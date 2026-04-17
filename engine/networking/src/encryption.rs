//! Network packet encryption using the crypto primitives from `genovo_core`.
//!
//! Provides authenticated encryption for game network packets using AES-128-CBC
//! with HMAC-SHA256 authentication tags, Diffie-Hellman key exchange for session
//! key establishment, and a [`SecureChannel`] wrapper that handles the complete
//! handshake-and-encrypt lifecycle.
//!
//! ## Packet wire format (encrypted)
//!
//! ```text
//! Offset  Size   Field
//! ------  -----  -----
//!   0       8    sequence number (u64 big-endian)
//!   8       2    payload length (u16 big-endian, of ciphertext)
//!  10       N    AES-128-CBC ciphertext (PKCS7-padded)
//!  10+N    32    HMAC-SHA256 tag (over bytes 0..10+N)
//! ```
//!
//! ## Anti-replay
//!
//! Each packet carries a monotonically increasing 64-bit sequence number. The
//! IV for AES-CBC is derived deterministically from this sequence, preventing
//! IV reuse. The receiver maintains a sliding window of accepted sequence
//! numbers and rejects duplicates.
//!
//! ## Key exchange
//!
//! A simplified Diffie-Hellman key agreement is provided for establishing
//! shared secrets without transmitting keys in the clear. The implementation
//! uses modular exponentiation over a 256-bit safe prime (suitable for game
//! engine use; production systems should use larger primes or ECDH).

use genovo_core::crypto::{Aes128, Hmac, Sha256};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Size of the sequence number field in the encrypted packet header.
const SEQUENCE_SIZE: usize = 8;

/// Size of the payload-length field.
const PAYLOAD_LEN_SIZE: usize = 2;

/// Size of the encrypted packet header (sequence + payload_len).
const ENCRYPTED_HEADER_SIZE: usize = SEQUENCE_SIZE + PAYLOAD_LEN_SIZE;

/// Size of the HMAC-SHA256 authentication tag.
const HMAC_TAG_SIZE: usize = 32;

/// AES block size.
const AES_BLOCK_SIZE: usize = 16;

/// Maximum plaintext payload size per packet.
const MAX_PLAINTEXT_SIZE: usize = 1200;

/// Size of the anti-replay window (bits).
const REPLAY_WINDOW_SIZE: usize = 256;

/// Number of u64 words needed for the replay bitmap.
const REPLAY_BITMAP_WORDS: usize = REPLAY_WINDOW_SIZE / 64;

// ---------------------------------------------------------------------------
// Encryption errors
// ---------------------------------------------------------------------------

/// Errors that can occur during packet encryption/decryption.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncryptionError {
    /// The HMAC authentication tag did not match.
    AuthenticationFailed,
    /// The packet data is too short or malformed.
    InvalidPacketFormat,
    /// Decryption failed (bad padding or corrupted data).
    DecryptionFailed,
    /// The sequence number has already been seen (replay attack).
    ReplayDetected,
    /// The sequence number is too old (outside the replay window).
    SequenceTooOld,
    /// The plaintext exceeds the maximum allowed size.
    PayloadTooLarge,
    /// Key exchange has not been completed.
    HandshakeIncomplete,
    /// Invalid key material.
    InvalidKeyMaterial,
}

impl std::fmt::Display for EncryptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AuthenticationFailed => write!(f, "HMAC authentication failed"),
            Self::InvalidPacketFormat => write!(f, "invalid encrypted packet format"),
            Self::DecryptionFailed => write!(f, "AES decryption failed"),
            Self::ReplayDetected => write!(f, "duplicate sequence number (replay)"),
            Self::SequenceTooOld => write!(f, "sequence number too old"),
            Self::PayloadTooLarge => write!(f, "plaintext payload too large"),
            Self::HandshakeIncomplete => write!(f, "key exchange not completed"),
            Self::InvalidKeyMaterial => write!(f, "invalid key material"),
        }
    }
}

impl std::error::Error for EncryptionError {}

/// Result type for encryption operations.
pub type EncryptionResult<T> = Result<T, EncryptionError>;

// ---------------------------------------------------------------------------
// IV derivation
// ---------------------------------------------------------------------------

/// Derive a 16-byte IV from a sequence number and a key.
///
/// The IV is computed as `SHA-256(key || sequence_bytes)[0..16]`. This ensures:
/// - Each sequence number produces a unique IV (no IV reuse).
/// - The IV is unpredictable without knowledge of the key.
fn derive_iv(key: &[u8; 16], sequence: u64) -> [u8; 16] {
    let mut material = Vec::with_capacity(24);
    material.extend_from_slice(key);
    material.extend_from_slice(&sequence.to_be_bytes());
    let hash = Sha256::hash(&material);
    let mut iv = [0u8; 16];
    iv.copy_from_slice(&hash[..16]);
    iv
}

// ===========================================================================
// PacketEncryptor
// ===========================================================================

/// Stateless packet encryption and decryption using AES-128-CBC + HMAC-SHA256.
///
/// Each call to `encrypt_packet` / `decrypt_packet` is self-contained — the
/// sequence number is embedded in the packet and used to derive the IV.
///
/// # Wire format
///
/// ```text
/// [8 bytes sequence] [2 bytes ciphertext_len] [N bytes ciphertext] [32 bytes HMAC]
/// ```
///
/// The HMAC is computed over the sequence + length + ciphertext, providing
/// authentication of both the header and the encrypted payload.
pub struct PacketEncryptor;

impl PacketEncryptor {
    /// Encrypt a plaintext payload into an authenticated encrypted packet.
    ///
    /// # Arguments
    /// * `plaintext` — the payload to encrypt (max 1200 bytes)
    /// * `key` — 16-byte AES-128 encryption key
    /// * `sequence` — monotonically increasing packet sequence number
    ///
    /// # Returns
    /// The complete encrypted packet (header + ciphertext + HMAC tag).
    pub fn encrypt_packet(
        plaintext: &[u8],
        key: &[u8; 16],
        sequence: u64,
    ) -> EncryptionResult<Vec<u8>> {
        if plaintext.len() > MAX_PLAINTEXT_SIZE {
            return Err(EncryptionError::PayloadTooLarge);
        }

        // 1. Derive IV from sequence number.
        let iv = derive_iv(key, sequence);

        // 2. Encrypt with AES-128-CBC (PKCS7 padding is applied internally).
        let ciphertext = Aes128::encrypt_cbc(plaintext, key, &iv);

        // 3. Build the packet: header + ciphertext.
        let ciphertext_len = ciphertext.len() as u16;
        let packet_len = ENCRYPTED_HEADER_SIZE + ciphertext.len() + HMAC_TAG_SIZE;
        let mut packet = Vec::with_capacity(packet_len);

        // Sequence number (8 bytes, big-endian).
        packet.extend_from_slice(&sequence.to_be_bytes());

        // Ciphertext length (2 bytes, big-endian).
        packet.extend_from_slice(&ciphertext_len.to_be_bytes());

        // Ciphertext.
        packet.extend_from_slice(&ciphertext);

        // 4. Compute HMAC over everything so far (authenticate-then-encrypt
        //    equivalent: we authenticate the ciphertext, which is the
        //    encrypt-then-MAC construction).
        let hmac_tag = Hmac::compute(key, &packet);

        // 5. Append HMAC tag.
        packet.extend_from_slice(&hmac_tag);

        Ok(packet)
    }

    /// Decrypt an authenticated encrypted packet.
    ///
    /// Verifies the HMAC tag first (constant-time comparison), then decrypts.
    /// This is the authenticate-then-decrypt pattern, which prevents padding
    /// oracle attacks.
    ///
    /// # Arguments
    /// * `packet` — the complete encrypted packet
    /// * `key` — 16-byte AES-128 encryption key
    /// * `sequence` — expected sequence number (must match the one in the packet)
    ///
    /// # Returns
    /// The decrypted plaintext payload.
    pub fn decrypt_packet(
        packet: &[u8],
        key: &[u8; 16],
        sequence: u64,
    ) -> EncryptionResult<Vec<u8>> {
        // Minimum packet size: header + 1 AES block + HMAC.
        let min_size = ENCRYPTED_HEADER_SIZE + AES_BLOCK_SIZE + HMAC_TAG_SIZE;
        if packet.len() < min_size {
            return Err(EncryptionError::InvalidPacketFormat);
        }

        // 1. Split packet into header+ciphertext and HMAC tag.
        let tag_start = packet.len() - HMAC_TAG_SIZE;
        let authenticated_data = &packet[..tag_start];
        let received_tag = &packet[tag_start..];

        // 2. Verify HMAC tag (constant-time comparison).
        let mut expected_tag = [0u8; 32];
        expected_tag.copy_from_slice(received_tag);
        if !Hmac::verify(key, authenticated_data, &expected_tag) {
            return Err(EncryptionError::AuthenticationFailed);
        }

        // 3. Parse header.
        let pkt_sequence = u64::from_be_bytes([
            packet[0], packet[1], packet[2], packet[3],
            packet[4], packet[5], packet[6], packet[7],
        ]);

        // Verify sequence number matches.
        if pkt_sequence != sequence {
            return Err(EncryptionError::ReplayDetected);
        }

        let ciphertext_len = u16::from_be_bytes([packet[8], packet[9]]) as usize;

        // Validate ciphertext length.
        if ENCRYPTED_HEADER_SIZE + ciphertext_len + HMAC_TAG_SIZE != packet.len() {
            return Err(EncryptionError::InvalidPacketFormat);
        }

        if ciphertext_len % AES_BLOCK_SIZE != 0 {
            return Err(EncryptionError::InvalidPacketFormat);
        }

        let ciphertext = &packet[ENCRYPTED_HEADER_SIZE..ENCRYPTED_HEADER_SIZE + ciphertext_len];

        // 4. Derive IV from sequence number.
        let iv = derive_iv(key, sequence);

        // 5. Decrypt with AES-128-CBC.
        let plaintext = Aes128::decrypt_cbc(ciphertext, key, &iv)
            .ok_or(EncryptionError::DecryptionFailed)?;

        Ok(plaintext)
    }

    /// Decrypt a packet without checking the sequence number against an expected
    /// value. Instead, extracts the sequence number from the packet and returns it.
    ///
    /// The HMAC is still verified.
    pub fn decrypt_packet_any_sequence(
        packet: &[u8],
        key: &[u8; 16],
    ) -> EncryptionResult<(u64, Vec<u8>)> {
        let min_size = ENCRYPTED_HEADER_SIZE + AES_BLOCK_SIZE + HMAC_TAG_SIZE;
        if packet.len() < min_size {
            return Err(EncryptionError::InvalidPacketFormat);
        }

        // Extract sequence from the packet.
        let sequence = u64::from_be_bytes([
            packet[0], packet[1], packet[2], packet[3],
            packet[4], packet[5], packet[6], packet[7],
        ]);

        let plaintext = Self::decrypt_packet(packet, key, sequence)?;
        Ok((sequence, plaintext))
    }
}

// ===========================================================================
// Anti-replay window
// ===========================================================================

/// Sliding window anti-replay protection.
///
/// Maintains a bitmap of the last `REPLAY_WINDOW_SIZE` sequence numbers.
/// Rejects packets that:
/// - Have a sequence number that has already been accepted.
/// - Have a sequence number older than `highest_seen - REPLAY_WINDOW_SIZE`.
///
/// This is the same algorithm used by IPsec (RFC 2401).
#[derive(Debug, Clone)]
pub struct ReplayWindow {
    /// The highest sequence number seen so far.
    highest_seq: u64,
    /// Bitmap of received sequence numbers within the window.
    /// Bit i represents `highest_seq - i`.
    bitmap: [u64; REPLAY_BITMAP_WORDS],
}

impl ReplayWindow {
    /// Create a new replay window.
    pub fn new() -> Self {
        Self {
            highest_seq: 0,
            bitmap: [0u64; REPLAY_BITMAP_WORDS],
        }
    }

    /// Check whether a sequence number is acceptable (not a replay).
    ///
    /// Returns `true` if the sequence number is new (not seen before and
    /// within the valid window). Does **not** mark it as seen — call
    /// [`accept`] for that.
    pub fn check(&self, seq: u64) -> bool {
        if seq == 0 {
            return self.highest_seq == 0;
        }

        if seq > self.highest_seq {
            // New packet ahead of the window — always acceptable.
            return true;
        }

        let diff = self.highest_seq - seq;
        if diff >= REPLAY_WINDOW_SIZE as u64 {
            // Too old.
            return false;
        }

        let word_idx = (diff as usize) / 64;
        let bit_idx = (diff as usize) % 64;

        if word_idx >= REPLAY_BITMAP_WORDS {
            return false;
        }

        // Check if the bit is NOT set (i.e., we haven't seen this seq before).
        (self.bitmap[word_idx] & (1u64 << bit_idx)) == 0
    }

    /// Mark a sequence number as seen.
    ///
    /// Returns `Ok(())` if accepted, `Err` if it's a replay or too old.
    pub fn accept(&mut self, seq: u64) -> EncryptionResult<()> {
        if !self.check(seq) {
            if seq <= self.highest_seq {
                let diff = self.highest_seq - seq;
                if diff >= REPLAY_WINDOW_SIZE as u64 {
                    return Err(EncryptionError::SequenceTooOld);
                }
                return Err(EncryptionError::ReplayDetected);
            }
        }

        if seq > self.highest_seq {
            let shift = seq - self.highest_seq;
            self.shift_window(shift);
            self.highest_seq = seq;
            // Mark the current seq as seen (bit 0).
            self.bitmap[0] |= 1;
        } else {
            let diff = self.highest_seq - seq;
            let word_idx = (diff as usize) / 64;
            let bit_idx = (diff as usize) % 64;
            if word_idx < REPLAY_BITMAP_WORDS {
                self.bitmap[word_idx] |= 1u64 << bit_idx;
            }
        }

        Ok(())
    }

    /// Shift the bitmap window forward by `count` positions.
    fn shift_window(&mut self, count: u64) {
        if count >= REPLAY_WINDOW_SIZE as u64 {
            // Clear everything.
            self.bitmap = [0u64; REPLAY_BITMAP_WORDS];
            return;
        }

        let word_shift = (count as usize) / 64;
        let bit_shift = (count as usize) % 64;

        if word_shift > 0 {
            // Shift words.
            for i in (word_shift..REPLAY_BITMAP_WORDS).rev() {
                self.bitmap[i] = self.bitmap[i - word_shift];
            }
            for i in 0..word_shift.min(REPLAY_BITMAP_WORDS) {
                self.bitmap[i] = 0;
            }
        }

        if bit_shift > 0 {
            // Shift bits within words.
            for i in (1..REPLAY_BITMAP_WORDS).rev() {
                self.bitmap[i] = (self.bitmap[i] << bit_shift)
                    | (self.bitmap[i - 1] >> (64 - bit_shift));
            }
            self.bitmap[0] <<= bit_shift;
        }
    }

    /// Reset the window.
    pub fn reset(&mut self) {
        self.highest_seq = 0;
        self.bitmap = [0u64; REPLAY_BITMAP_WORDS];
    }

    /// Get the highest sequence number seen.
    pub fn highest_sequence(&self) -> u64 {
        self.highest_seq
    }
}

impl Default for ReplayWindow {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Key Exchange (Diffie-Hellman over a 256-bit safe prime)
// ===========================================================================

/// A 256-bit unsigned integer for DH arithmetic, stored as 4 x u64 (little-endian words).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct U256 {
    pub words: [u64; 4],
}

impl U256 {
    /// Zero.
    pub const ZERO: Self = Self { words: [0; 4] };

    /// One.
    pub const ONE: Self = Self {
        words: [1, 0, 0, 0],
    };

    /// Create from a single u64.
    pub fn from_u64(val: u64) -> Self {
        Self {
            words: [val, 0, 0, 0],
        }
    }

    /// Create from big-endian bytes (32 bytes).
    pub fn from_be_bytes(bytes: &[u8; 32]) -> Self {
        let mut words = [0u64; 4];
        // bytes[0..8] is the most significant.
        words[3] = u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        words[2] = u64::from_be_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11],
            bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        words[1] = u64::from_be_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19],
            bytes[20], bytes[21], bytes[22], bytes[23],
        ]);
        words[0] = u64::from_be_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27],
            bytes[28], bytes[29], bytes[30], bytes[31],
        ]);
        Self { words }
    }

    /// Convert to big-endian bytes.
    pub fn to_be_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[0..8].copy_from_slice(&self.words[3].to_be_bytes());
        bytes[8..16].copy_from_slice(&self.words[2].to_be_bytes());
        bytes[16..24].copy_from_slice(&self.words[1].to_be_bytes());
        bytes[24..32].copy_from_slice(&self.words[0].to_be_bytes());
        bytes
    }

    /// Check if zero.
    pub fn is_zero(&self) -> bool {
        self.words == [0; 4]
    }

    /// Compare: returns -1, 0, or 1.
    pub fn cmp_u256(&self, other: &Self) -> std::cmp::Ordering {
        for i in (0..4).rev() {
            match self.words[i].cmp(&other.words[i]) {
                std::cmp::Ordering::Equal => continue,
                ord => return ord,
            }
        }
        std::cmp::Ordering::Equal
    }

    /// Addition with carry (wrapping at 2^256).
    pub fn wrapping_add(&self, other: &Self) -> Self {
        let mut result = [0u64; 4];
        let mut carry = 0u64;
        for i in 0..4 {
            let (sum1, c1) = self.words[i].overflowing_add(other.words[i]);
            let (sum2, c2) = sum1.overflowing_add(carry);
            result[i] = sum2;
            carry = (c1 as u64) + (c2 as u64);
        }
        Self { words: result }
    }

    /// Subtraction with borrow (wrapping).
    pub fn wrapping_sub(&self, other: &Self) -> Self {
        let mut result = [0u64; 4];
        let mut borrow = 0u64;
        for i in 0..4 {
            let (diff1, b1) = self.words[i].overflowing_sub(other.words[i]);
            let (diff2, b2) = diff1.overflowing_sub(borrow);
            result[i] = diff2;
            borrow = (b1 as u64) + (b2 as u64);
        }
        Self { words: result }
    }

    /// Test if bit `n` is set.
    pub fn bit(&self, n: usize) -> bool {
        if n >= 256 {
            return false;
        }
        let word = n / 64;
        let bit = n % 64;
        (self.words[word] >> bit) & 1 == 1
    }

    /// Number of significant bits.
    pub fn bits(&self) -> usize {
        for i in (0..4).rev() {
            if self.words[i] != 0 {
                return i * 64 + (64 - self.words[i].leading_zeros() as usize);
            }
        }
        0
    }

    /// Modular reduction: `self mod modulus`.
    ///
    /// Simple repeated-subtraction approach for demonstration. For production
    /// use, Montgomery or Barrett reduction would be used.
    pub fn mod_reduce(&self, modulus: &Self) -> Self {
        if modulus.is_zero() {
            return Self::ZERO;
        }
        // Use schoolbook division approach.
        let mut remainder = *self;
        let mod_bits = modulus.bits();
        let self_bits = self.bits();

        if self_bits == 0 || self.cmp_u256(modulus) == std::cmp::Ordering::Less {
            return remainder;
        }

        // Align modulus to the MSB of self, then subtract downward.
        let shift = if self_bits > mod_bits {
            self_bits - mod_bits
        } else {
            0
        };

        let mut shifted_mod = shl_u256(modulus, shift);

        for _ in 0..=shift {
            if remainder.cmp_u256(&shifted_mod) != std::cmp::Ordering::Less {
                remainder = remainder.wrapping_sub(&shifted_mod);
            }
            shifted_mod = shr_u256(&shifted_mod, 1);
        }

        remainder
    }
}

/// Shift a U256 left by `n` bits.
fn shl_u256(val: &U256, n: usize) -> U256 {
    if n >= 256 {
        return U256::ZERO;
    }
    let word_shift = n / 64;
    let bit_shift = n % 64;
    let mut result = [0u64; 4];

    for i in word_shift..4 {
        result[i] = val.words[i - word_shift] << bit_shift;
        if bit_shift > 0 && i > word_shift {
            result[i] |= val.words[i - word_shift - 1] >> (64 - bit_shift);
        }
    }

    U256 { words: result }
}

/// Shift a U256 right by `n` bits.
fn shr_u256(val: &U256, n: usize) -> U256 {
    if n >= 256 {
        return U256::ZERO;
    }
    let word_shift = n / 64;
    let bit_shift = n % 64;
    let mut result = [0u64; 4];

    for i in 0..(4 - word_shift) {
        result[i] = val.words[i + word_shift] >> bit_shift;
        if bit_shift > 0 && i + word_shift + 1 < 4 {
            result[i] |= val.words[i + word_shift + 1] << (64 - bit_shift);
        }
    }

    U256 { words: result }
}

/// Multiply two U256 values, returning the lower 512 bits as two U256 (lo, hi).
fn mul_u256(a: &U256, b: &U256) -> (U256, U256) {
    // Schoolbook multiplication into 8 u64 words.
    let mut result = [0u128; 8];

    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..4 {
            let prod = (a.words[i] as u128) * (b.words[j] as u128) + result[i + j] + carry;
            result[i + j] = prod & 0xFFFFFFFFFFFFFFFF;
            carry = prod >> 64;
        }
        if i + 4 < 8 {
            result[i + 4] += carry;
        }
    }

    let lo = U256 {
        words: [
            result[0] as u64,
            result[1] as u64,
            result[2] as u64,
            result[3] as u64,
        ],
    };
    let hi = U256 {
        words: [
            result[4] as u64,
            result[5] as u64,
            result[6] as u64,
            result[7] as u64,
        ],
    };

    (lo, hi)
}

/// Modular multiplication: `(a * b) mod m`.
///
/// Uses schoolbook multiply followed by reduction.
fn mulmod_u256(a: &U256, b: &U256, m: &U256) -> U256 {
    if m.is_zero() {
        return U256::ZERO;
    }

    // For simplicity, use the repeated-doubling approach (Montgomery
    // multiplication would be faster for production).
    let mut result = U256::ZERO;
    let mut base = a.mod_reduce(m);
    let bits = b.bits();

    for i in 0..bits {
        if b.bit(i) {
            result = result.wrapping_add(&base).mod_reduce(m);
        }
        base = base.wrapping_add(&base).mod_reduce(m);
    }

    result
}

/// Modular exponentiation: `base^exp mod modulus` using binary method (right-to-left).
pub fn modpow_u256(base: &U256, exp: &U256, modulus: &U256) -> U256 {
    if modulus.is_zero() {
        return U256::ZERO;
    }
    if exp.is_zero() {
        return U256::ONE.mod_reduce(modulus);
    }

    let mut result = U256::ONE;
    let mut b = base.mod_reduce(modulus);
    let bits = exp.bits();

    for i in 0..bits {
        if exp.bit(i) {
            result = mulmod_u256(&result, &b, modulus);
        }
        b = mulmod_u256(&b, &b, modulus);
    }

    result
}

/// A 256-bit safe prime for Diffie-Hellman.
///
/// p = 2^255 - 19 (the Curve25519 prime). While this is a Mersenne-like prime
/// used for elliptic curves, it also works as a DH prime for demonstration.
/// A generator g=2 is used.
fn dh_prime() -> U256 {
    // 2^255 - 19 =
    // 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFED
    U256 {
        words: [
            0xFFFFFFFFFFFFFFED,
            0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
            0x7FFFFFFFFFFFFFFF,
        ],
    }
}

/// The DH generator.
fn dh_generator() -> U256 {
    U256::from_u64(2)
}

/// A private key for Diffie-Hellman (256-bit random value).
#[derive(Debug, Clone)]
pub struct PrivateKey(pub U256);

/// A public key for Diffie-Hellman (g^private mod p).
#[derive(Debug, Clone)]
pub struct PublicKey(pub U256);

/// A shared secret (other_public^my_private mod p).
#[derive(Debug, Clone)]
pub struct SharedSecret(pub U256);

/// Diffie-Hellman key exchange.
pub struct KeyExchange;

impl KeyExchange {
    /// Generate a DH key pair from random bytes.
    ///
    /// # Arguments
    /// * `random_bytes` — 32 bytes of random data (from a CSPRNG).
    ///
    /// # Returns
    /// (private_key, public_key) tuple.
    pub fn generate_keypair(random_bytes: &[u8; 32]) -> (PrivateKey, PublicKey) {
        let p = dh_prime();
        let g = dh_generator();

        // Clamp the private key (clear high bit to stay < p, set low bits).
        let mut key_bytes = *random_bytes;
        key_bytes[0] &= 0x7F; // Clear bit 255 to ensure < p.
        key_bytes[31] &= 0xFC; // Clear low 2 bits (standard clamping for Curve25519-like ops).

        let private = U256::from_be_bytes(&key_bytes);
        let public = modpow_u256(&g, &private, &p);

        (PrivateKey(private), PublicKey(public))
    }

    /// Compute the shared secret from our private key and their public key.
    ///
    /// `shared = their_public ^ my_private mod p`
    pub fn compute_shared_secret(
        my_private: &PrivateKey,
        their_public: &PublicKey,
    ) -> SharedSecret {
        let p = dh_prime();
        let shared = modpow_u256(&their_public.0, &my_private.0, &p);
        SharedSecret(shared)
    }

    /// Derive a 16-byte AES session key from a shared secret.
    ///
    /// Uses SHA-256 of the shared secret bytes, taking the first 16 bytes.
    pub fn derive_session_key(shared_secret: &SharedSecret) -> [u8; 16] {
        let bytes = shared_secret.0.to_be_bytes();
        let hash = Sha256::hash(&bytes);
        let mut key = [0u8; 16];
        key.copy_from_slice(&hash[..16]);
        key
    }
}

// ===========================================================================
// SecureChannel
// ===========================================================================

/// State of a secure channel handshake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandshakeState {
    /// Initial state: no keys exchanged.
    NotStarted,
    /// We have generated our key pair and sent our public key.
    WaitingForPeerKey,
    /// We have received the peer's public key and derived the session key.
    Completed,
}

/// A secure communication channel that wraps a transport connection with
/// encryption.
///
/// The lifecycle is:
/// 1. Call `begin_handshake()` to generate a key pair. Send the returned
///    public key bytes to the peer.
/// 2. When the peer's public key bytes arrive, call `complete_handshake()`.
///    This derives the session key.
/// 3. All subsequent calls to `encrypt()` and `decrypt()` use the session key.
/// 4. Sequence numbers are tracked automatically for anti-replay.
///
/// # Example
/// ```ignore
/// let mut channel = SecureChannel::new();
/// let my_pub = channel.begin_handshake(&random_bytes);
/// // ... send my_pub to peer, receive their_pub ...
/// channel.complete_handshake(&their_pub_bytes)?;
/// let encrypted = channel.encrypt(b"hello")?;
/// let decrypted = channel.decrypt(&encrypted)?;
/// ```
pub struct SecureChannel {
    /// Handshake state.
    state: HandshakeState,
    /// Our private key.
    private_key: Option<PrivateKey>,
    /// Our public key.
    public_key: Option<PublicKey>,
    /// Derived session key (16 bytes for AES-128).
    session_key: Option<[u8; 16]>,
    /// Send sequence number (incremented for each sent packet).
    send_sequence: u64,
    /// Receive replay window.
    replay_window: ReplayWindow,
}

impl SecureChannel {
    /// Create a new secure channel.
    pub fn new() -> Self {
        Self {
            state: HandshakeState::NotStarted,
            private_key: None,
            public_key: None,
            session_key: None,
            send_sequence: 0,
            replay_window: ReplayWindow::new(),
        }
    }

    /// Begin the handshake by generating a key pair.
    ///
    /// Returns the public key bytes (32 bytes) that should be sent to the peer.
    pub fn begin_handshake(&mut self, random_bytes: &[u8; 32]) -> [u8; 32] {
        let (private, public) = KeyExchange::generate_keypair(random_bytes);
        let pub_bytes = public.0.to_be_bytes();
        self.private_key = Some(private);
        self.public_key = Some(public);
        self.state = HandshakeState::WaitingForPeerKey;
        pub_bytes
    }

    /// Complete the handshake upon receiving the peer's public key.
    ///
    /// Derives the session key from the shared secret.
    pub fn complete_handshake(&mut self, peer_public_bytes: &[u8; 32]) -> EncryptionResult<()> {
        let private = self
            .private_key
            .as_ref()
            .ok_or(EncryptionError::HandshakeIncomplete)?;

        let peer_public = PublicKey(U256::from_be_bytes(peer_public_bytes));
        let shared = KeyExchange::compute_shared_secret(private, &peer_public);

        // Check for degenerate shared secret (0 or 1).
        if shared.0.is_zero() || shared.0 == U256::ONE {
            return Err(EncryptionError::InvalidKeyMaterial);
        }

        let session_key = KeyExchange::derive_session_key(&shared);
        self.session_key = Some(session_key);
        self.state = HandshakeState::Completed;
        self.send_sequence = 0;
        self.replay_window.reset();

        Ok(())
    }

    /// Create a secure channel with a pre-shared key (skipping handshake).
    pub fn with_preshared_key(key: [u8; 16]) -> Self {
        Self {
            state: HandshakeState::Completed,
            private_key: None,
            public_key: None,
            session_key: Some(key),
            send_sequence: 0,
            replay_window: ReplayWindow::new(),
        }
    }

    /// Encrypt a payload. Automatically assigns and increments the sequence number.
    pub fn encrypt(&mut self, plaintext: &[u8]) -> EncryptionResult<Vec<u8>> {
        let key = self
            .session_key
            .as_ref()
            .ok_or(EncryptionError::HandshakeIncomplete)?;

        let seq = self.send_sequence;
        self.send_sequence += 1;

        PacketEncryptor::encrypt_packet(plaintext, key, seq)
    }

    /// Decrypt a received packet. Checks anti-replay.
    pub fn decrypt(&mut self, packet: &[u8]) -> EncryptionResult<Vec<u8>> {
        let key = self
            .session_key
            .as_ref()
            .ok_or(EncryptionError::HandshakeIncomplete)?;

        // Extract sequence from the packet header.
        if packet.len() < ENCRYPTED_HEADER_SIZE {
            return Err(EncryptionError::InvalidPacketFormat);
        }
        let sequence = u64::from_be_bytes([
            packet[0], packet[1], packet[2], packet[3],
            packet[4], packet[5], packet[6], packet[7],
        ]);

        // Check replay window.
        if !self.replay_window.check(sequence) {
            return Err(EncryptionError::ReplayDetected);
        }

        // Decrypt and verify.
        let plaintext = PacketEncryptor::decrypt_packet(packet, key, sequence)?;

        // Accept the sequence number.
        self.replay_window.accept(sequence)?;

        Ok(plaintext)
    }

    /// Get the current handshake state.
    pub fn state(&self) -> HandshakeState {
        self.state
    }

    /// Get the current send sequence number.
    pub fn send_sequence(&self) -> u64 {
        self.send_sequence
    }

    /// Whether the channel is ready for encrypted communication.
    pub fn is_ready(&self) -> bool {
        self.state == HandshakeState::Completed && self.session_key.is_some()
    }
}

impl Default for SecureChannel {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// PacketIntegrity (HMAC-only mode)
// ===========================================================================

/// HMAC-only packet authentication (no encryption).
///
/// Useful for game data that needs tamper protection but not confidentiality
/// (e.g., lobby chat, public game state). Cheaper than full encryption.
///
/// ## Wire format
///
/// ```text
/// [8 bytes sequence] [2 bytes payload_len] [N bytes plaintext] [32 bytes HMAC]
/// ```
pub struct PacketIntegrity;

impl PacketIntegrity {
    /// Authenticate a plaintext packet (no encryption).
    pub fn sign_packet(
        plaintext: &[u8],
        key: &[u8; 16],
        sequence: u64,
    ) -> EncryptionResult<Vec<u8>> {
        if plaintext.len() > MAX_PLAINTEXT_SIZE {
            return Err(EncryptionError::PayloadTooLarge);
        }

        let payload_len = plaintext.len() as u16;
        let mut packet = Vec::with_capacity(ENCRYPTED_HEADER_SIZE + plaintext.len() + HMAC_TAG_SIZE);

        // Header.
        packet.extend_from_slice(&sequence.to_be_bytes());
        packet.extend_from_slice(&payload_len.to_be_bytes());

        // Plaintext payload.
        packet.extend_from_slice(plaintext);

        // HMAC over header + plaintext.
        // Use a wider key by concatenating the AES key with a constant.
        let mut hmac_key = Vec::with_capacity(32);
        hmac_key.extend_from_slice(key);
        hmac_key.extend_from_slice(b"integrity-only\x00\x00");
        let tag = Hmac::compute(&hmac_key, &packet);
        packet.extend_from_slice(&tag);

        Ok(packet)
    }

    /// Verify and extract the plaintext from a signed packet.
    pub fn verify_packet(
        packet: &[u8],
        key: &[u8; 16],
        sequence: u64,
    ) -> EncryptionResult<Vec<u8>> {
        let min_size = ENCRYPTED_HEADER_SIZE + HMAC_TAG_SIZE;
        if packet.len() < min_size {
            return Err(EncryptionError::InvalidPacketFormat);
        }

        // Split off the HMAC tag.
        let tag_start = packet.len() - HMAC_TAG_SIZE;
        let authenticated_data = &packet[..tag_start];
        let received_tag = &packet[tag_start..];

        // Verify HMAC.
        let mut hmac_key = Vec::with_capacity(32);
        hmac_key.extend_from_slice(key);
        hmac_key.extend_from_slice(b"integrity-only\x00\x00");

        let mut expected_tag = [0u8; 32];
        expected_tag.copy_from_slice(received_tag);
        if !Hmac::verify(&hmac_key, authenticated_data, &expected_tag) {
            return Err(EncryptionError::AuthenticationFailed);
        }

        // Parse header.
        let pkt_sequence = u64::from_be_bytes([
            packet[0], packet[1], packet[2], packet[3],
            packet[4], packet[5], packet[6], packet[7],
        ]);

        if pkt_sequence != sequence {
            return Err(EncryptionError::ReplayDetected);
        }

        let payload_len = u16::from_be_bytes([packet[8], packet[9]]) as usize;
        if ENCRYPTED_HEADER_SIZE + payload_len + HMAC_TAG_SIZE != packet.len() {
            return Err(EncryptionError::InvalidPacketFormat);
        }

        let plaintext = packet[ENCRYPTED_HEADER_SIZE..ENCRYPTED_HEADER_SIZE + payload_len].to_vec();
        Ok(plaintext)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> [u8; 16] {
        [0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
         0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c]
    }

    // -- PacketEncryptor tests -----------------------------------------------

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let key = test_key();
        let plaintext = b"Hello, Genovo Engine!";
        let sequence = 42u64;

        let packet = PacketEncryptor::encrypt_packet(plaintext, &key, sequence).unwrap();

        // Packet should be larger than plaintext (header + padding + HMAC).
        assert!(packet.len() > plaintext.len());

        let decrypted = PacketEncryptor::decrypt_packet(&packet, &key, sequence).unwrap();
        assert_eq!(&decrypted[..], &plaintext[..]);
    }

    #[test]
    fn encrypt_decrypt_empty() {
        let key = test_key();
        let packet = PacketEncryptor::encrypt_packet(b"", &key, 0).unwrap();
        let decrypted = PacketEncryptor::decrypt_packet(&packet, &key, 0).unwrap();
        assert!(decrypted.is_empty());
    }

    #[test]
    fn encrypt_decrypt_max_size() {
        let key = test_key();
        let plaintext = vec![0xAA; MAX_PLAINTEXT_SIZE];
        let packet = PacketEncryptor::encrypt_packet(&plaintext, &key, 100).unwrap();
        let decrypted = PacketEncryptor::decrypt_packet(&packet, &key, 100).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn encrypt_too_large() {
        let key = test_key();
        let plaintext = vec![0xAA; MAX_PLAINTEXT_SIZE + 1];
        let result = PacketEncryptor::encrypt_packet(&plaintext, &key, 0);
        assert_eq!(result.unwrap_err(), EncryptionError::PayloadTooLarge);
    }

    #[test]
    fn decrypt_wrong_key() {
        let key = test_key();
        let wrong_key = [0xFF; 16];
        let packet = PacketEncryptor::encrypt_packet(b"secret", &key, 1).unwrap();
        let result = PacketEncryptor::decrypt_packet(&packet, &wrong_key, 1);
        assert_eq!(result.unwrap_err(), EncryptionError::AuthenticationFailed);
    }

    #[test]
    fn decrypt_wrong_sequence() {
        let key = test_key();
        let packet = PacketEncryptor::encrypt_packet(b"data", &key, 5).unwrap();
        let result = PacketEncryptor::decrypt_packet(&packet, &key, 6);
        assert_eq!(result.unwrap_err(), EncryptionError::ReplayDetected);
    }

    #[test]
    fn decrypt_tampered_ciphertext() {
        let key = test_key();
        let mut packet = PacketEncryptor::encrypt_packet(b"data", &key, 1).unwrap();
        // Tamper with a ciphertext byte.
        if packet.len() > ENCRYPTED_HEADER_SIZE + 2 {
            packet[ENCRYPTED_HEADER_SIZE + 1] ^= 0xFF;
        }
        let result = PacketEncryptor::decrypt_packet(&packet, &key, 1);
        assert_eq!(result.unwrap_err(), EncryptionError::AuthenticationFailed);
    }

    #[test]
    fn decrypt_tampered_hmac() {
        let key = test_key();
        let mut packet = PacketEncryptor::encrypt_packet(b"data", &key, 1).unwrap();
        // Tamper with the last byte of the HMAC tag.
        let last = packet.len() - 1;
        packet[last] ^= 0xFF;
        let result = PacketEncryptor::decrypt_packet(&packet, &key, 1);
        assert_eq!(result.unwrap_err(), EncryptionError::AuthenticationFailed);
    }

    #[test]
    fn decrypt_truncated() {
        let key = test_key();
        let packet = PacketEncryptor::encrypt_packet(b"data", &key, 1).unwrap();
        let truncated = &packet[..packet.len() - 10];
        let result = PacketEncryptor::decrypt_packet(truncated, &key, 1);
        assert!(result.is_err());
    }

    #[test]
    fn different_sequences_different_ciphertext() {
        let key = test_key();
        let data = b"same data";
        let p1 = PacketEncryptor::encrypt_packet(data, &key, 1).unwrap();
        let p2 = PacketEncryptor::encrypt_packet(data, &key, 2).unwrap();
        // Ciphertexts should differ (different IVs from different sequences).
        assert_ne!(p1, p2);
    }

    #[test]
    fn decrypt_any_sequence() {
        let key = test_key();
        let packet = PacketEncryptor::encrypt_packet(b"hello", &key, 99).unwrap();
        let (seq, plaintext) = PacketEncryptor::decrypt_packet_any_sequence(&packet, &key).unwrap();
        assert_eq!(seq, 99);
        assert_eq!(&plaintext[..], b"hello");
    }

    // -- ReplayWindow tests --------------------------------------------------

    #[test]
    fn replay_window_sequential() {
        let mut window = ReplayWindow::new();
        for i in 1..=100 {
            assert!(window.check(i));
            window.accept(i).unwrap();
        }
    }

    #[test]
    fn replay_window_duplicate() {
        let mut window = ReplayWindow::new();
        window.accept(1).unwrap();
        assert!(!window.check(1));
        assert_eq!(
            window.accept(1).unwrap_err(),
            EncryptionError::ReplayDetected
        );
    }

    #[test]
    fn replay_window_out_of_order() {
        let mut window = ReplayWindow::new();
        window.accept(5).unwrap();
        window.accept(3).unwrap();
        window.accept(7).unwrap();
        window.accept(1).unwrap();

        // All should be marked as seen.
        assert!(!window.check(5));
        assert!(!window.check(3));
        assert!(!window.check(7));
        assert!(!window.check(1));

        // 2, 4, 6 should still be acceptable.
        assert!(window.check(2));
        assert!(window.check(4));
        assert!(window.check(6));
    }

    #[test]
    fn replay_window_too_old() {
        let mut window = ReplayWindow::new();
        window.accept(REPLAY_WINDOW_SIZE as u64 + 10).unwrap();
        // Sequence 1 is now outside the window.
        assert!(!window.check(1));
        assert_eq!(
            window.accept(1).unwrap_err(),
            EncryptionError::SequenceTooOld
        );
    }

    #[test]
    fn replay_window_edge_of_window() {
        let mut window = ReplayWindow::new();
        let high = REPLAY_WINDOW_SIZE as u64;
        window.accept(high).unwrap();
        // Sequence 1 should be at the edge of the window.
        assert!(window.check(1));
        window.accept(1).unwrap();
        assert!(!window.check(1));
    }

    // -- KeyExchange tests ---------------------------------------------------

    #[test]
    fn key_exchange_shared_secret() {
        // Two parties should derive the same shared secret.
        let alice_random: [u8; 32] = {
            let mut b = [0u8; 32];
            for (i, byte) in b.iter_mut().enumerate() {
                *byte = (i as u8).wrapping_mul(7).wrapping_add(13);
            }
            b
        };
        let bob_random: [u8; 32] = {
            let mut b = [0u8; 32];
            for (i, byte) in b.iter_mut().enumerate() {
                *byte = (i as u8).wrapping_mul(11).wrapping_add(37);
            }
            b
        };

        let (alice_priv, alice_pub) = KeyExchange::generate_keypair(&alice_random);
        let (bob_priv, bob_pub) = KeyExchange::generate_keypair(&bob_random);

        let alice_shared = KeyExchange::compute_shared_secret(&alice_priv, &bob_pub);
        let bob_shared = KeyExchange::compute_shared_secret(&bob_priv, &alice_pub);

        assert_eq!(alice_shared.0, bob_shared.0);

        // Derived session keys should be the same.
        let alice_key = KeyExchange::derive_session_key(&alice_shared);
        let bob_key = KeyExchange::derive_session_key(&bob_shared);
        assert_eq!(alice_key, bob_key);
    }

    #[test]
    fn key_exchange_different_keys_different_secrets() {
        let rand1: [u8; 32] = [0x01; 32];
        let rand2: [u8; 32] = [0x02; 32];
        let rand3: [u8; 32] = [0x03; 32];

        let (priv1, pub1) = KeyExchange::generate_keypair(&rand1);
        let (_, pub2) = KeyExchange::generate_keypair(&rand2);

        let secret_a = KeyExchange::compute_shared_secret(&priv1, &pub2);

        let (priv3, _) = KeyExchange::generate_keypair(&rand3);
        let secret_b = KeyExchange::compute_shared_secret(&priv3, &pub2);

        // Different private keys should yield different shared secrets.
        assert_ne!(secret_a.0, secret_b.0);
    }

    // -- SecureChannel tests -------------------------------------------------

    #[test]
    fn secure_channel_preshared_key() {
        let key = test_key();
        let mut sender = SecureChannel::with_preshared_key(key);
        let mut receiver = SecureChannel::with_preshared_key(key);

        assert!(sender.is_ready());
        assert!(receiver.is_ready());

        let encrypted = sender.encrypt(b"hello from sender").unwrap();
        let decrypted = receiver.decrypt(&encrypted).unwrap();
        assert_eq!(&decrypted[..], b"hello from sender");
    }

    #[test]
    fn secure_channel_multiple_messages() {
        let key = test_key();
        let mut sender = SecureChannel::with_preshared_key(key);
        let mut receiver = SecureChannel::with_preshared_key(key);

        for i in 0..10 {
            let msg = format!("message {}", i);
            let encrypted = sender.encrypt(msg.as_bytes()).unwrap();
            let decrypted = receiver.decrypt(&encrypted).unwrap();
            assert_eq!(decrypted, msg.as_bytes());
        }

        assert_eq!(sender.send_sequence(), 10);
    }

    #[test]
    fn secure_channel_replay_rejected() {
        let key = test_key();
        let mut sender = SecureChannel::with_preshared_key(key);
        let mut receiver = SecureChannel::with_preshared_key(key);

        let encrypted = sender.encrypt(b"test").unwrap();
        receiver.decrypt(&encrypted).unwrap();

        // Replaying the same packet should fail.
        let result = receiver.decrypt(&encrypted);
        assert_eq!(result.unwrap_err(), EncryptionError::ReplayDetected);
    }

    #[test]
    fn secure_channel_not_ready() {
        let mut channel = SecureChannel::new();
        assert!(!channel.is_ready());
        let result = channel.encrypt(b"test");
        assert_eq!(result.unwrap_err(), EncryptionError::HandshakeIncomplete);
    }

    #[test]
    fn secure_channel_handshake() {
        let alice_rand: [u8; 32] = {
            let mut b = [0u8; 32];
            for (i, byte) in b.iter_mut().enumerate() {
                *byte = (i as u8).wrapping_mul(17).wrapping_add(42);
            }
            b
        };
        let bob_rand: [u8; 32] = {
            let mut b = [0u8; 32];
            for (i, byte) in b.iter_mut().enumerate() {
                *byte = (i as u8).wrapping_mul(23).wrapping_add(7);
            }
            b
        };

        let mut alice = SecureChannel::new();
        let mut bob = SecureChannel::new();

        let alice_pub = alice.begin_handshake(&alice_rand);
        let bob_pub = bob.begin_handshake(&bob_rand);

        alice.complete_handshake(&bob_pub).unwrap();
        bob.complete_handshake(&alice_pub).unwrap();

        assert!(alice.is_ready());
        assert!(bob.is_ready());

        // Alice sends to Bob.
        let encrypted = alice.encrypt(b"Hello Bob!").unwrap();
        let decrypted = bob.decrypt(&encrypted).unwrap();
        assert_eq!(&decrypted[..], b"Hello Bob!");

        // Bob sends to Alice.
        let encrypted = bob.encrypt(b"Hello Alice!").unwrap();
        let decrypted = alice.decrypt(&encrypted).unwrap();
        assert_eq!(&decrypted[..], b"Hello Alice!");
    }

    // -- PacketIntegrity tests -----------------------------------------------

    #[test]
    fn integrity_sign_verify() {
        let key = test_key();
        let packet = PacketIntegrity::sign_packet(b"public data", &key, 1).unwrap();
        let verified = PacketIntegrity::verify_packet(&packet, &key, 1).unwrap();
        assert_eq!(&verified[..], b"public data");
    }

    #[test]
    fn integrity_tampered() {
        let key = test_key();
        let mut packet = PacketIntegrity::sign_packet(b"data", &key, 1).unwrap();
        // Tamper with a payload byte.
        packet[ENCRYPTED_HEADER_SIZE] ^= 0xFF;
        let result = PacketIntegrity::verify_packet(&packet, &key, 1);
        assert_eq!(result.unwrap_err(), EncryptionError::AuthenticationFailed);
    }

    #[test]
    fn integrity_wrong_sequence() {
        let key = test_key();
        let packet = PacketIntegrity::sign_packet(b"data", &key, 5).unwrap();
        let result = PacketIntegrity::verify_packet(&packet, &key, 6);
        assert_eq!(result.unwrap_err(), EncryptionError::ReplayDetected);
    }

    // -- U256 tests ----------------------------------------------------------

    #[test]
    fn u256_be_bytes_roundtrip() {
        let val = U256 {
            words: [0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0],
        };
        let bytes = val.to_be_bytes();
        let recovered = U256::from_be_bytes(&bytes);
        assert_eq!(val, recovered);
    }

    #[test]
    fn u256_add_basic() {
        let a = U256::from_u64(100);
        let b = U256::from_u64(200);
        let c = a.wrapping_add(&b);
        assert_eq!(c.words[0], 300);
    }

    #[test]
    fn u256_sub_basic() {
        let a = U256::from_u64(300);
        let b = U256::from_u64(100);
        let c = a.wrapping_sub(&b);
        assert_eq!(c.words[0], 200);
    }

    #[test]
    fn u256_mod_reduce() {
        let a = U256::from_u64(17);
        let m = U256::from_u64(5);
        let r = a.mod_reduce(&m);
        assert_eq!(r.words[0], 2);
    }

    #[test]
    fn u256_modpow_small() {
        // 2^10 mod 1000 = 1024 mod 1000 = 24
        let base = U256::from_u64(2);
        let exp = U256::from_u64(10);
        let modulus = U256::from_u64(1000);
        let result = modpow_u256(&base, &exp, &modulus);
        assert_eq!(result.words[0], 24);
    }

    #[test]
    fn u256_modpow_fermat() {
        // Fermat's little theorem: a^(p-1) = 1 mod p for prime p.
        // Use a small prime: p = 97, a = 3.
        let a = U256::from_u64(3);
        let p = U256::from_u64(97);
        let exp = U256::from_u64(96); // p - 1
        let result = modpow_u256(&a, &exp, &p);
        assert_eq!(result.words[0], 1);
    }

    #[test]
    fn u256_bits() {
        assert_eq!(U256::ZERO.bits(), 0);
        assert_eq!(U256::ONE.bits(), 1);
        assert_eq!(U256::from_u64(255).bits(), 8);
        assert_eq!(U256::from_u64(256).bits(), 9);
    }

    // -- IV derivation tests -------------------------------------------------

    #[test]
    fn iv_derivation_unique() {
        let key = test_key();
        let iv1 = derive_iv(&key, 1);
        let iv2 = derive_iv(&key, 2);
        assert_ne!(iv1, iv2);
    }

    #[test]
    fn iv_derivation_deterministic() {
        let key = test_key();
        let iv1 = derive_iv(&key, 42);
        let iv2 = derive_iv(&key, 42);
        assert_eq!(iv1, iv2);
    }
}
