//! # Basic Cryptography
//!
//! Self-contained cryptographic primitives for network packet encryption and
//! save-file integrity verification. These implementations follow published
//! standards and are suitable for game engine use cases where external crate
//! dependencies are undesirable.
//!
//! **Warning**: These implementations prioritize correctness and clarity over
//! side-channel resistance. Do not use for security-critical applications
//! outside game engine contexts.
//!
//! - [`Sha256`] — SHA-256 hash (FIPS 180-4)
//! - [`Aes128`] — AES-128 block cipher with CBC mode
//! - [`Crc32`] — CRC-32/ISO-HDLC (Ethernet)
//! - [`Hmac`] — HMAC-SHA256

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4)
// ---------------------------------------------------------------------------

/// SHA-256 round constants (first 32 bits of the fractional parts of the
/// cube roots of the first 64 primes).
const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 initial hash values (first 32 bits of the fractional parts of the
/// square roots of the first 8 primes).
const SHA256_H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// SHA-256 hash function (FIPS 180-4).
///
/// # Example
/// ```
/// use genovo_core::crypto::Sha256;
///
/// let hash = Sha256::hash(b"hello");
/// // Expected: 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
/// assert_eq!(hash[0], 0x2c);
/// ```
pub struct Sha256;

impl Sha256 {
    /// Ch(x, y, z) = (x AND y) XOR (NOT x AND z)
    #[inline]
    fn ch(x: u32, y: u32, z: u32) -> u32 {
        (x & y) ^ (!x & z)
    }

    /// Maj(x, y, z) = (x AND y) XOR (x AND z) XOR (y AND z)
    #[inline]
    fn maj(x: u32, y: u32, z: u32) -> u32 {
        (x & y) ^ (x & z) ^ (y & z)
    }

    /// Upper-case Sigma_0: ROTR^2(x) XOR ROTR^13(x) XOR ROTR^22(x)
    #[inline]
    fn big_sigma0(x: u32) -> u32 {
        x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
    }

    /// Upper-case Sigma_1: ROTR^6(x) XOR ROTR^11(x) XOR ROTR^25(x)
    #[inline]
    fn big_sigma1(x: u32) -> u32 {
        x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
    }

    /// Lower-case sigma_0: ROTR^7(x) XOR ROTR^18(x) XOR SHR^3(x)
    #[inline]
    fn small_sigma0(x: u32) -> u32 {
        x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
    }

    /// Lower-case sigma_1: ROTR^17(x) XOR ROTR^19(x) XOR SHR^10(x)
    #[inline]
    fn small_sigma1(x: u32) -> u32 {
        x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
    }

    /// Pads the message according to SHA-256 spec:
    /// append 0x80, pad zeros, append 64-bit big-endian length.
    fn pad_message(data: &[u8]) -> Vec<u8> {
        let bit_len = (data.len() as u64) * 8;
        let mut padded = data.to_vec();

        // Append the bit '1' (0x80 byte).
        padded.push(0x80);

        // Pad with zeros until length is 56 mod 64.
        while padded.len() % 64 != 56 {
            padded.push(0x00);
        }

        // Append the original message length as 64-bit big-endian.
        padded.extend_from_slice(&bit_len.to_be_bytes());

        padded
    }

    /// Processes a single 512-bit (64-byte) block, updating the hash state.
    fn process_block(state: &mut [u32; 8], block: &[u8]) {
        // Prepare the message schedule W[0..63].
        let mut w = [0u32; 64];

        // W[0..15] = the 16 big-endian 32-bit words from the block.
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }

        // W[16..63] = sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16]
        for t in 16..64 {
            w[t] = Self::small_sigma1(w[t - 2])
                .wrapping_add(w[t - 7])
                .wrapping_add(Self::small_sigma0(w[t - 15]))
                .wrapping_add(w[t - 16]);
        }

        // Initialize working variables.
        let mut a = state[0];
        let mut b = state[1];
        let mut c = state[2];
        let mut d = state[3];
        let mut e = state[4];
        let mut f = state[5];
        let mut g = state[6];
        let mut h = state[7];

        // 64 compression rounds.
        for t in 0..64 {
            let t1 = h
                .wrapping_add(Self::big_sigma1(e))
                .wrapping_add(Self::ch(e, f, g))
                .wrapping_add(SHA256_K[t])
                .wrapping_add(w[t]);
            let t2 = Self::big_sigma0(a).wrapping_add(Self::maj(a, b, c));

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }

        // Add compressed chunk to running hash value.
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    /// Computes the SHA-256 hash of `data` in a single call.
    pub fn hash(data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256Hasher::new();
        hasher.update(data);
        hasher.finalize()
    }

    /// Returns a hex string representation of a hash.
    pub fn hash_hex(data: &[u8]) -> String {
        let hash = Self::hash(data);
        hash.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

/// Incremental SHA-256 hasher. Allows feeding data in chunks.
///
/// # Example
/// ```
/// use genovo_core::crypto::Sha256Hasher;
///
/// let mut hasher = Sha256Hasher::new();
/// hasher.update(b"hello");
/// hasher.update(b" world");
/// let hash = hasher.finalize();
/// ```
pub struct Sha256Hasher {
    state: [u32; 8],
    buffer: [u8; 64],
    buffer_len: usize,
    total_len: u64,
}

impl Sha256Hasher {
    /// Creates a new hasher initialized with SHA-256 IV.
    pub fn new() -> Self {
        Self {
            state: SHA256_H0,
            buffer: [0u8; 64],
            buffer_len: 0,
            total_len: 0,
        }
    }

    /// Feeds more data into the hasher.
    pub fn update(&mut self, data: &[u8]) {
        self.total_len += data.len() as u64;
        let mut offset = 0;

        // If we have buffered data, try to fill the buffer to 64 bytes.
        if self.buffer_len > 0 {
            let to_copy = (64 - self.buffer_len).min(data.len());
            self.buffer[self.buffer_len..self.buffer_len + to_copy]
                .copy_from_slice(&data[..to_copy]);
            self.buffer_len += to_copy;
            offset = to_copy;

            if self.buffer_len == 64 {
                let block = self.buffer;
                Sha256::process_block(&mut self.state, &block);
                self.buffer_len = 0;
            }
        }

        // Process complete 64-byte blocks directly from input.
        while offset + 64 <= data.len() {
            Sha256::process_block(&mut self.state, &data[offset..offset + 64]);
            offset += 64;
        }

        // Buffer any remaining bytes.
        let remaining = data.len() - offset;
        if remaining > 0 {
            self.buffer[..remaining].copy_from_slice(&data[offset..]);
            self.buffer_len = remaining;
        }
    }

    /// Finalizes the hash computation and returns the 32-byte digest.
    ///
    /// This consumes the hasher state; call `new()` to start fresh.
    pub fn finalize(&mut self) -> [u8; 32] {
        let bit_len = self.total_len * 8;

        // Pad: append 0x80.
        self.buffer[self.buffer_len] = 0x80;
        self.buffer_len += 1;

        // If there isn't room for the 8-byte length, process this block and start a new one.
        if self.buffer_len > 56 {
            // Zero-fill the rest of this block.
            for i in self.buffer_len..64 {
                self.buffer[i] = 0;
            }
            let block = self.buffer;
            Sha256::process_block(&mut self.state, &block);
            self.buffer_len = 0;
        }

        // Zero-fill up to byte 56.
        for i in self.buffer_len..56 {
            self.buffer[i] = 0;
        }

        // Append the bit length as big-endian u64.
        self.buffer[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buffer;
        Sha256::process_block(&mut self.state, &block);

        // Produce the final hash.
        let mut hash = [0u8; 32];
        for (i, &word) in self.state.iter().enumerate() {
            hash[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        hash
    }

    /// Resets the hasher to its initial state.
    pub fn reset(&mut self) {
        self.state = SHA256_H0;
        self.buffer = [0u8; 64];
        self.buffer_len = 0;
        self.total_len = 0;
    }
}

impl Default for Sha256Hasher {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AES-128 (FIPS 197)
// ---------------------------------------------------------------------------

/// AES S-box (SubBytes forward substitution table).
#[rustfmt::skip]
const AES_SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

/// AES inverse S-box (InvSubBytes substitution table).
#[rustfmt::skip]
const AES_INV_SBOX: [u8; 256] = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d,
];

/// AES round constants (Rcon). Only indices 1..10 are used for AES-128.
const AES_RCON: [u8; 11] = [
    0x00, // unused
    0x01, 0x02, 0x04, 0x08, 0x10,
    0x20, 0x40, 0x80, 0x1b, 0x36,
];

/// AES-128 block cipher (FIPS 197).
///
/// Implements single-block encrypt/decrypt as well as CBC mode with PKCS7
/// padding for multi-block data.
///
/// # Example
/// ```
/// use genovo_core::crypto::Aes128;
///
/// let key = [0u8; 16];
/// let plaintext = [0u8; 16];
/// let ciphertext = Aes128::encrypt_block(&plaintext, &key);
/// let decrypted = Aes128::decrypt_block(&ciphertext, &key);
/// assert_eq!(decrypted, plaintext);
/// ```
pub struct Aes128;

impl Aes128 {
    /// Number of rounds for AES-128.
    const NR: usize = 10;
    /// Key length in 32-bit words.
    const NK: usize = 4;
    /// Block size in 32-bit words.
    const NB: usize = 4;

    /// Galois Field multiplication by 2 in GF(2^8) with irreducible polynomial
    /// x^8 + x^4 + x^3 + x + 1 (0x11B).
    #[inline]
    fn gf_mul2(x: u8) -> u8 {
        if x & 0x80 != 0 {
            (x << 1) ^ 0x1b
        } else {
            x << 1
        }
    }

    /// Galois Field multiplication by 3 in GF(2^8).
    #[inline]
    fn gf_mul3(x: u8) -> u8 {
        Self::gf_mul2(x) ^ x
    }

    /// Galois Field multiplication by 9 in GF(2^8).
    #[inline]
    fn gf_mul9(x: u8) -> u8 {
        Self::gf_mul2(Self::gf_mul2(Self::gf_mul2(x))) ^ x
    }

    /// Galois Field multiplication by 11 (0x0B) in GF(2^8).
    #[inline]
    fn gf_mul11(x: u8) -> u8 {
        Self::gf_mul2(Self::gf_mul2(Self::gf_mul2(x)) ^ x) ^ x
    }

    /// Galois Field multiplication by 13 (0x0D) in GF(2^8).
    #[inline]
    fn gf_mul13(x: u8) -> u8 {
        Self::gf_mul2(Self::gf_mul2(Self::gf_mul2(x) ^ x)) ^ x
    }

    /// Galois Field multiplication by 14 (0x0E) in GF(2^8).
    #[inline]
    fn gf_mul14(x: u8) -> u8 {
        Self::gf_mul2(Self::gf_mul2(Self::gf_mul2(x) ^ x) ^ x)
    }

    /// AES key expansion: expands a 16-byte key into 11 round keys (176 bytes).
    fn key_expansion(key: &[u8; 16]) -> [[u8; 16]; 11] {
        let mut expanded = [[0u8; 16]; 11];
        let mut w = [0u32; 44]; // 4 * (10 + 1) = 44 words

        // Copy original key into first 4 words.
        for i in 0..Self::NK {
            w[i] = u32::from_be_bytes([
                key[4 * i],
                key[4 * i + 1],
                key[4 * i + 2],
                key[4 * i + 3],
            ]);
        }

        // Expand remaining words.
        for i in Self::NK..44 {
            let mut temp = w[i - 1];

            if i % Self::NK == 0 {
                // RotWord: rotate left by one byte.
                temp = (temp << 8) | (temp >> 24);
                // SubWord: apply S-box to each byte.
                let b = temp.to_be_bytes();
                temp = u32::from_be_bytes([
                    AES_SBOX[b[0] as usize],
                    AES_SBOX[b[1] as usize],
                    AES_SBOX[b[2] as usize],
                    AES_SBOX[b[3] as usize],
                ]);
                // XOR with Rcon.
                temp ^= (AES_RCON[i / Self::NK] as u32) << 24;
            }

            w[i] = w[i - Self::NK] ^ temp;
        }

        // Pack words into round key arrays.
        for round in 0..11 {
            for word in 0..4 {
                let bytes = w[round * 4 + word].to_be_bytes();
                expanded[round][word * 4] = bytes[0];
                expanded[round][word * 4 + 1] = bytes[1];
                expanded[round][word * 4 + 2] = bytes[2];
                expanded[round][word * 4 + 3] = bytes[3];
            }
        }

        expanded
    }

    /// AddRoundKey: XOR state with round key.
    #[inline]
    fn add_round_key(state: &mut [u8; 16], round_key: &[u8; 16]) {
        for i in 0..16 {
            state[i] ^= round_key[i];
        }
    }

    /// SubBytes: apply S-box substitution.
    #[inline]
    fn sub_bytes(state: &mut [u8; 16]) {
        for byte in state.iter_mut() {
            *byte = AES_SBOX[*byte as usize];
        }
    }

    /// InvSubBytes: apply inverse S-box substitution.
    #[inline]
    fn inv_sub_bytes(state: &mut [u8; 16]) {
        for byte in state.iter_mut() {
            *byte = AES_INV_SBOX[*byte as usize];
        }
    }

    /// ShiftRows: cyclically shift rows of the state matrix.
    ///
    /// State is stored column-major: index = row + 4 * col.
    fn shift_rows(state: &mut [u8; 16]) {
        // Row 0: no shift.
        // Row 1: shift left by 1.
        let tmp = state[1];
        state[1] = state[5];
        state[5] = state[9];
        state[9] = state[13];
        state[13] = tmp;
        // Row 2: shift left by 2.
        let tmp1 = state[2];
        let tmp2 = state[6];
        state[2] = state[10];
        state[6] = state[14];
        state[10] = tmp1;
        state[14] = tmp2;
        // Row 3: shift left by 3 (= shift right by 1).
        let tmp = state[3];
        state[3] = state[15];
        state[15] = state[11];
        state[11] = state[7];
        state[7] = tmp;
    }

    /// InvShiftRows: inverse of ShiftRows.
    fn inv_shift_rows(state: &mut [u8; 16]) {
        // Row 1: shift right by 1.
        let tmp = state[13];
        state[13] = state[9];
        state[9] = state[5];
        state[5] = state[1];
        state[1] = tmp;
        // Row 2: shift right by 2.
        let tmp1 = state[2];
        let tmp2 = state[6];
        state[2] = state[10];
        state[6] = state[14];
        state[10] = tmp1;
        state[14] = tmp2;
        // Row 3: shift right by 3 (= shift left by 1).
        let tmp = state[3];
        state[3] = state[7];
        state[7] = state[11];
        state[11] = state[15];
        state[15] = tmp;
    }

    /// MixColumns: mix columns of the state matrix.
    ///
    /// Each column is treated as a polynomial over GF(2^8) and multiplied
    /// modulo x^4+1 by {03}x^3 + {01}x^2 + {01}x + {02}.
    fn mix_columns(state: &mut [u8; 16]) {
        for col in 0..4 {
            let i = col * 4;
            let s0 = state[i];
            let s1 = state[i + 1];
            let s2 = state[i + 2];
            let s3 = state[i + 3];

            state[i]     = Self::gf_mul2(s0) ^ Self::gf_mul3(s1) ^ s2 ^ s3;
            state[i + 1] = s0 ^ Self::gf_mul2(s1) ^ Self::gf_mul3(s2) ^ s3;
            state[i + 2] = s0 ^ s1 ^ Self::gf_mul2(s2) ^ Self::gf_mul3(s3);
            state[i + 3] = Self::gf_mul3(s0) ^ s1 ^ s2 ^ Self::gf_mul2(s3);
        }
    }

    /// InvMixColumns: inverse of MixColumns.
    ///
    /// Multiplies by {0B}x^3 + {0D}x^2 + {09}x + {0E}.
    fn inv_mix_columns(state: &mut [u8; 16]) {
        for col in 0..4 {
            let i = col * 4;
            let s0 = state[i];
            let s1 = state[i + 1];
            let s2 = state[i + 2];
            let s3 = state[i + 3];

            state[i]     = Self::gf_mul14(s0) ^ Self::gf_mul11(s1) ^ Self::gf_mul13(s2) ^ Self::gf_mul9(s3);
            state[i + 1] = Self::gf_mul9(s0) ^ Self::gf_mul14(s1) ^ Self::gf_mul11(s2) ^ Self::gf_mul13(s3);
            state[i + 2] = Self::gf_mul13(s0) ^ Self::gf_mul9(s1) ^ Self::gf_mul14(s2) ^ Self::gf_mul11(s3);
            state[i + 3] = Self::gf_mul11(s0) ^ Self::gf_mul13(s1) ^ Self::gf_mul9(s2) ^ Self::gf_mul14(s3);
        }
    }

    /// Encrypts a single 16-byte block with AES-128.
    pub fn encrypt_block(plaintext: &[u8; 16], key: &[u8; 16]) -> [u8; 16] {
        let round_keys = Self::key_expansion(key);
        let mut state = *plaintext;

        // Initial round key addition.
        Self::add_round_key(&mut state, &round_keys[0]);

        // Rounds 1 through 9.
        for round in 1..Self::NR {
            Self::sub_bytes(&mut state);
            Self::shift_rows(&mut state);
            Self::mix_columns(&mut state);
            Self::add_round_key(&mut state, &round_keys[round]);
        }

        // Final round (no MixColumns).
        Self::sub_bytes(&mut state);
        Self::shift_rows(&mut state);
        Self::add_round_key(&mut state, &round_keys[Self::NR]);

        state
    }

    /// Decrypts a single 16-byte block with AES-128.
    pub fn decrypt_block(ciphertext: &[u8; 16], key: &[u8; 16]) -> [u8; 16] {
        let round_keys = Self::key_expansion(key);
        let mut state = *ciphertext;

        // Initial round key addition (last round key).
        Self::add_round_key(&mut state, &round_keys[Self::NR]);

        // Rounds 9 down through 1.
        for round in (1..Self::NR).rev() {
            Self::inv_shift_rows(&mut state);
            Self::inv_sub_bytes(&mut state);
            Self::add_round_key(&mut state, &round_keys[round]);
            Self::inv_mix_columns(&mut state);
        }

        // Final inverse round (no InvMixColumns).
        Self::inv_shift_rows(&mut state);
        Self::inv_sub_bytes(&mut state);
        Self::add_round_key(&mut state, &round_keys[0]);

        state
    }

    /// PKCS7 padding: pads data to a multiple of 16 bytes.
    fn pkcs7_pad(data: &[u8]) -> Vec<u8> {
        let pad_len = 16 - (data.len() % 16);
        let mut padded = data.to_vec();
        for _ in 0..pad_len {
            padded.push(pad_len as u8);
        }
        padded
    }

    /// PKCS7 unpadding: removes padding from decrypted data.
    fn pkcs7_unpad(data: &[u8]) -> Option<Vec<u8>> {
        if data.is_empty() || data.len() % 16 != 0 {
            return None;
        }
        let pad_byte = *data.last()? as usize;
        if pad_byte == 0 || pad_byte > 16 {
            return None;
        }
        // Verify all padding bytes.
        for &b in &data[data.len() - pad_byte..] {
            if b as usize != pad_byte {
                return None;
            }
        }
        Some(data[..data.len() - pad_byte].to_vec())
    }

    /// XORs two 16-byte blocks.
    #[inline]
    fn xor_blocks(a: &mut [u8; 16], b: &[u8; 16]) {
        for i in 0..16 {
            a[i] ^= b[i];
        }
    }

    /// Encrypts data using AES-128 in CBC mode with PKCS7 padding.
    ///
    /// # Arguments
    /// * `data` — plaintext of any length
    /// * `key` — 16-byte encryption key
    /// * `iv` — 16-byte initialization vector
    pub fn encrypt_cbc(data: &[u8], key: &[u8; 16], iv: &[u8; 16]) -> Vec<u8> {
        let padded = Self::pkcs7_pad(data);
        let num_blocks = padded.len() / 16;
        let mut output = Vec::with_capacity(padded.len());
        let mut prev_cipher = *iv;

        for block_idx in 0..num_blocks {
            let mut block = [0u8; 16];
            block.copy_from_slice(&padded[block_idx * 16..(block_idx + 1) * 16]);

            // XOR with previous ciphertext block (or IV for the first).
            Self::xor_blocks(&mut block, &prev_cipher);

            // Encrypt.
            let cipher_block = Self::encrypt_block(&block, key);
            output.extend_from_slice(&cipher_block);
            prev_cipher = cipher_block;
        }

        output
    }

    /// Decrypts data encrypted with AES-128 CBC mode.
    ///
    /// Returns `None` if the ciphertext length is not a multiple of 16 or
    /// if padding is invalid.
    pub fn decrypt_cbc(ciphertext: &[u8], key: &[u8; 16], iv: &[u8; 16]) -> Option<Vec<u8>> {
        if ciphertext.is_empty() || ciphertext.len() % 16 != 0 {
            return None;
        }

        let num_blocks = ciphertext.len() / 16;
        let mut decrypted = Vec::with_capacity(ciphertext.len());
        let mut prev_cipher = *iv;

        for block_idx in 0..num_blocks {
            let mut cipher_block = [0u8; 16];
            cipher_block.copy_from_slice(&ciphertext[block_idx * 16..(block_idx + 1) * 16]);

            let mut plain_block = Self::decrypt_block(&cipher_block, key);
            Self::xor_blocks(&mut plain_block, &prev_cipher);
            decrypted.extend_from_slice(&plain_block);
            prev_cipher = cipher_block;
        }

        Self::pkcs7_unpad(&decrypted)
    }
}

// ---------------------------------------------------------------------------
// CRC-32 (ISO-HDLC / Ethernet)
// ---------------------------------------------------------------------------

/// Pre-computed CRC-32 lookup table (polynomial 0xEDB88320, reflected).
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
};

/// CRC-32/ISO-HDLC (Ethernet CRC) calculator.
///
/// Uses a 256-entry pre-computed table for fast byte-at-a-time processing.
///
/// # Example
/// ```
/// use genovo_core::crypto::Crc32;
///
/// let checksum = Crc32::compute(b"123456789");
/// assert_eq!(checksum, 0xCBF43926);
/// ```
pub struct Crc32;

impl Crc32 {
    /// Computes the CRC-32 checksum of `data`.
    pub fn compute(data: &[u8]) -> u32 {
        let mut hasher = Crc32Hasher::new();
        hasher.update(data);
        hasher.finalize()
    }

    /// Returns the CRC-32 checksum as a hex string.
    pub fn compute_hex(data: &[u8]) -> String {
        format!("{:08x}", Self::compute(data))
    }
}

/// Incremental CRC-32 hasher. Allows feeding data in chunks.
///
/// # Example
/// ```
/// use genovo_core::crypto::Crc32Hasher;
///
/// let mut hasher = Crc32Hasher::new();
/// hasher.update(b"hello");
/// hasher.update(b" world");
/// let crc = hasher.finalize();
/// ```
pub struct Crc32Hasher {
    crc: u32,
}

impl Crc32Hasher {
    /// Creates a new CRC-32 hasher.
    pub fn new() -> Self {
        Self { crc: 0xFFFFFFFF }
    }

    /// Feeds more data into the hasher.
    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            let index = ((self.crc ^ byte as u32) & 0xFF) as usize;
            self.crc = (self.crc >> 8) ^ CRC32_TABLE[index];
        }
    }

    /// Finalizes and returns the CRC-32 value.
    pub fn finalize(&self) -> u32 {
        self.crc ^ 0xFFFFFFFF
    }

    /// Resets the hasher.
    pub fn reset(&mut self) {
        self.crc = 0xFFFFFFFF;
    }
}

impl Default for Crc32Hasher {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HMAC-SHA256
// ---------------------------------------------------------------------------

/// HMAC-SHA256 (RFC 2104) message authentication code.
///
/// # Example
/// ```
/// use genovo_core::crypto::Hmac;
///
/// let mac = Hmac::compute(b"secret-key", b"message");
/// assert_eq!(mac.len(), 32);
/// ```
pub struct Hmac;

impl Hmac {
    /// SHA-256 block size (bytes).
    const BLOCK_SIZE: usize = 64;

    /// Computes HMAC-SHA256.
    ///
    /// If the key is longer than 64 bytes, it is first hashed with SHA-256.
    /// If shorter, it is zero-padded to 64 bytes.
    pub fn compute(key: &[u8], message: &[u8]) -> [u8; 32] {
        // Step 1: Normalize key to BLOCK_SIZE bytes.
        let mut key_block = [0u8; Self::BLOCK_SIZE];
        if key.len() > Self::BLOCK_SIZE {
            let hashed = Sha256::hash(key);
            key_block[..32].copy_from_slice(&hashed);
        } else {
            key_block[..key.len()].copy_from_slice(key);
        }

        // Step 2: Create inner and outer padded keys.
        let mut ipad = [0x36u8; Self::BLOCK_SIZE];
        let mut opad = [0x5cu8; Self::BLOCK_SIZE];
        for i in 0..Self::BLOCK_SIZE {
            ipad[i] ^= key_block[i];
            opad[i] ^= key_block[i];
        }

        // Step 3: Inner hash = SHA256(ipad || message).
        let mut inner_hasher = Sha256Hasher::new();
        inner_hasher.update(&ipad);
        inner_hasher.update(message);
        let inner_hash = inner_hasher.finalize();

        // Step 4: Outer hash = SHA256(opad || inner_hash).
        let mut outer_hasher = Sha256Hasher::new();
        outer_hasher.update(&opad);
        outer_hasher.update(&inner_hash);
        outer_hasher.finalize()
    }

    /// Verifies an HMAC by computing it and comparing in constant time.
    pub fn verify(key: &[u8], message: &[u8], expected: &[u8; 32]) -> bool {
        let computed = Self::compute(key, message);
        // Constant-time comparison to prevent timing attacks.
        let mut diff = 0u8;
        for i in 0..32 {
            diff |= computed[i] ^ expected[i];
        }
        diff == 0
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Converts a byte slice to a hex string.
pub fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Converts a hex string to bytes. Returns `None` if the string is invalid.
pub fn hex_to_bytes(hex: &str) -> Option<Vec<u8>> {
    if hex.len() % 2 != 0 {
        return None;
    }
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    for i in (0..hex.len()).step_by(2) {
        let byte = u8::from_str_radix(&hex[i..i + 2], 16).ok()?;
        bytes.push(byte);
    }
    Some(bytes)
}

/// Generates a simple deterministic "key" from a seed (NOT cryptographically
/// secure — use for testing and save-file obfuscation only).
pub fn derive_key_simple(seed: &[u8]) -> [u8; 16] {
    let hash = Sha256::hash(seed);
    let mut key = [0u8; 16];
    key.copy_from_slice(&hash[..16]);
    key
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SHA-256 tests --------------------------------------------------------

    #[test]
    fn sha256_empty() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let hash = Sha256::hash(b"");
        let hex = Sha256::hash_hex(b"");
        assert_eq!(hex, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(hash[0], 0xe3);
        assert_eq!(hash[31], 0x55);
    }

    #[test]
    fn sha256_hello() {
        // SHA-256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        let hex = Sha256::hash_hex(b"hello");
        assert_eq!(hex, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");
    }

    #[test]
    fn sha256_abc() {
        // SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        let hex = Sha256::hash_hex(b"abc");
        assert_eq!(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn sha256_long_message() {
        // SHA-256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
        // = 248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1
        let msg = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        let hex = Sha256::hash_hex(msg);
        assert_eq!(hex, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    }

    #[test]
    fn sha256_incremental() {
        let mut hasher = Sha256Hasher::new();
        hasher.update(b"hello");
        hasher.update(b" ");
        hasher.update(b"world");
        let hash1 = hasher.finalize();

        let hash2 = Sha256::hash(b"hello world");
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn sha256_incremental_many_chunks() {
        // Feed data one byte at a time.
        let data = b"The quick brown fox jumps over the lazy dog";
        let expected = Sha256::hash(data);

        let mut hasher = Sha256Hasher::new();
        for &byte in data.iter() {
            hasher.update(&[byte]);
        }
        let result = hasher.finalize();
        assert_eq!(result, expected);
    }

    #[test]
    fn sha256_reset() {
        let mut hasher = Sha256Hasher::new();
        hasher.update(b"garbage");
        hasher.reset();
        hasher.update(b"hello");
        let hash = hasher.finalize();
        assert_eq!(hash, Sha256::hash(b"hello"));
    }

    // -- AES-128 tests --------------------------------------------------------

    #[test]
    fn aes128_encrypt_decrypt_block() {
        let key = [0u8; 16];
        let plaintext = [0u8; 16];
        let ciphertext = Aes128::encrypt_block(&plaintext, &key);
        let decrypted = Aes128::decrypt_block(&ciphertext, &key);
        assert_eq!(decrypted, plaintext);
        assert_ne!(ciphertext, plaintext); // should be different
    }

    #[test]
    fn aes128_known_vector() {
        // NIST AES-128 test vector:
        // Key:       000102030405060708090a0b0c0d0e0f
        // Plaintext: 00112233445566778899aabbccddeeff
        // Expected:  69c4e0d86a7b0430d8cdb78070b4c55a
        let key: [u8; 16] = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                              0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f];
        let plaintext: [u8; 16] = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                                    0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff];
        let expected: [u8; 16] = [0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30,
                                   0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a];

        let ciphertext = Aes128::encrypt_block(&plaintext, &key);
        assert_eq!(ciphertext, expected);

        let decrypted = Aes128::decrypt_block(&ciphertext, &key);
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn aes128_cbc_roundtrip() {
        let key = [0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
                   0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c];
        let iv = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                  0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f];
        let plaintext = b"Hello, AES-128 CBC mode! This is a test.";

        let ciphertext = Aes128::encrypt_cbc(plaintext, &key, &iv);
        assert_ne!(&ciphertext[..], &plaintext[..]);
        assert_eq!(ciphertext.len() % 16, 0);

        let decrypted = Aes128::decrypt_cbc(&ciphertext, &key, &iv).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn aes128_cbc_empty() {
        let key = [0u8; 16];
        let iv = [0u8; 16];
        let ciphertext = Aes128::encrypt_cbc(b"", &key, &iv);
        // Empty plaintext still produces one block due to PKCS7 padding.
        assert_eq!(ciphertext.len(), 16);
        let decrypted = Aes128::decrypt_cbc(&ciphertext, &key, &iv).unwrap();
        assert!(decrypted.is_empty());
    }

    #[test]
    fn aes128_cbc_exact_block() {
        let key = [0xAA; 16];
        let iv = [0xBB; 16];
        let plaintext = [0xCC; 16]; // exactly one block
        let ciphertext = Aes128::encrypt_cbc(&plaintext, &key, &iv);
        // PKCS7 adds a full padding block when input is block-aligned.
        assert_eq!(ciphertext.len(), 32);
        let decrypted = Aes128::decrypt_cbc(&ciphertext, &key, &iv).unwrap();
        assert_eq!(&decrypted[..], &plaintext[..]);
    }

    #[test]
    fn aes128_pkcs7_padding() {
        // Test padding for various sizes.
        for size in 0..=32 {
            let data = vec![0x42u8; size];
            let padded = Aes128::pkcs7_pad(&data);
            assert_eq!(padded.len() % 16, 0);
            assert!(padded.len() >= data.len() + 1);
            let unpadded = Aes128::pkcs7_unpad(&padded).unwrap();
            assert_eq!(unpadded, data);
        }
    }

    #[test]
    fn aes128_gf_mul() {
        // Test GF(2^8) multiplication properties.
        assert_eq!(Aes128::gf_mul2(0), 0);
        assert_eq!(Aes128::gf_mul2(1), 2);
        assert_eq!(Aes128::gf_mul2(0x80), 0x1b); // reduction by x^8+x^4+x^3+x+1
        assert_eq!(Aes128::gf_mul3(1), 3);
    }

    // -- CRC-32 tests ---------------------------------------------------------

    #[test]
    fn crc32_check_value() {
        // The CRC-32 of "123456789" should be 0xCBF43926.
        let crc = Crc32::compute(b"123456789");
        assert_eq!(crc, 0xCBF43926);
    }

    #[test]
    fn crc32_empty() {
        let crc = Crc32::compute(b"");
        assert_eq!(crc, 0x00000000);
    }

    #[test]
    fn crc32_incremental() {
        let mut hasher = Crc32Hasher::new();
        hasher.update(b"1234");
        hasher.update(b"56789");
        assert_eq!(hasher.finalize(), 0xCBF43926);
    }

    #[test]
    fn crc32_hex() {
        let hex = Crc32::compute_hex(b"123456789");
        assert_eq!(hex, "cbf43926");
    }

    #[test]
    fn crc32_reset() {
        let mut hasher = Crc32Hasher::new();
        hasher.update(b"garbage");
        hasher.reset();
        hasher.update(b"123456789");
        assert_eq!(hasher.finalize(), 0xCBF43926);
    }

    // -- HMAC tests -----------------------------------------------------------

    #[test]
    fn hmac_basic() {
        let mac = Hmac::compute(b"key", b"message");
        assert_eq!(mac.len(), 32);
        // Should be deterministic.
        let mac2 = Hmac::compute(b"key", b"message");
        assert_eq!(mac, mac2);
    }

    #[test]
    fn hmac_different_keys() {
        let mac1 = Hmac::compute(b"key1", b"message");
        let mac2 = Hmac::compute(b"key2", b"message");
        assert_ne!(mac1, mac2);
    }

    #[test]
    fn hmac_different_messages() {
        let mac1 = Hmac::compute(b"key", b"message1");
        let mac2 = Hmac::compute(b"key", b"message2");
        assert_ne!(mac1, mac2);
    }

    #[test]
    fn hmac_verify() {
        let mac = Hmac::compute(b"secret", b"data");
        assert!(Hmac::verify(b"secret", b"data", &mac));
        assert!(!Hmac::verify(b"wrong", b"data", &mac));
    }

    #[test]
    fn hmac_long_key() {
        // Key longer than block size should be hashed first.
        let long_key = vec![0xAA; 100];
        let mac = Hmac::compute(&long_key, b"test");
        assert_eq!(mac.len(), 32);
        // Verify it's consistent.
        let mac2 = Hmac::compute(&long_key, b"test");
        assert_eq!(mac, mac2);
    }

    #[test]
    fn hmac_rfc4231_test1() {
        // RFC 4231 Test Case 1:
        // Key  = 0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b (20 bytes)
        // Data = "Hi There"
        // HMAC-SHA-256 = b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7
        let key = vec![0x0b; 20];
        let mac = Hmac::compute(&key, b"Hi There");
        let hex: String = mac.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(hex, "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
    }

    // -- Utility tests --------------------------------------------------------

    #[test]
    fn hex_roundtrip() {
        let data = [0xDE, 0xAD, 0xBE, 0xEF];
        let hex = bytes_to_hex(&data);
        assert_eq!(hex, "deadbeef");
        let bytes = hex_to_bytes(&hex).unwrap();
        assert_eq!(bytes, data);
    }

    #[test]
    fn hex_invalid() {
        assert!(hex_to_bytes("xyz").is_none());
        assert!(hex_to_bytes("abc").is_none()); // odd length
    }

    #[test]
    fn derive_key() {
        let key = derive_key_simple(b"my-game-save-v1");
        assert_eq!(key.len(), 16);
        // Deterministic.
        let key2 = derive_key_simple(b"my-game-save-v1");
        assert_eq!(key, key2);
    }
}
