//! Network protocol framework.
//!
//! Defines a message-based protocol layer with type registration, serialization,
//! channels, fragmentation/reassembly, compression, and encryption stubs.
//!
//! ## Architecture
//!
//! Messages are registered with unique IDs in a [`MessageRegistry`]. Each
//! message type implements the [`NetMessage`] trait for serialization. Messages
//! are assigned to [`Channel`]s that control reliability, ordering, and priority.
//!
//! Large messages are fragmented across multiple packets and reassembled at the
//! receiver. An optional compression pass reduces bandwidth for large payloads.
//!
//! ## Wire format (fragmented message)
//!
//! ```text
//! Fragment header (8 bytes):
//!   [u16 message_id]
//!   [u16 fragment_id]        // unique per fragmented message
//!   [u8  fragment_index]     // 0-based index of this fragment
//!   [u8  fragment_count]     // total fragments for this message
//!   [u16 fragment_data_len]  // length of payload in this fragment
//!   [... fragment data ...]
//! ```

use std::collections::HashMap;
use std::time::Instant;

use genovo_core::{EngineError, EngineResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum payload per fragment.
pub const MAX_FRAGMENT_PAYLOAD: usize = 1024;

/// Fragment header size.
pub const FRAGMENT_HEADER_SIZE: usize = 8;

/// Maximum number of fragments per message.
pub const MAX_FRAGMENTS: u8 = 255;

/// Maximum reassembly buffer entries.
pub const MAX_REASSEMBLY_ENTRIES: usize = 64;

/// Reassembly timeout (if all fragments haven't arrived within this time, discard).
pub const REASSEMBLY_TIMEOUT_SECS: f64 = 5.0;

/// Compression threshold: messages larger than this are compressed.
pub const COMPRESSION_THRESHOLD: usize = 256;

/// Maximum message size before fragmentation.
pub const MAX_UNFRAGMENTED_SIZE: usize = MAX_FRAGMENT_PAYLOAD;

// ---------------------------------------------------------------------------
// NetMessage trait
// ---------------------------------------------------------------------------

/// Trait for network messages.
///
/// All message types must implement this trait to be sent over the network.
/// Each message type has a unique ID for dispatch on the receiving side.
pub trait NetMessage: Send + Sync {
    /// Returns the unique message type ID.
    fn message_id(&self) -> u16;

    /// Serialize this message to bytes.
    fn serialize(&self) -> Vec<u8>;

    /// Returns the channel this message should be sent on.
    fn channel(&self) -> u8 {
        0 // default channel
    }

    /// Returns a human-readable name for this message type.
    fn message_name(&self) -> &str {
        "unknown"
    }
}

// ---------------------------------------------------------------------------
// MessageFactory
// ---------------------------------------------------------------------------

/// A factory function that creates a message from raw bytes.
pub type MessageDeserializer = Box<dyn Fn(&[u8]) -> Option<Box<dyn NetMessage>> + Send + Sync>;

// ---------------------------------------------------------------------------
// MessageRegistration
// ---------------------------------------------------------------------------

/// Registration entry for a message type.
struct MessageRegistration {
    /// The unique message ID.
    id: u16,
    /// Human-readable name.
    name: String,
    /// Channel assignment.
    channel: u8,
    /// Deserializer factory.
    deserializer: MessageDeserializer,
}

// ---------------------------------------------------------------------------
// MessageRegistry
// ---------------------------------------------------------------------------

/// Registry for network message types.
///
/// Maps message IDs to deserialization factories so incoming bytes can be
/// decoded into the correct message type.
pub struct MessageRegistry {
    /// Registered message types, keyed by ID.
    registrations: HashMap<u16, MessageRegistration>,
    /// Name to ID mapping.
    name_to_id: HashMap<String, u16>,
}

impl MessageRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            registrations: HashMap::new(),
            name_to_id: HashMap::new(),
        }
    }

    /// Register a message type with a unique ID.
    pub fn register(
        &mut self,
        id: u16,
        name: impl Into<String>,
        channel: u8,
        deserializer: MessageDeserializer,
    ) -> EngineResult<()> {
        let name = name.into();

        if self.registrations.contains_key(&id) {
            return Err(EngineError::InvalidArgument(format!(
                "message ID {} already registered",
                id
            )));
        }

        if self.name_to_id.contains_key(&name) {
            return Err(EngineError::InvalidArgument(format!(
                "message name '{}' already registered",
                name
            )));
        }

        self.name_to_id.insert(name.clone(), id);
        self.registrations.insert(
            id,
            MessageRegistration {
                id,
                name,
                channel,
                deserializer,
            },
        );

        Ok(())
    }

    /// Unregister a message type by ID.
    pub fn unregister(&mut self, id: u16) -> bool {
        if let Some(reg) = self.registrations.remove(&id) {
            self.name_to_id.remove(&reg.name);
            true
        } else {
            false
        }
    }

    /// Deserialize a message from its ID and raw data.
    pub fn deserialize(&self, id: u16, data: &[u8]) -> Option<Box<dyn NetMessage>> {
        let reg = self.registrations.get(&id)?;
        (reg.deserializer)(data)
    }

    /// Serialize a message to bytes with its ID prefix.
    pub fn serialize_message(&self, msg: &dyn NetMessage) -> Vec<u8> {
        let id = msg.message_id();
        let payload = msg.serialize();
        let mut buf = Vec::with_capacity(2 + payload.len());
        buf.extend_from_slice(&id.to_be_bytes());
        buf.extend_from_slice(&payload);
        buf
    }

    /// Returns the message ID for a given name.
    pub fn get_id(&self, name: &str) -> Option<u16> {
        self.name_to_id.get(name).copied()
    }

    /// Returns the message name for a given ID.
    pub fn get_name(&self, id: u16) -> Option<&str> {
        self.registrations.get(&id).map(|r| r.name.as_str())
    }

    /// Returns the channel for a given message ID.
    pub fn get_channel(&self, id: u16) -> Option<u8> {
        self.registrations.get(&id).map(|r| r.channel)
    }

    /// Returns the number of registered message types.
    pub fn count(&self) -> usize {
        self.registrations.len()
    }

    /// Returns true if a message ID is registered.
    pub fn is_registered(&self, id: u16) -> bool {
        self.registrations.contains_key(&id)
    }
}

impl Default for MessageRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Channel
// ---------------------------------------------------------------------------

/// A network channel with specific reliability and ordering guarantees.
#[derive(Debug, Clone)]
pub struct Channel {
    /// Channel ID.
    pub id: u8,
    /// Human-readable name.
    pub name: String,
    /// Reliability mode.
    pub reliability: ChannelReliability,
    /// Ordering mode.
    pub ordering: ChannelOrdering,
    /// Priority (higher = more important).
    pub priority: u8,
    /// Maximum bandwidth per second (bytes).
    pub max_bandwidth_bps: u32,
    /// Bytes sent this frame.
    bytes_this_frame: u32,
    /// Next sequence number for ordered delivery.
    next_sequence: u16,
}

/// Reliability mode for a channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelReliability {
    /// No delivery guarantee.
    Unreliable,
    /// Guaranteed delivery.
    Reliable,
}

/// Ordering mode for a channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelOrdering {
    /// No ordering guarantee.
    Unordered,
    /// Messages delivered in order (out-of-order messages are buffered).
    Ordered,
    /// Only the most recent message is delivered (older ones are dropped).
    Sequenced,
}

impl Channel {
    /// Create a new channel.
    pub fn new(
        id: u8,
        name: impl Into<String>,
        reliability: ChannelReliability,
        ordering: ChannelOrdering,
        priority: u8,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            reliability,
            ordering,
            priority,
            max_bandwidth_bps: 1_000_000,
            bytes_this_frame: 0,
            next_sequence: 0,
        }
    }

    /// Create a reliable ordered channel (most common).
    pub fn reliable_ordered(id: u8, name: impl Into<String>) -> Self {
        Self::new(
            id,
            name,
            ChannelReliability::Reliable,
            ChannelOrdering::Ordered,
            128,
        )
    }

    /// Create an unreliable sequenced channel (good for state updates).
    pub fn unreliable_sequenced(id: u8, name: impl Into<String>) -> Self {
        Self::new(
            id,
            name,
            ChannelReliability::Unreliable,
            ChannelOrdering::Sequenced,
            64,
        )
    }

    /// Allocate a sequence number.
    pub fn next_sequence(&mut self) -> u16 {
        let seq = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1);
        seq
    }

    /// Check if there is bandwidth budget remaining.
    pub fn has_budget(&self, bytes: u32) -> bool {
        let budget_per_frame = self.max_bandwidth_bps / 60;
        self.bytes_this_frame + bytes <= budget_per_frame
    }

    /// Consume bandwidth budget.
    pub fn consume(&mut self, bytes: u32) {
        self.bytes_this_frame += bytes;
    }

    /// Reset per-frame counters.
    pub fn reset_frame(&mut self) {
        self.bytes_this_frame = 0;
    }
}

// ---------------------------------------------------------------------------
// ChannelConfig
// ---------------------------------------------------------------------------

/// Configuration for the channel system.
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    /// Registered channels.
    pub channels: Vec<Channel>,
}

impl ChannelConfig {
    /// Create a default channel configuration.
    pub fn default_config() -> Self {
        Self {
            channels: vec![
                Channel::reliable_ordered(0, "reliable"),
                Channel::unreliable_sequenced(1, "unreliable"),
                Channel::new(
                    2,
                    "voice",
                    ChannelReliability::Unreliable,
                    ChannelOrdering::Unordered,
                    32,
                ),
            ],
        }
    }

    /// Add a channel.
    pub fn add_channel(&mut self, channel: Channel) {
        self.channels.push(channel);
    }

    /// Get a channel by ID.
    pub fn get_channel(&self, id: u8) -> Option<&Channel> {
        self.channels.iter().find(|c| c.id == id)
    }

    /// Get a mutable channel by ID.
    pub fn get_channel_mut(&mut self, id: u8) -> Option<&mut Channel> {
        self.channels.iter_mut().find(|c| c.id == id)
    }

    /// Reset all channels for a new frame.
    pub fn reset_all(&mut self) {
        for ch in &mut self.channels {
            ch.reset_frame();
        }
    }
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

// ---------------------------------------------------------------------------
// Fragment
// ---------------------------------------------------------------------------

/// A single fragment of a larger message.
#[derive(Debug, Clone)]
pub struct Fragment {
    /// The message ID this fragment belongs to.
    pub message_id: u16,
    /// Unique ID for this fragmented message (to distinguish concurrent messages).
    pub fragment_id: u16,
    /// Index of this fragment (0-based).
    pub fragment_index: u8,
    /// Total number of fragments.
    pub fragment_count: u8,
    /// The fragment payload data.
    pub data: Vec<u8>,
}

impl Fragment {
    /// Encode a fragment to wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(FRAGMENT_HEADER_SIZE + self.data.len());
        buf.extend_from_slice(&self.message_id.to_be_bytes());
        buf.extend_from_slice(&self.fragment_id.to_be_bytes());
        buf.push(self.fragment_index);
        buf.push(self.fragment_count);
        buf.extend_from_slice(&(self.data.len() as u16).to_be_bytes());
        buf.extend_from_slice(&self.data);
        buf
    }

    /// Decode a fragment from wire bytes.
    pub fn decode(data: &[u8]) -> Option<(Self, usize)> {
        if data.len() < FRAGMENT_HEADER_SIZE {
            return None;
        }

        let message_id = u16::from_be_bytes([data[0], data[1]]);
        let fragment_id = u16::from_be_bytes([data[2], data[3]]);
        let fragment_index = data[4];
        let fragment_count = data[5];
        let payload_len = u16::from_be_bytes([data[6], data[7]]) as usize;

        let total = FRAGMENT_HEADER_SIZE + payload_len;
        if data.len() < total {
            return None;
        }

        let payload = data[FRAGMENT_HEADER_SIZE..total].to_vec();

        Some((
            Self {
                message_id,
                fragment_id,
                fragment_index,
                fragment_count,
                data: payload,
            },
            total,
        ))
    }
}

// ---------------------------------------------------------------------------
// Fragmenter
// ---------------------------------------------------------------------------

/// Splits large messages into fragments and reassembles them.
pub struct Fragmenter {
    /// Next fragment ID to assign.
    next_fragment_id: u16,
}

impl Fragmenter {
    /// Create a new fragmenter.
    pub fn new() -> Self {
        Self {
            next_fragment_id: 0,
        }
    }

    /// Fragment a message into multiple pieces.
    ///
    /// If the message fits in a single fragment, returns a vec with one element.
    pub fn fragment(&mut self, message_id: u16, data: &[u8]) -> Vec<Fragment> {
        let fragment_id = self.next_fragment_id;
        self.next_fragment_id = self.next_fragment_id.wrapping_add(1);

        if data.len() <= MAX_FRAGMENT_PAYLOAD {
            return vec![Fragment {
                message_id,
                fragment_id,
                fragment_index: 0,
                fragment_count: 1,
                data: data.to_vec(),
            }];
        }

        let chunk_count = (data.len() + MAX_FRAGMENT_PAYLOAD - 1) / MAX_FRAGMENT_PAYLOAD;
        let fragment_count = chunk_count.min(MAX_FRAGMENTS as usize) as u8;

        let mut fragments = Vec::with_capacity(fragment_count as usize);
        for i in 0..fragment_count {
            let start = i as usize * MAX_FRAGMENT_PAYLOAD;
            let end = (start + MAX_FRAGMENT_PAYLOAD).min(data.len());
            fragments.push(Fragment {
                message_id,
                fragment_id,
                fragment_index: i,
                fragment_count,
                data: data[start..end].to_vec(),
            });
        }

        fragments
    }
}

impl Default for Fragmenter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ReassemblyBuffer
// ---------------------------------------------------------------------------

/// Tracks the state of a message being reassembled from fragments.
#[derive(Debug)]
struct ReassemblyEntry {
    /// The message ID.
    message_id: u16,
    /// The fragment ID.
    fragment_id: u16,
    /// Total expected fragments.
    fragment_count: u8,
    /// Received fragments, indexed by fragment_index.
    fragments: HashMap<u8, Vec<u8>>,
    /// When the first fragment was received.
    started_at: Instant,
}

impl ReassemblyEntry {
    fn new(fragment: &Fragment) -> Self {
        let mut fragments = HashMap::new();
        fragments.insert(fragment.fragment_index, fragment.data.clone());

        Self {
            message_id: fragment.message_id,
            fragment_id: fragment.fragment_id,
            fragment_count: fragment.fragment_count,
            fragments,
            started_at: Instant::now(),
        }
    }

    /// Add a fragment. Returns true if the message is now complete.
    fn add_fragment(&mut self, fragment: &Fragment) -> bool {
        self.fragments
            .insert(fragment.fragment_index, fragment.data.clone());
        self.is_complete()
    }

    /// Returns true if all fragments have been received.
    fn is_complete(&self) -> bool {
        self.fragments.len() == self.fragment_count as usize
    }

    /// Returns true if this entry has timed out.
    fn is_timed_out(&self) -> bool {
        Instant::now()
            .duration_since(self.started_at)
            .as_secs_f64()
            > REASSEMBLY_TIMEOUT_SECS
    }

    /// Assemble the complete message from all fragments.
    fn assemble(&self) -> Option<(u16, Vec<u8>)> {
        if !self.is_complete() {
            return None;
        }

        let mut data = Vec::new();
        for i in 0..self.fragment_count {
            let chunk = self.fragments.get(&i)?;
            data.extend_from_slice(chunk);
        }

        Some((self.message_id, data))
    }
}

/// Buffer for reassembling fragmented messages.
pub struct ReassemblyBuffer {
    /// In-progress reassemblies, keyed by fragment_id.
    entries: HashMap<u16, ReassemblyEntry>,
    /// Completed messages ready for delivery.
    completed: Vec<(u16, Vec<u8>)>,
    /// Total messages reassembled successfully.
    pub total_reassembled: u64,
    /// Total messages dropped due to timeout.
    pub total_timed_out: u64,
}

impl ReassemblyBuffer {
    /// Create a new reassembly buffer.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            completed: Vec::new(),
            total_reassembled: 0,
            total_timed_out: 0,
        }
    }

    /// Process an incoming fragment.
    ///
    /// Returns true if a message was completed.
    pub fn receive_fragment(&mut self, fragment: Fragment) -> bool {
        // Single-fragment messages don't need reassembly.
        if fragment.fragment_count == 1 {
            self.completed
                .push((fragment.message_id, fragment.data.clone()));
            self.total_reassembled += 1;
            return true;
        }

        let frag_id = fragment.fragment_id;

        if let Some(entry) = self.entries.get_mut(&frag_id) {
            if entry.add_fragment(&fragment) {
                // Message complete.
                if let Some((msg_id, data)) = entry.assemble() {
                    self.completed.push((msg_id, data));
                    self.total_reassembled += 1;
                    self.entries.remove(&frag_id);
                    return true;
                }
            }
        } else {
            // New fragmented message.
            if self.entries.len() >= MAX_REASSEMBLY_ENTRIES {
                // Evict the oldest entry.
                let oldest = self
                    .entries
                    .iter()
                    .min_by_key(|(_, e)| e.started_at)
                    .map(|(&k, _)| k);
                if let Some(k) = oldest {
                    self.entries.remove(&k);
                    self.total_timed_out += 1;
                }
            }

            let entry = ReassemblyEntry::new(&fragment);
            if entry.is_complete() {
                if let Some((msg_id, data)) = entry.assemble() {
                    self.completed.push((msg_id, data));
                    self.total_reassembled += 1;
                    return true;
                }
            }
            self.entries.insert(frag_id, entry);
        }

        false
    }

    /// Clean up timed-out reassembly entries.
    pub fn cleanup(&mut self) {
        let timed_out: Vec<u16> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_timed_out())
            .map(|(&k, _)| k)
            .collect();

        for k in timed_out {
            self.entries.remove(&k);
            self.total_timed_out += 1;
        }
    }

    /// Take all completed messages.
    pub fn take_completed(&mut self) -> Vec<(u16, Vec<u8>)> {
        std::mem::take(&mut self.completed)
    }

    /// Returns the number of in-progress reassemblies.
    pub fn pending_count(&self) -> usize {
        self.entries.len()
    }
}

impl Default for ReassemblyBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Compression (simple LZ4-style RLE)
// ---------------------------------------------------------------------------

/// Simple run-length compression for network messages.
///
/// This is a lightweight compression scheme suitable for game state data
/// which often has long runs of identical bytes (e.g., zeroed fields).
pub struct Compressor;

impl Compressor {
    /// Compress data using simple run-length encoding.
    ///
    /// Format: [u8 tag, ...data]
    ///   tag < 128: literal run of `tag + 1` bytes follows
    ///   tag >= 128: repeat next byte `(tag - 128 + 3)` times
    pub fn compress(data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut output = Vec::with_capacity(data.len());
        let mut i = 0;

        while i < data.len() {
            // Check for a run of identical bytes.
            let run_start = i;
            let run_byte = data[i];
            while i < data.len() && data[i] == run_byte && (i - run_start) < 130 {
                i += 1;
            }
            let run_len = i - run_start;

            if run_len >= 3 {
                // RLE: tag >= 128.
                let tag = 128 + (run_len as u8 - 3);
                output.push(tag);
                output.push(run_byte);
            } else {
                // Literal: collect non-run bytes.
                i = run_start;
                let literal_start = i;

                loop {
                    if i >= data.len() || (i - literal_start) >= 128 {
                        break;
                    }

                    // Check if the next bytes form a run of 3+.
                    if i + 2 < data.len()
                        && data[i] == data[i + 1]
                        && data[i] == data[i + 2]
                    {
                        break;
                    }

                    i += 1;
                }

                let literal_len = i - literal_start;
                if literal_len > 0 {
                    let tag = (literal_len as u8) - 1;
                    output.push(tag);
                    output.extend_from_slice(&data[literal_start..literal_start + literal_len]);
                }
            }
        }

        output
    }

    /// Decompress data that was compressed with `compress`.
    pub fn decompress(data: &[u8]) -> EngineResult<Vec<u8>> {
        let mut output = Vec::new();
        let mut i = 0;

        while i < data.len() {
            let tag = data[i];
            i += 1;

            if tag >= 128 {
                // RLE run.
                let count = (tag - 128 + 3) as usize;
                if i >= data.len() {
                    return Err(EngineError::InvalidArgument(
                        "compressed data truncated (RLE)".into(),
                    ));
                }
                let byte = data[i];
                i += 1;
                for _ in 0..count {
                    output.push(byte);
                }
            } else {
                // Literal run.
                let count = (tag + 1) as usize;
                if i + count > data.len() {
                    return Err(EngineError::InvalidArgument(
                        "compressed data truncated (literal)".into(),
                    ));
                }
                output.extend_from_slice(&data[i..i + count]);
                i += count;
            }
        }

        Ok(output)
    }

    /// Returns true if the data is worth compressing (above threshold).
    pub fn should_compress(data: &[u8]) -> bool {
        data.len() >= COMPRESSION_THRESHOLD
    }
}

// ---------------------------------------------------------------------------
// Encryption (placeholder)
// ---------------------------------------------------------------------------

/// Placeholder encryption wrapper.
///
/// In a production implementation this would use AES-256-GCM or ChaCha20-Poly1305.
/// For now it provides the interface without actual cryptographic operations.
pub struct Encryption {
    /// Whether encryption is enabled.
    enabled: bool,
    /// The shared key (placeholder -- 32 bytes for AES-256).
    key: [u8; 32],
    /// Nonce counter for unique IVs.
    nonce_counter: u64,
}

impl Encryption {
    /// Create a new encryption wrapper (disabled by default).
    pub fn new() -> Self {
        Self {
            enabled: false,
            key: [0u8; 32],
            nonce_counter: 0,
        }
    }

    /// Enable encryption with the given key.
    pub fn enable(&mut self, key: [u8; 32]) {
        self.key = key;
        self.enabled = true;
    }

    /// Disable encryption.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Returns true if encryption is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// "Encrypt" data (placeholder: XOR with key for demonstration).
    ///
    /// In production, use a proper AEAD cipher.
    pub fn encrypt(&mut self, data: &[u8]) -> Vec<u8> {
        if !self.enabled {
            return data.to_vec();
        }

        self.nonce_counter += 1;
        let nonce = self.nonce_counter;

        // Prepend nonce (8 bytes) + XOR "encryption".
        let mut output = Vec::with_capacity(8 + data.len());
        output.extend_from_slice(&nonce.to_be_bytes());

        for (i, &byte) in data.iter().enumerate() {
            let key_byte = self.key[i % 32];
            let nonce_byte = ((nonce >> ((i % 8) * 8)) & 0xFF) as u8;
            output.push(byte ^ key_byte ^ nonce_byte);
        }

        output
    }

    /// "Decrypt" data (placeholder: reverse XOR).
    pub fn decrypt(&self, data: &[u8]) -> EngineResult<Vec<u8>> {
        if !self.enabled {
            return Ok(data.to_vec());
        }

        if data.len() < 8 {
            return Err(EngineError::InvalidArgument(
                "encrypted data too short".into(),
            ));
        }

        let nonce = u64::from_be_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);

        let mut output = Vec::with_capacity(data.len() - 8);
        for (i, &byte) in data[8..].iter().enumerate() {
            let key_byte = self.key[i % 32];
            let nonce_byte = ((nonce >> ((i % 8) * 8)) & 0xFF) as u8;
            output.push(byte ^ key_byte ^ nonce_byte);
        }

        Ok(output)
    }
}

impl Default for Encryption {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PacketStats
// ---------------------------------------------------------------------------

/// Network statistics.
#[derive(Debug, Clone, Default)]
pub struct PacketStats {
    /// Total bytes sent.
    pub bytes_sent: u64,
    /// Total bytes received.
    pub bytes_received: u64,
    /// Total packets sent.
    pub packets_sent: u64,
    /// Total packets received.
    pub packets_received: u64,
    /// Estimated packet loss ratio (0.0 - 1.0).
    pub packet_loss: f32,
    /// Round-trip time in milliseconds.
    pub rtt_ms: f32,
    /// RTT jitter in milliseconds.
    pub jitter_ms: f32,
    /// Bandwidth usage (bytes per second, outgoing).
    pub bandwidth_out_bps: u64,
    /// Bandwidth usage (bytes per second, incoming).
    pub bandwidth_in_bps: u64,
    /// Fragments sent.
    pub fragments_sent: u64,
    /// Fragments received.
    pub fragments_received: u64,
    /// Messages compressed.
    pub messages_compressed: u64,
    /// Bytes saved by compression.
    pub compression_savings: u64,
}

impl PacketStats {
    /// Reset all counters.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Record bytes sent.
    pub fn record_sent(&mut self, bytes: u64) {
        self.bytes_sent += bytes;
        self.packets_sent += 1;
    }

    /// Record bytes received.
    pub fn record_received(&mut self, bytes: u64) {
        self.bytes_received += bytes;
        self.packets_received += 1;
    }

    /// Update RTT estimate using EWMA.
    pub fn update_rtt(&mut self, sample_ms: f32) {
        let diff = (sample_ms - self.rtt_ms).abs();
        self.jitter_ms = 0.875 * self.jitter_ms + 0.125 * diff;
        self.rtt_ms = 0.875 * self.rtt_ms + 0.125 * sample_ms;
    }

    /// Returns the total data transferred (sent + received).
    pub fn total_bytes(&self) -> u64 {
        self.bytes_sent + self.bytes_received
    }

    /// Returns the total packets (sent + received).
    pub fn total_packets(&self) -> u64 {
        self.packets_sent + self.packets_received
    }
}

// ---------------------------------------------------------------------------
// Protocol
// ---------------------------------------------------------------------------

/// The main protocol manager tying together messages, channels, fragmentation,
/// compression, and encryption.
pub struct Protocol {
    /// Message type registry.
    pub registry: MessageRegistry,
    /// Channel configuration.
    pub channels: ChannelConfig,
    /// Message fragmenter.
    pub fragmenter: Fragmenter,
    /// Fragment reassembly buffer.
    pub reassembly: ReassemblyBuffer,
    /// Compression helper.
    pub compressor_enabled: bool,
    /// Encryption wrapper.
    pub encryption: Encryption,
    /// Network statistics.
    pub stats: PacketStats,
}

impl Protocol {
    /// Create a new protocol instance with default configuration.
    pub fn new() -> Self {
        Self {
            registry: MessageRegistry::new(),
            channels: ChannelConfig::default_config(),
            fragmenter: Fragmenter::new(),
            reassembly: ReassemblyBuffer::new(),
            compressor_enabled: true,
            encryption: Encryption::new(),
            stats: PacketStats::default(),
        }
    }

    /// Register a message type.
    pub fn register_message(
        &mut self,
        id: u16,
        name: impl Into<String>,
        channel: u8,
        deserializer: MessageDeserializer,
    ) -> EngineResult<()> {
        self.registry.register(id, name, channel, deserializer)
    }

    /// Prepare a message for sending: serialize, compress, encrypt, fragment.
    ///
    /// Returns a list of encoded fragments ready for the transport layer.
    pub fn prepare_send(&mut self, msg: &dyn NetMessage) -> Vec<Vec<u8>> {
        let msg_id = msg.message_id();
        let mut payload = msg.serialize();

        // Compress if beneficial.
        if self.compressor_enabled && Compressor::should_compress(&payload) {
            let compressed = Compressor::compress(&payload);
            if compressed.len() < payload.len() {
                let savings = (payload.len() - compressed.len()) as u64;
                self.stats.compression_savings += savings;
                self.stats.messages_compressed += 1;

                // Prepend a compression flag (1 byte: 1 = compressed, 0 = uncompressed).
                let mut flagged = Vec::with_capacity(1 + compressed.len());
                flagged.push(1);
                flagged.extend_from_slice(&compressed);
                payload = flagged;
            } else {
                let mut flagged = Vec::with_capacity(1 + payload.len());
                flagged.push(0);
                flagged.extend_from_slice(&payload);
                payload = flagged;
            }
        } else {
            // No compression: flag = 0.
            let mut flagged = Vec::with_capacity(1 + payload.len());
            flagged.push(0);
            flagged.extend_from_slice(&payload);
            payload = flagged;
        }

        // Encrypt.
        payload = self.encryption.encrypt(&payload);

        // Fragment.
        let fragments = self.fragmenter.fragment(msg_id, &payload);
        self.stats.fragments_sent += fragments.len() as u64;

        let mut encoded = Vec::with_capacity(fragments.len());
        for frag in fragments {
            let wire = frag.encode();
            self.stats.record_sent(wire.len() as u64);
            encoded.push(wire);
        }

        encoded
    }

    /// Process a received fragment, returning a completed message if reassembly
    /// is done.
    pub fn receive_fragment(
        &mut self,
        data: &[u8],
    ) -> EngineResult<Option<Box<dyn NetMessage>>> {
        let (fragment, _consumed) = Fragment::decode(data).ok_or_else(|| {
            EngineError::InvalidArgument("failed to decode fragment".into())
        })?;

        self.stats.record_received(data.len() as u64);
        self.stats.fragments_received += 1;

        let completed = self.reassembly.receive_fragment(fragment);

        if completed {
            // Get the completed message.
            let messages = self.reassembly.take_completed();
            if let Some((msg_id, mut payload)) = messages.into_iter().next() {
                // Decrypt.
                payload = self.encryption.decrypt(&payload)?;

                // Decompress.
                if !payload.is_empty() {
                    let flag = payload[0];
                    payload = if flag == 1 {
                        Compressor::decompress(&payload[1..])?
                    } else {
                        payload[1..].to_vec()
                    };
                }

                // Deserialize.
                if let Some(msg) = self.registry.deserialize(msg_id, &payload) {
                    return Ok(Some(msg));
                }
            }
        }

        Ok(None)
    }

    /// Perform periodic cleanup (reassembly timeouts, stat updates).
    pub fn update(&mut self) {
        self.reassembly.cleanup();
        self.channels.reset_all();
    }
}

impl Default for Protocol {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Test message type ---

    struct TestMessage {
        id: u16,
        data: Vec<u8>,
    }

    impl TestMessage {
        fn new(data: Vec<u8>) -> Self {
            Self { id: 1, data }
        }
    }

    impl NetMessage for TestMessage {
        fn message_id(&self) -> u16 {
            self.id
        }
        fn serialize(&self) -> Vec<u8> {
            self.data.clone()
        }
        fn message_name(&self) -> &str {
            "TestMessage"
        }
    }

    fn test_deserializer() -> MessageDeserializer {
        Box::new(|data: &[u8]| {
            Some(Box::new(TestMessage {
                id: 1,
                data: data.to_vec(),
            }))
        })
    }

    // -----------------------------------------------------------------------
    // MessageRegistry
    // -----------------------------------------------------------------------

    #[test]
    fn test_registry_register_and_lookup() {
        let mut reg = MessageRegistry::new();
        reg.register(1, "TestMsg", 0, test_deserializer()).unwrap();

        assert_eq!(reg.count(), 1);
        assert!(reg.is_registered(1));
        assert_eq!(reg.get_id("TestMsg"), Some(1));
        assert_eq!(reg.get_name(1), Some("TestMsg"));
        assert_eq!(reg.get_channel(1), Some(0));
    }

    #[test]
    fn test_registry_duplicate_id() {
        let mut reg = MessageRegistry::new();
        reg.register(1, "Msg1", 0, test_deserializer()).unwrap();
        let result = reg.register(1, "Msg2", 0, test_deserializer());
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_duplicate_name() {
        let mut reg = MessageRegistry::new();
        reg.register(1, "Msg", 0, test_deserializer()).unwrap();
        let result = reg.register(2, "Msg", 0, test_deserializer());
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_unregister() {
        let mut reg = MessageRegistry::new();
        reg.register(1, "Msg", 0, test_deserializer()).unwrap();
        assert!(reg.unregister(1));
        assert!(!reg.is_registered(1));
        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn test_registry_deserialize() {
        let mut reg = MessageRegistry::new();
        reg.register(1, "Msg", 0, test_deserializer()).unwrap();

        let msg = reg.deserialize(1, &[10, 20, 30]).unwrap();
        assert_eq!(msg.message_id(), 1);
        assert_eq!(msg.serialize(), vec![10, 20, 30]);
    }

    #[test]
    fn test_registry_serialize() {
        let reg = MessageRegistry::new();
        let msg = TestMessage::new(vec![1, 2, 3]);
        let encoded = reg.serialize_message(&msg);
        assert_eq!(encoded[0..2], [0, 1]); // message_id = 1
        assert_eq!(encoded[2..], [1, 2, 3]);
    }

    // -----------------------------------------------------------------------
    // Channel
    // -----------------------------------------------------------------------

    #[test]
    fn test_channel_budget() {
        let mut ch = Channel::reliable_ordered(0, "test");
        ch.max_bandwidth_bps = 60_000; // 1000 bytes per frame at 60fps

        assert!(ch.has_budget(500));
        ch.consume(500);
        assert!(ch.has_budget(500));
        ch.consume(500);
        assert!(!ch.has_budget(1));

        ch.reset_frame();
        assert!(ch.has_budget(1000));
    }

    #[test]
    fn test_channel_sequence() {
        let mut ch = Channel::reliable_ordered(0, "test");
        assert_eq!(ch.next_sequence(), 0);
        assert_eq!(ch.next_sequence(), 1);
        assert_eq!(ch.next_sequence(), 2);
    }

    #[test]
    fn test_channel_config() {
        let config = ChannelConfig::default_config();
        assert_eq!(config.channels.len(), 3);
        assert!(config.get_channel(0).is_some());
        assert!(config.get_channel(1).is_some());
        assert!(config.get_channel(2).is_some());
        assert!(config.get_channel(99).is_none());
    }

    // -----------------------------------------------------------------------
    // Fragment
    // -----------------------------------------------------------------------

    #[test]
    fn test_fragment_roundtrip() {
        let frag = Fragment {
            message_id: 42,
            fragment_id: 7,
            fragment_index: 2,
            fragment_count: 5,
            data: vec![10, 20, 30, 40],
        };

        let encoded = frag.encode();
        let (decoded, consumed) = Fragment::decode(&encoded).unwrap();

        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.message_id, 42);
        assert_eq!(decoded.fragment_id, 7);
        assert_eq!(decoded.fragment_index, 2);
        assert_eq!(decoded.fragment_count, 5);
        assert_eq!(decoded.data, vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_fragment_decode_too_short() {
        assert!(Fragment::decode(&[0, 1, 2]).is_none());
    }

    // -----------------------------------------------------------------------
    // Fragmenter
    // -----------------------------------------------------------------------

    #[test]
    fn test_fragmenter_small_message() {
        let mut frag = Fragmenter::new();
        let data = vec![1, 2, 3, 4, 5];
        let fragments = frag.fragment(1, &data);

        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].fragment_count, 1);
        assert_eq!(fragments[0].data, data);
    }

    #[test]
    fn test_fragmenter_large_message() {
        let mut frag = Fragmenter::new();
        let data = vec![0xAB; MAX_FRAGMENT_PAYLOAD * 3 + 100];
        let fragments = frag.fragment(1, &data);

        assert_eq!(fragments.len(), 4);
        for (i, f) in fragments.iter().enumerate() {
            assert_eq!(f.fragment_index, i as u8);
            assert_eq!(f.fragment_count, 4);
        }

        // Verify total data.
        let total: usize = fragments.iter().map(|f| f.data.len()).sum();
        assert_eq!(total, data.len());
    }

    // -----------------------------------------------------------------------
    // ReassemblyBuffer
    // -----------------------------------------------------------------------

    #[test]
    fn test_reassembly_single_fragment() {
        let mut buf = ReassemblyBuffer::new();
        let frag = Fragment {
            message_id: 1,
            fragment_id: 0,
            fragment_index: 0,
            fragment_count: 1,
            data: vec![42],
        };

        assert!(buf.receive_fragment(frag));
        let completed = buf.take_completed();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0], (1, vec![42]));
    }

    #[test]
    fn test_reassembly_multi_fragment() {
        let mut buf = ReassemblyBuffer::new();

        let frag0 = Fragment {
            message_id: 1,
            fragment_id: 5,
            fragment_index: 0,
            fragment_count: 3,
            data: vec![10, 20],
        };
        let frag1 = Fragment {
            message_id: 1,
            fragment_id: 5,
            fragment_index: 1,
            fragment_count: 3,
            data: vec![30, 40],
        };
        let frag2 = Fragment {
            message_id: 1,
            fragment_id: 5,
            fragment_index: 2,
            fragment_count: 3,
            data: vec![50],
        };

        assert!(!buf.receive_fragment(frag0));
        assert!(!buf.receive_fragment(frag1));
        assert!(buf.receive_fragment(frag2));

        let completed = buf.take_completed();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0], (1, vec![10, 20, 30, 40, 50]));
    }

    #[test]
    fn test_reassembly_out_of_order() {
        let mut buf = ReassemblyBuffer::new();

        // Receive fragments out of order.
        let frag2 = Fragment {
            message_id: 1,
            fragment_id: 3,
            fragment_index: 2,
            fragment_count: 3,
            data: vec![50],
        };
        let frag0 = Fragment {
            message_id: 1,
            fragment_id: 3,
            fragment_index: 0,
            fragment_count: 3,
            data: vec![10, 20],
        };
        let frag1 = Fragment {
            message_id: 1,
            fragment_id: 3,
            fragment_index: 1,
            fragment_count: 3,
            data: vec![30, 40],
        };

        assert!(!buf.receive_fragment(frag2));
        assert!(!buf.receive_fragment(frag0));
        assert!(buf.receive_fragment(frag1));

        let completed = buf.take_completed();
        assert_eq!(completed.len(), 1);
        // Data should be in correct order regardless of fragment arrival order.
        assert_eq!(completed[0], (1, vec![10, 20, 30, 40, 50]));
    }

    // -----------------------------------------------------------------------
    // Compressor
    // -----------------------------------------------------------------------

    #[test]
    fn test_compressor_roundtrip() {
        let data = b"Hello, World! This is a test of the compression system.".to_vec();
        let compressed = Compressor::compress(&data);
        let decompressed = Compressor::decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_compressor_runs() {
        // Data with long runs should compress well.
        let mut data = Vec::new();
        data.extend_from_slice(&[0u8; 50]);
        data.extend_from_slice(&[0xFF; 30]);
        data.extend_from_slice(&[0x42; 10]);

        let compressed = Compressor::compress(&data);
        assert!(compressed.len() < data.len());

        let decompressed = Compressor::decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_compressor_empty() {
        let compressed = Compressor::compress(&[]);
        assert!(compressed.is_empty());
        let decompressed = Compressor::decompress(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_compressor_single_byte() {
        let data = vec![42];
        let compressed = Compressor::compress(&data);
        let decompressed = Compressor::decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_compressor_no_runs() {
        // Sequential bytes with no runs.
        let data: Vec<u8> = (0..100).map(|i| (i % 256) as u8).collect();
        let compressed = Compressor::compress(&data);
        let decompressed = Compressor::decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_compressor_should_compress() {
        assert!(!Compressor::should_compress(&[0; 100]));
        assert!(Compressor::should_compress(&[0; 256]));
    }

    // -----------------------------------------------------------------------
    // Encryption (placeholder)
    // -----------------------------------------------------------------------

    #[test]
    fn test_encryption_disabled() {
        let mut enc = Encryption::new();
        let data = vec![1, 2, 3, 4, 5];
        let encrypted = enc.encrypt(&data);
        assert_eq!(encrypted, data); // no-op when disabled

        let decrypted = enc.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_encryption_roundtrip() {
        let mut enc = Encryption::new();
        let key = [42u8; 32];
        enc.enable(key);

        let data = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let encrypted = enc.encrypt(&data);
        assert_ne!(encrypted, data); // should be different

        let decrypted = enc.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_encryption_different_nonces() {
        let mut enc = Encryption::new();
        enc.enable([1u8; 32]);

        let data = vec![42; 16];
        let enc1 = enc.encrypt(&data);
        let enc2 = enc.encrypt(&data);
        // Different nonces should produce different ciphertext.
        assert_ne!(enc1, enc2);

        // Both should decrypt correctly.
        assert_eq!(enc.decrypt(&enc1).unwrap(), data);
        assert_eq!(enc.decrypt(&enc2).unwrap(), data);
    }

    // -----------------------------------------------------------------------
    // PacketStats
    // -----------------------------------------------------------------------

    #[test]
    fn test_packet_stats() {
        let mut stats = PacketStats::default();
        stats.record_sent(100);
        stats.record_sent(200);
        stats.record_received(150);

        assert_eq!(stats.bytes_sent, 300);
        assert_eq!(stats.bytes_received, 150);
        assert_eq!(stats.packets_sent, 2);
        assert_eq!(stats.packets_received, 1);
        assert_eq!(stats.total_bytes(), 450);
        assert_eq!(stats.total_packets(), 3);
    }

    #[test]
    fn test_packet_stats_rtt() {
        let mut stats = PacketStats::default();
        // Feed many samples to let EWMA converge from 0.0 toward 100.0.
        for _ in 0..30 {
            stats.update_rtt(100.0);
        }
        // After many identical samples, RTT should have converged.
        assert!((stats.rtt_ms - 100.0).abs() < 5.0);
    }

    // -----------------------------------------------------------------------
    // Protocol integration
    // -----------------------------------------------------------------------

    #[test]
    fn test_protocol_send_receive() {
        let mut sender = Protocol::new();
        let mut receiver = Protocol::new();

        // Register the same message type on both sides.
        sender
            .register_message(1, "TestMsg", 0, test_deserializer())
            .unwrap();
        receiver
            .register_message(1, "TestMsg", 0, test_deserializer())
            .unwrap();

        // Send a small message.
        let msg = TestMessage::new(vec![10, 20, 30]);
        let fragments = sender.prepare_send(&msg);
        assert_eq!(fragments.len(), 1);

        // Receive.
        let received = receiver.receive_fragment(&fragments[0]).unwrap();
        assert!(received.is_some());
        let received_msg = received.unwrap();
        assert_eq!(received_msg.message_id(), 1);
        assert_eq!(received_msg.serialize(), vec![10, 20, 30]);
    }

    #[test]
    fn test_protocol_large_message() {
        let mut sender = Protocol::new();
        let mut receiver = Protocol::new();

        sender
            .register_message(1, "Big", 0, test_deserializer())
            .unwrap();
        receiver
            .register_message(1, "Big", 0, test_deserializer())
            .unwrap();

        // Send a message that requires fragmentation. Use random-ish data
        // that does not compress well to ensure it stays large after compression.
        let data: Vec<u8> = (0..MAX_FRAGMENT_PAYLOAD * 3)
            .map(|i| ((i * 7 + 13) % 256) as u8)
            .collect();
        let msg = TestMessage::new(data.clone());
        let fragments = sender.prepare_send(&msg);
        assert!(fragments.len() > 1);

        // Receive all fragments.
        let mut result = None;
        for frag in &fragments {
            if let Some(msg) = receiver.receive_fragment(frag).unwrap() {
                result = Some(msg);
            }
        }

        assert!(result.is_some());
        assert_eq!(result.unwrap().serialize(), data);
    }

    #[test]
    fn test_protocol_with_encryption() {
        let mut sender = Protocol::new();
        let mut receiver = Protocol::new();

        let key = [99u8; 32];
        sender.encryption.enable(key);
        receiver.encryption.enable(key);

        sender
            .register_message(1, "Enc", 0, test_deserializer())
            .unwrap();
        receiver
            .register_message(1, "Enc", 0, test_deserializer())
            .unwrap();

        let msg = TestMessage::new(vec![1, 2, 3, 4, 5]);
        let fragments = sender.prepare_send(&msg);

        let received = receiver.receive_fragment(&fragments[0]).unwrap();
        assert!(received.is_some());
        assert_eq!(received.unwrap().serialize(), vec![1, 2, 3, 4, 5]);
    }
}
