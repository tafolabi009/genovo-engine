//! Network transport layer.
//!
//! Provides an abstract [`Transport`] trait and concrete implementations for
//! UDP and WebSocket protocols. Includes a [`ReliabilityLayer`] that adds
//! reliable, ordered delivery on top of unreliable transports (UDP).
//!
//! ## Wire format
//!
//! Every datagram on the wire starts with a fixed-size header:
//!
//! ```text
//! Offset  Size  Field
//! ------  ----  -----
//!   0       1   PacketType (u8)
//!   1       2   sequence_number (u16 big-endian)
//!   3       2   ack (u16 big-endian)
//!   5       4   ack_bitfield (u32 big-endian)
//!   9       2   payload_len (u16 big-endian)
//!  11       N   payload bytes
//! ```

use std::collections::{HashMap, VecDeque};
use std::io;
use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

use genovo_core::{EngineError, EngineResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Size of the wire packet header in bytes.
pub const HEADER_SIZE: usize = 11;

/// Maximum datagram payload (MTU minus IP/UDP headers minus our header).
pub const MAX_PAYLOAD_SIZE: usize = 1200;

/// Maximum datagram size on the wire.
pub const MAX_DATAGRAM_SIZE: usize = HEADER_SIZE + MAX_PAYLOAD_SIZE;

/// Default heartbeat interval.
pub const DEFAULT_HEARTBEAT_INTERVAL: Duration = Duration::from_millis(500);

/// Default connection timeout (no data received for this long => timed out).
pub const DEFAULT_CONNECTION_TIMEOUT: Duration = Duration::from_secs(10);

/// Maximum retransmissions before we declare a reliable packet lost.
pub const MAX_RETRANSMIT_COUNT: u32 = 10;

/// Default retransmission timeout.
pub const DEFAULT_RTO: Duration = Duration::from_millis(200);

// ---------------------------------------------------------------------------
// PacketType
// ---------------------------------------------------------------------------

/// Discriminant for the type of network packet on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PacketType {
    /// Connection request from client to server.
    Connect = 0,
    /// Server accepts a connection request.
    ConnectAccept = 1,
    /// Graceful disconnect notification.
    Disconnect = 2,
    /// Application data payload.
    Data = 3,
    /// Keep-alive heartbeat (empty payload).
    Heartbeat = 4,
    /// Standalone acknowledgement (no app data, just acks).
    Ack = 5,
}

impl PacketType {
    /// Decode a byte into a `PacketType`.
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::Connect),
            1 => Some(Self::ConnectAccept),
            2 => Some(Self::Disconnect),
            3 => Some(Self::Data),
            4 => Some(Self::Heartbeat),
            5 => Some(Self::Ack),
            _ => None,
        }
    }

    /// Encode as a single byte.
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

// ---------------------------------------------------------------------------
// ConnectionState
// ---------------------------------------------------------------------------

/// The state of a network connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConnectionState {
    /// No connection established.
    Disconnected,
    /// A connection attempt is in progress.
    Connecting,
    /// The connection is established and active.
    Connected,
    /// We are in the process of disconnecting.
    Disconnecting,
    /// The connection was lost due to timeout.
    TimedOut,
}

// ---------------------------------------------------------------------------
// PacketHeader
// ---------------------------------------------------------------------------

/// Header prepended to every network packet for sequencing and acknowledgment.
///
/// Uses a compact acknowledgment scheme: the `ack` field identifies the most
/// recent packet received from the remote peer, and `ack_bitfield` encodes
/// receipt of the 32 packets preceding `ack`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketHeader {
    /// The type of this packet.
    pub packet_type: PacketType,
    /// Sequence number of this packet (monotonically increasing per sender).
    pub sequence: u16,
    /// The most recent remote sequence number we have received.
    pub ack: u16,
    /// Bitmask acknowledging the 32 packets before `ack`.
    /// Bit 0 = ack-1, bit 1 = ack-2, ..., bit 31 = ack-32.
    pub ack_bitfield: u32,
    /// Length of the payload that follows.
    pub payload_len: u16,
}

impl PacketHeader {
    /// Creates a new packet header.
    pub fn new(
        packet_type: PacketType,
        sequence: u16,
        ack: u16,
        ack_bitfield: u32,
        payload_len: u16,
    ) -> Self {
        Self {
            packet_type,
            sequence,
            ack,
            ack_bitfield,
            payload_len,
        }
    }

    /// Returns `true` if the given sequence number is acknowledged by this header.
    pub fn is_acked(&self, sequence: u16) -> bool {
        if sequence == self.ack {
            return true;
        }
        let diff = self.ack.wrapping_sub(sequence);
        if diff > 0 && diff <= 32 {
            (self.ack_bitfield & (1 << (diff - 1))) != 0
        } else {
            false
        }
    }

    /// Encode the header into the first [`HEADER_SIZE`] bytes of `buf`.
    /// Returns the number of bytes written (always `HEADER_SIZE`).
    pub fn encode(&self, buf: &mut [u8]) -> usize {
        assert!(
            buf.len() >= HEADER_SIZE,
            "buffer too small for packet header"
        );
        buf[0] = self.packet_type.as_u8();
        buf[1..3].copy_from_slice(&self.sequence.to_be_bytes());
        buf[3..5].copy_from_slice(&self.ack.to_be_bytes());
        buf[5..9].copy_from_slice(&self.ack_bitfield.to_be_bytes());
        buf[9..11].copy_from_slice(&self.payload_len.to_be_bytes());
        HEADER_SIZE
    }

    /// Decode a header from the first [`HEADER_SIZE`] bytes of `buf`.
    pub fn decode(buf: &[u8]) -> EngineResult<Self> {
        if buf.len() < HEADER_SIZE {
            return Err(EngineError::InvalidArgument(format!(
                "packet too short: {} bytes, need at least {}",
                buf.len(),
                HEADER_SIZE
            )));
        }
        let packet_type = PacketType::from_u8(buf[0]).ok_or_else(|| {
            EngineError::InvalidArgument(format!("unknown packet type: {}", buf[0]))
        })?;
        let sequence = u16::from_be_bytes([buf[1], buf[2]]);
        let ack = u16::from_be_bytes([buf[3], buf[4]]);
        let ack_bitfield = u32::from_be_bytes([buf[5], buf[6], buf[7], buf[8]]);
        let payload_len = u16::from_be_bytes([buf[9], buf[10]]);
        Ok(Self {
            packet_type,
            sequence,
            ack,
            ack_bitfield,
            payload_len,
        })
    }
}

// ---------------------------------------------------------------------------
// DeliveryMode
// ---------------------------------------------------------------------------

/// Delivery guarantee for a packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryMode {
    /// No delivery guarantee; may be lost, duplicated, or reordered.
    Unreliable,
    /// Guaranteed delivery, but may arrive out of order.
    Reliable,
    /// Guaranteed delivery and in-order processing.
    ReliableOrdered,
}

// ---------------------------------------------------------------------------
// Packet
// ---------------------------------------------------------------------------

/// A network packet with header, payload, and delivery metadata.
#[derive(Debug, Clone)]
pub struct Packet {
    /// The packet header (sequence / ack info).
    pub header: PacketHeader,
    /// The payload data.
    pub payload: Vec<u8>,
    /// Delivery guarantee requested for this packet.
    pub delivery: DeliveryMode,
    /// Optional channel ID for multiplexing.
    pub channel: u8,
}

impl Packet {
    /// Creates a new packet with the given payload and delivery mode.
    pub fn new(payload: Vec<u8>, delivery: DeliveryMode) -> Self {
        Self {
            header: PacketHeader::new(PacketType::Data, 0, 0, 0, payload.len() as u16),
            payload,
            delivery,
            channel: 0,
        }
    }

    /// Creates a control packet (Connect, Heartbeat, etc.) with no payload.
    pub fn control(packet_type: PacketType) -> Self {
        Self {
            header: PacketHeader::new(packet_type, 0, 0, 0, 0),
            payload: Vec::new(),
            delivery: DeliveryMode::Reliable,
            channel: 0,
        }
    }

    /// Returns the total size of the packet on the wire (header + payload).
    pub fn wire_size(&self) -> usize {
        HEADER_SIZE + self.payload.len()
    }

    /// Encode this packet into a byte buffer suitable for sending on the wire.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = vec![0u8; self.wire_size()];
        self.header.encode(&mut buf);
        if !self.payload.is_empty() {
            buf[HEADER_SIZE..].copy_from_slice(&self.payload);
        }
        buf
    }

    /// Decode a packet from a raw byte buffer received from the wire.
    pub fn decode(buf: &[u8]) -> EngineResult<Self> {
        let header = PacketHeader::decode(buf)?;
        let payload_len = header.payload_len as usize;
        let total = HEADER_SIZE + payload_len;
        if buf.len() < total {
            return Err(EngineError::InvalidArgument(format!(
                "packet truncated: header says {} payload bytes, but only {} available",
                payload_len,
                buf.len().saturating_sub(HEADER_SIZE)
            )));
        }
        let payload = buf[HEADER_SIZE..total].to_vec();
        Ok(Self {
            header,
            payload,
            delivery: DeliveryMode::Unreliable, // delivery mode is a local concern
            channel: 0,
        })
    }
}

// ---------------------------------------------------------------------------
// Transport trait
// ---------------------------------------------------------------------------

/// Abstract network transport layer.
///
/// Implementations handle the low-level details of establishing connections,
/// sending/receiving raw data, and polling for events.
pub trait Transport: Send + Sync {
    /// Initiates a connection to the given remote address.
    fn connect(&mut self, addr: SocketAddr) -> EngineResult<()>;

    /// Disconnects from the remote peer (if connected).
    fn disconnect(&mut self) -> EngineResult<()>;

    /// Sends a packet to the connected peer.
    fn send(&mut self, packet: Packet) -> EngineResult<()>;

    /// Receives the next available packet, if any.
    fn recv(&mut self) -> EngineResult<Option<Packet>>;

    /// Polls the transport for state changes and processes internal timers.
    ///
    /// Should be called once per frame. Returns `true` if there is data to read.
    fn poll(&mut self, timeout: Duration) -> EngineResult<bool>;

    /// Returns the current connection state.
    fn connection_state(&self) -> ConnectionState;

    /// Returns the remote address, if connected.
    fn remote_addr(&self) -> Option<SocketAddr>;

    /// Returns an estimate of the round-trip time (RTT) in milliseconds.
    fn rtt_ms(&self) -> f32;

    /// Returns the estimated packet loss ratio (0.0 = no loss, 1.0 = total loss).
    fn packet_loss(&self) -> f32;
}

// ---------------------------------------------------------------------------
// SentPacketRecord
// ---------------------------------------------------------------------------

/// Sentinel entry tracking a sent packet awaiting acknowledgment.
#[derive(Debug, Clone)]
struct SentPacketRecord {
    /// The sequence number of the sent packet.
    sequence: u16,
    /// When this packet was sent (for RTT calculation).
    send_time: Instant,
    /// The full encoded packet (for retransmission).
    wire_data: Vec<u8>,
    /// Whether this packet has been acknowledged.
    acked: bool,
    /// Number of times this packet has been (re)transmitted.
    send_count: u32,
    /// Whether this packet requires reliable delivery.
    reliable: bool,
    /// The packet type (so we know what it is).
    packet_type: PacketType,
}

// ---------------------------------------------------------------------------
// ReliabilityLayer
// ---------------------------------------------------------------------------

/// Adds reliable delivery and acknowledgment tracking on top of an unreliable
/// transport (UDP).
///
/// Wraps sequence numbering, ack processing, RTT estimation, retransmission,
/// and packet loss tracking.
pub struct ReliabilityLayer {
    /// Next sequence number to assign to outgoing packets.
    local_sequence: u16,
    /// The latest remote sequence number we have received.
    remote_sequence: u16,
    /// Whether we have received any remote packet yet.
    remote_sequence_initialized: bool,
    /// Bitfield tracking receipt of the 32 packets before `remote_sequence`.
    ack_bitfield: u32,
    /// Sent packets awaiting acknowledgment.
    sent_packets: VecDeque<SentPacketRecord>,
    /// Maximum number of sent packets to track.
    max_tracked: usize,
    /// Retransmission timeout (adaptive).
    rto: Duration,
    /// Smoothed RTT estimate.
    smoothed_rtt: f64,
    /// RTT jitter (mean deviation).
    jitter: f64,
    /// Running count of sent packets (for loss ratio).
    total_sent: u64,
    /// Running count of acknowledged packets.
    total_acked: u64,
    /// Running count of packets declared lost.
    total_lost: u64,
    /// Received sequence numbers (for duplicate detection). Tracks last 256.
    received_sequences: VecDeque<u16>,
    /// Packets received out of order, waiting for in-order delivery.
    ordered_recv_buffer: VecDeque<(u16, Vec<u8>)>,
    /// Next expected sequence for ordered delivery.
    next_ordered_sequence: u16,
}

impl ReliabilityLayer {
    /// Creates a new reliability layer.
    pub fn new() -> Self {
        Self {
            local_sequence: 0,
            remote_sequence: 0,
            remote_sequence_initialized: false,
            ack_bitfield: 0,
            sent_packets: VecDeque::with_capacity(256),
            max_tracked: 256,
            rto: DEFAULT_RTO,
            smoothed_rtt: 100.0,
            jitter: 50.0,
            total_sent: 0,
            total_acked: 0,
            total_lost: 0,
            received_sequences: VecDeque::with_capacity(256),
            ordered_recv_buffer: VecDeque::new(),
            next_ordered_sequence: 0,
        }
    }

    /// Returns the current local sequence number (next to be assigned).
    pub fn local_sequence(&self) -> u16 {
        self.local_sequence
    }

    /// Returns the latest remote sequence number received.
    pub fn remote_sequence(&self) -> u16 {
        self.remote_sequence
    }

    /// Allocates a sequence number and builds the header for an outgoing packet.
    /// Records the packet for potential retransmission if `reliable` is true.
    pub fn prepare_send(
        &mut self,
        packet_type: PacketType,
        payload: &[u8],
        reliable: bool,
    ) -> Packet {
        let seq = self.local_sequence;
        self.local_sequence = self.local_sequence.wrapping_add(1);

        let header = PacketHeader::new(
            packet_type,
            seq,
            self.remote_sequence,
            self.ack_bitfield,
            payload.len() as u16,
        );

        let packet = Packet {
            header,
            payload: payload.to_vec(),
            delivery: if reliable {
                DeliveryMode::Reliable
            } else {
                DeliveryMode::Unreliable
            },
            channel: 0,
        };

        let wire_data = packet.encode();

        self.sent_packets.push_back(SentPacketRecord {
            sequence: seq,
            send_time: Instant::now(),
            wire_data,
            acked: false,
            send_count: 1,
            reliable,
            packet_type,
        });

        // Evict old records.
        while self.sent_packets.len() > self.max_tracked {
            if let Some(old) = self.sent_packets.pop_front() {
                if !old.acked && old.reliable {
                    self.total_lost += 1;
                }
            }
        }

        self.total_sent += 1;
        packet
    }

    /// Processes the header of an incoming packet, updating ack state and RTT.
    /// Returns `true` if this is a new (non-duplicate) packet.
    pub fn process_recv(&mut self, header: &PacketHeader) -> bool {
        let seq = header.sequence;

        // Duplicate detection.
        if self.received_sequences.contains(&seq) {
            return false;
        }
        self.received_sequences.push_back(seq);
        if self.received_sequences.len() > 256 {
            self.received_sequences.pop_front();
        }

        // Update remote sequence tracking.
        if !self.remote_sequence_initialized {
            self.remote_sequence = seq;
            self.remote_sequence_initialized = true;
            self.ack_bitfield = 0;
        } else if Self::sequence_greater_than(seq, self.remote_sequence) {
            let diff = seq.wrapping_sub(self.remote_sequence) as u32;
            if diff <= 32 {
                // Shift ack_bitfield left and set bit for the old remote_sequence.
                self.ack_bitfield = (self.ack_bitfield << diff) | (1 << (diff - 1));
            } else {
                self.ack_bitfield = 0;
            }
            self.remote_sequence = seq;
        } else {
            // Older packet; set the appropriate bit.
            let diff = self.remote_sequence.wrapping_sub(seq) as u32;
            if diff > 0 && diff <= 32 {
                self.ack_bitfield |= 1 << (diff - 1);
            }
        }

        // Process acks from the remote peer (they ack our sent packets).
        // Collect RTT samples first, then apply them, to avoid double-borrowing self.
        let now = Instant::now();
        let mut rtt_samples = Vec::new();
        for record in self.sent_packets.iter_mut() {
            if !record.acked && header.is_acked(record.sequence) {
                record.acked = true;
                let sample_ms =
                    now.duration_since(record.send_time).as_secs_f64() * 1000.0;
                rtt_samples.push(sample_ms);
            }
        }
        self.total_acked += rtt_samples.len() as u64;
        for sample in rtt_samples {
            self.update_rtt(sample);
        }

        true
    }

    /// Returns packets that need retransmission (unacked, reliable, and past RTO).
    pub fn get_retransmissions(&mut self) -> Vec<Vec<u8>> {
        let now = Instant::now();
        let mut retransmits = Vec::new();

        for record in self.sent_packets.iter_mut() {
            if !record.acked
                && record.reliable
                && record.send_count <= MAX_RETRANSMIT_COUNT
                && now.duration_since(record.send_time) > self.rto
            {
                // Update the ack fields in the wire data before retransmitting.
                let mut wire = record.wire_data.clone();
                // Patch ack and ack_bitfield in the wire data.
                wire[3..5].copy_from_slice(&self.remote_sequence.to_be_bytes());
                wire[5..9].copy_from_slice(&self.ack_bitfield.to_be_bytes());

                retransmits.push(wire);
                record.send_time = now;
                record.send_count += 1;
            }
        }

        retransmits
    }

    /// Checks if any reliable packet has exceeded the maximum retransmission count.
    /// Returns `true` if a packet has been permanently lost.
    pub fn has_permanent_loss(&self) -> bool {
        self.sent_packets
            .iter()
            .any(|r| !r.acked && r.reliable && r.send_count > MAX_RETRANSMIT_COUNT)
    }

    /// Removes acknowledged and permanently-lost packets from the tracking buffer.
    pub fn cleanup(&mut self) {
        while self
            .sent_packets
            .front()
            .is_some_and(|r| r.acked || r.send_count > MAX_RETRANSMIT_COUNT)
        {
            let record = self.sent_packets.pop_front().unwrap();
            if !record.acked && record.reliable {
                self.total_lost += 1;
            }
        }
    }

    /// Returns the current smoothed RTT estimate in milliseconds.
    pub fn smoothed_rtt_ms(&self) -> f64 {
        self.smoothed_rtt
    }

    /// Returns the current jitter estimate in milliseconds.
    pub fn jitter_ms(&self) -> f64 {
        self.jitter
    }

    /// Returns the current retransmission timeout.
    pub fn rto(&self) -> Duration {
        self.rto
    }

    /// Returns the estimated packet loss ratio.
    pub fn packet_loss_ratio(&self) -> f32 {
        if self.total_sent == 0 {
            0.0
        } else {
            self.total_lost as f32 / self.total_sent as f32
        }
    }

    /// Returns the number of unacknowledged packets currently tracked.
    pub fn unacked_count(&self) -> usize {
        self.sent_packets.iter().filter(|r| !r.acked).count()
    }

    /// Updates RTT estimate using EWMA.
    /// `rtt = 0.875 * rtt + 0.125 * sample`
    /// `jitter = 0.875 * jitter + 0.125 * |sample - rtt|`
    fn update_rtt(&mut self, sample_ms: f64) {
        let diff = (sample_ms - self.smoothed_rtt).abs();
        self.jitter = 0.875 * self.jitter + 0.125 * diff;
        self.smoothed_rtt = 0.875 * self.smoothed_rtt + 0.125 * sample_ms;

        // Adaptive RTO: SRTT + 4 * jitter, clamped.
        let rto_ms = self.smoothed_rtt + 4.0 * self.jitter;
        let rto_ms = rto_ms.clamp(50.0, 5000.0);
        self.rto = Duration::from_millis(rto_ms as u64);
    }

    /// Sequence number comparison accounting for 16-bit wrapping.
    fn sequence_greater_than(s1: u16, s2: u16) -> bool {
        let half = u16::MAX / 2;
        (s1 > s2 && s1 - s2 <= half) || (s1 < s2 && s2 - s1 > half)
    }
}

impl Default for ReliabilityLayer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------

/// A connection state machine managing a single peer-to-peer UDP link.
///
/// Handles the connection handshake (Connect -> ConnectAccept -> Connected),
/// heartbeat keep-alives, and timeout detection.
pub struct Connection {
    /// Remote address of the peer.
    remote_addr: SocketAddr,
    /// Current state.
    state: ConnectionState,
    /// When the last packet was received from the peer.
    last_recv_time: Instant,
    /// When the last packet was sent to the peer.
    last_send_time: Instant,
    /// When we entered the Connecting state (for handshake timeout).
    connect_start_time: Option<Instant>,
    /// Heartbeat interval.
    heartbeat_interval: Duration,
    /// Connection timeout (how long without receiving before we time out).
    connection_timeout: Duration,
    /// Number of heartbeats sent without receiving a response.
    missed_heartbeats: u32,
    /// Maximum missed heartbeats before timeout.
    max_missed_heartbeats: u32,
    /// Reliability layer for this connection.
    reliability: ReliabilityLayer,
}

impl Connection {
    /// Creates a new connection to the given remote address.
    pub fn new(remote_addr: SocketAddr) -> Self {
        let now = Instant::now();
        Self {
            remote_addr,
            state: ConnectionState::Disconnected,
            last_recv_time: now,
            last_send_time: now,
            connect_start_time: None,
            heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
            connection_timeout: DEFAULT_CONNECTION_TIMEOUT,
            missed_heartbeats: 0,
            max_missed_heartbeats: 6,
            reliability: ReliabilityLayer::new(),
        }
    }

    /// Returns the current connection state.
    pub fn state(&self) -> ConnectionState {
        self.state
    }

    /// Returns the remote address.
    pub fn remote_addr(&self) -> SocketAddr {
        self.remote_addr
    }

    /// Returns a reference to the reliability layer.
    pub fn reliability(&self) -> &ReliabilityLayer {
        &self.reliability
    }

    /// Returns a mutable reference to the reliability layer.
    pub fn reliability_mut(&mut self) -> &mut ReliabilityLayer {
        &mut self.reliability
    }

    /// Initiate a connection (move to Connecting state).
    pub fn initiate(&mut self) {
        self.state = ConnectionState::Connecting;
        self.connect_start_time = Some(Instant::now());
        self.last_recv_time = Instant::now();
        log::info!("Connection: initiating connection to {}", self.remote_addr);
    }

    /// Accept a connection (move directly to Connected state, server-side).
    pub fn accept(&mut self) {
        self.state = ConnectionState::Connected;
        self.connect_start_time = None;
        self.last_recv_time = Instant::now();
        self.missed_heartbeats = 0;
        log::info!("Connection: accepted connection from {}", self.remote_addr);
    }

    /// Begin graceful disconnection.
    pub fn begin_disconnect(&mut self) {
        self.state = ConnectionState::Disconnecting;
        log::info!(
            "Connection: disconnecting from {}",
            self.remote_addr
        );
    }

    /// Mark as fully disconnected.
    pub fn finish_disconnect(&mut self) {
        self.state = ConnectionState::Disconnected;
    }

    /// Handle the receipt of a ConnectAccept (client-side handshake completion).
    pub fn on_connect_accepted(&mut self) {
        if self.state == ConnectionState::Connecting {
            self.state = ConnectionState::Connected;
            self.connect_start_time = None;
            self.missed_heartbeats = 0;
            self.last_recv_time = Instant::now();
            log::info!(
                "Connection: established with {}",
                self.remote_addr
            );
        }
    }

    /// Record that we received data from the peer.
    pub fn on_data_received(&mut self) {
        self.last_recv_time = Instant::now();
        self.missed_heartbeats = 0;
    }

    /// Record that we sent data to the peer.
    pub fn on_data_sent(&mut self) {
        self.last_send_time = Instant::now();
    }

    /// Returns `true` if a heartbeat should be sent now.
    pub fn should_send_heartbeat(&self) -> bool {
        if self.state != ConnectionState::Connected {
            return false;
        }
        Instant::now().duration_since(self.last_send_time) >= self.heartbeat_interval
    }

    /// Update connection timers. Returns `true` if the connection is still alive.
    pub fn update(&mut self) -> bool {
        let now = Instant::now();
        let since_recv = now.duration_since(self.last_recv_time);

        match self.state {
            ConnectionState::Connecting => {
                // Check handshake timeout.
                if since_recv >= self.connection_timeout {
                    self.state = ConnectionState::TimedOut;
                    log::warn!(
                        "Connection: handshake timed out for {}",
                        self.remote_addr
                    );
                    return false;
                }
            }
            ConnectionState::Connected => {
                // Check for connection timeout.
                if since_recv >= self.connection_timeout {
                    self.state = ConnectionState::TimedOut;
                    log::warn!(
                        "Connection: timed out for {} (no data for {:?})",
                        self.remote_addr,
                        since_recv
                    );
                    return false;
                }
                // Track missed heartbeats.
                let expected_heartbeats =
                    (since_recv.as_millis() / self.heartbeat_interval.as_millis()) as u32;
                if expected_heartbeats > self.missed_heartbeats {
                    self.missed_heartbeats = expected_heartbeats;
                }
                if self.missed_heartbeats >= self.max_missed_heartbeats {
                    self.state = ConnectionState::TimedOut;
                    log::warn!(
                        "Connection: {} missed heartbeats, timing out {}",
                        self.missed_heartbeats,
                        self.remote_addr
                    );
                    return false;
                }
            }
            ConnectionState::Disconnecting => {
                // Wait briefly for disconnect ack, then force.
                if since_recv >= Duration::from_secs(2) {
                    self.state = ConnectionState::Disconnected;
                    return false;
                }
            }
            ConnectionState::Disconnected | ConnectionState::TimedOut => {
                return false;
            }
        }

        // Clean up the reliability layer.
        self.reliability.cleanup();

        true
    }

    /// Set the heartbeat interval.
    pub fn set_heartbeat_interval(&mut self, interval: Duration) {
        self.heartbeat_interval = interval;
    }

    /// Set the connection timeout.
    pub fn set_connection_timeout(&mut self, timeout: Duration) {
        self.connection_timeout = timeout;
    }

    /// Returns the time since the last received packet.
    pub fn time_since_last_recv(&self) -> Duration {
        Instant::now().duration_since(self.last_recv_time)
    }
}

// ---------------------------------------------------------------------------
// UdpTransport
// ---------------------------------------------------------------------------

/// UDP-based network transport.
///
/// Provides low-latency communication suitable for real-time game state.
/// Manages a real `std::net::UdpSocket` in non-blocking mode with the full
/// connection lifecycle and reliability layer.
pub struct UdpTransport {
    /// The raw UDP socket.
    socket: Option<UdpSocket>,
    /// Local bind address.
    local_addr: Option<SocketAddr>,
    /// Active connection (only one peer for client-mode).
    connection: Option<Connection>,
    /// Incoming packet queue (decoded packets ready for the application).
    recv_queue: VecDeque<Packet>,
    /// Receive buffer for raw datagrams.
    recv_buf: Vec<u8>,
    /// Whether we are acting as the listening/server side.
    is_server: bool,
    /// For server mode: multiple connections keyed by remote address.
    server_connections: HashMap<SocketAddr, Connection>,
}

impl UdpTransport {
    /// Creates a new UDP transport (not yet bound or connected).
    pub fn new() -> Self {
        Self {
            socket: None,
            local_addr: None,
            connection: None,
            recv_queue: VecDeque::new(),
            recv_buf: vec![0u8; MAX_DATAGRAM_SIZE],
            is_server: false,
            server_connections: HashMap::new(),
        }
    }

    /// Binds the transport to a local address for listening.
    ///
    /// Creates a real UDP socket bound to `addr` in non-blocking mode.
    pub fn bind(addr: SocketAddr) -> EngineResult<Self> {
        let socket = UdpSocket::bind(addr)?;
        socket.set_nonblocking(true)?;
        let local = socket.local_addr()?;

        log::info!("UdpTransport: bound to {}", local);

        Ok(Self {
            socket: Some(socket),
            local_addr: Some(local),
            connection: None,
            recv_queue: VecDeque::new(),
            recv_buf: vec![0u8; MAX_DATAGRAM_SIZE],
            is_server: false,
            server_connections: HashMap::new(),
        })
    }

    /// Enables server mode — the transport will accept incoming connections.
    pub fn set_server_mode(&mut self, server: bool) {
        self.is_server = server;
    }

    /// Returns the local address this transport is bound to.
    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.local_addr
    }

    /// Sets the socket to non-blocking mode.
    pub fn set_nonblocking(&self, nonblocking: bool) -> EngineResult<()> {
        if let Some(ref sock) = self.socket {
            sock.set_nonblocking(nonblocking)?;
        }
        Ok(())
    }

    /// Sends raw bytes to a specific remote address.
    fn send_raw(&self, data: &[u8], addr: SocketAddr) -> EngineResult<usize> {
        let sock = self.socket.as_ref().ok_or_else(|| {
            EngineError::InvalidState("socket not bound".into())
        })?;
        let n = sock.send_to(data, addr)?;
        Ok(n)
    }

    /// Receive a raw datagram from the socket. Returns None if WouldBlock.
    fn recv_raw(&mut self) -> EngineResult<Option<(usize, SocketAddr)>> {
        let sock = self.socket.as_ref().ok_or_else(|| {
            EngineError::InvalidState("socket not bound".into())
        })?;
        match sock.recv_from(&mut self.recv_buf) {
            Ok((n, addr)) => Ok(Some((n, addr))),
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(EngineError::Io(e)),
        }
    }

    /// Processes a received datagram in client mode.
    fn process_client_datagram(&mut self, n: usize, from: SocketAddr) {
        let buf = &self.recv_buf[..n];

        let packet = match Packet::decode(buf) {
            Ok(p) => p,
            Err(e) => {
                log::warn!("UdpTransport: failed to decode packet from {}: {}", from, e);
                return;
            }
        };

        let conn = match self.connection.as_mut() {
            Some(c) if c.remote_addr() == from => c,
            _ => {
                log::warn!("UdpTransport: ignoring packet from unknown peer {}", from);
                return;
            }
        };

        // Process reliability (acks, sequence tracking).
        let is_new = conn.reliability_mut().process_recv(&packet.header);
        if !is_new {
            return; // duplicate
        }

        conn.on_data_received();

        match packet.header.packet_type {
            PacketType::ConnectAccept => {
                conn.on_connect_accepted();
            }
            PacketType::Disconnect => {
                conn.finish_disconnect();
                log::info!("UdpTransport: peer {} disconnected", from);
            }
            PacketType::Data => {
                self.recv_queue.push_back(packet);
            }
            PacketType::Heartbeat | PacketType::Ack => {
                // Just the ack processing above is sufficient.
            }
            PacketType::Connect => {
                // Client shouldn't receive Connect; ignore.
            }
        }
    }

    /// Processes a received datagram in server mode.
    fn process_server_datagram(&mut self, n: usize, from: SocketAddr) {
        let buf = self.recv_buf[..n].to_vec(); // copy so we can borrow self mutably

        let packet = match Packet::decode(&buf) {
            Ok(p) => p,
            Err(e) => {
                log::warn!("UdpTransport: failed to decode packet from {}: {}", from, e);
                return;
            }
        };

        match packet.header.packet_type {
            PacketType::Connect => {
                // New connection request.
                if !self.server_connections.contains_key(&from) {
                    let mut conn = Connection::new(from);
                    conn.accept();
                    conn.reliability_mut().process_recv(&packet.header);
                    self.server_connections.insert(from, conn);
                    log::info!("UdpTransport: new client connected from {}", from);
                }

                // Send ConnectAccept.
                if let Some(conn) = self.server_connections.get_mut(&from) {
                    let pkt =
                        conn.reliability_mut()
                            .prepare_send(PacketType::ConnectAccept, &[], true);
                    let wire = pkt.encode();
                    conn.on_data_sent();
                    let _ = self.send_raw_inner(&wire, from);
                }
            }
            PacketType::Disconnect => {
                if let Some(mut conn) = self.server_connections.remove(&from) {
                    conn.finish_disconnect();
                    log::info!("UdpTransport: client {} disconnected", from);
                }
            }
            _ => {
                if let Some(conn) = self.server_connections.get_mut(&from) {
                    let is_new = conn.reliability_mut().process_recv(&packet.header);
                    if !is_new {
                        return;
                    }
                    conn.on_data_received();
                    if packet.header.packet_type == PacketType::Data {
                        // Tag the packet with source info via channel byte.
                        self.recv_queue.push_back(packet);
                    }
                }
            }
        }
    }

    /// Internal raw send that doesn't need &mut self for the socket.
    fn send_raw_inner(&self, data: &[u8], addr: SocketAddr) -> EngineResult<usize> {
        self.send_raw(data, addr)
    }

    /// Perform retransmissions for all connections.
    fn do_retransmissions(&mut self) {
        // Client connection.
        if let Some(ref mut conn) = self.connection {
            let retransmits = conn.reliability_mut().get_retransmissions();
            let addr = conn.remote_addr();
            for wire in retransmits {
                let _ = self.send_raw(&wire, addr);
            }
        }

        // Server connections.
        let addrs: Vec<SocketAddr> = self.server_connections.keys().copied().collect();
        for addr in addrs {
            if let Some(conn) = self.server_connections.get_mut(&addr) {
                let retransmits = conn.reliability_mut().get_retransmissions();
                for wire in retransmits {
                    let _ = self.send_raw_inner(&wire, addr);
                }
            }
        }
    }

    /// Send heartbeats for connections that need them.
    fn do_heartbeats(&mut self) {
        // Client connection.
        if let Some(ref mut conn) = self.connection {
            if conn.should_send_heartbeat() {
                let pkt =
                    conn.reliability_mut()
                        .prepare_send(PacketType::Heartbeat, &[], false);
                let wire = pkt.encode();
                let addr = conn.remote_addr();
                conn.on_data_sent();
                let _ = self.send_raw(&wire, addr);
            }
        }

        // Server connections.
        let addrs: Vec<SocketAddr> = self.server_connections.keys().copied().collect();
        for addr in addrs {
            if let Some(conn) = self.server_connections.get_mut(&addr) {
                if conn.should_send_heartbeat() {
                    let pkt = conn
                        .reliability_mut()
                        .prepare_send(PacketType::Heartbeat, &[], false);
                    let wire = pkt.encode();
                    conn.on_data_sent();
                    let _ = self.send_raw_inner(&wire, addr);
                }
            }
        }
    }

    /// Update all connection state machines.
    fn update_connections(&mut self) {
        // Client connection.
        if let Some(ref mut conn) = self.connection {
            conn.update();
        }

        // Server connections: remove timed-out connections.
        self.server_connections
            .retain(|addr, conn| {
                let alive = conn.update();
                if !alive {
                    log::info!("UdpTransport: removing dead connection to {}", addr);
                }
                alive
            });
    }

    /// Send a packet to a specific address (server-mode helper).
    pub fn send_to(&mut self, data: &[u8], addr: SocketAddr) -> EngineResult<()> {
        if let Some(conn) = self.server_connections.get_mut(&addr) {
            let pkt =
                conn.reliability_mut()
                    .prepare_send(PacketType::Data, data, true);
            let wire = pkt.encode();
            conn.on_data_sent();
            self.send_raw(&wire, addr)?;
            Ok(())
        } else {
            Err(EngineError::NotFound(format!(
                "no connection to {}",
                addr
            )))
        }
    }

    /// Broadcast data to all connected clients (server mode).
    pub fn broadcast(&mut self, data: &[u8]) -> EngineResult<()> {
        let addrs: Vec<SocketAddr> = self.server_connections.keys().copied().collect();
        for addr in addrs {
            self.send_to(data, addr)?;
        }
        Ok(())
    }

    /// Returns an iterator over connected client addresses (server mode).
    pub fn connected_clients(&self) -> impl Iterator<Item = &SocketAddr> {
        self.server_connections.keys()
    }

    /// Returns the number of connected clients.
    pub fn client_count(&self) -> usize {
        self.server_connections.len()
    }
}

impl Default for UdpTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl Transport for UdpTransport {
    fn connect(&mut self, addr: SocketAddr) -> EngineResult<()> {
        // If we don't have a socket yet, bind to an ephemeral port.
        if self.socket.is_none() {
            let bind_addr: SocketAddr = if addr.is_ipv6() {
                "[::]:0".parse().unwrap()
            } else {
                "0.0.0.0:0".parse().unwrap()
            };
            let socket = UdpSocket::bind(bind_addr)?;
            socket.set_nonblocking(true)?;
            self.local_addr = Some(socket.local_addr()?);
            self.socket = Some(socket);
        }

        let mut conn = Connection::new(addr);
        conn.initiate();

        // Send the Connect packet.
        let pkt =
            conn.reliability_mut()
                .prepare_send(PacketType::Connect, &[], true);
        let wire = pkt.encode();
        conn.on_data_sent();
        self.send_raw(&wire, addr)?;

        self.connection = Some(conn);
        log::info!("UdpTransport: connecting to {}", addr);

        Ok(())
    }

    fn disconnect(&mut self) -> EngineResult<()> {
        // Build the disconnect packet and extract the remote address before
        // calling send_raw, to avoid borrow conflicts with self.
        let wire_and_addr = if let Some(ref mut conn) = self.connection {
            let addr = conn.remote_addr();
            conn.begin_disconnect();

            let pkt =
                conn.reliability_mut()
                    .prepare_send(PacketType::Disconnect, &[], true);
            let wire = pkt.encode();
            conn.on_data_sent();
            conn.finish_disconnect();
            Some((wire, addr))
        } else {
            None
        };

        if let Some((wire, addr)) = wire_and_addr {
            let _ = self.send_raw(&wire, addr);
        }

        self.connection = None;
        log::info!("UdpTransport: disconnected");
        Ok(())
    }

    fn send(&mut self, packet: Packet) -> EngineResult<()> {
        let conn = self.connection.as_mut().ok_or_else(|| {
            EngineError::InvalidState("not connected".into())
        })?;

        if conn.state() != ConnectionState::Connected {
            return Err(EngineError::InvalidState(
                "cannot send: not in Connected state".into(),
            ));
        }

        let reliable = packet.delivery != DeliveryMode::Unreliable;
        let addr = conn.remote_addr();

        let pkt = conn
            .reliability_mut()
            .prepare_send(PacketType::Data, &packet.payload, reliable);
        let wire = pkt.encode();
        conn.on_data_sent();
        self.send_raw(&wire, addr)?;
        Ok(())
    }

    fn recv(&mut self) -> EngineResult<Option<Packet>> {
        Ok(self.recv_queue.pop_front())
    }

    fn poll(&mut self, _timeout: Duration) -> EngineResult<bool> {
        // Drain all pending datagrams from the socket.
        loop {
            match self.recv_raw()? {
                Some((n, from)) => {
                    if self.is_server {
                        self.process_server_datagram(n, from);
                    } else {
                        self.process_client_datagram(n, from);
                    }
                }
                None => break,
            }
        }

        // Do heartbeats.
        self.do_heartbeats();

        // Do retransmissions.
        self.do_retransmissions();

        // Update connection state machines.
        self.update_connections();

        Ok(!self.recv_queue.is_empty())
    }

    fn connection_state(&self) -> ConnectionState {
        self.connection
            .as_ref()
            .map(|c| c.state())
            .unwrap_or(ConnectionState::Disconnected)
    }

    fn remote_addr(&self) -> Option<SocketAddr> {
        self.connection.as_ref().map(|c| c.remote_addr())
    }

    fn rtt_ms(&self) -> f32 {
        self.connection
            .as_ref()
            .map(|c| c.reliability().smoothed_rtt_ms() as f32)
            .unwrap_or(0.0)
    }

    fn packet_loss(&self) -> f32 {
        self.connection
            .as_ref()
            .map(|c| c.reliability().packet_loss_ratio())
            .unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// WebSocketTransport (stub, as it requires an external crate)
// ---------------------------------------------------------------------------

/// WebSocket-based network transport.
///
/// Suitable for web builds (WASM targets) and scenarios where UDP is blocked.
/// Provides reliable, ordered delivery natively via TCP.
///
/// This is currently a placeholder -- a full implementation would use
/// `tungstenite` or a similar WebSocket crate.
pub struct WebSocketTransport {
    /// The connection state.
    state: ConnectionState,
    /// Remote peer address / URL.
    remote: Option<SocketAddr>,
    /// Estimated round-trip time (milliseconds).
    smoothed_rtt_ms: f32,
}

impl WebSocketTransport {
    /// Creates a new WebSocket transport.
    pub fn new() -> Self {
        Self {
            state: ConnectionState::Disconnected,
            remote: None,
            smoothed_rtt_ms: 0.0,
        }
    }
}

impl Default for WebSocketTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl Transport for WebSocketTransport {
    fn connect(&mut self, addr: SocketAddr) -> EngineResult<()> {
        self.state = ConnectionState::Connecting;
        self.remote = Some(addr);
        log::info!("WebSocketTransport: connecting to {addr} (stub)");
        Ok(())
    }

    fn disconnect(&mut self) -> EngineResult<()> {
        self.state = ConnectionState::Disconnected;
        self.remote = None;
        Ok(())
    }

    fn send(&mut self, _packet: Packet) -> EngineResult<()> {
        if self.state != ConnectionState::Connected {
            return Err(EngineError::InvalidState(
                "Cannot send: not connected".into(),
            ));
        }
        Ok(())
    }

    fn recv(&mut self) -> EngineResult<Option<Packet>> {
        Ok(None)
    }

    fn poll(&mut self, _timeout: Duration) -> EngineResult<bool> {
        Ok(false)
    }

    fn connection_state(&self) -> ConnectionState {
        self.state
    }

    fn remote_addr(&self) -> Option<SocketAddr> {
        self.remote
    }

    fn rtt_ms(&self) -> f32 {
        self.smoothed_rtt_ms
    }

    fn packet_loss(&self) -> f32 {
        0.0 // WebSocket uses TCP, so no packet loss at transport level
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // PacketHeader encoding / decoding
    // -----------------------------------------------------------------------

    #[test]
    fn test_packet_header_roundtrip() {
        let header = PacketHeader::new(PacketType::Data, 1234, 999, 0xDEADBEEF, 42);
        let mut buf = [0u8; HEADER_SIZE];
        header.encode(&mut buf);

        let decoded = PacketHeader::decode(&buf).unwrap();
        assert_eq!(decoded.packet_type, PacketType::Data);
        assert_eq!(decoded.sequence, 1234);
        assert_eq!(decoded.ack, 999);
        assert_eq!(decoded.ack_bitfield, 0xDEADBEEF);
        assert_eq!(decoded.payload_len, 42);
    }

    #[test]
    fn test_packet_header_all_types() {
        let types = [
            PacketType::Connect,
            PacketType::ConnectAccept,
            PacketType::Disconnect,
            PacketType::Data,
            PacketType::Heartbeat,
            PacketType::Ack,
        ];
        for pt in types {
            let header = PacketHeader::new(pt, 0, 0, 0, 0);
            let mut buf = [0u8; HEADER_SIZE];
            header.encode(&mut buf);
            let decoded = PacketHeader::decode(&buf).unwrap();
            assert_eq!(decoded.packet_type, pt);
        }
    }

    #[test]
    fn test_packet_header_decode_too_short() {
        let buf = [0u8; 5];
        assert!(PacketHeader::decode(&buf).is_err());
    }

    #[test]
    fn test_packet_header_decode_invalid_type() {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0] = 255; // invalid packet type
        assert!(PacketHeader::decode(&buf).is_err());
    }

    // -----------------------------------------------------------------------
    // PacketHeader ack checking
    // -----------------------------------------------------------------------

    #[test]
    fn test_header_is_acked_exact() {
        let header = PacketHeader::new(PacketType::Ack, 0, 50, 0, 0);
        assert!(header.is_acked(50));
        assert!(!header.is_acked(49));
        assert!(!header.is_acked(51));
    }

    #[test]
    fn test_header_is_acked_bitfield() {
        // ack=50, ack_bitfield=0b101 => acks packets 50, 49, 47
        let header = PacketHeader::new(PacketType::Ack, 0, 50, 0b101, 0);
        assert!(header.is_acked(50)); // exact ack
        assert!(header.is_acked(49)); // bit 0 set (ack - 1)
        assert!(!header.is_acked(48)); // bit 1 not set
        assert!(header.is_acked(47)); // bit 2 set (ack - 3)
        assert!(!header.is_acked(46));
    }

    #[test]
    fn test_header_is_acked_wrap() {
        // ack=2, ack_bitfield with bit 0 set => acks 2 and 1
        let header = PacketHeader::new(PacketType::Ack, 0, 2, 0b11, 0);
        assert!(header.is_acked(2));
        assert!(header.is_acked(1)); // bit 0 = ack - 1
        assert!(header.is_acked(0)); // bit 1 = ack - 2
    }

    // -----------------------------------------------------------------------
    // Packet encoding / decoding
    // -----------------------------------------------------------------------

    #[test]
    fn test_packet_roundtrip() {
        let payload = b"hello, network!".to_vec();
        let mut pkt = Packet::new(payload.clone(), DeliveryMode::Reliable);
        pkt.header = PacketHeader::new(
            PacketType::Data,
            42,
            10,
            0xFF,
            payload.len() as u16,
        );

        let encoded = pkt.encode();
        assert_eq!(encoded.len(), HEADER_SIZE + payload.len());

        let decoded = Packet::decode(&encoded).unwrap();
        assert_eq!(decoded.header.packet_type, PacketType::Data);
        assert_eq!(decoded.header.sequence, 42);
        assert_eq!(decoded.header.ack, 10);
        assert_eq!(decoded.header.ack_bitfield, 0xFF);
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn test_packet_empty_payload() {
        let pkt = Packet::control(PacketType::Heartbeat);
        let encoded = pkt.encode();
        assert_eq!(encoded.len(), HEADER_SIZE);

        let decoded = Packet::decode(&encoded).unwrap();
        assert_eq!(decoded.header.packet_type, PacketType::Heartbeat);
        assert!(decoded.payload.is_empty());
    }

    #[test]
    fn test_packet_decode_truncated() {
        // Header says 100 bytes of payload but buffer is too short.
        let header = PacketHeader::new(PacketType::Data, 0, 0, 0, 100);
        let mut buf = [0u8; HEADER_SIZE + 10]; // only 10 payload bytes
        header.encode(&mut buf);
        assert!(Packet::decode(&buf).is_err());
    }

    #[test]
    fn test_packet_max_payload() {
        let payload = vec![0xAB; MAX_PAYLOAD_SIZE];
        let mut pkt = Packet::new(payload.clone(), DeliveryMode::Unreliable);
        pkt.header.payload_len = payload.len() as u16;
        pkt.header.packet_type = PacketType::Data;

        let encoded = pkt.encode();
        let decoded = Packet::decode(&encoded).unwrap();
        assert_eq!(decoded.payload.len(), MAX_PAYLOAD_SIZE);
        assert_eq!(decoded.payload, payload);
    }

    // -----------------------------------------------------------------------
    // ReliabilityLayer
    // -----------------------------------------------------------------------

    #[test]
    fn test_reliability_sequence_allocation() {
        let mut rel = ReliabilityLayer::new();
        let p1 = rel.prepare_send(PacketType::Data, b"first", true);
        assert_eq!(p1.header.sequence, 0);

        let p2 = rel.prepare_send(PacketType::Data, b"second", true);
        assert_eq!(p2.header.sequence, 1);

        let p3 = rel.prepare_send(PacketType::Data, b"third", false);
        assert_eq!(p3.header.sequence, 2);

        assert_eq!(rel.local_sequence(), 3);
    }

    #[test]
    fn test_reliability_ack_processing() {
        let mut sender = ReliabilityLayer::new();
        let mut receiver = ReliabilityLayer::new();

        // Sender sends three packets.
        let p1 = sender.prepare_send(PacketType::Data, b"a", true);
        let p2 = sender.prepare_send(PacketType::Data, b"b", true);
        let p3 = sender.prepare_send(PacketType::Data, b"c", true);

        // Receiver gets all three.
        assert!(receiver.process_recv(&p1.header));
        assert!(receiver.process_recv(&p2.header));
        assert!(receiver.process_recv(&p3.header));

        // Receiver sends a response, which includes ack info.
        let response = receiver.prepare_send(PacketType::Data, b"reply", true);

        // The response should ack sequence 2 (latest from sender) and have
        // bits set for 1 and 0.
        assert_eq!(response.header.ack, 2);
        assert!(response.header.is_acked(2));
        assert!(response.header.is_acked(1));
        assert!(response.header.is_acked(0));

        // Sender processes the response, which acks all three sent packets.
        sender.process_recv(&response.header);

        // All three should now be acked.
        assert_eq!(sender.unacked_count(), 0);
    }

    #[test]
    fn test_reliability_duplicate_detection() {
        let mut rel = ReliabilityLayer::new();

        let header = PacketHeader::new(PacketType::Data, 5, 0, 0, 3);

        assert!(rel.process_recv(&header)); // first time: new
        assert!(!rel.process_recv(&header)); // second time: duplicate
    }

    #[test]
    fn test_reliability_rtt_update() {
        let mut rel = ReliabilityLayer::new();
        // Initial RTT is 100ms.
        assert!((rel.smoothed_rtt_ms() - 100.0).abs() < 1.0);

        // Simulate receiving an ack.
        let pkt = rel.prepare_send(PacketType::Data, b"test", true);
        // We can't easily control timing in unit tests, but we can verify the
        // method exists and doesn't panic.
        std::thread::sleep(Duration::from_millis(5));
        let ack_header = PacketHeader::new(PacketType::Ack, 0, pkt.header.sequence, 0, 0);
        rel.process_recv(&ack_header);

        // RTT should have moved towards the measured sample (which is very small).
        assert!(rel.smoothed_rtt_ms() < 100.0);
    }

    #[test]
    fn test_reliability_packet_loss_ratio() {
        let mut rel = ReliabilityLayer::new();
        assert_eq!(rel.packet_loss_ratio(), 0.0);

        // Send 10 packets.
        for _ in 0..10 {
            rel.prepare_send(PacketType::Data, b"x", true);
        }

        // Ack only half of them (0, 2, 4, 6, 8).
        // Each ack must have a unique sequence number to avoid duplicate detection.
        for (i, seq) in (0u16..10).step_by(2).enumerate() {
            let ack_header =
                PacketHeader::new(PacketType::Ack, i as u16, seq, 0, 0);
            rel.process_recv(&ack_header);
        }

        // The unacked packets (1, 3, 5, 7, 9) are still tracked.
        assert_eq!(rel.unacked_count(), 5);
    }

    #[test]
    fn test_reliability_sequence_wrapping() {
        let mut rel = ReliabilityLayer::new();
        // Jump near the wrapping point.
        rel.local_sequence = u16::MAX - 1;

        let p1 = rel.prepare_send(PacketType::Data, b"a", true);
        assert_eq!(p1.header.sequence, u16::MAX - 1);

        let p2 = rel.prepare_send(PacketType::Data, b"b", true);
        assert_eq!(p2.header.sequence, u16::MAX);

        let p3 = rel.prepare_send(PacketType::Data, b"c", true);
        assert_eq!(p3.header.sequence, 0); // wrapped!

        let p4 = rel.prepare_send(PacketType::Data, b"d", true);
        assert_eq!(p4.header.sequence, 1);
    }

    #[test]
    fn test_reliability_remote_sequence_ordering() {
        let mut rel = ReliabilityLayer::new();

        // Receive packets out of order: 0, 2, 1
        let h0 = PacketHeader::new(PacketType::Data, 0, 0, 0, 1);
        let h2 = PacketHeader::new(PacketType::Data, 2, 0, 0, 1);
        let h1 = PacketHeader::new(PacketType::Data, 1, 0, 0, 1);

        rel.process_recv(&h0);
        assert_eq!(rel.remote_sequence(), 0);

        rel.process_recv(&h2);
        assert_eq!(rel.remote_sequence(), 2);
        // Bit 1 should be set (for seq 0, which is ack-2)
        // Bit 0 should NOT be set (seq 1 not received yet)

        rel.process_recv(&h1);
        assert_eq!(rel.remote_sequence(), 2); // unchanged, 1 is older
    }

    #[test]
    fn test_reliability_cleanup() {
        let mut rel = ReliabilityLayer::new();

        // Send a packet and ack it.
        let pkt = rel.prepare_send(PacketType::Data, b"test", true);
        let ack_header =
            PacketHeader::new(PacketType::Ack, 0, pkt.header.sequence, 0, 0);
        rel.process_recv(&ack_header);

        assert_eq!(rel.unacked_count(), 0);
        rel.cleanup();

        // The sent_packets deque should be empty after cleanup.
        assert_eq!(rel.sent_packets.len(), 0);
    }

    // -----------------------------------------------------------------------
    // Connection state machine
    // -----------------------------------------------------------------------

    #[test]
    fn test_connection_lifecycle() {
        let addr: SocketAddr = "127.0.0.1:9999".parse().unwrap();
        let mut conn = Connection::new(addr);

        assert_eq!(conn.state(), ConnectionState::Disconnected);

        conn.initiate();
        assert_eq!(conn.state(), ConnectionState::Connecting);

        conn.on_connect_accepted();
        assert_eq!(conn.state(), ConnectionState::Connected);

        conn.begin_disconnect();
        assert_eq!(conn.state(), ConnectionState::Disconnecting);

        conn.finish_disconnect();
        assert_eq!(conn.state(), ConnectionState::Disconnected);
    }

    #[test]
    fn test_connection_server_accept() {
        let addr: SocketAddr = "127.0.0.1:9999".parse().unwrap();
        let mut conn = Connection::new(addr);

        conn.accept();
        assert_eq!(conn.state(), ConnectionState::Connected);
    }

    #[test]
    fn test_connection_heartbeat_check() {
        let addr: SocketAddr = "127.0.0.1:9999".parse().unwrap();
        let mut conn = Connection::new(addr);

        // Not connected -> no heartbeat needed.
        assert!(!conn.should_send_heartbeat());

        conn.accept();
        // Just accepted -> last_send_time is recent.
        assert!(!conn.should_send_heartbeat());
    }

    #[test]
    fn test_connection_update_alive() {
        let addr: SocketAddr = "127.0.0.1:9999".parse().unwrap();
        let mut conn = Connection::new(addr);
        conn.accept();
        conn.on_data_received(); // reset timer

        assert!(conn.update()); // should be alive
        assert_eq!(conn.state(), ConnectionState::Connected);
    }

    #[test]
    fn test_connection_connect_accepted_idempotent() {
        let addr: SocketAddr = "127.0.0.1:9999".parse().unwrap();
        let mut conn = Connection::new(addr);

        // Already connected, on_connect_accepted should be ignored.
        conn.accept();
        assert_eq!(conn.state(), ConnectionState::Connected);

        conn.on_connect_accepted(); // should not change state
        assert_eq!(conn.state(), ConnectionState::Connected);
    }

    // -----------------------------------------------------------------------
    // UdpTransport - real socket tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_udp_transport_bind() {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let transport = UdpTransport::bind(addr).unwrap();
        assert!(transport.local_addr().is_some());
    }

    #[test]
    fn test_udp_transport_client_server_roundtrip() {
        // Start a server.
        let server_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let mut server = UdpTransport::bind(server_addr).unwrap();
        server.set_server_mode(true);
        let server_addr = server.local_addr().unwrap();

        // Start a client and connect to the server.
        let mut client = UdpTransport::new();
        client.connect(server_addr).unwrap();

        // Poll the server to process the Connect packet.
        std::thread::sleep(Duration::from_millis(10));
        server.poll(Duration::from_millis(0)).unwrap();

        // Poll the client to process the ConnectAccept.
        std::thread::sleep(Duration::from_millis(10));
        client.poll(Duration::from_millis(0)).unwrap();

        // The client should now be connected.
        assert_eq!(client.connection_state(), ConnectionState::Connected);

        // Send data from client to server.
        let data = b"hello server!".to_vec();
        let pkt = Packet::new(data.clone(), DeliveryMode::Reliable);
        client.send(pkt).unwrap();

        // Poll the server.
        std::thread::sleep(Duration::from_millis(10));
        server.poll(Duration::from_millis(0)).unwrap();

        // Server should have received the data.
        let received = server.recv().unwrap();
        assert!(received.is_some());
        assert_eq!(received.unwrap().payload, data);

        // Send data from server to client.
        // The server tracks the client by the address the Connect datagram came
        // from, which is the actual source address (127.0.0.1:port), not the
        // client's local_addr (which may be 0.0.0.0:port).
        let reply = b"hello client!".to_vec();
        let client_addr_on_server =
            *server.connected_clients().next().unwrap();
        server.send_to(&reply, client_addr_on_server).unwrap();

        std::thread::sleep(Duration::from_millis(10));
        client.poll(Duration::from_millis(0)).unwrap();

        let client_recv = client.recv().unwrap();
        assert!(client_recv.is_some());
        assert_eq!(client_recv.unwrap().payload, reply);
    }

    #[test]
    fn test_udp_transport_disconnect() {
        let server_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let mut server = UdpTransport::bind(server_addr).unwrap();
        server.set_server_mode(true);
        let server_addr = server.local_addr().unwrap();

        let mut client = UdpTransport::new();
        client.connect(server_addr).unwrap();

        std::thread::sleep(Duration::from_millis(10));
        server.poll(Duration::from_millis(0)).unwrap();
        std::thread::sleep(Duration::from_millis(10));
        client.poll(Duration::from_millis(0)).unwrap();

        assert_eq!(client.connection_state(), ConnectionState::Connected);

        // Disconnect.
        client.disconnect().unwrap();
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);

        // Server processes the disconnect.
        std::thread::sleep(Duration::from_millis(10));
        server.poll(Duration::from_millis(0)).unwrap();
        assert_eq!(server.client_count(), 0);
    }

    #[test]
    fn test_udp_transport_send_without_connect() {
        let mut transport = UdpTransport::new();
        let pkt = Packet::new(b"test".to_vec(), DeliveryMode::Unreliable);
        assert!(transport.send(pkt).is_err());
    }

    #[test]
    fn test_udp_transport_recv_empty() {
        let transport = UdpTransport::new();
        let result = transport.recv_queue.front();
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // PacketType conversions
    // -----------------------------------------------------------------------

    #[test]
    fn test_packet_type_roundtrip() {
        for val in 0u8..=5 {
            let pt = PacketType::from_u8(val).unwrap();
            assert_eq!(pt.as_u8(), val);
        }
        assert!(PacketType::from_u8(6).is_none());
        assert!(PacketType::from_u8(255).is_none());
    }

    // -----------------------------------------------------------------------
    // Sequence comparison
    // -----------------------------------------------------------------------

    #[test]
    fn test_sequence_greater_than() {
        assert!(ReliabilityLayer::sequence_greater_than(1, 0));
        assert!(ReliabilityLayer::sequence_greater_than(100, 50));
        assert!(!ReliabilityLayer::sequence_greater_than(50, 100));

        // Wrapping: 0 is "greater" than 65535 (just wrapped).
        assert!(ReliabilityLayer::sequence_greater_than(0, u16::MAX));
        assert!(ReliabilityLayer::sequence_greater_than(1, u16::MAX));
        // But 65535 is NOT greater than 0 (it would require going backward).
        assert!(!ReliabilityLayer::sequence_greater_than(u16::MAX, 0));
    }

    #[test]
    fn test_sequence_greater_than_half_range() {
        let half = u16::MAX / 2;
        // Values within half the range are ordered normally.
        assert!(ReliabilityLayer::sequence_greater_than(half, 0));
        // Values more than half the range apart are considered to have wrapped.
        assert!(!ReliabilityLayer::sequence_greater_than(half + 2, 0));
    }

    // -----------------------------------------------------------------------
    // Delivery mode and wire size
    // -----------------------------------------------------------------------

    #[test]
    fn test_packet_wire_size() {
        let pkt = Packet::new(vec![0; 100], DeliveryMode::Unreliable);
        assert_eq!(pkt.wire_size(), HEADER_SIZE + 100);

        let pkt_empty = Packet::control(PacketType::Heartbeat);
        assert_eq!(pkt_empty.wire_size(), HEADER_SIZE);
    }

    // -----------------------------------------------------------------------
    // Multiple server connections
    // -----------------------------------------------------------------------

    #[test]
    fn test_udp_transport_multiple_clients() {
        let server_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let mut server = UdpTransport::bind(server_addr).unwrap();
        server.set_server_mode(true);
        let server_addr = server.local_addr().unwrap();

        // Connect two clients.
        let mut client1 = UdpTransport::new();
        let mut client2 = UdpTransport::new();
        client1.connect(server_addr).unwrap();
        client2.connect(server_addr).unwrap();

        std::thread::sleep(Duration::from_millis(20));
        server.poll(Duration::from_millis(0)).unwrap();

        assert_eq!(server.client_count(), 2);

        // Both clients get ConnectAccept.
        std::thread::sleep(Duration::from_millis(10));
        client1.poll(Duration::from_millis(0)).unwrap();
        client2.poll(Duration::from_millis(0)).unwrap();

        assert_eq!(client1.connection_state(), ConnectionState::Connected);
        assert_eq!(client2.connection_state(), ConnectionState::Connected);
    }

    // -----------------------------------------------------------------------
    // WebSocketTransport
    // -----------------------------------------------------------------------

    #[test]
    fn test_websocket_transport_default_state() {
        let ws = WebSocketTransport::new();
        assert_eq!(ws.connection_state(), ConnectionState::Disconnected);
        assert!(ws.remote_addr().is_none());
        assert_eq!(ws.rtt_ms(), 0.0);
        assert_eq!(ws.packet_loss(), 0.0);
    }
}
