//! WebSocket implementation for the Genovo networking module.
//!
//! Provides a WebSocket client following RFC 6455, built directly on TCP sockets.
//! Suitable for real-time communication between the game client and backend
//! services (matchmaking servers, leaderboards, live events).
//!
//! # Features
//!
//! - HTTP upgrade handshake with Sec-WebSocket-Key validation
//! - Frame parsing: opcodes (text, binary, ping, pong, close), masking, payload
//! - Fragmented message reassembly
//! - Ping/pong keepalive with configurable interval
//! - Close handshake (send close frame, wait for server close)
//! - Auto-reconnect with configurable exponential backoff
//! - Message queue for outbound messages during reconnect
//! - Connection state machine (Connecting, Open, Closing, Closed)
//!
//! # Example
//!
//! ```ignore
//! let mut ws = WebSocket::connect("ws://localhost:8080/game")?;
//! ws.send_text("hello server")?;
//! while let Some(msg) = ws.poll()? {
//!     match msg {
//!         WsMessage::Text(t) => println!("got: {t}"),
//!         WsMessage::Binary(b) => println!("got {} bytes", b.len()),
//!         WsMessage::Close(code, reason) => break,
//!         _ => {}
//!     }
//! }
//! ws.close(WsCloseCode::Normal, "bye")?;
//! ```

use std::collections::VecDeque;
use std::fmt;
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// WebSocket GUID used during the handshake (RFC 6455 Section 4.2.2).
const WS_GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/// Maximum frame payload size (16 MiB).
const MAX_FRAME_PAYLOAD: usize = 16 * 1024 * 1024;

/// Maximum message size after reassembly (64 MiB).
const MAX_MESSAGE_SIZE: usize = 64 * 1024 * 1024;

/// Default ping interval in seconds.
const DEFAULT_PING_INTERVAL_SECS: u64 = 30;

/// Default pong timeout in seconds.
const DEFAULT_PONG_TIMEOUT_SECS: u64 = 10;

/// Default connect timeout in seconds.
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;

/// Default initial reconnect delay in milliseconds.
const DEFAULT_RECONNECT_BASE_MS: u64 = 500;

/// Default maximum reconnect delay in milliseconds.
const DEFAULT_RECONNECT_MAX_MS: u64 = 30_000;

/// Default reconnect backoff multiplier.
const DEFAULT_RECONNECT_MULTIPLIER: f64 = 2.0;

/// Default maximum reconnect attempts (0 = infinite).
const DEFAULT_MAX_RECONNECT_ATTEMPTS: u32 = 10;

/// Read buffer size.
const READ_BUF_SIZE: usize = 8192;

/// Maximum outbound queue size during reconnect.
const MAX_QUEUE_SIZE: usize = 256;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during WebSocket operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WsError {
    /// The URL is invalid or uses an unsupported scheme.
    InvalidUrl(String),
    /// TCP connection failed.
    ConnectionFailed(String),
    /// Connection timed out.
    ConnectTimeout,
    /// The HTTP upgrade handshake failed.
    HandshakeFailed(String),
    /// The server rejected the upgrade request.
    UpgradeRejected(u16, String),
    /// The Sec-WebSocket-Accept value did not match.
    InvalidAcceptKey,
    /// An I/O error occurred.
    IoError(String),
    /// A received frame was malformed.
    InvalidFrame(String),
    /// A received frame payload exceeded the maximum size.
    FrameTooLarge,
    /// A reassembled message exceeded the maximum size.
    MessageTooLarge,
    /// The connection is not in the Open state.
    NotConnected,
    /// The connection was closed by the remote end.
    ConnectionClosed(u16, String),
    /// A protocol violation occurred.
    ProtocolError(String),
    /// Maximum reconnect attempts exceeded.
    ReconnectExhausted,
    /// The outbound message queue is full.
    QueueFull,
    /// Read timed out.
    ReadTimeout,
}

impl fmt::Display for WsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUrl(u) => write!(f, "invalid WebSocket URL: {u}"),
            Self::ConnectionFailed(msg) => write!(f, "connection failed: {msg}"),
            Self::ConnectTimeout => write!(f, "connection timed out"),
            Self::HandshakeFailed(msg) => write!(f, "handshake failed: {msg}"),
            Self::UpgradeRejected(code, reason) => {
                write!(f, "upgrade rejected: {code} {reason}")
            }
            Self::InvalidAcceptKey => write!(f, "invalid Sec-WebSocket-Accept"),
            Self::IoError(msg) => write!(f, "I/O error: {msg}"),
            Self::InvalidFrame(msg) => write!(f, "invalid frame: {msg}"),
            Self::FrameTooLarge => write!(f, "frame payload too large"),
            Self::MessageTooLarge => write!(f, "reassembled message too large"),
            Self::NotConnected => write!(f, "WebSocket not connected"),
            Self::ConnectionClosed(code, reason) => {
                write!(f, "connection closed: {code} {reason}")
            }
            Self::ProtocolError(msg) => write!(f, "protocol error: {msg}"),
            Self::ReconnectExhausted => write!(f, "max reconnect attempts exceeded"),
            Self::QueueFull => write!(f, "outbound message queue full"),
            Self::ReadTimeout => write!(f, "read timed out"),
        }
    }
}

/// Result type alias for WebSocket operations.
pub type WsResult<T> = Result<T, WsError>;

// ---------------------------------------------------------------------------
// WebSocket close codes (RFC 6455 Section 7.4.1)
// ---------------------------------------------------------------------------

/// Standard WebSocket close status codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WsCloseCode {
    /// 1000 - Normal closure.
    Normal,
    /// 1001 - Going away (e.g. server shutting down).
    GoingAway,
    /// 1002 - Protocol error.
    ProtocolError,
    /// 1003 - Unsupported data type.
    UnsupportedData,
    /// 1005 - No status code present (should not be sent on wire).
    NoStatus,
    /// 1006 - Abnormal closure (should not be sent on wire).
    Abnormal,
    /// 1007 - Invalid frame payload data (e.g. non-UTF-8 text).
    InvalidPayload,
    /// 1008 - Policy violation.
    PolicyViolation,
    /// 1009 - Message too big.
    MessageTooBig,
    /// 1010 - Missing extension.
    MissingExtension,
    /// 1011 - Internal server error.
    InternalError,
    /// 1012 - Service restart.
    ServiceRestart,
    /// 1013 - Try again later.
    TryAgainLater,
    /// Application-defined close code.
    Custom(u16),
}

impl WsCloseCode {
    /// Convert to the numeric code.
    pub fn as_u16(&self) -> u16 {
        match self {
            Self::Normal => 1000,
            Self::GoingAway => 1001,
            Self::ProtocolError => 1002,
            Self::UnsupportedData => 1003,
            Self::NoStatus => 1005,
            Self::Abnormal => 1006,
            Self::InvalidPayload => 1007,
            Self::PolicyViolation => 1008,
            Self::MessageTooBig => 1009,
            Self::MissingExtension => 1010,
            Self::InternalError => 1011,
            Self::ServiceRestart => 1012,
            Self::TryAgainLater => 1013,
            Self::Custom(code) => *code,
        }
    }

    /// Parse from a numeric code.
    pub fn from_u16(code: u16) -> Self {
        match code {
            1000 => Self::Normal,
            1001 => Self::GoingAway,
            1002 => Self::ProtocolError,
            1003 => Self::UnsupportedData,
            1005 => Self::NoStatus,
            1006 => Self::Abnormal,
            1007 => Self::InvalidPayload,
            1008 => Self::PolicyViolation,
            1009 => Self::MessageTooBig,
            1010 => Self::MissingExtension,
            1011 => Self::InternalError,
            1012 => Self::ServiceRestart,
            1013 => Self::TryAgainLater,
            other => Self::Custom(other),
        }
    }

    /// Returns `true` if this is a valid code to send in a close frame.
    pub fn is_sendable(&self) -> bool {
        let code = self.as_u16();
        // Reserved codes that should not be sent on the wire.
        code != 1005 && code != 1006 && (code >= 1000)
    }
}

impl fmt::Display for WsCloseCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_u16())
    }
}

// ---------------------------------------------------------------------------
// WebSocket opcodes
// ---------------------------------------------------------------------------

/// WebSocket frame opcodes (4-bit field in the frame header).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WsOpcode {
    /// Continuation frame (0x0).
    Continuation,
    /// Text frame (0x1).
    Text,
    /// Binary frame (0x2).
    Binary,
    /// Connection close (0x8).
    Close,
    /// Ping (0x9).
    Ping,
    /// Pong (0xA).
    Pong,
    /// Unknown/reserved opcode.
    Unknown(u8),
}

impl WsOpcode {
    /// Parse from a 4-bit value.
    pub fn from_u8(val: u8) -> Self {
        match val & 0x0F {
            0x0 => Self::Continuation,
            0x1 => Self::Text,
            0x2 => Self::Binary,
            0x8 => Self::Close,
            0x9 => Self::Ping,
            0xA => Self::Pong,
            other => Self::Unknown(other),
        }
    }

    /// Convert to the 4-bit value.
    pub fn as_u8(&self) -> u8 {
        match self {
            Self::Continuation => 0x0,
            Self::Text => 0x1,
            Self::Binary => 0x2,
            Self::Close => 0x8,
            Self::Ping => 0x9,
            Self::Pong => 0xA,
            Self::Unknown(v) => *v,
        }
    }

    /// Returns `true` if this is a control opcode (close, ping, pong).
    pub fn is_control(&self) -> bool {
        matches!(self, Self::Close | Self::Ping | Self::Pong)
    }

    /// Returns `true` if this is a data opcode (text, binary, continuation).
    pub fn is_data(&self) -> bool {
        matches!(self, Self::Text | Self::Binary | Self::Continuation)
    }
}

impl fmt::Display for WsOpcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Continuation => write!(f, "Continuation"),
            Self::Text => write!(f, "Text"),
            Self::Binary => write!(f, "Binary"),
            Self::Close => write!(f, "Close"),
            Self::Ping => write!(f, "Ping"),
            Self::Pong => write!(f, "Pong"),
            Self::Unknown(v) => write!(f, "Unknown(0x{v:X})"),
        }
    }
}

// ---------------------------------------------------------------------------
// WebSocket Frame
// ---------------------------------------------------------------------------

/// A parsed WebSocket frame.
#[derive(Debug, Clone)]
pub struct WsFrame {
    /// Whether this is the final fragment.
    pub fin: bool,
    /// RSV1 bit (used for extensions like per-message compression).
    pub rsv1: bool,
    /// RSV2 bit (reserved).
    pub rsv2: bool,
    /// RSV3 bit (reserved).
    pub rsv3: bool,
    /// The frame opcode.
    pub opcode: WsOpcode,
    /// Whether the payload is masked.
    pub masked: bool,
    /// The masking key (4 bytes), if masked.
    pub mask_key: [u8; 4],
    /// The payload data (unmasked).
    pub payload: Vec<u8>,
}

impl WsFrame {
    /// Create a new frame with the given opcode and payload.
    pub fn new(opcode: WsOpcode, payload: Vec<u8>) -> Self {
        Self {
            fin: true,
            rsv1: false,
            rsv2: false,
            rsv3: false,
            opcode,
            masked: false,
            mask_key: [0; 4],
            payload,
        }
    }

    /// Create a text frame.
    pub fn text(data: &str) -> Self {
        Self::new(WsOpcode::Text, data.as_bytes().to_vec())
    }

    /// Create a binary frame.
    pub fn binary(data: Vec<u8>) -> Self {
        Self::new(WsOpcode::Binary, data)
    }

    /// Create a ping frame.
    pub fn ping(data: Vec<u8>) -> Self {
        Self::new(WsOpcode::Ping, data)
    }

    /// Create a pong frame (echoes the ping payload).
    pub fn pong(data: Vec<u8>) -> Self {
        Self::new(WsOpcode::Pong, data)
    }

    /// Create a close frame.
    pub fn close(code: WsCloseCode, reason: &str) -> Self {
        let mut payload = Vec::with_capacity(2 + reason.len());
        let code_val = code.as_u16();
        payload.push((code_val >> 8) as u8);
        payload.push((code_val & 0xFF) as u8);
        payload.extend_from_slice(reason.as_bytes());
        Self::new(WsOpcode::Close, payload)
    }

    /// Create a continuation frame.
    pub fn continuation(data: Vec<u8>, fin: bool) -> Self {
        let mut frame = Self::new(WsOpcode::Continuation, data);
        frame.fin = fin;
        frame
    }

    /// Set the masking key and enable masking.
    pub fn with_mask(mut self, key: [u8; 4]) -> Self {
        self.masked = true;
        self.mask_key = key;
        self
    }

    /// Returns the payload as a UTF-8 string (lossy).
    pub fn payload_text(&self) -> String {
        String::from_utf8_lossy(&self.payload).into_owned()
    }

    /// Returns the total wire size of this frame.
    pub fn wire_size(&self) -> usize {
        let mut size = 2; // header bytes
        let payload_len = self.payload.len();
        if payload_len <= 125 {
            // length fits in 7 bits
        } else if payload_len <= 65535 {
            size += 2;
        } else {
            size += 8;
        }
        if self.masked {
            size += 4;
        }
        size + payload_len
    }

    /// Serialize this frame to bytes for sending.
    pub fn encode(&self, apply_mask: bool) -> Vec<u8> {
        let payload_len = self.payload.len();
        let mut buf = Vec::with_capacity(self.wire_size());

        // First byte: FIN | RSV1 | RSV2 | RSV3 | opcode.
        let mut b0 = self.opcode.as_u8();
        if self.fin {
            b0 |= 0x80;
        }
        if self.rsv1 {
            b0 |= 0x40;
        }
        if self.rsv2 {
            b0 |= 0x20;
        }
        if self.rsv3 {
            b0 |= 0x10;
        }
        buf.push(b0);

        // Second byte: MASK | payload length.
        let mask_bit: u8 = if apply_mask { 0x80 } else { 0x00 };
        if payload_len <= 125 {
            buf.push(mask_bit | payload_len as u8);
        } else if payload_len <= 65535 {
            buf.push(mask_bit | 126);
            buf.push((payload_len >> 8) as u8);
            buf.push((payload_len & 0xFF) as u8);
        } else {
            buf.push(mask_bit | 127);
            let len64 = payload_len as u64;
            for i in (0..8).rev() {
                buf.push((len64 >> (i * 8)) as u8);
            }
        }

        // Masking key.
        if apply_mask {
            buf.extend_from_slice(&self.mask_key);
            // Masked payload.
            for (i, byte) in self.payload.iter().enumerate() {
                buf.push(byte ^ self.mask_key[i % 4]);
            }
        } else {
            buf.extend_from_slice(&self.payload);
        }

        buf
    }

    /// Parse a frame from a byte buffer. Returns the frame and number of bytes consumed.
    pub fn decode(buf: &[u8]) -> WsResult<Option<(Self, usize)>> {
        if buf.len() < 2 {
            return Ok(None); // Need more data.
        }

        let b0 = buf[0];
        let b1 = buf[1];

        let fin = (b0 & 0x80) != 0;
        let rsv1 = (b0 & 0x40) != 0;
        let rsv2 = (b0 & 0x20) != 0;
        let rsv3 = (b0 & 0x10) != 0;
        let opcode = WsOpcode::from_u8(b0 & 0x0F);
        let masked = (b1 & 0x80) != 0;
        let len7 = (b1 & 0x7F) as usize;

        let mut offset = 2;

        // Extended payload length.
        let payload_len = if len7 <= 125 {
            len7
        } else if len7 == 126 {
            if buf.len() < offset + 2 {
                return Ok(None);
            }
            let len = ((buf[offset] as usize) << 8) | (buf[offset + 1] as usize);
            offset += 2;
            len
        } else {
            // len7 == 127
            if buf.len() < offset + 8 {
                return Ok(None);
            }
            let mut len = 0u64;
            for i in 0..8 {
                len = (len << 8) | (buf[offset + i] as u64);
            }
            offset += 8;
            if len > MAX_FRAME_PAYLOAD as u64 {
                return Err(WsError::FrameTooLarge);
            }
            len as usize
        };

        if payload_len > MAX_FRAME_PAYLOAD {
            return Err(WsError::FrameTooLarge);
        }

        // Masking key.
        let mut mask_key = [0u8; 4];
        if masked {
            if buf.len() < offset + 4 {
                return Ok(None);
            }
            mask_key.copy_from_slice(&buf[offset..offset + 4]);
            offset += 4;
        }

        // Payload.
        if buf.len() < offset + payload_len {
            return Ok(None); // Need more data.
        }

        let mut payload = buf[offset..offset + payload_len].to_vec();
        if masked {
            for (i, byte) in payload.iter_mut().enumerate() {
                *byte ^= mask_key[i % 4];
            }
        }

        let total_consumed = offset + payload_len;

        let frame = WsFrame {
            fin,
            rsv1,
            rsv2,
            rsv3,
            opcode,
            masked,
            mask_key,
            payload,
        };

        // Validate control frames.
        if opcode.is_control() {
            if !fin {
                return Err(WsError::ProtocolError(
                    "control frame must not be fragmented".to_string(),
                ));
            }
            if payload_len > 125 {
                return Err(WsError::ProtocolError(
                    "control frame payload must be <= 125 bytes".to_string(),
                ));
            }
        }

        Ok(Some((frame, total_consumed)))
    }
}

// ---------------------------------------------------------------------------
// WebSocket Messages (reassembled from frames)
// ---------------------------------------------------------------------------

/// A complete WebSocket message (may be reassembled from fragments).
#[derive(Debug, Clone)]
pub enum WsMessage {
    /// A text message.
    Text(String),
    /// A binary message.
    Binary(Vec<u8>),
    /// A ping message.
    Ping(Vec<u8>),
    /// A pong message.
    Pong(Vec<u8>),
    /// A close message with status code and reason.
    Close(u16, String),
}

impl WsMessage {
    /// Returns `true` if this is a text message.
    pub fn is_text(&self) -> bool {
        matches!(self, WsMessage::Text(_))
    }

    /// Returns `true` if this is a binary message.
    pub fn is_binary(&self) -> bool {
        matches!(self, WsMessage::Binary(_))
    }

    /// Returns `true` if this is a close message.
    pub fn is_close(&self) -> bool {
        matches!(self, WsMessage::Close(_, _))
    }

    /// Returns `true` if this is a control message.
    pub fn is_control(&self) -> bool {
        matches!(self, WsMessage::Ping(_) | WsMessage::Pong(_) | WsMessage::Close(_, _))
    }

    /// Get the text content, if this is a text message.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            WsMessage::Text(t) => Some(t),
            _ => None,
        }
    }

    /// Get the binary content, if this is a binary message.
    pub fn as_binary(&self) -> Option<&[u8]> {
        match self {
            WsMessage::Binary(b) => Some(b),
            _ => None,
        }
    }

    /// Approximate size in bytes of this message.
    pub fn size(&self) -> usize {
        match self {
            WsMessage::Text(t) => t.len(),
            WsMessage::Binary(b) => b.len(),
            WsMessage::Ping(d) | WsMessage::Pong(d) => d.len(),
            WsMessage::Close(_, r) => 2 + r.len(),
        }
    }
}

impl fmt::Display for WsMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WsMessage::Text(t) => {
                let preview = if t.len() > 64 {
                    format!("{}...", &t[..64])
                } else {
                    t.clone()
                };
                write!(f, "Text({preview})")
            }
            WsMessage::Binary(b) => write!(f, "Binary({} bytes)", b.len()),
            WsMessage::Ping(d) => write!(f, "Ping({} bytes)", d.len()),
            WsMessage::Pong(d) => write!(f, "Pong({} bytes)", d.len()),
            WsMessage::Close(code, reason) => write!(f, "Close({code}, {reason})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Connection State
// ---------------------------------------------------------------------------

/// WebSocket connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WsState {
    /// Establishing TCP connection and performing handshake.
    Connecting,
    /// Connection is open and ready for data transfer.
    Open,
    /// Close handshake has been initiated (sent close frame).
    Closing,
    /// Connection is fully closed.
    Closed,
    /// Attempting to reconnect after a disconnection.
    Reconnecting,
}

impl fmt::Display for WsState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Connecting => write!(f, "Connecting"),
            Self::Open => write!(f, "Open"),
            Self::Closing => write!(f, "Closing"),
            Self::Closed => write!(f, "Closed"),
            Self::Reconnecting => write!(f, "Reconnecting"),
        }
    }
}

// ---------------------------------------------------------------------------
// Reconnect policy
// ---------------------------------------------------------------------------

/// Configuration for automatic reconnection.
#[derive(Debug, Clone)]
pub struct ReconnectPolicy {
    /// Whether auto-reconnect is enabled.
    pub enabled: bool,
    /// Initial delay before the first reconnect attempt.
    pub base_delay: Duration,
    /// Maximum delay between reconnect attempts.
    pub max_delay: Duration,
    /// Backoff multiplier applied after each failed attempt.
    pub multiplier: f64,
    /// Maximum number of reconnect attempts (0 = unlimited).
    pub max_attempts: u32,
    /// Whether to add random jitter to the delay.
    pub jitter: bool,
}

impl Default for ReconnectPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            base_delay: Duration::from_millis(DEFAULT_RECONNECT_BASE_MS),
            max_delay: Duration::from_millis(DEFAULT_RECONNECT_MAX_MS),
            multiplier: DEFAULT_RECONNECT_MULTIPLIER,
            max_attempts: DEFAULT_MAX_RECONNECT_ATTEMPTS,
            jitter: true,
        }
    }
}

impl ReconnectPolicy {
    /// Create a policy with reconnection disabled.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Calculate the delay for the given attempt number (0-based).
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let base_ms = self.base_delay.as_millis() as f64;
        let multiplied = base_ms * self.multiplier.powi(attempt as i32);
        let capped = multiplied.min(self.max_delay.as_millis() as f64);
        let with_jitter = if self.jitter {
            // Simple deterministic pseudo-jitter based on attempt number.
            let jitter_factor = 0.5 + 0.5 * ((attempt as f64 * 7.31).sin().abs());
            capped * jitter_factor
        } else {
            capped
        };
        Duration::from_millis(with_jitter as u64)
    }

    /// Returns `true` if we should attempt another reconnect.
    pub fn should_retry(&self, attempt: u32) -> bool {
        if !self.enabled {
            return false;
        }
        self.max_attempts == 0 || attempt < self.max_attempts
    }
}

// ---------------------------------------------------------------------------
// WebSocket statistics
// ---------------------------------------------------------------------------

/// Statistics about a WebSocket connection.
#[derive(Debug, Clone, Default)]
pub struct WsStats {
    /// Total messages sent.
    pub messages_sent: u64,
    /// Total messages received.
    pub messages_received: u64,
    /// Total bytes sent (payload only).
    pub bytes_sent: u64,
    /// Total bytes received (payload only).
    pub bytes_received: u64,
    /// Total frames sent.
    pub frames_sent: u64,
    /// Total frames received.
    pub frames_received: u64,
    /// Number of ping frames sent.
    pub pings_sent: u64,
    /// Number of pong frames received.
    pub pongs_received: u64,
    /// Number of reconnect attempts.
    pub reconnect_attempts: u32,
    /// Number of successful reconnections.
    pub reconnections: u32,
    /// Connection uptime in seconds (approximate).
    pub uptime_secs: f64,
    /// Time of last message received.
    pub last_message_time: Option<Instant>,
    /// Time of last pong received.
    pub last_pong_time: Option<Instant>,
}

impl WsStats {
    /// Reset all statistics.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Returns the average message size sent (bytes per message).
    pub fn avg_send_size(&self) -> f64 {
        if self.messages_sent == 0 {
            0.0
        } else {
            self.bytes_sent as f64 / self.messages_sent as f64
        }
    }

    /// Returns the average message size received (bytes per message).
    pub fn avg_recv_size(&self) -> f64 {
        if self.messages_received == 0 {
            0.0
        } else {
            self.bytes_received as f64 / self.messages_received as f64
        }
    }
}

impl fmt::Display for WsStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WsStats[sent={}/{:.1}KB, recv={}/{:.1}KB, pings={}, reconnects={}]",
            self.messages_sent,
            self.bytes_sent as f64 / 1024.0,
            self.messages_received,
            self.bytes_received as f64 / 1024.0,
            self.pings_sent,
            self.reconnections
        )
    }
}

// ---------------------------------------------------------------------------
// WebSocket Configuration
// ---------------------------------------------------------------------------

/// Configuration for a WebSocket connection.
#[derive(Debug, Clone)]
pub struct WsConfig {
    /// Connect timeout.
    pub connect_timeout: Duration,
    /// Read timeout for non-blocking reads.
    pub read_timeout: Duration,
    /// Ping interval (how often to send ping frames).
    pub ping_interval: Duration,
    /// How long to wait for a pong before considering the connection dead.
    pub pong_timeout: Duration,
    /// Reconnection policy.
    pub reconnect_policy: ReconnectPolicy,
    /// Maximum payload size for a single frame.
    pub max_frame_payload: usize,
    /// Maximum size for a reassembled message.
    pub max_message_size: usize,
    /// Maximum outbound queue size during reconnect.
    pub max_queue_size: usize,
    /// Additional headers to send during the handshake.
    pub extra_headers: Vec<(String, String)>,
    /// Subprotocols to request (Sec-WebSocket-Protocol).
    pub subprotocols: Vec<String>,
}

impl Default for WsConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS),
            read_timeout: Duration::from_millis(100),
            ping_interval: Duration::from_secs(DEFAULT_PING_INTERVAL_SECS),
            pong_timeout: Duration::from_secs(DEFAULT_PONG_TIMEOUT_SECS),
            reconnect_policy: ReconnectPolicy::default(),
            max_frame_payload: MAX_FRAME_PAYLOAD,
            max_message_size: MAX_MESSAGE_SIZE,
            max_queue_size: MAX_QUEUE_SIZE,
            extra_headers: Vec::new(),
            subprotocols: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Fragment Assembler
// ---------------------------------------------------------------------------

/// Reassembles fragmented WebSocket messages.
struct FragmentAssembler {
    /// The opcode from the first fragment.
    opcode: Option<WsOpcode>,
    /// Accumulated payload bytes.
    buffer: Vec<u8>,
    /// Maximum message size.
    max_size: usize,
}

impl FragmentAssembler {
    fn new(max_size: usize) -> Self {
        Self {
            opcode: None,
            buffer: Vec::new(),
            max_size,
        }
    }

    /// Begin a new fragmented message.
    fn begin(&mut self, opcode: WsOpcode, data: &[u8]) -> WsResult<()> {
        if self.opcode.is_some() {
            return Err(WsError::ProtocolError(
                "new message started while fragment in progress".to_string(),
            ));
        }
        self.opcode = Some(opcode);
        self.buffer.clear();
        self.buffer.extend_from_slice(data);
        if self.buffer.len() > self.max_size {
            return Err(WsError::MessageTooLarge);
        }
        Ok(())
    }

    /// Append a continuation fragment.
    fn append(&mut self, data: &[u8]) -> WsResult<()> {
        if self.opcode.is_none() {
            return Err(WsError::ProtocolError(
                "continuation frame without initial fragment".to_string(),
            ));
        }
        self.buffer.extend_from_slice(data);
        if self.buffer.len() > self.max_size {
            return Err(WsError::MessageTooLarge);
        }
        Ok(())
    }

    /// Finalize the fragmented message and return it.
    fn finish(&mut self) -> WsResult<WsMessage> {
        let opcode = self.opcode.take().ok_or_else(|| {
            WsError::ProtocolError("finish called without begin".to_string())
        })?;
        let data = std::mem::take(&mut self.buffer);
        match opcode {
            WsOpcode::Text => {
                let text = String::from_utf8(data).map_err(|_| {
                    WsError::ProtocolError("invalid UTF-8 in text message".to_string())
                })?;
                Ok(WsMessage::Text(text))
            }
            WsOpcode::Binary => Ok(WsMessage::Binary(data)),
            _ => Err(WsError::ProtocolError(format!(
                "unexpected opcode for fragmented message: {opcode}"
            ))),
        }
    }

    /// Returns `true` if a fragmented message is in progress.
    fn in_progress(&self) -> bool {
        self.opcode.is_some()
    }

    /// Reset the assembler, discarding any partial message.
    fn reset(&mut self) {
        self.opcode = None;
        self.buffer.clear();
    }
}

// ---------------------------------------------------------------------------
// Parsed WebSocket URL
// ---------------------------------------------------------------------------

/// A parsed WebSocket URL.
#[derive(Debug, Clone)]
pub struct WsUrl {
    /// Whether this is a secure connection (wss://).
    pub secure: bool,
    /// The hostname.
    pub host: String,
    /// The port.
    pub port: u16,
    /// The path (including query string).
    pub path: String,
}

impl WsUrl {
    /// Parse a WebSocket URL.
    pub fn parse(url: &str) -> WsResult<Self> {
        let url = url.trim();

        let (secure, rest) = if let Some(stripped) = url.strip_prefix("wss://") {
            (true, stripped)
        } else if let Some(stripped) = url.strip_prefix("ws://") {
            (false, stripped)
        } else {
            return Err(WsError::InvalidUrl(
                "expected ws:// or wss:// scheme".to_string(),
            ));
        };

        // Split host and path.
        let (authority, path) = if let Some(idx) = rest.find('/') {
            (&rest[..idx], &rest[idx..])
        } else {
            (rest, "/")
        };

        // Split host and port.
        let (host, port) = if let Some(idx) = authority.rfind(':') {
            let port_str = &authority[idx + 1..];
            let port: u16 = port_str
                .parse()
                .map_err(|_| WsError::InvalidUrl(format!("invalid port: {port_str}")))?;
            (authority[..idx].to_string(), port)
        } else {
            let default_port = if secure { 443 } else { 80 };
            (authority.to_string(), default_port)
        };

        if host.is_empty() {
            return Err(WsError::InvalidUrl("empty hostname".to_string()));
        }

        Ok(Self {
            secure,
            host,
            port,
            path: path.to_string(),
        })
    }

    /// Returns the authority string for the Host header.
    pub fn authority(&self) -> String {
        let default_port = if self.secure { 443 } else { 80 };
        if self.port == default_port {
            self.host.clone()
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }

    /// Returns the address string for TCP connection.
    pub fn socket_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

// ---------------------------------------------------------------------------
// WebSocket Client
// ---------------------------------------------------------------------------

/// A WebSocket client connection.
pub struct WebSocket {
    /// The URL we are connecting to.
    url: WsUrl,
    /// The original URL string (for reconnection).
    url_string: String,
    /// The TCP stream (if connected).
    stream: Option<TcpStream>,
    /// Current connection state.
    state: WsState,
    /// Configuration.
    config: WsConfig,
    /// Read buffer for incoming data.
    read_buf: Vec<u8>,
    /// Fragment assembler.
    fragments: FragmentAssembler,
    /// Outbound message queue (used during reconnection).
    outbound_queue: VecDeque<Vec<u8>>,
    /// Connection statistics.
    stats: WsStats,
    /// Masking key counter (simple incrementing for determinism in tests).
    mask_counter: u32,
    /// Time of last ping sent.
    last_ping_time: Option<Instant>,
    /// Whether we are waiting for a pong.
    awaiting_pong: bool,
    /// Time the connection was established.
    connected_at: Option<Instant>,
    /// Number of consecutive reconnect attempts.
    reconnect_attempt: u32,
    /// Time of last reconnect attempt.
    last_reconnect_time: Option<Instant>,
    /// The selected subprotocol (from server response).
    selected_protocol: Option<String>,
    /// Close code received from remote.
    remote_close_code: Option<u16>,
    /// Close reason received from remote.
    remote_close_reason: Option<String>,
}

impl WebSocket {
    /// Connect to a WebSocket server.
    pub fn connect(url: &str) -> WsResult<Self> {
        Self::connect_with_config(url, WsConfig::default())
    }

    /// Connect to a WebSocket server with custom configuration.
    pub fn connect_with_config(url: &str, config: WsConfig) -> WsResult<Self> {
        let ws_url = WsUrl::parse(url)?;

        if ws_url.secure {
            return Err(WsError::ConnectionFailed(
                "TLS/WSS not supported in this implementation".to_string(),
            ));
        }

        let mut ws = Self {
            url: ws_url,
            url_string: url.to_string(),
            stream: None,
            state: WsState::Connecting,
            config,
            read_buf: Vec::with_capacity(READ_BUF_SIZE),
            fragments: FragmentAssembler::new(MAX_MESSAGE_SIZE),
            outbound_queue: VecDeque::new(),
            stats: WsStats::default(),
            mask_counter: 0,
            last_ping_time: None,
            awaiting_pong: false,
            connected_at: None,
            reconnect_attempt: 0,
            last_reconnect_time: None,
            selected_protocol: None,
            remote_close_code: None,
            remote_close_reason: None,
        };

        ws.perform_connect()?;
        Ok(ws)
    }

    /// Perform the TCP connection and WebSocket handshake.
    fn perform_connect(&mut self) -> WsResult<()> {
        self.state = WsState::Connecting;

        // TCP connect.
        let addr = self.url.socket_addr();
        let stream = TcpStream::connect_timeout(
            &addr
                .parse()
                .map_err(|_| {
                    // Try DNS resolution.
                    WsError::ConnectionFailed(format!("cannot resolve address: {addr}"))
                })
                .or_else(|_| {
                    use std::net::ToSocketAddrs;
                    let addrs: Vec<_> = addr
                        .to_socket_addrs()
                        .map_err(|e| WsError::ConnectionFailed(e.to_string()))?
                        .collect();
                    addrs
                        .first()
                        .copied()
                        .ok_or_else(|| WsError::ConnectionFailed("no addresses found".to_string()))
                })?,
            self.config.connect_timeout,
        )
        .map_err(|e| {
            if e.kind() == io::ErrorKind::TimedOut {
                WsError::ConnectTimeout
            } else {
                WsError::ConnectionFailed(e.to_string())
            }
        })?;

        stream
            .set_read_timeout(Some(self.config.read_timeout))
            .map_err(|e| WsError::IoError(e.to_string()))?;
        stream
            .set_nodelay(true)
            .map_err(|e| WsError::IoError(e.to_string()))?;

        self.stream = Some(stream);

        // Perform HTTP upgrade handshake.
        self.perform_handshake()?;

        self.state = WsState::Open;
        self.connected_at = Some(Instant::now());
        self.reconnect_attempt = 0;
        self.fragments.reset();
        self.read_buf.clear();

        // Flush any queued messages.
        self.flush_queue()?;

        Ok(())
    }

    /// Perform the WebSocket HTTP upgrade handshake.
    fn perform_handshake(&mut self) -> WsResult<()> {
        // Generate a Sec-WebSocket-Key (16 random bytes, base64 encoded).
        let ws_key = self.generate_ws_key();

        let stream = self
            .stream
            .as_mut()
            .ok_or(WsError::NotConnected)?;

        // Build the HTTP upgrade request.
        let mut request = format!(
            "GET {} HTTP/1.1\r\n\
             Host: {}\r\n\
             Upgrade: websocket\r\n\
             Connection: Upgrade\r\n\
             Sec-WebSocket-Key: {}\r\n\
             Sec-WebSocket-Version: 13\r\n",
            self.url.path,
            self.url.authority(),
            ws_key,
        );

        // Add subprotocols.
        if !self.config.subprotocols.is_empty() {
            request.push_str(&format!(
                "Sec-WebSocket-Protocol: {}\r\n",
                self.config.subprotocols.join(", ")
            ));
        }

        // Add extra headers.
        for (name, value) in &self.config.extra_headers {
            request.push_str(&format!("{name}: {value}\r\n"));
        }

        request.push_str("\r\n");

        stream
            .write_all(request.as_bytes())
            .map_err(|e| WsError::IoError(e.to_string()))?;
        stream
            .flush()
            .map_err(|e| WsError::IoError(e.to_string()))?;

        // Read the response.
        let mut response_buf = Vec::with_capacity(4096);
        let mut tmp = [0u8; 1024];

        // Set a longer timeout for the handshake response.
        stream
            .set_read_timeout(Some(self.config.connect_timeout))
            .map_err(|e| WsError::IoError(e.to_string()))?;

        loop {
            let n = stream
                .read(&mut tmp)
                .map_err(|e| WsError::HandshakeFailed(e.to_string()))?;
            if n == 0 {
                return Err(WsError::HandshakeFailed(
                    "connection closed during handshake".to_string(),
                ));
            }
            response_buf.extend_from_slice(&tmp[..n]);

            // Check if we have the full header.
            if let Some(header_end) = find_header_end(&response_buf) {
                let header_text =
                    String::from_utf8_lossy(&response_buf[..header_end]).into_owned();

                // Parse the status line.
                let first_line = header_text
                    .lines()
                    .next()
                    .ok_or_else(|| WsError::HandshakeFailed("empty response".to_string()))?;

                let parts: Vec<&str> = first_line.splitn(3, ' ').collect();
                if parts.len() < 2 {
                    return Err(WsError::HandshakeFailed(format!(
                        "malformed status line: {first_line}"
                    )));
                }

                let status_code: u16 = parts[1]
                    .parse()
                    .map_err(|_| WsError::HandshakeFailed("invalid status code".to_string()))?;

                if status_code != 101 {
                    let reason = parts.get(2).unwrap_or(&"");
                    return Err(WsError::UpgradeRejected(
                        status_code,
                        reason.to_string(),
                    ));
                }

                // Parse response headers for Sec-WebSocket-Accept validation.
                let mut accept_key = None;
                for line in header_text.lines().skip(1) {
                    if let Some(colon) = line.find(':') {
                        let name = line[..colon].trim().to_lowercase();
                        let value = line[colon + 1..].trim();
                        if name == "sec-websocket-accept" {
                            accept_key = Some(value.to_string());
                        }
                        if name == "sec-websocket-protocol" {
                            self.selected_protocol = Some(value.to_string());
                        }
                    }
                }

                // Validate the accept key.
                let expected_accept = compute_accept_key(&ws_key);
                if let Some(ref actual) = accept_key {
                    if *actual != expected_accept {
                        return Err(WsError::InvalidAcceptKey);
                    }
                }
                // Note: some servers might not send the accept key; we allow it.

                // Save any trailing data (after the header) into the read buffer.
                let trailing_start = header_end + 4;
                if trailing_start < response_buf.len() {
                    self.read_buf
                        .extend_from_slice(&response_buf[trailing_start..]);
                }

                // Restore the normal read timeout.
                stream
                    .set_read_timeout(Some(self.config.read_timeout))
                    .map_err(|e| WsError::IoError(e.to_string()))?;

                return Ok(());
            }

            if response_buf.len() > 16384 {
                return Err(WsError::HandshakeFailed(
                    "response headers too large".to_string(),
                ));
            }
        }
    }

    /// Generate a Sec-WebSocket-Key (deterministic for simplicity).
    fn generate_ws_key(&mut self) -> String {
        // Use a simple counter-based approach for test reproducibility.
        // In production, use a cryptographically secure random source.
        self.mask_counter = self.mask_counter.wrapping_add(1);
        let mut bytes = [0u8; 16];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ((self.mask_counter as usize * 31 + i * 17 + 53) % 256) as u8;
        }
        simple_base64_encode(&bytes)
    }

    /// Generate a masking key for outgoing frames.
    fn generate_mask_key(&mut self) -> [u8; 4] {
        self.mask_counter = self.mask_counter.wrapping_add(1);
        let n = self.mask_counter;
        [
            (n & 0xFF) as u8,
            ((n >> 8) & 0xFF) as u8,
            ((n >> 16) & 0xFF) as u8,
            ((n >> 24) & 0xFF) as u8,
        ]
    }

    /// Send a text message.
    pub fn send_text(&mut self, text: &str) -> WsResult<()> {
        let frame = WsFrame::text(text);
        self.send_frame(frame)?;
        self.stats.messages_sent += 1;
        self.stats.bytes_sent += text.len() as u64;
        Ok(())
    }

    /// Send a binary message.
    pub fn send_binary(&mut self, data: &[u8]) -> WsResult<()> {
        let frame = WsFrame::binary(data.to_vec());
        self.send_frame(frame)?;
        self.stats.messages_sent += 1;
        self.stats.bytes_sent += data.len() as u64;
        Ok(())
    }

    /// Send a ping frame.
    pub fn send_ping(&mut self, data: &[u8]) -> WsResult<()> {
        let frame = WsFrame::ping(data.to_vec());
        self.send_frame(frame)?;
        self.stats.pings_sent += 1;
        self.last_ping_time = Some(Instant::now());
        self.awaiting_pong = true;
        Ok(())
    }

    /// Initiate the close handshake.
    pub fn close(&mut self, code: WsCloseCode, reason: &str) -> WsResult<()> {
        if self.state != WsState::Open {
            return Err(WsError::NotConnected);
        }
        let frame = WsFrame::close(code, reason);
        self.send_frame(frame)?;
        self.state = WsState::Closing;
        Ok(())
    }

    /// Send a raw frame.
    fn send_frame(&mut self, frame: WsFrame) -> WsResult<()> {
        // Client frames must be masked (RFC 6455 Section 5.1).
        let mask_key = self.generate_mask_key();
        let masked_frame = frame.with_mask(mask_key);
        let encoded = masked_frame.encode(true);

        match self.state {
            WsState::Open | WsState::Closing => {
                if let Some(ref mut stream) = self.stream {
                    stream
                        .write_all(&encoded)
                        .map_err(|e| WsError::IoError(e.to_string()))?;
                    stream
                        .flush()
                        .map_err(|e| WsError::IoError(e.to_string()))?;
                    self.stats.frames_sent += 1;
                } else {
                    return Err(WsError::NotConnected);
                }
            }
            WsState::Reconnecting | WsState::Connecting => {
                // Queue the message for later.
                if self.outbound_queue.len() >= self.config.max_queue_size {
                    return Err(WsError::QueueFull);
                }
                self.outbound_queue.push_back(encoded);
            }
            WsState::Closed => {
                return Err(WsError::NotConnected);
            }
        }
        Ok(())
    }

    /// Flush the outbound queue (after reconnection).
    fn flush_queue(&mut self) -> WsResult<()> {
        while let Some(data) = self.outbound_queue.pop_front() {
            if let Some(ref mut stream) = self.stream {
                stream
                    .write_all(&data)
                    .map_err(|e| WsError::IoError(e.to_string()))?;
                self.stats.frames_sent += 1;
            }
        }
        if let Some(ref mut stream) = self.stream {
            stream
                .flush()
                .map_err(|e| WsError::IoError(e.to_string()))?;
        }
        Ok(())
    }

    /// Poll for incoming messages. Returns `None` if no message is available.
    pub fn poll(&mut self) -> WsResult<Option<WsMessage>> {
        // Check keepalive.
        self.check_keepalive()?;

        // Check reconnect.
        if self.state == WsState::Reconnecting {
            self.try_reconnect()?;
            if self.state != WsState::Open {
                return Ok(None);
            }
        }

        if self.state != WsState::Open && self.state != WsState::Closing {
            return Ok(None);
        }

        // Read data from the socket.
        if let Err(e) = self.read_from_socket() {
            match e {
                WsError::ReadTimeout => {} // Expected for non-blocking reads.
                _ => {
                    self.handle_disconnect()?;
                    return Ok(None);
                }
            }
        }

        // Try to parse a frame from the read buffer.
        self.try_parse_frame()
    }

    /// Read data from the socket into the read buffer.
    fn read_from_socket(&mut self) -> WsResult<()> {
        let stream = self
            .stream
            .as_mut()
            .ok_or(WsError::NotConnected)?;

        let mut tmp = [0u8; READ_BUF_SIZE];
        match stream.read(&mut tmp) {
            Ok(0) => {
                // Connection closed.
                Err(WsError::ConnectionClosed(1006, "connection reset".to_string()))
            }
            Ok(n) => {
                self.read_buf.extend_from_slice(&tmp[..n]);
                Ok(())
            }
            Err(e) => {
                if e.kind() == io::ErrorKind::WouldBlock || e.kind() == io::ErrorKind::TimedOut {
                    Err(WsError::ReadTimeout)
                } else {
                    Err(WsError::IoError(e.to_string()))
                }
            }
        }
    }

    /// Try to parse a complete frame from the read buffer.
    fn try_parse_frame(&mut self) -> WsResult<Option<WsMessage>> {
        loop {
            match WsFrame::decode(&self.read_buf)? {
                None => return Ok(None),
                Some((frame, consumed)) => {
                    self.read_buf.drain(..consumed);
                    self.stats.frames_received += 1;

                    match frame.opcode {
                        WsOpcode::Ping => {
                            // Respond with pong.
                            let pong = WsFrame::pong(frame.payload.clone());
                            self.send_frame(pong)?;
                            return Ok(Some(WsMessage::Ping(frame.payload)));
                        }
                        WsOpcode::Pong => {
                            self.awaiting_pong = false;
                            self.stats.pongs_received += 1;
                            self.stats.last_pong_time = Some(Instant::now());
                            return Ok(Some(WsMessage::Pong(frame.payload)));
                        }
                        WsOpcode::Close => {
                            let (code, reason) = parse_close_payload(&frame.payload);
                            self.remote_close_code = Some(code);
                            self.remote_close_reason = Some(reason.clone());

                            if self.state == WsState::Open {
                                // Server initiated close — echo it back.
                                let close_frame = WsFrame::close(
                                    WsCloseCode::from_u16(code),
                                    &reason,
                                );
                                let _ = self.send_frame(close_frame);
                            }
                            self.state = WsState::Closed;
                            self.stream = None;
                            return Ok(Some(WsMessage::Close(code, reason)));
                        }
                        WsOpcode::Text | WsOpcode::Binary => {
                            if frame.fin {
                                // Complete message in a single frame.
                                let msg = if frame.opcode == WsOpcode::Text {
                                    let text = String::from_utf8(frame.payload)
                                        .map_err(|_| {
                                            WsError::ProtocolError(
                                                "invalid UTF-8 in text frame".to_string(),
                                            )
                                        })?;
                                    WsMessage::Text(text)
                                } else {
                                    WsMessage::Binary(frame.payload)
                                };
                                self.stats.messages_received += 1;
                                self.stats.bytes_received += msg.size() as u64;
                                self.stats.last_message_time = Some(Instant::now());
                                return Ok(Some(msg));
                            } else {
                                // Start of a fragmented message.
                                self.fragments.begin(frame.opcode, &frame.payload)?;
                            }
                        }
                        WsOpcode::Continuation => {
                            self.fragments.append(&frame.payload)?;
                            if frame.fin {
                                let msg = self.fragments.finish()?;
                                self.stats.messages_received += 1;
                                self.stats.bytes_received += msg.size() as u64;
                                self.stats.last_message_time = Some(Instant::now());
                                return Ok(Some(msg));
                            }
                        }
                        WsOpcode::Unknown(op) => {
                            return Err(WsError::ProtocolError(format!(
                                "unknown opcode: 0x{op:X}"
                            )));
                        }
                    }
                }
            }
        }
    }

    /// Check keepalive timers and send pings if needed.
    fn check_keepalive(&mut self) -> WsResult<()> {
        if self.state != WsState::Open {
            return Ok(());
        }

        let now = Instant::now();

        // Check pong timeout.
        if self.awaiting_pong {
            if let Some(ping_time) = self.last_ping_time {
                if now.duration_since(ping_time) > self.config.pong_timeout {
                    // Pong timeout — connection is dead.
                    self.handle_disconnect()?;
                    return Ok(());
                }
            }
        }

        // Send periodic ping.
        let should_ping = match self.last_ping_time {
            Some(t) => now.duration_since(t) >= self.config.ping_interval,
            None => {
                if let Some(connected) = self.connected_at {
                    now.duration_since(connected) >= self.config.ping_interval
                } else {
                    false
                }
            }
        };

        if should_ping && !self.awaiting_pong {
            self.send_ping(&[])?;
        }

        Ok(())
    }

    /// Handle a disconnection (potentially triggering reconnect).
    fn handle_disconnect(&mut self) -> WsResult<()> {
        self.stream = None;
        self.fragments.reset();

        if self.config.reconnect_policy.enabled && self.state == WsState::Open {
            self.state = WsState::Reconnecting;
            self.reconnect_attempt = 0;
            self.last_reconnect_time = None;
        } else {
            self.state = WsState::Closed;
        }

        Ok(())
    }

    /// Try to reconnect if the backoff delay has elapsed.
    fn try_reconnect(&mut self) -> WsResult<()> {
        if !self.config.reconnect_policy.should_retry(self.reconnect_attempt) {
            self.state = WsState::Closed;
            return Err(WsError::ReconnectExhausted);
        }

        let delay = self
            .config
            .reconnect_policy
            .delay_for_attempt(self.reconnect_attempt);

        if let Some(last_attempt) = self.last_reconnect_time {
            if Instant::now().duration_since(last_attempt) < delay {
                return Ok(()); // Not time yet.
            }
        }

        self.last_reconnect_time = Some(Instant::now());
        self.reconnect_attempt += 1;
        self.stats.reconnect_attempts += 1;

        match self.perform_connect() {
            Ok(()) => {
                self.stats.reconnections += 1;
                Ok(())
            }
            Err(_) => {
                self.state = WsState::Reconnecting;
                Ok(()) // Will retry later.
            }
        }
    }

    /// Returns the current connection state.
    pub fn state(&self) -> WsState {
        self.state
    }

    /// Returns `true` if the connection is open.
    pub fn is_connected(&self) -> bool {
        self.state == WsState::Open
    }

    /// Returns the connection statistics.
    pub fn stats(&self) -> &WsStats {
        &self.stats
    }

    /// Returns the selected subprotocol, if any.
    pub fn selected_protocol(&self) -> Option<&str> {
        self.selected_protocol.as_deref()
    }

    /// Returns the URL this WebSocket is connected to.
    pub fn url(&self) -> &str {
        &self.url_string
    }

    /// Returns the remote close code, if the connection was closed by the remote.
    pub fn remote_close_code(&self) -> Option<u16> {
        self.remote_close_code
    }

    /// Returns the remote close reason, if the connection was closed by the remote.
    pub fn remote_close_reason(&self) -> Option<&str> {
        self.remote_close_reason.as_deref()
    }

    /// Update the uptime statistic.
    pub fn update_uptime(&mut self) {
        if let Some(connected) = self.connected_at {
            self.stats.uptime_secs = connected.elapsed().as_secs_f64();
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Find the \r\n\r\n header terminator in a byte buffer.
fn find_header_end(buf: &[u8]) -> Option<usize> {
    for i in 0..buf.len().saturating_sub(3) {
        if buf[i] == b'\r' && buf[i + 1] == b'\n' && buf[i + 2] == b'\r' && buf[i + 3] == b'\n' {
            return Some(i);
        }
    }
    None
}

/// Parse the close frame payload into (code, reason).
fn parse_close_payload(payload: &[u8]) -> (u16, String) {
    if payload.len() < 2 {
        return (1005, String::new());
    }
    let code = ((payload[0] as u16) << 8) | (payload[1] as u16);
    let reason = if payload.len() > 2 {
        String::from_utf8_lossy(&payload[2..]).into_owned()
    } else {
        String::new()
    };
    (code, reason)
}

/// Compute the expected Sec-WebSocket-Accept value.
fn compute_accept_key(key: &str) -> String {
    // SHA-1 hash of key + GUID.
    // Using a simple SHA-1 implementation since we only need it for the handshake.
    let input = format!("{key}{WS_GUID}");
    let hash = simple_sha1(input.as_bytes());
    simple_base64_encode(&hash)
}

/// Minimal SHA-1 implementation for WebSocket handshake.
fn simple_sha1(data: &[u8]) -> [u8; 20] {
    let mut h0: u32 = 0x67452301;
    let mut h1: u32 = 0xEFCDAB89;
    let mut h2: u32 = 0x98BADCFE;
    let mut h3: u32 = 0x10325476;
    let mut h4: u32 = 0xC3D2E1F0;

    let bit_len = (data.len() as u64) * 8;

    // Pre-processing: add padding.
    let mut padded = data.to_vec();
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0x00);
    }
    // Append original length in bits as 64-bit big-endian.
    for i in (0..8).rev() {
        padded.push((bit_len >> (i * 8)) as u8);
    }

    // Process each 512-bit (64-byte) block.
    for block_start in (0..padded.len()).step_by(64) {
        let block = &padded[block_start..block_start + 64];
        let mut w = [0u32; 80];

        for i in 0..16 {
            w[i] = ((block[i * 4] as u32) << 24)
                | ((block[i * 4 + 1] as u32) << 16)
                | ((block[i * 4 + 2] as u32) << 8)
                | (block[i * 4 + 3] as u32);
        }

        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let mut a = h0;
        let mut b = h1;
        let mut c = h2;
        let mut d = h3;
        let mut e = h4;

        for i in 0..80 {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A827999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1u32),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDCu32),
                _ => (b ^ c ^ d, 0xCA62C1D6u32),
            };

            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(w[i]);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut result = [0u8; 20];
    for (i, &h) in [h0, h1, h2, h3, h4].iter().enumerate() {
        result[i * 4] = (h >> 24) as u8;
        result[i * 4 + 1] = (h >> 16) as u8;
        result[i * 4 + 2] = (h >> 8) as u8;
        result[i * 4 + 3] = h as u8;
    }
    result
}

/// Minimal base64 encoder.
fn simple_base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    let mut i = 0;
    while i + 2 < data.len() {
        let n = ((data[i] as u32) << 16) | ((data[i + 1] as u32) << 8) | (data[i + 2] as u32);
        out.push(CHARS[((n >> 18) & 63) as usize] as char);
        out.push(CHARS[((n >> 12) & 63) as usize] as char);
        out.push(CHARS[((n >> 6) & 63) as usize] as char);
        out.push(CHARS[(n & 63) as usize] as char);
        i += 3;
    }
    let rem = data.len() - i;
    if rem == 1 {
        let n = (data[i] as u32) << 16;
        out.push(CHARS[((n >> 18) & 63) as usize] as char);
        out.push(CHARS[((n >> 12) & 63) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let n = ((data[i] as u32) << 16) | ((data[i + 1] as u32) << 8);
        out.push(CHARS[((n >> 18) & 63) as usize] as char);
        out.push(CHARS[((n >> 12) & 63) as usize] as char);
        out.push(CHARS[((n >> 6) & 63) as usize] as char);
        out.push('=');
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ws_url_parse() {
        let url = WsUrl::parse("ws://localhost:8080/game").unwrap();
        assert_eq!(url.host, "localhost");
        assert_eq!(url.port, 8080);
        assert_eq!(url.path, "/game");
        assert!(!url.secure);
    }

    #[test]
    fn test_ws_url_default_port() {
        let url = WsUrl::parse("ws://example.com/chat").unwrap();
        assert_eq!(url.port, 80);
    }

    #[test]
    fn test_ws_url_wss() {
        let url = WsUrl::parse("wss://secure.example.com").unwrap();
        assert!(url.secure);
        assert_eq!(url.port, 443);
    }

    #[test]
    fn test_ws_url_invalid() {
        assert!(WsUrl::parse("http://example.com").is_err());
        assert!(WsUrl::parse("not-a-url").is_err());
    }

    #[test]
    fn test_close_code_roundtrip() {
        for code_val in [1000, 1001, 1002, 1003, 1007, 1008, 1009, 1010, 1011] {
            let code = WsCloseCode::from_u16(code_val);
            assert_eq!(code.as_u16(), code_val);
        }
        let custom = WsCloseCode::Custom(4000);
        assert_eq!(custom.as_u16(), 4000);
    }

    #[test]
    fn test_opcode_parsing() {
        assert_eq!(WsOpcode::from_u8(0x0), WsOpcode::Continuation);
        assert_eq!(WsOpcode::from_u8(0x1), WsOpcode::Text);
        assert_eq!(WsOpcode::from_u8(0x2), WsOpcode::Binary);
        assert_eq!(WsOpcode::from_u8(0x8), WsOpcode::Close);
        assert_eq!(WsOpcode::from_u8(0x9), WsOpcode::Ping);
        assert_eq!(WsOpcode::from_u8(0xA), WsOpcode::Pong);
        assert!(WsOpcode::Close.is_control());
        assert!(WsOpcode::Text.is_data());
    }

    #[test]
    fn test_frame_encode_decode_text() {
        let frame = WsFrame::text("hello");
        let encoded = frame.encode(false);
        let (decoded, consumed) = WsFrame::decode(&encoded).unwrap().unwrap();
        assert_eq!(consumed, encoded.len());
        assert!(decoded.fin);
        assert_eq!(decoded.opcode, WsOpcode::Text);
        assert_eq!(decoded.payload_text(), "hello");
    }

    #[test]
    fn test_frame_encode_decode_binary() {
        let data = vec![1, 2, 3, 4, 5];
        let frame = WsFrame::binary(data.clone());
        let encoded = frame.encode(false);
        let (decoded, _) = WsFrame::decode(&encoded).unwrap().unwrap();
        assert_eq!(decoded.opcode, WsOpcode::Binary);
        assert_eq!(decoded.payload, data);
    }

    #[test]
    fn test_frame_masked() {
        let frame = WsFrame::text("masked test").with_mask([0x12, 0x34, 0x56, 0x78]);
        let encoded = frame.encode(true);
        let (decoded, _) = WsFrame::decode(&encoded).unwrap().unwrap();
        assert_eq!(decoded.payload_text(), "masked test");
    }

    #[test]
    fn test_frame_close() {
        let frame = WsFrame::close(WsCloseCode::Normal, "goodbye");
        let encoded = frame.encode(false);
        let (decoded, _) = WsFrame::decode(&encoded).unwrap().unwrap();
        assert_eq!(decoded.opcode, WsOpcode::Close);
        let (code, reason) = parse_close_payload(&decoded.payload);
        assert_eq!(code, 1000);
        assert_eq!(reason, "goodbye");
    }

    #[test]
    fn test_frame_large_payload() {
        let data = vec![0xABu8; 300];
        let frame = WsFrame::binary(data.clone());
        let encoded = frame.encode(false);
        let (decoded, _) = WsFrame::decode(&encoded).unwrap().unwrap();
        assert_eq!(decoded.payload.len(), 300);
        assert_eq!(decoded.payload, data);
    }

    #[test]
    fn test_frame_very_large_payload() {
        let data = vec![0xCDu8; 70_000];
        let frame = WsFrame::binary(data.clone());
        let encoded = frame.encode(false);
        let (decoded, _) = WsFrame::decode(&encoded).unwrap().unwrap();
        assert_eq!(decoded.payload.len(), 70_000);
    }

    #[test]
    fn test_frame_incomplete_data() {
        let result = WsFrame::decode(&[0x81]).unwrap();
        assert!(result.is_none());

        let result = WsFrame::decode(&[]).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_sha1_known_vector() {
        // SHA-1("abc") = A9993E36 4706816A BA3E2571 7850C26C 9CD0D89D
        let hash = simple_sha1(b"abc");
        assert_eq!(hash[0], 0xA9);
        assert_eq!(hash[1], 0x99);
        assert_eq!(hash[2], 0x3E);
        assert_eq!(hash[3], 0x36);
    }

    #[test]
    fn test_reconnect_policy_delay() {
        let policy = ReconnectPolicy {
            enabled: true,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            multiplier: 2.0,
            max_attempts: 5,
            jitter: false,
        };
        assert_eq!(policy.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(policy.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(policy.delay_for_attempt(2), Duration::from_millis(400));
        assert!(policy.should_retry(4));
        assert!(!policy.should_retry(5));
    }

    #[test]
    fn test_reconnect_policy_disabled() {
        let policy = ReconnectPolicy::disabled();
        assert!(!policy.should_retry(0));
    }

    #[test]
    fn test_ws_message_types() {
        let text = WsMessage::Text("hello".to_string());
        assert!(text.is_text());
        assert_eq!(text.as_text(), Some("hello"));

        let binary = WsMessage::Binary(vec![1, 2, 3]);
        assert!(binary.is_binary());
        assert_eq!(binary.as_binary(), Some(&[1u8, 2, 3][..]));

        let close = WsMessage::Close(1000, "bye".to_string());
        assert!(close.is_close());
        assert!(close.is_control());
    }

    #[test]
    fn test_ws_stats_default() {
        let stats = WsStats::default();
        assert_eq!(stats.messages_sent, 0);
        assert_eq!(stats.avg_send_size(), 0.0);
    }

    #[test]
    fn test_fragment_assembler() {
        let mut asm = FragmentAssembler::new(1024);
        assert!(!asm.in_progress());

        asm.begin(WsOpcode::Text, b"hello ").unwrap();
        assert!(asm.in_progress());

        asm.append(b"world").unwrap();
        let msg = asm.finish().unwrap();
        assert!(!asm.in_progress());

        match msg {
            WsMessage::Text(t) => assert_eq!(t, "hello world"),
            _ => panic!("expected text message"),
        }
    }

    #[test]
    fn test_fragment_assembler_too_large() {
        let mut asm = FragmentAssembler::new(10);
        assert!(asm.begin(WsOpcode::Binary, &[0u8; 20]).is_err());
    }

    #[test]
    fn test_base64_encode() {
        assert_eq!(simple_base64_encode(b""), "");
        assert_eq!(simple_base64_encode(b"f"), "Zg==");
        assert_eq!(simple_base64_encode(b"fo"), "Zm8=");
        assert_eq!(simple_base64_encode(b"foo"), "Zm9v");
    }

    #[test]
    fn test_ws_config_defaults() {
        let config = WsConfig::default();
        assert_eq!(config.ping_interval.as_secs(), DEFAULT_PING_INTERVAL_SECS);
        assert_eq!(config.max_queue_size, MAX_QUEUE_SIZE);
    }
}
