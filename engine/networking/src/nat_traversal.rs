//! NAT traversal utilities for the Genovo engine.
//!
//! Provides a STUN client for determining a host's public IP and port mapping,
//! NAT type detection (full cone, restricted, port-restricted, symmetric),
//! UDP hole punching with a coordination server, and ICE-lite candidate
//! gathering.
//!
//! # NAT Types
//!
//! | NAT Type          | Same Port? | Filtering           | Hole Punch? |
//! |--------------------|-----------|---------------------|-------------|
//! | Full Cone          | Yes       | None                | Easy        |
//! | Restricted Cone    | Yes       | IP-based            | Moderate    |
//! | Port-Restricted    | Yes       | IP+Port-based       | Moderate    |
//! | Symmetric          | No        | Per-destination     | Hard/Relay  |

use std::collections::HashMap;
use std::fmt;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// STUN Message Types (RFC 5389)
// ---------------------------------------------------------------------------

/// STUN message class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StunClass {
    Request,
    Indication,
    SuccessResponse,
    ErrorResponse,
}

impl fmt::Display for StunClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StunClass::Request => write!(f, "Request"),
            StunClass::Indication => write!(f, "Indication"),
            StunClass::SuccessResponse => write!(f, "Success"),
            StunClass::ErrorResponse => write!(f, "Error"),
        }
    }
}

/// STUN message method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StunMethod {
    Binding,
    Allocate,
    Refresh,
    Send,
    Data,
    CreatePermission,
    ChannelBind,
}

impl fmt::Display for StunMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StunMethod::Binding => write!(f, "Binding"),
            StunMethod::Allocate => write!(f, "Allocate"),
            StunMethod::Refresh => write!(f, "Refresh"),
            StunMethod::Send => write!(f, "Send"),
            StunMethod::Data => write!(f, "Data"),
            StunMethod::CreatePermission => write!(f, "CreatePermission"),
            StunMethod::ChannelBind => write!(f, "ChannelBind"),
        }
    }
}

/// STUN message type (class + method combined).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StunMessageType {
    pub class: StunClass,
    pub method: StunMethod,
}

impl StunMessageType {
    pub fn binding_request() -> Self {
        Self {
            class: StunClass::Request,
            method: StunMethod::Binding,
        }
    }

    pub fn binding_response() -> Self {
        Self {
            class: StunClass::SuccessResponse,
            method: StunMethod::Binding,
        }
    }

    pub fn binding_error() -> Self {
        Self {
            class: StunClass::ErrorResponse,
            method: StunMethod::Binding,
        }
    }

    /// Encode to the wire format (16-bit type value).
    pub fn to_u16(self) -> u16 {
        let method_bits = match self.method {
            StunMethod::Binding => 0x0001,
            StunMethod::Allocate => 0x0003,
            StunMethod::Refresh => 0x0004,
            StunMethod::Send => 0x0006,
            StunMethod::Data => 0x0007,
            StunMethod::CreatePermission => 0x0008,
            StunMethod::ChannelBind => 0x0009,
        };

        let class_bits = match self.class {
            StunClass::Request => 0x0000,
            StunClass::Indication => 0x0010,
            StunClass::SuccessResponse => 0x0100,
            StunClass::ErrorResponse => 0x0110,
        };

        method_bits | class_bits
    }

    /// Decode from wire format.
    pub fn from_u16(value: u16) -> Option<Self> {
        let class_bits = value & 0x0110;
        let method_bits = value & !0x0110;

        let class = match class_bits {
            0x0000 => StunClass::Request,
            0x0010 => StunClass::Indication,
            0x0100 => StunClass::SuccessResponse,
            0x0110 => StunClass::ErrorResponse,
            _ => return None,
        };

        let method = match method_bits {
            0x0001 => StunMethod::Binding,
            0x0003 => StunMethod::Allocate,
            0x0004 => StunMethod::Refresh,
            0x0006 => StunMethod::Send,
            0x0007 => StunMethod::Data,
            0x0008 => StunMethod::CreatePermission,
            0x0009 => StunMethod::ChannelBind,
            _ => return None,
        };

        Some(Self { class, method })
    }
}

// ---------------------------------------------------------------------------
// STUN Attributes
// ---------------------------------------------------------------------------

/// STUN attribute types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StunAttributeType {
    MappedAddress,
    XorMappedAddress,
    Username,
    MessageIntegrity,
    ErrorCode,
    Realm,
    Nonce,
    Software,
    Fingerprint,
    ChangeRequest,
    ResponseOrigin,
    OtherAddress,
}

impl StunAttributeType {
    pub fn to_u16(self) -> u16 {
        match self {
            StunAttributeType::MappedAddress => 0x0001,
            StunAttributeType::XorMappedAddress => 0x0020,
            StunAttributeType::Username => 0x0006,
            StunAttributeType::MessageIntegrity => 0x0008,
            StunAttributeType::ErrorCode => 0x0009,
            StunAttributeType::Realm => 0x0014,
            StunAttributeType::Nonce => 0x0015,
            StunAttributeType::Software => 0x8022,
            StunAttributeType::Fingerprint => 0x8028,
            StunAttributeType::ChangeRequest => 0x0003,
            StunAttributeType::ResponseOrigin => 0x802B,
            StunAttributeType::OtherAddress => 0x802C,
        }
    }

    pub fn from_u16(value: u16) -> Option<Self> {
        match value {
            0x0001 => Some(StunAttributeType::MappedAddress),
            0x0020 => Some(StunAttributeType::XorMappedAddress),
            0x0006 => Some(StunAttributeType::Username),
            0x0008 => Some(StunAttributeType::MessageIntegrity),
            0x0009 => Some(StunAttributeType::ErrorCode),
            0x0014 => Some(StunAttributeType::Realm),
            0x0015 => Some(StunAttributeType::Nonce),
            0x8022 => Some(StunAttributeType::Software),
            0x8028 => Some(StunAttributeType::Fingerprint),
            0x0003 => Some(StunAttributeType::ChangeRequest),
            0x802B => Some(StunAttributeType::ResponseOrigin),
            0x802C => Some(StunAttributeType::OtherAddress),
            _ => None,
        }
    }
}

/// A single STUN attribute.
#[derive(Debug, Clone)]
pub struct StunAttribute {
    pub attr_type: StunAttributeType,
    pub data: Vec<u8>,
}

impl StunAttribute {
    pub fn new(attr_type: StunAttributeType, data: Vec<u8>) -> Self {
        Self { attr_type, data }
    }

    /// Create an XOR-MAPPED-ADDRESS attribute.
    pub fn xor_mapped_address(addr: SocketAddr, transaction_id: &[u8; 12]) -> Self {
        let magic_cookie: u32 = 0x2112A442;
        let mut data = Vec::new();

        data.push(0); // Reserved.
        match addr {
            SocketAddr::V4(v4) => {
                data.push(0x01); // IPv4 family.
                let port = v4.port() ^ (magic_cookie >> 16) as u16;
                data.extend_from_slice(&port.to_be_bytes());
                let ip_bytes = v4.ip().octets();
                let magic_bytes = magic_cookie.to_be_bytes();
                for i in 0..4 {
                    data.push(ip_bytes[i] ^ magic_bytes[i]);
                }
            }
            SocketAddr::V6(v6) => {
                data.push(0x02); // IPv6 family.
                let port = v6.port() ^ (magic_cookie >> 16) as u16;
                data.extend_from_slice(&port.to_be_bytes());
                let ip_bytes = v6.ip().octets();
                let mut xor_key = [0u8; 16];
                xor_key[0..4].copy_from_slice(&magic_cookie.to_be_bytes());
                xor_key[4..16].copy_from_slice(transaction_id);
                for i in 0..16 {
                    data.push(ip_bytes[i] ^ xor_key[i]);
                }
            }
        }

        Self {
            attr_type: StunAttributeType::XorMappedAddress,
            data,
        }
    }

    /// Parse an XOR-MAPPED-ADDRESS to get the mapped address.
    pub fn parse_xor_mapped_address(
        &self,
        transaction_id: &[u8; 12],
    ) -> Option<SocketAddr> {
        let magic_cookie: u32 = 0x2112A442;
        if self.data.len() < 8 {
            return None;
        }

        let family = self.data[1];
        let xor_port =
            u16::from_be_bytes([self.data[2], self.data[3]]);
        let port = xor_port ^ (magic_cookie >> 16) as u16;

        match family {
            0x01 => {
                // IPv4
                if self.data.len() < 8 {
                    return None;
                }
                let magic_bytes = magic_cookie.to_be_bytes();
                let ip = [
                    self.data[4] ^ magic_bytes[0],
                    self.data[5] ^ magic_bytes[1],
                    self.data[6] ^ magic_bytes[2],
                    self.data[7] ^ magic_bytes[3],
                ];
                let addr = std::net::Ipv4Addr::new(ip[0], ip[1], ip[2], ip[3]);
                Some(SocketAddr::new(std::net::IpAddr::V4(addr), port))
            }
            0x02 => {
                // IPv6
                if self.data.len() < 20 {
                    return None;
                }
                let mut xor_key = [0u8; 16];
                xor_key[0..4].copy_from_slice(&magic_cookie.to_be_bytes());
                xor_key[4..16].copy_from_slice(transaction_id);
                let mut ip_bytes = [0u8; 16];
                for i in 0..16 {
                    ip_bytes[i] = self.data[4 + i] ^ xor_key[i];
                }
                let addr = std::net::Ipv6Addr::from(ip_bytes);
                Some(SocketAddr::new(std::net::IpAddr::V6(addr), port))
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// STUN Message
// ---------------------------------------------------------------------------

/// STUN magic cookie constant.
pub const STUN_MAGIC_COOKIE: u32 = 0x2112A442;

/// A STUN protocol message.
#[derive(Debug, Clone)]
pub struct StunMessage {
    pub msg_type: StunMessageType,
    pub transaction_id: [u8; 12],
    pub attributes: Vec<StunAttribute>,
}

impl StunMessage {
    /// Create a new binding request with a random transaction ID.
    pub fn binding_request() -> Self {
        let mut tid = [0u8; 12];
        // Simple pseudo-random fill.
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        for (i, byte) in tid.iter_mut().enumerate() {
            *byte = ((seed >> (i * 8)) & 0xFF) as u8;
        }
        Self {
            msg_type: StunMessageType::binding_request(),
            transaction_id: tid,
            attributes: Vec::new(),
        }
    }

    /// Create a binding success response.
    pub fn binding_response(
        transaction_id: [u8; 12],
        mapped_addr: SocketAddr,
    ) -> Self {
        let attr = StunAttribute::xor_mapped_address(mapped_addr, &transaction_id);
        Self {
            msg_type: StunMessageType::binding_response(),
            transaction_id,
            attributes: vec![attr],
        }
    }

    /// Add an attribute.
    pub fn add_attribute(&mut self, attr: StunAttribute) {
        self.attributes.push(attr);
    }

    /// Find an attribute by type.
    pub fn find_attribute(&self, attr_type: StunAttributeType) -> Option<&StunAttribute> {
        self.attributes.iter().find(|a| a.attr_type == attr_type)
    }

    /// Extract the XOR-MAPPED-ADDRESS from a binding response.
    pub fn mapped_address(&self) -> Option<SocketAddr> {
        self.find_attribute(StunAttributeType::XorMappedAddress)
            .and_then(|attr| attr.parse_xor_mapped_address(&self.transaction_id))
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Compute total attribute length.
        let attr_len: usize = self
            .attributes
            .iter()
            .map(|a| 4 + a.data.len() + (4 - a.data.len() % 4) % 4)
            .sum();

        // Header: type (2), length (2), magic cookie (4), transaction ID (12).
        buf.extend_from_slice(&self.msg_type.to_u16().to_be_bytes());
        buf.extend_from_slice(&(attr_len as u16).to_be_bytes());
        buf.extend_from_slice(&STUN_MAGIC_COOKIE.to_be_bytes());
        buf.extend_from_slice(&self.transaction_id);

        // Attributes.
        for attr in &self.attributes {
            buf.extend_from_slice(&attr.attr_type.to_u16().to_be_bytes());
            buf.extend_from_slice(&(attr.data.len() as u16).to_be_bytes());
            buf.extend_from_slice(&attr.data);
            // Padding to 4-byte boundary.
            let pad = (4 - attr.data.len() % 4) % 4;
            for _ in 0..pad {
                buf.push(0);
            }
        }

        buf
    }

    /// Parse from bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 20 {
            return None;
        }

        let msg_type_raw = u16::from_be_bytes([data[0], data[1]]);
        let msg_type = StunMessageType::from_u16(msg_type_raw)?;
        let _length = u16::from_be_bytes([data[2], data[3]]) as usize;
        let cookie = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        if cookie != STUN_MAGIC_COOKIE {
            return None;
        }

        let mut tid = [0u8; 12];
        tid.copy_from_slice(&data[8..20]);

        // Parse attributes.
        let mut attributes = Vec::new();
        let mut offset = 20;
        while offset + 4 <= data.len() {
            let attr_type_raw = u16::from_be_bytes([data[offset], data[offset + 1]]);
            let attr_len =
                u16::from_be_bytes([data[offset + 2], data[offset + 3]]) as usize;
            offset += 4;

            if offset + attr_len > data.len() {
                break;
            }

            let attr_data = data[offset..offset + attr_len].to_vec();
            if let Some(attr_type) = StunAttributeType::from_u16(attr_type_raw) {
                attributes.push(StunAttribute {
                    attr_type,
                    data: attr_data,
                });
            }

            offset += attr_len;
            offset += (4 - attr_len % 4) % 4; // Padding.
        }

        Some(Self {
            msg_type,
            transaction_id: tid,
            attributes,
        })
    }
}

// ---------------------------------------------------------------------------
// NAT Type
// ---------------------------------------------------------------------------

/// Detected NAT type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NatType {
    /// No NAT (public IP).
    NoNat,
    /// Full cone NAT: any external host can send to the mapped port.
    FullCone,
    /// Restricted cone: only hosts we've sent to can reply (any port).
    RestrictedCone,
    /// Port-restricted cone: only the specific IP:port we sent to can reply.
    PortRestricted,
    /// Symmetric NAT: different mapping for each destination.
    Symmetric,
    /// Unable to determine.
    Unknown,
    /// UDP is blocked.
    UdpBlocked,
}

impl NatType {
    /// Whether direct hole punching is likely to succeed.
    pub fn can_hole_punch(self) -> bool {
        matches!(
            self,
            NatType::NoNat
                | NatType::FullCone
                | NatType::RestrictedCone
                | NatType::PortRestricted
        )
    }

    /// Whether a relay is recommended.
    pub fn needs_relay(self) -> bool {
        matches!(self, NatType::Symmetric | NatType::UdpBlocked | NatType::Unknown)
    }
}

impl fmt::Display for NatType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NatType::NoNat => write!(f, "No NAT (Public)"),
            NatType::FullCone => write!(f, "Full Cone"),
            NatType::RestrictedCone => write!(f, "Restricted Cone"),
            NatType::PortRestricted => write!(f, "Port-Restricted Cone"),
            NatType::Symmetric => write!(f, "Symmetric"),
            NatType::Unknown => write!(f, "Unknown"),
            NatType::UdpBlocked => write!(f, "UDP Blocked"),
        }
    }
}

// ---------------------------------------------------------------------------
// NAT Detection Result
// ---------------------------------------------------------------------------

/// Result of a NAT type detection test.
#[derive(Debug, Clone)]
pub struct NatDetectionResult {
    /// Detected NAT type.
    pub nat_type: NatType,
    /// Public (mapped) address as seen by the first STUN server.
    pub public_addr: Option<SocketAddr>,
    /// Local (source) address.
    pub local_addr: Option<SocketAddr>,
    /// Public address as seen by the second STUN server (for symmetric check).
    pub alt_public_addr: Option<SocketAddr>,
    /// Whether hairpin routing is supported.
    pub hairpin_supported: bool,
    /// RTT to the STUN server.
    pub rtt: Option<Duration>,
    /// The STUN servers used for detection.
    pub servers_used: Vec<String>,
    /// Any errors encountered during detection.
    pub errors: Vec<String>,
}

impl NatDetectionResult {
    pub fn new(nat_type: NatType) -> Self {
        Self {
            nat_type,
            public_addr: None,
            local_addr: None,
            alt_public_addr: None,
            hairpin_supported: false,
            rtt: None,
            servers_used: Vec::new(),
            errors: Vec::new(),
        }
    }
}

impl fmt::Display for NatDetectionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "NAT Detection Result:")?;
        writeln!(f, "  type:    {}", self.nat_type)?;
        if let Some(addr) = self.public_addr {
            writeln!(f, "  public:  {}", addr)?;
        }
        if let Some(addr) = self.local_addr {
            writeln!(f, "  local:   {}", addr)?;
        }
        if let Some(rtt) = self.rtt {
            writeln!(f, "  RTT:     {:.1}ms", rtt.as_secs_f64() * 1000.0)?;
        }
        writeln!(f, "  hairpin: {}", self.hairpin_supported)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ICE Candidate
// ---------------------------------------------------------------------------

/// ICE candidate type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateType {
    /// Host candidate (local address).
    Host,
    /// Server-reflexive (STUN-discovered public address).
    ServerReflexive,
    /// Peer-reflexive (discovered during connectivity checks).
    PeerReflexive,
    /// Relay candidate (TURN-allocated address).
    Relay,
}

impl CandidateType {
    /// Default priority for this candidate type (higher is preferred).
    pub fn default_priority(self) -> u32 {
        match self {
            CandidateType::Host => 126,
            CandidateType::ServerReflexive => 100,
            CandidateType::PeerReflexive => 110,
            CandidateType::Relay => 0,
        }
    }
}

impl fmt::Display for CandidateType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CandidateType::Host => write!(f, "host"),
            CandidateType::ServerReflexive => write!(f, "srflx"),
            CandidateType::PeerReflexive => write!(f, "prflx"),
            CandidateType::Relay => write!(f, "relay"),
        }
    }
}

/// An ICE candidate for connectivity establishment.
#[derive(Debug, Clone)]
pub struct IceCandidate {
    /// Candidate foundation (identifier).
    pub foundation: String,
    /// Component ID (1 = RTP, 2 = RTCP in standard ICE).
    pub component: u32,
    /// Transport protocol ("udp" or "tcp").
    pub transport: String,
    /// Priority.
    pub priority: u32,
    /// Address.
    pub addr: SocketAddr,
    /// Candidate type.
    pub candidate_type: CandidateType,
    /// Related address (for reflexive/relay candidates).
    pub related_addr: Option<SocketAddr>,
}

impl IceCandidate {
    /// Create a host candidate.
    pub fn host(addr: SocketAddr) -> Self {
        Self {
            foundation: format!("host-{}", addr),
            component: 1,
            transport: "udp".to_string(),
            priority: Self::compute_priority(CandidateType::Host, 1, 1),
            addr,
            candidate_type: CandidateType::Host,
            related_addr: None,
        }
    }

    /// Create a server-reflexive candidate.
    pub fn server_reflexive(public_addr: SocketAddr, local_addr: SocketAddr) -> Self {
        Self {
            foundation: format!("srflx-{}", public_addr),
            component: 1,
            transport: "udp".to_string(),
            priority: Self::compute_priority(CandidateType::ServerReflexive, 1, 1),
            addr: public_addr,
            candidate_type: CandidateType::ServerReflexive,
            related_addr: Some(local_addr),
        }
    }

    /// Create a relay candidate.
    pub fn relay(relay_addr: SocketAddr, local_addr: SocketAddr) -> Self {
        Self {
            foundation: format!("relay-{}", relay_addr),
            component: 1,
            transport: "udp".to_string(),
            priority: Self::compute_priority(CandidateType::Relay, 1, 1),
            addr: relay_addr,
            candidate_type: CandidateType::Relay,
            related_addr: Some(local_addr),
        }
    }

    /// Compute ICE priority per RFC 8445.
    pub fn compute_priority(
        candidate_type: CandidateType,
        component: u32,
        local_pref: u32,
    ) -> u32 {
        let type_pref = candidate_type.default_priority();
        (type_pref << 24) | (local_pref << 8) | (256 - component)
    }

    /// Format as SDP attribute line.
    pub fn to_sdp(&self) -> String {
        let mut s = format!(
            "a=candidate:{} {} {} {} {} {} typ {}",
            self.foundation,
            self.component,
            self.transport,
            self.priority,
            self.addr.ip(),
            self.addr.port(),
            self.candidate_type,
        );
        if let Some(raddr) = self.related_addr {
            s.push_str(&format!(" raddr {} rport {}", raddr.ip(), raddr.port()));
        }
        s
    }
}

impl fmt::Display for IceCandidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_sdp())
    }
}

// ---------------------------------------------------------------------------
// Hole Punch Coordinator
// ---------------------------------------------------------------------------

/// State of a hole-punch attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HolePunchState {
    /// Waiting for both peers to register.
    WaitingForPeers,
    /// Exchanging candidates.
    ExchangingCandidates,
    /// Punching (sending probes).
    Punching,
    /// Connected (hole punch succeeded).
    Connected,
    /// Failed (timed out or NAT incompatible).
    Failed,
}

impl fmt::Display for HolePunchState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HolePunchState::WaitingForPeers => write!(f, "WaitingForPeers"),
            HolePunchState::ExchangingCandidates => write!(f, "ExchangingCandidates"),
            HolePunchState::Punching => write!(f, "Punching"),
            HolePunchState::Connected => write!(f, "Connected"),
            HolePunchState::Failed => write!(f, "Failed"),
        }
    }
}

/// A hole-punch session between two peers.
#[derive(Debug, Clone)]
pub struct HolePunchSession {
    /// Session identifier.
    pub session_id: u64,
    /// Peer A identifier.
    pub peer_a: String,
    /// Peer B identifier.
    pub peer_b: String,
    /// Peer A's candidates.
    pub candidates_a: Vec<IceCandidate>,
    /// Peer B's candidates.
    pub candidates_b: Vec<IceCandidate>,
    /// Current state.
    pub state: HolePunchState,
    /// Number of probe packets sent.
    pub probes_sent: u32,
    /// Maximum probes before giving up.
    pub max_probes: u32,
    /// Interval between probes.
    pub probe_interval: Duration,
    /// When the session was created.
    pub created_at: Instant,
    /// Timeout duration.
    pub timeout: Duration,
    /// Successfully connected addresses (if any).
    pub connected_addr_a: Option<SocketAddr>,
    pub connected_addr_b: Option<SocketAddr>,
}

impl HolePunchSession {
    /// Create a new hole-punch session.
    pub fn new(session_id: u64, peer_a: &str, peer_b: &str) -> Self {
        Self {
            session_id,
            peer_a: peer_a.to_string(),
            peer_b: peer_b.to_string(),
            candidates_a: Vec::new(),
            candidates_b: Vec::new(),
            state: HolePunchState::WaitingForPeers,
            probes_sent: 0,
            max_probes: 30,
            probe_interval: Duration::from_millis(100),
            created_at: Instant::now(),
            timeout: Duration::from_secs(30),
            connected_addr_a: None,
            connected_addr_b: None,
        }
    }

    /// Submit candidates for peer A.
    pub fn set_candidates_a(&mut self, candidates: Vec<IceCandidate>) {
        self.candidates_a = candidates;
        self.check_ready();
    }

    /// Submit candidates for peer B.
    pub fn set_candidates_b(&mut self, candidates: Vec<IceCandidate>) {
        self.candidates_b = candidates;
        self.check_ready();
    }

    /// Check if both peers have submitted candidates.
    fn check_ready(&mut self) {
        if !self.candidates_a.is_empty()
            && !self.candidates_b.is_empty()
            && self.state == HolePunchState::WaitingForPeers
        {
            self.state = HolePunchState::ExchangingCandidates;
        }
    }

    /// Check if the session has timed out.
    pub fn is_timed_out(&self) -> bool {
        self.created_at.elapsed() >= self.timeout
    }

    /// Get candidate pairs sorted by priority.
    pub fn candidate_pairs(&self) -> Vec<(usize, usize, u64)> {
        let mut pairs = Vec::new();
        for (i, ca) in self.candidates_a.iter().enumerate() {
            for (j, cb) in self.candidates_b.iter().enumerate() {
                let pair_priority =
                    ca.priority as u64 * cb.priority as u64;
                pairs.push((i, j, pair_priority));
            }
        }
        pairs.sort_by(|a, b| b.2.cmp(&a.2));
        pairs
    }

    /// Mark the connection as established.
    pub fn set_connected(&mut self, addr_a: SocketAddr, addr_b: SocketAddr) {
        self.state = HolePunchState::Connected;
        self.connected_addr_a = Some(addr_a);
        self.connected_addr_b = Some(addr_b);
    }

    /// Mark as failed.
    pub fn set_failed(&mut self) {
        self.state = HolePunchState::Failed;
    }

    /// Age of this session.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Coordinator that manages hole-punch sessions.
pub struct HolePunchCoordinator {
    sessions: HashMap<u64, HolePunchSession>,
    next_session_id: u64,
}

impl HolePunchCoordinator {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            next_session_id: 1,
        }
    }

    /// Create a new session.
    pub fn create_session(&mut self, peer_a: &str, peer_b: &str) -> u64 {
        let id = self.next_session_id;
        self.next_session_id += 1;
        let session = HolePunchSession::new(id, peer_a, peer_b);
        self.sessions.insert(id, session);
        id
    }

    /// Get a session by ID.
    pub fn get_session(&self, id: u64) -> Option<&HolePunchSession> {
        self.sessions.get(&id)
    }

    /// Get a mutable session.
    pub fn get_session_mut(&mut self, id: u64) -> Option<&mut HolePunchSession> {
        self.sessions.get_mut(&id)
    }

    /// Remove completed/failed sessions.
    pub fn cleanup(&mut self) {
        self.sessions.retain(|_, s| {
            !matches!(s.state, HolePunchState::Connected | HolePunchState::Failed)
                && !s.is_timed_out()
        });
    }

    /// Number of active sessions.
    pub fn active_session_count(&self) -> usize {
        self.sessions.len()
    }
}

impl Default for HolePunchCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stun_message_roundtrip() {
        let msg = StunMessage::binding_request();
        let bytes = msg.to_bytes();
        let parsed = StunMessage::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.msg_type, msg.msg_type);
        assert_eq!(parsed.transaction_id, msg.transaction_id);
    }

    #[test]
    fn test_xor_mapped_address() {
        let addr: SocketAddr = "192.168.1.100:12345".parse().unwrap();
        let tid = [1u8; 12];
        let attr = StunAttribute::xor_mapped_address(addr, &tid);
        let decoded = attr.parse_xor_mapped_address(&tid).unwrap();
        assert_eq!(decoded, addr);
    }

    #[test]
    fn test_binding_response() {
        let addr: SocketAddr = "203.0.113.5:54321".parse().unwrap();
        let tid = [42u8; 12];
        let msg = StunMessage::binding_response(tid, addr);
        let mapped = msg.mapped_address().unwrap();
        assert_eq!(mapped, addr);
    }

    #[test]
    fn test_nat_type_properties() {
        assert!(NatType::FullCone.can_hole_punch());
        assert!(!NatType::Symmetric.can_hole_punch());
        assert!(NatType::Symmetric.needs_relay());
        assert!(!NatType::FullCone.needs_relay());
    }

    #[test]
    fn test_ice_candidate_priority() {
        let host = IceCandidate::host("192.168.1.1:5000".parse().unwrap());
        let srflx = IceCandidate::server_reflexive(
            "203.0.113.5:5000".parse().unwrap(),
            "192.168.1.1:5000".parse().unwrap(),
        );
        let relay = IceCandidate::relay(
            "10.0.0.1:3478".parse().unwrap(),
            "192.168.1.1:5000".parse().unwrap(),
        );

        assert!(host.priority > srflx.priority);
        assert!(srflx.priority > relay.priority);
    }

    #[test]
    fn test_hole_punch_session() {
        let mut coord = HolePunchCoordinator::new();
        let id = coord.create_session("alice", "bob");

        let session = coord.get_session_mut(id).unwrap();
        session.set_candidates_a(vec![IceCandidate::host(
            "192.168.1.1:5000".parse().unwrap(),
        )]);
        session.set_candidates_b(vec![IceCandidate::host(
            "192.168.1.2:5000".parse().unwrap(),
        )]);

        assert_eq!(session.state, HolePunchState::ExchangingCandidates);
        assert_eq!(session.candidate_pairs().len(), 1);
    }

    #[test]
    fn test_stun_message_type() {
        let mt = StunMessageType::binding_request();
        let raw = mt.to_u16();
        let decoded = StunMessageType::from_u16(raw).unwrap();
        assert_eq!(decoded.class, StunClass::Request);
        assert_eq!(decoded.method, StunMethod::Binding);
    }
}
