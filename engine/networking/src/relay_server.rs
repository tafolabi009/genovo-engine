//! Relay/TURN server for NAT traversal in the Genovo engine.
//!
//! Relays packets between clients that cannot establish a direct peer-to-peer
//! connection due to NAT restrictions. Manages allocations, client
//! authentication, bandwidth throttling, relay statistics, and connection
//! pairing.
//!
//! # Architecture
//!
//! ```text
//!   Client A                Relay Server               Client B
//!   (behind NAT)                                        (behind NAT)
//!       │                       │                           │
//!       │──── Allocate ────────►│                           │
//!       │◄─── AllocationOK ────│                           │
//!       │                       │◄──── Allocate ────────────│
//!       │                       │───── AllocationOK ───────►│
//!       │                       │                           │
//!       │──── Data(to B) ──────►│──── Data(from A) ────────►│
//!       │◄─── Data(from B) ────│◄──── Data(to A) ──────────│
//! ```

use std::collections::HashMap;
use std::fmt;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// RelayId / AllocationId
// ---------------------------------------------------------------------------

/// Unique identifier for a relay client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RelayClientId(u64);

impl RelayClientId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for RelayClientId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RelayClient({})", self.0)
    }
}

/// Unique identifier for a relay allocation (port reservation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AllocationId(u64);

impl AllocationId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for AllocationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Alloc({})", self.0)
    }
}

/// Unique identifier for a pairing between two clients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PairingId(u64);

impl PairingId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

impl fmt::Display for PairingId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pair({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Authentication
// ---------------------------------------------------------------------------

/// Authentication credentials for relay access.
#[derive(Debug, Clone)]
pub struct RelayCredentials {
    /// Username or client token.
    pub username: String,
    /// HMAC or password-based credential.
    pub credential: Vec<u8>,
    /// Expiration time (Unix timestamp).
    pub expires_at: u64,
    /// Realm / namespace.
    pub realm: String,
}

impl RelayCredentials {
    /// Create new credentials.
    pub fn new(username: &str, credential: &[u8], expires_at: u64) -> Self {
        Self {
            username: username.to_string(),
            credential: credential.to_vec(),
            expires_at,
            realm: "genovo".to_string(),
        }
    }

    /// Check if credentials are expired.
    pub fn is_expired(&self, current_time: u64) -> bool {
        current_time > self.expires_at
    }
}

/// Result of an authentication attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthResult {
    /// Authentication succeeded.
    Accepted,
    /// Invalid credentials.
    InvalidCredentials,
    /// Credentials have expired.
    Expired,
    /// Too many active connections.
    QuotaExceeded,
    /// Authentication server is unavailable.
    ServiceUnavailable,
}

impl fmt::Display for AuthResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AuthResult::Accepted => write!(f, "Accepted"),
            AuthResult::InvalidCredentials => write!(f, "InvalidCredentials"),
            AuthResult::Expired => write!(f, "Expired"),
            AuthResult::QuotaExceeded => write!(f, "QuotaExceeded"),
            AuthResult::ServiceUnavailable => write!(f, "ServiceUnavailable"),
        }
    }
}

/// Authentication provider trait.
pub trait RelayAuthenticator: Send + Sync {
    /// Validate credentials.
    fn authenticate(&self, credentials: &RelayCredentials) -> AuthResult;

    /// Check if a username has permission to create allocations.
    fn can_allocate(&self, username: &str) -> bool;

    /// Get the maximum allowed bandwidth for a user (bytes/sec).
    fn max_bandwidth(&self, username: &str) -> u64;

    /// Get the maximum number of allocations for a user.
    fn max_allocations(&self, username: &str) -> usize;
}

/// Simple in-memory authenticator for testing.
pub struct SimpleAuthenticator {
    /// Registered users: username -> credential bytes.
    users: HashMap<String, Vec<u8>>,
    /// Default bandwidth limit.
    default_bandwidth: u64,
    /// Default max allocations.
    default_max_allocations: usize,
}

impl SimpleAuthenticator {
    pub fn new() -> Self {
        Self {
            users: HashMap::new(),
            default_bandwidth: 1024 * 1024, // 1 MB/s
            default_max_allocations: 5,
        }
    }

    pub fn add_user(&mut self, username: &str, credential: &[u8]) {
        self.users.insert(username.to_string(), credential.to_vec());
    }

    pub fn set_bandwidth_limit(&mut self, limit: u64) {
        self.default_bandwidth = limit;
    }
}

impl Default for SimpleAuthenticator {
    fn default() -> Self {
        Self::new()
    }
}

impl RelayAuthenticator for SimpleAuthenticator {
    fn authenticate(&self, credentials: &RelayCredentials) -> AuthResult {
        match self.users.get(&credentials.username) {
            Some(expected) => {
                if credentials.credential == *expected {
                    AuthResult::Accepted
                } else {
                    AuthResult::InvalidCredentials
                }
            }
            None => AuthResult::InvalidCredentials,
        }
    }

    fn can_allocate(&self, username: &str) -> bool {
        self.users.contains_key(username)
    }

    fn max_bandwidth(&self, _username: &str) -> u64 {
        self.default_bandwidth
    }

    fn max_allocations(&self, _username: &str) -> usize {
        self.default_max_allocations
    }
}

// ---------------------------------------------------------------------------
// BandwidthThrottle
// ---------------------------------------------------------------------------

/// Token-bucket bandwidth throttle.
#[derive(Debug, Clone)]
pub struct BandwidthThrottle {
    /// Maximum bytes per second.
    pub rate_limit: u64,
    /// Current token count.
    tokens: f64,
    /// Maximum token burst.
    max_tokens: f64,
    /// Last time tokens were refilled.
    last_refill: Instant,
    /// Total bytes relayed.
    total_bytes: u64,
    /// Bytes relayed in current window.
    window_bytes: u64,
    /// Window start time.
    window_start: Instant,
}

impl BandwidthThrottle {
    /// Create a new throttle with the given rate limit (bytes/sec).
    pub fn new(rate_limit: u64) -> Self {
        Self {
            rate_limit,
            tokens: rate_limit as f64,
            max_tokens: (rate_limit * 2) as f64,
            last_refill: Instant::now(),
            total_bytes: 0,
            window_bytes: 0,
            window_start: Instant::now(),
        }
    }

    /// Refill tokens based on elapsed time.
    pub fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens += elapsed * self.rate_limit as f64;
        if self.tokens > self.max_tokens {
            self.tokens = self.max_tokens;
        }
        self.last_refill = now;

        // Reset window every second.
        if now.duration_since(self.window_start).as_secs() >= 1 {
            self.window_bytes = 0;
            self.window_start = now;
        }
    }

    /// Try to consume `bytes` tokens. Returns `true` if allowed.
    pub fn try_consume(&mut self, bytes: u64) -> bool {
        self.refill();
        let b = bytes as f64;
        if self.tokens >= b {
            self.tokens -= b;
            self.total_bytes += bytes;
            self.window_bytes += bytes;
            true
        } else {
            false
        }
    }

    /// Current available tokens.
    pub fn available(&self) -> u64 {
        self.tokens as u64
    }

    /// Total bytes relayed through this throttle.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Current throughput (bytes/sec in current window).
    pub fn current_throughput(&self) -> f64 {
        let elapsed = self.window_start.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.window_bytes as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Utilization ratio (0.0 to 1.0).
    pub fn utilization(&self) -> f64 {
        (self.current_throughput() / self.rate_limit as f64).min(1.0)
    }
}

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

/// A relay allocation: a reserved relay port for a specific client.
#[derive(Debug, Clone)]
pub struct Allocation {
    /// Unique allocation ID.
    pub id: AllocationId,
    /// Client that owns this allocation.
    pub client_id: RelayClientId,
    /// Client's reported address.
    pub client_addr: SocketAddr,
    /// Allocated relay address.
    pub relay_addr: SocketAddr,
    /// When the allocation was created.
    pub created_at: Instant,
    /// When the allocation expires.
    pub expires_at: Instant,
    /// Bandwidth throttle for this allocation.
    pub throttle: BandwidthThrottle,
    /// Total packets relayed through this allocation.
    pub packets_relayed: u64,
    /// Total bytes relayed.
    pub bytes_relayed: u64,
    /// Whether the allocation is active.
    pub active: bool,
    /// Username of the owner.
    pub username: String,
}

impl Allocation {
    /// Create a new allocation.
    pub fn new(
        id: AllocationId,
        client_id: RelayClientId,
        client_addr: SocketAddr,
        relay_addr: SocketAddr,
        lifetime: Duration,
        bandwidth_limit: u64,
        username: String,
    ) -> Self {
        let now = Instant::now();
        Self {
            id,
            client_id,
            client_addr,
            relay_addr,
            created_at: now,
            expires_at: now + lifetime,
            throttle: BandwidthThrottle::new(bandwidth_limit),
            packets_relayed: 0,
            bytes_relayed: 0,
            active: true,
            username,
        }
    }

    /// Check if the allocation has expired.
    pub fn is_expired(&self) -> bool {
        Instant::now() >= self.expires_at
    }

    /// Refresh the allocation's lifetime.
    pub fn refresh(&mut self, lifetime: Duration) {
        self.expires_at = Instant::now() + lifetime;
    }

    /// Time remaining before expiration.
    pub fn time_remaining(&self) -> Duration {
        self.expires_at
            .checked_duration_since(Instant::now())
            .unwrap_or(Duration::ZERO)
    }

    /// Record a relayed packet.
    pub fn record_relay(&mut self, bytes: u64) {
        self.packets_relayed += 1;
        self.bytes_relayed += bytes;
    }
}

// ---------------------------------------------------------------------------
// ConnectionPairing
// ---------------------------------------------------------------------------

/// A pairing between two relay clients.
#[derive(Debug, Clone)]
pub struct ConnectionPairing {
    /// Unique pairing ID.
    pub id: PairingId,
    /// First client.
    pub client_a: RelayClientId,
    /// Second client.
    pub client_b: RelayClientId,
    /// Allocation for client A.
    pub allocation_a: AllocationId,
    /// Allocation for client B.
    pub allocation_b: AllocationId,
    /// When the pairing was established.
    pub created_at: Instant,
    /// Whether both clients have confirmed.
    pub confirmed: bool,
    /// Packets relayed A -> B.
    pub packets_a_to_b: u64,
    /// Packets relayed B -> A.
    pub packets_b_to_a: u64,
    /// Bytes relayed A -> B.
    pub bytes_a_to_b: u64,
    /// Bytes relayed B -> A.
    pub bytes_b_to_a: u64,
}

impl ConnectionPairing {
    pub fn new(
        id: PairingId,
        client_a: RelayClientId,
        client_b: RelayClientId,
        allocation_a: AllocationId,
        allocation_b: AllocationId,
    ) -> Self {
        Self {
            id,
            client_a,
            client_b,
            allocation_a,
            allocation_b,
            created_at: Instant::now(),
            confirmed: false,
            packets_a_to_b: 0,
            packets_b_to_a: 0,
            bytes_a_to_b: 0,
            bytes_b_to_a: 0,
        }
    }

    /// Total packets relayed in this pairing.
    pub fn total_packets(&self) -> u64 {
        self.packets_a_to_b + self.packets_b_to_a
    }

    /// Total bytes relayed.
    pub fn total_bytes(&self) -> u64 {
        self.bytes_a_to_b + self.bytes_b_to_a
    }

    /// Record a relay from A to B.
    pub fn record_a_to_b(&mut self, bytes: u64) {
        self.packets_a_to_b += 1;
        self.bytes_a_to_b += bytes;
    }

    /// Record a relay from B to A.
    pub fn record_b_to_a(&mut self, bytes: u64) {
        self.packets_b_to_a += 1;
        self.bytes_b_to_a += bytes;
    }

    /// Elapsed time since creation.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

// ---------------------------------------------------------------------------
// Relay Message
// ---------------------------------------------------------------------------

/// Messages exchanged with the relay server.
#[derive(Debug, Clone)]
pub enum RelayMessage {
    /// Request a relay allocation.
    AllocateRequest {
        credentials: RelayCredentials,
        lifetime_secs: u32,
    },
    /// Allocation response.
    AllocateResponse {
        result: AllocateResult,
    },
    /// Refresh an existing allocation's lifetime.
    RefreshRequest {
        allocation_id: AllocationId,
        lifetime_secs: u32,
    },
    /// Refresh response.
    RefreshResponse {
        success: bool,
    },
    /// Request to pair with another client.
    PairRequest {
        target_client: RelayClientId,
    },
    /// Pairing response.
    PairResponse {
        result: PairResult,
    },
    /// Data packet to relay.
    RelayData {
        allocation_id: AllocationId,
        target: RelayClientId,
        payload: Vec<u8>,
    },
    /// Data received from relay.
    RelayedData {
        source: RelayClientId,
        payload: Vec<u8>,
    },
    /// Keepalive / ping.
    Ping {
        timestamp: u64,
    },
    /// Pong response.
    Pong {
        timestamp: u64,
    },
    /// Disconnect from the relay.
    Disconnect,
}

/// Result of an allocation request.
#[derive(Debug, Clone)]
pub enum AllocateResult {
    /// Allocation succeeded.
    Success {
        allocation_id: AllocationId,
        relay_addr: SocketAddr,
        lifetime_secs: u32,
    },
    /// Authentication failed.
    AuthFailed(AuthResult),
    /// Quota exceeded.
    QuotaExceeded,
    /// Server error.
    ServerError(String),
}

/// Result of a pairing request.
#[derive(Debug, Clone)]
pub enum PairResult {
    /// Pairing succeeded.
    Success {
        pairing_id: PairingId,
        peer_addr: SocketAddr,
    },
    /// Target client not found.
    PeerNotFound,
    /// Pairing denied.
    Denied(String),
}

// ---------------------------------------------------------------------------
// RelayStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the relay server.
#[derive(Debug, Clone, Default)]
pub struct RelayStats {
    /// Total allocations ever created.
    pub total_allocations: u64,
    /// Currently active allocations.
    pub active_allocations: usize,
    /// Total pairings ever created.
    pub total_pairings: u64,
    /// Currently active pairings.
    pub active_pairings: usize,
    /// Total packets relayed.
    pub total_packets_relayed: u64,
    /// Total bytes relayed.
    pub total_bytes_relayed: u64,
    /// Total authentication attempts.
    pub auth_attempts: u64,
    /// Successful authentication attempts.
    pub auth_successes: u64,
    /// Failed authentication attempts.
    pub auth_failures: u64,
    /// Total allocations expired.
    pub allocations_expired: u64,
    /// Total allocations explicitly closed.
    pub allocations_closed: u64,
    /// Peak concurrent allocations.
    pub peak_allocations: usize,
    /// Peak concurrent pairings.
    pub peak_pairings: usize,
    /// Current aggregate bandwidth usage (bytes/sec).
    pub current_bandwidth: f64,
}

impl fmt::Display for RelayStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Relay Server Statistics:")?;
        writeln!(f, "  allocations: {} active / {} total", self.active_allocations, self.total_allocations)?;
        writeln!(f, "  pairings:    {} active / {} total", self.active_pairings, self.total_pairings)?;
        writeln!(f, "  packets:     {}", self.total_packets_relayed)?;
        writeln!(f, "  bytes:       {} MB", self.total_bytes_relayed / (1024 * 1024))?;
        writeln!(f, "  auth:        {} ok / {} fail", self.auth_successes, self.auth_failures)?;
        writeln!(f, "  bandwidth:   {:.1} KB/s", self.current_bandwidth / 1024.0)?;
        writeln!(f, "  peak alloc:  {}", self.peak_allocations)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RelayServerConfig
// ---------------------------------------------------------------------------

/// Configuration for the relay server.
#[derive(Debug, Clone)]
pub struct RelayServerConfig {
    /// Maximum concurrent allocations.
    pub max_allocations: usize,
    /// Maximum concurrent pairings.
    pub max_pairings: usize,
    /// Default allocation lifetime.
    pub default_lifetime: Duration,
    /// Maximum allocation lifetime.
    pub max_lifetime: Duration,
    /// Default bandwidth limit per allocation (bytes/sec).
    pub default_bandwidth: u64,
    /// Maximum packet size.
    pub max_packet_size: usize,
    /// Keepalive interval.
    pub keepalive_interval: Duration,
    /// Cleanup interval for expired allocations.
    pub cleanup_interval: Duration,
    /// Bind address for the relay server.
    pub bind_addr: SocketAddr,
}

impl Default for RelayServerConfig {
    fn default() -> Self {
        Self {
            max_allocations: 1000,
            max_pairings: 500,
            default_lifetime: Duration::from_secs(600),
            max_lifetime: Duration::from_secs(3600),
            default_bandwidth: 256 * 1024, // 256 KB/s
            max_packet_size: 65536,
            keepalive_interval: Duration::from_secs(30),
            cleanup_interval: Duration::from_secs(60),
            bind_addr: "0.0.0.0:3478".parse().unwrap(),
        }
    }
}

// ---------------------------------------------------------------------------
// RelayServer
// ---------------------------------------------------------------------------

/// The relay server that manages allocations, pairings, and packet forwarding.
pub struct RelayServer {
    /// Server configuration.
    config: RelayServerConfig,
    /// Active allocations keyed by ID.
    allocations: HashMap<AllocationId, Allocation>,
    /// Active pairings keyed by ID.
    pairings: HashMap<PairingId, ConnectionPairing>,
    /// Client -> allocation mapping.
    client_allocations: HashMap<RelayClientId, Vec<AllocationId>>,
    /// Authenticator.
    authenticator: Box<dyn RelayAuthenticator>,
    /// Next allocation ID.
    next_alloc_id: u64,
    /// Next pairing ID.
    next_pair_id: u64,
    /// Statistics.
    stats: RelayStats,
    /// Last cleanup time.
    last_cleanup: Instant,
}

impl RelayServer {
    /// Create a new relay server with the given configuration and authenticator.
    pub fn new(config: RelayServerConfig, authenticator: Box<dyn RelayAuthenticator>) -> Self {
        Self {
            config,
            allocations: HashMap::new(),
            pairings: HashMap::new(),
            client_allocations: HashMap::new(),
            authenticator,
            next_alloc_id: 1,
            next_pair_id: 1,
            stats: RelayStats::default(),
            last_cleanup: Instant::now(),
        }
    }

    /// Create with default configuration and a simple authenticator.
    pub fn with_simple_auth() -> Self {
        Self::new(
            RelayServerConfig::default(),
            Box::new(SimpleAuthenticator::new()),
        )
    }

    /// Handle an allocation request.
    pub fn handle_allocate(
        &mut self,
        client_id: RelayClientId,
        client_addr: SocketAddr,
        credentials: &RelayCredentials,
        lifetime_secs: u32,
    ) -> AllocateResult {
        self.stats.auth_attempts += 1;

        // Authenticate.
        let auth_result = self.authenticator.authenticate(credentials);
        if auth_result != AuthResult::Accepted {
            self.stats.auth_failures += 1;
            return AllocateResult::AuthFailed(auth_result);
        }
        self.stats.auth_successes += 1;

        // Check quota.
        let current_count = self
            .client_allocations
            .get(&client_id)
            .map_or(0, |v| v.len());
        let max_alloc = self.authenticator.max_allocations(&credentials.username);
        if current_count >= max_alloc {
            return AllocateResult::QuotaExceeded;
        }

        // Check server capacity.
        if self.allocations.len() >= self.config.max_allocations {
            return AllocateResult::QuotaExceeded;
        }

        // Create allocation.
        let lifetime = Duration::from_secs(lifetime_secs as u64)
            .min(self.config.max_lifetime);
        let alloc_id = AllocationId::new(self.next_alloc_id);
        self.next_alloc_id += 1;

        // Compute a relay address (in a real implementation, this binds a port).
        let relay_port = 49152 + (alloc_id.raw() % 16384) as u16;
        let relay_addr: SocketAddr = format!("0.0.0.0:{}", relay_port).parse().unwrap();

        let bandwidth = self.authenticator.max_bandwidth(&credentials.username);
        let allocation = Allocation::new(
            alloc_id,
            client_id,
            client_addr,
            relay_addr,
            lifetime,
            bandwidth,
            credentials.username.clone(),
        );

        self.allocations.insert(alloc_id, allocation);
        self.client_allocations
            .entry(client_id)
            .or_default()
            .push(alloc_id);

        self.stats.total_allocations += 1;
        self.stats.active_allocations = self.allocations.len();
        if self.stats.active_allocations > self.stats.peak_allocations {
            self.stats.peak_allocations = self.stats.active_allocations;
        }

        AllocateResult::Success {
            allocation_id: alloc_id,
            relay_addr,
            lifetime_secs: lifetime.as_secs() as u32,
        }
    }

    /// Refresh an allocation's lifetime.
    pub fn handle_refresh(
        &mut self,
        allocation_id: AllocationId,
        lifetime_secs: u32,
    ) -> bool {
        if let Some(alloc) = self.allocations.get_mut(&allocation_id) {
            let lifetime = Duration::from_secs(lifetime_secs as u64)
                .min(self.config.max_lifetime);
            alloc.refresh(lifetime);
            true
        } else {
            false
        }
    }

    /// Relay data from one client to another.
    pub fn handle_relay_data(
        &mut self,
        from_allocation: AllocationId,
        target: RelayClientId,
        payload: &[u8],
    ) -> Result<Vec<u8>, RelayError> {
        // Validate source allocation.
        let alloc = self
            .allocations
            .get_mut(&from_allocation)
            .ok_or(RelayError::InvalidAllocation)?;

        if !alloc.active || alloc.is_expired() {
            return Err(RelayError::AllocationExpired);
        }

        // Check bandwidth.
        if !alloc.throttle.try_consume(payload.len() as u64) {
            return Err(RelayError::BandwidthExceeded);
        }

        // Check packet size.
        if payload.len() > self.config.max_packet_size {
            return Err(RelayError::PacketTooLarge);
        }

        alloc.record_relay(payload.len() as u64);

        self.stats.total_packets_relayed += 1;
        self.stats.total_bytes_relayed += payload.len() as u64;

        // In a real implementation, this would forward to the target's socket.
        Ok(payload.to_vec())
    }

    /// Create a pairing between two clients.
    pub fn create_pairing(
        &mut self,
        client_a: RelayClientId,
        client_b: RelayClientId,
        alloc_a: AllocationId,
        alloc_b: AllocationId,
    ) -> Result<PairingId, RelayError> {
        if self.pairings.len() >= self.config.max_pairings {
            return Err(RelayError::ServerFull);
        }

        // Verify both allocations exist.
        if !self.allocations.contains_key(&alloc_a) {
            return Err(RelayError::InvalidAllocation);
        }
        if !self.allocations.contains_key(&alloc_b) {
            return Err(RelayError::InvalidAllocation);
        }

        let pair_id = PairingId::new(self.next_pair_id);
        self.next_pair_id += 1;

        let pairing = ConnectionPairing::new(pair_id, client_a, client_b, alloc_a, alloc_b);
        self.pairings.insert(pair_id, pairing);

        self.stats.total_pairings += 1;
        self.stats.active_pairings = self.pairings.len();
        if self.stats.active_pairings > self.stats.peak_pairings {
            self.stats.peak_pairings = self.stats.active_pairings;
        }

        Ok(pair_id)
    }

    /// Close an allocation.
    pub fn close_allocation(&mut self, allocation_id: AllocationId) {
        if let Some(alloc) = self.allocations.remove(&allocation_id) {
            if let Some(client_allocs) = self.client_allocations.get_mut(&alloc.client_id) {
                client_allocs.retain(|id| *id != allocation_id);
            }
            self.stats.allocations_closed += 1;
            self.stats.active_allocations = self.allocations.len();
        }
    }

    /// Clean up expired allocations and pairings.
    pub fn cleanup(&mut self) {
        // Expired allocations.
        let expired: Vec<AllocationId> = self
            .allocations
            .iter()
            .filter(|(_, a)| a.is_expired())
            .map(|(id, _)| *id)
            .collect();

        for id in expired {
            self.close_allocation(id);
            self.stats.allocations_expired += 1;
        }

        // Remove pairings where either allocation is gone.
        let dead_pairings: Vec<PairingId> = self
            .pairings
            .iter()
            .filter(|(_, p)| {
                !self.allocations.contains_key(&p.allocation_a)
                    || !self.allocations.contains_key(&p.allocation_b)
            })
            .map(|(id, _)| *id)
            .collect();

        for id in dead_pairings {
            self.pairings.remove(&id);
        }

        self.stats.active_pairings = self.pairings.len();
        self.last_cleanup = Instant::now();
    }

    /// Update the relay server (call periodically).
    pub fn update(&mut self) {
        if self.last_cleanup.elapsed() >= self.config.cleanup_interval {
            self.cleanup();
        }

        // Update bandwidth stats.
        let total_bw: f64 = self
            .allocations
            .values()
            .map(|a| a.throttle.current_throughput())
            .sum();
        self.stats.current_bandwidth = total_bw;
    }

    /// Returns a reference to the server statistics.
    pub fn stats(&self) -> &RelayStats {
        &self.stats
    }

    /// Returns the server configuration.
    pub fn config(&self) -> &RelayServerConfig {
        &self.config
    }

    /// Get an allocation by ID.
    pub fn get_allocation(&self, id: AllocationId) -> Option<&Allocation> {
        self.allocations.get(&id)
    }

    /// Get a pairing by ID.
    pub fn get_pairing(&self, id: PairingId) -> Option<&ConnectionPairing> {
        self.pairings.get(&id)
    }

    /// Number of active allocations.
    pub fn active_allocation_count(&self) -> usize {
        self.allocations.len()
    }

    /// Number of active pairings.
    pub fn active_pairing_count(&self) -> usize {
        self.pairings.len()
    }
}

impl fmt::Debug for RelayServer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RelayServer")
            .field("allocations", &self.allocations.len())
            .field("pairings", &self.pairings.len())
            .finish()
    }
}

/// Errors from relay operations.
#[derive(Debug, Clone)]
pub enum RelayError {
    InvalidAllocation,
    AllocationExpired,
    BandwidthExceeded,
    PacketTooLarge,
    ServerFull,
    AuthenticationFailed(AuthResult),
    PeerNotFound,
}

impl fmt::Display for RelayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RelayError::InvalidAllocation => write!(f, "invalid allocation"),
            RelayError::AllocationExpired => write!(f, "allocation expired"),
            RelayError::BandwidthExceeded => write!(f, "bandwidth limit exceeded"),
            RelayError::PacketTooLarge => write!(f, "packet too large"),
            RelayError::ServerFull => write!(f, "server at capacity"),
            RelayError::AuthenticationFailed(r) => write!(f, "auth failed: {}", r),
            RelayError::PeerNotFound => write!(f, "peer not found"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_server() -> RelayServer {
        let mut auth = SimpleAuthenticator::new();
        auth.add_user("alice", b"secret_a");
        auth.add_user("bob", b"secret_b");

        RelayServer::new(RelayServerConfig::default(), Box::new(auth))
    }

    #[test]
    fn test_allocation() {
        let mut server = make_server();
        let client = RelayClientId::new(1);
        let addr: SocketAddr = "127.0.0.1:5000".parse().unwrap();
        let creds = RelayCredentials::new("alice", b"secret_a", u64::MAX);

        let result = server.handle_allocate(client, addr, &creds, 300);
        match result {
            AllocateResult::Success { allocation_id, .. } => {
                assert!(server.get_allocation(allocation_id).is_some());
            }
            _ => panic!("expected success"),
        }
        assert_eq!(server.active_allocation_count(), 1);
    }

    #[test]
    fn test_auth_failure() {
        let mut server = make_server();
        let client = RelayClientId::new(1);
        let addr: SocketAddr = "127.0.0.1:5000".parse().unwrap();
        let creds = RelayCredentials::new("alice", b"wrong_password", u64::MAX);

        let result = server.handle_allocate(client, addr, &creds, 300);
        match result {
            AllocateResult::AuthFailed(_) => {}
            _ => panic!("expected auth failure"),
        }
    }

    #[test]
    fn test_relay_data() {
        let mut server = make_server();
        let client = RelayClientId::new(1);
        let addr: SocketAddr = "127.0.0.1:5000".parse().unwrap();
        let creds = RelayCredentials::new("alice", b"secret_a", u64::MAX);

        let alloc_id = match server.handle_allocate(client, addr, &creds, 300) {
            AllocateResult::Success { allocation_id, .. } => allocation_id,
            _ => panic!("expected success"),
        };

        let target = RelayClientId::new(2);
        let payload = b"hello world";
        let result = server.handle_relay_data(alloc_id, target, payload);
        assert!(result.is_ok());
        assert_eq!(server.stats().total_packets_relayed, 1);
    }

    #[test]
    fn test_close_allocation() {
        let mut server = make_server();
        let client = RelayClientId::new(1);
        let addr: SocketAddr = "127.0.0.1:5000".parse().unwrap();
        let creds = RelayCredentials::new("alice", b"secret_a", u64::MAX);

        let alloc_id = match server.handle_allocate(client, addr, &creds, 300) {
            AllocateResult::Success { allocation_id, .. } => allocation_id,
            _ => panic!("expected success"),
        };

        server.close_allocation(alloc_id);
        assert_eq!(server.active_allocation_count(), 0);
    }

    #[test]
    fn test_bandwidth_throttle() {
        let mut throttle = BandwidthThrottle::new(1000); // 1000 bytes/sec
        assert!(throttle.try_consume(500));
        assert!(throttle.try_consume(400));
        // Next 200 should fail (only ~100 tokens left).
        assert!(!throttle.try_consume(200));
    }
}
