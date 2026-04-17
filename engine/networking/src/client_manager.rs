// engine/networking/src/client_manager.rs
//
// Client connection manager for the Genovo engine.
//
// Manages client connections from the server's perspective:
//
// - Connection lifecycle: connect, authenticate, play, disconnect.
// - Authentication with session tokens.
// - Kick and ban management.
// - Timeout handling with configurable thresholds.
// - Reconnection support with session persistence.
// - Connection quality monitoring.
// - Client capability negotiation.
// - Rate limiting per client.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum pending connections.
const MAX_PENDING: usize = 32;

/// Default authentication timeout in seconds.
const AUTH_TIMEOUT: f32 = 10.0;

/// Default session token length.
const SESSION_TOKEN_LENGTH: usize = 32;

/// Maximum reconnection attempts.
const MAX_RECONNECT_ATTEMPTS: u32 = 5;

/// Session persistence duration after disconnect (seconds).
const SESSION_PERSIST_DURATION: f32 = 120.0;

/// Default rate limit (messages per second).
const DEFAULT_RATE_LIMIT: f32 = 60.0;

/// Ban duration: permanent.
const PERMANENT_BAN: f32 = -1.0;

// ---------------------------------------------------------------------------
// Client ID
// ---------------------------------------------------------------------------

/// Unique identifier for a managed client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ManagedClientId(pub u64);

// ---------------------------------------------------------------------------
// Connection Phase
// ---------------------------------------------------------------------------

/// Current phase of the client connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionPhase {
    /// TCP/UDP handshake in progress.
    Handshake,
    /// Waiting for authentication credentials.
    Authenticating,
    /// Loading game state.
    Loading,
    /// Fully connected and active.
    Active,
    /// Gracefully disconnecting.
    Disconnecting,
    /// Connection lost, awaiting potential reconnection.
    Reconnecting,
    /// Fully disconnected.
    Disconnected,
}

// ---------------------------------------------------------------------------
// Session Token
// ---------------------------------------------------------------------------

/// A session token for client authentication and reconnection.
#[derive(Debug, Clone)]
pub struct SessionToken {
    /// Token string.
    pub token: String,
    /// Client it belongs to.
    pub client_id: ManagedClientId,
    /// Creation time (seconds since server start).
    pub created_at: f64,
    /// Expiry time (seconds since server start).
    pub expires_at: f64,
    /// Whether this token has been used.
    pub used: bool,
    /// Number of times this token has been refreshed.
    pub refresh_count: u32,
}

impl SessionToken {
    /// Create a new session token.
    pub fn new(client_id: ManagedClientId, current_time: f64, duration: f32) -> Self {
        // Generate a pseudo-random token.
        let hash = (client_id.0 as f64 * 7.31 + current_time * 13.37).to_bits();
        let token = format!("{:016X}{:016X}", hash, hash.wrapping_mul(0x517CC1B727220A95));

        Self {
            token,
            client_id,
            created_at: current_time,
            expires_at: current_time + duration as f64,
            used: false,
            refresh_count: 0,
        }
    }

    /// Check if the token is expired.
    pub fn is_expired(&self, current_time: f64) -> bool {
        current_time >= self.expires_at
    }

    /// Refresh the token expiry.
    pub fn refresh(&mut self, current_time: f64, duration: f32) {
        self.expires_at = current_time + duration as f64;
        self.refresh_count += 1;
    }
}

// ---------------------------------------------------------------------------
// Ban Entry
// ---------------------------------------------------------------------------

/// A ban record.
#[derive(Debug, Clone)]
pub struct BanEntry {
    /// Banned identifier (address or player name).
    pub identifier: String,
    /// Ban reason.
    pub reason: String,
    /// When the ban was issued.
    pub issued_at: f64,
    /// Ban duration in seconds (negative = permanent).
    pub duration: f32,
    /// Who issued the ban.
    pub issued_by: String,
}

impl BanEntry {
    /// Check if this ban is still active.
    pub fn is_active(&self, current_time: f64) -> bool {
        if self.duration < 0.0 {
            return true; // Permanent ban.
        }
        current_time < self.issued_at + self.duration as f64
    }

    /// Remaining ban time in seconds.
    pub fn remaining(&self, current_time: f64) -> f32 {
        if self.duration < 0.0 {
            return f32::INFINITY;
        }
        ((self.issued_at + self.duration as f64) - current_time).max(0.0) as f32
    }
}

// ---------------------------------------------------------------------------
// Client Capabilities
// ---------------------------------------------------------------------------

/// Client capabilities negotiated during handshake.
#[derive(Debug, Clone)]
pub struct ClientCapabilities {
    /// Protocol version supported.
    pub protocol_version: u32,
    /// Maximum receive rate (bytes per second).
    pub max_receive_rate: u32,
    /// Whether the client supports compression.
    pub supports_compression: bool,
    /// Whether the client supports encryption.
    pub supports_encryption: bool,
    /// Client platform.
    pub platform: String,
    /// Client version string.
    pub client_version: String,
    /// Maximum entities the client can track.
    pub max_entities: u32,
}

impl Default for ClientCapabilities {
    fn default() -> Self {
        Self {
            protocol_version: 1,
            max_receive_rate: 1_000_000,
            supports_compression: true,
            supports_encryption: true,
            platform: "unknown".to_string(),
            client_version: "0.0.0".to_string(),
            max_entities: 10000,
        }
    }
}

// ---------------------------------------------------------------------------
// Managed Client
// ---------------------------------------------------------------------------

/// A managed client connection.
#[derive(Debug, Clone)]
pub struct ManagedClient {
    /// Client ID.
    pub id: ManagedClientId,
    /// Player name.
    pub name: String,
    /// Connection phase.
    pub phase: ConnectionPhase,
    /// Network address.
    pub address: String,
    /// Session token.
    pub session_token: Option<String>,
    /// Authentication state.
    pub authenticated: bool,
    /// Time in current phase.
    pub phase_time: f32,
    /// Total connection time.
    pub total_time: f64,
    /// Time since last message.
    pub idle_time: f32,
    /// Connection timeout.
    pub timeout: f32,
    /// Auth timeout.
    pub auth_timeout: f32,
    /// Capabilities.
    pub capabilities: ClientCapabilities,
    /// Reconnection attempts.
    pub reconnect_attempts: u32,
    /// Maximum reconnection attempts.
    pub max_reconnect_attempts: u32,
    /// Rate limiter state.
    pub rate_limiter: RateLimiterState,
    /// Whether the client is muted (cannot send chat).
    pub muted: bool,
    /// Custom properties.
    pub properties: HashMap<String, String>,
    /// Ping history (last N ping values).
    pub ping_history: Vec<f32>,
    /// Average ping.
    pub avg_ping: f32,
}

/// Rate limiter state.
#[derive(Debug, Clone)]
pub struct RateLimiterState {
    /// Messages sent in the current window.
    pub messages_this_window: u32,
    /// Window start time.
    pub window_start: f32,
    /// Maximum messages per second.
    pub max_rate: f32,
    /// Whether currently rate-limited.
    pub limited: bool,
    /// Total violations.
    pub violations: u32,
}

impl Default for RateLimiterState {
    fn default() -> Self {
        Self {
            messages_this_window: 0,
            window_start: 0.0,
            max_rate: DEFAULT_RATE_LIMIT,
            limited: false,
            violations: 0,
        }
    }
}

impl RateLimiterState {
    /// Record a message and check if rate-limited.
    pub fn check(&mut self, current_time: f32) -> bool {
        if current_time - self.window_start >= 1.0 {
            self.window_start = current_time;
            self.messages_this_window = 0;
            self.limited = false;
        }

        self.messages_this_window += 1;
        if self.messages_this_window as f32 > self.max_rate {
            self.limited = true;
            self.violations += 1;
            return true;
        }
        false
    }
}

impl ManagedClient {
    /// Create a new managed client.
    pub fn new(id: ManagedClientId, name: &str, address: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            phase: ConnectionPhase::Handshake,
            address: address.to_string(),
            session_token: None,
            authenticated: false,
            phase_time: 0.0,
            total_time: 0.0,
            idle_time: 0.0,
            timeout: 30.0,
            auth_timeout: AUTH_TIMEOUT,
            capabilities: ClientCapabilities::default(),
            reconnect_attempts: 0,
            max_reconnect_attempts: MAX_RECONNECT_ATTEMPTS,
            rate_limiter: RateLimiterState::default(),
            muted: false,
            properties: HashMap::new(),
            ping_history: Vec::with_capacity(60),
            avg_ping: 0.0,
        }
    }

    /// Advance to the next phase.
    pub fn advance_phase(&mut self, new_phase: ConnectionPhase) {
        self.phase = new_phase;
        self.phase_time = 0.0;
    }

    /// Record a ping measurement.
    pub fn record_ping(&mut self, ping_ms: f32) {
        self.ping_history.push(ping_ms);
        if self.ping_history.len() > 60 {
            self.ping_history.remove(0);
        }
        self.avg_ping = self.ping_history.iter().sum::<f32>() / self.ping_history.len() as f32;
    }

    /// Whether the client needs timeout handling.
    pub fn is_timed_out(&self) -> bool {
        self.idle_time >= self.timeout
    }

    /// Whether authentication has timed out.
    pub fn auth_timed_out(&self) -> bool {
        self.phase == ConnectionPhase::Authenticating && self.phase_time >= self.auth_timeout
    }

    /// Whether the client is active.
    pub fn is_active(&self) -> bool {
        self.phase == ConnectionPhase::Active
    }

    /// Whether the client can attempt reconnection.
    pub fn can_reconnect(&self) -> bool {
        self.reconnect_attempts < self.max_reconnect_attempts
    }
}

// ---------------------------------------------------------------------------
// Client Manager Events
// ---------------------------------------------------------------------------

/// Events from the client manager.
#[derive(Debug, Clone)]
pub enum ClientManagerEvent {
    /// Client completed handshake.
    HandshakeComplete { client_id: ManagedClientId },
    /// Client authenticated.
    Authenticated { client_id: ManagedClientId, name: String },
    /// Authentication failed.
    AuthFailed { client_id: ManagedClientId, reason: String },
    /// Client became active.
    ClientActive { client_id: ManagedClientId },
    /// Client disconnected.
    ClientDisconnected { client_id: ManagedClientId, reason: String },
    /// Client reconnected.
    ClientReconnected { client_id: ManagedClientId },
    /// Client kicked.
    ClientKicked { client_id: ManagedClientId, reason: String },
    /// Client banned.
    ClientBanned { client_id: ManagedClientId, reason: String },
    /// Client timed out.
    ClientTimedOut { client_id: ManagedClientId },
    /// Rate limit exceeded.
    RateLimitExceeded { client_id: ManagedClientId },
}

// ---------------------------------------------------------------------------
// Client Manager
// ---------------------------------------------------------------------------

/// Manages all client connections.
#[derive(Debug)]
pub struct ClientManager {
    /// Active clients.
    pub clients: HashMap<ManagedClientId, ManagedClient>,
    /// Session tokens.
    pub sessions: HashMap<String, SessionToken>,
    /// Ban list.
    pub bans: Vec<BanEntry>,
    /// Next client ID.
    next_id: u64,
    /// Maximum concurrent clients.
    pub max_clients: usize,
    /// Maximum pending connections.
    pub max_pending: usize,
    /// Current time.
    pub current_time: f64,
    /// Events.
    pub events: Vec<ClientManagerEvent>,
    /// Statistics.
    pub stats: ClientManagerStats,
}

impl ClientManager {
    /// Create a new client manager.
    pub fn new(max_clients: usize) -> Self {
        Self {
            clients: HashMap::new(),
            sessions: HashMap::new(),
            bans: Vec::new(),
            next_id: 1,
            max_clients,
            max_pending: MAX_PENDING,
            current_time: 0.0,
            events: Vec::new(),
            stats: ClientManagerStats::default(),
        }
    }

    /// Accept a new connection.
    pub fn accept_connection(&mut self, name: &str, address: &str) -> Option<ManagedClientId> {
        // Check ban list.
        if self.is_banned(address) {
            return None;
        }

        // Check capacity.
        let active_count = self.clients.values().filter(|c| c.is_active()).count();
        if active_count >= self.max_clients {
            return None;
        }

        let id = ManagedClientId(self.next_id);
        self.next_id += 1;

        let client = ManagedClient::new(id, name, address);
        self.clients.insert(id, client);
        self.stats.total_connections += 1;

        Some(id)
    }

    /// Authenticate a client.
    pub fn authenticate(&mut self, client_id: ManagedClientId, credentials: &str) -> bool {
        let client = match self.clients.get_mut(&client_id) {
            Some(c) => c,
            None => return false,
        };

        // Simple authentication: accept any non-empty credentials.
        if credentials.is_empty() {
            self.events.push(ClientManagerEvent::AuthFailed {
                client_id,
                reason: "Empty credentials".to_string(),
            });
            return false;
        }

        client.authenticated = true;
        client.advance_phase(ConnectionPhase::Loading);

        // Generate session token.
        let token = SessionToken::new(client_id, self.current_time, SESSION_PERSIST_DURATION);
        let token_str = token.token.clone();
        client.session_token = Some(token_str.clone());
        self.sessions.insert(token_str, token);

        self.events.push(ClientManagerEvent::Authenticated {
            client_id,
            name: client.name.clone(),
        });

        true
    }

    /// Mark a client as active (loading complete).
    pub fn activate_client(&mut self, client_id: ManagedClientId) {
        if let Some(client) = self.clients.get_mut(&client_id) {
            client.advance_phase(ConnectionPhase::Active);
            self.events.push(ClientManagerEvent::ClientActive { client_id });
        }
    }

    /// Attempt reconnection with a session token.
    pub fn reconnect(&mut self, token: &str, address: &str) -> Option<ManagedClientId> {
        let session = self.sessions.get(token)?.clone();

        if session.is_expired(self.current_time) {
            return None;
        }

        let client_id = session.client_id;
        if let Some(client) = self.clients.get_mut(&client_id) {
            if !client.can_reconnect() {
                return None;
            }

            client.reconnect_attempts += 1;
            client.advance_phase(ConnectionPhase::Active);
            client.address = address.to_string();
            client.idle_time = 0.0;

            self.events.push(ClientManagerEvent::ClientReconnected { client_id });
            self.stats.reconnections += 1;

            Some(client_id)
        } else {
            None
        }
    }

    /// Disconnect a client.
    pub fn disconnect(&mut self, client_id: ManagedClientId, reason: &str) {
        if let Some(client) = self.clients.get_mut(&client_id) {
            client.advance_phase(ConnectionPhase::Disconnected);
            self.events.push(ClientManagerEvent::ClientDisconnected {
                client_id,
                reason: reason.to_string(),
            });
        }
    }

    /// Kick a client.
    pub fn kick(&mut self, client_id: ManagedClientId, reason: &str) {
        self.events.push(ClientManagerEvent::ClientKicked {
            client_id,
            reason: reason.to_string(),
        });
        self.disconnect(client_id, reason);
        // Invalidate session.
        if let Some(client) = self.clients.get(&client_id) {
            if let Some(token) = &client.session_token {
                self.sessions.remove(token);
            }
        }
    }

    /// Ban a client.
    pub fn ban(&mut self, client_id: ManagedClientId, reason: &str, duration: f32) {
        if let Some(client) = self.clients.get(&client_id) {
            self.bans.push(BanEntry {
                identifier: client.address.clone(),
                reason: reason.to_string(),
                issued_at: self.current_time,
                duration,
                issued_by: "system".to_string(),
            });
            self.events.push(ClientManagerEvent::ClientBanned {
                client_id,
                reason: reason.to_string(),
            });
        }
        self.kick(client_id, reason);
    }

    /// Check if an address is banned.
    pub fn is_banned(&self, address: &str) -> bool {
        self.bans.iter().any(|b| b.identifier == address && b.is_active(self.current_time))
    }

    /// Update all clients.
    pub fn update(&mut self, dt: f32) {
        self.current_time += dt as f64;

        let client_ids: Vec<ManagedClientId> = self.clients.keys().copied().collect();
        for id in client_ids {
            if let Some(client) = self.clients.get_mut(&id) {
                client.phase_time += dt;
                client.total_time += dt as f64;
                client.idle_time += dt;

                // Check timeouts.
                if client.is_timed_out() && client.is_active() {
                    client.advance_phase(ConnectionPhase::Reconnecting);
                    self.events.push(ClientManagerEvent::ClientTimedOut { client_id: id });
                }

                if client.auth_timed_out() {
                    self.events.push(ClientManagerEvent::AuthFailed {
                        client_id: id,
                        reason: "Authentication timeout".to_string(),
                    });
                }
            }
        }

        // Clean up expired sessions.
        self.sessions.retain(|_, s| !s.is_expired(self.current_time));

        // Clean up expired bans.
        self.bans.retain(|b| b.is_active(self.current_time));

        // Update stats.
        self.stats.active_clients = self.clients.values().filter(|c| c.is_active()).count() as u32;
        self.stats.pending_clients = self.clients.values()
            .filter(|c| matches!(c.phase, ConnectionPhase::Handshake | ConnectionPhase::Authenticating))
            .count() as u32;
    }

    /// Record a message from a client (resets idle timer, checks rate limit).
    pub fn on_client_message(&mut self, client_id: ManagedClientId) {
        if let Some(client) = self.clients.get_mut(&client_id) {
            client.idle_time = 0.0;

            let limited = client.rate_limiter.check(client.total_time as f32);
            if limited {
                self.events.push(ClientManagerEvent::RateLimitExceeded { client_id });
            }
        }
    }

    /// Get a client by ID.
    pub fn get(&self, id: ManagedClientId) -> Option<&ManagedClient> {
        self.clients.get(&id)
    }

    /// Get active client count.
    pub fn active_count(&self) -> usize {
        self.clients.values().filter(|c| c.is_active()).count()
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<ClientManagerEvent> {
        std::mem::take(&mut self.events)
    }
}

/// Client manager statistics.
#[derive(Debug, Clone, Default)]
pub struct ClientManagerStats {
    /// Total connections ever accepted.
    pub total_connections: u64,
    /// Currently active clients.
    pub active_clients: u32,
    /// Pending clients.
    pub pending_clients: u32,
    /// Successful reconnections.
    pub reconnections: u64,
    /// Total kicks.
    pub kicks: u64,
    /// Total bans.
    pub bans: u64,
    /// Total auth failures.
    pub auth_failures: u64,
}
