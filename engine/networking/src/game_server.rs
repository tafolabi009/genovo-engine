// engine/networking/src/game_server.rs
//
// Game server framework for the Genovo engine.
//
// Provides a complete authoritative game server:
//
// - Server tick loop with fixed timestep.
// - Client management: connect, disconnect, timeout.
// - World state management and replication.
// - Authority model: server-authoritative with client prediction.
// - Anti-cheat hooks for validation.
// - Admin commands.
// - Server logging and metrics.
// - Graceful shutdown.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default server tick rate (ticks per second).
const DEFAULT_TICK_RATE: u32 = 60;

/// Default client timeout in seconds.
const DEFAULT_CLIENT_TIMEOUT: f32 = 30.0;

/// Maximum connected clients.
const MAX_CLIENTS: usize = 64;

/// Server log buffer size.
const LOG_BUFFER_SIZE: usize = 1024;

/// Default max entities in the world.
const DEFAULT_MAX_ENTITIES: usize = 10000;

/// Anti-cheat validation interval (ticks).
const ANTICHEAT_INTERVAL: u32 = 60;

// ---------------------------------------------------------------------------
// Client ID
// ---------------------------------------------------------------------------

/// Unique identifier for a connected client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ServerClientId(pub u64);

/// Unique identifier for a server entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ServerEntityId(pub u64);

// ---------------------------------------------------------------------------
// Client State
// ---------------------------------------------------------------------------

/// State of a connected client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientConnectionState {
    /// Connection handshake in progress.
    Connecting,
    /// Authenticated and loading.
    Loading,
    /// Fully connected and playing.
    Connected,
    /// Disconnecting gracefully.
    Disconnecting,
    /// Timed out.
    TimedOut,
    /// Kicked by admin.
    Kicked,
    /// Banned.
    Banned,
}

/// Information about a connected client.
#[derive(Debug, Clone)]
pub struct ServerClient {
    /// Client identifier.
    pub id: ServerClientId,
    /// Player name.
    pub name: String,
    /// Connection state.
    pub state: ClientConnectionState,
    /// Authentication token.
    pub auth_token: String,
    /// IP address (as string).
    pub address: String,
    /// Time since last message received.
    pub time_since_last_msg: f32,
    /// Timeout threshold.
    pub timeout: f32,
    /// Round-trip time in milliseconds.
    pub rtt_ms: f32,
    /// Packet loss percentage.
    pub packet_loss: f32,
    /// Player entity in the world.
    pub player_entity: Option<ServerEntityId>,
    /// Tick of the last input received.
    pub last_input_tick: u64,
    /// Connection time.
    pub connected_at: f64,
    /// Total bytes sent.
    pub bytes_sent: u64,
    /// Total bytes received.
    pub bytes_received: u64,
    /// Warnings issued (for anti-cheat).
    pub warnings: u32,
    /// Whether this client has admin privileges.
    pub is_admin: bool,
    /// Team assignment.
    pub team: u32,
}

impl ServerClient {
    /// Create a new client.
    pub fn new(id: ServerClientId, name: &str, address: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            state: ClientConnectionState::Connecting,
            auth_token: String::new(),
            address: address.to_string(),
            time_since_last_msg: 0.0,
            timeout: DEFAULT_CLIENT_TIMEOUT,
            rtt_ms: 0.0,
            packet_loss: 0.0,
            player_entity: None,
            last_input_tick: 0,
            connected_at: 0.0,
            bytes_sent: 0,
            bytes_received: 0,
            warnings: 0,
            is_admin: false,
            team: 0,
        }
    }

    /// Check if the client has timed out.
    pub fn is_timed_out(&self) -> bool {
        self.time_since_last_msg >= self.timeout
    }

    /// Check if the client is fully connected.
    pub fn is_connected(&self) -> bool {
        self.state == ClientConnectionState::Connected
    }

    /// Connection duration in seconds.
    pub fn connection_duration(&self, current_time: f64) -> f64 {
        current_time - self.connected_at
    }
}

// ---------------------------------------------------------------------------
// Server Entity
// ---------------------------------------------------------------------------

/// A server-side entity.
#[derive(Debug, Clone)]
pub struct ServerEntity {
    /// Entity ID.
    pub id: ServerEntityId,
    /// Entity type.
    pub entity_type: String,
    /// Position.
    pub position: [f32; 3],
    /// Rotation (quaternion).
    pub rotation: [f32; 4],
    /// Velocity.
    pub velocity: [f32; 3],
    /// Health.
    pub health: f32,
    /// Owner client (None = server-owned).
    pub owner: Option<ServerClientId>,
    /// Whether this entity is replicated to clients.
    pub replicated: bool,
    /// Replication priority.
    pub replication_priority: f32,
    /// Last update tick.
    pub last_update_tick: u64,
    /// Whether this entity has changed since last replication.
    pub dirty: bool,
    /// Custom properties.
    pub properties: HashMap<String, String>,
}

impl ServerEntity {
    /// Create a new server entity.
    pub fn new(id: ServerEntityId, entity_type: &str) -> Self {
        Self {
            id,
            entity_type: entity_type.to_string(),
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            velocity: [0.0; 3],
            health: 100.0,
            owner: None,
            replicated: true,
            replication_priority: 1.0,
            last_update_tick: 0,
            dirty: true,
            properties: HashMap::new(),
        }
    }

    /// Mark as dirty (needs replication).
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Distance to a point.
    pub fn distance_to(&self, point: [f32; 3]) -> f32 {
        let dx = self.position[0] - point[0];
        let dy = self.position[1] - point[1];
        let dz = self.position[2] - point[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Server Events
// ---------------------------------------------------------------------------

/// Events emitted by the game server.
#[derive(Debug, Clone)]
pub enum ServerEvent {
    /// Client connected.
    ClientConnected { client_id: ServerClientId, name: String },
    /// Client disconnected.
    ClientDisconnected { client_id: ServerClientId, reason: String },
    /// Client timed out.
    ClientTimedOut { client_id: ServerClientId },
    /// Client kicked.
    ClientKicked { client_id: ServerClientId, reason: String },
    /// Entity spawned.
    EntitySpawned { entity_id: ServerEntityId, entity_type: String },
    /// Entity destroyed.
    EntityDestroyed { entity_id: ServerEntityId },
    /// Anti-cheat violation detected.
    AntiCheatViolation { client_id: ServerClientId, violation: String },
    /// Admin command executed.
    AdminCommand { client_id: ServerClientId, command: String },
    /// Server started.
    ServerStarted { tick_rate: u32, max_clients: usize },
    /// Server shutting down.
    ServerShuttingDown { reason: String },
}

// ---------------------------------------------------------------------------
// Admin Commands
// ---------------------------------------------------------------------------

/// Admin command types.
#[derive(Debug, Clone)]
pub enum AdminCommand {
    /// Kick a client.
    Kick { client_id: ServerClientId, reason: String },
    /// Ban a client.
    Ban { client_id: ServerClientId, reason: String, duration_hours: f32 },
    /// Change map/level.
    ChangeMap { map_name: String },
    /// Set server variable.
    SetVar { key: String, value: String },
    /// Send message to all clients.
    Broadcast { message: String },
    /// Restart the server.
    Restart,
    /// Shutdown the server.
    Shutdown { reason: String },
    /// Spawn an entity.
    SpawnEntity { entity_type: String, position: [f32; 3] },
    /// Teleport a player.
    TeleportPlayer { client_id: ServerClientId, position: [f32; 3] },
    /// Set player health.
    SetHealth { client_id: ServerClientId, health: f32 },
    /// List connected clients.
    ListClients,
    /// Get server status.
    Status,
}

// ---------------------------------------------------------------------------
// Anti-Cheat
// ---------------------------------------------------------------------------

/// Anti-cheat validation result.
#[derive(Debug, Clone)]
pub struct AntiCheatResult {
    /// Client being validated.
    pub client_id: ServerClientId,
    /// Whether the check passed.
    pub passed: bool,
    /// Violation type (if any).
    pub violation: Option<String>,
    /// Severity (0 = minor, 1 = severe).
    pub severity: f32,
}

/// Anti-cheat hook configuration.
#[derive(Debug, Clone)]
pub struct AntiCheatConfig {
    /// Maximum allowed speed.
    pub max_speed: f32,
    /// Maximum teleport distance per tick.
    pub max_teleport_distance: f32,
    /// Maximum damage per hit.
    pub max_damage_per_hit: f32,
    /// Maximum fire rate (shots per second).
    pub max_fire_rate: f32,
    /// Warnings before kick.
    pub warnings_before_kick: u32,
    /// Whether anti-cheat is enabled.
    pub enabled: bool,
}

impl Default for AntiCheatConfig {
    fn default() -> Self {
        Self {
            max_speed: 20.0,
            max_teleport_distance: 5.0,
            max_damage_per_hit: 200.0,
            max_fire_rate: 15.0,
            warnings_before_kick: 5,
            enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Server Log Entry
// ---------------------------------------------------------------------------

/// A server log entry.
#[derive(Debug, Clone)]
pub struct ServerLogEntry {
    /// Timestamp (seconds since server start).
    pub timestamp: f64,
    /// Log level.
    pub level: LogLevel,
    /// Message.
    pub message: String,
    /// Category.
    pub category: String,
}

/// Log level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

// ---------------------------------------------------------------------------
// Game Server
// ---------------------------------------------------------------------------

/// The main game server.
#[derive(Debug)]
pub struct GameServer {
    /// Server tick rate.
    pub tick_rate: u32,
    /// Current tick number.
    pub current_tick: u64,
    /// Elapsed time since server start.
    pub elapsed_time: f64,
    /// Delta time per tick.
    pub dt: f32,
    /// Connected clients.
    pub clients: HashMap<ServerClientId, ServerClient>,
    /// Server entities.
    pub entities: HashMap<ServerEntityId, ServerEntity>,
    /// Next client ID.
    next_client_id: u64,
    /// Next entity ID.
    next_entity_id: u64,
    /// Maximum clients.
    pub max_clients: usize,
    /// Server name.
    pub name: String,
    /// Current map.
    pub map_name: String,
    /// Server variables.
    pub vars: HashMap<String, String>,
    /// Anti-cheat configuration.
    pub anti_cheat: AntiCheatConfig,
    /// Server events.
    pub events: Vec<ServerEvent>,
    /// Log buffer.
    pub log: Vec<ServerLogEntry>,
    /// Whether the server is running.
    pub running: bool,
    /// Shutdown requested.
    pub shutdown_requested: bool,
    /// Statistics.
    pub stats: ServerStats,
    /// Ban list (client addresses).
    pub ban_list: Vec<String>,
}

impl GameServer {
    /// Create a new game server.
    pub fn new(name: &str, tick_rate: u32) -> Self {
        let dt = 1.0 / tick_rate as f32;
        Self {
            tick_rate,
            current_tick: 0,
            elapsed_time: 0.0,
            dt,
            clients: HashMap::new(),
            entities: HashMap::new(),
            next_client_id: 1,
            next_entity_id: 1,
            max_clients: MAX_CLIENTS,
            name: name.to_string(),
            map_name: String::new(),
            vars: HashMap::new(),
            anti_cheat: AntiCheatConfig::default(),
            events: Vec::new(),
            log: Vec::new(),
            running: false,
            shutdown_requested: false,
            stats: ServerStats::default(),
            ban_list: Vec::new(),
        }
    }

    /// Start the server.
    pub fn start(&mut self) {
        self.running = true;
        self.log_info("Server", &format!("Server '{}' started at {} tick/s", self.name, self.tick_rate));
        self.events.push(ServerEvent::ServerStarted {
            tick_rate: self.tick_rate,
            max_clients: self.max_clients,
        });
    }

    /// Process a single server tick.
    pub fn tick(&mut self) {
        if !self.running { return; }

        self.current_tick += 1;
        self.elapsed_time += self.dt as f64;

        // Update client timeouts.
        let mut timed_out = Vec::new();
        for client in self.clients.values_mut() {
            client.time_since_last_msg += self.dt;
            if client.is_timed_out() && client.is_connected() {
                client.state = ClientConnectionState::TimedOut;
                timed_out.push(client.id);
            }
        }
        for id in timed_out {
            self.events.push(ServerEvent::ClientTimedOut { client_id: id });
            self.log_warning("Network", &format!("Client {:?} timed out", id));
        }

        // Anti-cheat checks.
        if self.anti_cheat.enabled && self.current_tick % ANTICHEAT_INTERVAL as u64 == 0 {
            self.run_anti_cheat();
        }

        // Update statistics.
        self.stats.current_tick = self.current_tick;
        self.stats.connected_clients = self.clients.values().filter(|c| c.is_connected()).count() as u32;
        self.stats.entity_count = self.entities.len() as u32;
        self.stats.uptime_seconds = self.elapsed_time;

        // Check for shutdown.
        if self.shutdown_requested {
            self.shutdown("Shutdown requested");
        }
    }

    /// Connect a new client.
    pub fn connect_client(&mut self, name: &str, address: &str) -> Option<ServerClientId> {
        if self.clients.values().filter(|c| c.is_connected()).count() >= self.max_clients {
            self.log_warning("Network", "Connection rejected: server full");
            return None;
        }

        if self.ban_list.contains(&address.to_string()) {
            self.log_warning("Network", &format!("Connection rejected: {} is banned", address));
            return None;
        }

        let id = ServerClientId(self.next_client_id);
        self.next_client_id += 1;

        let mut client = ServerClient::new(id, name, address);
        client.state = ClientConnectionState::Connected;
        client.connected_at = self.elapsed_time;

        self.clients.insert(id, client);
        self.events.push(ServerEvent::ClientConnected { client_id: id, name: name.to_string() });
        self.log_info("Network", &format!("Client '{}' connected from {}", name, address));

        Some(id)
    }

    /// Disconnect a client.
    pub fn disconnect_client(&mut self, id: ServerClientId, reason: &str) {
        let client_name = if let Some(client) = self.clients.get_mut(&id) {
            client.state = ClientConnectionState::Disconnecting;
            let name = client.name.clone();
            self.events.push(ServerEvent::ClientDisconnected {
                client_id: id,
                reason: reason.to_string(),
            });
            Some(name)
        } else {
            None
        };
        if let Some(name) = client_name {
            self.log_info("Network", &format!("Client '{}' disconnected: {}", name, reason));
        }
    }

    /// Kick a client.
    pub fn kick_client(&mut self, id: ServerClientId, reason: &str) {
        let client_name = if let Some(client) = self.clients.get_mut(&id) {
            client.state = ClientConnectionState::Kicked;
            let name = client.name.clone();
            self.events.push(ServerEvent::ClientKicked {
                client_id: id,
                reason: reason.to_string(),
            });
            Some(name)
        } else {
            None
        };
        if let Some(name) = client_name {
            self.log_warning("Admin", &format!("Client '{}' kicked: {}", name, reason));
        }
    }

    /// Ban a client.
    pub fn ban_client(&mut self, id: ServerClientId, reason: &str) {
        if let Some(client) = self.clients.get(&id) {
            self.ban_list.push(client.address.clone());
            self.log_warning("Admin", &format!("Client '{}' banned: {}", client.name, reason));
        }
        self.kick_client(id, reason);
    }

    /// Spawn a server entity.
    pub fn spawn_entity(&mut self, entity_type: &str) -> ServerEntityId {
        let id = ServerEntityId(self.next_entity_id);
        self.next_entity_id += 1;

        let entity = ServerEntity::new(id, entity_type);
        self.entities.insert(id, entity);
        self.events.push(ServerEvent::EntitySpawned {
            entity_id: id,
            entity_type: entity_type.to_string(),
        });

        id
    }

    /// Destroy a server entity.
    pub fn destroy_entity(&mut self, id: ServerEntityId) {
        if self.entities.remove(&id).is_some() {
            self.events.push(ServerEvent::EntityDestroyed { entity_id: id });
        }
    }

    /// Run anti-cheat validation.
    fn run_anti_cheat(&mut self) {
        let client_ids: Vec<ServerClientId> = self.clients.keys().copied().collect();
        for client_id in client_ids {
            if let Some(client) = self.clients.get(&client_id) {
                if !client.is_connected() { continue; }

                if let Some(entity_id) = client.player_entity {
                    if let Some(entity) = self.entities.get(&entity_id) {
                        let speed = (entity.velocity[0] * entity.velocity[0]
                            + entity.velocity[1] * entity.velocity[1]
                            + entity.velocity[2] * entity.velocity[2]).sqrt();

                        if speed > self.anti_cheat.max_speed {
                            self.events.push(ServerEvent::AntiCheatViolation {
                                client_id,
                                violation: format!("Speed hack detected: {:.1} > {:.1}", speed, self.anti_cheat.max_speed),
                            });

                            if let Some(c) = self.clients.get_mut(&client_id) {
                                c.warnings += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Shutdown the server.
    pub fn shutdown(&mut self, reason: &str) {
        self.events.push(ServerEvent::ServerShuttingDown { reason: reason.to_string() });
        self.log_info("Server", &format!("Server shutting down: {}", reason));
        self.running = false;
    }

    /// Log a message.
    fn log_entry(&mut self, level: LogLevel, category: &str, message: &str) {
        let entry = ServerLogEntry {
            timestamp: self.elapsed_time,
            level,
            message: message.to_string(),
            category: category.to_string(),
        };
        self.log.push(entry);
        if self.log.len() > LOG_BUFFER_SIZE {
            self.log.remove(0);
        }
    }

    fn log_info(&mut self, category: &str, message: &str) {
        self.log_entry(LogLevel::Info, category, message);
    }

    fn log_warning(&mut self, category: &str, message: &str) {
        self.log_entry(LogLevel::Warning, category, message);
    }

    /// Get a client by ID.
    pub fn get_client(&self, id: ServerClientId) -> Option<&ServerClient> {
        self.clients.get(&id)
    }

    /// Get connected client count.
    pub fn connected_count(&self) -> usize {
        self.clients.values().filter(|c| c.is_connected()).count()
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<ServerEvent> {
        std::mem::take(&mut self.events)
    }
}

/// Server statistics.
#[derive(Debug, Clone, Default)]
pub struct ServerStats {
    /// Current tick.
    pub current_tick: u64,
    /// Connected clients.
    pub connected_clients: u32,
    /// Total entities.
    pub entity_count: u32,
    /// Uptime in seconds.
    pub uptime_seconds: f64,
    /// Total bytes sent.
    pub total_bytes_sent: u64,
    /// Total bytes received.
    pub total_bytes_received: u64,
    /// Peak client count.
    pub peak_clients: u32,
}
