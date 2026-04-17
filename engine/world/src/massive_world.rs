//! # Massive World
//!
//! EVE Online-scale world management providing universe-scale structure,
//! server instancing, hierarchical spatial queries, and persistent world state.
//!
//! ## Architecture
//!
//! ```text
//! GalaxyMap (universe structure)
//!     |
//!     +-- SolarSystem (node)
//!     |       |-- Star, Planets, Stations, AsteroidBelts
//!     |       |-- System LOD (full only if occupied)
//!     |
//!     +-- WarpLane / JumpGate (edge)
//!
//! InstanceManager (server sharding)
//!     |
//!     +-- Instance (isolated world shard)
//!     |       |-- Player limit, spin-up/down
//!     |       |-- Cross-instance travel
//!     |       |-- Instance migration
//!
//! SpatialDatabase (efficient large-world queries)
//!     |
//!     +-- HierarchicalSpatialHash (multi-resolution grid)
//!     |       |-- Range queries
//!     |       |-- Nearest-neighbor
//!     |       |-- Aggregate queries
//!
//! WorldPersistence (save massive world state)
//!     |
//!     +-- Incremental sector saves
//!     +-- Background save thread signaling
//!     +-- Save versioning
//! ```

use std::collections::{HashMap, HashSet, VecDeque, BTreeMap};
use std::fmt;

// ===========================================================================
// Constants
// ===========================================================================

/// Maximum solar systems in a galaxy.
pub const MAX_SOLAR_SYSTEMS: usize = 10_000;

/// Maximum connections per solar system.
pub const MAX_CONNECTIONS_PER_SYSTEM: usize = 8;

/// Default instance player limit.
pub const DEFAULT_INSTANCE_PLAYER_LIMIT: usize = 200;

/// Maximum concurrent instances.
pub const MAX_INSTANCES: usize = 1000;

/// Default spatial hash grid levels.
pub const SPATIAL_HASH_LEVELS: usize = 4;

/// Smallest cell size for the spatial hash (world units).
pub const SPATIAL_HASH_MIN_CELL: f64 = 100.0;

/// Default save version.
pub const SAVE_FORMAT_VERSION: u32 = 1;

/// Maximum entities per spatial hash cell.
pub const MAX_ENTITIES_PER_HASH_CELL: usize = 256;

/// Background save flush interval (seconds).
pub const SAVE_FLUSH_INTERVAL: f64 = 30.0;

/// Maximum dirty sectors to save per flush.
pub const MAX_SECTORS_PER_FLUSH: usize = 16;

// ===========================================================================
// Type aliases
// ===========================================================================

/// Solar system unique identifier.
pub type SystemId = u64;

/// Instance unique identifier.
pub type InstanceId = u64;

/// Entity identifier used in spatial queries.
pub type SpatialEntityId = u64;

/// Sector identifier for persistence.
pub type SectorId = u64;

/// Player identifier.
pub type PlayerId = u64;

// ===========================================================================
// GalaxyPosition — simple 3D position for galaxy-scale
// ===========================================================================

/// Position in galaxy space (light-years or arbitrary units).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GalaxyPosition {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl GalaxyPosition {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn distance(a: Self, b: Self) -> f64 {
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let dz = a.z - b.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    #[inline]
    pub fn distance_sq(a: Self, b: Self) -> f64 {
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let dz = a.z - b.z;
        dx * dx + dy * dy + dz * dz
    }

    #[inline]
    pub fn lerp(a: Self, b: Self, t: f64) -> Self {
        Self {
            x: a.x + (b.x - a.x) * t,
            y: a.y + (b.y - a.y) * t,
            z: a.z + (b.z - a.z) * t,
        }
    }

    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    #[inline]
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

impl Default for GalaxyPosition {
    fn default() -> Self {
        Self::ZERO
    }
}

impl fmt::Display for GalaxyPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.2}, {:.2}, {:.2})", self.x, self.y, self.z)
    }
}

// ===========================================================================
// CelestialBody — stars, planets, stations, etc.
// ===========================================================================

/// Type of celestial body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CelestialType {
    Star,
    Planet,
    Moon,
    AsteroidBelt,
    Station,
    Gate,
    Anomaly,
    Nebula,
    WormholeEntrance,
}

/// A celestial body within a solar system.
#[derive(Debug, Clone)]
pub struct CelestialBody {
    /// Unique ID within the system.
    pub id: u64,
    /// Name.
    pub name: String,
    /// Type.
    pub body_type: CelestialType,
    /// Position relative to the system center.
    pub local_position: GalaxyPosition,
    /// Radius (km).
    pub radius: f64,
    /// Mass (arbitrary units).
    pub mass: f64,
    /// Orbital radius (distance from star).
    pub orbital_radius: f64,
    /// Orbital period (seconds for one revolution).
    pub orbital_period: f64,
    /// Current orbital angle (radians).
    pub orbital_angle: f64,
    /// Whether this body is dockable (stations, some moons).
    pub dockable: bool,
    /// Associated resources (mining, etc.).
    pub resources: Vec<String>,
    /// Security level (0.0 = dangerous, 1.0 = safe).
    pub security: f32,
}

impl CelestialBody {
    /// Create a star.
    pub fn star(name: impl Into<String>, radius: f64, mass: f64) -> Self {
        Self {
            id: 0,
            name: name.into(),
            body_type: CelestialType::Star,
            local_position: GalaxyPosition::ZERO,
            radius,
            mass,
            orbital_radius: 0.0,
            orbital_period: 0.0,
            orbital_angle: 0.0,
            dockable: false,
            resources: Vec::new(),
            security: 1.0,
        }
    }

    /// Create a planet.
    pub fn planet(
        name: impl Into<String>,
        radius: f64,
        orbital_radius: f64,
        orbital_period: f64,
    ) -> Self {
        Self {
            id: 0,
            name: name.into(),
            body_type: CelestialType::Planet,
            local_position: GalaxyPosition::ZERO,
            radius,
            mass: 1.0,
            orbital_radius,
            orbital_period,
            orbital_angle: 0.0,
            dockable: false,
            resources: Vec::new(),
            security: 0.8,
        }
    }

    /// Create a station.
    pub fn station(
        name: impl Into<String>,
        orbital_radius: f64,
    ) -> Self {
        Self {
            id: 0,
            name: name.into(),
            body_type: CelestialType::Station,
            local_position: GalaxyPosition::ZERO,
            radius: 10.0,
            mass: 0.001,
            orbital_radius,
            orbital_period: 3600.0,
            orbital_angle: 0.0,
            dockable: true,
            resources: Vec::new(),
            security: 0.9,
        }
    }

    /// Update orbital position based on elapsed time.
    pub fn update_orbit(&mut self, dt: f64) {
        if self.orbital_period > 0.0 {
            let angular_speed = 2.0 * std::f64::consts::PI / self.orbital_period;
            self.orbital_angle += angular_speed * dt;
            if self.orbital_angle > 2.0 * std::f64::consts::PI {
                self.orbital_angle -= 2.0 * std::f64::consts::PI;
            }

            self.local_position = GalaxyPosition::new(
                self.orbital_radius * self.orbital_angle.cos(),
                0.0,
                self.orbital_radius * self.orbital_angle.sin(),
            );
        }
    }
}

// ===========================================================================
// SolarSystem — a system within the galaxy
// ===========================================================================

/// LOD level for a solar system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemLOD {
    /// Full simulation: all entities active, physics, AI, etc.
    Full,
    /// Reduced: orbits update, but no detailed physics/AI.
    Reduced,
    /// Minimal: only strategic-level state (population, resources).
    Minimal,
    /// Dormant: frozen state, no updates.
    Dormant,
}

/// A solar system in the galaxy.
#[derive(Debug, Clone)]
pub struct SolarSystem {
    /// Unique system ID.
    pub id: SystemId,
    /// Name.
    pub name: String,
    /// Position in galaxy space.
    pub position: GalaxyPosition,
    /// Celestial bodies.
    pub bodies: Vec<CelestialBody>,
    /// Connected system IDs (warp lanes / jump gates).
    pub connections: Vec<SystemId>,
    /// Current LOD level.
    pub lod: SystemLOD,
    /// Number of players currently in this system.
    pub player_count: u32,
    /// Security level (0.0 = lawless, 1.0 = high security).
    pub security: f32,
    /// Instance ID managing this system (if any).
    pub instance_id: Option<InstanceId>,
    /// Whether this system has been discovered.
    pub discovered: bool,
    /// Total population (NPCs + players).
    pub population: u64,
    /// Economic value index.
    pub economic_value: f64,
    /// Control faction (if any).
    pub faction: Option<String>,
    /// Last update timestamp.
    pub last_update: f64,
}

impl SolarSystem {
    /// Create a new solar system.
    pub fn new(id: SystemId, name: impl Into<String>, position: GalaxyPosition) -> Self {
        Self {
            id,
            name: name.into(),
            position,
            bodies: Vec::new(),
            connections: Vec::new(),
            lod: SystemLOD::Dormant,
            player_count: 0,
            security: 0.5,
            instance_id: None,
            discovered: false,
            population: 0,
            economic_value: 0.0,
            faction: None,
            last_update: 0.0,
        }
    }

    /// Add a celestial body.
    pub fn add_body(&mut self, mut body: CelestialBody) -> u64 {
        let id = self.bodies.len() as u64;
        body.id = id;
        self.bodies.push(body);
        id
    }

    /// Add a connection to another system.
    pub fn connect_to(&mut self, other_id: SystemId) {
        if !self.connections.contains(&other_id)
            && self.connections.len() < MAX_CONNECTIONS_PER_SYSTEM
        {
            self.connections.push(other_id);
        }
    }

    /// Update all orbital bodies.
    pub fn update_orbits(&mut self, dt: f64) {
        for body in &mut self.bodies {
            body.update_orbit(dt);
        }
    }

    /// Whether this system should run at full simulation.
    pub fn should_be_full_sim(&self) -> bool {
        self.player_count > 0
    }

    /// Whether this system is occupied.
    pub fn is_occupied(&self) -> bool {
        self.player_count > 0
    }

    /// Get all dockable bodies.
    pub fn dockable_bodies(&self) -> Vec<&CelestialBody> {
        self.bodies.iter().filter(|b| b.dockable).collect()
    }

    /// Find a body by name.
    pub fn find_body(&self, name: &str) -> Option<&CelestialBody> {
        self.bodies.iter().find(|b| b.name == name)
    }
}

// ===========================================================================
// GalaxyMap — universe-scale structure
// ===========================================================================

/// Represents the entire galaxy as a graph of solar systems.
#[derive(Debug)]
pub struct GalaxyMap {
    /// All solar systems indexed by ID.
    pub systems: HashMap<SystemId, SolarSystem>,
    /// Next system ID.
    next_system_id: SystemId,
    /// Edges: (system_a, system_b) representing warp lanes.
    pub warp_lanes: Vec<(SystemId, SystemId)>,
    /// Galaxy name.
    pub name: String,
    /// Galaxy seed (for procedural generation).
    pub seed: u64,
    /// Total player count across all systems.
    pub total_players: u32,
    /// Statistics.
    pub stats: GalaxyStats,
}

/// Galaxy statistics.
#[derive(Debug, Clone, Default)]
pub struct GalaxyStats {
    pub total_systems: usize,
    pub occupied_systems: usize,
    pub full_sim_systems: usize,
    pub reduced_sim_systems: usize,
    pub dormant_systems: usize,
    pub total_warp_lanes: usize,
    pub total_celestial_bodies: usize,
}

impl GalaxyMap {
    /// Create a new galaxy.
    pub fn new(name: impl Into<String>, seed: u64) -> Self {
        Self {
            systems: HashMap::new(),
            next_system_id: 1,
            warp_lanes: Vec::new(),
            name: name.into(),
            seed,
            total_players: 0,
            stats: GalaxyStats::default(),
        }
    }

    /// Add a solar system and return its ID.
    pub fn add_system(&mut self, mut system: SolarSystem) -> SystemId {
        let id = self.next_system_id;
        self.next_system_id += 1;
        system.id = id;
        self.systems.insert(id, system);
        id
    }

    /// Get a system by ID.
    pub fn get_system(&self, id: SystemId) -> Option<&SolarSystem> {
        self.systems.get(&id)
    }

    /// Get a mutable system.
    pub fn get_system_mut(&mut self, id: SystemId) -> Option<&mut SolarSystem> {
        self.systems.get_mut(&id)
    }

    /// Create a bidirectional warp lane between two systems.
    pub fn create_warp_lane(&mut self, a: SystemId, b: SystemId) -> bool {
        if !self.systems.contains_key(&a) || !self.systems.contains_key(&b) {
            return false;
        }
        if a == b {
            return false;
        }

        // Add connection to both systems.
        if let Some(sys) = self.systems.get_mut(&a) {
            sys.connect_to(b);
        }
        if let Some(sys) = self.systems.get_mut(&b) {
            sys.connect_to(a);
        }

        self.warp_lanes.push((a, b));
        true
    }

    /// Find the shortest path between two systems using BFS.
    pub fn find_route(&self, from: SystemId, to: SystemId) -> Option<Vec<SystemId>> {
        if from == to {
            return Some(vec![from]);
        }

        let mut visited: HashSet<SystemId> = HashSet::new();
        let mut queue: VecDeque<(SystemId, Vec<SystemId>)> = VecDeque::new();

        visited.insert(from);
        queue.push_back((from, vec![from]));

        while let Some((current, path)) = queue.pop_front() {
            if let Some(system) = self.systems.get(&current) {
                for &neighbor in &system.connections {
                    if neighbor == to {
                        let mut full_path = path.clone();
                        full_path.push(to);
                        return Some(full_path);
                    }

                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        let mut new_path = path.clone();
                        new_path.push(neighbor);
                        queue.push_back((neighbor, new_path));
                    }
                }
            }
        }

        None // No route found.
    }

    /// Distance between two systems in galaxy space.
    pub fn system_distance(&self, a: SystemId, b: SystemId) -> Option<f64> {
        let sys_a = self.systems.get(&a)?;
        let sys_b = self.systems.get(&b)?;
        Some(GalaxyPosition::distance(sys_a.position, sys_b.position))
    }

    /// Update LOD levels for all systems based on player presence.
    pub fn update_lod(&mut self) {
        let occupied: HashSet<SystemId> = self
            .systems
            .iter()
            .filter(|(_, s)| s.is_occupied())
            .map(|(&id, _)| id)
            .collect();

        // Gather neighbor sets for occupied systems.
        let mut near_occupied: HashSet<SystemId> = HashSet::new();
        for &sys_id in &occupied {
            if let Some(sys) = self.systems.get(&sys_id) {
                for &conn in &sys.connections {
                    near_occupied.insert(conn);
                }
            }
        }

        for (id, system) in &mut self.systems {
            if occupied.contains(id) {
                system.lod = SystemLOD::Full;
            } else if near_occupied.contains(id) {
                system.lod = SystemLOD::Reduced;
            } else {
                system.lod = SystemLOD::Dormant;
            }
        }
    }

    /// Update all systems that need simulation.
    pub fn update(&mut self, dt: f64) {
        self.total_players = 0;

        for system in self.systems.values_mut() {
            self.total_players += system.player_count;

            match system.lod {
                SystemLOD::Full => {
                    system.update_orbits(dt);
                    system.last_update = dt; // Would be absolute time in production.
                }
                SystemLOD::Reduced => {
                    // Update orbits at reduced frequency.
                    system.update_orbits(dt);
                }
                SystemLOD::Minimal | SystemLOD::Dormant => {
                    // No orbital updates.
                }
            }
        }

        // Update statistics.
        self.update_stats();
    }

    /// Update statistics.
    fn update_stats(&mut self) {
        self.stats.total_systems = self.systems.len();
        self.stats.occupied_systems = self.systems.values().filter(|s| s.is_occupied()).count();
        self.stats.full_sim_systems = self
            .systems
            .values()
            .filter(|s| s.lod == SystemLOD::Full)
            .count();
        self.stats.reduced_sim_systems = self
            .systems
            .values()
            .filter(|s| s.lod == SystemLOD::Reduced)
            .count();
        self.stats.dormant_systems = self
            .systems
            .values()
            .filter(|s| s.lod == SystemLOD::Dormant)
            .count();
        self.stats.total_warp_lanes = self.warp_lanes.len();
        self.stats.total_celestial_bodies =
            self.systems.values().map(|s| s.bodies.len()).sum();
    }

    /// Find the nearest system to a galaxy position.
    pub fn nearest_system(&self, pos: GalaxyPosition) -> Option<SystemId> {
        self.systems
            .iter()
            .min_by(|a, b| {
                let da = GalaxyPosition::distance_sq(pos, a.1.position);
                let db = GalaxyPosition::distance_sq(pos, b.1.position);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(&id, _)| id)
    }

    /// Get all systems within a radius.
    pub fn systems_in_radius(&self, center: GalaxyPosition, radius: f64) -> Vec<SystemId> {
        let r_sq = radius * radius;
        self.systems
            .iter()
            .filter(|(_, s)| GalaxyPosition::distance_sq(center, s.position) <= r_sq)
            .map(|(&id, _)| id)
            .collect()
    }

    /// System count.
    pub fn system_count(&self) -> usize {
        self.systems.len()
    }

    /// Remove a system.
    pub fn remove_system(&mut self, id: SystemId) -> Option<SolarSystem> {
        let system = self.systems.remove(&id)?;

        // Remove warp lanes.
        self.warp_lanes.retain(|&(a, b)| a != id && b != id);

        // Remove connections from other systems.
        for sys in self.systems.values_mut() {
            sys.connections.retain(|&c| c != id);
        }

        Some(system)
    }
}

// ===========================================================================
// InstanceManager — server instancing
// ===========================================================================

/// State of a server instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceState {
    /// Starting up.
    SpinningUp,
    /// Active and accepting players.
    Active,
    /// Full, not accepting new players.
    Full,
    /// Shutting down (draining players).
    SpinningDown,
    /// Stopped.
    Stopped,
}

/// A server instance (isolated world shard).
#[derive(Debug, Clone)]
pub struct Instance {
    /// Instance ID.
    pub id: InstanceId,
    /// Name.
    pub name: String,
    /// Current state.
    pub state: InstanceState,
    /// Maximum player count.
    pub max_players: usize,
    /// Current players.
    pub players: HashSet<PlayerId>,
    /// Associated solar system (if any).
    pub system_id: Option<SystemId>,
    /// Server address/endpoint.
    pub endpoint: String,
    /// Creation timestamp.
    pub created_at: f64,
    /// Last activity timestamp.
    pub last_activity: f64,
    /// CPU usage estimate (0-1).
    pub cpu_usage: f32,
    /// Memory usage (bytes).
    pub memory_usage: u64,
    /// Tick rate.
    pub tick_rate: u32,
    /// Custom metadata.
    pub metadata: HashMap<String, String>,
}

impl Instance {
    /// Create a new instance.
    pub fn new(id: InstanceId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            state: InstanceState::SpinningUp,
            max_players: DEFAULT_INSTANCE_PLAYER_LIMIT,
            players: HashSet::new(),
            system_id: None,
            endpoint: String::new(),
            created_at: 0.0,
            last_activity: 0.0,
            cpu_usage: 0.0,
            memory_usage: 0,
            tick_rate: 60,
            metadata: HashMap::new(),
        }
    }

    /// Whether the instance can accept a new player.
    pub fn can_accept(&self) -> bool {
        self.state == InstanceState::Active && self.players.len() < self.max_players
    }

    /// Add a player.
    pub fn add_player(&mut self, player_id: PlayerId) -> bool {
        if !self.can_accept() {
            return false;
        }
        self.players.insert(player_id);
        self.last_activity = self.last_activity; // Would be current time.
        if self.players.len() >= self.max_players {
            self.state = InstanceState::Full;
        }
        true
    }

    /// Remove a player.
    pub fn remove_player(&mut self, player_id: PlayerId) -> bool {
        let removed = self.players.remove(&player_id);
        if removed && self.state == InstanceState::Full {
            self.state = InstanceState::Active;
        }
        removed
    }

    /// Player count.
    pub fn player_count(&self) -> usize {
        self.players.len()
    }

    /// Whether the instance is empty.
    pub fn is_empty(&self) -> bool {
        self.players.is_empty()
    }

    /// Whether the instance should be considered for shutdown.
    pub fn should_spin_down(&self, idle_timeout: f64, current_time: f64) -> bool {
        self.is_empty() && (current_time - self.last_activity) > idle_timeout
    }
}

/// Cross-instance travel request.
#[derive(Debug, Clone)]
pub struct TransferRequest {
    /// Player being transferred.
    pub player_id: PlayerId,
    /// Source instance.
    pub from_instance: InstanceId,
    /// Target instance.
    pub to_instance: InstanceId,
    /// Serialized player state.
    pub player_state: Vec<u8>,
    /// Request timestamp.
    pub timestamp: f64,
    /// Whether the transfer is in progress.
    pub in_progress: bool,
    /// Whether the transfer completed.
    pub completed: bool,
}

/// Manages all server instances.
#[derive(Debug)]
pub struct InstanceManager {
    /// All instances.
    instances: HashMap<InstanceId, Instance>,
    /// Next instance ID.
    next_id: InstanceId,
    /// Pending transfer requests.
    pub pending_transfers: Vec<TransferRequest>,
    /// Idle timeout before spinning down empty instances (seconds).
    pub idle_timeout: f64,
    /// Statistics.
    pub stats: InstanceManagerStats,
}

/// Instance manager statistics.
#[derive(Debug, Clone, Default)]
pub struct InstanceManagerStats {
    pub total_instances: usize,
    pub active_instances: usize,
    pub total_players: usize,
    pub pending_transfers: usize,
    pub avg_load: f32,
}

impl InstanceManager {
    /// Create a new instance manager.
    pub fn new() -> Self {
        Self {
            instances: HashMap::new(),
            next_id: 1,
            pending_transfers: Vec::new(),
            idle_timeout: 300.0, // 5 minutes
            stats: InstanceManagerStats::default(),
        }
    }

    /// Spin up a new instance.
    pub fn create_instance(&mut self, name: impl Into<String>) -> InstanceId {
        let id = self.next_id;
        self.next_id += 1;
        let instance = Instance::new(id, name);
        self.instances.insert(id, instance);
        id
    }

    /// Activate an instance (transition from SpinningUp to Active).
    pub fn activate_instance(&mut self, id: InstanceId) -> bool {
        if let Some(inst) = self.instances.get_mut(&id) {
            if inst.state == InstanceState::SpinningUp {
                inst.state = InstanceState::Active;
                return true;
            }
        }
        false
    }

    /// Spin down an instance.
    pub fn spin_down(&mut self, id: InstanceId) -> bool {
        if let Some(inst) = self.instances.get_mut(&id) {
            inst.state = InstanceState::SpinningDown;
            true
        } else {
            false
        }
    }

    /// Remove a stopped instance.
    pub fn remove_instance(&mut self, id: InstanceId) -> Option<Instance> {
        self.instances.remove(&id)
    }

    /// Get an instance.
    pub fn get_instance(&self, id: InstanceId) -> Option<&Instance> {
        self.instances.get(&id)
    }

    /// Get a mutable instance.
    pub fn get_instance_mut(&mut self, id: InstanceId) -> Option<&mut Instance> {
        self.instances.get_mut(&id)
    }

    /// Find the best instance for a player (least loaded active instance
    /// associated with the target system).
    pub fn find_instance_for_system(&self, system_id: SystemId) -> Option<InstanceId> {
        self.instances
            .iter()
            .filter(|(_, inst)| {
                inst.state == InstanceState::Active
                    && inst.system_id == Some(system_id)
                    && inst.can_accept()
            })
            .min_by_key(|(_, inst)| inst.player_count())
            .map(|(&id, _)| id)
    }

    /// Request a cross-instance transfer.
    pub fn request_transfer(
        &mut self,
        player_id: PlayerId,
        from: InstanceId,
        to: InstanceId,
        player_state: Vec<u8>,
        timestamp: f64,
    ) -> bool {
        // Validate instances exist and target can accept.
        if !self.instances.contains_key(&from) {
            return false;
        }
        if let Some(target) = self.instances.get(&to) {
            if !target.can_accept() {
                return false;
            }
        } else {
            return false;
        }

        self.pending_transfers.push(TransferRequest {
            player_id,
            from_instance: from,
            to_instance: to,
            player_state,
            timestamp,
            in_progress: false,
            completed: false,
        });

        true
    }

    /// Process pending transfers.
    pub fn process_transfers(&mut self) -> Vec<TransferRequest> {
        let mut completed = Vec::new();

        self.pending_transfers.retain_mut(|transfer| {
            if transfer.completed {
                completed.push(transfer.clone());
                return false;
            }

            if !transfer.in_progress {
                // Start the transfer: remove from source, add to target.
                if let Some(source) = self.instances.get_mut(&transfer.from_instance) {
                    source.remove_player(transfer.player_id);
                }
                if let Some(target) = self.instances.get_mut(&transfer.to_instance) {
                    target.add_player(transfer.player_id);
                }
                transfer.in_progress = true;
                transfer.completed = true;
            }

            !transfer.completed
        });

        completed
    }

    /// Update the manager: check for idle instances, process transfers.
    pub fn update(&mut self, current_time: f64) {
        // Find instances that should be spun down.
        let idle_timeout = self.idle_timeout;
        let to_spin_down: Vec<InstanceId> = self
            .instances
            .iter()
            .filter(|(_, inst)| {
                inst.state == InstanceState::Active
                    && inst.should_spin_down(idle_timeout, current_time)
            })
            .map(|(&id, _)| id)
            .collect();

        for id in to_spin_down {
            self.spin_down(id);
        }

        // Process transfers.
        self.process_transfers();

        // Update stats.
        self.update_stats();
    }

    /// Update statistics.
    fn update_stats(&mut self) {
        self.stats.total_instances = self.instances.len();
        self.stats.active_instances = self
            .instances
            .values()
            .filter(|i| i.state == InstanceState::Active || i.state == InstanceState::Full)
            .count();
        self.stats.total_players = self
            .instances
            .values()
            .map(|i| i.player_count())
            .sum();
        self.stats.pending_transfers = self.pending_transfers.len();

        if !self.instances.is_empty() {
            let total_load: f32 = self
                .instances
                .values()
                .map(|i| i.player_count() as f32 / i.max_players as f32)
                .sum();
            self.stats.avg_load = total_load / self.instances.len() as f32;
        }
    }

    /// Instance count.
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Total players across all instances.
    pub fn total_players(&self) -> usize {
        self.stats.total_players
    }
}

impl Default for InstanceManager {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// SpatialDatabase — efficient large-world queries
// ===========================================================================

/// An entry in the spatial database.
#[derive(Debug, Clone)]
pub struct SpatialEntry {
    /// Entity ID.
    pub entity_id: SpatialEntityId,
    /// Position.
    pub position: GalaxyPosition,
    /// Bounding radius (for sphere queries).
    pub radius: f64,
    /// Entity type tag (for filtered queries).
    pub type_tag: u32,
}

/// Cell coordinate for the spatial hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SpatialHashCell {
    x: i64,
    y: i64,
    z: i64,
    level: u8,
}

/// Hierarchical spatial hash for efficient multi-scale queries.
///
/// Uses multiple grid levels with geometrically increasing cell sizes.
/// Level 0 = smallest cells, Level N = largest cells.
#[derive(Debug)]
pub struct SpatialDatabase {
    /// Cell sizes for each level.
    cell_sizes: Vec<f64>,
    /// Number of levels.
    levels: usize,
    /// Per-level grid: cell -> entity list.
    grids: Vec<HashMap<SpatialHashCell, Vec<SpatialEntry>>>,
    /// Entity -> level mapping for fast removal.
    entity_levels: HashMap<SpatialEntityId, u8>,
    /// Total entities.
    entity_count: usize,
    /// Statistics.
    pub stats: SpatialDatabaseStats,
}

/// Spatial database statistics.
#[derive(Debug, Clone, Default)]
pub struct SpatialDatabaseStats {
    pub total_entities: usize,
    pub total_cells: usize,
    pub queries_performed: u64,
    pub avg_query_results: f32,
}

impl SpatialDatabase {
    /// Create a new hierarchical spatial hash.
    pub fn new(levels: usize, min_cell_size: f64) -> Self {
        let mut cell_sizes = Vec::with_capacity(levels);
        let mut size = min_cell_size;
        for _ in 0..levels {
            cell_sizes.push(size);
            size *= 4.0; // Each level is 4x the previous.
        }

        let grids = (0..levels).map(|_| HashMap::new()).collect();

        Self {
            cell_sizes,
            levels,
            grids,
            entity_levels: HashMap::new(),
            entity_count: 0,
            stats: SpatialDatabaseStats::default(),
        }
    }

    /// Create with default settings.
    pub fn with_defaults() -> Self {
        Self::new(SPATIAL_HASH_LEVELS, SPATIAL_HASH_MIN_CELL)
    }

    /// Determine the best level for an entity based on its radius.
    fn level_for_radius(&self, radius: f64) -> u8 {
        for (i, &cell_size) in self.cell_sizes.iter().enumerate() {
            if radius <= cell_size * 0.5 {
                return i as u8;
            }
        }
        (self.levels - 1) as u8
    }

    /// Convert a position to a cell coordinate at a given level.
    fn pos_to_cell(&self, pos: &GalaxyPosition, level: u8) -> SpatialHashCell {
        let size = self.cell_sizes[level as usize];
        SpatialHashCell {
            x: (pos.x / size).floor() as i64,
            y: (pos.y / size).floor() as i64,
            z: (pos.z / size).floor() as i64,
            level,
        }
    }

    /// Insert or update an entity.
    pub fn insert(&mut self, entry: SpatialEntry) {
        let level = self.level_for_radius(entry.radius);

        // Remove from old position if it exists.
        self.remove(entry.entity_id);

        let cell = self.pos_to_cell(&entry.position, level);
        self.grids[level as usize]
            .entry(cell)
            .or_default()
            .push(entry.clone());

        self.entity_levels.insert(entry.entity_id, level);
        self.entity_count += 1;
        self.stats.total_entities = self.entity_count;
    }

    /// Remove an entity.
    pub fn remove(&mut self, entity_id: SpatialEntityId) -> bool {
        if let Some(level) = self.entity_levels.remove(&entity_id) {
            let grid = &mut self.grids[level as usize];
            for cell_entries in grid.values_mut() {
                if let Some(pos) = cell_entries.iter().position(|e| e.entity_id == entity_id) {
                    cell_entries.swap_remove(pos);
                    self.entity_count -= 1;
                    self.stats.total_entities = self.entity_count;
                    return true;
                }
            }
        }
        false
    }

    /// Range query: find all entities within a radius of a center point.
    pub fn query_range(
        &mut self,
        center: GalaxyPosition,
        radius: f64,
        max_results: usize,
    ) -> Vec<(SpatialEntityId, f64)> {
        self.stats.queries_performed += 1;
        let r_sq = radius * radius;
        let mut results: Vec<(SpatialEntityId, f64)> = Vec::new();

        // Query all levels (larger entities may only be in higher levels).
        for level in 0..self.levels {
            let cell_size = self.cell_sizes[level];
            let cell_radius = (radius / cell_size).ceil() as i64;

            let center_cell = self.pos_to_cell(&center, level as u8);

            for dx in -cell_radius..=cell_radius {
                for dy in -cell_radius..=cell_radius {
                    for dz in -cell_radius..=cell_radius {
                        let cell = SpatialHashCell {
                            x: center_cell.x + dx,
                            y: center_cell.y + dy,
                            z: center_cell.z + dz,
                            level: level as u8,
                        };

                        if let Some(entries) = self.grids[level].get(&cell) {
                            for entry in entries {
                                let dist_sq =
                                    GalaxyPosition::distance_sq(center, entry.position);
                                if dist_sq <= r_sq {
                                    results.push((entry.entity_id, dist_sq.sqrt()));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sort by distance and truncate.
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(max_results);

        if !results.is_empty() {
            self.stats.avg_query_results = self.stats.avg_query_results * 0.95
                + results.len() as f32 * 0.05;
        }

        results
    }

    /// Nearest-neighbor query.
    pub fn nearest(
        &mut self,
        center: GalaxyPosition,
        max_distance: f64,
    ) -> Option<(SpatialEntityId, f64)> {
        let results = self.query_range(center, max_distance, 1);
        results.into_iter().next()
    }

    /// Count entities in a region (aggregate query).
    pub fn count_in_range(&self, center: GalaxyPosition, radius: f64) -> usize {
        let r_sq = radius * radius;
        let mut count = 0;

        for level in 0..self.levels {
            let cell_size = self.cell_sizes[level];
            let cell_radius = (radius / cell_size).ceil() as i64;
            let center_cell = self.pos_to_cell(&center, level as u8);

            for dx in -cell_radius..=cell_radius {
                for dy in -cell_radius..=cell_radius {
                    for dz in -cell_radius..=cell_radius {
                        let cell = SpatialHashCell {
                            x: center_cell.x + dx,
                            y: center_cell.y + dy,
                            z: center_cell.z + dz,
                            level: level as u8,
                        };

                        if let Some(entries) = self.grids[level].get(&cell) {
                            for entry in entries {
                                if GalaxyPosition::distance_sq(center, entry.position) <= r_sq {
                                    count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        count
    }

    /// Query by type tag within a range.
    pub fn query_by_type(
        &mut self,
        center: GalaxyPosition,
        radius: f64,
        type_tag: u32,
        max_results: usize,
    ) -> Vec<(SpatialEntityId, f64)> {
        let mut all = self.query_range(center, radius, max_results * 10);
        // Filter by type -- we need access to the entries to check type.
        // For efficiency, collect all type-matching entity IDs first.
        let type_ids: HashSet<SpatialEntityId> = self
            .grids
            .iter()
            .flat_map(|grid| grid.values())
            .flat_map(|entries| entries.iter())
            .filter(|e| e.type_tag == type_tag)
            .map(|e| e.entity_id)
            .collect();

        all.retain(|(id, _)| type_ids.contains(id));
        all.truncate(max_results);
        all
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        for grid in &mut self.grids {
            grid.clear();
        }
        self.entity_levels.clear();
        self.entity_count = 0;
        self.stats.total_entities = 0;
    }

    /// Total entity count.
    pub fn entity_count(&self) -> usize {
        self.entity_count
    }

    /// Total occupied cells across all levels.
    pub fn cell_count(&self) -> usize {
        self.grids.iter().map(|g| g.len()).sum()
    }
}

// ===========================================================================
// WorldPersistence — save massive world state
// ===========================================================================

/// State of a save operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveState {
    /// No save in progress.
    Idle,
    /// Save is being prepared.
    Preparing,
    /// Save is in progress (background thread).
    Saving,
    /// Save completed successfully.
    Completed,
    /// Save failed.
    Failed,
}

/// A saved sector's data.
#[derive(Debug, Clone)]
pub struct SectorSaveData {
    /// Sector ID.
    pub sector_id: SectorId,
    /// Serialized sector state.
    pub data: Vec<u8>,
    /// Save version.
    pub version: u32,
    /// Timestamp when this data was captured.
    pub timestamp: f64,
    /// Checksum (CRC32).
    pub checksum: u32,
    /// Whether the data has been flushed to disk.
    pub flushed: bool,
    /// Size in bytes.
    pub size_bytes: u64,
}

impl SectorSaveData {
    /// Create new save data.
    pub fn new(sector_id: SectorId, data: Vec<u8>, timestamp: f64) -> Self {
        let size = data.len() as u64;
        let checksum = Self::compute_checksum(&data);
        Self {
            sector_id,
            data,
            version: SAVE_FORMAT_VERSION,
            timestamp,
            checksum,
            flushed: false,
            size_bytes: size,
        }
    }

    /// Compute a simple CRC32-like checksum.
    fn compute_checksum(data: &[u8]) -> u32 {
        let mut hash: u32 = 0xFFFF_FFFF;
        for &byte in data {
            hash ^= byte as u32;
            for _ in 0..8 {
                if hash & 1 == 1 {
                    hash = (hash >> 1) ^ 0xEDB8_8320;
                } else {
                    hash >>= 1;
                }
            }
        }
        !hash
    }

    /// Verify the checksum.
    pub fn verify_checksum(&self) -> bool {
        Self::compute_checksum(&self.data) == self.checksum
    }
}

/// Save header for a complete world save.
#[derive(Debug, Clone)]
pub struct SaveHeader {
    /// Save version.
    pub version: u32,
    /// Galaxy name.
    pub galaxy_name: String,
    /// Number of sectors.
    pub sector_count: u32,
    /// Total save size (bytes).
    pub total_size: u64,
    /// Timestamp.
    pub timestamp: f64,
    /// Save slot name.
    pub slot_name: String,
    /// Description.
    pub description: String,
    /// Play time (seconds).
    pub play_time: f64,
}

impl SaveHeader {
    /// Create a new save header.
    pub fn new(galaxy_name: impl Into<String>, slot_name: impl Into<String>) -> Self {
        Self {
            version: SAVE_FORMAT_VERSION,
            galaxy_name: galaxy_name.into(),
            sector_count: 0,
            total_size: 0,
            timestamp: 0.0,
            slot_name: slot_name.into(),
            description: String::new(),
            play_time: 0.0,
        }
    }
}

/// Manages persistent storage of the massive world.
#[derive(Debug)]
pub struct WorldPersistence {
    /// Dirty sectors that need saving.
    dirty_sectors: HashMap<SectorId, SectorSaveData>,
    /// Save version history.
    pub save_versions: Vec<SaveHeader>,
    /// Current save state.
    pub state: SaveState,
    /// Time since last flush.
    pub time_since_flush: f64,
    /// Flush interval.
    pub flush_interval: f64,
    /// Maximum sectors to save per flush.
    pub max_sectors_per_flush: usize,
    /// Total bytes saved.
    pub total_bytes_saved: u64,
    /// Total sectors saved.
    pub total_sectors_saved: u64,
    /// Background save queue.
    save_queue: VecDeque<SectorSaveData>,
    /// Statistics.
    pub stats: PersistenceStats,
}

/// Persistence statistics.
#[derive(Debug, Clone, Default)]
pub struct PersistenceStats {
    pub dirty_sector_count: usize,
    pub queued_saves: usize,
    pub total_saves: u64,
    pub total_bytes: u64,
    pub last_save_duration_ms: f64,
    pub saves_per_minute: f32,
}

impl WorldPersistence {
    /// Create a new persistence manager.
    pub fn new() -> Self {
        Self {
            dirty_sectors: HashMap::new(),
            save_versions: Vec::new(),
            state: SaveState::Idle,
            time_since_flush: 0.0,
            flush_interval: SAVE_FLUSH_INTERVAL,
            max_sectors_per_flush: MAX_SECTORS_PER_FLUSH,
            total_bytes_saved: 0,
            total_sectors_saved: 0,
            save_queue: VecDeque::new(),
            stats: PersistenceStats::default(),
        }
    }

    /// Mark a sector as dirty (needs saving).
    pub fn mark_dirty(&mut self, sector_id: SectorId, data: Vec<u8>, timestamp: f64) {
        let save_data = SectorSaveData::new(sector_id, data, timestamp);
        self.dirty_sectors.insert(sector_id, save_data);
    }

    /// Check if a sector is dirty.
    pub fn is_dirty(&self, sector_id: SectorId) -> bool {
        self.dirty_sectors.contains_key(&sector_id)
    }

    /// Get the number of dirty sectors.
    pub fn dirty_count(&self) -> usize {
        self.dirty_sectors.len()
    }

    /// Queue dirty sectors for background saving.
    pub fn flush(&mut self) -> Vec<SectorId> {
        let mut flushed = Vec::new();
        let count = self.max_sectors_per_flush.min(self.dirty_sectors.len());

        let sector_ids: Vec<SectorId> = self.dirty_sectors.keys().take(count).copied().collect();

        for id in sector_ids {
            if let Some(save_data) = self.dirty_sectors.remove(&id) {
                self.total_bytes_saved += save_data.size_bytes;
                self.total_sectors_saved += 1;
                self.save_queue.push_back(save_data);
                flushed.push(id);
            }
        }

        self.state = if self.save_queue.is_empty() {
            SaveState::Idle
        } else {
            SaveState::Saving
        };

        flushed
    }

    /// Process the background save queue.
    ///
    /// In a real implementation, this would write to disk in a background thread.
    /// Here, we simulate completion.
    pub fn process_save_queue(&mut self) -> Vec<SectorId> {
        let mut completed = Vec::new();

        while let Some(mut save_data) = self.save_queue.pop_front() {
            save_data.flushed = true;
            completed.push(save_data.sector_id);
            self.stats.total_saves += 1;
            self.stats.total_bytes += save_data.size_bytes;
        }

        if self.save_queue.is_empty() && !completed.is_empty() {
            self.state = SaveState::Completed;
        }

        completed
    }

    /// Update the persistence system.
    pub fn update(&mut self, dt: f64) -> Vec<SectorId> {
        self.time_since_flush += dt;

        let mut saved = Vec::new();

        if self.time_since_flush >= self.flush_interval && !self.dirty_sectors.is_empty() {
            self.time_since_flush = 0.0;
            let flushed = self.flush();
            saved.extend(flushed);

            // Process immediately in this simplified implementation.
            let completed = self.process_save_queue();
            saved.extend(completed);
        }

        // Update stats.
        self.stats.dirty_sector_count = self.dirty_sectors.len();
        self.stats.queued_saves = self.save_queue.len();

        saved
    }

    /// Create a full world save header.
    pub fn create_save_header(
        &self,
        galaxy_name: &str,
        slot_name: &str,
        timestamp: f64,
        play_time: f64,
    ) -> SaveHeader {
        let mut header = SaveHeader::new(galaxy_name, slot_name);
        header.sector_count = self.total_sectors_saved as u32;
        header.total_size = self.total_bytes_saved;
        header.timestamp = timestamp;
        header.play_time = play_time;
        header
    }

    /// Record a save header in version history.
    pub fn record_save(&mut self, header: SaveHeader) {
        self.save_versions.push(header);
    }

    /// Get the latest save header.
    pub fn latest_save(&self) -> Option<&SaveHeader> {
        self.save_versions.last()
    }

    /// Number of save versions.
    pub fn save_version_count(&self) -> usize {
        self.save_versions.len()
    }

    /// Clear all dirty data and queues.
    pub fn clear(&mut self) {
        self.dirty_sectors.clear();
        self.save_queue.clear();
        self.state = SaveState::Idle;
        self.time_since_flush = 0.0;
    }
}

impl Default for WorldPersistence {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// MassiveWorldSystem — top-level coordinator
// ===========================================================================

/// Coordinates all massive-world subsystems.
#[derive(Debug)]
pub struct MassiveWorldSystem {
    /// Galaxy map.
    pub galaxy: GalaxyMap,
    /// Instance manager.
    pub instances: InstanceManager,
    /// Spatial database.
    pub spatial_db: SpatialDatabase,
    /// World persistence.
    pub persistence: WorldPersistence,
    /// Current time.
    pub time: f64,
}

impl MassiveWorldSystem {
    /// Create a new massive world system.
    pub fn new(galaxy_name: &str, galaxy_seed: u64) -> Self {
        Self {
            galaxy: GalaxyMap::new(galaxy_name, galaxy_seed),
            instances: InstanceManager::new(),
            spatial_db: SpatialDatabase::with_defaults(),
            persistence: WorldPersistence::new(),
            time: 0.0,
        }
    }

    /// Main update tick.
    pub fn update(&mut self, dt: f64) {
        self.time += dt;

        // Update galaxy (orbits, LOD).
        self.galaxy.update(dt);
        self.galaxy.update_lod();

        // Update instances.
        self.instances.update(self.time);

        // Update persistence.
        self.persistence.update(dt);
    }

    /// Add a solar system with standard bodies (star + random planets).
    pub fn add_standard_system(
        &mut self,
        name: &str,
        position: GalaxyPosition,
        planet_count: usize,
    ) -> SystemId {
        let mut system = SolarSystem::new(0, name, position);

        // Add a star.
        system.add_body(CelestialBody::star(
            format!("{} Star", name),
            100_000.0,
            1.0,
        ));

        // Add planets at increasing orbital radii.
        for i in 0..planet_count {
            let orbital_radius = (i + 1) as f64 * 50_000_000.0; // 50M km per AU slot
            let orbital_period = ((i + 1) as f64).powf(1.5) * 365.0 * 24.0 * 3600.0;
            system.add_body(CelestialBody::planet(
                format!("{} {}", name, ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
                    .get(i)
                    .unwrap_or(&"IX")),
                5000.0 + i as f64 * 2000.0,
                orbital_radius,
                orbital_period,
            ));
        }

        // Add a station.
        system.add_body(CelestialBody::station(
            format!("{} Station", name),
            60_000_000.0,
        ));

        self.galaxy.add_system(system)
    }

    /// Get the total player count across all instances.
    pub fn total_players(&self) -> usize {
        self.instances.total_players()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_galaxy_map_routing() {
        let mut galaxy = GalaxyMap::new("TestGalaxy", 42);

        let s1 = galaxy.add_system(SolarSystem::new(0, "Sol", GalaxyPosition::new(0.0, 0.0, 0.0)));
        let s2 = galaxy.add_system(SolarSystem::new(0, "Alpha", GalaxyPosition::new(10.0, 0.0, 0.0)));
        let s3 = galaxy.add_system(SolarSystem::new(0, "Beta", GalaxyPosition::new(20.0, 0.0, 0.0)));
        let s4 = galaxy.add_system(SolarSystem::new(0, "Gamma", GalaxyPosition::new(30.0, 0.0, 0.0)));

        galaxy.create_warp_lane(s1, s2);
        galaxy.create_warp_lane(s2, s3);
        galaxy.create_warp_lane(s3, s4);

        // Route from Sol to Gamma.
        let route = galaxy.find_route(s1, s4);
        assert!(route.is_some());
        let path = route.unwrap();
        assert_eq!(path.len(), 4);
        assert_eq!(path[0], s1);
        assert_eq!(path[3], s4);
    }

    #[test]
    fn test_galaxy_no_route() {
        let mut galaxy = GalaxyMap::new("TestGalaxy", 42);
        let s1 = galaxy.add_system(SolarSystem::new(0, "Sol", GalaxyPosition::ZERO));
        let s2 = galaxy.add_system(SolarSystem::new(0, "Isolated", GalaxyPosition::new(100.0, 0.0, 0.0)));

        // No warp lane connecting them.
        let route = galaxy.find_route(s1, s2);
        assert!(route.is_none());
    }

    #[test]
    fn test_instance_management() {
        let mut manager = InstanceManager::new();
        let inst1 = manager.create_instance("Instance-1");
        manager.activate_instance(inst1);

        // Add players.
        if let Some(inst) = manager.get_instance_mut(inst1) {
            assert!(inst.add_player(1));
            assert!(inst.add_player(2));
            assert_eq!(inst.player_count(), 2);
        }

        assert_eq!(manager.total_players(), 2);
    }

    #[test]
    fn test_instance_transfer() {
        let mut manager = InstanceManager::new();
        let inst1 = manager.create_instance("From");
        let inst2 = manager.create_instance("To");
        manager.activate_instance(inst1);
        manager.activate_instance(inst2);

        // Add player to inst1.
        manager.get_instance_mut(inst1).unwrap().add_player(42);

        // Request transfer.
        assert!(manager.request_transfer(42, inst1, inst2, vec![1, 2, 3], 0.0));

        // Process.
        let completed = manager.process_transfers();
        assert_eq!(completed.len(), 1);

        // Player should now be in inst2.
        assert!(!manager.get_instance(inst1).unwrap().players.contains(&42));
        assert!(manager.get_instance(inst2).unwrap().players.contains(&42));
    }

    #[test]
    fn test_spatial_database_range_query() {
        let mut db = SpatialDatabase::with_defaults();

        for i in 0..100 {
            db.insert(SpatialEntry {
                entity_id: i,
                position: GalaxyPosition::new(i as f64 * 10.0, 0.0, 0.0),
                radius: 5.0,
                type_tag: 0,
            });
        }

        assert_eq!(db.entity_count(), 100);

        // Query near the origin.
        let results = db.query_range(GalaxyPosition::ZERO, 50.0, 100);
        assert!(!results.is_empty());
        // Should find entities at x=0, 10, 20, 30, 40 (within 50 units).
        assert!(results.len() >= 5);
    }

    #[test]
    fn test_spatial_database_nearest() {
        let mut db = SpatialDatabase::with_defaults();

        db.insert(SpatialEntry {
            entity_id: 1,
            position: GalaxyPosition::new(100.0, 0.0, 0.0),
            radius: 5.0,
            type_tag: 0,
        });
        db.insert(SpatialEntry {
            entity_id: 2,
            position: GalaxyPosition::new(50.0, 0.0, 0.0),
            radius: 5.0,
            type_tag: 0,
        });

        let nearest = db.nearest(GalaxyPosition::ZERO, 200.0);
        assert!(nearest.is_some());
        assert_eq!(nearest.unwrap().0, 2); // Closer entity.
    }

    #[test]
    fn test_world_persistence() {
        let mut persistence = WorldPersistence::new();
        persistence.flush_interval = 0.0; // Flush immediately.

        // Mark some sectors dirty.
        persistence.mark_dirty(1, vec![1, 2, 3, 4], 1.0);
        persistence.mark_dirty(2, vec![5, 6, 7, 8, 9], 1.0);

        assert_eq!(persistence.dirty_count(), 2);

        // Update should trigger flush.
        let saved = persistence.update(1.0);
        assert!(!saved.is_empty());
        assert_eq!(persistence.dirty_count(), 0);
    }

    #[test]
    fn test_sector_save_checksum() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let save = SectorSaveData::new(1, data, 0.0);
        assert!(save.verify_checksum());

        // Corrupt the data.
        let mut corrupted = save.clone();
        corrupted.data[0] = 99;
        assert!(!corrupted.verify_checksum());
    }

    #[test]
    fn test_celestial_orbit() {
        let mut planet = CelestialBody::planet("TestPlanet", 5000.0, 100.0, 100.0);
        assert_eq!(planet.orbital_angle, 0.0);

        // Simulate half an orbit.
        planet.update_orbit(50.0);
        assert!(planet.orbital_angle > 0.0);
        assert!(planet.local_position.x != 0.0 || planet.local_position.z != 0.0);
    }

    #[test]
    fn test_galaxy_lod_update() {
        let mut galaxy = GalaxyMap::new("TestGalaxy", 42);
        let s1 = galaxy.add_system(SolarSystem::new(0, "Sol", GalaxyPosition::ZERO));
        let s2 = galaxy.add_system(SolarSystem::new(0, "Alpha", GalaxyPosition::new(10.0, 0.0, 0.0)));
        let s3 = galaxy.add_system(SolarSystem::new(0, "Far", GalaxyPosition::new(1000.0, 0.0, 0.0)));

        galaxy.create_warp_lane(s1, s2);

        // Put players in s1.
        galaxy.get_system_mut(s1).unwrap().player_count = 5;

        galaxy.update_lod();

        // s1 should be Full (has players).
        assert_eq!(galaxy.get_system(s1).unwrap().lod, SystemLOD::Full);
        // s2 should be Reduced (neighbor of occupied).
        assert_eq!(galaxy.get_system(s2).unwrap().lod, SystemLOD::Reduced);
        // s3 should be Dormant (far away, no connection).
        assert_eq!(galaxy.get_system(s3).unwrap().lod, SystemLOD::Dormant);
    }

    #[test]
    fn test_massive_world_system() {
        let mut world = MassiveWorldSystem::new("TestUniverse", 12345);

        let s1 = world.add_standard_system("Sol", GalaxyPosition::ZERO, 4);
        let s2 = world.add_standard_system("Proxima", GalaxyPosition::new(4.24, 0.0, 0.0), 2);

        world.galaxy.create_warp_lane(s1, s2);

        // System should have star + 4 planets + 1 station = 6 bodies.
        assert_eq!(world.galaxy.get_system(s1).unwrap().bodies.len(), 6);

        world.update(1.0);
        assert_eq!(world.galaxy.system_count(), 2);
    }
}
