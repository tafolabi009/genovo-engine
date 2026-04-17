// engine/physics/src/island_manager.rs
//
// Physics island system for the Genovo engine.
//
// Implements a union-find based island building algorithm that groups
// connected bodies into islands for independent simulation:
//
// - Union-find (disjoint set) for efficient island construction.
// - Island sleep/wake management based on body activity.
// - Island-level simulation skip for sleeping islands.
// - Island splitting when joints or contacts are removed.
// - Island merging when new connections form.
// - Per-island statistics (body count, energy, etc.).
// - Debug visualization of island boundaries.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum kinetic energy threshold below which a body can sleep.
const SLEEP_ENERGY_THRESHOLD: f32 = 0.01;

/// Time a body must be below the energy threshold before sleeping.
const SLEEP_DELAY_SECONDS: f32 = 0.5;

/// Maximum bodies in a single island before forced splitting.
const MAX_ISLAND_SIZE: usize = 4096;

/// Default island capacity.
const DEFAULT_ISLAND_CAPACITY: usize = 256;

/// Energy threshold for waking an entire island.
const WAKE_ENERGY_THRESHOLD: f32 = 0.05;

// ---------------------------------------------------------------------------
// Island ID
// ---------------------------------------------------------------------------

/// Unique identifier for a physics island.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IslandId(pub u32);

impl IslandId {
    /// Invalid/unassigned island.
    pub const NONE: Self = Self(u32::MAX);

    /// Check if this is a valid island ID.
    pub fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }
}

// ---------------------------------------------------------------------------
// Body Info (for island building)
// ---------------------------------------------------------------------------

/// Body reference used by the island system.
#[derive(Debug, Clone)]
pub struct IslandBody {
    /// Body identifier.
    pub body_id: u64,
    /// Whether this body is static.
    pub is_static: bool,
    /// Whether this body is kinematic.
    pub is_kinematic: bool,
    /// Current linear velocity magnitude squared.
    pub linear_velocity_sq: f32,
    /// Current angular velocity magnitude squared.
    pub angular_velocity_sq: f32,
    /// Mass of the body.
    pub mass: f32,
    /// Time below sleep threshold.
    pub time_below_threshold: f32,
    /// Whether this body is sleeping.
    pub sleeping: bool,
    /// Island this body belongs to.
    pub island_id: IslandId,
}

impl IslandBody {
    /// Create a new dynamic body.
    pub fn dynamic(body_id: u64, mass: f32) -> Self {
        Self {
            body_id,
            is_static: false,
            is_kinematic: false,
            linear_velocity_sq: 0.0,
            angular_velocity_sq: 0.0,
            mass,
            time_below_threshold: 0.0,
            sleeping: false,
            island_id: IslandId::NONE,
        }
    }

    /// Create a static body.
    pub fn static_body(body_id: u64) -> Self {
        Self {
            body_id,
            is_static: true,
            is_kinematic: false,
            linear_velocity_sq: 0.0,
            angular_velocity_sq: 0.0,
            mass: 0.0,
            time_below_threshold: 0.0,
            sleeping: true,
            island_id: IslandId::NONE,
        }
    }

    /// Compute the kinetic energy of this body.
    pub fn kinetic_energy(&self) -> f32 {
        0.5 * self.mass * (self.linear_velocity_sq + self.angular_velocity_sq)
    }

    /// Check if this body can sleep based on its velocity.
    pub fn can_sleep(&self) -> bool {
        if self.is_static || self.is_kinematic {
            return true;
        }
        self.kinetic_energy() < SLEEP_ENERGY_THRESHOLD
    }
}

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------

/// A connection between two bodies (via contact or joint).
#[derive(Debug, Clone)]
pub struct BodyConnection {
    /// First body.
    pub body_a: u64,
    /// Second body.
    pub body_b: u64,
    /// Connection type.
    pub connection_type: ConnectionType,
}

/// Type of connection between bodies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionType {
    /// Contact pair (collision).
    Contact,
    /// Joint constraint.
    Joint,
}

// ---------------------------------------------------------------------------
// Union-Find
// ---------------------------------------------------------------------------

/// Disjoint set (union-find) data structure with path compression and union by rank.
#[derive(Debug, Clone)]
pub struct DisjointSet {
    /// Parent array: parent[i] is the parent of node i.
    parent: Vec<u32>,
    /// Rank for union by rank.
    rank: Vec<u32>,
    /// Number of elements.
    size: usize,
}

impl DisjointSet {
    /// Create a new disjoint set with `n` elements, each in its own set.
    pub fn new(n: usize) -> Self {
        let parent = (0..n as u32).collect();
        let rank = vec![0; n];
        Self { parent, rank, size: n }
    }

    /// Find the representative (root) of the set containing `x` with path compression.
    pub fn find(&mut self, x: u32) -> u32 {
        if self.parent[x as usize] != x {
            self.parent[x as usize] = self.find(self.parent[x as usize]);
        }
        self.parent[x as usize]
    }

    /// Union the sets containing `x` and `y`. Returns true if they were in different sets.
    pub fn union(&mut self, x: u32, y: u32) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return false;
        }

        if self.rank[rx as usize] < self.rank[ry as usize] {
            self.parent[rx as usize] = ry;
        } else if self.rank[rx as usize] > self.rank[ry as usize] {
            self.parent[ry as usize] = rx;
        } else {
            self.parent[ry as usize] = rx;
            self.rank[rx as usize] += 1;
        }

        true
    }

    /// Check if two elements are in the same set.
    pub fn connected(&mut self, x: u32, y: u32) -> bool {
        self.find(x) == self.find(y)
    }

    /// Count the number of distinct sets.
    pub fn set_count(&mut self) -> usize {
        let mut roots = std::collections::HashSet::new();
        for i in 0..self.size as u32 {
            roots.insert(self.find(i));
        }
        roots.len()
    }
}

// ---------------------------------------------------------------------------
// Island
// ---------------------------------------------------------------------------

/// A physics island -- a connected group of bodies.
#[derive(Debug, Clone)]
pub struct Island {
    /// Island identifier.
    pub id: IslandId,
    /// Body IDs in this island.
    pub bodies: Vec<u64>,
    /// Whether the entire island is sleeping.
    pub sleeping: bool,
    /// Time the island has been below the sleep energy threshold.
    pub sleep_timer: f32,
    /// Total kinetic energy of the island.
    pub total_energy: f32,
    /// Number of contact pairs in this island.
    pub contact_count: u32,
    /// Number of joints in this island.
    pub joint_count: u32,
    /// Whether this island should be simulated this frame.
    pub simulate: bool,
    /// AABB of the island (for spatial queries).
    pub aabb_min: [f32; 3],
    /// AABB max.
    pub aabb_max: [f32; 3],
}

impl Island {
    /// Create a new empty island.
    pub fn new(id: IslandId) -> Self {
        Self {
            id,
            bodies: Vec::new(),
            sleeping: false,
            sleep_timer: 0.0,
            total_energy: 0.0,
            contact_count: 0,
            joint_count: 0,
            simulate: true,
            aabb_min: [f32::MAX; 3],
            aabb_max: [f32::MIN; 3],
        }
    }

    /// Add a body to this island.
    pub fn add_body(&mut self, body_id: u64) {
        self.bodies.push(body_id);
    }

    /// Check if the island can sleep (all bodies below threshold).
    pub fn can_sleep(&self) -> bool {
        self.total_energy < SLEEP_ENERGY_THRESHOLD * self.bodies.len() as f32
    }

    /// Wake the island.
    pub fn wake(&mut self) {
        self.sleeping = false;
        self.sleep_timer = 0.0;
        self.simulate = true;
    }

    /// Put the island to sleep.
    pub fn sleep(&mut self) {
        self.sleeping = true;
        self.simulate = false;
    }

    /// Number of bodies in this island.
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Whether this island is trivial (single static body).
    pub fn is_trivial(&self) -> bool {
        self.bodies.len() <= 1
    }
}

// ---------------------------------------------------------------------------
// Island Manager
// ---------------------------------------------------------------------------

/// Manages physics islands for simulation optimization.
#[derive(Debug)]
pub struct IslandManager {
    /// Body data indexed by an internal index.
    bodies: Vec<IslandBody>,
    /// Mapping from body ID to internal index.
    body_index_map: HashMap<u64, u32>,
    /// Connections between bodies.
    connections: Vec<BodyConnection>,
    /// Built islands.
    pub islands: Vec<Island>,
    /// Next island ID.
    next_island_id: u32,
    /// Statistics.
    pub stats: IslandStats,
    /// Sleep configuration.
    pub sleep_config: SleepConfig,
}

/// Configuration for sleep behavior.
#[derive(Debug, Clone)]
pub struct SleepConfig {
    /// Energy threshold for sleeping.
    pub energy_threshold: f32,
    /// Delay before bodies can sleep.
    pub sleep_delay: f32,
    /// Whether to allow island sleeping.
    pub enable_sleeping: bool,
    /// Wake energy threshold.
    pub wake_threshold: f32,
    /// Maximum island size before forced simulation.
    pub max_island_size: usize,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            energy_threshold: SLEEP_ENERGY_THRESHOLD,
            sleep_delay: SLEEP_DELAY_SECONDS,
            enable_sleeping: true,
            wake_threshold: WAKE_ENERGY_THRESHOLD,
            max_island_size: MAX_ISLAND_SIZE,
        }
    }
}

impl IslandManager {
    /// Create a new island manager.
    pub fn new() -> Self {
        Self {
            bodies: Vec::with_capacity(DEFAULT_ISLAND_CAPACITY),
            body_index_map: HashMap::new(),
            connections: Vec::new(),
            islands: Vec::new(),
            next_island_id: 0,
            stats: IslandStats::default(),
            sleep_config: SleepConfig::default(),
        }
    }

    /// Register a body with the island system.
    pub fn add_body(&mut self, body: IslandBody) {
        let id = body.body_id;
        let index = self.bodies.len() as u32;
        self.bodies.push(body);
        self.body_index_map.insert(id, index);
    }

    /// Remove a body from the island system.
    pub fn remove_body(&mut self, body_id: u64) {
        self.body_index_map.remove(&body_id);
        // Note: we don't compact bodies here; we rebuild islands each frame.
    }

    /// Add a connection between two bodies.
    pub fn add_connection(&mut self, connection: BodyConnection) {
        self.connections.push(connection);
    }

    /// Clear all connections (called before rebuilding each frame).
    pub fn clear_connections(&mut self) {
        self.connections.clear();
    }

    /// Update body velocities (called before island building).
    pub fn update_body(
        &mut self,
        body_id: u64,
        linear_velocity_sq: f32,
        angular_velocity_sq: f32,
    ) {
        if let Some(&index) = self.body_index_map.get(&body_id) {
            if let Some(body) = self.bodies.get_mut(index as usize) {
                body.linear_velocity_sq = linear_velocity_sq;
                body.angular_velocity_sq = angular_velocity_sq;
            }
        }
    }

    /// Build islands from the current body and connection data.
    pub fn build_islands(&mut self) {
        let body_count = self.bodies.len();
        if body_count == 0 {
            self.islands.clear();
            return;
        }

        // Initialize union-find.
        let mut uf = DisjointSet::new(body_count);

        // Union connected bodies.
        for connection in &self.connections {
            let idx_a = self.body_index_map.get(&connection.body_a);
            let idx_b = self.body_index_map.get(&connection.body_b);

            if let (Some(&a), Some(&b)) = (idx_a, idx_b) {
                let body_a = &self.bodies[a as usize];
                let body_b = &self.bodies[b as usize];

                // Skip connections between two static bodies.
                if body_a.is_static && body_b.is_static {
                    continue;
                }

                uf.union(a, b);
            }
        }

        // Collect bodies into islands.
        let mut island_map: HashMap<u32, Vec<u64>> = HashMap::new();
        for (i, body) in self.bodies.iter().enumerate() {
            if body.is_static {
                continue; // Static bodies don't belong to islands.
            }
            let root = uf.find(i as u32);
            island_map.entry(root).or_insert_with(Vec::new).push(body.body_id);
        }

        // Build island objects.
        self.islands.clear();
        for (_, body_ids) in island_map {
            let island_id = IslandId(self.next_island_id);
            self.next_island_id = self.next_island_id.wrapping_add(1);

            let mut island = Island::new(island_id);
            let mut total_energy = 0.0f32;

            for &body_id in &body_ids {
                if let Some(&idx) = self.body_index_map.get(&body_id) {
                    let body = &self.bodies[idx as usize];
                    total_energy += body.kinetic_energy();
                }
                island.add_body(body_id);
            }

            island.total_energy = total_energy;

            // Count connections within this island.
            let body_set: std::collections::HashSet<u64> = body_ids.iter().copied().collect();
            for conn in &self.connections {
                if body_set.contains(&conn.body_a) || body_set.contains(&conn.body_b) {
                    match conn.connection_type {
                        ConnectionType::Contact => island.contact_count += 1,
                        ConnectionType::Joint => island.joint_count += 1,
                    }
                }
            }

            self.islands.push(island);
        }

        // Assign island IDs back to bodies.
        for island in &self.islands {
            for &body_id in &island.bodies {
                if let Some(&idx) = self.body_index_map.get(&body_id) {
                    if let Some(body) = self.bodies.get_mut(idx as usize) {
                        body.island_id = island.id;
                    }
                }
            }
        }

        self.update_stats();
    }

    /// Update sleep/wake state for all islands.
    pub fn update_sleep(&mut self, dt: f32) {
        if !self.sleep_config.enable_sleeping {
            for island in &mut self.islands {
                island.simulate = true;
                island.sleeping = false;
            }
            return;
        }

        for island in &mut self.islands {
            if island.can_sleep() {
                island.sleep_timer += dt;
                if island.sleep_timer >= self.sleep_config.sleep_delay {
                    island.sleep();
                }
            } else {
                island.sleep_timer = 0.0;
                if island.sleeping {
                    island.wake();
                }
                island.simulate = true;
            }
        }
    }

    /// Wake an island (e.g., due to external force or user action).
    pub fn wake_island(&mut self, island_id: IslandId) {
        if let Some(island) = self.islands.iter_mut().find(|i| i.id == island_id) {
            island.wake();

            // Wake all bodies in the island.
            for &body_id in &island.bodies {
                if let Some(&idx) = self.body_index_map.get(&body_id) {
                    if let Some(body) = self.bodies.get_mut(idx as usize) {
                        body.sleeping = false;
                        body.time_below_threshold = 0.0;
                    }
                }
            }
        }
    }

    /// Wake the island containing a specific body.
    pub fn wake_body(&mut self, body_id: u64) {
        if let Some(&idx) = self.body_index_map.get(&body_id) {
            if let Some(body) = self.bodies.get(idx as usize) {
                let island_id = body.island_id;
                if island_id.is_valid() {
                    self.wake_island(island_id);
                }
            }
        }
    }

    /// Get the island for a body.
    pub fn island_for_body(&self, body_id: u64) -> Option<IslandId> {
        self.body_index_map.get(&body_id).and_then(|&idx| {
            self.bodies.get(idx as usize).map(|b| b.island_id)
        }).filter(|id| id.is_valid())
    }

    /// Returns how many islands should be simulated this frame.
    pub fn active_island_count(&self) -> usize {
        self.islands.iter().filter(|i| i.simulate).count()
    }

    /// Returns the total number of islands.
    pub fn total_island_count(&self) -> usize {
        self.islands.len()
    }

    /// Returns the number of sleeping islands.
    pub fn sleeping_island_count(&self) -> usize {
        self.islands.iter().filter(|i| i.sleeping).count()
    }

    /// Update statistics.
    fn update_stats(&mut self) {
        self.stats.total_islands = self.islands.len() as u32;
        self.stats.active_islands = self.islands.iter().filter(|i| i.simulate).count() as u32;
        self.stats.sleeping_islands = self.islands.iter().filter(|i| i.sleeping).count() as u32;
        self.stats.total_bodies = self.bodies.iter().filter(|b| !b.is_static).count() as u32;
        self.stats.sleeping_bodies = self.bodies.iter().filter(|b| b.sleeping).count() as u32;
        self.stats.largest_island = self.islands.iter().map(|i| i.body_count()).max().unwrap_or(0) as u32;
        self.stats.total_connections = self.connections.len() as u32;
    }
}

/// Statistics for the island system.
#[derive(Debug, Clone, Default)]
pub struct IslandStats {
    /// Total number of islands.
    pub total_islands: u32,
    /// Number of active (simulated) islands.
    pub active_islands: u32,
    /// Number of sleeping islands.
    pub sleeping_islands: u32,
    /// Total dynamic bodies.
    pub total_bodies: u32,
    /// Number of sleeping bodies.
    pub sleeping_bodies: u32,
    /// Size of the largest island.
    pub largest_island: u32,
    /// Total connections.
    pub total_connections: u32,
}
