//! State synchronization with interest management and delta compression.
//!
//! Provides advanced replication features beyond the basic [`ReplicationManager`]:
//!
//! - **Interest management**: only replicate entities relevant to each client
//!   (area-of-interest spatial queries)
//! - **Priority-based bandwidth allocation**: nearby/important entities get more
//!   bandwidth
//! - **Snapshot interpolation**: smooth jitter-buffered interpolation with
//!   adaptive delay and Hermite spline support
//! - **Delta compression**: only send fields that changed, using per-field
//!   bit-flags

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default area-of-interest radius (world units).
pub const DEFAULT_AOI_RADIUS: f32 = 100.0;

/// Default AOI grid cell size.
pub const DEFAULT_CELL_SIZE: f32 = 50.0;

/// Maximum number of sync groups.
pub const MAX_SYNC_GROUPS: usize = 64;

/// Maximum entities per sync group.
pub const MAX_ENTITIES_PER_GROUP: usize = 256;

/// Default jitter buffer size.
pub const DEFAULT_JITTER_BUFFER_SIZE: usize = 3;

/// Minimum jitter buffer size.
pub const MIN_JITTER_BUFFER_SIZE: usize = 1;

/// Maximum jitter buffer size.
pub const MAX_JITTER_BUFFER_SIZE: usize = 10;

/// Maximum delta fields per component.
pub const MAX_DELTA_FIELDS: usize = 64;

// ---------------------------------------------------------------------------
// SyncGroup
// ---------------------------------------------------------------------------

/// A group of entities that synchronize together.
///
/// Sync groups allow controlling replication granularity. All entities in a
/// group share the same replication frequency, priority, and relevance rules.
#[derive(Debug, Clone)]
pub struct SyncGroup {
    /// Unique group ID.
    pub id: u32,
    /// Human-readable name.
    pub name: String,
    /// Entity IDs in this group (network IDs as u64).
    pub entities: HashSet<u64>,
    /// Replication frequency (updates per second).
    pub update_rate: f32,
    /// Base priority for entities in this group.
    pub base_priority: f32,
    /// Whether this group is active.
    pub active: bool,
    /// Last update tick.
    pub last_update_tick: u64,
}

impl SyncGroup {
    /// Create a new sync group.
    pub fn new(id: u32, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            entities: HashSet::new(),
            update_rate: 20.0,
            base_priority: 1.0,
            active: true,
            last_update_tick: 0,
        }
    }

    /// Add an entity to the group.
    pub fn add_entity(&mut self, entity_id: u64) -> bool {
        if self.entities.len() >= MAX_ENTITIES_PER_GROUP {
            return false;
        }
        self.entities.insert(entity_id)
    }

    /// Remove an entity from the group.
    pub fn remove_entity(&mut self, entity_id: u64) -> bool {
        self.entities.remove(&entity_id)
    }

    /// Returns true if the entity is in this group.
    pub fn contains(&self, entity_id: u64) -> bool {
        self.entities.contains(&entity_id)
    }

    /// Returns the number of entities.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Returns true if this group should update at the given tick.
    pub fn should_update(&self, current_tick: u64, tick_rate: f32) -> bool {
        if !self.active {
            return false;
        }
        let ticks_between_updates = (tick_rate / self.update_rate).max(1.0) as u64;
        current_tick >= self.last_update_tick + ticks_between_updates
    }
}

// ---------------------------------------------------------------------------
// SyncGroupManager
// ---------------------------------------------------------------------------

/// Manages multiple sync groups.
pub struct SyncGroupManager {
    /// Sync groups, keyed by ID.
    groups: HashMap<u32, SyncGroup>,
    /// Entity to group mapping.
    entity_to_group: HashMap<u64, u32>,
    /// Next group ID.
    next_group_id: u32,
}

impl SyncGroupManager {
    /// Create a new sync group manager.
    pub fn new() -> Self {
        Self {
            groups: HashMap::new(),
            entity_to_group: HashMap::new(),
            next_group_id: 1,
        }
    }

    /// Create a new sync group.
    pub fn create_group(&mut self, name: impl Into<String>) -> u32 {
        let id = self.next_group_id;
        self.next_group_id += 1;
        self.groups.insert(id, SyncGroup::new(id, name));
        id
    }

    /// Add an entity to a group.
    pub fn add_to_group(&mut self, entity_id: u64, group_id: u32) -> bool {
        // Remove from old group if any.
        if let Some(old_group) = self.entity_to_group.get(&entity_id).copied() {
            if let Some(group) = self.groups.get_mut(&old_group) {
                group.remove_entity(entity_id);
            }
        }

        if let Some(group) = self.groups.get_mut(&group_id) {
            if group.add_entity(entity_id) {
                self.entity_to_group.insert(entity_id, group_id);
                return true;
            }
        }
        false
    }

    /// Remove an entity from its group.
    pub fn remove_entity(&mut self, entity_id: u64) {
        if let Some(group_id) = self.entity_to_group.remove(&entity_id) {
            if let Some(group) = self.groups.get_mut(&group_id) {
                group.remove_entity(entity_id);
            }
        }
    }

    /// Get the group for an entity.
    pub fn get_entity_group(&self, entity_id: u64) -> Option<&SyncGroup> {
        let group_id = self.entity_to_group.get(&entity_id)?;
        self.groups.get(group_id)
    }

    /// Get a group by ID.
    pub fn get_group(&self, group_id: u32) -> Option<&SyncGroup> {
        self.groups.get(&group_id)
    }

    /// Get groups that should update this tick.
    pub fn groups_to_update(&self, current_tick: u64, tick_rate: f32) -> Vec<u32> {
        self.groups
            .values()
            .filter(|g| g.should_update(current_tick, tick_rate))
            .map(|g| g.id)
            .collect()
    }

    /// Mark a group as updated.
    pub fn mark_updated(&mut self, group_id: u32, tick: u64) {
        if let Some(group) = self.groups.get_mut(&group_id) {
            group.last_update_tick = tick;
        }
    }

    /// Returns the total number of groups.
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }
}

impl Default for SyncGroupManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AOIGrid — spatial grid for Area of Interest queries
// ---------------------------------------------------------------------------

/// An entity's position and metadata in the AOI grid.
#[derive(Debug, Clone)]
pub struct AOIEntity {
    /// Network entity ID.
    pub entity_id: u64,
    /// World position.
    pub position: [f32; 3],
    /// Importance weight (higher = more likely to be replicated).
    pub importance: f32,
    /// Whether this entity is always relevant (e.g., the player's own entity).
    pub always_relevant: bool,
}

/// A cell coordinate in the spatial grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CellCoord {
    x: i32,
    y: i32,
    z: i32,
}

/// Spatial grid for efficient area-of-interest queries.
///
/// The grid divides the world into cells of fixed size. Each cell stores the
/// entities within it. Relevance queries return entities within a radius by
/// checking only the cells that overlap the query region.
pub struct AOIGrid {
    /// Cell size (world units per cell edge).
    cell_size: f32,
    /// Cells, keyed by coordinate.
    cells: HashMap<CellCoord, Vec<u64>>,
    /// Entity positions for quick lookup.
    entity_positions: HashMap<u64, AOIEntity>,
}

impl AOIGrid {
    /// Create a new AOI grid with the given cell size.
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size: cell_size.max(1.0),
            cells: HashMap::new(),
            entity_positions: HashMap::new(),
        }
    }

    /// Convert a world position to a cell coordinate.
    fn world_to_cell(&self, pos: &[f32; 3]) -> CellCoord {
        CellCoord {
            x: (pos[0] / self.cell_size).floor() as i32,
            y: (pos[1] / self.cell_size).floor() as i32,
            z: (pos[2] / self.cell_size).floor() as i32,
        }
    }

    /// Insert or update an entity in the grid.
    pub fn update_entity(&mut self, entity: AOIEntity) {
        let entity_id = entity.entity_id;

        // Remove from old cell if position changed.
        if let Some(old) = self.entity_positions.get(&entity_id) {
            let old_cell = self.world_to_cell(&old.position);
            let new_cell = self.world_to_cell(&entity.position);

            if old_cell != new_cell {
                if let Some(cell) = self.cells.get_mut(&old_cell) {
                    cell.retain(|&id| id != entity_id);
                    if cell.is_empty() {
                        self.cells.remove(&old_cell);
                    }
                }
                self.cells
                    .entry(new_cell)
                    .or_insert_with(Vec::new)
                    .push(entity_id);
            }
        } else {
            // New entity.
            let cell = self.world_to_cell(&entity.position);
            self.cells
                .entry(cell)
                .or_insert_with(Vec::new)
                .push(entity_id);
        }

        self.entity_positions.insert(entity_id, entity);
    }

    /// Remove an entity from the grid.
    pub fn remove_entity(&mut self, entity_id: u64) {
        if let Some(entity) = self.entity_positions.remove(&entity_id) {
            let cell = self.world_to_cell(&entity.position);
            if let Some(cell_entities) = self.cells.get_mut(&cell) {
                cell_entities.retain(|&id| id != entity_id);
                if cell_entities.is_empty() {
                    self.cells.remove(&cell);
                }
            }
        }
    }

    /// Query entities within a radius of a position.
    ///
    /// Returns entity IDs sorted by distance (closest first).
    /// Always-relevant entities are included regardless of distance.
    pub fn query_radius(&self, center: &[f32; 3], radius: f32) -> Vec<u64> {
        let radius_sq = radius * radius;

        // Determine which cells to check.
        let cells_radius = (radius / self.cell_size).ceil() as i32 + 1;
        let center_cell = self.world_to_cell(center);

        let mut result_set: HashMap<u64, f32> = HashMap::new();

        // First, collect all always-relevant entities (from any cell).
        for entity in self.entity_positions.values() {
            if entity.always_relevant {
                let dist_sq = distance_squared(center, &entity.position);
                result_set.insert(entity.entity_id, dist_sq);
            }
        }

        // Then, collect entities within the radius from nearby cells.
        for dx in -cells_radius..=cells_radius {
            for dy in -cells_radius..=cells_radius {
                for dz in -cells_radius..=cells_radius {
                    let cell = CellCoord {
                        x: center_cell.x + dx,
                        y: center_cell.y + dy,
                        z: center_cell.z + dz,
                    };

                    if let Some(entities) = self.cells.get(&cell) {
                        for &eid in entities {
                            if let Some(entity) = self.entity_positions.get(&eid) {
                                let dist_sq = distance_squared(center, &entity.position);
                                if dist_sq <= radius_sq {
                                    result_set.insert(eid, dist_sq);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sort by distance.
        let mut results: Vec<(u64, f32)> = result_set.into_iter().collect();
        results.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.into_iter().map(|(id, _)| id).collect()
    }

    /// Check if an entity is relevant to a given position.
    pub fn is_relevant(&self, entity_id: u64, client_pos: &[f32; 3], radius: f32) -> bool {
        if let Some(entity) = self.entity_positions.get(&entity_id) {
            if entity.always_relevant {
                return true;
            }
            let dist_sq = distance_squared(client_pos, &entity.position);
            dist_sq <= radius * radius
        } else {
            false
        }
    }

    /// Get all always-relevant entities.
    pub fn always_relevant_entities(&self) -> Vec<u64> {
        self.entity_positions
            .values()
            .filter(|e| e.always_relevant)
            .map(|e| e.entity_id)
            .collect()
    }

    /// Returns the total number of entities tracked.
    pub fn entity_count(&self) -> usize {
        self.entity_positions.len()
    }

    /// Returns the number of non-empty cells.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Returns the cell size.
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }
}

/// Squared distance between two 3D points.
fn distance_squared(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Euclidean distance between two 3D points.
fn _distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    distance_squared(a, b).sqrt()
}

impl Default for AOIGrid {
    fn default() -> Self {
        Self::new(DEFAULT_CELL_SIZE)
    }
}

// ---------------------------------------------------------------------------
// InterestManagement
// ---------------------------------------------------------------------------

/// Per-client interest management.
///
/// Tracks which entities are relevant to each client and manages relevance
/// transitions (enter/exit relevance set).
pub struct InterestManagement {
    /// AOI grid for spatial queries.
    grid: AOIGrid,
    /// Per-client relevant entity sets.
    client_relevance: HashMap<u32, HashSet<u64>>,
    /// Per-client positions.
    client_positions: HashMap<u32, [f32; 3]>,
    /// Per-client AOI radius.
    client_radius: HashMap<u32, f32>,
    /// Default AOI radius.
    default_radius: f32,
}

/// Result of an interest update for a client.
#[derive(Debug, Clone)]
pub struct InterestUpdate {
    /// Client ID.
    pub client_id: u32,
    /// Entities that just became relevant (need spawn).
    pub entered: Vec<u64>,
    /// Entities that just became irrelevant (need despawn).
    pub exited: Vec<u64>,
    /// Entities that remain relevant (need state update).
    pub maintained: Vec<u64>,
}

impl InterestManagement {
    /// Create a new interest management system.
    pub fn new(cell_size: f32, default_radius: f32) -> Self {
        Self {
            grid: AOIGrid::new(cell_size),
            client_relevance: HashMap::new(),
            client_positions: HashMap::new(),
            client_radius: HashMap::new(),
            default_radius,
        }
    }

    /// Update an entity's position in the grid.
    pub fn update_entity(&mut self, entity: AOIEntity) {
        self.grid.update_entity(entity);
    }

    /// Remove an entity from the grid.
    pub fn remove_entity(&mut self, entity_id: u64) {
        self.grid.remove_entity(entity_id);

        // Remove from all clients' relevance sets.
        for relevance in self.client_relevance.values_mut() {
            relevance.remove(&entity_id);
        }
    }

    /// Register a client.
    pub fn add_client(&mut self, client_id: u32, position: [f32; 3]) {
        self.client_positions.insert(client_id, position);
        self.client_relevance.insert(client_id, HashSet::new());
    }

    /// Remove a client.
    pub fn remove_client(&mut self, client_id: u32) {
        self.client_positions.remove(&client_id);
        self.client_relevance.remove(&client_id);
        self.client_radius.remove(&client_id);
    }

    /// Update a client's position.
    pub fn update_client_position(&mut self, client_id: u32, position: [f32; 3]) {
        self.client_positions.insert(client_id, position);
    }

    /// Set a client's AOI radius.
    pub fn set_client_radius(&mut self, client_id: u32, radius: f32) {
        self.client_radius.insert(client_id, radius);
    }

    /// Compute interest updates for all clients.
    ///
    /// Returns a list of interest updates (entered, exited, maintained entities
    /// per client).
    pub fn compute_updates(&mut self) -> Vec<InterestUpdate> {
        let mut updates = Vec::new();

        let client_ids: Vec<u32> = self.client_positions.keys().copied().collect();

        for client_id in client_ids {
            let pos = match self.client_positions.get(&client_id) {
                Some(p) => *p,
                None => continue,
            };

            let radius = self
                .client_radius
                .get(&client_id)
                .copied()
                .unwrap_or(self.default_radius);

            // Query the grid.
            let new_relevant: HashSet<u64> =
                self.grid.query_radius(&pos, radius).into_iter().collect();

            // Compute diff.
            let old_relevant = self
                .client_relevance
                .entry(client_id)
                .or_insert_with(HashSet::new);

            let entered: Vec<u64> = new_relevant.difference(old_relevant).copied().collect();
            let exited: Vec<u64> = old_relevant.difference(&new_relevant).copied().collect();
            let maintained: Vec<u64> = new_relevant.intersection(old_relevant).copied().collect();

            if !entered.is_empty() || !exited.is_empty() || !maintained.is_empty() {
                updates.push(InterestUpdate {
                    client_id,
                    entered: entered.clone(),
                    exited: exited.clone(),
                    maintained: maintained.clone(),
                });
            }

            // Update the relevance set.
            *old_relevant = new_relevant;
        }

        updates
    }

    /// Returns the entities currently relevant to a client.
    pub fn client_entities(&self, client_id: u32) -> Option<&HashSet<u64>> {
        self.client_relevance.get(&client_id)
    }

    /// Returns the AOI grid.
    pub fn grid(&self) -> &AOIGrid {
        &self.grid
    }
}

impl Default for InterestManagement {
    fn default() -> Self {
        Self::new(DEFAULT_CELL_SIZE, DEFAULT_AOI_RADIUS)
    }
}

// ---------------------------------------------------------------------------
// StatePriority
// ---------------------------------------------------------------------------

/// Calculates replication priority for entities based on distance, visibility,
/// and importance.
pub struct StatePriority;

impl StatePriority {
    /// Calculate priority based on distance.
    ///
    /// Returns a value between 0.0 (lowest) and 1.0 (highest).
    /// Closer entities have higher priority.
    pub fn distance_priority(entity_pos: &[f32; 3], client_pos: &[f32; 3], max_dist: f32) -> f32 {
        let dist = distance_squared(entity_pos, client_pos).sqrt();
        (1.0 - (dist / max_dist).min(1.0)).max(0.0)
    }

    /// Calculate combined priority.
    ///
    /// `distance_weight`: weight for distance factor (0.0-1.0)
    /// `importance_weight`: weight for entity importance (0.0-1.0)
    pub fn combined_priority(
        entity_pos: &[f32; 3],
        client_pos: &[f32; 3],
        max_dist: f32,
        importance: f32,
        distance_weight: f32,
        importance_weight: f32,
    ) -> f32 {
        let dist_p = Self::distance_priority(entity_pos, client_pos, max_dist);
        let imp_p = importance.clamp(0.0, 1.0);
        (dist_p * distance_weight + imp_p * importance_weight)
            / (distance_weight + importance_weight).max(0.001)
    }

    /// Allocate bandwidth budget based on priorities.
    ///
    /// Given a list of (entity_id, priority) pairs and a total budget,
    /// returns (entity_id, allocated_bytes) pairs.
    pub fn allocate_bandwidth(
        entities: &[(u64, f32)],
        total_budget: u32,
    ) -> Vec<(u64, u32)> {
        if entities.is_empty() {
            return Vec::new();
        }

        let total_priority: f32 = entities.iter().map(|(_, p)| p).sum();
        if total_priority <= 0.0 {
            // Equal distribution.
            let per_entity = total_budget / entities.len() as u32;
            return entities.iter().map(|(id, _)| (*id, per_entity)).collect();
        }

        entities
            .iter()
            .map(|(id, priority)| {
                let fraction = priority / total_priority;
                let bytes = (fraction * total_budget as f32) as u32;
                (*id, bytes.max(1))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// SnapshotInterpolation — improved with jitter buffer
// ---------------------------------------------------------------------------

/// A state sample for interpolation.
#[derive(Debug, Clone)]
pub struct InterpolationSample {
    /// Tick number.
    pub tick: u64,
    /// Position (or generic state vector).
    pub position: [f32; 3],
    /// Velocity (for Hermite interpolation).
    pub velocity: [f32; 3],
    /// Receive timestamp (local time, seconds).
    pub receive_time: f64,
}

/// Snapshot interpolation with adaptive jitter buffer.
pub struct SnapshotInterpolation {
    /// Buffered samples.
    buffer: Vec<InterpolationSample>,
    /// Maximum buffer size.
    max_buffer_size: usize,
    /// Current adaptive buffer target count.
    target_buffer_count: usize,
    /// Measured jitter (standard deviation of inter-arrival times).
    jitter: f64,
    /// Last arrival time (for jitter measurement).
    last_arrival_time: Option<f64>,
    /// EWMA of inter-arrival time.
    avg_interarrival: f64,
    /// Tick rate (ticks per second).
    tick_rate: f64,
    /// Whether to use Hermite interpolation (uses velocity).
    use_hermite: bool,
}

impl SnapshotInterpolation {
    /// Create a new snapshot interpolation buffer.
    pub fn new(tick_rate: f64) -> Self {
        Self {
            buffer: Vec::new(),
            max_buffer_size: MAX_JITTER_BUFFER_SIZE,
            target_buffer_count: DEFAULT_JITTER_BUFFER_SIZE,
            jitter: 0.0,
            last_arrival_time: None,
            avg_interarrival: 1.0 / tick_rate,
            tick_rate,
            use_hermite: false,
        }
    }

    /// Enable Hermite interpolation (uses velocity data).
    pub fn set_hermite(&mut self, enabled: bool) {
        self.use_hermite = enabled;
    }

    /// Push a new sample into the buffer.
    pub fn push(&mut self, sample: InterpolationSample) {
        let now = sample.receive_time;

        // Measure jitter.
        if let Some(last) = self.last_arrival_time {
            let interarrival = now - last;
            let diff = (interarrival - self.avg_interarrival).abs();
            self.jitter = 0.875 * self.jitter + 0.125 * diff;
            self.avg_interarrival = 0.875 * self.avg_interarrival + 0.125 * interarrival;

            // Adapt buffer size based on jitter.
            let expected_interarrival = 1.0 / self.tick_rate;
            let jitter_ratio = self.jitter / expected_interarrival;
            if jitter_ratio > 1.0 {
                self.target_buffer_count =
                    (self.target_buffer_count + 1).min(MAX_JITTER_BUFFER_SIZE);
            } else if jitter_ratio < 0.3 && self.target_buffer_count > MIN_JITTER_BUFFER_SIZE {
                self.target_buffer_count -= 1;
            }
        }
        self.last_arrival_time = Some(now);

        // Insert in tick order.
        let tick = sample.tick;
        let pos = self.buffer.iter().position(|s| s.tick > tick);
        match pos {
            Some(p) => self.buffer.insert(p, sample),
            None => self.buffer.push(sample),
        }

        // Trim.
        while self.buffer.len() > self.max_buffer_size {
            self.buffer.remove(0);
        }
    }

    /// Interpolate at a given render tick (fractional).
    ///
    /// The render tick should be delayed by `target_buffer_count` ticks behind
    /// the newest sample to absorb jitter.
    pub fn interpolate(&self, render_tick: f64) -> Option<[f32; 3]> {
        if self.buffer.len() < 2 {
            return None;
        }

        // Find bracketing samples.
        let mut before = None;
        let mut after = None;

        for sample in &self.buffer {
            if (sample.tick as f64) <= render_tick {
                before = Some(sample);
            } else if after.is_none() {
                after = Some(sample);
            }
        }

        match (before, after) {
            (Some(a), Some(b)) => {
                let range = (b.tick as f64) - (a.tick as f64);
                let t = if range > 0.0 {
                    ((render_tick - a.tick as f64) / range) as f32
                } else {
                    0.0
                };
                let t = t.clamp(0.0, 1.0);

                if self.use_hermite {
                    Some(hermite_interpolate(
                        &a.position,
                        &a.velocity,
                        &b.position,
                        &b.velocity,
                        t,
                        range as f32 / self.tick_rate as f32,
                    ))
                } else {
                    Some(lerp_position(&a.position, &b.position, t))
                }
            }
            (Some(a), None) => {
                // Extrapolation: use latest position.
                Some(a.position)
            }
            _ => None,
        }
    }

    /// Returns the recommended render tick (newest_tick - buffer_count).
    pub fn render_tick(&self) -> Option<f64> {
        let newest = self.buffer.last()?.tick;
        Some(newest as f64 - self.target_buffer_count as f64)
    }

    /// Returns the current jitter estimate.
    pub fn jitter(&self) -> f64 {
        self.jitter
    }

    /// Returns the current adaptive buffer size.
    pub fn buffer_count(&self) -> usize {
        self.target_buffer_count
    }

    /// Returns the number of buffered samples.
    pub fn sample_count(&self) -> usize {
        self.buffer.len()
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.last_arrival_time = None;
    }
}

/// Linear interpolation between two positions.
fn lerp_position(a: &[f32; 3], b: &[f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Hermite interpolation using position and velocity at both endpoints.
///
/// Provides smoother curves than linear interpolation when velocity data
/// is available. `dt` is the time span between the two samples in seconds.
fn hermite_interpolate(
    p0: &[f32; 3],
    v0: &[f32; 3],
    p1: &[f32; 3],
    v1: &[f32; 3],
    t: f32,
    dt: f32,
) -> [f32; 3] {
    let t2 = t * t;
    let t3 = t2 * t;

    // Hermite basis functions.
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    [
        h00 * p0[0] + h10 * dt * v0[0] + h01 * p1[0] + h11 * dt * v1[0],
        h00 * p0[1] + h10 * dt * v0[1] + h01 * p1[1] + h11 * dt * v1[1],
        h00 * p0[2] + h10 * dt * v0[2] + h01 * p1[2] + h11 * dt * v1[2],
    ]
}

// ---------------------------------------------------------------------------
// DeltaCompression
// ---------------------------------------------------------------------------

/// Per-field delta encoder/decoder.
///
/// Uses a bit-flag header to indicate which fields have changed. Only changed
/// fields are included in the encoded output.
pub struct DeltaCompression;

impl DeltaCompression {
    /// Encode a delta between old and new state.
    ///
    /// `fields` is a list of (field_data_old, field_data_new) pairs.
    /// Returns the encoded delta bytes.
    ///
    /// Format: [u64 changed_bits (up to 64 fields)] [changed field data...]
    pub fn encode_delta(fields: &[(&[u8], &[u8])]) -> Vec<u8> {
        assert!(fields.len() <= MAX_DELTA_FIELDS);

        let mut changed_bits: u64 = 0;
        let mut changed_data = Vec::new();

        for (i, (old, new)) in fields.iter().enumerate() {
            if old != new {
                changed_bits |= 1 << i;
                // Length-prefixed field data.
                changed_data.extend_from_slice(&(new.len() as u16).to_be_bytes());
                changed_data.extend_from_slice(new);
            }
        }

        let mut buf = Vec::with_capacity(8 + changed_data.len());
        buf.extend_from_slice(&changed_bits.to_be_bytes());
        buf.extend_from_slice(&changed_data);
        buf
    }

    /// Decode a delta and apply it to the current state.
    ///
    /// `current_fields` contains the current field values (will be updated).
    /// Returns the number of fields changed.
    pub fn decode_delta(
        data: &[u8],
        current_fields: &mut [Vec<u8>],
    ) -> Option<usize> {
        if data.len() < 8 {
            return None;
        }

        let changed_bits = u64::from_be_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);

        let mut offset = 8;
        let mut changes = 0;

        for (i, field) in current_fields.iter_mut().enumerate() {
            if changed_bits & (1 << i) != 0 {
                if data.len() < offset + 2 {
                    return None;
                }
                let field_len =
                    u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
                offset += 2;

                if data.len() < offset + field_len {
                    return None;
                }
                *field = data[offset..offset + field_len].to_vec();
                offset += field_len;
                changes += 1;
            }
        }

        Some(changes)
    }

    /// Check if any fields differ between old and new state.
    pub fn has_changes(fields: &[(&[u8], &[u8])]) -> bool {
        fields.iter().any(|(old, new)| old != new)
    }

    /// Count the number of changed fields.
    pub fn change_count(fields: &[(&[u8], &[u8])]) -> usize {
        fields.iter().filter(|(old, new)| old != new).count()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // SyncGroup
    // -----------------------------------------------------------------------

    #[test]
    fn test_sync_group_basic() {
        let mut group = SyncGroup::new(1, "test");
        assert!(group.add_entity(100));
        assert!(group.add_entity(200));
        assert_eq!(group.entity_count(), 2);
        assert!(group.contains(100));
        assert!(!group.contains(300));

        assert!(group.remove_entity(100));
        assert_eq!(group.entity_count(), 1);
    }

    #[test]
    fn test_sync_group_should_update() {
        let mut group = SyncGroup::new(1, "test");
        group.update_rate = 20.0; // 20 Hz
        group.last_update_tick = 0;

        // At 60 Hz tick rate, should update every 3 ticks.
        assert!(group.should_update(3, 60.0));
        assert!(!group.should_update(2, 60.0));
    }

    #[test]
    fn test_sync_group_manager() {
        let mut mgr = SyncGroupManager::new();
        let g1 = mgr.create_group("players");
        let g2 = mgr.create_group("npcs");

        mgr.add_to_group(100, g1);
        mgr.add_to_group(200, g2);

        assert_eq!(mgr.get_entity_group(100).unwrap().id, g1);
        assert_eq!(mgr.get_entity_group(200).unwrap().id, g2);

        mgr.remove_entity(100);
        assert!(mgr.get_entity_group(100).is_none());
    }

    // -----------------------------------------------------------------------
    // AOIGrid
    // -----------------------------------------------------------------------

    #[test]
    fn test_aoi_grid_insert_and_query() {
        let mut grid = AOIGrid::new(10.0);

        grid.update_entity(AOIEntity {
            entity_id: 1,
            position: [5.0, 0.0, 5.0],
            importance: 1.0,
            always_relevant: false,
        });
        grid.update_entity(AOIEntity {
            entity_id: 2,
            position: [100.0, 0.0, 100.0],
            importance: 1.0,
            always_relevant: false,
        });
        grid.update_entity(AOIEntity {
            entity_id: 3,
            position: [8.0, 0.0, 3.0],
            importance: 1.0,
            always_relevant: false,
        });

        assert_eq!(grid.entity_count(), 3);

        // Query near origin -- should find entities 1 and 3.
        let nearby = grid.query_radius(&[0.0, 0.0, 0.0], 20.0);
        assert!(nearby.contains(&1));
        assert!(nearby.contains(&3));
        assert!(!nearby.contains(&2)); // too far

        // Query near (100, 0, 100) -- should find entity 2.
        let nearby = grid.query_radius(&[100.0, 0.0, 100.0], 20.0);
        assert!(nearby.contains(&2));
        assert!(!nearby.contains(&1));
    }

    #[test]
    fn test_aoi_grid_always_relevant() {
        let mut grid = AOIGrid::new(10.0);

        grid.update_entity(AOIEntity {
            entity_id: 1,
            position: [1000.0, 0.0, 1000.0],
            importance: 1.0,
            always_relevant: true,
        });

        // Should be found even though it's far away.
        let nearby = grid.query_radius(&[0.0, 0.0, 0.0], 10.0);
        assert!(nearby.contains(&1));
    }

    #[test]
    fn test_aoi_grid_move_entity() {
        let mut grid = AOIGrid::new(10.0);

        grid.update_entity(AOIEntity {
            entity_id: 1,
            position: [5.0, 0.0, 5.0],
            importance: 1.0,
            always_relevant: false,
        });

        // Move entity to a different cell.
        grid.update_entity(AOIEntity {
            entity_id: 1,
            position: [50.0, 0.0, 50.0],
            importance: 1.0,
            always_relevant: false,
        });

        // Should NOT be near origin anymore.
        let nearby = grid.query_radius(&[0.0, 0.0, 0.0], 10.0);
        assert!(!nearby.contains(&1));

        // Should be near its new position.
        let nearby = grid.query_radius(&[50.0, 0.0, 50.0], 10.0);
        assert!(nearby.contains(&1));
    }

    #[test]
    fn test_aoi_grid_remove() {
        let mut grid = AOIGrid::new(10.0);

        grid.update_entity(AOIEntity {
            entity_id: 1,
            position: [5.0, 0.0, 5.0],
            importance: 1.0,
            always_relevant: false,
        });

        grid.remove_entity(1);
        assert_eq!(grid.entity_count(), 0);

        let nearby = grid.query_radius(&[0.0, 0.0, 0.0], 100.0);
        assert!(nearby.is_empty());
    }

    #[test]
    fn test_aoi_is_relevant() {
        let mut grid = AOIGrid::new(10.0);

        grid.update_entity(AOIEntity {
            entity_id: 1,
            position: [5.0, 0.0, 5.0],
            importance: 1.0,
            always_relevant: false,
        });

        assert!(grid.is_relevant(1, &[0.0, 0.0, 0.0], 20.0));
        assert!(!grid.is_relevant(1, &[0.0, 0.0, 0.0], 3.0));
        assert!(!grid.is_relevant(99, &[0.0, 0.0, 0.0], 100.0));
    }

    #[test]
    fn test_aoi_grid_sorted_by_distance() {
        let mut grid = AOIGrid::new(100.0);

        grid.update_entity(AOIEntity {
            entity_id: 1,
            position: [30.0, 0.0, 0.0],
            importance: 1.0,
            always_relevant: false,
        });
        grid.update_entity(AOIEntity {
            entity_id: 2,
            position: [10.0, 0.0, 0.0],
            importance: 1.0,
            always_relevant: false,
        });
        grid.update_entity(AOIEntity {
            entity_id: 3,
            position: [20.0, 0.0, 0.0],
            importance: 1.0,
            always_relevant: false,
        });

        let nearby = grid.query_radius(&[0.0, 0.0, 0.0], 50.0);
        assert_eq!(nearby, vec![2, 3, 1]); // sorted by distance
    }

    // -----------------------------------------------------------------------
    // InterestManagement
    // -----------------------------------------------------------------------

    #[test]
    fn test_interest_management_basic() {
        let mut im = InterestManagement::new(10.0, 50.0);

        im.add_client(1, [0.0, 0.0, 0.0]);

        im.update_entity(AOIEntity {
            entity_id: 100,
            position: [10.0, 0.0, 0.0],
            importance: 1.0,
            always_relevant: false,
        });

        im.update_entity(AOIEntity {
            entity_id: 200,
            position: [1000.0, 0.0, 0.0],
            importance: 1.0,
            always_relevant: false,
        });

        let updates = im.compute_updates();
        assert_eq!(updates.len(), 1);
        assert!(updates[0].entered.contains(&100));
        assert!(!updates[0].entered.contains(&200));
    }

    #[test]
    fn test_interest_management_enter_exit() {
        let mut im = InterestManagement::new(10.0, 30.0);

        im.add_client(1, [0.0, 0.0, 0.0]);

        im.update_entity(AOIEntity {
            entity_id: 100,
            position: [10.0, 0.0, 0.0],
            importance: 1.0,
            always_relevant: false,
        });

        // First update: entity enters.
        let updates = im.compute_updates();
        assert!(updates[0].entered.contains(&100));
        assert!(updates[0].exited.is_empty());

        // Move entity far away.
        im.update_entity(AOIEntity {
            entity_id: 100,
            position: [1000.0, 0.0, 0.0],
            importance: 1.0,
            always_relevant: false,
        });

        // Second update: entity exits.
        let updates = im.compute_updates();
        assert!(updates[0].exited.contains(&100));
        assert!(updates[0].entered.is_empty());
    }

    // -----------------------------------------------------------------------
    // StatePriority
    // -----------------------------------------------------------------------

    #[test]
    fn test_distance_priority() {
        // At origin, priority should be 1.0.
        let p = StatePriority::distance_priority(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], 100.0);
        assert!((p - 1.0).abs() < 0.001);

        // At max distance, priority should be 0.0.
        let p = StatePriority::distance_priority(&[100.0, 0.0, 0.0], &[0.0, 0.0, 0.0], 100.0);
        assert!((p - 0.0).abs() < 0.01);

        // At half distance, priority should be ~0.5.
        let p = StatePriority::distance_priority(&[50.0, 0.0, 0.0], &[0.0, 0.0, 0.0], 100.0);
        assert!((p - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_bandwidth_allocation() {
        let entities = vec![(1, 0.3_f32), (2, 0.5), (3, 0.2)];
        let allocations = StatePriority::allocate_bandwidth(&entities, 1000);

        assert_eq!(allocations.len(), 3);

        // Entity 2 should get the most bandwidth.
        let e2_alloc = allocations.iter().find(|(id, _)| *id == 2).unwrap().1;
        let e1_alloc = allocations.iter().find(|(id, _)| *id == 1).unwrap().1;
        assert!(e2_alloc > e1_alloc);

        // Total should approximately equal budget.
        let total: u32 = allocations.iter().map(|(_, b)| b).sum();
        assert!(total <= 1000 + 3); // allow rounding
    }

    // -----------------------------------------------------------------------
    // SnapshotInterpolation
    // -----------------------------------------------------------------------

    #[test]
    fn test_snapshot_interpolation_lerp() {
        let mut interp = SnapshotInterpolation::new(60.0);

        interp.push(InterpolationSample {
            tick: 0,
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            receive_time: 0.0,
        });
        interp.push(InterpolationSample {
            tick: 10,
            position: [100.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            receive_time: 0.1,
        });

        // Midpoint.
        let pos = interp.interpolate(5.0).unwrap();
        assert!((pos[0] - 50.0).abs() < 0.1);
        assert!((pos[1] - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_snapshot_interpolation_hermite() {
        let mut interp = SnapshotInterpolation::new(60.0);
        interp.set_hermite(true);

        interp.push(InterpolationSample {
            tick: 0,
            position: [0.0, 0.0, 0.0],
            velocity: [10.0, 0.0, 0.0],
            receive_time: 0.0,
        });
        interp.push(InterpolationSample {
            tick: 60,
            position: [10.0, 0.0, 0.0],
            velocity: [10.0, 0.0, 0.0],
            receive_time: 1.0,
        });

        // Hermite should produce a smooth curve.
        let pos = interp.interpolate(30.0).unwrap();
        // With constant velocity, Hermite should give ~5.0 at midpoint.
        assert!((pos[0] - 5.0).abs() < 1.0);
    }

    #[test]
    fn test_snapshot_interpolation_render_tick() {
        let mut interp = SnapshotInterpolation::new(60.0);

        interp.push(InterpolationSample {
            tick: 0,
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            receive_time: 0.0,
        });
        interp.push(InterpolationSample {
            tick: 10,
            position: [100.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            receive_time: 0.1,
        });

        let rt = interp.render_tick().unwrap();
        assert!(rt < 10.0); // Should be behind the newest tick.
    }

    // -----------------------------------------------------------------------
    // DeltaCompression
    // -----------------------------------------------------------------------

    #[test]
    fn test_delta_no_changes() {
        let fields: Vec<(&[u8], &[u8])> = vec![
            (&[1, 2, 3], &[1, 2, 3]),
            (&[4, 5], &[4, 5]),
        ];

        assert!(!DeltaCompression::has_changes(&fields));
        assert_eq!(DeltaCompression::change_count(&fields), 0);

        let encoded = DeltaCompression::encode_delta(&fields);
        // Changed bits should be 0.
        let changed_bits = u64::from_be_bytes([
            encoded[0], encoded[1], encoded[2], encoded[3],
            encoded[4], encoded[5], encoded[6], encoded[7],
        ]);
        assert_eq!(changed_bits, 0);
    }

    #[test]
    fn test_delta_with_changes() {
        let fields: Vec<(&[u8], &[u8])> = vec![
            (&[1, 2, 3], &[1, 2, 3]),  // unchanged
            (&[4, 5], &[6, 7]),          // changed
            (&[8], &[8]),                // unchanged
            (&[9, 10], &[11, 12]),       // changed
        ];

        assert!(DeltaCompression::has_changes(&fields));
        assert_eq!(DeltaCompression::change_count(&fields), 2);

        let encoded = DeltaCompression::encode_delta(&fields);

        // Decode.
        let mut current = vec![
            vec![1, 2, 3],
            vec![4, 5],
            vec![8],
            vec![9, 10],
        ];

        let changes = DeltaCompression::decode_delta(&encoded, &mut current).unwrap();
        assert_eq!(changes, 2);

        // Field 0 unchanged.
        assert_eq!(current[0], vec![1, 2, 3]);
        // Field 1 changed.
        assert_eq!(current[1], vec![6, 7]);
        // Field 2 unchanged.
        assert_eq!(current[2], vec![8]);
        // Field 3 changed.
        assert_eq!(current[3], vec![11, 12]);
    }

    #[test]
    fn test_delta_all_changed() {
        let fields: Vec<(&[u8], &[u8])> = vec![
            (&[1], &[10]),
            (&[2], &[20]),
            (&[3], &[30]),
        ];

        let encoded = DeltaCompression::encode_delta(&fields);

        let mut current = vec![vec![1], vec![2], vec![3]];
        let changes = DeltaCompression::decode_delta(&encoded, &mut current).unwrap();
        assert_eq!(changes, 3);
        assert_eq!(current, vec![vec![10], vec![20], vec![30]]);
    }

    #[test]
    fn test_delta_decode_too_short() {
        let result = DeltaCompression::decode_delta(&[0; 4], &mut []);
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // Helper functions
    // -----------------------------------------------------------------------

    #[test]
    fn test_lerp_position() {
        let a = [0.0, 0.0, 0.0];
        let b = [10.0, 20.0, 30.0];

        let mid = lerp_position(&a, &b, 0.5);
        assert!((mid[0] - 5.0).abs() < 0.001);
        assert!((mid[1] - 10.0).abs() < 0.001);
        assert!((mid[2] - 15.0).abs() < 0.001);

        let start = lerp_position(&a, &b, 0.0);
        assert_eq!(start, a);

        let end = lerp_position(&a, &b, 1.0);
        assert!((end[0] - b[0]).abs() < 0.001);
    }

    #[test]
    fn test_hermite_at_endpoints() {
        let p0 = [0.0, 0.0, 0.0];
        let v0 = [1.0, 0.0, 0.0];
        let p1 = [10.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];

        let at_0 = hermite_interpolate(&p0, &v0, &p1, &v1, 0.0, 1.0);
        assert!((at_0[0] - p0[0]).abs() < 0.001);

        let at_1 = hermite_interpolate(&p0, &v0, &p1, &v1, 1.0, 1.0);
        assert!((at_1[0] - p1[0]).abs() < 0.001);
    }

    #[test]
    fn test_distance_squared_fn() {
        let a = [0.0, 0.0, 0.0];
        let b = [3.0, 4.0, 0.0];
        assert!((distance_squared(&a, &b) - 25.0).abs() < 0.001);
    }
}
