//! World Partition
//!
//! Divides the game world into a grid of cells that can be independently
//! loaded and unloaded. The partition system tracks the camera position and
//! produces streaming operations indicating which cells should be loaded or
//! unloaded based on configurable distance thresholds with hysteresis.
//!
//! # Design
//!
//! The world is divided into axis-aligned square cells on the XZ plane. Each
//! cell is identified by an integer coordinate pair [`CellCoord`]. Cells
//! transition through a state machine managed by the [`StreamingManager`]:
//!
//! ```text
//! Unloaded -> Loading -> Loaded -> Active -> PendingUnload -> Unloaded
//! ```
//!
//! Distance hysteresis prevents thrashing: cells are loaded when they enter
//! the load radius and unloaded only when they exit a larger unload radius.
//! Each [`WorldLayer`] can override these distances so that high-detail
//! layers (foliage, audio) load at shorter ranges than coarser layers
//! (terrain geometry, AI navigation).

use std::collections::{HashMap, HashSet};
use std::fmt;

use glam::Vec3;
use serde::{Deserialize, Serialize};

use genovo_ecs::Entity;

// ---------------------------------------------------------------------------
// CellCoord
// ---------------------------------------------------------------------------

/// Integer coordinate identifying a world cell on the XZ grid.
///
/// The origin cell (0, 0) covers world positions [0 .. cell_size) on both
/// axes. Negative coordinates extend in the negative X/Z directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CellCoord {
    /// Column index (X axis).
    pub x: i32,
    /// Row index (Z axis).
    pub z: i32,
}

impl CellCoord {
    /// Construct a new cell coordinate.
    #[inline]
    pub const fn new(x: i32, z: i32) -> Self {
        Self { x, z }
    }

    /// Returns the squared Chebyshev (chessboard) distance to `other`.
    #[inline]
    pub fn chebyshev_distance(&self, other: &CellCoord) -> i32 {
        let dx = (self.x - other.x).abs();
        let dz = (self.z - other.z).abs();
        dx.max(dz)
    }

    /// Euclidean distance squared between cell centers (requires cell_size).
    #[inline]
    pub fn distance_sq(&self, other: &CellCoord, cell_size: f32) -> f32 {
        let dx = (self.x - other.x) as f32 * cell_size;
        let dz = (self.z - other.z) as f32 * cell_size;
        dx * dx + dz * dz
    }

    /// Euclidean distance between cell centers.
    #[inline]
    pub fn distance(&self, other: &CellCoord, cell_size: f32) -> f32 {
        self.distance_sq(other, cell_size).sqrt()
    }

    /// Iterator over all cells within Chebyshev radius `r` of this cell.
    pub fn neighbors(&self, radius: i32) -> CellNeighborIter {
        CellNeighborIter {
            center: *self,
            radius,
            cx: -radius,
            cz: -radius,
        }
    }
}

impl fmt::Display for CellCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.z)
    }
}

// ---------------------------------------------------------------------------
// CellNeighborIter
// ---------------------------------------------------------------------------

/// Iterator that yields all cells within a Chebyshev radius of a center cell.
pub struct CellNeighborIter {
    center: CellCoord,
    radius: i32,
    cx: i32,
    cz: i32,
}

impl Iterator for CellNeighborIter {
    type Item = CellCoord;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cx > self.radius {
            return None;
        }
        let coord = CellCoord::new(self.center.x + self.cx, self.center.z + self.cz);
        self.cz += 1;
        if self.cz > self.radius {
            self.cz = -self.radius;
            self.cx += 1;
        }
        Some(coord)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let side = (self.radius * 2 + 1) as usize;
        let total = side * side;
        (0, Some(total))
    }
}

// ---------------------------------------------------------------------------
// CellState
// ---------------------------------------------------------------------------

/// Lifecycle state of a world cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CellState {
    /// Cell data is not in memory.
    Unloaded,
    /// Cell data is being asynchronously loaded.
    Loading,
    /// Cell data is in memory but not yet active (pending activation).
    Loaded,
    /// Cell is fully active: entities are ticked, rendered, and collidable.
    Active,
    /// Cell has been flagged for unloading; serialization may be in progress.
    PendingUnload,
}

impl CellState {
    /// Returns `true` if the cell has any data in memory.
    #[inline]
    pub fn is_resident(&self) -> bool {
        matches!(self, CellState::Loaded | CellState::Active | CellState::PendingUnload)
    }

    /// Returns `true` if the cell is actively simulated.
    #[inline]
    pub fn is_active(&self) -> bool {
        matches!(self, CellState::Active)
    }
}

// ---------------------------------------------------------------------------
// WorldCell
// ---------------------------------------------------------------------------

/// A single cell in the world partition grid.
///
/// Each cell tracks its streaming state, the entities it contains, and
/// per-layer load status. Cells are the atomic unit of streaming: they are
/// loaded and unloaded as indivisible groups.
#[derive(Debug, Clone)]
pub struct WorldCell {
    /// Grid coordinate of this cell.
    pub coord: CellCoord,
    /// Center position in world space.
    pub center: Vec3,
    /// Half-extent of the cell (cell_size / 2).
    pub half_extent: f32,
    /// Current streaming state.
    pub state: CellState,
    /// Entities that belong to this cell.
    pub entities: Vec<Entity>,
    /// Per-layer streaming state.
    pub layer_states: HashMap<String, CellState>,
    /// Estimated memory usage in bytes when loaded.
    pub memory_usage: usize,
    /// Priority value for streaming (lower = higher priority). Updated each
    /// frame based on distance to camera.
    pub streaming_priority: f32,
    /// Number of frames this cell has been in its current state.
    pub state_duration_frames: u32,
    /// Whether the cell data has been modified since last save.
    pub dirty: bool,
}

impl WorldCell {
    /// Create a new unloaded cell.
    pub fn new(coord: CellCoord, cell_size: f32) -> Self {
        let center = Vec3::new(
            coord.x as f32 * cell_size + cell_size * 0.5,
            0.0,
            coord.z as f32 * cell_size + cell_size * 0.5,
        );

        Self {
            coord,
            center,
            half_extent: cell_size * 0.5,
            state: CellState::Unloaded,
            entities: Vec::new(),
            layer_states: HashMap::new(),
            memory_usage: 0,
            streaming_priority: f32::MAX,
            state_duration_frames: 0,
            dirty: false,
        }
    }

    /// Returns the axis-aligned bounding box of this cell on the XZ plane.
    /// Y bounds are set to a large range since cells are infinite vertically.
    pub fn bounds_min(&self) -> Vec3 {
        Vec3::new(
            self.center.x - self.half_extent,
            -1e6,
            self.center.z - self.half_extent,
        )
    }

    /// Upper corner of the cell AABB.
    pub fn bounds_max(&self) -> Vec3 {
        Vec3::new(
            self.center.x + self.half_extent,
            1e6,
            self.center.z + self.half_extent,
        )
    }

    /// Squared distance from the cell center to a world-space point (XZ only).
    #[inline]
    pub fn distance_sq_xz(&self, pos: Vec3) -> f32 {
        let dx = self.center.x - pos.x;
        let dz = self.center.z - pos.z;
        dx * dx + dz * dz
    }

    /// Distance from the cell center to a world-space point (XZ only).
    #[inline]
    pub fn distance_xz(&self, pos: Vec3) -> f32 {
        self.distance_sq_xz(pos).sqrt()
    }

    /// Test whether a world-space point lies within this cell on the XZ plane.
    pub fn contains_xz(&self, pos: Vec3) -> bool {
        let dx = (pos.x - self.center.x).abs();
        let dz = (pos.z - self.center.z).abs();
        dx <= self.half_extent && dz <= self.half_extent
    }

    /// Transition to a new state, resetting the duration counter.
    pub fn set_state(&mut self, new_state: CellState) {
        if self.state != new_state {
            log::trace!(
                "Cell {} transitioning {:?} -> {:?}",
                self.coord,
                self.state,
                new_state,
            );
            self.state = new_state;
            self.state_duration_frames = 0;
        }
    }

    /// Add an entity to this cell.
    pub fn add_entity(&mut self, entity: Entity) {
        if !self.entities.contains(&entity) {
            self.entities.push(entity);
        }
    }

    /// Remove an entity from this cell. Returns `true` if it was found.
    pub fn remove_entity(&mut self, entity: Entity) -> bool {
        if let Some(idx) = self.entities.iter().position(|e| *e == entity) {
            self.entities.swap_remove(idx);
            true
        } else {
            false
        }
    }

    /// Update layer state for a specific layer.
    pub fn set_layer_state(&mut self, layer: &str, state: CellState) {
        self.layer_states.insert(layer.to_owned(), state);
    }

    /// Get the state of a specific layer, defaulting to Unloaded.
    pub fn layer_state(&self, layer: &str) -> CellState {
        self.layer_states
            .get(layer)
            .copied()
            .unwrap_or(CellState::Unloaded)
    }

    /// Check if all layers are loaded / active.
    pub fn all_layers_active(&self, layer_names: &[String]) -> bool {
        layer_names.iter().all(|name| {
            self.layer_states
                .get(name)
                .map_or(false, |s| *s == CellState::Active)
        })
    }

    /// Tick: increment state duration.
    pub fn tick(&mut self) {
        self.state_duration_frames = self.state_duration_frames.saturating_add(1);
    }
}

// ---------------------------------------------------------------------------
// WorldLayer
// ---------------------------------------------------------------------------

/// A named layer of world data with its own loading distance.
///
/// Layers let different categories of content stream at different ranges.
/// For example, terrain might load 5 km out, large props at 2 km, foliage
/// at 500 m, and audio at 200 m.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldLayer {
    /// Human-readable name of this layer (e.g., "terrain", "foliage").
    pub name: String,
    /// Maximum distance (world units) at which this layer is loaded.
    pub load_distance: f32,
    /// Distance at which this layer is unloaded (should be > load_distance).
    pub unload_distance: f32,
    /// Priority order: lower values load first.
    pub priority: u32,
    /// Whether this layer is enabled.
    pub enabled: bool,
    /// Optional data path override for this layer.
    pub data_path: Option<String>,
}

impl WorldLayer {
    /// Create a new world layer with default hysteresis.
    pub fn new(name: impl Into<String>, load_distance: f32, priority: u32) -> Self {
        let name = name.into();
        Self {
            name,
            load_distance,
            unload_distance: load_distance * 1.2, // 20% hysteresis by default
            priority,
            enabled: true,
            data_path: None,
        }
    }

    /// Create a layer with explicit unload distance.
    pub fn with_distances(
        name: impl Into<String>,
        load_distance: f32,
        unload_distance: f32,
        priority: u32,
    ) -> Self {
        assert!(
            unload_distance >= load_distance,
            "unload_distance must be >= load_distance"
        );
        Self {
            name: name.into(),
            load_distance,
            unload_distance,
            priority,
            enabled: true,
            data_path: None,
        }
    }

    /// Set a data path override for this layer.
    pub fn with_data_path(mut self, path: impl Into<String>) -> Self {
        self.data_path = Some(path.into());
        self
    }

    /// Returns `true` if a cell at the given distance should be loaded for
    /// this layer.
    #[inline]
    pub fn should_load(&self, distance: f32) -> bool {
        self.enabled && distance <= self.load_distance
    }

    /// Returns `true` if a cell at the given distance should be unloaded for
    /// this layer.
    #[inline]
    pub fn should_unload(&self, distance: f32) -> bool {
        distance > self.unload_distance
    }
}

impl Default for WorldLayer {
    fn default() -> Self {
        Self {
            name: "default".to_owned(),
            load_distance: 1000.0,
            unload_distance: 1200.0,
            priority: 100,
            enabled: true,
            data_path: None,
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingOps
// ---------------------------------------------------------------------------

/// Output of a partition update: lists of cells that should be loaded or
/// unloaded.
#[derive(Debug, Clone, Default)]
pub struct StreamingOps {
    /// Cells that should be loaded, ordered by priority (highest first).
    pub load: Vec<CellCoord>,
    /// Cells that should be unloaded, ordered by priority (lowest first).
    pub unload: Vec<CellCoord>,
    /// Per-layer operations: (coord, layer_name, should_load).
    pub layer_ops: Vec<(CellCoord, String, bool)>,
}

impl StreamingOps {
    /// Returns `true` if there is nothing to do.
    pub fn is_empty(&self) -> bool {
        self.load.is_empty() && self.unload.is_empty() && self.layer_ops.is_empty()
    }

    /// Total number of operations.
    pub fn operation_count(&self) -> usize {
        self.load.len() + self.unload.len() + self.layer_ops.len()
    }
}

// ---------------------------------------------------------------------------
// WorldPartition
// ---------------------------------------------------------------------------

/// Spatial partition of the game world into a regular grid.
///
/// The partition tracks all cells that have been created, evaluates distances
/// from the camera each frame, and produces [`StreamingOps`] describing which
/// cells should be loaded or unloaded.
///
/// # Hysteresis
///
/// To prevent rapid load/unload cycling when the camera sits near a boundary,
/// the unload radius is set larger than the load radius. A cell is loaded when
/// it enters the load radius and only unloaded once it exits the unload radius.
///
/// # Cell lifecycle
///
/// ```text
///           load requested          data ready           activation
///  Unloaded ──────────────> Loading ──────────> Loaded ─────────────> Active
///                                                                      │
///                                   pending_unload                     │
///                           Unloaded <──────────── PendingUnload <─────┘
///                                     serialize if dirty
/// ```
pub struct WorldPartition {
    /// Size of each cell in world units (square).
    cell_size: f32,
    /// Inverse of cell_size, cached for fast coordinate lookups.
    inv_cell_size: f32,
    /// Base load radius in world units. Cells within this distance of the
    /// camera are candidates for loading.
    load_radius: f32,
    /// Base unload radius in world units. Cells beyond this distance are
    /// candidates for unloading. Must be > load_radius.
    unload_radius: f32,
    /// Named layers with per-layer distances.
    layers: Vec<WorldLayer>,
    /// All known cells, keyed by coordinate.
    cells: HashMap<CellCoord, WorldCell>,
    /// Set of currently loaded cell coordinates (state is Loaded or Active).
    loaded_cells: HashSet<CellCoord>,
    /// Set of cells currently in the Loading state.
    loading_cells: HashSet<CellCoord>,
    /// The cell coordinate the camera was in last frame (for change detection).
    last_camera_cell: Option<CellCoord>,
    /// Maximum number of cells that can be loaded concurrently.
    max_concurrent_loads: usize,
    /// Maximum number of cells that can be unloaded per frame.
    max_unloads_per_frame: usize,
    /// Total memory budget for loaded cells (bytes).
    memory_budget: usize,
    /// Total memory currently consumed by loaded cells.
    current_memory: usize,
    /// Total number of update calls (frame counter).
    frame_counter: u64,
}

impl WorldPartition {
    /// Create a new world partition.
    ///
    /// # Parameters
    ///
    /// - `cell_size`: The side length of each square cell in world units.
    /// - `load_radius`: Cells within this distance are loaded.
    /// - `unload_radius`: Cells beyond this distance are unloaded (must be >= load_radius).
    pub fn new(cell_size: f32, load_radius: f32, unload_radius: f32) -> Self {
        assert!(cell_size > 0.0, "cell_size must be positive");
        assert!(load_radius > 0.0, "load_radius must be positive");
        assert!(
            unload_radius >= load_radius,
            "unload_radius must be >= load_radius"
        );

        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            load_radius,
            unload_radius,
            layers: Vec::new(),
            cells: HashMap::new(),
            loaded_cells: HashSet::new(),
            loading_cells: HashSet::new(),
            last_camera_cell: None,
            max_concurrent_loads: 4,
            max_unloads_per_frame: 2,
            memory_budget: 512 * 1024 * 1024, // 512 MiB default
            current_memory: 0,
            frame_counter: 0,
        }
    }

    /// Builder method: set maximum concurrent loads.
    pub fn with_max_concurrent_loads(mut self, n: usize) -> Self {
        self.max_concurrent_loads = n;
        self
    }

    /// Builder method: set maximum unloads per frame.
    pub fn with_max_unloads_per_frame(mut self, n: usize) -> Self {
        self.max_unloads_per_frame = n;
        self
    }

    /// Builder method: set memory budget in bytes.
    pub fn with_memory_budget(mut self, bytes: usize) -> Self {
        self.memory_budget = bytes;
        self
    }

    /// Add a streaming layer.
    pub fn add_layer(&mut self, layer: WorldLayer) {
        // Insert sorted by priority.
        let idx = self
            .layers
            .binary_search_by_key(&layer.priority, |l| l.priority)
            .unwrap_or_else(|i| i);
        self.layers.insert(idx, layer);
    }

    /// Remove a layer by name.
    pub fn remove_layer(&mut self, name: &str) -> bool {
        if let Some(idx) = self.layers.iter().position(|l| l.name == name) {
            self.layers.remove(idx);
            true
        } else {
            false
        }
    }

    /// Get a reference to a layer by name.
    pub fn layer(&self, name: &str) -> Option<&WorldLayer> {
        self.layers.iter().find(|l| l.name == name)
    }

    /// Get a mutable reference to a layer by name.
    pub fn layer_mut(&mut self, name: &str) -> Option<&mut WorldLayer> {
        self.layers.iter_mut().find(|l| l.name == name)
    }

    /// All layers.
    pub fn layers(&self) -> &[WorldLayer] {
        &self.layers
    }

    // -- Coordinate conversion ---------------------------------------------

    /// Convert a world-space position to a cell coordinate.
    #[inline]
    pub fn world_to_cell(&self, position: Vec3) -> CellCoord {
        CellCoord {
            x: (position.x * self.inv_cell_size).floor() as i32,
            z: (position.z * self.inv_cell_size).floor() as i32,
        }
    }

    /// Convert a cell coordinate to the world-space center of that cell.
    #[inline]
    pub fn cell_to_world(&self, coord: CellCoord) -> Vec3 {
        Vec3::new(
            coord.x as f32 * self.cell_size + self.cell_size * 0.5,
            0.0,
            coord.z as f32 * self.cell_size + self.cell_size * 0.5,
        )
    }

    /// Cell size in world units.
    #[inline]
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    /// Current load radius.
    #[inline]
    pub fn load_radius(&self) -> f32 {
        self.load_radius
    }

    /// Current unload radius.
    #[inline]
    pub fn unload_radius(&self) -> f32 {
        self.unload_radius
    }

    /// Set new load/unload radii.
    pub fn set_radii(&mut self, load: f32, unload: f32) {
        assert!(unload >= load);
        self.load_radius = load;
        self.unload_radius = unload;
    }

    // -- Cell access -------------------------------------------------------

    /// Get a reference to a cell by coordinate.
    pub fn cell(&self, coord: &CellCoord) -> Option<&WorldCell> {
        self.cells.get(coord)
    }

    /// Get a mutable reference to a cell by coordinate.
    pub fn cell_mut(&mut self, coord: &CellCoord) -> Option<&mut WorldCell> {
        self.cells.get_mut(coord)
    }

    /// Get or create a cell at the given coordinate.
    pub fn get_or_create_cell(&mut self, coord: CellCoord) -> &mut WorldCell {
        self.cells
            .entry(coord)
            .or_insert_with(|| WorldCell::new(coord, self.cell_size))
    }

    /// Returns an iterator over all cells.
    pub fn cells(&self) -> impl Iterator<Item = (&CellCoord, &WorldCell)> {
        self.cells.iter()
    }

    /// Returns the set of currently loaded cell coordinates.
    pub fn loaded_cells(&self) -> &HashSet<CellCoord> {
        &self.loaded_cells
    }

    /// Number of loaded cells.
    pub fn loaded_cell_count(&self) -> usize {
        self.loaded_cells.len()
    }

    /// Number of cells currently loading.
    pub fn loading_cell_count(&self) -> usize {
        self.loading_cells.len()
    }

    /// Total cell count (including unloaded ones that have been created).
    pub fn total_cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Current memory usage from loaded cells.
    pub fn current_memory_usage(&self) -> usize {
        self.current_memory
    }

    /// Memory budget.
    pub fn memory_budget(&self) -> usize {
        self.memory_budget
    }

    // -- Entity registration -----------------------------------------------

    /// Register an entity at a world-space position. The entity is added to
    /// the cell that contains the position.
    pub fn register_entity(&mut self, entity: Entity, position: Vec3) -> CellCoord {
        let coord = self.world_to_cell(position);
        let cell = self.get_or_create_cell(coord);
        cell.add_entity(entity);
        coord
    }

    /// Move an entity from one position to another, potentially changing cells.
    /// Returns `(old_cell, new_cell)`.
    pub fn move_entity(
        &mut self,
        entity: Entity,
        old_position: Vec3,
        new_position: Vec3,
    ) -> (CellCoord, CellCoord) {
        let old_coord = self.world_to_cell(old_position);
        let new_coord = self.world_to_cell(new_position);

        if old_coord != new_coord {
            // Remove from old cell.
            if let Some(old_cell) = self.cells.get_mut(&old_coord) {
                old_cell.remove_entity(entity);
            }
            // Add to new cell.
            let new_cell = self.get_or_create_cell(new_coord);
            new_cell.add_entity(entity);
        }

        (old_coord, new_coord)
    }

    /// Unregister an entity from its cell.
    pub fn unregister_entity(&mut self, entity: Entity, position: Vec3) {
        let coord = self.world_to_cell(position);
        if let Some(cell) = self.cells.get_mut(&coord) {
            cell.remove_entity(entity);
        }
    }

    // -- Main update -------------------------------------------------------

    /// Evaluate streaming based on the current camera position.
    ///
    /// Returns a [`StreamingOps`] struct describing which cells should be
    /// loaded and which should be unloaded. The caller (usually
    /// [`StreamingManager`]) is responsible for actually performing the
    /// load/unload work.
    ///
    /// This method:
    /// 1. Determines which cell the camera is in.
    /// 2. Computes the set of cells within the load radius.
    /// 3. Identifies cells that are loaded but outside the unload radius.
    /// 4. Applies per-layer distance overrides.
    /// 5. Respects concurrent load limits and memory budget.
    /// 6. Sorts results by priority (closest cells load first).
    pub fn update(&mut self, camera_pos: Vec3) -> StreamingOps {
        self.frame_counter += 1;
        let camera_cell = self.world_to_cell(camera_pos);
        let camera_changed = self.last_camera_cell != Some(camera_cell);
        self.last_camera_cell = Some(camera_cell);

        // Tick all loaded cells.
        for coord in &self.loaded_cells {
            if let Some(cell) = self.cells.get_mut(coord) {
                cell.tick();
            }
        }

        let mut ops = StreamingOps::default();

        // Calculate how many cells fit within the load radius.
        let cell_radius = (self.load_radius * self.inv_cell_size).ceil() as i32 + 1;
        let unload_radius_sq = self.unload_radius * self.unload_radius;
        let load_radius_sq = self.load_radius * self.load_radius;

        // --- Pass 1: Find cells that should be loaded ----------------------
        let mut candidates: Vec<(CellCoord, f32)> = Vec::new();

        for dx in -cell_radius..=cell_radius {
            for dz in -cell_radius..=cell_radius {
                let coord = CellCoord::new(camera_cell.x + dx, camera_cell.z + dz);
                let world_center = self.cell_to_world(coord);
                let dist_sq = {
                    let ddx = world_center.x - camera_pos.x;
                    let ddz = world_center.z - camera_pos.z;
                    ddx * ddx + ddz * ddz
                };

                if dist_sq > load_radius_sq {
                    continue;
                }

                // Check if already loaded or loading.
                let already_loaded = self.loaded_cells.contains(&coord);
                let already_loading = self.loading_cells.contains(&coord);

                if !already_loaded && !already_loading {
                    candidates.push((coord, dist_sq));
                }

                // Update streaming priority on existing cells.
                if let Some(cell) = self.cells.get_mut(&coord) {
                    cell.streaming_priority = dist_sq;
                }
            }
        }

        // Sort candidates by distance (closest first).
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Limit concurrent loads.
        let available_load_slots = self
            .max_concurrent_loads
            .saturating_sub(self.loading_cells.len());
        candidates.truncate(available_load_slots);

        // Check memory budget.
        for (coord, _dist_sq) in &candidates {
            // Estimate memory for this cell (use existing data or default).
            let estimated = self
                .cells
                .get(coord)
                .map(|c| c.memory_usage)
                .unwrap_or(1024 * 1024); // 1 MiB default estimate

            if self.current_memory + estimated > self.memory_budget {
                // Over budget -- stop loading more cells.
                break;
            }

            ops.load.push(*coord);
        }

        // --- Pass 2: Find cells that should be unloaded --------------------
        let mut unload_candidates: Vec<(CellCoord, f32)> = Vec::new();

        for coord in &self.loaded_cells {
            let world_center = self.cell_to_world(*coord);
            let dist_sq = {
                let dx = world_center.x - camera_pos.x;
                let dz = world_center.z - camera_pos.z;
                dx * dx + dz * dz
            };

            if dist_sq > unload_radius_sq {
                unload_candidates.push((*coord, dist_sq));
            }
        }

        // Sort by distance (farthest first) so we unload the most distant cells.
        unload_candidates
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        unload_candidates.truncate(self.max_unloads_per_frame);

        for (coord, _dist_sq) in &unload_candidates {
            ops.unload.push(*coord);
        }

        // --- Pass 3: Per-layer operations ----------------------------------
        if !self.layers.is_empty() {
            for coord in self.loaded_cells.iter().chain(ops.load.iter()) {
                let world_center = self.cell_to_world(*coord);
                let dist = {
                    let dx = world_center.x - camera_pos.x;
                    let dz = world_center.z - camera_pos.z;
                    (dx * dx + dz * dz).sqrt()
                };

                for layer in &self.layers {
                    if !layer.enabled {
                        continue;
                    }

                    let current_layer_state = self
                        .cells
                        .get(coord)
                        .map(|c| c.layer_state(&layer.name))
                        .unwrap_or(CellState::Unloaded);

                    if layer.should_load(dist) && current_layer_state == CellState::Unloaded {
                        ops.layer_ops.push((*coord, layer.name.clone(), true));
                    } else if layer.should_unload(dist) && current_layer_state.is_resident() {
                        ops.layer_ops.push((*coord, layer.name.clone(), false));
                    }
                }
            }
        }

        // --- Memory pressure unloads --------------------------------------
        // If we are over budget even after regular unloads, force-unload the
        // farthest loaded cells until we are back under budget.
        if self.current_memory > self.memory_budget {
            let mut all_loaded: Vec<(CellCoord, f32)> = self
                .loaded_cells
                .iter()
                .filter(|c| !ops.unload.contains(c))
                .map(|c| {
                    let center = self.cell_to_world(*c);
                    let d = {
                        let dx = center.x - camera_pos.x;
                        let dz = center.z - camera_pos.z;
                        dx * dx + dz * dz
                    };
                    (*c, d)
                })
                .collect();

            all_loaded.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut freed = 0usize;
            let overshoot = self.current_memory.saturating_sub(self.memory_budget);

            for (coord, _) in all_loaded {
                if freed >= overshoot {
                    break;
                }
                let mem = self
                    .cells
                    .get(&coord)
                    .map(|c| c.memory_usage)
                    .unwrap_or(0);
                freed += mem;
                if !ops.unload.contains(&coord) {
                    ops.unload.push(coord);
                }
            }
        }

        ops
    }

    /// Notify the partition that a cell has started loading.
    pub fn notify_loading(&mut self, coord: CellCoord) {
        self.loading_cells.insert(coord);
        let cell = self.get_or_create_cell(coord);
        cell.set_state(CellState::Loading);
    }

    /// Notify the partition that a cell has finished loading.
    pub fn notify_loaded(&mut self, coord: CellCoord, memory_bytes: usize) {
        self.loading_cells.remove(&coord);
        self.loaded_cells.insert(coord);
        self.current_memory += memory_bytes;

        if let Some(cell) = self.cells.get_mut(&coord) {
            cell.set_state(CellState::Loaded);
            cell.memory_usage = memory_bytes;
        }
    }

    /// Notify the partition that a cell has been activated (entities ticking).
    pub fn notify_activated(&mut self, coord: CellCoord) {
        if let Some(cell) = self.cells.get_mut(&coord) {
            cell.set_state(CellState::Active);
        }
    }

    /// Notify the partition that a cell has been fully unloaded.
    pub fn notify_unloaded(&mut self, coord: CellCoord) {
        self.loaded_cells.remove(&coord);
        self.loading_cells.remove(&coord);

        if let Some(cell) = self.cells.get_mut(&coord) {
            self.current_memory = self.current_memory.saturating_sub(cell.memory_usage);
            cell.memory_usage = 0;
            cell.entities.clear();
            cell.layer_states.clear();
            cell.set_state(CellState::Unloaded);
        }
    }

    /// Notify the partition that a cell has entered the pending-unload state.
    pub fn notify_pending_unload(&mut self, coord: CellCoord) {
        if let Some(cell) = self.cells.get_mut(&coord) {
            cell.set_state(CellState::PendingUnload);
        }
    }

    /// Force-unload all cells.
    pub fn unload_all(&mut self) -> Vec<CellCoord> {
        let coords: Vec<CellCoord> = self.loaded_cells.iter().copied().collect();
        for coord in &coords {
            self.notify_unloaded(*coord);
        }
        self.loading_cells.clear();
        self.current_memory = 0;
        coords
    }

    /// Reset the partition entirely, clearing all cells.
    pub fn reset(&mut self) {
        self.cells.clear();
        self.loaded_cells.clear();
        self.loading_cells.clear();
        self.last_camera_cell = None;
        self.current_memory = 0;
        self.frame_counter = 0;
    }

    /// Get statistics about the current partition state.
    pub fn stats(&self) -> PartitionStats {
        PartitionStats {
            total_cells: self.cells.len(),
            loaded_cells: self.loaded_cells.len(),
            loading_cells: self.loading_cells.len(),
            memory_used: self.current_memory,
            memory_budget: self.memory_budget,
            frame_counter: self.frame_counter,
            cell_size: self.cell_size,
            load_radius: self.load_radius,
            unload_radius: self.unload_radius,
        }
    }
}

// ---------------------------------------------------------------------------
// PartitionStats
// ---------------------------------------------------------------------------

/// Summary statistics for a [`WorldPartition`].
#[derive(Debug, Clone)]
pub struct PartitionStats {
    /// Total number of cells ever created.
    pub total_cells: usize,
    /// Number of cells currently loaded or active.
    pub loaded_cells: usize,
    /// Number of cells currently in the loading pipeline.
    pub loading_cells: usize,
    /// Memory currently consumed by loaded cells (bytes).
    pub memory_used: usize,
    /// Total memory budget (bytes).
    pub memory_budget: usize,
    /// Frame counter.
    pub frame_counter: u64,
    /// Cell size in world units.
    pub cell_size: f32,
    /// Load radius.
    pub load_radius: f32,
    /// Unload radius.
    pub unload_radius: f32,
}

impl fmt::Display for PartitionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Partition: {} loaded / {} loading / {} total cells, \
             mem {:.1} MiB / {:.1} MiB, cell_size={}, radii={}/{}",
            self.loaded_cells,
            self.loading_cells,
            self.total_cells,
            self.memory_used as f64 / (1024.0 * 1024.0),
            self.memory_budget as f64 / (1024.0 * 1024.0),
            self.cell_size,
            self.load_radius,
            self.unload_radius,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_to_cell_positive() {
        let partition = WorldPartition::new(256.0, 1024.0, 1280.0);
        let coord = partition.world_to_cell(Vec3::new(100.0, 0.0, 300.0));
        assert_eq!(coord, CellCoord::new(0, 1));
    }

    #[test]
    fn world_to_cell_negative() {
        let partition = WorldPartition::new(256.0, 1024.0, 1280.0);
        let coord = partition.world_to_cell(Vec3::new(-1.0, 0.0, -1.0));
        assert_eq!(coord, CellCoord::new(-1, -1));
    }

    #[test]
    fn cell_to_world_round_trip() {
        let partition = WorldPartition::new(100.0, 500.0, 600.0);
        let coord = CellCoord::new(3, -2);
        let world_pos = partition.cell_to_world(coord);
        let back = partition.world_to_cell(world_pos);
        assert_eq!(coord, back);
    }

    #[test]
    fn update_loads_nearby_cells() {
        let mut partition = WorldPartition::new(100.0, 250.0, 300.0);
        let ops = partition.update(Vec3::new(50.0, 0.0, 50.0));
        assert!(!ops.load.is_empty(), "should request loading nearby cells");
        assert!(ops.unload.is_empty(), "nothing to unload yet");
    }

    #[test]
    fn update_unloads_far_cells() {
        let mut partition = WorldPartition::new(100.0, 200.0, 250.0);

        // Load a cell far away by manually marking it loaded.
        let far_coord = CellCoord::new(100, 100);
        partition.get_or_create_cell(far_coord);
        partition.notify_loaded(far_coord, 1024);

        let ops = partition.update(Vec3::ZERO);
        assert!(
            ops.unload.contains(&far_coord),
            "far cell should be unloaded"
        );
    }

    #[test]
    fn hysteresis_prevents_thrashing() {
        let mut partition = WorldPartition::new(100.0, 200.0, 300.0);

        // Camera at origin: load cells within 200 units.
        let ops = partition.update(Vec3::ZERO);
        for coord in &ops.load {
            partition.notify_loaded(*coord, 1024);
        }

        // Move camera to the edge of the load radius.
        let ops2 = partition.update(Vec3::new(200.0, 0.0, 0.0));
        // Cells near origin should NOT be unloaded yet because they are still
        // within the 300-unit unload radius.
        let origin_cell = CellCoord::new(0, 0);
        assert!(
            !ops2.unload.contains(&origin_cell),
            "origin cell should remain loaded due to hysteresis"
        );
    }

    #[test]
    fn cell_neighbor_iterator() {
        let center = CellCoord::new(0, 0);
        let neighbors: Vec<_> = center.neighbors(1).collect();
        assert_eq!(neighbors.len(), 9); // 3x3 grid
    }

    #[test]
    fn cell_contains_point() {
        let cell = WorldCell::new(CellCoord::new(0, 0), 100.0);
        assert!(cell.contains_xz(Vec3::new(50.0, 0.0, 50.0)));
        assert!(!cell.contains_xz(Vec3::new(150.0, 0.0, 50.0)));
    }

    #[test]
    fn layer_distance_thresholds() {
        let layer = WorldLayer::new("foliage", 500.0, 0);
        assert!(layer.should_load(400.0));
        assert!(!layer.should_load(600.0));
        assert!(!layer.should_unload(550.0));
        assert!(layer.should_unload(700.0));
    }
}
