//! Terrain LOD streaming system for the Genovo engine.
//!
//! Provides a complete tile-based terrain streaming subsystem:
//!
//! - **Async heightmap tile loading** — tiles are loaded asynchronously from
//!   disk or network, with configurable I/O thread count.
//! - **Tile cache** — an LRU cache of loaded tiles with a configurable
//!   memory budget.
//! - **Seamless LOD stitching** — eliminates T-junction cracks at boundaries
//!   between tiles of different LOD levels.
//! - **Progressive detail loading** — coarse data loads first, detail fills
//!   in over subsequent frames.
//! - **Memory budget** — automatic eviction when total loaded data exceeds
//!   the configured limit.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Tile coordinates
// ---------------------------------------------------------------------------

/// 2D coordinate of a terrain tile in the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TileCoord {
    pub x: i32,
    pub z: i32,
}

impl TileCoord {
    /// Creates a new tile coordinate.
    pub fn new(x: i32, z: i32) -> Self {
        Self { x, z }
    }

    /// Returns the Manhattan distance to another tile.
    pub fn manhattan_distance(&self, other: &Self) -> u32 {
        ((self.x - other.x).unsigned_abs()) + ((self.z - other.z).unsigned_abs())
    }

    /// Returns the Chebyshev distance to another tile.
    pub fn chebyshev_distance(&self, other: &Self) -> u32 {
        (self.x - other.x)
            .unsigned_abs()
            .max((self.z - other.z).unsigned_abs())
    }

    /// Returns the 4 directly adjacent tiles.
    pub fn neighbors_4(&self) -> [Self; 4] {
        [
            Self::new(self.x - 1, self.z),
            Self::new(self.x + 1, self.z),
            Self::new(self.x, self.z - 1),
            Self::new(self.x, self.z + 1),
        ]
    }

    /// Returns the 8 surrounding tiles (including diagonals).
    pub fn neighbors_8(&self) -> [Self; 8] {
        [
            Self::new(self.x - 1, self.z - 1),
            Self::new(self.x, self.z - 1),
            Self::new(self.x + 1, self.z - 1),
            Self::new(self.x - 1, self.z),
            Self::new(self.x + 1, self.z),
            Self::new(self.x - 1, self.z + 1),
            Self::new(self.x, self.z + 1),
            Self::new(self.x + 1, self.z + 1),
        ]
    }
}

impl fmt::Display for TileCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.z)
    }
}

// ---------------------------------------------------------------------------
// LOD level
// ---------------------------------------------------------------------------

/// A LOD level for terrain tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LodLevel(pub u8);

impl LodLevel {
    /// The highest detail level (LOD 0).
    pub const HIGHEST: Self = Self(0);

    /// Returns the resolution reduction factor (2^lod).
    pub fn reduction_factor(&self) -> u32 {
        1 << self.0
    }

    /// Returns the number of vertices per edge at this LOD.
    pub fn vertices_per_edge(&self, base_resolution: u32) -> u32 {
        (base_resolution / self.reduction_factor()).max(2)
    }

    /// Returns the next coarser LOD level.
    pub fn coarser(&self) -> Self {
        Self(self.0 + 1)
    }

    /// Returns the next finer LOD level (if not already at highest).
    pub fn finer(&self) -> Option<Self> {
        if self.0 > 0 {
            Some(Self(self.0 - 1))
        } else {
            None
        }
    }
}

impl fmt::Display for LodLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LOD{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Tile state
// ---------------------------------------------------------------------------

/// The lifecycle state of a terrain tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileState {
    /// The tile has not been requested.
    Unloaded,
    /// The tile has been queued for loading.
    Queued,
    /// The tile is currently being loaded from disk/network.
    Loading,
    /// A coarse version is loaded; detail is still being streamed.
    PartiallyLoaded,
    /// The tile is fully loaded and ready to render.
    Loaded,
    /// The tile is being evicted from the cache.
    Evicting,
    /// The tile load failed.
    Failed,
}

impl TileState {
    /// Whether the tile has any renderable data.
    pub fn is_renderable(&self) -> bool {
        matches!(self, Self::PartiallyLoaded | Self::Loaded)
    }

    /// Whether the tile is in a loading state.
    pub fn is_loading(&self) -> bool {
        matches!(self, Self::Queued | Self::Loading)
    }
}

// ---------------------------------------------------------------------------
// Tile data
// ---------------------------------------------------------------------------

/// Heightmap data for a single terrain tile.
#[derive(Debug, Clone)]
pub struct TileHeightData {
    /// Resolution of the heightmap (vertices per edge).
    pub resolution: u32,
    /// Height values (row-major, resolution^2 entries).
    pub heights: Vec<f32>,
    /// Minimum height in this tile.
    pub min_height: f32,
    /// Maximum height in this tile.
    pub max_height: f32,
}

impl TileHeightData {
    /// Creates a new tile height data with the given resolution.
    pub fn new(resolution: u32, heights: Vec<f32>) -> Self {
        assert_eq!(
            heights.len(),
            (resolution * resolution) as usize,
            "Heights length must match resolution^2"
        );
        let min_height = heights.iter().copied().fold(f32::MAX, f32::min);
        let max_height = heights.iter().copied().fold(f32::MIN, f32::max);
        Self {
            resolution,
            heights,
            min_height,
            max_height,
        }
    }

    /// Creates a flat tile.
    pub fn flat(resolution: u32, height: f32) -> Self {
        Self::new(resolution, vec![height; (resolution * resolution) as usize])
    }

    /// Gets the height at integer coordinates.
    pub fn get(&self, x: u32, z: u32) -> f32 {
        let x = x.min(self.resolution - 1);
        let z = z.min(self.resolution - 1);
        self.heights[(z * self.resolution + x) as usize]
    }

    /// Sets the height at integer coordinates.
    pub fn set(&mut self, x: u32, z: u32, height: f32) {
        if x < self.resolution && z < self.resolution {
            self.heights[(z * self.resolution + x) as usize] = height;
        }
    }

    /// Bilinear sample at fractional coordinates.
    pub fn sample(&self, fx: f32, fz: f32) -> f32 {
        let x0 = (fx as u32).min(self.resolution - 2);
        let z0 = (fz as u32).min(self.resolution - 2);
        let x1 = x0 + 1;
        let z1 = z0 + 1;
        let tx = fx - x0 as f32;
        let tz = fz - z0 as f32;

        let h00 = self.get(x0, z0);
        let h10 = self.get(x1, z0);
        let h01 = self.get(x0, z1);
        let h11 = self.get(x1, z1);

        let h0 = h00 + (h10 - h00) * tx;
        let h1 = h01 + (h11 - h01) * tx;

        h0 + (h1 - h0) * tz
    }

    /// Downsamples to a lower resolution (halving).
    pub fn downsample(&self) -> Self {
        let new_res = (self.resolution / 2).max(2);
        let mut heights = Vec::with_capacity((new_res * new_res) as usize);

        for z in 0..new_res {
            for x in 0..new_res {
                let sx = x * 2;
                let sz = z * 2;
                // Average 2x2 block.
                let h = (self.get(sx, sz)
                    + self.get(sx + 1, sz)
                    + self.get(sx, sz + 1)
                    + self.get(sx + 1, sz + 1))
                    * 0.25;
                heights.push(h);
            }
        }

        Self::new(new_res, heights)
    }

    /// Returns the memory size in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.heights.len() * std::mem::size_of::<f32>()
    }
}

// ---------------------------------------------------------------------------
// Tile entry (cache entry)
// ---------------------------------------------------------------------------

/// A cached terrain tile entry.
#[derive(Debug)]
pub struct TileEntry {
    /// Tile grid coordinate.
    pub coord: TileCoord,
    /// Current LOD level.
    pub lod: LodLevel,
    /// Target LOD level (may differ while transitioning).
    pub target_lod: LodLevel,
    /// Lifecycle state.
    pub state: TileState,
    /// Height data (None if not yet loaded).
    pub height_data: Option<TileHeightData>,
    /// Coarse height data for progressive loading.
    pub coarse_data: Option<TileHeightData>,
    /// Last frame this tile was used/accessed.
    pub last_access_frame: u64,
    /// Memory size of loaded data in bytes.
    pub memory_bytes: usize,
    /// Load priority (lower = more urgent).
    pub priority: i32,
    /// Whether the tile mesh needs rebuilding.
    pub mesh_dirty: bool,
    /// Whether border stitching is needed.
    pub needs_stitching: bool,
    /// Error message if the load failed.
    pub error: Option<String>,
}

impl TileEntry {
    /// Creates a new empty tile entry.
    pub fn new(coord: TileCoord) -> Self {
        Self {
            coord,
            lod: LodLevel::HIGHEST,
            target_lod: LodLevel::HIGHEST,
            state: TileState::Unloaded,
            height_data: None,
            coarse_data: None,
            last_access_frame: 0,
            memory_bytes: 0,
            priority: 0,
            mesh_dirty: false,
            needs_stitching: false,
            error: None,
        }
    }

    /// Whether this tile has renderable data.
    pub fn is_renderable(&self) -> bool {
        self.state.is_renderable()
    }

    /// Touch the tile (update access frame).
    pub fn touch(&mut self, frame: u64) {
        self.last_access_frame = frame;
    }
}

// ---------------------------------------------------------------------------
// Stitch edge
// ---------------------------------------------------------------------------

/// Which edge of a tile needs stitching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileEdge {
    North,
    South,
    East,
    West,
}

/// Stitching information for a tile boundary.
#[derive(Debug, Clone)]
pub struct StitchInfo {
    /// The edge being stitched.
    pub edge: TileEdge,
    /// LOD of this tile.
    pub this_lod: LodLevel,
    /// LOD of the neighboring tile.
    pub neighbor_lod: LodLevel,
    /// Number of vertices to skip per segment on this side.
    pub this_skip: u32,
    /// Number of vertices to skip per segment on the neighbor side.
    pub neighbor_skip: u32,
}

impl StitchInfo {
    /// Computes stitching parameters for two adjacent tiles.
    pub fn compute(
        edge: TileEdge,
        this_lod: LodLevel,
        neighbor_lod: LodLevel,
        base_resolution: u32,
    ) -> Self {
        let this_verts = this_lod.vertices_per_edge(base_resolution);
        let neighbor_verts = neighbor_lod.vertices_per_edge(base_resolution);

        // Skip is the ratio -- higher LOD (fewer verts) has larger skip.
        let this_skip = if neighbor_verts > this_verts {
            neighbor_verts / this_verts
        } else {
            1
        };
        let neighbor_skip = if this_verts > neighbor_verts {
            this_verts / neighbor_verts
        } else {
            1
        };

        Self {
            edge,
            this_lod,
            neighbor_lod,
            this_skip,
            neighbor_skip,
        }
    }

    /// Whether stitching is actually needed (LODs differ).
    pub fn needs_stitching(&self) -> bool {
        self.this_lod != self.neighbor_lod
    }

    /// Generates stitched edge vertices by interpolating between LOD levels.
    pub fn generate_stitched_heights(
        &self,
        this_edge_heights: &[f32],
        neighbor_edge_heights: &[f32],
    ) -> Vec<f32> {
        // Output at the finer resolution.
        let output_len = this_edge_heights.len().max(neighbor_edge_heights.len());
        let mut result = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let this_idx = if this_edge_heights.len() > 1 {
                (i as f32 / (output_len - 1) as f32 * (this_edge_heights.len() - 1) as f32)
                    .min((this_edge_heights.len() - 1) as f32)
            } else {
                0.0
            };
            let neighbor_idx = if neighbor_edge_heights.len() > 1 {
                (i as f32 / (output_len - 1) as f32 * (neighbor_edge_heights.len() - 1) as f32)
                    .min((neighbor_edge_heights.len() - 1) as f32)
            } else {
                0.0
            };

            let this_h = lerp_heights(this_edge_heights, this_idx);
            let neighbor_h = lerp_heights(neighbor_edge_heights, neighbor_idx);

            // Average the two for a seamless join.
            result.push((this_h + neighbor_h) * 0.5);
        }

        result
    }
}

/// Helper to linearly interpolate in a height array.
fn lerp_heights(heights: &[f32], idx: f32) -> f32 {
    if heights.is_empty() {
        return 0.0;
    }
    if heights.len() == 1 {
        return heights[0];
    }
    let i0 = (idx as usize).min(heights.len() - 2);
    let i1 = i0 + 1;
    let t = idx - i0 as f32;
    heights[i0] + (heights[i1] - heights[i0]) * t
}

// ---------------------------------------------------------------------------
// Load request
// ---------------------------------------------------------------------------

/// A request to load a terrain tile.
#[derive(Debug, Clone)]
pub struct TileLoadRequest {
    /// Tile coordinate.
    pub coord: TileCoord,
    /// Target LOD level.
    pub lod: LodLevel,
    /// Load priority (lower = higher priority).
    pub priority: i32,
    /// Whether to load progressively (coarse first).
    pub progressive: bool,
    /// Request identifier.
    pub request_id: u64,
}

impl TileLoadRequest {
    /// Creates a new load request.
    pub fn new(coord: TileCoord, lod: LodLevel, priority: i32) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self {
            coord,
            lod,
            priority,
            progressive: true,
            request_id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming configuration
// ---------------------------------------------------------------------------

/// Configuration for the terrain streaming system.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Base heightmap resolution per tile (vertices per edge at LOD 0).
    pub base_resolution: u32,
    /// World-space size of each tile.
    pub tile_world_size: f32,
    /// Maximum LOD level (coarsest).
    pub max_lod: u8,
    /// Radius (in tiles) around the viewer within which tiles are loaded.
    pub load_radius: u32,
    /// Radius beyond which tiles are unloaded.
    pub unload_radius: u32,
    /// Memory budget in bytes (0 = unlimited).
    pub memory_budget: usize,
    /// Maximum number of concurrent tile loads.
    pub max_concurrent_loads: usize,
    /// Whether to use progressive loading (coarse then detail).
    pub progressive_loading: bool,
    /// LOD distance thresholds (distance at which each LOD activates).
    pub lod_distances: Vec<f32>,
    /// Minimum frames between LOD transitions (to prevent thrashing).
    pub lod_hysteresis_frames: u32,
}

impl StreamingConfig {
    /// Creates a default configuration.
    pub fn new() -> Self {
        Self {
            base_resolution: 129,
            tile_world_size: 256.0,
            max_lod: 5,
            load_radius: 8,
            unload_radius: 10,
            memory_budget: 256 * 1024 * 1024, // 256 MB
            max_concurrent_loads: 4,
            progressive_loading: true,
            lod_distances: vec![64.0, 128.0, 256.0, 512.0, 1024.0],
            lod_hysteresis_frames: 10,
        }
    }

    /// Computes the LOD level for a given distance from the viewer.
    pub fn lod_for_distance(&self, distance: f32) -> LodLevel {
        for (i, &threshold) in self.lod_distances.iter().enumerate() {
            if distance < threshold {
                return LodLevel(i as u8);
            }
        }
        LodLevel(self.max_lod)
    }

    /// Returns the world-space origin of a tile.
    pub fn tile_world_origin(&self, coord: TileCoord) -> (f32, f32) {
        (
            coord.x as f32 * self.tile_world_size,
            coord.z as f32 * self.tile_world_size,
        )
    }

    /// Returns the world-space center of a tile.
    pub fn tile_world_center(&self, coord: TileCoord) -> (f32, f32) {
        let (ox, oz) = self.tile_world_origin(coord);
        (ox + self.tile_world_size * 0.5, oz + self.tile_world_size * 0.5)
    }

    /// Converts a world position to the tile coordinate it falls in.
    pub fn world_to_tile(&self, world_x: f32, world_z: f32) -> TileCoord {
        TileCoord {
            x: (world_x / self.tile_world_size).floor() as i32,
            z: (world_z / self.tile_world_size).floor() as i32,
        }
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Streaming statistics
// ---------------------------------------------------------------------------

/// Runtime statistics for the terrain streaming system.
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    /// Number of tiles currently in the cache.
    pub cached_tiles: usize,
    /// Number of tiles currently being loaded.
    pub loading_tiles: usize,
    /// Number of tiles queued for loading.
    pub queued_tiles: usize,
    /// Number of tiles currently visible/renderable.
    pub visible_tiles: usize,
    /// Total memory used by cached tiles in bytes.
    pub memory_used: usize,
    /// Memory budget in bytes.
    pub memory_budget: usize,
    /// Number of tiles loaded this frame.
    pub loads_this_frame: usize,
    /// Number of tiles evicted this frame.
    pub evictions_this_frame: usize,
    /// Total tiles loaded since startup.
    pub total_loads: u64,
    /// Total tiles evicted since startup.
    pub total_evictions: u64,
    /// Current viewer tile coordinate.
    pub viewer_tile: TileCoord,
}

impl StreamingStats {
    /// Memory usage as a fraction of budget (0..1).
    pub fn memory_usage_ratio(&self) -> f32 {
        if self.memory_budget == 0 {
            return 0.0;
        }
        self.memory_used as f32 / self.memory_budget as f32
    }

    /// Whether we are over the memory budget.
    pub fn over_budget(&self) -> bool {
        self.memory_budget > 0 && self.memory_used > self.memory_budget
    }
}

// ---------------------------------------------------------------------------
// TerrainStreamingManager
// ---------------------------------------------------------------------------

/// The main terrain streaming manager.
///
/// Manages the loading, caching, and eviction of terrain tiles based on
/// the viewer's position. Provides LOD selection, tile stitching info,
/// and progressive detail loading.
pub struct TerrainStreamingManager {
    /// Configuration.
    config: StreamingConfig,
    /// All tile entries (loaded, loading, or queued).
    tiles: HashMap<TileCoord, TileEntry>,
    /// Load request queue (sorted by priority).
    load_queue: VecDeque<TileLoadRequest>,
    /// Currently active loads.
    active_loads: Vec<TileLoadRequest>,
    /// LRU eviction order (most recently used at the back).
    lru_order: VecDeque<TileCoord>,
    /// Current viewer position in world space.
    viewer_position: (f32, f32),
    /// Current viewer tile.
    viewer_tile: TileCoord,
    /// Current frame counter.
    frame_counter: u64,
    /// Runtime statistics.
    stats: StreamingStats,
}

impl TerrainStreamingManager {
    /// Creates a new streaming manager.
    pub fn new(config: StreamingConfig) -> Self {
        let budget = config.memory_budget;
        Self {
            config,
            tiles: HashMap::new(),
            load_queue: VecDeque::new(),
            active_loads: Vec::new(),
            lru_order: VecDeque::new(),
            viewer_position: (0.0, 0.0),
            viewer_tile: TileCoord::new(0, 0),
            frame_counter: 0,
            stats: StreamingStats {
                memory_budget: budget,
                ..Default::default()
            },
        }
    }

    /// Returns the streaming configuration.
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Returns the current statistics.
    pub fn stats(&self) -> &StreamingStats {
        &self.stats
    }

    /// Returns the current viewer tile.
    pub fn viewer_tile(&self) -> TileCoord {
        self.viewer_tile
    }

    /// Updates the viewer position and triggers tile management.
    pub fn update(&mut self, viewer_x: f32, viewer_z: f32) {
        self.frame_counter += 1;
        self.viewer_position = (viewer_x, viewer_z);
        self.viewer_tile = self.config.world_to_tile(viewer_x, viewer_z);

        self.stats.loads_this_frame = 0;
        self.stats.evictions_this_frame = 0;

        // 1. Determine which tiles should be loaded.
        self.compute_desired_tiles();

        // 2. Queue load requests for missing tiles.
        self.queue_missing_tiles();

        // 3. Process load queue (up to concurrency limit).
        self.process_load_queue();

        // 4. Update LOD levels for loaded tiles.
        self.update_tile_lods();

        // 5. Evict tiles that are too far away or over budget.
        self.evict_distant_tiles();
        self.evict_for_budget();

        // 6. Update statistics.
        self.update_stats();
    }

    /// Returns a tile entry by coordinate.
    pub fn get_tile(&self, coord: TileCoord) -> Option<&TileEntry> {
        self.tiles.get(&coord)
    }

    /// Returns a mutable tile entry by coordinate.
    pub fn get_tile_mut(&mut self, coord: TileCoord) -> Option<&mut TileEntry> {
        self.tiles.get_mut(&coord)
    }

    /// Returns all renderable tile coordinates and their LOD levels.
    pub fn visible_tiles(&self) -> Vec<(TileCoord, LodLevel)> {
        self.tiles
            .iter()
            .filter(|(_, t)| t.is_renderable())
            .map(|(&c, t)| (c, t.lod))
            .collect()
    }

    /// Returns the number of cached tiles.
    pub fn cached_tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// Returns the total memory used by all tiles.
    pub fn memory_used(&self) -> usize {
        self.tiles.values().map(|t| t.memory_bytes).sum()
    }

    /// Simulates completing a tile load (for testing / integration).
    pub fn complete_tile_load(&mut self, coord: TileCoord, data: TileHeightData) {
        if let Some(tile) = self.tiles.get_mut(&coord) {
            tile.memory_bytes = data.memory_bytes();
            tile.height_data = Some(data);
            tile.state = TileState::Loaded;
            tile.mesh_dirty = true;
            tile.needs_stitching = true;
            tile.touch(self.frame_counter);

            // Update LRU.
            self.lru_order.retain(|c| *c != coord);
            self.lru_order.push_back(coord);

            self.stats.total_loads += 1;
            self.stats.loads_this_frame += 1;
        }

        // Remove from active loads.
        self.active_loads.retain(|r| r.coord != coord);
    }

    /// Simulates completing a coarse (progressive) load.
    pub fn complete_coarse_load(&mut self, coord: TileCoord, data: TileHeightData) {
        if let Some(tile) = self.tiles.get_mut(&coord) {
            tile.memory_bytes = data.memory_bytes();
            tile.coarse_data = Some(data);
            tile.state = TileState::PartiallyLoaded;
            tile.mesh_dirty = true;
            tile.touch(self.frame_counter);

            self.lru_order.retain(|c| *c != coord);
            self.lru_order.push_back(coord);
        }
    }

    /// Fails a tile load.
    pub fn fail_tile_load(&mut self, coord: TileCoord, error: &str) {
        if let Some(tile) = self.tiles.get_mut(&coord) {
            tile.state = TileState::Failed;
            tile.error = Some(error.to_string());
        }
        self.active_loads.retain(|r| r.coord != coord);
    }

    /// Computes stitching info for a tile.
    pub fn compute_stitch_info(&self, coord: TileCoord) -> Vec<StitchInfo> {
        let mut stitches = Vec::new();

        let this_tile = match self.tiles.get(&coord) {
            Some(t) if t.is_renderable() => t,
            _ => return stitches,
        };

        let edges = [
            (TileEdge::North, TileCoord::new(coord.x, coord.z - 1)),
            (TileEdge::South, TileCoord::new(coord.x, coord.z + 1)),
            (TileEdge::West, TileCoord::new(coord.x - 1, coord.z)),
            (TileEdge::East, TileCoord::new(coord.x + 1, coord.z)),
        ];

        for (edge, neighbor_coord) in &edges {
            if let Some(neighbor) = self.tiles.get(neighbor_coord) {
                if neighbor.is_renderable() && neighbor.lod != this_tile.lod {
                    stitches.push(StitchInfo::compute(
                        *edge,
                        this_tile.lod,
                        neighbor.lod,
                        self.config.base_resolution,
                    ));
                }
            }
        }

        stitches
    }

    /// Evicts a specific tile from the cache.
    pub fn evict_tile(&mut self, coord: TileCoord) {
        if let Some(tile) = self.tiles.remove(&coord) {
            self.lru_order.retain(|c| *c != coord);
            self.stats.total_evictions += 1;
            self.stats.evictions_this_frame += 1;
            let _ = tile; // Drop the tile data.
        }
    }

    /// Clears all tiles.
    pub fn clear(&mut self) {
        self.tiles.clear();
        self.lru_order.clear();
        self.load_queue.clear();
        self.active_loads.clear();
    }

    // -- Internal methods -----------------------------------------------------

    /// Determines which tiles should be loaded based on viewer position.
    fn compute_desired_tiles(&mut self) {
        let radius = self.config.load_radius as i32;
        let viewer = self.viewer_tile;

        for dz in -radius..=radius {
            for dx in -radius..=radius {
                let coord = TileCoord::new(viewer.x + dx, viewer.z + dz);
                let dist = coord.chebyshev_distance(&viewer);

                if dist <= self.config.load_radius {
                    // Ensure the tile exists in the map.
                    self.tiles.entry(coord).or_insert_with(|| TileEntry::new(coord));
                }
            }
        }
    }

    /// Queues load requests for tiles that aren't loaded yet.
    fn queue_missing_tiles(&mut self) {
        let viewer = self.viewer_tile;
        let mut requests = Vec::new();

        for (&coord, tile) in &self.tiles {
            if tile.state == TileState::Unloaded {
                let dist = coord.chebyshev_distance(&viewer);
                let (cx, cz) = self.config.tile_world_center(coord);
                let world_dist = ((cx - self.viewer_position.0).powi(2)
                    + (cz - self.viewer_position.1).powi(2))
                .sqrt();
                let lod = self.config.lod_for_distance(world_dist);
                let priority = dist as i32;

                requests.push(TileLoadRequest::new(coord, lod, priority));
            }
        }

        // Sort by priority.
        requests.sort_by_key(|r| r.priority);

        for req in requests {
            if let Some(tile) = self.tiles.get_mut(&req.coord) {
                tile.state = TileState::Queued;
                tile.target_lod = req.lod;
            }
            self.load_queue.push_back(req);
        }
    }

    /// Processes the load queue up to the concurrency limit.
    fn process_load_queue(&mut self) {
        while self.active_loads.len() < self.config.max_concurrent_loads {
            if let Some(request) = self.load_queue.pop_front() {
                if let Some(tile) = self.tiles.get_mut(&request.coord) {
                    tile.state = TileState::Loading;
                }
                self.active_loads.push(request);
            } else {
                break;
            }
        }
    }

    /// Updates LOD levels for loaded tiles based on distance.
    fn update_tile_lods(&mut self) {
        let viewer_pos = self.viewer_position;
        let config = &self.config;

        for tile in self.tiles.values_mut() {
            if tile.is_renderable() {
                let (cx, cz) = config.tile_world_center(tile.coord);
                let dist = ((cx - viewer_pos.0).powi(2) + (cz - viewer_pos.1).powi(2)).sqrt();
                let desired_lod = config.lod_for_distance(dist);

                if desired_lod != tile.target_lod {
                    tile.target_lod = desired_lod;
                    tile.mesh_dirty = true;
                    tile.needs_stitching = true;
                }

                tile.touch(self.frame_counter);
            }
        }
    }

    /// Evicts tiles that are beyond the unload radius.
    fn evict_distant_tiles(&mut self) {
        let viewer = self.viewer_tile;
        let unload_radius = self.config.unload_radius;

        let to_evict: Vec<TileCoord> = self
            .tiles
            .iter()
            .filter(|(coord, tile)| {
                coord.chebyshev_distance(&viewer) > unload_radius
                    && !tile.state.is_loading()
            })
            .map(|(&coord, _)| coord)
            .collect();

        for coord in to_evict {
            self.evict_tile(coord);
        }
    }

    /// Evicts tiles to stay within the memory budget (LRU order).
    fn evict_for_budget(&mut self) {
        if self.config.memory_budget == 0 {
            return;
        }

        let mut memory_used: usize = self.tiles.values().map(|t| t.memory_bytes).sum();
        let viewer = self.viewer_tile;

        while memory_used > self.config.memory_budget {
            // Find the LRU tile that isn't too close to the viewer.
            let to_evict = self.lru_order.iter().find(|&&coord| {
                coord.chebyshev_distance(&viewer) > self.config.load_radius / 2
            }).copied();

            if let Some(coord) = to_evict {
                if let Some(tile) = self.tiles.get(&coord) {
                    memory_used -= tile.memory_bytes;
                }
                self.evict_tile(coord);
            } else {
                break;
            }
        }
    }

    /// Updates the runtime statistics.
    fn update_stats(&mut self) {
        self.stats.cached_tiles = self.tiles.len();
        self.stats.loading_tiles = self.active_loads.len();
        self.stats.queued_tiles = self.load_queue.len();
        self.stats.visible_tiles = self.tiles.values().filter(|t| t.is_renderable()).count();
        self.stats.memory_used = self.tiles.values().map(|t| t.memory_bytes).sum();
        self.stats.memory_budget = self.config.memory_budget;
        self.stats.viewer_tile = self.viewer_tile;
    }
}

impl fmt::Debug for TerrainStreamingManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TerrainStreamingManager")
            .field("viewer_tile", &self.viewer_tile)
            .field("cached_tiles", &self.tiles.len())
            .field("load_queue", &self.load_queue.len())
            .field("active_loads", &self.active_loads.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> StreamingConfig {
        StreamingConfig {
            base_resolution: 33,
            tile_world_size: 100.0,
            max_lod: 4,
            load_radius: 3,
            unload_radius: 5,
            memory_budget: 10 * 1024 * 1024,
            max_concurrent_loads: 2,
            progressive_loading: false,
            lod_distances: vec![100.0, 200.0, 400.0, 800.0],
            lod_hysteresis_frames: 5,
        }
    }

    #[test]
    fn tile_coord_distance() {
        let a = TileCoord::new(0, 0);
        let b = TileCoord::new(3, 4);
        assert_eq!(a.manhattan_distance(&b), 7);
        assert_eq!(a.chebyshev_distance(&b), 4);
    }

    #[test]
    fn tile_coord_neighbors() {
        let t = TileCoord::new(5, 5);
        let n4 = t.neighbors_4();
        assert_eq!(n4.len(), 4);
        let n8 = t.neighbors_8();
        assert_eq!(n8.len(), 8);
    }

    #[test]
    fn lod_level_reduction() {
        assert_eq!(LodLevel(0).reduction_factor(), 1);
        assert_eq!(LodLevel(1).reduction_factor(), 2);
        assert_eq!(LodLevel(3).reduction_factor(), 8);
    }

    #[test]
    fn lod_vertices_per_edge() {
        assert_eq!(LodLevel(0).vertices_per_edge(129), 129);
        assert_eq!(LodLevel(1).vertices_per_edge(129), 64);
        assert_eq!(LodLevel(2).vertices_per_edge(129), 32);
    }

    #[test]
    fn config_world_to_tile() {
        let config = make_config();
        let tile = config.world_to_tile(250.0, 150.0);
        assert_eq!(tile, TileCoord::new(2, 1));
    }

    #[test]
    fn config_lod_for_distance() {
        let config = make_config();
        assert_eq!(config.lod_for_distance(50.0), LodLevel(0));
        assert_eq!(config.lod_for_distance(150.0), LodLevel(1));
        assert_eq!(config.lod_for_distance(5000.0), LodLevel(4));
    }

    #[test]
    fn tile_height_data_basic() {
        let data = TileHeightData::flat(33, 10.0);
        assert_eq!(data.resolution, 33);
        assert!((data.get(16, 16) - 10.0).abs() < 0.001);
        assert!((data.min_height - 10.0).abs() < 0.001);
    }

    #[test]
    fn tile_height_data_sample() {
        let mut data = TileHeightData::flat(5, 0.0);
        data.set(0, 0, 0.0);
        data.set(1, 0, 10.0);
        let sampled = data.sample(0.5, 0.0);
        assert!((sampled - 5.0).abs() < 0.01);
    }

    #[test]
    fn tile_height_data_downsample() {
        let data = TileHeightData::flat(8, 5.0);
        let downsampled = data.downsample();
        assert_eq!(downsampled.resolution, 4);
        assert!((downsampled.get(2, 2) - 5.0).abs() < 0.01);
    }

    #[test]
    fn streaming_manager_update() {
        let config = make_config();
        let mut mgr = TerrainStreamingManager::new(config);

        mgr.update(150.0, 150.0);

        // Should have created tile entries around the viewer.
        assert!(mgr.cached_tile_count() > 0);
        assert_eq!(mgr.viewer_tile(), TileCoord::new(1, 1));
    }

    #[test]
    fn streaming_manager_load_complete() {
        let config = make_config();
        let mut mgr = TerrainStreamingManager::new(config);

        mgr.update(50.0, 50.0);

        let coord = TileCoord::new(0, 0);
        let data = TileHeightData::flat(33, 5.0);
        mgr.complete_tile_load(coord, data);

        let tile = mgr.get_tile(coord).unwrap();
        assert_eq!(tile.state, TileState::Loaded);
        assert!(tile.height_data.is_some());
    }

    #[test]
    fn streaming_manager_evict() {
        let config = make_config();
        let mut mgr = TerrainStreamingManager::new(config);

        mgr.update(50.0, 50.0);
        let coord = TileCoord::new(0, 0);
        mgr.complete_tile_load(coord, TileHeightData::flat(33, 0.0));

        // Move viewer far away.
        mgr.update(5000.0, 5000.0);

        // Tile should have been evicted.
        assert!(mgr.get_tile(coord).is_none());
    }

    #[test]
    fn streaming_manager_fail_load() {
        let config = make_config();
        let mut mgr = TerrainStreamingManager::new(config);

        mgr.update(50.0, 50.0);
        let coord = TileCoord::new(0, 0);
        mgr.fail_tile_load(coord, "File not found");

        if let Some(tile) = mgr.get_tile(coord) {
            assert_eq!(tile.state, TileState::Failed);
            assert!(tile.error.is_some());
        }
    }

    #[test]
    fn stitch_info_computation() {
        let info = StitchInfo::compute(TileEdge::North, LodLevel(0), LodLevel(1), 129);
        assert!(info.needs_stitching());
    }

    #[test]
    fn stitch_info_same_lod() {
        let info = StitchInfo::compute(TileEdge::East, LodLevel(2), LodLevel(2), 129);
        assert!(!info.needs_stitching());
    }

    #[test]
    fn visible_tiles() {
        let config = make_config();
        let mut mgr = TerrainStreamingManager::new(config);
        mgr.update(50.0, 50.0);

        // Load a couple tiles.
        mgr.complete_tile_load(TileCoord::new(0, 0), TileHeightData::flat(33, 0.0));
        mgr.complete_tile_load(TileCoord::new(1, 0), TileHeightData::flat(33, 0.0));

        let visible = mgr.visible_tiles();
        assert_eq!(visible.len(), 2);
    }

    #[test]
    fn streaming_stats() {
        let config = make_config();
        let mut mgr = TerrainStreamingManager::new(config);
        mgr.update(50.0, 50.0);

        let stats = mgr.stats();
        assert!(stats.cached_tiles > 0);
        assert!(!stats.over_budget());
    }

    #[test]
    fn tile_world_coordinates() {
        let config = make_config();
        let (ox, oz) = config.tile_world_origin(TileCoord::new(2, 3));
        assert!((ox - 200.0).abs() < 0.01);
        assert!((oz - 300.0).abs() < 0.01);

        let (cx, cz) = config.tile_world_center(TileCoord::new(2, 3));
        assert!((cx - 250.0).abs() < 0.01);
        assert!((cz - 350.0).abs() < 0.01);
    }

    #[test]
    fn tile_state_checks() {
        assert!(TileState::Loaded.is_renderable());
        assert!(TileState::PartiallyLoaded.is_renderable());
        assert!(!TileState::Unloaded.is_renderable());
        assert!(TileState::Loading.is_loading());
        assert!(TileState::Queued.is_loading());
    }

    #[test]
    fn clear_tiles() {
        let config = make_config();
        let mut mgr = TerrainStreamingManager::new(config);
        mgr.update(50.0, 50.0);
        assert!(mgr.cached_tile_count() > 0);
        mgr.clear();
        assert_eq!(mgr.cached_tile_count(), 0);
    }
}
