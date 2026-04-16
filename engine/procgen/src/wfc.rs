//! # Wave Function Collapse
//!
//! A full implementation of the Wave Function Collapse (WFC) algorithm for
//! procedural generation of 2D and 3D tile-based content. WFC works by
//! iteratively collapsing cells in a grid to a single tile, then propagating
//! constraints to neighboring cells until the entire grid is solved or a
//! contradiction is reached.
//!
//! ## Key concepts
//!
//! - **Tile**: A content unit with labeled sockets on each face/edge.
//! - **Socket**: A connection label. Two tiles may be adjacent only if their
//!   facing sockets are compatible.
//! - **Cell**: A position in the grid that starts with all tiles possible and
//!   is gradually constrained until only one tile remains (collapsed).
//! - **Entropy**: Shannon entropy of the remaining tile possibilities,
//!   weighted by tile frequency. The cell with minimum entropy is collapsed
//!   next (minimum-entropy heuristic).
//! - **Propagation**: After collapsing a cell, remove incompatible tiles from
//!   its neighbors, and recursively propagate those removals.
//! - **Backtracking**: On contradiction (a cell with zero possibilities),
//!   undo the last collapse and try a different tile.

use genovo_core::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;

// ===========================================================================
// Error types
// ===========================================================================

/// Errors that can occur during WFC solving.
#[derive(Debug, Error)]
pub enum WFCError {
    /// A cell was reduced to zero possible tiles with no backtracking options.
    #[error("contradiction at cell ({x}, {y}, {z}): no valid tiles remain")]
    Contradiction { x: usize, y: usize, z: usize },

    /// The solver exceeded the maximum number of backtracking attempts.
    #[error("exceeded maximum backtrack attempts ({max})")]
    MaxBacktracksExceeded { max: usize },

    /// No tiles were provided to the solver.
    #[error("tile set is empty")]
    EmptyTileSet,

    /// Grid dimensions are invalid (zero size).
    #[error("invalid grid dimensions: {width}x{height}x{depth}")]
    InvalidDimensions {
        width: usize,
        height: usize,
        depth: usize,
    },

    /// A tile references a socket ID that has no compatibility rules.
    #[error("unknown socket ID: {0}")]
    UnknownSocket(String),
}

/// Result type alias for WFC operations.
pub type WFCStdResult<T> = Result<T, WFCError>;

// ===========================================================================
// Direction / Axis
// ===========================================================================

/// Cardinal directions for 2D grids (4-connectivity) and 3D grids (6-connectivity).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Direction {
    /// +X (East)
    PosX,
    /// -X (West)
    NegX,
    /// +Y (North in 2D, or Up in 3D)
    PosY,
    /// -Y (South in 2D, or Down in 3D)
    NegY,
    /// +Z (only used in 3D grids)
    PosZ,
    /// -Z (only used in 3D grids)
    NegZ,
}

impl Direction {
    /// All 2D directions.
    pub const DIRS_2D: [Direction; 4] = [
        Direction::PosX,
        Direction::NegX,
        Direction::PosY,
        Direction::NegY,
    ];

    /// All 3D directions.
    pub const DIRS_3D: [Direction; 6] = [
        Direction::PosX,
        Direction::NegX,
        Direction::PosY,
        Direction::NegY,
        Direction::PosZ,
        Direction::NegZ,
    ];

    /// Return the opposite direction.
    pub fn opposite(self) -> Direction {
        match self {
            Direction::PosX => Direction::NegX,
            Direction::NegX => Direction::PosX,
            Direction::PosY => Direction::NegY,
            Direction::NegY => Direction::PosY,
            Direction::PosZ => Direction::NegZ,
            Direction::NegZ => Direction::PosZ,
        }
    }

    /// Return the (dx, dy, dz) offset for this direction.
    pub fn offset(self) -> (i32, i32, i32) {
        match self {
            Direction::PosX => (1, 0, 0),
            Direction::NegX => (-1, 0, 0),
            Direction::PosY => (0, 1, 0),
            Direction::NegY => (0, -1, 0),
            Direction::PosZ => (0, 0, 1),
            Direction::NegZ => (0, 0, -1),
        }
    }
}

// ===========================================================================
// Symmetry groups
// ===========================================================================

/// Symmetry group of a tile, determining how it can be rotated/reflected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymmetryGroup {
    /// No symmetry — tile has a unique orientation (e.g., an L-shaped piece).
    None,
    /// 2-fold rotational symmetry (180 degrees looks the same).
    TwoFold,
    /// 4-fold rotational symmetry (90 degree rotations all look the same).
    FourFold,
    /// Full symmetry — all rotations and reflections are equivalent.
    Full,
}

// ===========================================================================
// WFCTile
// ===========================================================================

/// A tile definition for Wave Function Collapse.
///
/// Each tile has labeled sockets on its edges/faces. Two tiles may be placed
/// adjacent to each other only if the facing sockets are compatible.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WFCTile {
    /// Unique identifier for this tile.
    pub id: u32,

    /// Human-readable name (for debugging).
    pub name: String,

    /// Socket labels for each direction. The key is a `Direction`, and the
    /// value is the socket identifier string. For 2D tiles, only PosX, NegX,
    /// PosY, NegY are used. For 3D tiles, all six directions are used.
    pub sockets: HashMap<Direction, String>,

    /// Relative probability weight of this tile appearing. Higher values make
    /// this tile more likely to be chosen when collapsing a cell. Must be > 0.
    pub weight: f32,

    /// Symmetry group for auto-rotation generation.
    pub symmetry: SymmetryGroup,

    /// Optional metadata (e.g., terrain type, room type).
    pub metadata: HashMap<String, String>,
}

impl WFCTile {
    /// Create a new 2D tile with the given ID, name, and edge sockets.
    ///
    /// Sockets are specified in order: +X, -X, +Y, -Y (East, West, North, South).
    pub fn new_2d(id: u32, name: &str, px: &str, nx: &str, py: &str, ny: &str) -> Self {
        let mut sockets = HashMap::new();
        sockets.insert(Direction::PosX, px.to_string());
        sockets.insert(Direction::NegX, nx.to_string());
        sockets.insert(Direction::PosY, py.to_string());
        sockets.insert(Direction::NegY, ny.to_string());

        Self {
            id,
            name: name.to_string(),
            sockets,
            weight: 1.0,
            symmetry: SymmetryGroup::None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new 3D tile with the given ID, name, and face sockets.
    ///
    /// Sockets are specified in order: +X, -X, +Y, -Y, +Z, -Z.
    pub fn new_3d(
        id: u32,
        name: &str,
        px: &str,
        nx: &str,
        py: &str,
        ny: &str,
        pz: &str,
        nz: &str,
    ) -> Self {
        let mut sockets = HashMap::new();
        sockets.insert(Direction::PosX, px.to_string());
        sockets.insert(Direction::NegX, nx.to_string());
        sockets.insert(Direction::PosY, py.to_string());
        sockets.insert(Direction::NegY, ny.to_string());
        sockets.insert(Direction::PosZ, pz.to_string());
        sockets.insert(Direction::NegZ, nz.to_string());

        Self {
            id,
            name: name.to_string(),
            sockets,
            weight: 1.0,
            symmetry: SymmetryGroup::None,
            metadata: HashMap::new(),
        }
    }

    /// Set the weight of this tile and return self (builder pattern).
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.max(0.001);
        self
    }

    /// Set the symmetry group and return self.
    pub fn with_symmetry(mut self, symmetry: SymmetryGroup) -> Self {
        self.symmetry = symmetry;
        self
    }

    /// Add a metadata key-value pair.
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Get the socket label for a given direction, or empty string if not set.
    pub fn socket(&self, dir: Direction) -> &str {
        self.sockets.get(&dir).map(|s| s.as_str()).unwrap_or("")
    }

    /// Create a 90-degree clockwise rotation of this 2D tile.
    ///
    /// The rotation maps: +X -> +Y, +Y -> -X, -X -> -Y, -Y -> +X.
    pub fn rotate_cw_2d(&self, new_id: u32) -> Self {
        let mut sockets = HashMap::new();
        // Clockwise rotation: what was facing +X now faces +Y, etc.
        sockets.insert(
            Direction::PosX,
            self.socket(Direction::NegY).to_string(),
        );
        sockets.insert(
            Direction::PosY,
            self.socket(Direction::PosX).to_string(),
        );
        sockets.insert(
            Direction::NegX,
            self.socket(Direction::PosY).to_string(),
        );
        sockets.insert(
            Direction::NegY,
            self.socket(Direction::NegX).to_string(),
        );

        Self {
            id: new_id,
            name: format!("{}_rot90", self.name),
            sockets,
            weight: self.weight,
            symmetry: self.symmetry,
            metadata: self.metadata.clone(),
        }
    }

    /// Create a horizontal mirror of this 2D tile (flip along Y axis).
    pub fn mirror_x_2d(&self, new_id: u32) -> Self {
        let mut sockets = HashMap::new();
        sockets.insert(Direction::PosX, self.socket(Direction::NegX).to_string());
        sockets.insert(Direction::NegX, self.socket(Direction::PosX).to_string());
        sockets.insert(Direction::PosY, self.socket(Direction::PosY).to_string());
        sockets.insert(Direction::NegY, self.socket(Direction::NegY).to_string());

        Self {
            id: new_id,
            name: format!("{}_mirX", self.name),
            sockets,
            weight: self.weight,
            symmetry: self.symmetry,
            metadata: self.metadata.clone(),
        }
    }
}

// ===========================================================================
// WFCConstraints
// ===========================================================================

/// Adjacency constraints for WFC: which sockets can connect to which.
///
/// By default, sockets with the same name are compatible (self-matching).
/// Additional compatibility rules can be added manually.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WFCConstraints {
    /// Set of compatible socket pairs. If (a, b) is in the set, then socket
    /// `a` on one tile can face socket `b` on an adjacent tile.
    compatible_pairs: HashSet<(String, String)>,

    /// If true, sockets with the same name automatically match.
    pub auto_match_same: bool,

    /// Per-direction overrides: some socket pairs may only be valid in
    /// specific directions.
    direction_overrides: HashMap<Direction, HashSet<(String, String)>>,
}

impl WFCConstraints {
    /// Create a new constraint set with auto-matching enabled.
    pub fn new() -> Self {
        Self {
            compatible_pairs: HashSet::new(),
            auto_match_same: true,
            direction_overrides: HashMap::new(),
        }
    }

    /// Create constraints with auto-matching disabled.
    pub fn manual() -> Self {
        Self {
            compatible_pairs: HashSet::new(),
            auto_match_same: false,
            direction_overrides: HashMap::new(),
        }
    }

    /// Add a bidirectional compatibility rule: socket `a` can face socket `b`
    /// and vice versa.
    pub fn add_compatible(&mut self, a: &str, b: &str) {
        self.compatible_pairs
            .insert((a.to_string(), b.to_string()));
        self.compatible_pairs
            .insert((b.to_string(), a.to_string()));
    }

    /// Add a unidirectional compatibility rule: socket `a` can face socket `b`
    /// but not necessarily the reverse.
    pub fn add_compatible_one_way(&mut self, a: &str, b: &str) {
        self.compatible_pairs
            .insert((a.to_string(), b.to_string()));
    }

    /// Add a direction-specific compatibility override.
    pub fn add_direction_compatible(&mut self, dir: Direction, a: &str, b: &str) {
        let entry = self.direction_overrides.entry(dir).or_default();
        entry.insert((a.to_string(), b.to_string()));
        // Also add the reverse for the opposite direction.
        let opp_entry = self.direction_overrides.entry(dir.opposite()).or_default();
        opp_entry.insert((b.to_string(), a.to_string()));
    }

    /// Auto-derive constraints from a tile set by matching socket names.
    ///
    /// This scans all tiles and for each direction, records which tile pairs
    /// have compatible sockets facing each other.
    pub fn derive_from_tiles(&mut self, tiles: &[WFCTile]) {
        for tile in tiles {
            for (_, socket) in &tile.sockets {
                // Every socket is self-compatible when auto_match_same is true.
                if self.auto_match_same {
                    self.compatible_pairs
                        .insert((socket.clone(), socket.clone()));
                }
            }
        }
    }

    /// Check if two sockets are compatible in the given direction.
    ///
    /// `from_socket` is the socket on the source cell's face pointing toward
    /// the neighbor. `to_socket` is the socket on the neighbor's face pointing
    /// back toward the source.
    pub fn are_compatible(&self, from_socket: &str, to_socket: &str, dir: Direction) -> bool {
        // Check direction-specific overrides first.
        if let Some(dir_set) = self.direction_overrides.get(&dir) {
            if dir_set.contains(&(from_socket.to_string(), to_socket.to_string())) {
                return true;
            }
        }

        // Check global compatibility.
        if self
            .compatible_pairs
            .contains(&(from_socket.to_string(), to_socket.to_string()))
        {
            return true;
        }

        // Auto-match same sockets.
        if self.auto_match_same && from_socket == to_socket {
            return true;
        }

        false
    }
}

impl Default for WFCConstraints {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// WFC Cell
// ===========================================================================

/// A single cell in the WFC grid.
#[derive(Debug, Clone)]
struct WFCCell {
    /// Set of tile IDs that are still possible for this cell.
    possible: Vec<u32>,
    /// Whether this cell has been collapsed to a single tile.
    collapsed: bool,
    /// The chosen tile ID (valid only when `collapsed` is true).
    chosen_tile: Option<u32>,
}

impl WFCCell {
    /// Create a new cell with all tiles possible.
    fn new(all_tile_ids: &[u32]) -> Self {
        Self {
            possible: all_tile_ids.to_vec(),
            collapsed: false,
            chosen_tile: None,
        }
    }

    /// Calculate Shannon entropy of this cell based on tile weights.
    ///
    /// Entropy = log(sum_w) - (sum(w * log(w))) / sum_w
    /// where w is the weight of each possible tile.
    fn entropy(&self, tile_weights: &HashMap<u32, f32>) -> f64 {
        if self.collapsed || self.possible.len() <= 1 {
            return 0.0;
        }

        let mut sum_w: f64 = 0.0;
        let mut sum_w_log_w: f64 = 0.0;

        for &tid in &self.possible {
            let w = *tile_weights.get(&tid).unwrap_or(&1.0) as f64;
            if w > 0.0 {
                sum_w += w;
                sum_w_log_w += w * w.ln();
            }
        }

        if sum_w <= 0.0 {
            return 0.0;
        }

        sum_w.ln() - sum_w_log_w / sum_w
    }

    /// Collapse this cell to a specific tile.
    fn collapse_to(&mut self, tile_id: u32) {
        self.possible = vec![tile_id];
        self.collapsed = true;
        self.chosen_tile = Some(tile_id);
    }

    /// Remove a tile from the possible set. Returns true if it was removed.
    fn remove_tile(&mut self, tile_id: u32) -> bool {
        if let Some(pos) = self.possible.iter().position(|&t| t == tile_id) {
            self.possible.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Check if this cell has contradicted (no possibilities left).
    fn is_contradiction(&self) -> bool {
        self.possible.is_empty() && !self.collapsed
    }

    /// Number of remaining possibilities.
    fn num_possible(&self) -> usize {
        self.possible.len()
    }
}

// ===========================================================================
// WFCGrid
// ===========================================================================

/// A grid of WFC cells, supporting both 2D and 3D layouts.
///
/// For 2D grids, `depth` is 1. The grid uses a flat array indexed as
/// `z * (width * height) + y * width + x`.
#[derive(Debug, Clone)]
pub struct WFCGrid {
    /// Grid width (X dimension).
    pub width: usize,
    /// Grid height (Y dimension).
    pub height: usize,
    /// Grid depth (Z dimension; 1 for 2D grids).
    pub depth: usize,

    /// Flat array of cells.
    cells: Vec<WFCCell>,
}

impl WFCGrid {
    /// Create a new 2D grid.
    pub fn new_2d(width: usize, height: usize, all_tile_ids: &[u32]) -> Self {
        let total = width * height;
        let cells = (0..total)
            .map(|_| WFCCell::new(all_tile_ids))
            .collect();

        Self {
            width,
            height,
            depth: 1,
            cells,
        }
    }

    /// Create a new 3D grid.
    pub fn new_3d(width: usize, height: usize, depth: usize, all_tile_ids: &[u32]) -> Self {
        let total = width * height * depth;
        let cells = (0..total)
            .map(|_| WFCCell::new(all_tile_ids))
            .collect();

        Self {
            width,
            height,
            depth,
            cells,
        }
    }

    /// Convert (x, y, z) to a flat index.
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        z * (self.width * self.height) + y * self.width + x
    }

    /// Check if coordinates are within bounds.
    fn in_bounds(&self, x: i32, y: i32, z: i32) -> bool {
        x >= 0
            && y >= 0
            && z >= 0
            && (x as usize) < self.width
            && (y as usize) < self.height
            && (z as usize) < self.depth
    }

    /// Get the cell at (x, y, z).
    fn cell(&self, x: usize, y: usize, z: usize) -> &WFCCell {
        &self.cells[self.index(x, y, z)]
    }

    /// Get a mutable reference to the cell at (x, y, z).
    fn cell_mut(&mut self, x: usize, y: usize, z: usize) -> &mut WFCCell {
        let idx = self.index(x, y, z);
        &mut self.cells[idx]
    }

    /// Get the neighbors of a cell, returned as (x, y, z, direction).
    fn neighbors_2d(&self, x: usize, y: usize) -> Vec<(usize, usize, usize, Direction)> {
        let mut result = Vec::new();
        for dir in &Direction::DIRS_2D {
            let (dx, dy, _) = dir.offset();
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if self.in_bounds(nx, ny, 0) {
                result.push((nx as usize, ny as usize, 0, *dir));
            }
        }
        result
    }

    /// Get the neighbors of a cell in 3D.
    fn neighbors_3d(&self, x: usize, y: usize, z: usize) -> Vec<(usize, usize, usize, Direction)> {
        let mut result = Vec::new();
        for dir in &Direction::DIRS_3D {
            let (dx, dy, dz) = dir.offset();
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            let nz = z as i32 + dz;
            if self.in_bounds(nx, ny, nz) {
                result.push((nx as usize, ny as usize, nz as usize, *dir));
            }
        }
        result
    }

    /// Check if the entire grid is fully collapsed.
    fn is_fully_collapsed(&self) -> bool {
        self.cells.iter().all(|c| c.collapsed)
    }

    /// Find the uncollapsed cell with minimum entropy. Returns (x, y, z).
    /// Adds a small random noise to break ties.
    fn find_min_entropy_cell(
        &self,
        tile_weights: &HashMap<u32, f32>,
        rng: &mut Rng,
    ) -> Option<(usize, usize, usize)> {
        let mut min_entropy = f64::MAX;
        let mut best = None;

        for z in 0..self.depth {
            for y in 0..self.height {
                for x in 0..self.width {
                    let cell = self.cell(x, y, z);
                    if cell.collapsed || cell.possible.is_empty() {
                        continue;
                    }
                    if cell.possible.len() == 1 {
                        // Single possibility — entropy is zero, collapse immediately.
                        return Some((x, y, z));
                    }
                    let entropy = cell.entropy(tile_weights);
                    // Add noise to break ties randomly.
                    let noisy = entropy - rng.next_f64() * 0.0001;
                    if noisy < min_entropy {
                        min_entropy = noisy;
                        best = Some((x, y, z));
                    }
                }
            }
        }

        best
    }

    /// Get the tile ID at a collapsed cell.
    pub fn tile_at(&self, x: usize, y: usize, z: usize) -> Option<u32> {
        self.cell(x, y, z).chosen_tile
    }

    /// Get all possible tiles at a cell (before or after collapse).
    pub fn possibilities_at(&self, x: usize, y: usize, z: usize) -> &[u32] {
        &self.cell(x, y, z).possible
    }

    /// Check if a specific cell is collapsed.
    pub fn is_collapsed(&self, x: usize, y: usize, z: usize) -> bool {
        self.cell(x, y, z).collapsed
    }

    /// Total number of cells in the grid.
    pub fn total_cells(&self) -> usize {
        self.width * self.height * self.depth
    }

    /// Number of collapsed cells.
    pub fn collapsed_count(&self) -> usize {
        self.cells.iter().filter(|c| c.collapsed).count()
    }
}

// ===========================================================================
// Snapshot for backtracking
// ===========================================================================

/// A snapshot of the grid state at a specific point, used for backtracking.
#[derive(Debug, Clone)]
struct GridSnapshot {
    /// The cell states at the time of the snapshot.
    cells: Vec<WFCCell>,
    /// The (x, y, z) of the cell that was collapsed in this step.
    collapsed_cell: (usize, usize, usize),
    /// The tile that was chosen when collapsing.
    chosen_tile: u32,
    /// Tiles that have already been tried and failed at this cell.
    tried_tiles: Vec<u32>,
}

// ===========================================================================
// WFCResult
// ===========================================================================

/// The result of a WFC solve attempt.
#[derive(Debug, Clone)]
pub struct WFCResult {
    /// The solved grid (may be partially solved if an error occurred).
    pub grid: WFCGrid,

    /// Whether the grid was fully solved without contradiction.
    pub success: bool,

    /// Number of backtracking steps performed.
    pub backtracks: usize,

    /// Number of propagation steps performed.
    pub propagation_steps: usize,

    /// Total number of cells collapsed.
    pub cells_collapsed: usize,
}

impl WFCResult {
    /// Get the tile at (x, y) for 2D grids. Returns `None` if not collapsed.
    pub fn tile_at_2d(&self, x: usize, y: usize) -> Option<u32> {
        self.grid.tile_at(x, y, 0)
    }

    /// Get the tile at (x, y, z) for 3D grids.
    pub fn tile_at_3d(&self, x: usize, y: usize, z: usize) -> Option<u32> {
        self.grid.tile_at(x, y, z)
    }

    /// Convert the result to a 2D array of tile IDs.
    pub fn to_2d_array(&self) -> Vec<Vec<Option<u32>>> {
        let mut result = Vec::with_capacity(self.grid.height);
        for y in 0..self.grid.height {
            let mut row = Vec::with_capacity(self.grid.width);
            for x in 0..self.grid.width {
                row.push(self.grid.tile_at(x, y, 0));
            }
            result.push(row);
        }
        result
    }
}

// ===========================================================================
// WFCSolver
// ===========================================================================

/// Configuration for the WFC solver.
#[derive(Debug, Clone)]
pub struct WFCSolverConfig {
    /// Maximum number of backtracking attempts before giving up.
    pub max_backtracks: usize,
    /// Whether to use 3D adjacency (6 neighbors) or 2D (4 neighbors).
    pub is_3d: bool,
    /// Seed for the PRNG.
    pub seed: u64,
}

impl Default for WFCSolverConfig {
    fn default() -> Self {
        Self {
            max_backtracks: 1000,
            is_3d: false,
            seed: 42,
        }
    }
}

/// The Wave Function Collapse solver.
///
/// Manages the tile set, constraints, and grid, and runs the collapse-propagate
/// loop with optional backtracking.
pub struct WFCSolver {
    /// The set of available tiles.
    tiles: Vec<WFCTile>,
    /// Adjacency constraints.
    constraints: WFCConstraints,
    /// Map from tile ID to weight for entropy calculation.
    tile_weights: HashMap<u32, f32>,
    /// Map from tile ID to tile data for fast lookup.
    tile_map: HashMap<u32, usize>,
    /// Solver configuration.
    config: WFCSolverConfig,
    /// Pre-computed adjacency: for each tile, for each direction, which tiles
    /// are compatible neighbors.
    adjacency_cache: HashMap<(u32, Direction), Vec<u32>>,
}

impl WFCSolver {
    /// Create a new WFC solver with the given tiles, constraints, and config.
    pub fn new(
        tiles: Vec<WFCTile>,
        constraints: WFCConstraints,
        config: WFCSolverConfig,
    ) -> Self {
        let tile_weights: HashMap<u32, f32> =
            tiles.iter().map(|t| (t.id, t.weight)).collect();
        let tile_map: HashMap<u32, usize> =
            tiles.iter().enumerate().map(|(i, t)| (t.id, i)).collect();

        let mut solver = Self {
            tiles,
            constraints,
            tile_weights,
            tile_map,
            config,
            adjacency_cache: HashMap::new(),
        };
        solver.build_adjacency_cache();
        solver
    }

    /// Pre-compute which tiles can be adjacent to each other tile in each direction.
    fn build_adjacency_cache(&mut self) {
        let dirs = if self.config.is_3d {
            &Direction::DIRS_3D[..]
        } else {
            &Direction::DIRS_2D[..]
        };

        for tile in &self.tiles {
            for &dir in dirs {
                let socket = tile.socket(dir);
                let opp = dir.opposite();
                let mut compatible = Vec::new();

                for other in &self.tiles {
                    let other_socket = other.socket(opp);
                    if self.constraints.are_compatible(socket, other_socket, dir) {
                        compatible.push(other.id);
                    }
                }

                self.adjacency_cache.insert((tile.id, dir), compatible);
            }
        }
    }

    /// Get the set of tiles that are compatible with `tile_id` in direction `dir`.
    fn compatible_tiles(&self, tile_id: u32, dir: Direction) -> &[u32] {
        self.adjacency_cache
            .get(&(tile_id, dir))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get tiles compatible with ANY of the given tile IDs in a direction.
    fn compatible_with_any(&self, tile_ids: &[u32], dir: Direction) -> HashSet<u32> {
        let mut result = HashSet::new();
        for &tid in tile_ids {
            for &compatible in self.compatible_tiles(tid, dir) {
                result.insert(compatible);
            }
        }
        result
    }

    /// Run the full WFC solve loop on a grid.
    ///
    /// Returns a `WFCResult` containing the solved grid and statistics.
    pub fn collapse(&self, grid: &mut WFCGrid) -> WFCStdResult<WFCResult> {
        if self.tiles.is_empty() {
            return Err(WFCError::EmptyTileSet);
        }
        if grid.width == 0 || grid.height == 0 || grid.depth == 0 {
            return Err(WFCError::InvalidDimensions {
                width: grid.width,
                height: grid.height,
                depth: grid.depth,
            });
        }

        let mut rng = Rng::new(self.config.seed);
        let mut backtracks = 0usize;
        let mut propagation_steps = 0usize;
        let mut cells_collapsed = 0usize;
        let mut history: Vec<GridSnapshot> = Vec::new();

        // Initial constraint propagation to remove obviously impossible tiles.
        self.initial_propagation(grid);

        loop {
            // Find the cell with minimum entropy.
            let min_cell = grid.find_min_entropy_cell(&self.tile_weights, &mut rng);

            let (cx, cy, cz) = match min_cell {
                Some(pos) => pos,
                None => {
                    // No uncollapsed cells remain — we're done.
                    if grid.is_fully_collapsed() {
                        break;
                    }
                    // There might be contradicted cells.
                    break;
                }
            };

            // Save snapshot for backtracking.
            let snapshot = GridSnapshot {
                cells: grid.cells.clone(),
                collapsed_cell: (cx, cy, cz),
                chosen_tile: 0, // Will be set below.
                tried_tiles: Vec::new(),
            };

            // Choose a tile for this cell, weighted by probability.
            let cell = grid.cell(cx, cy, cz);
            let possible = cell.possible.clone();
            let weights: Vec<f32> = possible
                .iter()
                .map(|&tid| *self.tile_weights.get(&tid).unwrap_or(&1.0))
                .collect();
            let chosen_idx = rng.weighted_pick(&weights);
            let chosen_tile = possible[chosen_idx];

            // Record the snapshot with the chosen tile.
            let mut snap = snapshot;
            snap.chosen_tile = chosen_tile;
            history.push(snap);

            // Collapse the cell.
            grid.cell_mut(cx, cy, cz).collapse_to(chosen_tile);
            cells_collapsed += 1;

            // Propagate constraints from this cell.
            let prop_result = self.propagate(grid, cx, cy, cz);
            propagation_steps += prop_result.steps;

            if prop_result.contradiction {
                // Contradiction detected — try to backtrack.
                let bt_result = self.backtrack(
                    grid,
                    &mut history,
                    &mut rng,
                    &mut backtracks,
                    &mut propagation_steps,
                    &mut cells_collapsed,
                );

                if !bt_result {
                    // Cannot backtrack further — return partial result.
                    return Ok(WFCResult {
                        grid: grid.clone(),
                        success: false,
                        backtracks,
                        propagation_steps,
                        cells_collapsed,
                    });
                }

                if backtracks > self.config.max_backtracks {
                    return Err(WFCError::MaxBacktracksExceeded {
                        max: self.config.max_backtracks,
                    });
                }
            }
        }

        Ok(WFCResult {
            grid: grid.clone(),
            success: grid.is_fully_collapsed(),
            backtracks,
            propagation_steps,
            cells_collapsed,
        })
    }

    /// Perform initial propagation pass to remove tiles that cannot possibly
    /// be placed anywhere based on the available neighbors.
    fn initial_propagation(&self, grid: &mut WFCGrid) {
        // Run a full pass: for each cell, for each direction, constrain based
        // on what tiles could possibly appear at the neighbor.
        let mut changed = true;
        let mut iteration = 0;
        while changed && iteration < 100 {
            changed = false;
            iteration += 1;

            for z in 0..grid.depth {
                for y in 0..grid.height {
                    for x in 0..grid.width {
                        let neighbors = if self.config.is_3d {
                            grid.neighbors_3d(x, y, z)
                        } else {
                            grid.neighbors_2d(x, y)
                        };

                        for &(nx, ny, nz, dir) in &neighbors {
                            let neighbor_possible = grid.cell(nx, ny, nz).possible.clone();
                            let allowed =
                                self.compatible_with_any(&neighbor_possible, dir.opposite());

                            let cell = grid.cell(x, y, z);
                            let mut to_remove = Vec::new();
                            for &tid in &cell.possible {
                                if !allowed.contains(&tid) {
                                    to_remove.push(tid);
                                }
                            }

                            for tid in to_remove {
                                if grid.cell_mut(x, y, z).remove_tile(tid) {
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Observe and propagate a single step: collapse one cell and propagate.
    ///
    /// Returns `true` if a cell was collapsed, `false` if the grid is fully
    /// collapsed or an error occurred.
    pub fn observe_and_propagate(
        &self,
        grid: &mut WFCGrid,
        rng: &mut Rng,
    ) -> WFCStdResult<bool> {
        if grid.is_fully_collapsed() {
            return Ok(false);
        }

        let min_cell = grid.find_min_entropy_cell(&self.tile_weights, rng);
        let (cx, cy, cz) = match min_cell {
            Some(pos) => pos,
            None => return Ok(false),
        };

        let cell = grid.cell(cx, cy, cz);
        if cell.possible.is_empty() {
            return Err(WFCError::Contradiction {
                x: cx,
                y: cy,
                z: cz,
            });
        }

        let possible = cell.possible.clone();
        let weights: Vec<f32> = possible
            .iter()
            .map(|&tid| *self.tile_weights.get(&tid).unwrap_or(&1.0))
            .collect();
        let chosen_idx = rng.weighted_pick(&weights);
        let chosen_tile = possible[chosen_idx];

        grid.cell_mut(cx, cy, cz).collapse_to(chosen_tile);
        let _prop = self.propagate(grid, cx, cy, cz);

        Ok(true)
    }

    /// Propagate constraints from a recently collapsed cell using BFS.
    ///
    /// For each neighbor of the collapsed cell, remove tiles that are
    /// incompatible with the collapsed tile. If a neighbor's possibilities
    /// change, add its neighbors to the propagation queue.
    fn propagate(&self, grid: &mut WFCGrid, start_x: usize, start_y: usize, start_z: usize) -> PropagationResult {
        let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();
        queue.push_back((start_x, start_y, start_z));

        let mut steps = 0usize;
        let mut visited: HashSet<(usize, usize, usize)> = HashSet::new();

        while let Some((x, y, z)) = queue.pop_front() {
            if !visited.insert((x, y, z)) {
                continue;
            }
            steps += 1;

            let neighbors = if self.config.is_3d {
                grid.neighbors_3d(x, y, z)
            } else {
                grid.neighbors_2d(x, y)
            };

            let current_possible = grid.cell(x, y, z).possible.clone();

            for &(nx, ny, nz, dir) in &neighbors {
                let neighbor = grid.cell(nx, ny, nz);
                if neighbor.collapsed {
                    continue;
                }

                // Compute which tiles are allowed at the neighbor based on
                // what is possible at the current cell.
                let allowed = self.compatible_with_any(&current_possible, dir);

                let neighbor_possible = grid.cell(nx, ny, nz).possible.clone();
                let mut changed = false;

                for &ntid in &neighbor_possible {
                    if !allowed.contains(&ntid) {
                        grid.cell_mut(nx, ny, nz).remove_tile(ntid);
                        changed = true;
                    }
                }

                // Check for contradiction.
                if grid.cell(nx, ny, nz).is_contradiction() {
                    return PropagationResult {
                        steps,
                        contradiction: true,
                    };
                }

                // If the neighbor was single-possibility, auto-collapse.
                if grid.cell(nx, ny, nz).num_possible() == 1 && !grid.cell(nx, ny, nz).collapsed {
                    let tile = grid.cell(nx, ny, nz).possible[0];
                    grid.cell_mut(nx, ny, nz).collapse_to(tile);
                }

                if changed {
                    // Re-add to queue for further propagation.
                    visited.remove(&(nx, ny, nz));
                    queue.push_back((nx, ny, nz));
                }
            }
        }

        PropagationResult {
            steps,
            contradiction: false,
        }
    }

    /// Attempt to backtrack by restoring the previous grid state and trying
    /// a different tile.
    fn backtrack(
        &self,
        grid: &mut WFCGrid,
        history: &mut Vec<GridSnapshot>,
        rng: &mut Rng,
        backtracks: &mut usize,
        propagation_steps: &mut usize,
        cells_collapsed: &mut usize,
    ) -> bool {
        while let Some(mut snapshot) = history.pop() {
            *backtracks += 1;

            // Mark the previously chosen tile as tried.
            snapshot.tried_tiles.push(snapshot.chosen_tile);

            // Restore grid state.
            grid.cells = snapshot.cells.clone();

            let (cx, cy, cz) = snapshot.collapsed_cell;
            let cell = grid.cell(cx, cy, cz);

            // Find remaining untried tiles.
            let untried: Vec<u32> = cell
                .possible
                .iter()
                .filter(|&&tid| !snapshot.tried_tiles.contains(&tid))
                .copied()
                .collect();

            if untried.is_empty() {
                // All tiles tried at this cell — backtrack further.
                continue;
            }

            // Choose a different tile.
            let weights: Vec<f32> = untried
                .iter()
                .map(|&tid| *self.tile_weights.get(&tid).unwrap_or(&1.0))
                .collect();
            let chosen_idx = rng.weighted_pick(&weights);
            let chosen_tile = untried[chosen_idx];

            snapshot.chosen_tile = chosen_tile;
            history.push(snapshot);

            // Collapse to the new choice.
            grid.cell_mut(cx, cy, cz).collapse_to(chosen_tile);
            *cells_collapsed += 1;

            // Propagate.
            let prop = self.propagate(grid, cx, cy, cz);
            *propagation_steps += prop.steps;

            if !prop.contradiction {
                return true; // Successfully backtracked.
            }
            // Contradiction again — loop continues to backtrack further.
        }

        false // History exhausted.
    }

    /// Get the tile data for a given tile ID.
    pub fn tile(&self, id: u32) -> Option<&WFCTile> {
        self.tile_map.get(&id).map(|&idx| &self.tiles[idx])
    }

    /// Get all tiles.
    pub fn tiles(&self) -> &[WFCTile] {
        &self.tiles
    }
}

/// Internal result of a propagation pass.
#[derive(Debug)]
struct PropagationResult {
    steps: usize,
    contradiction: bool,
}

// ===========================================================================
// Pre-built tile sets
// ===========================================================================

/// Pre-built tile sets for common use cases.
pub struct WFCPresets;

impl WFCPresets {
    /// Create a simple maze tile set for 2D.
    ///
    /// Tiles: empty, wall, corridor-horizontal, corridor-vertical,
    /// corner pieces, T-junctions, crossroads.
    pub fn simple_maze() -> (Vec<WFCTile>, WFCConstraints) {
        let mut tiles = Vec::new();
        let mut constraints = WFCConstraints::new();

        // Socket types: "w" = wall, "o" = open
        // Empty (all walls)
        tiles.push(WFCTile::new_2d(0, "empty", "w", "w", "w", "w").with_weight(2.0));
        // Horizontal corridor
        tiles.push(WFCTile::new_2d(1, "corridor_h", "o", "o", "w", "w").with_weight(1.0));
        // Vertical corridor
        tiles.push(WFCTile::new_2d(2, "corridor_v", "w", "w", "o", "o").with_weight(1.0));
        // Corner: open on +X and +Y
        tiles.push(WFCTile::new_2d(3, "corner_ne", "o", "w", "o", "w").with_weight(0.8));
        // Corner: open on -X and +Y
        tiles.push(WFCTile::new_2d(4, "corner_nw", "w", "o", "o", "w").with_weight(0.8));
        // Corner: open on +X and -Y
        tiles.push(WFCTile::new_2d(5, "corner_se", "o", "w", "w", "o").with_weight(0.8));
        // Corner: open on -X and -Y
        tiles.push(WFCTile::new_2d(6, "corner_sw", "w", "o", "w", "o").with_weight(0.8));
        // T-junction: open on +X, -X, +Y
        tiles.push(WFCTile::new_2d(7, "t_north", "o", "o", "o", "w").with_weight(0.5));
        // T-junction: open on +X, -X, -Y
        tiles.push(WFCTile::new_2d(8, "t_south", "o", "o", "w", "o").with_weight(0.5));
        // T-junction: open on +X, +Y, -Y
        tiles.push(WFCTile::new_2d(9, "t_east", "o", "w", "o", "o").with_weight(0.5));
        // T-junction: open on -X, +Y, -Y
        tiles.push(WFCTile::new_2d(10, "t_west", "w", "o", "o", "o").with_weight(0.5));
        // Crossroads: all open
        tiles.push(WFCTile::new_2d(11, "cross", "o", "o", "o", "o").with_weight(0.3));

        constraints.derive_from_tiles(&tiles);

        (tiles, constraints)
    }

    /// Create a dungeon room tile set for 2D.
    ///
    /// Uses socket types: "w" (wall), "d" (door), "f" (floor/open).
    pub fn dungeon_rooms() -> (Vec<WFCTile>, WFCConstraints) {
        let mut tiles = Vec::new();
        let mut constraints = WFCConstraints::new();

        // Add compatibility: doors connect to doors, walls to walls.
        constraints.add_compatible("d", "d");
        constraints.add_compatible("w", "w");
        constraints.add_compatible("f", "f");

        // Solid wall tile.
        tiles.push(WFCTile::new_2d(0, "wall", "w", "w", "w", "w").with_weight(3.0));
        // Room floor.
        tiles.push(WFCTile::new_2d(1, "floor", "f", "f", "f", "f").with_weight(2.0));
        // Wall-to-door transitions.
        tiles.push(WFCTile::new_2d(2, "door_e", "d", "f", "w", "w").with_weight(0.5));
        tiles.push(WFCTile::new_2d(3, "door_w", "f", "d", "w", "w").with_weight(0.5));
        tiles.push(WFCTile::new_2d(4, "door_n", "w", "w", "d", "f").with_weight(0.5));
        tiles.push(WFCTile::new_2d(5, "door_s", "w", "w", "f", "d").with_weight(0.5));
        // Wall edge tiles (wall on one side, floor on others).
        tiles.push(WFCTile::new_2d(6, "wall_e", "w", "f", "f", "f").with_weight(1.0));
        tiles.push(WFCTile::new_2d(7, "wall_w", "f", "w", "f", "f").with_weight(1.0));
        tiles.push(WFCTile::new_2d(8, "wall_n", "f", "f", "w", "f").with_weight(1.0));
        tiles.push(WFCTile::new_2d(9, "wall_s", "f", "f", "f", "w").with_weight(1.0));
        // Corner walls.
        tiles.push(WFCTile::new_2d(10, "corner_ne", "w", "f", "w", "f").with_weight(0.8));
        tiles.push(WFCTile::new_2d(11, "corner_nw", "f", "w", "w", "f").with_weight(0.8));
        tiles.push(WFCTile::new_2d(12, "corner_se", "w", "f", "f", "w").with_weight(0.8));
        tiles.push(WFCTile::new_2d(13, "corner_sw", "f", "w", "f", "w").with_weight(0.8));

        (tiles, constraints)
    }

    /// Create a simple terrain heightmap tile set.
    ///
    /// Socket types encode height levels: "0", "1", "2", "3" for increasing
    /// elevation. Adjacent tiles must have the same edge height.
    pub fn terrain_heightmap() -> (Vec<WFCTile>, WFCConstraints) {
        let mut tiles = Vec::new();
        let mut constraints = WFCConstraints::new();

        // Heights can only differ by at most 1 level on adjacent edges.
        constraints.add_compatible("0", "0");
        constraints.add_compatible("0", "1");
        constraints.add_compatible("1", "0");
        constraints.add_compatible("1", "1");
        constraints.add_compatible("1", "2");
        constraints.add_compatible("2", "1");
        constraints.add_compatible("2", "2");
        constraints.add_compatible("2", "3");
        constraints.add_compatible("3", "2");
        constraints.add_compatible("3", "3");
        constraints.auto_match_same = false;

        let mut id = 0u32;

        // Flat tiles at each level.
        for level in 0..4 {
            let s = level.to_string();
            tiles.push(
                WFCTile::new_2d(id, &format!("flat_{level}"), &s, &s, &s, &s)
                    .with_weight(2.0),
            );
            id += 1;
        }

        // Slope tiles (one side higher than the other).
        for low in 0..3 {
            let high = low + 1;
            let sl = low.to_string();
            let sh = high.to_string();

            // Slope going east (low west, high east).
            tiles.push(
                WFCTile::new_2d(id, &format!("slope_e_{low}_{high}"), &sh, &sl, &sl, &sl)
                    .with_weight(1.0),
            );
            id += 1;

            // Slope going north.
            tiles.push(
                WFCTile::new_2d(id, &format!("slope_n_{low}_{high}"), &sl, &sl, &sh, &sl)
                    .with_weight(1.0),
            );
            id += 1;

            // Slope going west.
            tiles.push(
                WFCTile::new_2d(id, &format!("slope_w_{low}_{high}"), &sl, &sh, &sl, &sl)
                    .with_weight(1.0),
            );
            id += 1;

            // Slope going south.
            tiles.push(
                WFCTile::new_2d(id, &format!("slope_s_{low}_{high}"), &sl, &sl, &sl, &sh)
                    .with_weight(1.0),
            );
            id += 1;
        }

        (tiles, constraints)
    }

    /// Create a simple 3D tile set for building interiors.
    ///
    /// Uses sockets: "w" (wall), "o" (open), "f" (floor), "c" (ceiling).
    pub fn building_3d() -> (Vec<WFCTile>, WFCConstraints) {
        let mut tiles = Vec::new();
        let mut constraints = WFCConstraints::new();

        constraints.add_compatible("w", "w");
        constraints.add_compatible("o", "o");
        constraints.add_compatible("f", "c"); // Floor on top connects to ceiling below.
        constraints.add_compatible("c", "f");

        // Solid block.
        tiles.push(
            WFCTile::new_3d(0, "solid", "w", "w", "w", "w", "w", "w").with_weight(3.0),
        );
        // Room interior.
        tiles.push(
            WFCTile::new_3d(1, "room", "o", "o", "c", "f", "o", "o").with_weight(1.5),
        );
        // Wall segment (wall on +X, open on other horizontal faces).
        tiles.push(
            WFCTile::new_3d(2, "wall_e", "w", "o", "c", "f", "o", "o").with_weight(1.0),
        );
        tiles.push(
            WFCTile::new_3d(3, "wall_w", "o", "w", "c", "f", "o", "o").with_weight(1.0),
        );
        tiles.push(
            WFCTile::new_3d(4, "wall_s", "o", "o", "c", "f", "w", "o").with_weight(1.0),
        );
        tiles.push(
            WFCTile::new_3d(5, "wall_n", "o", "o", "c", "f", "o", "w").with_weight(1.0),
        );
        // Corner wall (two adjacent walls).
        tiles.push(
            WFCTile::new_3d(6, "corner_ne", "w", "o", "c", "f", "o", "w").with_weight(0.8),
        );
        tiles.push(
            WFCTile::new_3d(7, "corner_nw", "o", "w", "c", "f", "o", "w").with_weight(0.8),
        );
        tiles.push(
            WFCTile::new_3d(8, "corner_se", "w", "o", "c", "f", "w", "o").with_weight(0.8),
        );
        tiles.push(
            WFCTile::new_3d(9, "corner_sw", "o", "w", "c", "f", "w", "o").with_weight(0.8),
        );
        // Doorway (open passage through a wall).
        tiles.push(
            WFCTile::new_3d(10, "door_ew", "o", "o", "c", "f", "w", "w").with_weight(0.5),
        );
        tiles.push(
            WFCTile::new_3d(11, "door_ns", "w", "w", "c", "f", "o", "o").with_weight(0.5),
        );

        (tiles, constraints)
    }
}

// ===========================================================================
// Convenience constructors
// ===========================================================================

impl WFCSolver {
    /// Create a solver and grid for 2D WFC with a pre-built tile set.
    pub fn from_preset_2d(
        tiles: Vec<WFCTile>,
        constraints: WFCConstraints,
        width: usize,
        height: usize,
        seed: u64,
    ) -> (Self, WFCGrid) {
        let tile_ids: Vec<u32> = tiles.iter().map(|t| t.id).collect();
        let config = WFCSolverConfig {
            max_backtracks: 1000,
            is_3d: false,
            seed,
        };
        let solver = WFCSolver::new(tiles, constraints, config);
        let grid = WFCGrid::new_2d(width, height, &tile_ids);
        (solver, grid)
    }

    /// Create a solver and grid for 3D WFC with a pre-built tile set.
    pub fn from_preset_3d(
        tiles: Vec<WFCTile>,
        constraints: WFCConstraints,
        width: usize,
        height: usize,
        depth: usize,
        seed: u64,
    ) -> (Self, WFCGrid) {
        let tile_ids: Vec<u32> = tiles.iter().map(|t| t.id).collect();
        let config = WFCSolverConfig {
            max_backtracks: 2000,
            is_3d: true,
            seed,
        };
        let solver = WFCSolver::new(tiles, constraints, config);
        let grid = WFCGrid::new_3d(width, height, depth, &tile_ids);
        (solver, grid)
    }

    /// Solve a 2D maze grid using the simple maze preset.
    pub fn solve_maze_2d(width: usize, height: usize, seed: u64) -> WFCStdResult<WFCResult> {
        let (tiles, constraints) = WFCPresets::simple_maze();
        let (solver, mut grid) = WFCSolver::from_preset_2d(tiles, constraints, width, height, seed);
        solver.collapse(&mut grid)
    }

    /// Solve a 2D dungeon grid using the dungeon rooms preset.
    pub fn solve_dungeon_2d(width: usize, height: usize, seed: u64) -> WFCStdResult<WFCResult> {
        let (tiles, constraints) = WFCPresets::dungeon_rooms();
        let (solver, mut grid) = WFCSolver::from_preset_2d(tiles, constraints, width, height, seed);
        solver.collapse(&mut grid)
    }

    /// Solve a terrain heightmap grid using the terrain preset.
    pub fn solve_terrain_2d(width: usize, height: usize, seed: u64) -> WFCStdResult<WFCResult> {
        let (tiles, constraints) = WFCPresets::terrain_heightmap();
        let (solver, mut grid) = WFCSolver::from_preset_2d(tiles, constraints, width, height, seed);
        solver.collapse(&mut grid)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direction_opposite() {
        assert_eq!(Direction::PosX.opposite(), Direction::NegX);
        assert_eq!(Direction::NegY.opposite(), Direction::PosY);
        assert_eq!(Direction::PosZ.opposite(), Direction::NegZ);
    }

    #[test]
    fn test_tile_creation() {
        let tile = WFCTile::new_2d(1, "test", "a", "b", "c", "d");
        assert_eq!(tile.id, 1);
        assert_eq!(tile.socket(Direction::PosX), "a");
        assert_eq!(tile.socket(Direction::NegX), "b");
        assert_eq!(tile.socket(Direction::PosY), "c");
        assert_eq!(tile.socket(Direction::NegY), "d");
    }

    #[test]
    fn test_tile_rotation() {
        let tile = WFCTile::new_2d(1, "test", "a", "b", "c", "d");
        let rotated = tile.rotate_cw_2d(2);
        assert_eq!(rotated.socket(Direction::PosX), "d");
        assert_eq!(rotated.socket(Direction::PosY), "a");
        assert_eq!(rotated.socket(Direction::NegX), "c");
        assert_eq!(rotated.socket(Direction::NegY), "b");
    }

    #[test]
    fn test_constraints_auto_match() {
        let constraints = WFCConstraints::new();
        assert!(constraints.are_compatible("wall", "wall", Direction::PosX));
        assert!(!constraints.are_compatible("wall", "floor", Direction::PosX));
    }

    #[test]
    fn test_constraints_manual() {
        let mut constraints = WFCConstraints::manual();
        constraints.add_compatible("door", "door");
        assert!(constraints.are_compatible("door", "door", Direction::PosX));
        assert!(!constraints.are_compatible("door", "wall", Direction::PosX));
        assert!(!constraints.are_compatible("wall", "wall", Direction::PosX));
    }

    #[test]
    fn test_cell_entropy() {
        let cell = WFCCell::new(&[0, 1, 2]);
        let mut weights = HashMap::new();
        weights.insert(0, 1.0);
        weights.insert(1, 1.0);
        weights.insert(2, 1.0);
        let entropy = cell.entropy(&weights);
        // For 3 equally weighted tiles, entropy = ln(3) ~ 1.099.
        assert!((entropy - 3.0_f64.ln()).abs() < 0.01);
    }

    #[test]
    fn test_grid_creation() {
        let grid = WFCGrid::new_2d(5, 5, &[0, 1, 2]);
        assert_eq!(grid.total_cells(), 25);
        assert_eq!(grid.collapsed_count(), 0);
        assert_eq!(grid.possibilities_at(0, 0, 0).len(), 3);
    }

    #[test]
    fn test_grid_neighbors_2d() {
        let grid = WFCGrid::new_2d(3, 3, &[0]);
        // Center cell should have 4 neighbors.
        let neighbors = grid.neighbors_2d(1, 1);
        assert_eq!(neighbors.len(), 4);
        // Corner cell should have 2 neighbors.
        let corner_neighbors = grid.neighbors_2d(0, 0);
        assert_eq!(corner_neighbors.len(), 2);
    }

    #[test]
    fn test_grid_3d() {
        let grid = WFCGrid::new_3d(3, 3, 3, &[0, 1]);
        assert_eq!(grid.total_cells(), 27);
        let neighbors = grid.neighbors_3d(1, 1, 1);
        assert_eq!(neighbors.len(), 6);
    }

    #[test]
    fn test_solve_simple_maze() {
        let result = WFCSolver::solve_maze_2d(4, 4, 42);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
        // All cells should be collapsed.
        assert_eq!(result.grid.collapsed_count(), 16);
    }

    #[test]
    fn test_solve_deterministic() {
        let r1 = WFCSolver::solve_maze_2d(5, 5, 42).unwrap();
        let r2 = WFCSolver::solve_maze_2d(5, 5, 42).unwrap();
        // Same seed should produce same result.
        for y in 0..5 {
            for x in 0..5 {
                assert_eq!(r1.tile_at_2d(x, y), r2.tile_at_2d(x, y));
            }
        }
    }

    #[test]
    fn test_solve_different_seeds() {
        let r1 = WFCSolver::solve_maze_2d(6, 6, 1).unwrap();
        let r2 = WFCSolver::solve_maze_2d(6, 6, 2).unwrap();
        // Different seeds should produce different results (with high probability).
        let mut differ = false;
        for y in 0..6 {
            for x in 0..6 {
                if r1.tile_at_2d(x, y) != r2.tile_at_2d(x, y) {
                    differ = true;
                }
            }
        }
        assert!(differ, "Different seeds should produce different grids");
    }

    #[test]
    fn test_terrain_preset() {
        let (tiles, constraints) = WFCPresets::terrain_heightmap();
        assert!(!tiles.is_empty());
        // Heights 0-0 should be compatible.
        assert!(constraints.are_compatible("0", "0", Direction::PosX));
        // Heights 0-2 should NOT be compatible (differ by 2).
        assert!(!constraints.are_compatible("0", "2", Direction::PosX));
    }

    #[test]
    fn test_wfc_result_to_2d_array() {
        let result = WFCSolver::solve_maze_2d(3, 3, 42).unwrap();
        let array = result.to_2d_array();
        assert_eq!(array.len(), 3);
        assert_eq!(array[0].len(), 3);
        for row in &array {
            for cell in row {
                assert!(cell.is_some());
            }
        }
    }

    #[test]
    fn test_building_3d_preset() {
        let (tiles, constraints) = WFCPresets::building_3d();
        assert!(!tiles.is_empty());
        assert!(constraints.are_compatible("f", "c", Direction::PosY));
        assert!(constraints.are_compatible("c", "f", Direction::PosY));
    }

    #[test]
    fn test_tile_mirror() {
        let tile = WFCTile::new_2d(1, "test", "a", "b", "c", "d");
        let mirrored = tile.mirror_x_2d(2);
        assert_eq!(mirrored.socket(Direction::PosX), "b");
        assert_eq!(mirrored.socket(Direction::NegX), "a");
        assert_eq!(mirrored.socket(Direction::PosY), "c");
        assert_eq!(mirrored.socket(Direction::NegY), "d");
    }

    #[test]
    fn test_observe_and_propagate() {
        let (tiles, constraints) = WFCPresets::simple_maze();
        let tile_ids: Vec<u32> = tiles.iter().map(|t| t.id).collect();
        let config = WFCSolverConfig {
            max_backtracks: 100,
            is_3d: false,
            seed: 42,
        };
        let solver = WFCSolver::new(tiles, constraints, config);
        let mut grid = WFCGrid::new_2d(4, 4, &tile_ids);
        let mut rng = Rng::new(42);

        // Run a single step.
        let result = solver.observe_and_propagate(&mut grid, &mut rng);
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should have collapsed one cell.
        assert!(grid.collapsed_count() >= 1);
    }
}
