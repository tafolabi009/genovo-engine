//! # Large World Coordinates
//!
//! Provides 64-bit double-precision world positioning, floating-origin rebasing,
//! spatial subdivision into cells, streaming volumes, world bounds management,
//! a multi-system LOD framework, and sector-based variable-rate simulation.
//!
//! These primitives enable EVE Online-scale worlds spanning millions of kilometers
//! while maintaining sub-millimeter precision near the camera by converting to
//! f32-relative coordinates for rendering and physics.
//!
//! ## Architecture
//!
//! ```text
//! WorldPosition (f64)
//!       |
//!       v
//! WorldOrigin (floating origin)
//!       |
//!       v
//! to_local() -> Vec3 (f32 relative)  -->  Renderer / Physics / Audio
//!       |
//!       v
//! WorldCell grid  -->  StreamingVolume  -->  SectorManager
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use glam::Vec3;

// ===========================================================================
// Constants
// ===========================================================================

/// Default threshold distance from origin before a rebase is triggered.
pub const DEFAULT_REBASE_THRESHOLD: f64 = 10_000.0;

/// Default world cell size in world units (1 km).
pub const DEFAULT_CELL_SIZE: f64 = 1000.0;

/// Maximum coordinate value per axis (practical world limit).
pub const MAX_COORDINATE: f64 = 2_000_000_000.0; // 2 billion units

/// Minimum coordinate value per axis.
pub const MIN_COORDINATE: f64 = -2_000_000_000.0;

/// Default preload radius multiplier (fraction beyond load radius).
pub const DEFAULT_PRELOAD_MULTIPLIER: f64 = 1.5;

/// LOD hysteresis factor to prevent thrashing (unitless multiplier).
pub const LOD_HYSTERESIS_FACTOR: f64 = 1.1;

/// Active sector simulation rate (Hz).
pub const ACTIVE_SECTOR_RATE: f32 = 60.0;

/// Near sector simulation rate (Hz).
pub const NEAR_SECTOR_RATE: f32 = 10.0;

/// Far sector simulation rate (Hz).
pub const FAR_SECTOR_RATE: f32 = 1.0;

/// Maximum entities tracked per cell for streaming decisions.
pub const MAX_ENTITIES_PER_CELL: usize = 4096;

/// Default LOD distance thresholds (in world units).
pub const LOD_DISTANCE_FULL: f64 = 50.0;
pub const LOD_DISTANCE_REDUCED: f64 = 200.0;
pub const LOD_DISTANCE_MINIMAL: f64 = 800.0;
pub const LOD_DISTANCE_PROXY: f64 = 3000.0;

// ===========================================================================
// WorldPosition — 64-bit double-precision position
// ===========================================================================

/// A position in world space using 64-bit double-precision floating point.
///
/// This provides sub-millimeter precision even at distances of millions of
/// kilometers from the origin. All large-world calculations operate on
/// `WorldPosition` values; conversion to f32 `Vec3` happens only when
/// feeding data to the renderer, physics, or audio systems via [`to_local`].
///
/// [`to_local`]: WorldPosition::to_local
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldPosition {
    /// X coordinate in world space (f64).
    pub x: f64,
    /// Y coordinate in world space (f64).
    pub y: f64,
    /// Z coordinate in world space (f64).
    pub z: f64,
}

impl WorldPosition {
    /// Zero position.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Unit X.
    pub const UNIT_X: Self = Self {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };

    /// Unit Y.
    pub const UNIT_Y: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    /// Unit Z.
    pub const UNIT_Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    /// Create a new world position.
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Create from an f32 Vec3 (promoted to f64).
    #[inline]
    pub fn from_vec3(v: Vec3) -> Self {
        Self {
            x: v.x as f64,
            y: v.y as f64,
            z: v.z as f64,
        }
    }

    /// Convert to a camera-relative f32 Vec3.
    ///
    /// Subtracts `origin` (typically the camera or floating-origin position)
    /// to produce a small offset that fits comfortably in f32.  This is the
    /// primary bridge between the 64-bit world and 32-bit subsystems.
    #[inline]
    pub fn to_local(&self, origin: WorldPosition) -> Vec3 {
        Vec3::new(
            (self.x - origin.x) as f32,
            (self.y - origin.y) as f32,
            (self.z - origin.z) as f32,
        )
    }

    /// Reconstruct a `WorldPosition` from a local Vec3 offset and an origin.
    #[inline]
    pub fn from_local(origin: WorldPosition, local: Vec3) -> Self {
        Self {
            x: origin.x + local.x as f64,
            y: origin.y + local.y as f64,
            z: origin.z + local.z as f64,
        }
    }

    /// Precise distance between two world positions.
    #[inline]
    pub fn distance(a: WorldPosition, b: WorldPosition) -> f64 {
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let dz = a.z - b.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Squared distance (avoids the sqrt for comparisons).
    #[inline]
    pub fn distance_sq(a: WorldPosition, b: WorldPosition) -> f64 {
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let dz = a.z - b.z;
        dx * dx + dy * dy + dz * dz
    }

    /// Manhattan distance.
    #[inline]
    pub fn manhattan_distance(a: WorldPosition, b: WorldPosition) -> f64 {
        (a.x - b.x).abs() + (a.y - b.y).abs() + (a.z - b.z).abs()
    }

    /// Chebyshev (chessboard) distance.
    #[inline]
    pub fn chebyshev_distance(a: WorldPosition, b: WorldPosition) -> f64 {
        let dx = (a.x - b.x).abs();
        let dy = (a.y - b.y).abs();
        let dz = (a.z - b.z).abs();
        dx.max(dy).max(dz)
    }

    /// Linear interpolation between two world positions.
    #[inline]
    pub fn lerp(a: WorldPosition, b: WorldPosition, t: f64) -> Self {
        Self {
            x: a.x + (b.x - a.x) * t,
            y: a.y + (b.y - a.y) * t,
            z: a.z + (b.z - a.z) * t,
        }
    }

    /// Magnitude (length from origin).
    #[inline]
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Squared magnitude.
    #[inline]
    pub fn magnitude_sq(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Normalize to unit length.
    #[inline]
    pub fn normalized(&self) -> Self {
        let m = self.magnitude();
        if m < 1e-15 {
            return Self::ZERO;
        }
        Self {
            x: self.x / m,
            y: self.y / m,
            z: self.z / m,
        }
    }

    /// Dot product.
    #[inline]
    pub fn dot(a: WorldPosition, b: WorldPosition) -> f64 {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    /// Cross product.
    #[inline]
    pub fn cross(a: WorldPosition, b: WorldPosition) -> Self {
        Self {
            x: a.y * b.z - a.z * b.y,
            y: a.z * b.x - a.x * b.z,
            z: a.x * b.y - a.y * b.x,
        }
    }

    /// Component-wise minimum.
    #[inline]
    pub fn min(a: WorldPosition, b: WorldPosition) -> Self {
        Self {
            x: a.x.min(b.x),
            y: a.y.min(b.y),
            z: a.z.min(b.z),
        }
    }

    /// Component-wise maximum.
    #[inline]
    pub fn max(a: WorldPosition, b: WorldPosition) -> Self {
        Self {
            x: a.x.max(b.x),
            y: a.y.max(b.y),
            z: a.z.max(b.z),
        }
    }

    /// Clamp each component independently.
    #[inline]
    pub fn clamp(&self, min_val: f64, max_val: f64) -> Self {
        Self {
            x: self.x.clamp(min_val, max_val),
            y: self.y.clamp(min_val, max_val),
            z: self.z.clamp(min_val, max_val),
        }
    }

    /// Whether this position is within world bounds.
    #[inline]
    pub fn is_within_bounds(&self) -> bool {
        self.x >= MIN_COORDINATE
            && self.x <= MAX_COORDINATE
            && self.y >= MIN_COORDINATE
            && self.y <= MAX_COORDINATE
            && self.z >= MIN_COORDINATE
            && self.z <= MAX_COORDINATE
    }

    /// Scale by a scalar.
    #[inline]
    pub fn scale(&self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// Add another position (used as a vector addition).
    #[inline]
    pub fn add(&self, other: WorldPosition) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    /// Subtract another position (used as vector subtraction).
    #[inline]
    pub fn sub(&self, other: WorldPosition) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    /// Negate.
    #[inline]
    pub fn neg(&self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Convert to the cell coordinate this position falls within.
    #[inline]
    pub fn to_cell_coord(&self, cell_size: f64) -> CellCoord3D {
        CellCoord3D {
            x: (self.x / cell_size).floor() as i32,
            y: (self.y / cell_size).floor() as i32,
            z: (self.z / cell_size).floor() as i32,
        }
    }

    /// Returns true if any component is NaN.
    #[inline]
    pub fn is_nan(&self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }

    /// Returns true if any component is infinite.
    #[inline]
    pub fn is_infinite(&self) -> bool {
        self.x.is_infinite() || self.y.is_infinite() || self.z.is_infinite()
    }

    /// Returns true if all components are finite (not NaN or infinite).
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Angle between two position-vectors in radians.
    #[inline]
    pub fn angle_between(a: WorldPosition, b: WorldPosition) -> f64 {
        let d = Self::dot(a, b);
        let ma = a.magnitude();
        let mb = b.magnitude();
        if ma < 1e-15 || mb < 1e-15 {
            return 0.0;
        }
        (d / (ma * mb)).clamp(-1.0, 1.0).acos()
    }

    /// Project self onto the direction defined by `onto`.
    #[inline]
    pub fn project_onto(&self, onto: WorldPosition) -> Self {
        let d = Self::dot(*self, onto);
        let m = onto.magnitude_sq();
        if m < 1e-15 {
            return Self::ZERO;
        }
        onto.scale(d / m)
    }

    /// Reflect self across a normal.
    #[inline]
    pub fn reflect(&self, normal: WorldPosition) -> Self {
        let d = Self::dot(*self, normal);
        Self {
            x: self.x - 2.0 * d * normal.x,
            y: self.y - 2.0 * d * normal.y,
            z: self.z - 2.0 * d * normal.z,
        }
    }
}

impl Default for WorldPosition {
    fn default() -> Self {
        Self::ZERO
    }
}

impl fmt::Display for WorldPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.6}, {:.6}, {:.6})", self.x, self.y, self.z)
    }
}

impl std::ops::Add for WorldPosition {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Sub for WorldPosition {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Mul<f64> for WorldPosition {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl std::ops::Neg for WorldPosition {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl std::ops::AddAssign for WorldPosition {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl std::ops::SubAssign for WorldPosition {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

// ===========================================================================
// WorldOrigin — Floating origin system
// ===========================================================================

/// Event produced when the floating origin is rebased.
#[derive(Debug, Clone, Copy)]
pub struct RebaseEvent {
    /// The old origin position.
    pub old_origin: WorldPosition,
    /// The new origin position.
    pub new_origin: WorldPosition,
    /// The delta that must be subtracted from all entity positions.
    pub delta: WorldPosition,
    /// Frame number when the rebase occurred.
    pub frame: u64,
}

/// Manages the floating origin to keep the camera near (0,0,0) in f32 space.
///
/// When the camera moves far from the current origin, all entities are shifted
/// by the inverse of the delta so that the camera returns to near-zero. This
/// prevents f32 precision degradation at large coordinates.
#[derive(Debug, Clone)]
pub struct WorldOrigin {
    /// Current origin in absolute world coordinates.
    pub origin: WorldPosition,
    /// Distance threshold from origin that triggers rebase.
    pub rebase_threshold: f64,
    /// Squared threshold (cached to avoid sqrt in checks).
    rebase_threshold_sq: f64,
    /// Number of rebases performed.
    pub rebase_count: u64,
    /// History of rebase events for debugging.
    pub rebase_history: VecDeque<RebaseEvent>,
    /// Maximum history entries to keep.
    pub max_history: usize,
    /// Current frame counter.
    pub frame: u64,
    /// Whether rebasing is enabled.
    pub enabled: bool,
    /// Minimum time between rebases in frames (prevents rapid rebasing).
    pub min_rebase_interval: u64,
    /// Frame of last rebase.
    last_rebase_frame: u64,
}

impl WorldOrigin {
    /// Create a new floating origin system.
    pub fn new(threshold: f64) -> Self {
        Self {
            origin: WorldPosition::ZERO,
            rebase_threshold: threshold,
            rebase_threshold_sq: threshold * threshold,
            rebase_count: 0,
            rebase_history: VecDeque::new(),
            max_history: 64,
            frame: 0,
            enabled: true,
            min_rebase_interval: 10,
            last_rebase_frame: 0,
        }
    }

    /// Create with default threshold.
    pub fn with_default_threshold() -> Self {
        Self::new(DEFAULT_REBASE_THRESHOLD)
    }

    /// Set the rebase threshold.
    pub fn set_threshold(&mut self, threshold: f64) {
        self.rebase_threshold = threshold;
        self.rebase_threshold_sq = threshold * threshold;
    }

    /// Check if the camera position requires a rebase.
    ///
    /// Returns `true` when the camera's distance from the current origin
    /// exceeds the threshold, subject to the minimum rebase interval.
    pub fn should_rebase(&self, camera_pos: WorldPosition) -> bool {
        if !self.enabled {
            return false;
        }
        if self.frame - self.last_rebase_frame < self.min_rebase_interval {
            return false;
        }
        let local = camera_pos.to_local(self.origin);
        let dist_sq = (local.x * local.x + local.y * local.y + local.z * local.z) as f64;
        dist_sq > self.rebase_threshold_sq
    }

    /// Perform a rebase to a new origin.
    ///
    /// Returns a `RebaseEvent` containing the delta that must be applied to
    /// every entity's local position. The caller is responsible for iterating
    /// all entities and subtracting `delta` from their transforms.
    pub fn rebase(&mut self, new_origin: WorldPosition) -> RebaseEvent {
        let old_origin = self.origin;
        let delta = new_origin - old_origin;

        let event = RebaseEvent {
            old_origin,
            new_origin,
            delta,
            frame: self.frame,
        };

        self.origin = new_origin;
        self.rebase_count += 1;
        self.last_rebase_frame = self.frame;

        self.rebase_history.push_back(event);
        while self.rebase_history.len() > self.max_history {
            self.rebase_history.pop_front();
        }

        event
    }

    /// Convenience: check and rebase if necessary, using the camera position
    /// as the new origin. Returns `Some(RebaseEvent)` if a rebase occurred.
    pub fn update(&mut self, camera_pos: WorldPosition) -> Option<RebaseEvent> {
        self.frame += 1;
        if self.should_rebase(camera_pos) {
            Some(self.rebase(camera_pos))
        } else {
            None
        }
    }

    /// Convert an absolute world position to local f32 coordinates.
    #[inline]
    pub fn to_local(&self, world_pos: WorldPosition) -> Vec3 {
        world_pos.to_local(self.origin)
    }

    /// Convert local f32 coordinates back to absolute world position.
    #[inline]
    pub fn to_world(&self, local: Vec3) -> WorldPosition {
        WorldPosition::from_local(self.origin, local)
    }

    /// Get the cumulative shift applied by all rebases since the beginning.
    pub fn cumulative_shift(&self) -> WorldPosition {
        self.origin
    }

    /// Reset the origin to (0,0,0) and clear history.
    pub fn reset(&mut self) {
        self.origin = WorldPosition::ZERO;
        self.rebase_count = 0;
        self.rebase_history.clear();
        self.last_rebase_frame = 0;
    }

    /// Apply a rebase event to a set of world positions, shifting them by the delta.
    pub fn apply_rebase_to_positions(
        event: &RebaseEvent,
        positions: &mut [WorldPosition],
    ) {
        for pos in positions.iter_mut() {
            // Positions in local space need the delta subtracted so they
            // remain at the same absolute world location.
            *pos = *pos - event.delta;
        }
    }

    /// Apply a rebase event to a set of local Vec3 positions.
    pub fn apply_rebase_to_local_positions(event: &RebaseEvent, positions: &mut [Vec3]) {
        let dx = event.delta.x as f32;
        let dy = event.delta.y as f32;
        let dz = event.delta.z as f32;
        for pos in positions.iter_mut() {
            pos.x -= dx;
            pos.y -= dy;
            pos.z -= dz;
        }
    }
}

impl Default for WorldOrigin {
    fn default() -> Self {
        Self::with_default_threshold()
    }
}

// ===========================================================================
// CellCoord3D — 3D cell coordinate
// ===========================================================================

/// A 3D integer cell coordinate for spatial subdivision.
///
/// With i32 components and a default cell size of 1 km, this supports
/// a world spanning ~4.3 billion km on each axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellCoord3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl CellCoord3D {
    /// New cell coordinate.
    #[inline]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// The zero cell.
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };

    /// Chebyshev distance to another cell.
    #[inline]
    pub fn chebyshev_distance(&self, other: &CellCoord3D) -> i32 {
        let dx = (self.x - other.x).abs();
        let dy = (self.y - other.y).abs();
        let dz = (self.z - other.z).abs();
        dx.max(dy).max(dz)
    }

    /// Manhattan distance.
    #[inline]
    pub fn manhattan_distance(&self, other: &CellCoord3D) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs() + (self.z - other.z).abs()
    }

    /// Squared Euclidean distance.
    #[inline]
    pub fn distance_sq(&self, other: &CellCoord3D) -> i64 {
        let dx = (self.x - other.x) as i64;
        let dy = (self.y - other.y) as i64;
        let dz = (self.z - other.z) as i64;
        dx * dx + dy * dy + dz * dz
    }

    /// Euclidean distance as f64.
    #[inline]
    pub fn distance_f64(&self, other: &CellCoord3D) -> f64 {
        (self.distance_sq(other) as f64).sqrt()
    }

    /// World-space center of this cell.
    pub fn center(&self, cell_size: f64) -> WorldPosition {
        WorldPosition {
            x: (self.x as f64 + 0.5) * cell_size,
            y: (self.y as f64 + 0.5) * cell_size,
            z: (self.z as f64 + 0.5) * cell_size,
        }
    }

    /// World-space minimum corner of this cell.
    pub fn min_corner(&self, cell_size: f64) -> WorldPosition {
        WorldPosition {
            x: self.x as f64 * cell_size,
            y: self.y as f64 * cell_size,
            z: self.z as f64 * cell_size,
        }
    }

    /// World-space maximum corner of this cell.
    pub fn max_corner(&self, cell_size: f64) -> WorldPosition {
        WorldPosition {
            x: (self.x + 1) as f64 * cell_size,
            y: (self.y + 1) as f64 * cell_size,
            z: (self.z + 1) as f64 * cell_size,
        }
    }

    /// Iterate over all neighbors within a Chebyshev radius.
    pub fn neighbors(&self, radius: i32) -> Vec<CellCoord3D> {
        let mut result = Vec::new();
        for dx in -radius..=radius {
            for dy in -radius..=radius {
                for dz in -radius..=radius {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    result.push(CellCoord3D {
                        x: self.x + dx,
                        y: self.y + dy,
                        z: self.z + dz,
                    });
                }
            }
        }
        result
    }

    /// Iterate over all cells in the box defined by `min_cell..=max_cell`.
    pub fn cells_in_range(min_cell: CellCoord3D, max_cell: CellCoord3D) -> Vec<CellCoord3D> {
        let mut result = Vec::new();
        for x in min_cell.x..=max_cell.x {
            for y in min_cell.y..=max_cell.y {
                for z in min_cell.z..=max_cell.z {
                    result.push(CellCoord3D { x, y, z });
                }
            }
        }
        result
    }
}

impl fmt::Display for CellCoord3D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}]", self.x, self.y, self.z)
    }
}

// ===========================================================================
// WorldCell — spatial subdivision cell
// ===========================================================================

/// State of a world cell in the streaming pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CellLoadState {
    /// Not loaded, no data in memory.
    Unloaded,
    /// Load request queued.
    Queued,
    /// Actively loading from disk/network.
    Loading,
    /// Loaded and ready.
    Loaded,
    /// Active (visible, simulated).
    Active,
    /// Pending unload (deferred to avoid hitches).
    PendingUnload,
    /// Error state (load failed).
    Error,
}

/// A cell in the world grid.
#[derive(Debug, Clone)]
pub struct WorldCell {
    /// Cell coordinate.
    pub coord: CellCoord3D,
    /// Current load state.
    pub state: CellLoadState,
    /// Number of entities in this cell.
    pub entity_count: u32,
    /// Entity IDs (lightweight handles).
    pub entities: Vec<u64>,
    /// Whether this cell has been modified since last save.
    pub dirty: bool,
    /// Priority for loading (higher = load sooner).
    pub load_priority: f32,
    /// Distance from camera cell (updated each frame).
    pub camera_distance: f64,
    /// Timestamp of last state change.
    pub last_state_change: f64,
    /// Memory footprint estimate in bytes.
    pub memory_bytes: u64,
    /// LOD level currently applied to this cell's content.
    pub lod_level: LODLevel,
    /// Custom user data tag.
    pub tag: u32,
}

impl WorldCell {
    /// Create a new unloaded cell.
    pub fn new(coord: CellCoord3D) -> Self {
        Self {
            coord,
            state: CellLoadState::Unloaded,
            entity_count: 0,
            entities: Vec::new(),
            dirty: false,
            load_priority: 0.0,
            camera_distance: f64::MAX,
            last_state_change: 0.0,
            memory_bytes: 0,
            lod_level: LODLevel::Hidden,
            tag: 0,
        }
    }

    /// Add an entity to this cell.
    pub fn add_entity(&mut self, entity_id: u64) {
        if self.entities.len() < MAX_ENTITIES_PER_CELL {
            self.entities.push(entity_id);
            self.entity_count = self.entities.len() as u32;
            self.dirty = true;
        }
    }

    /// Remove an entity from this cell.
    pub fn remove_entity(&mut self, entity_id: u64) -> bool {
        if let Some(pos) = self.entities.iter().position(|&e| e == entity_id) {
            self.entities.swap_remove(pos);
            self.entity_count = self.entities.len() as u32;
            self.dirty = true;
            true
        } else {
            false
        }
    }

    /// Transition to a new state.
    pub fn set_state(&mut self, new_state: CellLoadState, timestamp: f64) {
        self.state = new_state;
        self.last_state_change = timestamp;
    }

    /// Whether this cell has any entities.
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Whether this cell is loaded or active.
    pub fn is_loaded(&self) -> bool {
        matches!(self.state, CellLoadState::Loaded | CellLoadState::Active)
    }
}

// ===========================================================================
// WorldCellGrid — manages the grid of cells
// ===========================================================================

/// Configuration for the world cell grid.
#[derive(Debug, Clone)]
pub struct WorldCellConfig {
    /// Size of each cell in world units (applies to all axes).
    pub cell_size: f64,
    /// Radius in cells around the camera to keep loaded.
    pub load_radius: i32,
    /// Radius in cells beyond which cells are unloaded (must be > load_radius).
    pub unload_radius: i32,
    /// Maximum cells to load per frame.
    pub max_loads_per_frame: usize,
    /// Maximum cells to unload per frame.
    pub max_unloads_per_frame: usize,
    /// Total memory budget for loaded cells (bytes).
    pub memory_budget: u64,
    /// Whether to use vertical (Y) cell subdivision.
    pub use_vertical_cells: bool,
    /// Vertical load radius (if vertical cells enabled).
    pub vertical_load_radius: i32,
}

impl Default for WorldCellConfig {
    fn default() -> Self {
        Self {
            cell_size: DEFAULT_CELL_SIZE,
            load_radius: 5,
            unload_radius: 7,
            max_loads_per_frame: 2,
            max_unloads_per_frame: 4,
            memory_budget: 512 * 1024 * 1024, // 512 MB
            use_vertical_cells: false,
            vertical_load_radius: 2,
        }
    }
}

/// Manages the spatial grid of world cells, handling load/unload decisions.
#[derive(Debug)]
pub struct WorldCellGrid {
    /// Configuration.
    pub config: WorldCellConfig,
    /// All known cells indexed by coordinate.
    cells: HashMap<CellCoord3D, WorldCell>,
    /// Current camera cell.
    camera_cell: CellCoord3D,
    /// Cells queued for loading.
    load_queue: VecDeque<CellCoord3D>,
    /// Cells queued for unloading.
    unload_queue: VecDeque<CellCoord3D>,
    /// Total memory used by loaded cells.
    total_memory: u64,
    /// Statistics.
    pub stats: WorldCellGridStats,
}

/// Statistics for the cell grid.
#[derive(Debug, Clone, Default)]
pub struct WorldCellGridStats {
    /// Total cells known.
    pub total_cells: usize,
    /// Cells currently loaded.
    pub loaded_cells: usize,
    /// Cells currently active.
    pub active_cells: usize,
    /// Cells loaded this frame.
    pub loads_this_frame: usize,
    /// Cells unloaded this frame.
    pub unloads_this_frame: usize,
    /// Total memory used (bytes).
    pub memory_used: u64,
    /// Memory budget remaining.
    pub memory_remaining: u64,
}

impl WorldCellGrid {
    /// Create a new cell grid.
    pub fn new(config: WorldCellConfig) -> Self {
        Self {
            config,
            cells: HashMap::new(),
            camera_cell: CellCoord3D::ZERO,
            load_queue: VecDeque::new(),
            unload_queue: VecDeque::new(),
            total_memory: 0,
            stats: WorldCellGridStats::default(),
        }
    }

    /// Get a cell by coordinate.
    pub fn get_cell(&self, coord: &CellCoord3D) -> Option<&WorldCell> {
        self.cells.get(coord)
    }

    /// Get a mutable cell by coordinate.
    pub fn get_cell_mut(&mut self, coord: &CellCoord3D) -> Option<&mut WorldCell> {
        self.cells.get_mut(coord)
    }

    /// Get or create a cell at the given coordinate.
    pub fn get_or_create_cell(&mut self, coord: CellCoord3D) -> &mut WorldCell {
        self.cells.entry(coord).or_insert_with(|| WorldCell::new(coord))
    }

    /// Update the grid based on the camera's world position.
    ///
    /// This evaluates which cells should be loaded/unloaded and populates
    /// the internal queues. Call `process_queues` to perform the actual
    /// load/unload operations.
    pub fn update_camera(&mut self, camera_world_pos: WorldPosition, timestamp: f64) {
        let new_camera_cell = camera_world_pos.to_cell_coord(self.config.cell_size);
        self.camera_cell = new_camera_cell;

        // Determine which cells should be loaded.
        let load_r = self.config.load_radius;
        let unload_r = self.config.unload_radius;
        let vert_r = if self.config.use_vertical_cells {
            self.config.vertical_load_radius
        } else {
            0
        };

        let mut desired_loaded: HashSet<CellCoord3D> = HashSet::new();
        for dx in -load_r..=load_r {
            let y_range = if self.config.use_vertical_cells {
                -vert_r..=vert_r
            } else {
                0..=0
            };
            for dy in y_range {
                for dz in -load_r..=load_r {
                    desired_loaded.insert(CellCoord3D {
                        x: new_camera_cell.x + dx,
                        y: new_camera_cell.y + dy,
                        z: new_camera_cell.z + dz,
                    });
                }
            }
        }

        // Queue loads for cells that should be loaded but aren't.
        for &coord in &desired_loaded {
            let needs_load = match self.cells.get(&coord) {
                Some(cell) => cell.state == CellLoadState::Unloaded,
                None => true,
            };
            if needs_load && !self.load_queue.contains(&coord) {
                self.load_queue.push_back(coord);
            }
        }

        // Queue unloads for cells that are loaded but too far away.
        let cells_to_check: Vec<CellCoord3D> = self
            .cells
            .iter()
            .filter(|(_, cell)| cell.is_loaded())
            .map(|(coord, _)| *coord)
            .collect();

        for coord in cells_to_check {
            let dist = coord.chebyshev_distance(&new_camera_cell);
            if dist > unload_r && !self.unload_queue.contains(&coord) {
                self.unload_queue.push_back(coord);
            }
        }

        // Update camera distances for all loaded cells.
        for (coord, cell) in &mut self.cells {
            cell.camera_distance = coord.distance_f64(&new_camera_cell);
        }

        // Sort load queue by priority (closer cells first).
        let cam = new_camera_cell;
        let mut queue_vec: Vec<CellCoord3D> = self.load_queue.drain(..).collect();
        queue_vec.sort_by(|a, b| {
            let da = a.distance_sq(&cam);
            let db = b.distance_sq(&cam);
            da.cmp(&db)
        });
        self.load_queue = VecDeque::from(queue_vec);

        // Update statistics.
        self.update_stats();
    }

    /// Process load/unload queues up to the per-frame limits.
    ///
    /// Returns lists of cells that were loaded and unloaded.
    pub fn process_queues(&mut self, timestamp: f64) -> (Vec<CellCoord3D>, Vec<CellCoord3D>) {
        let mut loaded = Vec::new();
        let mut unloaded = Vec::new();

        // Process loads.
        let max_loads = self.config.max_loads_per_frame;
        for _ in 0..max_loads {
            if let Some(coord) = self.load_queue.pop_front() {
                let cell = self.cells.entry(coord).or_insert_with(|| WorldCell::new(coord));
                if cell.state == CellLoadState::Unloaded || cell.state == CellLoadState::Error {
                    cell.set_state(CellLoadState::Loaded, timestamp);
                    loaded.push(coord);
                }
            } else {
                break;
            }
        }

        // Process unloads.
        let max_unloads = self.config.max_unloads_per_frame;
        for _ in 0..max_unloads {
            if let Some(coord) = self.unload_queue.pop_front() {
                if let Some(cell) = self.cells.get_mut(&coord) {
                    if cell.is_loaded() {
                        self.total_memory = self.total_memory.saturating_sub(cell.memory_bytes);
                        cell.set_state(CellLoadState::Unloaded, timestamp);
                        cell.entities.clear();
                        cell.entity_count = 0;
                        unloaded.push(coord);
                    }
                }
            } else {
                break;
            }
        }

        self.update_stats();
        (loaded, unloaded)
    }

    /// Update internal statistics.
    fn update_stats(&mut self) {
        self.stats.total_cells = self.cells.len();
        self.stats.loaded_cells = self
            .cells
            .values()
            .filter(|c| c.state == CellLoadState::Loaded)
            .count();
        self.stats.active_cells = self
            .cells
            .values()
            .filter(|c| c.state == CellLoadState::Active)
            .count();
        self.stats.memory_used = self.total_memory;
        self.stats.memory_remaining = self.config.memory_budget.saturating_sub(self.total_memory);
    }

    /// Get all loaded cells.
    pub fn loaded_cells(&self) -> Vec<&WorldCell> {
        self.cells.values().filter(|c| c.is_loaded()).collect()
    }

    /// Get all active cells.
    pub fn active_cells(&self) -> Vec<&WorldCell> {
        self.cells
            .values()
            .filter(|c| c.state == CellLoadState::Active)
            .collect()
    }

    /// Get the current camera cell coordinate.
    pub fn camera_cell(&self) -> CellCoord3D {
        self.camera_cell
    }

    /// Count of pending loads in the queue.
    pub fn pending_loads(&self) -> usize {
        self.load_queue.len()
    }

    /// Count of pending unloads in the queue.
    pub fn pending_unloads(&self) -> usize {
        self.unload_queue.len()
    }

    /// Remove a cell entirely from the grid.
    pub fn remove_cell(&mut self, coord: &CellCoord3D) -> Option<WorldCell> {
        if let Some(cell) = self.cells.remove(coord) {
            self.total_memory = self.total_memory.saturating_sub(cell.memory_bytes);
            Some(cell)
        } else {
            None
        }
    }

    /// Clear all cells and queues.
    pub fn clear(&mut self) {
        self.cells.clear();
        self.load_queue.clear();
        self.unload_queue.clear();
        self.total_memory = 0;
        self.update_stats();
    }

    /// Check if memory budget allows loading another cell.
    pub fn has_memory_budget(&self, estimated_bytes: u64) -> bool {
        self.total_memory + estimated_bytes <= self.config.memory_budget
    }

    /// Get an iterator over all cells.
    pub fn iter_cells(&self) -> impl Iterator<Item = (&CellCoord3D, &WorldCell)> {
        self.cells.iter()
    }
}

// ===========================================================================
// StreamingVolumeShape — shape definitions for streaming triggers
// ===========================================================================

/// Shape of a streaming volume trigger.
#[derive(Debug, Clone, Copy)]
pub enum StreamingVolumeShape {
    /// Axis-aligned box defined by half-extents.
    Box {
        half_extents: WorldPosition,
    },
    /// Sphere with a radius.
    Sphere {
        radius: f64,
    },
    /// Cylinder along the Y axis.
    Cylinder {
        radius: f64,
        half_height: f64,
    },
}

impl StreamingVolumeShape {
    /// Check if a world position is inside this shape, given the shape center.
    pub fn contains(&self, center: WorldPosition, point: WorldPosition) -> bool {
        match self {
            StreamingVolumeShape::Box { half_extents } => {
                let dx = (point.x - center.x).abs();
                let dy = (point.y - center.y).abs();
                let dz = (point.z - center.z).abs();
                dx <= half_extents.x && dy <= half_extents.y && dz <= half_extents.z
            }
            StreamingVolumeShape::Sphere { radius } => {
                WorldPosition::distance_sq(center, point) <= radius * radius
            }
            StreamingVolumeShape::Cylinder {
                radius,
                half_height,
            } => {
                let dy = (point.y - center.y).abs();
                if dy > *half_height {
                    return false;
                }
                let dx = point.x - center.x;
                let dz = point.z - center.z;
                (dx * dx + dz * dz) <= radius * radius
            }
        }
    }

    /// Approximate volume of the shape.
    pub fn volume(&self) -> f64 {
        match self {
            StreamingVolumeShape::Box { half_extents } => {
                8.0 * half_extents.x * half_extents.y * half_extents.z
            }
            StreamingVolumeShape::Sphere { radius } => {
                (4.0 / 3.0) * std::f64::consts::PI * radius * radius * radius
            }
            StreamingVolumeShape::Cylinder {
                radius,
                half_height,
            } => std::f64::consts::PI * radius * radius * 2.0 * half_height,
        }
    }
}

// ===========================================================================
// StreamingVolume — triggers level/asset loading
// ===========================================================================

/// Priority level for streaming volumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StreamingPriority {
    /// Low priority: background loading, non-essential content.
    Low = 0,
    /// Normal priority: standard game content.
    Normal = 1,
    /// High priority: player-visible, gameplay-critical.
    High = 2,
    /// Critical: must be loaded before player can interact.
    Critical = 3,
}

/// A volume in the world that triggers streaming of associated content.
#[derive(Debug, Clone)]
pub struct StreamingVolume {
    /// Unique identifier.
    pub id: u64,
    /// Human-readable name.
    pub name: String,
    /// Center position in world space.
    pub center: WorldPosition,
    /// Shape and dimensions.
    pub shape: StreamingVolumeShape,
    /// Priority for loading.
    pub priority: StreamingPriority,
    /// Preload radius multiplier: content starts loading when the player
    /// is within `shape_bounds * preload_radius` even if not inside yet.
    pub preload_radius: f64,
    /// Associated level/asset paths to load when triggered.
    pub associated_assets: Vec<String>,
    /// Whether this volume is currently active.
    pub active: bool,
    /// Whether the player is currently inside this volume.
    pub player_inside: bool,
    /// Whether the content is loaded.
    pub content_loaded: bool,
    /// Distance to the nearest player (updated each frame).
    pub player_distance: f64,
    /// Whether this volume should unload content when the player leaves.
    pub auto_unload: bool,
    /// Delay before unloading after the player leaves (seconds).
    pub unload_delay: f64,
    /// Time since the player left (for unload delay tracking).
    pub time_since_exit: f64,
    /// Whether this is a one-shot trigger (loads once, never unloads).
    pub one_shot: bool,
    /// Whether this trigger has already fired (for one-shot volumes).
    pub has_triggered: bool,
}

impl StreamingVolume {
    /// Create a new streaming volume.
    pub fn new(
        id: u64,
        name: impl Into<String>,
        center: WorldPosition,
        shape: StreamingVolumeShape,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            center,
            shape,
            priority: StreamingPriority::Normal,
            preload_radius: DEFAULT_PRELOAD_MULTIPLIER,
            associated_assets: Vec::new(),
            active: true,
            player_inside: false,
            content_loaded: false,
            player_distance: f64::MAX,
            auto_unload: true,
            unload_delay: 5.0,
            time_since_exit: 0.0,
            one_shot: false,
            has_triggered: false,
        }
    }

    /// Check if a position is inside the volume.
    pub fn contains(&self, pos: WorldPosition) -> bool {
        self.shape.contains(self.center, pos)
    }

    /// Check if a position is within preload range.
    pub fn is_in_preload_range(&self, pos: WorldPosition) -> bool {
        let dist = WorldPosition::distance(self.center, pos);
        let preload_dist = self.preload_distance();
        dist <= preload_dist
    }

    /// Calculate the effective preload distance.
    pub fn preload_distance(&self) -> f64 {
        match self.shape {
            StreamingVolumeShape::Sphere { radius } => radius * self.preload_radius,
            StreamingVolumeShape::Box { half_extents } => {
                let max_extent = half_extents.x.max(half_extents.y).max(half_extents.z);
                max_extent * self.preload_radius
            }
            StreamingVolumeShape::Cylinder {
                radius,
                half_height,
            } => radius.max(half_height) * self.preload_radius,
        }
    }

    /// Update the volume's state based on player position.
    pub fn update(&mut self, player_pos: WorldPosition, dt: f64) -> StreamingVolumeEvent {
        if !self.active {
            return StreamingVolumeEvent::None;
        }

        if self.one_shot && self.has_triggered {
            return StreamingVolumeEvent::None;
        }

        self.player_distance = WorldPosition::distance(self.center, player_pos);
        let was_inside = self.player_inside;
        self.player_inside = self.contains(player_pos);

        // Player entered the volume.
        if self.player_inside && !was_inside {
            self.time_since_exit = 0.0;
            if self.one_shot {
                self.has_triggered = true;
            }
            return StreamingVolumeEvent::Enter { volume_id: self.id };
        }

        // Player exited the volume.
        if !self.player_inside && was_inside {
            self.time_since_exit = 0.0;
            return StreamingVolumeEvent::Exit { volume_id: self.id };
        }

        // Player is outside and auto-unload is pending.
        if !self.player_inside && self.content_loaded && self.auto_unload {
            self.time_since_exit += dt;
            if self.time_since_exit >= self.unload_delay {
                return StreamingVolumeEvent::Unload { volume_id: self.id };
            }
        }

        // Check preload.
        if !self.content_loaded && self.is_in_preload_range(player_pos) {
            return StreamingVolumeEvent::Preload { volume_id: self.id };
        }

        StreamingVolumeEvent::None
    }
}

/// Events emitted by streaming volumes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StreamingVolumeEvent {
    /// No event.
    None,
    /// Player entered the volume.
    Enter { volume_id: u64 },
    /// Player exited the volume.
    Exit { volume_id: u64 },
    /// Content should be preloaded.
    Preload { volume_id: u64 },
    /// Content should be unloaded.
    Unload { volume_id: u64 },
}

/// Manages all streaming volumes in the world.
#[derive(Debug)]
pub struct StreamingVolumeManager {
    /// All registered volumes.
    volumes: Vec<StreamingVolume>,
    /// Next ID to assign.
    next_id: u64,
    /// Events from the current frame.
    pub pending_events: Vec<StreamingVolumeEvent>,
}

impl StreamingVolumeManager {
    /// Create a new manager.
    pub fn new() -> Self {
        Self {
            volumes: Vec::new(),
            next_id: 1,
            pending_events: Vec::new(),
        }
    }

    /// Add a streaming volume and return its ID.
    pub fn add_volume(&mut self, mut volume: StreamingVolume) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        volume.id = id;
        self.volumes.push(volume);
        id
    }

    /// Remove a volume by ID.
    pub fn remove_volume(&mut self, id: u64) -> bool {
        if let Some(pos) = self.volumes.iter().position(|v| v.id == id) {
            self.volumes.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Get a volume by ID.
    pub fn get_volume(&self, id: u64) -> Option<&StreamingVolume> {
        self.volumes.iter().find(|v| v.id == id)
    }

    /// Get a mutable volume by ID.
    pub fn get_volume_mut(&mut self, id: u64) -> Option<&mut StreamingVolume> {
        self.volumes.iter_mut().find(|v| v.id == id)
    }

    /// Update all volumes based on the player position.
    pub fn update(&mut self, player_pos: WorldPosition, dt: f64) {
        self.pending_events.clear();

        // Sort volumes by priority (critical first) for processing.
        self.volumes.sort_by(|a, b| b.priority.cmp(&a.priority));

        for volume in &mut self.volumes {
            let event = volume.update(player_pos, dt);
            if event != StreamingVolumeEvent::None {
                self.pending_events.push(event);
            }
        }
    }

    /// Get all volumes containing a position.
    pub fn volumes_at(&self, pos: WorldPosition) -> Vec<u64> {
        self.volumes
            .iter()
            .filter(|v| v.active && v.contains(pos))
            .map(|v| v.id)
            .collect()
    }

    /// Get the number of registered volumes.
    pub fn count(&self) -> usize {
        self.volumes.len()
    }

    /// Clear all volumes.
    pub fn clear(&mut self) {
        self.volumes.clear();
        self.pending_events.clear();
    }
}

impl Default for StreamingVolumeManager {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// WorldBounds — practical limits and precision info
// ===========================================================================

/// Describes the practical limits of the world.
#[derive(Debug, Clone)]
pub struct WorldBounds {
    /// Minimum corner of the world.
    pub min: WorldPosition,
    /// Maximum corner of the world.
    pub max: WorldPosition,
    /// Cell size used for subdivision.
    pub cell_size: f64,
}

impl WorldBounds {
    /// Create world bounds from min/max corners.
    pub fn new(min: WorldPosition, max: WorldPosition, cell_size: f64) -> Self {
        Self {
            min,
            max,
            cell_size,
        }
    }

    /// Create default world bounds (2 billion units on each axis).
    pub fn default_bounds() -> Self {
        Self {
            min: WorldPosition::new(MIN_COORDINATE, MIN_COORDINATE, MIN_COORDINATE),
            max: WorldPosition::new(MAX_COORDINATE, MAX_COORDINATE, MAX_COORDINATE),
            cell_size: DEFAULT_CELL_SIZE,
        }
    }

    /// Check if a position is within bounds.
    pub fn contains(&self, pos: WorldPosition) -> bool {
        pos.x >= self.min.x
            && pos.x <= self.max.x
            && pos.y >= self.min.y
            && pos.y <= self.max.y
            && pos.z >= self.min.z
            && pos.z <= self.max.z
    }

    /// Clamp a position to be within bounds.
    pub fn clamp(&self, pos: WorldPosition) -> WorldPosition {
        WorldPosition {
            x: pos.x.clamp(self.min.x, self.max.x),
            y: pos.y.clamp(self.min.y, self.max.y),
            z: pos.z.clamp(self.min.z, self.max.z),
        }
    }

    /// World extents (max - min on each axis).
    pub fn extents(&self) -> WorldPosition {
        WorldPosition {
            x: self.max.x - self.min.x,
            y: self.max.y - self.min.y,
            z: self.max.z - self.min.z,
        }
    }

    /// Total number of cells that fit in the world.
    pub fn total_cells(&self) -> u64 {
        let ext = self.extents();
        let cx = (ext.x / self.cell_size).ceil() as u64;
        let cy = (ext.y / self.cell_size).ceil() as u64;
        let cz = (ext.z / self.cell_size).ceil() as u64;
        cx * cy * cz
    }

    /// Center of the world bounds.
    pub fn center(&self) -> WorldPosition {
        WorldPosition::lerp(self.min, self.max, 0.5)
    }

    /// Calculate the f32 precision at a given distance from the origin.
    ///
    /// Returns the approximate smallest representable step in meters.
    /// f32 provides ~7 decimal digits of precision, so at distance D the
    /// smallest step is approximately D * 1.2e-7.
    pub fn precision_at_distance(distance_from_origin: f64) -> f64 {
        distance_from_origin * 1.192_092_9e-7 // f32 machine epsilon
    }

    /// Calculate the f64 precision at a given distance from the origin.
    pub fn precision_at_distance_f64(distance_from_origin: f64) -> f64 {
        distance_from_origin * 2.220_446_049_250_313e-16 // f64 machine epsilon
    }

    /// Report on precision at various distances.
    pub fn precision_report() -> Vec<(f64, f64, f64)> {
        let distances = [
            1.0,
            10.0,
            100.0,
            1_000.0,
            10_000.0,
            100_000.0,
            1_000_000.0,
            10_000_000.0,
            100_000_000.0,
            1_000_000_000.0,
        ];
        distances
            .iter()
            .map(|&d| {
                (
                    d,
                    Self::precision_at_distance(d),
                    Self::precision_at_distance_f64(d),
                )
            })
            .collect()
    }
}

impl Default for WorldBounds {
    fn default() -> Self {
        Self::default_bounds()
    }
}

// ===========================================================================
// LODLevel — level-of-detail classification
// ===========================================================================

/// Level of detail for entities or systems.
///
/// Used across rendering, physics, AI, and audio to scale fidelity based
/// on distance. Lower LOD levels reduce resource consumption.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LODLevel {
    /// Full quality: all features enabled.
    Full = 0,
    /// Reduced quality: simplified meshes, fewer particles, reduced physics.
    Reduced = 1,
    /// Minimal quality: impostor rendering, no physics, simplified AI.
    Minimal = 2,
    /// Proxy: billboards, no simulation at all.
    Proxy = 3,
    /// Hidden: not rendered, not simulated.
    Hidden = 4,
}

impl LODLevel {
    /// Get the physics simulation rate multiplier for this LOD.
    pub fn physics_rate_multiplier(&self) -> f32 {
        match self {
            LODLevel::Full => 1.0,
            LODLevel::Reduced => 0.5,
            LODLevel::Minimal => 0.1,
            LODLevel::Proxy => 0.0,
            LODLevel::Hidden => 0.0,
        }
    }

    /// Get the AI update rate multiplier.
    pub fn ai_rate_multiplier(&self) -> f32 {
        match self {
            LODLevel::Full => 1.0,
            LODLevel::Reduced => 0.5,
            LODLevel::Minimal => 0.2,
            LODLevel::Proxy => 0.05,
            LODLevel::Hidden => 0.0,
        }
    }

    /// Get the audio processing level.
    pub fn audio_quality(&self) -> AudioLODQuality {
        match self {
            LODLevel::Full => AudioLODQuality::Full,
            LODLevel::Reduced => AudioLODQuality::Reduced,
            LODLevel::Minimal => AudioLODQuality::Ambient,
            LODLevel::Proxy => AudioLODQuality::Silent,
            LODLevel::Hidden => AudioLODQuality::Silent,
        }
    }

    /// Get the rendering mesh quality.
    pub fn mesh_quality(&self) -> f32 {
        match self {
            LODLevel::Full => 1.0,
            LODLevel::Reduced => 0.5,
            LODLevel::Minimal => 0.15,
            LODLevel::Proxy => 0.0,
            LODLevel::Hidden => 0.0,
        }
    }

    /// Get the particle emission rate multiplier.
    pub fn particle_rate_multiplier(&self) -> f32 {
        match self {
            LODLevel::Full => 1.0,
            LODLevel::Reduced => 0.5,
            LODLevel::Minimal => 0.1,
            LODLevel::Proxy => 0.0,
            LODLevel::Hidden => 0.0,
        }
    }
}

impl Default for LODLevel {
    fn default() -> Self {
        LODLevel::Full
    }
}

/// Audio LOD quality levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioLODQuality {
    /// Full 3D spatialized audio with HRTF.
    Full,
    /// Simplified panning, reduced effects.
    Reduced,
    /// Ambient only (no per-entity sounds).
    Ambient,
    /// No audio.
    Silent,
}

// ===========================================================================
// LODThresholds — configurable distance thresholds
// ===========================================================================

/// Distance thresholds for LOD transitions.
///
/// Hysteresis is applied: the threshold to transition *back* to a higher LOD
/// is smaller than the threshold to drop to a lower LOD, preventing rapid
/// switching (thrashing) at boundary distances.
#[derive(Debug, Clone, Copy)]
pub struct LODThresholds {
    /// Distance beyond which Full -> Reduced.
    pub full_to_reduced: f64,
    /// Distance beyond which Reduced -> Minimal.
    pub reduced_to_minimal: f64,
    /// Distance beyond which Minimal -> Proxy.
    pub minimal_to_proxy: f64,
    /// Distance beyond which Proxy -> Hidden.
    pub proxy_to_hidden: f64,
    /// Hysteresis factor (multiplicative, e.g., 1.1 = 10% margin).
    pub hysteresis: f64,
}

impl LODThresholds {
    /// Create default LOD thresholds.
    pub fn defaults() -> Self {
        Self {
            full_to_reduced: LOD_DISTANCE_FULL,
            reduced_to_minimal: LOD_DISTANCE_REDUCED,
            minimal_to_proxy: LOD_DISTANCE_MINIMAL,
            proxy_to_hidden: LOD_DISTANCE_PROXY,
            hysteresis: LOD_HYSTERESIS_FACTOR,
        }
    }

    /// Create scaled thresholds (multiply all distances by a factor).
    pub fn scaled(base: &LODThresholds, scale: f64) -> Self {
        Self {
            full_to_reduced: base.full_to_reduced * scale,
            reduced_to_minimal: base.reduced_to_minimal * scale,
            minimal_to_proxy: base.minimal_to_proxy * scale,
            proxy_to_hidden: base.proxy_to_hidden * scale,
            hysteresis: base.hysteresis,
        }
    }

    /// Evaluate the desired LOD level for a given distance.
    ///
    /// Takes the current LOD level to apply hysteresis properly.
    pub fn evaluate(&self, distance: f64, current_lod: LODLevel) -> LODLevel {
        // When transitioning to a worse LOD, use the raw threshold.
        // When transitioning to a better LOD, require the distance to be
        // *below* threshold / hysteresis.
        let h = self.hysteresis;

        // Check from worst to best.
        if distance > self.proxy_to_hidden {
            return LODLevel::Hidden;
        }

        if distance > self.minimal_to_proxy {
            if current_lod == LODLevel::Hidden && distance > self.proxy_to_hidden / h {
                return LODLevel::Hidden;
            }
            return LODLevel::Proxy;
        }

        if distance > self.reduced_to_minimal {
            if current_lod == LODLevel::Proxy && distance > self.minimal_to_proxy / h {
                return LODLevel::Proxy;
            }
            return LODLevel::Minimal;
        }

        if distance > self.full_to_reduced {
            if current_lod == LODLevel::Minimal && distance > self.reduced_to_minimal / h {
                return LODLevel::Minimal;
            }
            return LODLevel::Reduced;
        }

        if current_lod == LODLevel::Reduced && distance > self.full_to_reduced / h {
            return LODLevel::Reduced;
        }

        LODLevel::Full
    }
}

impl Default for LODThresholds {
    fn default() -> Self {
        Self::defaults()
    }
}

// ===========================================================================
// LODEntity — per-entity LOD tracking
// ===========================================================================

/// Per-entity LOD state.
#[derive(Debug, Clone)]
pub struct LODEntity {
    /// Entity identifier.
    pub entity_id: u64,
    /// World position (f64).
    pub position: WorldPosition,
    /// Current LOD level.
    pub current_lod: LODLevel,
    /// Previous LOD level (for transition detection).
    pub previous_lod: LODLevel,
    /// Custom threshold overrides (if any).
    pub custom_thresholds: Option<LODThresholds>,
    /// Whether LOD is locked (e.g., always Full for important NPCs).
    pub locked: bool,
    /// LOD importance bias (multiplied to distance; <1 = stay high LOD longer).
    pub importance_bias: f64,
    /// Frame of last LOD evaluation.
    pub last_eval_frame: u64,
    /// Whether the LOD changed this frame.
    pub lod_changed: bool,
}

impl LODEntity {
    /// Create a new LOD entity.
    pub fn new(entity_id: u64, position: WorldPosition) -> Self {
        Self {
            entity_id,
            position,
            current_lod: LODLevel::Full,
            previous_lod: LODLevel::Full,
            custom_thresholds: None,
            locked: false,
            importance_bias: 1.0,
            last_eval_frame: 0,
            lod_changed: false,
        }
    }
}

// ===========================================================================
// LODSystem — manages LOD for all entities
// ===========================================================================

/// Manages LOD evaluation for all tracked entities.
#[derive(Debug)]
pub struct LODSystem {
    /// Default LOD thresholds.
    pub default_thresholds: LODThresholds,
    /// Tracked entities.
    entities: Vec<LODEntity>,
    /// Current frame.
    frame: u64,
    /// Maximum entities to evaluate per frame (for spreading the cost).
    pub max_evals_per_frame: usize,
    /// Index of the next entity to evaluate (round-robin).
    next_eval_index: usize,
    /// Statistics.
    pub stats: LODSystemStats,
}

/// LOD system statistics.
#[derive(Debug, Clone, Default)]
pub struct LODSystemStats {
    /// Total entities tracked.
    pub total_entities: usize,
    /// Count per LOD level.
    pub count_full: usize,
    pub count_reduced: usize,
    pub count_minimal: usize,
    pub count_proxy: usize,
    pub count_hidden: usize,
    /// Number of LOD transitions this frame.
    pub transitions_this_frame: usize,
    /// Entities evaluated this frame.
    pub evals_this_frame: usize,
}

impl LODSystem {
    /// Create a new LOD system.
    pub fn new(thresholds: LODThresholds) -> Self {
        Self {
            default_thresholds: thresholds,
            entities: Vec::new(),
            frame: 0,
            max_evals_per_frame: 1024,
            next_eval_index: 0,
            stats: LODSystemStats::default(),
        }
    }

    /// Create with default thresholds.
    pub fn with_defaults() -> Self {
        Self::new(LODThresholds::defaults())
    }

    /// Register an entity for LOD tracking.
    pub fn add_entity(&mut self, entity: LODEntity) {
        self.entities.push(entity);
    }

    /// Remove an entity by ID.
    pub fn remove_entity(&mut self, entity_id: u64) -> bool {
        if let Some(pos) = self.entities.iter().position(|e| e.entity_id == entity_id) {
            self.entities.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Get an entity's LOD state.
    pub fn get_entity(&self, entity_id: u64) -> Option<&LODEntity> {
        self.entities.iter().find(|e| e.entity_id == entity_id)
    }

    /// Update LOD for all entities based on camera position.
    ///
    /// Evaluates up to `max_evals_per_frame` entities in a round-robin
    /// fashion to spread the cost across frames.
    pub fn update(&mut self, camera_pos: WorldPosition) {
        self.frame += 1;
        self.stats.transitions_this_frame = 0;
        self.stats.evals_this_frame = 0;

        if self.entities.is_empty() {
            self.update_counts();
            return;
        }

        let count = self.entities.len();
        let evals = self.max_evals_per_frame.min(count);

        for _ in 0..evals {
            if self.next_eval_index >= count {
                self.next_eval_index = 0;
            }

            let entity = &mut self.entities[self.next_eval_index];

            if !entity.locked {
                let distance = WorldPosition::distance(camera_pos, entity.position);
                let biased_distance = distance * entity.importance_bias;

                let thresholds = entity
                    .custom_thresholds
                    .as_ref()
                    .unwrap_or(&self.default_thresholds);

                let new_lod = thresholds.evaluate(biased_distance, entity.current_lod);

                entity.previous_lod = entity.current_lod;
                entity.lod_changed = new_lod != entity.current_lod;
                entity.current_lod = new_lod;
                entity.last_eval_frame = self.frame;

                if entity.lod_changed {
                    self.stats.transitions_this_frame += 1;
                }
            }

            self.stats.evals_this_frame += 1;
            self.next_eval_index += 1;
        }

        self.update_counts();
    }

    /// Force-evaluate all entities immediately (expensive).
    pub fn evaluate_all(&mut self, camera_pos: WorldPosition) {
        self.frame += 1;
        self.stats.transitions_this_frame = 0;

        for entity in &mut self.entities {
            if entity.locked {
                continue;
            }

            let distance = WorldPosition::distance(camera_pos, entity.position);
            let biased_distance = distance * entity.importance_bias;

            let thresholds = entity
                .custom_thresholds
                .as_ref()
                .unwrap_or(&self.default_thresholds);

            let new_lod = thresholds.evaluate(biased_distance, entity.current_lod);

            entity.previous_lod = entity.current_lod;
            entity.lod_changed = new_lod != entity.current_lod;
            entity.current_lod = new_lod;
            entity.last_eval_frame = self.frame;

            if entity.lod_changed {
                self.stats.transitions_this_frame += 1;
            }
        }

        self.stats.evals_this_frame = self.entities.len();
        self.update_counts();
    }

    /// Update per-LOD-level counts.
    fn update_counts(&mut self) {
        self.stats.total_entities = self.entities.len();
        self.stats.count_full = 0;
        self.stats.count_reduced = 0;
        self.stats.count_minimal = 0;
        self.stats.count_proxy = 0;
        self.stats.count_hidden = 0;

        for entity in &self.entities {
            match entity.current_lod {
                LODLevel::Full => self.stats.count_full += 1,
                LODLevel::Reduced => self.stats.count_reduced += 1,
                LODLevel::Minimal => self.stats.count_minimal += 1,
                LODLevel::Proxy => self.stats.count_proxy += 1,
                LODLevel::Hidden => self.stats.count_hidden += 1,
            }
        }
    }

    /// Get entities that changed LOD this frame.
    pub fn changed_entities(&self) -> Vec<&LODEntity> {
        self.entities.iter().filter(|e| e.lod_changed).collect()
    }

    /// Get all entities at a specific LOD level.
    pub fn entities_at_lod(&self, lod: LODLevel) -> Vec<&LODEntity> {
        self.entities
            .iter()
            .filter(|e| e.current_lod == lod)
            .collect()
    }

    /// Total entity count.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Clear all entities.
    pub fn clear(&mut self) {
        self.entities.clear();
        self.next_eval_index = 0;
    }
}

// ===========================================================================
// SectorSimRate — simulation rates for sectors
// ===========================================================================

/// The simulation tier for a sector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SectorTier {
    /// Full simulation (60 Hz default).
    Active,
    /// Reduced simulation (10 Hz default).
    Near,
    /// Minimal simulation (1 Hz default).
    Far,
    /// No simulation; state is preserved but not updated.
    Dormant,
}

impl SectorTier {
    /// Get the simulation rate in Hz for this tier.
    pub fn rate_hz(&self) -> f32 {
        match self {
            SectorTier::Active => ACTIVE_SECTOR_RATE,
            SectorTier::Near => NEAR_SECTOR_RATE,
            SectorTier::Far => FAR_SECTOR_RATE,
            SectorTier::Dormant => 0.0,
        }
    }

    /// Get the time step for this tier's simulation rate.
    pub fn time_step(&self) -> f32 {
        let rate = self.rate_hz();
        if rate > 0.0 {
            1.0 / rate
        } else {
            0.0
        }
    }

    /// Whether this tier has any simulation.
    pub fn is_simulated(&self) -> bool {
        !matches!(self, SectorTier::Dormant)
    }
}

// ===========================================================================
// Sector — a region of the world with its own simulation rate
// ===========================================================================

/// A sector of the world with its own simulation tier and state.
#[derive(Debug, Clone)]
pub struct Sector {
    /// Unique sector identifier.
    pub id: u64,
    /// Human-readable name.
    pub name: String,
    /// Center position in world space.
    pub center: WorldPosition,
    /// Radius of the sector.
    pub radius: f64,
    /// Current simulation tier.
    pub tier: SectorTier,
    /// Previous simulation tier (for transition detection).
    pub previous_tier: SectorTier,
    /// Entities in this sector.
    pub entities: Vec<u64>,
    /// Time accumulator for reduced-rate simulation.
    pub time_accumulator: f64,
    /// Whether the sector is active (can be disabled independently).
    pub active: bool,
    /// Last simulation timestamp.
    pub last_sim_time: f64,
    /// Number of simulation steps performed this session.
    pub sim_step_count: u64,
    /// Serialized dormant state (for dormant sectors).
    pub dormant_state: Option<Vec<u8>>,
    /// Distance to camera (updated each frame).
    pub camera_distance: f64,
    /// Custom data tag.
    pub tag: u32,
}

impl Sector {
    /// Create a new sector.
    pub fn new(id: u64, name: impl Into<String>, center: WorldPosition, radius: f64) -> Self {
        Self {
            id,
            name: name.into(),
            center,
            radius,
            tier: SectorTier::Dormant,
            previous_tier: SectorTier::Dormant,
            entities: Vec::new(),
            time_accumulator: 0.0,
            active: true,
            last_sim_time: 0.0,
            sim_step_count: 0,
            dormant_state: None,
            camera_distance: f64::MAX,
            tag: 0,
        }
    }

    /// Check if a position is within this sector.
    pub fn contains(&self, pos: WorldPosition) -> bool {
        WorldPosition::distance_sq(self.center, pos) <= self.radius * self.radius
    }

    /// Whether this sector should simulate this frame.
    pub fn should_simulate(&self, dt: f64) -> bool {
        if !self.active || self.tier == SectorTier::Dormant {
            return false;
        }
        let step = self.tier.time_step() as f64;
        if step <= 0.0 {
            return false;
        }
        self.time_accumulator + dt >= step
    }

    /// Advance the time accumulator and return the number of simulation
    /// steps to perform this frame.
    pub fn advance_time(&mut self, dt: f64) -> u32 {
        if !self.active || self.tier == SectorTier::Dormant {
            return 0;
        }
        let step = self.tier.time_step() as f64;
        if step <= 0.0 {
            return 0;
        }

        self.time_accumulator += dt;
        let steps = (self.time_accumulator / step).floor() as u32;
        self.time_accumulator -= steps as f64 * step;
        self.sim_step_count += steps as u64;
        steps
    }

    /// Add an entity to this sector.
    pub fn add_entity(&mut self, entity_id: u64) {
        if !self.entities.contains(&entity_id) {
            self.entities.push(entity_id);
        }
    }

    /// Remove an entity from this sector.
    pub fn remove_entity(&mut self, entity_id: u64) -> bool {
        if let Some(pos) = self.entities.iter().position(|&e| e == entity_id) {
            self.entities.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Store dormant state snapshot.
    pub fn enter_dormant(&mut self, state_snapshot: Vec<u8>) {
        self.dormant_state = Some(state_snapshot);
        self.previous_tier = self.tier;
        self.tier = SectorTier::Dormant;
        self.time_accumulator = 0.0;
    }

    /// Retrieve and clear dormant state snapshot.
    pub fn exit_dormant(&mut self, new_tier: SectorTier) -> Option<Vec<u8>> {
        self.previous_tier = self.tier;
        self.tier = new_tier;
        self.dormant_state.take()
    }
}

// ===========================================================================
// SectorManager — manages sectors with variable simulation rates
// ===========================================================================

/// Configuration for sector tier transitions based on distance.
#[derive(Debug, Clone)]
pub struct SectorManagerConfig {
    /// Distance below which a sector is Active.
    pub active_radius: f64,
    /// Distance below which a sector is Near.
    pub near_radius: f64,
    /// Distance below which a sector is Far.
    pub far_radius: f64,
    /// Distance beyond which a sector becomes Dormant.
    pub dormant_radius: f64,
    /// Hysteresis factor for tier transitions.
    pub hysteresis: f64,
}

impl Default for SectorManagerConfig {
    fn default() -> Self {
        Self {
            active_radius: 500.0,
            near_radius: 2000.0,
            far_radius: 10_000.0,
            dormant_radius: 50_000.0,
            hysteresis: 1.15,
        }
    }
}

/// Manages all world sectors and their simulation rates.
#[derive(Debug)]
pub struct SectorManager {
    /// Configuration.
    pub config: SectorManagerConfig,
    /// All sectors.
    sectors: Vec<Sector>,
    /// Next sector ID.
    next_id: u64,
    /// Statistics.
    pub stats: SectorManagerStats,
}

/// Statistics for the sector manager.
#[derive(Debug, Clone, Default)]
pub struct SectorManagerStats {
    /// Total sectors.
    pub total_sectors: usize,
    /// Active sectors.
    pub active_sectors: usize,
    /// Near sectors.
    pub near_sectors: usize,
    /// Far sectors.
    pub far_sectors: usize,
    /// Dormant sectors.
    pub dormant_sectors: usize,
    /// Total entities across all sectors.
    pub total_entities: usize,
    /// Tier transitions this frame.
    pub transitions_this_frame: usize,
    /// Simulation steps performed this frame.
    pub sim_steps_this_frame: u32,
}

impl SectorManager {
    /// Create a new sector manager.
    pub fn new(config: SectorManagerConfig) -> Self {
        Self {
            config,
            sectors: Vec::new(),
            next_id: 1,
            stats: SectorManagerStats::default(),
        }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(SectorManagerConfig::default())
    }

    /// Add a sector and return its ID.
    pub fn add_sector(&mut self, mut sector: Sector) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        sector.id = id;
        self.sectors.push(sector);
        id
    }

    /// Remove a sector by ID.
    pub fn remove_sector(&mut self, id: u64) -> Option<Sector> {
        if let Some(pos) = self.sectors.iter().position(|s| s.id == id) {
            Some(self.sectors.swap_remove(pos))
        } else {
            None
        }
    }

    /// Get a sector by ID.
    pub fn get_sector(&self, id: u64) -> Option<&Sector> {
        self.sectors.iter().find(|s| s.id == id)
    }

    /// Get a mutable sector by ID.
    pub fn get_sector_mut(&mut self, id: u64) -> Option<&mut Sector> {
        self.sectors.iter_mut().find(|s| s.id == id)
    }

    /// Determine the appropriate tier for a sector based on distance.
    fn tier_for_distance(&self, distance: f64, current_tier: SectorTier) -> SectorTier {
        let h = self.config.hysteresis;

        if distance <= self.config.active_radius {
            return SectorTier::Active;
        }
        if distance <= self.config.near_radius {
            // Apply hysteresis: if currently Active, require crossing
            // active_radius * hysteresis before downgrading.
            if current_tier == SectorTier::Active
                && distance <= self.config.active_radius * h
            {
                return SectorTier::Active;
            }
            return SectorTier::Near;
        }
        if distance <= self.config.far_radius {
            if current_tier == SectorTier::Near
                && distance <= self.config.near_radius * h
            {
                return SectorTier::Near;
            }
            return SectorTier::Far;
        }
        if distance <= self.config.dormant_radius {
            if current_tier == SectorTier::Far
                && distance <= self.config.far_radius * h
            {
                return SectorTier::Far;
            }
            return SectorTier::Far;
        }

        SectorTier::Dormant
    }

    /// Update all sectors based on camera position.
    ///
    /// Adjusts tier assignments and returns simulation step counts per sector.
    pub fn update(
        &mut self,
        camera_pos: WorldPosition,
        dt: f64,
    ) -> Vec<(u64, SectorTier, u32)> {
        self.stats.transitions_this_frame = 0;
        self.stats.sim_steps_this_frame = 0;

        let mut results = Vec::new();

        // Cache config values to avoid borrow conflict
        let cfg_active_radius = self.config.active_radius;
        let cfg_near_radius = self.config.near_radius;
        let cfg_far_radius = self.config.far_radius;
        let cfg_dormant_radius = self.config.dormant_radius;
        let cfg_hysteresis = self.config.hysteresis;

        for sector in &mut self.sectors {
            if !sector.active {
                continue;
            }

            // Update distance.
            sector.camera_distance = WorldPosition::distance(camera_pos, sector.center);

            // Determine new tier (inlined to avoid self borrow conflict).
            let new_tier = {
                let distance = sector.camera_distance;
                let current_tier = sector.tier;
                let h = cfg_hysteresis;
                if distance <= cfg_active_radius {
                    SectorTier::Active
                } else if distance <= cfg_near_radius {
                    if current_tier == SectorTier::Active && distance <= cfg_active_radius * h {
                        SectorTier::Active
                    } else {
                        SectorTier::Near
                    }
                } else if distance <= cfg_far_radius {
                    if current_tier == SectorTier::Near && distance <= cfg_near_radius * h {
                        SectorTier::Near
                    } else {
                        SectorTier::Far
                    }
                } else if distance <= cfg_dormant_radius {
                    if current_tier == SectorTier::Far && distance <= cfg_far_radius * h {
                        SectorTier::Far
                    } else {
                        SectorTier::Far
                    }
                } else {
                    SectorTier::Dormant
                }
            };

            if new_tier != sector.tier {
                sector.previous_tier = sector.tier;
                sector.tier = new_tier;
                self.stats.transitions_this_frame += 1;
            }

            // Advance simulation.
            let steps = sector.advance_time(dt);
            self.stats.sim_steps_this_frame += steps;

            results.push((sector.id, sector.tier, steps));
        }

        // Update statistics.
        self.stats.total_sectors = self.sectors.len();
        self.stats.active_sectors = self
            .sectors
            .iter()
            .filter(|s| s.tier == SectorTier::Active)
            .count();
        self.stats.near_sectors = self
            .sectors
            .iter()
            .filter(|s| s.tier == SectorTier::Near)
            .count();
        self.stats.far_sectors = self
            .sectors
            .iter()
            .filter(|s| s.tier == SectorTier::Far)
            .count();
        self.stats.dormant_sectors = self
            .sectors
            .iter()
            .filter(|s| s.tier == SectorTier::Dormant)
            .count();
        self.stats.total_entities = self.sectors.iter().map(|s| s.entities.len()).sum();

        results
    }

    /// Find which sector a position belongs to.
    pub fn sector_at(&self, pos: WorldPosition) -> Option<u64> {
        self.sectors
            .iter()
            .filter(|s| s.contains(pos))
            .min_by(|a, b| {
                a.camera_distance
                    .partial_cmp(&b.camera_distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.id)
    }

    /// Get all sectors of a specific tier.
    pub fn sectors_of_tier(&self, tier: SectorTier) -> Vec<&Sector> {
        self.sectors.iter().filter(|s| s.tier == tier).collect()
    }

    /// Get all sector IDs that need simulation this frame.
    pub fn sectors_needing_simulation(&self) -> Vec<u64> {
        self.sectors
            .iter()
            .filter(|s| s.active && s.tier.is_simulated())
            .map(|s| s.id)
            .collect()
    }

    /// Total sector count.
    pub fn sector_count(&self) -> usize {
        self.sectors.len()
    }

    /// Clear all sectors.
    pub fn clear(&mut self) {
        self.sectors.clear();
    }

    /// Iterate over all sectors.
    pub fn iter(&self) -> impl Iterator<Item = &Sector> {
        self.sectors.iter()
    }
}

// ===========================================================================
// LargeWorldSystem — top-level coordinator
// ===========================================================================

/// Coordinates all large-world subsystems: floating origin, cell grid,
/// streaming volumes, LOD, and sectors.
#[derive(Debug)]
pub struct LargeWorldSystem {
    /// Floating origin manager.
    pub origin: WorldOrigin,
    /// Cell grid for spatial subdivision.
    pub cell_grid: WorldCellGrid,
    /// Streaming volume manager.
    pub streaming_volumes: StreamingVolumeManager,
    /// World bounds.
    pub bounds: WorldBounds,
    /// LOD system.
    pub lod: LODSystem,
    /// Sector manager.
    pub sectors: SectorManager,
    /// Current frame number.
    pub frame: u64,
    /// Current simulation time.
    pub time: f64,
}

impl LargeWorldSystem {
    /// Create a new large world system with default configuration.
    pub fn new() -> Self {
        Self {
            origin: WorldOrigin::with_default_threshold(),
            cell_grid: WorldCellGrid::new(WorldCellConfig::default()),
            streaming_volumes: StreamingVolumeManager::new(),
            bounds: WorldBounds::default(),
            lod: LODSystem::with_defaults(),
            sectors: SectorManager::with_defaults(),
            frame: 0,
            time: 0.0,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(
        origin_threshold: f64,
        cell_config: WorldCellConfig,
        lod_thresholds: LODThresholds,
        sector_config: SectorManagerConfig,
    ) -> Self {
        Self {
            origin: WorldOrigin::new(origin_threshold),
            cell_grid: WorldCellGrid::new(cell_config),
            streaming_volumes: StreamingVolumeManager::new(),
            bounds: WorldBounds::default(),
            lod: LODSystem::new(lod_thresholds),
            sectors: SectorManager::new(sector_config),
            frame: 0,
            time: 0.0,
        }
    }

    /// Main update tick. Call once per frame with the camera's world position.
    ///
    /// Returns a `LargeWorldUpdateResult` describing what happened.
    pub fn update(
        &mut self,
        camera_pos: WorldPosition,
        dt: f64,
    ) -> LargeWorldUpdateResult {
        self.frame += 1;
        self.time += dt;

        let mut result = LargeWorldUpdateResult::default();

        // 1. Floating origin check/rebase.
        if let Some(rebase_event) = self.origin.update(camera_pos) {
            result.rebase_event = Some(rebase_event);
        }

        // 2. Update cell grid.
        self.cell_grid.update_camera(camera_pos, self.time);
        let (loaded, unloaded) = self.cell_grid.process_queues(self.time);
        result.cells_loaded = loaded;
        result.cells_unloaded = unloaded;

        // 3. Update streaming volumes.
        self.streaming_volumes.update(camera_pos, dt);
        result.streaming_events = self.streaming_volumes.pending_events.clone();

        // 4. Update LOD system.
        self.lod.update(camera_pos);

        // 5. Update sectors.
        result.sector_updates = self.sectors.update(camera_pos, dt);

        result
    }

    /// Get the current local-space position of the camera.
    pub fn camera_local(&self, camera_world: WorldPosition) -> Vec3 {
        self.origin.to_local(camera_world)
    }

    /// Convert any world position to local space relative to the current origin.
    pub fn world_to_local(&self, pos: WorldPosition) -> Vec3 {
        self.origin.to_local(pos)
    }

    /// Convert a local position back to world space.
    pub fn local_to_world(&self, local: Vec3) -> WorldPosition {
        self.origin.to_world(local)
    }
}

impl Default for LargeWorldSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a large world system update tick.
#[derive(Debug, Clone, Default)]
pub struct LargeWorldUpdateResult {
    /// Rebase event if floating origin was shifted.
    pub rebase_event: Option<RebaseEvent>,
    /// Cells that were loaded this frame.
    pub cells_loaded: Vec<CellCoord3D>,
    /// Cells that were unloaded this frame.
    pub cells_unloaded: Vec<CellCoord3D>,
    /// Streaming volume events.
    pub streaming_events: Vec<StreamingVolumeEvent>,
    /// Sector update results: (sector_id, tier, simulation_steps).
    pub sector_updates: Vec<(u64, SectorTier, u32)>,
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_position_distance() {
        let a = WorldPosition::new(0.0, 0.0, 0.0);
        let b = WorldPosition::new(3.0, 4.0, 0.0);
        let d = WorldPosition::distance(a, b);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_world_position_to_local() {
        let origin = WorldPosition::new(1_000_000.0, 0.0, 0.0);
        let pos = WorldPosition::new(1_000_001.0, 2.0, 3.0);
        let local = pos.to_local(origin);
        assert!((local.x - 1.0).abs() < 1e-5);
        assert!((local.y - 2.0).abs() < 1e-5);
        assert!((local.z - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_world_position_from_local() {
        let origin = WorldPosition::new(1_000_000.0, 0.0, 0.0);
        let local = Vec3::new(1.0, 2.0, 3.0);
        let world = WorldPosition::from_local(origin, local);
        assert!((world.x - 1_000_001.0).abs() < 1e-10);
    }

    #[test]
    fn test_world_origin_rebase() {
        let mut origin = WorldOrigin::new(100.0);
        origin.min_rebase_interval = 0;
        let camera = WorldPosition::new(200.0, 0.0, 0.0);
        assert!(origin.should_rebase(camera));
        let event = origin.rebase(camera);
        assert!((event.delta.x - 200.0).abs() < 1e-10);
        assert!((origin.origin.x - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_cell_coord_3d() {
        let pos = WorldPosition::new(1500.0, 500.0, -200.0);
        let cell = pos.to_cell_coord(1000.0);
        assert_eq!(cell.x, 1);
        assert_eq!(cell.y, 0);
        assert_eq!(cell.z, -1);
    }

    #[test]
    fn test_lod_thresholds_hysteresis() {
        let thresholds = LODThresholds::defaults();

        // Well beyond the Full threshold -> Reduced.
        let lod = thresholds.evaluate(100.0, LODLevel::Full);
        assert_eq!(lod, LODLevel::Reduced);

        // Just beyond Full threshold but within hysteresis if currently Reduced.
        let lod2 = thresholds.evaluate(
            LOD_DISTANCE_FULL * 1.05,
            LODLevel::Reduced,
        );
        // Should stay Reduced because it's still above threshold.
        assert_eq!(lod2, LODLevel::Reduced);
    }

    #[test]
    fn test_sector_simulation_steps() {
        let mut sector = Sector::new(
            1,
            "test",
            WorldPosition::ZERO,
            1000.0,
        );
        sector.tier = SectorTier::Near; // 10 Hz = 0.1s step
        sector.active = true;

        // Accumulate 0.25 seconds -> 2 steps at 10Hz.
        let steps = sector.advance_time(0.25);
        assert_eq!(steps, 2);
    }

    #[test]
    fn test_streaming_volume_contains() {
        let vol = StreamingVolume::new(
            1,
            "test",
            WorldPosition::new(100.0, 0.0, 100.0),
            StreamingVolumeShape::Sphere { radius: 50.0 },
        );

        assert!(vol.contains(WorldPosition::new(120.0, 0.0, 100.0)));
        assert!(!vol.contains(WorldPosition::new(200.0, 0.0, 100.0)));
    }

    #[test]
    fn test_world_bounds_precision() {
        let report = WorldBounds::precision_report();
        // At 1 unit distance, f32 precision should be ~1.19e-7.
        assert!(report[0].1 < 1e-6);
        // At 1 billion units, f32 precision should be ~119 units.
        let last = report.last().unwrap();
        assert!(last.1 > 10.0);
        // f64 is always much better.
        assert!(last.2 < 0.001);
    }

    #[test]
    fn test_world_position_operations() {
        let a = WorldPosition::new(1.0, 2.0, 3.0);
        let b = WorldPosition::new(4.0, 5.0, 6.0);
        let c = a + b;
        assert!((c.x - 5.0).abs() < 1e-10);
        assert!((c.y - 7.0).abs() < 1e-10);
        assert!((c.z - 9.0).abs() < 1e-10);

        let d = b - a;
        assert!((d.x - 3.0).abs() < 1e-10);

        let e = a * 2.0;
        assert!((e.x - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_large_world_system_update() {
        let mut system = LargeWorldSystem::new();
        let camera = WorldPosition::new(50.0, 0.0, 50.0);
        let result = system.update(camera, 1.0 / 60.0);
        assert!(result.rebase_event.is_none()); // 50 units < 10000 threshold
    }
}
