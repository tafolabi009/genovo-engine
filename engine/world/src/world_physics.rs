//! World-scale physics system for the Genovo engine.
//!
//! Provides physics management features for large, open worlds:
//!
//! - **Physics regions** — define areas with different gravity, time scale,
//!   and physics parameters (e.g. underwater, low-gravity zones, time bubbles).
//! - **Physics LOD** — simplified physics simulation at distance from the
//!   player to save CPU budget.
//! - **Physics streaming** — load/unload physics bodies with world cells as
//!   the player moves through the world.
//! - **Physics proxy objects** — lightweight stand-ins for distant or sleeping
//!   physics bodies.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

/// Minimal 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };
    pub const GRAVITY: Self = Self { x: 0.0, y: -9.81, z: 0.0 };

    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn length_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len < 1e-9 { Self::ZERO } else { Self::new(self.x / len, self.y / len, self.z / len) }
    }

    pub fn add(&self, o: &Self) -> Self { Self::new(self.x + o.x, self.y + o.y, self.z + o.z) }
    pub fn sub(&self, o: &Self) -> Self { Self::new(self.x - o.x, self.y - o.y, self.z - o.z) }
    pub fn scale(&self, s: f32) -> Self { Self::new(self.x * s, self.y * s, self.z * s) }
    pub fn dot(&self, o: &Self) -> f32 { self.x * o.x + self.y * o.y + self.z * o.z }

    pub fn distance(&self, o: &Self) -> f32 { self.sub(o).length() }
    pub fn distance_squared(&self, o: &Self) -> f32 { self.sub(o).length_squared() }
}

impl Default for Vec3 { fn default() -> Self { Self::ZERO } }

// ---------------------------------------------------------------------------
// AABB
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }

    pub fn center(&self) -> Vec3 { self.min.add(&self.max).scale(0.5) }

    pub fn extents(&self) -> Vec3 { self.max.sub(&self.min).scale(0.5) }

    pub fn contains_point(&self, p: Vec3) -> bool {
        p.x >= self.min.x && p.x <= self.max.x
            && p.y >= self.min.y && p.y <= self.max.y
            && p.z >= self.min.z && p.z <= self.max.z
    }

    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
            && self.min.y <= other.max.y && self.max.y >= other.min.y
            && self.min.z <= other.max.z && self.max.z >= other.min.z
    }

    pub fn volume(&self) -> f32 {
        let d = self.max.sub(&self.min);
        d.x * d.y * d.z
    }

    pub fn expand(&self, margin: f32) -> Self {
        Self {
            min: Vec3::new(self.min.x - margin, self.min.y - margin, self.min.z - margin),
            max: Vec3::new(self.max.x + margin, self.max.y + margin, self.max.z + margin),
        }
    }
}

// ---------------------------------------------------------------------------
// Physics body identifiers
// ---------------------------------------------------------------------------

/// Unique identifier for a physics body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicsBodyId(u64);

impl PhysicsBodyId {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    pub fn from_raw(raw: u64) -> Self { Self(raw) }
    pub fn raw(&self) -> u64 { self.0 }
}

impl fmt::Display for PhysicsBodyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PhysBody({})", self.0)
    }
}

/// Unique identifier for a physics region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionId(u64);

impl RegionId {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    pub fn from_raw(raw: u64) -> Self { Self(raw) }
    pub fn raw(&self) -> u64 { self.0 }
}

/// World cell coordinate for physics streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl CellCoord {
    pub fn new(x: i32, y: i32, z: i32) -> Self { Self { x, y, z } }

    pub fn chebyshev_distance(&self, other: &Self) -> u32 {
        (self.x - other.x).unsigned_abs()
            .max((self.y - other.y).unsigned_abs())
            .max((self.z - other.z).unsigned_abs())
    }
}

impl fmt::Display for CellCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({},{},{})", self.x, self.y, self.z)
    }
}

// ---------------------------------------------------------------------------
// Physics region
// ---------------------------------------------------------------------------

/// Physics region shape.
#[derive(Debug, Clone)]
pub enum RegionShape {
    /// Axis-aligned box region.
    Box(AABB),
    /// Spherical region.
    Sphere { center: Vec3, radius: f32 },
    /// Infinite half-space (everything below a Y level).
    HalfSpace { y_level: f32 },
    /// Entire world.
    Global,
}

impl RegionShape {
    /// Tests whether a point is inside the region.
    pub fn contains_point(&self, p: Vec3) -> bool {
        match self {
            Self::Box(aabb) => aabb.contains_point(p),
            Self::Sphere { center, radius } => center.distance(&p) <= *radius,
            Self::HalfSpace { y_level } => p.y <= *y_level,
            Self::Global => true,
        }
    }

    /// Tests whether an AABB intersects the region.
    pub fn intersects_aabb(&self, aabb: &AABB) -> bool {
        match self {
            Self::Box(region_aabb) => region_aabb.intersects(aabb),
            Self::Sphere { center, radius } => {
                // Closest point on AABB to sphere center.
                let closest = Vec3::new(
                    center.x.clamp(aabb.min.x, aabb.max.x),
                    center.y.clamp(aabb.min.y, aabb.max.y),
                    center.z.clamp(aabb.min.z, aabb.max.z),
                );
                center.distance_squared(&closest) <= radius * radius
            }
            Self::HalfSpace { y_level } => aabb.min.y <= *y_level,
            Self::Global => true,
        }
    }
}

/// A physics region with modified physical properties.
#[derive(Debug, Clone)]
pub struct PhysicsRegion {
    /// Unique region id.
    pub id: RegionId,
    /// Human-readable name.
    pub name: String,
    /// Region shape.
    pub shape: RegionShape,
    /// Gravity override (None = use world default).
    pub gravity: Option<Vec3>,
    /// Time scale multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double).
    pub time_scale: f32,
    /// Air density multiplier (affects drag).
    pub air_density: f32,
    /// Whether this region applies buoyancy (water).
    pub buoyancy: bool,
    /// Water surface Y level (for buoyancy calculations).
    pub water_surface_y: f32,
    /// Drag coefficient override.
    pub drag_coefficient: Option<f32>,
    /// Wind direction and strength.
    pub wind: Vec3,
    /// Priority (higher priority region overrides lower).
    pub priority: i32,
    /// Whether the region is active.
    pub enabled: bool,
    /// Linear damping applied to bodies in this region.
    pub linear_damping: f32,
    /// Angular damping applied to bodies in this region.
    pub angular_damping: f32,
}

impl PhysicsRegion {
    /// Creates a standard gravity region.
    pub fn standard(name: &str, shape: RegionShape) -> Self {
        Self {
            id: RegionId::new(),
            name: name.to_string(),
            shape,
            gravity: None,
            time_scale: 1.0,
            air_density: 1.225,
            buoyancy: false,
            water_surface_y: 0.0,
            drag_coefficient: None,
            wind: Vec3::ZERO,
            priority: 0,
            enabled: true,
            linear_damping: 0.01,
            angular_damping: 0.05,
        }
    }

    /// Creates a low-gravity region.
    pub fn low_gravity(name: &str, shape: RegionShape, gravity_factor: f32) -> Self {
        Self {
            gravity: Some(Vec3::GRAVITY.scale(gravity_factor)),
            ..Self::standard(name, shape)
        }
    }

    /// Creates a zero-gravity region.
    pub fn zero_gravity(name: &str, shape: RegionShape) -> Self {
        Self {
            gravity: Some(Vec3::ZERO),
            linear_damping: 0.005,
            angular_damping: 0.01,
            ..Self::standard(name, shape)
        }
    }

    /// Creates an underwater region.
    pub fn underwater(name: &str, shape: RegionShape, surface_y: f32) -> Self {
        Self {
            buoyancy: true,
            water_surface_y: surface_y,
            air_density: 1000.0, // Water density
            drag_coefficient: Some(0.5),
            linear_damping: 0.5,
            angular_damping: 0.5,
            ..Self::standard(name, shape)
        }
    }

    /// Creates a time-dilated region.
    pub fn time_bubble(name: &str, shape: RegionShape, time_scale: f32) -> Self {
        Self {
            time_scale,
            priority: 10,
            ..Self::standard(name, shape)
        }
    }

    /// Returns the effective gravity for this region.
    pub fn effective_gravity(&self) -> Vec3 {
        self.gravity.unwrap_or(Vec3::GRAVITY)
    }

    /// Computes buoyancy force for a body.
    pub fn compute_buoyancy(&self, body_position: Vec3, body_volume: f32) -> Vec3 {
        if !self.buoyancy {
            return Vec3::ZERO;
        }

        // Fraction of body submerged.
        let submersion = if body_position.y < self.water_surface_y {
            1.0
        } else {
            0.0
        };

        if submersion <= 0.0 {
            return Vec3::ZERO;
        }

        // Buoyancy = density * volume * gravity (upward).
        let buoyancy_magnitude = self.air_density * body_volume * 9.81 * submersion;
        Vec3::UP.scale(buoyancy_magnitude)
    }

    /// Computes drag force for a body.
    pub fn compute_drag(&self, velocity: Vec3, cross_section: f32) -> Vec3 {
        let speed = velocity.length();
        if speed < 0.001 {
            return Vec3::ZERO;
        }

        let cd = self.drag_coefficient.unwrap_or(0.47);
        let force_magnitude = 0.5 * self.air_density * speed * speed * cd * cross_section;

        // Drag opposes velocity.
        velocity.normalized().scale(-force_magnitude)
    }
}

// ---------------------------------------------------------------------------
// Physics LOD
// ---------------------------------------------------------------------------

/// Physics simulation level of detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysicsLOD {
    /// Full simulation (rigid body dynamics, collision response).
    Full,
    /// Reduced simulation (simplified collision, lower update rate).
    Reduced,
    /// Minimal simulation (AABB-only collision, very low update rate).
    Minimal,
    /// Sleeping -- no simulation, wakes on proximity or event.
    Sleeping,
    /// Proxy -- replaced by a lightweight proxy object.
    Proxy,
}

impl PhysicsLOD {
    /// Returns the update rate divisor (every N frames).
    pub fn update_divisor(&self) -> u32 {
        match self {
            Self::Full => 1,
            Self::Reduced => 2,
            Self::Minimal => 8,
            Self::Sleeping => 0,
            Self::Proxy => 0,
        }
    }

    /// Whether this LOD level performs any simulation.
    pub fn is_simulated(&self) -> bool {
        matches!(self, Self::Full | Self::Reduced | Self::Minimal)
    }

    /// Whether this LOD uses full collision detection.
    pub fn uses_full_collision(&self) -> bool {
        matches!(self, Self::Full)
    }
}

/// Configuration for distance-based physics LOD.
#[derive(Debug, Clone)]
pub struct PhysicsLODConfig {
    /// Distance thresholds for each LOD level.
    pub full_distance: f32,
    pub reduced_distance: f32,
    pub minimal_distance: f32,
    pub sleep_distance: f32,
    /// Whether to use LOD for static bodies.
    pub lod_statics: bool,
    /// Hysteresis factor to prevent LOD thrashing.
    pub hysteresis: f32,
}

impl PhysicsLODConfig {
    /// Default LOD configuration.
    pub fn default_config() -> Self {
        Self {
            full_distance: 50.0,
            reduced_distance: 100.0,
            minimal_distance: 200.0,
            sleep_distance: 400.0,
            lod_statics: false,
            hysteresis: 5.0,
        }
    }

    /// Determines the LOD level for a given distance.
    pub fn lod_for_distance(&self, distance: f32) -> PhysicsLOD {
        if distance < self.full_distance {
            PhysicsLOD::Full
        } else if distance < self.reduced_distance {
            PhysicsLOD::Reduced
        } else if distance < self.minimal_distance {
            PhysicsLOD::Minimal
        } else if distance < self.sleep_distance {
            PhysicsLOD::Sleeping
        } else {
            PhysicsLOD::Proxy
        }
    }

    /// Determines the LOD level with hysteresis.
    pub fn lod_for_distance_with_hysteresis(
        &self,
        distance: f32,
        current_lod: PhysicsLOD,
    ) -> PhysicsLOD {
        let target = self.lod_for_distance(distance);

        // Only change if the distance is past threshold + hysteresis.
        if target as u8 > current_lod as u8 {
            // Getting farther -- apply hysteresis.
            let threshold_with_hysteresis = match current_lod {
                PhysicsLOD::Full => self.full_distance + self.hysteresis,
                PhysicsLOD::Reduced => self.reduced_distance + self.hysteresis,
                PhysicsLOD::Minimal => self.minimal_distance + self.hysteresis,
                PhysicsLOD::Sleeping => self.sleep_distance + self.hysteresis,
                PhysicsLOD::Proxy => f32::MAX,
            };
            if distance > threshold_with_hysteresis {
                target
            } else {
                current_lod
            }
        } else {
            target
        }
    }
}

impl Default for PhysicsLODConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

// ---------------------------------------------------------------------------
// Physics body descriptor
// ---------------------------------------------------------------------------

/// Type of physics body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyType {
    /// Dynamic body -- fully simulated.
    Dynamic,
    /// Static body -- never moves.
    Static,
    /// Kinematic body -- moved by code, collides with dynamic.
    Kinematic,
    /// Trigger -- detects overlap, no collision response.
    Trigger,
}

/// A physics body tracked by the world physics system.
#[derive(Debug, Clone)]
pub struct PhysicsBody {
    /// Unique body id.
    pub id: PhysicsBodyId,
    /// Body type.
    pub body_type: BodyType,
    /// Current position.
    pub position: Vec3,
    /// Current linear velocity.
    pub velocity: Vec3,
    /// Angular velocity.
    pub angular_velocity: Vec3,
    /// Mass (kg).
    pub mass: f32,
    /// Bounding box.
    pub bounds: AABB,
    /// Volume (for buoyancy).
    pub volume: f32,
    /// Cross-section area (for drag).
    pub cross_section: f32,
    /// Current physics LOD.
    pub lod: PhysicsLOD,
    /// The world cell this body is in.
    pub cell: CellCoord,
    /// Whether the body is active.
    pub active: bool,
    /// Whether the body is sleeping (no movement for a while).
    pub sleeping: bool,
    /// Sleep timer (seconds of inactivity).
    pub sleep_timer: f32,
    /// Entity id this body is associated with.
    pub entity_id: Option<u64>,
    /// The region this body is currently in.
    pub current_region: Option<RegionId>,
}

impl PhysicsBody {
    /// Creates a new dynamic body.
    pub fn dynamic(position: Vec3, mass: f32) -> Self {
        let half = Vec3::new(0.5, 0.5, 0.5);
        Self {
            id: PhysicsBodyId::new(),
            body_type: BodyType::Dynamic,
            position,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass,
            bounds: AABB::new(position.sub(&half), position.add(&half)),
            volume: 1.0,
            cross_section: 1.0,
            lod: PhysicsLOD::Full,
            cell: CellCoord::new(0, 0, 0),
            active: true,
            sleeping: false,
            sleep_timer: 0.0,
            entity_id: None,
            current_region: None,
        }
    }

    /// Creates a new static body.
    pub fn static_body(position: Vec3, bounds: AABB) -> Self {
        Self {
            id: PhysicsBodyId::new(),
            body_type: BodyType::Static,
            position,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 0.0,
            bounds,
            volume: bounds.volume(),
            cross_section: 0.0,
            lod: PhysicsLOD::Full,
            cell: CellCoord::new(0, 0, 0),
            active: true,
            sleeping: true,
            sleep_timer: 0.0,
            entity_id: None,
            current_region: None,
        }
    }

    /// Creates a kinematic body.
    pub fn kinematic(position: Vec3) -> Self {
        let half = Vec3::new(0.5, 0.5, 0.5);
        Self {
            body_type: BodyType::Kinematic,
            ..Self::dynamic(position, 0.0)
        }
    }

    /// Returns the kinetic energy of the body.
    pub fn kinetic_energy(&self) -> f32 {
        0.5 * self.mass * self.velocity.length_squared()
    }

    /// Whether the body should be put to sleep.
    pub fn should_sleep(&self, threshold: f32, time_threshold: f32) -> bool {
        self.velocity.length_squared() < threshold * threshold
            && self.sleep_timer >= time_threshold
    }

    /// Updates the sleep timer.
    pub fn update_sleep(&mut self, dt: f32, velocity_threshold: f32) {
        if self.velocity.length_squared() < velocity_threshold * velocity_threshold {
            self.sleep_timer += dt;
        } else {
            self.sleep_timer = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Physics proxy
// ---------------------------------------------------------------------------

/// A lightweight proxy that replaces a distant physics body.
///
/// Proxies store minimal state and can be reconstituted into full bodies
/// when the player gets close enough.
#[derive(Debug, Clone)]
pub struct PhysicsProxy {
    /// The original body id.
    pub body_id: PhysicsBodyId,
    /// Last known position.
    pub position: Vec3,
    /// Last known velocity.
    pub velocity: Vec3,
    /// Bounding box.
    pub bounds: AABB,
    /// Body type.
    pub body_type: BodyType,
    /// Mass.
    pub mass: f32,
    /// The cell this proxy is in.
    pub cell: CellCoord,
    /// Entity id.
    pub entity_id: Option<u64>,
    /// Time since the body was proxied (for interpolation).
    pub time_since_proxy: f32,
}

impl PhysicsProxy {
    /// Creates a proxy from a full body.
    pub fn from_body(body: &PhysicsBody) -> Self {
        Self {
            body_id: body.id,
            position: body.position,
            velocity: body.velocity,
            bounds: body.bounds,
            body_type: body.body_type,
            mass: body.mass,
            cell: body.cell,
            entity_id: body.entity_id,
            time_since_proxy: 0.0,
        }
    }

    /// Extrapolates the proxy position forward in time.
    pub fn extrapolated_position(&self) -> Vec3 {
        self.position.add(&self.velocity.scale(self.time_since_proxy))
    }

    /// Reconstitutes into a full body.
    pub fn to_body(&self) -> PhysicsBody {
        PhysicsBody {
            id: self.body_id,
            body_type: self.body_type,
            position: self.extrapolated_position(),
            velocity: self.velocity,
            angular_velocity: Vec3::ZERO,
            mass: self.mass,
            bounds: self.bounds,
            volume: self.bounds.volume(),
            cross_section: 1.0,
            lod: PhysicsLOD::Full,
            cell: self.cell,
            active: true,
            sleeping: false,
            sleep_timer: 0.0,
            entity_id: self.entity_id,
            current_region: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Physics cell
// ---------------------------------------------------------------------------

/// A physics cell containing bodies for a section of the world.
#[derive(Debug)]
pub struct PhysicsCell {
    /// Cell coordinate.
    pub coord: CellCoord,
    /// Bodies in this cell.
    pub bodies: Vec<PhysicsBodyId>,
    /// Proxies in this cell.
    pub proxies: Vec<PhysicsProxy>,
    /// Whether this cell is loaded (active simulation).
    pub loaded: bool,
    /// Memory estimate for this cell.
    pub memory_bytes: usize,
}

impl PhysicsCell {
    pub fn new(coord: CellCoord) -> Self {
        Self {
            coord,
            bodies: Vec::new(),
            proxies: Vec::new(),
            loaded: false,
            memory_bytes: 0,
        }
    }

    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    pub fn proxy_count(&self) -> usize {
        self.proxies.len()
    }
}

// ---------------------------------------------------------------------------
// World physics statistics
// ---------------------------------------------------------------------------

/// Runtime statistics for the world physics system.
#[derive(Debug, Clone, Default)]
pub struct WorldPhysicsStats {
    pub total_bodies: usize,
    pub active_bodies: usize,
    pub sleeping_bodies: usize,
    pub proxy_bodies: usize,
    pub loaded_cells: usize,
    pub total_cells: usize,
    pub regions: usize,
    pub lod_full: usize,
    pub lod_reduced: usize,
    pub lod_minimal: usize,
    pub lod_sleeping: usize,
    pub lod_proxy: usize,
}

// ---------------------------------------------------------------------------
// World physics manager
// ---------------------------------------------------------------------------

/// The world-scale physics manager.
///
/// Coordinates physics regions, LOD management, cell streaming, and proxy
/// conversion for a large open world.
pub struct WorldPhysicsManager {
    /// All physics bodies keyed by id.
    bodies: HashMap<PhysicsBodyId, PhysicsBody>,
    /// Physics regions.
    regions: Vec<PhysicsRegion>,
    /// Physics cells.
    cells: HashMap<CellCoord, PhysicsCell>,
    /// Proxied bodies.
    proxies: HashMap<PhysicsBodyId, PhysicsProxy>,
    /// LOD configuration.
    lod_config: PhysicsLODConfig,
    /// World cell size.
    cell_size: f32,
    /// Load radius (in cells).
    load_radius: u32,
    /// Unload radius (in cells).
    unload_radius: u32,
    /// Current viewer position.
    viewer_position: Vec3,
    /// Current viewer cell.
    viewer_cell: CellCoord,
    /// Default gravity.
    default_gravity: Vec3,
    /// Current frame.
    frame: u64,
    /// Sleep velocity threshold.
    sleep_velocity_threshold: f32,
    /// Sleep time threshold.
    sleep_time_threshold: f32,
}

impl WorldPhysicsManager {
    /// Creates a new world physics manager.
    pub fn new(cell_size: f32, load_radius: u32, unload_radius: u32) -> Self {
        Self {
            bodies: HashMap::new(),
            regions: Vec::new(),
            cells: HashMap::new(),
            proxies: HashMap::new(),
            lod_config: PhysicsLODConfig::default(),
            cell_size,
            load_radius,
            unload_radius,
            viewer_position: Vec3::ZERO,
            viewer_cell: CellCoord::new(0, 0, 0),
            default_gravity: Vec3::GRAVITY,
            frame: 0,
            sleep_velocity_threshold: 0.05,
            sleep_time_threshold: 2.0,
        }
    }

    // -- Body management ------------------------------------------------------

    /// Adds a body to the world.
    pub fn add_body(&mut self, mut body: PhysicsBody) -> PhysicsBodyId {
        let id = body.id;
        body.cell = self.world_to_cell(body.position);
        self.bodies.insert(id, body);

        // Add to cell.
        let cell = self.bodies[&id].cell;
        self.cells.entry(cell).or_insert_with(|| PhysicsCell::new(cell));
        self.cells.get_mut(&cell).unwrap().bodies.push(id);

        id
    }

    /// Removes a body from the world.
    pub fn remove_body(&mut self, id: PhysicsBodyId) {
        if let Some(body) = self.bodies.remove(&id) {
            if let Some(cell) = self.cells.get_mut(&body.cell) {
                cell.bodies.retain(|&b| b != id);
            }
        }
        self.proxies.remove(&id);
    }

    /// Returns a body by id.
    pub fn get_body(&self, id: PhysicsBodyId) -> Option<&PhysicsBody> {
        self.bodies.get(&id)
    }

    /// Returns a mutable body by id.
    pub fn get_body_mut(&mut self, id: PhysicsBodyId) -> Option<&mut PhysicsBody> {
        self.bodies.get_mut(&id)
    }

    /// Returns all body ids.
    pub fn body_ids(&self) -> Vec<PhysicsBodyId> {
        self.bodies.keys().copied().collect()
    }

    /// Returns the total number of bodies.
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    // -- Region management ----------------------------------------------------

    /// Adds a physics region.
    pub fn add_region(&mut self, region: PhysicsRegion) -> RegionId {
        let id = region.id;
        self.regions.push(region);
        id
    }

    /// Removes a region.
    pub fn remove_region(&mut self, id: RegionId) {
        self.regions.retain(|r| r.id != id);
    }

    /// Returns a region by id.
    pub fn get_region(&self, id: RegionId) -> Option<&PhysicsRegion> {
        self.regions.iter().find(|r| r.id == id)
    }

    /// Returns the region count.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Finds the highest-priority active region containing a point.
    pub fn region_at_point(&self, point: Vec3) -> Option<&PhysicsRegion> {
        self.regions
            .iter()
            .filter(|r| r.enabled && r.shape.contains_point(point))
            .max_by_key(|r| r.priority)
    }

    /// Returns the effective gravity at a point.
    pub fn gravity_at(&self, point: Vec3) -> Vec3 {
        match self.region_at_point(point) {
            Some(region) => region.effective_gravity(),
            None => self.default_gravity,
        }
    }

    /// Returns the effective time scale at a point.
    pub fn time_scale_at(&self, point: Vec3) -> f32 {
        match self.region_at_point(point) {
            Some(region) => region.time_scale,
            None => 1.0,
        }
    }

    // -- LOD management -------------------------------------------------------

    /// Sets the LOD configuration.
    pub fn set_lod_config(&mut self, config: PhysicsLODConfig) {
        self.lod_config = config;
    }

    /// Updates LOD levels for all bodies based on distance from viewer.
    fn update_body_lods(&mut self) {
        let viewer = self.viewer_position;
        let config = self.lod_config.clone();

        for body in self.bodies.values_mut() {
            if body.body_type == BodyType::Static && !config.lod_statics {
                continue;
            }

            let distance = body.position.distance(&viewer);
            let new_lod = config.lod_for_distance_with_hysteresis(distance, body.lod);
            body.lod = new_lod;
        }
    }

    // -- Cell streaming -------------------------------------------------------

    /// Converts a world position to a cell coordinate.
    pub fn world_to_cell(&self, pos: Vec3) -> CellCoord {
        CellCoord {
            x: (pos.x / self.cell_size).floor() as i32,
            y: (pos.y / self.cell_size).floor() as i32,
            z: (pos.z / self.cell_size).floor() as i32,
        }
    }

    /// Loads a physics cell -- converts proxies back to full bodies.
    pub fn load_cell(&mut self, coord: CellCoord) {
        if let Some(cell) = self.cells.get_mut(&coord) {
            if cell.loaded {
                return;
            }
            cell.loaded = true;

            // Convert proxies to full bodies.
            let proxies: Vec<PhysicsProxy> = cell.proxies.drain(..).collect();
            for proxy in proxies {
                let body = proxy.to_body();
                let id = body.id;
                cell.bodies.push(id);
                self.bodies.insert(id, body);
                self.proxies.remove(&id);
            }
        }
    }

    /// Unloads a physics cell -- converts bodies to proxies.
    pub fn unload_cell(&mut self, coord: CellCoord) {
        if let Some(cell) = self.cells.get_mut(&coord) {
            if !cell.loaded {
                return;
            }
            cell.loaded = false;

            let body_ids: Vec<PhysicsBodyId> = cell.bodies.drain(..).collect();
            for id in body_ids {
                if let Some(body) = self.bodies.remove(&id) {
                    if body.body_type == BodyType::Dynamic {
                        let proxy = PhysicsProxy::from_body(&body);
                        cell.proxies.push(proxy.clone());
                        self.proxies.insert(id, proxy);
                    }
                }
            }
        }
    }

    /// Streams cells based on viewer position.
    fn stream_cells(&mut self) {
        let viewer_cell = self.viewer_cell;

        // Load cells within radius.
        let load_r = self.load_radius as i32;
        for dz in -load_r..=load_r {
            for dy in -1..=1 {
                for dx in -load_r..=load_r {
                    let coord = CellCoord::new(
                        viewer_cell.x + dx,
                        viewer_cell.y + dy,
                        viewer_cell.z + dz,
                    );
                    let dist = coord.chebyshev_distance(&viewer_cell);
                    if dist <= self.load_radius {
                        self.cells.entry(coord).or_insert_with(|| {
                            let mut cell = PhysicsCell::new(coord);
                            cell.loaded = true;
                            cell
                        });
                        self.load_cell(coord);
                    }
                }
            }
        }

        // Unload cells beyond unload radius.
        let to_unload: Vec<CellCoord> = self
            .cells
            .keys()
            .filter(|&&coord| coord.chebyshev_distance(&viewer_cell) > self.unload_radius)
            .copied()
            .collect();

        for coord in to_unload {
            self.unload_cell(coord);
        }
    }

    // -- Per-frame update -----------------------------------------------------

    /// Main per-frame update.
    pub fn update(&mut self, dt: f32, viewer_position: Vec3) {
        self.frame += 1;
        self.viewer_position = viewer_position;
        self.viewer_cell = self.world_to_cell(viewer_position);

        // 1. Stream cells.
        self.stream_cells();

        // 2. Update body LODs.
        self.update_body_lods();

        // 3. Update region assignments.
        self.update_region_assignments();

        // 4. Simulate active bodies.
        self.simulate_bodies(dt);

        // 5. Update proxy times.
        for proxy in self.proxies.values_mut() {
            proxy.time_since_proxy += dt;
        }
    }

    /// Updates which region each body is in.
    fn update_region_assignments(&mut self) {
        let regions = &self.regions;
        for body in self.bodies.values_mut() {
            body.current_region = regions
                .iter()
                .filter(|r| r.enabled && r.shape.contains_point(body.position))
                .max_by_key(|r| r.priority)
                .map(|r| r.id);
        }
    }

    /// Simulates all active bodies.
    fn simulate_bodies(&mut self, dt: f32) {
        let default_gravity = self.default_gravity;
        let frame = self.frame;
        let sleep_vel = self.sleep_velocity_threshold;
        let sleep_time = self.sleep_time_threshold;

        // Collect region data for applying forces.
        let regions: HashMap<RegionId, PhysicsRegion> = self
            .regions
            .iter()
            .map(|r| (r.id, r.clone()))
            .collect();

        for body in self.bodies.values_mut() {
            if body.body_type != BodyType::Dynamic || !body.active {
                continue;
            }
            if body.sleeping {
                continue;
            }

            // Check LOD update rate.
            let divisor = body.lod.update_divisor();
            if divisor == 0 {
                continue;
            }
            if frame % divisor as u64 != 0 {
                continue;
            }

            let effective_dt = dt * divisor as f32;

            // Get region parameters.
            let (gravity, time_scale, damping) = if let Some(region_id) = body.current_region {
                if let Some(region) = regions.get(&region_id) {
                    (
                        region.effective_gravity(),
                        region.time_scale,
                        region.linear_damping,
                    )
                } else {
                    (default_gravity, 1.0, 0.01)
                }
            } else {
                (default_gravity, 1.0, 0.01)
            };

            let sim_dt = effective_dt * time_scale;

            // Apply gravity.
            if body.mass > 0.0 {
                let gravity_impulse = gravity.scale(sim_dt);
                body.velocity = body.velocity.add(&gravity_impulse);
            }

            // Apply damping.
            body.velocity = body.velocity.scale(1.0 - damping * sim_dt);

            // Integrate position.
            let displacement = body.velocity.scale(sim_dt);
            body.position = body.position.add(&displacement);

            // Update bounds.
            let half = body.bounds.extents();
            body.bounds = AABB::new(
                body.position.sub(&half),
                body.position.add(&half),
            );

            // Update cell assignment.
            let new_cell = CellCoord::new(
                (body.position.x / self.cell_size).floor() as i32,
                (body.position.y / self.cell_size).floor() as i32,
                (body.position.z / self.cell_size).floor() as i32,
            );
            body.cell = new_cell;

            // Sleep check.
            body.update_sleep(sim_dt, sleep_vel);
            if body.should_sleep(sleep_vel, sleep_time) {
                body.sleeping = true;
            }
        }
    }

    /// Wakes a sleeping body.
    pub fn wake_body(&mut self, id: PhysicsBodyId) {
        if let Some(body) = self.bodies.get_mut(&id) {
            body.sleeping = false;
            body.sleep_timer = 0.0;
        }
    }

    /// Wakes all bodies in a radius.
    pub fn wake_bodies_in_radius(&mut self, center: Vec3, radius: f32) {
        let radius_sq = radius * radius;
        for body in self.bodies.values_mut() {
            if body.position.distance_squared(&center) < radius_sq {
                body.sleeping = false;
                body.sleep_timer = 0.0;
            }
        }
    }

    // -- Statistics -----------------------------------------------------------

    /// Computes current statistics.
    pub fn stats(&self) -> WorldPhysicsStats {
        let mut stats = WorldPhysicsStats::default();
        stats.total_bodies = self.bodies.len();
        stats.proxy_bodies = self.proxies.len();
        stats.loaded_cells = self.cells.values().filter(|c| c.loaded).count();
        stats.total_cells = self.cells.len();
        stats.regions = self.regions.len();

        for body in self.bodies.values() {
            if body.sleeping {
                stats.sleeping_bodies += 1;
            } else if body.active {
                stats.active_bodies += 1;
            }

            match body.lod {
                PhysicsLOD::Full => stats.lod_full += 1,
                PhysicsLOD::Reduced => stats.lod_reduced += 1,
                PhysicsLOD::Minimal => stats.lod_minimal += 1,
                PhysicsLOD::Sleeping => stats.lod_sleeping += 1,
                PhysicsLOD::Proxy => stats.lod_proxy += 1,
            }
        }

        stats
    }

    /// Sets the default gravity.
    pub fn set_default_gravity(&mut self, gravity: Vec3) {
        self.default_gravity = gravity;
    }
}

impl Default for WorldPhysicsManager {
    fn default() -> Self {
        Self::new(64.0, 4, 6)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_shape_box_contains() {
        let shape = RegionShape::Box(AABB::new(
            Vec3::new(-10.0, -10.0, -10.0),
            Vec3::new(10.0, 10.0, 10.0),
        ));
        assert!(shape.contains_point(Vec3::ZERO));
        assert!(!shape.contains_point(Vec3::new(20.0, 0.0, 0.0)));
    }

    #[test]
    fn region_shape_sphere() {
        let shape = RegionShape::Sphere {
            center: Vec3::ZERO,
            radius: 5.0,
        };
        assert!(shape.contains_point(Vec3::new(3.0, 0.0, 0.0)));
        assert!(!shape.contains_point(Vec3::new(6.0, 0.0, 0.0)));
    }

    #[test]
    fn region_gravity() {
        let region = PhysicsRegion::low_gravity(
            "Moon",
            RegionShape::Global,
            0.16,
        );
        let g = region.effective_gravity();
        assert!(g.y.abs() < 2.0);
    }

    #[test]
    fn region_buoyancy() {
        let region = PhysicsRegion::underwater(
            "Lake",
            RegionShape::Box(AABB::new(Vec3::ZERO, Vec3::new(100.0, 50.0, 100.0))),
            50.0,
        );
        let force = region.compute_buoyancy(Vec3::new(50.0, 25.0, 50.0), 1.0);
        assert!(force.y > 0.0); // Should push up.
    }

    #[test]
    fn physics_lod_selection() {
        let config = PhysicsLODConfig::default_config();
        assert_eq!(config.lod_for_distance(10.0), PhysicsLOD::Full);
        assert_eq!(config.lod_for_distance(75.0), PhysicsLOD::Reduced);
        assert_eq!(config.lod_for_distance(150.0), PhysicsLOD::Minimal);
        assert_eq!(config.lod_for_distance(300.0), PhysicsLOD::Sleeping);
        assert_eq!(config.lod_for_distance(500.0), PhysicsLOD::Proxy);
    }

    #[test]
    fn body_creation() {
        let body = PhysicsBody::dynamic(Vec3::new(10.0, 5.0, 10.0), 50.0);
        assert_eq!(body.body_type, BodyType::Dynamic);
        assert_eq!(body.mass, 50.0);
    }

    #[test]
    fn body_sleep() {
        let mut body = PhysicsBody::dynamic(Vec3::ZERO, 1.0);
        body.velocity = Vec3::new(0.001, 0.0, 0.0);
        body.update_sleep(1.0, 0.05);
        body.update_sleep(1.0, 0.05);
        body.update_sleep(1.0, 0.05);
        assert!(body.should_sleep(0.05, 2.0));
    }

    #[test]
    fn proxy_round_trip() {
        let mut body = PhysicsBody::dynamic(Vec3::new(10.0, 5.0, 10.0), 50.0);
        body.velocity = Vec3::new(1.0, 0.0, 0.0);
        body.entity_id = Some(42);

        let proxy = PhysicsProxy::from_body(&body);
        assert_eq!(proxy.body_id, body.id);

        let reconstituted = proxy.to_body();
        assert_eq!(reconstituted.id, body.id);
        assert_eq!(reconstituted.entity_id, Some(42));
    }

    #[test]
    fn world_physics_add_remove() {
        let mut mgr = WorldPhysicsManager::new(64.0, 2, 3);
        let body = PhysicsBody::dynamic(Vec3::new(10.0, 0.0, 10.0), 1.0);
        let id = mgr.add_body(body);
        assert_eq!(mgr.body_count(), 1);

        mgr.remove_body(id);
        assert_eq!(mgr.body_count(), 0);
    }

    #[test]
    fn world_physics_regions() {
        let mut mgr = WorldPhysicsManager::new(64.0, 2, 3);

        let region = PhysicsRegion::low_gravity(
            "Low-G Zone",
            RegionShape::Sphere {
                center: Vec3::new(50.0, 0.0, 50.0),
                radius: 20.0,
            },
            0.3,
        );
        mgr.add_region(region);

        let g = mgr.gravity_at(Vec3::new(50.0, 0.0, 50.0));
        assert!(g.y.abs() < 5.0);

        let g_outside = mgr.gravity_at(Vec3::new(200.0, 0.0, 200.0));
        assert!((g_outside.y - Vec3::GRAVITY.y).abs() < 0.01);
    }

    #[test]
    fn world_physics_update() {
        let mut mgr = WorldPhysicsManager::new(64.0, 2, 3);
        let body = PhysicsBody::dynamic(Vec3::new(0.0, 10.0, 0.0), 1.0);
        let id = mgr.add_body(body);

        mgr.update(0.016, Vec3::ZERO);

        // Body should have fallen due to gravity.
        let body = mgr.get_body(id).unwrap();
        assert!(body.velocity.y < 0.0);
    }

    #[test]
    fn world_to_cell() {
        let mgr = WorldPhysicsManager::new(64.0, 2, 3);
        let cell = mgr.world_to_cell(Vec3::new(100.0, 0.0, 200.0));
        assert_eq!(cell.x, 1);
        assert_eq!(cell.z, 3);
    }

    #[test]
    fn wake_bodies() {
        let mut mgr = WorldPhysicsManager::new(64.0, 2, 3);
        let body = PhysicsBody::dynamic(Vec3::new(10.0, 0.0, 10.0), 1.0);
        let id = mgr.add_body(body);

        // Put to sleep.
        mgr.get_body_mut(id).unwrap().sleeping = true;
        assert!(mgr.get_body(id).unwrap().sleeping);

        // Wake.
        mgr.wake_body(id);
        assert!(!mgr.get_body(id).unwrap().sleeping);
    }

    #[test]
    fn stats() {
        let mut mgr = WorldPhysicsManager::new(64.0, 2, 3);
        mgr.add_body(PhysicsBody::dynamic(Vec3::ZERO, 1.0));
        mgr.add_body(PhysicsBody::static_body(
            Vec3::new(10.0, 0.0, 10.0),
            AABB::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0)),
        ));

        let stats = mgr.stats();
        assert_eq!(stats.total_bodies, 2);
    }

    #[test]
    fn time_scale_region() {
        let mut mgr = WorldPhysicsManager::new(64.0, 2, 3);
        let region = PhysicsRegion::time_bubble(
            "Slow Zone",
            RegionShape::Sphere { center: Vec3::ZERO, radius: 10.0 },
            0.5,
        );
        mgr.add_region(region);

        assert!((mgr.time_scale_at(Vec3::ZERO) - 0.5).abs() < 0.01);
        assert!((mgr.time_scale_at(Vec3::new(100.0, 0.0, 0.0)) - 1.0).abs() < 0.01);
    }
}
