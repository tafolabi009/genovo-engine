// engine/physics/src/physics_world_v2.rs
//
// Enhanced physics world for the Genovo physics engine.
//
// Provides sub-worlds for different regions, physics islands, sleeping
// island optimization, broad-phase switching (SAP/grid/BVH), narrow-phase
// cache, and constraint groups.
//
// # Architecture
//
// `PhysicsWorldV2` wraps a collection of `PhysicsIsland`s. Each island is a
// connected sub-graph of bodies linked by contacts or constraints. Islands
// that have been still for a configurable time are put to sleep, skipping
// simulation entirely. Broad-phase algorithms can be switched at runtime
// depending on scene characteristics.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Math types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const GRAVITY: Self = Self { x: 0.0, y: -9.81, z: 0.0 };

    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    #[inline]
    pub fn add(self, o: Self) -> Self { Self::new(self.x + o.x, self.y + o.y, self.z + o.z) }
    #[inline]
    pub fn sub(self, o: Self) -> Self { Self::new(self.x - o.x, self.y - o.y, self.z - o.z) }
    #[inline]
    pub fn scale(self, s: f32) -> Self { Self::new(self.x * s, self.y * s, self.z * s) }
    #[inline]
    pub fn dot(self, o: Self) -> f32 { self.x * o.x + self.y * o.y + self.z * o.z }
    #[inline]
    pub fn length(self) -> f32 { self.dot(self).sqrt() }
    #[inline]
    pub fn length_sq(self) -> f32 { self.dot(self) }
    #[inline]
    pub fn cross(self, o: Self) -> Self {
        Self::new(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)
    }
    #[inline]
    pub fn normalize(self) -> Self {
        let l = self.length();
        if l > 1e-7 { self.scale(1.0 / l) } else { Self::ZERO }
    }
}

/// Quaternion for orientation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    pub const IDENTITY: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let half = angle * 0.5;
        let s = half.sin();
        let a = axis.normalize();
        Self { x: a.x * s, y: a.y * s, z: a.z * s, w: half.cos() }
    }

    pub fn normalize(self) -> Self {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt();
        if len > 1e-7 {
            let inv = 1.0 / len;
            Self { x: self.x * inv, y: self.y * inv, z: self.z * inv, w: self.w * inv }
        } else {
            Self::IDENTITY
        }
    }

    pub fn rotate_vec(&self, v: Vec3) -> Vec3 {
        let u = Vec3::new(self.x, self.y, self.z);
        let s = self.w;
        let dot_uv = u.dot(v);
        let dot_uu = u.dot(u);
        let cross_uv = u.cross(v);
        v.scale(s * s - dot_uu).add(u.scale(2.0 * dot_uv)).add(cross_uv.scale(2.0 * s))
    }
}

/// AABB.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }

    pub fn from_center_half(center: Vec3, half: Vec3) -> Self {
        Self { min: center.sub(half), max: center.add(half) }
    }

    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
            && self.min.y <= other.max.y && self.max.y >= other.min.y
            && self.min.z <= other.max.z && self.max.z >= other.min.z
    }

    pub fn expand(&mut self, p: Vec3) {
        self.min.x = self.min.x.min(p.x);
        self.min.y = self.min.y.min(p.y);
        self.min.z = self.min.z.min(p.z);
        self.max.x = self.max.x.max(p.x);
        self.max.y = self.max.y.max(p.y);
        self.max.z = self.max.z.max(p.z);
    }

    pub fn center(&self) -> Vec3 {
        Vec3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    pub fn volume(&self) -> f32 {
        let d = self.max.sub(self.min);
        d.x * d.y * d.z
    }

    pub fn surface_area(&self) -> f32 {
        let d = self.max.sub(self.min);
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    pub fn merge(&self, other: &Self) -> Self {
        Self {
            min: Vec3::new(self.min.x.min(other.min.x), self.min.y.min(other.min.y), self.min.z.min(other.min.z)),
            max: Vec3::new(self.max.x.max(other.max.x), self.max.y.max(other.max.y), self.max.z.max(other.max.z)),
        }
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub type BodyHandle = u64;
pub type ConstraintHandle = u64;
pub type IslandId = u32;
pub type SubWorldId = u32;

/// Default sleep velocity threshold (m/s).
pub const SLEEP_VELOCITY_THRESHOLD: f32 = 0.05;
/// Default sleep angular velocity threshold (rad/s).
pub const SLEEP_ANGULAR_THRESHOLD: f32 = 0.05;
/// Default frames before a body sleeps.
pub const SLEEP_FRAMES_THRESHOLD: u32 = 60;
/// Default solver iterations.
pub const DEFAULT_SOLVER_ITERATIONS: u32 = 8;
/// Default solver position iterations.
pub const DEFAULT_POSITION_ITERATIONS: u32 = 3;
/// Maximum sub-worlds.
pub const MAX_SUB_WORLDS: usize = 16;
/// Maximum bodies per island before splitting consideration.
pub const MAX_ISLAND_BODIES: usize = 512;
/// Narrow-phase cache capacity.
pub const NARROW_PHASE_CACHE_CAPACITY: usize = 16384;

// ---------------------------------------------------------------------------
// Broad-phase algorithm selection
// ---------------------------------------------------------------------------

/// Available broad-phase algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BroadPhaseAlgorithm {
    /// Sweep-and-Prune (good for mostly-static scenes).
    SweepAndPrune,
    /// Uniform grid (good for uniform density).
    UniformGrid,
    /// Bounding Volume Hierarchy (good for dynamic scenes).
    Bvh,
    /// Multi-SAP (spatial hashing with SAP buckets).
    MultiSap,
}

impl fmt::Display for BroadPhaseAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SweepAndPrune => write!(f, "SAP"),
            Self::UniformGrid => write!(f, "Grid"),
            Self::Bvh => write!(f, "BVH"),
            Self::MultiSap => write!(f, "Multi-SAP"),
        }
    }
}

/// Broad-phase configuration.
#[derive(Debug, Clone)]
pub struct BroadPhaseConfig {
    pub algorithm: BroadPhaseAlgorithm,
    /// Grid cell size (for UniformGrid).
    pub grid_cell_size: f32,
    /// BVH refit frequency (frames between full rebuilds).
    pub bvh_refit_interval: u32,
    /// SAP axis to sort on (0=X, 1=Y, 2=Z).
    pub sap_primary_axis: u32,
    /// Whether to auto-switch algorithm based on scene characteristics.
    pub auto_switch: bool,
    /// Body count threshold for switching from Grid to BVH.
    pub bvh_threshold: usize,
}

impl Default for BroadPhaseConfig {
    fn default() -> Self {
        Self {
            algorithm: BroadPhaseAlgorithm::SweepAndPrune,
            grid_cell_size: 2.0,
            bvh_refit_interval: 30,
            sap_primary_axis: 0,
            auto_switch: true,
            bvh_threshold: 1000,
        }
    }
}

// ---------------------------------------------------------------------------
// Body state
// ---------------------------------------------------------------------------

/// Body type in the physics world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyType {
    Dynamic,
    Kinematic,
    Static,
}

/// Sleep state of a body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SleepState {
    Awake,
    Sleeping,
    /// Will go to sleep after a few more quiet frames.
    WantsSleep,
}

/// Physical body in the world.
#[derive(Debug, Clone)]
pub struct RigidBodyState {
    pub handle: BodyHandle,
    pub body_type: BodyType,
    pub position: Vec3,
    pub orientation: Quat,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub mass: f32,
    pub inv_mass: f32,
    pub inertia: Vec3,
    pub inv_inertia: Vec3,
    pub friction: f32,
    pub restitution: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub aabb: Aabb,
    pub sleep_state: SleepState,
    pub sleep_timer: u32,
    pub island_id: Option<IslandId>,
    pub sub_world_id: SubWorldId,
    pub collision_layer: u32,
    pub collision_mask: u32,
    pub gravity_scale: f32,
    pub ccd_enabled: bool,
    pub user_data: u64,
}

impl RigidBodyState {
    /// Create a new dynamic body.
    pub fn new_dynamic(handle: BodyHandle, position: Vec3, mass: f32) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self {
            handle,
            body_type: BodyType::Dynamic,
            position,
            orientation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass,
            inv_mass,
            inertia: Vec3::new(mass, mass, mass),
            inv_inertia: Vec3::new(inv_mass, inv_mass, inv_mass),
            friction: 0.5,
            restitution: 0.3,
            linear_damping: 0.01,
            angular_damping: 0.05,
            aabb: Aabb::from_center_half(position, Vec3::new(0.5, 0.5, 0.5)),
            sleep_state: SleepState::Awake,
            sleep_timer: 0,
            island_id: None,
            sub_world_id: 0,
            collision_layer: 1,
            collision_mask: 0xFFFFFFFF,
            gravity_scale: 1.0,
            ccd_enabled: false,
            user_data: 0,
        }
    }

    /// Create a new static body.
    pub fn new_static(handle: BodyHandle, position: Vec3) -> Self {
        Self {
            handle,
            body_type: BodyType::Static,
            position,
            orientation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 0.0,
            inv_mass: 0.0,
            inertia: Vec3::ZERO,
            inv_inertia: Vec3::ZERO,
            friction: 0.5,
            restitution: 0.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
            aabb: Aabb::from_center_half(position, Vec3::new(0.5, 0.5, 0.5)),
            sleep_state: SleepState::Sleeping,
            sleep_timer: u32::MAX,
            island_id: None,
            sub_world_id: 0,
            collision_layer: 1,
            collision_mask: 0xFFFFFFFF,
            gravity_scale: 0.0,
            ccd_enabled: false,
            user_data: 0,
        }
    }

    /// Check if the body is awake.
    pub fn is_awake(&self) -> bool { self.sleep_state == SleepState::Awake }

    /// Check if the body should be considered for sleeping.
    pub fn should_sleep(&self, vel_threshold: f32, ang_threshold: f32, frame_threshold: u32) -> bool {
        if self.body_type != BodyType::Dynamic { return false; }
        self.linear_velocity.length_sq() < vel_threshold * vel_threshold
            && self.angular_velocity.length_sq() < ang_threshold * ang_threshold
            && self.sleep_timer >= frame_threshold
    }

    /// Apply a force at the center of mass.
    pub fn apply_force(&mut self, force: Vec3) {
        if self.body_type != BodyType::Dynamic { return; }
        self.linear_velocity = self.linear_velocity.add(force.scale(self.inv_mass));
        self.wake_up();
    }

    /// Apply an impulse at the center of mass.
    pub fn apply_impulse(&mut self, impulse: Vec3) {
        if self.body_type != BodyType::Dynamic { return; }
        self.linear_velocity = self.linear_velocity.add(impulse.scale(self.inv_mass));
        self.wake_up();
    }

    /// Apply torque.
    pub fn apply_torque(&mut self, torque: Vec3) {
        if self.body_type != BodyType::Dynamic { return; }
        self.angular_velocity = self.angular_velocity.add(Vec3::new(
            torque.x * self.inv_inertia.x,
            torque.y * self.inv_inertia.y,
            torque.z * self.inv_inertia.z,
        ));
        self.wake_up();
    }

    /// Wake up the body.
    pub fn wake_up(&mut self) {
        self.sleep_state = SleepState::Awake;
        self.sleep_timer = 0;
    }

    /// Put the body to sleep.
    pub fn put_to_sleep(&mut self) {
        self.sleep_state = SleepState::Sleeping;
        self.linear_velocity = Vec3::ZERO;
        self.angular_velocity = Vec3::ZERO;
    }
}

// ---------------------------------------------------------------------------
// Constraint
// ---------------------------------------------------------------------------

/// Type of physics constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintType {
    Contact,
    Distance,
    Hinge,
    BallSocket,
    Fixed,
    Slider,
    Spring,
    Motor,
}

/// A physics constraint between two bodies.
#[derive(Debug, Clone)]
pub struct Constraint {
    pub handle: ConstraintHandle,
    pub constraint_type: ConstraintType,
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub axis: Vec3,
    pub stiffness: f32,
    pub damping: f32,
    pub max_force: f32,
    pub error: f32,
    pub group_id: u32,
    pub enabled: bool,
}

impl Constraint {
    pub fn new_contact(handle: ConstraintHandle, a: BodyHandle, b: BodyHandle) -> Self {
        Self {
            handle,
            constraint_type: ConstraintType::Contact,
            body_a: a,
            body_b: b,
            anchor_a: Vec3::ZERO,
            anchor_b: Vec3::ZERO,
            axis: Vec3::new(0.0, 1.0, 0.0),
            stiffness: 0.0,
            damping: 0.0,
            max_force: f32::MAX,
            error: 0.0,
            group_id: 0,
            enabled: true,
        }
    }

    pub fn new_distance(handle: ConstraintHandle, a: BodyHandle, b: BodyHandle, distance: f32) -> Self {
        Self {
            handle,
            constraint_type: ConstraintType::Distance,
            body_a: a,
            body_b: b,
            anchor_a: Vec3::ZERO,
            anchor_b: Vec3::ZERO,
            axis: Vec3::ZERO,
            stiffness: distance,
            damping: 0.1,
            max_force: f32::MAX,
            error: 0.0,
            group_id: 0,
            enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Constraint group
// ---------------------------------------------------------------------------

/// A group of constraints that are solved together.
#[derive(Debug, Clone)]
pub struct ConstraintGroup {
    pub id: u32,
    pub name: String,
    pub constraints: Vec<ConstraintHandle>,
    pub solver_iterations: u32,
    pub enabled: bool,
    pub priority: u32,
}

impl ConstraintGroup {
    pub fn new(id: u32, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            constraints: Vec::new(),
            solver_iterations: DEFAULT_SOLVER_ITERATIONS,
            enabled: true,
            priority: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Physics island
// ---------------------------------------------------------------------------

/// A connected group of bodies and constraints that can be simulated independently.
#[derive(Debug, Clone)]
pub struct PhysicsIsland {
    pub id: IslandId,
    pub bodies: Vec<BodyHandle>,
    pub constraints: Vec<ConstraintHandle>,
    pub sleeping: bool,
    pub sleep_timer: u32,
    pub aabb: Aabb,
    pub body_count_dynamic: u32,
    pub body_count_static: u32,
    pub body_count_kinematic: u32,
    pub total_energy: f32,
}

impl PhysicsIsland {
    pub fn new(id: IslandId) -> Self {
        Self {
            id,
            bodies: Vec::new(),
            constraints: Vec::new(),
            sleeping: false,
            sleep_timer: 0,
            aabb: Aabb::new(Vec3::ZERO, Vec3::ZERO),
            body_count_dynamic: 0,
            body_count_static: 0,
            body_count_kinematic: 0,
            total_energy: 0.0,
        }
    }

    pub fn body_count(&self) -> usize { self.bodies.len() }

    pub fn constraint_count(&self) -> usize { self.constraints.len() }

    /// Check if all dynamic bodies in this island are below the sleep threshold.
    pub fn can_sleep(&self, vel_threshold: f32) -> bool {
        self.total_energy < vel_threshold * vel_threshold * self.body_count_dynamic as f32
    }

    /// Wake up the island.
    pub fn wake_up(&mut self) {
        self.sleeping = false;
        self.sleep_timer = 0;
    }
}

// ---------------------------------------------------------------------------
// Narrow-phase cache
// ---------------------------------------------------------------------------

/// Cached result from a narrow-phase collision test.
#[derive(Debug, Clone)]
pub struct NarrowPhaseEntry {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    pub contact_normal: Vec3,
    pub contact_depth: f32,
    pub contact_point: Vec3,
    pub valid: bool,
    pub frame: u64,
}

/// Cache for narrow-phase results to avoid re-computation.
pub struct NarrowPhaseCache {
    entries: HashMap<(BodyHandle, BodyHandle), NarrowPhaseEntry>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl NarrowPhaseCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            capacity,
            hits: 0,
            misses: 0,
        }
    }

    /// Look up a cached entry.
    pub fn get(&mut self, a: BodyHandle, b: BodyHandle, current_frame: u64) -> Option<&NarrowPhaseEntry> {
        let key = if a <= b { (a, b) } else { (b, a) };
        if let Some(entry) = self.entries.get(&key) {
            if entry.valid && entry.frame >= current_frame.saturating_sub(1) {
                self.hits += 1;
                return Some(entry);
            }
        }
        self.misses += 1;
        None
    }

    /// Insert or update a cache entry.
    pub fn insert(&mut self, entry: NarrowPhaseEntry) {
        let key = if entry.body_a <= entry.body_b {
            (entry.body_a, entry.body_b)
        } else {
            (entry.body_b, entry.body_a)
        };

        if self.entries.len() >= self.capacity && !self.entries.contains_key(&key) {
            // Evict oldest entry.
            let oldest_key = self.entries
                .iter()
                .min_by_key(|(_, e)| e.frame)
                .map(|(k, _)| *k);
            if let Some(k) = oldest_key {
                self.entries.remove(&k);
            }
        }

        self.entries.insert(key, entry);
    }

    /// Invalidate entries for a specific body.
    pub fn invalidate_body(&mut self, body: BodyHandle) {
        self.entries.retain(|&(a, b), _| a != body && b != body);
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Get cache hit rate.
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total > 0 { self.hits as f32 / total as f32 } else { 0.0 }
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
}

// ---------------------------------------------------------------------------
// Sub-world
// ---------------------------------------------------------------------------

/// A sub-world representing a distinct simulation region.
#[derive(Debug, Clone)]
pub struct SubWorld {
    pub id: SubWorldId,
    pub name: String,
    pub bounds: Aabb,
    pub gravity: Vec3,
    pub time_scale: f32,
    pub enabled: bool,
    pub body_handles: HashSet<BodyHandle>,
    pub solver_iterations: u32,
    pub position_iterations: u32,
}

impl SubWorld {
    pub fn new(id: SubWorldId, name: &str, bounds: Aabb) -> Self {
        Self {
            id,
            name: name.to_string(),
            bounds,
            gravity: Vec3::GRAVITY,
            time_scale: 1.0,
            enabled: true,
            body_handles: HashSet::new(),
            solver_iterations: DEFAULT_SOLVER_ITERATIONS,
            position_iterations: DEFAULT_POSITION_ITERATIONS,
        }
    }

    pub fn contains_point(&self, point: Vec3) -> bool {
        point.x >= self.bounds.min.x && point.x <= self.bounds.max.x
            && point.y >= self.bounds.min.y && point.y <= self.bounds.max.y
            && point.z >= self.bounds.min.z && point.z <= self.bounds.max.z
    }

    pub fn body_count(&self) -> usize { self.body_handles.len() }
}

// ---------------------------------------------------------------------------
// World configuration
// ---------------------------------------------------------------------------

/// Configuration for the physics world.
#[derive(Debug, Clone)]
pub struct PhysicsWorldConfig {
    /// Global gravity.
    pub gravity: Vec3,
    /// Fixed timestep for simulation.
    pub fixed_timestep: f32,
    /// Maximum substeps per frame.
    pub max_substeps: u32,
    /// Solver velocity iterations.
    pub solver_iterations: u32,
    /// Solver position iterations.
    pub position_iterations: u32,
    /// Broad-phase configuration.
    pub broad_phase: BroadPhaseConfig,
    /// Sleep velocity threshold.
    pub sleep_velocity_threshold: f32,
    /// Sleep angular velocity threshold.
    pub sleep_angular_threshold: f32,
    /// Frames before sleep.
    pub sleep_frames_threshold: u32,
    /// Enable island splitting.
    pub island_splitting: bool,
    /// Enable sleeping optimization.
    pub sleeping_enabled: bool,
    /// Enable CCD (continuous collision detection).
    pub ccd_enabled: bool,
    /// Narrow-phase cache capacity.
    pub narrow_phase_cache_size: usize,
    /// Allow simulation of multiple sub-worlds.
    pub multi_world: bool,
}

impl Default for PhysicsWorldConfig {
    fn default() -> Self {
        Self {
            gravity: Vec3::GRAVITY,
            fixed_timestep: 1.0 / 60.0,
            max_substeps: 4,
            solver_iterations: DEFAULT_SOLVER_ITERATIONS,
            position_iterations: DEFAULT_POSITION_ITERATIONS,
            broad_phase: BroadPhaseConfig::default(),
            sleep_velocity_threshold: SLEEP_VELOCITY_THRESHOLD,
            sleep_angular_threshold: SLEEP_ANGULAR_THRESHOLD,
            sleep_frames_threshold: SLEEP_FRAMES_THRESHOLD,
            island_splitting: true,
            sleeping_enabled: true,
            ccd_enabled: false,
            narrow_phase_cache_size: NARROW_PHASE_CACHE_CAPACITY,
            multi_world: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Physics world statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhysicsWorldStats {
    pub total_bodies: u32,
    pub dynamic_bodies: u32,
    pub static_bodies: u32,
    pub kinematic_bodies: u32,
    pub awake_bodies: u32,
    pub sleeping_bodies: u32,
    pub total_constraints: u32,
    pub total_islands: u32,
    pub sleeping_islands: u32,
    pub awake_islands: u32,
    pub broad_phase_pairs: u32,
    pub narrow_phase_tests: u32,
    pub narrow_phase_cache_hits: u32,
    pub narrow_phase_cache_rate: f32,
    pub solver_iterations_used: u32,
    pub sub_world_count: u32,
    pub step_time_ms: f32,
    pub broad_phase_time_ms: f32,
    pub narrow_phase_time_ms: f32,
    pub solver_time_ms: f32,
    pub island_time_ms: f32,
}

// ---------------------------------------------------------------------------
// PhysicsWorldV2
// ---------------------------------------------------------------------------

/// Enhanced physics world with islands, sub-worlds, and configurable broad-phase.
pub struct PhysicsWorldV2 {
    config: PhysicsWorldConfig,
    bodies: HashMap<BodyHandle, RigidBodyState>,
    constraints: HashMap<ConstraintHandle, Constraint>,
    constraint_groups: HashMap<u32, ConstraintGroup>,
    islands: HashMap<IslandId, PhysicsIsland>,
    sub_worlds: HashMap<SubWorldId, SubWorld>,
    narrow_cache: NarrowPhaseCache,
    stats: PhysicsWorldStats,
    next_body_handle: BodyHandle,
    next_constraint_handle: ConstraintHandle,
    next_island_id: IslandId,
    next_sub_world_id: SubWorldId,
    frame: u64,
    accumulated_time: f32,
    broad_phase_dirty: bool,
}

impl PhysicsWorldV2 {
    /// Create a new physics world.
    pub fn new(config: PhysicsWorldConfig) -> Self {
        let cache_size = config.narrow_phase_cache_size;
        let mut world = Self {
            config,
            bodies: HashMap::new(),
            constraints: HashMap::new(),
            constraint_groups: HashMap::new(),
            islands: HashMap::new(),
            sub_worlds: HashMap::new(),
            narrow_cache: NarrowPhaseCache::new(cache_size),
            stats: PhysicsWorldStats::default(),
            next_body_handle: 1,
            next_constraint_handle: 1,
            next_island_id: 1,
            next_sub_world_id: 1,
            frame: 0,
            accumulated_time: 0.0,
            broad_phase_dirty: true,
        };

        // Create default sub-world.
        let default_bounds = Aabb::new(
            Vec3::new(-10000.0, -10000.0, -10000.0),
            Vec3::new(10000.0, 10000.0, 10000.0),
        );
        world.create_sub_world("default", default_bounds);

        world
    }

    // ----- Body management -----

    /// Add a dynamic body.
    pub fn add_dynamic_body(&mut self, position: Vec3, mass: f32) -> BodyHandle {
        let handle = self.next_body_handle;
        self.next_body_handle += 1;
        let body = RigidBodyState::new_dynamic(handle, position, mass);
        self.bodies.insert(handle, body);
        self.broad_phase_dirty = true;
        handle
    }

    /// Add a static body.
    pub fn add_static_body(&mut self, position: Vec3) -> BodyHandle {
        let handle = self.next_body_handle;
        self.next_body_handle += 1;
        let body = RigidBodyState::new_static(handle, position);
        self.bodies.insert(handle, body);
        self.broad_phase_dirty = true;
        handle
    }

    /// Remove a body.
    pub fn remove_body(&mut self, handle: BodyHandle) {
        self.bodies.remove(&handle);
        self.narrow_cache.invalidate_body(handle);
        // Remove from sub-worlds.
        for sw in self.sub_worlds.values_mut() {
            sw.body_handles.remove(&handle);
        }
        // Remove constraints involving this body.
        let to_remove: Vec<ConstraintHandle> = self.constraints
            .values()
            .filter(|c| c.body_a == handle || c.body_b == handle)
            .map(|c| c.handle)
            .collect();
        for ch in to_remove {
            self.constraints.remove(&ch);
        }
        self.broad_phase_dirty = true;
    }

    /// Get a body by handle.
    pub fn get_body(&self, handle: BodyHandle) -> Option<&RigidBodyState> {
        self.bodies.get(&handle)
    }

    /// Get a mutable body by handle.
    pub fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBodyState> {
        self.bodies.get_mut(&handle)
    }

    /// Get all body handles.
    pub fn body_handles(&self) -> Vec<BodyHandle> {
        self.bodies.keys().cloned().collect()
    }

    /// Number of bodies.
    pub fn body_count(&self) -> usize { self.bodies.len() }

    // ----- Constraint management -----

    /// Add a contact constraint.
    pub fn add_contact_constraint(&mut self, a: BodyHandle, b: BodyHandle) -> ConstraintHandle {
        let handle = self.next_constraint_handle;
        self.next_constraint_handle += 1;
        self.constraints.insert(handle, Constraint::new_contact(handle, a, b));
        handle
    }

    /// Add a distance constraint.
    pub fn add_distance_constraint(&mut self, a: BodyHandle, b: BodyHandle, distance: f32) -> ConstraintHandle {
        let handle = self.next_constraint_handle;
        self.next_constraint_handle += 1;
        self.constraints.insert(handle, Constraint::new_distance(handle, a, b, distance));
        handle
    }

    /// Remove a constraint.
    pub fn remove_constraint(&mut self, handle: ConstraintHandle) {
        self.constraints.remove(&handle);
    }

    /// Number of constraints.
    pub fn constraint_count(&self) -> usize { self.constraints.len() }

    // ----- Constraint groups -----

    /// Create a constraint group.
    pub fn create_constraint_group(&mut self, name: &str) -> u32 {
        let id = self.constraint_groups.len() as u32;
        self.constraint_groups.insert(id, ConstraintGroup::new(id, name));
        id
    }

    /// Add a constraint to a group.
    pub fn add_to_group(&mut self, group_id: u32, constraint: ConstraintHandle) {
        if let Some(group) = self.constraint_groups.get_mut(&group_id) {
            group.constraints.push(constraint);
        }
        if let Some(c) = self.constraints.get_mut(&constraint) {
            c.group_id = group_id;
        }
    }

    // ----- Sub-world management -----

    /// Create a sub-world.
    pub fn create_sub_world(&mut self, name: &str, bounds: Aabb) -> SubWorldId {
        let id = self.next_sub_world_id;
        self.next_sub_world_id += 1;
        let mut sw = SubWorld::new(id, name, bounds);
        sw.gravity = self.config.gravity;
        self.sub_worlds.insert(id, sw);
        id
    }

    /// Assign a body to a sub-world.
    pub fn assign_to_sub_world(&mut self, body: BodyHandle, sub_world: SubWorldId) {
        // Remove from current sub-world.
        for sw in self.sub_worlds.values_mut() {
            sw.body_handles.remove(&body);
        }
        // Add to new sub-world.
        if let Some(sw) = self.sub_worlds.get_mut(&sub_world) {
            sw.body_handles.insert(body);
        }
        if let Some(b) = self.bodies.get_mut(&body) {
            b.sub_world_id = sub_world;
        }
    }

    /// Get a sub-world.
    pub fn get_sub_world(&self, id: SubWorldId) -> Option<&SubWorld> {
        self.sub_worlds.get(&id)
    }

    // ----- Island management -----

    /// Rebuild islands from connectivity graph.
    pub fn rebuild_islands(&mut self) {
        self.islands.clear();

        // Union-Find for island assignment.
        let handles: Vec<BodyHandle> = self.bodies.keys().cloned().collect();
        let mut parent: HashMap<BodyHandle, BodyHandle> = HashMap::new();
        for &h in &handles {
            parent.insert(h, h);
        }

        // Find root with path compression.
        fn find(parent: &mut HashMap<BodyHandle, BodyHandle>, x: BodyHandle) -> BodyHandle {
            let p = *parent.get(&x).unwrap_or(&x);
            if p != x {
                let root = find(parent, p);
                parent.insert(x, root);
                root
            } else {
                x
            }
        }

        // Union by constraint connections.
        let constraints: Vec<(BodyHandle, BodyHandle)> = self.constraints
            .values()
            .filter(|c| c.enabled)
            .map(|c| (c.body_a, c.body_b))
            .collect();

        for (a, b) in &constraints {
            let ra = find(&mut parent, *a);
            let rb = find(&mut parent, *b);
            if ra != rb {
                parent.insert(ra, rb);
            }
        }

        // Group bodies by island root.
        let mut island_bodies: HashMap<BodyHandle, Vec<BodyHandle>> = HashMap::new();
        for &h in &handles {
            let root = find(&mut parent, h);
            island_bodies.entry(root).or_default().push(h);
        }

        // Create islands.
        for (_, body_handles) in island_bodies {
            let island_id = self.next_island_id;
            self.next_island_id += 1;

            let mut island = PhysicsIsland::new(island_id);
            let mut aabb = Aabb::new(
                Vec3::new(f32::MAX, f32::MAX, f32::MAX),
                Vec3::new(f32::MIN, f32::MIN, f32::MIN),
            );

            for &bh in &body_handles {
                if let Some(body) = self.bodies.get_mut(&bh) {
                    body.island_id = Some(island_id);
                    aabb = aabb.merge(&body.aabb);
                    match body.body_type {
                        BodyType::Dynamic => island.body_count_dynamic += 1,
                        BodyType::Static => island.body_count_static += 1,
                        BodyType::Kinematic => island.body_count_kinematic += 1,
                    }
                    island.total_energy += body.linear_velocity.length_sq() + body.angular_velocity.length_sq();
                }
            }

            island.bodies = body_handles;
            island.aabb = aabb;

            // Find constraints for this island.
            let island_body_set: HashSet<BodyHandle> = island.bodies.iter().cloned().collect();
            for c in self.constraints.values() {
                if c.enabled
                    && island_body_set.contains(&c.body_a)
                    && island_body_set.contains(&c.body_b)
                {
                    island.constraints.push(c.handle);
                }
            }

            self.islands.insert(island_id, island);
        }
    }

    /// Update island sleep states.
    pub fn update_island_sleep(&mut self) {
        if !self.config.sleeping_enabled { return; }

        let vel_threshold = self.config.sleep_velocity_threshold;
        let ang_threshold = self.config.sleep_angular_threshold;
        let frame_threshold = self.config.sleep_frames_threshold;

        for island in self.islands.values_mut() {
            if island.body_count_dynamic == 0 { continue; }

            let can_sleep = island.can_sleep(vel_threshold);

            if can_sleep {
                island.sleep_timer += 1;
                if island.sleep_timer >= frame_threshold && !island.sleeping {
                    island.sleeping = true;
                }
            } else {
                island.sleep_timer = 0;
                island.sleeping = false;
            }
        }

        // Apply sleep state to bodies.
        for island in self.islands.values() {
            for &bh in &island.bodies {
                if let Some(body) = self.bodies.get_mut(&bh) {
                    if island.sleeping && body.body_type == BodyType::Dynamic {
                        body.put_to_sleep();
                    }
                }
            }
        }
    }

    // ----- Simulation step -----

    /// Step the physics simulation.
    pub fn step(&mut self, dt: f32) {
        self.frame += 1;
        self.accumulated_time += dt;

        let fixed_dt = self.config.fixed_timestep;
        let mut substeps = 0u32;

        while self.accumulated_time >= fixed_dt && substeps < self.config.max_substeps {
            self.substep(fixed_dt);
            self.accumulated_time -= fixed_dt;
            substeps += 1;
        }

        // Rebuild islands periodically.
        if self.frame % 10 == 0 || self.broad_phase_dirty {
            self.rebuild_islands();
            self.broad_phase_dirty = false;
        }

        // Update sleep.
        self.update_island_sleep();

        // Update stats.
        self.update_stats();
    }

    /// Single fixed-timestep substep.
    fn substep(&mut self, dt: f32) {
        // Integrate forces.
        let gravity = self.config.gravity;
        let handles: Vec<BodyHandle> = self.bodies.keys().cloned().collect();

        for &handle in &handles {
            if let Some(body) = self.bodies.get_mut(&handle) {
                if body.body_type != BodyType::Dynamic || !body.is_awake() {
                    continue;
                }

                // Apply gravity.
                let grav = gravity.scale(body.gravity_scale);
                body.linear_velocity = body.linear_velocity.add(grav.scale(dt));

                // Apply damping.
                body.linear_velocity = body.linear_velocity.scale(1.0 - body.linear_damping * dt);
                body.angular_velocity = body.angular_velocity.scale(1.0 - body.angular_damping * dt);

                // Integrate position.
                body.position = body.position.add(body.linear_velocity.scale(dt));

                // Integrate orientation (simplified).
                let w = body.angular_velocity;
                let dq = Quat {
                    x: w.x * 0.5 * dt,
                    y: w.y * 0.5 * dt,
                    z: w.z * 0.5 * dt,
                    w: 0.0,
                };
                let q = body.orientation;
                body.orientation = Quat {
                    x: q.x + dq.w * q.x + dq.x * q.w + dq.y * q.z - dq.z * q.y,
                    y: q.y + dq.w * q.y - dq.x * q.z + dq.y * q.w + dq.z * q.x,
                    z: q.z + dq.w * q.z + dq.x * q.y - dq.y * q.x + dq.z * q.w,
                    w: q.w + dq.w * q.w - dq.x * q.x - dq.y * q.y - dq.z * q.z,
                }.normalize();

                // Update AABB.
                body.aabb = Aabb::from_center_half(body.position, Vec3::new(0.5, 0.5, 0.5));

                // Update sleep timer.
                if body.linear_velocity.length_sq() < self.config.sleep_velocity_threshold * self.config.sleep_velocity_threshold
                    && body.angular_velocity.length_sq() < self.config.sleep_angular_threshold * self.config.sleep_angular_threshold
                {
                    body.sleep_timer += 1;
                } else {
                    body.sleep_timer = 0;
                }
            }
        }
    }

    /// Update world statistics.
    fn update_stats(&mut self) {
        self.stats = PhysicsWorldStats::default();
        self.stats.total_bodies = self.bodies.len() as u32;
        self.stats.total_constraints = self.constraints.len() as u32;
        self.stats.total_islands = self.islands.len() as u32;
        self.stats.sub_world_count = self.sub_worlds.len() as u32;
        self.stats.narrow_phase_cache_rate = self.narrow_cache.hit_rate();

        for body in self.bodies.values() {
            match body.body_type {
                BodyType::Dynamic => self.stats.dynamic_bodies += 1,
                BodyType::Static => self.stats.static_bodies += 1,
                BodyType::Kinematic => self.stats.kinematic_bodies += 1,
            }
            match body.sleep_state {
                SleepState::Awake => self.stats.awake_bodies += 1,
                SleepState::Sleeping => self.stats.sleeping_bodies += 1,
                SleepState::WantsSleep => self.stats.awake_bodies += 1,
            }
        }

        for island in self.islands.values() {
            if island.sleeping {
                self.stats.sleeping_islands += 1;
            } else {
                self.stats.awake_islands += 1;
            }
        }
    }

    /// Get world statistics.
    pub fn stats(&self) -> &PhysicsWorldStats {
        &self.stats
    }

    /// Get the current configuration.
    pub fn config(&self) -> &PhysicsWorldConfig {
        &self.config
    }

    /// Set the gravity.
    pub fn set_gravity(&mut self, gravity: Vec3) {
        self.config.gravity = gravity;
    }

    /// Get the gravity.
    pub fn gravity(&self) -> Vec3 {
        self.config.gravity
    }

    /// Get the current frame.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Get all islands.
    pub fn islands(&self) -> &HashMap<IslandId, PhysicsIsland> {
        &self.islands
    }

    /// Wake up all bodies in an island.
    pub fn wake_island(&mut self, island_id: IslandId) {
        if let Some(island) = self.islands.get_mut(&island_id) {
            island.wake_up();
            let bodies = island.bodies.clone();
            for bh in bodies {
                if let Some(body) = self.bodies.get_mut(&bh) {
                    body.wake_up();
                }
            }
        }
    }

    /// Clear the narrow-phase cache.
    pub fn clear_narrow_cache(&mut self) {
        self.narrow_cache.clear();
    }

    /// Get the narrow-phase cache hit rate.
    pub fn narrow_cache_hit_rate(&self) -> f32 {
        self.narrow_cache.hit_rate()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_world() {
        let config = PhysicsWorldConfig::default();
        let world = PhysicsWorldV2::new(config);
        assert_eq!(world.body_count(), 0);
        assert_eq!(world.constraint_count(), 0);
    }

    #[test]
    fn test_add_bodies() {
        let mut world = PhysicsWorldV2::new(PhysicsWorldConfig::default());
        let h1 = world.add_dynamic_body(Vec3::ZERO, 1.0);
        let h2 = world.add_static_body(Vec3::new(0.0, -5.0, 0.0));
        assert_eq!(world.body_count(), 2);
        assert!(world.get_body(h1).is_some());
        assert_eq!(world.get_body(h2).unwrap().body_type, BodyType::Static);
    }

    #[test]
    fn test_remove_body() {
        let mut world = PhysicsWorldV2::new(PhysicsWorldConfig::default());
        let h = world.add_dynamic_body(Vec3::ZERO, 1.0);
        world.remove_body(h);
        assert_eq!(world.body_count(), 0);
    }

    #[test]
    fn test_simulation_step() {
        let mut world = PhysicsWorldV2::new(PhysicsWorldConfig::default());
        let h = world.add_dynamic_body(Vec3::new(0.0, 10.0, 0.0), 1.0);
        world.step(1.0 / 60.0);
        let body = world.get_body(h).unwrap();
        // Body should have moved down due to gravity.
        assert!(body.position.y < 10.0);
    }

    #[test]
    fn test_island_rebuild() {
        let mut world = PhysicsWorldV2::new(PhysicsWorldConfig::default());
        let h1 = world.add_dynamic_body(Vec3::ZERO, 1.0);
        let h2 = world.add_dynamic_body(Vec3::new(1.0, 0.0, 0.0), 1.0);
        world.add_distance_constraint(h1, h2, 1.0);
        world.rebuild_islands();
        // h1 and h2 should be in the same island.
        let b1 = world.get_body(h1).unwrap();
        let b2 = world.get_body(h2).unwrap();
        assert_eq!(b1.island_id, b2.island_id);
    }

    #[test]
    fn test_sub_world() {
        let mut world = PhysicsWorldV2::new(PhysicsWorldConfig::default());
        let sw = world.create_sub_world("test", Aabb::new(Vec3::ZERO, Vec3::new(100.0, 100.0, 100.0)));
        let h = world.add_dynamic_body(Vec3::new(50.0, 50.0, 50.0), 1.0);
        world.assign_to_sub_world(h, sw);
        let sub = world.get_sub_world(sw).unwrap();
        assert!(sub.body_handles.contains(&h));
    }

    #[test]
    fn test_narrow_phase_cache() {
        let mut cache = NarrowPhaseCache::new(100);
        cache.insert(NarrowPhaseEntry {
            body_a: 1,
            body_b: 2,
            contact_normal: Vec3::new(0.0, 1.0, 0.0),
            contact_depth: 0.1,
            contact_point: Vec3::ZERO,
            valid: true,
            frame: 10,
        });

        assert!(cache.get(1, 2, 10).is_some());
        assert!(cache.get(2, 1, 10).is_some()); // Canonical order.
        assert!(cache.get(3, 4, 10).is_none());
    }

    #[test]
    fn test_aabb_operations() {
        let a = Aabb::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        let b = Aabb::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.5, 1.5, 1.5));
        assert!(a.intersects(&b));

        let c = Aabb::new(Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0));
        assert!(!a.intersects(&c));
    }
}
