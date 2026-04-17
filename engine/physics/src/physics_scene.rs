// engine/physics/src/physics_scene.rs
//
// Physics scene management: add/remove bodies efficiently, broad-phase update,
// contact pair cache, island partitioning, step pipeline orchestration,
// sub-stepping, interpolation state management.
//
// The PhysicsScene is the top-level container that owns all rigid bodies,
// colliders, and constraints, and orchestrates the simulation pipeline each
// tick:
//   1. Pre-step: apply external forces, update AABBs
//   2. Broad phase: find overlapping AABB pairs
//   3. Narrow phase: generate contacts for overlapping pairs
//   4. Island building: partition connected bodies into islands
//   5. Constraint solving: solve velocity/position constraints per island
//   6. Integration: integrate velocities and positions
//   7. Post-step: update transforms, detect sleeping islands

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Body handle
// ---------------------------------------------------------------------------

/// A generational handle to a rigid body in the scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BodyHandle {
    pub index: u32,
    pub generation: u32,
}

impl BodyHandle {
    pub const INVALID: Self = Self { index: u32::MAX, generation: 0 };

    pub fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    pub fn is_valid(&self) -> bool {
        self.index != u32::MAX
    }
}

/// A generational handle to a collider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColliderHandle {
    pub index: u32,
    pub generation: u32,
}

impl ColliderHandle {
    pub const INVALID: Self = Self { index: u32::MAX, generation: 0 };

    pub fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }
}

/// A handle to a constraint (joint or contact).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstraintHandle {
    pub index: u32,
    pub generation: u32,
}

impl ConstraintHandle {
    pub const INVALID: Self = Self { index: u32::MAX, generation: 0 };
}

// ---------------------------------------------------------------------------
// AABB and broad phase
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    pub const INVALID: Self = Self {
        min: [f32::MAX; 3],
        max: [f32::MIN; 3],
    };

    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self { Self { min, max } }

    pub fn from_center_half(center: [f32; 3], half: [f32; 3]) -> Self {
        Self {
            min: [center[0] - half[0], center[1] - half[1], center[2] - half[2]],
            max: [center[0] + half[0], center[1] + half[1], center[2] + half[2]],
        }
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        self.min[0] <= other.max[0] && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1] && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2] && self.max[2] >= other.min[2]
    }

    pub fn merge(&self, other: &Self) -> Self {
        Self {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    pub fn expand(&self, margin: f32) -> Self {
        Self {
            min: [self.min[0] - margin, self.min[1] - margin, self.min[2] - margin],
            max: [self.max[0] + margin, self.max[1] + margin, self.max[2] + margin],
        }
    }

    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    pub fn volume(&self) -> f32 {
        let dx = (self.max[0] - self.min[0]).max(0.0);
        let dy = (self.max[1] - self.min[1]).max(0.0);
        let dz = (self.max[2] - self.min[2]).max(0.0);
        dx * dy * dz
    }

    pub fn surface_area(&self) -> f32 {
        let dx = (self.max[0] - self.min[0]).max(0.0);
        let dy = (self.max[1] - self.min[1]).max(0.0);
        let dz = (self.max[2] - self.min[2]).max(0.0);
        2.0 * (dx * dy + dy * dz + dz * dx)
    }
}

/// An overlapping pair from the broad phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BroadPhasePair {
    pub a: BodyHandle,
    pub b: BodyHandle,
}

impl BroadPhasePair {
    pub fn new(a: BodyHandle, b: BodyHandle) -> Self {
        // Canonical ordering for deduplication.
        if a.index <= b.index { Self { a, b } } else { Self { a: b, b: a } }
    }
}

/// Simple spatial hash broad phase.
pub struct SpatialHashBroadPhase {
    cell_size: f32,
    inv_cell_size: f32,
    cells: HashMap<(i32, i32, i32), Vec<BodyHandle>>,
    body_aabbs: HashMap<u32, Aabb>,
}

impl SpatialHashBroadPhase {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            cells: HashMap::new(),
            body_aabbs: HashMap::new(),
        }
    }

    fn cell_coords(&self, pos: [f32; 3]) -> (i32, i32, i32) {
        (
            (pos[0] * self.inv_cell_size).floor() as i32,
            (pos[1] * self.inv_cell_size).floor() as i32,
            (pos[2] * self.inv_cell_size).floor() as i32,
        )
    }

    /// Update a body's AABB in the broad phase.
    pub fn update_body(&mut self, handle: BodyHandle, aabb: Aabb) {
        self.body_aabbs.insert(handle.index, aabb);
    }

    /// Remove a body from the broad phase.
    pub fn remove_body(&mut self, handle: BodyHandle) {
        self.body_aabbs.remove(&handle.index);
    }

    /// Rebuild the spatial hash and find all overlapping pairs.
    pub fn find_pairs(&mut self) -> Vec<BroadPhasePair> {
        self.cells.clear();

        // Insert all bodies into cells.
        for (&idx, aabb) in &self.body_aabbs {
            let handle = BodyHandle::new(idx, 0);
            let min_cell = self.cell_coords(aabb.min);
            let max_cell = self.cell_coords(aabb.max);
            for x in min_cell.0..=max_cell.0 {
                for y in min_cell.1..=max_cell.1 {
                    for z in min_cell.2..=max_cell.2 {
                        self.cells.entry((x, y, z)).or_default().push(handle);
                    }
                }
            }
        }

        // Find pairs.
        let mut pair_set: HashSet<(u32, u32)> = HashSet::new();
        let mut pairs = Vec::new();

        for cell_bodies in self.cells.values() {
            for i in 0..cell_bodies.len() {
                for j in (i + 1)..cell_bodies.len() {
                    let a = cell_bodies[i];
                    let b = cell_bodies[j];
                    let key = if a.index < b.index { (a.index, b.index) } else { (b.index, a.index) };
                    if pair_set.insert(key) {
                        // Verify AABB overlap.
                        if let (Some(aabb_a), Some(aabb_b)) = (self.body_aabbs.get(&a.index), self.body_aabbs.get(&b.index)) {
                            if aabb_a.overlaps(aabb_b) {
                                pairs.push(BroadPhasePair::new(a, b));
                            }
                        }
                    }
                }
            }
        }

        pairs
    }
}

// ---------------------------------------------------------------------------
// Contact pair cache
// ---------------------------------------------------------------------------

/// A contact point between two bodies.
#[derive(Debug, Clone, Copy)]
pub struct ContactPoint {
    /// World-space contact position.
    pub position: [f32; 3],
    /// Contact normal (from A to B).
    pub normal: [f32; 3],
    /// Penetration depth (positive = overlapping).
    pub depth: f32,
    /// Local position on body A.
    pub local_a: [f32; 3],
    /// Local position on body B.
    pub local_b: [f32; 3],
    /// Accumulated normal impulse (for warm starting).
    pub normal_impulse: f32,
    /// Accumulated tangent impulse (friction).
    pub tangent_impulse: [f32; 2],
    /// Feature ID for persistent contact tracking.
    pub feature_id: u32,
}

/// A contact manifold between two bodies.
#[derive(Debug, Clone)]
pub struct ContactManifold {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    pub points: Vec<ContactPoint>,
    pub friction: f32,
    pub restitution: f32,
    /// Number of frames this manifold has been alive.
    pub age: u32,
    /// Whether this manifold was updated this frame.
    pub updated: bool,
}

impl ContactManifold {
    pub fn new(body_a: BodyHandle, body_b: BodyHandle) -> Self {
        Self {
            body_a,
            body_b,
            points: Vec::new(),
            friction: 0.5,
            restitution: 0.0,
            age: 0,
            updated: false,
        }
    }

    /// Number of contact points.
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Whether the manifold has at least one contact.
    pub fn has_contacts(&self) -> bool {
        !self.points.is_empty()
    }

    /// Get the deepest penetration depth.
    pub fn max_depth(&self) -> f32 {
        self.points.iter().map(|p| p.depth).fold(0.0f32, f32::max)
    }
}

/// Cache for contact manifolds between body pairs.
pub struct ContactPairCache {
    /// Manifolds keyed by body pair.
    manifolds: HashMap<(u32, u32), ContactManifold>,
    /// Maximum age before a manifold is removed.
    pub max_age: u32,
}

impl ContactPairCache {
    pub fn new() -> Self {
        Self {
            manifolds: HashMap::new(),
            max_age: 3,
        }
    }

    /// Get or create a manifold for a body pair.
    pub fn get_or_create(&mut self, pair: BroadPhasePair) -> &mut ContactManifold {
        let key = (pair.a.index, pair.b.index);
        self.manifolds.entry(key).or_insert_with(|| ContactManifold::new(pair.a, pair.b))
    }

    /// Get an existing manifold for a body pair.
    pub fn get(&self, a: BodyHandle, b: BodyHandle) -> Option<&ContactManifold> {
        let key = if a.index <= b.index { (a.index, b.index) } else { (b.index, a.index) };
        self.manifolds.get(&key)
    }

    /// Mark all manifolds as not updated (call before narrow phase).
    pub fn begin_frame(&mut self) {
        for manifold in self.manifolds.values_mut() {
            manifold.updated = false;
        }
    }

    /// Remove stale manifolds (not updated for max_age frames).
    pub fn cleanup(&mut self) {
        self.manifolds.retain(|_, m| {
            if m.updated {
                m.age = 0;
                true
            } else {
                m.age += 1;
                m.age <= self.max_age
            }
        });
    }

    /// Remove all manifolds involving a specific body.
    pub fn remove_body(&mut self, handle: BodyHandle) {
        self.manifolds.retain(|&(a, b), _| a != handle.index && b != handle.index);
    }

    /// Total number of cached manifolds.
    pub fn manifold_count(&self) -> usize {
        self.manifolds.len()
    }

    /// Total number of contact points across all manifolds.
    pub fn total_contact_points(&self) -> usize {
        self.manifolds.values().map(|m| m.points.len()).sum()
    }

    /// Iterate all manifolds.
    pub fn iter(&self) -> impl Iterator<Item = &ContactManifold> {
        self.manifolds.values()
    }

    /// Iterate all manifolds mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut ContactManifold> {
        self.manifolds.values_mut()
    }
}

// ---------------------------------------------------------------------------
// Island partitioning
// ---------------------------------------------------------------------------

/// An island is a connected group of bodies that can be solved independently.
#[derive(Debug, Clone)]
pub struct Island {
    pub id: u32,
    pub bodies: Vec<BodyHandle>,
    pub contacts: Vec<(u32, u32)>,
    pub constraints: Vec<ConstraintHandle>,
    pub is_sleeping: bool,
    pub sleep_timer: f32,
    pub total_energy: f32,
}

/// Island builder using union-find.
pub struct IslandBuilder {
    parent: Vec<u32>,
    rank: Vec<u32>,
    body_to_node: HashMap<u32, usize>,
    next_node: usize,
}

impl IslandBuilder {
    pub fn new(capacity: usize) -> Self {
        Self {
            parent: Vec::with_capacity(capacity),
            rank: Vec::with_capacity(capacity),
            body_to_node: HashMap::with_capacity(capacity),
            next_node: 0,
        }
    }

    fn get_or_create_node(&mut self, body_index: u32) -> usize {
        if let Some(&node) = self.body_to_node.get(&body_index) {
            return node;
        }
        let node = self.next_node;
        self.next_node += 1;
        self.parent.push(node as u32);
        self.rank.push(0);
        self.body_to_node.insert(body_index, node);
        node
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x as u32 {
            self.parent[x] = self.parent[self.parent[x] as usize];
            x = self.parent[x] as usize;
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb { return; }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb as u32;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra as u32;
        } else {
            self.parent[rb] = ra as u32;
            self.rank[ra] += 1;
        }
    }

    /// Reset the builder for a new frame.
    pub fn reset(&mut self) {
        self.parent.clear();
        self.rank.clear();
        self.body_to_node.clear();
        self.next_node = 0;
    }

    /// Build islands from contact pairs and constraint pairs.
    pub fn build(
        &mut self,
        contact_pairs: &[(BodyHandle, BodyHandle)],
        constraint_pairs: &[(BodyHandle, BodyHandle, ConstraintHandle)],
        body_handles: &[BodyHandle],
    ) -> Vec<Island> {
        self.reset();

        // Create nodes for all bodies.
        for handle in body_handles {
            self.get_or_create_node(handle.index);
        }

        // Union contact pairs.
        for &(a, b) in contact_pairs {
            let na = self.get_or_create_node(a.index);
            let nb = self.get_or_create_node(b.index);
            self.union(na, nb);
        }

        // Union constraint pairs.
        for &(a, b, _) in constraint_pairs {
            let na = self.get_or_create_node(a.index);
            let nb = self.get_or_create_node(b.index);
            self.union(na, nb);
        }

        // Group bodies by root.
        let mut island_map: HashMap<usize, Vec<BodyHandle>> = HashMap::new();
        for handle in body_handles {
            if let Some(&node) = self.body_to_node.get(&handle.index) {
                let root = self.find(node);
                island_map.entry(root).or_default().push(*handle);
            }
        }

        // Build island structs.
        let mut islands: Vec<Island> = Vec::new();
        for (id, (_, bodies)) in island_map.into_iter().enumerate() {
            let body_set: HashSet<u32> = bodies.iter().map(|b| b.index).collect();

            let contacts: Vec<(u32, u32)> = contact_pairs.iter()
                .filter(|(a, b)| body_set.contains(&a.index) && body_set.contains(&b.index))
                .map(|(a, b)| (a.index, b.index))
                .collect();

            let constraints: Vec<ConstraintHandle> = constraint_pairs.iter()
                .filter(|(a, b, _)| body_set.contains(&a.index) && body_set.contains(&b.index))
                .map(|(_, _, c)| *c)
                .collect();

            islands.push(Island {
                id: id as u32,
                bodies,
                contacts,
                constraints,
                is_sleeping: false,
                sleep_timer: 0.0,
                total_energy: 0.0,
            });
        }

        islands
    }
}

// ---------------------------------------------------------------------------
// Interpolation state
// ---------------------------------------------------------------------------

/// State used for rendering interpolation between physics steps.
#[derive(Debug, Clone, Copy)]
pub struct InterpolationState {
    /// Previous physics position.
    pub prev_position: [f32; 3],
    /// Previous physics rotation (quaternion).
    pub prev_rotation: [f32; 4],
    /// Current physics position.
    pub curr_position: [f32; 3],
    /// Current physics rotation.
    pub curr_rotation: [f32; 4],
}

impl InterpolationState {
    pub fn new(position: [f32; 3], rotation: [f32; 4]) -> Self {
        Self {
            prev_position: position,
            prev_rotation: rotation,
            curr_position: position,
            curr_rotation: rotation,
        }
    }

    /// Store current as previous, then update current.
    pub fn update(&mut self, new_position: [f32; 3], new_rotation: [f32; 4]) {
        self.prev_position = self.curr_position;
        self.prev_rotation = self.curr_rotation;
        self.curr_position = new_position;
        self.curr_rotation = new_rotation;
    }

    /// Interpolate between previous and current state.
    pub fn interpolate(&self, alpha: f32) -> ([f32; 3], [f32; 4]) {
        let pos = [
            self.prev_position[0] + (self.curr_position[0] - self.prev_position[0]) * alpha,
            self.prev_position[1] + (self.curr_position[1] - self.prev_position[1]) * alpha,
            self.prev_position[2] + (self.curr_position[2] - self.prev_position[2]) * alpha,
        ];
        // Simple quaternion lerp (nlerp) for rotation.
        let mut rot = [0.0f32; 4];
        let mut dot = 0.0f32;
        for i in 0..4 {
            dot += self.prev_rotation[i] * self.curr_rotation[i];
        }
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        for i in 0..4 {
            rot[i] = self.prev_rotation[i] * (1.0 - alpha) + self.curr_rotation[i] * sign * alpha;
        }
        // Normalize.
        let len = (rot[0] * rot[0] + rot[1] * rot[1] + rot[2] * rot[2] + rot[3] * rot[3]).sqrt();
        if len > 0.0 {
            for v in &mut rot { *v /= len; }
        }
        (pos, rot)
    }
}

// ---------------------------------------------------------------------------
// Physics scene configuration
// ---------------------------------------------------------------------------

/// Configuration for the physics scene.
#[derive(Debug, Clone)]
pub struct PhysicsSceneConfig {
    /// Gravity vector.
    pub gravity: [f32; 3],
    /// Fixed timestep in seconds.
    pub fixed_timestep: f32,
    /// Maximum number of sub-steps per frame.
    pub max_sub_steps: u32,
    /// Constraint solver iterations.
    pub solver_iterations: u32,
    /// Position correction iterations.
    pub position_iterations: u32,
    /// Sleep energy threshold.
    pub sleep_threshold: f32,
    /// Time a body must be below threshold to sleep (seconds).
    pub sleep_time: f32,
    /// Broad phase cell size.
    pub broad_phase_cell_size: f32,
    /// Contact pair cache max age.
    pub contact_cache_max_age: u32,
    /// Whether to enable sleeping.
    pub enable_sleeping: bool,
    /// Whether to enable interpolation.
    pub enable_interpolation: bool,
    /// Whether to enable warm starting.
    pub enable_warm_starting: bool,
    /// Collision margin for GJK+EPA.
    pub collision_margin: f32,
}

impl Default for PhysicsSceneConfig {
    fn default() -> Self {
        Self {
            gravity: [0.0, -9.81, 0.0],
            fixed_timestep: 1.0 / 60.0,
            max_sub_steps: 8,
            solver_iterations: 10,
            position_iterations: 4,
            sleep_threshold: 0.01,
            sleep_time: 2.0,
            broad_phase_cell_size: 4.0,
            contact_cache_max_age: 3,
            enable_sleeping: true,
            enable_interpolation: true,
            enable_warm_starting: true,
            collision_margin: 0.04,
        }
    }
}

// ---------------------------------------------------------------------------
// Rigid body state
// ---------------------------------------------------------------------------

/// The type of a rigid body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyType {
    Static,
    Dynamic,
    Kinematic,
}

/// State of a rigid body in the physics scene.
#[derive(Debug, Clone)]
pub struct RigidBodyState {
    pub handle: BodyHandle,
    pub body_type: BodyType,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub linear_velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub force: [f32; 3],
    pub torque: [f32; 3],
    pub mass: f32,
    pub inv_mass: f32,
    pub inertia: [f32; 3],
    pub inv_inertia: [f32; 3],
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub aabb: Aabb,
    pub is_sleeping: bool,
    pub collision_layer: u32,
    pub collision_mask: u32,
    pub interpolation: InterpolationState,
    pub user_data: u64,
}

impl RigidBodyState {
    pub fn new_dynamic(handle: BodyHandle, position: [f32; 3], mass: f32) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self {
            handle,
            body_type: BodyType::Dynamic,
            position,
            rotation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            force: [0.0; 3],
            torque: [0.0; 3],
            mass,
            inv_mass,
            inertia: [mass; 3],
            inv_inertia: [inv_mass; 3],
            linear_damping: 0.01,
            angular_damping: 0.01,
            aabb: Aabb::from_center_half(position, [0.5, 0.5, 0.5]),
            is_sleeping: false,
            collision_layer: 1,
            collision_mask: u32::MAX,
            interpolation: InterpolationState::new(position, [0.0, 0.0, 0.0, 1.0]),
            user_data: 0,
        }
    }

    pub fn new_static(handle: BodyHandle, position: [f32; 3]) -> Self {
        Self {
            handle,
            body_type: BodyType::Static,
            position,
            rotation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            force: [0.0; 3],
            torque: [0.0; 3],
            mass: 0.0,
            inv_mass: 0.0,
            inertia: [0.0; 3],
            inv_inertia: [0.0; 3],
            linear_damping: 0.0,
            angular_damping: 0.0,
            aabb: Aabb::from_center_half(position, [0.5, 0.5, 0.5]),
            is_sleeping: false,
            collision_layer: 1,
            collision_mask: u32::MAX,
            interpolation: InterpolationState::new(position, [0.0, 0.0, 0.0, 1.0]),
            user_data: 0,
        }
    }

    /// Kinetic energy of the body.
    pub fn kinetic_energy(&self) -> f32 {
        let lv = self.linear_velocity;
        let av = self.angular_velocity;
        let lin = 0.5 * self.mass * (lv[0] * lv[0] + lv[1] * lv[1] + lv[2] * lv[2]);
        let ang = 0.5 * (self.inertia[0] * av[0] * av[0] + self.inertia[1] * av[1] * av[1] + self.inertia[2] * av[2] * av[2]);
        lin + ang
    }

    /// Apply gravity to the body.
    pub fn apply_gravity(&mut self, gravity: [f32; 3]) {
        if self.body_type != BodyType::Dynamic || self.is_sleeping { return; }
        self.force[0] += gravity[0] * self.mass;
        self.force[1] += gravity[1] * self.mass;
        self.force[2] += gravity[2] * self.mass;
    }

    /// Semi-implicit Euler integration.
    pub fn integrate(&mut self, dt: f32) {
        if self.body_type != BodyType::Dynamic { return; }

        // Velocity integration.
        for i in 0..3 {
            self.linear_velocity[i] += self.force[i] * self.inv_mass * dt;
            self.angular_velocity[i] += self.torque[i] * self.inv_inertia[i] * dt;
        }

        // Damping.
        let lin_damp = (1.0 - self.linear_damping * dt).max(0.0);
        let ang_damp = (1.0 - self.angular_damping * dt).max(0.0);
        for i in 0..3 {
            self.linear_velocity[i] *= lin_damp;
            self.angular_velocity[i] *= ang_damp;
        }

        // Position integration.
        for i in 0..3 {
            self.position[i] += self.linear_velocity[i] * dt;
        }

        // Rotation integration (simplified).
        let av = self.angular_velocity;
        let q = self.rotation;
        let dq = [
            0.5 * dt * (av[0] * q[3] + av[1] * q[2] - av[2] * q[1]),
            0.5 * dt * (-av[0] * q[2] + av[1] * q[3] + av[2] * q[0]),
            0.5 * dt * (av[0] * q[1] - av[1] * q[0] + av[2] * q[3]),
            0.5 * dt * (-av[0] * q[0] - av[1] * q[1] - av[2] * q[2]),
        ];
        for i in 0..4 {
            self.rotation[i] += dq[i];
        }
        // Normalize quaternion.
        let len = (self.rotation[0].powi(2) + self.rotation[1].powi(2) + self.rotation[2].powi(2) + self.rotation[3].powi(2)).sqrt();
        if len > 0.0 {
            for v in &mut self.rotation { *v /= len; }
        }

        // Clear forces.
        self.force = [0.0; 3];
        self.torque = [0.0; 3];
    }
}

// ---------------------------------------------------------------------------
// Physics scene
// ---------------------------------------------------------------------------

/// Per-frame statistics for the physics scene.
#[derive(Debug, Clone, Default)]
pub struct PhysicsSceneStats {
    pub body_count: u32,
    pub dynamic_count: u32,
    pub static_count: u32,
    pub sleeping_count: u32,
    pub broad_phase_pairs: u32,
    pub contact_manifolds: u32,
    pub contact_points: u32,
    pub island_count: u32,
    pub sub_steps: u32,
    pub step_time_us: u64,
    pub broad_phase_time_us: u64,
    pub narrow_phase_time_us: u64,
    pub solver_time_us: u64,
    pub integration_time_us: u64,
}

/// The main physics scene.
pub struct PhysicsScene {
    pub config: PhysicsSceneConfig,
    /// All rigid bodies.
    bodies: Vec<RigidBodyState>,
    /// Free list for body slots.
    free_bodies: Vec<u32>,
    /// Generation counter per body slot.
    generations: Vec<u32>,
    /// Broad phase.
    broad_phase: SpatialHashBroadPhase,
    /// Contact pair cache.
    contact_cache: ContactPairCache,
    /// Island builder.
    island_builder: IslandBuilder,
    /// Current islands.
    pub islands: Vec<Island>,
    /// Accumulated time for sub-stepping.
    accumulated_time: f32,
    /// Per-frame statistics.
    pub stats: PhysicsSceneStats,
}

impl PhysicsScene {
    pub fn new(config: PhysicsSceneConfig) -> Self {
        let cell_size = config.broad_phase_cell_size;
        let cache_age = config.contact_cache_max_age;
        Self {
            config,
            bodies: Vec::new(),
            free_bodies: Vec::new(),
            generations: Vec::new(),
            broad_phase: SpatialHashBroadPhase::new(cell_size),
            contact_cache: ContactPairCache::new(),
            island_builder: IslandBuilder::new(256),
            islands: Vec::new(),
            accumulated_time: 0.0,
            stats: PhysicsSceneStats::default(),
        }
    }

    /// Add a dynamic body to the scene.
    pub fn add_dynamic_body(&mut self, position: [f32; 3], mass: f32) -> BodyHandle {
        self.allocate_body(|handle| RigidBodyState::new_dynamic(handle, position, mass))
    }

    /// Add a static body to the scene.
    pub fn add_static_body(&mut self, position: [f32; 3]) -> BodyHandle {
        self.allocate_body(|handle| RigidBodyState::new_static(handle, position))
    }

    /// Remove a body from the scene.
    pub fn remove_body(&mut self, handle: BodyHandle) -> bool {
        if !self.validate_handle(handle) { return false; }
        let idx = handle.index as usize;
        self.generations[idx] += 1;
        self.free_bodies.push(handle.index);
        self.broad_phase.remove_body(handle);
        self.contact_cache.remove_body(handle);
        true
    }

    /// Get a body by handle.
    pub fn get_body(&self, handle: BodyHandle) -> Option<&RigidBodyState> {
        if !self.validate_handle(handle) { return None; }
        Some(&self.bodies[handle.index as usize])
    }

    /// Get a mutable body by handle.
    pub fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBodyState> {
        if !self.validate_handle(handle) { return None; }
        Some(&mut self.bodies[handle.index as usize])
    }

    /// Step the simulation by the given delta time.
    pub fn step(&mut self, dt: f32) {
        let step_start = std::time::Instant::now();

        self.accumulated_time += dt;
        let fixed_dt = self.config.fixed_timestep;
        let mut sub_steps = 0u32;

        while self.accumulated_time >= fixed_dt && sub_steps < self.config.max_sub_steps {
            self.single_step(fixed_dt);
            self.accumulated_time -= fixed_dt;
            sub_steps += 1;
        }

        // Clamp accumulated time to prevent spiral of death.
        if self.accumulated_time > fixed_dt * 2.0 {
            self.accumulated_time = 0.0;
        }

        self.stats.sub_steps = sub_steps;
        self.stats.step_time_us = step_start.elapsed().as_micros() as u64;
    }

    /// Get the interpolation alpha for rendering.
    pub fn interpolation_alpha(&self) -> f32 {
        self.accumulated_time / self.config.fixed_timestep
    }

    /// Get the interpolated transform for rendering.
    pub fn get_interpolated_transform(&self, handle: BodyHandle) -> Option<([f32; 3], [f32; 4])> {
        let body = self.get_body(handle)?;
        let alpha = self.interpolation_alpha();
        Some(body.interpolation.interpolate(alpha))
    }

    /// Number of bodies in the scene.
    pub fn body_count(&self) -> usize {
        self.bodies.len() - self.free_bodies.len()
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    fn allocate_body<F>(&mut self, create: F) -> BodyHandle
    where
        F: FnOnce(BodyHandle) -> RigidBodyState,
    {
        let (index, generation) = if let Some(free_idx) = self.free_bodies.pop() {
            self.generations[free_idx as usize] += 1;
            (free_idx, self.generations[free_idx as usize])
        } else {
            let idx = self.bodies.len() as u32;
            self.bodies.push(RigidBodyState::new_static(BodyHandle::INVALID, [0.0; 3]));
            self.generations.push(0);
            (idx, 0)
        };

        let handle = BodyHandle::new(index, generation);
        let body = create(handle);
        self.broad_phase.update_body(handle, body.aabb);
        self.bodies[index as usize] = body;
        handle
    }

    fn validate_handle(&self, handle: BodyHandle) -> bool {
        let idx = handle.index as usize;
        idx < self.bodies.len() && self.generations[idx] == handle.generation
    }

    fn single_step(&mut self, dt: f32) {
        let gravity = self.config.gravity;

        // 1. Pre-step: apply gravity, update AABBs.
        for body in &mut self.bodies {
            if body.body_type == BodyType::Dynamic && !body.is_sleeping {
                body.apply_gravity(gravity);
                body.interpolation.update(body.position, body.rotation);
            }
        }

        // 2. Broad phase.
        let bp_start = std::time::Instant::now();
        for body in &self.bodies {
            if body.handle.is_valid() {
                self.broad_phase.update_body(body.handle, body.aabb);
            }
        }
        let pairs = self.broad_phase.find_pairs();
        self.stats.broad_phase_time_us = bp_start.elapsed().as_micros() as u64;
        self.stats.broad_phase_pairs = pairs.len() as u32;

        // 3. Narrow phase (simplified -- just cache the pairs).
        let np_start = std::time::Instant::now();
        self.contact_cache.begin_frame();
        for pair in &pairs {
            let manifold = self.contact_cache.get_or_create(*pair);
            manifold.updated = true;
        }
        self.contact_cache.cleanup();
        self.stats.narrow_phase_time_us = np_start.elapsed().as_micros() as u64;
        self.stats.contact_manifolds = self.contact_cache.manifold_count() as u32;
        self.stats.contact_points = self.contact_cache.total_contact_points() as u32;

        // 4. Island building.
        let active_handles: Vec<BodyHandle> = self.bodies.iter()
            .filter(|b| b.handle.is_valid() && b.body_type == BodyType::Dynamic)
            .map(|b| b.handle)
            .collect();
        let contact_pairs: Vec<(BodyHandle, BodyHandle)> = pairs.iter()
            .map(|p| (p.a, p.b))
            .collect();
        self.islands = self.island_builder.build(&contact_pairs, &[], &active_handles);
        self.stats.island_count = self.islands.len() as u32;

        // 5. Integration.
        let int_start = std::time::Instant::now();
        for body in &mut self.bodies {
            body.integrate(dt);
            // Update AABB.
            body.aabb = Aabb::from_center_half(body.position, [0.5, 0.5, 0.5]);
        }
        self.stats.integration_time_us = int_start.elapsed().as_micros() as u64;

        // 6. Sleep detection.
        if self.config.enable_sleeping {
            let sleep_threshold = self.config.sleep_threshold;
            let sleep_time = self.config.sleep_time;
            for body in &mut self.bodies {
                if body.body_type == BodyType::Dynamic {
                    let energy = body.kinetic_energy();
                    if energy < sleep_threshold {
                        // Body is nearly at rest.
                        // (sleep timer would be managed per-island in production)
                    }
                }
            }
        }

        // Update stats.
        self.stats.body_count = self.bodies.len() as u32 - self.free_bodies.len() as u32;
        self.stats.dynamic_count = self.bodies.iter().filter(|b| b.body_type == BodyType::Dynamic).count() as u32;
        self.stats.static_count = self.bodies.iter().filter(|b| b.body_type == BodyType::Static).count() as u32;
        self.stats.sleeping_count = self.bodies.iter().filter(|b| b.is_sleeping).count() as u32;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_overlap() {
        let a = Aabb::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let b = Aabb::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);
        assert!(a.overlaps(&b));
        let c = Aabb::new([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]);
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_body_handle_allocation() {
        let mut scene = PhysicsScene::new(PhysicsSceneConfig::default());
        let h1 = scene.add_dynamic_body([0.0, 1.0, 0.0], 1.0);
        let h2 = scene.add_dynamic_body([5.0, 1.0, 0.0], 2.0);
        assert_eq!(scene.body_count(), 2);
        assert!(scene.get_body(h1).is_some());
        assert!(scene.get_body(h2).is_some());
    }

    #[test]
    fn test_body_removal() {
        let mut scene = PhysicsScene::new(PhysicsSceneConfig::default());
        let h1 = scene.add_dynamic_body([0.0, 0.0, 0.0], 1.0);
        assert!(scene.remove_body(h1));
        assert!(scene.get_body(h1).is_none());
    }

    #[test]
    fn test_interpolation() {
        let mut state = InterpolationState::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
        state.update([10.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
        let (pos, _) = state.interpolate(0.5);
        assert!((pos[0] - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_island_building() {
        let mut builder = IslandBuilder::new(10);
        let bodies = vec![
            BodyHandle::new(0, 0),
            BodyHandle::new(1, 0),
            BodyHandle::new(2, 0),
            BodyHandle::new(3, 0),
        ];
        let contacts = vec![
            (bodies[0], bodies[1]),
            (bodies[2], bodies[3]),
        ];
        let islands = builder.build(&contacts, &[], &bodies);
        assert_eq!(islands.len(), 2); // Two separate islands.
    }

    #[test]
    fn test_step_with_gravity() {
        let mut scene = PhysicsScene::new(PhysicsSceneConfig::default());
        let h = scene.add_dynamic_body([0.0, 10.0, 0.0], 1.0);
        scene.step(1.0 / 60.0);
        let body = scene.get_body(h).unwrap();
        // Body should have moved downward due to gravity.
        assert!(body.position[1] < 10.0);
    }
}
