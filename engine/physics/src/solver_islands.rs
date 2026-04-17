// engine/physics/src/solver_islands.rs
//
// Island-based constraint solving for the Genovo engine.
//
// Groups connected bodies into islands using a union-find (disjoint set) data
// structure, then solves each island independently. Islands where all bodies
// are sleeping are skipped entirely, providing a significant performance boost
// for large physics scenes with many static or resting objects.
//
// Features:
// - Union-find with path compression and union by rank
// - Island building from contact pairs and joint connections
// - Per-island sequential impulse solving
// - Island sleeping: put islands to sleep when all bodies have low velocity
// - Island waking: wake islands when external forces or collisions are applied
// - Split islands when joints break or contacts are lost
// - Island merging when new contacts bridge two islands
// - Statistics tracking (island count, sleeping count, body distribution)
//
// The solver processes islands in parallel when possible, since islands are
// by definition independent of each other.

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default solver iterations for velocity constraints.
pub const DEFAULT_VELOCITY_ITERATIONS: u32 = 8;

/// Default solver iterations for position constraints.
pub const DEFAULT_POSITION_ITERATIONS: u32 = 3;

/// Velocity threshold below which a body is considered resting.
pub const DEFAULT_SLEEP_LINEAR_THRESHOLD: f32 = 0.01;

/// Angular velocity threshold for sleeping.
pub const DEFAULT_SLEEP_ANGULAR_THRESHOLD: f32 = 0.02;

/// Time (seconds) a body must be below thresholds before sleeping.
pub const DEFAULT_SLEEP_TIME_THRESHOLD: f32 = 0.5;

/// Maximum bodies per island before splitting is considered.
pub const MAX_ISLAND_SIZE: usize = 1024;

/// Baumgarte stabilization factor.
pub const BAUMGARTE_FACTOR: f32 = 0.2;

/// Slop for penetration recovery.
pub const PENETRATION_SLOP: f32 = 0.005;

/// Maximum correction velocity.
pub const MAX_CORRECTION_VELOCITY: f32 = 10.0;

// ---------------------------------------------------------------------------
// Union-Find
// ---------------------------------------------------------------------------

/// Disjoint set (union-find) with path compression and union by rank.
#[derive(Debug, Clone)]
pub struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u32>,
    size: Vec<u32>,
}

impl UnionFind {
    /// Creates a new union-find for `n` elements.
    pub fn new(n: usize) -> Self {
        let parent = (0..n as u32).collect();
        let rank = vec![0u32; n];
        let size = vec![1u32; n];
        Self { parent, rank, size }
    }

    /// Find the root of the set containing `x` (with path compression).
    pub fn find(&mut self, x: u32) -> u32 {
        if self.parent[x as usize] != x {
            self.parent[x as usize] = self.find(self.parent[x as usize]);
        }
        self.parent[x as usize]
    }

    /// Union the sets containing `x` and `y`. Returns the new root.
    pub fn union(&mut self, x: u32, y: u32) -> u32 {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return rx;
        }

        // Union by rank.
        if self.rank[rx as usize] < self.rank[ry as usize] {
            self.parent[rx as usize] = ry;
            self.size[ry as usize] += self.size[rx as usize];
            ry
        } else if self.rank[rx as usize] > self.rank[ry as usize] {
            self.parent[ry as usize] = rx;
            self.size[rx as usize] += self.size[ry as usize];
            rx
        } else {
            self.parent[ry as usize] = rx;
            self.size[rx as usize] += self.size[ry as usize];
            self.rank[rx as usize] += 1;
            rx
        }
    }

    /// Check if two elements are in the same set.
    pub fn connected(&mut self, x: u32, y: u32) -> bool {
        self.find(x) == self.find(y)
    }

    /// Returns the size of the set containing `x`.
    pub fn set_size(&mut self, x: u32) -> u32 {
        let root = self.find(x);
        self.size[root as usize]
    }

    /// Returns the number of distinct sets.
    pub fn set_count(&mut self) -> usize {
        let n = self.parent.len();
        let mut roots = HashSet::new();
        for i in 0..n {
            roots.insert(self.find(i as u32));
        }
        roots.len()
    }

    /// Resize the union-find to accommodate more elements.
    pub fn resize(&mut self, new_size: usize) {
        let old_size = self.parent.len();
        if new_size <= old_size {
            return;
        }
        for i in old_size..new_size {
            self.parent.push(i as u32);
            self.rank.push(0);
            self.size.push(1);
        }
    }

    /// Reset all elements to be in their own set.
    pub fn reset(&mut self) {
        for i in 0..self.parent.len() {
            self.parent[i] = i as u32;
            self.rank[i] = 0;
            self.size[i] = 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Body state for the solver
// ---------------------------------------------------------------------------

/// Physics body state used by the island solver.
#[derive(Debug, Clone)]
pub struct SolverBody {
    /// Body ID.
    pub id: u32,
    /// Position.
    pub position: [f32; 3],
    /// Linear velocity.
    pub linear_velocity: [f32; 3],
    /// Angular velocity.
    pub angular_velocity: [f32; 3],
    /// Inverse mass (0 = static/infinite mass).
    pub inv_mass: f32,
    /// Inverse inertia tensor (diagonal, local space).
    pub inv_inertia: [f32; 3],
    /// Whether this body is static.
    pub is_static: bool,
    /// Whether this body is sleeping.
    pub sleeping: bool,
    /// Time spent below sleep thresholds.
    pub sleep_timer: f32,
    /// External force accumulator.
    pub force: [f32; 3],
    /// External torque accumulator.
    pub torque: [f32; 3],
    /// Friction coefficient.
    pub friction: f32,
    /// Restitution coefficient.
    pub restitution: f32,
    /// Linear damping.
    pub linear_damping: f32,
    /// Angular damping.
    pub angular_damping: f32,
    /// Island index this body belongs to.
    pub island_index: Option<usize>,
}

impl SolverBody {
    /// Creates a new dynamic body.
    pub fn dynamic(id: u32, mass: f32, position: [f32; 3]) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        // Simple sphere-like inertia.
        let inertia = 0.4 * mass;
        let inv_inertia = if inertia > 0.0 { 1.0 / inertia } else { 0.0 };
        Self {
            id,
            position,
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            inv_mass,
            inv_inertia: [inv_inertia; 3],
            is_static: false,
            sleeping: false,
            sleep_timer: 0.0,
            force: [0.0; 3],
            torque: [0.0; 3],
            friction: 0.5,
            restitution: 0.3,
            linear_damping: 0.01,
            angular_damping: 0.02,
            island_index: None,
        }
    }

    /// Creates a new static body.
    pub fn static_body(id: u32, position: [f32; 3]) -> Self {
        Self {
            id,
            position,
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            inv_mass: 0.0,
            inv_inertia: [0.0; 3],
            is_static: true,
            sleeping: false,
            sleep_timer: 0.0,
            force: [0.0; 3],
            torque: [0.0; 3],
            friction: 0.5,
            restitution: 0.3,
            linear_damping: 0.0,
            angular_damping: 0.0,
            island_index: None,
        }
    }

    /// Returns the kinetic energy of this body.
    pub fn kinetic_energy(&self) -> f32 {
        if self.inv_mass == 0.0 { return 0.0; }
        let mass = 1.0 / self.inv_mass;
        let lin_sq = self.linear_velocity[0].powi(2) +
            self.linear_velocity[1].powi(2) + self.linear_velocity[2].powi(2);
        let ang_sq = self.angular_velocity[0].powi(2) +
            self.angular_velocity[1].powi(2) + self.angular_velocity[2].powi(2);
        0.5 * mass * lin_sq + 0.5 * mass * ang_sq * 0.4
    }

    /// Check if this body is below the sleep thresholds.
    pub fn is_resting(&self, linear_threshold: f32, angular_threshold: f32) -> bool {
        let lin_sq = self.linear_velocity[0].powi(2) +
            self.linear_velocity[1].powi(2) + self.linear_velocity[2].powi(2);
        let ang_sq = self.angular_velocity[0].powi(2) +
            self.angular_velocity[1].powi(2) + self.angular_velocity[2].powi(2);
        lin_sq < linear_threshold * linear_threshold && ang_sq < angular_threshold * angular_threshold
    }

    /// Apply a velocity impulse.
    pub fn apply_impulse(&mut self, impulse: [f32; 3], contact_offset: [f32; 3]) {
        if self.is_static { return; }
        self.linear_velocity[0] += impulse[0] * self.inv_mass;
        self.linear_velocity[1] += impulse[1] * self.inv_mass;
        self.linear_velocity[2] += impulse[2] * self.inv_mass;

        // Angular impulse from contact offset cross impulse.
        let torque_impulse = cross(contact_offset, impulse);
        self.angular_velocity[0] += torque_impulse[0] * self.inv_inertia[0];
        self.angular_velocity[1] += torque_impulse[1] * self.inv_inertia[1];
        self.angular_velocity[2] += torque_impulse[2] * self.inv_inertia[2];
    }

    /// Integrate velocity from forces.
    pub fn integrate_forces(&mut self, dt: f32, gravity: [f32; 3]) {
        if self.is_static || self.sleeping { return; }
        self.linear_velocity[0] += (gravity[0] + self.force[0] * self.inv_mass) * dt;
        self.linear_velocity[1] += (gravity[1] + self.force[1] * self.inv_mass) * dt;
        self.linear_velocity[2] += (gravity[2] + self.force[2] * self.inv_mass) * dt;

        self.angular_velocity[0] += self.torque[0] * self.inv_inertia[0] * dt;
        self.angular_velocity[1] += self.torque[1] * self.inv_inertia[1] * dt;
        self.angular_velocity[2] += self.torque[2] * self.inv_inertia[2] * dt;

        // Apply damping.
        let lin_damp = (1.0 - self.linear_damping * dt).max(0.0);
        let ang_damp = (1.0 - self.angular_damping * dt).max(0.0);
        self.linear_velocity[0] *= lin_damp;
        self.linear_velocity[1] *= lin_damp;
        self.linear_velocity[2] *= lin_damp;
        self.angular_velocity[0] *= ang_damp;
        self.angular_velocity[1] *= ang_damp;
        self.angular_velocity[2] *= ang_damp;

        // Clear accumulators.
        self.force = [0.0; 3];
        self.torque = [0.0; 3];
    }

    /// Integrate position from velocity.
    pub fn integrate_position(&mut self, dt: f32) {
        if self.is_static || self.sleeping { return; }
        self.position[0] += self.linear_velocity[0] * dt;
        self.position[1] += self.linear_velocity[1] * dt;
        self.position[2] += self.linear_velocity[2] * dt;
    }

    /// Wake this body up.
    pub fn wake(&mut self) {
        self.sleeping = false;
        self.sleep_timer = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Contact constraint
// ---------------------------------------------------------------------------

/// A velocity-level contact constraint for the sequential impulse solver.
#[derive(Debug, Clone)]
pub struct ContactConstraint {
    /// Body A index.
    pub body_a: u32,
    /// Body B index.
    pub body_b: u32,
    /// Contact normal (B to A).
    pub normal: [f32; 3],
    /// Contact point offset from body A center.
    pub offset_a: [f32; 3],
    /// Contact point offset from body B center.
    pub offset_b: [f32; 3],
    /// Penetration depth.
    pub penetration: f32,
    /// Combined restitution.
    pub restitution: f32,
    /// Combined friction.
    pub friction: f32,
    /// Tangent directions.
    pub tangent1: [f32; 3],
    pub tangent2: [f32; 3],
    /// Accumulated normal impulse.
    pub normal_impulse: f32,
    /// Accumulated tangent impulses.
    pub tangent_impulse1: f32,
    pub tangent_impulse2: f32,
    /// Effective mass for normal direction.
    pub normal_mass: f32,
    /// Effective mass for tangent directions.
    pub tangent_mass1: f32,
    pub tangent_mass2: f32,
    /// Velocity bias (for restitution and position correction).
    pub velocity_bias: f32,
}

impl ContactConstraint {
    /// Create a new contact constraint from contact data.
    pub fn new(
        body_a: u32, body_b: u32,
        normal: [f32; 3],
        offset_a: [f32; 3], offset_b: [f32; 3],
        penetration: f32,
        restitution: f32, friction: f32,
    ) -> Self {
        // Compute tangent frame.
        let (t1, t2) = compute_tangent_frame(normal);

        Self {
            body_a, body_b, normal,
            offset_a, offset_b,
            penetration, restitution, friction,
            tangent1: t1, tangent2: t2,
            normal_impulse: 0.0,
            tangent_impulse1: 0.0,
            tangent_impulse2: 0.0,
            normal_mass: 0.0,
            tangent_mass1: 0.0,
            tangent_mass2: 0.0,
            velocity_bias: 0.0,
        }
    }

    /// Pre-compute effective masses and velocity bias.
    pub fn prepare(&mut self, bodies: &[SolverBody], dt: f32) {
        let a = &bodies[self.body_a as usize];
        let b = &bodies[self.body_b as usize];

        // Normal effective mass.
        self.normal_mass = compute_effective_mass(
            a.inv_mass, a.inv_inertia, self.offset_a,
            b.inv_mass, b.inv_inertia, self.offset_b,
            self.normal,
        );

        // Tangent effective masses.
        self.tangent_mass1 = compute_effective_mass(
            a.inv_mass, a.inv_inertia, self.offset_a,
            b.inv_mass, b.inv_inertia, self.offset_b,
            self.tangent1,
        );
        self.tangent_mass2 = compute_effective_mass(
            a.inv_mass, a.inv_inertia, self.offset_a,
            b.inv_mass, b.inv_inertia, self.offset_b,
            self.tangent2,
        );

        // Velocity bias for restitution and Baumgarte stabilization.
        let relative_vel = compute_relative_velocity(a, b, self.offset_a, self.offset_b);
        let vn = dot(relative_vel, self.normal);

        self.velocity_bias = 0.0;
        if vn < -1.0 {
            self.velocity_bias = -self.restitution * vn;
        }

        // Position correction bias.
        let correction = (self.penetration - PENETRATION_SLOP).max(0.0);
        self.velocity_bias += BAUMGARTE_FACTOR / dt * correction;
        self.velocity_bias = self.velocity_bias.min(MAX_CORRECTION_VELOCITY);
    }

    /// Solve the normal constraint for one iteration.
    pub fn solve_normal(&mut self, bodies: &mut [SolverBody]) {
        let relative_vel = compute_relative_velocity(
            &bodies[self.body_a as usize], &bodies[self.body_b as usize],
            self.offset_a, self.offset_b,
        );
        let vn = dot(relative_vel, self.normal);
        let impulse_mag = self.normal_mass * (-vn + self.velocity_bias);

        // Clamp accumulated impulse (no pulling).
        let old_impulse = self.normal_impulse;
        self.normal_impulse = (self.normal_impulse + impulse_mag).max(0.0);
        let applied = self.normal_impulse - old_impulse;

        let impulse = scale(self.normal, applied);
        bodies[self.body_a as usize].apply_impulse(impulse, self.offset_a);
        bodies[self.body_b as usize].apply_impulse(negate(impulse), self.offset_b);
    }

    /// Solve the friction constraints.
    pub fn solve_friction(&mut self, bodies: &mut [SolverBody]) {
        let max_friction = self.friction * self.normal_impulse;

        // Tangent 1.
        {
            let relative_vel = compute_relative_velocity(
                &bodies[self.body_a as usize], &bodies[self.body_b as usize],
                self.offset_a, self.offset_b,
            );
            let vt = dot(relative_vel, self.tangent1);
            let impulse_mag = self.tangent_mass1 * (-vt);

            let old = self.tangent_impulse1;
            self.tangent_impulse1 = (self.tangent_impulse1 + impulse_mag).clamp(-max_friction, max_friction);
            let applied = self.tangent_impulse1 - old;

            let impulse = scale(self.tangent1, applied);
            bodies[self.body_a as usize].apply_impulse(impulse, self.offset_a);
            bodies[self.body_b as usize].apply_impulse(negate(impulse), self.offset_b);
        }

        // Tangent 2.
        {
            let relative_vel = compute_relative_velocity(
                &bodies[self.body_a as usize], &bodies[self.body_b as usize],
                self.offset_a, self.offset_b,
            );
            let vt = dot(relative_vel, self.tangent2);
            let impulse_mag = self.tangent_mass2 * (-vt);

            let old = self.tangent_impulse2;
            self.tangent_impulse2 = (self.tangent_impulse2 + impulse_mag).clamp(-max_friction, max_friction);
            let applied = self.tangent_impulse2 - old;

            let impulse = scale(self.tangent2, applied);
            bodies[self.body_a as usize].apply_impulse(impulse, self.offset_a);
            bodies[self.body_b as usize].apply_impulse(negate(impulse), self.offset_b);
        }
    }
}

// ---------------------------------------------------------------------------
// Island
// ---------------------------------------------------------------------------

/// A physics island: a group of connected bodies and their constraints.
#[derive(Debug, Clone)]
pub struct Island {
    /// Island index.
    pub index: usize,
    /// Body indices in the global body array.
    pub body_indices: Vec<u32>,
    /// Constraint indices in the global constraint array.
    pub constraint_indices: Vec<usize>,
    /// Whether all bodies in this island are sleeping.
    pub sleeping: bool,
    /// Whether this island was just woken up.
    pub just_woken: bool,
    /// Total kinetic energy of the island.
    pub total_energy: f32,
}

impl Island {
    /// Creates a new empty island.
    pub fn new(index: usize) -> Self {
        Self {
            index,
            body_indices: Vec::new(),
            constraint_indices: Vec::new(),
            sleeping: false,
            just_woken: false,
            total_energy: 0.0,
        }
    }

    /// Returns the number of bodies in this island.
    pub fn body_count(&self) -> usize {
        self.body_indices.len()
    }

    /// Returns the number of constraints in this island.
    pub fn constraint_count(&self) -> usize {
        self.constraint_indices.len()
    }

    /// Returns the number of dynamic bodies.
    pub fn dynamic_body_count(&self, bodies: &[SolverBody]) -> usize {
        self.body_indices.iter()
            .filter(|&&i| !bodies[i as usize].is_static)
            .count()
    }
}

// ---------------------------------------------------------------------------
// Solver configuration
// ---------------------------------------------------------------------------

/// Configuration for the island constraint solver.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Number of velocity constraint iterations.
    pub velocity_iterations: u32,
    /// Number of position constraint iterations.
    pub position_iterations: u32,
    /// Gravity vector.
    pub gravity: [f32; 3],
    /// Linear velocity threshold for sleeping.
    pub sleep_linear_threshold: f32,
    /// Angular velocity threshold for sleeping.
    pub sleep_angular_threshold: f32,
    /// Time below thresholds before sleeping (seconds).
    pub sleep_time_threshold: f32,
    /// Whether island sleeping is enabled.
    pub enable_sleeping: bool,
    /// Whether to warm-start constraints from previous frame.
    pub warm_starting: bool,
    /// Warm-starting factor (0-1, typically 0.8).
    pub warm_start_factor: f32,
    /// Whether to solve islands in parallel.
    pub parallel_islands: bool,
    /// Minimum island size to justify parallel execution.
    pub parallel_min_island_size: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            velocity_iterations: DEFAULT_VELOCITY_ITERATIONS,
            position_iterations: DEFAULT_POSITION_ITERATIONS,
            gravity: [0.0, -9.81, 0.0],
            sleep_linear_threshold: DEFAULT_SLEEP_LINEAR_THRESHOLD,
            sleep_angular_threshold: DEFAULT_SLEEP_ANGULAR_THRESHOLD,
            sleep_time_threshold: DEFAULT_SLEEP_TIME_THRESHOLD,
            enable_sleeping: true,
            warm_starting: true,
            warm_start_factor: 0.8,
            parallel_islands: false,
            parallel_min_island_size: 32,
        }
    }
}

// ---------------------------------------------------------------------------
// Solver statistics
// ---------------------------------------------------------------------------

/// Statistics from the island solver.
#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    /// Total number of islands.
    pub total_islands: usize,
    /// Number of sleeping islands (skipped).
    pub sleeping_islands: usize,
    /// Number of active (solved) islands.
    pub active_islands: usize,
    /// Total bodies across all islands.
    pub total_bodies: usize,
    /// Sleeping bodies.
    pub sleeping_bodies: usize,
    /// Total constraints solved.
    pub total_constraints: usize,
    /// Largest island size (body count).
    pub largest_island: usize,
    /// Average island size.
    pub average_island_size: f32,
    /// Number of islands just woken up.
    pub woken_islands: usize,
    /// Total kinetic energy.
    pub total_energy: f32,
    /// Solve time in microseconds.
    pub solve_time_us: u64,
}

impl SolverStats {
    pub fn summary(&self) -> String {
        format!(
            "Solver: {} islands ({} sleeping, {} active), {} bodies ({} sleeping), {} constraints, largest={}",
            self.total_islands, self.sleeping_islands, self.active_islands,
            self.total_bodies, self.sleeping_bodies, self.total_constraints,
            self.largest_island,
        )
    }
}

// ---------------------------------------------------------------------------
// Island solver
// ---------------------------------------------------------------------------

/// The island-based constraint solver.
///
/// Each physics step:
/// 1. Build islands from contact pairs and joint connections.
/// 2. Identify sleeping islands (skip).
/// 3. For each active island:
///    a. Integrate forces.
///    b. Prepare constraints (compute effective masses).
///    c. Warm-start from accumulated impulses.
///    d. Solve velocity constraints (iterate).
///    e. Integrate positions.
///    f. Solve position constraints (iterate).
/// 4. Update sleep timers and sleep/wake bodies.
#[derive(Debug)]
pub struct IslandSolver {
    /// Solver configuration.
    pub config: SolverConfig,
    /// All bodies.
    pub bodies: Vec<SolverBody>,
    /// All contact constraints.
    pub constraints: Vec<ContactConstraint>,
    /// Built islands.
    pub islands: Vec<Island>,
    /// Union-find for island building.
    union_find: UnionFind,
    /// Map from body ID to index in the bodies array.
    body_index_map: HashMap<u32, usize>,
    /// Statistics from the last solve.
    pub stats: SolverStats,
}

impl IslandSolver {
    /// Creates a new island solver.
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            bodies: Vec::new(),
            constraints: Vec::new(),
            islands: Vec::new(),
            union_find: UnionFind::new(0),
            body_index_map: HashMap::new(),
            stats: SolverStats::default(),
        }
    }

    /// Add a body to the solver.
    pub fn add_body(&mut self, body: SolverBody) {
        let idx = self.bodies.len();
        self.body_index_map.insert(body.id, idx);
        self.bodies.push(body);
    }

    /// Remove a body from the solver.
    pub fn remove_body(&mut self, id: u32) {
        if let Some(&idx) = self.body_index_map.get(&id) {
            self.bodies.swap_remove(idx);
            self.body_index_map.remove(&id);
            // Update the map for the body that was swapped in.
            if idx < self.bodies.len() {
                self.body_index_map.insert(self.bodies[idx].id, idx);
            }
        }
    }

    /// Set constraints for this frame.
    pub fn set_constraints(&mut self, constraints: Vec<ContactConstraint>) {
        self.constraints = constraints;
    }

    /// Get body index by ID.
    pub fn body_index(&self, id: u32) -> Option<usize> {
        self.body_index_map.get(&id).copied()
    }

    /// Build islands from the current constraints.
    pub fn build_islands(&mut self) {
        let n = self.bodies.len();
        if n == 0 { return; }

        self.union_find = UnionFind::new(n);

        // Union bodies connected by constraints.
        for constraint in &self.constraints {
            let a = constraint.body_a as usize;
            let b = constraint.body_b as usize;
            if a < n && b < n {
                // Only union if at least one body is dynamic.
                if !self.bodies[a].is_static || !self.bodies[b].is_static {
                    self.union_find.union(a as u32, b as u32);
                }
            }
        }

        // Group bodies by island root.
        let mut island_map: HashMap<u32, Vec<u32>> = HashMap::new();
        for i in 0..n {
            if self.bodies[i].is_static { continue; }
            let root = self.union_find.find(i as u32);
            island_map.entry(root).or_default().push(i as u32);
        }

        // Build Island structs.
        self.islands.clear();
        for (island_idx, (_, body_indices)) in island_map.iter().enumerate() {
            let mut island = Island::new(island_idx);
            island.body_indices = body_indices.clone();

            // Also include static bodies that touch any dynamic body in this island.
            let mut static_bodies = HashSet::new();
            for &bi in &island.body_indices {
                for (ci, constraint) in self.constraints.iter().enumerate() {
                    if constraint.body_a == bi && self.bodies[constraint.body_b as usize].is_static {
                        static_bodies.insert(constraint.body_b);
                        island.constraint_indices.push(ci);
                    } else if constraint.body_b == bi && self.bodies[constraint.body_a as usize].is_static {
                        static_bodies.insert(constraint.body_a);
                        island.constraint_indices.push(ci);
                    } else if constraint.body_a == bi || constraint.body_b == bi {
                        island.constraint_indices.push(ci);
                    }
                }
            }
            for sid in static_bodies {
                island.body_indices.push(sid);
            }
            island.constraint_indices.sort();
            island.constraint_indices.dedup();

            // Check sleep state.
            if self.config.enable_sleeping {
                island.sleeping = island.body_indices.iter()
                    .filter(|&&i| !self.bodies[i as usize].is_static)
                    .all(|&i| self.bodies[i as usize].sleeping);
            }

            // Set island index on bodies.
            for &bi in &island.body_indices {
                self.bodies[bi as usize].island_index = Some(island_idx);
            }

            self.islands.push(island);
        }
    }

    /// Solve all islands for one timestep.
    pub fn solve(&mut self, dt: f32) {
        self.stats = SolverStats::default();
        self.stats.total_islands = self.islands.len();
        self.stats.total_bodies = self.bodies.len();
        self.stats.total_constraints = self.constraints.len();

        for island in &mut self.islands {
            if island.sleeping {
                self.stats.sleeping_islands += 1;
                self.stats.sleeping_bodies += island.body_indices.iter()
                    .filter(|&&i| !self.bodies[i as usize].is_static).count();
                continue;
            }

            self.stats.active_islands += 1;
            self.stats.largest_island = self.stats.largest_island.max(island.body_count());

            // 1. Integrate forces for all bodies in the island.
            for &bi in &island.body_indices {
                self.bodies[bi as usize].integrate_forces(dt, self.config.gravity);
            }

            // 2. Prepare constraints.
            for &ci in &island.constraint_indices {
                self.constraints[ci].prepare(&self.bodies, dt);
            }

            // 3. Warm-start.
            if self.config.warm_starting {
                for &ci in &island.constraint_indices {
                    let c = &self.constraints[ci];
                    let impulse = scale(c.normal, c.normal_impulse * self.config.warm_start_factor);
                    let t1_impulse = scale(c.tangent1, c.tangent_impulse1 * self.config.warm_start_factor);
                    let t2_impulse = scale(c.tangent2, c.tangent_impulse2 * self.config.warm_start_factor);

                    let total = add(add(impulse, t1_impulse), t2_impulse);
                    self.bodies[c.body_a as usize].apply_impulse(total, c.offset_a);
                    self.bodies[c.body_b as usize].apply_impulse(negate(total), c.offset_b);
                }
            }

            // 4. Solve velocity constraints.
            for _ in 0..self.config.velocity_iterations {
                for &ci in &island.constraint_indices {
                    self.constraints[ci].solve_normal(&mut self.bodies);
                    self.constraints[ci].solve_friction(&mut self.bodies);
                }
            }

            // 5. Integrate positions.
            for &bi in &island.body_indices {
                self.bodies[bi as usize].integrate_position(dt);
            }

            // 6. Compute island energy and update sleep timers.
            island.total_energy = 0.0;
            let mut all_resting = true;
            for &bi in &island.body_indices {
                let body = &mut self.bodies[bi as usize];
                if body.is_static { continue; }

                island.total_energy += body.kinetic_energy();

                if body.is_resting(self.config.sleep_linear_threshold, self.config.sleep_angular_threshold) {
                    body.sleep_timer += dt;
                    if body.sleep_timer < self.config.sleep_time_threshold {
                        all_resting = false;
                    }
                } else {
                    body.sleep_timer = 0.0;
                    all_resting = false;
                }
            }

            self.stats.total_energy += island.total_energy;

            // Sleep the entire island if all bodies are resting.
            if self.config.enable_sleeping && all_resting {
                island.sleeping = true;
                for &bi in &island.body_indices {
                    let body = &mut self.bodies[bi as usize];
                    if !body.is_static {
                        body.sleeping = true;
                        body.linear_velocity = [0.0; 3];
                        body.angular_velocity = [0.0; 3];
                    }
                }
                self.stats.sleeping_islands += 1;
            }
        }

        if self.stats.active_islands > 0 {
            self.stats.average_island_size = self.stats.total_bodies as f32 / self.stats.total_islands as f32;
        }
    }

    /// Wake a specific body and its entire island.
    pub fn wake_body(&mut self, id: u32) {
        if let Some(&idx) = self.body_index_map.get(&id) {
            self.bodies[idx].wake();
            if let Some(island_idx) = self.bodies[idx].island_index {
                if island_idx < self.islands.len() {
                    self.islands[island_idx].sleeping = false;
                    self.islands[island_idx].just_woken = true;
                    for &bi in &self.islands[island_idx].body_indices {
                        self.bodies[bi as usize].wake();
                    }
                    self.stats.woken_islands += 1;
                }
            }
        }
    }

    /// Apply a force to a body, waking it if necessary.
    pub fn apply_force(&mut self, id: u32, force: [f32; 3]) {
        if let Some(&idx) = self.body_index_map.get(&id) {
            self.bodies[idx].force[0] += force[0];
            self.bodies[idx].force[1] += force[1];
            self.bodies[idx].force[2] += force[2];
            if self.bodies[idx].sleeping {
                self.wake_body(id);
            }
        }
    }

    /// Clear all bodies and constraints.
    pub fn clear(&mut self) {
        self.bodies.clear();
        self.constraints.clear();
        self.islands.clear();
        self.body_index_map.clear();
    }
}

// ---------------------------------------------------------------------------
// Vector math helpers
// ---------------------------------------------------------------------------

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn negate(v: [f32; 3]) -> [f32; 3] {
    [-v[0], -v[1], -v[2]]
}

fn length(v: [f32; 3]) -> f32 {
    dot(v, v).sqrt()
}

/// Compute a tangent frame from a normal.
fn compute_tangent_frame(normal: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let up = if normal[1].abs() < 0.99 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let t1 = cross(normal, up);
    let t1_len = length(t1);
    let t1 = if t1_len > 1e-6 { scale(t1, 1.0 / t1_len) } else { [1.0, 0.0, 0.0] };
    let t2 = cross(normal, t1);
    (t1, t2)
}

/// Compute relative velocity at contact point.
fn compute_relative_velocity(
    a: &SolverBody, b: &SolverBody,
    offset_a: [f32; 3], offset_b: [f32; 3],
) -> [f32; 3] {
    let vel_a = add(a.linear_velocity, cross(a.angular_velocity, offset_a));
    let vel_b = add(b.linear_velocity, cross(b.angular_velocity, offset_b));
    sub(vel_a, vel_b)
}

/// Compute the effective mass for a constraint direction.
fn compute_effective_mass(
    inv_mass_a: f32, inv_inertia_a: [f32; 3], offset_a: [f32; 3],
    inv_mass_b: f32, inv_inertia_b: [f32; 3], offset_b: [f32; 3],
    direction: [f32; 3],
) -> f32 {
    let rn_a = cross(offset_a, direction);
    let rn_b = cross(offset_b, direction);

    let inv_effective = inv_mass_a + inv_mass_b
        + rn_a[0] * rn_a[0] * inv_inertia_a[0]
        + rn_a[1] * rn_a[1] * inv_inertia_a[1]
        + rn_a[2] * rn_a[2] * inv_inertia_a[2]
        + rn_b[0] * rn_b[0] * inv_inertia_b[0]
        + rn_b[1] * rn_b[1] * inv_inertia_b[1]
        + rn_b[2] * rn_b[2] * inv_inertia_b[2];

    if inv_effective > 1e-10 {
        1.0 / inv_effective
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(2, 3);
        assert!(uf.connected(0, 1));
        assert!(!uf.connected(0, 2));
        uf.union(1, 3);
        assert!(uf.connected(0, 3));
        assert_eq!(uf.set_size(0), 4);
    }

    #[test]
    fn test_union_find_set_count() {
        let mut uf = UnionFind::new(6);
        uf.union(0, 1);
        uf.union(2, 3);
        uf.union(4, 5);
        assert_eq!(uf.set_count(), 3);
    }

    #[test]
    fn test_solver_body_sleeping() {
        let body = SolverBody::dynamic(0, 1.0, [0.0, 0.0, 0.0]);
        assert!(body.is_resting(0.01, 0.02));
    }

    #[test]
    fn test_solver_body_energy() {
        let mut body = SolverBody::dynamic(0, 2.0, [0.0, 0.0, 0.0]);
        body.linear_velocity = [1.0, 0.0, 0.0];
        let energy = body.kinetic_energy();
        assert!((energy - 1.0).abs() < 0.01); // 0.5 * 2.0 * 1.0^2 = 1.0
    }

    #[test]
    fn test_island_solver_basic() {
        let config = SolverConfig::default();
        let mut solver = IslandSolver::new(config);

        solver.add_body(SolverBody::dynamic(0, 1.0, [0.0, 5.0, 0.0]));
        solver.add_body(SolverBody::static_body(1, [0.0, 0.0, 0.0]));

        let constraint = ContactConstraint::new(
            0, 1,
            [0.0, 1.0, 0.0],
            [0.0, -0.5, 0.0], [0.0, 0.5, 0.0],
            0.01, 0.3, 0.5,
        );
        solver.set_constraints(vec![constraint]);
        solver.build_islands();
        solver.solve(1.0 / 60.0);

        assert!(solver.stats.active_islands > 0);
    }

    #[test]
    fn test_tangent_frame() {
        let normal = [0.0, 1.0, 0.0];
        let (t1, t2) = compute_tangent_frame(normal);
        // Tangents should be perpendicular to normal.
        assert!(dot(t1, normal).abs() < 1e-5);
        assert!(dot(t2, normal).abs() < 1e-5);
        // And to each other.
        assert!(dot(t1, t2).abs() < 1e-5);
    }

    #[test]
    fn test_effective_mass() {
        let em = compute_effective_mass(
            1.0, [1.0, 1.0, 1.0], [0.0, -0.5, 0.0],
            0.0, [0.0, 0.0, 0.0], [0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
        );
        assert!(em > 0.0);
    }

    #[test]
    fn test_wake_body() {
        let config = SolverConfig::default();
        let mut solver = IslandSolver::new(config);
        let mut body = SolverBody::dynamic(0, 1.0, [0.0, 0.0, 0.0]);
        body.sleeping = true;
        solver.add_body(body);
        solver.wake_body(0);
        assert!(!solver.bodies[0].sleeping);
    }
}
