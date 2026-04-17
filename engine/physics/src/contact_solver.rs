// engine/physics/src/contact_solver.rs
//
// Contact constraint solver: velocity-level constraint, position correction
// (split impulse or Baumgarte), friction model (box/cone/ellipse), restitution
// with velocity threshold, warm starting from previous frame, configurable
// iterations, convergence tracking.
//
// This is a Sequential Impulse (SI) solver that processes contact constraints
// one at a time, applying corrective impulses to maintain non-penetration and
// friction constraints. The solver supports warm starting by reusing impulses
// from the previous frame as initial guesses.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn neg3(a: [f32; 3]) -> [f32; 3] {
    [-a[0], -a[1], -a[2]]
}

#[inline]
fn len3(a: [f32; 3]) -> f32 {
    dot3(a, a).sqrt()
}

#[inline]
fn normalize3(a: [f32; 3]) -> [f32; 3] {
    let l = len3(a);
    if l < 1e-12 { [0.0, 1.0, 0.0] } else { scale3(a, 1.0 / l) }
}

// ---------------------------------------------------------------------------
// Solver body
// ---------------------------------------------------------------------------

/// A body as seen by the solver. Contains the velocity state that the solver
/// reads and writes.
#[derive(Debug, Clone)]
pub struct SolverBody {
    /// Body index.
    pub index: u32,
    /// Linear velocity.
    pub linear_velocity: [f32; 3],
    /// Angular velocity.
    pub angular_velocity: [f32; 3],
    /// Inverse mass.
    pub inv_mass: f32,
    /// Inverse inertia tensor (diagonal approximation in world space).
    pub inv_inertia: [f32; 3],
    /// Position (for position correction).
    pub position: [f32; 3],
    /// Rotation quaternion (for position correction).
    pub rotation: [f32; 4],
    /// Whether this body is static (infinite mass).
    pub is_static: bool,
    /// Position correction delta (split impulse).
    pub delta_position: [f32; 3],
    pub delta_rotation: [f32; 3],
}

impl SolverBody {
    pub fn new_dynamic(
        index: u32,
        position: [f32; 3],
        rotation: [f32; 4],
        linear_velocity: [f32; 3],
        angular_velocity: [f32; 3],
        inv_mass: f32,
        inv_inertia: [f32; 3],
    ) -> Self {
        Self {
            index,
            linear_velocity,
            angular_velocity,
            inv_mass,
            inv_inertia,
            position,
            rotation,
            is_static: false,
            delta_position: [0.0; 3],
            delta_rotation: [0.0; 3],
        }
    }

    pub fn new_static(index: u32, position: [f32; 3]) -> Self {
        Self {
            index,
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            inv_mass: 0.0,
            inv_inertia: [0.0; 3],
            position,
            rotation: [0.0, 0.0, 0.0, 1.0],
            is_static: true,
            delta_position: [0.0; 3],
            delta_rotation: [0.0; 3],
        }
    }

    /// Compute the velocity at a world-space point on this body.
    #[inline]
    pub fn velocity_at_point(&self, r: [f32; 3]) -> [f32; 3] {
        add3(self.linear_velocity, cross3(self.angular_velocity, r))
    }

    /// Apply a linear impulse at a world-space offset from the center of mass.
    #[inline]
    pub fn apply_impulse(&mut self, impulse: [f32; 3], r: [f32; 3]) {
        if self.is_static { return; }
        self.linear_velocity = add3(self.linear_velocity, scale3(impulse, self.inv_mass));
        let torque = cross3(r, impulse);
        self.angular_velocity[0] += torque[0] * self.inv_inertia[0];
        self.angular_velocity[1] += torque[1] * self.inv_inertia[1];
        self.angular_velocity[2] += torque[2] * self.inv_inertia[2];
    }

    /// Apply a position correction impulse (split impulse method).
    #[inline]
    pub fn apply_position_impulse(&mut self, impulse: [f32; 3], r: [f32; 3]) {
        if self.is_static { return; }
        self.delta_position = add3(self.delta_position, scale3(impulse, self.inv_mass));
        let torque = cross3(r, impulse);
        self.delta_rotation[0] += torque[0] * self.inv_inertia[0];
        self.delta_rotation[1] += torque[1] * self.inv_inertia[1];
        self.delta_rotation[2] += torque[2] * self.inv_inertia[2];
    }
}

// ---------------------------------------------------------------------------
// Contact constraint
// ---------------------------------------------------------------------------

/// A single contact constraint between two bodies.
#[derive(Debug, Clone)]
pub struct ContactConstraint {
    /// Index of body A in the solver body array.
    pub body_a: usize,
    /// Index of body B in the solver body array.
    pub body_b: usize,
    /// Contact normal (from A to B).
    pub normal: [f32; 3],
    /// Contact point offset from body A center.
    pub r_a: [f32; 3],
    /// Contact point offset from body B center.
    pub r_b: [f32; 3],
    /// Penetration depth (positive = overlapping).
    pub depth: f32,
    /// Combined friction coefficient.
    pub friction: f32,
    /// Combined restitution.
    pub restitution: f32,
    /// Tangent direction 1 (perpendicular to normal).
    pub tangent1: [f32; 3],
    /// Tangent direction 2 (perpendicular to normal and tangent1).
    pub tangent2: [f32; 3],
    /// Accumulated normal impulse.
    pub normal_impulse: f32,
    /// Accumulated tangent impulses.
    pub tangent1_impulse: f32,
    pub tangent2_impulse: f32,
    /// Effective mass for normal direction.
    pub normal_mass: f32,
    /// Effective mass for tangent directions.
    pub tangent1_mass: f32,
    pub tangent2_mass: f32,
    /// Velocity bias for restitution.
    pub velocity_bias: f32,
    /// Position correction bias.
    pub position_bias: f32,
    /// Whether this constraint was warm-started.
    pub warm_started: bool,
    /// Feature ID for warm-start matching.
    pub feature_id: u32,
}

impl ContactConstraint {
    pub fn new(
        body_a: usize,
        body_b: usize,
        normal: [f32; 3],
        r_a: [f32; 3],
        r_b: [f32; 3],
        depth: f32,
        friction: f32,
        restitution: f32,
    ) -> Self {
        // Compute tangent basis.
        let (t1, t2) = compute_tangent_basis(normal);

        Self {
            body_a,
            body_b,
            normal,
            r_a,
            r_b,
            depth,
            friction,
            restitution,
            tangent1: t1,
            tangent2: t2,
            normal_impulse: 0.0,
            tangent1_impulse: 0.0,
            tangent2_impulse: 0.0,
            normal_mass: 0.0,
            tangent1_mass: 0.0,
            tangent2_mass: 0.0,
            velocity_bias: 0.0,
            position_bias: 0.0,
            warm_started: false,
            feature_id: 0,
        }
    }

    /// Pre-compute effective masses and velocity biases.
    pub fn prepare(&mut self, bodies: &[SolverBody], config: &SolverConfig) {
        let a = &bodies[self.body_a];
        let b = &bodies[self.body_b];

        // Normal effective mass: 1 / (1/ma + 1/mb + (ra x n)^2 / Ia + (rb x n)^2 / Ib).
        self.normal_mass = compute_effective_mass(a, b, self.r_a, self.r_b, self.normal);
        self.tangent1_mass = compute_effective_mass(a, b, self.r_a, self.r_b, self.tangent1);
        self.tangent2_mass = compute_effective_mass(a, b, self.r_a, self.r_b, self.tangent2);

        // Relative velocity at contact point along normal.
        let vel_a = a.velocity_at_point(self.r_a);
        let vel_b = b.velocity_at_point(self.r_b);
        let rel_vel = sub3(vel_b, vel_a);
        let normal_vel = dot3(rel_vel, self.normal);

        // Restitution velocity bias (only for separating velocities above threshold).
        self.velocity_bias = 0.0;
        if normal_vel < -config.restitution_velocity_threshold {
            self.velocity_bias = -self.restitution * normal_vel;
        }

        // Position correction bias.
        match config.position_correction {
            PositionCorrection::Baumgarte(beta) => {
                let slop = config.penetration_slop;
                let correction = (self.depth - slop).max(0.0);
                self.position_bias = beta * correction / config.dt;
            }
            PositionCorrection::SplitImpulse(beta) => {
                let slop = config.penetration_slop;
                let correction = (self.depth - slop).max(0.0);
                self.position_bias = beta * correction / config.dt;
            }
            PositionCorrection::None => {
                self.position_bias = 0.0;
            }
        }
    }

    /// Warm start: apply cached impulses from the previous frame.
    pub fn warm_start(&mut self, bodies: &mut [SolverBody]) {
        if self.normal_impulse == 0.0 && self.tangent1_impulse == 0.0 && self.tangent2_impulse == 0.0 {
            return;
        }

        let impulse = add3(
            add3(
                scale3(self.normal, self.normal_impulse),
                scale3(self.tangent1, self.tangent1_impulse),
            ),
            scale3(self.tangent2, self.tangent2_impulse),
        );

        bodies[self.body_a].apply_impulse(neg3(impulse), self.r_a);
        bodies[self.body_b].apply_impulse(impulse, self.r_b);
        self.warm_started = true;
    }

    /// Solve the velocity constraint (normal + friction).
    pub fn solve_velocity(&mut self, bodies: &mut [SolverBody], config: &SolverConfig) {
        // --- Normal constraint ---
        let vel_a = bodies[self.body_a].velocity_at_point(self.r_a);
        let vel_b = bodies[self.body_b].velocity_at_point(self.r_b);
        let rel_vel = sub3(vel_b, vel_a);
        let normal_vel = dot3(rel_vel, self.normal);

        // Compute impulse magnitude.
        let mut lambda = self.normal_mass * (-(normal_vel - self.velocity_bias - self.position_bias));

        // Accumulate and clamp (non-negative).
        let old_impulse = self.normal_impulse;
        self.normal_impulse = (self.normal_impulse + lambda).max(0.0);
        lambda = self.normal_impulse - old_impulse;

        // Apply normal impulse.
        let normal_impulse = scale3(self.normal, lambda);
        bodies[self.body_a].apply_impulse(neg3(normal_impulse), self.r_a);
        bodies[self.body_b].apply_impulse(normal_impulse, self.r_b);

        // --- Friction constraint ---
        let max_friction = self.friction * self.normal_impulse;

        match config.friction_model {
            FrictionModel::Box => {
                self.solve_friction_box(bodies, max_friction);
            }
            FrictionModel::Cone => {
                self.solve_friction_cone(bodies, max_friction);
            }
            FrictionModel::Ellipse(aniso) => {
                self.solve_friction_ellipse(bodies, max_friction, aniso);
            }
        }
    }

    /// Solve friction using the box friction model (independent per tangent).
    fn solve_friction_box(&mut self, bodies: &mut [SolverBody], max_friction: f32) {
        // Tangent 1.
        {
            let vel_a = bodies[self.body_a].velocity_at_point(self.r_a);
            let vel_b = bodies[self.body_b].velocity_at_point(self.r_b);
            let rel_vel = sub3(vel_b, vel_a);
            let tangent_vel = dot3(rel_vel, self.tangent1);

            let mut lambda = self.tangent1_mass * (-tangent_vel);
            let old = self.tangent1_impulse;
            self.tangent1_impulse = (self.tangent1_impulse + lambda).clamp(-max_friction, max_friction);
            lambda = self.tangent1_impulse - old;

            let impulse = scale3(self.tangent1, lambda);
            bodies[self.body_a].apply_impulse(neg3(impulse), self.r_a);
            bodies[self.body_b].apply_impulse(impulse, self.r_b);
        }

        // Tangent 2.
        {
            let vel_a = bodies[self.body_a].velocity_at_point(self.r_a);
            let vel_b = bodies[self.body_b].velocity_at_point(self.r_b);
            let rel_vel = sub3(vel_b, vel_a);
            let tangent_vel = dot3(rel_vel, self.tangent2);

            let mut lambda = self.tangent2_mass * (-tangent_vel);
            let old = self.tangent2_impulse;
            self.tangent2_impulse = (self.tangent2_impulse + lambda).clamp(-max_friction, max_friction);
            lambda = self.tangent2_impulse - old;

            let impulse = scale3(self.tangent2, lambda);
            bodies[self.body_a].apply_impulse(neg3(impulse), self.r_a);
            bodies[self.body_b].apply_impulse(impulse, self.r_b);
        }
    }

    /// Solve friction using the cone friction model.
    fn solve_friction_cone(&mut self, bodies: &mut [SolverBody], max_friction: f32) {
        let vel_a = bodies[self.body_a].velocity_at_point(self.r_a);
        let vel_b = bodies[self.body_b].velocity_at_point(self.r_b);
        let rel_vel = sub3(vel_b, vel_a);

        let t1_vel = dot3(rel_vel, self.tangent1);
        let t2_vel = dot3(rel_vel, self.tangent2);

        let mut lambda1 = self.tangent1_mass * (-t1_vel);
        let mut lambda2 = self.tangent2_mass * (-t2_vel);

        let new_t1 = self.tangent1_impulse + lambda1;
        let new_t2 = self.tangent2_impulse + lambda2;

        // Project onto friction cone.
        let magnitude = (new_t1 * new_t1 + new_t2 * new_t2).sqrt();
        if magnitude > max_friction && magnitude > 1e-12 {
            let scale = max_friction / magnitude;
            let clamped_t1 = new_t1 * scale;
            let clamped_t2 = new_t2 * scale;
            lambda1 = clamped_t1 - self.tangent1_impulse;
            lambda2 = clamped_t2 - self.tangent2_impulse;
            self.tangent1_impulse = clamped_t1;
            self.tangent2_impulse = clamped_t2;
        } else {
            self.tangent1_impulse = new_t1;
            self.tangent2_impulse = new_t2;
        }

        let impulse = add3(
            scale3(self.tangent1, lambda1),
            scale3(self.tangent2, lambda2),
        );
        bodies[self.body_a].apply_impulse(neg3(impulse), self.r_a);
        bodies[self.body_b].apply_impulse(impulse, self.r_b);
    }

    /// Solve friction using an elliptical friction model (anisotropic).
    fn solve_friction_ellipse(&mut self, bodies: &mut [SolverBody], max_friction: f32, aniso: f32) {
        let vel_a = bodies[self.body_a].velocity_at_point(self.r_a);
        let vel_b = bodies[self.body_b].velocity_at_point(self.r_b);
        let rel_vel = sub3(vel_b, vel_a);

        let t1_vel = dot3(rel_vel, self.tangent1);
        let t2_vel = dot3(rel_vel, self.tangent2);

        let mut lambda1 = self.tangent1_mass * (-t1_vel);
        let mut lambda2 = self.tangent2_mass * (-t2_vel);

        let new_t1 = self.tangent1_impulse + lambda1;
        let new_t2 = self.tangent2_impulse + lambda2;

        // Project onto friction ellipse.
        let max_t1 = max_friction;
        let max_t2 = max_friction * aniso;
        let ellipse_val = if max_t1 > 0.0 && max_t2 > 0.0 {
            (new_t1 / max_t1).powi(2) + (new_t2 / max_t2).powi(2)
        } else {
            0.0
        };

        if ellipse_val > 1.0 && ellipse_val > 1e-12 {
            let scale = 1.0 / ellipse_val.sqrt();
            let clamped_t1 = new_t1 * scale;
            let clamped_t2 = new_t2 * scale;
            lambda1 = clamped_t1 - self.tangent1_impulse;
            lambda2 = clamped_t2 - self.tangent2_impulse;
            self.tangent1_impulse = clamped_t1;
            self.tangent2_impulse = clamped_t2;
        } else {
            self.tangent1_impulse = new_t1;
            self.tangent2_impulse = new_t2;
        }

        let impulse = add3(
            scale3(self.tangent1, lambda1),
            scale3(self.tangent2, lambda2),
        );
        bodies[self.body_a].apply_impulse(neg3(impulse), self.r_a);
        bodies[self.body_b].apply_impulse(impulse, self.r_b);
    }

    /// Solve position constraint (for split impulse method).
    pub fn solve_position(&mut self, bodies: &mut [SolverBody], config: &SolverConfig) {
        if !matches!(config.position_correction, PositionCorrection::SplitImpulse(_)) {
            return;
        }

        let slop = config.penetration_slop;
        let correction = (self.depth - slop).max(0.0);
        if correction < 1e-6 { return; }

        // Compute position-level effective mass.
        let pos_a = add3(bodies[self.body_a].position, bodies[self.body_a].delta_position);
        let pos_b = add3(bodies[self.body_b].position, bodies[self.body_b].delta_position);
        let _ = (pos_a, pos_b); // These would be used with full position correction.

        let beta = match config.position_correction {
            PositionCorrection::SplitImpulse(b) => b,
            _ => 0.2,
        };
        let lambda = self.normal_mass * beta * correction;

        let impulse = scale3(self.normal, lambda);
        bodies[self.body_a].apply_position_impulse(neg3(impulse), self.r_a);
        bodies[self.body_b].apply_position_impulse(impulse, self.r_b);
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Compute a tangent basis (two perpendicular vectors) from a normal.
pub fn compute_tangent_basis(normal: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let n = normal;
    let t1 = if n[0].abs() > 0.9 {
        normalize3(cross3(n, [0.0, 1.0, 0.0]))
    } else {
        normalize3(cross3(n, [1.0, 0.0, 0.0]))
    };
    let t2 = cross3(n, t1);
    (t1, t2)
}

/// Compute the effective mass for a constraint direction.
fn compute_effective_mass(
    a: &SolverBody,
    b: &SolverBody,
    r_a: [f32; 3],
    r_b: [f32; 3],
    direction: [f32; 3],
) -> f32 {
    let rn_a = cross3(r_a, direction);
    let rn_b = cross3(r_b, direction);

    let k = a.inv_mass + b.inv_mass
        + rn_a[0] * rn_a[0] * a.inv_inertia[0]
        + rn_a[1] * rn_a[1] * a.inv_inertia[1]
        + rn_a[2] * rn_a[2] * a.inv_inertia[2]
        + rn_b[0] * rn_b[0] * b.inv_inertia[0]
        + rn_b[1] * rn_b[1] * b.inv_inertia[1]
        + rn_b[2] * rn_b[2] * b.inv_inertia[2];

    if k > 1e-12 { 1.0 / k } else { 0.0 }
}

// ---------------------------------------------------------------------------
// Solver configuration
// ---------------------------------------------------------------------------

/// Friction model used by the solver.
#[derive(Debug, Clone, Copy)]
pub enum FrictionModel {
    /// Box friction: independent friction per tangent direction.
    Box,
    /// Cone friction: combined friction limited by a circular cone.
    Cone,
    /// Elliptical friction: anisotropic friction with a scaling factor.
    Ellipse(f32),
}

/// Position correction method.
#[derive(Debug, Clone, Copy)]
pub enum PositionCorrection {
    /// No position correction (allows sinking).
    None,
    /// Baumgarte stabilization: add velocity bias to prevent penetration.
    /// Parameter is the Baumgarte factor (typically 0.1-0.3).
    Baumgarte(f32),
    /// Split impulse: separate position and velocity solves.
    /// Parameter is the position correction factor.
    SplitImpulse(f32),
}

/// Configuration for the contact solver.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Number of velocity iterations.
    pub velocity_iterations: u32,
    /// Number of position iterations.
    pub position_iterations: u32,
    /// Timestep (set per step).
    pub dt: f32,
    /// Friction model.
    pub friction_model: FrictionModel,
    /// Position correction method.
    pub position_correction: PositionCorrection,
    /// Penetration slop (allowed penetration before correction).
    pub penetration_slop: f32,
    /// Restitution velocity threshold (velocities below this don't bounce).
    pub restitution_velocity_threshold: f32,
    /// Whether to enable warm starting.
    pub warm_starting: bool,
    /// Warm starting factor (0..1, typically 0.8-1.0).
    pub warm_start_factor: f32,
    /// Whether to enable convergence tracking.
    pub track_convergence: bool,
    /// Early exit threshold for convergence.
    pub convergence_threshold: f32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            velocity_iterations: 10,
            position_iterations: 4,
            dt: 1.0 / 60.0,
            friction_model: FrictionModel::Cone,
            position_correction: PositionCorrection::Baumgarte(0.2),
            penetration_slop: 0.005,
            restitution_velocity_threshold: 0.5,
            warm_starting: true,
            warm_start_factor: 0.85,
            track_convergence: false,
            convergence_threshold: 0.001,
        }
    }
}

// ---------------------------------------------------------------------------
// Solver statistics
// ---------------------------------------------------------------------------

/// Statistics from the solver.
#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    /// Number of constraints solved.
    pub constraint_count: u32,
    /// Number of velocity iterations performed.
    pub velocity_iterations: u32,
    /// Number of position iterations performed.
    pub position_iterations: u32,
    /// Convergence error per iteration (for tracking).
    pub convergence_history: Vec<f32>,
    /// Final convergence error.
    pub final_error: f32,
    /// Whether convergence was achieved (error < threshold).
    pub converged: bool,
    /// Total time spent solving in microseconds.
    pub solve_time_us: u64,
    /// Number of warm-started constraints.
    pub warm_started_count: u32,
    /// Maximum impulse applied this step.
    pub max_impulse: f32,
    /// Average impulse magnitude.
    pub avg_impulse: f32,
}

// ---------------------------------------------------------------------------
// Contact solver
// ---------------------------------------------------------------------------

/// The main contact constraint solver.
pub struct ContactSolver {
    pub config: SolverConfig,
    /// Solver bodies (temporary per frame).
    pub bodies: Vec<SolverBody>,
    /// Contact constraints.
    pub constraints: Vec<ContactConstraint>,
    /// Warm start cache: feature_id -> (normal_impulse, tangent1, tangent2).
    warm_cache: HashMap<u64, (f32, f32, f32)>,
    /// Statistics from the last solve.
    pub stats: SolverStats,
}

impl ContactSolver {
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            bodies: Vec::new(),
            constraints: Vec::new(),
            warm_cache: HashMap::new(),
            stats: SolverStats::default(),
        }
    }

    /// Clear solver state for a new frame.
    pub fn begin(&mut self) {
        // Save warm start data from previous frame.
        self.warm_cache.clear();
        for c in &self.constraints {
            if c.normal_impulse.abs() > 1e-6 || c.tangent1_impulse.abs() > 1e-6 || c.tangent2_impulse.abs() > 1e-6 {
                let key = warm_key(c.body_a as u32, c.body_b as u32, c.feature_id);
                self.warm_cache.insert(key, (c.normal_impulse, c.tangent1_impulse, c.tangent2_impulse));
            }
        }

        self.bodies.clear();
        self.constraints.clear();
        self.stats = SolverStats::default();
    }

    /// Add a solver body.
    pub fn add_body(&mut self, body: SolverBody) -> usize {
        let idx = self.bodies.len();
        self.bodies.push(body);
        idx
    }

    /// Add a contact constraint.
    pub fn add_constraint(&mut self, mut constraint: ContactConstraint) {
        // Apply warm start from cache.
        if self.config.warm_starting {
            let key = warm_key(
                self.bodies[constraint.body_a].index,
                self.bodies[constraint.body_b].index,
                constraint.feature_id,
            );
            if let Some(&(n, t1, t2)) = self.warm_cache.get(&key) {
                let factor = self.config.warm_start_factor;
                constraint.normal_impulse = n * factor;
                constraint.tangent1_impulse = t1 * factor;
                constraint.tangent2_impulse = t2 * factor;
            }
        }
        self.constraints.push(constraint);
    }

    /// Solve all constraints.
    pub fn solve(&mut self) {
        let start = std::time::Instant::now();
        let config = self.config.clone();
        self.stats.constraint_count = self.constraints.len() as u32;

        // Prepare constraints.
        for i in 0..self.constraints.len() {
            // Need to borrow bodies and constraint separately.
            let bodies_ptr = &self.bodies as *const Vec<SolverBody>;
            unsafe {
                self.constraints[i].prepare(&*bodies_ptr, &config);
            }
        }

        // Warm start.
        let mut warm_count = 0u32;
        for i in 0..self.constraints.len() {
            if self.constraints[i].normal_impulse != 0.0
                || self.constraints[i].tangent1_impulse != 0.0
                || self.constraints[i].tangent2_impulse != 0.0
            {
                let bodies_ptr = &mut self.bodies as *mut Vec<SolverBody>;
                unsafe {
                    self.constraints[i].warm_start(&mut *bodies_ptr);
                }
                warm_count += 1;
            }
        }
        self.stats.warm_started_count = warm_count;

        // Velocity iterations.
        for iter in 0..config.velocity_iterations {
            let mut max_error = 0.0f32;

            for i in 0..self.constraints.len() {
                let old_normal = self.constraints[i].normal_impulse;
                let bodies_ptr = &mut self.bodies as *mut Vec<SolverBody>;
                unsafe {
                    self.constraints[i].solve_velocity(&mut *bodies_ptr, &config);
                }
                let delta = (self.constraints[i].normal_impulse - old_normal).abs();
                max_error = max_error.max(delta);
            }

            if config.track_convergence {
                self.stats.convergence_history.push(max_error);
            }

            // Early exit if converged.
            if max_error < config.convergence_threshold {
                self.stats.velocity_iterations = iter + 1;
                self.stats.converged = true;
                break;
            }
            self.stats.velocity_iterations = iter + 1;
        }

        // Position iterations (split impulse).
        for _ in 0..config.position_iterations {
            for i in 0..self.constraints.len() {
                let bodies_ptr = &mut self.bodies as *mut Vec<SolverBody>;
                unsafe {
                    self.constraints[i].solve_position(&mut *bodies_ptr, &config);
                }
            }
            self.stats.position_iterations += 1;
        }

        // Apply position corrections.
        if matches!(config.position_correction, PositionCorrection::SplitImpulse(_)) {
            for body in &mut self.bodies {
                body.position = add3(body.position, body.delta_position);
                // Apply delta rotation would go here for full implementation.
                body.delta_position = [0.0; 3];
                body.delta_rotation = [0.0; 3];
            }
        }

        // Compute impulse statistics.
        let mut max_imp = 0.0f32;
        let mut total_imp = 0.0f32;
        for c in &self.constraints {
            let imp = c.normal_impulse.abs();
            max_imp = max_imp.max(imp);
            total_imp += imp;
        }
        self.stats.max_impulse = max_imp;
        self.stats.avg_impulse = if self.constraints.is_empty() { 0.0 } else { total_imp / self.constraints.len() as f32 };
        self.stats.final_error = self.stats.convergence_history.last().copied().unwrap_or(0.0);
        self.stats.solve_time_us = start.elapsed().as_micros() as u64;
    }

    /// Get the solved velocity for a body.
    pub fn get_body_velocity(&self, body_idx: usize) -> Option<([f32; 3], [f32; 3])> {
        self.bodies.get(body_idx).map(|b| (b.linear_velocity, b.angular_velocity))
    }

    /// Get the solved position for a body.
    pub fn get_body_position(&self, body_idx: usize) -> Option<[f32; 3]> {
        self.bodies.get(body_idx).map(|b| b.position)
    }
}

/// Compute a warm-start cache key from body indices and feature ID.
fn warm_key(body_a: u32, body_b: u32, feature_id: u32) -> u64 {
    let a = body_a.min(body_b) as u64;
    let b = body_a.max(body_b) as u64;
    (a << 40) | (b << 8) | (feature_id as u64 & 0xFF)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tangent_basis() {
        let (t1, t2) = compute_tangent_basis([0.0, 1.0, 0.0]);
        // t1 and t2 should be perpendicular to normal and each other.
        assert!(dot3(t1, [0.0, 1.0, 0.0]).abs() < 1e-6);
        assert!(dot3(t2, [0.0, 1.0, 0.0]).abs() < 1e-6);
        assert!(dot3(t1, t2).abs() < 1e-6);
    }

    #[test]
    fn test_solver_body_impulse() {
        let mut body = SolverBody::new_dynamic(
            0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            1.0, [1.0, 1.0, 1.0],
        );
        body.apply_impulse([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!((body.linear_velocity[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_static_body_no_impulse() {
        let mut body = SolverBody::new_static(0, [0.0, 0.0, 0.0]);
        body.apply_impulse([100.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert_eq!(body.linear_velocity[0], 0.0);
    }

    #[test]
    fn test_solver_two_bodies() {
        let mut solver = ContactSolver::new(SolverConfig {
            velocity_iterations: 10,
            position_iterations: 2,
            warm_starting: false,
            ..Default::default()
        });
        solver.begin();

        // Body A: dynamic, falling.
        let a = solver.add_body(SolverBody::new_dynamic(
            0, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
            [0.0, -5.0, 0.0], [0.0, 0.0, 0.0],
            1.0, [1.0, 1.0, 1.0],
        ));
        // Body B: static floor.
        let b = solver.add_body(SolverBody::new_static(1, [0.0, 0.0, 0.0]));

        // Contact: A is penetrating B.
        let constraint = ContactConstraint::new(
            a, b,
            [0.0, 1.0, 0.0], // Normal pointing up.
            [0.0, -0.5, 0.0], // Contact at bottom of A.
            [0.0, 0.0, 0.0],  // Contact at top of B.
            0.01,  // Small penetration.
            0.5,   // Friction.
            0.0,   // No restitution.
        );
        solver.add_constraint(constraint);
        solver.solve();

        // Body A's downward velocity should be reduced or zeroed.
        let (vel, _) = solver.get_body_velocity(a).unwrap();
        assert!(vel[1] >= -0.1, "Body A should not be moving down significantly: {}", vel[1]);
    }

    #[test]
    fn test_warm_starting() {
        let config = SolverConfig {
            warm_starting: true,
            warm_start_factor: 1.0,
            ..Default::default()
        };
        let mut solver = ContactSolver::new(config);

        // First solve.
        solver.begin();
        let a = solver.add_body(SolverBody::new_dynamic(
            0, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
            [0.0, -10.0, 0.0], [0.0, 0.0, 0.0],
            1.0, [1.0, 1.0, 1.0],
        ));
        let b = solver.add_body(SolverBody::new_static(1, [0.0, 0.0, 0.0]));
        let mut constraint = ContactConstraint::new(
            a, b, [0.0, 1.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.0],
            0.02, 0.5, 0.0,
        );
        constraint.feature_id = 42;
        solver.add_constraint(constraint);
        solver.solve();

        // Verify some impulse was accumulated.
        let first_impulse = solver.constraints[0].normal_impulse;
        assert!(first_impulse > 0.0);

        // Second solve -- warm starting should use cached impulse.
        solver.begin();
        let a2 = solver.add_body(SolverBody::new_dynamic(
            0, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
            [0.0, -10.0, 0.0], [0.0, 0.0, 0.0],
            1.0, [1.0, 1.0, 1.0],
        ));
        let b2 = solver.add_body(SolverBody::new_static(1, [0.0, 0.0, 0.0]));
        let mut constraint2 = ContactConstraint::new(
            a2, b2, [0.0, 1.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.0],
            0.02, 0.5, 0.0,
        );
        constraint2.feature_id = 42;
        solver.add_constraint(constraint2);

        assert_eq!(solver.stats.warm_started_count, 0); // Not solved yet.
        solver.solve();
        assert!(solver.stats.warm_started_count > 0, "Expected warm starting to be applied");
    }

    #[test]
    fn test_convergence_tracking() {
        let config = SolverConfig {
            velocity_iterations: 20,
            track_convergence: true,
            ..Default::default()
        };
        let mut solver = ContactSolver::new(config);
        solver.begin();

        let a = solver.add_body(SolverBody::new_dynamic(
            0, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
            [0.0, -5.0, 0.0], [0.0, 0.0, 0.0],
            1.0, [1.0, 1.0, 1.0],
        ));
        let b = solver.add_body(SolverBody::new_static(1, [0.0, 0.0, 0.0]));
        solver.add_constraint(ContactConstraint::new(
            a, b, [0.0, 1.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.0],
            0.01, 0.3, 0.0,
        ));
        solver.solve();

        // Convergence history should show decreasing errors.
        assert!(!solver.stats.convergence_history.is_empty());
    }
}
