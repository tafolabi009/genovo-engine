//! Rigid body dynamics, integration, constraint solving, joints, and sleeping.
//!
//! This module implements the full dynamics pipeline:
//! - Semi-implicit Euler integration with damping
//! - Sequential impulse constraint solver for contacts and joints
//! - Joint types: Fixed, Hinge, Ball
//! - Body sleeping after sustained low velocity

use glam::{Mat3, Quat, Vec3};

use crate::collision::ContactManifold;
use crate::interface::RigidBodyHandle;

// ---------------------------------------------------------------------------
// Force application mode
// ---------------------------------------------------------------------------

/// How a force or impulse is applied to a rigid body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForceMode {
    /// Continuous force (N) applied over the time step. Affected by mass.
    Force,
    /// Instantaneous impulse (N*s). Affected by mass.
    Impulse,
    /// Continuous acceleration (m/s^2). Ignores mass.
    Acceleration,
    /// Instantaneous velocity change (m/s). Ignores mass.
    VelocityChange,
}

// ---------------------------------------------------------------------------
// Sleep constants
// ---------------------------------------------------------------------------

/// Velocity threshold below which a body is a candidate for sleeping.
const SLEEP_LINEAR_THRESHOLD: f32 = 0.05;
/// Angular velocity threshold for sleep candidacy.
const SLEEP_ANGULAR_THRESHOLD: f32 = 0.05;
/// Number of frames a body must remain below thresholds before sleeping.
const SLEEP_FRAMES_REQUIRED: u32 = 60;

// ---------------------------------------------------------------------------
// Rigid body state
// ---------------------------------------------------------------------------

/// Runtime state of a rigid body in the simulation.
#[derive(Debug, Clone)]
pub struct RigidBody {
    /// The handle identifying this body within the physics world.
    pub handle: RigidBodyHandle,

    // -- Transform --
    /// World-space position of the center of mass.
    pub position: Vec3,
    /// World-space orientation.
    pub rotation: Quat,

    // -- Linear dynamics --
    /// Linear velocity in m/s.
    pub linear_velocity: Vec3,
    /// Accumulated force to be applied during the next integration step.
    pub accumulated_force: Vec3,

    // -- Angular dynamics --
    /// Angular velocity in rad/s.
    pub angular_velocity: Vec3,
    /// Accumulated torque to be applied during the next integration step.
    pub accumulated_torque: Vec3,

    // -- Mass properties --
    /// Mass in kilograms (0.0 for static bodies).
    pub mass: f32,
    /// Inverse mass (cached for performance; 0.0 for static bodies).
    pub inv_mass: f32,
    /// Inertia tensor in local space.
    pub inertia_tensor: Mat3,
    /// Inverse inertia tensor in local space.
    pub inv_inertia_tensor: Mat3,

    // -- Damping --
    /// Linear velocity damping factor [0, 1].
    pub linear_damping: f32,
    /// Angular velocity damping factor [0, 1].
    pub angular_damping: f32,

    // -- State flags --
    /// Whether the body is currently sleeping (deactivated).
    pub is_sleeping: bool,
    /// Counter of consecutive frames below sleep thresholds.
    pub sleep_counter: u32,
    /// Gravity scale multiplier (0.0 = no gravity, 1.0 = normal, 2.0 = double).
    pub gravity_scale: f32,
    /// Whether the body is static (infinite mass, never moves).
    pub is_static: bool,
    /// Whether the body is kinematic (moved programmatically, not by forces).
    pub is_kinematic: bool,
}

impl RigidBody {
    /// Apply a force according to the specified [`ForceMode`].
    pub fn apply_force(&mut self, force: Vec3, mode: ForceMode) {
        if self.is_static {
            return;
        }
        match mode {
            ForceMode::Force => {
                self.accumulated_force += force;
            }
            ForceMode::Impulse => {
                self.linear_velocity += force * self.inv_mass;
            }
            ForceMode::Acceleration => {
                self.accumulated_force += force * self.mass;
            }
            ForceMode::VelocityChange => {
                self.linear_velocity += force;
            }
        }
        self.wake_up();
    }

    /// Apply a torque according to the specified [`ForceMode`].
    pub fn apply_torque(&mut self, torque: Vec3, mode: ForceMode) {
        if self.is_static {
            return;
        }
        match mode {
            ForceMode::Force => {
                self.accumulated_torque += torque;
            }
            ForceMode::Impulse => {
                self.angular_velocity += self.world_inv_inertia() * torque;
            }
            ForceMode::Acceleration => {
                self.accumulated_torque += torque * self.mass;
            }
            ForceMode::VelocityChange => {
                self.angular_velocity += torque;
            }
        }
        self.wake_up();
    }

    /// Apply a force at a world-space point, producing both linear force and torque.
    pub fn apply_force_at_point(&mut self, force: Vec3, point: Vec3, mode: ForceMode) {
        self.apply_force(force, mode);
        let torque = (point - self.position).cross(force);
        self.apply_torque(torque, mode);
    }

    /// Wake the body from sleep.
    pub fn wake_up(&mut self) {
        self.is_sleeping = false;
        self.sleep_counter = 0;
    }

    /// Clear all accumulated forces and torques (called after integration).
    pub fn clear_forces(&mut self) {
        self.accumulated_force = Vec3::ZERO;
        self.accumulated_torque = Vec3::ZERO;
    }

    /// Get the velocity at a world-space point on the body.
    pub fn velocity_at_point(&self, point: Vec3) -> Vec3 {
        self.linear_velocity + self.angular_velocity.cross(point - self.position)
    }

    /// Kinetic energy of the body: 0.5 * m * v^2 + 0.5 * w^T * I * w
    pub fn kinetic_energy(&self) -> f32 {
        let linear = 0.5 * self.mass * self.linear_velocity.length_squared();
        let angular =
            0.5 * self.angular_velocity.dot(self.inertia_tensor * self.angular_velocity);
        linear + angular
    }

    /// Compute the world-space inverse inertia tensor.
    pub fn world_inv_inertia(&self) -> Mat3 {
        if self.is_static || self.is_kinematic {
            return Mat3::ZERO;
        }
        let rot = Mat3::from_quat(self.rotation);
        rot * self.inv_inertia_tensor * rot.transpose()
    }

    /// Check sleep candidacy and potentially put the body to sleep.
    pub fn update_sleep(&mut self) {
        if self.is_static || self.is_kinematic {
            return;
        }

        let lin_speed = self.linear_velocity.length();
        let ang_speed = self.angular_velocity.length();

        if lin_speed < SLEEP_LINEAR_THRESHOLD && ang_speed < SLEEP_ANGULAR_THRESHOLD {
            self.sleep_counter += 1;
            if self.sleep_counter >= SLEEP_FRAMES_REQUIRED {
                self.is_sleeping = true;
                self.linear_velocity = Vec3::ZERO;
                self.angular_velocity = Vec3::ZERO;
            }
        } else {
            self.sleep_counter = 0;
        }
    }

    /// Returns true if this body has zero inverse mass (static or kinematic).
    #[inline]
    pub fn is_immovable(&self) -> bool {
        self.is_static || self.is_kinematic
    }
}

impl Default for RigidBody {
    fn default() -> Self {
        Self {
            handle: RigidBodyHandle(0),
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            accumulated_force: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            accumulated_torque: Vec3::ZERO,
            mass: 1.0,
            inv_mass: 1.0,
            inertia_tensor: Mat3::IDENTITY,
            inv_inertia_tensor: Mat3::IDENTITY,
            linear_damping: 0.01,
            angular_damping: 0.05,
            is_sleeping: false,
            sleep_counter: 0,
            gravity_scale: 1.0,
            is_static: false,
            is_kinematic: false,
        }
    }
}

// ===========================================================================
// Semi-implicit Euler Integrator
// ===========================================================================

/// Integrate all bodies forward by `dt` seconds using semi-implicit Euler.
///
/// Steps:
/// 1. Apply gravity (scaled per body)
/// 2. Integrate acceleration -> velocity (using accumulated forces)
/// 3. Apply damping to velocities
/// 4. Integrate velocity -> position
/// 5. Integrate angular velocity -> orientation (quaternion update)
/// 6. Clear accumulated forces
pub fn integrate_bodies(bodies: &mut [RigidBody], gravity: Vec3, dt: f32) {
    for body in bodies.iter_mut() {
        if body.is_static || body.is_kinematic || body.is_sleeping {
            body.clear_forces();
            continue;
        }

        // 1. Apply gravity
        let gravity_force = gravity * body.mass * body.gravity_scale;
        body.accumulated_force += gravity_force;

        // 2. Integrate forces -> velocity (semi-implicit Euler: update velocity first)
        let linear_accel = body.accumulated_force * body.inv_mass;
        body.linear_velocity += linear_accel * dt;

        let world_inv_inertia = body.world_inv_inertia();
        let angular_accel = world_inv_inertia * body.accumulated_torque;
        body.angular_velocity += angular_accel * dt;

        // 3. Apply damping
        body.linear_velocity *= (1.0 - body.linear_damping).max(0.0);
        body.angular_velocity *= (1.0 - body.angular_damping).max(0.0);

        // 4. Integrate velocity -> position
        body.position += body.linear_velocity * dt;

        // 5. Integrate angular velocity -> orientation
        // dq/dt = 0.5 * omega_quat * q
        let omega = body.angular_velocity;
        let omega_quat = Quat::from_xyzw(omega.x, omega.y, omega.z, 0.0);
        let dq = omega_quat * body.rotation * 0.5;
        body.rotation = Quat::from_xyzw(
            body.rotation.x + dq.x * dt,
            body.rotation.y + dq.y * dt,
            body.rotation.z + dq.z * dt,
            body.rotation.w + dq.w * dt,
        )
        .normalize();

        // 6. Clear forces for next step
        body.clear_forces();
    }
}

// ===========================================================================
// Contact Constraint (for the sequential impulse solver)
// ===========================================================================

/// Cached data for a single contact constraint between two bodies.
#[derive(Debug, Clone)]
pub struct ContactConstraint {
    pub body_a_idx: usize,
    pub body_b_idx: usize,
    /// World-space contact point.
    pub point: Vec3,
    /// Contact normal (A -> B).
    pub normal: Vec3,
    /// Penetration depth.
    pub penetration: f32,
    /// Combined friction.
    pub friction: f32,
    /// Combined restitution.
    pub restitution: f32,
    /// Offset from body A center of mass to contact point.
    pub r_a: Vec3,
    /// Offset from body B center of mass to contact point.
    pub r_b: Vec3,
    /// Effective mass along the normal (precomputed).
    pub normal_mass: f32,
    /// Effective mass along tangent 1.
    pub tangent1_mass: f32,
    /// Effective mass along tangent 2.
    pub tangent2_mass: f32,
    /// Tangent direction 1 (perpendicular to normal).
    pub tangent1: Vec3,
    /// Tangent direction 2 (perpendicular to normal and tangent1).
    pub tangent2: Vec3,
    /// Accumulated normal impulse (for warm starting and clamping).
    pub accumulated_normal_impulse: f32,
    /// Accumulated tangent impulse 1.
    pub accumulated_tangent1_impulse: f32,
    /// Accumulated tangent impulse 2.
    pub accumulated_tangent2_impulse: f32,
    /// Velocity bias for restitution.
    pub velocity_bias: f32,
}

impl ContactConstraint {
    /// Precompute data needed for solving: effective masses, tangent directions, bias.
    pub fn pre_solve(
        &mut self,
        body_a: &RigidBody,
        body_b: &RigidBody,
        dt: f32,
        baumgarte: f32,
        slop: f32,
    ) {
        let safe_dt = dt.max(1e-6);

        self.r_a = self.point - body_a.position;
        self.r_b = self.point - body_b.position;

        let inv_mass_a = body_a.inv_mass;
        let inv_mass_b = body_b.inv_mass;
        let inv_inertia_a = body_a.world_inv_inertia();
        let inv_inertia_b = body_b.world_inv_inertia();

        // Compute effective mass for normal direction
        // K = 1/ma + 1/mb + (ra x n)^T * Ia^-1 * (ra x n) + (rb x n)^T * Ib^-1 * (rb x n)
        let rn_a = self.r_a.cross(self.normal);
        let rn_b = self.r_b.cross(self.normal);
        let k_normal = inv_mass_a
            + inv_mass_b
            + rn_a.dot(inv_inertia_a * rn_a)
            + rn_b.dot(inv_inertia_b * rn_b);
        self.normal_mass = if k_normal > 1e-10 { 1.0 / k_normal } else { 0.0 };

        // Compute tangent directions
        self.tangent1 = compute_tangent(self.normal);
        self.tangent2 = self.normal.cross(self.tangent1);

        // Effective mass for tangent 1
        let rt1_a = self.r_a.cross(self.tangent1);
        let rt1_b = self.r_b.cross(self.tangent1);
        let k_t1 = inv_mass_a
            + inv_mass_b
            + rt1_a.dot(inv_inertia_a * rt1_a)
            + rt1_b.dot(inv_inertia_b * rt1_b);
        self.tangent1_mass = if k_t1 > 1e-10 { 1.0 / k_t1 } else { 0.0 };

        // Effective mass for tangent 2
        let rt2_a = self.r_a.cross(self.tangent2);
        let rt2_b = self.r_b.cross(self.tangent2);
        let k_t2 = inv_mass_a
            + inv_mass_b
            + rt2_a.dot(inv_inertia_a * rt2_a)
            + rt2_b.dot(inv_inertia_b * rt2_b);
        self.tangent2_mass = if k_t2 > 1e-10 { 1.0 / k_t2 } else { 0.0 };

        // Velocity bias for position correction (Baumgarte stabilization)
        // and restitution
        let rel_vel = body_b.velocity_at_point(self.point)
            - body_a.velocity_at_point(self.point);
        let closing_vel = rel_vel.dot(self.normal);

        self.velocity_bias = 0.0;

        // Baumgarte stabilization: push objects apart when penetrating
        let pen_correction = (self.penetration - slop).max(0.0);
        self.velocity_bias += baumgarte * pen_correction / safe_dt;

        // Restitution (bounce) -- only apply if closing velocity is significant
        if closing_vel < -1.0 {
            self.velocity_bias += -self.restitution * closing_vel;
        }
    }
}

/// Compute a tangent vector perpendicular to the given normal.
fn compute_tangent(normal: Vec3) -> Vec3 {
    if normal.x.abs() < 0.9 {
        normal.cross(Vec3::X).normalize()
    } else {
        normal.cross(Vec3::Y).normalize()
    }
}

// ===========================================================================
// Sequential Impulse Solver
// ===========================================================================

/// Number of solver iterations per physics step.
pub const SOLVER_ITERATIONS: u32 = 10;
/// Baumgarte stabilization factor for penetration correction.
pub const BAUMGARTE_FACTOR: f32 = 0.2;
/// Penetration slop -- allowed penetration before correction kicks in.
pub const PENETRATION_SLOP: f32 = 0.005;

/// Build contact constraints from manifolds. Returns a Vec of constraints, each
/// referencing bodies by their index in the `bodies` slice.
pub fn build_contact_constraints(
    manifolds: &[ContactManifold],
    _bodies: &[RigidBody],
    body_index_map: &std::collections::HashMap<RigidBodyHandle, usize>,
) -> Vec<ContactConstraint> {
    let mut constraints = Vec::new();

    for manifold in manifolds {
        let body_a_idx = match body_index_map.get(&manifold.body_a) {
            Some(&idx) => idx,
            None => continue,
        };
        let body_b_idx = match body_index_map.get(&manifold.body_b) {
            Some(&idx) => idx,
            None => continue,
        };

        for contact in &manifold.contacts {
            constraints.push(ContactConstraint {
                body_a_idx,
                body_b_idx,
                point: contact.position,
                normal: contact.normal,
                penetration: contact.penetration_depth,
                friction: manifold.friction,
                restitution: manifold.restitution,
                r_a: Vec3::ZERO,
                r_b: Vec3::ZERO,
                normal_mass: 0.0,
                tangent1_mass: 0.0,
                tangent2_mass: 0.0,
                tangent1: Vec3::ZERO,
                tangent2: Vec3::ZERO,
                accumulated_normal_impulse: 0.0,
                accumulated_tangent1_impulse: 0.0,
                accumulated_tangent2_impulse: 0.0,
                velocity_bias: 0.0,
            });
        }
    }

    constraints
}

/// Pre-solve all contact constraints (compute effective masses, biases).
pub fn pre_solve_contacts(
    constraints: &mut [ContactConstraint],
    bodies: &[RigidBody],
    dt: f32,
) {
    for c in constraints.iter_mut() {
        let body_a = &bodies[c.body_a_idx];
        let body_b = &bodies[c.body_b_idx];
        c.pre_solve(body_a, body_b, dt, BAUMGARTE_FACTOR, PENETRATION_SLOP);
    }
}

/// Apply warm-starting impulses from the previous frame's accumulated impulses.
pub fn warm_start_contacts(constraints: &[ContactConstraint], bodies: &mut [RigidBody]) {
    for c in constraints {
        if c.accumulated_normal_impulse.abs() < 1e-10
            && c.accumulated_tangent1_impulse.abs() < 1e-10
            && c.accumulated_tangent2_impulse.abs() < 1e-10
        {
            continue;
        }

        let impulse = c.normal * c.accumulated_normal_impulse
            + c.tangent1 * c.accumulated_tangent1_impulse
            + c.tangent2 * c.accumulated_tangent2_impulse;

        let body_a = &bodies[c.body_a_idx];
        let body_b = &bodies[c.body_b_idx];

        let inv_mass_a = body_a.inv_mass;
        let inv_mass_b = body_b.inv_mass;
        let inv_inertia_a = body_a.world_inv_inertia();
        let inv_inertia_b = body_b.world_inv_inertia();
        let r_a = c.r_a;
        let r_b = c.r_b;

        // We need to split the borrow, so compute changes then apply
        let dv_a = -impulse * inv_mass_a;
        let dw_a = -(inv_inertia_a * r_a.cross(impulse));
        let dv_b = impulse * inv_mass_b;
        let dw_b = inv_inertia_b * r_b.cross(impulse);

        let body_a = &mut bodies[c.body_a_idx];
        body_a.linear_velocity += dv_a;
        body_a.angular_velocity += dw_a;

        let body_b = &mut bodies[c.body_b_idx];
        body_b.linear_velocity += dv_b;
        body_b.angular_velocity += dw_b;
    }
}

/// Solve contact constraints for one iteration. Updates velocities of all bodies.
pub fn solve_contacts_iteration(
    constraints: &mut [ContactConstraint],
    bodies: &mut [RigidBody],
) {
    for c_idx in 0..constraints.len() {
        let c = &constraints[c_idx];
        let body_a_idx = c.body_a_idx;
        let body_b_idx = c.body_b_idx;

        // Read body data
        let inv_mass_a = bodies[body_a_idx].inv_mass;
        let inv_mass_b = bodies[body_b_idx].inv_mass;
        let inv_inertia_a = bodies[body_a_idx].world_inv_inertia();
        let inv_inertia_b = bodies[body_b_idx].world_inv_inertia();
        let r_a = c.r_a;
        let r_b = c.r_b;
        let normal = c.normal;
        let tangent1 = c.tangent1;
        let tangent2 = c.tangent2;
        let normal_mass = c.normal_mass;
        let tangent1_mass = c.tangent1_mass;
        let tangent2_mass = c.tangent2_mass;
        let friction = c.friction;
        let velocity_bias = c.velocity_bias;
        let old_normal_impulse = c.accumulated_normal_impulse;
        let old_t1_impulse = c.accumulated_tangent1_impulse;
        let old_t2_impulse = c.accumulated_tangent2_impulse;

        // Compute relative velocity at contact point
        let vel_a = bodies[body_a_idx].linear_velocity
            + bodies[body_a_idx].angular_velocity.cross(r_a);
        let vel_b = bodies[body_b_idx].linear_velocity
            + bodies[body_b_idx].angular_velocity.cross(r_b);
        let rel_vel = vel_b - vel_a;

        // --- Normal impulse ---
        let vn = rel_vel.dot(normal);
        let lambda_n = normal_mass * (-vn + velocity_bias);

        // Clamp: accumulated normal impulse must be >= 0
        let new_normal_impulse = (old_normal_impulse + lambda_n).max(0.0);
        let applied_lambda_n = new_normal_impulse - old_normal_impulse;

        // Apply normal impulse
        let impulse_n = normal * applied_lambda_n;
        bodies[body_a_idx].linear_velocity -= impulse_n * inv_mass_a;
        bodies[body_a_idx].angular_velocity -= inv_inertia_a * r_a.cross(impulse_n);
        bodies[body_b_idx].linear_velocity += impulse_n * inv_mass_b;
        bodies[body_b_idx].angular_velocity += inv_inertia_b * r_b.cross(impulse_n);

        // Re-read relative velocity after normal impulse for friction
        let vel_a = bodies[body_a_idx].linear_velocity
            + bodies[body_a_idx].angular_velocity.cross(r_a);
        let vel_b = bodies[body_b_idx].linear_velocity
            + bodies[body_b_idx].angular_velocity.cross(r_b);
        let rel_vel = vel_b - vel_a;

        // --- Friction impulse (tangent 1) ---
        let vt1 = rel_vel.dot(tangent1);
        let lambda_t1 = tangent1_mass * (-vt1);
        let max_friction = friction * new_normal_impulse;
        let new_t1_impulse = (old_t1_impulse + lambda_t1).clamp(-max_friction, max_friction);
        let applied_lambda_t1 = new_t1_impulse - old_t1_impulse;

        let impulse_t1 = tangent1 * applied_lambda_t1;
        bodies[body_a_idx].linear_velocity -= impulse_t1 * inv_mass_a;
        bodies[body_a_idx].angular_velocity -= inv_inertia_a * r_a.cross(impulse_t1);
        bodies[body_b_idx].linear_velocity += impulse_t1 * inv_mass_b;
        bodies[body_b_idx].angular_velocity += inv_inertia_b * r_b.cross(impulse_t1);

        // --- Friction impulse (tangent 2) ---
        let vel_a = bodies[body_a_idx].linear_velocity
            + bodies[body_a_idx].angular_velocity.cross(r_a);
        let vel_b = bodies[body_b_idx].linear_velocity
            + bodies[body_b_idx].angular_velocity.cross(r_b);
        let rel_vel = vel_b - vel_a;
        let vt2 = rel_vel.dot(tangent2);
        let lambda_t2 = tangent2_mass * (-vt2);
        let new_t2_impulse = (old_t2_impulse + lambda_t2).clamp(-max_friction, max_friction);
        let applied_lambda_t2 = new_t2_impulse - old_t2_impulse;

        let impulse_t2 = tangent2 * applied_lambda_t2;
        bodies[body_a_idx].linear_velocity -= impulse_t2 * inv_mass_a;
        bodies[body_a_idx].angular_velocity -= inv_inertia_a * r_a.cross(impulse_t2);
        bodies[body_b_idx].linear_velocity += impulse_t2 * inv_mass_b;
        bodies[body_b_idx].angular_velocity += inv_inertia_b * r_b.cross(impulse_t2);

        // Store accumulated impulses
        let c = &mut constraints[c_idx];
        c.accumulated_normal_impulse = new_normal_impulse;
        c.accumulated_tangent1_impulse = new_t1_impulse;
        c.accumulated_tangent2_impulse = new_t2_impulse;
    }
}

/// Run the full sequential impulse solver: pre-solve, warm-start, iterate.
pub fn solve_contacts(
    manifolds: &[ContactManifold],
    bodies: &mut Vec<RigidBody>,
    body_index_map: &std::collections::HashMap<RigidBodyHandle, usize>,
    dt: f32,
    warm_start_data: &mut Vec<ContactConstraint>,
) {
    let mut constraints = build_contact_constraints(manifolds, bodies, body_index_map);

    // Transfer warm start impulses from previous frame
    if warm_start_data.len() == constraints.len() {
        for (new_c, old_c) in constraints.iter_mut().zip(warm_start_data.iter()) {
            new_c.accumulated_normal_impulse = old_c.accumulated_normal_impulse * 0.8;
            new_c.accumulated_tangent1_impulse = old_c.accumulated_tangent1_impulse * 0.8;
            new_c.accumulated_tangent2_impulse = old_c.accumulated_tangent2_impulse * 0.8;
        }
    }

    pre_solve_contacts(&mut constraints, bodies, dt);
    warm_start_contacts(&constraints, bodies);

    for _ in 0..SOLVER_ITERATIONS {
        solve_contacts_iteration(&mut constraints, bodies);
    }

    // Save constraints for next frame's warm starting
    *warm_start_data = constraints;
}

// ===========================================================================
// Joint / Constraint Types
// ===========================================================================

/// Opaque handle to a constraint in the physics world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstraintHandle(pub(crate) u64);

/// Describes the type and configuration of a physics joint.
#[derive(Debug, Clone)]
pub enum JointDesc {
    /// Hinge joint: allows rotation around a single axis.
    Hinge {
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis: Vec3,
        limits: Option<(f32, f32)>,
    },
    /// Ball-and-socket joint: allows rotation around all axes.
    Ball {
        anchor_a: Vec3,
        anchor_b: Vec3,
        cone_limit: Option<f32>,
    },
    /// Fixed joint: locks two bodies together with no relative movement.
    Fixed { anchor_a: Vec3, anchor_b: Vec3 },
    /// Slider (prismatic) joint: allows translation along a single axis.
    Slider {
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis: Vec3,
        limits: Option<(f32, f32)>,
    },
    /// Spring joint: elastic connection between two anchor points.
    Spring {
        anchor_a: Vec3,
        anchor_b: Vec3,
        rest_length: f32,
        stiffness: f32,
        damping: f32,
    },
}

/// Trait for physics constraint (joint) solvers.
pub trait Constraint: Send + Sync {
    /// The two bodies connected by this constraint.
    fn bodies(&self) -> (RigidBodyHandle, RigidBodyHandle);

    /// Prepare internal data for the solver iteration (called once per step).
    fn pre_solve(&mut self, body_a: &RigidBody, body_b: &RigidBody, dt: f32);

    /// Apply corrective impulses to satisfy the constraint (called each solver iteration).
    fn solve(&mut self, body_a: &mut RigidBody, body_b: &mut RigidBody);

    /// Whether this constraint has been broken.
    fn is_broken(&self) -> bool;

    /// Optional break force threshold in Newtons. `None` means unbreakable.
    fn break_force(&self) -> Option<f32>;
}

// ---------------------------------------------------------------------------
// Fixed Joint
// ---------------------------------------------------------------------------

/// Fixed joint: locks two bodies together with no relative movement.
pub struct FixedJoint {
    pub body_a: RigidBodyHandle,
    pub body_b: RigidBodyHandle,
    pub local_anchor_a: Vec3,
    pub local_anchor_b: Vec3,
    pub break_force_threshold: Option<f32>,
    pub broken: bool,
    // Pre-solved data
    effective_mass: Mat3,
    bias: Vec3,
    accumulated_impulse: Vec3,
    r_a: Vec3,
    r_b: Vec3,
}

impl FixedJoint {
    pub fn new(
        body_a: RigidBodyHandle,
        body_b: RigidBodyHandle,
        anchor_a: Vec3,
        anchor_b: Vec3,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: anchor_a,
            local_anchor_b: anchor_b,
            break_force_threshold: None,
            broken: false,
            effective_mass: Mat3::IDENTITY,
            bias: Vec3::ZERO,
            accumulated_impulse: Vec3::ZERO,
            r_a: Vec3::ZERO,
            r_b: Vec3::ZERO,
        }
    }
}

impl Constraint for FixedJoint {
    fn bodies(&self) -> (RigidBodyHandle, RigidBodyHandle) {
        (self.body_a, self.body_b)
    }

    fn pre_solve(&mut self, body_a: &RigidBody, body_b: &RigidBody, dt: f32) {
        let safe_dt = dt.max(1e-6);

        self.r_a = body_a.rotation * self.local_anchor_a;
        self.r_b = body_b.rotation * self.local_anchor_b;

        let world_anchor_a = body_a.position + self.r_a;
        let world_anchor_b = body_b.position + self.r_b;

        // Position error
        let error = world_anchor_b - world_anchor_a;
        self.bias = error * (BAUMGARTE_FACTOR / safe_dt);

        // Compute effective mass matrix K^-1 where
        // K = (1/ma + 1/mb)*I - [ra]x*Ia^-1*[ra]x - [rb]x*Ib^-1*[rb]x
        let inv_mass_sum = body_a.inv_mass + body_b.inv_mass;
        let inv_inertia_a = body_a.world_inv_inertia();
        let inv_inertia_b = body_b.world_inv_inertia();

        let skew_ra = skew_matrix(self.r_a);
        let skew_rb = skew_matrix(self.r_b);

        let k = Mat3::from_diagonal(Vec3::splat(inv_mass_sum))
            - skew_ra * inv_inertia_a * skew_ra.transpose()
            - skew_rb * inv_inertia_b * skew_rb.transpose();

        self.effective_mass = safe_inverse_mat3(k);
    }

    fn solve(&mut self, body_a: &mut RigidBody, body_b: &mut RigidBody) {
        if self.broken {
            return;
        }

        let vel_a = body_a.linear_velocity + body_a.angular_velocity.cross(self.r_a);
        let vel_b = body_b.linear_velocity + body_b.angular_velocity.cross(self.r_b);
        let cdot = vel_b - vel_a;

        let lambda = self.effective_mass * (-cdot + self.bias);
        self.accumulated_impulse += lambda;

        // Check break force
        if let Some(threshold) = self.break_force_threshold {
            if self.accumulated_impulse.length() > threshold {
                self.broken = true;
                return;
            }
        }

        body_a.linear_velocity -= lambda * body_a.inv_mass;
        body_a.angular_velocity -= body_a.world_inv_inertia() * self.r_a.cross(lambda);
        body_b.linear_velocity += lambda * body_b.inv_mass;
        body_b.angular_velocity += body_b.world_inv_inertia() * self.r_b.cross(lambda);
    }

    fn is_broken(&self) -> bool {
        self.broken
    }

    fn break_force(&self) -> Option<f32> {
        self.break_force_threshold
    }
}

// ---------------------------------------------------------------------------
// Ball Joint (Ball-and-Socket)
// ---------------------------------------------------------------------------

/// Ball joint: allows rotation around all axes, constrains positions to coincide.
pub struct BallJoint {
    pub body_a: RigidBodyHandle,
    pub body_b: RigidBodyHandle,
    pub local_anchor_a: Vec3,
    pub local_anchor_b: Vec3,
    pub cone_limit: Option<f32>,
    pub break_force_threshold: Option<f32>,
    pub broken: bool,
    effective_mass: Mat3,
    bias: Vec3,
    accumulated_impulse: Vec3,
    r_a: Vec3,
    r_b: Vec3,
}

impl BallJoint {
    pub fn new(
        body_a: RigidBodyHandle,
        body_b: RigidBodyHandle,
        anchor_a: Vec3,
        anchor_b: Vec3,
        cone_limit: Option<f32>,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: anchor_a,
            local_anchor_b: anchor_b,
            cone_limit,
            break_force_threshold: None,
            broken: false,
            effective_mass: Mat3::IDENTITY,
            bias: Vec3::ZERO,
            accumulated_impulse: Vec3::ZERO,
            r_a: Vec3::ZERO,
            r_b: Vec3::ZERO,
        }
    }
}

impl Constraint for BallJoint {
    fn bodies(&self) -> (RigidBodyHandle, RigidBodyHandle) {
        (self.body_a, self.body_b)
    }

    fn pre_solve(&mut self, body_a: &RigidBody, body_b: &RigidBody, dt: f32) {
        let safe_dt = dt.max(1e-6);

        self.r_a = body_a.rotation * self.local_anchor_a;
        self.r_b = body_b.rotation * self.local_anchor_b;

        let world_anchor_a = body_a.position + self.r_a;
        let world_anchor_b = body_b.position + self.r_b;

        let error = world_anchor_b - world_anchor_a;
        self.bias = error * (BAUMGARTE_FACTOR / safe_dt);

        let inv_mass_sum = body_a.inv_mass + body_b.inv_mass;
        let inv_inertia_a = body_a.world_inv_inertia();
        let inv_inertia_b = body_b.world_inv_inertia();

        let skew_ra = skew_matrix(self.r_a);
        let skew_rb = skew_matrix(self.r_b);

        let k = Mat3::from_diagonal(Vec3::splat(inv_mass_sum))
            - skew_ra * inv_inertia_a * skew_ra.transpose()
            - skew_rb * inv_inertia_b * skew_rb.transpose();

        self.effective_mass = safe_inverse_mat3(k);
    }

    fn solve(&mut self, body_a: &mut RigidBody, body_b: &mut RigidBody) {
        if self.broken {
            return;
        }

        let vel_a = body_a.linear_velocity + body_a.angular_velocity.cross(self.r_a);
        let vel_b = body_b.linear_velocity + body_b.angular_velocity.cross(self.r_b);
        let cdot = vel_b - vel_a;

        let lambda = self.effective_mass * (-cdot + self.bias);
        self.accumulated_impulse += lambda;

        if let Some(threshold) = self.break_force_threshold {
            if self.accumulated_impulse.length() > threshold {
                self.broken = true;
                return;
            }
        }

        body_a.linear_velocity -= lambda * body_a.inv_mass;
        body_a.angular_velocity -= body_a.world_inv_inertia() * self.r_a.cross(lambda);
        body_b.linear_velocity += lambda * body_b.inv_mass;
        body_b.angular_velocity += body_b.world_inv_inertia() * self.r_b.cross(lambda);
    }

    fn is_broken(&self) -> bool {
        self.broken
    }

    fn break_force(&self) -> Option<f32> {
        self.break_force_threshold
    }
}

// ---------------------------------------------------------------------------
// Hinge Joint
// ---------------------------------------------------------------------------

/// Hinge joint: constrains position and allows rotation around a single axis.
pub struct HingeJoint {
    pub body_a: RigidBodyHandle,
    pub body_b: RigidBodyHandle,
    pub local_anchor_a: Vec3,
    pub local_anchor_b: Vec3,
    /// Hinge axis in body A's local space.
    pub local_axis_a: Vec3,
    pub limits: Option<(f32, f32)>,
    pub break_force_threshold: Option<f32>,
    pub broken: bool,
    // Position constraint data
    effective_mass_pos: Mat3,
    bias_pos: Vec3,
    accumulated_impulse_pos: Vec3,
    r_a: Vec3,
    r_b: Vec3,
    // Angular constraint data (2 axes perpendicular to hinge)
    perp1: Vec3,
    perp2: Vec3,
    effective_mass_ang1: f32,
    effective_mass_ang2: f32,
    bias_ang1: f32,
    bias_ang2: f32,
    accumulated_impulse_ang1: f32,
    accumulated_impulse_ang2: f32,
}

impl HingeJoint {
    pub fn new(
        body_a: RigidBodyHandle,
        body_b: RigidBodyHandle,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis: Vec3,
        limits: Option<(f32, f32)>,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: anchor_a,
            local_anchor_b: anchor_b,
            local_axis_a: axis.normalize(),
            limits,
            break_force_threshold: None,
            broken: false,
            effective_mass_pos: Mat3::IDENTITY,
            bias_pos: Vec3::ZERO,
            accumulated_impulse_pos: Vec3::ZERO,
            r_a: Vec3::ZERO,
            r_b: Vec3::ZERO,
            perp1: Vec3::ZERO,
            perp2: Vec3::ZERO,
            effective_mass_ang1: 0.0,
            effective_mass_ang2: 0.0,
            bias_ang1: 0.0,
            bias_ang2: 0.0,
            accumulated_impulse_ang1: 0.0,
            accumulated_impulse_ang2: 0.0,
        }
    }
}

impl Constraint for HingeJoint {
    fn bodies(&self) -> (RigidBodyHandle, RigidBodyHandle) {
        (self.body_a, self.body_b)
    }

    fn pre_solve(&mut self, body_a: &RigidBody, body_b: &RigidBody, dt: f32) {
        let safe_dt = dt.max(1e-6);

        // Position constraint (same as ball joint)
        self.r_a = body_a.rotation * self.local_anchor_a;
        self.r_b = body_b.rotation * self.local_anchor_b;

        let world_anchor_a = body_a.position + self.r_a;
        let world_anchor_b = body_b.position + self.r_b;

        let error = world_anchor_b - world_anchor_a;
        self.bias_pos = error * (BAUMGARTE_FACTOR / safe_dt);

        let inv_mass_sum = body_a.inv_mass + body_b.inv_mass;
        let inv_inertia_a = body_a.world_inv_inertia();
        let inv_inertia_b = body_b.world_inv_inertia();

        let skew_ra = skew_matrix(self.r_a);
        let skew_rb = skew_matrix(self.r_b);

        let k = Mat3::from_diagonal(Vec3::splat(inv_mass_sum))
            - skew_ra * inv_inertia_a * skew_ra.transpose()
            - skew_rb * inv_inertia_b * skew_rb.transpose();

        self.effective_mass_pos = safe_inverse_mat3(k);

        // Angular constraint: restrict rotation to the hinge axis only.
        // The world-space hinge axis
        let world_axis = (body_a.rotation * self.local_axis_a).normalize();

        // Two perpendicular axes
        self.perp1 = compute_tangent(world_axis);
        self.perp2 = world_axis.cross(self.perp1);

        // Effective mass for angular constraints along perp1 and perp2
        let k1 = self.perp1.dot(inv_inertia_a * self.perp1)
            + self.perp1.dot(inv_inertia_b * self.perp1);
        self.effective_mass_ang1 = if k1 > 1e-10 { 1.0 / k1 } else { 0.0 };

        let k2 = self.perp2.dot(inv_inertia_a * self.perp2)
            + self.perp2.dot(inv_inertia_b * self.perp2);
        self.effective_mass_ang2 = if k2 > 1e-10 { 1.0 / k2 } else { 0.0 };

        // Angular error: the relative angular velocity perpendicular to hinge axis
        // should be zero. We correct the angular drift.
        let rel_omega = body_b.angular_velocity - body_a.angular_velocity;
        self.bias_ang1 = rel_omega.dot(self.perp1) * 0.0; // No position-level angular correction for velocity-only constraint
        self.bias_ang2 = rel_omega.dot(self.perp2) * 0.0;
    }

    fn solve(&mut self, body_a: &mut RigidBody, body_b: &mut RigidBody) {
        if self.broken {
            return;
        }

        // Solve position constraint
        let vel_a = body_a.linear_velocity + body_a.angular_velocity.cross(self.r_a);
        let vel_b = body_b.linear_velocity + body_b.angular_velocity.cross(self.r_b);
        let cdot = vel_b - vel_a;

        let lambda = self.effective_mass_pos * (-cdot + self.bias_pos);
        self.accumulated_impulse_pos += lambda;

        body_a.linear_velocity -= lambda * body_a.inv_mass;
        body_a.angular_velocity -= body_a.world_inv_inertia() * self.r_a.cross(lambda);
        body_b.linear_velocity += lambda * body_b.inv_mass;
        body_b.angular_velocity += body_b.world_inv_inertia() * self.r_b.cross(lambda);

        // Solve angular constraint: zero out angular velocity perpendicular to hinge axis
        let rel_omega = body_b.angular_velocity - body_a.angular_velocity;

        let cdot1 = rel_omega.dot(self.perp1);
        let lambda1 = self.effective_mass_ang1 * (-cdot1);
        self.accumulated_impulse_ang1 += lambda1;

        let impulse1 = self.perp1 * lambda1;
        body_a.angular_velocity -= body_a.world_inv_inertia() * impulse1;
        body_b.angular_velocity += body_b.world_inv_inertia() * impulse1;

        let rel_omega = body_b.angular_velocity - body_a.angular_velocity;
        let cdot2 = rel_omega.dot(self.perp2);
        let lambda2 = self.effective_mass_ang2 * (-cdot2);
        self.accumulated_impulse_ang2 += lambda2;

        let impulse2 = self.perp2 * lambda2;
        body_a.angular_velocity -= body_a.world_inv_inertia() * impulse2;
        body_b.angular_velocity += body_b.world_inv_inertia() * impulse2;

        // Check break force
        if let Some(threshold) = self.break_force_threshold {
            if self.accumulated_impulse_pos.length() > threshold {
                self.broken = true;
            }
        }
    }

    fn is_broken(&self) -> bool {
        self.broken
    }

    fn break_force(&self) -> Option<f32> {
        self.break_force_threshold
    }
}

// ---------------------------------------------------------------------------
// Spring Joint (explicit force, not impulse-based)
// ---------------------------------------------------------------------------

/// Spring joint: applies Hooke's law forces between two anchor points.
pub struct SpringJoint {
    pub body_a: RigidBodyHandle,
    pub body_b: RigidBodyHandle,
    pub local_anchor_a: Vec3,
    pub local_anchor_b: Vec3,
    pub rest_length: f32,
    pub stiffness: f32,
    pub damping: f32,
    pub break_force_threshold: Option<f32>,
    pub broken: bool,
    r_a: Vec3,
    r_b: Vec3,
}

impl SpringJoint {
    pub fn new(
        body_a: RigidBodyHandle,
        body_b: RigidBodyHandle,
        anchor_a: Vec3,
        anchor_b: Vec3,
        rest_length: f32,
        stiffness: f32,
        damping: f32,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: anchor_a,
            local_anchor_b: anchor_b,
            rest_length,
            stiffness,
            damping,
            break_force_threshold: None,
            broken: false,
            r_a: Vec3::ZERO,
            r_b: Vec3::ZERO,
        }
    }
}

impl Constraint for SpringJoint {
    fn bodies(&self) -> (RigidBodyHandle, RigidBodyHandle) {
        (self.body_a, self.body_b)
    }

    fn pre_solve(&mut self, body_a: &RigidBody, body_b: &RigidBody, _dt: f32) {
        self.r_a = body_a.rotation * self.local_anchor_a;
        self.r_b = body_b.rotation * self.local_anchor_b;
    }

    fn solve(&mut self, body_a: &mut RigidBody, body_b: &mut RigidBody) {
        if self.broken {
            return;
        }

        let world_anchor_a = body_a.position + self.r_a;
        let world_anchor_b = body_b.position + self.r_b;

        let diff = world_anchor_b - world_anchor_a;
        let dist = diff.length();
        if dist < 1e-7 {
            return;
        }

        let dir = diff / dist;
        let displacement = dist - self.rest_length;

        // Spring force: F = -k * x
        let spring_force = self.stiffness * displacement;

        // Damping force: F = -c * v_rel_along_dir
        let vel_a = body_a.velocity_at_point(world_anchor_a);
        let vel_b = body_b.velocity_at_point(world_anchor_b);
        let rel_vel = (vel_b - vel_a).dot(dir);
        let damping_force = self.damping * rel_vel;

        let total_force = (spring_force + damping_force) * dir;

        // Check break force
        if let Some(threshold) = self.break_force_threshold {
            if total_force.length() > threshold {
                self.broken = true;
                return;
            }
        }

        // Apply as impulse (force * dt approximated in a single solve call)
        // Since this is called each iteration, scale by 1/SOLVER_ITERATIONS
        let scale = 1.0 / SOLVER_ITERATIONS as f32;
        let impulse = total_force * scale;

        body_a.linear_velocity += impulse * body_a.inv_mass;
        body_a.angular_velocity += body_a.world_inv_inertia() * self.r_a.cross(impulse);
        body_b.linear_velocity -= impulse * body_b.inv_mass;
        body_b.angular_velocity -= body_b.world_inv_inertia() * self.r_b.cross(impulse);
    }

    fn is_broken(&self) -> bool {
        self.broken
    }

    fn break_force(&self) -> Option<f32> {
        self.break_force_threshold
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Compute the skew-symmetric matrix [v]x such that [v]x * u = v.cross(u).
fn skew_matrix(v: Vec3) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(0.0, v.z, -v.y),
        Vec3::new(-v.z, 0.0, v.x),
        Vec3::new(v.y, -v.x, 0.0),
    )
}

/// Safe inverse of a 3x3 matrix. Returns zero matrix if singular.
fn safe_inverse_mat3(m: Mat3) -> Mat3 {
    let det = m.determinant();
    if det.abs() < 1e-10 {
        Mat3::ZERO
    } else {
        m.inverse()
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gravity_integration() {
        let mut bodies = vec![RigidBody {
            mass: 1.0,
            inv_mass: 1.0,
            ..Default::default()
        }];

        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let dt = 1.0 / 60.0;

        // Step one frame
        integrate_bodies(&mut bodies, gravity, dt);

        // After one step, velocity should be approximately -9.81 * dt
        let expected_vel = -9.81 * dt;
        assert!(
            (bodies[0].linear_velocity.y - expected_vel).abs() < 0.1,
            "velocity y = {}, expected ~{}",
            bodies[0].linear_velocity.y,
            expected_vel
        );

        // Position should have moved by v*dt (semi-implicit Euler)
        assert!(bodies[0].position.y < 0.0);
    }

    #[test]
    fn test_integration_no_gravity() {
        let mut bodies = vec![RigidBody {
            mass: 1.0,
            inv_mass: 1.0,
            linear_velocity: Vec3::new(1.0, 0.0, 0.0),
            ..Default::default()
        }];

        let dt = 0.1;
        integrate_bodies(&mut bodies, Vec3::ZERO, dt);

        // Position should advance by roughly velocity * dt (minus damping)
        assert!(bodies[0].position.x > 0.0);
        assert!((bodies[0].position.x - 0.1).abs() < 0.02); // Small damping effect
    }

    #[test]
    fn test_static_body_no_movement() {
        let mut bodies = vec![RigidBody {
            mass: 0.0,
            inv_mass: 0.0,
            is_static: true,
            ..Default::default()
        }];

        integrate_bodies(&mut bodies, Vec3::new(0.0, -9.81, 0.0), 1.0 / 60.0);

        assert_eq!(bodies[0].position, Vec3::ZERO);
        assert_eq!(bodies[0].linear_velocity, Vec3::ZERO);
    }

    #[test]
    fn test_sleeping_body_no_movement() {
        let mut bodies = vec![RigidBody {
            mass: 1.0,
            inv_mass: 1.0,
            is_sleeping: true,
            ..Default::default()
        }];

        integrate_bodies(&mut bodies, Vec3::new(0.0, -9.81, 0.0), 1.0 / 60.0);

        assert_eq!(bodies[0].position, Vec3::ZERO);
        assert_eq!(bodies[0].linear_velocity, Vec3::ZERO);
    }

    #[test]
    fn test_force_mode_impulse() {
        let mut body = RigidBody {
            mass: 2.0,
            inv_mass: 0.5,
            ..Default::default()
        };

        body.apply_force(Vec3::new(4.0, 0.0, 0.0), ForceMode::Impulse);

        // velocity = impulse * inv_mass = 4 * 0.5 = 2.0
        assert!((body.linear_velocity.x - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_force_mode_velocity_change() {
        let mut body = RigidBody {
            mass: 10.0,
            inv_mass: 0.1,
            ..Default::default()
        };

        body.apply_force(Vec3::new(5.0, 0.0, 0.0), ForceMode::VelocityChange);

        // Direct velocity change, ignoring mass
        assert!((body.linear_velocity.x - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_quaternion_integration() {
        let mut bodies = vec![RigidBody {
            mass: 1.0,
            inv_mass: 1.0,
            angular_velocity: Vec3::new(0.0, 1.0, 0.0), // Spin around Y
            ..Default::default()
        }];

        let dt = 0.1;
        integrate_bodies(&mut bodies, Vec3::ZERO, dt);

        // Rotation should have changed from identity
        assert!(bodies[0].rotation != Quat::IDENTITY);
        // And should still be normalized
        assert!((bodies[0].rotation.length() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_sleep_counter() {
        let mut body = RigidBody {
            mass: 1.0,
            inv_mass: 1.0,
            linear_velocity: Vec3::new(0.01, 0.0, 0.0), // Below threshold
            angular_velocity: Vec3::ZERO,
            ..Default::default()
        };

        // Not sleeping yet
        assert!(!body.is_sleeping);

        for _ in 0..59 {
            body.update_sleep();
        }
        assert!(!body.is_sleeping);
        assert_eq!(body.sleep_counter, 59);

        body.update_sleep(); // 60th frame
        assert!(body.is_sleeping);
    }

    #[test]
    fn test_wake_up_resets_counter() {
        let mut body = RigidBody {
            sleep_counter: 50,
            ..Default::default()
        };

        body.wake_up();
        assert_eq!(body.sleep_counter, 0);
        assert!(!body.is_sleeping);
    }

    #[test]
    fn test_velocity_at_point() {
        let body = RigidBody {
            position: Vec3::ZERO,
            linear_velocity: Vec3::new(1.0, 0.0, 0.0),
            angular_velocity: Vec3::new(0.0, 0.0, 1.0),
            ..Default::default()
        };

        // Point at (0, 1, 0): v = (1,0,0) + (0,0,1) x (0,1,0) = (1,0,0) + (-1,0,0) = (0,0,0)
        let v = body.velocity_at_point(Vec3::new(0.0, 1.0, 0.0));
        assert!((v - Vec3::ZERO).length() < 1e-4);
    }

    #[test]
    fn test_kinetic_energy() {
        let body = RigidBody {
            mass: 2.0,
            linear_velocity: Vec3::new(3.0, 0.0, 0.0),
            angular_velocity: Vec3::ZERO,
            ..Default::default()
        };

        // KE = 0.5 * 2 * 9 = 9
        assert!((body.kinetic_energy() - 9.0).abs() < 1e-4);
    }

    #[test]
    fn test_contact_constraint_normal_impulse() {
        // Two bodies: A at origin, B offset along X. Contact at the midpoint.
        let mut bodies = vec![
            RigidBody {
                handle: RigidBodyHandle(0),
                position: Vec3::new(0.0, 0.0, 0.0),
                mass: 1.0,
                inv_mass: 1.0,
                linear_velocity: Vec3::new(1.0, 0.0, 0.0), // Moving toward B
                ..Default::default()
            },
            RigidBody {
                handle: RigidBodyHandle(1),
                position: Vec3::new(2.0, 0.0, 0.0),
                mass: 1.0,
                inv_mass: 1.0,
                linear_velocity: Vec3::new(-1.0, 0.0, 0.0), // Moving toward A
                ..Default::default()
            },
        ];

        let mut c = ContactConstraint {
            body_a_idx: 0,
            body_b_idx: 1,
            point: Vec3::new(1.0, 0.0, 0.0),
            normal: Vec3::X,
            penetration: 0.01,
            friction: 0.5,
            restitution: 0.0,
            r_a: Vec3::ZERO,
            r_b: Vec3::ZERO,
            normal_mass: 0.0,
            tangent1_mass: 0.0,
            tangent2_mass: 0.0,
            tangent1: Vec3::ZERO,
            tangent2: Vec3::ZERO,
            accumulated_normal_impulse: 0.0,
            accumulated_tangent1_impulse: 0.0,
            accumulated_tangent2_impulse: 0.0,
            velocity_bias: 0.0,
        };

        c.pre_solve(&bodies[0], &bodies[1], 1.0 / 60.0, BAUMGARTE_FACTOR, PENETRATION_SLOP);

        // Normal mass for two equal-mass bodies with no rotation:
        // K = 1/m_a + 1/m_b = 2.0, so effective mass = 0.5
        assert!((c.normal_mass - 0.5).abs() < 1e-4);

        // After solving, bodies should stop moving toward each other
        let mut constraints = vec![c];
        for _ in 0..SOLVER_ITERATIONS {
            solve_contacts_iteration(&mut constraints, &mut bodies);
        }

        // Bodies should no longer have closing velocity
        let rel_vel = bodies[1].linear_velocity.x - bodies[0].linear_velocity.x;
        assert!(rel_vel >= -0.01, "rel_vel = {}", rel_vel);
    }

    #[test]
    fn test_contact_constraint_with_restitution() {
        let mut bodies = vec![
            RigidBody {
                handle: RigidBodyHandle(0),
                position: Vec3::ZERO,
                mass: 1.0,
                inv_mass: 1.0,
                linear_velocity: Vec3::new(5.0, 0.0, 0.0),
                ..Default::default()
            },
            RigidBody {
                handle: RigidBodyHandle(1),
                position: Vec3::new(2.0, 0.0, 0.0),
                mass: 1.0,
                inv_mass: 1.0,
                linear_velocity: Vec3::new(-5.0, 0.0, 0.0),
                ..Default::default()
            },
        ];

        let mut c = ContactConstraint {
            body_a_idx: 0,
            body_b_idx: 1,
            point: Vec3::new(1.0, 0.0, 0.0),
            normal: Vec3::X,
            penetration: 0.01,
            friction: 0.0,
            restitution: 1.0, // Perfect bounce
            r_a: Vec3::ZERO,
            r_b: Vec3::ZERO,
            normal_mass: 0.0,
            tangent1_mass: 0.0,
            tangent2_mass: 0.0,
            tangent1: Vec3::ZERO,
            tangent2: Vec3::ZERO,
            accumulated_normal_impulse: 0.0,
            accumulated_tangent1_impulse: 0.0,
            accumulated_tangent2_impulse: 0.0,
            velocity_bias: 0.0,
        };

        c.pre_solve(&bodies[0], &bodies[1], 1.0 / 60.0, BAUMGARTE_FACTOR, PENETRATION_SLOP);

        let mut constraints = vec![c];
        for _ in 0..SOLVER_ITERATIONS {
            solve_contacts_iteration(&mut constraints, &mut bodies);
        }

        // With perfect restitution, bodies should bounce back
        // Body A was moving +X, should now move -X (or at least reduced significantly)
        // Body B was moving -X, should now move +X
        assert!(
            bodies[0].linear_velocity.x < 0.0,
            "A velocity = {}",
            bodies[0].linear_velocity.x
        );
        assert!(
            bodies[1].linear_velocity.x > 0.0,
            "B velocity = {}",
            bodies[1].linear_velocity.x
        );
    }

    #[test]
    fn test_damping() {
        let mut bodies = vec![RigidBody {
            mass: 1.0,
            inv_mass: 1.0,
            linear_velocity: Vec3::new(10.0, 0.0, 0.0),
            linear_damping: 0.1,
            ..Default::default()
        }];

        let initial_speed = bodies[0].linear_velocity.length();
        integrate_bodies(&mut bodies, Vec3::ZERO, 1.0 / 60.0);
        let final_speed = bodies[0].linear_velocity.length();

        assert!(final_speed < initial_speed);
    }

    #[test]
    fn test_skew_matrix() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let u = Vec3::new(4.0, 5.0, 6.0);
        let cross = v.cross(u);
        let skew_result = skew_matrix(v) * u;
        assert!((cross - skew_result).length() < 1e-4);
    }
}
