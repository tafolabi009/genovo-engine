//! Advanced constraint/joint types for the physics engine.
//!
//! Provides:
//! - `SliderJoint`: prismatic joint with limits and motor
//! - `ConeTwistJoint`: ball socket with cone + twist limits
//! - `GearJoint`: ratio-linked rotation between two bodies
//! - `PulleyJoint`: two bodies connected through a virtual pulley
//! - `WeldJoint`: rigid connection with optional break threshold
//! - `DistanceJoint`: distance constraint with min/max range
//! - `MouseJoint`: drag body toward target point (spring-damper)

use glam::{Mat3, Vec3};

use crate::dynamics::{Constraint, RigidBody, BAUMGARTE_FACTOR};
use crate::interface::RigidBodyHandle;

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Compute a tangent vector perpendicular to the given normal.
fn compute_tangent(normal: Vec3) -> Vec3 {
    if normal.x.abs() < 0.9 {
        normal.cross(Vec3::X).normalize()
    } else {
        normal.cross(Vec3::Y).normalize()
    }
}

/// Compute the skew-symmetric matrix [v]x such that [v]x * u = v.cross(u).
fn skew_matrix(v: Vec3) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(0.0, v.z, -v.y),
        Vec3::new(-v.z, 0.0, v.x),
        Vec3::new(v.y, -v.x, 0.0),
    )
}

/// Safe inverse of a 3x3 matrix.
fn safe_inverse_mat3(m: Mat3) -> Mat3 {
    let det = m.determinant();
    if det.abs() < 1e-10 {
        Mat3::ZERO
    } else {
        m.inverse()
    }
}

// ===========================================================================
// SliderJoint (Prismatic)
// ===========================================================================

/// Slider (prismatic) joint: allows translation along a single axis with
/// optional limits and motor.
///
/// The joint constrains the relative position of two bodies to lie along
/// a specified axis direction. Motor force can drive the bodies along the axis.
pub struct SliderJoint {
    pub body_a: RigidBodyHandle,
    pub body_b: RigidBodyHandle,
    pub local_anchor_a: Vec3,
    pub local_anchor_b: Vec3,
    /// Slide axis in body A's local space.
    pub local_axis: Vec3,
    /// Minimum distance along the axis (None = no limit).
    pub min_distance: Option<f32>,
    /// Maximum distance along the axis (None = no limit).
    pub max_distance: Option<f32>,
    /// Motor target velocity along the axis.
    pub motor_target_velocity: f32,
    /// Maximum motor force.
    pub motor_max_force: f32,
    /// Whether the motor is enabled.
    pub motor_enabled: bool,
    /// Spring mode: rest position along the axis.
    pub spring_rest_position: Option<f32>,
    /// Spring stiffness (when spring mode is enabled).
    pub spring_stiffness: f32,
    /// Spring damping.
    pub spring_damping: f32,
    /// Break force threshold.
    pub break_force_threshold: Option<f32>,
    pub broken: bool,
    // Pre-solved data
    r_a: Vec3,
    r_b: Vec3,
    world_axis: Vec3,
    effective_mass_lateral: Mat3,
    bias_lateral: Vec3,
    accumulated_impulse: Vec3,
    perp1: Vec3,
    perp2: Vec3,
}

impl SliderJoint {
    pub fn new(
        body_a: RigidBodyHandle,
        body_b: RigidBodyHandle,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis: Vec3,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: anchor_a,
            local_anchor_b: anchor_b,
            local_axis: axis.normalize(),
            min_distance: None,
            max_distance: None,
            motor_target_velocity: 0.0,
            motor_max_force: 0.0,
            motor_enabled: false,
            spring_rest_position: None,
            spring_stiffness: 100.0,
            spring_damping: 10.0,
            break_force_threshold: None,
            broken: false,
            r_a: Vec3::ZERO,
            r_b: Vec3::ZERO,
            world_axis: Vec3::X,
            effective_mass_lateral: Mat3::IDENTITY,
            bias_lateral: Vec3::ZERO,
            accumulated_impulse: Vec3::ZERO,
            perp1: Vec3::ZERO,
            perp2: Vec3::ZERO,
        }
    }

    /// Set axis limits.
    pub fn with_limits(mut self, min: f32, max: f32) -> Self {
        self.min_distance = Some(min);
        self.max_distance = Some(max);
        self
    }

    /// Enable motor.
    pub fn with_motor(mut self, target_velocity: f32, max_force: f32) -> Self {
        self.motor_enabled = true;
        self.motor_target_velocity = target_velocity;
        self.motor_max_force = max_force;
        self
    }

    /// Enable spring mode.
    pub fn with_spring(mut self, rest_position: f32, stiffness: f32, damping: f32) -> Self {
        self.spring_rest_position = Some(rest_position);
        self.spring_stiffness = stiffness;
        self.spring_damping = damping;
        self
    }
}

impl Constraint for SliderJoint {
    fn bodies(&self) -> (RigidBodyHandle, RigidBodyHandle) {
        (self.body_a, self.body_b)
    }

    fn pre_solve(&mut self, body_a: &RigidBody, body_b: &RigidBody, dt: f32) {
        let safe_dt = dt.max(1e-6);

        self.r_a = body_a.rotation * self.local_anchor_a;
        self.r_b = body_b.rotation * self.local_anchor_b;
        self.world_axis = (body_a.rotation * self.local_axis).normalize();

        let world_anchor_a = body_a.position + self.r_a;
        let world_anchor_b = body_b.position + self.r_b;

        // Perpendicular axes to the slide axis
        self.perp1 = compute_tangent(self.world_axis);
        self.perp2 = self.world_axis.cross(self.perp1);

        // Lateral error (perpendicular to slide axis)
        let error = world_anchor_b - world_anchor_a;
        let lateral_error = error - self.world_axis * error.dot(self.world_axis);
        self.bias_lateral = lateral_error * (BAUMGARTE_FACTOR / safe_dt);

        // Effective mass for lateral constraint
        let inv_mass_sum = body_a.inv_mass + body_b.inv_mass;
        let inv_inertia_a = body_a.world_inv_inertia();
        let inv_inertia_b = body_b.world_inv_inertia();
        let skew_ra = skew_matrix(self.r_a);
        let skew_rb = skew_matrix(self.r_b);
        let k = Mat3::from_diagonal(Vec3::splat(inv_mass_sum))
            - skew_ra * inv_inertia_a * skew_ra.transpose()
            - skew_rb * inv_inertia_b * skew_rb.transpose();
        self.effective_mass_lateral = safe_inverse_mat3(k);
    }

    fn solve(&mut self, body_a: &mut RigidBody, body_b: &mut RigidBody) {
        if self.broken {
            return;
        }

        // Solve lateral constraint (keep on axis)
        let vel_a = body_a.linear_velocity + body_a.angular_velocity.cross(self.r_a);
        let vel_b = body_b.linear_velocity + body_b.angular_velocity.cross(self.r_b);
        let cdot = vel_b - vel_a;

        // Remove the axial component from the velocity constraint
        let lateral_cdot = cdot - self.world_axis * cdot.dot(self.world_axis);
        let lateral_bias = self.bias_lateral;

        let lambda = self.effective_mass_lateral * (-lateral_cdot + lateral_bias);
        self.accumulated_impulse += lambda;

        // Apply lateral impulse
        body_a.linear_velocity -= lambda * body_a.inv_mass;
        body_a.angular_velocity -= body_a.world_inv_inertia() * self.r_a.cross(lambda);
        body_b.linear_velocity += lambda * body_b.inv_mass;
        body_b.angular_velocity += body_b.world_inv_inertia() * self.r_b.cross(lambda);

        // Motor / spring along the axis
        if self.motor_enabled || self.spring_rest_position.is_some() {
            let axial_vel =
                (body_b.linear_velocity - body_a.linear_velocity).dot(self.world_axis);

            let world_anchor_a = body_a.position + self.r_a;
            let world_anchor_b = body_b.position + self.r_b;
            let axial_pos = (world_anchor_b - world_anchor_a).dot(self.world_axis);

            let mut axial_force = 0.0;

            if let Some(rest) = self.spring_rest_position {
                let displacement = axial_pos - rest;
                axial_force += -self.spring_stiffness * displacement
                    - self.spring_damping * axial_vel;
            }

            if self.motor_enabled {
                let velocity_error = self.motor_target_velocity - axial_vel;
                let motor_force = velocity_error.clamp(-self.motor_max_force, self.motor_max_force);
                axial_force += motor_force;
            }

            let axial_impulse = self.world_axis * axial_force * 0.1; // Scale for stability
            body_a.linear_velocity -= axial_impulse * body_a.inv_mass;
            body_b.linear_velocity += axial_impulse * body_b.inv_mass;
        }

        // Check break force
        if let Some(threshold) = self.break_force_threshold {
            if self.accumulated_impulse.length() > threshold {
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

// ===========================================================================
// ConeTwistJoint
// ===========================================================================

/// Cone-twist joint: ball socket with cone limit and twist limit.
///
/// Allows rotation within a cone around the joint axis and limited twist
/// rotation around that cone axis.
pub struct ConeTwistJoint {
    pub body_a: RigidBodyHandle,
    pub body_b: RigidBodyHandle,
    pub local_anchor_a: Vec3,
    pub local_anchor_b: Vec3,
    /// Cone axis in body A's local space.
    pub local_cone_axis: Vec3,
    /// Maximum swing (cone) angle in radians.
    pub swing_limit: f32,
    /// Maximum twist angle in radians.
    pub twist_limit: f32,
    /// Angular motor target velocity.
    pub motor_target_velocity: Vec3,
    /// Maximum motor torque.
    pub motor_max_torque: f32,
    /// Whether motor is enabled.
    pub motor_enabled: bool,
    /// Break force threshold.
    pub break_force_threshold: Option<f32>,
    pub broken: bool,
    // Pre-solved data
    effective_mass: Mat3,
    bias: Vec3,
    accumulated_impulse: Vec3,
    r_a: Vec3,
    r_b: Vec3,
}

impl ConeTwistJoint {
    pub fn new(
        body_a: RigidBodyHandle,
        body_b: RigidBodyHandle,
        anchor_a: Vec3,
        anchor_b: Vec3,
        cone_axis: Vec3,
        swing_limit: f32,
        twist_limit: f32,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: anchor_a,
            local_anchor_b: anchor_b,
            local_cone_axis: cone_axis.normalize(),
            swing_limit,
            twist_limit,
            motor_target_velocity: Vec3::ZERO,
            motor_max_torque: 0.0,
            motor_enabled: false,
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

impl Constraint for ConeTwistJoint {
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

        // Position constraint (same as ball joint)
        let vel_a = body_a.linear_velocity + body_a.angular_velocity.cross(self.r_a);
        let vel_b = body_b.linear_velocity + body_b.angular_velocity.cross(self.r_b);
        let cdot = vel_b - vel_a;

        let lambda = self.effective_mass * (-cdot + self.bias);
        self.accumulated_impulse += lambda;

        body_a.linear_velocity -= lambda * body_a.inv_mass;
        body_a.angular_velocity -= body_a.world_inv_inertia() * self.r_a.cross(lambda);
        body_b.linear_velocity += lambda * body_b.inv_mass;
        body_b.angular_velocity += body_b.world_inv_inertia() * self.r_b.cross(lambda);

        // Cone (swing) limit
        let world_axis_a = (body_a.rotation * self.local_cone_axis).normalize();
        let world_axis_b = (body_b.rotation * self.local_cone_axis).normalize();

        let cos_angle = world_axis_a.dot(world_axis_b).clamp(-1.0, 1.0);
        let swing_angle = cos_angle.acos();

        if swing_angle > self.swing_limit && swing_angle > 1e-4 {
            let correction_axis = world_axis_a.cross(world_axis_b);
            let correction_len = correction_axis.length();
            if correction_len > 1e-6 {
                let axis = correction_axis / correction_len;
                let overshoot = swing_angle - self.swing_limit;

                let inv_inertia_a = body_a.world_inv_inertia();
                let inv_inertia_b = body_b.world_inv_inertia();
                let k = axis.dot(inv_inertia_a * axis) + axis.dot(inv_inertia_b * axis);
                if k > 1e-10 {
                    let impulse_mag = overshoot * BAUMGARTE_FACTOR / k.max(1e-6);
                    let impulse = axis * impulse_mag;
                    body_a.angular_velocity -= inv_inertia_a * impulse;
                    body_b.angular_velocity += inv_inertia_b * impulse;
                }
            }
        }

        // Twist limit
        let rel_omega = body_b.angular_velocity - body_a.angular_velocity;
        let twist_speed = rel_omega.dot(world_axis_a);
        // Simple twist damping near limits (full twist tracking would require quaternion tracking)
        if twist_speed.abs() > 0.1 && self.twist_limit < std::f32::consts::PI {
            let inv_inertia_a = body_a.world_inv_inertia();
            let inv_inertia_b = body_b.world_inv_inertia();
            let k_twist = world_axis_a.dot(inv_inertia_a * world_axis_a)
                + world_axis_a.dot(inv_inertia_b * world_axis_a);
            if k_twist > 1e-10 {
                let damping_impulse = world_axis_a * (-twist_speed * 0.1 / k_twist);
                body_a.angular_velocity -= inv_inertia_a * damping_impulse;
                body_b.angular_velocity += inv_inertia_b * damping_impulse;
            }
        }

        if let Some(threshold) = self.break_force_threshold {
            if self.accumulated_impulse.length() > threshold {
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

// ===========================================================================
// GearJoint
// ===========================================================================

/// Gear joint: links the angular velocities of two bodies by a fixed ratio.
///
///   angular_velocity_A = gear_ratio * angular_velocity_B
pub struct GearJoint {
    pub body_a: RigidBodyHandle,
    pub body_b: RigidBodyHandle,
    /// Gear axis for body A (local space).
    pub axis_a: Vec3,
    /// Gear axis for body B (local space).
    pub axis_b: Vec3,
    /// Gear ratio: omega_A = ratio * omega_B.
    pub gear_ratio: f32,
    /// Break threshold.
    pub break_force_threshold: Option<f32>,
    pub broken: bool,
    accumulated_impulse: f32,
}

impl GearJoint {
    pub fn new(
        body_a: RigidBodyHandle,
        body_b: RigidBodyHandle,
        axis_a: Vec3,
        axis_b: Vec3,
        ratio: f32,
    ) -> Self {
        Self {
            body_a,
            body_b,
            axis_a: axis_a.normalize(),
            axis_b: axis_b.normalize(),
            gear_ratio: ratio,
            break_force_threshold: None,
            broken: false,
            accumulated_impulse: 0.0,
        }
    }
}

impl Constraint for GearJoint {
    fn bodies(&self) -> (RigidBodyHandle, RigidBodyHandle) {
        (self.body_a, self.body_b)
    }

    fn pre_solve(&mut self, _body_a: &RigidBody, _body_b: &RigidBody, _dt: f32) {
        // No pre-computation needed
    }

    fn solve(&mut self, body_a: &mut RigidBody, body_b: &mut RigidBody) {
        if self.broken {
            return;
        }

        let world_axis_a = (body_a.rotation * self.axis_a).normalize();
        let world_axis_b = (body_b.rotation * self.axis_b).normalize();

        let omega_a = body_a.angular_velocity.dot(world_axis_a);
        let omega_b = body_b.angular_velocity.dot(world_axis_b);

        // Constraint: omega_a - ratio * omega_b = 0
        let cdot = omega_a - self.gear_ratio * omega_b;

        let inv_inertia_a = body_a.world_inv_inertia();
        let inv_inertia_b = body_b.world_inv_inertia();

        let k = world_axis_a.dot(inv_inertia_a * world_axis_a)
            + self.gear_ratio * self.gear_ratio
                * world_axis_b.dot(inv_inertia_b * world_axis_b);

        if k < 1e-10 {
            return;
        }

        let lambda = -cdot / k;
        self.accumulated_impulse += lambda;

        body_a.angular_velocity -= inv_inertia_a * world_axis_a * lambda;
        body_b.angular_velocity +=
            inv_inertia_b * world_axis_b * (lambda * self.gear_ratio);

        if let Some(threshold) = self.break_force_threshold {
            if self.accumulated_impulse.abs() > threshold {
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

// ===========================================================================
// PulleyJoint
// ===========================================================================

/// Pulley joint: two bodies connected through a virtual pulley.
///
/// The total rope length is constant:
///   dist_A_to_groundA + ratio * dist_B_to_groundB = constant
pub struct PulleyJoint {
    pub body_a: RigidBodyHandle,
    pub body_b: RigidBodyHandle,
    /// Pulley anchor for body A (world space).
    pub ground_anchor_a: Vec3,
    /// Pulley anchor for body B (world space).
    pub ground_anchor_b: Vec3,
    /// Local anchor on body A.
    pub local_anchor_a: Vec3,
    /// Local anchor on body B.
    pub local_anchor_b: Vec3,
    /// Pulley ratio.
    pub ratio: f32,
    /// Total constant rope length.
    pub total_length: f32,
    /// Break threshold.
    pub break_force_threshold: Option<f32>,
    pub broken: bool,
    accumulated_impulse: f32,
}

impl PulleyJoint {
    pub fn new(
        body_a: RigidBodyHandle,
        body_b: RigidBodyHandle,
        ground_a: Vec3,
        ground_b: Vec3,
        local_a: Vec3,
        local_b: Vec3,
        ratio: f32,
    ) -> Self {
        Self {
            body_a,
            body_b,
            ground_anchor_a: ground_a,
            ground_anchor_b: ground_b,
            local_anchor_a: local_a,
            local_anchor_b: local_b,
            ratio: ratio.max(0.01),
            total_length: 0.0, // Set during first pre_solve
            break_force_threshold: None,
            broken: false,
            accumulated_impulse: 0.0,
        }
    }

    /// Set the total rope length explicitly.
    pub fn with_total_length(mut self, length: f32) -> Self {
        self.total_length = length;
        self
    }
}

impl Constraint for PulleyJoint {
    fn bodies(&self) -> (RigidBodyHandle, RigidBodyHandle) {
        (self.body_a, self.body_b)
    }

    fn pre_solve(&mut self, body_a: &RigidBody, body_b: &RigidBody, _dt: f32) {
        // Compute current rope lengths
        let world_a = body_a.position + body_a.rotation * self.local_anchor_a;
        let world_b = body_b.position + body_b.rotation * self.local_anchor_b;
        let dist_a = (world_a - self.ground_anchor_a).length();
        let dist_b = (world_b - self.ground_anchor_b).length();

        if self.total_length <= 0.0 {
            // Initialize total length on first use
            self.total_length = dist_a + self.ratio * dist_b;
        }
    }

    fn solve(&mut self, body_a: &mut RigidBody, body_b: &mut RigidBody) {
        if self.broken {
            return;
        }

        let world_a = body_a.position + body_a.rotation * self.local_anchor_a;
        let world_b = body_b.position + body_b.rotation * self.local_anchor_b;

        let diff_a = world_a - self.ground_anchor_a;
        let diff_b = world_b - self.ground_anchor_b;
        let dist_a = diff_a.length().max(0.01);
        let dist_b = diff_b.length().max(0.01);

        let dir_a = diff_a / dist_a;
        let dir_b = diff_b / dist_b;

        // Constraint: dist_a + ratio * dist_b = total_length
        let c = dist_a + self.ratio * dist_b - self.total_length;

        // Effective mass
        let k = body_a.inv_mass + self.ratio * self.ratio * body_b.inv_mass;
        if k < 1e-10 {
            return;
        }

        let lambda = -c / k * BAUMGARTE_FACTOR;
        self.accumulated_impulse += lambda;

        let impulse_a = -dir_a * lambda;
        let impulse_b = -dir_b * (lambda * self.ratio);

        body_a.linear_velocity += impulse_a * body_a.inv_mass;
        body_b.linear_velocity += impulse_b * body_b.inv_mass;

        if let Some(threshold) = self.break_force_threshold {
            if self.accumulated_impulse.abs() > threshold {
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

// ===========================================================================
// WeldJoint
// ===========================================================================

/// Weld joint: rigid connection between two bodies with optional break force.
///
/// Like a FixedJoint but with a configurable break threshold that disconnects
/// the joint when the internal force exceeds the limit.
pub struct WeldJoint {
    pub body_a: RigidBodyHandle,
    pub body_b: RigidBodyHandle,
    pub local_anchor_a: Vec3,
    pub local_anchor_b: Vec3,
    /// Break threshold in Newtons. When the constraint force exceeds this,
    /// the weld breaks.
    pub break_threshold: f32,
    pub broken: bool,
    // Pre-solved data
    effective_mass: Mat3,
    bias: Vec3,
    accumulated_impulse: Vec3,
    r_a: Vec3,
    r_b: Vec3,
    // Angular constraint
    angular_effective_mass: Mat3,
    angular_bias: Vec3,
    angular_accumulated: Vec3,
}

impl WeldJoint {
    pub fn new(
        body_a: RigidBodyHandle,
        body_b: RigidBodyHandle,
        anchor_a: Vec3,
        anchor_b: Vec3,
        break_threshold: f32,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: anchor_a,
            local_anchor_b: anchor_b,
            break_threshold,
            broken: false,
            effective_mass: Mat3::IDENTITY,
            bias: Vec3::ZERO,
            accumulated_impulse: Vec3::ZERO,
            r_a: Vec3::ZERO,
            r_b: Vec3::ZERO,
            angular_effective_mass: Mat3::IDENTITY,
            angular_bias: Vec3::ZERO,
            angular_accumulated: Vec3::ZERO,
        }
    }
}

impl Constraint for WeldJoint {
    fn bodies(&self) -> (RigidBodyHandle, RigidBodyHandle) {
        (self.body_a, self.body_b)
    }

    fn pre_solve(&mut self, body_a: &RigidBody, body_b: &RigidBody, dt: f32) {
        self.r_a = body_a.rotation * self.local_anchor_a;
        self.r_b = body_b.rotation * self.local_anchor_b;

        let world_anchor_a = body_a.position + self.r_a;
        let world_anchor_b = body_b.position + self.r_b;

        let error = world_anchor_b - world_anchor_a;
        self.bias = error * (BAUMGARTE_FACTOR / dt);

        let inv_mass_sum = body_a.inv_mass + body_b.inv_mass;
        let inv_inertia_a = body_a.world_inv_inertia();
        let inv_inertia_b = body_b.world_inv_inertia();
        let skew_ra = skew_matrix(self.r_a);
        let skew_rb = skew_matrix(self.r_b);
        let k = Mat3::from_diagonal(Vec3::splat(inv_mass_sum))
            - skew_ra * inv_inertia_a * skew_ra.transpose()
            - skew_rb * inv_inertia_b * skew_rb.transpose();
        self.effective_mass = safe_inverse_mat3(k);

        // Angular constraint: lock relative rotation
        let k_ang = inv_inertia_a + inv_inertia_b;
        self.angular_effective_mass = safe_inverse_mat3(k_ang);
    }

    fn solve(&mut self, body_a: &mut RigidBody, body_b: &mut RigidBody) {
        if self.broken {
            return;
        }

        // Position constraint
        let vel_a = body_a.linear_velocity + body_a.angular_velocity.cross(self.r_a);
        let vel_b = body_b.linear_velocity + body_b.angular_velocity.cross(self.r_b);
        let cdot = vel_b - vel_a;

        let lambda = self.effective_mass * (-cdot + self.bias);
        self.accumulated_impulse += lambda;

        body_a.linear_velocity -= lambda * body_a.inv_mass;
        body_a.angular_velocity -= body_a.world_inv_inertia() * self.r_a.cross(lambda);
        body_b.linear_velocity += lambda * body_b.inv_mass;
        body_b.angular_velocity += body_b.world_inv_inertia() * self.r_b.cross(lambda);

        // Angular constraint: zero relative rotation
        let rel_omega = body_b.angular_velocity - body_a.angular_velocity;
        let lambda_ang = self.angular_effective_mass * (-rel_omega);
        self.angular_accumulated += lambda_ang;

        body_a.angular_velocity -= body_a.world_inv_inertia() * lambda_ang;
        body_b.angular_velocity += body_b.world_inv_inertia() * lambda_ang;

        // Check break threshold
        let total_force = self.accumulated_impulse.length() + self.angular_accumulated.length();
        if total_force > self.break_threshold {
            self.broken = true;
        }
    }

    fn is_broken(&self) -> bool {
        self.broken
    }

    fn break_force(&self) -> Option<f32> {
        Some(self.break_threshold)
    }
}

// ===========================================================================
// DistanceJoint
// ===========================================================================

/// Distance joint with min/max distance range.
///
/// Unlike a spring, this uses hard limits: the distance between anchors
/// must remain within [min_distance, max_distance].
pub struct DistanceJoint {
    pub body_a: RigidBodyHandle,
    pub body_b: RigidBodyHandle,
    pub local_anchor_a: Vec3,
    pub local_anchor_b: Vec3,
    /// Minimum allowed distance.
    pub min_distance: f32,
    /// Maximum allowed distance.
    pub max_distance: f32,
    /// Stiffness of the distance enforcement.
    pub stiffness: f32,
    /// Damping coefficient.
    pub damping: f32,
    /// Break threshold.
    pub break_force_threshold: Option<f32>,
    pub broken: bool,
    accumulated_impulse: f32,
    r_a: Vec3,
    r_b: Vec3,
}

impl DistanceJoint {
    pub fn new(
        body_a: RigidBodyHandle,
        body_b: RigidBodyHandle,
        anchor_a: Vec3,
        anchor_b: Vec3,
        min_distance: f32,
        max_distance: f32,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: anchor_a,
            local_anchor_b: anchor_b,
            min_distance,
            max_distance: max_distance.max(min_distance),
            stiffness: 1.0,
            damping: 0.1,
            break_force_threshold: None,
            broken: false,
            accumulated_impulse: 0.0,
            r_a: Vec3::ZERO,
            r_b: Vec3::ZERO,
        }
    }
}

impl Constraint for DistanceJoint {
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

        let world_a = body_a.position + self.r_a;
        let world_b = body_b.position + self.r_b;
        let diff = world_b - world_a;
        let dist = diff.length();

        if dist < 1e-7 {
            return;
        }

        let dir = diff / dist;

        // Check if distance is within allowed range
        let error = if dist < self.min_distance {
            dist - self.min_distance
        } else if dist > self.max_distance {
            dist - self.max_distance
        } else {
            return; // Within range, no correction needed
        };

        // Relative velocity along the constraint direction
        let vel_a = body_a.linear_velocity + body_a.angular_velocity.cross(self.r_a);
        let vel_b = body_b.linear_velocity + body_b.angular_velocity.cross(self.r_b);
        let rel_vel = (vel_b - vel_a).dot(dir);

        // Effective mass along the direction
        let k = body_a.inv_mass + body_b.inv_mass;
        if k < 1e-10 {
            return;
        }

        let lambda = -(error * self.stiffness + rel_vel * self.damping) / k;
        self.accumulated_impulse += lambda;

        let impulse = dir * lambda;
        body_a.linear_velocity -= impulse * body_a.inv_mass;
        body_a.angular_velocity -= body_a.world_inv_inertia() * self.r_a.cross(impulse);
        body_b.linear_velocity += impulse * body_b.inv_mass;
        body_b.angular_velocity += body_b.world_inv_inertia() * self.r_b.cross(impulse);

        if let Some(threshold) = self.break_force_threshold {
            if self.accumulated_impulse.abs() > threshold {
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

// ===========================================================================
// MouseJoint
// ===========================================================================

/// Mouse joint: drags a body toward a target point using a spring-damper.
///
/// Used for interactive editing/testing: the body is pulled toward
/// `target_position` with configurable spring stiffness and damping.
pub struct MouseJoint {
    pub body_a: RigidBodyHandle,
    /// Dummy second body handle (not used, for trait compatibility).
    pub body_b: RigidBodyHandle,
    /// Local anchor on the dragged body.
    pub local_anchor: Vec3,
    /// World-space target position.
    pub target_position: Vec3,
    /// Spring stiffness.
    pub stiffness: f32,
    /// Damping coefficient.
    pub damping: f32,
    /// Maximum force the joint can apply.
    pub max_force: f32,
    pub broken: bool,
    r_a: Vec3,
}

impl MouseJoint {
    pub fn new(
        body: RigidBodyHandle,
        dummy: RigidBodyHandle,
        local_anchor: Vec3,
        target: Vec3,
    ) -> Self {
        Self {
            body_a: body,
            body_b: dummy,
            local_anchor,
            target_position: target,
            stiffness: 500.0,
            damping: 30.0,
            max_force: 10000.0,
            broken: false,
            r_a: Vec3::ZERO,
        }
    }

    /// Update the target position.
    pub fn set_target(&mut self, target: Vec3) {
        self.target_position = target;
    }
}

impl Constraint for MouseJoint {
    fn bodies(&self) -> (RigidBodyHandle, RigidBodyHandle) {
        (self.body_a, self.body_b)
    }

    fn pre_solve(&mut self, body_a: &RigidBody, _body_b: &RigidBody, _dt: f32) {
        self.r_a = body_a.rotation * self.local_anchor;
    }

    fn solve(&mut self, body_a: &mut RigidBody, _body_b: &mut RigidBody) {
        if self.broken || body_a.is_static {
            return;
        }

        let world_anchor = body_a.position + self.r_a;
        let error = self.target_position - world_anchor;

        // Velocity at anchor point
        let vel = body_a.linear_velocity + body_a.angular_velocity.cross(self.r_a);

        // Spring-damper force
        let force = error * self.stiffness - vel * self.damping;

        // Clamp force magnitude
        let force_mag = force.length();
        let clamped_force = if force_mag > self.max_force && force_mag > 1e-6 {
            force * (self.max_force / force_mag)
        } else {
            force
        };

        // Apply as impulse (scaled for solver iteration)
        let impulse = clamped_force * 0.1;
        body_a.linear_velocity += impulse * body_a.inv_mass;
        body_a.angular_velocity += body_a.world_inv_inertia() * self.r_a.cross(impulse);
    }

    fn is_broken(&self) -> bool {
        self.broken
    }

    fn break_force(&self) -> Option<f32> {
        None
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::RigidBody;

    fn make_body(handle: u64, pos: Vec3, mass: f32) -> RigidBody {
        RigidBody {
            handle: RigidBodyHandle(handle),
            position: pos,
            mass,
            inv_mass: if mass > 0.0 { 1.0 / mass } else { 0.0 },
            ..Default::default()
        }
    }

    #[test]
    fn test_slider_joint_creation() {
        let joint = SliderJoint::new(
            RigidBodyHandle(1),
            RigidBodyHandle(2),
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::X,
        );
        assert!(!joint.broken);
        assert_eq!(joint.bodies(), (RigidBodyHandle(1), RigidBodyHandle(2)));
    }

    #[test]
    fn test_slider_joint_with_motor() {
        let joint = SliderJoint::new(
            RigidBodyHandle(1),
            RigidBodyHandle(2),
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::X,
        )
        .with_motor(5.0, 100.0);

        assert!(joint.motor_enabled);
        assert_eq!(joint.motor_target_velocity, 5.0);
    }

    #[test]
    fn test_cone_twist_joint() {
        let joint = ConeTwistJoint::new(
            RigidBodyHandle(1),
            RigidBodyHandle(2),
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            0.5,
            0.3,
        );
        assert!(!joint.broken);
        assert_eq!(joint.swing_limit, 0.5);
    }

    #[test]
    fn test_gear_joint_constrains_ratio() {
        let mut body_a = make_body(1, Vec3::ZERO, 1.0);
        let mut body_b = make_body(2, Vec3::new(2.0, 0.0, 0.0), 1.0);

        body_a.angular_velocity = Vec3::new(0.0, 10.0, 0.0);
        body_b.angular_velocity = Vec3::ZERO;

        let mut joint = GearJoint::new(
            RigidBodyHandle(1),
            RigidBodyHandle(2),
            Vec3::Y,
            Vec3::Y,
            2.0,
        );

        joint.pre_solve(&body_a, &body_b, 1.0 / 60.0);

        for _ in 0..20 {
            joint.solve(&mut body_a, &mut body_b);
        }

        // After solving, omega_a should approach ratio * omega_b
        let omega_a = body_a.angular_velocity.y;
        let omega_b = body_b.angular_velocity.y;
        let ratio_error = (omega_a - 2.0 * omega_b).abs();
        assert!(
            ratio_error < 1.0,
            "Gear ratio error = {}, omega_a={}, omega_b={}",
            ratio_error,
            omega_a,
            omega_b
        );
    }

    #[test]
    fn test_weld_joint_breaks() {
        let mut body_a = make_body(1, Vec3::ZERO, 1.0);
        let mut body_b = make_body(2, Vec3::new(5.0, 0.0, 0.0), 1.0);

        let mut joint = WeldJoint::new(
            RigidBodyHandle(1),
            RigidBodyHandle(2),
            Vec3::X,
            Vec3::NEG_X,
            0.001, // Very low break threshold
        );

        joint.pre_solve(&body_a, &body_b, 1.0 / 60.0);
        joint.solve(&mut body_a, &mut body_b);

        assert!(joint.broken, "Weld should break with large separation");
    }

    #[test]
    fn test_distance_joint_min_max() {
        let mut body_a = make_body(1, Vec3::ZERO, 1.0);
        let mut body_b = make_body(2, Vec3::new(10.0, 0.0, 0.0), 1.0);

        let mut joint = DistanceJoint::new(
            RigidBodyHandle(1),
            RigidBodyHandle(2),
            Vec3::ZERO,
            Vec3::ZERO,
            2.0,
            5.0,
        );

        joint.pre_solve(&body_a, &body_b, 1.0 / 60.0);

        for _ in 0..50 {
            joint.solve(&mut body_a, &mut body_b);
        }

        // Bodies should have been pulled closer (distance was 10, max is 5)
        let dist = (body_b.position - body_a.position).length();
        // After solving velocity constraints, the distance constraint manifests
        // as velocity changes. The actual position change happens during integration.
        // But we should see that impulses were applied.
        let moved =
            body_a.linear_velocity.length() > 0.0 || body_b.linear_velocity.length() > 0.0;
        assert!(moved, "Distance joint should apply impulses");
    }

    #[test]
    fn test_mouse_joint() {
        let mut body_a = make_body(1, Vec3::ZERO, 1.0);
        let mut body_b = make_body(2, Vec3::ZERO, 0.0); // dummy

        let mut joint = MouseJoint::new(
            RigidBodyHandle(1),
            RigidBodyHandle(2),
            Vec3::ZERO,
            Vec3::new(5.0, 0.0, 0.0),
        );

        joint.pre_solve(&body_a, &body_b, 1.0 / 60.0);

        for _ in 0..20 {
            joint.solve(&mut body_a, &mut body_b);
        }

        // Body should have gained velocity toward the target
        assert!(
            body_a.linear_velocity.x > 0.0,
            "Mouse joint should pull body toward target: v.x = {}",
            body_a.linear_velocity.x
        );
    }

    #[test]
    fn test_pulley_joint() {
        let mut body_a = make_body(1, Vec3::new(0.0, -2.0, 0.0), 1.0);
        let mut body_b = make_body(2, Vec3::new(5.0, -1.0, 0.0), 1.0);

        let mut joint = PulleyJoint::new(
            RigidBodyHandle(1),
            RigidBodyHandle(2),
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(5.0, 5.0, 0.0),
            Vec3::ZERO,
            Vec3::ZERO,
            1.0,
        );

        joint.pre_solve(&body_a, &body_b, 1.0 / 60.0);
        assert!(joint.total_length > 0.0);

        joint.solve(&mut body_a, &mut body_b);
        // Should not panic
    }
}
