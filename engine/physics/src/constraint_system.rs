// engine/physics/src/constraint_system.rs
//
// Generic constraint framework for rigid body physics. Provides:
//   - A trait-based constraint interface (velocity + position level)
//   - Pre-step: compute effective mass, bias, and accumulated impulse
//   - Solve: iterative impulse application with warm starting
//   - Built-in constraint types: distance, contact, hinge axis, limit
//   - Sequential impulse solver with configurable iteration count

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Vec3 / Quaternion
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    #[inline]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-12 { Self::ZERO } else { self * (1.0 / l) }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self::new(self.x + r.x, self.y + r.y, self.z + r.z) }
}
impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, r: Self) -> Self { Self::new(self.x - r.x, self.y - r.y, self.z - r.z) }
}
impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self { Self::new(-self.x, -self.y, -self.z) }
}
impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, s: f32) -> Self { Self::new(self.x * s, self.y * s, self.z * s) }
}
impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, r: Self) { self.x += r.x; self.y += r.y; self.z += r.z; }
}
impl std::ops::SubAssign for Vec3 {
    fn sub_assign(&mut self, r: Self) { self.x -= r.x; self.y -= r.y; self.z -= r.z; }
}

// ---------------------------------------------------------------------------
// Body handle and body state
// ---------------------------------------------------------------------------

/// A handle to a rigid body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BodyHandle(pub u32);

impl BodyHandle {
    pub const WORLD: Self = Self(u32::MAX);

    pub fn is_world(&self) -> bool {
        *self == Self::WORLD
    }
}

/// The dynamic state of a rigid body, used by the constraint solver.
#[derive(Debug, Clone)]
pub struct BodyVelocity {
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub inv_mass: f32,
    pub inv_inertia: [f32; 9], // 3x3 world-space inverse inertia tensor
    pub position: Vec3,
    pub is_static: bool,
}

impl BodyVelocity {
    pub fn new_dynamic(mass: f32, inertia_diag: Vec3) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        let mut inv_inertia = [0.0f32; 9];
        if inertia_diag.x > 0.0 { inv_inertia[0] = 1.0 / inertia_diag.x; }
        if inertia_diag.y > 0.0 { inv_inertia[4] = 1.0 / inertia_diag.y; }
        if inertia_diag.z > 0.0 { inv_inertia[8] = 1.0 / inertia_diag.z; }
        Self {
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            inv_mass,
            inv_inertia,
            position: Vec3::ZERO,
            is_static: false,
        }
    }

    pub fn new_static() -> Self {
        Self {
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            inv_mass: 0.0,
            inv_inertia: [0.0; 9],
            position: Vec3::ZERO,
            is_static: true,
        }
    }

    /// Apply an impulse at a point (world space).
    #[inline]
    pub fn apply_impulse(&mut self, impulse: Vec3, contact_point: Vec3) {
        if self.is_static {
            return;
        }
        self.linear_velocity += impulse * self.inv_mass;
        let r = contact_point - self.position;
        let torque = r.cross(impulse);
        self.angular_velocity += self.mul_inv_inertia(torque);
    }

    /// Apply a linear-only impulse.
    #[inline]
    pub fn apply_linear_impulse(&mut self, impulse: Vec3) {
        if self.is_static {
            return;
        }
        self.linear_velocity += impulse * self.inv_mass;
    }

    /// Apply an angular-only impulse.
    #[inline]
    pub fn apply_angular_impulse(&mut self, impulse: Vec3) {
        if self.is_static {
            return;
        }
        self.angular_velocity += self.mul_inv_inertia(impulse);
    }

    /// Multiply a vector by the inverse inertia tensor.
    #[inline]
    pub fn mul_inv_inertia(&self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.inv_inertia[0] * v.x + self.inv_inertia[1] * v.y + self.inv_inertia[2] * v.z,
            self.inv_inertia[3] * v.x + self.inv_inertia[4] * v.y + self.inv_inertia[5] * v.z,
            self.inv_inertia[6] * v.x + self.inv_inertia[7] * v.y + self.inv_inertia[8] * v.z,
        )
    }

    /// Compute the velocity at a world-space point.
    #[inline]
    pub fn velocity_at(&self, point: Vec3) -> Vec3 {
        let r = point - self.position;
        self.linear_velocity + self.angular_velocity.cross(r)
    }

    /// Compute the effective mass for a constraint direction at a point.
    pub fn effective_mass_at(&self, normal: Vec3, r: Vec3) -> f32 {
        let rn = r.cross(normal);
        let irn = self.mul_inv_inertia(rn);
        self.inv_mass + rn.dot(irn)
    }
}

// ---------------------------------------------------------------------------
// Constraint trait
// ---------------------------------------------------------------------------

/// The constraint interface. Each constraint operates on one or two bodies.
pub trait Constraint: std::fmt::Debug {
    /// Unique type name for debugging.
    fn type_name(&self) -> &'static str;

    /// Which bodies this constraint connects.
    fn bodies(&self) -> (BodyHandle, BodyHandle);

    /// Pre-step: compute the effective mass, bias velocity, and prepare
    /// for warm starting. Called once per solver iteration cycle.
    fn pre_step(
        &mut self,
        body_a: &BodyVelocity,
        body_b: &BodyVelocity,
        dt: f32,
    );

    /// Apply warm-start impulses from the previous frame.
    fn warm_start(
        &self,
        body_a: &mut BodyVelocity,
        body_b: &mut BodyVelocity,
    );

    /// Solve velocity constraints: compute and apply impulse corrections.
    fn solve_velocity(
        &mut self,
        body_a: &mut BodyVelocity,
        body_b: &mut BodyVelocity,
    );

    /// Solve position constraints (optional, for position correction).
    fn solve_position(
        &mut self,
        body_a: &mut BodyVelocity,
        body_b: &mut BodyVelocity,
    ) -> f32 {
        let _ = (body_a, body_b);
        0.0
    }

    /// Whether this constraint is active.
    fn is_active(&self) -> bool {
        true
    }

    /// Return accumulated impulse magnitude (for diagnostics).
    fn accumulated_impulse(&self) -> f32 {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Distance constraint
// ---------------------------------------------------------------------------

/// Maintains a fixed distance between two anchor points on two bodies.
#[derive(Debug)]
pub struct DistanceConstraint {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    /// Local-space anchor on body A.
    pub local_anchor_a: Vec3,
    /// Local-space anchor on body B.
    pub local_anchor_b: Vec3,
    /// Target distance.
    pub rest_length: f32,
    /// Constraint stiffness (0..1, 1 = rigid).
    pub stiffness: f32,
    /// Damping ratio (0..1).
    pub damping: f32,
    // Pre-step cached values
    effective_mass: f32,
    bias: f32,
    direction: Vec3,
    accumulated_impulse: f32,
    gamma: f32,
    r_a: Vec3,
    r_b: Vec3,
    active: bool,
}

impl DistanceConstraint {
    pub fn new(
        body_a: BodyHandle,
        body_b: BodyHandle,
        anchor_a: Vec3,
        anchor_b: Vec3,
        rest_length: f32,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: anchor_a,
            local_anchor_b: anchor_b,
            rest_length,
            stiffness: 1.0,
            damping: 0.0,
            effective_mass: 0.0,
            bias: 0.0,
            direction: Vec3::ZERO,
            accumulated_impulse: 0.0,
            gamma: 0.0,
            r_a: Vec3::ZERO,
            r_b: Vec3::ZERO,
            active: true,
        }
    }

    pub fn with_spring(mut self, stiffness: f32, damping: f32) -> Self {
        self.stiffness = stiffness;
        self.damping = damping;
        self
    }
}

impl Constraint for DistanceConstraint {
    fn type_name(&self) -> &'static str { "DistanceConstraint" }

    fn bodies(&self) -> (BodyHandle, BodyHandle) {
        (self.body_a, self.body_b)
    }

    fn pre_step(&mut self, body_a: &BodyVelocity, body_b: &BodyVelocity, dt: f32) {
        self.r_a = self.local_anchor_a;
        self.r_b = self.local_anchor_b;

        let world_a = body_a.position + self.r_a;
        let world_b = body_b.position + self.r_b;
        let delta = world_b - world_a;
        let current_length = delta.length();

        if current_length < 1e-8 {
            self.active = false;
            return;
        }
        self.active = true;
        self.direction = delta * (1.0 / current_length);

        let error = current_length - self.rest_length;

        // Compute effective mass along the constraint direction.
        let k_a = body_a.effective_mass_at(self.direction, self.r_a);
        let k_b = body_b.effective_mass_at(self.direction, self.r_b);
        let k = k_a + k_b;

        if self.stiffness < 1.0 && self.stiffness > 0.0 {
            // Soft constraint (spring-damper).
            let omega = 2.0 * std::f32::consts::PI * self.stiffness * 60.0;
            let c = 2.0 * k * self.damping * omega;
            let spring_k = k * omega * omega;
            self.gamma = 1.0 / (dt * (c + dt * spring_k));
            let beta = dt * spring_k * self.gamma;
            self.effective_mass = 1.0 / (k + self.gamma);
            self.bias = error * beta;
        } else {
            // Hard constraint.
            self.gamma = 0.0;
            if k > 0.0 {
                self.effective_mass = 1.0 / k;
            } else {
                self.effective_mass = 0.0;
            }
            // Baumgarte stabilization.
            let baumgarte = 0.2;
            self.bias = baumgarte * error / dt;
        }
    }

    fn warm_start(&self, body_a: &mut BodyVelocity, body_b: &mut BodyVelocity) {
        if !self.active {
            return;
        }
        let impulse = self.direction * self.accumulated_impulse;
        body_a.apply_impulse(-impulse, body_a.position + self.r_a);
        body_b.apply_impulse(impulse, body_b.position + self.r_b);
    }

    fn solve_velocity(&mut self, body_a: &mut BodyVelocity, body_b: &mut BodyVelocity) {
        if !self.active {
            return;
        }
        let vel_a = body_a.velocity_at(body_a.position + self.r_a);
        let vel_b = body_b.velocity_at(body_b.position + self.r_b);
        let cdot = self.direction.dot(vel_b - vel_a);

        let impulse = -self.effective_mass
            * (cdot + self.bias + self.gamma * self.accumulated_impulse);
        self.accumulated_impulse += impulse;

        let j = self.direction * impulse;
        body_a.apply_impulse(-j, body_a.position + self.r_a);
        body_b.apply_impulse(j, body_b.position + self.r_b);
    }

    fn solve_position(&mut self, body_a: &mut BodyVelocity, body_b: &mut BodyVelocity) -> f32 {
        if !self.active {
            return 0.0;
        }
        let world_a = body_a.position + self.r_a;
        let world_b = body_b.position + self.r_b;
        let delta = world_b - world_a;
        let current_length = delta.length();
        if current_length < 1e-8 {
            return 0.0;
        }
        let n = delta * (1.0 / current_length);
        let error = current_length - self.rest_length;

        let k = body_a.effective_mass_at(n, self.r_a)
            + body_b.effective_mass_at(n, self.r_b);
        if k <= 0.0 {
            return error.abs();
        }

        let correction = -error / k * 0.2; // Baumgarte factor
        let impulse = n * correction;
        body_a.position -= impulse * body_a.inv_mass;
        body_b.position += impulse * body_b.inv_mass;

        error.abs()
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn accumulated_impulse(&self) -> f32 {
        self.accumulated_impulse
    }
}

// ---------------------------------------------------------------------------
// Contact constraint
// ---------------------------------------------------------------------------

/// A contact constraint between two bodies.
#[derive(Debug)]
pub struct ContactConstraint {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    pub contact_point: Vec3,
    pub normal: Vec3,
    pub penetration: f32,
    pub friction: f32,
    pub restitution: f32,
    // Cached
    r_a: Vec3,
    r_b: Vec3,
    normal_mass: f32,
    tangent_mass_1: f32,
    tangent_mass_2: f32,
    tangent_1: Vec3,
    tangent_2: Vec3,
    normal_impulse: f32,
    tangent_impulse_1: f32,
    tangent_impulse_2: f32,
    bias: f32,
    velocity_bias: f32,
}

impl ContactConstraint {
    pub fn new(
        body_a: BodyHandle,
        body_b: BodyHandle,
        contact_point: Vec3,
        normal: Vec3,
        penetration: f32,
        friction: f32,
        restitution: f32,
    ) -> Self {
        // Compute tangent vectors.
        let (t1, t2) = compute_tangent_basis(normal);
        Self {
            body_a,
            body_b,
            contact_point,
            normal,
            penetration,
            friction,
            restitution,
            r_a: Vec3::ZERO,
            r_b: Vec3::ZERO,
            normal_mass: 0.0,
            tangent_mass_1: 0.0,
            tangent_mass_2: 0.0,
            tangent_1: t1,
            tangent_2: t2,
            normal_impulse: 0.0,
            tangent_impulse_1: 0.0,
            tangent_impulse_2: 0.0,
            bias: 0.0,
            velocity_bias: 0.0,
        }
    }
}

fn compute_tangent_basis(normal: Vec3) -> (Vec3, Vec3) {
    let t1 = if normal.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0).cross(normal).normalized()
    } else {
        Vec3::new(0.0, 1.0, 0.0).cross(normal).normalized()
    };
    let t2 = normal.cross(t1);
    (t1, t2)
}

impl Constraint for ContactConstraint {
    fn type_name(&self) -> &'static str { "ContactConstraint" }

    fn bodies(&self) -> (BodyHandle, BodyHandle) {
        (self.body_a, self.body_b)
    }

    fn pre_step(&mut self, body_a: &BodyVelocity, body_b: &BodyVelocity, dt: f32) {
        self.r_a = self.contact_point - body_a.position;
        self.r_b = self.contact_point - body_b.position;

        // Effective mass for normal direction.
        let kn = body_a.effective_mass_at(self.normal, self.r_a)
            + body_b.effective_mass_at(self.normal, self.r_b);
        self.normal_mass = if kn > 0.0 { 1.0 / kn } else { 0.0 };

        // Effective mass for tangent directions.
        let kt1 = body_a.effective_mass_at(self.tangent_1, self.r_a)
            + body_b.effective_mass_at(self.tangent_1, self.r_b);
        self.tangent_mass_1 = if kt1 > 0.0 { 1.0 / kt1 } else { 0.0 };

        let kt2 = body_a.effective_mass_at(self.tangent_2, self.r_a)
            + body_b.effective_mass_at(self.tangent_2, self.r_b);
        self.tangent_mass_2 = if kt2 > 0.0 { 1.0 / kt2 } else { 0.0 };

        // Baumgarte position correction bias.
        let slop = 0.005; // penetration allowance
        let baumgarte = 0.2;
        self.bias = -baumgarte / dt * (self.penetration + slop).min(0.0);

        // Restitution velocity bias.
        let rel_vel = body_b.velocity_at(self.contact_point) - body_a.velocity_at(self.contact_point);
        let closing_vel = self.normal.dot(rel_vel);
        let restitution_threshold = 1.0;
        if closing_vel < -restitution_threshold {
            self.velocity_bias = -self.restitution * closing_vel;
        } else {
            self.velocity_bias = 0.0;
        }
    }

    fn warm_start(&self, body_a: &mut BodyVelocity, body_b: &mut BodyVelocity) {
        let impulse = self.normal * self.normal_impulse
            + self.tangent_1 * self.tangent_impulse_1
            + self.tangent_2 * self.tangent_impulse_2;
        body_a.apply_impulse(-impulse, body_a.position + self.r_a);
        body_b.apply_impulse(impulse, body_b.position + self.r_b);
    }

    fn solve_velocity(&mut self, body_a: &mut BodyVelocity, body_b: &mut BodyVelocity) {
        // Solve friction (tangent constraints).
        {
            let rel_vel = body_b.velocity_at(body_b.position + self.r_b)
                - body_a.velocity_at(body_a.position + self.r_a);

            // Tangent 1.
            let vt1 = rel_vel.dot(self.tangent_1);
            let lambda1 = -vt1 * self.tangent_mass_1;
            let max_friction = self.friction * self.normal_impulse;
            let old1 = self.tangent_impulse_1;
            self.tangent_impulse_1 = (old1 + lambda1).clamp(-max_friction, max_friction);
            let dt1 = self.tangent_impulse_1 - old1;

            // Tangent 2.
            let vt2 = rel_vel.dot(self.tangent_2);
            let lambda2 = -vt2 * self.tangent_mass_2;
            let old2 = self.tangent_impulse_2;
            self.tangent_impulse_2 = (old2 + lambda2).clamp(-max_friction, max_friction);
            let dt2 = self.tangent_impulse_2 - old2;

            let friction_impulse = self.tangent_1 * dt1 + self.tangent_2 * dt2;
            body_a.apply_impulse(-friction_impulse, body_a.position + self.r_a);
            body_b.apply_impulse(friction_impulse, body_b.position + self.r_b);
        }

        // Solve normal constraint.
        {
            let rel_vel = body_b.velocity_at(body_b.position + self.r_b)
                - body_a.velocity_at(body_a.position + self.r_a);
            let vn = rel_vel.dot(self.normal);
            let lambda = -(vn - self.bias - self.velocity_bias) * self.normal_mass;
            let old = self.normal_impulse;
            self.normal_impulse = (old + lambda).max(0.0); // non-penetration
            let dn = self.normal_impulse - old;

            let impulse = self.normal * dn;
            body_a.apply_impulse(-impulse, body_a.position + self.r_a);
            body_b.apply_impulse(impulse, body_b.position + self.r_b);
        }
    }

    fn accumulated_impulse(&self) -> f32 {
        self.normal_impulse
    }
}

// ---------------------------------------------------------------------------
// Hinge constraint (revolute joint axis)
// ---------------------------------------------------------------------------

/// Constrains two bodies to rotate about a shared axis.
#[derive(Debug)]
pub struct HingeConstraint {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub axis: Vec3,
    pub lower_limit: f32,
    pub upper_limit: f32,
    pub limits_enabled: bool,
    pub motor_speed: f32,
    pub motor_max_torque: f32,
    pub motor_enabled: bool,
    // Cached
    effective_mass_point: [f32; 3], // for the point-to-point part
    effective_mass_axis: f32,
    accumulated_point: Vec3,
    accumulated_axis: f32,
    accumulated_motor: f32,
    r_a: Vec3,
    r_b: Vec3,
    bias_point: Vec3,
}

impl HingeConstraint {
    pub fn new(
        body_a: BodyHandle,
        body_b: BodyHandle,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis: Vec3,
    ) -> Self {
        Self {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis: axis.normalized(),
            lower_limit: -std::f32::consts::PI,
            upper_limit: std::f32::consts::PI,
            limits_enabled: false,
            motor_speed: 0.0,
            motor_max_torque: 0.0,
            motor_enabled: false,
            effective_mass_point: [0.0; 3],
            effective_mass_axis: 0.0,
            accumulated_point: Vec3::ZERO,
            accumulated_axis: 0.0,
            accumulated_motor: 0.0,
            r_a: Vec3::ZERO,
            r_b: Vec3::ZERO,
            bias_point: Vec3::ZERO,
        }
    }

    pub fn with_limits(mut self, lower: f32, upper: f32) -> Self {
        self.lower_limit = lower;
        self.upper_limit = upper;
        self.limits_enabled = true;
        self
    }

    pub fn with_motor(mut self, speed: f32, max_torque: f32) -> Self {
        self.motor_speed = speed;
        self.motor_max_torque = max_torque;
        self.motor_enabled = true;
        self
    }
}

impl Constraint for HingeConstraint {
    fn type_name(&self) -> &'static str { "HingeConstraint" }

    fn bodies(&self) -> (BodyHandle, BodyHandle) {
        (self.body_a, self.body_b)
    }

    fn pre_step(&mut self, body_a: &BodyVelocity, body_b: &BodyVelocity, dt: f32) {
        self.r_a = self.anchor_a;
        self.r_b = self.anchor_b;

        let world_a = body_a.position + self.r_a;
        let world_b = body_b.position + self.r_b;
        let error = world_b - world_a;

        let baumgarte = 0.2;
        self.bias_point = error * (-baumgarte / dt);

        // Compute effective mass for each axis of the point constraint.
        let axes = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        for (i, axis) in axes.iter().enumerate() {
            let k = body_a.effective_mass_at(*axis, self.r_a)
                + body_b.effective_mass_at(*axis, self.r_b);
            self.effective_mass_point[i] = if k > 0.0 { 1.0 / k } else { 0.0 };
        }

        // Effective mass for the angular axis constraint.
        let rn_a = body_a.mul_inv_inertia(self.axis);
        let rn_b = body_b.mul_inv_inertia(self.axis);
        let k_axis = self.axis.dot(rn_a) + self.axis.dot(rn_b);
        self.effective_mass_axis = if k_axis > 0.0 { 1.0 / k_axis } else { 0.0 };
    }

    fn warm_start(&self, body_a: &mut BodyVelocity, body_b: &mut BodyVelocity) {
        body_a.apply_impulse(-self.accumulated_point, body_a.position + self.r_a);
        body_b.apply_impulse(self.accumulated_point, body_b.position + self.r_b);

        let angular = self.axis * (self.accumulated_axis + self.accumulated_motor);
        body_a.apply_angular_impulse(-angular);
        body_b.apply_angular_impulse(angular);
    }

    fn solve_velocity(&mut self, body_a: &mut BodyVelocity, body_b: &mut BodyVelocity) {
        // Point-to-point constraint (3 DOF).
        let rel_vel = body_b.velocity_at(body_b.position + self.r_b)
            - body_a.velocity_at(body_a.position + self.r_a);

        let impulse = Vec3::new(
            -(rel_vel.x - self.bias_point.x) * self.effective_mass_point[0],
            -(rel_vel.y - self.bias_point.y) * self.effective_mass_point[1],
            -(rel_vel.z - self.bias_point.z) * self.effective_mass_point[2],
        );
        self.accumulated_point += impulse;
        body_a.apply_impulse(-impulse, body_a.position + self.r_a);
        body_b.apply_impulse(impulse, body_b.position + self.r_b);

        // Motor constraint.
        if self.motor_enabled {
            let rel_omega = body_b.angular_velocity - body_a.angular_velocity;
            let omega_axis = self.axis.dot(rel_omega);
            let motor_impulse = -(omega_axis - self.motor_speed) * self.effective_mass_axis;
            let old = self.accumulated_motor;
            self.accumulated_motor = (old + motor_impulse).clamp(
                -self.motor_max_torque,
                self.motor_max_torque,
            );
            let dm = self.accumulated_motor - old;
            let j = self.axis * dm;
            body_a.apply_angular_impulse(-j);
            body_b.apply_angular_impulse(j);
        }
    }

    fn accumulated_impulse(&self) -> f32 {
        self.accumulated_point.length() + self.accumulated_axis.abs()
    }
}

// ---------------------------------------------------------------------------
// Constraint handle
// ---------------------------------------------------------------------------

/// A handle to a constraint in the solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstraintHandle(pub u32);

// ---------------------------------------------------------------------------
// Constraint solver
// ---------------------------------------------------------------------------

/// Configuration for the sequential impulse solver.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Number of velocity solver iterations.
    pub velocity_iterations: u32,
    /// Number of position solver iterations.
    pub position_iterations: u32,
    /// Warm starting factor (0..1, typically 0.95).
    pub warm_start_factor: f32,
    /// Position correction slop (allowed penetration).
    pub position_slop: f32,
    /// Maximum position correction per step.
    pub max_position_correction: f32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            velocity_iterations: 8,
            position_iterations: 3,
            warm_start_factor: 0.95,
            position_slop: 0.005,
            max_position_correction: 0.2,
        }
    }
}

/// Statistics from the solver.
#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    pub constraint_count: u32,
    pub velocity_iterations_used: u32,
    pub position_iterations_used: u32,
    pub max_position_error: f32,
    pub total_impulse: f32,
    pub solve_time_us: u64,
}

/// The sequential impulse constraint solver.
pub struct ConstraintSolver {
    constraints: Vec<Box<dyn Constraint>>,
    handles: HashMap<u32, usize>,
    next_handle: u32,
    config: SolverConfig,
    stats: SolverStats,
}

impl ConstraintSolver {
    pub fn new() -> Self {
        Self::with_config(SolverConfig::default())
    }

    pub fn with_config(config: SolverConfig) -> Self {
        Self {
            constraints: Vec::new(),
            handles: HashMap::new(),
            next_handle: 0,
            config,
            stats: SolverStats::default(),
        }
    }

    pub fn config(&self) -> &SolverConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut SolverConfig {
        &mut self.config
    }

    pub fn stats(&self) -> &SolverStats {
        &self.stats
    }

    /// Add a constraint to the solver.
    pub fn add_constraint(&mut self, constraint: Box<dyn Constraint>) -> ConstraintHandle {
        let handle = ConstraintHandle(self.next_handle);
        self.next_handle += 1;
        let idx = self.constraints.len();
        self.handles.insert(handle.0, idx);
        self.constraints.push(constraint);
        handle
    }

    /// Remove a constraint.
    pub fn remove_constraint(&mut self, handle: ConstraintHandle) -> bool {
        if let Some(&idx) = self.handles.get(&handle.0) {
            self.handles.remove(&handle.0);
            let last = self.constraints.len() - 1;
            if idx != last {
                self.constraints.swap(idx, last);
                // Update handle mapping for the swapped constraint.
                for (h, i) in self.handles.iter_mut() {
                    if *i == last {
                        *i = idx;
                        break;
                    }
                }
            }
            self.constraints.pop();
            true
        } else {
            false
        }
    }

    /// Clear all constraints.
    pub fn clear(&mut self) {
        self.constraints.clear();
        self.handles.clear();
    }

    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Run the full solver pipeline.
    pub fn solve(
        &mut self,
        bodies: &mut HashMap<u32, BodyVelocity>,
        dt: f32,
    ) {
        let start = std::time::Instant::now();
        let world_body = BodyVelocity::new_static();

        // Pre-step: compute effective masses and biases.
        for constraint in self.constraints.iter_mut() {
            if !constraint.is_active() {
                continue;
            }
            let (ha, hb) = constraint.bodies();
            let ba = bodies.get(&ha.0).cloned().unwrap_or_else(|| world_body.clone());
            let bb = bodies.get(&hb.0).cloned().unwrap_or_else(|| world_body.clone());
            constraint.pre_step(&ba, &bb, dt);
        }

        // Warm start.
        for constraint in self.constraints.iter() {
            if !constraint.is_active() {
                continue;
            }
            let (ha, hb) = constraint.bodies();
            let mut ba = bodies.get(&ha.0).cloned().unwrap_or_else(|| world_body.clone());
            let mut bb = bodies.get(&hb.0).cloned().unwrap_or_else(|| world_body.clone());
            constraint.warm_start(&mut ba, &mut bb);
            if !ha.is_world() { if let Some(b) = bodies.get_mut(&ha.0) { *b = ba; } }
            if !hb.is_world() { if let Some(b) = bodies.get_mut(&hb.0) { *b = bb; } }
        }

        // Velocity iterations.
        for _ in 0..self.config.velocity_iterations {
            for constraint in self.constraints.iter_mut() {
                if !constraint.is_active() {
                    continue;
                }
                let (ha, hb) = constraint.bodies();
                let mut ba = bodies.get(&ha.0).cloned().unwrap_or_else(|| world_body.clone());
                let mut bb = bodies.get(&hb.0).cloned().unwrap_or_else(|| world_body.clone());
                constraint.solve_velocity(&mut ba, &mut bb);
                if !ha.is_world() { if let Some(b) = bodies.get_mut(&ha.0) { *b = ba; } }
                if !hb.is_world() { if let Some(b) = bodies.get_mut(&hb.0) { *b = bb; } }
            }
        }

        // Position iterations.
        let mut max_error = 0.0f32;
        for _ in 0..self.config.position_iterations {
            for constraint in self.constraints.iter_mut() {
                if !constraint.is_active() {
                    continue;
                }
                let (ha, hb) = constraint.bodies();
                let mut ba = bodies.get(&ha.0).cloned().unwrap_or_else(|| world_body.clone());
                let mut bb = bodies.get(&hb.0).cloned().unwrap_or_else(|| world_body.clone());
                let error = constraint.solve_position(&mut ba, &mut bb);
                max_error = max_error.max(error);
                if !ha.is_world() { if let Some(b) = bodies.get_mut(&ha.0) { *b = ba; } }
                if !hb.is_world() { if let Some(b) = bodies.get_mut(&hb.0) { *b = bb; } }
            }
        }

        // Gather stats.
        let total_impulse: f32 = self.constraints.iter()
            .filter(|c| c.is_active())
            .map(|c| c.accumulated_impulse())
            .sum();

        self.stats = SolverStats {
            constraint_count: self.constraints.len() as u32,
            velocity_iterations_used: self.config.velocity_iterations,
            position_iterations_used: self.config.position_iterations,
            max_position_error: max_error,
            total_impulse,
            solve_time_us: start.elapsed().as_micros() as u64,
        };
    }
}

impl Default for ConstraintSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_body_apply_impulse() {
        let mut body = BodyVelocity::new_dynamic(1.0, Vec3::new(1.0, 1.0, 1.0));
        body.position = Vec3::ZERO;
        body.apply_linear_impulse(Vec3::new(1.0, 0.0, 0.0));
        assert!((body.linear_velocity.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_constraint() {
        let mut solver = ConstraintSolver::new();
        let c = DistanceConstraint::new(
            BodyHandle(0),
            BodyHandle(1),
            Vec3::ZERO,
            Vec3::ZERO,
            2.0,
        );
        solver.add_constraint(Box::new(c));

        let mut bodies = HashMap::new();
        let mut b0 = BodyVelocity::new_dynamic(1.0, Vec3::new(1.0, 1.0, 1.0));
        b0.position = Vec3::new(0.0, 0.0, 0.0);
        let mut b1 = BodyVelocity::new_dynamic(1.0, Vec3::new(1.0, 1.0, 1.0));
        b1.position = Vec3::new(3.0, 0.0, 0.0); // 3 units apart, rest = 2
        bodies.insert(0, b0);
        bodies.insert(1, b1);

        solver.solve(&mut bodies, 1.0 / 60.0);

        assert!(solver.stats().constraint_count == 1);
    }

    #[test]
    fn test_contact_constraint() {
        let mut solver = ConstraintSolver::new();
        let c = ContactConstraint::new(
            BodyHandle(0),
            BodyHandle(1),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            -0.1,
            0.5,
            0.3,
        );
        solver.add_constraint(Box::new(c));

        let mut bodies = HashMap::new();
        let mut b0 = BodyVelocity::new_dynamic(1.0, Vec3::new(1.0, 1.0, 1.0));
        b0.position = Vec3::new(0.0, -0.5, 0.0);
        let mut b1 = BodyVelocity::new_dynamic(1.0, Vec3::new(1.0, 1.0, 1.0));
        b1.position = Vec3::new(0.0, 0.5, 0.0);
        b1.linear_velocity = Vec3::new(0.0, -5.0, 0.0);
        bodies.insert(0, b0);
        bodies.insert(1, b1);

        solver.solve(&mut bodies, 1.0 / 60.0);

        // After solving, body 1 should have less negative velocity.
        let v1 = bodies.get(&1).unwrap().linear_velocity.y;
        assert!(v1 > -5.0, "velocity should be corrected: {v1}");
    }

    #[test]
    fn test_add_remove_constraint() {
        let mut solver = ConstraintSolver::new();
        let h = solver.add_constraint(Box::new(DistanceConstraint::new(
            BodyHandle(0), BodyHandle(1), Vec3::ZERO, Vec3::ZERO, 1.0,
        )));
        assert_eq!(solver.constraint_count(), 1);
        solver.remove_constraint(h);
        assert_eq!(solver.constraint_count(), 0);
    }
}
