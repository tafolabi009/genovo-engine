//! Advanced motorized and configurable joint types for the Genovo physics engine.
//!
//! Provides:
//! - **`MotorJoint`** — PD-controlled motorized joint with position/velocity targeting
//! - **`RopeJoint`** — maximum-distance constraint (slack when closer)
//! - **`WheelJoint`** — vehicle wheel with suspension spring-damper and spin motor
//! - **`PrismaticJoint`** — slider along a fixed axis with limits and motor
//! - **`GenericJoint`** — fully configurable 6-DOF joint with per-axis modes
//!
//! All joints implement the engine's constraint solving interface and can be
//! integrated into the sequential-impulse solver.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Vec3 helper (standalone, no external dependency required)
// ---------------------------------------------------------------------------

/// Minimal 3D vector for joint computations.
///
/// When the full `glam` crate is available these can be replaced with
/// `glam::Vec3`; this standalone version keeps the module self-contained.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const X: Self = Self {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };
    pub const Y: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    pub const Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-10 {
            Self::ZERO
        } else {
            Self {
                x: self.x / len,
                y: self.y / len,
                z: self.z / len,
            }
        }
    }

    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        Self {
            x: self.x + (rhs.x - self.x) * t,
            y: self.y + (rhs.y - self.y) * t,
            z: self.z + (rhs.z - self.z) * t,
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

// ---------------------------------------------------------------------------
// Body handle and body data
// ---------------------------------------------------------------------------

/// Opaque handle identifying a rigid body in the physics world.
pub type BodyHandle = u32;

/// Minimal rigid body state needed by joint solvers.
///
/// In a full engine this would reference the dynamics system; here we
/// store enough data for the constraint math.
#[derive(Debug, Clone)]
pub struct BodyState {
    /// World-space position of the body's centre of mass.
    pub position: Vec3,
    /// Linear velocity.
    pub velocity: Vec3,
    /// Angular velocity.
    pub angular_velocity: Vec3,
    /// Inverse mass (0 = infinite mass / static).
    pub inv_mass: f32,
    /// Inverse inertia (scalar approximation).
    pub inv_inertia: f32,
    /// Orientation angle around primary axis (radians). Simplified single-axis.
    pub angle: f32,
}

impl BodyState {
    /// Create a dynamic body at the given position with the given mass.
    pub fn dynamic(position: Vec3, mass: f32) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self {
            position,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            inv_mass,
            inv_inertia: inv_mass * 0.4, // rough sphere-ish approximation
            angle: 0.0,
        }
    }

    /// Create a static (immovable) body.
    pub fn static_body(position: Vec3) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            inv_mass: 0.0,
            inv_inertia: 0.0,
            angle: 0.0,
        }
    }

    /// Whether this body is static (infinite mass).
    pub fn is_static(&self) -> bool {
        self.inv_mass == 0.0
    }
}

// ---------------------------------------------------------------------------
// Motor mode
// ---------------------------------------------------------------------------

/// The operating mode of a motor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MotorMode {
    /// Motor drives toward a target position.
    Position,
    /// Motor drives toward a target velocity.
    Velocity,
    /// Motor is disabled.
    Disabled,
}

// ===========================================================================
// MotorJoint
// ===========================================================================

/// Proportional-Derivative (PD) controller for computing motor forces.
///
/// The PD controller produces a force/torque value:
///   `output = Kp * error + Kd * error_derivative`
///
/// - `Kp` (proportional gain) controls how aggressively the joint seeks
///   its target.
/// - `Kd` (derivative gain) provides damping to prevent oscillation.
#[derive(Debug, Clone, Copy)]
pub struct PdController {
    /// Proportional gain.
    pub kp: f32,
    /// Derivative gain.
    pub kd: f32,
}

impl PdController {
    /// Create a PD controller with the given gains.
    pub fn new(kp: f32, kd: f32) -> Self {
        Self { kp, kd }
    }

    /// Create a critically damped PD controller for the given frequency.
    ///
    /// `frequency` is the desired natural frequency in Hz.
    /// `damping_ratio` should be 1.0 for critical damping.
    pub fn critically_damped(frequency: f32, damping_ratio: f32) -> Self {
        let omega = 2.0 * PI * frequency;
        Self {
            kp: omega * omega,
            kd: 2.0 * damping_ratio * omega,
        }
    }

    /// Compute the PD output given current error and its time derivative.
    pub fn compute(&self, error: f32, error_velocity: f32) -> f32 {
        self.kp * error + self.kd * error_velocity
    }

    /// Compute PD output clamped to a maximum magnitude.
    pub fn compute_clamped(&self, error: f32, error_velocity: f32, max_force: f32) -> f32 {
        let raw = self.compute(error, error_velocity);
        raw.clamp(-max_force, max_force)
    }
}

impl Default for PdController {
    fn default() -> Self {
        Self::critically_damped(5.0, 1.0)
    }
}

/// Configurable motorized joint with PD control.
///
/// Can operate in position-targeting or velocity-targeting mode on
/// both linear and angular axes independently. The motor applies
/// forces/torques computed via a PD controller, subject to a maximum
/// force limit.
pub struct MotorJoint {
    /// Handle of the first body.
    pub body_a: BodyHandle,
    /// Handle of the second body.
    pub body_b: BodyHandle,
    /// Anchor point on body A (local space).
    pub local_anchor_a: Vec3,
    /// Anchor point on body B (local space).
    pub local_anchor_b: Vec3,
    /// Linear motor mode.
    pub linear_mode: MotorMode,
    /// Angular motor mode.
    pub angular_mode: MotorMode,
    /// Target position for linear motor (world offset).
    pub linear_target: Vec3,
    /// Target velocity for linear motor.
    pub linear_target_velocity: Vec3,
    /// Target angle for angular motor (radians).
    pub angular_target: f32,
    /// Target angular velocity for angular motor (rad/s).
    pub angular_target_velocity: f32,
    /// Maximum linear force the motor can apply.
    pub max_linear_force: f32,
    /// Maximum angular torque the motor can apply.
    pub max_angular_torque: f32,
    /// PD controller for linear axis.
    pub linear_pd: PdController,
    /// PD controller for angular axis.
    pub angular_pd: PdController,
    /// Correction factor for position drift (Baumgarte stabilisation).
    pub correction_factor: f32,
    /// Accumulated linear impulse (for warm-starting).
    accumulated_linear_impulse: Vec3,
    /// Accumulated angular impulse.
    accumulated_angular_impulse: f32,
}

impl MotorJoint {
    /// Create a new motor joint between two bodies.
    pub fn new(body_a: BodyHandle, body_b: BodyHandle) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
            linear_mode: MotorMode::Position,
            angular_mode: MotorMode::Disabled,
            linear_target: Vec3::ZERO,
            linear_target_velocity: Vec3::ZERO,
            angular_target: 0.0,
            angular_target_velocity: 0.0,
            max_linear_force: 1000.0,
            max_angular_torque: 500.0,
            linear_pd: PdController::default(),
            angular_pd: PdController::default(),
            correction_factor: 0.2,
            accumulated_linear_impulse: Vec3::ZERO,
            accumulated_angular_impulse: 0.0,
        }
    }

    /// Set linear motor to position mode with the given target.
    pub fn set_linear_position_target(&mut self, target: Vec3) {
        self.linear_mode = MotorMode::Position;
        self.linear_target = target;
    }

    /// Set linear motor to velocity mode with the given target velocity.
    pub fn set_linear_velocity_target(&mut self, velocity: Vec3) {
        self.linear_mode = MotorMode::Velocity;
        self.linear_target_velocity = velocity;
    }

    /// Set angular motor to position mode with the given target angle.
    pub fn set_angular_position_target(&mut self, angle: f32) {
        self.angular_mode = MotorMode::Position;
        self.angular_target = angle;
    }

    /// Set angular motor to velocity mode.
    pub fn set_angular_velocity_target(&mut self, velocity: f32) {
        self.angular_mode = MotorMode::Velocity;
        self.angular_target_velocity = velocity;
    }

    /// Disable the linear motor.
    pub fn disable_linear_motor(&mut self) {
        self.linear_mode = MotorMode::Disabled;
    }

    /// Disable the angular motor.
    pub fn disable_angular_motor(&mut self) {
        self.angular_mode = MotorMode::Disabled;
    }

    /// Compute the linear motor force for one timestep.
    pub fn compute_linear_force(&self, a: &BodyState, b: &BodyState, dt: f32) -> Vec3 {
        match self.linear_mode {
            MotorMode::Disabled => Vec3::ZERO,
            MotorMode::Position => {
                let current_offset = b.position - a.position;
                let error = self.linear_target - current_offset;
                let error_velocity = b.velocity - a.velocity;
                // PD per axis
                let fx = self
                    .linear_pd
                    .compute_clamped(error.x, -error_velocity.x, self.max_linear_force);
                let fy = self
                    .linear_pd
                    .compute_clamped(error.y, -error_velocity.y, self.max_linear_force);
                let fz = self
                    .linear_pd
                    .compute_clamped(error.z, -error_velocity.z, self.max_linear_force);
                let force = Vec3::new(fx, fy, fz);
                // Clamp total force magnitude
                let mag = force.length();
                if mag > self.max_linear_force {
                    force * (self.max_linear_force / mag)
                } else {
                    force
                }
            }
            MotorMode::Velocity => {
                let relative_vel = b.velocity - a.velocity;
                let error_vel = self.linear_target_velocity - relative_vel;
                let fx = (self.linear_pd.kd * error_vel.x).clamp(
                    -self.max_linear_force,
                    self.max_linear_force,
                );
                let fy = (self.linear_pd.kd * error_vel.y).clamp(
                    -self.max_linear_force,
                    self.max_linear_force,
                );
                let fz = (self.linear_pd.kd * error_vel.z).clamp(
                    -self.max_linear_force,
                    self.max_linear_force,
                );
                Vec3::new(fx, fy, fz)
            }
        }
    }

    /// Compute the angular motor torque for one timestep.
    pub fn compute_angular_torque(&self, a: &BodyState, b: &BodyState, _dt: f32) -> f32 {
        match self.angular_mode {
            MotorMode::Disabled => 0.0,
            MotorMode::Position => {
                let current_angle = b.angle - a.angle;
                let error = self.angular_target - current_angle;
                // Wrap error to [-PI, PI]
                let wrapped = wrap_angle(error);
                let error_velocity =
                    b.angular_velocity.length() - a.angular_velocity.length();
                self.angular_pd
                    .compute_clamped(wrapped, -error_velocity, self.max_angular_torque)
            }
            MotorMode::Velocity => {
                let current_omega =
                    b.angular_velocity.length() - a.angular_velocity.length();
                let error = self.angular_target_velocity - current_omega;
                (self.angular_pd.kd * error).clamp(
                    -self.max_angular_torque,
                    self.max_angular_torque,
                )
            }
        }
    }

    /// Apply the motor joint for one solver iteration.
    ///
    /// Computes PD forces/torques and applies impulses to both bodies.
    pub fn solve(&mut self, a: &mut BodyState, b: &mut BodyState, dt: f32) {
        if dt <= 0.0 {
            return;
        }

        // Linear motor
        let linear_force = self.compute_linear_force(a, b, dt);
        let linear_impulse = linear_force * dt;
        a.velocity = a.velocity - linear_impulse * a.inv_mass;
        b.velocity = b.velocity + linear_impulse * b.inv_mass;
        self.accumulated_linear_impulse = self.accumulated_linear_impulse + linear_impulse;

        // Angular motor
        let angular_torque = self.compute_angular_torque(a, b, dt);
        let angular_impulse = angular_torque * dt;
        let ang_a = a.angular_velocity.length() - angular_impulse * a.inv_inertia;
        let ang_b = b.angular_velocity.length() + angular_impulse * b.inv_inertia;
        a.angular_velocity = Vec3::new(0.0, ang_a, 0.0);
        b.angular_velocity = Vec3::new(0.0, ang_b, 0.0);
        self.accumulated_angular_impulse += angular_impulse;
    }

    /// Reset accumulated impulses (call at the start of each physics step).
    pub fn reset_impulses(&mut self) {
        self.accumulated_linear_impulse = Vec3::ZERO;
        self.accumulated_angular_impulse = 0.0;
    }
}

/// Wrap an angle to the range [-PI, PI].
fn wrap_angle(angle: f32) -> f32 {
    let mut a = angle % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    }
    if a < -PI {
        a += 2.0 * PI;
    }
    a
}

// ===========================================================================
// RopeJoint
// ===========================================================================

/// Maximum-distance (rope) constraint.
///
/// The rope joint enforces that the distance between two anchor points
/// never exceeds `max_length`. When the bodies are closer than `max_length`
/// the joint is slack and applies no force. This is a unilateral constraint
/// — it only pulls, never pushes.
pub struct RopeJoint {
    /// Handle of the first body.
    pub body_a: BodyHandle,
    /// Handle of the second body.
    pub body_b: BodyHandle,
    /// Anchor on body A (local space).
    pub local_anchor_a: Vec3,
    /// Anchor on body B (local space).
    pub local_anchor_b: Vec3,
    /// Maximum allowed distance between anchors.
    pub max_length: f32,
    /// Baumgarte correction factor.
    pub correction_factor: f32,
    /// Damping coefficient for the constraint.
    pub damping: f32,
    /// Accumulated impulse for warm starting.
    accumulated_impulse: f32,
    /// Whether the rope is currently taut.
    is_taut: bool,
}

impl RopeJoint {
    /// Create a new rope joint with the given maximum length.
    pub fn new(body_a: BodyHandle, body_b: BodyHandle, max_length: f32) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
            max_length: max_length.max(0.0),
            correction_factor: 0.3,
            damping: 0.1,
            accumulated_impulse: 0.0,
            is_taut: false,
        }
    }

    /// Set anchor points in local space.
    pub fn set_anchors(&mut self, anchor_a: Vec3, anchor_b: Vec3) {
        self.local_anchor_a = anchor_a;
        self.local_anchor_b = anchor_b;
    }

    /// Check if the rope is currently taut.
    pub fn is_taut(&self) -> bool {
        self.is_taut
    }

    /// Get the current distance between anchor points (world space).
    pub fn current_distance(&self, a: &BodyState, b: &BodyState) -> f32 {
        let world_a = a.position + self.local_anchor_a;
        let world_b = b.position + self.local_anchor_b;
        (world_b - world_a).length()
    }

    /// Solve the rope constraint for one iteration.
    ///
    /// Only applies corrective impulses when the distance exceeds `max_length`.
    pub fn solve(&mut self, a: &mut BodyState, b: &mut BodyState, dt: f32) {
        if dt <= 0.0 {
            return;
        }

        let world_a = a.position + self.local_anchor_a;
        let world_b = b.position + self.local_anchor_b;
        let delta = world_b - world_a;
        let distance = delta.length();

        if distance <= self.max_length || distance < 1e-6 {
            // Slack — no constraint force
            self.is_taut = false;
            return;
        }

        self.is_taut = true;
        let direction = delta * (1.0 / distance);

        // Position error: how far beyond max_length
        let error = distance - self.max_length;

        // Relative velocity along the constraint axis
        let rel_vel = b.velocity - a.velocity;
        let rel_vel_along = rel_vel.dot(direction);

        // Effective mass
        let eff_mass_inv = a.inv_mass + b.inv_mass;
        if eff_mass_inv < 1e-10 {
            return;
        }
        let eff_mass = 1.0 / eff_mass_inv;

        // Baumgarte position correction + velocity damping
        let bias = (self.correction_factor / dt) * error;
        let impulse_mag = eff_mass * (rel_vel_along + bias + self.damping * rel_vel_along);

        // Rope can only pull (positive impulse along direction)
        let impulse_mag = impulse_mag.max(0.0);
        let impulse = direction * impulse_mag;

        // Apply impulses
        a.velocity = a.velocity + impulse * a.inv_mass;
        b.velocity = b.velocity - impulse * b.inv_mass;

        self.accumulated_impulse += impulse_mag;
    }

    /// Reset accumulated impulses.
    pub fn reset_impulses(&mut self) {
        self.accumulated_impulse = 0.0;
    }
}

// ===========================================================================
// WheelJoint
// ===========================================================================

/// Spring-damper suspension parameters for the wheel joint.
#[derive(Debug, Clone, Copy)]
pub struct SuspensionSettings {
    /// Spring stiffness (N/m).
    pub stiffness: f32,
    /// Damping coefficient (Ns/m).
    pub damping: f32,
    /// Rest length of the suspension (meters).
    pub rest_length: f32,
    /// Maximum compression (meters below rest length).
    pub max_compression: f32,
    /// Maximum extension (meters above rest length).
    pub max_extension: f32,
}

impl Default for SuspensionSettings {
    fn default() -> Self {
        Self {
            stiffness: 30000.0,
            damping: 4500.0,
            rest_length: 0.3,
            max_compression: 0.15,
            max_extension: 0.2,
        }
    }
}

/// Motor settings for the wheel's spin axis.
#[derive(Debug, Clone, Copy)]
pub struct WheelMotorSettings {
    /// Whether the motor is enabled.
    pub enabled: bool,
    /// Target angular velocity (rad/s).
    pub target_speed: f32,
    /// Maximum torque the motor can apply (Nm).
    pub max_torque: f32,
    /// Brake torque (applied when braking, opposes rotation).
    pub brake_torque: f32,
    /// Whether the brake is currently engaged.
    pub braking: bool,
}

impl Default for WheelMotorSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            target_speed: 0.0,
            max_torque: 1500.0,
            brake_torque: 3000.0,
            braking: false,
        }
    }
}

/// Vehicle wheel joint combining suspension and spin motor.
///
/// The wheel joint constrains a wheel body relative to a chassis body:
/// - **Suspension axis**: spring-damper perpendicular to ground (typically Y-up)
/// - **Spin axis**: free rotation for wheel rolling (typically X-local)
/// - **Steering**: rotation of the suspension axis about the chassis up-vector
pub struct WheelJoint {
    /// Chassis body handle.
    pub body_chassis: BodyHandle,
    /// Wheel body handle.
    pub body_wheel: BodyHandle,
    /// Suspension axis in chassis local space (typically up).
    pub suspension_axis: Vec3,
    /// Spin axis in wheel local space (rotation axis).
    pub spin_axis: Vec3,
    /// Anchor on chassis (local space).
    pub local_anchor_chassis: Vec3,
    /// Anchor on wheel (local space).
    pub local_anchor_wheel: Vec3,
    /// Suspension spring-damper settings.
    pub suspension: SuspensionSettings,
    /// Wheel motor settings.
    pub motor: WheelMotorSettings,
    /// Current steering angle (radians, positive = left turn).
    pub steering_angle: f32,
    /// Current suspension compression (meters from rest).
    pub current_compression: f32,
    /// Current wheel spin angular velocity (rad/s).
    pub current_spin_speed: f32,
    /// Wheel radius (for ground contact calculations).
    pub wheel_radius: f32,
    /// Whether the wheel is in contact with the ground.
    pub ground_contact: bool,
}

impl WheelJoint {
    /// Create a new wheel joint.
    pub fn new(body_chassis: BodyHandle, body_wheel: BodyHandle) -> Self {
        Self {
            body_chassis,
            body_wheel,
            suspension_axis: Vec3::Y,
            spin_axis: Vec3::X,
            local_anchor_chassis: Vec3::ZERO,
            local_anchor_wheel: Vec3::ZERO,
            suspension: SuspensionSettings::default(),
            motor: WheelMotorSettings::default(),
            steering_angle: 0.0,
            current_compression: 0.0,
            current_spin_speed: 0.0,
            wheel_radius: 0.35,
            ground_contact: false,
        }
    }

    /// Set the steering angle (radians).
    pub fn set_steering_angle(&mut self, angle: f32) {
        self.steering_angle = angle.clamp(-PI / 3.0, PI / 3.0); // max 60 degrees
    }

    /// Set motor speed target (rad/s).
    pub fn set_motor_speed(&mut self, speed: f32) {
        self.motor.enabled = true;
        self.motor.target_speed = speed;
    }

    /// Apply brakes.
    pub fn set_braking(&mut self, braking: bool) {
        self.motor.braking = braking;
    }

    /// Get the effective suspension axis accounting for steering.
    ///
    /// Rotates the suspension axis around the chassis up-vector by the
    /// steering angle.
    pub fn effective_suspension_axis(&self) -> Vec3 {
        // Rotate suspension_axis around Y by steering_angle
        let cos_a = self.steering_angle.cos();
        let sin_a = self.steering_angle.sin();
        Vec3::new(
            self.suspension_axis.x * cos_a + self.suspension_axis.z * sin_a,
            self.suspension_axis.y,
            -self.suspension_axis.x * sin_a + self.suspension_axis.z * cos_a,
        )
    }

    /// Compute the suspension spring-damper force.
    ///
    /// Returns the force along the suspension axis based on compression
    /// and compression velocity. Uses Hooke's law plus viscous damping:
    ///   `F = -k * compression - c * compression_velocity`
    pub fn compute_suspension_force(&self, compression: f32, compression_velocity: f32) -> f32 {
        let clamped_compression = compression.clamp(
            -self.suspension.max_extension,
            self.suspension.max_compression,
        );
        let spring_force = self.suspension.stiffness * clamped_compression;
        let damping_force = self.suspension.damping * compression_velocity;
        spring_force - damping_force
    }

    /// Compute the motor torque to apply to the wheel.
    pub fn compute_motor_torque(&self) -> f32 {
        if self.motor.braking {
            // Brake: oppose current rotation
            let brake = if self.current_spin_speed > 0.0 {
                -self.motor.brake_torque
            } else if self.current_spin_speed < 0.0 {
                self.motor.brake_torque
            } else {
                0.0
            };
            return brake.clamp(-self.motor.brake_torque, self.motor.brake_torque);
        }

        if !self.motor.enabled {
            return 0.0;
        }

        let speed_error = self.motor.target_speed - self.current_spin_speed;
        let torque = speed_error * 10.0; // simple proportional control
        torque.clamp(-self.motor.max_torque, self.motor.max_torque)
    }

    /// Solve the wheel joint for one physics step.
    pub fn solve(
        &mut self,
        chassis: &mut BodyState,
        wheel: &mut BodyState,
        dt: f32,
    ) {
        if dt <= 0.0 {
            return;
        }

        let susp_axis = self.effective_suspension_axis();

        // Compute current suspension compression
        let chassis_anchor = chassis.position + self.local_anchor_chassis;
        let wheel_anchor = wheel.position + self.local_anchor_wheel;
        let displacement = wheel_anchor - chassis_anchor;
        let projection = displacement.dot(susp_axis);
        let compression = self.suspension.rest_length - projection;
        self.current_compression = compression;

        // Compute compression velocity
        let rel_vel = wheel.velocity - chassis.velocity;
        let compression_vel = -rel_vel.dot(susp_axis);

        // Suspension force
        let susp_force = self.compute_suspension_force(compression, compression_vel);
        let force_vector = susp_axis * susp_force;

        // Apply suspension impulses
        let impulse = force_vector * dt;
        chassis.velocity = chassis.velocity - impulse * chassis.inv_mass;
        wheel.velocity = wheel.velocity + impulse * wheel.inv_mass;

        // Motor torque
        let torque = self.compute_motor_torque();
        let angular_impulse = torque * dt;
        self.current_spin_speed += angular_impulse * wheel.inv_inertia;
    }
}

// ===========================================================================
// PrismaticJoint
// ===========================================================================

/// Prismatic (slider) joint: constrains relative motion to a single axis.
///
/// The joint allows translation along a defined axis with optional
/// limits and a motor for powered linear motion. All other degrees of
/// freedom are locked.
pub struct PrismaticJoint {
    /// Handle of the first body.
    pub body_a: BodyHandle,
    /// Handle of the second body.
    pub body_b: BodyHandle,
    /// The sliding axis in world space.
    pub axis: Vec3,
    /// Anchor on body A (local space).
    pub local_anchor_a: Vec3,
    /// Anchor on body B (local space).
    pub local_anchor_b: Vec3,
    /// Whether translation limits are enabled.
    pub limits_enabled: bool,
    /// Minimum translation along axis.
    pub min_translation: f32,
    /// Maximum translation along axis.
    pub max_translation: f32,
    /// Whether the motor is enabled.
    pub motor_enabled: bool,
    /// Motor mode (position or velocity).
    pub motor_mode: MotorMode,
    /// Motor target position (meters along axis).
    pub motor_target_position: f32,
    /// Motor target velocity (m/s along axis).
    pub motor_target_velocity: f32,
    /// Maximum motor force.
    pub motor_max_force: f32,
    /// PD controller for the motor.
    pub motor_pd: PdController,
    /// Baumgarte correction factor.
    pub correction_factor: f32,
    /// Current translation along the axis.
    current_translation: f32,
    /// Accumulated impulse.
    accumulated_impulse: f32,
}

impl PrismaticJoint {
    /// Create a new prismatic joint along the given axis.
    pub fn new(body_a: BodyHandle, body_b: BodyHandle, axis: Vec3) -> Self {
        Self {
            body_a,
            body_b,
            axis: axis.normalize(),
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
            limits_enabled: false,
            min_translation: -1.0,
            max_translation: 1.0,
            motor_enabled: false,
            motor_mode: MotorMode::Disabled,
            motor_target_position: 0.0,
            motor_target_velocity: 0.0,
            motor_max_force: 500.0,
            motor_pd: PdController::default(),
            correction_factor: 0.2,
            current_translation: 0.0,
            accumulated_impulse: 0.0,
        }
    }

    /// Enable translation limits.
    pub fn set_limits(&mut self, min: f32, max: f32) {
        self.limits_enabled = true;
        self.min_translation = min.min(max);
        self.max_translation = min.max(max);
    }

    /// Disable translation limits.
    pub fn disable_limits(&mut self) {
        self.limits_enabled = false;
    }

    /// Enable position motor.
    pub fn set_motor_position(&mut self, target: f32, max_force: f32) {
        self.motor_enabled = true;
        self.motor_mode = MotorMode::Position;
        self.motor_target_position = target;
        self.motor_max_force = max_force;
    }

    /// Enable velocity motor.
    pub fn set_motor_velocity(&mut self, velocity: f32, max_force: f32) {
        self.motor_enabled = true;
        self.motor_mode = MotorMode::Velocity;
        self.motor_target_velocity = velocity;
        self.motor_max_force = max_force;
    }

    /// Disable the motor.
    pub fn disable_motor(&mut self) {
        self.motor_enabled = false;
        self.motor_mode = MotorMode::Disabled;
    }

    /// Get the current translation along the axis.
    pub fn current_translation(&self) -> f32 {
        self.current_translation
    }

    /// Solve the prismatic joint constraint.
    pub fn solve(&mut self, a: &mut BodyState, b: &mut BodyState, dt: f32) {
        if dt <= 0.0 {
            return;
        }

        let world_a = a.position + self.local_anchor_a;
        let world_b = b.position + self.local_anchor_b;
        let delta = world_b - world_a;

        // Current translation along axis
        let translation = delta.dot(self.axis);
        self.current_translation = translation;

        // -- Constrain perpendicular motion (lock off-axis DOFs) --
        let on_axis = self.axis * translation;
        let off_axis = delta - on_axis;
        let off_axis_len = off_axis.length();

        let eff_mass_inv = a.inv_mass + b.inv_mass;
        if eff_mass_inv < 1e-10 {
            return;
        }
        let eff_mass = 1.0 / eff_mass_inv;

        // Correct off-axis drift
        if off_axis_len > 1e-6 {
            let correction_dir = off_axis * (1.0 / off_axis_len);
            let rel_vel = b.velocity - a.velocity;
            let off_vel = rel_vel.dot(correction_dir);
            let bias = (self.correction_factor / dt) * off_axis_len;
            let impulse_mag = eff_mass * (off_vel + bias);
            let impulse = correction_dir * impulse_mag;
            a.velocity = a.velocity + impulse * a.inv_mass;
            b.velocity = b.velocity - impulse * b.inv_mass;
        }

        // -- Translation limits --
        if self.limits_enabled {
            if translation < self.min_translation {
                let error = self.min_translation - translation;
                let rel_vel_axis = (b.velocity - a.velocity).dot(self.axis);
                let bias = (self.correction_factor / dt) * error;
                let impulse_mag = (eff_mass * (-rel_vel_axis + bias)).max(0.0);
                let impulse = self.axis * impulse_mag;
                a.velocity = a.velocity - impulse * a.inv_mass;
                b.velocity = b.velocity + impulse * b.inv_mass;
            } else if translation > self.max_translation {
                let error = translation - self.max_translation;
                let rel_vel_axis = (b.velocity - a.velocity).dot(self.axis);
                let bias = (self.correction_factor / dt) * error;
                let impulse_mag = (eff_mass * (rel_vel_axis + bias)).max(0.0);
                let impulse = self.axis * impulse_mag;
                a.velocity = a.velocity + impulse * a.inv_mass;
                b.velocity = b.velocity - impulse * b.inv_mass;
            }
        }

        // -- Motor --
        if self.motor_enabled {
            let rel_vel_axis = (b.velocity - a.velocity).dot(self.axis);
            let motor_impulse = match self.motor_mode {
                MotorMode::Position => {
                    let error = self.motor_target_position - translation;
                    let force = self.motor_pd.compute_clamped(
                        error,
                        -rel_vel_axis,
                        self.motor_max_force,
                    );
                    force * dt
                }
                MotorMode::Velocity => {
                    let vel_error = self.motor_target_velocity - rel_vel_axis;
                    let force = (self.motor_pd.kd * vel_error).clamp(
                        -self.motor_max_force,
                        self.motor_max_force,
                    );
                    force * dt
                }
                MotorMode::Disabled => 0.0,
            };

            let impulse = self.axis * motor_impulse;
            a.velocity = a.velocity - impulse * a.inv_mass;
            b.velocity = b.velocity + impulse * b.inv_mass;
            self.accumulated_impulse += motor_impulse;
        }
    }

    /// Reset accumulated impulses.
    pub fn reset_impulses(&mut self) {
        self.accumulated_impulse = 0.0;
    }
}

// ===========================================================================
// GenericJoint (6-DOF)
// ===========================================================================

/// Configuration for a single joint axis.
///
/// Each of the six degrees of freedom (3 linear, 3 angular) can be
/// independently configured as locked, limited, free, or motor-driven.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointAxisConfig {
    /// Axis is completely locked (no relative motion).
    Locked,
    /// Axis has limited range of motion.
    Limited {
        /// Minimum value (meters for linear, radians for angular).
        min: f32,
        /// Maximum value.
        max: f32,
    },
    /// Axis is completely free (no constraint).
    Free,
    /// Axis is motor-driven toward a target.
    Motor {
        /// Target value (position or angle).
        target: f32,
        /// Maximum force/torque the motor can apply.
        max_force: f32,
    },
}

impl JointAxisConfig {
    /// Create a limited axis config.
    pub fn limited(min: f32, max: f32) -> Self {
        Self::Limited {
            min: min.min(max),
            max: min.max(max),
        }
    }

    /// Create a motor axis config.
    pub fn motor(target: f32, max_force: f32) -> Self {
        Self::Motor {
            target,
            max_force: max_force.abs(),
        }
    }
}

/// Axis index for the 6-DOF joint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JointAxis {
    /// Linear X axis.
    LinearX = 0,
    /// Linear Y axis.
    LinearY = 1,
    /// Linear Z axis.
    LinearZ = 2,
    /// Angular X axis (pitch).
    AngularX = 3,
    /// Angular Y axis (yaw).
    AngularY = 4,
    /// Angular Z axis (roll).
    AngularZ = 5,
}

impl JointAxis {
    /// Return the unit direction vector for this axis.
    pub fn direction(&self) -> Vec3 {
        match self {
            Self::LinearX | Self::AngularX => Vec3::X,
            Self::LinearY | Self::AngularY => Vec3::Y,
            Self::LinearZ | Self::AngularZ => Vec3::Z,
        }
    }

    /// Whether this axis is linear (translation).
    pub fn is_linear(&self) -> bool {
        matches!(self, Self::LinearX | Self::LinearY | Self::LinearZ)
    }

    /// Whether this axis is angular (rotation).
    pub fn is_angular(&self) -> bool {
        !self.is_linear()
    }
}

/// Fully configurable 6 degree-of-freedom joint.
///
/// Each axis (LinearX, LinearY, LinearZ, AngularX, AngularY, AngularZ)
/// can be independently configured as Locked, Limited, Free, or Motor.
/// This joint can represent any of the other joint types:
///
/// - **Fixed joint**: all axes Locked
/// - **Ball joint**: linear axes Locked, angular axes Free
/// - **Hinge joint**: linear Locked, one angular Free, others Locked
/// - **Prismatic**: one linear Free/Limited, rest Locked
/// - **Motor joint**: axes set to Motor mode
pub struct GenericJoint {
    /// Handle of the first body.
    pub body_a: BodyHandle,
    /// Handle of the second body.
    pub body_b: BodyHandle,
    /// Per-axis configuration (indexed by JointAxis ordinal).
    pub axes: [JointAxisConfig; 6],
    /// Anchor on body A (local space).
    pub local_anchor_a: Vec3,
    /// Anchor on body B (local space).
    pub local_anchor_b: Vec3,
    /// Baumgarte correction factor.
    pub correction_factor: f32,
    /// PD controller gains for motor axes.
    pub motor_pd: PdController,
    /// Accumulated impulses per axis.
    accumulated_impulses: [f32; 6],
}

impl GenericJoint {
    /// Create a new generic joint with all axes locked (fixed joint).
    pub fn new_fixed(body_a: BodyHandle, body_b: BodyHandle) -> Self {
        Self {
            body_a,
            body_b,
            axes: [JointAxisConfig::Locked; 6],
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
            correction_factor: 0.2,
            motor_pd: PdController::default(),
            accumulated_impulses: [0.0; 6],
        }
    }

    /// Create a ball-socket joint (linear locked, angular free).
    pub fn new_ball(body_a: BodyHandle, body_b: BodyHandle) -> Self {
        let mut joint = Self::new_fixed(body_a, body_b);
        joint.axes[JointAxis::AngularX as usize] = JointAxisConfig::Free;
        joint.axes[JointAxis::AngularY as usize] = JointAxisConfig::Free;
        joint.axes[JointAxis::AngularZ as usize] = JointAxisConfig::Free;
        joint
    }

    /// Create a hinge joint (rotation around one axis).
    pub fn new_hinge(body_a: BodyHandle, body_b: BodyHandle, hinge_axis: JointAxis) -> Self {
        let mut joint = Self::new_fixed(body_a, body_b);
        if hinge_axis.is_angular() {
            joint.axes[hinge_axis as usize] = JointAxisConfig::Free;
        }
        joint
    }

    /// Create a prismatic joint (translation along one axis).
    pub fn new_prismatic(body_a: BodyHandle, body_b: BodyHandle, slide_axis: JointAxis) -> Self {
        let mut joint = Self::new_fixed(body_a, body_b);
        if slide_axis.is_linear() {
            joint.axes[slide_axis as usize] = JointAxisConfig::Free;
        }
        joint
    }

    /// Set the configuration for a specific axis.
    pub fn set_axis(&mut self, axis: JointAxis, config: JointAxisConfig) {
        self.axes[axis as usize] = config;
    }

    /// Get the configuration for a specific axis.
    pub fn get_axis(&self, axis: JointAxis) -> JointAxisConfig {
        self.axes[axis as usize]
    }

    /// Set all linear axes to the same config.
    pub fn set_all_linear(&mut self, config: JointAxisConfig) {
        self.axes[JointAxis::LinearX as usize] = config;
        self.axes[JointAxis::LinearY as usize] = config;
        self.axes[JointAxis::LinearZ as usize] = config;
    }

    /// Set all angular axes to the same config.
    pub fn set_all_angular(&mut self, config: JointAxisConfig) {
        self.axes[JointAxis::AngularX as usize] = config;
        self.axes[JointAxis::AngularY as usize] = config;
        self.axes[JointAxis::AngularZ as usize] = config;
    }

    /// Solve the generic joint for one iteration.
    pub fn solve(&mut self, a: &mut BodyState, b: &mut BodyState, dt: f32) {
        if dt <= 0.0 {
            return;
        }

        let eff_mass_inv = a.inv_mass + b.inv_mass;
        if eff_mass_inv < 1e-10 {
            return;
        }
        let eff_mass = 1.0 / eff_mass_inv;

        let world_a = a.position + self.local_anchor_a;
        let world_b = b.position + self.local_anchor_b;
        let delta = world_b - world_a;
        let rel_vel = b.velocity - a.velocity;

        // Process linear axes
        let linear_axes = [
            (JointAxis::LinearX, Vec3::X),
            (JointAxis::LinearY, Vec3::Y),
            (JointAxis::LinearZ, Vec3::Z),
        ];

        for (axis, dir) in &linear_axes {
            let idx = *axis as usize;
            let config = self.axes[idx];
            let projection = delta.dot(*dir);
            let vel_proj = rel_vel.dot(*dir);

            match config {
                JointAxisConfig::Locked => {
                    // Correct position error and kill velocity along this axis
                    let bias = (self.correction_factor / dt) * projection;
                    let impulse_mag = eff_mass * (vel_proj + bias);
                    let impulse = *dir * impulse_mag;
                    a.velocity = a.velocity + impulse * a.inv_mass;
                    b.velocity = b.velocity - impulse * b.inv_mass;
                    self.accumulated_impulses[idx] += impulse_mag;
                }
                JointAxisConfig::Limited { min, max } => {
                    if projection < min {
                        let error = min - projection;
                        let bias = (self.correction_factor / dt) * error;
                        let impulse_mag = (eff_mass * (-vel_proj + bias)).max(0.0);
                        let impulse = *dir * impulse_mag;
                        a.velocity = a.velocity - impulse * a.inv_mass;
                        b.velocity = b.velocity + impulse * b.inv_mass;
                        self.accumulated_impulses[idx] += impulse_mag;
                    } else if projection > max {
                        let error = projection - max;
                        let bias = (self.correction_factor / dt) * error;
                        let impulse_mag = (eff_mass * (vel_proj + bias)).max(0.0);
                        let impulse = *dir * impulse_mag;
                        a.velocity = a.velocity + impulse * a.inv_mass;
                        b.velocity = b.velocity - impulse * b.inv_mass;
                        self.accumulated_impulses[idx] += impulse_mag;
                    }
                }
                JointAxisConfig::Free => {
                    // No constraint
                }
                JointAxisConfig::Motor { target, max_force } => {
                    let error = target - projection;
                    let force =
                        self.motor_pd.compute_clamped(error, -vel_proj, max_force);
                    let impulse_mag = force * dt;
                    let impulse = *dir * impulse_mag;
                    a.velocity = a.velocity - impulse * a.inv_mass;
                    b.velocity = b.velocity + impulse * b.inv_mass;
                    self.accumulated_impulses[idx] += impulse_mag;
                }
            }
        }

        // Process angular axes (simplified — using scalar angle differences)
        let angular_axes = [
            JointAxis::AngularX,
            JointAxis::AngularY,
            JointAxis::AngularZ,
        ];

        let eff_inertia_inv = a.inv_inertia + b.inv_inertia;
        if eff_inertia_inv < 1e-10 {
            return;
        }
        let eff_inertia = 1.0 / eff_inertia_inv;

        for axis in &angular_axes {
            let idx = *axis as usize;
            let config = self.axes[idx];
            let dir = axis.direction();

            // Project angular velocity onto this axis
            let omega_a = a.angular_velocity.dot(dir);
            let omega_b = b.angular_velocity.dot(dir);
            let rel_omega = omega_b - omega_a;

            // Approximate angle difference from angular velocities
            // (in a full engine we'd use quaternion-based angle extraction)
            let angle_diff = (b.angle - a.angle) * dir.dot(Vec3::Y).abs();

            match config {
                JointAxisConfig::Locked => {
                    let bias = (self.correction_factor / dt) * angle_diff;
                    let impulse_mag = eff_inertia * (rel_omega + bias);
                    a.angular_velocity = a.angular_velocity + dir * (impulse_mag * a.inv_inertia);
                    b.angular_velocity = b.angular_velocity - dir * (impulse_mag * b.inv_inertia);
                    self.accumulated_impulses[idx] += impulse_mag;
                }
                JointAxisConfig::Limited { min, max } => {
                    if angle_diff < min {
                        let error = min - angle_diff;
                        let bias = (self.correction_factor / dt) * error;
                        let impulse_mag = (eff_inertia * (-rel_omega + bias)).max(0.0);
                        a.angular_velocity =
                            a.angular_velocity - dir * (impulse_mag * a.inv_inertia);
                        b.angular_velocity =
                            b.angular_velocity + dir * (impulse_mag * b.inv_inertia);
                        self.accumulated_impulses[idx] += impulse_mag;
                    } else if angle_diff > max {
                        let error = angle_diff - max;
                        let bias = (self.correction_factor / dt) * error;
                        let impulse_mag = (eff_inertia * (rel_omega + bias)).max(0.0);
                        a.angular_velocity =
                            a.angular_velocity + dir * (impulse_mag * a.inv_inertia);
                        b.angular_velocity =
                            b.angular_velocity - dir * (impulse_mag * b.inv_inertia);
                        self.accumulated_impulses[idx] += impulse_mag;
                    }
                }
                JointAxisConfig::Free => {}
                JointAxisConfig::Motor { target, max_force } => {
                    let error = wrap_angle(target - angle_diff);
                    let torque =
                        self.motor_pd.compute_clamped(error, -rel_omega, max_force);
                    let impulse_mag = torque * dt;
                    a.angular_velocity =
                        a.angular_velocity - dir * (impulse_mag * a.inv_inertia);
                    b.angular_velocity =
                        b.angular_velocity + dir * (impulse_mag * b.inv_inertia);
                    self.accumulated_impulses[idx] += impulse_mag;
                }
            }
        }
    }

    /// Reset accumulated impulses for all axes.
    pub fn reset_impulses(&mut self) {
        self.accumulated_impulses = [0.0; 6];
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pd_controller_zero_error() {
        let pd = PdController::new(100.0, 10.0);
        let output = pd.compute(0.0, 0.0);
        assert_eq!(output, 0.0);
    }

    #[test]
    fn pd_controller_proportional() {
        let pd = PdController::new(100.0, 0.0);
        let output = pd.compute(1.0, 0.0);
        assert_eq!(output, 100.0);
    }

    #[test]
    fn pd_controller_derivative() {
        let pd = PdController::new(0.0, 10.0);
        let output = pd.compute(0.0, 5.0);
        assert_eq!(output, 50.0);
    }

    #[test]
    fn pd_controller_clamped() {
        let pd = PdController::new(1000.0, 0.0);
        let output = pd.compute_clamped(10.0, 0.0, 500.0);
        assert_eq!(output, 500.0);
    }

    #[test]
    fn motor_joint_position_mode() {
        let mut a = BodyState::dynamic(Vec3::ZERO, 1.0);
        let mut b = BodyState::dynamic(Vec3::new(1.0, 0.0, 0.0), 1.0);
        let mut joint = MotorJoint::new(0, 1);
        joint.set_linear_position_target(Vec3::new(2.0, 0.0, 0.0));
        joint.linear_pd = PdController::new(100.0, 10.0);
        joint.solve(&mut a, &mut b, 0.016);
        // B should gain velocity toward target
        assert!(b.velocity.x > 0.0, "motor should push B in +X");
    }

    #[test]
    fn rope_joint_slack() {
        let mut a = BodyState::dynamic(Vec3::ZERO, 1.0);
        let mut b = BodyState::dynamic(Vec3::new(1.0, 0.0, 0.0), 1.0);
        let mut rope = RopeJoint::new(0, 1, 5.0);
        let vel_before = b.velocity;
        rope.solve(&mut a, &mut b, 0.016);
        // Should be slack — no velocity change
        assert!(!rope.is_taut());
        assert_eq!(b.velocity.x, vel_before.x);
    }

    #[test]
    fn rope_joint_taut() {
        let mut a = BodyState::dynamic(Vec3::ZERO, 1.0);
        let mut b = BodyState::dynamic(Vec3::new(10.0, 0.0, 0.0), 1.0);
        let mut rope = RopeJoint::new(0, 1, 5.0);
        rope.solve(&mut a, &mut b, 0.016);
        assert!(rope.is_taut());
    }

    #[test]
    fn wheel_joint_suspension_force() {
        let wheel = WheelJoint::new(0, 1);
        // Positive compression = compressed
        let force = wheel.compute_suspension_force(0.1, 0.0);
        assert!(force > 0.0, "compressed spring should push out");
        // Negative compression = extended
        let force_ext = wheel.compute_suspension_force(-0.1, 0.0);
        assert!(force_ext < 0.0, "extended spring should pull in");
    }

    #[test]
    fn prismatic_joint_limits() {
        let mut a = BodyState::dynamic(Vec3::ZERO, 1.0);
        let mut b = BodyState::dynamic(Vec3::new(2.0, 0.0, 0.0), 1.0);
        let mut joint = PrismaticJoint::new(0, 1, Vec3::X);
        joint.set_limits(0.0, 1.0); // B can be at most 1m from A along X
        joint.solve(&mut a, &mut b, 0.016);
        // B should be pushed back (velocity should decrease toward A)
        assert!(b.velocity.x < 0.0 || a.velocity.x > 0.0);
    }

    #[test]
    fn generic_joint_fixed() {
        let mut a = BodyState::dynamic(Vec3::ZERO, 1.0);
        let mut b = BodyState::dynamic(Vec3::new(0.1, 0.0, 0.0), 1.0);
        let mut joint = GenericJoint::new_fixed(0, 1);
        joint.solve(&mut a, &mut b, 0.016);
        // Bodies should be pushed toward each other
        assert!(a.velocity.x > 0.0 || b.velocity.x < 0.0);
    }

    #[test]
    fn generic_joint_free_axis() {
        let mut a = BodyState::dynamic(Vec3::ZERO, 1.0);
        let mut b = BodyState::dynamic(Vec3::new(1.0, 0.0, 0.0), 1.0);
        b.velocity = Vec3::new(5.0, 0.0, 0.0);
        let mut joint = GenericJoint::new_fixed(0, 1);
        joint.set_axis(JointAxis::LinearX, JointAxisConfig::Free);
        // Y and Z are locked but X is free — only Y/Z should be constrained
        let vel_before = b.velocity.x;
        joint.solve(&mut a, &mut b, 0.016);
        // X velocity should not be affected by the free axis
        // (only Y and Z are locked, and they have zero offset)
        assert!((b.velocity.x - vel_before).abs() < 0.01);
    }

    #[test]
    fn wrap_angle_test() {
        assert!((wrap_angle(0.0)).abs() < 1e-6);
        assert!((wrap_angle(PI) - PI).abs() < 1e-6);
        assert!((wrap_angle(3.0 * PI) - PI).abs() < 1e-4);
        assert!((wrap_angle(-3.0 * PI) + PI).abs() < 1e-4);
    }

    #[test]
    fn vec3_operations() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let sum = a + b;
        assert_eq!(sum.x, 5.0);
        assert_eq!(sum.y, 7.0);
        assert_eq!(sum.z, 9.0);

        let dot = a.dot(b);
        assert_eq!(dot, 32.0); // 4+10+18

        let cross = Vec3::X.cross(Vec3::Y);
        assert!((cross.z - 1.0).abs() < 1e-6);
    }
}
