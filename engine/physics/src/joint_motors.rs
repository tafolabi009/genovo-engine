// engine/physics/src/joint_motors.rs
//
// Joint motor system for the Genovo engine.
//
// Provides motor controllers for physics joints to drive controlled motion:
//
// - **Velocity motors** -- Drive a joint axis at a target angular/linear velocity.
// - **Position motors (servo)** -- Move a joint to a target position/angle with
//   PD control.
// - **Spring motors** -- Simulate spring-damper behavior on joint axes.
// - **Motor limits** -- Configurable force/torque limits to prevent unrealistic
//   forces.
// - **Motor force feedback** -- Report the force/torque being applied by each
//   motor for gameplay use (e.g., steering feel).
// - **Motor enable/disable** -- Toggle motors without removing them.
// - **Motor profiles** -- Pre-built motor configurations for common use cases.

use std::fmt;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EPSILON: f32 = 1e-6;
const DEFAULT_MAX_FORCE: f32 = 1000.0;
const DEFAULT_MAX_TORQUE: f32 = 500.0;
const PI: f32 = std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Motor axis
// ---------------------------------------------------------------------------

/// Which joint axis the motor acts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MotorAxis {
    /// Linear axis X.
    LinearX,
    /// Linear axis Y.
    LinearY,
    /// Linear axis Z.
    LinearZ,
    /// Angular axis X (roll).
    AngularX,
    /// Angular axis Y (yaw).
    AngularY,
    /// Angular axis Z (pitch).
    AngularZ,
}

impl MotorAxis {
    /// Whether this is a linear axis.
    pub fn is_linear(&self) -> bool {
        matches!(self, Self::LinearX | Self::LinearY | Self::LinearZ)
    }

    /// Whether this is an angular axis.
    pub fn is_angular(&self) -> bool {
        !self.is_linear()
    }

    /// Get the axis index (0=X, 1=Y, 2=Z).
    pub fn index(&self) -> usize {
        match self {
            Self::LinearX | Self::AngularX => 0,
            Self::LinearY | Self::AngularY => 1,
            Self::LinearZ | Self::AngularZ => 2,
        }
    }
}

impl fmt::Display for MotorAxis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LinearX => write!(f, "LinearX"),
            Self::LinearY => write!(f, "LinearY"),
            Self::LinearZ => write!(f, "LinearZ"),
            Self::AngularX => write!(f, "AngularX"),
            Self::AngularY => write!(f, "AngularY"),
            Self::AngularZ => write!(f, "AngularZ"),
        }
    }
}

// ---------------------------------------------------------------------------
// Motor mode
// ---------------------------------------------------------------------------

/// The control mode for a motor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MotorMode {
    /// Disabled -- motor applies no force.
    Disabled,
    /// Velocity mode -- drive towards a target velocity.
    Velocity,
    /// Position (servo) mode -- drive towards a target position with PD control.
    Position,
    /// Spring mode -- apply spring-damper forces.
    Spring,
    /// Free-spin mode -- no resistance (acts as a free joint).
    Free,
}

impl Default for MotorMode {
    fn default() -> Self {
        Self::Disabled
    }
}

impl fmt::Display for MotorMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => write!(f, "Disabled"),
            Self::Velocity => write!(f, "Velocity"),
            Self::Position => write!(f, "Position"),
            Self::Spring => write!(f, "Spring"),
            Self::Free => write!(f, "Free"),
        }
    }
}

// ---------------------------------------------------------------------------
// Motor limits
// ---------------------------------------------------------------------------

/// Force/torque limits for a motor.
#[derive(Debug, Clone, Copy)]
pub struct MotorLimits {
    /// Maximum force (for linear motors) in Newtons.
    pub max_force: f32,
    /// Maximum torque (for angular motors) in Newton-meters.
    pub max_torque: f32,
    /// Minimum position/angle limit.
    pub min_position: f32,
    /// Maximum position/angle limit.
    pub max_position: f32,
    /// Whether position limits are enabled.
    pub limits_enabled: bool,
}

impl Default for MotorLimits {
    fn default() -> Self {
        Self {
            max_force: DEFAULT_MAX_FORCE,
            max_torque: DEFAULT_MAX_TORQUE,
            min_position: -f32::MAX,
            max_position: f32::MAX,
            limits_enabled: false,
        }
    }
}

impl MotorLimits {
    /// Create limits with angular range.
    pub fn angular(min_deg: f32, max_deg: f32, max_torque: f32) -> Self {
        Self {
            max_force: DEFAULT_MAX_FORCE,
            max_torque,
            min_position: min_deg.to_radians(),
            max_position: max_deg.to_radians(),
            limits_enabled: true,
        }
    }

    /// Create limits with linear range.
    pub fn linear(min_pos: f32, max_pos: f32, max_force: f32) -> Self {
        Self {
            max_force,
            max_torque: DEFAULT_MAX_TORQUE,
            min_position: min_pos,
            max_position: max_pos,
            limits_enabled: true,
        }
    }

    /// Clamp a force/torque value to the limits.
    pub fn clamp_force(&self, force: f32, is_angular: bool) -> f32 {
        let max = if is_angular { self.max_torque } else { self.max_force };
        force.clamp(-max, max)
    }

    /// Clamp a position to the limits.
    pub fn clamp_position(&self, position: f32) -> f32 {
        if self.limits_enabled {
            position.clamp(self.min_position, self.max_position)
        } else {
            position
        }
    }
}

// ---------------------------------------------------------------------------
// Spring-damper parameters
// ---------------------------------------------------------------------------

/// Spring-damper parameters for spring motor mode.
#[derive(Debug, Clone, Copy)]
pub struct SpringDamperParams {
    /// Spring stiffness (N/m for linear, Nm/rad for angular).
    pub stiffness: f32,
    /// Damping coefficient.
    pub damping: f32,
    /// Rest position (equilibrium).
    pub rest_position: f32,
}

impl Default for SpringDamperParams {
    fn default() -> Self {
        Self {
            stiffness: 100.0,
            damping: 10.0,
            rest_position: 0.0,
        }
    }
}

impl SpringDamperParams {
    /// Compute the spring-damper force given current state.
    pub fn compute_force(&self, position: f32, velocity: f32) -> f32 {
        let displacement = position - self.rest_position;
        -self.stiffness * displacement - self.damping * velocity
    }

    /// Compute the natural frequency.
    pub fn natural_frequency(&self, mass: f32) -> f32 {
        if mass > EPSILON {
            (self.stiffness / mass).sqrt()
        } else {
            0.0
        }
    }

    /// Compute the damping ratio (1.0 = critical, <1.0 = underdamped).
    pub fn damping_ratio(&self, mass: f32) -> f32 {
        if mass > EPSILON && self.stiffness > EPSILON {
            self.damping / (2.0 * (self.stiffness * mass).sqrt())
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// PD controller
// ---------------------------------------------------------------------------

/// PD controller for position (servo) motor mode.
#[derive(Debug, Clone, Copy)]
pub struct PdController {
    /// Proportional gain.
    pub kp: f32,
    /// Derivative gain.
    pub kd: f32,
    /// Target position.
    pub target: f32,
    /// Previous error (for derivative term).
    prev_error: f32,
}

impl Default for PdController {
    fn default() -> Self {
        Self {
            kp: 100.0,
            kd: 20.0,
            target: 0.0,
            prev_error: 0.0,
        }
    }
}

impl PdController {
    /// Create a PD controller with the given gains.
    pub fn new(kp: f32, kd: f32) -> Self {
        Self {
            kp,
            kd,
            target: 0.0,
            prev_error: 0.0,
        }
    }

    /// Set the target position.
    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }

    /// Compute the control force.
    pub fn compute(&mut self, current_position: f32, current_velocity: f32, dt: f32) -> f32 {
        let error = self.target - current_position;
        let derivative = if dt > EPSILON {
            (error - self.prev_error) / dt
        } else {
            -current_velocity
        };
        self.prev_error = error;
        self.kp * error + self.kd * derivative
    }

    /// Reset the controller state.
    pub fn reset(&mut self) {
        self.prev_error = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Motor feedback
// ---------------------------------------------------------------------------

/// Force/torque feedback data from a motor.
#[derive(Debug, Clone, Copy, Default)]
pub struct MotorFeedback {
    /// Applied force/torque this frame.
    pub applied_force: f32,
    /// Clamped (limited) force/torque.
    pub clamped_force: f32,
    /// Current position on the motor axis.
    pub current_position: f32,
    /// Current velocity on the motor axis.
    pub current_velocity: f32,
    /// Position error (target - current).
    pub position_error: f32,
    /// Velocity error (target - current).
    pub velocity_error: f32,
    /// Whether the motor is at its force limit.
    pub at_limit: bool,
}

// ---------------------------------------------------------------------------
// Joint motor
// ---------------------------------------------------------------------------

/// Unique identifier for a joint motor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MotorId(pub u32);

/// A motor attached to a joint axis.
#[derive(Debug, Clone)]
pub struct JointMotor {
    /// Unique ID.
    pub id: MotorId,
    /// Display name.
    pub name: String,
    /// Which axis this motor controls.
    pub axis: MotorAxis,
    /// Motor control mode.
    pub mode: MotorMode,
    /// Target velocity (for velocity mode).
    pub target_velocity: f32,
    /// Target position (for position mode).
    pub target_position: f32,
    /// Force/torque limits.
    pub limits: MotorLimits,
    /// Spring-damper parameters (for spring mode).
    pub spring: SpringDamperParams,
    /// PD controller (for position mode).
    pub pd: PdController,
    /// Force feedback from the last simulation step.
    pub feedback: MotorFeedback,
    /// Whether the motor is enabled.
    pub enabled: bool,
    /// Ramp-up time (seconds to reach full force from zero).
    pub ramp_time: f32,
    /// Current ramp factor (0..1).
    ramp_factor: f32,
}

impl JointMotor {
    /// Create a new motor.
    pub fn new(id: MotorId, name: &str, axis: MotorAxis, mode: MotorMode) -> Self {
        Self {
            id,
            name: name.to_string(),
            axis,
            mode,
            target_velocity: 0.0,
            target_position: 0.0,
            limits: MotorLimits::default(),
            spring: SpringDamperParams::default(),
            pd: PdController::default(),
            feedback: MotorFeedback::default(),
            enabled: true,
            ramp_time: 0.0,
            ramp_factor: 1.0,
        }
    }

    /// Create a velocity motor.
    pub fn velocity(id: MotorId, name: &str, axis: MotorAxis, target_vel: f32) -> Self {
        let mut motor = Self::new(id, name, axis, MotorMode::Velocity);
        motor.target_velocity = target_vel;
        motor
    }

    /// Create a position (servo) motor.
    pub fn position(id: MotorId, name: &str, axis: MotorAxis, target_pos: f32) -> Self {
        let mut motor = Self::new(id, name, axis, MotorMode::Position);
        motor.target_position = target_pos;
        motor.pd.target = target_pos;
        motor
    }

    /// Create a spring motor.
    pub fn spring(id: MotorId, name: &str, axis: MotorAxis, stiffness: f32, damping: f32) -> Self {
        let mut motor = Self::new(id, name, axis, MotorMode::Spring);
        motor.spring.stiffness = stiffness;
        motor.spring.damping = damping;
        motor
    }

    /// Compute the motor force for the current frame.
    pub fn compute_force(&mut self, current_pos: f32, current_vel: f32, dt: f32) -> f32 {
        if !self.enabled || self.mode == MotorMode::Disabled || self.mode == MotorMode::Free {
            self.feedback = MotorFeedback::default();
            return 0.0;
        }

        // Update ramp.
        if self.ramp_time > EPSILON {
            self.ramp_factor = (self.ramp_factor + dt / self.ramp_time).min(1.0);
        } else {
            self.ramp_factor = 1.0;
        }

        let raw_force = match self.mode {
            MotorMode::Velocity => {
                let vel_error = self.target_velocity - current_vel;
                vel_error * 50.0 // Simple proportional velocity control.
            }
            MotorMode::Position => {
                self.pd.target = self.limits.clamp_position(self.target_position);
                self.pd.compute(current_pos, current_vel, dt)
            }
            MotorMode::Spring => {
                self.spring.compute_force(current_pos, current_vel)
            }
            _ => 0.0,
        };

        let ramped = raw_force * self.ramp_factor;
        let clamped = self.limits.clamp_force(ramped, self.axis.is_angular());
        let at_limit = (clamped - ramped).abs() > EPSILON;

        self.feedback = MotorFeedback {
            applied_force: ramped,
            clamped_force: clamped,
            current_position: current_pos,
            current_velocity: current_vel,
            position_error: self.target_position - current_pos,
            velocity_error: self.target_velocity - current_vel,
            at_limit,
        };

        clamped
    }

    /// Enable/disable the motor.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.ramp_factor = 0.0;
            self.pd.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// Motor profiles
// ---------------------------------------------------------------------------

/// Pre-built motor configurations for common use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotorProfile {
    /// Door hinge: slow angular, position-controlled, with limits.
    DoorHinge,
    /// Wheel: velocity-controlled angular, high torque.
    Wheel,
    /// Steering: position-controlled angular, fast response.
    Steering,
    /// Elevator: position-controlled linear Y, smooth.
    Elevator,
    /// Crane: position-controlled linear XZ, spring-damped.
    Crane,
    /// Robot arm: position-controlled angular, precise.
    RobotArm,
}

impl MotorProfile {
    /// Apply the profile to a motor.
    pub fn apply(&self, motor: &mut JointMotor) {
        match self {
            Self::DoorHinge => {
                motor.mode = MotorMode::Position;
                motor.limits = MotorLimits::angular(0.0, 90.0, 200.0);
                motor.pd = PdController::new(50.0, 15.0);
            }
            Self::Wheel => {
                motor.mode = MotorMode::Velocity;
                motor.limits.max_torque = 1000.0;
                motor.target_velocity = 0.0;
            }
            Self::Steering => {
                motor.mode = MotorMode::Position;
                motor.limits = MotorLimits::angular(-35.0, 35.0, 500.0);
                motor.pd = PdController::new(200.0, 30.0);
            }
            Self::Elevator => {
                motor.mode = MotorMode::Position;
                motor.limits = MotorLimits::linear(0.0, 50.0, 5000.0);
                motor.pd = PdController::new(300.0, 100.0);
                motor.ramp_time = 0.5;
            }
            Self::Crane => {
                motor.mode = MotorMode::Spring;
                motor.spring = SpringDamperParams {
                    stiffness: 500.0,
                    damping: 100.0,
                    rest_position: 0.0,
                };
            }
            Self::RobotArm => {
                motor.mode = MotorMode::Position;
                motor.limits = MotorLimits::angular(-180.0, 180.0, 800.0);
                motor.pd = PdController::new(500.0, 50.0);
            }
        }
    }

    /// Create a motor with this profile.
    pub fn create(&self, id: MotorId, name: &str, axis: MotorAxis) -> JointMotor {
        let mut motor = JointMotor::new(id, name, axis, MotorMode::Disabled);
        self.apply(&mut motor);
        motor
    }
}

// ---------------------------------------------------------------------------
// Motor manager
// ---------------------------------------------------------------------------

/// Statistics for the motor system.
#[derive(Debug, Clone, Default)]
pub struct MotorSystemStats {
    /// Total number of motors.
    pub total_motors: usize,
    /// Number of enabled motors.
    pub enabled_motors: usize,
    /// Number of motors at their force limit.
    pub motors_at_limit: usize,
    /// Total force magnitude applied across all motors.
    pub total_applied_force: f32,
    /// Maximum position error across all motors.
    pub max_position_error: f32,
}

/// Manages all joint motors.
pub struct MotorManager {
    /// All motors indexed by ID.
    motors: HashMap<MotorId, JointMotor>,
    /// Next motor ID.
    next_id: u32,
    /// Statistics.
    stats: MotorSystemStats,
}

impl MotorManager {
    /// Create a new motor manager.
    pub fn new() -> Self {
        Self {
            motors: HashMap::new(),
            next_id: 0,
            stats: MotorSystemStats::default(),
        }
    }

    /// Create and register a new motor.
    pub fn create_motor(&mut self, name: &str, axis: MotorAxis, mode: MotorMode) -> MotorId {
        let id = MotorId(self.next_id);
        self.next_id += 1;
        let motor = JointMotor::new(id, name, axis, mode);
        self.motors.insert(id, motor);
        id
    }

    /// Create a motor from a profile.
    pub fn create_from_profile(
        &mut self,
        name: &str,
        axis: MotorAxis,
        profile: MotorProfile,
    ) -> MotorId {
        let id = MotorId(self.next_id);
        self.next_id += 1;
        let motor = profile.create(id, name, axis);
        self.motors.insert(id, motor);
        id
    }

    /// Remove a motor.
    pub fn remove_motor(&mut self, id: MotorId) -> bool {
        self.motors.remove(&id).is_some()
    }

    /// Get a motor.
    pub fn motor(&self, id: MotorId) -> Option<&JointMotor> {
        self.motors.get(&id)
    }

    /// Get a mutable motor.
    pub fn motor_mut(&mut self, id: MotorId) -> Option<&mut JointMotor> {
        self.motors.get_mut(&id)
    }

    /// Compute forces for all motors.
    pub fn compute_all(&mut self, dt: f32, states: &HashMap<MotorId, (f32, f32)>) -> HashMap<MotorId, f32> {
        let mut forces = HashMap::new();
        self.stats = MotorSystemStats::default();
        self.stats.total_motors = self.motors.len();

        for (id, motor) in &mut self.motors {
            if !motor.enabled {
                continue;
            }
            self.stats.enabled_motors += 1;

            let (pos, vel) = states.get(id).copied().unwrap_or((0.0, 0.0));
            let force = motor.compute_force(pos, vel, dt);
            forces.insert(*id, force);

            self.stats.total_applied_force += force.abs();
            if motor.feedback.at_limit {
                self.stats.motors_at_limit += 1;
            }
            let err = motor.feedback.position_error.abs();
            if err > self.stats.max_position_error {
                self.stats.max_position_error = err;
            }
        }

        forces
    }

    /// Get statistics.
    pub fn stats(&self) -> &MotorSystemStats {
        &self.stats
    }

    /// Get all motor IDs.
    pub fn motor_ids(&self) -> Vec<MotorId> {
        self.motors.keys().copied().collect()
    }
}

impl Default for MotorManager {
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
    fn test_velocity_motor() {
        let mut motor = JointMotor::velocity(MotorId(0), "test", MotorAxis::AngularX, 10.0);
        let force = motor.compute_force(0.0, 0.0, 0.016);
        // Should produce positive force to reach target velocity.
        assert!(force > 0.0);
    }

    #[test]
    fn test_position_motor() {
        let mut motor = JointMotor::position(MotorId(0), "test", MotorAxis::AngularY, 1.0);
        let force = motor.compute_force(0.0, 0.0, 0.016);
        // Should produce positive force to reach target position.
        assert!(force > 0.0);
    }

    #[test]
    fn test_spring_motor() {
        let mut motor = JointMotor::spring(MotorId(0), "test", MotorAxis::LinearY, 100.0, 10.0);
        // Displaced from rest: should produce restoring force.
        let force = motor.compute_force(1.0, 0.0, 0.016);
        assert!(force < 0.0); // Push back toward 0.
    }

    #[test]
    fn test_force_limits() {
        let limits = MotorLimits {
            max_force: 10.0,
            max_torque: 5.0,
            ..Default::default()
        };
        assert!((limits.clamp_force(100.0, false) - 10.0).abs() < EPSILON);
        assert!((limits.clamp_force(100.0, true) - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_spring_damper() {
        let spring = SpringDamperParams {
            stiffness: 100.0,
            damping: 10.0,
            rest_position: 0.0,
        };
        let force = spring.compute_force(1.0, 0.0);
        assert!((force - (-100.0)).abs() < EPSILON);
    }

    #[test]
    fn test_motor_profile() {
        let mut motor = JointMotor::new(MotorId(0), "door", MotorAxis::AngularZ, MotorMode::Disabled);
        MotorProfile::DoorHinge.apply(&mut motor);
        assert_eq!(motor.mode, MotorMode::Position);
        assert!(motor.limits.limits_enabled);
    }

    #[test]
    fn test_disabled_motor() {
        let mut motor = JointMotor::new(MotorId(0), "test", MotorAxis::LinearX, MotorMode::Disabled);
        let force = motor.compute_force(0.0, 0.0, 0.016);
        assert!(force.abs() < EPSILON);
    }
}
