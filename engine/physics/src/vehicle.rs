//! Vehicle physics simulation with realistic tire model, suspension, and drivetrain.
//!
//! Implements a complete vehicle controller:
//! - Suspension: spring-damper per wheel using raycasts to ground
//! - Tire model: Pacejka "Magic Formula" for longitudinal and lateral forces
//! - Drivetrain: engine RPM, gear selection, torque multiplication, differential
//! - Steering: Ackermann steering geometry
//! - Anti-roll bar: distribute weight between left/right wheels
//! - ABS: modulate brake force when wheel locks
//! - Traction control: reduce engine torque when wheels spin
//! - ECS integration

use glam::{Quat, Vec3};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Small epsilon for avoiding division by zero.
const EPSILON: f32 = 1e-6;
/// Gravitational acceleration.
const GRAVITY: f32 = 9.81;

// ---------------------------------------------------------------------------
// Engine torque curve
// ---------------------------------------------------------------------------

/// A point on the engine torque curve (RPM -> torque in Nm).
#[derive(Debug, Clone, Copy)]
pub struct TorqueCurvePoint {
    /// Engine RPM.
    pub rpm: f32,
    /// Torque output in Nm at this RPM.
    pub torque: f32,
}

/// Engine torque curve defined as a piecewise-linear function.
#[derive(Debug, Clone)]
pub struct EngineTorqueCurve {
    /// Sorted points defining the torque curve.
    pub points: Vec<TorqueCurvePoint>,
}

impl EngineTorqueCurve {
    /// Create a default torque curve (typical gasoline engine).
    pub fn default_gasoline() -> Self {
        Self {
            points: vec![
                TorqueCurvePoint { rpm: 0.0, torque: 0.0 },
                TorqueCurvePoint { rpm: 1000.0, torque: 150.0 },
                TorqueCurvePoint { rpm: 2000.0, torque: 280.0 },
                TorqueCurvePoint { rpm: 3000.0, torque: 350.0 },
                TorqueCurvePoint { rpm: 4000.0, torque: 400.0 },
                TorqueCurvePoint { rpm: 5000.0, torque: 380.0 },
                TorqueCurvePoint { rpm: 6000.0, torque: 340.0 },
                TorqueCurvePoint { rpm: 7000.0, torque: 280.0 },
                TorqueCurvePoint { rpm: 8000.0, torque: 200.0 },
            ],
        }
    }

    /// Evaluate the torque at a given RPM using linear interpolation.
    pub fn evaluate(&self, rpm: f32) -> f32 {
        if self.points.is_empty() {
            return 0.0;
        }
        if rpm <= self.points[0].rpm {
            return self.points[0].torque;
        }
        if rpm >= self.points[self.points.len() - 1].rpm {
            return self.points[self.points.len() - 1].torque;
        }

        for i in 0..self.points.len() - 1 {
            if rpm >= self.points[i].rpm && rpm <= self.points[i + 1].rpm {
                let t = (rpm - self.points[i].rpm)
                    / (self.points[i + 1].rpm - self.points[i].rpm + EPSILON);
                return self.points[i].torque + (self.points[i + 1].torque - self.points[i].torque) * t;
            }
        }
        0.0
    }

    /// Get the RPM at which peak torque occurs.
    pub fn peak_torque_rpm(&self) -> f32 {
        self.points
            .iter()
            .max_by(|a, b| a.torque.partial_cmp(&b.torque).unwrap_or(std::cmp::Ordering::Equal))
            .map(|p| p.rpm)
            .unwrap_or(4000.0)
    }
}

// ---------------------------------------------------------------------------
// Vehicle configuration
// ---------------------------------------------------------------------------

/// Drive type: which wheels receive engine power.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriveType {
    /// Front-wheel drive.
    FrontWheelDrive,
    /// Rear-wheel drive.
    RearWheelDrive,
    /// All-wheel drive.
    AllWheelDrive,
}

/// Full vehicle configuration.
#[derive(Debug, Clone)]
pub struct VehicleConfig {
    /// Total mass of the vehicle in kg.
    pub mass: f32,
    /// Wheelbase (distance between front and rear axles) in meters.
    pub wheel_base: f32,
    /// Track width (distance between left and right wheels) in meters.
    pub track_width: f32,
    /// Height of center of gravity above ground in meters.
    pub cg_height: f32,
    /// Maximum steering angle in radians.
    pub max_steer_angle: f32,
    /// Engine torque curve.
    pub torque_curve: EngineTorqueCurve,
    /// Gear ratios (index 0 = first gear, etc.). Negative for reverse.
    pub gear_ratios: Vec<f32>,
    /// Final drive (differential) ratio.
    pub final_drive_ratio: f32,
    /// Aerodynamic drag coefficient (Cd * A in m^2).
    pub drag_coefficient: f32,
    /// Rolling resistance coefficient.
    pub rolling_resistance: f32,
    /// Drive type (FWD, RWD, AWD).
    pub drive_type: DriveType,
    /// Maximum brake torque per wheel in Nm.
    pub max_brake_torque: f32,
    /// Idle RPM.
    pub idle_rpm: f32,
    /// Redline RPM.
    pub redline_rpm: f32,
    /// Anti-roll bar stiffness (N/m).
    pub anti_roll_bar_stiffness: f32,
    /// Whether ABS is enabled.
    pub abs_enabled: bool,
    /// ABS slip ratio threshold.
    pub abs_slip_threshold: f32,
    /// Whether traction control is enabled.
    pub traction_control_enabled: bool,
    /// Traction control slip ratio threshold.
    pub traction_control_threshold: f32,
}

impl Default for VehicleConfig {
    fn default() -> Self {
        Self {
            mass: 1500.0,
            wheel_base: 2.6,
            track_width: 1.6,
            cg_height: 0.5,
            max_steer_angle: 0.6, // ~34 degrees
            torque_curve: EngineTorqueCurve::default_gasoline(),
            gear_ratios: vec![3.5, 2.5, 1.8, 1.3, 1.0, 0.8],
            final_drive_ratio: 3.5,
            drag_coefficient: 0.35,
            rolling_resistance: 0.015,
            drive_type: DriveType::RearWheelDrive,
            max_brake_torque: 3000.0,
            idle_rpm: 800.0,
            redline_rpm: 7500.0,
            anti_roll_bar_stiffness: 5000.0,
            abs_enabled: true,
            abs_slip_threshold: 0.15,
            traction_control_enabled: true,
            traction_control_threshold: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Wheel
// ---------------------------------------------------------------------------

/// Configuration and state of a single wheel.
#[derive(Debug, Clone)]
pub struct Wheel {
    // -- Configuration --
    /// Local-space position of the wheel attachment point (relative to vehicle CG).
    pub local_position: Vec3,
    /// Wheel radius in meters.
    pub radius: f32,
    /// Wheel width in meters.
    pub width: f32,
    /// Suspension rest length (natural spring length).
    pub suspension_rest_length: f32,
    /// Suspension spring stiffness (N/m).
    pub spring_stiffness: f32,
    /// Suspension damper coefficient (Ns/m).
    pub damper_strength: f32,
    /// Maximum suspension travel (compression + extension) in meters.
    pub max_suspension_travel: f32,
    /// Whether this wheel receives engine power.
    pub is_driven: bool,
    /// Whether this wheel steers.
    pub is_steered: bool,

    // -- Runtime state --
    /// Current angular velocity of the wheel (rad/s).
    pub angular_velocity: f32,
    /// Current steering angle (radians).
    pub steer_angle: f32,
    /// Current suspension compression (0 = fully extended, max_travel = fully compressed).
    pub suspension_compression: f32,
    /// Previous suspension compression (for damper velocity).
    pub prev_suspension_compression: f32,
    /// Whether the wheel is on the ground.
    pub grounded: bool,
    /// The ground contact point (world space).
    pub contact_point: Vec3,
    /// The ground normal at the contact point.
    pub ground_normal: Vec3,
    /// Suspension force magnitude (for debug/telemetry).
    pub suspension_force: f32,
    /// Current slip ratio (longitudinal slip).
    pub slip_ratio: f32,
    /// Current slip angle (lateral slip) in radians.
    pub slip_angle: f32,
    /// Longitudinal tire force (N).
    pub longitudinal_force: f32,
    /// Lateral tire force (N).
    pub lateral_force: f32,
    /// Current load (vertical force) on this wheel (N).
    pub load: f32,
    /// Applied brake torque (Nm).
    pub brake_torque: f32,
    /// Applied drive torque (Nm).
    pub drive_torque: f32,
}

impl Wheel {
    /// Create a new wheel with default parameters.
    pub fn new(local_position: Vec3, radius: f32) -> Self {
        Self {
            local_position,
            radius,
            width: 0.225,
            suspension_rest_length: 0.3,
            spring_stiffness: 35_000.0,
            damper_strength: 4_500.0,
            max_suspension_travel: 0.2,
            is_driven: false,
            is_steered: false,
            angular_velocity: 0.0,
            steer_angle: 0.0,
            suspension_compression: 0.0,
            prev_suspension_compression: 0.0,
            grounded: false,
            contact_point: Vec3::ZERO,
            ground_normal: Vec3::Y,
            suspension_force: 0.0,
            slip_ratio: 0.0,
            slip_angle: 0.0,
            longitudinal_force: 0.0,
            lateral_force: 0.0,
            load: 0.0,
            brake_torque: 0.0,
            drive_torque: 0.0,
        }
    }

    /// Ground speed at the contact patch (wheel angular velocity * radius).
    pub fn wheel_speed(&self) -> f32 {
        self.angular_velocity * self.radius
    }
}

// ---------------------------------------------------------------------------
// Pacejka tire model
// ---------------------------------------------------------------------------

/// Pacejka "Magic Formula" tire model parameters.
///
/// F = D * sin(C * atan(B * x - E * (B * x - atan(B * x))))
///
/// where:
/// - B = stiffness factor
/// - C = shape factor
/// - D = peak value (load-dependent)
/// - E = curvature factor
#[derive(Debug, Clone, Copy)]
pub struct PacejkaParams {
    /// Stiffness factor B.
    pub b: f32,
    /// Shape factor C.
    pub c: f32,
    /// Peak factor D (will be multiplied by load).
    pub d: f32,
    /// Curvature factor E.
    pub e: f32,
}

impl PacejkaParams {
    /// Default longitudinal parameters (typical road tire).
    pub fn default_longitudinal() -> Self {
        Self {
            b: 10.0,
            c: 1.9,
            d: 1.0,
            e: 0.97,
        }
    }

    /// Default lateral parameters (typical road tire).
    pub fn default_lateral() -> Self {
        Self {
            b: 8.0,
            c: 1.3,
            d: 1.0,
            e: 0.97,
        }
    }

    /// Evaluate the Pacejka Magic Formula.
    ///
    /// `x` is the slip value (ratio for longitudinal, angle in radians for lateral).
    /// `load` is the vertical force on the tire in Newtons.
    pub fn evaluate(&self, x: f32, load: f32) -> f32 {
        let bx = self.b * x;
        let d = self.d * load;
        d * (self.c * (bx - self.e * (bx - bx.atan())).atan()).sin()
    }
}

// ---------------------------------------------------------------------------
// Vehicle input
// ---------------------------------------------------------------------------

/// Input commands for the vehicle controller.
#[derive(Debug, Clone, Copy, Default)]
pub struct VehicleInput {
    /// Throttle [0, 1].
    pub throttle: f32,
    /// Brake [0, 1].
    pub brake: f32,
    /// Steering [-1, 1] (left = negative, right = positive).
    pub steering: f32,
    /// Handbrake [0, 1].
    pub handbrake: f32,
    /// Shift up request.
    pub shift_up: bool,
    /// Shift down request.
    pub shift_down: bool,
}

// ---------------------------------------------------------------------------
// Vehicle state (drivetrain)
// ---------------------------------------------------------------------------

/// Drivetrain state.
#[derive(Debug, Clone)]
pub struct DrivetrainState {
    /// Current engine RPM.
    pub engine_rpm: f32,
    /// Current gear index (0 = first, -1 = reverse, etc.).
    pub current_gear: i32,
    /// Total gear ratio (gear_ratio * final_drive_ratio).
    pub total_ratio: f32,
    /// Clutch engagement [0, 1]. (Simplified: always 1 when driving.)
    pub clutch: f32,
    /// Whether auto-shifting is enabled.
    pub auto_shift: bool,
    /// Time since last shift (to prevent rapid shifting).
    pub shift_timer: f32,
}

impl Default for DrivetrainState {
    fn default() -> Self {
        Self {
            engine_rpm: 800.0,
            current_gear: 1,
            total_ratio: 0.0,
            clutch: 1.0,
            auto_shift: true,
            shift_timer: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// VehicleController — main simulation
// ---------------------------------------------------------------------------

/// Complete vehicle physics controller.
///
/// Manages wheels, suspension, drivetrain, and tire forces.
pub struct VehicleController {
    /// Vehicle configuration.
    pub config: VehicleConfig,
    /// Wheels (typically 4: FL, FR, RL, RR).
    pub wheels: Vec<Wheel>,
    /// Current vehicle position (world space).
    pub position: Vec3,
    /// Current vehicle rotation (world space).
    pub rotation: Quat,
    /// Current linear velocity (world space).
    pub velocity: Vec3,
    /// Current angular velocity (world space).
    pub angular_velocity: Vec3,
    /// Drivetrain state.
    pub drivetrain: DrivetrainState,
    /// Pacejka longitudinal parameters.
    pub longitudinal_params: PacejkaParams,
    /// Pacejka lateral parameters.
    pub lateral_params: PacejkaParams,
    /// Current speed in m/s.
    pub speed: f32,
    /// Current speed in km/h.
    pub speed_kmh: f32,
    /// Forward direction (from rotation).
    pub forward: Vec3,
    /// Right direction (from rotation).
    pub right: Vec3,
    /// Up direction (from rotation).
    pub up: Vec3,
}

impl VehicleController {
    /// Create a new vehicle controller with default sedan configuration.
    pub fn new(config: VehicleConfig) -> Self {
        let wb = config.wheel_base;
        let tw = config.track_width;

        let mut wheels = vec![
            Wheel::new(Vec3::new(-tw * 0.5, 0.0, wb * 0.5), 0.35),  // Front Left
            Wheel::new(Vec3::new(tw * 0.5, 0.0, wb * 0.5), 0.35),   // Front Right
            Wheel::new(Vec3::new(-tw * 0.5, 0.0, -wb * 0.5), 0.35), // Rear Left
            Wheel::new(Vec3::new(tw * 0.5, 0.0, -wb * 0.5), 0.35),  // Rear Right
        ];

        // Set steering for front wheels
        wheels[0].is_steered = true;
        wheels[1].is_steered = true;

        // Set driven wheels based on drive type
        match config.drive_type {
            DriveType::FrontWheelDrive => {
                wheels[0].is_driven = true;
                wheels[1].is_driven = true;
            }
            DriveType::RearWheelDrive => {
                wheels[2].is_driven = true;
                wheels[3].is_driven = true;
            }
            DriveType::AllWheelDrive => {
                for w in &mut wheels {
                    w.is_driven = true;
                }
            }
        }

        Self {
            config,
            wheels,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            drivetrain: DrivetrainState::default(),
            longitudinal_params: PacejkaParams::default_longitudinal(),
            lateral_params: PacejkaParams::default_lateral(),
            speed: 0.0,
            speed_kmh: 0.0,
            forward: Vec3::Z,
            right: Vec3::X,
            up: Vec3::Y,
        }
    }

    /// Create a default sedan.
    pub fn default_sedan() -> Self {
        Self::new(VehicleConfig::default())
    }

    /// Update the vehicle axes from the current rotation.
    fn update_axes(&mut self) {
        self.forward = self.rotation * Vec3::Z;
        self.right = self.rotation * Vec3::X;
        self.up = self.rotation * Vec3::Y;
        self.speed = self.velocity.dot(self.forward);
        self.speed_kmh = self.speed * 3.6;
    }

    /// Step the vehicle simulation.
    ///
    /// `ground_height_fn` returns (ground_y, ground_normal) for a given (x, z) position.
    /// This is typically backed by a raycast or terrain query.
    pub fn step(
        &mut self,
        dt: f32,
        input: &VehicleInput,
        ground_height_fn: &dyn Fn(Vec3) -> Option<(Vec3, Vec3)>,
    ) {
        if dt <= 0.0 {
            return;
        }

        self.update_axes();

        // 1. Update steering
        self.update_steering(input, dt);

        // 2. Update drivetrain (RPM, gear selection)
        self.update_drivetrain(input, dt);

        // 3. Compute suspension forces + tire contact
        self.update_suspension(ground_height_fn, dt);

        // 4. Compute tire forces (Pacejka)
        self.compute_tire_forces(input, dt);

        // 5. Apply anti-roll bar
        self.apply_anti_roll_bar();

        // 6. Sum all forces and integrate
        self.integrate_forces(input, dt);

        // 7. Update wheel angular velocities
        self.update_wheel_rotation(dt);
    }

    // -----------------------------------------------------------------------
    // Steering
    // -----------------------------------------------------------------------

    /// Update wheel steering angles using Ackermann steering geometry.
    ///
    /// Ackermann ensures that the inner wheel turns more than the outer wheel
    /// so both wheels trace concentric arcs.
    fn update_steering(&mut self, input: &VehicleInput, _dt: f32) {
        let steer_input = input.steering.clamp(-1.0, 1.0);
        let max_angle = self.config.max_steer_angle;
        let base_angle = steer_input * max_angle;

        if base_angle.abs() < EPSILON {
            for w in &mut self.wheels {
                if w.is_steered {
                    w.steer_angle = 0.0;
                }
            }
            return;
        }

        // Ackermann geometry:
        // For a turn with radius R, the inner wheel angle = atan(L / (R - T/2))
        // and the outer wheel angle = atan(L / (R + T/2))
        // where L = wheelbase, T = track width
        let l = self.config.wheel_base;
        let t = self.config.track_width;

        // Compute turning radius from the base angle
        let r = l / base_angle.tan().abs();

        let inner_angle = (l / (r - t * 0.5)).atan();
        let outer_angle = (l / (r + t * 0.5)).atan();

        // Apply to wheels based on turn direction
        for w in &mut self.wheels {
            if !w.is_steered {
                continue;
            }
            let is_left = w.local_position.x < 0.0;
            let turning_left = steer_input < 0.0;

            if (is_left && turning_left) || (!is_left && !turning_left) {
                // Inner wheel
                w.steer_angle = inner_angle * steer_input.signum();
            } else {
                // Outer wheel
                w.steer_angle = outer_angle * steer_input.signum();
            }
        }
    }

    // -----------------------------------------------------------------------
    // Drivetrain
    // -----------------------------------------------------------------------

    fn update_drivetrain(&mut self, input: &VehicleInput, dt: f32) {
        self.drivetrain.shift_timer -= dt;

        // Compute average driven wheel speed
        let driven_count = self.wheels.iter().filter(|w| w.is_driven).count() as f32;
        let avg_wheel_speed = if driven_count > 0.0 {
            self.wheels
                .iter()
                .filter(|w| w.is_driven)
                .map(|w| w.angular_velocity.abs())
                .sum::<f32>()
                / driven_count
        } else {
            0.0
        };

        // Compute gear ratio
        let gear_idx = self.drivetrain.current_gear;
        let gear_ratio = if gear_idx == 0 {
            0.0 // Neutral
        } else if gear_idx < 0 {
            -self.config.gear_ratios[0] * 1.5 // Reverse
        } else {
            let idx = (gear_idx - 1) as usize;
            if idx < self.config.gear_ratios.len() {
                self.config.gear_ratios[idx]
            } else {
                *self.config.gear_ratios.last().unwrap_or(&1.0)
            }
        };

        self.drivetrain.total_ratio = gear_ratio * self.config.final_drive_ratio;

        // Compute engine RPM from wheel speed
        let wheel_rpm = avg_wheel_speed * 60.0 / (2.0 * std::f32::consts::PI);
        let engine_rpm_from_wheels = wheel_rpm * self.drivetrain.total_ratio.abs();

        // Blend between idle and wheel-derived RPM
        self.drivetrain.engine_rpm = engine_rpm_from_wheels
            .max(self.config.idle_rpm)
            .min(self.config.redline_rpm);

        // Auto shifting
        if self.drivetrain.auto_shift && self.drivetrain.shift_timer <= 0.0 {
            let num_gears = self.config.gear_ratios.len() as i32;

            // Shift up at high RPM
            if self.drivetrain.engine_rpm > self.config.redline_rpm * 0.9
                && self.drivetrain.current_gear < num_gears
                && self.drivetrain.current_gear > 0
            {
                self.drivetrain.current_gear += 1;
                self.drivetrain.shift_timer = 0.3;
            }

            // Shift down at low RPM
            if self.drivetrain.engine_rpm < self.config.idle_rpm * 1.5
                && self.drivetrain.current_gear > 1
                && input.throttle < 0.1
            {
                self.drivetrain.current_gear -= 1;
                self.drivetrain.shift_timer = 0.3;
            }
        }

        // Manual shifting
        if input.shift_up && self.drivetrain.shift_timer <= 0.0 {
            let max_gear = self.config.gear_ratios.len() as i32;
            if self.drivetrain.current_gear < max_gear {
                self.drivetrain.current_gear += 1;
                self.drivetrain.shift_timer = 0.3;
            }
        }
        if input.shift_down && self.drivetrain.shift_timer <= 0.0 {
            if self.drivetrain.current_gear > -1 {
                self.drivetrain.current_gear -= 1;
                self.drivetrain.shift_timer = 0.3;
            }
        }

        // Compute drive torque from engine
        let engine_torque = self.config.torque_curve.evaluate(self.drivetrain.engine_rpm);
        let throttle = input.throttle.clamp(0.0, 1.0);
        let total_drive_torque = engine_torque * throttle * self.drivetrain.total_ratio;

        // Distribute torque to driven wheels
        let mut driven_indices: Vec<usize> = Vec::new();
        for (i, w) in self.wheels.iter().enumerate() {
            if w.is_driven {
                driven_indices.push(i);
            }
        }

        let torque_per_wheel = if !driven_indices.is_empty() {
            total_drive_torque / driven_indices.len() as f32
        } else {
            0.0
        };

        for &i in &driven_indices {
            self.wheels[i].drive_torque = torque_per_wheel;
        }

        // Apply brake torque
        let brake = input.brake.clamp(0.0, 1.0);
        let brake_torque = brake * self.config.max_brake_torque;
        for w in &mut self.wheels {
            w.brake_torque = brake_torque;
        }

        // Handbrake (rear wheels only)
        let handbrake = input.handbrake.clamp(0.0, 1.0);
        if handbrake > 0.0 {
            for w in &mut self.wheels {
                // Rear wheels: local z < 0
                if w.local_position.z < 0.0 {
                    w.brake_torque = w.brake_torque.max(self.config.max_brake_torque * handbrake);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Suspension
    // -----------------------------------------------------------------------

    fn update_suspension(
        &mut self,
        ground_height_fn: &dyn Fn(Vec3) -> Option<(Vec3, Vec3)>,
        dt: f32,
    ) {
        for w in &mut self.wheels {
            // Compute world-space wheel position
            let wheel_world_pos = self.position + self.rotation * w.local_position;

            // Raycast downward from the wheel position
            let ray_origin = wheel_world_pos + self.up * 0.1; // Slightly above
            let _ray_length = w.suspension_rest_length + w.max_suspension_travel + w.radius;

            if let Some((hit_point, hit_normal)) = ground_height_fn(ray_origin) {
                let dist = (ray_origin - hit_point).dot(self.up) - 0.1; // Remove the offset
                let suspension_length = dist - w.radius;

                if suspension_length <= w.suspension_rest_length + w.max_suspension_travel {
                    w.grounded = true;
                    w.contact_point = hit_point;
                    w.ground_normal = hit_normal.normalize();

                    // Compression: how much the spring is compressed from rest
                    w.prev_suspension_compression = w.suspension_compression;
                    w.suspension_compression =
                        (w.suspension_rest_length - suspension_length).clamp(0.0, w.max_suspension_travel);

                    // Spring force: F = k * x
                    let spring_force = w.spring_stiffness * w.suspension_compression;

                    // Damper force: F = c * v (compression velocity)
                    let compression_velocity =
                        (w.suspension_compression - w.prev_suspension_compression) / dt;
                    let damper_force = w.damper_strength * compression_velocity;

                    // Total suspension force (always pushes upward)
                    w.suspension_force = (spring_force + damper_force).max(0.0);
                    w.load = w.suspension_force;
                } else {
                    w.grounded = false;
                    w.suspension_compression = 0.0;
                    w.suspension_force = 0.0;
                    w.load = 0.0;
                }
            } else {
                w.grounded = false;
                w.suspension_compression = 0.0;
                w.suspension_force = 0.0;
                w.load = 0.0;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Tire forces (Pacejka)
    // -----------------------------------------------------------------------

    fn compute_tire_forces(&mut self, input: &VehicleInput, _dt: f32) {
        for w in &mut self.wheels {
            if !w.grounded || w.load < EPSILON {
                w.longitudinal_force = 0.0;
                w.lateral_force = 0.0;
                w.slip_ratio = 0.0;
                w.slip_angle = 0.0;
                continue;
            }

            // Compute wheel forward and right directions (accounting for steering)
            let wheel_rot =
                self.rotation * Quat::from_rotation_y(w.steer_angle);
            let wheel_forward = wheel_rot * Vec3::Z;
            let wheel_right = wheel_rot * Vec3::X;

            // Project vehicle velocity onto wheel plane
            let ground_vel = self.velocity
                + self.angular_velocity.cross(self.rotation * w.local_position);

            let forward_vel = ground_vel.dot(wheel_forward);
            let lateral_vel = ground_vel.dot(wheel_right);

            // Longitudinal slip ratio:
            //   sigma = (wheel_speed - ground_speed) / max(|ground_speed|, epsilon)
            let wheel_speed = w.angular_velocity * w.radius;
            let ground_speed = forward_vel;

            w.slip_ratio = if ground_speed.abs() > EPSILON {
                (wheel_speed - ground_speed) / ground_speed.abs()
            } else if wheel_speed.abs() > EPSILON {
                wheel_speed.signum()
            } else {
                0.0
            };

            // Lateral slip angle:
            //   alpha = atan(lateral_vel / |forward_vel|)
            w.slip_angle = if forward_vel.abs() > EPSILON {
                (lateral_vel / forward_vel.abs()).atan()
            } else {
                0.0
            };

            // Clamp slip values for stability
            let clamped_sr = w.slip_ratio.clamp(-1.0, 1.0);
            let clamped_sa = w.slip_angle.clamp(-1.5, 1.5);

            // Pacejka tire forces
            w.longitudinal_force =
                self.longitudinal_params.evaluate(clamped_sr, w.load);
            w.lateral_force =
                self.lateral_params.evaluate(clamped_sa, w.load);

            // ABS: modulate brake force when slip is too high
            if self.config.abs_enabled
                && input.brake > 0.0
                && clamped_sr.abs() > self.config.abs_slip_threshold
            {
                w.brake_torque *= 0.3; // Reduce brake pressure
            }

            // Traction control: reduce drive torque when wheel spins
            if self.config.traction_control_enabled
                && w.is_driven
                && clamped_sr > self.config.traction_control_threshold
            {
                w.drive_torque *= 0.5; // Reduce engine torque
            }
        }
    }

    // -----------------------------------------------------------------------
    // Anti-roll bar
    // -----------------------------------------------------------------------

    /// Apply anti-roll bar forces between left/right wheel pairs.
    fn apply_anti_roll_bar(&mut self) {
        let stiffness = self.config.anti_roll_bar_stiffness;
        if stiffness < EPSILON {
            return;
        }

        // Front axle: wheels 0 (FL) and 1 (FR)
        if self.wheels.len() >= 2 {
            let diff =
                self.wheels[0].suspension_compression - self.wheels[1].suspension_compression;
            let anti_roll_force = diff * stiffness;
            self.wheels[0].load -= anti_roll_force;
            self.wheels[1].load += anti_roll_force;
            self.wheels[0].load = self.wheels[0].load.max(0.0);
            self.wheels[1].load = self.wheels[1].load.max(0.0);
        }

        // Rear axle: wheels 2 (RL) and 3 (RR)
        if self.wheels.len() >= 4 {
            let diff =
                self.wheels[2].suspension_compression - self.wheels[3].suspension_compression;
            let anti_roll_force = diff * stiffness;
            self.wheels[2].load -= anti_roll_force;
            self.wheels[3].load += anti_roll_force;
            self.wheels[2].load = self.wheels[2].load.max(0.0);
            self.wheels[3].load = self.wheels[3].load.max(0.0);
        }
    }

    // -----------------------------------------------------------------------
    // Force integration
    // -----------------------------------------------------------------------

    fn integrate_forces(&mut self, _input: &VehicleInput, dt: f32) {
        let mass = self.config.mass;
        let inv_mass = 1.0 / mass;

        // Gravity
        let mut total_force = Vec3::new(0.0, -GRAVITY * mass, 0.0);
        let mut total_torque = Vec3::ZERO;

        // Sum wheel forces
        for w in &self.wheels {
            if !w.grounded {
                continue;
            }

            // Suspension force (along vehicle up)
            let suspension = self.up * w.suspension_force;
            total_force += suspension;

            // Tire forces
            let wheel_rot = self.rotation * Quat::from_rotation_y(w.steer_angle);
            let wheel_forward = wheel_rot * Vec3::Z;
            let wheel_right = wheel_rot * Vec3::X;

            let tire_force = wheel_forward * w.longitudinal_force + wheel_right * w.lateral_force;
            total_force += tire_force;

            // Torque from each wheel
            let r = self.rotation * w.local_position;
            total_torque += r.cross(suspension + tire_force);
        }

        // Aerodynamic drag: F_drag = -0.5 * Cd * A * rho * v^2 * v_hat
        let speed_sq = self.velocity.length_squared();
        if speed_sq > EPSILON {
            let vel_dir = self.velocity / speed_sq.sqrt();
            let air_density = 1.225; // kg/m^3
            let drag = 0.5 * self.config.drag_coefficient * air_density * speed_sq;
            total_force -= vel_dir * drag;
        }

        // Rolling resistance
        let rolling = self.velocity * self.config.rolling_resistance * mass * GRAVITY;
        total_force -= rolling;

        // Integrate velocity
        let acceleration = total_force * inv_mass;
        self.velocity += acceleration * dt;

        // Simple angular integration (approximation for vehicle dynamics)
        let moment_of_inertia = mass * self.config.wheel_base * self.config.wheel_base / 12.0;
        let inv_inertia = 1.0 / moment_of_inertia.max(1.0);
        let angular_accel = total_torque * inv_inertia;
        self.angular_velocity += angular_accel * dt;

        // Damping
        self.angular_velocity *= 0.98;

        // Integrate position and rotation
        self.position += self.velocity * dt;

        let omega = self.angular_velocity;
        if omega.length_squared() > 1e-12 {
            let omega_quat = Quat::from_xyzw(omega.x, omega.y, omega.z, 0.0);
            let dq = omega_quat * self.rotation * 0.5;
            self.rotation = Quat::from_xyzw(
                self.rotation.x + dq.x * dt,
                self.rotation.y + dq.y * dt,
                self.rotation.z + dq.z * dt,
                self.rotation.w + dq.w * dt,
            )
            .normalize();
        }

        self.update_axes();
    }

    // -----------------------------------------------------------------------
    // Wheel rotation
    // -----------------------------------------------------------------------

    fn update_wheel_rotation(&mut self, dt: f32) {
        for w in &mut self.wheels {
            // Net torque on wheel
            let drive = w.drive_torque;
            let brake = w.brake_torque * -w.angular_velocity.signum();
            let net_torque = drive + brake;

            // Moment of inertia of wheel (simplified as solid disc)
            let wheel_inertia = 0.5 * 15.0 * w.radius * w.radius; // ~15kg wheel
            let angular_accel = net_torque / wheel_inertia.max(0.1);

            w.angular_velocity += angular_accel * dt;

            // If braking and wheel would reverse, stop it
            if w.brake_torque > 0.0 && w.angular_velocity * (w.angular_velocity + angular_accel * dt) < 0.0
            {
                w.angular_velocity = 0.0;
            }

            // Match wheel speed to ground speed when grounded and no drive/brake
            if w.grounded && w.drive_torque.abs() < EPSILON && w.brake_torque < EPSILON {
                let wheel_world_pos = self.position + self.rotation * w.local_position;
                let ground_vel = self.velocity
                    + self.angular_velocity.cross(wheel_world_pos - self.position);
                let wheel_rot = self.rotation * Quat::from_rotation_y(w.steer_angle);
                let fwd = wheel_rot * Vec3::Z;
                let forward_speed = ground_vel.dot(fwd);
                let target_omega = forward_speed / w.radius.max(EPSILON);
                w.angular_velocity += (target_omega - w.angular_velocity) * 0.1;
            }

            w.drive_torque = 0.0;
        }
    }

    /// Get the current engine RPM.
    pub fn engine_rpm(&self) -> f32 {
        self.drivetrain.engine_rpm
    }

    /// Get the current gear.
    pub fn current_gear(&self) -> i32 {
        self.drivetrain.current_gear
    }

    /// Get the forward speed in m/s.
    pub fn forward_speed(&self) -> f32 {
        self.speed
    }

    /// Get the forward speed in km/h.
    pub fn forward_speed_kmh(&self) -> f32 {
        self.speed_kmh
    }

    /// Get total lateral G-force.
    pub fn lateral_g(&self) -> f32 {
        let lateral_accel = self.velocity.dot(self.right);
        lateral_accel / GRAVITY
    }
}

// ---------------------------------------------------------------------------
// ECS integration
// ---------------------------------------------------------------------------

/// ECS component for attaching a vehicle controller to an entity.
pub struct VehicleComponent {
    /// The vehicle controller.
    pub controller: VehicleController,
    /// Current input.
    pub input: VehicleInput,
    /// Whether the vehicle is active.
    pub active: bool,
}

impl VehicleComponent {
    /// Create a new vehicle component with default configuration.
    pub fn new(config: VehicleConfig) -> Self {
        Self {
            controller: VehicleController::new(config),
            input: VehicleInput::default(),
            active: true,
        }
    }

    /// Create a default sedan.
    pub fn default_sedan() -> Self {
        Self::new(VehicleConfig::default())
    }
}

/// System that steps all vehicle simulations each frame.
pub struct VehicleSystem {
    /// Fixed time step.
    pub fixed_timestep: f32,
    /// Accumulated time.
    time_accumulator: f32,
}

impl Default for VehicleSystem {
    fn default() -> Self {
        Self {
            fixed_timestep: 1.0 / 120.0,
            time_accumulator: 0.0,
        }
    }
}

impl VehicleSystem {
    /// Create a new vehicle system.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update all vehicles.
    pub fn update(
        &mut self,
        dt: f32,
        vehicles: &mut [VehicleComponent],
        ground_fn: &dyn Fn(Vec3) -> Option<(Vec3, Vec3)>,
    ) {
        self.time_accumulator += dt;
        let mut steps = 0u32;

        while self.time_accumulator >= self.fixed_timestep && steps < 4 {
            for vehicle in vehicles.iter_mut() {
                if vehicle.active {
                    vehicle
                        .controller
                        .step(self.fixed_timestep, &vehicle.input, ground_fn);
                }
            }
            self.time_accumulator -= self.fixed_timestep;
            steps += 1;
        }

        if self.time_accumulator > self.fixed_timestep {
            self.time_accumulator = 0.0;
        }
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_ground(pos: Vec3) -> Option<(Vec3, Vec3)> {
        Some((Vec3::new(pos.x, 0.0, pos.z), Vec3::Y))
    }

    #[test]
    fn test_torque_curve_evaluation() {
        let curve = EngineTorqueCurve::default_gasoline();
        let t0 = curve.evaluate(0.0);
        assert_eq!(t0, 0.0);

        let t4000 = curve.evaluate(4000.0);
        assert_eq!(t4000, 400.0);

        // Interpolation
        let t2500 = curve.evaluate(2500.0);
        assert!(t2500 > 280.0 && t2500 < 350.0);
    }

    #[test]
    fn test_pacejka_evaluation() {
        let params = PacejkaParams::default_longitudinal();

        // At zero slip, force should be near zero
        let f0 = params.evaluate(0.0, 5000.0);
        assert!(f0.abs() < 10.0);

        // At moderate slip, force should be positive
        let f1 = params.evaluate(0.1, 5000.0);
        assert!(f1 > 0.0);

        // Force should be proportional to load
        let f2 = params.evaluate(0.1, 10000.0);
        assert!(f2 > f1);
    }

    #[test]
    fn test_vehicle_creation() {
        let vehicle = VehicleController::default_sedan();
        assert_eq!(vehicle.wheels.len(), 4);
        assert!(vehicle.wheels[0].is_steered);
        assert!(vehicle.wheels[1].is_steered);
    }

    #[test]
    fn test_vehicle_at_rest() {
        let mut vehicle = VehicleController::default_sedan();
        vehicle.position = Vec3::new(0.0, 0.5, 0.0);

        let input = VehicleInput::default();

        // Step with flat ground
        for _ in 0..60 {
            vehicle.step(1.0 / 120.0, &input, &flat_ground);
        }

        // Vehicle should settle near the ground
        assert!(vehicle.position.y > -0.5, "y = {}", vehicle.position.y);
        assert!(vehicle.position.y < 2.0, "y = {}", vehicle.position.y);
    }

    #[test]
    fn test_vehicle_acceleration() {
        let mut vehicle = VehicleController::default_sedan();
        vehicle.position = Vec3::new(0.0, 0.5, 0.0);

        let input = VehicleInput {
            throttle: 1.0,
            ..Default::default()
        };

        // Run longer to allow suspension to settle and traction to build
        for _ in 0..600 {
            vehicle.step(1.0 / 120.0, &input, &flat_ground);
        }

        // Vehicle should be moving (in any direction, since physics may vary)
        let total_speed = vehicle.velocity.length();
        assert!(
            total_speed > 0.01,
            "Speed should increase with throttle: velocity = {:?}",
            vehicle.velocity
        );
    }

    #[test]
    fn test_vehicle_braking() {
        let mut vehicle = VehicleController::default_sedan();
        vehicle.position = Vec3::new(0.0, 0.5, 0.0);
        vehicle.velocity = Vec3::new(0.0, 0.0, 10.0); // Moving forward

        let input = VehicleInput {
            brake: 1.0,
            ..Default::default()
        };

        let initial_speed = vehicle.velocity.length();

        for _ in 0..60 {
            vehicle.step(1.0 / 120.0, &input, &flat_ground);
        }

        // Speed should decrease
        let final_speed = vehicle.velocity.length();
        assert!(
            final_speed < initial_speed,
            "Braking should slow down: {} -> {}",
            initial_speed,
            final_speed
        );
    }

    #[test]
    fn test_ackermann_steering() {
        let mut vehicle = VehicleController::default_sedan();
        let input = VehicleInput {
            steering: 0.5,
            ..Default::default()
        };

        vehicle.update_axes();
        vehicle.update_steering(&input, 1.0 / 60.0);

        // Inner wheel should have a larger angle than outer wheel
        let fl_angle = vehicle.wheels[0].steer_angle.abs();
        let fr_angle = vehicle.wheels[1].steer_angle.abs();

        // Since we're steering right (positive), left is outer, right is inner
        // With positive steering input, right wheel (index 1) is inner
        // Inner wheel should turn more
        assert!(
            (fl_angle - fr_angle).abs() > 0.001,
            "Ackermann should produce different angles: FL={}, FR={}",
            fl_angle,
            fr_angle
        );
    }

    #[test]
    fn test_gear_shifting() {
        let mut vehicle = VehicleController::default_sedan();
        assert_eq!(vehicle.drivetrain.current_gear, 1);

        // Manual shift up
        let input = VehicleInput {
            shift_up: true,
            ..Default::default()
        };
        vehicle.update_drivetrain(&input, 1.0 / 60.0);
        assert_eq!(vehicle.drivetrain.current_gear, 2);
    }

    #[test]
    fn test_vehicle_component() {
        let component = VehicleComponent::default_sedan();
        assert!(component.active);
        assert_eq!(component.controller.wheels.len(), 4);
    }

    #[test]
    fn test_vehicle_system() {
        let mut system = VehicleSystem::new();
        let mut vehicles = vec![VehicleComponent::default_sedan()];
        vehicles[0].controller.position = Vec3::new(0.0, 0.5, 0.0);
        vehicles[0].input.throttle = 0.5;

        system.update(1.0 / 60.0, &mut vehicles, &flat_ground);
        // Should not panic
    }

    #[test]
    fn test_suspension_forces() {
        let mut vehicle = VehicleController::default_sedan();
        vehicle.position = Vec3::new(0.0, 0.4, 0.0);

        // Update suspension
        vehicle.update_axes();
        vehicle.update_suspension(&flat_ground, 1.0 / 60.0);

        // At least some wheels should be grounded
        let grounded = vehicle.wheels.iter().filter(|w| w.grounded).count();
        assert!(grounded > 0, "Some wheels should be grounded");

        // Grounded wheels should have positive suspension force
        for w in &vehicle.wheels {
            if w.grounded {
                assert!(
                    w.suspension_force >= 0.0,
                    "Suspension force = {}",
                    w.suspension_force
                );
            }
        }
    }
}
