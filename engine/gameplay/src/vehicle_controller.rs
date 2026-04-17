//! High-level vehicle gameplay controller wrapping the physics vehicle system.
//!
//! This module builds on the physics-level vehicle simulation (`engine/physics/vehicle`)
//! and adds gameplay features:
//!
//! - **Speed display**: converts internal velocity to km/h or mph for HUD
//! - **RPM and gear display**: exposes engine state for instrument panels
//! - **Nitro boost**: temporary speed multiplier with resource management
//! - **Drift detection**: monitors tire slip angle to detect and score drifts
//! - **Vehicle camera**: chase camera with speed-based distance and look-ahead
//! - **Vehicle audio**: engine pitch from RPM, tire screech from slip
//!
//! The controller acts as a bridge between raw player input and the physics
//! vehicle controller, applying gameplay transformations like nitro boost,
//! automatic transmission, and HUD data extraction.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Meters per second to km/h conversion factor.
const MPS_TO_KMH: f32 = 3.6;

/// Meters per second to mph conversion factor.
const MPS_TO_MPH: f32 = 2.23694;

/// Minimum slip angle (radians) to be considered drifting.
const DRIFT_THRESHOLD: f32 = 0.15;

/// Minimum speed (m/s) to be considered drifting.
const DRIFT_MIN_SPEED: f32 = 5.0;

/// Slip angle at which drift is fully established (radians).
const DRIFT_FULL_ANGLE: f32 = 0.5;

/// Small value for avoiding division by zero.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// Vehicle input
// ---------------------------------------------------------------------------

/// Raw player input for vehicle control.
#[derive(Debug, Clone, Default)]
pub struct VehicleInput {
    /// Throttle input (-1.0 to 1.0). Positive = forward, negative = reverse.
    pub throttle: f32,
    /// Brake input (0.0 to 1.0).
    pub brake: f32,
    /// Steering input (-1.0 to 1.0). Negative = left, positive = right.
    pub steering: f32,
    /// Whether the handbrake is engaged.
    pub handbrake: bool,
    /// Shift gear up request.
    pub gear_up: bool,
    /// Shift gear down request.
    pub gear_down: bool,
    /// Whether nitro boost is requested.
    pub nitro: bool,
    /// Whether to look behind (rear view).
    pub look_behind: bool,
    /// Horn button.
    pub horn: bool,
    /// Headlights toggle.
    pub headlights: bool,
}

impl VehicleInput {
    /// Clamp all input values to valid ranges.
    pub fn clamp(&mut self) {
        self.throttle = self.throttle.clamp(-1.0, 1.0);
        self.brake = self.brake.clamp(0.0, 1.0);
        self.steering = self.steering.clamp(-1.0, 1.0);
    }

    /// Whether the player is providing any throttle input.
    pub fn has_throttle(&self) -> bool {
        self.throttle.abs() > 0.05
    }

    /// Whether the player is providing any steering input.
    pub fn has_steering(&self) -> bool {
        self.steering.abs() > 0.05
    }
}

// ---------------------------------------------------------------------------
// Speed unit
// ---------------------------------------------------------------------------

/// Unit for speed display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpeedUnit {
    /// Kilometers per hour.
    Kmh,
    /// Miles per hour.
    Mph,
    /// Meters per second (raw).
    Mps,
}

impl SpeedUnit {
    /// Convert meters per second to this unit.
    pub fn from_mps(&self, mps: f32) -> f32 {
        match self {
            Self::Kmh => mps * MPS_TO_KMH,
            Self::Mph => mps * MPS_TO_MPH,
            Self::Mps => mps,
        }
    }

    /// Convert from this unit to meters per second.
    pub fn to_mps(&self, value: f32) -> f32 {
        match self {
            Self::Kmh => value / MPS_TO_KMH,
            Self::Mph => value / MPS_TO_MPH,
            Self::Mps => value,
        }
    }

    /// Unit suffix string.
    pub fn suffix(&self) -> &'static str {
        match self {
            Self::Kmh => "km/h",
            Self::Mph => "mph",
            Self::Mps => "m/s",
        }
    }
}

impl Default for SpeedUnit {
    fn default() -> Self {
        Self::Kmh
    }
}

// ---------------------------------------------------------------------------
// Drift state
// ---------------------------------------------------------------------------

/// Drift state machine tracking the stages of a drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftState {
    /// Not drifting.
    NotDrifting,
    /// Just started drifting (initiating the slide).
    Initiating,
    /// Fully drifting (sustained slide).
    Drifting,
    /// Coming out of a drift (recovering grip).
    Recovering,
}

impl DriftState {
    /// Whether the vehicle is in any form of drift.
    pub fn is_drifting(&self) -> bool {
        matches!(self, Self::Initiating | Self::Drifting)
    }
}

impl Default for DriftState {
    fn default() -> Self {
        Self::NotDrifting
    }
}

// ---------------------------------------------------------------------------
// Drift tracker
// ---------------------------------------------------------------------------

/// Tracks drift metrics for scoring.
#[derive(Debug, Clone, Default)]
pub struct DriftTracker {
    /// Current drift state.
    pub state: DriftState,
    /// Duration of the current drift (seconds).
    pub duration: f32,
    /// Maximum slip angle during the current drift (radians).
    pub max_angle: f32,
    /// Current slip angle (radians).
    pub current_angle: f32,
    /// Total distance traveled while drifting (meters).
    pub distance: f32,
    /// Score accumulated during the current drift.
    pub score: f32,
    /// Total score from all drifts.
    pub total_score: f32,
    /// Number of completed drifts.
    pub drift_count: u32,
    /// Time since the last drift ended (for combo window).
    pub time_since_last_drift: f32,
    /// Combo multiplier (increases for consecutive drifts).
    pub combo_multiplier: f32,
    /// Maximum combo window (seconds between drifts to maintain combo).
    pub combo_window: f32,
}

impl DriftTracker {
    /// Create a new drift tracker.
    pub fn new() -> Self {
        Self {
            combo_window: 2.0,
            combo_multiplier: 1.0,
            ..Default::default()
        }
    }

    /// Update drift state based on current slip angle and speed.
    pub fn update(&mut self, slip_angle: f32, speed: f32, dt: f32) {
        self.current_angle = slip_angle.abs();
        let is_drifting = self.current_angle > DRIFT_THRESHOLD && speed > DRIFT_MIN_SPEED;

        match self.state {
            DriftState::NotDrifting => {
                if is_drifting {
                    self.state = DriftState::Initiating;
                    self.duration = 0.0;
                    self.max_angle = 0.0;
                    self.distance = 0.0;
                    self.score = 0.0;
                }

                // Combo timeout.
                self.time_since_last_drift += dt;
                if self.time_since_last_drift > self.combo_window {
                    self.combo_multiplier = 1.0;
                }
            }
            DriftState::Initiating => {
                if !is_drifting {
                    self.state = DriftState::NotDrifting;
                } else {
                    self.duration += dt;
                    if self.duration > 0.3 {
                        self.state = DriftState::Drifting;
                    }
                    self.max_angle = self.max_angle.max(self.current_angle);
                }
            }
            DriftState::Drifting => {
                if !is_drifting {
                    self.state = DriftState::Recovering;
                } else {
                    self.duration += dt;
                    self.distance += speed * dt;
                    self.max_angle = self.max_angle.max(self.current_angle);

                    // Score: based on angle, speed, and duration.
                    let angle_factor = (self.current_angle / DRIFT_FULL_ANGLE).min(1.0);
                    let speed_factor = speed / 20.0;
                    self.score += angle_factor * speed_factor * dt * 100.0 * self.combo_multiplier;
                }
            }
            DriftState::Recovering => {
                if is_drifting {
                    self.state = DriftState::Drifting;
                } else {
                    // Drift complete.
                    self.total_score += self.score;
                    self.drift_count += 1;
                    self.combo_multiplier += 0.5;
                    self.time_since_last_drift = 0.0;
                    self.state = DriftState::NotDrifting;

                    log::trace!(
                        "Drift complete: score={:.0}, duration={:.1}s, max_angle={:.1}deg, combo=x{:.1}",
                        self.score,
                        self.duration,
                        self.max_angle.to_degrees(),
                        self.combo_multiplier,
                    );
                }
            }
        }
    }

    /// Reset all drift tracking data.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ---------------------------------------------------------------------------
// Nitro boost
// ---------------------------------------------------------------------------

/// Nitro boost system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NitroBoost {
    /// Current nitro fuel (0.0 to max_fuel).
    pub fuel: f32,
    /// Maximum nitro fuel capacity.
    pub max_fuel: f32,
    /// Fuel consumption rate (units/sec while active).
    pub consumption_rate: f32,
    /// Fuel regeneration rate (units/sec while inactive).
    pub regen_rate: f32,
    /// Speed multiplier when nitro is active.
    pub speed_multiplier: f32,
    /// Acceleration multiplier when nitro is active.
    pub accel_multiplier: f32,
    /// Whether nitro is currently active.
    pub active: bool,
    /// Whether nitro is available (not on cooldown).
    pub available: bool,
    /// Cooldown after fuel is depleted (seconds).
    pub cooldown: f32,
    /// Remaining cooldown time.
    pub cooldown_remaining: f32,
    /// Minimum fuel required to activate.
    pub min_activate_fuel: f32,
}

impl Default for NitroBoost {
    fn default() -> Self {
        Self {
            fuel: 100.0,
            max_fuel: 100.0,
            consumption_rate: 25.0,
            regen_rate: 10.0,
            speed_multiplier: 1.5,
            accel_multiplier: 2.0,
            active: false,
            available: true,
            cooldown: 3.0,
            cooldown_remaining: 0.0,
            min_activate_fuel: 20.0,
        }
    }
}

impl NitroBoost {
    /// Update nitro state.
    pub fn update(&mut self, requested: bool, dt: f32) {
        // Cooldown.
        if self.cooldown_remaining > 0.0 {
            self.cooldown_remaining -= dt;
            if self.cooldown_remaining <= 0.0 {
                self.cooldown_remaining = 0.0;
                self.available = true;
            }
        }

        // Activate/deactivate.
        if requested && self.available && self.fuel >= self.min_activate_fuel {
            self.active = true;
        } else if !requested || self.fuel <= 0.0 {
            if self.active && self.fuel <= 0.0 {
                // Fuel depleted: start cooldown.
                self.cooldown_remaining = self.cooldown;
                self.available = false;
            }
            self.active = false;
        }

        // Consume or regenerate fuel.
        if self.active {
            self.fuel = (self.fuel - self.consumption_rate * dt).max(0.0);
        } else if self.fuel < self.max_fuel {
            self.fuel = (self.fuel + self.regen_rate * dt).min(self.max_fuel);
        }
    }

    /// Get the current fuel as a fraction (0.0 to 1.0).
    pub fn fuel_fraction(&self) -> f32 {
        self.fuel / self.max_fuel.max(EPSILON)
    }
}

// ---------------------------------------------------------------------------
// Vehicle HUD data
// ---------------------------------------------------------------------------

/// Data extracted from the vehicle state for HUD display.
#[derive(Debug, Clone, Default)]
pub struct VehicleHudData {
    /// Speed in the configured display unit.
    pub speed_display: f32,
    /// Speed unit suffix string.
    pub speed_unit: &'static str,
    /// Engine RPM (0 to redline).
    pub rpm: f32,
    /// Maximum RPM (redline).
    pub max_rpm: f32,
    /// Current gear (0 = neutral, -1 = reverse, 1+ = forward gears).
    pub gear: i32,
    /// Total number of forward gears.
    pub gear_count: u32,
    /// Nitro fuel fraction (0.0 to 1.0).
    pub nitro_fuel: f32,
    /// Whether nitro is active.
    pub nitro_active: bool,
    /// Current drift state.
    pub drift_state: DriftState,
    /// Current drift score.
    pub drift_score: f32,
    /// Drift combo multiplier.
    pub drift_combo: f32,
    /// Whether the handbrake is engaged.
    pub handbrake: bool,
    /// Whether headlights are on.
    pub headlights: bool,
}

// ---------------------------------------------------------------------------
// Vehicle controller
// ---------------------------------------------------------------------------

/// High-level vehicle gameplay controller.
///
/// Wraps the physics-level vehicle controller with gameplay features like
/// nitro boost, drift scoring, speed display, and automatic transmission.
#[derive(Debug, Clone)]
pub struct VehicleController {
    /// Current world-space position.
    pub position: Vec3,
    /// Current rotation.
    pub rotation: Quat,
    /// Current velocity (world space).
    pub velocity: Vec3,
    /// Forward direction.
    pub forward: Vec3,
    /// Speed in m/s.
    pub speed_mps: f32,
    /// Engine RPM.
    pub rpm: f32,
    /// Maximum RPM (redline).
    pub max_rpm: f32,
    /// Idle RPM.
    pub idle_rpm: f32,
    /// Current gear (-1 = reverse, 0 = neutral, 1+ = forward).
    pub gear: i32,
    /// Number of forward gears.
    pub gear_count: u32,
    /// Gear ratios (index 0 = first gear).
    pub gear_ratios: Vec<f32>,
    /// Reverse gear ratio.
    pub reverse_ratio: f32,
    /// Final drive ratio.
    pub final_drive: f32,
    /// Whether to use automatic transmission.
    pub automatic: bool,
    /// RPM at which to upshift (automatic).
    pub upshift_rpm: f32,
    /// RPM at which to downshift (automatic).
    pub downshift_rpm: f32,
    /// Nitro boost system.
    pub nitro: NitroBoost,
    /// Drift tracker.
    pub drift: DriftTracker,
    /// Speed display unit.
    pub speed_unit: SpeedUnit,
    /// Rear wheel slip angle (for drift detection).
    pub rear_slip_angle: f32,
    /// Front wheel slip angle.
    pub front_slip_angle: f32,
    /// Whether the handbrake is engaged.
    pub handbrake_engaged: bool,
    /// Whether headlights are on.
    pub headlights_on: bool,
    /// Effective throttle after nitro multiplier.
    pub effective_throttle: f32,
    /// Maximum speed (m/s).
    pub max_speed: f32,
    /// Maximum reverse speed (m/s).
    pub max_reverse_speed: f32,
    /// Steering sensitivity at speed (reduces at high speed).
    pub steering_speed_factor: f32,
    /// Wheel base (distance between front and rear axles, meters).
    pub wheel_base: f32,
    /// Track width (distance between left and right wheels, meters).
    pub track_width: f32,
}

impl VehicleController {
    /// Create a new vehicle controller with default settings.
    pub fn new() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            forward: Vec3::Z,
            speed_mps: 0.0,
            rpm: 800.0,
            max_rpm: 7000.0,
            idle_rpm: 800.0,
            gear: 1,
            gear_count: 6,
            gear_ratios: vec![3.5, 2.5, 1.8, 1.4, 1.1, 0.9],
            reverse_ratio: 3.2,
            final_drive: 3.7,
            automatic: true,
            upshift_rpm: 6000.0,
            downshift_rpm: 2500.0,
            nitro: NitroBoost::default(),
            drift: DriftTracker::new(),
            speed_unit: SpeedUnit::Kmh,
            rear_slip_angle: 0.0,
            front_slip_angle: 0.0,
            handbrake_engaged: false,
            headlights_on: false,
            effective_throttle: 0.0,
            max_speed: 80.0,
            max_reverse_speed: 20.0,
            steering_speed_factor: 1.0,
            wheel_base: 2.5,
            track_width: 1.5,
        }
    }

    /// Process input and update the controller state.
    pub fn update(&mut self, mut input: VehicleInput, dt: f32) {
        if dt <= 0.0 {
            return;
        }

        input.clamp();

        // Toggle headlights.
        if input.headlights {
            self.headlights_on = !self.headlights_on;
        }

        // Handbrake.
        self.handbrake_engaged = input.handbrake;

        // Nitro boost.
        self.nitro.update(input.nitro, dt);

        // Compute effective throttle.
        self.effective_throttle = input.throttle;
        if self.nitro.active {
            self.effective_throttle *= self.nitro.accel_multiplier;
        }

        // Automatic transmission.
        if self.automatic {
            self.auto_shift(dt);
        } else {
            if input.gear_up {
                self.shift_up();
            }
            if input.gear_down {
                self.shift_down();
            }
        }

        // Update engine RPM.
        self.update_rpm(input.throttle, dt);

        // Compute steering (speed-dependent).
        let speed_factor = 1.0 / (1.0 + self.speed_mps * 0.02);
        self.steering_speed_factor = speed_factor;
        let _effective_steering = input.steering * speed_factor;

        // Update speed.
        self.speed_mps = self.velocity.length();

        // Compute forward direction.
        self.forward = self.rotation * Vec3::Z;

        // Compute slip angles.
        self.compute_slip_angles();

        // Update drift tracker.
        self.drift.update(self.rear_slip_angle, self.speed_mps, dt);

        // Apply nitro speed cap boost.
        let _effective_max_speed = if self.nitro.active {
            self.max_speed * self.nitro.speed_multiplier
        } else {
            self.max_speed
        };
    }

    /// Get the display speed.
    pub fn display_speed(&self) -> f32 {
        self.speed_unit.from_mps(self.speed_mps)
    }

    /// Get the RPM as a fraction of max RPM (0.0 to 1.0).
    pub fn rpm_fraction(&self) -> f32 {
        self.rpm / self.max_rpm.max(EPSILON)
    }

    /// Get HUD data.
    pub fn hud_data(&self) -> VehicleHudData {
        VehicleHudData {
            speed_display: self.display_speed(),
            speed_unit: self.speed_unit.suffix(),
            rpm: self.rpm,
            max_rpm: self.max_rpm,
            gear: self.gear,
            gear_count: self.gear_count,
            nitro_fuel: self.nitro.fuel_fraction(),
            nitro_active: self.nitro.active,
            drift_state: self.drift.state,
            drift_score: self.drift.score,
            drift_combo: self.drift.combo_multiplier,
            handbrake: self.handbrake_engaged,
            headlights: self.headlights_on,
        }
    }

    /// Get the current gear ratio.
    pub fn current_gear_ratio(&self) -> f32 {
        if self.gear <= 0 {
            if self.gear == 0 {
                return 0.0;
            }
            return self.reverse_ratio;
        }
        let idx = (self.gear as usize).saturating_sub(1);
        self.gear_ratios.get(idx).copied().unwrap_or(1.0)
    }

    /// Shift up one gear.
    pub fn shift_up(&mut self) {
        if self.gear < self.gear_count as i32 {
            self.gear += 1;
            // Rev-match: drop RPM on upshift.
            if self.gear > 1 {
                let ratio_change = self.gear_ratios.get(self.gear as usize - 2)
                    .copied().unwrap_or(1.0)
                    / self.gear_ratios.get(self.gear as usize - 1)
                    .copied().unwrap_or(1.0);
                self.rpm /= ratio_change.max(EPSILON);
            }
            log::trace!("Shifted up to gear {}", self.gear);
        }
    }

    /// Shift down one gear.
    pub fn shift_down(&mut self) {
        if self.gear > -1 {
            self.gear -= 1;
            // Rev-match: raise RPM on downshift.
            if self.gear >= 1 {
                let ratio_change = self.gear_ratios.get(self.gear as usize)
                    .copied().unwrap_or(1.0)
                    / self.gear_ratios.get(self.gear as usize - 1)
                    .copied().unwrap_or(1.0);
                self.rpm /= ratio_change.max(EPSILON);
                self.rpm = self.rpm.clamp(self.idle_rpm, self.max_rpm);
            }
            log::trace!("Shifted down to gear {}", self.gear);
        }
    }

    /// Automatic transmission logic.
    fn auto_shift(&mut self, _dt: f32) {
        if self.gear >= 1 && self.rpm >= self.upshift_rpm && self.gear < self.gear_count as i32 {
            self.shift_up();
        } else if self.gear > 1 && self.rpm <= self.downshift_rpm {
            self.shift_down();
        }

        // Engage reverse when backing up from a stop.
        if self.gear == 1 && self.speed_mps < 1.0 && self.effective_throttle < -0.1 {
            self.gear = -1;
        } else if self.gear == -1 && self.effective_throttle > 0.1 {
            self.gear = 1;
        }
    }

    /// Update engine RPM based on speed and gear.
    fn update_rpm(&mut self, throttle: f32, dt: f32) {
        let gear_ratio = self.current_gear_ratio().abs().max(EPSILON);
        let wheel_rpm = self.speed_mps * 60.0 / (std::f32::consts::TAU * 0.33); // 0.33m wheel radius
        let engine_rpm_from_speed = wheel_rpm * gear_ratio * self.final_drive;

        // Blend between idle and speed-based RPM.
        let target_rpm = if throttle.abs() > 0.05 {
            engine_rpm_from_speed.max(self.idle_rpm + throttle.abs() * 2000.0)
        } else {
            engine_rpm_from_speed.max(self.idle_rpm)
        };

        // Smooth RPM transition.
        let rpm_rate = 5.0;
        let t = 1.0 - (-rpm_rate * dt).exp();
        self.rpm = self.rpm + (target_rpm - self.rpm) * t;
        self.rpm = self.rpm.clamp(self.idle_rpm, self.max_rpm);
    }

    /// Compute tire slip angles for drift detection.
    fn compute_slip_angles(&mut self) {
        if self.speed_mps < 1.0 {
            self.rear_slip_angle = 0.0;
            self.front_slip_angle = 0.0;
            return;
        }

        let velocity_dir = self.velocity.normalize_or_zero();
        let forward = self.forward;
        let right = self.rotation * Vec3::X;

        // Slip angle = angle between velocity direction and forward direction.
        let lateral_vel = velocity_dir.dot(right);
        let longitudinal_vel = velocity_dir.dot(forward).max(EPSILON);

        self.rear_slip_angle = lateral_vel.atan2(longitudinal_vel);
        self.front_slip_angle = self.rear_slip_angle * 0.7; // Simplified: front has less slip.
    }

    /// Set the vehicle's transform (called from physics integration).
    pub fn set_transform(&mut self, position: Vec3, rotation: Quat, velocity: Vec3) {
        self.position = position;
        self.rotation = rotation;
        self.velocity = velocity;
        self.speed_mps = velocity.length();
        self.forward = rotation * Vec3::Z;
    }

    /// Reset the vehicle to initial state.
    pub fn reset(&mut self) {
        self.velocity = Vec3::ZERO;
        self.speed_mps = 0.0;
        self.rpm = self.idle_rpm;
        self.gear = 1;
        self.nitro = NitroBoost::default();
        self.drift.reset();
        self.handbrake_engaged = false;
        self.rear_slip_angle = 0.0;
        self.front_slip_angle = 0.0;
    }
}

impl Default for VehicleController {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Vehicle camera
// ---------------------------------------------------------------------------

/// Chase camera for vehicles with dynamic distance based on speed.
#[derive(Debug, Clone)]
pub struct VehicleCamera {
    /// Current camera position (world space).
    pub position: Vec3,
    /// Current look-at target (world space).
    pub look_at: Vec3,
    /// Base distance behind the vehicle.
    pub base_distance: f32,
    /// Base height above the vehicle.
    pub base_height: f32,
    /// Additional distance at max speed.
    pub speed_distance_bonus: f32,
    /// Additional height at max speed.
    pub speed_height_bonus: f32,
    /// Maximum speed for camera distance scaling (m/s).
    pub reference_speed: f32,
    /// How much the camera looks ahead (toward velocity direction).
    /// 0.0 = always look at vehicle, 1.0 = fully look ahead.
    pub look_ahead_factor: f32,
    /// Camera smoothing factor (0 = instant, 1 = very slow).
    pub smooth_speed: f32,
    /// Camera shake intensity at max speed.
    pub speed_shake_intensity: f32,
    /// Current shake offset.
    pub shake_offset: Vec3,
    /// Shake timer (for pseudo-random shake).
    shake_timer: f32,
    /// FOV (field of view in degrees).
    pub fov: f32,
    /// Base FOV.
    pub base_fov: f32,
    /// FOV increase at max speed (degrees).
    pub speed_fov_bonus: f32,
}

impl VehicleCamera {
    /// Create a new vehicle camera with default settings.
    pub fn new() -> Self {
        Self {
            position: Vec3::new(0.0, 5.0, -10.0),
            look_at: Vec3::ZERO,
            base_distance: 8.0,
            base_height: 3.0,
            speed_distance_bonus: 4.0,
            speed_height_bonus: 1.5,
            reference_speed: 60.0,
            look_ahead_factor: 0.3,
            smooth_speed: 6.0,
            speed_shake_intensity: 0.05,
            shake_offset: Vec3::ZERO,
            shake_timer: 0.0,
            fov: 60.0,
            base_fov: 60.0,
            speed_fov_bonus: 15.0,
        }
    }

    /// Update the camera position and look-at target.
    pub fn update(
        &mut self,
        vehicle_pos: Vec3,
        vehicle_rot: Quat,
        vehicle_velocity: Vec3,
        speed_mps: f32,
        look_behind: bool,
        dt: f32,
    ) {
        let speed_fraction = (speed_mps / self.reference_speed).min(1.0);

        // Compute dynamic distance and height.
        let distance = self.base_distance + self.speed_distance_bonus * speed_fraction;
        let height = self.base_height + self.speed_height_bonus * speed_fraction;

        // Camera direction: behind the vehicle (or in front if looking behind).
        let forward = vehicle_rot * Vec3::Z;
        let camera_forward = if look_behind { forward } else { -forward };

        // Look-ahead: bias toward velocity direction.
        let velocity_dir = vehicle_velocity.normalize_or_zero();
        let look_dir = if velocity_dir.length_squared() > 0.1 {
            camera_forward
                .lerp(-velocity_dir * camera_forward.dot(-velocity_dir).signum(), self.look_ahead_factor)
                .normalize_or_zero()
        } else {
            camera_forward
        };

        // Target camera position.
        let target_pos =
            vehicle_pos + look_dir * distance + Vec3::Y * height;

        // Smooth interpolation.
        let t = 1.0 - (-self.smooth_speed * dt).exp();
        self.position = self.position.lerp(target_pos, t);

        // Look-at target.
        let look_ahead_offset = if speed_mps > 5.0 {
            velocity_dir * 5.0 * speed_fraction
        } else {
            Vec3::ZERO
        };
        self.look_at = vehicle_pos + Vec3::Y * 1.0 + look_ahead_offset;

        // Speed shake.
        self.shake_timer += dt * 20.0;
        if speed_fraction > 0.7 {
            let intensity = self.speed_shake_intensity * (speed_fraction - 0.7) / 0.3;
            self.shake_offset = Vec3::new(
                (self.shake_timer * 7.3).sin() * intensity,
                (self.shake_timer * 11.7).cos() * intensity,
                (self.shake_timer * 5.1).sin() * intensity * 0.5,
            );
            self.position += self.shake_offset;
        } else {
            self.shake_offset = Vec3::ZERO;
        }

        // Dynamic FOV.
        self.fov = self.base_fov + self.speed_fov_bonus * speed_fraction;
    }

    /// Get the camera's forward direction.
    pub fn camera_forward(&self) -> Vec3 {
        (self.look_at - self.position).normalize_or_zero()
    }

    /// Get the camera's right direction.
    pub fn camera_right(&self) -> Vec3 {
        self.camera_forward().cross(Vec3::Y).normalize_or_zero()
    }

    /// Get the camera's up direction.
    pub fn camera_up(&self) -> Vec3 {
        self.camera_right()
            .cross(self.camera_forward())
            .normalize_or_zero()
    }
}

impl Default for VehicleCamera {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Vehicle audio
// ---------------------------------------------------------------------------

/// Audio parameters derived from vehicle state for the audio system.
#[derive(Debug, Clone, Default)]
pub struct VehicleAudio {
    /// Engine pitch multiplier (1.0 = base pitch).
    pub engine_pitch: f32,
    /// Engine volume (0.0 to 1.0).
    pub engine_volume: f32,
    /// Tire screech volume (0.0 to 1.0, based on slip).
    pub tire_screech_volume: f32,
    /// Tire screech pitch.
    pub tire_screech_pitch: f32,
    /// Wind volume (based on speed).
    pub wind_volume: f32,
    /// Whether the horn is playing.
    pub horn_playing: bool,
    /// Whether to play a gear shift sound.
    pub gear_shift_trigger: bool,
    /// Whether to play the nitro activation sound.
    pub nitro_trigger: bool,
    /// Whether to play the nitro loop.
    pub nitro_loop: bool,
    /// Exhaust backfire trigger (on deceleration).
    pub backfire_trigger: bool,
}

impl VehicleAudio {
    /// Update audio parameters from vehicle state.
    pub fn update(
        &mut self,
        rpm: f32,
        max_rpm: f32,
        speed_mps: f32,
        slip_angle: f32,
        handbrake: bool,
        nitro_active: bool,
        horn: bool,
        gear_changed: bool,
        throttle: f32,
    ) {
        // Engine pitch: proportional to RPM.
        let rpm_fraction = rpm / max_rpm.max(EPSILON);
        self.engine_pitch = 0.5 + rpm_fraction * 1.5;
        self.engine_volume = 0.3 + rpm_fraction * 0.7;

        // Tire screech: based on slip angle and handbrake.
        let slip_factor = (slip_angle.abs() / DRIFT_FULL_ANGLE).min(1.0);
        let handbrake_screech = if handbrake && speed_mps > 3.0 { 0.5 } else { 0.0 };
        self.tire_screech_volume = (slip_factor * 0.8 + handbrake_screech).min(1.0);
        self.tire_screech_pitch = 0.8 + slip_factor * 0.4;

        // Wind volume: based on speed.
        self.wind_volume = (speed_mps / 40.0).min(1.0);

        // Horn.
        self.horn_playing = horn;

        // Gear shift.
        self.gear_shift_trigger = gear_changed;

        // Nitro.
        self.nitro_trigger = nitro_active && !self.nitro_loop;
        self.nitro_loop = nitro_active;

        // Backfire: sudden throttle release at high RPM.
        self.backfire_trigger = throttle.abs() < 0.1 && rpm_fraction > 0.7;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_unit_conversion() {
        let kmh = SpeedUnit::Kmh;
        let mph = SpeedUnit::Mph;

        let mps = 10.0;
        assert!((kmh.from_mps(mps) - 36.0).abs() < 0.1);
        assert!((mph.from_mps(mps) - 22.37).abs() < 0.1);

        // Round-trip.
        let value = kmh.from_mps(mps);
        assert!((kmh.to_mps(value) - mps).abs() < 0.01);
    }

    #[test]
    fn drift_tracker_enter_exit() {
        let mut tracker = DriftTracker::new();

        // Not drifting at low speed.
        tracker.update(0.3, 2.0, 1.0 / 60.0);
        assert_eq!(tracker.state, DriftState::NotDrifting);

        // Initiate drift.
        tracker.update(0.3, 20.0, 1.0 / 60.0);
        assert_eq!(tracker.state, DriftState::Initiating);

        // Sustain drift.
        for _ in 0..30 {
            tracker.update(0.3, 20.0, 1.0 / 60.0);
        }
        assert_eq!(tracker.state, DriftState::Drifting);

        // End drift.
        tracker.update(0.0, 20.0, 1.0 / 60.0);
        assert!(matches!(
            tracker.state,
            DriftState::Recovering | DriftState::NotDrifting
        ));
    }

    #[test]
    fn drift_score_accumulates() {
        let mut tracker = DriftTracker::new();

        // Initiate and sustain drift.
        for _ in 0..100 {
            tracker.update(0.4, 25.0, 1.0 / 60.0);
        }

        assert!(tracker.score > 0.0, "Score should accumulate during drift");
    }

    #[test]
    fn nitro_fuel_consumption() {
        let mut nitro = NitroBoost::default();
        let initial_fuel = nitro.fuel;

        nitro.update(true, 1.0);
        assert!(nitro.fuel < initial_fuel);
        assert!(nitro.active);
    }

    #[test]
    fn nitro_fuel_depletion_cooldown() {
        let mut nitro = NitroBoost::default();
        nitro.fuel = 1.0;

        // Deplete fuel.
        nitro.update(true, 1.0);
        assert!(!nitro.available);
        assert!(nitro.cooldown_remaining > 0.0);
    }

    #[test]
    fn nitro_fuel_regeneration() {
        let mut nitro = NitroBoost::default();
        nitro.fuel = 50.0;

        nitro.update(false, 1.0);
        assert!(nitro.fuel > 50.0);
    }

    #[test]
    fn vehicle_controller_gear_shift() {
        let mut vehicle = VehicleController::new();
        vehicle.automatic = false;

        assert_eq!(vehicle.gear, 1);
        vehicle.shift_up();
        assert_eq!(vehicle.gear, 2);
        vehicle.shift_down();
        assert_eq!(vehicle.gear, 1);
    }

    #[test]
    fn vehicle_controller_max_gear() {
        let mut vehicle = VehicleController::new();
        vehicle.automatic = false;

        for _ in 0..10 {
            vehicle.shift_up();
        }
        assert_eq!(vehicle.gear, vehicle.gear_count as i32);
    }

    #[test]
    fn vehicle_controller_reverse() {
        let mut vehicle = VehicleController::new();
        vehicle.automatic = false;

        vehicle.shift_down(); // To neutral.
        assert_eq!(vehicle.gear, 0);
        vehicle.shift_down(); // To reverse.
        assert_eq!(vehicle.gear, -1);
    }

    #[test]
    fn vehicle_controller_display_speed() {
        let mut vehicle = VehicleController::new();
        vehicle.speed_mps = 27.78; // ~100 km/h

        let display = vehicle.display_speed();
        assert!((display - 100.0).abs() < 1.0);
    }

    #[test]
    fn vehicle_controller_hud_data() {
        let vehicle = VehicleController::new();
        let hud = vehicle.hud_data();
        assert_eq!(hud.gear, 1);
        assert!(hud.rpm > 0.0);
    }

    #[test]
    fn vehicle_camera_update() {
        let mut camera = VehicleCamera::new();
        camera.update(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(0.0, 0.0, 20.0),
            20.0,
            false,
            1.0 / 60.0,
        );

        // Camera should be behind and above the vehicle.
        assert!(camera.position.y > 0.0);
    }

    #[test]
    fn vehicle_camera_speed_fov() {
        let mut camera = VehicleCamera::new();
        camera.update(Vec3::ZERO, Quat::IDENTITY, Vec3::Z * 60.0, 60.0, false, 1.0);
        assert!(camera.fov > camera.base_fov);
    }

    #[test]
    fn vehicle_audio_engine_pitch() {
        let mut audio = VehicleAudio::default();
        audio.update(3500.0, 7000.0, 20.0, 0.0, false, false, false, false, 0.5);
        assert!(audio.engine_pitch > 0.5);
        assert!(audio.engine_volume > 0.0);
    }

    #[test]
    fn vehicle_audio_tire_screech() {
        let mut audio = VehicleAudio::default();
        audio.update(3000.0, 7000.0, 20.0, 0.4, false, false, false, false, 0.5);
        assert!(audio.tire_screech_volume > 0.0);
    }

    #[test]
    fn vehicle_input_clamp() {
        let mut input = VehicleInput {
            throttle: 2.0,
            brake: -0.5,
            steering: 3.0,
            ..Default::default()
        };
        input.clamp();
        assert_eq!(input.throttle, 1.0);
        assert_eq!(input.brake, 0.0);
        assert_eq!(input.steering, 1.0);
    }

    #[test]
    fn vehicle_reset() {
        let mut vehicle = VehicleController::new();
        vehicle.speed_mps = 50.0;
        vehicle.gear = 4;
        vehicle.rpm = 6000.0;
        vehicle.reset();

        assert_eq!(vehicle.speed_mps, 0.0);
        assert_eq!(vehicle.gear, 1);
        assert_eq!(vehicle.rpm, vehicle.idle_rpm);
    }
}
