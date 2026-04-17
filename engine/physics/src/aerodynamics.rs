//! Aerodynamic simulation for lift, drag, and airborne physics.
//!
//! Provides:
//! - Lift/drag coefficient computation from angle of attack
//! - NACA 4-digit airfoil profile generation and lookup
//! - Angle of attack and stall behavior modeling
//! - Wind resistance on arbitrary surfaces (flat plates, curved)
//! - Parachute drag simulation with deployment states
//! - Glider physics with pitch/roll/yaw aerodynamic response
//! - Paper airplane tumble and flutter dynamics
//! - Reynolds number estimation for flow regime selection
//! - ECS integration via `AerodynamicsComponent` and `AerodynamicsSystem`

use glam::{Mat3, Quat, Vec3};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Air density at sea level in kg/m^3.
const AIR_DENSITY_SEA_LEVEL: f32 = 1.225;
/// Kinematic viscosity of air at 20 C in m^2/s.
const AIR_VISCOSITY: f32 = 1.5e-5;
/// Small epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-7;
/// Gravitational acceleration magnitude.
const GRAVITY: f32 = 9.81;
/// Pi constant.
const PI: f32 = std::f32::consts::PI;
/// Critical angle of attack in radians (approx 15 degrees) for typical stall.
const DEFAULT_STALL_ANGLE: f32 = 0.2618;
/// Post-stall lift coefficient reduction factor.
const POST_STALL_LIFT_FACTOR: f32 = 0.5;
/// Maximum Cl for a thin plate (theoretical).
const MAX_CL_FLAT_PLATE: f32 = 2.0 * PI;
/// Default Oswald efficiency factor for induced drag.
const DEFAULT_OSWALD: f32 = 0.85;
/// Default aspect ratio for wing surfaces.
const DEFAULT_ASPECT_RATIO: f32 = 6.0;

// ---------------------------------------------------------------------------
// NACA 4-digit airfoil
// ---------------------------------------------------------------------------

/// Parameters for a NACA 4-digit airfoil (e.g., NACA 2412).
#[derive(Debug, Clone, Copy)]
pub struct NacaAirfoil {
    /// Maximum camber as a fraction of chord (0-0.09, first digit / 100).
    pub max_camber: f32,
    /// Position of maximum camber as a fraction of chord (0-0.9, second digit / 10).
    pub camber_position: f32,
    /// Maximum thickness as a fraction of chord (0-0.40, last two digits / 100).
    pub max_thickness: f32,
}

impl NacaAirfoil {
    /// Create a NACA airfoil from its 4-digit designation (e.g., 2412).
    pub fn from_designation(digits: u16) -> Self {
        let d1 = (digits / 1000) as f32;
        let d2 = ((digits / 100) % 10) as f32;
        let d34 = (digits % 100) as f32;
        Self {
            max_camber: d1 / 100.0,
            camber_position: d2 / 10.0,
            max_thickness: d34 / 100.0,
        }
    }

    /// Create common airfoil presets.
    pub fn symmetric(thickness: f32) -> Self {
        Self {
            max_camber: 0.0,
            camber_position: 0.0,
            max_thickness: thickness,
        }
    }

    /// NACA 0012 (symmetric, common for tails).
    pub fn naca_0012() -> Self { Self::from_designation(12) }
    /// NACA 2412 (slight camber, general aviation).
    pub fn naca_2412() -> Self { Self::from_designation(2412) }
    /// NACA 4412 (moderate camber, high lift).
    pub fn naca_4412() -> Self { Self::from_designation(4412) }
    /// NACA 6412 (high camber).
    pub fn naca_6412() -> Self { Self::from_designation(6412) }

    /// Compute the half-thickness at a given chord position x in [0,1].
    pub fn half_thickness_at(&self, x: f32) -> f32 {
        let t = self.max_thickness;
        let x = x.clamp(0.0, 1.0);
        let sqrt_x = x.sqrt();
        // NACA thickness distribution
        t / 0.2
            * (0.2969 * sqrt_x
                - 0.1260 * x
                - 0.3516 * x * x
                + 0.2843 * x * x * x
                - 0.1015 * x * x * x * x)
    }

    /// Compute the camber line height at a given chord position x in [0,1].
    pub fn camber_at(&self, x: f32) -> f32 {
        let m = self.max_camber;
        let p = self.camber_position;
        let x = x.clamp(0.0, 1.0);

        if m < EPSILON || p < EPSILON {
            return 0.0;
        }

        if x < p {
            m / (p * p) * (2.0 * p * x - x * x)
        } else {
            m / ((1.0 - p) * (1.0 - p)) * (1.0 - 2.0 * p + 2.0 * p * x - x * x)
        }
    }

    /// Compute the camber line slope (dy/dx) at a chord position.
    pub fn camber_slope_at(&self, x: f32) -> f32 {
        let m = self.max_camber;
        let p = self.camber_position;
        let x = x.clamp(0.0, 1.0);

        if m < EPSILON || p < EPSILON {
            return 0.0;
        }

        if x < p {
            2.0 * m / (p * p) * (p - x)
        } else {
            2.0 * m / ((1.0 - p) * (1.0 - p)) * (p - x)
        }
    }

    /// Generate upper and lower surface points for rendering/analysis.
    /// Returns (upper_points, lower_points) as Vec<(f32, f32)> in chord coords.
    pub fn generate_profile(&self, num_points: usize) -> (Vec<(f32, f32)>, Vec<(f32, f32)>) {
        let mut upper = Vec::with_capacity(num_points);
        let mut lower = Vec::with_capacity(num_points);

        for i in 0..num_points {
            // Cosine spacing for better leading-edge resolution
            let beta = PI * i as f32 / (num_points - 1) as f32;
            let x = 0.5 * (1.0 - beta.cos());

            let yc = self.camber_at(x);
            let yt = self.half_thickness_at(x);
            let theta = self.camber_slope_at(x).atan();

            upper.push((x - yt * theta.sin(), yc + yt * theta.cos()));
            lower.push((x + yt * theta.sin(), yc - yt * theta.cos()));
        }

        (upper, lower)
    }

    /// Estimate the zero-lift angle of attack in radians.
    /// For cambered airfoils, this is approximately -2 * max_camber (thin airfoil theory).
    pub fn zero_lift_aoa(&self) -> f32 {
        -2.0 * self.max_camber
    }

    /// Estimate the 2D lift curve slope (dCl/d_alpha) in per-radian.
    /// Thin airfoil theory gives 2*pi, real airfoils are slightly less.
    pub fn lift_slope_2d(&self) -> f32 {
        2.0 * PI * 0.9 // 90% of theoretical for real airfoils
    }
}

// ---------------------------------------------------------------------------
// Aerodynamic coefficients
// ---------------------------------------------------------------------------

/// Computed aerodynamic coefficients at a given flight condition.
#[derive(Debug, Clone, Copy, Default)]
pub struct AeroCoefficients {
    /// Lift coefficient (perpendicular to relative airflow).
    pub cl: f32,
    /// Drag coefficient (parallel to relative airflow).
    pub cd: f32,
    /// Pitching moment coefficient.
    pub cm: f32,
    /// Angle of attack in radians.
    pub alpha: f32,
    /// Whether the wing is stalled.
    pub stalled: bool,
    /// Reynolds number.
    pub reynolds: f32,
}

/// Compute aerodynamic coefficients for a wing section.
#[derive(Debug, Clone)]
pub struct AeroModel {
    /// Airfoil profile.
    pub airfoil: NacaAirfoil,
    /// Stall angle in radians (positive).
    pub stall_angle: f32,
    /// Parasitic drag coefficient at zero lift.
    pub cd0: f32,
    /// Oswald efficiency factor for induced drag.
    pub oswald: f32,
    /// Wing aspect ratio (span^2 / area).
    pub aspect_ratio: f32,
    /// Chord length in meters.
    pub chord: f32,
    /// Span in meters.
    pub span: f32,
    /// Reference area (planform area) in m^2.
    pub area: f32,
}

impl AeroModel {
    /// Create a new aero model with a given airfoil and wing geometry.
    pub fn new(airfoil: NacaAirfoil, chord: f32, span: f32) -> Self {
        let area = chord * span;
        let aspect_ratio = if area > EPSILON { span * span / area } else { DEFAULT_ASPECT_RATIO };
        Self {
            airfoil,
            stall_angle: DEFAULT_STALL_ANGLE,
            cd0: 0.02,
            oswald: DEFAULT_OSWALD,
            aspect_ratio,
            chord,
            span,
            area,
        }
    }

    /// Create a flat plate aero model (no airfoil, just thin plate).
    pub fn flat_plate(chord: f32, span: f32) -> Self {
        Self::new(NacaAirfoil::symmetric(0.01), chord, span)
    }

    /// Compute 3D lift coefficient accounting for finite wing effects.
    /// Uses Prandtl's lifting-line correction.
    fn lift_coefficient_3d(&self, alpha: f32) -> (f32, bool) {
        let a0 = self.airfoil.lift_slope_2d();
        // 3D lift slope: a = a0 / (1 + a0 / (pi * AR))
        let a = a0 / (1.0 + a0 / (PI * self.aspect_ratio));

        let alpha_zl = self.airfoil.zero_lift_aoa();
        let effective_alpha = alpha - alpha_zl;

        let stalled = effective_alpha.abs() > self.stall_angle;

        let cl = if !stalled {
            a * effective_alpha
        } else {
            // Post-stall: Viterna-Corrigan model approximation
            let sign = effective_alpha.signum();
            let abs_alpha = effective_alpha.abs();
            let cl_max = a * self.stall_angle;

            // Smooth transition from linear to flat plate
            let stall_progress = ((abs_alpha - self.stall_angle) / (PI * 0.25 - self.stall_angle))
                .clamp(0.0, 1.0);
            let cl_flat = 2.0 * abs_alpha.sin() * abs_alpha.cos(); // flat plate
            let cl = cl_max * (1.0 - stall_progress) + cl_flat * stall_progress;
            cl * sign * POST_STALL_LIFT_FACTOR
        };

        (cl, stalled)
    }

    /// Compute drag coefficient (parasitic + induced + post-stall).
    fn drag_coefficient(&self, cl: f32, alpha: f32, stalled: bool) -> f32 {
        // Induced drag: Cd_i = Cl^2 / (pi * e * AR)
        let cd_induced = cl * cl / (PI * self.oswald * self.aspect_ratio);

        // Profile drag increases with angle of attack
        let alpha_factor = 1.0 + alpha.abs() * 0.5;
        let cd_profile = self.cd0 * alpha_factor;

        // Post-stall drag increase
        let cd_stall = if stalled {
            let excess = (alpha.abs() - self.stall_angle).max(0.0);
            2.0 * excess.sin() * excess.sin()
        } else {
            0.0
        };

        cd_profile + cd_induced + cd_stall
    }

    /// Compute pitching moment coefficient.
    fn moment_coefficient(&self, alpha: f32, cl: f32) -> f32 {
        // Quarter-chord moment from camber
        let cm_0 = -PI * self.airfoil.max_camber / 2.0;
        // Stability contribution
        cm_0 - 0.025 * alpha
    }

    /// Compute all aerodynamic coefficients at a given angle of attack and speed.
    pub fn compute_coefficients(&self, alpha: f32, speed: f32) -> AeroCoefficients {
        let reynolds = speed * self.chord / AIR_VISCOSITY;
        let (cl, stalled) = self.lift_coefficient_3d(alpha);
        let cd = self.drag_coefficient(cl, alpha, stalled);
        let cm = self.moment_coefficient(alpha, cl);

        AeroCoefficients {
            cl,
            cd,
            cm,
            alpha,
            stalled,
            reynolds,
        }
    }

    /// Compute the lift force magnitude.
    /// L = 0.5 * rho * V^2 * S * Cl
    pub fn lift_force(&self, dynamic_pressure: f32, cl: f32) -> f32 {
        dynamic_pressure * self.area * cl
    }

    /// Compute the drag force magnitude.
    /// D = 0.5 * rho * V^2 * S * Cd
    pub fn drag_force(&self, dynamic_pressure: f32, cd: f32) -> f32 {
        dynamic_pressure * self.area * cd
    }

    /// Compute the pitching moment.
    /// M = 0.5 * rho * V^2 * S * c * Cm
    pub fn pitching_moment(&self, dynamic_pressure: f32, cm: f32) -> f32 {
        dynamic_pressure * self.area * self.chord * cm
    }
}

// ---------------------------------------------------------------------------
// Wind resistance (surface drag)
// ---------------------------------------------------------------------------

/// Configuration for computing wind resistance on a surface.
#[derive(Debug, Clone)]
pub struct SurfaceDrag {
    /// Reference area exposed to the wind in m^2.
    pub area: f32,
    /// Drag coefficient (depends on shape).
    pub cd: f32,
    /// Normal vector of the surface (used to compute effective area).
    pub normal: Vec3,
}

impl SurfaceDrag {
    /// Create a flat plate drag surface.
    pub fn flat_plate(area: f32, normal: Vec3) -> Self {
        Self { area, cd: 1.28, normal: normal.normalize_or_zero() }
    }

    /// Create a streamlined body drag surface.
    pub fn streamlined(area: f32, normal: Vec3) -> Self {
        Self { area, cd: 0.04, normal: normal.normalize_or_zero() }
    }

    /// Create a sphere drag surface.
    pub fn sphere(radius: f32) -> Self {
        let area = PI * radius * radius;
        Self { area, cd: 0.47, normal: Vec3::Z }
    }

    /// Create a cylinder drag surface (perpendicular to axis).
    pub fn cylinder(radius: f32, length: f32) -> Self {
        let area = 2.0 * radius * length;
        Self { area, cd: 1.2, normal: Vec3::Z }
    }

    /// Compute the drag force vector given relative wind velocity and air density.
    pub fn compute_drag(&self, relative_wind: Vec3, air_density: f32) -> Vec3 {
        let speed = relative_wind.length();
        if speed < EPSILON {
            return Vec3::ZERO;
        }

        let wind_dir = relative_wind / speed;

        // Effective area based on angle between wind and surface normal
        let cos_angle = wind_dir.dot(self.normal).abs();
        let effective_area = self.area * cos_angle.max(0.1); // minimum 10% effectiveness

        let dynamic_pressure = 0.5 * air_density * speed * speed;
        let drag_magnitude = dynamic_pressure * effective_area * self.cd;

        // Drag opposes relative wind direction
        -wind_dir * drag_magnitude
    }
}

// ---------------------------------------------------------------------------
// Parachute
// ---------------------------------------------------------------------------

/// Deployment state of a parachute.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParachuteState {
    /// Packed, not deployed.
    Packed,
    /// Currently deploying (inflating).
    Deploying { progress: f32 },
    /// Fully deployed.
    FullyDeployed,
    /// Cut away / detached.
    CutAway,
}

/// Parachute drag simulation.
#[derive(Debug, Clone)]
pub struct Parachute {
    /// Current state.
    pub state: ParachuteState,
    /// Canopy area when fully deployed in m^2.
    pub canopy_area: f32,
    /// Drag coefficient when fully deployed (hemispherical: ~1.3-1.5).
    pub cd_deployed: f32,
    /// Drag coefficient when packed (very small).
    pub cd_packed: f32,
    /// Time to fully deploy in seconds.
    pub deploy_time: f32,
    /// Oscillation amplitude (radians) — canopy sway.
    pub oscillation_amplitude: f32,
    /// Oscillation frequency in Hz.
    pub oscillation_frequency: f32,
    /// Time accumulator for deployment.
    deploy_timer: f32,
    /// Time accumulator for oscillation.
    oscillation_timer: f32,
}

impl Parachute {
    /// Create a new packed parachute.
    pub fn new(canopy_area: f32) -> Self {
        Self {
            state: ParachuteState::Packed,
            canopy_area,
            cd_deployed: 1.4,
            cd_packed: 0.3,
            deploy_time: 2.0,
            oscillation_amplitude: 0.1,
            oscillation_frequency: 0.5,
            deploy_timer: 0.0,
            oscillation_timer: 0.0,
        }
    }

    /// Begin deploying the parachute.
    pub fn deploy(&mut self) {
        if self.state == ParachuteState::Packed {
            self.state = ParachuteState::Deploying { progress: 0.0 };
            self.deploy_timer = 0.0;
        }
    }

    /// Cut away the parachute.
    pub fn cut_away(&mut self) {
        self.state = ParachuteState::CutAway;
    }

    /// Update the parachute state.
    pub fn update(&mut self, dt: f32) {
        match &mut self.state {
            ParachuteState::Deploying { progress } => {
                self.deploy_timer += dt;
                *progress = (self.deploy_timer / self.deploy_time).clamp(0.0, 1.0);
                if *progress >= 1.0 {
                    self.state = ParachuteState::FullyDeployed;
                }
            }
            ParachuteState::FullyDeployed => {
                self.oscillation_timer += dt;
            }
            _ => {}
        }
    }

    /// Get the effective drag coefficient based on current state.
    pub fn effective_cd(&self) -> f32 {
        match self.state {
            ParachuteState::Packed => self.cd_packed,
            ParachuteState::Deploying { progress } => {
                // Smooth interpolation during deployment
                let t = progress * progress * (3.0 - 2.0 * progress); // smoothstep
                self.cd_packed + (self.cd_deployed - self.cd_packed) * t
            }
            ParachuteState::FullyDeployed => self.cd_deployed,
            ParachuteState::CutAway => 0.0,
        }
    }

    /// Get the effective canopy area based on deployment state.
    pub fn effective_area(&self) -> f32 {
        match self.state {
            ParachuteState::Packed => 0.01 * self.canopy_area,
            ParachuteState::Deploying { progress } => {
                let t = progress * progress;
                0.01 * self.canopy_area + self.canopy_area * 0.99 * t
            }
            ParachuteState::FullyDeployed => self.canopy_area,
            ParachuteState::CutAway => 0.0,
        }
    }

    /// Compute the drag force on the parachute.
    pub fn drag_force(&self, velocity: Vec3, air_density: f32) -> Vec3 {
        let speed = velocity.length();
        if speed < EPSILON {
            return Vec3::ZERO;
        }

        let cd = self.effective_cd();
        let area = self.effective_area();
        let dynamic_pressure = 0.5 * air_density * speed * speed;
        let drag = dynamic_pressure * area * cd;

        // Drag opposes velocity
        let drag_dir = -velocity / speed;

        // Add oscillation for fully deployed chute
        let oscillation_offset = if matches!(self.state, ParachuteState::FullyDeployed) {
            let phase = self.oscillation_timer * self.oscillation_frequency * 2.0 * PI;
            let lateral = Vec3::new(
                phase.sin() * self.oscillation_amplitude,
                0.0,
                (phase * 1.3).cos() * self.oscillation_amplitude,
            );
            lateral
        } else {
            Vec3::ZERO
        };

        drag_dir * drag + oscillation_offset * drag * 0.1
    }

    /// Estimate the terminal velocity for a given payload mass.
    pub fn terminal_velocity(&self, payload_mass: f32, air_density: f32) -> f32 {
        let cd = self.effective_cd();
        let area = self.effective_area();
        if cd * area < EPSILON {
            return f32::INFINITY;
        }
        ((2.0 * payload_mass * GRAVITY) / (air_density * cd * area)).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Glider physics
// ---------------------------------------------------------------------------

/// Glider flight model with pitch, roll, and yaw aerodynamic response.
#[derive(Debug, Clone)]
pub struct GliderModel {
    /// Main wing aero model.
    pub main_wing: AeroModel,
    /// Horizontal tail aero model.
    pub tail: AeroModel,
    /// Vertical fin aero model (for yaw stability).
    pub fin: AeroModel,
    /// Tail moment arm (distance from CG to tail aerodynamic center).
    pub tail_arm: f32,
    /// Fin moment arm.
    pub fin_arm: f32,
    /// Total mass in kg.
    pub mass: f32,
    /// Moment of inertia (diagonal approximation).
    pub inertia: Vec3,
    /// Current orientation quaternion.
    pub orientation: Quat,
    /// Current angular velocity (body frame).
    pub angular_velocity: Vec3,
    /// Current world-space velocity.
    pub velocity: Vec3,
    /// Current world-space position.
    pub position: Vec3,
    /// Aileron deflection in radians (roll control).
    pub aileron: f32,
    /// Elevator deflection in radians (pitch control).
    pub elevator: f32,
    /// Rudder deflection in radians (yaw control).
    pub rudder: f32,
    /// Glide ratio (L/D) cache.
    cached_glide_ratio: f32,
    /// Whether the main wing is stalled.
    pub is_stalled: bool,
}

impl GliderModel {
    /// Create a new glider with basic geometry.
    pub fn new(wing_span: f32, wing_chord: f32, mass: f32) -> Self {
        let main_wing = AeroModel::new(NacaAirfoil::naca_2412(), wing_chord, wing_span);
        let tail = AeroModel::new(
            NacaAirfoil::naca_0012(),
            wing_chord * 0.4,
            wing_span * 0.3,
        );
        let fin = AeroModel::new(
            NacaAirfoil::naca_0012(),
            wing_chord * 0.4,
            wing_span * 0.15,
        );

        let tail_arm = wing_chord * 3.0;
        let fin_arm = wing_chord * 3.0;

        // Approximate inertia for a glider
        let ixx = mass * wing_span * wing_span / 12.0; // roll
        let iyy = mass * (wing_span * wing_span + wing_chord * wing_chord) / 12.0; // pitch
        let izz = mass * wing_chord * wing_chord / 12.0; // yaw

        Self {
            main_wing,
            tail,
            fin,
            tail_arm,
            fin_arm,
            mass,
            inertia: Vec3::new(ixx, iyy, izz),
            orientation: Quat::IDENTITY,
            angular_velocity: Vec3::ZERO,
            velocity: Vec3::ZERO,
            position: Vec3::ZERO,
            aileron: 0.0,
            elevator: 0.0,
            rudder: 0.0,
            cached_glide_ratio: 0.0,
            is_stalled: false,
        }
    }

    /// Compute aerodynamic forces and moments, then integrate one step.
    pub fn step(&mut self, dt: f32, wind: Vec3, air_density: f32) {
        if dt <= 0.0 {
            return;
        }

        // Relative airflow in world space
        let relative_wind = self.velocity - wind;
        let airspeed = relative_wind.length();

        if airspeed < EPSILON {
            // Apply gravity only
            self.velocity += Vec3::new(0.0, -GRAVITY, 0.0) * dt;
            self.position += self.velocity * dt;
            return;
        }

        // Transform wind to body frame
        let inv_orient = self.orientation.inverse();
        let body_wind = inv_orient * relative_wind;

        // Angle of attack (pitch angle of airflow in body frame)
        let alpha = (-body_wind.y).atan2(body_wind.z.abs().max(EPSILON));

        // Sideslip angle
        let beta = body_wind.x.atan2(body_wind.z.abs().max(EPSILON));

        let dynamic_pressure = 0.5 * air_density * airspeed * airspeed;

        // --- Main wing ---
        let wing_coeffs = self.main_wing.compute_coefficients(alpha, airspeed);
        self.is_stalled = wing_coeffs.stalled;

        let lift = self.main_wing.lift_force(dynamic_pressure, wing_coeffs.cl);
        let drag = self.main_wing.drag_force(dynamic_pressure, wing_coeffs.cd);

        // Roll moment from aileron (differential lift)
        let roll_moment = dynamic_pressure * self.main_wing.area * self.main_wing.span
            * 0.5 * self.aileron * 0.3;

        // --- Tail ---
        let tail_alpha = alpha + self.elevator;
        let tail_coeffs = self.tail.compute_coefficients(tail_alpha, airspeed);
        let tail_lift = self.tail.lift_force(dynamic_pressure, tail_coeffs.cl);
        let pitch_moment = -tail_lift * self.tail_arm
            + self.main_wing.pitching_moment(dynamic_pressure, wing_coeffs.cm);

        // --- Fin ---
        let fin_alpha = beta + self.rudder;
        let fin_coeffs = self.fin.compute_coefficients(fin_alpha, airspeed);
        let fin_force = self.fin.lift_force(dynamic_pressure, fin_coeffs.cl);
        let yaw_moment = -fin_force * self.fin_arm;

        // Build total force in body frame
        // Lift is perpendicular to airflow (body Y-up), drag is opposite to airflow
        let wind_dir = body_wind / airspeed;
        let lift_dir = Vec3::new(0.0, 1.0, 0.0); // body up
        let drag_dir = wind_dir; // along airflow

        let body_force = lift_dir * lift - drag_dir * drag
            + Vec3::new(fin_force, tail_lift, 0.0);

        // Transform force to world frame
        let world_force = self.orientation * body_force;

        // Add gravity
        let gravity_force = Vec3::new(0.0, -GRAVITY * self.mass, 0.0);
        let total_force = world_force + gravity_force;

        // Angular acceleration (body frame)
        let torque = Vec3::new(roll_moment, yaw_moment, pitch_moment);
        let angular_accel = Vec3::new(
            if self.inertia.x > EPSILON { torque.x / self.inertia.x } else { 0.0 },
            if self.inertia.y > EPSILON { torque.y / self.inertia.y } else { 0.0 },
            if self.inertia.z > EPSILON { torque.z / self.inertia.z } else { 0.0 },
        );

        // Angular damping
        let angular_damping = 0.98_f32;
        self.angular_velocity = self.angular_velocity * angular_damping + angular_accel * dt;

        // Update orientation
        let angle = self.angular_velocity.length() * dt;
        if angle > EPSILON {
            let axis = self.angular_velocity.normalize_or_zero();
            let dq = Quat::from_axis_angle(axis, angle);
            self.orientation = (self.orientation * dq).normalize();
        }

        // Linear integration
        let acceleration = total_force / self.mass;
        self.velocity += acceleration * dt;
        self.position += self.velocity * dt;

        // Cache glide ratio
        if drag > EPSILON {
            self.cached_glide_ratio = lift / drag;
        }
    }

    /// Get the current glide ratio (L/D).
    pub fn glide_ratio(&self) -> f32 {
        self.cached_glide_ratio
    }

    /// Get the current airspeed.
    pub fn airspeed(&self) -> f32 {
        self.velocity.length()
    }

    /// Get the current altitude (Y position).
    pub fn altitude(&self) -> f32 {
        self.position.y
    }

    /// Get the current sink rate (negative Y velocity).
    pub fn sink_rate(&self) -> f32 {
        -self.velocity.y
    }

    /// Compute the best glide speed for maximum L/D.
    pub fn best_glide_speed(&self, air_density: f32) -> f32 {
        // V_best = sqrt(2 * W / (rho * S * sqrt(pi * e * AR * Cd0)))
        let w = self.mass * GRAVITY;
        let s = self.main_wing.area;
        let ar = self.main_wing.aspect_ratio;
        let e = self.main_wing.oswald;
        let cd0 = self.main_wing.cd0;

        let cl_best = (PI * e * ar * cd0).sqrt();
        if cl_best < EPSILON || s < EPSILON || air_density < EPSILON {
            return 0.0;
        }
        ((2.0 * w) / (air_density * s * cl_best)).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Paper airplane tumble
// ---------------------------------------------------------------------------

/// Paper airplane physics with tumble, flutter, and glide modes.
#[derive(Debug, Clone)]
pub struct PaperAirplane {
    /// Position in world space.
    pub position: Vec3,
    /// Velocity in world space.
    pub velocity: Vec3,
    /// Orientation quaternion.
    pub orientation: Quat,
    /// Angular velocity (body frame).
    pub angular_velocity: Vec3,
    /// Mass in kg (very light).
    pub mass: f32,
    /// Moment of inertia (diagonal).
    pub inertia: Vec3,
    /// Wing area in m^2.
    pub wing_area: f32,
    /// Chord length.
    pub chord: f32,
    /// Span.
    pub span: f32,
    /// Current flight mode.
    pub mode: PaperFlightMode,
    /// Tumble rotation rate (rad/s) when tumbling.
    pub tumble_rate: f32,
    /// Flutter amplitude.
    pub flutter_amplitude: f32,
    /// Flutter frequency.
    pub flutter_frequency: f32,
    /// Time accumulator.
    time: f32,
}

/// Flight mode for a paper airplane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PaperFlightMode {
    /// Stable gliding flight.
    Glide,
    /// Tumbling end over end.
    Tumble,
    /// Fluttering side to side (leaf-like).
    Flutter,
    /// Spiraling downward.
    Spiral,
}

impl PaperAirplane {
    /// Create a standard paper airplane.
    pub fn new(position: Vec3, throw_velocity: Vec3) -> Self {
        let mass = 0.005; // 5 grams
        let chord = 0.15;
        let span = 0.10;
        let wing_area = chord * span * 0.7; // not full rectangle

        let orientation = if throw_velocity.length() > EPSILON {
            Quat::from_rotation_arc(Vec3::Z, throw_velocity.normalize_or_zero())
        } else {
            Quat::IDENTITY
        };

        Self {
            position,
            velocity: throw_velocity,
            orientation,
            angular_velocity: Vec3::ZERO,
            mass,
            inertia: Vec3::new(
                mass * span * span / 12.0,
                mass * chord * chord / 12.0,
                mass * (span * span + chord * chord) / 12.0,
            ),
            wing_area,
            chord,
            span,
            mode: PaperFlightMode::Glide,
            tumble_rate: 15.0,
            flutter_amplitude: 0.8,
            flutter_frequency: 2.0,
            time: 0.0,
        }
    }

    /// Determine the flight mode based on current state.
    fn update_mode(&mut self) {
        let speed = self.velocity.length();
        let angular_speed = self.angular_velocity.length();

        if angular_speed > 8.0 {
            self.mode = PaperFlightMode::Tumble;
        } else if speed < 1.0 {
            self.mode = PaperFlightMode::Flutter;
        } else {
            // Check angle of attack
            let inv_orient = self.orientation.inverse();
            let body_vel = inv_orient * self.velocity;
            let alpha = (-body_vel.y).atan2(body_vel.z.abs().max(EPSILON));
            if alpha.abs() > 0.6 {
                self.mode = PaperFlightMode::Tumble;
            } else {
                self.mode = PaperFlightMode::Glide;
            }
        }
    }

    /// Step the paper airplane simulation.
    pub fn step(&mut self, dt: f32, wind: Vec3, air_density: f32) {
        if dt <= 0.0 {
            return;
        }
        self.time += dt;

        self.update_mode();

        let relative_wind = self.velocity - wind;
        let airspeed = relative_wind.length();
        let dynamic_pressure = 0.5 * air_density * airspeed * airspeed;

        let inv_orient = self.orientation.inverse();
        let body_wind = inv_orient * relative_wind;

        let mut body_force = Vec3::ZERO;
        let mut body_torque = Vec3::ZERO;

        match self.mode {
            PaperFlightMode::Glide => {
                // Standard aero forces
                let alpha = (-body_wind.y).atan2(body_wind.z.abs().max(EPSILON));

                // Simplified lift/drag for paper
                let cl = 2.0 * PI * alpha * 0.7; // reduced efficiency
                let cd = 0.05 + cl * cl / (PI * 4.0); // low AR

                let lift = dynamic_pressure * self.wing_area * cl;
                let drag = dynamic_pressure * self.wing_area * cd;

                body_force.y += lift;
                body_force.z -= drag;

                // Pitch stability
                body_torque.x -= alpha * dynamic_pressure * self.wing_area * self.chord * 0.1;
            }
            PaperFlightMode::Tumble => {
                // High drag, tumbling rotation
                let cd = 1.5; // high drag during tumble
                let drag = dynamic_pressure * self.wing_area * cd;
                if airspeed > EPSILON {
                    body_force -= body_wind / airspeed * drag;
                }

                // Tumble torque (around pitch axis)
                let target_rate = self.tumble_rate;
                let pitch_error = target_rate - self.angular_velocity.x;
                body_torque.x += pitch_error * 0.01;
            }
            PaperFlightMode::Flutter => {
                // Leaf-like flutter: oscillating lateral drift
                let phase = self.time * self.flutter_frequency * 2.0 * PI;
                let flutter_force = self.flutter_amplitude * phase.sin();

                body_force.x += flutter_force * self.mass * GRAVITY * 0.3;
                body_force.y -= self.mass * GRAVITY * 0.3; // slow descent

                // Drag
                let cd = 1.2;
                let drag = dynamic_pressure * self.wing_area * cd;
                if airspeed > EPSILON {
                    body_force -= body_wind / airspeed * drag;
                }

                // Oscillating roll
                body_torque.z += flutter_force * 0.001;
            }
            PaperFlightMode::Spiral => {
                // Spiral: combination of lift and constant yaw rate
                let alpha = (-body_wind.y).atan2(body_wind.z.abs().max(EPSILON));
                let cl = 2.0 * PI * alpha * 0.5;
                let cd = 0.1 + cl * cl / (PI * 3.0);

                body_force.y += dynamic_pressure * self.wing_area * cl;
                body_force.z -= dynamic_pressure * self.wing_area * cd;

                // Constant yaw
                body_torque.y += 0.001;
                // Banking
                body_torque.z += 0.0005;
            }
        }

        // Transform forces to world frame
        let world_force = self.orientation * body_force;
        let gravity = Vec3::new(0.0, -GRAVITY * self.mass, 0.0);
        let total_force = world_force + gravity;

        // Angular integration
        let angular_accel = Vec3::new(
            body_torque.x / self.inertia.x.max(EPSILON),
            body_torque.y / self.inertia.y.max(EPSILON),
            body_torque.z / self.inertia.z.max(EPSILON),
        );

        let angular_damping = 0.95_f32;
        self.angular_velocity = self.angular_velocity * angular_damping + angular_accel * dt;

        let angle = self.angular_velocity.length() * dt;
        if angle > EPSILON {
            let axis = self.angular_velocity.normalize_or_zero();
            let dq = Quat::from_axis_angle(axis, angle);
            self.orientation = (self.orientation * dq).normalize();
        }

        // Linear integration
        let accel = total_force / self.mass;
        self.velocity += accel * dt;
        self.position += self.velocity * dt;
    }
}

// ---------------------------------------------------------------------------
// Aerodynamic body (general purpose)
// ---------------------------------------------------------------------------

/// A general-purpose aerodynamic body that computes lift and drag.
#[derive(Debug, Clone)]
pub struct AerodynamicBody {
    /// Wing/surface aero model.
    pub model: AeroModel,
    /// Additional drag surfaces (fuselage, landing gear, etc.).
    pub extra_drag: Vec<SurfaceDrag>,
    /// Current computed coefficients.
    pub coefficients: AeroCoefficients,
    /// Last computed total aerodynamic force (world space).
    pub total_force: Vec3,
    /// Last computed total aerodynamic torque (body space).
    pub total_torque: Vec3,
    /// Air density at current altitude.
    pub air_density: f32,
}

impl AerodynamicBody {
    /// Create a new aerodynamic body with a wing.
    pub fn new(model: AeroModel) -> Self {
        Self {
            model,
            extra_drag: Vec::new(),
            coefficients: AeroCoefficients::default(),
            total_force: Vec3::ZERO,
            total_torque: Vec3::ZERO,
            air_density: AIR_DENSITY_SEA_LEVEL,
        }
    }

    /// Add an additional drag surface.
    pub fn add_drag_surface(&mut self, surface: SurfaceDrag) {
        self.extra_drag.push(surface);
    }

    /// Compute air density at altitude (simple exponential atmosphere model).
    pub fn density_at_altitude(altitude: f32) -> f32 {
        // Scale height ~8500m
        AIR_DENSITY_SEA_LEVEL * (-altitude / 8500.0).exp()
    }

    /// Update air density based on altitude.
    pub fn update_density(&mut self, altitude: f32) {
        self.air_density = Self::density_at_altitude(altitude);
    }

    /// Compute all aerodynamic forces given body orientation and velocity.
    /// Returns (force_world, torque_body).
    pub fn compute_forces(
        &mut self,
        velocity: Vec3,
        orientation: Quat,
        wind: Vec3,
    ) -> (Vec3, Vec3) {
        let relative_wind = velocity - wind;
        let airspeed = relative_wind.length();

        if airspeed < EPSILON {
            self.total_force = Vec3::ZERO;
            self.total_torque = Vec3::ZERO;
            self.coefficients = AeroCoefficients::default();
            return (Vec3::ZERO, Vec3::ZERO);
        }

        // Body-frame airflow
        let inv_orient = orientation.inverse();
        let body_wind = inv_orient * relative_wind;

        // Angle of attack
        let alpha = (-body_wind.y).atan2(body_wind.z.abs().max(EPSILON));

        let dynamic_pressure = 0.5 * self.air_density * airspeed * airspeed;

        // Main wing coefficients
        self.coefficients = self.model.compute_coefficients(alpha, airspeed);

        let lift = self.model.lift_force(dynamic_pressure, self.coefficients.cl);
        let drag = self.model.drag_force(dynamic_pressure, self.coefficients.cd);
        let moment = self.model.pitching_moment(dynamic_pressure, self.coefficients.cm);

        // Lift perpendicular to airflow, drag along airflow (body frame)
        let body_force = Vec3::new(0.0, lift, -drag);
        let body_torque = Vec3::new(moment, 0.0, 0.0);

        // Extra drag surfaces
        let mut extra_force = Vec3::ZERO;
        for surface in &self.extra_drag {
            extra_force += surface.compute_drag(relative_wind, self.air_density);
        }

        let world_force = orientation * body_force + extra_force;

        self.total_force = world_force;
        self.total_torque = body_torque;

        (world_force, body_torque)
    }

    /// Estimate the Reynolds number at the current conditions.
    pub fn reynolds_number(&self) -> f32 {
        self.coefficients.reynolds
    }

    /// Check if the wing is currently stalled.
    pub fn is_stalled(&self) -> bool {
        self.coefficients.stalled
    }
}

// ---------------------------------------------------------------------------
// ECS Components
// ---------------------------------------------------------------------------

/// ECS component for aerodynamic simulation on an entity.
#[derive(Debug, Clone)]
pub struct AerodynamicsComponent {
    /// The aerodynamic body model.
    pub body: AerodynamicBody,
    /// Whether this component is enabled.
    pub enabled: bool,
    /// Whether to auto-update air density based on entity Y position.
    pub altitude_density: bool,
}

impl AerodynamicsComponent {
    /// Create a new aerodynamics component with a flat plate model.
    pub fn flat_plate(area: f32, chord: f32) -> Self {
        let span = area / chord.max(EPSILON);
        let model = AeroModel::flat_plate(chord, span);
        Self {
            body: AerodynamicBody::new(model),
            enabled: true,
            altitude_density: true,
        }
    }

    /// Create a new aerodynamics component with a wing profile.
    pub fn wing(airfoil: NacaAirfoil, chord: f32, span: f32) -> Self {
        let model = AeroModel::new(airfoil, chord, span);
        Self {
            body: AerodynamicBody::new(model),
            enabled: true,
            altitude_density: true,
        }
    }
}

/// System that updates all aerodynamics components.
pub struct AerodynamicsSystem {
    /// Global wind vector (can be overridden per-entity).
    pub global_wind: Vec3,
}

impl AerodynamicsSystem {
    /// Create a new aerodynamics system.
    pub fn new() -> Self {
        Self {
            global_wind: Vec3::ZERO,
        }
    }

    /// Set the global wind vector.
    pub fn set_wind(&mut self, wind: Vec3) {
        self.global_wind = wind;
    }
}
