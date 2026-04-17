//! Global wind system for physics-driven environmental effects.
//!
//! Provides:
//! - Directional wind with configurable speed and direction
//! - Gust patterns: periodic sine-wave gusts and random bursts
//! - Turbulence zones: localized chaotic wind regions
//! - Wind affecting particles, cloth, vegetation, and audio systems
//! - Beaufort scale presets (calm to hurricane)
//! - Spatial wind field sampling at arbitrary world positions
//! - Wind source composition (multiple sources, superposition)
//! - ECS integration via `WindComponent` and `WindSystem`

use glam::Vec3;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Small epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-7;
/// Default global wind direction (positive X axis).
const DEFAULT_WIND_DIRECTION: Vec3 = Vec3::new(1.0, 0.0, 0.0);
/// Default global wind speed in m/s.
const DEFAULT_WIND_SPEED: f32 = 5.0;
/// Default gust period in seconds.
const DEFAULT_GUST_PERIOD: f32 = 4.0;
/// Default gust strength multiplier.
const DEFAULT_GUST_STRENGTH: f32 = 1.5;
/// Maximum number of turbulence zones.
const MAX_TURBULENCE_ZONES: usize = 32;
/// Maximum number of wind sources.
const MAX_WIND_SOURCES: usize = 64;
/// Pi constant.
const PI: f32 = std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Beaufort scale
// ---------------------------------------------------------------------------

/// Beaufort wind scale levels (0-12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BeaufortScale {
    /// 0: Calm (< 0.5 m/s).
    Calm = 0,
    /// 1: Light air (0.5-1.5 m/s).
    LightAir = 1,
    /// 2: Light breeze (1.6-3.3 m/s).
    LightBreeze = 2,
    /// 3: Gentle breeze (3.4-5.5 m/s).
    GentleBreeze = 3,
    /// 4: Moderate breeze (5.5-7.9 m/s).
    ModerateBreeze = 4,
    /// 5: Fresh breeze (8.0-10.7 m/s).
    FreshBreeze = 5,
    /// 6: Strong breeze (10.8-13.8 m/s).
    StrongBreeze = 6,
    /// 7: Near gale (13.9-17.1 m/s).
    NearGale = 7,
    /// 8: Gale (17.2-20.7 m/s).
    Gale = 8,
    /// 9: Strong gale (20.8-24.4 m/s).
    StrongGale = 9,
    /// 10: Storm (24.5-28.4 m/s).
    Storm = 10,
    /// 11: Violent storm (28.5-32.6 m/s).
    ViolentStorm = 11,
    /// 12: Hurricane (>= 32.7 m/s).
    Hurricane = 12,
}

impl BeaufortScale {
    /// Get the average wind speed for this Beaufort level in m/s.
    pub fn speed(&self) -> f32 {
        match self {
            BeaufortScale::Calm => 0.25,
            BeaufortScale::LightAir => 1.0,
            BeaufortScale::LightBreeze => 2.5,
            BeaufortScale::GentleBreeze => 4.5,
            BeaufortScale::ModerateBreeze => 6.7,
            BeaufortScale::FreshBreeze => 9.3,
            BeaufortScale::StrongBreeze => 12.3,
            BeaufortScale::NearGale => 15.5,
            BeaufortScale::Gale => 19.0,
            BeaufortScale::StrongGale => 22.6,
            BeaufortScale::Storm => 26.5,
            BeaufortScale::ViolentStorm => 30.5,
            BeaufortScale::Hurricane => 35.0,
        }
    }

    /// Get a descriptive name for this Beaufort level.
    pub fn description(&self) -> &'static str {
        match self {
            BeaufortScale::Calm => "Calm",
            BeaufortScale::LightAir => "Light air",
            BeaufortScale::LightBreeze => "Light breeze",
            BeaufortScale::GentleBreeze => "Gentle breeze",
            BeaufortScale::ModerateBreeze => "Moderate breeze",
            BeaufortScale::FreshBreeze => "Fresh breeze",
            BeaufortScale::StrongBreeze => "Strong breeze",
            BeaufortScale::NearGale => "Near gale",
            BeaufortScale::Gale => "Gale",
            BeaufortScale::StrongGale => "Strong gale",
            BeaufortScale::Storm => "Storm",
            BeaufortScale::ViolentStorm => "Violent storm",
            BeaufortScale::Hurricane => "Hurricane",
        }
    }

    /// Get the gust factor (typical gust/mean wind ratio) for this level.
    pub fn gust_factor(&self) -> f32 {
        match self {
            BeaufortScale::Calm => 1.0,
            BeaufortScale::LightAir => 1.2,
            BeaufortScale::LightBreeze => 1.3,
            BeaufortScale::GentleBreeze => 1.4,
            BeaufortScale::ModerateBreeze => 1.5,
            BeaufortScale::FreshBreeze => 1.5,
            BeaufortScale::StrongBreeze => 1.6,
            BeaufortScale::NearGale => 1.7,
            BeaufortScale::Gale => 1.8,
            BeaufortScale::StrongGale => 1.8,
            BeaufortScale::Storm => 1.9,
            BeaufortScale::ViolentStorm => 2.0,
            BeaufortScale::Hurricane => 2.0,
        }
    }

    /// Get the turbulence intensity (ratio of wind speed standard deviation to mean).
    pub fn turbulence_intensity(&self) -> f32 {
        match self {
            BeaufortScale::Calm => 0.0,
            BeaufortScale::LightAir => 0.05,
            BeaufortScale::LightBreeze => 0.1,
            BeaufortScale::GentleBreeze => 0.12,
            BeaufortScale::ModerateBreeze => 0.15,
            BeaufortScale::FreshBreeze => 0.18,
            BeaufortScale::StrongBreeze => 0.20,
            BeaufortScale::NearGale => 0.22,
            BeaufortScale::Gale => 0.25,
            BeaufortScale::StrongGale => 0.27,
            BeaufortScale::Storm => 0.30,
            BeaufortScale::ViolentStorm => 0.33,
            BeaufortScale::Hurricane => 0.35,
        }
    }

    /// Determine the Beaufort scale from a wind speed in m/s.
    pub fn from_speed(speed: f32) -> Self {
        if speed < 0.5 { BeaufortScale::Calm }
        else if speed < 1.6 { BeaufortScale::LightAir }
        else if speed < 3.4 { BeaufortScale::LightBreeze }
        else if speed < 5.5 { BeaufortScale::GentleBreeze }
        else if speed < 8.0 { BeaufortScale::ModerateBreeze }
        else if speed < 10.8 { BeaufortScale::FreshBreeze }
        else if speed < 13.9 { BeaufortScale::StrongBreeze }
        else if speed < 17.2 { BeaufortScale::NearGale }
        else if speed < 20.8 { BeaufortScale::Gale }
        else if speed < 24.5 { BeaufortScale::StrongGale }
        else if speed < 28.5 { BeaufortScale::Storm }
        else if speed < 32.7 { BeaufortScale::ViolentStorm }
        else { BeaufortScale::Hurricane }
    }
}

// ---------------------------------------------------------------------------
// Gust pattern
// ---------------------------------------------------------------------------

/// A gust pattern that modulates the base wind.
#[derive(Debug, Clone)]
pub enum GustPattern {
    /// No gusts.
    None,
    /// Periodic sinusoidal gusts.
    Periodic {
        /// Period in seconds.
        period: f32,
        /// Strength multiplier at peak (e.g., 1.5 = 50% stronger).
        strength: f32,
        /// Phase offset in seconds.
        phase: f32,
    },
    /// Random gusts with configurable frequency and intensity.
    Random {
        /// Average time between gusts in seconds.
        interval: f32,
        /// Gust duration in seconds.
        duration: f32,
        /// Gust strength multiplier.
        strength: f32,
        /// Pseudo-random state.
        rng_state: u32,
        /// Time of next gust.
        next_gust_time: f32,
        /// Whether currently gusting.
        is_gusting: bool,
        /// Current gust timer.
        gust_timer: f32,
    },
    /// Combined periodic and random gusts.
    Combined {
        /// Periodic component period.
        periodic_period: f32,
        /// Periodic component strength.
        periodic_strength: f32,
        /// Random component interval.
        random_interval: f32,
        /// Random component duration.
        random_duration: f32,
        /// Random component strength.
        random_strength: f32,
        /// Random state.
        rng_state: u32,
        /// Next random gust time.
        next_gust_time: f32,
        /// Whether randomly gusting.
        is_gusting: bool,
        /// Current random gust timer.
        gust_timer: f32,
    },
}

impl GustPattern {
    /// Create a periodic gust pattern.
    pub fn periodic(period: f32, strength: f32) -> Self {
        GustPattern::Periodic {
            period,
            strength,
            phase: 0.0,
        }
    }

    /// Create a random gust pattern.
    pub fn random(interval: f32, duration: f32, strength: f32) -> Self {
        GustPattern::Random {
            interval,
            duration,
            strength,
            rng_state: 12345,
            next_gust_time: interval,
            is_gusting: false,
            gust_timer: 0.0,
        }
    }

    /// Create a combined gust pattern.
    pub fn combined(
        periodic_period: f32,
        periodic_strength: f32,
        random_interval: f32,
        random_strength: f32,
    ) -> Self {
        GustPattern::Combined {
            periodic_period,
            periodic_strength,
            random_interval,
            random_duration: 1.0,
            random_strength,
            rng_state: 67890,
            next_gust_time: random_interval,
            is_gusting: false,
            gust_timer: 0.0,
        }
    }

    /// Simple xorshift32 for random gust timing.
    fn xorshift(state: &mut u32) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        (*state as f32) / (u32::MAX as f32)
    }

    /// Evaluate the gust multiplier at the given time. Returns a value >= 1.0
    /// (1.0 = no gust, higher = gustier).
    pub fn evaluate(&mut self, time: f32, dt: f32) -> f32 {
        match self {
            GustPattern::None => 1.0,
            GustPattern::Periodic { period, strength, phase } => {
                let t = (time + *phase) / period.max(EPSILON);
                let sine = (t * 2.0 * PI).sin();
                // Gusts are only additive (positive sine portion)
                1.0 + (*strength - 1.0) * sine.max(0.0)
            }
            GustPattern::Random {
                interval,
                duration,
                strength,
                rng_state,
                next_gust_time,
                is_gusting,
                gust_timer,
            } => {
                if *is_gusting {
                    *gust_timer += dt;
                    if *gust_timer >= *duration {
                        *is_gusting = false;
                        *next_gust_time = time + *interval * (0.5 + Self::xorshift(rng_state));
                    }
                    // Smooth gust envelope
                    let progress = *gust_timer / duration.max(EPSILON);
                    let envelope = (progress * PI).sin(); // bell curve
                    1.0 + (*strength - 1.0) * envelope
                } else if time >= *next_gust_time {
                    *is_gusting = true;
                    *gust_timer = 0.0;
                    1.0
                } else {
                    1.0
                }
            }
            GustPattern::Combined {
                periodic_period,
                periodic_strength,
                random_interval,
                random_duration,
                random_strength,
                rng_state,
                next_gust_time,
                is_gusting,
                gust_timer,
            } => {
                // Periodic component
                let periodic = {
                    let t = time / periodic_period.max(EPSILON);
                    1.0 + (*periodic_strength - 1.0) * (t * 2.0 * PI).sin().max(0.0)
                };

                // Random component
                let random = {
                    if *is_gusting {
                        *gust_timer += dt;
                        if *gust_timer >= *random_duration {
                            *is_gusting = false;
                            *next_gust_time = time + *random_interval
                                * (0.5 + Self::xorshift(rng_state));
                        }
                        let progress = *gust_timer / random_duration.max(EPSILON);
                        let envelope = (progress * PI).sin();
                        1.0 + (*random_strength - 1.0) * envelope
                    } else if time >= *next_gust_time {
                        *is_gusting = true;
                        *gust_timer = 0.0;
                        1.0
                    } else {
                        1.0
                    }
                };

                // Multiply: both can amplify independently
                periodic * random
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Turbulence zone
// ---------------------------------------------------------------------------

/// A localized region of chaotic wind.
#[derive(Debug, Clone)]
pub struct TurbulenceZone {
    /// Center of the turbulence zone in world space.
    pub center: Vec3,
    /// Radius of the zone (spherical).
    pub radius: f32,
    /// Turbulence intensity (0 = calm, 1 = maximum chaos).
    pub intensity: f32,
    /// Frequency of turbulent oscillations in Hz.
    pub frequency: f32,
    /// Whether this zone is active.
    pub active: bool,
    /// Pseudo-random seed for deterministic turbulence.
    pub seed: u32,
    /// Internal phase accumulators (3 axes).
    phases: [f32; 3],
}

impl TurbulenceZone {
    /// Create a new turbulence zone.
    pub fn new(center: Vec3, radius: f32, intensity: f32) -> Self {
        Self {
            center,
            radius,
            intensity,
            frequency: 2.0,
            active: true,
            seed: 42,
            phases: [0.0, 1.37, 2.89], // offset phases for variety
        }
    }

    /// Sample the turbulent wind contribution at a world-space position.
    pub fn sample(&self, position: Vec3, time: f32) -> Vec3 {
        if !self.active {
            return Vec3::ZERO;
        }

        let diff = position - self.center;
        let dist = diff.length();
        if dist > self.radius {
            return Vec3::ZERO;
        }

        // Falloff from center to edge
        let falloff = 1.0 - (dist / self.radius);
        let falloff = falloff * falloff; // quadratic falloff

        // Position-dependent phase offsets for spatial variation
        let px = position.x * 0.3;
        let py = position.y * 0.3;
        let pz = position.z * 0.3;

        let freq = self.frequency;
        let t = time * freq;

        // Multi-octave noise-like turbulence using sine combinations
        let wx = (t + self.phases[0] + px).sin() * 0.5
            + (t * 1.7 + self.phases[1] + px * 2.0).sin() * 0.3
            + (t * 3.1 + self.phases[2] + px * 4.0).sin() * 0.2;

        let wy = (t * 0.9 + self.phases[1] + py).sin() * 0.5
            + (t * 2.1 + self.phases[2] + py * 2.0).sin() * 0.3
            + (t * 3.7 + self.phases[0] + py * 4.0).sin() * 0.2;

        let wz = (t * 1.1 + self.phases[2] + pz).sin() * 0.5
            + (t * 1.9 + self.phases[0] + pz * 2.0).sin() * 0.3
            + (t * 2.9 + self.phases[1] + pz * 4.0).sin() * 0.2;

        Vec3::new(wx, wy, wz) * self.intensity * falloff
    }

    /// Check if a point is within this turbulence zone.
    pub fn contains(&self, point: Vec3) -> bool {
        (point - self.center).length() <= self.radius
    }
}

// ---------------------------------------------------------------------------
// Wind source
// ---------------------------------------------------------------------------

/// Unique identifier for a wind source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindSourceId(pub u32);

/// A wind source contributing to the global wind field.
#[derive(Debug, Clone)]
pub enum WindSource {
    /// Directional wind (global or bounded).
    Directional {
        id: WindSourceId,
        /// Wind direction (normalized) and speed (magnitude of the vector).
        direction: Vec3,
        speed: f32,
        /// Optional AABB bounds.
        bounds: Option<(Vec3, Vec3)>,
        /// Whether this source is active.
        active: bool,
    },
    /// Point source (radial wind from a point, like an explosion or fan).
    Point {
        id: WindSourceId,
        /// Position of the source.
        position: Vec3,
        /// Strength (positive = outward, negative = inward).
        strength: f32,
        /// Interaction radius.
        radius: f32,
        /// Falloff power (1 = linear, 2 = inverse square).
        falloff: f32,
        /// Whether this source is active.
        active: bool,
    },
    /// Vortex wind (circular/tornado-like).
    Vortex {
        id: WindSourceId,
        /// Center of the vortex.
        position: Vec3,
        /// Axis of rotation (typically Y-up).
        axis: Vec3,
        /// Tangential strength.
        strength: f32,
        /// Vortex radius.
        radius: f32,
        /// Upward draft strength (for tornado-like effects).
        updraft: f32,
        /// Whether this source is active.
        active: bool,
    },
}

impl WindSource {
    /// Get the ID of this wind source.
    pub fn id(&self) -> WindSourceId {
        match self {
            WindSource::Directional { id, .. } => *id,
            WindSource::Point { id, .. } => *id,
            WindSource::Vortex { id, .. } => *id,
        }
    }

    /// Check if this wind source is active.
    pub fn is_active(&self) -> bool {
        match self {
            WindSource::Directional { active, .. } => *active,
            WindSource::Point { active, .. } => *active,
            WindSource::Vortex { active, .. } => *active,
        }
    }

    /// Set the active state.
    pub fn set_active(&mut self, state: bool) {
        match self {
            WindSource::Directional { active, .. } => *active = state,
            WindSource::Point { active, .. } => *active = state,
            WindSource::Vortex { active, .. } => *active = state,
        }
    }

    /// Sample the wind contribution at a world-space position.
    pub fn sample(&self, position: Vec3) -> Vec3 {
        match self {
            WindSource::Directional {
                direction,
                speed,
                bounds,
                active,
                ..
            } => {
                if !active {
                    return Vec3::ZERO;
                }
                if let Some((min, max)) = bounds {
                    if position.x < min.x || position.x > max.x
                        || position.y < min.y || position.y > max.y
                        || position.z < min.z || position.z > max.z
                    {
                        return Vec3::ZERO;
                    }
                }
                direction.normalize_or_zero() * *speed
            }
            WindSource::Point {
                position: source_pos,
                strength,
                radius,
                falloff,
                active,
                ..
            } => {
                if !active {
                    return Vec3::ZERO;
                }
                let diff = position - *source_pos;
                let dist = diff.length();
                if dist > *radius || dist < EPSILON {
                    return Vec3::ZERO;
                }
                let dir = diff / dist;
                let factor = 1.0 - (dist / radius).powf(*falloff);
                dir * *strength * factor
            }
            WindSource::Vortex {
                position: vortex_pos,
                axis,
                strength,
                radius,
                updraft,
                active,
                ..
            } => {
                if !active {
                    return Vec3::ZERO;
                }
                let to_point = position - *vortex_pos;
                let axis_n = axis.normalize_or_zero();
                let projected = to_point - axis_n * to_point.dot(axis_n);
                let dist = projected.length();
                if dist > *radius || dist < EPSILON {
                    return Vec3::ZERO;
                }

                let factor = 1.0 - dist / radius;
                let tangent = axis_n.cross(projected.normalize_or_zero());
                let wind = tangent * *strength * factor + axis_n * *updraft * factor;
                wind
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Wind settings (global configuration)
// ---------------------------------------------------------------------------

/// Configuration for the global wind system.
#[derive(Debug, Clone)]
pub struct WindSettings {
    /// Base wind direction (normalized).
    pub direction: Vec3,
    /// Base wind speed in m/s.
    pub speed: f32,
    /// Gust pattern.
    pub gust_pattern: GustPattern,
    /// Whether the wind system is enabled.
    pub enabled: bool,
    /// Global time scale.
    pub time_scale: f32,
    /// Height gradient: wind speed increases with altitude.
    /// Factor per meter of elevation (e.g., 0.01 = 1% more per meter).
    pub height_gradient: f32,
    /// Ground level Y coordinate (wind is zero below this).
    pub ground_level: f32,
    /// Wind direction change rate in radians per second (slow drift).
    pub direction_drift_rate: f32,
    /// Current Beaufort scale level (informational, derived from speed).
    pub beaufort: BeaufortScale,
}

impl Default for WindSettings {
    fn default() -> Self {
        Self {
            direction: DEFAULT_WIND_DIRECTION,
            speed: DEFAULT_WIND_SPEED,
            gust_pattern: GustPattern::periodic(DEFAULT_GUST_PERIOD, DEFAULT_GUST_STRENGTH),
            enabled: true,
            time_scale: 1.0,
            height_gradient: 0.005,
            ground_level: 0.0,
            direction_drift_rate: 0.01,
            beaufort: BeaufortScale::GentleBreeze,
        }
    }
}

impl WindSettings {
    /// Create settings from a Beaufort scale level.
    pub fn from_beaufort(level: BeaufortScale) -> Self {
        let speed = level.speed();
        let gust_factor = level.gust_factor();
        let turbulence = level.turbulence_intensity();

        Self {
            direction: DEFAULT_WIND_DIRECTION,
            speed,
            gust_pattern: GustPattern::combined(
                4.0 + (12 - level as u8) as f32 * 0.3, // period varies
                gust_factor,
                6.0,
                1.0 + turbulence,
            ),
            enabled: true,
            time_scale: 1.0,
            height_gradient: 0.005,
            ground_level: 0.0,
            direction_drift_rate: 0.01 * (1.0 + turbulence),
            beaufort: level,
        }
    }

    /// Set the wind to a Beaufort level preset.
    pub fn set_beaufort(&mut self, level: BeaufortScale) {
        let preset = Self::from_beaufort(level);
        self.speed = preset.speed;
        self.gust_pattern = preset.gust_pattern;
        self.beaufort = level;
    }
}

// ---------------------------------------------------------------------------
// Wind field (the main system)
// ---------------------------------------------------------------------------

/// Sample result from the wind field.
#[derive(Debug, Clone, Copy)]
pub struct WindSample {
    /// Wind velocity at the sampled point.
    pub velocity: Vec3,
    /// Wind speed (magnitude of velocity).
    pub speed: f32,
    /// Wind direction (normalized velocity, or zero if calm).
    pub direction: Vec3,
    /// Gust multiplier at the sample time.
    pub gust_factor: f32,
    /// Turbulence contribution at the sample point.
    pub turbulence: Vec3,
    /// Current Beaufort scale of the sampled wind.
    pub beaufort: BeaufortScale,
}

/// The main wind field system that composes all wind sources, gusts,
/// and turbulence zones into a unified sampling interface.
#[derive(Debug)]
pub struct WindField {
    /// Global wind settings.
    pub settings: WindSettings,
    /// Additional wind sources (fans, explosions, vortices).
    pub sources: Vec<WindSource>,
    /// Turbulence zones.
    pub turbulence_zones: Vec<TurbulenceZone>,
    /// Current simulation time.
    time: f32,
    /// Next wind source ID.
    next_source_id: u32,
    /// Direction drift angle accumulator.
    drift_angle: f32,
    /// Pseudo-random state for direction drift.
    rng_state: u32,
}

impl WindField {
    /// Create a new wind field with default settings.
    pub fn new() -> Self {
        Self {
            settings: WindSettings::default(),
            sources: Vec::new(),
            turbulence_zones: Vec::new(),
            time: 0.0,
            next_source_id: 0,
            drift_angle: 0.0,
            rng_state: 314159,
        }
    }

    /// Create a wind field with specific settings.
    pub fn with_settings(settings: WindSettings) -> Self {
        Self {
            settings,
            sources: Vec::new(),
            turbulence_zones: Vec::new(),
            time: 0.0,
            next_source_id: 0,
            drift_angle: 0.0,
            rng_state: 314159,
        }
    }

    /// Create a wind field from a Beaufort preset.
    pub fn from_beaufort(level: BeaufortScale) -> Self {
        Self::with_settings(WindSettings::from_beaufort(level))
    }

    /// Add a directional wind source. Returns its ID.
    pub fn add_directional(
        &mut self,
        direction: Vec3,
        speed: f32,
        bounds: Option<(Vec3, Vec3)>,
    ) -> WindSourceId {
        let id = WindSourceId(self.next_source_id);
        self.next_source_id += 1;
        self.sources.push(WindSource::Directional {
            id,
            direction: direction.normalize_or_zero(),
            speed,
            bounds,
            active: true,
        });
        id
    }

    /// Add a point wind source (fan, explosion). Returns its ID.
    pub fn add_point_source(
        &mut self,
        position: Vec3,
        strength: f32,
        radius: f32,
    ) -> WindSourceId {
        let id = WindSourceId(self.next_source_id);
        self.next_source_id += 1;
        self.sources.push(WindSource::Point {
            id,
            position,
            strength,
            radius,
            falloff: 1.0,
            active: true,
        });
        id
    }

    /// Add a vortex wind source (tornado). Returns its ID.
    pub fn add_vortex(
        &mut self,
        position: Vec3,
        axis: Vec3,
        strength: f32,
        radius: f32,
        updraft: f32,
    ) -> WindSourceId {
        let id = WindSourceId(self.next_source_id);
        self.next_source_id += 1;
        self.sources.push(WindSource::Vortex {
            id,
            position,
            axis: axis.normalize_or_zero(),
            strength,
            radius,
            updraft,
            active: true,
        });
        id
    }

    /// Remove a wind source by ID.
    pub fn remove_source(&mut self, id: WindSourceId) {
        self.sources.retain(|s| s.id() != id);
    }

    /// Add a turbulence zone.
    pub fn add_turbulence_zone(&mut self, zone: TurbulenceZone) {
        if self.turbulence_zones.len() < MAX_TURBULENCE_ZONES {
            self.turbulence_zones.push(zone);
        }
    }

    /// Remove all turbulence zones.
    pub fn clear_turbulence_zones(&mut self) {
        self.turbulence_zones.clear();
    }

    /// Simple xorshift32 pseudo-random.
    fn next_random(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 17;
        self.rng_state ^= self.rng_state << 5;
        (self.rng_state as f32) / (u32::MAX as f32)
    }

    /// Update the wind system (advance time, update gusts, drift direction).
    pub fn update(&mut self, dt: f32) {
        let dt = dt * self.settings.time_scale;
        self.time += dt;

        // Drift the wind direction slowly
        if self.settings.direction_drift_rate > 0.0 {
            let drift = (self.next_random() - 0.5) * 2.0
                * self.settings.direction_drift_rate * dt;
            self.drift_angle += drift;

            let cos_d = self.drift_angle.cos();
            let sin_d = self.drift_angle.sin();
            let base = DEFAULT_WIND_DIRECTION;
            self.settings.direction = Vec3::new(
                base.x * cos_d - base.z * sin_d,
                base.y,
                base.x * sin_d + base.z * cos_d,
            ).normalize_or_zero();
        }

        // Update Beaufort level
        let current_speed = self.settings.speed
            * self.settings.gust_pattern.evaluate(self.time, dt);
        self.settings.beaufort = BeaufortScale::from_speed(current_speed);
    }

    /// Sample the complete wind field at a world-space position.
    /// This is the main query function for all wind-affected systems.
    pub fn sample_at(&mut self, position: Vec3) -> WindSample {
        if !self.settings.enabled {
            return WindSample {
                velocity: Vec3::ZERO,
                speed: 0.0,
                direction: Vec3::ZERO,
                gust_factor: 1.0,
                turbulence: Vec3::ZERO,
                beaufort: BeaufortScale::Calm,
            };
        }

        // Base wind with height gradient
        let height_above_ground = (position.y - self.settings.ground_level).max(0.0);
        let height_factor = 1.0 + self.settings.height_gradient * height_above_ground;
        let base_speed = self.settings.speed * height_factor;

        // Gust modulation - use a small dt approximation for the gust evaluation
        let gust_factor = self.settings.gust_pattern.evaluate(self.time, 0.0);
        let gusted_speed = base_speed * gust_factor;

        let mut velocity = self.settings.direction * gusted_speed;

        // Additional sources
        for source in &self.sources {
            velocity += source.sample(position);
        }

        // Turbulence
        let mut turbulence = Vec3::ZERO;
        for zone in &self.turbulence_zones {
            turbulence += zone.sample(position, self.time);
        }
        velocity += turbulence;

        let speed = velocity.length();
        let direction = if speed > EPSILON {
            velocity / speed
        } else {
            Vec3::ZERO
        };

        WindSample {
            velocity,
            speed,
            direction,
            gust_factor,
            turbulence,
            beaufort: BeaufortScale::from_speed(speed),
        }
    }

    /// Sample without mutating (uses last known gust state). Useful for
    /// parallel access.
    pub fn sample_at_readonly(&self, position: Vec3) -> WindSample {
        if !self.settings.enabled {
            return WindSample {
                velocity: Vec3::ZERO,
                speed: 0.0,
                direction: Vec3::ZERO,
                gust_factor: 1.0,
                turbulence: Vec3::ZERO,
                beaufort: BeaufortScale::Calm,
            };
        }

        let height_above_ground = (position.y - self.settings.ground_level).max(0.0);
        let height_factor = 1.0 + self.settings.height_gradient * height_above_ground;
        let base_speed = self.settings.speed * height_factor;

        let mut velocity = self.settings.direction * base_speed;

        for source in &self.sources {
            velocity += source.sample(position);
        }

        let mut turbulence = Vec3::ZERO;
        for zone in &self.turbulence_zones {
            turbulence += zone.sample(position, self.time);
        }
        velocity += turbulence;

        let speed = velocity.length();
        let direction = if speed > EPSILON {
            velocity / speed
        } else {
            Vec3::ZERO
        };

        WindSample {
            velocity,
            speed,
            direction,
            gust_factor: 1.0,
            turbulence,
            beaufort: BeaufortScale::from_speed(speed),
        }
    }

    /// Get the current simulation time.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Get the number of active wind sources.
    pub fn active_source_count(&self) -> usize {
        self.sources.iter().filter(|s| s.is_active()).count()
    }

    /// Get the current base wind vector (without gusts or turbulence).
    pub fn base_wind(&self) -> Vec3 {
        self.settings.direction * self.settings.speed
    }

    /// Clear all sources and zones.
    pub fn clear(&mut self) {
        self.sources.clear();
        self.turbulence_zones.clear();
    }

    /// Set the global wind speed and direction.
    pub fn set_wind(&mut self, direction: Vec3, speed: f32) {
        self.settings.direction = direction.normalize_or_zero();
        self.settings.speed = speed;
    }
}

// ---------------------------------------------------------------------------
// ECS Components
// ---------------------------------------------------------------------------

/// ECS component that marks an entity as affected by wind.
#[derive(Debug, Clone)]
pub struct WindReceiverComponent {
    /// How strongly the entity is affected by wind (0 = immune, 1 = full effect).
    pub influence: f32,
    /// Drag area exposed to wind (for force computation).
    pub drag_area: f32,
    /// Drag coefficient.
    pub drag_cd: f32,
    /// Whether this receiver is enabled.
    pub enabled: bool,
    /// Last sampled wind at this entity's position (cached).
    pub last_sample: WindSample,
}

impl WindReceiverComponent {
    /// Create a new wind receiver with default settings.
    pub fn new(drag_area: f32) -> Self {
        Self {
            influence: 1.0,
            drag_area,
            drag_cd: 1.0,
            enabled: true,
            last_sample: WindSample {
                velocity: Vec3::ZERO,
                speed: 0.0,
                direction: Vec3::ZERO,
                gust_factor: 1.0,
                turbulence: Vec3::ZERO,
                beaufort: BeaufortScale::Calm,
            },
        }
    }

    /// Compute the wind force on this receiver given the wind sample and air density.
    pub fn compute_force(&self, air_density: f32) -> Vec3 {
        if !self.enabled {
            return Vec3::ZERO;
        }
        let dynamic_pressure = 0.5 * air_density * self.last_sample.speed * self.last_sample.speed;
        self.last_sample.direction * dynamic_pressure * self.drag_area * self.drag_cd * self.influence
    }
}

/// ECS component that creates a wind source attached to an entity.
#[derive(Debug, Clone)]
pub struct WindSourceComponent {
    /// Type of wind source to create.
    pub source_type: WindSourceType,
    /// Wind source ID (assigned when registered with the system).
    pub source_id: Option<WindSourceId>,
    /// Whether the component is enabled.
    pub enabled: bool,
}

/// Type of wind source for the ECS component.
#[derive(Debug, Clone)]
pub enum WindSourceType {
    /// Directional wind.
    Directional { direction: Vec3, speed: f32 },
    /// Point source.
    Point { strength: f32, radius: f32 },
    /// Vortex source.
    Vortex { strength: f32, radius: f32, updraft: f32 },
}

impl WindSourceComponent {
    /// Create a directional wind source component.
    pub fn directional(direction: Vec3, speed: f32) -> Self {
        Self {
            source_type: WindSourceType::Directional { direction, speed },
            source_id: None,
            enabled: true,
        }
    }

    /// Create a point wind source component (fan).
    pub fn point(strength: f32, radius: f32) -> Self {
        Self {
            source_type: WindSourceType::Point { strength, radius },
            source_id: None,
            enabled: true,
        }
    }

    /// Create a vortex wind source component.
    pub fn vortex(strength: f32, radius: f32, updraft: f32) -> Self {
        Self {
            source_type: WindSourceType::Vortex {
                strength,
                radius,
                updraft,
            },
            source_id: None,
            enabled: true,
        }
    }
}
