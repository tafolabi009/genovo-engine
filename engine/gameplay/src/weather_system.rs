//! Dynamic weather system with state transitions, parameter interpolation,
//! and time-of-day integration.
//!
//! Provides a complete weather framework for open-world games:
//!
//! - **Weather states**: Clear, Cloudy, Overcast, LightRain, HeavyRain, Storm,
//!   Snow, Fog, Sandstorm -- each with characteristic visual and audio parameters
//! - **Smooth transitions**: weather changes are interpolated over configurable
//!   durations, blending all parameters continuously
//! - **Random weather changes**: optional system that schedules random weather
//!   transitions based on configurable probabilities
//! - **Time-of-day integration**: weather affects ambient light color and intensity,
//!   and time-of-day affects weather appearance (e.g., fog is denser at dawn)
//! - **Effect activation**: each weather state specifies which particle systems,
//!   post-processing effects, and audio sources should be active
//!
//! # Architecture
//!
//! The [`WeatherManager`] holds the current and target [`WeatherState`], and
//! interpolates [`WeatherParameters`] between them over the transition duration.
//! Each frame, `update(dt)` advances the interpolation and produces a
//! [`WeatherSnapshot`] containing the blended parameters that rendering,
//! audio, and gameplay systems can consume.
//!
//! The [`TimeOfDay`] system tracks the sun position and ambient light, and
//! the weather system modifies these values based on cloud density and
//! precipitation.

use glam::Vec3;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Weather state
// ---------------------------------------------------------------------------

/// Discrete weather states that the weather system can be in or transition between.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeatherState {
    /// Clear sky, full sunlight.
    Clear,
    /// Partially cloudy, some cloud coverage.
    Cloudy,
    /// Fully overcast, no direct sunlight.
    Overcast,
    /// Light rain with moderate cloud coverage.
    LightRain,
    /// Heavy rain with dark clouds.
    HeavyRain,
    /// Thunderstorm: heavy rain, lightning, strong wind.
    Storm,
    /// Snowfall with cold temperature.
    Snow,
    /// Dense fog reducing visibility.
    Fog,
    /// Desert sandstorm with extreme visibility reduction.
    Sandstorm,
}

impl WeatherState {
    /// All weather states.
    pub const ALL: &'static [WeatherState] = &[
        Self::Clear,
        Self::Cloudy,
        Self::Overcast,
        Self::LightRain,
        Self::HeavyRain,
        Self::Storm,
        Self::Snow,
        Self::Fog,
        Self::Sandstorm,
    ];

    /// Whether this state has precipitation.
    pub fn has_precipitation(&self) -> bool {
        matches!(
            self,
            Self::LightRain | Self::HeavyRain | Self::Storm | Self::Snow
        )
    }

    /// Whether this state has lightning.
    pub fn has_lightning(&self) -> bool {
        matches!(self, Self::Storm)
    }

    /// Whether this state has fog.
    pub fn has_fog(&self) -> bool {
        matches!(self, Self::Fog | Self::HeavyRain | Self::Storm)
    }

    /// Whether this state significantly reduces visibility.
    pub fn reduces_visibility(&self) -> bool {
        matches!(self, Self::Fog | Self::Storm | Self::Sandstorm | Self::Snow)
    }

    /// Get the default parameters for this weather state.
    pub fn default_parameters(&self) -> WeatherParameters {
        match self {
            Self::Clear => WeatherParameters {
                cloud_density: 0.1,
                cloud_coverage: 0.1,
                rain_intensity: 0.0,
                snow_intensity: 0.0,
                wind_speed: 2.0,
                wind_direction: Vec3::new(1.0, 0.0, 0.3),
                fog_density: 0.0,
                fog_start: 200.0,
                fog_end: 1000.0,
                fog_color: Vec3::new(0.7, 0.8, 0.9),
                lightning_frequency: 0.0,
                lightning_intensity: 0.0,
                temperature: 22.0,
                humidity: 0.3,
                ambient_light_multiplier: 1.0,
                ambient_light_color: Vec3::new(1.0, 1.0, 1.0),
                sun_intensity_multiplier: 1.0,
                sun_color_multiplier: Vec3::new(1.0, 0.98, 0.95),
                sky_exposure: 1.0,
                precipitation_color: Vec3::ZERO,
                sand_intensity: 0.0,
                wetness: 0.0,
                puddle_amount: 0.0,
                snow_coverage: 0.0,
            },
            Self::Cloudy => WeatherParameters {
                cloud_density: 0.5,
                cloud_coverage: 0.5,
                rain_intensity: 0.0,
                snow_intensity: 0.0,
                wind_speed: 5.0,
                wind_direction: Vec3::new(1.0, 0.0, 0.5),
                fog_density: 0.05,
                fog_start: 300.0,
                fog_end: 1500.0,
                fog_color: Vec3::new(0.75, 0.78, 0.82),
                lightning_frequency: 0.0,
                lightning_intensity: 0.0,
                temperature: 18.0,
                humidity: 0.5,
                ambient_light_multiplier: 0.8,
                ambient_light_color: Vec3::new(0.9, 0.92, 0.95),
                sun_intensity_multiplier: 0.7,
                sun_color_multiplier: Vec3::new(0.95, 0.95, 0.95),
                sky_exposure: 0.9,
                precipitation_color: Vec3::ZERO,
                sand_intensity: 0.0,
                wetness: 0.0,
                puddle_amount: 0.0,
                snow_coverage: 0.0,
            },
            Self::Overcast => WeatherParameters {
                cloud_density: 0.9,
                cloud_coverage: 0.9,
                rain_intensity: 0.0,
                snow_intensity: 0.0,
                wind_speed: 4.0,
                wind_direction: Vec3::new(0.8, 0.0, 0.6),
                fog_density: 0.1,
                fog_start: 150.0,
                fog_end: 800.0,
                fog_color: Vec3::new(0.6, 0.65, 0.7),
                lightning_frequency: 0.0,
                lightning_intensity: 0.0,
                temperature: 15.0,
                humidity: 0.6,
                ambient_light_multiplier: 0.5,
                ambient_light_color: Vec3::new(0.8, 0.82, 0.85),
                sun_intensity_multiplier: 0.3,
                sun_color_multiplier: Vec3::new(0.85, 0.85, 0.87),
                sky_exposure: 0.7,
                precipitation_color: Vec3::ZERO,
                sand_intensity: 0.0,
                wetness: 0.0,
                puddle_amount: 0.0,
                snow_coverage: 0.0,
            },
            Self::LightRain => WeatherParameters {
                cloud_density: 0.7,
                cloud_coverage: 0.75,
                rain_intensity: 0.3,
                snow_intensity: 0.0,
                wind_speed: 6.0,
                wind_direction: Vec3::new(0.7, -0.2, 0.7),
                fog_density: 0.15,
                fog_start: 100.0,
                fog_end: 600.0,
                fog_color: Vec3::new(0.55, 0.6, 0.65),
                lightning_frequency: 0.0,
                lightning_intensity: 0.0,
                temperature: 14.0,
                humidity: 0.75,
                ambient_light_multiplier: 0.6,
                ambient_light_color: Vec3::new(0.75, 0.78, 0.82),
                sun_intensity_multiplier: 0.4,
                sun_color_multiplier: Vec3::new(0.8, 0.8, 0.85),
                sky_exposure: 0.75,
                precipitation_color: Vec3::new(0.6, 0.65, 0.75),
                sand_intensity: 0.0,
                wetness: 0.3,
                puddle_amount: 0.1,
                snow_coverage: 0.0,
            },
            Self::HeavyRain => WeatherParameters {
                cloud_density: 0.95,
                cloud_coverage: 0.95,
                rain_intensity: 0.8,
                snow_intensity: 0.0,
                wind_speed: 12.0,
                wind_direction: Vec3::new(0.6, -0.3, 0.7),
                fog_density: 0.3,
                fog_start: 50.0,
                fog_end: 300.0,
                fog_color: Vec3::new(0.4, 0.45, 0.5),
                lightning_frequency: 0.0,
                lightning_intensity: 0.0,
                temperature: 12.0,
                humidity: 0.9,
                ambient_light_multiplier: 0.35,
                ambient_light_color: Vec3::new(0.6, 0.63, 0.7),
                sun_intensity_multiplier: 0.1,
                sun_color_multiplier: Vec3::new(0.7, 0.7, 0.75),
                sky_exposure: 0.5,
                precipitation_color: Vec3::new(0.5, 0.55, 0.65),
                sand_intensity: 0.0,
                wetness: 0.8,
                puddle_amount: 0.5,
                snow_coverage: 0.0,
            },
            Self::Storm => WeatherParameters {
                cloud_density: 1.0,
                cloud_coverage: 1.0,
                rain_intensity: 1.0,
                snow_intensity: 0.0,
                wind_speed: 20.0,
                wind_direction: Vec3::new(0.5, -0.4, 0.8),
                fog_density: 0.4,
                fog_start: 30.0,
                fog_end: 200.0,
                fog_color: Vec3::new(0.3, 0.33, 0.38),
                lightning_frequency: 0.15,
                lightning_intensity: 1.0,
                temperature: 10.0,
                humidity: 0.95,
                ambient_light_multiplier: 0.2,
                ambient_light_color: Vec3::new(0.5, 0.5, 0.6),
                sun_intensity_multiplier: 0.05,
                sun_color_multiplier: Vec3::new(0.6, 0.6, 0.65),
                sky_exposure: 0.35,
                precipitation_color: Vec3::new(0.4, 0.45, 0.55),
                sand_intensity: 0.0,
                wetness: 1.0,
                puddle_amount: 0.8,
                snow_coverage: 0.0,
            },
            Self::Snow => WeatherParameters {
                cloud_density: 0.8,
                cloud_coverage: 0.85,
                rain_intensity: 0.0,
                snow_intensity: 0.7,
                wind_speed: 8.0,
                wind_direction: Vec3::new(0.6, -0.1, 0.8),
                fog_density: 0.2,
                fog_start: 80.0,
                fog_end: 400.0,
                fog_color: Vec3::new(0.8, 0.85, 0.9),
                lightning_frequency: 0.0,
                lightning_intensity: 0.0,
                temperature: -5.0,
                humidity: 0.6,
                ambient_light_multiplier: 0.7,
                ambient_light_color: Vec3::new(0.85, 0.88, 0.95),
                sun_intensity_multiplier: 0.5,
                sun_color_multiplier: Vec3::new(0.9, 0.92, 1.0),
                sky_exposure: 0.8,
                precipitation_color: Vec3::new(0.9, 0.92, 0.97),
                sand_intensity: 0.0,
                wetness: 0.2,
                puddle_amount: 0.0,
                snow_coverage: 0.5,
            },
            Self::Fog => WeatherParameters {
                cloud_density: 0.6,
                cloud_coverage: 0.7,
                rain_intensity: 0.0,
                snow_intensity: 0.0,
                wind_speed: 1.0,
                wind_direction: Vec3::new(0.5, 0.0, 0.5),
                fog_density: 0.8,
                fog_start: 10.0,
                fog_end: 100.0,
                fog_color: Vec3::new(0.65, 0.7, 0.75),
                lightning_frequency: 0.0,
                lightning_intensity: 0.0,
                temperature: 10.0,
                humidity: 0.9,
                ambient_light_multiplier: 0.5,
                ambient_light_color: Vec3::new(0.75, 0.78, 0.82),
                sun_intensity_multiplier: 0.2,
                sun_color_multiplier: Vec3::new(0.8, 0.82, 0.85),
                sky_exposure: 0.6,
                precipitation_color: Vec3::ZERO,
                sand_intensity: 0.0,
                wetness: 0.4,
                puddle_amount: 0.1,
                snow_coverage: 0.0,
            },
            Self::Sandstorm => WeatherParameters {
                cloud_density: 0.3,
                cloud_coverage: 0.4,
                rain_intensity: 0.0,
                snow_intensity: 0.0,
                wind_speed: 25.0,
                wind_direction: Vec3::new(0.8, -0.1, 0.6),
                fog_density: 0.7,
                fog_start: 5.0,
                fog_end: 80.0,
                fog_color: Vec3::new(0.8, 0.65, 0.4),
                lightning_frequency: 0.0,
                lightning_intensity: 0.0,
                temperature: 35.0,
                humidity: 0.1,
                ambient_light_multiplier: 0.4,
                ambient_light_color: Vec3::new(0.9, 0.75, 0.5),
                sun_intensity_multiplier: 0.3,
                sun_color_multiplier: Vec3::new(1.0, 0.8, 0.5),
                sky_exposure: 0.5,
                precipitation_color: Vec3::ZERO,
                sand_intensity: 0.8,
                wetness: 0.0,
                puddle_amount: 0.0,
                snow_coverage: 0.0,
            },
        }
    }

    /// Human-readable name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Clear => "Clear",
            Self::Cloudy => "Cloudy",
            Self::Overcast => "Overcast",
            Self::LightRain => "Light Rain",
            Self::HeavyRain => "Heavy Rain",
            Self::Storm => "Thunderstorm",
            Self::Snow => "Snow",
            Self::Fog => "Fog",
            Self::Sandstorm => "Sandstorm",
        }
    }
}

impl Default for WeatherState {
    fn default() -> Self {
        Self::Clear
    }
}

// ---------------------------------------------------------------------------
// Weather parameters
// ---------------------------------------------------------------------------

/// Continuous parameters describing the weather at a point in time.
///
/// These parameters are interpolated between weather states during transitions.
/// Rendering, audio, and gameplay systems read these values each frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherParameters {
    /// Cloud density (0.0 = clear, 1.0 = fully covered).
    pub cloud_density: f32,
    /// Cloud coverage fraction.
    pub cloud_coverage: f32,
    /// Rain intensity (0.0 = none, 1.0 = heavy downpour).
    pub rain_intensity: f32,
    /// Snow intensity (0.0 = none, 1.0 = heavy snowfall).
    pub snow_intensity: f32,
    /// Wind speed in m/s.
    pub wind_speed: f32,
    /// Wind direction (normalized, XZ = horizontal, Y = vertical component).
    pub wind_direction: Vec3,
    /// Fog density (0.0 = none, 1.0 = extremely dense).
    pub fog_density: f32,
    /// Fog start distance (meters).
    pub fog_start: f32,
    /// Fog end distance (meters, fully fogged beyond this).
    pub fog_end: f32,
    /// Fog color.
    pub fog_color: Vec3,
    /// Lightning flash frequency (flashes per second, 0 = none).
    pub lightning_frequency: f32,
    /// Lightning intensity (0.0 to 1.0).
    pub lightning_intensity: f32,
    /// Ambient temperature in Celsius.
    pub temperature: f32,
    /// Humidity (0.0 to 1.0).
    pub humidity: f32,
    /// Ambient light intensity multiplier.
    pub ambient_light_multiplier: f32,
    /// Ambient light color tint.
    pub ambient_light_color: Vec3,
    /// Directional sun intensity multiplier.
    pub sun_intensity_multiplier: f32,
    /// Sun color multiplier.
    pub sun_color_multiplier: Vec3,
    /// Sky exposure (affects skybox brightness).
    pub sky_exposure: f32,
    /// Color tint for precipitation particles.
    pub precipitation_color: Vec3,
    /// Sand particle intensity (for sandstorm).
    pub sand_intensity: f32,
    /// Surface wetness (0.0 = dry, 1.0 = soaked).
    pub wetness: f32,
    /// Puddle coverage (0.0 = none, 1.0 = many puddles).
    pub puddle_amount: f32,
    /// Snow ground coverage (0.0 = none, 1.0 = fully covered).
    pub snow_coverage: f32,
}

impl WeatherParameters {
    /// Linearly interpolate between two parameter sets.
    pub fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let lerp_f32 = |a: f32, b: f32| a + (b - a) * t;
        let lerp_vec3 = |a: Vec3, b: Vec3| a + (b - a) * t;

        Self {
            cloud_density: lerp_f32(a.cloud_density, b.cloud_density),
            cloud_coverage: lerp_f32(a.cloud_coverage, b.cloud_coverage),
            rain_intensity: lerp_f32(a.rain_intensity, b.rain_intensity),
            snow_intensity: lerp_f32(a.snow_intensity, b.snow_intensity),
            wind_speed: lerp_f32(a.wind_speed, b.wind_speed),
            wind_direction: lerp_vec3(a.wind_direction, b.wind_direction).normalize_or_zero(),
            fog_density: lerp_f32(a.fog_density, b.fog_density),
            fog_start: lerp_f32(a.fog_start, b.fog_start),
            fog_end: lerp_f32(a.fog_end, b.fog_end),
            fog_color: lerp_vec3(a.fog_color, b.fog_color),
            lightning_frequency: lerp_f32(a.lightning_frequency, b.lightning_frequency),
            lightning_intensity: lerp_f32(a.lightning_intensity, b.lightning_intensity),
            temperature: lerp_f32(a.temperature, b.temperature),
            humidity: lerp_f32(a.humidity, b.humidity),
            ambient_light_multiplier: lerp_f32(
                a.ambient_light_multiplier,
                b.ambient_light_multiplier,
            ),
            ambient_light_color: lerp_vec3(a.ambient_light_color, b.ambient_light_color),
            sun_intensity_multiplier: lerp_f32(
                a.sun_intensity_multiplier,
                b.sun_intensity_multiplier,
            ),
            sun_color_multiplier: lerp_vec3(a.sun_color_multiplier, b.sun_color_multiplier),
            sky_exposure: lerp_f32(a.sky_exposure, b.sky_exposure),
            precipitation_color: lerp_vec3(a.precipitation_color, b.precipitation_color),
            sand_intensity: lerp_f32(a.sand_intensity, b.sand_intensity),
            wetness: lerp_f32(a.wetness, b.wetness),
            puddle_amount: lerp_f32(a.puddle_amount, b.puddle_amount),
            snow_coverage: lerp_f32(a.snow_coverage, b.snow_coverage),
        }
    }

    /// Smoothstep interpolation (smoother transitions).
    pub fn smoothstep(a: &Self, b: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let smooth_t = t * t * (3.0 - 2.0 * t);
        Self::lerp(a, b, smooth_t)
    }

    /// Get the wind vector (direction * speed).
    pub fn wind_vector(&self) -> Vec3 {
        self.wind_direction.normalize_or_zero() * self.wind_speed
    }

    /// Effective visibility distance based on fog settings.
    pub fn visibility_distance(&self) -> f32 {
        if self.fog_density > 0.01 {
            self.fog_end / (1.0 + self.fog_density * 2.0)
        } else {
            f32::INFINITY
        }
    }
}

impl Default for WeatherParameters {
    fn default() -> Self {
        WeatherState::Clear.default_parameters()
    }
}

// ---------------------------------------------------------------------------
// Weather effects
// ---------------------------------------------------------------------------

/// Specifies which visual/audio effects should be active for the current weather.
#[derive(Debug, Clone, Default)]
pub struct WeatherEffects {
    /// Whether to spawn rain particles.
    pub rain_particles: bool,
    /// Whether to spawn snow particles.
    pub snow_particles: bool,
    /// Whether to spawn sand particles (sandstorm).
    pub sand_particles: bool,
    /// Rain particle spawn rate multiplier.
    pub rain_particle_rate: f32,
    /// Snow particle spawn rate multiplier.
    pub snow_particle_rate: f32,
    /// Sand particle spawn rate multiplier.
    pub sand_particle_rate: f32,
    /// Whether to apply fog post-processing.
    pub fog_post_process: bool,
    /// Whether to apply wet surface shader.
    pub wet_surfaces: bool,
    /// Whether to render puddle reflections.
    pub puddle_reflections: bool,
    /// Whether to render snow on surfaces.
    pub snow_on_surfaces: bool,
    /// Whether to play rain audio.
    pub rain_audio: bool,
    /// Whether to play wind audio.
    pub wind_audio: bool,
    /// Whether to play thunder audio.
    pub thunder_audio: bool,
    /// Whether to play snow ambient audio.
    pub snow_audio: bool,
    /// Whether to play sandstorm audio.
    pub sand_audio: bool,
    /// Whether lightning flash effects are active.
    pub lightning_flash: bool,
    /// Rain audio volume.
    pub rain_volume: f32,
    /// Wind audio volume.
    pub wind_volume: f32,
    /// Thunder volume.
    pub thunder_volume: f32,
}

impl WeatherEffects {
    /// Compute effects from weather parameters.
    pub fn from_parameters(params: &WeatherParameters) -> Self {
        Self {
            rain_particles: params.rain_intensity > 0.05,
            snow_particles: params.snow_intensity > 0.05,
            sand_particles: params.sand_intensity > 0.05,
            rain_particle_rate: params.rain_intensity,
            snow_particle_rate: params.snow_intensity,
            sand_particle_rate: params.sand_intensity,
            fog_post_process: params.fog_density > 0.05,
            wet_surfaces: params.wetness > 0.1,
            puddle_reflections: params.puddle_amount > 0.1,
            snow_on_surfaces: params.snow_coverage > 0.1,
            rain_audio: params.rain_intensity > 0.1,
            wind_audio: params.wind_speed > 5.0,
            thunder_audio: params.lightning_frequency > 0.01,
            snow_audio: params.snow_intensity > 0.1,
            sand_audio: params.sand_intensity > 0.1,
            lightning_flash: params.lightning_intensity > 0.1,
            rain_volume: params.rain_intensity,
            wind_volume: (params.wind_speed / 25.0).min(1.0),
            thunder_volume: params.lightning_intensity,
        }
    }
}

// ---------------------------------------------------------------------------
// Weather transition
// ---------------------------------------------------------------------------

/// Tracks the current weather transition state.
#[derive(Debug, Clone)]
pub struct WeatherTransition {
    /// Source weather state.
    pub from_state: WeatherState,
    /// Target weather state.
    pub to_state: WeatherState,
    /// Source parameters (captured at transition start).
    pub from_params: WeatherParameters,
    /// Target parameters.
    pub to_params: WeatherParameters,
    /// Total transition duration (seconds).
    pub duration: f32,
    /// Elapsed time since transition started (seconds).
    pub elapsed: f32,
    /// Whether a transition is currently in progress.
    pub active: bool,
}

impl WeatherTransition {
    /// Create an inactive (completed) transition.
    pub fn inactive() -> Self {
        Self {
            from_state: WeatherState::Clear,
            to_state: WeatherState::Clear,
            from_params: WeatherState::Clear.default_parameters(),
            to_params: WeatherState::Clear.default_parameters(),
            duration: 1.0,
            elapsed: 1.0,
            active: false,
        }
    }

    /// Start a new transition.
    pub fn start(
        from_state: WeatherState,
        to_state: WeatherState,
        current_params: WeatherParameters,
        duration: f32,
    ) -> Self {
        Self {
            from_state,
            to_state,
            from_params: current_params,
            to_params: to_state.default_parameters(),
            duration: duration.max(0.1),
            elapsed: 0.0,
            active: true,
        }
    }

    /// Advance the transition by dt seconds.
    pub fn advance(&mut self, dt: f32) {
        if !self.active {
            return;
        }
        self.elapsed += dt;
        if self.elapsed >= self.duration {
            self.elapsed = self.duration;
            self.active = false;
        }
    }

    /// Interpolation factor (0.0 = from, 1.0 = to).
    pub fn factor(&self) -> f32 {
        if self.duration <= 0.0 {
            return 1.0;
        }
        (self.elapsed / self.duration).clamp(0.0, 1.0)
    }

    /// Get the interpolated parameters.
    pub fn interpolate(&self) -> WeatherParameters {
        WeatherParameters::smoothstep(&self.from_params, &self.to_params, self.factor())
    }

    /// Whether the transition is complete.
    pub fn is_complete(&self) -> bool {
        !self.active
    }
}

impl Default for WeatherTransition {
    fn default() -> Self {
        Self::inactive()
    }
}

// ---------------------------------------------------------------------------
// Time of day
// ---------------------------------------------------------------------------

/// Time-of-day system tracking sun position and ambient light.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeOfDay {
    /// Current time as fraction of day (0.0 = midnight, 0.5 = noon, 1.0 = midnight).
    pub time: f32,
    /// Day-night cycle duration in real seconds (0 = static time).
    pub cycle_duration: f32,
    /// Sun direction (normalized, computed from time).
    pub sun_direction: Vec3,
    /// Sun color at current time.
    pub sun_color: Vec3,
    /// Sun intensity at current time.
    pub sun_intensity: f32,
    /// Ambient sky color at current time.
    pub ambient_color: Vec3,
    /// Whether it is considered "night" for gameplay purposes.
    pub is_night: bool,
    /// Sunrise time (fraction of day, e.g., 0.25 = 6 AM).
    pub sunrise_time: f32,
    /// Sunset time (fraction of day, e.g., 0.75 = 6 PM).
    pub sunset_time: f32,
    /// Moon direction.
    pub moon_direction: Vec3,
    /// Moon intensity.
    pub moon_intensity: f32,
}

impl TimeOfDay {
    /// Create a new time-of-day system.
    pub fn new() -> Self {
        let mut tod = Self {
            time: 0.4, // 9:36 AM
            cycle_duration: 1200.0, // 20 minutes per day
            sun_direction: Vec3::Y,
            sun_color: Vec3::ONE,
            sun_intensity: 1.0,
            ambient_color: Vec3::new(0.4, 0.5, 0.7),
            is_night: false,
            sunrise_time: 0.25,
            sunset_time: 0.75,
            moon_direction: Vec3::NEG_Y,
            moon_intensity: 0.1,
        };
        tod.compute_sun_position();
        tod
    }

    /// Update time of day.
    pub fn update(&mut self, dt: f32) {
        if self.cycle_duration > 0.0 {
            self.time += dt / self.cycle_duration;
            if self.time >= 1.0 {
                self.time -= 1.0;
            }
        }
        self.compute_sun_position();
        self.compute_lighting();
    }

    /// Set the time directly (0.0 to 1.0).
    pub fn set_time(&mut self, time: f32) {
        self.time = time.rem_euclid(1.0);
        self.compute_sun_position();
        self.compute_lighting();
    }

    /// Set the time from 24-hour clock (0.0 to 24.0).
    pub fn set_time_24h(&mut self, hours: f32) {
        self.set_time(hours / 24.0);
    }

    /// Get the current time as 24-hour clock.
    pub fn time_24h(&self) -> f32 {
        self.time * 24.0
    }

    /// Get a formatted time string (e.g., "14:30").
    pub fn time_string(&self) -> String {
        let total_minutes = (self.time * 24.0 * 60.0) as u32;
        let hours = total_minutes / 60;
        let minutes = total_minutes % 60;
        format!("{:02}:{:02}", hours, minutes)
    }

    /// Compute sun direction from time of day.
    fn compute_sun_position(&mut self) {
        // Sun arc: rises in the east (negative X), peaks at noon (positive Y),
        // sets in the west (positive X).
        let sun_angle = (self.time - 0.25) * std::f32::consts::TAU;
        let sun_y = sun_angle.sin();
        let sun_x = sun_angle.cos();
        self.sun_direction = Vec3::new(sun_x, sun_y, 0.3).normalize();

        // Moon is opposite the sun.
        self.moon_direction = -self.sun_direction;

        // Night detection.
        self.is_night = self.time < self.sunrise_time || self.time > self.sunset_time;
    }

    /// Compute lighting parameters from sun position.
    fn compute_lighting(&mut self) {
        let sun_height = self.sun_direction.y;

        if sun_height > 0.1 {
            // Daytime.
            self.sun_intensity = sun_height.min(1.0);
            self.sun_color = Vec3::new(1.0, 0.95 + sun_height * 0.05, 0.9 + sun_height * 0.1);
            self.ambient_color = Vec3::new(0.4, 0.5, 0.7) * (0.3 + sun_height * 0.7);
            self.moon_intensity = 0.0;
        } else if sun_height > -0.1 {
            // Sunrise/sunset.
            let t = (sun_height + 0.1) / 0.2;
            self.sun_intensity = t * 0.6;
            self.sun_color = Vec3::new(1.0, 0.5 + t * 0.45, 0.2 + t * 0.7);
            self.ambient_color = Vec3::new(0.4, 0.3, 0.4) * (0.1 + t * 0.5);
            self.moon_intensity = (1.0 - t) * 0.15;
        } else {
            // Nighttime.
            self.sun_intensity = 0.0;
            self.sun_color = Vec3::ZERO;
            self.ambient_color = Vec3::new(0.05, 0.07, 0.15);
            self.moon_intensity = 0.15;
        }
    }

    /// Apply weather modifiers to the time-of-day lighting.
    pub fn apply_weather(&self, params: &WeatherParameters) -> (Vec3, f32, Vec3) {
        let modified_sun_color = self.sun_color * params.sun_color_multiplier;
        let modified_sun_intensity = self.sun_intensity * params.sun_intensity_multiplier;
        let modified_ambient = self.ambient_color * params.ambient_light_color * params.ambient_light_multiplier;
        (modified_sun_color, modified_sun_intensity, modified_ambient)
    }
}

impl Default for TimeOfDay {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Weather snapshot
// ---------------------------------------------------------------------------

/// Complete weather state snapshot for the current frame.
///
/// This is the output of the weather system, consumed by rendering, audio,
/// and gameplay systems.
#[derive(Debug, Clone)]
pub struct WeatherSnapshot {
    /// Current discrete weather state.
    pub state: WeatherState,
    /// Target weather state (same as state if not transitioning).
    pub target_state: WeatherState,
    /// Whether a transition is in progress.
    pub transitioning: bool,
    /// Transition progress (0.0 to 1.0).
    pub transition_progress: f32,
    /// Interpolated weather parameters.
    pub parameters: WeatherParameters,
    /// Active weather effects.
    pub effects: WeatherEffects,
    /// Time of day data.
    pub time_of_day: TimeOfDay,
    /// Modified sun color (after weather).
    pub sun_color: Vec3,
    /// Modified sun intensity (after weather).
    pub sun_intensity: f32,
    /// Modified ambient color (after weather).
    pub ambient_color: Vec3,
}

// ---------------------------------------------------------------------------
// Weather random scheduler
// ---------------------------------------------------------------------------

/// Configuration for random weather changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherSchedulerConfig {
    /// Whether random weather changes are enabled.
    pub enabled: bool,
    /// Minimum time between weather changes (seconds).
    pub min_interval: f32,
    /// Maximum time between weather changes (seconds).
    pub max_interval: f32,
    /// Transition duration for random changes (seconds).
    pub transition_duration: f32,
    /// Probabilities for each weather state (indexed by WeatherState ordinal).
    /// Values are relative weights (don't need to sum to 1).
    pub probabilities: Vec<f32>,
}

impl Default for WeatherSchedulerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_interval: 120.0,
            max_interval: 600.0,
            transition_duration: 30.0,
            probabilities: vec![
                3.0, // Clear
                2.0, // Cloudy
                1.5, // Overcast
                1.0, // LightRain
                0.5, // HeavyRain
                0.3, // Storm
                0.5, // Snow
                0.5, // Fog
                0.2, // Sandstorm
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Weather manager
// ---------------------------------------------------------------------------

/// Main weather manager that controls weather state, transitions, and
/// time of day.
///
/// # Example
///
/// ```ignore
/// let mut weather = WeatherManager::new();
/// weather.set_weather(WeatherState::LightRain, 30.0);
///
/// // Each frame:
/// let snapshot = weather.update(dt);
/// renderer.set_fog(snapshot.parameters.fog_density, snapshot.parameters.fog_color);
/// audio.set_rain_volume(snapshot.effects.rain_volume);
/// ```
pub struct WeatherManager {
    /// Current weather state.
    pub current_state: WeatherState,
    /// Current weather parameters (may be mid-transition).
    pub current_params: WeatherParameters,
    /// Active transition.
    transition: WeatherTransition,
    /// Time of day system.
    pub time_of_day: TimeOfDay,
    /// Random weather scheduler config.
    pub scheduler: WeatherSchedulerConfig,
    /// Timer for next random weather change.
    next_change_timer: f32,
    /// Pseudo-random seed for weather scheduling.
    random_seed: u32,
    /// Whether the weather system is paused.
    pub paused: bool,
    /// Global weather intensity multiplier (for difficulty scaling).
    pub intensity_multiplier: f32,
}

impl WeatherManager {
    /// Create a new weather manager starting with clear weather.
    pub fn new() -> Self {
        Self {
            current_state: WeatherState::Clear,
            current_params: WeatherState::Clear.default_parameters(),
            transition: WeatherTransition::inactive(),
            time_of_day: TimeOfDay::new(),
            scheduler: WeatherSchedulerConfig::default(),
            next_change_timer: 300.0,
            random_seed: 12345,
            paused: false,
            intensity_multiplier: 1.0,
        }
    }

    /// Start a smooth transition to a new weather state.
    pub fn set_weather(&mut self, state: WeatherState, transition_time: f32) {
        if state == self.current_state && !self.transition.active {
            return;
        }

        self.transition = WeatherTransition::start(
            self.current_state,
            state,
            self.current_params.clone(),
            transition_time,
        );

        log::trace!(
            "Weather transition: {:?} -> {:?} over {:.1}s",
            self.current_state,
            state,
            transition_time,
        );
    }

    /// Instantly set the weather without transition.
    pub fn set_weather_instant(&mut self, state: WeatherState) {
        self.current_state = state;
        self.current_params = state.default_parameters();
        self.transition = WeatherTransition::inactive();
    }

    /// Update the weather system. Returns a snapshot of the current state.
    pub fn update(&mut self, dt: f32) -> WeatherSnapshot {
        if self.paused {
            return self.create_snapshot();
        }

        // Update time of day.
        self.time_of_day.update(dt);

        // Advance transition.
        if self.transition.active {
            self.transition.advance(dt);
            self.current_params = self.transition.interpolate();

            // Apply intensity multiplier.
            self.current_params.rain_intensity *= self.intensity_multiplier;
            self.current_params.snow_intensity *= self.intensity_multiplier;
            self.current_params.lightning_intensity *= self.intensity_multiplier;
            self.current_params.sand_intensity *= self.intensity_multiplier;

            if self.transition.is_complete() {
                self.current_state = self.transition.to_state;
                log::trace!(
                    "Weather transition complete: now {:?}",
                    self.current_state
                );
            }
        }

        // Random weather scheduling.
        if self.scheduler.enabled {
            self.next_change_timer -= dt;
            if self.next_change_timer <= 0.0 {
                self.trigger_random_weather();
            }
        }

        self.create_snapshot()
    }

    /// Create a snapshot of the current weather state.
    fn create_snapshot(&self) -> WeatherSnapshot {
        let effects = WeatherEffects::from_parameters(&self.current_params);
        let (sun_color, sun_intensity, ambient_color) =
            self.time_of_day.apply_weather(&self.current_params);

        WeatherSnapshot {
            state: self.current_state,
            target_state: if self.transition.active {
                self.transition.to_state
            } else {
                self.current_state
            },
            transitioning: self.transition.active,
            transition_progress: self.transition.factor(),
            parameters: self.current_params.clone(),
            effects,
            time_of_day: self.time_of_day.clone(),
            sun_color,
            sun_intensity,
            ambient_color,
        }
    }

    /// Trigger a random weather change based on configured probabilities.
    fn trigger_random_weather(&mut self) {
        let states = WeatherState::ALL;
        let probs = &self.scheduler.probabilities;

        // Compute total weight.
        let total_weight: f32 = probs.iter().take(states.len()).sum();
        if total_weight <= 0.0 {
            return;
        }

        // Pick a random state.
        let r = self.next_random() * total_weight;
        let mut cumulative = 0.0;
        let mut chosen = WeatherState::Clear;
        for (i, &weight) in probs.iter().enumerate().take(states.len()) {
            cumulative += weight;
            if r <= cumulative {
                chosen = states[i];
                break;
            }
        }

        // Avoid transitioning to the same state.
        if chosen != self.current_state {
            self.set_weather(chosen, self.scheduler.transition_duration);
        }

        // Schedule next change.
        let range = self.scheduler.max_interval - self.scheduler.min_interval;
        self.next_change_timer = self.scheduler.min_interval + self.next_random() * range;
    }

    /// Simple pseudo-random number generator (0.0 to 1.0).
    fn next_random(&mut self) -> f32 {
        self.random_seed = self.random_seed.wrapping_mul(1103515245).wrapping_add(12345);
        let bits = (self.random_seed >> 16) & 0x7FFF;
        bits as f32 / 32768.0
    }

    /// Get the current weather state.
    pub fn current_state(&self) -> WeatherState {
        self.current_state
    }

    /// Get the current weather parameters.
    pub fn current_params(&self) -> &WeatherParameters {
        &self.current_params
    }

    /// Whether a transition is in progress.
    pub fn is_transitioning(&self) -> bool {
        self.transition.active
    }

    /// Get the transition progress (0.0 to 1.0), 1.0 if not transitioning.
    pub fn transition_progress(&self) -> f32 {
        self.transition.factor()
    }

    /// Get the current wind vector.
    pub fn wind_vector(&self) -> Vec3 {
        self.current_params.wind_vector()
    }

    /// Get the current visibility distance.
    pub fn visibility(&self) -> f32 {
        self.current_params.visibility_distance()
    }

    /// Get the current temperature.
    pub fn temperature(&self) -> f32 {
        self.current_params.temperature
    }

    /// Pause/unpause the weather system.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Enable random weather with default configuration.
    pub fn enable_random_weather(&mut self) {
        self.scheduler.enabled = true;
    }

    /// Disable random weather.
    pub fn disable_random_weather(&mut self) {
        self.scheduler.enabled = false;
    }

    /// Set the time-of-day cycle duration.
    pub fn set_day_cycle_duration(&mut self, seconds: f32) {
        self.time_of_day.cycle_duration = seconds;
    }

    /// Set the global weather intensity multiplier.
    pub fn set_intensity_multiplier(&mut self, multiplier: f32) {
        self.intensity_multiplier = multiplier.max(0.0);
    }
}

impl Default for WeatherManager {
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
    fn weather_state_defaults() {
        for state in WeatherState::ALL {
            let params = state.default_parameters();
            assert!(params.wind_speed >= 0.0);
            assert!(params.cloud_density >= 0.0 && params.cloud_density <= 1.0);
        }
    }

    #[test]
    fn weather_state_properties() {
        assert!(WeatherState::Storm.has_precipitation());
        assert!(WeatherState::Storm.has_lightning());
        assert!(WeatherState::Fog.has_fog());
        assert!(!WeatherState::Clear.has_precipitation());
    }

    #[test]
    fn parameter_interpolation() {
        let a = WeatherState::Clear.default_parameters();
        let b = WeatherState::Storm.default_parameters();

        let mid = WeatherParameters::lerp(&a, &b, 0.5);
        assert!(mid.cloud_density > a.cloud_density);
        assert!(mid.cloud_density < b.cloud_density);
        assert!(mid.rain_intensity > 0.0);
    }

    #[test]
    fn parameter_interpolation_endpoints() {
        let a = WeatherState::Clear.default_parameters();
        let b = WeatherState::Storm.default_parameters();

        let start = WeatherParameters::lerp(&a, &b, 0.0);
        let end = WeatherParameters::lerp(&a, &b, 1.0);

        assert!((start.cloud_density - a.cloud_density).abs() < 0.001);
        assert!((end.cloud_density - b.cloud_density).abs() < 0.001);
    }

    #[test]
    fn smoothstep_interpolation() {
        let a = WeatherState::Clear.default_parameters();
        let b = WeatherState::HeavyRain.default_parameters();

        let smooth = WeatherParameters::smoothstep(&a, &b, 0.5);
        assert!(smooth.rain_intensity > 0.0);
    }

    #[test]
    fn weather_transition() {
        let mut transition = WeatherTransition::start(
            WeatherState::Clear,
            WeatherState::Storm,
            WeatherState::Clear.default_parameters(),
            10.0,
        );

        assert!(transition.active);
        assert!((transition.factor() - 0.0).abs() < 0.01);

        transition.advance(5.0);
        assert!((transition.factor() - 0.5).abs() < 0.01);
        assert!(transition.active);

        transition.advance(5.0);
        assert!((transition.factor() - 1.0).abs() < 0.01);
        assert!(!transition.active);
    }

    #[test]
    fn weather_manager_instant() {
        let mut manager = WeatherManager::new();
        manager.set_weather_instant(WeatherState::Snow);

        assert_eq!(manager.current_state(), WeatherState::Snow);
        assert!(manager.current_params().snow_intensity > 0.0);
    }

    #[test]
    fn weather_manager_transition() {
        let mut manager = WeatherManager::new();
        manager.set_weather(WeatherState::HeavyRain, 10.0);

        assert!(manager.is_transitioning());

        // Advance to completion.
        for _ in 0..600 {
            manager.update(1.0 / 60.0);
        }

        assert!(!manager.is_transitioning());
        assert_eq!(manager.current_state(), WeatherState::HeavyRain);
    }

    #[test]
    fn weather_effects_from_params() {
        let storm = WeatherState::Storm.default_parameters();
        let effects = WeatherEffects::from_parameters(&storm);

        assert!(effects.rain_particles);
        assert!(effects.rain_audio);
        assert!(effects.thunder_audio);
        assert!(effects.lightning_flash);
        assert!(effects.fog_post_process);
        assert!(effects.wet_surfaces);
    }

    #[test]
    fn weather_effects_clear() {
        let clear = WeatherState::Clear.default_parameters();
        let effects = WeatherEffects::from_parameters(&clear);

        assert!(!effects.rain_particles);
        assert!(!effects.snow_particles);
        assert!(!effects.thunder_audio);
    }

    #[test]
    fn time_of_day_noon() {
        let mut tod = TimeOfDay::new();
        tod.set_time(0.5); // Noon
        assert!(tod.sun_direction.y > 0.9);
        assert!(tod.sun_intensity > 0.8);
        assert!(!tod.is_night);
    }

    #[test]
    fn time_of_day_midnight() {
        let mut tod = TimeOfDay::new();
        tod.set_time(0.0); // Midnight
        assert!(tod.sun_direction.y < 0.0);
        assert!(tod.is_night);
    }

    #[test]
    fn time_of_day_format() {
        let mut tod = TimeOfDay::new();
        tod.set_time_24h(14.5);
        let s = tod.time_string();
        assert_eq!(s, "14:30");
    }

    #[test]
    fn time_of_day_cycle() {
        let mut tod = TimeOfDay::new();
        tod.cycle_duration = 100.0;
        tod.set_time(0.9);
        tod.update(15.0); // 15/100 = 0.15, so 0.9 + 0.15 = 1.05 -> 0.05
        assert!(tod.time < 0.1);
    }

    #[test]
    fn weather_with_time_of_day() {
        let tod = TimeOfDay::new();
        let params = WeatherState::Clear.default_parameters();
        let (sun_color, sun_intensity, ambient) = tod.apply_weather(&params);

        assert!(sun_color.length() > 0.0);
        assert!(sun_intensity >= 0.0);
        assert!(ambient.length() > 0.0);
    }

    #[test]
    fn weather_visibility() {
        let clear = WeatherState::Clear.default_parameters();
        let fog = WeatherState::Fog.default_parameters();

        let clear_vis = clear.visibility_distance();
        let fog_vis = fog.visibility_distance();

        assert!(fog_vis < clear_vis);
    }

    #[test]
    fn weather_wind_vector() {
        let storm = WeatherState::Storm.default_parameters();
        let wind = storm.wind_vector();
        assert!(wind.length() > 10.0);
    }

    #[test]
    fn weather_manager_snapshot() {
        let mut manager = WeatherManager::new();
        let snapshot = manager.update(1.0 / 60.0);

        assert_eq!(snapshot.state, WeatherState::Clear);
        assert!(!snapshot.transitioning);
    }

    #[test]
    fn weather_scheduler_config_default() {
        let config = WeatherSchedulerConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.probabilities.len(), 9);
    }

    #[test]
    fn weather_display_names() {
        assert_eq!(WeatherState::Clear.display_name(), "Clear");
        assert_eq!(WeatherState::Storm.display_name(), "Thunderstorm");
        assert_eq!(WeatherState::Sandstorm.display_name(), "Sandstorm");
    }
}
