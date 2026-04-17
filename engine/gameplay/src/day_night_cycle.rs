//! Day/night cycle system with sky, lighting, and gameplay effects.
//!
//! Provides:
//! - **Configurable day length**: real-time to game-time ratio
//! - **Sun position**: calculated from game time for directional lighting
//! - **Moon phases**: 8-phase lunar cycle with configurable period
//! - **Sky color transitions**: sunrise/sunset gradients with configurable palettes
//! - **Ambient light changes**: dynamic ambient light intensity and color
//! - **NPC schedules tied to time**: NPCs change behavior based on time of day
//! - **Torch/lamp activation**: lights turn on/off at configurable times
//! - **Visibility reduction at night**: affects gameplay (stealth, detection range)
//! - **Stars and celestial objects**: star visibility based on sun position
//! - **ECS integration**: `DayNightComponent`, `DayNightSystem`
//!
//! # Coordinate system
//!
//! The sun direction is computed using a simple latitude-based model. The sun
//! rises in +X, reaches zenith at +Y, and sets in -X. Time is stored as
//! a fractional day (0.0 = midnight, 0.25 = 6 AM, 0.5 = noon, 0.75 = 6 PM).

use glam::Vec3;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default real-time seconds per in-game day.
pub const DEFAULT_DAY_LENGTH: f32 = 1200.0; // 20 real minutes = 1 game day
/// Default sunrise time (fraction of day: 0.25 = 6:00 AM).
pub const DEFAULT_SUNRISE_TIME: f32 = 0.25;
/// Default sunset time (0.75 = 6:00 PM).
pub const DEFAULT_SUNSET_TIME: f32 = 0.75;
/// Duration of twilight transition (fraction of day).
pub const TWILIGHT_DURATION: f32 = 0.04; // ~1 hour of game time
/// Maximum ambient light intensity (daytime).
pub const MAX_AMBIENT_INTENSITY: f32 = 1.0;
/// Minimum ambient light intensity (nighttime).
pub const MIN_AMBIENT_INTENSITY: f32 = 0.1;
/// Maximum sun intensity.
pub const MAX_SUN_INTENSITY: f32 = 1.0;
/// Minimum sun intensity (below horizon).
pub const MIN_SUN_INTENSITY: f32 = 0.0;
/// Moon cycle length in game days.
pub const DEFAULT_MOON_CYCLE: f32 = 28.0;
/// Number of moon phases.
pub const MOON_PHASE_COUNT: usize = 8;
/// Night visibility multiplier (affects detection range etc.).
pub const NIGHT_VISIBILITY_MULTIPLIER: f32 = 0.4;
/// Pi constant.
const PI: f32 = std::f32::consts::PI;
/// Two PI.
const TWO_PI: f32 = PI * 2.0;
/// Small epsilon.
const EPSILON: f32 = 1e-6;
/// Default latitude for sun angle calculation (degrees).
pub const DEFAULT_LATITUDE: f32 = 45.0;
/// Light activation threshold (ambient intensity below this -> activate lights).
pub const LIGHT_ACTIVATION_THRESHOLD: f32 = 0.35;
/// Star visibility threshold (ambient intensity below this -> show stars).
pub const STAR_VISIBILITY_THRESHOLD: f32 = 0.25;

// ---------------------------------------------------------------------------
// TimePeriod
// ---------------------------------------------------------------------------

/// Named time period of the day.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimePeriod {
    /// Night (midnight to dawn).
    Night,
    /// Dawn/sunrise transition.
    Dawn,
    /// Morning.
    Morning,
    /// Midday.
    Midday,
    /// Afternoon.
    Afternoon,
    /// Dusk/sunset transition.
    Dusk,
    /// Evening.
    Evening,
    /// Late night.
    LateNight,
}

impl TimePeriod {
    /// Get a human-readable name for this period.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Night => "Night",
            Self::Dawn => "Dawn",
            Self::Morning => "Morning",
            Self::Midday => "Midday",
            Self::Afternoon => "Afternoon",
            Self::Dusk => "Dusk",
            Self::Evening => "Evening",
            Self::LateNight => "Late Night",
        }
    }

    /// Determine the time period from a fractional day value.
    pub fn from_day_fraction(t: f32) -> Self {
        let t = t.rem_euclid(1.0);
        match t {
            x if x < 0.20 => Self::LateNight,
            x if x < 0.27 => Self::Dawn,
            x if x < 0.40 => Self::Morning,
            x if x < 0.55 => Self::Midday,
            x if x < 0.70 => Self::Afternoon,
            x if x < 0.77 => Self::Dusk,
            x if x < 0.88 => Self::Evening,
            _ => Self::Night,
        }
    }
}

// ---------------------------------------------------------------------------
// MoonPhase
// ---------------------------------------------------------------------------

/// Phase of the moon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoonPhase {
    NewMoon,
    WaxingCrescent,
    FirstQuarter,
    WaxingGibbous,
    FullMoon,
    WaningGibbous,
    LastQuarter,
    WaningCrescent,
}

impl MoonPhase {
    /// Get the moon phase from a fractional cycle (0..1).
    pub fn from_cycle_fraction(t: f32) -> Self {
        let t = t.rem_euclid(1.0);
        let phase_index = (t * MOON_PHASE_COUNT as f32) as usize % MOON_PHASE_COUNT;
        match phase_index {
            0 => Self::NewMoon,
            1 => Self::WaxingCrescent,
            2 => Self::FirstQuarter,
            3 => Self::WaxingGibbous,
            4 => Self::FullMoon,
            5 => Self::WaningGibbous,
            6 => Self::LastQuarter,
            7 => Self::WaningCrescent,
            _ => Self::NewMoon,
        }
    }

    /// Get the moon brightness (0..1).
    pub fn brightness(&self) -> f32 {
        match self {
            Self::NewMoon => 0.0,
            Self::WaxingCrescent => 0.15,
            Self::FirstQuarter => 0.3,
            Self::WaxingGibbous => 0.6,
            Self::FullMoon => 1.0,
            Self::WaningGibbous => 0.6,
            Self::LastQuarter => 0.3,
            Self::WaningCrescent => 0.15,
        }
    }

    /// Get a display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::NewMoon => "New Moon",
            Self::WaxingCrescent => "Waxing Crescent",
            Self::FirstQuarter => "First Quarter",
            Self::WaxingGibbous => "Waxing Gibbous",
            Self::FullMoon => "Full Moon",
            Self::WaningGibbous => "Waning Gibbous",
            Self::LastQuarter => "Last Quarter",
            Self::WaningCrescent => "Waning Crescent",
        }
    }

    /// Get the illumination fraction (0..1).
    pub fn illumination(&self) -> f32 {
        self.brightness()
    }
}

// ---------------------------------------------------------------------------
// SkyColor
// ---------------------------------------------------------------------------

/// RGB color for sky rendering.
#[derive(Debug, Clone, Copy)]
pub struct SkyColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl SkyColor {
    /// Create a new sky color.
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// Linearly interpolate between two colors.
    pub fn lerp(a: SkyColor, b: SkyColor, t: f32) -> SkyColor {
        let t = t.clamp(0.0, 1.0);
        SkyColor {
            r: a.r + (b.r - a.r) * t,
            g: a.g + (b.g - a.g) * t,
            b: a.b + (b.b - a.b) * t,
        }
    }

    /// Convert to Vec3 (for shader uniforms).
    pub fn to_vec3(&self) -> Vec3 {
        Vec3::new(self.r, self.g, self.b)
    }

    /// Multiply by a scalar (intensity).
    pub fn scale(&self, s: f32) -> SkyColor {
        SkyColor {
            r: self.r * s,
            g: self.g * s,
            b: self.b * s,
        }
    }
}

/// Predefined sky colors for different times of day.
pub mod sky_colors {
    use super::SkyColor;

    /// Deep night sky.
    pub const NIGHT: SkyColor = SkyColor::new(0.02, 0.02, 0.08);
    /// Dawn/dusk horizon color.
    pub const DAWN_HORIZON: SkyColor = SkyColor::new(0.9, 0.5, 0.2);
    /// Dawn zenith.
    pub const DAWN_ZENITH: SkyColor = SkyColor::new(0.3, 0.3, 0.6);
    /// Morning sky.
    pub const MORNING: SkyColor = SkyColor::new(0.5, 0.7, 1.0);
    /// Midday sky.
    pub const MIDDAY: SkyColor = SkyColor::new(0.4, 0.6, 1.0);
    /// Afternoon sky (slightly warmer).
    pub const AFTERNOON: SkyColor = SkyColor::new(0.5, 0.65, 0.95);
    /// Sunset horizon.
    pub const SUNSET_HORIZON: SkyColor = SkyColor::new(1.0, 0.4, 0.1);
    /// Sunset zenith.
    pub const SUNSET_ZENITH: SkyColor = SkyColor::new(0.4, 0.2, 0.5);
    /// Evening sky.
    pub const EVENING: SkyColor = SkyColor::new(0.1, 0.1, 0.3);
}

// ---------------------------------------------------------------------------
// SkyPalette
// ---------------------------------------------------------------------------

/// A configurable color palette for sky rendering across the day.
#[derive(Debug, Clone)]
pub struct SkyPalette {
    /// Colors at specific day fractions (sorted by time).
    pub gradient_stops: Vec<(f32, SkyColor)>,
}

impl SkyPalette {
    /// Create a default palette.
    pub fn default_palette() -> Self {
        Self {
            gradient_stops: vec![
                (0.0, sky_colors::NIGHT),
                (0.22, sky_colors::NIGHT),
                (0.25, sky_colors::DAWN_HORIZON),
                (0.30, sky_colors::MORNING),
                (0.45, sky_colors::MIDDAY),
                (0.55, sky_colors::MIDDAY),
                (0.70, sky_colors::AFTERNOON),
                (0.74, sky_colors::SUNSET_HORIZON),
                (0.78, sky_colors::EVENING),
                (0.85, sky_colors::NIGHT),
                (1.0, sky_colors::NIGHT),
            ],
        }
    }

    /// Sample the palette at a given day fraction (0..1).
    pub fn sample(&self, t: f32) -> SkyColor {
        let t = t.rem_euclid(1.0);
        if self.gradient_stops.is_empty() {
            return sky_colors::MIDDAY;
        }
        if self.gradient_stops.len() == 1 {
            return self.gradient_stops[0].1;
        }

        // Find surrounding stops
        for i in 0..self.gradient_stops.len() - 1 {
            let (t0, c0) = self.gradient_stops[i];
            let (t1, c1) = self.gradient_stops[i + 1];
            if t >= t0 && t <= t1 {
                let local_t = if (t1 - t0).abs() < EPSILON {
                    0.0
                } else {
                    (t - t0) / (t1 - t0)
                };
                return SkyColor::lerp(c0, c1, local_t);
            }
        }

        self.gradient_stops.last().unwrap().1
    }
}

// ---------------------------------------------------------------------------
// DayNightConfig
// ---------------------------------------------------------------------------

/// Configuration for the day/night cycle.
#[derive(Debug, Clone)]
pub struct DayNightConfig {
    /// Real-time seconds per in-game day.
    pub day_length_seconds: f32,
    /// Sunrise time (fraction of day).
    pub sunrise_time: f32,
    /// Sunset time (fraction of day).
    pub sunset_time: f32,
    /// Twilight transition duration (fraction of day).
    pub twilight_duration: f32,
    /// Moon cycle length in game days.
    pub moon_cycle_days: f32,
    /// Latitude for sun angle calculation (degrees).
    pub latitude: f32,
    /// Sky color palette.
    pub sky_palette: SkyPalette,
    /// Maximum sun directional light intensity.
    pub max_sun_intensity: f32,
    /// Minimum ambient light at night.
    pub min_ambient_intensity: f32,
    /// Maximum ambient light at day.
    pub max_ambient_intensity: f32,
    /// Moon light intensity multiplier.
    pub moon_intensity_multiplier: f32,
    /// Night visibility multiplier.
    pub night_visibility: f32,
    /// Whether the time is paused.
    pub paused: bool,
    /// Time scale multiplier (1.0 = normal, 2.0 = double speed).
    pub time_scale: f32,
    /// Whether lights should auto-activate at night.
    pub auto_lights: bool,
    /// Ambient intensity below which lights activate.
    pub light_threshold: f32,
}

impl Default for DayNightConfig {
    fn default() -> Self {
        Self {
            day_length_seconds: DEFAULT_DAY_LENGTH,
            sunrise_time: DEFAULT_SUNRISE_TIME,
            sunset_time: DEFAULT_SUNSET_TIME,
            twilight_duration: TWILIGHT_DURATION,
            moon_cycle_days: DEFAULT_MOON_CYCLE,
            latitude: DEFAULT_LATITUDE,
            sky_palette: SkyPalette::default_palette(),
            max_sun_intensity: MAX_SUN_INTENSITY,
            min_ambient_intensity: MIN_AMBIENT_INTENSITY,
            max_ambient_intensity: MAX_AMBIENT_INTENSITY,
            moon_intensity_multiplier: 0.3,
            night_visibility: NIGHT_VISIBILITY_MULTIPLIER,
            paused: false,
            time_scale: 1.0,
            auto_lights: true,
            light_threshold: LIGHT_ACTIVATION_THRESHOLD,
        }
    }
}

// ---------------------------------------------------------------------------
// DayNightState
// ---------------------------------------------------------------------------

/// Current state of the day/night cycle.
#[derive(Debug, Clone)]
pub struct DayNightState {
    /// Current game time (fractional days since epoch).
    pub game_time: f64,
    /// Current day fraction (0..1, 0 = midnight).
    pub day_fraction: f32,
    /// Current day number (integer).
    pub day_number: u32,
    /// Current time period.
    pub period: TimePeriod,
    /// Sun direction (normalized, pointing toward the sun).
    pub sun_direction: Vec3,
    /// Sun intensity (0..1).
    pub sun_intensity: f32,
    /// Sun color.
    pub sun_color: SkyColor,
    /// Ambient light intensity.
    pub ambient_intensity: f32,
    /// Ambient light color.
    pub ambient_color: SkyColor,
    /// Sky color (at zenith).
    pub sky_color: SkyColor,
    /// Horizon color.
    pub horizon_color: SkyColor,
    /// Moon phase.
    pub moon_phase: MoonPhase,
    /// Moon direction.
    pub moon_direction: Vec3,
    /// Moon brightness (0..1).
    pub moon_brightness: f32,
    /// Whether it is currently night.
    pub is_night: bool,
    /// Whether it is twilight (dawn or dusk).
    pub is_twilight: bool,
    /// Visibility multiplier (1.0 = full visibility, <1 = reduced).
    pub visibility_multiplier: f32,
    /// Whether outdoor lights should be on.
    pub lights_active: bool,
    /// Whether stars are visible.
    pub stars_visible: bool,
    /// Star visibility intensity (0..1).
    pub star_intensity: f32,
    /// Hour of the day (0..23).
    pub hour: u8,
    /// Minute of the hour (0..59).
    pub minute: u8,
    /// Clock string (e.g., "14:30").
    pub clock_string: String,
}

impl DayNightState {
    /// Create initial state at midnight of day 0.
    pub fn new() -> Self {
        Self {
            game_time: 0.0,
            day_fraction: 0.0,
            day_number: 0,
            period: TimePeriod::Night,
            sun_direction: Vec3::new(0.0, -1.0, 0.0),
            sun_intensity: 0.0,
            sun_color: sky_colors::NIGHT,
            ambient_intensity: MIN_AMBIENT_INTENSITY,
            ambient_color: sky_colors::NIGHT,
            sky_color: sky_colors::NIGHT,
            horizon_color: sky_colors::NIGHT,
            moon_phase: MoonPhase::FullMoon,
            moon_direction: Vec3::new(0.0, 1.0, 0.0),
            moon_brightness: 1.0,
            is_night: true,
            is_twilight: false,
            visibility_multiplier: NIGHT_VISIBILITY_MULTIPLIER,
            lights_active: true,
            stars_visible: true,
            star_intensity: 1.0,
            hour: 0,
            minute: 0,
            clock_string: "00:00".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// LightSchedule
// ---------------------------------------------------------------------------

/// Schedule for an auto-activating light.
#[derive(Debug, Clone)]
pub struct LightSchedule {
    /// Entity/light identifier.
    pub light_id: u64,
    /// Whether the light is currently on.
    pub is_on: bool,
    /// Custom activation time override (day fraction, or None for default).
    pub activate_time: Option<f32>,
    /// Custom deactivation time override.
    pub deactivate_time: Option<f32>,
    /// Whether to use ambient threshold instead of fixed time.
    pub use_ambient_threshold: bool,
    /// Light color when active.
    pub light_color: SkyColor,
    /// Light intensity when active.
    pub light_intensity: f32,
    /// Flicker effect.
    pub flicker: bool,
    /// Flicker speed.
    pub flicker_speed: f32,
    /// Flicker intensity range.
    pub flicker_range: f32,
}

impl LightSchedule {
    /// Create a new light schedule with default settings.
    pub fn new(light_id: u64) -> Self {
        Self {
            light_id,
            is_on: false,
            activate_time: None,
            deactivate_time: None,
            use_ambient_threshold: true,
            light_color: SkyColor::new(1.0, 0.85, 0.6),
            light_intensity: 1.0,
            flicker: false,
            flicker_speed: 5.0,
            flicker_range: 0.2,
        }
    }

    /// Create a torch (with flicker).
    pub fn torch(light_id: u64) -> Self {
        Self {
            light_id,
            is_on: false,
            activate_time: None,
            deactivate_time: None,
            use_ambient_threshold: true,
            light_color: SkyColor::new(1.0, 0.7, 0.3),
            light_intensity: 1.2,
            flicker: true,
            flicker_speed: 8.0,
            flicker_range: 0.3,
        }
    }

    /// Update the light state based on the day/night state.
    pub fn update(&mut self, state: &DayNightState, config: &DayNightConfig) {
        if self.use_ambient_threshold {
            self.is_on = state.ambient_intensity < config.light_threshold;
        } else {
            let t = state.day_fraction;
            let activate = self.activate_time.unwrap_or(config.sunset_time);
            let deactivate = self.deactivate_time.unwrap_or(config.sunrise_time);

            if activate > deactivate {
                // Wraps around midnight
                self.is_on = t >= activate || t <= deactivate;
            } else {
                self.is_on = t >= activate && t <= deactivate;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NpcTimeSchedule
// ---------------------------------------------------------------------------

/// An NPC's behavior changes based on time of day.
#[derive(Debug, Clone)]
pub struct NpcTimeSchedule {
    /// NPC entity ID.
    pub npc_id: u64,
    /// Schedule entries.
    pub entries: Vec<NpcScheduleEntry>,
    /// Current active entry index.
    pub current_entry: Option<usize>,
}

/// A single entry in an NPC's time-based schedule.
#[derive(Debug, Clone)]
pub struct NpcScheduleEntry {
    /// Start time (day fraction).
    pub start_time: f32,
    /// End time (day fraction).
    pub end_time: f32,
    /// Location to move to.
    pub location: Vec3,
    /// Activity name.
    pub activity: String,
    /// Whether the NPC is interactable during this time.
    pub interactable: bool,
    /// Movement speed override.
    pub speed_override: Option<f32>,
}

impl NpcTimeSchedule {
    /// Create a new NPC schedule.
    pub fn new(npc_id: u64) -> Self {
        Self {
            npc_id,
            entries: Vec::new(),
            current_entry: None,
        }
    }

    /// Add a schedule entry.
    pub fn add_entry(&mut self, entry: NpcScheduleEntry) {
        self.entries.push(entry);
    }

    /// Get the current entry for a given time.
    pub fn entry_at(&self, day_fraction: f32) -> Option<(usize, &NpcScheduleEntry)> {
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.start_time <= entry.end_time {
                if day_fraction >= entry.start_time && day_fraction < entry.end_time {
                    return Some((i, entry));
                }
            } else {
                // Wraps around midnight
                if day_fraction >= entry.start_time || day_fraction < entry.end_time {
                    return Some((i, entry));
                }
            }
        }
        None
    }

    /// Update the schedule (check for entry transitions).
    pub fn update(&mut self, day_fraction: f32) -> Option<DayNightEvent> {
        let new_entry = self.entry_at(day_fraction).map(|(i, _)| i);
        if new_entry != self.current_entry {
            let old = self.current_entry;
            self.current_entry = new_entry;
            if let Some(idx) = new_entry {
                return Some(DayNightEvent::NpcScheduleChanged {
                    npc_id: self.npc_id,
                    activity: self.entries[idx].activity.clone(),
                    location: self.entries[idx].location,
                });
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// DayNightEvent
// ---------------------------------------------------------------------------

/// Events emitted by the day/night system.
#[derive(Debug, Clone)]
pub enum DayNightEvent {
    /// Time period changed.
    PeriodChanged { old: TimePeriod, new: TimePeriod },
    /// A new day started.
    NewDay { day_number: u32 },
    /// Moon phase changed.
    MoonPhaseChanged { phase: MoonPhase },
    /// Lights were activated.
    LightsActivated,
    /// Lights were deactivated.
    LightsDeactivated,
    /// NPC schedule changed.
    NpcScheduleChanged { npc_id: u64, activity: String, location: Vec3 },
    /// Dawn began.
    DawnBegan,
    /// Dusk began.
    DuskBegan,
    /// Hour changed.
    HourChanged { hour: u8 },
}

// ---------------------------------------------------------------------------
// DayNightSystem
// ---------------------------------------------------------------------------

/// The main day/night cycle system.
pub struct DayNightSystem {
    /// Configuration.
    pub config: DayNightConfig,
    /// Current state.
    pub state: DayNightState,
    /// Light schedules.
    lights: Vec<LightSchedule>,
    /// NPC schedules.
    npc_schedules: Vec<NpcTimeSchedule>,
    /// Events from last update.
    events: Vec<DayNightEvent>,
    /// Previous period (for change detection).
    prev_period: TimePeriod,
    /// Previous hour.
    prev_hour: u8,
    /// Previous day number.
    prev_day: u32,
    /// Previous lights_active state.
    prev_lights_active: bool,
    /// Previous moon phase.
    prev_moon_phase: MoonPhase,
}

impl DayNightSystem {
    /// Create a new day/night system with default config.
    pub fn new() -> Self {
        Self {
            config: DayNightConfig::default(),
            state: DayNightState::new(),
            lights: Vec::new(),
            npc_schedules: Vec::new(),
            events: Vec::new(),
            prev_period: TimePeriod::Night,
            prev_hour: 0,
            prev_day: 0,
            prev_lights_active: true,
            prev_moon_phase: MoonPhase::FullMoon,
        }
    }

    /// Create with a specific configuration.
    pub fn with_config(config: DayNightConfig) -> Self {
        let mut sys = Self::new();
        sys.config = config;
        sys
    }

    /// Set the current time (day fraction).
    pub fn set_time(&mut self, day_fraction: f32) {
        let day_number = self.state.day_number;
        self.state.game_time = day_number as f64 + day_fraction as f64;
        self.update_state();
    }

    /// Set the time to a specific hour and minute.
    pub fn set_clock(&mut self, hour: u8, minute: u8) {
        let fraction = (hour as f32 * 60.0 + minute as f32) / 1440.0;
        self.set_time(fraction);
    }

    /// Skip to a specific time period.
    pub fn skip_to_period(&mut self, period: TimePeriod) {
        let target = match period {
            TimePeriod::Night => 0.0,
            TimePeriod::Dawn => 0.24,
            TimePeriod::Morning => 0.30,
            TimePeriod::Midday => 0.50,
            TimePeriod::Afternoon => 0.60,
            TimePeriod::Dusk => 0.74,
            TimePeriod::Evening => 0.80,
            TimePeriod::LateNight => 0.92,
        };
        self.set_time(target);
    }

    /// Advance time by a specific number of hours.
    pub fn advance_hours(&mut self, hours: f32) {
        let fraction = hours / 24.0;
        self.state.game_time += fraction as f64;
        self.update_state();
    }

    /// Main update (call each frame).
    pub fn update(&mut self, dt: f32) {
        self.events.clear();

        if !self.config.paused {
            // Advance game time
            let day_fraction = dt * self.config.time_scale / self.config.day_length_seconds;
            self.state.game_time += day_fraction as f64;
        }

        self.update_state();
        self.check_events();
    }

    /// Recalculate all state from game_time.
    fn update_state(&mut self) {
        let t = self.state.game_time;
        self.state.day_number = t.floor() as u32;
        self.state.day_fraction = (t.fract()) as f32;

        // Calculate hour/minute
        let total_minutes = (self.state.day_fraction * 1440.0) as u32;
        self.state.hour = (total_minutes / 60).min(23) as u8;
        self.state.minute = (total_minutes % 60) as u8;
        self.state.clock_string = format!("{:02}:{:02}", self.state.hour, self.state.minute);

        // Time period
        self.state.period = TimePeriod::from_day_fraction(self.state.day_fraction);

        // Sun position
        let sun_angle = (self.state.day_fraction - self.config.sunrise_time)
            / (self.config.sunset_time - self.config.sunrise_time);
        let sun_elevation = (sun_angle * PI).sin();
        let sun_azimuth = sun_angle * PI;

        self.state.sun_direction = Vec3::new(
            sun_azimuth.cos(),
            sun_elevation.max(0.0),
            sun_azimuth.sin() * 0.3,
        ).normalize_or_zero();

        // Sun intensity
        let in_daylight = self.state.day_fraction >= self.config.sunrise_time
            && self.state.day_fraction <= self.config.sunset_time;
        let twilight_start_dawn = self.config.sunrise_time - self.config.twilight_duration;
        let twilight_end_dawn = self.config.sunrise_time + self.config.twilight_duration;
        let twilight_start_dusk = self.config.sunset_time - self.config.twilight_duration;
        let twilight_end_dusk = self.config.sunset_time + self.config.twilight_duration;

        let df = self.state.day_fraction;
        self.state.sun_intensity = if in_daylight {
            let mid_day = (self.config.sunrise_time + self.config.sunset_time) * 0.5;
            let half_day = (self.config.sunset_time - self.config.sunrise_time) * 0.5;
            let from_mid = (df - mid_day).abs() / half_day;
            self.config.max_sun_intensity * (1.0 - from_mid * 0.3)
        } else if df >= twilight_start_dawn && df < self.config.sunrise_time {
            let t = (df - twilight_start_dawn) / self.config.twilight_duration;
            t * 0.3
        } else if df > self.config.sunset_time && df <= twilight_end_dusk {
            let t = 1.0 - (df - self.config.sunset_time) / self.config.twilight_duration;
            t * 0.3
        } else {
            MIN_SUN_INTENSITY
        };

        // Ambient intensity
        self.state.ambient_intensity = if in_daylight {
            self.config.max_ambient_intensity
        } else if df >= twilight_start_dawn && df <= twilight_end_dawn {
            let t = (df - twilight_start_dawn) / (2.0 * self.config.twilight_duration);
            self.config.min_ambient_intensity + t * (self.config.max_ambient_intensity - self.config.min_ambient_intensity)
        } else if df >= twilight_start_dusk && df <= twilight_end_dusk {
            let t = 1.0 - (df - twilight_start_dusk) / (2.0 * self.config.twilight_duration);
            self.config.min_ambient_intensity + t * (self.config.max_ambient_intensity - self.config.min_ambient_intensity)
        } else {
            self.config.min_ambient_intensity
        };

        // Sky color
        self.state.sky_color = self.config.sky_palette.sample(df);
        self.state.horizon_color = self.config.sky_palette.sample(df);

        // Sun color
        self.state.sun_color = if self.state.sun_intensity > 0.5 {
            SkyColor::new(1.0, 0.95, 0.9)
        } else if self.state.sun_intensity > 0.1 {
            SkyColor::lerp(
                SkyColor::new(1.0, 0.5, 0.2),
                SkyColor::new(1.0, 0.95, 0.9),
                (self.state.sun_intensity - 0.1) / 0.4,
            )
        } else {
            SkyColor::new(0.8, 0.3, 0.1)
        };

        // Ambient color
        self.state.ambient_color = self.state.sky_color.scale(self.state.ambient_intensity);

        // Night/twilight flags
        self.state.is_night = !in_daylight && !(df >= twilight_start_dawn && df <= twilight_end_dusk);
        self.state.is_twilight = (df >= twilight_start_dawn && df < twilight_end_dawn)
            || (df >= twilight_start_dusk && df <= twilight_end_dusk);

        // Moon
        let moon_cycle_fraction = (self.state.game_time / self.config.moon_cycle_days as f64).fract() as f32;
        self.state.moon_phase = MoonPhase::from_cycle_fraction(moon_cycle_fraction);
        self.state.moon_brightness = self.state.moon_phase.brightness()
            * self.config.moon_intensity_multiplier;

        let moon_angle = (df + 0.5).rem_euclid(1.0) * TWO_PI;
        self.state.moon_direction = Vec3::new(
            moon_angle.cos(),
            (moon_angle * 0.5).sin().max(0.0),
            moon_angle.sin() * 0.3,
        ).normalize_or_zero();

        // Visibility
        self.state.visibility_multiplier = if self.state.is_night {
            self.config.night_visibility + self.state.moon_brightness * 0.2
        } else if self.state.is_twilight {
            0.7
        } else {
            1.0
        };

        // Lights
        self.state.lights_active = self.state.ambient_intensity < self.config.light_threshold;

        // Stars
        self.state.stars_visible = self.state.ambient_intensity < STAR_VISIBILITY_THRESHOLD;
        self.state.star_intensity = if self.state.stars_visible {
            1.0 - (self.state.ambient_intensity / STAR_VISIBILITY_THRESHOLD)
        } else {
            0.0
        };

        // Update lights
        for light in &mut self.lights {
            light.update(&self.state, &self.config);
        }

        // Update NPC schedules
        for schedule in &mut self.npc_schedules {
            if let Some(event) = schedule.update(df) {
                self.events.push(event);
            }
        }
    }

    /// Check for events (period changes, etc.).
    fn check_events(&mut self) {
        if self.state.period != self.prev_period {
            self.events.push(DayNightEvent::PeriodChanged {
                old: self.prev_period,
                new: self.state.period,
            });
            if self.state.period == TimePeriod::Dawn {
                self.events.push(DayNightEvent::DawnBegan);
            }
            if self.state.period == TimePeriod::Dusk {
                self.events.push(DayNightEvent::DuskBegan);
            }
            self.prev_period = self.state.period;
        }

        if self.state.hour != self.prev_hour {
            self.events.push(DayNightEvent::HourChanged { hour: self.state.hour });
            self.prev_hour = self.state.hour;
        }

        if self.state.day_number != self.prev_day {
            self.events.push(DayNightEvent::NewDay { day_number: self.state.day_number });
            self.prev_day = self.state.day_number;
        }

        if self.state.lights_active != self.prev_lights_active {
            if self.state.lights_active {
                self.events.push(DayNightEvent::LightsActivated);
            } else {
                self.events.push(DayNightEvent::LightsDeactivated);
            }
            self.prev_lights_active = self.state.lights_active;
        }

        if self.state.moon_phase != self.prev_moon_phase {
            self.events.push(DayNightEvent::MoonPhaseChanged {
                phase: self.state.moon_phase,
            });
            self.prev_moon_phase = self.state.moon_phase;
        }
    }

    /// Register a light schedule.
    pub fn add_light(&mut self, light: LightSchedule) {
        self.lights.push(light);
    }

    /// Register an NPC schedule.
    pub fn add_npc_schedule(&mut self, schedule: NpcTimeSchedule) {
        self.npc_schedules.push(schedule);
    }

    /// Get events from last update.
    pub fn events(&self) -> &[DayNightEvent] {
        &self.events
    }

    /// Get the current time period.
    pub fn period(&self) -> TimePeriod {
        self.state.period
    }

    /// Check if it is currently nighttime.
    pub fn is_night(&self) -> bool {
        self.state.is_night
    }

    /// Get the current game day number.
    pub fn day_number(&self) -> u32 {
        self.state.day_number
    }

    /// Get the clock string.
    pub fn clock(&self) -> &str {
        &self.state.clock_string
    }

    /// Get the visibility multiplier.
    pub fn visibility(&self) -> f32 {
        self.state.visibility_multiplier
    }
}

// ---------------------------------------------------------------------------
// DayNightComponent (ECS)
// ---------------------------------------------------------------------------

/// ECS component for entities affected by day/night.
#[derive(Debug, Clone)]
pub struct DayNightComponent {
    /// Whether this entity is affected by visibility changes.
    pub affected_by_visibility: bool,
    /// Whether this entity activates at night (e.g., torches).
    pub night_activation: bool,
    /// Whether to hide this entity during the day (e.g., nocturnal creatures).
    pub nocturnal: bool,
    /// Visibility modifier from the day/night system.
    pub visibility_modifier: f32,
}

impl Default for DayNightComponent {
    fn default() -> Self {
        Self {
            affected_by_visibility: true,
            night_activation: false,
            nocturnal: false,
            visibility_modifier: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_period() {
        assert_eq!(TimePeriod::from_day_fraction(0.0), TimePeriod::LateNight);
        assert_eq!(TimePeriod::from_day_fraction(0.25), TimePeriod::Dawn);
        assert_eq!(TimePeriod::from_day_fraction(0.50), TimePeriod::Midday);
        assert_eq!(TimePeriod::from_day_fraction(0.75), TimePeriod::Dusk);
    }

    #[test]
    fn test_moon_phase() {
        assert_eq!(MoonPhase::from_cycle_fraction(0.0), MoonPhase::NewMoon);
        assert_eq!(MoonPhase::from_cycle_fraction(0.5), MoonPhase::FullMoon);
        assert!(MoonPhase::FullMoon.brightness() > MoonPhase::NewMoon.brightness());
    }

    #[test]
    fn test_sky_color_lerp() {
        let a = SkyColor::new(0.0, 0.0, 0.0);
        let b = SkyColor::new(1.0, 1.0, 1.0);
        let mid = SkyColor::lerp(a, b, 0.5);
        assert!((mid.r - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_day_night_system_time_advance() {
        let mut system = DayNightSystem::new();
        system.set_time(0.0);
        assert_eq!(system.state.hour, 0);
        system.set_time(0.5);
        assert_eq!(system.state.hour, 12);
    }

    #[test]
    fn test_visibility_day_vs_night() {
        let mut system = DayNightSystem::new();
        system.set_time(0.5); // noon
        let day_vis = system.state.visibility_multiplier;

        system.set_time(0.0); // midnight
        let night_vis = system.state.visibility_multiplier;

        assert!(day_vis > night_vis);
    }

    #[test]
    fn test_set_clock() {
        let mut system = DayNightSystem::new();
        system.set_clock(14, 30);
        assert_eq!(system.state.hour, 14);
        assert_eq!(system.state.minute, 30);
    }
}
