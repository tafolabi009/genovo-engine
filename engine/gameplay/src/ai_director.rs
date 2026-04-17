//! AI Director / Game Master system for dynamic difficulty and pacing.
//!
//! Inspired by Left 4 Dead's AI Director, this module monitors player
//! performance and emotional state in real time, then adjusts game intensity
//! by issuing spawn directives, pacing events, and difficulty multipliers.
//!
//! # Overview
//!
//! - [`AIDirector`] — the central brain that runs a five-state pacing loop
//!   (Idle → BuildUp → Peak → Relax → Cooldown) and emits [`SpawnDirective`]s.
//! - [`PlayerStats`] — a snapshot of the player's current state fed into the
//!   director each tick.
//! - [`TensionTracker`] — accumulates tension from combat events and drains
//!   during calm periods.
//! - [`PacingCurve`] — a designer-authored time→intensity envelope with
//!   interpolation.
//! - [`DifficultyMultiplier`] — per-attribute scaling factors derived from
//!   player performance.
//!
//! All values are unitless floats in the range [0, 1] unless otherwise noted.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Intensity state
// ---------------------------------------------------------------------------

/// The five phases of the director's pacing loop.
///
/// The loop cycles: Idle → BuildUp → Peak → Relax → Cooldown → BuildUp → …
/// Each state has a minimum and maximum duration. Transitions happen when
/// the timer expires **and** intensity conditions are met.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntensityState {
    /// Nothing happening yet — waiting for the game to start or for the
    /// previous cooldown to expire.
    Idle,
    /// Ramping up: spawning increasing numbers of enemies, tightening the
    /// pressure on the player.
    BuildUp,
    /// Maximum intensity: boss waves, elite spawns, environmental hazards.
    Peak,
    /// Backing off: fewer spawns, more pickups, breathing room.
    Relax,
    /// Full calm: no active threats, health/ammo drops, ambient atmosphere.
    Cooldown,
}

impl Default for IntensityState {
    fn default() -> Self {
        Self::Idle
    }
}

impl IntensityState {
    /// The next state in the pacing loop.
    pub fn next(self) -> Self {
        match self {
            Self::Idle => Self::BuildUp,
            Self::BuildUp => Self::Peak,
            Self::Peak => Self::Relax,
            Self::Relax => Self::Cooldown,
            Self::Cooldown => Self::BuildUp,
        }
    }

    /// Target intensity range for this state. Returns `(min, max)`.
    pub fn target_intensity_range(self) -> (f32, f32) {
        match self {
            Self::Idle => (0.0, 0.05),
            Self::BuildUp => (0.2, 0.7),
            Self::Peak => (0.7, 1.0),
            Self::Relax => (0.1, 0.35),
            Self::Cooldown => (0.0, 0.1),
        }
    }

    /// Whether spawning is allowed in this state.
    pub fn allows_spawning(self) -> bool {
        matches!(self, Self::BuildUp | Self::Peak)
    }

    /// Whether pickup/recovery spawning is encouraged.
    pub fn encourages_recovery(self) -> bool {
        matches!(self, Self::Relax | Self::Cooldown)
    }
}

// ---------------------------------------------------------------------------
// State duration configuration
// ---------------------------------------------------------------------------

/// Min/max duration (in seconds) for a single pacing state.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StateDuration {
    /// Minimum seconds to stay in this state.
    pub min_secs: f32,
    /// Maximum seconds before forcing a transition.
    pub max_secs: f32,
}

impl StateDuration {
    pub fn new(min: f32, max: f32) -> Self {
        Self {
            min_secs: min,
            max_secs: max,
        }
    }
}

impl Default for StateDuration {
    fn default() -> Self {
        Self {
            min_secs: 10.0,
            max_secs: 30.0,
        }
    }
}

/// Duration configuration for every pacing state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacingConfig {
    pub idle: StateDuration,
    pub build_up: StateDuration,
    pub peak: StateDuration,
    pub relax: StateDuration,
    pub cooldown: StateDuration,
    /// How quickly the director can ramp intensity per second during BuildUp.
    pub build_up_rate: f32,
    /// How quickly the director reduces intensity per second during Relax.
    pub relax_rate: f32,
    /// Base spawn interval in seconds (modified by intensity).
    pub base_spawn_interval: f32,
    /// Maximum simultaneous active enemies.
    pub max_active_enemies: u32,
    /// Intensity threshold above which elites may spawn.
    pub elite_threshold: f32,
    /// Intensity threshold above which bosses may spawn.
    pub boss_threshold: f32,
    /// Minimum player health fraction before the director forces recovery.
    pub emergency_health_threshold: f32,
    /// Minimum player ammo fraction before the director spawns ammo.
    pub emergency_ammo_threshold: f32,
}

impl Default for PacingConfig {
    fn default() -> Self {
        Self {
            idle: StateDuration::new(3.0, 8.0),
            build_up: StateDuration::new(20.0, 60.0),
            peak: StateDuration::new(10.0, 30.0),
            relax: StateDuration::new(15.0, 40.0),
            cooldown: StateDuration::new(10.0, 25.0),
            build_up_rate: 0.03,
            relax_rate: 0.05,
            base_spawn_interval: 3.0,
            max_active_enemies: 30,
            elite_threshold: 0.6,
            boss_threshold: 0.85,
            emergency_health_threshold: 0.2,
            emergency_ammo_threshold: 0.15,
        }
    }
}

impl PacingConfig {
    /// Get the duration config for a specific state.
    pub fn duration_for(&self, state: IntensityState) -> &StateDuration {
        match state {
            IntensityState::Idle => &self.idle,
            IntensityState::BuildUp => &self.build_up,
            IntensityState::Peak => &self.peak,
            IntensityState::Relax => &self.relax,
            IntensityState::Cooldown => &self.cooldown,
        }
    }
}

// ---------------------------------------------------------------------------
// Player stats snapshot
// ---------------------------------------------------------------------------

/// A snapshot of the player's current state, fed to the director every tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerStats {
    /// Current health as a fraction [0, 1].
    pub health_fraction: f32,
    /// Current ammo as a fraction [0, 1] (1 = full ammo for all weapons).
    pub ammo_fraction: f32,
    /// Cumulative score.
    pub score: u64,
    /// Total kills this session.
    pub kills: u32,
    /// Total deaths this session.
    pub deaths: u32,
    /// Subjective stress level [0, 1] — derived from recent damage taken,
    /// nearby enemies, low resources, etc.
    pub stress_level: f32,
    /// Seconds the player has been idle (not moving or shooting).
    pub idle_time: f32,
    /// Current active enemy count near the player.
    pub nearby_enemies: u32,
    /// Seconds since last kill.
    pub time_since_last_kill: f32,
    /// Seconds since last damage taken.
    pub time_since_last_hit: f32,
    /// Player's accuracy over the last 30 seconds [0, 1].
    pub recent_accuracy: f32,
    /// Average time to kill an enemy (seconds, rolling window).
    pub avg_time_to_kill: f32,
}

impl Default for PlayerStats {
    fn default() -> Self {
        Self {
            health_fraction: 1.0,
            ammo_fraction: 1.0,
            score: 0,
            kills: 0,
            deaths: 0,
            stress_level: 0.0,
            idle_time: 0.0,
            nearby_enemies: 0,
            time_since_last_kill: 0.0,
            time_since_last_hit: 0.0,
            recent_accuracy: 0.5,
            avg_time_to_kill: 3.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Desired intensity
// ---------------------------------------------------------------------------

/// The intensity target the director is steering towards.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DesiredIntensity {
    /// Target intensity [0, 1].
    pub target: f32,
    /// How aggressively to steer towards the target (units/sec).
    pub convergence_rate: f32,
}

impl Default for DesiredIntensity {
    fn default() -> Self {
        Self {
            target: 0.0,
            convergence_rate: 0.02,
        }
    }
}

// ---------------------------------------------------------------------------
// Spawn directive
// ---------------------------------------------------------------------------

/// The type of ambient event the director can trigger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AmbientEventKind {
    /// Play a distant sound effect (growl, explosion, etc.).
    Sound(String),
    /// Flicker or change lighting in an area.
    LightingChange { area_id: String, intensity: f32 },
    /// Trigger a weather change.
    Weather(String),
    /// Trigger a screen shake or rumble.
    Shake { intensity: f32, duration: f32 },
    /// Custom event with a string tag.
    Custom(String),
}

/// A directive issued by the AI director to the game's spawner/event systems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpawnDirective {
    /// Adjust the base spawn rate multiplier.
    SetSpawnRateMultiplier(f32),
    /// Spawn a batch of regular enemies.
    SpawnEnemies {
        count: u32,
        entity_type: String,
    },
    /// Spawn an elite enemy.
    SpawnElite {
        entity_type: String,
        difficulty_scale: f32,
    },
    /// Spawn a boss encounter.
    SpawnBoss {
        entity_type: String,
        difficulty_scale: f32,
    },
    /// Spawn a health pickup.
    SpawnHealthPickup {
        amount: f32,
    },
    /// Spawn an ammo pickup.
    SpawnAmmoPickup {
        amount: f32,
    },
    /// Trigger an ambient event.
    TriggerAmbientEvent(AmbientEventKind),
    /// Stop all active spawning.
    HaltSpawning,
    /// Resume spawning after a halt.
    ResumeSpawning,
    /// Set the difficulty multiplier for all future spawns.
    SetDifficultyMultiplier(DifficultyMultiplier),
}

// ---------------------------------------------------------------------------
// Difficulty multiplier
// ---------------------------------------------------------------------------

/// Scaling factors applied to enemies based on player performance.
///
/// All values default to 1.0 (no change). Values > 1 make the game harder,
/// values < 1 make it easier.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DifficultyMultiplier {
    /// Enemy health multiplier.
    pub enemy_health: f32,
    /// Enemy damage output multiplier.
    pub enemy_damage: f32,
    /// Enemy movement speed multiplier.
    pub enemy_speed: f32,
    /// Spawn rate multiplier (higher = more enemies per unit time).
    pub spawn_rate: f32,
    /// Elite spawn chance multiplier.
    pub elite_chance: f32,
    /// Pickup drop rate multiplier (higher = more generous).
    pub pickup_rate: f32,
}

impl Default for DifficultyMultiplier {
    fn default() -> Self {
        Self {
            enemy_health: 1.0,
            enemy_damage: 1.0,
            enemy_speed: 1.0,
            spawn_rate: 1.0,
            elite_chance: 1.0,
            pickup_rate: 1.0,
        }
    }
}

impl DifficultyMultiplier {
    /// Compute difficulty scaling from the player's kill/death ratio and
    /// average time-to-kill.
    ///
    /// Good players (high KD, fast TTK) get harder enemies; struggling
    /// players get easier ones.
    pub fn from_performance(kills: u32, deaths: u32, avg_ttk: f32, accuracy: f32) -> Self {
        let kd_ratio = if deaths == 0 {
            kills as f32
        } else {
            kills as f32 / deaths as f32
        };

        // Normalize KD: 1.0 is "average", >2 is strong, <0.5 is struggling
        let kd_factor = (kd_ratio / 1.5).clamp(0.5, 2.0);

        // Fast TTK means the player is good — reference TTK is 3 seconds
        let ttk_factor = (3.0 / avg_ttk.max(0.5)).clamp(0.5, 2.0);

        // High accuracy means the player is skilled
        let acc_factor = (accuracy / 0.4).clamp(0.5, 1.8);

        // Composite performance score
        let perf = (kd_factor * 0.4 + ttk_factor * 0.3 + acc_factor * 0.3).clamp(0.4, 2.0);

        Self {
            enemy_health: (0.7 + perf * 0.3).clamp(0.5, 1.8),
            enemy_damage: (0.6 + perf * 0.4).clamp(0.4, 1.6),
            enemy_speed: (0.8 + perf * 0.2).clamp(0.7, 1.4),
            spawn_rate: (0.5 + perf * 0.5).clamp(0.3, 2.0),
            elite_chance: (0.3 + perf * 0.7).clamp(0.1, 2.5),
            pickup_rate: (1.8 - perf * 0.6).clamp(0.5, 2.0),
        }
    }

    /// Linearly interpolate between two difficulty multipliers.
    pub fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            enemy_health: a.enemy_health + (b.enemy_health - a.enemy_health) * t,
            enemy_damage: a.enemy_damage + (b.enemy_damage - a.enemy_damage) * t,
            enemy_speed: a.enemy_speed + (b.enemy_speed - a.enemy_speed) * t,
            spawn_rate: a.spawn_rate + (b.spawn_rate - a.spawn_rate) * t,
            elite_chance: a.elite_chance + (b.elite_chance - a.elite_chance) * t,
            pickup_rate: a.pickup_rate + (b.pickup_rate - a.pickup_rate) * t,
        }
    }
}

// ---------------------------------------------------------------------------
// Tension tracker
// ---------------------------------------------------------------------------

/// Tracks accumulated "tension" from combat events. Tension rises when
/// enemies are nearby, the player takes damage, or the player is in danger.
/// It drains gradually during calm periods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensionTracker {
    current_tension: f32,
    max_tension: f32,
    drain_rate: f32,
    /// History of tension values for smoothing (circular buffer).
    history: VecDeque<f32>,
    history_capacity: usize,
    /// Accumulated time for history sampling.
    sample_timer: f32,
    sample_interval: f32,
}

impl Default for TensionTracker {
    fn default() -> Self {
        Self {
            current_tension: 0.0,
            max_tension: 100.0,
            drain_rate: 2.0,
            history: VecDeque::new(),
            history_capacity: 60,
            sample_timer: 0.0,
            sample_interval: 1.0,
        }
    }
}

impl TensionTracker {
    /// Create a new tension tracker.
    pub fn new(max_tension: f32, drain_rate: f32) -> Self {
        Self {
            max_tension,
            drain_rate,
            ..Default::default()
        }
    }

    /// Add tension from a combat event (damage taken, nearby explosion, etc.).
    pub fn add_tension(&mut self, amount: f32) {
        self.current_tension = (self.current_tension + amount).min(self.max_tension);
    }

    /// Drain tension over time. Called with `dt` each frame during calm.
    pub fn drain(&mut self, dt: f32) {
        self.current_tension = (self.current_tension - self.drain_rate * dt).max(0.0);
    }

    /// Drain tension at a custom rate.
    pub fn drain_at_rate(&mut self, rate: f32, dt: f32) {
        self.current_tension = (self.current_tension - rate * dt).max(0.0);
    }

    /// Current tension value [0, max_tension].
    pub fn current_tension(&self) -> f32 {
        self.current_tension
    }

    /// Current tension as a normalized value [0, 1].
    pub fn normalized_tension(&self) -> f32 {
        if self.max_tension <= 0.0 {
            return 0.0;
        }
        self.current_tension / self.max_tension
    }

    /// Update the history buffer. Should be called each frame with `dt`.
    pub fn update(&mut self, dt: f32) {
        self.sample_timer += dt;
        if self.sample_timer >= self.sample_interval {
            self.sample_timer -= self.sample_interval;
            self.history.push_back(self.current_tension);
            if self.history.len() > self.history_capacity {
                self.history.pop_front();
            }
        }
    }

    /// Average tension over the recorded history.
    pub fn average_tension(&self) -> f32 {
        if self.history.is_empty() {
            return self.current_tension;
        }
        let sum: f32 = self.history.iter().sum();
        sum / self.history.len() as f32
    }

    /// Peak tension in the recorded history.
    pub fn peak_tension(&self) -> f32 {
        self.history
            .iter()
            .copied()
            .fold(self.current_tension, f32::max)
    }

    /// Variance of tension over the history (measure of volatility).
    pub fn tension_variance(&self) -> f32 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let mean = self.average_tension();
        let sum_sq: f32 = self.history.iter().map(|&v| (v - mean) * (v - mean)).sum();
        sum_sq / self.history.len() as f32
    }

    /// Whether the player is currently "in combat" (tension above threshold).
    pub fn is_in_combat(&self, threshold: f32) -> bool {
        self.normalized_tension() > threshold
    }

    /// Reset tension to zero and clear history.
    pub fn reset(&mut self) {
        self.current_tension = 0.0;
        self.history.clear();
        self.sample_timer = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Pacing curve
// ---------------------------------------------------------------------------

/// A single waypoint on the pacing curve.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PacingWaypoint {
    /// Time in seconds from the start of the level/session.
    pub time: f32,
    /// Target intensity at this time [0, 1].
    pub intensity: f32,
}

/// A designer-authored time→intensity envelope.
///
/// The curve is defined by a sequence of waypoints, with linear interpolation
/// between them. The director can use this as a "suggested" pacing and
/// override it based on player performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacingCurve {
    waypoints: Vec<PacingWaypoint>,
    /// How strongly the curve influences the director [0, 1].
    /// 0 = director ignores the curve entirely.
    /// 1 = director follows the curve exactly.
    pub influence: f32,
}

impl Default for PacingCurve {
    fn default() -> Self {
        // A default 5-minute pacing curve with a classic "roller coaster".
        Self {
            waypoints: vec![
                PacingWaypoint { time: 0.0, intensity: 0.0 },
                PacingWaypoint { time: 10.0, intensity: 0.1 },
                PacingWaypoint { time: 30.0, intensity: 0.4 },
                PacingWaypoint { time: 60.0, intensity: 0.7 },
                PacingWaypoint { time: 90.0, intensity: 1.0 },
                PacingWaypoint { time: 120.0, intensity: 0.5 },
                PacingWaypoint { time: 150.0, intensity: 0.2 },
                PacingWaypoint { time: 180.0, intensity: 0.6 },
                PacingWaypoint { time: 210.0, intensity: 0.9 },
                PacingWaypoint { time: 240.0, intensity: 0.3 },
                PacingWaypoint { time: 270.0, intensity: 0.7 },
                PacingWaypoint { time: 300.0, intensity: 1.0 },
            ],
            influence: 0.5,
        }
    }
}

impl PacingCurve {
    /// Create a new pacing curve from waypoints. They will be sorted by time.
    pub fn new(mut waypoints: Vec<PacingWaypoint>, influence: f32) -> Self {
        waypoints.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
        Self {
            waypoints,
            influence: influence.clamp(0.0, 1.0),
        }
    }

    /// Add a waypoint. The list will be re-sorted.
    pub fn add_waypoint(&mut self, time: f32, intensity: f32) {
        self.waypoints.push(PacingWaypoint {
            time,
            intensity: intensity.clamp(0.0, 1.0),
        });
        self.waypoints
            .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Sample the curve at a given time, using linear interpolation.
    pub fn sample(&self, time: f32) -> f32 {
        if self.waypoints.is_empty() {
            return 0.0;
        }

        // Before first waypoint
        if time <= self.waypoints[0].time {
            return self.waypoints[0].intensity;
        }

        // After last waypoint
        if time >= self.waypoints[self.waypoints.len() - 1].time {
            return self.waypoints[self.waypoints.len() - 1].intensity;
        }

        // Find the two surrounding waypoints and interpolate
        for i in 0..self.waypoints.len() - 1 {
            let a = &self.waypoints[i];
            let b = &self.waypoints[i + 1];
            if time >= a.time && time <= b.time {
                let range = b.time - a.time;
                if range <= f32::EPSILON {
                    return a.intensity;
                }
                let t = (time - a.time) / range;
                return a.intensity + (b.intensity - a.intensity) * t;
            }
        }

        self.waypoints.last().map_or(0.0, |w| w.intensity)
    }

    /// Sample with smoothstep interpolation for softer transitions.
    pub fn sample_smooth(&self, time: f32) -> f32 {
        if self.waypoints.is_empty() {
            return 0.0;
        }

        if time <= self.waypoints[0].time {
            return self.waypoints[0].intensity;
        }

        if time >= self.waypoints[self.waypoints.len() - 1].time {
            return self.waypoints[self.waypoints.len() - 1].intensity;
        }

        for i in 0..self.waypoints.len() - 1 {
            let a = &self.waypoints[i];
            let b = &self.waypoints[i + 1];
            if time >= a.time && time <= b.time {
                let range = b.time - a.time;
                if range <= f32::EPSILON {
                    return a.intensity;
                }
                let t = (time - a.time) / range;
                // Smoothstep: 3t^2 - 2t^3
                let t_smooth = t * t * (3.0 - 2.0 * t);
                return a.intensity + (b.intensity - a.intensity) * t_smooth;
            }
        }

        self.waypoints.last().map_or(0.0, |w| w.intensity)
    }

    /// Total duration of the curve.
    pub fn duration(&self) -> f32 {
        self.waypoints.last().map_or(0.0, |w| w.time)
    }

    /// Number of waypoints.
    pub fn waypoint_count(&self) -> usize {
        self.waypoints.len()
    }
}

// ---------------------------------------------------------------------------
// Intensity calculator
// ---------------------------------------------------------------------------

/// Computes a composite intensity score from player stats.
///
/// The score is a weighted combination of several factors:
/// - Inverse health (low health = high intensity)
/// - Inverse ammo (low ammo = high intensity)
/// - Stress level
/// - Nearby enemy count
/// - Recency of combat (time since last kill/hit)
pub fn calculate_intensity(stats: &PlayerStats) -> f32 {
    let health_factor = 1.0 - stats.health_fraction.clamp(0.0, 1.0);
    let ammo_factor = 1.0 - stats.ammo_fraction.clamp(0.0, 1.0);
    let stress_factor = stats.stress_level.clamp(0.0, 1.0);

    // Nearby enemies: 0 → 0, 5 → 0.5, 10+ → 1.0
    let enemy_factor = (stats.nearby_enemies as f32 / 10.0).clamp(0.0, 1.0);

    // Recent combat: high if the player was recently in a fight
    let kill_recency = (1.0 - stats.time_since_last_kill / 15.0).clamp(0.0, 1.0);
    let hit_recency = (1.0 - stats.time_since_last_hit / 10.0).clamp(0.0, 1.0);
    let combat_factor = kill_recency.max(hit_recency);

    // Idle penalty: long idle time means the intensity should be low
    let idle_factor = (1.0 - stats.idle_time / 30.0).clamp(0.0, 1.0);

    // Weighted composite
    let intensity = health_factor * 0.15
        + ammo_factor * 0.10
        + stress_factor * 0.25
        + enemy_factor * 0.20
        + combat_factor * 0.20
        + idle_factor * 0.10;

    intensity.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Director event log
// ---------------------------------------------------------------------------

/// An event recorded by the director for debugging and analytics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectorEvent {
    /// Timestamp (seconds from session start).
    pub time: f32,
    /// The state the director was in when this event occurred.
    pub state: IntensityState,
    /// Description of what happened.
    pub description: String,
    /// The current intensity at the time of the event.
    pub intensity: f32,
}

/// A rolling log of director events, capped at a maximum size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectorLog {
    events: VecDeque<DirectorEvent>,
    capacity: usize,
}

impl Default for DirectorLog {
    fn default() -> Self {
        Self {
            events: VecDeque::new(),
            capacity: 200,
        }
    }
}

impl DirectorLog {
    pub fn new(capacity: usize) -> Self {
        Self {
            events: VecDeque::new(),
            capacity,
        }
    }

    pub fn push(&mut self, event: DirectorEvent) {
        if self.events.len() >= self.capacity {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    pub fn events(&self) -> &VecDeque<DirectorEvent> {
        &self.events
    }

    pub fn recent(&self, count: usize) -> impl Iterator<Item = &DirectorEvent> {
        self.events.iter().rev().take(count)
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }
}

// ---------------------------------------------------------------------------
// AI Director
// ---------------------------------------------------------------------------

/// The AI Director monitors the player and dynamically adjusts game intensity.
///
/// # Usage
///
/// ```rust,ignore
/// let mut director = AIDirector::new(PacingConfig::default());
/// // each frame:
/// let directives = director.update(dt, &player_stats);
/// for directive in directives {
///     // forward to spawner, event system, etc.
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIDirector {
    config: PacingConfig,
    state: IntensityState,
    /// Time spent in the current state.
    state_timer: f32,
    /// Total elapsed time since the director started.
    elapsed: f32,
    /// Current intensity level [0, 1].
    current_intensity: f32,
    /// The desired intensity based on state + curve.
    desired: DesiredIntensity,
    /// Tension tracker for combat events.
    tension: TensionTracker,
    /// Optional pacing curve for designer-authored intensity envelopes.
    pacing_curve: Option<PacingCurve>,
    /// Current difficulty multiplier.
    difficulty: DifficultyMultiplier,
    /// Event log for debugging.
    log: DirectorLog,
    /// Time since last spawn directive was issued.
    spawn_timer: f32,
    /// Whether spawning is currently halted.
    spawning_halted: bool,
    /// Rolling count of enemies spawned in the current state.
    enemies_spawned_in_state: u32,
    /// Whether a boss has been spawned in the current peak.
    boss_spawned_this_peak: bool,
    /// Number of peak phases completed.
    peaks_completed: u32,
    /// Emergency mode: the player is in critical danger.
    emergency_mode: bool,
    /// Time since last recovery pickup was spawned.
    recovery_timer: f32,
}

impl AIDirector {
    /// Create a new AI director with the given pacing configuration.
    pub fn new(config: PacingConfig) -> Self {
        Self {
            config,
            state: IntensityState::Idle,
            state_timer: 0.0,
            elapsed: 0.0,
            current_intensity: 0.0,
            desired: DesiredIntensity::default(),
            tension: TensionTracker::default(),
            pacing_curve: None,
            difficulty: DifficultyMultiplier::default(),
            log: DirectorLog::default(),
            spawn_timer: 0.0,
            spawning_halted: false,
            enemies_spawned_in_state: 0,
            boss_spawned_this_peak: false,
            peaks_completed: 0,
            emergency_mode: false,
            recovery_timer: 0.0,
        }
    }

    /// Create a director with a pacing curve attached.
    pub fn with_pacing_curve(config: PacingConfig, curve: PacingCurve) -> Self {
        let mut director = Self::new(config);
        director.pacing_curve = Some(curve);
        director
    }

    // -- Accessors --

    pub fn state(&self) -> IntensityState {
        self.state
    }

    pub fn current_intensity(&self) -> f32 {
        self.current_intensity
    }

    pub fn elapsed(&self) -> f32 {
        self.elapsed
    }

    pub fn difficulty(&self) -> &DifficultyMultiplier {
        &self.difficulty
    }

    pub fn tension(&self) -> &TensionTracker {
        &self.tension
    }

    pub fn tension_mut(&mut self) -> &mut TensionTracker {
        &mut self.tension
    }

    pub fn log(&self) -> &DirectorLog {
        &self.log
    }

    pub fn is_emergency(&self) -> bool {
        self.emergency_mode
    }

    pub fn peaks_completed(&self) -> u32 {
        self.peaks_completed
    }

    pub fn set_pacing_curve(&mut self, curve: PacingCurve) {
        self.pacing_curve = Some(curve);
    }

    /// Report a combat event to the director (adds tension).
    pub fn report_combat_event(&mut self, tension_amount: f32) {
        self.tension.add_tension(tension_amount);
    }

    // -- Core update --

    /// Advance the director by `dt` seconds. Returns a list of directives
    /// to be executed by the game's spawner and event systems.
    pub fn update(&mut self, dt: f32, stats: &PlayerStats) -> Vec<SpawnDirective> {
        self.elapsed += dt;
        self.state_timer += dt;
        self.spawn_timer += dt;
        self.recovery_timer += dt;

        // Update tension: drain during calm, the game code should call
        // report_combat_event() to add tension.
        let in_combat = self.tension.is_in_combat(0.2);
        if !in_combat {
            self.tension.drain(dt);
        }
        self.tension.update(dt);

        // Calculate measured intensity from player stats
        let measured = calculate_intensity(stats);

        // Get pacing curve suggestion (if any)
        let curve_intensity = self
            .pacing_curve
            .as_ref()
            .map(|c| c.sample_smooth(self.elapsed) * c.influence)
            .unwrap_or(0.0);

        // Determine desired intensity from state
        let (target_min, target_max) = self.state.target_intensity_range();
        let state_target = (target_min + target_max) * 0.5;

        // Blend state target with curve suggestion
        let curve_weight = self.pacing_curve.as_ref().map_or(0.0, |c| c.influence);
        let blended_target =
            state_target * (1.0 - curve_weight) + curve_intensity;
        self.desired.target = blended_target.clamp(0.0, 1.0);

        // Steer current intensity towards desired
        let diff = self.desired.target - self.current_intensity;
        let max_change = match self.state {
            IntensityState::BuildUp => self.config.build_up_rate * dt,
            IntensityState::Relax | IntensityState::Cooldown => self.config.relax_rate * dt,
            _ => 0.02 * dt,
        };

        if diff.abs() > max_change {
            self.current_intensity += max_change * diff.signum();
        } else {
            self.current_intensity = self.desired.target;
        }
        self.current_intensity = self.current_intensity.clamp(0.0, 1.0);

        // Check for emergency conditions
        let was_emergency = self.emergency_mode;
        self.emergency_mode = stats.health_fraction < self.config.emergency_health_threshold
            || stats.ammo_fraction < self.config.emergency_ammo_threshold;

        let mut directives = Vec::new();

        if self.emergency_mode && !was_emergency {
            self.log.push(DirectorEvent {
                time: self.elapsed,
                state: self.state,
                description: "Emergency mode activated — player critically low on resources"
                    .into(),
                intensity: self.current_intensity,
            });
            // In emergency mode, halt spawning and provide pickups
            directives.push(SpawnDirective::HaltSpawning);
            self.spawning_halted = true;
        } else if !self.emergency_mode && was_emergency {
            self.log.push(DirectorEvent {
                time: self.elapsed,
                state: self.state,
                description: "Emergency mode deactivated — player recovered".into(),
                intensity: self.current_intensity,
            });
            directives.push(SpawnDirective::ResumeSpawning);
            self.spawning_halted = false;
        }

        // Spawn recovery pickups during emergency or relaxation
        if self.emergency_mode || self.state.encourages_recovery() {
            if self.recovery_timer >= 5.0 {
                self.recovery_timer = 0.0;
                if stats.health_fraction < 0.7 {
                    directives.push(SpawnDirective::SpawnHealthPickup {
                        amount: if self.emergency_mode { 0.4 } else { 0.2 },
                    });
                }
                if stats.ammo_fraction < 0.5 {
                    directives.push(SpawnDirective::SpawnAmmoPickup {
                        amount: if self.emergency_mode { 0.5 } else { 0.25 },
                    });
                }
            }
        }

        // State machine transitions
        let dur = self.config.duration_for(self.state);
        let should_transition = if self.state_timer >= dur.max_secs {
            true // Forced transition
        } else if self.state_timer >= dur.min_secs {
            // Check intensity conditions
            match self.state {
                IntensityState::Idle => true,
                IntensityState::BuildUp => {
                    self.current_intensity >= 0.65 || measured >= 0.7
                }
                IntensityState::Peak => {
                    // Stay in peak as long as tension is high, unless max time hit
                    self.tension.normalized_tension() < 0.3
                        || self.current_intensity < 0.5
                }
                IntensityState::Relax => {
                    self.current_intensity < 0.2 && self.tension.normalized_tension() < 0.15
                }
                IntensityState::Cooldown => {
                    self.tension.normalized_tension() < 0.05
                }
            }
        } else {
            false
        };

        if should_transition {
            let old_state = self.state;
            self.state = self.state.next();
            self.state_timer = 0.0;
            self.enemies_spawned_in_state = 0;

            if self.state == IntensityState::Peak {
                self.boss_spawned_this_peak = false;
            }
            if old_state == IntensityState::Peak {
                self.peaks_completed += 1;
            }

            self.log.push(DirectorEvent {
                time: self.elapsed,
                state: self.state,
                description: format!("State transition: {:?} -> {:?}", old_state, self.state),
                intensity: self.current_intensity,
            });
        }

        // Update difficulty based on player performance
        let new_difficulty = DifficultyMultiplier::from_performance(
            stats.kills,
            stats.deaths,
            stats.avg_time_to_kill,
            stats.recent_accuracy,
        );
        self.difficulty = DifficultyMultiplier::lerp(&self.difficulty, &new_difficulty, 0.01);

        // Generate spawn directives based on current state
        if !self.spawning_halted && self.state.allows_spawning() {
            let spawn_interval = self.config.base_spawn_interval
                / (self.current_intensity.max(0.1) * self.difficulty.spawn_rate);

            if self.spawn_timer >= spawn_interval {
                self.spawn_timer = 0.0;

                // Determine spawn count based on intensity
                let base_count = (self.current_intensity * 4.0).ceil() as u32;
                let count = base_count.max(1).min(self.config.max_active_enemies - stats.nearby_enemies);

                if count > 0 && stats.nearby_enemies < self.config.max_active_enemies {
                    directives.push(SpawnDirective::SpawnEnemies {
                        count,
                        entity_type: "enemy_basic".into(),
                    });
                    self.enemies_spawned_in_state += count;
                }

                // Elite spawn chance
                if self.current_intensity >= self.config.elite_threshold {
                    let elite_chance = (self.current_intensity - self.config.elite_threshold)
                        * self.difficulty.elite_chance
                        * 0.5;
                    // Use a simple deterministic check based on spawn count
                    if self.enemies_spawned_in_state % 7 == 0 && elite_chance > 0.1 {
                        directives.push(SpawnDirective::SpawnElite {
                            entity_type: "enemy_elite".into(),
                            difficulty_scale: self.difficulty.enemy_health,
                        });
                    }
                }

                // Boss spawn during peak
                if self.state == IntensityState::Peak
                    && !self.boss_spawned_this_peak
                    && self.current_intensity >= self.config.boss_threshold
                    && self.state_timer > 5.0
                {
                    self.boss_spawned_this_peak = true;
                    directives.push(SpawnDirective::SpawnBoss {
                        entity_type: "enemy_boss".into(),
                        difficulty_scale: self.difficulty.enemy_health * 1.5,
                    });
                    self.log.push(DirectorEvent {
                        time: self.elapsed,
                        state: self.state,
                        description: "Boss spawned!".into(),
                        intensity: self.current_intensity,
                    });
                }
            }
        }

        // Ambient events based on state
        if self.state == IntensityState::BuildUp && self.state_timer > 3.0 {
            // Occasional ambient tension events
            let ambient_interval = 10.0 - self.current_intensity * 5.0;
            if self.state_timer % ambient_interval < dt {
                directives.push(SpawnDirective::TriggerAmbientEvent(
                    AmbientEventKind::Sound("distant_growl".into()),
                ));
            }
        }

        // Propagate difficulty multiplier periodically
        if self.elapsed as u32 % 10 == 0 && self.elapsed.fract() < dt {
            directives.push(SpawnDirective::SetDifficultyMultiplier(self.difficulty));
            directives.push(SpawnDirective::SetSpawnRateMultiplier(
                self.difficulty.spawn_rate,
            ));
        }

        directives
    }

    /// Force a state transition (useful for scripted events).
    pub fn force_state(&mut self, state: IntensityState) {
        let old = self.state;
        self.state = state;
        self.state_timer = 0.0;
        self.enemies_spawned_in_state = 0;
        self.log.push(DirectorEvent {
            time: self.elapsed,
            state: self.state,
            description: format!("Forced state transition: {:?} -> {:?}", old, state),
            intensity: self.current_intensity,
        });
    }

    /// Force the intensity to a specific value (useful for cutscenes).
    pub fn force_intensity(&mut self, intensity: f32) {
        self.current_intensity = intensity.clamp(0.0, 1.0);
        self.desired.target = self.current_intensity;
    }

    /// Reset the director to its initial state.
    pub fn reset(&mut self) {
        self.state = IntensityState::Idle;
        self.state_timer = 0.0;
        self.elapsed = 0.0;
        self.current_intensity = 0.0;
        self.desired = DesiredIntensity::default();
        self.tension.reset();
        self.difficulty = DifficultyMultiplier::default();
        self.log.clear();
        self.spawn_timer = 0.0;
        self.spawning_halted = false;
        self.enemies_spawned_in_state = 0;
        self.boss_spawned_this_peak = false;
        self.peaks_completed = 0;
        self.emergency_mode = false;
        self.recovery_timer = 0.0;
    }

    /// Get a human-readable summary of the director's current state.
    pub fn debug_summary(&self) -> String {
        format!(
            "Director: state={:?} timer={:.1}s intensity={:.2} tension={:.2} \
             emergency={} peaks={} spawns_in_state={}",
            self.state,
            self.state_timer,
            self.current_intensity,
            self.tension.normalized_tension(),
            self.emergency_mode,
            self.peaks_completed,
            self.enemies_spawned_in_state,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intensity_state_cycle() {
        let mut state = IntensityState::Idle;
        let expected = [
            IntensityState::BuildUp,
            IntensityState::Peak,
            IntensityState::Relax,
            IntensityState::Cooldown,
            IntensityState::BuildUp,
        ];
        for &exp in &expected {
            state = state.next();
            assert_eq!(state, exp);
        }
    }

    #[test]
    fn test_calculate_intensity_baseline() {
        let stats = PlayerStats::default();
        let intensity = calculate_intensity(&stats);
        // Default stats: full health/ammo, no stress, no enemies → low intensity
        assert!(intensity < 0.3, "Expected low intensity, got {intensity}");
    }

    #[test]
    fn test_calculate_intensity_high_danger() {
        let stats = PlayerStats {
            health_fraction: 0.1,
            ammo_fraction: 0.1,
            stress_level: 0.9,
            nearby_enemies: 10,
            time_since_last_kill: 1.0,
            time_since_last_hit: 1.0,
            idle_time: 0.0,
            ..Default::default()
        };
        let intensity = calculate_intensity(&stats);
        assert!(intensity > 0.7, "Expected high intensity, got {intensity}");
    }

    #[test]
    fn test_tension_tracker_add_and_drain() {
        let mut tracker = TensionTracker::new(100.0, 5.0);
        tracker.add_tension(50.0);
        assert!((tracker.current_tension() - 50.0).abs() < f32::EPSILON);

        tracker.drain(2.0);
        assert!((tracker.current_tension() - 40.0).abs() < f32::EPSILON);

        tracker.drain(20.0);
        assert!(tracker.current_tension() >= 0.0);
        assert!(tracker.current_tension() < f32::EPSILON);
    }

    #[test]
    fn test_tension_tracker_max_cap() {
        let mut tracker = TensionTracker::new(100.0, 1.0);
        tracker.add_tension(200.0);
        assert!((tracker.current_tension() - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_pacing_curve_interpolation() {
        let curve = PacingCurve::new(
            vec![
                PacingWaypoint { time: 0.0, intensity: 0.0 },
                PacingWaypoint { time: 10.0, intensity: 1.0 },
            ],
            1.0,
        );
        assert!((curve.sample(5.0) - 0.5).abs() < 0.01);
        assert!((curve.sample(0.0) - 0.0).abs() < 0.01);
        assert!((curve.sample(10.0) - 1.0).abs() < 0.01);
        assert!((curve.sample(-1.0) - 0.0).abs() < 0.01);
        assert!((curve.sample(15.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pacing_curve_smooth() {
        let curve = PacingCurve::new(
            vec![
                PacingWaypoint { time: 0.0, intensity: 0.0 },
                PacingWaypoint { time: 10.0, intensity: 1.0 },
            ],
            1.0,
        );
        // Smoothstep at t=0.5 → 0.5 (symmetric)
        let val = curve.sample_smooth(5.0);
        assert!((val - 0.5).abs() < 0.01, "Smoothstep midpoint should be ~0.5, got {val}");
    }

    #[test]
    fn test_difficulty_from_performance() {
        // Average player
        let avg = DifficultyMultiplier::from_performance(10, 10, 3.0, 0.4);
        assert!((avg.enemy_health - 1.0).abs() < 0.3);
        assert!((avg.spawn_rate - 1.0).abs() < 0.3);

        // Very skilled player
        let pro = DifficultyMultiplier::from_performance(50, 2, 1.0, 0.8);
        assert!(pro.enemy_health > avg.enemy_health);
        assert!(pro.spawn_rate > avg.spawn_rate);
        assert!(pro.pickup_rate < avg.pickup_rate);

        // Struggling player
        let noob = DifficultyMultiplier::from_performance(3, 20, 8.0, 0.15);
        assert!(noob.enemy_health < avg.enemy_health);
        assert!(noob.pickup_rate > avg.pickup_rate);
    }

    #[test]
    fn test_difficulty_lerp() {
        let a = DifficultyMultiplier::default();
        let b = DifficultyMultiplier {
            enemy_health: 2.0,
            enemy_damage: 2.0,
            enemy_speed: 2.0,
            spawn_rate: 2.0,
            elite_chance: 2.0,
            pickup_rate: 2.0,
        };
        let mid = DifficultyMultiplier::lerp(&a, &b, 0.5);
        assert!((mid.enemy_health - 1.5).abs() < 0.01);
        assert!((mid.spawn_rate - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_director_initial_state() {
        let director = AIDirector::new(PacingConfig::default());
        assert_eq!(director.state(), IntensityState::Idle);
        assert!(director.current_intensity() < f32::EPSILON);
        assert!(!director.is_emergency());
        assert_eq!(director.peaks_completed(), 0);
    }

    #[test]
    fn test_director_transitions_from_idle() {
        let mut director = AIDirector::new(PacingConfig {
            idle: StateDuration::new(0.0, 0.1),
            ..Default::default()
        });
        let stats = PlayerStats::default();
        // Advance past the idle min+max duration
        let _ = director.update(0.2, &stats);
        assert_eq!(director.state(), IntensityState::BuildUp);
    }

    #[test]
    fn test_director_emergency_mode() {
        let mut director = AIDirector::new(PacingConfig::default());
        let stats = PlayerStats {
            health_fraction: 0.05,
            ammo_fraction: 0.05,
            ..Default::default()
        };
        let directives = director.update(0.1, &stats);
        assert!(director.is_emergency());
        assert!(
            directives
                .iter()
                .any(|d| matches!(d, SpawnDirective::HaltSpawning)),
            "Expected HaltSpawning directive in emergency"
        );
    }

    #[test]
    fn test_director_force_state() {
        let mut director = AIDirector::new(PacingConfig::default());
        director.force_state(IntensityState::Peak);
        assert_eq!(director.state(), IntensityState::Peak);
    }

    #[test]
    fn test_director_reset() {
        let mut director = AIDirector::new(PacingConfig::default());
        let stats = PlayerStats::default();
        for _ in 0..100 {
            director.update(0.1, &stats);
        }
        director.reset();
        assert_eq!(director.state(), IntensityState::Idle);
        assert!(director.current_intensity() < f32::EPSILON);
        assert_eq!(director.elapsed(), 0.0);
    }

    #[test]
    fn test_tension_tracker_history() {
        let mut tracker = TensionTracker {
            sample_interval: 0.1,
            ..Default::default()
        };
        tracker.add_tension(50.0);
        for _ in 0..10 {
            tracker.update(0.11);
        }
        assert!(tracker.average_tension() > 0.0);
        assert!(tracker.peak_tension() >= 50.0);
    }

    #[test]
    fn test_director_log() {
        let mut director = AIDirector::new(PacingConfig {
            idle: StateDuration::new(0.0, 0.1),
            ..Default::default()
        });
        let stats = PlayerStats::default();
        let _ = director.update(0.2, &stats);
        assert!(
            !director.log().events().is_empty(),
            "Director log should record state transitions"
        );
    }
}
