//! AI perception and sensing system.
//!
//! Manages what each AI agent can see, hear, smell, touch, or sense via
//! custom modalities. Each sense produces stimuli that are stored in a
//! per-agent perception memory, which fades over time to simulate an AI
//! that "forgets" when targets leave its sensory range.
//!
//! # Key concepts
//!
//! - **SenseType**: An enumeration of sensory modalities.
//! - **SightSense**: Field-of-view perception with distance falloff,
//!   peripheral vision, detection ramp-up, and line-of-sight raycast.
//! - **HearingSense**: Sound propagation with distance attenuation and
//!   wall-based occlusion.
//! - **AwarenessLevel**: Graduated awareness from Unaware through Engaged.
//! - **PerceptionMemory**: Time-limited memory of perceived entities.
//! - **PerceptionSystem**: Top-level manager that ticks all agents.

use std::collections::HashMap;

use glam::{Vec2, Vec3};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default field-of-view half-angle (degrees).
pub const DEFAULT_FOV_DEGREES: f32 = 110.0;

/// Default maximum sight distance (world units).
pub const DEFAULT_SIGHT_DISTANCE: f32 = 50.0;

/// Default hearing falloff exponent (inverse-square law would be 2.0).
pub const DEFAULT_HEARING_FALLOFF: f32 = 2.0;

/// Default memory duration before an unseen entity is forgotten (seconds).
pub const DEFAULT_MEMORY_DURATION: f32 = 10.0;

/// Rate at which awareness ramps up per second when a target is continuously
/// perceived (0..1 per second).
pub const DEFAULT_AWARENESS_RAMP_RATE: f32 = 0.5;

/// Rate at which awareness decays per second when a target is *not* perceived.
pub const DEFAULT_AWARENESS_DECAY_RATE: f32 = 0.25;

/// Peripheral vision multiplier for detection speed at the edge of FOV.
pub const PERIPHERAL_DETECTION_MULTIPLIER: f32 = 0.3;

/// Minimum wall thickness for occlusion calculations (world units).
pub const MIN_WALL_THICKNESS: f32 = 0.1;

/// Maximum number of stimuli a single perception component can hold.
pub const MAX_STIMULI_PER_AGENT: usize = 128;

/// Maximum number of concurrent sound events processed per tick.
pub const MAX_SOUND_EVENTS_PER_TICK: usize = 256;

/// Epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// SenseType
// ---------------------------------------------------------------------------

/// The different sensory modalities an AI agent can possess.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SenseType {
    /// Visual perception with field of view and line of sight.
    Sight,
    /// Auditory perception with distance attenuation and occlusion.
    Hearing,
    /// Olfactory perception (e.g., tracking scent trails).
    Smell,
    /// Tactile perception (close-range contact detection).
    Touch,
    /// Radar/sonar sweep (360-degree, no LOS required).
    Radar,
    /// Game-specific custom sense identified by a u32 key.
    Custom(u32),
}

impl SenseType {
    /// Returns `true` if this sense requires line-of-sight checks.
    pub fn requires_line_of_sight(&self) -> bool {
        matches!(self, SenseType::Sight)
    }

    /// Returns `true` if this sense is omnidirectional (ignores facing).
    pub fn is_omnidirectional(&self) -> bool {
        matches!(
            self,
            SenseType::Hearing | SenseType::Smell | SenseType::Radar | SenseType::Touch
        )
    }
}

// ---------------------------------------------------------------------------
// AwarenessLevel
// ---------------------------------------------------------------------------

/// Graduated awareness of a perceived target.
///
/// Awareness ramps up as a target is continuously perceived and decays when
/// the target is no longer sensed. Game logic can react differently at each
/// threshold — for example, transitioning from patrol to investigation to
/// combat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AwarenessLevel {
    /// Agent has no knowledge of the target.
    Unaware,
    /// Agent has noticed something but is not certain.
    Suspicious,
    /// Agent is confident a target is present and actively searching.
    Alerted,
    /// Agent is fully engaged with the target (e.g., in combat).
    Engaged,
}

impl AwarenessLevel {
    /// Returns the numeric threshold for this level (0.0..1.0).
    pub fn threshold(&self) -> f32 {
        match self {
            AwarenessLevel::Unaware => 0.0,
            AwarenessLevel::Suspicious => 0.25,
            AwarenessLevel::Alerted => 0.6,
            AwarenessLevel::Engaged => 0.9,
        }
    }

    /// Determine the awareness level from a raw awareness value.
    pub fn from_value(value: f32) -> Self {
        if value >= 0.9 {
            AwarenessLevel::Engaged
        } else if value >= 0.6 {
            AwarenessLevel::Alerted
        } else if value >= 0.25 {
            AwarenessLevel::Suspicious
        } else {
            AwarenessLevel::Unaware
        }
    }

    /// Returns the next higher awareness level, or self if already at max.
    pub fn escalate(&self) -> Self {
        match self {
            AwarenessLevel::Unaware => AwarenessLevel::Suspicious,
            AwarenessLevel::Suspicious => AwarenessLevel::Alerted,
            AwarenessLevel::Alerted => AwarenessLevel::Engaged,
            AwarenessLevel::Engaged => AwarenessLevel::Engaged,
        }
    }

    /// Returns the next lower awareness level, or self if already at min.
    pub fn deescalate(&self) -> Self {
        match self {
            AwarenessLevel::Engaged => AwarenessLevel::Alerted,
            AwarenessLevel::Alerted => AwarenessLevel::Suspicious,
            AwarenessLevel::Suspicious => AwarenessLevel::Unaware,
            AwarenessLevel::Unaware => AwarenessLevel::Unaware,
        }
    }
}

// ---------------------------------------------------------------------------
// LastKnownPosition
// ---------------------------------------------------------------------------

/// Records where a target was last perceived.
#[derive(Debug, Clone)]
pub struct LastKnownPosition {
    /// The world-space position where the target was last detected.
    pub position: Vec3,
    /// The velocity the target had when last seen (for extrapolation).
    pub velocity: Vec3,
    /// Simulation time when this position was recorded.
    pub timestamp: f64,
    /// Which sense produced this sighting.
    pub sense: SenseType,
}

impl LastKnownPosition {
    /// Create a new last-known-position record.
    pub fn new(position: Vec3, velocity: Vec3, timestamp: f64, sense: SenseType) -> Self {
        Self {
            position,
            velocity,
            timestamp,
            sense,
        }
    }

    /// Extrapolate the target's position forward from the last known state.
    pub fn extrapolate(&self, current_time: f64) -> Vec3 {
        let dt = (current_time - self.timestamp) as f32;
        self.position + self.velocity * dt
    }

    /// Returns how long ago this position was recorded relative to `now`.
    pub fn age(&self, now: f64) -> f64 {
        now - self.timestamp
    }
}

// ---------------------------------------------------------------------------
// SoundType
// ---------------------------------------------------------------------------

/// Categories of sounds that the hearing sense can recognize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SoundType {
    /// Quiet footstep (walking).
    Footstep,
    /// Running footstep (louder).
    FootstepRunning,
    /// Weapon discharge.
    Gunshot,
    /// Explosion.
    Explosion,
    /// Spoken voice / callout.
    Voice,
    /// Environmental ambient sound (wind, water, etc.).
    Ambient,
    /// Impact sound (door slam, breaking glass, etc.).
    Impact,
    /// Custom game-specific sound type.
    Custom(u32),
}

impl SoundType {
    /// Returns a base loudness multiplier for this sound type.
    pub fn base_loudness(&self) -> f32 {
        match self {
            SoundType::Footstep => 0.2,
            SoundType::FootstepRunning => 0.4,
            SoundType::Gunshot => 1.0,
            SoundType::Explosion => 1.5,
            SoundType::Voice => 0.5,
            SoundType::Ambient => 0.1,
            SoundType::Impact => 0.7,
            SoundType::Custom(_) => 0.5,
        }
    }

    /// Returns `true` if this sound type is considered high-priority.
    pub fn is_high_priority(&self) -> bool {
        matches!(self, SoundType::Gunshot | SoundType::Explosion)
    }
}

// ---------------------------------------------------------------------------
// SoundEvent
// ---------------------------------------------------------------------------

/// A sound event generated in the world.
///
/// Sound events are produced by entities and consumed by the hearing sense.
/// Loudness attenuates with distance and can be occluded by walls.
#[derive(Debug, Clone)]
pub struct SoundEvent {
    /// Unique identifier for this sound event.
    pub id: u64,
    /// World-space origin of the sound.
    pub position: Vec3,
    /// Base loudness at the source (0.0..∞). Typical range: 0.1–2.0.
    pub loudness: f32,
    /// Category of the sound.
    pub sound_type: SoundType,
    /// Entity that produced the sound (if any).
    pub source_entity: Option<u64>,
    /// Maximum distance at which the sound can be heard.
    pub max_range: f32,
    /// Timestamp when the sound was emitted.
    pub timestamp: f64,
}

impl SoundEvent {
    /// Create a new sound event.
    pub fn new(
        id: u64,
        position: Vec3,
        loudness: f32,
        sound_type: SoundType,
        source_entity: Option<u64>,
    ) -> Self {
        Self {
            id,
            position,
            loudness,
            sound_type,
            source_entity,
            max_range: loudness * 50.0,
            timestamp: 0.0,
        }
    }

    /// Set the timestamp on this event.
    pub fn with_timestamp(mut self, t: f64) -> Self {
        self.timestamp = t;
        self
    }

    /// Set a custom max range.
    pub fn with_max_range(mut self, range: f32) -> Self {
        self.max_range = range;
        self
    }

    /// Compute the loudness at a given distance from the source, using
    /// inverse-power falloff.
    pub fn loudness_at_distance(&self, distance: f32, falloff_exponent: f32) -> f32 {
        if distance <= 1.0 {
            return self.loudness;
        }
        self.loudness / distance.powf(falloff_exponent)
    }
}

// ---------------------------------------------------------------------------
// PerceivedSound
// ---------------------------------------------------------------------------

/// A sound as perceived by a listener, after distance and occlusion are applied.
#[derive(Debug, Clone)]
pub struct PerceivedSound {
    /// The original sound event.
    pub event_id: u64,
    /// World-space direction from listener toward the sound source.
    pub direction: Vec3,
    /// Perceived loudness after distance falloff and occlusion.
    pub perceived_loudness: f32,
    /// Distance from listener to source.
    pub distance: f32,
    /// Sound type for AI decision-making.
    pub sound_type: SoundType,
    /// Source entity if known.
    pub source_entity: Option<u64>,
}

// ---------------------------------------------------------------------------
// Wall (for occlusion / LOS checks)
// ---------------------------------------------------------------------------

/// A wall segment used for line-of-sight and sound occlusion checks.
///
/// Walls are axis-aligned or arbitrary line segments in the XZ plane
/// (assuming Y is up).
#[derive(Debug, Clone)]
pub struct OcclusionWall {
    /// Start point of the wall (XZ plane).
    pub start: Vec2,
    /// End point of the wall (XZ plane).
    pub end: Vec2,
    /// Thickness of the wall for sound occlusion (affects loudness reduction).
    pub thickness: f32,
    /// Sound absorption factor (0.0 = fully transparent, 1.0 = fully opaque).
    pub absorption: f32,
}

impl OcclusionWall {
    /// Create a new wall segment.
    pub fn new(start: Vec2, end: Vec2) -> Self {
        Self {
            start,
            end,
            thickness: 0.3,
            absorption: 0.8,
        }
    }

    /// Set custom thickness.
    pub fn with_thickness(mut self, thickness: f32) -> Self {
        self.thickness = thickness.max(MIN_WALL_THICKNESS);
        self
    }

    /// Set custom absorption.
    pub fn with_absorption(mut self, absorption: f32) -> Self {
        self.absorption = absorption.clamp(0.0, 1.0);
        self
    }

    /// Check if a ray from `a` to `b` (both in XZ) intersects this wall.
    /// Returns the parametric `t` along `a->b` if there is an intersection.
    pub fn ray_intersect(&self, a: Vec2, b: Vec2) -> Option<f32> {
        let dir = b - a;
        let wall_dir = self.end - self.start;
        let denom = dir.x * wall_dir.y - dir.y * wall_dir.x;

        if denom.abs() < EPSILON {
            return None; // Parallel lines.
        }

        let diff = self.start - a;
        let t = (diff.x * wall_dir.y - diff.y * wall_dir.x) / denom;
        let u = (diff.x * dir.y - diff.y * dir.x) / denom;

        if t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 {
            Some(t)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// SightSense
// ---------------------------------------------------------------------------

/// Configuration for visual perception.
///
/// Implements field-of-view perception with peripheral vision, detection
/// speed ramping, and line-of-sight obstruction. The detection model works
/// as follows:
///
/// 1. Check if the target is within `max_distance`.
/// 2. Check if the target falls within `fov_half_angle` of the perceiver's
///    forward direction.
/// 3. Cast a ray for line-of-sight; if blocked by a wall, the target
///    cannot be seen.
/// 4. Compute a detection multiplier based on the angle from center
///    (peripheral vision penalty) and distance.
/// 5. Ramp awareness by `detection_speed * multiplier * dt` each tick.
#[derive(Debug, Clone)]
pub struct SightSense {
    /// Half-angle of the field of view (radians).
    pub fov_half_angle: f32,
    /// Maximum sight distance.
    pub max_distance: f32,
    /// Speed at which awareness ramps up (units per second at center of FOV).
    pub detection_speed: f32,
    /// Multiplier applied at the edge of the FOV (0.0–1.0).
    pub peripheral_multiplier: f32,
    /// Height offset for the "eye" position above the agent's origin.
    pub eye_height: f32,
    /// Minimum detection distance (targets closer than this are always at full
    /// detection speed regardless of angle).
    pub close_range_override: f32,
}

impl SightSense {
    /// Create a new sight sense with default parameters.
    pub fn new() -> Self {
        Self {
            fov_half_angle: (DEFAULT_FOV_DEGREES * 0.5_f32).to_radians(),
            max_distance: DEFAULT_SIGHT_DISTANCE,
            detection_speed: DEFAULT_AWARENESS_RAMP_RATE,
            peripheral_multiplier: PERIPHERAL_DETECTION_MULTIPLIER,
            eye_height: 1.7,
            close_range_override: 2.0,
        }
    }

    /// Set the full field-of-view angle in degrees.
    pub fn with_fov_degrees(mut self, degrees: f32) -> Self {
        self.fov_half_angle = (degrees * 0.5).to_radians();
        self
    }

    /// Set the maximum sight distance.
    pub fn with_max_distance(mut self, d: f32) -> Self {
        self.max_distance = d;
        self
    }

    /// Set the detection speed.
    pub fn with_detection_speed(mut self, s: f32) -> Self {
        self.detection_speed = s;
        self
    }

    /// Set the peripheral vision multiplier.
    pub fn with_peripheral_multiplier(mut self, m: f32) -> Self {
        self.peripheral_multiplier = m.clamp(0.0, 1.0);
        self
    }

    /// Set the eye height offset.
    pub fn with_eye_height(mut self, h: f32) -> Self {
        self.eye_height = h;
        self
    }

    /// Check whether the perceiver can *potentially* see the target based on
    /// distance and field-of-view angle (ignoring line-of-sight obstruction).
    ///
    /// Returns the angle (radians) from the perceiver's forward direction to
    /// the target if the target is within FOV and range, or `None` otherwise.
    pub fn can_see(
        &self,
        perceiver_pos: Vec3,
        perceiver_forward: Vec3,
        target_pos: Vec3,
    ) -> Option<f32> {
        let to_target = target_pos - perceiver_pos;
        let dist = to_target.length();

        // Out of range?
        if dist > self.max_distance || dist < EPSILON {
            return None;
        }

        let dir = to_target / dist;
        let fwd = perceiver_forward.normalize();

        // Compute angle between forward and to_target.
        let dot = fwd.dot(dir).clamp(-1.0, 1.0);
        let angle = dot.acos();

        // Close-range override: always visible if very close.
        if dist <= self.close_range_override {
            return Some(angle);
        }

        if angle <= self.fov_half_angle {
            Some(angle)
        } else {
            None
        }
    }

    /// Compute the detection speed multiplier for a target at the given angle
    /// from the perceiver's forward direction and at the given distance.
    ///
    /// The multiplier is 1.0 at the center of the FOV and linearly
    /// interpolates down to `peripheral_multiplier` at the edge of the FOV.
    /// It also scales inversely with distance.
    pub fn detection_multiplier(&self, angle: f32, distance: f32) -> f32 {
        // Angular falloff: lerp from 1.0 at center to peripheral_multiplier
        // at the edge of the FOV.
        let angle_ratio = if self.fov_half_angle > EPSILON {
            (angle / self.fov_half_angle).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let angular = 1.0 - angle_ratio * (1.0 - self.peripheral_multiplier);

        // Distance falloff: linear from 1.0 at close range to 0.3 at max.
        let dist_ratio = if self.max_distance > EPSILON {
            (distance / self.max_distance).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let distance_factor = 1.0 - dist_ratio * 0.7;

        angular * distance_factor
    }

    /// Check line of sight between two points against a set of walls.
    /// Returns `true` if the line of sight is clear (no walls block the ray).
    pub fn has_line_of_sight(
        &self,
        from: Vec3,
        to: Vec3,
        walls: &[OcclusionWall],
    ) -> bool {
        let from_xz = Vec2::new(from.x, from.z);
        let to_xz = Vec2::new(to.x, to.z);

        for wall in walls {
            if wall.ray_intersect(from_xz, to_xz).is_some() {
                return false;
            }
        }
        true
    }

    /// Full visibility test: FOV + distance + LOS.
    /// Returns `Some((angle, distance, multiplier))` if the target is visible.
    pub fn test_visibility(
        &self,
        perceiver_pos: Vec3,
        perceiver_forward: Vec3,
        target_pos: Vec3,
        walls: &[OcclusionWall],
    ) -> Option<(f32, f32, f32)> {
        let angle = self.can_see(perceiver_pos, perceiver_forward, target_pos)?;
        let dist = (target_pos - perceiver_pos).length();

        // Eye-height adjusted LOS.
        let eye_pos = perceiver_pos + Vec3::new(0.0, self.eye_height, 0.0);
        if !self.has_line_of_sight(eye_pos, target_pos, walls) {
            return None;
        }

        let multiplier = self.detection_multiplier(angle, dist);
        Some((angle, dist, multiplier))
    }
}

impl Default for SightSense {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HearingSense
// ---------------------------------------------------------------------------

/// Configuration for auditory perception.
///
/// Sounds propagate from a source with inverse-power distance falloff and
/// are attenuated further by intervening walls (occlusion). Each wall's
/// absorption factor reduces the perceived loudness.
#[derive(Debug, Clone)]
pub struct HearingSense {
    /// Loudness threshold below which sounds are not perceived.
    pub hearing_threshold: f32,
    /// Exponent for distance falloff (2.0 = inverse square).
    pub falloff_exponent: f32,
    /// Maximum distance at which any sound can be heard, regardless of
    /// loudness.
    pub max_distance: f32,
    /// Multiplier for wall occlusion (higher = walls block more sound).
    pub occlusion_multiplier: f32,
}

impl HearingSense {
    /// Create a new hearing sense with default parameters.
    pub fn new() -> Self {
        Self {
            hearing_threshold: 0.01,
            falloff_exponent: DEFAULT_HEARING_FALLOFF,
            max_distance: 200.0,
            occlusion_multiplier: 1.0,
        }
    }

    /// Set the hearing threshold.
    pub fn with_threshold(mut self, t: f32) -> Self {
        self.hearing_threshold = t.max(0.0);
        self
    }

    /// Set the falloff exponent.
    pub fn with_falloff(mut self, f: f32) -> Self {
        self.falloff_exponent = f.max(0.0);
        self
    }

    /// Set the max hearing distance.
    pub fn with_max_distance(mut self, d: f32) -> Self {
        self.max_distance = d;
        self
    }

    /// Compute occlusion factor for a sound passing through walls between
    /// `listener` and `source`. Each wall reduces loudness by its
    /// absorption factor.
    pub fn compute_occlusion(
        &self,
        listener_pos: Vec3,
        source_pos: Vec3,
        walls: &[OcclusionWall],
    ) -> f32 {
        let from_xz = Vec2::new(listener_pos.x, listener_pos.z);
        let to_xz = Vec2::new(source_pos.x, source_pos.z);

        let mut factor = 1.0_f32;
        for wall in walls {
            if wall.ray_intersect(from_xz, to_xz).is_some() {
                factor *= 1.0 - (wall.absorption * self.occlusion_multiplier).clamp(0.0, 1.0);
            }
        }
        factor
    }

    /// Process a sound event for a listener at the given position.
    /// Returns `Some(PerceivedSound)` if the sound is loud enough to hear.
    pub fn hear_sound(
        &self,
        listener_pos: Vec3,
        sound: &SoundEvent,
        walls: &[OcclusionWall],
    ) -> Option<PerceivedSound> {
        let to_source = sound.position - listener_pos;
        let distance = to_source.length();

        // Beyond max hearing distance or the sound's own range.
        if distance > self.max_distance || distance > sound.max_range {
            return None;
        }

        // Compute base loudness at this distance.
        let base = sound.loudness_at_distance(distance, self.falloff_exponent);

        // Apply wall occlusion.
        let occlusion = self.compute_occlusion(listener_pos, sound.position, walls);
        let perceived = base * occlusion;

        if perceived < self.hearing_threshold {
            return None;
        }

        let direction = if distance > EPSILON {
            to_source / distance
        } else {
            Vec3::ZERO
        };

        Some(PerceivedSound {
            event_id: sound.id,
            direction,
            perceived_loudness: perceived,
            distance,
            sound_type: sound.sound_type,
            source_entity: sound.source_entity,
        })
    }
}

impl Default for HearingSense {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SmellSense
// ---------------------------------------------------------------------------

/// Configuration for olfactory perception.
#[derive(Debug, Clone)]
pub struct SmellSense {
    /// Maximum distance at which scents can be detected.
    pub max_distance: f32,
    /// How fast scent trails decay (multiplier per second, <1.0).
    pub decay_rate: f32,
    /// Minimum scent intensity to trigger detection.
    pub detection_threshold: f32,
}

impl SmellSense {
    /// Create with default values.
    pub fn new() -> Self {
        Self {
            max_distance: 30.0,
            decay_rate: 0.9,
            detection_threshold: 0.05,
        }
    }

    /// Test whether a scent at `source_pos` with the given `intensity`
    /// (already accounting for decay) can be detected by a sniffer at
    /// `perceiver_pos`.
    pub fn can_smell(
        &self,
        perceiver_pos: Vec3,
        source_pos: Vec3,
        intensity: f32,
    ) -> Option<f32> {
        let dist = (source_pos - perceiver_pos).length();
        if dist > self.max_distance {
            return None;
        }
        let falloff = if dist > 1.0 { 1.0 / dist } else { 1.0 };
        let perceived = intensity * falloff;
        if perceived >= self.detection_threshold {
            Some(perceived)
        } else {
            None
        }
    }
}

impl Default for SmellSense {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TouchSense
// ---------------------------------------------------------------------------

/// Configuration for tactile / proximity detection.
#[derive(Debug, Clone)]
pub struct TouchSense {
    /// Detection radius around the agent.
    pub radius: f32,
}

impl TouchSense {
    /// Create with default radius.
    pub fn new() -> Self {
        Self { radius: 1.5 }
    }

    /// Check if the target position is within touch range.
    pub fn can_touch(&self, perceiver_pos: Vec3, target_pos: Vec3) -> bool {
        (target_pos - perceiver_pos).length() <= self.radius
    }
}

impl Default for TouchSense {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RadarSense
// ---------------------------------------------------------------------------

/// Configuration for radar/sonar: 360-degree detection, ignoring LOS.
#[derive(Debug, Clone)]
pub struct RadarSense {
    /// Maximum detection range.
    pub range: f32,
    /// Sweep interval (seconds between pings). Targets are detected only
    /// on each sweep.
    pub sweep_interval: f32,
    /// Time accumulator for sweep timing.
    pub sweep_timer: f32,
}

impl RadarSense {
    /// Create with default range and 1-second sweep.
    pub fn new(range: f32) -> Self {
        Self {
            range,
            sweep_interval: 1.0,
            sweep_timer: 0.0,
        }
    }

    /// Advance the sweep timer. Returns `true` if a sweep occurs this tick.
    pub fn tick(&mut self, dt: f32) -> bool {
        self.sweep_timer += dt;
        if self.sweep_timer >= self.sweep_interval {
            self.sweep_timer -= self.sweep_interval;
            true
        } else {
            false
        }
    }

    /// Check if a target is within radar range.
    pub fn in_range(&self, perceiver_pos: Vec3, target_pos: Vec3) -> bool {
        (target_pos - perceiver_pos).length() <= self.range
    }
}

// ---------------------------------------------------------------------------
// StimulusSource
// ---------------------------------------------------------------------------

/// An entity that generates stimuli for the perception system.
///
/// Attach this to player characters, NPCs, or environmental objects that
/// should be detectable by AI senses.
#[derive(Debug, Clone)]
pub struct StimulusSource {
    /// Entity identifier.
    pub entity_id: u64,
    /// Current world position.
    pub position: Vec3,
    /// Current velocity (for last-known-position extrapolation).
    pub velocity: Vec3,
    /// Visual visibility multiplier (0.0 = invisible, 1.0 = normal).
    pub visual_visibility: f32,
    /// Sound emission multiplier (0.0 = silent, 1.0 = normal).
    pub noise_level: f32,
    /// Scent emission intensity.
    pub scent_intensity: f32,
    /// Team/faction identifier (perception may ignore same-team entities).
    pub team: u32,
    /// Whether this source is currently active.
    pub active: bool,
}

impl StimulusSource {
    /// Create a new stimulus source for an entity.
    pub fn new(entity_id: u64, position: Vec3) -> Self {
        Self {
            entity_id,
            position,
            velocity: Vec3::ZERO,
            visual_visibility: 1.0,
            noise_level: 1.0,
            scent_intensity: 0.0,
            team: 0,
            active: true,
        }
    }

    /// Set the entity's team.
    pub fn with_team(mut self, team: u32) -> Self {
        self.team = team;
        self
    }

    /// Set visual visibility.
    pub fn with_visibility(mut self, v: f32) -> Self {
        self.visual_visibility = v.clamp(0.0, 1.0);
        self
    }

    /// Update position and velocity each tick.
    pub fn update_transform(&mut self, position: Vec3, velocity: Vec3) {
        self.position = position;
        self.velocity = velocity;
    }
}

// ---------------------------------------------------------------------------
// PerceptionMemoryEntry
// ---------------------------------------------------------------------------

/// A single entry in an agent's perception memory: tracks one entity.
#[derive(Debug, Clone)]
pub struct PerceptionMemoryEntry {
    /// The perceived entity's ID.
    pub entity_id: u64,
    /// Last known position (and velocity) of the entity.
    pub last_known: LastKnownPosition,
    /// Current raw awareness value (0.0..1.0).
    pub awareness: f32,
    /// Discrete awareness level derived from the raw value.
    pub level: AwarenessLevel,
    /// Whether the entity is currently being perceived (this frame).
    pub currently_perceived: bool,
    /// Timestamp of first perception in this continuous sighting.
    pub first_seen_time: f64,
    /// How long until this entry is purged after losing perception (seconds).
    pub forget_after: f32,
    /// Time spent without perception (resets when perceived again).
    pub time_without_perception: f32,
    /// Which senses have detected this entity this frame.
    pub active_senses: Vec<SenseType>,
    /// Accumulated number of frames this entity has been perceived.
    pub perception_count: u32,
}

impl PerceptionMemoryEntry {
    /// Create a fresh memory entry for a newly detected entity.
    pub fn new(
        entity_id: u64,
        position: Vec3,
        velocity: Vec3,
        timestamp: f64,
        sense: SenseType,
        forget_after: f32,
    ) -> Self {
        Self {
            entity_id,
            last_known: LastKnownPosition::new(position, velocity, timestamp, sense),
            awareness: 0.0,
            level: AwarenessLevel::Unaware,
            currently_perceived: true,
            first_seen_time: timestamp,
            forget_after,
            time_without_perception: 0.0,
            active_senses: vec![sense],
            perception_count: 1,
        }
    }

    /// Update the entry when the entity is perceived again.
    pub fn update_perceived(
        &mut self,
        position: Vec3,
        velocity: Vec3,
        timestamp: f64,
        sense: SenseType,
    ) {
        self.last_known = LastKnownPosition::new(position, velocity, timestamp, sense);
        self.currently_perceived = true;
        self.time_without_perception = 0.0;
        self.perception_count += 1;
        if !self.active_senses.contains(&sense) {
            self.active_senses.push(sense);
        }
    }

    /// Ramp awareness up by the given delta (clamped to 0..1).
    pub fn ramp_awareness(&mut self, delta: f32) {
        self.awareness = (self.awareness + delta).clamp(0.0, 1.0);
        self.level = AwarenessLevel::from_value(self.awareness);
    }

    /// Decay awareness by the given delta (clamped to 0..1).
    pub fn decay_awareness(&mut self, delta: f32) {
        self.awareness = (self.awareness - delta).clamp(0.0, 1.0);
        self.level = AwarenessLevel::from_value(self.awareness);
    }

    /// Returns `true` if this entry should be purged from memory.
    pub fn should_forget(&self) -> bool {
        !self.currently_perceived
            && self.time_without_perception >= self.forget_after
            && self.awareness <= 0.0
    }
}

// ---------------------------------------------------------------------------
// PerceptionMemory
// ---------------------------------------------------------------------------

/// An agent's memory of all perceived entities.
///
/// Each entry represents knowledge of one entity, including when/where it
/// was last seen and the agent's current awareness level. Entries fade
/// over time when the entity is no longer perceived, and are eventually
/// purged.
#[derive(Debug, Clone)]
pub struct PerceptionMemory {
    /// All memory entries, keyed by entity ID.
    pub entries: HashMap<u64, PerceptionMemoryEntry>,
    /// Default time after which an unseen entity is forgotten.
    pub default_forget_duration: f32,
    /// Awareness ramp-up rate per second.
    pub ramp_rate: f32,
    /// Awareness decay rate per second.
    pub decay_rate: f32,
}

impl PerceptionMemory {
    /// Create a new empty perception memory.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            default_forget_duration: DEFAULT_MEMORY_DURATION,
            ramp_rate: DEFAULT_AWARENESS_RAMP_RATE,
            decay_rate: DEFAULT_AWARENESS_DECAY_RATE,
        }
    }

    /// Set the default forget duration.
    pub fn with_forget_duration(mut self, d: f32) -> Self {
        self.default_forget_duration = d;
        self
    }

    /// Set the awareness ramp rate.
    pub fn with_ramp_rate(mut self, r: f32) -> Self {
        self.ramp_rate = r;
        self
    }

    /// Set the awareness decay rate.
    pub fn with_decay_rate(mut self, r: f32) -> Self {
        self.decay_rate = r;
        self
    }

    /// Record that an entity was perceived this frame.
    pub fn record_perception(
        &mut self,
        entity_id: u64,
        position: Vec3,
        velocity: Vec3,
        timestamp: f64,
        sense: SenseType,
    ) {
        if let Some(entry) = self.entries.get_mut(&entity_id) {
            entry.update_perceived(position, velocity, timestamp, sense);
        } else {
            let entry = PerceptionMemoryEntry::new(
                entity_id,
                position,
                velocity,
                timestamp,
                sense,
                self.default_forget_duration,
            );
            self.entries.insert(entity_id, entry);
        }
    }

    /// Begin a new frame: mark all entries as not currently perceived.
    /// Call this at the start of each perception tick, before running senses.
    pub fn begin_frame(&mut self) {
        for entry in self.entries.values_mut() {
            entry.currently_perceived = false;
            entry.active_senses.clear();
        }
    }

    /// Update awareness for all entries: ramp up currently perceived,
    /// decay non-perceived, and purge forgotten entries.
    pub fn update(&mut self, dt: f32) {
        let ramp = self.ramp_rate;
        let decay = self.decay_rate;

        for entry in self.entries.values_mut() {
            if entry.currently_perceived {
                // Ramp awareness up.
                entry.ramp_awareness(ramp * dt);
            } else {
                // Decay awareness and track time without perception.
                entry.decay_awareness(decay * dt);
                entry.time_without_perception += dt;
            }
        }

        // Purge entries that should be forgotten.
        self.entries.retain(|_, entry| !entry.should_forget());
    }

    /// Update awareness with a custom detection multiplier (e.g., from
    /// peripheral vision or distance). Only affects currently perceived
    /// entries.
    pub fn update_with_multipliers(
        &mut self,
        dt: f32,
        multipliers: &HashMap<u64, f32>,
    ) {
        let ramp = self.ramp_rate;
        let decay = self.decay_rate;

        for entry in self.entries.values_mut() {
            if entry.currently_perceived {
                let mult = multipliers.get(&entry.entity_id).copied().unwrap_or(1.0);
                entry.ramp_awareness(ramp * mult * dt);
            } else {
                entry.decay_awareness(decay * dt);
                entry.time_without_perception += dt;
            }
        }

        self.entries.retain(|_, entry| !entry.should_forget());
    }

    /// Get the memory entry for a specific entity.
    pub fn get(&self, entity_id: u64) -> Option<&PerceptionMemoryEntry> {
        self.entries.get(&entity_id)
    }

    /// Get all entities at or above a given awareness level.
    pub fn entities_at_level(&self, min_level: AwarenessLevel) -> Vec<u64> {
        self.entries
            .iter()
            .filter(|(_, e)| e.level >= min_level)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get the most threatening entity (highest awareness).
    pub fn highest_awareness_entity(&self) -> Option<(u64, f32)> {
        self.entries
            .iter()
            .max_by(|a, b| a.1.awareness.partial_cmp(&b.1.awareness).unwrap())
            .map(|(id, e)| (*id, e.awareness))
    }

    /// Returns the number of entities currently in memory.
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Clear all memory entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns all entities currently being perceived this frame.
    pub fn currently_perceived(&self) -> Vec<u64> {
        self.entries
            .iter()
            .filter(|(_, e)| e.currently_perceived)
            .map(|(id, _)| *id)
            .collect()
    }
}

impl Default for PerceptionMemory {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PerceptionComponent
// ---------------------------------------------------------------------------

/// ECS component that gives an entity perception capabilities.
///
/// Attach this to AI agents to enable them to detect other entities through
/// one or more sensory modalities.
#[derive(Debug, Clone)]
pub struct PerceptionComponent {
    /// Entity that owns this perception component.
    pub entity_id: u64,
    /// World-space position of the perceiver.
    pub position: Vec3,
    /// Forward-facing direction (normalized).
    pub forward: Vec3,
    /// Team/faction (used to filter same-team entities).
    pub team: u32,
    /// Sight sense (if any).
    pub sight: Option<SightSense>,
    /// Hearing sense (if any).
    pub hearing: Option<HearingSense>,
    /// Smell sense (if any).
    pub smell: Option<SmellSense>,
    /// Touch sense (if any).
    pub touch: Option<TouchSense>,
    /// Radar sense (if any).
    pub radar: Option<RadarSense>,
    /// This agent's perception memory.
    pub memory: PerceptionMemory,
    /// Whether this component is active.
    pub active: bool,
}

impl PerceptionComponent {
    /// Create a new perception component for the given entity.
    pub fn new(entity_id: u64) -> Self {
        Self {
            entity_id,
            position: Vec3::ZERO,
            forward: Vec3::Z,
            team: 0,
            sight: None,
            hearing: None,
            smell: None,
            touch: None,
            radar: None,
            memory: PerceptionMemory::new(),
            active: true,
        }
    }

    /// Add a sight sense.
    pub fn with_sight(mut self, sight: SightSense) -> Self {
        self.sight = Some(sight);
        self
    }

    /// Add a hearing sense.
    pub fn with_hearing(mut self, hearing: HearingSense) -> Self {
        self.hearing = Some(hearing);
        self
    }

    /// Add a smell sense.
    pub fn with_smell(mut self, smell: SmellSense) -> Self {
        self.smell = Some(smell);
        self
    }

    /// Add a touch sense.
    pub fn with_touch(mut self, touch: TouchSense) -> Self {
        self.touch = Some(touch);
        self
    }

    /// Add a radar sense.
    pub fn with_radar(mut self, radar: RadarSense) -> Self {
        self.radar = Some(radar);
        self
    }

    /// Set the team.
    pub fn with_team(mut self, team: u32) -> Self {
        self.team = team;
        self
    }

    /// Set the memory parameters.
    pub fn with_memory(mut self, memory: PerceptionMemory) -> Self {
        self.memory = memory;
        self
    }

    /// Update the perceiver's transform.
    pub fn update_transform(&mut self, position: Vec3, forward: Vec3) {
        self.position = position;
        self.forward = forward.normalize();
    }

    /// Returns which sense types are enabled on this component.
    pub fn enabled_senses(&self) -> Vec<SenseType> {
        let mut senses = Vec::new();
        if self.sight.is_some() {
            senses.push(SenseType::Sight);
        }
        if self.hearing.is_some() {
            senses.push(SenseType::Hearing);
        }
        if self.smell.is_some() {
            senses.push(SenseType::Smell);
        }
        if self.touch.is_some() {
            senses.push(SenseType::Touch);
        }
        if self.radar.is_some() {
            senses.push(SenseType::Radar);
        }
        senses
    }
}

// ---------------------------------------------------------------------------
// PerceptionConfig
// ---------------------------------------------------------------------------

/// Global configuration for the perception system.
#[derive(Debug, Clone)]
pub struct PerceptionConfig {
    /// Whether same-team entities are filtered out of perception results.
    pub ignore_same_team: bool,
    /// Maximum number of sound events processed per tick.
    pub max_sounds_per_tick: usize,
    /// Whether to enable debug visualization hooks.
    pub debug_draw: bool,
    /// Global awareness ramp multiplier (applies on top of per-sense speeds).
    pub awareness_multiplier: f32,
    /// Maximum number of stimuli sources to process per agent per tick.
    pub max_stimuli_per_agent: usize,
}

impl PerceptionConfig {
    /// Create default configuration.
    pub fn new() -> Self {
        Self {
            ignore_same_team: true,
            max_sounds_per_tick: MAX_SOUND_EVENTS_PER_TICK,
            debug_draw: false,
            awareness_multiplier: 1.0,
            max_stimuli_per_agent: MAX_STIMULI_PER_AGENT,
        }
    }
}

impl Default for PerceptionConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PerceptionEvent
// ---------------------------------------------------------------------------

/// Events emitted by the perception system for game logic to react to.
#[derive(Debug, Clone)]
pub enum PerceptionEvent {
    /// An entity was first detected by a perceiver.
    FirstDetection {
        perceiver: u64,
        target: u64,
        sense: SenseType,
        position: Vec3,
    },
    /// An entity's awareness level changed.
    AwarenessChanged {
        perceiver: u64,
        target: u64,
        old_level: AwarenessLevel,
        new_level: AwarenessLevel,
    },
    /// An entity was lost from perception (but may still be in memory).
    LostPerception {
        perceiver: u64,
        target: u64,
        last_position: Vec3,
    },
    /// An entity was completely forgotten.
    Forgotten {
        perceiver: u64,
        target: u64,
    },
    /// A sound was heard.
    SoundHeard {
        listener: u64,
        sound_event_id: u64,
        direction: Vec3,
        loudness: f32,
    },
}

// ---------------------------------------------------------------------------
// PerceptionSystem
// ---------------------------------------------------------------------------

/// Top-level perception manager.
///
/// Runs all perception checks each tick, updates awareness in each agent's
/// memory, and emits perception events.
///
/// # Usage
///
/// ```ignore
/// let mut system = PerceptionSystem::new(PerceptionConfig::default());
///
/// // Register agents and sources...
/// system.add_perceiver(agent_component);
/// system.add_source(player_source);
///
/// // Each tick:
/// system.update(dt, current_time, &walls, &sound_events);
/// for event in system.drain_events() {
///     // React to perception events...
/// }
/// ```
pub struct PerceptionSystem {
    /// Configuration.
    pub config: PerceptionConfig,
    /// All perceiving agents, keyed by entity ID.
    pub perceivers: HashMap<u64, PerceptionComponent>,
    /// All stimulus sources, keyed by entity ID.
    pub sources: HashMap<u64, StimulusSource>,
    /// Events generated during the last update.
    events: Vec<PerceptionEvent>,
    /// Monotonically increasing event counter for sound IDs.
    next_sound_id: u64,
}

impl PerceptionSystem {
    /// Create a new perception system.
    pub fn new(config: PerceptionConfig) -> Self {
        Self {
            config,
            perceivers: HashMap::new(),
            sources: HashMap::new(),
            events: Vec::new(),
            next_sound_id: 1,
        }
    }

    /// Add or replace a perceiving agent.
    pub fn add_perceiver(&mut self, component: PerceptionComponent) {
        self.perceivers.insert(component.entity_id, component);
    }

    /// Remove a perceiver by entity ID.
    pub fn remove_perceiver(&mut self, entity_id: u64) -> Option<PerceptionComponent> {
        self.perceivers.remove(&entity_id)
    }

    /// Add or replace a stimulus source.
    pub fn add_source(&mut self, source: StimulusSource) {
        self.sources.insert(source.entity_id, source);
    }

    /// Remove a stimulus source by entity ID.
    pub fn remove_source(&mut self, entity_id: u64) -> Option<StimulusSource> {
        self.sources.remove(&entity_id)
    }

    /// Get a perceiver by entity ID.
    pub fn get_perceiver(&self, entity_id: u64) -> Option<&PerceptionComponent> {
        self.perceivers.get(&entity_id)
    }

    /// Get a mutable reference to a perceiver.
    pub fn get_perceiver_mut(&mut self, entity_id: u64) -> Option<&mut PerceptionComponent> {
        self.perceivers.get_mut(&entity_id)
    }

    /// Allocate a new unique sound event ID.
    pub fn next_sound_id(&mut self) -> u64 {
        let id = self.next_sound_id;
        self.next_sound_id += 1;
        id
    }

    /// Drain all perception events generated during the last update.
    pub fn drain_events(&mut self) -> Vec<PerceptionEvent> {
        std::mem::take(&mut self.events)
    }

    /// Run a full perception update.
    ///
    /// Processes sight, hearing, smell, touch, and radar for every perceiver
    /// against every active stimulus source. Updates awareness levels in
    /// each agent's memory and emits events for state transitions.
    pub fn update(
        &mut self,
        dt: f32,
        current_time: f64,
        walls: &[OcclusionWall],
        sound_events: &[SoundEvent],
    ) {
        // Collect source data so we don't borrow self mutably and immutably.
        let sources: Vec<StimulusSource> = self
            .sources
            .values()
            .filter(|s| s.active)
            .cloned()
            .collect();

        // Limit sound events to configured max.
        let max_sounds = self.config.max_sounds_per_tick;
        let sounds: &[SoundEvent] = if sound_events.len() > max_sounds {
            &sound_events[..max_sounds]
        } else {
            sound_events
        };

        let ignore_team = self.config.ignore_same_team;
        let awareness_mult = self.config.awareness_multiplier;

        // Process each perceiver.
        let perceiver_ids: Vec<u64> = self.perceivers.keys().copied().collect();
        for pid in perceiver_ids {
            let perceiver = match self.perceivers.get_mut(&pid) {
                Some(p) if p.active => p,
                _ => continue,
            };

            // Begin frame: mark all memory entries as not currently perceived.
            perceiver.memory.begin_frame();

            // Collect detection multipliers for awareness ramping.
            let mut multipliers: HashMap<u64, f32> = HashMap::new();

            // --- Sight sense ---
            if let Some(ref sight) = perceiver.sight {
                let sight = sight.clone();
                let p_pos = perceiver.position;
                let p_fwd = perceiver.forward;

                for source in &sources {
                    if source.entity_id == pid {
                        continue;
                    }
                    if ignore_team && source.team == perceiver.team && source.team != 0 {
                        continue;
                    }
                    if source.visual_visibility <= 0.0 {
                        continue;
                    }

                    if let Some((angle, dist, mult)) =
                        sight.test_visibility(p_pos, p_fwd, source.position, walls)
                    {
                        let effective_mult = mult * source.visual_visibility * awareness_mult;
                        let is_new = !perceiver.memory.entries.contains_key(&source.entity_id);

                        perceiver.memory.record_perception(
                            source.entity_id,
                            source.position,
                            source.velocity,
                            current_time,
                            SenseType::Sight,
                        );

                        multipliers
                            .entry(source.entity_id)
                            .and_modify(|m| *m = m.max(effective_mult))
                            .or_insert(effective_mult);

                        if is_new {
                            self.events.push(PerceptionEvent::FirstDetection {
                                perceiver: pid,
                                target: source.entity_id,
                                sense: SenseType::Sight,
                                position: source.position,
                            });
                        }

                        let _ = (angle, dist); // Used in visibility test.
                    }
                }
            }

            // --- Hearing sense ---
            if let Some(ref hearing) = perceiver.hearing {
                let hearing = hearing.clone();
                let l_pos = perceiver.position;

                for sound in sounds {
                    if let Some(perceived) = hearing.hear_sound(l_pos, sound, walls) {
                        // Attribute to source entity if known.
                        if let Some(src_id) = perceived.source_entity {
                            if src_id == pid {
                                continue;
                            }
                            if ignore_team {
                                if let Some(src) = self.sources.get(&src_id) {
                                    if src.team == perceiver.team && src.team != 0 {
                                        continue;
                                    }
                                }
                            }

                            let is_new = !perceiver.memory.entries.contains_key(&src_id);
                            perceiver.memory.record_perception(
                                src_id,
                                sound.position,
                                Vec3::ZERO,
                                current_time,
                                SenseType::Hearing,
                            );

                            // Hearing gives a partial awareness boost proportional
                            // to perceived loudness.
                            let hearing_mult =
                                (perceived.perceived_loudness * 0.5).min(1.0) * awareness_mult;
                            multipliers
                                .entry(src_id)
                                .and_modify(|m| *m = m.max(hearing_mult))
                                .or_insert(hearing_mult);

                            if is_new {
                                self.events.push(PerceptionEvent::FirstDetection {
                                    perceiver: pid,
                                    target: src_id,
                                    sense: SenseType::Hearing,
                                    position: sound.position,
                                });
                            }
                        }

                        self.events.push(PerceptionEvent::SoundHeard {
                            listener: pid,
                            sound_event_id: perceived.event_id,
                            direction: perceived.direction,
                            loudness: perceived.perceived_loudness,
                        });
                    }
                }
            }

            // --- Smell sense ---
            if let Some(ref smell) = perceiver.smell {
                let smell = smell.clone();
                let p_pos = perceiver.position;

                for source in &sources {
                    if source.entity_id == pid {
                        continue;
                    }
                    if ignore_team && source.team == perceiver.team && source.team != 0 {
                        continue;
                    }
                    if source.scent_intensity <= 0.0 {
                        continue;
                    }

                    if let Some(intensity) =
                        smell.can_smell(p_pos, source.position, source.scent_intensity)
                    {
                        let is_new = !perceiver.memory.entries.contains_key(&source.entity_id);
                        perceiver.memory.record_perception(
                            source.entity_id,
                            source.position,
                            source.velocity,
                            current_time,
                            SenseType::Smell,
                        );

                        let smell_mult = (intensity * 0.3).min(1.0) * awareness_mult;
                        multipliers
                            .entry(source.entity_id)
                            .and_modify(|m| *m = m.max(smell_mult))
                            .or_insert(smell_mult);

                        if is_new {
                            self.events.push(PerceptionEvent::FirstDetection {
                                perceiver: pid,
                                target: source.entity_id,
                                sense: SenseType::Smell,
                                position: source.position,
                            });
                        }
                    }
                }
            }

            // --- Touch sense ---
            if let Some(ref touch) = perceiver.touch {
                let touch = touch.clone();
                let p_pos = perceiver.position;

                for source in &sources {
                    if source.entity_id == pid {
                        continue;
                    }
                    if ignore_team && source.team == perceiver.team && source.team != 0 {
                        continue;
                    }

                    if touch.can_touch(p_pos, source.position) {
                        let is_new = !perceiver.memory.entries.contains_key(&source.entity_id);
                        perceiver.memory.record_perception(
                            source.entity_id,
                            source.position,
                            source.velocity,
                            current_time,
                            SenseType::Touch,
                        );

                        // Touch gives full awareness immediately.
                        multipliers
                            .entry(source.entity_id)
                            .and_modify(|m| *m = m.max(2.0))
                            .or_insert(2.0);

                        if is_new {
                            self.events.push(PerceptionEvent::FirstDetection {
                                perceiver: pid,
                                target: source.entity_id,
                                sense: SenseType::Touch,
                                position: source.position,
                            });
                        }
                    }
                }
            }

            // --- Radar sense ---
            if let Some(ref mut radar) = perceiver.radar {
                if radar.tick(dt) {
                    let p_pos = perceiver.position;
                    for source in &sources {
                        if source.entity_id == pid {
                            continue;
                        }
                        if ignore_team && source.team == perceiver.team && source.team != 0 {
                            continue;
                        }
                        if radar.in_range(p_pos, source.position) {
                            let is_new =
                                !perceiver.memory.entries.contains_key(&source.entity_id);
                            perceiver.memory.record_perception(
                                source.entity_id,
                                source.position,
                                source.velocity,
                                current_time,
                                SenseType::Radar,
                            );

                            let radar_mult = 0.8 * awareness_mult;
                            multipliers
                                .entry(source.entity_id)
                                .and_modify(|m| *m = m.max(radar_mult))
                                .or_insert(radar_mult);

                            if is_new {
                                self.events.push(PerceptionEvent::FirstDetection {
                                    perceiver: pid,
                                    target: source.entity_id,
                                    sense: SenseType::Radar,
                                    position: source.position,
                                });
                            }
                        }
                    }
                }
            }

            // --- Update awareness with multipliers ---
            // Capture old awareness levels for event generation.
            let old_levels: HashMap<u64, AwarenessLevel> = perceiver
                .memory
                .entries
                .iter()
                .map(|(id, e)| (*id, e.level))
                .collect();

            perceiver.memory.update_with_multipliers(dt, &multipliers);

            // Generate awareness-change and lost-perception events.
            for (entity_id, entry) in &perceiver.memory.entries {
                if let Some(&old_level) = old_levels.get(entity_id) {
                    if entry.level != old_level {
                        self.events.push(PerceptionEvent::AwarenessChanged {
                            perceiver: pid,
                            target: *entity_id,
                            old_level,
                            new_level: entry.level,
                        });
                    }
                }

                if !entry.currently_perceived {
                    if let Some(&old_level) = old_levels.get(entity_id) {
                        // Only emit LostPerception when transitioning from perceived to not.
                        if old_level >= AwarenessLevel::Suspicious
                            && entry.time_without_perception < dt * 1.5
                        {
                            self.events.push(PerceptionEvent::LostPerception {
                                perceiver: pid,
                                target: *entity_id,
                                last_position: entry.last_known.position,
                            });
                        }
                    }
                }
            }

            // Check for entities that were forgotten (were in old_levels but
            // no longer in entries after update).
            for (entity_id, _) in &old_levels {
                if !perceiver.memory.entries.contains_key(entity_id) {
                    self.events.push(PerceptionEvent::Forgotten {
                        perceiver: pid,
                        target: *entity_id,
                    });
                }
            }
        }
    }

    /// Get the awareness level of `perceiver` toward `target`.
    pub fn get_awareness(
        &self,
        perceiver: u64,
        target: u64,
    ) -> Option<AwarenessLevel> {
        self.perceivers
            .get(&perceiver)
            .and_then(|p| p.memory.get(target))
            .map(|e| e.level)
    }

    /// Get the last known position of `target` as seen by `perceiver`.
    pub fn get_last_known_position(
        &self,
        perceiver: u64,
        target: u64,
    ) -> Option<&LastKnownPosition> {
        self.perceivers
            .get(&perceiver)
            .and_then(|p| p.memory.get(target))
            .map(|e| &e.last_known)
    }

    /// Get all targets that `perceiver` is currently aware of at or above
    /// the given level.
    pub fn get_known_targets(
        &self,
        perceiver: u64,
        min_level: AwarenessLevel,
    ) -> Vec<u64> {
        self.perceivers
            .get(&perceiver)
            .map(|p| p.memory.entities_at_level(min_level))
            .unwrap_or_default()
    }

    /// Return the total number of active perceivers.
    pub fn perceiver_count(&self) -> usize {
        self.perceivers.len()
    }

    /// Return the total number of active sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Clear all perceivers, sources, and events.
    pub fn clear(&mut self) {
        self.perceivers.clear();
        self.sources.clear();
        self.events.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_awareness_level_from_value() {
        assert_eq!(AwarenessLevel::from_value(0.0), AwarenessLevel::Unaware);
        assert_eq!(AwarenessLevel::from_value(0.24), AwarenessLevel::Unaware);
        assert_eq!(AwarenessLevel::from_value(0.25), AwarenessLevel::Suspicious);
        assert_eq!(AwarenessLevel::from_value(0.59), AwarenessLevel::Suspicious);
        assert_eq!(AwarenessLevel::from_value(0.6), AwarenessLevel::Alerted);
        assert_eq!(AwarenessLevel::from_value(0.89), AwarenessLevel::Alerted);
        assert_eq!(AwarenessLevel::from_value(0.9), AwarenessLevel::Engaged);
        assert_eq!(AwarenessLevel::from_value(1.0), AwarenessLevel::Engaged);
    }

    #[test]
    fn test_sight_can_see_in_fov() {
        let sight = SightSense::new().with_fov_degrees(90.0);
        let pos = Vec3::ZERO;
        let fwd = Vec3::Z;
        let target = Vec3::new(0.0, 0.0, 10.0);

        assert!(sight.can_see(pos, fwd, target).is_some());
    }

    #[test]
    fn test_sight_cannot_see_behind() {
        let sight = SightSense::new().with_fov_degrees(90.0);
        let pos = Vec3::ZERO;
        let fwd = Vec3::Z;
        let target = Vec3::new(0.0, 0.0, -10.0);

        assert!(sight.can_see(pos, fwd, target).is_none());
    }

    #[test]
    fn test_sight_cannot_see_beyond_range() {
        let sight = SightSense::new().with_max_distance(20.0);
        let pos = Vec3::ZERO;
        let fwd = Vec3::Z;
        let target = Vec3::new(0.0, 0.0, 30.0);

        assert!(sight.can_see(pos, fwd, target).is_none());
    }

    #[test]
    fn test_sight_close_range_override() {
        let sight = SightSense::new().with_fov_degrees(10.0);
        let pos = Vec3::ZERO;
        let fwd = Vec3::Z;
        // Target is behind but very close (within close_range_override).
        let target = Vec3::new(0.0, 0.0, -1.0);

        assert!(sight.can_see(pos, fwd, target).is_some());
    }

    #[test]
    fn test_sight_line_of_sight_blocked() {
        let sight = SightSense::new();
        let from = Vec3::new(0.0, 1.7, 0.0);
        let to = Vec3::new(0.0, 1.7, 20.0);
        let walls = vec![OcclusionWall::new(
            Vec2::new(-5.0, 10.0),
            Vec2::new(5.0, 10.0),
        )];

        assert!(!sight.has_line_of_sight(from, to, &walls));
    }

    #[test]
    fn test_sight_line_of_sight_clear() {
        let sight = SightSense::new();
        let from = Vec3::new(0.0, 1.7, 0.0);
        let to = Vec3::new(0.0, 1.7, 5.0);
        let walls = vec![OcclusionWall::new(
            Vec2::new(-5.0, 10.0),
            Vec2::new(5.0, 10.0),
        )];

        assert!(sight.has_line_of_sight(from, to, &walls));
    }

    #[test]
    fn test_detection_multiplier_center() {
        let sight = SightSense::new();
        let mult = sight.detection_multiplier(0.0, 1.0);
        assert!((mult - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_detection_multiplier_edge() {
        let sight = SightSense::new();
        let mult = sight.detection_multiplier(sight.fov_half_angle, 1.0);
        assert!(mult < 0.5);
    }

    #[test]
    fn test_sound_loudness_falloff() {
        let event = SoundEvent::new(1, Vec3::ZERO, 1.0, SoundType::Gunshot, None);
        let loud_close = event.loudness_at_distance(1.0, 2.0);
        let loud_far = event.loudness_at_distance(10.0, 2.0);

        assert!(loud_close > loud_far);
        assert!((loud_close - 1.0).abs() < 0.01);
        assert!((loud_far - 0.01).abs() < 0.01);
    }

    #[test]
    fn test_hearing_occlusion() {
        let hearing = HearingSense::new();
        let listener = Vec3::ZERO;
        let source = Vec3::new(0.0, 0.0, 20.0);
        let wall = OcclusionWall::new(Vec2::new(-5.0, 10.0), Vec2::new(5.0, 10.0))
            .with_absorption(0.5);

        let factor = hearing.compute_occlusion(listener, source, &[wall]);
        assert!(factor < 1.0);
        assert!(factor > 0.0);
    }

    #[test]
    fn test_hearing_hear_sound() {
        let hearing = HearingSense::new();
        let listener = Vec3::ZERO;
        let sound = SoundEvent::new(1, Vec3::new(0.0, 0.0, 5.0), 1.0, SoundType::Gunshot, Some(42));

        let result = hearing.hear_sound(listener, &sound, &[]);
        assert!(result.is_some());
        let perceived = result.unwrap();
        assert_eq!(perceived.source_entity, Some(42));
        assert!(perceived.perceived_loudness > 0.0);
    }

    #[test]
    fn test_hearing_too_far() {
        let hearing = HearingSense::new().with_max_distance(10.0);
        let listener = Vec3::ZERO;
        let sound = SoundEvent::new(1, Vec3::new(0.0, 0.0, 50.0), 0.1, SoundType::Footstep, None);

        assert!(hearing.hear_sound(listener, &sound, &[]).is_none());
    }

    #[test]
    fn test_perception_memory_ramp_and_decay() {
        let mut memory = PerceptionMemory::new()
            .with_ramp_rate(1.0)
            .with_decay_rate(0.5);

        // Record perception.
        memory.record_perception(42, Vec3::ZERO, Vec3::ZERO, 0.0, SenseType::Sight);
        memory.update(1.0);

        let entry = memory.get(42).unwrap();
        assert!(entry.awareness >= 0.9);

        // Stop perceiving.
        memory.begin_frame();
        memory.update(1.0);

        let entry = memory.get(42).unwrap();
        assert!(entry.awareness < 0.9);
    }

    #[test]
    fn test_perception_memory_forget() {
        let mut memory = PerceptionMemory::new()
            .with_ramp_rate(1.0)
            .with_decay_rate(10.0)
            .with_forget_duration(0.1);

        memory.record_perception(42, Vec3::ZERO, Vec3::ZERO, 0.0, SenseType::Sight);
        memory.update(0.1);

        // Stop perceiving; after enough time it should be forgotten.
        memory.begin_frame();
        memory.update(1.0); // Decay fast, time_without_perception exceeds forget_after.

        assert!(memory.get(42).is_none());
    }

    #[test]
    fn test_perception_system_full() {
        let mut system = PerceptionSystem::new(PerceptionConfig::new());

        let agent = PerceptionComponent::new(1)
            .with_sight(SightSense::new().with_fov_degrees(180.0).with_max_distance(100.0))
            .with_hearing(HearingSense::new())
            .with_team(1);
        let mut agent = agent;
        agent.position = Vec3::ZERO;
        agent.forward = Vec3::Z;
        system.add_perceiver(agent);

        let source = StimulusSource::new(2, Vec3::new(0.0, 0.0, 10.0)).with_team(2);
        system.add_source(source);

        system.update(0.1, 0.1, &[], &[]);

        let awareness = system.get_awareness(1, 2);
        assert!(awareness.is_some());
    }

    #[test]
    fn test_wall_ray_intersect() {
        let wall = OcclusionWall::new(Vec2::new(-5.0, 10.0), Vec2::new(5.0, 10.0));

        // Ray crossing the wall.
        let hit = wall.ray_intersect(Vec2::new(0.0, 0.0), Vec2::new(0.0, 20.0));
        assert!(hit.is_some());

        // Ray not crossing the wall.
        let miss = wall.ray_intersect(Vec2::new(0.0, 0.0), Vec2::new(10.0, 5.0));
        assert!(miss.is_none());
    }

    #[test]
    fn test_smell_sense() {
        let smell = SmellSense::new();
        let perceiver = Vec3::ZERO;

        // Close and strong scent.
        let result = smell.can_smell(perceiver, Vec3::new(0.0, 0.0, 5.0), 1.0);
        assert!(result.is_some());

        // Too far.
        let result = smell.can_smell(perceiver, Vec3::new(0.0, 0.0, 100.0), 0.1);
        assert!(result.is_none());
    }

    #[test]
    fn test_touch_sense() {
        let touch = TouchSense::new();
        assert!(touch.can_touch(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0)));
        assert!(!touch.can_touch(Vec3::ZERO, Vec3::new(0.0, 0.0, 5.0)));
    }

    #[test]
    fn test_radar_sense_sweep() {
        let mut radar = RadarSense::new(50.0);
        radar.sweep_interval = 0.5;

        assert!(!radar.tick(0.3));
        assert!(radar.tick(0.3)); // 0.6 >= 0.5
        assert!(!radar.tick(0.3));
    }

    #[test]
    fn test_last_known_position_extrapolation() {
        let lkp = LastKnownPosition::new(
            Vec3::new(10.0, 0.0, 10.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.0,
            SenseType::Sight,
        );

        let extrapolated = lkp.extrapolate(3.0);
        assert!((extrapolated.x - 12.0).abs() < 0.01);
        assert!((extrapolated.z - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_ignore_same_team() {
        let mut system = PerceptionSystem::new(PerceptionConfig::new());

        let agent = PerceptionComponent::new(1)
            .with_sight(SightSense::new().with_fov_degrees(180.0))
            .with_team(1);
        let mut agent = agent;
        agent.position = Vec3::ZERO;
        agent.forward = Vec3::Z;
        system.add_perceiver(agent);

        // Same team source should be ignored.
        let source = StimulusSource::new(2, Vec3::new(0.0, 0.0, 10.0)).with_team(1);
        system.add_source(source);

        system.update(0.1, 0.1, &[], &[]);
        assert!(system.get_awareness(1, 2).is_none());
    }
}
