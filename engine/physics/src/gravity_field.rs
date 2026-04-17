//! Custom gravity field system for non-uniform gravitational environments.
//!
//! Provides:
//! - **Gravity zones**: spherical planets, directional overrides, cylindrical zones
//! - **Gravity wells**: point-source attractors with configurable falloff
//! - **Anti-gravity volumes**: regions that negate or reverse gravity
//! - **Gravity transitions**: smooth blending between overlapping gravity zones
//! - **Radial gravity**: spherical world gravity where "down" points toward center
//! - **Priority-based resolution**: when multiple zones overlap, highest priority wins
//!   (or zones can blend according to their blend weights)
//! - **ECS integration**: `GravityFieldComponent` and `GravityFieldSystem`
//!
//! # Design
//!
//! The world can contain an arbitrary number of [`GravityZone`]s, each with a shape,
//! a gravity direction/magnitude strategy, and blending parameters. Bodies query
//! the [`GravityFieldManager`] for the effective gravity at their position, and
//! the manager resolves overlaps according to priority and blend settings.

use glam::{Vec3, Quat, Mat3};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default gravity magnitude (Earth-like), m/s^2.
pub const DEFAULT_GRAVITY: f32 = 9.81;
/// Default gravity direction (negative Y axis).
pub const DEFAULT_GRAVITY_DIR: Vec3 = Vec3::new(0.0, -1.0, 0.0);
/// Maximum number of gravity zones.
pub const MAX_GRAVITY_ZONES: usize = 256;
/// Maximum number of gravity wells.
pub const MAX_GRAVITY_WELLS: usize = 128;
/// Minimum blend weight to be considered active.
pub const MIN_BLEND_WEIGHT: f32 = 0.001;
/// Small epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-7;
/// Pi constant.
const PI: f32 = std::f32::consts::PI;
/// Maximum blend radius ratio (transition_radius / zone_radius).
pub const MAX_BLEND_RATIO: f32 = 2.0;
/// Default transition width for gravity zone edges (meters).
pub const DEFAULT_TRANSITION_WIDTH: f32 = 5.0;
/// Maximum number of overlapping zones to consider for blending.
pub const MAX_OVERLAP_ZONES: usize = 8;
/// Gravitational constant for mass-based gravity wells (scaled for game use).
pub const GRAVITATIONAL_CONSTANT: f32 = 6.674e-2;
/// Minimum distance for gravity well calculations to prevent singularities.
pub const MIN_WELL_DISTANCE: f32 = 0.5;

// ---------------------------------------------------------------------------
// GravityZoneId
// ---------------------------------------------------------------------------

/// Unique identifier for a gravity zone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GravityZoneId(pub u64);

impl GravityZoneId {
    /// Create a new zone ID.
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

// ---------------------------------------------------------------------------
// GravityWellId
// ---------------------------------------------------------------------------

/// Unique identifier for a gravity well.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GravityWellId(pub u64);

impl GravityWellId {
    /// Create a new well ID.
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

// ---------------------------------------------------------------------------
// GravityZoneShape
// ---------------------------------------------------------------------------

/// Shape of a gravity zone volume.
#[derive(Debug, Clone)]
pub enum GravityZoneShape {
    /// Spherical zone centered at a point with a radius.
    Sphere {
        center: Vec3,
        radius: f32,
    },
    /// Axis-aligned box zone.
    Box {
        min: Vec3,
        max: Vec3,
    },
    /// Oriented box zone with rotation.
    OrientedBox {
        center: Vec3,
        half_extents: Vec3,
        rotation: Quat,
    },
    /// Cylindrical zone along an axis.
    Cylinder {
        base_center: Vec3,
        axis: Vec3,
        radius: f32,
        height: f32,
    },
    /// Infinite half-space defined by a plane.
    HalfSpace {
        point_on_plane: Vec3,
        normal: Vec3,
    },
    /// Capsule-shaped zone (cylinder with hemispherical caps).
    Capsule {
        start: Vec3,
        end: Vec3,
        radius: f32,
    },
    /// Global zone that covers the entire world (used for default gravity).
    Global,
}

impl GravityZoneShape {
    /// Test if a point is inside this shape.
    pub fn contains(&self, point: Vec3) -> bool {
        match self {
            GravityZoneShape::Sphere { center, radius } => {
                (point - *center).length_squared() <= radius * radius
            }
            GravityZoneShape::Box { min, max } => {
                point.x >= min.x && point.x <= max.x
                    && point.y >= min.y && point.y <= max.y
                    && point.z >= min.z && point.z <= max.z
            }
            GravityZoneShape::OrientedBox { center, half_extents, rotation } => {
                let inv_rot = rotation.inverse();
                let local = inv_rot * (point - *center);
                local.x.abs() <= half_extents.x
                    && local.y.abs() <= half_extents.y
                    && local.z.abs() <= half_extents.z
            }
            GravityZoneShape::Cylinder { base_center, axis, radius, height } => {
                let axis_norm = axis.normalize_or_zero();
                let to_point = point - *base_center;
                let along = to_point.dot(axis_norm);
                if along < 0.0 || along > *height {
                    return false;
                }
                let radial = to_point - axis_norm * along;
                radial.length_squared() <= radius * radius
            }
            GravityZoneShape::HalfSpace { point_on_plane, normal } => {
                let norm = normal.normalize_or_zero();
                (point - *point_on_plane).dot(norm) <= 0.0
            }
            GravityZoneShape::Capsule { start, end, radius } => {
                let seg = *end - *start;
                let seg_len_sq = seg.length_squared();
                let t = if seg_len_sq < EPSILON {
                    0.0
                } else {
                    ((point - *start).dot(seg) / seg_len_sq).clamp(0.0, 1.0)
                };
                let closest = *start + seg * t;
                (point - closest).length_squared() <= radius * radius
            }
            GravityZoneShape::Global => true,
        }
    }

    /// Compute the signed distance from a point to the shape boundary.
    /// Negative means inside, positive means outside.
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        match self {
            GravityZoneShape::Sphere { center, radius } => {
                (point - *center).length() - radius
            }
            GravityZoneShape::Box { min, max } => {
                let center = (*min + *max) * 0.5;
                let half = (*max - *min) * 0.5;
                let local = point - center;
                let d = Vec3::new(
                    local.x.abs() - half.x,
                    local.y.abs() - half.y,
                    local.z.abs() - half.z,
                );
                let outside = Vec3::new(d.x.max(0.0), d.y.max(0.0), d.z.max(0.0)).length();
                let inside = d.x.max(d.y).max(d.z).min(0.0);
                outside + inside
            }
            GravityZoneShape::OrientedBox { center, half_extents, rotation } => {
                let inv_rot = rotation.inverse();
                let local = inv_rot * (point - *center);
                let d = Vec3::new(
                    local.x.abs() - half_extents.x,
                    local.y.abs() - half_extents.y,
                    local.z.abs() - half_extents.z,
                );
                let outside = Vec3::new(d.x.max(0.0), d.y.max(0.0), d.z.max(0.0)).length();
                let inside = d.x.max(d.y).max(d.z).min(0.0);
                outside + inside
            }
            GravityZoneShape::Cylinder { base_center, axis, radius, height } => {
                let axis_norm = axis.normalize_or_zero();
                let to_point = point - *base_center;
                let along = to_point.dot(axis_norm);
                let radial_vec = to_point - axis_norm * along;
                let radial_dist = radial_vec.length();
                let axial_dist = if along < 0.0 {
                    -along
                } else if along > *height {
                    along - *height
                } else {
                    0.0_f32.min(along).min(*height - along)
                };
                let r_dist = radial_dist - *radius;
                if along >= 0.0 && along <= *height {
                    if radial_dist <= *radius {
                        // Inside: negative of minimum distance to boundary
                        let to_cap = (along).min(*height - along);
                        let to_wall = *radius - radial_dist;
                        -(to_cap.min(to_wall))
                    } else {
                        r_dist
                    }
                } else {
                    (axial_dist * axial_dist + r_dist.max(0.0).powi(2)).sqrt()
                }
            }
            GravityZoneShape::HalfSpace { point_on_plane, normal } => {
                let norm = normal.normalize_or_zero();
                (point - *point_on_plane).dot(norm)
            }
            GravityZoneShape::Capsule { start, end, radius } => {
                let seg = *end - *start;
                let seg_len_sq = seg.length_squared();
                let t = if seg_len_sq < EPSILON {
                    0.0
                } else {
                    ((point - *start).dot(seg) / seg_len_sq).clamp(0.0, 1.0)
                };
                let closest = *start + seg * t;
                (point - closest).length() - radius
            }
            GravityZoneShape::Global => f32::NEG_INFINITY,
        }
    }

    /// Get the approximate center of the shape.
    pub fn center(&self) -> Vec3 {
        match self {
            GravityZoneShape::Sphere { center, .. } => *center,
            GravityZoneShape::Box { min, max } => (*min + *max) * 0.5,
            GravityZoneShape::OrientedBox { center, .. } => *center,
            GravityZoneShape::Cylinder { base_center, axis, height, .. } => {
                *base_center + axis.normalize_or_zero() * (*height * 0.5)
            }
            GravityZoneShape::HalfSpace { point_on_plane, .. } => *point_on_plane,
            GravityZoneShape::Capsule { start, end, .. } => (*start + *end) * 0.5,
            GravityZoneShape::Global => Vec3::ZERO,
        }
    }
}

// ---------------------------------------------------------------------------
// GravityMode
// ---------------------------------------------------------------------------

/// How gravity is calculated within a zone.
#[derive(Debug, Clone)]
pub enum GravityMode {
    /// Constant directional gravity (e.g. standard Earth-like).
    Directional {
        direction: Vec3,
        magnitude: f32,
    },
    /// Radial gravity toward a center point (spherical planet).
    Radial {
        center: Vec3,
        magnitude: f32,
    },
    /// Radial gravity away from a center point (anti-gravity repulsor).
    RadialRepulsor {
        center: Vec3,
        magnitude: f32,
    },
    /// Zero gravity zone.
    ZeroG,
    /// Anti-gravity (reverse of the global default).
    AntiGravity {
        magnitude: f32,
    },
    /// Custom gravity specified by a function index (for game code to evaluate).
    Custom {
        function_id: u32,
        parameter: f32,
    },
    /// Gravity that varies with distance from a center (inverse square law).
    InverseSquare {
        center: Vec3,
        mass: f32,
    },
    /// Gravity that follows a spline or path (for roller-coaster like sections).
    AlongAxis {
        axis: Vec3,
        magnitude: f32,
    },
    /// Gravity that rotates around an axis (centrifuge stations).
    Centrifugal {
        axis_point: Vec3,
        axis_dir: Vec3,
        angular_speed: f32,
    },
}

impl GravityMode {
    /// Calculate the gravity vector at a given world position.
    pub fn gravity_at(&self, position: Vec3) -> Vec3 {
        match self {
            GravityMode::Directional { direction, magnitude } => {
                direction.normalize_or_zero() * *magnitude
            }
            GravityMode::Radial { center, magnitude } => {
                let diff = *center - position;
                let len = diff.length();
                if len < EPSILON {
                    Vec3::ZERO
                } else {
                    (diff / len) * *magnitude
                }
            }
            GravityMode::RadialRepulsor { center, magnitude } => {
                let diff = position - *center;
                let len = diff.length();
                if len < EPSILON {
                    Vec3::ZERO
                } else {
                    (diff / len) * *magnitude
                }
            }
            GravityMode::ZeroG => Vec3::ZERO,
            GravityMode::AntiGravity { magnitude } => {
                Vec3::new(0.0, 1.0, 0.0) * *magnitude
            }
            GravityMode::Custom { .. } => {
                // Game code should handle this externally
                Vec3::ZERO
            }
            GravityMode::InverseSquare { center, mass } => {
                let diff = *center - position;
                let dist_sq = diff.length_squared().max(MIN_WELL_DISTANCE * MIN_WELL_DISTANCE);
                let len = dist_sq.sqrt();
                let force_mag = GRAVITATIONAL_CONSTANT * *mass / dist_sq;
                (diff / len) * force_mag
            }
            GravityMode::AlongAxis { axis, magnitude } => {
                axis.normalize_or_zero() * *magnitude
            }
            GravityMode::Centrifugal { axis_point, axis_dir, angular_speed } => {
                let axis_norm = axis_dir.normalize_or_zero();
                let to_pos = position - *axis_point;
                let along_axis = to_pos.dot(axis_norm);
                let radial = to_pos - axis_norm * along_axis;
                let radial_dist = radial.length();
                if radial_dist < EPSILON {
                    Vec3::ZERO
                } else {
                    // Centrifugal: push outward proportional to omega^2 * r
                    let centrifugal = (radial / radial_dist) * angular_speed * angular_speed * radial_dist;
                    // Also add "downward" (toward axis) component for artificial gravity
                    let inward = -(radial / radial_dist) * DEFAULT_GRAVITY;
                    inward + centrifugal * 0.0 // Pure centrifugal
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GravityBlendMode
// ---------------------------------------------------------------------------

/// How overlapping gravity zones blend with each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GravityBlendMode {
    /// Higher priority zone completely overrides lower priority.
    Override,
    /// Zones blend based on their weights (smooth transition).
    Blend,
    /// Zones are added together.
    Additive,
    /// Use the zone closest to the query point.
    Nearest,
}

// ---------------------------------------------------------------------------
// GravityTransition
// ---------------------------------------------------------------------------

/// Configuration for smooth transitions at gravity zone boundaries.
#[derive(Debug, Clone)]
pub struct GravityTransition {
    /// Width of the transition region in meters.
    pub width: f32,
    /// Easing function for the transition.
    pub easing: TransitionEasing,
    /// Whether to interpolate direction as well as magnitude.
    pub interpolate_direction: bool,
}

impl Default for GravityTransition {
    fn default() -> Self {
        Self {
            width: DEFAULT_TRANSITION_WIDTH,
            easing: TransitionEasing::SmoothStep,
            interpolate_direction: true,
        }
    }
}

/// Easing function for gravity transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionEasing {
    /// Linear interpolation.
    Linear,
    /// Smooth step (Hermite interpolation).
    SmoothStep,
    /// Smoother step (Ken Perlin's improved version).
    SmootherStep,
    /// Ease in (quadratic).
    EaseIn,
    /// Ease out (quadratic).
    EaseOut,
    /// Ease in-out (quadratic).
    EaseInOut,
}

impl TransitionEasing {
    /// Evaluate the easing function for t in [0, 1].
    pub fn evaluate(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            TransitionEasing::Linear => t,
            TransitionEasing::SmoothStep => t * t * (3.0 - 2.0 * t),
            TransitionEasing::SmootherStep => t * t * t * (t * (t * 6.0 - 15.0) + 10.0),
            TransitionEasing::EaseIn => t * t,
            TransitionEasing::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
            TransitionEasing::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GravityZone
// ---------------------------------------------------------------------------

/// A region of space with custom gravity behavior.
#[derive(Debug, Clone)]
pub struct GravityZone {
    /// Unique identifier for this zone.
    pub id: GravityZoneId,
    /// Human-readable name for debugging.
    pub name: String,
    /// The spatial shape of this zone.
    pub shape: GravityZoneShape,
    /// How gravity behaves inside this zone.
    pub mode: GravityMode,
    /// Priority for resolving overlaps (higher = takes precedence).
    pub priority: i32,
    /// How this zone blends with overlapping zones.
    pub blend_mode: GravityBlendMode,
    /// Blend weight (0..1) for weighted blending.
    pub blend_weight: f32,
    /// Transition configuration for the zone boundary.
    pub transition: GravityTransition,
    /// Whether this zone is currently active.
    pub active: bool,
    /// Optional tag for game code queries.
    pub tag: Option<String>,
    /// Fade-in duration in seconds when activating.
    pub fade_in_duration: f32,
    /// Fade-out duration in seconds when deactivating.
    pub fade_out_duration: f32,
    /// Current fade progress (0 = fully out, 1 = fully in).
    fade_progress: f32,
    /// Whether this zone is currently fading in or out.
    fade_direction: FadeDirection,
}

/// Direction of zone fade animation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FadeDirection {
    None,
    FadingIn,
    FadingOut,
}

impl GravityZone {
    /// Create a new gravity zone with default settings.
    pub fn new(id: GravityZoneId, name: impl Into<String>, shape: GravityZoneShape, mode: GravityMode) -> Self {
        Self {
            id,
            name: name.into(),
            shape,
            mode,
            priority: 0,
            blend_mode: GravityBlendMode::Override,
            blend_weight: 1.0,
            transition: GravityTransition::default(),
            active: true,
            tag: None,
            fade_in_duration: 0.0,
            fade_out_duration: 0.0,
            fade_progress: 1.0,
            fade_direction: FadeDirection::None,
        }
    }

    /// Builder: set priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Builder: set blend mode.
    pub fn with_blend_mode(mut self, mode: GravityBlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Builder: set blend weight.
    pub fn with_blend_weight(mut self, weight: f32) -> Self {
        self.blend_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Builder: set transition.
    pub fn with_transition(mut self, transition: GravityTransition) -> Self {
        self.transition = transition;
        self
    }

    /// Builder: set tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    /// Builder: set fade durations.
    pub fn with_fade(mut self, fade_in: f32, fade_out: f32) -> Self {
        self.fade_in_duration = fade_in.max(0.0);
        self.fade_out_duration = fade_out.max(0.0);
        self
    }

    /// Activate this zone (optionally with fade).
    pub fn activate(&mut self) {
        if self.active {
            return;
        }
        self.active = true;
        if self.fade_in_duration > EPSILON {
            self.fade_direction = FadeDirection::FadingIn;
        } else {
            self.fade_progress = 1.0;
            self.fade_direction = FadeDirection::None;
        }
    }

    /// Deactivate this zone (optionally with fade).
    pub fn deactivate(&mut self) {
        if !self.active && self.fade_progress <= EPSILON {
            return;
        }
        if self.fade_out_duration > EPSILON {
            self.fade_direction = FadeDirection::FadingOut;
        } else {
            self.active = false;
            self.fade_progress = 0.0;
            self.fade_direction = FadeDirection::None;
        }
    }

    /// Update fade progress.
    pub fn update_fade(&mut self, dt: f32) {
        match self.fade_direction {
            FadeDirection::None => {}
            FadeDirection::FadingIn => {
                if self.fade_in_duration > EPSILON {
                    self.fade_progress += dt / self.fade_in_duration;
                }
                if self.fade_progress >= 1.0 {
                    self.fade_progress = 1.0;
                    self.fade_direction = FadeDirection::None;
                }
            }
            FadeDirection::FadingOut => {
                if self.fade_out_duration > EPSILON {
                    self.fade_progress -= dt / self.fade_out_duration;
                }
                if self.fade_progress <= 0.0 {
                    self.fade_progress = 0.0;
                    self.active = false;
                    self.fade_direction = FadeDirection::None;
                }
            }
        }
    }

    /// Get the effective weight of this zone at a given position.
    /// Considers boundary transition, blend weight, and fade progress.
    pub fn effective_weight(&self, position: Vec3) -> f32 {
        if !self.active && self.fade_progress <= EPSILON {
            return 0.0;
        }

        let sd = self.shape.signed_distance(position);
        let boundary_weight = if sd <= -self.transition.width {
            // Fully inside
            1.0
        } else if sd >= 0.0 {
            // Outside
            0.0
        } else {
            // In transition region
            let t = (-sd) / self.transition.width;
            self.transition.easing.evaluate(t)
        };

        boundary_weight * self.blend_weight * self.fade_progress
    }

    /// Get the gravity vector at a position (unweighted).
    pub fn gravity_at(&self, position: Vec3) -> Vec3 {
        self.mode.gravity_at(position)
    }
}

// ---------------------------------------------------------------------------
// GravityWell
// ---------------------------------------------------------------------------

/// A point-source gravity attractor.
#[derive(Debug, Clone)]
pub struct GravityWell {
    /// Unique identifier.
    pub id: GravityWellId,
    /// Name for debugging.
    pub name: String,
    /// Position in world space.
    pub position: Vec3,
    /// Mass of the gravity well (affects pull strength).
    pub mass: f32,
    /// Maximum range of influence.
    pub max_range: f32,
    /// Falloff type for the gravity well.
    pub falloff: GravityFalloff,
    /// Whether this well is currently active.
    pub active: bool,
    /// Whether this is an attractor (true) or repulsor (false).
    pub attractor: bool,
    /// Maximum force cap to prevent extreme values.
    pub max_force: f32,
    /// Optional tag.
    pub tag: Option<String>,
    /// Time-varying pulsation settings.
    pub pulsation: Option<GravityPulsation>,
    /// Internal time accumulator for pulsation.
    pulsation_time: f32,
}

/// Falloff function for gravity wells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GravityFalloff {
    /// Inverse square law (physically accurate).
    InverseSquare,
    /// Linear falloff from max_force to zero over max_range.
    Linear,
    /// Constant force within range.
    Constant,
    /// Smooth falloff (cubic).
    Smooth,
    /// Exponential decay.
    Exponential,
}

/// Pulsation settings for time-varying gravity wells.
#[derive(Debug, Clone)]
pub struct GravityPulsation {
    /// Frequency of pulsation in Hz.
    pub frequency: f32,
    /// Amplitude of pulsation (0..1, multiplied against base force).
    pub amplitude: f32,
    /// Waveform type.
    pub waveform: PulsationWaveform,
    /// Phase offset in radians.
    pub phase: f32,
}

/// Waveform type for gravity pulsation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PulsationWaveform {
    /// Sine wave.
    Sine,
    /// Square wave.
    Square,
    /// Triangle wave.
    Triangle,
    /// Sawtooth wave.
    Sawtooth,
}

impl GravityWell {
    /// Create a new gravity well.
    pub fn new(id: GravityWellId, name: impl Into<String>, position: Vec3, mass: f32) -> Self {
        Self {
            id,
            name: name.into(),
            position,
            mass,
            max_range: 50.0,
            falloff: GravityFalloff::InverseSquare,
            active: true,
            attractor: true,
            max_force: 100.0,
            tag: None,
            pulsation: None,
            pulsation_time: 0.0,
        }
    }

    /// Builder: set max range.
    pub fn with_max_range(mut self, range: f32) -> Self {
        self.max_range = range.max(0.0);
        self
    }

    /// Builder: set falloff type.
    pub fn with_falloff(mut self, falloff: GravityFalloff) -> Self {
        self.falloff = falloff;
        self
    }

    /// Builder: set as repulsor.
    pub fn as_repulsor(mut self) -> Self {
        self.attractor = false;
        self
    }

    /// Builder: set max force cap.
    pub fn with_max_force(mut self, max_force: f32) -> Self {
        self.max_force = max_force.max(0.0);
        self
    }

    /// Builder: set pulsation.
    pub fn with_pulsation(mut self, pulsation: GravityPulsation) -> Self {
        self.pulsation = Some(pulsation);
        self
    }

    /// Update internal time for pulsation.
    pub fn update(&mut self, dt: f32) {
        self.pulsation_time += dt;
    }

    /// Get the pulsation multiplier at the current time.
    fn pulsation_multiplier(&self) -> f32 {
        match &self.pulsation {
            None => 1.0,
            Some(pulse) => {
                let t = self.pulsation_time * pulse.frequency * 2.0 * PI + pulse.phase;
                let wave = match pulse.waveform {
                    PulsationWaveform::Sine => t.sin(),
                    PulsationWaveform::Square => {
                        if t.sin() >= 0.0 { 1.0 } else { -1.0 }
                    }
                    PulsationWaveform::Triangle => {
                        let phase = (t / PI).rem_euclid(2.0);
                        if phase < 1.0 { 2.0 * phase - 1.0 } else { 3.0 - 2.0 * phase }
                    }
                    PulsationWaveform::Sawtooth => {
                        (t / PI).rem_euclid(2.0) - 1.0
                    }
                };
                1.0 + wave * pulse.amplitude
            }
        }
    }

    /// Calculate the gravity force vector at a position.
    pub fn force_at(&self, position: Vec3) -> Vec3 {
        if !self.active {
            return Vec3::ZERO;
        }

        let diff = self.position - position;
        let dist = diff.length();
        if dist < EPSILON {
            return Vec3::ZERO;
        }
        if dist > self.max_range {
            return Vec3::ZERO;
        }

        let direction = if self.attractor { diff / dist } else { -diff / dist };
        let normalized_dist = dist / self.max_range;

        let force_magnitude = match self.falloff {
            GravityFalloff::InverseSquare => {
                let safe_dist = dist.max(MIN_WELL_DISTANCE);
                GRAVITATIONAL_CONSTANT * self.mass / (safe_dist * safe_dist)
            }
            GravityFalloff::Linear => {
                self.mass * (1.0 - normalized_dist)
            }
            GravityFalloff::Constant => {
                self.mass
            }
            GravityFalloff::Smooth => {
                let t = 1.0 - normalized_dist;
                self.mass * t * t * (3.0 - 2.0 * t)
            }
            GravityFalloff::Exponential => {
                self.mass * (-3.0 * normalized_dist).exp()
            }
        };

        let clamped = force_magnitude.min(self.max_force);
        let pulsed = clamped * self.pulsation_multiplier();

        direction * pulsed
    }
}

// ---------------------------------------------------------------------------
// GravityFieldManager
// ---------------------------------------------------------------------------

/// Manages all gravity zones and wells, resolves gravity queries.
#[derive(Debug)]
pub struct GravityFieldManager {
    /// Registered gravity zones.
    zones: Vec<GravityZone>,
    /// Registered gravity wells.
    wells: Vec<GravityWell>,
    /// Default gravity when no zones apply.
    default_gravity: Vec3,
    /// Next zone ID.
    next_zone_id: u64,
    /// Next well ID.
    next_well_id: u64,
    /// Cache of last query results (entity_id -> gravity).
    gravity_cache: HashMap<u64, GravityCacheEntry>,
    /// Whether the cache is valid.
    cache_dirty: bool,
    /// Global gravity multiplier.
    pub global_multiplier: f32,
    /// Whether to include wells in gravity calculations.
    pub wells_enabled: bool,
    /// Debug: last query position.
    last_query_position: Vec3,
    /// Debug: last query result.
    last_query_result: Vec3,
    /// Debug: number of zones considered in last query.
    last_zones_considered: usize,
}

/// Cached gravity result for an entity.
#[derive(Debug, Clone)]
struct GravityCacheEntry {
    gravity: Vec3,
    position: Vec3,
    frame: u64,
}

/// Result of a gravity query with debug info.
#[derive(Debug, Clone)]
pub struct GravityQueryResult {
    /// The effective gravity vector.
    pub gravity: Vec3,
    /// The "down" direction (normalized gravity, or default if zero-g).
    pub down: Vec3,
    /// The magnitude of gravity.
    pub magnitude: f32,
    /// Number of zones that contributed.
    pub zone_count: usize,
    /// Number of wells that contributed.
    pub well_count: usize,
    /// IDs of zones that contributed.
    pub contributing_zones: Vec<GravityZoneId>,
    /// Whether the point is in zero or near-zero gravity.
    pub is_zero_g: bool,
}

impl GravityFieldManager {
    /// Create a new gravity field manager with default Earth-like gravity.
    pub fn new() -> Self {
        Self {
            zones: Vec::new(),
            wells: Vec::new(),
            default_gravity: DEFAULT_GRAVITY_DIR * DEFAULT_GRAVITY,
            next_zone_id: 1,
            next_well_id: 1,
            gravity_cache: HashMap::new(),
            cache_dirty: true,
            global_multiplier: 1.0,
            wells_enabled: true,
            last_query_position: Vec3::ZERO,
            last_query_result: Vec3::ZERO,
            last_zones_considered: 0,
        }
    }

    /// Create with custom default gravity.
    pub fn with_default_gravity(gravity: Vec3) -> Self {
        let mut mgr = Self::new();
        mgr.default_gravity = gravity;
        mgr
    }

    /// Set the default gravity vector.
    pub fn set_default_gravity(&mut self, gravity: Vec3) {
        self.default_gravity = gravity;
        self.cache_dirty = true;
    }

    /// Get the default gravity vector.
    pub fn default_gravity(&self) -> Vec3 {
        self.default_gravity
    }

    /// Add a gravity zone and return its ID.
    pub fn add_zone(&mut self, mut zone: GravityZone) -> GravityZoneId {
        let id = GravityZoneId::new(self.next_zone_id);
        self.next_zone_id += 1;
        zone.id = id;
        self.zones.push(zone);
        self.cache_dirty = true;
        id
    }

    /// Remove a gravity zone by ID.
    pub fn remove_zone(&mut self, id: GravityZoneId) -> bool {
        let before = self.zones.len();
        self.zones.retain(|z| z.id != id);
        let removed = self.zones.len() < before;
        if removed {
            self.cache_dirty = true;
        }
        removed
    }

    /// Get a mutable reference to a gravity zone by ID.
    pub fn zone_mut(&mut self, id: GravityZoneId) -> Option<&mut GravityZone> {
        self.zones.iter_mut().find(|z| z.id == id)
    }

    /// Get a reference to a gravity zone by ID.
    pub fn zone(&self, id: GravityZoneId) -> Option<&GravityZone> {
        self.zones.iter().find(|z| z.id == id)
    }

    /// Get all zones.
    pub fn zones(&self) -> &[GravityZone] {
        &self.zones
    }

    /// Add a gravity well and return its ID.
    pub fn add_well(&mut self, mut well: GravityWell) -> GravityWellId {
        let id = GravityWellId::new(self.next_well_id);
        self.next_well_id += 1;
        well.id = id;
        self.wells.push(well);
        self.cache_dirty = true;
        id
    }

    /// Remove a gravity well by ID.
    pub fn remove_well(&mut self, id: GravityWellId) -> bool {
        let before = self.wells.len();
        self.wells.retain(|w| w.id != id);
        let removed = self.wells.len() < before;
        if removed {
            self.cache_dirty = true;
        }
        removed
    }

    /// Get a mutable reference to a gravity well.
    pub fn well_mut(&mut self, id: GravityWellId) -> Option<&mut GravityWell> {
        self.wells.iter_mut().find(|w| w.id == id)
    }

    /// Get all wells.
    pub fn wells(&self) -> &[GravityWell] {
        &self.wells
    }

    /// Find zones by tag.
    pub fn zones_with_tag(&self, tag: &str) -> Vec<GravityZoneId> {
        self.zones.iter()
            .filter(|z| z.tag.as_deref() == Some(tag))
            .map(|z| z.id)
            .collect()
    }

    /// Update all zones and wells (fade animations, pulsation).
    pub fn update(&mut self, dt: f32) {
        for zone in &mut self.zones {
            zone.update_fade(dt);
        }
        for well in &mut self.wells {
            well.update(dt);
        }
        self.cache_dirty = true;
    }

    /// Query gravity at a position (simple version, returns just the vector).
    pub fn gravity_at(&self, position: Vec3) -> Vec3 {
        let result = self.query_gravity(position);
        result.gravity
    }

    /// Query gravity with full debug information.
    pub fn query_gravity(&self, position: Vec3) -> GravityQueryResult {
        let mut total_gravity = Vec3::ZERO;
        let mut total_weight = 0.0;
        let mut contributing_zones = Vec::new();
        let mut zone_count = 0;
        let mut well_count = 0;

        // Collect active zones sorted by priority (highest first)
        let mut active_zones: Vec<(usize, f32, i32)> = Vec::new();
        for (i, zone) in self.zones.iter().enumerate() {
            let w = zone.effective_weight(position);
            if w > MIN_BLEND_WEIGHT {
                active_zones.push((i, w, zone.priority));
            }
        }
        active_zones.sort_by(|a, b| b.2.cmp(&a.2));

        // Resolve zone contributions
        if active_zones.is_empty() {
            // No zones apply: use default gravity
            total_gravity = self.default_gravity;
        } else {
            // Process zones by blend mode
            let highest_priority = active_zones[0].2;

            for &(idx, weight, priority) in &active_zones {
                let zone = &self.zones[idx];

                match zone.blend_mode {
                    GravityBlendMode::Override => {
                        if priority == highest_priority {
                            total_gravity = zone.gravity_at(position) * weight;
                            total_weight = weight;
                            contributing_zones.push(zone.id);
                            zone_count += 1;
                            break;
                        }
                    }
                    GravityBlendMode::Blend => {
                        total_gravity += zone.gravity_at(position) * weight;
                        total_weight += weight;
                        contributing_zones.push(zone.id);
                        zone_count += 1;
                    }
                    GravityBlendMode::Additive => {
                        total_gravity += zone.gravity_at(position) * weight;
                        contributing_zones.push(zone.id);
                        zone_count += 1;
                    }
                    GravityBlendMode::Nearest => {
                        let dist = zone.shape.signed_distance(position);
                        // The more negative (deeper inside), the closer
                        let closeness = (-dist).max(0.0);
                        total_gravity += zone.gravity_at(position) * closeness;
                        total_weight += closeness;
                        contributing_zones.push(zone.id);
                        zone_count += 1;
                    }
                }
            }

            // Normalize blended gravity
            if total_weight > EPSILON && active_zones.iter().any(|&(i, _, _)| {
                matches!(self.zones[i].blend_mode, GravityBlendMode::Blend | GravityBlendMode::Nearest)
            }) {
                total_gravity /= total_weight;
            }

            // If total weight is less than 1, blend with default gravity
            if total_weight < 1.0 - EPSILON {
                let default_contribution = self.default_gravity * (1.0 - total_weight);
                total_gravity += default_contribution;
            }
        }

        // Add gravity well contributions
        if self.wells_enabled {
            for well in &self.wells {
                let force = well.force_at(position);
                if force.length_squared() > EPSILON {
                    total_gravity += force;
                    well_count += 1;
                }
            }
        }

        // Apply global multiplier
        total_gravity *= self.global_multiplier;

        let magnitude = total_gravity.length();
        let down = if magnitude > EPSILON {
            total_gravity / magnitude
        } else {
            DEFAULT_GRAVITY_DIR
        };

        GravityQueryResult {
            gravity: total_gravity,
            down,
            magnitude,
            zone_count,
            well_count,
            contributing_zones,
            is_zero_g: magnitude < 0.1,
        }
    }

    /// Get the "up" direction at a position (opposite of gravity).
    pub fn up_direction(&self, position: Vec3) -> Vec3 {
        let result = self.query_gravity(position);
        -result.down
    }

    /// Get the "down" direction at a position.
    pub fn down_direction(&self, position: Vec3) -> Vec3 {
        self.query_gravity(position).down
    }

    /// Check if a position is in zero-gravity.
    pub fn is_zero_g(&self, position: Vec3) -> bool {
        self.query_gravity(position).is_zero_g
    }

    /// Get the total number of active zones.
    pub fn active_zone_count(&self) -> usize {
        self.zones.iter().filter(|z| z.active).count()
    }

    /// Get the total number of active wells.
    pub fn active_well_count(&self) -> usize {
        self.wells.iter().filter(|w| w.active).count()
    }

    /// Clear all zones and wells.
    pub fn clear(&mut self) {
        self.zones.clear();
        self.wells.clear();
        self.gravity_cache.clear();
        self.cache_dirty = true;
    }

    /// Debug statistics.
    pub fn stats(&self) -> GravityFieldStats {
        GravityFieldStats {
            total_zones: self.zones.len(),
            active_zones: self.active_zone_count(),
            total_wells: self.wells.len(),
            active_wells: self.active_well_count(),
            cache_entries: self.gravity_cache.len(),
        }
    }
}

/// Statistics for the gravity field system.
#[derive(Debug, Clone)]
pub struct GravityFieldStats {
    /// Total number of zones.
    pub total_zones: usize,
    /// Number of active zones.
    pub active_zones: usize,
    /// Total number of wells.
    pub total_wells: usize,
    /// Number of active wells.
    pub active_wells: usize,
    /// Number of cache entries.
    pub cache_entries: usize,
}

// ---------------------------------------------------------------------------
// GravityFieldComponent (ECS)
// ---------------------------------------------------------------------------

/// ECS component for entities that are affected by gravity fields.
#[derive(Debug, Clone)]
pub struct GravityFieldComponent {
    /// Whether this entity is affected by gravity zones.
    pub affected_by_zones: bool,
    /// Whether this entity is affected by gravity wells.
    pub affected_by_wells: bool,
    /// Custom gravity multiplier for this entity.
    pub gravity_multiplier: f32,
    /// Cached effective gravity vector.
    pub effective_gravity: Vec3,
    /// Cached "up" direction.
    pub effective_up: Vec3,
    /// Whether the entity is in zero-g.
    pub in_zero_g: bool,
    /// Override gravity (ignores field system entirely if set).
    pub override_gravity: Option<Vec3>,
    /// IDs of zones the entity is currently inside.
    pub current_zones: Vec<GravityZoneId>,
}

impl Default for GravityFieldComponent {
    fn default() -> Self {
        Self {
            affected_by_zones: true,
            affected_by_wells: true,
            gravity_multiplier: 1.0,
            effective_gravity: DEFAULT_GRAVITY_DIR * DEFAULT_GRAVITY,
            effective_up: Vec3::new(0.0, 1.0, 0.0),
            in_zero_g: false,
            override_gravity: None,
            current_zones: Vec::new(),
        }
    }
}

impl GravityFieldComponent {
    /// Create a new gravity field component.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a gravity override (bypass field system).
    pub fn set_override(&mut self, gravity: Vec3) {
        self.override_gravity = Some(gravity);
    }

    /// Clear gravity override.
    pub fn clear_override(&mut self) {
        self.override_gravity = None;
    }
}

// ---------------------------------------------------------------------------
// GravitySourceComponent (ECS)
// ---------------------------------------------------------------------------

/// ECS component for entities that generate gravity (e.g., planets).
#[derive(Debug, Clone)]
pub struct GravitySourceComponent {
    /// Zone ID in the gravity field manager (if zone-based).
    pub zone_id: Option<GravityZoneId>,
    /// Well ID in the gravity field manager (if well-based).
    pub well_id: Option<GravityWellId>,
    /// Whether this source auto-updates its position from the entity transform.
    pub track_transform: bool,
}

impl GravitySourceComponent {
    /// Create a new gravity source component.
    pub fn new() -> Self {
        Self {
            zone_id: None,
            well_id: None,
            track_transform: true,
        }
    }

    /// Create for a zone source.
    pub fn from_zone(zone_id: GravityZoneId) -> Self {
        Self {
            zone_id: Some(zone_id),
            well_id: None,
            track_transform: true,
        }
    }

    /// Create for a well source.
    pub fn from_well(well_id: GravityWellId) -> Self {
        Self {
            zone_id: None,
            well_id: Some(well_id),
            track_transform: true,
        }
    }
}

// ---------------------------------------------------------------------------
// GravityFieldSystem (ECS system)
// ---------------------------------------------------------------------------

/// ECS system that updates gravity field components each frame.
pub struct GravityFieldSystem {
    /// The gravity field manager.
    pub manager: GravityFieldManager,
    /// Performance: max entities to update per frame (0 = unlimited).
    pub max_updates_per_frame: usize,
    /// Internal frame counter.
    frame_counter: u64,
}

impl GravityFieldSystem {
    /// Create a new gravity field system.
    pub fn new() -> Self {
        Self {
            manager: GravityFieldManager::new(),
            max_updates_per_frame: 0,
            frame_counter: 0,
        }
    }

    /// Create with a custom gravity field manager.
    pub fn with_manager(manager: GravityFieldManager) -> Self {
        Self {
            manager,
            max_updates_per_frame: 0,
            frame_counter: 0,
        }
    }

    /// Update the system (call once per frame).
    pub fn update(&mut self, dt: f32) {
        self.manager.update(dt);
        self.frame_counter += 1;
    }

    /// Update a single entity's gravity field component.
    pub fn update_entity(&self, position: Vec3, component: &mut GravityFieldComponent) {
        if let Some(override_g) = component.override_gravity {
            component.effective_gravity = override_g;
            let mag = override_g.length();
            component.effective_up = if mag > EPSILON {
                -override_g / mag
            } else {
                Vec3::new(0.0, 1.0, 0.0)
            };
            component.in_zero_g = mag < 0.1;
            return;
        }

        let result = self.manager.query_gravity(position);
        component.effective_gravity = result.gravity * component.gravity_multiplier;
        component.effective_up = -result.down;
        component.in_zero_g = result.is_zero_g;
        component.current_zones = result.contributing_zones;
    }

    /// Query gravity at a position.
    pub fn gravity_at(&self, position: Vec3) -> Vec3 {
        self.manager.gravity_at(position)
    }

    /// Get the manager reference.
    pub fn manager(&self) -> &GravityFieldManager {
        &self.manager
    }

    /// Get the manager mutably.
    pub fn manager_mut(&mut self) -> &mut GravityFieldManager {
        &mut self.manager
    }
}

// ---------------------------------------------------------------------------
// Preset helpers
// ---------------------------------------------------------------------------

/// Create a spherical planet gravity zone.
pub fn spherical_planet(
    name: impl Into<String>,
    center: Vec3,
    radius: f32,
    surface_gravity: f32,
) -> GravityZone {
    GravityZone::new(
        GravityZoneId::new(0),
        name,
        GravityZoneShape::Sphere { center, radius },
        GravityMode::Radial { center, magnitude: surface_gravity },
    )
    .with_priority(10)
    .with_transition(GravityTransition {
        width: radius * 0.2,
        easing: TransitionEasing::SmoothStep,
        interpolate_direction: true,
    })
}

/// Create a zero-gravity box zone (e.g., space station interior).
pub fn zero_g_box(
    name: impl Into<String>,
    min: Vec3,
    max: Vec3,
) -> GravityZone {
    GravityZone::new(
        GravityZoneId::new(0),
        name,
        GravityZoneShape::Box { min, max },
        GravityMode::ZeroG,
    )
    .with_priority(20)
}

/// Create a directional override zone (e.g., gravity room puzzle).
pub fn directional_override(
    name: impl Into<String>,
    shape: GravityZoneShape,
    direction: Vec3,
    magnitude: f32,
) -> GravityZone {
    GravityZone::new(
        GravityZoneId::new(0),
        name,
        shape,
        GravityMode::Directional { direction, magnitude },
    )
    .with_priority(15)
}

/// Create an anti-gravity volume.
pub fn anti_gravity_volume(
    name: impl Into<String>,
    shape: GravityZoneShape,
    strength: f32,
) -> GravityZone {
    GravityZone::new(
        GravityZoneId::new(0),
        name,
        shape,
        GravityMode::AntiGravity { magnitude: strength },
    )
    .with_priority(25)
}

/// Create a gravity well with inverse-square falloff.
pub fn gravity_well(
    name: impl Into<String>,
    position: Vec3,
    mass: f32,
    range: f32,
) -> GravityWell {
    GravityWell::new(
        GravityWellId::new(0),
        name,
        position,
        mass,
    )
    .with_max_range(range)
    .with_falloff(GravityFalloff::InverseSquare)
}

/// Create a pulsating gravity well.
pub fn pulsating_well(
    name: impl Into<String>,
    position: Vec3,
    mass: f32,
    range: f32,
    frequency: f32,
    amplitude: f32,
) -> GravityWell {
    GravityWell::new(
        GravityWellId::new(0),
        name,
        position,
        mass,
    )
    .with_max_range(range)
    .with_falloff(GravityFalloff::Smooth)
    .with_pulsation(GravityPulsation {
        frequency,
        amplitude,
        waveform: PulsationWaveform::Sine,
        phase: 0.0,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_gravity() {
        let mgr = GravityFieldManager::new();
        let g = mgr.gravity_at(Vec3::ZERO);
        assert!((g.y - (-DEFAULT_GRAVITY)).abs() < EPSILON);
        assert!(g.x.abs() < EPSILON);
        assert!(g.z.abs() < EPSILON);
    }

    #[test]
    fn test_spherical_zone_containment() {
        let shape = GravityZoneShape::Sphere {
            center: Vec3::ZERO,
            radius: 10.0,
        };
        assert!(shape.contains(Vec3::new(5.0, 0.0, 0.0)));
        assert!(shape.contains(Vec3::ZERO));
        assert!(!shape.contains(Vec3::new(11.0, 0.0, 0.0)));
    }

    #[test]
    fn test_box_zone_containment() {
        let shape = GravityZoneShape::Box {
            min: Vec3::new(-5.0, -5.0, -5.0),
            max: Vec3::new(5.0, 5.0, 5.0),
        };
        assert!(shape.contains(Vec3::ZERO));
        assert!(shape.contains(Vec3::new(4.9, 4.9, 4.9)));
        assert!(!shape.contains(Vec3::new(6.0, 0.0, 0.0)));
    }

    #[test]
    fn test_radial_gravity() {
        let mode = GravityMode::Radial {
            center: Vec3::ZERO,
            magnitude: 10.0,
        };
        let g = mode.gravity_at(Vec3::new(5.0, 0.0, 0.0));
        assert!(g.x < 0.0); // Should point toward center
        assert!((g.length() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_gravity_well_force() {
        let well = GravityWell::new(
            GravityWellId::new(1),
            "test_well",
            Vec3::ZERO,
            100.0,
        ).with_max_range(50.0).with_falloff(GravityFalloff::Linear);

        let force = well.force_at(Vec3::new(25.0, 0.0, 0.0));
        assert!(force.x < 0.0); // Should pull toward well
        assert!(force.y.abs() < EPSILON);
    }

    #[test]
    fn test_zone_priority_override() {
        let mut mgr = GravityFieldManager::new();
        let zone1 = GravityZone::new(
            GravityZoneId::new(0),
            "low_priority",
            GravityZoneShape::Global,
            GravityMode::Directional {
                direction: Vec3::new(0.0, -1.0, 0.0),
                magnitude: 5.0,
            },
        ).with_priority(0);

        let zone2 = GravityZone::new(
            GravityZoneId::new(0),
            "high_priority",
            GravityZoneShape::Global,
            GravityMode::Directional {
                direction: Vec3::new(0.0, 1.0, 0.0),
                magnitude: 20.0,
            },
        ).with_priority(10);

        mgr.add_zone(zone1);
        mgr.add_zone(zone2);

        let g = mgr.gravity_at(Vec3::ZERO);
        // High priority zone should override
        assert!(g.y > 0.0);
    }

    #[test]
    fn test_zero_g_zone() {
        let mut mgr = GravityFieldManager::new();
        let zone = zero_g_box(
            "zero_g_room",
            Vec3::new(-10.0, -10.0, -10.0),
            Vec3::new(10.0, 10.0, 10.0),
        );
        mgr.add_zone(zone);

        let result = mgr.query_gravity(Vec3::ZERO);
        assert!(result.is_zero_g);
    }

    #[test]
    fn test_transition_easing() {
        let linear = TransitionEasing::Linear;
        assert!((linear.evaluate(0.0)).abs() < EPSILON);
        assert!((linear.evaluate(0.5) - 0.5).abs() < EPSILON);
        assert!((linear.evaluate(1.0) - 1.0).abs() < EPSILON);

        let smooth = TransitionEasing::SmoothStep;
        assert!((smooth.evaluate(0.0)).abs() < EPSILON);
        assert!((smooth.evaluate(0.5) - 0.5).abs() < EPSILON);
        assert!((smooth.evaluate(1.0) - 1.0).abs() < EPSILON);
    }
}
