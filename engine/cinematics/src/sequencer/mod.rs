//! # Sequencer Engine
//!
//! The sequencer provides typed keyframe tracks for animating arbitrary
//! properties over time. A [`Sequence`] contains multiple tracks, each
//! targeting a specific entity and property. The [`SequencePlayer`] handles
//! playback state (play, pause, seek) and evaluates all tracks every frame.
//!
//! ## Track types
//!
//! | Track               | Animates                                     |
//! |---------------------|----------------------------------------------|
//! | [`TransformTrack`]  | Position, rotation, and scale                |
//! | [`FloatTrack`]      | Any single `f32` parameter                   |
//! | [`ColorTrack`]      | RGBA colour (with colour-space interpolation) |
//! | [`BoolTrack`]       | On/off toggle events                         |
//! | [`EventTrack`]      | Named event firing at specific times         |
//! | [`CameraTrack`]     | FOV, near/far plane, focus distance          |
//! | [`AudioTrack`]      | Audio playback triggers                      |
//! | [`SubSequenceTrack`]| Embedded child sequences                     |
//!
//! ## Interpolation
//!
//! Each keyframe carries an [`Interpolation`] mode that controls how values
//! transition *from* that keyframe to the next:
//!
//! - **Step** -- hold current value until the next keyframe.
//! - **Linear** -- straight-line interpolation.
//! - **Cubic** -- cubic Bezier with explicit in/out tangent handles.

use std::collections::HashMap;
use std::fmt;

use genovo_ecs::Entity;
use glam::{Quat, Vec3, Vec4};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Returns a placeholder entity for serde deserialization default.
fn default_entity() -> Entity {
    Entity::PLACEHOLDER
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the sequencer subsystem.
#[derive(Debug, thiserror::Error)]
pub enum SequencerError {
    #[error("Track {0} not found in sequence")]
    TrackNotFound(Uuid),
    #[error("Keyframe index {0} out of range for track with {1} keyframes")]
    KeyframeOutOfRange(usize, usize),
    #[error("Sequence duration must be positive, got {0}")]
    InvalidDuration(f32),
    #[error("Cannot bind track to entity: {0}")]
    BindingError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

/// Interpolation mode between two keyframes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Interpolation {
    /// Hold the value of the current keyframe until the next.
    Step,
    /// Linearly interpolate between keyframes.
    Linear,
    /// Cubic Bezier interpolation using tangent handles.
    Cubic,
}

impl Default for Interpolation {
    fn default() -> Self {
        Self::Linear
    }
}

// ---------------------------------------------------------------------------
// Keyframe
// ---------------------------------------------------------------------------

/// A single keyframe storing a value of type `T` at a specific time.
///
/// The `in_tangent` and `out_tangent` fields are used only when the
/// interpolation mode is [`Interpolation::Cubic`]; they encode the slope
/// of the Bezier curve entering and leaving this keyframe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Keyframe<T: Clone> {
    /// Time in seconds within the sequence.
    pub time: f32,
    /// The value at this keyframe.
    pub value: T,
    /// Incoming tangent (Cubic only).
    pub in_tangent: Option<T>,
    /// Outgoing tangent (Cubic only).
    pub out_tangent: Option<T>,
    /// How to interpolate *from* this keyframe to the next.
    pub interpolation: Interpolation,
}

impl<T: Clone> Keyframe<T> {
    /// Create a new keyframe with linear interpolation and no tangents.
    pub fn new(time: f32, value: T) -> Self {
        Self {
            time,
            value,
            in_tangent: None,
            out_tangent: None,
            interpolation: Interpolation::Linear,
        }
    }

    /// Create a keyframe with step interpolation.
    pub fn step(time: f32, value: T) -> Self {
        Self {
            time,
            value,
            in_tangent: None,
            out_tangent: None,
            interpolation: Interpolation::Step,
        }
    }

    /// Create a keyframe with cubic interpolation and explicit tangents.
    pub fn cubic(time: f32, value: T, in_tangent: T, out_tangent: T) -> Self {
        Self {
            time,
            value,
            in_tangent: Some(in_tangent),
            out_tangent: Some(out_tangent),
            interpolation: Interpolation::Cubic,
        }
    }
}

// ---------------------------------------------------------------------------
// SequenceTrack<T>
// ---------------------------------------------------------------------------

/// A typed track of keyframes targeting a property on an entity.
///
/// Keyframes are kept sorted by time. The [`evaluate`](SequenceTrack::evaluate)
/// method interpolates between the two surrounding keyframes at a given time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceTrack<T: Clone> {
    /// Unique identifier for this track.
    pub track_id: Uuid,
    /// The entity this track targets.
    #[serde(skip, default = "default_entity")]
    pub target_entity: Entity,
    /// Dot-separated property path (e.g. `"transform.position.x"`).
    pub property_path: String,
    /// Sorted list of keyframes.
    pub keyframes: Vec<Keyframe<T>>,
    /// Human-readable name for editor display.
    pub name: String,
    /// Whether this track is enabled. Disabled tracks are skipped during
    /// evaluation.
    pub enabled: bool,
}

impl<T: Clone> SequenceTrack<T> {
    /// Create a new empty track.
    pub fn new(name: impl Into<String>, target: Entity, property: impl Into<String>) -> Self {
        Self {
            track_id: Uuid::new_v4(),
            target_entity: target,
            property_path: property.into(),
            keyframes: Vec::new(),
            name: name.into(),
            enabled: true,
        }
    }

    /// Insert a keyframe, maintaining time-sorted order.
    pub fn add_keyframe(&mut self, kf: Keyframe<T>) {
        let pos = self
            .keyframes
            .binary_search_by(|probe| probe.time.partial_cmp(&kf.time).unwrap())
            .unwrap_or_else(|e| e);
        self.keyframes.insert(pos, kf);
    }

    /// Remove the keyframe at the given index.
    pub fn remove_keyframe(&mut self, index: usize) -> Result<Keyframe<T>, SequencerError> {
        if index >= self.keyframes.len() {
            return Err(SequencerError::KeyframeOutOfRange(
                index,
                self.keyframes.len(),
            ));
        }
        Ok(self.keyframes.remove(index))
    }

    /// Number of keyframes.
    #[inline]
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// Duration covered by the keyframes (time of last - time of first).
    pub fn duration(&self) -> f32 {
        match (self.keyframes.first(), self.keyframes.last()) {
            (Some(first), Some(last)) => last.time - first.time,
            _ => 0.0,
        }
    }

    /// Find the indices of the two keyframes surrounding `time`.
    /// Returns `(left_index, right_index, local_t)` where `local_t` is in
    /// `[0, 1]` representing the position between the two keyframes.
    pub fn find_surrounding(&self, time: f32) -> Option<(usize, usize, f32)> {
        if self.keyframes.is_empty() {
            return None;
        }
        if self.keyframes.len() == 1 {
            return Some((0, 0, 0.0));
        }
        // Before first keyframe
        if time <= self.keyframes[0].time {
            return Some((0, 0, 0.0));
        }
        // After last keyframe
        if time >= self.keyframes[self.keyframes.len() - 1].time {
            let last = self.keyframes.len() - 1;
            return Some((last, last, 0.0));
        }
        // Binary search for the interval
        for i in 0..self.keyframes.len() - 1 {
            let a = &self.keyframes[i];
            let b = &self.keyframes[i + 1];
            if time >= a.time && time <= b.time {
                let span = b.time - a.time;
                let t = if span > 0.0 {
                    (time - a.time) / span
                } else {
                    0.0
                };
                return Some((i, i + 1, t));
            }
        }
        None
    }

    /// Move all keyframes by a time offset.
    pub fn shift_keyframes(&mut self, offset: f32) {
        for kf in &mut self.keyframes {
            kf.time += offset;
        }
    }

    /// Scale all keyframe times by a factor around a pivot time.
    pub fn scale_keyframes(&mut self, factor: f32, pivot: f32) {
        for kf in &mut self.keyframes {
            kf.time = pivot + (kf.time - pivot) * factor;
        }
    }
}

// ---------------------------------------------------------------------------
// Interpolation helpers
// ---------------------------------------------------------------------------

/// Hermite basis functions for cubic interpolation.
fn hermite_basis(t: f32) -> (f32, f32, f32, f32) {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    (h00, h10, h01, h11)
}

/// Cubic Hermite interpolation for f32.
fn cubic_interp_f32(p0: f32, m0: f32, p1: f32, m1: f32, t: f32) -> f32 {
    let (h00, h10, h01, h11) = hermite_basis(t);
    h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
}

/// Cubic Hermite interpolation for Vec3.
fn cubic_interp_vec3(p0: Vec3, m0: Vec3, p1: Vec3, m1: Vec3, t: f32) -> Vec3 {
    let (h00, h10, h01, h11) = hermite_basis(t);
    p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11
}

/// Cubic Hermite interpolation for Vec4 (used for color).
fn cubic_interp_vec4(p0: Vec4, m0: Vec4, p1: Vec4, m1: Vec4, t: f32) -> Vec4 {
    let (h00, h10, h01, h11) = hermite_basis(t);
    p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11
}

// ---------------------------------------------------------------------------
// Float track evaluation
// ---------------------------------------------------------------------------

/// Track that animates a single float value.
pub type FloatTrack = SequenceTrack<f32>;

impl FloatTrack {
    /// Evaluate the float value at the given time.
    pub fn evaluate(&self, time: f32) -> f32 {
        let Some((i, j, t)) = self.find_surrounding(time) else {
            return 0.0;
        };
        if i == j {
            return self.keyframes[i].value;
        }
        let a = &self.keyframes[i];
        let b = &self.keyframes[j];
        match a.interpolation {
            Interpolation::Step => a.value,
            Interpolation::Linear => {
                let v0 = a.value;
                let v1 = b.value;
                v0 + (v1 - v0) * t
            }
            Interpolation::Cubic => {
                let p0 = a.value;
                let p1 = b.value;
                let m0 = a.out_tangent.unwrap_or(0.0);
                let m1 = b.in_tangent.unwrap_or(0.0);
                let span = b.time - a.time;
                cubic_interp_f32(p0, m0 * span, p1, m1 * span, t)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Transform track
// ---------------------------------------------------------------------------

/// Decomposed transform for keyframing.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TransformValue {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for TransformValue {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

/// Track that animates position, rotation, and scale.
pub type TransformTrack = SequenceTrack<TransformValue>;

impl TransformTrack {
    /// Evaluate the transform at the given time.
    ///
    /// Position and scale are interpolated component-wise; rotation uses
    /// spherical linear interpolation (slerp) for linear mode and a
    /// cubic squad approximation for cubic mode.
    pub fn evaluate(&self, time: f32) -> TransformValue {
        let Some((i, j, t)) = self.find_surrounding(time) else {
            return TransformValue::default();
        };
        if i == j {
            return self.keyframes[i].value;
        }
        let a = &self.keyframes[i];
        let b = &self.keyframes[j];
        match a.interpolation {
            Interpolation::Step => a.value,
            Interpolation::Linear => {
                let pos = a.value.position.lerp(b.value.position, t);
                let rot = a.value.rotation.slerp(b.value.rotation, t);
                let scl = a.value.scale.lerp(b.value.scale, t);
                TransformValue {
                    position: pos,
                    rotation: rot,
                    scale: scl,
                }
            }
            Interpolation::Cubic => {
                // For position and scale we use cubic Hermite;
                // for rotation we still slerp (squad requires 4 quaternions).
                let span = b.time - a.time;
                let default_tan = TransformValue::default();
                let m0 = a.out_tangent.as_ref().unwrap_or(&default_tan);
                let m1 = b.in_tangent.as_ref().unwrap_or(&default_tan);

                let pos = cubic_interp_vec3(
                    a.value.position,
                    m0.position * span,
                    b.value.position,
                    m1.position * span,
                    t,
                );
                let scl = cubic_interp_vec3(
                    a.value.scale,
                    m0.scale * span,
                    b.value.scale,
                    m1.scale * span,
                    t,
                );
                // Quaternion squad approximation: use slerp with smoothed t
                let smooth_t = t * t * (3.0 - 2.0 * t);
                let rot = a.value.rotation.slerp(b.value.rotation, smooth_t);
                TransformValue {
                    position: pos,
                    rotation: rot,
                    scale: scl,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Color track
// ---------------------------------------------------------------------------

/// RGBA color for keyframing. Components in `[0, 1]`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const TRANSPARENT: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Convert sRGB to linear for interpolation.
    pub fn to_linear(self) -> Self {
        Self {
            r: srgb_to_linear(self.r),
            g: srgb_to_linear(self.g),
            b: srgb_to_linear(self.b),
            a: self.a,
        }
    }

    /// Convert linear back to sRGB for display.
    pub fn to_srgb(self) -> Self {
        Self {
            r: linear_to_srgb(self.r),
            g: linear_to_srgb(self.g),
            b: linear_to_srgb(self.b),
            a: self.a,
        }
    }

    fn to_vec4(self) -> Vec4 {
        Vec4::new(self.r, self.g, self.b, self.a)
    }

    fn from_vec4(v: Vec4) -> Self {
        Self {
            r: v.x,
            g: v.y,
            b: v.z,
            a: v.w,
        }
    }

    /// Linearly interpolate two colors in linear color space.
    pub fn lerp_linear(a: Self, b: Self, t: f32) -> Self {
        let a_lin = a.to_linear().to_vec4();
        let b_lin = b.to_linear().to_vec4();
        let result = a_lin.lerp(b_lin, t);
        Self::from_vec4(result).to_srgb()
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::WHITE
    }
}

/// sRGB gamma curve: sRGB -> linear.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear -> sRGB gamma curve.
fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Track that animates an RGBA color with proper color-space interpolation.
pub type ColorTrack = SequenceTrack<Color>;

impl ColorTrack {
    /// Evaluate the color at the given time.
    ///
    /// Interpolation is performed in linear color space and the result
    /// is converted back to sRGB.
    pub fn evaluate(&self, time: f32) -> Color {
        let Some((i, j, t)) = self.find_surrounding(time) else {
            return Color::WHITE;
        };
        if i == j {
            return self.keyframes[i].value;
        }
        let a = &self.keyframes[i];
        let b = &self.keyframes[j];
        match a.interpolation {
            Interpolation::Step => a.value,
            Interpolation::Linear => Color::lerp_linear(a.value, b.value, t),
            Interpolation::Cubic => {
                let span = b.time - a.time;
                let p0 = a.value.to_linear().to_vec4();
                let p1 = b.value.to_linear().to_vec4();
                let m0 = a
                    .out_tangent
                    .map(|c| c.to_vec4())
                    .unwrap_or(Vec4::ZERO)
                    * span;
                let m1 = b
                    .in_tangent
                    .map(|c| c.to_vec4())
                    .unwrap_or(Vec4::ZERO)
                    * span;
                let result = cubic_interp_vec4(p0, m0, p1, m1, t);
                Color::from_vec4(result).to_srgb()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Bool track
// ---------------------------------------------------------------------------

/// Track that toggles a boolean value at specific times.
pub type BoolTrack = SequenceTrack<bool>;

impl BoolTrack {
    /// Evaluate the boolean at the given time.
    ///
    /// Bool tracks always use step interpolation regardless of the keyframe
    /// setting -- a bool cannot be "between" true and false.
    pub fn evaluate(&self, time: f32) -> bool {
        if self.keyframes.is_empty() {
            return false;
        }
        // Find the last keyframe at or before `time`.
        let mut result = self.keyframes[0].value;
        for kf in &self.keyframes {
            if kf.time <= time {
                result = kf.value;
            } else {
                break;
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Event track
// ---------------------------------------------------------------------------

/// A named event that fires at a specific time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceEvent {
    /// Identifier for the event.
    pub name: String,
    /// Arbitrary string payload (JSON, command name, etc.).
    pub payload: String,
}

impl Default for SequenceEvent {
    fn default() -> Self {
        Self {
            name: String::new(),
            payload: String::new(),
        }
    }
}

/// Track that fires named events at specific times.
pub type EventTrack = SequenceTrack<SequenceEvent>;

impl EventTrack {
    /// Collect all events whose time falls within `(prev_time, current_time]`.
    ///
    /// This is used every frame to determine which events should fire.
    pub fn collect_events(&self, prev_time: f32, current_time: f32) -> Vec<&SequenceEvent> {
        let (lo, hi) = if current_time >= prev_time {
            (prev_time, current_time)
        } else {
            // Handle wraparound for looping sequences.
            (prev_time, current_time)
        };
        self.keyframes
            .iter()
            .filter(|kf| kf.time > lo && kf.time <= hi)
            .map(|kf| &kf.value)
            .collect()
    }

    /// Collect all events in the track (for editor preview).
    pub fn all_events(&self) -> Vec<(f32, &SequenceEvent)> {
        self.keyframes
            .iter()
            .map(|kf| (kf.time, &kf.value))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Camera track
// ---------------------------------------------------------------------------

/// Camera properties that can be animated over time.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CameraProperties {
    /// Field of view in degrees.
    pub fov: f32,
    /// Near clipping plane distance.
    pub near: f32,
    /// Far clipping plane distance.
    pub far: f32,
    /// Focus distance for depth-of-field effects.
    pub focus_distance: f32,
    /// Aperture (f-stop) for depth-of-field.
    pub aperture: f32,
    /// Focal length in mm (for cinematic look).
    pub focal_length: f32,
}

impl Default for CameraProperties {
    fn default() -> Self {
        Self {
            fov: 60.0,
            near: 0.1,
            far: 1000.0,
            focus_distance: 10.0,
            aperture: 2.8,
            focal_length: 50.0,
        }
    }
}

/// Track that animates camera properties (FOV, near/far, focus).
pub type CameraTrack = SequenceTrack<CameraProperties>;

impl CameraTrack {
    /// Evaluate camera properties at the given time.
    pub fn evaluate(&self, time: f32) -> CameraProperties {
        let Some((i, j, t)) = self.find_surrounding(time) else {
            return CameraProperties::default();
        };
        if i == j {
            return self.keyframes[i].value;
        }
        let a = &self.keyframes[i];
        let b = &self.keyframes[j];
        match a.interpolation {
            Interpolation::Step => a.value,
            Interpolation::Linear => CameraProperties {
                fov: a.value.fov + (b.value.fov - a.value.fov) * t,
                near: a.value.near + (b.value.near - a.value.near) * t,
                far: a.value.far + (b.value.far - a.value.far) * t,
                focus_distance: a.value.focus_distance
                    + (b.value.focus_distance - a.value.focus_distance) * t,
                aperture: a.value.aperture + (b.value.aperture - a.value.aperture) * t,
                focal_length: a.value.focal_length
                    + (b.value.focal_length - a.value.focal_length) * t,
            },
            Interpolation::Cubic => {
                let span = b.time - a.time;
                let def = CameraProperties::default();
                let m0 = a.out_tangent.as_ref().unwrap_or(&def);
                let m1 = b.in_tangent.as_ref().unwrap_or(&def);
                CameraProperties {
                    fov: cubic_interp_f32(a.value.fov, m0.fov * span, b.value.fov, m1.fov * span, t),
                    near: cubic_interp_f32(
                        a.value.near,
                        m0.near * span,
                        b.value.near,
                        m1.near * span,
                        t,
                    ),
                    far: cubic_interp_f32(a.value.far, m0.far * span, b.value.far, m1.far * span, t),
                    focus_distance: cubic_interp_f32(
                        a.value.focus_distance,
                        m0.focus_distance * span,
                        b.value.focus_distance,
                        m1.focus_distance * span,
                        t,
                    ),
                    aperture: cubic_interp_f32(
                        a.value.aperture,
                        m0.aperture * span,
                        b.value.aperture,
                        m1.aperture * span,
                        t,
                    ),
                    focal_length: cubic_interp_f32(
                        a.value.focal_length,
                        m0.focal_length * span,
                        b.value.focal_length,
                        m1.focal_length * span,
                        t,
                    ),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Audio track
// ---------------------------------------------------------------------------

/// An audio cue to trigger at a specific time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioCue {
    /// Asset path or identifier for the audio clip.
    pub clip_path: String,
    /// Volume multiplier in `[0, 1]`.
    pub volume: f32,
    /// Pitch multiplier (1.0 = normal).
    pub pitch: f32,
    /// Whether the audio is spatial (3D) or 2D.
    pub spatial: bool,
    /// Fade-in duration in seconds.
    pub fade_in: f32,
    /// Fade-out duration in seconds.
    pub fade_out: f32,
    /// If true, loop the audio clip.
    pub looping: bool,
}

impl Default for AudioCue {
    fn default() -> Self {
        Self {
            clip_path: String::new(),
            volume: 1.0,
            pitch: 1.0,
            spatial: false,
            fade_in: 0.0,
            fade_out: 0.0,
            looping: false,
        }
    }
}

/// Track that triggers audio playback at specific times.
pub type AudioTrack = SequenceTrack<AudioCue>;

impl AudioTrack {
    /// Collect audio cues that should start within `(prev_time, current_time]`.
    pub fn collect_cues(&self, prev_time: f32, current_time: f32) -> Vec<&AudioCue> {
        self.keyframes
            .iter()
            .filter(|kf| kf.time > prev_time && kf.time <= current_time)
            .map(|kf| &kf.value)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Sub-sequence track
// ---------------------------------------------------------------------------

/// Reference to a child sequence to embed within a parent sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubSequenceRef {
    /// Identifier for the child sequence asset.
    pub sequence_id: Uuid,
    /// Name for display.
    pub name: String,
    /// Time offset within the child to start playback.
    pub start_offset: f32,
    /// Playback speed multiplier for the child.
    pub speed: f32,
    /// Whether to apply the child's loop mode or clamp.
    pub inherit_loop: bool,
}

impl Default for SubSequenceRef {
    fn default() -> Self {
        Self {
            sequence_id: Uuid::nil(),
            name: String::new(),
            start_offset: 0.0,
            speed: 1.0,
            inherit_loop: false,
        }
    }
}

/// Track that embeds child sequences within a parent sequence.
pub type SubSequenceTrack = SequenceTrack<SubSequenceRef>;

impl SubSequenceTrack {
    /// Collect sub-sequences that should be active at the given time.
    pub fn active_subsequences(&self, time: f32) -> Vec<&SubSequenceRef> {
        // A sub-sequence is considered "active" if time >= its keyframe time.
        // In practice, the actual end-time depends on the child duration,
        // but we don't have that here. The player layer handles stopping.
        self.keyframes
            .iter()
            .filter(|kf| kf.time <= time)
            .map(|kf| &kf.value)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// TrackKind (type-erased wrapper)
// ---------------------------------------------------------------------------

/// A type-erased container for any track type, used by [`Sequence`] to store
/// heterogeneous tracks in a single list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrackKind {
    Transform(TransformTrack),
    Float(FloatTrack),
    Color(ColorTrack),
    Bool(BoolTrack),
    Event(EventTrack),
    Camera(CameraTrack),
    Audio(AudioTrack),
    SubSequence(SubSequenceTrack),
}

impl TrackKind {
    /// Get the track ID regardless of variant.
    pub fn track_id(&self) -> Uuid {
        match self {
            Self::Transform(t) => t.track_id,
            Self::Float(t) => t.track_id,
            Self::Color(t) => t.track_id,
            Self::Bool(t) => t.track_id,
            Self::Event(t) => t.track_id,
            Self::Camera(t) => t.track_id,
            Self::Audio(t) => t.track_id,
            Self::SubSequence(t) => t.track_id,
        }
    }

    /// Get the track name regardless of variant.
    pub fn name(&self) -> &str {
        match self {
            Self::Transform(t) => &t.name,
            Self::Float(t) => &t.name,
            Self::Color(t) => &t.name,
            Self::Bool(t) => &t.name,
            Self::Event(t) => &t.name,
            Self::Camera(t) => &t.name,
            Self::Audio(t) => &t.name,
            Self::SubSequence(t) => &t.name,
        }
    }

    /// Get the property path regardless of variant.
    pub fn property_path(&self) -> &str {
        match self {
            Self::Transform(t) => &t.property_path,
            Self::Float(t) => &t.property_path,
            Self::Color(t) => &t.property_path,
            Self::Bool(t) => &t.property_path,
            Self::Event(t) => &t.property_path,
            Self::Camera(t) => &t.property_path,
            Self::Audio(t) => &t.property_path,
            Self::SubSequence(t) => &t.property_path,
        }
    }

    /// Is this track enabled?
    pub fn is_enabled(&self) -> bool {
        match self {
            Self::Transform(t) => t.enabled,
            Self::Float(t) => t.enabled,
            Self::Color(t) => t.enabled,
            Self::Bool(t) => t.enabled,
            Self::Event(t) => t.enabled,
            Self::Camera(t) => t.enabled,
            Self::Audio(t) => t.enabled,
            Self::SubSequence(t) => t.enabled,
        }
    }

    /// Set the enabled state.
    pub fn set_enabled(&mut self, enabled: bool) {
        match self {
            Self::Transform(t) => t.enabled = enabled,
            Self::Float(t) => t.enabled = enabled,
            Self::Color(t) => t.enabled = enabled,
            Self::Bool(t) => t.enabled = enabled,
            Self::Event(t) => t.enabled = enabled,
            Self::Camera(t) => t.enabled = enabled,
            Self::Audio(t) => t.enabled = enabled,
            Self::SubSequence(t) => t.enabled = enabled,
        }
    }

    /// Get the number of keyframes.
    pub fn keyframe_count(&self) -> usize {
        match self {
            Self::Transform(t) => t.keyframe_count(),
            Self::Float(t) => t.keyframe_count(),
            Self::Color(t) => t.keyframe_count(),
            Self::Bool(t) => t.keyframe_count(),
            Self::Event(t) => t.keyframe_count(),
            Self::Camera(t) => t.keyframe_count(),
            Self::Audio(t) => t.keyframe_count(),
            Self::SubSequence(t) => t.keyframe_count(),
        }
    }

    /// Type name for display.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Transform(_) => "Transform",
            Self::Float(_) => "Float",
            Self::Color(_) => "Color",
            Self::Bool(_) => "Bool",
            Self::Event(_) => "Event",
            Self::Camera(_) => "Camera",
            Self::Audio(_) => "Audio",
            Self::SubSequence(_) => "SubSequence",
        }
    }
}

// ---------------------------------------------------------------------------
// Loop mode
// ---------------------------------------------------------------------------

/// Playback loop behaviour for a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoopMode {
    /// Play once and clamp at the end.
    Clamp,
    /// Loop from the beginning when the end is reached.
    Loop,
    /// Alternate forward/backward on each cycle.
    PingPong,
}

impl Default for LoopMode {
    fn default() -> Self {
        Self::Clamp
    }
}

// ---------------------------------------------------------------------------
// Sequence
// ---------------------------------------------------------------------------

/// A complete cinematic sequence containing multiple typed tracks.
///
/// A sequence has a fixed duration and a collection of tracks. During
/// playback the [`SequencePlayer`] evaluates every enabled track at the
/// current time and applies the results to the targeted entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sequence {
    /// Unique identifier.
    pub id: Uuid,
    /// Human-readable name.
    pub name: String,
    /// Total duration in seconds.
    pub duration: f32,
    /// All tracks in this sequence.
    pub tracks: Vec<TrackKind>,
    /// Playback loop mode.
    pub loop_mode: LoopMode,
    /// Default playback speed.
    pub playback_speed: f32,
    /// Metadata tags for categorization.
    pub tags: Vec<String>,
}

impl Sequence {
    /// Create a new empty sequence.
    pub fn new(name: impl Into<String>, duration: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            duration,
            tracks: Vec::new(),
            loop_mode: LoopMode::Clamp,
            playback_speed: 1.0,
            tags: Vec::new(),
        }
    }

    /// Add a track to the sequence.
    pub fn add_track(&mut self, track: TrackKind) {
        self.tracks.push(track);
    }

    /// Remove a track by ID.
    pub fn remove_track(&mut self, track_id: Uuid) -> Option<TrackKind> {
        if let Some(pos) = self.tracks.iter().position(|t| t.track_id() == track_id) {
            Some(self.tracks.remove(pos))
        } else {
            None
        }
    }

    /// Find a track by ID.
    pub fn find_track(&self, track_id: Uuid) -> Option<&TrackKind> {
        self.tracks.iter().find(|t| t.track_id() == track_id)
    }

    /// Find a track by ID (mutable).
    pub fn find_track_mut(&mut self, track_id: Uuid) -> Option<&mut TrackKind> {
        self.tracks.iter_mut().find(|t| t.track_id() == track_id)
    }

    /// Get all tracks of a specific type.
    pub fn tracks_of_type(&self, type_name: &str) -> Vec<&TrackKind> {
        self.tracks
            .iter()
            .filter(|t| t.type_name() == type_name)
            .collect()
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, SequencerError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| SequencerError::SerializationError(e.to_string()))
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, SequencerError> {
        serde_json::from_str(json)
            .map_err(|e| SequencerError::SerializationError(e.to_string()))
    }
}

impl fmt::Display for Sequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Sequence(\"{}\" {:.2}s, {} tracks, {:?})",
            self.name,
            self.duration,
            self.tracks.len(),
            self.loop_mode
        )
    }
}

// ---------------------------------------------------------------------------
// SequencePlayer
// ---------------------------------------------------------------------------

/// Playback state for a sequence player.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackState {
    /// Not playing; time frozen.
    Stopped,
    /// Actively advancing time.
    Playing,
    /// Time frozen but position remembered.
    Paused,
}

/// Callback type for track events.
pub type EventCallback = Box<dyn Fn(&SequenceEvent) + Send + Sync>;

/// Runtime player that evaluates a [`Sequence`] over time.
///
/// The player manages playback state, applies loop/ping-pong wrapping, and
/// evaluates every enabled track each frame. Track results are stored in
/// a result cache that can be queried or applied to entities.
pub struct SequencePlayer {
    /// The sequence being played.
    sequence: Sequence,
    /// Current playback time in seconds.
    current_time: f32,
    /// Previous frame's time (for event detection).
    previous_time: f32,
    /// Playback state.
    state: PlaybackState,
    /// Speed multiplier (negative = reverse).
    speed: f32,
    /// Whether the current ping-pong cycle is going backwards.
    ping_pong_reverse: bool,
    /// Number of completed loops.
    loop_count: u32,
    /// Cached float results: track_id -> value.
    float_results: HashMap<Uuid, f32>,
    /// Cached transform results: track_id -> value.
    transform_results: HashMap<Uuid, TransformValue>,
    /// Cached color results: track_id -> value.
    color_results: HashMap<Uuid, Color>,
    /// Cached bool results: track_id -> value.
    bool_results: HashMap<Uuid, bool>,
    /// Cached camera results: track_id -> value.
    camera_results: HashMap<Uuid, CameraProperties>,
    /// Pending events collected this frame.
    pending_events: Vec<SequenceEvent>,
    /// Pending audio cues collected this frame.
    pending_audio: Vec<AudioCue>,
    /// Event callbacks registered by name.
    event_callbacks: HashMap<String, Vec<EventCallback>>,
    /// Entity bindings: track_id -> entity override.
    entity_bindings: HashMap<Uuid, Entity>,
}

impl SequencePlayer {
    /// Create a new player for the given sequence.
    pub fn new(sequence: Sequence) -> Self {
        let speed = sequence.playback_speed;
        Self {
            sequence,
            current_time: 0.0,
            previous_time: 0.0,
            state: PlaybackState::Stopped,
            speed,
            ping_pong_reverse: false,
            loop_count: 0,
            float_results: HashMap::new(),
            transform_results: HashMap::new(),
            color_results: HashMap::new(),
            bool_results: HashMap::new(),
            camera_results: HashMap::new(),
            pending_events: Vec::new(),
            pending_audio: Vec::new(),
            event_callbacks: HashMap::new(),
            entity_bindings: HashMap::new(),
        }
    }

    /// Start or resume playback from current position.
    pub fn play(&mut self) {
        self.state = PlaybackState::Playing;
        log::debug!(
            "Sequence \"{}\" playing at t={:.3}",
            self.sequence.name,
            self.current_time
        );
    }

    /// Pause playback (retains current time).
    pub fn pause(&mut self) {
        self.state = PlaybackState::Paused;
        log::debug!(
            "Sequence \"{}\" paused at t={:.3}",
            self.sequence.name,
            self.current_time
        );
    }

    /// Stop playback and reset to the beginning.
    pub fn stop(&mut self) {
        self.state = PlaybackState::Stopped;
        self.current_time = 0.0;
        self.previous_time = 0.0;
        self.ping_pong_reverse = false;
        self.loop_count = 0;
        self.clear_results();
        log::debug!("Sequence \"{}\" stopped", self.sequence.name);
    }

    /// Seek to a specific time.
    pub fn seek(&mut self, time: f32) {
        self.previous_time = self.current_time;
        self.current_time = time.clamp(0.0, self.sequence.duration);
    }

    /// Set the playback speed multiplier.
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
    }

    /// Get the current playback speed.
    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// Whether the player is currently playing.
    pub fn is_playing(&self) -> bool {
        self.state == PlaybackState::Playing
    }

    /// Whether the player is paused.
    pub fn is_paused(&self) -> bool {
        self.state == PlaybackState::Paused
    }

    /// Whether the player is stopped.
    pub fn is_stopped(&self) -> bool {
        self.state == PlaybackState::Stopped
    }

    /// Current playback state.
    pub fn playback_state(&self) -> PlaybackState {
        self.state
    }

    /// Current time in seconds.
    pub fn current_time(&self) -> f32 {
        self.current_time
    }

    /// Normalized time in `[0, 1]` (current_time / duration).
    pub fn normalized_time(&self) -> f32 {
        if self.sequence.duration > 0.0 {
            self.current_time / self.sequence.duration
        } else {
            0.0
        }
    }

    /// Number of completed loops.
    pub fn loop_count(&self) -> u32 {
        self.loop_count
    }

    /// Duration of the underlying sequence.
    pub fn duration(&self) -> f32 {
        self.sequence.duration
    }

    /// Immutable reference to the underlying sequence.
    pub fn sequence(&self) -> &Sequence {
        &self.sequence
    }

    /// Mutable reference to the underlying sequence.
    pub fn sequence_mut(&mut self) -> &mut Sequence {
        &mut self.sequence
    }

    /// Bind a track to a different entity at runtime.
    pub fn bind_entity(&mut self, track_id: Uuid, entity: Entity) {
        self.entity_bindings.insert(track_id, entity);
    }

    /// Remove an entity binding.
    pub fn unbind_entity(&mut self, track_id: Uuid) {
        self.entity_bindings.remove(&track_id);
    }

    /// Register a callback for a named event.
    pub fn on_event(&mut self, event_name: impl Into<String>, callback: EventCallback) {
        self.event_callbacks
            .entry(event_name.into())
            .or_default()
            .push(callback);
    }

    /// Get pending events from the last `update` call.
    pub fn pending_events(&self) -> &[SequenceEvent] {
        &self.pending_events
    }

    /// Get pending audio cues from the last `update` call.
    pub fn pending_audio(&self) -> &[AudioCue] {
        &self.pending_audio
    }

    /// Get a cached float result by track ID.
    pub fn get_float(&self, track_id: Uuid) -> Option<f32> {
        self.float_results.get(&track_id).copied()
    }

    /// Get a cached transform result by track ID.
    pub fn get_transform(&self, track_id: Uuid) -> Option<&TransformValue> {
        self.transform_results.get(&track_id)
    }

    /// Get a cached color result by track ID.
    pub fn get_color(&self, track_id: Uuid) -> Option<&Color> {
        self.color_results.get(&track_id)
    }

    /// Get a cached bool result by track ID.
    pub fn get_bool(&self, track_id: Uuid) -> Option<bool> {
        self.bool_results.get(&track_id).copied()
    }

    /// Get a cached camera properties result by track ID.
    pub fn get_camera(&self, track_id: Uuid) -> Option<&CameraProperties> {
        self.camera_results.get(&track_id)
    }

    /// Advance time by `dt` seconds and evaluate all tracks.
    ///
    /// This is the main update entry point. Call once per frame.
    pub fn update(&mut self, dt: f32) {
        if self.state != PlaybackState::Playing {
            return;
        }

        self.previous_time = self.current_time;
        self.pending_events.clear();
        self.pending_audio.clear();

        // Advance time
        let effective_dt = dt * self.speed;
        if self.ping_pong_reverse {
            self.current_time -= effective_dt;
        } else {
            self.current_time += effective_dt;
        }

        // Handle end-of-sequence
        self.apply_loop_mode();

        // Evaluate all tracks
        self.evaluate_all_tracks();

        // Fire event callbacks
        self.fire_event_callbacks();
    }

    /// Apply the loop mode when time goes out of bounds.
    fn apply_loop_mode(&mut self) {
        let dur = self.sequence.duration;
        if dur <= 0.0 {
            self.current_time = 0.0;
            return;
        }

        match self.sequence.loop_mode {
            LoopMode::Clamp => {
                if self.current_time >= dur {
                    self.current_time = dur;
                    self.state = PlaybackState::Stopped;
                    log::debug!(
                        "Sequence \"{}\" finished (clamped)",
                        self.sequence.name
                    );
                } else if self.current_time < 0.0 {
                    self.current_time = 0.0;
                    self.state = PlaybackState::Stopped;
                }
            }
            LoopMode::Loop => {
                if self.current_time >= dur {
                    self.loop_count += 1;
                    self.current_time %= dur;
                    log::trace!(
                        "Sequence \"{}\" looped (count={})",
                        self.sequence.name,
                        self.loop_count
                    );
                } else if self.current_time < 0.0 {
                    self.loop_count += 1;
                    self.current_time = dur + (self.current_time % dur);
                }
            }
            LoopMode::PingPong => {
                if self.current_time >= dur {
                    self.current_time = dur - (self.current_time - dur);
                    self.ping_pong_reverse = true;
                    self.loop_count += 1;
                    log::trace!(
                        "Sequence \"{}\" ping-pong reversed (count={})",
                        self.sequence.name,
                        self.loop_count
                    );
                } else if self.current_time <= 0.0 {
                    self.current_time = -self.current_time;
                    self.ping_pong_reverse = false;
                    self.loop_count += 1;
                }
            }
        }
    }

    /// Evaluate every enabled track at the current time.
    fn evaluate_all_tracks(&mut self) {
        let time = self.current_time;
        let prev = self.previous_time;

        for track in &self.sequence.tracks {
            if !track.is_enabled() {
                continue;
            }
            match track {
                TrackKind::Float(t) => {
                    let val = t.evaluate(time);
                    self.float_results.insert(t.track_id, val);
                }
                TrackKind::Transform(t) => {
                    let val = t.evaluate(time);
                    self.transform_results.insert(t.track_id, val);
                }
                TrackKind::Color(t) => {
                    let val = t.evaluate(time);
                    self.color_results.insert(t.track_id, val);
                }
                TrackKind::Bool(t) => {
                    let val = t.evaluate(time);
                    self.bool_results.insert(t.track_id, val);
                }
                TrackKind::Camera(t) => {
                    let val = t.evaluate(time);
                    self.camera_results.insert(t.track_id, val);
                }
                TrackKind::Event(t) => {
                    let events = t.collect_events(prev, time);
                    for evt in events {
                        self.pending_events.push(evt.clone());
                    }
                }
                TrackKind::Audio(t) => {
                    let cues = t.collect_cues(prev, time);
                    for cue in cues {
                        self.pending_audio.push(cue.clone());
                    }
                }
                TrackKind::SubSequence(_) => {
                    // Sub-sequence activation is handled by the outer
                    // orchestration layer, not the track evaluator.
                }
            }
        }
    }

    /// Invoke registered callbacks for pending events.
    fn fire_event_callbacks(&self) {
        for evt in &self.pending_events {
            if let Some(callbacks) = self.event_callbacks.get(&evt.name) {
                for cb in callbacks {
                    cb(evt);
                }
            }
        }
    }

    /// Clear all cached results.
    fn clear_results(&mut self) {
        self.float_results.clear();
        self.transform_results.clear();
        self.color_results.clear();
        self.bool_results.clear();
        self.camera_results.clear();
        self.pending_events.clear();
        self.pending_audio.clear();
    }
}

impl fmt::Debug for SequencePlayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SequencePlayer")
            .field("sequence", &self.sequence.name)
            .field("state", &self.state)
            .field("current_time", &self.current_time)
            .field("speed", &self.speed)
            .field("loop_count", &self.loop_count)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entity() -> Entity {
        Entity::new(0, 0)
    }

    #[test]
    fn float_track_linear() {
        let mut track = FloatTrack::new("opacity", make_entity(), "material.opacity");
        track.add_keyframe(Keyframe::new(0.0, 0.0));
        track.add_keyframe(Keyframe::new(1.0, 1.0));

        assert!((track.evaluate(0.0) - 0.0).abs() < 1e-5);
        assert!((track.evaluate(0.5) - 0.5).abs() < 1e-5);
        assert!((track.evaluate(1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn float_track_step() {
        let mut track = FloatTrack::new("toggle_val", make_entity(), "prop");
        track.add_keyframe(Keyframe::step(0.0, 10.0));
        track.add_keyframe(Keyframe::step(1.0, 20.0));

        assert!((track.evaluate(0.0) - 10.0).abs() < 1e-5);
        assert!((track.evaluate(0.5) - 10.0).abs() < 1e-5);
        assert!((track.evaluate(1.0) - 20.0).abs() < 1e-5);
    }

    #[test]
    fn float_track_cubic() {
        let mut track = FloatTrack::new("smooth", make_entity(), "prop");
        track.add_keyframe(Keyframe::cubic(0.0, 0.0, 0.0, 1.0));
        track.add_keyframe(Keyframe::cubic(1.0, 1.0, 1.0, 0.0));

        let mid = track.evaluate(0.5);
        assert!(mid > 0.0 && mid < 1.0, "cubic midpoint should be in (0,1): {mid}");
    }

    #[test]
    fn bool_track() {
        let mut track = BoolTrack::new("visible", make_entity(), "renderer.visible");
        track.add_keyframe(Keyframe::new(0.0, false));
        track.add_keyframe(Keyframe::new(1.0, true));

        assert!(!track.evaluate(0.0));
        assert!(!track.evaluate(0.5));
        assert!(track.evaluate(1.0));
    }

    #[test]
    fn event_track_collection() {
        let mut track = EventTrack::new("events", make_entity(), "events");
        track.add_keyframe(Keyframe::new(
            0.5,
            SequenceEvent {
                name: "explosion".into(),
                payload: "{}".into(),
            },
        ));
        track.add_keyframe(Keyframe::new(
            1.5,
            SequenceEvent {
                name: "shake".into(),
                payload: "{}".into(),
            },
        ));

        let events = track.collect_events(0.0, 1.0);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "explosion");
    }

    #[test]
    fn sequence_player_clamp() {
        let mut seq = Sequence::new("test", 1.0);
        let mut ft = FloatTrack::new("val", make_entity(), "prop");
        ft.add_keyframe(Keyframe::new(0.0, 0.0));
        ft.add_keyframe(Keyframe::new(1.0, 100.0));
        let tid = ft.track_id;
        seq.add_track(TrackKind::Float(ft));

        let mut player = SequencePlayer::new(seq);
        player.play();
        player.update(0.5);
        assert!((player.get_float(tid).unwrap() - 50.0).abs() < 1.0);
        player.update(1.0);
        assert!(player.is_stopped());
    }

    #[test]
    fn sequence_player_loop() {
        let mut seq = Sequence::new("loop_test", 1.0);
        seq.loop_mode = LoopMode::Loop;
        let player_seq = seq.clone();
        let mut player = SequencePlayer::new(player_seq);
        player.play();
        player.update(1.5);
        assert!(player.is_playing());
        assert!(player.current_time() < 1.0);
        assert_eq!(player.loop_count(), 1);
    }

    #[test]
    fn transform_track_linear() {
        let mut track = TransformTrack::new("cam_xform", make_entity(), "transform");
        track.add_keyframe(Keyframe::new(
            0.0,
            TransformValue {
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
        ));
        track.add_keyframe(Keyframe::new(
            1.0,
            TransformValue {
                position: Vec3::new(10.0, 0.0, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
        ));

        let mid = track.evaluate(0.5);
        assert!((mid.position.x - 5.0).abs() < 1e-4);
    }

    #[test]
    fn color_track_linear() {
        let mut track = ColorTrack::new("light_color", make_entity(), "light.color");
        track.add_keyframe(Keyframe::new(0.0, Color::BLACK));
        track.add_keyframe(Keyframe::new(1.0, Color::WHITE));

        let mid = track.evaluate(0.5);
        // Due to sRGB <-> linear conversion the midpoint will not be exactly 0.5
        assert!(mid.r > 0.1 && mid.r < 0.9);
    }
}
