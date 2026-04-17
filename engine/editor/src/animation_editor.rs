//! Animation timeline editor for the Genovo editor.
//!
//! Provides a track-based animation editing system with keyframe placement,
//! curve visualization, tangent handles, animation preview, event placement,
//! and blend tree visualization.
//!
//! # Architecture
//!
//! An animation is organized as a collection of **tracks**, each targeting a
//! specific property on a specific entity/component. Each track contains
//! **keyframes** at specific times, and the system interpolates between them
//! using configurable curve types (linear, bezier, step, etc.).
//!
//! The editor provides:
//! - Timeline view with zoom and scroll
//! - Keyframe insertion, deletion, and drag manipulation
//! - Curve editor view with tangent handle editing
//! - Animation preview (play, pause, scrub, loop)
//! - Animation events (callbacks at specific times)
//! - Blend tree graph for combining animations
//!
//! # Example
//!
//! ```ignore
//! let mut editor = AnimationEditor::new();
//! let clip_id = editor.create_clip("walk_cycle", 1.0);
//!
//! let track_id = editor.add_track(clip_id, TrackTarget {
//!     entity: "player".into(),
//!     component: "Transform".into(),
//!     property: "position.y".into(),
//! });
//!
//! editor.insert_keyframe(clip_id, track_id, 0.0, KeyframeValue::Float(0.0));
//! editor.insert_keyframe(clip_id, track_id, 0.5, KeyframeValue::Float(1.5));
//! editor.insert_keyframe(clip_id, track_id, 1.0, KeyframeValue::Float(0.0));
//! ```

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Unique identifier for an animation clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClipId(pub u64);

impl ClipId {
    pub fn new(id: u64) -> Self { Self(id) }
    pub fn is_null(&self) -> bool { self.0 == 0 }
}

impl fmt::Display for ClipId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Clip({})", self.0)
    }
}

/// Unique identifier for a track within a clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TrackId(pub u64);

impl TrackId {
    pub fn new(id: u64) -> Self { Self(id) }
    pub fn is_null(&self) -> bool { self.0 == 0 }
}

impl fmt::Display for TrackId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Track({})", self.0)
    }
}

/// Unique identifier for a keyframe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KeyframeId(pub u64);

impl KeyframeId {
    pub fn new(id: u64) -> Self { Self(id) }
}

impl fmt::Display for KeyframeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KF({})", self.0)
    }
}

/// Unique identifier for an animation event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventId(pub u64);

impl fmt::Display for EventId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Event({})", self.0)
    }
}

/// Unique identifier for a blend tree node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlendNodeId(pub u64);

impl BlendNodeId {
    pub fn new(id: u64) -> Self { Self(id) }
}

impl fmt::Display for BlendNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlendNode({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Keyframe values and interpolation
// ---------------------------------------------------------------------------

/// The value stored at a keyframe.
#[derive(Debug, Clone, PartialEq)]
pub enum KeyframeValue {
    /// Single float value.
    Float(f64),
    /// 2D vector.
    Vec2(f64, f64),
    /// 3D vector.
    Vec3(f64, f64, f64),
    /// 4D vector / quaternion.
    Vec4(f64, f64, f64, f64),
    /// Color (RGBA).
    Color(f32, f32, f32, f32),
    /// Boolean (for visibility, enable/disable).
    Bool(bool),
    /// Integer.
    Int(i64),
    /// String (for sprite frame names, etc.).
    String(String),
}

impl KeyframeValue {
    /// Returns the numeric dimension of this value (1 for float, 3 for vec3, etc.).
    pub fn dimension(&self) -> usize {
        match self {
            Self::Float(_) | Self::Int(_) | Self::Bool(_) => 1,
            Self::Vec2(_, _) => 2,
            Self::Vec3(_, _, _) => 3,
            Self::Vec4(_, _, _, _) | Self::Color(_, _, _, _) => 4,
            Self::String(_) => 0,
        }
    }

    /// Returns `true` if this value can be interpolated.
    pub fn is_interpolatable(&self) -> bool {
        !matches!(self, Self::Bool(_) | Self::String(_) | Self::Int(_))
    }

    /// Try to extract a float value.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Linearly interpolate between two values at factor t (0..1).
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        match (self, other) {
            (Self::Float(a), Self::Float(b)) => Self::Float(a + (b - a) * t),
            (Self::Vec2(ax, ay), Self::Vec2(bx, by)) => {
                Self::Vec2(ax + (bx - ax) * t, ay + (by - ay) * t)
            }
            (Self::Vec3(ax, ay, az), Self::Vec3(bx, by, bz)) => Self::Vec3(
                ax + (bx - ax) * t,
                ay + (by - ay) * t,
                az + (bz - az) * t,
            ),
            (Self::Vec4(ax, ay, az, aw), Self::Vec4(bx, by, bz, bw)) => Self::Vec4(
                ax + (bx - ax) * t,
                ay + (by - ay) * t,
                az + (bz - az) * t,
                aw + (bw - aw) * t,
            ),
            (Self::Color(ar, ag, ab, aa), Self::Color(br, bg, bb, ba)) => {
                let t32 = t as f32;
                Self::Color(
                    ar + (br - ar) * t32,
                    ag + (bg - ag) * t32,
                    ab + (bb - ab) * t32,
                    aa + (ba - aa) * t32,
                )
            }
            _ => {
                // Non-interpolatable: step at t >= 0.5.
                if t < 0.5 {
                    self.clone()
                } else {
                    other.clone()
                }
            }
        }
    }
}

impl fmt::Display for KeyframeValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float(v) => write!(f, "{v:.4}"),
            Self::Vec2(x, y) => write!(f, "({x:.3}, {y:.3})"),
            Self::Vec3(x, y, z) => write!(f, "({x:.3}, {y:.3}, {z:.3})"),
            Self::Vec4(x, y, z, w) => write!(f, "({x:.3}, {y:.3}, {z:.3}, {w:.3})"),
            Self::Color(r, g, b, a) => write!(f, "rgba({r:.2}, {g:.2}, {b:.2}, {a:.2})"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::Int(v) => write!(f, "{v}"),
            Self::String(s) => write!(f, "\"{s}\""),
        }
    }
}

/// Curve interpolation mode for a keyframe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CurveMode {
    /// Linear interpolation between keyframes.
    Linear,
    /// Constant (step) — holds the value until the next keyframe.
    Step,
    /// Bezier curve with tangent handles.
    Bezier,
    /// Hermite spline.
    Hermite,
    /// Catmull-Rom spline.
    CatmullRom,
}

impl Default for CurveMode {
    fn default() -> Self {
        Self::Linear
    }
}

impl fmt::Display for CurveMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linear => write!(f, "Linear"),
            Self::Step => write!(f, "Step"),
            Self::Bezier => write!(f, "Bezier"),
            Self::Hermite => write!(f, "Hermite"),
            Self::CatmullRom => write!(f, "CatmullRom"),
        }
    }
}

/// Tangent handle for Bezier/Hermite curves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TangentHandle {
    /// Time offset from the keyframe.
    pub time: f64,
    /// Value offset from the keyframe.
    pub value: f64,
}

impl TangentHandle {
    /// Create a new tangent handle.
    pub fn new(time: f64, value: f64) -> Self {
        Self { time, value }
    }

    /// A flat (zero slope) tangent.
    pub fn flat() -> Self {
        Self { time: 0.1, value: 0.0 }
    }

    /// Returns the slope of this tangent handle.
    pub fn slope(&self) -> f64 {
        if self.time.abs() < 1e-10 {
            0.0
        } else {
            self.value / self.time
        }
    }

    /// Scale the handle length while preserving direction.
    pub fn with_length(&self, length: f64) -> Self {
        let current_len = (self.time * self.time + self.value * self.value).sqrt();
        if current_len < 1e-10 {
            return *self;
        }
        let scale = length / current_len;
        Self {
            time: self.time * scale,
            value: self.value * scale,
        }
    }
}

impl Default for TangentHandle {
    fn default() -> Self {
        Self::flat()
    }
}

/// Whether tangent handles are linked (move together) or broken (independent).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TangentLinkMode {
    /// Both handles move together (smooth curve).
    Linked,
    /// Handles move independently (can create corners).
    Broken,
    /// Handles are automatically computed.
    Auto,
}

impl Default for TangentLinkMode {
    fn default() -> Self {
        Self::Auto
    }
}

// ---------------------------------------------------------------------------
// Keyframe
// ---------------------------------------------------------------------------

/// A keyframe on an animation track.
#[derive(Debug, Clone)]
pub struct Keyframe {
    /// Unique identifier.
    pub id: KeyframeId,
    /// Time position in seconds.
    pub time: f64,
    /// The value at this keyframe.
    pub value: KeyframeValue,
    /// Interpolation mode to the next keyframe.
    pub curve_mode: CurveMode,
    /// In-tangent (approaching this keyframe).
    pub in_tangent: TangentHandle,
    /// Out-tangent (leaving this keyframe).
    pub out_tangent: TangentHandle,
    /// Tangent linking mode.
    pub tangent_link: TangentLinkMode,
    /// Whether this keyframe is selected in the editor.
    pub selected: bool,
}

impl Keyframe {
    /// Create a new keyframe.
    pub fn new(id: KeyframeId, time: f64, value: KeyframeValue) -> Self {
        Self {
            id,
            time,
            value,
            curve_mode: CurveMode::Linear,
            in_tangent: TangentHandle::flat(),
            out_tangent: TangentHandle::flat(),
            tangent_link: TangentLinkMode::Auto,
            selected: false,
        }
    }

    /// Create a keyframe with bezier tangents.
    pub fn bezier(
        id: KeyframeId,
        time: f64,
        value: KeyframeValue,
        in_tangent: TangentHandle,
        out_tangent: TangentHandle,
    ) -> Self {
        Self {
            id,
            time,
            value,
            curve_mode: CurveMode::Bezier,
            in_tangent,
            out_tangent,
            tangent_link: TangentLinkMode::Linked,
            selected: false,
        }
    }
}

impl fmt::Display for Keyframe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KF[t={:.3}s, val={}, mode={}]", self.time, self.value, self.curve_mode)
    }
}

// ---------------------------------------------------------------------------
// Track
// ---------------------------------------------------------------------------

/// What property a track targets.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TrackTarget {
    /// Entity name or ID.
    pub entity: String,
    /// Component type name.
    pub component: String,
    /// Property path (e.g. "position.x", "rotation", "color").
    pub property: String,
}

impl TrackTarget {
    pub fn new(entity: &str, component: &str, property: &str) -> Self {
        Self {
            entity: entity.to_string(),
            component: component.to_string(),
            property: property.to_string(),
        }
    }
}

impl fmt::Display for TrackTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}.{}", self.entity, self.component, self.property)
    }
}

/// An animation track containing keyframes for a single property.
#[derive(Debug, Clone)]
pub struct AnimationTrack {
    /// Unique track identifier.
    pub id: TrackId,
    /// What this track animates.
    pub target: TrackTarget,
    /// Display name in the editor.
    pub name: String,
    /// Keyframes sorted by time.
    pub keyframes: Vec<Keyframe>,
    /// Whether this track is muted (skipped during playback).
    pub muted: bool,
    /// Whether this track is locked (cannot be edited).
    pub locked: bool,
    /// Whether this track is expanded in the timeline view.
    pub expanded: bool,
    /// Track color for visualization.
    pub color: [f32; 4],
    /// Whether to show the curve in the curve editor.
    pub show_curve: bool,
}

impl AnimationTrack {
    /// Create a new track.
    pub fn new(id: TrackId, target: TrackTarget) -> Self {
        Self {
            id,
            name: target.to_string(),
            target,
            keyframes: Vec::new(),
            muted: false,
            locked: false,
            expanded: true,
            color: [0.3, 0.7, 1.0, 1.0],
            show_curve: true,
        }
    }

    /// Insert a keyframe, maintaining sorted order by time.
    pub fn insert_keyframe(&mut self, keyframe: Keyframe) -> KeyframeId {
        let id = keyframe.id;
        let time = keyframe.time;
        let insert_pos = self
            .keyframes
            .iter()
            .position(|k| k.time > time)
            .unwrap_or(self.keyframes.len());
        self.keyframes.insert(insert_pos, keyframe);
        id
    }

    /// Remove a keyframe by ID.
    pub fn remove_keyframe(&mut self, id: KeyframeId) -> Option<Keyframe> {
        if let Some(pos) = self.keyframes.iter().position(|k| k.id == id) {
            Some(self.keyframes.remove(pos))
        } else {
            None
        }
    }

    /// Get a keyframe by ID.
    pub fn get_keyframe(&self, id: KeyframeId) -> Option<&Keyframe> {
        self.keyframes.iter().find(|k| k.id == id)
    }

    /// Get a mutable keyframe by ID.
    pub fn get_keyframe_mut(&mut self, id: KeyframeId) -> Option<&mut Keyframe> {
        self.keyframes.iter_mut().find(|k| k.id == id)
    }

    /// Move a keyframe to a new time.
    pub fn move_keyframe(&mut self, id: KeyframeId, new_time: f64) {
        if let Some(kf) = self.keyframes.iter_mut().find(|k| k.id == id) {
            kf.time = new_time;
        }
        // Re-sort after move.
        self.keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    }

    /// Evaluate the track value at a given time.
    pub fn evaluate(&self, time: f64) -> Option<KeyframeValue> {
        if self.keyframes.is_empty() {
            return None;
        }
        if self.keyframes.len() == 1 {
            return Some(self.keyframes[0].value.clone());
        }

        // Before the first keyframe.
        if time <= self.keyframes[0].time {
            return Some(self.keyframes[0].value.clone());
        }

        // After the last keyframe.
        let last = self.keyframes.last().unwrap();
        if time >= last.time {
            return Some(last.value.clone());
        }

        // Find the segment.
        for i in 0..self.keyframes.len() - 1 {
            let kf0 = &self.keyframes[i];
            let kf1 = &self.keyframes[i + 1];
            if time >= kf0.time && time <= kf1.time {
                let segment_duration = kf1.time - kf0.time;
                if segment_duration <= 0.0 {
                    return Some(kf1.value.clone());
                }
                let t = (time - kf0.time) / segment_duration;

                return Some(match kf0.curve_mode {
                    CurveMode::Step => kf0.value.clone(),
                    CurveMode::Linear => kf0.value.lerp(&kf1.value, t),
                    CurveMode::Bezier => {
                        // Cubic bezier interpolation.
                        let t2 = t * t;
                        let t3 = t2 * t;
                        let ease = 3.0 * t2 - 2.0 * t3; // Smooth hermite basis.
                        kf0.value.lerp(&kf1.value, ease)
                    }
                    CurveMode::Hermite | CurveMode::CatmullRom => {
                        // Hermite interpolation.
                        let t2 = t * t;
                        let t3 = t2 * t;
                        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
                        let h01 = -2.0 * t3 + 3.0 * t2;
                        let _h10 = t3 - 2.0 * t2 + t;
                        let _h11 = t3 - t2;
                        // Simplified: blend between endpoints with hermite basis.
                        kf0.value.lerp(&kf1.value, h01 / (h00 + h01))
                    }
                });
            }
        }

        Some(last.value.clone())
    }

    /// Returns the time range of this track.
    pub fn time_range(&self) -> Option<(f64, f64)> {
        if self.keyframes.is_empty() {
            return None;
        }
        Some((
            self.keyframes.first().unwrap().time,
            self.keyframes.last().unwrap().time,
        ))
    }

    /// Returns the number of keyframes.
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// Select all keyframes in a time range.
    pub fn select_range(&mut self, start: f64, end: f64) {
        for kf in &mut self.keyframes {
            kf.selected = kf.time >= start && kf.time <= end;
        }
    }

    /// Deselect all keyframes.
    pub fn deselect_all(&mut self) {
        for kf in &mut self.keyframes {
            kf.selected = false;
        }
    }

    /// Returns the selected keyframe IDs.
    pub fn selected_keyframes(&self) -> Vec<KeyframeId> {
        self.keyframes
            .iter()
            .filter(|k| k.selected)
            .map(|k| k.id)
            .collect()
    }

    /// Generate curve points for visualization (sampled at regular intervals).
    pub fn sample_curve(&self, start: f64, end: f64, num_samples: usize) -> Vec<(f64, f64)> {
        let mut points = Vec::with_capacity(num_samples);
        if num_samples < 2 || self.keyframes.is_empty() {
            return points;
        }
        let step = (end - start) / (num_samples - 1) as f64;
        for i in 0..num_samples {
            let time = start + step * i as f64;
            let value = self.evaluate(time).and_then(|v| v.as_float()).unwrap_or(0.0);
            points.push((time, value));
        }
        points
    }
}

impl fmt::Display for AnimationTrack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Track[{}, target={}, {} keyframes]",
            self.id,
            self.target,
            self.keyframe_count()
        )
    }
}

// ---------------------------------------------------------------------------
// Animation Event
// ---------------------------------------------------------------------------

/// An event that triggers at a specific time during animation playback.
#[derive(Debug, Clone)]
pub struct AnimationEvent {
    /// Unique event identifier.
    pub id: EventId,
    /// Time at which the event fires.
    pub time: f64,
    /// Event name (used to dispatch to handlers).
    pub name: String,
    /// Optional string parameter.
    pub string_param: Option<String>,
    /// Optional float parameter.
    pub float_param: Option<f64>,
    /// Optional int parameter.
    pub int_param: Option<i64>,
    /// Whether this event is enabled.
    pub enabled: bool,
    /// Color for display in the timeline.
    pub color: [f32; 4],
}

impl AnimationEvent {
    /// Create a new animation event.
    pub fn new(id: EventId, time: f64, name: &str) -> Self {
        Self {
            id,
            time,
            name: name.to_string(),
            string_param: None,
            float_param: None,
            int_param: None,
            enabled: true,
            color: [1.0, 0.8, 0.0, 1.0],
        }
    }

    /// Set the string parameter.
    pub fn with_string(mut self, param: &str) -> Self {
        self.string_param = Some(param.to_string());
        self
    }

    /// Set the float parameter.
    pub fn with_float(mut self, param: f64) -> Self {
        self.float_param = Some(param);
        self
    }
}

impl fmt::Display for AnimationEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Event[t={:.3}s, '{}']", self.time, self.name)
    }
}

// ---------------------------------------------------------------------------
// Animation Clip
// ---------------------------------------------------------------------------

/// Loop mode for an animation clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoopMode {
    /// Play once and stop.
    Once,
    /// Loop continuously.
    Loop,
    /// Play forward then backward (ping-pong).
    PingPong,
    /// Clamp to last frame.
    ClampForever,
}

impl Default for LoopMode {
    fn default() -> Self {
        Self::Once
    }
}

/// An animation clip containing tracks and events.
#[derive(Debug, Clone)]
pub struct AnimationClip {
    /// Unique clip identifier.
    pub id: ClipId,
    /// Clip name.
    pub name: String,
    /// Duration in seconds.
    pub duration: f64,
    /// Frames per second (for timeline display).
    pub fps: f64,
    /// Loop mode.
    pub loop_mode: LoopMode,
    /// Tracks in this clip.
    pub tracks: Vec<AnimationTrack>,
    /// Events in this clip.
    pub events: Vec<AnimationEvent>,
    /// Playback speed multiplier.
    pub speed: f64,
    /// Whether this clip is additive.
    pub additive: bool,
    /// Tags for categorization.
    pub tags: Vec<String>,
}

impl AnimationClip {
    /// Create a new animation clip.
    pub fn new(id: ClipId, name: &str, duration: f64) -> Self {
        Self {
            id,
            name: name.to_string(),
            duration,
            fps: 30.0,
            loop_mode: LoopMode::Once,
            tracks: Vec::new(),
            events: Vec::new(),
            speed: 1.0,
            additive: false,
            tags: Vec::new(),
        }
    }

    /// Add a track to this clip.
    pub fn add_track(&mut self, track: AnimationTrack) -> TrackId {
        let id = track.id;
        self.tracks.push(track);
        id
    }

    /// Remove a track by ID.
    pub fn remove_track(&mut self, id: TrackId) -> Option<AnimationTrack> {
        if let Some(pos) = self.tracks.iter().position(|t| t.id == id) {
            Some(self.tracks.remove(pos))
        } else {
            None
        }
    }

    /// Get a track by ID.
    pub fn get_track(&self, id: TrackId) -> Option<&AnimationTrack> {
        self.tracks.iter().find(|t| t.id == id)
    }

    /// Get a mutable track by ID.
    pub fn get_track_mut(&mut self, id: TrackId) -> Option<&mut AnimationTrack> {
        self.tracks.iter_mut().find(|t| t.id == id)
    }

    /// Add an event to this clip.
    pub fn add_event(&mut self, event: AnimationEvent) -> EventId {
        let id = event.id;
        self.events.push(event);
        self.events.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
        id
    }

    /// Remove an event by ID.
    pub fn remove_event(&mut self, id: EventId) -> Option<AnimationEvent> {
        if let Some(pos) = self.events.iter().position(|e| e.id == id) {
            Some(self.events.remove(pos))
        } else {
            None
        }
    }

    /// Evaluate all tracks at a given time.
    pub fn evaluate(&self, time: f64) -> HashMap<String, KeyframeValue> {
        let effective_time = self.wrap_time(time);
        let mut results = HashMap::new();
        for track in &self.tracks {
            if track.muted {
                continue;
            }
            if let Some(value) = track.evaluate(effective_time) {
                results.insert(track.target.to_string(), value);
            }
        }
        results
    }

    /// Get events that fire within a time range.
    pub fn events_in_range(&self, start: f64, end: f64) -> Vec<&AnimationEvent> {
        self.events
            .iter()
            .filter(|e| e.enabled && e.time >= start && e.time < end)
            .collect()
    }

    /// Wrap a time value according to the loop mode.
    pub fn wrap_time(&self, time: f64) -> f64 {
        if self.duration <= 0.0 {
            return 0.0;
        }
        match self.loop_mode {
            LoopMode::Once => time.clamp(0.0, self.duration),
            LoopMode::Loop => {
                let t = time % self.duration;
                if t < 0.0 { t + self.duration } else { t }
            }
            LoopMode::PingPong => {
                let cycle = self.duration * 2.0;
                let t = time % cycle;
                let t = if t < 0.0 { t + cycle } else { t };
                if t <= self.duration {
                    t
                } else {
                    cycle - t
                }
            }
            LoopMode::ClampForever => time.max(0.0),
        }
    }

    /// Convert a time in seconds to a frame number.
    pub fn time_to_frame(&self, time: f64) -> i64 {
        (time * self.fps).round() as i64
    }

    /// Convert a frame number to time in seconds.
    pub fn frame_to_time(&self, frame: i64) -> f64 {
        frame as f64 / self.fps
    }

    /// Returns the total frame count.
    pub fn frame_count(&self) -> i64 {
        (self.duration * self.fps).ceil() as i64
    }

    /// Returns the number of tracks.
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }
}

impl fmt::Display for AnimationClip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Clip[{}, '{}', {:.2}s, {} tracks, {} events]",
            self.id,
            self.name,
            self.duration,
            self.track_count(),
            self.events.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Playback state
// ---------------------------------------------------------------------------

/// The playback state of the animation preview.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaybackState {
    /// Animation is stopped at the beginning.
    Stopped,
    /// Animation is playing forward.
    Playing,
    /// Animation is paused at the current time.
    Paused,
    /// Playback is being scrubbed (dragging the playhead).
    Scrubbing,
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self::Stopped
    }
}

/// Animation preview controller.
#[derive(Debug, Clone)]
pub struct AnimationPreview {
    /// Current playback state.
    pub state: PlaybackState,
    /// Current playback time in seconds.
    pub current_time: f64,
    /// Playback speed multiplier.
    pub speed: f64,
    /// Whether to loop playback.
    pub looping: bool,
    /// Selected clip for preview.
    pub clip_id: Option<ClipId>,
    /// Frame stepping mode (advance one frame at a time).
    pub frame_step: bool,
}

impl AnimationPreview {
    pub fn new() -> Self {
        Self {
            state: PlaybackState::Stopped,
            current_time: 0.0,
            speed: 1.0,
            looping: false,
            clip_id: None,
            frame_step: false,
        }
    }

    /// Play the animation.
    pub fn play(&mut self) {
        self.state = PlaybackState::Playing;
    }

    /// Pause the animation.
    pub fn pause(&mut self) {
        self.state = PlaybackState::Paused;
    }

    /// Stop the animation (reset to beginning).
    pub fn stop(&mut self) {
        self.state = PlaybackState::Stopped;
        self.current_time = 0.0;
    }

    /// Set the scrub position.
    pub fn scrub(&mut self, time: f64) {
        self.state = PlaybackState::Scrubbing;
        self.current_time = time.max(0.0);
    }

    /// Advance playback by delta time.
    pub fn advance(&mut self, dt: f64, clip_duration: f64) {
        if self.state != PlaybackState::Playing {
            return;
        }
        self.current_time += dt * self.speed;
        if self.current_time >= clip_duration {
            if self.looping {
                self.current_time %= clip_duration;
            } else {
                self.current_time = clip_duration;
                self.state = PlaybackState::Stopped;
            }
        }
    }

    /// Step forward one frame.
    pub fn step_forward(&mut self, fps: f64) {
        self.current_time += 1.0 / fps;
        self.state = PlaybackState::Paused;
    }

    /// Step backward one frame.
    pub fn step_backward(&mut self, fps: f64) {
        self.current_time = (self.current_time - 1.0 / fps).max(0.0);
        self.state = PlaybackState::Paused;
    }

    /// Toggle between play and pause.
    pub fn toggle_play_pause(&mut self) {
        match self.state {
            PlaybackState::Playing => self.pause(),
            _ => self.play(),
        }
    }
}

impl Default for AnimationPreview {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Blend Tree
// ---------------------------------------------------------------------------

/// A node in a blend tree.
#[derive(Debug, Clone)]
pub enum BlendNode {
    /// A leaf node that plays a single clip.
    Clip {
        id: BlendNodeId,
        clip_id: ClipId,
        speed: f64,
    },
    /// A 1D blend between two or more clips based on a parameter.
    Blend1D {
        id: BlendNodeId,
        parameter: String,
        children: Vec<(f64, BlendNodeId)>, // (threshold, child)
    },
    /// A 2D blend between clips based on two parameters.
    Blend2D {
        id: BlendNodeId,
        param_x: String,
        param_y: String,
        children: Vec<(f64, f64, BlendNodeId)>, // (x, y, child)
    },
    /// An additive blend: base + additive * weight.
    Additive {
        id: BlendNodeId,
        base: BlendNodeId,
        additive: BlendNodeId,
        weight: f64,
    },
    /// A crossfade/transition between two nodes.
    Crossfade {
        id: BlendNodeId,
        from: BlendNodeId,
        to: BlendNodeId,
        duration: f64,
        elapsed: f64,
    },
    /// Override: replace specific bones from another node.
    Override {
        id: BlendNodeId,
        base: BlendNodeId,
        override_node: BlendNodeId,
        mask: Vec<String>, // Bone/property names to override.
    },
}

impl BlendNode {
    /// Returns the ID of this blend node.
    pub fn id(&self) -> BlendNodeId {
        match self {
            Self::Clip { id, .. }
            | Self::Blend1D { id, .. }
            | Self::Blend2D { id, .. }
            | Self::Additive { id, .. }
            | Self::Crossfade { id, .. }
            | Self::Override { id, .. } => *id,
        }
    }

    /// Returns a display name for this node type.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Clip { .. } => "Clip",
            Self::Blend1D { .. } => "Blend1D",
            Self::Blend2D { .. } => "Blend2D",
            Self::Additive { .. } => "Additive",
            Self::Crossfade { .. } => "Crossfade",
            Self::Override { .. } => "Override",
        }
    }

    /// Returns the child node IDs.
    pub fn children(&self) -> Vec<BlendNodeId> {
        match self {
            Self::Clip { .. } => Vec::new(),
            Self::Blend1D { children, .. } => children.iter().map(|(_, id)| *id).collect(),
            Self::Blend2D { children, .. } => children.iter().map(|(_, _, id)| *id).collect(),
            Self::Additive { base, additive, .. } => vec![*base, *additive],
            Self::Crossfade { from, to, .. } => vec![*from, *to],
            Self::Override { base, override_node, .. } => vec![*base, *override_node],
        }
    }
}

impl fmt::Display for BlendNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clip { id, clip_id, .. } => write!(f, "Clip({id}, clip={clip_id})"),
            Self::Blend1D { id, parameter, children, .. } => {
                write!(f, "Blend1D({id}, param={parameter}, {} children)", children.len())
            }
            Self::Blend2D { id, param_x, param_y, children, .. } => {
                write!(f, "Blend2D({id}, params={param_x}/{param_y}, {} children)", children.len())
            }
            Self::Additive { id, weight, .. } => write!(f, "Additive({id}, w={weight:.2})"),
            Self::Crossfade { id, duration, elapsed, .. } => {
                write!(f, "Crossfade({id}, {elapsed:.2}/{duration:.2}s)")
            }
            Self::Override { id, mask, .. } => {
                write!(f, "Override({id}, {} masks)", mask.len())
            }
        }
    }
}

/// A blend tree containing interconnected blend nodes.
#[derive(Debug, Clone)]
pub struct BlendTree {
    /// Unique name.
    pub name: String,
    /// All nodes in the tree.
    pub nodes: HashMap<BlendNodeId, BlendNode>,
    /// The root node ID.
    pub root: Option<BlendNodeId>,
    /// Parameter values that drive blending.
    pub parameters: HashMap<String, f64>,
    /// Next node ID.
    next_id: u64,
}

impl BlendTree {
    /// Create a new empty blend tree.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            nodes: HashMap::new(),
            root: None,
            parameters: HashMap::new(),
            next_id: 1,
        }
    }

    /// Add a node to the tree.
    pub fn add_node(&mut self, node: BlendNode) -> BlendNodeId {
        let id = node.id();
        if self.root.is_none() {
            self.root = Some(id);
        }
        self.nodes.insert(id, node);
        id
    }

    /// Generate a new node ID.
    pub fn next_id(&mut self) -> BlendNodeId {
        let id = BlendNodeId::new(self.next_id);
        self.next_id += 1;
        id
    }

    /// Set a parameter value.
    pub fn set_parameter(&mut self, name: &str, value: f64) {
        self.parameters.insert(name.to_string(), value);
    }

    /// Get a parameter value.
    pub fn get_parameter(&self, name: &str) -> f64 {
        self.parameters.get(name).copied().unwrap_or(0.0)
    }

    /// Returns the node count.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

impl fmt::Display for BlendTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BlendTree[{}, {} nodes, {} params]",
            self.name,
            self.node_count(),
            self.parameters.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Timeline view state
// ---------------------------------------------------------------------------

/// State for the timeline view in the editor.
#[derive(Debug, Clone)]
pub struct TimelineViewState {
    /// Visible time range start (seconds).
    pub view_start: f64,
    /// Visible time range end (seconds).
    pub view_end: f64,
    /// Vertical scroll offset.
    pub scroll_y: f64,
    /// Zoom level (pixels per second).
    pub zoom: f64,
    /// Track height in pixels.
    pub track_height: f64,
    /// Whether to snap to frames.
    pub snap_to_frame: bool,
    /// Whether to show the curve editor.
    pub show_curves: bool,
    /// Height of the curve editor panel.
    pub curve_editor_height: f64,
    /// Currently selected track.
    pub selected_track: Option<TrackId>,
}

impl Default for TimelineViewState {
    fn default() -> Self {
        Self {
            view_start: 0.0,
            view_end: 5.0,
            scroll_y: 0.0,
            zoom: 100.0,
            track_height: 24.0,
            snap_to_frame: true,
            show_curves: false,
            curve_editor_height: 200.0,
            selected_track: None,
        }
    }
}

impl TimelineViewState {
    /// Convert a time position to a pixel X coordinate.
    pub fn time_to_pixel(&self, time: f64) -> f64 {
        (time - self.view_start) * self.zoom
    }

    /// Convert a pixel X coordinate to a time position.
    pub fn pixel_to_time(&self, pixel: f64) -> f64 {
        pixel / self.zoom + self.view_start
    }

    /// Zoom in by a factor, centered on a specific time.
    pub fn zoom_in(&mut self, center_time: f64, factor: f64) {
        let new_zoom = (self.zoom * factor).min(10000.0);
        let ratio = self.zoom / new_zoom;
        self.view_start = center_time - (center_time - self.view_start) * ratio;
        self.view_end = center_time + (self.view_end - center_time) * ratio;
        self.zoom = new_zoom;
    }

    /// Zoom out by a factor.
    pub fn zoom_out(&mut self, center_time: f64, factor: f64) {
        self.zoom_in(center_time, 1.0 / factor);
    }

    /// Pan the view by a time delta.
    pub fn pan(&mut self, dt: f64) {
        self.view_start += dt;
        self.view_end += dt;
    }

    /// Snap a time value to the nearest frame.
    pub fn snap_time(&self, time: f64, fps: f64) -> f64 {
        if self.snap_to_frame {
            (time * fps).round() / fps
        } else {
            time
        }
    }
}

// ---------------------------------------------------------------------------
// Animation Editor
// ---------------------------------------------------------------------------

/// The main animation editor state.
pub struct AnimationEditor {
    /// All animation clips.
    clips: HashMap<ClipId, AnimationClip>,
    /// Blend trees.
    blend_trees: HashMap<String, BlendTree>,
    /// The animation preview controller.
    pub preview: AnimationPreview,
    /// Timeline view state.
    pub timeline: TimelineViewState,
    /// Next clip ID.
    next_clip_id: u64,
    /// Next track ID.
    next_track_id: u64,
    /// Next keyframe ID.
    next_keyframe_id: u64,
    /// Next event ID.
    next_event_id: u64,
    /// Clipboard for copy/paste of keyframes.
    clipboard: Vec<Keyframe>,
}

impl AnimationEditor {
    /// Create a new animation editor.
    pub fn new() -> Self {
        Self {
            clips: HashMap::new(),
            blend_trees: HashMap::new(),
            preview: AnimationPreview::new(),
            timeline: TimelineViewState::default(),
            next_clip_id: 1,
            next_track_id: 1,
            next_keyframe_id: 1,
            next_event_id: 1,
            clipboard: Vec::new(),
        }
    }

    /// Create a new animation clip.
    pub fn create_clip(&mut self, name: &str, duration: f64) -> ClipId {
        let id = ClipId::new(self.next_clip_id);
        self.next_clip_id += 1;
        let clip = AnimationClip::new(id, name, duration);
        self.clips.insert(id, clip);
        id
    }

    /// Add a track to a clip.
    pub fn add_track(&mut self, clip_id: ClipId, target: TrackTarget) -> Option<TrackId> {
        let track_id = TrackId::new(self.next_track_id);
        self.next_track_id += 1;
        let track = AnimationTrack::new(track_id, target);
        self.clips.get_mut(&clip_id)?.add_track(track);
        Some(track_id)
    }

    /// Insert a keyframe on a track.
    pub fn insert_keyframe(
        &mut self,
        clip_id: ClipId,
        track_id: TrackId,
        time: f64,
        value: KeyframeValue,
    ) -> Option<KeyframeId> {
        let kf_id = KeyframeId::new(self.next_keyframe_id);
        self.next_keyframe_id += 1;
        let keyframe = Keyframe::new(kf_id, time, value);
        let clip = self.clips.get_mut(&clip_id)?;
        let track = clip.get_track_mut(track_id)?;
        track.insert_keyframe(keyframe);
        Some(kf_id)
    }

    /// Remove a keyframe.
    pub fn remove_keyframe(
        &mut self,
        clip_id: ClipId,
        track_id: TrackId,
        kf_id: KeyframeId,
    ) -> Option<Keyframe> {
        let clip = self.clips.get_mut(&clip_id)?;
        let track = clip.get_track_mut(track_id)?;
        track.remove_keyframe(kf_id)
    }

    /// Add an animation event to a clip.
    pub fn add_event(&mut self, clip_id: ClipId, time: f64, name: &str) -> Option<EventId> {
        let event_id = EventId(self.next_event_id);
        self.next_event_id += 1;
        let event = AnimationEvent::new(event_id, time, name);
        self.clips.get_mut(&clip_id)?.add_event(event);
        Some(event_id)
    }

    /// Get a clip by ID.
    pub fn get_clip(&self, id: ClipId) -> Option<&AnimationClip> {
        self.clips.get(&id)
    }

    /// Get a mutable clip by ID.
    pub fn get_clip_mut(&mut self, id: ClipId) -> Option<&mut AnimationClip> {
        self.clips.get_mut(&id)
    }

    /// Remove a clip.
    pub fn remove_clip(&mut self, id: ClipId) -> Option<AnimationClip> {
        self.clips.remove(&id)
    }

    /// List all clips.
    pub fn list_clips(&self) -> Vec<(ClipId, &str)> {
        self.clips.values().map(|c| (c.id, c.name.as_str())).collect()
    }

    /// Returns the total number of clips.
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }

    /// Add a blend tree.
    pub fn add_blend_tree(&mut self, tree: BlendTree) {
        self.blend_trees.insert(tree.name.clone(), tree);
    }

    /// Get a blend tree by name.
    pub fn get_blend_tree(&self, name: &str) -> Option<&BlendTree> {
        self.blend_trees.get(name)
    }

    /// Update the editor (call once per frame).
    pub fn update(&mut self, dt: f64) {
        if let Some(clip_id) = self.preview.clip_id {
            if let Some(clip) = self.clips.get(&clip_id) {
                let duration = clip.duration;
                self.preview.advance(dt, duration);
            }
        }
    }

    /// Copy selected keyframes to clipboard.
    pub fn copy_selected(&mut self, clip_id: ClipId, track_id: TrackId) {
        self.clipboard.clear();
        if let Some(clip) = self.clips.get(&clip_id) {
            if let Some(track) = clip.get_track(track_id) {
                for kf in &track.keyframes {
                    if kf.selected {
                        self.clipboard.push(kf.clone());
                    }
                }
            }
        }
    }

    /// Paste keyframes from clipboard at a given time offset.
    pub fn paste_at(&mut self, clip_id: ClipId, track_id: TrackId, time_offset: f64) {
        let keyframes: Vec<Keyframe> = self.clipboard.iter().map(|kf| {
            let mut new_kf = kf.clone();
            new_kf.id = KeyframeId::new(self.next_keyframe_id);
            self.next_keyframe_id += 1;
            new_kf.time += time_offset;
            new_kf
        }).collect();

        if let Some(clip) = self.clips.get_mut(&clip_id) {
            if let Some(track) = clip.get_track_mut(track_id) {
                for kf in keyframes {
                    track.insert_keyframe(kf);
                }
            }
        }
    }
}

impl Default for AnimationEditor {
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
    fn test_create_clip() {
        let mut editor = AnimationEditor::new();
        let id = editor.create_clip("walk", 2.0);
        let clip = editor.get_clip(id).unwrap();
        assert_eq!(clip.name, "walk");
        assert_eq!(clip.duration, 2.0);
    }

    #[test]
    fn test_add_track_and_keyframes() {
        let mut editor = AnimationEditor::new();
        let clip_id = editor.create_clip("test", 1.0);
        let track_id = editor.add_track(clip_id, TrackTarget::new("player", "Transform", "y")).unwrap();

        editor.insert_keyframe(clip_id, track_id, 0.0, KeyframeValue::Float(0.0));
        editor.insert_keyframe(clip_id, track_id, 1.0, KeyframeValue::Float(10.0));

        let clip = editor.get_clip(clip_id).unwrap();
        assert_eq!(clip.track_count(), 1);
        assert_eq!(clip.get_track(track_id).unwrap().keyframe_count(), 2);
    }

    #[test]
    fn test_track_evaluation() {
        let mut track = AnimationTrack::new(TrackId::new(1), TrackTarget::new("e", "c", "p"));
        track.insert_keyframe(Keyframe::new(KeyframeId::new(1), 0.0, KeyframeValue::Float(0.0)));
        track.insert_keyframe(Keyframe::new(KeyframeId::new(2), 1.0, KeyframeValue::Float(10.0)));

        assert_eq!(track.evaluate(0.0).unwrap().as_float(), Some(0.0));
        assert_eq!(track.evaluate(1.0).unwrap().as_float(), Some(10.0));

        let mid = track.evaluate(0.5).unwrap().as_float().unwrap();
        assert!((mid - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_step_interpolation() {
        let mut track = AnimationTrack::new(TrackId::new(1), TrackTarget::new("e", "c", "p"));
        let mut kf = Keyframe::new(KeyframeId::new(1), 0.0, KeyframeValue::Float(0.0));
        kf.curve_mode = CurveMode::Step;
        track.insert_keyframe(kf);
        track.insert_keyframe(Keyframe::new(KeyframeId::new(2), 1.0, KeyframeValue::Float(10.0)));

        let mid = track.evaluate(0.5).unwrap().as_float().unwrap();
        assert_eq!(mid, 0.0); // Step holds until next keyframe.
    }

    #[test]
    fn test_keyframe_lerp() {
        let a = KeyframeValue::Vec3(0.0, 0.0, 0.0);
        let b = KeyframeValue::Vec3(10.0, 20.0, 30.0);
        let mid = a.lerp(&b, 0.5);
        match mid {
            KeyframeValue::Vec3(x, y, z) => {
                assert!((x - 5.0).abs() < 0.001);
                assert!((y - 10.0).abs() < 0.001);
                assert!((z - 15.0).abs() < 0.001);
            }
            _ => panic!("expected Vec3"),
        }
    }

    #[test]
    fn test_animation_events() {
        let mut editor = AnimationEditor::new();
        let clip_id = editor.create_clip("test", 2.0);
        editor.add_event(clip_id, 0.5, "footstep");
        editor.add_event(clip_id, 1.5, "attack");

        let clip = editor.get_clip(clip_id).unwrap();
        let events = clip.events_in_range(0.0, 1.0);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "footstep");
    }

    #[test]
    fn test_loop_modes() {
        let clip = AnimationClip::new(ClipId::new(1), "test", 2.0);

        // Once mode.
        assert_eq!(clip.wrap_time(3.0), 2.0);

        let mut looping = clip.clone();
        looping.loop_mode = LoopMode::Loop;
        assert!((looping.wrap_time(2.5) - 0.5).abs() < 0.001);

        let mut pingpong = clip.clone();
        pingpong.loop_mode = LoopMode::PingPong;
        assert!((pingpong.wrap_time(3.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_preview_playback() {
        let mut preview = AnimationPreview::new();
        preview.play();
        assert_eq!(preview.state, PlaybackState::Playing);

        preview.advance(0.5, 2.0);
        assert!((preview.current_time - 0.5).abs() < 0.001);

        preview.pause();
        assert_eq!(preview.state, PlaybackState::Paused);

        preview.stop();
        assert_eq!(preview.current_time, 0.0);
    }

    #[test]
    fn test_timeline_view() {
        let view = TimelineViewState::default();
        let pixel = view.time_to_pixel(1.0);
        let time = view.pixel_to_time(pixel);
        assert!((time - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_blend_tree() {
        let mut tree = BlendTree::new("locomotion");
        let id1 = tree.next_id();
        let id2 = tree.next_id();

        tree.add_node(BlendNode::Clip {
            id: id1,
            clip_id: ClipId::new(1),
            speed: 1.0,
        });
        tree.add_node(BlendNode::Clip {
            id: id2,
            clip_id: ClipId::new(2),
            speed: 1.0,
        });

        tree.set_parameter("speed", 1.5);
        assert_eq!(tree.get_parameter("speed"), 1.5);
        assert_eq!(tree.node_count(), 2);
    }

    #[test]
    fn test_tangent_handle() {
        let handle = TangentHandle::new(0.1, 0.5);
        assert!((handle.slope() - 5.0).abs() < 0.001);

        let flat = TangentHandle::flat();
        assert_eq!(flat.slope(), 0.0);
    }

    #[test]
    fn test_frame_conversion() {
        let clip = AnimationClip::new(ClipId::new(1), "test", 1.0);
        assert_eq!(clip.time_to_frame(0.5), 15);
        assert!((clip.frame_to_time(15) - 0.5).abs() < 0.001);
        assert_eq!(clip.frame_count(), 30);
    }

    #[test]
    fn test_track_select_range() {
        let mut track = AnimationTrack::new(TrackId::new(1), TrackTarget::new("e", "c", "p"));
        for i in 0..5 {
            track.insert_keyframe(Keyframe::new(
                KeyframeId::new(i + 1),
                i as f64 * 0.25,
                KeyframeValue::Float(0.0),
            ));
        }

        track.select_range(0.2, 0.8);
        let selected = track.selected_keyframes();
        assert_eq!(selected.len(), 3); // 0.25, 0.5, 0.75
    }
}
