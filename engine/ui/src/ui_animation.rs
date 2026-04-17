//! Widget animation system for the Genovo UI framework.
//!
//! Provides smooth transitions and reactive motion for UI elements, inspired
//! by Unreal Engine Slate's `FCurveSequence` and Epic's motion design
//! principles.
//!
//! # Components
//!
//! - [`CurveSequence`]: a timeline of animation curves that can be played
//!   forward, backward, paused, and resumed. Each segment defines a start
//!   time, duration, and easing function.
//!
//! - [`SpringAnimation`]: a critically damped spring for smooth reactive
//!   motion. Used for hover effects, scroll, and resize animations.
//!
//! - [`AnimatedValue<T>`]: a wrapper that smoothly interpolates any `Lerp`
//!   value from its current state to a target over a configurable duration.
//!
//! - [`TransformAnimation`]: animates position, rotation, and scale as a
//!   combined render transform.
//!
//! # Usage
//!
//! ```rust,ignore
//! let mut seq = CurveSequence::new();
//! seq.add_segment(0.0, 0.3, EasingKind::CubicOut);
//! seq.play();
//!
//! // Each frame:
//! seq.tick(dt);
//! let alpha = seq.get_lerp(); // 0.0 .. 1.0
//! ```

use std::collections::HashMap;

use glam::Vec2;
use serde::{Deserialize, Serialize};

// We re-use the existing EasingFunction from the animation module rather
// than duplicating it, but wrap it in a local alias for the richer set.
use crate::animation::EasingFunction;

// ---------------------------------------------------------------------------
// EasingKind — extended easing catalogue
// ---------------------------------------------------------------------------

/// Extended easing catalogue covering all common motion curves.
///
/// This wraps the existing [`EasingFunction`] enum and adds convenience
/// constructors for the most commonly used animation curves.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EasingKind {
    Linear,
    // Quad
    QuadIn,
    QuadOut,
    QuadInOut,
    // Cubic
    CubicIn,
    CubicOut,
    CubicInOut,
    // Quart
    QuartIn,
    QuartOut,
    QuartInOut,
    // Quint
    QuintIn,
    QuintOut,
    QuintInOut,
    // Sine
    SinIn,
    SinOut,
    SinInOut,
    // Expo
    ExpoIn,
    ExpoOut,
    ExpoInOut,
    // Circ
    CircIn,
    CircOut,
    CircInOut,
    // Special
    ElasticOut,
    BounceOut,
    BackOut,
    BackIn,
}

impl EasingKind {
    /// Evaluate the easing curve at `t` (0.0 to 1.0).
    pub fn evaluate(&self, t: f32) -> f32 {
        self.to_easing_function().evaluate(t)
    }

    /// Convert to the core [`EasingFunction`] enum.
    pub fn to_easing_function(&self) -> EasingFunction {
        match self {
            Self::Linear => EasingFunction::Linear,
            Self::QuadIn => EasingFunction::EaseInQuad,
            Self::QuadOut => EasingFunction::EaseOutQuad,
            Self::QuadInOut => EasingFunction::EaseInOutQuad,
            Self::CubicIn => EasingFunction::EaseInCubic,
            Self::CubicOut => EasingFunction::EaseOutCubic,
            Self::CubicInOut => EasingFunction::EaseInOutCubic,
            Self::QuartIn => EasingFunction::EaseInQuart,
            Self::QuartOut => EasingFunction::EaseOutQuart,
            Self::QuartInOut => EasingFunction::EaseInOutQuart,
            Self::QuintIn => EasingFunction::EaseInQuint,
            Self::QuintOut => EasingFunction::EaseOutQuint,
            Self::QuintInOut => EasingFunction::EaseInOutQuint,
            Self::SinIn => EasingFunction::EaseInSine,
            Self::SinOut => EasingFunction::EaseOutSine,
            Self::SinInOut => EasingFunction::EaseInOutSine,
            Self::ExpoIn => EasingFunction::EaseInExpo,
            Self::ExpoOut => EasingFunction::EaseOutExpo,
            Self::ExpoInOut => EasingFunction::EaseInOutExpo,
            Self::CircIn => EasingFunction::EaseInCirc,
            Self::CircOut => EasingFunction::EaseOutCirc,
            Self::CircInOut => EasingFunction::EaseInOutCirc,
            Self::ElasticOut => EasingFunction::Elastic,
            Self::BounceOut => EasingFunction::Bounce,
            Self::BackOut => EasingFunction::Back,
            Self::BackIn => EasingFunction::BackIn,
        }
    }

    /// Returns the "inverse" easing: if this is an In, returns the Out, etc.
    pub fn inverse(&self) -> Self {
        match self {
            Self::Linear => Self::Linear,
            Self::QuadIn => Self::QuadOut,
            Self::QuadOut => Self::QuadIn,
            Self::QuadInOut => Self::QuadInOut,
            Self::CubicIn => Self::CubicOut,
            Self::CubicOut => Self::CubicIn,
            Self::CubicInOut => Self::CubicInOut,
            Self::QuartIn => Self::QuartOut,
            Self::QuartOut => Self::QuartIn,
            Self::QuartInOut => Self::QuartInOut,
            Self::QuintIn => Self::QuintOut,
            Self::QuintOut => Self::QuintIn,
            Self::QuintInOut => Self::QuintInOut,
            Self::SinIn => Self::SinOut,
            Self::SinOut => Self::SinIn,
            Self::SinInOut => Self::SinInOut,
            Self::ExpoIn => Self::ExpoOut,
            Self::ExpoOut => Self::ExpoIn,
            Self::ExpoInOut => Self::ExpoInOut,
            Self::CircIn => Self::CircOut,
            Self::CircOut => Self::CircIn,
            Self::CircInOut => Self::CircInOut,
            Self::ElasticOut => Self::ElasticOut,
            Self::BounceOut => Self::BounceOut,
            Self::BackOut => Self::BackIn,
            Self::BackIn => Self::BackOut,
        }
    }
}

impl Default for EasingKind {
    fn default() -> Self {
        Self::Linear
    }
}

// ---------------------------------------------------------------------------
// CurveHandle — a single segment of a CurveSequence
// ---------------------------------------------------------------------------

/// A single curve segment within a [`CurveSequence`].
///
/// Each handle defines a time interval and easing function. During that
/// interval, the interpolation value ramps from 0.0 to 1.0 (or 1.0 to 0.0
/// if the sequence is playing in reverse).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurveHandle {
    /// Start time of this segment within the sequence (in seconds).
    pub start_time: f32,
    /// Duration of this segment (in seconds).
    pub duration: f32,
    /// Easing function for this segment.
    pub easing: EasingKind,
}

impl CurveHandle {
    /// Creates a new curve handle.
    pub fn new(start_time: f32, duration: f32, easing: EasingKind) -> Self {
        Self {
            start_time,
            duration: duration.max(0.001),
            easing,
        }
    }

    /// End time of this segment.
    pub fn end_time(&self) -> f32 {
        self.start_time + self.duration
    }

    /// Evaluate the easing function at the given sequence time.
    /// Returns a value in [0, 1] or `None` if outside this segment's range.
    pub fn evaluate(&self, sequence_time: f32) -> Option<f32> {
        if sequence_time < self.start_time {
            return Some(0.0);
        }
        if sequence_time >= self.end_time() {
            return Some(1.0);
        }
        let local_t = (sequence_time - self.start_time) / self.duration;
        Some(self.easing.evaluate(local_t.clamp(0.0, 1.0)))
    }

    /// Evaluate in reverse.
    pub fn evaluate_reverse(&self, sequence_time: f32) -> Option<f32> {
        self.evaluate(sequence_time).map(|v| 1.0 - v)
    }
}

// ---------------------------------------------------------------------------
// PlaybackState
// ---------------------------------------------------------------------------

/// Playback state of an animation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlaybackState {
    /// Not started or stopped.
    Stopped,
    /// Currently playing forward.
    PlayingForward,
    /// Currently playing in reverse.
    PlayingReverse,
    /// Paused (retains current position).
    Paused,
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self::Stopped
    }
}

// ---------------------------------------------------------------------------
// CurveSequence — animation timeline
// ---------------------------------------------------------------------------

/// A timeline of animation curve segments.
///
/// The sequence maintains a current time position and can be played forward,
/// reversed, paused, and resumed. Each segment defines a contiguous time
/// interval with an easing function.
///
/// The primary output is `get_lerp()`, which returns a value in [0, 1]
/// representing the current interpolation position of the active segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurveSequence {
    /// The curve segments that make up this sequence.
    segments: Vec<CurveHandle>,
    /// Current playback time (in seconds).
    current_time: f32,
    /// Total duration (computed from segments).
    total_duration: f32,
    /// Current playback state.
    state: PlaybackState,
    /// Whether the sequence should loop.
    pub looping: bool,
    /// Playback speed multiplier (1.0 = normal speed).
    pub speed: f32,
}

impl CurveSequence {
    /// Creates a new empty sequence.
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            current_time: 0.0,
            total_duration: 0.0,
            state: PlaybackState::Stopped,
            looping: false,
            speed: 1.0,
        }
    }

    /// Creates a sequence with a single segment.
    pub fn simple(duration: f32, easing: EasingKind) -> Self {
        let mut seq = Self::new();
        seq.add_segment(0.0, duration, easing);
        seq
    }

    /// Creates a sequence with two segments (e.g., fade in then hold).
    pub fn two_phase(
        phase1_duration: f32,
        phase1_easing: EasingKind,
        phase2_duration: f32,
        phase2_easing: EasingKind,
    ) -> Self {
        let mut seq = Self::new();
        seq.add_segment(0.0, phase1_duration, phase1_easing);
        seq.add_segment(phase1_duration, phase2_duration, phase2_easing);
        seq
    }

    /// Add a curve segment to the sequence.
    pub fn add_segment(&mut self, start_time: f32, duration: f32, easing: EasingKind) -> &mut Self {
        let handle = CurveHandle::new(start_time, duration, easing);
        let end = handle.end_time();
        self.segments.push(handle);
        if end > self.total_duration {
            self.total_duration = end;
        }
        self
    }

    /// Returns the total duration of the sequence.
    pub fn duration(&self) -> f32 {
        self.total_duration
    }

    /// Returns the number of segments.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Start playing forward from the beginning.
    pub fn play(&mut self) {
        self.current_time = 0.0;
        self.state = PlaybackState::PlayingForward;
    }

    /// Start playing forward from the current position.
    pub fn play_from_current(&mut self) {
        self.state = PlaybackState::PlayingForward;
    }

    /// Start playing in reverse from the end.
    pub fn play_reverse(&mut self) {
        self.current_time = self.total_duration;
        self.state = PlaybackState::PlayingReverse;
    }

    /// Start playing in reverse from the current position.
    pub fn play_reverse_from_current(&mut self) {
        self.state = PlaybackState::PlayingReverse;
    }

    /// Pause playback, retaining the current position.
    pub fn pause(&mut self) {
        if self.state == PlaybackState::PlayingForward
            || self.state == PlaybackState::PlayingReverse
        {
            self.state = PlaybackState::Paused;
        }
    }

    /// Resume playback from a paused state.
    pub fn resume(&mut self) {
        if self.state == PlaybackState::Paused {
            // Resume in the direction we were going before pause
            self.state = PlaybackState::PlayingForward;
        }
    }

    /// Stop playback and reset to the beginning.
    pub fn stop(&mut self) {
        self.current_time = 0.0;
        self.state = PlaybackState::Stopped;
    }

    /// Jump to a specific time position.
    pub fn seek(&mut self, time: f32) {
        self.current_time = time.clamp(0.0, self.total_duration);
    }

    /// Returns `true` if the sequence is currently playing.
    pub fn is_playing(&self) -> bool {
        matches!(
            self.state,
            PlaybackState::PlayingForward | PlaybackState::PlayingReverse
        )
    }

    /// Returns `true` if playback is paused.
    pub fn is_paused(&self) -> bool {
        self.state == PlaybackState::Paused
    }

    /// Returns `true` if the sequence is at the start (time 0).
    pub fn is_at_start(&self) -> bool {
        self.current_time <= 0.0
    }

    /// Returns `true` if the sequence is at the end.
    pub fn is_at_end(&self) -> bool {
        self.current_time >= self.total_duration
    }

    /// Returns the current playback state.
    pub fn state(&self) -> PlaybackState {
        self.state
    }

    /// Returns the current time position.
    pub fn current_time(&self) -> f32 {
        self.current_time
    }

    /// Returns the normalised position (0.0 to 1.0) within the total
    /// duration.
    pub fn progress(&self) -> f32 {
        if self.total_duration <= 0.0 {
            return 0.0;
        }
        (self.current_time / self.total_duration).clamp(0.0, 1.0)
    }

    /// Advance the sequence by `dt` seconds.
    pub fn tick(&mut self, dt: f32) {
        match self.state {
            PlaybackState::PlayingForward => {
                self.current_time += dt * self.speed;
                if self.current_time >= self.total_duration {
                    if self.looping {
                        self.current_time -= self.total_duration;
                    } else {
                        self.current_time = self.total_duration;
                        self.state = PlaybackState::Stopped;
                    }
                }
            }
            PlaybackState::PlayingReverse => {
                self.current_time -= dt * self.speed;
                if self.current_time <= 0.0 {
                    if self.looping {
                        self.current_time += self.total_duration;
                    } else {
                        self.current_time = 0.0;
                        self.state = PlaybackState::Stopped;
                    }
                }
            }
            PlaybackState::Stopped | PlaybackState::Paused => {}
        }
    }

    /// Get the current interpolation value for the first (or only) segment.
    /// Returns a value in [0, 1].
    pub fn get_lerp(&self) -> f32 {
        self.get_lerp_for_segment(0)
    }

    /// Get the interpolation value for a specific segment by index.
    pub fn get_lerp_for_segment(&self, index: usize) -> f32 {
        if let Some(segment) = self.segments.get(index) {
            match self.state {
                PlaybackState::PlayingReverse | PlaybackState::Paused
                    if matches!(self.state, PlaybackState::PlayingReverse) =>
                {
                    segment
                        .evaluate_reverse(self.current_time)
                        .unwrap_or(0.0)
                }
                _ => segment.evaluate(self.current_time).unwrap_or(0.0),
            }
        } else {
            0.0
        }
    }

    /// Get interpolation values for all segments as a vector.
    pub fn get_all_lerps(&self) -> Vec<f32> {
        self.segments
            .iter()
            .map(|seg| seg.evaluate(self.current_time).unwrap_or(0.0))
            .collect()
    }
}

impl Default for CurveSequence {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WidgetSpringAnimation — critically damped spring for UI
// ---------------------------------------------------------------------------

/// A critically damped spring animation for smooth reactive UI motion.
///
/// Spring animations feel more natural than tweens for interactive elements
/// because they respond gracefully to target changes mid-flight. This is used
/// for scroll position, resize, hover effects, and any other value that
/// reacts to user input.
///
/// Unlike the core `SpringAnimation`, this version supports 2D springs and
/// provides higher-level configuration presets.
#[derive(Debug, Clone)]
pub struct WidgetSpring {
    /// Current value.
    pub value: f32,
    /// Current velocity.
    velocity: f32,
    /// Target value.
    target: f32,
    /// Spring stiffness (higher = snappier, default 300).
    stiffness: f32,
    /// Damping coefficient (higher = less oscillation, default critical).
    damping: f32,
    /// Mass of the spring (default 1.0).
    mass: f32,
    /// Threshold for considering the spring at rest.
    rest_threshold: f32,
    /// Whether the spring has settled at the target.
    at_rest: bool,
}

impl WidgetSpring {
    /// Creates a new spring starting at `initial_value`.
    pub fn new(stiffness: f32, damping: f32) -> Self {
        Self {
            value: 0.0,
            velocity: 0.0,
            target: 0.0,
            stiffness,
            damping,
            mass: 1.0,
            rest_threshold: 0.01,
            at_rest: true,
        }
    }

    /// Creates a critically damped spring (fastest settling without
    /// oscillation).
    pub fn critically_damped(stiffness: f32) -> Self {
        let damping = 2.0 * stiffness.sqrt();
        Self::new(stiffness, damping)
    }

    /// Creates a slightly under-damped spring (small overshoot, bouncy feel).
    pub fn bouncy(stiffness: f32) -> Self {
        let damping = 1.4 * stiffness.sqrt();
        Self::new(stiffness, damping)
    }

    /// Creates a slow, smooth spring (for background transitions).
    pub fn slow() -> Self {
        Self::new(50.0, 14.0)
    }

    /// Creates a snappy spring (for hover effects).
    pub fn snappy() -> Self {
        Self::critically_damped(500.0)
    }

    /// Creates a responsive spring (for scroll and resize).
    pub fn responsive() -> Self {
        Self::critically_damped(300.0)
    }

    /// Set the initial value without triggering animation.
    pub fn set_value(&mut self, value: f32) {
        self.value = value;
        self.target = value;
        self.velocity = 0.0;
        self.at_rest = true;
    }

    /// Set a new target value. The spring will animate toward it.
    pub fn set_target(&mut self, target: f32) {
        if (self.target - target).abs() > 0.001 {
            self.target = target;
            self.at_rest = false;
        }
    }

    /// Returns the current target.
    pub fn target(&self) -> f32 {
        self.target
    }

    /// Returns `true` if the spring has settled.
    pub fn is_at_rest(&self) -> bool {
        self.at_rest
    }

    /// Returns the current value.
    pub fn current(&self) -> f32 {
        self.value
    }

    /// Advance the spring simulation by `dt` seconds.
    pub fn tick(&mut self, dt: f32) -> f32 {
        if self.at_rest {
            return self.value;
        }

        // Semi-implicit Euler integration (good stability for springs).
        let displacement = self.value - self.target;
        let spring_force = -self.stiffness * displacement;
        let damping_force = -self.damping * self.velocity;
        let acceleration = (spring_force + damping_force) / self.mass;

        self.velocity += acceleration * dt;
        self.value += self.velocity * dt;

        // Rest detection: if both displacement and velocity are below
        // threshold, snap to target.
        if displacement.abs() < self.rest_threshold
            && self.velocity.abs() < self.rest_threshold
        {
            self.value = self.target;
            self.velocity = 0.0;
            self.at_rest = true;
        }

        self.value
    }

    /// Instantly snap to the target (no animation).
    pub fn snap_to_target(&mut self) {
        self.value = self.target;
        self.velocity = 0.0;
        self.at_rest = true;
    }

    /// Apply an impulse (instantaneous velocity change).
    pub fn impulse(&mut self, velocity_delta: f32) {
        self.velocity += velocity_delta;
        self.at_rest = false;
    }
}

// ---------------------------------------------------------------------------
// WidgetSpring2D — 2D spring
// ---------------------------------------------------------------------------

/// A 2D spring animation for position-based effects.
#[derive(Debug, Clone)]
pub struct WidgetSpring2D {
    pub x: WidgetSpring,
    pub y: WidgetSpring,
}

impl WidgetSpring2D {
    /// Creates a 2D spring with the same parameters for both axes.
    pub fn new(stiffness: f32, damping: f32) -> Self {
        Self {
            x: WidgetSpring::new(stiffness, damping),
            y: WidgetSpring::new(stiffness, damping),
        }
    }

    pub fn critically_damped(stiffness: f32) -> Self {
        Self {
            x: WidgetSpring::critically_damped(stiffness),
            y: WidgetSpring::critically_damped(stiffness),
        }
    }

    pub fn set_value(&mut self, value: Vec2) {
        self.x.set_value(value.x);
        self.y.set_value(value.y);
    }

    pub fn set_target(&mut self, target: Vec2) {
        self.x.set_target(target.x);
        self.y.set_target(target.y);
    }

    pub fn current(&self) -> Vec2 {
        Vec2::new(self.x.current(), self.y.current())
    }

    pub fn target(&self) -> Vec2 {
        Vec2::new(self.x.target(), self.y.target())
    }

    pub fn is_at_rest(&self) -> bool {
        self.x.is_at_rest() && self.y.is_at_rest()
    }

    pub fn tick(&mut self, dt: f32) -> Vec2 {
        Vec2::new(self.x.tick(dt), self.y.tick(dt))
    }

    pub fn snap_to_target(&mut self) {
        self.x.snap_to_target();
        self.y.snap_to_target();
    }

    pub fn impulse(&mut self, velocity: Vec2) {
        self.x.impulse(velocity.x);
        self.y.impulse(velocity.y);
    }
}

// ---------------------------------------------------------------------------
// Lerp trait
// ---------------------------------------------------------------------------

/// Trait for types that can be linearly interpolated.
pub trait Lerp {
    fn lerp(&self, target: &Self, t: f32) -> Self;
}

impl Lerp for f32 {
    fn lerp(&self, target: &Self, t: f32) -> Self {
        self + (target - self) * t
    }
}

impl Lerp for Vec2 {
    fn lerp(&self, target: &Self, t: f32) -> Self {
        Vec2::new(
            self.x + (target.x - self.x) * t,
            self.y + (target.y - self.y) * t,
        )
    }
}

impl Lerp for crate::render_commands::Color {
    fn lerp(&self, target: &Self, t: f32) -> Self {
        crate::render_commands::Color::new(
            self.r + (target.r - self.r) * t,
            self.g + (target.g - self.g) * t,
            self.b + (target.b - self.b) * t,
            self.a + (target.a - self.a) * t,
        )
    }
}

// ---------------------------------------------------------------------------
// AnimatedValue<T> — generic animated value wrapper
// ---------------------------------------------------------------------------

/// Wraps a value and smoothly animates it from current to target when set.
///
/// The animation uses a configurable duration and easing function. Setting a
/// new value starts a transition from the current interpolated value to the
/// new target.
#[derive(Debug, Clone)]
pub struct AnimatedValue<T: Lerp + Clone + std::fmt::Debug> {
    /// Current (start of transition) value.
    from: T,
    /// Target value.
    to: T,
    /// Elapsed time since the last `set`.
    elapsed: f32,
    /// Duration of the transition.
    duration: f32,
    /// Easing function.
    easing: EasingKind,
    /// Whether a transition is in progress.
    animating: bool,
}

impl<T: Lerp + Clone + std::fmt::Debug> AnimatedValue<T> {
    /// Creates a new animated value starting at `initial`.
    pub fn new(initial: T, duration: f32, easing: EasingKind) -> Self {
        Self {
            from: initial.clone(),
            to: initial,
            elapsed: 0.0,
            duration: duration.max(0.001),
            easing,
            animating: false,
        }
    }

    /// Creates with default easing (cubic out) and 0.2s duration.
    pub fn default_animated(initial: T) -> Self {
        Self::new(initial, 0.2, EasingKind::CubicOut)
    }

    /// Set the target value. If different from the current interpolated value,
    /// starts a new transition.
    pub fn set(&mut self, value: T) {
        self.from = self.get();
        self.to = value;
        self.elapsed = 0.0;
        self.animating = true;
    }

    /// Set the value instantly without animation.
    pub fn set_immediate(&mut self, value: T) {
        self.from = value.clone();
        self.to = value;
        self.elapsed = self.duration;
        self.animating = false;
    }

    /// Get the current interpolated value.
    pub fn get(&self) -> T {
        if !self.animating {
            return self.to.clone();
        }
        let t = (self.elapsed / self.duration).clamp(0.0, 1.0);
        let eased = self.easing.evaluate(t);
        self.from.lerp(&self.to, eased)
    }

    /// Returns the target value (the value being animated toward).
    pub fn target(&self) -> &T {
        &self.to
    }

    /// Returns `true` if the animation has finished.
    pub fn is_done(&self) -> bool {
        !self.animating
    }

    /// Returns `true` if an animation is in progress.
    pub fn is_animating(&self) -> bool {
        self.animating
    }

    /// Advance the animation by `dt` seconds.
    pub fn tick(&mut self, dt: f32) {
        if !self.animating {
            return;
        }
        self.elapsed += dt;
        if self.elapsed >= self.duration {
            self.elapsed = self.duration;
            self.animating = false;
            self.from = self.to.clone();
        }
    }

    /// Change the transition duration.
    pub fn set_duration(&mut self, duration: f32) {
        self.duration = duration.max(0.001);
    }

    /// Change the easing function.
    pub fn set_easing(&mut self, easing: EasingKind) {
        self.easing = easing;
    }
}

// ---------------------------------------------------------------------------
// TransformAnimation — position + rotation + scale
// ---------------------------------------------------------------------------

/// Animates a 2D transform (position, rotation, scale) for use as a widget
/// render transform.
#[derive(Debug, Clone)]
pub struct TransformAnimation {
    /// Animated position offset.
    pub position: AnimatedValue<Vec2>,
    /// Animated rotation in radians.
    pub rotation: AnimatedValue<f32>,
    /// Animated uniform scale.
    pub scale: AnimatedValue<f32>,
}

impl TransformAnimation {
    /// Creates a new transform animation with identity transform.
    pub fn new(duration: f32, easing: EasingKind) -> Self {
        Self {
            position: AnimatedValue::new(Vec2::ZERO, duration, easing),
            rotation: AnimatedValue::new(0.0, duration, easing),
            scale: AnimatedValue::new(1.0, duration, easing),
        }
    }

    /// Creates with default parameters.
    pub fn default_transform() -> Self {
        Self::new(0.25, EasingKind::CubicOut)
    }

    /// Set the target position.
    pub fn set_position(&mut self, pos: Vec2) {
        self.position.set(pos);
    }

    /// Set the target rotation (in radians).
    pub fn set_rotation(&mut self, radians: f32) {
        self.rotation.set(radians);
    }

    /// Set the target scale.
    pub fn set_scale(&mut self, scale: f32) {
        self.scale.set(scale);
    }

    /// Set all components immediately (no animation).
    pub fn set_immediate(&mut self, pos: Vec2, rotation: f32, scale: f32) {
        self.position.set_immediate(pos);
        self.rotation.set_immediate(rotation);
        self.scale.set_immediate(scale);
    }

    /// Advance all animations.
    pub fn tick(&mut self, dt: f32) {
        self.position.tick(dt);
        self.rotation.tick(dt);
        self.scale.tick(dt);
    }

    /// Returns `true` if all animations are done.
    pub fn is_done(&self) -> bool {
        self.position.is_done() && self.rotation.is_done() && self.scale.is_done()
    }

    /// Returns `true` if any animation is in progress.
    pub fn is_animating(&self) -> bool {
        self.position.is_animating()
            || self.rotation.is_animating()
            || self.scale.is_animating()
    }

    /// Compute the current transform matrix.
    pub fn to_matrix(&self) -> glam::Mat3 {
        let pos = self.position.get();
        let rot = self.rotation.get();
        let scl = self.scale.get();

        let cos_r = rot.cos();
        let sin_r = rot.sin();

        // Construct: translate * rotate * scale
        glam::Mat3::from_cols(
            glam::Vec3::new(scl * cos_r, scl * sin_r, 0.0),
            glam::Vec3::new(-scl * sin_r, scl * cos_r, 0.0),
            glam::Vec3::new(pos.x, pos.y, 1.0),
        )
    }

    /// Compute the current position.
    pub fn current_position(&self) -> Vec2 {
        self.position.get()
    }

    /// Compute the current rotation in radians.
    pub fn current_rotation(&self) -> f32 {
        self.rotation.get()
    }

    /// Compute the current scale.
    pub fn current_scale(&self) -> f32 {
        self.scale.get()
    }
}

// ---------------------------------------------------------------------------
// AnimationController — manages multiple named animations
// ---------------------------------------------------------------------------

/// Manages a set of named animations for a widget.
///
/// Widgets can register named curve sequences and control them by name. This
/// is useful for complex widgets that have multiple independent animations
/// (e.g., a button with hover fade, press scale, and ripple).
pub struct AnimationController {
    sequences: HashMap<String, CurveSequence>,
    springs: HashMap<String, WidgetSpring>,
}

impl AnimationController {
    pub fn new() -> Self {
        Self {
            sequences: HashMap::new(),
            springs: HashMap::new(),
        }
    }

    /// Register a named curve sequence.
    pub fn add_sequence(&mut self, name: impl Into<String>, sequence: CurveSequence) {
        self.sequences.insert(name.into(), sequence);
    }

    /// Register a named spring.
    pub fn add_spring(&mut self, name: impl Into<String>, spring: WidgetSpring) {
        self.springs.insert(name.into(), spring);
    }

    /// Get a sequence by name.
    pub fn sequence(&self, name: &str) -> Option<&CurveSequence> {
        self.sequences.get(name)
    }

    /// Get a mutable sequence by name.
    pub fn sequence_mut(&mut self, name: &str) -> Option<&mut CurveSequence> {
        self.sequences.get_mut(name)
    }

    /// Get a spring by name.
    pub fn spring(&self, name: &str) -> Option<&WidgetSpring> {
        self.springs.get(name)
    }

    /// Get a mutable spring by name.
    pub fn spring_mut(&mut self, name: &str) -> Option<&mut WidgetSpring> {
        self.springs.get_mut(name)
    }

    /// Play a named sequence.
    pub fn play(&mut self, name: &str) {
        if let Some(seq) = self.sequences.get_mut(name) {
            seq.play();
        }
    }

    /// Play a named sequence in reverse.
    pub fn play_reverse(&mut self, name: &str) {
        if let Some(seq) = self.sequences.get_mut(name) {
            seq.play_reverse();
        }
    }

    /// Stop a named sequence.
    pub fn stop(&mut self, name: &str) {
        if let Some(seq) = self.sequences.get_mut(name) {
            seq.stop();
        }
    }

    /// Stop all animations.
    pub fn stop_all(&mut self) {
        for seq in self.sequences.values_mut() {
            seq.stop();
        }
        for spring in self.springs.values_mut() {
            spring.snap_to_target();
        }
    }

    /// Get the lerp value of a named sequence.
    pub fn get_lerp(&self, name: &str) -> f32 {
        self.sequences
            .get(name)
            .map(|s| s.get_lerp())
            .unwrap_or(0.0)
    }

    /// Get the current value of a named spring.
    pub fn get_spring_value(&self, name: &str) -> f32 {
        self.springs
            .get(name)
            .map(|s| s.current())
            .unwrap_or(0.0)
    }

    /// Set a spring target by name.
    pub fn set_spring_target(&mut self, name: &str, target: f32) {
        if let Some(spring) = self.springs.get_mut(name) {
            spring.set_target(target);
        }
    }

    /// Returns `true` if any animation is currently active.
    pub fn is_any_playing(&self) -> bool {
        self.sequences.values().any(|s| s.is_playing())
            || self.springs.values().any(|s| !s.is_at_rest())
    }

    /// Advance all animations by `dt` seconds.
    pub fn tick(&mut self, dt: f32) {
        for seq in self.sequences.values_mut() {
            seq.tick(dt);
        }
        for spring in self.springs.values_mut() {
            spring.tick(dt);
        }
    }
}

impl Default for AnimationController {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Common animation presets
// ---------------------------------------------------------------------------

/// Pre-built animation presets for common UI transitions.
pub struct AnimPresets;

impl AnimPresets {
    /// Fade-in animation (0.25s, cubic out).
    pub fn fade_in() -> CurveSequence {
        CurveSequence::simple(0.25, EasingKind::CubicOut)
    }

    /// Fade-out animation (0.2s, cubic in).
    pub fn fade_out() -> CurveSequence {
        CurveSequence::simple(0.2, EasingKind::CubicIn)
    }

    /// Slide-in from left (0.3s, cubic out).
    pub fn slide_in() -> CurveSequence {
        CurveSequence::simple(0.3, EasingKind::CubicOut)
    }

    /// Expand animation (0.35s, expo out).
    pub fn expand() -> CurveSequence {
        CurveSequence::simple(0.35, EasingKind::ExpoOut)
    }

    /// Collapse animation (0.25s, expo in).
    pub fn collapse() -> CurveSequence {
        CurveSequence::simple(0.25, EasingKind::ExpoIn)
    }

    /// Bounce-in animation (0.5s, elastic out).
    pub fn bounce_in() -> CurveSequence {
        CurveSequence::simple(0.5, EasingKind::ElasticOut)
    }

    /// Quick pop animation (0.15s, back out).
    pub fn pop() -> CurveSequence {
        CurveSequence::simple(0.15, EasingKind::BackOut)
    }

    /// Hover highlight spring.
    pub fn hover_spring() -> WidgetSpring {
        WidgetSpring::snappy()
    }

    /// Scroll position spring.
    pub fn scroll_spring() -> WidgetSpring {
        WidgetSpring::responsive()
    }

    /// Resize spring.
    pub fn resize_spring() -> WidgetSpring {
        WidgetSpring::critically_damped(200.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_easing_kind_evaluate() {
        assert!((EasingKind::Linear.evaluate(0.5) - 0.5).abs() < 0.001);
        assert!((EasingKind::Linear.evaluate(0.0)).abs() < 0.001);
        assert!((EasingKind::Linear.evaluate(1.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_easing_kind_inverse() {
        assert_eq!(EasingKind::QuadIn.inverse(), EasingKind::QuadOut);
        assert_eq!(EasingKind::QuadOut.inverse(), EasingKind::QuadIn);
        assert_eq!(EasingKind::Linear.inverse(), EasingKind::Linear);
    }

    #[test]
    fn test_curve_handle() {
        let handle = CurveHandle::new(0.0, 1.0, EasingKind::Linear);
        assert_eq!(handle.end_time(), 1.0);

        let v = handle.evaluate(0.5).unwrap();
        assert!((v - 0.5).abs() < 0.01);

        let v = handle.evaluate(0.0).unwrap();
        assert!(v.abs() < 0.01);

        let v = handle.evaluate(1.0).unwrap();
        assert!((v - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_curve_sequence_basic() {
        let mut seq = CurveSequence::simple(1.0, EasingKind::Linear);

        assert!(!seq.is_playing());
        assert!(seq.is_at_start());

        seq.play();
        assert!(seq.is_playing());

        // Advance halfway
        seq.tick(0.5);
        let lerp = seq.get_lerp();
        assert!((lerp - 0.5).abs() < 0.01);

        // Advance to end
        seq.tick(0.5);
        assert!(seq.is_at_end());
        let lerp = seq.get_lerp();
        assert!((lerp - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_curve_sequence_reverse() {
        let mut seq = CurveSequence::simple(1.0, EasingKind::Linear);
        seq.play_reverse();

        seq.tick(0.5);
        assert!(!seq.is_at_start());

        seq.tick(0.5);
        assert!(seq.is_at_start());
    }

    #[test]
    fn test_curve_sequence_pause_resume() {
        let mut seq = CurveSequence::simple(1.0, EasingKind::Linear);
        seq.play();

        seq.tick(0.3);
        let time_before_pause = seq.current_time();

        seq.pause();
        assert!(seq.is_paused());

        seq.tick(0.5);
        assert_eq!(seq.current_time(), time_before_pause);

        seq.resume();
        seq.tick(0.2);
        assert!(seq.current_time() > time_before_pause);
    }

    #[test]
    fn test_curve_sequence_looping() {
        let mut seq = CurveSequence::simple(0.5, EasingKind::Linear);
        seq.looping = true;
        seq.play();

        seq.tick(0.6);
        assert!(seq.is_playing());
        assert!(seq.current_time() < 0.5);
    }

    #[test]
    fn test_curve_sequence_two_phase() {
        let seq = CurveSequence::two_phase(
            0.3,
            EasingKind::CubicOut,
            0.5,
            EasingKind::CubicIn,
        );
        assert_eq!(seq.segment_count(), 2);
        assert!((seq.duration() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_widget_spring() {
        let mut spring = WidgetSpring::critically_damped(300.0);
        spring.set_value(0.0);
        spring.set_target(100.0);

        // Simulate for 1 second
        for _ in 0..60 {
            spring.tick(1.0 / 60.0);
        }

        // Should be close to target
        assert!((spring.current() - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_widget_spring_at_rest() {
        let mut spring = WidgetSpring::snappy();
        spring.set_value(50.0);
        assert!(spring.is_at_rest());

        spring.set_target(100.0);
        assert!(!spring.is_at_rest());

        // Simulate until at rest
        for _ in 0..300 {
            spring.tick(1.0 / 60.0);
            if spring.is_at_rest() {
                break;
            }
        }

        assert!(spring.is_at_rest());
        assert!((spring.current() - 100.0).abs() < 0.02);
    }

    #[test]
    fn test_widget_spring_impulse() {
        let mut spring = WidgetSpring::bouncy(200.0);
        spring.set_value(0.0);
        spring.impulse(50.0);

        assert!(!spring.is_at_rest());
        spring.tick(0.016);
        assert!(spring.current() > 0.0);
    }

    #[test]
    fn test_widget_spring_2d() {
        let mut spring = WidgetSpring2D::critically_damped(300.0);
        spring.set_value(Vec2::ZERO);
        spring.set_target(Vec2::new(100.0, 200.0));

        for _ in 0..120 {
            spring.tick(1.0 / 60.0);
        }

        let pos = spring.current();
        assert!((pos.x - 100.0).abs() < 1.0);
        assert!((pos.y - 200.0).abs() < 1.0);
    }

    #[test]
    fn test_animated_value_f32() {
        let mut val = AnimatedValue::new(0.0_f32, 1.0, EasingKind::Linear);

        val.set(100.0);
        assert!(val.is_animating());

        val.tick(0.5);
        let current = val.get();
        assert!((current - 50.0).abs() < 1.0);

        val.tick(0.5);
        assert!(val.is_done());
        assert!((val.get() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_animated_value_set_immediate() {
        let mut val = AnimatedValue::new(0.0_f32, 1.0, EasingKind::Linear);
        val.set_immediate(42.0);
        assert!(val.is_done());
        assert!((val.get() - 42.0).abs() < 0.01);
    }

    #[test]
    fn test_animated_value_retarget() {
        let mut val = AnimatedValue::new(0.0_f32, 1.0, EasingKind::Linear);
        val.set(100.0);
        val.tick(0.5);

        // Retarget mid-animation
        val.set(200.0);
        assert!(val.is_animating());

        // Current "from" should be about 50
        let current = val.get();
        assert!((current - 50.0).abs() < 2.0);
    }

    #[test]
    fn test_animated_value_color() {
        let mut val = AnimatedValue::new(
            crate::render_commands::Color::RED,
            0.5,
            EasingKind::Linear,
        );
        val.set(crate::render_commands::Color::BLUE);
        val.tick(0.25);
        let c = val.get();
        assert!(c.r > 0.0);
        assert!(c.b > 0.0);
    }

    #[test]
    fn test_transform_animation() {
        let mut xform = TransformAnimation::default_transform();
        xform.set_position(Vec2::new(100.0, 50.0));
        xform.set_scale(2.0);
        xform.set_rotation(std::f32::consts::FRAC_PI_4);

        assert!(xform.is_animating());

        for _ in 0..60 {
            xform.tick(1.0 / 60.0);
        }

        assert!((xform.current_position().x - 100.0).abs() < 1.0);
        assert!((xform.current_scale() - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_transform_matrix() {
        let mut xform = TransformAnimation::default_transform();
        xform.set_immediate(Vec2::new(10.0, 20.0), 0.0, 1.0);
        let mat = xform.to_matrix();
        // Translation should be in the third column
        assert!((mat.z_axis.x - 10.0).abs() < 0.01);
        assert!((mat.z_axis.y - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_animation_controller() {
        let mut ctrl = AnimationController::new();
        ctrl.add_sequence("fade", AnimPresets::fade_in());
        ctrl.add_spring("hover", AnimPresets::hover_spring());

        ctrl.play("fade");
        assert!(ctrl.is_any_playing());

        ctrl.set_spring_target("hover", 1.0);
        ctrl.tick(0.1);

        let fade_val = ctrl.get_lerp("fade");
        assert!(fade_val > 0.0);

        let spring_val = ctrl.get_spring_value("hover");
        assert!(spring_val > 0.0);
    }

    #[test]
    fn test_animation_controller_stop_all() {
        let mut ctrl = AnimationController::new();
        ctrl.add_sequence("a", AnimPresets::fade_in());
        ctrl.add_sequence("b", AnimPresets::slide_in());
        ctrl.play("a");
        ctrl.play("b");

        ctrl.stop_all();
        assert!(!ctrl.is_any_playing());
    }

    #[test]
    fn test_anim_presets() {
        let fade = AnimPresets::fade_in();
        assert!(fade.duration() > 0.0);

        let spring = AnimPresets::hover_spring();
        assert!(spring.is_at_rest());
    }

    #[test]
    fn test_spring_snap() {
        let mut spring = WidgetSpring::slow();
        spring.set_value(0.0);
        spring.set_target(100.0);
        spring.snap_to_target();
        assert!(spring.is_at_rest());
        assert!((spring.current() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_curve_sequence_speed() {
        let mut seq = CurveSequence::simple(1.0, EasingKind::Linear);
        seq.speed = 2.0;
        seq.play();

        seq.tick(0.25);
        // At 2x speed, 0.25s real time = 0.5s sequence time
        assert!((seq.progress() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_curve_sequence_seek() {
        let mut seq = CurveSequence::simple(1.0, EasingKind::Linear);
        seq.seek(0.75);
        assert!((seq.progress() - 0.75).abs() < 0.01);
    }
}
