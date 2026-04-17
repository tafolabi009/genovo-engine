//! Widget animation system for the Slate UI framework.
//!
//! Provides curve sequences, easing functions (all 24 standard types),
//! critically damped spring physics, animated values, and widget transitions.
//! This module is the animation foundation that makes every hover, expand,
//! collapse, and slide feel polished.

use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::render_commands::Color;

// ---------------------------------------------------------------------------
// SlateEasing -- 24 easing functions
// ---------------------------------------------------------------------------

/// Easing function enumeration with all 24+ standard types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SlateEasing {
    Linear,
    QuadIn,
    QuadOut,
    QuadInOut,
    CubicIn,
    CubicOut,
    CubicInOut,
    QuartIn,
    QuartOut,
    QuartInOut,
    QuintIn,
    QuintOut,
    QuintInOut,
    SinIn,
    SinOut,
    SinInOut,
    ExpoIn,
    ExpoOut,
    ExpoInOut,
    CircIn,
    CircOut,
    CircInOut,
    ElasticOut,
    BounceOut,
    BackIn,
    BackOut,
    BackInOut,
}

impl Default for SlateEasing {
    fn default() -> Self {
        Self::Linear
    }
}

impl SlateEasing {
    /// Evaluate the easing function at normalized time t in [0, 1].
    pub fn evaluate(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::QuadIn => t * t,
            Self::QuadOut => 1.0 - (1.0 - t) * (1.0 - t),
            Self::QuadInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
                }
            }
            Self::CubicIn => t * t * t,
            Self::CubicOut => 1.0 - (1.0 - t).powi(3),
            Self::CubicInOut => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
                }
            }
            Self::QuartIn => t * t * t * t,
            Self::QuartOut => 1.0 - (1.0 - t).powi(4),
            Self::QuartInOut => {
                if t < 0.5 {
                    8.0 * t * t * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(4) / 2.0
                }
            }
            Self::QuintIn => t * t * t * t * t,
            Self::QuintOut => 1.0 - (1.0 - t).powi(5),
            Self::QuintInOut => {
                if t < 0.5 {
                    16.0 * t * t * t * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(5) / 2.0
                }
            }
            Self::SinIn => 1.0 - (t * std::f32::consts::FRAC_PI_2).cos(),
            Self::SinOut => (t * std::f32::consts::FRAC_PI_2).sin(),
            Self::SinInOut => -((t * std::f32::consts::PI).cos() - 1.0) / 2.0,
            Self::ExpoIn => {
                if t == 0.0 {
                    0.0
                } else {
                    (2.0_f32).powf(10.0 * t - 10.0)
                }
            }
            Self::ExpoOut => {
                if t == 1.0 {
                    1.0
                } else {
                    1.0 - (2.0_f32).powf(-10.0 * t)
                }
            }
            Self::ExpoInOut => {
                if t == 0.0 {
                    0.0
                } else if t == 1.0 {
                    1.0
                } else if t < 0.5 {
                    (2.0_f32).powf(20.0 * t - 10.0) / 2.0
                } else {
                    (2.0 - (2.0_f32).powf(-20.0 * t + 10.0)) / 2.0
                }
            }
            Self::CircIn => 1.0 - (1.0 - t * t).sqrt(),
            Self::CircOut => (1.0 - (t - 1.0).powi(2)).sqrt(),
            Self::CircInOut => {
                if t < 0.5 {
                    (1.0 - (1.0 - (2.0 * t).powi(2)).sqrt()) / 2.0
                } else {
                    ((1.0 - (-2.0 * t + 2.0).powi(2)).sqrt() + 1.0) / 2.0
                }
            }
            Self::ElasticOut => {
                if t == 0.0 || t == 1.0 {
                    return t;
                }
                let c4 = (2.0 * std::f32::consts::PI) / 3.0;
                (2.0_f32).powf(-10.0 * t)
                    * ((t * 10.0 - 0.75) * c4).sin()
                    + 1.0
            }
            Self::BounceOut => bounce_out(t),
            Self::BackIn => {
                let c1: f32 = 1.70158;
                let c3 = c1 + 1.0;
                c3 * t * t * t - c1 * t * t
            }
            Self::BackOut => {
                let c1: f32 = 1.70158;
                let c3 = c1 + 1.0;
                1.0 + c3 * (t - 1.0).powi(3) + c1 * (t - 1.0).powi(2)
            }
            Self::BackInOut => {
                let c1: f32 = 1.70158;
                let c2 = c1 * 1.525;
                if t < 0.5 {
                    ((2.0 * t).powi(2) * ((c2 + 1.0) * 2.0 * t - c2))
                        / 2.0
                } else {
                    ((2.0 * t - 2.0).powi(2)
                        * ((c2 + 1.0) * (2.0 * t - 2.0) + c2)
                        + 2.0)
                        / 2.0
                }
            }
        }
    }

    /// Returns the name of this easing function.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::QuadIn => "QuadIn",
            Self::QuadOut => "QuadOut",
            Self::QuadInOut => "QuadInOut",
            Self::CubicIn => "CubicIn",
            Self::CubicOut => "CubicOut",
            Self::CubicInOut => "CubicInOut",
            Self::QuartIn => "QuartIn",
            Self::QuartOut => "QuartOut",
            Self::QuartInOut => "QuartInOut",
            Self::QuintIn => "QuintIn",
            Self::QuintOut => "QuintOut",
            Self::QuintInOut => "QuintInOut",
            Self::SinIn => "SinIn",
            Self::SinOut => "SinOut",
            Self::SinInOut => "SinInOut",
            Self::ExpoIn => "ExpoIn",
            Self::ExpoOut => "ExpoOut",
            Self::ExpoInOut => "ExpoInOut",
            Self::CircIn => "CircIn",
            Self::CircOut => "CircOut",
            Self::CircInOut => "CircInOut",
            Self::ElasticOut => "ElasticOut",
            Self::BounceOut => "BounceOut",
            Self::BackIn => "BackIn",
            Self::BackOut => "BackOut",
            Self::BackInOut => "BackInOut",
        }
    }

    /// All easing types.
    pub fn all() -> &'static [SlateEasing] {
        &[
            Self::Linear,
            Self::QuadIn,
            Self::QuadOut,
            Self::QuadInOut,
            Self::CubicIn,
            Self::CubicOut,
            Self::CubicInOut,
            Self::QuartIn,
            Self::QuartOut,
            Self::QuartInOut,
            Self::QuintIn,
            Self::QuintOut,
            Self::QuintInOut,
            Self::SinIn,
            Self::SinOut,
            Self::SinInOut,
            Self::ExpoIn,
            Self::ExpoOut,
            Self::ExpoInOut,
            Self::CircIn,
            Self::CircOut,
            Self::CircInOut,
            Self::ElasticOut,
            Self::BounceOut,
            Self::BackIn,
            Self::BackOut,
            Self::BackInOut,
        ]
    }
}

/// Bounce-out helper function.
fn bounce_out(t: f32) -> f32 {
    let n1: f32 = 7.5625;
    let d1: f32 = 2.75;
    if t < 1.0 / d1 {
        n1 * t * t
    } else if t < 2.0 / d1 {
        let t = t - 1.5 / d1;
        n1 * t * t + 0.75
    } else if t < 2.5 / d1 {
        let t = t - 2.25 / d1;
        n1 * t * t + 0.9375
    } else {
        let t = t - 2.625 / d1;
        n1 * t * t + 0.984375
    }
}

// ---------------------------------------------------------------------------
// CurveHandle
// ---------------------------------------------------------------------------

/// Handle referencing a specific curve in a CurveSequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CurveHandle(pub usize);

// ---------------------------------------------------------------------------
// CurveEntry
// ---------------------------------------------------------------------------

/// A single curve within a CurveSequence timeline.
#[derive(Debug, Clone)]
struct CurveEntry {
    start_time: f32,
    duration: f32,
    easing: SlateEasing,
}

// ---------------------------------------------------------------------------
// CurveSequence
// ---------------------------------------------------------------------------

/// A timeline of easing curves that can be played forward and backward.
///
/// Multiple curves can overlap in time. Each curve independently interpolates
/// from 0 to 1 (or 1 to 0 when reversed) over its duration using its easing.
#[derive(Debug, Clone)]
pub struct CurveSequence {
    curves: Vec<CurveEntry>,
    current_time: f32,
    total_duration: f32,
    playing: bool,
    reversed: bool,
    looping: bool,
    playback_rate: f32,
}

impl CurveSequence {
    /// Creates a new empty curve sequence.
    pub fn new() -> Self {
        Self {
            curves: Vec::new(),
            current_time: 0.0,
            total_duration: 0.0,
            playing: false,
            reversed: false,
            looping: false,
            playback_rate: 1.0,
        }
    }

    /// Add a curve to the timeline. Returns a handle for querying its value.
    pub fn add_curve(
        &mut self,
        start_time: f32,
        duration: f32,
        easing: SlateEasing,
    ) -> CurveHandle {
        let handle = CurveHandle(self.curves.len());
        let end = start_time + duration;
        if end > self.total_duration {
            self.total_duration = end;
        }
        self.curves.push(CurveEntry {
            start_time,
            duration,
            easing,
        });
        handle
    }

    /// Start playing forward from the beginning.
    pub fn play(&mut self) {
        self.current_time = 0.0;
        self.reversed = false;
        self.playing = true;
    }

    /// Start playing in reverse from the end.
    pub fn play_reverse(&mut self) {
        self.current_time = self.total_duration;
        self.reversed = true;
        self.playing = true;
    }

    /// Pause playback.
    pub fn pause(&mut self) {
        self.playing = false;
    }

    /// Resume playback.
    pub fn resume(&mut self) {
        self.playing = true;
    }

    /// Jump to a specific time.
    pub fn jump_to(&mut self, time: f32) {
        self.current_time = time.clamp(0.0, self.total_duration);
    }

    /// Set looping.
    pub fn set_loop(&mut self, looping: bool) {
        self.looping = looping;
    }

    /// Set playback rate (1.0 = normal speed).
    pub fn set_playback_rate(&mut self, rate: f32) {
        self.playback_rate = rate;
    }

    /// Advance the sequence by `dt` seconds.
    pub fn update(&mut self, dt: f32) {
        if !self.playing {
            return;
        }
        let delta = dt * self.playback_rate;
        if self.reversed {
            self.current_time -= delta;
            if self.current_time <= 0.0 {
                if self.looping {
                    self.current_time = self.total_duration;
                } else {
                    self.current_time = 0.0;
                    self.playing = false;
                }
            }
        } else {
            self.current_time += delta;
            if self.current_time >= self.total_duration {
                if self.looping {
                    self.current_time = 0.0;
                } else {
                    self.current_time = self.total_duration;
                    self.playing = false;
                }
            }
        }
    }

    /// Get the interpolated value [0, 1] for a specific curve handle.
    pub fn get_lerp(&self, handle: CurveHandle) -> f32 {
        if let Some(curve) = self.curves.get(handle.0) {
            let local_time = self.current_time - curve.start_time;
            if local_time <= 0.0 {
                return 0.0;
            }
            if local_time >= curve.duration {
                return 1.0;
            }
            let t = local_time / curve.duration;
            curve.easing.evaluate(t)
        } else {
            0.0
        }
    }

    /// Whether the sequence is currently playing.
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Whether the sequence is at the start (time = 0).
    pub fn is_at_start(&self) -> bool {
        self.current_time <= 0.0
    }

    /// Whether the sequence is at the end.
    pub fn is_at_end(&self) -> bool {
        self.current_time >= self.total_duration
    }

    /// Current time.
    pub fn current_time(&self) -> f32 {
        self.current_time
    }

    /// Total duration.
    pub fn total_duration(&self) -> f32 {
        self.total_duration
    }
}

impl Default for CurveSequence {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CriticalSpring -- critically damped spring physics
// ---------------------------------------------------------------------------

/// A critically damped spring animation.
///
/// The spring equation: `x'' + 2*zeta*omega*x' + omega^2*x = 0`
/// where zeta is the damping ratio and omega is the natural frequency.
/// When zeta = 1.0 (critically damped), the system reaches the target as
/// fast as possible without oscillation.
#[derive(Debug, Clone)]
pub struct CriticalSpring {
    /// Current value.
    pub value: f32,
    /// Current velocity.
    pub velocity: f32,
    /// Target value.
    pub target: f32,
    /// Spring stiffness (higher = snappier).
    pub stiffness: f32,
    /// Damping ratio. 1.0 = critically damped, <1.0 = underdamped (bouncy),
    /// >1.0 = overdamped (sluggish).
    pub damping_ratio: f32,
    /// Rest threshold.
    pub rest_threshold: f32,
    /// Whether the spring has settled at the target.
    pub settled: bool,
}

impl CriticalSpring {
    /// Create a new spring with the given stiffness and damping ratio.
    pub fn new(stiffness: f32, damping_ratio: f32) -> Self {
        Self {
            value: 0.0,
            velocity: 0.0,
            target: 0.0,
            stiffness,
            damping_ratio,
            rest_threshold: 0.001,
            settled: true,
        }
    }

    /// Create a critically damped spring (ratio = 1.0).
    pub fn critically_damped(stiffness: f32) -> Self {
        Self::new(stiffness, 1.0)
    }

    /// Set the target value. Wakes the spring if settled.
    pub fn set_target(&mut self, target: f32) {
        self.target = target;
        self.settled = false;
    }

    /// Set the current value directly (jump).
    pub fn set_value(&mut self, value: f32) {
        self.value = value;
        self.velocity = 0.0;
        self.settled =
            (self.value - self.target).abs() < self.rest_threshold;
    }

    /// Set both value and target (instant snap).
    pub fn snap_to(&mut self, value: f32) {
        self.value = value;
        self.target = value;
        self.velocity = 0.0;
        self.settled = true;
    }

    /// Advance the spring by dt seconds. Returns the current value.
    ///
    /// Uses semi-implicit Euler integration with the damped harmonic
    /// oscillator equation:
    ///   F = -k * x - c * v
    /// where k = stiffness, c = 2 * zeta * omega, omega = sqrt(k).
    pub fn tick(&mut self, dt: f32) -> f32 {
        if self.settled {
            return self.value;
        }

        let omega = self.stiffness.sqrt();
        let damping = 2.0 * self.damping_ratio * omega;

        let displacement = self.value - self.target;
        let spring_force = -self.stiffness * displacement;
        let damping_force = -damping * self.velocity;
        let acceleration = spring_force + damping_force;

        self.velocity += acceleration * dt;
        self.value += self.velocity * dt;

        let disp = (self.value - self.target).abs();
        let vel = self.velocity.abs();
        if disp < self.rest_threshold && vel < self.rest_threshold {
            self.value = self.target;
            self.velocity = 0.0;
            self.settled = true;
        }

        self.value
    }

    /// Whether the spring has settled at the target.
    pub fn is_settled(&self) -> bool {
        self.settled
    }

    /// Current value.
    pub fn get(&self) -> f32 {
        self.value
    }
}

// ---------------------------------------------------------------------------
// AnimatedFloat
// ---------------------------------------------------------------------------

/// An animated float that smoothly transitions between values using an
/// easing function.
#[derive(Debug, Clone)]
pub struct AnimatedFloat {
    current: f32,
    start: f32,
    target: f32,
    elapsed: f32,
    duration: f32,
    easing: SlateEasing,
    animating: bool,
}

impl AnimatedFloat {
    /// Create a new animated float at the initial value.
    pub fn new(initial: f32) -> Self {
        Self {
            current: initial,
            start: initial,
            target: initial,
            elapsed: 0.0,
            duration: 0.3,
            easing: SlateEasing::CubicOut,
            animating: false,
        }
    }

    /// Start animating toward a target value.
    pub fn animate_to(
        &mut self,
        target: f32,
        duration: f32,
        easing: SlateEasing,
    ) {
        self.start = self.current;
        self.target = target;
        self.duration = duration.max(0.001);
        self.easing = easing;
        self.elapsed = 0.0;
        self.animating = true;
    }

    /// Set the value immediately (no animation).
    pub fn set_immediate(&mut self, value: f32) {
        self.current = value;
        self.start = value;
        self.target = value;
        self.animating = false;
    }

    /// Advance the animation by dt seconds.
    pub fn update(&mut self, dt: f32) {
        if !self.animating {
            return;
        }
        self.elapsed += dt;
        let t = (self.elapsed / self.duration).min(1.0);
        let eased = self.easing.evaluate(t);
        self.current = self.start + (self.target - self.start) * eased;
        if t >= 1.0 {
            self.current = self.target;
            self.animating = false;
        }
    }

    /// Get the current interpolated value.
    pub fn get(&self) -> f32 {
        self.current
    }

    /// Get the target value.
    pub fn target(&self) -> f32 {
        self.target
    }

    /// Whether the animation is in progress.
    pub fn is_animating(&self) -> bool {
        self.animating
    }
}

// ---------------------------------------------------------------------------
// AnimatedVec2
// ---------------------------------------------------------------------------

/// An animated Vec2 value. Each component is animated independently.
#[derive(Debug, Clone)]
pub struct AnimatedVec2 {
    pub x: AnimatedFloat,
    pub y: AnimatedFloat,
}

impl AnimatedVec2 {
    /// Create a new animated Vec2 at the initial value.
    pub fn new(initial: Vec2) -> Self {
        Self {
            x: AnimatedFloat::new(initial.x),
            y: AnimatedFloat::new(initial.y),
        }
    }

    /// Start animating toward a target value.
    pub fn animate_to(
        &mut self,
        target: Vec2,
        duration: f32,
        easing: SlateEasing,
    ) {
        self.x.animate_to(target.x, duration, easing);
        self.y.animate_to(target.y, duration, easing);
    }

    /// Set immediately.
    pub fn set_immediate(&mut self, value: Vec2) {
        self.x.set_immediate(value.x);
        self.y.set_immediate(value.y);
    }

    /// Advance the animation.
    pub fn update(&mut self, dt: f32) {
        self.x.update(dt);
        self.y.update(dt);
    }

    /// Get the current value.
    pub fn get(&self) -> Vec2 {
        Vec2::new(self.x.get(), self.y.get())
    }

    /// Whether either component is still animating.
    pub fn is_animating(&self) -> bool {
        self.x.is_animating() || self.y.is_animating()
    }
}

// ---------------------------------------------------------------------------
// AnimatedColor
// ---------------------------------------------------------------------------

/// An animated colour value. Each RGBA channel is animated independently.
#[derive(Debug, Clone)]
pub struct AnimatedColor {
    pub r: AnimatedFloat,
    pub g: AnimatedFloat,
    pub b: AnimatedFloat,
    pub a: AnimatedFloat,
}

impl AnimatedColor {
    /// Create a new animated colour at the initial value.
    pub fn new(initial: Color) -> Self {
        Self {
            r: AnimatedFloat::new(initial.r),
            g: AnimatedFloat::new(initial.g),
            b: AnimatedFloat::new(initial.b),
            a: AnimatedFloat::new(initial.a),
        }
    }

    /// Start animating toward a target colour.
    pub fn animate_to(
        &mut self,
        target: Color,
        duration: f32,
        easing: SlateEasing,
    ) {
        self.r.animate_to(target.r, duration, easing);
        self.g.animate_to(target.g, duration, easing);
        self.b.animate_to(target.b, duration, easing);
        self.a.animate_to(target.a, duration, easing);
    }

    /// Advance the animation.
    pub fn update(&mut self, dt: f32) {
        self.r.update(dt);
        self.g.update(dt);
        self.b.update(dt);
        self.a.update(dt);
    }

    /// Get the current colour.
    pub fn get(&self) -> Color {
        Color::new(
            self.r.get(),
            self.g.get(),
            self.b.get(),
            self.a.get(),
        )
    }

    /// Whether any channel is still animating.
    pub fn is_animating(&self) -> bool {
        self.r.is_animating()
            || self.g.is_animating()
            || self.b.is_animating()
            || self.a.is_animating()
    }
}

// ---------------------------------------------------------------------------
// TransitionManager
// ---------------------------------------------------------------------------

/// Manages a set of named animated values, automatically removing completed
/// animations.
pub struct TransitionManager {
    floats: Vec<(String, AnimatedFloat)>,
    vec2s: Vec<(String, AnimatedVec2)>,
    colors: Vec<(String, AnimatedColor)>,
    springs: Vec<(String, CriticalSpring)>,
}

impl TransitionManager {
    /// Create an empty transition manager.
    pub fn new() -> Self {
        Self {
            floats: Vec::new(),
            vec2s: Vec::new(),
            colors: Vec::new(),
            springs: Vec::new(),
        }
    }

    /// Add a float animation.
    pub fn add_float(
        &mut self,
        name: impl Into<String>,
        anim: AnimatedFloat,
    ) {
        self.floats.push((name.into(), anim));
    }

    /// Add a Vec2 animation.
    pub fn add_vec2(
        &mut self,
        name: impl Into<String>,
        anim: AnimatedVec2,
    ) {
        self.vec2s.push((name.into(), anim));
    }

    /// Add a colour animation.
    pub fn add_color(
        &mut self,
        name: impl Into<String>,
        anim: AnimatedColor,
    ) {
        self.colors.push((name.into(), anim));
    }

    /// Add a spring animation.
    pub fn add_spring(
        &mut self,
        name: impl Into<String>,
        spring: CriticalSpring,
    ) {
        self.springs.push((name.into(), spring));
    }

    /// Advance all animations by dt seconds and clean up completed ones.
    pub fn update(&mut self, dt: f32) {
        for (_, anim) in &mut self.floats {
            anim.update(dt);
        }
        for (_, anim) in &mut self.vec2s {
            anim.update(dt);
        }
        for (_, anim) in &mut self.colors {
            anim.update(dt);
        }
        for (_, spring) in &mut self.springs {
            spring.tick(dt);
        }

        self.floats.retain(|(_, a)| a.is_animating());
        self.vec2s.retain(|(_, a)| a.is_animating());
        self.colors.retain(|(_, a)| a.is_animating());
        self.springs.retain(|(_, s)| !s.is_settled());
    }

    /// Get a float value by name.
    pub fn get_float(&self, name: &str) -> Option<f32> {
        self.floats
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, a)| a.get())
    }

    /// Get a Vec2 value by name.
    pub fn get_vec2(&self, name: &str) -> Option<Vec2> {
        self.vec2s
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, a)| a.get())
    }

    /// Get a colour value by name.
    pub fn get_color(&self, name: &str) -> Option<Color> {
        self.colors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, a)| a.get())
    }

    /// Get a spring value by name.
    pub fn get_spring(&self, name: &str) -> Option<f32> {
        self.springs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, s)| s.get())
    }

    /// Whether there are no active animations.
    pub fn is_empty(&self) -> bool {
        self.floats.is_empty()
            && self.vec2s.is_empty()
            && self.colors.is_empty()
            && self.springs.is_empty()
    }

    /// Number of active animations.
    pub fn active_count(&self) -> usize {
        self.floats.len()
            + self.vec2s.len()
            + self.colors.len()
            + self.springs.len()
    }
}

impl Default for TransitionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Widget Transitions
// ---------------------------------------------------------------------------

/// Direction for slide transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SlideDirection {
    Left,
    Right,
    Up,
    Down,
}

/// Predefined widget transition types.
#[derive(Debug, Clone)]
pub enum WidgetTransition {
    /// Fade from transparent to opaque.
    FadeIn {
        duration: f32,
        easing: SlateEasing,
    },
    /// Fade from opaque to transparent.
    FadeOut {
        duration: f32,
        easing: SlateEasing,
    },
    /// Slide in from an edge.
    SlideIn {
        direction: SlideDirection,
        duration: f32,
        distance: f32,
        easing: SlateEasing,
    },
    /// Slide out to an edge.
    SlideOut {
        direction: SlideDirection,
        duration: f32,
        distance: f32,
        easing: SlateEasing,
    },
    /// Scale from 0 to 1.
    ScaleIn {
        duration: f32,
        easing: SlateEasing,
    },
    /// Scale from 1 to 0.
    ScaleOut {
        duration: f32,
        easing: SlateEasing,
    },
    /// Expand from zero height to full height.
    ExpandHeight {
        duration: f32,
        easing: SlateEasing,
    },
    /// Collapse from full height to zero.
    CollapseHeight {
        duration: f32,
        easing: SlateEasing,
    },
}

/// State of an active widget transition.
#[derive(Debug, Clone)]
pub struct ActiveTransition {
    /// The transition definition.
    pub transition: WidgetTransition,
    /// Elapsed time since the transition started.
    pub elapsed: f32,
    /// Whether the transition has completed.
    pub completed: bool,
}

impl ActiveTransition {
    /// Create a new active transition.
    pub fn new(transition: WidgetTransition) -> Self {
        Self {
            transition,
            elapsed: 0.0,
            completed: false,
        }
    }

    /// Get the duration of the transition.
    pub fn duration(&self) -> f32 {
        match &self.transition {
            WidgetTransition::FadeIn { duration, .. } => *duration,
            WidgetTransition::FadeOut { duration, .. } => *duration,
            WidgetTransition::SlideIn { duration, .. } => *duration,
            WidgetTransition::SlideOut { duration, .. } => *duration,
            WidgetTransition::ScaleIn { duration, .. } => *duration,
            WidgetTransition::ScaleOut { duration, .. } => *duration,
            WidgetTransition::ExpandHeight { duration, .. } => *duration,
            WidgetTransition::CollapseHeight { duration, .. } => *duration,
        }
    }

    /// Get the easing function.
    pub fn easing(&self) -> SlateEasing {
        match &self.transition {
            WidgetTransition::FadeIn { easing, .. } => *easing,
            WidgetTransition::FadeOut { easing, .. } => *easing,
            WidgetTransition::SlideIn { easing, .. } => *easing,
            WidgetTransition::SlideOut { easing, .. } => *easing,
            WidgetTransition::ScaleIn { easing, .. } => *easing,
            WidgetTransition::ScaleOut { easing, .. } => *easing,
            WidgetTransition::ExpandHeight { easing, .. } => *easing,
            WidgetTransition::CollapseHeight { easing, .. } => *easing,
        }
    }

    /// Advance the transition by dt seconds.
    pub fn update(&mut self, dt: f32) {
        if self.completed {
            return;
        }
        self.elapsed += dt;
        if self.elapsed >= self.duration() {
            self.completed = true;
        }
    }

    /// Get the normalized progress [0, 1] with easing applied.
    pub fn progress(&self) -> f32 {
        let t = (self.elapsed / self.duration()).clamp(0.0, 1.0);
        self.easing().evaluate(t)
    }

    /// Compute the opacity for this transition.
    pub fn opacity(&self) -> f32 {
        let p = self.progress();
        match &self.transition {
            WidgetTransition::FadeIn { .. } => p,
            WidgetTransition::FadeOut { .. } => 1.0 - p,
            _ => 1.0,
        }
    }

    /// Compute the scale for this transition.
    pub fn scale(&self) -> f32 {
        let p = self.progress();
        match &self.transition {
            WidgetTransition::ScaleIn { .. } => p,
            WidgetTransition::ScaleOut { .. } => 1.0 - p,
            _ => 1.0,
        }
    }

    /// Compute the translation offset for this transition.
    pub fn offset(&self) -> Vec2 {
        let p = self.progress();
        match &self.transition {
            WidgetTransition::SlideIn {
                direction,
                distance,
                ..
            } => {
                let remaining = 1.0 - p;
                match direction {
                    SlideDirection::Left => {
                        Vec2::new(-distance * remaining, 0.0)
                    }
                    SlideDirection::Right => {
                        Vec2::new(distance * remaining, 0.0)
                    }
                    SlideDirection::Up => {
                        Vec2::new(0.0, -distance * remaining)
                    }
                    SlideDirection::Down => {
                        Vec2::new(0.0, distance * remaining)
                    }
                }
            }
            WidgetTransition::SlideOut {
                direction,
                distance,
                ..
            } => match direction {
                SlideDirection::Left => Vec2::new(-distance * p, 0.0),
                SlideDirection::Right => Vec2::new(distance * p, 0.0),
                SlideDirection::Up => Vec2::new(0.0, -distance * p),
                SlideDirection::Down => Vec2::new(0.0, distance * p),
            },
            _ => Vec2::ZERO,
        }
    }

    /// Compute the height fraction for expand/collapse transitions.
    pub fn height_fraction(&self) -> f32 {
        let p = self.progress();
        match &self.transition {
            WidgetTransition::ExpandHeight { .. } => p,
            WidgetTransition::CollapseHeight { .. } => 1.0 - p,
            _ => 1.0,
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
    fn test_all_easings_boundary() {
        for easing in SlateEasing::all() {
            let v0 = easing.evaluate(0.0);
            let v1 = easing.evaluate(1.0);
            assert!(
                (v0 - 0.0).abs() < 0.01,
                "{}: f(0) = {} (expected ~0)",
                easing.name(),
                v0
            );
            assert!(
                (v1 - 1.0).abs() < 0.01,
                "{}: f(1) = {} (expected ~1)",
                easing.name(),
                v1
            );
        }
    }

    #[test]
    fn test_linear_easing() {
        assert!(
            (SlateEasing::Linear.evaluate(0.5) - 0.5).abs() < 0.001
        );
    }

    #[test]
    fn test_quad_in() {
        assert!(
            (SlateEasing::QuadIn.evaluate(0.5) - 0.25).abs() < 0.001
        );
    }

    #[test]
    fn test_cubic_out() {
        assert!(
            (SlateEasing::CubicOut.evaluate(0.5) - 0.875).abs() < 0.001
        );
    }

    #[test]
    fn test_curve_sequence() {
        let mut seq = CurveSequence::new();
        let h0 = seq.add_curve(0.0, 1.0, SlateEasing::Linear);
        let h1 = seq.add_curve(0.5, 1.0, SlateEasing::Linear);

        seq.play();
        seq.update(0.5);
        assert!((seq.get_lerp(h0) - 0.5).abs() < 0.01);
        assert!((seq.get_lerp(h1) - 0.0).abs() < 0.01);

        seq.update(0.5);
        assert!((seq.get_lerp(h0) - 1.0).abs() < 0.01);
        assert!((seq.get_lerp(h1) - 0.5).abs() < 0.02);
    }

    #[test]
    fn test_curve_sequence_reverse() {
        let mut seq = CurveSequence::new();
        let h0 = seq.add_curve(0.0, 1.0, SlateEasing::Linear);
        seq.play_reverse();
        assert!((seq.get_lerp(h0) - 1.0).abs() < 0.01);
        seq.update(0.5);
        assert!((seq.get_lerp(h0) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_critical_spring() {
        let mut spring = CriticalSpring::critically_damped(200.0);
        spring.snap_to(0.0);
        spring.set_target(100.0);
        assert!(!spring.is_settled());

        for _ in 0..1000 {
            spring.tick(0.016);
        }
        assert!(spring.is_settled());
        assert!((spring.get() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_underdamped_spring() {
        let mut spring = CriticalSpring::new(200.0, 0.3);
        spring.set_target(1.0);
        let mut max_value: f32 = 0.0;
        for _ in 0..500 {
            let v = spring.tick(0.016);
            max_value = max_value.max(v);
        }
        assert!(
            max_value > 1.0,
            "Underdamped spring should overshoot"
        );
    }

    #[test]
    fn test_animated_float() {
        let mut anim = AnimatedFloat::new(0.0);
        anim.animate_to(100.0, 1.0, SlateEasing::Linear);
        assert!(anim.is_animating());

        anim.update(0.5);
        assert!((anim.get() - 50.0).abs() < 1.0);

        anim.update(0.6);
        assert!(!anim.is_animating());
        assert!((anim.get() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_animated_vec2() {
        let mut anim = AnimatedVec2::new(Vec2::ZERO);
        anim.animate_to(
            Vec2::new(100.0, 200.0),
            1.0,
            SlateEasing::Linear,
        );
        anim.update(0.5);
        let v = anim.get();
        assert!((v.x - 50.0).abs() < 1.0);
        assert!((v.y - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_animated_color() {
        let mut anim = AnimatedColor::new(Color::BLACK);
        anim.animate_to(Color::WHITE, 1.0, SlateEasing::Linear);
        anim.update(0.5);
        let c = anim.get();
        assert!((c.r - 0.5).abs() < 0.01);
        assert!((c.g - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_transition_manager() {
        let mut mgr = TransitionManager::new();
        let mut anim = AnimatedFloat::new(0.0);
        anim.animate_to(1.0, 0.5, SlateEasing::Linear);
        mgr.add_float("test", anim);
        assert_eq!(mgr.active_count(), 1);

        mgr.update(0.25);
        let v = mgr.get_float("test").unwrap();
        assert!((v - 0.5).abs() < 0.01);

        mgr.update(0.3);
        assert!(mgr.is_empty());
    }

    #[test]
    fn test_fade_transition() {
        let mut t = ActiveTransition::new(WidgetTransition::FadeIn {
            duration: 1.0,
            easing: SlateEasing::Linear,
        });
        assert!((t.opacity() - 0.0).abs() < 0.01);
        t.update(0.5);
        assert!((t.opacity() - 0.5).abs() < 0.01);
        t.update(0.6);
        assert!((t.opacity() - 1.0).abs() < 0.01);
        assert!(t.completed);
    }

    #[test]
    fn test_slide_transition() {
        let mut t = ActiveTransition::new(WidgetTransition::SlideIn {
            direction: SlideDirection::Left,
            duration: 1.0,
            distance: 200.0,
            easing: SlateEasing::Linear,
        });
        let offset = t.offset();
        assert!((offset.x - (-200.0)).abs() < 1.0);

        t.update(0.5);
        let offset = t.offset();
        assert!((offset.x - (-100.0)).abs() < 1.0);
    }

    #[test]
    fn test_expand_collapse() {
        let mut t =
            ActiveTransition::new(WidgetTransition::ExpandHeight {
                duration: 0.3,
                easing: SlateEasing::CubicOut,
            });
        assert!((t.height_fraction() - 0.0).abs() < 0.01);
        t.update(0.3);
        assert!((t.height_fraction() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_bounce_out_values() {
        assert!(
            (SlateEasing::BounceOut.evaluate(1.0) - 1.0).abs() < 0.001
        );
        assert!(
            (SlateEasing::BounceOut.evaluate(0.0) - 0.0).abs() < 0.001
        );
    }

    #[test]
    fn test_elastic_out() {
        let v = SlateEasing::ElasticOut.evaluate(0.5);
        assert!(v > 0.5);
    }

    #[test]
    fn test_back_in() {
        let v_early = SlateEasing::BackIn.evaluate(0.1);
        assert!(v_early < 0.0, "BackIn should undershoot early");
    }
}
