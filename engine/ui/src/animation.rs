//! UI animation system.
//!
//! Provides easing functions, property-based animations, spring physics, and
//! animation sequencing/grouping. Animations drive smooth transitions in the
//! styling and layout systems.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// EasingFunction
// ---------------------------------------------------------------------------

/// Predefined easing curves. Each maps an input `t` in `[0, 1]` to an output
/// value, typically also in `[0, 1]` (though Elastic / Back may overshoot).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EasingFunction {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    EaseInQuad,
    EaseOutQuad,
    EaseInOutQuad,
    EaseInCubic,
    EaseOutCubic,
    EaseInOutCubic,
    EaseInQuart,
    EaseOutQuart,
    EaseInOutQuart,
    EaseInQuint,
    EaseOutQuint,
    EaseInOutQuint,
    EaseInSine,
    EaseOutSine,
    EaseInOutSine,
    EaseInExpo,
    EaseOutExpo,
    EaseInOutExpo,
    EaseInCirc,
    EaseOutCirc,
    EaseInOutCirc,
    Bounce,
    BounceIn,
    Elastic,
    ElasticIn,
    Back,
    BackIn,
    /// Cubic bezier with two control points (x1, y1, x2, y2).
    CubicBezier(f32, f32, f32, f32),
}

impl EasingFunction {
    /// Evaluate the easing function at `t` (should be in `[0, 1]`).
    pub fn evaluate(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseIn => ease_in_cubic(t),
            Self::EaseOut => ease_out_cubic(t),
            Self::EaseInOut => ease_in_out_cubic(t),
            Self::EaseInQuad => t * t,
            Self::EaseOutQuad => 1.0 - (1.0 - t) * (1.0 - t),
            Self::EaseInOutQuad => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
                }
            }
            Self::EaseInCubic => ease_in_cubic(t),
            Self::EaseOutCubic => ease_out_cubic(t),
            Self::EaseInOutCubic => ease_in_out_cubic(t),
            Self::EaseInQuart => t * t * t * t,
            Self::EaseOutQuart => 1.0 - (1.0 - t).powi(4),
            Self::EaseInOutQuart => {
                if t < 0.5 {
                    8.0 * t * t * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(4) / 2.0
                }
            }
            Self::EaseInQuint => t * t * t * t * t,
            Self::EaseOutQuint => 1.0 - (1.0 - t).powi(5),
            Self::EaseInOutQuint => {
                if t < 0.5 {
                    16.0 * t * t * t * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(5) / 2.0
                }
            }
            Self::EaseInSine => {
                1.0 - (t * std::f32::consts::FRAC_PI_2).cos()
            }
            Self::EaseOutSine => (t * std::f32::consts::FRAC_PI_2).sin(),
            Self::EaseInOutSine => -(((t * std::f32::consts::PI).cos() - 1.0) / 2.0),
            Self::EaseInExpo => {
                if t == 0.0 {
                    0.0
                } else {
                    (2.0_f32).powf(10.0 * t - 10.0)
                }
            }
            Self::EaseOutExpo => {
                if t == 1.0 {
                    1.0
                } else {
                    1.0 - (2.0_f32).powf(-10.0 * t)
                }
            }
            Self::EaseInOutExpo => {
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
            Self::EaseInCirc => 1.0 - (1.0 - t * t).sqrt(),
            Self::EaseOutCirc => (1.0 - (t - 1.0).powi(2)).sqrt(),
            Self::EaseInOutCirc => {
                if t < 0.5 {
                    (1.0 - (1.0 - (2.0 * t).powi(2)).sqrt()) / 2.0
                } else {
                    ((1.0 - (-2.0 * t + 2.0).powi(2)).sqrt() + 1.0) / 2.0
                }
            }
            Self::Bounce => bounce_out(t),
            Self::BounceIn => 1.0 - bounce_out(1.0 - t),
            Self::Elastic => elastic_out(t),
            Self::ElasticIn => elastic_in(t),
            Self::Back => back_out(t),
            Self::BackIn => back_in(t),
            Self::CubicBezier(x1, y1, x2, y2) => {
                cubic_bezier_eval(*x1, *y1, *x2, *y2, t)
            }
        }
    }
}

impl Default for EasingFunction {
    fn default() -> Self {
        Self::Linear
    }
}

// ---------------------------------------------------------------------------
// Easing math helpers
// ---------------------------------------------------------------------------

fn ease_in_cubic(t: f32) -> f32 {
    t * t * t
}

fn ease_out_cubic(t: f32) -> f32 {
    1.0 - (1.0 - t).powi(3)
}

fn ease_in_out_cubic(t: f32) -> f32 {
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
    }
}

fn bounce_out(t: f32) -> f32 {
    const N1: f32 = 7.5625;
    const D1: f32 = 2.75;
    if t < 1.0 / D1 {
        N1 * t * t
    } else if t < 2.0 / D1 {
        let t = t - 1.5 / D1;
        N1 * t * t + 0.75
    } else if t < 2.5 / D1 {
        let t = t - 2.25 / D1;
        N1 * t * t + 0.9375
    } else {
        let t = t - 2.625 / D1;
        N1 * t * t + 0.984375
    }
}

fn elastic_out(t: f32) -> f32 {
    if t == 0.0 || t == 1.0 {
        return t;
    }
    let c4 = (2.0 * std::f32::consts::PI) / 3.0;
    (2.0_f32).powf(-10.0 * t) * ((t * 10.0 - 0.75) * c4).sin() + 1.0
}

fn elastic_in(t: f32) -> f32 {
    if t == 0.0 || t == 1.0 {
        return t;
    }
    let c4 = (2.0 * std::f32::consts::PI) / 3.0;
    -(2.0_f32).powf(10.0 * t - 10.0) * ((t * 10.0 - 10.75) * c4).sin()
}

fn back_out(t: f32) -> f32 {
    const C1: f32 = 1.70158;
    const C3: f32 = C1 + 1.0;
    1.0 + C3 * (t - 1.0).powi(3) + C1 * (t - 1.0).powi(2)
}

fn back_in(t: f32) -> f32 {
    const C1: f32 = 1.70158;
    const C3: f32 = C1 + 1.0;
    C3 * t * t * t - C1 * t * t
}

/// Evaluate a cubic bezier curve defined by control points `(x1,y1)` and
/// `(x2,y2)` at input `t`. Uses Newton's method to invert the x(t) mapping.
fn cubic_bezier_eval(x1: f32, y1: f32, x2: f32, y2: f32, x: f32) -> f32 {
    // Find the parametric t that corresponds to `x` using Newton-Raphson.
    let mut t = x; // initial guess
    for _ in 0..8 {
        let cx = cubic_bezier_sample(x1, x2, t) - x;
        if cx.abs() < 1e-6 {
            break;
        }
        let dx = cubic_bezier_slope(x1, x2, t);
        if dx.abs() < 1e-6 {
            break;
        }
        t -= cx / dx;
        t = t.clamp(0.0, 1.0);
    }
    cubic_bezier_sample(y1, y2, t)
}

fn cubic_bezier_sample(a: f32, b: f32, t: f32) -> f32 {
    // B(t) = 3(1-t)^2*t*a + 3(1-t)*t^2*b + t^3
    let t2 = t * t;
    let t3 = t2 * t;
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    3.0 * mt2 * t * a + 3.0 * mt * t2 * b + t3
}

fn cubic_bezier_slope(a: f32, b: f32, t: f32) -> f32 {
    let mt = 1.0 - t;
    3.0 * mt * mt * a + 6.0 * mt * t * (b - a) + 3.0 * t * t * (1.0 - b)
}

// ---------------------------------------------------------------------------
// UIAnimation
// ---------------------------------------------------------------------------

/// State of a running animation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnimationState {
    /// Waiting to start (delayed).
    Pending,
    /// Actively interpolating.
    Running,
    /// Reached the end; may loop.
    Completed,
    /// Explicitly cancelled.
    Cancelled,
}

/// Which property of a UI element is being animated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnimatedProperty {
    /// X position offset.
    PositionX,
    /// Y position offset.
    PositionY,
    /// Width.
    Width,
    /// Height.
    Height,
    /// Opacity (0..1).
    Opacity,
    /// Rotation in radians.
    Rotation,
    /// Uniform scale factor.
    Scale,
    /// Background color red channel.
    ColorR,
    /// Background color green channel.
    ColorG,
    /// Background color blue channel.
    ColorB,
    /// Background color alpha channel.
    ColorA,
    /// Border radius.
    BorderRadius,
    /// Padding (all sides).
    PaddingAll,
    /// Margin (all sides).
    MarginAll,
    /// Font size.
    FontSize,
    /// Scroll offset X.
    ScrollX,
    /// Scroll offset Y.
    ScrollY,
    /// Custom named property.
    Custom(String),
}

/// Loop behaviour for an animation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoopMode {
    /// Play once and stop.
    Once,
    /// Play forward, then backward, then forward, etc.
    PingPong,
    /// Restart from the beginning when done.
    Loop,
    /// Loop a fixed number of times.
    Count(u32),
}

impl Default for LoopMode {
    fn default() -> Self {
        Self::Once
    }
}

/// Animates a single numeric UI property from `start` to `end` over a
/// duration, using an easing function.
#[derive(Debug, Clone)]
pub struct UIAnimation {
    /// Which property to animate.
    pub property: AnimatedProperty,
    /// Starting value.
    pub start_value: f32,
    /// Ending value.
    pub end_value: f32,
    /// Total duration in seconds.
    pub duration: f32,
    /// Delay before starting, in seconds.
    pub delay: f32,
    /// Easing curve.
    pub easing: EasingFunction,
    /// Loop behaviour.
    pub loop_mode: LoopMode,
    /// Current elapsed time (including delay).
    pub elapsed: f32,
    /// Current state.
    pub state: AnimationState,
    /// Number of loops completed.
    pub loops_completed: u32,
    /// Whether playing in reverse (for PingPong).
    pub reversed: bool,
}

impl UIAnimation {
    /// Creates a new animation.
    pub fn new(
        property: AnimatedProperty,
        start: f32,
        end: f32,
        duration: f32,
    ) -> Self {
        Self {
            property,
            start_value: start,
            end_value: end,
            duration: duration.max(0.001),
            delay: 0.0,
            easing: EasingFunction::EaseInOut,
            loop_mode: LoopMode::Once,
            elapsed: 0.0,
            state: AnimationState::Pending,
            loops_completed: 0,
            reversed: false,
        }
    }

    /// Builder: set delay.
    pub fn with_delay(mut self, delay: f32) -> Self {
        self.delay = delay;
        self
    }

    /// Builder: set easing function.
    pub fn with_easing(mut self, easing: EasingFunction) -> Self {
        self.easing = easing;
        self
    }

    /// Builder: set loop mode.
    pub fn with_loop(mut self, mode: LoopMode) -> Self {
        self.loop_mode = mode;
        self
    }

    /// Advance the animation by `dt` seconds. Returns the current interpolated
    /// value.
    pub fn update(&mut self, dt: f32) -> f32 {
        if self.state == AnimationState::Completed || self.state == AnimationState::Cancelled {
            return if self.reversed {
                self.start_value
            } else {
                self.end_value
            };
        }

        self.elapsed += dt;

        // Handle delay.
        if self.elapsed < self.delay {
            self.state = AnimationState::Pending;
            return self.start_value;
        }

        self.state = AnimationState::Running;
        let active_time = self.elapsed - self.delay;
        let raw_t = (active_time / self.duration).min(1.0);

        let t = if self.reversed { 1.0 - raw_t } else { raw_t };
        let eased = self.easing.evaluate(t);
        let value = self.start_value + (self.end_value - self.start_value) * eased;

        // Check if this pass is complete.
        if raw_t >= 1.0 {
            match self.loop_mode {
                LoopMode::Once => {
                    self.state = AnimationState::Completed;
                }
                LoopMode::PingPong => {
                    self.reversed = !self.reversed;
                    self.elapsed = self.delay;
                    self.loops_completed += 1;
                }
                LoopMode::Loop => {
                    self.elapsed = self.delay;
                    self.loops_completed += 1;
                }
                LoopMode::Count(max) => {
                    self.loops_completed += 1;
                    if self.loops_completed >= max {
                        self.state = AnimationState::Completed;
                    } else {
                        self.elapsed = self.delay;
                    }
                }
            }
        }

        value
    }

    /// Current interpolated value without advancing time.
    pub fn current_value(&self) -> f32 {
        if self.state == AnimationState::Pending {
            return self.start_value;
        }
        let active_time = (self.elapsed - self.delay).max(0.0);
        let raw_t = (active_time / self.duration).clamp(0.0, 1.0);
        let t = if self.reversed { 1.0 - raw_t } else { raw_t };
        let eased = self.easing.evaluate(t);
        self.start_value + (self.end_value - self.start_value) * eased
    }

    /// Cancel the animation.
    pub fn cancel(&mut self) {
        self.state = AnimationState::Cancelled;
    }

    /// Reset the animation to the beginning.
    pub fn reset(&mut self) {
        self.elapsed = 0.0;
        self.state = AnimationState::Pending;
        self.loops_completed = 0;
        self.reversed = false;
    }

    /// Whether the animation has finished.
    pub fn is_done(&self) -> bool {
        matches!(
            self.state,
            AnimationState::Completed | AnimationState::Cancelled
        )
    }
}

// ---------------------------------------------------------------------------
// AnimationSequence
// ---------------------------------------------------------------------------

/// Plays a list of animations one after the other.
pub struct AnimationSequence {
    animations: Vec<UIAnimation>,
    current_index: usize,
    pub state: AnimationState,
}

impl AnimationSequence {
    pub fn new(animations: Vec<UIAnimation>) -> Self {
        Self {
            animations,
            current_index: 0,
            state: AnimationState::Pending,
        }
    }

    /// Advance the sequence. Returns `(property, value)` pairs for each
    /// animation that is currently producing output.
    pub fn update(&mut self, dt: f32) -> Vec<(AnimatedProperty, f32)> {
        let mut results = Vec::new();
        if self.state == AnimationState::Completed || self.state == AnimationState::Cancelled {
            return results;
        }
        self.state = AnimationState::Running;

        if self.current_index >= self.animations.len() {
            self.state = AnimationState::Completed;
            return results;
        }

        let anim = &mut self.animations[self.current_index];
        let value = anim.update(dt);
        results.push((anim.property.clone(), value));

        if anim.is_done() {
            self.current_index += 1;
            if self.current_index >= self.animations.len() {
                self.state = AnimationState::Completed;
            }
        }

        results
    }

    pub fn is_done(&self) -> bool {
        matches!(
            self.state,
            AnimationState::Completed | AnimationState::Cancelled
        )
    }

    pub fn reset(&mut self) {
        self.current_index = 0;
        self.state = AnimationState::Pending;
        for anim in &mut self.animations {
            anim.reset();
        }
    }

    pub fn cancel(&mut self) {
        self.state = AnimationState::Cancelled;
    }
}

// ---------------------------------------------------------------------------
// AnimationGroup
// ---------------------------------------------------------------------------

/// Plays multiple animations simultaneously.
pub struct AnimationGroup {
    animations: Vec<UIAnimation>,
    pub state: AnimationState,
}

impl AnimationGroup {
    pub fn new(animations: Vec<UIAnimation>) -> Self {
        Self {
            animations,
            state: AnimationState::Pending,
        }
    }

    /// Advance all animations. Returns `(property, value)` for each.
    pub fn update(&mut self, dt: f32) -> Vec<(AnimatedProperty, f32)> {
        if self.state == AnimationState::Completed || self.state == AnimationState::Cancelled {
            return Vec::new();
        }
        self.state = AnimationState::Running;

        let mut results = Vec::with_capacity(self.animations.len());
        let mut all_done = true;
        for anim in &mut self.animations {
            let value = anim.update(dt);
            results.push((anim.property.clone(), value));
            if !anim.is_done() {
                all_done = false;
            }
        }

        if all_done {
            self.state = AnimationState::Completed;
        }

        results
    }

    pub fn is_done(&self) -> bool {
        matches!(
            self.state,
            AnimationState::Completed | AnimationState::Cancelled
        )
    }

    pub fn reset(&mut self) {
        self.state = AnimationState::Pending;
        for anim in &mut self.animations {
            anim.reset();
        }
    }

    pub fn cancel(&mut self) {
        self.state = AnimationState::Cancelled;
    }
}

// ---------------------------------------------------------------------------
// SpringAnimation — damped harmonic oscillator
// ---------------------------------------------------------------------------

/// A spring-based animation that simulates a damped harmonic oscillator.
///
/// Spring animations feel more natural than tween-based ones for interactive
/// elements because they respond to target changes mid-flight without
/// discontinuities.
#[derive(Debug, Clone)]
pub struct SpringAnimation {
    /// Current value.
    pub value: f32,
    /// Current velocity.
    pub velocity: f32,
    /// Target value the spring is pulling toward.
    pub target: f32,
    /// Stiffness coefficient (higher = snappier).
    pub stiffness: f32,
    /// Damping coefficient (higher = less oscillation).
    pub damping: f32,
    /// Mass of the simulated object.
    pub mass: f32,
    /// Threshold below which the spring is considered at rest.
    pub rest_threshold: f32,
    /// Whether the spring has settled.
    pub at_rest: bool,
}

impl SpringAnimation {
    /// Creates a spring animation starting at `initial` heading toward
    /// `target`.
    pub fn new(initial: f32, target: f32) -> Self {
        Self {
            value: initial,
            velocity: 0.0,
            target,
            stiffness: 200.0,
            damping: 20.0,
            mass: 1.0,
            rest_threshold: 0.01,
            at_rest: false,
        }
    }

    /// Builder: set stiffness.
    pub fn with_stiffness(mut self, stiffness: f32) -> Self {
        self.stiffness = stiffness;
        self
    }

    /// Builder: set damping.
    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping;
        self
    }

    /// Builder: set mass.
    pub fn with_mass(mut self, mass: f32) -> Self {
        self.mass = mass.max(0.01);
        self
    }

    /// Sets a new target; the spring immediately starts pulling toward it.
    pub fn set_target(&mut self, target: f32) {
        self.target = target;
        self.at_rest = false;
    }

    /// Simulate one time-step using semi-implicit Euler integration.
    pub fn update(&mut self, dt: f32) -> f32 {
        if self.at_rest {
            return self.value;
        }

        // F = -k * displacement - c * velocity
        let displacement = self.value - self.target;
        let spring_force = -self.stiffness * displacement;
        let damping_force = -self.damping * self.velocity;
        let acceleration = (spring_force + damping_force) / self.mass;

        // Semi-implicit Euler: update velocity first, then position.
        self.velocity += acceleration * dt;
        self.value += self.velocity * dt;

        // Rest detection.
        if displacement.abs() < self.rest_threshold
            && self.velocity.abs() < self.rest_threshold
        {
            self.value = self.target;
            self.velocity = 0.0;
            self.at_rest = true;
        }

        self.value
    }

    /// Reset to a new initial value.
    pub fn reset(&mut self, initial: f32) {
        self.value = initial;
        self.velocity = 0.0;
        self.at_rest = false;
    }

    /// Critically damped coefficient for the current stiffness and mass.
    pub fn critical_damping(&self) -> f32 {
        2.0 * (self.stiffness * self.mass).sqrt()
    }

    /// Sets damping to the critical value (no oscillation, fastest settling).
    pub fn set_critical_damping(&mut self) {
        self.damping = self.critical_damping();
    }
}
