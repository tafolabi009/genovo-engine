//! Animation curves and spline types for the Genovo engine.
//!
//! Provides multiple curve/spline types used throughout the engine:
//!
//! - **AnimationCurve** — keyframed float curves with Hermite interpolation
//! - **BezierCurve** — cubic Bezier curves with De Casteljau evaluation
//! - **CatmullRomSpline** — smooth spline that passes through control points
//! - **BSpline** — B-spline with arbitrary knot vectors and De Boor evaluation
//!
//! These are used for animation easing, camera paths, procedural motion,
//! particle parameter curves, and UI animation.

use glam::Vec3;

// ===========================================================================
// AnimationCurve
// ===========================================================================

/// How the curve handles time values outside its keyframe range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurveMode {
    /// Clamp to the first/last keyframe value.
    Clamp,
    /// Loop: wrap time around the curve duration.
    Loop,
    /// PingPong: alternate forward and backward.
    PingPong,
}

impl Default for CurveMode {
    fn default() -> Self {
        Self::Clamp
    }
}

/// A keyframe in an animation curve.
///
/// Stores a time-value pair along with incoming and outgoing tangents
/// for Hermite (cubic) interpolation between keyframes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveKeyframe {
    /// Time position of this keyframe.
    pub time: f32,
    /// Value at this keyframe.
    pub value: f32,
    /// Incoming tangent (slope arriving at this keyframe).
    pub in_tangent: f32,
    /// Outgoing tangent (slope leaving this keyframe).
    pub out_tangent: f32,
}

impl CurveKeyframe {
    /// Create a new keyframe with zero tangents (flat).
    pub fn new(time: f32, value: f32) -> Self {
        Self {
            time,
            value,
            in_tangent: 0.0,
            out_tangent: 0.0,
        }
    }

    /// Create a keyframe with explicit tangents.
    pub fn with_tangents(time: f32, value: f32, in_tangent: f32, out_tangent: f32) -> Self {
        Self {
            time,
            value,
            in_tangent,
            out_tangent,
        }
    }
}

/// A keyframed float curve with Hermite interpolation.
///
/// Used for animating single float properties (opacity, scale, rotation
/// angles, blend weights, etc.) over time. Supports multiple interpolation
/// modes and wrap behavior.
///
/// # Interpolation
///
/// Between adjacent keyframes, the curve uses cubic Hermite interpolation:
///
/// ```text
/// p(t) = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
/// ```
///
/// where h00..h11 are the Hermite basis functions and m0/m1 are the
/// scaled tangents of the surrounding keyframes.
#[derive(Debug, Clone)]
pub struct AnimationCurve {
    /// Keyframes sorted by time.
    keys: Vec<CurveKeyframe>,
    /// How the curve handles out-of-range time values.
    pub mode: CurveMode,
}

impl AnimationCurve {
    /// Create an empty animation curve.
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            mode: CurveMode::Clamp,
        }
    }

    /// Create a curve with the given keys and mode.
    pub fn from_keys(mut keys: Vec<CurveKeyframe>, mode: CurveMode) -> Self {
        keys.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
        Self { keys, mode }
    }

    /// Add a keyframe. The curve is re-sorted by time.
    pub fn add_key(&mut self, key: CurveKeyframe) {
        self.keys.push(key);
        self.keys
            .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Add a keyframe at the given time and value with zero tangents.
    pub fn add_key_simple(&mut self, time: f32, value: f32) {
        self.add_key(CurveKeyframe::new(time, value));
    }

    /// Remove the keyframe at the given index.
    pub fn remove_key(&mut self, index: usize) {
        if index < self.keys.len() {
            self.keys.remove(index);
        }
    }

    /// Number of keyframes.
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// Get a reference to a keyframe by index.
    pub fn key(&self, index: usize) -> Option<&CurveKeyframe> {
        self.keys.get(index)
    }

    /// Duration of the curve (time of last key - time of first key).
    pub fn duration(&self) -> f32 {
        if self.keys.len() < 2 {
            return 0.0;
        }
        self.keys.last().unwrap().time - self.keys.first().unwrap().time
    }

    /// Wrap the input time according to the curve mode.
    fn wrap_time(&self, time: f32) -> f32 {
        if self.keys.len() < 2 {
            return time;
        }
        let first = self.keys[0].time;
        let last = self.keys[self.keys.len() - 1].time;
        let duration = last - first;

        if duration <= 0.0 {
            return first;
        }

        match self.mode {
            CurveMode::Clamp => time.clamp(first, last),
            CurveMode::Loop => {
                let t = (time - first).rem_euclid(duration);
                first + t
            }
            CurveMode::PingPong => {
                let t = (time - first) / duration;
                let cycle = t.floor() as i32;
                let frac = t - t.floor();
                if cycle % 2 == 0 {
                    first + frac * duration
                } else {
                    last - frac * duration
                }
            }
        }
    }

    /// Evaluate the curve at the given time.
    ///
    /// Uses cubic Hermite interpolation between keyframes.
    pub fn evaluate(&self, time: f32) -> f32 {
        if self.keys.is_empty() {
            return 0.0;
        }
        if self.keys.len() == 1 {
            return self.keys[0].value;
        }

        let time = self.wrap_time(time);

        // Before first key.
        if time <= self.keys[0].time {
            return self.keys[0].value;
        }
        // After last key.
        let last = self.keys.len() - 1;
        if time >= self.keys[last].time {
            return self.keys[last].value;
        }

        // Find the bracketing keyframes.
        let mut i = 0;
        for k in 0..last {
            if self.keys[k + 1].time >= time {
                i = k;
                break;
            }
        }

        let k0 = &self.keys[i];
        let k1 = &self.keys[i + 1];
        let dt = k1.time - k0.time;
        if dt < f32::EPSILON {
            return k0.value;
        }
        let t = (time - k0.time) / dt;

        // Hermite interpolation.
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        h00 * k0.value + h10 * (k0.out_tangent * dt) + h01 * k1.value + h11 * (k1.in_tangent * dt)
    }

    /// Auto-compute smooth tangents for all keyframes using Catmull-Rom style.
    pub fn auto_tangents(&mut self) {
        let n = self.keys.len();
        if n < 2 {
            return;
        }

        for i in 0..n {
            let tangent = if i == 0 {
                // Forward difference.
                let dt = self.keys[1].time - self.keys[0].time;
                if dt > f32::EPSILON {
                    (self.keys[1].value - self.keys[0].value) / dt
                } else {
                    0.0
                }
            } else if i == n - 1 {
                // Backward difference.
                let dt = self.keys[n - 1].time - self.keys[n - 2].time;
                if dt > f32::EPSILON {
                    (self.keys[n - 1].value - self.keys[n - 2].value) / dt
                } else {
                    0.0
                }
            } else {
                // Central difference.
                let dt = self.keys[i + 1].time - self.keys[i - 1].time;
                if dt > f32::EPSILON {
                    (self.keys[i + 1].value - self.keys[i - 1].value) / dt
                } else {
                    0.0
                }
            };

            self.keys[i].in_tangent = tangent;
            self.keys[i].out_tangent = tangent;
        }
    }

    // -- Pre-built curves ---------------------------------------------------

    /// Create a linear ramp from 0 to 1 over the range [0, 1].
    pub fn linear() -> Self {
        Self::from_keys(
            vec![
                CurveKeyframe::with_tangents(0.0, 0.0, 1.0, 1.0),
                CurveKeyframe::with_tangents(1.0, 1.0, 1.0, 1.0),
            ],
            CurveMode::Clamp,
        )
    }

    /// Create a quadratic ease-in curve (slow start, fast end).
    pub fn ease_in() -> Self {
        Self::from_keys(
            vec![
                CurveKeyframe::with_tangents(0.0, 0.0, 0.0, 0.0),
                CurveKeyframe::with_tangents(1.0, 1.0, 2.0, 2.0),
            ],
            CurveMode::Clamp,
        )
    }

    /// Create a quadratic ease-out curve (fast start, slow end).
    pub fn ease_out() -> Self {
        Self::from_keys(
            vec![
                CurveKeyframe::with_tangents(0.0, 0.0, 2.0, 2.0),
                CurveKeyframe::with_tangents(1.0, 1.0, 0.0, 0.0),
            ],
            CurveMode::Clamp,
        )
    }

    /// Create an ease-in-out curve (smooth start and end).
    pub fn ease_in_out() -> Self {
        Self::from_keys(
            vec![
                CurveKeyframe::with_tangents(0.0, 0.0, 0.0, 0.0),
                CurveKeyframe::with_tangents(1.0, 1.0, 0.0, 0.0),
            ],
            CurveMode::Clamp,
        )
    }

    /// Create a constant-value curve.
    pub fn constant(value: f32) -> Self {
        Self::from_keys(
            vec![
                CurveKeyframe::new(0.0, value),
                CurveKeyframe::new(1.0, value),
            ],
            CurveMode::Clamp,
        )
    }

    /// Create a bell/pulse curve: 0 -> 1 -> 0 over [0, 1].
    pub fn bell() -> Self {
        let mut curve = Self::from_keys(
            vec![
                CurveKeyframe::new(0.0, 0.0),
                CurveKeyframe::new(0.5, 1.0),
                CurveKeyframe::new(1.0, 0.0),
            ],
            CurveMode::Clamp,
        );
        curve.auto_tangents();
        curve
    }
}

impl Default for AnimationCurve {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// BezierCurve
// ===========================================================================

/// A cubic Bezier curve defined by four control points.
///
/// The curve starts at P0, ends at P1, and is shaped by the tangent
/// handles P1 and P2. Evaluated using De Casteljau's algorithm for
/// numerical stability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BezierCurve {
    /// First control point (start).
    pub p0: Vec3,
    /// Second control point (tangent handle for start).
    pub p1: Vec3,
    /// Third control point (tangent handle for end).
    pub p2: Vec3,
    /// Fourth control point (end).
    pub p3: Vec3,
}

impl BezierCurve {
    /// Create a new cubic Bezier curve.
    pub fn new(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> Self {
        Self { p0, p1, p2, p3 }
    }

    /// Create a linear Bezier (straight line from p0 to p3).
    pub fn linear(p0: Vec3, p3: Vec3) -> Self {
        let d = p3 - p0;
        Self {
            p0,
            p1: p0 + d * (1.0 / 3.0),
            p2: p0 + d * (2.0 / 3.0),
            p3,
        }
    }

    /// Evaluate the curve at parameter `t` in [0, 1] using De Casteljau's algorithm.
    ///
    /// De Casteljau's is numerically more stable than direct polynomial
    /// evaluation, especially for values of t near the endpoints.
    pub fn evaluate(&self, t: f32) -> Vec3 {
        let t = t.clamp(0.0, 1.0);
        let one_minus_t = 1.0 - t;

        // First level of interpolation.
        let a = self.p0 * one_minus_t + self.p1 * t;
        let b = self.p1 * one_minus_t + self.p2 * t;
        let c = self.p2 * one_minus_t + self.p3 * t;

        // Second level.
        let d = a * one_minus_t + b * t;
        let e = b * one_minus_t + c * t;

        // Final point.
        d * one_minus_t + e * t
    }

    /// Evaluate the tangent (first derivative) at parameter `t`.
    pub fn evaluate_tangent(&self, t: f32) -> Vec3 {
        let t = t.clamp(0.0, 1.0);
        let one_minus_t = 1.0 - t;

        // First derivative of a cubic Bezier:
        // B'(t) = 3(1-t)^2(P1-P0) + 6(1-t)t(P2-P1) + 3t^2(P3-P2)
        let a = (self.p1 - self.p0) * (3.0 * one_minus_t * one_minus_t);
        let b = (self.p2 - self.p1) * (6.0 * one_minus_t * t);
        let c = (self.p3 - self.p2) * (3.0 * t * t);

        a + b + c
    }

    /// Split the curve at parameter `t` into two sub-curves.
    ///
    /// Uses De Casteljau's algorithm to compute the split, which
    /// is exact (no approximation error).
    pub fn split(&self, t: f32) -> (BezierCurve, BezierCurve) {
        let t = t.clamp(0.0, 1.0);
        let one_minus_t = 1.0 - t;

        let a = self.p0 * one_minus_t + self.p1 * t;
        let b = self.p1 * one_minus_t + self.p2 * t;
        let c = self.p2 * one_minus_t + self.p3 * t;

        let d = a * one_minus_t + b * t;
        let e = b * one_minus_t + c * t;

        let f = d * one_minus_t + e * t;

        let left = BezierCurve::new(self.p0, a, d, f);
        let right = BezierCurve::new(f, e, c, self.p3);
        (left, right)
    }

    /// Approximate the arc length of the curve by sampling `n` segments.
    pub fn length(&self, samples: usize) -> f32 {
        let n = samples.max(1);
        let mut total = 0.0f32;
        let mut prev = self.evaluate(0.0);

        for i in 1..=n {
            let t = i as f32 / n as f32;
            let curr = self.evaluate(t);
            total += (curr - prev).length();
            prev = curr;
        }

        total
    }

    /// Find the point at a given arc length `s` along the curve.
    ///
    /// Uses binary search with arc-length approximation for constant-speed
    /// parameterization. This is essential for smooth camera paths and
    /// uniform particle distribution along curves.
    pub fn point_at_arc_length(&self, s: f32, samples: usize) -> Vec3 {
        let total_length = self.length(samples);
        if total_length < f32::EPSILON {
            return self.p0;
        }

        let target = s.clamp(0.0, total_length);

        // Build an arc-length table.
        let n = samples.max(10);
        let mut table = Vec::with_capacity(n + 1);
        let mut prev = self.evaluate(0.0);
        let mut cumulative = 0.0f32;
        table.push((0.0f32, 0.0f32)); // (t, arc_length)

        for i in 1..=n {
            let t = i as f32 / n as f32;
            let curr = self.evaluate(t);
            cumulative += (curr - prev).length();
            table.push((t, cumulative));
            prev = curr;
        }

        // Binary search for the parameter t at the target arc length.
        let mut lo = 0;
        let mut hi = table.len() - 1;
        while lo < hi - 1 {
            let mid = (lo + hi) / 2;
            if table[mid].1 < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        // Linear interpolation between the two table entries.
        let s0 = table[lo].1;
        let s1 = table[hi].1;
        let ds = s1 - s0;
        let t0 = table[lo].0;
        let t1 = table[hi].0;

        let t = if ds > f32::EPSILON {
            t0 + (target - s0) / ds * (t1 - t0)
        } else {
            t0
        };

        self.evaluate(t)
    }

    /// Get the bounding box of the curve (approximate, using control points).
    pub fn bounding_box(&self) -> (Vec3, Vec3) {
        let min = self.p0.min(self.p1).min(self.p2).min(self.p3);
        let max = self.p0.max(self.p1).max(self.p2).max(self.p3);
        (min, max)
    }
}

// ===========================================================================
// CatmullRomSpline
// ===========================================================================

/// A Catmull-Rom spline that smoothly passes through all control points.
///
/// Each segment is a cubic curve between two adjacent control points,
/// with tangents computed from the neighboring points. The spline is
/// C1-continuous (continuous first derivative) at all interior points.
///
/// This is the most common spline for camera paths, character movement
/// waypoints, and smooth interpolation through data points.
#[derive(Debug, Clone)]
pub struct CatmullRomSpline {
    /// Control points that the spline passes through.
    pub points: Vec<Vec3>,
    /// Tension parameter. 0.0 = Catmull-Rom, 1.0 = zero tangents.
    /// Default is 0.0.
    pub tension: f32,
    /// Whether the spline loops (last point connects to first).
    pub closed: bool,
}

impl CatmullRomSpline {
    /// Create a new open Catmull-Rom spline through the given points.
    pub fn new(points: Vec<Vec3>) -> Self {
        Self {
            points,
            tension: 0.0,
            closed: false,
        }
    }

    /// Create a closed (looping) Catmull-Rom spline.
    pub fn closed(points: Vec<Vec3>) -> Self {
        Self {
            points,
            tension: 0.0,
            closed: true,
        }
    }

    /// Set the tension parameter.
    pub fn with_tension(mut self, tension: f32) -> Self {
        self.tension = tension;
        self
    }

    /// Number of segments in the spline.
    pub fn segment_count(&self) -> usize {
        if self.points.len() < 2 {
            return 0;
        }
        if self.closed {
            self.points.len()
        } else {
            self.points.len() - 1
        }
    }

    /// Get the four control points for a given segment.
    fn segment_points(&self, segment: usize) -> (Vec3, Vec3, Vec3, Vec3) {
        let n = self.points.len();
        if self.closed {
            let p0 = self.points[(segment + n - 1) % n];
            let p1 = self.points[segment % n];
            let p2 = self.points[(segment + 1) % n];
            let p3 = self.points[(segment + 2) % n];
            (p0, p1, p2, p3)
        } else {
            let p1_idx = segment;
            let p2_idx = segment + 1;
            let p0 = if p1_idx > 0 {
                self.points[p1_idx - 1]
            } else {
                // Extrapolate.
                self.points[0] * 2.0 - self.points[1]
            };
            let p3 = if p2_idx + 1 < n {
                self.points[p2_idx + 1]
            } else {
                self.points[n - 1] * 2.0 - self.points[n - 2]
            };
            (p0, self.points[p1_idx], self.points[p2_idx], p3)
        }
    }

    /// Evaluate the spline at the given global parameter `t` in [0, 1].
    ///
    /// `t = 0` is the first point, `t = 1` is the last point (or wraps
    /// back to the first for closed splines).
    pub fn evaluate(&self, t: f32) -> Vec3 {
        let n = self.segment_count();
        if n == 0 {
            return if self.points.is_empty() {
                Vec3::ZERO
            } else {
                self.points[0]
            };
        }

        let t = if self.closed {
            t.rem_euclid(1.0)
        } else {
            t.clamp(0.0, 1.0)
        };

        let scaled = t * n as f32;
        let segment = (scaled.floor() as usize).min(n - 1);
        let local_t = scaled - segment as f32;

        self.evaluate_segment(segment, local_t)
    }

    /// Evaluate a specific segment at local parameter `t` in [0, 1].
    fn evaluate_segment(&self, segment: usize, t: f32) -> Vec3 {
        let (p0, p1, p2, p3) = self.segment_points(segment);
        let s = 0.5 * (1.0 - self.tension);

        let t2 = t * t;
        let t3 = t2 * t;

        // Catmull-Rom matrix multiplication.
        // The standard matrix form below is clearer and avoids intermediates.

        // Standard Catmull-Rom formula:
        // q(t) = 0.5 * [(2*P1) + (-P0+P2)*t + (2*P0-5*P1+4*P2-P3)*t^2 + (-P0+3*P1-3*P2+P3)*t^3]
        let result = (p1 * 2.0
            + (p2 - p0) * t
            + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * t2
            + (p1 * 3.0 - p0 - p2 * 3.0 + p3) * t3)
            * s;

        // Adjust for tension: when tension != 0, blend toward linear.
        if (self.tension).abs() > f32::EPSILON {
            let linear = p1 + (p2 - p1) * t;
            result * (1.0 - self.tension) + linear * self.tension
        } else {
            result
        }
    }

    /// Evaluate the tangent at the given global parameter `t`.
    pub fn evaluate_tangent(&self, t: f32) -> Vec3 {
        let n = self.segment_count();
        if n == 0 {
            return Vec3::ZERO;
        }

        let t = if self.closed {
            t.rem_euclid(1.0)
        } else {
            t.clamp(0.0, 1.0)
        };

        let scaled = t * n as f32;
        let segment = (scaled.floor() as usize).min(n - 1);
        let local_t = scaled - segment as f32;

        self.tangent_segment(segment, local_t)
    }

    /// Evaluate the tangent for a specific segment at local t.
    fn tangent_segment(&self, segment: usize, t: f32) -> Vec3 {
        let (p0, p1, p2, p3) = self.segment_points(segment);
        let s = 0.5 * (1.0 - self.tension);

        let t2 = t * t;

        // Derivative of the Catmull-Rom formula:
        // q'(t) = 0.5 * [(-P0+P2) + 2*(2*P0-5*P1+4*P2-P3)*t + 3*(-P0+3*P1-3*P2+P3)*t^2]
        let result = ((p2 - p0)
            + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * (2.0 * t)
            + (p1 * 3.0 - p0 - p2 * 3.0 + p3) * (3.0 * t2))
            * s;

        result
    }

    /// Approximate the total arc length of the spline.
    pub fn length(&self, samples_per_segment: usize) -> f32 {
        let n = self.segment_count();
        if n == 0 {
            return 0.0;
        }

        let total_samples = n * samples_per_segment.max(1);
        let mut total = 0.0f32;
        let mut prev = self.evaluate(0.0);

        for i in 1..=total_samples {
            let t = i as f32 / total_samples as f32;
            let curr = self.evaluate(t);
            total += (curr - prev).length();
            prev = curr;
        }

        total
    }

    /// Find the point at a given arc length along the spline.
    pub fn point_at_arc_length(&self, s: f32, samples_per_segment: usize) -> Vec3 {
        let n = self.segment_count();
        if n == 0 {
            return if self.points.is_empty() { Vec3::ZERO } else { self.points[0] };
        }

        let total_samples = n * samples_per_segment.max(10);
        let mut table = Vec::with_capacity(total_samples + 1);
        let mut prev = self.evaluate(0.0);
        let mut cumulative = 0.0f32;
        table.push((0.0f32, 0.0f32));

        for i in 1..=total_samples {
            let t = i as f32 / total_samples as f32;
            let curr = self.evaluate(t);
            cumulative += (curr - prev).length();
            table.push((t, cumulative));
            prev = curr;
        }

        let total_length = cumulative;
        let target = s.clamp(0.0, total_length);

        let mut lo = 0;
        let mut hi = table.len() - 1;
        while lo < hi - 1 {
            let mid = (lo + hi) / 2;
            if table[mid].1 < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let s0 = table[lo].1;
        let s1 = table[hi].1;
        let t0 = table[lo].0;
        let t1 = table[hi].0;
        let ds = s1 - s0;

        let t = if ds > f32::EPSILON {
            t0 + (target - s0) / ds * (t1 - t0)
        } else {
            t0
        };

        self.evaluate(t)
    }
}

// ===========================================================================
// BSpline
// ===========================================================================

/// A B-spline curve with an arbitrary knot vector.
///
/// B-splines provide local control: moving a single control point only
/// affects a bounded region of the curve. The curve is C^(p-1) continuous
/// where p is the degree.
///
/// Evaluated using De Boor's algorithm, which is the B-spline analog
/// of De Casteljau's algorithm for Bezier curves.
#[derive(Debug, Clone)]
pub struct BSpline {
    /// Control points.
    pub control_points: Vec<Vec3>,
    /// Knot vector. Length must be `control_points.len() + degree + 1`.
    pub knots: Vec<f32>,
    /// Degree of the spline (typically 3 for cubic).
    pub degree: usize,
}

impl BSpline {
    /// Create a new B-spline with the given control points, knots, and degree.
    ///
    /// # Panics
    ///
    /// Panics if the knot vector length doesn't match `points.len() + degree + 1`.
    pub fn new(control_points: Vec<Vec3>, knots: Vec<f32>, degree: usize) -> Self {
        assert_eq!(
            knots.len(),
            control_points.len() + degree + 1,
            "Knot vector length ({}) must be control_points ({}) + degree ({}) + 1",
            knots.len(),
            control_points.len(),
            degree
        );
        Self {
            control_points,
            knots,
            degree,
        }
    }

    /// Create a uniform B-spline with automatically generated knot vector.
    ///
    /// The knot vector is clamped (first and last `degree+1` knots are
    /// repeated) so the curve starts at the first point and ends at the
    /// last point.
    pub fn uniform(control_points: Vec<Vec3>, degree: usize) -> Self {
        let n = control_points.len();
        let m = n + degree + 1;
        let mut knots = Vec::with_capacity(m);

        // Clamped uniform knot vector.
        for _ in 0..=degree {
            knots.push(0.0);
        }
        let interior_knots = n - degree; // n - p = number of internal knot spans
        if interior_knots > 1 {
            for i in 1..interior_knots {
                knots.push(i as f32 / interior_knots as f32);
            }
        }
        for _ in 0..=degree {
            knots.push(1.0);
        }

        // Pad if necessary (can happen for small numbers of control points).
        while knots.len() < m {
            knots.insert(knots.len() - degree - 1, 0.5);
        }
        knots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Self {
            control_points,
            knots,
            degree,
        }
    }

    /// Evaluate the B-spline at parameter `t` using De Boor's algorithm.
    ///
    /// `t` should be in the range `[knots[degree], knots[n]]` where `n`
    /// is the number of control points.
    pub fn evaluate(&self, t: f32) -> Vec3 {
        let n = self.control_points.len();
        let p = self.degree;

        if n == 0 {
            return Vec3::ZERO;
        }
        if n == 1 {
            return self.control_points[0];
        }

        // Clamp t to the valid range.
        let t_min = self.knots[p];
        let t_max = self.knots[n];
        let t = t.clamp(t_min, t_max);

        // Handle exact endpoint.
        if (t - t_max).abs() < f32::EPSILON {
            return *self.control_points.last().unwrap();
        }

        // Find the knot span index k such that knots[k] <= t < knots[k+1].
        let k = self.find_knot_span(t);

        // De Boor's algorithm: triangular computation.
        // We need control points d[k-p], d[k-p+1], ..., d[k].
        let mut d: Vec<Vec3> = (0..=p)
            .map(|j| {
                let idx = k.wrapping_sub(p).wrapping_add(j);
                if idx < n {
                    self.control_points[idx]
                } else {
                    Vec3::ZERO
                }
            })
            .collect();

        for r in 1..=p {
            for j in (r..=p).rev() {
                let i = k.wrapping_sub(p).wrapping_add(j);
                let knot_i_plus_p_minus_r_plus_1 = if i + p + 1 - r < self.knots.len() {
                    self.knots[i + p + 1 - r]
                } else {
                    1.0
                };
                let knot_i = if i < self.knots.len() {
                    self.knots[i]
                } else {
                    0.0
                };

                let denom = knot_i_plus_p_minus_r_plus_1 - knot_i;
                let alpha = if denom.abs() > f32::EPSILON {
                    (t - knot_i) / denom
                } else {
                    0.0
                };

                d[j] = d[j - 1] * (1.0 - alpha) + d[j] * alpha;
            }
        }

        d[p]
    }

    /// Find the knot span index for the given parameter value.
    fn find_knot_span(&self, t: f32) -> usize {
        let n = self.control_points.len();
        let p = self.degree;

        // Special case: t at the end.
        if t >= self.knots[n] {
            return n - 1;
        }

        // Binary search for the span.
        let mut low = p;
        let mut high = n;
        while low < high {
            let mid = (low + high) / 2;
            if t < self.knots[mid] {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        low - 1
    }

    /// Evaluate the tangent at parameter `t` via finite differences.
    pub fn evaluate_tangent(&self, t: f32) -> Vec3 {
        let eps = 0.0001;
        let p1 = self.evaluate(t - eps);
        let p2 = self.evaluate(t + eps);
        (p2 - p1) / (2.0 * eps)
    }

    /// Insert a knot at the given parameter value (knot insertion).
    ///
    /// Returns a new B-spline with one additional knot and one additional
    /// control point, representing the same curve.
    pub fn insert_knot(&self, t: f32) -> Self {
        let n = self.control_points.len();
        let p = self.degree;
        let k = self.find_knot_span(t);

        // New knot vector: insert t.
        let mut new_knots = self.knots.clone();
        new_knots.insert(k + 1, t);

        // Compute new control points.
        let mut new_points = Vec::with_capacity(n + 1);

        for i in 0..=n {
            if i <= k.saturating_sub(p) {
                new_points.push(self.control_points[i]);
            } else if i > k {
                if i - 1 < n {
                    new_points.push(self.control_points[i - 1]);
                }
            } else {
                let knot_i = self.knots[i];
                let knot_i_plus_p = if i + p < self.knots.len() {
                    self.knots[i + p]
                } else {
                    1.0
                };
                let denom = knot_i_plus_p - knot_i;
                let alpha = if denom.abs() > f32::EPSILON {
                    (t - knot_i) / denom
                } else {
                    0.0
                };

                let pi_minus_1 = if i > 0 && i - 1 < n {
                    self.control_points[i - 1]
                } else {
                    Vec3::ZERO
                };
                let pi = if i < n {
                    self.control_points[i]
                } else {
                    Vec3::ZERO
                };

                new_points.push(pi_minus_1 * (1.0 - alpha) + pi * alpha);
            }
        }

        // Ensure correct length.
        while new_points.len() > n + 1 {
            new_points.pop();
        }
        while new_points.len() < n + 1 {
            new_points.push(*self.control_points.last().unwrap_or(&Vec3::ZERO));
        }

        Self::new(new_points, new_knots, p)
    }

    /// Approximate arc length.
    pub fn length(&self, samples: usize) -> f32 {
        let n = samples.max(1);
        let t_min = self.knots[self.degree];
        let t_max = self.knots[self.control_points.len()];
        let mut total = 0.0f32;
        let mut prev = self.evaluate(t_min);

        for i in 1..=n {
            let t = t_min + (t_max - t_min) * i as f32 / n as f32;
            let curr = self.evaluate(t);
            total += (curr - prev).length();
            prev = curr;
        }

        total
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- AnimationCurve tests --

    #[test]
    fn curve_empty() {
        let curve = AnimationCurve::new();
        assert_eq!(curve.evaluate(0.5), 0.0);
    }

    #[test]
    fn curve_single_key() {
        let mut curve = AnimationCurve::new();
        curve.add_key_simple(0.0, 5.0);
        assert!((curve.evaluate(0.0) - 5.0).abs() < f32::EPSILON);
        assert!((curve.evaluate(1.0) - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn curve_linear() {
        let curve = AnimationCurve::linear();
        assert!((curve.evaluate(0.0) - 0.0).abs() < 0.01);
        assert!((curve.evaluate(1.0) - 1.0).abs() < 0.01);
        assert!((curve.evaluate(0.5) - 0.5).abs() < 0.1);
    }

    #[test]
    fn curve_ease_in() {
        let curve = AnimationCurve::ease_in();
        let at_quarter = curve.evaluate(0.25);
        // Ease-in should be below the linear value at 0.25.
        assert!(at_quarter < 0.3, "Ease-in at 0.25 should be < 0.3, got {at_quarter}");
    }

    #[test]
    fn curve_ease_out() {
        let curve = AnimationCurve::ease_out();
        let at_quarter = curve.evaluate(0.25);
        // Ease-out should be above the linear value at 0.25.
        assert!(at_quarter > 0.2, "Ease-out at 0.25 should be > 0.2, got {at_quarter}");
    }

    #[test]
    fn curve_ease_in_out_endpoints() {
        let curve = AnimationCurve::ease_in_out();
        assert!((curve.evaluate(0.0) - 0.0).abs() < 0.01);
        assert!((curve.evaluate(1.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn curve_constant() {
        let curve = AnimationCurve::constant(7.5);
        assert!((curve.evaluate(0.0) - 7.5).abs() < f32::EPSILON);
        assert!((curve.evaluate(0.5) - 7.5).abs() < f32::EPSILON);
        assert!((curve.evaluate(1.0) - 7.5).abs() < f32::EPSILON);
    }

    #[test]
    fn curve_loop_mode() {
        let curve = AnimationCurve::from_keys(
            vec![
                CurveKeyframe::with_tangents(0.0, 0.0, 1.0, 1.0),
                CurveKeyframe::with_tangents(1.0, 1.0, 1.0, 1.0),
            ],
            CurveMode::Loop,
        );
        let at_1_5 = curve.evaluate(1.5);
        let at_0_5 = curve.evaluate(0.5);
        assert!((at_1_5 - at_0_5).abs() < 0.1, "Looped value at 1.5 should equal 0.5");
    }

    #[test]
    fn curve_clamp_mode() {
        let curve = AnimationCurve::from_keys(
            vec![
                CurveKeyframe::new(0.0, 0.0),
                CurveKeyframe::new(1.0, 1.0),
            ],
            CurveMode::Clamp,
        );
        assert!((curve.evaluate(-1.0) - 0.0).abs() < f32::EPSILON);
        assert!((curve.evaluate(2.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn curve_remove_key() {
        let mut curve = AnimationCurve::new();
        curve.add_key_simple(0.0, 0.0);
        curve.add_key_simple(0.5, 0.5);
        curve.add_key_simple(1.0, 1.0);
        assert_eq!(curve.key_count(), 3);
        curve.remove_key(1);
        assert_eq!(curve.key_count(), 2);
    }

    #[test]
    fn curve_auto_tangents() {
        let mut curve = AnimationCurve::new();
        curve.add_key_simple(0.0, 0.0);
        curve.add_key_simple(0.5, 1.0);
        curve.add_key_simple(1.0, 0.0);
        curve.auto_tangents();
        // Middle key should have zero tangent (peak).
        assert!((curve.key(1).unwrap().in_tangent - 0.0).abs() < f32::EPSILON);
    }

    // -- BezierCurve tests --

    #[test]
    fn bezier_endpoints() {
        let curve = BezierCurve::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        );
        let start = curve.evaluate(0.0);
        let end = curve.evaluate(1.0);
        assert!((start - Vec3::ZERO).length() < f32::EPSILON);
        assert!((end - Vec3::new(3.0, 0.0, 0.0)).length() < f32::EPSILON);
    }

    #[test]
    fn bezier_linear() {
        let curve = BezierCurve::linear(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0));
        let mid = curve.evaluate(0.5);
        assert!((mid.x - 5.0).abs() < 0.01);
    }

    #[test]
    fn bezier_split() {
        let curve = BezierCurve::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        );
        let (left, right) = curve.split(0.5);
        let left_end = left.evaluate(1.0);
        let right_start = right.evaluate(0.0);
        assert!((left_end - right_start).length() < 0.001, "Split point should match");
        let original_mid = curve.evaluate(0.5);
        assert!((left_end - original_mid).length() < 0.001);
    }

    #[test]
    fn bezier_arc_length() {
        let curve = BezierCurve::linear(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0));
        let len = curve.length(100);
        assert!((len - 10.0).abs() < 0.1, "Linear bezier should have length 10, got {len}");
    }

    #[test]
    fn bezier_point_at_arc_length() {
        let curve = BezierCurve::linear(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0));
        let point = curve.point_at_arc_length(5.0, 100);
        assert!((point.x - 5.0).abs() < 0.2);
    }

    #[test]
    fn bezier_tangent() {
        let curve = BezierCurve::linear(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0));
        let tan = curve.evaluate_tangent(0.5);
        assert!(tan.x > 0.0, "Tangent should point in +X direction");
        assert!(tan.y.abs() < 0.01);
    }

    // -- CatmullRomSpline tests --

    #[test]
    fn catmull_rom_endpoints() {
        let spline = CatmullRomSpline::new(vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 1.0, 0.0),
        ]);
        let start = spline.evaluate(0.0);
        let end = spline.evaluate(1.0);
        assert!((start - Vec3::ZERO).length() < 0.01);
        assert!((end - Vec3::new(3.0, 1.0, 0.0)).length() < 0.01);
    }

    #[test]
    fn catmull_rom_passes_through_points() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(3.0, 1.0, 0.0),
            Vec3::new(4.0, 3.0, 0.0),
        ];
        let spline = CatmullRomSpline::new(points.clone());
        // The spline should pass through each internal point.
        for i in 0..points.len() {
            let t = i as f32 / (points.len() - 1) as f32;
            let p = spline.evaluate(t);
            let expected = points[i];
            assert!(
                (p - expected).length() < 0.5,
                "Spline at t={t} should be near point {i}: got {p:?}, expected {expected:?}"
            );
        }
    }

    #[test]
    fn catmull_rom_closed() {
        let spline = CatmullRomSpline::closed(vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ]);
        // At t=0 and t=1 the closed spline should be at the same point.
        let start = spline.evaluate(0.0);
        let end = spline.evaluate(0.999);
        assert!((start - end).length() < 0.5);
    }

    #[test]
    fn catmull_rom_tangent() {
        let spline = CatmullRomSpline::new(vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ]);
        let tang = spline.evaluate_tangent(0.5);
        assert!(tang.x > 0.0, "Tangent should point in +X");
    }

    #[test]
    fn catmull_rom_length() {
        let spline = CatmullRomSpline::new(vec![
            Vec3::ZERO,
            Vec3::new(10.0, 0.0, 0.0),
        ]);
        let len = spline.length(100);
        assert!((len - 10.0).abs() < 1.0, "Length should be ~10, got {len}");
    }

    // -- BSpline tests --

    #[test]
    fn bspline_uniform_endpoints() {
        let spline = BSpline::uniform(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(3.0, 1.0, 0.0),
            ],
            3,
        );
        let start = spline.evaluate(0.0);
        let end = spline.evaluate(1.0);
        assert!((start - Vec3::ZERO).length() < 0.1, "Start should be near P0: {start:?}");
        assert!((end - Vec3::new(3.0, 1.0, 0.0)).length() < 0.1, "End should be near P3: {end:?}");
    }

    #[test]
    fn bspline_single_point() {
        let spline = BSpline::uniform(vec![Vec3::new(5.0, 5.0, 5.0)], 0);
        let p = spline.evaluate(0.0);
        assert!((p - Vec3::new(5.0, 5.0, 5.0)).length() < f32::EPSILON);
    }

    #[test]
    fn bspline_tangent() {
        let spline = BSpline::uniform(
            vec![
                Vec3::ZERO,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(3.0, 0.0, 0.0),
            ],
            3,
        );
        let tang = spline.evaluate_tangent(0.5);
        assert!(tang.length() > 0.0, "Tangent should be non-zero");
    }

    #[test]
    fn bspline_length() {
        let spline = BSpline::uniform(
            vec![
                Vec3::ZERO,
                Vec3::new(5.0, 0.0, 0.0),
                Vec3::new(10.0, 0.0, 0.0),
                Vec3::new(15.0, 0.0, 0.0),
            ],
            3,
        );
        let len = spline.length(200);
        assert!(len > 5.0 && len < 20.0, "Length should be reasonable, got {len}");
    }

    #[test]
    fn bspline_knot_insertion() {
        let spline = BSpline::uniform(
            vec![
                Vec3::ZERO,
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(3.0, 1.0, 0.0),
            ],
            3,
        );
        let refined = spline.insert_knot(0.5);
        // The refined spline should have one more control point.
        assert_eq!(refined.control_points.len(), spline.control_points.len() + 1);
        // And one more knot.
        assert_eq!(refined.knots.len(), spline.knots.len() + 1);
        // The curves should be equivalent.
        let p_orig = spline.evaluate(0.5);
        let p_refined = refined.evaluate(0.5);
        assert!(
            (p_orig - p_refined).length() < 0.5,
            "Knot insertion should preserve the curve: {p_orig:?} vs {p_refined:?}"
        );
    }

    #[test]
    fn curve_bell() {
        let curve = AnimationCurve::bell();
        let at_0 = curve.evaluate(0.0);
        let at_half = curve.evaluate(0.5);
        let at_1 = curve.evaluate(1.0);
        assert!((at_0).abs() < 0.01);
        assert!((at_half - 1.0).abs() < 0.1);
        assert!((at_1).abs() < 0.01);
    }
}
