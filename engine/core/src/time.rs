//! Time management utilities.
//!
//! Provides a [`Clock`] that tracks per-frame timing, a [`Timer`] for
//! one-shot countdowns, a [`Stopwatch`] for elapsed-time measurement, and
//! a fixed-timestep accumulator for deterministic physics updates.

use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Clock
// ---------------------------------------------------------------------------

/// Central time source for the engine.
///
/// Call [`Clock::tick`] exactly once per frame (at the top of the game loop).
/// All downstream systems read timing information from the clock rather than
/// querying the OS directly, which keeps behaviour deterministic and
/// makes time manipulation (pause, slow-mo) trivial.
pub struct Clock {
    /// Wall-clock time at engine start.
    start_instant: Instant,
    /// Wall-clock time at the previous [`tick`](Clock::tick) call.
    last_instant: Instant,
    /// Duration of the most recent frame.
    delta: Duration,
    /// Accumulated wall-clock time since engine start (respects time scale).
    total: Duration,
    /// Monotonically increasing frame counter.
    frame_count: u64,
    /// Multiplier applied to raw delta before it is stored.
    /// `1.0` = real-time, `0.0` = paused, `0.5` = half-speed, etc.
    time_scale: f64,
    /// Upper bound on `delta` to prevent spiral-of-death when the application
    /// hitches.
    max_delta: Duration,
    /// Accumulator for fixed-timestep updates.
    fixed_accumulator: Duration,
    /// The desired fixed timestep interval.
    fixed_timestep: Duration,
}

impl Clock {
    /// Creates a new clock.
    ///
    /// `fixed_timestep` sets the interval used by [`should_run_fixed_update`]
    /// (e.g., `Duration::from_secs_f64(1.0 / 60.0)` for 60 Hz physics).
    pub fn new(fixed_timestep: Duration) -> Self {
        let now = Instant::now();
        Self {
            start_instant: now,
            last_instant: now,
            delta: Duration::ZERO,
            total: Duration::ZERO,
            frame_count: 0,
            time_scale: 1.0,
            max_delta: Duration::from_millis(250),
            fixed_accumulator: Duration::ZERO,
            fixed_timestep,
        }
    }

    /// Advances the clock by one frame. Call once at the top of the game loop.
    pub fn tick(&mut self) {
        let now = Instant::now();
        let mut raw_delta = now - self.last_instant;
        self.last_instant = now;

        // Clamp to avoid spiral-of-death.
        if raw_delta > self.max_delta {
            raw_delta = self.max_delta;
        }

        // Apply time scale.
        self.delta = raw_delta.mul_f64(self.time_scale);
        self.total += self.delta;
        self.frame_count += 1;

        // Feed the fixed-timestep accumulator.
        self.fixed_accumulator += self.delta;
    }

    /// Returns `true` and consumes one quantum from the accumulator each time
    /// a fixed update should run. Typical usage:
    ///
    /// ```ignore
    /// while clock.should_run_fixed_update() {
    ///     physics.step(clock.fixed_timestep());
    /// }
    /// ```
    pub fn should_run_fixed_update(&mut self) -> bool {
        if self.fixed_accumulator >= self.fixed_timestep {
            self.fixed_accumulator -= self.fixed_timestep;
            true
        } else {
            false
        }
    }

    /// Duration of the most recent frame (scaled).
    #[inline]
    pub fn delta(&self) -> Duration {
        self.delta
    }

    /// `delta` as `f32` seconds — the most common form used by gameplay code.
    #[inline]
    pub fn delta_secs(&self) -> f32 {
        self.delta.as_secs_f32()
    }

    /// Total elapsed time since engine start (scaled).
    #[inline]
    pub fn total(&self) -> Duration {
        self.total
    }

    /// Total elapsed time as `f64` seconds.
    #[inline]
    pub fn total_secs_f64(&self) -> f64 {
        self.total.as_secs_f64()
    }

    /// Number of frames elapsed.
    #[inline]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Returns the configured fixed timestep interval.
    #[inline]
    pub fn fixed_timestep(&self) -> Duration {
        self.fixed_timestep
    }

    /// Current time scale multiplier.
    #[inline]
    pub fn time_scale(&self) -> f64 {
        self.time_scale
    }

    /// Sets the time scale. Clamped to `[0.0, 100.0]`.
    pub fn set_time_scale(&mut self, scale: f64) {
        self.time_scale = scale.clamp(0.0, 100.0);
    }

    /// Interpolation alpha for rendering between fixed updates.
    ///
    /// Returns a value in `[0.0, 1.0)` representing how far into the next
    /// fixed tick the current frame is. Use this to interpolate visual state
    /// between the last two physics snapshots.
    #[inline]
    pub fn fixed_interpolation_alpha(&self) -> f64 {
        self.fixed_accumulator.as_secs_f64() / self.fixed_timestep.as_secs_f64()
    }
}

impl Default for Clock {
    /// Defaults to a 60 Hz fixed timestep.
    fn default() -> Self {
        Self::new(Duration::from_secs_f64(1.0 / 60.0))
    }
}

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------

/// A countdown timer.
///
/// Counts down from a configured duration. Once elapsed, [`is_finished`]
/// returns `true`. Optionally repeats.
pub struct Timer {
    /// Total duration of the timer.
    duration: Duration,
    /// Time remaining.
    remaining: Duration,
    /// Whether the timer restarts automatically after finishing.
    repeating: bool,
    /// `true` once the timer has reached zero at least once since the last
    /// reset.
    finished: bool,
}

impl Timer {
    /// Creates a one-shot timer with the given duration.
    pub fn once(duration: Duration) -> Self {
        Self {
            duration,
            remaining: duration,
            repeating: false,
            finished: false,
        }
    }

    /// Creates a repeating timer with the given duration.
    pub fn repeating(duration: Duration) -> Self {
        Self {
            duration,
            remaining: duration,
            repeating: true,
            finished: false,
        }
    }

    /// Advances the timer by `delta`. Call once per frame.
    pub fn tick(&mut self, delta: Duration) {
        // For very short timers a single delta may span multiple full
        // durations. The current implementation handles one wrap-around per
        // tick via the overshoot calculation below, which is sufficient for
        // game timers where dt << duration. Multi-expiration counting can be
        // added if sub-millisecond repeating timers are required.
        if self.finished && !self.repeating {
            return;
        }

        if let Some(new_remaining) = self.remaining.checked_sub(delta) {
            self.remaining = new_remaining;
        } else {
            self.finished = true;
            if self.repeating {
                // Wrap around, preserving leftover time.
                let overshoot = delta - self.remaining;
                self.remaining = self.duration.saturating_sub(overshoot);
            } else {
                self.remaining = Duration::ZERO;
            }
        }
    }

    /// Returns `true` if the timer has finished (at least once).
    #[inline]
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Fraction of the duration that has elapsed, in `[0.0, 1.0]`.
    #[inline]
    pub fn fraction_elapsed(&self) -> f32 {
        if self.duration.is_zero() {
            return 1.0;
        }
        1.0 - (self.remaining.as_secs_f32() / self.duration.as_secs_f32())
    }

    /// Resets the timer to its full duration.
    pub fn reset(&mut self) {
        self.remaining = self.duration;
        self.finished = false;
    }
}

// ---------------------------------------------------------------------------
// Stopwatch
// ---------------------------------------------------------------------------

/// A simple elapsed-time stopwatch.
///
/// Unlike [`Timer`], a stopwatch counts *up* with no target and never finishes.
pub struct Stopwatch {
    /// Accumulated elapsed time.
    elapsed: Duration,
    /// Whether the stopwatch is currently running.
    running: bool,
}

impl Stopwatch {
    /// Creates a stopped stopwatch at zero.
    pub fn new() -> Self {
        Self {
            elapsed: Duration::ZERO,
            running: false,
        }
    }

    /// Starts (or resumes) the stopwatch.
    pub fn start(&mut self) {
        self.running = true;
    }

    /// Pauses the stopwatch without resetting elapsed time.
    pub fn pause(&mut self) {
        self.running = false;
    }

    /// Resets elapsed time to zero and stops the stopwatch.
    pub fn reset(&mut self) {
        self.elapsed = Duration::ZERO;
        self.running = false;
    }

    /// Advances the stopwatch by `delta` if it is running.
    pub fn tick(&mut self, delta: Duration) {
        if self.running {
            self.elapsed += delta;
        }
    }

    /// Total elapsed time while running.
    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// Elapsed time as `f32` seconds.
    #[inline]
    pub fn elapsed_secs(&self) -> f32 {
        self.elapsed.as_secs_f32()
    }

    /// Whether the stopwatch is currently running.
    #[inline]
    pub fn is_running(&self) -> bool {
        self.running
    }
}

impl Default for Stopwatch {
    fn default() -> Self {
        Self::new()
    }
}
