// engine/core/src/engine_loop.rs
//
// Main engine loop for the Genovo engine.
//
// Implements fixed-timestep physics with variable-rate rendering, frame
// pacing, vsync support, frame time smoothing, and spiral-of-death protection.

use std::collections::VecDeque;

pub const DEFAULT_FIXED_DT: f32 = 1.0 / 60.0;
pub const DEFAULT_MAX_DT: f32 = 0.25;
pub const DEFAULT_TARGET_FPS: u32 = 60;
pub const DEFAULT_SMOOTH_WINDOW: usize = 11;
pub const MAX_FIXED_STEPS_PER_FRAME: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum VsyncMode { Off, On, Adaptive, TripleBuffer }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum FramePacingMode { None, FrameLimiter, SleepAndSpin }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum LoopState { Running, Paused, Stopping, Stopped }

#[derive(Debug, Clone)]
pub struct EngineLoopConfig {
    pub fixed_timestep: f32,
    pub max_delta_time: f32,
    pub target_fps: u32,
    pub vsync: VsyncMode,
    pub frame_pacing: FramePacingMode,
    pub smooth_window_size: usize,
    pub max_fixed_steps: u32,
    pub enable_interpolation: bool,
    pub panic_on_spiral: bool,
}

impl Default for EngineLoopConfig {
    fn default() -> Self {
        Self { fixed_timestep: DEFAULT_FIXED_DT, max_delta_time: DEFAULT_MAX_DT, target_fps: DEFAULT_TARGET_FPS,
            vsync: VsyncMode::On, frame_pacing: FramePacingMode::None, smooth_window_size: DEFAULT_SMOOTH_WINDOW,
            max_fixed_steps: MAX_FIXED_STEPS_PER_FRAME, enable_interpolation: true, panic_on_spiral: false }
    }
}

#[derive(Debug, Clone, Default)]
pub struct FrameStats {
    pub frame_number: u64,
    pub raw_dt: f32,
    pub smoothed_dt: f32,
    pub fixed_steps_this_frame: u32,
    pub interpolation_alpha: f32,
    pub fps: f32,
    pub avg_fps: f32,
    pub frame_time_ms: f32,
    pub fixed_time_ms: f32,
    pub render_time_ms: f32,
    pub idle_time_ms: f32,
    pub spiral_of_death: bool,
    pub total_fixed_ticks: u64,
    pub time_since_start: f64,
}

#[derive(Debug, Clone)]
pub struct FrameTimeSmoother {
    samples: VecDeque<f32>,
    window_size: usize,
}

impl FrameTimeSmoother {
    pub fn new(window_size: usize) -> Self { Self { samples: VecDeque::with_capacity(window_size + 1), window_size } }

    pub fn add_sample(&mut self, dt: f32) {
        self.samples.push_back(dt);
        while self.samples.len() > self.window_size { self.samples.pop_front(); }
    }

    pub fn smoothed(&self) -> f32 {
        if self.samples.is_empty() { return DEFAULT_FIXED_DT; }
        // Median filter: sort samples, take median.
        let mut sorted: Vec<f32> = self.samples.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    }

    pub fn average(&self) -> f32 {
        if self.samples.is_empty() { return DEFAULT_FIXED_DT; }
        self.samples.iter().sum::<f32>() / self.samples.len() as f32
    }

    pub fn min(&self) -> f32 { self.samples.iter().copied().fold(f32::MAX, f32::min) }
    pub fn max(&self) -> f32 { self.samples.iter().copied().fold(0.0f32, f32::max) }
    pub fn jitter(&self) -> f32 { self.max() - self.min() }
}

/// Phases of the engine loop, called in order each frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnginePhase { Input, FixedUpdate, Update, LateUpdate, PreRender, Render, PostRender, FrameEnd }

/// Callback signature for engine phases.
pub type PhaseCallback = Box<dyn FnMut(f32) + Send>;

/// The main engine loop driver.
#[derive(Debug)]
pub struct EngineLoop {
    pub config: EngineLoopConfig,
    pub state: LoopState,
    pub stats: FrameStats,
    smoother: FrameTimeSmoother,
    accumulator: f32,
    total_time: f64,
    frame_count: u64,
    fps_counter: FpsCounter,
    fixed_tick_count: u64,
}

#[derive(Debug, Clone)]
struct FpsCounter {
    frame_times: VecDeque<f32>,
    timer: f32,
    current_fps: f32,
    avg_fps: f32,
}

impl FpsCounter {
    fn new() -> Self { Self { frame_times: VecDeque::new(), timer: 0.0, current_fps: 0.0, avg_fps: 0.0 } }
    fn update(&mut self, dt: f32) {
        self.frame_times.push_back(dt);
        self.timer += dt;
        while self.timer > 1.0 { if let Some(old) = self.frame_times.pop_front() { self.timer -= old; } else { break; } }
        self.current_fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
        self.avg_fps = if self.timer > 0.0 { self.frame_times.len() as f32 / self.timer } else { 0.0 };
    }
}

impl EngineLoop {
    pub fn new(config: EngineLoopConfig) -> Self {
        let smoother = FrameTimeSmoother::new(config.smooth_window_size);
        Self { config, state: LoopState::Running, stats: FrameStats::default(), smoother, accumulator: 0.0, total_time: 0.0, frame_count: 0, fps_counter: FpsCounter::new(), fixed_tick_count: 0 }
    }

    /// Called each frame with the raw delta time from the platform timer.
    /// Returns a FrameTick describing what to do this frame.
    pub fn tick(&mut self, raw_dt: f32) -> FrameTick {
        if self.state != LoopState::Running { return FrameTick::default(); }

        // Clamp raw dt to prevent spiral of death.
        let clamped_dt = raw_dt.min(self.config.max_delta_time);
        self.smoother.add_sample(clamped_dt);
        let smoothed_dt = self.smoother.smoothed();
        self.fps_counter.update(smoothed_dt);

        self.frame_count += 1;
        self.total_time += smoothed_dt as f64;

        // Fixed timestep accumulation.
        self.accumulator += smoothed_dt;
        let fixed_dt = self.config.fixed_timestep;
        let mut fixed_steps = 0u32;

        while self.accumulator >= fixed_dt && fixed_steps < self.config.max_fixed_steps {
            self.accumulator -= fixed_dt;
            fixed_steps += 1;
            self.fixed_tick_count += 1;
        }

        let spiral = self.accumulator >= fixed_dt * 2.0;
        if spiral && self.config.panic_on_spiral {
            // Reset accumulator to prevent spiral of death.
            self.accumulator = 0.0;
        }

        let alpha = if self.config.enable_interpolation { self.accumulator / fixed_dt } else { 1.0 };

        self.stats = FrameStats {
            frame_number: self.frame_count,
            raw_dt,
            smoothed_dt,
            fixed_steps_this_frame: fixed_steps,
            interpolation_alpha: alpha,
            fps: self.fps_counter.current_fps,
            avg_fps: self.fps_counter.avg_fps,
            frame_time_ms: smoothed_dt * 1000.0,
            fixed_time_ms: fixed_steps as f32 * fixed_dt * 1000.0,
            render_time_ms: 0.0,
            idle_time_ms: 0.0,
            spiral_of_death: spiral,
            total_fixed_ticks: self.fixed_tick_count,
            time_since_start: self.total_time,
        };

        FrameTick { dt: smoothed_dt, fixed_dt, fixed_steps, interpolation_alpha: alpha, frame_number: self.frame_count, spiral_of_death: spiral, total_time: self.total_time }
    }

    pub fn pause(&mut self) { self.state = LoopState::Paused; }
    pub fn resume(&mut self) { self.state = LoopState::Running; self.accumulator = 0.0; }
    pub fn stop(&mut self) { self.state = LoopState::Stopping; }
    pub fn is_running(&self) -> bool { self.state == LoopState::Running }
    pub fn frame_number(&self) -> u64 { self.frame_count }
    pub fn fps(&self) -> f32 { self.fps_counter.avg_fps }
    pub fn set_target_fps(&mut self, fps: u32) { self.config.target_fps = fps; }
    pub fn set_fixed_timestep(&mut self, dt: f32) { self.config.fixed_timestep = dt.max(0.001); }
}

/// Result of an engine loop tick.
#[derive(Debug, Clone, Default)]
pub struct FrameTick {
    pub dt: f32,
    pub fixed_dt: f32,
    pub fixed_steps: u32,
    pub interpolation_alpha: f32,
    pub frame_number: u64,
    pub spiral_of_death: bool,
    pub total_time: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_loop_basic() {
        let mut el = EngineLoop::new(EngineLoopConfig::default());
        let tick = el.tick(1.0 / 60.0);
        assert!(tick.fixed_steps >= 1);
        assert!(tick.interpolation_alpha >= 0.0 && tick.interpolation_alpha <= 1.0);
    }

    #[test]
    fn test_spiral_of_death_protection() {
        let mut el = EngineLoop::new(EngineLoopConfig { max_fixed_steps: 4, ..Default::default() });
        let tick = el.tick(0.5); // Very long frame.
        assert!(tick.fixed_steps <= 4);
    }

    #[test]
    fn test_frame_smoother() {
        let mut smoother = FrameTimeSmoother::new(5);
        for _ in 0..10 { smoother.add_sample(1.0 / 60.0); }
        assert!((smoother.smoothed() - 1.0 / 60.0).abs() < 0.001);
    }

    #[test]
    fn test_pause_resume() {
        let mut el = EngineLoop::new(EngineLoopConfig::default());
        el.pause();
        let tick = el.tick(1.0 / 60.0);
        assert_eq!(tick.fixed_steps, 0);
        el.resume();
        let tick = el.tick(1.0 / 60.0);
        assert!(tick.fixed_steps >= 1);
    }
}
