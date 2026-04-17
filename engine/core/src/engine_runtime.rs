//! Engine runtime: initializes all subsystems, runs the main loop with fixed
//! timestep + variable render, and provides orderly shutdown.
//!
//! # Architecture
//!
//! The `EngineRuntime` owns (or holds references to) all major subsystems:
//! input, physics, audio, scripting, rendering, and the ECS world. It drives
//! the main loop using a fixed-timestep model for physics/gameplay and a
//! variable-timestep render call. This prevents physics instability from
//! variable frame rates while still rendering as fast as possible.
//!
//! # Fixed timestep model
//!
//! Each frame:
//! 1. Accumulate real elapsed time.
//! 2. While accumulated time >= fixed_dt:
//!    a. Poll input
//!    b. Run gameplay/AI systems
//!    c. Step physics
//!    d. Advance scripting
//!    e. Subtract fixed_dt from accumulator
//! 3. Compute interpolation alpha = accumulator / fixed_dt
//! 4. Render with interpolated state
//! 5. Present
//!
//! # Spiral-of-death protection
//!
//! If the game falls behind (accumulated time > max_frame_time), the
//! accumulator is clamped to prevent an unbounded number of fixed steps.

use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Subsystem state
// ---------------------------------------------------------------------------

/// Initialization state of a subsystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubsystemState {
    /// Not yet initialized.
    Uninitialized,
    /// Currently initializing.
    Initializing,
    /// Running normally.
    Running,
    /// Paused (still initialized, not ticking).
    Paused,
    /// Shutting down.
    ShuttingDown,
    /// Fully shut down.
    Shutdown,
    /// Encountered a fatal error.
    Error,
}

/// Identifies an engine subsystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubsystemId {
    Core,
    Input,
    Physics,
    Audio,
    Rendering,
    Scripting,
    ECS,
    Animation,
    Networking,
    AI,
    Assets,
    UI,
}

impl std::fmt::Display for SubsystemId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Core => write!(f, "Core"),
            Self::Input => write!(f, "Input"),
            Self::Physics => write!(f, "Physics"),
            Self::Audio => write!(f, "Audio"),
            Self::Rendering => write!(f, "Rendering"),
            Self::Scripting => write!(f, "Scripting"),
            Self::ECS => write!(f, "ECS"),
            Self::Animation => write!(f, "Animation"),
            Self::Networking => write!(f, "Networking"),
            Self::AI => write!(f, "AI"),
            Self::Assets => write!(f, "Assets"),
            Self::UI => write!(f, "UI"),
        }
    }
}

/// Record of a subsystem's state and initialization timing.
#[derive(Debug, Clone)]
pub struct SubsystemRecord {
    pub id: SubsystemId,
    pub state: SubsystemState,
    pub init_time_ms: f64,
    pub shutdown_order: u32,
}

// ---------------------------------------------------------------------------
// Runtime configuration
// ---------------------------------------------------------------------------

/// Configuration for the engine runtime loop.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Fixed timestep for physics/gameplay (default: 1/60 sec).
    pub fixed_timestep: f64,
    /// Maximum frame time before spiral-of-death clamp (default: 0.25 sec).
    pub max_frame_time: f64,
    /// Target render framerate (0 = unlimited).
    pub target_fps: u32,
    /// Whether to enable VSync.
    pub vsync: bool,
    /// Whether to run physics in a separate thread.
    pub threaded_physics: bool,
    /// Whether to enable the profiler.
    pub profiler_enabled: bool,
    /// Number of fixed-timestep iterations before forcing a render.
    pub max_fixed_steps_per_frame: u32,
    /// Whether to smooth delta time over multiple frames.
    pub smooth_delta_time: bool,
    /// Number of frames to average for smooth delta time.
    pub smooth_window: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            fixed_timestep: 1.0 / 60.0,
            max_frame_time: 0.25,
            target_fps: 0,
            vsync: true,
            threaded_physics: false,
            profiler_enabled: false,
            max_fixed_steps_per_frame: 5,
            smooth_delta_time: true,
            smooth_window: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Frame timing
// ---------------------------------------------------------------------------

/// Timing information for the current frame.
#[derive(Debug, Clone, Copy)]
pub struct FrameTiming {
    /// Real delta time (wall clock) in seconds.
    pub raw_delta_time: f64,
    /// Smoothed delta time (if enabled) in seconds.
    pub delta_time: f64,
    /// Fixed timestep size.
    pub fixed_delta_time: f64,
    /// Interpolation alpha for rendering (0.0 to 1.0).
    pub interpolation_alpha: f64,
    /// Total elapsed time since engine start.
    pub total_time: f64,
    /// Current FPS (smoothed).
    pub fps: f64,
    /// Frame index.
    pub frame: u64,
    /// Number of fixed steps taken this frame.
    pub fixed_steps_this_frame: u32,
}

// ---------------------------------------------------------------------------
// Runtime statistics
// ---------------------------------------------------------------------------

/// Cumulative runtime statistics.
#[derive(Debug, Clone, Default)]
pub struct RuntimeStats {
    /// Total frames rendered.
    pub total_frames: u64,
    /// Total fixed-step ticks executed.
    pub total_fixed_ticks: u64,
    /// Total elapsed time in seconds.
    pub total_time: f64,
    /// Average FPS over lifetime.
    pub average_fps: f64,
    /// Minimum frame time observed (seconds).
    pub min_frame_time: f64,
    /// Maximum frame time observed (seconds).
    pub max_frame_time: f64,
    /// Number of spiral-of-death clamps.
    pub spiral_clamps: u64,
    /// Number of frame pacing sleeps.
    pub pacing_sleeps: u64,
}

// ---------------------------------------------------------------------------
// EngineRuntime
// ---------------------------------------------------------------------------

/// The engine runtime manages subsystem lifecycle and drives the main loop.
///
/// # Usage
///
/// ```ignore
/// let mut runtime = EngineRuntime::new(RuntimeConfig::default());
/// runtime.initialize()?;
/// runtime.run(|timing| {
///     // Game logic here
///     true // return false to exit
/// });
/// runtime.shutdown();
/// ```
pub struct EngineRuntime {
    config: RuntimeConfig,
    subsystems: Vec<SubsystemRecord>,
    stats: RuntimeStats,
    running: bool,
    accumulator: f64,
    total_time: f64,
    frame_index: u64,
    last_frame_time: Option<Instant>,
    start_time: Option<Instant>,
    delta_history: Vec<f64>,
    smoothed_fps: f64,
}

impl EngineRuntime {
    /// Create a new engine runtime with the given configuration.
    pub fn new(config: RuntimeConfig) -> Self {
        Self {
            config,
            subsystems: Vec::new(),
            stats: RuntimeStats {
                min_frame_time: f64::MAX,
                max_frame_time: 0.0,
                ..Default::default()
            },
            running: false,
            accumulator: 0.0,
            total_time: 0.0,
            frame_index: 0,
            last_frame_time: None,
            start_time: None,
            delta_history: Vec::new(),
            smoothed_fps: 60.0,
        }
    }

    /// Initialize all subsystems in dependency order.
    ///
    /// Subsystems are initialized in the order they are registered. Each
    /// subsystem's init time is recorded for startup profiling.
    pub fn initialize(&mut self) -> Result<(), String> {
        let init_start = Instant::now();
        self.start_time = Some(init_start);

        // Register subsystems in initialization order
        let subsystem_order = [
            (SubsystemId::Core, 0),
            (SubsystemId::Assets, 1),
            (SubsystemId::Input, 2),
            (SubsystemId::Audio, 3),
            (SubsystemId::Physics, 4),
            (SubsystemId::ECS, 5),
            (SubsystemId::Animation, 6),
            (SubsystemId::Scripting, 7),
            (SubsystemId::AI, 8),
            (SubsystemId::Networking, 9),
            (SubsystemId::Rendering, 10),
            (SubsystemId::UI, 11),
        ];

        for (id, shutdown_order) in &subsystem_order {
            let sub_start = Instant::now();
            // In a real engine, this would call each subsystem's init()
            let init_time = sub_start.elapsed().as_secs_f64() * 1000.0;

            self.subsystems.push(SubsystemRecord {
                id: *id,
                state: SubsystemState::Running,
                init_time_ms: init_time,
                // Shutdown in reverse order
                shutdown_order: (subsystem_order.len() as u32) - shutdown_order,
            });
        }

        let total_init = init_start.elapsed().as_secs_f64() * 1000.0;
        log::info!("Engine initialized in {:.1}ms ({} subsystems)", total_init, self.subsystems.len());

        self.running = true;
        Ok(())
    }

    /// Run the main loop with a user-provided tick callback.
    ///
    /// The callback receives timing information and returns `true` to continue
    /// or `false` to exit the loop.
    pub fn run<F>(&mut self, mut tick: F)
    where
        F: FnMut(&FrameTiming) -> bool,
    {
        self.last_frame_time = Some(Instant::now());

        while self.running {
            let timing = self.begin_frame();

            // Run fixed-timestep ticks
            let fixed_steps = timing.fixed_steps_this_frame;
            self.stats.total_fixed_ticks += fixed_steps as u64;

            // Call user tick
            if !tick(&timing) {
                self.running = false;
                break;
            }

            self.end_frame(&timing);
        }
    }

    /// Compute frame timing for the current frame.
    fn begin_frame(&mut self) -> FrameTiming {
        let now = Instant::now();
        let raw_dt = self.last_frame_time
            .map(|t| now.duration_since(t).as_secs_f64())
            .unwrap_or(self.config.fixed_timestep);
        self.last_frame_time = Some(now);

        // Spiral-of-death protection
        let clamped_dt = if raw_dt > self.config.max_frame_time {
            self.stats.spiral_clamps += 1;
            self.config.max_frame_time
        } else {
            raw_dt
        };

        // Smooth delta time
        let smoothed_dt = if self.config.smooth_delta_time {
            self.delta_history.push(clamped_dt);
            if self.delta_history.len() > self.config.smooth_window {
                self.delta_history.remove(0);
            }
            let sum: f64 = self.delta_history.iter().sum();
            sum / self.delta_history.len() as f64
        } else {
            clamped_dt
        };

        // Fixed timestep accumulation
        self.accumulator += clamped_dt;
        let fixed_dt = self.config.fixed_timestep;
        let mut fixed_steps = 0u32;
        while self.accumulator >= fixed_dt && fixed_steps < self.config.max_fixed_steps_per_frame {
            self.accumulator -= fixed_dt;
            fixed_steps += 1;
        }

        // Compute interpolation alpha
        let alpha = self.accumulator / fixed_dt;

        self.total_time += clamped_dt;

        // Update FPS
        if clamped_dt > 0.0 {
            let instant_fps = 1.0 / clamped_dt;
            self.smoothed_fps = self.smoothed_fps * 0.95 + instant_fps * 0.05;
        }

        // Update stats
        self.stats.total_frames += 1;
        self.stats.total_time = self.total_time;
        if clamped_dt < self.stats.min_frame_time {
            self.stats.min_frame_time = clamped_dt;
        }
        if clamped_dt > self.stats.max_frame_time {
            self.stats.max_frame_time = clamped_dt;
        }
        if self.stats.total_frames > 0 {
            self.stats.average_fps = self.stats.total_frames as f64 / self.total_time;
        }

        let timing = FrameTiming {
            raw_delta_time: raw_dt,
            delta_time: smoothed_dt,
            fixed_delta_time: fixed_dt,
            interpolation_alpha: alpha,
            total_time: self.total_time,
            fps: self.smoothed_fps,
            frame: self.frame_index,
            fixed_steps_this_frame: fixed_steps,
        };

        self.frame_index += 1;
        timing
    }

    /// End-of-frame: frame pacing (sleep to target FPS if configured).
    fn end_frame(&mut self, _timing: &FrameTiming) {
        if self.config.target_fps > 0 {
            let target_frame_time = 1.0 / self.config.target_fps as f64;
            let elapsed = self.last_frame_time
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(0.0);
            let sleep_time = target_frame_time - elapsed;
            if sleep_time > 0.001 {
                std::thread::sleep(Duration::from_secs_f64(sleep_time));
                self.stats.pacing_sleeps += 1;
            }
        }
    }

    /// Shutdown all subsystems in reverse initialization order.
    pub fn shutdown(&mut self) {
        self.running = false;

        // Sort by shutdown order (reverse of init)
        self.subsystems.sort_by_key(|s| s.shutdown_order);

        for sub in &mut self.subsystems {
            sub.state = SubsystemState::ShuttingDown;
            // In a real engine, call sub.shutdown()
            sub.state = SubsystemState::Shutdown;
        }

        let total_time = self.start_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        log::info!(
            "Engine shutdown after {:.1}s ({} frames, avg {:.1} FPS)",
            total_time, self.stats.total_frames, self.stats.average_fps
        );
    }

    /// Request the runtime to stop at the end of the current frame.
    pub fn request_exit(&mut self) {
        self.running = false;
    }

    /// Whether the runtime is currently running.
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Get the runtime configuration.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Update the runtime configuration. Takes effect next frame.
    pub fn set_config(&mut self, config: RuntimeConfig) {
        self.config = config;
    }

    /// Get cumulative runtime statistics.
    pub fn stats(&self) -> &RuntimeStats {
        &self.stats
    }

    /// Get subsystem records.
    pub fn subsystems(&self) -> &[SubsystemRecord] {
        &self.subsystems
    }

    /// Get the state of a specific subsystem.
    pub fn subsystem_state(&self, id: SubsystemId) -> SubsystemState {
        self.subsystems
            .iter()
            .find(|s| s.id == id)
            .map(|s| s.state)
            .unwrap_or(SubsystemState::Uninitialized)
    }

    /// Get the current frame index.
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Get total elapsed time.
    pub fn total_time(&self) -> f64 {
        self.total_time
    }

    /// Get the smoothed FPS.
    pub fn fps(&self) -> f64 {
        self.smoothed_fps
    }
}

impl Default for EngineRuntime {
    fn default() -> Self {
        Self::new(RuntimeConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_creation() {
        let rt = EngineRuntime::new(RuntimeConfig::default());
        assert!(!rt.is_running());
        assert_eq!(rt.frame_index(), 0);
    }

    #[test]
    fn test_runtime_initialize() {
        let mut rt = EngineRuntime::new(RuntimeConfig::default());
        rt.initialize().unwrap();
        assert!(rt.is_running());
        assert!(rt.subsystems().len() > 0);
        for sub in rt.subsystems() {
            assert_eq!(sub.state, SubsystemState::Running);
        }
    }

    #[test]
    fn test_runtime_run_and_exit() {
        let mut rt = EngineRuntime::new(RuntimeConfig::default());
        rt.initialize().unwrap();
        let mut count = 0;
        rt.run(|_timing| {
            count += 1;
            count < 5
        });
        assert_eq!(count, 5);
        assert!(!rt.is_running());
    }

    #[test]
    fn test_runtime_shutdown() {
        let mut rt = EngineRuntime::new(RuntimeConfig::default());
        rt.initialize().unwrap();
        rt.shutdown();
        for sub in rt.subsystems() {
            assert_eq!(sub.state, SubsystemState::Shutdown);
        }
    }

    #[test]
    fn test_frame_timing() {
        let mut rt = EngineRuntime::new(RuntimeConfig {
            fixed_timestep: 1.0 / 60.0,
            ..Default::default()
        });
        rt.initialize().unwrap();
        rt.run(|timing| {
            assert!(timing.delta_time > 0.0);
            assert!(timing.fixed_delta_time > 0.0);
            assert!(timing.interpolation_alpha >= 0.0);
            assert!(timing.interpolation_alpha <= 1.0);
            timing.frame < 3
        });
    }

    #[test]
    fn test_runtime_stats() {
        let mut rt = EngineRuntime::new(RuntimeConfig::default());
        rt.initialize().unwrap();
        rt.run(|t| t.frame < 10);
        let stats = rt.stats();
        assert!(stats.total_frames > 0);
        assert!(stats.total_time > 0.0);
    }

    #[test]
    fn test_subsystem_state() {
        let mut rt = EngineRuntime::new(RuntimeConfig::default());
        assert_eq!(rt.subsystem_state(SubsystemId::Core), SubsystemState::Uninitialized);
        rt.initialize().unwrap();
        assert_eq!(rt.subsystem_state(SubsystemId::Core), SubsystemState::Running);
    }

    #[test]
    fn test_config_update() {
        let mut rt = EngineRuntime::new(RuntimeConfig::default());
        let mut config = rt.config().clone();
        config.target_fps = 30;
        rt.set_config(config);
        assert_eq!(rt.config().target_fps, 30);
    }

    #[test]
    fn test_request_exit() {
        let mut rt = EngineRuntime::new(RuntimeConfig::default());
        rt.initialize().unwrap();
        rt.request_exit();
        assert!(!rt.is_running());
    }
}
