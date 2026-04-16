//! Engine configuration types.

use genovo_render::RenderBackend;

/// Top-level engine configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Application name (shown in window title, logs, GPU debugger).
    pub app_name: String,

    /// Which rendering backend to use.
    pub backend: RenderBackend,

    /// Window configuration.
    pub window: WindowConfig,

    /// Target frames per second (0 = uncapped).
    pub target_fps: u32,

    /// Fixed timestep for physics simulation (in seconds).
    pub fixed_timestep: f64,

    /// Enable editor mode (includes editor UI, gizmos, hot-reload).
    pub editor_mode: bool,

    /// Enable validation/debug layers for rendering.
    pub render_debug: bool,

    /// Asset root directory.
    pub asset_root: String,

    /// Log level filter.
    pub log_level: LogLevel,

    /// Maximum simultaneous audio voices.
    pub max_audio_voices: u32,

    /// Audio output sample rate.
    pub audio_sample_rate: u32,

    /// Audio mixing buffer size in frames.
    pub audio_buffer_size: usize,

    /// Default gravity vector for physics (x, y, z).
    pub gravity: [f32; 3],
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            app_name: "Genovo Application".to_string(),
            backend: RenderBackend::Auto,
            window: WindowConfig::default(),
            target_fps: 0,
            fixed_timestep: 1.0 / 60.0,
            editor_mode: false,
            render_debug: cfg!(debug_assertions),
            asset_root: "assets".to_string(),
            log_level: LogLevel::Info,
            max_audio_voices: 64,
            audio_sample_rate: 44100,
            audio_buffer_size: 1024,
            gravity: [0.0, -9.81, 0.0],
        }
    }
}

/// Window creation configuration.
#[derive(Debug, Clone)]
pub struct WindowConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub fullscreen: bool,
    pub vsync: bool,
    pub resizable: bool,
    pub maximized: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "Genovo Engine".to_string(),
            width: 1920,
            height: 1080,
            fullscreen: false,
            vsync: true,
            resizable: true,
            maximized: false,
        }
    }
}

/// Log severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}
