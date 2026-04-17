//! On-Screen Debug Logger
//!
//! Provides an in-game overlay for displaying debug messages, warnings,
//! and errors directly on screen. Messages can be persistent or transient,
//! categorized, filtered, grouped, and styled.
//!
//! Features:
//! - Persistent messages with categories
//! - Auto-fade after configurable duration
//! - Message stacking and grouping
//! - Scrollable in-game log overlay
//! - Category-based filtering
//! - Screenshot capture with timestamp
//! - Video capture stub
//!
//! # Usage
//!
//! ```ignore
//! let mut logger = ScreenLogger::new();
//! logger.log(LogCategory::Gameplay, "Player spawned at origin");
//! logger.warn(LogCategory::Physics, "Collision solver exceeded iteration limit");
//! logger.error(LogCategory::Rendering, "Shader compilation failed: basic.frag");
//! ```

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// LogCategory
// ---------------------------------------------------------------------------

/// Category for organizing on-screen log messages.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LogCategory {
    /// General-purpose messages.
    General,
    /// Gameplay events.
    Gameplay,
    /// AI behavior.
    AI,
    /// Physics simulation.
    Physics,
    /// Rendering and graphics.
    Rendering,
    /// Audio.
    Audio,
    /// UI events.
    UI,
    /// Network events.
    Network,
    /// Performance warnings.
    Performance,
    /// Script errors.
    Scripting,
    /// Asset loading.
    Assets,
    /// System/engine messages.
    System,
    /// Custom category.
    Custom(String),
}

impl fmt::Display for LogCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::General => write!(f, "General"),
            Self::Gameplay => write!(f, "Gameplay"),
            Self::AI => write!(f, "AI"),
            Self::Physics => write!(f, "Physics"),
            Self::Rendering => write!(f, "Rendering"),
            Self::Audio => write!(f, "Audio"),
            Self::UI => write!(f, "UI"),
            Self::Network => write!(f, "Network"),
            Self::Performance => write!(f, "Performance"),
            Self::Scripting => write!(f, "Scripting"),
            Self::Assets => write!(f, "Assets"),
            Self::System => write!(f, "System"),
            Self::Custom(name) => write!(f, "{}", name),
        }
    }
}

impl Default for LogCategory {
    fn default() -> Self {
        Self::General
    }
}

// ---------------------------------------------------------------------------
// LogSeverity
// ---------------------------------------------------------------------------

/// Severity level for a log message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogSeverity {
    /// Verbose debug information.
    Trace,
    /// Normal informational messages.
    Info,
    /// Warnings that may indicate problems.
    Warning,
    /// Errors that need attention.
    Error,
    /// Critical/fatal errors.
    Fatal,
}

impl LogSeverity {
    /// Returns a color associated with this severity level.
    pub fn color(&self) -> [f32; 4] {
        match self {
            Self::Trace => [0.6, 0.6, 0.6, 1.0],
            Self::Info => [1.0, 1.0, 1.0, 1.0],
            Self::Warning => [1.0, 0.8, 0.0, 1.0],
            Self::Error => [1.0, 0.2, 0.2, 1.0],
            Self::Fatal => [1.0, 0.0, 0.5, 1.0],
        }
    }

    /// Returns a prefix string for this severity.
    pub fn prefix(&self) -> &'static str {
        match self {
            Self::Trace => "[TRACE]",
            Self::Info => "[INFO]",
            Self::Warning => "[WARN]",
            Self::Error => "[ERROR]",
            Self::Fatal => "[FATAL]",
        }
    }
}

impl fmt::Display for LogSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.prefix())
    }
}

// ---------------------------------------------------------------------------
// ScreenLogMessage
// ---------------------------------------------------------------------------

/// A single message displayed on screen.
#[derive(Debug, Clone)]
pub struct ScreenLogMessage {
    /// Unique message ID.
    pub id: u64,
    /// The message text.
    pub text: String,
    /// Category of the message.
    pub category: LogCategory,
    /// Severity level.
    pub severity: LogSeverity,
    /// Time when the message was created (engine time in seconds).
    pub timestamp: f64,
    /// Frame number when the message was created.
    pub frame: u64,
    /// Duration in seconds before the message starts fading (0 = persistent).
    pub display_duration: f32,
    /// Duration of the fade-out effect in seconds.
    pub fade_duration: f32,
    /// Current opacity (1.0 = fully visible, 0.0 = invisible).
    pub opacity: f32,
    /// Whether this message is persistent (never auto-fades).
    pub persistent: bool,
    /// Whether this message has been acknowledged/dismissed.
    pub dismissed: bool,
    /// Number of times this identical message has been repeated.
    pub repeat_count: u32,
    /// Optional key for grouping/deduplication.
    pub group_key: Option<String>,
    /// Custom color override.
    pub custom_color: Option<[f32; 4]>,
    /// Optional source location (file:line).
    pub source_location: Option<String>,
}

impl ScreenLogMessage {
    /// Creates a new log message.
    pub fn new(
        id: u64,
        text: impl Into<String>,
        category: LogCategory,
        severity: LogSeverity,
        timestamp: f64,
        frame: u64,
    ) -> Self {
        Self {
            id,
            text: text.into(),
            category,
            severity,
            timestamp,
            display_duration: 5.0,
            fade_duration: 1.0,
            opacity: 1.0,
            persistent: false,
            dismissed: false,
            repeat_count: 1,
            group_key: None,
            custom_color: None,
            source_location: None,
            frame,
        }
    }

    /// Sets the display duration.
    pub fn with_duration(mut self, duration: f32) -> Self {
        self.display_duration = duration;
        self
    }

    /// Makes the message persistent.
    pub fn with_persistent(mut self) -> Self {
        self.persistent = true;
        self
    }

    /// Sets a group key for deduplication.
    pub fn with_group_key(mut self, key: impl Into<String>) -> Self {
        self.group_key = Some(key.into());
        self
    }

    /// Sets a custom color.
    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.custom_color = Some(color);
        self
    }

    /// Sets the source location.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source_location = Some(source.into());
        self
    }

    /// Returns the effective color for this message.
    pub fn effective_color(&self) -> [f32; 4] {
        let base = self.custom_color.unwrap_or_else(|| self.severity.color());
        [base[0], base[1], base[2], base[3] * self.opacity]
    }

    /// Returns the formatted display text.
    pub fn display_text(&self) -> String {
        let mut parts = Vec::new();
        parts.push(format!("{} [{}]", self.severity.prefix(), self.category));

        if self.repeat_count > 1 {
            parts.push(format!("(x{})", self.repeat_count));
        }

        parts.push(self.text.clone());

        if let Some(ref source) = self.source_location {
            parts.push(format!("({})", source));
        }

        parts.join(" ")
    }

    /// Updates the message's opacity based on elapsed time.
    pub fn update_fade(&mut self, current_time: f64) {
        if self.persistent || self.dismissed {
            return;
        }

        let age = (current_time - self.timestamp) as f32;
        if age > self.display_duration + self.fade_duration {
            self.opacity = 0.0;
        } else if age > self.display_duration {
            let fade_progress = (age - self.display_duration) / self.fade_duration;
            self.opacity = (1.0 - fade_progress).clamp(0.0, 1.0);
        } else {
            self.opacity = 1.0;
        }
    }

    /// Returns `true` if this message should be removed.
    pub fn should_remove(&self) -> bool {
        self.dismissed || (!self.persistent && self.opacity <= 0.0)
    }
}

// ---------------------------------------------------------------------------
// LogOverlayConfig
// ---------------------------------------------------------------------------

/// Configuration for the on-screen log overlay.
#[derive(Debug, Clone)]
pub struct LogOverlayConfig {
    /// Whether the overlay is visible.
    pub visible: bool,
    /// Maximum number of messages to display on screen.
    pub max_visible_messages: usize,
    /// Maximum number of messages to keep in the history buffer.
    pub max_history: usize,
    /// Position of the overlay on screen.
    pub position: OverlayPosition,
    /// Padding from the screen edges (in logical pixels).
    pub edge_padding: f32,
    /// Spacing between messages (in logical pixels).
    pub line_spacing: f32,
    /// Font size for messages.
    pub font_size: f32,
    /// Background opacity for the overlay panel (0 = transparent).
    pub background_opacity: f32,
    /// Default display duration for transient messages.
    pub default_display_duration: f32,
    /// Default fade duration.
    pub default_fade_duration: f32,
    /// Minimum severity level to display.
    pub min_severity: LogSeverity,
    /// Whether to show timestamps.
    pub show_timestamps: bool,
    /// Whether to show category labels.
    pub show_categories: bool,
    /// Whether to show frame numbers.
    pub show_frame_numbers: bool,
    /// Whether to group duplicate messages.
    pub group_duplicates: bool,
    /// Whether the overlay is scrollable.
    pub scrollable: bool,
    /// Width of the overlay panel (0 = auto).
    pub width: f32,
}

/// Position of the overlay on screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayPosition {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    TopCenter,
    BottomCenter,
}

impl Default for OverlayPosition {
    fn default() -> Self {
        Self::TopLeft
    }
}

impl Default for LogOverlayConfig {
    fn default() -> Self {
        Self {
            visible: true,
            max_visible_messages: 20,
            max_history: 500,
            position: OverlayPosition::TopLeft,
            edge_padding: 10.0,
            line_spacing: 2.0,
            font_size: 14.0,
            background_opacity: 0.3,
            default_display_duration: 5.0,
            default_fade_duration: 1.0,
            min_severity: LogSeverity::Info,
            show_timestamps: false,
            show_categories: true,
            show_frame_numbers: false,
            group_duplicates: true,
            scrollable: true,
            width: 500.0,
        }
    }
}

// ---------------------------------------------------------------------------
// CategoryFilter
// ---------------------------------------------------------------------------

/// Filter for controlling which categories are visible.
#[derive(Debug, Clone)]
pub struct CategoryFilter {
    /// Enabled categories. If empty, all categories are shown.
    enabled: HashMap<LogCategory, bool>,
    /// Default state for categories not explicitly set.
    default_enabled: bool,
}

impl CategoryFilter {
    /// Creates a new filter with all categories enabled.
    pub fn all_enabled() -> Self {
        Self {
            enabled: HashMap::new(),
            default_enabled: true,
        }
    }

    /// Creates a new filter with all categories disabled.
    pub fn all_disabled() -> Self {
        Self {
            enabled: HashMap::new(),
            default_enabled: false,
        }
    }

    /// Enables a specific category.
    pub fn enable(&mut self, category: LogCategory) {
        self.enabled.insert(category, true);
    }

    /// Disables a specific category.
    pub fn disable(&mut self, category: LogCategory) {
        self.enabled.insert(category, false);
    }

    /// Toggles a category.
    pub fn toggle(&mut self, category: &LogCategory) {
        let current = self.is_enabled(category);
        self.enabled.insert(category.clone(), !current);
    }

    /// Returns whether a category is enabled.
    pub fn is_enabled(&self, category: &LogCategory) -> bool {
        self.enabled
            .get(category)
            .copied()
            .unwrap_or(self.default_enabled)
    }

    /// Resets all filters to default.
    pub fn reset(&mut self) {
        self.enabled.clear();
    }
}

impl Default for CategoryFilter {
    fn default() -> Self {
        Self::all_enabled()
    }
}

// ---------------------------------------------------------------------------
// ScreenshotCapture
// ---------------------------------------------------------------------------

/// Configuration and state for screenshot capture.
#[derive(Debug, Clone)]
pub struct ScreenshotCapture {
    /// Directory where screenshots are saved.
    pub output_directory: PathBuf,
    /// Filename prefix.
    pub prefix: String,
    /// Image format.
    pub format: ScreenshotFormat,
    /// JPEG quality (1-100).
    pub jpeg_quality: u8,
    /// Whether to include the debug overlay in screenshots.
    pub include_overlay: bool,
    /// Whether to add a timestamp watermark.
    pub add_timestamp: bool,
    /// Counter for unique filenames.
    pub counter: u32,
    /// Whether a capture is pending for this frame.
    pub pending_capture: bool,
}

/// Screenshot image format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScreenshotFormat {
    Png,
    Jpeg,
    Bmp,
    Tga,
}

impl ScreenshotFormat {
    /// Returns the file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::Bmp => "bmp",
            Self::Tga => "tga",
        }
    }
}

impl Default for ScreenshotFormat {
    fn default() -> Self {
        Self::Png
    }
}

impl ScreenshotCapture {
    /// Creates a new screenshot capture configuration.
    pub fn new(output_directory: impl Into<PathBuf>) -> Self {
        Self {
            output_directory: output_directory.into(),
            prefix: "screenshot".to_string(),
            format: ScreenshotFormat::Png,
            jpeg_quality: 90,
            include_overlay: false,
            add_timestamp: true,
            counter: 0,
            pending_capture: false,
        }
    }

    /// Generates the next filename for a screenshot.
    pub fn next_filename(&mut self) -> PathBuf {
        self.counter += 1;

        let timestamp = chrono_timestamp();
        let filename = format!(
            "{}_{:04}_{}.{}",
            self.prefix,
            self.counter,
            timestamp,
            self.format.extension()
        );

        self.output_directory.join(filename)
    }

    /// Requests a screenshot capture on the next frame.
    pub fn request_capture(&mut self) {
        self.pending_capture = true;
    }

    /// Returns `true` if a capture is pending.
    pub fn is_capture_pending(&self) -> bool {
        self.pending_capture
    }

    /// Clears the pending capture flag.
    pub fn clear_pending(&mut self) {
        self.pending_capture = false;
    }
}

/// Generates a simple timestamp string (YYYYMMDD_HHMMSS format stub).
fn chrono_timestamp() -> String {
    // In a real engine, this would use the system clock.
    // For now, return a placeholder format.
    format!("20260416_120000")
}

// ---------------------------------------------------------------------------
// VideoCaptureState
// ---------------------------------------------------------------------------

/// Stub for video capture functionality.
#[derive(Debug, Clone)]
pub struct VideoCaptureState {
    /// Whether video capture is currently recording.
    pub recording: bool,
    /// Output file path.
    pub output_path: Option<PathBuf>,
    /// Frame rate for the recording.
    pub frame_rate: u32,
    /// Resolution width.
    pub width: u32,
    /// Resolution height.
    pub height: u32,
    /// Number of frames captured so far.
    pub frames_captured: u64,
    /// Total recording duration in seconds.
    pub duration: f64,
    /// Video codec identifier.
    pub codec: String,
}

impl VideoCaptureState {
    /// Creates a new video capture state.
    pub fn new() -> Self {
        Self {
            recording: false,
            output_path: None,
            frame_rate: 30,
            width: 1920,
            height: 1080,
            frames_captured: 0,
            duration: 0.0,
            codec: "h264".to_string(),
        }
    }

    /// Starts a video recording.
    pub fn start_recording(&mut self, output_path: impl Into<PathBuf>) {
        self.recording = true;
        self.output_path = Some(output_path.into());
        self.frames_captured = 0;
        self.duration = 0.0;
    }

    /// Stops the current recording.
    pub fn stop_recording(&mut self) {
        self.recording = false;
    }

    /// Records a frame (stub).
    pub fn capture_frame(&mut self, dt: f64) {
        if !self.recording {
            return;
        }
        self.frames_captured += 1;
        self.duration += dt;
    }

    /// Returns the estimated file size in bytes (rough estimate).
    pub fn estimated_size_bytes(&self) -> u64 {
        // Very rough estimate: ~5 MB per minute at 1080p/30fps
        let minutes = self.duration / 60.0;
        (minutes * 5.0 * 1024.0 * 1024.0) as u64
    }
}

impl Default for VideoCaptureState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ScreenLogger
// ---------------------------------------------------------------------------

/// Central manager for on-screen debug messages.
///
/// Provides methods for logging messages at various severity levels,
/// managing the message lifecycle (display, fade, removal), filtering,
/// and screenshot/video capture.
pub struct ScreenLogger {
    /// Active messages currently displayed or fading.
    messages: Vec<ScreenLogMessage>,
    /// Full message history (including removed messages).
    history: Vec<ScreenLogMessage>,
    /// Configuration for the overlay display.
    pub config: LogOverlayConfig,
    /// Category filter.
    pub filter: CategoryFilter,
    /// Screenshot capture state.
    pub screenshot: ScreenshotCapture,
    /// Video capture state.
    pub video: VideoCaptureState,
    /// Next message ID.
    next_id: u64,
    /// Current engine time.
    current_time: f64,
    /// Current frame number.
    current_frame: u64,
    /// Scroll offset for the scrollable overlay.
    pub scroll_offset: f32,
    /// Whether the overlay is in expanded (full log) mode.
    pub expanded: bool,
}

impl ScreenLogger {
    /// Creates a new screen logger with default configuration.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            history: Vec::new(),
            config: LogOverlayConfig::default(),
            filter: CategoryFilter::all_enabled(),
            screenshot: ScreenshotCapture::new("./screenshots"),
            video: VideoCaptureState::new(),
            next_id: 1,
            current_time: 0.0,
            current_frame: 0,
            scroll_offset: 0.0,
            expanded: false,
        }
    }

    /// Logs a message at the Info severity level.
    pub fn log(&mut self, category: LogCategory, text: impl Into<String>) -> u64 {
        self.add_message(category, LogSeverity::Info, text)
    }

    /// Logs a warning message.
    pub fn warn(&mut self, category: LogCategory, text: impl Into<String>) -> u64 {
        self.add_message(category, LogSeverity::Warning, text)
    }

    /// Logs an error message.
    pub fn error(&mut self, category: LogCategory, text: impl Into<String>) -> u64 {
        self.add_message(category, LogSeverity::Error, text)
    }

    /// Logs a fatal message.
    pub fn fatal(&mut self, category: LogCategory, text: impl Into<String>) -> u64 {
        self.add_message(category, LogSeverity::Fatal, text)
    }

    /// Logs a trace/verbose message.
    pub fn trace(&mut self, category: LogCategory, text: impl Into<String>) -> u64 {
        self.add_message(category, LogSeverity::Trace, text)
    }

    /// Logs a persistent message that does not auto-fade.
    pub fn log_persistent(
        &mut self,
        category: LogCategory,
        severity: LogSeverity,
        text: impl Into<String>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let msg = ScreenLogMessage::new(id, text, category, severity, self.current_time, self.current_frame)
            .with_persistent();
        self.messages.push(msg);
        id
    }

    /// Logs a message with a group key for deduplication.
    pub fn log_grouped(
        &mut self,
        category: LogCategory,
        severity: LogSeverity,
        text: impl Into<String>,
        group_key: impl Into<String>,
    ) -> u64 {
        let key = group_key.into();
        let text_str = text.into();

        // Check if a message with this group key already exists.
        if self.config.group_duplicates {
            if let Some(existing) = self.messages.iter_mut().find(|m| {
                m.group_key.as_deref() == Some(&key) && !m.dismissed
            }) {
                existing.repeat_count += 1;
                existing.timestamp = self.current_time;
                existing.opacity = 1.0;
                existing.text = text_str;
                return existing.id;
            }
        }

        let id = self.next_id;
        self.next_id += 1;
        let msg = ScreenLogMessage::new(id, text_str, category, severity, self.current_time, self.current_frame)
            .with_group_key(key);
        self.messages.push(msg);
        id
    }

    /// Dismisses a message by ID.
    pub fn dismiss(&mut self, id: u64) -> bool {
        if let Some(msg) = self.messages.iter_mut().find(|m| m.id == id) {
            msg.dismissed = true;
            true
        } else {
            false
        }
    }

    /// Dismisses all messages.
    pub fn dismiss_all(&mut self) {
        for msg in &mut self.messages {
            msg.dismissed = true;
        }
    }

    /// Clears all messages and history.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.history.clear();
    }

    /// Updates the logger (call once per frame).
    ///
    /// Updates fade timers, removes expired messages, and processes
    /// pending captures.
    pub fn update(&mut self, dt: f64, frame: u64) {
        self.current_time += dt;
        self.current_frame = frame;

        // Update message fades.
        for msg in &mut self.messages {
            msg.update_fade(self.current_time);
        }

        // Move expired messages to history.
        let mut i = 0;
        while i < self.messages.len() {
            if self.messages[i].should_remove() {
                let msg = self.messages.remove(i);
                self.history.push(msg);
            } else {
                i += 1;
            }
        }

        // Trim history.
        while self.history.len() > self.config.max_history {
            self.history.remove(0);
        }

        // Process video capture.
        self.video.capture_frame(dt);
    }

    /// Returns the messages that should currently be displayed, filtered
    /// by category and severity.
    pub fn visible_messages(&self) -> Vec<&ScreenLogMessage> {
        self.messages
            .iter()
            .filter(|m| m.opacity > 0.0 && !m.dismissed)
            .filter(|m| m.severity >= self.config.min_severity)
            .filter(|m| self.filter.is_enabled(&m.category))
            .take(self.config.max_visible_messages)
            .collect()
    }

    /// Returns the full message history.
    pub fn history(&self) -> &[ScreenLogMessage] {
        &self.history
    }

    /// Returns the number of active (visible) messages.
    pub fn active_message_count(&self) -> usize {
        self.messages.iter().filter(|m| !m.dismissed).count()
    }

    /// Returns the total number of messages in history.
    pub fn history_count(&self) -> usize {
        self.history.len()
    }

    /// Requests a screenshot.
    pub fn capture_screenshot(&mut self) -> PathBuf {
        let path = self.screenshot.next_filename();
        self.screenshot.request_capture();
        self.log(LogCategory::System, format!("Screenshot saved: {:?}", path));
        path
    }

    /// Starts video recording.
    pub fn start_video_recording(&mut self, path: impl Into<PathBuf>) {
        let p = path.into();
        self.video.start_recording(p.clone());
        self.log(LogCategory::System, format!("Recording started: {:?}", p));
    }

    /// Stops video recording.
    pub fn stop_video_recording(&mut self) {
        if self.video.recording {
            self.video.stop_recording();
            self.log(LogCategory::System, format!(
                "Recording stopped. {} frames, {:.1}s",
                self.video.frames_captured,
                self.video.duration
            ));
        }
    }

    /// Toggles the expanded log view.
    pub fn toggle_expanded(&mut self) {
        self.expanded = !self.expanded;
    }

    /// Scrolls the log overlay.
    pub fn scroll(&mut self, delta: f32) {
        self.scroll_offset = (self.scroll_offset + delta).max(0.0);
    }

    /// Returns debug statistics about the logger.
    pub fn stats(&self) -> ScreenLoggerStats {
        let mut category_counts: HashMap<LogCategory, usize> = HashMap::new();
        let mut severity_counts: HashMap<LogSeverity, usize> = HashMap::new();

        for msg in &self.messages {
            *category_counts.entry(msg.category.clone()).or_insert(0) += 1;
            *severity_counts.entry(msg.severity).or_insert(0) += 1;
        }

        ScreenLoggerStats {
            active_messages: self.messages.len(),
            history_size: self.history.len(),
            category_counts,
            severity_counts,
            is_recording: self.video.recording,
            capture_pending: self.screenshot.pending_capture,
        }
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    /// Adds a message and returns its ID.
    fn add_message(
        &mut self,
        category: LogCategory,
        severity: LogSeverity,
        text: impl Into<String>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let mut msg = ScreenLogMessage::new(
            id,
            text,
            category,
            severity,
            self.current_time,
            self.current_frame,
        );
        msg.display_duration = self.config.default_display_duration;
        msg.fade_duration = self.config.default_fade_duration;

        self.messages.push(msg);
        id
    }
}

impl Default for ScreenLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ScreenLogger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScreenLogger")
            .field("active_messages", &self.messages.len())
            .field("history_size", &self.history.len())
            .field("visible", &self.config.visible)
            .field("expanded", &self.expanded)
            .finish()
    }
}

/// Statistics about the screen logger.
#[derive(Debug, Clone)]
pub struct ScreenLoggerStats {
    /// Number of active messages.
    pub active_messages: usize,
    /// Number of messages in history.
    pub history_size: usize,
    /// Message counts per category.
    pub category_counts: HashMap<LogCategory, usize>,
    /// Message counts per severity.
    pub severity_counts: HashMap<LogSeverity, usize>,
    /// Whether video recording is active.
    pub is_recording: bool,
    /// Whether a screenshot capture is pending.
    pub capture_pending: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_message() {
        let mut logger = ScreenLogger::new();
        let id = logger.log(LogCategory::General, "Hello, world!");
        assert!(id > 0);
        assert_eq!(logger.active_message_count(), 1);
    }

    #[test]
    fn test_log_severities() {
        let mut logger = ScreenLogger::new();
        logger.trace(LogCategory::General, "trace");
        logger.log(LogCategory::General, "info");
        logger.warn(LogCategory::General, "warning");
        logger.error(LogCategory::General, "error");
        logger.fatal(LogCategory::General, "fatal");
        assert_eq!(logger.active_message_count(), 5);
    }

    #[test]
    fn test_message_fade() {
        let mut msg = ScreenLogMessage::new(1, "test", LogCategory::General, LogSeverity::Info, 0.0, 0);
        msg.display_duration = 2.0;
        msg.fade_duration = 1.0;

        msg.update_fade(1.0);
        assert!((msg.opacity - 1.0).abs() < 0.01);

        msg.update_fade(2.5);
        assert!(msg.opacity < 1.0);
        assert!(msg.opacity > 0.0);

        msg.update_fade(4.0);
        assert!(msg.opacity <= 0.0);
    }

    #[test]
    fn test_dismiss() {
        let mut logger = ScreenLogger::new();
        let id = logger.log(LogCategory::General, "dismissable");
        assert!(logger.dismiss(id));
        assert_eq!(logger.active_message_count(), 0);
    }

    #[test]
    fn test_category_filter() {
        let mut filter = CategoryFilter::all_enabled();
        assert!(filter.is_enabled(&LogCategory::Physics));
        filter.disable(LogCategory::Physics);
        assert!(!filter.is_enabled(&LogCategory::Physics));
        filter.toggle(&LogCategory::Physics);
        assert!(filter.is_enabled(&LogCategory::Physics));
    }

    #[test]
    fn test_grouped_messages() {
        let mut logger = ScreenLogger::new();
        logger.log_grouped(LogCategory::General, LogSeverity::Warning, "Warn 1", "warn_group");
        logger.log_grouped(LogCategory::General, LogSeverity::Warning, "Warn 2", "warn_group");
        // Should have only 1 active message with repeat_count = 2.
        assert_eq!(logger.active_message_count(), 1);
    }

    #[test]
    fn test_persistent_message() {
        let mut logger = ScreenLogger::new();
        let id = logger.log_persistent(LogCategory::System, LogSeverity::Info, "Persistent");
        // Should not be removed even after a long time.
        logger.update(100.0, 6000);
        assert_eq!(logger.active_message_count(), 1);
    }

    #[test]
    fn test_screenshot_filename() {
        let mut capture = ScreenshotCapture::new("./screenshots");
        let path = capture.next_filename();
        assert!(path.to_str().unwrap().contains("screenshot"));
        assert!(path.to_str().unwrap().ends_with(".png"));
    }

    #[test]
    fn test_video_capture() {
        let mut video = VideoCaptureState::new();
        assert!(!video.recording);
        video.start_recording("./output.mp4");
        assert!(video.recording);
        video.capture_frame(1.0 / 30.0);
        assert_eq!(video.frames_captured, 1);
        video.stop_recording();
        assert!(!video.recording);
    }
}
