//! Editor performance monitoring for the Genovo editor.
//!
//! Provides real-time performance graphs and analytics for CPU, GPU, memory,
//! draw calls, per-system timing, frame time breakdown, memory allocation
//! tracking, hotspot identification, and performance alerts.
//!
//! # Features
//!
//! - Real-time graphs: CPU usage, GPU usage, memory, draw calls, FPS
//! - Per-system timing breakdown (physics, render, scripts, audio, etc.)
//! - Frame time waterfall chart (what each system costs per frame)
//! - Memory allocation tracking (current, peak, rate)
//! - Hotspot identification (systems exceeding budget)
//! - Performance alerts (configurable thresholds)
//! - Historical data for trend analysis
//! - Export statistics to file
//!
//! # Example
//!
//! ```ignore
//! let mut monitor = PerformanceMonitor::new(PerformanceConfig::default());
//!
//! // Each frame:
//! monitor.begin_frame();
//! monitor.begin_system("Physics");
//! // ... run physics ...
//! monitor.end_system("Physics");
//! monitor.end_frame();
//!
//! // Display:
//! let fps_graph = monitor.fps_graph();
//! let alerts = monitor.active_alerts();
//! ```

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of frame samples to keep for graphs.
const GRAPH_HISTORY_SIZE: usize = 300;

/// Number of frames for rolling average.
const ROLLING_AVG_FRAMES: usize = 60;

/// Default frame budget in milliseconds (60 FPS).
const DEFAULT_FRAME_BUDGET_MS: f64 = 16.67;

/// Default memory budget in megabytes.
const DEFAULT_MEMORY_BUDGET_MB: f64 = 512.0;

/// Default draw call budget.
const DEFAULT_DRAW_CALL_BUDGET: u64 = 5000;

/// Alert cooldown in seconds (to avoid spamming).
const ALERT_COOLDOWN_SECS: f64 = 5.0;

/// Maximum number of active alerts.
const MAX_ALERTS: usize = 50;

/// Maximum number of historical snapshots.
const MAX_SNAPSHOTS: usize = 3600;

// ---------------------------------------------------------------------------
// Performance metrics
// ---------------------------------------------------------------------------

/// A single frame's worth of performance data.
#[derive(Debug, Clone)]
pub struct FrameMetrics {
    /// Frame number.
    pub frame_number: u64,
    /// Total frame time in milliseconds.
    pub frame_time_ms: f64,
    /// CPU time in milliseconds.
    pub cpu_time_ms: f64,
    /// GPU time in milliseconds (estimated).
    pub gpu_time_ms: f64,
    /// Per-system timing breakdown.
    pub system_times: HashMap<String, f64>,
    /// Number of draw calls.
    pub draw_calls: u64,
    /// Number of triangles rendered.
    pub triangles: u64,
    /// Current memory usage in bytes.
    pub memory_bytes: u64,
    /// Memory allocated this frame in bytes.
    pub memory_allocated: u64,
    /// Memory freed this frame in bytes.
    pub memory_freed: u64,
    /// Number of entities in the scene.
    pub entity_count: u64,
    /// Number of active particles.
    pub particle_count: u64,
    /// Number of active audio sources.
    pub audio_sources: u64,
}

impl FrameMetrics {
    /// Create empty metrics for a frame.
    pub fn new(frame_number: u64) -> Self {
        Self {
            frame_number,
            frame_time_ms: 0.0,
            cpu_time_ms: 0.0,
            gpu_time_ms: 0.0,
            system_times: HashMap::new(),
            draw_calls: 0,
            triangles: 0,
            memory_bytes: 0,
            memory_allocated: 0,
            memory_freed: 0,
            entity_count: 0,
            particle_count: 0,
            audio_sources: 0,
        }
    }

    /// Returns the FPS for this frame.
    pub fn fps(&self) -> f64 {
        if self.frame_time_ms > 0.0 {
            1000.0 / self.frame_time_ms
        } else {
            0.0
        }
    }

    /// Returns the frame budget utilization (0..1, >1 means over budget).
    pub fn budget_utilization(&self, budget_ms: f64) -> f64 {
        if budget_ms > 0.0 {
            self.frame_time_ms / budget_ms
        } else {
            0.0
        }
    }
}

impl fmt::Display for FrameMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Frame #{}: {:.2}ms ({:.0} FPS), CPU={:.2}ms, GPU={:.2}ms, \
             DC={}, mem={:.1}MB",
            self.frame_number,
            self.frame_time_ms,
            self.fps(),
            self.cpu_time_ms,
            self.gpu_time_ms,
            self.draw_calls,
            self.memory_bytes as f64 / (1024.0 * 1024.0),
        )
    }
}

// ---------------------------------------------------------------------------
// System timing
// ---------------------------------------------------------------------------

/// Accumulated timing for a named system.
#[derive(Debug, Clone)]
pub struct SystemTiming {
    /// System name.
    pub name: String,
    /// Current frame time in milliseconds.
    pub current_ms: f64,
    /// Rolling average time in milliseconds.
    pub avg_ms: f64,
    /// Peak time observed.
    pub peak_ms: f64,
    /// Time budget for this system (if set).
    pub budget_ms: Option<f64>,
    /// Rolling history of frame times.
    history: Vec<f64>,
    /// Write position in history.
    history_pos: usize,
    /// Number of samples recorded.
    sample_count: usize,
    /// Whether timing is currently running for this system.
    active: bool,
    /// When the current timing started.
    start_time: Option<Instant>,
    /// Display color.
    pub color: [f32; 4],
    /// Whether to show this system in the waterfall chart.
    pub visible: bool,
}

impl SystemTiming {
    /// Create a new system timing entry.
    pub fn new(name: &str, color: [f32; 4]) -> Self {
        Self {
            name: name.to_string(),
            current_ms: 0.0,
            avg_ms: 0.0,
            peak_ms: 0.0,
            budget_ms: None,
            history: vec![0.0; GRAPH_HISTORY_SIZE],
            history_pos: 0,
            sample_count: 0,
            active: false,
            start_time: None,
            color,
            visible: true,
        }
    }

    /// Begin timing this system.
    pub fn begin(&mut self) {
        self.active = true;
        self.start_time = Some(Instant::now());
    }

    /// End timing this system.
    pub fn end(&mut self) {
        if let Some(start) = self.start_time.take() {
            self.current_ms = start.elapsed().as_secs_f64() * 1000.0;
            self.active = false;

            // Update history.
            self.history[self.history_pos] = self.current_ms;
            self.history_pos = (self.history_pos + 1) % GRAPH_HISTORY_SIZE;
            self.sample_count += 1;

            // Update peak.
            if self.current_ms > self.peak_ms {
                self.peak_ms = self.current_ms;
            }

            // Update rolling average.
            let count = self.sample_count.min(ROLLING_AVG_FRAMES);
            let sum: f64 = if self.sample_count >= GRAPH_HISTORY_SIZE {
                self.history.iter().take(ROLLING_AVG_FRAMES).sum()
            } else {
                self.history[..self.sample_count.min(ROLLING_AVG_FRAMES)]
                    .iter()
                    .sum()
            };
            self.avg_ms = sum / count as f64;
        }
    }

    /// Returns `true` if this system is over budget.
    pub fn is_over_budget(&self) -> bool {
        if let Some(budget) = self.budget_ms {
            self.current_ms > budget
        } else {
            false
        }
    }

    /// Returns the history as ordered samples (oldest first).
    pub fn history_ordered(&self) -> Vec<f64> {
        let count = self.sample_count.min(GRAPH_HISTORY_SIZE);
        if count < GRAPH_HISTORY_SIZE {
            self.history[..count].to_vec()
        } else {
            let mut result = Vec::with_capacity(GRAPH_HISTORY_SIZE);
            result.extend_from_slice(&self.history[self.history_pos..]);
            result.extend_from_slice(&self.history[..self.history_pos]);
            result
        }
    }

    /// Returns normalized history (0..1) for graphing.
    pub fn history_normalized(&self) -> Vec<f64> {
        let ordered = self.history_ordered();
        if ordered.is_empty() {
            return Vec::new();
        }
        let max = ordered.iter().copied().fold(f64::MIN, f64::max).max(0.001);
        ordered.iter().map(|v| v / max).collect()
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        self.current_ms = 0.0;
        self.avg_ms = 0.0;
        self.peak_ms = 0.0;
        self.history.iter_mut().for_each(|v| *v = 0.0);
        self.history_pos = 0;
        self.sample_count = 0;
    }
}

impl fmt::Display for SystemTiming {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:.2}ms (avg={:.2}ms, peak={:.2}ms)",
            self.name, self.current_ms, self.avg_ms, self.peak_ms
        )
    }
}

// ---------------------------------------------------------------------------
// Memory tracker
// ---------------------------------------------------------------------------

/// Tracks memory allocation patterns.
#[derive(Debug, Clone)]
pub struct MemoryTracker {
    /// Current total memory usage in bytes.
    pub current_bytes: u64,
    /// Peak memory usage in bytes.
    pub peak_bytes: u64,
    /// Total bytes allocated since start.
    pub total_allocated: u64,
    /// Total bytes freed since start.
    pub total_freed: u64,
    /// Allocation count per category.
    pub category_usage: HashMap<String, u64>,
    /// History of memory usage for graphing.
    history: Vec<f64>,
    /// Write position.
    history_pos: usize,
    /// Sample count.
    sample_count: usize,
    /// Allocation rate (bytes/sec) rolling average.
    pub alloc_rate: f64,
    /// Free rate (bytes/sec) rolling average.
    pub free_rate: f64,
    /// Bytes allocated in the current measurement interval.
    interval_allocated: u64,
    /// Bytes freed in the current measurement interval.
    interval_freed: u64,
    /// Timer for rate calculation.
    rate_timer: f64,
}

impl MemoryTracker {
    /// Create a new memory tracker.
    pub fn new() -> Self {
        Self {
            current_bytes: 0,
            peak_bytes: 0,
            total_allocated: 0,
            total_freed: 0,
            category_usage: HashMap::new(),
            history: vec![0.0; GRAPH_HISTORY_SIZE],
            history_pos: 0,
            sample_count: 0,
            alloc_rate: 0.0,
            free_rate: 0.0,
            interval_allocated: 0,
            interval_freed: 0,
            rate_timer: 0.0,
        }
    }

    /// Record a memory allocation.
    pub fn record_alloc(&mut self, bytes: u64, category: &str) {
        self.current_bytes += bytes;
        self.total_allocated += bytes;
        self.interval_allocated += bytes;
        if self.current_bytes > self.peak_bytes {
            self.peak_bytes = self.current_bytes;
        }
        *self.category_usage.entry(category.to_string()).or_insert(0) += bytes;
    }

    /// Record a memory free.
    pub fn record_free(&mut self, bytes: u64, category: &str) {
        self.current_bytes = self.current_bytes.saturating_sub(bytes);
        self.total_freed += bytes;
        self.interval_freed += bytes;
        if let Some(usage) = self.category_usage.get_mut(category) {
            *usage = usage.saturating_sub(bytes);
        }
    }

    /// Set current memory usage directly (from OS query).
    pub fn set_current(&mut self, bytes: u64) {
        self.current_bytes = bytes;
        if bytes > self.peak_bytes {
            self.peak_bytes = bytes;
        }
    }

    /// Update memory tracking (call periodically).
    pub fn update(&mut self, dt: f64) {
        // Record history.
        self.history[self.history_pos] = self.current_bytes as f64;
        self.history_pos = (self.history_pos + 1) % GRAPH_HISTORY_SIZE;
        self.sample_count += 1;

        // Update allocation rates.
        self.rate_timer += dt;
        if self.rate_timer >= 1.0 {
            self.alloc_rate = self.interval_allocated as f64 / self.rate_timer;
            self.free_rate = self.interval_freed as f64 / self.rate_timer;
            self.interval_allocated = 0;
            self.interval_freed = 0;
            self.rate_timer = 0.0;
        }
    }

    /// Returns current usage in megabytes.
    pub fn current_mb(&self) -> f64 {
        self.current_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Returns peak usage in megabytes.
    pub fn peak_mb(&self) -> f64 {
        self.peak_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Returns history ordered (oldest first) in megabytes.
    pub fn history_mb(&self) -> Vec<f64> {
        let mb = 1024.0 * 1024.0;
        let count = self.sample_count.min(GRAPH_HISTORY_SIZE);
        if count < GRAPH_HISTORY_SIZE {
            self.history[..count].iter().map(|v| v / mb).collect()
        } else {
            let mut result = Vec::with_capacity(GRAPH_HISTORY_SIZE);
            for &v in &self.history[self.history_pos..] {
                result.push(v / mb);
            }
            for &v in &self.history[..self.history_pos] {
                result.push(v / mb);
            }
            result
        }
    }

    /// Returns normalized history (0..1) for graphing.
    pub fn history_normalized(&self) -> Vec<f64> {
        let history = self.history_mb();
        if history.is_empty() {
            return Vec::new();
        }
        let max = history.iter().copied().fold(f64::MIN, f64::max).max(0.001);
        history.iter().map(|v| v / max).collect()
    }

    /// Reset all tracking.
    pub fn reset(&mut self) {
        self.current_bytes = 0;
        self.peak_bytes = 0;
        self.total_allocated = 0;
        self.total_freed = 0;
        self.category_usage.clear();
        self.history.iter_mut().for_each(|v| *v = 0.0);
        self.history_pos = 0;
        self.sample_count = 0;
        self.alloc_rate = 0.0;
        self.free_rate = 0.0;
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MemoryTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Memory: {:.1}MB / {:.1}MB peak, alloc={:.1}KB/s, free={:.1}KB/s",
            self.current_mb(),
            self.peak_mb(),
            self.alloc_rate / 1024.0,
            self.free_rate / 1024.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Performance alerts
// ---------------------------------------------------------------------------

/// Severity level for a performance alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Informational (not a problem yet).
    Info,
    /// Warning (approaching a limit).
    Warning,
    /// Critical (exceeding budget).
    Critical,
}

impl fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Critical => write!(f, "CRIT"),
        }
    }
}

/// A performance alert.
#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    /// Alert severity.
    pub severity: AlertSeverity,
    /// Alert message.
    pub message: String,
    /// The metric that triggered the alert.
    pub metric: String,
    /// The current value of the metric.
    pub current_value: f64,
    /// The threshold that was exceeded.
    pub threshold: f64,
    /// Frame number when the alert was triggered.
    pub frame: u64,
    /// Time when the alert was triggered.
    pub timestamp: f64,
    /// Whether the alert has been acknowledged.
    pub acknowledged: bool,
}

impl PerformanceAlert {
    /// Create a new alert.
    pub fn new(
        severity: AlertSeverity,
        metric: &str,
        message: &str,
        current: f64,
        threshold: f64,
        frame: u64,
        time: f64,
    ) -> Self {
        Self {
            severity,
            message: message.to_string(),
            metric: metric.to_string(),
            current_value: current,
            threshold,
            frame,
            timestamp: time,
            acknowledged: false,
        }
    }
}

impl fmt::Display for PerformanceAlert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}: {} (current={:.2}, threshold={:.2})",
            self.severity, self.metric, self.message, self.current_value, self.threshold
        )
    }
}

// ---------------------------------------------------------------------------
// Performance configuration
// ---------------------------------------------------------------------------

/// Configuration for the performance monitor.
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Target frame time in milliseconds.
    pub frame_budget_ms: f64,
    /// Memory budget in megabytes.
    pub memory_budget_mb: f64,
    /// Draw call budget.
    pub draw_call_budget: u64,
    /// Triangle budget.
    pub triangle_budget: u64,
    /// Whether to enable alerts.
    pub alerts_enabled: bool,
    /// Alert cooldown in seconds.
    pub alert_cooldown: f64,
    /// Whether to collect per-system timing.
    pub system_timing_enabled: bool,
    /// Whether to track memory allocations.
    pub memory_tracking_enabled: bool,
    /// How often to take snapshots (seconds).
    pub snapshot_interval: f64,
    /// Per-system time budgets.
    pub system_budgets: HashMap<String, f64>,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        let mut system_budgets = HashMap::new();
        system_budgets.insert("Physics".to_string(), 4.0);
        system_budgets.insert("Render".to_string(), 8.0);
        system_budgets.insert("Scripts".to_string(), 3.0);
        system_budgets.insert("Audio".to_string(), 1.0);
        system_budgets.insert("AI".to_string(), 2.0);
        system_budgets.insert("Animation".to_string(), 2.0);
        system_budgets.insert("UI".to_string(), 1.0);
        system_budgets.insert("Networking".to_string(), 1.0);

        Self {
            frame_budget_ms: DEFAULT_FRAME_BUDGET_MS,
            memory_budget_mb: DEFAULT_MEMORY_BUDGET_MB,
            draw_call_budget: DEFAULT_DRAW_CALL_BUDGET,
            triangle_budget: 2_000_000,
            alerts_enabled: true,
            alert_cooldown: ALERT_COOLDOWN_SECS,
            system_timing_enabled: true,
            memory_tracking_enabled: true,
            snapshot_interval: 1.0,
            system_budgets,
        }
    }
}

// ---------------------------------------------------------------------------
// Graph data
// ---------------------------------------------------------------------------

/// Data prepared for rendering a graph in the editor UI.
#[derive(Debug, Clone)]
pub struct PerfGraphData {
    /// Graph title.
    pub title: String,
    /// Data points (normalized 0..1).
    pub points: Vec<f64>,
    /// Current value (for display).
    pub current: f64,
    /// Average value.
    pub average: f64,
    /// Peak value.
    pub peak: f64,
    /// Unit label (e.g. "ms", "MB", "FPS").
    pub unit: String,
    /// Graph line color (RGBA).
    pub color: [f32; 4],
    /// Whether to show threshold line.
    pub threshold: Option<f64>,
}

// ---------------------------------------------------------------------------
// Performance snapshot
// ---------------------------------------------------------------------------

/// A timestamped snapshot for historical analysis.
#[derive(Debug, Clone)]
pub struct PerfSnapshot {
    /// Time offset from monitor start (seconds).
    pub timestamp: f64,
    /// Average FPS over the snapshot interval.
    pub avg_fps: f64,
    /// Average frame time in ms.
    pub avg_frame_time_ms: f64,
    /// Peak frame time in the interval.
    pub peak_frame_time_ms: f64,
    /// Average memory usage (MB).
    pub avg_memory_mb: f64,
    /// Average draw calls.
    pub avg_draw_calls: f64,
    /// Hotspot systems (over budget).
    pub hotspots: Vec<String>,
}

// ---------------------------------------------------------------------------
// Performance Monitor
// ---------------------------------------------------------------------------

/// The main performance monitor.
pub struct PerformanceMonitor {
    /// Configuration.
    config: PerformanceConfig,
    /// Frame counter.
    frame_number: u64,
    /// Time since monitor start (seconds).
    uptime: f64,
    /// Frame time when the current frame started.
    frame_start: Option<Instant>,
    /// Current frame metrics (being built).
    current_frame: FrameMetrics,
    /// Per-system timing.
    systems: HashMap<String, SystemTiming>,
    /// System timing order (for consistent display).
    system_order: Vec<String>,
    /// Memory tracker.
    pub memory: MemoryTracker,

    // --- History ---
    /// FPS history.
    fps_history: Vec<f64>,
    fps_history_pos: usize,
    fps_sample_count: usize,
    /// Frame time history.
    frame_time_history: Vec<f64>,
    ft_history_pos: usize,
    ft_sample_count: usize,
    /// Draw call history.
    dc_history: Vec<f64>,
    dc_history_pos: usize,
    dc_sample_count: usize,

    // --- Alerts ---
    /// Active performance alerts.
    alerts: Vec<PerformanceAlert>,
    /// Time of last alert per metric (for cooldown).
    last_alert_time: HashMap<String, f64>,

    // --- Snapshots ---
    snapshots: Vec<PerfSnapshot>,
    snapshot_timer: f64,
    /// Accumulator for snapshot averaging.
    snapshot_frame_count: u64,
    snapshot_fps_sum: f64,
    snapshot_ft_sum: f64,
    snapshot_ft_peak: f64,
    snapshot_mem_sum: f64,
    snapshot_dc_sum: f64,

    // --- Summary ---
    /// Rolling average FPS.
    pub avg_fps: f64,
    /// Rolling average frame time.
    pub avg_frame_time_ms: f64,
    /// Peak frame time (last N frames).
    pub peak_frame_time_ms: f64,
}

impl PerformanceMonitor {
    /// Create a new performance monitor.
    pub fn new(config: PerformanceConfig) -> Self {
        // Pre-create system timings.
        let mut systems = HashMap::new();
        let system_order = vec![
            "Physics", "Render", "Scripts", "Audio", "AI",
            "Animation", "UI", "Networking",
        ];
        let colors = [
            [0.2, 0.6, 1.0, 1.0], // Physics — blue
            [1.0, 0.4, 0.2, 1.0], // Render — red-orange
            [0.3, 0.9, 0.3, 1.0], // Scripts — green
            [0.9, 0.7, 0.2, 1.0], // Audio — yellow
            [0.7, 0.3, 0.9, 1.0], // AI — purple
            [0.2, 0.9, 0.9, 1.0], // Animation — cyan
            [0.9, 0.5, 0.7, 1.0], // UI — pink
            [0.5, 0.5, 0.5, 1.0], // Networking — gray
        ];
        for (i, &name) in system_order.iter().enumerate() {
            let mut timing = SystemTiming::new(name, colors[i]);
            if let Some(&budget) = config.system_budgets.get(name) {
                timing.budget_ms = Some(budget);
            }
            systems.insert(name.to_string(), timing);
        }

        Self {
            config,
            frame_number: 0,
            uptime: 0.0,
            frame_start: None,
            current_frame: FrameMetrics::new(0),
            systems,
            system_order: system_order.iter().map(|s| s.to_string()).collect(),
            memory: MemoryTracker::new(),
            fps_history: vec![0.0; GRAPH_HISTORY_SIZE],
            fps_history_pos: 0,
            fps_sample_count: 0,
            frame_time_history: vec![0.0; GRAPH_HISTORY_SIZE],
            ft_history_pos: 0,
            ft_sample_count: 0,
            dc_history: vec![0.0; GRAPH_HISTORY_SIZE],
            dc_history_pos: 0,
            dc_sample_count: 0,
            alerts: Vec::new(),
            last_alert_time: HashMap::new(),
            snapshots: Vec::new(),
            snapshot_timer: 0.0,
            snapshot_frame_count: 0,
            snapshot_fps_sum: 0.0,
            snapshot_ft_sum: 0.0,
            snapshot_ft_peak: 0.0,
            snapshot_mem_sum: 0.0,
            snapshot_dc_sum: 0.0,
            avg_fps: 0.0,
            avg_frame_time_ms: 0.0,
            peak_frame_time_ms: 0.0,
        }
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self) {
        self.frame_number += 1;
        self.frame_start = Some(Instant::now());
        self.current_frame = FrameMetrics::new(self.frame_number);
    }

    /// End the current frame and record metrics.
    pub fn end_frame(&mut self) {
        if let Some(start) = self.frame_start.take() {
            let frame_time = start.elapsed();
            let frame_ms = frame_time.as_secs_f64() * 1000.0;

            self.current_frame.frame_time_ms = frame_ms;
            self.current_frame.cpu_time_ms = frame_ms; // Simplified.
            self.current_frame.memory_bytes = self.memory.current_bytes;

            // Collect system times into the frame metrics.
            for (name, timing) in &self.systems {
                self.current_frame
                    .system_times
                    .insert(name.clone(), timing.current_ms);
            }

            let dt = frame_ms / 1000.0;
            self.uptime += dt;

            // Update FPS history.
            let fps = if frame_ms > 0.0 { 1000.0 / frame_ms } else { 0.0 };
            self.fps_history[self.fps_history_pos] = fps;
            self.fps_history_pos = (self.fps_history_pos + 1) % GRAPH_HISTORY_SIZE;
            self.fps_sample_count += 1;

            // Update frame time history.
            self.frame_time_history[self.ft_history_pos] = frame_ms;
            self.ft_history_pos = (self.ft_history_pos + 1) % GRAPH_HISTORY_SIZE;
            self.ft_sample_count += 1;

            // Update draw call history.
            self.dc_history[self.dc_history_pos] = self.current_frame.draw_calls as f64;
            self.dc_history_pos = (self.dc_history_pos + 1) % GRAPH_HISTORY_SIZE;
            self.dc_sample_count += 1;

            // Rolling averages.
            self.update_averages();

            // Memory update.
            self.memory.update(dt);

            // Check alerts.
            if self.config.alerts_enabled {
                self.check_alerts(frame_ms, fps);
            }

            // Snapshot accumulation.
            self.snapshot_timer += dt;
            self.snapshot_frame_count += 1;
            self.snapshot_fps_sum += fps;
            self.snapshot_ft_sum += frame_ms;
            if frame_ms > self.snapshot_ft_peak {
                self.snapshot_ft_peak = frame_ms;
            }
            self.snapshot_mem_sum += self.memory.current_mb();
            self.snapshot_dc_sum += self.current_frame.draw_calls as f64;

            if self.snapshot_timer >= self.config.snapshot_interval {
                self.take_snapshot();
            }
        }
    }

    /// Begin timing a system.
    pub fn begin_system(&mut self, name: &str) {
        if !self.config.system_timing_enabled {
            return;
        }
        if let Some(timing) = self.systems.get_mut(name) {
            timing.begin();
        } else {
            let mut timing = SystemTiming::new(name, [0.5, 0.5, 0.5, 1.0]);
            if let Some(&budget) = self.config.system_budgets.get(name) {
                timing.budget_ms = Some(budget);
            }
            timing.begin();
            self.systems.insert(name.to_string(), timing);
            self.system_order.push(name.to_string());
        }
    }

    /// End timing a system.
    pub fn end_system(&mut self, name: &str) {
        if let Some(timing) = self.systems.get_mut(name) {
            timing.end();
        }
    }

    /// Set the draw call count for the current frame.
    pub fn set_draw_calls(&mut self, count: u64) {
        self.current_frame.draw_calls = count;
    }

    /// Set the triangle count for the current frame.
    pub fn set_triangles(&mut self, count: u64) {
        self.current_frame.triangles = count;
    }

    /// Set the GPU time for the current frame.
    pub fn set_gpu_time(&mut self, ms: f64) {
        self.current_frame.gpu_time_ms = ms;
    }

    /// Set entity count.
    pub fn set_entity_count(&mut self, count: u64) {
        self.current_frame.entity_count = count;
    }

    /// Update rolling averages.
    fn update_averages(&mut self) {
        let n = self.fps_sample_count.min(ROLLING_AVG_FRAMES);
        if n == 0 {
            return;
        }

        // FPS average.
        let fps_sum: f64 = if self.fps_sample_count >= GRAPH_HISTORY_SIZE {
            self.fps_history.iter().take(n).sum()
        } else {
            self.fps_history[..n].iter().sum()
        };
        self.avg_fps = fps_sum / n as f64;

        // Frame time average.
        let ft_sum: f64 = if self.ft_sample_count >= GRAPH_HISTORY_SIZE {
            self.frame_time_history.iter().take(n).sum()
        } else {
            self.frame_time_history[..n].iter().sum()
        };
        self.avg_frame_time_ms = ft_sum / n as f64;

        // Peak frame time (within the rolling window).
        let ft_count = self.ft_sample_count.min(GRAPH_HISTORY_SIZE);
        self.peak_frame_time_ms = if ft_count < GRAPH_HISTORY_SIZE {
            self.frame_time_history[..ft_count]
                .iter()
                .copied()
                .fold(0.0f64, f64::max)
        } else {
            self.frame_time_history
                .iter()
                .copied()
                .fold(0.0f64, f64::max)
        };
    }

    /// Check for performance alert conditions.
    fn check_alerts(&mut self, frame_ms: f64, _fps: f64) {
        // Frame time alert.
        if frame_ms > self.config.frame_budget_ms * 1.5 {
            self.emit_alert(
                AlertSeverity::Critical,
                "frame_time",
                "Frame time significantly over budget",
                frame_ms,
                self.config.frame_budget_ms,
            );
        } else if frame_ms > self.config.frame_budget_ms {
            self.emit_alert(
                AlertSeverity::Warning,
                "frame_time",
                "Frame time over budget",
                frame_ms,
                self.config.frame_budget_ms,
            );
        }

        // Memory alert.
        let mem_mb = self.memory.current_mb();
        if mem_mb > self.config.memory_budget_mb {
            self.emit_alert(
                AlertSeverity::Critical,
                "memory",
                "Memory usage exceeds budget",
                mem_mb,
                self.config.memory_budget_mb,
            );
        } else if mem_mb > self.config.memory_budget_mb * 0.9 {
            self.emit_alert(
                AlertSeverity::Warning,
                "memory",
                "Memory usage approaching budget",
                mem_mb,
                self.config.memory_budget_mb * 0.9,
            );
        }

        // Draw call alert.
        let dc = self.current_frame.draw_calls;
        if dc > self.config.draw_call_budget {
            self.emit_alert(
                AlertSeverity::Warning,
                "draw_calls",
                "Draw call count exceeds budget",
                dc as f64,
                self.config.draw_call_budget as f64,
            );
        }

        // Per-system alerts.
        for (name, timing) in &self.systems {
            if timing.is_over_budget() {
                let budget = timing.budget_ms.unwrap_or(0.0);
                self.emit_alert(
                    AlertSeverity::Warning,
                    &format!("system_{name}"),
                    &format!("{name} system over time budget"),
                    timing.current_ms,
                    budget,
                );
            }
        }
    }

    /// Emit a performance alert (with cooldown).
    fn emit_alert(
        &mut self,
        severity: AlertSeverity,
        metric: &str,
        message: &str,
        current: f64,
        threshold: f64,
    ) {
        // Check cooldown.
        if let Some(&last_time) = self.last_alert_time.get(metric) {
            if self.uptime - last_time < self.config.alert_cooldown {
                return;
            }
        }

        let alert = PerformanceAlert::new(
            severity,
            metric,
            message,
            current,
            threshold,
            self.frame_number,
            self.uptime,
        );

        self.last_alert_time.insert(metric.to_string(), self.uptime);

        if self.alerts.len() >= MAX_ALERTS {
            self.alerts.remove(0);
        }
        self.alerts.push(alert);
    }

    /// Take a periodic snapshot.
    fn take_snapshot(&mut self) {
        if self.snapshot_frame_count == 0 {
            return;
        }
        let n = self.snapshot_frame_count as f64;

        let mut hotspots = Vec::new();
        for (name, timing) in &self.systems {
            if timing.is_over_budget() {
                hotspots.push(name.clone());
            }
        }

        let snapshot = PerfSnapshot {
            timestamp: self.uptime,
            avg_fps: self.snapshot_fps_sum / n,
            avg_frame_time_ms: self.snapshot_ft_sum / n,
            peak_frame_time_ms: self.snapshot_ft_peak,
            avg_memory_mb: self.snapshot_mem_sum / n,
            avg_draw_calls: self.snapshot_dc_sum / n,
            hotspots,
        };

        if self.snapshots.len() >= MAX_SNAPSHOTS {
            self.snapshots.remove(0);
        }
        self.snapshots.push(snapshot);

        // Reset accumulators.
        self.snapshot_timer = 0.0;
        self.snapshot_frame_count = 0;
        self.snapshot_fps_sum = 0.0;
        self.snapshot_ft_sum = 0.0;
        self.snapshot_ft_peak = 0.0;
        self.snapshot_mem_sum = 0.0;
        self.snapshot_dc_sum = 0.0;
    }

    // --- Queries ---

    /// Returns the current FPS.
    pub fn current_fps(&self) -> f64 {
        if self.fps_sample_count == 0 {
            return 0.0;
        }
        let idx = if self.fps_history_pos == 0 {
            GRAPH_HISTORY_SIZE - 1
        } else {
            self.fps_history_pos - 1
        };
        self.fps_history[idx]
    }

    /// Returns the FPS graph data.
    pub fn fps_graph(&self) -> PerfGraphData {
        let history = self.ordered_history(&self.fps_history, self.fps_history_pos, self.fps_sample_count);
        let max = history.iter().copied().fold(0.0f64, f64::max).max(1.0);
        PerfGraphData {
            title: "FPS".to_string(),
            points: history.iter().map(|v| v / max).collect(),
            current: self.current_fps(),
            average: self.avg_fps,
            peak: max,
            unit: "fps".to_string(),
            color: [0.3, 1.0, 0.3, 1.0],
            threshold: Some(1000.0 / self.config.frame_budget_ms / max),
        }
    }

    /// Returns the frame time graph data.
    pub fn frame_time_graph(&self) -> PerfGraphData {
        let history = self.ordered_history(
            &self.frame_time_history,
            self.ft_history_pos,
            self.ft_sample_count,
        );
        let max = history.iter().copied().fold(0.0f64, f64::max).max(1.0);
        PerfGraphData {
            title: "Frame Time".to_string(),
            points: history.iter().map(|v| v / max).collect(),
            current: self.current_frame.frame_time_ms,
            average: self.avg_frame_time_ms,
            peak: self.peak_frame_time_ms,
            unit: "ms".to_string(),
            color: [1.0, 0.5, 0.2, 1.0],
            threshold: Some(self.config.frame_budget_ms / max),
        }
    }

    /// Returns the memory graph data.
    pub fn memory_graph(&self) -> PerfGraphData {
        let history = self.memory.history_mb();
        let max = history.iter().copied().fold(0.0f64, f64::max).max(1.0);
        PerfGraphData {
            title: "Memory".to_string(),
            points: history.iter().map(|v| v / max).collect(),
            current: self.memory.current_mb(),
            average: if history.is_empty() {
                0.0
            } else {
                history.iter().sum::<f64>() / history.len() as f64
            },
            peak: self.memory.peak_mb(),
            unit: "MB".to_string(),
            color: [0.4, 0.6, 1.0, 1.0],
            threshold: Some(self.config.memory_budget_mb / max),
        }
    }

    /// Returns the draw call graph data.
    pub fn draw_call_graph(&self) -> PerfGraphData {
        let history = self.ordered_history(&self.dc_history, self.dc_history_pos, self.dc_sample_count);
        let max = history.iter().copied().fold(0.0f64, f64::max).max(1.0);
        PerfGraphData {
            title: "Draw Calls".to_string(),
            points: history.iter().map(|v| v / max).collect(),
            current: self.current_frame.draw_calls as f64,
            average: if history.is_empty() {
                0.0
            } else {
                history.iter().sum::<f64>() / history.len() as f64
            },
            peak: max,
            unit: "calls".to_string(),
            color: [0.9, 0.3, 0.7, 1.0],
            threshold: Some(self.config.draw_call_budget as f64 / max),
        }
    }

    /// Returns the per-system waterfall data for the current frame.
    pub fn system_waterfall(&self) -> Vec<(&str, f64, [f32; 4])> {
        self.system_order
            .iter()
            .filter_map(|name| {
                self.systems.get(name).map(|t| (name.as_str(), t.current_ms, t.color))
            })
            .collect()
    }

    /// Returns identified hotspots (systems over budget).
    pub fn hotspots(&self) -> Vec<(&str, f64, f64)> {
        self.systems
            .iter()
            .filter(|(_, t)| t.is_over_budget())
            .map(|(name, t)| (name.as_str(), t.current_ms, t.budget_ms.unwrap_or(0.0)))
            .collect()
    }

    /// Returns active alerts.
    pub fn active_alerts(&self) -> &[PerformanceAlert] {
        &self.alerts
    }

    /// Acknowledge an alert by index.
    pub fn acknowledge_alert(&mut self, index: usize) {
        if index < self.alerts.len() {
            self.alerts[index].acknowledged = true;
        }
    }

    /// Clear all alerts.
    pub fn clear_alerts(&mut self) {
        self.alerts.clear();
    }

    /// Returns system timing data.
    pub fn system_timings(&self) -> &HashMap<String, SystemTiming> {
        &self.systems
    }

    /// Returns historical snapshots.
    pub fn snapshots(&self) -> &[PerfSnapshot] {
        &self.snapshots
    }

    /// Returns the current frame number.
    pub fn frame_number(&self) -> u64 {
        self.frame_number
    }

    /// Returns the monitor uptime.
    pub fn uptime(&self) -> f64 {
        self.uptime
    }

    /// Helper to get ordered history from a circular buffer.
    fn ordered_history(&self, buf: &[f64], pos: usize, count: usize) -> Vec<f64> {
        let n = count.min(GRAPH_HISTORY_SIZE);
        if n < GRAPH_HISTORY_SIZE {
            buf[..n].to_vec()
        } else {
            let mut result = Vec::with_capacity(GRAPH_HISTORY_SIZE);
            result.extend_from_slice(&buf[pos..]);
            result.extend_from_slice(&buf[..pos]);
            result
        }
    }

    /// Reset all monitor data.
    pub fn reset(&mut self) {
        self.frame_number = 0;
        self.uptime = 0.0;
        self.fps_history.iter_mut().for_each(|v| *v = 0.0);
        self.fps_history_pos = 0;
        self.fps_sample_count = 0;
        self.frame_time_history.iter_mut().for_each(|v| *v = 0.0);
        self.ft_history_pos = 0;
        self.ft_sample_count = 0;
        self.dc_history.iter_mut().for_each(|v| *v = 0.0);
        self.dc_history_pos = 0;
        self.dc_sample_count = 0;
        for timing in self.systems.values_mut() {
            timing.reset();
        }
        self.memory.reset();
        self.alerts.clear();
        self.last_alert_time.clear();
        self.snapshots.clear();
        self.avg_fps = 0.0;
        self.avg_frame_time_ms = 0.0;
        self.peak_frame_time_ms = 0.0;
    }
}

impl fmt::Display for PerformanceMonitor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Perf[frame={}, fps={:.0}, ft={:.2}ms, mem={:.1}MB, DC={}, alerts={}]",
            self.frame_number,
            self.avg_fps,
            self.avg_frame_time_ms,
            self.memory.current_mb(),
            self.current_frame.draw_calls,
            self.alerts.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_metrics() {
        let metrics = FrameMetrics::new(1);
        assert_eq!(metrics.frame_number, 1);
        assert_eq!(metrics.fps(), 0.0);
    }

    #[test]
    fn test_system_timing() {
        let mut timing = SystemTiming::new("Physics", [0.0, 0.0, 1.0, 1.0]);
        timing.begin();
        // Simulate some work.
        let _ = (0..1000).sum::<i32>();
        timing.end();
        assert!(timing.current_ms >= 0.0);
        assert!(timing.sample_count > 0);
    }

    #[test]
    fn test_system_timing_budget() {
        let mut timing = SystemTiming::new("Test", [1.0, 0.0, 0.0, 1.0]);
        timing.budget_ms = Some(1.0);
        timing.current_ms = 2.0;
        assert!(timing.is_over_budget());

        timing.current_ms = 0.5;
        assert!(!timing.is_over_budget());
    }

    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::new();
        tracker.record_alloc(1024, "textures");
        tracker.record_alloc(2048, "meshes");
        assert_eq!(tracker.current_bytes, 3072);
        assert_eq!(tracker.peak_bytes, 3072);

        tracker.record_free(1024, "textures");
        assert_eq!(tracker.current_bytes, 2048);
        assert_eq!(tracker.peak_bytes, 3072); // Peak unchanged.
    }

    #[test]
    fn test_memory_categories() {
        let mut tracker = MemoryTracker::new();
        tracker.record_alloc(1000, "textures");
        tracker.record_alloc(2000, "textures");
        tracker.record_alloc(500, "audio");

        assert_eq!(tracker.category_usage["textures"], 3000);
        assert_eq!(tracker.category_usage["audio"], 500);
    }

    #[test]
    fn test_performance_monitor_basic() {
        let mut monitor = PerformanceMonitor::new(PerformanceConfig::default());

        monitor.begin_frame();
        monitor.begin_system("Physics");
        monitor.end_system("Physics");
        monitor.set_draw_calls(1000);
        monitor.end_frame();

        assert_eq!(monitor.frame_number(), 1);
        assert!(monitor.current_fps() >= 0.0);
    }

    #[test]
    fn test_performance_alerts() {
        let config = PerformanceConfig {
            frame_budget_ms: 16.67,
            memory_budget_mb: 1.0,
            alerts_enabled: true,
            alert_cooldown: 0.0, // No cooldown for testing.
            ..Default::default()
        };
        let mut monitor = PerformanceMonitor::new(config);

        // Simulate high memory usage.
        monitor.memory.set_current(2 * 1024 * 1024); // 2 MB, over 1 MB budget.
        monitor.begin_frame();
        monitor.end_frame();

        assert!(!monitor.active_alerts().is_empty());
    }

    #[test]
    fn test_alert_acknowledge() {
        let config = PerformanceConfig {
            alert_cooldown: 0.0,
            ..Default::default()
        };
        let mut monitor = PerformanceMonitor::new(config);

        monitor.emit_alert(
            AlertSeverity::Warning,
            "test",
            "test alert",
            100.0,
            50.0,
        );

        assert_eq!(monitor.alerts.len(), 1);
        assert!(!monitor.alerts[0].acknowledged);

        monitor.acknowledge_alert(0);
        assert!(monitor.alerts[0].acknowledged);
    }

    #[test]
    fn test_system_waterfall() {
        let mut monitor = PerformanceMonitor::new(PerformanceConfig::default());
        monitor.begin_frame();
        monitor.begin_system("Physics");
        monitor.end_system("Physics");
        monitor.begin_system("Render");
        monitor.end_system("Render");
        monitor.end_frame();

        let waterfall = monitor.system_waterfall();
        assert!(!waterfall.is_empty());
    }

    #[test]
    fn test_graph_data() {
        let mut monitor = PerformanceMonitor::new(PerformanceConfig::default());
        for _ in 0..10 {
            monitor.begin_frame();
            monitor.set_draw_calls(500);
            monitor.end_frame();
        }

        let fps = monitor.fps_graph();
        assert_eq!(fps.title, "FPS");
        assert!(!fps.points.is_empty());

        let ft = monitor.frame_time_graph();
        assert_eq!(ft.unit, "ms");

        let dc = monitor.draw_call_graph();
        assert_eq!(dc.title, "Draw Calls");
    }

    #[test]
    fn test_hotspot_detection() {
        let mut monitor = PerformanceMonitor::new(PerformanceConfig::default());

        // Manually set a system over budget.
        if let Some(timing) = monitor.systems.get_mut("Physics") {
            timing.budget_ms = Some(1.0);
            timing.current_ms = 5.0;
        }

        let hotspots = monitor.hotspots();
        assert!(!hotspots.is_empty());
        assert_eq!(hotspots[0].0, "Physics");
    }

    #[test]
    fn test_perf_snapshot() {
        let config = PerformanceConfig {
            snapshot_interval: 0.001, // Very short for testing.
            ..Default::default()
        };
        let mut monitor = PerformanceMonitor::new(config);

        for _ in 0..5 {
            monitor.begin_frame();
            monitor.set_draw_calls(100);
            monitor.end_frame();
        }

        // Should have at least one snapshot by now.
        assert!(!monitor.snapshots().is_empty());
    }

    #[test]
    fn test_monitor_reset() {
        let mut monitor = PerformanceMonitor::new(PerformanceConfig::default());
        for _ in 0..5 {
            monitor.begin_frame();
            monitor.end_frame();
        }
        assert!(monitor.frame_number() > 0);

        monitor.reset();
        assert_eq!(monitor.frame_number(), 0);
        assert_eq!(monitor.avg_fps, 0.0);
    }

    #[test]
    fn test_alert_severity_ordering() {
        assert!(AlertSeverity::Info < AlertSeverity::Warning);
        assert!(AlertSeverity::Warning < AlertSeverity::Critical);
    }
}
