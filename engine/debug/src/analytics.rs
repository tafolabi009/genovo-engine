//! Analytics and Telemetry
//!
//! Provides event tracking, metric recording, session management, heatmap
//! generation, and data export for both development profiling and player
//! behavior analysis.
//!
//! # Data Collection
//!
//! The analytics system collects several categories of data:
//!
//! - **Events**: Named occurrences with arbitrary string properties.
//! - **Metrics**: Numeric values tracked over time (FPS, frame time, etc.).
//! - **Sessions**: Start/end times with duration and termination reason.
//! - **Heatmaps**: 2D grids accumulating position frequency data.
//!
//! All data is stored in memory using circular buffers to limit memory
//! consumption. Data can be exported to CSV or JSON for offline analysis.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// AnalyticsEvent
// ---------------------------------------------------------------------------

/// A single tracked event with name, timestamp, and properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsEvent {
    /// Name of the event (e.g., "player_kill", "level_complete").
    pub name: String,
    /// Timestamp in seconds since session start.
    pub timestamp: f64,
    /// Frame number when the event occurred.
    pub frame: u64,
    /// Arbitrary key-value properties.
    pub properties: HashMap<String, String>,
}

impl AnalyticsEvent {
    /// Create a new event.
    pub fn new(name: impl Into<String>, timestamp: f64, frame: u64) -> Self {
        Self {
            name: name.into(),
            timestamp,
            frame,
            properties: HashMap::new(),
        }
    }

    /// Add a property to this event.
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Estimated memory usage.
    pub fn estimated_size(&self) -> usize {
        let mut size = 48 + self.name.len();
        for (k, v) in &self.properties {
            size += k.len() + v.len() + 16;
        }
        size
    }
}

// ---------------------------------------------------------------------------
// MetricSample
// ---------------------------------------------------------------------------

/// A single metric sample.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MetricSample {
    /// Value of the metric.
    pub value: f64,
    /// Timestamp when the sample was recorded.
    pub timestamp: f64,
    /// Frame number.
    pub frame: u64,
}

// ---------------------------------------------------------------------------
// MetricStats
// ---------------------------------------------------------------------------

/// Running statistics for a metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStats {
    /// Name of the metric.
    pub name: String,
    /// Total number of samples.
    pub count: u64,
    /// Running sum.
    pub sum: f64,
    /// Running sum of squares (for variance).
    pub sum_sq: f64,
    /// Minimum value observed.
    pub min: f64,
    /// Maximum value observed.
    pub max: f64,
    /// Most recent value.
    pub last: f64,
    /// Exponential moving average.
    pub ema: f64,
    /// EMA smoothing factor.
    pub ema_alpha: f64,
}

impl MetricStats {
    /// Create new metric statistics.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            min: f64::MAX,
            max: f64::MIN,
            last: 0.0,
            ema: 0.0,
            ema_alpha: 0.05,
        }
    }

    /// Record a new sample.
    pub fn record(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_sq += value * value;
        self.last = value;

        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }

        if self.count == 1 {
            self.ema = value;
        } else {
            self.ema = self.ema * (1.0 - self.ema_alpha) + value * self.ema_alpha;
        }
    }

    /// Calculate the mean.
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum / self.count as f64
    }

    /// Calculate the variance.
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let mean = self.mean();
        self.sum_sq / self.count as f64 - mean * mean
    }

    /// Calculate the standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        self.count = 0;
        self.sum = 0.0;
        self.sum_sq = 0.0;
        self.min = f64::MAX;
        self.max = f64::MIN;
        self.last = 0.0;
        self.ema = 0.0;
    }
}

// ---------------------------------------------------------------------------
// MetricTimeSeries
// ---------------------------------------------------------------------------

/// Time-series data for a metric, stored in a circular buffer.
#[derive(Debug, Clone)]
pub struct MetricTimeSeries {
    /// Name of the metric.
    pub name: String,
    /// Circular buffer of samples.
    samples: Vec<MetricSample>,
    /// Write head in the circular buffer.
    head: usize,
    /// Number of samples stored (up to capacity).
    count: usize,
    /// Capacity of the circular buffer.
    capacity: usize,
    /// Running statistics.
    pub stats: MetricStats,
}

impl MetricTimeSeries {
    /// Create a new time series with the given capacity.
    pub fn new(name: impl Into<String>, capacity: usize) -> Self {
        let name = name.into();
        Self {
            stats: MetricStats::new(name.clone()),
            name,
            samples: Vec::with_capacity(capacity),
            head: 0,
            count: 0,
            capacity,
        }
    }

    /// Record a sample.
    pub fn record(&mut self, value: f64, timestamp: f64, frame: u64) {
        let sample = MetricSample {
            value,
            timestamp,
            frame,
        };

        if self.samples.len() < self.capacity {
            self.samples.push(sample);
        } else {
            self.samples[self.head] = sample;
        }

        self.head = (self.head + 1) % self.capacity;
        self.count += 1;
        self.stats.record(value);
    }

    /// Get the most recent N samples.
    pub fn recent_samples(&self, n: usize) -> Vec<MetricSample> {
        let stored = self.samples.len();
        let take = n.min(stored);

        let mut result = Vec::with_capacity(take);
        for i in 0..take {
            let idx = if self.head >= take {
                self.head - take + i
            } else {
                (self.capacity + self.head - take + i) % self.capacity
            };
            if idx < stored {
                result.push(self.samples[idx]);
            }
        }
        result
    }

    /// Get all stored samples in chronological order.
    pub fn all_samples(&self) -> Vec<MetricSample> {
        let stored = self.samples.len();
        if stored < self.capacity {
            self.samples.clone()
        } else {
            let mut result = Vec::with_capacity(stored);
            for i in 0..stored {
                let idx = (self.head + i) % self.capacity;
                result.push(self.samples[idx]);
            }
            result
        }
    }

    /// Number of samples stored.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Total number of samples ever recorded.
    pub fn total_count(&self) -> u64 {
        self.stats.count
    }

    /// Clear all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.head = 0;
        self.count = 0;
        self.stats.reset();
    }

    /// Calculate the percentile value from stored samples.
    pub fn percentile(&self, p: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }

        let mut sorted: Vec<f64> = self.samples.iter().map(|s| s.value).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
}

// ---------------------------------------------------------------------------
// HeatmapData
// ---------------------------------------------------------------------------

/// 2D grid that accumulates frequency data at world positions.
///
/// Used for tracking player position frequency, death locations, item
/// pickup hotspots, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapData {
    /// Name of this heatmap.
    pub name: String,
    /// Grid of accumulated values.
    pub grid: Vec<f32>,
    /// Width of the grid (number of cells in X).
    pub width: usize,
    /// Height of the grid (number of cells in Z).
    pub height: usize,
    /// World-space minimum X coordinate.
    pub world_min_x: f32,
    /// World-space minimum Z coordinate.
    pub world_min_z: f32,
    /// World-space maximum X coordinate.
    pub world_max_x: f32,
    /// World-space maximum Z coordinate.
    pub world_max_z: f32,
    /// Cell size in world units.
    pub cell_size: f32,
    /// Total number of samples added.
    pub total_samples: u64,
    /// Peak value in the grid.
    pub peak_value: f32,
}

impl HeatmapData {
    /// Create a new heatmap covering a world-space region.
    ///
    /// # Parameters
    ///
    /// - `name`: Name for this heatmap.
    /// - `min_x`, `min_z`: Minimum world coordinates.
    /// - `max_x`, `max_z`: Maximum world coordinates.
    /// - `cell_size`: Size of each grid cell in world units.
    pub fn new(
        name: impl Into<String>,
        min_x: f32,
        min_z: f32,
        max_x: f32,
        max_z: f32,
        cell_size: f32,
    ) -> Self {
        let width = ((max_x - min_x) / cell_size).ceil() as usize;
        let height = ((max_z - min_z) / cell_size).ceil() as usize;

        Self {
            name: name.into(),
            grid: vec![0.0; width * height],
            width,
            height,
            world_min_x: min_x,
            world_min_z: min_z,
            world_max_x: max_x,
            world_max_z: max_z,
            cell_size,
            total_samples: 0,
            peak_value: 0.0,
        }
    }

    /// Increment the heatmap at a world-space position.
    pub fn increment(&mut self, x: f32, z: f32) {
        self.add_value(x, z, 1.0);
    }

    /// Add a value at a world-space position.
    pub fn add_value(&mut self, x: f32, z: f32, value: f32) {
        if let Some(idx) = self.world_to_index(x, z) {
            self.grid[idx] += value;
            if self.grid[idx] > self.peak_value {
                self.peak_value = self.grid[idx];
            }
            self.total_samples += 1;
        }
    }

    /// Add a value with Gaussian splat (affects neighboring cells).
    pub fn add_gaussian(&mut self, x: f32, z: f32, value: f32, radius: f32) {
        let cx = ((x - self.world_min_x) / self.cell_size).floor() as i32;
        let cz = ((z - self.world_min_z) / self.cell_size).floor() as i32;
        let cell_radius = (radius / self.cell_size).ceil() as i32;

        let sigma_sq = (radius * 0.5) * (radius * 0.5);

        for dz in -cell_radius..=cell_radius {
            for dx in -cell_radius..=cell_radius {
                let gx = cx + dx;
                let gz = cz + dz;

                if gx >= 0 && gx < self.width as i32 && gz >= 0 && gz < self.height as i32 {
                    let world_gx = self.world_min_x + (gx as f32 + 0.5) * self.cell_size;
                    let world_gz = self.world_min_z + (gz as f32 + 0.5) * self.cell_size;

                    let ddx = world_gx - x;
                    let ddz = world_gz - z;
                    let dist_sq = ddx * ddx + ddz * ddz;

                    let weight = (-dist_sq / (2.0 * sigma_sq)).exp();
                    let idx = gz as usize * self.width + gx as usize;

                    self.grid[idx] += value * weight;
                    if self.grid[idx] > self.peak_value {
                        self.peak_value = self.grid[idx];
                    }
                }
            }
        }

        self.total_samples += 1;
    }

    /// Get the value at a world-space position.
    pub fn value_at(&self, x: f32, z: f32) -> f32 {
        self.world_to_index(x, z)
            .map(|idx| self.grid[idx])
            .unwrap_or(0.0)
    }

    /// Convert world coordinates to grid index.
    fn world_to_index(&self, x: f32, z: f32) -> Option<usize> {
        if x < self.world_min_x || x >= self.world_max_x
            || z < self.world_min_z || z >= self.world_max_z
        {
            return None;
        }

        let gx = ((x - self.world_min_x) / self.cell_size).floor() as usize;
        let gz = ((z - self.world_min_z) / self.cell_size).floor() as usize;

        if gx < self.width && gz < self.height {
            Some(gz * self.width + gx)
        } else {
            None
        }
    }

    /// Get the normalized value (0..1) at a grid cell.
    pub fn normalized_value(&self, x: usize, z: usize) -> f32 {
        if self.peak_value <= 0.0 || x >= self.width || z >= self.height {
            return 0.0;
        }
        self.grid[z * self.width + x] / self.peak_value
    }

    /// Clear all heatmap data.
    pub fn clear(&mut self) {
        self.grid.fill(0.0);
        self.total_samples = 0;
        self.peak_value = 0.0;
    }

    /// Get the sum of all values.
    pub fn total_value(&self) -> f64 {
        self.grid.iter().map(|&v| v as f64).sum()
    }

    /// Export heatmap as a flat array of normalized floats (for rendering).
    pub fn as_normalized_grid(&self) -> Vec<f32> {
        if self.peak_value <= 0.0 {
            return vec![0.0; self.grid.len()];
        }
        self.grid.iter().map(|v| v / self.peak_value).collect()
    }
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// Tracks a play session from start to end.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique session identifier.
    pub id: String,
    /// Start time (ISO 8601 or timestamp).
    pub start_time: String,
    /// Duration in seconds (updated as the session progresses).
    pub duration: f64,
    /// Reason the session ended.
    pub end_reason: SessionEndReason,
    /// Map / level name.
    pub map_name: String,
    /// Game mode.
    pub game_mode: String,
    /// Player name.
    pub player_name: String,
    /// Custom session properties.
    pub properties: HashMap<String, String>,
    /// Whether the session is still active.
    pub active: bool,
}

/// Why a session ended.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SessionEndReason {
    /// Session is still active.
    Active,
    /// Player quit normally.
    NormalQuit,
    /// Player disconnected.
    Disconnect,
    /// Game crashed.
    Crash,
    /// Session timed out.
    Timeout,
    /// Server shut down.
    ServerShutdown,
    /// Other reason.
    Other(String),
}

impl Default for SessionEndReason {
    fn default() -> Self {
        SessionEndReason::Active
    }
}

impl Session {
    /// Create a new active session.
    pub fn new(
        id: impl Into<String>,
        player_name: impl Into<String>,
        map_name: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            start_time: String::new(),
            duration: 0.0,
            end_reason: SessionEndReason::Active,
            map_name: map_name.into(),
            game_mode: String::new(),
            player_name: player_name.into(),
            properties: HashMap::new(),
            active: true,
        }
    }

    /// End the session with a reason.
    pub fn end(&mut self, reason: SessionEndReason) {
        self.end_reason = reason;
        self.active = false;
    }

    /// Update the session duration.
    pub fn update_duration(&mut self, dt: f64) {
        if self.active {
            self.duration += dt;
        }
    }
}

// ---------------------------------------------------------------------------
// PerformanceMetrics
// ---------------------------------------------------------------------------

/// Pre-defined performance metric names.
pub struct PerfMetrics;

impl PerfMetrics {
    pub const FPS: &'static str = "perf.fps";
    pub const FRAME_TIME_MS: &'static str = "perf.frame_time_ms";
    pub const FRAME_TIME_P99_MS: &'static str = "perf.frame_time_p99_ms";
    pub const GPU_TIME_MS: &'static str = "perf.gpu_time_ms";
    pub const GPU_MEMORY_MB: &'static str = "perf.gpu_memory_mb";
    pub const CPU_USAGE: &'static str = "perf.cpu_usage";
    pub const DRAW_CALLS: &'static str = "perf.draw_calls";
    pub const TRIANGLES: &'static str = "perf.triangles";
    pub const ENTITIES: &'static str = "perf.entities";
    pub const MEMORY_MB: &'static str = "perf.memory_mb";
}

// ---------------------------------------------------------------------------
// Analytics
// ---------------------------------------------------------------------------

/// Central analytics system for tracking events, metrics, sessions, and
/// heatmaps.
///
/// # Usage
///
/// ```ignore
/// let mut analytics = Analytics::new(10000);
///
/// // Track events.
/// analytics.track_event("player_kill", &[("weapon", "rifle"), ("distance", "42.5")]);
///
/// // Track metrics.
/// analytics.track_metric("fps", 60.0);
/// analytics.track_metric("frame_time_ms", 16.67);
///
/// // Session management.
/// analytics.start_session("player1", "map_01");
///
/// // Heatmaps.
/// analytics.create_heatmap("deaths", -5000.0, -5000.0, 5000.0, 5000.0, 10.0);
/// analytics.record_heatmap("deaths", 123.0, 456.0);
///
/// // Export.
/// analytics.export_csv("analytics.csv");
/// analytics.export_json("analytics.json");
/// ```
pub struct Analytics {
    /// Event circular buffer.
    events: Vec<AnalyticsEvent>,
    /// Write head for event buffer.
    event_head: usize,
    /// Maximum number of events to keep in memory.
    max_events: usize,
    /// Total number of events ever recorded.
    total_events: u64,
    /// Metric time series, keyed by name.
    metrics: HashMap<String, MetricTimeSeries>,
    /// Default capacity for new metric time series.
    default_metric_capacity: usize,
    /// Named heatmaps.
    heatmaps: HashMap<String, HeatmapData>,
    /// Current active session.
    current_session: Option<Session>,
    /// Session history.
    session_history: Vec<Session>,
    /// Current timestamp (seconds since analytics start).
    current_time: f64,
    /// Current frame number.
    current_frame: u64,
    /// Action frequency counter: action_name -> count.
    action_counts: HashMap<String, u64>,
    /// Whether analytics is enabled.
    enabled: bool,
}

impl Analytics {
    /// Create a new analytics system.
    ///
    /// # Parameters
    ///
    /// - `max_events`: Maximum number of events to keep in the circular buffer.
    pub fn new(max_events: usize) -> Self {
        Self {
            events: Vec::with_capacity(max_events.min(1024)),
            event_head: 0,
            max_events,
            total_events: 0,
            metrics: HashMap::new(),
            default_metric_capacity: 3600, // 1 minute at 60 FPS
            heatmaps: HashMap::new(),
            current_session: None,
            session_history: Vec::new(),
            current_time: 0.0,
            current_frame: 0,
            action_counts: HashMap::new(),
            enabled: true,
        }
    }

    /// Enable or disable analytics.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Update timestamp and frame counter.
    pub fn update(&mut self, dt: f64) {
        self.current_time += dt;
        self.current_frame += 1;

        if let Some(ref mut session) = self.current_session {
            session.update_duration(dt);
        }
    }

    // -- Event tracking ----------------------------------------------------

    /// Track a named event with properties.
    pub fn track_event(&mut self, name: &str, properties: &[(&str, &str)]) {
        if !self.enabled {
            return;
        }

        let mut event = AnalyticsEvent::new(name, self.current_time, self.current_frame);
        for &(k, v) in properties {
            event.properties.insert(k.to_owned(), v.to_owned());
        }

        self.push_event(event);
    }

    /// Track a named event with a HashMap of properties.
    pub fn track_event_map(&mut self, name: &str, properties: HashMap<String, String>) {
        if !self.enabled {
            return;
        }

        let event = AnalyticsEvent {
            name: name.to_owned(),
            timestamp: self.current_time,
            frame: self.current_frame,
            properties,
        };

        self.push_event(event);
    }

    /// Push an event into the circular buffer.
    fn push_event(&mut self, event: AnalyticsEvent) {
        if self.events.len() < self.max_events {
            self.events.push(event);
        } else {
            self.events[self.event_head] = event;
        }
        self.event_head = (self.event_head + 1) % self.max_events;
        self.total_events += 1;
    }

    /// Get all stored events.
    pub fn events(&self) -> &[AnalyticsEvent] {
        &self.events
    }

    /// Get the total number of events ever recorded.
    pub fn total_events(&self) -> u64 {
        self.total_events
    }

    /// Get events matching a name filter.
    pub fn events_by_name(&self, name: &str) -> Vec<&AnalyticsEvent> {
        self.events.iter().filter(|e| e.name == name).collect()
    }

    /// Track an action (increments a counter).
    pub fn track_action(&mut self, action: &str) {
        if !self.enabled {
            return;
        }
        *self.action_counts.entry(action.to_owned()).or_insert(0) += 1;
    }

    /// Get the count for an action.
    pub fn action_count(&self, action: &str) -> u64 {
        self.action_counts.get(action).copied().unwrap_or(0)
    }

    /// Get all action counts.
    pub fn action_counts(&self) -> &HashMap<String, u64> {
        &self.action_counts
    }

    // -- Metric tracking ---------------------------------------------------

    /// Track a numeric metric.
    pub fn track_metric(&mut self, name: &str, value: f64) {
        if !self.enabled {
            return;
        }

        let timestamp = self.current_time;
        let frame = self.current_frame;
        let capacity = self.default_metric_capacity;

        self.metrics
            .entry(name.to_owned())
            .or_insert_with(|| MetricTimeSeries::new(name, capacity))
            .record(value, timestamp, frame);
    }

    /// Get metric statistics.
    pub fn metric_stats(&self, name: &str) -> Option<&MetricStats> {
        self.metrics.get(name).map(|m| &m.stats)
    }

    /// Get metric time series.
    pub fn metric_series(&self, name: &str) -> Option<&MetricTimeSeries> {
        self.metrics.get(name)
    }

    /// Get the current value of a metric.
    pub fn metric_value(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).map(|m| m.stats.last)
    }

    /// Get the average value of a metric.
    pub fn metric_average(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).map(|m| m.stats.mean())
    }

    /// Get the P99 value of a metric from stored samples.
    pub fn metric_p99(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).map(|m| m.percentile(99.0))
    }

    /// Get all metric names.
    pub fn metric_names(&self) -> Vec<&str> {
        self.metrics.keys().map(|s| s.as_str()).collect()
    }

    /// Clear a specific metric.
    pub fn clear_metric(&mut self, name: &str) {
        if let Some(m) = self.metrics.get_mut(name) {
            m.clear();
        }
    }

    // -- Performance shortcuts ---------------------------------------------

    /// Record standard performance metrics for this frame.
    pub fn record_frame_performance(&mut self, fps: f64, frame_time_ms: f64) {
        self.track_metric(PerfMetrics::FPS, fps);
        self.track_metric(PerfMetrics::FRAME_TIME_MS, frame_time_ms);
    }

    /// Record GPU performance metrics.
    pub fn record_gpu_performance(
        &mut self,
        gpu_time_ms: f64,
        gpu_memory_mb: f64,
        draw_calls: u32,
        triangles: u64,
    ) {
        self.track_metric(PerfMetrics::GPU_TIME_MS, gpu_time_ms);
        self.track_metric(PerfMetrics::GPU_MEMORY_MB, gpu_memory_mb);
        self.track_metric(PerfMetrics::DRAW_CALLS, draw_calls as f64);
        self.track_metric(PerfMetrics::TRIANGLES, triangles as f64);
    }

    // -- Session management ------------------------------------------------

    /// Start a new session.
    pub fn start_session(&mut self, player_name: &str, map_name: &str) {
        // End the current session if active.
        if let Some(mut session) = self.current_session.take() {
            session.end(SessionEndReason::Other("new session started".to_owned()));
            self.session_history.push(session);
        }

        let session_id = format!("session_{}", self.total_events);
        let session = Session::new(session_id, player_name, map_name);

        log::info!(
            "Analytics session started: player={}, map={}",
            player_name,
            map_name,
        );

        self.current_session = Some(session);
    }

    /// End the current session.
    pub fn end_session(&mut self, reason: SessionEndReason) {
        if let Some(mut session) = self.current_session.take() {
            log::info!(
                "Analytics session ended: duration={:.1}s, reason={:?}",
                session.duration,
                reason,
            );
            session.end(reason);
            self.session_history.push(session);
        }
    }

    /// Get the current session.
    pub fn current_session(&self) -> Option<&Session> {
        self.current_session.as_ref()
    }

    /// Get session history.
    pub fn session_history(&self) -> &[Session] {
        &self.session_history
    }

    // -- Heatmap management ------------------------------------------------

    /// Create a new heatmap.
    pub fn create_heatmap(
        &mut self,
        name: &str,
        min_x: f32,
        min_z: f32,
        max_x: f32,
        max_z: f32,
        cell_size: f32,
    ) {
        let heatmap = HeatmapData::new(name, min_x, min_z, max_x, max_z, cell_size);
        self.heatmaps.insert(name.to_owned(), heatmap);
    }

    /// Record a position on a heatmap.
    pub fn record_heatmap(&mut self, name: &str, x: f32, z: f32) {
        if !self.enabled {
            return;
        }
        if let Some(heatmap) = self.heatmaps.get_mut(name) {
            heatmap.increment(x, z);
        }
    }

    /// Record a position with Gaussian splat on a heatmap.
    pub fn record_heatmap_gaussian(&mut self, name: &str, x: f32, z: f32, radius: f32) {
        if !self.enabled {
            return;
        }
        if let Some(heatmap) = self.heatmaps.get_mut(name) {
            heatmap.add_gaussian(x, z, 1.0, radius);
        }
    }

    /// Get a heatmap by name.
    pub fn heatmap(&self, name: &str) -> Option<&HeatmapData> {
        self.heatmaps.get(name)
    }

    /// Get a mutable heatmap by name.
    pub fn heatmap_mut(&mut self, name: &str) -> Option<&mut HeatmapData> {
        self.heatmaps.get_mut(name)
    }

    /// Clear a heatmap.
    pub fn clear_heatmap(&mut self, name: &str) {
        if let Some(heatmap) = self.heatmaps.get_mut(name) {
            heatmap.clear();
        }
    }

    // -- Export -------------------------------------------------------------

    /// Export all metrics to CSV format.
    pub fn export_csv(&self, path: &Path) -> Result<(), String> {
        let mut output = String::new();

        // Header.
        output.push_str("type,name,timestamp,frame,value,properties\n");

        // Events.
        for event in &self.events {
            let props: Vec<String> = event
                .properties
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            output.push_str(&format!(
                "event,{},{:.6},{},{},{}\n",
                event.name,
                event.timestamp,
                event.frame,
                "",
                props.join(";"),
            ));
        }

        // Metrics.
        for (name, series) in &self.metrics {
            for sample in series.all_samples() {
                output.push_str(&format!(
                    "metric,{},{:.6},{},{:.6},\n",
                    name, sample.timestamp, sample.frame, sample.value,
                ));
            }
        }

        // Metric summaries.
        for (name, series) in &self.metrics {
            let stats = &series.stats;
            output.push_str(&format!(
                "metric_summary,{},0,0,,count={};mean={:.4};min={:.4};max={:.4};stddev={:.4}\n",
                name, stats.count, stats.mean(), stats.min, stats.max, stats.std_dev(),
            ));
        }

        // Sessions.
        for session in &self.session_history {
            output.push_str(&format!(
                "session,{},{},0,{:.2},player={};map={};reason={:?}\n",
                session.id,
                session.start_time,
                session.duration,
                session.player_name,
                session.map_name,
                session.end_reason,
            ));
        }

        // Action counts.
        for (action, count) in &self.action_counts {
            output.push_str(&format!(
                "action,{},0,0,{},\n",
                action, count,
            ));
        }

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        std::fs::write(path, output)
            .map_err(|e| format!("Failed to write CSV file: {}", e))?;

        log::info!("Analytics exported to CSV: {}", path.display());
        Ok(())
    }

    /// Export all analytics data to JSON format.
    pub fn export_json(&self, path: &Path) -> Result<(), String> {
        let export = AnalyticsExport {
            events: self.events.clone(),
            metrics: self
                .metrics
                .iter()
                .map(|(name, series)| {
                    (
                        name.clone(),
                        MetricExport {
                            stats: series.stats.clone(),
                            samples: series.all_samples(),
                        },
                    )
                })
                .collect(),
            sessions: self.session_history.clone(),
            heatmaps: self.heatmaps.clone(),
            action_counts: self.action_counts.clone(),
            total_events: self.total_events,
        };

        let json = serde_json::to_string_pretty(&export)
            .map_err(|e| format!("Failed to serialize analytics: {}", e))?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        std::fs::write(path, json)
            .map_err(|e| format!("Failed to write JSON file: {}", e))?;

        log::info!("Analytics exported to JSON: {}", path.display());
        Ok(())
    }

    /// Reset all analytics data.
    pub fn reset(&mut self) {
        self.events.clear();
        self.event_head = 0;
        self.total_events = 0;
        self.metrics.clear();
        self.heatmaps.clear();
        self.action_counts.clear();
        self.current_session = None;
        self.session_history.clear();
        self.current_time = 0.0;
        self.current_frame = 0;
    }
}

// ---------------------------------------------------------------------------
// Export types
// ---------------------------------------------------------------------------

/// Serializable analytics export.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnalyticsExport {
    events: Vec<AnalyticsEvent>,
    metrics: HashMap<String, MetricExport>,
    sessions: Vec<Session>,
    heatmaps: HashMap<String, HeatmapData>,
    action_counts: HashMap<String, u64>,
    total_events: u64,
}

/// Serializable metric export.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MetricExport {
    stats: MetricStats,
    samples: Vec<MetricSample>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_event() {
        let mut analytics = Analytics::new(100);
        analytics.track_event("test_event", &[("key", "value")]);
        assert_eq!(analytics.total_events(), 1);
        assert_eq!(analytics.events().len(), 1);
        assert_eq!(analytics.events()[0].name, "test_event");
        assert_eq!(analytics.events()[0].properties.get("key").unwrap(), "value");
    }

    #[test]
    fn circular_buffer_wraps() {
        let mut analytics = Analytics::new(5);
        for i in 0..10 {
            analytics.track_event(&format!("event_{}", i), &[]);
        }
        assert_eq!(analytics.total_events(), 10);
        assert_eq!(analytics.events().len(), 5);
    }

    #[test]
    fn track_metric() {
        let mut analytics = Analytics::new(100);
        for i in 0..100 {
            analytics.track_metric("fps", 60.0 + (i % 5) as f64);
        }
        let stats = analytics.metric_stats("fps").unwrap();
        assert_eq!(stats.count, 100);
        assert!(stats.mean() > 59.0 && stats.mean() < 63.0);
        assert!(stats.min >= 60.0);
        assert!(stats.max <= 64.0);
    }

    #[test]
    fn metric_percentile() {
        let mut analytics = Analytics::new(100);
        for i in 0..100 {
            analytics.track_metric("test", i as f64);
        }
        let p99 = analytics.metric_p99("test").unwrap();
        assert!(p99 >= 98.0);
    }

    #[test]
    fn heatmap_increment() {
        let mut analytics = Analytics::new(100);
        analytics.create_heatmap("test", 0.0, 0.0, 100.0, 100.0, 10.0);

        analytics.record_heatmap("test", 15.0, 15.0);
        analytics.record_heatmap("test", 15.0, 15.0);

        let heatmap = analytics.heatmap("test").unwrap();
        assert_eq!(heatmap.value_at(15.0, 15.0), 2.0);
        assert_eq!(heatmap.value_at(50.0, 50.0), 0.0);
    }

    #[test]
    fn heatmap_gaussian() {
        let mut analytics = Analytics::new(100);
        analytics.create_heatmap("test", 0.0, 0.0, 100.0, 100.0, 5.0);

        analytics.record_heatmap_gaussian("test", 50.0, 50.0, 15.0);

        let heatmap = analytics.heatmap("test").unwrap();
        let center = heatmap.value_at(50.0, 50.0);
        let edge = heatmap.value_at(35.0, 50.0);
        assert!(center > edge);
    }

    #[test]
    fn session_lifecycle() {
        let mut analytics = Analytics::new(100);

        analytics.start_session("player1", "map_01");
        assert!(analytics.current_session().is_some());

        analytics.update(10.0);
        assert!(analytics.current_session().unwrap().duration >= 10.0);

        analytics.end_session(SessionEndReason::NormalQuit);
        assert!(analytics.current_session().is_none());
        assert_eq!(analytics.session_history().len(), 1);
    }

    #[test]
    fn action_tracking() {
        let mut analytics = Analytics::new(100);
        analytics.track_action("jump");
        analytics.track_action("jump");
        analytics.track_action("shoot");

        assert_eq!(analytics.action_count("jump"), 2);
        assert_eq!(analytics.action_count("shoot"), 1);
        assert_eq!(analytics.action_count("crouch"), 0);
    }

    #[test]
    fn metric_stats_math() {
        let mut stats = MetricStats::new("test");
        stats.record(10.0);
        stats.record(20.0);
        stats.record(30.0);

        assert_eq!(stats.count, 3);
        assert!((stats.mean() - 20.0).abs() < 0.01);
        assert!(stats.min <= 10.0);
        assert!(stats.max >= 30.0);
        assert!(stats.std_dev() > 0.0);
    }

    #[test]
    fn heatmap_out_of_bounds() {
        let heatmap = HeatmapData::new("test", 0.0, 0.0, 100.0, 100.0, 10.0);
        assert_eq!(heatmap.value_at(-10.0, 50.0), 0.0);
        assert_eq!(heatmap.value_at(110.0, 50.0), 0.0);
    }

    #[test]
    fn disabled_analytics_no_tracking() {
        let mut analytics = Analytics::new(100);
        analytics.set_enabled(false);
        analytics.track_event("test", &[]);
        analytics.track_metric("fps", 60.0);
        assert_eq!(analytics.total_events(), 0);
        assert!(analytics.metric_stats("fps").is_none());
    }
}
