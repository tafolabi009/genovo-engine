//! GPU profiler for tracking GPU-side timing, pipeline statistics, and memory.
//!
//! This module provides a software-side abstraction for GPU profiling. Actual
//! GPU timestamp queries would be issued through the rendering backend; this
//! module collects and presents the results.
//!
//! # Design
//!
//! The [`GpuProfiler`] maintains a set of named [`GpuScope`] entries. The
//! rendering backend calls [`GpuProfiler::begin_scope`] and
//! [`GpuProfiler::end_scope`] when it issues GPU timestamp queries, and later
//! resolves the results via [`GpuProfiler::resolve_scope`]. At the end of the
//! frame, [`GpuProfiler::end_frame`] finalizes the data and pushes it into a
//! rolling history.

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use parking_lot::Mutex;

// ---------------------------------------------------------------------------
// GpuScope
// ---------------------------------------------------------------------------

/// A single GPU timing scope.
#[derive(Debug, Clone)]
pub struct GpuScope {
    /// Human-readable name for this GPU scope.
    pub name: String,
    /// Measured GPU duration (resolved from timestamp queries).
    pub duration: Duration,
    /// Begin timestamp in nanoseconds (GPU clock).
    pub begin_ns: u64,
    /// End timestamp in nanoseconds (GPU clock).
    pub end_ns: u64,
    /// Whether the scope has been resolved (timestamps read back).
    pub resolved: bool,
}

impl GpuScope {
    /// Create a new unresolved GPU scope.
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            duration: Duration::ZERO,
            begin_ns: 0,
            end_ns: 0,
            resolved: false,
        }
    }

    /// Resolve the scope with the given begin/end timestamps (in nanoseconds).
    fn resolve(&mut self, begin_ns: u64, end_ns: u64) {
        self.begin_ns = begin_ns;
        self.end_ns = end_ns;
        self.duration = Duration::from_nanos(end_ns.saturating_sub(begin_ns));
        self.resolved = true;
    }
}

// ---------------------------------------------------------------------------
// GpuPipelineStats
// ---------------------------------------------------------------------------

/// GPU pipeline statistics counters for a single frame.
#[derive(Debug, Clone, Default)]
pub struct GpuPipelineStats {
    /// Number of vertices processed by the input assembler.
    pub vertex_count: u64,
    /// Number of primitives (triangles) generated.
    pub primitive_count: u64,
    /// Number of fragment shader invocations.
    pub fragment_count: u64,
    /// Number of compute shader invocations.
    pub compute_invocations: u64,
    /// Number of draw calls issued.
    pub draw_calls: u32,
    /// Number of dispatch calls issued.
    pub dispatch_calls: u32,
    /// Number of render passes.
    pub render_passes: u32,
    /// Number of pipeline state changes.
    pub pipeline_changes: u32,
    /// Number of descriptor set binds.
    pub descriptor_binds: u32,
    /// Number of buffer uploads / transfers.
    pub buffer_transfers: u32,
}

impl GpuPipelineStats {
    /// Reset all counters to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Increment draw call count.
    pub fn record_draw_call(&mut self) {
        self.draw_calls += 1;
    }

    /// Increment dispatch call count.
    pub fn record_dispatch(&mut self) {
        self.dispatch_calls += 1;
    }

    /// Record vertex/primitive counts for a draw call.
    pub fn record_geometry(&mut self, vertices: u64, primitives: u64) {
        self.vertex_count += vertices;
        self.primitive_count += primitives;
    }

    /// Record fragment invocations.
    pub fn record_fragments(&mut self, count: u64) {
        self.fragment_count += count;
    }

    /// Record a pipeline state change.
    pub fn record_pipeline_change(&mut self) {
        self.pipeline_changes += 1;
    }

    /// Record a render pass.
    pub fn record_render_pass(&mut self) {
        self.render_passes += 1;
    }

    /// Record a buffer transfer.
    pub fn record_buffer_transfer(&mut self) {
        self.buffer_transfers += 1;
    }
}

impl fmt::Display for GpuPipelineStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GPU Pipeline Statistics:")?;
        writeln!(f, "  Vertices:       {}", self.vertex_count)?;
        writeln!(f, "  Primitives:     {}", self.primitive_count)?;
        writeln!(f, "  Fragments:      {}", self.fragment_count)?;
        writeln!(f, "  Compute Invoc:  {}", self.compute_invocations)?;
        writeln!(f, "  Draw Calls:     {}", self.draw_calls)?;
        writeln!(f, "  Dispatches:     {}", self.dispatch_calls)?;
        writeln!(f, "  Render Passes:  {}", self.render_passes)?;
        writeln!(f, "  Pipeline Binds: {}", self.pipeline_changes)?;
        writeln!(f, "  Descriptor Binds:{}", self.descriptor_binds)?;
        writeln!(f, "  Buffer Xfers:   {}", self.buffer_transfers)
    }
}

// ---------------------------------------------------------------------------
// GpuFrameTimeBreakdown
// ---------------------------------------------------------------------------

/// Per-frame GPU time broken down by pipeline stage.
#[derive(Debug, Clone, Default)]
pub struct GpuFrameTimeBreakdown {
    /// Time spent in vertex processing.
    pub vertex_time: Duration,
    /// Time spent in fragment / pixel processing.
    pub fragment_time: Duration,
    /// Time spent in compute shaders.
    pub compute_time: Duration,
    /// Time spent in transfer / copy operations.
    pub transfer_time: Duration,
    /// Total GPU frame time.
    pub total_time: Duration,
}

impl GpuFrameTimeBreakdown {
    /// Compute total from components.
    pub fn compute_total(&mut self) {
        self.total_time =
            self.vertex_time + self.fragment_time + self.compute_time + self.transfer_time;
    }

    /// Reset all durations to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl fmt::Display for GpuFrameTimeBreakdown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GPU Frame Time Breakdown:")?;
        writeln!(
            f,
            "  Vertex:   {:.3}ms",
            self.vertex_time.as_secs_f64() * 1000.0
        )?;
        writeln!(
            f,
            "  Fragment: {:.3}ms",
            self.fragment_time.as_secs_f64() * 1000.0
        )?;
        writeln!(
            f,
            "  Compute:  {:.3}ms",
            self.compute_time.as_secs_f64() * 1000.0
        )?;
        writeln!(
            f,
            "  Transfer: {:.3}ms",
            self.transfer_time.as_secs_f64() * 1000.0
        )?;
        writeln!(
            f,
            "  Total:    {:.3}ms",
            self.total_time.as_secs_f64() * 1000.0
        )
    }
}

// ---------------------------------------------------------------------------
// GpuMemoryUsage
// ---------------------------------------------------------------------------

/// Tracks GPU memory usage across different categories.
#[derive(Debug, Clone, Default)]
pub struct GpuMemoryUsage {
    /// Total device-local (VRAM) memory in bytes.
    pub device_local_total: u64,
    /// Used device-local memory in bytes.
    pub device_local_used: u64,
    /// Total host-visible (shared/upload) memory in bytes.
    pub host_visible_total: u64,
    /// Used host-visible memory in bytes.
    pub host_visible_used: u64,
    /// Memory used by textures.
    pub texture_memory: u64,
    /// Memory used by vertex/index buffers.
    pub buffer_memory: u64,
    /// Memory used by render targets / framebuffers.
    pub render_target_memory: u64,
    /// Memory used by staging / upload buffers.
    pub staging_memory: u64,
    /// Number of active allocations.
    pub allocation_count: u32,
}

impl GpuMemoryUsage {
    /// Total used memory (device-local + host-visible).
    pub fn total_used(&self) -> u64 {
        self.device_local_used + self.host_visible_used
    }

    /// Total available memory.
    pub fn total_available(&self) -> u64 {
        self.device_local_total + self.host_visible_total
    }

    /// Usage percentage of device-local memory.
    pub fn device_local_usage_pct(&self) -> f64 {
        if self.device_local_total == 0 {
            return 0.0;
        }
        self.device_local_used as f64 / self.device_local_total as f64 * 100.0
    }

    /// Format a byte count as a human-readable string.
    fn format_bytes(bytes: u64) -> String {
        const KB: u64 = 1024;
        const MB: u64 = 1024 * 1024;
        const GB: u64 = 1024 * 1024 * 1024;
        if bytes >= GB {
            format!("{:.2} GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.2} MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.2} KB", bytes as f64 / KB as f64)
        } else {
            format!("{} B", bytes)
        }
    }
}

impl fmt::Display for GpuMemoryUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GPU Memory Usage:")?;
        writeln!(
            f,
            "  Device Local: {} / {} ({:.1}%)",
            Self::format_bytes(self.device_local_used),
            Self::format_bytes(self.device_local_total),
            self.device_local_usage_pct(),
        )?;
        writeln!(
            f,
            "  Host Visible: {} / {}",
            Self::format_bytes(self.host_visible_used),
            Self::format_bytes(self.host_visible_total),
        )?;
        writeln!(f, "  Textures:       {}", Self::format_bytes(self.texture_memory))?;
        writeln!(f, "  Buffers:        {}", Self::format_bytes(self.buffer_memory))?;
        writeln!(
            f,
            "  Render Targets: {}",
            Self::format_bytes(self.render_target_memory)
        )?;
        writeln!(f, "  Staging:        {}", Self::format_bytes(self.staging_memory))?;
        writeln!(f, "  Allocations:    {}", self.allocation_count)
    }
}

// ---------------------------------------------------------------------------
// GpuProfileFrame
// ---------------------------------------------------------------------------

/// A complete frame of GPU profiling data.
#[derive(Debug, Clone)]
pub struct GpuProfileFrame {
    /// Frame index.
    pub frame_index: u64,
    /// Resolved GPU scopes.
    pub scopes: Vec<GpuScope>,
    /// Pipeline statistics for this frame.
    pub pipeline_stats: GpuPipelineStats,
    /// Time breakdown by stage.
    pub time_breakdown: GpuFrameTimeBreakdown,
    /// Memory usage snapshot.
    pub memory_usage: GpuMemoryUsage,
}

// ---------------------------------------------------------------------------
// GpuProfileReport
// ---------------------------------------------------------------------------

/// Formatted GPU profile report.
#[derive(Debug, Clone)]
pub struct GpuProfileReport {
    /// Report title.
    pub title: String,
    /// Scope timing lines.
    pub scope_lines: Vec<GpuScopeReportLine>,
    /// Averaged pipeline stats.
    pub avg_stats: GpuPipelineStats,
    /// Averaged time breakdown.
    pub avg_breakdown: GpuFrameTimeBreakdown,
    /// Latest memory snapshot.
    pub memory: GpuMemoryUsage,
    /// Number of frames in the report.
    pub frame_count: usize,
}

/// A line in the GPU scope report.
#[derive(Debug, Clone)]
pub struct GpuScopeReportLine {
    /// Scope name.
    pub name: String,
    /// Average GPU time.
    pub avg_duration: Duration,
    /// Maximum GPU time.
    pub max_duration: Duration,
    /// Percentage of total GPU frame time.
    pub pct_of_frame: f64,
}

impl fmt::Display for GpuProfileReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== {} ===", self.title)?;
        writeln!(f, "Frames sampled: {}", self.frame_count)?;
        writeln!(f)?;

        writeln!(
            f,
            "{:<30} {:>10} {:>10} {:>8}",
            "GPU Scope", "Avg (ms)", "Max (ms)", "% Frame"
        )?;
        writeln!(f, "{}", "-".repeat(62))?;
        for line in &self.scope_lines {
            writeln!(
                f,
                "{:<30} {:>10.3} {:>10.3} {:>7.1}%",
                line.name,
                line.avg_duration.as_secs_f64() * 1000.0,
                line.max_duration.as_secs_f64() * 1000.0,
                line.pct_of_frame,
            )?;
        }

        writeln!(f)?;
        write!(f, "{}", self.avg_breakdown)?;
        writeln!(f)?;
        write!(f, "{}", self.avg_stats)?;
        writeln!(f)?;
        write!(f, "{}", self.memory)?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GpuProfiler
// ---------------------------------------------------------------------------

/// GPU profiler that tracks timing scopes, pipeline statistics, and memory
/// usage.
///
/// # Usage
///
/// ```ignore
/// let mut gpu_profiler = GpuProfiler::new();
///
/// gpu_profiler.begin_frame();
///
/// // Issue a GPU timestamp query, get back an opaque scope id.
/// let scope = gpu_profiler.begin_scope("ShadowPass");
/// // ... render shadow maps ...
/// gpu_profiler.end_scope(scope);
///
/// // After the GPU results are available (1-2 frames later):
/// gpu_profiler.resolve_scope("ShadowPass", begin_ns, end_ns);
///
/// gpu_profiler.end_frame();
/// ```
pub struct GpuProfiler {
    /// Active (current frame) scopes being timed.
    active_scopes: Mutex<HashMap<String, GpuScope>>,
    /// Current frame pipeline statistics.
    current_stats: Mutex<GpuPipelineStats>,
    /// Current frame time breakdown.
    current_breakdown: Mutex<GpuFrameTimeBreakdown>,
    /// Latest memory usage snapshot.
    memory_usage: Mutex<GpuMemoryUsage>,
    /// Rolling history of completed GPU frames.
    history: Mutex<Vec<GpuProfileFrame>>,
    /// Maximum history size.
    history_size: usize,
    /// Frame counter.
    frame_counter: u64,
    /// Whether the profiler is enabled.
    enabled: bool,
}

impl GpuProfiler {
    /// Create a new GPU profiler with default history (120 frames).
    pub fn new() -> Self {
        Self::with_history_size(120)
    }

    /// Create a GPU profiler with a custom history size.
    pub fn with_history_size(size: usize) -> Self {
        Self {
            active_scopes: Mutex::new(HashMap::new()),
            current_stats: Mutex::new(GpuPipelineStats::default()),
            current_breakdown: Mutex::new(GpuFrameTimeBreakdown::default()),
            memory_usage: Mutex::new(GpuMemoryUsage::default()),
            history: Mutex::new(Vec::with_capacity(size)),
            history_size: size,
            frame_counter: 0,
            enabled: true,
        }
    }

    /// Enable or disable the GPU profiler.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Returns `true` if the GPU profiler is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Begin a new GPU profiling frame.
    pub fn begin_frame(&mut self) {
        if !self.enabled {
            return;
        }
        self.active_scopes.lock().clear();
        self.current_stats.lock().reset();
        self.current_breakdown.lock().reset();
    }

    /// Begin a named GPU scope. Returns the scope name for use with
    /// `end_scope`.
    pub fn begin_scope(&self, name: &str) -> String {
        if !self.enabled {
            return name.to_string();
        }
        let scope = GpuScope::new(name);
        self.active_scopes
            .lock()
            .insert(name.to_string(), scope);
        name.to_string()
    }

    /// End a GPU scope (mark the end timestamp query position).
    pub fn end_scope(&self, _name: &str) {
        // In a real implementation, this would record the position of the
        // end timestamp query. The actual resolution happens asynchronously.
    }

    /// Resolve a GPU scope with actual timestamp values from GPU readback.
    pub fn resolve_scope(&self, name: &str, begin_ns: u64, end_ns: u64) {
        if let Some(scope) = self.active_scopes.lock().get_mut(name) {
            scope.resolve(begin_ns, end_ns);
        }
    }

    /// Record a draw call.
    pub fn record_draw_call(&self, vertex_count: u64, primitive_count: u64) {
        let mut stats = self.current_stats.lock();
        stats.record_draw_call();
        stats.record_geometry(vertex_count, primitive_count);
    }

    /// Record a compute dispatch.
    pub fn record_dispatch(&self) {
        self.current_stats.lock().record_dispatch();
    }

    /// Record fragment shader invocations.
    pub fn record_fragments(&self, count: u64) {
        self.current_stats.lock().record_fragments(count);
    }

    /// Record a pipeline state change.
    pub fn record_pipeline_change(&self) {
        self.current_stats.lock().record_pipeline_change();
    }

    /// Record a render pass begin.
    pub fn record_render_pass(&self) {
        self.current_stats.lock().record_render_pass();
    }

    /// Record a buffer transfer.
    pub fn record_buffer_transfer(&self) {
        self.current_stats.lock().record_buffer_transfer();
    }

    /// Update GPU time breakdown for a specific stage.
    pub fn set_vertex_time(&self, duration: Duration) {
        self.current_breakdown.lock().vertex_time = duration;
    }

    /// Update fragment time in the breakdown.
    pub fn set_fragment_time(&self, duration: Duration) {
        self.current_breakdown.lock().fragment_time = duration;
    }

    /// Update compute time in the breakdown.
    pub fn set_compute_time(&self, duration: Duration) {
        self.current_breakdown.lock().compute_time = duration;
    }

    /// Update transfer time in the breakdown.
    pub fn set_transfer_time(&self, duration: Duration) {
        self.current_breakdown.lock().transfer_time = duration;
    }

    /// Update memory usage snapshot.
    pub fn update_memory(&self, usage: GpuMemoryUsage) {
        *self.memory_usage.lock() = usage;
    }

    /// Get the current memory usage snapshot.
    pub fn get_memory_usage(&self) -> GpuMemoryUsage {
        self.memory_usage.lock().clone()
    }

    /// End the current GPU profiling frame: finalize data and push to history.
    pub fn end_frame(&mut self) {
        if !self.enabled {
            return;
        }

        let scopes: Vec<GpuScope> = self
            .active_scopes
            .lock()
            .values()
            .cloned()
            .collect();

        let stats = self.current_stats.lock().clone();
        let mut breakdown = self.current_breakdown.lock().clone();
        breakdown.compute_total();
        let memory = self.memory_usage.lock().clone();

        let frame = GpuProfileFrame {
            frame_index: self.frame_counter,
            scopes,
            pipeline_stats: stats,
            time_breakdown: breakdown,
            memory_usage: memory,
        };

        let mut history = self.history.lock();
        if history.len() >= self.history_size {
            history.remove(0);
        }
        history.push(frame);
        self.frame_counter += 1;
    }

    /// Get the last N GPU profile frames.
    pub fn get_history(&self, n: usize) -> Vec<GpuProfileFrame> {
        let history = self.history.lock();
        let start = history.len().saturating_sub(n);
        history[start..].to_vec()
    }

    /// Get the most recent GPU frame.
    pub fn get_last_frame(&self) -> Option<GpuProfileFrame> {
        self.history.lock().last().cloned()
    }

    /// Generate a text report from the last `n` frames.
    pub fn generate_report(&self, n: usize) -> GpuProfileReport {
        let history = self.history.lock();
        let frames = if n == 0 || n > history.len() {
            &history[..]
        } else {
            &history[history.len() - n..]
        };

        let frame_count = frames.len();
        if frame_count == 0 {
            return GpuProfileReport {
                title: "GPU Profile Report (no data)".into(),
                scope_lines: Vec::new(),
                avg_stats: GpuPipelineStats::default(),
                avg_breakdown: GpuFrameTimeBreakdown::default(),
                memory: GpuMemoryUsage::default(),
                frame_count: 0,
            };
        }

        // Aggregate scope stats.
        let mut scope_agg: HashMap<String, (Duration, Duration, u64)> = HashMap::new();
        let mut total_gpu_time = Duration::ZERO;

        for frame in frames {
            for scope in &frame.scopes {
                if !scope.resolved {
                    continue;
                }
                let entry = scope_agg
                    .entry(scope.name.clone())
                    .or_insert((Duration::ZERO, Duration::ZERO, 0));
                entry.0 += scope.duration;
                if scope.duration > entry.1 {
                    entry.1 = scope.duration;
                }
                entry.2 += 1;
            }
            total_gpu_time += frame.time_breakdown.total_time;
        }

        let avg_total = total_gpu_time / frame_count as u32;

        let mut scope_lines: Vec<GpuScopeReportLine> = scope_agg
            .iter()
            .map(|(name, (total, max, count))| {
                let avg = *total / (*count).max(1) as u32;
                let pct = if avg_total.as_nanos() > 0 {
                    avg.as_secs_f64() / avg_total.as_secs_f64() * 100.0
                } else {
                    0.0
                };
                GpuScopeReportLine {
                    name: name.clone(),
                    avg_duration: avg,
                    max_duration: *max,
                    pct_of_frame: pct,
                }
            })
            .collect();
        scope_lines.sort_by(|a, b| b.avg_duration.cmp(&a.avg_duration));

        // Average pipeline stats.
        let mut avg_stats = GpuPipelineStats::default();
        for frame in frames {
            avg_stats.vertex_count += frame.pipeline_stats.vertex_count;
            avg_stats.primitive_count += frame.pipeline_stats.primitive_count;
            avg_stats.fragment_count += frame.pipeline_stats.fragment_count;
            avg_stats.compute_invocations += frame.pipeline_stats.compute_invocations;
            avg_stats.draw_calls += frame.pipeline_stats.draw_calls;
            avg_stats.dispatch_calls += frame.pipeline_stats.dispatch_calls;
            avg_stats.render_passes += frame.pipeline_stats.render_passes;
            avg_stats.pipeline_changes += frame.pipeline_stats.pipeline_changes;
        }
        let fc = frame_count as u64;
        avg_stats.vertex_count /= fc;
        avg_stats.primitive_count /= fc;
        avg_stats.fragment_count /= fc;
        avg_stats.compute_invocations /= fc;
        avg_stats.draw_calls /= fc as u32;
        avg_stats.dispatch_calls /= fc as u32;
        avg_stats.render_passes /= fc as u32;
        avg_stats.pipeline_changes /= fc as u32;

        // Average breakdown.
        let mut avg_breakdown = GpuFrameTimeBreakdown::default();
        for frame in frames {
            avg_breakdown.vertex_time += frame.time_breakdown.vertex_time;
            avg_breakdown.fragment_time += frame.time_breakdown.fragment_time;
            avg_breakdown.compute_time += frame.time_breakdown.compute_time;
            avg_breakdown.transfer_time += frame.time_breakdown.transfer_time;
        }
        avg_breakdown.vertex_time /= frame_count as u32;
        avg_breakdown.fragment_time /= frame_count as u32;
        avg_breakdown.compute_time /= frame_count as u32;
        avg_breakdown.transfer_time /= frame_count as u32;
        avg_breakdown.compute_total();

        let memory = frames
            .last()
            .map(|f| f.memory_usage.clone())
            .unwrap_or_default();

        GpuProfileReport {
            title: format!("GPU Profile Report ({} frames)", frame_count),
            scope_lines,
            avg_stats,
            avg_breakdown,
            memory,
            frame_count,
        }
    }
}

impl Default for GpuProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_profiler_basic() {
        let mut profiler = GpuProfiler::new();
        profiler.begin_frame();

        let name = profiler.begin_scope("ShadowPass");
        profiler.end_scope(&name);
        profiler.resolve_scope("ShadowPass", 0, 2_000_000); // 2ms

        profiler.record_draw_call(1000, 333);
        profiler.record_render_pass();

        profiler.end_frame();

        let frame = profiler.get_last_frame().unwrap();
        assert_eq!(frame.scopes.len(), 1);
        assert_eq!(frame.scopes[0].name, "ShadowPass");
        assert!(frame.scopes[0].resolved);
        assert_eq!(frame.pipeline_stats.draw_calls, 1);
    }

    #[test]
    fn gpu_memory_formatting() {
        let mem = GpuMemoryUsage {
            device_local_total: 8 * 1024 * 1024 * 1024,
            device_local_used: 2 * 1024 * 1024 * 1024,
            host_visible_total: 16 * 1024 * 1024 * 1024,
            host_visible_used: 512 * 1024 * 1024,
            texture_memory: 1024 * 1024 * 1024,
            buffer_memory: 256 * 1024 * 1024,
            render_target_memory: 512 * 1024 * 1024,
            staging_memory: 64 * 1024 * 1024,
            allocation_count: 1500,
        };
        let text = format!("{}", mem);
        assert!(text.contains("GPU Memory Usage"));
        assert!(text.contains("GB"));
    }

    #[test]
    fn gpu_report_generation() {
        let mut profiler = GpuProfiler::new();
        for _ in 0..3 {
            profiler.begin_frame();
            profiler.begin_scope("MainPass");
            profiler.resolve_scope("MainPass", 0, 5_000_000);
            profiler.set_fragment_time(Duration::from_millis(3));
            profiler.set_vertex_time(Duration::from_millis(1));
            profiler.end_frame();
        }

        let report = profiler.generate_report(0);
        assert_eq!(report.frame_count, 3);
        let text = format!("{}", report);
        assert!(text.contains("MainPass"));
    }

    #[test]
    fn pipeline_stats_tracking() {
        let mut stats = GpuPipelineStats::default();
        stats.record_draw_call();
        stats.record_draw_call();
        stats.record_geometry(300, 100);
        stats.record_dispatch();
        stats.record_fragments(10000);

        assert_eq!(stats.draw_calls, 2);
        assert_eq!(stats.vertex_count, 300);
        assert_eq!(stats.dispatch_calls, 1);
        assert_eq!(stats.fragment_count, 10000);
    }
}
