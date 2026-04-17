// engine/render/src/render_statistics.rs
//
// Detailed render statistics tracking for the Genovo engine.
//
// Provides comprehensive GPU and CPU-side rendering metrics:
//
// - **Per-pass timing** — GPU and CPU time for each render pass.
// - **Draw call counts** — Total draw calls per frame, per pass.
// - **Triangle counts** — Total triangles submitted and rasterised.
// - **State change counts** — Pipeline, descriptor set, vertex buffer binds.
// - **GPU memory usage** — Total, per-category memory consumption.
// - **Texture memory breakdown** — Per-format and per-usage memory.
// - **Shader compile times** — Tracking of shader compilation costs.
// - **Buffer upload sizes** — Data transferred to the GPU per frame.
// - **Batching efficiency** — Ratio of logical draws to actual draw calls.
// - **Overdraw ratio** — Average number of fragment writes per pixel.
//
// # Usage
//
// The `RenderStats` struct is updated each frame by the renderer and render
// graph. It provides both instantaneous and rolling-average values.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Per-pass statistics
// ---------------------------------------------------------------------------

/// Timing and resource usage for a single render pass.
#[derive(Debug, Clone)]
pub struct PassStats {
    /// Pass name.
    pub name: String,
    /// GPU time in milliseconds.
    pub gpu_time_ms: f64,
    /// CPU time in milliseconds (command recording).
    pub cpu_time_ms: f64,
    /// Number of draw calls.
    pub draw_calls: u32,
    /// Number of compute dispatches.
    pub compute_dispatches: u32,
    /// Number of triangles submitted.
    pub triangles: u64,
    /// Number of vertices submitted.
    pub vertices: u64,
    /// Number of pipeline state changes.
    pub pipeline_changes: u32,
    /// Number of descriptor set binds.
    pub descriptor_binds: u32,
    /// Number of vertex buffer binds.
    pub vertex_buffer_binds: u32,
    /// Number of index buffer binds.
    pub index_buffer_binds: u32,
    /// Number of render target switches.
    pub render_target_switches: u32,
    /// Number of texture binds.
    pub texture_binds: u32,
    /// Bytes uploaded to GPU buffers during this pass.
    pub buffer_upload_bytes: u64,
    /// Number of instances drawn (for instanced rendering).
    pub instances: u64,
    /// Whether this pass is active (was executed this frame).
    pub active: bool,
}

impl PassStats {
    /// Create a new pass stats entry.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            gpu_time_ms: 0.0,
            cpu_time_ms: 0.0,
            draw_calls: 0,
            compute_dispatches: 0,
            triangles: 0,
            vertices: 0,
            pipeline_changes: 0,
            descriptor_binds: 0,
            vertex_buffer_binds: 0,
            index_buffer_binds: 0,
            render_target_switches: 0,
            texture_binds: 0,
            buffer_upload_bytes: 0,
            instances: 0,
            active: false,
        }
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        self.gpu_time_ms = 0.0;
        self.cpu_time_ms = 0.0;
        self.draw_calls = 0;
        self.compute_dispatches = 0;
        self.triangles = 0;
        self.vertices = 0;
        self.pipeline_changes = 0;
        self.descriptor_binds = 0;
        self.vertex_buffer_binds = 0;
        self.index_buffer_binds = 0;
        self.render_target_switches = 0;
        self.texture_binds = 0;
        self.buffer_upload_bytes = 0;
        self.instances = 0;
        self.active = false;
    }

    /// Total state changes.
    pub fn total_state_changes(&self) -> u32 {
        self.pipeline_changes
            + self.descriptor_binds
            + self.vertex_buffer_binds
            + self.index_buffer_binds
            + self.texture_binds
    }
}

// ---------------------------------------------------------------------------
// GPU memory statistics
// ---------------------------------------------------------------------------

/// Memory category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryCategory {
    /// Vertex and index buffers.
    Geometry,
    /// Texture data (including render targets).
    Textures,
    /// Render targets specifically.
    RenderTargets,
    /// Uniform/constant buffers.
    Uniforms,
    /// Structured/storage buffers.
    Storage,
    /// Staging/upload buffers.
    Staging,
    /// Shader program binaries.
    Shaders,
    /// Acceleration structures (ray tracing).
    AccelerationStructures,
    /// Other/unclassified.
    Other,
}

/// GPU memory usage per category.
#[derive(Debug, Clone)]
pub struct GpuMemoryStats {
    /// Per-category memory usage in bytes.
    pub categories: HashMap<MemoryCategory, u64>,
    /// Total allocated GPU memory.
    pub total_allocated: u64,
    /// Total GPU memory budget (device-reported).
    pub total_budget: u64,
    /// Peak memory usage this session.
    pub peak_usage: u64,
    /// Number of allocations.
    pub allocation_count: u32,
    /// Number of deallocations this frame.
    pub deallocation_count: u32,
}

impl GpuMemoryStats {
    /// Create empty stats.
    pub fn new() -> Self {
        Self {
            categories: HashMap::new(),
            total_allocated: 0,
            total_budget: 0,
            peak_usage: 0,
            allocation_count: 0,
            deallocation_count: 0,
        }
    }

    /// Record a memory allocation.
    pub fn record_allocation(&mut self, category: MemoryCategory, bytes: u64) {
        *self.categories.entry(category).or_insert(0) += bytes;
        self.total_allocated += bytes;
        self.peak_usage = self.peak_usage.max(self.total_allocated);
        self.allocation_count += 1;
    }

    /// Record a memory deallocation.
    pub fn record_deallocation(&mut self, category: MemoryCategory, bytes: u64) {
        if let Some(cat) = self.categories.get_mut(&category) {
            *cat = cat.saturating_sub(bytes);
        }
        self.total_allocated = self.total_allocated.saturating_sub(bytes);
        self.deallocation_count += 1;
    }

    /// Get memory usage for a category in megabytes.
    pub fn category_mb(&self, category: MemoryCategory) -> f64 {
        self.categories.get(&category).copied().unwrap_or(0) as f64 / (1024.0 * 1024.0)
    }

    /// Get total allocated in megabytes.
    pub fn total_mb(&self) -> f64 {
        self.total_allocated as f64 / (1024.0 * 1024.0)
    }

    /// Get utilisation ratio (0-1).
    pub fn utilisation(&self) -> f64 {
        if self.total_budget > 0 {
            self.total_allocated as f64 / self.total_budget as f64
        } else {
            0.0
        }
    }
}

impl Default for GpuMemoryStats {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Texture memory breakdown
// ---------------------------------------------------------------------------

/// Texture memory entry.
#[derive(Debug, Clone)]
pub struct TextureMemoryEntry {
    /// Texture name/label.
    pub name: String,
    /// Format name.
    pub format: String,
    /// Dimensions (width x height x depth/layers).
    pub dimensions: (u32, u32, u32),
    /// Mip count.
    pub mip_count: u32,
    /// Memory usage in bytes.
    pub memory_bytes: u64,
    /// Whether this is a render target.
    pub is_render_target: bool,
    /// Whether this is streaming.
    pub is_streaming: bool,
}

/// Texture memory breakdown.
#[derive(Debug, Clone)]
pub struct TextureMemoryBreakdown {
    /// All tracked textures.
    pub entries: Vec<TextureMemoryEntry>,
    /// Total texture memory.
    pub total_bytes: u64,
    /// Render target memory.
    pub render_target_bytes: u64,
    /// Streaming texture memory.
    pub streaming_bytes: u64,
}

impl TextureMemoryBreakdown {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            total_bytes: 0,
            render_target_bytes: 0,
            streaming_bytes: 0,
        }
    }

    /// Add a texture entry.
    pub fn add(&mut self, entry: TextureMemoryEntry) {
        self.total_bytes += entry.memory_bytes;
        if entry.is_render_target {
            self.render_target_bytes += entry.memory_bytes;
        }
        if entry.is_streaming {
            self.streaming_bytes += entry.memory_bytes;
        }
        self.entries.push(entry);
    }

    /// Sort entries by memory usage (descending).
    pub fn sort_by_size(&mut self) {
        self.entries.sort_by(|a, b| b.memory_bytes.cmp(&a.memory_bytes));
    }

    /// Get the top N memory consumers.
    pub fn top_n(&self, n: usize) -> &[TextureMemoryEntry] {
        &self.entries[..n.min(self.entries.len())]
    }

    /// Total in megabytes.
    pub fn total_mb(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0)
    }
}

impl Default for TextureMemoryBreakdown {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Shader compile tracking
// ---------------------------------------------------------------------------

/// Shader compilation record.
#[derive(Debug, Clone)]
pub struct ShaderCompileRecord {
    /// Shader name.
    pub name: String,
    /// Variant hash.
    pub variant_hash: u64,
    /// Compilation time in milliseconds.
    pub compile_time_ms: f64,
    /// Whether compilation was successful.
    pub success: bool,
    /// Error message (if any).
    pub error: Option<String>,
    /// Frame when compiled.
    pub frame: u64,
    /// Binary size in bytes.
    pub binary_size: u32,
}

/// Shader compilation statistics.
#[derive(Debug, Clone)]
pub struct ShaderCompileStats {
    /// All compilation records.
    pub records: Vec<ShaderCompileRecord>,
    /// Total compilation time this session.
    pub total_compile_time_ms: f64,
    /// Number of successful compilations.
    pub success_count: u32,
    /// Number of failed compilations.
    pub failure_count: u32,
    /// Total unique variants compiled.
    pub unique_variants: u32,
    /// Peak compilation time for a single shader.
    pub peak_compile_ms: f64,
}

impl ShaderCompileStats {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            total_compile_time_ms: 0.0,
            success_count: 0,
            failure_count: 0,
            unique_variants: 0,
            peak_compile_ms: 0.0,
        }
    }

    /// Record a shader compilation.
    pub fn record(&mut self, record: ShaderCompileRecord) {
        self.total_compile_time_ms += record.compile_time_ms;
        self.peak_compile_ms = self.peak_compile_ms.max(record.compile_time_ms);
        if record.success {
            self.success_count += 1;
        } else {
            self.failure_count += 1;
        }
        self.unique_variants += 1;
        self.records.push(record);
    }

    /// Average compilation time.
    pub fn avg_compile_time_ms(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total > 0 {
            self.total_compile_time_ms / total as f64
        } else {
            0.0
        }
    }
}

impl Default for ShaderCompileStats {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Frame statistics
// ---------------------------------------------------------------------------

/// Complete render statistics for a single frame.
#[derive(Debug, Clone)]
pub struct FrameStats {
    /// Frame number.
    pub frame: u64,
    /// Total GPU frame time in milliseconds.
    pub gpu_frame_time_ms: f64,
    /// Total CPU render time in milliseconds.
    pub cpu_frame_time_ms: f64,
    /// Per-pass statistics.
    pub passes: Vec<PassStats>,
    /// Total draw calls.
    pub total_draw_calls: u32,
    /// Total compute dispatches.
    pub total_compute_dispatches: u32,
    /// Total triangles submitted.
    pub total_triangles: u64,
    /// Total vertices submitted.
    pub total_vertices: u64,
    /// Total instances drawn.
    pub total_instances: u64,
    /// Total state changes.
    pub total_state_changes: u32,
    /// Bytes uploaded to GPU this frame.
    pub total_upload_bytes: u64,
    /// Number of visible objects after culling.
    pub visible_objects: u32,
    /// Number of culled objects.
    pub culled_objects: u32,
    /// Overdraw ratio (average fragment writes per pixel).
    pub overdraw_ratio: f32,
    /// Batching efficiency (logical draws / actual draw calls).
    pub batching_efficiency: f32,
    /// Number of render target switches.
    pub render_target_switches: u32,
    /// Number of active render passes.
    pub active_pass_count: u32,
}

impl FrameStats {
    /// Create a new frame stats.
    pub fn new(frame: u64) -> Self {
        Self {
            frame,
            gpu_frame_time_ms: 0.0,
            cpu_frame_time_ms: 0.0,
            passes: Vec::new(),
            total_draw_calls: 0,
            total_compute_dispatches: 0,
            total_triangles: 0,
            total_vertices: 0,
            total_instances: 0,
            total_state_changes: 0,
            total_upload_bytes: 0,
            visible_objects: 0,
            culled_objects: 0,
            overdraw_ratio: 1.0,
            batching_efficiency: 1.0,
            render_target_switches: 0,
            active_pass_count: 0,
        }
    }

    /// Aggregate totals from pass stats.
    pub fn aggregate_from_passes(&mut self) {
        self.total_draw_calls = 0;
        self.total_compute_dispatches = 0;
        self.total_triangles = 0;
        self.total_vertices = 0;
        self.total_instances = 0;
        self.total_state_changes = 0;
        self.total_upload_bytes = 0;
        self.render_target_switches = 0;
        self.active_pass_count = 0;

        for pass in &self.passes {
            if pass.active {
                self.total_draw_calls += pass.draw_calls;
                self.total_compute_dispatches += pass.compute_dispatches;
                self.total_triangles += pass.triangles;
                self.total_vertices += pass.vertices;
                self.total_instances += pass.instances;
                self.total_state_changes += pass.total_state_changes();
                self.total_upload_bytes += pass.buffer_upload_bytes;
                self.render_target_switches += pass.render_target_switches;
                self.active_pass_count += 1;
            }
        }
    }

    /// GPU FPS estimate.
    pub fn gpu_fps(&self) -> f64 {
        if self.gpu_frame_time_ms > 0.0 {
            1000.0 / self.gpu_frame_time_ms
        } else {
            0.0
        }
    }

    /// CPU FPS estimate.
    pub fn cpu_fps(&self) -> f64 {
        if self.cpu_frame_time_ms > 0.0 {
            1000.0 / self.cpu_frame_time_ms
        } else {
            0.0
        }
    }

    /// Culling efficiency (percentage of objects culled).
    pub fn culling_efficiency(&self) -> f32 {
        let total = self.visible_objects + self.culled_objects;
        if total > 0 {
            self.culled_objects as f32 / total as f32 * 100.0
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Rolling statistics
// ---------------------------------------------------------------------------

/// Rolling average statistics over a configurable window.
#[derive(Debug, Clone)]
pub struct RollingStats {
    /// Window size (number of frames to average over).
    pub window_size: usize,
    /// GPU frame times (circular buffer).
    gpu_times: Vec<f64>,
    /// CPU frame times (circular buffer).
    cpu_times: Vec<f64>,
    /// Draw call counts.
    draw_calls: Vec<u32>,
    /// Triangle counts.
    triangles: Vec<u64>,
    /// Write cursor.
    cursor: usize,
    /// Number of samples collected so far.
    count: usize,
}

impl RollingStats {
    /// Create a new rolling stats tracker.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            gpu_times: vec![0.0; window_size],
            cpu_times: vec![0.0; window_size],
            draw_calls: vec![0; window_size],
            triangles: vec![0; window_size],
            cursor: 0,
            count: 0,
        }
    }

    /// Record a frame's stats.
    pub fn record(&mut self, frame: &FrameStats) {
        self.gpu_times[self.cursor] = frame.gpu_frame_time_ms;
        self.cpu_times[self.cursor] = frame.cpu_frame_time_ms;
        self.draw_calls[self.cursor] = frame.total_draw_calls;
        self.triangles[self.cursor] = frame.total_triangles;

        self.cursor = (self.cursor + 1) % self.window_size;
        self.count = (self.count + 1).min(self.window_size);
    }

    /// Average GPU frame time.
    pub fn avg_gpu_time(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        self.gpu_times[..self.count].iter().sum::<f64>() / self.count as f64
    }

    /// Average CPU frame time.
    pub fn avg_cpu_time(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        self.cpu_times[..self.count].iter().sum::<f64>() / self.count as f64
    }

    /// Average FPS (based on GPU time).
    pub fn avg_fps(&self) -> f64 {
        let t = self.avg_gpu_time();
        if t > 0.0 { 1000.0 / t } else { 0.0 }
    }

    /// Average draw calls.
    pub fn avg_draw_calls(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        self.draw_calls[..self.count].iter().map(|&v| v as f64).sum::<f64>() / self.count as f64
    }

    /// Average triangle count.
    pub fn avg_triangles(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        self.triangles[..self.count].iter().map(|&v| v as f64).sum::<f64>() / self.count as f64
    }

    /// Min/max GPU time in the window.
    pub fn gpu_time_range(&self) -> (f64, f64) {
        if self.count == 0 { return (0.0, 0.0); }
        let slice = &self.gpu_times[..self.count];
        let min = slice.iter().cloned().fold(f64::MAX, f64::min);
        let max = slice.iter().cloned().fold(f64::MIN, f64::max);
        (min, max)
    }

    /// 99th percentile GPU frame time.
    pub fn gpu_time_p99(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        let mut sorted: Vec<f64> = self.gpu_times[..self.count].to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((self.count as f64 * 0.99) as usize).min(self.count - 1);
        sorted[idx]
    }
}

// ---------------------------------------------------------------------------
// Render statistics manager
// ---------------------------------------------------------------------------

/// The main render statistics manager.
#[derive(Debug)]
pub struct RenderStats {
    /// Current frame statistics.
    pub current: FrameStats,
    /// Previous frame statistics.
    pub previous: FrameStats,
    /// Rolling average statistics.
    pub rolling: RollingStats,
    /// GPU memory statistics.
    pub memory: GpuMemoryStats,
    /// Texture memory breakdown.
    pub textures: TextureMemoryBreakdown,
    /// Shader compilation statistics.
    pub shaders: ShaderCompileStats,
    /// Frame counter.
    pub frame: u64,
    /// Whether statistics collection is enabled.
    pub enabled: bool,
    /// Whether detailed per-pass GPU timing is enabled.
    pub gpu_timing_enabled: bool,
}

impl RenderStats {
    /// Create a new statistics manager.
    pub fn new() -> Self {
        Self {
            current: FrameStats::new(0),
            previous: FrameStats::new(0),
            rolling: RollingStats::new(120),
            memory: GpuMemoryStats::new(),
            textures: TextureMemoryBreakdown::new(),
            shaders: ShaderCompileStats::new(),
            frame: 0,
            enabled: true,
            gpu_timing_enabled: true,
        }
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self) {
        self.previous = self.current.clone();
        self.frame += 1;
        self.current = FrameStats::new(self.frame);
    }

    /// End the current frame.
    pub fn end_frame(&mut self) {
        self.current.aggregate_from_passes();
        self.rolling.record(&self.current);
    }

    /// Begin a render pass.
    pub fn begin_pass(&mut self, name: &str) {
        let mut pass = PassStats::new(name);
        pass.active = true;
        self.current.passes.push(pass);
    }

    /// End the current render pass.
    pub fn end_pass(&mut self) {
        // In a real implementation, this would read GPU timestamp queries.
    }

    /// Record a draw call.
    pub fn record_draw(&mut self, triangles: u64, vertices: u64, instances: u64) {
        if let Some(pass) = self.current.passes.last_mut() {
            pass.draw_calls += 1;
            pass.triangles += triangles;
            pass.vertices += vertices;
            pass.instances += instances;
        }
    }

    /// Record a compute dispatch.
    pub fn record_dispatch(&mut self) {
        if let Some(pass) = self.current.passes.last_mut() {
            pass.compute_dispatches += 1;
        }
    }

    /// Record a state change.
    pub fn record_pipeline_change(&mut self) {
        if let Some(pass) = self.current.passes.last_mut() {
            pass.pipeline_changes += 1;
        }
    }

    /// Record a descriptor bind.
    pub fn record_descriptor_bind(&mut self) {
        if let Some(pass) = self.current.passes.last_mut() {
            pass.descriptor_binds += 1;
        }
    }

    /// Record a buffer upload.
    pub fn record_buffer_upload(&mut self, bytes: u64) {
        if let Some(pass) = self.current.passes.last_mut() {
            pass.buffer_upload_bytes += bytes;
        }
    }

    /// Set the overdraw ratio.
    pub fn set_overdraw(&mut self, ratio: f32) {
        self.current.overdraw_ratio = ratio;
    }

    /// Set culling stats.
    pub fn set_culling(&mut self, visible: u32, culled: u32) {
        self.current.visible_objects = visible;
        self.current.culled_objects = culled;
    }

    /// Format a summary string.
    pub fn summary_string(&self) -> String {
        format!(
            "Frame {} | GPU: {:.2}ms ({:.0} FPS) | CPU: {:.2}ms | \
             Draws: {} | Tris: {} | States: {} | Mem: {:.1}MB",
            self.current.frame,
            self.rolling.avg_gpu_time(),
            self.rolling.avg_fps(),
            self.rolling.avg_cpu_time(),
            self.current.total_draw_calls,
            self.current.total_triangles,
            self.current.total_state_changes,
            self.memory.total_mb(),
        )
    }

    /// Detailed pass breakdown string.
    pub fn pass_breakdown(&self) -> String {
        let mut lines = Vec::new();
        for pass in &self.current.passes {
            if pass.active {
                lines.push(format!(
                    "  {} | GPU: {:.2}ms | Draws: {} | Tris: {} | States: {}",
                    pass.name,
                    pass.gpu_time_ms,
                    pass.draw_calls,
                    pass.triangles,
                    pass.total_state_changes(),
                ));
            }
        }
        lines.join("\n")
    }
}

impl Default for RenderStats {
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
    fn test_pass_stats() {
        let mut pass = PassStats::new("Shadow");
        pass.draw_calls = 100;
        pass.pipeline_changes = 5;
        pass.descriptor_binds = 200;
        assert_eq!(pass.total_state_changes(), 205);
    }

    #[test]
    fn test_gpu_memory() {
        let mut mem = GpuMemoryStats::new();
        mem.record_allocation(MemoryCategory::Textures, 1024 * 1024);
        assert_eq!(mem.total_allocated, 1024 * 1024);
        assert!((mem.category_mb(MemoryCategory::Textures) - 1.0).abs() < 0.01);

        mem.record_deallocation(MemoryCategory::Textures, 512 * 1024);
        assert_eq!(mem.total_allocated, 512 * 1024);
        assert_eq!(mem.peak_usage, 1024 * 1024);
    }

    #[test]
    fn test_rolling_stats() {
        let mut rolling = RollingStats::new(4);
        for i in 0..4 {
            let mut frame = FrameStats::new(i);
            frame.gpu_frame_time_ms = 16.0;
            frame.total_draw_calls = 100;
            rolling.record(&frame);
        }
        assert!((rolling.avg_gpu_time() - 16.0).abs() < 0.01);
        assert!((rolling.avg_fps() - 62.5).abs() < 0.1);
    }

    #[test]
    fn test_render_stats_lifecycle() {
        let mut stats = RenderStats::new();

        stats.begin_frame();
        stats.begin_pass("Depth");
        stats.record_draw(100, 300, 1);
        stats.record_draw(200, 600, 1);
        stats.record_pipeline_change();
        stats.end_pass();
        stats.end_frame();

        assert_eq!(stats.current.total_draw_calls, 2);
        assert_eq!(stats.current.total_triangles, 300);
    }

    #[test]
    fn test_shader_compile_stats() {
        let mut stats = ShaderCompileStats::new();
        stats.record(ShaderCompileRecord {
            name: "PBR".to_string(),
            variant_hash: 12345,
            compile_time_ms: 50.0,
            success: true,
            error: None,
            frame: 1,
            binary_size: 4096,
        });
        assert_eq!(stats.success_count, 1);
        assert!((stats.avg_compile_time_ms() - 50.0).abs() < 0.01);
    }
}
