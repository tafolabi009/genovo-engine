//! Master render loop that orchestrates all render passes in order.
//!
//! This module defines the `RenderLoop` which is the central integration
//! point for the rendering pipeline. It executes all render passes in the
//! correct order each frame:
//!
//! 1. **Shadow pass** -- Renders depth from each light's perspective for shadow mapping.
//! 2. **Depth prepass** -- Renders scene depth for early-Z and SSAO.
//! 3. **Scene render** -- Main forward or deferred shading pass with PBR lighting.
//! 4. **Skeletal animation upload** -- Updates bone palette buffers.
//! 5. **Post-process** -- Bloom, tone mapping, FXAA, color grading.
//! 6. **UI overlay** -- egui or custom UI rendered on top.
//! 7. **Present** -- Submits the final framebuffer to the swapchain.
//!
//! # Architecture
//!
//! The `RenderLoop` does not own GPU resources directly. Instead, it
//! references external pass implementations and invokes them in sequence.
//! Each pass receives a `RenderContext` with the current frame's camera,
//! lights, and command encoder.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Render pass identifiers
// ---------------------------------------------------------------------------

/// Identifies a render pass in the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderPassId {
    ShadowMap,
    DepthPrepass,
    GBuffer,
    Lighting,
    ForwardOpaque,
    ForwardTransparent,
    Skybox,
    Particles,
    SkeletalUpload,
    PostProcess,
    Bloom,
    ToneMap,
    FXAA,
    UIOverlay,
    DebugOverlay,
    Present,
    Custom(u32),
}

/// Priority for pass ordering (lower = earlier).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PassPriority(pub u32);

impl PassPriority {
    pub const SHADOW: Self = Self(100);
    pub const DEPTH_PREPASS: Self = Self(200);
    pub const GBUFFER: Self = Self(300);
    pub const LIGHTING: Self = Self(400);
    pub const FORWARD_OPAQUE: Self = Self(500);
    pub const SKYBOX: Self = Self(550);
    pub const FORWARD_TRANSPARENT: Self = Self(600);
    pub const PARTICLES: Self = Self(650);
    pub const SKELETAL_UPLOAD: Self = Self(150);
    pub const POST_PROCESS: Self = Self(700);
    pub const BLOOM: Self = Self(710);
    pub const TONE_MAP: Self = Self(720);
    pub const FXAA: Self = Self(730);
    pub const UI_OVERLAY: Self = Self(800);
    pub const DEBUG_OVERLAY: Self = Self(850);
    pub const PRESENT: Self = Self(900);
}

// ---------------------------------------------------------------------------
// Render context
// ---------------------------------------------------------------------------

/// Per-frame data passed to each render pass.
#[derive(Debug, Clone)]
pub struct FrameData {
    /// Frame index (monotonically increasing).
    pub frame_index: u64,
    /// Delta time in seconds.
    pub delta_time: f32,
    /// Total elapsed time in seconds.
    pub total_time: f32,
    /// Viewport width in pixels.
    pub viewport_width: u32,
    /// Viewport height in pixels.
    pub viewport_height: u32,
    /// Camera view matrix (4x4, column-major).
    pub view_matrix: [f32; 16],
    /// Camera projection matrix (4x4, column-major).
    pub projection_matrix: [f32; 16],
    /// Camera position in world space.
    pub camera_position: [f32; 3],
    /// Number of visible entities after culling.
    pub visible_entity_count: u32,
    /// Number of active lights.
    pub light_count: u32,
    /// Whether shadows are enabled.
    pub shadows_enabled: bool,
    /// Whether post-processing is enabled.
    pub post_process_enabled: bool,
}

impl Default for FrameData {
    fn default() -> Self {
        let identity = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        Self {
            frame_index: 0,
            delta_time: 1.0 / 60.0,
            total_time: 0.0,
            viewport_width: 1920,
            viewport_height: 1080,
            view_matrix: identity,
            projection_matrix: identity,
            camera_position: [0.0; 3],
            visible_entity_count: 0,
            light_count: 0,
            shadows_enabled: true,
            post_process_enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Pass result
// ---------------------------------------------------------------------------

/// Result of executing a render pass.
#[derive(Debug, Clone)]
pub struct PassResult {
    /// Pass identifier.
    pub pass_id: RenderPassId,
    /// Whether the pass executed successfully.
    pub success: bool,
    /// Execution time in microseconds.
    pub time_us: u64,
    /// Number of draw calls issued.
    pub draw_calls: u32,
    /// Number of triangles rendered.
    pub triangle_count: u64,
    /// Optional error message.
    pub error: Option<String>,
}

impl PassResult {
    /// Create a successful result.
    pub fn success(pass_id: RenderPassId, time_us: u64, draw_calls: u32, triangles: u64) -> Self {
        Self {
            pass_id,
            success: true,
            time_us,
            draw_calls,
            triangle_count: triangles,
            error: None,
        }
    }

    /// Create a failed result.
    pub fn failed(pass_id: RenderPassId, error: String) -> Self {
        Self {
            pass_id,
            success: false,
            time_us: 0,
            draw_calls: 0,
            triangle_count: 0,
            error: Some(error),
        }
    }

    /// Create a skipped result (pass was disabled).
    pub fn skipped(pass_id: RenderPassId) -> Self {
        Self {
            pass_id,
            success: true,
            time_us: 0,
            draw_calls: 0,
            triangle_count: 0,
            error: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Pass descriptor
// ---------------------------------------------------------------------------

/// Describes a render pass in the pipeline.
#[derive(Debug)]
pub struct PassDescriptor {
    /// Unique identifier.
    pub id: RenderPassId,
    /// Display name for profiling.
    pub name: String,
    /// Execution priority (lower = earlier).
    pub priority: PassPriority,
    /// Whether this pass is currently enabled.
    pub enabled: bool,
    /// Dependencies: passes that must complete before this one.
    pub dependencies: Vec<RenderPassId>,
}

// ---------------------------------------------------------------------------
// Render statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics for a complete frame render.
#[derive(Debug, Clone, Default)]
pub struct FrameRenderStats {
    /// Total frame render time in microseconds.
    pub total_time_us: u64,
    /// Total draw calls across all passes.
    pub total_draw_calls: u32,
    /// Total triangles across all passes.
    pub total_triangles: u64,
    /// Number of passes executed.
    pub passes_executed: u32,
    /// Number of passes skipped.
    pub passes_skipped: u32,
    /// Number of passes that failed.
    pub passes_failed: u32,
    /// Per-pass timing breakdown.
    pub pass_timings: Vec<(String, u64)>,
}

// ---------------------------------------------------------------------------
// Render loop configuration
// ---------------------------------------------------------------------------

/// Configuration for the render loop.
#[derive(Debug, Clone)]
pub struct RenderLoopConfig {
    /// Maximum number of shadow-casting lights.
    pub max_shadow_lights: u32,
    /// Shadow map resolution.
    pub shadow_resolution: u32,
    /// Number of shadow cascades for directional lights.
    pub shadow_cascades: u32,
    /// Whether to use depth prepass.
    pub depth_prepass: bool,
    /// Whether to use deferred rendering (vs forward).
    pub deferred_rendering: bool,
    /// Post-process quality level (0=off, 1=low, 2=medium, 3=high).
    pub post_process_quality: u32,
    /// Whether bloom is enabled.
    pub bloom_enabled: bool,
    /// Bloom intensity.
    pub bloom_intensity: f32,
    /// Whether FXAA is enabled.
    pub fxaa_enabled: bool,
    /// Whether the debug overlay is shown.
    pub debug_overlay: bool,
    /// Target framerate for frame pacing.
    pub target_framerate: u32,
    /// Whether VSync is enabled.
    pub vsync: bool,
}

impl Default for RenderLoopConfig {
    fn default() -> Self {
        Self {
            max_shadow_lights: 4,
            shadow_resolution: 2048,
            shadow_cascades: 4,
            depth_prepass: true,
            deferred_rendering: false,
            post_process_quality: 2,
            bloom_enabled: true,
            bloom_intensity: 0.5,
            fxaa_enabled: true,
            debug_overlay: false,
            target_framerate: 60,
            vsync: true,
        }
    }
}

// ---------------------------------------------------------------------------
// RenderLoop
// ---------------------------------------------------------------------------

/// The master render loop that orchestrates all render passes.
///
/// This is the central integration point for the rendering pipeline. Each
/// frame, `execute_frame()` is called which runs all enabled passes in
/// priority order, collects statistics, and produces a `FrameRenderStats`.
pub struct RenderLoop {
    /// Configuration.
    config: RenderLoopConfig,
    /// Registered passes, sorted by priority.
    passes: Vec<PassDescriptor>,
    /// Per-frame statistics history (ring buffer).
    stats_history: Vec<FrameRenderStats>,
    /// Maximum history length.
    max_history: usize,
    /// Current frame index.
    frame_index: u64,
    /// Average frame time (smoothed).
    avg_frame_time_us: f64,
    /// Pass enable overrides.
    pass_overrides: HashMap<RenderPassId, bool>,
}

impl RenderLoop {
    /// Create a new render loop with default configuration.
    pub fn new() -> Self {
        Self::with_config(RenderLoopConfig::default())
    }

    /// Create a new render loop with the given configuration.
    pub fn with_config(config: RenderLoopConfig) -> Self {
        let mut loop_inst = Self {
            config,
            passes: Vec::new(),
            stats_history: Vec::new(),
            max_history: 300,
            frame_index: 0,
            avg_frame_time_us: 16667.0,
            pass_overrides: HashMap::new(),
        };
        loop_inst.register_default_passes();
        loop_inst
    }

    /// Register the default set of render passes.
    fn register_default_passes(&mut self) {
        self.passes.push(PassDescriptor {
            id: RenderPassId::ShadowMap,
            name: "Shadow Map".to_string(),
            priority: PassPriority::SHADOW,
            enabled: true,
            dependencies: vec![],
        });
        self.passes.push(PassDescriptor {
            id: RenderPassId::SkeletalUpload,
            name: "Skeletal Upload".to_string(),
            priority: PassPriority::SKELETAL_UPLOAD,
            enabled: true,
            dependencies: vec![],
        });
        self.passes.push(PassDescriptor {
            id: RenderPassId::DepthPrepass,
            name: "Depth Prepass".to_string(),
            priority: PassPriority::DEPTH_PREPASS,
            enabled: true,
            dependencies: vec![RenderPassId::SkeletalUpload],
        });
        self.passes.push(PassDescriptor {
            id: RenderPassId::ForwardOpaque,
            name: "Forward Opaque".to_string(),
            priority: PassPriority::FORWARD_OPAQUE,
            enabled: true,
            dependencies: vec![RenderPassId::ShadowMap, RenderPassId::DepthPrepass],
        });
        self.passes.push(PassDescriptor {
            id: RenderPassId::Skybox,
            name: "Skybox".to_string(),
            priority: PassPriority::SKYBOX,
            enabled: true,
            dependencies: vec![RenderPassId::ForwardOpaque],
        });
        self.passes.push(PassDescriptor {
            id: RenderPassId::ForwardTransparent,
            name: "Forward Transparent".to_string(),
            priority: PassPriority::FORWARD_TRANSPARENT,
            enabled: true,
            dependencies: vec![RenderPassId::ForwardOpaque],
        });
        self.passes.push(PassDescriptor {
            id: RenderPassId::Particles,
            name: "Particles".to_string(),
            priority: PassPriority::PARTICLES,
            enabled: true,
            dependencies: vec![RenderPassId::ForwardTransparent],
        });
        self.passes.push(PassDescriptor {
            id: RenderPassId::PostProcess,
            name: "Post Process".to_string(),
            priority: PassPriority::POST_PROCESS,
            enabled: true,
            dependencies: vec![RenderPassId::Particles],
        });
        self.passes.push(PassDescriptor {
            id: RenderPassId::Bloom,
            name: "Bloom".to_string(),
            priority: PassPriority::BLOOM,
            enabled: true,
            dependencies: vec![RenderPassId::PostProcess],
        });
        self.passes.push(PassDescriptor {
            id: RenderPassId::ToneMap,
            name: "Tone Map".to_string(),
            priority: PassPriority::TONE_MAP,
            enabled: true,
            dependencies: vec![RenderPassId::Bloom],
        });
        self.passes.push(PassDescriptor {
            id: RenderPassId::FXAA,
            name: "FXAA".to_string(),
            priority: PassPriority::FXAA,
            enabled: true,
            dependencies: vec![RenderPassId::ToneMap],
        });
        self.passes.push(PassDescriptor {
            id: RenderPassId::UIOverlay,
            name: "UI Overlay".to_string(),
            priority: PassPriority::UI_OVERLAY,
            enabled: true,
            dependencies: vec![RenderPassId::FXAA],
        });
        self.passes.push(PassDescriptor {
            id: RenderPassId::DebugOverlay,
            name: "Debug Overlay".to_string(),
            priority: PassPriority::DEBUG_OVERLAY,
            enabled: false,
            dependencies: vec![RenderPassId::UIOverlay],
        });
        self.passes.push(PassDescriptor {
            id: RenderPassId::Present,
            name: "Present".to_string(),
            priority: PassPriority::PRESENT,
            enabled: true,
            dependencies: vec![RenderPassId::UIOverlay],
        });

        // Sort by priority
        self.passes.sort_by_key(|p| p.priority);
    }

    /// Execute a complete frame render.
    ///
    /// Runs all enabled passes in priority order, collects per-pass timing,
    /// and returns aggregate frame statistics.
    pub fn execute_frame(&mut self, frame_data: &FrameData) -> FrameRenderStats {
        let frame_start = std::time::Instant::now();
        let mut stats = FrameRenderStats::default();

        for pass in &self.passes {
            let enabled = self.pass_overrides
                .get(&pass.id)
                .copied()
                .unwrap_or(pass.enabled);

            if !enabled || !self.should_execute_pass(pass, frame_data) {
                stats.passes_skipped += 1;
                continue;
            }

            let pass_start = std::time::Instant::now();

            // Simulate pass execution (in a real engine, this would invoke the
            // actual GPU work via the pass implementation)
            let result = self.execute_pass(pass, frame_data);

            let pass_time = pass_start.elapsed().as_micros() as u64;

            if result.success {
                stats.passes_executed += 1;
                stats.total_draw_calls += result.draw_calls;
                stats.total_triangles += result.triangle_count;
                stats.pass_timings.push((pass.name.clone(), pass_time));
            } else {
                stats.passes_failed += 1;
                if let Some(err) = &result.error {
                    log::error!("Render pass '{}' failed: {}", pass.name, err);
                }
            }
        }

        let total_time = frame_start.elapsed().as_micros() as u64;
        stats.total_time_us = total_time;

        // Update smoothed average
        self.avg_frame_time_us = self.avg_frame_time_us * 0.95 + total_time as f64 * 0.05;

        // Store in history
        if self.stats_history.len() >= self.max_history {
            self.stats_history.remove(0);
        }
        self.stats_history.push(stats.clone());

        self.frame_index += 1;
        stats
    }

    /// Check if a pass should execute based on configuration and frame data.
    fn should_execute_pass(&self, pass: &PassDescriptor, frame_data: &FrameData) -> bool {
        match pass.id {
            RenderPassId::ShadowMap => frame_data.shadows_enabled && self.config.max_shadow_lights > 0,
            RenderPassId::DepthPrepass => self.config.depth_prepass,
            RenderPassId::Bloom => self.config.bloom_enabled && frame_data.post_process_enabled,
            RenderPassId::FXAA => self.config.fxaa_enabled && frame_data.post_process_enabled,
            RenderPassId::PostProcess => frame_data.post_process_enabled,
            RenderPassId::ToneMap => frame_data.post_process_enabled,
            RenderPassId::DebugOverlay => self.config.debug_overlay,
            _ => true,
        }
    }

    /// Execute a single render pass. In a real engine, this would dispatch
    /// to the actual GPU pass implementation. Here we produce a synthetic result.
    fn execute_pass(&self, pass: &PassDescriptor, frame_data: &FrameData) -> PassResult {
        // Estimate draw calls and triangles based on pass type and scene size
        let entity_count = frame_data.visible_entity_count;
        let (draws, tris) = match pass.id {
            RenderPassId::ShadowMap => (entity_count * frame_data.light_count, entity_count as u64 * 1000),
            RenderPassId::DepthPrepass => (entity_count, entity_count as u64 * 500),
            RenderPassId::ForwardOpaque => (entity_count, entity_count as u64 * 2000),
            RenderPassId::ForwardTransparent => (entity_count / 4, entity_count as u64 * 200),
            RenderPassId::Skybox => (1, 12),
            RenderPassId::Particles => (entity_count / 8, entity_count as u64 * 100),
            RenderPassId::PostProcess | RenderPassId::Bloom | RenderPassId::ToneMap | RenderPassId::FXAA => (1, 2),
            RenderPassId::UIOverlay => (10, 100),
            RenderPassId::DebugOverlay => (5, 50),
            RenderPassId::Present => (0, 0),
            _ => (0, 0),
        };

        PassResult::success(pass.id, 0, draws, tris)
    }

    // -- Configuration --

    /// Get the current configuration.
    pub fn config(&self) -> &RenderLoopConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: RenderLoopConfig) {
        self.config = config;
    }

    /// Enable or disable a specific pass.
    pub fn set_pass_enabled(&mut self, pass_id: RenderPassId, enabled: bool) {
        self.pass_overrides.insert(pass_id, enabled);
    }

    /// Get the average frame time in microseconds (smoothed).
    pub fn avg_frame_time_us(&self) -> f64 {
        self.avg_frame_time_us
    }

    /// Get the estimated FPS based on average frame time.
    pub fn estimated_fps(&self) -> f64 {
        if self.avg_frame_time_us > 0.0 {
            1_000_000.0 / self.avg_frame_time_us
        } else {
            0.0
        }
    }

    /// Get the statistics history.
    pub fn stats_history(&self) -> &[FrameRenderStats] {
        &self.stats_history
    }

    /// Get the last frame's statistics.
    pub fn last_stats(&self) -> Option<&FrameRenderStats> {
        self.stats_history.last()
    }

    /// Get the current frame index.
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Get the number of registered passes.
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// Get the names of all registered passes in execution order.
    pub fn pass_names(&self) -> Vec<&str> {
        self.passes.iter().map(|p| p.name.as_str()).collect()
    }

    /// Register a custom pass.
    pub fn register_pass(&mut self, descriptor: PassDescriptor) {
        self.passes.push(descriptor);
        self.passes.sort_by_key(|p| p.priority);
    }
}

impl Default for RenderLoop {
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
    fn test_render_loop_creation() {
        let rl = RenderLoop::new();
        assert!(rl.pass_count() > 0);
        assert_eq!(rl.frame_index(), 0);
    }

    #[test]
    fn test_execute_frame() {
        let mut rl = RenderLoop::new();
        let frame = FrameData {
            visible_entity_count: 100,
            light_count: 2,
            ..Default::default()
        };
        let stats = rl.execute_frame(&frame);
        assert!(stats.passes_executed > 0);
        assert_eq!(rl.frame_index(), 1);
    }

    #[test]
    fn test_pass_disable() {
        let mut rl = RenderLoop::new();
        rl.set_pass_enabled(RenderPassId::Bloom, false);
        let frame = FrameData {
            visible_entity_count: 10,
            light_count: 1,
            ..Default::default()
        };
        let stats = rl.execute_frame(&frame);
        assert!(stats.passes_skipped > 0);
    }

    #[test]
    fn test_config_update() {
        let mut rl = RenderLoop::new();
        let mut config = rl.config().clone();
        config.bloom_enabled = false;
        config.fxaa_enabled = false;
        rl.set_config(config);
        assert!(!rl.config().bloom_enabled);
        assert!(!rl.config().fxaa_enabled);
    }

    #[test]
    fn test_stats_history() {
        let mut rl = RenderLoop::new();
        let frame = FrameData::default();
        for _ in 0..5 {
            rl.execute_frame(&frame);
        }
        assert_eq!(rl.stats_history().len(), 5);
        assert_eq!(rl.frame_index(), 5);
    }

    #[test]
    fn test_pass_names() {
        let rl = RenderLoop::new();
        let names = rl.pass_names();
        assert!(names.contains(&"Shadow Map"));
        assert!(names.contains(&"Present"));
    }

    #[test]
    fn test_estimated_fps() {
        let rl = RenderLoop::new();
        assert!(rl.estimated_fps() > 0.0);
    }

    #[test]
    fn test_pass_result_constructors() {
        let s = PassResult::success(RenderPassId::ForwardOpaque, 100, 50, 10000);
        assert!(s.success);
        assert_eq!(s.draw_calls, 50);

        let f = PassResult::failed(RenderPassId::ShadowMap, "GPU error".to_string());
        assert!(!f.success);
        assert!(f.error.is_some());

        let sk = PassResult::skipped(RenderPassId::Bloom);
        assert!(sk.success);
        assert_eq!(sk.draw_calls, 0);
    }
}
