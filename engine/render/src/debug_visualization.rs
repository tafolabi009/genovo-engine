// engine/render/src/debug_visualization.rs
//
// Runtime debug overlays and visualization modes for the renderer.
// Provides wireframe, normals, UVs, overdraw, light complexity,
// shadow cascade, depth, motion vectors, mip level, light probes,
// and cluster visualizations. Also includes a HUD overlay for FPS,
// GPU memory, draw call stats, and per-pass timing.
//
// # Usage
//
// The `DebugVisualization` struct holds the active debug mode and overlay
// settings. Each frame, the renderer queries it to decide which debug
// shaders to bind and whether to draw the stats overlay.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Debug visualization modes
// ---------------------------------------------------------------------------

/// Available debug visualization modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DebugMode {
    /// No debug visualization (normal rendering).
    None,
    /// Render everything as wireframe.
    Wireframe,
    /// Visualize world-space normals as RGB.
    Normals,
    /// Checker pattern on all surfaces (UV debugging).
    UVs,
    /// Heat-map of overdraw count.
    Overdraw,
    /// Color by number of lights affecting each pixel.
    LightComplexity,
    /// Color each shadow cascade a different color.
    ShadowCascades,
    /// Linearized depth visualization.
    Depth,
    /// Visualize motion vectors as colours.
    MotionVectors,
    /// Color by texture mip level used.
    MipLevel,
    /// Render light probe locations and SH as spheres.
    LightProbes,
    /// Visualize the light cluster grid.
    Clusters,
    /// Visualize ambient occlusion buffer.
    AmbientOcclusion,
    /// Visualize G-Buffer albedo.
    GBufferAlbedo,
    /// Visualize G-Buffer normals.
    GBufferNormals,
    /// Visualize G-Buffer metallic/roughness.
    GBufferMetallicRoughness,
    /// Visualize G-Buffer emissive.
    GBufferEmissive,
    /// Visualize screen-space reflections.
    SSR,
}

impl DebugMode {
    /// All available modes (for UI enumeration).
    pub fn all() -> &'static [DebugMode] {
        &[
            DebugMode::None,
            DebugMode::Wireframe,
            DebugMode::Normals,
            DebugMode::UVs,
            DebugMode::Overdraw,
            DebugMode::LightComplexity,
            DebugMode::ShadowCascades,
            DebugMode::Depth,
            DebugMode::MotionVectors,
            DebugMode::MipLevel,
            DebugMode::LightProbes,
            DebugMode::Clusters,
            DebugMode::AmbientOcclusion,
            DebugMode::GBufferAlbedo,
            DebugMode::GBufferNormals,
            DebugMode::GBufferMetallicRoughness,
            DebugMode::GBufferEmissive,
            DebugMode::SSR,
        ]
    }

    /// User-friendly name.
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Wireframe => "Wireframe",
            Self::Normals => "Normals",
            Self::UVs => "UV Checker",
            Self::Overdraw => "Overdraw",
            Self::LightComplexity => "Light Complexity",
            Self::ShadowCascades => "Shadow Cascades",
            Self::Depth => "Depth",
            Self::MotionVectors => "Motion Vectors",
            Self::MipLevel => "Mip Level",
            Self::LightProbes => "Light Probes",
            Self::Clusters => "Light Clusters",
            Self::AmbientOcclusion => "Ambient Occlusion",
            Self::GBufferAlbedo => "GBuffer Albedo",
            Self::GBufferNormals => "GBuffer Normals",
            Self::GBufferMetallicRoughness => "GBuffer Metal/Rough",
            Self::GBufferEmissive => "GBuffer Emissive",
            Self::SSR => "SSR",
        }
    }

    /// Whether this mode requires a fullscreen pass (as opposed to modifying
    /// the geometry pass).
    pub fn is_fullscreen_pass(self) -> bool {
        matches!(
            self,
            Self::Overdraw
                | Self::LightComplexity
                | Self::ShadowCascades
                | Self::Depth
                | Self::MotionVectors
                | Self::MipLevel
                | Self::Clusters
                | Self::AmbientOcclusion
                | Self::GBufferAlbedo
                | Self::GBufferNormals
                | Self::GBufferMetallicRoughness
                | Self::GBufferEmissive
                | Self::SSR
        )
    }

    /// Whether this mode modifies the geometry pass pipeline state.
    pub fn modifies_geometry_pass(self) -> bool {
        matches!(self, Self::Wireframe | Self::Normals | Self::UVs)
    }

    /// Get the WGSL shader source for this debug mode.
    pub fn shader_source(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Wireframe => WGSL_DEBUG_WIREFRAME,
            Self::Normals => WGSL_DEBUG_NORMALS,
            Self::UVs => WGSL_DEBUG_UVS,
            Self::Overdraw => WGSL_DEBUG_OVERDRAW,
            Self::LightComplexity => WGSL_DEBUG_LIGHT_COMPLEXITY,
            Self::ShadowCascades => WGSL_DEBUG_SHADOW_CASCADES,
            Self::Depth => WGSL_DEBUG_DEPTH,
            Self::MotionVectors => WGSL_DEBUG_MOTION_VECTORS,
            Self::MipLevel => WGSL_DEBUG_MIP_LEVEL,
            Self::LightProbes => WGSL_DEBUG_LIGHT_PROBES,
            Self::Clusters => WGSL_DEBUG_CLUSTERS,
            Self::AmbientOcclusion => WGSL_DEBUG_AO,
            Self::GBufferAlbedo => WGSL_DEBUG_GBUFFER_ALBEDO,
            Self::GBufferNormals => WGSL_DEBUG_GBUFFER_NORMALS,
            Self::GBufferMetallicRoughness => WGSL_DEBUG_GBUFFER_MR,
            Self::GBufferEmissive => WGSL_DEBUG_GBUFFER_EMISSIVE,
            Self::SSR => WGSL_DEBUG_SSR,
        }
    }
}

impl fmt::Display for DebugMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// Overlay settings
// ---------------------------------------------------------------------------

/// Which stats to show in the debug overlay.
#[derive(Debug, Clone)]
pub struct OverlaySettings {
    /// Show FPS counter.
    pub show_fps: bool,
    /// Show frame time graph.
    pub show_frame_time_graph: bool,
    /// Number of frame time samples in the graph.
    pub frame_time_graph_samples: usize,
    /// Show GPU memory usage.
    pub show_gpu_memory: bool,
    /// Show draw call count.
    pub show_draw_calls: bool,
    /// Show triangle count.
    pub show_triangles: bool,
    /// Show per-pass timing.
    pub show_pass_timing: bool,
    /// Show state change count.
    pub show_state_changes: bool,
    /// Show visible/culled object counts.
    pub show_culling_stats: bool,
    /// Show render resolution.
    pub show_resolution: bool,
    /// Overlay position.
    pub position: OverlayPosition,
    /// Overlay opacity (0..1).
    pub opacity: f32,
    /// Font size in pixels.
    pub font_size: f32,
    /// Background color (RGBA).
    pub background_color: [f32; 4],
    /// Text color (RGBA).
    pub text_color: [f32; 4],
}

impl Default for OverlaySettings {
    fn default() -> Self {
        Self {
            show_fps: true,
            show_frame_time_graph: false,
            frame_time_graph_samples: 120,
            show_gpu_memory: true,
            show_draw_calls: true,
            show_triangles: true,
            show_pass_timing: false,
            show_state_changes: false,
            show_culling_stats: false,
            show_resolution: false,
            position: OverlayPosition::TopLeft,
            opacity: 0.8,
            font_size: 14.0,
            background_color: [0.0, 0.0, 0.0, 0.7],
            text_color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

/// Position of the debug overlay on screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayPosition {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    TopCenter,
    BottomCenter,
}

impl OverlayPosition {
    /// Compute the screen-space origin for the overlay.
    /// Returns (x, y) in pixels from the top-left corner.
    pub fn compute_origin(
        self,
        screen_width: f32,
        screen_height: f32,
        overlay_width: f32,
        overlay_height: f32,
        margin: f32,
    ) -> (f32, f32) {
        match self {
            Self::TopLeft => (margin, margin),
            Self::TopRight => (screen_width - overlay_width - margin, margin),
            Self::BottomLeft => (margin, screen_height - overlay_height - margin),
            Self::BottomRight => (
                screen_width - overlay_width - margin,
                screen_height - overlay_height - margin,
            ),
            Self::TopCenter => ((screen_width - overlay_width) * 0.5, margin),
            Self::BottomCenter => (
                (screen_width - overlay_width) * 0.5,
                screen_height - overlay_height - margin,
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// DebugOverlay (runtime data)
// ---------------------------------------------------------------------------

/// Runtime data for the debug overlay HUD.
pub struct DebugOverlay {
    /// Circular buffer of frame times (in ms).
    frame_times: Vec<f32>,
    /// Current write index in the circular buffer.
    frame_time_index: usize,
    /// Smoothed FPS value.
    smoothed_fps: f32,
    /// Smoothed frame time (ms).
    smoothed_frame_time: f32,
    /// Alpha for exponential smoothing.
    smoothing_alpha: f32,
    /// Per-pass timing data from last frame.
    pass_timings: Vec<(String, f64)>,
    /// GPU memory usage (MB).
    gpu_memory_used: f32,
    /// GPU memory budget (MB).
    gpu_memory_budget: f32,
    /// Draw call count.
    draw_calls: u32,
    /// Triangle count.
    triangles: u32,
    /// State changes.
    state_changes: u32,
    /// Visible objects.
    visible_objects: u32,
    /// Culled objects.
    culled_objects: u32,
    /// Render resolution.
    render_width: u32,
    render_height: u32,
}

impl DebugOverlay {
    /// Create a new debug overlay.
    pub fn new(max_samples: usize) -> Self {
        Self {
            frame_times: vec![0.0; max_samples],
            frame_time_index: 0,
            smoothed_fps: 0.0,
            smoothed_frame_time: 0.0,
            smoothing_alpha: 0.05,
            pass_timings: Vec::new(),
            gpu_memory_used: 0.0,
            gpu_memory_budget: 0.0,
            draw_calls: 0,
            triangles: 0,
            state_changes: 0,
            visible_objects: 0,
            culled_objects: 0,
            render_width: 0,
            render_height: 0,
        }
    }

    /// Update with a new frame's data.
    pub fn update(&mut self, frame_time_ms: f32, stats: &FrameDebugStats) {
        // Update circular buffer.
        self.frame_times[self.frame_time_index] = frame_time_ms;
        self.frame_time_index = (self.frame_time_index + 1) % self.frame_times.len();

        // Exponential smoothing.
        let alpha = self.smoothing_alpha;
        self.smoothed_frame_time =
            alpha * frame_time_ms + (1.0 - alpha) * self.smoothed_frame_time;
        self.smoothed_fps = if self.smoothed_frame_time > 0.0 {
            1000.0 / self.smoothed_frame_time
        } else {
            0.0
        };

        // Update stats.
        self.pass_timings = stats.pass_timings.clone();
        self.gpu_memory_used = stats.gpu_memory_used_mb;
        self.gpu_memory_budget = stats.gpu_memory_budget_mb;
        self.draw_calls = stats.draw_calls;
        self.triangles = stats.triangles;
        self.state_changes = stats.state_changes;
        self.visible_objects = stats.visible_objects;
        self.culled_objects = stats.culled_objects;
        self.render_width = stats.render_width;
        self.render_height = stats.render_height;
    }

    /// Get the current smoothed FPS.
    pub fn fps(&self) -> f32 {
        self.smoothed_fps
    }

    /// Get the current smoothed frame time in ms.
    pub fn frame_time_ms(&self) -> f32 {
        self.smoothed_frame_time
    }

    /// Get the frame time graph data.
    pub fn frame_time_graph(&self) -> &[f32] {
        &self.frame_times
    }

    /// Get the min/max/avg frame times over the buffer.
    pub fn frame_time_stats(&self) -> (f32, f32, f32) {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        let mut sum = 0.0f32;
        let mut count = 0;
        for &t in &self.frame_times {
            if t > 0.0 {
                min = min.min(t);
                max = max.max(t);
                sum += t;
                count += 1;
            }
        }
        let avg = if count > 0 { sum / count as f32 } else { 0.0 };
        (min, max, avg)
    }

    /// Build text lines for the overlay.
    pub fn build_text(&self, settings: &OverlaySettings) -> Vec<String> {
        let mut lines = Vec::new();

        if settings.show_fps {
            lines.push(format!(
                "FPS: {:.0}  ({:.2} ms)",
                self.smoothed_fps, self.smoothed_frame_time
            ));
        }

        if settings.show_resolution {
            lines.push(format!(
                "Resolution: {}x{}",
                self.render_width, self.render_height
            ));
        }

        if settings.show_gpu_memory {
            let usage_pct = if self.gpu_memory_budget > 0.0 {
                (self.gpu_memory_used / self.gpu_memory_budget) * 100.0
            } else {
                0.0
            };
            lines.push(format!(
                "GPU Memory: {:.0}/{:.0} MB ({:.0}%)",
                self.gpu_memory_used, self.gpu_memory_budget, usage_pct
            ));
        }

        if settings.show_draw_calls {
            lines.push(format!("Draw Calls: {}", self.draw_calls));
        }

        if settings.show_triangles {
            let tri_str = if self.triangles >= 1_000_000 {
                format!("{:.1}M", self.triangles as f32 / 1_000_000.0)
            } else if self.triangles >= 1_000 {
                format!("{:.1}K", self.triangles as f32 / 1_000.0)
            } else {
                format!("{}", self.triangles)
            };
            lines.push(format!("Triangles: {}", tri_str));
        }

        if settings.show_state_changes {
            lines.push(format!("State Changes: {}", self.state_changes));
        }

        if settings.show_culling_stats {
            lines.push(format!(
                "Objects: {} visible, {} culled",
                self.visible_objects, self.culled_objects
            ));
        }

        if settings.show_pass_timing {
            lines.push("--- Pass Timing ---".to_string());
            for (name, time_ms) in &self.pass_timings {
                lines.push(format!("  {}: {:.2} ms", name, time_ms));
            }
        }

        if settings.show_frame_time_graph {
            let (min, max, avg) = self.frame_time_stats();
            lines.push(format!(
                "Frame Time: min={:.1} max={:.1} avg={:.1} ms",
                min, max, avg
            ));
        }

        lines
    }
}

impl Default for DebugOverlay {
    fn default() -> Self {
        Self::new(120)
    }
}

/// Statistics fed into the debug overlay each frame.
#[derive(Debug, Clone, Default)]
pub struct FrameDebugStats {
    pub pass_timings: Vec<(String, f64)>,
    pub gpu_memory_used_mb: f32,
    pub gpu_memory_budget_mb: f32,
    pub draw_calls: u32,
    pub triangles: u32,
    pub state_changes: u32,
    pub visible_objects: u32,
    pub culled_objects: u32,
    pub render_width: u32,
    pub render_height: u32,
}

// ---------------------------------------------------------------------------
// DebugVisualization (master controller)
// ---------------------------------------------------------------------------

/// Master controller for debug visualizations and overlays.
pub struct DebugVisualization {
    /// Current debug visualization mode.
    mode: DebugMode,
    /// Overlay settings.
    overlay_settings: OverlaySettings,
    /// Runtime overlay data.
    overlay: DebugOverlay,
    /// Whether the overlay is visible.
    overlay_visible: bool,
    /// Custom parameters for the active debug shader.
    debug_params: HashMap<String, f32>,
    /// Color map for heat-map visualizations.
    heat_map_colors: Vec<[f32; 4]>,
    /// Cascade visualization colors.
    cascade_colors: Vec<[f32; 4]>,
    /// Whether wireframe overlay is additive (draws wireframe on top of
    /// normal shading).
    wireframe_overlay: bool,
    /// Wireframe line width (if supported).
    wireframe_line_width: f32,
    /// Depth visualization range (near, far).
    depth_range: (f32, f32),
    /// Motion vector scale (for visualization).
    motion_vector_scale: f32,
}

impl DebugVisualization {
    /// Create with no debug visualization active.
    pub fn new() -> Self {
        Self {
            mode: DebugMode::None,
            overlay_settings: OverlaySettings::default(),
            overlay: DebugOverlay::new(120),
            overlay_visible: false,
            debug_params: HashMap::new(),
            heat_map_colors: Self::default_heat_map(),
            cascade_colors: Self::default_cascade_colors(),
            wireframe_overlay: false,
            wireframe_line_width: 1.0,
            depth_range: (0.1, 1000.0),
            motion_vector_scale: 10.0,
        }
    }

    /// Default heat map gradient (blue -> green -> yellow -> red -> white).
    fn default_heat_map() -> Vec<[f32; 4]> {
        vec![
            [0.0, 0.0, 0.5, 1.0], // dark blue (0 overdraw / 0 lights)
            [0.0, 0.0, 1.0, 1.0], // blue
            [0.0, 1.0, 0.0, 1.0], // green
            [1.0, 1.0, 0.0, 1.0], // yellow
            [1.0, 0.5, 0.0, 1.0], // orange
            [1.0, 0.0, 0.0, 1.0], // red
            [1.0, 1.0, 1.0, 1.0], // white (very high)
        ]
    }

    /// Default cascade colors.
    fn default_cascade_colors() -> Vec<[f32; 4]> {
        vec![
            [1.0, 0.0, 0.0, 0.3], // cascade 0: red
            [0.0, 1.0, 0.0, 0.3], // cascade 1: green
            [0.0, 0.0, 1.0, 0.3], // cascade 2: blue
            [1.0, 1.0, 0.0, 0.3], // cascade 3: yellow
            [1.0, 0.0, 1.0, 0.3], // cascade 4: magenta
            [0.0, 1.0, 1.0, 0.3], // cascade 5: cyan
        ]
    }

    /// Set the active debug mode.
    pub fn set_mode(&mut self, mode: DebugMode) {
        self.mode = mode;
    }

    /// Get the active debug mode.
    pub fn mode(&self) -> DebugMode {
        self.mode
    }

    /// Cycle to the next debug mode.
    pub fn cycle_mode(&mut self) {
        let modes = DebugMode::all();
        let current_idx = modes
            .iter()
            .position(|&m| m == self.mode)
            .unwrap_or(0);
        let next_idx = (current_idx + 1) % modes.len();
        self.mode = modes[next_idx];
    }

    /// Cycle to the previous debug mode.
    pub fn cycle_mode_reverse(&mut self) {
        let modes = DebugMode::all();
        let current_idx = modes
            .iter()
            .position(|&m| m == self.mode)
            .unwrap_or(0);
        let prev_idx = if current_idx == 0 {
            modes.len() - 1
        } else {
            current_idx - 1
        };
        self.mode = modes[prev_idx];
    }

    /// Toggle the debug overlay visibility.
    pub fn toggle_overlay(&mut self) {
        self.overlay_visible = !self.overlay_visible;
    }

    /// Set overlay visibility.
    pub fn set_overlay_visible(&mut self, visible: bool) {
        self.overlay_visible = visible;
    }

    /// Whether the overlay is visible.
    pub fn is_overlay_visible(&self) -> bool {
        self.overlay_visible
    }

    /// Get the overlay settings.
    pub fn overlay_settings(&self) -> &OverlaySettings {
        &self.overlay_settings
    }

    /// Get a mutable reference to overlay settings.
    pub fn overlay_settings_mut(&mut self) -> &mut OverlaySettings {
        &mut self.overlay_settings
    }

    /// Update the overlay with new frame data.
    pub fn update_overlay(&mut self, frame_time_ms: f32, stats: &FrameDebugStats) {
        self.overlay.update(frame_time_ms, stats);
    }

    /// Get the overlay text lines.
    pub fn overlay_text(&self) -> Vec<String> {
        self.overlay.build_text(&self.overlay_settings)
    }

    /// Get the overlay data.
    pub fn overlay(&self) -> &DebugOverlay {
        &self.overlay
    }

    /// Set a debug shader parameter.
    pub fn set_param(&mut self, key: &str, value: f32) {
        self.debug_params.insert(key.to_string(), value);
    }

    /// Get a debug shader parameter.
    pub fn param(&self, key: &str) -> Option<f32> {
        self.debug_params.get(key).copied()
    }

    /// Get the heat map colors.
    pub fn heat_map_colors(&self) -> &[[f32; 4]] {
        &self.heat_map_colors
    }

    /// Set custom heat map colors.
    pub fn set_heat_map_colors(&mut self, colors: Vec<[f32; 4]>) {
        self.heat_map_colors = colors;
    }

    /// Get the cascade colors.
    pub fn cascade_colors(&self) -> &[[f32; 4]] {
        &self.cascade_colors
    }

    /// Set wireframe overlay mode.
    pub fn set_wireframe_overlay(&mut self, overlay: bool) {
        self.wireframe_overlay = overlay;
    }

    /// Whether wireframe is in overlay mode.
    pub fn is_wireframe_overlay(&self) -> bool {
        self.wireframe_overlay
    }

    /// Set the depth visualization range.
    pub fn set_depth_range(&mut self, near: f32, far: f32) {
        self.depth_range = (near, far);
    }

    /// Get the depth visualization range.
    pub fn depth_range(&self) -> (f32, f32) {
        self.depth_range
    }

    /// Set the motion vector display scale.
    pub fn set_motion_vector_scale(&mut self, scale: f32) {
        self.motion_vector_scale = scale;
    }

    /// Get the motion vector display scale.
    pub fn motion_vector_scale(&self) -> f32 {
        self.motion_vector_scale
    }

    /// Get the WGSL shader source for the current debug mode.
    pub fn current_shader(&self) -> &'static str {
        self.mode.shader_source()
    }

    /// Whether any debug visualization is active.
    pub fn is_active(&self) -> bool {
        self.mode != DebugMode::None
    }

    // -----------------------------------------------------------------------
    // GPU uniform data
    // -----------------------------------------------------------------------

    /// Build the debug uniform buffer data for the GPU.
    pub fn build_uniform_data(&self) -> DebugUniformData {
        DebugUniformData {
            mode: self.mode as u32,
            depth_near: self.depth_range.0,
            depth_far: self.depth_range.1,
            motion_vector_scale: self.motion_vector_scale,
            wireframe_overlay: if self.wireframe_overlay { 1 } else { 0 },
            wireframe_line_width: self.wireframe_line_width,
            heat_map_steps: self.heat_map_colors.len() as u32,
            cascade_count: self.cascade_colors.len() as u32,
            heat_map_color_0: self.heat_map_colors.first().copied().unwrap_or([0.0; 4]),
            heat_map_color_1: self.heat_map_colors.get(1).copied().unwrap_or([0.0; 4]),
            heat_map_color_2: self.heat_map_colors.get(2).copied().unwrap_or([0.0; 4]),
            heat_map_color_3: self.heat_map_colors.get(3).copied().unwrap_or([0.0; 4]),
            heat_map_color_4: self.heat_map_colors.get(4).copied().unwrap_or([0.0; 4]),
            heat_map_color_5: self.heat_map_colors.get(5).copied().unwrap_or([0.0; 4]),
            heat_map_color_6: self.heat_map_colors.get(6).copied().unwrap_or([0.0; 4]),
            cascade_color_0: self.cascade_colors.first().copied().unwrap_or([0.0; 4]),
            cascade_color_1: self.cascade_colors.get(1).copied().unwrap_or([0.0; 4]),
            cascade_color_2: self.cascade_colors.get(2).copied().unwrap_or([0.0; 4]),
            cascade_color_3: self.cascade_colors.get(3).copied().unwrap_or([0.0; 4]),
            cascade_color_4: self.cascade_colors.get(4).copied().unwrap_or([0.0; 4]),
            cascade_color_5: self.cascade_colors.get(5).copied().unwrap_or([0.0; 4]),
            _padding: [0.0; 2],
        }
    }

    /// Sample the heat map at a normalised value (0..1).
    pub fn sample_heat_map(&self, t: f32) -> [f32; 4] {
        if self.heat_map_colors.is_empty() {
            return [1.0, 1.0, 1.0, 1.0];
        }
        if self.heat_map_colors.len() == 1 {
            return self.heat_map_colors[0];
        }

        let t = t.clamp(0.0, 1.0);
        let max_idx = (self.heat_map_colors.len() - 1) as f32;
        let scaled = t * max_idx;
        let idx0 = scaled.floor() as usize;
        let idx1 = (idx0 + 1).min(self.heat_map_colors.len() - 1);
        let frac = scaled - scaled.floor();

        let c0 = self.heat_map_colors[idx0];
        let c1 = self.heat_map_colors[idx1];

        [
            c0[0] + (c1[0] - c0[0]) * frac,
            c0[1] + (c1[1] - c0[1]) * frac,
            c0[2] + (c1[2] - c0[2]) * frac,
            c0[3] + (c1[3] - c0[3]) * frac,
        ]
    }
}

impl Default for DebugVisualization {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU-uploadable debug uniform data.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DebugUniformData {
    pub mode: u32,
    pub depth_near: f32,
    pub depth_far: f32,
    pub motion_vector_scale: f32,
    pub wireframe_overlay: u32,
    pub wireframe_line_width: f32,
    pub heat_map_steps: u32,
    pub cascade_count: u32,
    pub heat_map_color_0: [f32; 4],
    pub heat_map_color_1: [f32; 4],
    pub heat_map_color_2: [f32; 4],
    pub heat_map_color_3: [f32; 4],
    pub heat_map_color_4: [f32; 4],
    pub heat_map_color_5: [f32; 4],
    pub heat_map_color_6: [f32; 4],
    pub cascade_color_0: [f32; 4],
    pub cascade_color_1: [f32; 4],
    pub cascade_color_2: [f32; 4],
    pub cascade_color_3: [f32; 4],
    pub cascade_color_4: [f32; 4],
    pub cascade_color_5: [f32; 4],
    pub _padding: [f32; 2],
}

// ---------------------------------------------------------------------------
// WGSL debug shaders
// ---------------------------------------------------------------------------

/// Wireframe debug shader -- outputs a solid color per triangle edge.
pub const WGSL_DEBUG_WIREFRAME: &str = r#"
// Debug wireframe shader.
// Uses barycentric coordinates to draw triangle edges.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) bary: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @builtin(vertex_index) vid: u32,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.mvp * vec4<f32>(pos, 1.0);
    // Assign barycentric coords based on vertex index within triangle.
    let bary_idx = vid % 3u;
    out.bary = vec3<f32>(
        select(0.0, 1.0, bary_idx == 0u),
        select(0.0, 1.0, bary_idx == 1u),
        select(0.0, 1.0, bary_idx == 2u),
    );
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = min(in.bary.x, min(in.bary.y, in.bary.z));
    let line_width = 0.02;
    let edge = smoothstep(0.0, line_width, d);
    let wire_color = vec3<f32>(0.0, 1.0, 0.0);
    let fill_color = vec3<f32>(0.05, 0.05, 0.05);
    let color = mix(wire_color, fill_color, edge);
    return vec4<f32>(color, 1.0);
}
"#;

/// Normal visualization shader -- maps world normals to RGB.
pub const WGSL_DEBUG_NORMALS: &str = r#"
// Debug normals shader.
// Maps world-space normals to RGB: X->R, Y->G, Z->B.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.mvp * vec4<f32>(pos, 1.0);
    out.normal = (uniforms.model * vec4<f32>(normal, 0.0)).xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let color = n * 0.5 + 0.5;
    return vec4<f32>(color, 1.0);
}
"#;

/// UV checker pattern shader.
pub const WGSL_DEBUG_UVS: &str = r#"
// Debug UV shader.
// Renders a checker pattern based on UV coordinates.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.mvp * vec4<f32>(pos, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let scale = 8.0;
    let checker = floor(in.uv * scale);
    let parity = (checker.x + checker.y) % 2.0;
    let c = select(0.8, 0.2, parity > 0.5);
    let uv_color = vec3<f32>(in.uv.x, in.uv.y, 0.0);
    let color = mix(vec3<f32>(c, c, c), uv_color, 0.3);
    return vec4<f32>(color, 1.0);
}
"#;

/// Overdraw visualization (fullscreen).
pub const WGSL_DEBUG_OVERDRAW: &str = r#"
// Debug overdraw heatmap shader.
// Reads the overdraw count buffer and maps to heat-map colors.

@group(0) @binding(0) var overdraw_tex: texture_2d<u32>;
@group(0) @binding(1) var<uniform> debug: DebugUniforms;

struct DebugUniforms {
    mode: u32,
    depth_near: f32,
    depth_far: f32,
    motion_vector_scale: f32,
    heat_map_steps: u32,
};

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(frag_coord.xy);
    let count = textureLoad(overdraw_tex, coord, 0).r;
    let t = clamp(f32(count) / 10.0, 0.0, 1.0);
    // Simple gradient: blue -> green -> red.
    var color: vec3<f32>;
    if (t < 0.5) {
        color = mix(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), t * 2.0);
    } else {
        color = mix(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), (t - 0.5) * 2.0);
    }
    return vec4<f32>(color, 1.0);
}
"#;

/// Light complexity visualization.
pub const WGSL_DEBUG_LIGHT_COMPLEXITY: &str = r#"
// Debug light complexity shader.
// Reads the light count per pixel from the light grid and maps to colors.

@group(0) @binding(0) var light_count_tex: texture_2d<u32>;

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(frag_coord.xy);
    let count = textureLoad(light_count_tex, coord, 0).r;
    let t = clamp(f32(count) / 32.0, 0.0, 1.0);
    var color: vec3<f32>;
    if (t < 0.25) {
        color = mix(vec3<f32>(0.0, 0.0, 0.2), vec3<f32>(0.0, 0.0, 1.0), t * 4.0);
    } else if (t < 0.5) {
        color = mix(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), (t - 0.25) * 4.0);
    } else if (t < 0.75) {
        color = mix(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), (t - 0.5) * 4.0);
    } else {
        color = mix(vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), (t - 0.75) * 4.0);
    }
    return vec4<f32>(color, 1.0);
}
"#;

/// Shadow cascade visualization.
pub const WGSL_DEBUG_SHADOW_CASCADES: &str = r#"
// Debug shadow cascade shader.
// Colors each pixel by which shadow cascade it falls into.

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var<uniform> camera: CameraUniforms;

struct CameraUniforms {
    inv_view_proj: mat4x4<f32>,
    cascade_splits: vec4<f32>,
    near: f32,
    far: f32,
};

fn linearize_depth(d: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - d * (far - near));
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(frag_coord.xy);
    let depth = textureLoad(depth_tex, coord, 0);
    let linear_z = linearize_depth(depth, camera.near, camera.far);

    var color: vec3<f32>;
    if (linear_z < camera.cascade_splits.x) {
        color = vec3<f32>(1.0, 0.0, 0.0); // cascade 0: red
    } else if (linear_z < camera.cascade_splits.y) {
        color = vec3<f32>(0.0, 1.0, 0.0); // cascade 1: green
    } else if (linear_z < camera.cascade_splits.z) {
        color = vec3<f32>(0.0, 0.0, 1.0); // cascade 2: blue
    } else if (linear_z < camera.cascade_splits.w) {
        color = vec3<f32>(1.0, 1.0, 0.0); // cascade 3: yellow
    } else {
        color = vec3<f32>(0.5, 0.5, 0.5); // beyond cascades
    }
    return vec4<f32>(color, 0.3);
}
"#;

/// Depth visualization.
pub const WGSL_DEBUG_DEPTH: &str = r#"
// Debug linearized depth shader.
// Visualizes the depth buffer as a greyscale gradient.

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var<uniform> debug: DebugDepthUniforms;

struct DebugDepthUniforms {
    near: f32,
    far: f32,
};

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(frag_coord.xy);
    let depth = textureLoad(depth_tex, coord, 0);
    let linear = debug.near * debug.far / (debug.far - depth * (debug.far - debug.near));
    let t = clamp((linear - debug.near) / (debug.far - debug.near), 0.0, 1.0);
    let inv = 1.0 - t;
    return vec4<f32>(inv, inv, inv, 1.0);
}
"#;

/// Motion vector visualization.
pub const WGSL_DEBUG_MOTION_VECTORS: &str = r#"
// Debug motion vectors shader.
// Visualizes 2D motion vectors as colours: X->R, Y->G.

@group(0) @binding(0) var motion_tex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> debug: DebugMotionUniforms;

struct DebugMotionUniforms {
    scale: f32,
};

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(frag_coord.xy);
    let mv = textureLoad(motion_tex, coord, 0).rg;
    let scaled = mv * debug.scale;
    let color = vec3<f32>(
        abs(scaled.x),
        abs(scaled.y),
        0.2,
    );
    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
"#;

/// Mip level visualization.
pub const WGSL_DEBUG_MIP_LEVEL: &str = r#"
// Debug mip level shader.
// Colors surfaces by which mip level the GPU would select.
// Requires textureQueryLod or derivative-based estimation.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var albedo_tex: texture_2d<f32>;
@group(0) @binding(1) var albedo_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Estimate mip level from UV derivatives.
    let dx = dpdx(in.uv);
    let dy = dpdy(in.uv);
    let tex_size = vec2<f32>(textureDimensions(albedo_tex, 0));
    let delta = max(length(dx * tex_size), length(dy * tex_size));
    let mip = log2(max(delta, 1.0));
    let max_mip = log2(max(tex_size.x, tex_size.y));
    let t = clamp(mip / max_mip, 0.0, 1.0);

    // Color gradient: green (mip 0) -> yellow -> red (high mip).
    var color: vec3<f32>;
    if (t < 0.5) {
        color = mix(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), t * 2.0);
    } else {
        color = mix(vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), (t - 0.5) * 2.0);
    }
    return vec4<f32>(color, 1.0);
}
"#;

/// Light probe visualization.
pub const WGSL_DEBUG_LIGHT_PROBES: &str = r#"
// Debug light probes shader.
// Renders each probe as a sphere colored by its dominant SH direction.

struct ProbeInstance {
    position: vec3<f32>,
    radius: f32,
    sh_r: vec4<f32>,
    sh_g: vec4<f32>,
    sh_b: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> probes: array<ProbeInstance>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @builtin(instance_index) iid: u32,
) -> VertexOutput {
    let probe = probes[iid];
    let world_pos = pos * probe.radius + probe.position;

    var out: VertexOutput;
    out.position = uniforms.vp * vec4<f32>(world_pos, 1.0);
    // Use L0 (DC) SH coefficient as color.
    let sh_scale = 0.282095;
    out.color = vec3<f32>(
        probe.sh_r.x * sh_scale,
        probe.sh_g.x * sh_scale,
        probe.sh_b.x * sh_scale,
    );
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(max(in.color, vec3<f32>(0.0)), 0.7);
}
"#;

/// Light cluster visualization.
pub const WGSL_DEBUG_CLUSTERS: &str = r#"
// Debug cluster visualization shader.
// Colors each pixel by its cluster index using a hash.

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var<uniform> cluster_info: ClusterInfo;

struct ClusterInfo {
    tile_size: vec2<u32>,
    num_slices: u32,
    near: f32,
    far: f32,
    log_far_over_near: f32,
};

fn hash_u32(n: u32) -> vec3<f32> {
    var x = n;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = (x >> 16u) ^ x;
    return vec3<f32>(
        f32((x >> 0u) & 0xFFu) / 255.0,
        f32((x >> 8u) & 0xFFu) / 255.0,
        f32((x >> 16u) & 0xFFu) / 255.0,
    );
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(frag_coord.xy);
    let tile_x = u32(coord.x) / cluster_info.tile_size.x;
    let tile_y = u32(coord.y) / cluster_info.tile_size.y;
    let depth = textureLoad(depth_tex, coord, 0);
    let linear_z = cluster_info.near * cluster_info.far /
        (cluster_info.far - depth * (cluster_info.far - cluster_info.near));
    let slice = u32(log2(linear_z / cluster_info.near) /
        cluster_info.log_far_over_near * f32(cluster_info.num_slices));

    let cluster_id = tile_x + tile_y * 64u + slice * 64u * 64u;
    let color = hash_u32(cluster_id);
    return vec4<f32>(color, 0.5);
}
"#;

/// AO buffer visualization.
pub const WGSL_DEBUG_AO: &str = r#"
// Debug AO buffer shader.
// Displays the SSAO buffer as greyscale.

@group(0) @binding(0) var ao_tex: texture_2d<f32>;

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(frag_coord.xy);
    let ao = textureLoad(ao_tex, coord, 0).r;
    return vec4<f32>(ao, ao, ao, 1.0);
}
"#;

/// G-Buffer albedo visualization.
pub const WGSL_DEBUG_GBUFFER_ALBEDO: &str = r#"
// Debug G-Buffer albedo shader.

@group(0) @binding(0) var albedo_tex: texture_2d<f32>;

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(frag_coord.xy);
    return textureLoad(albedo_tex, coord, 0);
}
"#;

/// G-Buffer normals visualization.
pub const WGSL_DEBUG_GBUFFER_NORMALS: &str = r#"
// Debug G-Buffer normals shader.
// Remaps normals from [-1,1] to [0,1] for display.

@group(0) @binding(0) var normal_tex: texture_2d<f32>;

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(frag_coord.xy);
    let n = textureLoad(normal_tex, coord, 0).xyz;
    return vec4<f32>(n * 0.5 + 0.5, 1.0);
}
"#;

/// G-Buffer metallic/roughness visualization.
pub const WGSL_DEBUG_GBUFFER_MR: &str = r#"
// Debug G-Buffer metallic/roughness shader.
// R = metallic, G = roughness, B = AO.

@group(0) @binding(0) var mr_tex: texture_2d<f32>;

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(frag_coord.xy);
    return textureLoad(mr_tex, coord, 0);
}
"#;

/// G-Buffer emissive visualization.
pub const WGSL_DEBUG_GBUFFER_EMISSIVE: &str = r#"
// Debug G-Buffer emissive shader.

@group(0) @binding(0) var emissive_tex: texture_2d<f32>;

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(frag_coord.xy);
    let e = textureLoad(emissive_tex, coord, 0);
    // Tonemap for display since emissive can be HDR.
    let mapped = e.rgb / (e.rgb + vec3<f32>(1.0));
    return vec4<f32>(mapped, 1.0);
}
"#;

/// SSR visualization.
pub const WGSL_DEBUG_SSR: &str = r#"
// Debug SSR shader.
// Displays the screen-space reflections buffer.

@group(0) @binding(0) var ssr_tex: texture_2d<f32>;

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(frag_coord.xy);
    let ssr = textureLoad(ssr_tex, coord, 0);
    return vec4<f32>(ssr.rgb, 1.0);
}
"#;

// ---------------------------------------------------------------------------
// Debug draw primitives
// ---------------------------------------------------------------------------

/// A debug line to be rendered.
#[derive(Debug, Clone)]
pub struct DebugLine {
    pub start: [f32; 3],
    pub end: [f32; 3],
    pub color: [f32; 4],
    pub width: f32,
    /// Duration in seconds (0 = one frame only).
    pub duration: f32,
    /// Whether to depth-test.
    pub depth_test: bool,
}

/// A debug sphere to be rendered.
#[derive(Debug, Clone)]
pub struct DebugSphere {
    pub center: [f32; 3],
    pub radius: f32,
    pub color: [f32; 4],
    pub segments: u32,
    pub duration: f32,
    pub depth_test: bool,
}

/// A debug box (AABB) to be rendered.
#[derive(Debug, Clone)]
pub struct DebugBox {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub color: [f32; 4],
    pub duration: f32,
    pub depth_test: bool,
}

/// A debug frustum to be rendered.
#[derive(Debug, Clone)]
pub struct DebugFrustum {
    /// The 8 corners of the frustum in world space.
    pub corners: [[f32; 3]; 8],
    pub color: [f32; 4],
    pub duration: f32,
}

/// Accumulates debug draw primitives for a frame.
pub struct DebugDrawQueue {
    lines: Vec<DebugLine>,
    spheres: Vec<DebugSphere>,
    boxes: Vec<DebugBox>,
    frustums: Vec<DebugFrustum>,
    /// Remaining time for timed primitives.
    timed_lines: Vec<(DebugLine, f32)>,
    timed_spheres: Vec<(DebugSphere, f32)>,
    timed_boxes: Vec<(DebugBox, f32)>,
}

impl DebugDrawQueue {
    pub fn new() -> Self {
        Self {
            lines: Vec::new(),
            spheres: Vec::new(),
            boxes: Vec::new(),
            frustums: Vec::new(),
            timed_lines: Vec::new(),
            timed_spheres: Vec::new(),
            timed_boxes: Vec::new(),
        }
    }

    /// Add a debug line.
    pub fn line(&mut self, start: [f32; 3], end: [f32; 3], color: [f32; 4]) {
        self.lines.push(DebugLine {
            start,
            end,
            color,
            width: 1.0,
            duration: 0.0,
            depth_test: true,
        });
    }

    /// Add a debug line with duration.
    pub fn line_timed(
        &mut self,
        start: [f32; 3],
        end: [f32; 3],
        color: [f32; 4],
        duration: f32,
    ) {
        let line = DebugLine {
            start,
            end,
            color,
            width: 1.0,
            duration,
            depth_test: true,
        };
        self.timed_lines.push((line, duration));
    }

    /// Add a debug sphere.
    pub fn sphere(&mut self, center: [f32; 3], radius: f32, color: [f32; 4]) {
        self.spheres.push(DebugSphere {
            center,
            radius,
            color,
            segments: 16,
            duration: 0.0,
            depth_test: true,
        });
    }

    /// Add a debug AABB.
    pub fn aabb(&mut self, min: [f32; 3], max: [f32; 3], color: [f32; 4]) {
        self.boxes.push(DebugBox {
            min,
            max,
            color,
            duration: 0.0,
            depth_test: true,
        });
    }

    /// Add a debug frustum.
    pub fn frustum(&mut self, corners: [[f32; 3]; 8], color: [f32; 4]) {
        self.frustums.push(DebugFrustum {
            corners,
            color,
            duration: 0.0,
        });
    }

    /// Generate line vertices for all AABB boxes.
    pub fn generate_box_lines(&self) -> Vec<DebugLine> {
        let mut lines = Vec::new();
        for b in &self.boxes {
            let corners = [
                [b.min[0], b.min[1], b.min[2]],
                [b.max[0], b.min[1], b.min[2]],
                [b.max[0], b.max[1], b.min[2]],
                [b.min[0], b.max[1], b.min[2]],
                [b.min[0], b.min[1], b.max[2]],
                [b.max[0], b.min[1], b.max[2]],
                [b.max[0], b.max[1], b.max[2]],
                [b.min[0], b.max[1], b.max[2]],
            ];
            let edges: [(usize, usize); 12] = [
                (0, 1), (1, 2), (2, 3), (3, 0), // front face
                (4, 5), (5, 6), (6, 7), (7, 4), // back face
                (0, 4), (1, 5), (2, 6), (3, 7), // connecting edges
            ];
            for (a, b_idx) in &edges {
                lines.push(DebugLine {
                    start: corners[*a],
                    end: corners[*b_idx],
                    color: b.color,
                    width: 1.0,
                    duration: b.duration,
                    depth_test: b.depth_test,
                });
            }
        }
        lines
    }

    /// Generate line vertices for all frustums.
    pub fn generate_frustum_lines(&self) -> Vec<DebugLine> {
        let mut lines = Vec::new();
        for f in &self.frustums {
            // Near quad: 0-1-2-3, Far quad: 4-5-6-7.
            let near_edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
            let far_edges = [(4, 5), (5, 6), (6, 7), (7, 4)];
            let connecting = [(0, 4), (1, 5), (2, 6), (3, 7)];

            for &(a, b) in near_edges.iter().chain(far_edges.iter()).chain(connecting.iter()) {
                lines.push(DebugLine {
                    start: f.corners[a],
                    end: f.corners[b],
                    color: f.color,
                    width: 1.0,
                    duration: f.duration,
                    depth_test: false,
                });
            }
        }
        lines
    }

    /// Tick timed primitives. Removes expired ones.
    pub fn tick(&mut self, delta_time: f32) {
        // Promote timed primitives to the one-frame lists.
        for (line, _) in &self.timed_lines {
            self.lines.push(line.clone());
        }
        for (sphere, _) in &self.timed_spheres {
            self.spheres.push(sphere.clone());
        }
        for (b, _) in &self.timed_boxes {
            self.boxes.push(b.clone());
        }

        // Decay timers.
        self.timed_lines
            .iter_mut()
            .for_each(|(_, t)| *t -= delta_time);
        self.timed_spheres
            .iter_mut()
            .for_each(|(_, t)| *t -= delta_time);
        self.timed_boxes
            .iter_mut()
            .for_each(|(_, t)| *t -= delta_time);

        // Remove expired.
        self.timed_lines.retain(|(_, t)| *t > 0.0);
        self.timed_spheres.retain(|(_, t)| *t > 0.0);
        self.timed_boxes.retain(|(_, t)| *t > 0.0);
    }

    /// Clear one-frame primitives (call at end of frame).
    pub fn clear_frame(&mut self) {
        self.lines.clear();
        self.spheres.clear();
        self.boxes.clear();
        self.frustums.clear();
    }

    /// Total number of primitives this frame.
    pub fn primitive_count(&self) -> usize {
        self.lines.len() + self.spheres.len() + self.boxes.len() + self.frustums.len()
    }

    /// Get all lines (including generated from boxes/frustums).
    pub fn all_lines(&self) -> Vec<DebugLine> {
        let mut all = self.lines.clone();
        all.extend(self.generate_box_lines());
        all.extend(self.generate_frustum_lines());
        all
    }

    /// Get all spheres.
    pub fn spheres(&self) -> &[DebugSphere] {
        &self.spheres
    }
}

impl Default for DebugDrawQueue {
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
    fn test_debug_mode_basics() {
        let modes = DebugMode::all();
        assert!(modes.len() >= 18);
        assert_eq!(modes[0], DebugMode::None);
        assert_eq!(DebugMode::Wireframe.name(), "Wireframe");
        assert!(!DebugMode::Wireframe.is_fullscreen_pass());
        assert!(DebugMode::Wireframe.modifies_geometry_pass());
        assert!(DebugMode::Overdraw.is_fullscreen_pass());
    }

    #[test]
    fn test_debug_mode_cycling() {
        let mut viz = DebugVisualization::new();
        assert_eq!(viz.mode(), DebugMode::None);
        viz.cycle_mode();
        assert_eq!(viz.mode(), DebugMode::Wireframe);
        viz.cycle_mode();
        assert_eq!(viz.mode(), DebugMode::Normals);
        viz.cycle_mode_reverse();
        assert_eq!(viz.mode(), DebugMode::Wireframe);
        viz.cycle_mode_reverse();
        assert_eq!(viz.mode(), DebugMode::None);
        // Wrap around backwards.
        viz.cycle_mode_reverse();
        assert_eq!(viz.mode(), *DebugMode::all().last().unwrap());
    }

    #[test]
    fn test_overlay_toggle() {
        let mut viz = DebugVisualization::new();
        assert!(!viz.is_overlay_visible());
        viz.toggle_overlay();
        assert!(viz.is_overlay_visible());
        viz.toggle_overlay();
        assert!(!viz.is_overlay_visible());
    }

    #[test]
    fn test_overlay_text() {
        let mut overlay = DebugOverlay::new(60);
        let stats = FrameDebugStats {
            draw_calls: 150,
            triangles: 500_000,
            gpu_memory_used_mb: 512.0,
            gpu_memory_budget_mb: 2048.0,
            ..Default::default()
        };
        overlay.update(16.67, &stats);
        overlay.update(16.0, &stats);
        overlay.update(17.0, &stats);

        let settings = OverlaySettings::default();
        let lines = overlay.build_text(&settings);
        assert!(!lines.is_empty());
        // FPS line should be present.
        assert!(lines[0].contains("FPS"));
    }

    #[test]
    fn test_heat_map_sampling() {
        let viz = DebugVisualization::new();
        let c0 = viz.sample_heat_map(0.0);
        let c1 = viz.sample_heat_map(1.0);
        // At 0 we should get the first color (dark blue).
        assert!(c0[2] > c0[0]); // more blue than red
        // At 1 we should get the last color (white).
        assert!(c1[0] > 0.9);
        assert!(c1[1] > 0.9);
        assert!(c1[2] > 0.9);
    }

    #[test]
    fn test_debug_uniform_data() {
        let viz = DebugVisualization::new();
        let data = viz.build_uniform_data();
        assert_eq!(data.mode, DebugMode::None as u32);
        assert_eq!(data.depth_near, 0.1);
        assert_eq!(data.depth_far, 1000.0);
    }

    #[test]
    fn test_shader_sources_non_empty() {
        for &mode in DebugMode::all() {
            if mode == DebugMode::None {
                assert!(mode.shader_source().is_empty());
            } else {
                assert!(
                    !mode.shader_source().is_empty(),
                    "Shader for {:?} should not be empty",
                    mode
                );
            }
        }
    }

    #[test]
    fn test_overlay_position() {
        let (x, y) = OverlayPosition::TopLeft.compute_origin(1920.0, 1080.0, 200.0, 100.0, 10.0);
        assert_eq!(x, 10.0);
        assert_eq!(y, 10.0);

        let (x, y) =
            OverlayPosition::BottomRight.compute_origin(1920.0, 1080.0, 200.0, 100.0, 10.0);
        assert!((x - 1710.0).abs() < 0.01);
        assert!((y - 970.0).abs() < 0.01);
    }

    #[test]
    fn test_debug_draw_queue() {
        let mut queue = DebugDrawQueue::new();
        queue.line([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
        queue.sphere([0.0, 0.0, 0.0], 1.0, [0.0, 1.0, 0.0, 1.0]);
        queue.aabb(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        );
        assert_eq!(queue.primitive_count(), 3);

        let box_lines = queue.generate_box_lines();
        assert_eq!(box_lines.len(), 12); // 12 edges per box

        let all = queue.all_lines();
        assert_eq!(all.len(), 1 + 12); // 1 line + 12 box edges

        queue.clear_frame();
        assert_eq!(queue.primitive_count(), 0);
    }

    #[test]
    fn test_debug_draw_timed() {
        let mut queue = DebugDrawQueue::new();
        queue.line_timed(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            1.0,
        );
        assert_eq!(queue.timed_lines.len(), 1);

        queue.tick(0.5);
        // Timed line should still exist.
        assert_eq!(queue.timed_lines.len(), 1);

        queue.clear_frame();
        queue.tick(0.6);
        // Now it should be expired.
        assert_eq!(queue.timed_lines.len(), 0);
    }

    #[test]
    fn test_frustum_line_generation() {
        let mut queue = DebugDrawQueue::new();
        let corners = [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-2.0, -2.0, -10.0],
            [2.0, -2.0, -10.0],
            [2.0, 2.0, -10.0],
            [-2.0, 2.0, -10.0],
        ];
        queue.frustum(corners, [1.0, 1.0, 0.0, 1.0]);
        let lines = queue.generate_frustum_lines();
        assert_eq!(lines.len(), 12); // 4 near + 4 far + 4 connecting
    }

    #[test]
    fn test_frame_time_stats() {
        let mut overlay = DebugOverlay::new(10);
        for i in 0..10 {
            let stats = FrameDebugStats::default();
            overlay.update(10.0 + i as f32, &stats);
        }
        let (min, max, _avg) = overlay.frame_time_stats();
        assert!(min >= 10.0);
        assert!(max <= 19.0);
    }
}
