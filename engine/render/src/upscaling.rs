// engine/render/src/upscaling.rs
//
// Temporal upscaling and spatial upscaling for the Genovo engine.
// Implements FSR 1.0-style spatial upscaling (EASU + RCAS) and a full
// temporal upscaling pipeline with motion-vector reprojection, neighbourhood
// clamping, disocclusion detection, and sharpening.

use glam::{Mat4, Vec2, Vec3, Vec4};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Quality modes and configuration
// ---------------------------------------------------------------------------

/// Upscaling quality preset. Each mode defines a render resolution scale
/// factor relative to the output resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualityMode {
    /// 1.3x scale (render at ~77% resolution). Highest quality.
    Ultra,
    /// 1.5x scale (render at ~67% resolution). Good balance.
    Quality,
    /// 1.7x scale (render at ~59% resolution). Moderate quality.
    Balanced,
    /// 2.0x scale (render at 50% resolution). Maximum performance.
    Performance,
}

impl QualityMode {
    /// Get the scale factor for this quality mode.
    /// The internal render resolution is output_resolution / scale_factor.
    pub fn scale_factor(&self) -> f32 {
        match self {
            Self::Ultra => 1.3,
            Self::Quality => 1.5,
            Self::Balanced => 1.7,
            Self::Performance => 2.0,
        }
    }

    /// Compute the internal render resolution for a given output resolution.
    pub fn internal_resolution(&self, output_width: u32, output_height: u32) -> (u32, u32) {
        let factor = self.scale_factor();
        let w = ((output_width as f32 / factor).ceil() as u32).max(1);
        let h = ((output_height as f32 / factor).ceil() as u32).max(1);
        (w, h)
    }

    /// All quality modes in order from highest to lowest quality.
    pub fn all() -> &'static [QualityMode] {
        &[Self::Ultra, Self::Quality, Self::Balanced, Self::Performance]
    }
}

/// Overall upscaling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UpscaleMode {
    /// No upscaling -- render at native resolution.
    Native,
    /// FSR 1.0 spatial upscaling (EASU + RCAS).
    FSR1,
    /// Full temporal upscaling with accumulation and motion reprojection.
    TemporalUpscale,
}

/// Settings for the upscaling system.
#[derive(Debug, Clone)]
pub struct UpscaleSettings {
    /// Active upscaling mode.
    pub mode: UpscaleMode,
    /// Quality preset.
    pub quality_mode: QualityMode,
    /// Sharpness for RCAS (0.0 = no sharpening, 1.0 = maximum).
    pub sharpness: f32,
    /// Whether to apply RCAS after temporal upscale.
    pub apply_rcas: bool,
    /// Number of Halton samples per cycle.
    pub jitter_sample_count: u32,
    /// Temporal accumulation blend factor (lower = more temporal stability).
    pub temporal_blend_factor: f32,
    /// Neighbourhood clamping aggressiveness (1.0 = tight, 2.0 = loose).
    pub clamp_aggressiveness: f32,
    /// Depth threshold for disocclusion detection.
    pub disocclusion_depth_threshold: f32,
    /// Velocity weighting for temporal blend.
    pub velocity_weight: f32,
}

impl Default for UpscaleSettings {
    fn default() -> Self {
        Self {
            mode: UpscaleMode::TemporalUpscale,
            quality_mode: QualityMode::Quality,
            sharpness: 0.5,
            apply_rcas: true,
            jitter_sample_count: 16,
            temporal_blend_factor: 0.05,
            clamp_aggressiveness: 1.25,
            disocclusion_depth_threshold: 0.01,
            velocity_weight: 10.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Halton sequence for jitter
// ---------------------------------------------------------------------------

/// Generate the n-th element of a Halton sequence with given base.
pub fn halton(index: u32, base: u32) -> f32 {
    let mut result = 0.0f32;
    let mut f = 1.0f32;
    let mut i = index;

    while i > 0 {
        f /= base as f32;
        result += f * (i % base) as f32;
        i /= base;
    }

    result
}

/// Generate a 2D Halton jitter sample (base 2, base 3).
/// Returns jitter in [-0.5, 0.5] range relative to a pixel.
pub fn halton_jitter(frame_index: u32, sample_count: u32) -> Vec2 {
    let idx = (frame_index % sample_count) + 1;
    Vec2::new(halton(idx, 2) - 0.5, halton(idx, 3) - 0.5)
}

/// Apply jitter to a projection matrix.
pub fn jitter_projection_matrix(
    projection: &Mat4,
    jitter: Vec2,
    render_width: u32,
    render_height: u32,
) -> Mat4 {
    let jitter_x = 2.0 * jitter.x / render_width as f32;
    let jitter_y = 2.0 * jitter.y / render_height as f32;

    let jitter_mat = Mat4::from_cols(
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(jitter_x, jitter_y, 0.0, 1.0),
    );

    jitter_mat * *projection
}

// ---------------------------------------------------------------------------
// FSR 1.0: EASU (Edge-Adaptive Spatial Upsampling)
// ---------------------------------------------------------------------------

/// EASU filter parameters computed per-pixel.
#[derive(Debug, Clone, Copy)]
pub struct EASUParams {
    /// Input texture size.
    pub input_size: Vec2,
    /// Output texture size.
    pub output_size: Vec2,
    /// Reciprocal of input size.
    pub input_size_rcp: Vec2,
    /// Reciprocal of output size.
    pub output_size_rcp: Vec2,
}

impl EASUParams {
    pub fn new(input_width: u32, input_height: u32, output_width: u32, output_height: u32) -> Self {
        let input_size = Vec2::new(input_width as f32, input_height as f32);
        let output_size = Vec2::new(output_width as f32, output_height as f32);
        Self {
            input_size,
            output_size,
            input_size_rcp: Vec2::new(1.0 / input_size.x, 1.0 / input_size.y),
            output_size_rcp: Vec2::new(1.0 / output_size.x, 1.0 / output_size.y),
        }
    }
}

/// Compute the Lanczos2 weight for EASU filtering.
fn lanczos2(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        return 1.0;
    }
    if x.abs() >= 2.0 {
        return 0.0;
    }
    let pi_x = PI * x;
    let pi_x_half = PI * x * 0.5;
    (pi_x.sin() * pi_x_half.sin()) / (pi_x * pi_x_half)
}

/// Compute edge direction from a 3x3 neighbourhood of luminances.
/// Returns (direction_x, direction_y, edge_strength).
fn detect_edge(luma: &[[f32; 3]; 3]) -> (f32, f32, f32) {
    // Sobel-like edge detection.
    let gx = -luma[0][0] + luma[0][2]
        - 2.0 * luma[1][0] + 2.0 * luma[1][2]
        - luma[2][0] + luma[2][2];

    let gy = -luma[0][0] - 2.0 * luma[0][1] - luma[0][2]
        + luma[2][0] + 2.0 * luma[2][1] + luma[2][2];

    let strength = (gx * gx + gy * gy).sqrt();
    let len = strength.max(1e-8);

    (gx / len, gy / len, strength)
}

/// Perform EASU on a single pixel (CPU reference implementation).
///
/// This upsamples from `input` to `output` coordinates using edge-directed
/// Lanczos filtering.
pub fn easu_sample(
    input: &[Vec3],
    input_width: u32,
    input_height: u32,
    output_x: u32,
    output_y: u32,
    params: &EASUParams,
) -> Vec3 {
    // Map output pixel to input texture coordinate.
    let src_x = (output_x as f32 + 0.5) * params.input_size.x / params.output_size.x;
    let src_y = (output_y as f32 + 0.5) * params.input_size.y / params.output_size.y;

    let ix = src_x.floor() as i32;
    let iy = src_y.floor() as i32;
    let fx = src_x - src_x.floor();
    let fy = src_y - src_y.floor();

    // Sample 5x5 neighbourhood for luminance edge detection.
    let sample = |x: i32, y: i32| -> Vec3 {
        let cx = x.clamp(0, input_width as i32 - 1) as usize;
        let cy = y.clamp(0, input_height as i32 - 1) as usize;
        input[cy * input_width as usize + cx]
    };

    let luma = |c: Vec3| -> f32 { 0.299 * c.x + 0.587 * c.y + 0.114 * c.z };

    // 3x3 luminance grid centred on the closest input texel.
    let mut luma_grid = [[0.0f32; 3]; 3];
    for dy in 0..3i32 {
        for dx in 0..3i32 {
            luma_grid[dy as usize][dx as usize] = luma(sample(ix - 1 + dx, iy - 1 + dy));
        }
    }

    let (edge_x, edge_y, edge_strength) = detect_edge(&luma_grid);

    // Edge-directed kernel: warp the filter along the detected edge.
    let edge_scale = (edge_strength * 4.0).min(1.0);
    let stretch_x = 1.0 + edge_y.abs() * edge_scale * 0.5;
    let stretch_y = 1.0 + edge_x.abs() * edge_scale * 0.5;

    // 4x4 Lanczos filter with edge-direction warping.
    let mut color = Vec3::ZERO;
    let mut weight_sum = 0.0f32;

    for dy in -1..=2i32 {
        for dx in -1..=2i32 {
            let sx = dx as f32 - fx;
            let sy = dy as f32 - fy;

            // Rotate sample offset by edge direction.
            let rotated_x = sx * stretch_x;
            let rotated_y = sy * stretch_y;

            let dist = (rotated_x * rotated_x + rotated_y * rotated_y).sqrt();
            let w = lanczos2(dist);

            if w > 1e-8 {
                let s = sample(ix + dx, iy + dy);
                color += s * w;
                weight_sum += w;
            }
        }
    }

    if weight_sum > 1e-8 {
        color / weight_sum
    } else {
        sample(ix, iy)
    }
}

// ---------------------------------------------------------------------------
// FSR 1.0: RCAS (Robust Contrast-Adaptive Sharpening)
// ---------------------------------------------------------------------------

/// RCAS sharpening parameters.
#[derive(Debug, Clone, Copy)]
pub struct RCASParams {
    /// Sharpness (0.0 to 1.0). Higher = sharper.
    pub sharpness: f32,
}

impl RCASParams {
    pub fn new(sharpness: f32) -> Self {
        Self { sharpness: sharpness.clamp(0.0, 1.0) }
    }
}

/// Perform RCAS on a single pixel (CPU reference implementation).
///
/// This applies contrast-adaptive sharpening using a cross-shaped kernel
/// that avoids ringing on edges.
pub fn rcas_sample(
    input: &[Vec3],
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    params: &RCASParams,
) -> Vec3 {
    let sample = |px: i32, py: i32| -> Vec3 {
        let cx = px.clamp(0, width as i32 - 1) as usize;
        let cy = py.clamp(0, height as i32 - 1) as usize;
        input[cy * width as usize + cx]
    };

    let luma = |c: Vec3| -> f32 { 0.299 * c.x + 0.587 * c.y + 0.114 * c.z };

    let ix = x as i32;
    let iy = y as i32;

    // Cross-shaped sample pattern.
    let center = sample(ix, iy);
    let north = sample(ix, iy - 1);
    let south = sample(ix, iy + 1);
    let west = sample(ix - 1, iy);
    let east = sample(ix + 1, iy);

    let lc = luma(center);
    let ln = luma(north);
    let ls = luma(south);
    let lw = luma(west);
    let le = luma(east);

    // Compute min and max luma in the neighbourhood.
    let luma_min = lc.min(ln).min(ls).min(lw).min(le);
    let luma_max = lc.max(ln).max(ls).max(lw).max(le);

    // Compute sharpening weight based on local contrast.
    let luma_range = luma_max - luma_min;
    let luma_avg = (ln + ls + lw + le) * 0.25;
    let contrast = luma_range / (luma_max + 1e-5);

    // Adaptive sharpening: reduce sharpening in high-contrast areas to
    // prevent ringing.
    let sharp_scale = params.sharpness * (1.0 - contrast.min(1.0));

    // The sharpening weight is negative to create the unsharp mask effect.
    // w = -1/(2*range + 1) scaled by sharpness.
    let w = -(sharp_scale / (4.0 * luma_range + 1e-5)).clamp(-0.25, 0.0);

    // Apply the 5-tap filter: center + w * (neighbours - 4 * center)
    // Simplified: (1 - 4w) * center + w * (N + S + W + E)
    let result = center * (1.0 - 4.0 * w) + (north + south + west + east) * w;

    // Clamp to prevent overshoot.
    Vec3::new(
        result.x.clamp(luma_min * 0.9, luma_max * 1.1 + 0.001),
        result.y.clamp(luma_min * 0.9, luma_max * 1.1 + 0.001),
        result.z.clamp(luma_min * 0.9, luma_max * 1.1 + 0.001),
    )
}

// ---------------------------------------------------------------------------
// Temporal Upscaling
// ---------------------------------------------------------------------------

/// State for the temporal upscaling system.
#[derive(Debug, Clone)]
pub struct TemporalUpscaleState {
    /// Accumulated colour buffer (output resolution).
    pub history_color: Vec<Vec3>,
    /// Previous frame's depth buffer (render resolution).
    pub prev_depth: Vec<f32>,
    /// Output resolution width.
    pub output_width: u32,
    /// Output resolution height.
    pub output_height: u32,
    /// Render resolution width.
    pub render_width: u32,
    /// Render resolution height.
    pub render_height: u32,
    /// Current jitter offset.
    pub jitter: Vec2,
    /// Previous jitter offset.
    pub prev_jitter: Vec2,
    /// Frame index.
    pub frame_index: u32,
    /// Whether history is valid (false on first frame or after reset).
    pub history_valid: bool,
    /// Settings.
    pub settings: UpscaleSettings,
}

impl TemporalUpscaleState {
    /// Create a new temporal upscale state.
    pub fn new(output_width: u32, output_height: u32, settings: &UpscaleSettings) -> Self {
        let (render_width, render_height) = settings.quality_mode.internal_resolution(output_width, output_height);
        let output_size = (output_width * output_height) as usize;
        let render_size = (render_width * render_height) as usize;

        Self {
            history_color: vec![Vec3::ZERO; output_size],
            prev_depth: vec![1.0; render_size],
            output_width,
            output_height,
            render_width,
            render_height,
            jitter: Vec2::ZERO,
            prev_jitter: Vec2::ZERO,
            frame_index: 0,
            history_valid: false,
            settings: settings.clone(),
        }
    }

    /// Update jitter for the current frame.
    pub fn update_jitter(&mut self) {
        self.prev_jitter = self.jitter;
        self.jitter = halton_jitter(self.frame_index, self.settings.jitter_sample_count);
        self.frame_index += 1;
    }

    /// Reset history (e.g. on camera cut or teleport).
    pub fn reset_history(&mut self) {
        for c in &mut self.history_color {
            *c = Vec3::ZERO;
        }
        self.history_valid = false;
    }

    /// Perform temporal upscaling on a frame (CPU reference implementation).
    ///
    /// # Arguments
    /// - `current_color`: Current frame rendered at render resolution (RGB).
    /// - `current_depth`: Current frame depth at render resolution.
    /// - `motion_vectors`: Per-pixel motion vectors at render resolution (in UV space).
    ///
    /// # Returns
    /// Upscaled colour buffer at output resolution.
    pub fn resolve(
        &mut self,
        current_color: &[Vec3],
        current_depth: &[f32],
        motion_vectors: &[Vec2],
    ) -> Vec<Vec3> {
        let out_w = self.output_width;
        let out_h = self.output_height;
        let in_w = self.render_width;
        let in_h = self.render_height;

        let mut output = vec![Vec3::ZERO; (out_w * out_h) as usize];

        let scale_x = in_w as f32 / out_w as f32;
        let scale_y = in_h as f32 / out_h as f32;

        for out_y in 0..out_h {
            for out_x in 0..out_w {
                let out_idx = (out_y * out_w + out_x) as usize;

                // Map output pixel to input pixel coordinate.
                let in_x_f = (out_x as f32 + 0.5) * scale_x;
                let in_y_f = (out_y as f32 + 0.5) * scale_y;

                let in_x = (in_x_f.floor() as u32).min(in_w - 1);
                let in_y = (in_y_f.floor() as u32).min(in_h - 1);
                let in_idx = (in_y * in_w + in_x) as usize;

                // Current frame sample (with jitter compensation).
                let current = bilinear_sample(
                    current_color, in_w, in_h,
                    in_x_f - self.jitter.x,
                    in_y_f - self.jitter.y,
                );

                let depth = current_depth[in_idx];

                if !self.history_valid {
                    // First frame: just use the current sample.
                    output[out_idx] = current;
                    continue;
                }

                // Motion-vector reprojection.
                let mv = if in_idx < motion_vectors.len() {
                    motion_vectors[in_idx]
                } else {
                    Vec2::ZERO
                };

                let prev_x_f = out_x as f32 + 0.5 - mv.x * out_w as f32;
                let prev_y_f = out_y as f32 + 0.5 - mv.y * out_h as f32;

                // Sample history at reprojected position.
                let history = bilinear_sample_output(
                    &self.history_color, out_w, out_h, prev_x_f, prev_y_f,
                );

                // Neighbourhood clamping to prevent ghosting.
                let (min_color, max_color) = compute_neighbourhood_bounds(
                    current_color, in_w, in_h, in_x, in_y,
                    self.settings.clamp_aggressiveness,
                );

                let clamped_history = Vec3::new(
                    history.x.clamp(min_color.x, max_color.x),
                    history.y.clamp(min_color.y, max_color.y),
                    history.z.clamp(min_color.z, max_color.z),
                );

                // Disocclusion detection via depth.
                let prev_in_x = (prev_x_f * scale_x) as u32;
                let prev_in_y = (prev_y_f * scale_y) as u32;
                let prev_in_x = prev_in_x.min(in_w.saturating_sub(1));
                let prev_in_y = prev_in_y.min(in_h.saturating_sub(1));
                let prev_depth = self.prev_depth[(prev_in_y * in_w + prev_in_x) as usize];

                let depth_diff = (depth - prev_depth).abs();
                let is_disoccluded = depth_diff > self.settings.disocclusion_depth_threshold;

                // Compute temporal blend factor.
                let velocity_mag = mv.length() * self.settings.velocity_weight;
                let velocity_factor = (1.0 - velocity_mag.min(1.0)).max(0.0);

                let mut blend = self.settings.temporal_blend_factor;
                blend = blend + (1.0 - velocity_factor) * 0.3;

                if is_disoccluded {
                    blend = 1.0; // Use current frame only.
                }

                blend = blend.clamp(0.0, 1.0);

                // Blend current and history.
                let resolved = current * blend + clamped_history * (1.0 - blend);
                output[out_idx] = resolved;
            }
        }

        // Update history.
        self.history_color = output.clone();
        self.prev_depth = current_depth.to_vec();
        self.history_valid = true;

        output
    }
}

/// Bilinear sample from a colour buffer at render resolution.
fn bilinear_sample(data: &[Vec3], width: u32, height: u32, x: f32, y: f32) -> Vec3 {
    let x = x.clamp(0.0, width as f32 - 1.0);
    let y = y.clamp(0.0, height as f32 - 1.0);

    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let s00 = data[(y0 * width + x0) as usize];
    let s10 = data[(y0 * width + x1) as usize];
    let s01 = data[(y1 * width + x0) as usize];
    let s11 = data[(y1 * width + x1) as usize];

    let top = s00 * (1.0 - fx) + s10 * fx;
    let bottom = s01 * (1.0 - fx) + s11 * fx;

    top * (1.0 - fy) + bottom * fy
}

/// Bilinear sample from a colour buffer at output resolution.
fn bilinear_sample_output(data: &[Vec3], width: u32, height: u32, x: f32, y: f32) -> Vec3 {
    bilinear_sample(data, width, height, x, y)
}

/// Compute min/max colour bounds from a 3x3 neighbourhood (for clamping).
fn compute_neighbourhood_bounds(
    data: &[Vec3],
    width: u32,
    height: u32,
    cx: u32,
    cy: u32,
    aggressiveness: f32,
) -> (Vec3, Vec3) {
    let mut min_c = Vec3::splat(f32::MAX);
    let mut max_c = Vec3::splat(f32::MIN);
    let mut mean = Vec3::ZERO;
    let mut count = 0.0f32;

    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            let px = (cx as i32 + dx).clamp(0, width as i32 - 1) as u32;
            let py = (cy as i32 + dy).clamp(0, height as i32 - 1) as u32;
            let c = data[(py * width + px) as usize];

            min_c = Vec3::new(min_c.x.min(c.x), min_c.y.min(c.y), min_c.z.min(c.z));
            max_c = Vec3::new(max_c.x.max(c.x), max_c.y.max(c.y), max_c.z.max(c.z));
            mean += c;
            count += 1.0;
        }
    }

    mean /= count;

    // Expand bounds based on aggressiveness.
    // Tight clamping (aggressiveness=1) uses raw min/max.
    // Loose clamping (aggressiveness>1) expands toward mean-relative bounds.
    let half_range = (max_c - min_c) * 0.5 * aggressiveness;
    let center = (min_c + max_c) * 0.5;

    let final_min = center - half_range;
    let final_max = center + half_range;

    (final_min, final_max)
}

// ---------------------------------------------------------------------------
// WGSL Compute Shaders
// ---------------------------------------------------------------------------

/// WGSL compute shader for EASU (Edge-Adaptive Spatial Upsampling).
pub const EASU_WGSL: &str = r#"
// EASU - Edge-Adaptive Spatial Upsampling
// Upsamples from render resolution to output resolution using edge-directed
// Lanczos-style filtering.

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var linear_sampler: sampler;

struct EASUUniforms {
    input_size: vec2<f32>,
    output_size: vec2<f32>,
    input_size_rcp: vec2<f32>,
    output_size_rcp: vec2<f32>,
}

@group(0) @binding(3) var<uniform> params: EASUUniforms;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

fn lanczos2_weight(x: f32) -> f32 {
    if (abs(x) < 0.001) { return 1.0; }
    if (abs(x) >= 2.0) { return 0.0; }
    let pi_x = 3.14159265 * x;
    let pi_x_half = 3.14159265 * x * 0.5;
    return (sin(pi_x) * sin(pi_x_half)) / (pi_x * pi_x_half);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u32(params.output_size.x) || gid.y >= u32(params.output_size.y)) {
        return;
    }

    let out_uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) * params.output_size_rcp;
    let src_pos = out_uv * params.input_size;

    let src_center = floor(src_pos);
    let frac_pos = src_pos - src_center;

    // Gather 3x3 luminance for edge detection.
    var luma: array<array<f32, 3>, 3>;
    for (var dy = 0; dy < 3; dy = dy + 1) {
        for (var dx = 0; dx < 3; dx = dx + 1) {
            let uv = (src_center + vec2<f32>(f32(dx) - 1.0, f32(dy) - 1.0) + 0.5) * params.input_size_rcp;
            let s = textureSampleLevel(input_tex, linear_sampler, uv, 0.0).rgb;
            luma[dy][dx] = luminance(s);
        }
    }

    // Sobel edge detection.
    let gx = -luma[0][0] + luma[0][2] - 2.0 * luma[1][0] + 2.0 * luma[1][2] - luma[2][0] + luma[2][2];
    let gy = -luma[0][0] - 2.0 * luma[0][1] - luma[0][2] + luma[2][0] + 2.0 * luma[2][1] + luma[2][2];
    let edge_str = sqrt(gx * gx + gy * gy);
    let edge_scale = min(edge_str * 4.0, 1.0);

    // Edge-directed Lanczos filter (4x4).
    var color = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var dy = -1; dy <= 2; dy = dy + 1) {
        for (var dx = -1; dx <= 2; dx = dx + 1) {
            let offset = vec2<f32>(f32(dx) - frac_pos.x, f32(dy) - frac_pos.y);
            let dist = length(offset);
            let w = lanczos2_weight(dist);

            if (w > 0.001) {
                let uv = (src_center + vec2<f32>(f32(dx), f32(dy)) + 0.5) * params.input_size_rcp;
                let s = textureSampleLevel(input_tex, linear_sampler, uv, 0.0).rgb;
                color = color + s * w;
                total_weight = total_weight + w;
            }
        }
    }

    if (total_weight > 0.001) {
        color = color / total_weight;
    }

    textureStore(output_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(color, 1.0));
}
"#;

/// WGSL compute shader for RCAS (Robust Contrast-Adaptive Sharpening).
pub const RCAS_WGSL: &str = r#"
// RCAS - Robust Contrast-Adaptive Sharpening
// Applies adaptive sharpening using a cross-shaped kernel.

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba16float, write>;

struct RCASUniforms {
    sharpness: f32,
    width: u32,
    height: u32,
    _pad: u32,
}

@group(0) @binding(2) var<uniform> params: RCASUniforms;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let pos = vec2<i32>(i32(gid.x), i32(gid.y));

    let center = textureLoad(input_tex, pos, 0).rgb;
    let north  = textureLoad(input_tex, pos + vec2<i32>(0, -1), 0).rgb;
    let south  = textureLoad(input_tex, pos + vec2<i32>(0, 1), 0).rgb;
    let west   = textureLoad(input_tex, pos + vec2<i32>(-1, 0), 0).rgb;
    let east   = textureLoad(input_tex, pos + vec2<i32>(1, 0), 0).rgb;

    let lc = luminance(center);
    let ln = luminance(north);
    let ls = luminance(south);
    let lw = luminance(west);
    let le = luminance(east);

    let luma_min = min(lc, min(ln, min(ls, min(lw, le))));
    let luma_max = max(lc, max(ln, max(ls, max(lw, le))));
    let luma_range = luma_max - luma_min;

    let contrast = luma_range / (luma_max + 0.00001);
    let sharp_scale = params.sharpness * (1.0 - min(contrast, 1.0));
    let w = clamp(-sharp_scale / (4.0 * luma_range + 0.00001), -0.25, 0.0);

    let result = center * (1.0 - 4.0 * w) + (north + south + west + east) * w;
    let clamped = clamp(result, vec3<f32>(luma_min * 0.9), vec3<f32>(luma_max * 1.1 + 0.001));

    textureStore(output_tex, pos, vec4<f32>(clamped, 1.0));
}
"#;

/// WGSL compute shader for temporal resolve.
pub const TEMPORAL_RESOLVE_WGSL: &str = r#"
// Temporal Upscale Resolve Shader
// Combines current frame with reprojected history using motion vectors.

@group(0) @binding(0) var current_tex: texture_2d<f32>;
@group(0) @binding(1) var history_tex: texture_2d<f32>;
@group(0) @binding(2) var motion_tex: texture_2d<f32>;
@group(0) @binding(3) var depth_tex: texture_2d<f32>;
@group(0) @binding(4) var prev_depth_tex: texture_2d<f32>;
@group(0) @binding(5) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var bilinear_sampler: sampler;

struct TemporalUniforms {
    output_size: vec2<f32>,
    render_size: vec2<f32>,
    jitter: vec2<f32>,
    prev_jitter: vec2<f32>,
    blend_factor: f32,
    clamp_aggressiveness: f32,
    disocclusion_threshold: f32,
    velocity_weight: f32,
}

@group(0) @binding(7) var<uniform> params: TemporalUniforms;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_size = vec2<u32>(u32(params.output_size.x), u32(params.output_size.y));
    if (gid.x >= out_size.x || gid.y >= out_size.y) {
        return;
    }

    let out_uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / params.output_size;
    let in_uv = out_uv;

    // Current frame sample (jitter-compensated).
    let jitter_offset = params.jitter / params.render_size;
    let current = textureSampleLevel(current_tex, bilinear_sampler, in_uv - jitter_offset, 0.0).rgb;

    // Motion vector.
    let mv = textureSampleLevel(motion_tex, bilinear_sampler, in_uv, 0.0).rg;

    // Reprojected history.
    let prev_uv = out_uv - mv;
    let history = textureSampleLevel(history_tex, bilinear_sampler, prev_uv, 0.0).rgb;

    // Neighbourhood clamping (3x3).
    var min_c = vec3<f32>(1e10);
    var max_c = vec3<f32>(-1e10);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let offset = vec2<f32>(f32(dx), f32(dy)) / params.render_size;
            let s = textureSampleLevel(current_tex, bilinear_sampler, in_uv + offset, 0.0).rgb;
            min_c = min(min_c, s);
            max_c = max(max_c, s);
        }
    }

    let half_range = (max_c - min_c) * 0.5 * params.clamp_aggressiveness;
    let center_c = (min_c + max_c) * 0.5;
    let clamp_min = center_c - half_range;
    let clamp_max = center_c + half_range;
    let clamped_history = clamp(history, clamp_min, clamp_max);

    // Disocclusion detection.
    let depth = textureSampleLevel(depth_tex, bilinear_sampler, in_uv, 0.0).r;
    let prev_depth = textureSampleLevel(prev_depth_tex, bilinear_sampler, prev_uv, 0.0).r;
    let is_disoccluded = abs(depth - prev_depth) > params.disocclusion_threshold;

    // Velocity-based blend factor.
    let vel_mag = length(mv) * params.velocity_weight;
    let vel_factor = max(1.0 - vel_mag, 0.0);
    var blend = params.blend_factor + (1.0 - vel_factor) * 0.3;
    if (is_disoccluded) { blend = 1.0; }
    blend = clamp(blend, 0.0, 1.0);

    let resolved = mix(clamped_history, current, blend);
    textureStore(output_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(resolved, 1.0));
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_mode_scale_factors() {
        assert!((QualityMode::Ultra.scale_factor() - 1.3).abs() < 0.01);
        assert!((QualityMode::Quality.scale_factor() - 1.5).abs() < 0.01);
        assert!((QualityMode::Balanced.scale_factor() - 1.7).abs() < 0.01);
        assert!((QualityMode::Performance.scale_factor() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_internal_resolution() {
        let (w, h) = QualityMode::Performance.internal_resolution(1920, 1080);
        assert_eq!(w, 960);
        assert_eq!(h, 540);

        let (w, h) = QualityMode::Quality.internal_resolution(1920, 1080);
        assert_eq!(w, 1280);
        assert_eq!(h, 720);
    }

    #[test]
    fn test_halton_sequence() {
        // Halton base 2 should give: 1/2, 1/4, 3/4, 1/8, ...
        let h1 = halton(1, 2);
        assert!((h1 - 0.5).abs() < 0.001);

        let h2 = halton(2, 2);
        assert!((h2 - 0.25).abs() < 0.001);

        let h3 = halton(3, 2);
        assert!((h3 - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_halton_jitter_range() {
        for i in 0..16 {
            let j = halton_jitter(i, 16);
            assert!(j.x >= -0.5 && j.x <= 0.5);
            assert!(j.y >= -0.5 && j.y <= 0.5);
        }
    }

    #[test]
    fn test_jitter_projection_matrix() {
        let proj = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0);
        let jittered = jitter_projection_matrix(&proj, Vec2::new(0.25, -0.25), 1920, 1080);
        // Should be different from original.
        assert!(jittered != proj);
    }

    #[test]
    fn test_lanczos2() {
        assert!((lanczos2(0.0) - 1.0).abs() < 0.001);
        assert!(lanczos2(2.0).abs() < 0.001);
        assert!(lanczos2(3.0).abs() < 0.001);
        // Should be positive near 0.
        assert!(lanczos2(0.5) > 0.0);
    }

    #[test]
    fn test_easu_sample() {
        let w = 4u32;
        let h = 4u32;
        let input: Vec<Vec3> = (0..16)
            .map(|i| Vec3::splat(i as f32 / 16.0))
            .collect();

        let params = EASUParams::new(w, h, 8, 8);
        let result = easu_sample(&input, w, h, 4, 4, &params);
        // Should produce a reasonable interpolated value.
        assert!(result.x >= 0.0 && result.x <= 1.0);
    }

    #[test]
    fn test_rcas_sample() {
        let w = 4u32;
        let h = 4u32;
        let input: Vec<Vec3> = (0..16)
            .map(|_| Vec3::splat(0.5))
            .collect();

        let params = RCASParams::new(0.5);
        let result = rcas_sample(&input, w, h, 2, 2, &params);
        // Uniform input should produce similar output.
        assert!((result.x - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_temporal_upscale_state() {
        let settings = UpscaleSettings::default();
        let mut state = TemporalUpscaleState::new(1920, 1080, &settings);
        assert_eq!(state.output_width, 1920);
        assert_eq!(state.output_height, 1080);
        assert!(!state.history_valid);

        state.update_jitter();
        assert_eq!(state.frame_index, 1);
    }

    #[test]
    fn test_temporal_resolve_small() {
        let settings = UpscaleSettings {
            quality_mode: QualityMode::Performance,
            ..Default::default()
        };

        let mut state = TemporalUpscaleState::new(8, 8, &settings);
        let render_size = (state.render_width * state.render_height) as usize;

        let color: Vec<Vec3> = (0..render_size)
            .map(|_| Vec3::new(0.5, 0.3, 0.2))
            .collect();
        let depth = vec![0.5f32; render_size];
        let motion = vec![Vec2::ZERO; render_size];

        state.update_jitter();
        let output = state.resolve(&color, &depth, &motion);

        assert_eq!(output.len(), 64); // 8x8
        assert!(state.history_valid);
    }

    #[test]
    fn test_neighbourhood_bounds() {
        let data = vec![
            Vec3::new(0.1, 0.1, 0.1), Vec3::new(0.5, 0.5, 0.5), Vec3::new(0.9, 0.9, 0.9),
            Vec3::new(0.2, 0.2, 0.2), Vec3::new(0.5, 0.5, 0.5), Vec3::new(0.8, 0.8, 0.8),
            Vec3::new(0.3, 0.3, 0.3), Vec3::new(0.5, 0.5, 0.5), Vec3::new(0.7, 0.7, 0.7),
        ];

        let (min_c, max_c) = compute_neighbourhood_bounds(&data, 3, 3, 1, 1, 1.0);
        assert!(min_c.x <= 0.2);
        assert!(max_c.x >= 0.8);
    }
}
