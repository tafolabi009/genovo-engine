// engine/render/src/postprocess/fxaa.rs
//
// Anti-aliasing: FXAA 3.11 (Fast Approximate Anti-Aliasing) and a Temporal
// Anti-Aliasing (TAA) framework.
//
// FXAA is a single-pass, image-based AA algorithm that detects contrast
// edges via luminance and shifts sub-pixel samples to smooth them.
// It's fast and requires no extra geometry passes.
//
// TAA accumulates multiple frames with sub-pixel jitter (Halton sequence)
// to reconstruct a higher-resolution signal over time. History reprojection,
// neighborhood clamping, and velocity rejection prevent ghosting and
// disocclusion artifacts.

use std::any::Any;

use super::{PostProcessEffect, PostProcessInput, PostProcessOutput, TextureId};

// ---------------------------------------------------------------------------
// FXAA Settings
// ---------------------------------------------------------------------------

/// FXAA quality preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FXAAQualityPreset {
    /// 5 edge search steps, fast.
    Low,
    /// 8 edge search steps, balanced.
    Medium,
    /// 12 edge search steps, high quality.
    High,
    /// 16 edge search steps + sub-pixel AA, best quality.
    Ultra,
}

impl FXAAQualityPreset {
    /// Number of edge search steps for this preset.
    pub fn edge_steps(self) -> u32 {
        match self {
            Self::Low => 5,
            Self::Medium => 8,
            Self::High => 12,
            Self::Ultra => 16,
        }
    }

    /// Edge detection threshold (minimum contrast to detect an edge).
    pub fn edge_threshold(self) -> f32 {
        match self {
            Self::Low => 0.250,
            Self::Medium => 0.166,
            Self::High => 0.125,
            Self::Ultra => 0.063,
        }
    }

    /// Minimum edge threshold (avoids processing dark areas).
    pub fn edge_threshold_min(self) -> f32 {
        match self {
            Self::Low => 0.0833,
            Self::Medium => 0.0625,
            Self::High => 0.0312,
            Self::Ultra => 0.0156,
        }
    }

    /// Sub-pixel quality (0 = off, 1 = max smoothing).
    pub fn subpixel_quality(self) -> f32 {
        match self {
            Self::Low => 0.25,
            Self::Medium => 0.50,
            Self::High => 0.75,
            Self::Ultra => 1.00,
        }
    }
}

/// FXAA configuration.
#[derive(Debug, Clone)]
pub struct FXAASettings {
    /// Quality preset.
    pub quality: FXAAQualityPreset,
    /// Whether the effect is enabled.
    pub enabled: bool,
}

impl Default for FXAASettings {
    fn default() -> Self {
        Self {
            quality: FXAAQualityPreset::High,
            enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// TAA Settings
// ---------------------------------------------------------------------------

/// Temporal Anti-Aliasing configuration.
#[derive(Debug, Clone)]
pub struct TAASettings {
    /// Blend factor for history accumulation (0 = no history, 1 = all
    /// history).
    pub blend_factor: f32,
    /// Neighborhood clamping mode.
    pub clamp_mode: TAAClampMode,
    /// Number of Halton sequence samples before repeating.
    pub jitter_sequence_length: u32,
    /// Whether to apply sharpening after TAA (to counter blur).
    pub sharpening: bool,
    /// Sharpening intensity (0..1).
    pub sharpening_intensity: f32,
    /// Velocity rejection threshold (pixels). Reject history when motion
    /// exceeds this.
    pub velocity_rejection_threshold: f32,
    /// Whether the TAA pass is enabled.
    pub enabled: bool,
}

/// Neighborhood clamping modes for TAA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TAAClampMode {
    /// Axis-aligned bounding box of the 3x3 neighborhood.
    AABB,
    /// Variance-based clipping (tighter, fewer artifacts).
    Variance,
}

impl Default for TAASettings {
    fn default() -> Self {
        Self {
            blend_factor: 0.9,
            clamp_mode: TAAClampMode::Variance,
            jitter_sequence_length: 16,
            sharpening: true,
            sharpening_intensity: 0.5,
            velocity_rejection_threshold: 10.0,
            enabled: false, // Off by default since FXAA is the primary.
        }
    }
}

// ---------------------------------------------------------------------------
// Halton sequence for TAA jitter
// ---------------------------------------------------------------------------

/// Generate a Halton sequence value for the given index and base.
pub fn halton(index: u32, base: u32) -> f32 {
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    let mut i = index;
    let b = base as f32;

    while i > 0 {
        f /= b;
        r += f * (i % base) as f32;
        i /= base;
    }

    r
}

/// Generate a 2D jitter offset for the given frame index.
///
/// Returns `(x, y)` in [-0.5, 0.5] pixel units, suitable for offsetting
/// the projection matrix.
pub fn halton_jitter(frame_index: u64, sequence_length: u32) -> [f32; 2] {
    let idx = (frame_index % sequence_length as u64) as u32 + 1;
    [halton(idx, 2) - 0.5, halton(idx, 3) - 0.5]
}

/// Apply a jitter offset to a projection matrix.
///
/// Modifies the projection matrix so the scene is rendered with a sub-pixel
/// offset. The offset is in pixel units and is converted to NDC.
pub fn apply_jitter_to_projection(
    projection: &mut [[f32; 4]; 4],
    jitter_pixels: [f32; 2],
    viewport_width: u32,
    viewport_height: u32,
) {
    let jitter_ndc_x = jitter_pixels[0] * 2.0 / viewport_width as f32;
    let jitter_ndc_y = jitter_pixels[1] * 2.0 / viewport_height as f32;

    // Offset the projection matrix (columns 2,0 and 2,1 for row-major;
    // we use the standard convention where [row][col]).
    projection[0][2] += jitter_ndc_x;
    projection[1][2] += jitter_ndc_y;
}

// ---------------------------------------------------------------------------
// FXAA luminance helpers (CPU reference)
// ---------------------------------------------------------------------------

/// Compute the perceived luminance (for edge detection).
pub fn fxaa_luminance(r: f32, g: f32, b: f32) -> f32 {
    // Green-weighted: human eyes are most sensitive to green.
    r * 0.299 + g * 0.587 + b * 0.114
}

/// Determine if a pixel lies on a contrast edge.
///
/// Samples the luminance of the pixel and its 4 immediate neighbors.
/// Returns `(is_edge, contrast)`.
pub fn detect_edge(
    center_luma: f32,
    north_luma: f32,
    south_luma: f32,
    east_luma: f32,
    west_luma: f32,
    threshold: f32,
    threshold_min: f32,
) -> (bool, f32) {
    let range_max = center_luma
        .max(north_luma)
        .max(south_luma)
        .max(east_luma)
        .max(west_luma);
    let range_min = center_luma
        .min(north_luma)
        .min(south_luma)
        .min(east_luma)
        .min(west_luma);
    let contrast = range_max - range_min;

    let is_edge = contrast >= threshold.max(threshold_min * range_max);
    (is_edge, contrast)
}

/// Determine if the edge is horizontal or vertical.
///
/// Uses the 3x3 neighborhood luminance to detect dominant edge direction.
pub fn edge_direction(
    n: f32,
    s: f32,
    e: f32,
    w: f32,
    ne: f32,
    nw: f32,
    se: f32,
    sw: f32,
) -> bool {
    // is_horizontal
    let horizontal = (n + s - 2.0 * 0.0).abs()   // approximation
        + (ne + se - 2.0 * e).abs()
        + (nw + sw - 2.0 * w).abs();
    let vertical = (e + w - 2.0 * 0.0).abs()
        + (ne + nw - 2.0 * n).abs()
        + (se + sw - 2.0 * s).abs();

    horizontal >= vertical
}

/// Compute the sub-pixel blend factor based on the low-pass filter
/// of the 3x3 neighborhood.
pub fn subpixel_blend(
    center: f32,
    n: f32,
    s: f32,
    e: f32,
    w: f32,
    ne: f32,
    nw: f32,
    se: f32,
    sw: f32,
    contrast: f32,
    quality: f32,
) -> f32 {
    // Low-pass filter of the 3x3 block.
    let average = (n + s + e + w) * 2.0 + (ne + nw + se + sw);
    let average = average / 12.0;
    let sub_pixel_offset = ((average - center).abs() / contrast).clamp(0.0, 1.0);
    // Smooth the blend factor.
    let smoothed = sub_pixel_offset * sub_pixel_offset * (3.0 - 2.0 * sub_pixel_offset);
    smoothed * smoothed * quality
}

// ---------------------------------------------------------------------------
// TAA helpers (CPU reference)
// ---------------------------------------------------------------------------

/// AABB neighborhood clamping.
///
/// Clamps the history color to the axis-aligned bounding box of the 3x3
/// neighborhood in the current frame.
pub fn clamp_aabb(history: [f32; 3], aabb_min: [f32; 3], aabb_max: [f32; 3]) -> [f32; 3] {
    [
        history[0].clamp(aabb_min[0], aabb_max[0]),
        history[1].clamp(aabb_min[1], aabb_max[1]),
        history[2].clamp(aabb_min[2], aabb_max[2]),
    ]
}

/// Variance-based neighborhood clipping.
///
/// Tighter than AABB: clips the history towards the mean of the
/// neighborhood, constrained by `gamma` standard deviations.
pub fn clip_variance(
    history: [f32; 3],
    mean: [f32; 3],
    stddev: [f32; 3],
    gamma: f32,
) -> [f32; 3] {
    let clip_min = [
        mean[0] - stddev[0] * gamma,
        mean[1] - stddev[1] * gamma,
        mean[2] - stddev[2] * gamma,
    ];
    let clip_max = [
        mean[0] + stddev[0] * gamma,
        mean[1] + stddev[1] * gamma,
        mean[2] + stddev[2] * gamma,
    ];

    // Clip towards the mean along the history-to-mean direction.
    let mut result = history;
    for i in 0..3 {
        result[i] = result[i].clamp(clip_min[i], clip_max[i]);
    }
    result
}

/// Compute mean and standard deviation of a set of color samples.
pub fn compute_neighborhood_stats(
    samples: &[[f32; 3]],
) -> ([f32; 3], [f32; 3]) {
    let n = samples.len() as f32;
    let mut mean = [0.0f32; 3];
    let mut mean_sq = [0.0f32; 3];

    for s in samples {
        for i in 0..3 {
            mean[i] += s[i];
            mean_sq[i] += s[i] * s[i];
        }
    }

    for i in 0..3 {
        mean[i] /= n;
        mean_sq[i] /= n;
    }

    let stddev = [
        (mean_sq[0] - mean[0] * mean[0]).max(0.0).sqrt(),
        (mean_sq[1] - mean[1] * mean[1]).max(0.0).sqrt(),
        (mean_sq[2] - mean[2] * mean[2]).max(0.0).sqrt(),
    ];

    (mean, stddev)
}

// ---------------------------------------------------------------------------
// FXAAEffect
// ---------------------------------------------------------------------------

/// FXAA anti-aliasing post-process effect.
pub struct FXAAEffect {
    pub fxaa_settings: FXAASettings,
    pub taa_settings: TAASettings,
    /// TAA history buffer texture.
    history_texture: TextureId,
    /// Resolved TAA output texture.
    resolved_texture: TextureId,
}

impl FXAAEffect {
    pub fn new(fxaa_settings: FXAASettings) -> Self {
        Self {
            fxaa_settings,
            taa_settings: TAASettings::default(),
            history_texture: TextureId(800),
            resolved_texture: TextureId(801),
        }
    }

    /// Create with both FXAA and TAA configured.
    pub fn with_taa(fxaa_settings: FXAASettings, taa_settings: TAASettings) -> Self {
        Self {
            fxaa_settings,
            taa_settings,
            history_texture: TextureId(800),
            resolved_texture: TextureId(801),
        }
    }

    /// Execute the FXAA pass.
    fn execute_fxaa(&self, _input: &PostProcessInput, _output: &mut PostProcessOutput) {
        // Dispatch FXAA compute shader:
        // 1. For each pixel, compute luminance.
        // 2. Sample 4 neighbors, detect edge.
        // 3. If on an edge, determine edge direction (horizontal/vertical).
        // 4. Walk along the edge in both directions to find endpoints.
        // 5. Compute the UV offset (perpendicular to edge) and sample.
        // 6. Apply sub-pixel shift.
    }

    /// Execute the TAA resolve pass.
    fn execute_taa(&self, _input: &PostProcessInput, _output: &mut PostProcessOutput) {
        if !self.taa_settings.enabled {
            return;
        }
        // 1. Read velocity at this pixel.
        // 2. Reproject to previous frame UV.
        // 3. Sample history at reprojected UV.
        // 4. Sample 3x3 neighborhood of current frame.
        // 5. Clamp/clip history to neighborhood (AABB or variance).
        // 6. Blend current and clamped history.
        // 7. Optional: apply sharpening.
    }
}

impl PostProcessEffect for FXAAEffect {
    fn name(&self) -> &str {
        "FXAA"
    }

    fn execute(&self, input: &PostProcessInput, output: &mut PostProcessOutput) {
        if !self.fxaa_settings.enabled && !self.taa_settings.enabled {
            return;
        }

        if self.taa_settings.enabled {
            self.execute_taa(input, output);
        }

        if self.fxaa_settings.enabled {
            self.execute_fxaa(input, output);
        }
    }

    fn is_enabled(&self) -> bool {
        self.fxaa_settings.enabled || self.taa_settings.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.fxaa_settings.enabled = enabled;
    }

    fn priority(&self) -> u32 {
        800
    }

    fn on_resize(&mut self, _width: u32, _height: u32) {
        // Reallocate history buffer at new resolution.
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// WGSL Shaders
// ---------------------------------------------------------------------------

/// FXAA 3.11 compute shader.
pub const FXAA_WGSL: &str = r#"
// FXAA 3.11 — Fast Approximate Anti-Aliasing (compute shader port)
// Based on the original algorithm by Timothy Lottes (NVIDIA).

struct FXAAParams {
    inv_width:        f32,
    inv_height:       f32,
    edge_threshold:   f32,
    edge_threshold_min: f32,
    subpixel_quality: f32,
    edge_steps:       u32,
    _pad0:            f32,
    _pad1:            f32,
};

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var<uniform> params: FXAAParams;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

// Edge search step offsets (up to 16 steps)
const STEP_OFFSETS: array<f32, 16> = array<f32, 16>(
    1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
    5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5
);

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_texture);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let texel = vec2<f32>(params.inv_width, params.inv_height);

    // Sample center and neighbors
    let color_c = textureSampleLevel(src_texture, tex_sampler, uv, 0.0).rgb;
    let luma_c = luma(color_c);

    let luma_n = luma(textureSampleLevel(src_texture, tex_sampler, uv + vec2<f32>(0.0, -1.0) * texel, 0.0).rgb);
    let luma_s = luma(textureSampleLevel(src_texture, tex_sampler, uv + vec2<f32>(0.0,  1.0) * texel, 0.0).rgb);
    let luma_e = luma(textureSampleLevel(src_texture, tex_sampler, uv + vec2<f32>( 1.0, 0.0) * texel, 0.0).rgb);
    let luma_w = luma(textureSampleLevel(src_texture, tex_sampler, uv + vec2<f32>(-1.0, 0.0) * texel, 0.0).rgb);

    let luma_max = max(luma_c, max(max(luma_n, luma_s), max(luma_e, luma_w)));
    let luma_min = min(luma_c, min(min(luma_n, luma_s), min(luma_e, luma_w)));
    let contrast = luma_max - luma_min;

    // Edge detection threshold
    if contrast < max(params.edge_threshold, params.edge_threshold_min * luma_max) {
        textureStore(dst_texture, gid.xy, vec4<f32>(color_c, 1.0));
        return;
    }

    // Diagonal neighbors for edge direction & sub-pixel shift
    let luma_ne = luma(textureSampleLevel(src_texture, tex_sampler, uv + vec2<f32>( 1.0, -1.0) * texel, 0.0).rgb);
    let luma_nw = luma(textureSampleLevel(src_texture, tex_sampler, uv + vec2<f32>(-1.0, -1.0) * texel, 0.0).rgb);
    let luma_se = luma(textureSampleLevel(src_texture, tex_sampler, uv + vec2<f32>( 1.0,  1.0) * texel, 0.0).rgb);
    let luma_sw = luma(textureSampleLevel(src_texture, tex_sampler, uv + vec2<f32>(-1.0,  1.0) * texel, 0.0).rgb);

    // Edge direction: horizontal vs. vertical
    let edge_h = abs(luma_n + luma_s - 2.0 * luma_c) * 2.0 +
                 abs(luma_ne + luma_se - 2.0 * luma_e) +
                 abs(luma_nw + luma_sw - 2.0 * luma_w);
    let edge_v = abs(luma_e + luma_w - 2.0 * luma_c) * 2.0 +
                 abs(luma_ne + luma_nw - 2.0 * luma_n) +
                 abs(luma_se + luma_sw - 2.0 * luma_s);
    let is_horizontal = edge_h >= edge_v;

    // Choose edge normal direction
    var step_dir: vec2<f32>;
    var luma_pos: f32;
    var luma_neg: f32;

    if is_horizontal {
        step_dir = vec2<f32>(0.0, texel.y);
        luma_pos = luma_s;
        luma_neg = luma_n;
    } else {
        step_dir = vec2<f32>(texel.x, 0.0);
        luma_pos = luma_e;
        luma_neg = luma_w;
    }

    let grad_pos = abs(luma_pos - luma_c);
    let grad_neg = abs(luma_neg - luma_c);
    var step_sign: f32;
    var luma_edge: f32;

    if grad_pos >= grad_neg {
        step_sign = 1.0;
        luma_edge = (luma_c + luma_pos) * 0.5;
    } else {
        step_sign = -1.0;
        luma_edge = (luma_c + luma_neg) * 0.5;
    }

    let gradient = max(grad_pos, grad_neg);
    let half_step = step_dir * step_sign * 0.5;

    // Start at the edge center
    var edge_uv = uv + half_step;

    // Edge search direction (along the edge)
    var search_dir: vec2<f32>;
    if is_horizontal {
        search_dir = vec2<f32>(texel.x, 0.0);
    } else {
        search_dir = vec2<f32>(0.0, texel.y);
    }

    // Walk along the edge in both directions
    var luma_end_pos: f32 = 0.0;
    var luma_end_neg: f32 = 0.0;
    var found_pos = false;
    var found_neg = false;
    var dist_pos: f32 = 0.0;
    var dist_neg: f32 = 0.0;

    for (var i = 0u; i < params.edge_steps; i++) {
        let offset = STEP_OFFSETS[i];

        if !found_pos {
            let sample_pos = edge_uv + search_dir * offset;
            luma_end_pos = luma(textureSampleLevel(src_texture, tex_sampler, sample_pos, 0.0).rgb) - luma_edge;
            found_pos = abs(luma_end_pos) >= gradient * 0.25;
            dist_pos = offset;
        }

        if !found_neg {
            let sample_neg = edge_uv - search_dir * offset;
            luma_end_neg = luma(textureSampleLevel(src_texture, tex_sampler, sample_neg, 0.0).rgb) - luma_edge;
            found_neg = abs(luma_end_neg) >= gradient * 0.25;
            dist_neg = offset;
        }

        if found_pos && found_neg {
            break;
        }
    }

    // Choose the closer end
    let use_pos = dist_pos <= dist_neg;
    let pixel_offset: f32;

    if use_pos {
        if (luma_c - luma_edge < 0.0) == (luma_end_pos < 0.0) {
            pixel_offset = 0.0;  // Wrong side of the edge
        } else {
            pixel_offset = 0.5 - dist_pos / (dist_pos + dist_neg);
        }
    } else {
        if (luma_c - luma_edge < 0.0) == (luma_end_neg < 0.0) {
            pixel_offset = 0.0;
        } else {
            pixel_offset = 0.5 - dist_neg / (dist_pos + dist_neg);
        }
    }

    // Sub-pixel anti-aliasing
    let filter = 2.0 * (luma_n + luma_s + luma_e + luma_w) + (luma_ne + luma_nw + luma_se + luma_sw);
    let filter_avg = filter / 12.0;
    let sub_pixel_offset = clamp(abs(filter_avg - luma_c) / contrast, 0.0, 1.0);
    let sub_pixel = smoothstep(0.0, 1.0, sub_pixel_offset);
    let sub_pixel_shift = sub_pixel * sub_pixel * params.subpixel_quality;

    let final_offset = max(pixel_offset, sub_pixel_shift);

    // Apply the offset perpendicular to the edge
    let final_uv = uv + step_dir * step_sign * final_offset;
    let result = textureSampleLevel(src_texture, tex_sampler, final_uv, 0.0).rgb;

    textureStore(dst_texture, gid.xy, vec4<f32>(result, 1.0));
}
"#;

/// TAA resolve compute shader.
pub const TAA_RESOLVE_WGSL: &str = r#"
// Temporal Anti-Aliasing — resolve compute shader

struct TAAParams {
    blend_factor:     f32,
    vel_reject_thresh: f32,
    clamp_mode:       u32,      // 0 = AABB, 1 = Variance
    sharpen:          u32,      // 0 = off, 1 = on
    sharpen_intensity: f32,
    inv_width:        f32,
    inv_height:       f32,
    _pad:             f32,
};

@group(0) @binding(0) var current_texture:  texture_2d<f32>;
@group(0) @binding(1) var history_texture:  texture_2d<f32>;
@group(0) @binding(2) var velocity_texture: texture_2d<f32>;
@group(0) @binding(3) var depth_texture:    texture_2d<f32>;
@group(0) @binding(4) var dst_texture:      texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var tex_sampler:      sampler;
@group(0) @binding(6) var<uniform> params:  TAAParams;

fn ycocg_encode(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>( 0.25, 0.50,  0.25)),
        dot(c, vec3<f32>( 0.50, 0.00, -0.50)),
        dot(c, vec3<f32>(-0.25, 0.50, -0.25)),
    );
}

fn ycocg_decode(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        c.x + c.y - c.z,
        c.x + c.z,
        c.x - c.y - c.z,
    );
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_texture);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let texel = vec2<f32>(params.inv_width, params.inv_height);

    // Current frame color
    let current = textureSampleLevel(current_texture, tex_sampler, uv, 0.0).rgb;

    // Reproject using velocity
    let velocity = textureSampleLevel(velocity_texture, tex_sampler, uv, 0.0).xy;
    let prev_uv = uv - velocity;

    // Velocity rejection: if object is moving too fast, reduce history weight
    let vel_mag = length(velocity * vec2<f32>(dims));
    let vel_reject = clamp(1.0 - vel_mag / params.vel_reject_thresh, 0.0, 1.0);

    // Out-of-screen rejection
    if any(prev_uv < vec2<f32>(0.0)) || any(prev_uv > vec2<f32>(1.0)) {
        textureStore(dst_texture, gid.xy, vec4<f32>(current, 1.0));
        return;
    }

    var history = textureSampleLevel(history_texture, tex_sampler, prev_uv, 0.0).rgb;

    // Gather 3x3 neighborhood for clamping
    var samples: array<vec3<f32>, 9>;
    var idx = 0u;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let offset = vec2<f32>(f32(dx), f32(dy)) * texel;
            samples[idx] = textureSampleLevel(current_texture, tex_sampler, uv + offset, 0.0).rgb;
            idx++;
        }
    }

    if params.clamp_mode == 0u {
        // AABB clamping
        var aabb_min = samples[0];
        var aabb_max = samples[0];
        for (var i = 1u; i < 9u; i++) {
            aabb_min = min(aabb_min, samples[i]);
            aabb_max = max(aabb_max, samples[i]);
        }
        history = clamp(history, aabb_min, aabb_max);
    } else {
        // Variance clipping (YCoCg space for better results)
        var mean = vec3<f32>(0.0);
        var mean_sq = vec3<f32>(0.0);
        for (var i = 0u; i < 9u; i++) {
            let ycocg = ycocg_encode(samples[i]);
            mean += ycocg;
            mean_sq += ycocg * ycocg;
        }
        mean /= 9.0;
        mean_sq /= 9.0;
        let stddev = sqrt(max(mean_sq - mean * mean, vec3<f32>(0.0)));

        let gamma = 1.25;
        let history_ycocg = ycocg_encode(history);
        let clipped = clamp(history_ycocg, mean - stddev * gamma, mean + stddev * gamma);
        history = ycocg_decode(clipped);
    }

    // Blend
    let blend = params.blend_factor * vel_reject;
    var result = mix(current, history, blend);

    // Optional sharpening (unsharp mask)
    if params.sharpen != 0u {
        let blur = (samples[1] + samples[3] + samples[5] + samples[7]) * 0.25;
        let sharp = result + (result - blur) * params.sharpen_intensity;
        result = max(sharp, vec3<f32>(0.0));
    }

    textureStore(dst_texture, gid.xy, vec4<f32>(result, 1.0));
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_halton_base2() {
        let h1 = halton(1, 2);
        assert!((h1 - 0.5).abs() < 1e-5);

        let h2 = halton(2, 2);
        assert!((h2 - 0.25).abs() < 1e-5);

        let h3 = halton(3, 2);
        assert!((h3 - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_halton_base3() {
        let h1 = halton(1, 3);
        assert!((h1 - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_halton_jitter_range() {
        for i in 0..16 {
            let jitter = halton_jitter(i, 16);
            assert!(jitter[0] >= -0.5 && jitter[0] <= 0.5);
            assert!(jitter[1] >= -0.5 && jitter[1] <= 0.5);
        }
    }

    #[test]
    fn test_fxaa_luminance() {
        // Pure white -> luminance = 1.0.
        let l = fxaa_luminance(1.0, 1.0, 1.0);
        assert!((l - 1.0).abs() < 1e-4);

        // Pure green should contribute most.
        let green = fxaa_luminance(0.0, 1.0, 0.0);
        let red = fxaa_luminance(1.0, 0.0, 0.0);
        assert!(green > red);
    }

    #[test]
    fn test_edge_detection() {
        // High contrast -> edge.
        let (is_edge, _) = detect_edge(0.9, 0.1, 0.1, 0.9, 0.1, 0.125, 0.0312);
        assert!(is_edge);

        // Low contrast -> no edge.
        let (is_edge, _) = detect_edge(0.5, 0.51, 0.49, 0.50, 0.50, 0.125, 0.0312);
        assert!(!is_edge);
    }

    #[test]
    fn test_aabb_clamping() {
        let history = [2.0, 0.5, 0.5];
        let clamped = clamp_aabb(history, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!((clamped[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_variance_clipping() {
        let history = [5.0, 5.0, 5.0];
        let mean = [0.5, 0.5, 0.5];
        let stddev = [0.1, 0.1, 0.1];

        let clipped = clip_variance(history, mean, stddev, 1.0);
        // Should be clamped to mean + stddev.
        assert!((clipped[0] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_neighborhood_stats() {
        let samples = vec![[1.0, 2.0, 3.0]; 9];
        let (mean, stddev) = compute_neighborhood_stats(&samples);

        assert!((mean[0] - 1.0).abs() < 1e-5);
        assert!((stddev[0]).abs() < 1e-5); // All same -> zero stddev.
    }

    #[test]
    fn test_quality_presets() {
        assert!(FXAAQualityPreset::Ultra.edge_steps() > FXAAQualityPreset::Low.edge_steps());
        assert!(
            FXAAQualityPreset::Ultra.edge_threshold()
                < FXAAQualityPreset::Low.edge_threshold()
        );
    }

    #[test]
    fn test_fxaa_effect_interface() {
        let effect = FXAAEffect::new(FXAASettings::default());
        assert_eq!(effect.name(), "FXAA");
        assert!(effect.is_enabled());
        assert_eq!(effect.priority(), 800);
    }
}
