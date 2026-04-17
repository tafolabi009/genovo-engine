// engine/render/src/post_process_runner.rs
//
// Post-processing execution pipeline for the Genovo engine.
//
// Executes a chain of screen-space post-processing effects after the main
// scene has been rendered. Manages ping-pong render targets, effect ordering,
// and the final blit to the swapchain.
//
// Built-in effects:
// - Bloom (dual-filter down/up sampling, threshold, soft knee)
// - Tonemapping (ACES, Reinhard, Uncharted2, Neutral, AgX)
// - FXAA (Fast Approximate Anti-Aliasing)
// - Vignette (circular and elliptical)
// - Chromatic aberration
// - Film grain
// - Color grading (LUT-based)
//
// Each effect has a WGSL shader and can be individually enabled, disabled,
// or weighted. Effects are executed in a configurable order with automatic
// ping-pong target management.

use glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of post-process effects in the chain.
pub const MAX_POST_PROCESS_EFFECTS: usize = 32;

/// Maximum number of bloom mip levels.
pub const MAX_BLOOM_MIP_LEVELS: usize = 8;

/// Default bloom threshold.
pub const DEFAULT_BLOOM_THRESHOLD: f32 = 1.0;

/// Default bloom intensity.
pub const DEFAULT_BLOOM_INTENSITY: f32 = 0.5;

// ---------------------------------------------------------------------------
// WGSL Shader: Bloom Threshold Extract
// ---------------------------------------------------------------------------

/// WGSL shader for bloom threshold extraction.
/// Extracts pixels above the brightness threshold with a soft knee transition.
pub const BLOOM_THRESHOLD_WGSL: &str = r#"
// Bloom threshold extract shader.
// Outputs only pixels above the brightness threshold with soft knee.

struct BloomThresholdParams {
    threshold: f32,
    soft_knee: f32,
    intensity: f32,
    _pad: f32,
};

@group(0) @binding(0)
var<uniform> params: BloomThresholdParams;

@group(0) @binding(1)
var input_texture: texture_2d<f32>;

@group(0) @binding(2)
var input_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(input_texture, input_sampler, input.uv).rgb;
    let lum = luminance(color);

    // Soft knee thresholding (Brian Karis style).
    let knee = params.threshold * params.soft_knee;
    let soft = lum - params.threshold + knee;
    let soft_clamped = clamp(soft, 0.0, 2.0 * knee);
    let contribution = select(0.0, soft_clamped * soft_clamped / (4.0 * knee + 0.00001), knee > 0.0);
    let weight = max(lum - params.threshold, contribution) / max(lum, 0.00001);

    return vec4<f32>(color * weight * params.intensity, 1.0);
}
"#;

// ---------------------------------------------------------------------------
// WGSL Shader: Bloom Downsample
// ---------------------------------------------------------------------------

/// WGSL shader for bloom downsampling using a 13-tap filter.
pub const BLOOM_DOWNSAMPLE_WGSL: &str = r#"
// Bloom downsample with 13-tap tent filter (Jimenez).
// Reduces fireflies by using a weighted box filter.

struct DownsampleParams {
    texel_size: vec2<f32>,
    mip_level: f32,
    _pad: f32,
};

@group(0) @binding(0)
var<uniform> params: DownsampleParams;

@group(0) @binding(1)
var input_texture: texture_2d<f32>;

@group(0) @binding(2)
var input_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let ts = params.texel_size;

    // 13-tap downsampling filter for anti-firefly bloom.
    var color = vec3<f32>(0.0);

    // Center sample (weight 0.125).
    color += textureSample(input_texture, input_sampler, uv).rgb * 0.125;

    // 4 corner samples (each weight 0.03125).
    color += textureSample(input_texture, input_sampler, uv + vec2<f32>(-ts.x, -ts.y)).rgb * 0.03125;
    color += textureSample(input_texture, input_sampler, uv + vec2<f32>( ts.x, -ts.y)).rgb * 0.03125;
    color += textureSample(input_texture, input_sampler, uv + vec2<f32>(-ts.x,  ts.y)).rgb * 0.03125;
    color += textureSample(input_texture, input_sampler, uv + vec2<f32>( ts.x,  ts.y)).rgb * 0.03125;

    // 4 edge samples (each weight 0.0625).
    color += textureSample(input_texture, input_sampler, uv + vec2<f32>(-ts.x, 0.0)).rgb * 0.0625;
    color += textureSample(input_texture, input_sampler, uv + vec2<f32>( ts.x, 0.0)).rgb * 0.0625;
    color += textureSample(input_texture, input_sampler, uv + vec2<f32>(0.0, -ts.y)).rgb * 0.0625;
    color += textureSample(input_texture, input_sampler, uv + vec2<f32>(0.0,  ts.y)).rgb * 0.0625;

    // 4 diagonal samples (each weight 0.125).
    color += textureSample(input_texture, input_sampler, uv + vec2<f32>(-ts.x * 2.0, -ts.y * 2.0)).rgb * 0.03125;
    color += textureSample(input_texture, input_sampler, uv + vec2<f32>( ts.x * 2.0, -ts.y * 2.0)).rgb * 0.03125;
    color += textureSample(input_texture, input_sampler, uv + vec2<f32>(-ts.x * 2.0,  ts.y * 2.0)).rgb * 0.03125;
    color += textureSample(input_texture, input_sampler, uv + vec2<f32>( ts.x * 2.0,  ts.y * 2.0)).rgb * 0.03125;

    return vec4<f32>(color, 1.0);
}
"#;

// ---------------------------------------------------------------------------
// WGSL Shader: Bloom Upsample
// ---------------------------------------------------------------------------

/// WGSL shader for bloom upsampling with tent filter.
pub const BLOOM_UPSAMPLE_WGSL: &str = r#"
// Bloom upsample with 9-tap tent filter.
// Additively blends with the higher-resolution mip.

struct UpsampleParams {
    texel_size: vec2<f32>,
    blend_factor: f32,
    _pad: f32,
};

@group(0) @binding(0)
var<uniform> params: UpsampleParams;

@group(0) @binding(1)
var low_res_texture: texture_2d<f32>;

@group(0) @binding(2)
var low_res_sampler: sampler;

@group(0) @binding(3)
var high_res_texture: texture_2d<f32>;

@group(0) @binding(4)
var high_res_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let ts = params.texel_size;

    // 9-tap tent filter for smooth upsampling.
    var upsampled = vec3<f32>(0.0);

    upsampled += textureSample(low_res_texture, low_res_sampler, uv + vec2<f32>(-ts.x, -ts.y)).rgb * 1.0;
    upsampled += textureSample(low_res_texture, low_res_sampler, uv + vec2<f32>( 0.0,  -ts.y)).rgb * 2.0;
    upsampled += textureSample(low_res_texture, low_res_sampler, uv + vec2<f32>( ts.x, -ts.y)).rgb * 1.0;

    upsampled += textureSample(low_res_texture, low_res_sampler, uv + vec2<f32>(-ts.x, 0.0)).rgb * 2.0;
    upsampled += textureSample(low_res_texture, low_res_sampler, uv).rgb * 4.0;
    upsampled += textureSample(low_res_texture, low_res_sampler, uv + vec2<f32>( ts.x, 0.0)).rgb * 2.0;

    upsampled += textureSample(low_res_texture, low_res_sampler, uv + vec2<f32>(-ts.x, ts.y)).rgb * 1.0;
    upsampled += textureSample(low_res_texture, low_res_sampler, uv + vec2<f32>( 0.0,  ts.y)).rgb * 2.0;
    upsampled += textureSample(low_res_texture, low_res_sampler, uv + vec2<f32>( ts.x, ts.y)).rgb * 1.0;

    upsampled = upsampled / 16.0;

    // Blend with the higher-resolution texture.
    let high_res = textureSample(high_res_texture, high_res_sampler, uv).rgb;
    let result = mix(high_res, upsampled, params.blend_factor);

    return vec4<f32>(result, 1.0);
}
"#;

// ---------------------------------------------------------------------------
// WGSL Shader: Tone Mapping
// ---------------------------------------------------------------------------

/// WGSL shader for tone mapping with multiple operator support.
pub const TONEMAP_WGSL: &str = r#"
// Tone mapping and final color grading shader.
// Supports ACES, Reinhard, Uncharted2, Neutral, and AgX operators.

struct TonemapParams {
    exposure: f32,
    gamma: f32,
    operator: u32,       // 0=ACES, 1=Reinhard, 2=Uncharted2, 3=Neutral, 4=AgX
    white_point: f32,
    contrast: f32,
    saturation: f32,
    lift: vec3<f32>,
    _pad1: f32,
    gain: vec3<f32>,
    _pad2: f32,
    vignette_intensity: f32,
    vignette_smoothness: f32,
    film_grain: f32,
    time: f32,
};

@group(0) @binding(0)
var<uniform> params: TonemapParams;

@group(0) @binding(1)
var input_texture: texture_2d<f32>;

@group(0) @binding(2)
var input_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// ACES Filmic Tone Mapping (Stephen Hill fit).
fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Reinhard tone mapping with white point.
fn reinhard_extended(x: vec3<f32>, white: f32) -> vec3<f32> {
    let white2 = white * white;
    return x * (1.0 + x / white2) / (1.0 + x);
}

// Uncharted 2 tone mapping (John Hable).
fn uncharted2_partial(x: vec3<f32>) -> vec3<f32> {
    let a = 0.15;
    let b = 0.50;
    let c = 0.10;
    let d = 0.20;
    let e = 0.02;
    let f = 0.30;
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f;
}

fn uncharted2(x: vec3<f32>) -> vec3<f32> {
    let curr = uncharted2_partial(x * 2.0);
    let white = uncharted2_partial(vec3<f32>(11.2));
    return curr / white;
}

// Neutral tone mapping (Khronos PBR Neutral).
fn neutral_tonemap(x: vec3<f32>) -> vec3<f32> {
    let start_comp = 0.8 - 0.04;
    let desaturation = 0.15;
    let a = -min(min(x.r, x.g), x.b) + start_comp;
    let b_val = clamp(a, 0.0, desaturation) / desaturation;
    let val = mix(x, vec3<f32>(dot(x, vec3<f32>(0.2126, 0.7152, 0.0722))), b_val * b_val);
    return val / (val + 1.0);
}

// AgX tone mapping.
fn agx_default_contrast(x: vec3<f32>) -> vec3<f32> {
    let lw = vec3<f32>(0.2126, 0.7152, 0.0722);
    let luma = dot(x, lw);
    return mix(vec3<f32>(luma), x, vec3<f32>(1.2));
}

fn agx_tonemap(x: vec3<f32>) -> vec3<f32> {
    let mapped = clamp(log2(max(x, vec3<f32>(1e-10))), vec3<f32>(-12.47393), vec3<f32>(4.026069));
    let normalized = (mapped - vec3<f32>(-12.47393)) / (vec3<f32>(4.026069) - vec3<f32>(-12.47393));
    let curved = normalized * normalized * (3.0 - 2.0 * normalized);
    return agx_default_contrast(curved);
}

// Interleaved gradient noise for film grain.
fn ign(pos: vec2<f32>, frame: f32) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(pos + frame * vec2<f32>(47.0, 17.0), magic.xy)));
}

// Vignette.
fn apply_vignette(color: vec3<f32>, uv: vec2<f32>, intensity: f32, smoothness: f32) -> vec3<f32> {
    let center = vec2<f32>(0.5, 0.5);
    let dist = distance(uv, center) * 1.414;
    let vig = smoothstep(1.0, 1.0 - smoothness, dist);
    return color * mix(1.0, vig, intensity);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(input_texture, input_sampler, input.uv).rgb;

    // Apply exposure.
    color = color * params.exposure;

    // Apply lift/gain color grading.
    color = color * params.gain + params.lift;

    // Apply contrast around midpoint.
    let mid = vec3<f32>(0.18);
    color = mid + (color - mid) * params.contrast;

    // Apply saturation.
    let lum = luminance(color);
    color = mix(vec3<f32>(lum), color, params.saturation);

    // Tone map.
    switch (params.operator) {
        case 0u: {
            color = aces_filmic(color);
        }
        case 1u: {
            color = reinhard_extended(color, params.white_point);
        }
        case 2u: {
            color = uncharted2(color);
        }
        case 3u: {
            color = neutral_tonemap(color);
        }
        case 4u: {
            color = agx_tonemap(color);
        }
        default: {
            color = aces_filmic(color);
        }
    }

    // Apply vignette.
    if (params.vignette_intensity > 0.001) {
        color = apply_vignette(color, input.uv, params.vignette_intensity, params.vignette_smoothness);
    }

    // Apply film grain.
    if (params.film_grain > 0.001) {
        let noise = ign(input.position.xy, params.time) - 0.5;
        color = color + noise * params.film_grain * (1.0 - luminance(color));
    }

    // Gamma correction.
    color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / params.gamma));

    return vec4<f32>(color, 1.0);
}
"#;

// ---------------------------------------------------------------------------
// WGSL Shader: FXAA
// ---------------------------------------------------------------------------

/// WGSL shader for Fast Approximate Anti-Aliasing (FXAA 3.11 quality).
pub const FXAA_WGSL: &str = r#"
// FXAA 3.11 quality implementation.
// Based on Timothy Lottes' original FXAA.

struct FxaaParams {
    texel_size: vec2<f32>,
    subpixel_quality: f32,
    edge_threshold: f32,
    edge_threshold_min: f32,
    _pad: vec3<f32>,
};

@group(0) @binding(0)
var<uniform> params: FxaaParams;

@group(0) @binding(1)
var input_texture: texture_2d<f32>;

@group(0) @binding(2)
var input_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

fn rgb_to_luma(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let ts = params.texel_size;

    // Sample center and neighbours.
    let color_center = textureSample(input_texture, input_sampler, uv).rgb;
    let luma_center = rgb_to_luma(color_center);

    let luma_d = rgb_to_luma(textureSample(input_texture, input_sampler, uv + vec2<f32>(0.0, ts.y)).rgb);
    let luma_u = rgb_to_luma(textureSample(input_texture, input_sampler, uv + vec2<f32>(0.0, -ts.y)).rgb);
    let luma_l = rgb_to_luma(textureSample(input_texture, input_sampler, uv + vec2<f32>(-ts.x, 0.0)).rgb);
    let luma_r = rgb_to_luma(textureSample(input_texture, input_sampler, uv + vec2<f32>(ts.x, 0.0)).rgb);

    let luma_min = min(luma_center, min(min(luma_d, luma_u), min(luma_l, luma_r)));
    let luma_max = max(luma_center, max(max(luma_d, luma_u), max(luma_l, luma_r)));
    let luma_range = luma_max - luma_min;

    // Early exit for low-contrast areas.
    if (luma_range < max(params.edge_threshold_min, luma_max * params.edge_threshold)) {
        return vec4<f32>(color_center, 1.0);
    }

    // Corner luma.
    let luma_dl = rgb_to_luma(textureSample(input_texture, input_sampler, uv + vec2<f32>(-ts.x, ts.y)).rgb);
    let luma_dr = rgb_to_luma(textureSample(input_texture, input_sampler, uv + vec2<f32>(ts.x, ts.y)).rgb);
    let luma_ul = rgb_to_luma(textureSample(input_texture, input_sampler, uv + vec2<f32>(-ts.x, -ts.y)).rgb);
    let luma_ur = rgb_to_luma(textureSample(input_texture, input_sampler, uv + vec2<f32>(ts.x, -ts.y)).rgb);

    let luma_du = luma_d + luma_u;
    let luma_lr = luma_l + luma_r;

    let luma_left_corners = luma_dl + luma_ul;
    let luma_down_corners = luma_dl + luma_dr;
    let luma_right_corners = luma_dr + luma_ur;
    let luma_up_corners = luma_ul + luma_ur;

    let edge_h = abs(-2.0 * luma_l + luma_left_corners) + abs(-2.0 * luma_center + luma_du) * 2.0 + abs(-2.0 * luma_r + luma_right_corners);
    let edge_v = abs(-2.0 * luma_u + luma_up_corners) + abs(-2.0 * luma_center + luma_lr) * 2.0 + abs(-2.0 * luma_d + luma_down_corners);

    let is_horizontal = edge_h >= edge_v;

    var step_length = select(ts.x, ts.y, is_horizontal);
    var luma1 = select(luma_l, luma_d, is_horizontal);
    var luma2 = select(luma_r, luma_u, is_horizontal);

    let gradient1 = luma1 - luma_center;
    let gradient2 = luma2 - luma_center;

    let is_1_steepest = abs(gradient1) >= abs(gradient2);

    let gradient_scaled = 0.25 * max(abs(gradient1), abs(gradient2));

    if (!is_1_steepest) {
        step_length = -step_length;
    }

    // Subpixel offset.
    var current_uv = uv;
    if (is_horizontal) {
        current_uv.y = current_uv.y + step_length * 0.5;
    } else {
        current_uv.x = current_uv.x + step_length * 0.5;
    }

    var offset = select(vec2<f32>(0.0, ts.y), vec2<f32>(ts.x, 0.0), is_horizontal);

    var uv1 = current_uv - offset;
    var uv2 = current_uv + offset;

    let luma_end1 = rgb_to_luma(textureSample(input_texture, input_sampler, uv1).rgb) - luma_center;
    let luma_end2 = rgb_to_luma(textureSample(input_texture, input_sampler, uv2).rgb) - luma_center;

    let reached1 = abs(luma_end1) >= gradient_scaled;
    let reached2 = abs(luma_end2) >= gradient_scaled;

    if (!reached1) { uv1 = uv1 - offset; }
    if (!reached2) { uv2 = uv2 + offset; }

    // Walk along the edge until both ends are found or max steps.
    for (var i = 0; i < 12; i = i + 1) {
        let r1 = abs(rgb_to_luma(textureSample(input_texture, input_sampler, uv1).rgb) - luma_center) >= gradient_scaled;
        let r2 = abs(rgb_to_luma(textureSample(input_texture, input_sampler, uv2).rgb) - luma_center) >= gradient_scaled;
        if (r1 && r2) { break; }
        if (!r1) { uv1 = uv1 - offset; }
        if (!r2) { uv2 = uv2 + offset; }
    }

    let dist1 = select(uv.x - uv1.x, uv.y - uv1.y, is_horizontal);
    let dist2 = select(uv2.x - uv.x, uv2.y - uv.y, is_horizontal);

    let is_direction1 = dist1 < dist2;
    let dist_final = min(dist1, dist2);
    let edge_len = dist1 + dist2;

    let pixel_offset = -dist_final / edge_len + 0.5;

    let luma_avg = (1.0 / 12.0) * (2.0 * (luma_du + luma_lr) + luma_left_corners + luma_right_corners);
    let sub_pixel_offset1 = clamp(abs(luma_avg - luma_center) / luma_range, 0.0, 1.0);
    let sub_pixel_offset2 = (-2.0 * sub_pixel_offset1 + 3.0) * sub_pixel_offset1 * sub_pixel_offset1;
    let sub_pixel_offset = sub_pixel_offset2 * sub_pixel_offset2 * params.subpixel_quality;

    let final_offset = max(pixel_offset, sub_pixel_offset);

    var final_uv = uv;
    if (is_horizontal) {
        final_uv.y = final_uv.y + final_offset * step_length;
    } else {
        final_uv.x = final_uv.x + final_offset * step_length;
    }

    let final_color = textureSample(input_texture, input_sampler, final_uv).rgb;
    return vec4<f32>(final_color, 1.0);
}
"#;

// ---------------------------------------------------------------------------
// WGSL Shader: Chromatic Aberration
// ---------------------------------------------------------------------------

/// WGSL shader for chromatic aberration effect.
pub const CHROMATIC_ABERRATION_WGSL: &str = r#"
// Chromatic aberration: shifts R, G, B channels radially from center.

struct ChromaticParams {
    intensity: f32,
    center: vec2<f32>,
    _pad: f32,
};

@group(0) @binding(0)
var<uniform> params: ChromaticParams;

@group(0) @binding(1)
var input_texture: texture_2d<f32>;

@group(0) @binding(2)
var input_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let dir = uv - params.center;
    let dist = length(dir) * params.intensity;

    let r = textureSample(input_texture, input_sampler, uv + dir * dist * 1.0).r;
    let g = textureSample(input_texture, input_sampler, uv).g;
    let b = textureSample(input_texture, input_sampler, uv - dir * dist * 1.0).b;

    return vec4<f32>(r, g, b, 1.0);
}
"#;

// ---------------------------------------------------------------------------
// WGSL Shader: Final Blit
// ---------------------------------------------------------------------------

/// WGSL shader for the final blit to the swapchain.
pub const FINAL_BLIT_WGSL: &str = r#"
// Final blit: copies the processed image to the swapchain surface.

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var input_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(input_texture, input_sampler, input.uv);
}
"#;

// ---------------------------------------------------------------------------
// Effect types
// ---------------------------------------------------------------------------

/// Post-processing effect identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PostProcessEffect {
    Bloom,
    Tonemap,
    Fxaa,
    Vignette,
    ChromaticAberration,
    FilmGrain,
    ColorGrading,
    DepthOfField,
    MotionBlur,
    Sharpen,
    Custom(u32),
}

impl std::fmt::Display for PostProcessEffect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bloom => write!(f, "Bloom"),
            Self::Tonemap => write!(f, "Tonemap"),
            Self::Fxaa => write!(f, "FXAA"),
            Self::Vignette => write!(f, "Vignette"),
            Self::ChromaticAberration => write!(f, "Chromatic Aberration"),
            Self::FilmGrain => write!(f, "Film Grain"),
            Self::ColorGrading => write!(f, "Color Grading"),
            Self::DepthOfField => write!(f, "Depth of Field"),
            Self::MotionBlur => write!(f, "Motion Blur"),
            Self::Sharpen => write!(f, "Sharpen"),
            Self::Custom(id) => write!(f, "Custom({})", id),
        }
    }
}

/// Tone mapping operator selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TonemapOperator {
    Aces,
    Reinhard,
    Uncharted2,
    Neutral,
    AgX,
}

impl Default for TonemapOperator {
    fn default() -> Self {
        Self::Aces
    }
}

impl TonemapOperator {
    /// Returns the GPU enum value.
    pub fn gpu_value(&self) -> u32 {
        match self {
            Self::Aces => 0,
            Self::Reinhard => 1,
            Self::Uncharted2 => 2,
            Self::Neutral => 3,
            Self::AgX => 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Effect parameters
// ---------------------------------------------------------------------------

/// Bloom effect parameters.
#[derive(Debug, Clone)]
pub struct BloomParams {
    pub enabled: bool,
    pub threshold: f32,
    pub soft_knee: f32,
    pub intensity: f32,
    pub mip_levels: u32,
    pub scatter: f32,
    pub lens_dirt_intensity: f32,
    pub lens_dirt_texture: Option<u64>,
    pub energy_conserving: bool,
}

impl Default for BloomParams {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: DEFAULT_BLOOM_THRESHOLD,
            soft_knee: 0.5,
            intensity: DEFAULT_BLOOM_INTENSITY,
            mip_levels: 5,
            scatter: 0.7,
            lens_dirt_intensity: 0.0,
            lens_dirt_texture: None,
            energy_conserving: true,
        }
    }
}

/// Tonemap effect parameters.
#[derive(Debug, Clone)]
pub struct TonemapParams {
    pub enabled: bool,
    pub operator: TonemapOperator,
    pub exposure: f32,
    pub gamma: f32,
    pub white_point: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub lift: Vec3,
    pub gain: Vec3,
    pub auto_exposure: bool,
    pub auto_exposure_speed: f32,
    pub min_ev: f32,
    pub max_ev: f32,
}

impl Default for TonemapParams {
    fn default() -> Self {
        Self {
            enabled: true,
            operator: TonemapOperator::default(),
            exposure: 1.0,
            gamma: 2.2,
            white_point: 11.2,
            contrast: 1.0,
            saturation: 1.0,
            lift: Vec3::ZERO,
            gain: Vec3::ONE,
            auto_exposure: false,
            auto_exposure_speed: 2.0,
            min_ev: -4.0,
            max_ev: 16.0,
        }
    }
}

/// FXAA effect parameters.
#[derive(Debug, Clone)]
pub struct FxaaParams {
    pub enabled: bool,
    pub subpixel_quality: f32,
    pub edge_threshold: f32,
    pub edge_threshold_min: f32,
}

impl Default for FxaaParams {
    fn default() -> Self {
        Self {
            enabled: true,
            subpixel_quality: 0.75,
            edge_threshold: 0.166,
            edge_threshold_min: 0.0833,
        }
    }
}

/// Vignette effect parameters.
#[derive(Debug, Clone)]
pub struct VignetteParams {
    pub enabled: bool,
    pub intensity: f32,
    pub smoothness: f32,
    pub color: Vec3,
}

impl Default for VignetteParams {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 0.3,
            smoothness: 0.5,
            color: Vec3::ZERO,
        }
    }
}

/// Chromatic aberration parameters.
#[derive(Debug, Clone)]
pub struct ChromaticAberrationParams {
    pub enabled: bool,
    pub intensity: f32,
}

impl Default for ChromaticAberrationParams {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 0.005,
        }
    }
}

/// Film grain parameters.
#[derive(Debug, Clone)]
pub struct FilmGrainParams {
    pub enabled: bool,
    pub intensity: f32,
}

impl Default for FilmGrainParams {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 0.05,
        }
    }
}

// ---------------------------------------------------------------------------
// Post-process chain entry
// ---------------------------------------------------------------------------

/// A single entry in the post-process chain.
#[derive(Debug, Clone)]
pub struct PostProcessEntry {
    /// Effect type.
    pub effect: PostProcessEffect,
    /// Whether this effect is enabled.
    pub enabled: bool,
    /// Weight / intensity override (0 = off, 1 = full).
    pub weight: f32,
    /// Priority (lower = earlier in chain).
    pub priority: i32,
    /// GPU pipeline handle (cached after creation).
    pub pipeline_handle: Option<u64>,
}

impl PostProcessEntry {
    /// Creates a new entry.
    pub fn new(effect: PostProcessEffect, priority: i32) -> Self {
        Self {
            effect,
            enabled: true,
            weight: 1.0,
            priority,
            pipeline_handle: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Render target pool
// ---------------------------------------------------------------------------

/// A render target for ping-pong processing.
#[derive(Debug, Clone)]
pub struct PostProcessTarget {
    /// Texture handle.
    pub texture_handle: u64,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Texture format.
    pub format: PostProcessFormat,
    /// Whether this target is currently in use.
    pub in_use: bool,
    /// Mip level count.
    pub mip_levels: u32,
}

/// Post-process texture format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostProcessFormat {
    Rgba8Unorm,
    Rgba16Float,
    R11G11B10Float,
}

/// Pool of reusable render targets for post-processing.
#[derive(Debug)]
pub struct RenderTargetPool {
    /// Available targets.
    targets: Vec<PostProcessTarget>,
    /// Next unique handle.
    next_handle: u64,
    /// Maximum number of pooled targets.
    max_targets: usize,
    /// Total GPU memory used by the pool.
    total_memory_bytes: u64,
}

impl RenderTargetPool {
    /// Creates a new render target pool.
    pub fn new(max_targets: usize) -> Self {
        Self {
            targets: Vec::new(),
            next_handle: 1,
            max_targets,
            total_memory_bytes: 0,
        }
    }

    /// Acquire a render target of the given dimensions and format.
    pub fn acquire(&mut self, width: u32, height: u32, format: PostProcessFormat, mip_levels: u32) -> u64 {
        // Try to find a matching unused target.
        for target in &mut self.targets {
            if !target.in_use && target.width == width && target.height == height
                && target.format == format && target.mip_levels == mip_levels
            {
                target.in_use = true;
                return target.texture_handle;
            }
        }

        // Create a new target.
        let handle = self.next_handle;
        self.next_handle += 1;

        let bytes_per_pixel: u64 = match format {
            PostProcessFormat::Rgba8Unorm => 4,
            PostProcessFormat::Rgba16Float => 8,
            PostProcessFormat::R11G11B10Float => 4,
        };
        let mut memory = 0u64;
        for mip in 0..mip_levels {
            let w = (width >> mip).max(1) as u64;
            let h = (height >> mip).max(1) as u64;
            memory += w * h * bytes_per_pixel;
        }
        self.total_memory_bytes += memory;

        self.targets.push(PostProcessTarget {
            texture_handle: handle,
            width,
            height,
            format,
            in_use: true,
            mip_levels,
        });

        handle
    }

    /// Release a render target back to the pool.
    pub fn release(&mut self, handle: u64) {
        for target in &mut self.targets {
            if target.texture_handle == handle {
                target.in_use = false;
                return;
            }
        }
    }

    /// Release all targets back to the pool.
    pub fn release_all(&mut self) {
        for target in &mut self.targets {
            target.in_use = false;
        }
    }

    /// Returns pool statistics.
    pub fn stats(&self) -> RenderTargetPoolStats {
        let in_use = self.targets.iter().filter(|t| t.in_use).count();
        RenderTargetPoolStats {
            total_targets: self.targets.len(),
            in_use_targets: in_use,
            available_targets: self.targets.len() - in_use,
            total_memory_bytes: self.total_memory_bytes,
        }
    }
}

/// Render target pool statistics.
#[derive(Debug, Clone, Default)]
pub struct RenderTargetPoolStats {
    pub total_targets: usize,
    pub in_use_targets: usize,
    pub available_targets: usize,
    pub total_memory_bytes: u64,
}

// ---------------------------------------------------------------------------
// Post-process runner
// ---------------------------------------------------------------------------

/// The main post-processing runner.
///
/// Manages the chain of post-processing effects, ping-pong render targets,
/// and final blit to the swapchain. Each frame:
///
/// 1. `begin_frame` — set up ping-pong targets.
/// 2. For each enabled effect in order:
///    a. Bind the current source target.
///    b. Set up the destination target.
///    c. Execute the effect shader.
///    d. Swap source and destination.
/// 3. `end_frame` — blit the final result to the swapchain.
#[derive(Debug)]
pub struct PostProcessRunner {
    /// Effect chain (sorted by priority).
    pub chain: Vec<PostProcessEntry>,
    /// Render target pool.
    pub target_pool: RenderTargetPool,
    /// Bloom parameters.
    pub bloom: BloomParams,
    /// Tonemap parameters.
    pub tonemap: TonemapParams,
    /// FXAA parameters.
    pub fxaa: FxaaParams,
    /// Vignette parameters.
    pub vignette: VignetteParams,
    /// Chromatic aberration parameters.
    pub chromatic_aberration: ChromaticAberrationParams,
    /// Film grain parameters.
    pub film_grain: FilmGrainParams,
    /// Output resolution.
    pub output_resolution: UVec2,
    /// Internal HDR format.
    pub hdr_format: PostProcessFormat,
    /// Current ping-pong source handle.
    current_source: u64,
    /// Current ping-pong destination handle.
    current_dest: u64,
    /// Bloom mip chain handles.
    bloom_mip_chain: Vec<u64>,
    /// Frame counter for temporal effects.
    frame_number: u64,
    /// Time accumulator for animated effects.
    time: f32,
    /// Pipeline cache: effect -> pipeline handle.
    pipeline_cache: HashMap<PostProcessEffect, u64>,
    /// Statistics.
    pub stats: PostProcessStats,
}

impl PostProcessRunner {
    /// Creates a new post-process runner.
    pub fn new(width: u32, height: u32) -> Self {
        let mut runner = Self {
            chain: Vec::new(),
            target_pool: RenderTargetPool::new(32),
            bloom: BloomParams::default(),
            tonemap: TonemapParams::default(),
            fxaa: FxaaParams::default(),
            vignette: VignetteParams::default(),
            chromatic_aberration: ChromaticAberrationParams::default(),
            film_grain: FilmGrainParams::default(),
            output_resolution: UVec2::new(width, height),
            hdr_format: PostProcessFormat::Rgba16Float,
            current_source: 0,
            current_dest: 0,
            bloom_mip_chain: Vec::new(),
            frame_number: 0,
            time: 0.0,
            pipeline_cache: HashMap::new(),
            stats: PostProcessStats::default(),
        };

        // Set up default chain.
        runner.add_default_chain();
        runner
    }

    /// Add the default post-process chain.
    fn add_default_chain(&mut self) {
        self.chain.push(PostProcessEntry::new(PostProcessEffect::Bloom, 100));
        self.chain.push(PostProcessEntry::new(PostProcessEffect::Tonemap, 200));
        self.chain.push(PostProcessEntry::new(PostProcessEffect::Fxaa, 300));
        self.sort_chain();
    }

    /// Sort the effect chain by priority.
    fn sort_chain(&mut self) {
        self.chain.sort_by_key(|e| e.priority);
    }

    /// Add an effect to the chain.
    pub fn add_effect(&mut self, effect: PostProcessEffect, priority: i32) {
        // Don't add duplicates.
        if self.chain.iter().any(|e| e.effect == effect) {
            return;
        }
        self.chain.push(PostProcessEntry::new(effect, priority));
        self.sort_chain();
    }

    /// Remove an effect from the chain.
    pub fn remove_effect(&mut self, effect: PostProcessEffect) {
        self.chain.retain(|e| e.effect != effect);
    }

    /// Enable or disable an effect.
    pub fn set_effect_enabled(&mut self, effect: PostProcessEffect, enabled: bool) {
        for entry in &mut self.chain {
            if entry.effect == effect {
                entry.enabled = enabled;
                break;
            }
        }
    }

    /// Set the weight of an effect.
    pub fn set_effect_weight(&mut self, effect: PostProcessEffect, weight: f32) {
        for entry in &mut self.chain {
            if entry.effect == effect {
                entry.weight = weight.clamp(0.0, 1.0);
                break;
            }
        }
    }

    /// Called at the beginning of each frame to set up render targets.
    pub fn begin_frame(&mut self, dt: f32) {
        self.frame_number += 1;
        self.time += dt;
        self.stats = PostProcessStats::default();

        let w = self.output_resolution.x;
        let h = self.output_resolution.y;

        // Acquire ping-pong targets.
        self.current_source = self.target_pool.acquire(w, h, self.hdr_format, 1);
        self.current_dest = self.target_pool.acquire(w, h, self.hdr_format, 1);

        // Acquire bloom mip chain if bloom is enabled.
        self.bloom_mip_chain.clear();
        if self.bloom.enabled {
            let mip_count = self.bloom.mip_levels.min(MAX_BLOOM_MIP_LEVELS as u32);
            for i in 0..mip_count {
                let mip_w = (w >> (i + 1)).max(1);
                let mip_h = (h >> (i + 1)).max(1);
                let handle = self.target_pool.acquire(mip_w, mip_h, self.hdr_format, 1);
                self.bloom_mip_chain.push(handle);
            }
        }
    }

    /// Execute all post-process effects for the current frame.
    ///
    /// Returns a list of effect executions for the GPU command encoder to record.
    pub fn execute(&mut self) -> Vec<PostProcessExecution> {
        let mut executions = Vec::new();

        for entry in &self.chain {
            if !entry.enabled || entry.weight < 0.001 {
                continue;
            }

            match entry.effect {
                PostProcessEffect::Bloom => {
                    if self.bloom.enabled {
                        let bloom_execs = self.build_bloom_executions();
                        executions.extend(bloom_execs);
                        self.stats.effects_executed += 1;
                    }
                }
                PostProcessEffect::Tonemap => {
                    if self.tonemap.enabled {
                        executions.push(self.build_tonemap_execution());
                        self.swap_targets();
                        self.stats.effects_executed += 1;
                    }
                }
                PostProcessEffect::Fxaa => {
                    if self.fxaa.enabled {
                        executions.push(self.build_fxaa_execution());
                        self.swap_targets();
                        self.stats.effects_executed += 1;
                    }
                }
                PostProcessEffect::ChromaticAberration => {
                    if self.chromatic_aberration.enabled {
                        executions.push(self.build_chromatic_aberration_execution());
                        self.swap_targets();
                        self.stats.effects_executed += 1;
                    }
                }
                _ => {
                    // Custom or unimplemented effects are skipped.
                    self.stats.effects_skipped += 1;
                }
            }
        }

        // Final blit.
        executions.push(PostProcessExecution {
            effect: PostProcessEffect::Custom(9999),
            shader_source: FINAL_BLIT_WGSL,
            source_texture: self.current_source,
            dest_texture: 0, // Swapchain.
            resolution: self.output_resolution,
            uniforms: PostProcessUniforms::None,
        });

        self.stats.total_passes = executions.len() as u32;
        executions
    }

    /// End the frame, releasing render targets.
    pub fn end_frame(&mut self) {
        self.target_pool.release(self.current_source);
        self.target_pool.release(self.current_dest);
        for &handle in &self.bloom_mip_chain {
            self.target_pool.release(handle);
        }
        self.bloom_mip_chain.clear();
    }

    /// Swap source and destination targets.
    fn swap_targets(&mut self) {
        std::mem::swap(&mut self.current_source, &mut self.current_dest);
    }

    /// Build bloom executions (threshold + downsample chain + upsample chain).
    fn build_bloom_executions(&mut self) -> Vec<PostProcessExecution> {
        let mut execs = Vec::new();
        let w = self.output_resolution.x;
        let h = self.output_resolution.y;

        // 1. Threshold extraction.
        if !self.bloom_mip_chain.is_empty() {
            execs.push(PostProcessExecution {
                effect: PostProcessEffect::Bloom,
                shader_source: BLOOM_THRESHOLD_WGSL,
                source_texture: self.current_source,
                dest_texture: self.bloom_mip_chain[0],
                resolution: UVec2::new((w >> 1).max(1), (h >> 1).max(1)),
                uniforms: PostProcessUniforms::Bloom {
                    threshold: self.bloom.threshold,
                    soft_knee: self.bloom.soft_knee,
                    intensity: self.bloom.intensity,
                },
            });
        }

        // 2. Downsample chain.
        for i in 1..self.bloom_mip_chain.len() {
            let mip_w = (w >> (i as u32 + 1)).max(1);
            let mip_h = (h >> (i as u32 + 1)).max(1);
            execs.push(PostProcessExecution {
                effect: PostProcessEffect::Bloom,
                shader_source: BLOOM_DOWNSAMPLE_WGSL,
                source_texture: self.bloom_mip_chain[i - 1],
                dest_texture: self.bloom_mip_chain[i],
                resolution: UVec2::new(mip_w, mip_h),
                uniforms: PostProcessUniforms::TexelSize {
                    texel_size: Vec2::new(1.0 / mip_w as f32, 1.0 / mip_h as f32),
                },
            });
        }

        // 3. Upsample chain (reverse).
        for i in (0..self.bloom_mip_chain.len().saturating_sub(1)).rev() {
            let mip_w = (w >> (i as u32 + 1)).max(1);
            let mip_h = (h >> (i as u32 + 1)).max(1);
            execs.push(PostProcessExecution {
                effect: PostProcessEffect::Bloom,
                shader_source: BLOOM_UPSAMPLE_WGSL,
                source_texture: self.bloom_mip_chain[i + 1],
                dest_texture: self.bloom_mip_chain[i],
                resolution: UVec2::new(mip_w, mip_h),
                uniforms: PostProcessUniforms::BloomUpsample {
                    texel_size: Vec2::new(1.0 / mip_w as f32, 1.0 / mip_h as f32),
                    blend_factor: self.bloom.scatter,
                },
            });
        }

        execs
    }

    /// Build tonemap execution.
    fn build_tonemap_execution(&self) -> PostProcessExecution {
        PostProcessExecution {
            effect: PostProcessEffect::Tonemap,
            shader_source: TONEMAP_WGSL,
            source_texture: self.current_source,
            dest_texture: self.current_dest,
            resolution: self.output_resolution,
            uniforms: PostProcessUniforms::Tonemap {
                exposure: self.tonemap.exposure,
                gamma: self.tonemap.gamma,
                operator: self.tonemap.operator.gpu_value(),
                white_point: self.tonemap.white_point,
                contrast: self.tonemap.contrast,
                saturation: self.tonemap.saturation,
                lift: self.tonemap.lift,
                gain: self.tonemap.gain,
                vignette_intensity: if self.vignette.enabled { self.vignette.intensity } else { 0.0 },
                vignette_smoothness: self.vignette.smoothness,
                film_grain: if self.film_grain.enabled { self.film_grain.intensity } else { 0.0 },
                time: self.time,
            },
        }
    }

    /// Build FXAA execution.
    fn build_fxaa_execution(&self) -> PostProcessExecution {
        let w = self.output_resolution.x as f32;
        let h = self.output_resolution.y as f32;
        PostProcessExecution {
            effect: PostProcessEffect::Fxaa,
            shader_source: FXAA_WGSL,
            source_texture: self.current_source,
            dest_texture: self.current_dest,
            resolution: self.output_resolution,
            uniforms: PostProcessUniforms::Fxaa {
                texel_size: Vec2::new(1.0 / w, 1.0 / h),
                subpixel_quality: self.fxaa.subpixel_quality,
                edge_threshold: self.fxaa.edge_threshold,
                edge_threshold_min: self.fxaa.edge_threshold_min,
            },
        }
    }

    /// Build chromatic aberration execution.
    fn build_chromatic_aberration_execution(&self) -> PostProcessExecution {
        PostProcessExecution {
            effect: PostProcessEffect::ChromaticAberration,
            shader_source: CHROMATIC_ABERRATION_WGSL,
            source_texture: self.current_source,
            dest_texture: self.current_dest,
            resolution: self.output_resolution,
            uniforms: PostProcessUniforms::ChromaticAberration {
                intensity: self.chromatic_aberration.intensity,
            },
        }
    }

    /// Handle window resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.output_resolution = UVec2::new(width, height);
        // Release all pooled targets so they get recreated at the new size.
        self.target_pool.release_all();
    }

    /// Returns the source texture handle (input to the post-process chain).
    pub fn source_texture(&self) -> u64 {
        self.current_source
    }

    /// Apply a preset for different quality levels.
    pub fn apply_preset(&mut self, preset: PostProcessPreset) {
        match preset {
            PostProcessPreset::Low => {
                self.bloom.enabled = false;
                self.fxaa.enabled = true;
                self.fxaa.subpixel_quality = 0.5;
                self.chromatic_aberration.enabled = false;
                self.film_grain.enabled = false;
                self.vignette.enabled = false;
            }
            PostProcessPreset::Medium => {
                self.bloom.enabled = true;
                self.bloom.mip_levels = 3;
                self.fxaa.enabled = true;
                self.fxaa.subpixel_quality = 0.75;
                self.chromatic_aberration.enabled = false;
                self.film_grain.enabled = false;
                self.vignette.enabled = true;
            }
            PostProcessPreset::High => {
                self.bloom.enabled = true;
                self.bloom.mip_levels = 5;
                self.fxaa.enabled = true;
                self.fxaa.subpixel_quality = 0.75;
                self.chromatic_aberration.enabled = true;
                self.film_grain.enabled = true;
                self.film_grain.intensity = 0.03;
                self.vignette.enabled = true;
            }
            PostProcessPreset::Ultra => {
                self.bloom.enabled = true;
                self.bloom.mip_levels = 7;
                self.fxaa.enabled = true;
                self.fxaa.subpixel_quality = 1.0;
                self.chromatic_aberration.enabled = true;
                self.chromatic_aberration.intensity = 0.003;
                self.film_grain.enabled = true;
                self.film_grain.intensity = 0.02;
                self.vignette.enabled = true;
                self.vignette.intensity = 0.25;
            }
        }
    }
}

/// Post-process quality preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostProcessPreset {
    Low,
    Medium,
    High,
    Ultra,
}

/// Describes a single post-process pass to execute on the GPU.
#[derive(Debug, Clone)]
pub struct PostProcessExecution {
    /// Which effect this execution belongs to.
    pub effect: PostProcessEffect,
    /// WGSL shader source for this pass.
    pub shader_source: &'static str,
    /// Source texture handle.
    pub source_texture: u64,
    /// Destination texture handle (0 = swapchain).
    pub dest_texture: u64,
    /// Resolution of the destination target.
    pub resolution: UVec2,
    /// Uniform data for this pass.
    pub uniforms: PostProcessUniforms,
}

/// Uniform data variants for post-process effects.
#[derive(Debug, Clone)]
pub enum PostProcessUniforms {
    None,
    Bloom {
        threshold: f32,
        soft_knee: f32,
        intensity: f32,
    },
    TexelSize {
        texel_size: Vec2,
    },
    BloomUpsample {
        texel_size: Vec2,
        blend_factor: f32,
    },
    Tonemap {
        exposure: f32,
        gamma: f32,
        operator: u32,
        white_point: f32,
        contrast: f32,
        saturation: f32,
        lift: Vec3,
        gain: Vec3,
        vignette_intensity: f32,
        vignette_smoothness: f32,
        film_grain: f32,
        time: f32,
    },
    Fxaa {
        texel_size: Vec2,
        subpixel_quality: f32,
        edge_threshold: f32,
        edge_threshold_min: f32,
    },
    ChromaticAberration {
        intensity: f32,
    },
}

/// Post-process runner statistics.
#[derive(Debug, Clone, Default)]
pub struct PostProcessStats {
    /// Total render passes this frame.
    pub total_passes: u32,
    /// Number of effects executed.
    pub effects_executed: u32,
    /// Number of effects skipped (disabled or unsupported).
    pub effects_skipped: u32,
}

// ---------------------------------------------------------------------------
// ECS component
// ---------------------------------------------------------------------------

/// ECS component for post-processing settings.
#[derive(Debug, Clone)]
pub struct PostProcessComponent {
    /// Whether post-processing is enabled.
    pub enabled: bool,
    /// Quality preset.
    pub preset: PostProcessPreset,
    /// Bloom override.
    pub bloom_override: Option<BloomParams>,
    /// Tonemap override.
    pub tonemap_override: Option<TonemapParams>,
    /// FXAA override.
    pub fxaa_override: Option<FxaaParams>,
}

impl Default for PostProcessComponent {
    fn default() -> Self {
        Self {
            enabled: true,
            preset: PostProcessPreset::High,
            bloom_override: None,
            tonemap_override: None,
            fxaa_override: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runner_creation() {
        let runner = PostProcessRunner::new(1920, 1080);
        assert_eq!(runner.output_resolution, UVec2::new(1920, 1080));
        assert!(!runner.chain.is_empty());
    }

    #[test]
    fn test_default_chain_order() {
        let runner = PostProcessRunner::new(800, 600);
        assert_eq!(runner.chain[0].effect, PostProcessEffect::Bloom);
        assert_eq!(runner.chain[1].effect, PostProcessEffect::Tonemap);
        assert_eq!(runner.chain[2].effect, PostProcessEffect::Fxaa);
    }

    #[test]
    fn test_add_remove_effect() {
        let mut runner = PostProcessRunner::new(800, 600);
        runner.add_effect(PostProcessEffect::ChromaticAberration, 250);
        assert_eq!(runner.chain.len(), 4);
        runner.remove_effect(PostProcessEffect::ChromaticAberration);
        assert_eq!(runner.chain.len(), 3);
    }

    #[test]
    fn test_render_target_pool() {
        let mut pool = RenderTargetPool::new(16);
        let h1 = pool.acquire(1920, 1080, PostProcessFormat::Rgba16Float, 1);
        let h2 = pool.acquire(1920, 1080, PostProcessFormat::Rgba16Float, 1);
        assert_ne!(h1, h2);

        pool.release(h1);
        let h3 = pool.acquire(1920, 1080, PostProcessFormat::Rgba16Float, 1);
        assert_eq!(h1, h3); // Reused from pool.
    }

    #[test]
    fn test_tonemap_operator_gpu_value() {
        assert_eq!(TonemapOperator::Aces.gpu_value(), 0);
        assert_eq!(TonemapOperator::AgX.gpu_value(), 4);
    }

    #[test]
    fn test_preset_application() {
        let mut runner = PostProcessRunner::new(800, 600);
        runner.apply_preset(PostProcessPreset::Low);
        assert!(!runner.bloom.enabled);
        assert!(runner.fxaa.enabled);

        runner.apply_preset(PostProcessPreset::Ultra);
        assert!(runner.bloom.enabled);
        assert_eq!(runner.bloom.mip_levels, 7);
    }

    #[test]
    fn test_resize() {
        let mut runner = PostProcessRunner::new(800, 600);
        runner.resize(1920, 1080);
        assert_eq!(runner.output_resolution, UVec2::new(1920, 1080));
    }
}
