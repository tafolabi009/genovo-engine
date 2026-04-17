// engine/render/src/gpu_post_process.rs
//
// GPU Post-Processing Pipeline for the Genovo engine.
//
// Provides a chain of fullscreen GPU passes using ping-pong render targets:
//
// - **Bloom**: Bright extraction, downsample (13-tap), upsample (9-tap tent),
//   composite with additive blend
// - **Tone mapping**: ACES filmic as a separate post-process pass for HDR
// - **FXAA**: Edge detection + sub-pixel antialiasing in a single WGSL pass
// - **Vignette**: Darken screen edges based on UV distance from center
// - **Fullscreen quad**: vertex shader that generates a triangle from vertex_index
//
// Each effect has its own WGSL shader, render pipeline, and bind group.

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3, Vec4};

// ============================================================================
// Constants
// ============================================================================

/// Maximum number of bloom downsample/upsample levels.
pub const MAX_BLOOM_LEVELS: usize = 8;

/// Default bloom intensity.
pub const DEFAULT_BLOOM_INTENSITY: f32 = 0.04;

/// Default bloom threshold.
pub const DEFAULT_BLOOM_THRESHOLD: f32 = 1.0;

/// Default FXAA edge threshold minimum.
pub const FXAA_EDGE_THRESHOLD_MIN: f32 = 0.0625;

/// Default FXAA edge threshold.
pub const FXAA_EDGE_THRESHOLD: f32 = 0.125;

/// Default FXAA sub-pixel quality.
pub const FXAA_SUBPIXEL_QUALITY: f32 = 0.75;

// ============================================================================
// Post-process configuration
// ============================================================================

/// Configuration for the post-processing pipeline.
#[derive(Debug, Clone)]
pub struct PostProcessConfig {
    /// Enable bloom effect.
    pub bloom_enabled: bool,
    /// Bloom brightness threshold.
    pub bloom_threshold: f32,
    /// Bloom soft threshold (knee).
    pub bloom_soft_threshold: f32,
    /// Bloom intensity multiplier.
    pub bloom_intensity: f32,
    /// Number of bloom downsample levels.
    pub bloom_levels: usize,

    /// Enable FXAA.
    pub fxaa_enabled: bool,
    /// FXAA edge detection threshold.
    pub fxaa_edge_threshold: f32,
    /// FXAA minimum edge threshold.
    pub fxaa_edge_threshold_min: f32,
    /// FXAA sub-pixel quality.
    pub fxaa_subpixel_quality: f32,

    /// Enable tone mapping.
    pub tone_mapping_enabled: bool,
    /// Exposure for tone mapping.
    pub exposure: f32,

    /// Enable vignette.
    pub vignette_enabled: bool,
    /// Vignette intensity (0 = none, 1 = max).
    pub vignette_intensity: f32,
    /// Vignette smoothness.
    pub vignette_smoothness: f32,
    /// Vignette roundness (1 = circular, 0 = rectangular).
    pub vignette_roundness: f32,

    /// Enable chromatic aberration.
    pub chromatic_aberration_enabled: bool,
    /// Chromatic aberration intensity.
    pub chromatic_aberration_intensity: f32,

    /// Enable film grain.
    pub film_grain_enabled: bool,
    /// Film grain intensity.
    pub film_grain_intensity: f32,
}

impl Default for PostProcessConfig {
    fn default() -> Self {
        Self {
            bloom_enabled: true,
            bloom_threshold: DEFAULT_BLOOM_THRESHOLD,
            bloom_soft_threshold: 0.5,
            bloom_intensity: DEFAULT_BLOOM_INTENSITY,
            bloom_levels: 5,
            fxaa_enabled: true,
            fxaa_edge_threshold: FXAA_EDGE_THRESHOLD,
            fxaa_edge_threshold_min: FXAA_EDGE_THRESHOLD_MIN,
            fxaa_subpixel_quality: FXAA_SUBPIXEL_QUALITY,
            tone_mapping_enabled: true,
            exposure: 1.0,
            vignette_enabled: true,
            vignette_intensity: 0.3,
            vignette_smoothness: 2.0,
            vignette_roundness: 1.0,
            chromatic_aberration_enabled: false,
            chromatic_aberration_intensity: 0.005,
            film_grain_enabled: false,
            film_grain_intensity: 0.05,
        }
    }
}

// ============================================================================
// Post-process uniform types
// ============================================================================

/// Bloom parameters uniform.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BloomParamsUniform {
    /// .x = threshold, .y = soft_threshold (knee), .z = intensity, .w = unused
    pub params: [f32; 4],
    /// .x = texel_size_x, .y = texel_size_y, .z = level, .w = max_levels
    pub texel_info: [f32; 4],
}

/// FXAA parameters uniform.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FxaaParamsUniform {
    /// .x = edge_threshold, .y = edge_threshold_min, .z = subpixel_quality, .w = unused
    pub params: [f32; 4],
    /// .x = texel_size_x, .y = texel_size_y, .z = unused, .w = unused
    pub texel_info: [f32; 4],
}

/// Tone mapping parameters uniform.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ToneMappingParamsUniform {
    /// .x = exposure, .y = gamma, .z = tone_map_mode (0=ACES, 1=Reinhard, 2=Uncharted2), .w = unused
    pub params: [f32; 4],
}

/// Vignette parameters uniform.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct VignetteParamsUniform {
    /// .x = intensity, .y = smoothness, .z = roundness, .w = unused
    pub params: [f32; 4],
}

/// Combined post-process parameters uniform.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PostProcessParamsUniform {
    /// Bloom: .x=threshold, .y=soft_knee, .z=intensity, .w=num_levels
    pub bloom_params: [f32; 4],
    /// FXAA: .x=edge_threshold, .y=edge_threshold_min, .z=subpixel, .w=enabled
    pub fxaa_params: [f32; 4],
    /// Tone map: .x=exposure, .y=gamma, .z=mode, .w=enabled
    pub tonemapping_params: [f32; 4],
    /// Vignette: .x=intensity, .y=smoothness, .z=roundness, .w=enabled
    pub vignette_params: [f32; 4],
    /// Screen: .x=width, .y=height, .z=texel_size_x, .w=texel_size_y
    pub screen_params: [f32; 4],
    /// Time: .x=time, .y=delta_time, .z=frame_count, .w=unused
    pub time_params: [f32; 4],
    /// Chromatic aberration: .x=intensity, .y=enabled, .zw=unused
    pub chromatic_params: [f32; 4],
    /// Film grain: .x=intensity, .y=enabled, .zw=unused
    pub grain_params: [f32; 4],
}

// ============================================================================
// Fullscreen vertex shader (shared by all post-process passes)
// ============================================================================

/// Fullscreen triangle vertex shader used by all post-process effects.
///
/// Generates a single triangle that covers the entire screen using only
/// the vertex index (no vertex buffer needed).
pub const FULLSCREEN_VERTEX_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;

    // Generate a fullscreen triangle from vertex index.
    // Vertex 0: (-1, -1), Vertex 1: (3, -1), Vertex 2: (-1, 3)
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);

    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>(
        (x + 1.0) * 0.5,
        (1.0 - y) * 0.5
    );

    return output;
}
"#;

// ============================================================================
// Bloom shaders
// ============================================================================

/// Bloom bright pass: extract pixels above the brightness threshold.
pub const BLOOM_BRIGHT_PASS_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Bloom Bright Pass
// ============================================================================
//
// Extracts pixels brighter than the threshold with a soft knee.

struct BloomParams {
    params: vec4<f32>,
    texel_info: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return output;
}

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> bloom: BloomParams;

// Luminance using perceptual weights.
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Soft thresholding with knee.
fn soft_threshold(color: vec3<f32>, threshold: f32, knee: f32) -> vec3<f32> {
    let brightness = luminance(color);
    let soft = brightness - threshold + knee;
    let soft_clamped = clamp(soft, 0.0, 2.0 * knee);
    let contribution = soft_clamped * soft_clamped / (4.0 * knee + 0.00001);
    let w = max(brightness - threshold, contribution) / max(brightness, 0.00001);
    return color * w;
}

@fragment
fn fs_bloom_bright(input: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(source_texture, source_sampler, input.uv).xyz;
    let threshold = bloom.params.x;
    let knee = bloom.params.y;

    let bright = soft_threshold(color, threshold, knee);

    return vec4<f32>(bright, 1.0);
}
"#;

/// Bloom downsample shader using a 13-tap filter (Jimenez 2014).
pub const BLOOM_DOWNSAMPLE_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Bloom Downsample (13-tap)
// ============================================================================
//
// High quality downsampling filter that reduces aliasing and firefly artifacts.
// Uses 13 bilinear taps arranged in a pattern that matches a box+tent filter.

struct BloomParams {
    params: vec4<f32>,
    texel_info: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return output;
}

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> bloom: BloomParams;

@fragment
fn fs_bloom_downsample(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel = vec2<f32>(bloom.texel_info.x, bloom.texel_info.y);
    let uv = input.uv;

    // 13-tap downsample filter.
    // Center cross (4 taps at half-texel offset).
    let a = textureSample(source_texture, source_sampler, uv + vec2<f32>(-1.0, -1.0) * texel).xyz;
    let b = textureSample(source_texture, source_sampler, uv + vec2<f32>( 0.0, -1.0) * texel).xyz;
    let c = textureSample(source_texture, source_sampler, uv + vec2<f32>( 1.0, -1.0) * texel).xyz;
    let d = textureSample(source_texture, source_sampler, uv + vec2<f32>(-0.5, -0.5) * texel).xyz;
    let e = textureSample(source_texture, source_sampler, uv + vec2<f32>( 0.5, -0.5) * texel).xyz;
    let f = textureSample(source_texture, source_sampler, uv + vec2<f32>(-1.0,  0.0) * texel).xyz;
    let g = textureSample(source_texture, source_sampler, uv).xyz;
    let h = textureSample(source_texture, source_sampler, uv + vec2<f32>( 1.0,  0.0) * texel).xyz;
    let i = textureSample(source_texture, source_sampler, uv + vec2<f32>(-0.5,  0.5) * texel).xyz;
    let j = textureSample(source_texture, source_sampler, uv + vec2<f32>( 0.5,  0.5) * texel).xyz;
    let k = textureSample(source_texture, source_sampler, uv + vec2<f32>(-1.0,  1.0) * texel).xyz;
    let l = textureSample(source_texture, source_sampler, uv + vec2<f32>( 0.0,  1.0) * texel).xyz;
    let m = textureSample(source_texture, source_sampler, uv + vec2<f32>( 1.0,  1.0) * texel).xyz;

    // Weighted combination following Jimenez 2014 pattern.
    var color = g * 0.125; // Center sample weight.
    color = color + (d + e + i + j) * 0.125; // Inner ring.
    color = color + (a + b + f + g) * 0.03125; // Outer ring NW.
    color = color + (b + c + g + h) * 0.03125; // Outer ring NE.
    color = color + (f + g + k + l) * 0.03125; // Outer ring SW.
    color = color + (g + h + l + m) * 0.03125; // Outer ring SE.

    return vec4<f32>(color, 1.0);
}
"#;

/// Bloom upsample shader using a 9-tap tent filter.
pub const BLOOM_UPSAMPLE_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Bloom Upsample (9-tap tent)
// ============================================================================
//
// Upsamples the bloom mip chain using a 3x3 tent filter for smooth results.

struct BloomParams {
    params: vec4<f32>,
    texel_info: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return output;
}

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> bloom: BloomParams;

@fragment
fn fs_bloom_upsample(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel = vec2<f32>(bloom.texel_info.x, bloom.texel_info.y);
    let uv = input.uv;
    let intensity = bloom.params.z;

    // 9-tap tent filter (3x3 bilinear).
    let a = textureSample(source_texture, source_sampler, uv + vec2<f32>(-1.0, -1.0) * texel).xyz;
    let b = textureSample(source_texture, source_sampler, uv + vec2<f32>( 0.0, -1.0) * texel).xyz;
    let c = textureSample(source_texture, source_sampler, uv + vec2<f32>( 1.0, -1.0) * texel).xyz;
    let d = textureSample(source_texture, source_sampler, uv + vec2<f32>(-1.0,  0.0) * texel).xyz;
    let e = textureSample(source_texture, source_sampler, uv).xyz;
    let f = textureSample(source_texture, source_sampler, uv + vec2<f32>( 1.0,  0.0) * texel).xyz;
    let g = textureSample(source_texture, source_sampler, uv + vec2<f32>(-1.0,  1.0) * texel).xyz;
    let h = textureSample(source_texture, source_sampler, uv + vec2<f32>( 0.0,  1.0) * texel).xyz;
    let i = textureSample(source_texture, source_sampler, uv + vec2<f32>( 1.0,  1.0) * texel).xyz;

    // Tent filter weights: corners=1, edges=2, center=4, total=16.
    var color = e * 4.0;
    color = color + (b + d + f + h) * 2.0;
    color = color + (a + c + g + i);
    color = color / 16.0;

    return vec4<f32>(color * intensity, 1.0);
}
"#;

/// Bloom composite shader: adds bloom to the scene.
pub const BLOOM_COMPOSITE_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Bloom Composite
// ============================================================================
//
// Additively blends the bloom result with the original scene.

struct BloomParams {
    params: vec4<f32>,
    texel_info: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return output;
}

@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var bloom_texture: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var<uniform> bloom: BloomParams;

@fragment
fn fs_bloom_composite(input: VertexOutput) -> @location(0) vec4<f32> {
    let scene = textureSample(scene_texture, tex_sampler, input.uv).xyz;
    let bloom_color = textureSample(bloom_texture, tex_sampler, input.uv).xyz;
    let intensity = bloom.params.z;

    let result = scene + bloom_color * intensity;
    return vec4<f32>(result, 1.0);
}
"#;

// ============================================================================
// Tone mapping shader
// ============================================================================

/// Tone mapping post-process shader.
pub const TONE_MAPPING_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Tone Mapping Post-Process
// ============================================================================
//
// Converts HDR scene colour to LDR with configurable tone mapping operators.
// Supports ACES filmic, Reinhard, and Uncharted 2 operators.

struct ToneMapParams {
    params: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return output;
}

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> tone_params: ToneMapParams;

const INV_GAMMA: f32 = 0.45454545454;

// ACES filmic tone mapping.
fn tone_map_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + vec3<f32>(b))) / (x * (c * x + vec3<f32>(d)) + vec3<f32>(e)),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

// Reinhard tone mapping.
fn tone_map_reinhard(color: vec3<f32>) -> vec3<f32> {
    return color / (color + vec3<f32>(1.0));
}

// Uncharted 2 / Hable tone mapping.
fn uncharted2_partial(x: vec3<f32>) -> vec3<f32> {
    let A = 0.15;
    let B = 0.50;
    let C = 0.10;
    let D = 0.20;
    let E = 0.02;
    let F = 0.30;
    return ((x * (A * x + vec3<f32>(C * B)) + vec3<f32>(D * E)) /
            (x * (A * x + vec3<f32>(B)) + vec3<f32>(D * F))) - vec3<f32>(E / F);
}

fn tone_map_uncharted2(color: vec3<f32>) -> vec3<f32> {
    let W = vec3<f32>(11.2);
    let numerator = uncharted2_partial(color * 2.0);
    let denominator = uncharted2_partial(W);
    return numerator / denominator;
}

// Linear to sRGB gamma correction.
fn linear_to_srgb_f(c: f32) -> f32 {
    if c <= 0.0031308 {
        return c * 12.92;
    }
    return 1.055 * pow(c, INV_GAMMA) - 0.055;
}

@fragment
fn fs_tone_mapping(input: VertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(source_texture, source_sampler, input.uv).xyz;
    let exposure = tone_params.params.x;
    let mode = i32(tone_params.params.z);

    // Apply exposure.
    color = color * exposure;

    // Apply tone mapping.
    switch mode {
        case 0: { color = tone_map_aces(color); }
        case 1: { color = tone_map_reinhard(color); }
        case 2: { color = tone_map_uncharted2(color); }
        default: { color = tone_map_aces(color); }
    }

    // Gamma correction.
    color = vec3<f32>(
        linear_to_srgb_f(color.x),
        linear_to_srgb_f(color.y),
        linear_to_srgb_f(color.z)
    );

    return vec4<f32>(color, 1.0);
}
"#;

// ============================================================================
// FXAA shader
// ============================================================================

/// FXAA (Fast Approximate Anti-Aliasing) shader.
pub const FXAA_SHADER_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- FXAA Post-Process
// ============================================================================
//
// Single-pass FXAA implementation based on NVIDIA FXAA 3.11.
// Detects edges using luminance contrast and applies sub-pixel smoothing.

struct FxaaParams {
    params: vec4<f32>,
    texel_info: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return output;
}

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> fxaa: FxaaParams;

// Perceptual luminance.
fn luma(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_fxaa(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel = vec2<f32>(fxaa.texel_info.x, fxaa.texel_info.y);
    let uv = input.uv;
    let edge_threshold = fxaa.params.x;
    let edge_threshold_min = fxaa.params.y;
    let subpixel_quality = fxaa.params.z;

    // Sample the 3x3 neighbourhood.
    let color_center = textureSample(source_texture, source_sampler, uv).xyz;
    let luma_center = luma(color_center);

    let luma_n = luma(textureSample(source_texture, source_sampler, uv + vec2<f32>( 0.0, -1.0) * texel).xyz);
    let luma_s = luma(textureSample(source_texture, source_sampler, uv + vec2<f32>( 0.0,  1.0) * texel).xyz);
    let luma_e = luma(textureSample(source_texture, source_sampler, uv + vec2<f32>( 1.0,  0.0) * texel).xyz);
    let luma_w = luma(textureSample(source_texture, source_sampler, uv + vec2<f32>(-1.0,  0.0) * texel).xyz);

    // Compute local contrast.
    let luma_min = min(luma_center, min(min(luma_n, luma_s), min(luma_e, luma_w)));
    let luma_max = max(luma_center, max(max(luma_n, luma_s), max(luma_e, luma_w)));
    let luma_range = luma_max - luma_min;

    // Early exit if contrast is too low.
    if luma_range < max(edge_threshold_min, luma_max * edge_threshold) {
        return vec4<f32>(color_center, 1.0);
    }

    // Sample corner neighbours.
    let luma_nw = luma(textureSample(source_texture, source_sampler, uv + vec2<f32>(-1.0, -1.0) * texel).xyz);
    let luma_ne = luma(textureSample(source_texture, source_sampler, uv + vec2<f32>( 1.0, -1.0) * texel).xyz);
    let luma_sw = luma(textureSample(source_texture, source_sampler, uv + vec2<f32>(-1.0,  1.0) * texel).xyz);
    let luma_se = luma(textureSample(source_texture, source_sampler, uv + vec2<f32>( 1.0,  1.0) * texel).xyz);

    // Determine edge direction (horizontal or vertical).
    let horizontal = abs(luma_n + luma_s - 2.0 * luma_center) * 2.0 + abs(luma_ne + luma_se - 2.0 * luma_e) + abs(luma_nw + luma_sw - 2.0 * luma_w);
    let vertical = abs(luma_e + luma_w - 2.0 * luma_center) * 2.0 + abs(luma_ne + luma_nw - 2.0 * luma_n) + abs(luma_se + luma_sw - 2.0 * luma_s);
    let is_horizontal = horizontal >= vertical;

    // Select the two neighbours along the edge gradient.
    var step_length: f32;
    var luma_negative: f32;
    var luma_positive: f32;

    if is_horizontal {
        step_length = texel.y;
        luma_negative = luma_n;
        luma_positive = luma_s;
    } else {
        step_length = texel.x;
        luma_negative = luma_w;
        luma_positive = luma_e;
    }

    let gradient_negative = abs(luma_negative - luma_center);
    let gradient_positive = abs(luma_positive - luma_center);

    var correct_variation = gradient_negative;
    var local_luma = luma_negative;

    if gradient_positive > gradient_negative {
        step_length = -step_length;
        correct_variation = gradient_positive;
        local_luma = luma_positive;
    }

    // Compute the sub-pixel offset.
    let luma_average_all = (luma_n + luma_s + luma_e + luma_w) * 0.25;
    let subpixel_offset = clamp(abs(luma_average_all - luma_center) / luma_range, 0.0, 1.0);
    let subpixel_factor = (-2.0 * subpixel_offset + 3.0) * subpixel_offset * subpixel_offset;
    let final_offset = subpixel_factor * subpixel_factor * subpixel_quality;

    // Apply the offset.
    var offset_uv = uv;
    if is_horizontal {
        offset_uv.y = offset_uv.y + step_length * final_offset;
    } else {
        offset_uv.x = offset_uv.x + step_length * final_offset;
    }

    // Edge walking: search along the edge for the end point.
    var edge_uv = uv;
    var edge_step: vec2<f32>;
    if is_horizontal {
        edge_uv.y = edge_uv.y + step_length * 0.5;
        edge_step = vec2<f32>(texel.x, 0.0);
    } else {
        edge_uv.x = edge_uv.x + step_length * 0.5;
        edge_step = vec2<f32>(0.0, texel.y);
    }

    var uv_neg = edge_uv - edge_step;
    var uv_pos = edge_uv + edge_step;
    let edge_threshold_inner = correct_variation * 0.25;

    var luma_end_neg = luma(textureSample(source_texture, source_sampler, uv_neg).xyz) - local_luma;
    var luma_end_pos = luma(textureSample(source_texture, source_sampler, uv_pos).xyz) - local_luma;

    var reached_neg = abs(luma_end_neg) >= edge_threshold_inner;
    var reached_pos = abs(luma_end_pos) >= edge_threshold_inner;

    // Walk up to 12 steps along the edge.
    let quality_steps = array<f32, 12>(1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0);

    for (var i = 0; i < 12; i = i + 1) {
        if reached_neg && reached_pos {
            break;
        }
        if !reached_neg {
            uv_neg = uv_neg - edge_step * quality_steps[i];
            luma_end_neg = luma(textureSample(source_texture, source_sampler, uv_neg).xyz) - local_luma;
            reached_neg = abs(luma_end_neg) >= edge_threshold_inner;
        }
        if !reached_pos {
            uv_pos = uv_pos + edge_step * quality_steps[i];
            luma_end_pos = luma(textureSample(source_texture, source_sampler, uv_pos).xyz) - local_luma;
            reached_pos = abs(luma_end_pos) >= edge_threshold_inner;
        }
    }

    // Compute distances.
    var dist_neg: f32;
    var dist_pos: f32;
    if is_horizontal {
        dist_neg = uv.x - uv_neg.x;
        dist_pos = uv_pos.x - uv.x;
    } else {
        dist_neg = uv.y - uv_neg.y;
        dist_pos = uv_pos.y - uv.y;
    }

    let is_closer_neg = dist_neg < dist_pos;
    let closer_dist = min(dist_neg, dist_pos);
    let edge_length = dist_neg + dist_pos;

    // Compute pixel offset based on edge position.
    var pixel_offset = -closer_dist / edge_length + 0.5;

    // Check if the endpoint luma indicates the current pixel is on the same side.
    let is_correct = (is_closer_neg && luma_end_neg < 0.0) || (!is_closer_neg && luma_end_pos < 0.0);
    if is_correct {
        pixel_offset = 0.0;
    }

    let final_pixel_offset = max(pixel_offset, final_offset);

    // Apply the final offset.
    var final_uv = uv;
    if is_horizontal {
        final_uv.y = final_uv.y + step_length * final_pixel_offset;
    } else {
        final_uv.x = final_uv.x + step_length * final_pixel_offset;
    }

    let final_color = textureSample(source_texture, source_sampler, final_uv).xyz;

    return vec4<f32>(final_color, 1.0);
}
"#;

// ============================================================================
// Vignette + Film Grain + Chromatic Aberration shader
// ============================================================================

/// Combined vignette, film grain, and chromatic aberration post-process shader.
pub const VIGNETTE_COMBINED_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Vignette + Film Grain + Chromatic Aberration
// ============================================================================

struct PostParams {
    vignette_params: vec4<f32>,
    chromatic_params: vec4<f32>,
    grain_params: vec4<f32>,
    time_params: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return output;
}

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> params: PostParams;

// Pseudo-random hash for film grain.
fn hash13(p3: vec3<f32>) -> f32 {
    var p = fract(p3 * 0.1031);
    p = p + dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

@fragment
fn fs_vignette_combined(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let vignette_intensity = params.vignette_params.x;
    let vignette_smoothness = params.vignette_params.y;
    let vignette_roundness = params.vignette_params.z;
    let chromatic_intensity = params.chromatic_params.x;
    let chromatic_enabled = params.chromatic_params.y;
    let grain_intensity = params.grain_params.x;
    let grain_enabled = params.grain_params.y;
    let time = params.time_params.x;

    // Chromatic aberration (if enabled).
    var color: vec3<f32>;
    if chromatic_enabled > 0.5 {
        let dist_from_center = length(uv - vec2<f32>(0.5));
        let offset = (uv - vec2<f32>(0.5)) * chromatic_intensity * dist_from_center;

        let r = textureSample(source_texture, source_sampler, uv - offset).x;
        let g = textureSample(source_texture, source_sampler, uv).y;
        let b = textureSample(source_texture, source_sampler, uv + offset).z;
        color = vec3<f32>(r, g, b);
    } else {
        color = textureSample(source_texture, source_sampler, uv).xyz;
    }

    // Vignette.
    if vignette_intensity > 0.0 {
        let center_dist = uv - vec2<f32>(0.5);
        var vignette_dist: f32;
        if vignette_roundness > 0.5 {
            // Circular vignette.
            vignette_dist = length(center_dist) * 1.414; // Normalize so corners = 1.
        } else {
            // Rectangular vignette.
            let abs_dist = abs(center_dist);
            vignette_dist = max(abs_dist.x, abs_dist.y) * 2.0;
        }

        let vignette = 1.0 - smoothstep(1.0 - vignette_intensity, 1.0 - vignette_intensity + vignette_smoothness * 0.5, vignette_dist);
        color = color * vignette;
    }

    // Film grain (if enabled).
    if grain_enabled > 0.5 {
        let grain = hash13(vec3<f32>(input.position.xy, time)) - 0.5;
        color = color + vec3<f32>(grain * grain_intensity);
    }

    return vec4<f32>(color, 1.0);
}
"#;

// ============================================================================
// Post-process pipeline
// ============================================================================

/// A post-processing render target (intermediate texture for ping-pong).
pub struct PostProcessTarget {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

impl PostProcessTarget {
    /// Create a new post-process render target.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            texture,
            view,
            width,
            height,
        }
    }
}

/// The main post-processing pipeline.
///
/// Chains multiple fullscreen passes: Bloom -> Tone Map -> FXAA -> Vignette.
pub struct PostProcessPipeline {
    /// Configuration.
    pub config: PostProcessConfig,

    /// Bloom bright pass pipeline.
    bloom_bright_pipeline: wgpu::RenderPipeline,
    /// Bloom downsample pipeline.
    bloom_downsample_pipeline: wgpu::RenderPipeline,
    /// Bloom upsample pipeline.
    bloom_upsample_pipeline: wgpu::RenderPipeline,
    /// Bloom composite pipeline.
    bloom_composite_pipeline: wgpu::RenderPipeline,

    /// Tone mapping pipeline.
    tone_mapping_pipeline: wgpu::RenderPipeline,

    /// FXAA pipeline.
    fxaa_pipeline: wgpu::RenderPipeline,

    /// Vignette + combined effects pipeline.
    vignette_pipeline: wgpu::RenderPipeline,

    /// Bind group layouts.
    single_texture_bgl: wgpu::BindGroupLayout,
    bloom_composite_bgl: wgpu::BindGroupLayout,

    /// Linear sampler for post-process effects.
    linear_sampler: wgpu::Sampler,

    /// Bloom mip chain targets (downsampled).
    bloom_mip_targets: Vec<PostProcessTarget>,

    /// Ping-pong render targets for chaining effects.
    ping_target: Option<PostProcessTarget>,
    pong_target: Option<PostProcessTarget>,

    /// Bloom parameter buffers.
    bloom_params_buffer: wgpu::Buffer,
    /// FXAA parameter buffer.
    fxaa_params_buffer: wgpu::Buffer,
    /// Tone map parameter buffer.
    tonemap_params_buffer: wgpu::Buffer,
    /// Vignette parameter buffer.
    vignette_params_buffer: wgpu::Buffer,

    /// Output format.
    color_format: wgpu::TextureFormat,
    /// HDR intermediate format.
    hdr_format: wgpu::TextureFormat,

    /// Current viewport dimensions.
    width: u32,
    height: u32,
}

impl PostProcessPipeline {
    /// Create a new post-processing pipeline.
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let hdr_format = wgpu::TextureFormat::Rgba16Float;

        // --- Bind group layout: single texture + sampler + params ---
        let single_texture_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pp_single_texture_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // --- Bloom composite BGL: two textures + sampler + params ---
        let bloom_composite_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pp_bloom_composite_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let single_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pp_single_layout"),
                bind_group_layouts: &[&single_texture_bgl],
                push_constant_ranges: &[],
            });

        let composite_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pp_composite_layout"),
                bind_group_layouts: &[&bloom_composite_bgl],
                push_constant_ranges: &[],
            });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("pp_linear_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // --- Create pipelines ---
        let create_fullscreen_pipeline =
            |device: &wgpu::Device,
             shader_src: &str,
             vs_entry: &str,
             fs_entry: &str,
             layout: &wgpu::PipelineLayout,
             target_format: wgpu::TextureFormat,
             label: &str|
             -> wgpu::RenderPipeline {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(shader_src.into()),
                });

                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some(vs_entry),
                        buffers: &[],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some(fs_entry),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: target_format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                })
            };

        let bloom_bright_pipeline = create_fullscreen_pipeline(
            device,
            BLOOM_BRIGHT_PASS_WGSL,
            "vs_fullscreen",
            "fs_bloom_bright",
            &single_layout,
            hdr_format,
            "bloom_bright_pipeline",
        );

        let bloom_downsample_pipeline = create_fullscreen_pipeline(
            device,
            BLOOM_DOWNSAMPLE_WGSL,
            "vs_fullscreen",
            "fs_bloom_downsample",
            &single_layout,
            hdr_format,
            "bloom_downsample_pipeline",
        );

        let bloom_upsample_pipeline = create_fullscreen_pipeline(
            device,
            BLOOM_UPSAMPLE_WGSL,
            "vs_fullscreen",
            "fs_bloom_upsample",
            &single_layout,
            hdr_format,
            "bloom_upsample_pipeline",
        );

        let bloom_composite_pipeline = create_fullscreen_pipeline(
            device,
            BLOOM_COMPOSITE_WGSL,
            "vs_fullscreen",
            "fs_bloom_composite",
            &composite_layout,
            hdr_format,
            "bloom_composite_pipeline",
        );

        let tone_mapping_pipeline = create_fullscreen_pipeline(
            device,
            TONE_MAPPING_WGSL,
            "vs_fullscreen",
            "fs_tone_mapping",
            &single_layout,
            color_format,
            "tone_mapping_pipeline",
        );

        let fxaa_pipeline = create_fullscreen_pipeline(
            device,
            FXAA_SHADER_WGSL,
            "vs_fullscreen",
            "fs_fxaa",
            &single_layout,
            color_format,
            "fxaa_pipeline",
        );

        let vignette_pipeline = create_fullscreen_pipeline(
            device,
            VIGNETTE_COMBINED_WGSL,
            "vs_fullscreen",
            "fs_vignette_combined",
            &single_layout,
            color_format,
            "vignette_pipeline",
        );

        // --- Create parameter buffers ---
        let bloom_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pp_bloom_params"),
            size: std::mem::size_of::<BloomParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let fxaa_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pp_fxaa_params"),
            size: std::mem::size_of::<FxaaParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tonemap_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pp_tonemap_params"),
            size: std::mem::size_of::<ToneMappingParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vignette_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pp_vignette_params"),
            size: 64, // PostParams struct in shader.
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Create bloom mip chain ---
        let bloom_mip_targets = Self::create_bloom_mip_chain(device, width, height, hdr_format, 5);

        // --- Create ping-pong targets ---
        let ping_target = Some(PostProcessTarget::new(
            device, width, height, hdr_format, "pp_ping",
        ));
        let pong_target = Some(PostProcessTarget::new(
            device, width, height, color_format, "pp_pong",
        ));

        Self {
            config: PostProcessConfig::default(),
            bloom_bright_pipeline,
            bloom_downsample_pipeline,
            bloom_upsample_pipeline,
            bloom_composite_pipeline,
            tone_mapping_pipeline,
            fxaa_pipeline,
            vignette_pipeline,
            single_texture_bgl,
            bloom_composite_bgl,
            linear_sampler,
            bloom_mip_targets,
            ping_target,
            pong_target,
            bloom_params_buffer,
            fxaa_params_buffer,
            tonemap_params_buffer,
            vignette_params_buffer,
            color_format,
            hdr_format,
            width,
            height,
        }
    }

    /// Create the bloom mip chain targets.
    fn create_bloom_mip_chain(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        levels: usize,
    ) -> Vec<PostProcessTarget> {
        let mut targets = Vec::with_capacity(levels);
        let mut w = width / 2;
        let mut h = height / 2;

        for i in 0..levels {
            w = w.max(1);
            h = h.max(1);
            targets.push(PostProcessTarget::new(
                device,
                w,
                h,
                format,
                &format!("bloom_mip_{}", i),
            ));
            w /= 2;
            h /= 2;
        }

        targets
    }

    /// Resize the post-processing pipeline for a new viewport size.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.width = width;
        self.height = height;

        self.bloom_mip_targets = Self::create_bloom_mip_chain(
            device,
            width,
            height,
            self.hdr_format,
            self.config.bloom_levels,
        );

        self.ping_target = Some(PostProcessTarget::new(
            device, width, height, self.hdr_format, "pp_ping",
        ));
        self.pong_target = Some(PostProcessTarget::new(
            device, width, height, self.color_format, "pp_pong",
        ));
    }

    /// Update the config and upload uniform parameters.
    pub fn update_params(&self, queue: &wgpu::Queue, config: &PostProcessConfig) {
        // Bloom params.
        let bloom_uniform = BloomParamsUniform {
            params: [
                config.bloom_threshold,
                config.bloom_soft_threshold,
                config.bloom_intensity,
                0.0,
            ],
            texel_info: [
                1.0 / self.width as f32,
                1.0 / self.height as f32,
                0.0,
                config.bloom_levels as f32,
            ],
        };
        queue.write_buffer(&self.bloom_params_buffer, 0, bytemuck::bytes_of(&bloom_uniform));

        // FXAA params.
        let fxaa_uniform = FxaaParamsUniform {
            params: [
                config.fxaa_edge_threshold,
                config.fxaa_edge_threshold_min,
                config.fxaa_subpixel_quality,
                0.0,
            ],
            texel_info: [
                1.0 / self.width as f32,
                1.0 / self.height as f32,
                0.0,
                0.0,
            ],
        };
        queue.write_buffer(&self.fxaa_params_buffer, 0, bytemuck::bytes_of(&fxaa_uniform));

        // Tone mapping params.
        let tonemap_uniform = ToneMappingParamsUniform {
            params: [config.exposure, 2.2, 0.0, 0.0],
        };
        queue.write_buffer(
            &self.tonemap_params_buffer,
            0,
            bytemuck::bytes_of(&tonemap_uniform),
        );

        // Vignette params.
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct VignetteShaderParams {
            vignette: [f32; 4],
            chromatic: [f32; 4],
            grain: [f32; 4],
            time: [f32; 4],
        }

        let vignette_uniform = VignetteShaderParams {
            vignette: [
                config.vignette_intensity,
                config.vignette_smoothness,
                config.vignette_roundness,
                0.0,
            ],
            chromatic: [
                config.chromatic_aberration_intensity,
                if config.chromatic_aberration_enabled { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
            grain: [
                config.film_grain_intensity,
                if config.film_grain_enabled { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
            time: [0.0, 0.016, 0.0, 0.0],
        };
        queue.write_buffer(
            &self.vignette_params_buffer,
            0,
            bytemuck::bytes_of(&vignette_uniform),
        );
    }

    /// Run a single fullscreen pass.
    fn run_fullscreen_pass(
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        bind_group: &wgpu::BindGroup,
        target_view: &wgpu::TextureView,
        label: &str,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Render the complete post-processing chain.
    ///
    /// `scene_view` is the HDR scene render target.
    /// `output_view` is the final output (swapchain).
    pub fn render_post_process(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
    ) {
        let ping = self.ping_target.as_ref().unwrap();
        let pong = self.pong_target.as_ref().unwrap();

        // --- Bloom ---
        if self.config.bloom_enabled && !self.bloom_mip_targets.is_empty() {
            // Bright pass: scene -> bloom_mip[0].
            let bright_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom_bright_bg"),
                layout: &self.single_texture_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(scene_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.bloom_params_buffer.as_entire_binding(),
                    },
                ],
            });

            Self::run_fullscreen_pass(
                encoder,
                &self.bloom_bright_pipeline,
                &bright_bg,
                &self.bloom_mip_targets[0].view,
                "bloom_bright_pass",
            );

            // Downsample chain.
            for i in 1..self.bloom_mip_targets.len() {
                let source_view = &self.bloom_mip_targets[i - 1].view;
                let down_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("bloom_down_{}", i)),
                    layout: &self.single_texture_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(source_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.bloom_params_buffer.as_entire_binding(),
                        },
                    ],
                });

                Self::run_fullscreen_pass(
                    encoder,
                    &self.bloom_downsample_pipeline,
                    &down_bg,
                    &self.bloom_mip_targets[i].view,
                    &format!("bloom_downsample_{}", i),
                );
            }

            // Upsample chain.
            for i in (0..self.bloom_mip_targets.len() - 1).rev() {
                let source_view = &self.bloom_mip_targets[i + 1].view;
                let up_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("bloom_up_{}", i)),
                    layout: &self.single_texture_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(source_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.bloom_params_buffer.as_entire_binding(),
                        },
                    ],
                });

                Self::run_fullscreen_pass(
                    encoder,
                    &self.bloom_upsample_pipeline,
                    &up_bg,
                    &self.bloom_mip_targets[i].view,
                    &format!("bloom_upsample_{}", i),
                );
            }

            // Composite: scene + bloom -> ping.
            let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom_composite_bg"),
                layout: &self.bloom_composite_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(scene_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &self.bloom_mip_targets[0].view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.bloom_params_buffer.as_entire_binding(),
                    },
                ],
            });

            Self::run_fullscreen_pass(
                encoder,
                &self.bloom_composite_pipeline,
                &composite_bg,
                &ping.view,
                "bloom_composite",
            );
        }

        // --- Tone mapping: ping -> pong ---
        if self.config.tone_mapping_enabled {
            let source = if self.config.bloom_enabled {
                &ping.view
            } else {
                scene_view
            };

            let tonemap_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tonemap_bg"),
                layout: &self.single_texture_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.tonemap_params_buffer.as_entire_binding(),
                    },
                ],
            });

            Self::run_fullscreen_pass(
                encoder,
                &self.tone_mapping_pipeline,
                &tonemap_bg,
                &pong.view,
                "tone_mapping",
            );
        }

        // --- FXAA: pong -> output ---
        if self.config.fxaa_enabled {
            let fxaa_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fxaa_bg"),
                layout: &self.single_texture_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&pong.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.fxaa_params_buffer.as_entire_binding(),
                    },
                ],
            });

            Self::run_fullscreen_pass(
                encoder,
                &self.fxaa_pipeline,
                &fxaa_bg,
                output_view,
                "fxaa",
            );
        }

        // --- Vignette: output -> output (final pass, reads from pong if FXAA disabled) ---
        if self.config.vignette_enabled {
            let source = if self.config.fxaa_enabled {
                output_view
            } else {
                &pong.view
            };

            let vignette_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vignette_bg"),
                layout: &self.single_texture_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.vignette_params_buffer.as_entire_binding(),
                    },
                ],
            });

            Self::run_fullscreen_pass(
                encoder,
                &self.vignette_pipeline,
                &vignette_bg,
                output_view,
                "vignette",
            );
        }
    }

    /// Get the current viewport dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}
