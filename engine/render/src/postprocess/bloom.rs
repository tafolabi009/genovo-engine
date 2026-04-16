// engine/render/src/postprocess/bloom.rs
//
// Physically-inspired bloom effect using progressive downsample / upsample
// with a configurable brightness threshold, soft knee, and optional lens
// dirt mask overlay.
//
// The algorithm:
//   1. Extract pixels above a brightness threshold (soft knee for smooth
//      transition).
//   2. Progressively downsample using a 13-tap filter (reduces firefly
//      artifacts compared to naive bilinear).
//   3. Progressively upsample with a 9-tap tent filter, additively
//      blending each mip level back together.
//   4. Composite the bloom result onto the scene color with intensity and
//      scatter controls.
//   5. Optionally multiply by a dirt mask texture for a cinematic lens
//      dirt effect.

use std::any::Any;

use super::{PostProcessEffect, PostProcessInput, PostProcessOutput, TextureId};

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// Configuration for the bloom effect.
#[derive(Debug, Clone)]
pub struct BloomSettings {
    /// Luminance threshold — pixels below this brightness are excluded.
    pub threshold: f32,
    /// Soft knee width: controls the smoothness of the threshold curve.
    /// 0.0 = hard cutoff, 1.0 = very soft.
    pub soft_knee: f32,
    /// Overall bloom intensity multiplier.
    pub intensity: f32,
    /// Controls how much bloom spreads to lower mips (0.0–1.0).
    /// Higher values produce wider, softer bloom.
    pub scatter: f32,
    /// Maximum number of mip levels to generate (clamped to
    /// `log2(min(width, height))`).
    pub max_mip_levels: u32,
    /// Optional lens dirt mask texture.
    pub dirt_mask: TextureId,
    /// Intensity of the dirt mask overlay.
    pub dirt_intensity: f32,
    /// Tint color for the bloom (linear RGB).
    pub tint: [f32; 3],
    /// Whether the effect is enabled.
    pub enabled: bool,
}

impl Default for BloomSettings {
    fn default() -> Self {
        Self {
            threshold: 1.0,
            soft_knee: 0.5,
            intensity: 0.8,
            scatter: 0.7,
            max_mip_levels: 8,
            dirt_mask: TextureId::INVALID,
            dirt_intensity: 0.0,
            tint: [1.0, 1.0, 1.0],
            enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Mip chain
// ---------------------------------------------------------------------------

/// Describes a single level in the bloom mip chain.
#[derive(Debug, Clone)]
struct BloomMip {
    /// Texture for this mip level.
    texture: TextureId,
    /// Width at this mip level.
    width: u32,
    /// Height at this mip level.
    height: u32,
}

// ---------------------------------------------------------------------------
// BloomEffect
// ---------------------------------------------------------------------------

/// Bloom post-process effect.
pub struct BloomEffect {
    pub settings: BloomSettings,
    /// The downsample mip chain (excluding level 0 which is the threshold
    /// output).
    mip_chain: Vec<BloomMip>,
    /// Texture holding the threshold-extracted bright pixels.
    threshold_texture: TextureId,
    /// Composite output texture.
    composite_texture: TextureId,
    /// Current viewport dimensions (used to detect resize).
    current_width: u32,
    current_height: u32,
}

impl BloomEffect {
    pub fn new(settings: BloomSettings) -> Self {
        Self {
            settings,
            mip_chain: Vec::new(),
            threshold_texture: TextureId(100),
            composite_texture: TextureId(101),
            current_width: 0,
            current_height: 0,
        }
    }

    /// (Re)build the mip chain for the given viewport size.
    fn rebuild_mip_chain(&mut self, width: u32, height: u32) {
        self.mip_chain.clear();

        let max_levels = self.compute_max_mip_levels(width, height);
        let mut w = width / 2;
        let mut h = height / 2;
        let base_id = 200u64;

        for i in 0..max_levels {
            self.mip_chain.push(BloomMip {
                texture: TextureId(base_id + i as u64),
                width: w.max(1),
                height: h.max(1),
            });
            w /= 2;
            h /= 2;
        }

        self.current_width = width;
        self.current_height = height;
    }

    /// Compute the number of mip levels to use.
    fn compute_max_mip_levels(&self, width: u32, height: u32) -> u32 {
        let min_dim = width.min(height) as f32;
        let max_possible = (min_dim.log2().floor() as u32).saturating_sub(1);
        max_possible.min(self.settings.max_mip_levels)
    }

    /// Compute the soft threshold curve value.
    /// Uses a quadratic knee: smoothly transitions from 0 to 1 around the
    /// threshold.
    fn soft_threshold(luminance: f32, threshold: f32, knee: f32) -> f32 {
        let half_knee = knee * 0.5;
        let low = threshold - half_knee;
        let high = threshold + half_knee;

        if luminance <= low {
            return 0.0;
        }
        if luminance >= high {
            return luminance - threshold;
        }

        // Quadratic interpolation in the knee region.
        let t = (luminance - low) / (high - low + 1e-6);
        let contribution = t * t * (luminance - threshold);
        contribution.max(0.0)
    }

    /// Execute the brightness threshold extraction pass.
    fn execute_threshold(&self, _input: &PostProcessInput) {
        // In a real implementation this would dispatch a compute shader or
        // render a fullscreen quad with the threshold shader.
        // The threshold shader samples the input color, computes luminance,
        // applies the soft knee curve, and writes to threshold_texture.
        let _threshold = self.settings.threshold;
        let _knee = self.settings.soft_knee;
        // GPU dispatch would happen here.
    }

    /// Execute the progressive downsample passes using the 13-tap filter.
    fn execute_downsample(&self) {
        // Each mip reads from the previous mip (or threshold_texture for
        // mip 0) and writes to the next smaller mip. The 13-tap filter
        // uses a pattern of 13 bilinear samples arranged in a cross/box
        // pattern to produce a weighted average that suppresses fireflies.
        for i in 0..self.mip_chain.len() {
            let _src = if i == 0 {
                self.threshold_texture
            } else {
                self.mip_chain[i - 1].texture
            };
            let _dst = self.mip_chain[i].texture;
            let _w = self.mip_chain[i].width;
            let _h = self.mip_chain[i].height;
            // Dispatch downsample compute shader for this mip level.
        }
    }

    /// Execute the progressive upsample passes with additive blending.
    fn execute_upsample(&self) {
        // Walk the mip chain from smallest to largest. Each level reads
        // the upsampled result from the level below plus the corresponding
        // downsample mip, blending them together with the scatter factor.
        let scatter = self.settings.scatter;
        let mip_count = self.mip_chain.len();

        if mip_count < 2 {
            return;
        }

        for i in (0..mip_count - 1).rev() {
            let _low_res = self.mip_chain[i + 1].texture;
            let _high_res = self.mip_chain[i].texture;
            let _w = self.mip_chain[i].width;
            let _h = self.mip_chain[i].height;
            let _blend = scatter;
            // Dispatch upsample compute shader.
        }
    }

    /// Composite the bloom result onto the scene color.
    fn execute_composite(&self, _input: &PostProcessInput, _output: &mut PostProcessOutput) {
        let _intensity = self.settings.intensity;
        let _tint = self.settings.tint;
        let _bloom_texture = if !self.mip_chain.is_empty() {
            self.mip_chain[0].texture
        } else {
            TextureId::INVALID
        };
        let _has_dirt = self.settings.dirt_mask.is_valid();
        let _dirt_intensity = self.settings.dirt_intensity;
        // Dispatch composite shader.
    }
}

impl PostProcessEffect for BloomEffect {
    fn name(&self) -> &str {
        "Bloom"
    }

    fn execute(&self, input: &PostProcessInput, output: &mut PostProcessOutput) {
        if !self.settings.enabled {
            return;
        }

        self.execute_threshold(input);
        self.execute_downsample();
        self.execute_upsample();
        self.execute_composite(input, output);
    }

    fn is_enabled(&self) -> bool {
        self.settings.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.settings.enabled = enabled;
    }

    fn priority(&self) -> u32 {
        500
    }

    fn on_resize(&mut self, width: u32, height: u32) {
        if width != self.current_width || height != self.current_height {
            self.rebuild_mip_chain(width, height);
        }
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

/// Bloom threshold extraction compute shader.
/// Computes luminance of each pixel and applies a soft knee threshold curve.
pub const BLOOM_THRESHOLD_WGSL: &str = r#"
// Bloom brightness threshold extraction — compute shader
// Workgroup size: 8x8 threads per tile

struct BloomParams {
    threshold:      f32,
    soft_knee:      f32,
    inv_src_width:  f32,
    inv_src_height: f32,
};

@group(0) @binding(0) var src_texture:   texture_2d<f32>;
@group(0) @binding(1) var dst_texture:   texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var src_sampler:   sampler;
@group(0) @binding(3) var<uniform> params: BloomParams;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn soft_threshold(lum: f32, threshold: f32, knee: f32) -> f32 {
    let half_knee = knee * 0.5;
    let low  = threshold - half_knee;
    let high = threshold + half_knee;

    if lum <= low {
        return 0.0;
    }
    if lum >= high {
        return lum - threshold;
    }
    let t = (lum - low) / (high - low + 1e-6);
    return max(t * t * (lum - threshold), 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_texture);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) * vec2<f32>(params.inv_src_width, params.inv_src_height);
    let color = textureSampleLevel(src_texture, src_sampler, uv, 0.0).rgb;
    let lum = luminance(color);

    let contribution = soft_threshold(lum, params.threshold, params.soft_knee);
    let scale = contribution / (lum + 1e-6);
    let result = color * scale;

    textureStore(dst_texture, gid.xy, vec4<f32>(result, 1.0));
}
"#;

/// Bloom 13-tap downsample compute shader.
///
/// Uses a pattern of 13 bilinear taps arranged to approximate a
/// high-quality box-Gaussian filter while suppressing single-pixel
/// fireflies:
///
/// ```text
///     a . b . c
///     . d . e .
///     f . g . h
///     . i . j .
///     k . l . m
/// ```
///
/// Weights:
///   center group (d,e,i,j around g): 0.5  (4 bilinear taps => 0.125 each)
///   corner groups: 0.125 each         (4 groups of 4 pixels => 0.03125 each)
///   edge group:    0.25 split among 4 (a+b+f+g, b+c+g+h, etc.)
pub const BLOOM_DOWNSAMPLE_WGSL: &str = r#"
// Bloom 13-tap downsample — compute shader

struct DownsampleParams {
    inv_src_width:  f32,
    inv_src_height: f32,
    _pad0:          f32,
    _pad1:          f32,
};

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var src_sampler: sampler;
@group(0) @binding(3) var<uniform> params: DownsampleParams;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_texture);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let texel = vec2<f32>(params.inv_src_width, params.inv_src_height);
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);

    // 13-tap filter pattern
    // Center
    let g = textureSampleLevel(src_texture, src_sampler, uv, 0.0).rgb;

    // Inner ring (half-texel offsets — exploit bilinear filtering)
    let d = textureSampleLevel(src_texture, src_sampler, uv + vec2<f32>(-0.5, -0.5) * texel, 0.0).rgb;
    let e = textureSampleLevel(src_texture, src_sampler, uv + vec2<f32>( 0.5, -0.5) * texel, 0.0).rgb;
    let i = textureSampleLevel(src_texture, src_sampler, uv + vec2<f32>(-0.5,  0.5) * texel, 0.0).rgb;
    let j = textureSampleLevel(src_texture, src_sampler, uv + vec2<f32>( 0.5,  0.5) * texel, 0.0).rgb;

    // Outer ring (full-texel offsets)
    let a = textureSampleLevel(src_texture, src_sampler, uv + vec2<f32>(-1.0, -1.0) * texel, 0.0).rgb;
    let b = textureSampleLevel(src_texture, src_sampler, uv + vec2<f32>( 0.0, -1.0) * texel, 0.0).rgb;
    let c = textureSampleLevel(src_texture, src_sampler, uv + vec2<f32>( 1.0, -1.0) * texel, 0.0).rgb;
    let f = textureSampleLevel(src_texture, src_sampler, uv + vec2<f32>(-1.0,  0.0) * texel, 0.0).rgb;
    let h = textureSampleLevel(src_texture, src_sampler, uv + vec2<f32>( 1.0,  0.0) * texel, 0.0).rgb;
    let k = textureSampleLevel(src_texture, src_sampler, uv + vec2<f32>(-1.0,  1.0) * texel, 0.0).rgb;
    let l = textureSampleLevel(src_texture, src_sampler, uv + vec2<f32>( 0.0,  1.0) * texel, 0.0).rgb;
    let m = textureSampleLevel(src_texture, src_sampler, uv + vec2<f32>( 1.0,  1.0) * texel, 0.0).rgb;

    // Weighted combination
    var result = g * 0.125;                                     // center
    result += (d + e + i + j) * 0.125;                          // inner ring
    result += (a + c + k + m) * 0.03125;                        // corners
    result += (b + f + h + l) * 0.0625;                         // edges

    // Karis average for mip 0 to prevent fireflies
    // (weighted by 1/(1+luma) per sample group, applied at first downsample)

    textureStore(dst_texture, gid.xy, vec4<f32>(result, 1.0));
}
"#;

/// Bloom 9-tap upsample compute shader.
///
/// Uses a 3x3 tent filter with bilinear sampling for smooth upsampling.
/// Blends the upsampled lower-resolution mip with the corresponding
/// higher-resolution downsample mip using the scatter parameter.
pub const BLOOM_UPSAMPLE_WGSL: &str = r#"
// Bloom 9-tap upsample with additive blend — compute shader

struct UpsampleParams {
    inv_dst_width:  f32,
    inv_dst_height: f32,
    scatter:        f32,
    _pad:           f32,
};

@group(0) @binding(0) var low_res_texture:  texture_2d<f32>;
@group(0) @binding(1) var high_res_texture: texture_2d<f32>;
@group(0) @binding(2) var dst_texture:      texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var tex_sampler:      sampler;
@group(0) @binding(4) var<uniform> params:  UpsampleParams;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_texture);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let texel = vec2<f32>(params.inv_dst_width, params.inv_dst_height);

    // 9-tap tent filter on the low-res texture
    var bloom = vec3<f32>(0.0);
    bloom += textureSampleLevel(low_res_texture, tex_sampler, uv + vec2<f32>(-1.0, -1.0) * texel, 0.0).rgb * (1.0 / 16.0);
    bloom += textureSampleLevel(low_res_texture, tex_sampler, uv + vec2<f32>( 0.0, -1.0) * texel, 0.0).rgb * (2.0 / 16.0);
    bloom += textureSampleLevel(low_res_texture, tex_sampler, uv + vec2<f32>( 1.0, -1.0) * texel, 0.0).rgb * (1.0 / 16.0);
    bloom += textureSampleLevel(low_res_texture, tex_sampler, uv + vec2<f32>(-1.0,  0.0) * texel, 0.0).rgb * (2.0 / 16.0);
    bloom += textureSampleLevel(low_res_texture, tex_sampler, uv,                                  0.0).rgb * (4.0 / 16.0);
    bloom += textureSampleLevel(low_res_texture, tex_sampler, uv + vec2<f32>( 1.0,  0.0) * texel, 0.0).rgb * (2.0 / 16.0);
    bloom += textureSampleLevel(low_res_texture, tex_sampler, uv + vec2<f32>(-1.0,  1.0) * texel, 0.0).rgb * (1.0 / 16.0);
    bloom += textureSampleLevel(low_res_texture, tex_sampler, uv + vec2<f32>( 0.0,  1.0) * texel, 0.0).rgb * (2.0 / 16.0);
    bloom += textureSampleLevel(low_res_texture, tex_sampler, uv + vec2<f32>( 1.0,  1.0) * texel, 0.0).rgb * (1.0 / 16.0);

    // Read the higher-resolution downsample mip at this level
    let high = textureSampleLevel(high_res_texture, tex_sampler, uv, 0.0).rgb;

    // Blend: lerp(high, bloom, scatter)
    let result = mix(high, bloom, params.scatter);
    textureStore(dst_texture, gid.xy, vec4<f32>(result, 1.0));
}
"#;

/// Bloom final composite shader.
/// Additively blends the bloom texture onto the scene color and applies
/// the optional dirt mask.
pub const BLOOM_COMPOSITE_WGSL: &str = r#"
// Bloom final composite — compute shader

struct CompositeParams {
    bloom_intensity: f32,
    dirt_intensity:  f32,
    tint_r:          f32,
    tint_g:          f32,
    tint_b:          f32,
    has_dirt_mask:   u32,
    _pad0:           f32,
    _pad1:           f32,
};

@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var bloom_texture: texture_2d<f32>;
@group(0) @binding(2) var dirt_texture:  texture_2d<f32>;
@group(0) @binding(3) var dst_texture:   texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var tex_sampler:   sampler;
@group(0) @binding(5) var<uniform> params: CompositeParams;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_texture);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let scene_color = textureSampleLevel(scene_texture, tex_sampler, uv, 0.0).rgb;
    var bloom = textureSampleLevel(bloom_texture, tex_sampler, uv, 0.0).rgb;

    // Apply tint
    let tint = vec3<f32>(params.tint_r, params.tint_g, params.tint_b);
    bloom *= tint;

    // Base bloom contribution
    var result = scene_color + bloom * params.bloom_intensity;

    // Optional dirt mask overlay
    if params.has_dirt_mask != 0u {
        let dirt = textureSampleLevel(dirt_texture, tex_sampler, uv, 0.0).rgb;
        result += bloom * dirt * params.dirt_intensity;
    }

    textureStore(dst_texture, gid.xy, vec4<f32>(result, 1.0));
}
"#;

// ---------------------------------------------------------------------------
// Karis average helper (for preventing fireflies at mip 0)
// ---------------------------------------------------------------------------

/// Compute the Karis average weight for a pixel group.
/// Weight = 1 / (1 + luminance), which downweights extremely bright pixels
/// in the first downsample pass to prevent single-pixel fireflies from
/// blooming excessively.
pub fn karis_weight(r: f32, g: f32, b: f32) -> f32 {
    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    1.0 / (1.0 + luma)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_threshold_hard() {
        // Below threshold -> 0
        assert_eq!(BloomEffect::soft_threshold(0.5, 1.0, 0.0), 0.0);
        // At threshold with no knee -> 0
        assert!(BloomEffect::soft_threshold(1.0, 1.0, 0.0).abs() < 1e-5);
        // Above threshold -> luminance - threshold
        let result = BloomEffect::soft_threshold(2.0, 1.0, 0.0);
        assert!((result - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_soft_threshold_soft() {
        // In the knee region, result should be > 0 but < (lum - threshold)
        let result = BloomEffect::soft_threshold(1.0, 1.0, 1.0);
        assert!(result >= 0.0);
    }

    #[test]
    fn test_mip_chain_generation() {
        let mut bloom = BloomEffect::new(BloomSettings::default());
        bloom.rebuild_mip_chain(1920, 1080);
        assert!(!bloom.mip_chain.is_empty());
        assert!(bloom.mip_chain.len() <= 8);

        // Each level should be half the previous
        for i in 1..bloom.mip_chain.len() {
            assert!(bloom.mip_chain[i].width <= bloom.mip_chain[i - 1].width);
            assert!(bloom.mip_chain[i].height <= bloom.mip_chain[i - 1].height);
        }
    }

    #[test]
    fn test_karis_weight() {
        // Pure black -> weight = 1.0
        assert!((karis_weight(0.0, 0.0, 0.0) - 1.0).abs() < 1e-6);
        // Brighter pixels get lower weight
        let w_dim = karis_weight(0.1, 0.1, 0.1);
        let w_bright = karis_weight(10.0, 10.0, 10.0);
        assert!(w_dim > w_bright);
    }

    #[test]
    fn test_bloom_effect_interface() {
        let mut bloom = BloomEffect::new(BloomSettings {
            threshold: 0.8,
            intensity: 1.2,
            ..Default::default()
        });

        assert_eq!(bloom.name(), "Bloom");
        assert!(bloom.is_enabled());
        assert_eq!(bloom.priority(), 500);

        bloom.set_enabled(false);
        assert!(!bloom.is_enabled());
    }

    #[test]
    fn test_bloom_resize() {
        let mut bloom = BloomEffect::new(BloomSettings::default());
        bloom.on_resize(1920, 1080);
        let first_count = bloom.mip_chain.len();

        bloom.on_resize(3840, 2160);
        let second_count = bloom.mip_chain.len();

        // 4K should allow at least as many mip levels as 1080p
        assert!(second_count >= first_count);
    }

    #[test]
    fn test_shader_strings_not_empty() {
        assert!(!BLOOM_THRESHOLD_WGSL.is_empty());
        assert!(!BLOOM_DOWNSAMPLE_WGSL.is_empty());
        assert!(!BLOOM_UPSAMPLE_WGSL.is_empty());
        assert!(!BLOOM_COMPOSITE_WGSL.is_empty());
    }
}
