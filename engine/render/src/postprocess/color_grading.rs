// engine/render/src/postprocess/color_grading.rs
//
// Color grading, LUT-based color transforms, and final image adjustments.
//
// This module provides:
//   - 3D Look-Up Table (LUT) color grading
//   - Lift/Gamma/Gain color wheels
//   - HSL adjustments (per luminance range)
//   - White balance / color temperature
//   - Split toning (shadow/highlight tints)
//   - Vignette
//   - Film grain
//   - Chromatic aberration

use std::any::Any;

use super::{PostProcessEffect, PostProcessInput, PostProcessOutput, TextureId};

// ---------------------------------------------------------------------------
// Settings structs
// ---------------------------------------------------------------------------

/// Split toning settings: separate color tints for shadows and highlights.
#[derive(Debug, Clone)]
pub struct SplitToningSettings {
    /// Tint color for shadows (linear RGB).
    pub shadow_tint: [f32; 3],
    /// Tint color for highlights (linear RGB).
    pub highlight_tint: [f32; 3],
    /// Balance between shadow and highlight tinting (-1..1).
    /// Negative = more shadow, positive = more highlight.
    pub balance: f32,
    /// Overall intensity of the split toning effect.
    pub intensity: f32,
}

impl Default for SplitToningSettings {
    fn default() -> Self {
        Self {
            shadow_tint: [0.5, 0.4, 0.3],
            highlight_tint: [1.0, 0.95, 0.8],
            balance: 0.0,
            intensity: 0.0,
        }
    }
}

/// Vignette settings.
#[derive(Debug, Clone)]
pub struct VignetteSettings {
    /// Intensity of the vignette darkening (0 = none, 1 = heavy).
    pub intensity: f32,
    /// Smoothness of the vignette transition.
    pub smoothness: f32,
    /// Roundness (0 = oval matching aspect ratio, 1 = perfect circle).
    pub roundness: f32,
    /// Center of the vignette (normalized, default [0.5, 0.5]).
    pub center: [f32; 2],
    /// Color of the vignette (default: black).
    pub color: [f32; 3],
}

impl Default for VignetteSettings {
    fn default() -> Self {
        Self {
            intensity: 0.3,
            smoothness: 0.5,
            roundness: 1.0,
            center: [0.5, 0.5],
            color: [0.0, 0.0, 0.0],
        }
    }
}

/// Film grain settings.
#[derive(Debug, Clone)]
pub struct FilmGrainSettings {
    /// Intensity of the grain (0 = none, 1 = heavy).
    pub intensity: f32,
    /// Size of the grain (1.0 = normal, 2.0 = coarser).
    pub size: f32,
    /// Whether grain is luminance-only (monochrome) or colored.
    pub luminance_only: bool,
    /// Response curve: how grain reacts to image brightness.
    /// 0 = uniform, 1 = more grain in midtones.
    pub response: f32,
}

impl Default for FilmGrainSettings {
    fn default() -> Self {
        Self {
            intensity: 0.0,
            size: 1.0,
            luminance_only: true,
            response: 0.8,
        }
    }
}

/// Chromatic aberration settings.
#[derive(Debug, Clone)]
pub struct ChromaticAberrationSettings {
    /// Intensity of the aberration (pixels of offset at the edge).
    pub intensity: f32,
    /// Center of the aberration (normalized UV, default [0.5, 0.5]).
    pub center: [f32; 2],
    /// Whether to use a more accurate 3-channel separation.
    pub three_channel: bool,
}

impl Default for ChromaticAberrationSettings {
    fn default() -> Self {
        Self {
            intensity: 0.0,
            center: [0.5, 0.5],
            three_channel: true,
        }
    }
}

/// Full color grading settings bundle.
#[derive(Debug, Clone)]
pub struct ColorGradingSettings {
    // -- Lift/Gamma/Gain --
    /// Lift: adds to the dark areas. Neutral = [0, 0, 0].
    pub lift: [f32; 3],
    /// Gamma: multiplies midtones. Neutral = [1, 1, 1].
    pub gamma: [f32; 3],
    /// Gain: multiplies highlights. Neutral = [1, 1, 1].
    pub gain: [f32; 3],

    // -- Global adjustments --
    /// Saturation multiplier (1.0 = unchanged).
    pub saturation: f32,
    /// Contrast multiplier (1.0 = unchanged, >1 = more contrast).
    pub contrast: f32,
    /// Brightness offset (0.0 = unchanged).
    pub brightness: f32,

    // -- White balance --
    /// Color temperature offset in Kelvin (6500 = daylight neutral).
    /// Lower = cooler (blue), higher = warmer (yellow).
    pub temperature: f32,
    /// Tint offset (-100..+100). Green vs. magenta.
    pub tint: f32,

    // -- HSL adjustments --
    /// Per-range hue shift (shadows, midtones, highlights) in degrees.
    pub hue_shift: [f32; 3],
    /// Per-range saturation multiplier.
    pub hsl_saturation: [f32; 3],
    /// Per-range lightness offset.
    pub hsl_lightness: [f32; 3],

    // -- Sub-effects --
    pub split_toning: SplitToningSettings,
    pub vignette: VignetteSettings,
    pub film_grain: FilmGrainSettings,
    pub chromatic_aberration: ChromaticAberrationSettings,

    // -- 3D LUT --
    /// External 3D LUT texture (INVALID if not used).
    pub lut_texture: TextureId,
    /// LUT size (typically 16, 32, or 64).
    pub lut_size: u32,
    /// LUT contribution (0 = original, 1 = fully LUT-graded).
    pub lut_contribution: f32,

    /// Whether the effect is enabled.
    pub enabled: bool,
}

impl Default for ColorGradingSettings {
    fn default() -> Self {
        Self {
            lift: [0.0; 3],
            gamma: [1.0; 3],
            gain: [1.0; 3],
            saturation: 1.0,
            contrast: 1.0,
            brightness: 0.0,
            temperature: 6500.0,
            tint: 0.0,
            hue_shift: [0.0; 3],
            hsl_saturation: [1.0; 3],
            hsl_lightness: [0.0; 3],
            split_toning: SplitToningSettings::default(),
            vignette: VignetteSettings::default(),
            film_grain: FilmGrainSettings::default(),
            chromatic_aberration: ChromaticAberrationSettings::default(),
            lut_texture: TextureId::INVALID,
            lut_size: 32,
            lut_contribution: 1.0,
            enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Color math
// ---------------------------------------------------------------------------

/// Apply Lift/Gamma/Gain to a linear color.
///
/// Formula (ASC CDL approximation):
///   output = gain * (lift * (1 - color) + color) ^ (1/gamma)
pub fn apply_lift_gamma_gain(
    color: [f32; 3],
    lift: [f32; 3],
    gamma: [f32; 3],
    gain: [f32; 3],
) -> [f32; 3] {
    let mut result = [0.0f32; 3];
    for i in 0..3 {
        let lifted = lift[i] * (1.0 - color[i]) + color[i];
        let inv_gamma = if gamma[i].abs() > 1e-6 {
            1.0 / gamma[i]
        } else {
            1.0
        };
        result[i] = gain[i] * lifted.max(0.0).powf(inv_gamma);
    }
    result
}

/// Adjust saturation of a linear color.
///
/// `saturation` = 1.0 means no change. 0.0 = fully desaturated.
pub fn adjust_saturation(color: [f32; 3], saturation: f32) -> [f32; 3] {
    let luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];
    [
        luminance + (color[0] - luminance) * saturation,
        luminance + (color[1] - luminance) * saturation,
        luminance + (color[2] - luminance) * saturation,
    ]
}

/// Adjust contrast around a midpoint (typically 0.18 for linear, 0.5 for
/// gamma-encoded).
pub fn adjust_contrast(color: [f32; 3], contrast: f32, midpoint: f32) -> [f32; 3] {
    [
        (color[0] - midpoint) * contrast + midpoint,
        (color[1] - midpoint) * contrast + midpoint,
        (color[2] - midpoint) * contrast + midpoint,
    ]
}

/// Convert color temperature in Kelvin to an RGB multiplier.
///
/// Uses the Tanner Helland approximation.
pub fn temperature_to_rgb(kelvin: f32) -> [f32; 3] {
    let temp = kelvin / 100.0;

    let r;
    let g;
    let b;

    if temp <= 66.0 {
        r = 1.0;
        g = (99.4708025861 * temp.ln() - 161.1195681661).clamp(0.0, 255.0) / 255.0;
    } else {
        r = (329.698727446 * (temp - 60.0).powf(-0.1332047592)).clamp(0.0, 255.0) / 255.0;
        g = (288.1221695283 * (temp - 60.0).powf(-0.0755148492)).clamp(0.0, 255.0) / 255.0;
    }

    if temp >= 66.0 {
        b = 1.0;
    } else if temp <= 19.0 {
        b = 0.0;
    } else {
        b = (138.5177312231 * (temp - 10.0).ln() - 305.0447927307).clamp(0.0, 255.0) / 255.0;
    }

    [r, g, b]
}

/// Apply white balance by shifting color temperature.
///
/// `target_kelvin` is the desired temperature. 6500K is daylight neutral.
pub fn apply_white_balance(color: [f32; 3], target_kelvin: f32) -> [f32; 3] {
    let multiplier = temperature_to_rgb(target_kelvin);
    // Normalize so that 6500K is a no-op.
    let reference = temperature_to_rgb(6500.0);
    let scale = [
        multiplier[0] / reference[0].max(1e-6),
        multiplier[1] / reference[1].max(1e-6),
        multiplier[2] / reference[2].max(1e-6),
    ];
    [
        color[0] * scale[0],
        color[1] * scale[1],
        color[2] * scale[2],
    ]
}

/// Apply split toning to a color.
///
/// Tints shadows with one color and highlights with another, based on
/// luminance.
pub fn apply_split_toning(
    color: [f32; 3],
    settings: &SplitToningSettings,
) -> [f32; 3] {
    if settings.intensity.abs() < 1e-6 {
        return color;
    }

    let luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];

    // Determine shadow/highlight blend weight based on luminance and
    // balance.
    let midpoint = 0.5 + settings.balance * 0.5;
    let shadow_weight = (1.0 - luminance / midpoint.max(0.01)).clamp(0.0, 1.0);
    let highlight_weight = ((luminance - midpoint) / (1.0 - midpoint).max(0.01)).clamp(0.0, 1.0);

    let mut result = color;
    for i in 0..3 {
        let shadow_tint = lerp(color[i], settings.shadow_tint[i] * luminance, shadow_weight);
        let highlight_tint = lerp(
            color[i],
            settings.highlight_tint[i] * luminance,
            highlight_weight,
        );
        result[i] = lerp(color[i], lerp(shadow_tint, highlight_tint, 0.5), settings.intensity);
    }

    result
}

/// Apply vignette to a color based on screen UV.
pub fn apply_vignette(color: [f32; 3], uv: [f32; 2], settings: &VignetteSettings) -> [f32; 3] {
    if settings.intensity.abs() < 1e-6 {
        return color;
    }

    let dx = (uv[0] - settings.center[0]) * 2.0;
    let dy = (uv[1] - settings.center[1]) * 2.0;

    // Adjust for roundness (0 = use aspect ratio, 1 = circle).
    let aspect = 1.0; // Would use viewport aspect ratio in practice.
    let adjusted_dx = dx * lerp(aspect, 1.0, settings.roundness);
    let adjusted_dy = dy;

    let dist = (adjusted_dx * adjusted_dx + adjusted_dy * adjusted_dy).sqrt();
    let vignette = smooth_step(1.0, 1.0 - settings.smoothness, dist);
    let factor = lerp(1.0, vignette, settings.intensity);

    [
        lerp(settings.color[0], color[0], factor),
        lerp(settings.color[1], color[1], factor),
        lerp(settings.color[2], color[2], factor),
    ]
}

/// Generate film grain noise value for a given pixel position and frame.
///
/// Uses a hash-based noise function for random-looking per-pixel grain.
pub fn film_grain_noise(x: u32, y: u32, frame: u64, size: f32) -> f32 {
    // Scale coordinates by grain size (larger = coarser).
    let sx = (x as f32 / size) as u32;
    let sy = (y as f32 / size) as u32;
    let sf = (frame % 256) as u32;

    // Simple integer hash.
    let mut hash = sx.wrapping_mul(374761393)
        .wrapping_add(sy.wrapping_mul(668265263))
        .wrapping_add(sf.wrapping_mul(1274126177));
    hash = (hash ^ (hash >> 13)).wrapping_mul(1103515245);
    hash = hash ^ (hash >> 16);

    // Map to [-1, 1].
    (hash as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// Apply film grain to a color.
pub fn apply_film_grain(
    color: [f32; 3],
    x: u32,
    y: u32,
    frame: u64,
    settings: &FilmGrainSettings,
) -> [f32; 3] {
    if settings.intensity.abs() < 1e-6 {
        return color;
    }

    let luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];

    // Response: more grain in midtones.
    let response_weight = if settings.response > 0.0 {
        let mid_dist = (luminance - 0.5).abs();
        let mid_factor = 1.0 - (mid_dist * 2.0).clamp(0.0, 1.0);
        lerp(1.0, mid_factor, settings.response)
    } else {
        1.0
    };

    let grain = film_grain_noise(x, y, frame, settings.size) * settings.intensity * response_weight;

    if settings.luminance_only {
        [color[0] + grain, color[1] + grain, color[2] + grain]
    } else {
        // Per-channel noise.
        let gr = film_grain_noise(x, y, frame.wrapping_add(1), settings.size)
            * settings.intensity
            * response_weight;
        let gg = film_grain_noise(x, y, frame.wrapping_add(2), settings.size)
            * settings.intensity
            * response_weight;
        let gb = film_grain_noise(x, y, frame.wrapping_add(3), settings.size)
            * settings.intensity
            * response_weight;
        [color[0] + gr, color[1] + gg, color[2] + gb]
    }
}

/// Compute chromatic aberration UV offsets.
///
/// Returns three UV coordinates: (red_uv, green_uv, blue_uv).
/// Green stays at the original UV, red shifts outward, blue shifts inward
/// (or vice versa).
pub fn chromatic_aberration_uvs(
    uv: [f32; 2],
    center: [f32; 2],
    intensity_pixels: f32,
    viewport_width: f32,
    viewport_height: f32,
) -> [[f32; 2]; 3] {
    let dx = uv[0] - center[0];
    let dy = uv[1] - center[1];
    let dist = (dx * dx + dy * dy).sqrt();

    if dist < 1e-6 || intensity_pixels.abs() < 1e-6 {
        return [uv, uv, uv];
    }

    let dir = [dx / dist, dy / dist];
    let offset = dist * intensity_pixels;

    let r_offset = [
        offset / viewport_width * dir[0],
        offset / viewport_height * dir[1],
    ];
    let b_offset = [
        -offset / viewport_width * dir[0] * 0.5,
        -offset / viewport_height * dir[1] * 0.5,
    ];

    [
        [uv[0] + r_offset[0], uv[1] + r_offset[1]], // Red: shifted outward
        uv,                                            // Green: center
        [uv[0] + b_offset[0], uv[1] + b_offset[1]], // Blue: shifted inward
    ]
}

// ---------------------------------------------------------------------------
// 3D LUT generation
// ---------------------------------------------------------------------------

/// Generate a neutral (identity) 3D LUT of the given size.
///
/// Returns a flat array of `size^3` RGB values. The LUT maps input color
/// to the same output color (no grading).
pub fn generate_identity_lut(size: u32) -> Vec<[f32; 3]> {
    let total = (size * size * size) as usize;
    let mut lut = Vec::with_capacity(total);
    let scale = 1.0 / (size - 1) as f32;

    for b in 0..size {
        for g in 0..size {
            for r in 0..size {
                lut.push([r as f32 * scale, g as f32 * scale, b as f32 * scale]);
            }
        }
    }

    lut
}

/// Apply a 3D LUT to a color using trilinear interpolation.
///
/// `color` should be in [0, 1] range.
/// `lut` is a flat array of size^3 RGB values.
pub fn sample_lut(color: [f32; 3], lut: &[[f32; 3]], size: u32) -> [f32; 3] {
    let max_idx = (size - 1) as f32;
    let r = (color[0] * max_idx).clamp(0.0, max_idx);
    let g = (color[1] * max_idx).clamp(0.0, max_idx);
    let b = (color[2] * max_idx).clamp(0.0, max_idx);

    let r0 = r.floor() as u32;
    let g0 = g.floor() as u32;
    let b0 = b.floor() as u32;
    let r1 = (r0 + 1).min(size - 1);
    let g1 = (g0 + 1).min(size - 1);
    let b1 = (b0 + 1).min(size - 1);

    let fr = r - r.floor();
    let fg = g - g.floor();
    let fb = b - b.floor();

    let idx = |r: u32, g: u32, b: u32| -> usize {
        (b * size * size + g * size + r) as usize
    };

    // Trilinear interpolation.
    let c000 = lut[idx(r0, g0, b0)];
    let c100 = lut[idx(r1, g0, b0)];
    let c010 = lut[idx(r0, g1, b0)];
    let c110 = lut[idx(r1, g1, b0)];
    let c001 = lut[idx(r0, g0, b1)];
    let c101 = lut[idx(r1, g0, b1)];
    let c011 = lut[idx(r0, g1, b1)];
    let c111 = lut[idx(r1, g1, b1)];

    let mut result = [0.0f32; 3];
    for i in 0..3 {
        let c00 = lerp(c000[i], c100[i], fr);
        let c01 = lerp(c001[i], c101[i], fr);
        let c10 = lerp(c010[i], c110[i], fr);
        let c11 = lerp(c011[i], c111[i], fr);

        let c0 = lerp(c00, c10, fg);
        let c1 = lerp(c01, c11, fg);

        result[i] = lerp(c0, c1, fb);
    }

    result
}

/// Generate a LUT from color grading parameters.
///
/// Bakes lift/gamma/gain, saturation, contrast, white balance, and split
/// toning into a 3D LUT for efficient single-pass application.
pub fn generate_graded_lut(settings: &ColorGradingSettings, size: u32) -> Vec<[f32; 3]> {
    let identity = generate_identity_lut(size);
    let mut graded = Vec::with_capacity(identity.len());

    for color in &identity {
        let mut c = *color;

        // White balance.
        c = apply_white_balance(c, settings.temperature);

        // Lift/Gamma/Gain.
        c = apply_lift_gamma_gain(c, settings.lift, settings.gamma, settings.gain);

        // Contrast.
        c = adjust_contrast(c, settings.contrast, 0.18);

        // Saturation.
        c = adjust_saturation(c, settings.saturation);

        // Brightness.
        c[0] += settings.brightness;
        c[1] += settings.brightness;
        c[2] += settings.brightness;

        // Split toning.
        c = apply_split_toning(c, &settings.split_toning);

        // Clamp.
        c[0] = c[0].clamp(0.0, 1.0);
        c[1] = c[1].clamp(0.0, 1.0);
        c[2] = c[2].clamp(0.0, 1.0);

        graded.push(c);
    }

    graded
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn smooth_step(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// ColorGradingEffect
// ---------------------------------------------------------------------------

/// Color grading post-process effect.
pub struct ColorGradingEffect {
    pub settings: ColorGradingSettings,
    /// Baked 3D LUT from settings (regenerated when settings change).
    baked_lut: Vec<[f32; 3]>,
    /// Whether the baked LUT needs regeneration.
    lut_dirty: bool,
}

impl ColorGradingEffect {
    pub fn new(settings: ColorGradingSettings) -> Self {
        let lut_size = settings.lut_size;
        let baked_lut = generate_graded_lut(&settings, lut_size);
        Self {
            settings,
            baked_lut,
            lut_dirty: false,
        }
    }

    /// Mark the LUT as needing regeneration (call after changing settings).
    pub fn invalidate_lut(&mut self) {
        self.lut_dirty = true;
    }

    /// Regenerate the baked LUT if dirty.
    fn ensure_lut(&mut self) {
        if self.lut_dirty {
            self.baked_lut =
                generate_graded_lut(&self.settings, self.settings.lut_size);
            self.lut_dirty = false;
        }
    }

    /// Apply all color grading to a single pixel (CPU reference).
    pub fn grade_pixel(
        &self,
        color: [f32; 3],
        uv: [f32; 2],
        pixel_x: u32,
        pixel_y: u32,
        frame: u64,
    ) -> [f32; 3] {
        let mut c = color;

        // Apply 3D LUT.
        if !self.baked_lut.is_empty() {
            let lut_color = sample_lut(c, &self.baked_lut, self.settings.lut_size);
            let contrib = self.settings.lut_contribution;
            c[0] = lerp(c[0], lut_color[0], contrib);
            c[1] = lerp(c[1], lut_color[1], contrib);
            c[2] = lerp(c[2], lut_color[2], contrib);
        }

        // Vignette.
        c = apply_vignette(c, uv, &self.settings.vignette);

        // Film grain.
        c = apply_film_grain(c, pixel_x, pixel_y, frame, &self.settings.film_grain);

        // Clamp final result.
        c[0] = c[0].clamp(0.0, 1.0);
        c[1] = c[1].clamp(0.0, 1.0);
        c[2] = c[2].clamp(0.0, 1.0);

        c
    }
}

impl PostProcessEffect for ColorGradingEffect {
    fn name(&self) -> &str {
        "ColorGrading"
    }

    fn execute(&self, _input: &PostProcessInput, _output: &mut PostProcessOutput) {
        if !self.settings.enabled {
            return;
        }

        // In a real implementation:
        // 1. If chromatic_aberration.intensity > 0, apply CA by splitting
        //    channels with offset UVs.
        // 2. Upload the baked 3D LUT texture (or use the external LUT).
        // 3. Dispatch the color grading compute shader which:
        //    a. Samples the scene color.
        //    b. Applies the 3D LUT via trilinear interpolation.
        //    c. Applies vignette.
        //    d. Applies film grain.
    }

    fn is_enabled(&self) -> bool {
        self.settings.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.settings.enabled = enabled;
    }

    fn priority(&self) -> u32 {
        700
    }

    fn on_resize(&mut self, _width: u32, _height: u32) {
        // No resize-dependent resources.
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

/// Color grading compute shader with LUT, vignette, grain, and chromatic
/// aberration.
pub const COLOR_GRADING_WGSL: &str = r#"
// Color Grading — compute shader
// Applies 3D LUT, vignette, film grain, and chromatic aberration.

struct ColorGradeParams {
    lut_size:           f32,
    lut_contribution:   f32,
    vignette_intensity: f32,
    vignette_smoothness: f32,
    vignette_roundness: f32,
    vignette_center_x:  f32,
    vignette_center_y:  f32,
    grain_intensity:    f32,
    grain_size:         f32,
    grain_response:     f32,
    grain_luminance_only: u32,
    ca_intensity:       f32,
    ca_center_x:        f32,
    ca_center_y:        f32,
    inv_width:          f32,
    inv_height:         f32,
    viewport_width:     f32,
    viewport_height:    f32,
    frame_index:        u32,
    _pad0:              u32,
    _pad1:              u32,
    _pad2:              u32,
};

@group(0) @binding(0) var src_texture:  texture_2d<f32>;
@group(0) @binding(1) var lut_texture:  texture_3d<f32>;
@group(0) @binding(2) var dst_texture:  texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var tex_sampler:  sampler;
@group(0) @binding(4) var lut_sampler:  sampler;
@group(0) @binding(5) var<uniform> params: ColorGradeParams;

fn hash_noise(p: vec3<u32>) -> f32 {
    var h = p.x * 374761393u + p.y * 668265263u + p.z * 1274126177u;
    h = (h ^ (h >> 13u)) * 1103515245u;
    h = h ^ (h >> 16u);
    return f32(h) / f32(0xFFFFFFFFu) * 2.0 - 1.0;
}

fn sample_lut(color: vec3<f32>) -> vec3<f32> {
    let size = params.lut_size;
    let scale = (size - 1.0) / size;
    let offset = 0.5 / size;
    let uv3d = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)) * scale + offset;
    return textureSampleLevel(lut_texture, lut_sampler, uv3d, 0.0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_texture);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    var color: vec3<f32>;

    // ---- Chromatic Aberration ----
    if params.ca_intensity > 0.001 {
        let center = vec2<f32>(params.ca_center_x, params.ca_center_y);
        let diff = uv - center;
        let dist = length(diff);
        let dir = diff / max(dist, 1e-6);
        let offset = dist * params.ca_intensity;

        let r_uv = uv + dir * offset / vec2<f32>(params.viewport_width, params.viewport_height);
        let b_uv = uv - dir * offset * 0.5 / vec2<f32>(params.viewport_width, params.viewport_height);

        let r = textureSampleLevel(src_texture, tex_sampler, r_uv, 0.0).r;
        let g = textureSampleLevel(src_texture, tex_sampler, uv, 0.0).g;
        let b = textureSampleLevel(src_texture, tex_sampler, b_uv, 0.0).b;
        color = vec3<f32>(r, g, b);
    } else {
        color = textureSampleLevel(src_texture, tex_sampler, uv, 0.0).rgb;
    }

    // ---- 3D LUT ----
    if params.lut_contribution > 0.001 {
        let lut_color = sample_lut(color);
        color = mix(color, lut_color, params.lut_contribution);
    }

    // ---- Vignette ----
    if params.vignette_intensity > 0.001 {
        let vc = vec2<f32>(params.vignette_center_x, params.vignette_center_y);
        let d = (uv - vc) * 2.0;
        let aspect = params.viewport_width / params.viewport_height;
        let adj_x = d.x * mix(aspect, 1.0, params.vignette_roundness);
        let vdist = length(vec2<f32>(adj_x, d.y));
        let vignette = smoothstep(1.0, 1.0 - params.vignette_smoothness, vdist);
        let factor = mix(1.0, vignette, params.vignette_intensity);
        color *= factor;
    }

    // ---- Film Grain ----
    if params.grain_intensity > 0.001 {
        let grain_coord = vec3<u32>(
            u32(f32(gid.x) / params.grain_size),
            u32(f32(gid.y) / params.grain_size),
            params.frame_index % 256u
        );
        var grain = hash_noise(grain_coord) * params.grain_intensity;

        // Response curve: reduce grain in very dark/bright areas
        let lum = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
        let mid_dist = abs(lum - 0.5);
        let response = mix(1.0, 1.0 - clamp(mid_dist * 2.0, 0.0, 1.0), params.grain_response);
        grain *= response;

        if params.grain_luminance_only != 0u {
            color += grain;
        } else {
            let gr = hash_noise(grain_coord + vec3<u32>(1u, 0u, 0u)) * params.grain_intensity * response;
            let gg = hash_noise(grain_coord + vec3<u32>(0u, 1u, 0u)) * params.grain_intensity * response;
            let gb = hash_noise(grain_coord + vec3<u32>(0u, 0u, 1u)) * params.grain_intensity * response;
            color += vec3<f32>(gr, gg, gb);
        }
    }

    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(dst_texture, gid.xy, vec4<f32>(color, 1.0));
}
"#;

/// LUT generation compute shader — bakes color grading parameters into a
/// 3D texture.
pub const LUT_BAKE_WGSL: &str = r#"
// 3D LUT bake — compute shader
// Generates a 3D LUT from lift/gamma/gain/saturation/contrast parameters.

struct LUTBakeParams {
    lift:        vec4<f32>,    // .rgb = lift, .a = unused
    gamma:       vec4<f32>,
    gain:        vec4<f32>,
    saturation:  f32,
    contrast:    f32,
    brightness:  f32,
    temperature: f32,
    lut_size:    u32,
    _pad0:       u32,
    _pad1:       u32,
    _pad2:       u32,
    shadow_tint:    vec4<f32>,
    highlight_tint: vec4<f32>,
    split_balance:  f32,
    split_intensity: f32,
    _pad3:       f32,
    _pad4:       f32,
};

@group(0) @binding(0) var lut_output: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: LUTBakeParams;

fn temperature_to_rgb(kelvin: f32) -> vec3<f32> {
    let temp = kelvin / 100.0;
    var r: f32;
    var g: f32;
    var b: f32;

    if temp <= 66.0 {
        r = 1.0;
        g = clamp(99.4708 * log(temp) - 161.1196, 0.0, 255.0) / 255.0;
    } else {
        r = clamp(329.6987 * pow(temp - 60.0, -0.1332), 0.0, 255.0) / 255.0;
        g = clamp(288.1222 * pow(temp - 60.0, -0.0755), 0.0, 255.0) / 255.0;
    }

    if temp >= 66.0 {
        b = 1.0;
    } else if temp <= 19.0 {
        b = 0.0;
    } else {
        b = clamp(138.5177 * log(temp - 10.0) - 305.0448, 0.0, 255.0) / 255.0;
    }

    return vec3<f32>(r, g, b);
}

@compute @workgroup_size(4, 4, 4)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.lut_size;
    if gid.x >= size || gid.y >= size || gid.z >= size {
        return;
    }

    let scale = 1.0 / f32(size - 1u);
    var color = vec3<f32>(f32(gid.x), f32(gid.y), f32(gid.z)) * scale;

    // White balance
    let wb = temperature_to_rgb(params.temperature);
    let ref_wb = temperature_to_rgb(6500.0);
    color *= wb / max(ref_wb, vec3<f32>(1e-6));

    // Lift/Gamma/Gain (ASC CDL approximation)
    let lifted = params.lift.rgb * (1.0 - color) + color;
    let inv_gamma = 1.0 / max(params.gamma.rgb, vec3<f32>(1e-6));
    color = params.gain.rgb * pow(max(lifted, vec3<f32>(0.0)), inv_gamma);

    // Contrast (around 0.18 midpoint for linear)
    color = (color - 0.18) * params.contrast + 0.18;

    // Saturation
    let lum = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    color = vec3<f32>(lum) + (color - lum) * params.saturation;

    // Brightness
    color += params.brightness;

    // Split toning
    if params.split_intensity > 0.001 {
        let midpoint = 0.5 + params.split_balance * 0.5;
        let shadow_w = clamp(1.0 - lum / max(midpoint, 0.01), 0.0, 1.0);
        let highlight_w = clamp((lum - midpoint) / max(1.0 - midpoint, 0.01), 0.0, 1.0);
        let shadow = mix(color, params.shadow_tint.rgb * lum, shadow_w);
        let highlight = mix(color, params.highlight_tint.rgb * lum, highlight_w);
        color = mix(color, mix(shadow, highlight, 0.5), params.split_intensity);
    }

    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(lut_output, gid, vec4<f32>(color, 1.0));
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_lut() {
        let lut = generate_identity_lut(4);
        assert_eq!(lut.len(), 64); // 4^3

        // Corner cases: (0,0,0) -> black, (1,1,1) -> white.
        assert!(lut[0][0].abs() < 1e-5);
        let last = lut.last().unwrap();
        assert!((last[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_lut_trilinear() {
        let lut = generate_identity_lut(16);
        // Identity LUT should return the same color.
        let color = [0.3, 0.6, 0.9];
        let result = sample_lut(color, &lut, 16);
        for i in 0..3 {
            assert!(
                (result[i] - color[i]).abs() < 0.02,
                "LUT identity failed: channel {i}, expected {}, got {}",
                color[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_lift_gamma_gain_neutral() {
        let color = [0.5, 0.3, 0.8];
        let result = apply_lift_gamma_gain(color, [0.0; 3], [1.0; 3], [1.0; 3]);
        for i in 0..3 {
            assert!((result[i] - color[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_saturation() {
        // Full desaturation -> grayscale.
        let color = [1.0, 0.0, 0.0]; // pure red
        let result = adjust_saturation(color, 0.0);
        // All channels should be equal (gray).
        assert!((result[0] - result[1]).abs() < 1e-5);
        assert!((result[1] - result[2]).abs() < 1e-5);
    }

    #[test]
    fn test_contrast() {
        let color = [0.18, 0.18, 0.18]; // midpoint
        let result = adjust_contrast(color, 2.0, 0.18);
        // At midpoint, contrast shouldn't change the value.
        for i in 0..3 {
            assert!((result[i] - 0.18).abs() < 1e-5);
        }
    }

    #[test]
    fn test_temperature_daylight() {
        let rgb = temperature_to_rgb(6500.0);
        // Daylight should be close to neutral white.
        assert!(rgb[0] > 0.9);
        assert!(rgb[1] > 0.9);
        assert!(rgb[2] > 0.8);
    }

    #[test]
    fn test_temperature_warm_vs_cool() {
        let warm = temperature_to_rgb(3000.0); // warm (incandescent)
        let cool = temperature_to_rgb(10000.0); // cool (overcast)

        // Warm should have more red than blue.
        assert!(warm[0] > warm[2]);
        // Cool should have more blue relative to warm.
        assert!(cool[2] > warm[2]);
    }

    #[test]
    fn test_vignette_center() {
        let color = [1.0, 1.0, 1.0];
        let settings = VignetteSettings {
            intensity: 1.0,
            ..Default::default()
        };

        // At center, vignette should have minimal effect.
        let result = apply_vignette(color, [0.5, 0.5], &settings);
        assert!(result[0] > 0.8);
    }

    #[test]
    fn test_film_grain_deterministic() {
        let a = film_grain_noise(100, 200, 0, 1.0);
        let b = film_grain_noise(100, 200, 0, 1.0);
        assert!((a - b).abs() < 1e-10);
    }

    #[test]
    fn test_chromatic_aberration_at_center() {
        let uvs = chromatic_aberration_uvs([0.5, 0.5], [0.5, 0.5], 5.0, 1920.0, 1080.0);
        // At the center, there should be minimal offset.
        for i in 0..3 {
            assert!((uvs[i][0] - 0.5).abs() < 0.01);
            assert!((uvs[i][1] - 0.5).abs() < 0.01);
        }
    }

    #[test]
    fn test_chromatic_aberration_at_edge() {
        let uvs =
            chromatic_aberration_uvs([0.9, 0.5], [0.5, 0.5], 5.0, 1920.0, 1080.0);
        // Red should be shifted further from center than blue.
        let r_dist = (uvs[0][0] - 0.5).abs();
        let b_dist = (uvs[2][0] - 0.5).abs();
        // Red shifts outward, blue shifts inward.
        assert!(r_dist > b_dist || (uvs[0][0] - uvs[2][0]).abs() > 0.001);
    }

    #[test]
    fn test_color_grading_effect_interface() {
        let effect = ColorGradingEffect::new(ColorGradingSettings::default());
        assert_eq!(effect.name(), "ColorGrading");
        assert!(effect.is_enabled());
        assert_eq!(effect.priority(), 700);
    }

    #[test]
    fn test_graded_lut_generation() {
        let settings = ColorGradingSettings {
            contrast: 1.2,
            saturation: 0.8,
            ..Default::default()
        };
        let lut = generate_graded_lut(&settings, 8);
        assert_eq!(lut.len(), 512); // 8^3

        // All values should be in [0, 1].
        for entry in &lut {
            for &c in entry {
                assert!(c >= 0.0 && c <= 1.0);
            }
        }
    }
}
