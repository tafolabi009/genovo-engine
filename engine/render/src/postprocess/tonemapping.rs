// engine/render/src/postprocess/tonemapping.rs
//
// Tone mapping transforms the HDR (high dynamic range) scene color into the
// displayable LDR (low dynamic range) range [0, 1]. This module provides
// multiple industry-standard tone-mapping operators, automatic exposure
// based on luminance histograms, and final gamma correction.

use std::any::Any;

use super::{PostProcessEffect, PostProcessInput, PostProcessOutput, TextureId};

// ---------------------------------------------------------------------------
// Tone map operators
// ---------------------------------------------------------------------------

/// Selects which tone-mapping curve to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToneMapOperator {
    /// No tonemapping — clamp to [0,1].
    None,
    /// Reinhard: `color / (1 + color)`.
    Reinhard,
    /// Reinhard extended: `color * (1 + color/white^2) / (1 + color)`.
    ReinhardExtended,
    /// ACES filmic: Academy Color Encoding System (RRT + ODT).
    AcesFilmic,
    /// Uncharted 2 / Hable filmic curve.
    Uncharted2,
    /// AgX — modern film-like tonemapping with better hue preservation.
    AgX,
    /// Khronos PBR Neutral — designed for glTF / PBR content.
    KhronosPbrNeutral,
}

impl ToneMapOperator {
    /// Apply the selected tone mapping operator to a linear HDR color.
    pub fn apply(&self, color: [f32; 3]) -> [f32; 3] {
        match self {
            Self::None => [
                color[0].clamp(0.0, 1.0),
                color[1].clamp(0.0, 1.0),
                color[2].clamp(0.0, 1.0),
            ],
            Self::Reinhard => Self::reinhard(color),
            Self::ReinhardExtended => Self::reinhard_extended(color, 4.0),
            Self::AcesFilmic => Self::aces_filmic(color),
            Self::Uncharted2 => Self::uncharted2(color),
            Self::AgX => Self::agx(color),
            Self::KhronosPbrNeutral => Self::khronos_pbr_neutral(color),
        }
    }

    /// Reinhard: `c / (1 + c)`
    fn reinhard(c: [f32; 3]) -> [f32; 3] {
        [
            c[0] / (1.0 + c[0]),
            c[1] / (1.0 + c[1]),
            c[2] / (1.0 + c[2]),
        ]
    }

    /// Reinhard extended: `c * (1 + c/w^2) / (1 + c)`
    /// where `w` is the white point.
    fn reinhard_extended(c: [f32; 3], white: f32) -> [f32; 3] {
        let w2 = white * white;
        [
            c[0] * (1.0 + c[0] / w2) / (1.0 + c[0]),
            c[1] * (1.0 + c[1] / w2) / (1.0 + c[1]),
            c[2] * (1.0 + c[2] / w2) / (1.0 + c[2]),
        ]
    }

    /// ACES filmic tone mapping (fitted curve by Stephen Hill).
    /// Uses the RRT (Reference Rendering Transform) + ODT (Output Device
    /// Transform) approximation.
    fn aces_filmic(c: [f32; 3]) -> [f32; 3] {
        // sRGB -> ACEScg input transform (simplified 3x3)
        let aces_input = Self::mul_mat3(
            c,
            [
                [0.59719, 0.35458, 0.04823],
                [0.07600, 0.90834, 0.01566],
                [0.02840, 0.13383, 0.83777],
            ],
        );

        // RRT + ODT fit
        let mapped = Self::aces_rrt_odt(aces_input);

        // ACEScg -> sRGB output transform (simplified 3x3)
        let result = Self::mul_mat3(
            mapped,
            [
                [1.60475, -0.53108, -0.07367],
                [-0.10208, 1.10813, -0.00605],
                [-0.00327, -0.07276, 1.07602],
            ],
        );

        [
            result[0].clamp(0.0, 1.0),
            result[1].clamp(0.0, 1.0),
            result[2].clamp(0.0, 1.0),
        ]
    }

    /// ACES RRT+ODT curve: `(x * (a*x + b)) / (x * (c*x + d) + e)`
    fn aces_rrt_odt(c: [f32; 3]) -> [f32; 3] {
        let a = 0.0245786;
        let b = 0.000090537;
        let c_val = 0.983729;
        let d = 0.4329510;
        let e = 0.238081;

        [
            (c[0] * (c[0] + a) - b) / (c[0] * (c_val * c[0] + d) + e),
            (c[1] * (c[1] + a) - b) / (c[1] * (c_val * c[1] + d) + e),
            (c[2] * (c[2] + a) - b) / (c[2] * (c_val * c[2] + d) + e),
        ]
    }

    /// Uncharted 2 / Hable filmic curve.
    /// `F(x) = ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F)) - E/F`
    fn uncharted2(color: [f32; 3]) -> [f32; 3] {
        // Curve parameters (John Hable's original values)
        const A: f32 = 0.15; // Shoulder strength
        const B: f32 = 0.50; // Linear strength
        const C: f32 = 0.10; // Linear angle
        const D: f32 = 0.20; // Toe strength
        const E: f32 = 0.02; // Toe numerator
        const F: f32 = 0.30; // Toe denominator
        const W: f32 = 11.2; // Linear white point

        fn curve(x: f32) -> f32 {
            ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
        }

        let white_scale = 1.0 / curve(W);
        [
            curve(color[0]) * white_scale,
            curve(color[1]) * white_scale,
            curve(color[2]) * white_scale,
        ]
    }

    /// AgX tone mapping — modern approach with better hue preservation
    /// under extreme exposures. Inspired by Blender's AgX implementation.
    fn agx(color: [f32; 3]) -> [f32; 3] {
        // AgX log encoding
        let min_ev: f32 = -12.47393;
        let max_ev: f32 = 4.026069;

        // Convert to AgX log space
        let agx_color = Self::mul_mat3(
            color,
            [
                [0.842479, 0.0784336, 0.0792237],
                [0.0423303, 0.878468, 0.0791916],
                [0.0423745, 0.0784336, 0.879142],
            ],
        );

        // Log2 encoding with clamping
        let log_color = [
            (agx_color[0].max(1e-10).ln() / std::f32::consts::LN_2)
                .clamp(min_ev, max_ev),
            (agx_color[1].max(1e-10).ln() / std::f32::consts::LN_2)
                .clamp(min_ev, max_ev),
            (agx_color[2].max(1e-10).ln() / std::f32::consts::LN_2)
                .clamp(min_ev, max_ev),
        ];

        // Normalize to [0, 1]
        let range = max_ev - min_ev;
        let encoded = [
            (log_color[0] - min_ev) / range,
            (log_color[1] - min_ev) / range,
            (log_color[2] - min_ev) / range,
        ];

        // 6th-order polynomial approximation of the AgX sigmoid
        // Attempt to approximate the S-curve used in the AgX default look.
        let mapped = [
            Self::agx_default_contrast(encoded[0]),
            Self::agx_default_contrast(encoded[1]),
            Self::agx_default_contrast(encoded[2]),
        ];

        // Convert back from AgX space
        let result = Self::mul_mat3(
            mapped,
            [
                [1.19687, -0.0980208, -0.0990297],
                [-0.0528968, 1.15190, -0.0989389],
                [-0.0529716, -0.0980434, 1.15107],
            ],
        );

        [
            result[0].clamp(0.0, 1.0),
            result[1].clamp(0.0, 1.0),
            result[2].clamp(0.0, 1.0),
        ]
    }

    /// AgX default contrast sigmoid approximation.
    fn agx_default_contrast(x: f32) -> f32 {
        // Attempt a polynomial fit to the AgX default look sigmoid.
        let x2 = x * x;
        let x4 = x2 * x2;
        // Attempt to fit: 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4
        //                 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232
        15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2
            + 0.1191 * x
            - 0.00232
    }

    /// Khronos PBR Neutral tone mapping.
    /// Designed to be neutral for PBR/glTF content — minimal color shift.
    fn khronos_pbr_neutral(color: [f32; 3]) -> [f32; 3] {
        let start_compression = 0.8 - 0.04;
        let desaturation = 0.15;

        let mut c = color;

        let x = c[0].min(c[1]).min(c[2]);
        let offset = if x < 0.08 {
            x - 6.25 * x * x
        } else {
            0.04
        };
        c[0] -= offset;
        c[1] -= offset;
        c[2] -= offset;

        let peak = c[0].max(c[1]).max(c[2]);
        if peak < start_compression {
            return c;
        }

        let d = 1.0 - start_compression;
        let new_peak = 1.0 - d * d / (peak + d - start_compression);
        let scale = new_peak / peak;
        c[0] *= scale;
        c[1] *= scale;
        c[2] *= scale;

        let t = ((peak - new_peak) / peak).max(0.0) * (1.0 / desaturation);
        let chroma_r = c[0] / new_peak;
        let chroma_g = c[1] / new_peak;
        let chroma_b = c[2] / new_peak;
        [
            Self::lerp(c[0], new_peak * chroma_r.powf(1.0 / (1.0 + t)), t.min(1.0)),
            Self::lerp(c[1], new_peak * chroma_g.powf(1.0 / (1.0 + t)), t.min(1.0)),
            Self::lerp(c[2], new_peak * chroma_b.powf(1.0 / (1.0 + t)), t.min(1.0)),
        ]
    }

    // -- Utility helpers --

    fn mul_mat3(v: [f32; 3], m: [[f32; 3]; 3]) -> [f32; 3] {
        [
            v[0] * m[0][0] + v[1] * m[0][1] + v[2] * m[0][2],
            v[0] * m[1][0] + v[1] * m[1][1] + v[2] * m[1][2],
            v[0] * m[2][0] + v[1] * m[2][1] + v[2] * m[2][2],
        ]
    }

    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }
}

// ---------------------------------------------------------------------------
// Auto-exposure
// ---------------------------------------------------------------------------

/// Settings for the automatic exposure system.
#[derive(Debug, Clone)]
pub struct ExposureSettings {
    /// Minimum exposure value (EV).
    pub min_ev: f32,
    /// Maximum exposure value (EV).
    pub max_ev: f32,
    /// How quickly exposure adapts to brightness changes (per second).
    /// Higher = faster. Separate speeds for brightening/darkening.
    pub adaptation_speed_up: f32,
    pub adaptation_speed_down: f32,
    /// Exposure compensation (added to the computed EV).
    pub compensation: f32,
    /// Whether auto-exposure is enabled.
    pub enabled: bool,
    /// Manual exposure override (EV100). Used when auto-exposure is disabled.
    pub manual_ev: f32,
    /// Histogram bin count for luminance analysis.
    pub histogram_bins: u32,
    /// Percentage of darkest pixels to ignore (shadow clipping).
    pub low_percentile: f32,
    /// Percentage of brightest pixels to ignore (highlight clipping).
    pub high_percentile: f32,
}

impl Default for ExposureSettings {
    fn default() -> Self {
        Self {
            min_ev: -4.0,
            max_ev: 16.0,
            adaptation_speed_up: 3.0,
            adaptation_speed_down: 1.5,
            compensation: 0.0,
            enabled: true,
            manual_ev: 8.0,
            histogram_bins: 256,
            low_percentile: 0.1,
            high_percentile: 0.95,
        }
    }
}

/// Persistent state for auto-exposure across frames.
#[derive(Debug, Clone)]
pub struct AutoExposureState {
    /// Current adapted exposure value (EV).
    pub current_ev: f32,
    /// Luminance histogram from the previous frame.
    pub histogram: Vec<u32>,
    /// Average scene luminance from the previous frame.
    pub avg_luminance: f32,
}

impl AutoExposureState {
    pub fn new() -> Self {
        Self {
            current_ev: 8.0,
            histogram: vec![0u32; 256],
            avg_luminance: 0.18,
        }
    }

    /// Compute the target EV from the luminance histogram.
    pub fn compute_target_ev(&self, settings: &ExposureSettings) -> f32 {
        if self.histogram.is_empty() {
            return settings.manual_ev;
        }

        let total_pixels: u32 = self.histogram.iter().sum();
        if total_pixels == 0 {
            return settings.manual_ev;
        }

        // Ignore the low and high percentiles (shadow/highlight clipping).
        let low_count = (total_pixels as f32 * settings.low_percentile) as u32;
        let high_count = (total_pixels as f32 * settings.high_percentile) as u32;

        let mut accumulated = 0u32;
        let mut weighted_sum = 0.0f64;
        let mut counted = 0u32;
        let bin_count = self.histogram.len() as f32;

        for (i, &count) in self.histogram.iter().enumerate() {
            let prev_accumulated = accumulated;
            accumulated += count;

            if accumulated < low_count {
                continue;
            }
            if prev_accumulated > high_count {
                break;
            }

            // Effective count within the valid range
            let effective = count.min(accumulated - low_count).min(high_count - prev_accumulated);
            let bin_center = (i as f32 + 0.5) / bin_count;
            // Map bin position to log2 luminance in [min_ev, max_ev]
            let log_luminance =
                settings.min_ev + bin_center * (settings.max_ev - settings.min_ev);
            weighted_sum += log_luminance as f64 * effective as f64;
            counted += effective;
        }

        if counted == 0 {
            return settings.manual_ev;
        }

        let avg_log = (weighted_sum / counted as f64) as f32;
        // Target EV is the negative of the average log luminance + compensation.
        let target_ev = -avg_log + settings.compensation;
        target_ev.clamp(settings.min_ev, settings.max_ev)
    }

    /// Smoothly adapt the current EV towards the target.
    pub fn adapt(&mut self, target_ev: f32, delta_time: f32, settings: &ExposureSettings) {
        let speed = if target_ev > self.current_ev {
            settings.adaptation_speed_up
        } else {
            settings.adaptation_speed_down
        };

        let t = 1.0 - (-speed * delta_time).exp();
        self.current_ev += (target_ev - self.current_ev) * t;
        self.current_ev = self.current_ev.clamp(settings.min_ev, settings.max_ev);
    }

    /// Convert the current EV to a linear exposure multiplier.
    pub fn exposure_multiplier(&self) -> f32 {
        // EV100 to luminance: L = 2^(EV - 3)
        // Exposure = 1 / L
        1.0 / 2.0f32.powf(self.current_ev - 3.0)
    }
}

impl Default for AutoExposureState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Gamma / sRGB
// ---------------------------------------------------------------------------

/// Convert a linear color component to sRGB gamma.
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Convert an sRGB gamma color component to linear.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

// ---------------------------------------------------------------------------
// ToneMappingSettings + ToneMappingEffect
// ---------------------------------------------------------------------------

/// Full settings for the tonemapping post-process pass.
#[derive(Debug, Clone)]
pub struct ToneMappingSettings {
    /// Which tone mapping operator to use.
    pub operator: ToneMapOperator,
    /// Auto-exposure settings.
    pub exposure: ExposureSettings,
    /// White point for Reinhard extended (units depend on scene).
    pub white_point: f32,
    /// Whether to apply sRGB gamma correction.
    pub gamma_correction: bool,
    /// Whether the effect is enabled.
    pub enabled: bool,
}

impl Default for ToneMappingSettings {
    fn default() -> Self {
        Self {
            operator: ToneMapOperator::AcesFilmic,
            exposure: ExposureSettings::default(),
            white_point: 4.0,
            gamma_correction: true,
            enabled: true,
        }
    }
}

/// Tone mapping post-process effect.
pub struct ToneMappingEffect {
    pub settings: ToneMappingSettings,
    pub exposure_state: AutoExposureState,
}

impl ToneMappingEffect {
    pub fn new(settings: ToneMappingSettings) -> Self {
        Self {
            settings,
            exposure_state: AutoExposureState::new(),
        }
    }

    /// Apply tone mapping to a single pixel (CPU reference implementation).
    pub fn tonemap_pixel(&self, hdr: [f32; 3]) -> [f32; 3] {
        // Apply exposure
        let exposure = if self.settings.exposure.enabled {
            self.exposure_state.exposure_multiplier()
        } else {
            1.0 / 2.0f32.powf(self.settings.exposure.manual_ev - 3.0)
        };

        let exposed = [
            hdr[0] * exposure,
            hdr[1] * exposure,
            hdr[2] * exposure,
        ];

        // Apply tone mapping operator
        let mapped = self.settings.operator.apply(exposed);

        // Gamma correction
        if self.settings.gamma_correction {
            [
                linear_to_srgb(mapped[0]),
                linear_to_srgb(mapped[1]),
                linear_to_srgb(mapped[2]),
            ]
        } else {
            mapped
        }
    }
}

impl PostProcessEffect for ToneMappingEffect {
    fn name(&self) -> &str {
        "ToneMapping"
    }

    fn execute(&self, _input: &PostProcessInput, _output: &mut PostProcessOutput) {
        // In a real implementation:
        // 1. If auto-exposure enabled, dispatch luminance histogram compute
        //    shader and readback average luminance.
        // 2. Adapt exposure.
        // 3. Dispatch the tonemapping compute shader which applies exposure,
        //    the selected ToneMapOperator, and gamma correction.
    }

    fn is_enabled(&self) -> bool {
        self.settings.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.settings.enabled = enabled;
    }

    fn priority(&self) -> u32 {
        600
    }

    fn on_resize(&mut self, _width: u32, _height: u32) {
        // Histogram texture may need resizing.
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

/// Luminance histogram compute shader.
/// Builds a 256-bin histogram of scene luminance values using atomic
/// operations on a storage buffer.
pub const LUMINANCE_HISTOGRAM_WGSL: &str = r#"
// Luminance histogram — compute shader
// Bins scene luminance into a 256-bin histogram using atomics.

struct HistogramParams {
    min_log_luminance: f32,
    inv_log_luminance_range: f32,
    num_pixels: u32,
    _pad: u32,
};

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>, 256>;
@group(0) @binding(2) var<uniform> params: HistogramParams;

var<workgroup> shared_histogram: array<atomic<u32>, 256>;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(16, 16, 1)
fn cs_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    // Clear shared histogram
    if lid < 256u {
        atomicStore(&shared_histogram[lid], 0u);
    }
    workgroupBarrier();

    let dims = textureDimensions(src_texture);
    if gid.x < dims.x && gid.y < dims.y {
        let color = textureLoad(src_texture, gid.xy, 0).rgb;
        let lum = luminance(color);

        var bin: u32;
        if lum < 1e-5 {
            bin = 0u;
        } else {
            let log_lum = clamp(
                (log2(lum) - params.min_log_luminance) * params.inv_log_luminance_range,
                0.0,
                1.0
            );
            bin = u32(log_lum * 254.0 + 1.0);
        }

        atomicAdd(&shared_histogram[bin], 1u);
    }
    workgroupBarrier();

    // Merge shared histogram into global histogram
    if lid < 256u {
        let val = atomicLoad(&shared_histogram[lid]);
        if val > 0u {
            atomicAdd(&histogram[lid], val);
        }
    }
}
"#;

/// Average luminance from histogram — reduces the histogram to a single
/// average luminance value using a weighted sum with percentile clipping.
pub const HISTOGRAM_AVERAGE_WGSL: &str = r#"
// Histogram average luminance — compute shader
// Computes weighted average luminance from the histogram,
// clipping low and high percentiles.

struct AverageParams {
    min_log_luminance: f32,
    log_luminance_range: f32,
    num_pixels: u32,
    low_clip_count: u32,
    high_clip_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> histogram: array<u32, 256>;
@group(0) @binding(1) var<storage, read_write> avg_luminance: f32;
@group(0) @binding(2) var<uniform> params: AverageParams;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn cs_main(@builtin(local_invocation_index) lid: u32) {
    let count = histogram[lid];
    let bin_center = (f32(lid) + 0.5) / 256.0;
    let log_lum = params.min_log_luminance + bin_center * params.log_luminance_range;

    shared_data[lid] = log_lum * f32(count);
    workgroupBarrier();

    // Parallel reduction
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if lid < stride {
            shared_data[lid] += shared_data[lid + stride];
        }
        workgroupBarrier();
    }

    if lid == 0u {
        let valid_pixels = max(params.num_pixels - params.low_clip_count - params.high_clip_count, 1u);
        let avg_log = shared_data[0] / f32(valid_pixels);
        avg_luminance = exp2(avg_log);
    }
}
"#;

/// Main tonemapping compute shader with all operator implementations.
pub const TONEMAPPING_WGSL: &str = r#"
// Tonemapping — compute shader
// Applies exposure, one of several tone-mapping curves, and gamma correction.

struct ToneMapParams {
    exposure:    f32,
    white_point: f32,
    operator:    u32,
    gamma:       u32,   // 0 = no gamma, 1 = sRGB
    inv_width:   f32,
    inv_height:  f32,
    _pad0:       f32,
    _pad1:       f32,
};

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var<uniform> params: ToneMapParams;

// ---- Reinhard ----
fn tonemap_reinhard(c: vec3<f32>) -> vec3<f32> {
    return c / (vec3<f32>(1.0) + c);
}

// ---- Reinhard extended ----
fn tonemap_reinhard_ext(c: vec3<f32>, w: f32) -> vec3<f32> {
    let w2 = w * w;
    return c * (vec3<f32>(1.0) + c / w2) / (vec3<f32>(1.0) + c);
}

// ---- ACES filmic ----
fn aces_input_matrix(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(0.59719, 0.35458, 0.04823)),
        dot(c, vec3<f32>(0.07600, 0.90834, 0.01566)),
        dot(c, vec3<f32>(0.02840, 0.13383, 0.83777)),
    );
}

fn aces_output_matrix(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>( 1.60475, -0.53108, -0.07367)),
        dot(c, vec3<f32>(-0.10208,  1.10813, -0.00605)),
        dot(c, vec3<f32>(-0.00327, -0.07276,  1.07602)),
    );
}

fn aces_rrt_odt(c: vec3<f32>) -> vec3<f32> {
    let a = c * (c + 0.0245786) - 0.000090537;
    let b = c * (0.983729 * c + 0.4329510) + 0.238081;
    return a / b;
}

fn tonemap_aces(c: vec3<f32>) -> vec3<f32> {
    let aces = aces_input_matrix(c);
    let mapped = aces_rrt_odt(aces);
    return clamp(aces_output_matrix(mapped), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ---- Uncharted 2 / Hable ----
fn uncharted2_partial(x: vec3<f32>) -> vec3<f32> {
    let A = 0.15;
    let B = 0.50;
    let C = 0.10;
    let D = 0.20;
    let E = 0.02;
    let F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

fn tonemap_uncharted2(c: vec3<f32>) -> vec3<f32> {
    let W = 11.2;
    let white_scale = vec3<f32>(1.0) / uncharted2_partial(vec3<f32>(W));
    return uncharted2_partial(c) * white_scale;
}

// ---- AgX ----
fn agx_default_contrast(x: f32) -> f32 {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 -
           6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn tonemap_agx(c: vec3<f32>) -> vec3<f32> {
    let agx_mat = mat3x3<f32>(
        vec3<f32>(0.842479, 0.0423303, 0.0423745),
        vec3<f32>(0.0784336, 0.878468, 0.0784336),
        vec3<f32>(0.0792237, 0.0791916, 0.879142),
    );
    let agx_mat_inv = mat3x3<f32>(
        vec3<f32>(1.19687, -0.0528968, -0.0529716),
        vec3<f32>(-0.0980208, 1.15190, -0.0980434),
        vec3<f32>(-0.0990297, -0.0989389, 1.15107),
    );

    let agx = agx_mat * c;
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    let range = max_ev - min_ev;

    let log_c = clamp(
        (log2(max(agx, vec3<f32>(1e-10))) - min_ev) / range,
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );

    let mapped = vec3<f32>(
        agx_default_contrast(log_c.x),
        agx_default_contrast(log_c.y),
        agx_default_contrast(log_c.z),
    );

    return clamp(agx_mat_inv * mapped, vec3<f32>(0.0), vec3<f32>(1.0));
}

// ---- Khronos PBR Neutral ----
fn tonemap_pbr_neutral(color: vec3<f32>) -> vec3<f32> {
    let start_compression = 0.8 - 0.04;
    let desaturation = 0.15;

    var c = color;
    let x = min(c.x, min(c.y, c.z));
    var offset: f32;
    if x < 0.08 {
        offset = x - 6.25 * x * x;
    } else {
        offset = 0.04;
    }
    c -= offset;

    let peak = max(c.x, max(c.y, c.z));
    if peak < start_compression {
        return c;
    }

    let d = 1.0 - start_compression;
    let new_peak = 1.0 - d * d / (peak + d - start_compression);
    c *= new_peak / peak;

    let t = clamp((peak - new_peak) / peak / desaturation, 0.0, 1.0);
    return mix(c, vec3<f32>(new_peak), t);
}

// ---- sRGB gamma ----
fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        return c * 12.92;
    }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

fn linear_to_srgb3(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(linear_to_srgb(c.x), linear_to_srgb(c.y), linear_to_srgb(c.z));
}

// ---- Main entry point ----
@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_texture);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) * vec2<f32>(params.inv_width, params.inv_height);
    var color = textureSampleLevel(src_texture, tex_sampler, uv, 0.0).rgb;

    // Apply exposure
    color *= params.exposure;

    // Apply selected tone mapping operator
    switch params.operator {
        case 0u: {
            // None — just clamp
            color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
        }
        case 1u: {
            color = tonemap_reinhard(color);
        }
        case 2u: {
            color = tonemap_reinhard_ext(color, params.white_point);
        }
        case 3u: {
            color = tonemap_aces(color);
        }
        case 4u: {
            color = tonemap_uncharted2(color);
        }
        case 5u: {
            color = tonemap_agx(color);
        }
        case 6u: {
            color = tonemap_pbr_neutral(color);
        }
        default: {
            color = tonemap_aces(color);
        }
    }

    // Apply gamma correction
    if params.gamma != 0u {
        color = linear_to_srgb3(color);
    }

    textureStore(dst_texture, gid.xy, vec4<f32>(color, 1.0));
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reinhard() {
        let result = ToneMapOperator::Reinhard.apply([1.0, 1.0, 1.0]);
        for c in &result {
            assert!((*c - 0.5).abs() < 1e-5, "Reinhard(1) should be 0.5");
        }
    }

    #[test]
    fn test_reinhard_extended() {
        let result = ToneMapOperator::ReinhardExtended.apply([0.0, 0.0, 0.0]);
        for c in &result {
            assert!(c.abs() < 1e-5);
        }
    }

    #[test]
    fn test_aces_preserves_black() {
        let result = ToneMapOperator::AcesFilmic.apply([0.0, 0.0, 0.0]);
        for c in &result {
            assert!(c.abs() < 0.01);
        }
    }

    #[test]
    fn test_uncharted2_range() {
        let result = ToneMapOperator::Uncharted2.apply([5.0, 5.0, 5.0]);
        for c in &result {
            assert!(*c >= 0.0 && *c <= 1.0);
        }
    }

    #[test]
    fn test_agx_range() {
        let result = ToneMapOperator::AgX.apply([10.0, 0.5, 0.1]);
        for c in &result {
            assert!(*c >= 0.0 && *c <= 1.0);
        }
    }

    #[test]
    fn test_pbr_neutral_range() {
        let result = ToneMapOperator::KhronosPbrNeutral.apply([3.0, 2.0, 1.0]);
        for c in &result {
            assert!(*c >= 0.0 && *c <= 1.0);
        }
    }

    #[test]
    fn test_linear_srgb_roundtrip() {
        for val in [0.0, 0.01, 0.5, 0.99, 1.0] {
            let srgb = linear_to_srgb(val);
            let back = srgb_to_linear(srgb);
            assert!(
                (val - back).abs() < 1e-4,
                "roundtrip failed for {val}: got {back}"
            );
        }
    }

    #[test]
    fn test_exposure_multiplier() {
        let state = AutoExposureState {
            current_ev: 3.0,
            ..Default::default()
        };
        let mult = state.exposure_multiplier();
        // 1 / 2^(3 - 3) = 1.0
        assert!((mult - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_adaptation() {
        let mut state = AutoExposureState::new();
        state.current_ev = 5.0;
        let settings = ExposureSettings::default();

        state.adapt(10.0, 0.016, &settings);
        assert!(state.current_ev > 5.0, "EV should increase towards target");
        assert!(state.current_ev < 10.0, "EV should not overshoot");
    }

    #[test]
    fn test_tonemapping_effect_interface() {
        let effect = ToneMappingEffect::new(ToneMappingSettings::default());
        assert_eq!(effect.name(), "ToneMapping");
        assert!(effect.is_enabled());
        assert_eq!(effect.priority(), 600);
    }
}
