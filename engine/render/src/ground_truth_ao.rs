// engine/render/src/ground_truth_ao.rs
//
// Ground Truth Ambient Occlusion (GTAO) implementation for the Genovo engine.
//
// Implements the GTAO algorithm from Jimenez et al. (2016) with extensions:
// - Multi-bounce AO approximation for energy-conserving indirect lighting
// - Bent normal computation for directional occlusion
// - Specular occlusion derived from AO and surface roughness
// - Temporal accumulation for noise reduction
// - Spatial denoising (bilateral or cross-bilateral)
// - Quality presets for scalable performance
//
// GTAO uses horizon-based methods in screen space, tracing directions in the
// view hemisphere and integrating the visible solid angle to produce a scalar
// AO term plus a bent normal direction.

use glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default AO radius in world units.
pub const DEFAULT_AO_RADIUS: f32 = 0.5;

/// Default number of directions sampled per pixel.
pub const DEFAULT_DIRECTION_COUNT: u32 = 4;

/// Default number of steps along each direction.
pub const DEFAULT_STEP_COUNT: u32 = 4;

/// Default temporal accumulation weight.
pub const DEFAULT_TEMPORAL_WEIGHT: f32 = 0.9;

/// Maximum number of samples for the denoiser kernel.
pub const MAX_DENOISE_KERNEL_SIZE: usize = 9;

/// Minimum AO value to prevent fully black areas.
pub const MIN_AO_VALUE: f32 = 0.0;

/// Small epsilon for numerical stability.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// Quality presets
// ---------------------------------------------------------------------------

/// Quality preset for GTAO rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GtaoQuality {
    /// Low quality: 2 directions, 2 steps. Suitable for mobile.
    Low,
    /// Medium quality: 4 directions, 3 steps. Good balance.
    Medium,
    /// High quality: 6 directions, 4 steps.
    High,
    /// Ultra quality: 8 directions, 6 steps. Maximum accuracy.
    Ultra,
    /// Custom quality with user-specified parameters.
    Custom,
}

impl GtaoQuality {
    /// Get the number of sample directions for this preset.
    pub fn direction_count(self) -> u32 {
        match self {
            Self::Low => 2,
            Self::Medium => 4,
            Self::High => 6,
            Self::Ultra => 8,
            Self::Custom => DEFAULT_DIRECTION_COUNT,
        }
    }

    /// Get the number of steps per direction for this preset.
    pub fn step_count(self) -> u32 {
        match self {
            Self::Low => 2,
            Self::Medium => 3,
            Self::High => 4,
            Self::Ultra => 6,
            Self::Custom => DEFAULT_STEP_COUNT,
        }
    }

    /// Get the recommended AO radius for this preset.
    pub fn radius(self) -> f32 {
        match self {
            Self::Low => 0.3,
            Self::Medium => 0.5,
            Self::High => 0.5,
            Self::Ultra => 0.7,
            Self::Custom => DEFAULT_AO_RADIUS,
        }
    }

    /// Whether to use half-resolution rendering.
    pub fn half_resolution(self) -> bool {
        matches!(self, Self::Low)
    }

    /// Whether to enable temporal accumulation.
    pub fn temporal_enabled(self) -> bool {
        !matches!(self, Self::Low)
    }

    /// Whether to compute bent normals.
    pub fn bent_normals_enabled(self) -> bool {
        matches!(self, Self::High | Self::Ultra)
    }

    /// Whether to compute multi-bounce AO.
    pub fn multi_bounce_enabled(self) -> bool {
        matches!(self, Self::High | Self::Ultra)
    }

    /// Whether to apply spatial denoising.
    pub fn denoise_enabled(self) -> bool {
        true
    }

    /// Denoise kernel radius (pixels).
    pub fn denoise_radius(self) -> u32 {
        match self {
            Self::Low => 2,
            Self::Medium => 3,
            Self::High => 4,
            Self::Ultra => 4,
            Self::Custom => 3,
        }
    }
}

// ---------------------------------------------------------------------------
// GTAO configuration
// ---------------------------------------------------------------------------

/// Full configuration for the GTAO system.
#[derive(Debug, Clone)]
pub struct GtaoConfig {
    /// Quality preset (used to initialise most parameters).
    pub quality: GtaoQuality,
    /// AO effect radius in world-space units.
    pub radius: f32,
    /// Number of directions sampled per pixel.
    pub direction_count: u32,
    /// Number of steps along each direction ray.
    pub step_count: u32,
    /// AO power/exponent (higher = darker occlusion).
    pub power: f32,
    /// AO intensity multiplier.
    pub intensity: f32,
    /// Minimum AO value (prevents fully black areas).
    pub min_ao: f32,
    /// Bias to avoid self-occlusion (in world units).
    pub bias: f32,
    /// Whether to render at half resolution.
    pub half_resolution: bool,
    /// Temporal accumulation settings.
    pub temporal: TemporalConfig,
    /// Spatial denoiser settings.
    pub denoise: DenoiseConfig,
    /// Multi-bounce AO settings.
    pub multi_bounce: MultiBounceConfig,
    /// Bent normal settings.
    pub bent_normals: BentNormalConfig,
    /// Specular occlusion settings.
    pub specular_occlusion: SpecularOcclusionConfig,
    /// Thickness heuristic for thin objects.
    pub thickness_heuristic: f32,
    /// Falloff mode for AO attenuation.
    pub falloff_mode: AoFalloffMode,
}

/// AO attenuation falloff mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AoFalloffMode {
    /// Linear falloff from center to radius.
    Linear,
    /// Smooth (cosine-based) falloff.
    Smooth,
    /// Squared falloff for sharper contact shadows.
    Squared,
    /// Exponential falloff.
    Exponential,
}

impl Default for GtaoConfig {
    fn default() -> Self {
        Self::from_quality(GtaoQuality::Medium)
    }
}

impl GtaoConfig {
    /// Create a configuration from a quality preset.
    pub fn from_quality(quality: GtaoQuality) -> Self {
        Self {
            quality,
            radius: quality.radius(),
            direction_count: quality.direction_count(),
            step_count: quality.step_count(),
            power: 1.5,
            intensity: 1.0,
            min_ao: MIN_AO_VALUE,
            bias: 0.01,
            half_resolution: quality.half_resolution(),
            temporal: TemporalConfig {
                enabled: quality.temporal_enabled(),
                weight: DEFAULT_TEMPORAL_WEIGHT,
                variance_clamp: 1.5,
                motion_rejection: true,
                motion_threshold: 0.01,
            },
            denoise: DenoiseConfig {
                enabled: quality.denoise_enabled(),
                radius: quality.denoise_radius(),
                sharpness: 8.0,
                normal_weight: 1.0,
                depth_weight: 1.0,
                bilateral: true,
                passes: if quality == GtaoQuality::Ultra { 2 } else { 1 },
            },
            multi_bounce: MultiBounceConfig {
                enabled: quality.multi_bounce_enabled(),
                albedo_influence: 0.8,
                bounce_power: 0.5,
            },
            bent_normals: BentNormalConfig {
                enabled: quality.bent_normals_enabled(),
                normalize: true,
                lerp_with_surface_normal: 0.5,
            },
            specular_occlusion: SpecularOcclusionConfig {
                enabled: true,
                strength: 1.0,
                roughness_influence: true,
            },
            thickness_heuristic: 0.5,
            falloff_mode: AoFalloffMode::Smooth,
        }
    }

    /// Total number of samples per pixel.
    pub fn total_samples(&self) -> u32 {
        self.direction_count * self.step_count
    }

    /// Estimated cost relative to the Low preset.
    pub fn relative_cost(&self) -> f32 {
        let base = GtaoQuality::Low.direction_count() * GtaoQuality::Low.step_count();
        let current = self.total_samples();
        let res_factor = if self.half_resolution { 0.25 } else { 1.0 };
        (current as f32 / base as f32) * res_factor
    }
}

// ---------------------------------------------------------------------------
// Temporal accumulation
// ---------------------------------------------------------------------------

/// Temporal accumulation configuration.
#[derive(Debug, Clone)]
pub struct TemporalConfig {
    /// Whether temporal accumulation is enabled.
    pub enabled: bool,
    /// Blend weight for history (0 = no history, 1 = only history).
    pub weight: f32,
    /// Variance clamp factor for neighbourhood clamping.
    pub variance_clamp: f32,
    /// Whether to reject history based on motion vectors.
    pub motion_rejection: bool,
    /// Motion vector magnitude threshold for rejection.
    pub motion_threshold: f32,
}

/// Runtime state for temporal accumulation.
#[derive(Debug)]
pub struct TemporalAccumulator {
    /// Configuration.
    pub config: TemporalConfig,
    /// Handle to the previous frame's AO texture.
    pub history_ao_rt: u64,
    /// Handle to the previous frame's bent normal texture.
    pub history_bent_normal_rt: u64,
    /// Previous frame's view-projection matrix (for reprojection).
    pub prev_view_projection: Mat4,
    /// Current frame's view-projection matrix.
    pub curr_view_projection: Mat4,
    /// Jitter offset for the current frame (sub-pixel jitter for TAA integration).
    pub jitter_offset: Vec2,
    /// Frame counter for jitter sequence.
    pub frame_count: u64,
    /// Resolution of the AO buffer.
    pub resolution: UVec2,
    /// Whether history is valid (invalid after resize, first frame, etc.).
    pub history_valid: bool,
}

impl TemporalAccumulator {
    /// Create a new temporal accumulator.
    pub fn new(config: TemporalConfig, width: u32, height: u32) -> Self {
        Self {
            config,
            history_ao_rt: 0,
            history_bent_normal_rt: 0,
            prev_view_projection: Mat4::IDENTITY,
            curr_view_projection: Mat4::IDENTITY,
            jitter_offset: Vec2::ZERO,
            frame_count: 0,
            resolution: UVec2::new(width, height),
            history_valid: false,
        }
    }

    /// Begin a new frame: swap history buffers and update matrices.
    pub fn begin_frame(&mut self, view_projection: Mat4) {
        self.prev_view_projection = self.curr_view_projection;
        self.curr_view_projection = view_projection;
        self.frame_count += 1;

        // Generate sub-pixel jitter from Halton sequence
        self.jitter_offset = halton_2d(self.frame_count);
    }

    /// Invalidate history (e.g. on camera cut or resize).
    pub fn invalidate(&mut self) {
        self.history_valid = false;
    }

    /// Mark history as valid after a successful accumulation pass.
    pub fn mark_valid(&mut self) {
        self.history_valid = true;
    }

    /// Resize the accumulator.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.resolution = UVec2::new(width, height);
        self.invalidate();
    }

    /// Compute the reprojected UV for a pixel from the current to the previous frame.
    pub fn reproject_uv(&self, uv: Vec2, depth: f32) -> Vec2 {
        // Current frame clip space
        let ndc = Vec4::new(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
        let inv_vp = self.curr_view_projection.inverse();
        let world = inv_vp * ndc;
        let world = world / world.w;

        // Previous frame clip space
        let prev_clip = self.prev_view_projection * world;
        if prev_clip.w.abs() < EPSILON {
            return uv;
        }
        let prev_ndc = prev_clip / prev_clip.w;
        Vec2::new(
            (prev_ndc.x + 1.0) * 0.5,
            1.0 - (prev_ndc.y + 1.0) * 0.5,
        )
    }

    /// Apply temporal accumulation to a single AO sample.
    pub fn accumulate(&self, current_ao: f32, history_ao: f32, motion_length: f32) -> f32 {
        if !self.config.enabled || !self.history_valid {
            return current_ao;
        }

        let mut weight = self.config.weight;

        // Reduce weight when there's significant motion
        if self.config.motion_rejection && motion_length > self.config.motion_threshold {
            let motion_factor = (1.0 - motion_length / (self.config.motion_threshold * 10.0)).max(0.0);
            weight *= motion_factor;
        }

        current_ao * (1.0 - weight) + history_ao * weight
    }

    /// Apply neighbourhood clamping to the history value.
    pub fn clamp_history(
        &self,
        history_ao: f32,
        neighbourhood_mean: f32,
        neighbourhood_variance: f32,
    ) -> f32 {
        let sigma = neighbourhood_variance.sqrt() * self.config.variance_clamp;
        let min_val = neighbourhood_mean - sigma;
        let max_val = neighbourhood_mean + sigma;
        history_ao.clamp(min_val, max_val)
    }
}

/// Generate a 2D Halton sequence sample.
fn halton_2d(index: u64) -> Vec2 {
    Vec2::new(halton_base(index, 2), halton_base(index, 3))
}

/// Compute Halton sequence value for a given base.
fn halton_base(mut index: u64, base: u64) -> f32 {
    let mut result = 0.0f32;
    let mut f = 1.0f32 / base as f32;
    let mut i = index;
    while i > 0 {
        result += f * (i % base) as f32;
        i /= base;
        f /= base as f32;
    }
    result
}

// ---------------------------------------------------------------------------
// Spatial denoising
// ---------------------------------------------------------------------------

/// Spatial denoising configuration.
#[derive(Debug, Clone)]
pub struct DenoiseConfig {
    /// Whether denoising is enabled.
    pub enabled: bool,
    /// Kernel radius in pixels.
    pub radius: u32,
    /// Sharpness parameter (higher = less blurring).
    pub sharpness: f32,
    /// Weight for normal-based rejection.
    pub normal_weight: f32,
    /// Weight for depth-based rejection.
    pub depth_weight: f32,
    /// Use bilateral (cross-bilateral with normals+depth) filtering.
    pub bilateral: bool,
    /// Number of filter passes.
    pub passes: u32,
}

/// A spatial denoiser for AO buffers.
#[derive(Debug)]
pub struct GtaoDenoiser {
    /// Configuration.
    pub config: DenoiseConfig,
    /// Intermediate buffer handle (for multi-pass).
    pub intermediate_rt: u64,
    /// Gaussian kernel weights (precomputed).
    pub kernel_weights: Vec<f32>,
    /// Resolution.
    pub resolution: UVec2,
}

impl GtaoDenoiser {
    /// Create a new denoiser.
    pub fn new(config: DenoiseConfig, width: u32, height: u32) -> Self {
        let kernel_weights = Self::compute_gaussian_kernel(config.radius, config.sharpness);
        Self {
            config,
            intermediate_rt: 0,
            kernel_weights,
            resolution: UVec2::new(width, height),
        }
    }

    /// Compute Gaussian kernel weights.
    fn compute_gaussian_kernel(radius: u32, sharpness: f32) -> Vec<f32> {
        let size = (radius * 2 + 1) as usize;
        let mut weights = vec![0.0f32; size];
        let sigma = radius as f32 / sharpness.max(0.1);
        let two_sigma_sq = 2.0 * sigma * sigma;
        let mut sum = 0.0f32;

        for i in 0..size {
            let offset = i as f32 - radius as f32;
            let w = (-offset * offset / two_sigma_sq).exp();
            weights[i] = w;
            sum += w;
        }

        // Normalize
        if sum > EPSILON {
            for w in &mut weights {
                *w /= sum;
            }
        }

        weights
    }

    /// Apply a single horizontal or vertical filter pass (CPU reference).
    pub fn filter_pass(
        &self,
        input: &[f32],
        output: &mut [f32],
        width: u32,
        height: u32,
        horizontal: bool,
        normals: Option<&[Vec3]>,
        depths: Option<&[f32]>,
    ) {
        let radius = self.config.radius as i32;

        for y in 0..height {
            for x in 0..width {
                let center_idx = (y * width + x) as usize;
                let center_normal = normals.map(|n| n[center_idx]);
                let center_depth = depths.map(|d| d[center_idx]);

                let mut weighted_sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                for k in -radius..=radius {
                    let sx = if horizontal { x as i32 + k } else { x as i32 };
                    let sy = if horizontal { y as i32 } else { y as i32 + k };

                    if sx < 0 || sx >= width as i32 || sy < 0 || sy >= height as i32 {
                        continue;
                    }

                    let sample_idx = (sy as u32 * width + sx as u32) as usize;
                    let kernel_idx = (k + radius) as usize;

                    let mut weight = if kernel_idx < self.kernel_weights.len() {
                        self.kernel_weights[kernel_idx]
                    } else {
                        0.0
                    };

                    // Bilateral: modulate weight by normal similarity
                    if self.config.bilateral {
                        if let (Some(cn), Some(ns)) = (center_normal, normals) {
                            let sn = ns[sample_idx];
                            let dot = cn.dot(sn).max(0.0);
                            weight *= dot.powf(self.config.normal_weight * 32.0);
                        }
                        if let (Some(cd), Some(ds)) = (center_depth, depths) {
                            let sd = ds[sample_idx];
                            let depth_diff = (cd - sd).abs();
                            let depth_factor = (-depth_diff * self.config.depth_weight * 100.0).exp();
                            weight *= depth_factor;
                        }
                    }

                    weighted_sum += input[sample_idx] * weight;
                    weight_sum += weight;
                }

                output[center_idx] = if weight_sum > EPSILON {
                    weighted_sum / weight_sum
                } else {
                    input[center_idx]
                };
            }
        }
    }

    /// Resize the denoiser.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.resolution = UVec2::new(width, height);
    }

    /// Reconfigure the kernel.
    pub fn reconfigure(&mut self, config: DenoiseConfig) {
        self.kernel_weights = Self::compute_gaussian_kernel(config.radius, config.sharpness);
        self.config = config;
    }
}

// ---------------------------------------------------------------------------
// Multi-bounce AO
// ---------------------------------------------------------------------------

/// Multi-bounce AO configuration.
#[derive(Debug, Clone)]
pub struct MultiBounceConfig {
    /// Whether multi-bounce is enabled.
    pub enabled: bool,
    /// How much the surface albedo influences the multi-bounce term.
    pub albedo_influence: f32,
    /// Power applied to the bounce term.
    pub bounce_power: f32,
}

/// Compute multi-bounce AO from single-bounce AO and surface albedo.
///
/// Uses the Jimenez et al. approximation to recover energy lost from
/// single-bounce AO by accounting for inter-reflected light.
pub fn multi_bounce_ao(ao: f32, albedo: Vec3, config: &MultiBounceConfig) -> Vec3 {
    if !config.enabled {
        return Vec3::splat(ao);
    }

    let a = 2.0404 * albedo * config.albedo_influence - 0.3324;
    let b = -4.7951 * albedo * config.albedo_influence + 0.6417;
    let c = 2.7552 * albedo * config.albedo_influence + 0.6903;

    let ao_pow = ao.powf(config.bounce_power);

    Vec3::new(
        (a.x * ao_pow * ao_pow + b.x * ao_pow + c.x).max(ao),
        (a.y * ao_pow * ao_pow + b.y * ao_pow + c.y).max(ao),
        (a.z * ao_pow * ao_pow + b.z * ao_pow + c.z).max(ao),
    )
}

/// Simplified multi-bounce AO using a scalar albedo.
pub fn multi_bounce_ao_scalar(ao: f32, albedo: f32, config: &MultiBounceConfig) -> f32 {
    if !config.enabled {
        return ao;
    }

    let a_albedo = albedo * config.albedo_influence;
    let a = 2.0404 * a_albedo - 0.3324;
    let b = -4.7951 * a_albedo + 0.6417;
    let c = 2.7552 * a_albedo + 0.6903;

    let ao_pow = ao.powf(config.bounce_power);
    (a * ao_pow * ao_pow + b * ao_pow + c).max(ao)
}

// ---------------------------------------------------------------------------
// Bent normals
// ---------------------------------------------------------------------------

/// Bent normal configuration.
#[derive(Debug, Clone)]
pub struct BentNormalConfig {
    /// Whether bent normal computation is enabled.
    pub enabled: bool,
    /// Whether to normalize the bent normal.
    pub normalize: bool,
    /// Lerp factor between bent normal and surface normal (0 = bent, 1 = surface).
    pub lerp_with_surface_normal: f32,
}

/// Result of a bent normal computation for a pixel.
#[derive(Debug, Clone, Copy)]
pub struct BentNormalResult {
    /// The computed bent normal direction (world space).
    pub bent_normal: Vec3,
    /// The AO value.
    pub ao: f32,
    /// The cone angle (in radians) representing the unoccluded cone.
    pub cone_angle: f32,
}

impl BentNormalResult {
    /// Compute the bent normal from sampled horizon angles.
    ///
    /// The bent normal points toward the average unoccluded direction
    /// in the hemisphere around the surface normal.
    pub fn compute(
        surface_normal: Vec3,
        sample_directions: &[Vec3],
        horizon_cos_angles: &[f32],
        config: &BentNormalConfig,
    ) -> Self {
        if !config.enabled || sample_directions.is_empty() {
            return Self {
                bent_normal: surface_normal,
                ao: 1.0,
                cone_angle: PI * 0.5,
            };
        }

        let mut accumulated = Vec3::ZERO;
        let mut ao_sum = 0.0f32;
        let n = sample_directions.len();

        for i in 0..n {
            let dir = sample_directions[i];
            let cos_horizon = horizon_cos_angles[i].clamp(-1.0, 1.0);
            let unoccluded = cos_horizon.acos(); // angle of unoccluded hemisphere portion

            // Weight by the unoccluded solid angle
            let weight = unoccluded / (PI * 0.5);
            accumulated += dir * weight;
            ao_sum += weight;
        }

        let ao = (ao_sum / n as f32).clamp(0.0, 1.0);

        let mut bent_normal = if accumulated.length_squared() > EPSILON {
            if config.normalize {
                accumulated.normalize()
            } else {
                accumulated / n as f32
            }
        } else {
            surface_normal
        };

        // Lerp with surface normal
        if config.lerp_with_surface_normal > 0.0 {
            bent_normal = Vec3::lerp(bent_normal, surface_normal, config.lerp_with_surface_normal);
            if config.normalize {
                bent_normal = bent_normal.normalize_or_zero();
            }
        }

        let cone_angle = ao * PI * 0.5;

        Self {
            bent_normal,
            ao,
            cone_angle,
        }
    }

    /// Use the bent normal for indirect lighting direction.
    pub fn indirect_lighting_direction(&self) -> Vec3 {
        self.bent_normal
    }
}

// ---------------------------------------------------------------------------
// Specular occlusion
// ---------------------------------------------------------------------------

/// Specular occlusion configuration.
#[derive(Debug, Clone)]
pub struct SpecularOcclusionConfig {
    /// Whether specular occlusion is enabled.
    pub enabled: bool,
    /// Strength of the specular occlusion effect.
    pub strength: f32,
    /// Whether roughness influences specular occlusion.
    pub roughness_influence: bool,
}

/// Compute specular occlusion from AO, bent normal, reflection vector, and roughness.
///
/// Uses the Lagarde & de Rousiers approximation.
pub fn specular_occlusion(
    ao: f32,
    bent_normal: Vec3,
    reflection: Vec3,
    roughness: f32,
    config: &SpecularOcclusionConfig,
) -> f32 {
    if !config.enabled {
        return 1.0;
    }

    let ndotv = bent_normal.dot(reflection).max(0.0);

    let roughness_factor = if config.roughness_influence {
        roughness
    } else {
        0.5
    };

    // Lagarde approximation
    let spec_occ = (ndotv + ao * ao - 1.0).max(0.0) / (ndotv + EPSILON);
    let adjusted = spec_occ.powf(1.0 / (1.0 + roughness_factor * 4.0));

    let result = ao.max(adjusted);
    1.0 - (1.0 - result) * config.strength
}

/// Compute a simplified specular occlusion from AO and roughness only.
pub fn specular_occlusion_simple(ao: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let spec_occ = ao.powf(1.0 / (1.0 + a * a * 50.0));
    spec_occ
}

// ---------------------------------------------------------------------------
// GTAO sample computation (CPU reference)
// ---------------------------------------------------------------------------

/// Per-pixel GTAO parameters for the horizon search.
#[derive(Debug, Clone, Copy)]
pub struct GtaoPixelParams {
    /// View-space position of the pixel.
    pub position: Vec3,
    /// View-space normal of the pixel.
    pub normal: Vec3,
    /// Screen UV of the pixel.
    pub uv: Vec2,
    /// Linear depth of the pixel.
    pub depth: f32,
}

/// Result of GTAO computation for a single pixel.
#[derive(Debug, Clone, Copy)]
pub struct GtaoPixelResult {
    /// Ambient occlusion value (0 = fully occluded, 1 = no occlusion).
    pub ao: f32,
    /// Bent normal (optional, zero if not computed).
    pub bent_normal: Vec3,
    /// Specular occlusion.
    pub specular_occlusion: f32,
}

/// Compute the AO falloff weight based on distance.
pub fn ao_falloff(distance: f32, radius: f32, mode: AoFalloffMode) -> f32 {
    let t = (distance / radius).clamp(0.0, 1.0);
    match mode {
        AoFalloffMode::Linear => 1.0 - t,
        AoFalloffMode::Smooth => {
            let s = 1.0 - t;
            s * s * (3.0 - 2.0 * s)
        }
        AoFalloffMode::Squared => {
            let s = 1.0 - t;
            s * s
        }
        AoFalloffMode::Exponential => {
            (-t * 3.0).exp()
        }
    }
}

/// Compute the cosine of the horizon angle for a direction.
///
/// This searches along a screen-space direction to find the highest
/// occluder horizon angle.
pub fn compute_horizon_angle(
    pixel: &GtaoPixelParams,
    direction: Vec2,
    step_count: u32,
    radius: f32,
    bias: f32,
    depth_buffer: &dyn Fn(Vec2) -> f32,
    position_from_depth: &dyn Fn(Vec2, f32) -> Vec3,
) -> f32 {
    let mut max_cos_horizon = -1.0f32;
    let step_size = radius / step_count as f32;

    for step in 1..=step_count {
        let offset = direction * (step as f32 * step_size);
        let sample_uv = pixel.uv + offset;

        // Clamp to screen
        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            break;
        }

        let sample_depth = depth_buffer(sample_uv);
        let sample_pos = position_from_depth(sample_uv, sample_depth);
        let delta = sample_pos - pixel.position;
        let dist = delta.length();

        if dist < EPSILON {
            continue;
        }

        let cos_angle = delta.dot(pixel.normal) / dist;
        let biased_cos = cos_angle - bias;

        if biased_cos > max_cos_horizon {
            max_cos_horizon = biased_cos;
        }
    }

    max_cos_horizon
}

/// Integrate the AO from horizon angles on both sides of a direction.
pub fn integrate_ao(cos_horizon_front: f32, cos_horizon_back: f32, cos_normal: f32) -> f32 {
    let h1 = cos_horizon_front.acos();
    let h2 = cos_horizon_back.acos();
    let n = cos_normal.acos();

    // Integrate the visible arc
    let inner_integral = |h: f32, n: f32| -> f32 {
        0.25 * (-(2.0 * h - n).cos() + (n).cos() + 2.0 * h * (n).sin())
    };

    let ao = inner_integral(h1, n) + inner_integral(h2, n);
    ao.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// GTAO system (ECS component)
// ---------------------------------------------------------------------------

/// The main GTAO system that ties everything together.
#[derive(Debug)]
pub struct GtaoSystem {
    /// Configuration.
    pub config: GtaoConfig,
    /// Temporal accumulator.
    pub temporal: TemporalAccumulator,
    /// Spatial denoiser.
    pub denoiser: GtaoDenoiser,
    /// AO render target handle.
    pub ao_rt: u64,
    /// Bent normal render target handle (if enabled).
    pub bent_normal_rt: u64,
    /// Resolution.
    pub resolution: UVec2,
    /// Effective resolution (may be half if half_resolution is enabled).
    pub effective_resolution: UVec2,
    /// Frame index.
    pub frame_index: u64,
    /// Statistics from the last frame.
    pub stats: GtaoStats,
}

/// GTAO statistics.
#[derive(Debug, Clone, Default)]
pub struct GtaoStats {
    /// Total pixel count processed.
    pub pixels_processed: u64,
    /// Total samples taken.
    pub samples_taken: u64,
    /// Average AO value across all pixels.
    pub avg_ao: f32,
    /// Minimum AO value.
    pub min_ao: f32,
    /// Maximum AO value.
    pub max_ao: f32,
    /// Processing time in microseconds.
    pub processing_time_us: f64,
    /// Whether temporal history was valid this frame.
    pub temporal_valid: bool,
}

impl GtaoSystem {
    /// Create a new GTAO system.
    pub fn new(config: GtaoConfig, width: u32, height: u32) -> Self {
        let effective_res = if config.half_resolution {
            UVec2::new(width / 2, height / 2)
        } else {
            UVec2::new(width, height)
        };
        Self {
            temporal: TemporalAccumulator::new(config.temporal.clone(), effective_res.x, effective_res.y),
            denoiser: GtaoDenoiser::new(config.denoise.clone(), effective_res.x, effective_res.y),
            config,
            ao_rt: 0,
            bent_normal_rt: 0,
            resolution: UVec2::new(width, height),
            effective_resolution: effective_res,
            frame_index: 0,
            stats: GtaoStats::default(),
        }
    }

    /// Resize the system.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.resolution = UVec2::new(width, height);
        self.effective_resolution = if self.config.half_resolution {
            UVec2::new(width / 2, height / 2)
        } else {
            UVec2::new(width, height)
        };
        self.temporal.resize(self.effective_resolution.x, self.effective_resolution.y);
        self.denoiser.resize(self.effective_resolution.x, self.effective_resolution.y);
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self, view_projection: Mat4) {
        self.frame_index += 1;
        self.temporal.begin_frame(view_projection);
        self.stats = GtaoStats::default();
    }

    /// Set the quality preset (reconfigures all parameters).
    pub fn set_quality(&mut self, quality: GtaoQuality) {
        self.config = GtaoConfig::from_quality(quality);
        let (w, h) = (self.resolution.x, self.resolution.y);
        self.resize(w, h);
        self.denoiser.reconfigure(self.config.denoise.clone());
    }

    /// Get the current quality preset.
    pub fn quality(&self) -> GtaoQuality {
        self.config.quality
    }

    /// Whether bent normals are currently being computed.
    pub fn has_bent_normals(&self) -> bool {
        self.config.bent_normals.enabled
    }

    /// Whether multi-bounce AO is enabled.
    pub fn has_multi_bounce(&self) -> bool {
        self.config.multi_bounce.enabled
    }

    /// Compute multi-bounce AO using the current configuration.
    pub fn apply_multi_bounce(&self, ao: f32, albedo: Vec3) -> Vec3 {
        multi_bounce_ao(ao, albedo, &self.config.multi_bounce)
    }

    /// Compute specular occlusion using the current configuration.
    pub fn apply_specular_occlusion(
        &self,
        ao: f32,
        bent_normal: Vec3,
        reflection: Vec3,
        roughness: f32,
    ) -> f32 {
        specular_occlusion(ao, bent_normal, reflection, roughness, &self.config.specular_occlusion)
    }
}

// ---------------------------------------------------------------------------
// ECS component wrapper
// ---------------------------------------------------------------------------

/// ECS component for attaching GTAO to a camera entity.
#[derive(Debug)]
pub struct GtaoComponent {
    /// Whether the component is enabled.
    pub enabled: bool,
    /// GTAO system (boxed for component storage).
    pub system: GtaoSystem,
}

impl GtaoComponent {
    /// Create a new GTAO component with default settings.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            enabled: true,
            system: GtaoSystem::new(GtaoConfig::default(), width, height),
        }
    }

    /// Create with a specific quality preset.
    pub fn with_quality(width: u32, height: u32, quality: GtaoQuality) -> Self {
        Self {
            enabled: true,
            system: GtaoSystem::new(GtaoConfig::from_quality(quality), width, height),
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
    fn test_quality_presets() {
        for quality in [GtaoQuality::Low, GtaoQuality::Medium, GtaoQuality::High, GtaoQuality::Ultra] {
            let config = GtaoConfig::from_quality(quality);
            assert!(config.direction_count >= 2);
            assert!(config.step_count >= 2);
            assert!(config.radius > 0.0);
        }
    }

    #[test]
    fn test_multi_bounce_ao() {
        let config = MultiBounceConfig {
            enabled: true,
            albedo_influence: 0.8,
            bounce_power: 0.5,
        };
        let result = multi_bounce_ao(0.5, Vec3::new(0.8, 0.2, 0.1), &config);
        // Multi-bounce should be >= single bounce
        assert!(result.x >= 0.5);
    }

    #[test]
    fn test_ao_falloff() {
        assert!((ao_falloff(0.0, 1.0, AoFalloffMode::Linear) - 1.0).abs() < EPSILON);
        assert!((ao_falloff(1.0, 1.0, AoFalloffMode::Linear) - 0.0).abs() < EPSILON);
        assert!((ao_falloff(0.5, 1.0, AoFalloffMode::Smooth) - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_halton_sequence() {
        let s0 = halton_2d(1);
        let s1 = halton_2d(2);
        assert!(s0.x >= 0.0 && s0.x <= 1.0);
        assert!(s1.y >= 0.0 && s1.y <= 1.0);
        // Different samples should be different
        assert!((s0 - s1).length() > EPSILON);
    }

    #[test]
    fn test_specular_occlusion() {
        let config = SpecularOcclusionConfig {
            enabled: true,
            strength: 1.0,
            roughness_influence: true,
        };
        let so = specular_occlusion(1.0, Vec3::Y, Vec3::Y, 0.5, &config);
        assert!((so - 1.0).abs() < 0.01); // No occlusion when AO=1

        let so_occluded = specular_occlusion(0.0, Vec3::Y, Vec3::Y, 0.5, &config);
        assert!(so_occluded < 0.5); // Should be heavily occluded
    }

    #[test]
    fn test_bent_normal_result() {
        let config = BentNormalConfig {
            enabled: true,
            normalize: true,
            lerp_with_surface_normal: 0.0,
        };
        let directions = vec![Vec3::Y, Vec3::X, Vec3::Z];
        let horizons = vec![0.5, 0.3, 0.7];
        let result = BentNormalResult::compute(Vec3::Y, &directions, &horizons, &config);
        assert!(result.ao >= 0.0 && result.ao <= 1.0);
        assert!((result.bent_normal.length() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_temporal_accumulator() {
        let config = TemporalConfig {
            enabled: true,
            weight: 0.9,
            variance_clamp: 1.5,
            motion_rejection: true,
            motion_threshold: 0.01,
        };
        let acc = TemporalAccumulator::new(config, 100, 100);
        let blended = acc.accumulate(0.5, 0.8, 0.0);
        // History not valid, should return current
        assert!((blended - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_gaussian_kernel() {
        let weights = GtaoDenoiser::compute_gaussian_kernel(3, 8.0);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
        // Center should be the largest weight
        assert!(weights[3] >= weights[0]);
    }

    #[test]
    fn test_gtao_system_lifecycle() {
        let mut sys = GtaoSystem::new(GtaoConfig::default(), 800, 600);
        sys.begin_frame(Mat4::IDENTITY);
        assert_eq!(sys.frame_index, 1);
        sys.set_quality(GtaoQuality::Ultra);
        assert_eq!(sys.quality(), GtaoQuality::Ultra);
        assert!(sys.has_bent_normals());
        assert!(sys.has_multi_bounce());
    }
}
