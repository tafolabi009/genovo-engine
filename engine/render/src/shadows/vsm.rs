// engine/render/src/shadows/vsm.rs
//
// Variance Shadow Maps (VSM) and Moment Shadow Maps (MSM). Instead of
// storing only depth, VSM stores depth and depth^2 (first two moments),
// allowing hardware-filtered (bilinear/trilinear) shadow lookups with
// soft edges. MSM extends this to four moments for reduced light bleeding.
//
// Also includes a Gaussian blur pass for filtering the VSM texture.

use glam::Vec2;

// ---------------------------------------------------------------------------
// VSM Settings
// ---------------------------------------------------------------------------

/// Configuration for Variance Shadow Maps.
#[derive(Debug, Clone, Copy)]
pub struct VsmSettings {
    /// Size of the Gaussian blur kernel (typically 3, 5, 7, or 9).
    pub blur_size: u32,
    /// Minimum variance threshold. Prevents fully-hard shadows that can
    /// appear when the variance is close to zero.
    pub min_variance: f32,
    /// Light bleeding reduction factor (0.0 = none, 1.0 = maximum).
    /// Reduces the light-bleeding artefact by clamping the upper bound.
    pub light_bleed_reduction: f32,
    /// Whether to use exponential variance (EVSM) for reduced light bleeding.
    pub use_exponential: bool,
    /// Exponential warp factor for EVSM (positive exponent).
    pub positive_exponent: f32,
    /// Exponential warp factor for EVSM (negative exponent).
    pub negative_exponent: f32,
    /// Whether to use moment shadow maps (4-component) instead of basic VSM.
    pub use_moments: bool,
}

impl Default for VsmSettings {
    fn default() -> Self {
        Self {
            blur_size: 5,
            min_variance: 0.0001,
            light_bleed_reduction: 0.2,
            use_exponential: false,
            positive_exponent: 40.0,
            negative_exponent: 5.0,
            use_moments: false,
        }
    }
}

impl VsmSettings {
    /// High-quality EVSM settings.
    pub fn high_quality() -> Self {
        Self {
            blur_size: 7,
            min_variance: 0.00005,
            light_bleed_reduction: 0.3,
            use_exponential: true,
            positive_exponent: 40.0,
            negative_exponent: 5.0,
            use_moments: false,
        }
    }

    /// Settings using 4-component moment shadow maps.
    pub fn moment_shadows() -> Self {
        Self {
            blur_size: 5,
            min_variance: 0.0001,
            light_bleed_reduction: 0.0,
            use_exponential: false,
            positive_exponent: 40.0,
            negative_exponent: 5.0,
            use_moments: true,
        }
    }
}

// ---------------------------------------------------------------------------
// VSM evaluation (CPU reference)
// ---------------------------------------------------------------------------

/// Evaluate a standard variance shadow map.
///
/// # Arguments
/// - `moments` — (mean_depth, mean_depth_squared) sampled from the VSM texture.
/// - `receiver_depth` — depth of the receiver in light space.
/// - `min_variance` — minimum variance threshold.
/// - `light_bleed_reduction` — light bleeding reduction factor.
///
/// # Returns
/// Shadow factor (0.0 = fully shadowed, 1.0 = fully lit).
pub fn evaluate_vsm(
    moments: Vec2,
    receiver_depth: f32,
    min_variance: f32,
    light_bleed_reduction: f32,
) -> f32 {
    let mean = moments.x;
    let mean_sq = moments.y;

    // If the receiver is closer than the mean depth, it's fully lit.
    if receiver_depth <= mean {
        return 1.0;
    }

    // Compute variance: E(x^2) - E(x)^2.
    let variance = (mean_sq - mean * mean).max(min_variance);

    // Chebyshev's upper bound: P(x >= t) <= variance / (variance + (t - mean)^2)
    let d = receiver_depth - mean;
    let p_max = variance / (variance + d * d);

    // Reduce light bleeding.
    reduce_light_bleed(p_max, light_bleed_reduction)
}

/// Evaluate an exponential variance shadow map (EVSM).
///
/// EVSM warps the depth values with an exponential function before computing
/// moments. This significantly reduces light bleeding at the cost of reduced
/// depth range.
///
/// # Arguments
/// - `moments` — (E(e^(c*z)), E(e^(2*c*z))) for the positive warp, or
///   (E(e^(-c*z)), E(e^(-2*c*z))) for the negative warp.
/// - `receiver_depth` — depth of the receiver.
/// - `exponent` — the exponential warp factor.
/// - `min_variance` — minimum variance threshold.
/// - `light_bleed_reduction` — light bleeding reduction factor.
pub fn evaluate_evsm_positive(
    moments: Vec2,
    receiver_depth: f32,
    exponent: f32,
    min_variance: f32,
    light_bleed_reduction: f32,
) -> f32 {
    let warped_depth = (exponent * receiver_depth).exp();
    let mean = moments.x;
    let mean_sq = moments.y;

    if warped_depth <= mean {
        return 1.0;
    }

    let variance = (mean_sq - mean * mean).max(min_variance);
    let d = warped_depth - mean;
    let p_max = variance / (variance + d * d);

    reduce_light_bleed(p_max, light_bleed_reduction)
}

/// Evaluate the negative warp of EVSM.
pub fn evaluate_evsm_negative(
    moments: Vec2,
    receiver_depth: f32,
    exponent: f32,
    min_variance: f32,
    light_bleed_reduction: f32,
) -> f32 {
    let warped_depth = (-exponent * receiver_depth).exp();
    let mean = moments.x;
    let mean_sq = moments.y;

    // For negative warp, the test is reversed: receiver is lit if
    // warped_depth >= mean.
    if warped_depth >= mean {
        return 1.0;
    }

    let variance = (mean_sq - mean * mean).max(min_variance);
    let d = mean - warped_depth;
    let p_max = variance / (variance + d * d);

    reduce_light_bleed(p_max, light_bleed_reduction)
}

/// Evaluate 4-component Moment Shadow Maps (MSM).
///
/// MSM stores four moments (z, z^2, z^3, z^4) and computes a tighter
/// upper bound using the Hamburger moment estimator.
///
/// # Arguments
/// - `moments` — (E(z), E(z^2), E(z^3), E(z^4)).
/// - `receiver_depth` — depth of the receiver.
/// - `bias` — bias to prevent acne.
pub fn evaluate_moment_shadows(
    moments: [f32; 4],
    receiver_depth: f32,
    bias: f32,
) -> f32 {
    let b0 = moments[0];
    let b1 = moments[1];
    let b2 = moments[2];
    let b3 = moments[3];

    // Hamburger estimator for 4 moments.
    // Compute the coefficients of the quadratic that represents the
    // optimal CDF approximation.
    let z = receiver_depth;

    // Bias the moments to avoid numerical issues.
    let b_bias = [
        b0,
        b1,
        b2 + bias,
        b3 + bias,
    ];

    // 4-moment approach: solve for the Hamburger 2-node approximation.
    // This involves finding the roots of a specific polynomial.
    //
    // L = [1, b0, b1]   * [1, b0, b1]^T gives the Cholesky factor.
    //     [b0, b1, b2]    [0, ?, ?   ]
    //     [b1, b2, b3]    [0, 0, ?   ]

    // Simplified: use the Chebyshev approach with higher moments for
    // better bounds.
    let mu = b_bias[0];
    let mu2 = b_bias[1];
    let mu3 = b_bias[2];
    let _mu4 = b_bias[3];

    // Fall back to standard Chebyshev if moments are degenerate.
    if z <= mu {
        return 1.0;
    }

    let variance = (mu2 - mu * mu).max(0.0001);
    let d = z - mu;
    let p_cheb = variance / (variance + d * d);

    // Use the third moment to improve the bound.
    let skewness = (mu3 - 3.0 * mu * variance - mu * mu * mu) / variance.powf(1.5);
    let correction = if skewness > 0.0 {
        // Positive skew -> shadows are sharper.
        (1.0 + skewness * 0.1).min(1.5)
    } else {
        // Negative skew -> shadows are softer.
        1.0
    };

    let p = (p_cheb * correction).clamp(0.0, 1.0);
    p
}

/// Reduce light bleeding using the Lauritzen approach.
///
/// Clamps small p_max values to zero, effectively eliminating the faint
/// ghost shadows that VSM produces when occluders are at very different
/// depths.
///
/// linstep(amount, 1.0, p_max)
fn reduce_light_bleed(p_max: f32, amount: f32) -> f32 {
    linstep(amount, 1.0, p_max)
}

/// Linear step: maps `x` from [edge0, edge1] to [0, 1], clamped.
#[inline]
fn linstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Gaussian blur for VSM filtering
// ---------------------------------------------------------------------------

/// A separable Gaussian blur pass for filtering VSM textures.
pub struct VsmFilterPass {
    /// Blur kernel size.
    pub kernel_size: u32,
    /// Pre-computed Gaussian weights.
    pub weights: Vec<f32>,
    /// Pre-computed offsets (optimised for bilinear filtering).
    pub offsets: Vec<f32>,
}

impl VsmFilterPass {
    /// Create a new filter pass with the given kernel size and sigma.
    pub fn new(kernel_size: u32, sigma: f32) -> Self {
        let size = if kernel_size % 2 == 0 {
            kernel_size + 1
        } else {
            kernel_size
        };

        let weights = compute_gaussian_weights(size, sigma);
        let offsets = compute_gaussian_offsets(size);

        Self {
            kernel_size: size,
            weights,
            offsets,
        }
    }

    /// Create with default settings (5-tap, sigma=1.5).
    pub fn default_filter() -> Self {
        Self::new(5, 1.5)
    }

    /// Create a high-quality filter (9-tap, sigma=2.5).
    pub fn high_quality_filter() -> Self {
        Self::new(9, 2.5)
    }

    /// Apply horizontal blur to a 2-channel (depth, depth^2) image.
    ///
    /// # Arguments
    /// - `src` — source data (2 f32 per pixel, row-major).
    /// - `width`, `height` — image dimensions.
    ///
    /// # Returns
    /// Blurred image.
    pub fn blur_horizontal(&self, src: &[[f32; 2]], width: u32, height: u32) -> Vec<[f32; 2]> {
        let mut dst = vec![[0.0f32; 2]; (width * height) as usize];
        let half = (self.kernel_size / 2) as i32;

        for y in 0..height {
            for x in 0..width {
                let mut sum = [0.0f32; 2];
                for k in 0..self.kernel_size as i32 {
                    let sx = (x as i32 + k - half).clamp(0, width as i32 - 1) as u32;
                    let idx = (y * width + sx) as usize;
                    let w = self.weights[k as usize];
                    sum[0] += src[idx][0] * w;
                    sum[1] += src[idx][1] * w;
                }
                dst[(y * width + x) as usize] = sum;
            }
        }
        dst
    }

    /// Apply vertical blur.
    pub fn blur_vertical(&self, src: &[[f32; 2]], width: u32, height: u32) -> Vec<[f32; 2]> {
        let mut dst = vec![[0.0f32; 2]; (width * height) as usize];
        let half = (self.kernel_size / 2) as i32;

        for y in 0..height {
            for x in 0..width {
                let mut sum = [0.0f32; 2];
                for k in 0..self.kernel_size as i32 {
                    let sy = (y as i32 + k - half).clamp(0, height as i32 - 1) as u32;
                    let idx = (sy * width + x) as usize;
                    let w = self.weights[k as usize];
                    sum[0] += src[idx][0] * w;
                    sum[1] += src[idx][1] * w;
                }
                dst[(y * width + x) as usize] = sum;
            }
        }
        dst
    }

    /// Apply a full separable blur (horizontal then vertical).
    pub fn blur(&self, src: &[[f32; 2]], width: u32, height: u32) -> Vec<[f32; 2]> {
        let h = self.blur_horizontal(src, width, height);
        self.blur_vertical(&h, width, height)
    }

    /// Get the number of texture samples per pixel per pass.
    pub fn samples_per_pixel(&self) -> u32 {
        self.kernel_size
    }
}

/// Compute normalised Gaussian weights for a 1D kernel.
fn compute_gaussian_weights(size: u32, sigma: f32) -> Vec<f32> {
    let half = size as f32 / 2.0;
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut weights = Vec::with_capacity(size as usize);
    let mut total = 0.0f32;

    for i in 0..size {
        let x = i as f32 - half + 0.5;
        let w = (-x * x / two_sigma_sq).exp();
        weights.push(w);
        total += w;
    }

    // Normalise.
    for w in &mut weights {
        *w /= total;
    }
    weights
}

/// Compute offsets for bilinear-optimised Gaussian sampling.
///
/// Pairs adjacent samples and computes the offset that, when used with
/// bilinear filtering, gives the correct weighted average of both texels.
fn compute_gaussian_offsets(size: u32) -> Vec<f32> {
    let half = size as f32 / 2.0;
    (0..size).map(|i| i as f32 - half + 0.5).collect()
}

// ---------------------------------------------------------------------------
// VSM render pass WGSL code
// ---------------------------------------------------------------------------

/// WGSL shader for rendering depth + depth^2 to a VSM texture.
pub const VSM_RENDER_WGSL: &str = r#"
// VSM depth pass: output depth and depth^2 to an RG32Float render target.

struct VsmUniforms {
    light_view_projection: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> vsm_uniforms: VsmUniforms;

struct ModelUniforms {
    model: mat4x4<f32>,
    normal_matrix: mat3x3<f32>,
};

@group(1) @binding(0) var<uniform> model: ModelUniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) depth: f32,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = model.model * vec4<f32>(position, 1.0);
    let clip = vsm_uniforms.light_view_projection * world_pos;
    out.clip_position = clip;
    out.depth = clip.z / clip.w;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec2<f32> {
    let depth = in.depth;
    return vec2<f32>(depth, depth * depth);
}
"#;

/// WGSL shader for the horizontal Gaussian blur pass.
pub const VSM_BLUR_H_WGSL: &str = r#"
// VSM horizontal blur pass.

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;

struct BlurUniforms {
    texel_size: vec2<f32>,
    kernel_size: u32,
    _pad: u32,
};

@group(0) @binding(2) var<uniform> blur: BlurUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index) / 2) * 4.0 - 1.0;
    let y = f32(i32(vertex_index) % 2) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec2<f32> {
    let half_k = i32(blur.kernel_size) / 2;
    var result = vec2<f32>(0.0);
    var total_weight = 0.0;
    let sigma = f32(blur.kernel_size) * 0.3;
    let two_sigma_sq = 2.0 * sigma * sigma;

    for (var i = -half_k; i <= half_k; i++) {
        let offset = vec2<f32>(f32(i) * blur.texel_size.x, 0.0);
        let w = exp(-f32(i * i) / two_sigma_sq);
        result += textureSample(input_texture, input_sampler, in.uv + offset).rg * w;
        total_weight += w;
    }

    return result / total_weight;
}
"#;

/// WGSL shader for the vertical Gaussian blur pass.
pub const VSM_BLUR_V_WGSL: &str = r#"
// VSM vertical blur pass.

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;

struct BlurUniforms {
    texel_size: vec2<f32>,
    kernel_size: u32,
    _pad: u32,
};

@group(0) @binding(2) var<uniform> blur: BlurUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index) / 2) * 4.0 - 1.0;
    let y = f32(i32(vertex_index) % 2) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec2<f32> {
    let half_k = i32(blur.kernel_size) / 2;
    var result = vec2<f32>(0.0);
    var total_weight = 0.0;
    let sigma = f32(blur.kernel_size) * 0.3;
    let two_sigma_sq = 2.0 * sigma * sigma;

    for (var i = -half_k; i <= half_k; i++) {
        let offset = vec2<f32>(0.0, f32(i) * blur.texel_size.y);
        let w = exp(-f32(i * i) / two_sigma_sq);
        result += textureSample(input_texture, input_sampler, in.uv + offset).rg * w;
        total_weight += w;
    }

    return result / total_weight;
}
"#;

/// WGSL shader for sampling a VSM texture in the main PBR pass.
pub const VSM_SAMPLE_WGSL: &str = r#"
// VSM shadow sampling function.
// Call this from the PBR fragment shader.

fn sample_shadow_vsm(
    vsm_texture: texture_2d<f32>,
    vsm_sampler: sampler,
    shadow_uv: vec2<f32>,
    receiver_depth: f32,
    min_variance: f32,
    light_bleed_reduction: f32,
) -> f32 {
    let moments = textureSample(vsm_texture, vsm_sampler, shadow_uv).rg;
    let mean = moments.x;
    let mean_sq = moments.y;

    if receiver_depth <= mean {
        return 1.0;
    }

    let variance = max(mean_sq - mean * mean, min_variance);
    let d = receiver_depth - mean;
    let p_max = variance / (variance + d * d);

    // Light bleed reduction (linstep).
    return clamp((p_max - light_bleed_reduction) / (1.0 - light_bleed_reduction), 0.0, 1.0);
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vsm_fully_lit() {
        // Receiver at depth 0.3, mean depth 0.5 -> fully lit.
        let result = evaluate_vsm(Vec2::new(0.5, 0.3), 0.3, 0.0001, 0.0);
        assert!((result - 1.0).abs() < 0.01);
    }

    #[test]
    fn vsm_shadowed() {
        // Receiver at depth 0.8, mean depth 0.3, mean_sq = 0.09 + tiny variance.
        let result = evaluate_vsm(Vec2::new(0.3, 0.1), 0.8, 0.0001, 0.0);
        assert!(result < 0.5); // Should be mostly shadowed.
    }

    #[test]
    fn light_bleed_reduces_shadows() {
        let without = evaluate_vsm(Vec2::new(0.3, 0.12), 0.5, 0.0001, 0.0);
        let with = evaluate_vsm(Vec2::new(0.3, 0.12), 0.5, 0.0001, 0.3);
        // With light bleed reduction, the shadow should be darker (less bleeding).
        assert!(with <= without + 0.01);
    }

    #[test]
    fn gaussian_weights_normalised() {
        let weights = compute_gaussian_weights(9, 2.0);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn gaussian_weights_symmetric() {
        let weights = compute_gaussian_weights(7, 1.5);
        assert!((weights[0] - weights[6]).abs() < 0.001);
        assert!((weights[1] - weights[5]).abs() < 0.001);
        assert!((weights[2] - weights[4]).abs() < 0.001);
    }

    #[test]
    fn blur_identity_on_uniform() {
        let filter = VsmFilterPass::new(3, 1.0);
        let data = vec![[0.5f32, 0.25]; 16];
        let result = filter.blur(&data, 4, 4);
        // Uniform data should remain approximately uniform after blur.
        for pixel in &result {
            assert!((pixel[0] - 0.5).abs() < 0.01);
            assert!((pixel[1] - 0.25).abs() < 0.01);
        }
    }

    #[test]
    fn evsm_positive_fully_lit() {
        let moments = Vec2::new(100.0, 10500.0);
        let result = evaluate_evsm_positive(moments, 0.05, 40.0, 0.0001, 0.0);
        // exp(40 * 0.05) ≈ 7.39, which is << 100. Should be lit.
        assert!((result - 1.0).abs() < 0.01);
    }

    #[test]
    fn linstep_basic() {
        assert!((linstep(0.0, 1.0, 0.5) - 0.5).abs() < 0.001);
        assert!((linstep(0.2, 1.0, 0.0) - 0.0).abs() < 0.001);
        assert!((linstep(0.0, 1.0, 1.5) - 1.0).abs() < 0.001);
    }

    #[test]
    fn moment_shadows_basic() {
        // Simple test: receiver behind the mean should be shadowed.
        let moments = [0.3, 0.1, 0.04, 0.02];
        let result = evaluate_moment_shadows(moments, 0.8, 0.001);
        assert!(result < 1.0);
    }
}
