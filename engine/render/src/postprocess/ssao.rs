// engine/render/src/postprocess/ssao.rs
//
// Screen Space Ambient Occlusion (SSAO) and Horizon-Based Ambient Occlusion
// (HBAO). Approximates global illumination shadowing by sampling the depth
// buffer around each pixel and checking how much nearby geometry occludes
// the hemisphere above the surface.
//
// Two algorithms are provided:
//   - **SSAO** (Crytek-style): Random hemisphere sampling with a cosine-
//     weighted kernel and a 4x4 noise texture for rotation.
//   - **HBAO** (Horizon-Based AO): Marches along several screen-space
//     directions to find the horizon angle, then computes AO from the
//     difference between the horizon and the surface tangent angle.
//
// Both variants include a bilateral blur pass to remove noise while
// preserving edges.

use std::any::Any;

use super::{PostProcessEffect, PostProcessInput, PostProcessOutput, TextureId};

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// Selects which AO algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SSAOMode {
    /// Crytek-style hemisphere sampling.
    SSAO,
    /// Horizon-Based Ambient Occlusion.
    HBAO,
}

/// Configuration for the SSAO/HBAO effect.
#[derive(Debug, Clone)]
pub struct SSAOSettings {
    /// Which algorithm to use.
    pub mode: SSAOMode,
    /// World-space radius of the AO sampling hemisphere.
    pub radius: f32,
    /// Depth bias to prevent self-occlusion artifacts.
    pub bias: f32,
    /// Intensity multiplier for the final AO term.
    pub intensity: f32,
    /// Power exponent applied to the raw AO (contrast control).
    pub power: f32,
    /// Number of samples per pixel for SSAO (typically 16, 32, or 64).
    pub sample_count: u32,
    /// Number of directions for HBAO (typically 4, 6, or 8).
    pub hbao_direction_count: u32,
    /// Number of steps per direction for HBAO.
    pub hbao_steps_per_direction: u32,
    /// Bilateral blur kernel size (half-width). 0 = no blur.
    pub blur_size: u32,
    /// Edge-stopping threshold for the bilateral blur (depth difference).
    pub blur_sharpness: f32,
    /// Whether the effect is enabled.
    pub enabled: bool,
    /// Half-resolution rendering for performance.
    pub half_resolution: bool,
}

impl Default for SSAOSettings {
    fn default() -> Self {
        Self {
            mode: SSAOMode::SSAO,
            radius: 0.5,
            bias: 0.025,
            intensity: 1.5,
            power: 2.0,
            sample_count: 64,
            hbao_direction_count: 8,
            hbao_steps_per_direction: 6,
            blur_size: 4,
            blur_sharpness: 40.0,
            enabled: true,
            half_resolution: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Sample kernel generation
// ---------------------------------------------------------------------------

/// Generate a hemisphere sample kernel with cosine-weighted distribution.
///
/// Samples are generated in tangent space (z >= 0 hemisphere) and are
/// distributed so that more samples are closer to the surface (the
/// `accelerating_interpolation` pushes samples towards the origin).
///
/// Returns `count` sample positions as `[x, y, z]` in tangent space.
pub fn generate_ssao_kernel(count: u32) -> Vec<[f32; 3]> {
    let mut samples = Vec::with_capacity(count as usize);

    // We use a deterministic low-discrepancy sequence for better coverage.
    for i in 0..count {
        let xi1 = radical_inverse_base2(i);
        let xi2 = (i as f32 + 0.5) / count as f32;

        // Map to hemisphere using cosine-weighted distribution.
        let phi = 2.0 * std::f32::consts::PI * xi1;
        let cos_theta = (1.0 - xi2).sqrt();
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let mut sample = [
            sin_theta * phi.cos(),
            sin_theta * phi.sin(),
            cos_theta,
        ];

        // Scale: samples closer to center of the hemisphere are more
        // important (they contribute more to ambient occlusion).
        let mut scale = (i as f32 + 1.0) / count as f32;
        // Accelerating interpolation: lerp(0.1, 1.0, scale * scale)
        scale = 0.1 + scale * scale * 0.9;

        sample[0] *= scale;
        sample[1] *= scale;
        sample[2] *= scale;

        samples.push(sample);
    }

    samples
}

/// Generate the 4x4 random rotation texture.
///
/// Returns 16 unit-length 2D vectors packed as `[x, y]`. These are used
/// to rotate the sample kernel per-pixel, breaking up banding without
/// increasing sample count.
pub fn generate_noise_texture() -> Vec<[f32; 2]> {
    let mut noise = Vec::with_capacity(16);
    // Deterministic rotations evenly distributed + slight perturbation
    // from a low-discrepancy sequence.
    for i in 0..16u32 {
        let angle = (i as f32 / 16.0 + radical_inverse_base2(i) * 0.5)
            * 2.0
            * std::f32::consts::PI;
        noise.push([angle.cos(), angle.sin()]);
    }
    noise
}

/// Radical inverse in base 2 (Van der Corput sequence).
fn radical_inverse_base2(mut n: u32) -> f32 {
    n = (n << 16) | (n >> 16);
    n = ((n & 0x55555555) << 1) | ((n & 0xAAAAAAAA) >> 1);
    n = ((n & 0x33333333) << 2) | ((n & 0xCCCCCCCC) >> 2);
    n = ((n & 0x0F0F0F0F) << 4) | ((n & 0xF0F0F0F0) >> 4);
    n = ((n & 0x00FF00FF) << 8) | ((n & 0xFF00FF00) >> 8);
    n as f32 * 2.328_306_4e-10 // 1 / 0x100000000
}

// ---------------------------------------------------------------------------
// Depth reconstruction helpers
// ---------------------------------------------------------------------------

/// Reconstruct view-space position from depth and screen UV.
///
/// `uv` is in [0, 1], `depth` is the non-linear depth buffer value,
/// `inv_proj` is the inverse projection matrix.
pub fn reconstruct_view_position(
    uv: [f32; 2],
    depth: f32,
    inv_proj: &[[f32; 4]; 4],
) -> [f32; 3] {
    // Convert to NDC: xy in [-1, 1], z = depth (assuming [0, 1] range).
    let ndc = [uv[0] * 2.0 - 1.0, (1.0 - uv[1]) * 2.0 - 1.0, depth, 1.0];

    // Multiply by inverse projection.
    let view = mat4_mul_vec4(inv_proj, &ndc);
    let w = view[3];
    if w.abs() < 1e-10 {
        return [0.0; 3];
    }
    [view[0] / w, view[1] / w, view[2] / w]
}

/// Linearize a depth buffer value.
///
/// Converts from the non-linear depth buffer value to a linear view-space
/// depth (positive, increasing away from camera).
pub fn linearize_depth(depth: f32, near: f32, far: f32) -> f32 {
    let z_ndc = depth;
    near * far / (far - z_ndc * (far - near))
}

fn mat4_mul_vec4(m: &[[f32; 4]; 4], v: &[f32; 4]) -> [f32; 4] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3] * v[3],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3] * v[3],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3] * v[3],
        m[3][0] * v[0] + m[3][1] * v[1] + m[3][2] * v[2] + m[3][3] * v[3],
    ]
}

// ---------------------------------------------------------------------------
// SSAO computation (CPU reference)
// ---------------------------------------------------------------------------

/// Compute SSAO for a single pixel (CPU reference implementation).
///
/// `position` is the view-space position of the pixel.
/// `normal` is the view-space normal.
/// `kernel` is the pre-generated sample kernel.
/// `rotation` is a 2D rotation vector from the noise texture.
/// `sample_depth_fn` is a closure that returns the depth at a given UV.
pub fn compute_ssao_pixel(
    position: [f32; 3],
    normal: [f32; 3],
    kernel: &[[f32; 3]],
    rotation: [f32; 2],
    radius: f32,
    bias: f32,
    projection: &[[f32; 4]; 4],
    sample_depth_fn: &dyn Fn(f32, f32) -> f32,
) -> f32 {
    let mut occlusion = 0.0f32;

    // Build a TBN matrix from the normal and the random rotation.
    let tangent = normalize([
        rotation[0] - normal[0] * dot3(&[rotation[0], rotation[1], 0.0], &normal),
        rotation[1] - normal[1] * dot3(&[rotation[0], rotation[1], 0.0], &normal),
        -normal[2] * dot3(&[rotation[0], rotation[1], 0.0], &normal),
    ]);
    let bitangent = cross(&normal, &tangent);

    for sample in kernel {
        // Rotate sample from tangent space to view space.
        let sample_pos = [
            position[0]
                + (tangent[0] * sample[0] + bitangent[0] * sample[1] + normal[0] * sample[2])
                    * radius,
            position[1]
                + (tangent[1] * sample[0] + bitangent[1] * sample[1] + normal[1] * sample[2])
                    * radius,
            position[2]
                + (tangent[2] * sample[0] + bitangent[2] * sample[1] + normal[2] * sample[2])
                    * radius,
        ];

        // Project sample position to screen space to get UV.
        let clip = [
            projection[0][0] * sample_pos[0]
                + projection[0][1] * sample_pos[1]
                + projection[0][2] * sample_pos[2]
                + projection[0][3],
            projection[1][0] * sample_pos[0]
                + projection[1][1] * sample_pos[1]
                + projection[1][2] * sample_pos[2]
                + projection[1][3],
            projection[2][0] * sample_pos[0]
                + projection[2][1] * sample_pos[1]
                + projection[2][2] * sample_pos[2]
                + projection[2][3],
            projection[3][0] * sample_pos[0]
                + projection[3][1] * sample_pos[1]
                + projection[3][2] * sample_pos[2]
                + projection[3][3],
        ];

        if clip[3].abs() < 1e-10 {
            continue;
        }

        let ndc_x = clip[0] / clip[3];
        let ndc_y = clip[1] / clip[3];
        let uv_x = ndc_x * 0.5 + 0.5;
        let uv_y = 1.0 - (ndc_y * 0.5 + 0.5);

        // Sample the depth buffer at this UV.
        let sample_depth = sample_depth_fn(uv_x, uv_y);

        // Range check: only occlude if the sample is within the radius.
        let range_check = smooth_step(
            0.0,
            1.0,
            radius / (sample_pos[2] - sample_depth).abs().max(1e-6),
        );

        // Occlusion test: is the sampled depth closer than our sample?
        if sample_depth >= sample_pos[2] + bias {
            occlusion += range_check;
        }
    }

    1.0 - (occlusion / kernel.len() as f32)
}

// ---------------------------------------------------------------------------
// HBAO computation (CPU reference)
// ---------------------------------------------------------------------------

/// Compute HBAO for a single pixel (CPU reference implementation).
///
/// Marches along `direction_count` directions in screen space, finding the
/// horizon angle for each direction and computing AO from the difference
/// between the horizon angle and the tangent angle.
pub fn compute_hbao_pixel(
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    direction_count: u32,
    steps_per_direction: u32,
    radius_pixels: f32,
    bias: f32,
    sample_position_fn: &dyn Fn(f32, f32) -> [f32; 3],
) -> f32 {
    let mut total_ao = 0.0f32;

    let angle_step = std::f32::consts::PI / direction_count as f32;

    for dir in 0..direction_count {
        let angle = dir as f32 * angle_step;
        let direction = [angle.cos(), angle.sin()];

        // Compute tangent angle from surface normal projected onto this
        // direction.
        let tangent_vec = [
            normal[0] * direction[0] + normal[1] * direction[1],
            normal[2],
        ];
        let tangent_angle = tangent_vec[1].atan2(tangent_vec[0]);

        let mut max_horizon_angle = tangent_angle - std::f32::consts::FRAC_PI_2;

        // March along the direction, finding the highest horizon angle.
        let step_size = radius_pixels / steps_per_direction as f32;

        for step in 1..=steps_per_direction {
            let offset = step as f32 * step_size;
            let sample_uv = [
                uv[0] + direction[0] * offset / 1920.0, // normalized
                uv[1] + direction[1] * offset / 1080.0,
            ];

            // Clamp to screen bounds.
            if sample_uv[0] < 0.0
                || sample_uv[0] > 1.0
                || sample_uv[1] < 0.0
                || sample_uv[1] > 1.0
            {
                continue;
            }

            let sample_pos = sample_position_fn(sample_uv[0], sample_uv[1]);
            let diff = [
                sample_pos[0] - position[0],
                sample_pos[1] - position[1],
                sample_pos[2] - position[2],
            ];

            let dist = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
            if dist < bias {
                continue;
            }

            let horizon_angle = (diff[2] / dist).asin();
            if horizon_angle > max_horizon_angle {
                max_horizon_angle = horizon_angle;
            }
        }

        // AO contribution from this direction: sin(horizon) - sin(tangent)
        let ao = (max_horizon_angle.sin() - tangent_angle.sin()).max(0.0);
        total_ao += ao;
    }

    1.0 - total_ao / direction_count as f32
}

// ---------------------------------------------------------------------------
// Bilateral blur
// ---------------------------------------------------------------------------

/// Parameters for the bilateral (edge-preserving) blur pass.
#[derive(Debug, Clone)]
pub struct BilateralBlurParams {
    /// Half-width of the blur kernel.
    pub kernel_half_size: u32,
    /// Controls how aggressively depth edges stop the blur.
    pub sharpness: f32,
    /// Whether this is a horizontal (true) or vertical (false) pass.
    pub horizontal: bool,
}

/// Compute a 1D bilateral blur weight for a spatial + depth kernel.
pub fn bilateral_weight(
    spatial_distance: f32,
    depth_center: f32,
    depth_sample: f32,
    sharpness: f32,
) -> f32 {
    // Gaussian spatial weight.
    let sigma = 3.0; // spatial sigma
    let spatial = (-spatial_distance * spatial_distance / (2.0 * sigma * sigma)).exp();

    // Edge-stopping range weight based on depth difference.
    let depth_diff = (depth_center - depth_sample).abs();
    let range = (-depth_diff * sharpness).exp();

    spatial * range
}

// ---------------------------------------------------------------------------
// SSAOEffect
// ---------------------------------------------------------------------------

/// Screen Space Ambient Occlusion post-process effect.
pub struct SSAOEffect {
    pub settings: SSAOSettings,
    /// Pre-computed sample kernel.
    kernel: Vec<[f32; 3]>,
    /// 4x4 noise texture values.
    noise: Vec<[f32; 2]>,
    /// AO result texture.
    ao_texture: TextureId,
    /// Blurred AO texture.
    blurred_texture: TextureId,
    /// Intermediate texture for two-pass blur.
    blur_intermediate: TextureId,
}

impl SSAOEffect {
    pub fn new(settings: SSAOSettings) -> Self {
        let kernel = generate_ssao_kernel(settings.sample_count);
        let noise = generate_noise_texture();
        Self {
            settings,
            kernel,
            noise,
            ao_texture: TextureId(300),
            blurred_texture: TextureId(301),
            blur_intermediate: TextureId(302),
        }
    }

    /// Regenerate the sample kernel (e.g., when sample_count changes).
    pub fn regenerate_kernel(&mut self) {
        self.kernel = generate_ssao_kernel(self.settings.sample_count);
    }

    /// Execute the SSAO/HBAO compute pass.
    fn execute_ao_pass(&self, _input: &PostProcessInput) {
        // In a real implementation:
        // 1. Upload kernel samples and noise texture.
        // 2. Bind depth texture, normal texture, and camera uniforms.
        // 3. Dispatch the SSAO or HBAO compute shader.
        // 4. Output goes to self.ao_texture.
        match self.settings.mode {
            SSAOMode::SSAO => {
                // Dispatch SSAO compute shader.
            }
            SSAOMode::HBAO => {
                // Dispatch HBAO compute shader.
            }
        }
    }

    /// Execute the bilateral blur pass (two-pass separable).
    fn execute_blur_pass(&self) {
        if self.settings.blur_size == 0 {
            return;
        }

        // Horizontal pass: ao_texture -> blur_intermediate
        // Vertical pass: blur_intermediate -> blurred_texture
    }
}

impl PostProcessEffect for SSAOEffect {
    fn name(&self) -> &str {
        "SSAO"
    }

    fn execute(&self, input: &PostProcessInput, _output: &mut PostProcessOutput) {
        if !self.settings.enabled {
            return;
        }

        self.execute_ao_pass(input);
        self.execute_blur_pass();

        // The pipeline should pick up self.blurred_texture via the named
        // texture system ("ssao_result").
    }

    fn is_enabled(&self) -> bool {
        self.settings.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.settings.enabled = enabled;
    }

    fn priority(&self) -> u32 {
        100
    }

    fn on_resize(&mut self, _width: u32, _height: u32) {
        // Reallocate AO textures at the new resolution (or half-res).
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Vector math helpers
// ---------------------------------------------------------------------------

fn dot3(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return [0.0; 3];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

fn smooth_step(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// WGSL Shaders
// ---------------------------------------------------------------------------

/// Crytek-style SSAO compute shader.
pub const SSAO_WGSL: &str = r#"
// Screen Space Ambient Occlusion — compute shader
// Crytek-style hemisphere sampling with random rotation.

struct SSAOParams {
    projection: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    radius: f32,
    bias: f32,
    intensity: f32,
    power: f32,
    sample_count: u32,
    viewport_width: f32,
    viewport_height: f32,
    near_plane: f32,
    far_plane: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

struct SSAOKernel {
    samples: array<vec4<f32>, 64>,
};

@group(0) @binding(0) var depth_texture:  texture_2d<f32>;
@group(0) @binding(1) var normal_texture: texture_2d<f32>;
@group(0) @binding(2) var noise_texture:  texture_2d<f32>;
@group(0) @binding(3) var ao_output:      texture_storage_2d<r16float, write>;
@group(0) @binding(4) var depth_sampler:  sampler;
@group(0) @binding(5) var<uniform> params: SSAOParams;
@group(0) @binding(6) var<uniform> kernel: SSAOKernel;

fn reconstruct_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let view = params.inv_projection * vec4<f32>(ndc.x, -ndc.y, ndc.z, 1.0);
    return view.xyz / view.w;
}

fn linearize_depth(d: f32) -> f32 {
    return params.near_plane * params.far_plane /
           (params.far_plane - d * (params.far_plane - params.near_plane));
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(ao_output);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);

    // Sample depth and reconstruct position
    let depth = textureSampleLevel(depth_texture, depth_sampler, uv, 0.0).r;
    if depth >= 1.0 {
        textureStore(ao_output, gid.xy, vec4<f32>(1.0, 0.0, 0.0, 0.0));
        return;
    }

    let frag_pos = reconstruct_view_pos(uv, depth);
    let normal = normalize(textureLoad(normal_texture, gid.xy, 0).xyz * 2.0 - 1.0);

    // Random rotation from 4x4 noise texture (tiled)
    let noise_uv = vec2<f32>(gid.xy) / 4.0;
    let noise_val = textureLoad(noise_texture, vec2<u32>(gid.xy % 4u), 0).xy;
    let random_vec = normalize(vec3<f32>(noise_val, 0.0));

    // Build TBN from normal + random vector (Gram-Schmidt)
    let tangent = normalize(random_vec - normal * dot(random_vec, normal));
    let bitangent = cross(normal, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, normal);

    var occlusion = 0.0;

    for (var i = 0u; i < params.sample_count; i++) {
        // Transform sample to view space
        let sample_dir = tbn * kernel.samples[i].xyz;
        let sample_pos = frag_pos + sample_dir * params.radius;

        // Project to screen space
        let projected = params.projection * vec4<f32>(sample_pos, 1.0);
        var sample_uv = projected.xy / projected.w;
        sample_uv = sample_uv * 0.5 + 0.5;
        sample_uv.y = 1.0 - sample_uv.y;

        // Sample depth at projected position
        let sample_depth = textureSampleLevel(depth_texture, depth_sampler, sample_uv, 0.0).r;
        let sample_view_z = reconstruct_view_pos(sample_uv, sample_depth).z;

        // Range check
        let range_check = smoothstep(0.0, 1.0, params.radius / abs(frag_pos.z - sample_view_z));

        // Occlusion test
        if sample_view_z >= sample_pos.z + params.bias {
            occlusion += range_check;
        }
    }

    occlusion = 1.0 - (occlusion / f32(params.sample_count));
    occlusion = pow(occlusion, params.power) * params.intensity;
    occlusion = clamp(occlusion, 0.0, 1.0);

    textureStore(ao_output, gid.xy, vec4<f32>(occlusion, 0.0, 0.0, 0.0));
}
"#;

/// HBAO compute shader.
pub const HBAO_WGSL: &str = r#"
// Horizon-Based Ambient Occlusion — compute shader

struct HBAOParams {
    inv_projection: mat4x4<f32>,
    radius: f32,
    bias: f32,
    intensity: f32,
    power: f32,
    direction_count: u32,
    steps_per_direction: u32,
    viewport_width: f32,
    viewport_height: f32,
    near_plane: f32,
    far_plane: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var depth_texture: texture_2d<f32>;
@group(0) @binding(1) var normal_texture: texture_2d<f32>;
@group(0) @binding(2) var noise_texture: texture_2d<f32>;
@group(0) @binding(3) var ao_output: texture_storage_2d<r16float, write>;
@group(0) @binding(4) var depth_sampler: sampler;
@group(0) @binding(5) var<uniform> params: HBAOParams;

fn reconstruct_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let view = params.inv_projection * vec4<f32>(ndc.x, -ndc.y, ndc.z, 1.0);
    return view.xyz / view.w;
}

fn compute_tangent_angle(normal: vec3<f32>, direction: vec2<f32>) -> f32 {
    let tangent_vec = vec2<f32>(
        normal.x * direction.x + normal.y * direction.y,
        normal.z
    );
    return atan2(tangent_vec.y, tangent_vec.x);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(ao_output);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let depth = textureSampleLevel(depth_texture, depth_sampler, uv, 0.0).r;
    if depth >= 1.0 {
        textureStore(ao_output, gid.xy, vec4<f32>(1.0, 0.0, 0.0, 0.0));
        return;
    }

    let frag_pos = reconstruct_view_pos(uv, depth);
    let normal = normalize(textureLoad(normal_texture, gid.xy, 0).xyz * 2.0 - 1.0);

    // Random rotation from noise texture
    let noise_val = textureLoad(noise_texture, vec2<u32>(gid.xy % 4u), 0).xy;
    let noise_angle = atan2(noise_val.y, noise_val.x);

    let angle_step = 3.14159265 / f32(params.direction_count);
    let pixel_radius = params.radius / abs(frag_pos.z) * params.viewport_height * 0.5;
    let step_size = max(pixel_radius / f32(params.steps_per_direction), 1.0);

    var total_ao = 0.0;

    for (var dir = 0u; dir < params.direction_count; dir++) {
        let angle = f32(dir) * angle_step + noise_angle;
        let direction = vec2<f32>(cos(angle), sin(angle));

        let tangent_angle = compute_tangent_angle(normal, direction);
        var max_horizon = tangent_angle - 1.5707963;  // - PI/2

        for (var step = 1u; step <= params.steps_per_direction; step++) {
            let offset = f32(step) * step_size;
            let sample_uv = uv + direction * offset / vec2<f32>(params.viewport_width, params.viewport_height);

            if any(sample_uv < vec2<f32>(0.0)) || any(sample_uv > vec2<f32>(1.0)) {
                continue;
            }

            let sample_depth = textureSampleLevel(depth_texture, depth_sampler, sample_uv, 0.0).r;
            let sample_pos = reconstruct_view_pos(sample_uv, sample_depth);

            let diff = sample_pos - frag_pos;
            let dist = length(diff);

            if dist < params.bias {
                continue;
            }

            let horizon = asin(diff.z / dist);
            max_horizon = max(max_horizon, horizon);
        }

        let ao_contrib = max(sin(max_horizon) - sin(tangent_angle), 0.0);
        total_ao += ao_contrib;
    }

    var ao = 1.0 - total_ao / f32(params.direction_count);
    ao = pow(clamp(ao, 0.0, 1.0), params.power) * params.intensity;

    textureStore(ao_output, gid.xy, vec4<f32>(ao, 0.0, 0.0, 0.0));
}
"#;

/// Bilateral blur compute shader (separable, one direction per dispatch).
pub const BILATERAL_BLUR_WGSL: &str = r#"
// Bilateral (edge-preserving) blur — compute shader
// Separable: dispatch once horizontal, once vertical.

struct BlurParams {
    direction: vec2<f32>,       // (1,0) for horizontal, (0,1) for vertical
    kernel_half_size: u32,
    sharpness: f32,
    inv_width: f32,
    inv_height: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var src_texture:   texture_2d<f32>;
@group(0) @binding(1) var depth_texture: texture_2d<f32>;
@group(0) @binding(2) var dst_texture:   texture_storage_2d<r16float, write>;
@group(0) @binding(3) var tex_sampler:   sampler;
@group(0) @binding(4) var<uniform> params: BlurParams;

fn linearize(d: f32) -> f32 {
    // Approximate linearization for edge detection.
    return 1.0 / (1.0 - d + 0.001);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_texture);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let texel = vec2<f32>(params.inv_width, params.inv_height);

    let center_ao = textureSampleLevel(src_texture, tex_sampler, uv, 0.0).r;
    let center_depth = linearize(textureSampleLevel(depth_texture, tex_sampler, uv, 0.0).r);

    var total_ao = center_ao;
    var total_weight = 1.0;

    let sigma = f32(params.kernel_half_size) * 0.5;

    for (var i = 1; i <= i32(params.kernel_half_size); i++) {
        let offset = f32(i);

        // Gaussian spatial weight
        let spatial = exp(-offset * offset / (2.0 * sigma * sigma));

        for (var sign = -1; sign <= 1; sign += 2) {
            let sample_uv = uv + params.direction * texel * f32(sign) * offset;
            let sample_ao = textureSampleLevel(src_texture, tex_sampler, sample_uv, 0.0).r;
            let sample_depth = linearize(textureSampleLevel(depth_texture, tex_sampler, sample_uv, 0.0).r);

            // Edge-stopping weight
            let depth_diff = abs(center_depth - sample_depth);
            let range_weight = exp(-depth_diff * params.sharpness);

            let w = spatial * range_weight;
            total_ao += sample_ao * w;
            total_weight += w;
        }
    }

    let result = total_ao / total_weight;
    textureStore(dst_texture, gid.xy, vec4<f32>(result, 0.0, 0.0, 0.0));
}
"#;

/// Composite shader that multiplies the scene color by the AO term.
pub const SSAO_COMPOSITE_WGSL: &str = r#"
// SSAO composite — compute shader
// Multiplies scene color by the ambient occlusion factor.

@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var ao_texture:    texture_2d<f32>;
@group(0) @binding(2) var dst_texture:   texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var tex_sampler:   sampler;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_texture);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let color = textureSampleLevel(scene_texture, tex_sampler, uv, 0.0);
    let ao = textureSampleLevel(ao_texture, tex_sampler, uv, 0.0).r;

    // Only apply AO to ambient/indirect lighting (approximate by
    // modulating all color channels).
    let result = vec4<f32>(color.rgb * ao, color.a);
    textureStore(dst_texture, gid.xy, result);
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_generation() {
        let kernel = generate_ssao_kernel(64);
        assert_eq!(kernel.len(), 64);

        for sample in &kernel {
            // All samples should be in the positive z hemisphere.
            assert!(sample[2] >= 0.0);
            // All samples should be within unit sphere (with scale).
            let len = (sample[0] * sample[0] + sample[1] * sample[1] + sample[2] * sample[2])
                .sqrt();
            assert!(len <= 1.01, "sample outside unit sphere: len={}", len);
        }
    }

    #[test]
    fn test_noise_texture() {
        let noise = generate_noise_texture();
        assert_eq!(noise.len(), 16);

        for n in &noise {
            let len = (n[0] * n[0] + n[1] * n[1]).sqrt();
            assert!(
                (len - 1.0).abs() < 0.01,
                "noise vector should be unit length: {len}"
            );
        }
    }

    #[test]
    fn test_linearize_depth() {
        let near = 0.1;
        let far = 1000.0;

        // Depth 0 -> near plane
        let d0 = linearize_depth(0.0, near, far);
        assert!((d0 - near).abs() < 0.01);

        // Depth 1 -> far plane
        let d1 = linearize_depth(1.0, near, far);
        assert!((d1 - far).abs() < 1.0);
    }

    #[test]
    fn test_bilateral_weight() {
        // Same depth -> weight should be close to spatial weight alone.
        let w_same = bilateral_weight(1.0, 10.0, 10.0, 40.0);
        let w_diff = bilateral_weight(1.0, 10.0, 15.0, 40.0);
        assert!(w_same > w_diff, "same depth should have higher weight");
    }

    #[test]
    fn test_ssao_effect_interface() {
        let effect = SSAOEffect::new(SSAOSettings::default());
        assert_eq!(effect.name(), "SSAO");
        assert!(effect.is_enabled());
        assert_eq!(effect.priority(), 100);
        assert_eq!(effect.kernel.len(), 64);
    }

    #[test]
    fn test_radical_inverse() {
        let v0 = radical_inverse_base2(0);
        assert!(v0.abs() < 1e-6);

        let v1 = radical_inverse_base2(1);
        assert!((v1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_smooth_step() {
        assert!((smooth_step(0.0, 1.0, 0.0)).abs() < 1e-6);
        assert!((smooth_step(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6);
        assert!((smooth_step(0.0, 1.0, 1.0) - 1.0).abs() < 1e-6);
    }
}
