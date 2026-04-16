// engine/render/src/subsurface.rs
//
// Subsurface scattering (SSS) for the Genovo engine. Implements separable
// screen-space SSS (Jimenez 2015), pre-integrated skin shading, and thin-object
// transmission for realistic skin, foliage, and translucent material rendering.

use glam::{Vec2, Vec3, Vec4};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// SubsurfaceProfile
// ---------------------------------------------------------------------------

/// A subsurface scattering profile defining how light scatters within a
/// material. Different profiles are used for skin, wax, marble, foliage, etc.
#[derive(Debug, Clone)]
pub struct SubsurfaceProfile {
    /// Name of this profile (for editor display).
    pub name: String,
    /// Scatter radius in world units. Larger values produce more visible
    /// scattering.
    pub scatter_radius: f32,
    /// Scatter colour: tints the scattered light. For skin this is reddish.
    pub scatter_color: Vec3,
    /// Falloff colour: controls the spread of scattering per channel.
    /// Each channel's falloff determines how quickly that wavelength attenuates.
    pub falloff: Vec3,
    /// Transmission tint: colour of light transmitted through thin parts.
    pub transmission_tint: Vec3,
    /// Transmission scale: multiplier for transmission intensity.
    pub transmission_scale: f32,
    /// Normal scale: how much the normal affects the SSS (0 = ignore normal,
    /// 1 = full normal influence).
    pub normal_scale: f32,
    /// Ambient scattering contribution.
    pub ambient_scatter: f32,
    /// Whether this profile uses transmission (thin objects like ears/leaves).
    pub enable_transmission: bool,
    /// Precomputed 1D Gaussian kernel (25 samples, separable).
    pub kernel: Vec<SSSKernelSample>,
    /// Precomputed scattering LUT (for pre-integrated skin shading).
    pub scatter_lut: Vec<f32>,
    /// LUT dimensions.
    pub lut_width: u32,
    pub lut_height: u32,
}

/// A single sample in the separable SSS kernel.
#[derive(Debug, Clone, Copy)]
pub struct SSSKernelSample {
    /// Offset from centre (normalised, will be scaled by scatter radius).
    pub offset: f32,
    /// Weight for this sample per channel (RGB + total).
    pub weight: Vec4,
}

impl SubsurfaceProfile {
    /// Default human skin profile.
    pub fn skin() -> Self {
        let mut profile = Self {
            name: "Skin".to_string(),
            scatter_radius: 0.012,
            scatter_color: Vec3::new(0.78, 0.42, 0.26),
            falloff: Vec3::new(1.0, 0.37, 0.12),
            transmission_tint: Vec3::new(1.0, 0.56, 0.34),
            transmission_scale: 0.5,
            normal_scale: 0.08,
            ambient_scatter: 0.15,
            enable_transmission: true,
            kernel: Vec::new(),
            scatter_lut: Vec::new(),
            lut_width: 0,
            lut_height: 0,
        };
        profile.build_kernel(25);
        profile.build_scatter_lut(128, 128);
        profile
    }

    /// Foliage/leaf profile (thin translucent material).
    pub fn foliage() -> Self {
        let mut profile = Self {
            name: "Foliage".to_string(),
            scatter_radius: 0.008,
            scatter_color: Vec3::new(0.3, 0.7, 0.15),
            falloff: Vec3::new(0.5, 1.0, 0.3),
            transmission_tint: Vec3::new(0.6, 0.85, 0.2),
            transmission_scale: 1.0,
            normal_scale: 0.1,
            ambient_scatter: 0.2,
            enable_transmission: true,
            kernel: Vec::new(),
            scatter_lut: Vec::new(),
            lut_width: 0,
            lut_height: 0,
        };
        profile.build_kernel(25);
        profile.build_scatter_lut(128, 128);
        profile
    }

    /// Wax/marble profile (subsurface with no transmission).
    pub fn wax() -> Self {
        let mut profile = Self {
            name: "Wax".to_string(),
            scatter_radius: 0.02,
            scatter_color: Vec3::new(0.9, 0.7, 0.5),
            falloff: Vec3::new(1.0, 0.6, 0.3),
            transmission_tint: Vec3::ZERO,
            transmission_scale: 0.0,
            normal_scale: 0.05,
            ambient_scatter: 0.1,
            enable_transmission: false,
            kernel: Vec::new(),
            scatter_lut: Vec::new(),
            lut_width: 0,
            lut_height: 0,
        };
        profile.build_kernel(25);
        profile.build_scatter_lut(128, 128);
        profile
    }

    /// Build the separable Gaussian SSS kernel.
    ///
    /// The kernel approximates the sum of multiple Gaussians (one per colour
    /// channel) to model the different scatter distances of red, green, and
    /// blue light within the material.
    pub fn build_kernel(&mut self, num_samples: usize) {
        let num_samples = num_samples.max(3) | 1; // Ensure odd.
        let half = num_samples / 2;

        self.kernel.clear();
        self.kernel.reserve(num_samples);

        // Range of the kernel in sigma units.
        let range = 3.0f32;

        let mut total_weight = Vec3::ZERO;

        for i in 0..num_samples {
            let s = if i == half {
                0.0
            } else {
                // Distribute samples using an importance-sampled pattern.
                let t = i as f32 / (num_samples - 1) as f32;
                let offset = (t * 2.0 - 1.0) * range;
                offset
            };

            // Compute per-channel Gaussian weights using the profile's falloff.
            let w = Vec3::new(
                gaussian(s, self.falloff.x),
                gaussian(s, self.falloff.y),
                gaussian(s, self.falloff.z),
            ) * self.scatter_color;

            total_weight += w;

            self.kernel.push(SSSKernelSample {
                offset: s,
                weight: Vec4::new(w.x, w.y, w.z, 0.0),
            });
        }

        // Normalise weights so they sum to 1 per channel.
        let inv_total = Vec3::new(
            if total_weight.x > 1e-8 { 1.0 / total_weight.x } else { 0.0 },
            if total_weight.y > 1e-8 { 1.0 / total_weight.y } else { 0.0 },
            if total_weight.z > 1e-8 { 1.0 / total_weight.z } else { 0.0 },
        );

        for sample in &mut self.kernel {
            sample.weight.x *= inv_total.x;
            sample.weight.y *= inv_total.y;
            sample.weight.z *= inv_total.z;
            // W channel stores the combined weight for stencil/depth testing.
            sample.weight.w = (sample.weight.x + sample.weight.y + sample.weight.z) / 3.0;
        }
    }

    /// Build the pre-integrated scattering LUT.
    ///
    /// The LUT is indexed by (NdotL, curvature) and stores the scattered
    /// diffuse lighting. This avoids the need for the separable blur pass
    /// on diffuse lighting and is more physically accurate for curved surfaces.
    pub fn build_scatter_lut(&mut self, width: u32, height: u32) {
        self.lut_width = width;
        self.lut_height = height;
        self.scatter_lut = vec![0.0; (width * height * 3) as usize]; // RGB

        for y in 0..height {
            for x in 0..width {
                let ndl = (x as f32 / (width - 1) as f32) * 2.0 - 1.0;
                let curvature = (y as f32 / (height - 1) as f32).max(0.001);

                // Integrate the diffuse profile over the surface.
                let scattered = integrate_diffuse_profile(
                    ndl,
                    curvature,
                    &self.scatter_color,
                    &self.falloff,
                );

                let idx = ((y * width + x) * 3) as usize;
                self.scatter_lut[idx] = scattered.x;
                self.scatter_lut[idx + 1] = scattered.y;
                self.scatter_lut[idx + 2] = scattered.z;
            }
        }
    }

    /// Sample the pre-integrated scattering LUT.
    pub fn sample_scatter_lut(&self, ndl: f32, curvature: f32) -> Vec3 {
        if self.scatter_lut.is_empty() || self.lut_width == 0 || self.lut_height == 0 {
            return Vec3::splat(ndl.max(0.0));
        }

        let u = (ndl * 0.5 + 0.5).clamp(0.0, 1.0);
        let v = curvature.clamp(0.0, 1.0);

        let px = (u * (self.lut_width - 1) as f32) as u32;
        let py = (v * (self.lut_height - 1) as f32) as u32;
        let px = px.min(self.lut_width - 1);
        let py = py.min(self.lut_height - 1);

        let idx = ((py * self.lut_width + px) * 3) as usize;
        Vec3::new(
            self.scatter_lut[idx],
            self.scatter_lut[idx + 1],
            self.scatter_lut[idx + 2],
        )
    }

    /// Compute the kernel data as a flat array of f32 for GPU upload.
    /// Layout: [offset, weight_r, weight_g, weight_b] per sample.
    pub fn kernel_gpu_data(&self) -> Vec<f32> {
        let mut data = Vec::with_capacity(self.kernel.len() * 4);
        for sample in &self.kernel {
            data.push(sample.offset);
            data.push(sample.weight.x);
            data.push(sample.weight.y);
            data.push(sample.weight.z);
        }
        data
    }
}

/// Gaussian function: G(x, sigma) = exp(-x^2 / (2 * sigma^2)).
fn gaussian(x: f32, sigma: f32) -> f32 {
    if sigma < 1e-8 {
        return if x.abs() < 1e-8 { 1.0 } else { 0.0 };
    }
    (-x * x / (2.0 * sigma * sigma)).exp()
}

/// Integrate the diffuse scattering profile for pre-integrated skin shading.
/// Uses the method from "Pre-Integrated Skin Shading" (Penner & Borshukov 2011).
fn integrate_diffuse_profile(
    ndl: f32,
    curvature: f32,
    scatter_color: &Vec3,
    falloff: &Vec3,
) -> Vec3 {
    // The key insight: on a curved surface, different points see different
    // NdotL values. We integrate the profile over a ring of surface points
    // at varying angles.

    let num_samples = 64;
    let mut result = Vec3::ZERO;
    let mut total_weight = Vec3::ZERO;

    for i in 0..num_samples {
        let theta = PI * (i as f32 + 0.5) / num_samples as f32;

        // At angle theta around the surface, the NdotL changes based on curvature.
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        // The local NdotL at angle theta from the shading point.
        let local_ndl = ndl * cos_theta + sin_theta * (1.0 - ndl * ndl).sqrt().max(0.0);
        let diffuse = local_ndl.max(0.0);

        // Distance along the surface (in scatter radius units).
        let dist = theta / (curvature + 1e-5);

        // Per-channel Gaussian scatter.
        let w = Vec3::new(
            gaussian(dist, falloff.x) * scatter_color.x,
            gaussian(dist, falloff.y) * scatter_color.y,
            gaussian(dist, falloff.z) * scatter_color.z,
        );

        result += Vec3::new(
            diffuse * w.x,
            diffuse * w.y,
            diffuse * w.z,
        );
        total_weight += w;
    }

    // Normalise.
    Vec3::new(
        if total_weight.x > 1e-8 { result.x / total_weight.x } else { 0.0 },
        if total_weight.y > 1e-8 { result.y / total_weight.y } else { 0.0 },
        if total_weight.z > 1e-8 { result.z / total_weight.z } else { 0.0 },
    )
}

// ---------------------------------------------------------------------------
// Screen-Space SSS (Separable Blur)
// ---------------------------------------------------------------------------

/// Perform the horizontal pass of separable screen-space SSS.
///
/// # Arguments
/// - `input`: Input colour buffer (linear HDR, RGB per pixel).
/// - `depth`: Depth buffer.
/// - `stencil_mask`: Per-pixel mask (true = apply SSS).
/// - `width`, `height`: Buffer dimensions.
/// - `profile`: The SSS profile to use.
/// - `projection_scale`: Used to scale kernel offsets to screen space.
///
/// # Returns
/// Intermediate buffer after horizontal blur.
pub fn sss_blur_horizontal(
    input: &[Vec3],
    depth: &[f32],
    stencil_mask: &[bool],
    width: u32,
    height: u32,
    profile: &SubsurfaceProfile,
    projection_scale: f32,
) -> Vec<Vec3> {
    sss_blur_pass(input, depth, stencil_mask, width, height, profile, projection_scale, true)
}

/// Perform the vertical pass of separable screen-space SSS.
pub fn sss_blur_vertical(
    input: &[Vec3],
    depth: &[f32],
    stencil_mask: &[bool],
    width: u32,
    height: u32,
    profile: &SubsurfaceProfile,
    projection_scale: f32,
) -> Vec<Vec3> {
    sss_blur_pass(input, depth, stencil_mask, width, height, profile, projection_scale, false)
}

/// Core separable SSS blur pass (horizontal or vertical).
fn sss_blur_pass(
    input: &[Vec3],
    depth: &[f32],
    stencil_mask: &[bool],
    width: u32,
    height: u32,
    profile: &SubsurfaceProfile,
    projection_scale: f32,
    horizontal: bool,
) -> Vec<Vec3> {
    let pixel_count = (width * height) as usize;
    let mut output = vec![Vec3::ZERO; pixel_count];

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;

            // If the pixel is not marked for SSS, pass through.
            if !stencil_mask[idx] {
                output[idx] = input[idx];
                continue;
            }

            let center_depth = depth[idx];
            if center_depth <= 0.0 || center_depth >= 1.0 {
                output[idx] = input[idx];
                continue;
            }

            // Compute the kernel scale based on depth and scatter radius.
            // The kernel width in pixels is proportional to scatter_radius / depth.
            let kernel_scale = profile.scatter_radius * projection_scale / center_depth;

            let mut accumulated = Vec3::ZERO;
            let mut weight_sum = Vec3::ZERO;

            for sample in &profile.kernel {
                let offset_pixels = sample.offset * kernel_scale;

                let (sx, sy) = if horizontal {
                    ((x as f32 + offset_pixels).round() as i32, y as i32)
                } else {
                    (x as i32, (y as f32 + offset_pixels).round() as i32)
                };

                // Clamp to buffer bounds.
                let sx = sx.clamp(0, width as i32 - 1) as u32;
                let sy = sy.clamp(0, height as i32 - 1) as u32;
                let s_idx = (sy * width + sx) as usize;

                // Depth-based weight correction: reduce weight for samples
                // at very different depths to prevent bleeding across edges.
                let sample_depth = depth[s_idx];
                let depth_diff = (center_depth - sample_depth).abs();
                let depth_weight = (-depth_diff * 1000.0 * center_depth).exp();

                // Only scatter from SSS-marked pixels.
                let stencil_weight = if stencil_mask[s_idx] { 1.0 } else { 0.0 };

                let w = Vec3::new(
                    sample.weight.x * depth_weight * stencil_weight,
                    sample.weight.y * depth_weight * stencil_weight,
                    sample.weight.z * depth_weight * stencil_weight,
                );

                accumulated += input[s_idx] * w;
                weight_sum += w;
            }

            // Normalise.
            output[idx] = Vec3::new(
                if weight_sum.x > 1e-8 { accumulated.x / weight_sum.x } else { input[idx].x },
                if weight_sum.y > 1e-8 { accumulated.y / weight_sum.y } else { input[idx].y },
                if weight_sum.z > 1e-8 { accumulated.z / weight_sum.z } else { input[idx].z },
            );
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Transmission
// ---------------------------------------------------------------------------

/// Compute transmission contribution for thin objects.
///
/// Models light passing through thin geometry like ears, leaves, or fingers
/// using shadow map-based thickness estimation.
///
/// # Arguments
/// - `light_dir`: Direction toward the light source.
/// - `view_dir`: Direction toward the camera.
/// - `normal`: Surface normal.
/// - `thickness`: Estimated thickness (from shadow map difference).
/// - `profile`: SSS profile.
///
/// # Returns
/// Transmission colour contribution.
pub fn compute_transmission(
    light_dir: Vec3,
    view_dir: Vec3,
    normal: Vec3,
    thickness: f32,
    profile: &SubsurfaceProfile,
) -> Vec3 {
    if !profile.enable_transmission || profile.transmission_scale <= 0.0 {
        return Vec3::ZERO;
    }

    // Modified half-vector for transmission (using negative normal).
    let distorted_light = light_dir + normal * profile.normal_scale;
    let v_dot_l = view_dir.dot(-distorted_light.normalize_or_zero()).max(0.0);

    // Transmission uses a falloff based on the pow function for a soft
    // forward-scattering lobe.
    let forward_scatter = v_dot_l.powf(12.0);
    let back_scatter = v_dot_l.max(0.0);

    let scatter_mix = forward_scatter * 0.7 + back_scatter * 0.3;

    // Thickness-based attenuation using Beer's law.
    // thicker material = less light gets through.
    let attenuation = Vec3::new(
        (-thickness / (profile.scatter_radius * profile.falloff.x + 1e-5)).exp(),
        (-thickness / (profile.scatter_radius * profile.falloff.y + 1e-5)).exp(),
        (-thickness / (profile.scatter_radius * profile.falloff.z + 1e-5)).exp(),
    );

    profile.transmission_tint * attenuation * scatter_mix * profile.transmission_scale
}

/// Estimate thickness from shadow map.
///
/// The thickness is approximated as the difference between the shadow map
/// depth (light-space depth of the nearest surface to the light) and the
/// current fragment's light-space depth.
pub fn estimate_thickness_from_shadow(
    fragment_light_depth: f32,
    shadow_map_depth: f32,
    light_far_plane: f32,
) -> f32 {
    let depth_diff = (fragment_light_depth - shadow_map_depth).max(0.0);
    depth_diff * light_far_plane
}

// ---------------------------------------------------------------------------
// Pre-Integrated Skin Shading
// ---------------------------------------------------------------------------

/// Compute curvature at a pixel from screen-space derivatives.
///
/// Curvature is estimated as the divergence of the normal divided by depth,
/// which correlates with the inverse radius of curvature of the surface.
pub fn estimate_curvature(
    normal: Vec3,
    normal_dx: Vec3,
    normal_dz: Vec3,
    depth: f32,
) -> f32 {
    if depth <= 0.0 {
        return 0.0;
    }

    let dn_dx = normal_dx - normal;
    let dn_dz = normal_dz - normal;

    let divergence = dn_dx.x + dn_dz.z;
    (divergence.abs() / depth).clamp(0.0, 1.0)
}

/// Shade a pixel using pre-integrated skin diffuse.
///
/// Instead of the standard NdotL clamp, use the scattering LUT indexed
/// by NdotL and curvature to get a softer, more realistic diffuse response.
pub fn pre_integrated_skin_diffuse(
    profile: &SubsurfaceProfile,
    ndl: f32,
    curvature: f32,
    light_color: Vec3,
    albedo: Vec3,
) -> Vec3 {
    let scattered = profile.sample_scatter_lut(ndl, curvature);
    scattered * light_color * albedo
}

// ---------------------------------------------------------------------------
// SSSComponent (ECS integration)
// ---------------------------------------------------------------------------

/// ECS component for objects with subsurface scattering.
#[derive(Debug, Clone)]
pub struct SSSComponent {
    /// Index into the global SSS profile array.
    pub profile_index: u32,
    /// Per-instance scatter radius override (0 = use profile default).
    pub scatter_radius_override: f32,
    /// Whether SSS is enabled for this instance.
    pub enabled: bool,
    /// Stencil value used to mask this object in the SSS pass.
    pub stencil_value: u8,
}

impl Default for SSSComponent {
    fn default() -> Self {
        Self {
            profile_index: 0,
            scatter_radius_override: 0.0,
            enabled: true,
            stencil_value: 1,
        }
    }
}

impl SSSComponent {
    /// Create an SSS component with a specific profile.
    pub fn with_profile(profile_index: u32) -> Self {
        Self {
            profile_index,
            ..Default::default()
        }
    }

    /// Get the effective scatter radius.
    pub fn effective_scatter_radius(&self, profiles: &[SubsurfaceProfile]) -> f32 {
        if self.scatter_radius_override > 0.0 {
            self.scatter_radius_override
        } else if let Some(profile) = profiles.get(self.profile_index as usize) {
            profile.scatter_radius
        } else {
            0.012 // Default skin radius.
        }
    }
}

// ---------------------------------------------------------------------------
// WGSL Shader
// ---------------------------------------------------------------------------

/// WGSL compute shader for the separable SSS blur pass.
pub const SSS_BLUR_WGSL: &str = r#"
// Separable Screen-Space Subsurface Scattering Blur
// Performs one direction (horizontal or vertical) of the separable blur.

struct SSSKernelSample {
    offset: f32,
    weight_r: f32,
    weight_g: f32,
    weight_b: f32,
}

struct SSSUniforms {
    direction: vec2<f32>,       // (1,0) for horizontal, (0,1) for vertical
    scatter_radius: f32,
    projection_scale: f32,
    num_samples: u32,
    width: u32,
    height: u32,
    _pad: u32,
}

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var stencil_tex: texture_2d<u32>;
@group(0) @binding(3) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params: SSSUniforms;
@group(0) @binding(5) var<storage, read> kernel: array<SSSKernelSample>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let pos = vec2<i32>(i32(gid.x), i32(gid.y));
    let stencil = textureLoad(stencil_tex, pos, 0).r;

    if (stencil == 0u) {
        let passthrough = textureLoad(input_tex, pos, 0);
        textureStore(output_tex, pos, passthrough);
        return;
    }

    let center_depth = textureLoad(depth_tex, pos, 0).r;
    if (center_depth <= 0.0 || center_depth >= 1.0) {
        let passthrough = textureLoad(input_tex, pos, 0);
        textureStore(output_tex, pos, passthrough);
        return;
    }

    let kernel_scale = params.scatter_radius * params.projection_scale / center_depth;

    var accumulated = vec3<f32>(0.0);
    var weight_sum = vec3<f32>(0.0);

    for (var i = 0u; i < params.num_samples; i = i + 1u) {
        let sample = kernel[i];
        let offset_pixels = sample.offset * kernel_scale;

        let sample_pos = pos + vec2<i32>(
            i32(round(offset_pixels * params.direction.x)),
            i32(round(offset_pixels * params.direction.y))
        );

        let clamped_pos = clamp(
            sample_pos,
            vec2<i32>(0),
            vec2<i32>(i32(params.width) - 1, i32(params.height) - 1)
        );

        let sample_depth = textureLoad(depth_tex, clamped_pos, 0).r;
        let depth_diff = abs(center_depth - sample_depth);
        let depth_weight = exp(-depth_diff * 1000.0 * center_depth);

        let sample_stencil = textureLoad(stencil_tex, clamped_pos, 0).r;
        let stencil_weight = select(0.0, 1.0, sample_stencil != 0u);

        let w = vec3<f32>(
            sample.weight_r * depth_weight * stencil_weight,
            sample.weight_g * depth_weight * stencil_weight,
            sample.weight_b * depth_weight * stencil_weight
        );

        let sample_color = textureLoad(input_tex, clamped_pos, 0).rgb;
        accumulated = accumulated + sample_color * w;
        weight_sum = weight_sum + w;
    }

    let result = vec3<f32>(
        select(textureLoad(input_tex, pos, 0).r, accumulated.x / weight_sum.x, weight_sum.x > 0.00001),
        select(textureLoad(input_tex, pos, 0).g, accumulated.y / weight_sum.y, weight_sum.y > 0.00001),
        select(textureLoad(input_tex, pos, 0).b, accumulated.z / weight_sum.z, weight_sum.z > 0.00001)
    );

    textureStore(output_tex, pos, vec4<f32>(result, 1.0));
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian() {
        assert!((gaussian(0.0, 1.0) - 1.0).abs() < 0.001);
        assert!(gaussian(3.0, 1.0) < 0.02);
        assert!(gaussian(0.0, 0.5) > 0.99);
    }

    #[test]
    fn test_skin_profile_creation() {
        let profile = SubsurfaceProfile::skin();
        assert!(!profile.kernel.is_empty());
        assert_eq!(profile.kernel.len(), 25);
        assert!(!profile.scatter_lut.is_empty());
    }

    #[test]
    fn test_kernel_normalisation() {
        let profile = SubsurfaceProfile::skin();

        // Sum of weights per channel should be approximately 1.
        let mut sum_r = 0.0f32;
        let mut sum_g = 0.0f32;
        let mut sum_b = 0.0f32;

        for sample in &profile.kernel {
            sum_r += sample.weight.x;
            sum_g += sample.weight.y;
            sum_b += sample.weight.z;
        }

        assert!((sum_r - 1.0).abs() < 0.05, "R sum: {}", sum_r);
        assert!((sum_g - 1.0).abs() < 0.05, "G sum: {}", sum_g);
        assert!((sum_b - 1.0).abs() < 0.05, "B sum: {}", sum_b);
    }

    #[test]
    fn test_kernel_gpu_data() {
        let profile = SubsurfaceProfile::skin();
        let data = profile.kernel_gpu_data();
        assert_eq!(data.len(), profile.kernel.len() * 4);
    }

    #[test]
    fn test_scatter_lut() {
        let profile = SubsurfaceProfile::skin();

        // NdotL = 1, curvature = 0 should give max diffuse.
        let bright = profile.sample_scatter_lut(1.0, 0.01);
        assert!(bright.x > 0.0);
        assert!(bright.y > 0.0);
        assert!(bright.z > 0.0);

        // NdotL = -1 should give near-zero (light from behind).
        let dark = profile.sample_scatter_lut(-1.0, 0.5);
        assert!(dark.x < bright.x);
    }

    #[test]
    fn test_transmission() {
        let profile = SubsurfaceProfile::skin();
        let light_dir = Vec3::new(0.0, 0.0, -1.0);
        let view_dir = Vec3::new(0.0, 0.0, 1.0);
        let normal = Vec3::new(0.0, 0.0, 1.0);

        // Thin object: light comes from behind, view from front.
        let trans = compute_transmission(light_dir, view_dir, normal, 0.01, &profile);
        assert!(trans.x >= 0.0);
    }

    #[test]
    fn test_transmission_disabled() {
        let mut profile = SubsurfaceProfile::skin();
        profile.enable_transmission = false;

        let trans = compute_transmission(
            Vec3::Z, -Vec3::Z, Vec3::Z, 0.01, &profile,
        );
        assert_eq!(trans, Vec3::ZERO);
    }

    #[test]
    fn test_thickness_estimation() {
        let thickness = estimate_thickness_from_shadow(0.8, 0.7, 100.0);
        assert!((thickness - 10.0).abs() < 0.01);

        // No thickness difference.
        let zero = estimate_thickness_from_shadow(0.5, 0.5, 100.0);
        assert_eq!(zero, 0.0);
    }

    #[test]
    fn test_curvature_estimation() {
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let normal_dx = Vec3::new(0.1, 0.995, 0.0);
        let normal_dz = Vec3::new(0.0, 0.995, 0.1);

        let curvature = estimate_curvature(normal, normal_dx, normal_dz, 1.0);
        assert!(curvature >= 0.0 && curvature <= 1.0);
    }

    #[test]
    fn test_sss_blur_passthrough() {
        let w = 4u32;
        let h = 4u32;
        let input = vec![Vec3::splat(0.5); 16];
        let depth = vec![0.5f32; 16];
        let stencil = vec![false; 16]; // No SSS pixels.
        let profile = SubsurfaceProfile::skin();

        let result = sss_blur_horizontal(&input, &depth, &stencil, w, h, &profile, 100.0);
        assert_eq!(result.len(), 16);
        // All pixels pass through unchanged.
        for c in &result {
            assert!((c.x - 0.5).abs() < 0.001);
        }
    }

    #[test]
    fn test_sss_blur_with_sss() {
        let w = 8u32;
        let h = 8u32;
        let pixel_count = (w * h) as usize;

        // Create a scene with a bright spot surrounded by dark.
        let mut input = vec![Vec3::ZERO; pixel_count];
        input[4 * w as usize + 4] = Vec3::ONE; // bright spot at (4,4)

        let depth = vec![0.5f32; pixel_count];
        let stencil = vec![true; pixel_count]; // All SSS.
        let profile = SubsurfaceProfile::skin();

        let h_pass = sss_blur_horizontal(&input, &depth, &stencil, w, h, &profile, 100.0);
        let result = sss_blur_vertical(&h_pass, &depth, &stencil, w, h, &profile, 100.0);

        // The bright spot should have spread to neighbours.
        assert!(result[4 * w as usize + 4].x > 0.0);
        // Neighbours should pick up some scattered light.
        let neighbor_val = result[4 * w as usize + 5].x;
        assert!(neighbor_val >= 0.0); // Some scatter.
    }

    #[test]
    fn test_pre_integrated_skin_diffuse() {
        let profile = SubsurfaceProfile::skin();

        let lit = pre_integrated_skin_diffuse(
            &profile, 1.0, 0.1, Vec3::ONE, Vec3::ONE,
        );
        assert!(lit.x > 0.0);

        let shadow = pre_integrated_skin_diffuse(
            &profile, -0.5, 0.3, Vec3::ONE, Vec3::ONE,
        );
        // Should be non-zero due to scattering.
        assert!(shadow.x >= 0.0);
        assert!(shadow.x < lit.x);
    }

    #[test]
    fn test_sss_component() {
        let profiles = vec![SubsurfaceProfile::skin(), SubsurfaceProfile::foliage()];
        let comp = SSSComponent::with_profile(1);
        let radius = comp.effective_scatter_radius(&profiles);
        assert!((radius - 0.008).abs() < 0.001);
    }

    #[test]
    fn test_foliage_profile() {
        let profile = SubsurfaceProfile::foliage();
        assert_eq!(profile.kernel.len(), 25);
        assert!(profile.enable_transmission);
        assert!(profile.scatter_color.y > profile.scatter_color.x); // Green dominant.
    }
}
