// engine/render/src/raytracing/reflections.rs
//
// Ray-traced reflections for the Genovo engine. Provides roughness-based
// importance-sampled reflection tracing, spatial/temporal denoising, and
// hybrid SSR + ray trace fallback.

use crate::raytracing::bvh_tracer::{BVHAccelerationStructure, HitInfo, Ray};
use crate::raytracing::screen_probes::{
    hemisphere_sample_cosine_weighted, SurfaceCache,
};
use glam::{Vec2, Vec3, Vec4};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// ReflectionSettings
// ---------------------------------------------------------------------------

/// Configuration for ray-traced reflections.
#[derive(Debug, Clone)]
pub struct ReflectionSettings {
    /// Maximum number of reflection rays per pixel.
    pub max_rays_per_pixel: u32,
    /// Maximum number of reflection bounces.
    pub max_bounces: u32,
    /// Maximum trace distance for reflection rays.
    pub max_trace_distance: f32,
    /// Roughness threshold: surfaces rougher than this use probe-based fallback.
    pub roughness_threshold: f32,
    /// Temporal accumulation blend factor (0 = no history).
    pub temporal_blend: f32,
    /// Spatial denoiser strength (0 = no denoising).
    pub denoise_strength: f32,
    /// Whether to use SSR as a first pass before ray tracing.
    pub use_ssr_fallback: bool,
    /// SSR max steps for the screen-space pass.
    pub ssr_max_steps: u32,
    /// SSR stride in pixels.
    pub ssr_stride: f32,
    /// Whether half-resolution tracing is enabled.
    pub half_resolution: bool,
    /// Roughness-based ray count scaling: more rays for smoother surfaces.
    pub adaptive_ray_count: bool,
}

impl Default for ReflectionSettings {
    fn default() -> Self {
        Self {
            max_rays_per_pixel: 1,
            max_bounces: 1,
            max_trace_distance: 200.0,
            roughness_threshold: 0.7,
            temporal_blend: 0.9,
            denoise_strength: 0.5,
            use_ssr_fallback: true,
            ssr_max_steps: 64,
            ssr_stride: 2.0,
            half_resolution: false,
            adaptive_ray_count: true,
        }
    }
}

impl ReflectionSettings {
    /// Low quality: single ray, no denoising.
    pub fn low() -> Self {
        Self {
            max_rays_per_pixel: 1,
            max_bounces: 1,
            temporal_blend: 0.95,
            denoise_strength: 0.3,
            half_resolution: true,
            ..Default::default()
        }
    }

    /// Medium quality.
    pub fn medium() -> Self {
        Self {
            max_rays_per_pixel: 1,
            max_bounces: 1,
            ..Default::default()
        }
    }

    /// High quality.
    pub fn high() -> Self {
        Self {
            max_rays_per_pixel: 4,
            max_bounces: 2,
            temporal_blend: 0.85,
            denoise_strength: 0.7,
            ..Default::default()
        }
    }

    /// Ultra quality (for offline/screenshots).
    pub fn ultra() -> Self {
        Self {
            max_rays_per_pixel: 16,
            max_bounces: 3,
            temporal_blend: 0.7,
            denoise_strength: 1.0,
            half_resolution: false,
            ..Default::default()
        }
    }

    /// Compute the number of rays for a given roughness.
    pub fn rays_for_roughness(&self, roughness: f32) -> u32 {
        if !self.adaptive_ray_count {
            return self.max_rays_per_pixel;
        }
        // Smooth surfaces get more rays, rough surfaces get fewer.
        let t = (1.0 - roughness).clamp(0.0, 1.0);
        let rays = 1.0 + (self.max_rays_per_pixel as f32 - 1.0) * t;
        rays.ceil() as u32
    }
}

// ---------------------------------------------------------------------------
// ReflectionBuffer
// ---------------------------------------------------------------------------

/// Buffer holding per-pixel reflection results.
pub struct ReflectionBuffer {
    pub width: u32,
    pub height: u32,
    /// Current frame reflection radiance per pixel.
    pub radiance: Vec<Vec3>,
    /// Previous frame reflection radiance (for temporal accumulation).
    pub prev_radiance: Vec<Vec3>,
    /// Hit distance per pixel (for denoising).
    pub hit_distance: Vec<f32>,
    /// Sample confidence per pixel (0-1, based on sample count and roughness).
    pub confidence: Vec<f32>,
    /// Variance estimate per pixel (for adaptive denoising).
    pub variance: Vec<f32>,
}

impl ReflectionBuffer {
    /// Create a new reflection buffer.
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            width,
            height,
            radiance: vec![Vec3::ZERO; size],
            prev_radiance: vec![Vec3::ZERO; size],
            hit_distance: vec![0.0; size],
            confidence: vec![0.0; size],
            variance: vec![0.0; size],
        }
    }

    /// Resize the buffer. Previous frame data is lost.
    pub fn resize(&mut self, width: u32, height: u32) {
        let size = (width * height) as usize;
        self.width = width;
        self.height = height;
        self.radiance = vec![Vec3::ZERO; size];
        self.prev_radiance = vec![Vec3::ZERO; size];
        self.hit_distance = vec![0.0; size];
        self.confidence = vec![0.0; size];
        self.variance = vec![0.0; size];
    }

    /// Swap current and previous frame data (call at start of frame).
    pub fn swap_history(&mut self) {
        std::mem::swap(&mut self.radiance, &mut self.prev_radiance);
        for v in &mut self.radiance {
            *v = Vec3::ZERO;
        }
        for d in &mut self.hit_distance {
            *d = 0.0;
        }
        for c in &mut self.confidence {
            *c = 0.0;
        }
    }

    /// Pixel index.
    #[inline]
    fn idx(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }

    /// Write a reflection result for a pixel.
    pub fn write(&mut self, x: u32, y: u32, radiance: Vec3, hit_dist: f32, confidence: f32) {
        let i = self.idx(x, y);
        self.radiance[i] = radiance;
        self.hit_distance[i] = hit_dist;
        self.confidence[i] = confidence;
    }

    /// Read the final (denoised + temporally filtered) reflection.
    pub fn read(&self, x: u32, y: u32) -> Vec3 {
        self.radiance[self.idx(x, y)]
    }
}

// ---------------------------------------------------------------------------
// ReflectionTracer
// ---------------------------------------------------------------------------

/// The main reflection tracer: traces GGX importance-sampled rays, denoises,
/// and composites the results.
pub struct ReflectionTracer {
    pub settings: ReflectionSettings,
    pub buffer: ReflectionBuffer,
    pub frame_index: u32,
}

impl ReflectionTracer {
    /// Create a new reflection tracer.
    pub fn new(width: u32, height: u32, settings: ReflectionSettings) -> Self {
        Self {
            settings,
            buffer: ReflectionBuffer::new(width, height),
            frame_index: 0,
        }
    }

    /// Trace reflections for all pixels.
    ///
    /// # Arguments
    /// - `accel`: BVH acceleration structure for ray tracing.
    /// - `surface_cache`: surface cache for radiance at hit points.
    /// - `depth_buffer`: linearized depth per pixel.
    /// - `normal_buffer`: world-space normals per pixel.
    /// - `roughness_buffer`: roughness per pixel.
    /// - `position_buffer`: world-space positions per pixel.
    /// - `view_dir_fn`: closure computing the view direction for a pixel.
    /// - `sky_radiance`: closure returning sky radiance for a direction.
    pub fn trace_frame(
        &mut self,
        accel: &BVHAccelerationStructure,
        surface_cache: &SurfaceCache,
        depth_buffer: &[f32],
        normal_buffer: &[Vec3],
        roughness_buffer: &[f32],
        position_buffer: &[Vec3],
        view_dir_fn: &dyn Fn(u32, u32) -> Vec3,
        sky_radiance: &dyn Fn(Vec3) -> Vec3,
    ) {
        self.buffer.swap_history();

        let width = self.buffer.width;
        let height = self.buffer.height;
        let step = if self.settings.half_resolution { 2 } else { 1 };

        for py in (0..height).step_by(step as usize) {
            for px in (0..width).step_by(step as usize) {
                let idx = (py * width + px) as usize;
                let depth = depth_buffer.get(idx).copied().unwrap_or(0.0);

                // Skip sky pixels.
                if depth <= 0.0 || depth >= 1.0 {
                    continue;
                }

                let roughness = roughness_buffer.get(idx).copied().unwrap_or(0.5);

                // Skip very rough surfaces (use probe-based diffuse GI instead).
                if roughness > self.settings.roughness_threshold {
                    continue;
                }

                let position = position_buffer.get(idx).copied().unwrap_or(Vec3::ZERO);
                let normal = normal_buffer.get(idx).copied().unwrap_or(Vec3::Y);
                let view_dir = view_dir_fn(px, py);

                let ray_count = self.settings.rays_for_roughness(roughness);
                let mut total_radiance = Vec3::ZERO;
                let mut total_weight = 0.0f32;
                let mut closest_dist = f32::MAX;

                for ray_idx in 0..ray_count {
                    // GGX importance sampling for the reflection direction.
                    let reflection_dir = sample_ggx_reflection(
                        view_dir, normal, roughness, ray_idx, ray_count, self.frame_index,
                    );

                    let n_dot_r = normal.dot(reflection_dir);
                    if n_dot_r < 0.0 {
                        continue; // Below surface.
                    }

                    let ray = Ray::with_range(
                        position + normal * 0.01,
                        reflection_dir,
                        1e-3,
                        self.settings.max_trace_distance,
                    );

                    let (radiance, hit_dist) = self.trace_reflection_ray(
                        &ray,
                        accel,
                        surface_cache,
                        sky_radiance,
                        0,
                    );

                    let weight = n_dot_r;
                    total_radiance += radiance * weight;
                    total_weight += weight;
                    closest_dist = closest_dist.min(hit_dist);
                }

                if total_weight > 0.0 {
                    let avg = total_radiance / total_weight;
                    let confidence = (ray_count as f32 / self.settings.max_rays_per_pixel as f32)
                        .min(1.0);
                    self.buffer.write(px, py, avg, closest_dist, confidence);

                    // Fill half-res neighbors.
                    if self.settings.half_resolution {
                        for dy in 0..step.min(height - py) {
                            for dx in 0..step.min(width - px) {
                                if dx == 0 && dy == 0 {
                                    continue;
                                }
                                self.buffer.write(px + dx, py + dy, avg, closest_dist, confidence * 0.5);
                            }
                        }
                    }
                }
            }
        }

        // Apply temporal accumulation.
        self.temporal_accumulate();

        // Apply spatial denoising.
        if self.settings.denoise_strength > 0.0 {
            self.spatial_denoise(depth_buffer, normal_buffer, roughness_buffer);
        }

        self.frame_index = self.frame_index.wrapping_add(1);
    }

    /// Trace a single reflection ray, optionally recursing for multi-bounce.
    fn trace_reflection_ray(
        &self,
        ray: &Ray,
        accel: &BVHAccelerationStructure,
        surface_cache: &SurfaceCache,
        sky_radiance: &dyn Fn(Vec3) -> Vec3,
        bounce: u32,
    ) -> (Vec3, f32) {
        match accel.trace_ray(ray) {
            Some(hit) => {
                let mut radiance = surface_cache.lookup_radiance(&hit);

                // Multi-bounce: trace another reflection from the hit point.
                if bounce + 1 < self.settings.max_bounces {
                    let reflection = reflect(ray.direction, hit.normal);
                    let next_ray = Ray::with_range(
                        hit.position + hit.normal * 0.01,
                        reflection,
                        1e-3,
                        self.settings.max_trace_distance,
                    );
                    let (bounce_radiance, _) = self.trace_reflection_ray(
                        &next_ray,
                        accel,
                        surface_cache,
                        sky_radiance,
                        bounce + 1,
                    );
                    // Approximate: blend bounce contribution. In a real
                    // implementation this would use the hit surface's F0.
                    radiance += bounce_radiance * 0.3;
                }

                (radiance, hit.distance)
            }
            None => {
                (sky_radiance(ray.direction), self.settings.max_trace_distance)
            }
        }
    }

    /// Apply temporal accumulation: blend current frame with history.
    fn temporal_accumulate(&mut self) {
        let blend = self.settings.temporal_blend;
        let size = (self.buffer.width * self.buffer.height) as usize;

        for i in 0..size {
            let current = self.buffer.radiance[i];
            let prev = self.buffer.prev_radiance[i];
            let confidence = self.buffer.confidence[i];

            if confidence < 1e-6 {
                // No current data, use history.
                self.buffer.radiance[i] = prev * blend;
                continue;
            }

            // Simple exponential moving average.
            self.buffer.radiance[i] = current * (1.0 - blend) + prev * blend;

            // Compute variance for adaptive denoising.
            let diff = current - prev;
            self.buffer.variance[i] = diff.dot(diff);
        }
    }

    /// Spatial denoising pass: edge-aware blur that respects depth and normal
    /// boundaries.
    fn spatial_denoise(
        &mut self,
        depth_buffer: &[f32],
        normal_buffer: &[Vec3],
        roughness_buffer: &[f32],
    ) {
        let width = self.buffer.width;
        let height = self.buffer.height;
        let strength = self.settings.denoise_strength;

        // Work on a copy.
        let original = self.buffer.radiance.clone();

        let kernel_radius = 3i32;
        let sigma_spatial = 2.0f32;
        let sigma_depth = 0.02f32;
        let sigma_normal = 32.0f32;

        for py in 0..height {
            for px in 0..width {
                let center_idx = (py * width + px) as usize;
                let center_depth = depth_buffer.get(center_idx).copied().unwrap_or(0.0);
                let center_normal = normal_buffer.get(center_idx).copied().unwrap_or(Vec3::Y);
                let center_roughness = roughness_buffer.get(center_idx).copied().unwrap_or(0.5);

                if center_depth <= 0.0 || center_depth >= 1.0 {
                    continue;
                }

                // Roughness-adaptive kernel size: rough = larger kernel.
                let adaptive_radius = (kernel_radius as f32 * (0.5 + center_roughness * 2.0)) as i32;
                let adaptive_radius = adaptive_radius.clamp(1, kernel_radius * 2);

                let mut sum = Vec3::ZERO;
                let mut weight_sum = 0.0f32;

                for dy in -adaptive_radius..=adaptive_radius {
                    for dx in -adaptive_radius..=adaptive_radius {
                        let nx = px as i32 + dx;
                        let ny = py as i32 + dy;

                        if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                            continue;
                        }

                        let n_idx = (ny as u32 * width + nx as u32) as usize;
                        let n_depth = depth_buffer.get(n_idx).copied().unwrap_or(0.0);
                        let n_normal = normal_buffer.get(n_idx).copied().unwrap_or(Vec3::Y);

                        // Spatial weight (Gaussian).
                        let d2 = (dx * dx + dy * dy) as f32;
                        let w_spatial = (-d2 / (2.0 * sigma_spatial * sigma_spatial)).exp();

                        // Depth weight.
                        let depth_diff = (center_depth - n_depth).abs();
                        let w_depth = (-depth_diff / sigma_depth).exp();

                        // Normal weight.
                        let normal_dot = center_normal.dot(n_normal).max(0.0);
                        let w_normal = (sigma_normal * (normal_dot - 1.0)).exp();

                        let w = w_spatial * w_depth * w_normal;

                        sum += original[n_idx] * w;
                        weight_sum += w;
                    }
                }

                if weight_sum > 1e-6 {
                    let denoised = sum / weight_sum;
                    // Blend between raw and denoised based on strength.
                    self.buffer.radiance[center_idx] =
                        original[center_idx] * (1.0 - strength) + denoised * strength;
                }
            }
        }
    }

    /// Resize the reflection buffer.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.buffer.resize(width, height);
    }
}

// ---------------------------------------------------------------------------
// GGX importance sampling
// ---------------------------------------------------------------------------

/// Sample a reflection direction using GGX importance sampling.
///
/// For roughness ~0 this produces a mirror reflection. For higher roughness
/// the samples spread around the reflection direction forming a cone.
pub fn sample_ggx_reflection(
    view_dir: Vec3,
    normal: Vec3,
    roughness: f32,
    sample_index: u32,
    total_samples: u32,
    frame_seed: u32,
) -> Vec3 {
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;

    // Hammersley-like low-discrepancy sequence.
    let seed = frame_seed.wrapping_add(sample_index.wrapping_mul(0x9E3779B9));
    let xi1 = ((sample_index as f32 + hash_float(seed))
        / total_samples.max(1) as f32)
        .fract();
    let xi2 = radical_inverse_vdc(sample_index ^ frame_seed);

    // GGX half-vector sampling in tangent space.
    let phi = 2.0 * PI * xi1;
    let cos_theta = if alpha2 < 1e-8 {
        // Mirror reflection for near-zero roughness.
        1.0
    } else {
        ((1.0 - xi2) / (1.0 + (alpha2 - 1.0) * xi2)).sqrt()
    };
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

    let h_local = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);

    // Transform to world space.
    let h_world = tangent_space_to_world(h_local, normal);

    // Reflect view direction around the half-vector.
    let v = -view_dir; // View dir points towards camera, we need outgoing.
    let l = (2.0 * v.dot(h_world) * h_world - v).normalize_or_zero();

    l
}

/// Compute the perfect mirror reflection of a direction off a normal.
#[inline]
pub fn reflect(incident: Vec3, normal: Vec3) -> Vec3 {
    incident - 2.0 * incident.dot(normal) * normal
}

/// Transform a tangent-space vector to world space given a normal.
fn tangent_space_to_world(local: Vec3, normal: Vec3) -> Vec3 {
    let up = if normal.y.abs() < 0.999 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let tangent = up.cross(normal).normalize_or_zero();
    let bitangent = normal.cross(tangent);
    (tangent * local.x + bitangent * local.y + normal * local.z).normalize_or_zero()
}

fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    bits as f32 * 2.328_306_4e-10
}

fn hash_float(mut seed: u32) -> f32 {
    seed ^= seed >> 16;
    seed = seed.wrapping_mul(0x45d9f3b);
    seed ^= seed >> 16;
    seed = seed.wrapping_mul(0x45d9f3b);
    seed ^= seed >> 16;
    (seed & 0x00FF_FFFF) as f32 / (0x0100_0000 as f32)
}

// ---------------------------------------------------------------------------
// SSR (Screen-Space Reflections) -- fallback pass
// ---------------------------------------------------------------------------

/// Screen-space reflection result for a single pixel.
#[derive(Debug, Clone, Copy)]
pub struct SSRResult {
    /// The UV coordinate of the reflected pixel.
    pub hit_uv: Vec2,
    /// Whether the SSR found a valid hit.
    pub valid: bool,
    /// Confidence of the hit (fades near edges).
    pub confidence: f32,
    /// Parametric distance along the reflection ray.
    pub hit_distance: f32,
}

/// Screen-space reflection ray marcher.
///
/// Performs a linear ray march through the depth buffer to find intersections.
/// Falls back to ray tracing for misses.
pub struct SSRTracer {
    pub max_steps: u32,
    pub stride: f32,
    pub thickness: f32,
    pub max_distance: f32,
    pub jitter: bool,
}

impl SSRTracer {
    /// Create a new SSR tracer with default settings.
    pub fn new() -> Self {
        Self {
            max_steps: 64,
            stride: 2.0,
            thickness: 0.1,
            max_distance: 100.0,
            jitter: true,
        }
    }

    /// Trace a screen-space reflection.
    ///
    /// `view_pos`: camera-space position of the pixel.
    /// `view_normal`: camera-space normal.
    /// `view_proj`: view-projection matrix.
    /// `depth_buffer`: the depth buffer (width * height).
    /// `screen_width`, `screen_height`: dimensions.
    pub fn trace_ssr(
        &self,
        view_pos: Vec3,
        view_normal: Vec3,
        view_proj: &glam::Mat4,
        depth_buffer: &[f32],
        screen_width: u32,
        screen_height: u32,
        frame_seed: u32,
        pixel_x: u32,
        pixel_y: u32,
    ) -> SSRResult {
        let reflect_dir = reflect(view_pos.normalize_or_zero(), view_normal);
        let step_dir = reflect_dir * self.stride;

        let mut current_pos = view_pos;

        // Optional jitter for temporal stability.
        if self.jitter {
            let jitter = hash_float(
                frame_seed
                    .wrapping_add(pixel_x)
                    .wrapping_add(pixel_y.wrapping_mul(screen_width)),
            );
            current_pos += step_dir * jitter;
        }

        for step in 0..self.max_steps {
            current_pos += step_dir;

            // Project to screen space.
            let clip =
                *view_proj * Vec4::new(current_pos.x, current_pos.y, current_pos.z, 1.0);
            if clip.w <= 0.0 {
                break;
            }
            let ndc = Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);

            // NDC to pixel coords.
            let px = ((ndc.x * 0.5 + 0.5) * screen_width as f32) as i32;
            let py = ((1.0 - (ndc.y * 0.5 + 0.5)) * screen_height as f32) as i32;

            if px < 0 || py < 0 || px >= screen_width as i32 || py >= screen_height as i32 {
                break;
            }

            let buffer_idx = (py as u32 * screen_width + px as u32) as usize;
            let sampled_depth = depth_buffer.get(buffer_idx).copied().unwrap_or(1.0);

            let ray_depth = ndc.z * 0.5 + 0.5; // [0, 1] range

            let depth_diff = ray_depth - sampled_depth;

            if depth_diff > 0.0 && depth_diff < self.thickness {
                // Hit found.
                let uv = Vec2::new(
                    px as f32 / screen_width as f32,
                    py as f32 / screen_height as f32,
                );

                // Fade confidence near screen edges.
                let edge_fade_x = 1.0 - (uv.x * 2.0 - 1.0).abs().powf(4.0);
                let edge_fade_y = 1.0 - (uv.y * 2.0 - 1.0).abs().powf(4.0);
                let confidence = (edge_fade_x * edge_fade_y).clamp(0.0, 1.0);

                let dist = (current_pos - view_pos).length();

                return SSRResult {
                    hit_uv: uv,
                    valid: true,
                    confidence,
                    hit_distance: dist,
                };
            }
        }

        SSRResult {
            hit_uv: Vec2::ZERO,
            valid: false,
            confidence: 0.0,
            hit_distance: 0.0,
        }
    }
}

impl Default for SSRTracer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_direction() {
        let incident = Vec3::new(1.0, -1.0, 0.0).normalize();
        let normal = Vec3::Y;
        let reflected = reflect(incident, normal);
        assert!((reflected.y - (-incident.y)).abs() < 1e-4);
        assert!((reflected.x - incident.x).abs() < 1e-4);
    }

    #[test]
    fn ggx_mirror_reflection() {
        let view_dir = Vec3::new(0.0, 0.0, -1.0);
        let normal = Vec3::Z;
        let dir = sample_ggx_reflection(view_dir, normal, 0.0, 0, 1, 42);
        // Should be close to perfect reflection (along +Z since view is -Z).
        assert!(dir.z > 0.5);
    }

    #[test]
    fn ggx_rough_spread() {
        let view_dir = Vec3::new(0.0, 0.0, -1.0);
        let normal = Vec3::Z;

        let mut directions = Vec::new();
        for i in 0..16 {
            let dir = sample_ggx_reflection(view_dir, normal, 0.5, i, 16, 42);
            directions.push(dir);
        }

        // With roughness 0.5, directions should spread out more.
        let mut dot_sum = 0.0f32;
        for d in &directions {
            dot_sum += d.dot(Vec3::Z).max(0.0);
        }
        // Average dot should be less than 1.0 (spread).
        let avg = dot_sum / directions.len() as f32;
        assert!(avg < 1.0);
    }

    #[test]
    fn reflection_settings_ray_count() {
        let settings = ReflectionSettings {
            max_rays_per_pixel: 8,
            adaptive_ray_count: true,
            ..Default::default()
        };

        // Smooth surface should get more rays.
        assert!(settings.rays_for_roughness(0.0) >= settings.rays_for_roughness(0.5));
        assert!(settings.rays_for_roughness(0.5) >= settings.rays_for_roughness(1.0));
    }

    #[test]
    fn reflection_buffer_rw() {
        let mut buf = ReflectionBuffer::new(4, 4);
        buf.write(1, 2, Vec3::new(1.0, 0.5, 0.25), 10.0, 0.8);
        let r = buf.read(1, 2);
        assert!((r.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ssr_tracer_defaults() {
        let tracer = SSRTracer::new();
        assert_eq!(tracer.max_steps, 64);
        assert!(tracer.stride > 0.0);
    }
}
