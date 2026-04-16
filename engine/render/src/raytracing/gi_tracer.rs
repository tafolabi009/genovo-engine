// engine/render/src/raytracing/gi_tracer.rs
//
// Global illumination via ray tracing for the Genovo engine.
// Supports diffuse GI via screen probes, multi-bounce recursive tracing,
// and a full unbiased path tracer for baking.

use crate::gi::SphericalHarmonics;
use crate::raytracing::bvh_tracer::{BVHAccelerationStructure, HitInfo, Ray};
use crate::raytracing::reflections::reflect;
use crate::raytracing::screen_probes::{
    hemisphere_sample_cosine_weighted, hemisphere_sample_uniform, ScreenProbeGrid,
    SurfaceCache,
};
use glam::{Vec2, Vec3, Vec4};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// GIMode
// ---------------------------------------------------------------------------

/// Global illumination mode selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GIMode {
    /// Screen-space probes only (fastest, Lumen Low).
    ScreenProbes,
    /// Screen probes + multi-bounce (Lumen Medium/High).
    MultiBounce,
    /// Full path tracing (offline / baking quality).
    PathTrace,
    /// Disabled (no GI).
    Off,
}

// ---------------------------------------------------------------------------
// GISettings
// ---------------------------------------------------------------------------

/// Configuration for the GI tracer.
#[derive(Debug, Clone)]
pub struct GISettings {
    /// GI mode.
    pub mode: GIMode,
    /// Number of bounces for multi-bounce mode.
    pub bounce_count: u32,
    /// Maximum trace distance.
    pub max_trace_distance: f32,
    /// Number of samples per pixel for path tracing mode.
    pub path_trace_samples: u32,
    /// Russian roulette start bounce (path tracing).
    pub russian_roulette_start: u32,
    /// Minimum Russian roulette survival probability.
    pub russian_roulette_min_prob: f32,
    /// GI intensity multiplier.
    pub intensity: f32,
    /// Whether to clamp fireflies.
    pub firefly_clamp: f32,
    /// Indirect lighting only (no direct, for debugging).
    pub indirect_only: bool,
}

impl Default for GISettings {
    fn default() -> Self {
        Self {
            mode: GIMode::ScreenProbes,
            bounce_count: 2,
            max_trace_distance: 200.0,
            path_trace_samples: 64,
            russian_roulette_start: 3,
            russian_roulette_min_prob: 0.05,
            intensity: 1.0,
            firefly_clamp: 10.0,
            indirect_only: false,
        }
    }
}

impl GISettings {
    /// Low quality: screen probes only, single bounce.
    pub fn low() -> Self {
        Self {
            mode: GIMode::ScreenProbes,
            bounce_count: 1,
            path_trace_samples: 16,
            ..Default::default()
        }
    }

    /// Medium quality: multi-bounce.
    pub fn medium() -> Self {
        Self {
            mode: GIMode::MultiBounce,
            bounce_count: 2,
            path_trace_samples: 64,
            ..Default::default()
        }
    }

    /// High quality: multi-bounce with more bounces.
    pub fn high() -> Self {
        Self {
            mode: GIMode::MultiBounce,
            bounce_count: 3,
            path_trace_samples: 256,
            ..Default::default()
        }
    }

    /// Epic quality: full path tracing.
    pub fn epic() -> Self {
        Self {
            mode: GIMode::PathTrace,
            bounce_count: 8,
            path_trace_samples: 512,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// GITracer
// ---------------------------------------------------------------------------

/// The main GI tracer orchestrating different GI approaches.
pub struct GITracer {
    pub settings: GISettings,
    /// Accumulated path trace results (for progressive rendering).
    pub path_trace_accumulator: Vec<Vec3>,
    /// Number of accumulated samples per pixel.
    pub path_trace_sample_count: Vec<u32>,
    /// Frame dimensions.
    pub width: u32,
    pub height: u32,
    /// Frame counter for jittering.
    pub frame_index: u32,
}

impl GITracer {
    /// Create a new GI tracer.
    pub fn new(width: u32, height: u32, settings: GISettings) -> Self {
        let size = (width * height) as usize;
        Self {
            settings,
            path_trace_accumulator: vec![Vec3::ZERO; size],
            path_trace_sample_count: vec![0; size],
            width,
            height,
            frame_index: 0,
        }
    }

    /// Resize the tracer.
    pub fn resize(&mut self, width: u32, height: u32) {
        let size = (width * height) as usize;
        self.width = width;
        self.height = height;
        self.path_trace_accumulator = vec![Vec3::ZERO; size];
        self.path_trace_sample_count = vec![0; size];
    }

    /// Reset path trace accumulation (e.g., when camera moves).
    pub fn reset_accumulation(&mut self) {
        for v in &mut self.path_trace_accumulator {
            *v = Vec3::ZERO;
        }
        for c in &mut self.path_trace_sample_count {
            *c = 0;
        }
    }

    /// Trace diffuse GI for a single hit point using multi-bounce.
    ///
    /// Returns the indirect radiance arriving at the hit point.
    pub fn trace_diffuse_gi(
        &self,
        position: Vec3,
        normal: Vec3,
        accel: &BVHAccelerationStructure,
        surface_cache: &SurfaceCache,
        sky_radiance: &dyn Fn(Vec3) -> Vec3,
        bounce: u32,
        seed: u32,
    ) -> Vec3 {
        if bounce >= self.settings.bounce_count {
            return Vec3::ZERO;
        }

        let max_dist = self.settings.max_trace_distance;
        let samples = if bounce == 0 { 8u32 } else { 4 };

        let mut total_radiance = Vec3::ZERO;

        for i in 0..samples {
            let direction = hemisphere_sample_cosine_weighted(
                normal,
                i,
                samples,
                seed.wrapping_add(bounce * 1000 + i),
            );

            let ray = Ray::with_range(
                position + normal * 0.01,
                direction,
                1e-3,
                max_dist,
            );

            match accel.trace_ray(&ray) {
                Some(hit) => {
                    // Get direct lighting from surface cache.
                    let direct = surface_cache.lookup_radiance(&hit);

                    // Recursively compute indirect at the hit point.
                    let indirect = if bounce + 1 < self.settings.bounce_count {
                        self.trace_diffuse_gi(
                            hit.position,
                            hit.normal,
                            accel,
                            surface_cache,
                            sky_radiance,
                            bounce + 1,
                            seed.wrapping_add(i * 7919),
                        )
                    } else {
                        Vec3::ZERO
                    };

                    total_radiance += direct + indirect;
                }
                None => {
                    total_radiance += sky_radiance(direction);
                }
            }
        }

        // Monte Carlo normalization (cosine-weighted hemisphere).
        // The PDF for cosine-weighted sampling is cos(theta) / pi,
        // so the estimator is: (1/N) * sum(L * cos / pdf) = (1/N) * sum(L * pi)
        // but we already weight by cos in the sampling, so just divide by N.
        let radiance = total_radiance / samples as f32;

        // Clamp fireflies.
        clamp_radiance(radiance, self.settings.firefly_clamp)
    }

    /// Full path tracing for a single pixel. Traces a complete light path
    /// with Russian roulette termination.
    ///
    /// This is the reference-quality path tracer used for baking or
    /// high-quality offline rendering.
    pub fn path_trace_pixel(
        &self,
        origin: Vec3,
        direction: Vec3,
        accel: &BVHAccelerationStructure,
        material_fn: &dyn Fn(&HitInfo) -> MaterialResponse,
        sky_radiance: &dyn Fn(Vec3) -> Vec3,
        light_sample_fn: &dyn Fn(Vec3) -> (Vec3, Vec3, f32),
        seed: u32,
    ) -> Vec3 {
        let mut throughput = Vec3::ONE;
        let mut radiance = Vec3::ZERO;
        let mut current_origin = origin;
        let mut current_dir = direction;

        for bounce in 0..self.settings.bounce_count.max(1) {
            let ray = Ray::with_range(
                current_origin,
                current_dir,
                1e-3,
                self.settings.max_trace_distance,
            );

            let hit = match accel.trace_ray(&ray) {
                Some(h) => h,
                None => {
                    // Ray escaped -- add sky contribution.
                    radiance += throughput * sky_radiance(current_dir);
                    break;
                }
            };

            // Evaluate material at the hit point.
            let material = material_fn(&hit);

            // Add emissive contribution.
            radiance += throughput * material.emission;

            // Direct lighting (next event estimation / shadow ray).
            let (light_dir, light_radiance, light_pdf) =
                light_sample_fn(hit.position);
            if light_pdf > 0.0 && light_dir.dot(hit.normal) > 0.0 {
                // Shadow test.
                let shadow_ray = Ray::with_range(
                    hit.position + hit.normal * 0.01,
                    light_dir,
                    1e-3,
                    self.settings.max_trace_distance,
                );
                if !accel.any_hit(&shadow_ray) {
                    let n_dot_l = hit.normal.dot(light_dir).max(0.0);
                    let brdf = material.albedo * (1.0 / PI); // Lambertian.
                    radiance += throughput * brdf * light_radiance * n_dot_l / light_pdf;
                }
            }

            // Russian roulette termination.
            if bounce >= self.settings.russian_roulette_start {
                let luminance = throughput.dot(Vec3::new(0.2126, 0.7152, 0.0722));
                let survive_prob = luminance.clamp(
                    self.settings.russian_roulette_min_prob,
                    0.95,
                );
                let rng = hash_float(seed.wrapping_add(bounce * 17 + 31));
                if rng >= survive_prob {
                    break;
                }
                throughput /= survive_prob;
            }

            // Sample next direction (cosine-weighted for Lambertian).
            let next_dir = hemisphere_sample_cosine_weighted(
                hit.normal,
                bounce,
                self.settings.bounce_count,
                seed.wrapping_add(bounce * 7919),
            );

            // Update throughput.
            // For Lambertian: BRDF = albedo / pi.
            // PDF = cos(theta) / pi.
            // throughput *= BRDF * cos / PDF = albedo.
            throughput *= material.albedo;

            // Advance ray.
            current_origin = hit.position + hit.normal * 0.01;
            current_dir = next_dir;

            // Throughput too low -- terminate.
            if throughput.length_squared() < 1e-8 {
                break;
            }
        }

        clamp_radiance(radiance, self.settings.firefly_clamp)
    }

    /// Run progressive path tracing: add one sample per pixel.
    ///
    /// Call this repeatedly for progressive refinement.
    pub fn path_trace_frame(
        &mut self,
        accel: &BVHAccelerationStructure,
        camera_origin: Vec3,
        pixel_ray_fn: &dyn Fn(u32, u32, u32) -> Vec3,
        material_fn: &dyn Fn(&HitInfo) -> MaterialResponse,
        sky_radiance: &dyn Fn(Vec3) -> Vec3,
        light_sample_fn: &dyn Fn(Vec3) -> (Vec3, Vec3, f32),
    ) {
        for py in 0..self.height {
            for px in 0..self.width {
                let idx = (py * self.width + px) as usize;
                let direction = pixel_ray_fn(px, py, self.frame_index);

                let sample = self.path_trace_pixel(
                    camera_origin,
                    direction,
                    accel,
                    material_fn,
                    sky_radiance,
                    light_sample_fn,
                    self.frame_index
                        .wrapping_mul(0x9E37_79B9)
                        .wrapping_add(px)
                        .wrapping_add(py.wrapping_mul(self.width)),
                );

                self.path_trace_accumulator[idx] += sample;
                self.path_trace_sample_count[idx] += 1;
            }
        }

        self.frame_index = self.frame_index.wrapping_add(1);
    }

    /// Get the current accumulated path trace result for a pixel.
    pub fn get_path_trace_result(&self, px: u32, py: u32) -> Vec3 {
        let idx = (py * self.width + px) as usize;
        let count = self.path_trace_sample_count[idx];
        if count == 0 {
            Vec3::ZERO
        } else {
            self.path_trace_accumulator[idx] / count as f32
        }
    }

    /// Get the total accumulated sample count for a pixel.
    pub fn get_sample_count(&self, px: u32, py: u32) -> u32 {
        let idx = (py * self.width + px) as usize;
        self.path_trace_sample_count[idx]
    }

    /// Get the average sample count across all pixels.
    pub fn average_sample_count(&self) -> f32 {
        let total: u64 = self
            .path_trace_sample_count
            .iter()
            .map(|&c| c as u64)
            .sum();
        let pixels = (self.width * self.height) as f64;
        if pixels < 1.0 {
            0.0
        } else {
            (total as f64 / pixels) as f32
        }
    }

    /// Check if all pixels have reached the target sample count.
    pub fn is_converged(&self) -> bool {
        let target = self.settings.path_trace_samples;
        self.path_trace_sample_count.iter().all(|&c| c >= target)
    }

    /// Advance the frame counter.
    pub fn next_frame(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }
}

// ---------------------------------------------------------------------------
// MaterialResponse
// ---------------------------------------------------------------------------

/// Material information needed by the path tracer at a hit point.
#[derive(Debug, Clone, Copy)]
pub struct MaterialResponse {
    /// Diffuse albedo (reflectance).
    pub albedo: Vec3,
    /// Emissive radiance (self-illumination).
    pub emission: Vec3,
    /// Roughness (0 = mirror, 1 = diffuse).
    pub roughness: f32,
    /// Metallic factor (0 = dielectric, 1 = metal).
    pub metallic: f32,
    /// Index of refraction (for dielectrics).
    pub ior: f32,
}

impl Default for MaterialResponse {
    fn default() -> Self {
        Self {
            albedo: Vec3::splat(0.8),
            emission: Vec3::ZERO,
            roughness: 0.5,
            metallic: 0.0,
            ior: 1.5,
        }
    }
}

impl MaterialResponse {
    /// Create a simple Lambertian material.
    pub fn lambertian(albedo: Vec3) -> Self {
        Self {
            albedo,
            roughness: 1.0,
            ..Default::default()
        }
    }

    /// Create an emissive material.
    pub fn emissive(emission: Vec3) -> Self {
        Self {
            albedo: Vec3::ZERO,
            emission,
            ..Default::default()
        }
    }

    /// Create a metal material.
    pub fn metal(albedo: Vec3, roughness: f32) -> Self {
        Self {
            albedo,
            roughness,
            metallic: 1.0,
            ..Default::default()
        }
    }

    /// Fresnel-Schlick approximation for the specular reflection coefficient.
    pub fn fresnel_schlick(&self, cos_theta: f32) -> Vec3 {
        let f0 = if self.metallic > 0.5 {
            self.albedo
        } else {
            let r = (self.ior - 1.0) / (self.ior + 1.0);
            Vec3::splat(r * r)
        };
        f0 + (Vec3::ONE - f0) * (1.0 - cos_theta).max(0.0).powi(5)
    }

    /// Evaluate the BRDF for a given pair of directions.
    ///
    /// Simplified Cook-Torrance BRDF with Lambertian diffuse.
    pub fn evaluate_brdf(
        &self,
        normal: Vec3,
        view_dir: Vec3,
        light_dir: Vec3,
    ) -> Vec3 {
        let n_dot_l = normal.dot(light_dir).max(0.0);
        let n_dot_v = normal.dot(view_dir).max(0.0);

        if n_dot_l < 1e-6 || n_dot_v < 1e-6 {
            return Vec3::ZERO;
        }

        // Diffuse component (Lambertian).
        let k_d = (1.0 - self.metallic).max(0.0);
        let diffuse = self.albedo * (k_d / PI);

        // Specular component (simplified GGX).
        let h = (view_dir + light_dir).normalize_or_zero();
        let n_dot_h = normal.dot(h).max(0.0);
        let v_dot_h = view_dir.dot(h).max(0.0);

        let alpha = self.roughness * self.roughness;
        let alpha2 = alpha * alpha;

        // GGX distribution.
        let denom = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
        let d = alpha2 / (PI * denom * denom).max(1e-7);

        // Fresnel.
        let f = self.fresnel_schlick(v_dot_h);

        // Schlick-GGX geometry term.
        let k = (self.roughness + 1.0).powi(2) / 8.0;
        let g1_v = n_dot_v / (n_dot_v * (1.0 - k) + k).max(1e-7);
        let g1_l = n_dot_l / (n_dot_l * (1.0 - k) + k).max(1e-7);
        let g = g1_v * g1_l;

        let specular = f * d * g / (4.0 * n_dot_v * n_dot_l).max(1e-7);

        diffuse + specular
    }
}

// ---------------------------------------------------------------------------
// MIS (Multiple Importance Sampling) utilities
// ---------------------------------------------------------------------------

/// Power heuristic for MIS combining (exponent 2).
#[inline]
pub fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    a2 / (a2 + b2).max(1e-10)
}

/// Balance heuristic for MIS combining.
#[inline]
pub fn balance_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    pdf_a / (pdf_a + pdf_b).max(1e-10)
}

// ---------------------------------------------------------------------------
// GI result buffer
// ---------------------------------------------------------------------------

/// Buffer holding per-pixel GI results.
pub struct GIResultBuffer {
    pub width: u32,
    pub height: u32,
    /// Indirect diffuse radiance per pixel.
    pub diffuse: Vec<Vec3>,
    /// Previous frame diffuse (for temporal).
    pub prev_diffuse: Vec<Vec3>,
    /// Confidence / sample weight.
    pub confidence: Vec<f32>,
}

impl GIResultBuffer {
    /// Create a new GI result buffer.
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            width,
            height,
            diffuse: vec![Vec3::ZERO; size],
            prev_diffuse: vec![Vec3::ZERO; size],
            confidence: vec![0.0; size],
        }
    }

    /// Swap history for temporal accumulation.
    pub fn swap_history(&mut self) {
        std::mem::swap(&mut self.diffuse, &mut self.prev_diffuse);
        for v in &mut self.diffuse {
            *v = Vec3::ZERO;
        }
        for c in &mut self.confidence {
            *c = 0.0;
        }
    }

    /// Write a GI result.
    pub fn write(&mut self, x: u32, y: u32, radiance: Vec3, confidence: f32) {
        let idx = (y * self.width + x) as usize;
        self.diffuse[idx] = radiance;
        self.confidence[idx] = confidence;
    }

    /// Read the GI result.
    pub fn read(&self, x: u32, y: u32) -> Vec3 {
        self.diffuse[(y * self.width + x) as usize]
    }

    /// Apply temporal accumulation.
    pub fn temporal_accumulate(&mut self, blend: f32) {
        let size = (self.width * self.height) as usize;
        for i in 0..size {
            self.diffuse[i] =
                self.diffuse[i] * (1.0 - blend) + self.prev_diffuse[i] * blend;
        }
    }

    /// Resize the buffer.
    pub fn resize(&mut self, width: u32, height: u32) {
        *self = Self::new(width, height);
    }
}

// ---------------------------------------------------------------------------
// Irradiance field for world-space GI caching
// ---------------------------------------------------------------------------

/// World-space irradiance field using a uniform 3D grid of probes.
/// Each probe stores SH coefficients for indirect lighting.
/// This supplements screen probes with off-screen GI.
pub struct IrradianceField {
    /// Grid dimensions.
    pub resolution: [u32; 3],
    /// World-space origin (min corner).
    pub origin: Vec3,
    /// Cell size.
    pub cell_size: Vec3,
    /// SH per probe.
    pub probes: Vec<SphericalHarmonics>,
    /// Sample count per probe (for progressive accumulation).
    pub sample_counts: Vec<u32>,
    /// Whether each probe needs updating.
    pub dirty: Vec<bool>,
}

impl IrradianceField {
    /// Create a new irradiance field.
    pub fn new(origin: Vec3, extent: Vec3, resolution: [u32; 3]) -> Self {
        let rx = resolution[0].max(2);
        let ry = resolution[1].max(2);
        let rz = resolution[2].max(2);
        let cell_size = Vec3::new(
            extent.x / (rx - 1) as f32,
            extent.y / (ry - 1) as f32,
            extent.z / (rz - 1) as f32,
        );

        let total = (rx * ry * rz) as usize;

        Self {
            resolution: [rx, ry, rz],
            origin,
            cell_size,
            probes: vec![SphericalHarmonics::new(); total],
            sample_counts: vec![0; total],
            dirty: vec![true; total],
        }
    }

    /// Linear index from grid coordinates.
    #[inline]
    pub fn linear_index(&self, x: u32, y: u32, z: u32) -> usize {
        (z * self.resolution[1] * self.resolution[0]
            + y * self.resolution[0]
            + x) as usize
    }

    /// World-space position of a probe.
    pub fn probe_position(&self, x: u32, y: u32, z: u32) -> Vec3 {
        self.origin
            + Vec3::new(
                x as f32 * self.cell_size.x,
                y as f32 * self.cell_size.y,
                z as f32 * self.cell_size.z,
            )
    }

    /// Total probe count.
    pub fn probe_count(&self) -> usize {
        self.probes.len()
    }

    /// Update a single probe by tracing rays.
    pub fn update_probe(
        &mut self,
        x: u32,
        y: u32,
        z: u32,
        accel: &BVHAccelerationStructure,
        surface_cache: &SurfaceCache,
        sky_radiance: &dyn Fn(Vec3) -> Vec3,
        rays: u32,
        seed: u32,
    ) {
        let idx = self.linear_index(x, y, z);
        let pos = self.probe_position(x, y, z);

        let mut sh = SphericalHarmonics::new();

        for i in 0..rays {
            // Uniform sphere sampling for omnidirectional probes.
            let direction = sphere_sample_uniform(i, rays, seed);

            let ray = Ray::with_range(pos, direction, 1e-3, 200.0);

            let radiance = match accel.trace_ray(&ray) {
                Some(hit) => surface_cache.lookup_radiance(&hit),
                None => sky_radiance(direction),
            };

            // Weight = 4*pi / N (uniform sphere measure).
            let weight = 4.0 * PI / rays as f32;
            sh.encode_weighted(direction, radiance, weight);
        }

        // Progressive accumulation.
        let count = self.sample_counts[idx];
        if count > 0 {
            let old_weight = count as f32 / (count + rays) as f32;
            let new_weight = rays as f32 / (count + rays) as f32;
            for ch in 0..3 {
                for c in 0..9 {
                    self.probes[idx].coeffs[ch][c] = self.probes[idx].coeffs[ch][c] * old_weight
                        + sh.coeffs[ch][c] * new_weight;
                }
            }
        } else {
            self.probes[idx] = sh;
        }

        self.sample_counts[idx] += rays;
        self.dirty[idx] = false;
    }

    /// Sample the irradiance field at a world position using trilinear interpolation.
    pub fn sample(&self, world_pos: Vec3, normal: Vec3) -> Vec3 {
        let local = world_pos - self.origin;
        let gx = (local.x / self.cell_size.x).clamp(0.0, (self.resolution[0] - 1) as f32);
        let gy = (local.y / self.cell_size.y).clamp(0.0, (self.resolution[1] - 1) as f32);
        let gz = (local.z / self.cell_size.z).clamp(0.0, (self.resolution[2] - 1) as f32);

        let x0 = gx.floor() as u32;
        let y0 = gy.floor() as u32;
        let z0 = gz.floor() as u32;
        let x1 = (x0 + 1).min(self.resolution[0] - 1);
        let y1 = (y0 + 1).min(self.resolution[1] - 1);
        let z1 = (z0 + 1).min(self.resolution[2] - 1);

        let fx = gx - gx.floor();
        let fy = gy - gy.floor();
        let fz = gz - gz.floor();

        // Trilinear interpolation of SH, then evaluate.
        let mut result_sh = SphericalHarmonics::new();
        let corners = [
            (x0, y0, z0, (1.0 - fx) * (1.0 - fy) * (1.0 - fz)),
            (x1, y0, z0, fx * (1.0 - fy) * (1.0 - fz)),
            (x0, y1, z0, (1.0 - fx) * fy * (1.0 - fz)),
            (x1, y1, z0, fx * fy * (1.0 - fz)),
            (x0, y0, z1, (1.0 - fx) * (1.0 - fy) * fz),
            (x1, y0, z1, fx * (1.0 - fy) * fz),
            (x0, y1, z1, (1.0 - fx) * fy * fz),
            (x1, y1, z1, fx * fy * fz),
        ];

        for &(cx, cy, cz, w) in &corners {
            let idx = self.linear_index(cx, cy, cz);
            let sh = &self.probes[idx];
            for ch in 0..3 {
                for c in 0..9 {
                    result_sh.coeffs[ch][c] += sh.coeffs[ch][c] * w;
                }
            }
        }

        result_sh.evaluate_irradiance(normal)
    }

    /// Mark all probes as dirty (needing update).
    pub fn invalidate_all(&mut self) {
        for d in &mut self.dirty {
            *d = true;
        }
    }

    /// Count dirty probes.
    pub fn dirty_count(&self) -> usize {
        self.dirty.iter().filter(|&&d| d).count()
    }
}

/// Uniform sphere sampling (full sphere, not hemisphere).
fn sphere_sample_uniform(index: u32, total: u32, seed: u32) -> Vec3 {
    let xi1 = (index as f32 + hash_float(seed)) / total.max(1) as f32;
    let xi2 = radical_inverse_vdc(index ^ seed);

    let phi = 2.0 * PI * xi1;
    let cos_theta = 1.0 - 2.0 * xi2;
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

    Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta)
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Clamp radiance to prevent fireflies.
#[inline]
fn clamp_radiance(r: Vec3, max_value: f32) -> Vec3 {
    let lum = r.x * 0.2126 + r.y * 0.7152 + r.z * 0.0722;
    if lum > max_value && lum > 0.0 {
        r * (max_value / lum)
    } else {
        Vec3::new(r.x.max(0.0), r.y.max(0.0), r.z.max(0.0))
    }
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gi_settings_presets() {
        let low = GISettings::low();
        let high = GISettings::high();
        assert!(low.bounce_count < high.bounce_count);
    }

    #[test]
    fn material_lambertian() {
        let mat = MaterialResponse::lambertian(Vec3::splat(0.5));
        assert!(mat.roughness > 0.9);
        assert!(mat.metallic < 0.1);
    }

    #[test]
    fn material_fresnel() {
        let mat = MaterialResponse::default();
        let f0 = mat.fresnel_schlick(1.0); // Normal incidence.
        let f90 = mat.fresnel_schlick(0.0); // Grazing angle.
        // At grazing angle, Fresnel should approach 1.
        assert!(f90.x > f0.x);
    }

    #[test]
    fn material_brdf_non_negative() {
        let mat = MaterialResponse::default();
        let brdf = mat.evaluate_brdf(Vec3::Y, Vec3::Y, Vec3::Y);
        assert!(brdf.x >= 0.0 && brdf.y >= 0.0 && brdf.z >= 0.0);
    }

    #[test]
    fn power_heuristic_symmetric() {
        let w = power_heuristic(1.0, 1.0);
        assert!((w - 0.5).abs() < 1e-6);
    }

    #[test]
    fn gi_result_buffer() {
        let mut buf = GIResultBuffer::new(4, 4);
        buf.write(1, 2, Vec3::ONE, 1.0);
        assert!((buf.read(1, 2).x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn irradiance_field_creation() {
        let field = IrradianceField::new(
            Vec3::ZERO,
            Vec3::splat(10.0),
            [4, 4, 4],
        );
        assert_eq!(field.probe_count(), 64);
        assert_eq!(field.dirty_count(), 64);
    }

    #[test]
    fn sphere_sampling_coverage() {
        let mut positive_z = 0;
        let mut negative_z = 0;
        for i in 0..100 {
            let dir = sphere_sample_uniform(i, 100, 42);
            assert!((dir.length() - 1.0).abs() < 0.1);
            if dir.z > 0.0 {
                positive_z += 1;
            } else {
                negative_z += 1;
            }
        }
        // Should cover both hemispheres.
        assert!(positive_z > 20);
        assert!(negative_z > 20);
    }

    #[test]
    fn firefly_clamp() {
        let r = clamp_radiance(Vec3::splat(100.0), 10.0);
        let lum = r.x * 0.2126 + r.y * 0.7152 + r.z * 0.0722;
        assert!(lum <= 10.01);
    }

    #[test]
    fn gi_tracer_creation() {
        let tracer = GITracer::new(32, 32, GISettings::default());
        assert_eq!(tracer.width, 32);
        assert!(!tracer.is_converged());
    }
}
