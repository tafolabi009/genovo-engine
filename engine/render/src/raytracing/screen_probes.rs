// engine/render/src/raytracing/screen_probes.rs
//
// Lumen-style screen-space probe system for the Genovo engine.
// Places probes on an 8x8 pixel grid, traces hemisphere rays from each
// probe, accumulates radiance into SH coefficients, and provides spatial
// and temporal filtering for smooth indirect lighting.

use crate::gi::{SphericalHarmonics, SH_COEFF_COUNT};
use crate::raytracing::bvh_tracer::{
    BVHAccelerationStructure, HitInfo, Ray, AABB,
};
use glam::{Vec2, Vec3, Vec4};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default probe tile size in pixels (each probe covers one tile).
pub const PROBE_TILE_SIZE: u32 = 8;

/// Default number of hemisphere rays per probe per frame.
pub const DEFAULT_RAYS_PER_PROBE: u32 = 32;

/// Maximum number of rays per probe.
pub const MAX_RAYS_PER_PROBE: u32 = 128;

/// Temporal blend factor (how much of the previous frame to keep).
pub const DEFAULT_TEMPORAL_BLEND: f32 = 0.9;

/// Spatial filter kernel radius (in probe tiles).
pub const SPATIAL_FILTER_RADIUS: u32 = 1;

// ---------------------------------------------------------------------------
// ScreenProbe
// ---------------------------------------------------------------------------

/// A single screen-space probe placed at a depth-buffer-derived position.
#[derive(Debug, Clone)]
pub struct ScreenProbe {
    /// Tile coordinates (probe grid position).
    pub tile_x: u32,
    pub tile_y: u32,
    /// World-space position derived from the depth buffer center pixel.
    pub position: Vec3,
    /// World-space surface normal at the probe position.
    pub normal: Vec3,
    /// Screen-space depth value at the probe center pixel.
    pub depth: f32,
    /// Accumulated SH radiance for this probe.
    pub radiance_sh: SphericalHarmonics,
    /// SH from the previous frame (for temporal accumulation).
    pub prev_radiance_sh: SphericalHarmonics,
    /// Number of valid samples accumulated.
    pub sample_count: u32,
    /// Whether this probe is valid (has geometry behind it).
    pub valid: bool,
    /// Linear index into the probe grid.
    pub linear_index: u32,
}

impl ScreenProbe {
    /// Create a new empty screen probe.
    pub fn new(tile_x: u32, tile_y: u32) -> Self {
        Self {
            tile_x,
            tile_y,
            position: Vec3::ZERO,
            normal: Vec3::Y,
            depth: 0.0,
            radiance_sh: SphericalHarmonics::new(),
            prev_radiance_sh: SphericalHarmonics::new(),
            sample_count: 0,
            valid: false,
            linear_index: 0,
        }
    }

    /// Reset the probe for a new frame (save previous SH, clear accumulator).
    pub fn begin_frame(&mut self) {
        self.prev_radiance_sh = self.radiance_sh.clone();
        self.radiance_sh = SphericalHarmonics::new();
        self.sample_count = 0;
    }

    /// Accumulate a single radiance sample from a hemisphere ray.
    ///
    /// `direction` is the world-space ray direction (in the hemisphere
    /// above the probe normal). `radiance` is the incoming radiance from
    /// that direction.
    pub fn accumulate_sample(&mut self, direction: Vec3, radiance: Vec3) {
        let weight = direction.dot(self.normal).max(0.0);
        if weight < 1e-6 {
            return;
        }

        // Project onto SH with cosine weighting.
        self.radiance_sh
            .encode_weighted(direction, radiance, weight);
        self.sample_count += 1;
    }

    /// Finalize the probe: normalize SH by sample count and apply
    /// temporal accumulation with the previous frame.
    pub fn finalize(&mut self, temporal_blend: f32) {
        if self.sample_count > 0 {
            // Monte Carlo normalization: hemisphere integral with cosine
            // weighting is pi, divided by the number of samples.
            let normalization = PI / self.sample_count as f32;
            self.radiance_sh.scale(normalization);
        }

        if self.valid {
            // Temporal blend: lerp between previous and current.
            self.radiance_sh = SphericalHarmonics::lerp(
                &self.radiance_sh,
                &self.prev_radiance_sh,
                temporal_blend,
            );
        }
    }

    /// Evaluate the irradiance at this probe for a given normal direction.
    pub fn evaluate_irradiance(&self, normal: Vec3) -> Vec3 {
        self.radiance_sh.evaluate_irradiance(normal)
    }

    /// Evaluate the raw radiance in a direction.
    pub fn evaluate(&self, direction: Vec3) -> Vec3 {
        self.radiance_sh.evaluate(direction)
    }
}

// ---------------------------------------------------------------------------
// ScreenProbeGrid
// ---------------------------------------------------------------------------

/// Grid of screen-space probes covering the render target.
pub struct ScreenProbeGrid {
    /// Width of the render target in pixels.
    pub screen_width: u32,
    /// Height of the render target in pixels.
    pub screen_height: u32,
    /// Probe tile size (probes are spaced every tile_size pixels).
    pub tile_size: u32,
    /// Number of probe columns (tiles in X).
    pub grid_width: u32,
    /// Number of probe rows (tiles in Y).
    pub grid_height: u32,
    /// All probes, stored in row-major order.
    pub probes: Vec<ScreenProbe>,
    /// Configuration: rays per probe per frame.
    pub rays_per_probe: u32,
    /// Configuration: temporal blend factor.
    pub temporal_blend: f32,
    /// Configuration: maximum trace distance.
    pub max_trace_distance: f32,
    /// Frame index for jittering / rotation.
    pub frame_index: u32,
}

impl ScreenProbeGrid {
    /// Create a new screen probe grid for the given render target size.
    pub fn new(screen_width: u32, screen_height: u32) -> Self {
        Self::with_tile_size(screen_width, screen_height, PROBE_TILE_SIZE)
    }

    /// Create a grid with a custom tile size.
    pub fn with_tile_size(screen_width: u32, screen_height: u32, tile_size: u32) -> Self {
        let tile_size = tile_size.max(1);
        let grid_width = (screen_width + tile_size - 1) / tile_size;
        let grid_height = (screen_height + tile_size - 1) / tile_size;
        let probe_count = (grid_width * grid_height) as usize;

        let mut probes = Vec::with_capacity(probe_count);
        for ty in 0..grid_height {
            for tx in 0..grid_width {
                let mut probe = ScreenProbe::new(tx, ty);
                probe.linear_index = ty * grid_width + tx;
                probes.push(probe);
            }
        }

        Self {
            screen_width,
            screen_height,
            tile_size,
            grid_width,
            grid_height,
            probes,
            rays_per_probe: DEFAULT_RAYS_PER_PROBE,
            temporal_blend: DEFAULT_TEMPORAL_BLEND,
            max_trace_distance: 200.0,
            frame_index: 0,
        }
    }

    /// Total number of probes.
    pub fn probe_count(&self) -> usize {
        self.probes.len()
    }

    /// Get a probe by tile coordinates.
    pub fn get_probe(&self, tx: u32, ty: u32) -> &ScreenProbe {
        let idx = (ty * self.grid_width + tx) as usize;
        &self.probes[idx]
    }

    /// Get a mutable probe by tile coordinates.
    pub fn get_probe_mut(&mut self, tx: u32, ty: u32) -> &mut ScreenProbe {
        let idx = (ty * self.grid_width + tx) as usize;
        &mut self.probes[idx]
    }

    /// Get the probe that covers a given pixel coordinate.
    pub fn probe_at_pixel(&self, px: u32, py: u32) -> &ScreenProbe {
        let tx = (px / self.tile_size).min(self.grid_width - 1);
        let ty = (py / self.tile_size).min(self.grid_height - 1);
        self.get_probe(tx, ty)
    }

    /// Resize the grid if the screen size has changed.
    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        if new_width == self.screen_width && new_height == self.screen_height {
            return;
        }
        *self = Self::with_tile_size(new_width, new_height, self.tile_size);
    }

    // -----------------------------------------------------------------------
    // Per-frame pipeline
    // -----------------------------------------------------------------------

    /// Place probes from the depth and normal buffers.
    ///
    /// `depth_buffer`: linearized depth values in row-major order (width * height).
    /// `normal_buffer`: world-space normals in row-major order.
    /// `unproject`: closure mapping (pixel_x, pixel_y, depth) -> world position.
    pub fn place_probes(
        &mut self,
        depth_buffer: &[f32],
        normal_buffer: &[Vec3],
        unproject: impl Fn(u32, u32, f32) -> Vec3,
    ) {
        for probe in &mut self.probes {
            probe.begin_frame();

            // Sample the center pixel of this tile.
            let center_px = probe.tile_x * self.tile_size + self.tile_size / 2;
            let center_py = probe.tile_y * self.tile_size + self.tile_size / 2;

            if center_px >= self.screen_width || center_py >= self.screen_height {
                probe.valid = false;
                continue;
            }

            let pixel_idx = (center_py * self.screen_width + center_px) as usize;

            let depth = depth_buffer.get(pixel_idx).copied().unwrap_or(0.0);
            let normal = normal_buffer.get(pixel_idx).copied().unwrap_or(Vec3::Y);

            if depth <= 0.0 || depth >= 1.0 {
                // Sky pixel or invalid depth -- mark probe invalid.
                probe.valid = false;
                probe.depth = depth;
                continue;
            }

            probe.position = unproject(center_px, center_py, depth);
            probe.normal = normal.normalize_or_zero();
            probe.depth = depth;
            probe.valid = true;
        }
    }

    /// Trace hemisphere rays from all probes and accumulate radiance.
    ///
    /// `accel`: the scene's BVH acceleration structure.
    /// `surface_cache`: closure returning radiance at a hit point.
    /// `sky_radiance`: closure returning sky radiance for a direction.
    pub fn trace_probes(
        &mut self,
        accel: &BVHAccelerationStructure,
        surface_cache: &dyn Fn(&HitInfo) -> Vec3,
        sky_radiance: &dyn Fn(Vec3) -> Vec3,
    ) {
        let frame = self.frame_index;
        let rays_per_probe = self.rays_per_probe;
        let max_dist = self.max_trace_distance;

        for probe_idx in 0..self.probes.len() {
            if !self.probes[probe_idx].valid {
                continue;
            }

            let position = self.probes[probe_idx].position;
            let normal = self.probes[probe_idx].normal;

            // Generate hemisphere directions using stratified sampling
            // with per-frame rotation for temporal variation.
            for ray_idx in 0..rays_per_probe {
                let direction = hemisphere_sample_cosine_weighted(
                    normal,
                    ray_idx,
                    rays_per_probe,
                    frame + probe_idx as u32,
                );

                let ray = Ray::with_range(
                    position + normal * 0.01, // offset to avoid self-intersection
                    direction,
                    1e-3,
                    max_dist,
                );

                let radiance = match accel.trace_ray(&ray) {
                    Some(hit) => {
                        // Read radiance from the surface cache at the hit point.
                        surface_cache(&hit)
                    }
                    None => {
                        // Ray escaped -- sample the sky.
                        sky_radiance(direction)
                    }
                };

                self.probes[probe_idx].accumulate_sample(direction, radiance);
            }
        }
    }

    /// Finalize all probes: normalize and apply temporal accumulation.
    pub fn finalize_probes(&mut self) {
        let blend = self.temporal_blend;
        for probe in &mut self.probes {
            if probe.valid {
                probe.finalize(blend);
            }
        }
    }

    /// Apply spatial filtering across neighboring probes to reduce noise.
    ///
    /// Uses a cross-bilateral filter that respects depth and normal
    /// discontinuities to avoid light leaking.
    pub fn spatial_filter(&mut self) {
        let grid_w = self.grid_width;
        let grid_h = self.grid_height;
        let radius = SPATIAL_FILTER_RADIUS as i32;

        // Work on a copy to avoid read-write conflicts.
        let original_probes = self.probes.clone();

        for ty in 0..grid_h {
            for tx in 0..grid_w {
                let center_idx = (ty * grid_w + tx) as usize;
                let center = &original_probes[center_idx];

                if !center.valid {
                    continue;
                }

                let mut filtered_sh = SphericalHarmonics::new();
                let mut total_weight = 0.0f32;

                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let nx = tx as i32 + dx;
                        let ny = ty as i32 + dy;

                        if nx < 0 || ny < 0 || nx >= grid_w as i32 || ny >= grid_h as i32 {
                            continue;
                        }

                        let neighbor_idx = (ny as u32 * grid_w + nx as u32) as usize;
                        let neighbor = &original_probes[neighbor_idx];

                        if !neighbor.valid {
                            continue;
                        }

                        // Depth similarity weight (bilateral).
                        let depth_diff = (center.depth - neighbor.depth).abs();
                        let depth_weight = (-depth_diff * 20.0).exp();
                        if depth_weight < 0.01 {
                            continue;
                        }

                        // Normal similarity weight.
                        let normal_dot = center.normal.dot(neighbor.normal).max(0.0);
                        let normal_weight = normal_dot.powi(4);
                        if normal_weight < 0.01 {
                            continue;
                        }

                        // Spatial distance weight (Gaussian).
                        let dist2 = (dx * dx + dy * dy) as f32;
                        let spatial_weight = (-dist2 / (2.0 * (radius as f32 + 0.5).powi(2))).exp();

                        let w = depth_weight * normal_weight * spatial_weight;

                        // Accumulate weighted SH.
                        for ch in 0..3 {
                            for i in 0..SH_COEFF_COUNT {
                                filtered_sh.coeffs[ch][i] +=
                                    neighbor.radiance_sh.coeffs[ch][i] * w;
                            }
                        }
                        total_weight += w;
                    }
                }

                if total_weight > 0.0 {
                    let inv_w = 1.0 / total_weight;
                    filtered_sh.scale(inv_w);
                    self.probes[center_idx].radiance_sh = filtered_sh;
                }
            }
        }
    }

    /// Interpolate probe irradiance for a pixel.
    ///
    /// Uses bilinear interpolation between the four nearest probes, weighted
    /// by depth and normal similarity to prevent light leaking.
    pub fn sample_irradiance(
        &self,
        pixel_x: u32,
        pixel_y: u32,
        pixel_depth: f32,
        pixel_normal: Vec3,
        surface_normal: Vec3,
    ) -> Vec3 {
        // Fractional probe coordinates.
        let fx = (pixel_x as f32 + 0.5) / self.tile_size as f32 - 0.5;
        let fy = (pixel_y as f32 + 0.5) / self.tile_size as f32 - 0.5;

        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let frac_x = fx - fx.floor();
        let frac_y = fy - fy.floor();

        let mut total_radiance = Vec3::ZERO;
        let mut total_weight = 0.0f32;

        for dy in 0..2 {
            for dx in 0..2 {
                let px = x0 + dx;
                let py = y0 + dy;

                if px < 0 || py < 0 || px >= self.grid_width as i32 || py >= self.grid_height as i32
                {
                    continue;
                }

                let probe = self.get_probe(px as u32, py as u32);
                if !probe.valid {
                    continue;
                }

                // Bilinear weight.
                let wx = if dx == 0 { 1.0 - frac_x } else { frac_x };
                let wy = if dy == 0 { 1.0 - frac_y } else { frac_y };
                let bilinear_w = wx * wy;

                // Depth similarity.
                let depth_diff = (pixel_depth - probe.depth).abs();
                let depth_w = (-depth_diff * 30.0).exp();

                // Normal similarity.
                let normal_w = pixel_normal.dot(probe.normal).max(0.0).powi(2);

                let w = bilinear_w * depth_w * normal_w;

                if w < 1e-6 {
                    continue;
                }

                let irradiance = probe.evaluate_irradiance(surface_normal);
                total_radiance += irradiance * w;
                total_weight += w;
            }
        }

        if total_weight > 1e-6 {
            total_radiance / total_weight
        } else {
            Vec3::ZERO
        }
    }

    /// Advance the frame counter.
    pub fn next_frame(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }

    /// Get the total number of rays that will be traced this frame.
    pub fn total_rays_per_frame(&self) -> u32 {
        let valid_count = self.probes.iter().filter(|p| p.valid).count() as u32;
        valid_count * self.rays_per_probe
    }
}

// ---------------------------------------------------------------------------
// SurfaceCache
// ---------------------------------------------------------------------------

/// A surface cache (surfel-based) storing pre-computed direct lighting at
/// world-space positions. This is the Lumen-style "surface cache" that
/// provides the radiance looked up when a screen probe ray hits geometry.
pub struct SurfaceCache {
    /// All surfels in the cache.
    pub surfels: Vec<Surfel>,
    /// Spatial acceleration: AABB of the entire cache.
    pub bounds: AABB,
    /// Maximum number of surfels.
    pub max_surfels: usize,
    /// Whether the cache needs a full rebuild.
    pub dirty: bool,
}

/// A single surfel in the surface cache.
#[derive(Debug, Clone)]
pub struct Surfel {
    /// World-space position of the surfel.
    pub position: Vec3,
    /// World-space normal.
    pub normal: Vec3,
    /// Albedo (diffuse reflectance) of the surface.
    pub albedo: Vec3,
    /// Pre-computed direct lighting (irradiance) at this surfel.
    pub direct_lighting: Vec3,
    /// Combined radiance (direct + indirect from previous frame).
    pub radiance: Vec3,
    /// Area represented by this surfel (for weighting).
    pub area: f32,
    /// Instance ID that owns this surfel.
    pub instance_id: u32,
    /// Triangle ID within the instance.
    pub triangle_id: u32,
    /// Whether this surfel is valid.
    pub valid: bool,
}

impl Surfel {
    /// Create a new surfel.
    pub fn new(position: Vec3, normal: Vec3, albedo: Vec3) -> Self {
        Self {
            position,
            normal,
            albedo,
            direct_lighting: Vec3::ZERO,
            radiance: Vec3::ZERO,
            area: 1.0,
            instance_id: 0,
            triangle_id: 0,
            valid: true,
        }
    }

    /// Compute the reflected radiance from direct lighting.
    pub fn direct_radiance(&self) -> Vec3 {
        // Lambertian BRDF: albedo / pi * irradiance.
        self.albedo * self.direct_lighting * (1.0 / PI)
    }

    /// Update the combined radiance with a new indirect contribution.
    pub fn update_radiance(&mut self, indirect: Vec3, blend: f32) {
        let direct = self.direct_radiance();
        let new_total = direct + indirect;
        self.radiance = self.radiance * blend + new_total * (1.0 - blend);
    }
}

impl SurfaceCache {
    /// Create a new empty surface cache.
    pub fn new(max_surfels: usize) -> Self {
        Self {
            surfels: Vec::with_capacity(max_surfels),
            bounds: AABB::EMPTY,
            max_surfels,
            dirty: true,
        }
    }

    /// Add a surfel to the cache. Returns the surfel index.
    pub fn add_surfel(&mut self, surfel: Surfel) -> Option<u32> {
        if self.surfels.len() >= self.max_surfels {
            return None; // Cache full.
        }
        let idx = self.surfels.len() as u32;
        self.bounds.expand_point(surfel.position);
        self.surfels.push(surfel);
        Some(idx)
    }

    /// Build surfels from mesh instance triangles.
    ///
    /// Samples one surfel per triangle (at the centroid). For dense meshes
    /// this provides good coverage.
    pub fn build_from_scene(
        &mut self,
        accel: &BVHAccelerationStructure,
        albedo_lookup: &dyn Fn(u32, u32) -> Vec3,
    ) {
        self.surfels.clear();
        self.bounds = AABB::EMPTY;

        for (inst_idx, instance) in accel.instances.iter().enumerate() {
            if !instance.visible {
                continue;
            }

            let blas = &accel.blas_list[instance.blas_index as usize];

            for (tri_idx, tri) in blas.triangles.iter().enumerate() {
                if self.surfels.len() >= self.max_surfels {
                    return;
                }

                let local_pos = tri.centroid();
                let local_normal = tri.face_normal();

                let world_pos = instance.transform.transform_point3(local_pos);
                let world_normal = instance
                    .inv_transform
                    .transpose()
                    .transform_vector3(local_normal)
                    .normalize_or_zero();

                let albedo = albedo_lookup(inst_idx as u32, tri_idx as u32);

                let mut surfel = Surfel::new(world_pos, world_normal, albedo);
                surfel.area = tri.area();
                surfel.instance_id = instance.instance_id;
                surfel.triangle_id = tri_idx as u32;

                self.add_surfel(surfel);
            }
        }

        self.dirty = false;
    }

    /// Update direct lighting on all surfels.
    ///
    /// `compute_direct`: closure computing direct irradiance at a surfel
    /// position given (position, normal).
    pub fn update_direct_lighting(
        &mut self,
        compute_direct: &dyn Fn(Vec3, Vec3) -> Vec3,
    ) {
        for surfel in &mut self.surfels {
            if surfel.valid {
                surfel.direct_lighting = compute_direct(surfel.position, surfel.normal);
            }
        }
    }

    /// Look up the cached radiance at a hit point by finding the nearest surfel.
    ///
    /// This is a brute-force lookup suitable for small caches. For production
    /// use, a spatial hash or BVH over surfels would be used.
    pub fn lookup_radiance(&self, hit: &HitInfo) -> Vec3 {
        let mut best_dist2 = f32::MAX;
        let mut best_radiance = Vec3::ZERO;

        for surfel in &self.surfels {
            if !surfel.valid {
                continue;
            }

            // Match by instance + triangle if possible.
            if surfel.instance_id == hit.instance_id
                && surfel.triangle_id == hit.triangle_id
            {
                return surfel.radiance;
            }

            let d2 = (surfel.position - hit.position).length_squared();
            if d2 < best_dist2 {
                let normal_dot = surfel.normal.dot(hit.normal);
                if normal_dot > 0.5 {
                    best_dist2 = d2;
                    best_radiance = surfel.radiance;
                }
            }
        }

        best_radiance
    }

    /// Total surfel count.
    pub fn surfel_count(&self) -> usize {
        self.surfels.len()
    }

    /// Clear all surfels.
    pub fn clear(&mut self) {
        self.surfels.clear();
        self.bounds = AABB::EMPTY;
        self.dirty = true;
    }

    /// Invalidate all surfels (mark for re-lighting).
    pub fn invalidate(&mut self) {
        for surfel in &mut self.surfels {
            surfel.direct_lighting = Vec3::ZERO;
            surfel.radiance = Vec3::ZERO;
        }
        self.dirty = true;
    }
}

// ---------------------------------------------------------------------------
// Hemisphere sampling utilities
// ---------------------------------------------------------------------------

/// Generate a cosine-weighted hemisphere sample direction.
///
/// Uses stratified sampling with Hammersley-like jittering for low
/// discrepancy across frames.
pub fn hemisphere_sample_cosine_weighted(
    normal: Vec3,
    sample_index: u32,
    total_samples: u32,
    seed: u32,
) -> Vec3 {
    // Stratified Hammersley sequence with per-frame rotation.
    let xi1 = ((sample_index as f32 + hash_float(seed)) % 1.0f32.max(0.001))
        / total_samples.max(1) as f32;
    let xi2 = radical_inverse_vdc(sample_index ^ seed);

    // Cosine-weighted hemisphere sampling.
    let phi = 2.0 * PI * xi1;
    let cos_theta = (1.0 - xi2).sqrt();
    let sin_theta = xi2.sqrt();

    let local_dir = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);

    // Build tangent frame from normal.
    tangent_space_to_world(local_dir, normal)
}

/// Generate a uniform hemisphere sample direction.
pub fn hemisphere_sample_uniform(
    normal: Vec3,
    sample_index: u32,
    total_samples: u32,
    seed: u32,
) -> Vec3 {
    let xi1 = ((sample_index as f32 + hash_float(seed)) % 1.0f32.max(0.001))
        / total_samples.max(1) as f32;
    let xi2 = radical_inverse_vdc(sample_index ^ seed);

    let phi = 2.0 * PI * xi1;
    let cos_theta = xi2;
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

    let local_dir = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);
    tangent_space_to_world(local_dir, normal)
}

/// Transform a tangent-space direction to world space given a normal.
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

/// Van der Corput radical inverse (bit reversal).
fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    bits as f32 * 2.328_306_4e-10
}

/// Simple hash function to generate a float in [0, 1) from a u32 seed.
fn hash_float(mut seed: u32) -> f32 {
    seed ^= seed >> 16;
    seed = seed.wrapping_mul(0x45d9f3b);
    seed ^= seed >> 16;
    seed = seed.wrapping_mul(0x45d9f3b);
    seed ^= seed >> 16;
    (seed & 0x00FF_FFFF) as f32 / (0x0100_0000 as f32)
}

// ---------------------------------------------------------------------------
// ScreenProbeSettings
// ---------------------------------------------------------------------------

/// Configuration for the screen probe system.
#[derive(Debug, Clone)]
pub struct ScreenProbeSettings {
    /// Tile size in pixels.
    pub tile_size: u32,
    /// Number of rays per probe per frame.
    pub rays_per_probe: u32,
    /// Temporal blend factor (0 = no history, 1 = full history).
    pub temporal_blend: f32,
    /// Maximum trace distance for probe rays.
    pub max_trace_distance: f32,
    /// Whether to apply spatial filtering.
    pub spatial_filter_enabled: bool,
    /// Spatial filter radius in tiles.
    pub spatial_filter_radius: u32,
    /// Depth threshold for bilateral filtering.
    pub depth_threshold: f32,
    /// Normal threshold for bilateral filtering.
    pub normal_threshold: f32,
}

impl Default for ScreenProbeSettings {
    fn default() -> Self {
        Self {
            tile_size: PROBE_TILE_SIZE,
            rays_per_probe: DEFAULT_RAYS_PER_PROBE,
            temporal_blend: DEFAULT_TEMPORAL_BLEND,
            max_trace_distance: 200.0,
            spatial_filter_enabled: true,
            spatial_filter_radius: SPATIAL_FILTER_RADIUS,
            depth_threshold: 0.05,
            normal_threshold: 0.7,
        }
    }
}

impl ScreenProbeSettings {
    /// Low quality settings (fewer rays, no spatial filter).
    pub fn low() -> Self {
        Self {
            rays_per_probe: 8,
            temporal_blend: 0.95,
            spatial_filter_enabled: false,
            ..Default::default()
        }
    }

    /// Medium quality settings.
    pub fn medium() -> Self {
        Self {
            rays_per_probe: 32,
            temporal_blend: 0.9,
            ..Default::default()
        }
    }

    /// High quality settings.
    pub fn high() -> Self {
        Self {
            rays_per_probe: 64,
            temporal_blend: 0.85,
            spatial_filter_radius: 2,
            ..Default::default()
        }
    }

    /// Ultra quality settings (for reference / screenshots).
    pub fn ultra() -> Self {
        Self {
            rays_per_probe: MAX_RAYS_PER_PROBE,
            temporal_blend: 0.8,
            spatial_filter_radius: 3,
            ..Default::default()
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
    fn hemisphere_samples_above_plane() {
        let normal = Vec3::Y;
        for i in 0..64 {
            let dir = hemisphere_sample_cosine_weighted(normal, i, 64, 42);
            assert!(
                dir.dot(normal) >= -0.01,
                "Sample {} is below hemisphere: dot = {}",
                i,
                dir.dot(normal)
            );
        }
    }

    #[test]
    fn probe_grid_dimensions() {
        let grid = ScreenProbeGrid::new(1920, 1080);
        assert_eq!(grid.grid_width, (1920 + 7) / 8);
        assert_eq!(grid.grid_height, (1080 + 7) / 8);
        assert_eq!(grid.probe_count(), (grid.grid_width * grid.grid_height) as usize);
    }

    #[test]
    fn probe_accumulation() {
        let mut probe = ScreenProbe::new(0, 0);
        probe.position = Vec3::ZERO;
        probe.normal = Vec3::Y;
        probe.valid = true;

        // Accumulate some samples.
        probe.accumulate_sample(Vec3::Y, Vec3::ONE);
        probe.accumulate_sample(Vec3::new(0.5, 0.866, 0.0).normalize(), Vec3::ONE);
        assert_eq!(probe.sample_count, 2);

        probe.finalize(0.0);

        // Should have some irradiance looking up.
        let irr = probe.evaluate_irradiance(Vec3::Y);
        assert!(irr.x > 0.0);
    }

    #[test]
    fn surfel_creation() {
        let surfel = Surfel::new(Vec3::ZERO, Vec3::Y, Vec3::ONE);
        assert!(surfel.valid);
        assert_eq!(surfel.direct_radiance(), Vec3::ZERO); // no lighting yet
    }

    #[test]
    fn surface_cache_add() {
        let mut cache = SurfaceCache::new(100);
        let idx = cache.add_surfel(Surfel::new(Vec3::ZERO, Vec3::Y, Vec3::ONE));
        assert_eq!(idx, Some(0));
        assert_eq!(cache.surfel_count(), 1);
    }

    #[test]
    fn probe_settings_presets() {
        let low = ScreenProbeSettings::low();
        let high = ScreenProbeSettings::high();
        assert!(low.rays_per_probe < high.rays_per_probe);
    }

    #[test]
    fn radical_inverse_range() {
        for i in 0..100 {
            let val = radical_inverse_vdc(i);
            assert!(val >= 0.0 && val < 1.0);
        }
    }
}
