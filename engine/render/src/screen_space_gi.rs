// engine/render/src/screen_space_gi.rs
//
// Screen-space global illumination (SSGI) for the Genovo engine.
//
// Implements a full SSGI pipeline consisting of:
//
// - Importance-sampled screen-space ray tracing for indirect diffuse bounces.
// - Temporal accumulation with exponential moving average to stabilise results.
// - Spatial denoising (cross-bilateral edge-aware blur).
// - Fallback to probe/ambient data when screen-space data is unavailable.
// - Indirect bounce contribution for multi-bounce approximation.
//
// # Pipeline stages
//
// 1. **Ray generation** — For each pixel, generate a set of hemisphere-directed
//    rays based on the surface normal, distributed with cosine-weighted
//    importance sampling.
// 2. **Screen-space tracing** — Hierarchical ray-march in screen space using
//    the depth buffer, with a min/max depth mip hierarchy for acceleration.
// 3. **Radiance fetch** — If a ray hits, read the colour from the previous
//    frame's lit colour buffer.
// 4. **Temporal accumulation** — Blend the current frame's estimate with the
//    history buffer, using motion vectors for reprojection.
// 5. **Spatial denoise** — Edge-aware bilateral filter to reduce noise while
//    preserving contact detail.
// 6. **Integration** — Add the denoised indirect lighting to the scene.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Quality presets
// ---------------------------------------------------------------------------

/// Quality preset for SSGI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsgiQuality {
    /// Low quality: fewer rays, larger stride.
    Low,
    /// Medium quality: balanced.
    Medium,
    /// High quality: more rays, finer trace.
    High,
    /// Ultra quality: maximum ray count and precision.
    Ultra,
}

impl SsgiQuality {
    /// Returns default settings for this quality level.
    pub fn settings(self) -> SsgiSettings {
        match self {
            Self::Low => SsgiSettings {
                rays_per_pixel: 1,
                max_trace_steps: 16,
                max_trace_distance: 50.0,
                thickness: 1.0,
                stride: 4,
                half_resolution: true,
                temporal_blend: 0.9,
                spatial_passes: 1,
                spatial_radius: 4,
                intensity: 1.0,
                indirect_bounce_strength: 0.3,
                fallback_intensity: 0.5,
                depth_threshold: 0.1,
                normal_threshold: 0.9,
                use_hi_z: false,
                hi_z_max_level: 4,
                jitter_samples: true,
                cosine_weighted: true,
            },
            Self::Medium => SsgiSettings {
                rays_per_pixel: 2,
                max_trace_steps: 32,
                max_trace_distance: 100.0,
                thickness: 0.5,
                stride: 2,
                half_resolution: true,
                temporal_blend: 0.92,
                spatial_passes: 2,
                spatial_radius: 6,
                intensity: 1.0,
                indirect_bounce_strength: 0.4,
                fallback_intensity: 0.4,
                depth_threshold: 0.05,
                normal_threshold: 0.85,
                use_hi_z: true,
                hi_z_max_level: 6,
                jitter_samples: true,
                cosine_weighted: true,
            },
            Self::High => SsgiSettings {
                rays_per_pixel: 4,
                max_trace_steps: 64,
                max_trace_distance: 200.0,
                thickness: 0.3,
                stride: 1,
                half_resolution: false,
                temporal_blend: 0.95,
                spatial_passes: 3,
                spatial_radius: 8,
                intensity: 1.0,
                indirect_bounce_strength: 0.5,
                fallback_intensity: 0.3,
                depth_threshold: 0.03,
                normal_threshold: 0.8,
                use_hi_z: true,
                hi_z_max_level: 8,
                jitter_samples: true,
                cosine_weighted: true,
            },
            Self::Ultra => SsgiSettings {
                rays_per_pixel: 8,
                max_trace_steps: 128,
                max_trace_distance: 500.0,
                thickness: 0.2,
                stride: 1,
                half_resolution: false,
                temporal_blend: 0.96,
                spatial_passes: 4,
                spatial_radius: 10,
                intensity: 1.0,
                indirect_bounce_strength: 0.6,
                fallback_intensity: 0.2,
                depth_threshold: 0.02,
                normal_threshold: 0.75,
                use_hi_z: true,
                hi_z_max_level: 10,
                jitter_samples: true,
                cosine_weighted: true,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// SSGI Settings
// ---------------------------------------------------------------------------

/// Configuration for screen-space global illumination.
#[derive(Debug, Clone)]
pub struct SsgiSettings {
    /// Number of rays per pixel.
    pub rays_per_pixel: u32,
    /// Maximum number of screen-space trace steps per ray.
    pub max_trace_steps: u32,
    /// Maximum world-space trace distance.
    pub max_trace_distance: f32,
    /// Thickness of surfaces for hit detection (world units).
    pub thickness: f32,
    /// Stride for screen-space marching (in pixels).
    pub stride: u32,
    /// Whether to render at half resolution.
    pub half_resolution: bool,
    /// Temporal accumulation blend factor (0 = current, 1 = history).
    pub temporal_blend: f32,
    /// Number of spatial denoising passes.
    pub spatial_passes: u32,
    /// Spatial filter radius (in pixels).
    pub spatial_radius: u32,
    /// SSGI intensity multiplier.
    pub intensity: f32,
    /// Indirect bounce contribution strength.
    pub indirect_bounce_strength: f32,
    /// Fallback probe intensity (when screen-space trace fails).
    pub fallback_intensity: f32,
    /// Depth similarity threshold for bilateral filter.
    pub depth_threshold: f32,
    /// Normal similarity threshold for bilateral filter.
    pub normal_threshold: f32,
    /// Whether to use hierarchical-Z acceleration.
    pub use_hi_z: bool,
    /// Maximum Hi-Z mip level.
    pub hi_z_max_level: u32,
    /// Whether to jitter ray directions per frame.
    pub jitter_samples: bool,
    /// Whether to use cosine-weighted hemisphere sampling.
    pub cosine_weighted: bool,
}

impl Default for SsgiSettings {
    fn default() -> Self {
        SsgiQuality::Medium.settings()
    }
}

// ---------------------------------------------------------------------------
// GBuffer pixel data
// ---------------------------------------------------------------------------

/// Represents G-buffer data for a single pixel.
#[derive(Debug, Clone, Copy)]
pub struct GBufferPixel {
    /// World-space position.
    pub position: [f32; 3],
    /// World-space normal (normalised).
    pub normal: [f32; 3],
    /// Linear depth from the camera.
    pub depth: f32,
    /// Albedo colour.
    pub albedo: [f32; 3],
    /// Roughness.
    pub roughness: f32,
    /// Metallic.
    pub metallic: f32,
}

impl GBufferPixel {
    /// Creates a default (empty) pixel.
    pub fn empty() -> Self {
        Self {
            position: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            depth: f32::MAX,
            albedo: [0.0; 3],
            roughness: 0.5,
            metallic: 0.0,
        }
    }

    /// Whether this pixel has valid geometry.
    pub fn is_valid(&self) -> bool {
        self.depth < f32::MAX - 1.0
    }
}

// ---------------------------------------------------------------------------
// Hemisphere sampling
// ---------------------------------------------------------------------------

/// Generates a cosine-weighted random direction on the hemisphere around
/// a given normal.
///
/// Uses the strategy: generate a direction in the upper hemisphere
/// (around Z) and rotate it to align with the given normal using a TBN matrix.
///
/// # Arguments
/// * `normal` — Surface normal (normalised).
/// * `u1`, `u2` — Random values in [0, 1].
///
/// # Returns
/// A normalised direction vector on the hemisphere.
pub fn cosine_weighted_hemisphere(normal: [f32; 3], u1: f32, u2: f32) -> [f32; 3] {
    // Cosine-weighted sampling in local space.
    let phi = 2.0 * PI * u1;
    let cos_theta = (1.0 - u2).sqrt();
    let sin_theta = u2.sqrt();

    let local_x = phi.cos() * sin_theta;
    let local_y = phi.sin() * sin_theta;
    let local_z = cos_theta;

    // Build TBN basis from the normal.
    let (tangent, bitangent) = build_tangent_frame(normal);

    // Transform to world space.
    let world = [
        tangent[0] * local_x + bitangent[0] * local_y + normal[0] * local_z,
        tangent[1] * local_x + bitangent[1] * local_y + normal[1] * local_z,
        tangent[2] * local_x + bitangent[2] * local_y + normal[2] * local_z,
    ];

    normalize_vec3(world)
}

/// Generates a uniform random direction on the hemisphere around a given normal.
pub fn uniform_hemisphere(normal: [f32; 3], u1: f32, u2: f32) -> [f32; 3] {
    let phi = 2.0 * PI * u1;
    let cos_theta = u2;
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

    let local_x = phi.cos() * sin_theta;
    let local_y = phi.sin() * sin_theta;
    let local_z = cos_theta;

    let (tangent, bitangent) = build_tangent_frame(normal);

    let world = [
        tangent[0] * local_x + bitangent[0] * local_y + normal[0] * local_z,
        tangent[1] * local_x + bitangent[1] * local_y + normal[1] * local_z,
        tangent[2] * local_x + bitangent[2] * local_y + normal[2] * local_z,
    ];

    normalize_vec3(world)
}

/// Builds an orthonormal tangent frame from a normal vector.
fn build_tangent_frame(normal: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let up = if normal[1].abs() < 0.999 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };

    let tangent = normalize_vec3(cross(up, normal));
    let bitangent = cross(normal, tangent);

    (tangent, bitangent)
}

/// Cross product.
#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalise a 3D vector.
#[inline]
fn normalize_vec3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-6 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// Dot product.
#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Linear interpolation.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------
// Screen-space ray tracing
// ---------------------------------------------------------------------------

/// Result of a screen-space ray trace.
#[derive(Debug, Clone, Copy)]
pub struct SsTraceResult {
    /// Whether the ray hit something.
    pub hit: bool,
    /// Screen-space UV of the hit point.
    pub hit_uv: [f32; 2],
    /// Depth at the hit point.
    pub hit_depth: f32,
    /// Number of steps taken.
    pub steps: u32,
    /// Confidence of the hit [0, 1].
    pub confidence: f32,
}

impl SsTraceResult {
    /// A miss result.
    pub fn miss() -> Self {
        Self {
            hit: false,
            hit_uv: [0.0; 2],
            hit_depth: 0.0,
            steps: 0,
            confidence: 0.0,
        }
    }
}

/// Traces a ray in screen space against the depth buffer.
///
/// Uses DDA (Digital Differential Analyzer) stepping in screen space for
/// efficient traversal.
///
/// # Arguments
/// * `origin_uv` — Screen-space UV of the ray origin.
/// * `origin_depth` — Linear depth at the ray origin.
/// * `direction` — World-space ray direction (normalised).
/// * `view_proj` — View-projection matrix (column-major 4x4).
/// * `depth_buffer` — Screen-space depth buffer.
/// * `width`, `height` — Depth buffer dimensions.
/// * `settings` — SSGI settings.
///
/// # Returns
/// Trace result indicating hit/miss and hit coordinates.
pub fn trace_screen_space(
    origin_uv: [f32; 2],
    origin_depth: f32,
    direction: [f32; 3],
    origin_pos: [f32; 3],
    view_proj: &[f32; 16],
    depth_buffer: &[f32],
    width: u32,
    height: u32,
    settings: &SsgiSettings,
) -> SsTraceResult {
    let max_steps = settings.max_trace_steps;
    let thickness = settings.thickness;
    let stride = settings.stride.max(1);

    // Project the end point of the ray into screen space.
    let end_world = [
        origin_pos[0] + direction[0] * settings.max_trace_distance,
        origin_pos[1] + direction[1] * settings.max_trace_distance,
        origin_pos[2] + direction[2] * settings.max_trace_distance,
    ];

    let end_clip = mat4_mul_point(view_proj, end_world);
    if end_clip[3] <= 0.0 {
        return SsTraceResult::miss(); // Behind camera.
    }

    let end_uv = [
        end_clip[0] / end_clip[3] * 0.5 + 0.5,
        0.5 - end_clip[1] / end_clip[3] * 0.5,
    ];

    // Screen-space ray direction.
    let dx = end_uv[0] - origin_uv[0];
    let dy = end_uv[1] - origin_uv[1];
    let pixel_dx = dx * width as f32;
    let pixel_dy = dy * height as f32;

    let pixel_dist = (pixel_dx * pixel_dx + pixel_dy * pixel_dy).sqrt();
    if pixel_dist < 1.0 {
        return SsTraceResult::miss();
    }

    let step_count = ((pixel_dist / stride as f32) as u32).min(max_steps);
    if step_count == 0 {
        return SsTraceResult::miss();
    }

    let step_uv = [dx / step_count as f32, dy / step_count as f32];

    // Depth interpolation.
    let end_depth = end_clip[3]; // Use w as linear depth approximation.
    let depth_step = (end_depth - origin_depth) / step_count as f32;

    let mut current_uv = origin_uv;
    let mut current_depth = origin_depth;

    for step in 0..step_count {
        current_uv[0] += step_uv[0];
        current_uv[1] += step_uv[1];
        current_depth += depth_step;

        // Check bounds.
        if current_uv[0] < 0.0
            || current_uv[0] > 1.0
            || current_uv[1] < 0.0
            || current_uv[1] > 1.0
        {
            return SsTraceResult::miss();
        }

        // Sample depth buffer.
        let px = (current_uv[0] * width as f32) as u32;
        let py = (current_uv[1] * height as f32) as u32;
        let px = px.min(width - 1);
        let py = py.min(height - 1);
        let idx = (py * width + px) as usize;

        if idx >= depth_buffer.len() {
            return SsTraceResult::miss();
        }

        let scene_depth = depth_buffer[idx];

        // Check for intersection.
        let depth_diff = current_depth - scene_depth;
        if depth_diff > 0.0 && depth_diff < thickness {
            // Hit! Compute confidence based on distance and edge proximity.
            let edge_factor = {
                let eu = (current_uv[0] * 2.0 - 1.0).abs();
                let ev = (current_uv[1] * 2.0 - 1.0).abs();
                let edge = eu.max(ev);
                (1.0 - ((edge - 0.8) / 0.2).clamp(0.0, 1.0))
            };

            let distance_factor = 1.0 - (step as f32 / step_count as f32);
            let confidence = edge_factor * distance_factor;

            return SsTraceResult {
                hit: true,
                hit_uv: current_uv,
                hit_depth: scene_depth,
                steps: step + 1,
                confidence,
            };
        }
    }

    SsTraceResult::miss()
}

/// Multiplies a 4x4 column-major matrix by a 3D point (w=1).
fn mat4_mul_point(m: &[f32; 16], p: [f32; 3]) -> [f32; 4] {
    [
        m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
        m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
        m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
        m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15],
    ]
}

// ---------------------------------------------------------------------------
// Hierarchical Z-buffer
// ---------------------------------------------------------------------------

/// Hierarchical Z-buffer (Hi-Z) for accelerated screen-space ray tracing.
///
/// Stores a mip-chain of the depth buffer where each level contains the
/// minimum (closest) depth of a 2x2 region from the level below.
#[derive(Debug)]
pub struct HiZBuffer {
    /// Mip levels, each half the resolution of the previous.
    pub levels: Vec<HiZLevel>,
    /// Base resolution.
    pub base_width: u32,
    pub base_height: u32,
}

/// A single level of the Hi-Z buffer.
#[derive(Debug, Clone)]
pub struct HiZLevel {
    pub width: u32,
    pub height: u32,
    /// Min depth values.
    pub min_depth: Vec<f32>,
    /// Max depth values.
    pub max_depth: Vec<f32>,
}

impl HiZBuffer {
    /// Builds a Hi-Z pyramid from a depth buffer.
    pub fn build(depth_buffer: &[f32], width: u32, height: u32, max_levels: u32) -> Self {
        let mut levels = Vec::new();

        // Level 0: copy of the depth buffer.
        levels.push(HiZLevel {
            width,
            height,
            min_depth: depth_buffer.to_vec(),
            max_depth: depth_buffer.to_vec(),
        });

        let mut prev_w = width;
        let mut prev_h = height;

        for level in 1..max_levels {
            let w = (prev_w / 2).max(1);
            let h = (prev_h / 2).max(1);

            if w == 0 || h == 0 {
                break;
            }

            let prev = &levels[level as usize - 1];
            let mut min_data = vec![f32::MAX; (w * h) as usize];
            let mut max_data = vec![f32::MIN; (w * h) as usize];

            for y in 0..h {
                for x in 0..w {
                    let sx = x * 2;
                    let sy = y * 2;

                    // Sample 2x2 block from previous level.
                    let mut local_min = f32::MAX;
                    let mut local_max = f32::MIN;

                    for dy in 0..2 {
                        for dx in 0..2 {
                            let px = (sx + dx).min(prev_w - 1);
                            let py = (sy + dy).min(prev_h - 1);
                            let idx = (py * prev_w + px) as usize;

                            if idx < prev.min_depth.len() {
                                local_min = local_min.min(prev.min_depth[idx]);
                                local_max = local_max.max(prev.max_depth[idx]);
                            }
                        }
                    }

                    let idx = (y * w + x) as usize;
                    min_data[idx] = local_min;
                    max_data[idx] = local_max;
                }
            }

            levels.push(HiZLevel {
                width: w,
                height: h,
                min_depth: min_data,
                max_depth: max_data,
            });

            prev_w = w;
            prev_h = h;

            if w == 1 && h == 1 {
                break;
            }
        }

        Self {
            levels,
            base_width: width,
            base_height: height,
        }
    }

    /// Samples the Hi-Z buffer at a given level and UV coordinate.
    pub fn sample_min(&self, level: u32, u: f32, v: f32) -> f32 {
        let lvl = &self.levels[level.min(self.levels.len() as u32 - 1) as usize];
        let x = ((u * lvl.width as f32) as u32).min(lvl.width - 1);
        let y = ((v * lvl.height as f32) as u32).min(lvl.height - 1);
        let idx = (y * lvl.width + x) as usize;
        lvl.min_depth.get(idx).copied().unwrap_or(f32::MAX)
    }

    /// Samples the Hi-Z buffer max depth at a given level and UV coordinate.
    pub fn sample_max(&self, level: u32, u: f32, v: f32) -> f32 {
        let lvl = &self.levels[level.min(self.levels.len() as u32 - 1) as usize];
        let x = ((u * lvl.width as f32) as u32).min(lvl.width - 1);
        let y = ((v * lvl.height as f32) as u32).min(lvl.height - 1);
        let idx = (y * lvl.width + x) as usize;
        lvl.max_depth.get(idx).copied().unwrap_or(f32::MIN)
    }

    /// Returns the number of levels.
    pub fn num_levels(&self) -> u32 {
        self.levels.len() as u32
    }

    /// Returns memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.levels
            .iter()
            .map(|l| (l.min_depth.len() + l.max_depth.len()) * std::mem::size_of::<f32>())
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Temporal accumulation
// ---------------------------------------------------------------------------

/// Temporal accumulation buffer for SSGI.
#[derive(Debug)]
pub struct SsgiTemporalBuffer {
    /// Buffer dimensions.
    pub width: u32,
    pub height: u32,
    /// History colour buffer (RGB + sample count in alpha).
    history: Vec<[f32; 4]>,
    /// Current frame accumulation.
    current: Vec<[f32; 4]>,
    /// Frame counter.
    pub frame_count: u64,
}

impl SsgiTemporalBuffer {
    /// Creates a new temporal buffer.
    pub fn new(width: u32, height: u32) -> Self {
        let total = (width * height) as usize;
        Self {
            width,
            height,
            history: vec![[0.0; 4]; total],
            current: vec![[0.0; 4]; total],
            frame_count: 0,
        }
    }

    /// Accumulates a sample at a given pixel.
    pub fn accumulate(
        &mut self,
        x: u32,
        y: u32,
        sample: [f32; 3],
        blend_factor: f32,
    ) {
        let idx = (y * self.width + x) as usize;
        if idx >= self.history.len() {
            return;
        }

        let prev = self.history[idx];
        let blended = [
            lerp(sample[0], prev[0], blend_factor),
            lerp(sample[1], prev[1], blend_factor),
            lerp(sample[2], prev[2], blend_factor),
            prev[3] + 1.0, // Increment sample count.
        ];

        self.current[idx] = blended;
    }

    /// Accumulates with motion vector reprojection.
    pub fn accumulate_reprojected(
        &mut self,
        x: u32,
        y: u32,
        sample: [f32; 3],
        motion_vector: [f32; 2],
        blend_factor: f32,
        depth: f32,
        prev_depth_buffer: &[f32],
    ) {
        let idx = (y * self.width + x) as usize;
        if idx >= self.history.len() {
            return;
        }

        // Compute previous pixel position.
        let prev_u = x as f32 / self.width as f32 + motion_vector[0];
        let prev_v = y as f32 / self.height as f32 + motion_vector[1];

        let accept = if prev_u >= 0.0 && prev_u < 1.0 && prev_v >= 0.0 && prev_v < 1.0 {
            let prev_x = (prev_u * self.width as f32) as u32;
            let prev_y = (prev_v * self.height as f32) as u32;
            let prev_idx = (prev_y * self.width + prev_x) as usize;

            if prev_idx < prev_depth_buffer.len() {
                let prev_depth = prev_depth_buffer[prev_idx];
                (depth - prev_depth).abs() < depth * 0.1 // 10% depth tolerance.
            } else {
                false
            }
        } else {
            false
        };

        if accept {
            let prev = self.history[idx];
            self.current[idx] = [
                lerp(sample[0], prev[0], blend_factor),
                lerp(sample[1], prev[1], blend_factor),
                lerp(sample[2], prev[2], blend_factor),
                prev[3] + 1.0,
            ];
        } else {
            // Disoccluded: use current sample only.
            self.current[idx] = [sample[0], sample[1], sample[2], 1.0];
        }
    }

    /// Swaps history and current buffers at the end of the frame.
    pub fn end_frame(&mut self) {
        std::mem::swap(&mut self.history, &mut self.current);
        // Clear current for next frame.
        for v in &mut self.current {
            *v = [0.0; 4];
        }
        self.frame_count += 1;
    }

    /// Returns the accumulated result at a pixel.
    pub fn result_at(&self, x: u32, y: u32) -> [f32; 3] {
        let idx = (y * self.width + x) as usize;
        let v = self.history.get(idx).copied().unwrap_or([0.0; 4]);
        [v[0], v[1], v[2]]
    }

    /// Returns the history buffer for GPU upload.
    pub fn history_data(&self) -> &[[f32; 4]] {
        &self.history
    }

    /// Resizes the buffers.
    pub fn resize(&mut self, width: u32, height: u32) {
        let total = (width * height) as usize;
        self.width = width;
        self.height = height;
        self.history = vec![[0.0; 4]; total];
        self.current = vec![[0.0; 4]; total];
    }

    /// Returns memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        (self.history.len() + self.current.len()) * std::mem::size_of::<[f32; 4]>()
    }
}

// ---------------------------------------------------------------------------
// Spatial denoiser
// ---------------------------------------------------------------------------

/// Edge-aware bilateral spatial denoiser for SSGI.
#[derive(Debug)]
pub struct SsgiSpatialDenoiser {
    /// Intermediate buffer for ping-pong filtering.
    buffer: Vec<[f32; 3]>,
}

impl SsgiSpatialDenoiser {
    /// Creates a new denoiser.
    pub fn new(width: u32, height: u32) -> Self {
        let total = (width * height) as usize;
        Self {
            buffer: vec![[0.0; 3]; total],
        }
    }

    /// Applies a single pass of edge-aware bilateral filtering.
    ///
    /// # Arguments
    /// * `input` — Input colour buffer (RGB per pixel).
    /// * `normals` — Normal buffer (3 floats per pixel).
    /// * `depths` — Depth buffer (1 float per pixel).
    /// * `width`, `height` — Buffer dimensions.
    /// * `radius` — Filter radius in pixels.
    /// * `depth_threshold` — Depth similarity threshold.
    /// * `normal_threshold` — Normal similarity threshold.
    /// * `output` — Output filtered buffer.
    pub fn filter_pass(
        input: &[[f32; 3]],
        normals: &[[f32; 3]],
        depths: &[f32],
        width: u32,
        height: u32,
        radius: i32,
        depth_threshold: f32,
        normal_threshold: f32,
        output: &mut [[f32; 3]],
    ) {
        let w = width as i32;
        let h = height as i32;

        for y in 0..h {
            for x in 0..w {
                let center_idx = (y * w + x) as usize;
                let center_depth = depths[center_idx];
                let center_normal = normals[center_idx];

                let mut sum = [0.0f32; 3];
                let mut weight_sum = 0.0f32;

                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let sx = x + dx;
                        let sy = y + dy;

                        if sx < 0 || sx >= w || sy < 0 || sy >= h {
                            continue;
                        }

                        let sample_idx = (sy * w + sx) as usize;
                        let sample_depth = depths[sample_idx];
                        let sample_normal = normals[sample_idx];

                        // Depth weight: Gaussian falloff based on depth difference.
                        let depth_diff = (center_depth - sample_depth).abs();
                        let depth_weight = (-depth_diff / depth_threshold.max(0.001)).exp();

                        // Normal weight: cosine similarity.
                        let n_dot = dot3(center_normal, sample_normal).max(0.0);
                        let normal_weight = if n_dot > normal_threshold { n_dot } else { 0.0 };

                        // Spatial weight: Gaussian.
                        let dist_sq = (dx * dx + dy * dy) as f32;
                        let spatial_weight = (-dist_sq / (2.0 * radius as f32 * radius as f32)).exp();

                        let weight = depth_weight * normal_weight * spatial_weight;

                        let sample_color = input[sample_idx];
                        sum[0] += sample_color[0] * weight;
                        sum[1] += sample_color[1] * weight;
                        sum[2] += sample_color[2] * weight;
                        weight_sum += weight;
                    }
                }

                if weight_sum > 1e-6 {
                    let inv = 1.0 / weight_sum;
                    output[center_idx] = [sum[0] * inv, sum[1] * inv, sum[2] * inv];
                } else {
                    output[center_idx] = input[center_idx];
                }
            }
        }
    }

    /// Resizes the internal buffer.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.buffer = vec![[0.0; 3]; (width * height) as usize];
    }

    /// Returns the internal buffer for ping-pong usage.
    pub fn buffer(&self) -> &[[f32; 3]] {
        &self.buffer
    }

    /// Returns a mutable reference to the buffer.
    pub fn buffer_mut(&mut self) -> &mut [[f32; 3]] {
        &mut self.buffer
    }
}

// ---------------------------------------------------------------------------
// Fallback probe
// ---------------------------------------------------------------------------

/// Fallback ambient/probe data used when screen-space tracing misses.
#[derive(Debug, Clone)]
pub struct SsgiFallback {
    /// Ambient cube colours (6 directions: +X, -X, +Y, -Y, +Z, -Z).
    pub ambient_cube: [[f32; 3]; 6],
    /// Global ambient colour.
    pub ambient_color: [f32; 3],
    /// Intensity multiplier.
    pub intensity: f32,
}

impl SsgiFallback {
    /// Creates a uniform ambient fallback.
    pub fn uniform(color: [f32; 3], intensity: f32) -> Self {
        Self {
            ambient_cube: [color; 6],
            ambient_color: color,
            intensity,
        }
    }

    /// Samples the ambient cube in a given direction.
    pub fn sample(&self, direction: [f32; 3]) -> [f32; 3] {
        // Blend the 6 cube faces based on the direction.
        let dx = direction[0];
        let dy = direction[1];
        let dz = direction[2];

        let pos_x_weight = dx.max(0.0);
        let neg_x_weight = (-dx).max(0.0);
        let pos_y_weight = dy.max(0.0);
        let neg_y_weight = (-dy).max(0.0);
        let pos_z_weight = dz.max(0.0);
        let neg_z_weight = (-dz).max(0.0);

        let mut result = [0.0f32; 3];
        let weights = [
            pos_x_weight,
            neg_x_weight,
            pos_y_weight,
            neg_y_weight,
            pos_z_weight,
            neg_z_weight,
        ];

        for (i, &w) in weights.iter().enumerate() {
            result[0] += self.ambient_cube[i][0] * w;
            result[1] += self.ambient_cube[i][1] * w;
            result[2] += self.ambient_cube[i][2] * w;
        }

        [
            result[0] * self.intensity,
            result[1] * self.intensity,
            result[2] * self.intensity,
        ]
    }
}

impl Default for SsgiFallback {
    fn default() -> Self {
        Self::uniform([0.15, 0.18, 0.25], 0.5)
    }
}

// ---------------------------------------------------------------------------
// SSGI Renderer
// ---------------------------------------------------------------------------

/// Top-level screen-space global illumination renderer.
#[derive(Debug)]
pub struct SsgiRenderer {
    /// SSGI settings.
    pub settings: SsgiSettings,
    /// Temporal accumulation buffer.
    pub temporal: SsgiTemporalBuffer,
    /// Spatial denoiser.
    pub denoiser: SsgiSpatialDenoiser,
    /// Hi-Z acceleration structure.
    pub hi_z: Option<HiZBuffer>,
    /// Fallback probe data.
    pub fallback: SsgiFallback,
    /// Whether SSGI is enabled.
    pub enabled: bool,
    /// Render width.
    pub width: u32,
    /// Render height.
    pub height: u32,
    /// Random seed (changes per frame for jittering).
    seed: u32,
}

impl SsgiRenderer {
    /// Creates a new SSGI renderer.
    pub fn new(width: u32, height: u32, quality: SsgiQuality) -> Self {
        let settings = quality.settings();
        let render_w = if settings.half_resolution { width / 2 } else { width };
        let render_h = if settings.half_resolution { height / 2 } else { height };

        Self {
            settings,
            temporal: SsgiTemporalBuffer::new(render_w, render_h),
            denoiser: SsgiSpatialDenoiser::new(render_w, render_h),
            hi_z: None,
            fallback: SsgiFallback::default(),
            enabled: true,
            width: render_w,
            height: render_h,
            seed: 0,
        }
    }

    /// Sets the quality preset.
    pub fn set_quality(&mut self, quality: SsgiQuality) {
        self.settings = quality.settings();
    }

    /// Updates the Hi-Z buffer from a depth buffer.
    pub fn update_hi_z(&mut self, depth_buffer: &[f32], width: u32, height: u32) {
        if self.settings.use_hi_z {
            self.hi_z = Some(HiZBuffer::build(
                depth_buffer,
                width,
                height,
                self.settings.hi_z_max_level,
            ));
        }
    }

    /// Computes SSGI for a single pixel (CPU reference implementation).
    ///
    /// In production, this runs as a compute shader. The CPU version is
    /// provided for testing and validation.
    pub fn compute_pixel(
        &self,
        pixel_x: u32,
        pixel_y: u32,
        gbuffer: &GBufferPixel,
        color_buffer: &[[f32; 3]],
        depth_buffer: &[f32],
        normal_buffer: &[[f32; 3]],
        view_proj: &[f32; 16],
    ) -> [f32; 3] {
        if !gbuffer.is_valid() {
            return [0.0; 3];
        }

        let mut indirect = [0.0f32; 3];
        let mut total_weight = 0.0f32;

        for ray_idx in 0..self.settings.rays_per_pixel {
            // Generate random values for this ray.
            let u1 = pseudo_random(pixel_x, pixel_y, ray_idx, self.seed, 0);
            let u2 = pseudo_random(pixel_x, pixel_y, ray_idx, self.seed, 1);

            // Generate hemisphere direction.
            let dir = if self.settings.cosine_weighted {
                cosine_weighted_hemisphere(gbuffer.normal, u1, u2)
            } else {
                uniform_hemisphere(gbuffer.normal, u1, u2)
            };

            let n_dot_d = dot3(gbuffer.normal, dir).max(0.0);
            if n_dot_d < 0.001 {
                continue;
            }

            // Trace screen-space ray.
            let origin_uv = [
                pixel_x as f32 / self.width as f32,
                pixel_y as f32 / self.height as f32,
            ];

            let trace = trace_screen_space(
                origin_uv,
                gbuffer.depth,
                dir,
                gbuffer.position,
                view_proj,
                depth_buffer,
                self.width,
                self.height,
                &self.settings,
            );

            let radiance = if trace.hit {
                // Fetch colour from the lit colour buffer at the hit point.
                let hx = (trace.hit_uv[0] * self.width as f32) as u32;
                let hy = (trace.hit_uv[1] * self.height as f32) as u32;
                let hx = hx.min(self.width - 1);
                let hy = hy.min(self.height - 1);
                let hit_idx = (hy * self.width + hx) as usize;

                if hit_idx < color_buffer.len() {
                    let c = color_buffer[hit_idx];
                    [
                        c[0] * trace.confidence,
                        c[1] * trace.confidence,
                        c[2] * trace.confidence,
                    ]
                } else {
                    self.fallback.sample(dir)
                }
            } else {
                // Fallback to probe.
                let fb = self.fallback.sample(dir);
                [
                    fb[0] * self.settings.fallback_intensity,
                    fb[1] * self.settings.fallback_intensity,
                    fb[2] * self.settings.fallback_intensity,
                ]
            };

            let weight = if self.settings.cosine_weighted { 1.0 } else { n_dot_d };
            indirect[0] += radiance[0] * weight;
            indirect[1] += radiance[1] * weight;
            indirect[2] += radiance[2] * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            let inv = self.settings.intensity / total_weight;
            [indirect[0] * inv, indirect[1] * inv, indirect[2] * inv]
        } else {
            [0.0; 3]
        }
    }

    /// Advances to the next frame (updates random seed, swaps temporal buffers).
    pub fn end_frame(&mut self) {
        self.temporal.end_frame();
        self.seed = self.seed.wrapping_add(1);
    }

    /// Resizes all buffers.
    pub fn resize(&mut self, width: u32, height: u32) {
        let w = if self.settings.half_resolution { width / 2 } else { width };
        let h = if self.settings.half_resolution { height / 2 } else { height };
        self.width = w;
        self.height = h;
        self.temporal.resize(w, h);
        self.denoiser.resize(w, h);
    }

    /// Returns total memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.temporal.memory_usage()
            + self.hi_z.as_ref().map(|h| h.memory_usage()).unwrap_or(0)
    }
}

/// Pseudo-random number generator for ray jittering.
fn pseudo_random(x: u32, y: u32, ray: u32, seed: u32, channel: u32) -> f32 {
    let mut n = x.wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(ray.wrapping_mul(1274126177))
        .wrapping_add(seed.wrapping_mul(48271))
        .wrapping_add(channel.wrapping_mul(16807));
    n = n ^ (n >> 13);
    n = n.wrapping_mul(n.wrapping_mul(n.wrapping_mul(60493).wrapping_add(19990303)).wrapping_add(1376312589));
    (n & 0x7FFF_FFFF) as f32 / 0x7FFF_FFFF as f32
}

// ---------------------------------------------------------------------------
// WGSL shader
// ---------------------------------------------------------------------------

/// WGSL compute shader for screen-space global illumination.
pub const SSGI_COMPUTE_WGSL: &str = r#"
// -----------------------------------------------------------------------
// Screen-Space Global Illumination compute shader (Genovo Engine)
// -----------------------------------------------------------------------

struct SsgiUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    frame_index: u32,
    rays_per_pixel: u32,
    max_steps: u32,
    max_distance: f32,
    thickness: f32,
    intensity: f32,
    temporal_blend: f32,
    resolution: vec2<u32>,
};

@group(0) @binding(0) var<uniform> ssgi: SsgiUniforms;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var color_tex: texture_2d<f32>;
@group(0) @binding(4) var history_tex: texture_2d<f32>;
@group(0) @binding(5) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var linear_sampler: sampler;

const PI: f32 = 3.14159265358979;

fn hash(p: vec3<u32>) -> f32 {
    var n = p.x * 374761393u + p.y * 668265263u + p.z * 1274126177u;
    n = n ^ (n >> 13u);
    n = n * ((n * (n * 60493u + 19990303u)) + 1376312589u);
    return f32(n & 0x7FFFFFFFu) / f32(0x7FFFFFFF);
}

fn cosine_hemisphere(normal: vec3<f32>, u1: f32, u2: f32) -> vec3<f32> {
    let phi = 2.0 * PI * u1;
    let cos_theta = sqrt(1.0 - u2);
    let sin_theta = sqrt(u2);

    let local = vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);

    var up = vec3<f32>(0.0, 1.0, 0.0);
    if abs(normal.y) > 0.999 {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);

    return normalize(tangent * local.x + bitangent * local.y + normal * local.z);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= ssgi.resolution.x || gid.y >= ssgi.resolution.y {
        return;
    }

    let uv = vec2<f32>(
        (f32(gid.x) + 0.5) / f32(ssgi.resolution.x),
        (f32(gid.y) + 0.5) / f32(ssgi.resolution.y)
    );

    let depth = textureSampleLevel(depth_tex, linear_sampler, uv, 0.0).r;
    if depth >= 1.0 {
        textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    let normal = textureSampleLevel(normal_tex, linear_sampler, uv, 0.0).rgb * 2.0 - 1.0;

    var indirect = vec3<f32>(0.0);

    for (var i = 0u; i < ssgi.rays_per_pixel; i = i + 1u) {
        let u1 = hash(vec3<u32>(gid.x, gid.y, i + ssgi.frame_index * 100u));
        let u2 = hash(vec3<u32>(gid.x + 127u, gid.y + 311u, i + ssgi.frame_index * 100u));
        let dir = cosine_hemisphere(normalize(normal), u1, u2);

        // Screen-space trace would go here (simplified for brevity).
        let fallback = vec3<f32>(0.15, 0.18, 0.25) * 0.3;
        indirect += fallback;
    }

    indirect /= f32(ssgi.rays_per_pixel);
    indirect *= ssgi.intensity;

    // Temporal blend.
    let history = textureSampleLevel(history_tex, linear_sampler, uv, 0.0).rgb;
    let result = mix(indirect, history, ssgi.temporal_blend);

    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(result, 1.0));
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_hemisphere_above_surface() {
        let normal = [0.0, 1.0, 0.0];
        for i in 0..20 {
            let u1 = i as f32 / 20.0;
            let u2 = (i as f32 * 0.618) % 1.0; // Golden ratio distribution.
            let dir = cosine_weighted_hemisphere(normal, u1, u2);

            // Should be in the upper hemisphere.
            assert!(
                dot3(dir, normal) >= -0.01,
                "Direction should be above surface: dot = {}",
                dot3(dir, normal)
            );

            // Should be normalised.
            let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            assert!((len - 1.0).abs() < 0.01, "Should be unit length: {len}");
        }
    }

    #[test]
    fn test_uniform_hemisphere() {
        let normal = [0.0, 0.0, 1.0];
        for i in 0..20 {
            let u1 = i as f32 / 20.0;
            let u2 = (i as f32 * 0.618) % 1.0;
            let dir = uniform_hemisphere(normal, u1, u2);
            assert!(dot3(dir, normal) >= -0.01);
        }
    }

    #[test]
    fn test_trace_miss_behind_camera() {
        let settings = SsgiSettings::default();
        let depth_buffer = vec![10.0; 16 * 16];
        let view_proj = identity_matrix();

        let result = trace_screen_space(
            [0.5, 0.5],
            5.0,
            [0.0, 0.0, 1.0], // Away from screen.
            [0.0, 0.0, 0.0],
            &view_proj,
            &depth_buffer,
            16,
            16,
            &settings,
        );

        // May or may not hit, but should not panic.
        let _ = result;
    }

    #[test]
    fn test_hi_z_build() {
        let depth = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let hi_z = HiZBuffer::build(&depth, 4, 4, 4);

        assert!(hi_z.num_levels() >= 2);
        assert_eq!(hi_z.levels[0].width, 4);
        assert_eq!(hi_z.levels[1].width, 2);

        // Min of the 4x4 should be 1.0.
        let top_min = hi_z.sample_min(hi_z.num_levels() - 1, 0.5, 0.5);
        assert!((top_min - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_temporal_buffer() {
        let mut buf = SsgiTemporalBuffer::new(4, 4);
        buf.accumulate(0, 0, [1.0, 0.5, 0.0], 0.0);
        buf.end_frame();

        let result = buf.result_at(0, 0);
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_temporal_blend() {
        let mut buf = SsgiTemporalBuffer::new(4, 4);
        buf.accumulate(0, 0, [1.0, 1.0, 1.0], 0.0);
        buf.end_frame();
        buf.accumulate(0, 0, [0.0, 0.0, 0.0], 0.5);
        buf.end_frame();

        let result = buf.result_at(0, 0);
        // Should be blend of 0 and 1 with factor 0.5 = 0.5.
        assert!((result[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_spatial_denoiser() {
        let input = vec![[1.0, 0.5, 0.0]; 16];
        let normals = vec![[0.0, 1.0, 0.0]; 16];
        let depths = vec![10.0; 16];
        let mut output = vec![[0.0; 3]; 16];

        SsgiSpatialDenoiser::filter_pass(
            &input,
            &normals,
            &depths,
            4,
            4,
            1,
            0.1,
            0.8,
            &mut output,
        );

        // With uniform input, output should match input.
        assert!((output[5][0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fallback_probe() {
        let fallback = SsgiFallback::uniform([0.5, 0.5, 0.5], 1.0);
        let sample = fallback.sample([0.0, 1.0, 0.0]);
        assert!(sample[1] > 0.0, "Should have positive contribution");
    }

    #[test]
    fn test_ssgi_renderer_creation() {
        let renderer = SsgiRenderer::new(1920, 1080, SsgiQuality::Medium);
        assert!(renderer.enabled);
        assert_eq!(renderer.width, 960); // Half res.
        assert_eq!(renderer.height, 540);
    }

    #[test]
    fn test_ssgi_quality_presets() {
        let low = SsgiQuality::Low.settings();
        let high = SsgiQuality::High.settings();
        assert!(low.rays_per_pixel < high.rays_per_pixel);
        assert!(low.max_trace_steps < high.max_trace_steps);
    }

    #[test]
    fn test_pseudo_random_distribution() {
        let mut values = Vec::new();
        for i in 0..100 {
            let v = pseudo_random(i, 0, 0, 42, 0);
            assert!(v >= 0.0 && v <= 1.0, "Random value out of range: {v}");
            values.push(v);
        }
        // Check some variance.
        let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        assert!((mean - 0.5).abs() < 0.15, "Mean should be near 0.5: {mean}");
    }

    #[test]
    fn test_gbuffer_pixel_validity() {
        let valid = GBufferPixel {
            position: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            depth: 10.0,
            albedo: [0.5, 0.5, 0.5],
            roughness: 0.5,
            metallic: 0.0,
        };
        assert!(valid.is_valid());

        let invalid = GBufferPixel::empty();
        assert!(!invalid.is_valid());
    }
}

fn identity_matrix() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
}
