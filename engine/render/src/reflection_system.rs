// engine/render/src/reflection_system.rs
//
// Unified reflection system for the Genovo engine.
//
// Provides a comprehensive set of reflection techniques that can be combined
// and blended:
//
// - **Planar reflections** — Render the scene from a mirrored camera for flat
//   reflective surfaces (mirrors, water).
// - **Cubemap reflections** — Pre-baked or captured cubemaps for environment
//   reflections.
// - **Real-time cubemap capture** — Per-frame cubemap capture at probe positions.
// - **Screen-space reflections (SSR)** — Ray-march in screen space as fallback.
// - **Reflection probe blending** — Blend between multiple probes based on
//   distance and priority.
// - **Box/sphere projection correction** — Parallax-correct cubemap lookups.
// - **Roughness-based mip selection** — Sample rougher reflections from lower
//   mip levels.
// - **Temporal filtering** — Accumulate SSR results across frames to reduce
//   noise.
//
// # Architecture
//
// The reflection system is designed as a multi-fallback pipeline:
//
// 1. Try SSR first (screen-space, cheapest, most accurate for nearby geometry).
// 2. Fall back to planar reflection if the surface is planar and a capture exists.
// 3. Fall back to probe cubemaps with parallax correction.
// 4. Ultimate fallback: sky cubemap.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Reflection probe
// ---------------------------------------------------------------------------

/// Type of projection volume for a reflection probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeProjectionType {
    /// Box projection — parallax correction using an AABB.
    Box,
    /// Sphere projection — parallax correction using a bounding sphere.
    Sphere,
}

/// A reflection probe captures the environment at a fixed position.
#[derive(Debug, Clone)]
pub struct ReflectionProbe {
    /// Unique identifier.
    pub id: u32,
    /// World-space position of the probe.
    pub position: [f32; 3],
    /// Projection volume type.
    pub projection: ProbeProjectionType,
    /// For `Box`: half-extents of the AABB. For `Sphere`: radius (stored in x).
    pub extents: [f32; 3],
    /// Priority (higher = preferred when overlapping).
    pub priority: i32,
    /// Cubemap texture handle (opaque index).
    pub cubemap_handle: u64,
    /// Number of mip levels in the cubemap.
    pub mip_count: u32,
    /// Cubemap resolution (per face).
    pub resolution: u32,
    /// Whether this probe updates every frame.
    pub real_time: bool,
    /// Update interval for real-time probes (frames). 1 = every frame.
    pub update_interval: u32,
    /// Capture near plane.
    pub near_plane: f32,
    /// Capture far plane.
    pub far_plane: f32,
    /// Influence radius — beyond this, the probe has zero weight.
    pub influence_radius: f32,
    /// Blend distance — soft edge within the influence radius.
    pub blend_distance: f32,
    /// Interior mode: if true, the probe only contributes inside its volume.
    pub interior: bool,
    /// Frame counter for update scheduling.
    pub last_update_frame: u64,
    /// Importance value for scheduling (higher = updated more often).
    pub importance: f32,
}

impl Default for ReflectionProbe {
    fn default() -> Self {
        Self {
            id: 0,
            position: [0.0; 3],
            projection: ProbeProjectionType::Box,
            extents: [5.0, 5.0, 5.0],
            priority: 0,
            cubemap_handle: 0,
            mip_count: 7,
            resolution: 128,
            real_time: false,
            update_interval: 1,
            near_plane: 0.1,
            far_plane: 100.0,
            influence_radius: 10.0,
            blend_distance: 1.0,
            interior: false,
            last_update_frame: 0,
            importance: 1.0,
        }
    }
}

impl ReflectionProbe {
    /// Create a static (baked) reflection probe.
    pub fn new_static(id: u32, position: [f32; 3], extents: [f32; 3], resolution: u32) -> Self {
        Self {
            id,
            position,
            extents,
            resolution,
            mip_count: compute_mip_count(resolution),
            ..Self::default()
        }
    }

    /// Create a real-time reflection probe.
    pub fn new_realtime(id: u32, position: [f32; 3], extents: [f32; 3], resolution: u32) -> Self {
        Self {
            id,
            position,
            extents,
            resolution,
            mip_count: compute_mip_count(resolution),
            real_time: true,
            ..Self::default()
        }
    }

    /// Compute the weight of this probe at a world-space position.
    ///
    /// Returns 0.0 if outside influence, up to 1.0 at the centre.
    pub fn compute_weight(&self, world_pos: [f32; 3]) -> f32 {
        let dist = distance_to_probe(world_pos, self.position);

        if dist > self.influence_radius {
            return 0.0;
        }

        // Check if inside the volume (for interior probes).
        if self.interior {
            if !self.is_inside_volume(world_pos) {
                return 0.0;
            }
        }

        // Smooth falloff near the edge.
        let fade_start = self.influence_radius - self.blend_distance;
        if dist > fade_start {
            let t = (dist - fade_start) / self.blend_distance.max(0.001);
            1.0 - t.clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Check if a world-space point is inside the probe's projection volume.
    pub fn is_inside_volume(&self, world_pos: [f32; 3]) -> bool {
        match self.projection {
            ProbeProjectionType::Box => {
                let dx = (world_pos[0] - self.position[0]).abs();
                let dy = (world_pos[1] - self.position[1]).abs();
                let dz = (world_pos[2] - self.position[2]).abs();
                dx <= self.extents[0] && dy <= self.extents[1] && dz <= self.extents[2]
            }
            ProbeProjectionType::Sphere => {
                let dist = distance_to_probe(world_pos, self.position);
                dist <= self.extents[0]
            }
        }
    }

    /// Apply box projection correction to a reflection direction.
    ///
    /// # Arguments
    /// * `world_pos` — Fragment world position.
    /// * `reflect_dir` — Reflection direction (normalised).
    ///
    /// # Returns
    /// Corrected cubemap lookup direction.
    pub fn box_project(&self, world_pos: [f32; 3], reflect_dir: [f32; 3]) -> [f32; 3] {
        let box_min = [
            self.position[0] - self.extents[0],
            self.position[1] - self.extents[1],
            self.position[2] - self.extents[2],
        ];
        let box_max = [
            self.position[0] + self.extents[0],
            self.position[1] + self.extents[1],
            self.position[2] + self.extents[2],
        ];

        // Ray-box intersection to find the closest face.
        let mut t_max = f32::MAX;
        for i in 0..3 {
            if reflect_dir[i].abs() > 1e-10 {
                let t1 = (box_min[i] - world_pos[i]) / reflect_dir[i];
                let t2 = (box_max[i] - world_pos[i]) / reflect_dir[i];
                let t_near = t1.max(t2); // We want the positive intersection.
                if t_near > 0.0 && t_near < t_max {
                    t_max = t_near;
                }
            }
        }

        if t_max == f32::MAX {
            return reflect_dir;
        }

        // Intersection point.
        let hit = [
            world_pos[0] + reflect_dir[0] * t_max,
            world_pos[1] + reflect_dir[1] * t_max,
            world_pos[2] + reflect_dir[2] * t_max,
        ];

        // Corrected direction: from probe position to intersection.
        let corrected = [
            hit[0] - self.position[0],
            hit[1] - self.position[1],
            hit[2] - self.position[2],
        ];

        normalize3(corrected)
    }

    /// Apply sphere projection correction to a reflection direction.
    pub fn sphere_project(&self, world_pos: [f32; 3], reflect_dir: [f32; 3]) -> [f32; 3] {
        let radius = self.extents[0];

        // Ray-sphere intersection.
        let oc = [
            world_pos[0] - self.position[0],
            world_pos[1] - self.position[1],
            world_pos[2] - self.position[2],
        ];

        let a = dot3(reflect_dir, reflect_dir);
        let b = 2.0 * dot3(oc, reflect_dir);
        let c = dot3(oc, oc) - radius * radius;
        let discriminant = b * b - 4.0 * a * c;

        if discriminant < 0.0 {
            return reflect_dir;
        }

        let t = (-b + discriminant.sqrt()) / (2.0 * a);
        if t <= 0.0 {
            return reflect_dir;
        }

        let hit = [
            world_pos[0] + reflect_dir[0] * t,
            world_pos[1] + reflect_dir[1] * t,
            world_pos[2] + reflect_dir[2] * t,
        ];

        let corrected = [
            hit[0] - self.position[0],
            hit[1] - self.position[1],
            hit[2] - self.position[2],
        ];

        normalize3(corrected)
    }

    /// Apply parallax correction based on the probe's projection type.
    pub fn parallax_correct(&self, world_pos: [f32; 3], reflect_dir: [f32; 3]) -> [f32; 3] {
        match self.projection {
            ProbeProjectionType::Box => self.box_project(world_pos, reflect_dir),
            ProbeProjectionType::Sphere => self.sphere_project(world_pos, reflect_dir),
        }
    }

    /// Determine whether this probe needs an update this frame.
    pub fn needs_update(&self, current_frame: u64) -> bool {
        if !self.real_time {
            return false;
        }
        current_frame - self.last_update_frame >= self.update_interval as u64
    }

    /// Compute the mip level for a given roughness value.
    ///
    /// # Arguments
    /// * `roughness` — Material roughness [0, 1].
    pub fn roughness_to_mip(&self, roughness: f32) -> f32 {
        let max_mip = (self.mip_count as f32 - 1.0).max(0.0);
        roughness * max_mip
    }
}

/// Compute mip count for a cubemap face resolution.
fn compute_mip_count(resolution: u32) -> u32 {
    ((resolution as f32).log2().floor() as u32 + 1).max(1)
}

// ---------------------------------------------------------------------------
// Probe blending
// ---------------------------------------------------------------------------

/// Result of blending multiple reflection probes.
#[derive(Debug, Clone)]
pub struct ProbeBlendResult {
    /// Up to 2 probes to blend between.
    pub probes: [(u32, f32); 2], // (probe_id, weight)
    /// Number of active probes (0, 1, or 2).
    pub count: usize,
}

impl ProbeBlendResult {
    /// No probes.
    pub fn none() -> Self {
        Self { probes: [(0, 0.0); 2], count: 0 }
    }
}

/// Find the two best probes for a given world position and blend between them.
///
/// # Arguments
/// * `probes` — All available reflection probes.
/// * `world_pos` — Fragment world position.
///
/// # Returns
/// The two highest-weight probes and their normalised blend weights.
pub fn blend_probes(probes: &[ReflectionProbe], world_pos: [f32; 3]) -> ProbeBlendResult {
    if probes.is_empty() {
        return ProbeBlendResult::none();
    }

    // Compute weights for all probes and find the top 2.
    let mut best: [(u32, f32, i32); 2] = [(0, 0.0, i32::MIN); 2]; // (id, weight, priority)

    for probe in probes {
        let w = probe.compute_weight(world_pos);
        if w <= 0.0 {
            continue;
        }

        // Insert into top-2 sorted by (priority, weight).
        let score = (probe.priority, w);
        if score.0 > best[0].2 || (score.0 == best[0].2 && score.1 > best[0].1) {
            best[1] = best[0];
            best[0] = (probe.id, w, probe.priority);
        } else if score.0 > best[1].2 || (score.0 == best[1].2 && score.1 > best[1].1) {
            best[1] = (probe.id, w, probe.priority);
        }
    }

    let mut result = ProbeBlendResult::none();

    if best[0].1 > 0.0 {
        result.probes[0] = (best[0].0, best[0].1);
        result.count = 1;

        if best[1].1 > 0.0 {
            result.probes[1] = (best[1].0, best[1].1);
            result.count = 2;

            // Normalise weights.
            let total = result.probes[0].1 + result.probes[1].1;
            if total > 0.0 {
                result.probes[0].1 /= total;
                result.probes[1].1 /= total;
            }
        } else {
            result.probes[0].1 = 1.0;
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Planar reflections
// ---------------------------------------------------------------------------

/// Configuration for planar reflections.
#[derive(Debug, Clone)]
pub struct PlanarReflectionConfig {
    /// Plane point (world space).
    pub plane_point: [f32; 3],
    /// Plane normal (normalised).
    pub plane_normal: [f32; 3],
    /// Render texture resolution (width, height).
    pub resolution: (u32, u32),
    /// Near clip plane.
    pub near_plane: f32,
    /// Far clip plane.
    pub far_plane: f32,
    /// Clip plane offset (push the clip plane slightly to avoid artefacts).
    pub clip_offset: f32,
    /// Texture handle for the reflection render target.
    pub texture_handle: u64,
    /// Whether to render LOD-reduced geometry in the reflection.
    pub use_lod_bias: bool,
    /// LOD bias for reflection rendering (positive = coarser).
    pub lod_bias: f32,
    /// Maximum draw distance for reflected objects.
    pub max_draw_distance: f32,
    /// Layer mask — bitfield of which layers appear in the reflection.
    pub layer_mask: u32,
}

impl Default for PlanarReflectionConfig {
    fn default() -> Self {
        Self {
            plane_point: [0.0, 0.0, 0.0],
            plane_normal: [0.0, 1.0, 0.0],
            resolution: (512, 512),
            near_plane: 0.1,
            far_plane: 200.0,
            clip_offset: 0.01,
            texture_handle: 0,
            use_lod_bias: true,
            lod_bias: 1.0,
            max_draw_distance: 200.0,
            layer_mask: 0xFFFF_FFFF,
        }
    }
}

impl PlanarReflectionConfig {
    /// Compute the reflected camera position.
    ///
    /// # Arguments
    /// * `camera_pos` — Original camera position.
    pub fn reflect_position(&self, camera_pos: [f32; 3]) -> [f32; 3] {
        let n = self.plane_normal;
        let d = dot3(
            [
                camera_pos[0] - self.plane_point[0],
                camera_pos[1] - self.plane_point[1],
                camera_pos[2] - self.plane_point[2],
            ],
            n,
        );
        [
            camera_pos[0] - 2.0 * d * n[0],
            camera_pos[1] - 2.0 * d * n[1],
            camera_pos[2] - 2.0 * d * n[2],
        ]
    }

    /// Compute the reflected camera forward direction.
    ///
    /// # Arguments
    /// * `camera_forward` — Original camera forward direction (normalised).
    pub fn reflect_direction(&self, camera_forward: [f32; 3]) -> [f32; 3] {
        let n = self.plane_normal;
        let d = dot3(camera_forward, n);
        normalize3([
            camera_forward[0] - 2.0 * d * n[0],
            camera_forward[1] - 2.0 * d * n[1],
            camera_forward[2] - 2.0 * d * n[2],
        ])
    }

    /// Compute the reflected camera up vector.
    pub fn reflect_up(&self, camera_up: [f32; 3]) -> [f32; 3] {
        self.reflect_direction(camera_up)
    }

    /// Build the oblique near-clip plane for the reflection camera.
    ///
    /// This clips geometry below the reflection plane so it doesn't appear
    /// above the surface.
    ///
    /// Returns (A, B, C, D) of the clip plane equation Ax + By + Cz + D = 0.
    pub fn oblique_clip_plane(&self) -> [f32; 4] {
        let n = self.plane_normal;
        let d = -(dot3(n, self.plane_point) + self.clip_offset);
        [n[0], n[1], n[2], d]
    }

    /// Compute the reflection matrix (4x4, column-major).
    pub fn reflection_matrix(&self) -> [f32; 16] {
        let n = self.plane_normal;
        let d = -dot3(n, self.plane_point);

        // Reflection matrix: I - 2 * n * nT (extended to 4x4 with plane offset).
        [
            1.0 - 2.0 * n[0] * n[0],
            -2.0 * n[1] * n[0],
            -2.0 * n[2] * n[0],
            0.0,
            -2.0 * n[0] * n[1],
            1.0 - 2.0 * n[1] * n[1],
            -2.0 * n[2] * n[1],
            0.0,
            -2.0 * n[0] * n[2],
            -2.0 * n[1] * n[2],
            1.0 - 2.0 * n[2] * n[2],
            0.0,
            -2.0 * d * n[0],
            -2.0 * d * n[1],
            -2.0 * d * n[2],
            1.0,
        ]
    }
}

// ---------------------------------------------------------------------------
// Screen-space reflections (SSR)
// ---------------------------------------------------------------------------

/// SSR configuration.
#[derive(Debug, Clone)]
pub struct SsrConfig {
    /// Maximum number of ray-march steps.
    pub max_steps: u32,
    /// Maximum ray distance (view space).
    pub max_distance: f32,
    /// Step stride (view-space units per step).
    pub stride: f32,
    /// Stride cutoff — distance at which stride increases.
    pub stride_cutoff: f32,
    /// Binary search refinement steps after a coarse hit.
    pub refinement_steps: u32,
    /// Thickness for depth comparison (view-space units).
    pub thickness: f32,
    /// Jitter amount for ray start (reduces banding).
    pub jitter: f32,
    /// Roughness threshold — surfaces rougher than this skip SSR.
    pub roughness_threshold: f32,
    /// Fade at screen edges (percentage of screen to fade out over).
    pub edge_fade: f32,
    /// Temporal filter blend factor (0 = no temporal, 1 = full history).
    pub temporal_blend: f32,
    /// Resolution scale (0.5 = half-res, 1.0 = full).
    pub resolution_scale: f32,
    /// Minimum roughness to apply SSR (skip perfectly smooth surfaces).
    pub min_roughness: f32,
}

impl Default for SsrConfig {
    fn default() -> Self {
        Self {
            max_steps: 64,
            max_distance: 50.0,
            stride: 0.5,
            stride_cutoff: 20.0,
            refinement_steps: 8,
            thickness: 0.1,
            jitter: 0.0,
            roughness_threshold: 0.6,
            edge_fade: 0.1,
            temporal_blend: 0.9,
            resolution_scale: 1.0,
            min_roughness: 0.0,
        }
    }
}

impl SsrConfig {
    /// High quality preset.
    pub fn high() -> Self {
        Self {
            max_steps: 128,
            max_distance: 100.0,
            stride: 0.25,
            refinement_steps: 16,
            thickness: 0.05,
            temporal_blend: 0.95,
            ..Self::default()
        }
    }

    /// Low quality / performance preset.
    pub fn low() -> Self {
        Self {
            max_steps: 32,
            max_distance: 30.0,
            stride: 1.0,
            refinement_steps: 4,
            thickness: 0.2,
            resolution_scale: 0.5,
            ..Self::default()
        }
    }
}

/// Screen-space reflection ray-march result.
#[derive(Debug, Clone, Copy)]
pub struct SsrHit {
    /// Was a reflection found?
    pub hit: bool,
    /// UV coordinates of the reflected pixel.
    pub uv: [f32; 2],
    /// Confidence/visibility factor [0, 1].
    pub confidence: f32,
    /// Number of march steps taken.
    pub steps: u32,
    /// View-space distance to the reflection hit.
    pub distance: f32,
}

impl SsrHit {
    pub fn miss() -> Self {
        Self { hit: false, uv: [0.0; 2], confidence: 0.0, steps: 0, distance: 0.0 }
    }
}

/// Perform a single SSR ray-march in screen/view space.
///
/// This is a CPU reference implementation; the actual SSR runs in a shader.
///
/// # Arguments
/// * `uv` — Screen UV of the fragment [0, 1].
/// * `view_pos` — View-space position of the fragment.
/// * `view_normal` — View-space normal.
/// * `view_dir` — View-space direction from camera to fragment (normalised).
/// * `roughness` — Surface roughness [0, 1].
/// * `depth_buffer` — Function that returns view-space depth at a UV.
/// * `config` — SSR configuration.
pub fn ssr_ray_march<F>(
    uv: [f32; 2],
    view_pos: [f32; 3],
    view_normal: [f32; 3],
    view_dir: [f32; 3],
    roughness: f32,
    depth_buffer: &F,
    projection: &[f32; 16],
    config: &SsrConfig,
) -> SsrHit
where
    F: Fn(f32, f32) -> f32,
{
    // Skip if roughness exceeds threshold.
    if roughness > config.roughness_threshold || roughness < config.min_roughness {
        return SsrHit::miss();
    }

    // Reflect the view direction.
    let d = dot3(view_dir, view_normal);
    let reflect_dir = [
        view_dir[0] - 2.0 * d * view_normal[0],
        view_dir[1] - 2.0 * d * view_normal[1],
        view_dir[2] - 2.0 * d * view_normal[2],
    ];
    let reflect_dir = normalize3(reflect_dir);

    let mut ray_pos = view_pos;
    let mut prev_depth_diff = 0.0_f32;

    for step in 0..config.max_steps {
        // Advance along the reflected ray.
        let stride = if length3(ray_pos) > config.stride_cutoff {
            config.stride * 2.0
        } else {
            config.stride
        };

        ray_pos[0] += reflect_dir[0] * stride;
        ray_pos[1] += reflect_dir[1] * stride;
        ray_pos[2] += reflect_dir[2] * stride;

        // Check distance limit.
        let ray_dist = length3([
            ray_pos[0] - view_pos[0],
            ray_pos[1] - view_pos[1],
            ray_pos[2] - view_pos[2],
        ]);
        if ray_dist > config.max_distance {
            break;
        }

        // Project to screen space.
        let screen_uv = project_to_screen(ray_pos, projection);

        // Out of screen bounds.
        if screen_uv[0] < 0.0 || screen_uv[0] > 1.0 || screen_uv[1] < 0.0 || screen_uv[1] > 1.0 {
            break;
        }

        // Compare depth.
        let scene_depth = depth_buffer(screen_uv[0], screen_uv[1]);
        let ray_depth = -ray_pos[2]; // View space Z is negative.
        let depth_diff = ray_depth - scene_depth;

        // Hit detection.
        if depth_diff > 0.0 && depth_diff < config.thickness {
            // Binary search refinement.
            let mut refined_pos = [
                ray_pos[0] - reflect_dir[0] * stride,
                ray_pos[1] - reflect_dir[1] * stride,
                ray_pos[2] - reflect_dir[2] * stride,
            ];
            let mut refined_uv = screen_uv;
            let mut half_stride = stride * 0.5;

            for _ in 0..config.refinement_steps {
                refined_pos[0] += reflect_dir[0] * half_stride;
                refined_pos[1] += reflect_dir[1] * half_stride;
                refined_pos[2] += reflect_dir[2] * half_stride;

                refined_uv = project_to_screen(refined_pos, projection);
                let sd = depth_buffer(refined_uv[0].clamp(0.0, 1.0), refined_uv[1].clamp(0.0, 1.0));
                let rd = -refined_pos[2];
                let dd = rd - sd;

                half_stride *= 0.5;
                if dd > 0.0 {
                    refined_pos[0] -= reflect_dir[0] * half_stride * 2.0;
                    refined_pos[1] -= reflect_dir[1] * half_stride * 2.0;
                    refined_pos[2] -= reflect_dir[2] * half_stride * 2.0;
                }
            }

            // Edge fade.
            let edge_x = 1.0 - ((refined_uv[0] - 0.5).abs() * 2.0 / (1.0 - config.edge_fade)).clamp(0.0, 1.0);
            let edge_y = 1.0 - ((refined_uv[1] - 0.5).abs() * 2.0 / (1.0 - config.edge_fade)).clamp(0.0, 1.0);
            let edge_fade = edge_x * edge_y;

            // Distance fade.
            let dist_fade = 1.0 - (ray_dist / config.max_distance).clamp(0.0, 1.0);

            return SsrHit {
                hit: true,
                uv: refined_uv,
                confidence: edge_fade * dist_fade,
                steps: step,
                distance: ray_dist,
            };
        }

        prev_depth_diff = depth_diff;
    }

    SsrHit::miss()
}

// ---------------------------------------------------------------------------
// Temporal filtering for SSR
// ---------------------------------------------------------------------------

/// Temporal SSR accumulation buffer.
#[derive(Debug, Clone)]
pub struct SsrTemporalBuffer {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Previous frame's SSR colour (RGBA linear).
    pub history: Vec<[f32; 4]>,
    /// Previous frame's motion vectors.
    pub prev_motion: Vec<[f32; 2]>,
    /// Temporal blend factor.
    pub blend_factor: f32,
    /// Frame counter.
    pub frame: u64,
}

impl SsrTemporalBuffer {
    /// Create a new temporal buffer.
    pub fn new(width: u32, height: u32, blend_factor: f32) -> Self {
        let count = (width * height) as usize;
        Self {
            width,
            height,
            history: vec![[0.0; 4]; count],
            prev_motion: vec![[0.0; 2]; count],
            blend_factor,
            frame: 0,
        }
    }

    /// Resize the buffer.
    pub fn resize(&mut self, width: u32, height: u32) {
        if self.width != width || self.height != height {
            self.width = width;
            self.height = height;
            let count = (width * height) as usize;
            self.history = vec![[0.0; 4]; count];
            self.prev_motion = vec![[0.0; 2]; count];
            self.frame = 0;
        }
    }

    /// Blend the current frame's SSR result with the temporal history.
    ///
    /// # Arguments
    /// * `current` — Current frame's SSR colour for this pixel.
    /// * `x`, `y` — Pixel coordinates.
    /// * `motion` — Motion vector at this pixel (screen UV offset).
    pub fn accumulate(
        &mut self,
        current: [f32; 4],
        x: u32,
        y: u32,
        motion: [f32; 2],
    ) -> [f32; 4] {
        let idx = (y * self.width + x) as usize;
        if idx >= self.history.len() {
            return current;
        }

        // Reproject to find the history pixel.
        let prev_u = (x as f32 / self.width as f32) - motion[0];
        let prev_v = (y as f32 / self.height as f32) - motion[1];

        let prev_x = (prev_u * self.width as f32).clamp(0.0, self.width as f32 - 1.0) as u32;
        let prev_y = (prev_v * self.height as f32).clamp(0.0, self.height as f32 - 1.0) as u32;
        let prev_idx = (prev_y * self.width + prev_x) as usize;

        let history = if prev_idx < self.history.len() && self.frame > 0 {
            self.history[prev_idx]
        } else {
            current
        };

        // Neighbourhood clamping to avoid ghosting.
        let blended = [
            lerp(current[0], history[0], self.blend_factor),
            lerp(current[1], history[1], self.blend_factor),
            lerp(current[2], history[2], self.blend_factor),
            lerp(current[3], history[3], self.blend_factor),
        ];

        self.history[idx] = blended;
        self.prev_motion[idx] = motion;

        blended
    }

    /// Mark the end of a frame.
    pub fn end_frame(&mut self) {
        self.frame += 1;
    }
}

// ---------------------------------------------------------------------------
// Real-time cubemap capture
// ---------------------------------------------------------------------------

/// Schedule for real-time cubemap capture.
#[derive(Debug, Clone)]
pub struct CubemapCaptureSchedule {
    /// Probes to capture this frame.
    pub pending_probes: Vec<u32>,
    /// Which face to capture next for each probe.
    pub face_index: Vec<u32>,
    /// Maximum number of faces to capture per frame (across all probes).
    pub max_faces_per_frame: u32,
    /// Total budget in milliseconds for probe capture.
    pub budget_ms: f32,
}

impl CubemapCaptureSchedule {
    /// Create a new capture schedule.
    pub fn new(max_faces_per_frame: u32) -> Self {
        Self {
            pending_probes: Vec::new(),
            face_index: Vec::new(),
            max_faces_per_frame,
            budget_ms: 2.0,
        }
    }

    /// Schedule probes for capture based on importance and update intervals.
    pub fn schedule(&mut self, probes: &mut [ReflectionProbe], current_frame: u64) {
        self.pending_probes.clear();
        self.face_index.clear();

        // Collect probes that need updating, sorted by importance.
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for (i, probe) in probes.iter().enumerate() {
            if probe.needs_update(current_frame) {
                candidates.push((i, probe.importance));
            }
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut faces_remaining = self.max_faces_per_frame;
        for (idx, _) in candidates {
            if faces_remaining == 0 {
                break;
            }

            self.pending_probes.push(probes[idx].id);
            self.face_index.push(0); // Start from face 0.
            faces_remaining = faces_remaining.saturating_sub(6);

            probes[idx].last_update_frame = current_frame;
        }
    }

    /// Get the camera directions for the 6 cubemap faces.
    pub fn face_directions() -> [([f32; 3], [f32; 3]); 6] {
        [
            ([1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),   // +X
            ([-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),   // -X
            ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),      // +Y
            ([0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),    // -Y
            ([0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),     // +Z
            ([0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),    // -Z
        ]
    }
}

// ---------------------------------------------------------------------------
// Roughness-based mip selection
// ---------------------------------------------------------------------------

/// Compute the cubemap mip level for a given roughness.
///
/// Uses the GGX roughness-to-mip mapping common in PBR renderers.
///
/// # Arguments
/// * `roughness` — Perceptual roughness [0, 1].
/// * `max_mip` — Maximum mip level of the cubemap.
pub fn roughness_to_mip(roughness: f32, max_mip: u32) -> f32 {
    // Perceptual roughness to linear roughness.
    let alpha = roughness * roughness;
    // Map to mip level (empirical fit matching the GGX lobe width).
    let mip = (alpha * max_mip as f32).sqrt();
    mip.clamp(0.0, max_mip as f32)
}

/// Alternative mip selection using the log2 of the solid angle ratio.
pub fn roughness_to_mip_log(roughness: f32, max_mip: u32) -> f32 {
    let alpha = roughness * roughness;
    if alpha < 1e-6 {
        return 0.0;
    }
    let mip = 0.5 * (alpha * (4.0 * PI)).log2() + 1.0;
    mip.clamp(0.0, max_mip as f32)
}

// ---------------------------------------------------------------------------
// Unified reflection evaluation
// ---------------------------------------------------------------------------

/// Reflection method priorities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReflectionMethod {
    /// Screen-space reflections.
    Ssr = 0,
    /// Planar reflection (for flat surfaces).
    Planar = 1,
    /// Reflection probe cubemap.
    Probe = 2,
    /// Sky cubemap fallback.
    Sky = 3,
}

/// Result of unified reflection evaluation.
#[derive(Debug, Clone)]
pub struct ReflectionResult {
    /// Reflection colour (linear HDR RGB).
    pub color: [f32; 3],
    /// Confidence/visibility [0, 1]. 0 = no reflection data, 1 = fully trusted.
    pub confidence: f32,
    /// Which method produced this result.
    pub method: ReflectionMethod,
    /// Cubemap mip level used (if applicable).
    pub mip_level: f32,
}

impl ReflectionResult {
    /// No reflection.
    pub fn none() -> Self {
        Self {
            color: [0.0; 3],
            confidence: 0.0,
            method: ReflectionMethod::Sky,
            mip_level: 0.0,
        }
    }

    /// Blend two reflection results.
    pub fn blend(a: &Self, b: &Self) -> Self {
        let total = a.confidence + b.confidence;
        if total <= 0.0 {
            return Self::none();
        }
        let wa = a.confidence / total;
        let wb = b.confidence / total;
        Self {
            color: [
                a.color[0] * wa + b.color[0] * wb,
                a.color[1] * wa + b.color[1] * wb,
                a.color[2] * wa + b.color[2] * wb,
            ],
            confidence: total.min(1.0),
            method: if a.confidence >= b.confidence { a.method } else { b.method },
            mip_level: a.mip_level * wa + b.mip_level * wb,
        }
    }
}

/// Unified reflection system configuration.
#[derive(Debug, Clone)]
pub struct ReflectionSystemConfig {
    /// Enable SSR.
    pub ssr_enabled: bool,
    /// SSR parameters.
    pub ssr: SsrConfig,
    /// Enable planar reflections.
    pub planar_enabled: bool,
    /// Maximum number of planar reflections per frame.
    pub max_planar_reflections: u32,
    /// Enable reflection probes.
    pub probes_enabled: bool,
    /// Maximum number of probes to blend per pixel.
    pub max_probe_blend: u32,
    /// Enable real-time probe updates.
    pub realtime_probes: bool,
    /// Cubemap capture schedule.
    pub capture_schedule: CubemapCaptureSchedule,
    /// Enable temporal filtering for SSR.
    pub temporal_enabled: bool,
    /// Fresnel effect intensity.
    pub fresnel_intensity: f32,
    /// Global reflection intensity multiplier.
    pub intensity: f32,
}

impl Default for ReflectionSystemConfig {
    fn default() -> Self {
        Self {
            ssr_enabled: true,
            ssr: SsrConfig::default(),
            planar_enabled: true,
            max_planar_reflections: 4,
            probes_enabled: true,
            max_probe_blend: 2,
            realtime_probes: true,
            capture_schedule: CubemapCaptureSchedule::new(6),
            temporal_enabled: true,
            fresnel_intensity: 1.0,
            intensity: 1.0,
        }
    }
}

/// Compute Fresnel reflectance (Schlick approximation).
///
/// # Arguments
/// * `n_dot_v` — Dot product of surface normal and view direction.
/// * `f0` — Reflectance at normal incidence.
pub fn fresnel_schlick(n_dot_v: f32, f0: f32) -> f32 {
    let one_minus = (1.0 - n_dot_v).max(0.0);
    let one_minus_5 = one_minus * one_minus * one_minus * one_minus * one_minus;
    f0 + (1.0 - f0) * one_minus_5
}

/// Compute Fresnel for a 3-component F0 (metallic workflows).
pub fn fresnel_schlick3(n_dot_v: f32, f0: [f32; 3]) -> [f32; 3] {
    let one_minus = (1.0 - n_dot_v).max(0.0);
    let one_minus_5 = one_minus * one_minus * one_minus * one_minus * one_minus;
    [
        f0[0] + (1.0 - f0[0]) * one_minus_5,
        f0[1] + (1.0 - f0[1]) * one_minus_5,
        f0[2] + (1.0 - f0[2]) * one_minus_5,
    ]
}

/// Compute Fresnel with roughness attenuation (Schlick-Roughness).
pub fn fresnel_schlick_roughness(n_dot_v: f32, f0: f32, roughness: f32) -> f32 {
    let one_minus = (1.0 - n_dot_v).max(0.0);
    let one_minus_5 = one_minus * one_minus * one_minus * one_minus * one_minus;
    let max_r = (1.0 - roughness).max(f0);
    f0 + (max_r - f0) * one_minus_5
}

// ---------------------------------------------------------------------------
// GPU uniform data
// ---------------------------------------------------------------------------

/// Packed reflection probe data for GPU upload.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ProbeGpuData {
    /// Position (xyz) + influence_radius (w).
    pub position_influence: [f32; 4],
    /// Extents (xyz) + blend_distance (w).
    pub extents_blend: [f32; 4],
    /// Cubemap handle (x), mip_count (y), projection type (z: 0=box, 1=sphere), priority (w).
    pub params: [f32; 4],
}

impl ProbeGpuData {
    /// Convert a probe to GPU-uploadable data.
    pub fn from_probe(probe: &ReflectionProbe) -> Self {
        Self {
            position_influence: [
                probe.position[0],
                probe.position[1],
                probe.position[2],
                probe.influence_radius,
            ],
            extents_blend: [
                probe.extents[0],
                probe.extents[1],
                probe.extents[2],
                probe.blend_distance,
            ],
            params: [
                probe.cubemap_handle as f32,
                probe.mip_count as f32,
                if probe.projection == ProbeProjectionType::Box { 0.0 } else { 1.0 },
                probe.priority as f32,
            ],
        }
    }
}

/// Packed SSR configuration for GPU upload.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SsrGpuConfig {
    /// max_steps (x), max_distance (y), stride (z), stride_cutoff (w).
    pub params0: [f32; 4],
    /// refinement_steps (x), thickness (y), jitter (z), roughness_threshold (w).
    pub params1: [f32; 4],
    /// edge_fade (x), temporal_blend (y), resolution_scale (z), min_roughness (w).
    pub params2: [f32; 4],
}

impl SsrGpuConfig {
    pub fn from_config(config: &SsrConfig) -> Self {
        Self {
            params0: [
                config.max_steps as f32,
                config.max_distance,
                config.stride,
                config.stride_cutoff,
            ],
            params1: [
                config.refinement_steps as f32,
                config.thickness,
                config.jitter,
                config.roughness_threshold,
            ],
            params2: [
                config.edge_fade,
                config.temporal_blend,
                config.resolution_scale,
                config.min_roughness,
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn length3(v: [f32; 3]) -> f32 {
    dot3(v, v).sqrt()
}

#[inline]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = length3(v);
    if len < 1e-12 {
        [0.0; 3]
    } else {
        let inv = 1.0 / len;
        [v[0] * inv, v[1] * inv, v[2] * inv]
    }
}

#[inline]
fn distance_to_probe(a: [f32; 3], b: [f32; 3]) -> f32 {
    length3([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Project a view-space position to screen UV [0, 1].
fn project_to_screen(view_pos: [f32; 3], projection: &[f32; 16]) -> [f32; 2] {
    // Multiply by projection matrix (column-major).
    let clip_x = projection[0] * view_pos[0] + projection[4] * view_pos[1]
        + projection[8] * view_pos[2] + projection[12];
    let clip_y = projection[1] * view_pos[0] + projection[5] * view_pos[1]
        + projection[9] * view_pos[2] + projection[13];
    let clip_w = projection[3] * view_pos[0] + projection[7] * view_pos[1]
        + projection[11] * view_pos[2] + projection[15];

    if clip_w.abs() < 1e-10 {
        return [0.5, 0.5];
    }

    let ndc_x = clip_x / clip_w;
    let ndc_y = clip_y / clip_w;

    [ndc_x * 0.5 + 0.5, ndc_y * 0.5 + 0.5]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_weight() {
        let probe = ReflectionProbe {
            influence_radius: 10.0,
            blend_distance: 2.0,
            position: [0.0, 0.0, 0.0],
            ..ReflectionProbe::default()
        };

        // At centre.
        assert!((probe.compute_weight([0.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);
        // Outside.
        assert!(probe.compute_weight([20.0, 0.0, 0.0]) <= 0.0);
        // On the edge.
        let w = probe.compute_weight([9.0, 0.0, 0.0]);
        assert!(w > 0.0 && w < 1.0);
    }

    #[test]
    fn test_box_projection() {
        let probe = ReflectionProbe {
            position: [0.0, 0.0, 0.0],
            extents: [5.0, 5.0, 5.0],
            projection: ProbeProjectionType::Box,
            ..ReflectionProbe::default()
        };

        let dir = probe.box_project([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!((dir[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fresnel() {
        // At normal incidence, should return F0.
        let f = fresnel_schlick(1.0, 0.04);
        assert!((f - 0.04).abs() < 1e-6);

        // At grazing angle, should approach 1.
        let f = fresnel_schlick(0.0, 0.04);
        assert!((f - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_roughness_mip() {
        let mip = roughness_to_mip(0.0, 7);
        assert!(mip < 0.01);

        let mip = roughness_to_mip(1.0, 7);
        assert!(mip > 2.0);
    }

    #[test]
    fn test_reflection_blend() {
        let a = ReflectionResult {
            color: [1.0, 0.0, 0.0],
            confidence: 0.5,
            method: ReflectionMethod::Ssr,
            mip_level: 0.0,
        };
        let b = ReflectionResult {
            color: [0.0, 0.0, 1.0],
            confidence: 0.5,
            method: ReflectionMethod::Probe,
            mip_level: 3.0,
        };
        let blended = ReflectionResult::blend(&a, &b);
        assert!((blended.color[0] - 0.5).abs() < 1e-6);
        assert!((blended.color[2] - 0.5).abs() < 1e-6);
    }
}
