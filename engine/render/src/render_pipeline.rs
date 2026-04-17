// engine/render/src/render_pipeline.rs
//
// The master rendering pipeline for the Genovo engine. Orchestrates all
// rendering subsystems -- shadow mapping, depth prepass, G-Buffer fill,
// SSAO, lighting resolve, SSR, transparent objects, particles, VFX,
// post-processing, and UI overlay -- into a coherent frame.
//
// The pipeline is configurable via `RenderFeatureFlags` to enable or
// disable individual passes (e.g. for quality scaling or platform
// limitations).
//
// # Key types
//
// - `RenderPipeline` -- the master renderer.
// - `RenderView` -- camera + viewport + render target.
// - `DrawItem` -- a single thing to draw (mesh + material + transform).
// - `RenderableGatherer` -- collects `DrawItem`s from the ECS world.
// - `FrustumCuller` -- tests renderables against the view frustum.
// - `RenderStats` -- timing and counter statistics.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Feature flags
// ---------------------------------------------------------------------------

/// Flags that control which render passes are active.
#[derive(Debug, Clone, Copy)]
pub struct RenderFeatureFlags {
    pub enable_shadows: bool,
    pub enable_cascaded_shadows: bool,
    pub shadow_cascade_count: u32,
    pub shadow_map_resolution: u32,
    pub enable_depth_prepass: bool,
    pub enable_gbuffer: bool,
    pub enable_ssao: bool,
    pub ssao_sample_count: u32,
    pub ssao_radius: f32,
    pub enable_ssr: bool,
    pub ssr_max_steps: u32,
    pub enable_bloom: bool,
    pub bloom_threshold: f32,
    pub bloom_intensity: f32,
    pub enable_tonemapping: bool,
    pub tonemapping_operator: TonemappingOperator,
    pub enable_fxaa: bool,
    pub enable_taa: bool,
    pub enable_motion_blur: bool,
    pub motion_blur_samples: u32,
    pub enable_dof: bool,
    pub dof_focal_distance: f32,
    pub dof_focal_range: f32,
    pub enable_chromatic_aberration: bool,
    pub enable_vignette: bool,
    pub enable_film_grain: bool,
    pub enable_volumetric_fog: bool,
    pub enable_particles: bool,
    pub enable_decals: bool,
    pub enable_ui_overlay: bool,
    pub max_render_distance: f32,
    pub enable_frustum_culling: bool,
    pub enable_occlusion_culling: bool,
    pub enable_lod: bool,
}

impl Default for RenderFeatureFlags {
    fn default() -> Self {
        Self {
            enable_shadows: true,
            enable_cascaded_shadows: true,
            shadow_cascade_count: 4,
            shadow_map_resolution: 2048,
            enable_depth_prepass: true,
            enable_gbuffer: true,
            enable_ssao: true,
            ssao_sample_count: 16,
            ssao_radius: 0.5,
            enable_ssr: false,
            ssr_max_steps: 64,
            enable_bloom: true,
            bloom_threshold: 1.0,
            bloom_intensity: 0.3,
            enable_tonemapping: true,
            tonemapping_operator: TonemappingOperator::Aces,
            enable_fxaa: true,
            enable_taa: false,
            enable_motion_blur: false,
            motion_blur_samples: 8,
            enable_dof: false,
            dof_focal_distance: 10.0,
            dof_focal_range: 5.0,
            enable_chromatic_aberration: false,
            enable_vignette: true,
            enable_film_grain: false,
            enable_volumetric_fog: false,
            enable_particles: true,
            enable_decals: true,
            enable_ui_overlay: true,
            max_render_distance: 1000.0,
            enable_frustum_culling: true,
            enable_occlusion_culling: false,
            enable_lod: true,
        }
    }
}

impl RenderFeatureFlags {
    /// High quality preset.
    pub fn high_quality() -> Self {
        Self {
            enable_ssr: true,
            enable_taa: true,
            enable_motion_blur: true,
            enable_dof: true,
            enable_volumetric_fog: true,
            shadow_cascade_count: 4,
            shadow_map_resolution: 4096,
            ssao_sample_count: 32,
            ..Default::default()
        }
    }

    /// Medium quality preset.
    pub fn medium_quality() -> Self {
        Self {
            shadow_cascade_count: 3,
            shadow_map_resolution: 2048,
            ssao_sample_count: 16,
            ..Default::default()
        }
    }

    /// Low quality preset.
    pub fn low_quality() -> Self {
        Self {
            enable_shadows: true,
            enable_cascaded_shadows: true,
            shadow_cascade_count: 2,
            shadow_map_resolution: 1024,
            enable_depth_prepass: true,
            enable_gbuffer: true,
            enable_ssao: false,
            ssao_sample_count: 8,
            ssao_radius: 0.5,
            enable_ssr: false,
            ssr_max_steps: 32,
            enable_bloom: false,
            bloom_threshold: 1.0,
            bloom_intensity: 0.3,
            enable_tonemapping: true,
            tonemapping_operator: TonemappingOperator::Reinhard,
            enable_fxaa: true,
            enable_taa: false,
            enable_motion_blur: false,
            motion_blur_samples: 4,
            enable_dof: false,
            dof_focal_distance: 10.0,
            dof_focal_range: 5.0,
            enable_chromatic_aberration: false,
            enable_vignette: false,
            enable_film_grain: false,
            enable_volumetric_fog: false,
            enable_particles: true,
            enable_decals: false,
            enable_ui_overlay: true,
            max_render_distance: 500.0,
            enable_frustum_culling: true,
            enable_occlusion_culling: false,
            enable_lod: true,
        }
    }
}

/// Tonemapping operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TonemappingOperator {
    /// Reinhard simple operator.
    Reinhard,
    /// ACES filmic tonemapping.
    Aces,
    /// Uncharted 2 filmic.
    Uncharted2,
    /// AgX tonemap.
    AgX,
    /// No tonemapping (linear).
    Linear,
}

// ---------------------------------------------------------------------------
// RenderView
// ---------------------------------------------------------------------------

/// A render view: camera + viewport + render target.
#[derive(Debug, Clone)]
pub struct RenderView {
    /// Unique identifier.
    pub id: u32,
    /// Human-readable name.
    pub name: String,
    /// View matrix (world -> view, column-major).
    pub view_matrix: [f32; 16],
    /// Projection matrix (column-major).
    pub projection_matrix: [f32; 16],
    /// Combined view-projection matrix.
    pub view_projection: [f32; 16],
    /// Inverse view-projection matrix.
    pub inv_view_projection: [f32; 16],
    /// Previous frame's view-projection (for motion vectors / TAA).
    pub prev_view_projection: [f32; 16],
    /// Camera world position.
    pub camera_position: [f32; 3],
    /// Camera forward direction.
    pub camera_forward: [f32; 3],
    /// Viewport rectangle: x, y, width, height in pixels.
    pub viewport: ViewportRect,
    /// Render target (None = swapchain).
    pub render_target: Option<u64>,
    /// Clear color.
    pub clear_color: [f32; 4],
    /// Near clip plane.
    pub near_clip: f32,
    /// Far clip plane.
    pub far_clip: f32,
    /// Vertical FOV in radians.
    pub fov_y: f32,
    /// Aspect ratio.
    pub aspect_ratio: f32,
    /// TAA jitter offset in pixels.
    pub jitter: [f32; 2],
    /// Priority (higher = rendered first).
    pub priority: i32,
    /// Layer mask.
    pub layer_mask: u32,
}

/// Viewport rectangle in pixels.
#[derive(Debug, Clone, Copy)]
pub struct ViewportRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl ViewportRect {
    pub fn full(width: f32, height: f32) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width,
            height,
        }
    }
}

impl RenderView {
    /// Create a simple render view from position, target, and projection params.
    pub fn new_perspective(
        id: u32,
        position: [f32; 3],
        forward: [f32; 3],
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
        viewport: ViewportRect,
    ) -> Self {
        Self {
            id,
            name: format!("View_{}", id),
            view_matrix: identity_matrix(),
            projection_matrix: identity_matrix(),
            view_projection: identity_matrix(),
            inv_view_projection: identity_matrix(),
            prev_view_projection: identity_matrix(),
            camera_position: position,
            camera_forward: forward,
            viewport,
            render_target: None,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            near_clip: near,
            far_clip: far,
            fov_y,
            aspect_ratio: aspect,
            jitter: [0.0, 0.0],
            priority: 0,
            layer_mask: u32::MAX,
        }
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

// ---------------------------------------------------------------------------
// Draw items and sort keys
// ---------------------------------------------------------------------------

/// Handle to a mesh resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshHandle(pub u64);

/// Handle to a material resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialHandle(pub u64);

/// A single renderable item in the draw list.
#[derive(Debug, Clone)]
pub struct DrawItem {
    /// Mesh to draw.
    pub mesh: MeshHandle,
    /// Material to use.
    pub material: MaterialHandle,
    /// World transform matrix (column-major).
    pub world_matrix: [f32; 16],
    /// Previous frame world matrix (for motion vectors).
    pub prev_world_matrix: [f32; 16],
    /// AABB in world space.
    pub world_aabb_min: [f32; 3],
    pub world_aabb_max: [f32; 3],
    /// Bounding sphere center.
    pub sphere_center: [f32; 3],
    /// Bounding sphere radius.
    pub sphere_radius: f32,
    /// Squared distance from camera.
    pub distance_sq: f32,
    /// Sort key for draw ordering.
    pub sort_key: u64,
    /// LOD level.
    pub lod_level: u32,
    /// Layer mask.
    pub layer_mask: u32,
    /// Whether this item is transparent.
    pub is_transparent: bool,
    /// Whether this item casts shadows.
    pub casts_shadow: bool,
    /// Entity ID (for debugging / picking).
    pub entity_id: u64,
    /// Sub-mesh index within the mesh.
    pub sub_mesh: u32,
    /// Instance count (for instanced drawing).
    pub instance_count: u32,
}

impl DrawItem {
    /// Compute the sort key for opaque objects.
    /// Sort by: pipeline state (material) then front-to-back.
    pub fn compute_opaque_sort_key(&mut self) {
        let material_bits = (self.material.0 & 0xFFFF_FFFF) << 32;
        let depth_bits = {
            let depth = self.distance_sq.sqrt();
            let quantised = (depth * 1000.0) as u32;
            quantised as u64
        };
        self.sort_key = material_bits | depth_bits;
    }

    /// Compute the sort key for transparent objects.
    /// Sort back-to-front.
    pub fn compute_transparent_sort_key(&mut self) {
        let depth = self.distance_sq.sqrt();
        let inv_depth = u32::MAX - (depth * 1000.0) as u32;
        self.sort_key = inv_depth as u64;
    }
}

// ---------------------------------------------------------------------------
// Draw list
// ---------------------------------------------------------------------------

/// A list of draw items, separated into opaque and transparent.
#[derive(Debug, Default)]
pub struct DrawList {
    /// Opaque items (sorted front-to-back by sort key).
    pub opaque: Vec<DrawItem>,
    /// Transparent items (sorted back-to-front by sort key).
    pub transparent: Vec<DrawItem>,
    /// Shadow caster items.
    pub shadow_casters: Vec<DrawItem>,
}

impl DrawList {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sort the draw lists.
    pub fn sort(&mut self) {
        // Opaque: front-to-back (lower sort key first).
        self.opaque.sort_unstable_by_key(|item| item.sort_key);
        // Transparent: back-to-front (lower sort key = farther away).
        self.transparent.sort_unstable_by_key(|item| item.sort_key);
        // Shadow casters: front-to-back.
        self.shadow_casters
            .sort_unstable_by_key(|item| item.sort_key);
    }

    /// Clear all lists.
    pub fn clear(&mut self) {
        self.opaque.clear();
        self.transparent.clear();
        self.shadow_casters.clear();
    }

    /// Total number of draw items.
    pub fn total_items(&self) -> usize {
        self.opaque.len() + self.transparent.len() + self.shadow_casters.len()
    }

    /// Total number of triangles (assuming each draw call is indexed).
    pub fn total_draw_calls(&self) -> u32 {
        (self.opaque.len() + self.transparent.len() + self.shadow_casters.len()) as u32
    }
}

// ---------------------------------------------------------------------------
// Frustum culler
// ---------------------------------------------------------------------------

/// Simple frustum plane for the pipeline culler.
#[derive(Debug, Clone, Copy)]
struct FrustumPlane {
    nx: f32,
    ny: f32,
    nz: f32,
    d: f32,
}

impl FrustumPlane {
    fn from_row_add(r3: [f32; 4], r: [f32; 4]) -> Self {
        let a = r3[0] + r[0];
        let b = r3[1] + r[1];
        let c = r3[2] + r[2];
        let d = r3[3] + r[3];
        let len = (a * a + b * b + c * c).sqrt();
        if len < 1e-7 {
            return Self { nx: 0.0, ny: 0.0, nz: 0.0, d: 0.0 };
        }
        let inv = 1.0 / len;
        Self {
            nx: a * inv,
            ny: b * inv,
            nz: c * inv,
            d: d * inv,
        }
    }

    fn from_row_sub(r3: [f32; 4], r: [f32; 4]) -> Self {
        let a = r3[0] - r[0];
        let b = r3[1] - r[1];
        let c = r3[2] - r[2];
        let d = r3[3] - r[3];
        let len = (a * a + b * b + c * c).sqrt();
        if len < 1e-7 {
            return Self { nx: 0.0, ny: 0.0, nz: 0.0, d: 0.0 };
        }
        let inv = 1.0 / len;
        Self {
            nx: a * inv,
            ny: b * inv,
            nz: c * inv,
            d: d * inv,
        }
    }

    /// Test if an AABB is on the positive side of this plane.
    #[inline]
    fn aabb_visible(&self, min: &[f32; 3], max: &[f32; 3]) -> bool {
        let px = if self.nx >= 0.0 { max[0] } else { min[0] };
        let py = if self.ny >= 0.0 { max[1] } else { min[1] };
        let pz = if self.nz >= 0.0 { max[2] } else { min[2] };
        self.nx * px + self.ny * py + self.nz * pz + self.d >= 0.0
    }
}

/// Frustum culler that tests draw items against the view frustum.
pub struct FrustumCuller {
    planes: [FrustumPlane; 6],
}

impl FrustumCuller {
    /// Extract frustum from a column-major view-projection matrix.
    pub fn from_view_projection(m: &[f32; 16]) -> Self {
        let r0 = [m[0], m[4], m[8], m[12]];
        let r1 = [m[1], m[5], m[9], m[13]];
        let r2 = [m[2], m[6], m[10], m[14]];
        let r3 = [m[3], m[7], m[11], m[15]];

        Self {
            planes: [
                FrustumPlane::from_row_add(r3, r0), // left
                FrustumPlane::from_row_sub(r3, r0), // right
                FrustumPlane::from_row_add(r3, r1), // bottom
                FrustumPlane::from_row_sub(r3, r1), // top
                FrustumPlane::from_row_add(r3, r2), // near
                FrustumPlane::from_row_sub(r3, r2), // far
            ],
        }
    }

    /// Test if an AABB is visible (at least partially inside the frustum).
    #[inline]
    pub fn is_visible(&self, min: &[f32; 3], max: &[f32; 3]) -> bool {
        for plane in &self.planes {
            if !plane.aabb_visible(min, max) {
                return false;
            }
        }
        true
    }

    /// Test if a sphere is visible.
    #[inline]
    pub fn is_sphere_visible(&self, center: &[f32; 3], radius: f32) -> bool {
        for plane in &self.planes {
            let dist = plane.nx * center[0] + plane.ny * center[1] + plane.nz * center[2] + plane.d;
            if dist < -radius {
                return false;
            }
        }
        true
    }

    /// Cull a batch of draw items, returning indices of visible ones.
    pub fn cull_draw_items(&self, items: &[DrawItem]) -> Vec<usize> {
        items
            .iter()
            .enumerate()
            .filter(|(_, item)| {
                self.is_visible(&item.world_aabb_min, &item.world_aabb_max)
            })
            .map(|(i, _)| i)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Renderable gatherer
// ---------------------------------------------------------------------------

/// A renderable entity from the ECS world.
#[derive(Debug, Clone)]
pub struct RenderableComponent {
    pub entity_id: u64,
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub world_matrix: [f32; 16],
    pub prev_world_matrix: [f32; 16],
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub is_transparent: bool,
    pub casts_shadow: bool,
    pub layer_mask: u32,
    pub lod_distances: Vec<f32>,
}

/// Gathers renderable entities from the world and builds draw lists.
pub struct RenderableGatherer {
    /// Cached draw items from last gather.
    draw_list: DrawList,
}

impl RenderableGatherer {
    pub fn new() -> Self {
        Self {
            draw_list: DrawList::new(),
        }
    }

    /// Gather all renderables and build a draw list.
    pub fn gather(
        &mut self,
        renderables: &[RenderableComponent],
        view: &RenderView,
        features: &RenderFeatureFlags,
    ) -> &DrawList {
        self.draw_list.clear();

        let camera_pos = view.camera_position;
        let culler = FrustumCuller::from_view_projection(&view.view_projection);

        for renderable in renderables {
            // Layer mask check.
            if (renderable.layer_mask & view.layer_mask) == 0 {
                continue;
            }

            // Compute distance.
            let cx = (renderable.aabb_min[0] + renderable.aabb_max[0]) * 0.5;
            let cy = (renderable.aabb_min[1] + renderable.aabb_max[1]) * 0.5;
            let cz = (renderable.aabb_min[2] + renderable.aabb_max[2]) * 0.5;
            let dx = cx - camera_pos[0];
            let dy = cy - camera_pos[1];
            let dz = cz - camera_pos[2];
            let distance_sq = dx * dx + dy * dy + dz * dz;
            let distance = distance_sq.sqrt();

            // Distance culling.
            if features.enable_frustum_culling && distance > features.max_render_distance {
                continue;
            }

            // Frustum culling.
            if features.enable_frustum_culling
                && !culler.is_visible(&renderable.aabb_min, &renderable.aabb_max)
            {
                continue;
            }

            // LOD selection.
            let lod_level = if features.enable_lod {
                let mut lod = 0u32;
                for (i, &threshold) in renderable.lod_distances.iter().enumerate() {
                    if distance > threshold {
                        lod = (i + 1) as u32;
                    }
                }
                lod
            } else {
                0
            };

            // Compute bounding sphere.
            let half_x = (renderable.aabb_max[0] - renderable.aabb_min[0]) * 0.5;
            let half_y = (renderable.aabb_max[1] - renderable.aabb_min[1]) * 0.5;
            let half_z = (renderable.aabb_max[2] - renderable.aabb_min[2]) * 0.5;
            let radius = (half_x * half_x + half_y * half_y + half_z * half_z).sqrt();

            let mut item = DrawItem {
                mesh: renderable.mesh,
                material: renderable.material,
                world_matrix: renderable.world_matrix,
                prev_world_matrix: renderable.prev_world_matrix,
                world_aabb_min: renderable.aabb_min,
                world_aabb_max: renderable.aabb_max,
                sphere_center: [cx, cy, cz],
                sphere_radius: radius,
                distance_sq,
                sort_key: 0,
                lod_level,
                layer_mask: renderable.layer_mask,
                is_transparent: renderable.is_transparent,
                casts_shadow: renderable.casts_shadow,
                entity_id: renderable.entity_id,
                sub_mesh: 0,
                instance_count: 1,
            };

            if item.is_transparent {
                item.compute_transparent_sort_key();
                self.draw_list.transparent.push(item);
            } else {
                item.compute_opaque_sort_key();
                self.draw_list.opaque.push(item.clone());
                if item.casts_shadow && features.enable_shadows {
                    self.draw_list.shadow_casters.push(item);
                }
            }
        }

        self.draw_list.sort();
        &self.draw_list
    }
}

impl Default for RenderableGatherer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Cascade shadow map data
// ---------------------------------------------------------------------------

/// Data for a single shadow cascade.
#[derive(Debug, Clone)]
pub struct ShadowCascade {
    /// Cascade index (0 = closest).
    pub index: u32,
    /// Near split distance.
    pub near_distance: f32,
    /// Far split distance.
    pub far_distance: f32,
    /// Light view-projection matrix for this cascade.
    pub light_vp: [f32; 16],
    /// Viewport within the shadow atlas.
    pub atlas_viewport: ViewportRect,
    /// Texel size (for PCF / bias computation).
    pub texel_size: f32,
    /// Depth bias.
    pub depth_bias: f32,
    /// Normal bias.
    pub normal_bias: f32,
}

/// Compute cascade split distances using the practical split scheme
/// (logarithmic / linear blend).
pub fn compute_cascade_splits(
    near: f32,
    far: f32,
    cascade_count: u32,
    lambda: f32,
) -> Vec<f32> {
    let mut splits = Vec::with_capacity(cascade_count as usize + 1);
    splits.push(near);

    for i in 1..=cascade_count {
        let p = i as f32 / cascade_count as f32;
        let log_split = near * (far / near).powf(p);
        let linear_split = near + (far - near) * p;
        let split = lambda * log_split + (1.0 - lambda) * linear_split;
        splits.push(split);
    }

    splits
}

/// Shadow configuration for a directional light.
#[derive(Debug, Clone)]
pub struct DirectionalShadowConfig {
    /// Light direction (normalised).
    pub light_direction: [f32; 3],
    /// Shadow map resolution per cascade.
    pub resolution: u32,
    /// Number of cascades.
    pub cascade_count: u32,
    /// Cascade split lambda (0 = linear, 1 = logarithmic).
    pub split_lambda: f32,
    /// Maximum shadow distance.
    pub max_distance: f32,
    /// Depth bias per cascade.
    pub depth_bias: f32,
    /// Normal bias per cascade.
    pub normal_bias: f32,
}

impl Default for DirectionalShadowConfig {
    fn default() -> Self {
        Self {
            light_direction: [0.0, -1.0, 0.0],
            resolution: 2048,
            cascade_count: 4,
            split_lambda: 0.75,
            max_distance: 200.0,
            depth_bias: 0.005,
            normal_bias: 0.02,
        }
    }
}

/// Build cascade data for a directional light.
pub fn build_cascades(
    config: &DirectionalShadowConfig,
    view: &RenderView,
) -> Vec<ShadowCascade> {
    let splits = compute_cascade_splits(
        view.near_clip,
        config.max_distance.min(view.far_clip),
        config.cascade_count,
        config.split_lambda,
    );

    let mut cascades = Vec::with_capacity(config.cascade_count as usize);

    for i in 0..config.cascade_count as usize {
        let near = splits[i];
        let far = splits[i + 1];

        // In a real engine, we would compute the light VP matrix from the
        // frustum sub-volume. Here we store placeholder identity matrices.
        cascades.push(ShadowCascade {
            index: i as u32,
            near_distance: near,
            far_distance: far,
            light_vp: identity_matrix(),
            atlas_viewport: ViewportRect {
                x: 0.0,
                y: (i as f32) * config.resolution as f32,
                width: config.resolution as f32,
                height: config.resolution as f32,
            },
            texel_size: 1.0 / config.resolution as f32,
            depth_bias: config.depth_bias,
            normal_bias: config.normal_bias,
        });
    }

    cascades
}

// ---------------------------------------------------------------------------
// Render stats
// ---------------------------------------------------------------------------

/// Per-pass timing and counter stats.
#[derive(Debug, Clone, Default)]
pub struct PassStats {
    pub name: String,
    pub gpu_time_ms: f64,
    pub cpu_time_ms: f64,
    pub draw_calls: u32,
    pub triangles: u32,
    pub state_changes: u32,
}

/// Overall render statistics for a frame.
#[derive(Debug, Clone, Default)]
pub struct RenderStats {
    pub frame_index: u64,
    pub total_gpu_time_ms: f64,
    pub total_cpu_time_ms: f64,
    pub total_draw_calls: u32,
    pub total_triangles: u32,
    pub total_state_changes: u32,
    pub visible_objects: u32,
    pub culled_objects: u32,
    pub shadow_casters: u32,
    pub transparent_objects: u32,
    pub opaque_objects: u32,
    pub pass_stats: Vec<PassStats>,
    pub gpu_memory_used_mb: f32,
    pub gpu_memory_budget_mb: f32,
    pub vertices_processed: u64,
    pub pixels_shaded: u64,
}

impl RenderStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a pass stat.
    pub fn add_pass(&mut self, pass: PassStats) {
        self.total_gpu_time_ms += pass.gpu_time_ms;
        self.total_cpu_time_ms += pass.cpu_time_ms;
        self.total_draw_calls += pass.draw_calls;
        self.total_triangles += pass.triangles;
        self.total_state_changes += pass.state_changes;
        self.pass_stats.push(pass);
    }
}

impl fmt::Display for RenderStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Render Stats (Frame {}) ===", self.frame_index)?;
        writeln!(f, "  GPU: {:.2} ms, CPU: {:.2} ms", self.total_gpu_time_ms, self.total_cpu_time_ms)?;
        writeln!(f, "  Draw calls: {}, Triangles: {}", self.total_draw_calls, self.total_triangles)?;
        writeln!(f, "  State changes: {}", self.total_state_changes)?;
        writeln!(
            f,
            "  Objects: {} visible, {} culled, {} shadow, {} opaque, {} transparent",
            self.visible_objects,
            self.culled_objects,
            self.shadow_casters,
            self.opaque_objects,
            self.transparent_objects,
        )?;
        writeln!(
            f,
            "  GPU Memory: {:.1}/{:.1} MB",
            self.gpu_memory_used_mb, self.gpu_memory_budget_mb
        )?;
        for pass in &self.pass_stats {
            writeln!(
                f,
                "    [{}] GPU: {:.2} ms, CPU: {:.2} ms, {} draws, {} tris",
                pass.name, pass.gpu_time_ms, pass.cpu_time_ms, pass.draw_calls, pass.triangles,
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Post-processing chain
// ---------------------------------------------------------------------------

/// A single post-processing effect.
#[derive(Debug, Clone)]
pub struct PostProcessEffect {
    pub name: String,
    pub enabled: bool,
    pub order: i32,
    pub parameters: HashMap<String, PostProcessParam>,
}

/// Parameter value for a post-process effect.
#[derive(Debug, Clone)]
pub enum PostProcessParam {
    Float(f32),
    Int(i32),
    Bool(bool),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Color([f32; 4]),
}

/// The post-processing chain.
#[derive(Debug, Clone)]
pub struct PostProcessChain {
    pub effects: Vec<PostProcessEffect>,
}

impl PostProcessChain {
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
        }
    }

    /// Add an effect.
    pub fn add(&mut self, effect: PostProcessEffect) {
        self.effects.push(effect);
        self.effects.sort_by_key(|e| e.order);
    }

    /// Get a mutable effect by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut PostProcessEffect> {
        self.effects.iter_mut().find(|e| e.name == name)
    }

    /// Build the default post-process chain based on feature flags.
    pub fn from_features(features: &RenderFeatureFlags) -> Self {
        let mut chain = Self::new();
        let mut order = 0;

        if features.enable_bloom {
            let mut params = HashMap::new();
            params.insert("threshold".into(), PostProcessParam::Float(features.bloom_threshold));
            params.insert("intensity".into(), PostProcessParam::Float(features.bloom_intensity));
            chain.add(PostProcessEffect {
                name: "Bloom".into(),
                enabled: true,
                order,
                parameters: params,
            });
            order += 10;
        }

        if features.enable_motion_blur {
            let mut params = HashMap::new();
            params.insert(
                "samples".into(),
                PostProcessParam::Int(features.motion_blur_samples as i32),
            );
            chain.add(PostProcessEffect {
                name: "MotionBlur".into(),
                enabled: true,
                order,
                parameters: params,
            });
            order += 10;
        }

        if features.enable_dof {
            let mut params = HashMap::new();
            params.insert(
                "focal_distance".into(),
                PostProcessParam::Float(features.dof_focal_distance),
            );
            params.insert(
                "focal_range".into(),
                PostProcessParam::Float(features.dof_focal_range),
            );
            chain.add(PostProcessEffect {
                name: "DepthOfField".into(),
                enabled: true,
                order,
                parameters: params,
            });
            order += 10;
        }

        if features.enable_tonemapping {
            let mut params = HashMap::new();
            params.insert(
                "operator".into(),
                PostProcessParam::Int(features.tonemapping_operator as i32),
            );
            chain.add(PostProcessEffect {
                name: "Tonemapping".into(),
                enabled: true,
                order,
                parameters: params,
            });
            order += 10;
        }

        if features.enable_chromatic_aberration {
            chain.add(PostProcessEffect {
                name: "ChromaticAberration".into(),
                enabled: true,
                order,
                parameters: HashMap::new(),
            });
            order += 10;
        }

        if features.enable_film_grain {
            chain.add(PostProcessEffect {
                name: "FilmGrain".into(),
                enabled: true,
                order,
                parameters: HashMap::new(),
            });
            order += 10;
        }

        if features.enable_vignette {
            chain.add(PostProcessEffect {
                name: "Vignette".into(),
                enabled: true,
                order,
                parameters: HashMap::new(),
            });
            order += 10;
        }

        if features.enable_fxaa {
            chain.add(PostProcessEffect {
                name: "FXAA".into(),
                enabled: true,
                order,
                parameters: HashMap::new(),
            });
            // order += 10;
        }

        chain
    }
}

impl Default for PostProcessChain {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RenderPipeline
// ---------------------------------------------------------------------------

/// The master rendering pipeline.
///
/// This orchestrates the entire frame rendering process:
///
/// 1. **Gather**: Collect renderables from the world, cull, sort.
/// 2. **Shadow pass**: Render cascaded shadow maps for each light.
/// 3. **Depth prepass**: Render opaque geometry depth-only.
/// 4. **G-Buffer fill**: Render deferred geometry to MRT.
/// 5. **SSAO**: Compute screen-space ambient occlusion.
/// 6. **Lighting resolve**: Evaluate all lights (clustered/deferred).
/// 7. **SSR**: Screen-space reflections.
/// 8. **Transparent pass**: Forward-render transparent objects (sorted).
/// 9. **Particles and VFX**: Render particle systems.
/// 10. **Post-processing**: Bloom, tonemapping, FXAA/TAA, etc.
/// 11. **UI overlay**: Render UI elements on top.
/// 12. **Present**: Submit to swapchain.
pub struct RenderPipeline {
    /// Feature flags.
    pub features: RenderFeatureFlags,
    /// Renderable gatherer.
    gatherer: RenderableGatherer,
    /// Cached draw list.
    draw_list: DrawList,
    /// Post-processing chain.
    post_process_chain: PostProcessChain,
    /// Shadow configuration.
    shadow_config: DirectionalShadowConfig,
    /// Current frame stats.
    stats: RenderStats,
    /// Frame counter.
    frame_index: u64,
    /// Cached cascade data.
    cascades: Vec<ShadowCascade>,
    /// Active render views.
    views: Vec<RenderView>,
}

impl RenderPipeline {
    /// Create a new render pipeline with default features.
    pub fn new() -> Self {
        let features = RenderFeatureFlags::default();
        let post_process_chain = PostProcessChain::from_features(&features);
        Self {
            features,
            gatherer: RenderableGatherer::new(),
            draw_list: DrawList::new(),
            post_process_chain,
            shadow_config: DirectionalShadowConfig::default(),
            stats: RenderStats::new(),
            frame_index: 0,
            cascades: Vec::new(),
            views: Vec::new(),
        }
    }

    /// Create with custom feature flags.
    pub fn with_features(features: RenderFeatureFlags) -> Self {
        let post_process_chain = PostProcessChain::from_features(&features);
        Self {
            features,
            gatherer: RenderableGatherer::new(),
            draw_list: DrawList::new(),
            post_process_chain,
            shadow_config: DirectionalShadowConfig::default(),
            stats: RenderStats::new(),
            frame_index: 0,
            cascades: Vec::new(),
            views: Vec::new(),
        }
    }

    /// Set the shadow configuration.
    pub fn set_shadow_config(&mut self, config: DirectionalShadowConfig) {
        self.shadow_config = config;
    }

    /// Set the feature flags.
    pub fn set_features(&mut self, features: RenderFeatureFlags) {
        self.features = features;
        self.post_process_chain = PostProcessChain::from_features(&features);
    }

    /// Add a render view.
    pub fn add_view(&mut self, view: RenderView) {
        self.views.push(view);
        self.views.sort_by_key(|v| std::cmp::Reverse(v.priority));
    }

    /// Clear all views.
    pub fn clear_views(&mut self) {
        self.views.clear();
    }

    /// Get the post-process chain.
    pub fn post_process_chain(&self) -> &PostProcessChain {
        &self.post_process_chain
    }

    /// Get a mutable reference to the post-process chain.
    pub fn post_process_chain_mut(&mut self) -> &mut PostProcessChain {
        &mut self.post_process_chain
    }

    // -----------------------------------------------------------------------
    // Frame lifecycle
    // -----------------------------------------------------------------------

    /// Prepare a frame: gather renderables, cull, sort.
    ///
    /// Call this once per frame before `render_frame`.
    pub fn prepare_frame(
        &mut self,
        renderables: &[RenderableComponent],
        primary_view: &RenderView,
    ) {
        self.stats = RenderStats::new();
        self.stats.frame_index = self.frame_index;
        self.frame_index += 1;

        // Gather and cull.
        let draw_list = self.gatherer.gather(renderables, primary_view, &self.features);

        // Copy draw list (in a real engine we'd move/swap instead).
        self.draw_list.opaque = draw_list.opaque.clone();
        self.draw_list.transparent = draw_list.transparent.clone();
        self.draw_list.shadow_casters = draw_list.shadow_casters.clone();

        // Update stats.
        self.stats.opaque_objects = self.draw_list.opaque.len() as u32;
        self.stats.transparent_objects = self.draw_list.transparent.len() as u32;
        self.stats.shadow_casters = self.draw_list.shadow_casters.len() as u32;
        self.stats.visible_objects =
            self.stats.opaque_objects + self.stats.transparent_objects;
        self.stats.culled_objects =
            renderables.len() as u32 - self.stats.visible_objects;

        // Build cascade data.
        if self.features.enable_shadows && self.features.enable_cascaded_shadows {
            self.cascades = build_cascades(&self.shadow_config, primary_view);
        }
    }

    /// Execute the full rendering pipeline.
    ///
    /// In a real engine, this would record GPU commands. Here it simulates
    /// the pipeline stages and collects statistics.
    pub fn render_frame(&mut self, primary_view: &RenderView) {
        // 1. Shadow pass.
        if self.features.enable_shadows {
            self.execute_shadow_pass(primary_view);
        }

        // 2. Depth prepass.
        if self.features.enable_depth_prepass {
            self.execute_depth_prepass(primary_view);
        }

        // 3. G-Buffer fill.
        if self.features.enable_gbuffer {
            self.execute_gbuffer_pass(primary_view);
        }

        // 4. SSAO.
        if self.features.enable_ssao {
            self.execute_ssao_pass();
        }

        // 5. Lighting resolve.
        self.execute_lighting_pass(primary_view);

        // 6. SSR.
        if self.features.enable_ssr {
            self.execute_ssr_pass();
        }

        // 7. Transparent objects.
        self.execute_transparent_pass(primary_view);

        // 8. Particles and VFX.
        if self.features.enable_particles {
            self.execute_particle_pass();
        }

        // 9. Post-processing chain.
        self.execute_post_process();

        // 10. UI overlay.
        if self.features.enable_ui_overlay {
            self.execute_ui_overlay();
        }

        // 11. Present (nothing to do in simulation).
        self.stats.add_pass(PassStats {
            name: "Present".into(),
            gpu_time_ms: 0.01,
            cpu_time_ms: 0.001,
            draw_calls: 0,
            triangles: 0,
            state_changes: 0,
        });
    }

    // -----------------------------------------------------------------------
    // Individual pass execution
    // -----------------------------------------------------------------------

    fn execute_shadow_pass(&mut self, _view: &RenderView) {
        let caster_count = self.draw_list.shadow_casters.len() as u32;
        let cascades = self.cascades.len() as u32;

        self.stats.add_pass(PassStats {
            name: "ShadowPass".into(),
            gpu_time_ms: caster_count as f64 * cascades as f64 * 0.01,
            cpu_time_ms: caster_count as f64 * 0.001,
            draw_calls: caster_count * cascades,
            triangles: caster_count * cascades * 1000, // estimate
            state_changes: cascades,
        });
    }

    fn execute_depth_prepass(&mut self, _view: &RenderView) {
        let count = self.draw_list.opaque.len() as u32;
        self.stats.add_pass(PassStats {
            name: "DepthPrepass".into(),
            gpu_time_ms: count as f64 * 0.005,
            cpu_time_ms: count as f64 * 0.001,
            draw_calls: count,
            triangles: count * 500,
            state_changes: 1,
        });
    }

    fn execute_gbuffer_pass(&mut self, _view: &RenderView) {
        let count = self.draw_list.opaque.len() as u32;
        self.stats.add_pass(PassStats {
            name: "GBufferFill".into(),
            gpu_time_ms: count as f64 * 0.015,
            cpu_time_ms: count as f64 * 0.002,
            draw_calls: count,
            triangles: count * 1000,
            state_changes: count / 4 + 1,
        });
    }

    fn execute_ssao_pass(&mut self) {
        self.stats.add_pass(PassStats {
            name: "SSAO".into(),
            gpu_time_ms: 0.8,
            cpu_time_ms: 0.05,
            draw_calls: 2, // downsample + blur
            triangles: 4,
            state_changes: 2,
        });
    }

    fn execute_lighting_pass(&mut self, _view: &RenderView) {
        self.stats.add_pass(PassStats {
            name: "LightingResolve".into(),
            gpu_time_ms: 1.2,
            cpu_time_ms: 0.1,
            draw_calls: 1,
            triangles: 2,
            state_changes: 1,
        });
    }

    fn execute_ssr_pass(&mut self) {
        self.stats.add_pass(PassStats {
            name: "SSR".into(),
            gpu_time_ms: 1.5,
            cpu_time_ms: 0.05,
            draw_calls: 1,
            triangles: 2,
            state_changes: 1,
        });
    }

    fn execute_transparent_pass(&mut self, _view: &RenderView) {
        let count = self.draw_list.transparent.len() as u32;
        self.stats.add_pass(PassStats {
            name: "TransparentPass".into(),
            gpu_time_ms: count as f64 * 0.02,
            cpu_time_ms: count as f64 * 0.003,
            draw_calls: count,
            triangles: count * 500,
            state_changes: count,
        });
    }

    fn execute_particle_pass(&mut self) {
        self.stats.add_pass(PassStats {
            name: "Particles".into(),
            gpu_time_ms: 0.4,
            cpu_time_ms: 0.1,
            draw_calls: 4,
            triangles: 2000,
            state_changes: 2,
        });
    }

    fn execute_post_process(&mut self) {
        let active_effects = self
            .post_process_chain
            .effects
            .iter()
            .filter(|e| e.enabled)
            .count() as u32;

        self.stats.add_pass(PassStats {
            name: "PostProcess".into(),
            gpu_time_ms: active_effects as f64 * 0.3,
            cpu_time_ms: active_effects as f64 * 0.01,
            draw_calls: active_effects,
            triangles: active_effects * 2,
            state_changes: active_effects,
        });
    }

    fn execute_ui_overlay(&mut self) {
        self.stats.add_pass(PassStats {
            name: "UIOverlay".into(),
            gpu_time_ms: 0.1,
            cpu_time_ms: 0.05,
            draw_calls: 10,
            triangles: 100,
            state_changes: 5,
        });
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Get the current frame stats.
    pub fn stats(&self) -> &RenderStats {
        &self.stats
    }

    /// Get the current draw list.
    pub fn draw_list(&self) -> &DrawList {
        &self.draw_list
    }

    /// Get cascade data.
    pub fn cascades(&self) -> &[ShadowCascade] {
        &self.cascades
    }

    /// Current frame index.
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }
}

impl Default for RenderPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Quality preset helper
// ---------------------------------------------------------------------------

/// Quality level presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityPreset {
    Low,
    Medium,
    High,
    Ultra,
    Custom,
}

impl QualityPreset {
    /// Get feature flags for this preset.
    pub fn features(self) -> RenderFeatureFlags {
        match self {
            Self::Low => RenderFeatureFlags::low_quality(),
            Self::Medium => RenderFeatureFlags::medium_quality(),
            Self::High | Self::Ultra => RenderFeatureFlags::high_quality(),
            Self::Custom => RenderFeatureFlags::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Instancing helpers
// ---------------------------------------------------------------------------

/// Groups draw items by mesh+material to enable instanced rendering.
pub struct InstanceBatcher;

impl InstanceBatcher {
    /// Batch a draw list into instanced groups.
    /// Returns groups of draw items that share the same mesh and material.
    pub fn batch(items: &[DrawItem]) -> Vec<InstanceGroup> {
        let mut groups: HashMap<(u64, u64), Vec<usize>> = HashMap::new();

        for (i, item) in items.iter().enumerate() {
            let key = (item.mesh.0, item.material.0);
            groups.entry(key).or_default().push(i);
        }

        groups
            .into_iter()
            .map(|((mesh, material), indices)| InstanceGroup {
                mesh: MeshHandle(mesh),
                material: MaterialHandle(material),
                instance_indices: indices,
            })
            .collect()
    }
}

/// A group of instances sharing the same mesh and material.
#[derive(Debug, Clone)]
pub struct InstanceGroup {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    /// Indices into the original draw list.
    pub instance_indices: Vec<usize>,
}

impl InstanceGroup {
    /// Number of instances in this group.
    pub fn count(&self) -> usize {
        self.instance_indices.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_view() -> RenderView {
        // Simple ortho VP: maps [-10,10]^3 to [-1,1]^3
        let s = 0.1f32;
        let vp = [
            s, 0.0, 0.0, 0.0,
            0.0, s, 0.0, 0.0,
            0.0, 0.0, s, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        RenderView {
            id: 0,
            name: "TestView".into(),
            view_matrix: identity_matrix(),
            projection_matrix: identity_matrix(),
            view_projection: vp,
            inv_view_projection: identity_matrix(),
            prev_view_projection: vp,
            camera_position: [0.0, 0.0, 0.0],
            camera_forward: [0.0, 0.0, -1.0],
            viewport: ViewportRect::full(1920.0, 1080.0),
            render_target: None,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            near_clip: 0.1,
            far_clip: 1000.0,
            fov_y: std::f32::consts::PI / 3.0,
            aspect_ratio: 16.0 / 9.0,
            jitter: [0.0, 0.0],
            priority: 0,
            layer_mask: u32::MAX,
        }
    }

    fn make_test_renderable(id: u64, pos: [f32; 3], transparent: bool) -> RenderableComponent {
        RenderableComponent {
            entity_id: id,
            mesh: MeshHandle(1),
            material: MaterialHandle(1),
            world_matrix: identity_matrix(),
            prev_world_matrix: identity_matrix(),
            aabb_min: [pos[0] - 0.5, pos[1] - 0.5, pos[2] - 0.5],
            aabb_max: [pos[0] + 0.5, pos[1] + 0.5, pos[2] + 0.5],
            is_transparent: transparent,
            casts_shadow: !transparent,
            layer_mask: u32::MAX,
            lod_distances: vec![50.0, 100.0],
        }
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = RenderPipeline::new();
        assert_eq!(pipeline.frame_index(), 0);
        assert!(pipeline.features.enable_shadows);
    }

    #[test]
    fn test_pipeline_prepare_and_render() {
        let mut pipeline = RenderPipeline::new();
        let view = make_test_view();

        let renderables = vec![
            make_test_renderable(0, [0.0, 0.0, -5.0], false),
            make_test_renderable(1, [2.0, 0.0, -3.0], false),
            make_test_renderable(2, [0.0, 0.0, -8.0], true),
        ];

        pipeline.prepare_frame(&renderables, &view);
        pipeline.render_frame(&view);

        let stats = pipeline.stats();
        assert_eq!(stats.frame_index, 0);
        assert!(stats.total_draw_calls > 0);
        assert!(stats.pass_stats.len() >= 5);
    }

    #[test]
    fn test_draw_item_sort_keys() {
        let mut opaque = DrawItem {
            mesh: MeshHandle(1),
            material: MaterialHandle(10),
            world_matrix: identity_matrix(),
            prev_world_matrix: identity_matrix(),
            world_aabb_min: [0.0, 0.0, 0.0],
            world_aabb_max: [1.0, 1.0, 1.0],
            sphere_center: [0.5, 0.5, 0.5],
            sphere_radius: 0.866,
            distance_sq: 25.0,
            sort_key: 0,
            lod_level: 0,
            layer_mask: u32::MAX,
            is_transparent: false,
            casts_shadow: true,
            entity_id: 0,
            sub_mesh: 0,
            instance_count: 1,
        };
        opaque.compute_opaque_sort_key();
        assert!(opaque.sort_key > 0);

        let mut transparent = opaque.clone();
        transparent.is_transparent = true;
        transparent.compute_transparent_sort_key();
        assert!(transparent.sort_key > 0);
    }

    #[test]
    fn test_frustum_culler() {
        let s = 0.1f32;
        let vp = [
            s, 0.0, 0.0, 0.0,
            0.0, s, 0.0, 0.0,
            0.0, 0.0, s, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let culler = FrustumCuller::from_view_projection(&vp);

        // Inside.
        assert!(culler.is_visible(&[-1.0, -1.0, -1.0], &[1.0, 1.0, 1.0]));
        // Outside.
        assert!(!culler.is_visible(&[100.0, 100.0, 100.0], &[101.0, 101.0, 101.0]));
    }

    #[test]
    fn test_cascade_splits() {
        let splits = compute_cascade_splits(0.1, 200.0, 4, 0.75);
        assert_eq!(splits.len(), 5);
        assert!((splits[0] - 0.1).abs() < 0.001);
        assert!(splits[1] < splits[2]);
        assert!(splits[2] < splits[3]);
        assert!(splits[3] < splits[4]);
        assert!((splits[4] - 200.0).abs() < 0.1);
    }

    #[test]
    fn test_quality_presets() {
        let low = QualityPreset::Low.features();
        let high = QualityPreset::High.features();
        assert!(high.shadow_map_resolution > low.shadow_map_resolution);
        assert!(high.enable_ssr);
        assert!(!low.enable_ssr);
    }

    #[test]
    fn test_post_process_chain() {
        let features = RenderFeatureFlags::high_quality();
        let chain = PostProcessChain::from_features(&features);
        assert!(!chain.effects.is_empty());
        // Check effects are in order.
        for i in 1..chain.effects.len() {
            assert!(chain.effects[i].order >= chain.effects[i - 1].order);
        }
    }

    #[test]
    fn test_instance_batcher() {
        let items = vec![
            DrawItem {
                mesh: MeshHandle(1),
                material: MaterialHandle(10),
                world_matrix: identity_matrix(),
                prev_world_matrix: identity_matrix(),
                world_aabb_min: [0.0; 3],
                world_aabb_max: [1.0; 3],
                sphere_center: [0.5; 3],
                sphere_radius: 1.0,
                distance_sq: 1.0,
                sort_key: 0,
                lod_level: 0,
                layer_mask: u32::MAX,
                is_transparent: false,
                casts_shadow: true,
                entity_id: 0,
                sub_mesh: 0,
                instance_count: 1,
            },
            DrawItem {
                mesh: MeshHandle(1),
                material: MaterialHandle(10),
                world_matrix: identity_matrix(),
                prev_world_matrix: identity_matrix(),
                world_aabb_min: [2.0; 3],
                world_aabb_max: [3.0; 3],
                sphere_center: [2.5; 3],
                sphere_radius: 1.0,
                distance_sq: 4.0,
                sort_key: 0,
                lod_level: 0,
                layer_mask: u32::MAX,
                is_transparent: false,
                casts_shadow: true,
                entity_id: 1,
                sub_mesh: 0,
                instance_count: 1,
            },
            DrawItem {
                mesh: MeshHandle(2),
                material: MaterialHandle(20),
                world_matrix: identity_matrix(),
                prev_world_matrix: identity_matrix(),
                world_aabb_min: [5.0; 3],
                world_aabb_max: [6.0; 3],
                sphere_center: [5.5; 3],
                sphere_radius: 1.0,
                distance_sq: 9.0,
                sort_key: 0,
                lod_level: 0,
                layer_mask: u32::MAX,
                is_transparent: false,
                casts_shadow: true,
                entity_id: 2,
                sub_mesh: 0,
                instance_count: 1,
            },
        ];

        let groups = InstanceBatcher::batch(&items);
        // 2 groups: (mesh=1, mat=10) with 2 instances, (mesh=2, mat=20) with 1.
        assert_eq!(groups.len(), 2);
        let big_group = groups.iter().find(|g| g.count() == 2).unwrap();
        assert_eq!(big_group.mesh.0, 1);
        assert_eq!(big_group.material.0, 10);
    }

    #[test]
    fn test_renderable_gatherer() {
        let view = make_test_view();
        let features = RenderFeatureFlags::default();
        let mut gatherer = RenderableGatherer::new();

        let renderables = vec![
            make_test_renderable(0, [0.0, 0.0, -5.0], false),
            make_test_renderable(1, [0.0, 0.0, -3.0], true),
        ];

        let draw_list = gatherer.gather(&renderables, &view, &features);
        assert!(!draw_list.opaque.is_empty());
        assert!(!draw_list.transparent.is_empty());
    }

    #[test]
    fn test_render_stats_display() {
        let mut stats = RenderStats::new();
        stats.frame_index = 42;
        stats.add_pass(PassStats {
            name: "TestPass".into(),
            gpu_time_ms: 1.5,
            cpu_time_ms: 0.3,
            draw_calls: 100,
            triangles: 50000,
            state_changes: 10,
        });
        let s = format!("{}", stats);
        assert!(s.contains("Frame 42"));
        assert!(s.contains("TestPass"));
    }

    #[test]
    fn test_feature_flags_low_quality() {
        let flags = RenderFeatureFlags::low_quality();
        assert!(!flags.enable_ssao);
        assert!(!flags.enable_ssr);
        assert!(!flags.enable_bloom);
        assert_eq!(flags.shadow_cascade_count, 2);
    }

    #[test]
    fn test_viewport_rect() {
        let vp = ViewportRect::full(1920.0, 1080.0);
        assert_eq!(vp.x, 0.0);
        assert_eq!(vp.y, 0.0);
        assert_eq!(vp.width, 1920.0);
        assert_eq!(vp.height, 1080.0);
    }

    #[test]
    fn test_draw_list_clear() {
        let mut dl = DrawList::new();
        dl.opaque.push(DrawItem {
            mesh: MeshHandle(1),
            material: MaterialHandle(1),
            world_matrix: identity_matrix(),
            prev_world_matrix: identity_matrix(),
            world_aabb_min: [0.0; 3],
            world_aabb_max: [1.0; 3],
            sphere_center: [0.5; 3],
            sphere_radius: 1.0,
            distance_sq: 1.0,
            sort_key: 0,
            lod_level: 0,
            layer_mask: u32::MAX,
            is_transparent: false,
            casts_shadow: true,
            entity_id: 0,
            sub_mesh: 0,
            instance_count: 1,
        });
        assert_eq!(dl.total_items(), 1);
        dl.clear();
        assert_eq!(dl.total_items(), 0);
    }
}
