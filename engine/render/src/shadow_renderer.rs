// engine/render/src/shadow_renderer.rs
//
// Shadow map rendering system for the Genovo engine.
//
// Implements a complete shadow mapping pipeline:
//
// - Depth-only render pass from each light's perspective
// - Cascaded shadow maps (CSM) for directional lights
// - Cubemap shadow maps for point lights
// - Single-face shadow maps for spot lights
// - PCF (Percentage Closer Filtering) with configurable kernel sizes
// - Poisson disk sampling for soft shadow edges
// - Slope-scaled depth bias to avoid shadow acne
// - Shadow atlas packing for multiple lights into a single large texture
// - Frustum-fit shadow matrices for optimal depth precision
// - Shadow caster culling per light frustum
// - GPU pipeline creation with actual WGSL shaders
//
// The shadow renderer is designed to be driven by the main scene renderer,
// which calls `prepare_frame` at the start of each frame, then renders shadow
// passes for each light, and finally provides the resulting shadow atlas to the
// main lighting pass.

use glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
use std::collections::HashMap;
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of shadow-casting lights per frame.
pub const MAX_SHADOW_LIGHTS: usize = 16;

/// Maximum number of cascades for directional lights.
pub const MAX_CASCADES: usize = 4;

/// Default shadow atlas dimensions.
pub const DEFAULT_ATLAS_SIZE: u32 = 4096;

/// Default shadow map resolution per light.
pub const DEFAULT_SHADOW_MAP_SIZE: u32 = 1024;

/// Default PCF kernel size (3x3).
pub const DEFAULT_PCF_KERNEL_SIZE: u32 = 3;

/// Default depth bias (constant).
pub const DEFAULT_DEPTH_BIAS_CONSTANT: f32 = 1.5;

/// Default depth bias (slope scaled).
pub const DEFAULT_DEPTH_BIAS_SLOPE: f32 = 2.0;

/// Default normal offset bias for shadow sampling.
pub const DEFAULT_NORMAL_BIAS: f32 = 0.02;

/// Maximum number of Poisson disk samples.
pub const MAX_POISSON_SAMPLES: usize = 64;

/// Epsilon for shadow calculations.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// WGSL Shader: Depth-Only Vertex Shader
// ---------------------------------------------------------------------------

/// WGSL shader for depth-only rendering from a light's perspective.
///
/// This shader transforms vertices into light-space clip coordinates using
/// a uniform light-view-projection matrix. No fragment output is needed for
/// the color attachment; only the depth buffer is written.
pub const SHADOW_DEPTH_WGSL: &str = r#"
// Shadow depth-only vertex shader.
// Renders geometry from the light's viewpoint into a depth texture.

struct ShadowUniforms {
    light_view_proj: mat4x4<f32>,
    depth_bias: vec2<f32>,     // x: constant bias, y: slope bias
    shadow_map_size: vec2<f32>, // x: width, y: height
};

@group(0) @binding(0)
var<uniform> shadow: ShadowUniforms;

struct ModelUniforms {
    model_matrix: mat4x4<f32>,
};

@group(1) @binding(0)
var<uniform> model: ModelUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) depth: f32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = model.model_matrix * vec4<f32>(input.position, 1.0);
    let clip_pos = shadow.light_view_proj * world_pos;

    // Apply slope-scaled depth bias using the surface normal.
    let world_normal = normalize((model.model_matrix * vec4<f32>(input.normal, 0.0)).xyz);
    let light_dir = normalize(shadow.light_view_proj[2].xyz);
    let cos_theta = abs(dot(world_normal, light_dir));
    let slope_factor = sqrt(1.0 - cos_theta * cos_theta) / max(cos_theta, 0.001);
    let bias = shadow.depth_bias.x + shadow.depth_bias.y * slope_factor;

    out.clip_position = clip_pos;
    out.clip_position.z = out.clip_position.z + bias / out.clip_position.w;
    out.depth = out.clip_position.z / out.clip_position.w;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) f32 {
    return input.depth;
}
"#;

// ---------------------------------------------------------------------------
// WGSL Shader: Shadow Sampling with PCF
// ---------------------------------------------------------------------------

/// WGSL shader for sampling shadow maps in the main lighting pass.
///
/// Provides functions for standard PCF, Poisson-disk PCF, and cascaded
/// shadow map lookup with cascade blending at boundaries.
pub const SHADOW_SAMPLING_WGSL: &str = r#"
// Shadow sampling utilities for the lighting pass.
// Includes PCF, Poisson disk sampling, and cascade selection.

struct ShadowSamplingUniforms {
    light_matrices: array<mat4x4<f32>, 16>,    // Up to 16 shadow-casting lights
    cascade_matrices: array<mat4x4<f32>, 4>,    // 4 cascades for directional
    cascade_splits: vec4<f32>,                   // View-space Z splits
    shadow_params: vec4<f32>,                    // x: pcf_radius, y: softness, z: bias, w: num_cascades
    atlas_size: vec2<f32>,                       // Shadow atlas dimensions
    texel_size: vec2<f32>,                       // 1.0 / atlas_size
};

@group(2) @binding(0)
var<uniform> shadow_sampling: ShadowSamplingUniforms;

@group(2) @binding(1)
var shadow_atlas: texture_depth_2d;

@group(2) @binding(2)
var shadow_sampler: sampler_comparison;

// Poisson disk sample offsets for soft shadows.
const POISSON_DISK: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2<f32>(-0.94201624, -0.39906216),
    vec2<f32>( 0.94558609, -0.76890725),
    vec2<f32>(-0.09418410, -0.92938870),
    vec2<f32>( 0.34495938,  0.29387760),
    vec2<f32>(-0.91588581,  0.45771432),
    vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543,  0.27676845),
    vec2<f32>( 0.97484398,  0.75648379),
    vec2<f32>( 0.44323325, -0.97511554),
    vec2<f32>( 0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023),
    vec2<f32>( 0.79197514,  0.19090188),
    vec2<f32>(-0.24188840,  0.99706507),
    vec2<f32>(-0.81409955,  0.91437590),
    vec2<f32>( 0.19984126,  0.78641367),
    vec2<f32>( 0.14383161, -0.14100790)
);

// Interleaved gradient noise for per-pixel jittering.
fn interleaved_gradient_noise(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(0.06711056 * pixel.x + 0.00583715 * pixel.y));
}

// Rotation matrix from an angle.
fn rotate_2d(angle: f32) -> mat2x2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return mat2x2<f32>(vec2<f32>(c, -s), vec2<f32>(s, c));
}

// Standard 2D PCF with configurable kernel size.
fn sample_shadow_pcf(shadow_coord: vec3<f32>, uv_offset: vec2<f32>, texel_size: vec2<f32>, kernel_radius: i32) -> f32 {
    var shadow_sum = 0.0;
    var sample_count = 0.0;
    let compare_depth = shadow_coord.z;

    for (var y = -kernel_radius; y <= kernel_radius; y = y + 1) {
        for (var x = -kernel_radius; x <= kernel_radius; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            let sample_uv = shadow_coord.xy + uv_offset + offset;
            shadow_sum += textureSampleCompare(
                shadow_atlas, shadow_sampler, sample_uv, compare_depth
            );
            sample_count += 1.0;
        }
    }
    return shadow_sum / sample_count;
}

// Poisson-disk PCF for soft shadows with per-pixel rotation.
fn sample_shadow_poisson(shadow_coord: vec3<f32>, uv_offset: vec2<f32>,
                          texel_size: vec2<f32>, screen_pos: vec2<f32>,
                          radius: f32, num_samples: i32) -> f32 {
    var shadow_sum = 0.0;
    let noise_angle = interleaved_gradient_noise(screen_pos) * 6.2831853;
    let rot = rotate_2d(noise_angle);
    let compare_depth = shadow_coord.z;
    let actual_samples = min(num_samples, 16);

    for (var i = 0; i < actual_samples; i = i + 1) {
        let disk_offset = rot * POISSON_DISK[i] * radius;
        let sample_uv = shadow_coord.xy + uv_offset + disk_offset * texel_size;
        shadow_sum += textureSampleCompare(
            shadow_atlas, shadow_sampler, sample_uv, compare_depth
        );
    }
    return shadow_sum / f32(actual_samples);
}

// Select the appropriate cascade for a world-space position.
fn select_cascade(view_depth: f32) -> i32 {
    let splits = shadow_sampling.cascade_splits;
    if (view_depth < splits.x) { return 0; }
    if (view_depth < splits.y) { return 1; }
    if (view_depth < splits.z) { return 2; }
    return 3;
}

// Compute cascade blend factor at cascade boundaries.
fn cascade_blend_factor(view_depth: f32, cascade_index: i32) -> f32 {
    let splits = shadow_sampling.cascade_splits;
    var split_end = 0.0;
    if (cascade_index == 0) { split_end = splits.x; }
    else if (cascade_index == 1) { split_end = splits.y; }
    else if (cascade_index == 2) { split_end = splits.z; }
    else { split_end = splits.w; }

    let blend_region = split_end * 0.1;
    let fade_start = split_end - blend_region;

    if (view_depth < fade_start) { return 0.0; }
    return saturate((view_depth - fade_start) / blend_region);
}

// Sample cascaded shadow map with optional inter-cascade blending.
fn sample_cascaded_shadow(world_pos: vec3<f32>, view_depth: f32,
                           screen_pos: vec2<f32>) -> f32 {
    let cascade_idx = select_cascade(view_depth);
    let light_space = shadow_sampling.cascade_matrices[cascade_idx] * vec4<f32>(world_pos, 1.0);
    let proj_coords = light_space.xyz / light_space.w;
    let shadow_uv = proj_coords.xy * 0.5 + 0.5;

    // Check bounds.
    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0) {
        return 1.0;
    }

    let pcf_radius = shadow_sampling.shadow_params.x;
    var shadow = sample_shadow_poisson(
        vec3<f32>(shadow_uv, proj_coords.z),
        vec2<f32>(0.0),
        shadow_sampling.texel_size,
        screen_pos,
        pcf_radius,
        16
    );

    // Blend with next cascade at boundaries.
    let blend = cascade_blend_factor(view_depth, cascade_idx);
    if (blend > 0.0 && cascade_idx < 3) {
        let next_cascade = cascade_idx + 1;
        let next_light_space = shadow_sampling.cascade_matrices[next_cascade] * vec4<f32>(world_pos, 1.0);
        let next_proj = next_light_space.xyz / next_light_space.w;
        let next_uv = next_proj.xy * 0.5 + 0.5;
        let next_shadow = sample_shadow_poisson(
            vec3<f32>(next_uv, next_proj.z),
            vec2<f32>(0.0),
            shadow_sampling.texel_size,
            screen_pos,
            pcf_radius * 0.5,
            8
        );
        shadow = mix(shadow, next_shadow, blend);
    }

    return shadow;
}
"#;

// ---------------------------------------------------------------------------
// WGSL Shader: Cubemap Shadow (Point Lights)
// ---------------------------------------------------------------------------

/// WGSL shader for point light cubemap shadow mapping.
pub const SHADOW_CUBEMAP_WGSL: &str = r#"
// Point light cubemap shadow mapping.
// Renders depth from the light's perspective onto 6 faces.

struct PointShadowUniforms {
    face_view_proj: mat4x4<f32>,
    light_position: vec3<f32>,
    far_plane: f32,
};

@group(0) @binding(0)
var<uniform> point_shadow: PointShadowUniforms;

struct ModelUniforms {
    model_matrix: mat4x4<f32>,
};

@group(1) @binding(0)
var<uniform> model: ModelUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = model.model_matrix * vec4<f32>(input.position, 1.0);
    out.world_position = world_pos.xyz;
    out.clip_position = point_shadow.face_view_proj * world_pos;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) f32 {
    // Store linear distance from light to fragment.
    let light_to_frag = input.world_position - point_shadow.light_position;
    let dist = length(light_to_frag);
    return dist / point_shadow.far_plane;
}
"#;

// ---------------------------------------------------------------------------
// Shadow light types
// ---------------------------------------------------------------------------

/// Type of shadow-casting light.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShadowLightType {
    /// Directional light with cascaded shadow maps.
    Directional,
    /// Point light with cubemap shadow.
    Point,
    /// Spot light with single frustum shadow.
    Spot,
    /// Area light with soft shadow estimation.
    Area,
}

impl std::fmt::Display for ShadowLightType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Directional => write!(f, "Directional"),
            Self::Point => write!(f, "Point"),
            Self::Spot => write!(f, "Spot"),
            Self::Area => write!(f, "Area"),
        }
    }
}

// ---------------------------------------------------------------------------
// Shadow quality preset
// ---------------------------------------------------------------------------

/// Quality preset for shadow rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShadowQuality {
    /// 512x512 per shadow, 2x2 PCF.
    Low,
    /// 1024x1024 per shadow, 3x3 PCF.
    Medium,
    /// 2048x2048 per shadow, 5x5 PCF with Poisson sampling.
    High,
    /// 4096x4096 per shadow, 7x7 PCF with Poisson + inter-cascade blend.
    Ultra,
}

impl ShadowQuality {
    /// Returns the shadow map resolution for this quality.
    pub fn resolution(&self) -> u32 {
        match self {
            Self::Low => 512,
            Self::Medium => 1024,
            Self::High => 2048,
            Self::Ultra => 4096,
        }
    }

    /// Returns the PCF kernel radius for this quality.
    pub fn pcf_kernel_radius(&self) -> u32 {
        match self {
            Self::Low => 1,
            Self::Medium => 1,
            Self::High => 2,
            Self::Ultra => 3,
        }
    }

    /// Returns the number of Poisson samples for this quality.
    pub fn poisson_samples(&self) -> u32 {
        match self {
            Self::Low => 4,
            Self::Medium => 8,
            Self::High => 16,
            Self::Ultra => 32,
        }
    }

    /// Whether to enable cascade blending at this quality level.
    pub fn cascade_blending(&self) -> bool {
        matches!(self, Self::High | Self::Ultra)
    }

    /// Whether to enable contact shadows at this quality level.
    pub fn contact_shadows(&self) -> bool {
        matches!(self, Self::High | Self::Ultra)
    }
}

impl Default for ShadowQuality {
    fn default() -> Self {
        Self::Medium
    }
}

// ---------------------------------------------------------------------------
// Shadow map region (atlas tile)
// ---------------------------------------------------------------------------

/// A rectangular region within the shadow atlas.
#[derive(Debug, Clone, Copy)]
pub struct ShadowAtlasTile {
    /// X offset in the atlas (pixels).
    pub x: u32,
    /// Y offset in the atlas (pixels).
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl ShadowAtlasTile {
    /// Creates a new tile at the given offset and size.
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self { x, y, width, height }
    }

    /// Returns the UV offset and scale for this tile within the atlas.
    pub fn uv_transform(&self, atlas_size: u32) -> (Vec2, Vec2) {
        let inv = 1.0 / atlas_size as f32;
        let offset = Vec2::new(self.x as f32 * inv, self.y as f32 * inv);
        let scale = Vec2::new(self.width as f32 * inv, self.height as f32 * inv);
        (offset, scale)
    }

    /// Returns the viewport for this tile.
    pub fn viewport(&self) -> [u32; 4] {
        [self.x, self.y, self.width, self.height]
    }

    /// Check if a point (in atlas pixel coordinates) is within this tile.
    pub fn contains(&self, px: u32, py: u32) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }
}

// ---------------------------------------------------------------------------
// Shadow atlas allocator
// ---------------------------------------------------------------------------

/// A simple shelf-based shadow atlas allocator.
///
/// Packs shadow map tiles into a large atlas texture using a shelf (row)
/// algorithm. Each shelf has a fixed height (tallest tile in the row).
/// When a tile doesn't fit in the current shelf, a new shelf is started.
#[derive(Debug, Clone)]
pub struct ShadowAtlasAllocator {
    /// Total atlas width in pixels.
    pub atlas_width: u32,
    /// Total atlas height in pixels.
    pub atlas_height: u32,
    /// Current X cursor within the current shelf.
    cursor_x: u32,
    /// Current Y cursor (top of current shelf).
    cursor_y: u32,
    /// Height of the current shelf.
    shelf_height: u32,
    /// All allocated tiles.
    tiles: Vec<ShadowAtlasTile>,
    /// Map from light ID to tile index.
    light_tile_map: HashMap<u64, usize>,
    /// Freed tile indices available for reuse.
    free_list: Vec<usize>,
}

impl ShadowAtlasAllocator {
    /// Creates a new allocator with the given atlas dimensions.
    pub fn new(atlas_width: u32, atlas_height: u32) -> Self {
        Self {
            atlas_width,
            atlas_height,
            cursor_x: 0,
            cursor_y: 0,
            shelf_height: 0,
            tiles: Vec::new(),
            light_tile_map: HashMap::new(),
            free_list: Vec::new(),
        }
    }

    /// Allocates a tile for a given light. Returns the tile index.
    pub fn allocate(&mut self, light_id: u64, width: u32, height: u32) -> Option<usize> {
        // Check if this light already has a tile.
        if let Some(&idx) = self.light_tile_map.get(&light_id) {
            let tile = &self.tiles[idx];
            if tile.width == width && tile.height == height {
                return Some(idx);
            }
            // Size changed, free the old tile.
            self.free_list.push(idx);
            self.light_tile_map.remove(&light_id);
        }

        // Try to reuse a freed tile of matching size.
        for (fi, &free_idx) in self.free_list.iter().enumerate() {
            let tile = &self.tiles[free_idx];
            if tile.width == width && tile.height == height {
                self.free_list.swap_remove(fi);
                self.light_tile_map.insert(light_id, free_idx);
                return Some(free_idx);
            }
        }

        // Try to fit in the current shelf.
        if self.cursor_x + width <= self.atlas_width {
            if self.cursor_y + height.max(self.shelf_height) <= self.atlas_height {
                let tile = ShadowAtlasTile::new(self.cursor_x, self.cursor_y, width, height);
                let idx = self.tiles.len();
                self.tiles.push(tile);
                self.light_tile_map.insert(light_id, idx);
                self.cursor_x += width;
                self.shelf_height = self.shelf_height.max(height);
                return Some(idx);
            }
        }

        // Start a new shelf.
        self.cursor_x = 0;
        self.cursor_y += self.shelf_height;
        self.shelf_height = 0;

        if self.cursor_x + width <= self.atlas_width
            && self.cursor_y + height <= self.atlas_height
        {
            let tile = ShadowAtlasTile::new(self.cursor_x, self.cursor_y, width, height);
            let idx = self.tiles.len();
            self.tiles.push(tile);
            self.light_tile_map.insert(light_id, idx);
            self.cursor_x += width;
            self.shelf_height = height;
            return Some(idx);
        }

        // Atlas is full.
        None
    }

    /// Frees the tile for a given light.
    pub fn free(&mut self, light_id: u64) {
        if let Some(idx) = self.light_tile_map.remove(&light_id) {
            self.free_list.push(idx);
        }
    }

    /// Returns the tile at a given index.
    pub fn get_tile(&self, idx: usize) -> Option<&ShadowAtlasTile> {
        self.tiles.get(idx)
    }

    /// Returns the tile for a given light.
    pub fn get_light_tile(&self, light_id: u64) -> Option<&ShadowAtlasTile> {
        self.light_tile_map.get(&light_id).and_then(|&idx| self.tiles.get(idx))
    }

    /// Resets the allocator, freeing all tiles.
    pub fn reset(&mut self) {
        self.tiles.clear();
        self.light_tile_map.clear();
        self.free_list.clear();
        self.cursor_x = 0;
        self.cursor_y = 0;
        self.shelf_height = 0;
    }

    /// Returns the number of active tiles.
    pub fn active_tile_count(&self) -> usize {
        self.tiles.len() - self.free_list.len()
    }

    /// Returns utilization as a fraction of atlas area used.
    pub fn utilization(&self) -> f32 {
        let total_area = self.atlas_width as f32 * self.atlas_height as f32;
        if total_area < 1.0 {
            return 0.0;
        }
        let used_area: f32 = self.tiles.iter().enumerate()
            .filter(|(i, _)| !self.free_list.contains(i))
            .map(|(_, t)| (t.width * t.height) as f32)
            .sum();
        used_area / total_area
    }
}

// ---------------------------------------------------------------------------
// Cascade configuration
// ---------------------------------------------------------------------------

/// Configuration for cascaded shadow map split calculation.
#[derive(Debug, Clone)]
pub struct CascadeConfig {
    /// Number of cascades (1-4).
    pub num_cascades: u32,
    /// Near plane distance (camera).
    pub near_plane: f32,
    /// Far plane distance for shadows (usually < camera far).
    pub shadow_distance: f32,
    /// Lambda for logarithmic/uniform blend (0 = uniform, 1 = logarithmic).
    pub split_lambda: f32,
    /// Per-cascade resolution override (if empty, use quality preset).
    pub cascade_resolutions: Vec<u32>,
    /// Extra padding around cascade frustums (world units).
    pub cascade_padding: f32,
    /// Whether to stabilize cascade edges to prevent shimmering during rotation.
    pub stabilize: bool,
    /// Whether to use tight frustum fitting per cascade.
    pub tight_fitting: bool,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            num_cascades: 4,
            near_plane: 0.1,
            shadow_distance: 200.0,
            split_lambda: 0.75,
            cascade_resolutions: Vec::new(),
            cascade_padding: 1.0,
            stabilize: true,
            tight_fitting: true,
        }
    }
}

impl CascadeConfig {
    /// Compute cascade split distances using the practical split scheme
    /// (blend between logarithmic and uniform).
    pub fn compute_splits(&self) -> [f32; MAX_CASCADES] {
        let mut splits = [0.0f32; MAX_CASCADES];
        let n = self.num_cascades.min(MAX_CASCADES as u32) as usize;
        let near = self.near_plane;
        let far = self.shadow_distance;
        let lambda = self.split_lambda;

        for i in 0..n {
            let p = (i + 1) as f32 / n as f32;
            let log_split = near * (far / near).powf(p);
            let uniform_split = near + (far - near) * p;
            splits[i] = lambda * log_split + (1.0 - lambda) * uniform_split;
        }

        splits
    }

    /// Compute the view-space Z range for a specific cascade.
    pub fn cascade_range(&self, cascade_index: u32) -> (f32, f32) {
        let splits = self.compute_splits();
        let near = if cascade_index == 0 {
            self.near_plane
        } else {
            splits[cascade_index as usize - 1]
        };
        let far = splits[cascade_index as usize];
        (near, far)
    }
}

// ---------------------------------------------------------------------------
// Shadow caster
// ---------------------------------------------------------------------------

/// Describes a shadow-casting object for culling and rendering.
#[derive(Debug, Clone)]
pub struct ShadowCaster {
    /// Unique entity/mesh ID.
    pub id: u64,
    /// Model transform matrix.
    pub model_matrix: Mat4,
    /// World-space AABB minimum.
    pub aabb_min: Vec3,
    /// World-space AABB maximum.
    pub aabb_max: Vec3,
    /// Number of vertices in the mesh.
    pub vertex_count: u32,
    /// Number of indices in the mesh.
    pub index_count: u32,
    /// Whether this caster is static (for shadow caching).
    pub is_static: bool,
    /// Last frame this caster was rendered.
    pub last_rendered_frame: u64,
}

impl ShadowCaster {
    /// Returns the AABB center.
    pub fn center(&self) -> Vec3 {
        (self.aabb_min + self.aabb_max) * 0.5
    }

    /// Returns the AABB half-extents.
    pub fn half_extents(&self) -> Vec3 {
        (self.aabb_max - self.aabb_min) * 0.5
    }

    /// Returns the AABB sphere radius (bounding sphere).
    pub fn bounding_radius(&self) -> f32 {
        self.half_extents().length()
    }

    /// Tests if this caster's AABB is inside or intersects a frustum.
    pub fn is_visible_to_frustum(&self, frustum_planes: &[Vec4; 6]) -> bool {
        let center = self.center();
        let extents = self.half_extents();

        for plane in frustum_planes {
            let normal = Vec3::new(plane.x, plane.y, plane.z);
            let d = plane.w;
            let r = extents.x * normal.x.abs()
                + extents.y * normal.y.abs()
                + extents.z * normal.z.abs();
            let dist = normal.dot(center) + d;
            if dist < -r {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Shadow light
// ---------------------------------------------------------------------------

/// A light that casts shadows, with its shadow configuration.
#[derive(Debug, Clone)]
pub struct ShadowLight {
    /// Unique light identifier.
    pub id: u64,
    /// Type of shadow light.
    pub light_type: ShadowLightType,
    /// World-space position.
    pub position: Vec3,
    /// Direction (for directional and spot lights).
    pub direction: Vec3,
    /// Light color (for debug visualization).
    pub color: Vec3,
    /// Light intensity (for importance scoring).
    pub intensity: f32,
    /// Light range (for point/spot lights).
    pub range: f32,
    /// Spot angle in radians (inner).
    pub inner_angle: f32,
    /// Spot angle in radians (outer).
    pub outer_angle: f32,
    /// Shadow resolution for this light.
    pub shadow_resolution: u32,
    /// Depth bias (constant term).
    pub depth_bias_constant: f32,
    /// Depth bias (slope-scaled term).
    pub depth_bias_slope: f32,
    /// Normal offset bias.
    pub normal_bias: f32,
    /// Shadow softness (affects PCF radius).
    pub softness: f32,
    /// View-projection matrix for this light.
    pub view_proj: Mat4,
    /// View matrix for this light.
    pub view: Mat4,
    /// Projection matrix for this light.
    pub projection: Mat4,
    /// Cascade configuration (for directional lights).
    pub cascade_config: Option<CascadeConfig>,
    /// Whether this light's shadow is currently cached.
    pub cached: bool,
    /// Shadow importance score (higher = more important).
    pub importance: f32,
    /// Whether this light is enabled.
    pub enabled: bool,
}

impl ShadowLight {
    /// Creates a new directional shadow light.
    pub fn directional(id: u64, direction: Vec3, cascade_config: CascadeConfig) -> Self {
        let dir = direction.normalize();
        let view = compute_directional_view(dir);
        Self {
            id,
            light_type: ShadowLightType::Directional,
            position: Vec3::ZERO,
            direction: dir,
            color: Vec3::ONE,
            intensity: 1.0,
            range: f32::MAX,
            inner_angle: 0.0,
            outer_angle: 0.0,
            shadow_resolution: DEFAULT_SHADOW_MAP_SIZE,
            depth_bias_constant: DEFAULT_DEPTH_BIAS_CONSTANT,
            depth_bias_slope: DEFAULT_DEPTH_BIAS_SLOPE,
            normal_bias: DEFAULT_NORMAL_BIAS,
            softness: 1.0,
            view_proj: Mat4::IDENTITY,
            view,
            projection: Mat4::IDENTITY,
            cascade_config: Some(cascade_config),
            cached: false,
            importance: 1.0,
            enabled: true,
        }
    }

    /// Creates a new point shadow light.
    pub fn point(id: u64, position: Vec3, range: f32) -> Self {
        Self {
            id,
            light_type: ShadowLightType::Point,
            position,
            direction: Vec3::NEG_Z,
            color: Vec3::ONE,
            intensity: 1.0,
            range,
            inner_angle: 0.0,
            outer_angle: PI * 2.0,
            shadow_resolution: DEFAULT_SHADOW_MAP_SIZE / 2,
            depth_bias_constant: DEFAULT_DEPTH_BIAS_CONSTANT,
            depth_bias_slope: DEFAULT_DEPTH_BIAS_SLOPE,
            normal_bias: DEFAULT_NORMAL_BIAS,
            softness: 1.0,
            view_proj: Mat4::IDENTITY,
            view: Mat4::IDENTITY,
            projection: Mat4::IDENTITY,
            cascade_config: None,
            cached: false,
            importance: 1.0,
            enabled: true,
        }
    }

    /// Creates a new spot shadow light.
    pub fn spot(id: u64, position: Vec3, direction: Vec3, inner_angle: f32, outer_angle: f32, range: f32) -> Self {
        let dir = direction.normalize();
        let view = compute_spot_view(position, dir);
        let projection = Mat4::perspective_rh(outer_angle * 2.0, 1.0, 0.1, range);
        Self {
            id,
            light_type: ShadowLightType::Spot,
            position,
            direction: dir,
            color: Vec3::ONE,
            intensity: 1.0,
            range,
            inner_angle,
            outer_angle,
            shadow_resolution: DEFAULT_SHADOW_MAP_SIZE,
            depth_bias_constant: DEFAULT_DEPTH_BIAS_CONSTANT,
            depth_bias_slope: DEFAULT_DEPTH_BIAS_SLOPE,
            normal_bias: DEFAULT_NORMAL_BIAS,
            softness: 1.0,
            view_proj: projection * view,
            view,
            projection,
            cascade_config: None,
            cached: false,
            importance: 1.0,
            enabled: true,
        }
    }

    /// Compute shadow importance based on screen-space contribution.
    pub fn compute_importance(&mut self, camera_pos: Vec3, screen_size: Vec2) {
        match self.light_type {
            ShadowLightType::Directional => {
                // Directional lights always have full importance.
                self.importance = self.intensity;
            }
            ShadowLightType::Point | ShadowLightType::Spot => {
                let dist = (self.position - camera_pos).length();
                if dist < EPSILON {
                    self.importance = self.intensity * 100.0;
                    return;
                }
                // Approximate screen coverage from distance and range.
                let angular_size = (self.range / dist).atan();
                let screen_fraction = angular_size / PI;
                self.importance = self.intensity * screen_fraction;
            }
            ShadowLightType::Area => {
                let dist = (self.position - camera_pos).length();
                let angular_size = (self.range / dist.max(EPSILON)).atan();
                self.importance = self.intensity * angular_size / PI;
            }
        }
    }

    /// Build the 6 face view-projection matrices for a point light cubemap.
    pub fn point_light_face_matrices(&self) -> [Mat4; 6] {
        let pos = self.position;
        let proj = Mat4::perspective_rh(PI * 0.5, 1.0, 0.1, self.range);

        let targets = [
            (Vec3::X, Vec3::NEG_Y),    // +X
            (Vec3::NEG_X, Vec3::NEG_Y),// -X
            (Vec3::Y, Vec3::Z),        // +Y
            (Vec3::NEG_Y, Vec3::NEG_Z),// -Y
            (Vec3::Z, Vec3::NEG_Y),    // +Z
            (Vec3::NEG_Z, Vec3::NEG_Y),// -Z
        ];

        let mut matrices = [Mat4::IDENTITY; 6];
        for (i, (target, up)) in targets.iter().enumerate() {
            let view = Mat4::look_at_rh(pos, pos + *target, *up);
            matrices[i] = proj * view;
        }
        matrices
    }

    /// Compute cascade view-projection matrices for a directional light.
    pub fn compute_cascade_matrices(
        &self,
        camera_view: &Mat4,
        camera_proj: &Mat4,
        camera_near: f32,
        camera_far: f32,
    ) -> Vec<CascadeData> {
        let config = match &self.cascade_config {
            Some(c) => c,
            None => return Vec::new(),
        };

        let splits = config.compute_splits();
        let num_cascades = config.num_cascades.min(MAX_CASCADES as u32) as usize;
        let inv_view_proj = (*camera_proj * *camera_view).inverse();
        let light_dir = self.direction.normalize();

        let mut cascades = Vec::with_capacity(num_cascades);

        for i in 0..num_cascades {
            let near = if i == 0 { camera_near } else { splits[i - 1] };
            let far = splits[i];

            // Compute frustum corners in world space for this cascade.
            let corners = compute_frustum_corners_world(&inv_view_proj, near, far, camera_near, camera_far);

            // Compute the center of the frustum.
            let center = corners.iter().fold(Vec3::ZERO, |acc, &c| acc + c) / corners.len() as f32;

            // Compute a bounding sphere radius for stable shadow maps.
            let radius = corners.iter()
                .map(|&c| (c - center).length())
                .fold(0.0f32, f32::max);

            let padded_radius = radius + config.cascade_padding;

            // Build light view matrix looking at the center.
            let light_view = Mat4::look_at_rh(
                center - light_dir * padded_radius,
                center,
                Vec3::Y,
            );

            // Build orthographic projection.
            let light_proj = Mat4::orthographic_rh(
                -padded_radius, padded_radius,
                -padded_radius, padded_radius,
                0.0, padded_radius * 2.0,
            );

            let mut view_proj = light_proj * light_view;

            // Stabilize: round the origin to texel boundaries.
            if config.stabilize {
                let res = config.cascade_resolutions.get(i)
                    .copied()
                    .unwrap_or(self.shadow_resolution);
                let texel_size = (padded_radius * 2.0) / res as f32;
                let origin = view_proj * Vec4::new(0.0, 0.0, 0.0, 1.0);
                let snapped_x = (origin.x / texel_size).round() * texel_size;
                let snapped_y = (origin.y / texel_size).round() * texel_size;
                let offset_x = snapped_x - origin.x;
                let offset_y = snapped_y - origin.y;
                let mut correction = Mat4::IDENTITY;
                // Apply translation to the projection to snap texels.
                correction.w_axis.x = offset_x;
                correction.w_axis.y = offset_y;
                view_proj = correction * view_proj;
            }

            cascades.push(CascadeData {
                index: i as u32,
                near_z: near,
                far_z: far,
                view_matrix: light_view,
                proj_matrix: light_proj,
                view_proj_matrix: view_proj,
                radius: padded_radius,
                center,
            });
        }

        cascades
    }
}

/// Data for a single cascade in a cascaded shadow map.
#[derive(Debug, Clone)]
pub struct CascadeData {
    /// Cascade index (0 = nearest).
    pub index: u32,
    /// Near Z in view space.
    pub near_z: f32,
    /// Far Z in view space.
    pub far_z: f32,
    /// Light view matrix.
    pub view_matrix: Mat4,
    /// Light projection matrix.
    pub proj_matrix: Mat4,
    /// Combined view-projection matrix.
    pub view_proj_matrix: Mat4,
    /// Bounding sphere radius.
    pub radius: f32,
    /// Frustum center in world space.
    pub center: Vec3,
}

// ---------------------------------------------------------------------------
// Shadow renderer configuration
// ---------------------------------------------------------------------------

/// Configuration for the shadow renderer.
#[derive(Debug, Clone)]
pub struct ShadowRendererConfig {
    /// Shadow quality preset.
    pub quality: ShadowQuality,
    /// Atlas dimensions.
    pub atlas_size: u32,
    /// Maximum shadow-casting lights per frame.
    pub max_shadow_lights: usize,
    /// Whether to use Poisson disk sampling.
    pub use_poisson: bool,
    /// Number of Poisson disk samples.
    pub poisson_samples: u32,
    /// Whether to enable cascade blending for directional lights.
    pub cascade_blending: bool,
    /// Whether to enable shadow caching for static objects.
    pub shadow_caching: bool,
    /// Minimum importance threshold for a light to cast shadows.
    pub importance_threshold: f32,
    /// Depth format for the shadow atlas.
    pub depth_format: ShadowDepthFormat,
    /// PCF filter mode.
    pub pcf_mode: PcfMode,
    /// Whether to render shadow debug visualization.
    pub debug_visualization: bool,
}

impl Default for ShadowRendererConfig {
    fn default() -> Self {
        let quality = ShadowQuality::default();
        Self {
            quality,
            atlas_size: DEFAULT_ATLAS_SIZE,
            max_shadow_lights: MAX_SHADOW_LIGHTS,
            use_poisson: true,
            poisson_samples: quality.poisson_samples(),
            cascade_blending: quality.cascade_blending(),
            shadow_caching: true,
            importance_threshold: 0.01,
            depth_format: ShadowDepthFormat::Depth32Float,
            pcf_mode: PcfMode::Poisson,
            debug_visualization: false,
        }
    }
}

/// Depth texture format options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowDepthFormat {
    Depth16Unorm,
    Depth24Plus,
    Depth32Float,
}

/// PCF filtering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcfMode {
    /// No filtering (hard shadows).
    None,
    /// Standard NxN box PCF.
    Standard,
    /// Poisson disk sampling (softer, lower sample count).
    Poisson,
    /// Rotated Poisson with per-pixel noise (highest quality).
    RotatedPoisson,
}

// ---------------------------------------------------------------------------
// Shadow render pass
// ---------------------------------------------------------------------------

/// Represents a single shadow render pass (one face, one cascade, etc.).
#[derive(Debug, Clone)]
pub struct ShadowRenderPass {
    /// Light ID this pass belongs to.
    pub light_id: u64,
    /// Cascade or face index.
    pub sub_index: u32,
    /// View-projection matrix for this pass.
    pub view_proj: Mat4,
    /// Atlas tile assigned to this pass.
    pub atlas_tile: ShadowAtlasTile,
    /// Caster IDs visible in this pass (after frustum culling).
    pub visible_casters: Vec<u64>,
    /// Frustum planes for culling (6 planes in world space).
    pub frustum_planes: [Vec4; 6],
    /// Depth bias settings.
    pub depth_bias: Vec2,
    /// Whether this pass uses cached results.
    pub use_cache: bool,
}

impl ShadowRenderPass {
    /// Cull shadow casters against this pass's frustum.
    pub fn cull_casters(&mut self, casters: &[ShadowCaster]) {
        self.visible_casters.clear();
        for caster in casters {
            if caster.is_visible_to_frustum(&self.frustum_planes) {
                self.visible_casters.push(caster.id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GPU pipeline descriptors
// ---------------------------------------------------------------------------

/// Vertex layout for shadow depth rendering.
#[derive(Debug, Clone)]
pub struct ShadowVertexLayout {
    /// Vertex stride in bytes.
    pub stride: u32,
    /// Position attribute offset.
    pub position_offset: u32,
    /// Normal attribute offset (for slope bias).
    pub normal_offset: u32,
    /// Whether normals are present (if not, skip slope bias).
    pub has_normals: bool,
}

impl Default for ShadowVertexLayout {
    fn default() -> Self {
        Self {
            stride: 24, // 3 floats position + 3 floats normal
            position_offset: 0,
            normal_offset: 12,
            has_normals: true,
        }
    }
}

/// Describes the GPU pipeline for shadow depth rendering.
#[derive(Debug, Clone)]
pub struct ShadowPipelineDesc {
    /// Vertex layout.
    pub vertex_layout: ShadowVertexLayout,
    /// Depth format.
    pub depth_format: ShadowDepthFormat,
    /// Whether to enable depth clamping.
    pub depth_clamp: bool,
    /// Front face winding.
    pub front_face_ccw: bool,
    /// Cull mode for shadow rendering.
    pub cull_back_face: bool,
    /// Whether to use polygon offset (hardware bias).
    pub polygon_offset: bool,
    /// Polygon offset constant factor.
    pub polygon_offset_constant: f32,
    /// Polygon offset slope factor.
    pub polygon_offset_slope: f32,
}

impl Default for ShadowPipelineDesc {
    fn default() -> Self {
        Self {
            vertex_layout: ShadowVertexLayout::default(),
            depth_format: ShadowDepthFormat::Depth32Float,
            depth_clamp: true,
            front_face_ccw: true,
            cull_back_face: true,
            polygon_offset: true,
            polygon_offset_constant: 2.0,
            polygon_offset_slope: 2.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Shadow statistics
// ---------------------------------------------------------------------------

/// Runtime statistics for shadow rendering.
#[derive(Debug, Clone, Default)]
pub struct ShadowStats {
    /// Total shadow passes this frame.
    pub pass_count: u32,
    /// Total shadow casters rendered this frame.
    pub total_casters_rendered: u32,
    /// Total shadow casters culled this frame.
    pub total_casters_culled: u32,
    /// Total triangles rendered for shadows.
    pub total_triangles: u64,
    /// Number of cached shadow passes (not re-rendered).
    pub cached_passes: u32,
    /// Atlas utilization fraction.
    pub atlas_utilization: f32,
    /// Number of active shadow lights.
    pub active_lights: u32,
    /// Time spent on shadow rendering (ms).
    pub render_time_ms: f32,
    /// Peak memory usage for shadow maps (bytes).
    pub memory_usage_bytes: u64,
}

impl ShadowStats {
    /// Returns a formatted summary string.
    pub fn summary(&self) -> String {
        format!(
            "Shadow: {} passes, {} casters ({} culled), {}k tris, {} cached, atlas {:.0}%, {:.2}ms",
            self.pass_count,
            self.total_casters_rendered,
            self.total_casters_culled,
            self.total_triangles / 1000,
            self.cached_passes,
            self.atlas_utilization * 100.0,
            self.render_time_ms,
        )
    }
}

// ---------------------------------------------------------------------------
// Shadow renderer
// ---------------------------------------------------------------------------

/// The main shadow renderer.
///
/// Manages shadow maps for all shadow-casting lights in the scene. Each frame:
///
/// 1. `prepare_frame` — sort lights by importance, allocate atlas tiles.
/// 2. `build_render_passes` — compute view-projection matrices, cull casters.
/// 3. `execute_passes` (external) — the scene renderer calls this to record
///    draw commands for each shadow pass.
/// 4. `get_shadow_data` — retrieve the final shadow atlas and sampling data
///    for the main lighting pass.
#[derive(Debug)]
pub struct ShadowRenderer {
    /// Configuration.
    pub config: ShadowRendererConfig,
    /// Atlas allocator.
    pub atlas: ShadowAtlasAllocator,
    /// Registered shadow lights.
    pub lights: Vec<ShadowLight>,
    /// Shadow casters in the scene.
    pub casters: Vec<ShadowCaster>,
    /// Active render passes for the current frame.
    pub active_passes: Vec<ShadowRenderPass>,
    /// Cascade data for the primary directional light.
    pub cascade_data: Vec<CascadeData>,
    /// Shadow statistics for the current frame.
    pub stats: ShadowStats,
    /// Frame counter for caching.
    pub frame_number: u64,
    /// Cached static shadow map validity flags (light_id -> frame_rendered).
    pub cache_validity: HashMap<u64, u64>,
    /// Shadow pipeline descriptor.
    pub pipeline_desc: ShadowPipelineDesc,
}

impl ShadowRenderer {
    /// Creates a new shadow renderer with the given configuration.
    pub fn new(config: ShadowRendererConfig) -> Self {
        let atlas = ShadowAtlasAllocator::new(config.atlas_size, config.atlas_size);
        Self {
            config,
            atlas,
            lights: Vec::new(),
            casters: Vec::new(),
            active_passes: Vec::new(),
            cascade_data: Vec::new(),
            stats: ShadowStats::default(),
            frame_number: 0,
            cache_validity: HashMap::new(),
            pipeline_desc: ShadowPipelineDesc::default(),
        }
    }

    /// Registers a shadow-casting light.
    pub fn add_light(&mut self, light: ShadowLight) {
        self.lights.push(light);
    }

    /// Removes a shadow-casting light by ID.
    pub fn remove_light(&mut self, light_id: u64) {
        self.lights.retain(|l| l.id != light_id);
        self.atlas.free(light_id);
        self.cache_validity.remove(&light_id);
    }

    /// Registers a shadow caster.
    pub fn add_caster(&mut self, caster: ShadowCaster) {
        self.casters.push(caster);
    }

    /// Removes a shadow caster by ID.
    pub fn remove_caster(&mut self, caster_id: u64) {
        self.casters.retain(|c| c.id != caster_id);
    }

    /// Clears all lights and casters.
    pub fn clear(&mut self) {
        self.lights.clear();
        self.casters.clear();
        self.atlas.reset();
        self.cache_validity.clear();
    }

    /// Prepare the frame: compute importance, sort lights, allocate atlas tiles.
    pub fn prepare_frame(&mut self, camera_pos: Vec3, screen_size: Vec2) {
        self.frame_number += 1;
        self.stats = ShadowStats::default();

        // Compute importance for all lights.
        for light in &mut self.lights {
            light.compute_importance(camera_pos, screen_size);
        }

        // Sort by importance (highest first).
        self.lights.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));

        // Filter out lights below importance threshold and disabled lights.
        let max_lights = self.config.max_shadow_lights;
        let threshold = self.config.importance_threshold;

        let active_lights: Vec<_> = self.lights.iter()
            .filter(|l| l.enabled && l.importance >= threshold)
            .take(max_lights)
            .cloned()
            .collect();

        self.stats.active_lights = active_lights.len() as u32;

        // Free atlas tiles for lights no longer active.
        let active_ids: Vec<u64> = active_lights.iter().map(|l| l.id).collect();
        let stale_ids: Vec<u64> = self.atlas.light_tile_map.keys()
            .filter(|id| !active_ids.contains(id))
            .copied()
            .collect();
        for id in stale_ids {
            self.atlas.free(id);
        }

        // Allocate atlas tiles for active lights.
        for light in &active_lights {
            let res = light.shadow_resolution;
            match light.light_type {
                ShadowLightType::Point => {
                    // Point lights need 6 faces, arranged in a 3x2 grid in the atlas.
                    let face_res = res / 2; // Half resolution per face.
                    let total_w = face_res * 3;
                    let total_h = face_res * 2;
                    self.atlas.allocate(light.id, total_w, total_h);
                }
                ShadowLightType::Directional => {
                    if let Some(ref config) = light.cascade_config {
                        let n = config.num_cascades.min(MAX_CASCADES as u32);
                        // Arrange cascades in a 2x2 grid if 4 cascades.
                        let cols = if n <= 2 { n } else { 2 };
                        let rows = (n + cols - 1) / cols;
                        let total_w = res * cols;
                        let total_h = res * rows;
                        self.atlas.allocate(light.id, total_w, total_h);
                    } else {
                        self.atlas.allocate(light.id, res, res);
                    }
                }
                ShadowLightType::Spot | ShadowLightType::Area => {
                    self.atlas.allocate(light.id, res, res);
                }
            }
        }

        self.stats.atlas_utilization = self.atlas.utilization();
    }

    /// Build render passes for the current frame.
    pub fn build_render_passes(
        &mut self,
        camera_view: &Mat4,
        camera_proj: &Mat4,
        camera_near: f32,
        camera_far: f32,
    ) {
        self.active_passes.clear();
        self.cascade_data.clear();

        let threshold = self.config.importance_threshold;
        let max_lights = self.config.max_shadow_lights;

        let active_lights: Vec<_> = self.lights.iter()
            .filter(|l| l.enabled && l.importance >= threshold)
            .take(max_lights)
            .cloned()
            .collect();

        for light in &active_lights {
            match light.light_type {
                ShadowLightType::Directional => {
                    self.build_directional_passes(light, camera_view, camera_proj, camera_near, camera_far);
                }
                ShadowLightType::Point => {
                    self.build_point_passes(light);
                }
                ShadowLightType::Spot => {
                    self.build_spot_pass(light);
                }
                ShadowLightType::Area => {
                    self.build_spot_pass(light); // Area uses similar technique.
                }
            }
        }

        // Cull casters against each pass.
        let casters = self.casters.clone();
        for pass in &mut self.active_passes {
            pass.cull_casters(&casters);
            self.stats.total_casters_rendered += pass.visible_casters.len() as u32;
            let total_in_scene = casters.len() as u32;
            self.stats.total_casters_culled += total_in_scene.saturating_sub(pass.visible_casters.len() as u32);
        }

        self.stats.pass_count = self.active_passes.len() as u32;
    }

    /// Build render passes for a directional light (one per cascade).
    fn build_directional_passes(
        &mut self,
        light: &ShadowLight,
        camera_view: &Mat4,
        camera_proj: &Mat4,
        camera_near: f32,
        camera_far: f32,
    ) {
        let cascades = light.compute_cascade_matrices(camera_view, camera_proj, camera_near, camera_far);
        let tile = match self.atlas.get_light_tile(light.id) {
            Some(t) => *t,
            None => return,
        };

        let num_cascades = cascades.len();
        let cols = if num_cascades <= 2 { num_cascades as u32 } else { 2 };
        let per_cascade_w = tile.width / cols;
        let per_cascade_h = if num_cascades <= 2 { tile.height } else { tile.height / 2 };

        for (i, cascade) in cascades.iter().enumerate() {
            let col = (i as u32) % cols;
            let row = (i as u32) / cols;
            let sub_tile = ShadowAtlasTile::new(
                tile.x + col * per_cascade_w,
                tile.y + row * per_cascade_h,
                per_cascade_w,
                per_cascade_h,
            );

            let use_cache = self.config.shadow_caching
                && self.cache_validity.get(&light.id).map_or(false, |&f| f == self.frame_number - 1);

            let frustum_planes = extract_frustum_planes(&cascade.view_proj_matrix);

            self.active_passes.push(ShadowRenderPass {
                light_id: light.id,
                sub_index: i as u32,
                view_proj: cascade.view_proj_matrix,
                atlas_tile: sub_tile,
                visible_casters: Vec::new(),
                frustum_planes,
                depth_bias: Vec2::new(light.depth_bias_constant, light.depth_bias_slope),
                use_cache,
            });
        }

        self.cascade_data = cascades;
    }

    /// Build 6 render passes for a point light.
    fn build_point_passes(&mut self, light: &ShadowLight) {
        let tile = match self.atlas.get_light_tile(light.id) {
            Some(t) => *t,
            None => return,
        };

        let face_matrices = light.point_light_face_matrices();
        let face_w = tile.width / 3;
        let face_h = tile.height / 2;

        for (i, vp) in face_matrices.iter().enumerate() {
            let col = (i % 3) as u32;
            let row = (i / 3) as u32;
            let sub_tile = ShadowAtlasTile::new(
                tile.x + col * face_w,
                tile.y + row * face_h,
                face_w,
                face_h,
            );

            let frustum_planes = extract_frustum_planes(vp);

            self.active_passes.push(ShadowRenderPass {
                light_id: light.id,
                sub_index: i as u32,
                view_proj: *vp,
                atlas_tile: sub_tile,
                visible_casters: Vec::new(),
                frustum_planes,
                depth_bias: Vec2::new(light.depth_bias_constant, light.depth_bias_slope),
                use_cache: false,
            });
        }
    }

    /// Build a single render pass for a spot light.
    fn build_spot_pass(&mut self, light: &ShadowLight) {
        let tile = match self.atlas.get_light_tile(light.id) {
            Some(t) => *t,
            None => return,
        };

        let frustum_planes = extract_frustum_planes(&light.view_proj);

        self.active_passes.push(ShadowRenderPass {
            light_id: light.id,
            sub_index: 0,
            view_proj: light.view_proj,
            atlas_tile: tile,
            visible_casters: Vec::new(),
            frustum_planes,
            depth_bias: Vec2::new(light.depth_bias_constant, light.depth_bias_slope),
            use_cache: false,
        });
    }

    /// Returns the GPU uniform data for shadow sampling in the lighting pass.
    pub fn get_shadow_sampling_data(&self) -> ShadowSamplingGpuData {
        let mut light_matrices = [Mat4::IDENTITY; MAX_SHADOW_LIGHTS];
        let mut cascade_matrices = [Mat4::IDENTITY; MAX_CASCADES];
        let mut cascade_splits = Vec4::ZERO;

        for (i, pass) in self.active_passes.iter().enumerate() {
            if i < MAX_SHADOW_LIGHTS {
                light_matrices[i] = pass.view_proj;
            }
        }

        for cascade in &self.cascade_data {
            let idx = cascade.index as usize;
            if idx < MAX_CASCADES {
                cascade_matrices[idx] = cascade.view_proj_matrix;
                match idx {
                    0 => cascade_splits.x = cascade.far_z,
                    1 => cascade_splits.y = cascade.far_z,
                    2 => cascade_splits.z = cascade.far_z,
                    3 => cascade_splits.w = cascade.far_z,
                    _ => {}
                }
            }
        }

        let atlas_size = self.config.atlas_size as f32;
        let texel_size = 1.0 / atlas_size;

        ShadowSamplingGpuData {
            light_matrices,
            cascade_matrices,
            cascade_splits,
            shadow_params: Vec4::new(
                self.config.poisson_samples as f32 * 0.001, // PCF radius
                1.0, // softness
                0.0, // bias (applied per-pass)
                self.cascade_data.len() as f32,
            ),
            atlas_size: Vec2::new(atlas_size, atlas_size),
            texel_size: Vec2::new(texel_size, texel_size),
        }
    }

    /// Apply quality preset, updating config accordingly.
    pub fn set_quality(&mut self, quality: ShadowQuality) {
        self.config.quality = quality;
        self.config.poisson_samples = quality.poisson_samples();
        self.config.cascade_blending = quality.cascade_blending();

        // Update all lights to the new resolution.
        let res = quality.resolution();
        for light in &mut self.lights {
            light.shadow_resolution = res;
        }
    }

    /// Mark a shadow pass as cached for the current frame.
    pub fn mark_cached(&mut self, light_id: u64) {
        self.cache_validity.insert(light_id, self.frame_number);
        self.stats.cached_passes += 1;
    }

    /// Returns the total GPU memory usage estimate for the shadow atlas.
    pub fn estimate_memory_usage(&self) -> u64 {
        let bytes_per_pixel: u64 = match self.config.depth_format {
            ShadowDepthFormat::Depth16Unorm => 2,
            ShadowDepthFormat::Depth24Plus => 4,
            ShadowDepthFormat::Depth32Float => 4,
        };
        let atlas_pixels = self.config.atlas_size as u64 * self.config.atlas_size as u64;
        atlas_pixels * bytes_per_pixel
    }
}

/// GPU data for shadow sampling in the lighting pass.
#[derive(Debug, Clone)]
pub struct ShadowSamplingGpuData {
    /// Light view-projection matrices.
    pub light_matrices: [Mat4; MAX_SHADOW_LIGHTS],
    /// Cascade view-projection matrices.
    pub cascade_matrices: [Mat4; MAX_CASCADES],
    /// Cascade split depths.
    pub cascade_splits: Vec4,
    /// Shadow parameters (PCF radius, softness, bias, num_cascades).
    pub shadow_params: Vec4,
    /// Atlas size.
    pub atlas_size: Vec2,
    /// Texel size (1/atlas_size).
    pub texel_size: Vec2,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Compute a view matrix for a directional light given its direction.
fn compute_directional_view(direction: Vec3) -> Mat4 {
    let up = if direction.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    Mat4::look_at_rh(Vec3::ZERO, direction, up)
}

/// Compute a view matrix for a spot light.
fn compute_spot_view(position: Vec3, direction: Vec3) -> Mat4 {
    let up = if direction.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    Mat4::look_at_rh(position, position + direction, up)
}

/// Extract 6 frustum planes from a view-projection matrix (Gribb-Hartmann).
fn extract_frustum_planes(vp: &Mat4) -> [Vec4; 6] {
    let m = vp.to_cols_array_2d();
    let mut planes = [Vec4::ZERO; 6];

    // Left
    planes[0] = Vec4::new(
        m[0][3] + m[0][0],
        m[1][3] + m[1][0],
        m[2][3] + m[2][0],
        m[3][3] + m[3][0],
    );
    // Right
    planes[1] = Vec4::new(
        m[0][3] - m[0][0],
        m[1][3] - m[1][0],
        m[2][3] - m[2][0],
        m[3][3] - m[3][0],
    );
    // Bottom
    planes[2] = Vec4::new(
        m[0][3] + m[0][1],
        m[1][3] + m[1][1],
        m[2][3] + m[2][1],
        m[3][3] + m[3][1],
    );
    // Top
    planes[3] = Vec4::new(
        m[0][3] - m[0][1],
        m[1][3] - m[1][1],
        m[2][3] - m[2][1],
        m[3][3] - m[3][1],
    );
    // Near
    planes[4] = Vec4::new(
        m[0][3] + m[0][2],
        m[1][3] + m[1][2],
        m[2][3] + m[2][2],
        m[3][3] + m[3][2],
    );
    // Far
    planes[5] = Vec4::new(
        m[0][3] - m[0][2],
        m[1][3] - m[1][2],
        m[2][3] - m[2][2],
        m[3][3] - m[3][2],
    );

    // Normalize all planes.
    for plane in &mut planes {
        let len = Vec3::new(plane.x, plane.y, plane.z).length();
        if len > EPSILON {
            *plane /= len;
        }
    }

    planes
}

/// Compute frustum corners in world space from an inverse VP matrix.
fn compute_frustum_corners_world(
    inv_view_proj: &Mat4,
    near: f32,
    far: f32,
    cam_near: f32,
    cam_far: f32,
) -> Vec<Vec3> {
    // NDC corners at near and far.
    let ndc_near = 2.0 * (near - cam_near) / (cam_far - cam_near) - 1.0;
    let ndc_far = 2.0 * (far - cam_near) / (cam_far - cam_near) - 1.0;

    let ndc_corners = [
        // Near plane corners.
        Vec4::new(-1.0, -1.0, ndc_near, 1.0),
        Vec4::new( 1.0, -1.0, ndc_near, 1.0),
        Vec4::new( 1.0,  1.0, ndc_near, 1.0),
        Vec4::new(-1.0,  1.0, ndc_near, 1.0),
        // Far plane corners.
        Vec4::new(-1.0, -1.0, ndc_far, 1.0),
        Vec4::new( 1.0, -1.0, ndc_far, 1.0),
        Vec4::new( 1.0,  1.0, ndc_far, 1.0),
        Vec4::new(-1.0,  1.0, ndc_far, 1.0),
    ];

    ndc_corners
        .iter()
        .map(|&ndc| {
            let world = *inv_view_proj * ndc;
            Vec3::new(world.x / world.w, world.y / world.w, world.z / world.w)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// ECS integration
// ---------------------------------------------------------------------------

/// ECS component for an entity that casts shadows.
#[derive(Debug, Clone)]
pub struct ShadowCasterComponent {
    /// Whether this entity casts shadows.
    pub casts_shadows: bool,
    /// Whether this is a static shadow caster (eligible for caching).
    pub is_static: bool,
    /// Shadow LOD bias (reduces shadow resolution for unimportant objects).
    pub lod_bias: f32,
    /// Maximum distance from camera at which this object casts shadows.
    pub max_shadow_distance: f32,
    /// Whether to use double-sided shadow rendering (for thin geometry).
    pub double_sided: bool,
}

impl Default for ShadowCasterComponent {
    fn default() -> Self {
        Self {
            casts_shadows: true,
            is_static: false,
            lod_bias: 0.0,
            max_shadow_distance: 200.0,
            double_sided: false,
        }
    }
}

/// ECS component for a shadow-receiving surface.
#[derive(Debug, Clone)]
pub struct ShadowReceiverComponent {
    /// Whether this entity receives shadows.
    pub receives_shadows: bool,
    /// Shadow intensity override (0 = no shadows, 1 = full shadows).
    pub shadow_intensity: f32,
    /// Whether to apply contact shadows to this surface.
    pub contact_shadows: bool,
}

impl Default for ShadowReceiverComponent {
    fn default() -> Self {
        Self {
            receives_shadows: true,
            shadow_intensity: 1.0,
            contact_shadows: true,
        }
    }
}

/// ECS system that drives the shadow renderer from entity data.
#[derive(Debug)]
pub struct ShadowSystem {
    /// The underlying shadow renderer.
    pub renderer: ShadowRenderer,
    /// Next unique light ID.
    next_light_id: u64,
    /// Next unique caster ID.
    next_caster_id: u64,
}

impl ShadowSystem {
    /// Creates a new shadow system with the given configuration.
    pub fn new(config: ShadowRendererConfig) -> Self {
        Self {
            renderer: ShadowRenderer::new(config),
            next_light_id: 1,
            next_caster_id: 1,
        }
    }

    /// Allocate a unique light ID.
    pub fn alloc_light_id(&mut self) -> u64 {
        let id = self.next_light_id;
        self.next_light_id += 1;
        id
    }

    /// Allocate a unique caster ID.
    pub fn alloc_caster_id(&mut self) -> u64 {
        let id = self.next_caster_id;
        self.next_caster_id += 1;
        id
    }

    /// Update the shadow system for the current frame.
    pub fn update(
        &mut self,
        camera_pos: Vec3,
        camera_view: &Mat4,
        camera_proj: &Mat4,
        camera_near: f32,
        camera_far: f32,
        screen_size: Vec2,
    ) {
        self.renderer.prepare_frame(camera_pos, screen_size);
        self.renderer.build_render_passes(camera_view, camera_proj, camera_near, camera_far);
    }

    /// Returns the shadow render passes to execute.
    pub fn render_passes(&self) -> &[ShadowRenderPass] {
        &self.renderer.active_passes
    }

    /// Returns the GPU data for the main lighting pass.
    pub fn shadow_data(&self) -> ShadowSamplingGpuData {
        self.renderer.get_shadow_sampling_data()
    }

    /// Returns shadow statistics.
    pub fn stats(&self) -> &ShadowStats {
        &self.renderer.stats
    }
}

// ---------------------------------------------------------------------------
// Debug visualization
// ---------------------------------------------------------------------------

/// Debug visualization data for shadow maps.
#[derive(Debug, Clone)]
pub struct ShadowDebugData {
    /// Cascade frustum corners for visualization.
    pub cascade_frustums: Vec<[Vec3; 8]>,
    /// Light frustum corners for spot/point lights.
    pub light_frustums: Vec<[Vec3; 8]>,
    /// Atlas tile rects for UI overlay.
    pub atlas_tiles: Vec<ShadowAtlasTile>,
    /// Color per cascade for debug rendering.
    pub cascade_colors: [Vec3; MAX_CASCADES],
}

impl ShadowDebugData {
    /// Create default cascade colors.
    pub fn default_cascade_colors() -> [Vec3; MAX_CASCADES] {
        [
            Vec3::new(1.0, 0.0, 0.0), // Red for cascade 0
            Vec3::new(0.0, 1.0, 0.0), // Green for cascade 1
            Vec3::new(0.0, 0.0, 1.0), // Blue for cascade 2
            Vec3::new(1.0, 1.0, 0.0), // Yellow for cascade 3
        ]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cascade_splits() {
        let config = CascadeConfig {
            num_cascades: 4,
            near_plane: 0.1,
            shadow_distance: 200.0,
            split_lambda: 0.75,
            ..Default::default()
        };
        let splits = config.compute_splits();
        // Each split should be increasing.
        assert!(splits[0] > 0.0);
        assert!(splits[1] > splits[0]);
        assert!(splits[2] > splits[1]);
        assert!(splits[3] > splits[2]);
        // Last split should equal shadow distance.
        assert!((splits[3] - 200.0).abs() < 0.1);
    }

    #[test]
    fn test_atlas_allocator() {
        let mut alloc = ShadowAtlasAllocator::new(4096, 4096);
        let tile0 = alloc.allocate(1, 1024, 1024);
        assert!(tile0.is_some());
        let tile1 = alloc.allocate(2, 1024, 1024);
        assert!(tile1.is_some());
        assert_ne!(tile0.unwrap(), tile1.unwrap());

        // Same light gets same tile.
        let tile0_again = alloc.allocate(1, 1024, 1024);
        assert_eq!(tile0.unwrap(), tile0_again.unwrap());

        assert_eq!(alloc.active_tile_count(), 2);
    }

    #[test]
    fn test_atlas_free_and_reuse() {
        let mut alloc = ShadowAtlasAllocator::new(2048, 2048);
        let idx = alloc.allocate(1, 512, 512).unwrap();
        alloc.free(1);
        assert_eq!(alloc.active_tile_count(), 0);
        let idx2 = alloc.allocate(2, 512, 512).unwrap();
        assert_eq!(idx, idx2); // Reuses the freed slot.
    }

    #[test]
    fn test_shadow_quality_presets() {
        assert_eq!(ShadowQuality::Low.resolution(), 512);
        assert_eq!(ShadowQuality::Ultra.resolution(), 4096);
        assert!(ShadowQuality::Ultra.cascade_blending());
        assert!(!ShadowQuality::Low.cascade_blending());
    }

    #[test]
    fn test_shadow_light_directional() {
        let config = CascadeConfig::default();
        let light = ShadowLight::directional(1, Vec3::new(-1.0, -1.0, -1.0), config);
        assert_eq!(light.light_type, ShadowLightType::Directional);
        assert!(light.cascade_config.is_some());
    }

    #[test]
    fn test_shadow_light_point() {
        let light = ShadowLight::point(2, Vec3::new(5.0, 10.0, 3.0), 25.0);
        assert_eq!(light.light_type, ShadowLightType::Point);
        let matrices = light.point_light_face_matrices();
        assert_eq!(matrices.len(), 6);
    }

    #[test]
    fn test_shadow_caster_frustum_culling() {
        let caster = ShadowCaster {
            id: 1,
            model_matrix: Mat4::IDENTITY,
            aabb_min: Vec3::new(-1.0, -1.0, -1.0),
            aabb_max: Vec3::new(1.0, 1.0, 1.0),
            vertex_count: 36,
            index_count: 36,
            is_static: true,
            last_rendered_frame: 0,
        };

        // A frustum that contains the origin should see this caster.
        let vp = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0)
            * Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let planes = extract_frustum_planes(&vp);
        assert!(caster.is_visible_to_frustum(&planes));
    }

    #[test]
    fn test_shadow_renderer_basic_flow() {
        let config = ShadowRendererConfig::default();
        let mut renderer = ShadowRenderer::new(config);

        let cascade_config = CascadeConfig::default();
        renderer.add_light(ShadowLight::directional(1, Vec3::new(-1.0, -1.0, -1.0), cascade_config));
        renderer.add_caster(ShadowCaster {
            id: 100,
            model_matrix: Mat4::IDENTITY,
            aabb_min: Vec3::new(-5.0, 0.0, -5.0),
            aabb_max: Vec3::new(5.0, 2.0, 5.0),
            vertex_count: 1000,
            index_count: 3000,
            is_static: true,
            last_rendered_frame: 0,
        });

        let camera_view = Mat4::look_at_rh(Vec3::new(0.0, 10.0, 20.0), Vec3::ZERO, Vec3::Y);
        let camera_proj = Mat4::perspective_rh(1.0, 16.0 / 9.0, 0.1, 1000.0);
        let screen = Vec2::new(1920.0, 1080.0);

        renderer.prepare_frame(Vec3::new(0.0, 10.0, 20.0), screen);
        renderer.build_render_passes(&camera_view, &camera_proj, 0.1, 1000.0);

        assert!(renderer.active_passes.len() > 0);
        assert!(renderer.stats.active_lights > 0);
    }

    #[test]
    fn test_atlas_tile_uv_transform() {
        let tile = ShadowAtlasTile::new(1024, 0, 1024, 1024);
        let (offset, scale) = tile.uv_transform(4096);
        assert!((offset.x - 0.25).abs() < EPSILON);
        assert!((offset.y).abs() < EPSILON);
        assert!((scale.x - 0.25).abs() < EPSILON);
    }

    #[test]
    fn test_shadow_stats_summary() {
        let stats = ShadowStats {
            pass_count: 4,
            total_casters_rendered: 50,
            total_casters_culled: 150,
            total_triangles: 250000,
            cached_passes: 1,
            atlas_utilization: 0.65,
            active_lights: 3,
            render_time_ms: 2.5,
            memory_usage_bytes: 64 * 1024 * 1024,
        };
        let summary = stats.summary();
        assert!(summary.contains("4 passes"));
        assert!(summary.contains("50 casters"));
    }
}
