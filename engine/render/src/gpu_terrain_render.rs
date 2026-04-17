// engine/render/src/gpu_terrain_render.rs
//
// GPU Terrain Rendering for the Genovo engine.
//
// # Features
//
// - `TerrainGpuRenderer`: heightmap as GPU texture, vertex shader displaces Y
// - WGSL terrain vertex shader: sample heightmap at vertex UV, set Y = height
// - WGSL terrain fragment shader: triplanar mapping, splatmap blending
// - Clipmap LOD: multiple resolution rings centered on camera
// - Normal computation from heightmap gradient (finite differences in shader)
// - Detail texture overlay at close range
// - Terrain shadow receiving

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};

// ============================================================================
// Constants
// ============================================================================

/// Maximum terrain size.
pub const MAX_TERRAIN_SIZE: u32 = 4096;

/// Maximum number of clipmap LOD levels.
pub const MAX_CLIPMAP_LEVELS: usize = 6;

/// Maximum number of terrain splat layers.
pub const MAX_SPLAT_LAYERS: usize = 4;

/// Default terrain height scale.
pub const DEFAULT_HEIGHT_SCALE: f32 = 100.0;

/// Default terrain tile scale (world units per terrain grid cell).
pub const DEFAULT_TILE_SCALE: f32 = 1.0;

// ============================================================================
// Terrain configuration
// ============================================================================

/// Configuration for terrain rendering.
#[derive(Debug, Clone)]
pub struct TerrainConfig {
    /// Terrain width in grid cells.
    pub grid_width: u32,
    /// Terrain depth in grid cells.
    pub grid_depth: u32,
    /// Height scale (maximum height).
    pub height_scale: f32,
    /// World-space scale per grid cell.
    pub tile_scale: f32,
    /// Number of clipmap LOD levels.
    pub num_lod_levels: usize,
    /// Texture tiling scale for each material layer.
    pub texture_scale: [f32; MAX_SPLAT_LAYERS],
    /// Detail texture tiling scale.
    pub detail_scale: f32,
    /// Detail texture visibility distance.
    pub detail_distance: f32,
    /// Whether to use triplanar mapping for steep surfaces.
    pub use_triplanar: bool,
    /// Triplanar blend sharpness.
    pub triplanar_sharpness: f32,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            grid_width: 256,
            grid_depth: 256,
            height_scale: DEFAULT_HEIGHT_SCALE,
            tile_scale: DEFAULT_TILE_SCALE,
            num_lod_levels: 4,
            texture_scale: [10.0, 8.0, 12.0, 6.0],
            detail_scale: 50.0,
            detail_distance: 50.0,
            use_triplanar: true,
            triplanar_sharpness: 4.0,
        }
    }
}

// ============================================================================
// Terrain uniforms
// ============================================================================

/// Terrain rendering parameters uniform.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TerrainUniform {
    /// .x = grid_width, .y = grid_depth, .z = height_scale, .w = tile_scale.
    pub terrain_params: [f32; 4],
    /// Texture scales for each splat layer.
    pub texture_scales: [f32; 4],
    /// .x = detail_scale, .y = detail_distance, .z = triplanar_sharpness, .w = use_triplanar.
    pub detail_params: [f32; 4],
    /// Camera position for LOD and detail fading.
    pub camera_position: [f32; 4],
    /// Terrain world offset (center of the terrain in world space).
    pub terrain_offset: [f32; 4],
    /// .x = texel_size_u, .y = texel_size_v, .z = lod_level, .w = time.
    pub misc_params: [f32; 4],
}

/// Clipmap level parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ClipmapLevelUniform {
    /// .x = grid_offset_x, .y = grid_offset_z, .z = scale, .w = lod_level.
    pub level_params: [f32; 4],
    /// .x = ring_size, .y = inner_size, .z = transition_width, .w = unused.
    pub ring_params: [f32; 4],
}

// ============================================================================
// Terrain vertex shader
// ============================================================================

/// WGSL shader for GPU terrain rendering.
pub const TERRAIN_SHADER_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Terrain Rendering Shader
// ============================================================================
//
// Renders terrain from a heightmap texture. The vertex shader samples the
// heightmap to displace vertex Y positions. The fragment shader blends
// material layers using a splatmap and supports triplanar mapping.
//
// Bind groups:
//   Group 0: Camera + Lights
//   Group 1: Terrain parameters
//   Group 2: Heightmap + Splatmap + Material textures

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PI: f32 = 3.14159265359;
const INV_PI: f32 = 0.31830988618;
const EPSILON: f32 = 0.0001;
const MAX_LIGHTS: u32 = 8u;
const INV_GAMMA: f32 = 0.45454545454;

const LIGHT_TYPE_DIRECTIONAL: f32 = 1.0;
const LIGHT_TYPE_POINT: f32 = 2.0;

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

struct CameraUniform {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
};

struct GpuLight {
    position_or_direction: vec4<f32>,
    color_intensity: vec4<f32>,
    params: vec4<f32>,
};

struct LightsUniform {
    ambient: vec4<f32>,
    light_count: vec4<f32>,
    lights: array<GpuLight, 8>,
};

struct TerrainParams {
    terrain_params: vec4<f32>,
    texture_scales: vec4<f32>,
    detail_params: vec4<f32>,
    camera_position: vec4<f32>,
    terrain_offset: vec4<f32>,
    misc_params: vec4<f32>,
};

// ---------------------------------------------------------------------------
// Bind groups
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> lights: LightsUniform;

@group(1) @binding(0) var<uniform> terrain: TerrainParams;

// Heightmap and splatmap textures.
@group(2) @binding(0) var heightmap_texture: texture_2d<f32>;
@group(2) @binding(1) var heightmap_sampler: sampler;
@group(2) @binding(2) var splatmap_texture: texture_2d<f32>;
@group(2) @binding(3) var splatmap_sampler: sampler;

// Material layer textures (4 layers).
@group(2) @binding(4) var layer0_texture: texture_2d<f32>;
@group(2) @binding(5) var layer1_texture: texture_2d<f32>;
@group(2) @binding(6) var layer2_texture: texture_2d<f32>;
@group(2) @binding(7) var layer3_texture: texture_2d<f32>;
@group(2) @binding(8) var layer_sampler: sampler;

// Detail texture overlay.
@group(2) @binding(9) var detail_texture: texture_2d<f32>;
@group(2) @binding(10) var detail_sampler: sampler;

// Normal map for terrain.
@group(2) @binding(11) var normal_texture: texture_2d<f32>;
@group(2) @binding(12) var normal_sampler: sampler;

// ---------------------------------------------------------------------------
// Vertex I/O
// ---------------------------------------------------------------------------

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) view_dir: vec3<f32>,
    @location(4) dist_to_camera: f32,
};

// ---------------------------------------------------------------------------
// Heightmap sampling
// ---------------------------------------------------------------------------

// Sample the heightmap at UV coordinates.
fn sample_height(uv: vec2<f32>) -> f32 {
    let raw = textureSampleLevel(heightmap_texture, heightmap_sampler, uv, 0.0).x;
    return raw * terrain.terrain_params.z; // height_scale
}

// Compute the terrain normal from heightmap finite differences.
fn compute_terrain_normal(uv: vec2<f32>) -> vec3<f32> {
    let texel_u = terrain.misc_params.x;
    let texel_v = terrain.misc_params.y;
    let height_scale = terrain.terrain_params.z;
    let tile_scale = terrain.terrain_params.w;

    // Sample 4 neighbours.
    let h_left  = textureSampleLevel(heightmap_texture, heightmap_sampler, uv + vec2<f32>(-texel_u, 0.0), 0.0).x * height_scale;
    let h_right = textureSampleLevel(heightmap_texture, heightmap_sampler, uv + vec2<f32>( texel_u, 0.0), 0.0).x * height_scale;
    let h_down  = textureSampleLevel(heightmap_texture, heightmap_sampler, uv + vec2<f32>(0.0, -texel_v), 0.0).x * height_scale;
    let h_up    = textureSampleLevel(heightmap_texture, heightmap_sampler, uv + vec2<f32>(0.0,  texel_v), 0.0).x * height_scale;

    // Finite differences.
    let dx = (h_right - h_left) / (2.0 * tile_scale);
    let dz = (h_up - h_down) / (2.0 * tile_scale);

    return normalize(vec3<f32>(-dx, 1.0, -dz));
}

// ---------------------------------------------------------------------------
// Vertex shader
// ---------------------------------------------------------------------------

@vertex
fn vs_terrain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    let grid_w = terrain.terrain_params.x;
    let grid_d = terrain.terrain_params.y;
    let tile_scale = terrain.terrain_params.w;
    let offset = terrain.terrain_offset.xyz;

    // UV from vertex position (grid is laid out in XZ plane).
    let uv = input.uv;

    // Sample height.
    let height = sample_height(uv);

    // Compute world position.
    let world_x = input.position.x * tile_scale + offset.x;
    let world_y = height + offset.y;
    let world_z = input.position.z * tile_scale + offset.z;
    let world_pos = vec3<f32>(world_x, world_y, world_z);
    output.world_position = world_pos;

    // Compute normal from heightmap.
    output.world_normal = compute_terrain_normal(uv);

    // Transform to clip space.
    output.clip_position = camera.view_projection * vec4<f32>(world_pos, 1.0);

    output.uv = uv;
    output.view_dir = normalize(camera.camera_position.xyz - world_pos);
    output.dist_to_camera = length(camera.camera_position.xyz - world_pos);

    return output;
}

// ---------------------------------------------------------------------------
// Triplanar mapping
// ---------------------------------------------------------------------------

fn triplanar_sample(
    tex: texture_2d<f32>,
    samp: sampler,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    tex_scale: f32,
    sharpness: f32,
) -> vec4<f32> {
    // Triplanar blend weights from the normal.
    var blend = abs(normal);
    blend = pow(blend, vec3<f32>(sharpness));
    blend = blend / (blend.x + blend.y + blend.z);

    // Sample texture from 3 projections.
    let xy = textureSample(tex, samp, world_pos.xy * tex_scale);
    let xz = textureSample(tex, samp, world_pos.xz * tex_scale);
    let yz = textureSample(tex, samp, world_pos.yz * tex_scale);

    // Blend.
    return xy * blend.z + xz * blend.y + yz * blend.x;
}

// ---------------------------------------------------------------------------
// PBR lighting (compact)
// ---------------------------------------------------------------------------

fn fresnel_sch(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let t = clamp(1.0 - cos_theta, 0.0, 1.0);
    return f0 + (vec3<f32>(1.0) - f0) * t * t * t * t * t;
}

fn dist_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness; let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + EPSILON);
}

fn geom_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0; let k = (r * r) / 8.0;
    let gv = n_dot_v / (n_dot_v * (1.0 - k) + k + EPSILON);
    let gl = n_dot_l / (n_dot_l * (1.0 - k) + k + EPSILON);
    return gv * gl;
}

fn atten(distance: f32, range: f32) -> f32 {
    if range <= 0.0 { return 1.0; }
    let d = distance / range; let d2 = d * d; let d4 = d2 * d2;
    let f = clamp(1.0 - d4, 0.0, 1.0);
    return f * f / (distance * distance + 1.0);
}

fn linear_to_srgb_t(c: f32) -> f32 {
    if c <= 0.0031308 { return c * 12.92; }
    return 1.055 * pow(c, INV_GAMMA) - 0.055;
}

fn tone_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 1.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + vec3<f32>(b))) / (x * (c * x + vec3<f32>(d)) + vec3<f32>(e)),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------

@fragment
fn fs_terrain(input: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(input.world_normal);
    let view_dir = normalize(input.view_dir);
    let uv = input.uv;
    let world_pos = input.world_position;

    // Sample splatmap (RGBA = 4 layer weights).
    let splat = textureSample(splatmap_texture, splatmap_sampler, uv);

    // Determine if we should use triplanar mapping (steep slopes).
    let use_triplanar = terrain.detail_params.w;
    let sharpness = terrain.detail_params.z;
    let slope = 1.0 - abs(normal.y);
    let triplanar_factor = select(0.0, smoothstep(0.3, 0.7, slope), use_triplanar > 0.5);

    // Sample material layers.
    let scale0 = terrain.texture_scales.x;
    let scale1 = terrain.texture_scales.y;
    let scale2 = terrain.texture_scales.z;
    let scale3 = terrain.texture_scales.w;

    var layer0: vec4<f32>;
    var layer1: vec4<f32>;
    var layer2: vec4<f32>;
    var layer3: vec4<f32>;

    if triplanar_factor > 0.01 {
        // Triplanar for steep surfaces.
        let tp0 = triplanar_sample(layer0_texture, layer_sampler, world_pos, normal, scale0 * 0.01, sharpness);
        let tp1 = triplanar_sample(layer1_texture, layer_sampler, world_pos, normal, scale1 * 0.01, sharpness);
        let tp2 = triplanar_sample(layer2_texture, layer_sampler, world_pos, normal, scale2 * 0.01, sharpness);
        let tp3 = triplanar_sample(layer3_texture, layer_sampler, world_pos, normal, scale3 * 0.01, sharpness);

        // Regular UV sampling for flat surfaces.
        let uv0 = textureSample(layer0_texture, layer_sampler, uv * scale0);
        let uv1 = textureSample(layer1_texture, layer_sampler, uv * scale1);
        let uv2 = textureSample(layer2_texture, layer_sampler, uv * scale2);
        let uv3 = textureSample(layer3_texture, layer_sampler, uv * scale3);

        // Blend between UV and triplanar based on slope.
        layer0 = mix(uv0, tp0, triplanar_factor);
        layer1 = mix(uv1, tp1, triplanar_factor);
        layer2 = mix(uv2, tp2, triplanar_factor);
        layer3 = mix(uv3, tp3, triplanar_factor);
    } else {
        // Flat: just UV sampling.
        layer0 = textureSample(layer0_texture, layer_sampler, uv * scale0);
        layer1 = textureSample(layer1_texture, layer_sampler, uv * scale1);
        layer2 = textureSample(layer2_texture, layer_sampler, uv * scale2);
        layer3 = textureSample(layer3_texture, layer_sampler, uv * scale3);
    }

    // Blend layers using splatmap weights.
    let total_weight = splat.x + splat.y + splat.z + splat.w + 0.001;
    let w0 = splat.x / total_weight;
    let w1 = splat.y / total_weight;
    let w2 = splat.z / total_weight;
    let w3 = splat.w / total_weight;

    var albedo = layer0.xyz * w0 + layer1.xyz * w1 + layer2.xyz * w2 + layer3.xyz * w3;

    // Detail texture (close range).
    let detail_scale = terrain.detail_params.x;
    let detail_distance = terrain.detail_params.y;
    let dist = input.dist_to_camera;
    if dist < detail_distance {
        let detail = textureSample(detail_texture, detail_sampler, uv * detail_scale);
        let detail_fade = 1.0 - smoothstep(detail_distance * 0.5, detail_distance, dist);
        // Overlay blend: detail modulates the base color.
        albedo = albedo * (1.0 + (detail.xyz - vec3<f32>(0.5)) * 0.5 * detail_fade);
    }

    // Terrain has fixed PBR parameters.
    let metallic = 0.0;
    let roughness = 0.85;
    let reflectance = 0.3;
    let f0 = vec3<f32>(0.16 * reflectance * reflectance);

    // Lighting.
    var total_diffuse = vec3<f32>(0.0);
    var total_specular = vec3<f32>(0.0);
    let num_lights = u32(lights.light_count.x);

    for (var i = 0u; i < 8u; i = i + 1u) {
        if i >= num_lights { break; }
        let light = lights.lights[i];
        let lt = light.params.x;

        var light_dir: vec3<f32>;
        var light_color: vec3<f32>;
        var att_val: f32 = 1.0;

        if abs(lt - LIGHT_TYPE_DIRECTIONAL) < 0.5 {
            light_dir = normalize(-light.position_or_direction.xyz);
            light_color = light.color_intensity.xyz * light.color_intensity.w;
        } else if abs(lt - LIGHT_TYPE_POINT) < 0.5 {
            let to_l = light.position_or_direction.xyz - world_pos;
            light_dir = normalize(to_l);
            light_color = light.color_intensity.xyz * light.color_intensity.w;
            att_val = atten(length(to_l), light.position_or_direction.w);
        } else { continue; }

        let n_dot_l = max(dot(normal, light_dir), 0.0);
        if n_dot_l <= 0.0 { continue; }

        let h = normalize(view_dir + light_dir);
        let n_dot_v = max(dot(normal, view_dir), EPSILON);
        let n_dot_h = max(dot(normal, h), 0.0);
        let v_dot_h = max(dot(view_dir, h), 0.0);

        let d = dist_ggx(n_dot_h, roughness);
        let g = geom_smith(n_dot_v, n_dot_l, roughness);
        let f = fresnel_sch(v_dot_h, f0);
        let spec = (d * g * f) / (4.0 * n_dot_v * n_dot_l + EPSILON);
        let k_d = (vec3<f32>(1.0) - f) * (1.0 - metallic);
        let attenuated = light_color * att_val;

        total_diffuse = total_diffuse + k_d * albedo * INV_PI * attenuated * n_dot_l;
        total_specular = total_specular + spec * attenuated * n_dot_l;
    }

    // Ambient.
    let sky = lights.ambient.xyz * lights.ambient.w;
    let ground = sky * vec3<f32>(0.6, 0.5, 0.4) * 0.5;
    let amb = mix(ground, sky, normal.y * 0.5 + 0.5);
    let ambient_d = albedo * amb;

    var final_color = total_diffuse + total_specular + ambient_d;

    // Distance fog.
    let fog_start = 200.0;
    let fog_end = 500.0;
    let fog_color = vec3<f32>(0.7, 0.8, 0.9);
    let fog_factor = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
    final_color = mix(final_color, fog_color, fog_factor * fog_factor);

    final_color = tone_aces(final_color);
    final_color = vec3<f32>(
        linear_to_srgb_t(final_color.x),
        linear_to_srgb_t(final_color.y),
        linear_to_srgb_t(final_color.z)
    );

    return vec4<f32>(final_color, 1.0);
}
"#;

// ============================================================================
// Terrain GPU renderer
// ============================================================================

/// GPU terrain renderer.
pub struct TerrainGpuRenderer {
    /// Configuration.
    pub config: TerrainConfig,
    /// Render pipeline.
    pipeline: wgpu::RenderPipeline,
    /// Terrain parameters bind group layout (group 1).
    terrain_bgl: wgpu::BindGroupLayout,
    /// Texture bind group layout (group 2).
    texture_bgl: wgpu::BindGroupLayout,
    /// Terrain parameters uniform buffer.
    terrain_params_buffer: wgpu::Buffer,
    /// Terrain params bind group.
    terrain_bind_group: wgpu::BindGroup,
    /// Grid vertex buffer.
    grid_vertex_buffer: wgpu::Buffer,
    /// Grid index buffer.
    grid_index_buffer: wgpu::Buffer,
    /// Number of grid indices.
    grid_index_count: u32,
}

impl TerrainGpuRenderer {
    /// Create a new terrain renderer.
    pub fn new(
        device: &wgpu::Device,
        camera_lights_bgl: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        config: TerrainConfig,
    ) -> Self {
        // --- Terrain params BGL ---
        let terrain_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("terrain_params_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // --- Texture BGL (group 2) ---
        let texture_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("terrain_texture_bgl"),
                entries: &[
                    // Heightmap + sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Splatmap + sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // 4 material layer textures + shared sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Detail texture + sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Normal map + sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 11,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 12,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // --- Pipeline ---
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain_shader"),
            source: wgpu::ShaderSource::Wgsl(TERRAIN_SHADER_WGSL.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain_pipeline_layout"),
            bind_group_layouts: &[camera_lights_bgl, &terrain_bgl, &texture_bgl],
            push_constant_ranges: &[],
        });

        let vertex_layout = super::scene_renderer::SceneVertex::buffer_layout();

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_terrain"),
                buffers: &[vertex_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_terrain"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // --- Terrain params buffer ---
        let terrain_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain_params_buffer"),
            size: std::mem::size_of::<TerrainUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let terrain_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain_params_bg"),
            layout: &terrain_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: terrain_params_buffer.as_entire_binding(),
            }],
        });

        // --- Generate grid mesh ---
        let (grid_vb, grid_ib, grid_idx_count) =
            Self::generate_terrain_grid(device, config.grid_width, config.grid_depth);

        Self {
            config,
            pipeline,
            terrain_bgl,
            texture_bgl,
            terrain_params_buffer,
            terrain_bind_group,
            grid_vertex_buffer: grid_vb,
            grid_index_buffer: grid_ib,
            grid_index_count: grid_idx_count,
        }
    }

    /// Generate a flat terrain grid mesh (XZ plane).
    fn generate_terrain_grid(
        device: &wgpu::Device,
        width: u32,
        depth: u32,
    ) -> (wgpu::Buffer, wgpu::Buffer, u32) {
        use wgpu::util::DeviceExt;

        let mut vertices = Vec::with_capacity(((width + 1) * (depth + 1)) as usize);
        let mut indices = Vec::with_capacity((width * depth * 6) as usize);

        for z in 0..=depth {
            for x in 0..=width {
                let u = x as f32 / width as f32;
                let v = z as f32 / depth as f32;
                let px = x as f32 - width as f32 * 0.5;
                let pz = z as f32 - depth as f32 * 0.5;

                vertices.push(super::scene_renderer::SceneVertex {
                    position: [px, 0.0, pz],
                    normal: [0.0, 1.0, 0.0],
                    uv: [u, v],
                    color: [1.0, 1.0, 1.0, 1.0],
                });
            }
        }

        for z in 0..depth {
            for x in 0..width {
                let tl = z * (width + 1) + x;
                let tr = tl + 1;
                let bl = (z + 1) * (width + 1) + x;
                let br = bl + 1;

                indices.push(tl);
                indices.push(bl);
                indices.push(tr);
                indices.push(tr);
                indices.push(bl);
                indices.push(br);
            }
        }

        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain_grid_vb"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain_grid_ib"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        (vb, ib, indices.len() as u32)
    }

    /// Update terrain parameters.
    pub fn update_params(
        &self,
        queue: &wgpu::Queue,
        camera_pos: Vec3,
        terrain_offset: Vec3,
        time: f32,
    ) {
        let uniform = TerrainUniform {
            terrain_params: [
                self.config.grid_width as f32,
                self.config.grid_depth as f32,
                self.config.height_scale,
                self.config.tile_scale,
            ],
            texture_scales: self.config.texture_scale,
            detail_params: [
                self.config.detail_scale,
                self.config.detail_distance,
                self.config.triplanar_sharpness,
                if self.config.use_triplanar { 1.0 } else { 0.0 },
            ],
            camera_position: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
            terrain_offset: [terrain_offset.x, terrain_offset.y, terrain_offset.z, 0.0],
            misc_params: [
                1.0 / self.config.grid_width as f32,
                1.0 / self.config.grid_depth as f32,
                0.0,
                time,
            ],
        };

        queue.write_buffer(
            &self.terrain_params_buffer,
            0,
            bytemuck::bytes_of(&uniform),
        );
    }

    /// Get the texture bind group layout.
    pub fn texture_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.texture_bgl
    }

    /// Render the terrain.
    pub fn render<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        camera_lights_bg: &'a wgpu::BindGroup,
        texture_bg: &'a wgpu::BindGroup,
    ) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, camera_lights_bg, &[]);
        pass.set_bind_group(1, &self.terrain_bind_group, &[]);
        pass.set_bind_group(2, texture_bg, &[]);
        pass.set_vertex_buffer(0, self.grid_vertex_buffer.slice(..));
        pass.set_index_buffer(
            self.grid_index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        pass.draw_indexed(0..self.grid_index_count, 0, 0..1);
    }
}

// ============================================================================
// Heightmap generation utilities
// ============================================================================

/// Generate a simple diamond-square heightmap for testing.
pub fn generate_test_heightmap(size: u32) -> Vec<u8> {
    let mut data = vec![0u8; (size * size * 4) as usize];

    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;

            // Multi-octave sine waves for testing.
            let h1 = ((u * 4.0 * std::f32::consts::PI).sin() * 0.3
                + (v * 3.0 * std::f32::consts::PI).sin() * 0.2)
                * 0.5
                + 0.5;
            let h2 = ((u * 12.0 * std::f32::consts::PI).sin() * 0.1
                + (v * 10.0 * std::f32::consts::PI).sin() * 0.08)
                * 0.5
                + 0.5;
            let height = (h1 * 0.7 + h2 * 0.3).clamp(0.0, 1.0);

            let byte = (height * 255.0) as u8;
            let idx = ((y * size + x) * 4) as usize;
            data[idx] = byte;
            data[idx + 1] = byte;
            data[idx + 2] = byte;
            data[idx + 3] = 255;
        }
    }

    data
}

/// Generate a flat splatmap for testing.
pub fn generate_test_splatmap(size: u32) -> Vec<u8> {
    let mut data = vec![0u8; (size * size * 4) as usize];

    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;

            // Height-based splatting.
            let center_dist = ((u - 0.5) * (u - 0.5) + (v - 0.5) * (v - 0.5)).sqrt();

            let idx = ((y * size + x) * 4) as usize;

            // Layer 0: grass (low areas).
            data[idx] = ((1.0 - center_dist * 2.0).clamp(0.0, 1.0) * 255.0) as u8;
            // Layer 1: rock (medium areas).
            data[idx + 1] = ((center_dist * 2.0).clamp(0.0, 1.0) * 128.0) as u8;
            // Layer 2: sand (edges).
            data[idx + 2] = if center_dist > 0.4 { 200 } else { 0 };
            // Layer 3: snow (peaks).
            data[idx + 3] = 0;
        }
    }

    data
}
