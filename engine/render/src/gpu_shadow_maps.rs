// engine/render/src/gpu_shadow_maps.rs
//
// Real GPU shadow map rendering for the Genovo engine.
//
// # Features
//
// - `ShadowMapRenderer`: depth-only render pipeline + depth texture
// - Shadow pass: render scene from light's perspective into depth texture
// - Directional light: orthographic projection, compute bounds from camera frustum
// - Point light: 6-face cubemap shadow (render to each face)
// - Spot light: perspective projection from light position
// - Cascaded shadow maps (CSM) for directional lights: 4 cascades with PSSM
// - Shadow sampling in PBR fragment shader with PCF 3x3 soft edges
// - WGSL depth-only vertex shader (no fragment output)
// - Shadow bias (constant + slope-scaled) to prevent acne
// - Cascade selection by fragment depth

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};
use std::collections::HashMap;

// ============================================================================
// Constants
// ============================================================================

/// Default shadow map resolution (per cascade / per face).
pub const SHADOW_MAP_SIZE: u32 = 2048;

/// Number of cascades for CSM.
pub const NUM_CASCADES: usize = 4;

/// Maximum number of shadow-casting lights.
pub const MAX_SHADOW_LIGHTS: usize = 4;

/// Default constant depth bias.
pub const DEFAULT_DEPTH_BIAS: f32 = 0.005;

/// Default slope-scaled bias.
pub const DEFAULT_SLOPE_BIAS: f32 = 0.002;

/// Default normal offset bias.
pub const DEFAULT_NORMAL_BIAS: f32 = 0.02;

/// PCF kernel radius.
pub const PCF_RADIUS: i32 = 1; // 3x3 kernel

/// PSSM split lambda (0 = uniform, 1 = logarithmic).
pub const PSSM_LAMBDA: f32 = 0.75;

// ============================================================================
// Shadow map configuration
// ============================================================================

/// Configuration for shadow map rendering.
#[derive(Debug, Clone)]
pub struct ShadowConfig {
    /// Resolution of each shadow map (width = height).
    pub resolution: u32,
    /// Number of cascades for directional light CSM.
    pub num_cascades: usize,
    /// Constant depth bias.
    pub depth_bias: f32,
    /// Slope-scaled depth bias.
    pub slope_bias: f32,
    /// Normal offset bias.
    pub normal_bias: f32,
    /// PSSM split lambda (0 = uniform, 1 = logarithmic).
    pub pssm_lambda: f32,
    /// Maximum shadow distance from camera.
    pub max_shadow_distance: f32,
    /// Enable PCF soft shadows.
    pub enable_pcf: bool,
    /// Enable cascade blending at boundaries.
    pub enable_cascade_blend: bool,
    /// Cascade blend width (world units).
    pub cascade_blend_width: f32,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            resolution: SHADOW_MAP_SIZE,
            num_cascades: NUM_CASCADES,
            depth_bias: DEFAULT_DEPTH_BIAS,
            slope_bias: DEFAULT_SLOPE_BIAS,
            normal_bias: DEFAULT_NORMAL_BIAS,
            pssm_lambda: PSSM_LAMBDA,
            max_shadow_distance: 200.0,
            enable_pcf: true,
            enable_cascade_blend: true,
            cascade_blend_width: 5.0,
        }
    }
}

// ============================================================================
// Shadow cascade uniform
// ============================================================================

/// Per-cascade data uploaded to the GPU.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CascadeUniform {
    /// Light view-projection matrix for this cascade.
    pub light_view_proj: [[f32; 4]; 4],
    /// Split distance in view space (near boundary of this cascade).
    /// .x = split_near, .y = split_far, .z = texel_size, .w = bias_multiplier
    pub split_info: [f32; 4],
}

impl Default for CascadeUniform {
    fn default() -> Self {
        Self {
            light_view_proj: [[0.0; 4]; 4],
            split_info: [0.0; 4],
        }
    }
}

/// Shadow uniform buffer containing all cascade data.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ShadowUniform {
    /// Cascade view-projection matrices and split info.
    pub cascades: [CascadeUniform; 4],
    /// .x = num_cascades, .y = depth_bias, .z = slope_bias, .w = normal_bias.
    pub shadow_params: [f32; 4],
    /// .x = max_shadow_distance, .y = cascade_blend_width, .z = pcf_enabled, .w = unused.
    pub shadow_params2: [f32; 4],
}

impl Default for ShadowUniform {
    fn default() -> Self {
        Self {
            cascades: [CascadeUniform::default(); 4],
            shadow_params: [4.0, DEFAULT_DEPTH_BIAS, DEFAULT_SLOPE_BIAS, DEFAULT_NORMAL_BIAS],
            shadow_params2: [200.0, 5.0, 1.0, 0.0],
        }
    }
}

// ============================================================================
// Depth-only shader (shadow pass)
// ============================================================================

/// WGSL shader for the shadow depth pass.
///
/// Only outputs the vertex position (depth is written automatically by the
/// rasterizer). No fragment shader needed for depth-only rendering.
pub const SHADOW_DEPTH_SHADER_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Shadow Depth Pass Shader
// ============================================================================
//
// Renders the scene from the light's perspective, writing only depth.
// Used for shadow map generation.
//
// Bind groups:
//   Group 0: Light view-projection matrix
//   Group 1: Model world matrix

struct LightViewProj {
    light_view_proj: mat4x4<f32>,
};

struct ModelUniform {
    world: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> light_vp: LightViewProj;
@group(1) @binding(0) var<uniform> model: ModelUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_shadow_depth(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // Transform to world space, then to light clip space.
    let world_pos = model.world * vec4<f32>(input.position, 1.0);
    output.clip_position = light_vp.light_view_proj * world_pos;

    return output;
}

// No fragment shader needed for depth-only rendering.
// The rasterizer automatically writes depth to the depth attachment.
// However, wgpu requires a fragment stage, so we provide an empty one.
@fragment
fn fs_shadow_depth() {
    // Depth is written by the rasterizer. Nothing to output.
}
"#;

/// WGSL shader for shadow depth pass with alpha testing.
pub const SHADOW_DEPTH_ALPHA_SHADER_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Shadow Depth Pass with Alpha Test
// ============================================================================
//
// Same as the basic shadow depth pass, but samples the albedo texture
// to perform alpha testing (for transparent/cutout objects).

struct LightViewProj {
    light_view_proj: mat4x4<f32>,
};

struct ModelUniform {
    world: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
};

struct MaterialUniform {
    albedo_color: vec4<f32>,
    metallic_roughness: vec4<f32>,
    emissive: vec4<f32>,
    flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> light_vp: LightViewProj;
@group(1) @binding(0) var<uniform> model: ModelUniform;
@group(2) @binding(0) var<uniform> material: MaterialUniform;
@group(2) @binding(1) var albedo_texture: texture_2d<f32>;
@group(2) @binding(2) var albedo_sampler_: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_shadow_alpha(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let world_pos = model.world * vec4<f32>(input.position, 1.0);
    output.clip_position = light_vp.light_view_proj * world_pos;
    output.uv = input.uv;
    return output;
}

@fragment
fn fs_shadow_alpha(input: VertexOutput) {
    let alpha = textureSample(albedo_texture, albedo_sampler_, input.uv).w;
    let alpha_cutoff = material.metallic_roughness.w;
    if alpha_cutoff > 0.0 && alpha < alpha_cutoff {
        discard;
    }
}
"#;

// ============================================================================
// Shadow sampling shader functions (to include in PBR shader)
// ============================================================================

/// WGSL functions for shadow sampling in the PBR fragment shader.
///
/// These functions should be included in the PBR shader when shadow maps
/// are enabled. They sample the shadow depth texture and compute visibility.
pub const SHADOW_SAMPLING_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Shadow Sampling Functions
// ============================================================================
//
// Include these functions in the PBR fragment shader to enable shadow
// receiving. Requires the shadow cascade depth texture and comparison sampler.

struct CascadeInfo {
    light_view_proj: mat4x4<f32>,
    split_info: vec4<f32>,
};

struct ShadowData {
    cascades: array<CascadeInfo, 4>,
    shadow_params: vec4<f32>,
    shadow_params2: vec4<f32>,
};

// Shadow map bindings (typically group 3).
@group(3) @binding(0) var<uniform> shadow_data: ShadowData;
@group(3) @binding(1) var shadow_map_cascade0: texture_depth_2d;
@group(3) @binding(2) var shadow_map_cascade1: texture_depth_2d;
@group(3) @binding(3) var shadow_map_cascade2: texture_depth_2d;
@group(3) @binding(4) var shadow_map_cascade3: texture_depth_2d;
@group(3) @binding(5) var shadow_comparison_sampler: sampler_comparison;

// Transform a world-space position to shadow UV + depth.
fn world_to_shadow_uv(world_pos: vec3<f32>, cascade_idx: u32) -> vec3<f32> {
    let light_vp = shadow_data.cascades[cascade_idx].light_view_proj;
    let light_clip = light_vp * vec4<f32>(world_pos, 1.0);
    let ndc = light_clip.xyz / light_clip.w;

    // Convert from NDC [-1,1] to UV [0,1].
    let uv = vec2<f32>(
        ndc.x * 0.5 + 0.5,
        ndc.y * -0.5 + 0.5  // Y is flipped for texture coordinates.
    );

    return vec3<f32>(uv.x, uv.y, ndc.z);
}

// Select the appropriate cascade for a given view-space depth.
fn select_cascade(view_depth: f32) -> u32 {
    let num_cascades = u32(shadow_data.shadow_params.x);

    for (var i = 0u; i < num_cascades; i = i + 1u) {
        if view_depth < shadow_data.cascades[i].split_info.y {
            return i;
        }
    }

    return num_cascades - 1u;
}

// Sample shadow map for cascade 0 with a comparison.
fn sample_shadow_cascade_0(uv: vec2<f32>, compare_depth: f32) -> f32 {
    return textureSampleCompare(shadow_map_cascade0, shadow_comparison_sampler, uv, compare_depth);
}

// Sample shadow map for cascade 1 with a comparison.
fn sample_shadow_cascade_1(uv: vec2<f32>, compare_depth: f32) -> f32 {
    return textureSampleCompare(shadow_map_cascade1, shadow_comparison_sampler, uv, compare_depth);
}

// Sample shadow map for cascade 2 with a comparison.
fn sample_shadow_cascade_2(uv: vec2<f32>, compare_depth: f32) -> f32 {
    return textureSampleCompare(shadow_map_cascade2, shadow_comparison_sampler, uv, compare_depth);
}

// Sample shadow map for cascade 3 with a comparison.
fn sample_shadow_cascade_3(uv: vec2<f32>, compare_depth: f32) -> f32 {
    return textureSampleCompare(shadow_map_cascade3, shadow_comparison_sampler, uv, compare_depth);
}

// Sample a cascade by index.
fn sample_shadow_cascade(cascade_idx: u32, uv: vec2<f32>, compare_depth: f32) -> f32 {
    switch cascade_idx {
        case 0u: { return sample_shadow_cascade_0(uv, compare_depth); }
        case 1u: { return sample_shadow_cascade_1(uv, compare_depth); }
        case 2u: { return sample_shadow_cascade_2(uv, compare_depth); }
        case 3u: { return sample_shadow_cascade_3(uv, compare_depth); }
        default: { return 1.0; }
    }
}

// PCF 3x3 shadow sampling for soft edges.
fn sample_shadow_pcf(cascade_idx: u32, uv: vec2<f32>, compare_depth: f32, texel_size: f32) -> f32 {
    var shadow = 0.0;
    let pcf_enabled = shadow_data.shadow_params2.z;

    if pcf_enabled < 0.5 {
        // No PCF: single sample.
        return sample_shadow_cascade(cascade_idx, uv, compare_depth);
    }

    // 3x3 PCF kernel.
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            shadow = shadow + sample_shadow_cascade(cascade_idx, uv + offset, compare_depth);
        }
    }

    return shadow / 9.0;
}

// Compute shadow visibility for a world-space fragment.
// Returns 1.0 = fully lit, 0.0 = fully shadowed.
fn compute_shadow_visibility(
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    view_depth: f32,
) -> f32 {
    let max_dist = shadow_data.shadow_params2.x;
    if view_depth > max_dist {
        return 1.0; // Beyond shadow distance.
    }

    // Select cascade.
    let cascade_idx = select_cascade(view_depth);
    let cascade = shadow_data.cascades[cascade_idx];

    // Apply normal offset bias.
    let normal_bias = shadow_data.shadow_params.w;
    let biased_pos = world_pos + world_normal * normal_bias;

    // Transform to shadow UV.
    let shadow_coord = world_to_shadow_uv(biased_pos, cascade_idx);
    let uv = shadow_coord.xy;
    let depth = shadow_coord.z;

    // Check bounds.
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
        return 1.0;
    }

    // Apply depth bias.
    let depth_bias = shadow_data.shadow_params.y;
    let slope_bias = shadow_data.shadow_params.z;
    let biased_depth = depth - depth_bias - slope_bias;

    let texel_size = cascade.split_info.z;

    // Sample shadow with PCF.
    var visibility = sample_shadow_pcf(cascade_idx, uv, biased_depth, texel_size);

    // Cascade blending at boundaries.
    let blend_width = shadow_data.shadow_params2.y;
    let split_far = cascade.split_info.y;
    let blend_start = split_far - blend_width;

    if view_depth > blend_start && cascade_idx < u32(shadow_data.shadow_params.x) - 1u {
        let next_cascade = cascade_idx + 1u;
        let next_coord = world_to_shadow_uv(biased_pos, next_cascade);
        let next_texel = shadow_data.cascades[next_cascade].split_info.z;
        let next_vis = sample_shadow_pcf(
            next_cascade, next_coord.xy,
            next_coord.z - depth_bias - slope_bias,
            next_texel
        );

        let blend_factor = (view_depth - blend_start) / blend_width;
        visibility = mix(visibility, next_vis, blend_factor);
    }

    // Fade out at max shadow distance.
    let fade_start = max_dist * 0.8;
    if view_depth > fade_start {
        let fade = (view_depth - fade_start) / (max_dist - fade_start);
        visibility = mix(visibility, 1.0, fade);
    }

    return visibility;
}
"#;

// ============================================================================
// Shadow map renderer
// ============================================================================

/// Manages shadow map rendering for the scene.
pub struct ShadowMapRenderer {
    /// Configuration.
    pub config: ShadowConfig,

    /// Depth-only render pipeline (opaque objects).
    depth_pipeline: wgpu::RenderPipeline,
    /// Depth-only pipeline with alpha test.
    depth_alpha_pipeline: wgpu::RenderPipeline,

    /// Bind group layout for light VP (group 0).
    light_vp_bgl: wgpu::BindGroupLayout,
    /// Bind group layout for model (group 1) -- shared with scene renderer.
    model_bgl: wgpu::BindGroupLayout,

    /// Shadow cascade depth textures.
    cascade_textures: Vec<wgpu::Texture>,
    /// Depth texture views for rendering.
    cascade_depth_views: Vec<wgpu::TextureView>,
    /// Texture views for sampling in the PBR shader.
    cascade_sample_views: Vec<wgpu::TextureView>,

    /// Per-cascade light VP uniform buffers.
    cascade_vp_buffers: Vec<wgpu::Buffer>,
    /// Per-cascade bind groups.
    cascade_vp_bind_groups: Vec<wgpu::BindGroup>,

    /// Shadow uniform buffer (cascade data for the PBR shader).
    shadow_uniform_buffer: wgpu::Buffer,
    /// Comparison sampler for shadow sampling.
    comparison_sampler: wgpu::Sampler,
    /// Shadow bind group for the PBR shader (group 3).
    shadow_bind_group: wgpu::BindGroup,
    /// Shadow bind group layout (group 3).
    shadow_bgl: wgpu::BindGroupLayout,

    /// Cascade split distances (view-space depths).
    cascade_splits: Vec<f32>,

    /// Model uniform pool for shadow pass.
    model_uniform_pool: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
}

impl ShadowMapRenderer {
    /// Create a new shadow map renderer.
    pub fn new(device: &wgpu::Device, config: ShadowConfig) -> Self {
        // --- Bind group layouts ---

        // Light VP bind group (group 0 for shadow pass).
        let light_vp_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_light_vp_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(64), // mat4x4
                    },
                    count: None,
                }],
            });

        // Model bind group (group 1 for shadow pass).
        let model_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_model_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(128), // 2x mat4x4
                    },
                    count: None,
                }],
            });

        // --- Depth-only pipeline ---
        let depth_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("shadow_depth_shader"),
                source: wgpu::ShaderSource::Wgsl(SHADOW_DEPTH_SHADER_WGSL.into()),
            });

        let depth_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow_depth_pipeline_layout"),
                bind_group_layouts: &[&light_vp_bgl, &model_bgl],
                push_constant_ranges: &[],
            });

        let vertex_layout = super::scene_renderer::SceneVertex::buffer_layout();

        let depth_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("shadow_depth_pipeline"),
                layout: Some(&depth_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &depth_shader,
                    entry_point: Some("vs_shadow_depth"),
                    buffers: &[vertex_layout.clone()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &depth_shader,
                    entry_point: Some("fs_shadow_depth"),
                    targets: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    // Use front-face culling for shadow maps to reduce
                    // peter-panning (shadow acne on back faces is acceptable).
                    cull_mode: Some(wgpu::Face::Front),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState {
                        constant: 2, // Constant bias in depth buffer units.
                        slope_scale: 2.0,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        // --- Alpha test depth pipeline ---
        let depth_alpha_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("shadow_depth_alpha_shader"),
                source: wgpu::ShaderSource::Wgsl(SHADOW_DEPTH_ALPHA_SHADER_WGSL.into()),
            });

        // For alpha test, we need material bindings too.
        let alpha_material_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_alpha_material_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let depth_alpha_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow_depth_alpha_pipeline_layout"),
                bind_group_layouts: &[&light_vp_bgl, &model_bgl, &alpha_material_bgl],
                push_constant_ranges: &[],
            });

        let depth_alpha_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("shadow_depth_alpha_pipeline"),
                layout: Some(&depth_alpha_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &depth_alpha_shader,
                    entry_point: Some("vs_shadow_alpha"),
                    buffers: &[vertex_layout],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &depth_alpha_shader,
                    entry_point: Some("fs_shadow_alpha"),
                    targets: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None, // No culling for alpha-tested geometry.
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState {
                        constant: 2,
                        slope_scale: 2.0,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        // --- Create cascade depth textures ---
        let mut cascade_textures = Vec::with_capacity(config.num_cascades);
        let mut cascade_depth_views = Vec::with_capacity(config.num_cascades);
        let mut cascade_sample_views = Vec::with_capacity(config.num_cascades);

        for i in 0..config.num_cascades {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("shadow_cascade_{}", i)),
                size: wgpu::Extent3d {
                    width: config.resolution,
                    height: config.resolution,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            let depth_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("shadow_cascade_{}_depth_view", i)),
                format: Some(wgpu::TextureFormat::Depth32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::DepthOnly,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
            });

            let sample_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("shadow_cascade_{}_sample_view", i)),
                format: Some(wgpu::TextureFormat::Depth32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::DepthOnly,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
            });

            cascade_textures.push(texture);
            cascade_depth_views.push(depth_view);
            cascade_sample_views.push(sample_view);
        }

        // --- Create per-cascade VP buffers ---
        let mut cascade_vp_buffers = Vec::with_capacity(config.num_cascades);
        let mut cascade_vp_bind_groups = Vec::with_capacity(config.num_cascades);

        for i in 0..config.num_cascades {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("shadow_cascade_vp_{}", i)),
                size: 64, // mat4x4<f32>
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("shadow_cascade_vp_bg_{}", i)),
                layout: &light_vp_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });

            cascade_vp_buffers.push(buffer);
            cascade_vp_bind_groups.push(bind_group);
        }

        // --- Shadow uniform buffer for PBR shader ---
        let shadow_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_uniform_buffer"),
            size: std::mem::size_of::<ShadowUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Comparison sampler ---
        let comparison_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_comparison_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // --- Shadow bind group layout (group 3 for PBR shader) ---
        let shadow_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_pbr_bgl"),
                entries: &[
                    // binding 0: ShadowUniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1-4: cascade depth textures
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 5: comparison sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                ],
            });

        // --- Shadow bind group ---
        let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_pbr_bg"),
            layout: &shadow_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shadow_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&cascade_sample_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        if config.num_cascades > 1 {
                            &cascade_sample_views[1]
                        } else {
                            &cascade_sample_views[0]
                        },
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        if config.num_cascades > 2 {
                            &cascade_sample_views[2]
                        } else {
                            &cascade_sample_views[0]
                        },
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(
                        if config.num_cascades > 3 {
                            &cascade_sample_views[3]
                        } else {
                            &cascade_sample_views[0]
                        },
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&comparison_sampler),
                },
            ],
        });

        Self {
            config: config.clone(),
            depth_pipeline,
            depth_alpha_pipeline,
            light_vp_bgl,
            model_bgl,
            cascade_textures,
            cascade_depth_views,
            cascade_sample_views,
            cascade_vp_buffers,
            cascade_vp_bind_groups,
            shadow_uniform_buffer,
            comparison_sampler,
            shadow_bind_group,
            shadow_bgl,
            cascade_splits: Vec::new(),
            model_uniform_pool: Vec::new(),
        }
    }

    /// Access the shadow bind group layout.
    pub fn shadow_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.shadow_bgl
    }

    /// Access the shadow bind group for the PBR shader.
    pub fn shadow_bind_group(&self) -> &wgpu::BindGroup {
        &self.shadow_bind_group
    }

    /// Compute PSSM cascade split distances.
    pub fn compute_cascade_splits(&mut self, near: f32, far: f32) {
        let num = self.config.num_cascades;
        let lambda = self.config.pssm_lambda;
        let max_dist = self.config.max_shadow_distance.min(far);

        self.cascade_splits.clear();
        self.cascade_splits.push(near);

        for i in 1..num {
            let p = i as f32 / num as f32;

            // Logarithmic split.
            let log_split = near * (max_dist / near).powf(p);
            // Uniform split.
            let uniform_split = near + (max_dist - near) * p;
            // PSSM: blend between logarithmic and uniform.
            let split = lambda * log_split + (1.0 - lambda) * uniform_split;

            self.cascade_splits.push(split);
        }

        self.cascade_splits.push(max_dist);
    }

    /// Compute the light view-projection matrix for a directional light cascade.
    pub fn compute_directional_light_vp(
        &self,
        light_dir: Vec3,
        cascade_idx: usize,
        camera_view: Mat4,
        camera_proj: Mat4,
    ) -> Mat4 {
        let near = self.cascade_splits[cascade_idx];
        let far = self.cascade_splits[cascade_idx + 1];

        // Compute the 8 corners of the view frustum slice.
        let inv_vp = (camera_proj * camera_view).inverse();

        let ndc_corners: [Vec3; 8] = [
            Vec3::new(-1.0, -1.0, 0.0), // near bottom-left
            Vec3::new(1.0, -1.0, 0.0),  // near bottom-right
            Vec3::new(-1.0, 1.0, 0.0),  // near top-left
            Vec3::new(1.0, 1.0, 0.0),   // near top-right
            Vec3::new(-1.0, -1.0, 1.0), // far bottom-left
            Vec3::new(1.0, -1.0, 1.0),  // far bottom-right
            Vec3::new(-1.0, 1.0, 1.0),  // far top-left
            Vec3::new(1.0, 1.0, 1.0),   // far top-right
        ];

        let mut world_corners = Vec::with_capacity(8);
        for ndc in &ndc_corners {
            let clip = inv_vp * Vec4::new(ndc.x, ndc.y, ndc.z, 1.0);
            let w = clip.truncate() / clip.w;
            world_corners.push(w);
        }

        // Interpolate between near and far planes based on cascade split.
        let total_near = 0.1f32; // Original camera near.
        let total_far = 1000.0f32;
        let near_t = (near - total_near) / (total_far - total_near);
        let far_t = (far - total_near) / (total_far - total_near);

        let mut cascade_corners = Vec::with_capacity(8);
        for i in 0..4 {
            let near_corner = world_corners[i]; // Near plane corners.
            let far_corner = world_corners[i + 4]; // Far plane corners.
            cascade_corners.push(near_corner + (far_corner - near_corner) * near_t);
            cascade_corners.push(near_corner + (far_corner - near_corner) * far_t);
        }

        // Compute the center of the cascade frustum.
        let mut center = Vec3::ZERO;
        for c in &cascade_corners {
            center += *c;
        }
        center /= cascade_corners.len() as f32;

        // Light view matrix.
        let light_up = if light_dir.y.abs() > 0.99 {
            Vec3::X
        } else {
            Vec3::Y
        };

        let light_view = Mat4::look_at_rh(
            center - light_dir * 100.0, // Position the light far back.
            center,
            light_up,
        );

        // Find the bounding box in light space.
        let mut min_ls = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max_ls = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        for corner in &cascade_corners {
            let ls = (light_view * Vec4::new(corner.x, corner.y, corner.z, 1.0)).truncate();
            min_ls = min_ls.min(ls);
            max_ls = max_ls.max(ls);
        }

        // Add some padding to prevent shadow swimming.
        let padding = 2.0;
        min_ls -= Vec3::splat(padding);
        max_ls += Vec3::splat(padding);

        // Orthographic projection that encompasses the cascade.
        let light_proj = Mat4::orthographic_rh(
            min_ls.x,
            max_ls.x,
            min_ls.y,
            max_ls.y,
            min_ls.z - 200.0, // Extend near plane to catch objects behind camera.
            max_ls.z + 200.0,
        );

        light_proj * light_view
    }

    /// Compute the light VP for a spot light shadow.
    pub fn compute_spot_light_vp(
        light_pos: Vec3,
        light_dir: Vec3,
        fov_radians: f32,
        near: f32,
        far: f32,
    ) -> Mat4 {
        let light_up = if light_dir.y.abs() > 0.99 {
            Vec3::X
        } else {
            Vec3::Y
        };

        let light_view = Mat4::look_at_rh(light_pos, light_pos + light_dir, light_up);
        let light_proj = Mat4::perspective_rh(fov_radians, 1.0, near, far);

        light_proj * light_view
    }

    /// Compute 6 face light VP matrices for a point light cubemap shadow.
    pub fn compute_point_light_cubemap_vps(
        light_pos: Vec3,
        near: f32,
        far: f32,
    ) -> [Mat4; 6] {
        let proj = Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_2, // 90 degrees.
            1.0,
            near,
            far,
        );

        // 6 cubemap face directions and up vectors.
        let directions: [(Vec3, Vec3); 6] = [
            (Vec3::X, Vec3::NEG_Y),     // +X
            (Vec3::NEG_X, Vec3::NEG_Y), // -X
            (Vec3::Y, Vec3::Z),         // +Y
            (Vec3::NEG_Y, Vec3::NEG_Z), // -Y
            (Vec3::Z, Vec3::NEG_Y),     // +Z
            (Vec3::NEG_Z, Vec3::NEG_Y), // -Z
        ];

        let mut vps = [Mat4::IDENTITY; 6];
        for (i, (dir, up)) in directions.iter().enumerate() {
            let view = Mat4::look_at_rh(light_pos, light_pos + *dir, *up);
            vps[i] = proj * view;
        }

        vps
    }

    /// Update cascade data and upload to GPU.
    pub fn update_cascades(
        &self,
        queue: &wgpu::Queue,
        light_dir: Vec3,
        camera_view: Mat4,
        camera_proj: Mat4,
        camera_near: f32,
        camera_far: f32,
    ) {
        let mut shadow_uniform = ShadowUniform::default();
        shadow_uniform.shadow_params = [
            self.config.num_cascades as f32,
            self.config.depth_bias,
            self.config.slope_bias,
            self.config.normal_bias,
        ];
        shadow_uniform.shadow_params2 = [
            self.config.max_shadow_distance,
            self.config.cascade_blend_width,
            if self.config.enable_pcf { 1.0 } else { 0.0 },
            0.0,
        ];

        let texel_size = 1.0 / self.config.resolution as f32;

        for i in 0..self.config.num_cascades {
            if i >= self.cascade_splits.len() - 1 {
                break;
            }

            let light_vp = self.compute_directional_light_vp(
                light_dir,
                i,
                camera_view,
                camera_proj,
            );

            // Upload the light VP to the cascade's uniform buffer.
            queue.write_buffer(
                &self.cascade_vp_buffers[i],
                0,
                bytemuck::bytes_of(&light_vp.to_cols_array_2d()),
            );

            // Fill cascade uniform.
            shadow_uniform.cascades[i] = CascadeUniform {
                light_view_proj: light_vp.to_cols_array_2d(),
                split_info: [
                    self.cascade_splits[i],
                    self.cascade_splits[i + 1],
                    texel_size,
                    1.0, // Bias multiplier.
                ],
            };
        }

        // Upload shadow uniform.
        queue.write_buffer(
            &self.shadow_uniform_buffer,
            0,
            bytemuck::bytes_of(&shadow_uniform),
        );
    }

    /// Ensure model uniform pool has enough entries.
    fn ensure_model_pool(&mut self, device: &wgpu::Device, count: usize) {
        while self.model_uniform_pool.len() < count {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!(
                    "shadow_model_ub_{}",
                    self.model_uniform_pool.len()
                )),
                size: std::mem::size_of::<super::scene_renderer::ModelUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!(
                    "shadow_model_bg_{}",
                    self.model_uniform_pool.len()
                )),
                layout: &self.model_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });

            self.model_uniform_pool.push((buffer, bind_group));
        }
    }

    /// Render the shadow pass for a single cascade.
    ///
    /// `objects` is a slice of (mesh_vertex_buffer, mesh_index_buffer,
    /// index_count, world_matrix) tuples for all shadow-casting objects.
    pub fn render_shadow_pass(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        cascade_idx: usize,
        objects: &[(&wgpu::Buffer, &wgpu::Buffer, u32, Mat4)],
    ) {
        if cascade_idx >= self.cascade_depth_views.len() {
            return;
        }

        // Ensure we have enough model uniform buffers.
        self.ensure_model_pool(device, objects.len());

        // Upload model matrices.
        for (i, (_, _, _, world_matrix)) in objects.iter().enumerate() {
            let model_uniform =
                super::scene_renderer::ModelUniform::from_world_matrix(*world_matrix);
            queue.write_buffer(
                &self.model_uniform_pool[i].0,
                0,
                bytemuck::bytes_of(&model_uniform),
            );
        }

        // Begin depth-only render pass for this cascade.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("shadow_pass_cascade_{}", cascade_idx)),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.cascade_depth_views[cascade_idx],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.depth_pipeline);
            pass.set_bind_group(0, &self.cascade_vp_bind_groups[cascade_idx], &[]);

            for (i, (vertex_buf, index_buf, index_count, _)) in objects.iter().enumerate()
            {
                pass.set_bind_group(1, &self.model_uniform_pool[i].1, &[]);
                pass.set_vertex_buffer(0, vertex_buf.slice(..));
                pass.set_index_buffer(index_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..*index_count, 0, 0..1);
            }
        }
    }

    /// Render all cascade shadow passes.
    pub fn render_all_cascades(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        objects: &[(&wgpu::Buffer, &wgpu::Buffer, u32, Mat4)],
    ) {
        for cascade_idx in 0..self.config.num_cascades {
            self.render_shadow_pass(device, queue, encoder, cascade_idx, objects);
        }
    }

    /// Get the shadow map resolution.
    pub fn resolution(&self) -> u32 {
        self.config.resolution
    }

    /// Get the number of cascades.
    pub fn num_cascades(&self) -> usize {
        self.config.num_cascades
    }

    /// Get cascade split distances.
    pub fn cascade_splits(&self) -> &[f32] {
        &self.cascade_splits
    }
}

// ============================================================================
// Shadow debug visualization shader
// ============================================================================

/// WGSL shader for visualizing shadow map cascades.
pub const SHADOW_DEBUG_SHADER_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Shadow Debug Visualization
// ============================================================================
//
// Renders a fullscreen quad showing the shadow map depth textures for
// debugging purposes. Displays all cascades side by side.

struct DebugParams {
    num_cascades: vec4<f32>,
    viewport: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: DebugParams;
@group(0) @binding(1) var shadow_tex: texture_depth_2d;
@group(0) @binding(2) var shadow_samp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_shadow_debug(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;

    // Fullscreen triangle.
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);

    return output;
}

@fragment
fn fs_shadow_debug(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample the depth texture.
    let depth = textureSample(shadow_tex, shadow_samp, input.uv);

    // Linearize depth for visualization.
    let near = 0.1;
    let far = 200.0;
    let linear_depth = near * far / (far - depth * (far - near));
    let normalized = linear_depth / far;

    // Display as greyscale.
    return vec4<f32>(normalized, normalized, normalized, 1.0);
}

// Cascade overlay coloring.
fn cascade_color(cascade_idx: u32) -> vec3<f32> {
    switch cascade_idx {
        case 0u: { return vec3<f32>(1.0, 0.0, 0.0); } // Red
        case 1u: { return vec3<f32>(0.0, 1.0, 0.0); } // Green
        case 2u: { return vec3<f32>(0.0, 0.0, 1.0); } // Blue
        case 3u: { return vec3<f32>(1.0, 1.0, 0.0); } // Yellow
        default: { return vec3<f32>(1.0, 1.0, 1.0); } // White
    }
}
"#;

// ============================================================================
// Shadow map utilities
// ============================================================================

/// Stabilize a shadow map by snapping to texel boundaries.
///
/// Prevents shadow swimming when the camera moves by aligning the
/// orthographic projection to shadow map texel grid.
pub fn stabilize_shadow_matrix(
    light_vp: Mat4,
    shadow_map_size: u32,
) -> Mat4 {
    // Get the origin in shadow space.
    let origin = light_vp * Vec4::new(0.0, 0.0, 0.0, 1.0);
    let origin_shadow = Vec2::new(origin.x / origin.w, origin.y / origin.w);

    // Compute texel size in shadow space.
    let half_size = shadow_map_size as f32 * 0.5;
    let texel_x = origin_shadow.x * half_size;
    let texel_y = origin_shadow.y * half_size;

    let offset_x = (texel_x.round() - texel_x) / half_size;
    let offset_y = (texel_y.round() - texel_y) / half_size;

    // Apply offset to the projection matrix.
    let offset_matrix = Mat4::from_translation(Vec3::new(offset_x, offset_y, 0.0));

    offset_matrix * light_vp
}

/// Compute a tight-fitting orthographic projection for a directional light
/// based on the scene's AABB.
pub fn compute_scene_shadow_projection(
    light_dir: Vec3,
    scene_min: Vec3,
    scene_max: Vec3,
    padding: f32,
) -> Mat4 {
    let light_up = if light_dir.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };

    let scene_center = (scene_min + scene_max) * 0.5;
    let scene_radius = (scene_max - scene_min).length() * 0.5;

    let light_view = Mat4::look_at_rh(
        scene_center - light_dir * (scene_radius + padding),
        scene_center,
        light_up,
    );

    let light_proj = Mat4::orthographic_rh(
        -scene_radius - padding,
        scene_radius + padding,
        -scene_radius - padding,
        scene_radius + padding,
        0.1,
        scene_radius * 2.0 + padding * 2.0,
    );

    light_proj * light_view
}

/// Compute the view-space depth of a world-space point.
pub fn world_to_view_depth(camera_view: Mat4, world_pos: Vec3) -> f32 {
    let view_pos = camera_view * Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);
    -view_pos.z // View space Z is negative (looking down -Z).
}

// ============================================================================
// Spot light shadow
// ============================================================================

/// A shadow map for a single spot light.
pub struct SpotLightShadow {
    /// Depth texture.
    pub texture: wgpu::Texture,
    /// Depth view for rendering.
    pub depth_view: wgpu::TextureView,
    /// Sample view for PBR shader.
    pub sample_view: wgpu::TextureView,
    /// Light VP uniform buffer.
    pub vp_buffer: wgpu::Buffer,
    /// VP bind group.
    pub vp_bind_group: wgpu::BindGroup,
    /// Light view-projection matrix.
    pub light_vp: Mat4,
}

impl SpotLightShadow {
    /// Create a new spot light shadow map.
    pub fn new(
        device: &wgpu::Device,
        resolution: u32,
        light_vp_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("spot_shadow_depth"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let depth_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("spot_shadow_depth_view"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });

        let sample_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("spot_shadow_sample_view"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });

        let vp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spot_shadow_vp"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vp_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spot_shadow_vp_bg"),
            layout: light_vp_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: vp_buffer.as_entire_binding(),
            }],
        });

        Self {
            texture,
            depth_view,
            sample_view,
            vp_buffer,
            vp_bind_group,
            light_vp: Mat4::IDENTITY,
        }
    }

    /// Update the spot light shadow VP.
    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        light_pos: Vec3,
        light_dir: Vec3,
        fov: f32,
        near: f32,
        far: f32,
    ) {
        self.light_vp =
            ShadowMapRenderer::compute_spot_light_vp(light_pos, light_dir, fov, near, far);
        queue.write_buffer(
            &self.vp_buffer,
            0,
            bytemuck::bytes_of(&self.light_vp.to_cols_array_2d()),
        );
    }
}

// ============================================================================
// Point light cubemap shadow
// ============================================================================

/// A cubemap shadow map for a point light.
pub struct PointLightCubemapShadow {
    /// 6 face depth textures.
    pub face_textures: [wgpu::Texture; 6],
    /// 6 face depth views.
    pub face_depth_views: [wgpu::TextureView; 6],
    /// 6 face sample views.
    pub face_sample_views: [wgpu::TextureView; 6],
    /// 6 face VP buffers.
    pub face_vp_buffers: [wgpu::Buffer; 6],
    /// 6 face VP bind groups.
    pub face_vp_bind_groups: [wgpu::BindGroup; 6],
    /// Light position.
    pub light_pos: Vec3,
    /// Near/far planes.
    pub near: f32,
    pub far: f32,
}

impl PointLightCubemapShadow {
    /// Create a new point light cubemap shadow.
    pub fn new(
        device: &wgpu::Device,
        resolution: u32,
        light_vp_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let face_names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"];

        let mut face_textures = Vec::with_capacity(6);
        let mut face_depth_views = Vec::with_capacity(6);
        let mut face_sample_views = Vec::with_capacity(6);
        let mut face_vp_buffers = Vec::with_capacity(6);
        let mut face_vp_bind_groups = Vec::with_capacity(6);

        for i in 0..6 {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("point_shadow_{}", face_names[i])),
                size: wgpu::Extent3d {
                    width: resolution,
                    height: resolution,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            let depth_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("point_shadow_{}_dv", face_names[i])),
                format: Some(wgpu::TextureFormat::Depth32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::DepthOnly,
                ..Default::default()
            });

            let sample_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("point_shadow_{}_sv", face_names[i])),
                format: Some(wgpu::TextureFormat::Depth32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::DepthOnly,
                ..Default::default()
            });

            let vp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("point_shadow_{}_vp", face_names[i])),
                size: 64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let vp_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("point_shadow_{}_vp_bg", face_names[i])),
                layout: light_vp_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vp_buffer.as_entire_binding(),
                }],
            });

            face_textures.push(texture);
            face_depth_views.push(depth_view);
            face_sample_views.push(sample_view);
            face_vp_buffers.push(vp_buffer);
            face_vp_bind_groups.push(vp_bind_group);
        }

        Self {
            face_textures: face_textures.try_into().unwrap_or_else(|_| panic!()),
            face_depth_views: face_depth_views.try_into().unwrap_or_else(|_| panic!()),
            face_sample_views: face_sample_views.try_into().unwrap_or_else(|_| panic!()),
            face_vp_buffers: face_vp_buffers.try_into().unwrap_or_else(|_| panic!()),
            face_vp_bind_groups: face_vp_bind_groups.try_into().unwrap_or_else(|_| panic!()),
            light_pos: Vec3::ZERO,
            near: 0.1,
            far: 100.0,
        }
    }

    /// Update the point light cubemap shadow matrices.
    pub fn update(&mut self, queue: &wgpu::Queue, light_pos: Vec3, near: f32, far: f32) {
        self.light_pos = light_pos;
        self.near = near;
        self.far = far;

        let vps = ShadowMapRenderer::compute_point_light_cubemap_vps(light_pos, near, far);

        for (i, vp) in vps.iter().enumerate() {
            queue.write_buffer(
                &self.face_vp_buffers[i],
                0,
                bytemuck::bytes_of(&vp.to_cols_array_2d()),
            );
        }
    }
}
