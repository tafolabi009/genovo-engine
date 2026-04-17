// engine/render/src/gpu_skeletal.rs
//
// GPU Skeletal Animation Rendering for the Genovo engine.
//
// # Features
//
// - `SkeletalMeshGpu`: vertex buffer with bone weights + bone indices
// - `SkinnedVertex`: position, normal, UV, bone_weights (vec4), bone_indices (uvec4)
// - `BonePaletteBuffer`: storage buffer of Mat4 bone matrices (max 256 bones)
// - WGSL skinned vertex shader with linear blend skinning
// - Dual quaternion skinning option for better joint twisting
// - Blend shape / morph target on GPU via additive vertex offsets
// - Skinned render pipeline with a different vertex layout than static mesh

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec2, Vec3, Vec4};

// ============================================================================
// Constants
// ============================================================================

/// Maximum number of bones per skeleton.
pub const MAX_BONES: usize = 256;

/// Maximum number of bone influences per vertex.
pub const MAX_BONE_INFLUENCES: usize = 4;

/// Maximum number of morph targets.
pub const MAX_MORPH_TARGETS: usize = 8;

/// Maximum number of morph target vertices (for the storage buffer).
pub const MAX_MORPH_VERTICES: usize = 65536;

// ============================================================================
// Skinned vertex type
// ============================================================================

/// GPU vertex layout for skeletal meshes.
///
/// Extends `SceneVertex` with bone weights and bone indices for skinning.
///
/// WGSL locations:
///   0: position    vec3<f32>
///   1: normal      vec3<f32>
///   2: uv          vec2<f32>
///   3: color       vec4<f32>
///   4: bone_weights vec4<f32>
///   5: bone_indices vec4<u32>
///   6: tangent     vec4<f32>
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SkinnedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    pub bone_weights: [f32; 4],
    pub bone_indices: [u32; 4],
    pub tangent: [f32; 4],
}

impl SkinnedVertex {
    /// Create a skinned vertex.
    pub fn new(
        position: Vec3,
        normal: Vec3,
        uv: Vec2,
        color: Vec4,
        bone_weights: Vec4,
        bone_indices: [u32; 4],
        tangent: Vec4,
    ) -> Self {
        Self {
            position: position.into(),
            normal: normal.into(),
            uv: uv.into(),
            color: color.into(),
            bone_weights: bone_weights.into(),
            bone_indices,
            tangent: tangent.into(),
        }
    }

    /// Create a skinned vertex with default values.
    pub fn with_pos_normal_uv(position: Vec3, normal: Vec3, uv: Vec2) -> Self {
        Self::new(
            position,
            normal,
            uv,
            Vec4::ONE,
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            [0, 0, 0, 0],
            Vec4::new(1.0, 0.0, 0.0, 1.0),
        )
    }

    /// The wgpu vertex buffer layout for skinned vertices.
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SkinnedVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position: vec3<f32>
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                // normal: vec3<f32>
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
                // uv: vec2<f32>
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 24,
                    shader_location: 2,
                },
                // color: vec4<f32>
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 3,
                },
                // bone_weights: vec4<f32>
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 48,
                    shader_location: 4,
                },
                // bone_indices: vec4<u32>
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32x4,
                    offset: 64,
                    shader_location: 5,
                },
                // tangent: vec4<f32>
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 80,
                    shader_location: 6,
                },
            ],
        }
    }
}

// ============================================================================
// Bone palette uniform
// ============================================================================

/// Bone palette: array of transformation matrices for all bones in a skeleton.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BonePaletteUniform {
    /// Bone transformation matrices (bind pose inverse * current pose).
    pub bones: [[[f32; 4]; 4]; MAX_BONES],
}

impl Default for BonePaletteUniform {
    fn default() -> Self {
        let identity = Mat4::IDENTITY.to_cols_array_2d();
        Self {
            bones: [identity; MAX_BONES],
        }
    }
}

/// Dual quaternion bone data for DQS (dual quaternion skinning).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DualQuaternionBone {
    /// Real part of the dual quaternion (rotation).
    pub real: [f32; 4],
    /// Dual part (translation encoded).
    pub dual: [f32; 4],
}

impl Default for DualQuaternionBone {
    fn default() -> Self {
        Self {
            real: [0.0, 0.0, 0.0, 1.0], // Identity quaternion.
            dual: [0.0; 4],
        }
    }
}

/// Dual quaternion bone palette.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DualQuaternionPaletteUniform {
    pub bones: [DualQuaternionBone; MAX_BONES],
}

impl Default for DualQuaternionPaletteUniform {
    fn default() -> Self {
        Self {
            bones: [DualQuaternionBone::default(); MAX_BONES],
        }
    }
}

/// Morph target parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MorphTargetParams {
    /// Morph target weights (up to 8).
    pub weights: [f32; 8],
    /// .x = num_targets, .y = num_vertices, .z = unused, .w = unused.
    pub morph_info: [f32; 4],
    /// Padding for alignment.
    pub _pad: [f32; 4],
}

impl Default for MorphTargetParams {
    fn default() -> Self {
        Self {
            weights: [0.0; 8],
            morph_info: [0.0; 4],
            _pad: [0.0; 4],
        }
    }
}

// ============================================================================
// Skinned vertex shader -- Linear Blend Skinning (LBS)
// ============================================================================

/// WGSL shader for GPU skeletal animation with Linear Blend Skinning.
pub const SKINNED_VERTEX_SHADER_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Skinned Vertex Shader (Linear Blend Skinning)
// ============================================================================
//
// Transforms vertices using a weighted combination of bone matrices.
// Each vertex is influenced by up to 4 bones.
//
// Bind groups:
//   Group 0: Camera + Lights
//   Group 1: Model
//   Group 2: Material
//   Group 3: Bone palette (storage buffer of mat4x4)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PI: f32 = 3.14159265359;
const INV_PI: f32 = 0.31830988618;
const EPSILON: f32 = 0.0001;
const MAX_LIGHTS: u32 = 8u;
const GAMMA: f32 = 2.2;
const INV_GAMMA: f32 = 0.45454545454;

const LIGHT_TYPE_DISABLED: f32 = 0.0;
const LIGHT_TYPE_DIRECTIONAL: f32 = 1.0;
const LIGHT_TYPE_POINT: f32 = 2.0;

// ---------------------------------------------------------------------------
// Uniform structs
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

// ---------------------------------------------------------------------------
// Bind groups
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> lights: LightsUniform;
@group(1) @binding(0) var<uniform> model: ModelUniform;
@group(2) @binding(0) var<uniform> material: MaterialUniform;

// Bone palette as a storage buffer (for large bone counts).
@group(3) @binding(0) var<storage, read> bone_matrices: array<mat4x4<f32>, 256>;

// ---------------------------------------------------------------------------
// Vertex input / output
// ---------------------------------------------------------------------------

struct SkinnedVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) bone_weights: vec4<f32>,
    @location(5) bone_indices: vec4<u32>,
    @location(6) tangent: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) vertex_color: vec4<f32>,
    @location(4) view_dir: vec3<f32>,
    @location(5) world_tangent: vec3<f32>,
    @location(6) world_bitangent: vec3<f32>,
};

// ---------------------------------------------------------------------------
// Skinning function
// ---------------------------------------------------------------------------

// Apply linear blend skinning to a position.
fn skin_position(
    position: vec3<f32>,
    weights: vec4<f32>,
    indices: vec4<u32>,
) -> vec4<f32> {
    var skinned = vec4<f32>(0.0);

    // Bone 0
    let bone0 = bone_matrices[indices.x];
    skinned = skinned + weights.x * (bone0 * vec4<f32>(position, 1.0));

    // Bone 1
    let bone1 = bone_matrices[indices.y];
    skinned = skinned + weights.y * (bone1 * vec4<f32>(position, 1.0));

    // Bone 2
    let bone2 = bone_matrices[indices.z];
    skinned = skinned + weights.z * (bone2 * vec4<f32>(position, 1.0));

    // Bone 3
    let bone3 = bone_matrices[indices.w];
    skinned = skinned + weights.w * (bone3 * vec4<f32>(position, 1.0));

    return skinned;
}

// Apply linear blend skinning to a normal (direction vector).
fn skin_normal(
    normal: vec3<f32>,
    weights: vec4<f32>,
    indices: vec4<u32>,
) -> vec3<f32> {
    var skinned = vec3<f32>(0.0);

    // For normals, we use the inverse-transpose of each bone matrix.
    // For orthogonal matrices (pure rotation + uniform scale), the
    // matrix itself can be used. We assume bones are rigid transforms.
    let bone0 = bone_matrices[indices.x];
    skinned = skinned + weights.x * (bone0 * vec4<f32>(normal, 0.0)).xyz;

    let bone1 = bone_matrices[indices.y];
    skinned = skinned + weights.y * (bone1 * vec4<f32>(normal, 0.0)).xyz;

    let bone2 = bone_matrices[indices.z];
    skinned = skinned + weights.z * (bone2 * vec4<f32>(normal, 0.0)).xyz;

    let bone3 = bone_matrices[indices.w];
    skinned = skinned + weights.w * (bone3 * vec4<f32>(normal, 0.0)).xyz;

    return normalize(skinned);
}

// Apply linear blend skinning to a tangent vector.
fn skin_tangent(
    tangent: vec3<f32>,
    weights: vec4<f32>,
    indices: vec4<u32>,
) -> vec3<f32> {
    var skinned = vec3<f32>(0.0);

    let bone0 = bone_matrices[indices.x];
    skinned = skinned + weights.x * (bone0 * vec4<f32>(tangent, 0.0)).xyz;

    let bone1 = bone_matrices[indices.y];
    skinned = skinned + weights.y * (bone1 * vec4<f32>(tangent, 0.0)).xyz;

    let bone2 = bone_matrices[indices.z];
    skinned = skinned + weights.z * (bone2 * vec4<f32>(tangent, 0.0)).xyz;

    let bone3 = bone_matrices[indices.w];
    skinned = skinned + weights.w * (bone3 * vec4<f32>(tangent, 0.0)).xyz;

    return normalize(skinned);
}

// ---------------------------------------------------------------------------
// Vertex shader
// ---------------------------------------------------------------------------

@vertex
fn vs_skinned(input: SkinnedVertexInput) -> VertexOutput {
    var output: VertexOutput;

    // Normalize bone weights.
    let total_weight = input.bone_weights.x + input.bone_weights.y +
                       input.bone_weights.z + input.bone_weights.w;
    var weights = input.bone_weights;
    if total_weight > 0.0 {
        weights = weights / total_weight;
    } else {
        weights = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    }

    // Apply skinning.
    let skinned_pos = skin_position(input.position, weights, input.bone_indices);
    let skinned_normal = skin_normal(input.normal, weights, input.bone_indices);
    let skinned_tangent = skin_tangent(input.tangent.xyz, weights, input.bone_indices);

    // Transform to world space.
    let world_pos = model.world * skinned_pos;
    output.world_position = world_pos.xyz;
    output.clip_position = camera.view_projection * world_pos;

    // Transform normal and tangent.
    let world_normal = normalize((model.normal_matrix * vec4<f32>(skinned_normal, 0.0)).xyz);
    output.world_normal = world_normal;

    let world_tangent = normalize((model.world * vec4<f32>(skinned_tangent, 0.0)).xyz);
    output.world_tangent = world_tangent;
    output.world_bitangent = normalize(cross(world_normal, world_tangent) * input.tangent.w);

    output.uv = input.uv;
    output.vertex_color = input.color;
    output.view_dir = normalize(camera.camera_position.xyz - world_pos.xyz);

    return output;
}

// ---------------------------------------------------------------------------
// PBR lighting functions (same as base shader)
// ---------------------------------------------------------------------------

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let t = clamp(1.0 - cos_theta, 0.0, 1.0);
    let t2 = t * t;
    let t5 = t2 * t2 * t;
    return f0 + (vec3<f32>(1.0) - f0) * t5;
}

fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let t = clamp(1.0 - cos_theta, 0.0, 1.0);
    let t2 = t * t;
    let t5 = t2 * t2 * t;
    let max_reflect = vec3<f32>(1.0 - roughness);
    return f0 + (max(max_reflect, f0) - f0) * t5;
}

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom_term = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom_term * denom_term + EPSILON);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k + EPSILON);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn attenuation_smooth(distance: f32, range: f32) -> f32 {
    if range <= 0.0 { return 1.0; }
    let d = distance / range;
    let d2 = d * d;
    let d4 = d2 * d2;
    let factor = clamp(1.0 - d4, 0.0, 1.0);
    return factor * factor / (distance * distance + 1.0);
}

fn compute_f0(albedo: vec3<f32>, metallic: f32, reflectance: f32) -> vec3<f32> {
    let dielectric_f0 = vec3<f32>(0.16 * reflectance * reflectance);
    return mix(dielectric_f0, albedo, metallic);
}

fn hemisphere_ambient(normal: vec3<f32>, ambient_color: vec3<f32>, ambient_intensity: f32) -> vec3<f32> {
    let sky_color = ambient_color * ambient_intensity;
    let ground_color = sky_color * vec3<f32>(0.6, 0.5, 0.4) * 0.5;
    let t = normal.y * 0.5 + 0.5;
    return mix(ground_color, sky_color, t);
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 { return c * 12.92; }
    return 1.055 * pow(c, INV_GAMMA) - 0.055;
}

fn tone_map_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 1.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + vec3<f32>(b))) / (x * (c * x + vec3<f32>(d)) + vec3<f32>(e)),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

// Light contribution types.
struct LightContribution { diffuse: vec3<f32>, specular: vec3<f32>, };

fn compute_directional_skinned(
    light: GpuLight, normal: vec3<f32>, view_dir: vec3<f32>,
    albedo: vec3<f32>, metallic: f32, roughness: f32, f0: vec3<f32>,
) -> LightContribution {
    var result: LightContribution;
    result.diffuse = vec3<f32>(0.0); result.specular = vec3<f32>(0.0);
    let light_dir = normalize(-light.position_or_direction.xyz);
    let light_color = light.color_intensity.xyz * light.color_intensity.w;
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    if n_dot_l <= 0.0 { return result; }
    let half_vec = normalize(view_dir + light_dir);
    let n_dot_v = max(dot(normal, view_dir), EPSILON);
    let n_dot_h = max(dot(normal, half_vec), 0.0);
    let v_dot_h = max(dot(view_dir, half_vec), 0.0);
    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(v_dot_h, f0);
    let spec_brdf = (d * g * f) / (4.0 * n_dot_v * n_dot_l + EPSILON);
    let k_s = f;
    let k_d = (vec3<f32>(1.0) - k_s) * (1.0 - metallic);
    result.diffuse = k_d * albedo * INV_PI * light_color * n_dot_l;
    result.specular = spec_brdf * light_color * n_dot_l;
    return result;
}

fn compute_point_skinned(
    light: GpuLight, world_pos: vec3<f32>, normal: vec3<f32>, view_dir: vec3<f32>,
    albedo: vec3<f32>, metallic: f32, roughness: f32, f0: vec3<f32>,
) -> LightContribution {
    var result: LightContribution;
    result.diffuse = vec3<f32>(0.0); result.specular = vec3<f32>(0.0);
    let to_light = light.position_or_direction.xyz - world_pos;
    let distance = length(to_light);
    let light_range = light.position_or_direction.w;
    if light_range > 0.0 && distance > light_range * 1.2 { return result; }
    let light_dir = normalize(to_light);
    let light_color = light.color_intensity.xyz * light.color_intensity.w;
    let att = attenuation_smooth(distance, light_range);
    if att < EPSILON { return result; }
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    if n_dot_l <= 0.0 { return result; }
    let half_vec = normalize(view_dir + light_dir);
    let n_dot_v = max(dot(normal, view_dir), EPSILON);
    let n_dot_h = max(dot(normal, half_vec), 0.0);
    let v_dot_h = max(dot(view_dir, half_vec), 0.0);
    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(v_dot_h, f0);
    let spec_brdf = (d * g * f) / (4.0 * n_dot_v * n_dot_l + EPSILON);
    let k_s = f;
    let k_d = (vec3<f32>(1.0) - k_s) * (1.0 - metallic);
    let attenuated = light_color * att;
    result.diffuse = k_d * albedo * INV_PI * attenuated * n_dot_l;
    result.specular = spec_brdf * attenuated * n_dot_l;
    return result;
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------

@fragment
fn fs_skinned(input: VertexOutput) -> @location(0) vec4<f32> {
    let albedo = material.albedo_color.xyz * input.vertex_color.xyz;
    let alpha = material.albedo_color.w * input.vertex_color.w;
    let metallic = clamp(material.metallic_roughness.x, 0.0, 1.0);
    let roughness = clamp(material.metallic_roughness.y, 0.04, 1.0);
    let reflectance = material.metallic_roughness.z;
    let f0 = compute_f0(albedo, metallic, reflectance);
    let normal = normalize(input.world_normal);
    let view_dir = normalize(input.view_dir);

    var total_diffuse = vec3<f32>(0.0);
    var total_specular = vec3<f32>(0.0);
    let num_lights = u32(lights.light_count.x);

    for (var i = 0u; i < MAX_LIGHTS; i = i + 1u) {
        if i >= num_lights { break; }
        let light = lights.lights[i];
        let light_type = light.params.x;
        if abs(light_type - LIGHT_TYPE_DIRECTIONAL) < 0.5 {
            let c = compute_directional_skinned(light, normal, view_dir, albedo, metallic, roughness, f0);
            total_diffuse = total_diffuse + c.diffuse;
            total_specular = total_specular + c.specular;
        } else if abs(light_type - LIGHT_TYPE_POINT) < 0.5 {
            let c = compute_point_skinned(light, input.world_position, normal, view_dir, albedo, metallic, roughness, f0);
            total_diffuse = total_diffuse + c.diffuse;
            total_specular = total_specular + c.specular;
        }
    }

    let ambient = hemisphere_ambient(normal, lights.ambient.xyz, lights.ambient.w);
    let n_dot_v_a = max(dot(normal, view_dir), 0.0);
    let f_a = fresnel_schlick_roughness(n_dot_v_a, f0, roughness);
    let k_d_a = (vec3<f32>(1.0) - f_a) * (1.0 - metallic);
    let ambient_d = k_d_a * albedo * ambient;
    let ambient_s = f_a * ambient * 0.2;
    let emissive = material.emissive.xyz * material.emissive.w;

    var final_color = total_diffuse + total_specular + ambient_d + ambient_s + emissive;
    final_color = tone_map_aces(final_color);
    final_color = vec3<f32>(linear_to_srgb(final_color.x), linear_to_srgb(final_color.y), linear_to_srgb(final_color.z));

    return vec4<f32>(final_color, alpha);
}
"#;

// ============================================================================
// Dual Quaternion Skinning shader
// ============================================================================

/// WGSL shader for Dual Quaternion Skinning (DQS).
///
/// DQS produces better results than LBS for twisting joints (e.g., forearms).
pub const DQS_SHADER_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Dual Quaternion Skinning
// ============================================================================
//
// Uses dual quaternions instead of matrices for smoother joint deformation.
// Avoids the "candy wrapper" artifact of linear blend skinning.

struct CameraUniform {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
};

struct ModelUniform {
    world: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
};

struct DualQuat {
    real: vec4<f32>,
    dual: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> model: ModelUniform;
@group(3) @binding(0) var<storage, read> bone_dqs: array<DualQuat, 256>;

struct SkinnedVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) bone_weights: vec4<f32>,
    @location(5) bone_indices: vec4<u32>,
    @location(6) tangent: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) vertex_color: vec4<f32>,
    @location(4) view_dir: vec3<f32>,
    @location(5) world_tangent: vec3<f32>,
    @location(6) world_bitangent: vec3<f32>,
};

// Quaternion multiplication.
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

// Conjugate of a quaternion.
fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

// Rotate a vector by a quaternion.
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = vec4<f32>(v, 0.0);
    let result = quat_mul(q, quat_mul(qv, quat_conj(q)));
    return result.xyz;
}

// Extract translation from a dual quaternion.
fn dq_translation(real: vec4<f32>, dual: vec4<f32>) -> vec3<f32> {
    let t = quat_mul(dual * 2.0, quat_conj(real));
    return t.xyz;
}

// Apply DQS to a position.
fn dqs_transform_position(
    position: vec3<f32>,
    weights: vec4<f32>,
    indices: vec4<u32>,
) -> vec3<f32> {
    // Blend dual quaternions.
    let dq0 = bone_dqs[indices.x];
    var blended_real = dq0.real * weights.x;
    var blended_dual = dq0.dual * weights.x;

    // Ensure shortest path by checking dot product with first bone.
    let dq1 = bone_dqs[indices.y];
    let sign1 = select(-1.0, 1.0, dot(dq0.real, dq1.real) >= 0.0);
    blended_real = blended_real + dq1.real * weights.y * sign1;
    blended_dual = blended_dual + dq1.dual * weights.y * sign1;

    let dq2 = bone_dqs[indices.z];
    let sign2 = select(-1.0, 1.0, dot(dq0.real, dq2.real) >= 0.0);
    blended_real = blended_real + dq2.real * weights.z * sign2;
    blended_dual = blended_dual + dq2.dual * weights.z * sign2;

    let dq3 = bone_dqs[indices.w];
    let sign3 = select(-1.0, 1.0, dot(dq0.real, dq3.real) >= 0.0);
    blended_real = blended_real + dq3.real * weights.w * sign3;
    blended_dual = blended_dual + dq3.dual * weights.w * sign3;

    // Normalize.
    let norm = length(blended_real);
    blended_real = blended_real / norm;
    blended_dual = blended_dual / norm;

    // Apply rotation + translation.
    let rotated = quat_rotate(blended_real, position);
    let translation = dq_translation(blended_real, blended_dual);

    return rotated + translation;
}

// Apply DQS to a normal.
fn dqs_transform_normal(
    normal: vec3<f32>,
    weights: vec4<f32>,
    indices: vec4<u32>,
) -> vec3<f32> {
    let dq0 = bone_dqs[indices.x];
    var blended_real = dq0.real * weights.x;

    let dq1 = bone_dqs[indices.y];
    let sign1 = select(-1.0, 1.0, dot(dq0.real, dq1.real) >= 0.0);
    blended_real = blended_real + dq1.real * weights.y * sign1;

    let dq2 = bone_dqs[indices.z];
    let sign2 = select(-1.0, 1.0, dot(dq0.real, dq2.real) >= 0.0);
    blended_real = blended_real + dq2.real * weights.z * sign2;

    let dq3 = bone_dqs[indices.w];
    let sign3 = select(-1.0, 1.0, dot(dq0.real, dq3.real) >= 0.0);
    blended_real = blended_real + dq3.real * weights.w * sign3;

    blended_real = normalize(blended_real);

    return normalize(quat_rotate(blended_real, normal));
}

@vertex
fn vs_dqs(input: SkinnedVertexInput) -> VertexOutput {
    var output: VertexOutput;

    var weights = input.bone_weights;
    let total = weights.x + weights.y + weights.z + weights.w;
    if total > 0.0 { weights = weights / total; }
    else { weights = vec4<f32>(1.0, 0.0, 0.0, 0.0); }

    let skinned_pos = dqs_transform_position(input.position, weights, input.bone_indices);
    let skinned_normal = dqs_transform_normal(input.normal, weights, input.bone_indices);
    let skinned_tangent = dqs_transform_normal(input.tangent.xyz, weights, input.bone_indices);

    let world_pos = model.world * vec4<f32>(skinned_pos, 1.0);
    output.world_position = world_pos.xyz;
    output.clip_position = camera.view_projection * world_pos;
    output.world_normal = normalize((model.normal_matrix * vec4<f32>(skinned_normal, 0.0)).xyz);
    output.world_tangent = normalize((model.world * vec4<f32>(skinned_tangent, 0.0)).xyz);
    output.world_bitangent = normalize(cross(output.world_normal, output.world_tangent) * input.tangent.w);
    output.uv = input.uv;
    output.vertex_color = input.color;
    output.view_dir = normalize(camera.camera_position.xyz - world_pos.xyz);

    return output;
}
"#;

// ============================================================================
// Morph target shader
// ============================================================================

/// WGSL compute shader for morph target application.
pub const MORPH_TARGET_COMPUTE_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Morph Target Compute Shader
// ============================================================================
//
// Applies morph target (blend shape) offsets to vertex positions and normals.
// Runs before the skinning vertex shader.

struct MorphParams {
    weights: array<f32, 8>,
    morph_info: vec4<f32>,
    _pad: vec4<f32>,
};

struct MorphDelta {
    position_delta: vec4<f32>,
    normal_delta: vec4<f32>,
};

@group(0) @binding(0) var<uniform> morph_params: MorphParams;
@group(0) @binding(1) var<storage, read> morph_deltas: array<MorphDelta>;
@group(0) @binding(2) var<storage, read_write> vertex_positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> vertex_normals: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> base_positions: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> base_normals: array<vec4<f32>>;

@compute @workgroup_size(64)
fn cs_morph_target(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vertex_idx = global_id.x;
    let num_vertices = u32(morph_params.morph_info.y);
    let num_targets = u32(morph_params.morph_info.x);

    if vertex_idx >= num_vertices {
        return;
    }

    // Start from base pose.
    var position = base_positions[vertex_idx].xyz;
    var normal = base_normals[vertex_idx].xyz;

    // Accumulate morph target deltas.
    for (var t = 0u; t < num_targets; t = t + 1u) {
        let weight = morph_params.weights[t];
        if abs(weight) < 0.0001 {
            continue;
        }

        let delta_idx = t * num_vertices + vertex_idx;
        let delta = morph_deltas[delta_idx];

        position = position + delta.position_delta.xyz * weight;
        normal = normal + delta.normal_delta.xyz * weight;
    }

    vertex_positions[vertex_idx] = vec4<f32>(position, 1.0);
    vertex_normals[vertex_idx] = vec4<f32>(normalize(normal), 0.0);
}
"#;

// ============================================================================
// Skeletal mesh GPU data
// ============================================================================

/// A skeletal mesh uploaded to the GPU.
pub struct SkeletalMeshGpu {
    /// Vertex buffer with skinned vertices.
    pub vertex_buffer: wgpu::Buffer,
    /// Index buffer.
    pub index_buffer: wgpu::Buffer,
    /// Number of indices.
    pub index_count: u32,
    /// Number of vertices.
    pub vertex_count: u32,
}

impl SkeletalMeshGpu {
    /// Upload a skinned mesh to the GPU.
    pub fn upload(
        device: &wgpu::Device,
        vertices: &[SkinnedVertex],
        indices: &[u32],
        label: &str,
    ) -> Self {
        use wgpu::util::DeviceExt;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{}_skinned_vb", label)),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{}_skinned_ib", label)),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            vertex_count: vertices.len() as u32,
        }
    }
}

// ============================================================================
// Bone palette buffer
// ============================================================================

/// GPU buffer for bone matrices, updated each frame.
pub struct BonePaletteBuffer {
    /// The GPU storage buffer containing bone matrices.
    pub buffer: wgpu::Buffer,
    /// Bind group for the bone palette.
    pub bind_group: wgpu::BindGroup,
    /// Bind group layout.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Number of active bones.
    pub bone_count: usize,
}

impl BonePaletteBuffer {
    /// Create a new bone palette buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer_size = (MAX_BONES * 64) as u64; // 256 * mat4x4 (64 bytes each)

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bone_palette_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bone_palette_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bone_palette_bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        Self {
            buffer,
            bind_group,
            bind_group_layout,
            bone_count: 0,
        }
    }

    /// Upload bone matrices to the GPU.
    pub fn upload_bone_matrices(&mut self, queue: &wgpu::Queue, matrices: &[Mat4]) {
        self.bone_count = matrices.len().min(MAX_BONES);
        let mut data = BonePaletteUniform::default();
        for (i, mat) in matrices.iter().take(MAX_BONES).enumerate() {
            data.bones[i] = mat.to_cols_array_2d();
        }
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&data));
    }

    /// Upload dual quaternion bone data.
    pub fn upload_dual_quaternions(
        &mut self,
        queue: &wgpu::Queue,
        dqs: &[(Quat, Vec3)],
    ) {
        self.bone_count = dqs.len().min(MAX_BONES);
        let mut data = DualQuaternionPaletteUniform::default();

        for (i, (rotation, translation)) in dqs.iter().take(MAX_BONES).enumerate() {
            let r = *rotation;
            data.bones[i].real = [r.x, r.y, r.z, r.w];

            // Dual part: 0.5 * t * r
            let t_quat = Quat::from_xyzw(
                translation.x,
                translation.y,
                translation.z,
                0.0,
            );
            let dual = t_quat * r * 0.5;
            data.bones[i].dual = [dual.x, dual.y, dual.z, dual.w];
        }

        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&data));
    }
}

// ============================================================================
// Skinned render pipeline
// ============================================================================

/// Create the skinned mesh render pipeline.
pub fn create_skinned_pipeline(
    device: &wgpu::Device,
    camera_lights_bgl: &wgpu::BindGroupLayout,
    model_bgl: &wgpu::BindGroupLayout,
    material_bgl: &wgpu::BindGroupLayout,
    bone_palette_bgl: &wgpu::BindGroupLayout,
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("skinned_pbr_shader"),
        source: wgpu::ShaderSource::Wgsl(SKINNED_VERTEX_SHADER_WGSL.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("skinned_pipeline_layout"),
        bind_group_layouts: &[camera_lights_bgl, model_bgl, material_bgl, bone_palette_bgl],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("skinned_pbr_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader_module,
            entry_point: Some("vs_skinned"),
            buffers: &[SkinnedVertex::buffer_layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_module,
            entry_point: Some("fs_skinned"),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
}

/// Create the morph target compute pipeline.
pub fn create_morph_target_pipeline(device: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("morph_target_compute_shader"),
        source: wgpu::ShaderSource::Wgsl(MORPH_TARGET_COMPUTE_WGSL.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("morph_target_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("morph_target_pipeline_layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("morph_target_compute_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("cs_morph_target"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    (pipeline, bgl)
}

// ============================================================================
// Skeleton hierarchy utilities
// ============================================================================

/// A bone in the skeleton hierarchy.
#[derive(Debug, Clone)]
pub struct Bone {
    /// Bone name.
    pub name: String,
    /// Parent bone index (-1 for root).
    pub parent_index: i32,
    /// Inverse bind pose matrix.
    pub inverse_bind_pose: Mat4,
    /// Local transform (relative to parent).
    pub local_transform: Mat4,
}

/// A complete skeleton with bone hierarchy.
#[derive(Debug, Clone)]
pub struct Skeleton {
    pub bones: Vec<Bone>,
}

impl Skeleton {
    /// Create a new empty skeleton.
    pub fn new() -> Self {
        Self { bones: Vec::new() }
    }

    /// Add a bone to the skeleton.
    pub fn add_bone(
        &mut self,
        name: &str,
        parent_index: i32,
        inverse_bind_pose: Mat4,
        local_transform: Mat4,
    ) -> usize {
        let idx = self.bones.len();
        self.bones.push(Bone {
            name: name.to_string(),
            parent_index,
            inverse_bind_pose,
            local_transform,
        });
        idx
    }

    /// Compute the final bone matrices for skinning.
    ///
    /// `pose_transforms` contains the current local transforms for each bone
    /// (e.g., from animation). Returns the final palette of bone matrices.
    pub fn compute_bone_palette(&self, pose_transforms: &[Mat4]) -> Vec<Mat4> {
        let bone_count = self.bones.len();
        let mut world_transforms = vec![Mat4::IDENTITY; bone_count];
        let mut palette = vec![Mat4::IDENTITY; bone_count];

        // Forward kinematics: compute world transform for each bone.
        for i in 0..bone_count {
            let local = if i < pose_transforms.len() {
                pose_transforms[i]
            } else {
                self.bones[i].local_transform
            };

            let parent = self.bones[i].parent_index;
            if parent >= 0 && (parent as usize) < bone_count {
                world_transforms[i] = world_transforms[parent as usize] * local;
            } else {
                world_transforms[i] = local;
            }

            // Final palette matrix = world_transform * inverse_bind_pose.
            palette[i] = world_transforms[i] * self.bones[i].inverse_bind_pose;
        }

        palette
    }

    /// Find a bone by name.
    pub fn find_bone(&self, name: &str) -> Option<usize> {
        self.bones.iter().position(|b| b.name == name)
    }

    /// Get the number of bones.
    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }
}

// ============================================================================
// Animation clip
// ============================================================================

/// A keyframe for a single bone.
#[derive(Debug, Clone)]
pub struct BoneKeyframe {
    /// Time in seconds.
    pub time: f32,
    /// Local transform at this keyframe.
    pub transform: Mat4,
}

/// An animation clip containing keyframes for each bone.
#[derive(Debug, Clone)]
pub struct AnimationClip {
    /// Clip name.
    pub name: String,
    /// Duration in seconds.
    pub duration: f32,
    /// Keyframes per bone (indexed by bone index).
    pub bone_tracks: Vec<Vec<BoneKeyframe>>,
    /// Whether the animation loops.
    pub looping: bool,
}

impl AnimationClip {
    /// Sample the animation at a given time, producing local transforms for each bone.
    pub fn sample(&self, time: f32) -> Vec<Mat4> {
        let t = if self.looping && self.duration > 0.0 {
            time % self.duration
        } else {
            time.min(self.duration)
        };

        let mut transforms = Vec::with_capacity(self.bone_tracks.len());

        for track in &self.bone_tracks {
            if track.is_empty() {
                transforms.push(Mat4::IDENTITY);
                continue;
            }

            if track.len() == 1 || t <= track[0].time {
                transforms.push(track[0].transform);
                continue;
            }

            let last = &track[track.len() - 1];
            if t >= last.time {
                transforms.push(last.transform);
                continue;
            }

            // Find the two keyframes to interpolate between.
            let mut idx = 0;
            for i in 0..track.len() - 1 {
                if t >= track[i].time && t < track[i + 1].time {
                    idx = i;
                    break;
                }
            }

            let k0 = &track[idx];
            let k1 = &track[idx + 1];
            let factor = (t - k0.time) / (k1.time - k0.time);

            // Simple linear interpolation of matrix columns.
            // In production, this should decompose into TRS and use
            // slerp for rotation, but matrix lerp works for demonstration.
            let cols0 = k0.transform.to_cols_array();
            let cols1 = k1.transform.to_cols_array();
            let mut lerped = [0.0f32; 16];
            for i in 0..16 {
                lerped[i] = cols0[i] * (1.0 - factor) + cols1[i] * factor;
            }
            transforms.push(Mat4::from_cols_array(&lerped));
        }

        transforms
    }
}

// ============================================================================
// Animation player
// ============================================================================

/// Plays animations on a skeleton, blending between clips.
pub struct AnimationPlayer {
    /// Current animation clip index.
    pub current_clip: Option<usize>,
    /// Current playback time.
    pub current_time: f32,
    /// Playback speed multiplier.
    pub speed: f32,
    /// Whether playback is paused.
    pub paused: bool,
    /// Blend weight for transitioning between clips.
    pub blend_weight: f32,
    /// Previous clip for blending.
    pub previous_clip: Option<usize>,
    /// Previous clip time.
    pub previous_time: f32,
    /// Blend duration for transitions.
    pub blend_duration: f32,
    /// Elapsed blend time.
    pub blend_elapsed: f32,
}

impl AnimationPlayer {
    /// Create a new animation player.
    pub fn new() -> Self {
        Self {
            current_clip: None,
            current_time: 0.0,
            speed: 1.0,
            paused: false,
            blend_weight: 1.0,
            previous_clip: None,
            previous_time: 0.0,
            blend_duration: 0.3,
            blend_elapsed: 0.0,
        }
    }

    /// Play a clip by index.
    pub fn play(&mut self, clip_index: usize) {
        if self.current_clip == Some(clip_index) {
            return;
        }
        self.previous_clip = self.current_clip;
        self.previous_time = self.current_time;
        self.current_clip = Some(clip_index);
        self.current_time = 0.0;
        self.blend_elapsed = 0.0;
        self.blend_weight = 0.0;
    }

    /// Advance the animation by `dt` seconds.
    pub fn update(&mut self, dt: f32) {
        if self.paused {
            return;
        }

        self.current_time += dt * self.speed;

        // Blend transition.
        if self.previous_clip.is_some() {
            self.blend_elapsed += dt;
            self.blend_weight = (self.blend_elapsed / self.blend_duration).min(1.0);

            if self.blend_weight >= 1.0 {
                self.previous_clip = None;
                self.blend_weight = 1.0;
            }

            self.previous_time += dt * self.speed;
        }
    }

    /// Sample the current pose (with blending).
    pub fn sample_pose(&self, clips: &[AnimationClip]) -> Vec<Mat4> {
        let current = self.current_clip.and_then(|idx| clips.get(idx));
        let previous = self.previous_clip.and_then(|idx| clips.get(idx));

        match (current, previous) {
            (Some(curr), Some(prev)) => {
                let curr_pose = curr.sample(self.current_time);
                let prev_pose = prev.sample(self.previous_time);

                // Blend between previous and current.
                let len = curr_pose.len().max(prev_pose.len());
                let mut blended = Vec::with_capacity(len);
                for i in 0..len {
                    let c = curr_pose.get(i).copied().unwrap_or(Mat4::IDENTITY);
                    let p = prev_pose.get(i).copied().unwrap_or(Mat4::IDENTITY);

                    let cc = c.to_cols_array();
                    let pp = p.to_cols_array();
                    let mut lerped = [0.0f32; 16];
                    for j in 0..16 {
                        lerped[j] = pp[j] * (1.0 - self.blend_weight) + cc[j] * self.blend_weight;
                    }
                    blended.push(Mat4::from_cols_array(&lerped));
                }
                blended
            }
            (Some(curr), None) => curr.sample(self.current_time),
            _ => Vec::new(),
        }
    }
}
