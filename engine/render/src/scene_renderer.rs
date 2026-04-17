// engine/render/src/scene_renderer.rs
//
// Complete scene renderer for the Genovo engine. Replaces the single-triangle
// demo with a real GPU rendering pipeline capable of drawing meshes with
// materials, PBR lighting, and built-in primitives.
//
// # Architecture
//
// The pipeline is structured in layers:
//
// 1. **Vertex** -- `SceneVertex` with position, normal, UV, and vertex color.
//    Uploaded to wgpu vertex/index buffers via `MeshGpuData`.
// 2. **Material** -- `MaterialParams` (albedo, metallic, roughness, emissive)
//    packed into a uniform buffer, exposed through a bind group.
// 3. **Camera** -- view/projection matrices + world-space eye position,
//    stored in a uniform buffer at bind group 0.
// 4. **Lights** -- array of up to `MAX_LIGHTS` directional and point lights,
//    stored in a uniform buffer at bind group 0 alongside the camera.
// 5. **Model** -- per-draw-call world matrix, stored in a uniform buffer at
//    bind group 1.
// 6. **PBR shader** -- an embedded WGSL shader that performs N-dot-L diffuse
//    plus Blinn-Phong specular, ambient term, and iterates over the light
//    array.
// 7. **Primitive generation** -- CPU-side mesh generators for cube, sphere,
//    plane, cylinder, and cone.
// 8. **Grid** -- a dedicated line-based grid rendered as thin quads.
// 9. **SceneRenderManager** -- top-level owner of all GPU data, queues
//    `RenderObject` submissions, and draws everything in a single pass.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Constants
// ============================================================================

/// Maximum number of lights the shader supports in a single pass.
pub const MAX_LIGHTS: usize = 8;

/// Maximum number of grid lines in each direction.
pub const GRID_LINE_COUNT: usize = 41; // -20 .. +20

/// Default ambient light colour.
pub const DEFAULT_AMBIENT: [f32; 3] = [0.03, 0.03, 0.03];

// ============================================================================
// Vertex types
// ============================================================================

/// GPU vertex layout used by every mesh in the scene renderer.
///
/// Matches the WGSL struct `VertexInput`:
///   location 0: position  vec3<f32>
///   location 1: normal    vec3<f32>
///   location 2: uv        vec2<f32>
///   location 3: color     vec4<f32>
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SceneVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

impl SceneVertex {
    /// Create a vertex with full data.
    pub fn new(position: Vec3, normal: Vec3, uv: Vec2, color: Vec4) -> Self {
        Self {
            position: position.into(),
            normal: normal.into(),
            uv: uv.into(),
            color: color.into(),
        }
    }

    /// Create a vertex with white colour.
    pub fn with_pos_normal_uv(position: Vec3, normal: Vec3, uv: Vec2) -> Self {
        Self::new(position, normal, uv, Vec4::new(1.0, 1.0, 1.0, 1.0))
    }

    /// The wgpu vertex buffer layout descriptor for this vertex type.
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SceneVertex>() as wgpu::BufferAddress,
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
            ],
        }
    }
}

// ============================================================================
// GPU Uniform Types  (all #[repr(C)] + Pod + Zeroable for bytemuck)
// ============================================================================

/// Camera uniform data (bind group 0, binding 0).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    /// Column-major 4x4 view matrix.
    pub view: [[f32; 4]; 4],
    /// Column-major 4x4 projection matrix.
    pub projection: [[f32; 4]; 4],
    /// Column-major 4x4 view-projection matrix (precomputed).
    pub view_projection: [[f32; 4]; 4],
    /// Camera world-space position, w unused (padding).
    pub camera_position: [f32; 4],
}

impl CameraUniform {
    pub fn from_camera(camera: &SceneCamera) -> Self {
        let vp = camera.projection * camera.view;
        Self {
            view: camera.view.to_cols_array_2d(),
            projection: camera.projection.to_cols_array_2d(),
            view_projection: vp.to_cols_array_2d(),
            camera_position: [
                camera.position.x,
                camera.position.y,
                camera.position.z,
                1.0,
            ],
        }
    }
}

/// Single light entry in the light array uniform.
///
/// The shader interprets `light_type`:
///   0 = disabled / unused slot
///   1 = directional light (direction in `position_or_direction.xyz`)
///   2 = point light (position in `position_or_direction.xyz`, range in `.w`)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuLight {
    /// For directional: normalised direction.  For point: world position.
    /// .w = range for point lights (0 for directional).
    pub position_or_direction: [f32; 4],
    /// RGB colour, .w = intensity multiplier.
    pub color_intensity: [f32; 4],
    /// .x = light type (0/1/2), .yzw reserved.
    pub params: [f32; 4],
}

impl Default for GpuLight {
    fn default() -> Self {
        Self {
            position_or_direction: [0.0; 4],
            color_intensity: [0.0; 4],
            params: [0.0; 4],
        }
    }
}

/// Lights uniform buffer (bind group 0, binding 1).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LightsUniform {
    /// Ambient light colour, .w = ambient intensity.
    pub ambient: [f32; 4],
    /// Number of active lights, padding.
    pub light_count: [f32; 4],
    /// Light array.
    pub lights: [GpuLight; MAX_LIGHTS],
}

impl Default for LightsUniform {
    fn default() -> Self {
        Self {
            ambient: [DEFAULT_AMBIENT[0], DEFAULT_AMBIENT[1], DEFAULT_AMBIENT[2], 1.0],
            light_count: [0.0; 4],
            lights: [GpuLight::default(); MAX_LIGHTS],
        }
    }
}

/// Per-object model uniform (bind group 1, binding 0).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ModelUniform {
    /// Column-major 4x4 world (model) matrix.
    pub world: [[f32; 4]; 4],
    /// Column-major 4x4 normal matrix (inverse-transpose of the upper-left 3x3,
    /// padded to mat4 for alignment).
    pub normal_matrix: [[f32; 4]; 4],
}

impl ModelUniform {
    pub fn from_world_matrix(world: Mat4) -> Self {
        // Normal matrix = transpose of inverse of upper-left 3x3.
        // We store it as a mat4 for shader simplicity (pad with 0/1).
        let inv = world.inverse();
        let normal_mat = inv.transpose();
        Self {
            world: world.to_cols_array_2d(),
            normal_matrix: normal_mat.to_cols_array_2d(),
        }
    }
}

/// Material uniform (bind group 2, binding 0).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MaterialUniform {
    /// Base albedo colour (linear RGB + alpha).
    pub albedo_color: [f32; 4],
    /// .x = metallic, .y = roughness, .z = reflectance, .w = alpha_cutoff.
    pub metallic_roughness: [f32; 4],
    /// Emissive colour (linear RGB), .w = emissive strength.
    pub emissive: [f32; 4],
    /// .x = has_albedo_texture (0/1), .y = has_normal_map (0/1), .zw = reserved.
    pub flags: [f32; 4],
}

impl Default for MaterialUniform {
    fn default() -> Self {
        Self {
            albedo_color: [0.8, 0.8, 0.8, 1.0],
            metallic_roughness: [0.0, 0.5, 0.5, 0.0],
            emissive: [0.0; 4],
            flags: [0.0; 4],
        }
    }
}

// ============================================================================
// High-level scene types
// ============================================================================

/// Camera data for rendering.
#[derive(Debug, Clone)]
pub struct SceneCamera {
    pub view: Mat4,
    pub projection: Mat4,
    pub position: Vec3,
}

impl Default for SceneCamera {
    fn default() -> Self {
        Self {
            view: Mat4::look_at_rh(
                Vec3::new(5.0, 5.0, 5.0),
                Vec3::ZERO,
                Vec3::Y,
            ),
            projection: Mat4::perspective_rh(
                std::f32::consts::FRAC_PI_4,
                16.0 / 9.0,
                0.1,
                1000.0,
            ),
            position: Vec3::new(5.0, 5.0, 5.0),
        }
    }
}

/// A directional light.
#[derive(Debug, Clone)]
pub struct DirectionalLight {
    /// Normalised world-space direction *toward* the light source.
    pub direction: Vec3,
    /// Linear RGB colour.
    pub color: Vec3,
    /// Intensity multiplier.
    pub intensity: f32,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.3, -1.0, -0.5).normalize(),
            color: Vec3::new(1.0, 0.95, 0.9),
            intensity: 1.5,
        }
    }
}

/// A point light.
#[derive(Debug, Clone)]
pub struct PointLight {
    /// World-space position.
    pub position: Vec3,
    /// Linear RGB colour.
    pub color: Vec3,
    /// Intensity multiplier.
    pub intensity: f32,
    /// Attenuation range (light falls to zero at this distance).
    pub range: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 3.0, 0.0),
            color: Vec3::ONE,
            intensity: 5.0,
            range: 20.0,
        }
    }
}

/// A collection of lights for the scene.
#[derive(Debug, Clone, Default)]
pub struct SceneLights {
    pub ambient_color: Vec3,
    pub ambient_intensity: f32,
    pub directional_lights: Vec<DirectionalLight>,
    pub point_lights: Vec<PointLight>,
}

impl SceneLights {
    /// Build the GPU uniform struct from the high-level light description.
    pub fn to_uniform(&self) -> LightsUniform {
        let mut uniform = LightsUniform::default();
        uniform.ambient = [
            self.ambient_color.x,
            self.ambient_color.y,
            self.ambient_color.z,
            self.ambient_intensity,
        ];

        let mut idx = 0usize;

        // Pack directional lights.
        for dl in &self.directional_lights {
            if idx >= MAX_LIGHTS {
                break;
            }
            let dir = dl.direction.normalize();
            uniform.lights[idx] = GpuLight {
                position_or_direction: [dir.x, dir.y, dir.z, 0.0],
                color_intensity: [dl.color.x, dl.color.y, dl.color.z, dl.intensity],
                params: [1.0, 0.0, 0.0, 0.0], // type = 1 (directional)
            };
            idx += 1;
        }

        // Pack point lights.
        for pl in &self.point_lights {
            if idx >= MAX_LIGHTS {
                break;
            }
            uniform.lights[idx] = GpuLight {
                position_or_direction: [pl.position.x, pl.position.y, pl.position.z, pl.range],
                color_intensity: [pl.color.x, pl.color.y, pl.color.z, pl.intensity],
                params: [2.0, 0.0, 0.0, 0.0], // type = 2 (point)
            };
            idx += 1;
        }

        uniform.light_count = [idx as f32, 0.0, 0.0, 0.0];
        uniform
    }

    /// Create a default outdoor lighting setup.
    pub fn default_outdoor() -> Self {
        Self {
            ambient_color: Vec3::new(0.05, 0.06, 0.08),
            ambient_intensity: 1.0,
            directional_lights: vec![
                DirectionalLight {
                    direction: Vec3::new(0.4, -0.8, -0.4).normalize(),
                    color: Vec3::new(1.0, 0.95, 0.85),
                    intensity: 2.0,
                },
                DirectionalLight {
                    direction: Vec3::new(-0.3, -0.5, 0.6).normalize(),
                    color: Vec3::new(0.3, 0.4, 0.6),
                    intensity: 0.5,
                },
            ],
            point_lights: vec![],
        }
    }
}

/// Material parameters for CPU-side specification.
#[derive(Debug, Clone)]
pub struct MaterialParams {
    /// Base albedo colour (linear RGB + alpha).
    pub albedo_color: Vec4,
    /// Metallic factor [0..1].
    pub metallic: f32,
    /// Roughness factor [0..1].
    pub roughness: f32,
    /// Reflectance at normal incidence [0..1] (default 0.5 = 4% F0).
    pub reflectance: f32,
    /// Emissive colour (linear RGB).
    pub emissive: Vec3,
    /// Emissive strength multiplier.
    pub emissive_strength: f32,
    /// Alpha cutoff for alpha testing (0 = no cutoff).
    pub alpha_cutoff: f32,
}

impl Default for MaterialParams {
    fn default() -> Self {
        Self {
            albedo_color: Vec4::new(0.8, 0.8, 0.8, 1.0),
            metallic: 0.0,
            roughness: 0.5,
            reflectance: 0.5,
            emissive: Vec3::ZERO,
            emissive_strength: 0.0,
            alpha_cutoff: 0.0,
        }
    }
}

impl MaterialParams {
    /// Convert to the GPU uniform struct.
    pub fn to_uniform(&self) -> MaterialUniform {
        MaterialUniform {
            albedo_color: self.albedo_color.into(),
            metallic_roughness: [self.metallic, self.roughness, self.reflectance, self.alpha_cutoff],
            emissive: [
                self.emissive.x,
                self.emissive.y,
                self.emissive.z,
                self.emissive_strength,
            ],
            flags: [0.0; 4],
        }
    }

    /// Create a solid colour material.
    pub fn solid_color(color: Vec4) -> Self {
        Self {
            albedo_color: color,
            ..Default::default()
        }
    }

    /// Create a metallic material.
    pub fn metallic(color: Vec4, metallic: f32, roughness: f32) -> Self {
        Self {
            albedo_color: color,
            metallic,
            roughness,
            ..Default::default()
        }
    }

    /// Create a glowing (emissive) material.
    pub fn emissive(color: Vec4, emissive: Vec3, strength: f32) -> Self {
        Self {
            albedo_color: color,
            emissive,
            emissive_strength: strength,
            ..Default::default()
        }
    }

    /// Red material preset.
    pub fn red() -> Self {
        Self::solid_color(Vec4::new(0.9, 0.15, 0.1, 1.0))
    }

    /// Green material preset.
    pub fn green() -> Self {
        Self::solid_color(Vec4::new(0.15, 0.85, 0.2, 1.0))
    }

    /// Blue material preset.
    pub fn blue() -> Self {
        Self::solid_color(Vec4::new(0.1, 0.3, 0.9, 1.0))
    }

    /// White material preset.
    pub fn white() -> Self {
        Self::solid_color(Vec4::new(0.95, 0.95, 0.95, 1.0))
    }

    /// Grey (concrete-like) material preset.
    pub fn grey() -> Self {
        Self {
            albedo_color: Vec4::new(0.5, 0.5, 0.5, 1.0),
            roughness: 0.85,
            ..Default::default()
        }
    }

    /// Gold metallic preset.
    pub fn gold() -> Self {
        Self::metallic(Vec4::new(1.0, 0.76, 0.33, 1.0), 1.0, 0.3)
    }

    /// Copper metallic preset.
    pub fn copper() -> Self {
        Self::metallic(Vec4::new(0.95, 0.64, 0.54, 1.0), 1.0, 0.4)
    }

    /// Chrome metallic preset.
    pub fn chrome() -> Self {
        Self::metallic(Vec4::new(0.55, 0.55, 0.55, 1.0), 1.0, 0.1)
    }
}

// ============================================================================
// Handle / ID types
// ============================================================================

/// Opaque handle to a GPU mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshId(pub u64);

/// Opaque handle to a GPU material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialId(pub u64);

// ============================================================================
// GPU mesh data
// ============================================================================

/// A mesh uploaded to the GPU (vertex + index buffers).
pub struct MeshGpuData {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub vertex_count: u32,
}

impl MeshGpuData {
    /// Upload mesh data to the GPU.
    pub fn upload(
        device: &wgpu::Device,
        vertices: &[SceneVertex],
        indices: &[u32],
        label: &str,
    ) -> Self {
        use wgpu::util::DeviceExt;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label}_vb")),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label}_ib")),
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
// GPU material data
// ============================================================================

/// A material uploaded to the GPU (uniform buffer + bind group).
pub struct MaterialGpuData {
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub params: MaterialParams,
}

impl MaterialGpuData {
    /// Create GPU material data from parameters.
    pub fn upload(
        device: &wgpu::Device,
        params: &MaterialParams,
        layout: &wgpu::BindGroupLayout,
        label: &str,
    ) -> Self {
        use wgpu::util::DeviceExt;

        let uniform = params.to_uniform();

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label}_material_ub")),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}_material_bg")),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            uniform_buffer,
            bind_group,
            params: params.clone(),
        }
    }

    /// Update the material parameters on the GPU.
    pub fn update(&mut self, queue: &wgpu::Queue, params: &MaterialParams) {
        self.params = params.clone();
        let uniform = params.to_uniform();
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }
}

// ============================================================================
// Render object -- a single draw call
// ============================================================================

/// A renderable object: mesh + material + transform.
#[derive(Debug, Clone)]
pub struct RenderObject {
    pub mesh_id: MeshId,
    pub material_id: MaterialId,
    pub world_matrix: Mat4,
}

// ============================================================================
// WGSL PBR Shader (embedded)
// ============================================================================

/// The complete PBR shader source in WGSL.
///
/// Bind group layout:
///   Group 0: Camera + Lights (global per frame)
///     binding 0: CameraUniform
///     binding 1: LightsUniform
///   Group 1: Model (per object)
///     binding 0: ModelUniform
///   Group 2: Material (per material)
///     binding 0: MaterialUniform
pub const PBR_SHADER_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- PBR Scene Shader
// ============================================================================
//
// A physically-based rendering shader implementing:
//   - Per-vertex transform with model * view * projection
//   - Normal transformation via inverse-transpose model matrix
//   - N-dot-L Lambertian diffuse
//   - Blinn-Phong specular with roughness-derived shininess
//   - Fresnel approximation (Schlick)
//   - Distance attenuation for point lights
//   - Ambient term with hemisphere approximation
//   - Support for up to 8 lights (directional + point)
//   - Emissive contribution
//   - Vertex colour modulation
//   - Gamma correction output
// ============================================================================

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PI: f32 = 3.14159265359;
const INV_PI: f32 = 0.31830988618;
const EPSILON: f32 = 0.0001;
const MAX_LIGHTS: u32 = 8u;
const GAMMA: f32 = 2.2;
const INV_GAMMA: f32 = 0.45454545454;

// Light type constants
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

// ---------------------------------------------------------------------------
// Vertex input / output
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
    @location(3) vertex_color: vec4<f32>,
    @location(4) view_dir: vec3<f32>,
};

// ---------------------------------------------------------------------------
// Vertex shader
// ---------------------------------------------------------------------------

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // Transform position to world space.
    let world_pos = model.world * vec4<f32>(input.position, 1.0);
    output.world_position = world_pos.xyz;

    // Transform position to clip space.
    output.clip_position = camera.view_projection * world_pos;

    // Transform normal to world space using the normal matrix.
    // We only use the upper 3x3 of the normal_matrix (inverse-transpose).
    let raw_normal = (model.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz;
    output.world_normal = normalize(raw_normal);

    // Pass through UV coordinates.
    output.uv = input.uv;

    // Pass through vertex colour.
    output.vertex_color = input.color;

    // Compute view direction (from surface to camera).
    output.view_dir = normalize(camera.camera_position.xyz - world_pos.xyz);

    return output;
}

// ---------------------------------------------------------------------------
// Lighting utility functions
// ---------------------------------------------------------------------------

// Schlick's Fresnel approximation.
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let t = clamp(1.0 - cos_theta, 0.0, 1.0);
    let t2 = t * t;
    let t5 = t2 * t2 * t;
    return f0 + (vec3<f32>(1.0) - f0) * t5;
}

// Fresnel-Schlick with roughness for ambient.
fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let t = clamp(1.0 - cos_theta, 0.0, 1.0);
    let t2 = t * t;
    let t5 = t2 * t2 * t;
    let max_reflect = vec3<f32>(1.0 - roughness);
    return f0 + (max(max_reflect, f0) - f0) * t5;
}

// GGX / Trowbridge-Reitz normal distribution function.
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom_term = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom_term * denom_term + EPSILON);
}

// Schlick-GGX geometry function (single direction).
fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k + EPSILON);
}

// Smith's geometry function combining both view and light directions.
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let ggx_v = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx_l = geometry_schlick_ggx(n_dot_l, roughness);
    return ggx_v * ggx_l;
}

// Blinn-Phong specular (fallback / additional specular).
fn blinn_phong_specular(
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    roughness: f32
) -> f32 {
    let half_dir = normalize(view_dir + light_dir);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    // Map roughness [0..1] to shininess exponent [2..2048].
    let shininess = exp2(10.0 * (1.0 - roughness) + 1.0);
    // Normalisation factor for energy conservation.
    let norm_factor = (shininess + 8.0) / (8.0 * PI);
    return norm_factor * pow(n_dot_h, shininess);
}

// Smooth distance attenuation for point lights.
fn attenuation_smooth(distance: f32, range: f32) -> f32 {
    if range <= 0.0 {
        return 1.0;
    }
    let d = distance / range;
    let d2 = d * d;
    let d4 = d2 * d2;
    let factor = clamp(1.0 - d4, 0.0, 1.0);
    return factor * factor / (distance * distance + 1.0);
}

// Inverse-square falloff with range clamping.
fn attenuation_inverse_square(distance: f32, range: f32) -> f32 {
    if range <= 0.0 {
        return 1.0;
    }
    let d_over_r = distance / range;
    if d_over_r >= 1.0 {
        return 0.0;
    }
    let att = 1.0 / (distance * distance + 1.0);
    // Smooth window function to avoid hard cutoff.
    let window = 1.0 - d_over_r * d_over_r;
    let window_sq = window * window;
    return att * window_sq;
}

// Compute hemisphere ambient (sky + ground interpolation).
fn hemisphere_ambient(
    normal: vec3<f32>,
    ambient_color: vec3<f32>,
    ambient_intensity: f32
) -> vec3<f32> {
    // Simple hemisphere: interpolate between ground colour (darker, warm)
    // and sky colour (ambient_color) based on the normal's Y component.
    let sky_color = ambient_color * ambient_intensity;
    let ground_color = sky_color * vec3<f32>(0.6, 0.5, 0.4) * 0.5;
    let t = normal.y * 0.5 + 0.5; // remap from [-1,1] to [0,1]
    return mix(ground_color, sky_color, t);
}

// Linearize an sRGB colour value.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, GAMMA);
}

// Convert linear to sRGB.
fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        return c * 12.92;
    }
    return 1.055 * pow(c, INV_GAMMA) - 0.055;
}

// Tone mapping (Reinhard).
fn tone_map_reinhard(color: vec3<f32>) -> vec3<f32> {
    return color / (color + vec3<f32>(1.0));
}

// ACES filmic tone mapping approximation.
fn tone_map_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 1.43; // originally named c but conflicts -- use c_
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + vec3<f32>(b))) / (x * (c * x + vec3<f32>(d)) + vec3<f32>(e)),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

// Compute the Fresnel F0 for dielectrics from reflectance, or from albedo for metals.
fn compute_f0(albedo: vec3<f32>, metallic: f32, reflectance: f32) -> vec3<f32> {
    // Dielectric F0 from reflectance factor.
    let dielectric_f0 = vec3<f32>(0.16 * reflectance * reflectance);
    // Metal F0 is the albedo itself.
    return mix(dielectric_f0, albedo, metallic);
}

// ---------------------------------------------------------------------------
// Compute lighting contribution from a single light
// ---------------------------------------------------------------------------

struct LightContribution {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
};

fn compute_directional_light(
    light: GpuLight,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    f0: vec3<f32>,
) -> LightContribution {
    var result: LightContribution;
    result.diffuse = vec3<f32>(0.0);
    result.specular = vec3<f32>(0.0);

    // Direction stored in the uniform points FROM surface TO light (negate for
    // "direction of incoming light").  However, our convention stores the
    // direction *toward* the light, so we can use it directly as L.
    let light_dir = normalize(-light.position_or_direction.xyz);
    let light_color = light.color_intensity.xyz * light.color_intensity.w;

    let n_dot_l = max(dot(normal, light_dir), 0.0);
    if n_dot_l <= 0.0 {
        return result;
    }

    let half_vec = normalize(view_dir + light_dir);
    let n_dot_v = max(dot(normal, view_dir), EPSILON);
    let n_dot_h = max(dot(normal, half_vec), 0.0);
    let v_dot_h = max(dot(view_dir, half_vec), 0.0);

    // Cook-Torrance BRDF components.
    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(v_dot_h, f0);

    // Specular BRDF.
    let numerator = d * g * f;
    let denominator = 4.0 * n_dot_v * n_dot_l + EPSILON;
    let spec_brdf = numerator / denominator;

    // Energy conservation: what is not reflected is refracted (diffuse).
    let k_s = f;
    let k_d = (vec3<f32>(1.0) - k_s) * (1.0 - metallic);

    // Lambertian diffuse.
    let diffuse_brdf = k_d * albedo * INV_PI;

    // Also add Blinn-Phong specular on top for more visible highlights.
    let bp_spec = blinn_phong_specular(normal, view_dir, light_dir, roughness);
    let bp_contribution = f * bp_spec * 0.15;

    result.diffuse = diffuse_brdf * light_color * n_dot_l;
    result.specular = (spec_brdf + bp_contribution) * light_color * n_dot_l;

    return result;
}

fn compute_point_light(
    light: GpuLight,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    f0: vec3<f32>,
) -> LightContribution {
    var result: LightContribution;
    result.diffuse = vec3<f32>(0.0);
    result.specular = vec3<f32>(0.0);

    let light_pos = light.position_or_direction.xyz;
    let light_range = light.position_or_direction.w;
    let to_light = light_pos - world_pos;
    let distance = length(to_light);

    // Early out if beyond range.
    if light_range > 0.0 && distance > light_range * 1.2 {
        return result;
    }

    let light_dir = normalize(to_light);
    let light_color = light.color_intensity.xyz * light.color_intensity.w;
    let att = attenuation_smooth(distance, light_range);

    if att < EPSILON {
        return result;
    }

    let n_dot_l = max(dot(normal, light_dir), 0.0);
    if n_dot_l <= 0.0 {
        return result;
    }

    let half_vec = normalize(view_dir + light_dir);
    let n_dot_v = max(dot(normal, view_dir), EPSILON);
    let n_dot_h = max(dot(normal, half_vec), 0.0);
    let v_dot_h = max(dot(view_dir, half_vec), 0.0);

    // Cook-Torrance BRDF.
    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(v_dot_h, f0);

    let spec_brdf = (d * g * f) / (4.0 * n_dot_v * n_dot_l + EPSILON);

    let k_s = f;
    let k_d = (vec3<f32>(1.0) - k_s) * (1.0 - metallic);
    let diffuse_brdf = k_d * albedo * INV_PI;

    let bp_spec = blinn_phong_specular(normal, view_dir, light_dir, roughness);
    let bp_contribution = f * bp_spec * 0.15;

    let attenuated_light = light_color * att;

    result.diffuse = diffuse_brdf * attenuated_light * n_dot_l;
    result.specular = (spec_brdf + bp_contribution) * attenuated_light * n_dot_l;

    return result;
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Extract material properties.
    let albedo = material.albedo_color.xyz * input.vertex_color.xyz;
    let alpha = material.albedo_color.w * input.vertex_color.w;
    let metallic = clamp(material.metallic_roughness.x, 0.0, 1.0);
    let roughness = clamp(material.metallic_roughness.y, 0.04, 1.0);
    let reflectance = material.metallic_roughness.z;

    // Compute F0 (reflectance at normal incidence).
    let f0 = compute_f0(albedo, metallic, reflectance);

    // Re-normalise interpolated normal.
    let normal = normalize(input.world_normal);
    let view_dir = normalize(input.view_dir);

    // Accumulate lighting.
    var total_diffuse = vec3<f32>(0.0);
    var total_specular = vec3<f32>(0.0);

    let num_lights = u32(lights.light_count.x);

    // Unrolled light loop (up to MAX_LIGHTS).
    for (var i = 0u; i < MAX_LIGHTS; i = i + 1u) {
        if i >= num_lights {
            break;
        }

        let light = lights.lights[i];
        let light_type = light.params.x;

        if abs(light_type - LIGHT_TYPE_DIRECTIONAL) < 0.5 {
            // Directional light.
            let contrib = compute_directional_light(
                light, normal, view_dir, albedo, metallic, roughness, f0
            );
            total_diffuse = total_diffuse + contrib.diffuse;
            total_specular = total_specular + contrib.specular;
        } else if abs(light_type - LIGHT_TYPE_POINT) < 0.5 {
            // Point light.
            let contrib = compute_point_light(
                light, input.world_position, normal, view_dir,
                albedo, metallic, roughness, f0
            );
            total_diffuse = total_diffuse + contrib.diffuse;
            total_specular = total_specular + contrib.specular;
        }
        // LIGHT_TYPE_DISABLED: skip
    }

    // Ambient lighting (hemisphere approximation).
    let ambient = hemisphere_ambient(
        normal,
        lights.ambient.xyz,
        lights.ambient.w
    );
    let n_dot_v_ambient = max(dot(normal, view_dir), 0.0);
    let f_ambient = fresnel_schlick_roughness(n_dot_v_ambient, f0, roughness);
    let k_d_ambient = (vec3<f32>(1.0) - f_ambient) * (1.0 - metallic);
    let ambient_diffuse = k_d_ambient * albedo * ambient;
    let ambient_specular = f_ambient * ambient * 0.2;

    // Emissive contribution.
    let emissive = material.emissive.xyz * material.emissive.w;

    // Combine all lighting.
    var final_color = total_diffuse + total_specular + ambient_diffuse + ambient_specular + emissive;

    // Tone mapping (ACES).
    final_color = tone_map_aces(final_color);

    // Gamma correction (linear -> sRGB).
    final_color = vec3<f32>(
        linear_to_srgb(final_color.x),
        linear_to_srgb(final_color.y),
        linear_to_srgb(final_color.z)
    );

    return vec4<f32>(final_color, alpha);
}
"#;

// ============================================================================
// Grid shader (embedded WGSL)
// ============================================================================

/// Shader for rendering the ground grid as line geometry.
pub const GRID_SHADER_WGSL: &str = r#"
// ---------------------------------------------------------------------------
// Genovo Engine -- Grid Line Shader
// ---------------------------------------------------------------------------
//
// Renders thin line segments as camera-facing quads. Each vertex carries a
// position and colour. The grid fades out with distance from the camera.
// ---------------------------------------------------------------------------

struct CameraUniform {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct GridVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct GridVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_position: vec3<f32>,
};

@vertex
fn vs_grid(input: GridVertexInput) -> GridVertexOutput {
    var output: GridVertexOutput;
    let world_pos = vec4<f32>(input.position, 1.0);
    output.clip_position = camera.view_projection * world_pos;
    output.color = input.color;
    output.world_position = input.position;
    return output;
}

@fragment
fn fs_grid(input: GridVertexOutput) -> @location(0) vec4<f32> {
    // Distance-based fade.
    let dist = length(input.world_position - camera.camera_position.xyz);
    let fade_start = 20.0;
    let fade_end = 80.0;
    let fade = 1.0 - clamp((dist - fade_start) / (fade_end - fade_start), 0.0, 1.0);

    var color = input.color;
    color.w = color.w * fade;

    // Discard fully transparent fragments.
    if color.w < 0.01 {
        discard;
    }

    return color;
}
"#;

// ============================================================================
// Grid vertex type
// ============================================================================

/// Vertex for grid line rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GridVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

impl GridVertex {
    pub fn new(position: Vec3, color: Vec4) -> Self {
        Self {
            position: position.into(),
            color: color.into(),
        }
    }

    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GridVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 12,
                    shader_location: 1,
                },
            ],
        }
    }
}

// ============================================================================
// Built-in primitive mesh generators
// ============================================================================

/// Generate a unit cube centered at the origin.
///
/// Each face has 4 unique vertices (for correct flat normals), totalling
/// 24 vertices and 36 indices.
pub fn generate_cube() -> (Vec<SceneVertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);

    // Face definitions: (normal, tangent_u, tangent_v).
    let faces: [(Vec3, Vec3, Vec3); 6] = [
        // +X face
        (Vec3::X, Vec3::Z, Vec3::Y),
        // -X face
        (Vec3::NEG_X, Vec3::NEG_Z, Vec3::Y),
        // +Y face (top)
        (Vec3::Y, Vec3::X, Vec3::NEG_Z),
        // -Y face (bottom)
        (Vec3::NEG_Y, Vec3::X, Vec3::Z),
        // +Z face
        (Vec3::Z, Vec3::NEG_X, Vec3::Y),
        // -Z face
        (Vec3::NEG_Z, Vec3::X, Vec3::Y),
    ];

    for (face_idx, (normal, u_axis, v_axis)) in faces.iter().enumerate() {
        let base = vertices.len() as u32;
        let center = *normal * 0.5;

        // Four corners of the face.
        let half_u = *u_axis * 0.5;
        let half_v = *v_axis * 0.5;

        let p0 = center - half_u - half_v;
        let p1 = center + half_u - half_v;
        let p2 = center + half_u + half_v;
        let p3 = center - half_u + half_v;

        vertices.push(SceneVertex::with_pos_normal_uv(p0, *normal, Vec2::new(0.0, 0.0)));
        vertices.push(SceneVertex::with_pos_normal_uv(p1, *normal, Vec2::new(1.0, 0.0)));
        vertices.push(SceneVertex::with_pos_normal_uv(p2, *normal, Vec2::new(1.0, 1.0)));
        vertices.push(SceneVertex::with_pos_normal_uv(p3, *normal, Vec2::new(0.0, 1.0)));

        // Two triangles per face (CCW winding).
        indices.push(base);
        indices.push(base + 1);
        indices.push(base + 2);

        indices.push(base);
        indices.push(base + 2);
        indices.push(base + 3);
    }

    debug_assert_eq!(vertices.len(), 24, "Cube should have 24 vertices");
    debug_assert_eq!(indices.len(), 36, "Cube should have 36 indices");

    (vertices, indices)
}

/// Generate a UV sphere with the given number of horizontal segments and
/// vertical rings.
///
/// `segments` = slices around the equator (longitude divisions).
/// `rings` = stacks from pole to pole (latitude divisions, excluding poles).
///
/// The sphere has radius 0.5 (unit diameter) centered at origin.
pub fn generate_sphere(segments: u32, rings: u32) -> (Vec<SceneVertex>, Vec<u32>) {
    let segments = segments.max(4);
    let rings = rings.max(2);

    let vert_count = ((rings + 1) * (segments + 1)) as usize;
    let idx_count = (rings * segments * 6) as usize;
    let mut vertices = Vec::with_capacity(vert_count);
    let mut indices = Vec::with_capacity(idx_count);

    let radius = 0.5f32;

    // Generate vertices.
    for ring in 0..=rings {
        let phi = std::f32::consts::PI * (ring as f32) / (rings as f32);
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();

        for seg in 0..=segments {
            let theta = 2.0 * std::f32::consts::PI * (seg as f32) / (segments as f32);
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();

            let x = cos_theta * sin_phi;
            let y = cos_phi;
            let z = sin_theta * sin_phi;

            let normal = Vec3::new(x, y, z);
            let position = normal * radius;
            let u = seg as f32 / segments as f32;
            let v = ring as f32 / rings as f32;

            vertices.push(SceneVertex::with_pos_normal_uv(
                position,
                normal,
                Vec2::new(u, v),
            ));
        }
    }

    // Generate indices.
    for ring in 0..rings {
        for seg in 0..segments {
            let current = ring * (segments + 1) + seg;
            let next = current + segments + 1;

            // First triangle.
            indices.push(current);
            indices.push(next);
            indices.push(current + 1);

            // Second triangle.
            indices.push(current + 1);
            indices.push(next);
            indices.push(next + 1);
        }
    }

    (vertices, indices)
}

/// Generate a flat ground plane in the XZ plane, centered at origin.
///
/// The plane faces upward (+Y). `size` is the half-extent in each direction,
/// `subdivisions` is the number of quads along each axis.
pub fn generate_plane(size: f32, subdivisions: u32) -> (Vec<SceneVertex>, Vec<u32>) {
    let subdivisions = subdivisions.max(1);
    let step = (size * 2.0) / subdivisions as f32;
    let vert_per_side = subdivisions + 1;
    let vert_count = (vert_per_side * vert_per_side) as usize;
    let idx_count = (subdivisions * subdivisions * 6) as usize;

    let mut vertices = Vec::with_capacity(vert_count);
    let mut indices = Vec::with_capacity(idx_count);

    let normal = Vec3::Y;

    for z_idx in 0..=subdivisions {
        for x_idx in 0..=subdivisions {
            let x = -size + x_idx as f32 * step;
            let z = -size + z_idx as f32 * step;
            let u = x_idx as f32 / subdivisions as f32;
            let v = z_idx as f32 / subdivisions as f32;

            vertices.push(SceneVertex::with_pos_normal_uv(
                Vec3::new(x, 0.0, z),
                normal,
                Vec2::new(u, v),
            ));
        }
    }

    for z_idx in 0..subdivisions {
        for x_idx in 0..subdivisions {
            let top_left = z_idx * vert_per_side + x_idx;
            let top_right = top_left + 1;
            let bottom_left = top_left + vert_per_side;
            let bottom_right = bottom_left + 1;

            // First triangle.
            indices.push(top_left);
            indices.push(bottom_left);
            indices.push(top_right);

            // Second triangle.
            indices.push(top_right);
            indices.push(bottom_left);
            indices.push(bottom_right);
        }
    }

    (vertices, indices)
}

/// Generate a cylinder along the Y axis.
///
/// `segments` = number of divisions around the circumference.
/// `radius` = cylinder radius (default 0.5).
/// `height` = total height (default 1.0, centered at origin).
///
/// Includes top and bottom caps.
pub fn generate_cylinder(segments: u32) -> (Vec<SceneVertex>, Vec<u32>) {
    generate_cylinder_ex(segments, 0.5, 1.0)
}

/// Generate a cylinder with explicit radius and height.
pub fn generate_cylinder_ex(
    segments: u32,
    radius: f32,
    height: f32,
) -> (Vec<SceneVertex>, Vec<u32>) {
    let segments = segments.max(3);
    let half_h = height * 0.5;

    // Pre-compute capacity.
    // Side: (segments+1)*2 vertices, segments*6 indices
    // Top cap: segments+1 (center + ring) vertices, segments*3 indices
    // Bottom cap: same
    let side_verts = ((segments + 1) * 2) as usize;
    let cap_verts = (segments + 1) as usize; // center + rim
    let total_verts = side_verts + cap_verts * 2;
    let side_indices = (segments * 6) as usize;
    let cap_indices = (segments * 3) as usize;
    let total_indices = side_indices + cap_indices * 2;

    let mut vertices = Vec::with_capacity(total_verts);
    let mut indices = Vec::with_capacity(total_indices);

    // --- Side ---
    for i in 0..=segments {
        let theta = 2.0 * std::f32::consts::PI * (i as f32) / (segments as f32);
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let normal = Vec3::new(cos_t, 0.0, sin_t);
        let u = i as f32 / segments as f32;

        // Bottom vertex.
        vertices.push(SceneVertex::with_pos_normal_uv(
            Vec3::new(cos_t * radius, -half_h, sin_t * radius),
            normal,
            Vec2::new(u, 1.0),
        ));

        // Top vertex.
        vertices.push(SceneVertex::with_pos_normal_uv(
            Vec3::new(cos_t * radius, half_h, sin_t * radius),
            normal,
            Vec2::new(u, 0.0),
        ));
    }

    for i in 0..segments {
        let base = i * 2;
        // bottom-left, top-left, top-right
        indices.push(base);
        indices.push(base + 1);
        indices.push(base + 3);
        // bottom-left, top-right, bottom-right
        indices.push(base);
        indices.push(base + 3);
        indices.push(base + 2);
    }

    // --- Top cap ---
    let top_center_idx = vertices.len() as u32;
    vertices.push(SceneVertex::with_pos_normal_uv(
        Vec3::new(0.0, half_h, 0.0),
        Vec3::Y,
        Vec2::new(0.5, 0.5),
    ));

    for i in 0..segments {
        let theta = 2.0 * std::f32::consts::PI * (i as f32) / (segments as f32);
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        vertices.push(SceneVertex::with_pos_normal_uv(
            Vec3::new(cos_t * radius, half_h, sin_t * radius),
            Vec3::Y,
            Vec2::new(cos_t * 0.5 + 0.5, sin_t * 0.5 + 0.5),
        ));
    }

    for i in 0..segments {
        let current = top_center_idx + 1 + i;
        let next = top_center_idx + 1 + ((i + 1) % segments);
        indices.push(top_center_idx);
        indices.push(current);
        indices.push(next);
    }

    // --- Bottom cap ---
    let bottom_center_idx = vertices.len() as u32;
    vertices.push(SceneVertex::with_pos_normal_uv(
        Vec3::new(0.0, -half_h, 0.0),
        Vec3::NEG_Y,
        Vec2::new(0.5, 0.5),
    ));

    for i in 0..segments {
        let theta = 2.0 * std::f32::consts::PI * (i as f32) / (segments as f32);
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        vertices.push(SceneVertex::with_pos_normal_uv(
            Vec3::new(cos_t * radius, -half_h, sin_t * radius),
            Vec3::NEG_Y,
            Vec2::new(cos_t * 0.5 + 0.5, sin_t * 0.5 + 0.5),
        ));
    }

    for i in 0..segments {
        let current = bottom_center_idx + 1 + i;
        let next = bottom_center_idx + 1 + ((i + 1) % segments);
        // Reverse winding for bottom cap.
        indices.push(bottom_center_idx);
        indices.push(next);
        indices.push(current);
    }

    (vertices, indices)
}

/// Generate a cone along the Y axis with the apex at the top.
///
/// `segments` = number of divisions around the circumference.
/// The cone has radius 0.5 at the base and height 1.0.
pub fn generate_cone(segments: u32) -> (Vec<SceneVertex>, Vec<u32>) {
    let segments = segments.max(3);
    let radius = 0.5f32;
    let half_h = 0.5f32;
    let slant = (radius * radius + (half_h * 2.0).powi(2)).sqrt();

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // --- Side ---
    // Each segment gets a triangle from apex to two base rim points.
    let apex_idx = vertices.len() as u32;
    vertices.push(SceneVertex::with_pos_normal_uv(
        Vec3::new(0.0, half_h, 0.0),
        Vec3::Y,
        Vec2::new(0.5, 0.0),
    ));

    for i in 0..=segments {
        let theta = 2.0 * std::f32::consts::PI * (i as f32) / (segments as f32);
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Normal for the cone side: the outward direction but tilted upward
        // by the cone's slope.
        let ny = radius / slant;
        let nxz = (half_h * 2.0) / slant;
        let normal = Vec3::new(cos_t * nxz, ny, sin_t * nxz).normalize();

        vertices.push(SceneVertex::with_pos_normal_uv(
            Vec3::new(cos_t * radius, -half_h, sin_t * radius),
            normal,
            Vec2::new(i as f32 / segments as f32, 1.0),
        ));
    }

    // We also need to duplicate the apex per triangle for correct normals.
    // For simplicity, use a shared apex and accept slightly incorrect
    // normals at the tip.
    for i in 0..segments {
        let base_rim = apex_idx + 1 + i;
        let next_rim = apex_idx + 1 + i + 1;
        indices.push(apex_idx);
        indices.push(base_rim);
        indices.push(next_rim);
    }

    // --- Bottom cap ---
    let bottom_center_idx = vertices.len() as u32;
    vertices.push(SceneVertex::with_pos_normal_uv(
        Vec3::new(0.0, -half_h, 0.0),
        Vec3::NEG_Y,
        Vec2::new(0.5, 0.5),
    ));

    for i in 0..segments {
        let theta = 2.0 * std::f32::consts::PI * (i as f32) / (segments as f32);
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        vertices.push(SceneVertex::with_pos_normal_uv(
            Vec3::new(cos_t * radius, -half_h, sin_t * radius),
            Vec3::NEG_Y,
            Vec2::new(cos_t * 0.5 + 0.5, sin_t * 0.5 + 0.5),
        ));
    }

    for i in 0..segments {
        let current = bottom_center_idx + 1 + i;
        let next = bottom_center_idx + 1 + ((i + 1) % segments);
        indices.push(bottom_center_idx);
        indices.push(next);
        indices.push(current);
    }

    (vertices, indices)
}

/// Generate a torus (donut) in the XZ plane.
///
/// `major_segments` = number of divisions around the main ring.
/// `minor_segments` = number of divisions around the tube cross-section.
/// `major_radius` = distance from center to the tube center.
/// `minor_radius` = radius of the tube itself.
pub fn generate_torus(
    major_segments: u32,
    minor_segments: u32,
    major_radius: f32,
    minor_radius: f32,
) -> (Vec<SceneVertex>, Vec<u32>) {
    let major_segments = major_segments.max(3);
    let minor_segments = minor_segments.max(3);

    let vert_count = ((major_segments + 1) * (minor_segments + 1)) as usize;
    let idx_count = (major_segments * minor_segments * 6) as usize;
    let mut vertices = Vec::with_capacity(vert_count);
    let mut indices = Vec::with_capacity(idx_count);

    for i in 0..=major_segments {
        let u_angle = 2.0 * std::f32::consts::PI * (i as f32) / (major_segments as f32);
        let cos_u = u_angle.cos();
        let sin_u = u_angle.sin();

        for j in 0..=minor_segments {
            let v_angle = 2.0 * std::f32::consts::PI * (j as f32) / (minor_segments as f32);
            let cos_v = v_angle.cos();
            let sin_v = v_angle.sin();

            let x = (major_radius + minor_radius * cos_v) * cos_u;
            let y = minor_radius * sin_v;
            let z = (major_radius + minor_radius * cos_v) * sin_u;

            // Normal: direction from the ring center to the surface point.
            let center_x = major_radius * cos_u;
            let center_z = major_radius * sin_u;
            let normal = Vec3::new(x - center_x, y, z - center_z).normalize();

            let uv = Vec2::new(
                i as f32 / major_segments as f32,
                j as f32 / minor_segments as f32,
            );

            vertices.push(SceneVertex::with_pos_normal_uv(
                Vec3::new(x, y, z),
                normal,
                uv,
            ));
        }
    }

    for i in 0..major_segments {
        for j in 0..minor_segments {
            let a = i * (minor_segments + 1) + j;
            let b = a + minor_segments + 1;

            indices.push(a);
            indices.push(b);
            indices.push(a + 1);

            indices.push(a + 1);
            indices.push(b);
            indices.push(b + 1);
        }
    }

    (vertices, indices)
}

/// Generate an icosphere by subdividing an icosahedron.
///
/// `subdivisions` = number of recursive subdivision steps (0 = icosahedron,
/// 1 = 80 faces, 2 = 320 faces, etc.).
pub fn generate_icosphere(subdivisions: u32) -> (Vec<SceneVertex>, Vec<u32>) {
    let radius = 0.5f32;

    // Golden ratio.
    let t = (1.0 + 5.0f32.sqrt()) / 2.0;
    let len = (1.0 + t * t).sqrt();
    let a = 1.0 / len;
    let b = t / len;

    // Initial icosahedron vertices (12 vertices, 20 faces).
    let base_positions: Vec<Vec3> = vec![
        Vec3::new(-a, b, 0.0),
        Vec3::new(a, b, 0.0),
        Vec3::new(-a, -b, 0.0),
        Vec3::new(a, -b, 0.0),
        Vec3::new(0.0, -a, b),
        Vec3::new(0.0, a, b),
        Vec3::new(0.0, -a, -b),
        Vec3::new(0.0, a, -b),
        Vec3::new(b, 0.0, -a),
        Vec3::new(b, 0.0, a),
        Vec3::new(-b, 0.0, -a),
        Vec3::new(-b, 0.0, a),
    ];

    let base_indices: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];

    let mut positions = base_positions;
    let mut tri_indices: Vec<[u32; 3]> = base_indices;

    // Midpoint cache for subdivision.
    let mut midpoint_cache: HashMap<u64, u32> = HashMap::new();

    let get_midpoint = |p: &mut Vec<Vec3>, cache: &mut HashMap<u64, u32>, i0: u32, i1: u32| -> u32 {
        let key = if i0 < i1 {
            (i0 as u64) << 32 | i1 as u64
        } else {
            (i1 as u64) << 32 | i0 as u64
        };
        if let Some(&idx) = cache.get(&key) {
            return idx;
        }
        let mid = ((p[i0 as usize] + p[i1 as usize]) * 0.5).normalize();
        let idx = p.len() as u32;
        p.push(mid);
        cache.insert(key, idx);
        idx
    };

    for _ in 0..subdivisions {
        let mut new_tris = Vec::with_capacity(tri_indices.len() * 4);
        midpoint_cache.clear();

        for tri in &tri_indices {
            let a = get_midpoint(&mut positions, &mut midpoint_cache, tri[0], tri[1]);
            let b = get_midpoint(&mut positions, &mut midpoint_cache, tri[1], tri[2]);
            let c = get_midpoint(&mut positions, &mut midpoint_cache, tri[2], tri[0]);

            new_tris.push([tri[0], a, c]);
            new_tris.push([tri[1], b, a]);
            new_tris.push([tri[2], c, b]);
            new_tris.push([a, b, c]);
        }

        tri_indices = new_tris;
    }

    // Build final vertex and index arrays.
    let mut vertices = Vec::with_capacity(positions.len());
    for pos in &positions {
        let normal = pos.normalize();
        let position = normal * radius;
        // Spherical UV mapping.
        let u = 0.5 + normal.z.atan2(normal.x) / (2.0 * std::f32::consts::PI);
        let v = 0.5 - normal.y.asin() / std::f32::consts::PI;
        vertices.push(SceneVertex::with_pos_normal_uv(position, normal, Vec2::new(u, v)));
    }

    let mut indices = Vec::with_capacity(tri_indices.len() * 3);
    for tri in &tri_indices {
        indices.push(tri[0]);
        indices.push(tri[1]);
        indices.push(tri[2]);
    }

    (vertices, indices)
}

// ============================================================================
// Grid mesh generation
// ============================================================================

/// Generate grid line geometry for a ground grid.
///
/// Creates thin line segments as pairs of vertices in the XZ plane at y=0.
/// Lines aligned with the axes (x=0, z=0) are highlighted.
///
/// `extent` = half-extent of the grid.
/// `spacing` = distance between grid lines.
pub fn generate_grid_lines(extent: f32, spacing: f32) -> (Vec<GridVertex>, Vec<u32>) {
    let line_count = ((extent * 2.0 / spacing) as u32) + 1;
    let half = extent;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let base_color = Vec4::new(0.35, 0.35, 0.35, 0.6);
    let axis_color_x = Vec4::new(0.8, 0.2, 0.2, 0.85);
    let axis_color_z = Vec4::new(0.2, 0.2, 0.8, 0.85);
    let major_color = Vec4::new(0.45, 0.45, 0.45, 0.7);

    // Lines along the X axis (varying Z).
    let start_z = -half;
    for i in 0..=line_count {
        let z = start_z + i as f32 * spacing;
        let z_rounded = (z / spacing).round() * spacing;

        let color = if z_rounded.abs() < spacing * 0.1 {
            axis_color_x // Red for the X axis line
        } else if (z_rounded / (spacing * 5.0)).fract().abs() < 0.1 {
            major_color // Brighter for major grid lines
        } else {
            base_color
        };

        // Line width via thin quad (two triangles).
        let width = if z_rounded.abs() < spacing * 0.1 { 0.02 } else { 0.005 };

        let base_idx = vertices.len() as u32;
        vertices.push(GridVertex::new(Vec3::new(-half, 0.001, z_rounded - width), color));
        vertices.push(GridVertex::new(Vec3::new(half, 0.001, z_rounded - width), color));
        vertices.push(GridVertex::new(Vec3::new(half, 0.001, z_rounded + width), color));
        vertices.push(GridVertex::new(Vec3::new(-half, 0.001, z_rounded + width), color));

        indices.push(base_idx);
        indices.push(base_idx + 1);
        indices.push(base_idx + 2);
        indices.push(base_idx);
        indices.push(base_idx + 2);
        indices.push(base_idx + 3);
    }

    // Lines along the Z axis (varying X).
    let start_x = -half;
    for i in 0..=line_count {
        let x = start_x + i as f32 * spacing;
        let x_rounded = (x / spacing).round() * spacing;

        let color = if x_rounded.abs() < spacing * 0.1 {
            axis_color_z // Blue for the Z axis line
        } else if (x_rounded / (spacing * 5.0)).fract().abs() < 0.1 {
            major_color
        } else {
            base_color
        };

        let width = if x_rounded.abs() < spacing * 0.1 { 0.02 } else { 0.005 };

        let base_idx = vertices.len() as u32;
        vertices.push(GridVertex::new(Vec3::new(x_rounded - width, 0.001, -half), color));
        vertices.push(GridVertex::new(Vec3::new(x_rounded + width, 0.001, -half), color));
        vertices.push(GridVertex::new(Vec3::new(x_rounded + width, 0.001, half), color));
        vertices.push(GridVertex::new(Vec3::new(x_rounded - width, 0.001, half), color));

        indices.push(base_idx);
        indices.push(base_idx + 1);
        indices.push(base_idx + 2);
        indices.push(base_idx);
        indices.push(base_idx + 2);
        indices.push(base_idx + 3);
    }

    (vertices, indices)
}

// ============================================================================
// Pipeline builder helpers
// ============================================================================

/// Create the bind group layout for camera + lights (group 0).
fn create_camera_lights_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene_camera_lights_bgl"),
        entries: &[
            // binding 0: CameraUniform
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<CameraUniform>() as u64,
                    ),
                },
                count: None,
            },
            // binding 1: LightsUniform
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<LightsUniform>() as u64,
                    ),
                },
                count: None,
            },
        ],
    })
}

/// Create the bind group layout for per-object model data (group 1).
fn create_model_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene_model_bgl"),
        entries: &[
            // binding 0: ModelUniform
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<ModelUniform>() as u64,
                    ),
                },
                count: None,
            },
        ],
    })
}

/// Create the bind group layout for material data (group 2).
fn create_material_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene_material_bgl"),
        entries: &[
            // binding 0: MaterialUniform
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<MaterialUniform>() as u64,
                    ),
                },
                count: None,
            },
        ],
    })
}

/// Create the grid bind group layout (group 0, camera only).
fn create_grid_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("grid_camera_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<CameraUniform>() as u64,
                    ),
                },
                count: None,
            },
        ],
    })
}

// ============================================================================
// SceneRenderer -- core render pipeline
// ============================================================================

/// The core scene rendering pipeline.
///
/// Owns the wgpu render pipeline, bind group layouts, and transient per-frame
/// uniform buffers. Call `render_scene()` each frame to draw a list of objects.
pub struct SceneRenderer {
    // --- PBR pipeline ---
    pbr_pipeline: wgpu::RenderPipeline,
    camera_lights_bgl: wgpu::BindGroupLayout,
    model_bgl: wgpu::BindGroupLayout,
    material_bgl: wgpu::BindGroupLayout,

    // --- Grid pipeline ---
    grid_pipeline: wgpu::RenderPipeline,
    grid_bgl: wgpu::BindGroupLayout,

    // --- Per-frame uniform buffers ---
    camera_uniform_buffer: wgpu::Buffer,
    lights_uniform_buffer: wgpu::Buffer,
    camera_lights_bind_group: wgpu::BindGroup,

    // --- Grid uniform buffer (shares camera data) ---
    grid_camera_buffer: wgpu::Buffer,
    grid_bind_group: wgpu::BindGroup,

    // --- Per-object model uniform pool ---
    /// Pool of model uniform buffers + bind groups, one per object per frame.
    model_uniform_pool: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
    /// Number of model uniforms currently allocated in the pool.
    model_pool_size: usize,

    // --- Surface format ---
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
}

impl SceneRenderer {
    /// Create a new scene renderer.
    ///
    /// `color_format` -- the swapchain/surface texture format.
    /// `depth_format` -- the depth buffer format (typically Depth32Float).
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        // Create bind group layouts.
        let camera_lights_bgl = create_camera_lights_bind_group_layout(device);
        let model_bgl = create_model_bind_group_layout(device);
        let material_bgl = create_material_bind_group_layout(device);
        let grid_bgl = create_grid_bind_group_layout(device);

        // --- PBR render pipeline ---
        let pbr_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene_pbr_shader"),
            source: wgpu::ShaderSource::Wgsl(PBR_SHADER_WGSL.into()),
        });

        let pbr_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scene_pbr_pipeline_layout"),
                bind_group_layouts: &[&camera_lights_bgl, &model_bgl, &material_bgl],
                push_constant_ranges: &[],
            });

        let pbr_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene_pbr_pipeline"),
            layout: Some(&pbr_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &pbr_shader_module,
                entry_point: Some("vs_main"),
                buffers: &[SceneVertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &pbr_shader_module,
                entry_point: Some("fs_main"),
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
        });

        // --- Grid render pipeline ---
        let grid_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene_grid_shader"),
            source: wgpu::ShaderSource::Wgsl(GRID_SHADER_WGSL.into()),
        });

        let grid_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scene_grid_pipeline_layout"),
                bind_group_layouts: &[&grid_bgl],
                push_constant_ranges: &[],
            });

        let grid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene_grid_pipeline"),
            layout: Some(&grid_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &grid_shader_module,
                entry_point: Some("vs_grid"),
                buffers: &[GridVertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &grid_shader_module,
                entry_point: Some("fs_grid"),
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
                cull_mode: None, // Grid is visible from both sides
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: false, // Grid does not write depth
                depth_compare: wgpu::CompareFunction::LessEqual,
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
        });

        // --- Create per-frame uniform buffers ---
        let camera_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene_camera_ub"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lights_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene_lights_ub"),
            size: std::mem::size_of::<LightsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_lights_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_camera_lights_bg"),
            layout: &camera_lights_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lights_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Grid camera buffer (separate because the grid bind group layout
        // only has one binding).
        let grid_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_camera_ub"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grid_camera_bg"),
            layout: &grid_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: grid_camera_buffer.as_entire_binding(),
            }],
        });

        Self {
            pbr_pipeline,
            camera_lights_bgl,
            model_bgl,
            material_bgl,
            grid_pipeline,
            grid_bgl,
            camera_uniform_buffer,
            lights_uniform_buffer,
            camera_lights_bind_group,
            grid_camera_buffer,
            grid_bind_group,
            model_uniform_pool: Vec::new(),
            model_pool_size: 0,
            color_format,
            depth_format,
        }
    }

    /// Access the material bind group layout (needed to create `MaterialGpuData`).
    pub fn material_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.material_bgl
    }

    /// Access the color format.
    pub fn color_format(&self) -> wgpu::TextureFormat {
        self.color_format
    }

    /// Access the depth format.
    pub fn depth_format(&self) -> wgpu::TextureFormat {
        self.depth_format
    }

    /// Ensure the model uniform pool has at least `count` entries.
    fn ensure_model_pool(&mut self, device: &wgpu::Device, count: usize) {
        while self.model_uniform_pool.len() < count {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("model_ub_{}", self.model_uniform_pool.len())),
                size: std::mem::size_of::<ModelUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("model_bg_{}", self.model_uniform_pool.len())),
                layout: &self.model_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });

            self.model_uniform_pool.push((buffer, bind_group));
        }
    }

    /// Update per-frame camera and light uniforms on the GPU.
    pub fn update_frame_uniforms(
        &self,
        queue: &wgpu::Queue,
        camera: &SceneCamera,
        lights: &SceneLights,
    ) {
        let camera_uniform = CameraUniform::from_camera(camera);
        queue.write_buffer(
            &self.camera_uniform_buffer,
            0,
            bytemuck::bytes_of(&camera_uniform),
        );

        let lights_uniform = lights.to_uniform();
        queue.write_buffer(
            &self.lights_uniform_buffer,
            0,
            bytemuck::bytes_of(&lights_uniform),
        );

        // Also update the grid camera buffer with the same data.
        queue.write_buffer(
            &self.grid_camera_buffer,
            0,
            bytemuck::bytes_of(&camera_uniform),
        );
    }

    /// Render a scene: all objects, followed by the grid.
    ///
    /// This method begins a render pass, draws all objects with the PBR pipeline,
    /// then draws the grid, and finishes the render pass.
    ///
    /// Returns the finished `wgpu::CommandBuffer`.
    pub fn render_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        objects: &[RenderObject],
        meshes: &HashMap<MeshId, MeshGpuData>,
        materials: &HashMap<MaterialId, MaterialGpuData>,
        camera: &SceneCamera,
        lights: &SceneLights,
        grid: Option<&GridGpuData>,
        clear_color: [f64; 4],
    ) -> wgpu::CommandBuffer {
        // Update camera and light uniforms.
        self.update_frame_uniforms(queue, camera, lights);

        // Ensure we have enough model uniform slots.
        self.ensure_model_pool(device, objects.len());

        // Upload per-object model uniforms.
        for (i, obj) in objects.iter().enumerate() {
            let model_uniform = ModelUniform::from_world_matrix(obj.world_matrix);
            queue.write_buffer(
                &self.model_uniform_pool[i].0,
                0,
                bytemuck::bytes_of(&model_uniform),
            );
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scene_render_encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: clear_color[0],
                            g: clear_color[1],
                            b: clear_color[2],
                            a: clear_color[3],
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // --- Draw all PBR objects ---
            render_pass.set_pipeline(&self.pbr_pipeline);
            render_pass.set_bind_group(0, Some(&self.camera_lights_bind_group), &[]);

            for (i, obj) in objects.iter().enumerate() {
                // Look up the mesh.
                let mesh = match meshes.get(&obj.mesh_id) {
                    Some(m) => m,
                    None => continue,
                };

                // Look up the material.
                let mat = match materials.get(&obj.material_id) {
                    Some(m) => m,
                    None => continue,
                };

                // Bind per-object model uniform.
                render_pass.set_bind_group(1, Some(&self.model_uniform_pool[i].1), &[]);
                // Bind material.
                render_pass.set_bind_group(2, Some(&mat.bind_group), &[]);

                // Set vertex and index buffers.
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

                // Draw.
                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }

            // --- Draw the grid ---
            if let Some(grid_data) = grid {
                render_pass.set_pipeline(&self.grid_pipeline);
                render_pass.set_bind_group(0, Some(&self.grid_bind_group), &[]);
                render_pass.set_vertex_buffer(0, grid_data.vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    grid_data.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                render_pass.draw_indexed(0..grid_data.index_count, 0, 0..1);
            }
        }

        encoder.finish()
    }
}

// ============================================================================
// Grid GPU data
// ============================================================================

/// Grid mesh uploaded to the GPU.
pub struct GridGpuData {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

impl GridGpuData {
    /// Generate and upload the grid.
    pub fn new(device: &wgpu::Device, extent: f32, spacing: f32) -> Self {
        use wgpu::util::DeviceExt;

        let (vertices, indices) = generate_grid_lines(extent, spacing);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grid_vb"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grid_ib"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
        }
    }
}

// ============================================================================
// SceneRenderManager -- top-level owner of all GPU rendering state
// ============================================================================

/// The top-level manager that owns all GPU mesh/material data, the scene
/// renderer pipeline, grid data, built-in primitives, and per-frame render
/// queue.
///
/// Usage pattern:
/// ```text
/// let mgr = SceneRenderManager::new(device, queue, surface_format);
///
/// // Add meshes and materials (returns IDs).
/// let cube_id = mgr.add_mesh(&device, &cube_verts, &cube_indices, "cube");
/// let red_id = mgr.add_material(&device, &MaterialParams::red(), "red");
///
/// // Each frame:
/// mgr.clear_queue();
/// mgr.submit(cube_id, red_id, Mat4::IDENTITY);
/// let cmd = mgr.render(&device, &queue, &surface_view, &depth_view, &camera, &lights);
/// queue.submit(std::iter::once(cmd));
/// ```
pub struct SceneRenderManager {
    /// The core scene renderer (pipelines, bind group layouts, uniforms).
    renderer: SceneRenderer,

    /// GPU mesh storage.
    meshes: HashMap<MeshId, MeshGpuData>,
    /// GPU material storage.
    materials: HashMap<MaterialId, MaterialGpuData>,
    /// Next mesh ID to assign.
    next_mesh_id: u64,
    /// Next material ID to assign.
    next_material_id: u64,

    /// Per-frame render queue.
    render_queue: Vec<RenderObject>,

    /// Grid GPU data.
    grid: Option<GridGpuData>,

    /// Clear colour for the frame.
    clear_color: [f64; 4],

    // Built-in primitive mesh IDs.
    /// Built-in unit cube.
    pub builtin_cube: Option<MeshId>,
    /// Built-in sphere (32 segments x 16 rings).
    pub builtin_sphere: Option<MeshId>,
    /// Built-in ground plane (10x10, 1 subdivision).
    pub builtin_plane: Option<MeshId>,
    /// Built-in cylinder (32 segments).
    pub builtin_cylinder: Option<MeshId>,
    /// Built-in cone (32 segments).
    pub builtin_cone: Option<MeshId>,
    /// Built-in torus (48 major, 24 minor).
    pub builtin_torus: Option<MeshId>,
    /// Built-in icosphere (2 subdivisions).
    pub builtin_icosphere: Option<MeshId>,

    // Built-in material IDs.
    /// Default grey material.
    pub builtin_material_default: Option<MaterialId>,
    /// Red material.
    pub builtin_material_red: Option<MaterialId>,
    /// Green material.
    pub builtin_material_green: Option<MaterialId>,
    /// Blue material.
    pub builtin_material_blue: Option<MaterialId>,
    /// White material.
    pub builtin_material_white: Option<MaterialId>,
    /// Gold metallic material.
    pub builtin_material_gold: Option<MaterialId>,
    /// Chrome metallic material.
    pub builtin_material_chrome: Option<MaterialId>,
    /// Copper metallic material.
    pub builtin_material_copper: Option<MaterialId>,
}

impl SceneRenderManager {
    /// Create a new scene render manager with default settings.
    ///
    /// Automatically creates:
    /// - The PBR + grid render pipelines
    /// - Built-in primitive meshes (cube, sphere, plane, cylinder, cone, torus)
    /// - Built-in materials (default, red, green, blue, white, gold, chrome)
    /// - The ground grid
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let renderer = SceneRenderer::new(device, color_format, depth_format);

        let mut mgr = Self {
            renderer,
            meshes: HashMap::new(),
            materials: HashMap::new(),
            next_mesh_id: 1,
            next_material_id: 1,
            render_queue: Vec::new(),
            grid: None,
            clear_color: [0.08, 0.08, 0.1, 1.0],
            builtin_cube: None,
            builtin_sphere: None,
            builtin_plane: None,
            builtin_cylinder: None,
            builtin_cone: None,
            builtin_torus: None,
            builtin_icosphere: None,
            builtin_material_default: None,
            builtin_material_red: None,
            builtin_material_green: None,
            builtin_material_blue: None,
            builtin_material_white: None,
            builtin_material_gold: None,
            builtin_material_chrome: None,
            builtin_material_copper: None,
        };

        // Create built-in meshes.
        mgr.create_builtin_meshes(device);

        // Create built-in materials.
        mgr.create_builtin_materials(device);

        // Create the ground grid.
        mgr.grid = Some(GridGpuData::new(device, 20.0, 1.0));

        mgr
    }

    /// Create all built-in primitive meshes.
    fn create_builtin_meshes(&mut self, device: &wgpu::Device) {
        // Cube
        {
            let (verts, idxs) = generate_cube();
            self.builtin_cube = Some(self.add_mesh(device, &verts, &idxs, "builtin_cube"));
        }
        // Sphere
        {
            let (verts, idxs) = generate_sphere(32, 16);
            self.builtin_sphere = Some(self.add_mesh(device, &verts, &idxs, "builtin_sphere"));
        }
        // Plane
        {
            let (verts, idxs) = generate_plane(5.0, 1);
            self.builtin_plane = Some(self.add_mesh(device, &verts, &idxs, "builtin_plane"));
        }
        // Cylinder
        {
            let (verts, idxs) = generate_cylinder(32);
            self.builtin_cylinder = Some(self.add_mesh(device, &verts, &idxs, "builtin_cylinder"));
        }
        // Cone
        {
            let (verts, idxs) = generate_cone(32);
            self.builtin_cone = Some(self.add_mesh(device, &verts, &idxs, "builtin_cone"));
        }
        // Torus
        {
            let (verts, idxs) = generate_torus(48, 24, 0.35, 0.15);
            self.builtin_torus = Some(self.add_mesh(device, &verts, &idxs, "builtin_torus"));
        }
        // Icosphere
        {
            let (verts, idxs) = generate_icosphere(2);
            self.builtin_icosphere = Some(self.add_mesh(device, &verts, &idxs, "builtin_icosphere"));
        }
    }

    /// Create all built-in materials.
    fn create_builtin_materials(&mut self, device: &wgpu::Device) {
        self.builtin_material_default =
            Some(self.add_material(device, &MaterialParams::default(), "builtin_default"));
        self.builtin_material_red =
            Some(self.add_material(device, &MaterialParams::red(), "builtin_red"));
        self.builtin_material_green =
            Some(self.add_material(device, &MaterialParams::green(), "builtin_green"));
        self.builtin_material_blue =
            Some(self.add_material(device, &MaterialParams::blue(), "builtin_blue"));
        self.builtin_material_white =
            Some(self.add_material(device, &MaterialParams::white(), "builtin_white"));
        self.builtin_material_gold =
            Some(self.add_material(device, &MaterialParams::gold(), "builtin_gold"));
        self.builtin_material_chrome =
            Some(self.add_material(device, &MaterialParams::chrome(), "builtin_chrome"));
        self.builtin_material_copper =
            Some(self.add_material(device, &MaterialParams::copper(), "builtin_copper"));
    }

    /// Add a mesh to the GPU and return its ID.
    pub fn add_mesh(
        &mut self,
        device: &wgpu::Device,
        vertices: &[SceneVertex],
        indices: &[u32],
        label: &str,
    ) -> MeshId {
        let id = MeshId(self.next_mesh_id);
        self.next_mesh_id += 1;

        let gpu_data = MeshGpuData::upload(device, vertices, indices, label);
        self.meshes.insert(id, gpu_data);
        id
    }

    /// Add a material to the GPU and return its ID.
    pub fn add_material(
        &mut self,
        device: &wgpu::Device,
        params: &MaterialParams,
        label: &str,
    ) -> MaterialId {
        let id = MaterialId(self.next_material_id);
        self.next_material_id += 1;

        let gpu_data = MaterialGpuData::upload(
            device,
            params,
            self.renderer.material_bind_group_layout(),
            label,
        );
        self.materials.insert(id, gpu_data);
        id
    }

    /// Update an existing material's parameters.
    pub fn update_material(
        &mut self,
        queue: &wgpu::Queue,
        material_id: MaterialId,
        params: &MaterialParams,
    ) {
        if let Some(mat) = self.materials.get_mut(&material_id) {
            mat.update(queue, params);
        }
    }

    /// Remove a mesh from GPU storage.
    pub fn remove_mesh(&mut self, mesh_id: MeshId) {
        self.meshes.remove(&mesh_id);
    }

    /// Remove a material from GPU storage.
    pub fn remove_material(&mut self, material_id: MaterialId) {
        self.materials.remove(&material_id);
    }

    /// Queue an object for rendering this frame.
    pub fn submit(&mut self, mesh_id: MeshId, material_id: MaterialId, transform: Mat4) {
        self.render_queue.push(RenderObject {
            mesh_id,
            material_id,
            world_matrix: transform,
        });
    }

    /// Queue a render object directly.
    pub fn submit_object(&mut self, obj: RenderObject) {
        self.render_queue.push(obj);
    }

    /// Clear the per-frame render queue.
    pub fn clear_queue(&mut self) {
        self.render_queue.clear();
    }

    /// Set the clear colour for the frame.
    pub fn set_clear_color(&mut self, color: [f64; 4]) {
        self.clear_color = color;
    }

    /// Enable or disable the ground grid.
    pub fn set_grid_enabled(&mut self, device: &wgpu::Device, enabled: bool) {
        if enabled && self.grid.is_none() {
            self.grid = Some(GridGpuData::new(device, 20.0, 1.0));
        } else if !enabled {
            self.grid = None;
        }
    }

    /// Reconfigure the grid with custom extent and spacing.
    pub fn configure_grid(&mut self, device: &wgpu::Device, extent: f32, spacing: f32) {
        self.grid = Some(GridGpuData::new(device, extent, spacing));
    }

    /// Get the number of queued render objects.
    pub fn queued_object_count(&self) -> usize {
        self.render_queue.len()
    }

    /// Get the number of registered meshes.
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// Get the number of registered materials.
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    /// Access a mesh ID for a built-in primitive by name.
    pub fn builtin_mesh(&self, name: &str) -> Option<MeshId> {
        match name {
            "cube" => self.builtin_cube,
            "sphere" => self.builtin_sphere,
            "plane" => self.builtin_plane,
            "cylinder" => self.builtin_cylinder,
            "cone" => self.builtin_cone,
            "torus" => self.builtin_torus,
            "icosphere" => self.builtin_icosphere,
            _ => None,
        }
    }

    /// Access a material ID for a built-in material by name.
    pub fn builtin_material(&self, name: &str) -> Option<MaterialId> {
        match name {
            "default" | "grey" | "gray" => self.builtin_material_default,
            "red" => self.builtin_material_red,
            "green" => self.builtin_material_green,
            "blue" => self.builtin_material_blue,
            "white" => self.builtin_material_white,
            "gold" => self.builtin_material_gold,
            "chrome" => self.builtin_material_chrome,
            "copper" => self.builtin_material_copper,
            _ => None,
        }
    }

    /// Render the current frame.
    ///
    /// Draws all queued objects with PBR lighting, followed by the grid.
    /// Returns a finished `wgpu::CommandBuffer` ready for queue submission.
    ///
    /// The render queue is NOT automatically cleared -- call `clear_queue()`
    /// before submitting next frame's objects.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera: &SceneCamera,
        lights: &SceneLights,
    ) -> wgpu::CommandBuffer {
        self.renderer.render_scene(
            device,
            queue,
            color_view,
            depth_view,
            &self.render_queue,
            &self.meshes,
            &self.materials,
            camera,
            lights,
            self.grid.as_ref(),
            self.clear_color,
        )
    }

    /// Convenience: render a demo scene.
    ///
    /// Places several built-in primitives in a grid pattern with different
    /// materials, adds default outdoor lighting, and renders the frame.
    pub fn render_demo_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera: &SceneCamera,
    ) -> wgpu::CommandBuffer {
        self.clear_queue();

        // Place primitives in a row.
        let primitives = [
            (self.builtin_cube, self.builtin_material_red),
            (self.builtin_sphere, self.builtin_material_gold),
            (self.builtin_cylinder, self.builtin_material_blue),
            (self.builtin_cone, self.builtin_material_green),
            (self.builtin_torus, self.builtin_material_chrome),
            (self.builtin_icosphere, self.builtin_material_copper),
        ];

        let spacing = 2.0f32;
        let start_x = -(primitives.len() as f32 - 1.0) * spacing * 0.5;

        for (i, (mesh_id, mat_id)) in primitives.iter().enumerate() {
            if let (Some(mesh), Some(mat)) = (mesh_id, mat_id) {
                let x = start_x + i as f32 * spacing;
                let transform = Mat4::from_translation(Vec3::new(x, 0.5, 0.0));
                self.submit(*mesh, *mat, transform);
            }
        }

        // Ground plane.
        if let (Some(plane), Some(mat)) = (self.builtin_plane, self.builtin_material_white) {
            let transform = Mat4::from_translation(Vec3::new(0.0, -0.01, 0.0));
            self.submit(plane, mat, transform);
        }

        let lights = SceneLights::default_outdoor();
        self.render(device, queue, color_view, depth_view, camera, &lights)
    }

    /// Access the underlying SceneRenderer.
    pub fn scene_renderer(&self) -> &SceneRenderer {
        &self.renderer
    }

    /// Access the underlying SceneRenderer mutably.
    pub fn scene_renderer_mut(&mut self) -> &mut SceneRenderer {
        &mut self.renderer
    }
}

// ============================================================================
// Depth texture helper
// ============================================================================

/// Create a depth texture and its view for the given dimensions.
pub fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene_depth_texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

// ============================================================================
// Camera construction helpers
// ============================================================================

impl SceneCamera {
    /// Create a camera with a perspective projection.
    pub fn perspective(
        eye: Vec3,
        target: Vec3,
        up: Vec3,
        fov_y_radians: f32,
        aspect_ratio: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self {
            view: Mat4::look_at_rh(eye, target, up),
            projection: Mat4::perspective_rh(fov_y_radians, aspect_ratio, near, far),
            position: eye,
        }
    }

    /// Create a camera with an orthographic projection.
    pub fn orthographic(
        eye: Vec3,
        target: Vec3,
        up: Vec3,
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self {
            view: Mat4::look_at_rh(eye, target, up),
            projection: Mat4::orthographic_rh(left, right, bottom, top, near, far),
            position: eye,
        }
    }

    /// Update the camera's aspect ratio (recomputes projection if perspective).
    pub fn set_aspect_ratio(&mut self, aspect: f32) {
        // Re-derive FOV from the current projection matrix.
        // projection[1][1] = 1 / tan(fov/2) for a perspective matrix.
        let proj_11 = self.projection.col(1).y;
        if proj_11 > 0.0 {
            let fov_y = 2.0 * (1.0 / proj_11).atan();
            let near = self.projection.col(3).z / (self.projection.col(2).z - 1.0);
            let far = self.projection.col(3).z / (self.projection.col(2).z + 1.0);
            self.projection = Mat4::perspective_rh(fov_y, aspect, near.abs(), far.abs());
        }
    }

    /// Orbit the camera around a target point.
    pub fn orbit(&mut self, target: Vec3, yaw: f32, pitch: f32, distance: f32) {
        let pitch = pitch.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );

        let x = distance * pitch.cos() * yaw.sin();
        let y = distance * pitch.sin();
        let z = distance * pitch.cos() * yaw.cos();

        self.position = target + Vec3::new(x, y, z);
        self.view = Mat4::look_at_rh(self.position, target, Vec3::Y);
    }
}

// ============================================================================
// Render statistics
// ============================================================================

/// Per-frame render statistics.
#[derive(Debug, Clone, Default)]
pub struct SceneRenderStats {
    /// Number of draw calls issued.
    pub draw_calls: u32,
    /// Total number of triangles submitted.
    pub total_triangles: u32,
    /// Total number of vertices submitted.
    pub total_vertices: u32,
    /// Number of unique meshes used.
    pub unique_meshes: u32,
    /// Number of unique materials used.
    pub unique_materials: u32,
}

impl SceneRenderManager {
    /// Compute render statistics for the current queue.
    pub fn compute_stats(&self) -> SceneRenderStats {
        let mut stats = SceneRenderStats::default();
        stats.draw_calls = self.render_queue.len() as u32;

        let mut seen_meshes = std::collections::HashSet::new();
        let mut seen_materials = std::collections::HashSet::new();

        for obj in &self.render_queue {
            seen_meshes.insert(obj.mesh_id);
            seen_materials.insert(obj.material_id);

            if let Some(mesh) = self.meshes.get(&obj.mesh_id) {
                stats.total_triangles += mesh.index_count / 3;
                stats.total_vertices += mesh.vertex_count;
            }
        }

        stats.unique_meshes = seen_meshes.len() as u32;
        stats.unique_materials = seen_materials.len() as u32;
        stats
    }
}

// ============================================================================
// Transform helper constructors
// ============================================================================

/// Utility functions for building common transform matrices.
pub struct Transform;

impl Transform {
    /// Translation only.
    pub fn translation(x: f32, y: f32, z: f32) -> Mat4 {
        Mat4::from_translation(Vec3::new(x, y, z))
    }

    /// Translation + uniform scale.
    pub fn translation_scale(x: f32, y: f32, z: f32, scale: f32) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            Vec3::splat(scale),
            glam::Quat::IDENTITY,
            Vec3::new(x, y, z),
        )
    }

    /// Translation + non-uniform scale.
    pub fn translation_scale_non_uniform(
        x: f32,
        y: f32,
        z: f32,
        sx: f32,
        sy: f32,
        sz: f32,
    ) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            Vec3::new(sx, sy, sz),
            glam::Quat::IDENTITY,
            Vec3::new(x, y, z),
        )
    }

    /// Translation + rotation (Euler angles in radians, YXZ order).
    pub fn translation_rotation(
        x: f32,
        y: f32,
        z: f32,
        rot_x: f32,
        rot_y: f32,
        rot_z: f32,
    ) -> Mat4 {
        let rotation = glam::Quat::from_euler(glam::EulerRot::YXZ, rot_y, rot_x, rot_z);
        Mat4::from_rotation_translation(rotation, Vec3::new(x, y, z))
    }

    /// Full TRS (translation + rotation + uniform scale).
    pub fn trs(
        x: f32,
        y: f32,
        z: f32,
        rot_x: f32,
        rot_y: f32,
        rot_z: f32,
        scale: f32,
    ) -> Mat4 {
        let rotation = glam::Quat::from_euler(glam::EulerRot::YXZ, rot_y, rot_x, rot_z);
        Mat4::from_scale_rotation_translation(
            Vec3::splat(scale),
            rotation,
            Vec3::new(x, y, z),
        )
    }

    /// Rotation around the Y axis.
    pub fn rotation_y(angle_radians: f32) -> Mat4 {
        Mat4::from_rotation_y(angle_radians)
    }

    /// Rotation around an arbitrary axis.
    pub fn rotation_axis(axis: Vec3, angle_radians: f32) -> Mat4 {
        Mat4::from_axis_angle(axis, angle_radians)
    }

    /// Look-at matrix suitable for placing an object that "faces" a target.
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
        // Inverse of the view matrix = the model matrix that places
        // something at `eye` looking toward `target`.
        Mat4::look_at_rh(eye, target, up).inverse()
    }
}

// ============================================================================
// Additional primitive: capsule
// ============================================================================

/// Generate a capsule (cylinder with hemisphere caps) along the Y axis.
///
/// `segments` = circumference divisions.
/// `rings` = rings per hemisphere.
/// `radius` = capsule radius.
/// `height` = total height including the hemispheres.
pub fn generate_capsule(
    segments: u32,
    rings: u32,
    radius: f32,
    height: f32,
) -> (Vec<SceneVertex>, Vec<u32>) {
    let segments = segments.max(4);
    let rings = rings.max(2);
    let half_cylinder_h = (height - 2.0 * radius).max(0.0) * 0.5;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // --- Top hemisphere ---
    for ring in 0..=rings {
        let phi = std::f32::consts::FRAC_PI_2 * (1.0 - ring as f32 / rings as f32);
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        let y = sin_phi * radius + half_cylinder_h;

        for seg in 0..=segments {
            let theta = 2.0 * std::f32::consts::PI * (seg as f32) / (segments as f32);
            let x = cos_phi * theta.cos();
            let z = cos_phi * theta.sin();

            let normal = Vec3::new(x, sin_phi, z).normalize();
            let position = Vec3::new(x * radius, y, z * radius);
            let u = seg as f32 / segments as f32;
            let v = ring as f32 / (rings as f32 * 2.0 + 1.0);

            vertices.push(SceneVertex::with_pos_normal_uv(position, normal, Vec2::new(u, v)));
        }
    }

    // --- Cylinder body ---
    let cylinder_rings = 2u32;
    for ring in 0..=cylinder_rings {
        let t = ring as f32 / cylinder_rings as f32;
        let y = half_cylinder_h - t * 2.0 * half_cylinder_h;

        for seg in 0..=segments {
            let theta = 2.0 * std::f32::consts::PI * (seg as f32) / (segments as f32);
            let x = theta.cos();
            let z = theta.sin();

            let normal = Vec3::new(x, 0.0, z);
            let position = Vec3::new(x * radius, y, z * radius);
            let u = seg as f32 / segments as f32;
            let v_base = rings as f32 / (rings as f32 * 2.0 + 1.0);
            let v = v_base + t * (1.0 / (rings as f32 * 2.0 + 1.0));

            vertices.push(SceneVertex::with_pos_normal_uv(position, normal, Vec2::new(u, v)));
        }
    }

    // --- Bottom hemisphere ---
    for ring in 0..=rings {
        let phi = -std::f32::consts::FRAC_PI_2 * (ring as f32 / rings as f32);
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        let y = sin_phi * radius - half_cylinder_h;

        for seg in 0..=segments {
            let theta = 2.0 * std::f32::consts::PI * (seg as f32) / (segments as f32);
            let x = cos_phi * theta.cos();
            let z = cos_phi * theta.sin();

            let normal = Vec3::new(x, sin_phi, z).normalize();
            let position = Vec3::new(x * radius, y, z * radius);
            let u = seg as f32 / segments as f32;
            let v = 1.0 - ring as f32 / (rings as f32 * 2.0 + 1.0);

            vertices.push(SceneVertex::with_pos_normal_uv(position, normal, Vec2::new(u, v)));
        }
    }

    // --- Generate indices ---
    let total_rings = rings + 1 + cylinder_rings + 1 + rings;
    let verts_per_ring = segments + 1;

    for ring in 0..total_rings {
        for seg in 0..segments {
            let current = ring * verts_per_ring + seg;
            let next = current + verts_per_ring;

            indices.push(current);
            indices.push(next);
            indices.push(current + 1);

            indices.push(current + 1);
            indices.push(next);
            indices.push(next + 1);
        }
    }

    (vertices, indices)
}

// ============================================================================
// Wireframe pipeline (for debug visualization)
// ============================================================================

/// WGSL shader for wireframe rendering with uniform colour.
pub const WIREFRAME_SHADER_WGSL: &str = r#"
// ---------------------------------------------------------------------------
// Genovo Engine -- Wireframe Debug Shader
// ---------------------------------------------------------------------------
//
// Simple unlit shader for drawing wireframe overlays. Uses the same camera
// uniform as the main scene, and a per-object model+color uniform.
// ---------------------------------------------------------------------------

struct CameraUniform {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
};

struct WireframeUniform {
    world: mat4x4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> wireframe: WireframeUniform;

struct WireVertexInput {
    @location(0) position: vec3<f32>,
};

struct WireVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_wireframe(input: WireVertexInput) -> WireVertexOutput {
    var output: WireVertexOutput;
    let world_pos = wireframe.world * vec4<f32>(input.position, 1.0);
    output.clip_position = camera.view_projection * world_pos;
    output.color = wireframe.color;
    return output;
}

@fragment
fn fs_wireframe(input: WireVertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
"#;

/// GPU uniform for wireframe objects (model + colour).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WireframeUniform {
    pub world: [[f32; 4]; 4],
    pub color: [f32; 4],
}

// ============================================================================
// Skybox shader (placeholder for future cubemap-based skybox)
// ============================================================================

/// WGSL shader for a simple gradient sky (no cubemap, just procedural).
pub const SKY_GRADIENT_SHADER_WGSL: &str = r#"
// ---------------------------------------------------------------------------
// Genovo Engine -- Gradient Sky Shader
// ---------------------------------------------------------------------------
//
// Draws a full-screen quad with a vertical gradient from ground colour to
// sky colour. Rendered at the far plane so everything draws on top.
// ---------------------------------------------------------------------------

struct SkyUniform {
    inv_view_projection: mat4x4<f32>,
    sky_color_top: vec4<f32>,
    sky_color_bottom: vec4<f32>,
    ground_color: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    params: vec4<f32>,  // .x = sun_size, .y = sun_intensity
};

@group(0) @binding(0) var<uniform> sky: SkyUniform;

struct SkyVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Full-screen triangle (no vertex buffer needed).
@vertex
fn vs_sky(@builtin(vertex_index) vertex_index: u32) -> SkyVertexOutput {
    var output: SkyVertexOutput;

    // Generate a full-screen triangle.
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );

    let pos = positions[vertex_index];
    output.position = vec4<f32>(pos, 0.99999, 1.0); // At near-far plane
    output.uv = pos * 0.5 + vec2<f32>(0.5);

    return output;
}

@fragment
fn fs_sky(input: SkyVertexOutput) -> @location(0) vec4<f32> {
    // Reconstruct world-space ray direction from UV.
    let ndc = vec4<f32>(input.uv.x * 2.0 - 1.0, (1.0 - input.uv.y) * 2.0 - 1.0, 1.0, 1.0);
    let world_dir_h = sky.inv_view_projection * ndc;
    let world_dir = normalize(world_dir_h.xyz / world_dir_h.w);

    // Vertical gradient.
    let t = world_dir.y * 0.5 + 0.5; // [-1,1] -> [0,1]

    // Below horizon: blend toward ground colour.
    var color: vec3<f32>;
    if world_dir.y < 0.0 {
        let ground_t = clamp(-world_dir.y * 5.0, 0.0, 1.0);
        color = mix(sky.sky_color_bottom.xyz, sky.ground_color.xyz, ground_t);
    } else {
        color = mix(sky.sky_color_bottom.xyz, sky.sky_color_top.xyz, t);
    }

    // Sun disc.
    let sun_dir = normalize(sky.sun_direction.xyz);
    let sun_dot = max(dot(world_dir, sun_dir), 0.0);
    let sun_size = sky.params.x;
    let sun_intensity = sky.params.y;

    if sun_dot > 1.0 - sun_size {
        let sun_t = (sun_dot - (1.0 - sun_size)) / sun_size;
        let sun_glow = pow(sun_t, 4.0) * sun_intensity;
        color = color + sky.sun_color.xyz * sun_glow;
    }

    // Atmospheric scattering glow around the sun.
    let glow = pow(sun_dot, 32.0) * 0.3;
    color = color + sky.sun_color.xyz * glow;

    return vec4<f32>(color, 1.0);
}
"#;

/// GPU uniform for the sky gradient shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SkyUniform {
    pub inv_view_projection: [[f32; 4]; 4],
    pub sky_color_top: [f32; 4],
    pub sky_color_bottom: [f32; 4],
    pub ground_color: [f32; 4],
    pub sun_direction: [f32; 4],
    pub sun_color: [f32; 4],
    pub params: [f32; 4],
}

impl Default for SkyUniform {
    fn default() -> Self {
        Self {
            inv_view_projection: Mat4::IDENTITY.to_cols_array_2d(),
            sky_color_top: [0.1, 0.3, 0.8, 1.0],
            sky_color_bottom: [0.5, 0.6, 0.8, 1.0],
            ground_color: [0.15, 0.12, 0.1, 1.0],
            sun_direction: [0.3, 0.8, 0.5, 0.0],
            sun_color: [1.0, 0.95, 0.8, 1.0],
            params: [0.005, 3.0, 0.0, 0.0],
        }
    }
}

// ============================================================================
// Additional mesh utilities
// ============================================================================

/// Recalculate smooth normals for a mesh in-place.
pub fn recalculate_normals(vertices: &mut [SceneVertex], indices: &[u32]) {
    // Zero all normals.
    for v in vertices.iter_mut() {
        v.normal = [0.0, 0.0, 0.0];
    }

    // Accumulate face normals (area-weighted).
    let tri_count = indices.len() / 3;
    for i in 0..tri_count {
        let i0 = indices[i * 3] as usize;
        let i1 = indices[i * 3 + 1] as usize;
        let i2 = indices[i * 3 + 2] as usize;

        let p0 = Vec3::from(vertices[i0].position);
        let p1 = Vec3::from(vertices[i1].position);
        let p2 = Vec3::from(vertices[i2].position);

        let edge1 = p1 - p0;
        let edge2 = p2 - p0;
        let face_normal = edge1.cross(edge2); // Not normalised = area-weighted.

        for idx in [i0, i1, i2] {
            vertices[idx].normal[0] += face_normal.x;
            vertices[idx].normal[1] += face_normal.y;
            vertices[idx].normal[2] += face_normal.z;
        }
    }

    // Normalise all normals.
    for v in vertices.iter_mut() {
        let n = Vec3::from(v.normal);
        let len = n.length();
        if len > 1e-8 {
            v.normal = (n / len).into();
        } else {
            v.normal = [0.0, 1.0, 0.0]; // Default to up.
        }
    }
}

/// Calculate tangent vectors for a mesh (Mikktspace-like, simplified).
pub fn calculate_tangents(vertices: &mut [SceneVertex], indices: &[u32]) {
    // This is a simplified tangent calculation. For production use,
    // you would want a full MikkTSpace implementation.

    let tri_count = indices.len() / 3;
    let mut tangents: Vec<Vec3> = vec![Vec3::ZERO; vertices.len()];
    let mut bitangents: Vec<Vec3> = vec![Vec3::ZERO; vertices.len()];

    for i in 0..tri_count {
        let i0 = indices[i * 3] as usize;
        let i1 = indices[i * 3 + 1] as usize;
        let i2 = indices[i * 3 + 2] as usize;

        let p0 = Vec3::from(vertices[i0].position);
        let p1 = Vec3::from(vertices[i1].position);
        let p2 = Vec3::from(vertices[i2].position);

        let uv0 = Vec2::from(vertices[i0].uv);
        let uv1 = Vec2::from(vertices[i1].uv);
        let uv2 = Vec2::from(vertices[i2].uv);

        let dp1 = p1 - p0;
        let dp2 = p2 - p0;

        let duv1 = uv1 - uv0;
        let duv2 = uv2 - uv0;

        let r = 1.0 / (duv1.x * duv2.y - duv1.y * duv2.x + 1e-10);

        let tangent = (dp1 * duv2.y - dp2 * duv1.y) * r;
        let bitangent = (dp2 * duv1.x - dp1 * duv2.x) * r;

        for idx in [i0, i1, i2] {
            tangents[idx] += tangent;
            bitangents[idx] += bitangent;
        }
    }

    // Orthogonalise and store.
    for (i, v) in vertices.iter_mut().enumerate() {
        let n = Vec3::from(v.normal);
        let t = tangents[i];
        let b = bitangents[i];

        // Gram-Schmidt orthogonalise.
        let tangent = (t - n * n.dot(t)).normalize();
        // Handedness.
        let handedness = if n.cross(t).dot(b) < 0.0 { -1.0 } else { 1.0 };

        // Store tangent in the color channel's xyz (since we don't have a
        // dedicated tangent attribute in SceneVertex). This is a pragmatic
        // choice for this renderer -- a production vertex format would have
        // a separate tangent vec4.
        // For now, we skip storing tangents as the PBR shader does not use
        // normal maps yet.
        let _ = (tangent, handedness);
    }
}

/// Merge multiple meshes into a single mesh. Each sub-mesh is transformed
/// by its corresponding matrix.
pub fn merge_meshes(
    mesh_list: &[(Vec<SceneVertex>, Vec<u32>, Mat4)],
) -> (Vec<SceneVertex>, Vec<u32>) {
    let total_verts: usize = mesh_list.iter().map(|(v, _, _)| v.len()).sum();
    let total_indices: usize = mesh_list.iter().map(|(_, i, _)| i.len()).sum();

    let mut out_verts = Vec::with_capacity(total_verts);
    let mut out_indices = Vec::with_capacity(total_indices);

    for (verts, idxs, transform) in mesh_list {
        let base_vertex = out_verts.len() as u32;

        // Transform and append vertices.
        for v in verts {
            let pos = *transform * Vec4::new(v.position[0], v.position[1], v.position[2], 1.0);
            let normal_mat = transform.inverse().transpose();
            let norm = normal_mat * Vec4::new(v.normal[0], v.normal[1], v.normal[2], 0.0);
            let n = Vec3::new(norm.x, norm.y, norm.z).normalize();

            out_verts.push(SceneVertex {
                position: [pos.x, pos.y, pos.z],
                normal: n.into(),
                uv: v.uv,
                color: v.color,
            });
        }

        // Offset and append indices.
        for idx in idxs {
            out_indices.push(base_vertex + idx);
        }
    }

    (out_verts, out_indices)
}

// ============================================================================
// Bounding geometry
// ============================================================================

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// Compute the AABB of a set of vertices.
    pub fn from_vertices(vertices: &[SceneVertex]) -> Self {
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);

        for v in vertices {
            let p = Vec3::from(v.position);
            min = min.min(p);
            max = max.max(p);
        }

        Self { min, max }
    }

    /// Center of the AABB.
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Half-extents of the AABB.
    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Bounding sphere radius (distance from center to a corner).
    pub fn bounding_radius(&self) -> f32 {
        self.half_extents().length()
    }

    /// Check if a point is inside the AABB.
    pub fn contains_point(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Check if this AABB intersects another.
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Transform the AABB by a matrix (produces a new, larger AABB).
    pub fn transform(&self, matrix: &Mat4) -> Self {
        // Transform all 8 corners and recompute the AABB.
        let corners = [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
        ];

        let mut new_min = Vec3::splat(f32::MAX);
        let mut new_max = Vec3::splat(f32::MIN);

        for corner in &corners {
            let transformed =
                *matrix * Vec4::new(corner.x, corner.y, corner.z, 1.0);
            let p = Vec3::new(transformed.x, transformed.y, transformed.z);
            new_min = new_min.min(p);
            new_max = new_max.max(p);
        }

        Self {
            min: new_min,
            max: new_max,
        }
    }
}

/// Bounding sphere.
#[derive(Debug, Clone, Copy)]
pub struct BoundingSphere {
    pub center: Vec3,
    pub radius: f32,
}

impl BoundingSphere {
    /// Compute a bounding sphere from an AABB.
    pub fn from_aabb(aabb: &Aabb) -> Self {
        Self {
            center: aabb.center(),
            radius: aabb.bounding_radius(),
        }
    }

    /// Compute a bounding sphere from vertices (using AABB-based approach).
    pub fn from_vertices(vertices: &[SceneVertex]) -> Self {
        Self::from_aabb(&Aabb::from_vertices(vertices))
    }

    /// Check if a point is inside the sphere.
    pub fn contains_point(&self, point: Vec3) -> bool {
        (point - self.center).length_squared() <= self.radius * self.radius
    }

    /// Check if this sphere intersects another.
    pub fn intersects(&self, other: &BoundingSphere) -> bool {
        let dist = (self.center - other.center).length();
        dist <= self.radius + other.radius
    }
}

// ============================================================================
// Frustum culling
// ============================================================================

/// A view frustum defined by 6 planes (left, right, bottom, top, near, far).
///
/// Each plane is stored as (a, b, c, d) where ax + by + cz + d = 0,
/// with the normal (a,b,c) pointing inward.
#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    pub planes: [Vec4; 6],
}

impl Frustum {
    /// Extract frustum planes from a view-projection matrix.
    pub fn from_view_projection(vp: Mat4) -> Self {
        let row0 = Vec4::new(vp.col(0).x, vp.col(1).x, vp.col(2).x, vp.col(3).x);
        let row1 = Vec4::new(vp.col(0).y, vp.col(1).y, vp.col(2).y, vp.col(3).y);
        let row2 = Vec4::new(vp.col(0).z, vp.col(1).z, vp.col(2).z, vp.col(3).z);
        let row3 = Vec4::new(vp.col(0).w, vp.col(1).w, vp.col(2).w, vp.col(3).w);

        let mut planes = [
            row3 + row0, // Left
            row3 - row0, // Right
            row3 + row1, // Bottom
            row3 - row1, // Top
            row3 + row2, // Near
            row3 - row2, // Far
        ];

        // Normalise each plane.
        for plane in &mut planes {
            let len = Vec3::new(plane.x, plane.y, plane.z).length();
            if len > 1e-8 {
                *plane /= len;
            }
        }

        Self { planes }
    }

    /// Test if a sphere is inside or intersecting the frustum.
    pub fn test_sphere(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            let dist = plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;
            if dist < -radius {
                return false; // Entirely outside this plane.
            }
        }
        true
    }

    /// Test if an AABB is inside or intersecting the frustum.
    pub fn test_aabb(&self, aabb: &Aabb) -> bool {
        for plane in &self.planes {
            // Find the corner most aligned with the plane normal.
            let px = if plane.x >= 0.0 { aabb.max.x } else { aabb.min.x };
            let py = if plane.y >= 0.0 { aabb.max.y } else { aabb.min.y };
            let pz = if plane.z >= 0.0 { aabb.max.z } else { aabb.min.z };

            let dist = plane.x * px + plane.y * py + plane.z * pz + plane.w;
            if dist < 0.0 {
                return false;
            }
        }
        true
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_cube_counts() {
        let (verts, idxs) = generate_cube();
        assert_eq!(verts.len(), 24);
        assert_eq!(idxs.len(), 36);
    }

    #[test]
    fn test_generate_sphere_counts() {
        let (verts, idxs) = generate_sphere(16, 8);
        assert_eq!(verts.len(), (8 + 1) * (16 + 1));
        assert_eq!(idxs.len(), (8 * 16 * 6) as usize);
    }

    #[test]
    fn test_generate_plane_counts() {
        let (verts, idxs) = generate_plane(5.0, 4);
        assert_eq!(verts.len(), 5 * 5);
        assert_eq!(idxs.len(), 4 * 4 * 6);
    }

    #[test]
    fn test_camera_uniform_size() {
        // Ensure the camera uniform is properly aligned for GPU usage.
        assert_eq!(std::mem::size_of::<CameraUniform>(), 256);
    }

    #[test]
    fn test_lights_uniform_size() {
        // Each GpuLight is 48 bytes (3 * vec4).
        // LightsUniform: ambient(16) + light_count(16) + 8 * 48 = 416
        assert_eq!(std::mem::size_of::<GpuLight>(), 48);
        assert_eq!(
            std::mem::size_of::<LightsUniform>(),
            16 + 16 + MAX_LIGHTS * 48
        );
    }

    #[test]
    fn test_model_uniform_size() {
        // Two mat4x4: 128 bytes.
        assert_eq!(std::mem::size_of::<ModelUniform>(), 128);
    }

    #[test]
    fn test_material_uniform_size() {
        // Four vec4: 64 bytes.
        assert_eq!(std::mem::size_of::<MaterialUniform>(), 64);
    }

    #[test]
    fn test_scene_vertex_size() {
        // 3 + 3 + 2 + 4 = 12 floats = 48 bytes.
        assert_eq!(std::mem::size_of::<SceneVertex>(), 48);
    }

    #[test]
    fn test_grid_vertex_size() {
        // 3 + 4 = 7 floats = 28 bytes.
        assert_eq!(std::mem::size_of::<GridVertex>(), 28);
    }

    #[test]
    fn test_aabb_from_vertices() {
        let verts = vec![
            SceneVertex::with_pos_normal_uv(Vec3::new(-1.0, 0.0, 0.0), Vec3::Y, Vec2::ZERO),
            SceneVertex::with_pos_normal_uv(Vec3::new(1.0, 2.0, 3.0), Vec3::Y, Vec2::ZERO),
        ];
        let aabb = Aabb::from_vertices(&verts);
        assert_eq!(aabb.min, Vec3::new(-1.0, 0.0, 0.0));
        assert_eq!(aabb.max, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_aabb_contains_point() {
        let aabb = Aabb {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        };
        assert!(aabb.contains_point(Vec3::splat(0.5)));
        assert!(!aabb.contains_point(Vec3::splat(1.5)));
    }

    #[test]
    fn test_bounding_sphere_from_aabb() {
        let aabb = Aabb {
            min: Vec3::new(-1.0, -1.0, -1.0),
            max: Vec3::new(1.0, 1.0, 1.0),
        };
        let sphere = BoundingSphere::from_aabb(&aabb);
        assert_eq!(sphere.center, Vec3::ZERO);
        assert!((sphere.radius - 3.0f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_material_presets() {
        let red = MaterialParams::red();
        assert!(red.albedo_color.x > 0.5);
        assert!(red.albedo_color.y < 0.3);

        let gold = MaterialParams::gold();
        assert_eq!(gold.metallic, 1.0);
        assert!(gold.roughness < 0.5);
    }

    #[test]
    fn test_lights_uniform_packing() {
        let lights = SceneLights {
            ambient_color: Vec3::new(0.1, 0.1, 0.1),
            ambient_intensity: 1.0,
            directional_lights: vec![DirectionalLight::default()],
            point_lights: vec![PointLight::default()],
        };
        let uniform = lights.to_uniform();
        assert_eq!(uniform.light_count[0] as u32, 2);
        assert_eq!(uniform.lights[0].params[0], 1.0); // directional
        assert_eq!(uniform.lights[1].params[0], 2.0); // point
    }

    #[test]
    fn test_frustum_sphere_inside() {
        let vp = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0)
            * Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let frustum = Frustum::from_view_projection(vp);

        // Origin should be visible.
        assert!(frustum.test_sphere(Vec3::ZERO, 0.5));

        // Very far behind camera should not be visible.
        assert!(!frustum.test_sphere(Vec3::new(0.0, 0.0, 200.0), 0.5));
    }

    #[test]
    fn test_generate_cylinder() {
        let (verts, idxs) = generate_cylinder(8);
        assert!(!verts.is_empty());
        assert!(!idxs.is_empty());
        // Check all indices are valid.
        for &idx in &idxs {
            assert!((idx as usize) < verts.len());
        }
    }

    #[test]
    fn test_generate_cone() {
        let (verts, idxs) = generate_cone(8);
        assert!(!verts.is_empty());
        assert!(!idxs.is_empty());
        for &idx in &idxs {
            assert!((idx as usize) < verts.len());
        }
    }

    #[test]
    fn test_generate_torus() {
        let (verts, idxs) = generate_torus(16, 8, 0.35, 0.15);
        assert!(!verts.is_empty());
        assert!(!idxs.is_empty());
        for &idx in &idxs {
            assert!((idx as usize) < verts.len());
        }
    }

    #[test]
    fn test_generate_icosphere() {
        let (verts, idxs) = generate_icosphere(1);
        assert!(!verts.is_empty());
        assert!(!idxs.is_empty());
        for &idx in &idxs {
            assert!((idx as usize) < verts.len());
        }
    }

    #[test]
    fn test_generate_capsule() {
        let (verts, idxs) = generate_capsule(8, 4, 0.5, 2.0);
        assert!(!verts.is_empty());
        assert!(!idxs.is_empty());
        for &idx in &idxs {
            assert!(
                (idx as usize) < verts.len(),
                "Index {} out of bounds (vertex count: {})",
                idx,
                verts.len()
            );
        }
    }

    #[test]
    fn test_grid_lines_generation() {
        let (verts, idxs) = generate_grid_lines(10.0, 1.0);
        assert!(!verts.is_empty());
        assert!(!idxs.is_empty());
        // Each grid line is 4 vertices + 6 indices (a thin quad).
        assert_eq!(verts.len() % 4, 0);
        assert_eq!(idxs.len() % 6, 0);
    }

    #[test]
    fn test_recalculate_normals() {
        // Simple triangle.
        let mut verts = vec![
            SceneVertex::with_pos_normal_uv(Vec3::ZERO, Vec3::ZERO, Vec2::ZERO),
            SceneVertex::with_pos_normal_uv(Vec3::X, Vec3::ZERO, Vec2::ZERO),
            SceneVertex::with_pos_normal_uv(Vec3::Y, Vec3::ZERO, Vec2::ZERO),
        ];
        let idxs = vec![0, 1, 2];

        recalculate_normals(&mut verts, &idxs);

        // Normal should point in +Z direction for a CCW triangle in XY plane.
        for v in &verts {
            let n = Vec3::from(v.normal);
            assert!((n.z - (-1.0)).abs() < 0.01 || (n.z - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_merge_meshes() {
        let (cube_v, cube_i) = generate_cube();
        let (sphere_v, sphere_i) = generate_sphere(4, 2);

        let merged = merge_meshes(&[
            (cube_v.clone(), cube_i.clone(), Mat4::IDENTITY),
            (
                sphere_v.clone(),
                sphere_i.clone(),
                Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)),
            ),
        ]);

        assert_eq!(merged.0.len(), cube_v.len() + sphere_v.len());
        assert_eq!(merged.1.len(), cube_i.len() + sphere_i.len());
    }

    #[test]
    fn test_transform_helpers() {
        let t = Transform::translation(1.0, 2.0, 3.0);
        let col3 = t.col(3);
        assert!((col3.x - 1.0).abs() < 1e-5);
        assert!((col3.y - 2.0).abs() < 1e-5);
        assert!((col3.z - 3.0).abs() < 1e-5);

        let ts = Transform::translation_scale(0.0, 0.0, 0.0, 2.0);
        assert!((ts.col(0).x - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_scene_camera_orbit() {
        let mut cam = SceneCamera::default();
        cam.orbit(Vec3::ZERO, 0.0, 0.3, 10.0);
        assert!((cam.position.length() - 10.0).abs() < 0.1);
    }
}
