// engine/render/src/gpu_instancing.rs
//
// GPU Instanced Rendering for the Genovo engine.
//
// # Features
//
// - `InstanceBuffer`: per-instance data (world matrix, colour/variation)
// - `InstancedMeshRenderer`: draw thousands of objects in one draw call
// - WGSL instanced vertex shader reading per-instance data from storage buffer
// - Instance data upload with dirty tracking
// - CPU frustum culling with instance compaction
// - LOD per instance based on distance
// - Foliage/grass instancing with wind animation parameter

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};

// ============================================================================
// Constants
// ============================================================================

/// Maximum instances per draw call.
pub const MAX_INSTANCES: usize = 65536;

/// Maximum LOD levels.
pub const MAX_LOD_LEVELS: usize = 4;

/// Wind animation speed.
pub const WIND_SPEED: f32 = 2.0;

/// Wind strength.
pub const WIND_STRENGTH: f32 = 0.1;

// ============================================================================
// Instance data types
// ============================================================================

/// Per-instance data for GPU instanced rendering.
///
/// This is stored in a storage buffer and read by the vertex shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct InstanceData {
    /// World transform matrix (column-major).
    pub world: [[f32; 4]; 4],
    /// Normal matrix (inverse-transpose of world, column-major).
    pub normal_matrix: [[f32; 4]; 4],
    /// Per-instance colour tint (RGBA).
    pub color: [f32; 4],
    /// .x = wind_phase, .y = wind_strength, .z = lod_level, .w = custom_param.
    pub params: [f32; 4],
}

impl InstanceData {
    /// Create instance data from a world matrix.
    pub fn from_matrix(world: Mat4) -> Self {
        let inv = world.inverse();
        let normal_mat = inv.transpose();
        Self {
            world: world.to_cols_array_2d(),
            normal_matrix: normal_mat.to_cols_array_2d(),
            color: [1.0, 1.0, 1.0, 1.0],
            params: [0.0; 4],
        }
    }

    /// Create instance data with colour tint.
    pub fn from_matrix_color(world: Mat4, color: Vec4) -> Self {
        let inv = world.inverse();
        let normal_mat = inv.transpose();
        Self {
            world: world.to_cols_array_2d(),
            normal_matrix: normal_mat.to_cols_array_2d(),
            color: color.into(),
            params: [0.0; 4],
        }
    }

    /// Create foliage instance data with wind parameters.
    pub fn foliage(world: Mat4, color: Vec4, wind_phase: f32, wind_strength: f32) -> Self {
        let inv = world.inverse();
        let normal_mat = inv.transpose();
        Self {
            world: world.to_cols_array_2d(),
            normal_matrix: normal_mat.to_cols_array_2d(),
            color: color.into(),
            params: [wind_phase, wind_strength, 0.0, 0.0],
        }
    }
}

/// Instancing uniform parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct InstancingParamsUniform {
    /// .x = time, .y = wind_direction_x, .z = wind_direction_z, .w = wind_speed.
    pub wind_params: [f32; 4],
    /// .x = instance_count, .y = lod_enabled, .z = frustum_cull_enabled, .w = unused.
    pub instance_params: [f32; 4],
    /// Camera position for LOD distance calculation.
    pub camera_position: [f32; 4],
    /// LOD distances: .x = lod0_dist, .y = lod1_dist, .z = lod2_dist, .w = lod3_dist.
    pub lod_distances: [f32; 4],
}

// ============================================================================
// WGSL instanced vertex shader
// ============================================================================

/// WGSL shader for instanced rendering with per-instance data.
pub const INSTANCED_SHADER_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Instanced Rendering Shader
// ============================================================================
//
// Draws multiple instances of the same mesh using per-instance data from a
// storage buffer. Each instance has its own world matrix and colour tint.
//
// Bind groups:
//   Group 0: Camera + Lights (per frame)
//   Group 1: Instance data storage buffer
//   Group 2: Material

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

struct MaterialUniform {
    albedo_color: vec4<f32>,
    metallic_roughness: vec4<f32>,
    emissive: vec4<f32>,
    flags: vec4<f32>,
};

struct InstanceDataGpu {
    world: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    color: vec4<f32>,
    params: vec4<f32>,
};

struct InstancingParams {
    wind_params: vec4<f32>,
    instance_params: vec4<f32>,
    camera_position: vec4<f32>,
    lod_distances: vec4<f32>,
};

// ---------------------------------------------------------------------------
// Bind groups
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> lights: LightsUniform;

@group(1) @binding(0) var<storage, read> instances: array<InstanceDataGpu>;
@group(1) @binding(1) var<uniform> instancing_params: InstancingParams;

@group(2) @binding(0) var<uniform> material: MaterialUniform;

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
    @location(3) vertex_color: vec4<f32>,
    @location(4) view_dir: vec3<f32>,
    @location(5) instance_color: vec4<f32>,
};

// ---------------------------------------------------------------------------
// Wind animation
// ---------------------------------------------------------------------------

fn apply_wind(position: vec3<f32>, world_y: f32, wind_phase: f32, wind_strength: f32) -> vec3<f32> {
    let time = instancing_params.wind_params.x;
    let wind_dir = vec2<f32>(instancing_params.wind_params.y, instancing_params.wind_params.z);
    let speed = instancing_params.wind_params.w;

    // Height-based wind influence (more at the top).
    let height_factor = clamp(position.y * 2.0, 0.0, 1.0);

    // Wind displacement.
    let phase = time * speed + wind_phase;
    let wave = sin(phase) * 0.5 + sin(phase * 2.3) * 0.3 + sin(phase * 0.7) * 0.2;
    let displacement = wind_dir * wave * wind_strength * height_factor;

    return vec3<f32>(
        position.x + displacement.x,
        position.y,
        position.z + displacement.y
    );
}

// ---------------------------------------------------------------------------
// Vertex shader
// ---------------------------------------------------------------------------

@vertex
fn vs_instanced(
    input: VertexInput,
    @builtin(instance_index) instance_id: u32,
) -> VertexOutput {
    var output: VertexOutput;

    let instance = instances[instance_id];

    // Apply wind animation if wind_strength > 0.
    var local_pos = input.position;
    let wind_strength = instance.params.y;
    if wind_strength > 0.001 {
        local_pos = apply_wind(local_pos, 0.0, instance.params.x, wind_strength);
    }

    // Transform to world space using instance matrix.
    let world_pos = instance.world * vec4<f32>(local_pos, 1.0);
    output.world_position = world_pos.xyz;
    output.clip_position = camera.view_projection * world_pos;

    // Transform normal.
    let raw_normal = (instance.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz;
    output.world_normal = normalize(raw_normal);

    output.uv = input.uv;
    output.vertex_color = input.color;
    output.instance_color = instance.color;
    output.view_dir = normalize(camera.camera_position.xyz - world_pos.xyz);

    return output;
}

// ---------------------------------------------------------------------------
// PBR lighting (compact version)
// ---------------------------------------------------------------------------

fn fresnel_schlick_inst(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let t = clamp(1.0 - cos_theta, 0.0, 1.0);
    let t5 = t * t * t * t * t;
    return f0 + (vec3<f32>(1.0) - f0) * t5;
}

fn distribution_ggx_inst(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + EPSILON);
}

fn geometry_smith_inst(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    let gv = n_dot_v / (n_dot_v * (1.0 - k) + k + EPSILON);
    let gl = n_dot_l / (n_dot_l * (1.0 - k) + k + EPSILON);
    return gv * gl;
}

fn attenuation_inst(distance: f32, range: f32) -> f32 {
    if range <= 0.0 { return 1.0; }
    let d = distance / range;
    let d2 = d * d; let d4 = d2 * d2;
    let factor = clamp(1.0 - d4, 0.0, 1.0);
    return factor * factor / (distance * distance + 1.0);
}

fn linear_to_srgb_inst(c: f32) -> f32 {
    if c <= 0.0031308 { return c * 12.92; }
    return 1.055 * pow(c, INV_GAMMA) - 0.055;
}

fn tone_map_aces_inst(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 1.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + vec3<f32>(b))) / (x * (c * x + vec3<f32>(d)) + vec3<f32>(e)),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------

@fragment
fn fs_instanced(input: VertexOutput) -> @location(0) vec4<f32> {
    // Combine material albedo with instance colour and vertex colour.
    let albedo = material.albedo_color.xyz * input.vertex_color.xyz * input.instance_color.xyz;
    let alpha = material.albedo_color.w * input.vertex_color.w * input.instance_color.w;
    let metallic = clamp(material.metallic_roughness.x, 0.0, 1.0);
    let roughness = clamp(material.metallic_roughness.y, 0.04, 1.0);
    let reflectance = material.metallic_roughness.z;

    let dielectric_f0 = vec3<f32>(0.16 * reflectance * reflectance);
    let f0 = mix(dielectric_f0, albedo, metallic);

    let normal = normalize(input.world_normal);
    let view_dir = normalize(input.view_dir);

    var total_diffuse = vec3<f32>(0.0);
    var total_specular = vec3<f32>(0.0);
    let num_lights = u32(lights.light_count.x);

    for (var i = 0u; i < 8u; i = i + 1u) {
        if i >= num_lights { break; }
        let light = lights.lights[i];
        let lt = light.params.x;

        var light_dir: vec3<f32>;
        var light_color: vec3<f32>;
        var att: f32 = 1.0;

        if abs(lt - LIGHT_TYPE_DIRECTIONAL) < 0.5 {
            light_dir = normalize(-light.position_or_direction.xyz);
            light_color = light.color_intensity.xyz * light.color_intensity.w;
        } else if abs(lt - LIGHT_TYPE_POINT) < 0.5 {
            let to_l = light.position_or_direction.xyz - input.world_position;
            let dist = length(to_l);
            light_dir = normalize(to_l);
            light_color = light.color_intensity.xyz * light.color_intensity.w;
            att = attenuation_inst(dist, light.position_or_direction.w);
        } else {
            continue;
        }

        let n_dot_l = max(dot(normal, light_dir), 0.0);
        if n_dot_l <= 0.0 { continue; }

        let half_vec = normalize(view_dir + light_dir);
        let n_dot_v = max(dot(normal, view_dir), EPSILON);
        let n_dot_h = max(dot(normal, half_vec), 0.0);
        let v_dot_h = max(dot(view_dir, half_vec), 0.0);

        let d = distribution_ggx_inst(n_dot_h, roughness);
        let g = geometry_smith_inst(n_dot_v, n_dot_l, roughness);
        let f = fresnel_schlick_inst(v_dot_h, f0);

        let spec = (d * g * f) / (4.0 * n_dot_v * n_dot_l + EPSILON);
        let k_d = (vec3<f32>(1.0) - f) * (1.0 - metallic);

        let attenuated = light_color * att;
        total_diffuse = total_diffuse + k_d * albedo * INV_PI * attenuated * n_dot_l;
        total_specular = total_specular + spec * attenuated * n_dot_l;
    }

    // Ambient.
    let sky = lights.ambient.xyz * lights.ambient.w;
    let ground = sky * vec3<f32>(0.6, 0.5, 0.4) * 0.5;
    let ambient = mix(ground, sky, normal.y * 0.5 + 0.5);
    let ambient_d = (vec3<f32>(1.0) - f0) * (1.0 - metallic) * albedo * ambient;

    let emissive = material.emissive.xyz * material.emissive.w;

    var final_color = total_diffuse + total_specular + ambient_d + emissive;
    final_color = tone_map_aces_inst(final_color);
    final_color = vec3<f32>(
        linear_to_srgb_inst(final_color.x),
        linear_to_srgb_inst(final_color.y),
        linear_to_srgb_inst(final_color.z)
    );

    return vec4<f32>(final_color, alpha);
}
"#;

// ============================================================================
// Instance buffer
// ============================================================================

/// Manages a GPU buffer of per-instance data.
pub struct InstanceBuffer {
    /// Storage buffer for instance data.
    pub buffer: wgpu::Buffer,
    /// Bind group.
    pub bind_group: wgpu::BindGroup,
    /// Bind group layout.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// CPU-side instance data.
    pub instances: Vec<InstanceData>,
    /// Number of active (visible) instances.
    pub visible_count: u32,
    /// Whether the buffer needs re-upload.
    pub dirty: bool,
    /// Instancing parameters uniform buffer.
    pub params_buffer: wgpu::Buffer,
    /// Maximum capacity.
    pub capacity: usize,
}

impl InstanceBuffer {
    /// Create a new instance buffer with the given capacity.
    pub fn new(device: &wgpu::Device, capacity: usize) -> Self {
        let buffer_size = (capacity * std::mem::size_of::<InstanceData>()) as u64;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instancing_params_buffer"),
            size: std::mem::size_of::<InstancingParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("instance_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("instance_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            buffer,
            bind_group,
            bind_group_layout,
            instances: Vec::with_capacity(capacity),
            visible_count: 0,
            dirty: true,
            params_buffer,
            capacity,
        }
    }

    /// Add an instance.
    pub fn add(&mut self, data: InstanceData) {
        if self.instances.len() < self.capacity {
            self.instances.push(data);
            self.dirty = true;
        }
    }

    /// Clear all instances.
    pub fn clear(&mut self) {
        self.instances.clear();
        self.visible_count = 0;
        self.dirty = true;
    }

    /// Set an instance's data.
    pub fn set(&mut self, index: usize, data: InstanceData) {
        if index < self.instances.len() {
            self.instances[index] = data;
            self.dirty = true;
        }
    }

    /// Upload instance data to the GPU.
    pub fn upload(&mut self, queue: &wgpu::Queue) {
        if !self.dirty || self.instances.is_empty() {
            return;
        }

        let data = bytemuck::cast_slice(&self.instances);
        queue.write_buffer(&self.buffer, 0, data);
        self.visible_count = self.instances.len() as u32;
        self.dirty = false;
    }

    /// Upload instancing parameters.
    pub fn upload_params(
        &self,
        queue: &wgpu::Queue,
        time: f32,
        wind_dir: Vec2,
        wind_speed: f32,
        camera_pos: Vec3,
        lod_distances: [f32; 4],
    ) {
        let params = InstancingParamsUniform {
            wind_params: [time, wind_dir.x, wind_dir.y, wind_speed],
            instance_params: [self.visible_count as f32, 1.0, 1.0, 0.0],
            camera_position: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
            lod_distances,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// Get the number of active instances.
    pub fn instance_count(&self) -> u32 {
        self.visible_count
    }

    /// Perform CPU frustum culling and compact the instance buffer.
    pub fn frustum_cull(
        &mut self,
        camera_view_proj: Mat4,
        bounding_radius: f32,
    ) {
        let planes = extract_frustum_planes(camera_view_proj);

        let mut visible = Vec::with_capacity(self.instances.len());
        for instance in &self.instances {
            let pos = Vec3::new(
                instance.world[3][0],
                instance.world[3][1],
                instance.world[3][2],
            );

            if is_sphere_in_frustum(&planes, pos, bounding_radius) {
                visible.push(*instance);
            }
        }

        self.instances = visible;
        self.visible_count = self.instances.len() as u32;
        self.dirty = true;
    }

    /// Assign LOD levels based on distance to camera.
    pub fn assign_lod_levels(&mut self, camera_pos: Vec3, lod_distances: &[f32; 4]) {
        for instance in &mut self.instances {
            let pos = Vec3::new(
                instance.world[3][0],
                instance.world[3][1],
                instance.world[3][2],
            );
            let dist = pos.distance(camera_pos);

            let lod = if dist < lod_distances[0] {
                0.0
            } else if dist < lod_distances[1] {
                1.0
            } else if dist < lod_distances[2] {
                2.0
            } else {
                3.0
            };

            instance.params[2] = lod;
        }
        self.dirty = true;
    }
}

// ============================================================================
// Frustum culling utilities
// ============================================================================

/// Frustum plane (ax + by + cz + d = 0).
#[derive(Debug, Clone, Copy)]
pub struct FrustumPlane {
    pub normal: Vec3,
    pub distance: f32,
}

/// Extract 6 frustum planes from a view-projection matrix.
pub fn extract_frustum_planes(vp: Mat4) -> [FrustumPlane; 6] {
    let cols = vp.to_cols_array_2d();

    let row = |r: usize| -> Vec4 {
        Vec4::new(cols[0][r], cols[1][r], cols[2][r], cols[3][r])
    };

    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let r3 = row(3);

    let mut planes = [FrustumPlane {
        normal: Vec3::ZERO,
        distance: 0.0,
    }; 6];

    let extract_plane = |v: Vec4| -> FrustumPlane {
        let len = Vec3::new(v.x, v.y, v.z).length();
        if len > 0.0 {
            FrustumPlane {
                normal: Vec3::new(v.x, v.y, v.z) / len,
                distance: v.w / len,
            }
        } else {
            FrustumPlane {
                normal: Vec3::ZERO,
                distance: 0.0,
            }
        }
    };

    planes[0] = extract_plane(r3 + r0); // Left
    planes[1] = extract_plane(r3 - r0); // Right
    planes[2] = extract_plane(r3 + r1); // Bottom
    planes[3] = extract_plane(r3 - r1); // Top
    planes[4] = extract_plane(r3 + r2); // Near
    planes[5] = extract_plane(r3 - r2); // Far

    planes
}

/// Test if a sphere intersects a frustum.
pub fn is_sphere_in_frustum(
    planes: &[FrustumPlane; 6],
    center: Vec3,
    radius: f32,
) -> bool {
    for plane in planes {
        let distance = plane.normal.dot(center) + plane.distance;
        if distance < -radius {
            return false;
        }
    }
    true
}

// ============================================================================
// Instanced mesh renderer
// ============================================================================

/// Renders instanced meshes using a single draw call.
pub struct InstancedMeshRenderer {
    /// Render pipeline for instanced drawing.
    pub pipeline: wgpu::RenderPipeline,
    /// Instance buffer.
    pub instance_buffer: InstanceBuffer,
}

impl InstancedMeshRenderer {
    /// Create a new instanced mesh renderer.
    pub fn new(
        device: &wgpu::Device,
        camera_lights_bgl: &wgpu::BindGroupLayout,
        material_bgl: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        max_instances: usize,
    ) -> Self {
        let instance_buffer = InstanceBuffer::new(device, max_instances);

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("instanced_shader"),
            source: wgpu::ShaderSource::Wgsl(INSTANCED_SHADER_WGSL.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("instanced_pipeline_layout"),
            bind_group_layouts: &[
                camera_lights_bgl,
                &instance_buffer.bind_group_layout,
                material_bgl,
            ],
            push_constant_ranges: &[],
        });

        let vertex_layout = super::scene_renderer::SceneVertex::buffer_layout();

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("instanced_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_instanced"),
                buffers: &[vertex_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_instanced"),
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

        Self {
            pipeline,
            instance_buffer,
        }
    }

    /// Render all instances.
    pub fn render<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        camera_lights_bg: &'a wgpu::BindGroup,
        material_bg: &'a wgpu::BindGroup,
        vertex_buffer: &'a wgpu::Buffer,
        index_buffer: &'a wgpu::Buffer,
        index_count: u32,
    ) {
        if self.instance_buffer.visible_count == 0 {
            return;
        }

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, camera_lights_bg, &[]);
        pass.set_bind_group(1, &self.instance_buffer.bind_group, &[]);
        pass.set_bind_group(2, material_bg, &[]);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..index_count, 0, 0..self.instance_buffer.visible_count);
    }
}

// ============================================================================
// Foliage instancing
// ============================================================================

/// Configuration for foliage/grass instancing.
#[derive(Debug, Clone)]
pub struct FoliageConfig {
    /// Density (instances per unit area).
    pub density: f32,
    /// Placement radius around camera.
    pub radius: f32,
    /// Minimum/maximum scale.
    pub scale_min: f32,
    pub scale_max: f32,
    /// Wind direction (XZ plane).
    pub wind_direction: Vec2,
    /// Wind speed.
    pub wind_speed: f32,
    /// Wind strength.
    pub wind_strength: f32,
    /// Colour variation.
    pub color_variation: f32,
    /// Base colour.
    pub base_color: Vec4,
    /// LOD distances.
    pub lod_distances: [f32; 4],
}

impl Default for FoliageConfig {
    fn default() -> Self {
        Self {
            density: 4.0,
            radius: 50.0,
            scale_min: 0.6,
            scale_max: 1.4,
            wind_direction: Vec2::new(1.0, 0.3).normalize(),
            wind_speed: WIND_SPEED,
            wind_strength: WIND_STRENGTH,
            color_variation: 0.15,
            base_color: Vec4::new(0.3, 0.6, 0.2, 1.0),
            lod_distances: [20.0, 40.0, 60.0, 80.0],
        }
    }
}

/// Generate foliage instance data within a radius of the camera.
pub fn generate_foliage_instances(
    config: &FoliageConfig,
    camera_pos: Vec3,
    seed: u32,
) -> Vec<InstanceData> {
    let mut instances = Vec::new();

    let area = config.radius * config.radius * std::f32::consts::PI;
    let count = (area * config.density) as usize;

    for i in 0..count {
        // Hash-based position.
        let hash1 = pseudo_hash(i as u32 ^ seed);
        let hash2 = pseudo_hash(hash1);
        let hash3 = pseudo_hash(hash2);

        let angle = (hash1 as f32 / u32::MAX as f32) * std::f32::consts::PI * 2.0;
        let dist = (hash2 as f32 / u32::MAX as f32).sqrt() * config.radius;

        let x = camera_pos.x + angle.cos() * dist;
        let z = camera_pos.z + angle.sin() * dist;
        let y = 0.0; // Ground level; in production, sample heightmap.

        // Random scale.
        let scale_t = hash3 as f32 / u32::MAX as f32;
        let scale = config.scale_min + (config.scale_max - config.scale_min) * scale_t;

        // Random rotation.
        let rotation = (hash1 as f32 / u32::MAX as f32) * std::f32::consts::PI * 2.0;

        let world = Mat4::from_translation(Vec3::new(x, y, z))
            * Mat4::from_rotation_y(rotation)
            * Mat4::from_scale(Vec3::splat(scale));

        // Colour variation.
        let color_var = (hash2 as f32 / u32::MAX as f32 - 0.5) * config.color_variation;
        let color = Vec4::new(
            (config.base_color.x + color_var).clamp(0.0, 1.0),
            (config.base_color.y + color_var * 0.5).clamp(0.0, 1.0),
            (config.base_color.z + color_var * 0.3).clamp(0.0, 1.0),
            config.base_color.w,
        );

        // Wind phase variation.
        let wind_phase = (hash3 as f32 / u32::MAX as f32) * std::f32::consts::PI * 2.0;

        instances.push(InstanceData::foliage(
            world,
            color,
            wind_phase,
            config.wind_strength,
        ));
    }

    instances
}

fn pseudo_hash(mut v: u32) -> u32 {
    v = v.wrapping_mul(0x45d9f3b);
    v = (v >> 16) ^ v;
    v = v.wrapping_mul(0x45d9f3b);
    v = (v >> 16) ^ v;
    v
}
