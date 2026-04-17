// engine/render/src/gpu_particles_render.rs
//
// GPU Particle Rendering for the Genovo engine.
//
// # Features
//
// - `ParticleBillboardRenderer`: render particles as camera-facing quads
// - `ParticleVertex`: position, size, color, rotation, UV
// - WGSL billboard vertex shader: expand point to quad facing camera
// - Soft particles: fade near depth buffer
// - Sprite sheet animation: select UV frame based on particle age
// - Additive / alpha blending modes
// - Back-to-front sorting for correct transparency

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};

// ============================================================================
// Constants
// ============================================================================

/// Maximum number of particles per emitter.
pub const MAX_PARTICLES: usize = 65536;

/// Maximum sprite sheet frames.
pub const MAX_SPRITE_FRAMES: usize = 64;

/// Soft particle fade distance.
pub const SOFT_PARTICLE_DISTANCE: f32 = 0.5;

// ============================================================================
// Particle data types
// ============================================================================

/// Per-particle data uploaded to the GPU as a storage buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuParticle {
    /// World-space position.
    pub position: [f32; 3],
    /// Size (width, height if non-square, or just uniform size).
    pub size: f32,
    /// RGBA colour.
    pub color: [f32; 4],
    /// Rotation in radians (around the view axis for billboards).
    pub rotation: f32,
    /// Normalized age (0 = just born, 1 = about to die).
    pub age: f32,
    /// Sprite sheet frame index (float for interpolation).
    pub frame: f32,
    /// Custom parameter (e.g., velocity magnitude for stretching).
    pub custom: f32,
}

impl Default for GpuParticle {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            size: 1.0,
            color: [1.0, 1.0, 1.0, 1.0],
            rotation: 0.0,
            age: 0.0,
            frame: 0.0,
            custom: 0.0,
        }
    }
}

/// Particle system rendering parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ParticleRenderParams {
    /// Camera right vector in world space.
    pub camera_right: [f32; 4],
    /// Camera up vector in world space.
    pub camera_up: [f32; 4],
    /// Camera position in world space.
    pub camera_position: [f32; 4],
    /// View-projection matrix.
    pub view_projection: [[f32; 4]; 4],
    /// Sprite sheet: .x = columns, .y = rows, .z = total_frames, .w = frame_rate.
    pub sprite_sheet: [f32; 4],
    /// .x = soft_particle_distance, .y = sort_enabled, .z = time, .w = particle_count.
    pub render_params: [f32; 4],
    /// .x = screen_width, .y = screen_height, .z = near, .w = far.
    pub screen_params: [f32; 4],
}

/// Blending mode for particles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleBlendMode {
    /// Standard alpha blending.
    AlphaBlend,
    /// Additive blending (for fire, sparks, etc.).
    Additive,
    /// Pre-multiplied alpha.
    PremultipliedAlpha,
}

impl ParticleBlendMode {
    /// Convert to wgpu blend state.
    pub fn to_wgpu_blend(&self) -> wgpu::BlendState {
        match self {
            ParticleBlendMode::AlphaBlend => wgpu::BlendState::ALPHA_BLENDING,
            ParticleBlendMode::Additive => wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            },
            ParticleBlendMode::PremultipliedAlpha => wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
        }
    }
}

// ============================================================================
// Billboard particle shader
// ============================================================================

/// WGSL shader for billboard particle rendering.
pub const PARTICLE_BILLBOARD_SHADER_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Particle Billboard Shader
// ============================================================================
//
// Renders particles as camera-facing quads (billboards).
// Each particle is a point expanded to 4 vertices forming a quad.
//
// Uses instance_index to select the particle and vertex_index for the
// quad corner (0-5 for two triangles).

struct ParticleData {
    position: vec3<f32>,
    size: f32,
    color: vec4<f32>,
    rotation: f32,
    age: f32,
    frame: f32,
    custom: f32,
};

struct RenderParams {
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    camera_position: vec4<f32>,
    view_projection: mat4x4<f32>,
    sprite_sheet: vec4<f32>,
    render_params: vec4<f32>,
    screen_params: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> particles: array<ParticleData>;
@group(0) @binding(1) var<uniform> params: RenderParams;

@group(1) @binding(0) var particle_texture: texture_2d<f32>;
@group(1) @binding(1) var particle_sampler: sampler;
@group(1) @binding(2) var depth_texture: texture_2d<f32>;
@group(1) @binding(3) var depth_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_position: vec3<f32>,
    @location(3) particle_depth: f32,
};

// Compute sprite sheet UV for a given frame.
fn sprite_uv(base_uv: vec2<f32>, frame: f32, cols: f32, rows: f32) -> vec2<f32> {
    let frame_idx = u32(frame);
    let col = f32(frame_idx % u32(cols));
    let row = f32(frame_idx / u32(cols));

    let frame_w = 1.0 / cols;
    let frame_h = 1.0 / rows;

    return vec2<f32>(
        (col + base_uv.x) * frame_w,
        (row + base_uv.y) * frame_h
    );
}

@vertex
fn vs_particle_billboard(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var output: VertexOutput;

    let particle = particles[instance_index];

    // 6 vertices per quad (2 triangles): 0,1,2,  2,1,3
    // Map vertex_index to corner: 0=BL, 1=BR, 2=TL, 3=TR
    let corner_map = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);
    let corner = corner_map[vertex_index % 6u];

    // UV for this corner.
    let uv_x = f32(corner & 1u);       // 0 or 1
    let uv_y = 1.0 - f32(corner >> 1u); // 1 or 0

    // Offset from center: [-0.5, 0.5]
    let offset_x = uv_x - 0.5;
    let offset_y = uv_y - 0.5;

    // Apply rotation.
    let cos_r = cos(particle.rotation);
    let sin_r = sin(particle.rotation);
    let rotated_x = offset_x * cos_r - offset_y * sin_r;
    let rotated_y = offset_x * sin_r + offset_y * cos_r;

    // Billboard: expand using camera right and up vectors.
    let right = params.camera_right.xyz * particle.size;
    let up = params.camera_up.xyz * particle.size;

    let world_pos = particle.position + right * rotated_x + up * rotated_y;
    output.world_position = world_pos;

    // Project to clip space.
    output.clip_position = params.view_projection * vec4<f32>(world_pos, 1.0);

    // Compute UV (with sprite sheet support).
    let cols = params.sprite_sheet.x;
    let rows = params.sprite_sheet.y;
    let total_frames = params.sprite_sheet.z;

    var uv = vec2<f32>(uv_x, uv_y);
    if total_frames > 1.0 {
        uv = sprite_uv(uv, particle.frame, cols, rows);
    }
    output.uv = uv;

    // Pass through colour.
    output.color = particle.color;

    // Store depth for soft particles.
    output.particle_depth = output.clip_position.z / output.clip_position.w;

    return output;
}

// Linearize a depth buffer value.
fn linearize_depth(depth: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - depth * (far - near));
}

@fragment
fn fs_particle_billboard(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample particle texture.
    let tex_color = textureSample(particle_texture, particle_sampler, input.uv);
    var color = tex_color * input.color;

    // Soft particle: fade near scene geometry.
    let soft_distance = params.render_params.x;
    if soft_distance > 0.0 {
        let screen_uv = vec2<f32>(
            input.clip_position.x / params.screen_params.x,
            input.clip_position.y / params.screen_params.y
        );

        let scene_depth_raw = textureSample(depth_texture, depth_sampler, screen_uv).x;
        let near = params.screen_params.z;
        let far = params.screen_params.w;

        let scene_depth = linearize_depth(scene_depth_raw, near, far);
        let particle_depth = linearize_depth(input.particle_depth, near, far);

        let depth_diff = scene_depth - particle_depth;
        let soft_factor = clamp(depth_diff / soft_distance, 0.0, 1.0);

        color.w = color.w * soft_factor;
    }

    // Discard fully transparent fragments.
    if color.w < 0.004 {
        discard;
    }

    return color;
}
"#;

/// WGSL shader for velocity-stretched particle billboards.
pub const PARTICLE_STRETCHED_SHADER_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Stretched Particle Billboard
// ============================================================================
//
// Renders particles stretched along their velocity direction.
// Used for rain, sparks, and fast-moving effects.

struct ParticleData {
    position: vec3<f32>,
    size: f32,
    color: vec4<f32>,
    rotation: f32,
    age: f32,
    frame: f32,
    custom: f32,  // velocity magnitude for stretch
};

struct RenderParams {
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    camera_position: vec4<f32>,
    view_projection: mat4x4<f32>,
    sprite_sheet: vec4<f32>,
    render_params: vec4<f32>,
    screen_params: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> particles: array<ParticleData>;
@group(0) @binding(1) var<uniform> params: RenderParams;

@group(1) @binding(0) var particle_texture: texture_2d<f32>;
@group(1) @binding(1) var particle_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_stretched(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var output: VertexOutput;
    let particle = particles[instance_index];

    let corner_map = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);
    let corner = corner_map[vertex_index % 6u];

    let uv_x = f32(corner & 1u);
    let uv_y = 1.0 - f32(corner >> 1u);

    let offset_x = uv_x - 0.5;
    let offset_y = uv_y - 0.5;

    // Use rotation as velocity direction angle.
    let vel_dir = vec3<f32>(cos(particle.rotation), 0.0, sin(particle.rotation));
    let stretch = max(particle.custom, 1.0);

    // Camera-facing with velocity stretch.
    let view_dir = normalize(params.camera_position.xyz - particle.position);
    let right = normalize(cross(vel_dir, view_dir)) * particle.size;
    let up = vel_dir * particle.size * stretch;

    let world_pos = particle.position + right * offset_x + up * offset_y;

    output.clip_position = params.view_projection * vec4<f32>(world_pos, 1.0);
    output.uv = vec2<f32>(uv_x, uv_y);
    output.color = particle.color;

    return output;
}

@fragment
fn fs_stretched(input: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(particle_texture, particle_sampler, input.uv);
    var color = tex_color * input.color;
    if color.w < 0.004 { discard; }
    return color;
}
"#;

// ============================================================================
// Particle billboard renderer
// ============================================================================

/// Renders particles as camera-facing billboards.
pub struct ParticleBillboardRenderer {
    /// Alpha blend pipeline.
    pipeline_alpha: wgpu::RenderPipeline,
    /// Additive blend pipeline.
    pipeline_additive: wgpu::RenderPipeline,
    /// Stretched billboard pipeline.
    pipeline_stretched: wgpu::RenderPipeline,
    /// Particle data bind group layout (group 0).
    particle_data_bgl: wgpu::BindGroupLayout,
    /// Texture bind group layout (group 1).
    texture_bgl: wgpu::BindGroupLayout,
    /// Particle storage buffer.
    particle_buffer: wgpu::Buffer,
    /// Render params uniform buffer.
    params_buffer: wgpu::Buffer,
    /// Bind group for particle data.
    particle_bind_group: wgpu::BindGroup,
    /// Maximum particles.
    max_particles: usize,
}

impl ParticleBillboardRenderer {
    /// Create a new particle billboard renderer.
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        max_particles: usize,
    ) -> Self {
        // --- Bind group layouts ---
        let particle_data_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("particle_data_bgl"),
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
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let texture_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("particle_texture_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
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
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle_pipeline_layout"),
            bind_group_layouts: &[&particle_data_bgl, &texture_bgl],
            push_constant_ranges: &[],
        });

        let billboard_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("particle_billboard_shader"),
                source: wgpu::ShaderSource::Wgsl(PARTICLE_BILLBOARD_SHADER_WGSL.into()),
            });

        let stretched_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("particle_stretched_shader"),
                source: wgpu::ShaderSource::Wgsl(PARTICLE_STRETCHED_SHADER_WGSL.into()),
            });

        // Create pipeline for each blend mode.
        let create_pipeline =
            |shader: &wgpu::ShaderModule,
             vs_entry: &str,
             fs_entry: &str,
             blend: wgpu::BlendState,
             label: &str|
             -> wgpu::RenderPipeline {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: shader,
                        entry_point: Some(vs_entry),
                        buffers: &[], // No vertex buffer, all data from storage.
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: shader,
                        entry_point: Some(fs_entry),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: Some(blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: None, // Particles are double-sided.
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: depth_format,
                        depth_write_enabled: false, // Particles don't write depth.
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                })
            };

        let pipeline_alpha = create_pipeline(
            &billboard_shader,
            "vs_particle_billboard",
            "fs_particle_billboard",
            ParticleBlendMode::AlphaBlend.to_wgpu_blend(),
            "particle_alpha_pipeline",
        );

        let pipeline_additive = create_pipeline(
            &billboard_shader,
            "vs_particle_billboard",
            "fs_particle_billboard",
            ParticleBlendMode::Additive.to_wgpu_blend(),
            "particle_additive_pipeline",
        );

        let pipeline_stretched = create_pipeline(
            &stretched_shader,
            "vs_stretched",
            "fs_stretched",
            ParticleBlendMode::AlphaBlend.to_wgpu_blend(),
            "particle_stretched_pipeline",
        );

        // --- Create buffers ---
        let particle_buffer_size =
            (max_particles * std::mem::size_of::<GpuParticle>()) as u64;

        let particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_storage_buffer"),
            size: particle_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_render_params"),
            size: std::mem::size_of::<ParticleRenderParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let particle_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle_data_bg"),
            layout: &particle_data_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline_alpha,
            pipeline_additive,
            pipeline_stretched,
            particle_data_bgl,
            texture_bgl,
            particle_buffer,
            params_buffer,
            particle_bind_group,
            max_particles,
        }
    }

    /// Upload particle data to the GPU.
    pub fn upload_particles(
        &self,
        queue: &wgpu::Queue,
        particles: &[GpuParticle],
    ) {
        let count = particles.len().min(self.max_particles);
        if count == 0 {
            return;
        }
        let data = bytemuck::cast_slice(&particles[..count]);
        queue.write_buffer(&self.particle_buffer, 0, data);
    }

    /// Upload render parameters.
    pub fn upload_params(
        &self,
        queue: &wgpu::Queue,
        camera_view: Mat4,
        camera_proj: Mat4,
        camera_pos: Vec3,
        particle_count: u32,
        sprite_cols: f32,
        sprite_rows: f32,
        soft_distance: f32,
        time: f32,
        screen_width: f32,
        screen_height: f32,
        near: f32,
        far: f32,
    ) {
        let view_inv = camera_view.inverse();
        let right = Vec3::new(view_inv.x_axis.x, view_inv.x_axis.y, view_inv.x_axis.z);
        let up = Vec3::new(view_inv.y_axis.x, view_inv.y_axis.y, view_inv.y_axis.z);

        let vp = camera_proj * camera_view;

        let params = ParticleRenderParams {
            camera_right: [right.x, right.y, right.z, 0.0],
            camera_up: [up.x, up.y, up.z, 0.0],
            camera_position: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
            view_projection: vp.to_cols_array_2d(),
            sprite_sheet: [sprite_cols, sprite_rows, sprite_cols * sprite_rows, 0.0],
            render_params: [soft_distance, 0.0, time, particle_count as f32],
            screen_params: [screen_width, screen_height, near, far],
        };

        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// Create a texture bind group for particle rendering.
    pub fn create_texture_bind_group(
        &self,
        device: &wgpu::Device,
        particle_texture_view: &wgpu::TextureView,
        particle_sampler: &wgpu::Sampler,
        depth_texture_view: &wgpu::TextureView,
        depth_sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle_texture_bg"),
            layout: &self.texture_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(particle_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(particle_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(depth_sampler),
                },
            ],
        })
    }

    /// Render particles with the specified blend mode.
    pub fn render<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        texture_bind_group: &'a wgpu::BindGroup,
        particle_count: u32,
        blend_mode: ParticleBlendMode,
        use_stretched: bool,
    ) {
        if particle_count == 0 {
            return;
        }

        let pipeline = if use_stretched {
            &self.pipeline_stretched
        } else {
            match blend_mode {
                ParticleBlendMode::Additive => &self.pipeline_additive,
                _ => &self.pipeline_alpha,
            }
        };

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.particle_bind_group, &[]);
        pass.set_bind_group(1, texture_bind_group, &[]);

        // 6 vertices per particle (2 triangles), instanced by particle count.
        pass.draw(0..6, 0..particle_count);
    }

    /// Access the texture bind group layout.
    pub fn texture_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.texture_bgl
    }
}

// ============================================================================
// Particle sorting
// ============================================================================

/// Sort particles back-to-front relative to the camera.
pub fn sort_particles_back_to_front(
    particles: &mut [GpuParticle],
    camera_pos: Vec3,
) {
    particles.sort_by(|a, b| {
        let dist_a = Vec3::from(a.position).distance_squared(camera_pos);
        let dist_b = Vec3::from(b.position).distance_squared(camera_pos);
        dist_b
            .partial_cmp(&dist_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Sort particles front-to-back (for depth pre-pass).
pub fn sort_particles_front_to_back(
    particles: &mut [GpuParticle],
    camera_pos: Vec3,
) {
    particles.sort_by(|a, b| {
        let dist_a = Vec3::from(a.position).distance_squared(camera_pos);
        let dist_b = Vec3::from(b.position).distance_squared(camera_pos);
        dist_a
            .partial_cmp(&dist_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

// ============================================================================
// Particle texture generation utilities
// ============================================================================

/// Generate a circular gradient particle texture.
pub fn generate_particle_circle_texture(size: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    let center = size as f32 * 0.5;

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - center + 0.5;
            let dy = y as f32 - center + 0.5;
            let dist = (dx * dx + dy * dy).sqrt() / center;

            let alpha = (1.0 - dist).max(0.0);
            let alpha_smooth = alpha * alpha * (3.0 - 2.0 * alpha); // Smoothstep.

            let byte = (alpha_smooth * 255.0) as u8;
            pixels.extend_from_slice(&[255, 255, 255, byte]);
        }
    }

    pixels
}

/// Generate a soft glow particle texture.
pub fn generate_particle_glow_texture(size: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    let center = size as f32 * 0.5;

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - center + 0.5;
            let dy = y as f32 - center + 0.5;
            let dist = (dx * dx + dy * dy).sqrt() / center;

            // Exponential falloff for glow.
            let alpha = (-dist * dist * 4.0).exp();

            let byte = (alpha.min(1.0) * 255.0) as u8;
            pixels.extend_from_slice(&[255, 255, 255, byte]);
        }
    }

    pixels
}

/// Generate a spark/streak particle texture.
pub fn generate_particle_spark_texture(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);

    for y in 0..height {
        for x in 0..width {
            let u = x as f32 / (width - 1).max(1) as f32;
            let v = y as f32 / (height - 1).max(1) as f32;

            // Centered V coordinate.
            let vc = (v - 0.5).abs() * 2.0;

            // Bright center line fading to edges.
            let cross_fade = (1.0 - vc * vc * 4.0).max(0.0);
            // Length fade.
            let length_fade = (1.0 - (u * 2.0 - 1.0).abs()).max(0.0);

            let alpha = cross_fade * length_fade;
            let byte = (alpha.min(1.0) * 255.0) as u8;
            pixels.extend_from_slice(&[255, 255, 255, byte]);
        }
    }

    pixels
}
