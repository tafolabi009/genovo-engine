// engine/render/src/gpu_texture_sampling.rs
//
// GPU Texture Sampling for the Genovo PBR pipeline.
//
// Adds full texture support to the existing PBR rendering system:
//
// - `TextureManager`: create GPU textures from raw pixel data, generate mipmaps
// - Sampler creation with configurable filtering and address modes
// - Mipmap generation via blit render passes (each mip level = half resolution)
// - WGSL PBR shader update: albedo, normal, metallic-roughness, emissive, AO
// - Normal map sampling with TBN matrix construction in the shader
// - Texture bind group layout compatible with scene_renderer's group 2
//
// # Bind Group Layout (group 2 -- textured material)
//
// | Binding | Type          | Description                          |
// |---------|---------------|--------------------------------------|
// | 0       | Uniform       | MaterialUniform                      |
// | 1       | Texture2D     | Albedo texture                       |
// | 2       | Sampler       | Albedo sampler                       |
// | 3       | Texture2D     | Normal map                           |
// | 4       | Sampler       | Normal sampler                       |
// | 5       | Texture2D     | Metallic-roughness map               |
// | 6       | Sampler       | Metallic-roughness sampler           |
// | 7       | Texture2D     | Emissive map                         |
// | 8       | Sampler       | Emissive sampler                     |
// | 9       | Texture2D     | AO map                               |
// | 10      | Sampler       | AO sampler                           |

use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Constants
// ============================================================================

/// Maximum texture dimension for mipmap generation.
pub const MAX_TEXTURE_DIMENSION: u32 = 8192;

/// Maximum mip levels we will ever generate.
pub const MAX_MIP_LEVELS: u32 = 14; // log2(8192) + 1

/// Default anisotropy level for samplers.
pub const DEFAULT_ANISOTROPY: u16 = 16;

// ============================================================================
// Texture handle
// ============================================================================

/// Opaque handle to a managed GPU texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuTextureHandle(pub u64);

/// Describes how a texture should be filtered and addressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFilterMode {
    /// Nearest-neighbour (pixelated).
    Nearest,
    /// Bilinear filtering.
    Linear,
    /// Trilinear filtering (linear mip interpolation).
    Trilinear,
}

/// Texture coordinate wrapping mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureAddressMode {
    /// Repeat tiling.
    Repeat,
    /// Mirror repeat.
    MirrorRepeat,
    /// Clamp to edge pixel.
    ClampToEdge,
    /// Clamp to a border colour.
    ClampToBorder,
}

impl TextureAddressMode {
    /// Convert to wgpu address mode.
    pub fn to_wgpu(self) -> wgpu::AddressMode {
        match self {
            TextureAddressMode::Repeat => wgpu::AddressMode::Repeat,
            TextureAddressMode::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
            TextureAddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            TextureAddressMode::ClampToBorder => wgpu::AddressMode::ClampToBorder,
        }
    }
}

// ============================================================================
// Texture descriptor for creation
// ============================================================================

/// Parameters for creating a GPU texture.
#[derive(Debug, Clone)]
pub struct TextureDescriptor {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel format.
    pub format: wgpu::TextureFormat,
    /// Whether to generate mipmaps.
    pub generate_mipmaps: bool,
    /// Label for debugging.
    pub label: String,
    /// Usage flags (texture binding is always included).
    pub usage: wgpu::TextureUsages,
}

impl Default for TextureDescriptor {
    fn default() -> Self {
        Self {
            width: 1,
            height: 1,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            generate_mipmaps: true,
            label: String::from("texture"),
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
        }
    }
}

// ============================================================================
// GPU texture wrapper
// ============================================================================

/// A GPU texture with its view, dimensions, and mip count.
pub struct GpuTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub format: wgpu::TextureFormat,
}

impl GpuTexture {
    /// Compute the number of mip levels for the given dimensions.
    pub fn compute_mip_levels(width: u32, height: u32) -> u32 {
        let max_dim = width.max(height) as f32;
        (max_dim.log2().floor() as u32 + 1).min(MAX_MIP_LEVELS)
    }
}

// ============================================================================
// Mipmap generation pipeline
// ============================================================================

/// WGSL shader for mipmap generation via blit.
///
/// Samples the previous mip level with bilinear filtering and writes to the
/// current level. This is a fullscreen triangle shader.
pub const MIPMAP_BLIT_SHADER_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Mipmap Blit Shader
// ============================================================================
//
// Generates a single mip level by sampling the previous level with bilinear
// filtering. Uses the fullscreen triangle technique (no vertex buffer needed).

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_mipmap(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;

    // Fullscreen triangle: 3 vertices cover the entire screen.
    // Vertex 0: (-1, -1), Vertex 1: (3, -1), Vertex 2: (-1, 3)
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);

    output.position = vec4<f32>(x, y, 0.0, 1.0);
    // UV: map from clip space to [0,1] with Y flip for texture coordinates.
    output.uv = vec2<f32>(
        (x + 1.0) * 0.5,
        (1.0 - y) * 0.5
    );

    return output;
}

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

@fragment
fn fs_mipmap(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample with bilinear filtering from the source (previous mip level).
    return textureSample(source_texture, source_sampler, input.uv);
}
"#;

/// Pipeline and resources for mipmap generation.
pub struct MipmapGenerator {
    /// The render pipeline that blits from one mip level to the next.
    pipeline_srgb: wgpu::RenderPipeline,
    /// Pipeline for linear (non-sRGB) textures.
    pipeline_linear: wgpu::RenderPipeline,
    /// Pipeline for Rgba16Float HDR textures.
    pipeline_hdr: wgpu::RenderPipeline,
    /// Bind group layout: source_texture + sampler.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Bilinear sampler used for downsampling.
    blit_sampler: wgpu::Sampler,
}

impl MipmapGenerator {
    /// Create the mipmap generation pipeline.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mipmap_blit_shader"),
            source: wgpu::ShaderSource::Wgsl(MIPMAP_BLIT_SHADER_WGSL.into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mipmap_bgl"),
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
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mipmap_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mipmap_blit_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let create_pipeline = |format: wgpu::TextureFormat, label: &str| -> wgpu::RenderPipeline {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: Some("vs_mipmap"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_module,
                    entry_point: Some("fs_mipmap"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };

        let pipeline_srgb = create_pipeline(
            wgpu::TextureFormat::Rgba8UnormSrgb,
            "mipmap_pipeline_srgb",
        );
        let pipeline_linear = create_pipeline(
            wgpu::TextureFormat::Rgba8Unorm,
            "mipmap_pipeline_linear",
        );
        let pipeline_hdr = create_pipeline(
            wgpu::TextureFormat::Rgba16Float,
            "mipmap_pipeline_hdr",
        );

        Self {
            pipeline_srgb,
            pipeline_linear,
            pipeline_hdr,
            bind_group_layout,
            blit_sampler,
        }
    }

    /// Select the appropriate pipeline for the texture format.
    fn select_pipeline(&self, format: wgpu::TextureFormat) -> &wgpu::RenderPipeline {
        match format {
            wgpu::TextureFormat::Rgba8UnormSrgb => &self.pipeline_srgb,
            wgpu::TextureFormat::Rgba16Float => &self.pipeline_hdr,
            _ => &self.pipeline_linear,
        }
    }

    /// Generate all mip levels for the given texture.
    ///
    /// The texture must have been created with `RENDER_ATTACHMENT` usage and
    /// mip level 0 must already contain the base image data.
    pub fn generate_mipmaps(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        mip_levels: u32,
    ) {
        let pipeline = self.select_pipeline(format);

        let mut mip_width = width;
        let mut mip_height = height;

        for target_mip in 1..mip_levels {
            let source_mip = target_mip - 1;

            // Create a view of the source mip level.
            let source_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("mip_source_{}", source_mip)),
                format: Some(format),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: source_mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
            });

            // Create a view of the target mip level.
            let target_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("mip_target_{}", target_mip)),
                format: Some(format),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: target_mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
            });

            // Create bind group for this blit pass.
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("mip_blit_bg_{}", target_mip)),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                    },
                ],
            });

            // Calculate the target mip dimensions.
            mip_width = (mip_width / 2).max(1);
            mip_height = (mip_height / 2).max(1);

            // Render pass: blit from source to target.
            {
                let mut render_pass =
                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some(&format!("mip_blit_pass_{}", target_mip)),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &target_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });

                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &bind_group, &[]);
                render_pass.draw(0..3, 0..1);
            }
        }
    }
}

// ============================================================================
// Texture Manager
// ============================================================================

/// Manages GPU textures for the engine.
///
/// Handles creation, mipmap generation, sampler creation, and provides
/// bind-group-ready texture views.
pub struct TextureManager {
    /// Mipmap generation pipeline.
    mipmap_gen: MipmapGenerator,
    /// Stored textures.
    textures: HashMap<GpuTextureHandle, GpuTexture>,
    /// Next handle ID.
    next_id: u64,
    /// Default white 1x1 texture (placeholder).
    pub default_white: GpuTextureHandle,
    /// Default flat normal map (0.5, 0.5, 1.0, 1.0) placeholder.
    pub default_normal: GpuTextureHandle,
    /// Default black 1x1 texture placeholder.
    pub default_black: GpuTextureHandle,
    /// Default metallic-roughness (0, 0.5, 0, 1) placeholder.
    pub default_metallic_roughness: GpuTextureHandle,
}

impl TextureManager {
    /// Create a new texture manager with default placeholder textures.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let mipmap_gen = MipmapGenerator::new(device);

        let mut mgr = Self {
            mipmap_gen,
            textures: HashMap::new(),
            next_id: 1,
            default_white: GpuTextureHandle(0),
            default_normal: GpuTextureHandle(0),
            default_black: GpuTextureHandle(0),
            default_metallic_roughness: GpuTextureHandle(0),
        };

        // Create default 1x1 white texture.
        let white_handle = mgr.create_texture_from_rgba(
            device,
            queue,
            1,
            1,
            &[255, 255, 255, 255],
            false,
            "default_white",
        );
        mgr.default_white = white_handle;

        // Create default 1x1 flat normal map (tangent space: 0,0,1).
        let normal_handle = mgr.create_texture_from_rgba(
            device,
            queue,
            1,
            1,
            &[128, 128, 255, 255],
            false,
            "default_normal",
        );
        mgr.default_normal = normal_handle;

        // Create default 1x1 black texture.
        let black_handle = mgr.create_texture_from_rgba(
            device,
            queue,
            1,
            1,
            &[0, 0, 0, 255],
            false,
            "default_black",
        );
        mgr.default_black = black_handle;

        // Create default metallic-roughness: G=0.5 roughness, B=0.0 metallic.
        let mr_handle = mgr.create_texture_from_rgba(
            device,
            queue,
            1,
            1,
            &[0, 128, 0, 255],
            false,
            "default_metallic_roughness",
        );
        mgr.default_metallic_roughness = mr_handle;

        mgr
    }

    /// Create a GPU texture from RGBA8 pixel data.
    ///
    /// The `pixels` slice must contain exactly `width * height * 4` bytes
    /// (RGBA8 format). If `generate_mipmaps` is true, mipmaps are generated
    /// on the GPU using the blit pipeline.
    pub fn create_texture_from_rgba(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        pixels: &[u8],
        generate_mipmaps: bool,
        label: &str,
    ) -> GpuTextureHandle {
        assert_eq!(
            pixels.len(),
            (width * height * 4) as usize,
            "Pixel data size mismatch: expected {} bytes, got {}",
            width * height * 4,
            pixels.len()
        );

        let mip_levels = if generate_mipmaps {
            GpuTexture::compute_mip_levels(width, height)
        } else {
            1
        };

        let format = wgpu::TextureFormat::Rgba8UnormSrgb;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        // Upload base mip level (level 0).
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        // Generate mipmaps on the GPU.
        if generate_mipmaps && mip_levels > 1 {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("{}_mipmap_encoder", label)),
                });

            self.mipmap_gen.generate_mipmaps(
                device,
                &mut encoder,
                &texture,
                format,
                width,
                height,
                mip_levels,
            );

            queue.submit(std::iter::once(encoder.finish()));
        }

        // Create the full texture view (all mip levels).
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("{}_view", label)),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None, // All mip levels.
            base_array_layer: 0,
            array_layer_count: None,
        });

        let handle = GpuTextureHandle(self.next_id);
        self.next_id += 1;

        self.textures.insert(
            handle,
            GpuTexture {
                texture,
                view,
                width,
                height,
                mip_levels,
                format,
            },
        );

        handle
    }

    /// Create a GPU texture from RGBA8 linear (non-sRGB) pixel data.
    ///
    /// Used for normal maps, metallic-roughness, and other data textures
    /// that should NOT be gamma-corrected.
    pub fn create_texture_from_rgba_linear(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        pixels: &[u8],
        generate_mipmaps: bool,
        label: &str,
    ) -> GpuTextureHandle {
        assert_eq!(
            pixels.len(),
            (width * height * 4) as usize,
            "Pixel data size mismatch"
        );

        let mip_levels = if generate_mipmaps {
            GpuTexture::compute_mip_levels(width, height)
        } else {
            1
        };

        let format = wgpu::TextureFormat::Rgba8Unorm; // Linear, not sRGB!

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        if generate_mipmaps && mip_levels > 1 {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("{}_mipmap_encoder", label)),
                });

            self.mipmap_gen.generate_mipmaps(
                device,
                &mut encoder,
                &texture,
                format,
                width,
                height,
                mip_levels,
            );

            queue.submit(std::iter::once(encoder.finish()));
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("{}_view", label)),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

        let handle = GpuTextureHandle(self.next_id);
        self.next_id += 1;

        self.textures.insert(
            handle,
            GpuTexture {
                texture,
                view,
                width,
                height,
                mip_levels,
                format,
            },
        );

        handle
    }

    /// Create a GPU texture from RGBA16Float (HDR) pixel data.
    pub fn create_texture_from_rgba16f(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        pixels: &[u8],
        generate_mipmaps: bool,
        label: &str,
    ) -> GpuTextureHandle {
        // Each pixel is 8 bytes (4 x f16).
        assert_eq!(
            pixels.len(),
            (width * height * 8) as usize,
            "HDR pixel data size mismatch"
        );

        let mip_levels = if generate_mipmaps {
            GpuTexture::compute_mip_levels(width, height)
        } else {
            1
        };

        let format = wgpu::TextureFormat::Rgba16Float;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        if generate_mipmaps && mip_levels > 1 {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("{}_mipmap_encoder", label)),
                });

            self.mipmap_gen.generate_mipmaps(
                device,
                &mut encoder,
                &texture,
                format,
                width,
                height,
                mip_levels,
            );

            queue.submit(std::iter::once(encoder.finish()));
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("{}_view", label)),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

        let handle = GpuTextureHandle(self.next_id);
        self.next_id += 1;

        self.textures.insert(
            handle,
            GpuTexture {
                texture,
                view,
                width,
                height,
                mip_levels,
                format,
            },
        );

        handle
    }

    /// Look up a texture by handle.
    pub fn get(&self, handle: GpuTextureHandle) -> Option<&GpuTexture> {
        self.textures.get(&handle)
    }

    /// Get the texture view for a handle (convenience).
    pub fn get_view(&self, handle: GpuTextureHandle) -> Option<&wgpu::TextureView> {
        self.textures.get(&handle).map(|t| &t.view)
    }

    /// Remove a texture from the manager, releasing GPU memory.
    pub fn remove(&mut self, handle: GpuTextureHandle) -> bool {
        self.textures.remove(&handle).is_some()
    }

    /// Total number of managed textures.
    pub fn texture_count(&self) -> usize {
        self.textures.len()
    }

    /// Estimate GPU memory usage for all managed textures.
    pub fn estimate_gpu_memory_bytes(&self) -> u64 {
        let mut total = 0u64;
        for tex in self.textures.values() {
            let bpp: u64 = match tex.format {
                wgpu::TextureFormat::Rgba16Float => 8,
                _ => 4,
            };
            let mut w = tex.width as u64;
            let mut h = tex.height as u64;
            for _ in 0..tex.mip_levels {
                total += w * h * bpp;
                w = (w / 2).max(1);
                h = (h / 2).max(1);
            }
        }
        total
    }
}

// ============================================================================
// Sampler creation helpers
// ============================================================================

/// Create a texture sampler with the given filter and address mode.
pub fn create_sampler(
    device: &wgpu::Device,
    filter: TextureFilterMode,
    address_mode: TextureAddressMode,
    label: &str,
) -> wgpu::Sampler {
    let (mag, min, mip) = match filter {
        TextureFilterMode::Nearest => (
            wgpu::FilterMode::Nearest,
            wgpu::FilterMode::Nearest,
            wgpu::FilterMode::Nearest,
        ),
        TextureFilterMode::Linear => (
            wgpu::FilterMode::Linear,
            wgpu::FilterMode::Linear,
            wgpu::FilterMode::Nearest,
        ),
        TextureFilterMode::Trilinear => (
            wgpu::FilterMode::Linear,
            wgpu::FilterMode::Linear,
            wgpu::FilterMode::Linear,
        ),
    };

    let addr = address_mode.to_wgpu();

    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: addr,
        address_mode_v: addr,
        address_mode_w: addr,
        mag_filter: mag,
        min_filter: min,
        mipmap_filter: mip,
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        compare: None,
        anisotropy_clamp: if matches!(filter, TextureFilterMode::Trilinear) {
            DEFAULT_ANISOTROPY
        } else {
            1
        },
        border_color: Some(wgpu::SamplerBorderColor::OpaqueBlack),
    })
}

/// Create a standard trilinear repeating sampler (most common for PBR textures).
pub fn create_default_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    create_sampler(
        device,
        TextureFilterMode::Trilinear,
        TextureAddressMode::Repeat,
        "default_trilinear_sampler",
    )
}

/// Create a sampler for normal maps (linear, repeat, no anisotropy).
pub fn create_normal_map_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    create_sampler(
        device,
        TextureFilterMode::Linear,
        TextureAddressMode::Repeat,
        "normal_map_sampler",
    )
}

/// Create a sampler suitable for shadow map depth comparison.
pub fn create_shadow_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("shadow_comparison_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    })
}

// ============================================================================
// Textured Material Bind Group Layout (group 2)
// ============================================================================

/// Create the bind group layout for a fully-textured PBR material.
///
/// This replaces the simple MaterialUniform-only layout in group 2 with one
/// that includes texture + sampler pairs for all PBR channels.
pub fn create_textured_material_bind_group_layout(
    device: &wgpu::Device,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("textured_material_bgl"),
        entries: &[
            // binding 0: MaterialUniform
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
            // binding 1: Albedo texture
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
            // binding 2: Albedo sampler
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // binding 3: Normal map
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // binding 4: Normal sampler
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // binding 5: Metallic-roughness map
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
            // binding 6: Metallic-roughness sampler
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // binding 7: Emissive map
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
            // binding 8: Emissive sampler
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // binding 9: AO map
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
            // binding 10: AO sampler
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

// ============================================================================
// Textured material bind group creation
// ============================================================================

/// Describes which textures to bind for a PBR material.
pub struct TexturedMaterialBindings<'a> {
    /// The material uniform buffer.
    pub material_uniform_buffer: &'a wgpu::Buffer,
    /// Albedo texture view.
    pub albedo_view: &'a wgpu::TextureView,
    /// Albedo sampler.
    pub albedo_sampler: &'a wgpu::Sampler,
    /// Normal map view.
    pub normal_view: &'a wgpu::TextureView,
    /// Normal sampler.
    pub normal_sampler: &'a wgpu::Sampler,
    /// Metallic-roughness map view.
    pub metallic_roughness_view: &'a wgpu::TextureView,
    /// Metallic-roughness sampler.
    pub metallic_roughness_sampler: &'a wgpu::Sampler,
    /// Emissive map view.
    pub emissive_view: &'a wgpu::TextureView,
    /// Emissive sampler.
    pub emissive_sampler: &'a wgpu::Sampler,
    /// Ambient occlusion map view.
    pub ao_view: &'a wgpu::TextureView,
    /// AO sampler.
    pub ao_sampler: &'a wgpu::Sampler,
}

/// Create a bind group for a fully-textured PBR material.
pub fn create_textured_material_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    bindings: &TexturedMaterialBindings,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: bindings.material_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(bindings.albedo_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(bindings.albedo_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(bindings.normal_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(bindings.normal_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(bindings.metallic_roughness_view),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::Sampler(bindings.metallic_roughness_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(bindings.emissive_view),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::Sampler(bindings.emissive_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::TextureView(bindings.ao_view),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: wgpu::BindingResource::Sampler(bindings.ao_sampler),
            },
        ],
    })
}

// ============================================================================
// Textured PBR shader
// ============================================================================

/// The complete textured PBR shader with all texture channels.
///
/// This extends the base PBR shader with:
/// - Albedo texture sampling (modulated by material uniform colour)
/// - Normal mapping with TBN matrix
/// - Metallic-roughness map (G=roughness, B=metallic)
/// - Emissive map
/// - Ambient occlusion map
pub const TEXTURED_PBR_SHADER_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Textured PBR Shader
// ============================================================================
//
// Extends the base PBR shader with full texture support:
//   - Albedo texture (sRGB, with mipmap sampling)
//   - Normal map (tangent space -> world space via TBN)
//   - Metallic-roughness map (G=roughness, B=metallic)
//   - Emissive map
//   - Ambient occlusion map
//
// Bind groups:
//   Group 0: Camera + Lights (shared with base PBR)
//   Group 1: Model (per object)
//   Group 2: Material + Textures (extended)

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

// Group 0: Camera + Lights (per frame).
@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> lights: LightsUniform;

// Group 1: Model (per object).
@group(1) @binding(0) var<uniform> model: ModelUniform;

// Group 2: Material + Textures.
@group(2) @binding(0) var<uniform> material: MaterialUniform;
@group(2) @binding(1) var albedo_texture: texture_2d<f32>;
@group(2) @binding(2) var albedo_sampler_: sampler;
@group(2) @binding(3) var normal_texture: texture_2d<f32>;
@group(2) @binding(4) var normal_sampler_: sampler;
@group(2) @binding(5) var metallic_roughness_texture: texture_2d<f32>;
@group(2) @binding(6) var metallic_roughness_sampler_: sampler;
@group(2) @binding(7) var emissive_texture: texture_2d<f32>;
@group(2) @binding(8) var emissive_sampler_: sampler;
@group(2) @binding(9) var ao_texture: texture_2d<f32>;
@group(2) @binding(10) var ao_sampler_: sampler;

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
    @location(5) world_tangent: vec3<f32>,
    @location(6) world_bitangent: vec3<f32>,
};

// ---------------------------------------------------------------------------
// Vertex shader
// ---------------------------------------------------------------------------

@vertex
fn vs_textured(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // Transform position to world space.
    let world_pos = model.world * vec4<f32>(input.position, 1.0);
    output.world_position = world_pos.xyz;

    // Transform position to clip space.
    output.clip_position = camera.view_projection * world_pos;

    // Transform normal to world space.
    let raw_normal = (model.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz;
    output.world_normal = normalize(raw_normal);

    // Construct tangent and bitangent from normal.
    // We compute an approximate tangent from the UV derivatives and the
    // world normal. For meshes without explicit tangent data, we use the
    // standard technique of choosing an arbitrary perpendicular vector.
    let n = output.world_normal;
    var tangent: vec3<f32>;
    if abs(n.y) < 0.999 {
        tangent = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), n));
    } else {
        tangent = normalize(cross(vec3<f32>(1.0, 0.0, 0.0), n));
    }
    let bitangent = normalize(cross(n, tangent));

    output.world_tangent = tangent;
    output.world_bitangent = bitangent;

    output.uv = input.uv;
    output.vertex_color = input.color;
    output.view_dir = normalize(camera.camera_position.xyz - world_pos.xyz);

    return output;
}

// ---------------------------------------------------------------------------
// TBN matrix construction
// ---------------------------------------------------------------------------

// Construct the tangent-bitangent-normal matrix and transform a normal map
// sample from tangent space to world space.
fn apply_normal_map(
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
    normal: vec3<f32>,
    normal_map_sample: vec3<f32>,
) -> vec3<f32> {
    // Convert from [0,1] to [-1,1].
    let tangent_normal = normal_map_sample * 2.0 - vec3<f32>(1.0);

    // Construct TBN matrix (columns = T, B, N).
    let tbn = mat3x3<f32>(
        normalize(tangent),
        normalize(bitangent),
        normalize(normal),
    );

    // Transform the tangent-space normal to world space.
    return normalize(tbn * tangent_normal);
}

// ---------------------------------------------------------------------------
// Lighting utility functions (same as base PBR)
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
    let ggx_v = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx_l = geometry_schlick_ggx(n_dot_l, roughness);
    return ggx_v * ggx_l;
}

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
    if c <= 0.0031308 {
        return c * 12.92;
    }
    return 1.055 * pow(c, INV_GAMMA) - 0.055;
}

fn tone_map_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 1.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + vec3<f32>(b))) / (x * (c * x + vec3<f32>(d)) + vec3<f32>(e)),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

// ---------------------------------------------------------------------------
// Light contribution
// ---------------------------------------------------------------------------

struct LightContribution {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
};

fn compute_directional_light_textured(
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

    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(v_dot_h, f0);

    let spec_brdf = (d * g * f) / (4.0 * n_dot_v * n_dot_l + EPSILON);

    let k_s = f;
    let k_d = (vec3<f32>(1.0) - k_s) * (1.0 - metallic);
    let diffuse_brdf = k_d * albedo * INV_PI;

    result.diffuse = diffuse_brdf * light_color * n_dot_l;
    result.specular = spec_brdf * light_color * n_dot_l;

    return result;
}

fn compute_point_light_textured(
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

    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(v_dot_h, f0);

    let spec_brdf = (d * g * f) / (4.0 * n_dot_v * n_dot_l + EPSILON);

    let k_s = f;
    let k_d = (vec3<f32>(1.0) - k_s) * (1.0 - metallic);
    let diffuse_brdf = k_d * albedo * INV_PI;

    let attenuated_light = light_color * att;

    result.diffuse = diffuse_brdf * attenuated_light * n_dot_l;
    result.specular = spec_brdf * attenuated_light * n_dot_l;

    return result;
}

// ---------------------------------------------------------------------------
// Fragment shader -- Textured PBR
// ---------------------------------------------------------------------------

@fragment
fn fs_textured(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample all textures.
    let albedo_sample = textureSample(albedo_texture, albedo_sampler_, input.uv);
    let normal_sample = textureSample(normal_texture, normal_sampler_, input.uv);
    let mr_sample = textureSample(metallic_roughness_texture, metallic_roughness_sampler_, input.uv);
    let emissive_sample = textureSample(emissive_texture, emissive_sampler_, input.uv);
    let ao_sample = textureSample(ao_texture, ao_sampler_, input.uv);

    // Compute albedo: texture * material colour * vertex colour.
    let has_albedo = material.flags.x;
    var albedo: vec3<f32>;
    if has_albedo > 0.5 {
        albedo = albedo_sample.xyz * material.albedo_color.xyz * input.vertex_color.xyz;
    } else {
        albedo = material.albedo_color.xyz * input.vertex_color.xyz;
    }
    let alpha = albedo_sample.w * material.albedo_color.w * input.vertex_color.w;

    // Alpha test.
    let alpha_cutoff = material.metallic_roughness.w;
    if alpha_cutoff > 0.0 && alpha < alpha_cutoff {
        discard;
    }

    // Metallic-roughness from texture (if flag set).
    let has_mr = material.flags.z;
    var metallic: f32;
    var roughness: f32;
    if has_mr > 0.5 {
        // glTF convention: G=roughness, B=metallic.
        roughness = clamp(mr_sample.y * material.metallic_roughness.y, 0.04, 1.0);
        metallic = clamp(mr_sample.z * material.metallic_roughness.x, 0.0, 1.0);
    } else {
        metallic = clamp(material.metallic_roughness.x, 0.0, 1.0);
        roughness = clamp(material.metallic_roughness.y, 0.04, 1.0);
    }
    let reflectance = material.metallic_roughness.z;

    // Compute F0.
    let f0 = compute_f0(albedo, metallic, reflectance);

    // Normal mapping.
    let has_normal = material.flags.y;
    var normal: vec3<f32>;
    if has_normal > 0.5 {
        normal = apply_normal_map(
            input.world_tangent,
            input.world_bitangent,
            input.world_normal,
            normal_sample.xyz,
        );
    } else {
        normal = normalize(input.world_normal);
    }
    let view_dir = normalize(input.view_dir);

    // Accumulate lighting.
    var total_diffuse = vec3<f32>(0.0);
    var total_specular = vec3<f32>(0.0);

    let num_lights = u32(lights.light_count.x);

    for (var i = 0u; i < MAX_LIGHTS; i = i + 1u) {
        if i >= num_lights {
            break;
        }

        let light = lights.lights[i];
        let light_type = light.params.x;

        if abs(light_type - LIGHT_TYPE_DIRECTIONAL) < 0.5 {
            let contrib = compute_directional_light_textured(
                light, normal, view_dir, albedo, metallic, roughness, f0
            );
            total_diffuse = total_diffuse + contrib.diffuse;
            total_specular = total_specular + contrib.specular;
        } else if abs(light_type - LIGHT_TYPE_POINT) < 0.5 {
            let contrib = compute_point_light_textured(
                light, input.world_position, normal, view_dir,
                albedo, metallic, roughness, f0
            );
            total_diffuse = total_diffuse + contrib.diffuse;
            total_specular = total_specular + contrib.specular;
        }
    }

    // Ambient lighting with AO.
    let ambient = hemisphere_ambient(normal, lights.ambient.xyz, lights.ambient.w);
    let n_dot_v_ambient = max(dot(normal, view_dir), 0.0);
    let f_ambient = fresnel_schlick_roughness(n_dot_v_ambient, f0, roughness);
    let k_d_ambient = (vec3<f32>(1.0) - f_ambient) * (1.0 - metallic);
    let ao = ao_sample.x; // Red channel of AO map.
    let ambient_diffuse = k_d_ambient * albedo * ambient * ao;
    let ambient_specular = f_ambient * ambient * 0.2 * ao;

    // Emissive.
    var emissive: vec3<f32>;
    let has_emissive = material.flags.w;
    if has_emissive > 0.5 {
        emissive = emissive_sample.xyz * material.emissive.xyz * material.emissive.w;
    } else {
        emissive = material.emissive.xyz * material.emissive.w;
    }

    // Combine.
    var final_color = total_diffuse + total_specular + ambient_diffuse + ambient_specular + emissive;

    // Tone mapping.
    final_color = tone_map_aces(final_color);

    // Gamma correction.
    final_color = vec3<f32>(
        linear_to_srgb(final_color.x),
        linear_to_srgb(final_color.y),
        linear_to_srgb(final_color.z)
    );

    return vec4<f32>(final_color, alpha);
}
"#;

// ============================================================================
// Textured PBR pipeline creation
// ============================================================================

/// Create the textured PBR render pipeline.
///
/// This pipeline uses the same vertex layout as the base PBR pipeline but
/// binds group 2 to the textured material layout.
pub fn create_textured_pbr_pipeline(
    device: &wgpu::Device,
    camera_lights_bgl: &wgpu::BindGroupLayout,
    model_bgl: &wgpu::BindGroupLayout,
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let textured_material_bgl = create_textured_material_bind_group_layout(device);

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("textured_pbr_shader"),
        source: wgpu::ShaderSource::Wgsl(TEXTURED_PBR_SHADER_WGSL.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("textured_pbr_pipeline_layout"),
        bind_group_layouts: &[camera_lights_bgl, model_bgl, &textured_material_bgl],
        push_constant_ranges: &[],
    });

    let vertex_layout = super::scene_renderer::SceneVertex::buffer_layout();

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("textured_pbr_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader_module,
            entry_point: Some("vs_textured"),
            buffers: &[vertex_layout],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_module,
            entry_point: Some("fs_textured"),
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

    (pipeline, textured_material_bgl)
}

// ============================================================================
// Texture checkerboard generator (utility for testing)
// ============================================================================

/// Generate a checkerboard pattern texture for testing.
///
/// Returns RGBA8 pixel data.
pub fn generate_checkerboard_texture(
    width: u32,
    height: u32,
    check_size: u32,
    color_a: [u8; 4],
    color_b: [u8; 4],
) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);

    for y in 0..height {
        for x in 0..width {
            let check_x = (x / check_size) % 2;
            let check_y = (y / check_size) % 2;
            let is_a = (check_x + check_y) % 2 == 0;
            let color = if is_a { color_a } else { color_b };
            pixels.extend_from_slice(&color);
        }
    }

    pixels
}

/// Generate a gradient texture for testing.
pub fn generate_gradient_texture(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);

    for y in 0..height {
        for x in 0..width {
            let u = x as f32 / (width - 1).max(1) as f32;
            let v = y as f32 / (height - 1).max(1) as f32;
            let r = (u * 255.0) as u8;
            let g = (v * 255.0) as u8;
            let b = ((1.0 - u) * 255.0) as u8;
            pixels.extend_from_slice(&[r, g, b, 255]);
        }
    }

    pixels
}

/// Generate a default normal map (flat, pointing up in tangent space).
pub fn generate_flat_normal_map(width: u32, height: u32) -> Vec<u8> {
    let pixel = [128u8, 128, 255, 255]; // Tangent-space Z-up
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for _ in 0..(width * height) {
        pixels.extend_from_slice(&pixel);
    }
    pixels
}

/// Generate a brick-pattern normal map for testing.
pub fn generate_brick_normal_map(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    let brick_w = width / 4;
    let brick_h = height / 8;

    for y in 0..height {
        for x in 0..width {
            let row = y / brick_h;
            let offset = if row % 2 == 0 { 0 } else { brick_w / 2 };
            let bx = (x + offset) % brick_w;
            let by = y % brick_h;

            // Check if we're in the mortar (border).
            let border = 2u32;
            let in_mortar = bx < border || bx >= brick_w - border
                || by < border || by >= brick_h - border;

            if in_mortar {
                // Mortar: slightly recessed. Normal = (0, 0, 1) = flat.
                pixels.extend_from_slice(&[128, 128, 255, 255]);
            } else {
                // Brick surface: slight variation.
                let nx = if bx < border + 2 {
                    100u8 // Left edge: normal points left.
                } else if bx >= brick_w - border - 2 {
                    156u8 // Right edge: normal points right.
                } else {
                    128u8
                };
                let ny = if by < border + 2 {
                    100u8 // Top edge.
                } else if by >= brick_h - border - 2 {
                    156u8 // Bottom edge.
                } else {
                    128u8
                };
                pixels.extend_from_slice(&[nx, ny, 255, 255]);
            }
        }
    }

    pixels
}

// ============================================================================
// Textured material GPU data
// ============================================================================

/// A fully textured PBR material uploaded to the GPU.
pub struct TexturedMaterialGpuData {
    /// Material uniform buffer.
    pub uniform_buffer: wgpu::Buffer,
    /// The bind group with all texture bindings.
    pub bind_group: wgpu::BindGroup,
    /// Material parameters.
    pub params: super::scene_renderer::MaterialParams,
    /// Texture handles for tracking.
    pub albedo_handle: GpuTextureHandle,
    pub normal_handle: GpuTextureHandle,
    pub metallic_roughness_handle: GpuTextureHandle,
    pub emissive_handle: GpuTextureHandle,
    pub ao_handle: GpuTextureHandle,
}

impl TexturedMaterialGpuData {
    /// Create a textured material with explicit texture handles.
    pub fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        params: &super::scene_renderer::MaterialParams,
        texture_mgr: &TextureManager,
        albedo_handle: GpuTextureHandle,
        normal_handle: GpuTextureHandle,
        metallic_roughness_handle: GpuTextureHandle,
        emissive_handle: GpuTextureHandle,
        ao_handle: GpuTextureHandle,
        albedo_sampler: &wgpu::Sampler,
        normal_sampler: &wgpu::Sampler,
        mr_sampler: &wgpu::Sampler,
        emissive_sampler: &wgpu::Sampler,
        ao_sampler: &wgpu::Sampler,
        label: &str,
    ) -> Self {
        use wgpu::util::DeviceExt;

        let mut uniform = params.to_uniform();
        // Set flags for which textures are active.
        uniform.flags[0] = if albedo_handle != texture_mgr.default_white {
            1.0
        } else {
            0.0
        };
        uniform.flags[1] = if normal_handle != texture_mgr.default_normal {
            1.0
        } else {
            0.0
        };
        uniform.flags[2] = if metallic_roughness_handle != texture_mgr.default_metallic_roughness {
            1.0
        } else {
            0.0
        };
        uniform.flags[3] = if emissive_handle != texture_mgr.default_black {
            1.0
        } else {
            0.0
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{}_material_ub", label)),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let albedo_view = texture_mgr.get_view(albedo_handle)
            .expect("Albedo texture not found");
        let normal_view = texture_mgr.get_view(normal_handle)
            .expect("Normal texture not found");
        let mr_view = texture_mgr.get_view(metallic_roughness_handle)
            .expect("Metallic-roughness texture not found");
        let emissive_view = texture_mgr.get_view(emissive_handle)
            .expect("Emissive texture not found");
        let ao_view = texture_mgr.get_view(ao_handle)
            .expect("AO texture not found");

        let bindings = TexturedMaterialBindings {
            material_uniform_buffer: &uniform_buffer,
            albedo_view,
            albedo_sampler,
            normal_view,
            normal_sampler,
            metallic_roughness_view: mr_view,
            metallic_roughness_sampler: mr_sampler,
            emissive_view,
            emissive_sampler,
            ao_view,
            ao_sampler,
        };

        let bind_group = create_textured_material_bind_group(device, layout, &bindings, label);

        Self {
            uniform_buffer,
            bind_group,
            params: params.clone(),
            albedo_handle,
            normal_handle,
            metallic_roughness_handle,
            emissive_handle,
            ao_handle,
        }
    }

    /// Update material parameters (uniform only, textures unchanged).
    pub fn update_params(
        &mut self,
        queue: &wgpu::Queue,
        params: &super::scene_renderer::MaterialParams,
    ) {
        self.params = params.clone();
        let uniform = params.to_uniform();
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }
}

// ============================================================================
// Texture array / atlas support
// ============================================================================

/// A texture atlas that packs multiple sub-textures into one GPU texture.
pub struct TextureAtlas {
    /// The GPU texture handle.
    pub texture_handle: GpuTextureHandle,
    /// UV regions for each sub-texture (u_min, v_min, u_max, v_max).
    pub regions: Vec<[f32; 4]>,
    /// Atlas dimensions.
    pub width: u32,
    pub height: u32,
}

impl TextureAtlas {
    /// Create a simple grid-based atlas.
    ///
    /// `tiles_x` and `tiles_y` define the grid layout.
    pub fn create_grid_atlas(
        texture_mgr: &mut TextureManager,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        atlas_pixels: &[u8],
        atlas_width: u32,
        atlas_height: u32,
        tiles_x: u32,
        tiles_y: u32,
        label: &str,
    ) -> Self {
        let handle = texture_mgr.create_texture_from_rgba(
            device,
            queue,
            atlas_width,
            atlas_height,
            atlas_pixels,
            true,
            label,
        );

        let tile_w = 1.0 / tiles_x as f32;
        let tile_h = 1.0 / tiles_y as f32;

        let mut regions = Vec::with_capacity((tiles_x * tiles_y) as usize);
        for y in 0..tiles_y {
            for x in 0..tiles_x {
                regions.push([
                    x as f32 * tile_w,
                    y as f32 * tile_h,
                    (x + 1) as f32 * tile_w,
                    (y + 1) as f32 * tile_h,
                ]);
            }
        }

        Self {
            texture_handle: handle,
            regions,
            width: atlas_width,
            height: atlas_height,
        }
    }

    /// Get the UV region for a specific tile index.
    pub fn get_region(&self, index: usize) -> Option<[f32; 4]> {
        self.regions.get(index).copied()
    }

    /// Total number of tiles.
    pub fn tile_count(&self) -> usize {
        self.regions.len()
    }
}

// ============================================================================
// Procedural texture generation utilities
// ============================================================================

/// Generate a Perlin-noise-like texture for roughness variation.
pub fn generate_noise_texture(width: u32, height: u32, scale: f32, seed: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);

    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 * scale / width as f32;
            let fy = y as f32 * scale / height as f32;

            // Simple hash-based noise.
            let hash = pseudo_hash(
                (x.wrapping_mul(73856093)) ^ (y.wrapping_mul(19349663)) ^ seed,
            );
            let fine = hash as f32 / u32::MAX as f32;

            // Larger-scale blobs.
            let ix = (fx.floor()) as u32;
            let iy = (fy.floor()) as u32;
            let blob_hash = pseudo_hash(
                (ix.wrapping_mul(73856093)) ^ (iy.wrapping_mul(19349663)) ^ seed,
            );
            let coarse = blob_hash as f32 / u32::MAX as f32;

            // Blend fine and coarse.
            let value = (fine * 0.3 + coarse * 0.7).clamp(0.0, 1.0);
            let byte = (value * 255.0) as u8;

            pixels.extend_from_slice(&[byte, byte, byte, 255]);
        }
    }

    pixels
}

/// Simple 32-bit hash function for procedural generation.
fn pseudo_hash(mut v: u32) -> u32 {
    v = v.wrapping_mul(0x45d9f3b);
    v = (v >> 16) ^ v;
    v = v.wrapping_mul(0x45d9f3b);
    v = (v >> 16) ^ v;
    v
}

/// Generate a circular gradient for spotlight cookie textures.
pub fn generate_spotlight_cookie(width: u32, height: u32, softness: f32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    let cx = width as f32 * 0.5;
    let cy = height as f32 * 0.5;
    let max_radius = cx.min(cy);

    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let normalized = dist / max_radius;

            // Smooth falloff.
            let edge_start = 1.0 - softness;
            let value = if normalized < edge_start {
                1.0
            } else if normalized < 1.0 {
                let t = (normalized - edge_start) / softness;
                1.0 - t * t * (3.0 - 2.0 * t) // Smoothstep falloff.
            } else {
                0.0
            };

            let byte = (value * 255.0) as u8;
            pixels.extend_from_slice(&[byte, byte, byte, 255]);
        }
    }

    pixels
}
