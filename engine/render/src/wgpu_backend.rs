// engine/render/src/wgpu_backend.rs
//
// wgpu-based GPU backend implementation. Uses wgpu's cross-platform abstraction
// to target Vulkan, DX12, Metal, and WebGPU from a single code path.
//
// This backend implements the engine's `RenderDevice` trait, provides surface
// management for presentation, and stores GPU resources in generational handle
// pools for safe, validated access.

use std::sync::Arc;

use parking_lot::Mutex;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use wgpu;

use crate::interface::command_buffer::CommandBuffer;
use crate::interface::device::{DeviceCapabilities, QueueType, RenderDevice};
use crate::interface::pipeline::{
    BlendFactor, BlendOp, CompareOp, ComputePipelineDesc, CullMode, FrontFace,
    GraphicsPipelineDesc, PolygonMode, PrimitiveTopology, VertexFormat, VertexStepMode,
};
use crate::interface::resource::{
    AddressMode, BufferDesc, BufferHandle, BufferUsage, FilterMode, FramebufferDesc,
    FramebufferHandle, FenceHandle, MemoryLocation, PipelineHandle, RenderPassDesc,
    RenderPassHandle, SamplerDesc, SamplerHandle, ShaderDesc, ShaderHandle, ShaderStage,
    TextureDesc, TextureDimension, TextureFormat, TextureHandle, TextureUsage,
};
use crate::RenderError;
use genovo_core::HandlePool;

// ---------------------------------------------------------------------------
// Internal resource wrapper types
// ---------------------------------------------------------------------------

/// Wrapper around a wgpu buffer with its descriptor metadata.
pub(crate) struct WgpuBuffer {
    pub buffer: wgpu::Buffer,
    pub size: u64,
    pub usage: BufferUsage,
    pub memory: MemoryLocation,
}

/// Wrapper around a wgpu texture with its descriptor metadata.
pub(crate) struct WgpuTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub format: TextureFormat,
    pub width: u32,
    pub height: u32,
    pub depth_or_array_layers: u32,
    pub mip_levels: u32,
}

/// Wrapper around a wgpu sampler.
pub(crate) struct WgpuSampler {
    pub sampler: wgpu::Sampler,
}

/// Wrapper around a compiled wgpu shader module.
pub(crate) struct WgpuShaderModule {
    pub module: wgpu::ShaderModule,
    pub stage: ShaderStage,
    pub entry_point: String,
}

/// Wrapper around a wgpu render pipeline.
pub(crate) struct WgpuPipeline {
    pub pipeline: WgpuPipelineKind,
}

pub(crate) enum WgpuPipelineKind {
    Render(wgpu::RenderPipeline),
    Compute(wgpu::ComputePipeline),
}

/// Wrapper around a render pass descriptor (stored for framebuffer compat).
pub(crate) struct WgpuRenderPass {
    pub color_format: Option<TextureFormat>,
    pub depth_format: Option<TextureFormat>,
    pub sample_count: u32,
}

/// Wrapper around a framebuffer (attachment references + dimensions).
pub(crate) struct WgpuFramebuffer {
    pub color_attachments: Vec<TextureHandle>,
    pub depth_attachment: Option<TextureHandle>,
    pub width: u32,
    pub height: u32,
}

// ---------------------------------------------------------------------------
// Fence tracking
// ---------------------------------------------------------------------------

/// Simple fence: a monotonically increasing submission counter.
struct FenceTracker {
    next_id: u64,
}

impl FenceTracker {
    fn new() -> Self {
        Self { next_id: 1 }
    }

    fn next(&mut self) -> FenceHandle {
        let id = self.next_id;
        self.next_id += 1;
        FenceHandle(id)
    }
}

// ---------------------------------------------------------------------------
// WgpuDevice
// ---------------------------------------------------------------------------

/// The wgpu implementation of the engine's `RenderDevice` trait.
///
/// This struct owns the wgpu instance, adapter, device, and queue, along with
/// handle pools for every GPU resource type. It is `Send + Sync` so it can be
/// shared across threads behind an `Arc`.
pub struct WgpuDevice {
    /// wgpu instance (entry point for adapter enumeration).
    instance: wgpu::Instance,
    /// The selected physical adapter.
    adapter: wgpu::Adapter,
    /// The logical GPU device.
    device: Arc<wgpu::Device>,
    /// The primary command queue.
    queue: Arc<wgpu::Queue>,

    // -- Resource pools (all behind Mutex for interior mutability) --
    buffers: Mutex<HandlePool<WgpuBuffer>>,
    textures: Mutex<HandlePool<WgpuTexture>>,
    samplers: Mutex<HandlePool<WgpuSampler>>,
    pipelines: Mutex<HandlePool<WgpuPipeline>>,
    shader_modules: Mutex<HandlePool<WgpuShaderModule>>,
    render_passes: Mutex<HandlePool<WgpuRenderPass>>,
    framebuffers: Mutex<HandlePool<WgpuFramebuffer>>,
    fence_tracker: Mutex<FenceTracker>,

    /// Cached device capabilities / limits.
    capabilities: DeviceCapabilities,
}

impl WgpuDevice {
    /// Create a new wgpu device **without** a surface.
    ///
    /// This is useful for headless rendering, testing, or when the surface
    /// will be configured separately via [`WgpuSurface`].
    pub fn new_headless() -> Result<Self, RenderError> {
        pollster::block_on(Self::new_headless_async())
    }

    async fn new_headless_async() -> Result<Self, RenderError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                RenderError::ResourceCreation("Failed to find a suitable GPU adapter".into())
            })?;

        let (device, queue) = Self::request_device(&adapter).await?;
        let capabilities = Self::query_capabilities(&adapter);

        log::info!(
            "WgpuDevice created (headless): {}",
            capabilities.device_name
        );

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            buffers: Mutex::new(HandlePool::new()),
            textures: Mutex::new(HandlePool::new()),
            samplers: Mutex::new(HandlePool::new()),
            pipelines: Mutex::new(HandlePool::new()),
            shader_modules: Mutex::new(HandlePool::new()),
            render_passes: Mutex::new(HandlePool::new()),
            framebuffers: Mutex::new(HandlePool::new()),
            fence_tracker: Mutex::new(FenceTracker::new()),
            capabilities,
        })
    }

    /// Create a new wgpu device from a pre-configured `wgpu::Instance`.
    ///
    /// This is used by backend wrappers (Vulkan, DX12, Metal) to create a
    /// `WgpuDevice` constrained to a specific backend. The instance should
    /// be created with the desired `wgpu::Backends` flags.
    pub fn new_with_instance(instance: wgpu::Instance) -> Result<Self, RenderError> {
        pollster::block_on(Self::new_with_instance_async(instance))
    }

    async fn new_with_instance_async(instance: wgpu::Instance) -> Result<Self, RenderError> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                RenderError::ResourceCreation(
                    "Failed to find a suitable GPU adapter for the requested backend".into(),
                )
            })?;

        let (device, queue) = Self::request_device(&adapter).await?;
        let capabilities = Self::query_capabilities(&adapter);

        log::info!(
            "WgpuDevice created (backend-specific): {}",
            capabilities.device_name
        );

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            buffers: Mutex::new(HandlePool::new()),
            textures: Mutex::new(HandlePool::new()),
            samplers: Mutex::new(HandlePool::new()),
            pipelines: Mutex::new(HandlePool::new()),
            shader_modules: Mutex::new(HandlePool::new()),
            render_passes: Mutex::new(HandlePool::new()),
            framebuffers: Mutex::new(HandlePool::new()),
            fence_tracker: Mutex::new(FenceTracker::new()),
            capabilities,
        })
    }

    /// Create a new wgpu device compatible with the given window surface.
    ///
    /// The surface is created from the provided window handle and returned
    /// alongside the device so that the caller (typically [`WgpuSurface`])
    /// can configure and present to it.
    pub fn new_with_surface<W>(window: Arc<W>) -> Result<(Self, wgpu::Surface<'static>), RenderError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        pollster::block_on(Self::new_with_surface_async(window))
    }

    async fn new_with_surface_async<W>(
        window: Arc<W>,
    ) -> Result<(Self, wgpu::Surface<'static>), RenderError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance
            .create_surface(window)
            .map_err(|e| RenderError::ResourceCreation(format!("Failed to create surface: {e}")))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                RenderError::ResourceCreation(
                    "Failed to find a GPU adapter compatible with the surface".into(),
                )
            })?;

        let (device, queue) = Self::request_device(&adapter).await?;
        let capabilities = Self::query_capabilities(&adapter);

        log::info!("WgpuDevice created: {}", capabilities.device_name);

        let wgpu_device = Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            buffers: Mutex::new(HandlePool::new()),
            textures: Mutex::new(HandlePool::new()),
            samplers: Mutex::new(HandlePool::new()),
            pipelines: Mutex::new(HandlePool::new()),
            shader_modules: Mutex::new(HandlePool::new()),
            render_passes: Mutex::new(HandlePool::new()),
            framebuffers: Mutex::new(HandlePool::new()),
            fence_tracker: Mutex::new(FenceTracker::new()),
            capabilities,
        };

        Ok((wgpu_device, surface))
    }

    /// Request a logical device and queue from the adapter.
    async fn request_device(
        adapter: &wgpu::Adapter,
    ) -> Result<(wgpu::Device, wgpu::Queue), RenderError> {
        let limits = adapter.limits();
        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Genovo WgpuDevice"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| RenderError::ResourceCreation(format!("Failed to create device: {e}")))
    }

    /// Build a `DeviceCapabilities` struct from the adapter's properties.
    fn query_capabilities(adapter: &wgpu::Adapter) -> DeviceCapabilities {
        let info = adapter.get_info();
        let limits = adapter.limits();
        let features = adapter.features();

        DeviceCapabilities {
            device_name: info.name.clone(),
            vendor_id: info.vendor as u32,
            device_id: info.device as u32,

            max_texture_dimension_2d: limits.max_texture_dimension_2d,
            max_texture_dimension_3d: limits.max_texture_dimension_3d,
            max_texture_dimension_cube: limits.max_texture_dimension_2d, // wgpu uses same limit
            max_texture_array_layers: limits.max_texture_array_layers,
            max_uniform_buffer_size: limits.max_uniform_buffer_binding_size as u64,
            max_storage_buffer_size: limits.max_storage_buffer_binding_size as u64,
            max_push_constant_size: limits.max_push_constant_size,
            max_bound_descriptor_sets: limits.max_bind_groups,
            max_color_attachments: limits.max_color_attachments,
            max_compute_workgroup_size: [
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ],
            max_compute_workgroup_invocations: limits.max_compute_invocations_per_workgroup,
            max_compute_workgroup_count: [
                limits.max_compute_workgroups_per_dimension,
                limits.max_compute_workgroups_per_dimension,
                limits.max_compute_workgroups_per_dimension,
            ],
            max_viewport_dimensions: [
                limits.max_texture_dimension_2d,
                limits.max_texture_dimension_2d,
            ],
            max_framebuffer_width: limits.max_texture_dimension_2d,
            max_framebuffer_height: limits.max_texture_dimension_2d,
            max_msaa_samples: 4, // wgpu commonly supports up to 4x

            geometry_shader: false, // wgpu does not expose geometry shaders
            tessellation_shader: false,
            mesh_shader: false,
            ray_tracing: false,
            descriptor_indexing: features.contains(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING),
            multi_draw_indirect: features.contains(wgpu::Features::MULTI_DRAW_INDIRECT),
            bc_texture_compression: features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC),
            astc_texture_compression: features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC),
            etc2_texture_compression: features.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2),
            sampler_anisotropy: true, // wgpu always supports basic anisotropy
            independent_blend: true,
            fill_mode_non_solid: features.contains(wgpu::Features::POLYGON_MODE_LINE),
            wide_lines: false,
            depth_clamp: features.contains(wgpu::Features::DEPTH_CLIP_CONTROL),
            depth_bias_clamp: true,
            shader_float64: features.contains(wgpu::Features::SHADER_F64),
            shader_float16: features.contains(wgpu::Features::SHADER_F16),
            shader_int16: false,
        }
    }

    /// Get a reference to the underlying wgpu device.
    pub fn raw_device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get a reference to the underlying wgpu queue.
    pub fn raw_queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get an `Arc` clone of the wgpu device.
    pub fn device_arc(&self) -> Arc<wgpu::Device> {
        Arc::clone(&self.device)
    }

    /// Get an `Arc` clone of the wgpu queue.
    pub fn queue_arc(&self) -> Arc<wgpu::Queue> {
        Arc::clone(&self.queue)
    }

    /// Get the adapter reference.
    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    /// Get the instance reference.
    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    // -----------------------------------------------------------------------
    // Format conversion helpers
    // -----------------------------------------------------------------------

    /// Convert the engine's `TextureFormat` to wgpu's `TextureFormat`.
    pub fn convert_texture_format(format: TextureFormat) -> wgpu::TextureFormat {
        match format {
            TextureFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
            TextureFormat::Rg8Unorm => wgpu::TextureFormat::Rg8Unorm,
            TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
            TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8UnormSrgb,

            TextureFormat::R8Snorm => wgpu::TextureFormat::R8Snorm,
            TextureFormat::Rg8Snorm => wgpu::TextureFormat::Rg8Snorm,
            TextureFormat::Rgba8Snorm => wgpu::TextureFormat::Rgba8Snorm,

            TextureFormat::R8Uint => wgpu::TextureFormat::R8Uint,
            TextureFormat::Rg8Uint => wgpu::TextureFormat::Rg8Uint,
            TextureFormat::Rgba8Uint => wgpu::TextureFormat::Rgba8Uint,
            TextureFormat::R16Uint => wgpu::TextureFormat::R16Uint,
            TextureFormat::Rg16Uint => wgpu::TextureFormat::Rg16Uint,
            TextureFormat::Rgba16Uint => wgpu::TextureFormat::Rgba16Uint,
            TextureFormat::R32Uint => wgpu::TextureFormat::R32Uint,
            TextureFormat::Rg32Uint => wgpu::TextureFormat::Rg32Uint,
            TextureFormat::Rgba32Uint => wgpu::TextureFormat::Rgba32Uint,

            TextureFormat::R8Sint => wgpu::TextureFormat::R8Sint,
            TextureFormat::Rg8Sint => wgpu::TextureFormat::Rg8Sint,
            TextureFormat::Rgba8Sint => wgpu::TextureFormat::Rgba8Sint,
            TextureFormat::R16Sint => wgpu::TextureFormat::R16Sint,
            TextureFormat::Rg16Sint => wgpu::TextureFormat::Rg16Sint,
            TextureFormat::Rgba16Sint => wgpu::TextureFormat::Rgba16Sint,
            TextureFormat::R32Sint => wgpu::TextureFormat::R32Sint,
            TextureFormat::Rg32Sint => wgpu::TextureFormat::Rg32Sint,
            TextureFormat::Rgba32Sint => wgpu::TextureFormat::Rgba32Sint,

            TextureFormat::R16Float => wgpu::TextureFormat::R16Float,
            TextureFormat::Rg16Float => wgpu::TextureFormat::Rg16Float,
            TextureFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
            TextureFormat::R32Float => wgpu::TextureFormat::R32Float,
            TextureFormat::Rg32Float => wgpu::TextureFormat::Rg32Float,
            TextureFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,

            TextureFormat::Rgb10A2Unorm => wgpu::TextureFormat::Rgb10a2Unorm,
            TextureFormat::Rg11B10Float => wgpu::TextureFormat::Rg11b10Ufloat,
            TextureFormat::Rgb9E5Float => wgpu::TextureFormat::Rgb9e5Ufloat,

            TextureFormat::Depth16Unorm => wgpu::TextureFormat::Depth16Unorm,
            TextureFormat::Depth24Plus => wgpu::TextureFormat::Depth24Plus,
            TextureFormat::Depth24PlusStencil8 => wgpu::TextureFormat::Depth24PlusStencil8,
            TextureFormat::Depth32Float => wgpu::TextureFormat::Depth32Float,
            TextureFormat::Depth32FloatStencil8 => wgpu::TextureFormat::Depth32FloatStencil8,
            TextureFormat::Stencil8 => wgpu::TextureFormat::Stencil8,

            TextureFormat::Bc1RgbaUnorm => wgpu::TextureFormat::Bc1RgbaUnorm,
            TextureFormat::Bc1RgbaUnormSrgb => wgpu::TextureFormat::Bc1RgbaUnormSrgb,
            TextureFormat::Bc2RgbaUnorm => wgpu::TextureFormat::Bc2RgbaUnorm,
            TextureFormat::Bc2RgbaUnormSrgb => wgpu::TextureFormat::Bc2RgbaUnormSrgb,
            TextureFormat::Bc3RgbaUnorm => wgpu::TextureFormat::Bc3RgbaUnorm,
            TextureFormat::Bc3RgbaUnormSrgb => wgpu::TextureFormat::Bc3RgbaUnormSrgb,
            TextureFormat::Bc4RUnorm => wgpu::TextureFormat::Bc4RUnorm,
            TextureFormat::Bc4RSnorm => wgpu::TextureFormat::Bc4RSnorm,
            TextureFormat::Bc5RgUnorm => wgpu::TextureFormat::Bc5RgUnorm,
            TextureFormat::Bc5RgSnorm => wgpu::TextureFormat::Bc5RgSnorm,
            TextureFormat::Bc6hRgbUfloat => wgpu::TextureFormat::Bc6hRgbUfloat,
            TextureFormat::Bc6hRgbSfloat => wgpu::TextureFormat::Bc6hRgbFloat,
            TextureFormat::Bc7RgbaUnorm => wgpu::TextureFormat::Bc7RgbaUnorm,
            TextureFormat::Bc7RgbaUnormSrgb => wgpu::TextureFormat::Bc7RgbaUnormSrgb,

            TextureFormat::Astc4x4Unorm => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B4x4, channel: wgpu::AstcChannel::Unorm },
            TextureFormat::Astc4x4UnormSrgb => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B4x4, channel: wgpu::AstcChannel::UnormSrgb },
            TextureFormat::Astc5x5Unorm => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x5, channel: wgpu::AstcChannel::Unorm },
            TextureFormat::Astc5x5UnormSrgb => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x5, channel: wgpu::AstcChannel::UnormSrgb },
            TextureFormat::Astc6x6Unorm => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x6, channel: wgpu::AstcChannel::Unorm },
            TextureFormat::Astc6x6UnormSrgb => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x6, channel: wgpu::AstcChannel::UnormSrgb },
            TextureFormat::Astc8x8Unorm => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x8, channel: wgpu::AstcChannel::Unorm },
            TextureFormat::Astc8x8UnormSrgb => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x8, channel: wgpu::AstcChannel::UnormSrgb },
            TextureFormat::Astc10x10Unorm => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x10, channel: wgpu::AstcChannel::Unorm },
            TextureFormat::Astc10x10UnormSrgb => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x10, channel: wgpu::AstcChannel::UnormSrgb },
            TextureFormat::Astc12x12Unorm => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x12, channel: wgpu::AstcChannel::Unorm },
            TextureFormat::Astc12x12UnormSrgb => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x12, channel: wgpu::AstcChannel::UnormSrgb },

            TextureFormat::Etc2Rgb8Unorm => wgpu::TextureFormat::Etc2Rgb8Unorm,
            TextureFormat::Etc2Rgb8UnormSrgb => wgpu::TextureFormat::Etc2Rgb8UnormSrgb,
            TextureFormat::Etc2Rgb8A1Unorm => wgpu::TextureFormat::Etc2Rgb8A1Unorm,
            TextureFormat::Etc2Rgb8A1UnormSrgb => wgpu::TextureFormat::Etc2Rgb8A1UnormSrgb,
            TextureFormat::Etc2Rgba8Unorm => wgpu::TextureFormat::Etc2Rgba8Unorm,
            TextureFormat::Etc2Rgba8UnormSrgb => wgpu::TextureFormat::Etc2Rgba8UnormSrgb,
        }
    }

    /// Convert the engine's `TextureFormat` back from wgpu's.
    pub fn convert_texture_format_from_wgpu(format: wgpu::TextureFormat) -> TextureFormat {
        match format {
            wgpu::TextureFormat::Bgra8UnormSrgb => TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Bgra8Unorm => TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8UnormSrgb => TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::Rgba8Unorm => TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::R8Unorm => TextureFormat::R8Unorm,
            _ => TextureFormat::Bgra8UnormSrgb, // safe fallback
        }
    }

    /// Convert engine `BufferUsage` flags to wgpu `BufferUsages`.
    fn convert_buffer_usage(usage: BufferUsage, memory: MemoryLocation) -> wgpu::BufferUsages {
        let mut wu = wgpu::BufferUsages::empty();
        if usage.contains(BufferUsage::VERTEX) {
            wu |= wgpu::BufferUsages::VERTEX;
        }
        if usage.contains(BufferUsage::INDEX) {
            wu |= wgpu::BufferUsages::INDEX;
        }
        if usage.contains(BufferUsage::UNIFORM) {
            wu |= wgpu::BufferUsages::UNIFORM;
        }
        if usage.contains(BufferUsage::STORAGE) {
            wu |= wgpu::BufferUsages::STORAGE;
        }
        if usage.contains(BufferUsage::INDIRECT) {
            wu |= wgpu::BufferUsages::INDIRECT;
        }
        if usage.contains(BufferUsage::TRANSFER_SRC) {
            wu |= wgpu::BufferUsages::COPY_SRC;
        }
        if usage.contains(BufferUsage::TRANSFER_DST) {
            wu |= wgpu::BufferUsages::COPY_DST;
        }
        // If the buffer is CPU-visible, enable MAP_READ / MAP_WRITE.
        match memory {
            MemoryLocation::CpuToGpu => {
                wu |= wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC;
            }
            MemoryLocation::GpuToCpu => {
                wu |= wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
            }
            MemoryLocation::GpuOnly => {}
        }
        wu
    }

    /// Convert engine `TextureUsage` flags to wgpu `TextureUsages`.
    fn convert_texture_usage(usage: TextureUsage) -> wgpu::TextureUsages {
        let mut wu = wgpu::TextureUsages::empty();
        if usage.contains(TextureUsage::SAMPLED) {
            wu |= wgpu::TextureUsages::TEXTURE_BINDING;
        }
        if usage.contains(TextureUsage::STORAGE) {
            wu |= wgpu::TextureUsages::STORAGE_BINDING;
        }
        if usage.contains(TextureUsage::COLOR_ATTACHMENT)
            || usage.contains(TextureUsage::DEPTH_STENCIL)
        {
            wu |= wgpu::TextureUsages::RENDER_ATTACHMENT;
        }
        if usage.contains(TextureUsage::TRANSFER_SRC) {
            wu |= wgpu::TextureUsages::COPY_SRC;
        }
        if usage.contains(TextureUsage::TRANSFER_DST) {
            wu |= wgpu::TextureUsages::COPY_DST;
        }
        // Ensure at least COPY_DST so textures can be written.
        if wu.is_empty() {
            wu = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        }
        wu
    }

    /// Convert engine `TextureDimension` to wgpu.
    fn convert_dimension(dim: TextureDimension) -> wgpu::TextureDimension {
        match dim {
            TextureDimension::D1 => wgpu::TextureDimension::D1,
            TextureDimension::D2 => wgpu::TextureDimension::D2,
            TextureDimension::D3 => wgpu::TextureDimension::D3,
        }
    }

    /// Convert engine `FilterMode` to wgpu.
    fn convert_filter(filter: FilterMode) -> wgpu::FilterMode {
        match filter {
            FilterMode::Nearest => wgpu::FilterMode::Nearest,
            FilterMode::Linear => wgpu::FilterMode::Linear,
        }
    }

    /// Convert engine `AddressMode` to wgpu.
    fn convert_address_mode(mode: AddressMode) -> wgpu::AddressMode {
        match mode {
            AddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            AddressMode::Repeat => wgpu::AddressMode::Repeat,
            AddressMode::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
            AddressMode::ClampToBorder => wgpu::AddressMode::ClampToBorder,
        }
    }

    /// Convert engine `CompareOp` to wgpu.
    fn convert_compare_op(op: CompareOp) -> wgpu::CompareFunction {
        match op {
            CompareOp::Never => wgpu::CompareFunction::Never,
            CompareOp::Less => wgpu::CompareFunction::Less,
            CompareOp::Equal => wgpu::CompareFunction::Equal,
            CompareOp::LessOrEqual => wgpu::CompareFunction::LessEqual,
            CompareOp::Greater => wgpu::CompareFunction::Greater,
            CompareOp::NotEqual => wgpu::CompareFunction::NotEqual,
            CompareOp::GreaterOrEqual => wgpu::CompareFunction::GreaterEqual,
            CompareOp::Always => wgpu::CompareFunction::Always,
        }
    }

    /// Convert engine `PrimitiveTopology` to wgpu.
    fn convert_topology(topo: PrimitiveTopology) -> wgpu::PrimitiveTopology {
        match topo {
            PrimitiveTopology::PointList => wgpu::PrimitiveTopology::PointList,
            PrimitiveTopology::LineList => wgpu::PrimitiveTopology::LineList,
            PrimitiveTopology::LineStrip => wgpu::PrimitiveTopology::LineStrip,
            PrimitiveTopology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
            PrimitiveTopology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
            // wgpu doesn't support these natively; fall back to TriangleList.
            PrimitiveTopology::TriangleFan
            | PrimitiveTopology::LineListWithAdjacency
            | PrimitiveTopology::LineStripWithAdjacency
            | PrimitiveTopology::TriangleListWithAdjacency
            | PrimitiveTopology::TriangleStripWithAdjacency
            | PrimitiveTopology::PatchList => {
                log::warn!("Unsupported topology {:?}, falling back to TriangleList", topo);
                wgpu::PrimitiveTopology::TriangleList
            }
        }
    }

    /// Convert engine `CullMode` to wgpu.
    fn convert_cull_mode(cull: CullMode) -> Option<wgpu::Face> {
        match cull {
            CullMode::None => None,
            CullMode::Front => Some(wgpu::Face::Front),
            CullMode::Back => Some(wgpu::Face::Back),
            CullMode::FrontAndBack => Some(wgpu::Face::Back), // wgpu has no FrontAndBack
        }
    }

    /// Convert engine `FrontFace` to wgpu.
    fn convert_front_face(ff: FrontFace) -> wgpu::FrontFace {
        match ff {
            FrontFace::CounterClockwise => wgpu::FrontFace::Ccw,
            FrontFace::Clockwise => wgpu::FrontFace::Cw,
        }
    }

    /// Convert engine `PolygonMode` to wgpu.
    fn convert_polygon_mode(pm: PolygonMode) -> wgpu::PolygonMode {
        match pm {
            PolygonMode::Fill => wgpu::PolygonMode::Fill,
            PolygonMode::Line => wgpu::PolygonMode::Line,
            PolygonMode::Point => wgpu::PolygonMode::Point,
        }
    }

    /// Convert engine `VertexFormat` to wgpu.
    fn convert_vertex_format(fmt: VertexFormat) -> wgpu::VertexFormat {
        match fmt {
            VertexFormat::Float32 => wgpu::VertexFormat::Float32,
            VertexFormat::Float32x2 => wgpu::VertexFormat::Float32x2,
            VertexFormat::Float32x3 => wgpu::VertexFormat::Float32x3,
            VertexFormat::Float32x4 => wgpu::VertexFormat::Float32x4,
            VertexFormat::Sint32 => wgpu::VertexFormat::Sint32,
            VertexFormat::Sint32x2 => wgpu::VertexFormat::Sint32x2,
            VertexFormat::Sint32x3 => wgpu::VertexFormat::Sint32x3,
            VertexFormat::Sint32x4 => wgpu::VertexFormat::Sint32x4,
            VertexFormat::Uint32 => wgpu::VertexFormat::Uint32,
            VertexFormat::Uint32x2 => wgpu::VertexFormat::Uint32x2,
            VertexFormat::Uint32x3 => wgpu::VertexFormat::Uint32x3,
            VertexFormat::Uint32x4 => wgpu::VertexFormat::Uint32x4,
            VertexFormat::Sint16x2 => wgpu::VertexFormat::Sint16x2,
            VertexFormat::Sint16x4 => wgpu::VertexFormat::Sint16x4,
            VertexFormat::Uint16x2 => wgpu::VertexFormat::Uint16x2,
            VertexFormat::Uint16x4 => wgpu::VertexFormat::Uint16x4,
            VertexFormat::Snorm16x2 => wgpu::VertexFormat::Snorm16x2,
            VertexFormat::Snorm16x4 => wgpu::VertexFormat::Snorm16x4,
            VertexFormat::Unorm16x2 => wgpu::VertexFormat::Unorm16x2,
            VertexFormat::Unorm16x4 => wgpu::VertexFormat::Unorm16x4,
            VertexFormat::Sint8x2 => wgpu::VertexFormat::Sint8x2,
            VertexFormat::Sint8x4 => wgpu::VertexFormat::Sint8x4,
            VertexFormat::Uint8x2 => wgpu::VertexFormat::Uint8x2,
            VertexFormat::Uint8x4 => wgpu::VertexFormat::Uint8x4,
            VertexFormat::Snorm8x2 => wgpu::VertexFormat::Snorm8x2,
            VertexFormat::Snorm8x4 => wgpu::VertexFormat::Snorm8x4,
            VertexFormat::Unorm8x2 => wgpu::VertexFormat::Unorm8x2,
            VertexFormat::Unorm8x4 => wgpu::VertexFormat::Unorm8x4,
            VertexFormat::Unorm10_10_10_2 => wgpu::VertexFormat::Unorm10_10_10_2,
        }
    }

    /// Convert engine `VertexStepMode` to wgpu.
    fn convert_step_mode(sm: VertexStepMode) -> wgpu::VertexStepMode {
        match sm {
            VertexStepMode::Vertex => wgpu::VertexStepMode::Vertex,
            VertexStepMode::Instance => wgpu::VertexStepMode::Instance,
        }
    }

    /// Convert engine `BlendFactor` to wgpu.
    fn convert_blend_factor(bf: BlendFactor) -> wgpu::BlendFactor {
        match bf {
            BlendFactor::Zero => wgpu::BlendFactor::Zero,
            BlendFactor::One => wgpu::BlendFactor::One,
            BlendFactor::SrcColor => wgpu::BlendFactor::Src,
            BlendFactor::OneMinusSrcColor => wgpu::BlendFactor::OneMinusSrc,
            BlendFactor::DstColor => wgpu::BlendFactor::Dst,
            BlendFactor::OneMinusDstColor => wgpu::BlendFactor::OneMinusDst,
            BlendFactor::SrcAlpha => wgpu::BlendFactor::SrcAlpha,
            BlendFactor::OneMinusSrcAlpha => wgpu::BlendFactor::OneMinusSrcAlpha,
            BlendFactor::DstAlpha => wgpu::BlendFactor::DstAlpha,
            BlendFactor::OneMinusDstAlpha => wgpu::BlendFactor::OneMinusDstAlpha,
            BlendFactor::ConstantColor => wgpu::BlendFactor::Constant,
            BlendFactor::OneMinusConstantColor => wgpu::BlendFactor::OneMinusConstant,
            BlendFactor::ConstantAlpha => wgpu::BlendFactor::Constant,
            BlendFactor::OneMinusConstantAlpha => wgpu::BlendFactor::OneMinusConstant,
            BlendFactor::SrcAlphaSaturate => wgpu::BlendFactor::SrcAlphaSaturated,
            BlendFactor::Src1Color => wgpu::BlendFactor::Src1,
            BlendFactor::OneMinusSrc1Color => wgpu::BlendFactor::OneMinusSrc1,
            BlendFactor::Src1Alpha => wgpu::BlendFactor::Src1Alpha,
            BlendFactor::OneMinusSrc1Alpha => wgpu::BlendFactor::OneMinusSrc1Alpha,
        }
    }

    /// Convert engine `BlendOp` to wgpu.
    fn convert_blend_op(bo: BlendOp) -> wgpu::BlendOperation {
        match bo {
            BlendOp::Add => wgpu::BlendOperation::Add,
            BlendOp::Subtract => wgpu::BlendOperation::Subtract,
            BlendOp::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
            BlendOp::Min => wgpu::BlendOperation::Min,
            BlendOp::Max => wgpu::BlendOperation::Max,
        }
    }

    /// Write data to a GPU buffer via the queue's staging mechanism.
    pub fn write_buffer(&self, handle: BufferHandle, offset: u64, data: &[u8]) -> Result<(), RenderError> {
        let pool = self.buffers.lock();
        let genovo_handle = genovo_core::Handle::<WgpuBuffer>::new(handle.0 as u32, (handle.0 >> 32) as u32);
        let buf = pool.get(genovo_handle).ok_or(RenderError::InvalidHandle)?;
        self.queue.write_buffer(&buf.buffer, offset, data);
        Ok(())
    }

    /// Write data to a GPU texture via the queue's staging mechanism.
    pub fn write_texture(
        &self,
        handle: TextureHandle,
        data: &[u8],
        bytes_per_row: u32,
        rows_per_image: u32,
        origin: [u32; 3],
        extent: [u32; 3],
    ) -> Result<(), RenderError> {
        let pool = self.textures.lock();
        let genovo_handle = genovo_core::Handle::<WgpuTexture>::new(handle.0 as u32, (handle.0 >> 32) as u32);
        let tex = pool.get(genovo_handle).ok_or(RenderError::InvalidHandle)?;
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: origin[0],
                    y: origin[1],
                    z: origin[2],
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(rows_per_image),
            },
            wgpu::Extent3d {
                width: extent[0],
                height: extent[1],
                depth_or_array_layers: extent[2],
            },
        );
        Ok(())
    }

    /// Create a render pipeline from raw WGSL source and a target format.
    ///
    /// This is a convenience method that bypasses the handle pool shader system
    /// and creates a pipeline directly. Used internally by the `Renderer` for
    /// built-in pipelines.
    pub fn create_pipeline_from_wgsl(
        &self,
        wgsl_source: &str,
        vertex_entry: &str,
        fragment_entry: &str,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        vertex_buffers: &[wgpu::VertexBufferLayout<'_>],
        label: &str,
    ) -> Result<PipelineHandle, RenderError> {
        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label}_layout")),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let depth_stencil = depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        let render_pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some(vertex_entry),
                buffers: vertex_buffers,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some(fragment_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
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
            depth_stencil,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let wrapper = WgpuPipeline {
            pipeline: WgpuPipelineKind::Render(render_pipeline),
        };

        let mut pool = self.pipelines.lock();
        let genovo_handle = pool.insert(wrapper);
        let packed = (genovo_handle.index() as u64) | ((genovo_handle.generation() as u64) << 32);
        Ok(PipelineHandle(packed))
    }

    /// Look up a render pipeline by handle and call `f` with it.
    pub fn with_render_pipeline<R>(
        &self,
        handle: PipelineHandle,
        f: impl FnOnce(&wgpu::RenderPipeline) -> R,
    ) -> Result<R, RenderError> {
        let pool = self.pipelines.lock();
        let genovo_handle = genovo_core::Handle::<WgpuPipeline>::new(
            handle.0 as u32,
            (handle.0 >> 32) as u32,
        );
        let wrapper = pool.get(genovo_handle).ok_or(RenderError::InvalidHandle)?;
        match &wrapper.pipeline {
            WgpuPipelineKind::Render(rp) => Ok(f(rp)),
            WgpuPipelineKind::Compute(_) => Err(RenderError::Internal(
                "Expected a render pipeline but found a compute pipeline".into(),
            )),
        }
    }

    /// Create a wgpu command encoder (low-level access for the renderer).
    pub fn create_command_encoder(&self, label: &str) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(label),
        })
    }

    /// Submit a list of wgpu command buffers directly.
    pub fn submit_wgpu_commands(
        &self,
        commands: impl IntoIterator<Item = wgpu::CommandBuffer>,
    ) {
        self.queue.submit(commands);
    }
}

// ---------------------------------------------------------------------------
// RenderDevice trait implementation
// ---------------------------------------------------------------------------

impl RenderDevice for WgpuDevice {
    fn create_buffer(&self, desc: &BufferDesc) -> crate::interface::device::Result<BufferHandle> {
        let wgpu_usage = Self::convert_buffer_usage(desc.usage, desc.memory);
        let mapped_at_creation = desc.memory == MemoryLocation::CpuToGpu;

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: desc.label.as_deref(),
            size: desc.size,
            usage: wgpu_usage,
            mapped_at_creation,
        });

        let wrapper = WgpuBuffer {
            buffer,
            size: desc.size,
            usage: desc.usage,
            memory: desc.memory,
        };

        let mut pool = self.buffers.lock();
        let genovo_handle = pool.insert(wrapper);
        let packed = (genovo_handle.index() as u64) | ((genovo_handle.generation() as u64) << 32);
        Ok(BufferHandle(packed))
    }

    fn create_texture(
        &self,
        desc: &TextureDesc,
    ) -> crate::interface::device::Result<TextureHandle> {
        let wgpu_format = Self::convert_texture_format(desc.format);
        let wgpu_dim = Self::convert_dimension(desc.dimension);
        let wgpu_usage = Self::convert_texture_usage(desc.usage);

        let size = wgpu::Extent3d {
            width: desc.width,
            height: desc.height,
            depth_or_array_layers: desc.depth_or_array_layers,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: desc.label.as_deref(),
            size,
            mip_level_count: desc.mip_levels,
            sample_count: desc.sample_count,
            dimension: wgpu_dim,
            format: wgpu_format,
            usage: wgpu_usage,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let wrapper = WgpuTexture {
            texture,
            view,
            format: desc.format,
            width: desc.width,
            height: desc.height,
            depth_or_array_layers: desc.depth_or_array_layers,
            mip_levels: desc.mip_levels,
        };

        let mut pool = self.textures.lock();
        let genovo_handle = pool.insert(wrapper);
        let packed = (genovo_handle.index() as u64) | ((genovo_handle.generation() as u64) << 32);
        Ok(TextureHandle(packed))
    }

    fn create_sampler(
        &self,
        desc: &SamplerDesc,
    ) -> crate::interface::device::Result<SamplerHandle> {
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: desc.label.as_deref(),
            address_mode_u: Self::convert_address_mode(desc.address_mode_u),
            address_mode_v: Self::convert_address_mode(desc.address_mode_v),
            address_mode_w: Self::convert_address_mode(desc.address_mode_w),
            mag_filter: Self::convert_filter(desc.mag_filter),
            min_filter: Self::convert_filter(desc.min_filter),
            mipmap_filter: Self::convert_filter(desc.mipmap_filter),
            lod_min_clamp: desc.lod_min_clamp,
            lod_max_clamp: desc.lod_max_clamp,
            compare: desc.compare.map(Self::convert_compare_op),
            anisotropy_clamp: desc.max_anisotropy.max(1) as u16,
            border_color: None,
        });

        let wrapper = WgpuSampler { sampler };

        let mut pool = self.samplers.lock();
        let genovo_handle = pool.insert(wrapper);
        let packed = (genovo_handle.index() as u64) | ((genovo_handle.generation() as u64) << 32);
        Ok(SamplerHandle(packed))
    }

    fn create_shader(
        &self,
        desc: &ShaderDesc,
    ) -> crate::interface::device::Result<ShaderHandle> {
        // Interpret the bytecode as UTF-8 WGSL source. wgpu v24 does not
        // enable SPIR-V input by default, so we require WGSL.
        let wgsl_str = std::str::from_utf8(&desc.bytecode).map_err(|_| {
            RenderError::ShaderLoad(
                "Shader bytecode is not valid WGSL (UTF-8). SPIR-V input requires the \
                 wgpu 'spirv' feature which is not enabled."
                    .into(),
            )
        })?;
        let source = wgpu::ShaderSource::Wgsl(wgsl_str.to_string().into());

        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: desc.label.as_deref(),
            source,
        });

        let wrapper = WgpuShaderModule {
            module,
            stage: desc.stage,
            entry_point: desc.entry_point.clone(),
        };

        let mut pool = self.shader_modules.lock();
        let genovo_handle = pool.insert(wrapper);
        let packed = (genovo_handle.index() as u64) | ((genovo_handle.generation() as u64) << 32);
        Ok(ShaderHandle(packed))
    }

    fn create_pipeline(
        &self,
        desc: &GraphicsPipelineDesc,
    ) -> crate::interface::device::Result<PipelineHandle> {
        // Look up vertex and fragment shader modules.
        let shaders = self.shader_modules.lock();

        let vs_genovo = genovo_core::Handle::<WgpuShaderModule>::new(
            desc.vertex_stage.module.0 as u32,
            (desc.vertex_stage.module.0 >> 32) as u32,
        );
        let vs = shaders.get(vs_genovo).ok_or_else(|| {
            RenderError::InvalidHandle
        })?;

        let fs_module;
        let fs_entry;
        if let Some(ref frag_stage) = desc.fragment_stage {
            let fs_genovo = genovo_core::Handle::<WgpuShaderModule>::new(
                frag_stage.module.0 as u32,
                (frag_stage.module.0 >> 32) as u32,
            );
            let fs_ref = shaders.get(fs_genovo).ok_or(RenderError::InvalidHandle)?;
            fs_module = Some(&fs_ref.module);
            fs_entry = Some(frag_stage.entry_point.clone());
        } else {
            fs_module = None;
            fs_entry = None;
        }

        // Build vertex buffer layouts.
        // We need to build the attribute arrays with stable addresses.
        let mut wgpu_attrs: Vec<Vec<wgpu::VertexAttribute>> = Vec::new();
        for buf_layout in &desc.vertex_input.buffers {
            let attrs: Vec<wgpu::VertexAttribute> = buf_layout
                .attributes
                .iter()
                .map(|a| wgpu::VertexAttribute {
                    format: Self::convert_vertex_format(a.format),
                    offset: a.offset as u64,
                    shader_location: a.location,
                })
                .collect();
            wgpu_attrs.push(attrs);
        }

        let wgpu_vb_layouts: Vec<wgpu::VertexBufferLayout<'_>> = desc
            .vertex_input
            .buffers
            .iter()
            .zip(wgpu_attrs.iter())
            .map(|(buf_layout, attrs)| wgpu::VertexBufferLayout {
                array_stride: buf_layout.stride as u64,
                step_mode: Self::convert_step_mode(buf_layout.step_mode),
                attributes: attrs,
            })
            .collect();

        // Colour targets.
        let color_targets: Vec<Option<wgpu::ColorTargetState>> = desc
            .blend
            .targets
            .iter()
            .map(|ct| {
                let blend = if ct.blend_enable {
                    Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: Self::convert_blend_factor(ct.src_color_blend_factor),
                            dst_factor: Self::convert_blend_factor(ct.dst_color_blend_factor),
                            operation: Self::convert_blend_op(ct.color_blend_op),
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: Self::convert_blend_factor(ct.src_alpha_blend_factor),
                            dst_factor: Self::convert_blend_factor(ct.dst_alpha_blend_factor),
                            operation: Self::convert_blend_op(ct.alpha_blend_op),
                        },
                    })
                } else {
                    None
                };
                Some(wgpu::ColorTargetState {
                    format: Self::convert_texture_format(ct.format),
                    blend,
                    write_mask: wgpu::ColorWrites::from_bits_truncate(ct.write_mask as u32),
                })
            })
            .collect();

        // If no explicit blend targets, use the color attachment formats.
        let effective_targets: Vec<Option<wgpu::ColorTargetState>> = if color_targets.is_empty() {
            desc.color_attachment_formats
                .iter()
                .map(|fmt| {
                    Some(wgpu::ColorTargetState {
                        format: Self::convert_texture_format(*fmt),
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })
                })
                .collect()
        } else {
            color_targets
        };

        // Depth / stencil.
        let depth_stencil = desc.depth_attachment_format.map(|fmt| {
            wgpu::DepthStencilState {
                format: Self::convert_texture_format(fmt),
                depth_write_enabled: desc.depth_stencil.depth_write_enable,
                depth_compare: if desc.depth_stencil.depth_test_enable {
                    Self::convert_compare_op(desc.depth_stencil.depth_compare)
                } else {
                    wgpu::CompareFunction::Always
                },
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: desc.rasterizer.depth_bias_constant as i32,
                    slope_scale: desc.rasterizer.depth_bias_slope,
                    clamp: desc.rasterizer.depth_bias_clamp,
                },
            }
        });

        // Pipeline layout (empty for now -- descriptor sets are not wired up yet).
        let pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: desc.label.as_deref(),
                    bind_group_layouts: &[],
                    push_constant_ranges: &[],
                });

        let vs_entry = desc.vertex_stage.entry_point.clone();

        let fragment_state;
        // We have to be careful with lifetimes here. Build the fragment state
        // only if both module and entry point are available.
        let fragment_state_ref = if let (Some(fs_mod), Some(entry)) = (fs_module, fs_entry.as_ref()) {
            fragment_state = wgpu::FragmentState {
                module: fs_mod,
                entry_point: Some(entry),
                targets: &effective_targets,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            };
            Some(fragment_state)
        } else {
            None
        };

        let render_pipeline =
            self.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: desc.label.as_deref(),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &vs.module,
                        entry_point: Some(&vs_entry),
                        buffers: &wgpu_vb_layouts,
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: fragment_state_ref,
                    primitive: wgpu::PrimitiveState {
                        topology: Self::convert_topology(desc.primitive_topology),
                        strip_index_format: None,
                        front_face: Self::convert_front_face(desc.rasterizer.front_face),
                        cull_mode: Self::convert_cull_mode(desc.rasterizer.cull_mode),
                        polygon_mode: Self::convert_polygon_mode(desc.rasterizer.polygon_mode),
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil,
                    multisample: wgpu::MultisampleState {
                        count: desc.multisample.count,
                        mask: desc.multisample.mask,
                        alpha_to_coverage_enabled: desc.multisample.alpha_to_coverage_enable,
                    },
                    multiview: None,
                    cache: None,
                });

        // Release shader lock before acquiring pipelines lock.
        drop(shaders);

        let wrapper = WgpuPipeline {
            pipeline: WgpuPipelineKind::Render(render_pipeline),
        };

        let mut pool = self.pipelines.lock();
        let genovo_handle = pool.insert(wrapper);
        let packed = (genovo_handle.index() as u64) | ((genovo_handle.generation() as u64) << 32);
        Ok(PipelineHandle(packed))
    }

    fn create_compute_pipeline(
        &self,
        desc: &ComputePipelineDesc,
    ) -> crate::interface::device::Result<PipelineHandle> {
        let shaders = self.shader_modules.lock();
        let cs_genovo = genovo_core::Handle::<WgpuShaderModule>::new(
            desc.compute_stage.module.0 as u32,
            (desc.compute_stage.module.0 >> 32) as u32,
        );
        let cs = shaders.get(cs_genovo).ok_or(RenderError::InvalidHandle)?;

        let pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: desc.label.as_deref(),
                    bind_group_layouts: &[],
                    push_constant_ranges: &[],
                });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: desc.label.as_deref(),
                    layout: Some(&pipeline_layout),
                    module: &cs.module,
                    entry_point: Some(&desc.compute_stage.entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        drop(shaders);

        let wrapper = WgpuPipeline {
            pipeline: WgpuPipelineKind::Compute(compute_pipeline),
        };

        let mut pool = self.pipelines.lock();
        let genovo_handle = pool.insert(wrapper);
        let packed = (genovo_handle.index() as u64) | ((genovo_handle.generation() as u64) << 32);
        Ok(PipelineHandle(packed))
    }

    fn create_render_pass(
        &self,
        desc: &RenderPassDesc,
    ) -> crate::interface::device::Result<RenderPassHandle> {
        // wgpu uses dynamic render passes so we just store the descriptor
        // metadata for compatibility checks.
        let color_format = desc.color_attachments.first().map(|a| a.format);
        let depth_format = desc.depth_stencil_attachment.as_ref().map(|a| a.format);
        let sample_count = desc
            .color_attachments
            .first()
            .map(|a| a.samples)
            .unwrap_or(1);

        let wrapper = WgpuRenderPass {
            color_format,
            depth_format,
            sample_count,
        };

        let mut pool = self.render_passes.lock();
        let genovo_handle = pool.insert(wrapper);
        let packed = (genovo_handle.index() as u64) | ((genovo_handle.generation() as u64) << 32);
        Ok(RenderPassHandle(packed))
    }

    fn create_framebuffer(
        &self,
        desc: &FramebufferDesc,
    ) -> crate::interface::device::Result<FramebufferHandle> {
        // wgpu does not have framebuffer objects; we store metadata.
        let wrapper = WgpuFramebuffer {
            color_attachments: desc.attachments.clone(),
            depth_attachment: None,
            width: desc.width,
            height: desc.height,
        };

        let mut pool = self.framebuffers.lock();
        let genovo_handle = pool.insert(wrapper);
        let packed = (genovo_handle.index() as u64) | ((genovo_handle.generation() as u64) << 32);
        Ok(FramebufferHandle(packed))
    }

    fn submit_commands(
        &self,
        _queue: QueueType,
        _cmds: &[CommandBuffer],
    ) -> crate::interface::device::Result<FenceHandle> {
        // The wgpu backend uses direct command encoder submission rather than
        // the engine's CommandBuffer abstraction. This implementation provides
        // a fence token for API compatibility.
        let fence = self.fence_tracker.lock().next();
        Ok(fence)
    }

    fn wait_idle(&self) -> crate::interface::device::Result<()> {
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }

    fn destroy_buffer(&self, handle: BufferHandle) {
        let mut pool = self.buffers.lock();
        let genovo_handle = genovo_core::Handle::<WgpuBuffer>::new(
            handle.0 as u32,
            (handle.0 >> 32) as u32,
        );
        if let Ok(wrapper) = pool.remove(genovo_handle) {
            wrapper.buffer.destroy();
        }
    }

    fn destroy_texture(&self, handle: TextureHandle) {
        let mut pool = self.textures.lock();
        let genovo_handle = genovo_core::Handle::<WgpuTexture>::new(
            handle.0 as u32,
            (handle.0 >> 32) as u32,
        );
        if let Ok(wrapper) = pool.remove(genovo_handle) {
            wrapper.texture.destroy();
        }
    }

    fn destroy_pipeline(&self, handle: PipelineHandle) {
        let mut pool = self.pipelines.lock();
        let genovo_handle = genovo_core::Handle::<WgpuPipeline>::new(
            handle.0 as u32,
            (handle.0 >> 32) as u32,
        );
        // wgpu pipelines are dropped automatically.
        let _ = pool.remove(genovo_handle);
    }

    fn get_capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    fn map_buffer(&self, handle: BufferHandle) -> crate::interface::device::Result<*mut u8> {
        let pool = self.buffers.lock();
        let genovo_handle = genovo_core::Handle::<WgpuBuffer>::new(
            handle.0 as u32,
            (handle.0 >> 32) as u32,
        );
        let buf = pool.get(genovo_handle).ok_or(RenderError::InvalidHandle)?;
        let slice = buf.buffer.slice(..);

        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Write, move |result| {
            let _ = tx.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|_| RenderError::Internal("Buffer map channel closed".into()))?
            .map_err(|e| RenderError::Internal(format!("Buffer map failed: {e}")))?;

        let mapped = slice.get_mapped_range_mut();
        Ok(mapped.as_ptr() as *mut u8)
    }

    fn unmap_buffer(&self, handle: BufferHandle) {
        let pool = self.buffers.lock();
        let genovo_handle = genovo_core::Handle::<WgpuBuffer>::new(
            handle.0 as u32,
            (handle.0 >> 32) as u32,
        );
        if let Some(buf) = pool.get(genovo_handle) {
            buf.buffer.unmap();
        }
    }
}

// ---------------------------------------------------------------------------
// WgpuSurface
// ---------------------------------------------------------------------------

/// Manages a wgpu presentation surface and its swapchain configuration.
///
/// This is separate from `WgpuDevice` so that the device can be used without
/// a surface (headless mode) and so that multiple surfaces can share a device
/// in the future.
pub struct WgpuSurface {
    /// The wgpu surface for presentation.
    surface: wgpu::Surface<'static>,
    /// Current surface configuration.
    config: wgpu::SurfaceConfiguration,
    /// The preferred surface texture format.
    format: wgpu::TextureFormat,
    /// The engine's representation of the surface format.
    engine_format: TextureFormat,
}

impl WgpuSurface {
    /// Create and configure a new surface.
    pub fn new(
        device: &WgpuDevice,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
    ) -> Self {
        let surface_caps = surface.get_capabilities(device.adapter());
        let format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let engine_format = WgpuDevice::convert_texture_format_from_wgpu(format);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(device.raw_device(), &config);

        Self {
            surface,
            config,
            format,
            engine_format,
        }
    }

    /// Reconfigure the surface after a resize.
    pub fn resize(&mut self, device: &WgpuDevice, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(device.raw_device(), &self.config);
    }

    /// Get the next swapchain texture for rendering.
    pub fn get_current_texture(&self) -> Result<wgpu::SurfaceTexture, RenderError> {
        self.surface.get_current_texture().map_err(|e| match e {
            wgpu::SurfaceError::OutOfMemory => {
                RenderError::OutOfMemory("Surface out of memory".into())
            }
            wgpu::SurfaceError::Lost => RenderError::SurfaceLost,
            wgpu::SurfaceError::Outdated => RenderError::SwapchainOutOfDate,
            wgpu::SurfaceError::Timeout => RenderError::Timeout,
            _ => RenderError::Internal(format!("Surface error: {e}")),
        })
    }

    /// The preferred wgpu texture format for this surface.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// The engine-side texture format.
    pub fn engine_format(&self) -> TextureFormat {
        self.engine_format
    }

    /// Current configured width.
    pub fn width(&self) -> u32 {
        self.config.width
    }

    /// Current configured height.
    pub fn height(&self) -> u32 {
        self.config.height
    }
}
