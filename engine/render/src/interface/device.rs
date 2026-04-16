// engine/render/src/interface/device.rs
//
// Core `RenderDevice` trait that every backend (Vulkan, DX12, Metal) must
// implement. This is the single point of abstraction between the engine's
// renderer and the platform GPU API.

use super::command_buffer::CommandBuffer;
use super::pipeline::{ComputePipelineDesc, GraphicsPipelineDesc};
use super::resource::{
    BufferDesc, BufferHandle, FramebufferDesc, FramebufferHandle, FenceHandle, PipelineHandle,
    RenderPassDesc, RenderPassHandle, SamplerDesc, SamplerHandle, ShaderDesc, ShaderHandle,
    TextureDesc, TextureHandle,
};
use crate::RenderError;

// ---------------------------------------------------------------------------
// Queue type
// ---------------------------------------------------------------------------

/// The type of GPU command queue a submission targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueueType {
    /// General-purpose queue supporting graphics, compute, and transfer.
    Graphics,
    /// Asynchronous compute queue.
    Compute,
    /// Dedicated transfer / DMA queue.
    Transfer,
}

// ---------------------------------------------------------------------------
// Device capabilities
// ---------------------------------------------------------------------------

/// Advertised capabilities and limits of the underlying GPU / driver.
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Human-readable device name (e.g. "NVIDIA GeForce RTX 4090").
    pub device_name: String,
    /// Vendor identifier.
    pub vendor_id: u32,
    /// Device identifier.
    pub device_id: u32,

    // -- Limits --
    /// Maximum 1-D/2-D texture dimension.
    pub max_texture_dimension_2d: u32,
    /// Maximum 3-D texture dimension.
    pub max_texture_dimension_3d: u32,
    /// Maximum cube-map texture dimension.
    pub max_texture_dimension_cube: u32,
    /// Maximum number of texture array layers.
    pub max_texture_array_layers: u32,
    /// Maximum uniform buffer binding size in bytes.
    pub max_uniform_buffer_size: u64,
    /// Maximum storage buffer binding size in bytes.
    pub max_storage_buffer_size: u64,
    /// Maximum push-constant size in bytes.
    pub max_push_constant_size: u32,
    /// Maximum bound descriptor sets.
    pub max_bound_descriptor_sets: u32,
    /// Maximum colour attachments per render pass.
    pub max_color_attachments: u32,
    /// Maximum compute workgroup size on each axis.
    pub max_compute_workgroup_size: [u32; 3],
    /// Maximum compute workgroup invocations.
    pub max_compute_workgroup_invocations: u32,
    /// Maximum compute workgroup count on each axis.
    pub max_compute_workgroup_count: [u32; 3],
    /// Maximum viewport dimensions.
    pub max_viewport_dimensions: [u32; 2],
    /// Maximum framebuffer width.
    pub max_framebuffer_width: u32,
    /// Maximum framebuffer height.
    pub max_framebuffer_height: u32,
    /// Maximum MSAA sample count for colour attachments.
    pub max_msaa_samples: u32,

    // -- Feature flags --
    /// Supports geometry shaders.
    pub geometry_shader: bool,
    /// Supports tessellation shaders.
    pub tessellation_shader: bool,
    /// Supports mesh shaders.
    pub mesh_shader: bool,
    /// Supports hardware ray tracing.
    pub ray_tracing: bool,
    /// Supports bindless / descriptor indexing.
    pub descriptor_indexing: bool,
    /// Supports multi-draw indirect.
    pub multi_draw_indirect: bool,
    /// Supports BC (block compression) texture formats.
    pub bc_texture_compression: bool,
    /// Supports ASTC texture formats.
    pub astc_texture_compression: bool,
    /// Supports ETC2 texture formats.
    pub etc2_texture_compression: bool,
    /// Supports anisotropic filtering.
    pub sampler_anisotropy: bool,
    /// Supports independent blending per colour attachment.
    pub independent_blend: bool,
    /// Supports non-fill polygon modes.
    pub fill_mode_non_solid: bool,
    /// Supports wide lines.
    pub wide_lines: bool,
    /// Supports depth clamping.
    pub depth_clamp: bool,
    /// Supports depth bias clamping.
    pub depth_bias_clamp: bool,
    /// Supports 64-bit floating-point operations in shaders.
    pub shader_float64: bool,
    /// Supports 16-bit floating-point operations in shaders.
    pub shader_float16: bool,
    /// Supports 16-bit integer operations in shaders.
    pub shader_int16: bool,
}

// ---------------------------------------------------------------------------
// RenderDevice trait
// ---------------------------------------------------------------------------

/// Result type alias used throughout the render crate.
pub type Result<T> = std::result::Result<T, RenderError>;

/// The abstract GPU device interface.
///
/// Backends implement this trait to expose resource creation, command
/// submission, and synchronisation to the rest of the engine. The trait is
/// object-safe so that a `Box<dyn RenderDevice>` can be held by the
/// high-level renderer.
pub trait RenderDevice: Send + Sync {
    // -- Resource creation --------------------------------------------------

    /// Allocate a GPU buffer.
    fn create_buffer(&self, desc: &BufferDesc) -> Result<BufferHandle>;

    /// Allocate a GPU texture (image).
    fn create_texture(&self, desc: &TextureDesc) -> Result<TextureHandle>;

    /// Create a sampler object.
    fn create_sampler(&self, desc: &SamplerDesc) -> Result<SamplerHandle>;

    /// Compile and load a shader module.
    fn create_shader(&self, desc: &ShaderDesc) -> Result<ShaderHandle>;

    /// Create a graphics pipeline.
    fn create_pipeline(&self, desc: &GraphicsPipelineDesc) -> Result<PipelineHandle>;

    /// Create a compute pipeline.
    fn create_compute_pipeline(&self, desc: &ComputePipelineDesc) -> Result<PipelineHandle>;

    /// Create a render pass.
    fn create_render_pass(&self, desc: &RenderPassDesc) -> Result<RenderPassHandle>;

    /// Create a framebuffer compatible with the given render pass.
    fn create_framebuffer(&self, desc: &FramebufferDesc) -> Result<FramebufferHandle>;

    // -- Command submission -------------------------------------------------

    /// Submit one or more recorded command buffers to the given queue.
    ///
    /// Returns a fence handle that can be used to wait for completion.
    fn submit_commands(
        &self,
        queue: QueueType,
        cmds: &[CommandBuffer],
    ) -> Result<FenceHandle>;

    // -- Synchronisation ----------------------------------------------------

    /// Block the calling thread until the device is idle (all queues drained).
    fn wait_idle(&self) -> Result<()>;

    // -- Resource destruction -----------------------------------------------

    /// Destroy a buffer. The handle must not be in use by any pending command.
    fn destroy_buffer(&self, handle: BufferHandle);

    /// Destroy a texture. The handle must not be in use by any pending command.
    fn destroy_texture(&self, handle: TextureHandle);

    /// Destroy a pipeline. The handle must not be in use by any pending command.
    fn destroy_pipeline(&self, handle: PipelineHandle);

    // -- Queries ------------------------------------------------------------

    /// Query the capabilities and limits of this device.
    fn get_capabilities(&self) -> &DeviceCapabilities;

    // -- Memory mapping -----------------------------------------------------

    /// Map a host-visible buffer into CPU address space.
    ///
    /// # Safety
    /// The returned pointer is only valid until [`unmap_buffer`](Self::unmap_buffer)
    /// is called or the buffer is destroyed.
    fn map_buffer(&self, handle: BufferHandle) -> Result<*mut u8>;

    /// Unmap a previously mapped buffer.
    fn unmap_buffer(&self, handle: BufferHandle);
}
