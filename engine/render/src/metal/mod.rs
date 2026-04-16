// engine/render/src/metal/mod.rs
//
// Metal backend implementation. Delegates to the wgpu backend configured
// with `wgpu::Backends::METAL`, so that wgpu handles all Metal API calls
// (device creation, command queues, render pipeline states, texture and
// buffer allocation) under the hood.

use crate::interface::command_buffer::CommandBuffer;
use crate::interface::device::{DeviceCapabilities, QueueType, RenderDevice, Result};
use crate::interface::pipeline::{ComputePipelineDesc, GraphicsPipelineDesc};
use crate::interface::resource::{
    BufferDesc, BufferHandle, FramebufferDesc, FramebufferHandle, FenceHandle, PipelineHandle,
    RenderPassDesc, RenderPassHandle, SamplerDesc, SamplerHandle, ShaderDesc, ShaderHandle,
    TextureDesc, TextureHandle,
};
use crate::wgpu_backend::WgpuDevice;
use crate::RenderError;

// ---------------------------------------------------------------------------
// MetalDevice
// ---------------------------------------------------------------------------

/// Metal implementation of [`RenderDevice`].
///
/// This is a thin wrapper around [`WgpuDevice`] configured to use only the
/// Metal backend. All resource creation, command submission, and
/// synchronisation is delegated to the inner wgpu device.
///
/// This backend targets Apple platforms (macOS, iOS, visionOS). When
/// Metal-specific functionality (e.g. MetalFX upscaling, argument buffers)
/// is needed beyond what wgpu exposes, it can be accessed through the raw
/// wgpu device's underlying Metal handles.
pub struct MetalDevice {
    /// The wgpu device restricted to the Metal backend.
    inner: WgpuDevice,
}

impl MetalDevice {
    /// Create a Metal device.
    ///
    /// The device is created headless (without a surface). Presentation
    /// can be set up separately via `WgpuSurface`.
    pub fn new() -> std::result::Result<Self, RenderError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..Default::default()
        });
        let inner = WgpuDevice::new_with_instance(instance)?;
        log::info!(
            "MetalDevice created: {}",
            inner.get_capabilities().device_name
        );
        Ok(Self { inner })
    }

    /// Access the underlying `WgpuDevice`.
    pub fn wgpu_device(&self) -> &WgpuDevice {
        &self.inner
    }
}

impl RenderDevice for MetalDevice {
    fn create_buffer(&self, desc: &BufferDesc) -> Result<BufferHandle> {
        self.inner.create_buffer(desc)
    }

    fn create_texture(&self, desc: &TextureDesc) -> Result<TextureHandle> {
        self.inner.create_texture(desc)
    }

    fn create_sampler(&self, desc: &SamplerDesc) -> Result<SamplerHandle> {
        self.inner.create_sampler(desc)
    }

    fn create_shader(&self, desc: &ShaderDesc) -> Result<ShaderHandle> {
        self.inner.create_shader(desc)
    }

    fn create_pipeline(&self, desc: &GraphicsPipelineDesc) -> Result<PipelineHandle> {
        self.inner.create_pipeline(desc)
    }

    fn create_compute_pipeline(&self, desc: &ComputePipelineDesc) -> Result<PipelineHandle> {
        self.inner.create_compute_pipeline(desc)
    }

    fn create_render_pass(&self, desc: &RenderPassDesc) -> Result<RenderPassHandle> {
        self.inner.create_render_pass(desc)
    }

    fn create_framebuffer(&self, desc: &FramebufferDesc) -> Result<FramebufferHandle> {
        self.inner.create_framebuffer(desc)
    }

    fn submit_commands(
        &self,
        queue: QueueType,
        cmds: &[CommandBuffer],
    ) -> Result<FenceHandle> {
        self.inner.submit_commands(queue, cmds)
    }

    fn wait_idle(&self) -> Result<()> {
        self.inner.wait_idle()
    }

    fn destroy_buffer(&self, handle: BufferHandle) {
        self.inner.destroy_buffer(handle);
    }

    fn destroy_texture(&self, handle: TextureHandle) {
        self.inner.destroy_texture(handle);
    }

    fn destroy_pipeline(&self, handle: PipelineHandle) {
        self.inner.destroy_pipeline(handle);
    }

    fn get_capabilities(&self) -> &DeviceCapabilities {
        self.inner.get_capabilities()
    }

    fn map_buffer(&self, handle: BufferHandle) -> Result<*mut u8> {
        self.inner.map_buffer(handle)
    }

    fn unmap_buffer(&self, handle: BufferHandle) {
        self.inner.unmap_buffer(handle);
    }
}
