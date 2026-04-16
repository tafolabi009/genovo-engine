// engine/render/src/vulkan/mod.rs
//
// Vulkan backend implementation. Delegates to the wgpu backend configured
// with `wgpu::Backends::VULKAN`, so that wgpu handles all Vulkan API calls
// (instance creation, device selection, memory allocation, command encoding,
// swapchain management) under the hood.

use std::sync::Arc;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::interface::command_buffer::CommandBuffer;
use crate::interface::device::{DeviceCapabilities, QueueType, RenderDevice, Result};
use crate::interface::pipeline::{ComputePipelineDesc, GraphicsPipelineDesc};
use crate::interface::resource::{
    BufferDesc, BufferHandle, FramebufferDesc, FramebufferHandle, FenceHandle, PipelineHandle,
    RenderPassDesc, RenderPassHandle, SamplerDesc, SamplerHandle, ShaderDesc, ShaderHandle,
    TextureDesc, TextureHandle,
};
use crate::wgpu_backend::{WgpuDevice, WgpuSurface};
use crate::RenderError;

// ---------------------------------------------------------------------------
// VulkanInstance
// ---------------------------------------------------------------------------

/// Wraps a wgpu instance configured exclusively for the Vulkan backend.
///
/// This provides the entry point for adapter enumeration and surface creation
/// on Vulkan-capable platforms.
pub struct VulkanInstance {
    /// The wgpu instance restricted to Vulkan.
    instance: wgpu::Instance,
}

impl VulkanInstance {
    /// Create a new Vulkan-only wgpu instance with the given application name.
    pub fn new(_app_name: &str) -> std::result::Result<Self, RenderError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        log::info!("VulkanInstance created (wgpu Vulkan backend)");
        Ok(Self { instance })
    }

    /// Access the underlying wgpu instance.
    pub fn raw_instance(&self) -> &wgpu::Instance {
        &self.instance
    }
}

// ---------------------------------------------------------------------------
// VulkanSwapchain
// ---------------------------------------------------------------------------

/// Manages a Vulkan presentation surface through the wgpu abstraction.
///
/// Wraps [`WgpuSurface`] so that callers get swapchain-like semantics
/// (acquire / present / resize) while wgpu handles the actual
/// `VkSwapchainKHR` lifecycle.
pub struct VulkanSwapchain {
    /// The wgpu surface wrapper.
    surface: WgpuSurface,
}

impl VulkanSwapchain {
    /// Create a new swapchain for the given surface.
    pub fn new(
        _instance: &VulkanInstance,
        device: &VulkanDevice,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
    ) -> std::result::Result<Self, RenderError> {
        let wgpu_surface = WgpuSurface::new(&device.inner, surface, width, height);
        log::info!("VulkanSwapchain created ({}x{})", width, height);
        Ok(Self {
            surface: wgpu_surface,
        })
    }

    /// Create a swapchain from a window handle.
    pub fn new_from_window<W>(
        instance: &VulkanInstance,
        device: &VulkanDevice,
        window: Arc<W>,
        width: u32,
        height: u32,
    ) -> std::result::Result<Self, RenderError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        let raw_surface = instance
            .instance
            .create_surface(window)
            .map_err(|e| RenderError::ResourceCreation(format!("Failed to create Vulkan surface: {e}")))?;

        Self::new(instance, device, raw_surface, width, height)
    }

    /// Acquire the next swapchain image for rendering.
    pub fn acquire_next_image(&self) -> std::result::Result<wgpu::SurfaceTexture, RenderError> {
        self.surface.get_current_texture()
    }

    /// Present the rendered frame. The `SurfaceTexture` is consumed.
    pub fn present(&self, surface_texture: wgpu::SurfaceTexture) -> std::result::Result<(), RenderError> {
        surface_texture.present();
        Ok(())
    }

    /// Recreate the swapchain after a window resize.
    pub fn resize(&mut self, device: &VulkanDevice, width: u32, height: u32) -> std::result::Result<(), RenderError> {
        self.surface.resize(&device.inner, width, height);
        Ok(())
    }

    /// The surface texture format (engine representation).
    pub fn format(&self) -> crate::interface::resource::TextureFormat {
        self.surface.engine_format()
    }

    /// Current configured width.
    pub fn width(&self) -> u32 {
        self.surface.width()
    }

    /// Current configured height.
    pub fn height(&self) -> u32 {
        self.surface.height()
    }
}

// ---------------------------------------------------------------------------
// VulkanDevice
// ---------------------------------------------------------------------------

/// Vulkan implementation of [`RenderDevice`].
///
/// This is a thin wrapper around [`WgpuDevice`] configured to use only the
/// Vulkan backend. All resource creation, command submission, and
/// synchronisation is delegated to the inner wgpu device.
pub struct VulkanDevice {
    /// The wgpu device restricted to the Vulkan backend.
    inner: WgpuDevice,
}

impl VulkanDevice {
    /// Create a Vulkan device from a `VulkanInstance`.
    ///
    /// The device is created headless (without a surface). Use
    /// [`VulkanSwapchain`] to set up presentation.
    pub fn new(_instance: &VulkanInstance) -> std::result::Result<Self, RenderError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let inner = WgpuDevice::new_with_instance(instance)?;
        log::info!(
            "VulkanDevice created: {}",
            inner.get_capabilities().device_name
        );
        Ok(Self { inner })
    }

    /// Access the underlying `WgpuDevice`.
    pub fn wgpu_device(&self) -> &WgpuDevice {
        &self.inner
    }
}

impl RenderDevice for VulkanDevice {
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
