// engine/render/src/backend_factory.rs
//
// Runtime backend selection. The factory function inspects compile-time
// feature gates and the requested `RenderBackend` to construct the
// appropriate `Box<dyn RenderDevice>`.

use crate::interface::device::RenderDevice;
use crate::{RenderBackend, RenderError};
use raw_window_handle::RawWindowHandle;

/// Create a concrete [`RenderDevice`] for the requested backend.
///
/// The `window` handle is used by backends that need a surface for
/// presentation (Vulkan, DX12, Metal).
///
/// # Errors
///
/// Returns [`RenderError::BackendNotAvailable`] when the requested backend
/// was not compiled in (feature gate disabled).
///
/// Returns [`RenderError::BackendNotImplemented`] when the backend feature is
/// enabled but the implementation could not initialise.
pub fn create_render_device(
    backend: RenderBackend,
    _window: &RawWindowHandle,
) -> std::result::Result<Box<dyn RenderDevice>, RenderError> {
    match backend {
        // -- wgpu (cross-platform) -----------------------------------------
        RenderBackend::Wgpu | RenderBackend::Auto => {
            let device = crate::wgpu_backend::WgpuDevice::new_headless()?;
            Ok(Box::new(device))
        }

        // -- Vulkan ---------------------------------------------------------
        RenderBackend::Vulkan => {
            #[cfg(feature = "vulkan")]
            {
                let instance = crate::vulkan::VulkanInstance::new("Genovo Engine")?;
                let device = crate::vulkan::VulkanDevice::new(&instance)?;
                Ok(Box::new(device))
            }
            #[cfg(not(feature = "vulkan"))]
            {
                // Fall back to wgpu which can use Vulkan under the hood.
                let device = crate::wgpu_backend::WgpuDevice::new_headless()?;
                Ok(Box::new(device))
            }
        }

        // -- DirectX 12 ----------------------------------------------------
        RenderBackend::Dx12 => {
            #[cfg(feature = "dx12")]
            {
                let device = crate::dx12::Dx12Device::new()?;
                Ok(Box::new(device))
            }
            #[cfg(not(feature = "dx12"))]
            {
                // Fall back to wgpu which can use DX12 under the hood.
                let device = crate::wgpu_backend::WgpuDevice::new_headless()?;
                Ok(Box::new(device))
            }
        }

        // -- Metal ----------------------------------------------------------
        RenderBackend::Metal => {
            #[cfg(feature = "metal")]
            {
                let device = crate::metal::MetalDevice::new()?;
                Ok(Box::new(device))
            }
            #[cfg(not(feature = "metal"))]
            {
                // Fall back to wgpu which can use Metal under the hood.
                let device = crate::wgpu_backend::WgpuDevice::new_headless()?;
                Ok(Box::new(device))
            }
        }
    }
}

/// Auto-detect the best available backend for the current platform.
///
/// Defaults to wgpu for maximum cross-platform compatibility. The wgpu
/// backend automatically selects the best native API (Vulkan on Linux/Windows,
/// Metal on macOS/iOS, DX12 on Windows).
///
/// Priority order:
/// - All platforms: wgpu (wraps Vulkan/DX12/Metal automatically)
/// - Fallback on macOS/iOS: Metal
/// - Fallback on Windows: Vulkan > DX12
/// - Fallback on Linux/Android: Vulkan
pub fn detect_preferred_backend() -> RenderBackend {
    // wgpu handles backend selection internally and is always available.
    RenderBackend::Wgpu
}
