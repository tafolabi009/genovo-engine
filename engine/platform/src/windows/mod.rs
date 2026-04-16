// =============================================================================
// Genovo Engine - Windows Platform Backend
// =============================================================================
//
// Windows platform implementation using the shared winit backend for window
// management and input. Uses DirectX 12 as the preferred rendering backend.
//
// For Windows-specific extensions (native Win32 APIs, XInput, Raw Input, DPI
// awareness, etc.) that go beyond what winit provides, add platform-specific
// code below. The core window management is handled by `WinitPlatform`.

#![cfg(target_os = "windows")]

use crate::interface::events::PlatformEvent;
use crate::interface::{
    CursorType, DisplayInfo, Platform, RawWindowHandle, RenderBackend, Result, SystemInfo,
    WindowDesc, WindowHandle,
};
use crate::winit_backend::WinitPlatform;

// -----------------------------------------------------------------------------
// Windows Platform
// -----------------------------------------------------------------------------

/// Windows platform implementation.
///
/// Wraps the shared `WinitPlatform` backend and configures it for Windows
/// with DirectX 12 as the preferred rendering backend.
///
/// # Windows-specific extensions
///
/// Features that go beyond winit's capabilities, planned for future integration:
///
/// ## XInput Gamepad Support
/// XInput provides higher-quality Xbox controller integration than generic
/// gamepad APIs. The plan is to use the `XInputGetState` function from
/// `xinput1_4.dll` for lower-latency polling, trigger rumble via
/// `XInputSetState`, and battery level queries via `XInputGetBatteryInformation`.
/// The newer GameInput API (Windows.Gaming.Input) is preferred on Windows 10+
/// for broader device support including DualSense.
///
/// ## Raw Input for High-Precision Mouse
/// `RegisterRawInputDevices` with `RIDEV_NOLEGACY` provides sub-pixel mouse
/// deltas without acceleration, essential for first-person camera control.
/// winit surfaces `DeviceEvent::MouseMotion` which uses Raw Input internally,
/// so this is already partially covered.
///
/// ## DPI Awareness
/// winit calls `SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)`
/// during initialization, which is the recommended approach for Windows 10+.
/// This ensures the engine receives physical pixel coordinates and correct
/// scale factor notifications via `WM_DPICHANGED`. The engine scales its UI
/// layer using `get_window_scale_factor()`. No additional Win32 calls are
/// needed beyond what winit provides.
///
/// ## COM Initialization
/// `CoInitializeEx(NULL, COINIT_MULTITHREADED)` is required before using
/// multimedia APIs (WASAPI audio, Media Foundation, DXGI adapter enumeration).
/// This should be called once during platform initialization on the main thread.
///
/// ## Hardware Enumeration
/// WMI (`Win32_Processor`, `Win32_VideoController`) or DXGI
/// (`IDXGIFactory::EnumAdapters`) provide detailed CPU model, GPU name,
/// dedicated VRAM, and driver version information.
pub struct WindowsPlatform {
    inner: WinitPlatform,
}

impl WindowsPlatform {
    /// Create a new Windows platform instance.
    ///
    /// Initializes the winit event loop with DirectX 12 as the preferred
    /// render backend.
    ///
    /// winit automatically sets Per-Monitor DPI Awareness V2 on Windows 10+,
    /// ensuring correct scale factor reporting and DPI change events.
    ///
    /// COM initialization (`CoInitializeEx`) should be performed here before
    /// any multimedia or DXGI calls. XInput and GameInput controller polling
    /// can be set up after the event loop is created.
    pub fn new() -> Self {
        log::info!("Initializing Windows platform (winit + DirectX12)");

        // winit handles SetProcessDpiAwarenessContext internally, so DPI
        // awareness is already configured by the time we create the event loop.
        //
        // COM initialization for multimedia support:
        //   CoInitializeEx(null, COINIT_MULTITHREADED) via the `windows` crate.
        //
        // XInput / GameInput controller setup:
        //   Load xinput1_4.dll dynamically or use Windows.Gaming.Input COM API.
        //   Poll connected controllers each frame in poll_events().
        //
        // Raw Input for high-precision mouse:
        //   Already surfaced by winit via DeviceEvent::MouseMotion; additional
        //   RegisterRawInputDevices calls are only needed for HID gamepads.

        Self {
            inner: WinitPlatform::new(RenderBackend::DirectX12),
        }
    }
}

impl Default for WindowsPlatform {
    fn default() -> Self {
        Self::new()
    }
}

impl Platform for WindowsPlatform {
    fn create_window(&self, desc: &WindowDesc) -> Result<WindowHandle> {
        self.inner.create_window(desc)
    }

    fn destroy_window(&self, handle: WindowHandle) {
        self.inner.destroy_window(handle);
    }

    fn poll_events(&mut self) -> Vec<PlatformEvent> {
        self.inner.poll_events()
    }

    fn get_render_backend(&self) -> RenderBackend {
        self.inner.get_render_backend()
    }

    fn get_display_info(&self) -> Vec<DisplayInfo> {
        self.inner.get_display_info()
    }

    fn set_cursor(&self, cursor: CursorType) {
        self.inner.set_cursor(cursor);
    }

    fn set_clipboard(&self, text: &str) {
        self.inner.set_clipboard(text);
    }

    fn get_clipboard(&self) -> Option<String> {
        self.inner.get_clipboard()
    }

    fn get_system_info(&self) -> SystemInfo {
        let mut info = self.inner.get_system_info();
        info.os_name = String::from("Windows");
        // Detailed hardware info can be obtained via:
        //   - CPU: WMI Win32_Processor.Name or CPUID instruction
        //   - GPU: IDXGIAdapter::GetDesc() for name and dedicated VRAM
        //   - RAM: GlobalMemoryStatusEx() for total physical memory
        //   - OS version: RtlGetVersion() or GetVersionEx()
        info
    }

    fn get_raw_window_handle(&self, handle: WindowHandle) -> RawWindowHandle {
        self.inner.get_raw_window_handle(handle)
    }

    fn set_window_title(&self, handle: WindowHandle, title: &str) {
        self.inner.set_window_title(handle, title);
    }

    fn set_window_size(&self, handle: WindowHandle, width: u32, height: u32) {
        self.inner.set_window_size(handle, width, height);
    }

    fn set_window_position(&self, handle: WindowHandle, x: i32, y: i32) {
        self.inner.set_window_position(handle, x, y);
    }

    fn set_fullscreen(&self, handle: WindowHandle, fullscreen: bool) {
        self.inner.set_fullscreen(handle, fullscreen);
    }

    fn set_cursor_locked(&self, handle: WindowHandle, locked: bool) {
        self.inner.set_cursor_locked(handle, locked);
    }

    fn set_cursor_visible(&self, visible: bool) {
        self.inner.set_cursor_visible(visible);
    }

    fn get_window_scale_factor(&self, handle: WindowHandle) -> f64 {
        self.inner.get_window_scale_factor(handle)
    }
}
