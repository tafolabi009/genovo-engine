// =============================================================================
// Genovo Engine - Linux Platform Backend
// =============================================================================
//
// Linux platform implementation using the shared winit backend for window
// management and input. Supports both X11 and Wayland (winit handles the
// display server selection automatically). Uses Vulkan as the preferred
// rendering backend.
//
// For Linux-specific extensions (evdev gamepads, X11 selections, Wayland
// protocols, Xrandr/wl_output queries, etc.) that go beyond what winit
// provides, add platform-specific code below.

#![cfg(target_os = "linux")]

use crate::interface::events::PlatformEvent;
use crate::interface::{
    CursorType, DisplayInfo, Platform, RawWindowHandle, RenderBackend, Result, SystemInfo,
    WindowDesc, WindowHandle,
};
use crate::winit_backend::WinitPlatform;

// -----------------------------------------------------------------------------
// Display Server Selection
// -----------------------------------------------------------------------------

/// Which Linux display server protocol is in use.
///
/// winit auto-detects this, but we expose it for engine-level queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinuxDisplayServer {
    /// X Window System (X11 / Xlib / Xcb).
    X11,
    /// Wayland compositor protocol.
    Wayland,
}

impl LinuxDisplayServer {
    /// Auto-detect the running display server from environment variables.
    pub fn detect() -> Self {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            LinuxDisplayServer::Wayland
        } else {
            LinuxDisplayServer::X11
        }
    }
}

// -----------------------------------------------------------------------------
// Linux Platform
// -----------------------------------------------------------------------------

/// Linux platform implementation.
///
/// Wraps the shared `WinitPlatform` backend and configures it for Linux
/// with Vulkan as the preferred rendering backend. winit handles the
/// X11/Wayland display server selection automatically.
///
/// # Linux-specific extensions
///
/// Features that go beyond winit's capabilities, planned for future integration:
///
/// ## evdev / libinput Gamepad Support
/// Linux gamepad support uses the evdev kernel interface (`/dev/input/eventN`).
/// The plan is to enumerate input devices via `/dev/input/event*`, filter for
/// those with `EV_ABS` (analog axes) and `EV_KEY` (buttons) capabilities using
/// `ioctl(EVIOCGBIT)`, and read events in a non-blocking loop. The `evdev`
/// crate provides safe Rust bindings. libinput is an alternative that handles
/// device hotplug and provides higher-level gesture recognition.
///
/// Device hotplug detection uses `libudev` (or `inotify` on `/dev/input/`)
/// to detect gamepads being connected or disconnected at runtime.
///
/// ## Clipboard (X11 / Wayland)
/// winit provides basic clipboard support. For full clipboard compatibility:
///   - X11: use `XSetSelectionOwner` / `XConvertSelection` with `CLIPBOARD`
///     and `PRIMARY` atoms, handling `SelectionRequest` / `SelectionNotify`
///     events for both copy and paste.
///   - Wayland: use `wl_data_device_manager` / `wl_data_source` /
///     `wl_data_offer` protocols for clipboard read/write.
/// The `arboard` or `clipboard` crate abstracts both protocols.
///
/// ## Display Enumeration (Xrandr / wl_output)
///   - X11: `XRRGetScreenResourcesCurrent` + `XRRGetOutputInfo` for monitor
///     names, resolutions, refresh rates, and positions.
///   - Wayland: listen for `wl_output` global events and `wl_output.geometry`
///     / `wl_output.mode` events for display properties.
/// winit's `MonitorHandle` covers most of this already.
///
/// ## System Information
///   - CPU: parse `/proc/cpuinfo` for "model name" and count logical cores
///   - RAM: parse `/proc/meminfo` for "MemTotal"
///   - GPU: parse `/sys/class/drm/card*/device/` for PCI vendor/device IDs,
///          or query Vulkan `vkGetPhysicalDeviceProperties` for device name
///          and `vkGetPhysicalDeviceMemoryProperties` for VRAM heap sizes
///   - OS: parse `/etc/os-release` for distribution name and version
pub struct LinuxPlatform {
    inner: WinitPlatform,
    /// The detected display server protocol.
    pub display_server: LinuxDisplayServer,
}

impl LinuxPlatform {
    /// Create a new Linux platform instance.
    ///
    /// Auto-detects X11 vs Wayland and initializes the winit event loop
    /// with Vulkan as the preferred render backend.
    ///
    /// Additional Linux-specific setup:
    ///
    /// - evdev device enumeration: scan `/dev/input/event*` for gamepad
    ///   devices and set up non-blocking reads. Use `libudev` or `inotify`
    ///   for hotplug detection.
    ///
    /// - Native clipboard: the `arboard` crate or direct X11/Wayland protocol
    ///   usage for clipboard operations. winit provides basic clipboard support
    ///   that is sufficient for most use cases.
    pub fn new() -> Self {
        let display_server = LinuxDisplayServer::detect();
        log::info!(
            "Initializing Linux platform (winit + Vulkan, display_server={:?})",
            display_server
        );

        Self {
            inner: WinitPlatform::new(RenderBackend::Vulkan),
            display_server,
        }
    }
}

impl Default for LinuxPlatform {
    fn default() -> Self {
        Self::new()
    }
}

impl Platform for LinuxPlatform {
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
        info.os_name = String::from("Linux");
        // Detailed system info can be obtained by parsing:
        //   - /proc/cpuinfo: "model name" field for CPU name, count entries
        //     for logical core count
        //   - /proc/meminfo: "MemTotal" field (in kB, convert to MB)
        //   - /sys/class/drm/card0/device/: PCI IDs for GPU identification,
        //     or VkPhysicalDeviceProperties.deviceName from Vulkan
        //   - /etc/os-release: PRETTY_NAME for distribution name and version
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
