// =============================================================================
// Genovo Engine - macOS Platform Backend
// =============================================================================
//
// macOS platform implementation using the shared winit backend for window
// management and input. Uses Metal as the preferred rendering backend.
//
// For macOS-specific extensions (Cocoa/AppKit APIs, IOKit HID for gamepads,
// trackpad gestures, Retina display handling, etc.) that go beyond what winit
// provides, add platform-specific code below.

#![cfg(target_os = "macos")]

use crate::interface::events::PlatformEvent;
use crate::interface::{
    CursorType, DisplayInfo, Platform, RawWindowHandle, RenderBackend, Result, SystemInfo,
    WindowDesc, WindowHandle,
};
use crate::winit_backend::WinitPlatform;

// -----------------------------------------------------------------------------
// macOS Platform
// -----------------------------------------------------------------------------

/// macOS platform implementation.
///
/// Wraps the shared `WinitPlatform` backend and configures it for macOS
/// with Metal as the preferred rendering backend.
///
/// # macOS-specific extensions
///
/// Features that go beyond winit's capabilities, planned for future integration:
///
/// ## IOKit HID Manager for Gamepads
/// IOKit HID Manager (`IOHIDManagerCreate`, `IOHIDManagerSetDeviceMatching`)
/// provides native MFi and Bluetooth gamepad discovery on macOS. The plan is
/// to enumerate HID devices matching `kHIDUsage_GD_GamePad` and
/// `kHIDUsage_GD_Joystick` usage pages, register value-changed callbacks,
/// and map the HID elements (buttons, axes, hat switches) to the engine's
/// `GamepadButton`/`GamepadAxis` enums. This gives lower latency and better
/// compatibility than going through higher-level frameworks. The `Game
/// Controller` framework (`GCController`) is an alternative with simpler
/// API but slightly higher latency.
///
/// ## Trackpad Gesture Events
/// macOS trackpads support multi-touch gestures that winit does not fully
/// expose. The plan is to use `NSEvent` gesture recognizers (or the
/// `NSTouchBar` / `NSTouch` API) to capture:
///   - Pinch/zoom (`magnification` property on `NSEvent`)
///   - Rotation (`rotation` property on `NSEvent`)
///   - Smart zoom (double-tap with two fingers)
///   - Three/four finger swipes (via `NSEvent.swipeWithEvent:`)
///
/// These would be surfaced as custom `PlatformEvent` variants or through
/// a macOS-specific extension trait. winit does surface scroll events from
/// the trackpad, which covers basic two-finger scroll.
///
/// ## NSApplication Activation Policy
/// `NSApp.setActivationPolicy(.regular)` ensures the app appears in the
/// Dock and receives focus correctly. winit handles this internally during
/// event loop creation on macOS.
pub struct MacOsPlatform {
    inner: WinitPlatform,
}

impl MacOsPlatform {
    /// Create a new macOS platform instance.
    ///
    /// Initializes the winit event loop with Metal as the preferred
    /// render backend.
    ///
    /// winit handles NSApplication setup (activation policy, main menu)
    /// internally. Additional macOS-specific setup:
    ///
    /// - IOKit HID Manager for gamepad discovery: call `IOHIDManagerCreate`,
    ///   set device matching dictionaries for gamepad/joystick usage pages,
    ///   schedule on the current run loop, and register connect/disconnect
    ///   callbacks.
    ///
    /// - Trackpad gesture configuration: winit surfaces scroll and magnify
    ///   events. For rotation and swipe gestures, an `NSEvent` local monitor
    ///   (`NSEvent.addLocalMonitorForEvents(matching:handler:)`) can intercept
    ///   gesture events before they reach the responder chain.
    pub fn new() -> Self {
        log::info!("Initializing macOS platform (winit + Metal)");

        Self {
            inner: WinitPlatform::new(RenderBackend::Metal),
        }
    }
}

impl Default for MacOsPlatform {
    fn default() -> Self {
        Self::new()
    }
}

impl Platform for MacOsPlatform {
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
        info.os_name = String::from("macOS");
        // Detailed hardware info can be obtained via:
        //   - CPU: sysctl("machdep.cpu.brand_string") for model name,
        //          sysctl("hw.ncpu") for logical core count
        //   - GPU: MTLCreateSystemDefaultDevice().name for Metal GPU name,
        //          IOKit registry for VRAM (IORegistryEntrySearchCFProperty
        //          with "VRAM,totalMB" key)
        //   - RAM: sysctl("hw.memsize") for total physical memory in bytes
        //   - OS: NSProcessInfo.processInfo.operatingSystemVersion for
        //         major/minor/patch version numbers
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
