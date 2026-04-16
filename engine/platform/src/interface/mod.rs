// =============================================================================
// Genovo Engine - Platform Abstraction Layer
// =============================================================================
//
// Defines the core `Platform` trait and associated types that all platform
// backends must implement. The engine interacts with the OS exclusively
// through this interface.

pub mod events;
pub mod input;

use std::fmt;

use events::PlatformEvent;

// -----------------------------------------------------------------------------
// Error types
// -----------------------------------------------------------------------------

/// Errors produced by platform operations.
#[derive(Debug, thiserror::Error)]
pub enum PlatformError {
    #[error("Failed to create window: {0}")]
    WindowCreationFailed(String),

    #[error("Invalid window handle: {0}")]
    InvalidWindowHandle(u64),

    #[error("Clipboard operation failed: {0}")]
    ClipboardError(String),

    #[error("Display enumeration failed: {0}")]
    DisplayError(String),

    #[error("Platform not supported: {0}")]
    Unsupported(String),

    #[error("Platform internal error: {0}")]
    Internal(String),
}

/// Convenience result alias for platform operations.
pub type Result<T> = std::result::Result<T, PlatformError>;

// -----------------------------------------------------------------------------
// Window Handle
// -----------------------------------------------------------------------------

/// Opaque handle to a platform window.
///
/// The inner `u64` is an engine-assigned identifier; it is *not* the native
/// OS handle (use [`Platform::get_raw_window_handle`] for that).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowHandle(pub u64);

impl fmt::Display for WindowHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WindowHandle({})", self.0)
    }
}

// -----------------------------------------------------------------------------
// Window Descriptor
// -----------------------------------------------------------------------------

/// Describes the desired properties of a new window.
#[derive(Debug, Clone)]
pub struct WindowDesc {
    /// Title text shown in the window's title bar.
    pub title: String,
    /// Initial width in logical pixels.
    pub width: u32,
    /// Initial height in logical pixels.
    pub height: u32,
    /// Whether the window should start in fullscreen mode.
    pub fullscreen: bool,
    /// Whether vertical sync should be enabled for this window's swap chain.
    pub vsync: bool,
    /// Whether the window should be user-resizable.
    pub resizable: bool,
    /// Whether the window should have OS-level decorations (title bar, borders).
    pub decorated: bool,
    /// Whether the window should be transparent.
    pub transparent: bool,
    /// Minimum window size in logical pixels, if any.
    pub min_size: Option<(u32, u32)>,
    /// Maximum window size in logical pixels, if any.
    pub max_size: Option<(u32, u32)>,
}

impl Default for WindowDesc {
    fn default() -> Self {
        Self {
            title: String::from("Genovo Engine"),
            width: 1280,
            height: 720,
            fullscreen: false,
            vsync: true,
            resizable: true,
            decorated: true,
            transparent: false,
            min_size: None,
            max_size: None,
        }
    }
}

// -----------------------------------------------------------------------------
// Display Info
// -----------------------------------------------------------------------------

/// Information about a connected display / monitor.
#[derive(Debug, Clone)]
pub struct DisplayInfo {
    /// Human-readable display name (may be empty on some platforms).
    pub name: String,
    /// Native resolution width in physical pixels.
    pub width: u32,
    /// Native resolution height in physical pixels.
    pub height: u32,
    /// Refresh rate in Hz.
    pub refresh_rate_hz: u32,
    /// DPI scale factor (1.0 = 96 DPI baseline on Windows).
    pub scale_factor: f64,
    /// Whether this is the primary / main display.
    pub is_primary: bool,
    /// Position of the display in the virtual desktop coordinate space.
    pub position: (i32, i32),
}

// -----------------------------------------------------------------------------
// System Info
// -----------------------------------------------------------------------------

/// Snapshot of system hardware and OS information.
#[derive(Debug, Clone)]
pub struct SystemInfo {
    /// CPU model string (e.g. "AMD Ryzen 9 7950X").
    pub cpu_name: String,
    /// Number of logical CPU cores.
    pub cpu_cores: u32,
    /// GPU model string (e.g. "NVIDIA GeForce RTX 4090").
    pub gpu_name: String,
    /// Dedicated GPU VRAM in megabytes.
    pub gpu_vram_mb: u64,
    /// Total system RAM in megabytes.
    pub total_ram_mb: u64,
    /// OS name and version string.
    pub os_name: String,
    /// OS version string.
    pub os_version: String,
}

// -----------------------------------------------------------------------------
// Render Backend
// -----------------------------------------------------------------------------

/// Identifies the rendering API available on the current platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderBackend {
    Vulkan,
    DirectX12,
    Metal,
    OpenGLES,
    WebGPU,
}

// -----------------------------------------------------------------------------
// Cursor Type
// -----------------------------------------------------------------------------

/// Standard OS cursor shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CursorType {
    Default,
    Arrow,
    IBeam,
    Crosshair,
    Hand,
    ResizeHorizontal,
    ResizeVertical,
    ResizeNESW,
    ResizeNWSE,
    Move,
    NotAllowed,
    Wait,
    Progress,
    Help,
    /// Hide the cursor entirely.
    Hidden,
}

// -----------------------------------------------------------------------------
// Raw Window Handle (re-export wrapper)
// -----------------------------------------------------------------------------

/// Wrapper around the `raw-window-handle` types for interop with rendering
/// backends.
#[derive(Debug, Clone)]
pub enum RawWindowHandle {
    /// Win32 HWND.
    Windows {
        hwnd: *mut std::ffi::c_void,
        hinstance: *mut std::ffi::c_void,
    },
    /// macOS NSView.
    MacOS {
        ns_view: *mut std::ffi::c_void,
        ns_window: *mut std::ffi::c_void,
    },
    /// X11 Window.
    X11 {
        window: u64,
        display: *mut std::ffi::c_void,
    },
    /// Wayland surface.
    Wayland {
        surface: *mut std::ffi::c_void,
        display: *mut std::ffi::c_void,
    },
    /// Android ANativeWindow.
    Android {
        a_native_window: *mut std::ffi::c_void,
    },
    /// iOS UIView.
    IOS {
        ui_view: *mut std::ffi::c_void,
    },
    /// Returned when the platform cannot provide a native window handle
    /// (e.g., console backends without SDK access).
    Unavailable,
}

// Safety: raw pointers in RawWindowHandle are opaque handles passed to the
// rendering backend; the platform implementation guarantees they remain valid
// for the lifetime of the window.
unsafe impl Send for RawWindowHandle {}
unsafe impl Sync for RawWindowHandle {}

// -----------------------------------------------------------------------------
// Platform Trait
// -----------------------------------------------------------------------------

/// Core abstraction over operating-system services required by the engine.
///
/// Each target platform provides a concrete implementation of this trait.
/// The engine obtains an instance via [`crate::create_platform`] and
/// interacts with the OS exclusively through these methods.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync`. Window creation and event polling
/// happen on the main thread; other methods may be called from any thread
/// unless documented otherwise on a specific platform.
pub trait Platform: Send + Sync {
    /// Create a new OS window according to the given descriptor.
    ///
    /// Returns a [`WindowHandle`] that can be used to reference this window
    /// in subsequent calls.
    fn create_window(&self, desc: &WindowDesc) -> Result<WindowHandle>;

    /// Destroy a previously created window.
    ///
    /// The handle becomes invalid after this call. Passing an invalid handle
    /// is a no-op (logged at warn level).
    fn destroy_window(&self, handle: WindowHandle);

    /// Poll the OS event queue and return all pending events.
    ///
    /// This must be called on the main thread once per frame.
    fn poll_events(&mut self) -> Vec<PlatformEvent>;

    /// Returns the preferred rendering backend for this platform.
    fn get_render_backend(&self) -> RenderBackend;

    /// Enumerate all connected displays.
    fn get_display_info(&self) -> Vec<DisplayInfo>;

    /// Change the OS cursor shape.
    fn set_cursor(&self, cursor: CursorType);

    /// Copy text to the OS clipboard.
    fn set_clipboard(&self, text: &str);

    /// Read text from the OS clipboard, if available.
    fn get_clipboard(&self) -> Option<String>;

    /// Query system hardware and OS information.
    fn get_system_info(&self) -> SystemInfo;

    /// Obtain the native window handle for rendering backend initialization.
    fn get_raw_window_handle(&self, handle: WindowHandle) -> RawWindowHandle;

    // -- Optional methods with default implementations ------------------------

    /// Set the window title at runtime.
    ///
    /// Desktop backends delegate to winit. Mobile and console backends ignore
    /// this call since those platforms do not display window title bars.
    fn set_window_title(&self, _handle: WindowHandle, _title: &str) {
        // No-op default for platforms without title bars (mobile, console).
    }

    /// Resize the window programmatically.
    ///
    /// Desktop backends delegate to winit. Mobile and console backends ignore
    /// this call since those platforms use fixed full-screen surfaces.
    fn set_window_size(&self, _handle: WindowHandle, _width: u32, _height: u32) {
        // No-op default for platforms with fixed-size surfaces.
    }

    /// Set the window position on screen.
    ///
    /// Desktop backends delegate to winit. Mobile and console backends ignore
    /// this call since window positioning is not applicable.
    fn set_window_position(&self, _handle: WindowHandle, _x: i32, _y: i32) {
        // No-op default for platforms without movable windows.
    }

    /// Toggle fullscreen mode for the given window.
    ///
    /// Desktop backends delegate to winit. Mobile and console platforms are
    /// always full-screen, so this is a no-op.
    fn set_fullscreen(&self, _handle: WindowHandle, _fullscreen: bool) {
        // No-op default for platforms that are always full-screen.
    }

    /// Lock or unlock the mouse cursor to the window.
    ///
    /// Desktop backends delegate to winit. Touch-only and console platforms
    /// ignore this call.
    fn set_cursor_locked(&self, _handle: WindowHandle, _locked: bool) {
        // No-op default for platforms without a mouse cursor.
    }

    /// Show or hide the OS cursor.
    ///
    /// Desktop backends delegate to winit. Touch-only and console platforms
    /// ignore this call.
    fn set_cursor_visible(&self, _visible: bool) {
        // No-op default for platforms without a mouse cursor.
    }

    /// Get the DPI scale factor of the window.
    fn get_window_scale_factor(&self, _handle: WindowHandle) -> f64 {
        1.0
    }
}
