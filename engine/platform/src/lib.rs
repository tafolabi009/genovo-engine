// =============================================================================
// Genovo Engine - Platform Module
// =============================================================================
//
// The `genovo-platform` crate provides a unified abstraction over all
// supported operating systems and hardware platforms. It owns the OS event
// loop, window management, input routing, clipboard, and system queries.
//
// Supported targets (in priority order):
//   - Windows  (Desktop, primary development target)
//   - macOS    (Desktop)
//   - Linux    (Desktop, X11 + Wayland)
//   - iOS      (Mobile)
//   - Android  (Mobile)
//   - Xbox     (Console, requires GDK)
//   - PlayStation (Console, requires PS SDK under NDA)

// -- Public modules -----------------------------------------------------------

pub mod clipboard;
pub mod file_dialog;
pub mod input_action;
pub mod interface;
pub mod intrinsics;
pub mod platform_factory;

// -- Shared desktop backend (winit) -------------------------------------------

#[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
pub mod winit_backend;

// -- Platform backend modules (conditionally compiled) ------------------------

#[cfg(target_os = "windows")]
pub mod windows;

#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "linux")]
pub mod linux;

#[cfg(target_os = "ios")]
pub mod ios;

#[cfg(target_os = "android")]
pub mod android;

// Console platforms are behind feature flags since they require proprietary SDKs.
#[cfg(feature = "xbox")]
pub mod xbox;

#[cfg(feature = "playstation")]
pub mod playstation;

// -- Re-exports for convenience -----------------------------------------------

pub use input_action::{
    ActionState, GestureRecognizer, GestureType, InputActionDef, InputActionMap, InputBinding,
    InputContext, InputPlayback, InputRecorder, InputSnapshot, MouseAxisType, RebindingState,
    SequenceTracker, SwipeDirection, Vec2,
};
pub use interface::events::PlatformEvent;
pub use interface::input::{
    GamepadAxis, GamepadButton, InputAction, InputManager, InputState, KeyCode, MouseButton,
};
pub use interface::{
    CursorType, DisplayInfo, Platform, PlatformError, RawWindowHandle, RenderBackend, Result,
    SystemInfo, WindowDesc, WindowHandle,
};
pub use intrinsics::{
    CpuFeatures, MemoryInfo, PrecisionTimer, SystemProfile, detect_cpu_features,
    detect_memory_info, memory_fence, pause, rdtsc, rdtscp,
};
pub use platform_factory::create_platform;
pub use clipboard::{
    Clipboard, ClipboardChangeEvent, ClipboardContent, ClipboardError, ClipboardFormat,
    ClipboardHistory, ClipboardImage, ClipboardResult, MacOsClipboard, Win32Clipboard,
    X11Clipboard, create_clipboard,
};
pub use file_dialog::{
    DialogType, DirectoryMemory, FileDialogBuilder, FileDialogError, FileDialogResponse,
    FileFilter, RecentFileEntry, RecentFiles, SoftwareFileDialog,
};

// -----------------------------------------------------------------------------
// Platform Type enum
// -----------------------------------------------------------------------------

/// Enumerates all platform targets the engine can be compiled for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlatformType {
    Windows,
    MacOS,
    Linux,
    IOS,
    Android,
    Xbox,
    PlayStation,
    Unknown,
}

impl PlatformType {
    /// Detect the current platform at compile time.
    pub const fn current() -> Self {
        #[cfg(target_os = "windows")]
        {
            PlatformType::Windows
        }
        #[cfg(target_os = "macos")]
        {
            PlatformType::MacOS
        }
        #[cfg(target_os = "linux")]
        {
            PlatformType::Linux
        }
        #[cfg(target_os = "ios")]
        {
            PlatformType::IOS
        }
        #[cfg(target_os = "android")]
        {
            PlatformType::Android
        }
        #[cfg(not(any(
            target_os = "windows",
            target_os = "macos",
            target_os = "linux",
            target_os = "ios",
            target_os = "android"
        )))]
        {
            PlatformType::Unknown
        }
    }

    /// Returns `true` if this is a desktop platform.
    pub const fn is_desktop(&self) -> bool {
        matches!(self, PlatformType::Windows | PlatformType::MacOS | PlatformType::Linux)
    }

    /// Returns `true` if this is a mobile platform.
    pub const fn is_mobile(&self) -> bool {
        matches!(self, PlatformType::IOS | PlatformType::Android)
    }

    /// Returns `true` if this is a console platform.
    pub const fn is_console(&self) -> bool {
        matches!(self, PlatformType::Xbox | PlatformType::PlayStation)
    }

    /// Returns the preferred rendering backend for this platform.
    pub const fn preferred_render_backend(&self) -> interface::RenderBackend {
        match self {
            PlatformType::Windows => interface::RenderBackend::DirectX12,
            PlatformType::MacOS => interface::RenderBackend::Metal,
            PlatformType::Linux => interface::RenderBackend::Vulkan,
            PlatformType::IOS => interface::RenderBackend::Metal,
            PlatformType::Android => interface::RenderBackend::Vulkan,
            PlatformType::Xbox => interface::RenderBackend::DirectX12,
            PlatformType::PlayStation => interface::RenderBackend::Vulkan, // GNM abstracted via Vulkan layer
            PlatformType::Unknown => interface::RenderBackend::Vulkan,
        }
    }
}

impl std::fmt::Display for PlatformType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlatformType::Windows => write!(f, "Windows"),
            PlatformType::MacOS => write!(f, "macOS"),
            PlatformType::Linux => write!(f, "Linux"),
            PlatformType::IOS => write!(f, "iOS"),
            PlatformType::Android => write!(f, "Android"),
            PlatformType::Xbox => write!(f, "Xbox"),
            PlatformType::PlayStation => write!(f, "PlayStation"),
            PlatformType::Unknown => write!(f, "Unknown"),
        }
    }
}

// -----------------------------------------------------------------------------
// Module-level constants
// -----------------------------------------------------------------------------

/// Maximum number of simultaneously connected gamepads the engine supports.
pub const MAX_GAMEPADS: u32 = 8;

/// Maximum number of simultaneous touch points tracked.
pub const MAX_TOUCH_POINTS: u32 = 10;

/// Maximum number of windows that can be open at once.
pub const MAX_WINDOWS: u32 = 16;
