// =============================================================================
// Genovo Engine - Platform Events
// =============================================================================
//
// Platform-level events surfaced from native OS event loops to the engine.

use std::path::PathBuf;

use super::input::{GamepadAxis, GamepadButton, KeyCode, MouseButton};
use super::WindowHandle;

// -----------------------------------------------------------------------------
// Platform Event
// -----------------------------------------------------------------------------

/// Top-level event enum encompassing all platform-originating events.
///
/// Events are polled per-frame via [`Platform::poll_events`] and dispatched
/// to engine subsystems (input manager, window manager, application lifecycle).
#[derive(Debug, Clone, PartialEq)]
pub enum PlatformEvent {
    // -- Window events --------------------------------------------------------

    /// The window has been resized to new logical dimensions.
    WindowResize {
        handle: WindowHandle,
        width: u32,
        height: u32,
    },

    /// The window's close button was pressed or a close was requested by the OS.
    WindowClose {
        handle: WindowHandle,
    },

    /// The window gained or lost input focus.
    WindowFocus {
        handle: WindowHandle,
        focused: bool,
    },

    /// The window was moved to a new position on screen.
    WindowMoved {
        handle: WindowHandle,
        x: i32,
        y: i32,
    },

    /// The window's DPI/scale factor changed (e.g. moved between monitors).
    WindowScaleFactorChanged {
        handle: WindowHandle,
        scale_factor: f64,
    },

    /// The window was minimized or restored.
    WindowMinimized {
        handle: WindowHandle,
        minimized: bool,
    },

    /// The window was maximized or restored.
    WindowMaximized {
        handle: WindowHandle,
        maximized: bool,
    },

    // -- Keyboard events ------------------------------------------------------

    /// A keyboard key was pressed or released.
    KeyInput {
        handle: WindowHandle,
        key: KeyCode,
        pressed: bool,
        /// Whether this is a repeat event from the OS key-repeat mechanism.
        repeat: bool,
    },

    /// A character was produced by the keyboard (after IME / dead-key processing).
    CharInput {
        handle: WindowHandle,
        character: char,
    },

    // -- Mouse events ---------------------------------------------------------

    /// A mouse button was pressed or released.
    MouseInput {
        handle: WindowHandle,
        button: MouseButton,
        pressed: bool,
    },

    /// The mouse cursor moved within the window.
    MouseMove {
        handle: WindowHandle,
        /// Logical x-position relative to the window's client area.
        x: f64,
        /// Logical y-position relative to the window's client area.
        y: f64,
    },

    /// The mouse scroll wheel moved.
    MouseScroll {
        handle: WindowHandle,
        /// Horizontal scroll delta.
        delta_x: f64,
        /// Vertical scroll delta.
        delta_y: f64,
    },

    /// The mouse cursor entered or left the window's client area.
    MouseEnterLeave {
        handle: WindowHandle,
        entered: bool,
    },

    // -- Gamepad events -------------------------------------------------------

    /// A gamepad button was pressed or released.
    GamepadInput {
        /// Gamepad index (0-based).
        gamepad_id: u32,
        button: GamepadButton,
        pressed: bool,
    },

    /// A gamepad axis value changed.
    GamepadAxisMotion {
        gamepad_id: u32,
        axis: GamepadAxis,
        /// Normalized value in [-1.0, 1.0] for sticks or [0.0, 1.0] for triggers.
        value: f32,
    },

    /// A gamepad was connected or disconnected.
    GamepadConnection {
        gamepad_id: u32,
        connected: bool,
        /// Human-readable name of the gamepad, if available.
        name: Option<String>,
    },

    // -- Touch events ---------------------------------------------------------

    /// A touch input event (mobile / touch-screen devices).
    TouchInput {
        handle: WindowHandle,
        /// Unique finger / pointer id for this touch sequence.
        touch_id: u64,
        phase: TouchPhase,
        x: f64,
        y: f64,
        /// Pressure in [0.0, 1.0] if supported by the device.
        pressure: Option<f32>,
    },

    // -- File events ----------------------------------------------------------

    /// One or more files were dragged and dropped onto the window.
    FileDrop {
        handle: WindowHandle,
        paths: Vec<PathBuf>,
    },

    /// Files are being hovered over the window (drag in progress).
    FileHover {
        handle: WindowHandle,
        paths: Vec<PathBuf>,
    },

    /// A file hover was cancelled (dragged away from the window).
    FileHoverCancelled {
        handle: WindowHandle,
    },

    // -- Application lifecycle ------------------------------------------------

    /// The application is being suspended (mobile background, console suspend).
    AppSuspend,

    /// The application is being resumed from a suspended state.
    AppResume,

    /// The OS has requested that the application quit.
    AppQuitRequested,

    /// Low-memory warning from the OS (primarily mobile platforms).
    AppLowMemory,
}

// -----------------------------------------------------------------------------
// Touch Phase
// -----------------------------------------------------------------------------

/// Phase of a touch event within a gesture sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TouchPhase {
    /// A finger touched the screen.
    Started,
    /// A finger moved on the screen.
    Moved,
    /// A finger was lifted from the screen.
    Ended,
    /// The touch was cancelled by the system.
    Cancelled,
}
