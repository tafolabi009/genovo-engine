// =============================================================================
// Genovo Engine - Input Abstraction
// =============================================================================
//
// Unified input system abstracting keyboard, mouse, gamepad, and touch across
// all supported platforms.

use std::collections::HashMap;

// -----------------------------------------------------------------------------
// Key Codes
// -----------------------------------------------------------------------------

/// Comprehensive keyboard key codes covering standard 104/105-key layouts,
/// numpads, media keys, and common international keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum KeyCode {
    // -- Letters --------------------------------------------------------------
    A, B, C, D, E, F, G, H, I, J, K, L, M,
    N, O, P, Q, R, S, T, U, V, W, X, Y, Z,

    // -- Digits (top row) -----------------------------------------------------
    Key0, Key1, Key2, Key3, Key4,
    Key5, Key6, Key7, Key8, Key9,

    // -- Function keys --------------------------------------------------------
    F1, F2, F3, F4, F5, F6,
    F7, F8, F9, F10, F11, F12,
    F13, F14, F15, F16, F17, F18,
    F19, F20, F21, F22, F23, F24,

    // -- Modifiers ------------------------------------------------------------
    LShift, RShift,
    LControl, RControl,
    LAlt, RAlt,
    LSuper, RSuper,

    // -- Navigation -----------------------------------------------------------
    Up, Down, Left, Right,
    Home, End,
    PageUp, PageDown,

    // -- Editing --------------------------------------------------------------
    Insert, Delete,
    Backspace,
    Enter,
    Tab,
    Space,

    // -- Lock keys ------------------------------------------------------------
    CapsLock,
    NumLock,
    ScrollLock,

    // -- Punctuation & symbols ------------------------------------------------
    Minus,       // -
    Equals,      // =
    LeftBracket, // [
    RightBracket,// ]
    Backslash,   // backslash
    Semicolon,   // ;
    Apostrophe,  // '
    Grave,       // `
    Comma,       // ,
    Period,      // .
    Slash,       // /

    // -- Numpad ---------------------------------------------------------------
    Numpad0, Numpad1, Numpad2, Numpad3, Numpad4,
    Numpad5, Numpad6, Numpad7, Numpad8, Numpad9,
    NumpadAdd,
    NumpadSubtract,
    NumpadMultiply,
    NumpadDivide,
    NumpadDecimal,
    NumpadEnter,

    // -- Special keys ---------------------------------------------------------
    Escape,
    PrintScreen,
    Pause,
    Menu,

    // -- Media keys -----------------------------------------------------------
    MediaPlay,
    MediaPause,
    MediaStop,
    MediaNext,
    MediaPrevious,
    VolumeUp,
    VolumeDown,
    VolumeMute,

    /// Catch-all for keys not yet mapped.
    Unknown(u16),
}

// -----------------------------------------------------------------------------
// Mouse Buttons
// -----------------------------------------------------------------------------

/// Mouse button identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    /// Extra buttons (forward / back, etc.) identified by index.
    Extra(u8),
}

// -----------------------------------------------------------------------------
// Gamepad Buttons
// -----------------------------------------------------------------------------

/// Standard gamepad button layout (Xbox-style naming convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GamepadButton {
    /// Bottom face button (A / Cross).
    South,
    /// Right face button (B / Circle).
    East,
    /// Left face button (X / Square).
    West,
    /// Top face button (Y / Triangle).
    North,

    LeftBumper,
    RightBumper,

    /// Left stick click (L3).
    LeftStickPress,
    /// Right stick click (R3).
    RightStickPress,

    Start,
    Select,
    /// Home / Guide / PS button.
    Home,

    DPadUp,
    DPadDown,
    DPadLeft,
    DPadRight,

    /// Left trigger pressed as digital button (platform dependent).
    LeftTriggerDigital,
    /// Right trigger pressed as digital button (platform dependent).
    RightTriggerDigital,

    /// Touchpad press (PlayStation).
    Touchpad,
    /// Share / Create button.
    Share,
}

// -----------------------------------------------------------------------------
// Gamepad Axes
// -----------------------------------------------------------------------------

/// Gamepad analog axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GamepadAxis {
    LeftStickX,
    LeftStickY,
    RightStickX,
    RightStickY,
    /// Analog left trigger [0.0, 1.0].
    LeftTrigger,
    /// Analog right trigger [0.0, 1.0].
    RightTrigger,
}

// -----------------------------------------------------------------------------
// Input Action
// -----------------------------------------------------------------------------

/// Describes the state of an input within a single frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InputAction {
    /// The input was pressed this frame.
    Pressed,
    /// The input was released this frame.
    Released,
    /// The input is being held down (subsequent frames after Pressed).
    Held,
    /// An analog axis value.
    Axis(f32),
}

// -----------------------------------------------------------------------------
// Input Device trait
// -----------------------------------------------------------------------------

/// Trait for abstracting a physical input device (keyboard, mouse, gamepad).
pub trait InputDevice: Send + Sync {
    /// Human-readable name of this device.
    fn name(&self) -> &str;

    /// Returns `true` if the device is currently connected and operational.
    fn is_connected(&self) -> bool;

    /// Poll the device for its current state. Platform implementations call
    /// this once per frame before dispatching events.
    fn poll(&mut self);

    /// Send a rumble/haptic feedback command to the device, if supported.
    ///
    /// `low_frequency` and `high_frequency` are motor intensities in [0.0, 1.0].
    /// `duration_ms` is the duration of the effect. Devices that do not support
    /// haptics should silently ignore this call.
    fn set_rumble(&mut self, _low_frequency: f32, _high_frequency: f32, _duration_ms: u32) {
        // Default no-op: not all input devices support rumble.
    }

    /// Query whether this device supports a specific capability.
    ///
    /// Common capabilities: "rumble", "motion", "touchpad", "adaptive_triggers".
    fn supports_capability(&self, _capability: &str) -> bool {
        false
    }
}

// -----------------------------------------------------------------------------
// Input State
// -----------------------------------------------------------------------------

/// Aggregated snapshot of all input state for the current frame.
///
/// Updated by the [`InputManager`] each frame from platform events.
#[derive(Debug, Clone)]
pub struct InputState {
    // -- Keyboard -------------------------------------------------------------
    /// Keys currently held down.
    pub keys_held: HashMap<KeyCode, bool>,
    /// Keys pressed this frame (edge-triggered).
    pub keys_pressed: HashMap<KeyCode, bool>,
    /// Keys released this frame (edge-triggered).
    pub keys_released: HashMap<KeyCode, bool>,

    // -- Mouse ----------------------------------------------------------------
    /// Mouse buttons currently held down.
    pub mouse_buttons_held: HashMap<MouseButton, bool>,
    /// Mouse buttons pressed this frame.
    pub mouse_buttons_pressed: HashMap<MouseButton, bool>,
    /// Mouse buttons released this frame.
    pub mouse_buttons_released: HashMap<MouseButton, bool>,

    /// Current mouse position (logical pixels relative to focused window).
    pub mouse_position: (f64, f64),
    /// Mouse movement delta since last frame.
    pub mouse_delta: (f64, f64),
    /// Scroll wheel delta (horizontal, vertical).
    pub scroll_delta: (f64, f64),

    // -- Gamepad --------------------------------------------------------------
    /// Per-gamepad button states indexed by gamepad id.
    pub gamepad_buttons: HashMap<u32, HashMap<GamepadButton, bool>>,
    /// Per-gamepad axis values indexed by gamepad id.
    pub gamepad_axes: HashMap<u32, HashMap<GamepadAxis, f32>>,

    // -- Touch ----------------------------------------------------------------
    /// Active touch points indexed by touch id.
    pub active_touches: HashMap<u64, TouchPoint>,
}

/// A currently active touch point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TouchPoint {
    pub id: u64,
    pub x: f64,
    pub y: f64,
    pub pressure: f32,
}

impl InputState {
    /// Create a new, empty input state.
    pub fn new() -> Self {
        Self {
            keys_held: HashMap::new(),
            keys_pressed: HashMap::new(),
            keys_released: HashMap::new(),
            mouse_buttons_held: HashMap::new(),
            mouse_buttons_pressed: HashMap::new(),
            mouse_buttons_released: HashMap::new(),
            mouse_position: (0.0, 0.0),
            mouse_delta: (0.0, 0.0),
            scroll_delta: (0.0, 0.0),
            gamepad_buttons: HashMap::new(),
            gamepad_axes: HashMap::new(),
            active_touches: HashMap::new(),
        }
    }

    /// Returns `true` if the given key is currently held down.
    pub fn is_key_held(&self, key: KeyCode) -> bool {
        self.keys_held.get(&key).copied().unwrap_or(false)
    }

    /// Returns `true` if the given key was pressed this frame.
    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.get(&key).copied().unwrap_or(false)
    }

    /// Returns `true` if the given key was released this frame.
    pub fn is_key_released(&self, key: KeyCode) -> bool {
        self.keys_released.get(&key).copied().unwrap_or(false)
    }

    /// Returns `true` if the given mouse button is currently held down.
    pub fn is_mouse_button_held(&self, button: MouseButton) -> bool {
        self.mouse_buttons_held.get(&button).copied().unwrap_or(false)
    }

    /// Returns `true` if the given mouse button was pressed this frame.
    pub fn is_mouse_button_pressed(&self, button: MouseButton) -> bool {
        self.mouse_buttons_pressed.get(&button).copied().unwrap_or(false)
    }

    /// Returns `true` if the given mouse button was released this frame.
    pub fn is_mouse_button_released(&self, button: MouseButton) -> bool {
        self.mouse_buttons_released.get(&button).copied().unwrap_or(false)
    }

    /// Returns the current value of a gamepad axis, or 0.0 if not available.
    pub fn gamepad_axis_value(&self, gamepad_id: u32, axis: GamepadAxis) -> f32 {
        self.gamepad_axes
            .get(&gamepad_id)
            .and_then(|axes| axes.get(&axis))
            .copied()
            .unwrap_or(0.0)
    }

    /// Returns `true` if the given gamepad button is currently held.
    pub fn is_gamepad_button_held(&self, gamepad_id: u32, button: GamepadButton) -> bool {
        self.gamepad_buttons
            .get(&gamepad_id)
            .and_then(|buttons| buttons.get(&button))
            .copied()
            .unwrap_or(false)
    }

    /// Clear per-frame edge-triggered state. Called at the start of each frame
    /// before new events are processed.
    pub fn clear_frame_state(&mut self) {
        self.keys_pressed.clear();
        self.keys_released.clear();
        self.mouse_buttons_pressed.clear();
        self.mouse_buttons_released.clear();
        self.mouse_delta = (0.0, 0.0);
        self.scroll_delta = (0.0, 0.0);
    }
}

impl Default for InputState {
    fn default() -> Self {
        Self::new()
    }
}

// -----------------------------------------------------------------------------
// Input Manager
// -----------------------------------------------------------------------------

/// Central input manager that routes platform events into a queryable
/// [`InputState`].
///
/// # Usage
///
/// Each frame the engine calls [`InputManager::begin_frame`] to clear
/// edge-triggered state, then feeds platform events via
/// [`InputManager::process_event`], and finally hands the resulting
/// [`InputState`] to gameplay systems.
///
/// # Future extensions
///
/// ## Action Mapping System
/// An action mapping layer will allow binding physical keys/buttons to named
/// actions (e.g., "jump", "fire") with support for multiple bindings per action,
/// configurable at runtime, and serializable to a user preferences file.
///
/// ## Input Contexts / Layers
/// Input contexts (e.g., "gameplay", "ui", "cinematic") will allow different
/// action mappings to be active depending on the game state. Contexts form a
/// stack; the topmost context consumes matching inputs.
///
/// ## Input Recording and Playback
/// Frame-accurate recording of all input events enables deterministic replay
/// for debugging, automated testing, and demo playback. Events are serialized
/// with frame numbers for exact reproduction.
pub struct InputManager {
    state: InputState,
    /// Deadzone threshold for gamepad analog sticks.
    pub stick_deadzone: f32,
    /// Deadzone threshold for gamepad triggers.
    pub trigger_deadzone: f32,
}

impl InputManager {
    /// Create a new input manager with default settings.
    pub fn new() -> Self {
        Self {
            state: InputState::new(),
            stick_deadzone: 0.15,
            trigger_deadzone: 0.05,
        }
    }

    /// Called at the start of each frame to reset per-frame state.
    pub fn begin_frame(&mut self) {
        self.state.clear_frame_state();
    }

    /// Process a single platform event and update internal state.
    pub fn process_event(&mut self, event: &super::events::PlatformEvent) {
        use super::events::PlatformEvent;

        match event {
            PlatformEvent::KeyInput { key, pressed, .. } => {
                if *pressed {
                    if !self.state.is_key_held(*key) {
                        self.state.keys_pressed.insert(*key, true);
                    }
                    self.state.keys_held.insert(*key, true);
                } else {
                    self.state.keys_released.insert(*key, true);
                    self.state.keys_held.remove(key);
                }
            }

            PlatformEvent::MouseInput { button, pressed, .. } => {
                if *pressed {
                    if !self.state.is_mouse_button_held(*button) {
                        self.state.mouse_buttons_pressed.insert(*button, true);
                    }
                    self.state.mouse_buttons_held.insert(*button, true);
                } else {
                    self.state.mouse_buttons_released.insert(*button, true);
                    self.state.mouse_buttons_held.remove(button);
                }
            }

            PlatformEvent::MouseMove { x, y, .. } => {
                let prev = self.state.mouse_position;
                self.state.mouse_delta = (x - prev.0, y - prev.1);
                self.state.mouse_position = (*x, *y);
            }

            PlatformEvent::MouseScroll { delta_x, delta_y, .. } => {
                self.state.scroll_delta.0 += delta_x;
                self.state.scroll_delta.1 += delta_y;
            }

            PlatformEvent::GamepadInput {
                gamepad_id,
                button,
                pressed,
            } => {
                self.state
                    .gamepad_buttons
                    .entry(*gamepad_id)
                    .or_default()
                    .insert(*button, *pressed);
            }

            PlatformEvent::GamepadAxisMotion {
                gamepad_id,
                axis,
                value,
            } => {
                let deadzone = match axis {
                    GamepadAxis::LeftTrigger | GamepadAxis::RightTrigger => self.trigger_deadzone,
                    _ => self.stick_deadzone,
                };
                let v = if value.abs() < deadzone { 0.0 } else { *value };
                self.state
                    .gamepad_axes
                    .entry(*gamepad_id)
                    .or_default()
                    .insert(*axis, v);
            }

            PlatformEvent::TouchInput {
                handle: _,
                touch_id,
                phase,
                x,
                y,
                pressure,
            } => {
                use super::events::TouchPhase;
                match phase {
                    TouchPhase::Started | TouchPhase::Moved => {
                        self.state.active_touches.insert(
                            *touch_id,
                            TouchPoint {
                                id: *touch_id,
                                x: *x,
                                y: *y,
                                pressure: pressure.unwrap_or(1.0),
                            },
                        );
                    }
                    TouchPhase::Ended | TouchPhase::Cancelled => {
                        self.state.active_touches.remove(touch_id);
                    }
                }
            }

            // Events that don't affect input state are ignored here.
            _ => {}
        }
    }

    /// Get a reference to the current input state.
    pub fn state(&self) -> &InputState {
        &self.state
    }

    /// Apply a deadzone to a 2D stick value using circular deadzone.
    fn _apply_circular_deadzone(x: f32, y: f32, deadzone: f32) -> (f32, f32) {
        let magnitude = (x * x + y * y).sqrt();
        if magnitude < deadzone {
            (0.0, 0.0)
        } else {
            let scale = (magnitude - deadzone) / (1.0 - deadzone);
            let normalized_scale = scale / magnitude;
            (x * normalized_scale, y * normalized_scale)
        }
    }
}

impl Default for InputManager {
    fn default() -> Self {
        Self::new()
    }
}
