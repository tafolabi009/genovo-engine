//! Centralized input system for the Genovo engine.
//!
//! Provides a unified polling-based input abstraction for keyboard, mouse,
//! and gamepad. Supports action mapping (name to key bindings), axis mapping
//! with configurable dead zones, input consumed flags, and input recording
//! for replay and debugging.
//!
//! # Architecture
//!
//! The `InputSystem` is updated once per frame at the start of the game loop.
//! It processes raw events from the platform layer and maintains the current
//! state of all input devices. Game code queries the system via action names
//! rather than raw key codes, allowing rebinding without code changes.
//!
//! # Usage
//!
//! ```ignore
//! let mut input = InputSystem::new();
//! input.map_action("jump", &[KeyBinding::Key(KeyCode::Space)]);
//! input.map_action("fire", &[KeyBinding::MouseButton(MouseButton::Left)]);
//! input.map_axis("move_x", AxisMapping::keys(KeyCode::D, KeyCode::A));
//! input.map_axis("move_y", AxisMapping::keys(KeyCode::W, KeyCode::S));
//!
//! // Each frame:
//! input.begin_frame();
//! input.process_event(&event);
//! // ...
//! if input.action_just_pressed("jump") { /* ... */ }
//! let move_dir = input.axis_value("move_x");
//! input.end_frame();
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Key / button identifiers
// ---------------------------------------------------------------------------

/// Keyboard key codes (subset covering common game keys).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyCode {
    A, B, C, D, E, F, G, H, I, J, K, L, M,
    N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    Key0, Key1, Key2, Key3, Key4, Key5, Key6, Key7, Key8, Key9,
    Space, Enter, Escape, Tab, Backspace, Delete, Insert,
    Up, Down, Left, Right,
    LShift, RShift, LCtrl, RCtrl, LAlt, RAlt,
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
    Home, End, PageUp, PageDown,
    NumPad0, NumPad1, NumPad2, NumPad3, NumPad4,
    NumPad5, NumPad6, NumPad7, NumPad8, NumPad9,
    Minus, Equals, LeftBracket, RightBracket, Semicolon,
    Apostrophe, Comma, Period, Slash, Backslash, Grave,
}

/// Mouse buttons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Button4,
    Button5,
}

/// Gamepad buttons (Xbox-style layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GamepadButton {
    South,       // A / Cross
    East,        // B / Circle
    West,        // X / Square
    North,       // Y / Triangle
    LeftBumper,
    RightBumper,
    LeftTrigger,
    RightTrigger,
    Select,
    Start,
    LeftStick,
    RightStick,
    DPadUp,
    DPadDown,
    DPadLeft,
    DPadRight,
}

/// Gamepad axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GamepadAxis {
    LeftStickX,
    LeftStickY,
    RightStickX,
    RightStickY,
    LeftTrigger,
    RightTrigger,
}

// ---------------------------------------------------------------------------
// Key binding types
// ---------------------------------------------------------------------------

/// A single input binding that can trigger an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyBinding {
    Key(KeyCode),
    MouseButton(MouseButton),
    Gamepad(GamepadButton),
}

/// Axis mapping: maps an axis name to positive/negative keys or a gamepad axis.
#[derive(Debug, Clone)]
pub enum AxisMapping {
    /// Two keyboard keys: positive direction and negative direction.
    Keys {
        positive: KeyCode,
        negative: KeyCode,
        /// Smoothing factor (0 = instant, 1 = maximum smooth). Default 0.
        gravity: f32,
        /// Sensitivity multiplier. Default 1.0.
        sensitivity: f32,
    },
    /// A gamepad analog axis.
    GamepadAxis {
        axis: GamepadAxis,
        /// Dead zone threshold (values below this are clamped to 0).
        dead_zone: f32,
        /// Whether to invert the axis.
        inverted: bool,
        /// Sensitivity multiplier.
        sensitivity: f32,
    },
    /// Mouse movement axis (delta per frame).
    MouseAxis {
        /// Which mouse axis: 0 = X (horizontal), 1 = Y (vertical).
        axis: u8,
        /// Sensitivity multiplier.
        sensitivity: f32,
    },
}

impl AxisMapping {
    /// Create a keyboard axis from two keys.
    pub fn keys(positive: KeyCode, negative: KeyCode) -> Self {
        Self::Keys {
            positive,
            negative,
            gravity: 0.0,
            sensitivity: 1.0,
        }
    }

    /// Create a gamepad axis mapping with default dead zone.
    pub fn gamepad(axis: GamepadAxis) -> Self {
        Self::GamepadAxis {
            axis,
            dead_zone: 0.15,
            inverted: false,
            sensitivity: 1.0,
        }
    }

    /// Create a mouse axis mapping.
    pub fn mouse(axis: u8) -> Self {
        Self::MouseAxis {
            axis,
            sensitivity: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Button state
// ---------------------------------------------------------------------------

/// Tracks the state of a single button across frames.
#[derive(Debug, Clone, Copy, Default)]
struct ButtonState {
    /// Whether the button is currently held down.
    pressed: bool,
    /// Whether the button was pressed this frame (transition from up to down).
    just_pressed: bool,
    /// Whether the button was released this frame (transition from down to up).
    just_released: bool,
    /// How many frames the button has been held.
    held_frames: u32,
    /// Timestamp (frame number) of the last press.
    last_press_frame: u64,
}

// ---------------------------------------------------------------------------
// Input recording
// ---------------------------------------------------------------------------

/// A recorded input event for replay.
#[derive(Debug, Clone)]
pub struct RecordedInput {
    /// Frame number when the input occurred.
    pub frame: u64,
    /// The input event.
    pub event: RecordedEvent,
}

/// Types of recorded input events.
#[derive(Debug, Clone)]
pub enum RecordedEvent {
    KeyDown(KeyCode),
    KeyUp(KeyCode),
    MouseDown(MouseButton),
    MouseUp(MouseButton),
    MouseMove { dx: f32, dy: f32 },
    MouseScroll(f32),
    GamepadButtonDown(GamepadButton),
    GamepadButtonUp(GamepadButton),
    GamepadAxisChanged(GamepadAxis, f32),
}

// ---------------------------------------------------------------------------
// InputSystem
// ---------------------------------------------------------------------------

/// Centralized input system that manages all input state.
///
/// Updated once per frame. Provides:
/// - Raw key/button polling (`is_key_pressed`, `is_key_just_pressed`)
/// - Action mapping (`action_pressed`, `action_just_pressed`)
/// - Axis mapping with dead zones and smoothing
/// - Mouse position and delta tracking
/// - Scroll wheel accumulator
/// - Input consumed flag (for UI priority)
/// - Input recording and playback
pub struct InputSystem {
    // -- Raw state --
    keys: HashMap<KeyCode, ButtonState>,
    mouse_buttons: HashMap<MouseButton, ButtonState>,
    gamepad_buttons: HashMap<GamepadButton, ButtonState>,
    gamepad_axes: HashMap<GamepadAxis, f32>,

    // -- Mouse --
    mouse_position: [f32; 2],
    mouse_delta: [f32; 2],
    scroll_delta: f32,

    // -- Action / axis mappings --
    action_bindings: HashMap<String, Vec<KeyBinding>>,
    axis_bindings: HashMap<String, Vec<AxisMapping>>,

    // -- Axis smoothed values (for keyboard gravity) --
    axis_values: HashMap<String, f32>,

    // -- Frame tracking --
    current_frame: u64,

    // -- Input consumed --
    consumed: bool,

    // -- Recording --
    recording: bool,
    recorded_inputs: Vec<RecordedInput>,

    // -- Playback --
    playing_back: bool,
    playback_inputs: Vec<RecordedInput>,
    playback_index: usize,
    playback_start_frame: u64,
}

impl InputSystem {
    /// Create a new input system with no bindings.
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
            mouse_buttons: HashMap::new(),
            gamepad_buttons: HashMap::new(),
            gamepad_axes: HashMap::new(),
            mouse_position: [0.0; 2],
            mouse_delta: [0.0; 2],
            scroll_delta: 0.0,
            action_bindings: HashMap::new(),
            axis_bindings: HashMap::new(),
            axis_values: HashMap::new(),
            current_frame: 0,
            consumed: false,
            recording: false,
            recorded_inputs: Vec::new(),
            playing_back: false,
            playback_inputs: Vec::new(),
            playback_index: 0,
            playback_start_frame: 0,
        }
    }

    // -- Frame lifecycle --

    /// Call at the start of each frame before processing events.
    ///
    /// Resets per-frame state (just_pressed, just_released, mouse delta, scroll).
    pub fn begin_frame(&mut self) {
        // Reset per-frame transitions
        for state in self.keys.values_mut() {
            state.just_pressed = false;
            state.just_released = false;
        }
        for state in self.mouse_buttons.values_mut() {
            state.just_pressed = false;
            state.just_released = false;
        }
        for state in self.gamepad_buttons.values_mut() {
            state.just_pressed = false;
            state.just_released = false;
        }
        self.mouse_delta = [0.0; 2];
        self.scroll_delta = 0.0;
        self.consumed = false;

        // Process playback inputs for this frame
        if self.playing_back {
            let relative_frame = self.current_frame - self.playback_start_frame;
            while self.playback_index < self.playback_inputs.len() {
                let rec = &self.playback_inputs[self.playback_index];
                if rec.frame > relative_frame {
                    break;
                }
                self.apply_recorded_event(&rec.event.clone());
                self.playback_index += 1;
            }
            if self.playback_index >= self.playback_inputs.len() {
                self.playing_back = false;
            }
        }
    }

    /// Call at the end of each frame after all game logic.
    ///
    /// Increments the frame counter and updates held durations.
    pub fn end_frame(&mut self) {
        for state in self.keys.values_mut() {
            if state.pressed {
                state.held_frames += 1;
            }
        }
        for state in self.mouse_buttons.values_mut() {
            if state.pressed {
                state.held_frames += 1;
            }
        }
        for state in self.gamepad_buttons.values_mut() {
            if state.pressed {
                state.held_frames += 1;
            }
        }
        self.current_frame += 1;
    }

    // -- Event processing --

    /// Process a key down event.
    pub fn key_down(&mut self, key: KeyCode) {
        let state = self.keys.entry(key).or_default();
        if !state.pressed {
            state.just_pressed = true;
            state.held_frames = 0;
            state.last_press_frame = self.current_frame;
        }
        state.pressed = true;
        if self.recording {
            self.recorded_inputs.push(RecordedInput {
                frame: self.current_frame,
                event: RecordedEvent::KeyDown(key),
            });
        }
    }

    /// Process a key up event.
    pub fn key_up(&mut self, key: KeyCode) {
        let state = self.keys.entry(key).or_default();
        if state.pressed {
            state.just_released = true;
        }
        state.pressed = false;
        state.held_frames = 0;
        if self.recording {
            self.recorded_inputs.push(RecordedInput {
                frame: self.current_frame,
                event: RecordedEvent::KeyUp(key),
            });
        }
    }

    /// Process a mouse button down event.
    pub fn mouse_button_down(&mut self, button: MouseButton) {
        let state = self.mouse_buttons.entry(button).or_default();
        if !state.pressed {
            state.just_pressed = true;
            state.held_frames = 0;
            state.last_press_frame = self.current_frame;
        }
        state.pressed = true;
        if self.recording {
            self.recorded_inputs.push(RecordedInput {
                frame: self.current_frame,
                event: RecordedEvent::MouseDown(button),
            });
        }
    }

    /// Process a mouse button up event.
    pub fn mouse_button_up(&mut self, button: MouseButton) {
        let state = self.mouse_buttons.entry(button).or_default();
        if state.pressed {
            state.just_released = true;
        }
        state.pressed = false;
        state.held_frames = 0;
        if self.recording {
            self.recorded_inputs.push(RecordedInput {
                frame: self.current_frame,
                event: RecordedEvent::MouseUp(button),
            });
        }
    }

    /// Process mouse movement.
    pub fn mouse_move(&mut self, x: f32, y: f32) {
        let dx = x - self.mouse_position[0];
        let dy = y - self.mouse_position[1];
        self.mouse_position = [x, y];
        self.mouse_delta[0] += dx;
        self.mouse_delta[1] += dy;
        if self.recording {
            self.recorded_inputs.push(RecordedInput {
                frame: self.current_frame,
                event: RecordedEvent::MouseMove { dx, dy },
            });
        }
    }

    /// Process mouse scroll.
    pub fn mouse_scroll(&mut self, delta: f32) {
        self.scroll_delta += delta;
        if self.recording {
            self.recorded_inputs.push(RecordedInput {
                frame: self.current_frame,
                event: RecordedEvent::MouseScroll(delta),
            });
        }
    }

    /// Process a gamepad button down event.
    pub fn gamepad_button_down(&mut self, button: GamepadButton) {
        let state = self.gamepad_buttons.entry(button).or_default();
        if !state.pressed {
            state.just_pressed = true;
            state.held_frames = 0;
            state.last_press_frame = self.current_frame;
        }
        state.pressed = true;
        if self.recording {
            self.recorded_inputs.push(RecordedInput {
                frame: self.current_frame,
                event: RecordedEvent::GamepadButtonDown(button),
            });
        }
    }

    /// Process a gamepad button up event.
    pub fn gamepad_button_up(&mut self, button: GamepadButton) {
        let state = self.gamepad_buttons.entry(button).or_default();
        if state.pressed {
            state.just_released = true;
        }
        state.pressed = false;
        state.held_frames = 0;
        if self.recording {
            self.recorded_inputs.push(RecordedInput {
                frame: self.current_frame,
                event: RecordedEvent::GamepadButtonUp(button),
            });
        }
    }

    /// Process a gamepad axis value change.
    pub fn gamepad_axis_changed(&mut self, axis: GamepadAxis, value: f32) {
        self.gamepad_axes.insert(axis, value);
        if self.recording {
            self.recorded_inputs.push(RecordedInput {
                frame: self.current_frame,
                event: RecordedEvent::GamepadAxisChanged(axis, value),
            });
        }
    }

    fn apply_recorded_event(&mut self, event: &RecordedEvent) {
        match event {
            RecordedEvent::KeyDown(k) => self.key_down(*k),
            RecordedEvent::KeyUp(k) => self.key_up(*k),
            RecordedEvent::MouseDown(b) => self.mouse_button_down(*b),
            RecordedEvent::MouseUp(b) => self.mouse_button_up(*b),
            RecordedEvent::MouseMove { dx, dy } => {
                self.mouse_delta[0] += dx;
                self.mouse_delta[1] += dy;
            }
            RecordedEvent::MouseScroll(d) => self.scroll_delta += d,
            RecordedEvent::GamepadButtonDown(b) => self.gamepad_button_down(*b),
            RecordedEvent::GamepadButtonUp(b) => self.gamepad_button_up(*b),
            RecordedEvent::GamepadAxisChanged(a, v) => { self.gamepad_axes.insert(*a, *v); }
        }
    }

    // -- Raw queries --

    /// Whether a key is currently held down.
    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.keys.get(&key).map_or(false, |s| s.pressed)
    }

    /// Whether a key was just pressed this frame.
    pub fn is_key_just_pressed(&self, key: KeyCode) -> bool {
        self.keys.get(&key).map_or(false, |s| s.just_pressed)
    }

    /// Whether a key was just released this frame.
    pub fn is_key_just_released(&self, key: KeyCode) -> bool {
        self.keys.get(&key).map_or(false, |s| s.just_released)
    }

    /// How many frames a key has been held.
    pub fn key_held_frames(&self, key: KeyCode) -> u32 {
        self.keys.get(&key).map_or(0, |s| s.held_frames)
    }

    /// Whether a mouse button is currently held down.
    pub fn is_mouse_pressed(&self, button: MouseButton) -> bool {
        self.mouse_buttons.get(&button).map_or(false, |s| s.pressed)
    }

    /// Whether a mouse button was just pressed this frame.
    pub fn is_mouse_just_pressed(&self, button: MouseButton) -> bool {
        self.mouse_buttons.get(&button).map_or(false, |s| s.just_pressed)
    }

    /// Whether a mouse button was just released this frame.
    pub fn is_mouse_just_released(&self, button: MouseButton) -> bool {
        self.mouse_buttons.get(&button).map_or(false, |s| s.just_released)
    }

    /// Current mouse position.
    pub fn mouse_position(&self) -> [f32; 2] {
        self.mouse_position
    }

    /// Mouse movement delta this frame.
    pub fn mouse_delta(&self) -> [f32; 2] {
        self.mouse_delta
    }

    /// Scroll wheel delta this frame.
    pub fn scroll_delta(&self) -> f32 {
        self.scroll_delta
    }

    /// Raw gamepad axis value.
    pub fn gamepad_axis_raw(&self, axis: GamepadAxis) -> f32 {
        self.gamepad_axes.get(&axis).copied().unwrap_or(0.0)
    }

    /// Whether a gamepad button is held.
    pub fn is_gamepad_pressed(&self, button: GamepadButton) -> bool {
        self.gamepad_buttons.get(&button).map_or(false, |s| s.pressed)
    }

    /// Whether a gamepad button was just pressed.
    pub fn is_gamepad_just_pressed(&self, button: GamepadButton) -> bool {
        self.gamepad_buttons.get(&button).map_or(false, |s| s.just_pressed)
    }

    // -- Action mapping --

    /// Map an action name to one or more key bindings.
    pub fn map_action(&mut self, name: &str, bindings: &[KeyBinding]) {
        self.action_bindings
            .insert(name.to_string(), bindings.to_vec());
    }

    /// Remove an action mapping.
    pub fn unmap_action(&mut self, name: &str) {
        self.action_bindings.remove(name);
    }

    /// Whether any binding for the named action is currently pressed.
    pub fn action_pressed(&self, name: &str) -> bool {
        if self.consumed { return false; }
        self.action_bindings.get(name).map_or(false, |bindings| {
            bindings.iter().any(|b| self.binding_pressed(b))
        })
    }

    /// Whether any binding for the named action was just pressed this frame.
    pub fn action_just_pressed(&self, name: &str) -> bool {
        if self.consumed { return false; }
        self.action_bindings.get(name).map_or(false, |bindings| {
            bindings.iter().any(|b| self.binding_just_pressed(b))
        })
    }

    /// Whether any binding for the named action was just released this frame.
    pub fn action_just_released(&self, name: &str) -> bool {
        if self.consumed { return false; }
        self.action_bindings.get(name).map_or(false, |bindings| {
            bindings.iter().any(|b| self.binding_just_released(b))
        })
    }

    fn binding_pressed(&self, binding: &KeyBinding) -> bool {
        match binding {
            KeyBinding::Key(k) => self.is_key_pressed(*k),
            KeyBinding::MouseButton(b) => self.is_mouse_pressed(*b),
            KeyBinding::Gamepad(b) => self.is_gamepad_pressed(*b),
        }
    }

    fn binding_just_pressed(&self, binding: &KeyBinding) -> bool {
        match binding {
            KeyBinding::Key(k) => self.is_key_just_pressed(*k),
            KeyBinding::MouseButton(b) => self.is_mouse_just_pressed(*b),
            KeyBinding::Gamepad(b) => self.is_gamepad_just_pressed(*b),
        }
    }

    fn binding_just_released(&self, binding: &KeyBinding) -> bool {
        match binding {
            KeyBinding::Key(k) => self.is_key_just_released(*k),
            KeyBinding::MouseButton(b) => self.is_mouse_just_released(*b),
            KeyBinding::Gamepad(b) => {
                self.gamepad_buttons.get(b).map_or(false, |s| s.just_released)
            }
        }
    }

    // -- Axis mapping --

    /// Map an axis name to one or more axis mappings.
    pub fn map_axis(&mut self, name: &str, mappings: Vec<AxisMapping>) {
        self.axis_bindings.insert(name.to_string(), mappings);
        self.axis_values.entry(name.to_string()).or_insert(0.0);
    }

    /// Get the current value of a named axis (typically -1.0 to 1.0).
    pub fn axis_value(&self, name: &str) -> f32 {
        if self.consumed { return 0.0; }
        let mut value = 0.0f32;
        if let Some(mappings) = self.axis_bindings.get(name) {
            for mapping in mappings {
                let v = self.evaluate_axis_mapping(mapping);
                if v.abs() > value.abs() {
                    value = v;
                }
            }
        }
        value
    }

    fn evaluate_axis_mapping(&self, mapping: &AxisMapping) -> f32 {
        match mapping {
            AxisMapping::Keys { positive, negative, sensitivity, .. } => {
                let pos = if self.is_key_pressed(*positive) { 1.0 } else { 0.0 };
                let neg = if self.is_key_pressed(*negative) { 1.0 } else { 0.0 };
                (pos - neg) * sensitivity
            }
            AxisMapping::GamepadAxis { axis, dead_zone, inverted, sensitivity } => {
                let raw = self.gamepad_axis_raw(*axis);
                let sign = if *inverted { -1.0 } else { 1.0 };
                let abs_val = raw.abs();
                if abs_val < *dead_zone {
                    0.0
                } else {
                    // Remap from [dead_zone, 1.0] to [0.0, 1.0]
                    let remapped = (abs_val - dead_zone) / (1.0 - dead_zone);
                    remapped * raw.signum() * sign * sensitivity
                }
            }
            AxisMapping::MouseAxis { axis, sensitivity } => {
                let delta = self.mouse_delta[*axis as usize];
                delta * sensitivity
            }
        }
    }

    // -- Input consumed flag --

    /// Mark all input as consumed for this frame (e.g., UI ate the input).
    pub fn consume(&mut self) {
        self.consumed = true;
    }

    /// Whether input has been consumed this frame.
    pub fn is_consumed(&self) -> bool {
        self.consumed
    }

    // -- Recording --

    /// Start recording input events.
    pub fn start_recording(&mut self) {
        self.recording = true;
        self.recorded_inputs.clear();
    }

    /// Stop recording and return the recorded events.
    pub fn stop_recording(&mut self) -> Vec<RecordedInput> {
        self.recording = false;
        std::mem::take(&mut self.recorded_inputs)
    }

    /// Whether recording is active.
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Start playback of recorded inputs.
    pub fn start_playback(&mut self, inputs: Vec<RecordedInput>) {
        self.playback_inputs = inputs;
        self.playback_index = 0;
        self.playback_start_frame = self.current_frame;
        self.playing_back = true;
    }

    /// Whether playback is active.
    pub fn is_playing_back(&self) -> bool {
        self.playing_back
    }

    /// Stop playback.
    pub fn stop_playback(&mut self) {
        self.playing_back = false;
    }

    // -- Utility --

    /// Get the current frame number.
    pub fn frame(&self) -> u64 {
        self.current_frame
    }

    /// Reset all input state (useful on focus loss).
    pub fn reset(&mut self) {
        self.keys.clear();
        self.mouse_buttons.clear();
        self.gamepad_buttons.clear();
        self.gamepad_axes.clear();
        self.mouse_delta = [0.0; 2];
        self.scroll_delta = 0.0;
        self.consumed = false;
    }

    /// Get all registered action names.
    pub fn action_names(&self) -> Vec<&str> {
        self.action_bindings.keys().map(|s| s.as_str()).collect()
    }

    /// Get all registered axis names.
    pub fn axis_names(&self) -> Vec<&str> {
        self.axis_bindings.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for InputSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_press_release() {
        let mut input = InputSystem::new();
        input.begin_frame();
        input.key_down(KeyCode::Space);
        assert!(input.is_key_pressed(KeyCode::Space));
        assert!(input.is_key_just_pressed(KeyCode::Space));
        input.end_frame();
        input.begin_frame();
        assert!(input.is_key_pressed(KeyCode::Space));
        assert!(!input.is_key_just_pressed(KeyCode::Space));
        input.key_up(KeyCode::Space);
        assert!(!input.is_key_pressed(KeyCode::Space));
        assert!(input.is_key_just_released(KeyCode::Space));
        input.end_frame();
    }

    #[test]
    fn test_action_mapping() {
        let mut input = InputSystem::new();
        input.map_action("jump", &[KeyBinding::Key(KeyCode::Space)]);
        input.begin_frame();
        assert!(!input.action_pressed("jump"));
        input.key_down(KeyCode::Space);
        assert!(input.action_pressed("jump"));
        assert!(input.action_just_pressed("jump"));
        input.end_frame();
    }

    #[test]
    fn test_axis_mapping_keys() {
        let mut input = InputSystem::new();
        input.map_axis("move_x", vec![AxisMapping::keys(KeyCode::D, KeyCode::A)]);
        input.begin_frame();
        assert!((input.axis_value("move_x")).abs() < 1e-5);
        input.key_down(KeyCode::D);
        assert!((input.axis_value("move_x") - 1.0).abs() < 1e-5);
        input.key_down(KeyCode::A);
        assert!((input.axis_value("move_x")).abs() < 1e-5); // both pressed = 0
        input.key_up(KeyCode::D);
        assert!((input.axis_value("move_x") + 1.0).abs() < 1e-5);
        input.end_frame();
    }

    #[test]
    fn test_mouse_delta() {
        let mut input = InputSystem::new();
        input.begin_frame();
        input.mouse_move(100.0, 200.0);
        input.mouse_move(110.0, 205.0);
        let delta = input.mouse_delta();
        assert!((delta[0] - 110.0).abs() < 1e-3); // total movement from 0
        input.end_frame();
        input.begin_frame();
        assert!((input.mouse_delta()[0]).abs() < 1e-5); // reset
    }

    #[test]
    fn test_consumed_flag() {
        let mut input = InputSystem::new();
        input.map_action("fire", &[KeyBinding::Key(KeyCode::Space)]);
        input.begin_frame();
        input.key_down(KeyCode::Space);
        assert!(input.action_pressed("fire"));
        input.consume();
        assert!(!input.action_pressed("fire"));
        input.end_frame();
    }

    #[test]
    fn test_recording_playback() {
        let mut input = InputSystem::new();
        input.start_recording();
        input.begin_frame();
        input.key_down(KeyCode::W);
        input.end_frame();
        input.begin_frame();
        input.key_up(KeyCode::W);
        input.end_frame();
        let recording = input.stop_recording();
        assert_eq!(recording.len(), 2);

        // Reset and playback
        input.reset();
        input.start_playback(recording);
        input.begin_frame();
        assert!(input.is_key_pressed(KeyCode::W));
        input.end_frame();
        input.begin_frame();
        assert!(!input.is_key_pressed(KeyCode::W));
        input.end_frame();
    }

    #[test]
    fn test_gamepad_dead_zone() {
        let mut input = InputSystem::new();
        input.map_axis("look_x", vec![AxisMapping::GamepadAxis {
            axis: GamepadAxis::RightStickX,
            dead_zone: 0.2,
            inverted: false,
            sensitivity: 1.0,
        }]);
        input.begin_frame();
        input.gamepad_axis_changed(GamepadAxis::RightStickX, 0.1);
        assert!((input.axis_value("look_x")).abs() < 1e-5); // below dead zone
        input.gamepad_axis_changed(GamepadAxis::RightStickX, 0.5);
        let val = input.axis_value("look_x");
        assert!(val > 0.0 && val < 1.0);
        input.end_frame();
    }

    #[test]
    fn test_held_frames() {
        let mut input = InputSystem::new();
        input.begin_frame();
        input.key_down(KeyCode::A);
        input.end_frame();
        assert_eq!(input.key_held_frames(KeyCode::A), 1);
        input.begin_frame();
        input.end_frame();
        assert_eq!(input.key_held_frames(KeyCode::A), 2);
    }
}
