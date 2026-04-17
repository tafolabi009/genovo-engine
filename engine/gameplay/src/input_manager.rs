// engine/gameplay/src/input_manager.rs
//
// Comprehensive input: action/axis bindings, dead zone processing, input
// smoothing, input recording/playback, virtual inputs (for AI), input device
// abstraction, input mapping serialization.

use std::collections::HashMap;

// --- Input device ---
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputDevice { Keyboard, Mouse, Gamepad(u8), Touch, Virtual }

// --- Key codes ---
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyCode {
    A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
    Space, Enter, Escape, Tab, Backspace, Delete, Insert, Home, End, PageUp, PageDown,
    Left, Right, Up, Down,
    LShift, RShift, LCtrl, RCtrl, LAlt, RAlt,
    MouseLeft, MouseRight, MouseMiddle, MouseX1, MouseX2,
    GamepadA, GamepadB, GamepadX, GamepadY,
    GamepadLB, GamepadRB, GamepadLT, GamepadRT,
    GamepadStart, GamepadSelect,
    GamepadDpadUp, GamepadDpadDown, GamepadDpadLeft, GamepadDpadRight,
    GamepadLStickPress, GamepadRStickPress,
}

// --- Mouse axes ---
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseAxis { X, Y, ScrollX, ScrollY }

// --- Gamepad axes ---
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GamepadAxis { LeftStickX, LeftStickY, RightStickX, RightStickY, LeftTrigger, RightTrigger }

// --- Input source ---
#[derive(Debug, Clone, PartialEq)]
pub enum InputSource {
    Key(KeyCode),
    MouseButton(KeyCode),
    MouseAxis(MouseAxis),
    GamepadButton(KeyCode),
    GamepadAxis(GamepadAxis),
    Virtual(String),
    Composite { positive: Box<InputSource>, negative: Box<InputSource> },
}

// --- Dead zone ---
#[derive(Debug, Clone, Copy)]
pub struct DeadZoneConfig {
    pub inner: f32,
    pub outer: f32,
    pub shape: DeadZoneShape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeadZoneShape { Axial, Radial, Cross }

impl Default for DeadZoneConfig {
    fn default() -> Self { Self { inner: 0.15, outer: 0.95, shape: DeadZoneShape::Radial } }
}

impl DeadZoneConfig {
    pub fn apply(&self, value: f32) -> f32 {
        let abs_val = value.abs();
        if abs_val < self.inner { return 0.0; }
        if abs_val > self.outer { return value.signum(); }
        let range = self.outer - self.inner;
        if range < 1e-6 { return 0.0; }
        let normalized = (abs_val - self.inner) / range;
        normalized * value.signum()
    }

    pub fn apply_radial(&self, x: f32, y: f32) -> (f32, f32) {
        let magnitude = (x * x + y * y).sqrt();
        if magnitude < self.inner { return (0.0, 0.0); }
        if magnitude > self.outer {
            let inv = 1.0 / magnitude;
            return (x * inv, y * inv);
        }
        let range = self.outer - self.inner;
        if range < 1e-6 { return (0.0, 0.0); }
        let normalized = (magnitude - self.inner) / range;
        let inv = normalized / magnitude;
        (x * inv, y * inv)
    }
}

// --- Input smoothing ---
#[derive(Debug, Clone)]
pub struct InputSmoother {
    pub factor: f32,
    current: f32,
}

impl InputSmoother {
    pub fn new(factor: f32) -> Self { Self { factor: factor.clamp(0.0, 1.0), current: 0.0 } }
    pub fn update(&mut self, target: f32, dt: f32) -> f32 {
        let speed = 1.0 - (-self.factor * dt * 60.0).exp();
        self.current += (target - self.current) * speed;
        self.current
    }
    pub fn reset(&mut self) { self.current = 0.0; }
    pub fn value(&self) -> f32 { self.current }
}

// --- Action binding ---
#[derive(Debug, Clone)]
pub struct ActionBinding {
    pub name: String,
    pub sources: Vec<InputSource>,
    pub consume_input: bool,
}

impl ActionBinding {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), sources: Vec::new(), consume_input: false }
    }
    pub fn bind(mut self, source: InputSource) -> Self { self.sources.push(source); self }
}

// --- Axis binding ---
#[derive(Debug, Clone)]
pub struct AxisBinding {
    pub name: String,
    pub sources: Vec<(InputSource, f32)>,
    pub dead_zone: DeadZoneConfig,
    pub sensitivity: f32,
    pub invert: bool,
    pub smoothing: Option<f32>,
}

impl AxisBinding {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(), sources: Vec::new(), dead_zone: DeadZoneConfig::default(),
            sensitivity: 1.0, invert: false, smoothing: None,
        }
    }
    pub fn bind(mut self, source: InputSource, scale: f32) -> Self {
        self.sources.push((source, scale)); self
    }
    pub fn with_dead_zone(mut self, dz: DeadZoneConfig) -> Self { self.dead_zone = dz; self }
    pub fn with_sensitivity(mut self, s: f32) -> Self { self.sensitivity = s; self }
    pub fn inverted(mut self) -> Self { self.invert = true; self }
}

// --- Input state ---
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ButtonState { Released, Pressed, JustPressed, JustReleased }

impl ButtonState {
    pub fn is_pressed(&self) -> bool { matches!(self, Self::Pressed | Self::JustPressed) }
    pub fn is_just_pressed(&self) -> bool { *self == Self::JustPressed }
    pub fn is_just_released(&self) -> bool { *self == Self::JustReleased }
}

// --- Input recording ---
#[derive(Debug, Clone)]
pub struct InputFrame {
    pub frame: u64,
    pub timestamp: f64,
    pub button_states: HashMap<KeyCode, bool>,
    pub axis_values: HashMap<String, f32>,
}

#[derive(Debug, Clone)]
pub struct InputRecording {
    pub frames: Vec<InputFrame>,
    pub total_time: f64,
    pub frame_count: u64,
}

impl InputRecording {
    pub fn new() -> Self { Self { frames: Vec::new(), total_time: 0.0, frame_count: 0 } }
    pub fn push_frame(&mut self, frame: InputFrame) {
        self.total_time = frame.timestamp;
        self.frame_count = frame.frame;
        self.frames.push(frame);
    }
}

// --- Virtual input (for AI) ---
#[derive(Debug, Clone)]
pub struct VirtualInput {
    pub name: String,
    pub button_value: bool,
    pub axis_value: f32,
}

impl VirtualInput {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), button_value: false, axis_value: 0.0 }
    }
    pub fn press(&mut self) { self.button_value = true; }
    pub fn release(&mut self) { self.button_value = false; }
    pub fn set_axis(&mut self, value: f32) { self.axis_value = value; }
}

// --- Input context ---
#[derive(Debug, Clone)]
pub struct InputContext {
    pub name: String,
    pub priority: i32,
    pub active: bool,
    pub actions: Vec<ActionBinding>,
    pub axes: Vec<AxisBinding>,
    pub consume_all: bool,
}

impl InputContext {
    pub fn new(name: &str, priority: i32) -> Self {
        Self {
            name: name.to_string(), priority, active: true,
            actions: Vec::new(), axes: Vec::new(), consume_all: false,
        }
    }
    pub fn add_action(&mut self, action: ActionBinding) { self.actions.push(action); }
    pub fn add_axis(&mut self, axis: AxisBinding) { self.axes.push(axis); }
}

// --- Input manager ---
pub struct InputManager {
    key_states: HashMap<KeyCode, ButtonState>,
    prev_key_states: HashMap<KeyCode, bool>,
    mouse_position: [f32; 2],
    mouse_delta: [f32; 2],
    mouse_scroll: [f32; 2],
    gamepad_axes: HashMap<GamepadAxis, f32>,
    contexts: Vec<InputContext>,
    action_states: HashMap<String, ButtonState>,
    axis_values: HashMap<String, f32>,
    axis_smoothers: HashMap<String, InputSmoother>,
    virtual_inputs: HashMap<String, VirtualInput>,
    recording: Option<InputRecording>,
    is_recording: bool,
    playback: Option<InputRecording>,
    playback_frame: usize,
    is_playing: bool,
    frame_count: u64,
    timestamp: f64,
    connected_gamepads: Vec<u8>,
}

impl InputManager {
    pub fn new() -> Self {
        Self {
            key_states: HashMap::new(), prev_key_states: HashMap::new(),
            mouse_position: [0.0; 2], mouse_delta: [0.0; 2], mouse_scroll: [0.0; 2],
            gamepad_axes: HashMap::new(), contexts: Vec::new(),
            action_states: HashMap::new(), axis_values: HashMap::new(),
            axis_smoothers: HashMap::new(), virtual_inputs: HashMap::new(),
            recording: None, is_recording: false,
            playback: None, playback_frame: 0, is_playing: false,
            frame_count: 0, timestamp: 0.0, connected_gamepads: Vec::new(),
        }
    }

    // --- Raw input ---
    pub fn set_key(&mut self, key: KeyCode, pressed: bool) {
        let prev = self.prev_key_states.get(&key).copied().unwrap_or(false);
        let state = match (prev, pressed) {
            (false, true) => ButtonState::JustPressed,
            (true, true) => ButtonState::Pressed,
            (true, false) => ButtonState::JustReleased,
            (false, false) => ButtonState::Released,
        };
        self.key_states.insert(key, state);
    }

    pub fn set_mouse_position(&mut self, x: f32, y: f32) {
        self.mouse_delta = [x - self.mouse_position[0], y - self.mouse_position[1]];
        self.mouse_position = [x, y];
    }

    pub fn set_mouse_scroll(&mut self, x: f32, y: f32) { self.mouse_scroll = [x, y]; }

    pub fn set_gamepad_axis(&mut self, axis: GamepadAxis, value: f32) {
        self.gamepad_axes.insert(axis, value);
    }

    // --- Context management ---
    pub fn add_context(&mut self, context: InputContext) { self.contexts.push(context); self.sort_contexts(); }

    pub fn set_context_active(&mut self, name: &str, active: bool) {
        for ctx in &mut self.contexts { if ctx.name == name { ctx.active = active; } }
    }

    fn sort_contexts(&mut self) { self.contexts.sort_by(|a, b| b.priority.cmp(&a.priority)); }

    // --- Query ---
    pub fn is_action_pressed(&self, name: &str) -> bool {
        self.action_states.get(name).map(|s| s.is_pressed()).unwrap_or(false)
    }

    pub fn is_action_just_pressed(&self, name: &str) -> bool {
        self.action_states.get(name).map(|s| s.is_just_pressed()).unwrap_or(false)
    }

    pub fn is_action_just_released(&self, name: &str) -> bool {
        self.action_states.get(name).map(|s| s.is_just_released()).unwrap_or(false)
    }

    pub fn axis_value(&self, name: &str) -> f32 { self.axis_values.get(name).copied().unwrap_or(0.0) }

    pub fn mouse_position(&self) -> [f32; 2] { self.mouse_position }
    pub fn mouse_delta(&self) -> [f32; 2] { self.mouse_delta }

    // --- Virtual input ---
    pub fn add_virtual_input(&mut self, input: VirtualInput) {
        self.virtual_inputs.insert(input.name.clone(), input);
    }

    pub fn get_virtual_input_mut(&mut self, name: &str) -> Option<&mut VirtualInput> {
        self.virtual_inputs.get_mut(name)
    }

    // --- Recording / Playback ---
    pub fn start_recording(&mut self) {
        self.recording = Some(InputRecording::new());
        self.is_recording = true;
    }

    pub fn stop_recording(&mut self) -> Option<InputRecording> {
        self.is_recording = false;
        self.recording.take()
    }

    pub fn start_playback(&mut self, recording: InputRecording) {
        self.playback = Some(recording);
        self.playback_frame = 0;
        self.is_playing = true;
    }

    pub fn stop_playback(&mut self) {
        self.is_playing = false;
        self.playback = None;
        self.playback_frame = 0;
    }

    // --- Frame update ---
    pub fn update(&mut self, dt: f32) {
        self.timestamp += dt as f64;

        // Playback.
        if self.is_playing {
            let frame_data = self.playback.as_ref().and_then(|recording| {
                if self.playback_frame < recording.frames.len() {
                    Some(recording.frames[self.playback_frame].button_states.clone())
                } else {
                    None
                }
            });
            if let Some(states) = frame_data {
                for (&key, &pressed) in &states {
                    self.set_key(key, pressed);
                }
                self.playback_frame += 1;
            } else if self.playback.is_some() {
                self.is_playing = false;
            }
        }

        // Process contexts (highest priority first).
        self.action_states.clear();
        self.axis_values.clear();

        for ctx in &self.contexts {
            if !ctx.active { continue; }

            // Process actions.
            for action in &ctx.actions {
                let mut state = ButtonState::Released;
                for source in &action.sources {
                    let source_state = self.evaluate_source_button(source);
                    if source_state.is_pressed() { state = source_state; break; }
                    if source_state == ButtonState::JustReleased && state == ButtonState::Released {
                        state = source_state;
                    }
                }
                self.action_states.entry(action.name.clone()).or_insert(state);
            }

            // Process axes.
            for axis in &ctx.axes {
                let mut value = 0.0f32;
                for (source, scale) in &axis.sources {
                    value += self.evaluate_source_axis(source) * scale;
                }
                value = axis.dead_zone.apply(value);
                value *= axis.sensitivity;
                if axis.invert { value = -value; }

                // Smoothing.
                if let Some(smooth_factor) = axis.smoothing {
                    let smoother = self.axis_smoothers.entry(axis.name.clone())
                        .or_insert_with(|| InputSmoother::new(smooth_factor));
                    value = smoother.update(value, dt);
                }

                self.axis_values.entry(axis.name.clone()).or_insert(value);
            }
        }

        // Recording.
        if self.is_recording {
            if let Some(ref mut recording) = self.recording {
                let frame = InputFrame {
                    frame: self.frame_count,
                    timestamp: self.timestamp,
                    button_states: self.key_states.iter()
                        .map(|(k, s)| (*k, s.is_pressed())).collect(),
                    axis_values: self.axis_values.clone(),
                };
                recording.push_frame(frame);
            }
        }

        // Update previous states.
        self.prev_key_states = self.key_states.iter()
            .map(|(k, s)| (*k, s.is_pressed())).collect();
        self.mouse_delta = [0.0; 2];
        self.mouse_scroll = [0.0; 2];
        self.frame_count += 1;
    }

    fn evaluate_source_button(&self, source: &InputSource) -> ButtonState {
        match source {
            InputSource::Key(key) | InputSource::MouseButton(key) | InputSource::GamepadButton(key) => {
                self.key_states.get(key).copied().unwrap_or(ButtonState::Released)
            }
            InputSource::Virtual(name) => {
                if let Some(vi) = self.virtual_inputs.get(name) {
                    if vi.button_value { ButtonState::Pressed } else { ButtonState::Released }
                } else { ButtonState::Released }
            }
            _ => ButtonState::Released,
        }
    }

    fn evaluate_source_axis(&self, source: &InputSource) -> f32 {
        match source {
            InputSource::Key(key) | InputSource::MouseButton(key) | InputSource::GamepadButton(key) => {
                if self.key_states.get(key).map(|s| s.is_pressed()).unwrap_or(false) { 1.0 } else { 0.0 }
            }
            InputSource::MouseAxis(axis) => match axis {
                MouseAxis::X => self.mouse_delta[0],
                MouseAxis::Y => self.mouse_delta[1],
                MouseAxis::ScrollX => self.mouse_scroll[0],
                MouseAxis::ScrollY => self.mouse_scroll[1],
            },
            InputSource::GamepadAxis(axis) => {
                self.gamepad_axes.get(axis).copied().unwrap_or(0.0)
            }
            InputSource::Virtual(name) => {
                self.virtual_inputs.get(name).map(|vi| vi.axis_value).unwrap_or(0.0)
            }
            InputSource::Composite { positive, negative } => {
                let pos = self.evaluate_source_axis(positive);
                let neg = self.evaluate_source_axis(negative);
                pos - neg
            }
        }
    }

    // --- Serialization ---
    pub fn serialize_mappings(&self) -> String {
        let mut s = String::new();
        for ctx in &self.contexts {
            s.push_str(&format!("[context:{}]\n", ctx.name));
            for action in &ctx.actions { s.push_str(&format!("  action: {}\n", action.name)); }
            for axis in &ctx.axes { s.push_str(&format!("  axis: {} (sensitivity={})\n", axis.name, axis.sensitivity)); }
        }
        s
    }
}

// --- Default FPS bindings ---
pub fn create_fps_context() -> InputContext {
    let mut ctx = InputContext::new("FPS", 0);
    ctx.add_action(ActionBinding::new("fire").bind(InputSource::Key(KeyCode::MouseLeft)));
    ctx.add_action(ActionBinding::new("aim").bind(InputSource::Key(KeyCode::MouseRight)));
    ctx.add_action(ActionBinding::new("jump").bind(InputSource::Key(KeyCode::Space)));
    ctx.add_action(ActionBinding::new("crouch").bind(InputSource::Key(KeyCode::LCtrl)));
    ctx.add_action(ActionBinding::new("sprint").bind(InputSource::Key(KeyCode::LShift)));
    ctx.add_action(ActionBinding::new("reload").bind(InputSource::Key(KeyCode::R)));
    ctx.add_action(ActionBinding::new("interact").bind(InputSource::Key(KeyCode::E)));
    ctx.add_axis(AxisBinding::new("move_forward")
        .bind(InputSource::Key(KeyCode::W), 1.0)
        .bind(InputSource::Key(KeyCode::S), -1.0));
    ctx.add_axis(AxisBinding::new("move_right")
        .bind(InputSource::Key(KeyCode::D), 1.0)
        .bind(InputSource::Key(KeyCode::A), -1.0));
    ctx.add_axis(AxisBinding::new("look_x").bind(InputSource::MouseAxis(MouseAxis::X), 1.0).with_sensitivity(0.3));
    ctx.add_axis(AxisBinding::new("look_y").bind(InputSource::MouseAxis(MouseAxis::Y), 1.0).with_sensitivity(0.3));
    ctx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dead_zone() {
        let dz = DeadZoneConfig { inner: 0.2, outer: 0.9, shape: DeadZoneShape::Axial };
        assert_eq!(dz.apply(0.1), 0.0);
        assert!(dz.apply(0.5) > 0.0);
        assert!((dz.apply(1.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_input_manager_action() {
        let mut mgr = InputManager::new();
        let mut ctx = InputContext::new("test", 0);
        ctx.add_action(ActionBinding::new("jump").bind(InputSource::Key(KeyCode::Space)));
        mgr.add_context(ctx);
        mgr.set_key(KeyCode::Space, true);
        mgr.update(1.0 / 60.0);
        assert!(mgr.is_action_pressed("jump"));
    }

    #[test]
    fn test_input_manager_axis() {
        let mut mgr = InputManager::new();
        let mut ctx = InputContext::new("test", 0);
        ctx.add_axis(AxisBinding::new("forward").bind(InputSource::Key(KeyCode::W), 1.0).bind(InputSource::Key(KeyCode::S), -1.0));
        mgr.add_context(ctx);
        mgr.set_key(KeyCode::W, true);
        mgr.update(1.0 / 60.0);
        assert!(mgr.axis_value("forward") > 0.0);
    }

    #[test]
    fn test_virtual_input() {
        let mut mgr = InputManager::new();
        let mut ctx = InputContext::new("ai", 0);
        ctx.add_action(ActionBinding::new("ai_fire").bind(InputSource::Virtual("shoot".into())));
        mgr.add_context(ctx);
        let mut vi = VirtualInput::new("shoot");
        vi.press();
        mgr.add_virtual_input(vi);
        mgr.update(1.0 / 60.0);
        assert!(mgr.is_action_pressed("ai_fire"));
    }
}
