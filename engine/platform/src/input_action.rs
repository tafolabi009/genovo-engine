// =============================================================================
// Genovo Engine - Input Action Mapping System
// =============================================================================
//
// A comprehensive input action mapping layer that sits on top of the raw
// `InputState` system. Provides named actions, multi-device bindings, input
// contexts/layers, combo and sequence detection, gesture recognition, input
// recording/playback, rebinding UI support, and JSON serialization.

use std::collections::{HashMap, VecDeque};

use super::interface::input::{GamepadAxis, GamepadButton, KeyCode, MouseButton};

// ---------------------------------------------------------------------------
// Vec2 (minimal 2D vector for axis output)
// ---------------------------------------------------------------------------

/// Simple 2D vector for axis output.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len < 1e-8 {
            Self::default()
        } else {
            Self {
                x: self.x / len,
                y: self.y / len,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Mouse Axis Type
// ---------------------------------------------------------------------------

/// The type of mouse axis for binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseAxisType {
    /// Horizontal mouse movement delta.
    X,
    /// Vertical mouse movement delta.
    Y,
    /// Scroll wheel (vertical).
    ScrollY,
    /// Scroll wheel (horizontal).
    ScrollX,
}

// ---------------------------------------------------------------------------
// Input Binding
// ---------------------------------------------------------------------------

/// A single input binding that maps a physical control to an action.
#[derive(Debug, Clone, PartialEq)]
pub enum InputBinding {
    /// A keyboard key.
    Key(KeyCode),
    /// A mouse button.
    MouseButton(MouseButton),
    /// A mouse axis (produces analog values).
    MouseAxis(MouseAxisType),
    /// A gamepad button.
    GamepadButton(GamepadButton),
    /// A gamepad analog axis.
    GamepadAxis(GamepadAxis),
    /// Multiple inputs that must all be active simultaneously (chord).
    Combo(Vec<InputBinding>),
    /// A timed sequence of inputs (e.g., fighting game combos).
    /// Each entry is (binding, max_time_window_seconds).
    Sequence(Vec<(InputBinding, f32)>),
    /// A modifier + key combination (convenience for Ctrl+S, etc.).
    Modified {
        modifier: Box<InputBinding>,
        key: Box<InputBinding>,
    },
}

impl InputBinding {
    /// Create a key binding.
    pub fn key(code: KeyCode) -> Self {
        InputBinding::Key(code)
    }

    /// Create a mouse button binding.
    pub fn mouse_button(button: MouseButton) -> Self {
        InputBinding::MouseButton(button)
    }

    /// Create a mouse axis binding.
    pub fn mouse_axis(axis: MouseAxisType) -> Self {
        InputBinding::MouseAxis(axis)
    }

    /// Create a gamepad button binding.
    pub fn gamepad_button(button: GamepadButton) -> Self {
        InputBinding::GamepadButton(button)
    }

    /// Create a gamepad axis binding.
    pub fn gamepad_axis(axis: GamepadAxis) -> Self {
        InputBinding::GamepadAxis(axis)
    }

    /// Create a combo binding (all must be pressed simultaneously).
    pub fn combo(bindings: Vec<InputBinding>) -> Self {
        InputBinding::Combo(bindings)
    }

    /// Create a sequence binding with time windows.
    pub fn sequence(steps: Vec<(InputBinding, f32)>) -> Self {
        InputBinding::Sequence(steps)
    }

    /// Create a modifier + key binding.
    pub fn modified(modifier: InputBinding, key: InputBinding) -> Self {
        InputBinding::Modified {
            modifier: Box::new(modifier),
            key: Box::new(key),
        }
    }
}

// ---------------------------------------------------------------------------
// Action State
// ---------------------------------------------------------------------------

/// The lifecycle state of an input action within a single frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionState {
    /// The action is not active.
    Idle,
    /// The action was just triggered this frame.
    Started,
    /// The action is continuously active (held).
    Ongoing,
    /// The action was just completed this frame (released after being held).
    Completed,
    /// The action was cancelled (e.g., context changed while active).
    Canceled,
}

impl Default for ActionState {
    fn default() -> Self {
        Self::Idle
    }
}

// ---------------------------------------------------------------------------
// Input Action
// ---------------------------------------------------------------------------

/// A named input action with one or more bindings.
#[derive(Debug, Clone)]
pub struct InputActionDef {
    /// Action name (e.g., "Jump", "Fire", "MoveForward").
    pub name: String,
    /// All bindings that can trigger this action.
    pub bindings: Vec<InputBinding>,
    /// Analog dead zone (below this threshold, the value is treated as 0).
    pub dead_zone: f32,
    /// Sensitivity multiplier for analog inputs.
    pub sensitivity: f32,
    /// Whether this is a "consume" action (blocks passthrough to lower contexts).
    pub consume: bool,
}

impl InputActionDef {
    /// Create a new action definition.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            bindings: Vec::new(),
            dead_zone: 0.1,
            sensitivity: 1.0,
            consume: false,
        }
    }

    /// Builder: add a binding.
    pub fn with_binding(mut self, binding: InputBinding) -> Self {
        self.bindings.push(binding);
        self
    }

    /// Builder: set dead zone.
    pub fn with_dead_zone(mut self, dz: f32) -> Self {
        self.dead_zone = dz;
        self
    }

    /// Builder: set sensitivity.
    pub fn with_sensitivity(mut self, s: f32) -> Self {
        self.sensitivity = s;
        self
    }

    /// Builder: mark as consuming.
    pub fn consuming(mut self) -> Self {
        self.consume = true;
        self
    }
}

// ---------------------------------------------------------------------------
// Input Action Map
// ---------------------------------------------------------------------------

/// A named set of input actions (e.g., "Gameplay", "Menu", "Vehicle").
#[derive(Debug, Clone)]
pub struct InputActionMap {
    /// Name of this action map.
    pub name: String,
    /// All action definitions in this map.
    actions: Vec<InputActionDef>,
    /// Runtime state of each action (indexed by action name).
    states: HashMap<String, ActionState>,
    /// Analog values for each action (for axis-like actions).
    axis_values: HashMap<String, f32>,
    /// Whether this map is currently enabled.
    pub enabled: bool,
}

impl InputActionMap {
    /// Create a new empty action map.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            actions: Vec::new(),
            states: HashMap::new(),
            axis_values: HashMap::new(),
            enabled: true,
        }
    }

    /// Define a new action in this map.
    pub fn define_action(&mut self, action: InputActionDef) {
        self.states.insert(action.name.clone(), ActionState::Idle);
        self.axis_values.insert(action.name.clone(), 0.0);
        self.actions.push(action);
    }

    /// Check if an action is currently pressed (Started or Ongoing).
    pub fn is_pressed(&self, name: &str) -> bool {
        matches!(
            self.states.get(name),
            Some(ActionState::Started) | Some(ActionState::Ongoing)
        )
    }

    /// Check if an action was just pressed this frame.
    pub fn is_just_pressed(&self, name: &str) -> bool {
        self.states.get(name) == Some(&ActionState::Started)
    }

    /// Check if an action was just released this frame.
    pub fn is_just_released(&self, name: &str) -> bool {
        self.states.get(name) == Some(&ActionState::Completed)
    }

    /// Get the analog axis value for an action.
    pub fn axis_value(&self, name: &str) -> f32 {
        self.axis_values.get(name).copied().unwrap_or(0.0)
    }

    /// Get a 2D axis value from two named actions.
    pub fn axis_value_2d(&self, name_x: &str, name_y: &str) -> Vec2 {
        Vec2 {
            x: self.axis_value(name_x),
            y: self.axis_value(name_y),
        }
    }

    /// Get the current state of an action.
    pub fn action_state(&self, name: &str) -> ActionState {
        self.states.get(name).copied().unwrap_or(ActionState::Idle)
    }

    /// Get all action definitions.
    pub fn actions(&self) -> &[InputActionDef] {
        &self.actions
    }

    /// Find an action definition by name.
    pub fn find_action(&self, name: &str) -> Option<&InputActionDef> {
        self.actions.iter().find(|a| a.name == name)
    }

    /// Find a mutable action definition by name.
    pub fn find_action_mut(&mut self, name: &str) -> Option<&mut InputActionDef> {
        self.actions.iter_mut().find(|a| a.name == name)
    }

    /// Update all action states from raw input. Called once per frame.
    pub fn update(&mut self, input: &InputSnapshot) {
        for action in &self.actions {
            let was_active = matches!(
                self.states.get(&action.name),
                Some(ActionState::Started) | Some(ActionState::Ongoing)
            );

            let (is_active, axis_val) = evaluate_bindings(&action.bindings, input, action.dead_zone);
            let scaled_axis = axis_val * action.sensitivity;

            let new_state = match (was_active, is_active) {
                (false, true) => ActionState::Started,
                (true, true) => ActionState::Ongoing,
                (true, false) => ActionState::Completed,
                (false, false) => ActionState::Idle,
            };

            self.states.insert(action.name.clone(), new_state);
            self.axis_values.insert(action.name.clone(), scaled_axis);
        }
    }

    /// Reset all states to Idle.
    pub fn reset_states(&mut self) {
        for state in self.states.values_mut() {
            *state = ActionState::Idle;
        }
        for val in self.axis_values.values_mut() {
            *val = 0.0;
        }
    }

    /// Rebind an action: replace all bindings with a new one.
    pub fn rebind_action(&mut self, name: &str, new_binding: InputBinding) {
        if let Some(action) = self.find_action_mut(name) {
            action.bindings.clear();
            action.bindings.push(new_binding);
        }
    }

    /// Add a binding to an existing action.
    pub fn add_binding(&mut self, name: &str, binding: InputBinding) {
        if let Some(action) = self.find_action_mut(name) {
            action.bindings.push(binding);
        }
    }
}

// ---------------------------------------------------------------------------
// Input Snapshot (simplified state for evaluation)
// ---------------------------------------------------------------------------

/// A snapshot of all input state used by the action mapping system.
/// This is a simplified view that the action system reads from.
#[derive(Debug, Clone, Default)]
pub struct InputSnapshot {
    /// Keys currently held.
    pub keys_held: Vec<KeyCode>,
    /// Keys pressed this frame.
    pub keys_pressed: Vec<KeyCode>,
    /// Keys released this frame.
    pub keys_released: Vec<KeyCode>,
    /// Mouse buttons held.
    pub mouse_buttons_held: Vec<MouseButton>,
    /// Mouse buttons pressed this frame.
    pub mouse_buttons_pressed: Vec<MouseButton>,
    /// Mouse buttons released this frame.
    pub mouse_buttons_released: Vec<MouseButton>,
    /// Mouse movement delta.
    pub mouse_delta: (f64, f64),
    /// Scroll delta.
    pub scroll_delta: (f64, f64),
    /// Gamepad buttons held (gamepad_id -> buttons).
    pub gamepad_buttons: HashMap<u32, Vec<GamepadButton>>,
    /// Gamepad axis values (gamepad_id -> (axis, value)).
    pub gamepad_axes: HashMap<u32, Vec<(GamepadAxis, f32)>>,
    /// Frame timestamp (seconds since start).
    pub timestamp: f64,
}

impl InputSnapshot {
    pub fn is_key_held(&self, key: KeyCode) -> bool {
        self.keys_held.contains(&key)
    }

    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }

    pub fn is_mouse_held(&self, button: MouseButton) -> bool {
        self.mouse_buttons_held.contains(&button)
    }

    pub fn is_mouse_pressed(&self, button: MouseButton) -> bool {
        self.mouse_buttons_pressed.contains(&button)
    }

    pub fn is_gamepad_held(&self, button: GamepadButton) -> bool {
        self.gamepad_buttons
            .values()
            .any(|buttons| buttons.contains(&button))
    }

    pub fn gamepad_axis_value(&self, axis: GamepadAxis) -> f32 {
        for axes in self.gamepad_axes.values() {
            for (a, v) in axes {
                if *a == axis {
                    return *v;
                }
            }
        }
        0.0
    }
}

/// Evaluate a set of bindings against the current input state.
/// Returns (is_active, axis_value).
fn evaluate_bindings(bindings: &[InputBinding], input: &InputSnapshot, dead_zone: f32) -> (bool, f32) {
    let mut any_active = false;
    let mut max_axis = 0.0_f32;

    for binding in bindings {
        let (active, axis) = evaluate_single_binding(binding, input, dead_zone);
        if active {
            any_active = true;
        }
        if axis.abs() > max_axis.abs() {
            max_axis = axis;
        }
    }

    (any_active, max_axis)
}

/// Evaluate a single binding.
fn evaluate_single_binding(
    binding: &InputBinding,
    input: &InputSnapshot,
    dead_zone: f32,
) -> (bool, f32) {
    match binding {
        InputBinding::Key(key) => {
            let held = input.is_key_held(*key);
            (held, if held { 1.0 } else { 0.0 })
        }
        InputBinding::MouseButton(button) => {
            let held = input.is_mouse_held(*button);
            (held, if held { 1.0 } else { 0.0 })
        }
        InputBinding::MouseAxis(axis) => {
            let value = match axis {
                MouseAxisType::X => input.mouse_delta.0 as f32,
                MouseAxisType::Y => input.mouse_delta.1 as f32,
                MouseAxisType::ScrollY => input.scroll_delta.1 as f32,
                MouseAxisType::ScrollX => input.scroll_delta.0 as f32,
            };
            let active = value.abs() > dead_zone;
            (active, if active { value } else { 0.0 })
        }
        InputBinding::GamepadButton(button) => {
            let held = input.is_gamepad_held(*button);
            (held, if held { 1.0 } else { 0.0 })
        }
        InputBinding::GamepadAxis(axis) => {
            let value = input.gamepad_axis_value(*axis);
            let active = value.abs() > dead_zone;
            (active, if active { value } else { 0.0 })
        }
        InputBinding::Combo(bindings) => {
            // All sub-bindings must be active simultaneously.
            let mut all_active = true;
            let mut max_axis = 0.0_f32;
            for sub in bindings {
                let (active, axis) = evaluate_single_binding(sub, input, dead_zone);
                if !active {
                    all_active = false;
                    break;
                }
                if axis.abs() > max_axis.abs() {
                    max_axis = axis;
                }
            }
            (all_active, if all_active { max_axis } else { 0.0 })
        }
        InputBinding::Sequence(_steps) => {
            // Sequence evaluation requires stateful tracking, handled by SequenceTracker.
            (false, 0.0)
        }
        InputBinding::Modified { modifier, key } => {
            let (mod_active, _) = evaluate_single_binding(modifier, input, dead_zone);
            let (key_active, axis) = evaluate_single_binding(key, input, dead_zone);
            let active = mod_active && key_active;
            (active, if active { axis } else { 0.0 })
        }
    }
}

// ---------------------------------------------------------------------------
// Input Context (layered action maps)
// ---------------------------------------------------------------------------

/// Manages a stack of input action maps with priority ordering.
/// Higher (later pushed) contexts take priority over lower ones.
#[derive(Debug, Clone)]
pub struct InputContext {
    /// Stack of action maps (topmost = highest priority).
    context_stack: Vec<InputActionMap>,
    /// Set of consumed action names this frame (from higher contexts).
    consumed: Vec<String>,
}

impl InputContext {
    /// Create a new empty input context.
    pub fn new() -> Self {
        Self {
            context_stack: Vec::new(),
            consumed: Vec::new(),
        }
    }

    /// Push an action map onto the context stack.
    pub fn push(&mut self, map: InputActionMap) {
        self.context_stack.push(map);
    }

    /// Pop the topmost action map.
    pub fn pop(&mut self) -> Option<InputActionMap> {
        self.context_stack.pop()
    }

    /// Remove an action map by name.
    pub fn remove(&mut self, name: &str) -> Option<InputActionMap> {
        if let Some(pos) = self.context_stack.iter().position(|m| m.name == name) {
            Some(self.context_stack.remove(pos))
        } else {
            None
        }
    }

    /// Get the topmost action map.
    pub fn top(&self) -> Option<&InputActionMap> {
        self.context_stack.last()
    }

    /// Get a mutable reference to the topmost action map.
    pub fn top_mut(&mut self) -> Option<&mut InputActionMap> {
        self.context_stack.last_mut()
    }

    /// Get an action map by name.
    pub fn get(&self, name: &str) -> Option<&InputActionMap> {
        self.context_stack.iter().find(|m| m.name == name)
    }

    /// Get a mutable action map by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut InputActionMap> {
        self.context_stack.iter_mut().find(|m| m.name == name)
    }

    /// Number of contexts in the stack.
    pub fn depth(&self) -> usize {
        self.context_stack.len()
    }

    /// Update all action maps from input, respecting consumption.
    /// Higher contexts consume inputs first.
    pub fn update(&mut self, input: &InputSnapshot) {
        self.consumed.clear();

        // Update from top to bottom.
        for i in (0..self.context_stack.len()).rev() {
            let map = &mut self.context_stack[i];
            if !map.enabled {
                continue;
            }
            map.update(input);

            // Check for consumed actions.
            for action in &map.actions {
                if action.consume && map.is_pressed(&action.name) {
                    self.consumed.push(action.name.clone());
                }
            }
        }
    }

    /// Check if an action is pressed in any active context.
    /// Respects consumption: a consumed action in a higher context blocks
    /// the same action name in lower contexts.
    pub fn is_pressed(&self, name: &str) -> bool {
        let mut found_consumer = false;
        for map in self.context_stack.iter().rev() {
            if !map.enabled {
                continue;
            }
            if map.is_pressed(name) {
                if let Some(action) = map.find_action(name) {
                    if action.consume {
                        return true;
                    }
                }
                if !found_consumer {
                    return true;
                }
            }
            // Check if a higher context consumed this action name.
            if self.consumed.contains(&name.to_string()) && !found_consumer {
                found_consumer = true;
            }
        }
        false
    }

    /// Check if an action was just pressed in any active context.
    pub fn is_just_pressed(&self, name: &str) -> bool {
        for map in self.context_stack.iter().rev() {
            if !map.enabled {
                continue;
            }
            if map.is_just_pressed(name) {
                return true;
            }
        }
        false
    }

    /// Check if an action was just released in any active context.
    pub fn is_just_released(&self, name: &str) -> bool {
        for map in self.context_stack.iter().rev() {
            if !map.enabled {
                continue;
            }
            if map.is_just_released(name) {
                return true;
            }
        }
        false
    }

    /// Get the axis value for an action from the highest-priority context.
    pub fn axis_value(&self, name: &str) -> f32 {
        for map in self.context_stack.iter().rev() {
            if !map.enabled {
                continue;
            }
            let val = map.axis_value(name);
            if val.abs() > 1e-6 {
                return val;
            }
        }
        0.0
    }

    /// Get a 2D axis value.
    pub fn axis_value_2d(&self, name_x: &str, name_y: &str) -> Vec2 {
        Vec2 {
            x: self.axis_value(name_x),
            y: self.axis_value(name_y),
        }
    }

    /// Cancel all active actions in all contexts.
    pub fn cancel_all(&mut self) {
        for map in &mut self.context_stack {
            map.reset_states();
        }
    }
}

impl Default for InputContext {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Sequence Tracker (combo detection)
// ---------------------------------------------------------------------------

/// Tracks timed input sequences for fighting-game-style combo detection.
#[derive(Debug, Clone)]
pub struct SequenceTracker {
    /// Active sequence being tracked.
    sequences: Vec<SequenceState>,
}

/// State of a single sequence being tracked.
#[derive(Debug, Clone)]
struct SequenceState {
    /// Name of the action this sequence belongs to.
    action_name: String,
    /// The sequence steps.
    steps: Vec<(InputBinding, f32)>,
    /// Current step index.
    current_step: usize,
    /// Time remaining for the current step.
    time_remaining: f32,
    /// Whether the sequence has been completed.
    completed: bool,
}

impl SequenceTracker {
    /// Create a new sequence tracker.
    pub fn new() -> Self {
        Self {
            sequences: Vec::new(),
        }
    }

    /// Register a sequence to track.
    pub fn register_sequence(
        &mut self,
        action_name: impl Into<String>,
        steps: Vec<(InputBinding, f32)>,
    ) {
        self.sequences.push(SequenceState {
            action_name: action_name.into(),
            steps,
            current_step: 0,
            time_remaining: 0.0,
            completed: false,
        });
    }

    /// Update all tracked sequences with the current input and elapsed time.
    /// Returns the names of sequences that completed this frame.
    pub fn update(&mut self, input: &InputSnapshot, dt: f32) -> Vec<String> {
        let mut completed = Vec::new();

        for seq in &mut self.sequences {
            if seq.completed {
                seq.completed = false;
            }

            if seq.current_step >= seq.steps.len() {
                // Already completed; reset.
                seq.current_step = 0;
                seq.time_remaining = 0.0;
                continue;
            }

            let (ref binding, time_window) = seq.steps[seq.current_step];

            if seq.current_step == 0 {
                // Waiting for the first input.
                let (active, _) = evaluate_single_binding(binding, input, 0.1);
                if active {
                    seq.current_step = 1;
                    if seq.current_step >= seq.steps.len() {
                        // Single-step sequence: completed.
                        seq.completed = true;
                        completed.push(seq.action_name.clone());
                        seq.current_step = 0;
                    } else {
                        seq.time_remaining = seq.steps[seq.current_step].1;
                    }
                }
            } else {
                // Waiting for subsequent inputs within the time window.
                seq.time_remaining -= dt;
                if seq.time_remaining <= 0.0 {
                    // Timed out: reset.
                    seq.current_step = 0;
                    seq.time_remaining = 0.0;
                    continue;
                }

                let (active, _) = evaluate_single_binding(binding, input, 0.1);
                if active {
                    seq.current_step += 1;
                    if seq.current_step >= seq.steps.len() {
                        // Sequence completed.
                        seq.completed = true;
                        completed.push(seq.action_name.clone());
                        seq.current_step = 0;
                        seq.time_remaining = 0.0;
                    } else {
                        seq.time_remaining = seq.steps[seq.current_step].1;
                    }
                }
            }
        }

        completed
    }

    /// Check if a sequence was completed this frame.
    pub fn was_completed(&self, action_name: &str) -> bool {
        self.sequences
            .iter()
            .any(|s| s.action_name == action_name && s.completed)
    }

    /// Reset all sequence tracking.
    pub fn reset(&mut self) {
        for seq in &mut self.sequences {
            seq.current_step = 0;
            seq.time_remaining = 0.0;
            seq.completed = false;
        }
    }
}

impl Default for SequenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Input Recorder
// ---------------------------------------------------------------------------

/// Records timestamped input events for replay, testing, or demo recording.
#[derive(Debug, Clone)]
pub struct InputRecorder {
    /// Recorded input frames.
    frames: Vec<RecordedFrame>,
    /// Whether recording is active.
    pub recording: bool,
    /// Time since recording started.
    elapsed: f64,
}

/// A single frame of recorded input.
#[derive(Debug, Clone)]
pub struct RecordedFrame {
    /// Timestamp (seconds from recording start).
    pub timestamp: f64,
    /// Input snapshot for this frame.
    pub snapshot: InputSnapshot,
}

impl InputRecorder {
    /// Create a new recorder.
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            recording: false,
            elapsed: 0.0,
        }
    }

    /// Start recording.
    pub fn start_recording(&mut self) {
        self.frames.clear();
        self.elapsed = 0.0;
        self.recording = true;
    }

    /// Record a frame of input.
    pub fn record_frame(&mut self, snapshot: InputSnapshot, dt: f64) {
        if !self.recording {
            return;
        }
        self.elapsed += dt;
        self.frames.push(RecordedFrame {
            timestamp: self.elapsed,
            snapshot,
        });
    }

    /// Stop recording.
    pub fn stop_recording(&mut self) {
        self.recording = false;
    }

    /// Get the recorded frames.
    pub fn frames(&self) -> &[RecordedFrame] {
        &self.frames
    }

    /// Number of recorded frames.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Total recording duration in seconds.
    pub fn duration(&self) -> f64 {
        self.frames.last().map(|f| f.timestamp).unwrap_or(0.0)
    }

    /// Clear recorded data.
    pub fn clear(&mut self) {
        self.frames.clear();
        self.elapsed = 0.0;
    }

    /// Serialize to a simple binary-compatible format (JSON for now).
    pub fn save_to_json(&self) -> String {
        // Simplified: just store frame count and duration.
        format!(
            "{{\"frame_count\":{},\"duration\":{}}}",
            self.frames.len(),
            self.duration()
        )
    }
}

impl Default for InputRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Plays back recorded input.
#[derive(Debug, Clone)]
pub struct InputPlayback {
    /// Frames to play back.
    frames: Vec<RecordedFrame>,
    /// Current playback position.
    current_frame: usize,
    /// Elapsed time since playback started.
    elapsed: f64,
    /// Whether playback is active.
    pub playing: bool,
    /// Whether to loop playback.
    pub looping: bool,
    /// Playback speed multiplier.
    pub speed: f64,
}

impl InputPlayback {
    /// Create a new playback from recorded frames.
    pub fn new(frames: Vec<RecordedFrame>) -> Self {
        Self {
            frames,
            current_frame: 0,
            elapsed: 0.0,
            playing: false,
            looping: false,
            speed: 1.0,
        }
    }

    /// Start playback.
    pub fn play(&mut self) {
        self.current_frame = 0;
        self.elapsed = 0.0;
        self.playing = true;
    }

    /// Stop playback.
    pub fn stop(&mut self) {
        self.playing = false;
    }

    /// Advance playback and return the current frame's input snapshot, if any.
    pub fn update(&mut self, dt: f64) -> Option<&InputSnapshot> {
        if !self.playing || self.frames.is_empty() {
            return None;
        }

        self.elapsed += dt * self.speed;

        // Find the frame at the current timestamp.
        while self.current_frame < self.frames.len() {
            if self.frames[self.current_frame].timestamp <= self.elapsed {
                let frame = &self.frames[self.current_frame];
                self.current_frame += 1;
                return Some(&frame.snapshot);
            } else {
                break;
            }
        }

        // Check for end of recording.
        if self.current_frame >= self.frames.len() {
            if self.looping {
                self.current_frame = 0;
                self.elapsed = 0.0;
            } else {
                self.playing = false;
            }
        }

        None
    }

    /// Whether playback has finished.
    pub fn is_finished(&self) -> bool {
        !self.playing && self.current_frame >= self.frames.len()
    }
}

// ---------------------------------------------------------------------------
// Gesture Recognizer (touch input)
// ---------------------------------------------------------------------------

/// Types of touch gestures that can be recognized.
#[derive(Debug, Clone, PartialEq)]
pub enum GestureType {
    /// Single tap.
    Tap { position: Vec2 },
    /// Double tap.
    DoubleTap { position: Vec2 },
    /// Long press.
    LongPress { position: Vec2, duration: f32 },
    /// Swipe in a direction.
    Swipe {
        direction: SwipeDirection,
        velocity: f32,
        start: Vec2,
        end: Vec2,
    },
    /// Pinch (two-finger zoom).
    Pinch {
        center: Vec2,
        scale: f32,
        delta_scale: f32,
    },
    /// Two-finger rotation.
    Rotate {
        center: Vec2,
        angle: f32,
        delta_angle: f32,
    },
    /// Custom gesture pattern matched.
    Custom { pattern_name: String },
}

/// Direction of a swipe gesture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SwipeDirection {
    Up,
    Down,
    Left,
    Right,
}

/// Internal tracking of touch points for gesture recognition.
#[derive(Debug, Clone)]
struct TouchTrack {
    /// Touch ID.
    id: u64,
    /// Starting position.
    start_pos: Vec2,
    /// Current position.
    current_pos: Vec2,
    /// Time the touch started.
    start_time: f64,
    /// Whether this touch has moved significantly.
    moved: bool,
}

/// Recognizes touch gestures from raw touch input.
#[derive(Debug, Clone)]
pub struct GestureRecognizer {
    /// Active touch tracking.
    touches: Vec<TouchTrack>,
    /// Completed gestures waiting to be consumed.
    pending_gestures: VecDeque<GestureType>,
    /// Minimum distance (pixels) for a swipe to be recognized.
    pub swipe_threshold: f32,
    /// Maximum time (seconds) for a tap.
    pub tap_max_duration: f32,
    /// Maximum time (seconds) between taps for a double-tap.
    pub double_tap_window: f32,
    /// Minimum time (seconds) for a long press.
    pub long_press_duration: f32,
    /// Minimum movement (pixels) before a tap becomes a drag/swipe.
    pub move_threshold: f32,
    /// Last tap time and position for double-tap detection.
    last_tap: Option<(f64, Vec2)>,
    /// Custom gesture patterns.
    custom_patterns: Vec<CustomGesturePattern>,
    /// Current accumulated path for custom gesture matching.
    current_path: Vec<Vec2>,
}

/// A custom gesture pattern defined by a sequence of directional segments.
#[derive(Debug, Clone)]
pub struct CustomGesturePattern {
    /// Pattern name.
    pub name: String,
    /// Sequence of direction segments.
    pub directions: Vec<SwipeDirection>,
    /// Tolerance angle in degrees.
    pub tolerance: f32,
}

impl Default for GestureRecognizer {
    fn default() -> Self {
        Self {
            touches: Vec::new(),
            pending_gestures: VecDeque::new(),
            swipe_threshold: 50.0,
            tap_max_duration: 0.3,
            double_tap_window: 0.4,
            long_press_duration: 0.5,
            move_threshold: 10.0,
            last_tap: None,
            custom_patterns: Vec::new(),
            current_path: Vec::new(),
        }
    }
}

impl GestureRecognizer {
    /// Create a new gesture recognizer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a custom gesture pattern.
    pub fn add_custom_pattern(&mut self, pattern: CustomGesturePattern) {
        self.custom_patterns.push(pattern);
    }

    /// Process a touch start event.
    pub fn on_touch_start(&mut self, id: u64, x: f32, y: f32, time: f64) {
        self.touches.push(TouchTrack {
            id,
            start_pos: Vec2::new(x, y),
            current_pos: Vec2::new(x, y),
            start_time: time,
            moved: false,
        });
        self.current_path.clear();
        self.current_path.push(Vec2::new(x, y));
    }

    /// Process a touch move event.
    pub fn on_touch_move(&mut self, id: u64, x: f32, y: f32) {
        if let Some(track) = self.touches.iter_mut().find(|t| t.id == id) {
            track.current_pos = Vec2::new(x, y);
            let dx = x - track.start_pos.x;
            let dy = y - track.start_pos.y;
            if (dx * dx + dy * dy).sqrt() > self.move_threshold {
                track.moved = true;
            }
        }

        // Record path for custom gesture detection.
        if let Some(last) = self.current_path.last() {
            let dx = x - last.x;
            let dy = y - last.y;
            if (dx * dx + dy * dy).sqrt() > 5.0 {
                self.current_path.push(Vec2::new(x, y));
            }
        }

        // Check for pinch/rotate with two fingers.
        if self.touches.len() == 2 {
            self.check_pinch_rotate();
        }
    }

    /// Process a touch end event.
    pub fn on_touch_end(&mut self, id: u64, time: f64) {
        let track = self.touches.iter().find(|t| t.id == id).cloned();

        if let Some(track) = track {
            let duration = (time - track.start_time) as f32;

            if !track.moved {
                // Could be a tap or long press.
                if duration < self.tap_max_duration {
                    // Check for double tap.
                    if let Some((last_time, last_pos)) = self.last_tap {
                        let dt = time - last_time;
                        let dx = track.start_pos.x - last_pos.x;
                        let dy = track.start_pos.y - last_pos.y;
                        let dist = (dx * dx + dy * dy).sqrt();

                        if dt < self.double_tap_window as f64 && dist < self.move_threshold * 2.0
                        {
                            self.pending_gestures.push_back(GestureType::DoubleTap {
                                position: track.start_pos,
                            });
                            self.last_tap = None;
                        } else {
                            self.pending_gestures.push_back(GestureType::Tap {
                                position: track.start_pos,
                            });
                            self.last_tap = Some((time, track.start_pos));
                        }
                    } else {
                        self.pending_gestures.push_back(GestureType::Tap {
                            position: track.start_pos,
                        });
                        self.last_tap = Some((time, track.start_pos));
                    }
                } else if duration >= self.long_press_duration {
                    self.pending_gestures.push_back(GestureType::LongPress {
                        position: track.start_pos,
                        duration,
                    });
                }
            } else {
                // Moved: check for swipe.
                let dx = track.current_pos.x - track.start_pos.x;
                let dy = track.current_pos.y - track.start_pos.y;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist >= self.swipe_threshold {
                    let velocity = dist / duration.max(0.001);
                    let direction = if dx.abs() > dy.abs() {
                        if dx > 0.0 {
                            SwipeDirection::Right
                        } else {
                            SwipeDirection::Left
                        }
                    } else if dy > 0.0 {
                        SwipeDirection::Down
                    } else {
                        SwipeDirection::Up
                    };

                    self.pending_gestures.push_back(GestureType::Swipe {
                        direction,
                        velocity,
                        start: track.start_pos,
                        end: track.current_pos,
                    });
                }

                // Check custom gesture patterns.
                self.check_custom_patterns();
            }
        }

        self.touches.retain(|t| t.id != id);
    }

    /// Check for pinch and rotate gestures with two active touches.
    fn check_pinch_rotate(&mut self) {
        if self.touches.len() != 2 {
            return;
        }

        let t0 = &self.touches[0];
        let t1 = &self.touches[1];

        let center = Vec2::new(
            (t0.current_pos.x + t1.current_pos.x) * 0.5,
            (t0.current_pos.y + t1.current_pos.y) * 0.5,
        );

        // Current distance between fingers.
        let dx = t1.current_pos.x - t0.current_pos.x;
        let dy = t1.current_pos.y - t0.current_pos.y;
        let current_dist = (dx * dx + dy * dy).sqrt();

        // Starting distance between fingers.
        let sdx = t1.start_pos.x - t0.start_pos.x;
        let sdy = t1.start_pos.y - t0.start_pos.y;
        let start_dist = (sdx * sdx + sdy * sdy).sqrt();

        if start_dist > 1.0 {
            let scale = current_dist / start_dist;
            let delta_scale = scale - 1.0;

            if delta_scale.abs() > 0.01 {
                self.pending_gestures.push_back(GestureType::Pinch {
                    center,
                    scale,
                    delta_scale,
                });
            }
        }

        // Rotation.
        let current_angle = dy.atan2(dx);
        let start_angle = sdy.atan2(sdx);
        let delta_angle = current_angle - start_angle;

        if delta_angle.abs() > 0.01 {
            self.pending_gestures.push_back(GestureType::Rotate {
                center,
                angle: current_angle,
                delta_angle,
            });
        }
    }

    /// Check accumulated path against custom gesture patterns.
    fn check_custom_patterns(&mut self) {
        if self.current_path.len() < 3 || self.custom_patterns.is_empty() {
            return;
        }

        // Convert path to a sequence of direction segments.
        let directions = path_to_directions(&self.current_path);

        for pattern in &self.custom_patterns {
            if directions_match(&directions, &pattern.directions) {
                self.pending_gestures.push_back(GestureType::Custom {
                    pattern_name: pattern.name.clone(),
                });
            }
        }

        self.current_path.clear();
    }

    /// Poll the next recognized gesture.
    pub fn poll_gesture(&mut self) -> Option<GestureType> {
        self.pending_gestures.pop_front()
    }

    /// Check if there are pending gestures.
    pub fn has_pending(&self) -> bool {
        !self.pending_gestures.is_empty()
    }

    /// Number of currently active touches.
    pub fn active_touch_count(&self) -> usize {
        self.touches.len()
    }

    /// Clear all state.
    pub fn clear(&mut self) {
        self.touches.clear();
        self.pending_gestures.clear();
        self.last_tap = None;
        self.current_path.clear();
    }
}

/// Convert a path of points into a sequence of cardinal directions.
fn path_to_directions(path: &[Vec2]) -> Vec<SwipeDirection> {
    let mut directions = Vec::new();
    let mut last_dir: Option<SwipeDirection> = None;

    for window in path.windows(2) {
        let dx = window[1].x - window[0].x;
        let dy = window[1].y - window[0].y;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < 5.0 {
            continue;
        }

        let dir = if dx.abs() > dy.abs() {
            if dx > 0.0 {
                SwipeDirection::Right
            } else {
                SwipeDirection::Left
            }
        } else if dy > 0.0 {
            SwipeDirection::Down
        } else {
            SwipeDirection::Up
        };

        // Only add if different from the last direction.
        if last_dir != Some(dir) {
            directions.push(dir);
            last_dir = Some(dir);
        }
    }

    directions
}

/// Check if a direction sequence matches a pattern.
fn directions_match(actual: &[SwipeDirection], pattern: &[SwipeDirection]) -> bool {
    if actual.len() != pattern.len() {
        return false;
    }
    actual.iter().zip(pattern.iter()).all(|(a, p)| a == p)
}

// ---------------------------------------------------------------------------
// Rebinding UI Support
// ---------------------------------------------------------------------------

/// State for the input rebinding UI.
#[derive(Debug, Clone)]
pub struct RebindingState {
    /// The action being rebound.
    pub action_name: String,
    /// The binding index being replaced.
    pub binding_index: usize,
    /// Whether we are waiting for user input.
    pub waiting_for_input: bool,
    /// Detected input (set when the user presses something).
    pub detected_binding: Option<InputBinding>,
    /// Timeout in seconds.
    pub timeout: f32,
    /// Elapsed time waiting.
    pub elapsed: f32,
}

impl RebindingState {
    /// Create a new rebinding state.
    pub fn new(action_name: impl Into<String>, binding_index: usize) -> Self {
        Self {
            action_name: action_name.into(),
            binding_index,
            waiting_for_input: true,
            detected_binding: None,
            timeout: 5.0,
            elapsed: 0.0,
        }
    }

    /// Update the rebinding detection. Returns `true` if a binding was detected.
    pub fn update(&mut self, input: &InputSnapshot, dt: f32) -> bool {
        if !self.waiting_for_input {
            return false;
        }

        self.elapsed += dt;
        if self.elapsed >= self.timeout {
            self.waiting_for_input = false;
            return false;
        }

        // Detect keyboard input.
        for key in &input.keys_pressed {
            // Skip modifier keys as standalone bindings.
            if matches!(
                key,
                KeyCode::LShift | KeyCode::RShift | KeyCode::LControl | KeyCode::RControl
                    | KeyCode::LAlt | KeyCode::RAlt
            ) {
                continue;
            }
            self.detected_binding = Some(InputBinding::Key(*key));
            self.waiting_for_input = false;
            return true;
        }

        // Detect mouse button input.
        for button in &input.mouse_buttons_pressed {
            self.detected_binding = Some(InputBinding::MouseButton(*button));
            self.waiting_for_input = false;
            return true;
        }

        // Detect gamepad button input.
        for buttons in input.gamepad_buttons.values() {
            for button in buttons {
                self.detected_binding = Some(InputBinding::GamepadButton(*button));
                self.waiting_for_input = false;
                return true;
            }
        }

        // Detect gamepad axis (significant movement).
        for axes in input.gamepad_axes.values() {
            for (axis, value) in axes {
                if value.abs() > 0.5 {
                    self.detected_binding = Some(InputBinding::GamepadAxis(*axis));
                    self.waiting_for_input = false;
                    return true;
                }
            }
        }

        false
    }

    /// Whether the detection timed out.
    pub fn timed_out(&self) -> bool {
        !self.waiting_for_input && self.detected_binding.is_none()
    }
}

// ---------------------------------------------------------------------------
// Default Action Presets
// ---------------------------------------------------------------------------

/// Create a default FPS action map.
pub fn fps_preset() -> InputActionMap {
    let mut map = InputActionMap::new("FPS");
    map.define_action(
        InputActionDef::new("MoveForward")
            .with_binding(InputBinding::key(KeyCode::W))
            .with_binding(InputBinding::gamepad_axis(GamepadAxis::LeftStickY)),
    );
    map.define_action(
        InputActionDef::new("MoveBackward")
            .with_binding(InputBinding::key(KeyCode::S)),
    );
    map.define_action(
        InputActionDef::new("MoveLeft")
            .with_binding(InputBinding::key(KeyCode::A)),
    );
    map.define_action(
        InputActionDef::new("MoveRight")
            .with_binding(InputBinding::key(KeyCode::D))
            .with_binding(InputBinding::gamepad_axis(GamepadAxis::LeftStickX)),
    );
    map.define_action(
        InputActionDef::new("Jump")
            .with_binding(InputBinding::key(KeyCode::Space))
            .with_binding(InputBinding::gamepad_button(GamepadButton::South)),
    );
    map.define_action(
        InputActionDef::new("Crouch")
            .with_binding(InputBinding::key(KeyCode::LControl))
            .with_binding(InputBinding::gamepad_button(GamepadButton::East)),
    );
    map.define_action(
        InputActionDef::new("Sprint")
            .with_binding(InputBinding::key(KeyCode::LShift))
            .with_binding(InputBinding::gamepad_button(GamepadButton::LeftStickPress)),
    );
    map.define_action(
        InputActionDef::new("Fire")
            .with_binding(InputBinding::mouse_button(MouseButton::Left))
            .with_binding(InputBinding::gamepad_axis(GamepadAxis::RightTrigger))
            .with_dead_zone(0.1),
    );
    map.define_action(
        InputActionDef::new("Aim")
            .with_binding(InputBinding::mouse_button(MouseButton::Right))
            .with_binding(InputBinding::gamepad_axis(GamepadAxis::LeftTrigger))
            .with_dead_zone(0.1),
    );
    map.define_action(
        InputActionDef::new("Reload")
            .with_binding(InputBinding::key(KeyCode::R))
            .with_binding(InputBinding::gamepad_button(GamepadButton::West)),
    );
    map.define_action(
        InputActionDef::new("Interact")
            .with_binding(InputBinding::key(KeyCode::E))
            .with_binding(InputBinding::gamepad_button(GamepadButton::North)),
    );
    map.define_action(
        InputActionDef::new("LookX")
            .with_binding(InputBinding::mouse_axis(MouseAxisType::X))
            .with_binding(InputBinding::gamepad_axis(GamepadAxis::RightStickX))
            .with_sensitivity(0.3),
    );
    map.define_action(
        InputActionDef::new("LookY")
            .with_binding(InputBinding::mouse_axis(MouseAxisType::Y))
            .with_binding(InputBinding::gamepad_axis(GamepadAxis::RightStickY))
            .with_sensitivity(0.3),
    );
    map
}

/// Create a default third-person action map.
pub fn third_person_preset() -> InputActionMap {
    let mut map = fps_preset();
    map.name = "ThirdPerson".into();
    map.define_action(
        InputActionDef::new("Dodge")
            .with_binding(InputBinding::combo(vec![
                InputBinding::key(KeyCode::LShift),
                InputBinding::key(KeyCode::Space),
            ]))
            .with_binding(InputBinding::gamepad_button(GamepadButton::RightBumper)),
    );
    map.define_action(
        InputActionDef::new("LockOn")
            .with_binding(InputBinding::key(KeyCode::Tab))
            .with_binding(InputBinding::gamepad_button(GamepadButton::RightStickPress)),
    );
    map
}

/// Create a default RTS action map.
pub fn rts_preset() -> InputActionMap {
    let mut map = InputActionMap::new("RTS");
    map.define_action(
        InputActionDef::new("Select")
            .with_binding(InputBinding::mouse_button(MouseButton::Left)),
    );
    map.define_action(
        InputActionDef::new("Command")
            .with_binding(InputBinding::mouse_button(MouseButton::Right)),
    );
    map.define_action(
        InputActionDef::new("PanCamera")
            .with_binding(InputBinding::mouse_button(MouseButton::Middle)),
    );
    map.define_action(
        InputActionDef::new("ZoomIn")
            .with_binding(InputBinding::mouse_axis(MouseAxisType::ScrollY))
            .with_sensitivity(1.0),
    );
    map.define_action(
        InputActionDef::new("SelectAll")
            .with_binding(InputBinding::modified(
                InputBinding::key(KeyCode::LControl),
                InputBinding::key(KeyCode::A),
            )),
    );
    map.define_action(
        InputActionDef::new("GroupAssign")
            .with_binding(InputBinding::modified(
                InputBinding::key(KeyCode::LControl),
                InputBinding::key(KeyCode::Key1),
            )),
    );
    map
}

/// Create a default menu navigation action map.
pub fn menu_preset() -> InputActionMap {
    let mut map = InputActionMap::new("Menu");
    map.define_action(
        InputActionDef::new("Confirm")
            .with_binding(InputBinding::key(KeyCode::Enter))
            .with_binding(InputBinding::gamepad_button(GamepadButton::South))
            .consuming(),
    );
    map.define_action(
        InputActionDef::new("Cancel")
            .with_binding(InputBinding::key(KeyCode::Escape))
            .with_binding(InputBinding::gamepad_button(GamepadButton::East))
            .consuming(),
    );
    map.define_action(
        InputActionDef::new("NavigateUp")
            .with_binding(InputBinding::key(KeyCode::Up))
            .with_binding(InputBinding::gamepad_button(GamepadButton::DPadUp))
            .consuming(),
    );
    map.define_action(
        InputActionDef::new("NavigateDown")
            .with_binding(InputBinding::key(KeyCode::Down))
            .with_binding(InputBinding::gamepad_button(GamepadButton::DPadDown))
            .consuming(),
    );
    map.define_action(
        InputActionDef::new("NavigateLeft")
            .with_binding(InputBinding::key(KeyCode::Left))
            .with_binding(InputBinding::gamepad_button(GamepadButton::DPadLeft))
            .consuming(),
    );
    map.define_action(
        InputActionDef::new("NavigateRight")
            .with_binding(InputBinding::key(KeyCode::Right))
            .with_binding(InputBinding::gamepad_button(GamepadButton::DPadRight))
            .consuming(),
    );
    map.define_action(
        InputActionDef::new("TabLeft")
            .with_binding(InputBinding::key(KeyCode::Q))
            .with_binding(InputBinding::gamepad_button(GamepadButton::LeftBumper)),
    );
    map.define_action(
        InputActionDef::new("TabRight")
            .with_binding(InputBinding::key(KeyCode::E))
            .with_binding(InputBinding::gamepad_button(GamepadButton::RightBumper)),
    );
    map
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input_with_key(key: KeyCode) -> InputSnapshot {
        InputSnapshot {
            keys_held: vec![key],
            keys_pressed: vec![key],
            ..Default::default()
        }
    }

    fn make_input_with_keys(keys: Vec<KeyCode>) -> InputSnapshot {
        InputSnapshot {
            keys_held: keys.clone(),
            keys_pressed: keys,
            ..Default::default()
        }
    }

    #[test]
    fn action_map_basic() {
        let mut map = InputActionMap::new("Test");
        map.define_action(
            InputActionDef::new("Jump")
                .with_binding(InputBinding::key(KeyCode::Space)),
        );

        // Not pressed.
        let empty = InputSnapshot::default();
        map.update(&empty);
        assert!(!map.is_pressed("Jump"));
        assert!(!map.is_just_pressed("Jump"));

        // Pressed.
        let pressed = make_input_with_key(KeyCode::Space);
        map.update(&pressed);
        assert!(map.is_pressed("Jump"));
        assert!(map.is_just_pressed("Jump"));

        // Held.
        map.update(&pressed);
        assert!(map.is_pressed("Jump"));
        assert!(!map.is_just_pressed("Jump")); // No longer "just" pressed.

        // Released.
        let empty = InputSnapshot::default();
        map.update(&empty);
        assert!(!map.is_pressed("Jump"));
        assert!(map.is_just_released("Jump"));
    }

    #[test]
    fn action_map_multiple_bindings() {
        let mut map = InputActionMap::new("Test");
        map.define_action(
            InputActionDef::new("Fire")
                .with_binding(InputBinding::key(KeyCode::F))
                .with_binding(InputBinding::mouse_button(MouseButton::Left)),
        );

        let key_input = make_input_with_key(KeyCode::F);
        map.update(&key_input);
        assert!(map.is_pressed("Fire"));

        map.update(&InputSnapshot::default());

        let mouse_input = InputSnapshot {
            mouse_buttons_held: vec![MouseButton::Left],
            mouse_buttons_pressed: vec![MouseButton::Left],
            ..Default::default()
        };
        map.update(&mouse_input);
        assert!(map.is_pressed("Fire"));
    }

    #[test]
    fn action_map_combo() {
        let mut map = InputActionMap::new("Test");
        map.define_action(
            InputActionDef::new("QuickSave")
                .with_binding(InputBinding::combo(vec![
                    InputBinding::key(KeyCode::LControl),
                    InputBinding::key(KeyCode::S),
                ])),
        );

        // Only Ctrl pressed.
        let ctrl_only = make_input_with_key(KeyCode::LControl);
        map.update(&ctrl_only);
        assert!(!map.is_pressed("QuickSave"));

        // Both pressed.
        let both = make_input_with_keys(vec![KeyCode::LControl, KeyCode::S]);
        map.update(&both);
        assert!(map.is_pressed("QuickSave"));
    }

    #[test]
    fn action_map_modified_binding() {
        let mut map = InputActionMap::new("Test");
        map.define_action(
            InputActionDef::new("Undo")
                .with_binding(InputBinding::modified(
                    InputBinding::key(KeyCode::LControl),
                    InputBinding::key(KeyCode::Z),
                )),
        );

        // Z alone should not trigger.
        let z_only = make_input_with_key(KeyCode::Z);
        map.update(&z_only);
        assert!(!map.is_pressed("Undo"));

        // Ctrl+Z should trigger.
        let ctrl_z = make_input_with_keys(vec![KeyCode::LControl, KeyCode::Z]);
        map.update(&ctrl_z);
        assert!(map.is_pressed("Undo"));
    }

    #[test]
    fn action_map_axis() {
        let mut map = InputActionMap::new("Test");
        map.define_action(
            InputActionDef::new("LookX")
                .with_binding(InputBinding::mouse_axis(MouseAxisType::X))
                .with_sensitivity(2.0)
                .with_dead_zone(0.0),
        );

        let input = InputSnapshot {
            mouse_delta: (10.0, 0.0),
            ..Default::default()
        };
        map.update(&input);
        assert!(map.axis_value("LookX") > 0.0);
        assert!((map.axis_value("LookX") - 20.0).abs() < 1e-3);
    }

    #[test]
    fn action_map_gamepad_axis() {
        let mut map = InputActionMap::new("Test");
        map.define_action(
            InputActionDef::new("MoveRight")
                .with_binding(InputBinding::gamepad_axis(GamepadAxis::LeftStickX))
                .with_dead_zone(0.1),
        );

        let mut axes = HashMap::new();
        axes.insert(0, vec![(GamepadAxis::LeftStickX, 0.8)]);
        let input = InputSnapshot {
            gamepad_axes: axes,
            ..Default::default()
        };
        map.update(&input);
        assert!(map.is_pressed("MoveRight"));
        assert!((map.axis_value("MoveRight") - 0.8).abs() < 1e-3);
    }

    #[test]
    fn axis_2d() {
        let mut map = InputActionMap::new("Test");
        map.define_action(
            InputActionDef::new("MoveX").with_binding(InputBinding::key(KeyCode::D)),
        );
        map.define_action(
            InputActionDef::new("MoveY").with_binding(InputBinding::key(KeyCode::W)),
        );

        let input = make_input_with_keys(vec![KeyCode::D, KeyCode::W]);
        map.update(&input);
        let v = map.axis_value_2d("MoveX", "MoveY");
        assert!((v.x - 1.0).abs() < 1e-3);
        assert!((v.y - 1.0).abs() < 1e-3);
    }

    #[test]
    fn input_context_stack() {
        let mut ctx = InputContext::new();

        let mut gameplay = InputActionMap::new("Gameplay");
        gameplay.define_action(
            InputActionDef::new("Jump").with_binding(InputBinding::key(KeyCode::Space)),
        );

        let mut menu = InputActionMap::new("Menu");
        menu.define_action(
            InputActionDef::new("Confirm")
                .with_binding(InputBinding::key(KeyCode::Enter))
                .consuming(),
        );

        ctx.push(gameplay);
        ctx.push(menu);

        assert_eq!(ctx.depth(), 2);

        let input = make_input_with_key(KeyCode::Space);
        ctx.update(&input);
        assert!(ctx.is_pressed("Jump"));

        let input = make_input_with_key(KeyCode::Enter);
        ctx.update(&input);
        assert!(ctx.is_pressed("Confirm"));
    }

    #[test]
    fn input_context_pop() {
        let mut ctx = InputContext::new();
        ctx.push(InputActionMap::new("A"));
        ctx.push(InputActionMap::new("B"));
        assert_eq!(ctx.depth(), 2);

        let popped = ctx.pop();
        assert_eq!(popped.unwrap().name, "B");
        assert_eq!(ctx.depth(), 1);
    }

    #[test]
    fn input_context_remove_by_name() {
        let mut ctx = InputContext::new();
        ctx.push(InputActionMap::new("Gameplay"));
        ctx.push(InputActionMap::new("Menu"));
        ctx.push(InputActionMap::new("Dialog"));

        let removed = ctx.remove("Menu");
        assert!(removed.is_some());
        assert_eq!(ctx.depth(), 2);
        assert!(ctx.get("Menu").is_none());
    }

    #[test]
    fn sequence_tracker_basic() {
        let mut tracker = SequenceTracker::new();
        tracker.register_sequence(
            "Hadouken",
            vec![
                (InputBinding::key(KeyCode::Down), 0.5),
                (InputBinding::key(KeyCode::Right), 0.5),
                (InputBinding::key(KeyCode::P), 0.5),
            ],
        );

        // Step 1: Down.
        let input1 = make_input_with_key(KeyCode::Down);
        let completed = tracker.update(&input1, 0.016);
        assert!(completed.is_empty());

        // Step 2: Right (within time window).
        let input2 = make_input_with_key(KeyCode::Right);
        let completed = tracker.update(&input2, 0.1);
        assert!(completed.is_empty());

        // Step 3: P (within time window).
        let input3 = make_input_with_key(KeyCode::P);
        let completed = tracker.update(&input3, 0.1);
        assert_eq!(completed, vec!["Hadouken"]);
    }

    #[test]
    fn sequence_tracker_timeout() {
        let mut tracker = SequenceTracker::new();
        tracker.register_sequence(
            "Combo",
            vec![
                (InputBinding::key(KeyCode::A), 0.3),
                (InputBinding::key(KeyCode::B), 0.3),
            ],
        );

        // Step 1: A.
        let input1 = make_input_with_key(KeyCode::A);
        tracker.update(&input1, 0.016);

        // Wait too long.
        let empty = InputSnapshot::default();
        tracker.update(&empty, 0.5);

        // Step 2: B (should fail - timed out).
        let input2 = make_input_with_key(KeyCode::B);
        let completed = tracker.update(&input2, 0.016);
        assert!(completed.is_empty());
    }

    #[test]
    fn input_recorder_basic() {
        let mut recorder = InputRecorder::new();
        recorder.start_recording();
        assert!(recorder.recording);

        let snapshot = make_input_with_key(KeyCode::W);
        recorder.record_frame(snapshot, 0.016);
        recorder.record_frame(InputSnapshot::default(), 0.016);

        recorder.stop_recording();
        assert!(!recorder.recording);
        assert_eq!(recorder.frame_count(), 2);
        assert!(recorder.duration() > 0.0);
    }

    #[test]
    fn input_playback_basic() {
        let frames = vec![
            RecordedFrame {
                timestamp: 0.0,
                snapshot: make_input_with_key(KeyCode::W),
            },
            RecordedFrame {
                timestamp: 0.016,
                snapshot: make_input_with_key(KeyCode::W),
            },
            RecordedFrame {
                timestamp: 0.032,
                snapshot: InputSnapshot::default(),
            },
        ];

        let mut playback = InputPlayback::new(frames);
        playback.play();
        assert!(playback.playing);

        let snap = playback.update(0.02);
        assert!(snap.is_some());
    }

    #[test]
    fn gesture_recognizer_tap() {
        let mut recognizer = GestureRecognizer::new();
        recognizer.on_touch_start(0, 100.0, 100.0, 0.0);
        recognizer.on_touch_end(0, 0.1);

        let gesture = recognizer.poll_gesture();
        assert!(gesture.is_some());
        match gesture.unwrap() {
            GestureType::Tap { position } => {
                assert!((position.x - 100.0).abs() < 1e-3);
            }
            _ => panic!("Expected Tap gesture"),
        }
    }

    #[test]
    fn gesture_recognizer_swipe() {
        let mut recognizer = GestureRecognizer::new();
        recognizer.swipe_threshold = 20.0;
        recognizer.move_threshold = 5.0;

        recognizer.on_touch_start(0, 100.0, 100.0, 0.0);
        recognizer.on_touch_move(0, 200.0, 100.0);
        recognizer.on_touch_end(0, 0.2);

        let gesture = recognizer.poll_gesture();
        assert!(gesture.is_some());
        match gesture.unwrap() {
            GestureType::Swipe { direction, .. } => {
                assert_eq!(direction, SwipeDirection::Right);
            }
            _ => panic!("Expected Swipe gesture"),
        }
    }

    #[test]
    fn gesture_recognizer_long_press() {
        let mut recognizer = GestureRecognizer::new();
        recognizer.long_press_duration = 0.5;
        recognizer.tap_max_duration = 0.3;

        recognizer.on_touch_start(0, 100.0, 100.0, 0.0);
        recognizer.on_touch_end(0, 0.6); // 0.6s > 0.5s threshold.

        let gesture = recognizer.poll_gesture();
        assert!(gesture.is_some());
        match gesture.unwrap() {
            GestureType::LongPress { duration, .. } => {
                assert!(duration >= 0.5);
            }
            _ => panic!("Expected LongPress gesture"),
        }
    }

    #[test]
    fn gesture_recognizer_pinch() {
        let mut recognizer = GestureRecognizer::new();

        recognizer.on_touch_start(0, 100.0, 100.0, 0.0);
        recognizer.on_touch_start(1, 200.0, 100.0, 0.0);

        // Move fingers apart.
        recognizer.on_touch_move(0, 50.0, 100.0);
        recognizer.on_touch_move(1, 250.0, 100.0);

        assert!(recognizer.has_pending());
        let gesture = recognizer.poll_gesture();
        assert!(gesture.is_some());
        match gesture.unwrap() {
            GestureType::Pinch { scale, .. } => {
                assert!(scale > 1.0); // Fingers moved apart = zoom in.
            }
            _ => panic!("Expected Pinch gesture"),
        }
    }

    #[test]
    fn rebinding_detects_key() {
        let mut state = RebindingState::new("Jump", 0);
        let input = make_input_with_key(KeyCode::Space);
        let detected = state.update(&input, 0.016);
        assert!(detected);
        assert!(!state.waiting_for_input);
        match &state.detected_binding {
            Some(InputBinding::Key(KeyCode::Space)) => {}
            _ => panic!("Expected Space key binding"),
        }
    }

    #[test]
    fn rebinding_timeout() {
        let mut state = RebindingState::new("Jump", 0);
        state.timeout = 1.0;

        let empty = InputSnapshot::default();
        for _ in 0..100 {
            state.update(&empty, 0.02);
        }

        assert!(state.timed_out());
        assert!(state.detected_binding.is_none());
    }

    #[test]
    fn rebind_action() {
        let mut map = InputActionMap::new("Test");
        map.define_action(
            InputActionDef::new("Jump").with_binding(InputBinding::key(KeyCode::Space)),
        );

        map.rebind_action("Jump", InputBinding::key(KeyCode::F));

        let input = make_input_with_key(KeyCode::F);
        map.update(&input);
        assert!(map.is_pressed("Jump"));

        let old_input = make_input_with_key(KeyCode::Space);
        map.update(&old_input);
        assert!(!map.is_pressed("Jump"));
    }

    #[test]
    fn fps_preset_has_actions() {
        let map = fps_preset();
        assert!(map.find_action("Jump").is_some());
        assert!(map.find_action("Fire").is_some());
        assert!(map.find_action("MoveForward").is_some());
        assert!(map.find_action("LookX").is_some());
    }

    #[test]
    fn third_person_preset_has_dodge() {
        let map = third_person_preset();
        assert!(map.find_action("Dodge").is_some());
        assert!(map.find_action("LockOn").is_some());
    }

    #[test]
    fn rts_preset_has_select() {
        let map = rts_preset();
        assert!(map.find_action("Select").is_some());
        assert!(map.find_action("SelectAll").is_some());
    }

    #[test]
    fn menu_preset_has_confirm() {
        let map = menu_preset();
        assert!(map.find_action("Confirm").is_some());
        assert!(map.find_action("Cancel").is_some());
        assert!(map.find_action("NavigateUp").is_some());
    }

    #[test]
    fn vec2_operations() {
        let v = Vec2::new(3.0, 4.0);
        assert!((v.length() - 5.0).abs() < 1e-5);

        let n = v.normalized();
        assert!((n.length() - 1.0).abs() < 1e-5);

        let zero = Vec2::default();
        assert!((zero.length() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn custom_gesture_pattern() {
        let mut recognizer = GestureRecognizer::new();
        recognizer.move_threshold = 1.0;
        recognizer.swipe_threshold = 5.0;

        recognizer.add_custom_pattern(CustomGesturePattern {
            name: "Z-Shape".into(),
            directions: vec![
                SwipeDirection::Right,
                SwipeDirection::Down,
                SwipeDirection::Right,
            ],
            tolerance: 30.0,
        });

        // Simulate a Z-shaped gesture.
        recognizer.on_touch_start(0, 0.0, 0.0, 0.0);
        recognizer.on_touch_move(0, 100.0, 0.0);    // Right
        recognizer.on_touch_move(0, 50.0, 100.0);   // Down
        recognizer.on_touch_move(0, 150.0, 100.0);  // Right
        recognizer.on_touch_end(0, 0.3);

        // The gesture should match the Z-Shape pattern.
        let mut found_custom = false;
        while let Some(gesture) = recognizer.poll_gesture() {
            if matches!(gesture, GestureType::Custom { .. }) {
                found_custom = true;
            }
        }
        assert!(found_custom);
    }

    #[test]
    fn path_to_directions_basic() {
        let path = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(100.0, 0.0),   // Right
            Vec2::new(100.0, 100.0), // Down
            Vec2::new(0.0, 100.0),   // Left
        ];
        let dirs = path_to_directions(&path);
        assert_eq!(dirs, vec![SwipeDirection::Right, SwipeDirection::Down, SwipeDirection::Left]);
    }

    #[test]
    fn action_state_lifecycle() {
        assert_eq!(ActionState::default(), ActionState::Idle);

        let mut map = InputActionMap::new("Test");
        map.define_action(
            InputActionDef::new("Act").with_binding(InputBinding::key(KeyCode::X)),
        );

        // Idle initially.
        assert_eq!(map.action_state("Act"), ActionState::Idle);

        // Started on press.
        map.update(&make_input_with_key(KeyCode::X));
        assert_eq!(map.action_state("Act"), ActionState::Started);

        // Ongoing while held.
        map.update(&make_input_with_key(KeyCode::X));
        assert_eq!(map.action_state("Act"), ActionState::Ongoing);

        // Completed on release.
        map.update(&InputSnapshot::default());
        assert_eq!(map.action_state("Act"), ActionState::Completed);

        // Back to idle.
        map.update(&InputSnapshot::default());
        assert_eq!(map.action_state("Act"), ActionState::Idle);
    }
}
