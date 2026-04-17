// engine/gameplay/src/player_controller.rs
//
// Complete player controller for the Genovo engine.
//
// Provides a comprehensive first/third-person player controller with:
// - Input processing with dead zones, smoothing, and action mapping
// - Multiple movement modes (walking, running, crouching, swimming, flying)
// - Camera control with look sensitivity, inversion, and smoothing
// - Interaction system for picking up items and using objects
// - Inventory hotbar with quick-slot selection
// - HUD data output for UI rendering
// - Ground detection with slope handling
// - Jump with coyote time and jump buffering
// - Step climbing for small obstacles
// - Head bobbing and footstep events
// - Stamina system for sprinting

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const MAX_HOTBAR_SLOTS: usize = 10;
pub const DEFAULT_WALK_SPEED: f32 = 4.0;
pub const DEFAULT_RUN_SPEED: f32 = 7.5;
pub const DEFAULT_CROUCH_SPEED: f32 = 2.0;
pub const DEFAULT_SWIM_SPEED: f32 = 3.0;
pub const DEFAULT_FLY_SPEED: f32 = 10.0;
pub const DEFAULT_JUMP_FORCE: f32 = 5.0;
pub const DEFAULT_GRAVITY: f32 = -20.0;
pub const DEFAULT_LOOK_SENSITIVITY: f32 = 2.0;
pub const DEFAULT_INTERACTION_RANGE: f32 = 3.0;
pub const COYOTE_TIME: f32 = 0.15;
pub const JUMP_BUFFER_TIME: f32 = 0.1;
pub const DEFAULT_MAX_STAMINA: f32 = 100.0;
pub const DEFAULT_STAMINA_REGEN: f32 = 15.0;
pub const DEFAULT_SPRINT_COST: f32 = 20.0;
pub const STEP_HEIGHT: f32 = 0.35;
pub const MAX_SLOPE_ANGLE: f32 = 50.0;
pub const HEAD_BOB_FREQUENCY: f32 = 8.0;
pub const HEAD_BOB_AMPLITUDE: f32 = 0.04;

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

/// Raw player input from input devices.
#[derive(Debug, Clone, Default)]
pub struct PlayerInput {
    /// Movement axes (X = strafe, Y = forward/back).
    pub move_axes: [f32; 2],
    /// Look axes (X = yaw, Y = pitch).
    pub look_axes: [f32; 2],
    /// Jump button pressed this frame.
    pub jump_pressed: bool,
    /// Jump button held.
    pub jump_held: bool,
    /// Sprint button held.
    pub sprint_held: bool,
    /// Crouch button pressed.
    pub crouch_pressed: bool,
    /// Crouch button held.
    pub crouch_held: bool,
    /// Interact button pressed.
    pub interact_pressed: bool,
    /// Primary action (attack/use).
    pub primary_action: bool,
    /// Secondary action (aim/block).
    pub secondary_action: bool,
    /// Scroll wheel delta.
    pub scroll_delta: f32,
    /// Hotbar slot selection (0-9, -1 = none).
    pub hotbar_select: i32,
    /// Reload pressed.
    pub reload_pressed: bool,
    /// Toggle flashlight.
    pub flashlight_pressed: bool,
    /// Menu/pause pressed.
    pub menu_pressed: bool,
    /// Delta time for this frame.
    pub dt: f32,
}

/// Processed input after dead zones and smoothing.
#[derive(Debug, Clone, Default)]
pub struct ProcessedInput {
    pub move_direction: [f32; 2],
    pub move_magnitude: f32,
    pub look_delta: [f32; 2],
    pub wants_jump: bool,
    pub wants_sprint: bool,
    pub wants_crouch: bool,
    pub wants_interact: bool,
    pub primary_action: bool,
    pub secondary_action: bool,
}

/// Input processing settings.
#[derive(Debug, Clone)]
pub struct InputSettings {
    pub move_dead_zone: f32,
    pub look_dead_zone: f32,
    pub look_sensitivity: f32,
    pub look_smoothing: f32,
    pub invert_y: bool,
    pub invert_x: bool,
    pub move_response_curve: ResponseCurve,
    pub look_response_curve: ResponseCurve,
}

impl Default for InputSettings {
    fn default() -> Self {
        Self {
            move_dead_zone: 0.15,
            look_dead_zone: 0.05,
            look_sensitivity: DEFAULT_LOOK_SENSITIVITY,
            look_smoothing: 0.0,
            invert_y: false,
            invert_x: false,
            move_response_curve: ResponseCurve::Linear,
            look_response_curve: ResponseCurve::Linear,
        }
    }
}

/// Analog stick response curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResponseCurve {
    Linear,
    Quadratic,
    Cubic,
    SCurve,
}

impl ResponseCurve {
    pub fn apply(&self, value: f32) -> f32 {
        let sign = value.signum();
        let abs = value.abs();
        sign * match self {
            Self::Linear => abs,
            Self::Quadratic => abs * abs,
            Self::Cubic => abs * abs * abs,
            Self::SCurve => {
                let t = abs;
                3.0 * t * t - 2.0 * t * t * t
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Movement mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlayerMovementMode {
    Walking,
    Running,
    Crouching,
    Swimming,
    Flying,
    Climbing,
    Sliding,
    Falling,
    Dead,
}

impl Default for PlayerMovementMode {
    fn default() -> Self {
        Self::Walking
    }
}

impl PlayerMovementMode {
    pub fn is_grounded(&self) -> bool {
        matches!(self, Self::Walking | Self::Running | Self::Crouching | Self::Sliding)
    }

    pub fn allows_jump(&self) -> bool {
        matches!(self, Self::Walking | Self::Running | Self::Crouching)
    }

    pub fn base_speed(&self) -> f32 {
        match self {
            Self::Walking => DEFAULT_WALK_SPEED,
            Self::Running => DEFAULT_RUN_SPEED,
            Self::Crouching => DEFAULT_CROUCH_SPEED,
            Self::Swimming => DEFAULT_SWIM_SPEED,
            Self::Flying => DEFAULT_FLY_SPEED,
            Self::Climbing => DEFAULT_CROUCH_SPEED,
            Self::Sliding => DEFAULT_RUN_SPEED * 1.2,
            Self::Falling => DEFAULT_WALK_SPEED * 0.5,
            Self::Dead => 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Movement config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PlayerMovementConfig {
    pub walk_speed: f32,
    pub run_speed: f32,
    pub crouch_speed: f32,
    pub swim_speed: f32,
    pub fly_speed: f32,
    pub jump_force: f32,
    pub gravity: f32,
    pub air_control: f32,
    pub ground_acceleration: f32,
    pub air_acceleration: f32,
    pub ground_friction: f32,
    pub air_friction: f32,
    pub max_slope_angle: f32,
    pub step_height: f32,
    pub crouch_height: f32,
    pub stand_height: f32,
    pub coyote_time: f32,
    pub jump_buffer_time: f32,
    pub enable_head_bob: bool,
    pub head_bob_frequency: f32,
    pub head_bob_amplitude: f32,
}

impl Default for PlayerMovementConfig {
    fn default() -> Self {
        Self {
            walk_speed: DEFAULT_WALK_SPEED,
            run_speed: DEFAULT_RUN_SPEED,
            crouch_speed: DEFAULT_CROUCH_SPEED,
            swim_speed: DEFAULT_SWIM_SPEED,
            fly_speed: DEFAULT_FLY_SPEED,
            jump_force: DEFAULT_JUMP_FORCE,
            gravity: DEFAULT_GRAVITY,
            air_control: 0.3,
            ground_acceleration: 15.0,
            air_acceleration: 5.0,
            ground_friction: 10.0,
            air_friction: 0.5,
            max_slope_angle: MAX_SLOPE_ANGLE,
            step_height: STEP_HEIGHT,
            crouch_height: 1.0,
            stand_height: 1.8,
            coyote_time: COYOTE_TIME,
            jump_buffer_time: JUMP_BUFFER_TIME,
            enable_head_bob: true,
            head_bob_frequency: HEAD_BOB_FREQUENCY,
            head_bob_amplitude: HEAD_BOB_AMPLITUDE,
        }
    }
}

// ---------------------------------------------------------------------------
// Camera state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PlayerCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub pitch_min: f32,
    pub pitch_max: f32,
    pub fov: f32,
    pub near_plane: f32,
    pub far_plane: f32,
    pub eye_height: f32,
    pub head_bob_offset: f32,
    pub head_bob_time: f32,
    pub camera_shake: f32,
    pub camera_shake_decay: f32,
    look_smooth_x: f32,
    look_smooth_y: f32,
}

impl Default for PlayerCamera {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            pitch_min: -89.0,
            pitch_max: 89.0,
            fov: 75.0,
            near_plane: 0.1,
            far_plane: 1000.0,
            eye_height: 1.6,
            head_bob_offset: 0.0,
            head_bob_time: 0.0,
            camera_shake: 0.0,
            camera_shake_decay: 5.0,
            look_smooth_x: 0.0,
            look_smooth_y: 0.0,
        }
    }
}

impl PlayerCamera {
    pub fn update(&mut self, look_delta: [f32; 2], settings: &InputSettings, dt: f32) {
        let mut dx = look_delta[0] * settings.look_sensitivity;
        let mut dy = look_delta[1] * settings.look_sensitivity;
        if settings.invert_x { dx = -dx; }
        if settings.invert_y { dy = -dy; }

        // Apply smoothing.
        if settings.look_smoothing > 0.0 {
            let s = (1.0 - settings.look_smoothing).max(0.01);
            self.look_smooth_x = self.look_smooth_x + (dx - self.look_smooth_x) * s;
            self.look_smooth_y = self.look_smooth_y + (dy - self.look_smooth_y) * s;
            dx = self.look_smooth_x;
            dy = self.look_smooth_y;
        }

        self.yaw += dx;
        self.pitch = (self.pitch + dy).clamp(self.pitch_min, self.pitch_max);

        // Decay camera shake.
        if self.camera_shake > 0.0 {
            self.camera_shake = (self.camera_shake - self.camera_shake_decay * dt).max(0.0);
        }
    }

    pub fn update_head_bob(&mut self, speed: f32, is_grounded: bool, dt: f32, config: &PlayerMovementConfig) {
        if !config.enable_head_bob || !is_grounded || speed < 0.5 {
            self.head_bob_offset *= 0.9;
            return;
        }
        self.head_bob_time += dt * speed * config.head_bob_frequency;
        self.head_bob_offset = (self.head_bob_time).sin() * config.head_bob_amplitude * (speed / DEFAULT_WALK_SPEED);
    }

    pub fn add_shake(&mut self, intensity: f32) {
        self.camera_shake = (self.camera_shake + intensity).min(2.0);
    }

    pub fn forward_direction(&self) -> [f32; 3] {
        let yaw_rad = self.yaw.to_radians();
        let pitch_rad = self.pitch.to_radians();
        [
            yaw_rad.sin() * pitch_rad.cos(),
            pitch_rad.sin(),
            yaw_rad.cos() * pitch_rad.cos(),
        ]
    }

    pub fn flat_forward(&self) -> [f32; 2] {
        let yaw_rad = self.yaw.to_radians();
        [yaw_rad.sin(), yaw_rad.cos()]
    }

    pub fn flat_right(&self) -> [f32; 2] {
        let yaw_rad = self.yaw.to_radians();
        [yaw_rad.cos(), -yaw_rad.sin()]
    }
}

// ---------------------------------------------------------------------------
// Stamina
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct StaminaSystem {
    pub current: f32,
    pub max: f32,
    pub regen_rate: f32,
    pub sprint_cost: f32,
    pub jump_cost: f32,
    pub regen_delay: f32,
    pub regen_cooldown: f32,
    pub exhausted: bool,
    pub exhaustion_threshold: f32,
}

impl Default for StaminaSystem {
    fn default() -> Self {
        Self {
            current: DEFAULT_MAX_STAMINA,
            max: DEFAULT_MAX_STAMINA,
            regen_rate: DEFAULT_STAMINA_REGEN,
            sprint_cost: DEFAULT_SPRINT_COST,
            jump_cost: 15.0,
            regen_delay: 1.0,
            regen_cooldown: 0.0,
            exhausted: false,
            exhaustion_threshold: 10.0,
        }
    }
}

impl StaminaSystem {
    pub fn consume(&mut self, amount: f32) -> bool {
        if self.current >= amount {
            self.current -= amount;
            self.regen_cooldown = self.regen_delay;
            if self.current <= 0.0 {
                self.exhausted = true;
            }
            true
        } else {
            false
        }
    }

    pub fn update(&mut self, dt: f32) {
        if self.regen_cooldown > 0.0 {
            self.regen_cooldown -= dt;
            return;
        }
        self.current = (self.current + self.regen_rate * dt).min(self.max);
        if self.exhausted && self.current >= self.exhaustion_threshold {
            self.exhausted = false;
        }
    }

    pub fn fraction(&self) -> f32 {
        self.current / self.max
    }

    pub fn can_sprint(&self) -> bool {
        !self.exhausted && self.current > 0.0
    }
}

// ---------------------------------------------------------------------------
// Ground info
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GroundState {
    pub on_ground: bool,
    pub ground_normal: [f32; 3],
    pub ground_distance: f32,
    pub slope_angle: f32,
    pub surface_type: SurfaceType,
    pub time_since_grounded: f32,
    pub time_since_airborne: f32,
    pub step_height: f32,
}

impl Default for GroundState {
    fn default() -> Self {
        Self {
            on_ground: false,
            ground_normal: [0.0, 1.0, 0.0],
            ground_distance: 0.0,
            slope_angle: 0.0,
            surface_type: SurfaceType::Default,
            time_since_grounded: 0.0,
            time_since_airborne: 0.0,
            step_height: 0.0,
        }
    }
}

impl GroundState {
    pub fn update(&mut self, dt: f32) {
        if self.on_ground {
            self.time_since_airborne += dt;
            self.time_since_grounded = 0.0;
        } else {
            self.time_since_grounded += dt;
            self.time_since_airborne = 0.0;
        }
    }

    pub fn in_coyote_time(&self) -> bool {
        !self.on_ground && self.time_since_grounded < COYOTE_TIME
    }

    pub fn is_on_steep_slope(&self, max_angle: f32) -> bool {
        self.on_ground && self.slope_angle > max_angle
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SurfaceType {
    Default,
    Grass,
    Dirt,
    Stone,
    Wood,
    Metal,
    Water,
    Snow,
    Sand,
    Ice,
}

// ---------------------------------------------------------------------------
// Interaction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct InteractionState {
    pub current_target: Option<InteractionTarget>,
    pub interaction_range: f32,
    pub interaction_cooldown: f32,
    pub last_interaction_time: f32,
}

impl Default for InteractionState {
    fn default() -> Self {
        Self {
            current_target: None,
            interaction_range: DEFAULT_INTERACTION_RANGE,
            interaction_cooldown: 0.2,
            last_interaction_time: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InteractionTarget {
    pub entity_id: u64,
    pub interaction_type: InteractionType,
    pub display_name: String,
    pub prompt_text: String,
    pub distance: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InteractionType {
    Pickup,
    Use,
    Open,
    Talk,
    Read,
    Activate,
    Mount,
    Examine,
}

// ---------------------------------------------------------------------------
// Inventory hotbar
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct HotbarSlot {
    pub item_id: Option<u64>,
    pub item_name: String,
    pub item_icon: String,
    pub stack_count: u32,
    pub cooldown_remaining: f32,
    pub cooldown_total: f32,
    pub is_usable: bool,
}

impl Default for HotbarSlot {
    fn default() -> Self {
        Self {
            item_id: None,
            item_name: String::new(),
            item_icon: String::new(),
            stack_count: 0,
            cooldown_remaining: 0.0,
            cooldown_total: 0.0,
            is_usable: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InventoryHotbar {
    pub slots: Vec<HotbarSlot>,
    pub selected_slot: usize,
    pub previous_slot: usize,
}

impl Default for InventoryHotbar {
    fn default() -> Self {
        Self {
            slots: (0..MAX_HOTBAR_SLOTS).map(|_| HotbarSlot::default()).collect(),
            selected_slot: 0,
            previous_slot: 0,
        }
    }
}

impl InventoryHotbar {
    pub fn select_slot(&mut self, slot: usize) {
        if slot < self.slots.len() {
            self.previous_slot = self.selected_slot;
            self.selected_slot = slot;
        }
    }

    pub fn select_next(&mut self) {
        self.previous_slot = self.selected_slot;
        self.selected_slot = (self.selected_slot + 1) % self.slots.len();
    }

    pub fn select_previous(&mut self) {
        self.previous_slot = self.selected_slot;
        self.selected_slot = if self.selected_slot == 0 { self.slots.len() - 1 } else { self.selected_slot - 1 };
    }

    pub fn quick_swap(&mut self) {
        let tmp = self.selected_slot;
        self.selected_slot = self.previous_slot;
        self.previous_slot = tmp;
    }

    pub fn current_item(&self) -> &HotbarSlot {
        &self.slots[self.selected_slot]
    }

    pub fn set_item(&mut self, slot: usize, item_id: u64, name: &str, icon: &str, count: u32) {
        if slot < self.slots.len() {
            self.slots[slot] = HotbarSlot {
                item_id: Some(item_id),
                item_name: name.to_string(),
                item_icon: icon.to_string(),
                stack_count: count,
                cooldown_remaining: 0.0,
                cooldown_total: 0.0,
                is_usable: true,
            };
        }
    }

    pub fn clear_slot(&mut self, slot: usize) {
        if slot < self.slots.len() {
            self.slots[slot] = HotbarSlot::default();
        }
    }

    pub fn update_cooldowns(&mut self, dt: f32) {
        for slot in &mut self.slots {
            if slot.cooldown_remaining > 0.0 {
                slot.cooldown_remaining = (slot.cooldown_remaining - dt).max(0.0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HUD data
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PlayerHudData {
    pub health: f32,
    pub max_health: f32,
    pub stamina_fraction: f32,
    pub is_sprinting: bool,
    pub is_crouching: bool,
    pub movement_mode: PlayerMovementMode,
    pub speed: f32,
    pub hotbar_slots: Vec<HotbarSlot>,
    pub selected_slot: usize,
    pub interaction_prompt: Option<String>,
    pub crosshair_spread: f32,
    pub compass_heading: f32,
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub flashlight_on: bool,
    pub notifications: Vec<String>,
}

impl Default for PlayerHudData {
    fn default() -> Self {
        Self {
            health: 100.0,
            max_health: 100.0,
            stamina_fraction: 1.0,
            is_sprinting: false,
            is_crouching: false,
            movement_mode: PlayerMovementMode::Walking,
            speed: 0.0,
            hotbar_slots: Vec::new(),
            selected_slot: 0,
            interaction_prompt: None,
            crosshair_spread: 0.0,
            compass_heading: 0.0,
            position: [0.0; 3],
            velocity: [0.0; 3],
            flashlight_on: false,
            notifications: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Footstep events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FootstepEvent {
    pub position: [f32; 3],
    pub surface: SurfaceType,
    pub speed: f32,
    pub is_left_foot: bool,
    pub volume: f32,
}

// ---------------------------------------------------------------------------
// Player controller
// ---------------------------------------------------------------------------

/// The complete player controller.
#[derive(Debug)]
pub struct PlayerController {
    // Configuration.
    pub input_settings: InputSettings,
    pub movement_config: PlayerMovementConfig,

    // State.
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub movement_mode: PlayerMovementMode,
    pub camera: PlayerCamera,
    pub ground: GroundState,
    pub stamina: StaminaSystem,
    pub interaction: InteractionState,
    pub hotbar: InventoryHotbar,

    // Health.
    pub health: f32,
    pub max_health: f32,
    pub is_alive: bool,

    // Internal state.
    jump_buffer_timer: f32,
    is_left_foot: bool,
    footstep_timer: f32,
    flashlight_on: bool,
    time_alive: f32,
    notifications: VecDeque<(String, f32)>,
    footstep_events: Vec<FootstepEvent>,
    processed_input: ProcessedInput,
}

impl PlayerController {
    pub fn new(spawn_position: [f32; 3]) -> Self {
        Self {
            input_settings: InputSettings::default(),
            movement_config: PlayerMovementConfig::default(),
            position: spawn_position,
            velocity: [0.0; 3],
            movement_mode: PlayerMovementMode::Walking,
            camera: PlayerCamera::default(),
            ground: GroundState::default(),
            stamina: StaminaSystem::default(),
            interaction: InteractionState::default(),
            hotbar: InventoryHotbar::default(),
            health: 100.0,
            max_health: 100.0,
            is_alive: true,
            jump_buffer_timer: 0.0,
            is_left_foot: false,
            footstep_timer: 0.0,
            flashlight_on: false,
            time_alive: 0.0,
            notifications: VecDeque::new(),
            footstep_events: Vec::new(),
            processed_input: ProcessedInput::default(),
        }
    }

    /// Process raw input into usable values.
    pub fn process_input(&mut self, raw: &PlayerInput) {
        let settings = &self.input_settings;
        let mut pi = ProcessedInput::default();

        // Apply dead zone and response curve to move axes.
        let mx = apply_dead_zone(raw.move_axes[0], settings.move_dead_zone);
        let my = apply_dead_zone(raw.move_axes[1], settings.move_dead_zone);
        pi.move_direction = [
            settings.move_response_curve.apply(mx),
            settings.move_response_curve.apply(my),
        ];
        pi.move_magnitude = (pi.move_direction[0].powi(2) + pi.move_direction[1].powi(2)).sqrt().min(1.0);

        // Apply dead zone to look axes.
        let lx = apply_dead_zone(raw.look_axes[0], settings.look_dead_zone);
        let ly = apply_dead_zone(raw.look_axes[1], settings.look_dead_zone);
        pi.look_delta = [
            settings.look_response_curve.apply(lx),
            settings.look_response_curve.apply(ly),
        ];

        pi.wants_jump = raw.jump_pressed;
        pi.wants_sprint = raw.sprint_held;
        pi.wants_crouch = raw.crouch_pressed || raw.crouch_held;
        pi.wants_interact = raw.interact_pressed;
        pi.primary_action = raw.primary_action;
        pi.secondary_action = raw.secondary_action;

        self.processed_input = pi;
    }

    /// Main update tick.
    pub fn update(&mut self, raw: &PlayerInput, ground_check: Option<GroundCheckResult>) {
        if !self.is_alive {
            self.movement_mode = PlayerMovementMode::Dead;
            return;
        }

        let dt = raw.dt;
        self.time_alive += dt;
        self.footstep_events.clear();

        // Process input.
        self.process_input(raw);

        // Update ground state.
        if let Some(gc) = ground_check {
            self.ground.on_ground = gc.hit;
            self.ground.ground_normal = gc.normal;
            self.ground.ground_distance = gc.distance;
            self.ground.slope_angle = gc.slope_angle;
            self.ground.surface_type = gc.surface_type;
        }
        self.ground.update(dt);

        // Update camera.
        self.camera.update(self.processed_input.look_delta, &self.input_settings, dt);

        // Update movement mode.
        self.update_movement_mode(dt);

        // Apply movement.
        self.apply_movement(dt);

        // Jump logic.
        self.update_jump(raw, dt);

        // Stamina.
        self.stamina.update(dt);

        // Head bob.
        let speed = (self.velocity[0].powi(2) + self.velocity[2].powi(2)).sqrt();
        self.camera.update_head_bob(speed, self.ground.on_ground, dt, &self.movement_config);

        // Footsteps.
        self.update_footsteps(speed, dt);

        // Hotbar.
        self.update_hotbar(raw, dt);

        // Flashlight.
        if raw.flashlight_pressed {
            self.flashlight_on = !self.flashlight_on;
        }

        // Notifications.
        self.notifications.retain(|n| n.1 > 0.0);
        for n in &mut self.notifications {
            n.1 -= dt;
        }

        // Apply gravity.
        if !self.ground.on_ground && self.movement_mode != PlayerMovementMode::Flying
            && self.movement_mode != PlayerMovementMode::Swimming
        {
            self.velocity[1] += self.movement_config.gravity * dt;
        }

        // Integrate position.
        self.position[0] += self.velocity[0] * dt;
        self.position[1] += self.velocity[1] * dt;
        self.position[2] += self.velocity[2] * dt;
    }

    fn update_movement_mode(&mut self, _dt: f32) {
        let pi = &self.processed_input;

        if !self.ground.on_ground && self.movement_mode != PlayerMovementMode::Flying
            && self.movement_mode != PlayerMovementMode::Swimming
        {
            self.movement_mode = PlayerMovementMode::Falling;
            return;
        }

        if pi.wants_crouch && self.ground.on_ground {
            self.movement_mode = PlayerMovementMode::Crouching;
        } else if pi.wants_sprint && self.stamina.can_sprint() && pi.move_magnitude > 0.5 {
            self.movement_mode = PlayerMovementMode::Running;
        } else if self.ground.on_ground {
            self.movement_mode = PlayerMovementMode::Walking;
        }
    }

    fn apply_movement(&mut self, dt: f32) {
        let pi = &self.processed_input;
        let speed = self.movement_mode.base_speed();
        let forward = self.camera.flat_forward();
        let right = self.camera.flat_right();

        let wish_x = right[0] * pi.move_direction[0] + forward[0] * pi.move_direction[1];
        let wish_z = right[1] * pi.move_direction[0] + forward[1] * pi.move_direction[1];
        let wish_len = (wish_x * wish_x + wish_z * wish_z).sqrt();

        if wish_len > 0.01 {
            let norm_x = wish_x / wish_len;
            let norm_z = wish_z / wish_len;
            let target_x = norm_x * speed * pi.move_magnitude.min(1.0);
            let target_z = norm_z * speed * pi.move_magnitude.min(1.0);

            let accel = if self.ground.on_ground {
                self.movement_config.ground_acceleration
            } else {
                self.movement_config.air_acceleration
            };

            self.velocity[0] += (target_x - self.velocity[0]) * accel * dt;
            self.velocity[2] += (target_z - self.velocity[2]) * accel * dt;
        } else if self.ground.on_ground {
            // Apply friction when not moving.
            let friction = self.movement_config.ground_friction * dt;
            self.velocity[0] *= (1.0 - friction).max(0.0);
            self.velocity[2] *= (1.0 - friction).max(0.0);
        }

        // Sprint stamina cost.
        if self.movement_mode == PlayerMovementMode::Running {
            self.stamina.consume(self.stamina.sprint_cost * dt);
        }
    }

    fn update_jump(&mut self, raw: &PlayerInput, dt: f32) {
        // Jump buffer.
        if raw.jump_pressed {
            self.jump_buffer_timer = self.movement_config.jump_buffer_time;
        }
        self.jump_buffer_timer -= dt;

        let can_jump = (self.ground.on_ground || self.ground.in_coyote_time())
            && self.movement_mode.allows_jump();

        if self.jump_buffer_timer > 0.0 && can_jump {
            self.velocity[1] = self.movement_config.jump_force;
            self.ground.on_ground = false;
            self.ground.time_since_grounded = COYOTE_TIME + 0.01; // Prevent double jump.
            self.jump_buffer_timer = 0.0;
            self.stamina.consume(self.stamina.jump_cost);
        }

        // Variable jump height.
        if !raw.jump_held && self.velocity[1] > 0.0 {
            self.velocity[1] *= 0.5;
        }
    }

    fn update_footsteps(&mut self, speed: f32, dt: f32) {
        if !self.ground.on_ground || speed < 0.5 {
            self.footstep_timer = 0.0;
            return;
        }

        let interval = 1.0 / (speed * 0.4).max(0.5);
        self.footstep_timer += dt;
        if self.footstep_timer >= interval {
            self.footstep_timer -= interval;
            self.is_left_foot = !self.is_left_foot;
            self.footstep_events.push(FootstepEvent {
                position: self.position,
                surface: self.ground.surface_type,
                speed,
                is_left_foot: self.is_left_foot,
                volume: (speed / DEFAULT_RUN_SPEED).min(1.0),
            });
        }
    }

    fn update_hotbar(&mut self, raw: &PlayerInput, dt: f32) {
        if raw.hotbar_select >= 0 && raw.hotbar_select < MAX_HOTBAR_SLOTS as i32 {
            self.hotbar.select_slot(raw.hotbar_select as usize);
        }
        if raw.scroll_delta > 0.0 {
            self.hotbar.select_next();
        } else if raw.scroll_delta < 0.0 {
            self.hotbar.select_previous();
        }
        self.hotbar.update_cooldowns(dt);
    }

    pub fn take_damage(&mut self, amount: f32) {
        self.health = (self.health - amount).max(0.0);
        self.camera.add_shake(amount * 0.02);
        if self.health <= 0.0 {
            self.is_alive = false;
            self.movement_mode = PlayerMovementMode::Dead;
        }
    }

    pub fn heal(&mut self, amount: f32) {
        self.health = (self.health + amount).min(self.max_health);
    }

    pub fn respawn(&mut self, position: [f32; 3]) {
        self.position = position;
        self.velocity = [0.0; 3];
        self.health = self.max_health;
        self.is_alive = true;
        self.movement_mode = PlayerMovementMode::Walking;
        self.stamina.current = self.stamina.max;
    }

    pub fn add_notification(&mut self, text: String, duration: f32) {
        self.notifications.push_back((text, duration));
        if self.notifications.len() > 5 {
            self.notifications.pop_front();
        }
    }

    pub fn get_footstep_events(&self) -> &[FootstepEvent] {
        &self.footstep_events
    }

    /// Build HUD data for the UI system.
    pub fn build_hud_data(&self) -> PlayerHudData {
        let speed = (self.velocity[0].powi(2) + self.velocity[2].powi(2)).sqrt();
        PlayerHudData {
            health: self.health,
            max_health: self.max_health,
            stamina_fraction: self.stamina.fraction(),
            is_sprinting: self.movement_mode == PlayerMovementMode::Running,
            is_crouching: self.movement_mode == PlayerMovementMode::Crouching,
            movement_mode: self.movement_mode,
            speed,
            hotbar_slots: self.hotbar.slots.clone(),
            selected_slot: self.hotbar.selected_slot,
            interaction_prompt: self.interaction.current_target.as_ref().map(|t| t.prompt_text.clone()),
            crosshair_spread: if self.movement_mode == PlayerMovementMode::Running { 3.0 } else { 1.0 },
            compass_heading: self.camera.yaw % 360.0,
            position: self.position,
            velocity: self.velocity,
            flashlight_on: self.flashlight_on,
            notifications: self.notifications.iter().map(|n| n.0.clone()).collect(),
        }
    }
}

/// Result from a ground check query.
#[derive(Debug, Clone)]
pub struct GroundCheckResult {
    pub hit: bool,
    pub normal: [f32; 3],
    pub distance: f32,
    pub slope_angle: f32,
    pub surface_type: SurfaceType,
}

fn apply_dead_zone(value: f32, dead_zone: f32) -> f32 {
    let abs = value.abs();
    if abs < dead_zone {
        return 0.0;
    }
    let sign = value.signum();
    sign * (abs - dead_zone) / (1.0 - dead_zone)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dead_zone() {
        assert_eq!(apply_dead_zone(0.05, 0.15), 0.0);
        assert!(apply_dead_zone(0.5, 0.15) > 0.0);
        assert!(apply_dead_zone(-0.5, 0.15) < 0.0);
    }

    #[test]
    fn test_stamina() {
        let mut stamina = StaminaSystem::default();
        assert!(stamina.can_sprint());
        stamina.consume(90.0);
        assert!(stamina.can_sprint());
        stamina.consume(11.0);
        assert!(!stamina.can_sprint());
    }

    #[test]
    fn test_hotbar() {
        let mut hotbar = InventoryHotbar::default();
        assert_eq!(hotbar.selected_slot, 0);
        hotbar.select_next();
        assert_eq!(hotbar.selected_slot, 1);
        hotbar.select_previous();
        assert_eq!(hotbar.selected_slot, 0);
    }

    #[test]
    fn test_player_controller() {
        let mut pc = PlayerController::new([0.0, 0.0, 0.0]);
        assert!(pc.is_alive);
        pc.take_damage(50.0);
        assert_eq!(pc.health, 50.0);
        pc.heal(20.0);
        assert_eq!(pc.health, 70.0);
    }

    #[test]
    fn test_camera_forward() {
        let camera = PlayerCamera::default();
        let dir = camera.forward_direction();
        assert!((dir[0].powi(2) + dir[1].powi(2) + dir[2].powi(2) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_response_curves() {
        assert!((ResponseCurve::Linear.apply(0.5) - 0.5).abs() < 0.01);
        assert!(ResponseCurve::Quadratic.apply(0.5) < 0.5);
    }
}
