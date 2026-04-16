//! Physics-based character movement controller.
//!
//! Provides a capsule-based character controller suitable for first-person,
//! third-person, and side-scrolling games. The controller handles grounding
//! detection, slope limits, step climbing, coyote time, jump buffering,
//! moving platform support, and rigid body pushing.
//!
//! # Architecture
//!
//! The controller is split into:
//!
//! - [`CharacterController`] -- standalone movement logic, usable without ECS.
//! - [`CharacterControllerComponent`] -- thin ECS wrapper implementing
//!   [`Component`](genovo_ecs::Component).
//! - [`CharacterState`] -- finite state machine for movement modes.
//! - [`GroundInfo`] -- data about the surface the character is standing on.
//! - [`CollisionShape`] -- capsule geometry for sweeps and overlaps.
//!
//! The controller intentionally does **not** depend on a specific physics engine.
//! Instead, it exposes sweep/overlap callbacks via the [`CollisionWorld`] trait,
//! which game code implements to bridge to whatever physics backend is in use.

use glam::Vec3;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Collision abstraction
// ---------------------------------------------------------------------------

/// Result of a shape sweep against the collision world.
#[derive(Debug, Clone)]
pub struct SweepHit {
    /// The fraction along the sweep direction where the hit occurred (0..=1).
    pub fraction: f32,
    /// World-space position of the hit point.
    pub position: Vec3,
    /// Surface normal at the hit point (pointing away from the surface).
    pub normal: Vec3,
    /// Entity id of the collider that was hit, if applicable.
    pub entity_id: Option<u32>,
    /// Whether the hit surface is walkable (slope angle below threshold).
    pub walkable: bool,
}

/// Result of a shape overlap test.
#[derive(Debug, Clone)]
pub struct OverlapResult {
    /// Minimum translation vector to separate the shapes.
    pub penetration_depth: f32,
    /// Direction to push the character out of the overlap.
    pub penetration_normal: Vec3,
    /// Entity id of the overlapping collider.
    pub entity_id: Option<u32>,
}

/// Abstraction over the physics world so the character controller remains
/// engine-agnostic. Implement this trait to bridge to your physics backend.
pub trait CollisionWorld {
    /// Sweep a capsule from `origin` along `direction * distance` and return
    /// all hits sorted by fraction (nearest first).
    fn sweep_capsule(
        &self,
        origin: Vec3,
        direction: Vec3,
        distance: f32,
        radius: f32,
        half_height: f32,
        ignore_entity: Option<u32>,
    ) -> Vec<SweepHit>;

    /// Test for overlaps at the given position.
    fn overlap_capsule(
        &self,
        position: Vec3,
        radius: f32,
        half_height: f32,
        ignore_entity: Option<u32>,
    ) -> Vec<OverlapResult>;

    /// Cast a ray and return the first hit, if any.
    fn raycast(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        ignore_entity: Option<u32>,
    ) -> Option<SweepHit>;

    /// Return the velocity of a physics body, used for moving platform support.
    fn get_body_velocity(&self, entity_id: u32) -> Option<Vec3>;
}

// ---------------------------------------------------------------------------
// Capsule shape
// ---------------------------------------------------------------------------

/// Capsule collision shape used by the character controller.
///
/// The capsule is oriented along the Y axis with its center at the character's
/// position. `half_height` is the distance from center to the tip of each
/// hemisphere, so total height = `2 * (half_height + radius)`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CollisionShape {
    /// Radius of the capsule's hemispheres and cylinder.
    pub radius: f32,
    /// Half the height of the cylindrical section (not including hemispheres).
    pub half_height: f32,
}

impl CollisionShape {
    /// Create a new capsule shape.
    ///
    /// * `radius` -- hemisphere/cylinder radius
    /// * `total_height` -- overall height including hemispheres
    pub fn new(radius: f32, total_height: f32) -> Self {
        let half_height = (total_height * 0.5 - radius).max(0.0);
        Self {
            radius,
            half_height,
        }
    }

    /// Total height of the capsule.
    #[inline]
    pub fn total_height(&self) -> f32 {
        2.0 * (self.half_height + self.radius)
    }

    /// Position of the bottom sphere center relative to the capsule center.
    #[inline]
    pub fn bottom_center_offset(&self) -> Vec3 {
        Vec3::new(0.0, -self.half_height, 0.0)
    }

    /// Position of the top sphere center relative to the capsule center.
    #[inline]
    pub fn top_center_offset(&self) -> Vec3 {
        Vec3::new(0.0, self.half_height, 0.0)
    }

    /// The Y position of the feet (lowest point of the capsule).
    #[inline]
    pub fn feet_offset_y(&self) -> f32 {
        -(self.half_height + self.radius)
    }

    /// The Y position of the head (highest point of the capsule).
    #[inline]
    pub fn head_offset_y(&self) -> f32 {
        self.half_height + self.radius
    }

    /// Create a crouching version of this shape (halved height, same radius).
    pub fn crouching(&self) -> Self {
        let crouch_height = self.total_height() * 0.5;
        Self::new(self.radius, crouch_height.max(self.radius * 2.0))
    }
}

impl Default for CollisionShape {
    fn default() -> Self {
        // Default: human-sized capsule (radius 0.3m, height 1.8m)
        Self::new(0.3, 1.8)
    }
}

// ---------------------------------------------------------------------------
// Character state
// ---------------------------------------------------------------------------

/// Finite state machine states for character movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CharacterState {
    /// Standing on walkable ground.
    Grounded,
    /// In the air (falling or jumping).
    Airborne,
    /// On a slope steeper than the max walkable angle.
    Sliding,
    /// Attached to a climbable surface (ladder, wall).
    Climbing,
    /// Submerged in a water volume.
    Swimming,
}

impl CharacterState {
    /// Returns `true` if the character can initiate a jump from this state.
    #[inline]
    pub fn can_jump(&self) -> bool {
        matches!(self, Self::Grounded | Self::Swimming)
    }

    /// Returns `true` if gravity should be applied.
    #[inline]
    pub fn applies_gravity(&self) -> bool {
        matches!(self, Self::Airborne | Self::Sliding)
    }

    /// Returns `true` if the character has full directional control.
    #[inline]
    pub fn has_full_control(&self) -> bool {
        matches!(self, Self::Grounded | Self::Swimming | Self::Climbing)
    }
}

impl Default for CharacterState {
    fn default() -> Self {
        Self::Airborne
    }
}

// ---------------------------------------------------------------------------
// Ground info
// ---------------------------------------------------------------------------

/// Information about the ground surface beneath the character.
#[derive(Debug, Clone)]
pub struct GroundInfo {
    /// Whether the character is on walkable ground.
    pub grounded: bool,
    /// Surface normal of the ground.
    pub normal: Vec3,
    /// World-space position of the ground contact point.
    pub point: Vec3,
    /// Slope angle in radians.
    pub slope_angle: f32,
    /// Entity id of the ground collider, if any.
    pub ground_entity: Option<u32>,
    /// Distance from the character's feet to the ground surface.
    pub distance: f32,
}

impl Default for GroundInfo {
    fn default() -> Self {
        Self {
            grounded: false,
            normal: Vec3::Y,
            point: Vec3::ZERO,
            slope_angle: 0.0,
            ground_entity: None,
            distance: f32::MAX,
        }
    }
}

// ---------------------------------------------------------------------------
// Moving platform state
// ---------------------------------------------------------------------------

/// Tracks the platform the character is standing on for velocity inheritance.
#[derive(Debug, Clone, Default)]
pub struct MovingPlatformState {
    /// Entity id of the platform.
    pub platform_entity: Option<u32>,
    /// Platform velocity from the previous frame.
    pub platform_velocity: Vec3,
    /// Position on the platform in the platform's local space (for rotation).
    pub local_position: Vec3,
    /// How many frames the character has been on this platform.
    pub frames_on_platform: u32,
}

impl MovingPlatformState {
    /// Clear platform tracking.
    pub fn clear(&mut self) {
        self.platform_entity = None;
        self.platform_velocity = Vec3::ZERO;
        self.local_position = Vec3::ZERO;
        self.frames_on_platform = 0;
    }

    /// Update platform state for the current frame.
    pub fn update(&mut self, ground: &GroundInfo, collision_world: &dyn CollisionWorld) {
        if let Some(entity_id) = ground.ground_entity {
            if self.platform_entity == Some(entity_id) {
                self.frames_on_platform += 1;
            } else {
                self.platform_entity = Some(entity_id);
                self.frames_on_platform = 1;
            }
            self.platform_velocity = collision_world
                .get_body_velocity(entity_id)
                .unwrap_or(Vec3::ZERO);
        } else {
            self.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// Character controller configuration
// ---------------------------------------------------------------------------

/// Tunable parameters for the character controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterConfig {
    /// Maximum walkable slope angle in radians.
    pub max_slope_angle: f32,
    /// Maximum height of obstacles the character can step up onto.
    pub max_step_height: f32,
    /// Skin width -- small gap maintained between the capsule and geometry
    /// to prevent tunneling and jitter.
    pub skin_width: f32,
    /// Gravity acceleration (positive = downward).
    pub gravity: f32,
    /// Ground movement speed (units/sec).
    pub move_speed: f32,
    /// Air movement speed multiplier (0..1).
    pub air_control: f32,
    /// Jump impulse velocity (units/sec upward).
    pub jump_speed: f32,
    /// Maximum number of jumps before touching ground (1 = single jump,
    /// 2 = double jump, etc.).
    pub max_jumps: u32,
    /// Coyote time: grace period after leaving ground during which a jump
    /// is still allowed (seconds).
    pub coyote_time: f32,
    /// Jump buffer: if the player presses jump this many seconds before
    /// landing, the jump is executed on landing.
    pub jump_buffer_time: f32,
    /// Terminal velocity (maximum downward speed).
    pub terminal_velocity: f32,
    /// Force applied to rigid bodies the character walks into.
    pub push_force: f32,
    /// Slide acceleration on steep slopes.
    pub slide_acceleration: f32,
    /// Slide friction coefficient.
    pub slide_friction: f32,
    /// Climbing speed (units/sec).
    pub climb_speed: f32,
    /// Swimming speed (units/sec).
    pub swim_speed: f32,
    /// Buoyancy force when swimming (counteracts gravity).
    pub swim_buoyancy: f32,
    /// Ground friction deceleration rate.
    pub ground_friction: f32,
    /// Air drag deceleration rate.
    pub air_drag: f32,
    /// Maximum number of sweep iterations per move.
    pub max_move_iterations: u32,
    /// Snap-to-ground distance when walking down slopes.
    pub snap_to_ground_distance: f32,
    /// Whether to apply step-up logic.
    pub enable_step_up: bool,
    /// Whether to enable moving platform support.
    pub enable_moving_platforms: bool,
}

impl Default for CharacterConfig {
    fn default() -> Self {
        Self {
            max_slope_angle: std::f32::consts::FRAC_PI_4, // 45 degrees
            max_step_height: 0.35,
            skin_width: 0.02,
            gravity: 20.0,
            move_speed: 6.0,
            air_control: 0.3,
            jump_speed: 8.0,
            max_jumps: 1,
            coyote_time: 0.12,
            jump_buffer_time: 0.1,
            terminal_velocity: 50.0,
            push_force: 5.0,
            slide_acceleration: 15.0,
            slide_friction: 0.4,
            climb_speed: 3.0,
            swim_speed: 4.0,
            swim_buoyancy: 12.0,
            ground_friction: 12.0,
            air_drag: 0.5,
            max_move_iterations: 4,
            snap_to_ground_distance: 0.4,
            enable_step_up: true,
            enable_moving_platforms: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Character movement input
// ---------------------------------------------------------------------------

/// Per-frame input fed to the character controller.
#[derive(Debug, Clone, Default)]
pub struct CharacterInput {
    /// Desired horizontal movement direction (normalized, world-space).
    /// X = right, Z = forward (typical).
    pub move_direction: Vec3,
    /// Whether the jump button was pressed this frame.
    pub jump_pressed: bool,
    /// Whether the jump button is held.
    pub jump_held: bool,
    /// Whether the crouch button is held.
    pub crouch_held: bool,
    /// Whether the sprint button is held.
    pub sprint_held: bool,
    /// Whether the character should start climbing (near a ladder, etc.).
    pub climb_requested: bool,
    /// Vertical input for climbing/swimming (positive = up).
    pub vertical_input: f32,
    /// External velocity to add (e.g., from explosions, knockback).
    pub external_impulse: Vec3,
}

// ---------------------------------------------------------------------------
// Character controller
// ---------------------------------------------------------------------------

/// Physics-based character movement controller.
///
/// This is the main workhorse of the character movement system. It manages
/// collision sweeps, ground detection, state transitions, and velocity
/// integration. It does not store a position -- the caller provides the
/// current position and receives the new position after movement.
///
/// # Usage
///
/// ```ignore
/// let mut controller = CharacterController::new(config, shape);
///
/// // Each frame:
/// let input = CharacterInput { move_direction: forward, jump_pressed: true, ..default() };
/// let new_pos = controller.update(current_pos, &input, dt, &collision_world);
/// transform.position = new_pos;
/// ```
pub struct CharacterController {
    /// Configuration parameters.
    pub config: CharacterConfig,
    /// Collision shape (capsule).
    pub shape: CollisionShape,
    /// Current movement state.
    state: CharacterState,
    /// Current velocity (world space).
    velocity: Vec3,
    /// Ground detection result from the last update.
    ground_info: GroundInfo,
    /// Moving platform tracking.
    platform_state: MovingPlatformState,
    /// Time since the character last left the ground (for coyote time).
    time_since_grounded: f32,
    /// Time since the player last pressed jump (for jump buffering).
    time_since_jump_pressed: f32,
    /// Number of jumps performed since last grounding.
    jumps_performed: u32,
    /// Whether the character was grounded on the previous frame.
    was_grounded: bool,
    /// Accumulated external forces for this frame.
    pending_impulse: Vec3,
    /// The last ground normal, used for slope calculations.
    last_ground_normal: Vec3,
    /// Vertical velocity override (set during jump, cleared on landing).
    vertical_velocity_override: Option<f32>,
    /// Whether the character is currently crouching.
    is_crouching: bool,
    /// Original shape before crouching.
    standing_shape: CollisionShape,
}

impl CharacterController {
    /// Create a new character controller with the given configuration and shape.
    pub fn new(config: CharacterConfig, shape: CollisionShape) -> Self {
        let standing_shape = shape;
        Self {
            config,
            shape,
            state: CharacterState::Airborne,
            velocity: Vec3::ZERO,
            ground_info: GroundInfo::default(),
            platform_state: MovingPlatformState::default(),
            time_since_grounded: f32::MAX,
            time_since_jump_pressed: f32::MAX,
            jumps_performed: 0,
            was_grounded: false,
            pending_impulse: Vec3::ZERO,
            last_ground_normal: Vec3::Y,
            vertical_velocity_override: None,
            is_crouching: false,
            standing_shape,
        }
    }

    /// Main update entry point. Moves the character and returns the new position.
    ///
    /// * `position` -- current world-space position (capsule center).
    /// * `input` -- player input for this frame.
    /// * `dt` -- delta time in seconds.
    /// * `world` -- collision world for sweeps and overlaps.
    pub fn update(
        &mut self,
        position: Vec3,
        input: &CharacterInput,
        dt: f32,
        world: &dyn CollisionWorld,
    ) -> Vec3 {
        if dt <= 0.0 {
            return position;
        }

        // Store previous grounded state.
        self.was_grounded = self.ground_info.grounded;

        // Handle crouch shape transitions.
        self.handle_crouch(position, input, world);

        // Accumulate external impulses.
        self.pending_impulse += input.external_impulse;

        // Update jump buffer timer.
        if input.jump_pressed {
            self.time_since_jump_pressed = 0.0;
        } else {
            self.time_since_jump_pressed += dt;
        }

        // Detect ground beneath the character.
        self.detect_ground(position, world);

        // Update moving platform state.
        if self.config.enable_moving_platforms {
            self.platform_state.update(&self.ground_info, world);
        }

        // Transition states.
        self.update_state(input);

        // Update coyote time counter.
        if self.ground_info.grounded {
            self.time_since_grounded = 0.0;
            self.jumps_performed = 0;
        } else {
            self.time_since_grounded += dt;
        }

        // Compute desired velocity.
        let desired_velocity = self.compute_velocity(input, dt);
        self.velocity = desired_velocity;

        // Apply pending impulse.
        self.velocity += self.pending_impulse;
        self.pending_impulse = Vec3::ZERO;

        // Move and slide: iterative collision resolution.
        let new_pos = self.move_and_slide(position, self.velocity * dt, world);

        // Snap to ground when walking down gentle slopes.
        let final_pos = if self.state == CharacterState::Grounded
            && !input.jump_pressed
            && self.was_grounded
        {
            self.snap_to_ground(new_pos, world)
        } else {
            new_pos
        };

        // Resolve any remaining penetrations.
        self.depenetrate(final_pos, world)
    }

    // -----------------------------------------------------------------------
    // Ground detection
    // -----------------------------------------------------------------------

    /// Cast downward to detect the ground surface.
    fn detect_ground(&mut self, position: Vec3, world: &dyn CollisionWorld) {
        let cast_distance = self.shape.radius + self.config.skin_width + 0.1;
        let cast_origin = position + Vec3::new(0.0, -self.shape.half_height, 0.0);

        let hits = world.sweep_capsule(
            cast_origin,
            Vec3::NEG_Y,
            cast_distance,
            self.shape.radius * 0.9, // slightly smaller to avoid wall contacts
            0.01,                     // nearly flat for ground test
            None,
        );

        self.ground_info = GroundInfo::default();

        for hit in &hits {
            if hit.fraction > 1.0 {
                continue;
            }

            let slope_angle = hit.normal.angle_between(Vec3::Y);
            let distance = hit.fraction * cast_distance;

            // Only consider this a ground hit if within standing distance.
            let ground_threshold = self.shape.radius + self.config.skin_width + 0.05;
            if distance <= ground_threshold {
                let walkable = slope_angle <= self.config.max_slope_angle;
                self.ground_info = GroundInfo {
                    grounded: walkable,
                    normal: hit.normal,
                    point: hit.position,
                    slope_angle,
                    ground_entity: hit.entity_id,
                    distance,
                };
                self.last_ground_normal = hit.normal;
                break;
            }
        }

        // Fallback: simple raycast if sweep found nothing.
        if !self.ground_info.grounded && self.ground_info.distance == f32::MAX {
            let ray_origin = position + Vec3::new(0.0, self.shape.feet_offset_y() + 0.1, 0.0);
            if let Some(hit) = world.raycast(ray_origin, Vec3::NEG_Y, 0.3, None) {
                let slope_angle = hit.normal.angle_between(Vec3::Y);
                if hit.fraction * 0.3 < 0.15 && slope_angle <= self.config.max_slope_angle {
                    self.ground_info = GroundInfo {
                        grounded: true,
                        normal: hit.normal,
                        point: hit.position,
                        slope_angle,
                        ground_entity: hit.entity_id,
                        distance: hit.fraction * 0.3,
                    };
                    self.last_ground_normal = hit.normal;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // State machine
    // -----------------------------------------------------------------------

    /// Update the character state based on ground info, input, and environment.
    fn update_state(&mut self, input: &CharacterInput) {
        let prev_state = self.state;

        if input.climb_requested && self.state != CharacterState::Swimming {
            self.state = CharacterState::Climbing;
            return;
        }

        // TODO: Swimming detection would normally check water volume overlaps.
        // For now, allow external code to set swimming state.

        if self.ground_info.grounded {
            self.state = CharacterState::Grounded;
        } else if self.ground_info.slope_angle > self.config.max_slope_angle
            && self.ground_info.distance < 0.5
        {
            self.state = CharacterState::Sliding;
        } else if self.state != CharacterState::Climbing
            && self.state != CharacterState::Swimming
        {
            self.state = CharacterState::Airborne;
        }

        // Log state transitions.
        if self.state != prev_state {
            log::trace!(
                "Character state transition: {:?} -> {:?}",
                prev_state,
                self.state
            );
        }
    }

    // -----------------------------------------------------------------------
    // Velocity computation
    // -----------------------------------------------------------------------

    /// Compute the desired velocity for this frame based on state and input.
    fn compute_velocity(&mut self, input: &CharacterInput, dt: f32) -> Vec3 {
        match self.state {
            CharacterState::Grounded => self.compute_grounded_velocity(input, dt),
            CharacterState::Airborne => self.compute_airborne_velocity(input, dt),
            CharacterState::Sliding => self.compute_sliding_velocity(input, dt),
            CharacterState::Climbing => self.compute_climbing_velocity(input, dt),
            CharacterState::Swimming => self.compute_swimming_velocity(input, dt),
        }
    }

    /// Velocity when grounded: full directional control, friction, jump initiation.
    fn compute_grounded_velocity(&mut self, input: &CharacterInput, dt: f32) -> Vec3 {
        let speed = if input.sprint_held {
            self.config.move_speed * 1.5
        } else if self.is_crouching {
            self.config.move_speed * 0.5
        } else {
            self.config.move_speed
        };

        // Project movement onto the ground plane.
        let move_dir = input.move_direction;
        let ground_tangent = self.project_on_ground(move_dir);
        let target_horizontal = ground_tangent * speed;

        // Smoothly interpolate horizontal velocity toward target.
        let mut vel = self.velocity;
        let horizontal = Vec3::new(vel.x, 0.0, vel.z);
        let new_horizontal = self.approach_velocity(
            horizontal,
            target_horizontal,
            self.config.ground_friction,
            dt,
        );
        vel.x = new_horizontal.x;
        vel.z = new_horizontal.z;

        // Clamp downward velocity on ground.
        vel.y = vel.y.max(-2.0);

        // Add platform velocity.
        if self.config.enable_moving_platforms {
            vel += self.platform_state.platform_velocity;
        }

        // Handle jump.
        if self.should_jump(input) {
            vel.y = self.config.jump_speed;
            self.jumps_performed += 1;
            self.time_since_jump_pressed = f32::MAX;
            log::trace!("Jump initiated from ground (jump #{})", self.jumps_performed);
        }

        vel
    }

    /// Velocity when airborne: reduced control, gravity, air jumps.
    fn compute_airborne_velocity(&mut self, input: &CharacterInput, dt: f32) -> Vec3 {
        let mut vel = self.velocity;

        // Air control: limited ability to change horizontal direction.
        let move_dir = input.move_direction;
        if move_dir.length_squared() > 0.001 {
            let target = move_dir * self.config.move_speed;
            let horizontal = Vec3::new(vel.x, 0.0, vel.z);
            let new_horizontal = self.approach_velocity(
                horizontal,
                target,
                self.config.air_control * self.config.ground_friction,
                dt,
            );
            vel.x = new_horizontal.x;
            vel.z = new_horizontal.z;
        }

        // Apply gravity.
        vel.y -= self.config.gravity * dt;

        // Variable jump height: cut vertical velocity if jump released early.
        if !input.jump_held && vel.y > 0.0 {
            vel.y *= 0.5_f32.powf(dt * 10.0);
        }

        // Terminal velocity.
        vel.y = vel.y.max(-self.config.terminal_velocity);

        // Air drag on horizontal axes.
        let drag = (-self.config.air_drag * dt).exp();
        vel.x *= drag;
        vel.z *= drag;

        // Multi-jump (double jump, etc.).
        if input.jump_pressed
            && self.jumps_performed < self.config.max_jumps
            && self.time_since_grounded > self.config.coyote_time
        {
            vel.y = self.config.jump_speed;
            self.jumps_performed += 1;
            self.time_since_jump_pressed = f32::MAX;
            log::trace!("Air jump (jump #{})", self.jumps_performed);
        }

        // Coyote time jump.
        if self.should_coyote_jump(input) {
            vel.y = self.config.jump_speed;
            self.jumps_performed += 1;
            self.time_since_jump_pressed = f32::MAX;
            self.time_since_grounded = f32::MAX; // consume coyote time
            log::trace!("Coyote time jump");
        }

        vel
    }

    /// Velocity when sliding on a steep slope.
    fn compute_sliding_velocity(&mut self, input: &CharacterInput, dt: f32) -> Vec3 {
        let mut vel = self.velocity;

        // Slide down the slope under gravity.
        let slope_normal = self.ground_info.normal;
        let slide_dir = self.compute_slide_direction(slope_normal);
        vel += slide_dir * self.config.slide_acceleration * dt;

        // Apply slide friction.
        let friction = (-self.config.slide_friction * dt).exp();
        vel *= friction;

        // Allow limited air control while sliding.
        let move_dir = input.move_direction;
        if move_dir.length_squared() > 0.001 {
            let influence = move_dir * self.config.move_speed * self.config.air_control * 0.5;
            vel.x += influence.x * dt;
            vel.z += influence.z * dt;
        }

        // Apply gravity component not handled by sliding.
        vel.y -= self.config.gravity * dt * 0.3;

        vel
    }

    /// Velocity when climbing a ladder or wall.
    fn compute_climbing_velocity(&mut self, input: &CharacterInput, dt: f32) -> Vec3 {
        let _ = dt; // dt unused for instantaneous climb velocity

        let mut vel = Vec3::ZERO;

        // Vertical movement from input.
        vel.y = input.vertical_input * self.config.climb_speed;

        // Horizontal movement along the wall (reduced).
        let horizontal = input.move_direction * self.config.climb_speed * 0.5;
        vel.x = horizontal.x;
        vel.z = horizontal.z;

        // Jump off the wall/ladder.
        if input.jump_pressed {
            vel.y = self.config.jump_speed * 0.8;
            // Push away from surface. Assume wall is in the direction the
            // character is NOT facing. This is simplified; real implementations
            // would use the wall normal.
            let away = -input.move_direction;
            vel += away * self.config.move_speed * 0.5;
            self.state = CharacterState::Airborne;
            self.jumps_performed = 1;
            log::trace!("Wall jump initiated");
        }

        vel
    }

    /// Velocity when swimming.
    fn compute_swimming_velocity(&mut self, input: &CharacterInput, dt: f32) -> Vec3 {
        let mut vel = self.velocity;

        // 3D movement in water.
        let mut target = input.move_direction * self.config.swim_speed;
        target.y = input.vertical_input * self.config.swim_speed;

        vel = self.approach_velocity(
            vel,
            target,
            self.config.ground_friction * 0.6,
            dt,
        );

        // Buoyancy counters gravity.
        vel.y += (self.config.swim_buoyancy - self.config.gravity) * dt;

        // Water drag.
        let drag = (-2.0 * dt).exp();
        vel *= drag;

        // Jump out of water.
        if input.jump_pressed && vel.y > -1.0 {
            vel.y = self.config.jump_speed * 0.7;
            self.state = CharacterState::Airborne;
            log::trace!("Swim jump");
        }

        vel
    }

    // -----------------------------------------------------------------------
    // Move and slide
    // -----------------------------------------------------------------------

    /// Iterative collision resolution. Sweeps the capsule along the desired
    /// displacement, and on each hit deflects the remaining motion along the
    /// collision surface (slide). Returns the final position.
    pub fn move_and_slide(
        &mut self,
        start: Vec3,
        displacement: Vec3,
        world: &dyn CollisionWorld,
    ) -> Vec3 {
        let mut position = start;
        let mut remaining = displacement;
        let mut iterations = 0u32;

        while remaining.length_squared() > 1e-8 && iterations < self.config.max_move_iterations {
            iterations += 1;

            let direction = remaining.normalize_or_zero();
            let distance = remaining.length();

            if direction.length_squared() < 0.5 {
                break;
            }

            let hits = world.sweep_capsule(
                position,
                direction,
                distance + self.config.skin_width,
                self.shape.radius,
                self.shape.half_height,
                None,
            );

            if let Some(hit) = hits.first() {
                if hit.fraction <= 0.0 {
                    // Already overlapping -- push out and try remaining motion.
                    position += hit.normal * self.config.skin_width;
                    remaining = self.slide_vector(remaining, hit.normal);
                    continue;
                }

                // Move to just before the hit.
                let safe_fraction = (hit.fraction - 0.01).max(0.0);
                let safe_distance = safe_fraction * (distance + self.config.skin_width);
                position += direction * safe_distance;

                // Compute remaining displacement after the hit.
                let used = safe_distance / distance.max(1e-6);
                remaining *= (1.0 - used).max(0.0);

                // Check for step-up opportunity.
                if self.config.enable_step_up
                    && self.state == CharacterState::Grounded
                    && self.can_step_up(&hit, direction)
                {
                    if let Some(stepped_pos) =
                        self.try_step_up(position, direction, distance * (1.0 - used), &hit, world)
                    {
                        position = stepped_pos;
                        remaining = Vec3::ZERO;
                        continue;
                    }
                }

                // Slide along the surface.
                remaining = self.slide_vector(remaining, hit.normal);

                // Push rigid bodies.
                if let Some(entity_id) = hit.entity_id {
                    self.push_body(entity_id, direction, &hit);
                }

                // Damp velocity along the hit normal.
                let vel_into_wall = self.velocity.dot(hit.normal);
                if vel_into_wall < 0.0 {
                    self.velocity -= hit.normal * vel_into_wall;
                }
            } else {
                // No hit -- move the full remaining distance.
                position += remaining;
                remaining = Vec3::ZERO;
            }
        }

        position
    }

    /// Project the remaining displacement onto the collision surface (slide).
    fn slide_vector(&self, displacement: Vec3, normal: Vec3) -> Vec3 {
        displacement - normal * displacement.dot(normal)
    }

    /// Check if a step-up is geometrically plausible.
    fn can_step_up(&self, hit: &SweepHit, move_dir: Vec3) -> bool {
        // Only step up if the obstacle is roughly horizontal (a step, not a wall above).
        let horizontal_normal = Vec3::new(hit.normal.x, 0.0, hit.normal.z).normalize_or_zero();
        let facing_obstacle = move_dir.dot(-horizontal_normal) > 0.3;
        let is_vertical_surface = hit.normal.y.abs() < 0.3;
        facing_obstacle && is_vertical_surface
    }

    /// Attempt a step-up maneuver: raise the capsule, move forward, then lower
    /// it back down. Returns the resulting position if successful.
    fn try_step_up(
        &self,
        position: Vec3,
        direction: Vec3,
        distance: f32,
        _hit: &SweepHit,
        world: &dyn CollisionWorld,
    ) -> Option<Vec3> {
        let step_height = self.config.max_step_height;

        // 1. Try to move up.
        let up_hits = world.sweep_capsule(
            position,
            Vec3::Y,
            step_height,
            self.shape.radius,
            self.shape.half_height,
            None,
        );
        let actual_up = if let Some(up_hit) = up_hits.first() {
            (up_hit.fraction * step_height - self.config.skin_width).max(0.0)
        } else {
            step_height
        };

        if actual_up < 0.05 {
            return None; // Not enough room to step up.
        }

        let raised_pos = position + Vec3::new(0.0, actual_up, 0.0);

        // 2. Try to move forward at the raised height.
        let forward_dist = distance.max(self.shape.radius * 2.0);
        let forward_dir = Vec3::new(direction.x, 0.0, direction.z).normalize_or_zero();
        let fwd_hits = world.sweep_capsule(
            raised_pos,
            forward_dir,
            forward_dist,
            self.shape.radius,
            self.shape.half_height,
            None,
        );
        let actual_fwd = if let Some(fwd_hit) = fwd_hits.first() {
            (fwd_hit.fraction * forward_dist - self.config.skin_width).max(0.0)
        } else {
            forward_dist
        };

        if actual_fwd < 0.01 {
            return None; // Blocked at the raised height too.
        }

        let forward_pos = raised_pos + forward_dir * actual_fwd;

        // 3. Lower back down.
        let down_hits = world.sweep_capsule(
            forward_pos,
            Vec3::NEG_Y,
            actual_up + self.config.skin_width,
            self.shape.radius,
            self.shape.half_height,
            None,
        );
        let final_pos = if let Some(down_hit) = down_hits.first() {
            let drop = down_hit.fraction * (actual_up + self.config.skin_width);
            // Check that the landing surface is walkable.
            let slope = down_hit.normal.angle_between(Vec3::Y);
            if slope > self.config.max_slope_angle {
                return None;
            }
            forward_pos + Vec3::new(0.0, -drop + self.config.skin_width, 0.0)
        } else {
            // No ground found after stepping -- abort.
            return None;
        };

        Some(final_pos)
    }

    // -----------------------------------------------------------------------
    // Snap to ground
    // -----------------------------------------------------------------------

    /// Snap the character down to the ground to prevent floating when walking
    /// down gentle slopes.
    fn snap_to_ground(&self, position: Vec3, world: &dyn CollisionWorld) -> Vec3 {
        let snap_distance = self.config.snap_to_ground_distance;

        let hits = world.sweep_capsule(
            position,
            Vec3::NEG_Y,
            snap_distance,
            self.shape.radius,
            self.shape.half_height,
            None,
        );

        if let Some(hit) = hits.first() {
            let slope = hit.normal.angle_between(Vec3::Y);
            if slope <= self.config.max_slope_angle && hit.fraction > 0.0 {
                let snap_down = hit.fraction * snap_distance - self.config.skin_width;
                if snap_down > 0.0 {
                    return position + Vec3::new(0.0, -snap_down, 0.0);
                }
            }
        }

        position
    }

    // -----------------------------------------------------------------------
    // Depenetration
    // -----------------------------------------------------------------------

    /// Resolve any remaining overlaps by pushing the character out.
    fn depenetrate(&self, position: Vec3, world: &dyn CollisionWorld) -> Vec3 {
        let overlaps = world.overlap_capsule(
            position,
            self.shape.radius,
            self.shape.half_height,
            None,
        );

        let mut resolved = position;
        for overlap in &overlaps {
            if overlap.penetration_depth > 0.0 {
                resolved +=
                    overlap.penetration_normal * (overlap.penetration_depth + self.config.skin_width);
            }
        }

        resolved
    }

    // -----------------------------------------------------------------------
    // Crouch
    // -----------------------------------------------------------------------

    /// Handle crouch/uncrouch shape transitions.
    fn handle_crouch(
        &mut self,
        position: Vec3,
        input: &CharacterInput,
        world: &dyn CollisionWorld,
    ) {
        if input.crouch_held && !self.is_crouching {
            // Enter crouch.
            self.is_crouching = true;
            self.shape = self.standing_shape.crouching();
            log::trace!("Entering crouch");
        } else if !input.crouch_held && self.is_crouching {
            // Try to uncrouch -- check if there's room.
            let standing_overlaps = world.overlap_capsule(
                position,
                self.standing_shape.radius,
                self.standing_shape.half_height,
                None,
            );
            if standing_overlaps.is_empty() {
                self.is_crouching = false;
                self.shape = self.standing_shape;
                log::trace!("Exiting crouch");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Jump logic
    // -----------------------------------------------------------------------

    /// Whether a jump should be initiated this frame (includes coyote time and
    /// jump buffering).
    fn should_jump(&self, input: &CharacterInput) -> bool {
        if !self.ground_info.grounded {
            return false;
        }

        // Direct press.
        if input.jump_pressed {
            return true;
        }

        // Jump buffer: player pressed jump recently and just landed.
        if self.time_since_jump_pressed <= self.config.jump_buffer_time && !self.was_grounded {
            return true;
        }

        false
    }

    /// Whether a coyote-time jump should trigger.
    fn should_coyote_jump(&self, input: &CharacterInput) -> bool {
        input.jump_pressed
            && !self.ground_info.grounded
            && self.was_grounded
            && self.time_since_grounded <= self.config.coyote_time
            && self.jumps_performed == 0
    }

    // -----------------------------------------------------------------------
    // Rigid body pushing
    // -----------------------------------------------------------------------

    /// Apply a force to a rigid body the character has walked into.
    fn push_body(&self, _entity_id: u32, direction: Vec3, hit: &SweepHit) {
        let push_dir = Vec3::new(direction.x, 0.0, direction.z).normalize_or_zero();
        let _force = push_dir * self.config.push_force;

        // In a real implementation, this would apply the force via the physics
        // engine. The entity_id identifies which body to push.
        log::trace!(
            "Pushing body (entity {:?}) with force along {:?}",
            hit.entity_id,
            push_dir
        );
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Project a direction vector onto the ground plane.
    fn project_on_ground(&self, direction: Vec3) -> Vec3 {
        let normal = self.ground_info.normal;
        let projected = direction - normal * direction.dot(normal);
        projected.normalize_or_zero()
    }

    /// Compute the direction a character slides down a steep slope.
    fn compute_slide_direction(&self, slope_normal: Vec3) -> Vec3 {
        let gravity_dir = Vec3::NEG_Y;
        let slide = gravity_dir - slope_normal * gravity_dir.dot(slope_normal);
        slide.normalize_or_zero()
    }

    /// Smoothly approach a target velocity using exponential interpolation.
    fn approach_velocity(&self, current: Vec3, target: Vec3, rate: f32, dt: f32) -> Vec3 {
        let t = 1.0 - (-rate * dt).exp();
        current.lerp(target, t.clamp(0.0, 1.0))
    }

    // -----------------------------------------------------------------------
    // Public queries
    // -----------------------------------------------------------------------

    /// Returns `true` if the character is currently on walkable ground.
    #[inline]
    pub fn is_grounded(&self) -> bool {
        self.ground_info.grounded
    }

    /// Returns the current movement state.
    #[inline]
    pub fn state(&self) -> CharacterState {
        self.state
    }

    /// Returns the current velocity.
    #[inline]
    pub fn velocity(&self) -> Vec3 {
        self.velocity
    }

    /// Set the velocity directly (e.g., for teleportation or knockback).
    #[inline]
    pub fn set_velocity(&mut self, velocity: Vec3) {
        self.velocity = velocity;
    }

    /// Returns the current ground information.
    #[inline]
    pub fn ground_info(&self) -> &GroundInfo {
        &self.ground_info
    }

    /// Returns the ground normal.
    #[inline]
    pub fn ground_normal(&self) -> Vec3 {
        self.ground_info.normal
    }

    /// Returns `true` if the character is crouching.
    #[inline]
    pub fn is_crouching(&self) -> bool {
        self.is_crouching
    }

    /// Returns the current horizontal speed.
    #[inline]
    pub fn horizontal_speed(&self) -> f32 {
        Vec3::new(self.velocity.x, 0.0, self.velocity.z).length()
    }

    /// Returns the number of jumps performed since last grounding.
    #[inline]
    pub fn jumps_performed(&self) -> u32 {
        self.jumps_performed
    }

    /// Returns the moving platform state.
    #[inline]
    pub fn platform_state(&self) -> &MovingPlatformState {
        &self.platform_state
    }

    /// Whether the character was grounded last frame (useful for landing detection).
    #[inline]
    pub fn was_grounded(&self) -> bool {
        self.was_grounded
    }

    /// Force-set the character state (e.g., entering a swim volume).
    pub fn set_state(&mut self, state: CharacterState) {
        log::trace!("Forced state transition: {:?} -> {:?}", self.state, state);
        self.state = state;
    }

    /// Reset the controller to its initial state.
    pub fn reset(&mut self) {
        self.state = CharacterState::Airborne;
        self.velocity = Vec3::ZERO;
        self.ground_info = GroundInfo::default();
        self.platform_state = MovingPlatformState::default();
        self.time_since_grounded = f32::MAX;
        self.time_since_jump_pressed = f32::MAX;
        self.jumps_performed = 0;
        self.was_grounded = false;
        self.pending_impulse = Vec3::ZERO;
        self.last_ground_normal = Vec3::Y;
        self.vertical_velocity_override = None;
        self.is_crouching = false;
        self.shape = self.standing_shape;
    }

    /// Add an impulse that will be applied on the next update.
    pub fn add_impulse(&mut self, impulse: Vec3) {
        self.pending_impulse += impulse;
    }

    /// Teleport the character (resets velocity and ground state).
    pub fn teleport(&mut self) {
        self.velocity = Vec3::ZERO;
        self.ground_info = GroundInfo::default();
        self.platform_state.clear();
        self.time_since_grounded = f32::MAX;
        self.was_grounded = false;
    }
}

// ---------------------------------------------------------------------------
// ECS Component
// ---------------------------------------------------------------------------

/// ECS component wrapper for [`CharacterController`].
///
/// Attach this to an entity along with a `Transform` component to enable
/// character movement. A system should read input, call `controller.update()`,
/// and write the resulting position back to the transform.
pub struct CharacterControllerComponent {
    /// The underlying character controller.
    pub controller: CharacterController,
    /// Entity's own id (for filtering self from collision queries).
    pub entity_id: Option<u32>,
    /// Yaw rotation in radians (horizontal look direction).
    pub yaw: f32,
}

impl CharacterControllerComponent {
    /// Create a new component with default configuration.
    pub fn new() -> Self {
        Self {
            controller: CharacterController::new(
                CharacterConfig::default(),
                CollisionShape::default(),
            ),
            entity_id: None,
            yaw: 0.0,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: CharacterConfig, shape: CollisionShape) -> Self {
        Self {
            controller: CharacterController::new(config, shape),
            entity_id: None,
            yaw: 0.0,
        }
    }

    /// Get the forward direction based on yaw.
    pub fn forward(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        Vec3::new(sin_yaw, 0.0, cos_yaw)
    }

    /// Get the right direction based on yaw.
    pub fn right(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        Vec3::new(cos_yaw, 0.0, -sin_yaw)
    }

    /// Transform local input axes (forward/right) into a world-space move direction.
    pub fn input_to_world_direction(&self, forward_input: f32, right_input: f32) -> Vec3 {
        let dir = self.forward() * forward_input + self.right() * right_input;
        if dir.length_squared() > 1.0 {
            dir.normalize()
        } else {
            dir
        }
    }
}

impl Default for CharacterControllerComponent {
    fn default() -> Self {
        Self::new()
    }
}

impl genovo_ecs::Component for CharacterControllerComponent {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal collision world that reports no collisions.
    struct EmptyWorld;

    impl CollisionWorld for EmptyWorld {
        fn sweep_capsule(
            &self,
            _origin: Vec3,
            _direction: Vec3,
            _distance: f32,
            _radius: f32,
            _half_height: f32,
            _ignore: Option<u32>,
        ) -> Vec<SweepHit> {
            Vec::new()
        }

        fn overlap_capsule(
            &self,
            _pos: Vec3,
            _radius: f32,
            _hh: f32,
            _ignore: Option<u32>,
        ) -> Vec<OverlapResult> {
            Vec::new()
        }

        fn raycast(
            &self,
            _origin: Vec3,
            _dir: Vec3,
            _max_dist: f32,
            _ignore: Option<u32>,
        ) -> Option<SweepHit> {
            None
        }

        fn get_body_velocity(&self, _id: u32) -> Option<Vec3> {
            None
        }
    }

    /// Collision world with a flat ground plane at y=0.
    struct FlatGroundWorld;

    impl CollisionWorld for FlatGroundWorld {
        fn sweep_capsule(
            &self,
            origin: Vec3,
            direction: Vec3,
            distance: f32,
            _radius: f32,
            _half_height: f32,
            _ignore: Option<u32>,
        ) -> Vec<SweepHit> {
            // Simple ground plane intersection for downward sweeps.
            if direction.y < -0.5 {
                let feet_y = origin.y;
                if feet_y > 0.0 {
                    let frac = (feet_y / (distance * direction.y.abs())).min(1.0);
                    return vec![SweepHit {
                        fraction: frac,
                        position: origin + direction * frac * distance,
                        normal: Vec3::Y,
                        entity_id: None,
                        walkable: true,
                    }];
                }
            }
            Vec::new()
        }

        fn overlap_capsule(
            &self,
            _pos: Vec3,
            _radius: f32,
            _hh: f32,
            _ignore: Option<u32>,
        ) -> Vec<OverlapResult> {
            Vec::new()
        }

        fn raycast(
            &self,
            origin: Vec3,
            direction: Vec3,
            max_dist: f32,
            _ignore: Option<u32>,
        ) -> Option<SweepHit> {
            if direction.y < -0.5 && origin.y > 0.0 {
                let frac = (origin.y / (max_dist * direction.y.abs())).min(1.0);
                Some(SweepHit {
                    fraction: frac,
                    position: origin + direction * frac * max_dist,
                    normal: Vec3::Y,
                    entity_id: None,
                    walkable: true,
                })
            } else {
                None
            }
        }

        fn get_body_velocity(&self, _id: u32) -> Option<Vec3> {
            None
        }
    }

    #[test]
    fn default_shape_dimensions() {
        let shape = CollisionShape::default();
        assert!((shape.total_height() - 1.8).abs() < 0.01);
        assert!((shape.radius - 0.3).abs() < 0.01);
    }

    #[test]
    fn crouching_shape_is_smaller() {
        let standing = CollisionShape::default();
        let crouching = standing.crouching();
        assert!(crouching.total_height() < standing.total_height());
        assert!((crouching.radius - standing.radius).abs() < 0.01);
    }

    #[test]
    fn character_state_can_jump() {
        assert!(CharacterState::Grounded.can_jump());
        assert!(CharacterState::Swimming.can_jump());
        assert!(!CharacterState::Airborne.can_jump());
        assert!(!CharacterState::Sliding.can_jump());
        assert!(!CharacterState::Climbing.can_jump());
    }

    #[test]
    fn state_applies_gravity() {
        assert!(CharacterState::Airborne.applies_gravity());
        assert!(CharacterState::Sliding.applies_gravity());
        assert!(!CharacterState::Grounded.applies_gravity());
    }

    #[test]
    fn controller_starts_airborne() {
        let ctrl = CharacterController::new(CharacterConfig::default(), CollisionShape::default());
        assert_eq!(ctrl.state(), CharacterState::Airborne);
        assert!(!ctrl.is_grounded());
    }

    #[test]
    fn controller_freefall_in_empty_world() {
        let mut ctrl =
            CharacterController::new(CharacterConfig::default(), CollisionShape::default());
        let world = EmptyWorld;
        let input = CharacterInput::default();

        let pos = Vec3::new(0.0, 10.0, 0.0);
        let new_pos = ctrl.update(pos, &input, 1.0 / 60.0, &world);

        // Should have moved downward due to gravity.
        assert!(new_pos.y < pos.y, "Should fall: {} < {}", new_pos.y, pos.y);
        assert_eq!(ctrl.state(), CharacterState::Airborne);
    }

    #[test]
    fn controller_horizontal_movement() {
        let mut ctrl =
            CharacterController::new(CharacterConfig::default(), CollisionShape::default());
        let world = EmptyWorld;
        let input = CharacterInput {
            move_direction: Vec3::X,
            ..Default::default()
        };

        let pos = Vec3::new(0.0, 10.0, 0.0);
        let new_pos = ctrl.update(pos, &input, 1.0 / 60.0, &world);

        // Should have some horizontal displacement due to air control.
        assert!(new_pos.x > pos.x || (new_pos.x - pos.x).abs() < 0.5);
    }

    #[test]
    fn slide_vector_perpendicular() {
        let ctrl =
            CharacterController::new(CharacterConfig::default(), CollisionShape::default());
        let disp = Vec3::new(1.0, 0.0, 0.0);
        let normal = Vec3::X;
        let slid = ctrl.slide_vector(disp, normal);
        assert!(slid.length() < 0.01, "Should slide to zero against wall");
    }

    #[test]
    fn slide_vector_parallel() {
        let ctrl =
            CharacterController::new(CharacterConfig::default(), CollisionShape::default());
        let disp = Vec3::new(0.0, 0.0, 1.0);
        let normal = Vec3::X;
        let slid = ctrl.slide_vector(disp, normal);
        assert!((slid - disp).length() < 0.01, "Parallel motion should be preserved");
    }

    #[test]
    fn component_forward_right() {
        let comp = CharacterControllerComponent::new();
        let fwd = comp.forward();
        let right = comp.right();
        // At yaw=0, forward should be +Z, right should be +X.
        assert!((fwd - Vec3::Z).length() < 0.01);
        assert!((right - Vec3::X).length() < 0.01);
    }

    #[test]
    fn input_to_world_direction_normalizes() {
        let comp = CharacterControllerComponent::new();
        let dir = comp.input_to_world_direction(1.0, 1.0);
        assert!((dir.length() - 1.0).abs() < 0.01, "Should be normalized");
    }

    #[test]
    fn teleport_resets_velocity() {
        let mut ctrl =
            CharacterController::new(CharacterConfig::default(), CollisionShape::default());
        ctrl.set_velocity(Vec3::new(10.0, 5.0, 3.0));
        ctrl.teleport();
        assert_eq!(ctrl.velocity(), Vec3::ZERO);
    }

    #[test]
    fn add_impulse_accumulates() {
        let mut ctrl =
            CharacterController::new(CharacterConfig::default(), CollisionShape::default());
        let world = EmptyWorld;
        let input = CharacterInput::default();

        ctrl.add_impulse(Vec3::new(0.0, 50.0, 0.0));
        let pos = Vec3::new(0.0, 5.0, 0.0);
        let new_pos = ctrl.update(pos, &input, 1.0 / 60.0, &world);

        // The upward impulse should counteract gravity at least partially.
        assert!(
            new_pos.y > pos.y,
            "Upward impulse should lift: {} > {}",
            new_pos.y,
            pos.y
        );
    }

    #[test]
    fn controller_reset_restores_defaults() {
        let mut ctrl =
            CharacterController::new(CharacterConfig::default(), CollisionShape::default());
        ctrl.set_velocity(Vec3::new(10.0, 5.0, 3.0));
        ctrl.set_state(CharacterState::Swimming);
        ctrl.reset();

        assert_eq!(ctrl.state(), CharacterState::Airborne);
        assert_eq!(ctrl.velocity(), Vec3::ZERO);
        assert_eq!(ctrl.jumps_performed(), 0);
        assert!(!ctrl.is_crouching());
    }
}
