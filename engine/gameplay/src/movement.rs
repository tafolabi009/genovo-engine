//! Complete movement system with Quake-style acceleration, multiple movement modes,
//! and physically-grounded locomotion.
//!
//! This module provides a high-level movement framework suitable for first-person
//! shooters, third-person action games, and platformers. It implements:
//!
//! - **Quake-style ground and air acceleration**: the classic acceleration model
//!   from Quake/Source engines that produces satisfying movement feel including
//!   bunny-hopping and air strafing
//! - **Multiple movement modes**: Walking, Running, Sprinting, Crouching, Swimming,
//!   Flying, and Climbing -- each with its own speed, acceleration, and constraints
//! - **Ground friction model**: velocity-dependent friction with separate static
//!   and dynamic coefficients
//! - **Air movement**: limited air control with separate air acceleration, enabling
//!   air strafing when wish direction differs from velocity
//! - **Water movement**: buoyancy, drag, and surface detection
//! - **Ladder movement**: vertical climbing with dismount
//!
//! # Quake-Style Acceleration
//!
//! The core acceleration function `accelerate()` implements the algorithm from
//! Quake III Arena (and later Source Engine). The key insight is that acceleration
//! is applied in the component of the wish direction that exceeds the current
//! velocity projection, which naturally produces:
//!
//! - Speed capping at `wish_speed` when moving in a straight line
//! - Air strafing: diagonal input while airborne curves the player's path and
//!   can increase speed beyond the normal cap
//! - Bunny-hopping: repeated jumps preserve horizontal speed from air strafing
//!
//! ```text
//! current_speed = dot(velocity, wish_dir)
//! add_speed = clamp(wish_speed - current_speed, 0, accel * wish_speed * dt)
//! velocity += wish_dir * add_speed
//! ```

use glam::Vec3;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Movement mode
// ---------------------------------------------------------------------------

/// Movement modes that affect speed, acceleration, and physics behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MovementMode {
    /// Normal walking speed.
    Walking,
    /// Faster than walking, default locomotion.
    Running,
    /// Maximum speed on foot, limited by stamina.
    Sprinting,
    /// Reduced speed and height, can fit through small spaces.
    Crouching,
    /// 3D movement in water, affected by buoyancy and drag.
    Swimming,
    /// Free 3D movement (noclip, jetpack, spectator).
    Flying,
    /// Vertical movement on ladders and climbable surfaces.
    Climbing,
}

impl MovementMode {
    /// Whether this mode allows full 3D movement (not constrained to ground plane).
    pub fn is_3d(&self) -> bool {
        matches!(self, Self::Swimming | Self::Flying)
    }

    /// Whether this mode is affected by gravity.
    pub fn has_gravity(&self) -> bool {
        matches!(self, Self::Walking | Self::Running | Self::Sprinting | Self::Crouching)
    }

    /// Whether this mode allows jumping.
    pub fn can_jump(&self) -> bool {
        matches!(
            self,
            Self::Walking | Self::Running | Self::Sprinting | Self::Crouching | Self::Swimming
        )
    }

    /// Whether the player can transition from this mode to sprinting.
    pub fn can_sprint(&self) -> bool {
        matches!(self, Self::Walking | Self::Running)
    }
}

impl Default for MovementMode {
    fn default() -> Self {
        Self::Running
    }
}

// ---------------------------------------------------------------------------
// Movement mode config
// ---------------------------------------------------------------------------

/// Per-mode speed and acceleration parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeConfig {
    /// Maximum speed in this mode (units/sec).
    pub max_speed: f32,
    /// Acceleration rate (units/sec^2).
    pub acceleration: f32,
    /// Deceleration rate when no input (units/sec^2).
    pub deceleration: f32,
    /// Multiplier applied to the base max_speed.
    pub speed_multiplier: f32,
    /// Whether the player can jump in this mode.
    pub can_jump: bool,
    /// Optional height multiplier (e.g., 0.5 for crouching).
    pub height_multiplier: f32,
    /// Optional FOV multiplier (e.g., 1.1 for sprinting).
    pub fov_multiplier: f32,
    /// Stamina cost per second (for sprinting).
    pub stamina_cost: f32,
}

impl Default for ModeConfig {
    fn default() -> Self {
        Self {
            max_speed: 6.0,
            acceleration: 10.0,
            deceleration: 10.0,
            speed_multiplier: 1.0,
            can_jump: true,
            height_multiplier: 1.0,
            fov_multiplier: 1.0,
            stamina_cost: 0.0,
        }
    }
}

impl ModeConfig {
    /// Create a walking mode config.
    pub fn walking() -> Self {
        Self {
            max_speed: 4.0,
            acceleration: 10.0,
            deceleration: 10.0,
            speed_multiplier: 1.0,
            ..Default::default()
        }
    }

    /// Create a running mode config.
    pub fn running() -> Self {
        Self {
            max_speed: 6.0,
            acceleration: 10.0,
            deceleration: 10.0,
            speed_multiplier: 1.0,
            ..Default::default()
        }
    }

    /// Create a sprinting mode config.
    pub fn sprinting() -> Self {
        Self {
            max_speed: 9.0,
            acceleration: 12.0,
            deceleration: 8.0,
            speed_multiplier: 1.5,
            fov_multiplier: 1.1,
            stamina_cost: 20.0,
            ..Default::default()
        }
    }

    /// Create a crouching mode config.
    pub fn crouching() -> Self {
        Self {
            max_speed: 3.0,
            acceleration: 8.0,
            deceleration: 12.0,
            speed_multiplier: 0.5,
            height_multiplier: 0.5,
            ..Default::default()
        }
    }

    /// Create a swimming mode config.
    pub fn swimming() -> Self {
        Self {
            max_speed: 4.0,
            acceleration: 6.0,
            deceleration: 4.0,
            speed_multiplier: 0.7,
            ..Default::default()
        }
    }

    /// Create a flying mode config.
    pub fn flying() -> Self {
        Self {
            max_speed: 12.0,
            acceleration: 15.0,
            deceleration: 8.0,
            speed_multiplier: 2.0,
            can_jump: false,
            ..Default::default()
        }
    }

    /// Create a climbing mode config.
    pub fn climbing() -> Self {
        Self {
            max_speed: 3.0,
            acceleration: 8.0,
            deceleration: 15.0,
            speed_multiplier: 0.5,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Movement component
// ---------------------------------------------------------------------------

/// Core movement parameters attached to an entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovementComponent {
    /// Base movement speed (units/sec).
    pub base_speed: f32,
    /// Ground acceleration (Quake-style `sv_accelerate`).
    pub ground_acceleration: f32,
    /// Ground deceleration / friction stop speed.
    pub ground_deceleration: f32,
    /// Air acceleration (Quake-style `sv_airaccelerate`).
    pub air_acceleration: f32,
    /// Maximum speed the player can reach.
    pub max_speed: f32,
    /// Gravity scale multiplier (1.0 = normal gravity).
    pub gravity_scale: f32,
    /// Jump velocity (units/sec upward).
    pub jump_speed: f32,
    /// Current velocity.
    pub velocity: Vec3,
    /// Current movement mode.
    pub mode: MovementMode,
    /// Per-mode configurations.
    pub mode_configs: ModeConfigs,
    /// Whether the entity is on the ground.
    pub grounded: bool,
    /// Ground surface normal (valid only when grounded).
    pub ground_normal: Vec3,
    /// Current stamina (for sprinting).
    pub stamina: f32,
    /// Maximum stamina.
    pub max_stamina: f32,
    /// Stamina regeneration rate (units/sec).
    pub stamina_regen: f32,
    /// Surface friction coefficient (from the ground material).
    pub surface_friction: f32,
    /// External velocity to add this frame (knockback, explosions, etc.).
    pub external_impulse: Vec3,
}

/// Per-mode configuration container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeConfigs {
    pub walking: ModeConfig,
    pub running: ModeConfig,
    pub sprinting: ModeConfig,
    pub crouching: ModeConfig,
    pub swimming: ModeConfig,
    pub flying: ModeConfig,
    pub climbing: ModeConfig,
}

impl Default for ModeConfigs {
    fn default() -> Self {
        Self {
            walking: ModeConfig::walking(),
            running: ModeConfig::running(),
            sprinting: ModeConfig::sprinting(),
            crouching: ModeConfig::crouching(),
            swimming: ModeConfig::swimming(),
            flying: ModeConfig::flying(),
            climbing: ModeConfig::climbing(),
        }
    }
}

impl ModeConfigs {
    /// Get the config for a given mode.
    pub fn get(&self, mode: MovementMode) -> &ModeConfig {
        match mode {
            MovementMode::Walking => &self.walking,
            MovementMode::Running => &self.running,
            MovementMode::Sprinting => &self.sprinting,
            MovementMode::Crouching => &self.crouching,
            MovementMode::Swimming => &self.swimming,
            MovementMode::Flying => &self.flying,
            MovementMode::Climbing => &self.climbing,
        }
    }

    /// Get a mutable reference to the config for a given mode.
    pub fn get_mut(&mut self, mode: MovementMode) -> &mut ModeConfig {
        match mode {
            MovementMode::Walking => &mut self.walking,
            MovementMode::Running => &mut self.running,
            MovementMode::Sprinting => &mut self.sprinting,
            MovementMode::Crouching => &mut self.crouching,
            MovementMode::Swimming => &mut self.swimming,
            MovementMode::Flying => &mut self.flying,
            MovementMode::Climbing => &mut self.climbing,
        }
    }
}

impl Default for MovementComponent {
    fn default() -> Self {
        Self {
            base_speed: 6.0,
            ground_acceleration: 10.0,
            ground_deceleration: 10.0,
            air_acceleration: 1.0,
            max_speed: 30.0,
            gravity_scale: 1.0,
            jump_speed: 8.0,
            velocity: Vec3::ZERO,
            mode: MovementMode::Running,
            mode_configs: ModeConfigs::default(),
            grounded: false,
            ground_normal: Vec3::Y,
            stamina: 100.0,
            max_stamina: 100.0,
            stamina_regen: 15.0,
            surface_friction: 1.0,
            external_impulse: Vec3::ZERO,
        }
    }
}

impl MovementComponent {
    /// Create a new movement component with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom base speed.
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.base_speed = speed;
        self
    }

    /// Create with custom gravity scale.
    pub fn with_gravity_scale(mut self, scale: f32) -> Self {
        self.gravity_scale = scale;
        self
    }

    /// Create with custom jump speed.
    pub fn with_jump_speed(mut self, speed: f32) -> Self {
        self.jump_speed = speed;
        self
    }

    /// Get the effective max speed for the current mode.
    pub fn effective_max_speed(&self) -> f32 {
        let config = self.mode_configs.get(self.mode);
        config.max_speed * config.speed_multiplier
    }

    /// Get the effective acceleration for the current mode.
    pub fn effective_acceleration(&self) -> f32 {
        self.mode_configs.get(self.mode).acceleration
    }

    /// Get horizontal speed (XZ plane).
    pub fn horizontal_speed(&self) -> f32 {
        Vec3::new(self.velocity.x, 0.0, self.velocity.z).length()
    }

    /// Get vertical speed (Y component).
    pub fn vertical_speed(&self) -> f32 {
        self.velocity.y
    }

    /// Whether the entity is moving (above a small threshold).
    pub fn is_moving(&self) -> bool {
        self.velocity.length_squared() > 0.01
    }

    /// Whether the entity can sprint right now.
    pub fn can_sprint(&self) -> bool {
        self.mode.can_sprint() && self.stamina > 0.0
    }

    /// Add an external impulse (accumulated until consumed by the movement system).
    pub fn add_impulse(&mut self, impulse: Vec3) {
        self.external_impulse += impulse;
    }
}

// ---------------------------------------------------------------------------
// Movement input
// ---------------------------------------------------------------------------

/// Per-frame movement input.
#[derive(Debug, Clone, Default)]
pub struct MovementInput {
    /// Desired movement direction (normalized, world-space or view-space
    /// depending on the system configuration).
    pub wish_direction: Vec3,
    /// Whether the jump button was pressed this frame.
    pub jump: bool,
    /// Whether the crouch button is held.
    pub crouch: bool,
    /// Whether the sprint button is held.
    pub sprint: bool,
    /// Vertical input for climbing/swimming (-1 to 1).
    pub vertical: f32,
    /// Whether to enter flying mode.
    pub fly: bool,
    /// Whether to enter climbing mode.
    pub climb: bool,
}

// ---------------------------------------------------------------------------
// Ground movement (Quake-style)
// ---------------------------------------------------------------------------

/// Quake-style ground movement functions.
pub struct GroundMovement;

impl GroundMovement {
    /// Apply ground friction to a velocity vector.
    ///
    /// Implements the Quake/Source friction model:
    /// 1. If speed < stop_speed, apply extra deceleration to bring to zero.
    /// 2. Otherwise, reduce speed by `friction * dt`.
    ///
    /// This produces a natural deceleration curve where the player comes to a
    /// crisp stop rather than sliding forever.
    pub fn apply_ground_friction(velocity: Vec3, friction: f32, stop_speed: f32, dt: f32) -> Vec3 {
        let speed = velocity.length();
        if speed < 0.001 {
            return Vec3::ZERO;
        }

        // The Quake friction model uses the higher of the current speed and
        // the stop speed to compute the drop amount. This ensures that at low
        // speeds the deceleration is at least stop_speed * friction * dt,
        // which brings the player to a crisp stop.
        let control = if speed < stop_speed { stop_speed } else { speed };
        let drop = control * friction * dt;

        let new_speed = (speed - drop).max(0.0);
        if new_speed < 0.001 {
            return Vec3::ZERO;
        }

        velocity * (new_speed / speed)
    }

    /// Quake-style acceleration: accelerate along wish_dir up to wish_speed.
    ///
    /// This is the core of Quake movement physics. The function computes the
    /// component of velocity along the wish direction, and adds acceleration
    /// only for the deficit (wish_speed minus that component). This naturally
    /// caps speed in the wish direction while allowing perpendicular velocity
    /// to remain -- which is what enables air strafing.
    ///
    /// ```text
    /// current_speed = dot(current_vel, wish_dir)
    /// add_speed = wish_speed - current_speed
    /// if add_speed <= 0: return current_vel  // already at or above wish speed
    /// accel_speed = min(accel * wish_speed * dt, add_speed)
    /// return current_vel + wish_dir * accel_speed
    /// ```
    pub fn accelerate(
        current_vel: Vec3,
        wish_dir: Vec3,
        wish_speed: f32,
        accel: f32,
        dt: f32,
    ) -> Vec3 {
        // Project current velocity onto wish direction.
        let current_speed = current_vel.dot(wish_dir);

        // Compute the speed to add. If we're already going faster than wish
        // speed in the wish direction, don't add anything.
        let add_speed = wish_speed - current_speed;
        if add_speed <= 0.0 {
            return current_vel;
        }

        // Compute the actual acceleration for this frame, capped at add_speed
        // to prevent overshooting.
        let accel_speed = (accel * wish_speed * dt).min(add_speed);

        current_vel + wish_dir * accel_speed
    }

    /// Compute the full ground movement update.
    ///
    /// 1. Apply friction to the current velocity.
    /// 2. Apply Quake-style acceleration toward the wish direction.
    /// 3. Clamp to max speed.
    pub fn update(
        current_vel: Vec3,
        wish_dir: Vec3,
        wish_speed: f32,
        accel: f32,
        friction: f32,
        stop_speed: f32,
        max_speed: f32,
        dt: f32,
    ) -> Vec3 {
        // Apply friction first.
        let vel_after_friction = Self::apply_ground_friction(current_vel, friction, stop_speed, dt);

        // Accelerate toward wish direction.
        let vel_after_accel = Self::accelerate(vel_after_friction, wish_dir, wish_speed, accel, dt);

        // Clamp to max speed.
        let speed = vel_after_accel.length();
        if speed > max_speed {
            vel_after_accel * (max_speed / speed)
        } else {
            vel_after_accel
        }
    }

    /// Project a velocity vector onto the ground plane defined by a normal.
    pub fn project_on_ground(velocity: Vec3, ground_normal: Vec3) -> Vec3 {
        velocity - ground_normal * velocity.dot(ground_normal)
    }

    /// Clip velocity against a surface normal (remove the component into the surface).
    pub fn clip_velocity(velocity: Vec3, normal: Vec3, overbounce: f32) -> Vec3 {
        let backoff = velocity.dot(normal) * overbounce;
        velocity - normal * backoff
    }
}

// ---------------------------------------------------------------------------
// Air movement
// ---------------------------------------------------------------------------

/// Air movement with limited air control and separate acceleration.
pub struct AirMovement;

impl AirMovement {
    /// Air strafe acceleration -- the Quake air acceleration model.
    ///
    /// Air acceleration in Quake uses the same `accelerate()` function but with
    /// a much lower `accel` value (typically 1.0 vs 10.0 for ground). This
    /// severely limits how fast you can change direction in the air, but because
    /// the function only caps speed in the wish direction, moving diagonally
    /// (holding forward + strafe) can increase total speed. This is the
    /// foundation of air strafing and bunny-hopping.
    pub fn accelerate(
        current_vel: Vec3,
        wish_dir: Vec3,
        wish_speed: f32,
        air_accel: f32,
        dt: f32,
    ) -> Vec3 {
        // Use the same Quake acceleration function, but with the air accel value.
        GroundMovement::accelerate(current_vel, wish_dir, wish_speed, air_accel, dt)
    }

    /// Apply air drag to horizontal velocity.
    pub fn apply_air_drag(velocity: Vec3, drag: f32, dt: f32) -> Vec3 {
        let factor = (-drag * dt).exp();
        Vec3::new(velocity.x * factor, velocity.y, velocity.z * factor)
    }

    /// Full air movement update.
    ///
    /// 1. Apply Quake-style air acceleration.
    /// 2. Apply gravity.
    /// 3. Apply air drag.
    /// 4. Clamp to terminal velocity.
    pub fn update(
        current_vel: Vec3,
        wish_dir: Vec3,
        wish_speed: f32,
        air_accel: f32,
        gravity: f32,
        air_drag: f32,
        terminal_velocity: f32,
        dt: f32,
    ) -> Vec3 {
        // Air acceleration.
        let mut vel = Self::accelerate(current_vel, wish_dir, wish_speed, air_accel, dt);

        // Gravity.
        vel.y -= gravity * dt;

        // Air drag on horizontal axes.
        vel = Self::apply_air_drag(vel, air_drag, dt);

        // Terminal velocity.
        vel.y = vel.y.max(-terminal_velocity);

        vel
    }

    /// Compute the air strafe speed gain.
    ///
    /// Returns the speed gain from a single frame of air strafing at 45 degrees.
    /// This is useful for UI feedback (showing speed in HUD).
    pub fn compute_strafe_gain(
        current_speed: f32,
        wish_speed: f32,
        air_accel: f32,
        dt: f32,
    ) -> f32 {
        // At 45 degrees, the projection onto wish_dir is current_speed * cos(45).
        let current_proj = current_speed * std::f32::consts::FRAC_1_SQRT_2;
        let add = wish_speed - current_proj;
        if add <= 0.0 {
            return 0.0;
        }
        (air_accel * wish_speed * dt).min(add)
    }
}

// ---------------------------------------------------------------------------
// Water movement
// ---------------------------------------------------------------------------

/// Water / swimming movement.
pub struct WaterMovement;

impl WaterMovement {
    /// Buoyancy force computation.
    ///
    /// Returns the upward force based on submersion depth and water density.
    pub fn compute_buoyancy(submersion_depth: f32, water_density: f32, body_volume: f32) -> f32 {
        let submerged_fraction = submersion_depth.clamp(0.0, 1.0);
        submerged_fraction * water_density * body_volume * 9.81
    }

    /// Water drag.
    pub fn apply_water_drag(velocity: Vec3, drag_coefficient: f32, dt: f32) -> Vec3 {
        let factor = (-drag_coefficient * dt).exp();
        velocity * factor
    }

    /// Swim speed with depth-based drag adjustment.
    pub fn swim_speed(base_speed: f32, depth: f32) -> f32 {
        // Speed reduces with depth (pressure / visibility).
        let depth_factor = 1.0 / (1.0 + depth * 0.05);
        base_speed * depth_factor
    }

    /// Surface detection: returns true if the entity is near the water surface.
    pub fn is_near_surface(entity_y: f32, water_surface_y: f32, threshold: f32) -> bool {
        (entity_y - water_surface_y).abs() < threshold
    }

    /// Full water movement update.
    pub fn update(
        current_vel: Vec3,
        wish_dir: Vec3,
        wish_speed: f32,
        accel: f32,
        buoyancy: f32,
        gravity: f32,
        drag: f32,
        dt: f32,
    ) -> Vec3 {
        // Accelerate toward wish direction (full 3D).
        let mut vel = GroundMovement::accelerate(current_vel, wish_dir, wish_speed, accel, dt);

        // Apply buoyancy (counteracts gravity).
        vel.y += (buoyancy - gravity) * dt;

        // Apply water drag.
        vel = Self::apply_water_drag(vel, drag, dt);

        vel
    }
}

// ---------------------------------------------------------------------------
// Ladder movement
// ---------------------------------------------------------------------------

/// Ladder / climbing movement.
pub struct LadderMovement;

impl LadderMovement {
    /// Compute ladder movement velocity.
    ///
    /// - `vertical_input`: -1 to 1 (down to up)
    /// - `horizontal_input`: movement direction along the ladder (left/right)
    /// - `ladder_normal`: the outward-facing normal of the ladder surface
    pub fn compute_velocity(
        vertical_input: f32,
        horizontal_input: Vec3,
        climb_speed: f32,
        ladder_normal: Vec3,
    ) -> Vec3 {
        let mut vel = Vec3::ZERO;

        // Vertical climbing.
        vel.y = vertical_input * climb_speed;

        // Horizontal movement along the ladder (perpendicular to normal).
        let lateral = horizontal_input - ladder_normal * horizontal_input.dot(ladder_normal);
        vel += lateral.normalize_or_zero() * climb_speed * 0.5;

        vel
    }

    /// Check if the player should dismount the ladder.
    ///
    /// Dismounts when the player jumps or moves away from the ladder.
    pub fn should_dismount(
        wish_dir: Vec3,
        ladder_normal: Vec3,
        jump_pressed: bool,
    ) -> bool {
        if jump_pressed {
            return true;
        }

        // Moving away from the ladder (wish direction has significant component
        // along the ladder normal).
        let away = wish_dir.dot(ladder_normal);
        away > 0.5
    }

    /// Compute the dismount velocity (jump off the ladder).
    pub fn dismount_velocity(
        ladder_normal: Vec3,
        up_speed: f32,
        push_speed: f32,
    ) -> Vec3 {
        ladder_normal * push_speed + Vec3::Y * up_speed
    }
}

// ---------------------------------------------------------------------------
// Movement system
// ---------------------------------------------------------------------------

/// ECS system that applies movement logic to all entities with a
/// `MovementComponent` each frame.
///
/// The system reads `MovementInput`, updates the velocity in the
/// `MovementComponent`, and produces a displacement vector that should be
/// applied to the entity's position (via a character controller or directly).
pub struct MovementSystem {
    /// Gravity vector (default: 9.81 m/s^2 downward).
    pub gravity: f32,
    /// Terminal velocity (maximum downward speed).
    pub terminal_velocity: f32,
    /// Friction stop speed threshold.
    pub stop_speed: f32,
    /// Default ground friction when no surface material is specified.
    pub default_friction: f32,
    /// Default air drag.
    pub air_drag: f32,
    /// Water buoyancy force.
    pub water_buoyancy: f32,
    /// Water drag coefficient.
    pub water_drag: f32,
    /// Maximum speed multiplier (hard cap on all movement).
    pub speed_cap: f32,
}

impl Default for MovementSystem {
    fn default() -> Self {
        Self {
            gravity: 9.81,
            terminal_velocity: 50.0,
            stop_speed: 1.0,
            default_friction: 6.0,
            air_drag: 0.1,
            water_buoyancy: 12.0,
            water_drag: 3.0,
            speed_cap: 100.0,
        }
    }
}

impl MovementSystem {
    /// Create a new movement system with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom gravity.
    pub fn with_gravity(mut self, gravity: f32) -> Self {
        self.gravity = gravity;
        self
    }

    /// Process a single entity's movement for one frame.
    ///
    /// Returns the displacement vector to apply to the entity's position.
    pub fn update_entity(
        &self,
        component: &mut MovementComponent,
        input: &MovementInput,
        dt: f32,
    ) -> Vec3 {
        if dt <= 0.0 {
            return Vec3::ZERO;
        }

        // Mode transitions.
        self.update_mode(component, input);

        // Update stamina.
        self.update_stamina(component, dt);

        // Get current mode config.
        let mode_config = component.mode_configs.get(component.mode).clone();
        let effective_speed = mode_config.max_speed * mode_config.speed_multiplier;

        // Compute wish direction and speed.
        let wish_dir = input.wish_direction.normalize_or_zero();
        let wish_speed = if wish_dir.length_squared() > 0.1 {
            effective_speed
        } else {
            0.0
        };

        // Update velocity based on current mode.
        let new_velocity = match component.mode {
            MovementMode::Walking | MovementMode::Running | MovementMode::Sprinting | MovementMode::Crouching => {
                if component.grounded {
                    self.update_ground_velocity(component, wish_dir, wish_speed, &mode_config, dt)
                } else {
                    self.update_air_velocity(component, wish_dir, wish_speed, dt)
                }
            }
            MovementMode::Swimming => {
                self.update_swim_velocity(component, input, wish_dir, wish_speed, dt)
            }
            MovementMode::Flying => {
                self.update_fly_velocity(component, input, wish_dir, wish_speed, &mode_config, dt)
            }
            MovementMode::Climbing => {
                self.update_climb_velocity(component, input, wish_speed, dt)
            }
        };

        component.velocity = new_velocity;

        // Apply external impulse.
        component.velocity += component.external_impulse;
        component.external_impulse = Vec3::ZERO;

        // Apply jump.
        if input.jump && component.mode.can_jump() && component.grounded {
            component.velocity.y = component.jump_speed;
            component.grounded = false;
        }

        // Speed cap.
        let speed = component.velocity.length();
        if speed > self.speed_cap {
            component.velocity *= self.speed_cap / speed;
        }

        // Return displacement.
        component.velocity * dt
    }

    /// Update movement mode based on input.
    fn update_mode(&self, component: &mut MovementComponent, input: &MovementInput) {
        let prev_mode = component.mode;

        // Handle special mode requests.
        if input.fly {
            component.mode = MovementMode::Flying;
            return;
        }
        if input.climb {
            component.mode = MovementMode::Climbing;
            return;
        }

        // Don't auto-transition out of swimming/flying/climbing.
        if matches!(
            component.mode,
            MovementMode::Swimming | MovementMode::Flying | MovementMode::Climbing
        ) {
            return;
        }

        // Ground-based mode transitions.
        if input.crouch {
            component.mode = MovementMode::Crouching;
        } else if input.sprint && component.can_sprint() {
            component.mode = MovementMode::Sprinting;
        } else if input.wish_direction.length_squared() > 0.1 {
            component.mode = MovementMode::Running;
        } else {
            component.mode = MovementMode::Walking;
        }

        if component.mode != prev_mode {
            log::trace!(
                "Movement mode: {:?} -> {:?}",
                prev_mode,
                component.mode
            );
        }
    }

    /// Update stamina (consumed by sprinting, regenerated otherwise).
    fn update_stamina(&self, component: &mut MovementComponent, dt: f32) {
        let config = component.mode_configs.get(component.mode);

        if config.stamina_cost > 0.0 {
            component.stamina -= config.stamina_cost * dt;
            if component.stamina <= 0.0 {
                component.stamina = 0.0;
                // Force out of sprint when stamina depleted.
                if component.mode == MovementMode::Sprinting {
                    component.mode = MovementMode::Running;
                }
            }
        } else {
            // Regenerate stamina.
            component.stamina =
                (component.stamina + component.stamina_regen * dt).min(component.max_stamina);
        }
    }

    /// Ground movement using Quake-style acceleration.
    fn update_ground_velocity(
        &self,
        component: &MovementComponent,
        wish_dir: Vec3,
        wish_speed: f32,
        mode_config: &ModeConfig,
        dt: f32,
    ) -> Vec3 {
        let friction = component.surface_friction * self.default_friction;
        let accel = mode_config.acceleration;

        // Project current velocity onto the ground plane.
        let horizontal_vel = Vec3::new(component.velocity.x, 0.0, component.velocity.z);

        // Quake-style ground update: friction then acceleration.
        let new_horizontal = GroundMovement::update(
            horizontal_vel,
            wish_dir,
            wish_speed,
            accel,
            friction,
            self.stop_speed,
            component.max_speed,
            dt,
        );

        // Keep vertical velocity minimal when grounded.
        Vec3::new(new_horizontal.x, component.velocity.y.max(-1.0), new_horizontal.z)
    }

    /// Air movement using Quake-style air acceleration.
    fn update_air_velocity(
        &self,
        component: &MovementComponent,
        wish_dir: Vec3,
        wish_speed: f32,
        dt: f32,
    ) -> Vec3 {
        AirMovement::update(
            component.velocity,
            wish_dir,
            wish_speed,
            component.air_acceleration,
            self.gravity * component.gravity_scale,
            self.air_drag,
            self.terminal_velocity,
            dt,
        )
    }

    /// Swimming movement.
    fn update_swim_velocity(
        &self,
        component: &MovementComponent,
        input: &MovementInput,
        wish_dir: Vec3,
        wish_speed: f32,
        dt: f32,
    ) -> Vec3 {
        // Full 3D wish direction including vertical input.
        let mut full_wish = wish_dir;
        full_wish.y = input.vertical;
        let full_wish = full_wish.normalize_or_zero();

        WaterMovement::update(
            component.velocity,
            full_wish,
            wish_speed,
            component.mode_configs.get(MovementMode::Swimming).acceleration,
            self.water_buoyancy,
            self.gravity * component.gravity_scale,
            self.water_drag,
            dt,
        )
    }

    /// Flying movement (noclip / jetpack).
    fn update_fly_velocity(
        &self,
        component: &MovementComponent,
        input: &MovementInput,
        wish_dir: Vec3,
        wish_speed: f32,
        mode_config: &ModeConfig,
        dt: f32,
    ) -> Vec3 {
        // Full 3D wish direction.
        let mut full_wish = wish_dir;
        full_wish.y = input.vertical;
        let full_wish = full_wish.normalize_or_zero();

        // Apply acceleration without friction in fly mode.
        let accel = mode_config.acceleration;
        let vel = GroundMovement::accelerate(
            component.velocity,
            full_wish,
            wish_speed,
            accel,
            dt,
        );

        // Apply slight drag to prevent infinite speed buildup.
        let drag = (-0.5 * dt).exp();
        if full_wish.length_squared() < 0.01 {
            vel * drag
        } else {
            vel
        }
    }

    /// Climbing movement.
    fn update_climb_velocity(
        &self,
        component: &MovementComponent,
        input: &MovementInput,
        wish_speed: f32,
        dt: f32,
    ) -> Vec3 {
        let climb_speed = component.mode_configs.get(MovementMode::Climbing).max_speed;

        // Simple ladder movement: vertical from input, horizontal limited.
        let mut vel = Vec3::ZERO;
        vel.y = input.vertical * climb_speed;

        let horizontal = Vec3::new(
            input.wish_direction.x,
            0.0,
            input.wish_direction.z,
        );
        vel += horizontal.normalize_or_zero() * climb_speed * 0.5;

        // Smooth toward target.
        let t = 1.0 - (-10.0 * dt).exp();
        component.velocity.lerp(vel, t)
    }

    /// Set the gravity for this system.
    pub fn set_gravity(&mut self, gravity: f32) {
        self.gravity = gravity;
    }

    /// Process multiple entities in batch.
    pub fn update_batch(
        &self,
        entities: &mut [(MovementComponent, MovementInput)],
        dt: f32,
    ) -> Vec<Vec3> {
        entities
            .iter_mut()
            .map(|(component, input)| self.update_entity(component, input, dt))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quake_accelerate_from_zero() {
        let vel = GroundMovement::accelerate(Vec3::ZERO, Vec3::X, 6.0, 10.0, 1.0 / 60.0);
        assert!(vel.x > 0.0, "Should accelerate along wish dir");
        assert!(vel.x <= 6.0, "Should not exceed wish speed");
    }

    #[test]
    fn quake_accelerate_caps_at_wish_speed() {
        let mut vel = Vec3::ZERO;
        for _ in 0..600 {
            vel = GroundMovement::accelerate(vel, Vec3::X, 6.0, 10.0, 1.0 / 60.0);
        }
        assert!(
            (vel.x - 6.0).abs() < 0.5,
            "Should converge to wish speed: {}",
            vel.x
        );
    }

    #[test]
    fn quake_accelerate_air_strafe() {
        // Moving forward at max speed, air strafe to the right should increase total speed.
        let vel = Vec3::new(0.0, 0.0, 6.0); // moving forward at 6 units/sec
        let wish_dir = Vec3::X; // strafing right
        let new_vel = AirMovement::accelerate(vel, wish_dir, 6.0, 1.0, 1.0 / 60.0);

        let old_speed = vel.length();
        let new_speed = new_vel.length();
        assert!(
            new_speed >= old_speed - 0.01,
            "Air strafe should not reduce speed: {} vs {}",
            new_speed,
            old_speed
        );
    }

    #[test]
    fn ground_friction_stops_player() {
        let vel = Vec3::new(1.0, 0.0, 0.0);
        let stopped = GroundMovement::apply_ground_friction(vel, 10.0, 2.0, 1.0);
        assert!(stopped.length() < vel.length(), "Friction should slow down");
    }

    #[test]
    fn ground_friction_zero_velocity() {
        let vel = Vec3::ZERO;
        let result = GroundMovement::apply_ground_friction(vel, 10.0, 2.0, 1.0 / 60.0);
        assert_eq!(result, Vec3::ZERO);
    }

    #[test]
    fn air_movement_applies_gravity() {
        let vel = AirMovement::update(
            Vec3::ZERO,
            Vec3::ZERO,
            0.0,
            1.0,
            9.81,
            0.0,
            50.0,
            1.0,
        );
        assert!(vel.y < 0.0, "Gravity should make velocity negative");
    }

    #[test]
    fn air_movement_terminal_velocity() {
        let mut vel = Vec3::new(0.0, -100.0, 0.0);
        vel = AirMovement::update(vel, Vec3::ZERO, 0.0, 1.0, 9.81, 0.0, 50.0, 1.0);
        assert!(
            vel.y >= -50.0,
            "Should be clamped to terminal velocity: {}",
            vel.y
        );
    }

    #[test]
    fn movement_mode_defaults() {
        assert_eq!(MovementMode::default(), MovementMode::Running);
        assert!(MovementMode::Swimming.is_3d());
        assert!(MovementMode::Flying.is_3d());
        assert!(!MovementMode::Running.is_3d());
    }

    #[test]
    fn movement_component_effective_speed() {
        let comp = MovementComponent::new();
        let speed = comp.effective_max_speed();
        assert!(speed > 0.0);
    }

    #[test]
    fn movement_system_ground_movement() {
        let system = MovementSystem::new();
        let mut comp = MovementComponent::new();
        comp.grounded = true;
        comp.velocity = Vec3::ZERO;

        let input = MovementInput {
            wish_direction: Vec3::X,
            ..Default::default()
        };

        let displacement = system.update_entity(&mut comp, &input, 1.0 / 60.0);
        assert!(displacement.x > 0.0, "Should move in wish direction");
    }

    #[test]
    fn movement_system_jump() {
        let system = MovementSystem::new();
        let mut comp = MovementComponent::new();
        comp.grounded = true;

        let input = MovementInput {
            jump: true,
            ..Default::default()
        };

        system.update_entity(&mut comp, &input, 1.0 / 60.0);
        assert!(comp.velocity.y > 0.0, "Jump should give upward velocity");
        assert!(!comp.grounded, "Should no longer be grounded after jump");
    }

    #[test]
    fn movement_system_sprint_stamina() {
        let system = MovementSystem::new();
        let mut comp = MovementComponent::new();
        comp.grounded = true;
        comp.stamina = 10.0;

        let input = MovementInput {
            sprint: true,
            wish_direction: Vec3::Z,
            ..Default::default()
        };

        // Sprint should consume stamina.
        system.update_entity(&mut comp, &input, 1.0);
        assert!(comp.stamina < 10.0, "Sprint should consume stamina");
    }

    #[test]
    fn water_movement_buoyancy() {
        let vel = WaterMovement::update(
            Vec3::ZERO,
            Vec3::ZERO,
            0.0,
            6.0,
            15.0,
            9.81,
            3.0,
            1.0 / 60.0,
        );
        assert!(vel.y > 0.0, "Buoyancy should push upward");
    }

    #[test]
    fn water_drag_reduces_speed() {
        let vel = Vec3::new(10.0, 5.0, 3.0);
        let dragged = WaterMovement::apply_water_drag(vel, 3.0, 1.0 / 60.0);
        assert!(dragged.length() < vel.length());
    }

    #[test]
    fn ladder_dismount_on_jump() {
        assert!(LadderMovement::should_dismount(Vec3::ZERO, Vec3::Z, true));
    }

    #[test]
    fn ladder_dismount_moving_away() {
        assert!(LadderMovement::should_dismount(Vec3::Z, Vec3::Z, false));
    }

    #[test]
    fn ladder_no_dismount_when_climbing() {
        assert!(!LadderMovement::should_dismount(Vec3::Y, Vec3::Z, false));
    }

    #[test]
    fn clip_velocity() {
        let vel = Vec3::new(5.0, -3.0, 0.0);
        let normal = Vec3::Y;
        let clipped = GroundMovement::clip_velocity(vel, normal, 1.0);
        assert!((clipped.y).abs() < 0.001, "Vertical component should be clipped");
        assert!((clipped.x - 5.0).abs() < 0.001, "Horizontal preserved");
    }

    #[test]
    fn project_on_ground_flat() {
        let vel = Vec3::new(5.0, -1.0, 3.0);
        let projected = GroundMovement::project_on_ground(vel, Vec3::Y);
        assert!((projected.y).abs() < 0.001);
        assert!((projected.x - 5.0).abs() < 0.001);
    }

    #[test]
    fn mode_config_presets() {
        let walking = ModeConfig::walking();
        let sprinting = ModeConfig::sprinting();
        assert!(sprinting.max_speed > walking.max_speed);
        assert!(sprinting.stamina_cost > 0.0);
    }

    #[test]
    fn movement_batch_update() {
        let system = MovementSystem::new();
        let mut entities = vec![
            (
                MovementComponent::new(),
                MovementInput {
                    wish_direction: Vec3::X,
                    ..Default::default()
                },
            ),
            (
                MovementComponent::new(),
                MovementInput {
                    wish_direction: Vec3::Z,
                    ..Default::default()
                },
            ),
        ];
        entities[0].0.grounded = true;
        entities[1].0.grounded = true;

        let displacements = system.update_batch(&mut entities, 1.0 / 60.0);
        assert_eq!(displacements.len(), 2);
    }

    #[test]
    fn external_impulse_applied() {
        let system = MovementSystem::new();
        let mut comp = MovementComponent::new();
        comp.grounded = true;
        comp.add_impulse(Vec3::new(0.0, 50.0, 0.0));

        let input = MovementInput::default();
        system.update_entity(&mut comp, &input, 1.0 / 60.0);
        assert!(
            comp.velocity.y > 0.0,
            "External impulse should be applied"
        );
    }

    #[test]
    fn strafe_gain_calculation() {
        let gain = AirMovement::compute_strafe_gain(6.0, 6.0, 1.0, 1.0 / 60.0);
        assert!(gain > 0.0, "Should have positive strafe gain at 45 degrees");
    }
}
