//! FPS/TPS weapon system with fire modes, recoil, projectiles, hitscan, and melee.
//!
//! Provides a complete weapon framework suitable for first-person shooters,
//! third-person shooters, and action RPGs. Features include:
//!
//! - **Multiple fire modes**: Single, Burst, and Automatic
//! - **Weapon state machine**: Idle, Firing, Reloading, Switching, Cooldown
//! - **Recoil system**: random rotation within a configurable spread cone
//! - **Hitscan weapons**: instant raycast with falloff and penetration
//! - **Projectile weapons**: moving entities with lifetime, gravity, and collision
//! - **Melee weapons**: sphere overlap in front of the player with combo support
//! - **Ammo and magazine management**: reload timers, reserve ammo
//! - **Weapon inventory**: equip, switch, drop, and pick up weapons
//!
//! # State Machine
//!
//! Each weapon has a state machine that governs what actions are available:
//!
//! ```text
//! Idle -> Firing -> Cooldown -> Idle
//!      -> Reloading -> Idle
//!      -> Switching -> Idle
//! ```
//!
//! State transitions are validated: you cannot fire while reloading, cannot
//! reload while switching, etc.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Fire mode
// ---------------------------------------------------------------------------

/// How a weapon fires when the trigger is held.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FireMode {
    /// One shot per trigger press.
    Single,
    /// Multiple shots per trigger press with a pause between bursts.
    Burst(u32),
    /// Continuous fire while trigger is held.
    Automatic,
}

impl FireMode {
    /// Number of shots per activation.
    pub fn shots_per_activation(&self) -> u32 {
        match self {
            Self::Single => 1,
            Self::Burst(count) => *count,
            Self::Automatic => 1, // Continuous, one per fire interval.
        }
    }

    /// Whether the weapon fires continuously while held.
    pub fn is_automatic(&self) -> bool {
        matches!(self, Self::Automatic)
    }
}

impl Default for FireMode {
    fn default() -> Self {
        Self::Single
    }
}

// ---------------------------------------------------------------------------
// Weapon state
// ---------------------------------------------------------------------------

/// Current state of a weapon in its state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeaponState {
    /// Ready to fire or perform other actions.
    Idle,
    /// Currently firing (animation/effect playing).
    Firing,
    /// Cooldown between shots (fire rate limiter).
    Cooldown,
    /// Reloading the magazine.
    Reloading,
    /// Being equipped/switching to this weapon.
    Switching,
}

impl WeaponState {
    /// Whether the weapon can initiate a fire action.
    pub fn can_fire(&self) -> bool {
        matches!(self, Self::Idle)
    }

    /// Whether the weapon can initiate a reload.
    pub fn can_reload(&self) -> bool {
        matches!(self, Self::Idle | Self::Cooldown)
    }

    /// Whether the weapon can be switched away from.
    pub fn can_switch(&self) -> bool {
        matches!(self, Self::Idle | Self::Cooldown)
    }

    /// Whether the weapon is busy (cannot accept most inputs).
    pub fn is_busy(&self) -> bool {
        matches!(self, Self::Firing | Self::Reloading | Self::Switching)
    }
}

impl Default for WeaponState {
    fn default() -> Self {
        Self::Idle
    }
}

// ---------------------------------------------------------------------------
// Weapon type
// ---------------------------------------------------------------------------

/// Category of weapon determining how damage is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeaponType {
    /// Instant raycast -- damage applied immediately along a ray.
    Hitscan,
    /// Spawns a moving projectile entity.
    Projectile,
    /// Close-range sphere overlap attack.
    Melee,
}

// ---------------------------------------------------------------------------
// Recoil pattern
// ---------------------------------------------------------------------------

/// Configures how recoil is applied after each shot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoilPattern {
    /// Vertical recoil per shot (pitch up, in degrees).
    pub vertical: f32,
    /// Horizontal recoil per shot (yaw left/right, in degrees).
    pub horizontal: f32,
    /// Recovery speed (degrees/sec returning to center).
    pub recovery_speed: f32,
    /// Random spread added per shot (degrees, cone half-angle).
    pub spread_per_shot: f32,
    /// Maximum accumulated spread (degrees).
    pub max_spread: f32,
    /// Spread recovery speed (degrees/sec).
    pub spread_recovery: f32,
    /// Recoil pattern: sequence of (vertical, horizontal) offsets.
    /// If empty, random recoil within the configured range is used.
    pub pattern: Vec<(f32, f32)>,
    /// Current index in the recoil pattern.
    pub pattern_index: usize,
}

impl Default for RecoilPattern {
    fn default() -> Self {
        Self {
            vertical: 1.5,
            horizontal: 0.3,
            recovery_speed: 5.0,
            spread_per_shot: 0.5,
            max_spread: 5.0,
            spread_recovery: 3.0,
            pattern: Vec::new(),
            pattern_index: 0,
        }
    }
}

impl RecoilPattern {
    /// Get the recoil for the next shot.
    pub fn next_recoil(&mut self) -> (f32, f32) {
        if self.pattern.is_empty() {
            // Random recoil within range.
            let v = self.vertical;
            let h = self.horizontal * (2.0 * pseudo_random_f32(self.pattern_index as u32) - 1.0);
            self.pattern_index += 1;
            (v, h)
        } else {
            let idx = self.pattern_index % self.pattern.len();
            self.pattern_index += 1;
            self.pattern[idx]
        }
    }

    /// Reset the pattern index (e.g., after stopping fire).
    pub fn reset(&mut self) {
        self.pattern_index = 0;
    }
}

/// Simple deterministic pseudo-random float from a seed, in [0, 1).
fn pseudo_random_f32(seed: u32) -> f32 {
    let x = seed.wrapping_mul(2654435761);
    let x = x ^ (x >> 16);
    (x & 0x00FFFFFF) as f32 / 16777216.0
}

// ---------------------------------------------------------------------------
// Weapon definition
// ---------------------------------------------------------------------------

/// Complete weapon definition with all parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Weapon {
    /// Unique weapon identifier.
    pub id: u32,
    /// Display name.
    pub name: String,
    /// Weapon type (Hitscan, Projectile, Melee).
    pub weapon_type: WeaponType,
    /// Base damage per shot/hit.
    pub damage: f32,
    /// Fire rate in rounds per minute.
    pub fire_rate: f32,
    /// Time to reload in seconds.
    pub reload_time: f32,
    /// Magazine capacity (shots before reload).
    pub ammo_capacity: u32,
    /// Current ammo in magazine.
    pub current_ammo: u32,
    /// Reserve ammo (total ammo not in magazine).
    pub reserve_ammo: u32,
    /// Maximum reserve ammo.
    pub max_reserve_ammo: u32,
    /// Base spread angle in degrees (accuracy cone half-angle).
    pub spread: f32,
    /// Effective range in units.
    pub range: f32,
    /// Fire mode.
    pub fire_mode: FireMode,
    /// Current weapon state.
    pub state: WeaponState,
    /// Timer for state transitions (counts down to 0).
    pub state_timer: f32,
    /// Recoil configuration.
    pub recoil: RecoilPattern,
    /// Current accumulated spread (increases when firing, recovers over time).
    pub current_spread: f32,
    /// Damage type identifier (matches the damage system's DamageType).
    pub damage_type: u32,
    /// Headshot damage multiplier.
    pub headshot_multiplier: f32,
    /// Number of pellets per shot (for shotguns).
    pub pellet_count: u32,
    /// Projectile speed (for projectile weapons, units/sec).
    pub projectile_speed: f32,
    /// Projectile gravity scale (for projectile weapons).
    pub projectile_gravity: f32,
    /// Melee attack duration (seconds).
    pub melee_duration: f32,
    /// Melee attack radius (for sphere overlap).
    pub melee_radius: f32,
    /// Weapon switch time (seconds to equip).
    pub switch_time: f32,
    /// Burst delay between shots in a burst (seconds).
    pub burst_delay: f32,
    /// Remaining shots in the current burst.
    pub burst_remaining: u32,
    /// Whether the weapon has unlimited ammo.
    pub infinite_ammo: bool,
    /// Damage falloff start distance (full damage up to here).
    pub falloff_start: f32,
    /// Damage falloff end distance (minimum damage at this range).
    pub falloff_end: f32,
    /// Minimum damage multiplier at max falloff.
    pub falloff_min_multiplier: f32,
    /// Number of surfaces this weapon can penetrate (hitscan only).
    pub penetration: u32,
    /// Damage multiplier per penetration.
    pub penetration_falloff: f32,
}

impl Weapon {
    /// Create a new weapon with the given parameters.
    pub fn new(id: u32, name: impl Into<String>, weapon_type: WeaponType) -> Self {
        Self {
            id,
            name: name.into(),
            weapon_type,
            damage: 25.0,
            fire_rate: 600.0,
            reload_time: 2.0,
            ammo_capacity: 30,
            current_ammo: 30,
            reserve_ammo: 90,
            max_reserve_ammo: 180,
            spread: 1.0,
            range: 100.0,
            fire_mode: FireMode::Automatic,
            state: WeaponState::Idle,
            state_timer: 0.0,
            recoil: RecoilPattern::default(),
            current_spread: 0.0,
            damage_type: 0,
            headshot_multiplier: 2.0,
            pellet_count: 1,
            projectile_speed: 50.0,
            projectile_gravity: 0.0,
            melee_duration: 0.3,
            melee_radius: 2.0,
            switch_time: 0.5,
            burst_delay: 0.05,
            burst_remaining: 0,
            infinite_ammo: false,
            falloff_start: 30.0,
            falloff_end: 80.0,
            falloff_min_multiplier: 0.5,
            penetration: 0,
            penetration_falloff: 0.7,
        }
    }

    /// Fire interval in seconds (derived from fire_rate in RPM).
    pub fn fire_interval(&self) -> f32 {
        if self.fire_rate <= 0.0 {
            return 1.0;
        }
        60.0 / self.fire_rate
    }

    /// Whether the weapon has ammo to fire.
    pub fn has_ammo(&self) -> bool {
        self.infinite_ammo || self.current_ammo > 0
    }

    /// Whether the weapon can be reloaded (not full, has reserve).
    pub fn can_reload(&self) -> bool {
        self.current_ammo < self.ammo_capacity
            && (self.reserve_ammo > 0 || self.infinite_ammo)
            && self.state.can_reload()
    }

    /// Whether the magazine is full.
    pub fn is_full(&self) -> bool {
        self.current_ammo >= self.ammo_capacity
    }

    /// Get the effective damage at a given distance.
    pub fn damage_at_distance(&self, distance: f32) -> f32 {
        if distance <= self.falloff_start {
            self.damage
        } else if distance >= self.falloff_end {
            self.damage * self.falloff_min_multiplier
        } else {
            let t = (distance - self.falloff_start) / (self.falloff_end - self.falloff_start);
            self.damage * (1.0 - t * (1.0 - self.falloff_min_multiplier))
        }
    }

    /// Get the total spread (base + accumulated).
    pub fn total_spread(&self) -> f32 {
        self.spread + self.current_spread
    }

    /// Create a standard assault rifle.
    pub fn assault_rifle(id: u32) -> Self {
        Self::new(id, "Assault Rifle", WeaponType::Hitscan)
    }

    /// Create a standard shotgun.
    pub fn shotgun(id: u32) -> Self {
        Self {
            damage: 8.0,
            fire_rate: 80.0,
            reload_time: 3.0,
            ammo_capacity: 8,
            current_ammo: 8,
            reserve_ammo: 32,
            max_reserve_ammo: 64,
            spread: 5.0,
            range: 30.0,
            fire_mode: FireMode::Single,
            pellet_count: 12,
            falloff_start: 10.0,
            falloff_end: 25.0,
            ..Self::new(id, "Shotgun", WeaponType::Hitscan)
        }
    }

    /// Create a standard sniper rifle.
    pub fn sniper_rifle(id: u32) -> Self {
        Self {
            damage: 100.0,
            fire_rate: 40.0,
            reload_time: 3.5,
            ammo_capacity: 5,
            current_ammo: 5,
            reserve_ammo: 20,
            max_reserve_ammo: 40,
            spread: 0.1,
            range: 500.0,
            fire_mode: FireMode::Single,
            headshot_multiplier: 3.0,
            falloff_start: 100.0,
            falloff_end: 400.0,
            falloff_min_multiplier: 0.8,
            penetration: 2,
            ..Self::new(id, "Sniper Rifle", WeaponType::Hitscan)
        }
    }

    /// Create a rocket launcher.
    pub fn rocket_launcher(id: u32) -> Self {
        Self {
            damage: 120.0,
            fire_rate: 60.0,
            reload_time: 2.5,
            ammo_capacity: 4,
            current_ammo: 4,
            reserve_ammo: 16,
            max_reserve_ammo: 32,
            spread: 0.5,
            range: 200.0,
            fire_mode: FireMode::Single,
            projectile_speed: 30.0,
            projectile_gravity: 0.5,
            ..Self::new(id, "Rocket Launcher", WeaponType::Projectile)
        }
    }

    /// Create a melee weapon (knife/sword).
    pub fn melee_weapon(id: u32, name: impl Into<String>) -> Self {
        Self {
            damage: 50.0,
            fire_rate: 120.0,
            reload_time: 0.0,
            ammo_capacity: 0,
            current_ammo: 0,
            reserve_ammo: 0,
            max_reserve_ammo: 0,
            spread: 0.0,
            range: 3.0,
            fire_mode: FireMode::Single,
            infinite_ammo: true,
            melee_duration: 0.25,
            melee_radius: 1.5,
            ..Self::new(id, name, WeaponType::Melee)
        }
    }

    /// Create a burst-fire weapon.
    pub fn burst_rifle(id: u32) -> Self {
        Self {
            damage: 30.0,
            fire_rate: 800.0,
            reload_time: 2.2,
            ammo_capacity: 24,
            current_ammo: 24,
            reserve_ammo: 72,
            max_reserve_ammo: 144,
            spread: 0.8,
            range: 80.0,
            fire_mode: FireMode::Burst(3),
            burst_delay: 0.06,
            ..Self::new(id, "Burst Rifle", WeaponType::Hitscan)
        }
    }

    /// Create a pistol.
    pub fn pistol(id: u32) -> Self {
        Self {
            damage: 20.0,
            fire_rate: 400.0,
            reload_time: 1.5,
            ammo_capacity: 12,
            current_ammo: 12,
            reserve_ammo: 48,
            max_reserve_ammo: 96,
            spread: 1.5,
            range: 50.0,
            fire_mode: FireMode::Single,
            ..Self::new(id, "Pistol", WeaponType::Hitscan)
        }
    }
}

// ---------------------------------------------------------------------------
// Fire event
// ---------------------------------------------------------------------------

/// Event produced when a weapon fires.
#[derive(Debug, Clone)]
pub struct FireEvent {
    /// Weapon that produced this event.
    pub weapon_id: u32,
    /// Entity that owns the weapon.
    pub owner_entity: u32,
    /// World-space origin of the shot.
    pub origin: Vec3,
    /// Direction of the shot (after spread is applied).
    pub direction: Vec3,
    /// Spread angle that was applied (degrees).
    pub spread: f32,
    /// Damage of this shot.
    pub damage: f32,
    /// Projectile speed (0 for hitscan).
    pub projectile_speed: f32,
    /// Weapon type.
    pub weapon_type: WeaponType,
    /// Maximum range.
    pub range: f32,
    /// Number of pellets (for shotguns).
    pub pellet_count: u32,
    /// Penetration count.
    pub penetration: u32,
}

// ---------------------------------------------------------------------------
// Hitscan result
// ---------------------------------------------------------------------------

/// Result of a hitscan weapon trace.
#[derive(Debug, Clone)]
pub struct HitscanResult {
    /// Whether the trace hit anything.
    pub hit: bool,
    /// World-space hit point.
    pub point: Vec3,
    /// Surface normal at the hit.
    pub normal: Vec3,
    /// Distance from origin to hit.
    pub distance: f32,
    /// Entity that was hit (if any).
    pub hit_entity: Option<u32>,
    /// Damage dealt (after falloff and penetration).
    pub damage: f32,
    /// Whether this was a headshot.
    pub headshot: bool,
    /// Material at the hit point.
    pub material_id: u32,
}

// ---------------------------------------------------------------------------
// Projectile
// ---------------------------------------------------------------------------

/// A moving projectile entity spawned by a projectile weapon.
#[derive(Debug, Clone)]
pub struct Projectile {
    /// Unique identifier.
    pub id: u32,
    /// Weapon that spawned this projectile.
    pub weapon_id: u32,
    /// Entity that owns the weapon.
    pub owner_entity: u32,
    /// Current world-space position.
    pub position: Vec3,
    /// Current velocity (direction * speed).
    pub velocity: Vec3,
    /// Damage on impact.
    pub damage: f32,
    /// Remaining lifetime in seconds.
    pub lifetime: f32,
    /// Collision radius (for sphere-cast collision detection).
    pub radius: f32,
    /// Gravity scale applied to this projectile.
    pub gravity_scale: f32,
    /// Whether the projectile has been destroyed (hit or expired).
    pub destroyed: bool,
    /// Whether the projectile should explode on impact (area damage).
    pub explosive: bool,
    /// Explosion radius (if explosive).
    pub explosion_radius: f32,
    /// Explosion damage (if explosive, applied as area damage).
    pub explosion_damage: f32,
    /// Whether to home toward a target.
    pub homing: bool,
    /// Target entity for homing.
    pub homing_target: Option<u32>,
    /// Homing turn rate (radians/sec).
    pub homing_turn_rate: f32,
    /// Distance traveled.
    pub distance_traveled: f32,
    /// Maximum distance before the projectile is destroyed.
    pub max_distance: f32,
}

impl Projectile {
    /// Create a new projectile.
    pub fn new(
        id: u32,
        weapon_id: u32,
        owner_entity: u32,
        position: Vec3,
        velocity: Vec3,
        damage: f32,
        lifetime: f32,
    ) -> Self {
        Self {
            id,
            weapon_id,
            owner_entity,
            position,
            velocity,
            damage,
            lifetime,
            radius: 0.1,
            gravity_scale: 0.0,
            destroyed: false,
            explosive: false,
            explosion_radius: 0.0,
            explosion_damage: 0.0,
            homing: false,
            homing_target: None,
            homing_turn_rate: 0.0,
            distance_traveled: 0.0,
            max_distance: 500.0,
        }
    }

    /// Update the projectile's position.
    pub fn update(&mut self, dt: f32, gravity: f32) {
        if self.destroyed {
            return;
        }

        // Apply gravity.
        self.velocity.y -= gravity * self.gravity_scale * dt;

        // Move.
        let displacement = self.velocity * dt;
        self.distance_traveled += displacement.length();
        self.position += displacement;

        // Lifetime.
        self.lifetime -= dt;
        if self.lifetime <= 0.0 || self.distance_traveled >= self.max_distance {
            self.destroyed = true;
        }
    }

    /// Mark the projectile as destroyed on impact.
    pub fn on_hit(&mut self, hit_point: Vec3) {
        self.position = hit_point;
        self.destroyed = true;
    }

    /// Get the forward direction of this projectile.
    pub fn forward(&self) -> Vec3 {
        self.velocity.normalize_or_zero()
    }

    /// Current speed.
    pub fn speed(&self) -> f32 {
        self.velocity.length()
    }
}

// ---------------------------------------------------------------------------
// Projectile system
// ---------------------------------------------------------------------------

/// System that manages all active projectiles.
pub struct ProjectileSystem {
    /// Active projectiles.
    projectiles: Vec<Projectile>,
    /// Next projectile ID.
    next_id: u32,
    /// Gravity for projectile physics.
    pub gravity: f32,
    /// Maximum number of simultaneous projectiles.
    pub max_projectiles: usize,
    /// Projectiles that were destroyed this frame (for effect spawning).
    destroyed_this_frame: Vec<Projectile>,
}

impl ProjectileSystem {
    /// Create a new projectile system.
    pub fn new() -> Self {
        Self {
            projectiles: Vec::new(),
            next_id: 0,
            gravity: 9.81,
            max_projectiles: 256,
            destroyed_this_frame: Vec::new(),
        }
    }

    /// Spawn a new projectile and return its ID.
    pub fn spawn(
        &mut self,
        weapon_id: u32,
        owner: u32,
        position: Vec3,
        velocity: Vec3,
        damage: f32,
        lifetime: f32,
    ) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        // Enforce max projectiles by removing oldest.
        if self.projectiles.len() >= self.max_projectiles {
            self.projectiles.remove(0);
        }

        self.projectiles.push(Projectile::new(
            id, weapon_id, owner, position, velocity, damage, lifetime,
        ));

        id
    }

    /// Spawn with full configuration.
    pub fn spawn_configured(&mut self, mut projectile: Projectile) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        projectile.id = id;

        if self.projectiles.len() >= self.max_projectiles {
            self.projectiles.remove(0);
        }

        self.projectiles.push(projectile);
        id
    }

    /// Update all projectiles. Returns a list of projectile IDs that were
    /// destroyed this frame.
    pub fn update(&mut self, dt: f32) -> Vec<u32> {
        self.destroyed_this_frame.clear();

        let gravity = self.gravity;
        let mut destroyed_ids = Vec::new();

        for projectile in &mut self.projectiles {
            projectile.update(dt, gravity);
            if projectile.destroyed {
                destroyed_ids.push(projectile.id);
            }
        }

        // Move destroyed projectiles to the destroyed list.
        let mut i = 0;
        while i < self.projectiles.len() {
            if self.projectiles[i].destroyed {
                let p = self.projectiles.swap_remove(i);
                self.destroyed_this_frame.push(p);
            } else {
                i += 1;
            }
        }

        destroyed_ids
    }

    /// Get an iterator over active projectiles.
    pub fn iter(&self) -> impl Iterator<Item = &Projectile> {
        self.projectiles.iter()
    }

    /// Get a mutable iterator over active projectiles.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Projectile> {
        self.projectiles.iter_mut()
    }

    /// Get projectiles destroyed this frame (for spawning effects).
    pub fn destroyed_this_frame(&self) -> &[Projectile] {
        &self.destroyed_this_frame
    }

    /// Get a projectile by ID.
    pub fn get(&self, id: u32) -> Option<&Projectile> {
        self.projectiles.iter().find(|p| p.id == id)
    }

    /// Get a mutable projectile by ID.
    pub fn get_mut(&mut self, id: u32) -> Option<&mut Projectile> {
        self.projectiles.iter_mut().find(|p| p.id == id)
    }

    /// Number of active projectiles.
    pub fn active_count(&self) -> usize {
        self.projectiles.len()
    }

    /// Remove all projectiles.
    pub fn clear(&mut self) {
        self.projectiles.clear();
        self.destroyed_this_frame.clear();
    }
}

impl Default for ProjectileSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Weapon system
// ---------------------------------------------------------------------------

/// Core weapon system that manages weapon state machines, firing logic,
/// reloading, and event generation.
pub struct WeaponSystem {
    /// Pending fire events to be processed by other systems.
    fire_events: Vec<FireEvent>,
    /// Hitscan results from the current frame.
    hitscan_results: Vec<HitscanResult>,
    /// Random seed for spread calculation.
    spread_seed: u32,
}

impl WeaponSystem {
    /// Create a new weapon system.
    pub fn new() -> Self {
        Self {
            fire_events: Vec::new(),
            hitscan_results: Vec::new(),
            spread_seed: 42,
        }
    }

    /// Attempt to fire a weapon. Returns a `FireEvent` if the weapon can fire.
    ///
    /// This function checks:
    /// 1. Is the weapon in a state that allows firing?
    /// 2. Does the weapon have ammo?
    /// 3. Is the fire mode cooldown elapsed?
    ///
    /// If all checks pass, the weapon transitions to the Firing state, ammo
    /// is consumed, recoil is applied, and a `FireEvent` is returned.
    pub fn try_fire(
        &mut self,
        weapon: &mut Weapon,
        owner_entity: u32,
        origin: Vec3,
        forward: Vec3,
    ) -> Option<FireEvent> {
        // Check state.
        if !weapon.state.can_fire() {
            return None;
        }

        // Check ammo.
        if !weapon.has_ammo() {
            // Auto-reload when trying to fire with empty magazine.
            if weapon.can_reload() {
                self.reload(weapon);
            }
            return None;
        }

        // Handle burst mode.
        if let FireMode::Burst(count) = weapon.fire_mode {
            weapon.burst_remaining = count;
        }

        // Fire the weapon.
        self.fire_shot(weapon, owner_entity, origin, forward)
    }

    /// Fire a single shot (internal, handles ammo, spread, recoil).
    fn fire_shot(
        &mut self,
        weapon: &mut Weapon,
        owner_entity: u32,
        origin: Vec3,
        forward: Vec3,
    ) -> Option<FireEvent> {
        // Consume ammo.
        if !weapon.infinite_ammo {
            if weapon.current_ammo == 0 {
                return None;
            }
            weapon.current_ammo -= 1;
        }

        // Apply spread.
        let spread_angle = weapon.total_spread();
        let direction = self.apply_spread(forward, spread_angle);

        // Apply recoil.
        let (_recoil_v, _recoil_h) = weapon.recoil.next_recoil();
        weapon.current_spread =
            (weapon.current_spread + weapon.recoil.spread_per_shot).min(weapon.recoil.max_spread);

        // Transition to firing state.
        weapon.state = WeaponState::Firing;
        weapon.state_timer = weapon.fire_interval() * 0.2; // Short firing state.

        // Create fire event.
        let event = FireEvent {
            weapon_id: weapon.id,
            owner_entity,
            origin,
            direction,
            spread: spread_angle,
            damage: weapon.damage,
            projectile_speed: weapon.projectile_speed,
            weapon_type: weapon.weapon_type,
            range: weapon.range,
            pellet_count: weapon.pellet_count,
            penetration: weapon.penetration,
        };

        self.fire_events.push(event.clone());

        // Handle burst.
        if let FireMode::Burst(_) = weapon.fire_mode {
            weapon.burst_remaining = weapon.burst_remaining.saturating_sub(1);
        }

        Some(event)
    }

    /// Apply spread to a direction vector.
    fn apply_spread(&mut self, forward: Vec3, spread_degrees: f32) -> Vec3 {
        if spread_degrees <= 0.001 {
            return forward;
        }

        let spread_rad = spread_degrees.to_radians();
        let r1 = pseudo_random_f32(self.spread_seed);
        self.spread_seed = self.spread_seed.wrapping_add(1);
        let r2 = pseudo_random_f32(self.spread_seed);
        self.spread_seed = self.spread_seed.wrapping_add(1);

        let theta = r1 * std::f32::consts::TAU;
        let phi = r2 * spread_rad;

        // Generate a random direction within the spread cone.
        let right = if forward.y.abs() < 0.99 {
            forward.cross(Vec3::Y).normalize()
        } else {
            forward.cross(Vec3::X).normalize()
        };
        let up = right.cross(forward).normalize();

        let offset = (right * theta.cos() + up * theta.sin()) * phi.sin();
        (forward * phi.cos() + offset).normalize()
    }

    /// Start reloading a weapon.
    pub fn reload(&mut self, weapon: &mut Weapon) {
        if !weapon.can_reload() {
            return;
        }

        weapon.state = WeaponState::Reloading;
        weapon.state_timer = weapon.reload_time;
        weapon.recoil.reset();
        weapon.current_spread = 0.0;

        log::trace!("Weapon '{}' reloading ({:.1}s)", weapon.name, weapon.reload_time);
    }

    /// Finish reloading: fill the magazine from reserve ammo.
    fn finish_reload(weapon: &mut Weapon) {
        let needed = weapon.ammo_capacity - weapon.current_ammo;
        if weapon.infinite_ammo {
            weapon.current_ammo = weapon.ammo_capacity;
        } else {
            let available = weapon.reserve_ammo.min(needed);
            weapon.current_ammo += available;
            weapon.reserve_ammo -= available;
        }
        weapon.state = WeaponState::Idle;
        log::trace!(
            "Weapon '{}' reload complete: {}/{}",
            weapon.name,
            weapon.current_ammo,
            weapon.ammo_capacity
        );
    }

    /// Start switching to a weapon (equip animation).
    pub fn start_switch(&mut self, weapon: &mut Weapon) {
        weapon.state = WeaponState::Switching;
        weapon.state_timer = weapon.switch_time;
        weapon.recoil.reset();
        weapon.current_spread = 0.0;
    }

    /// Update a weapon's state machine.
    ///
    /// Call this every frame for each weapon. Handles state transitions,
    /// cooldowns, spread recovery, and burst fire.
    pub fn update_weapon(
        &mut self,
        weapon: &mut Weapon,
        owner_entity: u32,
        origin: Vec3,
        forward: Vec3,
        trigger_held: bool,
        dt: f32,
    ) {
        // Tick state timer.
        if weapon.state_timer > 0.0 {
            weapon.state_timer -= dt;
        }

        // Spread recovery.
        if weapon.current_spread > 0.0 {
            weapon.current_spread =
                (weapon.current_spread - weapon.recoil.spread_recovery * dt).max(0.0);
        }

        // State machine transitions.
        match weapon.state {
            WeaponState::Idle => {
                // Check for automatic fire.
                if trigger_held && weapon.fire_mode.is_automatic() {
                    self.try_fire(weapon, owner_entity, origin, forward);
                }
            }
            WeaponState::Firing => {
                if weapon.state_timer <= 0.0 {
                    // Transition to cooldown.
                    weapon.state = WeaponState::Cooldown;
                    weapon.state_timer = weapon.fire_interval() * 0.8;
                }
            }
            WeaponState::Cooldown => {
                if weapon.state_timer <= 0.0 {
                    // Handle burst continuation.
                    if weapon.burst_remaining > 0 && weapon.has_ammo() {
                        self.fire_shot(weapon, owner_entity, origin, forward);
                        return;
                    }

                    weapon.state = WeaponState::Idle;

                    // Auto-fire if trigger is held and weapon is automatic.
                    if trigger_held && weapon.fire_mode.is_automatic() && weapon.has_ammo() {
                        self.try_fire(weapon, owner_entity, origin, forward);
                    }
                }
            }
            WeaponState::Reloading => {
                if weapon.state_timer <= 0.0 {
                    Self::finish_reload(weapon);
                }
            }
            WeaponState::Switching => {
                if weapon.state_timer <= 0.0 {
                    weapon.state = WeaponState::Idle;
                    log::trace!("Weapon '{}' switch complete", weapon.name);
                }
            }
        }

        // Reset recoil pattern when not firing.
        if !trigger_held && weapon.state == WeaponState::Idle {
            weapon.recoil.reset();
        }
    }

    /// Drain fire events from this frame.
    pub fn drain_fire_events(&mut self) -> Vec<FireEvent> {
        std::mem::take(&mut self.fire_events)
    }

    /// Get fire events without draining.
    pub fn fire_events(&self) -> &[FireEvent] {
        &self.fire_events
    }

    /// Drain hitscan results.
    pub fn drain_hitscan_results(&mut self) -> Vec<HitscanResult> {
        std::mem::take(&mut self.hitscan_results)
    }

    /// Process a hitscan fire event: trace a ray and record the hit.
    pub fn process_hitscan(
        &mut self,
        event: &FireEvent,
        hit_point: Vec3,
        hit_normal: Vec3,
        hit_distance: f32,
        hit_entity: Option<u32>,
        headshot: bool,
    ) {
        let damage = if hit_entity.is_some() {
            let base = Weapon::new(0, "", WeaponType::Hitscan).damage; // placeholder
            let _ = base;
            let falloff = if hit_distance <= 30.0 {
                1.0
            } else if hit_distance >= 80.0 {
                0.5
            } else {
                let t = (hit_distance - 30.0) / 50.0;
                1.0 - t * 0.5
            };
            let multiplier = if headshot { 2.0 } else { 1.0 };
            event.damage * falloff * multiplier
        } else {
            0.0
        };

        self.hitscan_results.push(HitscanResult {
            hit: hit_entity.is_some(),
            point: hit_point,
            normal: hit_normal,
            distance: hit_distance,
            hit_entity,
            damage,
            headshot,
            material_id: 0,
        });
    }

    /// Add ammo to a weapon's reserve.
    pub fn add_ammo(weapon: &mut Weapon, amount: u32) {
        weapon.reserve_ammo = (weapon.reserve_ammo + amount).min(weapon.max_reserve_ammo);
    }

    /// Clear all pending events.
    pub fn clear_events(&mut self) {
        self.fire_events.clear();
        self.hitscan_results.clear();
    }
}

impl Default for WeaponSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Weapon inventory
// ---------------------------------------------------------------------------

/// Manages a player's weapon loadout: equipping, switching, and cycling.
pub struct WeaponInventory {
    /// All weapons the player is carrying.
    pub weapons: Vec<Weapon>,
    /// Index of the currently equipped weapon.
    pub active_index: usize,
    /// Maximum number of weapons the player can carry.
    pub max_weapons: usize,
    /// Previous weapon index (for quick-switch).
    pub previous_index: usize,
}

impl WeaponInventory {
    /// Create a new empty inventory.
    pub fn new(max_weapons: usize) -> Self {
        Self {
            weapons: Vec::new(),
            active_index: 0,
            max_weapons,
            previous_index: 0,
        }
    }

    /// Add a weapon to the inventory. Returns true if added.
    pub fn add_weapon(&mut self, weapon: Weapon) -> bool {
        if self.weapons.len() >= self.max_weapons {
            return false;
        }
        self.weapons.push(weapon);
        true
    }

    /// Remove a weapon by index. Returns the removed weapon.
    pub fn remove_weapon(&mut self, index: usize) -> Option<Weapon> {
        if index >= self.weapons.len() {
            return None;
        }
        let weapon = self.weapons.remove(index);
        if self.active_index >= self.weapons.len() && !self.weapons.is_empty() {
            self.active_index = self.weapons.len() - 1;
        }
        Some(weapon)
    }

    /// Get the currently active weapon.
    pub fn active_weapon(&self) -> Option<&Weapon> {
        self.weapons.get(self.active_index)
    }

    /// Get the currently active weapon mutably.
    pub fn active_weapon_mut(&mut self) -> Option<&mut Weapon> {
        self.weapons.get_mut(self.active_index)
    }

    /// Switch to a weapon by index.
    pub fn switch_to(&mut self, index: usize) -> bool {
        if index >= self.weapons.len() || index == self.active_index {
            return false;
        }
        if let Some(weapon) = self.weapons.get(self.active_index) {
            if !weapon.state.can_switch() {
                return false;
            }
        }
        self.previous_index = self.active_index;
        self.active_index = index;
        true
    }

    /// Cycle to the next weapon.
    pub fn next_weapon(&mut self) -> bool {
        if self.weapons.len() <= 1 {
            return false;
        }
        let next = (self.active_index + 1) % self.weapons.len();
        self.switch_to(next)
    }

    /// Cycle to the previous weapon.
    pub fn prev_weapon(&mut self) -> bool {
        if self.weapons.len() <= 1 {
            return false;
        }
        let prev = if self.active_index == 0 {
            self.weapons.len() - 1
        } else {
            self.active_index - 1
        };
        self.switch_to(prev)
    }

    /// Quick-switch to the last used weapon.
    pub fn quick_switch(&mut self) -> bool {
        self.switch_to(self.previous_index)
    }

    /// Number of weapons in the inventory.
    pub fn weapon_count(&self) -> usize {
        self.weapons.len()
    }

    /// Whether the inventory is full.
    pub fn is_full(&self) -> bool {
        self.weapons.len() >= self.max_weapons
    }
}

impl Default for WeaponInventory {
    fn default() -> Self {
        Self::new(4)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weapon_fire_interval() {
        let weapon = Weapon::assault_rifle(0);
        let interval = weapon.fire_interval();
        assert!((interval - 0.1).abs() < 0.01, "600 RPM = 0.1s interval");
    }

    #[test]
    fn weapon_damage_at_distance() {
        let weapon = Weapon::assault_rifle(0);
        let full = weapon.damage_at_distance(10.0);
        let falloff = weapon.damage_at_distance(55.0);
        let min = weapon.damage_at_distance(100.0);

        assert_eq!(full, weapon.damage);
        assert!(falloff < weapon.damage);
        assert!(min <= weapon.damage * weapon.falloff_min_multiplier + 0.1);
    }

    #[test]
    fn weapon_state_machine_transitions() {
        assert!(WeaponState::Idle.can_fire());
        assert!(!WeaponState::Reloading.can_fire());
        assert!(!WeaponState::Switching.can_fire());
        assert!(WeaponState::Idle.can_reload());
        assert!(!WeaponState::Firing.can_reload());
    }

    #[test]
    fn try_fire_consumes_ammo() {
        let mut system = WeaponSystem::new();
        let mut weapon = Weapon::assault_rifle(0);
        weapon.current_ammo = 5;

        let event = system.try_fire(&mut weapon, 1, Vec3::ZERO, Vec3::Z);
        assert!(event.is_some());
        assert_eq!(weapon.current_ammo, 4);
    }

    #[test]
    fn try_fire_empty_magazine() {
        let mut system = WeaponSystem::new();
        let mut weapon = Weapon::assault_rifle(0);
        weapon.current_ammo = 0;

        let event = system.try_fire(&mut weapon, 1, Vec3::ZERO, Vec3::Z);
        assert!(event.is_none());
        // Should auto-reload.
        assert_eq!(weapon.state, WeaponState::Reloading);
    }

    #[test]
    fn reload_fills_magazine() {
        let mut weapon = Weapon::assault_rifle(0);
        weapon.current_ammo = 10;
        weapon.reserve_ammo = 50;

        WeaponSystem::finish_reload(&mut weapon);
        assert_eq!(weapon.current_ammo, weapon.ammo_capacity);
        assert_eq!(weapon.reserve_ammo, 50 - (weapon.ammo_capacity - 10));
    }

    #[test]
    fn reload_partial_reserve() {
        let mut weapon = Weapon::assault_rifle(0);
        weapon.current_ammo = 25;
        weapon.reserve_ammo = 2;

        WeaponSystem::finish_reload(&mut weapon);
        assert_eq!(weapon.current_ammo, 27);
        assert_eq!(weapon.reserve_ammo, 0);
    }

    #[test]
    fn melee_weapon_infinite_ammo() {
        let weapon = Weapon::melee_weapon(0, "Knife");
        assert!(weapon.infinite_ammo);
        assert!(weapon.has_ammo());
    }

    #[test]
    fn shotgun_pellets() {
        let weapon = Weapon::shotgun(0);
        assert!(weapon.pellet_count > 1);
    }

    #[test]
    fn projectile_update() {
        let mut proj = Projectile::new(0, 0, 1, Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), 50.0, 5.0);
        proj.update(1.0 / 60.0, 9.81);

        assert!(proj.position.x > 0.0);
        assert!(!proj.destroyed);
    }

    #[test]
    fn projectile_lifetime_expiry() {
        let mut proj = Projectile::new(0, 0, 1, Vec3::ZERO, Vec3::X, 50.0, 0.01);
        proj.update(0.1, 0.0);
        assert!(proj.destroyed);
    }

    #[test]
    fn projectile_system_spawn_and_update() {
        let mut system = ProjectileSystem::new();
        let id = system.spawn(0, 1, Vec3::ZERO, Vec3::new(50.0, 0.0, 0.0), 100.0, 5.0);

        assert_eq!(system.active_count(), 1);

        let destroyed = system.update(1.0 / 60.0);
        assert!(destroyed.is_empty());
        assert_eq!(system.active_count(), 1);
    }

    #[test]
    fn weapon_inventory_add_switch() {
        let mut inv = WeaponInventory::new(3);
        inv.add_weapon(Weapon::assault_rifle(0));
        inv.add_weapon(Weapon::shotgun(1));

        assert_eq!(inv.active_index, 0);
        assert!(inv.switch_to(1));
        assert_eq!(inv.active_index, 1);
    }

    #[test]
    fn weapon_inventory_cycle() {
        let mut inv = WeaponInventory::new(3);
        inv.add_weapon(Weapon::pistol(0));
        inv.add_weapon(Weapon::assault_rifle(1));
        inv.add_weapon(Weapon::shotgun(2));

        assert_eq!(inv.active_index, 0);
        inv.next_weapon();
        assert_eq!(inv.active_index, 1);
        inv.next_weapon();
        assert_eq!(inv.active_index, 2);
    }

    #[test]
    fn weapon_inventory_quick_switch() {
        let mut inv = WeaponInventory::new(3);
        inv.add_weapon(Weapon::pistol(0));
        inv.add_weapon(Weapon::assault_rifle(1));

        inv.switch_to(1);
        inv.quick_switch();
        assert_eq!(inv.active_index, 0);
    }

    #[test]
    fn weapon_inventory_full() {
        let mut inv = WeaponInventory::new(2);
        assert!(inv.add_weapon(Weapon::pistol(0)));
        assert!(inv.add_weapon(Weapon::assault_rifle(1)));
        assert!(!inv.add_weapon(Weapon::shotgun(2)));
        assert!(inv.is_full());
    }

    #[test]
    fn fire_mode_properties() {
        assert_eq!(FireMode::Single.shots_per_activation(), 1);
        assert_eq!(FireMode::Burst(3).shots_per_activation(), 3);
        assert!(FireMode::Automatic.is_automatic());
        assert!(!FireMode::Single.is_automatic());
    }

    #[test]
    fn recoil_pattern_reset() {
        let mut recoil = RecoilPattern::default();
        let _ = recoil.next_recoil();
        let _ = recoil.next_recoil();
        assert!(recoil.pattern_index > 0);
        recoil.reset();
        assert_eq!(recoil.pattern_index, 0);
    }

    #[test]
    fn weapon_update_state_machine() {
        let mut system = WeaponSystem::new();
        let mut weapon = Weapon::assault_rifle(0);
        weapon.state = WeaponState::Firing;
        weapon.state_timer = 0.01;

        // Update should transition to Cooldown.
        system.update_weapon(&mut weapon, 1, Vec3::ZERO, Vec3::Z, false, 0.02);
        assert_eq!(weapon.state, WeaponState::Cooldown);
    }

    #[test]
    fn weapon_presets() {
        let ar = Weapon::assault_rifle(0);
        let sg = Weapon::shotgun(1);
        let sniper = Weapon::sniper_rifle(2);
        let rl = Weapon::rocket_launcher(3);
        let knife = Weapon::melee_weapon(4, "Knife");
        let burst = Weapon::burst_rifle(5);

        assert_eq!(ar.weapon_type, WeaponType::Hitscan);
        assert!(sg.pellet_count > 1);
        assert!(sniper.range > 200.0);
        assert_eq!(rl.weapon_type, WeaponType::Projectile);
        assert_eq!(knife.weapon_type, WeaponType::Melee);
        assert!(matches!(burst.fire_mode, FireMode::Burst(3)));
    }

    #[test]
    fn add_ammo() {
        let mut weapon = Weapon::pistol(0);
        weapon.reserve_ammo = 10;
        WeaponSystem::add_ammo(&mut weapon, 20);
        assert_eq!(weapon.reserve_ammo, 30);
    }

    #[test]
    fn add_ammo_cap() {
        let mut weapon = Weapon::pistol(0);
        weapon.reserve_ammo = 90;
        weapon.max_reserve_ammo = 96;
        WeaponSystem::add_ammo(&mut weapon, 20);
        assert_eq!(weapon.reserve_ammo, 96);
    }
}
