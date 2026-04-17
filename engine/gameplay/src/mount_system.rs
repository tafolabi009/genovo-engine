//! Mount and riding system for player-controlled mounts.
//!
//! Provides:
//! - **Mount entity**: defines a rideable creature or vehicle with stats
//! - **Rider entity**: tracks who is riding and where
//! - **Mount points**: saddle positions with configurable attach offsets
//! - **Mount animations**: idle, walk, run, gallop with speed-based blending
//! - **Mount combat**: limited combat actions while mounted
//! - **Mount inventory**: shared or separate storage
//! - **Mount stamina**: sprint/gallop costs stamina
//! - **Mount stats**: speed, carry weight, health
//! - **Taming/befriending**: progressions for taming wild mounts
//! - **ECS integration**: `MountComponent`, `RiderComponent`, `MountSystem`
//!
//! # Design
//!
//! A [`Mount`] is an entity with speed, health, stamina, and animation state.
//! A [`Rider`] attaches to a mount at a [`MountPoint`]. The [`MountSystem`]
//! handles mounting/dismounting, movement input translation, stamina costs,
//! and animation state transitions.

use glam::{Vec3, Quat};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of mounts in the world.
pub const MAX_MOUNTS: usize = 256;
/// Maximum number of riders per mount.
pub const MAX_RIDERS_PER_MOUNT: usize = 4;
/// Default mount walk speed (m/s).
pub const DEFAULT_WALK_SPEED: f32 = 3.0;
/// Default mount run speed.
pub const DEFAULT_RUN_SPEED: f32 = 8.0;
/// Default mount gallop speed.
pub const DEFAULT_GALLOP_SPEED: f32 = 15.0;
/// Default mount health.
pub const DEFAULT_MOUNT_HEALTH: f32 = 100.0;
/// Default mount stamina.
pub const DEFAULT_MOUNT_STAMINA: f32 = 100.0;
/// Stamina cost per second of galloping.
pub const GALLOP_STAMINA_COST: f32 = 10.0;
/// Stamina cost per second of running.
pub const RUN_STAMINA_COST: f32 = 3.0;
/// Stamina regeneration rate per second (when not sprinting).
pub const STAMINA_REGEN_RATE: f32 = 5.0;
/// Default carry weight capacity.
pub const DEFAULT_CARRY_WEIGHT: f32 = 200.0;
/// Default mount interaction range.
pub const MOUNT_INTERACTION_RANGE: f32 = 3.0;
/// Taming progress per feed/pet action.
pub const TAMING_PROGRESS_PER_ACTION: f32 = 10.0;
/// Default dismount distance.
pub const DISMOUNT_OFFSET: f32 = 2.0;
/// Mount acceleration rate.
pub const MOUNT_ACCELERATION: f32 = 5.0;
/// Mount deceleration rate.
pub const MOUNT_DECELERATION: f32 = 8.0;
/// Minimum speed to use walk animation.
pub const WALK_ANIM_THRESHOLD: f32 = 0.5;
/// Minimum speed to use run animation.
pub const RUN_ANIM_THRESHOLD: f32 = 5.0;
/// Minimum speed to use gallop animation.
pub const GALLOP_ANIM_THRESHOLD: f32 = 10.0;
/// Small epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// MountId, RiderId
// ---------------------------------------------------------------------------

/// Unique mount identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MountId(pub u64);

/// Unique rider identifier (typically the entity ID of the rider).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RiderId(pub u64);

// ---------------------------------------------------------------------------
// MountType
// ---------------------------------------------------------------------------

/// Type of mount.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MountType {
    /// Horse-like mount.
    Horse,
    /// Wolf-like mount.
    Wolf,
    /// Bird (flying mount).
    Bird,
    /// Dragon (flying + combat).
    Dragon,
    /// Elephant/large beast.
    LargeBeast,
    /// Mechanical vehicle.
    Mechanical,
    /// Boat/water mount.
    Boat,
    /// Custom mount type.
    Custom(u32),
}

impl MountType {
    /// Whether this mount can fly.
    pub fn can_fly(&self) -> bool {
        matches!(self, MountType::Bird | MountType::Dragon)
    }

    /// Whether this mount can swim.
    pub fn can_swim(&self) -> bool {
        matches!(self, MountType::Boat | MountType::Dragon)
    }

    /// Display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Horse => "Horse",
            Self::Wolf => "Wolf",
            Self::Bird => "Bird",
            Self::Dragon => "Dragon",
            Self::LargeBeast => "Large Beast",
            Self::Mechanical => "Mechanical",
            Self::Boat => "Boat",
            Self::Custom(_) => "Custom",
        }
    }
}

// ---------------------------------------------------------------------------
// MountAnimState
// ---------------------------------------------------------------------------

/// Current animation state of a mount.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MountAnimState {
    /// Standing still.
    Idle,
    /// Walking.
    Walk,
    /// Running.
    Run,
    /// Full gallop.
    Gallop,
    /// Jumping.
    Jump,
    /// In the air (flying mount).
    Flying,
    /// Swimming.
    Swimming,
    /// Rearing up.
    Rear,
    /// Taking damage.
    Hit,
    /// Dying.
    Death,
    /// Eating/grazing.
    Eating,
    /// Being tamed.
    Taming,
}

impl MountAnimState {
    /// Get the animation clip name for this state.
    pub fn clip_name(&self) -> &'static str {
        match self {
            Self::Idle => "mount_idle",
            Self::Walk => "mount_walk",
            Self::Run => "mount_run",
            Self::Gallop => "mount_gallop",
            Self::Jump => "mount_jump",
            Self::Flying => "mount_fly",
            Self::Swimming => "mount_swim",
            Self::Rear => "mount_rear",
            Self::Hit => "mount_hit",
            Self::Death => "mount_death",
            Self::Eating => "mount_eat",
            Self::Taming => "mount_taming",
        }
    }
}

// ---------------------------------------------------------------------------
// MountPoint
// ---------------------------------------------------------------------------

/// A point on the mount where a rider can sit.
#[derive(Debug, Clone)]
pub struct MountPoint {
    /// Index of this mount point (0 = primary/saddle).
    pub index: usize,
    /// Local offset from mount origin.
    pub local_offset: Vec3,
    /// Local rotation offset.
    pub local_rotation: Quat,
    /// Whether this point is occupied.
    pub occupied: bool,
    /// Rider at this point (if occupied).
    pub rider_id: Option<RiderId>,
    /// Whether the rider can attack from this position.
    pub allows_combat: bool,
    /// Animation override for the rider at this point.
    pub rider_animation: String,
    /// Label (e.g., "saddle", "passenger_rear", "sidecar").
    pub label: String,
}

impl MountPoint {
    /// Create a new mount point.
    pub fn new(index: usize, local_offset: Vec3) -> Self {
        Self {
            index,
            local_offset,
            local_rotation: Quat::IDENTITY,
            occupied: false,
            rider_id: None,
            allows_combat: index == 0,
            rider_animation: "mounted_idle".to_string(),
            label: if index == 0 { "saddle".to_string() } else { format!("passenger_{}", index) },
        }
    }

    /// Assign a rider.
    pub fn assign_rider(&mut self, rider: RiderId) {
        self.occupied = true;
        self.rider_id = Some(rider);
    }

    /// Remove the rider.
    pub fn clear_rider(&mut self) {
        self.occupied = false;
        self.rider_id = None;
    }
}

// ---------------------------------------------------------------------------
// MountStats
// ---------------------------------------------------------------------------

/// Statistics and attributes for a mount.
#[derive(Debug, Clone)]
pub struct MountStats {
    /// Walk speed (m/s).
    pub walk_speed: f32,
    /// Run speed.
    pub run_speed: f32,
    /// Gallop speed.
    pub gallop_speed: f32,
    /// Flight speed (if applicable).
    pub fly_speed: f32,
    /// Swim speed.
    pub swim_speed: f32,
    /// Jump force.
    pub jump_force: f32,
    /// Maximum health.
    pub max_health: f32,
    /// Current health.
    pub health: f32,
    /// Maximum stamina.
    pub max_stamina: f32,
    /// Current stamina.
    pub stamina: f32,
    /// Carry weight capacity.
    pub carry_weight: f32,
    /// Current weight being carried.
    pub current_weight: f32,
    /// Turn rate (degrees per second).
    pub turn_rate: f32,
    /// Acceleration.
    pub acceleration: f32,
    /// Armor/defense rating.
    pub defense: f32,
    /// Attack damage (for combat mounts).
    pub attack_damage: f32,
}

impl Default for MountStats {
    fn default() -> Self {
        Self {
            walk_speed: DEFAULT_WALK_SPEED,
            run_speed: DEFAULT_RUN_SPEED,
            gallop_speed: DEFAULT_GALLOP_SPEED,
            fly_speed: 20.0,
            swim_speed: 5.0,
            jump_force: 8.0,
            max_health: DEFAULT_MOUNT_HEALTH,
            health: DEFAULT_MOUNT_HEALTH,
            max_stamina: DEFAULT_MOUNT_STAMINA,
            stamina: DEFAULT_MOUNT_STAMINA,
            carry_weight: DEFAULT_CARRY_WEIGHT,
            current_weight: 0.0,
            turn_rate: 120.0,
            acceleration: MOUNT_ACCELERATION,
            defense: 5.0,
            attack_damage: 10.0,
        }
    }
}

impl MountStats {
    /// Get the maximum speed at the current weight.
    pub fn effective_gallop_speed(&self) -> f32 {
        let weight_ratio = (self.current_weight / self.carry_weight).min(1.0);
        self.gallop_speed * (1.0 - weight_ratio * 0.3)
    }

    /// Check if the mount is overburdened.
    pub fn is_overburdened(&self) -> bool {
        self.current_weight > self.carry_weight
    }

    /// Check if the mount is alive.
    pub fn is_alive(&self) -> bool {
        self.health > 0.0
    }

    /// Apply damage.
    pub fn take_damage(&mut self, amount: f32) -> f32 {
        let actual = (amount - self.defense).max(1.0);
        self.health = (self.health - actual).max(0.0);
        actual
    }

    /// Heal the mount.
    pub fn heal(&mut self, amount: f32) {
        self.health = (self.health + amount).min(self.max_health);
    }

    /// Consume stamina. Returns false if not enough.
    pub fn consume_stamina(&mut self, amount: f32) -> bool {
        if self.stamina >= amount {
            self.stamina -= amount;
            true
        } else {
            false
        }
    }

    /// Regenerate stamina.
    pub fn regen_stamina(&mut self, dt: f32) {
        self.stamina = (self.stamina + STAMINA_REGEN_RATE * dt).min(self.max_stamina);
    }
}

// ---------------------------------------------------------------------------
// TamingState
// ---------------------------------------------------------------------------

/// Taming progress for a wild mount.
#[derive(Debug, Clone)]
pub struct TamingState {
    /// Whether this mount has been tamed.
    pub tamed: bool,
    /// Taming progress (0..100).
    pub progress: f32,
    /// Required progress to tame.
    pub required_progress: f32,
    /// Preferred food items for taming.
    pub preferred_foods: Vec<String>,
    /// Temperament (affects difficulty).
    pub temperament: MountTemperament,
    /// Number of failed taming attempts.
    pub failed_attempts: u32,
    /// Cooldown before next taming attempt.
    pub cooldown: f32,
    /// Current cooldown remaining.
    pub cooldown_remaining: f32,
    /// Trust level after taming (affects behavior).
    pub trust_level: f32,
}

/// Mount temperament affecting taming difficulty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MountTemperament {
    /// Easy to tame.
    Docile,
    /// Normal difficulty.
    Neutral,
    /// Hard to tame.
    Skittish,
    /// Very hard to tame.
    Aggressive,
    /// Cannot be tamed normally (requires special items).
    Wild,
}

impl MountTemperament {
    /// Get the taming difficulty multiplier.
    pub fn difficulty_multiplier(&self) -> f32 {
        match self {
            Self::Docile => 0.5,
            Self::Neutral => 1.0,
            Self::Skittish => 1.5,
            Self::Aggressive => 2.0,
            Self::Wild => 5.0,
        }
    }
}

impl Default for TamingState {
    fn default() -> Self {
        Self {
            tamed: false,
            progress: 0.0,
            required_progress: 100.0,
            preferred_foods: Vec::new(),
            temperament: MountTemperament::Neutral,
            failed_attempts: 0,
            cooldown: 30.0,
            cooldown_remaining: 0.0,
            trust_level: 0.0,
        }
    }
}

impl TamingState {
    /// Attempt a taming action (feed, pet, etc.).
    pub fn attempt_taming(&mut self, action: TamingAction) -> TamingResult {
        if self.tamed {
            return TamingResult::AlreadyTamed;
        }
        if self.cooldown_remaining > 0.0 {
            return TamingResult::OnCooldown;
        }

        let progress_gained = match action {
            TamingAction::Feed { food_id } => {
                let is_preferred = self.preferred_foods.iter().any(|f| f == &food_id);
                if is_preferred {
                    TAMING_PROGRESS_PER_ACTION * 2.0
                } else {
                    TAMING_PROGRESS_PER_ACTION
                }
            }
            TamingAction::Pet => TAMING_PROGRESS_PER_ACTION * 0.5,
            TamingAction::Approach => TAMING_PROGRESS_PER_ACTION * 0.2,
            TamingAction::UseSpecialItem => TAMING_PROGRESS_PER_ACTION * 5.0,
        };

        let adjusted = progress_gained / self.temperament.difficulty_multiplier();
        self.progress += adjusted;

        if self.progress >= self.required_progress {
            self.tamed = true;
            self.trust_level = 0.5;
            TamingResult::Success
        } else {
            self.cooldown_remaining = self.cooldown;
            TamingResult::Progress {
                current: self.progress,
                required: self.required_progress,
            }
        }
    }

    /// Update cooldown.
    pub fn update(&mut self, dt: f32) {
        if self.cooldown_remaining > 0.0 {
            self.cooldown_remaining = (self.cooldown_remaining - dt).max(0.0);
        }
    }
}

/// Taming action types.
#[derive(Debug, Clone)]
pub enum TamingAction {
    /// Feed the mount.
    Feed { food_id: String },
    /// Pet/stroke the mount.
    Pet,
    /// Approach slowly.
    Approach,
    /// Use a special taming item.
    UseSpecialItem,
}

/// Result of a taming attempt.
#[derive(Debug, Clone)]
pub enum TamingResult {
    /// Successfully tamed!
    Success,
    /// Made progress but not yet tamed.
    Progress { current: f32, required: f32 },
    /// Already tamed.
    AlreadyTamed,
    /// On cooldown.
    OnCooldown,
    /// Taming failed (mount fled or attacked).
    Failed,
}

// ---------------------------------------------------------------------------
// Mount
// ---------------------------------------------------------------------------

/// A rideable mount entity.
#[derive(Debug, Clone)]
pub struct Mount {
    /// Unique mount ID.
    pub id: MountId,
    /// Display name.
    pub name: String,
    /// Mount type.
    pub mount_type: MountType,
    /// Stats.
    pub stats: MountStats,
    /// Mount points (where riders can sit).
    pub mount_points: Vec<MountPoint>,
    /// Current animation state.
    pub anim_state: MountAnimState,
    /// Current position.
    pub position: Vec3,
    /// Current rotation.
    pub rotation: Quat,
    /// Current velocity.
    pub velocity: Vec3,
    /// Current speed (scalar).
    pub current_speed: f32,
    /// Target speed (based on input).
    pub target_speed: f32,
    /// Taming state.
    pub taming: TamingState,
    /// Whether the mount is currently being ridden.
    pub is_mounted: bool,
    /// Whether the mount has an inventory.
    pub has_inventory: bool,
    /// Inventory capacity (slots).
    pub inventory_slots: usize,
    /// Owner's rider ID (after taming).
    pub owner: Option<RiderId>,
    /// Tag for game code.
    pub tag: Option<String>,
    /// Whether combat is allowed from this mount.
    pub allows_combat: bool,
    /// Active buffs/debuffs on the mount.
    pub status_effects: Vec<MountStatusEffect>,
}

/// A status effect on a mount.
#[derive(Debug, Clone)]
pub struct MountStatusEffect {
    /// Effect name.
    pub name: String,
    /// Duration remaining.
    pub duration: f32,
    /// Speed modifier (multiplier).
    pub speed_modifier: f32,
    /// Stamina cost modifier.
    pub stamina_modifier: f32,
}

impl Mount {
    /// Create a new mount.
    pub fn new(id: MountId, name: impl Into<String>, mount_type: MountType) -> Self {
        let mut mount_points = vec![
            MountPoint::new(0, Vec3::new(0.0, 1.5, 0.0)),
        ];
        if matches!(mount_type, MountType::LargeBeast | MountType::Dragon) {
            mount_points.push(MountPoint::new(1, Vec3::new(0.0, 1.5, -1.0)));
        }

        Self {
            id,
            name: name.into(),
            mount_type,
            stats: MountStats::default(),
            mount_points,
            anim_state: MountAnimState::Idle,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            current_speed: 0.0,
            target_speed: 0.0,
            taming: TamingState::default(),
            is_mounted: false,
            has_inventory: true,
            inventory_slots: 8,
            owner: None,
            tag: None,
            allows_combat: true,
            status_effects: Vec::new(),
        }
    }

    /// Check if a rider can mount at any available point.
    pub fn can_mount(&self) -> bool {
        self.taming.tamed
            && self.stats.is_alive()
            && self.mount_points.iter().any(|p| !p.occupied)
    }

    /// Mount a rider at the first available point.
    pub fn mount_rider(&mut self, rider: RiderId) -> Option<usize> {
        if !self.can_mount() {
            return None;
        }
        for point in &mut self.mount_points {
            if !point.occupied {
                point.assign_rider(rider);
                self.is_mounted = true;
                return Some(point.index);
            }
        }
        None
    }

    /// Dismount a rider.
    pub fn dismount_rider(&mut self, rider: RiderId) -> bool {
        for point in &mut self.mount_points {
            if point.rider_id == Some(rider) {
                point.clear_rider();
                self.is_mounted = self.mount_points.iter().any(|p| p.occupied);
                return true;
            }
        }
        false
    }

    /// Get the primary rider (at mount point 0).
    pub fn primary_rider(&self) -> Option<RiderId> {
        self.mount_points.first().and_then(|p| p.rider_id)
    }

    /// Get all current riders.
    pub fn all_riders(&self) -> Vec<RiderId> {
        self.mount_points.iter()
            .filter_map(|p| p.rider_id)
            .collect()
    }

    /// Update mount state.
    pub fn update(&mut self, dt: f32) {
        // Update taming
        self.taming.update(dt);

        // Update status effects
        self.status_effects.retain_mut(|effect| {
            effect.duration -= dt;
            effect.duration > 0.0
        });

        // Update speed
        let speed_diff = self.target_speed - self.current_speed;
        if speed_diff.abs() > EPSILON {
            if speed_diff > 0.0 {
                self.current_speed = (self.current_speed + self.stats.acceleration * dt)
                    .min(self.target_speed);
            } else {
                self.current_speed = (self.current_speed - MOUNT_DECELERATION * dt)
                    .max(self.target_speed);
            }
        }

        // Apply speed modifiers from status effects
        let mut speed_mod = 1.0;
        for effect in &self.status_effects {
            speed_mod *= effect.speed_modifier;
        }
        let effective_speed = self.current_speed * speed_mod;

        // Stamina consumption
        if self.is_mounted {
            let stamina_cost = if effective_speed >= GALLOP_ANIM_THRESHOLD {
                GALLOP_STAMINA_COST * dt
            } else if effective_speed >= RUN_ANIM_THRESHOLD {
                RUN_STAMINA_COST * dt
            } else {
                0.0
            };

            let mut stamina_mod = 1.0;
            for effect in &self.status_effects {
                stamina_mod *= effect.stamina_modifier;
            }

            if stamina_cost > 0.0 {
                if !self.stats.consume_stamina(stamina_cost * stamina_mod) {
                    // Out of stamina, slow down
                    self.target_speed = self.stats.run_speed;
                }
            } else {
                self.stats.regen_stamina(dt);
            }
        } else {
            self.stats.regen_stamina(dt);
        }

        // Update velocity
        let forward = self.rotation * Vec3::new(0.0, 0.0, 1.0);
        self.velocity = forward * effective_speed;
        self.position += self.velocity * dt;

        // Update animation state
        self.update_animation(effective_speed);
    }

    /// Update the animation state based on speed.
    fn update_animation(&mut self, speed: f32) {
        if !self.stats.is_alive() {
            self.anim_state = MountAnimState::Death;
            return;
        }

        if speed < WALK_ANIM_THRESHOLD {
            self.anim_state = MountAnimState::Idle;
        } else if speed < RUN_ANIM_THRESHOLD {
            self.anim_state = MountAnimState::Walk;
        } else if speed < GALLOP_ANIM_THRESHOLD {
            self.anim_state = MountAnimState::Run;
        } else {
            self.anim_state = MountAnimState::Gallop;
        }
    }

    /// Set target speed based on input.
    pub fn set_input(&mut self, walk: bool, run: bool, gallop: bool) {
        if !self.is_mounted || !self.stats.is_alive() {
            self.target_speed = 0.0;
            return;
        }

        if gallop && self.stats.stamina > GALLOP_STAMINA_COST {
            self.target_speed = self.stats.effective_gallop_speed();
        } else if run {
            self.target_speed = self.stats.run_speed;
        } else if walk {
            self.target_speed = self.stats.walk_speed;
        } else {
            self.target_speed = 0.0;
        }
    }

    /// Turn the mount.
    pub fn turn(&mut self, angle_degrees: f32, dt: f32) {
        let max_turn = self.stats.turn_rate * dt;
        let clamped = angle_degrees.clamp(-max_turn, max_turn);
        let turn_radians = clamped.to_radians();
        self.rotation = self.rotation * Quat::from_rotation_y(turn_radians);
    }
}

// ---------------------------------------------------------------------------
// MountInput
// ---------------------------------------------------------------------------

/// Input for controlling a mount.
#[derive(Debug, Clone)]
pub struct MountInput {
    /// Forward/backward (-1..1).
    pub forward: f32,
    /// Turn left/right (-1..1).
    pub turn: f32,
    /// Sprint/gallop button.
    pub sprint: bool,
    /// Jump button.
    pub jump: bool,
    /// Dismount button.
    pub dismount: bool,
    /// Attack button (if mount combat).
    pub attack: bool,
}

impl Default for MountInput {
    fn default() -> Self {
        Self {
            forward: 0.0,
            turn: 0.0,
            sprint: false,
            jump: false,
            dismount: false,
            attack: false,
        }
    }
}

// ---------------------------------------------------------------------------
// MountComponent (ECS)
// ---------------------------------------------------------------------------

/// ECS component for mount entities.
#[derive(Debug, Clone)]
pub struct MountComponent {
    /// Mount data.
    pub mount: Mount,
    /// Whether to show the mount name above the entity.
    pub show_name: bool,
    /// Interaction prompt text.
    pub interaction_prompt: String,
}

impl MountComponent {
    /// Create a new mount component.
    pub fn new(mount: Mount) -> Self {
        Self {
            mount,
            show_name: true,
            interaction_prompt: "Press E to mount".to_string(),
        }
    }
}

/// ECS component for entities that can ride mounts.
#[derive(Debug, Clone)]
pub struct RiderComponent {
    /// Rider ID.
    pub rider_id: RiderId,
    /// Currently mounted mount ID.
    pub current_mount: Option<MountId>,
    /// Mount point index.
    pub mount_point_index: Option<usize>,
    /// Whether the rider is in control (primary rider).
    pub is_controller: bool,
    /// Rider position offset from mount point.
    pub position_offset: Vec3,
}

impl RiderComponent {
    /// Create a new rider component.
    pub fn new(rider_id: RiderId) -> Self {
        Self {
            rider_id,
            current_mount: None,
            mount_point_index: None,
            is_controller: false,
            position_offset: Vec3::ZERO,
        }
    }

    /// Check if this rider is currently mounted.
    pub fn is_mounted(&self) -> bool {
        self.current_mount.is_some()
    }
}

// ---------------------------------------------------------------------------
// MountSystem
// ---------------------------------------------------------------------------

/// System managing all mounts and rider interactions.
pub struct MountSystem {
    /// All mounts indexed by ID.
    mounts: HashMap<MountId, Mount>,
    /// Next mount ID.
    next_id: u64,
    /// Events from last update.
    events: Vec<MountEvent>,
}

/// Events emitted by the mount system.
#[derive(Debug, Clone)]
pub enum MountEvent {
    /// A rider mounted.
    Mounted { rider: RiderId, mount: MountId, point: usize },
    /// A rider dismounted.
    Dismounted { rider: RiderId, mount: MountId },
    /// A mount was tamed.
    Tamed { mount: MountId, owner: RiderId },
    /// A mount took damage.
    MountDamaged { mount: MountId, damage: f32 },
    /// A mount died.
    MountDied { mount: MountId },
    /// Mount stamina depleted.
    StaminaDepleted { mount: MountId },
    /// Taming progress.
    TamingProgress { mount: MountId, progress: f32, required: f32 },
}

impl MountSystem {
    /// Create a new mount system.
    pub fn new() -> Self {
        Self {
            mounts: HashMap::new(),
            next_id: 1,
            events: Vec::new(),
        }
    }

    /// Register a mount.
    pub fn register_mount(&mut self, mut mount: Mount) -> MountId {
        let id = MountId(self.next_id);
        self.next_id += 1;
        mount.id = id;
        self.mounts.insert(id, mount);
        id
    }

    /// Get a mount by ID.
    pub fn get_mount(&self, id: MountId) -> Option<&Mount> {
        self.mounts.get(&id)
    }

    /// Get a mount mutably.
    pub fn get_mount_mut(&mut self, id: MountId) -> Option<&mut Mount> {
        self.mounts.get_mut(&id)
    }

    /// Attempt to mount.
    pub fn try_mount(&mut self, rider: RiderId, mount_id: MountId) -> bool {
        if let Some(mount) = self.mounts.get_mut(&mount_id) {
            if let Some(point) = mount.mount_rider(rider) {
                self.events.push(MountEvent::Mounted {
                    rider,
                    mount: mount_id,
                    point,
                });
                return true;
            }
        }
        false
    }

    /// Dismount a rider.
    pub fn dismount(&mut self, rider: RiderId, mount_id: MountId) -> bool {
        if let Some(mount) = self.mounts.get_mut(&mount_id) {
            if mount.dismount_rider(rider) {
                self.events.push(MountEvent::Dismounted {
                    rider,
                    mount: mount_id,
                });
                return true;
            }
        }
        false
    }

    /// Update all mounts.
    pub fn update(&mut self, dt: f32) {
        self.events.clear();
        let ids: Vec<MountId> = self.mounts.keys().copied().collect();
        for id in ids {
            if let Some(mount) = self.mounts.get_mut(&id) {
                mount.update(dt);
            }
        }
    }

    /// Process mount input for a specific mount.
    pub fn process_input(&mut self, mount_id: MountId, input: &MountInput, dt: f32) {
        if let Some(mount) = self.mounts.get_mut(&mount_id) {
            if input.dismount {
                if let Some(rider) = mount.primary_rider() {
                    mount.dismount_rider(rider);
                    self.events.push(MountEvent::Dismounted {
                        rider,
                        mount: mount_id,
                    });
                    return;
                }
            }

            let walk = input.forward > 0.1;
            let run = input.forward > 0.5;
            let gallop = input.sprint && run;
            mount.set_input(walk, run, gallop);
            mount.turn(input.turn * mount.stats.turn_rate, dt);
        }
    }

    /// Get events from last update.
    pub fn events(&self) -> &[MountEvent] {
        &self.events
    }

    /// Get mount count.
    pub fn mount_count(&self) -> usize {
        self.mounts.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mount_creation() {
        let mount = Mount::new(MountId(1), "Roach", MountType::Horse);
        assert_eq!(mount.name, "Roach");
        assert_eq!(mount.mount_points.len(), 1);
        assert!(!mount.is_mounted);
    }

    #[test]
    fn test_mount_and_dismount() {
        let mut mount = Mount::new(MountId(1), "Roach", MountType::Horse);
        mount.taming.tamed = true;
        let rider = RiderId(1);

        assert!(mount.can_mount());
        let point = mount.mount_rider(rider);
        assert_eq!(point, Some(0));
        assert!(mount.is_mounted);
        assert_eq!(mount.primary_rider(), Some(rider));

        assert!(mount.dismount_rider(rider));
        assert!(!mount.is_mounted);
    }

    #[test]
    fn test_stamina_consumption() {
        let mut stats = MountStats::default();
        assert!(stats.consume_stamina(10.0));
        assert_eq!(stats.stamina, 90.0);
        assert!(!stats.consume_stamina(100.0));
    }

    #[test]
    fn test_taming_progress() {
        let mut taming = TamingState::default();
        taming.preferred_foods.push("apple".to_string());

        let result = taming.attempt_taming(TamingAction::Feed {
            food_id: "apple".to_string(),
        });
        assert!(matches!(result, TamingResult::Progress { .. }));
        assert!(taming.progress > 0.0);
    }

    #[test]
    fn test_mount_type_capabilities() {
        assert!(MountType::Dragon.can_fly());
        assert!(!MountType::Horse.can_fly());
        assert!(MountType::Boat.can_swim());
        assert!(!MountType::Wolf.can_swim());
    }
}
