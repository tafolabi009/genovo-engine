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
//! - **Taming/befriending**: full progression for taming wild mounts
//! - **Mount breeding**: combine traits from parent mounts
//! - **Mount equipment**: saddle, armor, bags affecting stats
//! - **Mount AI**: autonomous behavior when not ridden
//! - **Multi-rider mounts**: pilot and gunner seats
//! - **Mount special abilities**: charge, stomp, fly, swim
//! - **Mount fatigue and recovery**: long-distance travel costs
//! - **Stable management**: store, heal, feed mounts

use glam::{Vec3, Quat};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const MAX_MOUNTS: usize = 256;
pub const MAX_RIDERS_PER_MOUNT: usize = 4;
pub const DEFAULT_WALK_SPEED: f32 = 3.0;
pub const DEFAULT_RUN_SPEED: f32 = 8.0;
pub const DEFAULT_GALLOP_SPEED: f32 = 15.0;
pub const DEFAULT_MOUNT_HEALTH: f32 = 100.0;
pub const DEFAULT_MOUNT_STAMINA: f32 = 100.0;
pub const GALLOP_STAMINA_COST: f32 = 10.0;
pub const RUN_STAMINA_COST: f32 = 3.0;
pub const STAMINA_REGEN_RATE: f32 = 5.0;
pub const DEFAULT_CARRY_WEIGHT: f32 = 200.0;
pub const MOUNT_INTERACTION_RANGE: f32 = 3.0;
pub const TAMING_PROGRESS_PER_ACTION: f32 = 10.0;
pub const DISMOUNT_OFFSET: f32 = 2.0;
pub const MOUNT_ACCELERATION: f32 = 5.0;
pub const MOUNT_DECELERATION: f32 = 8.0;
pub const WALK_ANIM_THRESHOLD: f32 = 0.5;
pub const RUN_ANIM_THRESHOLD: f32 = 5.0;
pub const GALLOP_ANIM_THRESHOLD: f32 = 10.0;
const EPSILON: f32 = 1e-6;

// Breeding constants.
pub const BREEDING_COOLDOWN: f32 = 3600.0; // 1 hour game time.
pub const GESTATION_TIME: f32 = 7200.0; // 2 hours.

// Fatigue constants.
pub const FATIGUE_GALLOP_RATE: f32 = 0.05;
pub const FATIGUE_RUN_RATE: f32 = 0.02;
pub const FATIGUE_RECOVERY_RATE: f32 = 0.1;
pub const FATIGUE_SPEED_PENALTY: f32 = 0.3;
pub const FATIGUE_MAX: f32 = 1.0;

// Stable constants.
pub const STABLE_HEAL_RATE: f32 = 2.0;
pub const STABLE_STAMINA_REGEN: f32 = 10.0;
pub const STABLE_FATIGUE_RECOVERY: f32 = 0.5;
pub const STABLE_FEED_BOND_BONUS: f32 = 5.0;

// AI constants.
pub const AI_WANDER_RADIUS: f32 = 15.0;
pub const AI_FLEE_DISTANCE: f32 = 30.0;
pub const AI_RETURN_DISTANCE: f32 = 50.0;

// Ability constants.
pub const CHARGE_STAMINA_COST: f32 = 30.0;
pub const STOMP_STAMINA_COST: f32 = 20.0;
pub const FLY_STAMINA_COST_PER_SEC: f32 = 5.0;
pub const SWIM_STAMINA_COST_PER_SEC: f32 = 3.0;

// ---------------------------------------------------------------------------
// MountId, RiderId
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MountId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RiderId(pub u64);

// ---------------------------------------------------------------------------
// MountType
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MountType {
    Horse,
    Wolf,
    Bird,
    Dragon,
    LargeBeast,
    Mechanical,
    Boat,
    Custom(u32),
}

impl MountType {
    pub fn can_fly(&self) -> bool {
        matches!(self, MountType::Bird | MountType::Dragon)
    }

    pub fn can_swim(&self) -> bool {
        matches!(self, MountType::Boat | MountType::Dragon)
    }

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

    /// Maximum rider seats for this mount type.
    pub fn max_riders(&self) -> usize {
        match self {
            Self::Horse => 1,
            Self::Wolf => 1,
            Self::Bird => 1,
            Self::Dragon => 2,
            Self::LargeBeast => 3,
            Self::Mechanical => 2,
            Self::Boat => 4,
            Self::Custom(_) => 1,
        }
    }

    /// Available special abilities.
    pub fn available_abilities(&self) -> Vec<MountAbilityType> {
        match self {
            Self::Horse => vec![MountAbilityType::Charge],
            Self::Wolf => vec![MountAbilityType::Charge, MountAbilityType::Howl],
            Self::Bird => vec![MountAbilityType::Fly, MountAbilityType::Dive],
            Self::Dragon => vec![MountAbilityType::Fly, MountAbilityType::BreathAttack, MountAbilityType::Stomp],
            Self::LargeBeast => vec![MountAbilityType::Charge, MountAbilityType::Stomp],
            Self::Mechanical => vec![MountAbilityType::Boost],
            Self::Boat => vec![MountAbilityType::Swim],
            Self::Custom(_) => vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// MountAnimState
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MountAnimState {
    Idle,
    Walk,
    Run,
    Gallop,
    Jump,
    Flying,
    Swimming,
    Rear,
    Hit,
    Death,
    Eating,
    Taming,
    Charging,
    UsingAbility,
}

impl MountAnimState {
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
            Self::Charging => "mount_charge",
            Self::UsingAbility => "mount_ability",
        }
    }
}

// ---------------------------------------------------------------------------
// Mount Equipment
// ---------------------------------------------------------------------------

/// Equipment that can be attached to a mount.
#[derive(Debug, Clone)]
pub struct MountEquipment {
    pub item_id: String,
    pub name: String,
    pub slot: MountEquipmentSlot,
    pub speed_modifier: f32,
    pub defense_modifier: f32,
    pub stamina_modifier: f32,
    pub carry_weight_bonus: f32,
    pub inventory_slots_bonus: usize,
    pub durability: f32,
    pub max_durability: f32,
    pub quality: EquipmentQuality,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MountEquipmentSlot {
    Saddle,
    Armor,
    Barding,
    Bags,
    Horseshoes,
    Reins,
    Decoration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EquipmentQuality {
    Common,
    Uncommon,
    Rare,
    Epic,
    Legendary,
}

impl MountEquipment {
    pub fn new(item_id: &str, name: &str, slot: MountEquipmentSlot) -> Self {
        Self {
            item_id: item_id.to_string(),
            name: name.to_string(),
            slot,
            speed_modifier: 0.0,
            defense_modifier: 0.0,
            stamina_modifier: 0.0,
            carry_weight_bonus: 0.0,
            inventory_slots_bonus: 0,
            durability: 100.0,
            max_durability: 100.0,
            quality: EquipmentQuality::Common,
        }
    }

    pub fn is_broken(&self) -> bool {
        self.durability <= 0.0
    }

    pub fn wear(&mut self, amount: f32) {
        self.durability = (self.durability - amount).max(0.0);
    }

    pub fn repair(&mut self, amount: f32) {
        self.durability = (self.durability + amount).min(self.max_durability);
    }
}

// ---------------------------------------------------------------------------
// MountPoint
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MountPoint {
    pub index: usize,
    pub local_offset: Vec3,
    pub local_rotation: Quat,
    pub occupied: bool,
    pub rider_id: Option<RiderId>,
    pub allows_combat: bool,
    pub rider_animation: String,
    pub label: String,
    /// Role of this mount point (pilot, gunner, passenger).
    pub role: MountPointRole,
}

/// Role of a rider at a mount point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MountPointRole {
    /// Controls movement.
    Pilot,
    /// Can attack, doesn't control movement.
    Gunner,
    /// Passive passenger.
    Passenger,
}

impl MountPoint {
    pub fn new(index: usize, local_offset: Vec3) -> Self {
        Self {
            index,
            local_offset,
            local_rotation: Quat::IDENTITY,
            occupied: false,
            rider_id: None,
            allows_combat: index == 0,
            rider_animation: "mounted_idle".to_string(),
            label: if index == 0 {
                "saddle".to_string()
            } else {
                format!("passenger_{}", index)
            },
            role: if index == 0 {
                MountPointRole::Pilot
            } else {
                MountPointRole::Passenger
            },
        }
    }

    pub fn gunner(index: usize, local_offset: Vec3) -> Self {
        Self {
            index,
            local_offset,
            local_rotation: Quat::IDENTITY,
            occupied: false,
            rider_id: None,
            allows_combat: true,
            rider_animation: "mounted_gunner".to_string(),
            label: format!("gunner_{}", index),
            role: MountPointRole::Gunner,
        }
    }

    pub fn assign_rider(&mut self, rider: RiderId) {
        self.occupied = true;
        self.rider_id = Some(rider);
    }

    pub fn clear_rider(&mut self) {
        self.occupied = false;
        self.rider_id = None;
    }
}

// ---------------------------------------------------------------------------
// MountStats
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MountStats {
    pub walk_speed: f32,
    pub run_speed: f32,
    pub gallop_speed: f32,
    pub fly_speed: f32,
    pub swim_speed: f32,
    pub jump_force: f32,
    pub max_health: f32,
    pub health: f32,
    pub max_stamina: f32,
    pub stamina: f32,
    pub carry_weight: f32,
    pub current_weight: f32,
    pub turn_rate: f32,
    pub acceleration: f32,
    pub defense: f32,
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
    pub fn effective_gallop_speed(&self) -> f32 {
        let weight_ratio = (self.current_weight / self.carry_weight).min(1.0);
        self.gallop_speed * (1.0 - weight_ratio * 0.3)
    }

    pub fn is_overburdened(&self) -> bool {
        self.current_weight > self.carry_weight
    }

    pub fn is_alive(&self) -> bool {
        self.health > 0.0
    }

    pub fn take_damage(&mut self, amount: f32) -> f32 {
        let actual = (amount - self.defense).max(1.0);
        self.health = (self.health - actual).max(0.0);
        actual
    }

    pub fn heal(&mut self, amount: f32) {
        self.health = (self.health + amount).min(self.max_health);
    }

    pub fn consume_stamina(&mut self, amount: f32) -> bool {
        if self.stamina >= amount {
            self.stamina -= amount;
            true
        } else {
            false
        }
    }

    pub fn regen_stamina(&mut self, dt: f32) {
        self.stamina = (self.stamina + STAMINA_REGEN_RATE * dt).min(self.max_stamina);
    }

    /// Apply equipment modifiers.
    pub fn apply_equipment(&mut self, equipment: &[MountEquipment]) {
        for equip in equipment {
            if equip.is_broken() {
                continue;
            }
            self.gallop_speed += equip.speed_modifier;
            self.run_speed += equip.speed_modifier * 0.5;
            self.defense += equip.defense_modifier;
            self.max_stamina += equip.stamina_modifier;
            self.carry_weight += equip.carry_weight_bonus;
        }
    }
}

// ---------------------------------------------------------------------------
// TamingState
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TamingState {
    pub tamed: bool,
    pub progress: f32,
    pub required_progress: f32,
    pub preferred_foods: Vec<String>,
    pub temperament: MountTemperament,
    pub failed_attempts: u32,
    pub cooldown: f32,
    pub cooldown_remaining: f32,
    pub trust_level: f32,
    pub bond_level: f32,
    pub max_bond: f32,
    /// Approach distance tracking (how close the player got).
    pub approach_distance: f32,
    /// Whether the mount is currently fleeing from the player.
    pub fleeing: bool,
    /// Fear level (builds up from failed attempts).
    pub fear_level: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MountTemperament {
    Docile,
    Neutral,
    Skittish,
    Aggressive,
    Wild,
}

impl MountTemperament {
    pub fn difficulty_multiplier(&self) -> f32 {
        match self {
            Self::Docile => 0.5,
            Self::Neutral => 1.0,
            Self::Skittish => 1.5,
            Self::Aggressive => 2.0,
            Self::Wild => 5.0,
        }
    }

    /// How close the player can get before the mount reacts.
    pub fn reaction_distance(&self) -> f32 {
        match self {
            Self::Docile => 3.0,
            Self::Neutral => 5.0,
            Self::Skittish => 10.0,
            Self::Aggressive => 8.0,
            Self::Wild => 15.0,
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
            bond_level: 0.0,
            max_bond: 100.0,
            approach_distance: f32::MAX,
            fleeing: false,
            fear_level: 0.0,
        }
    }
}

impl TamingState {
    pub fn attempt_taming(&mut self, action: TamingAction) -> TamingResult {
        if self.tamed {
            return TamingResult::AlreadyTamed;
        }
        if self.cooldown_remaining > 0.0 {
            return TamingResult::OnCooldown;
        }
        if self.fleeing {
            return TamingResult::Failed { reason: "Mount is fleeing!".to_string() };
        }

        let progress_gained = match action {
            TamingAction::Feed { ref food_id } => {
                let is_preferred = self.preferred_foods.iter().any(|f| f == food_id);
                let base = if is_preferred {
                    TAMING_PROGRESS_PER_ACTION * 2.0
                } else {
                    TAMING_PROGRESS_PER_ACTION
                };
                // Fear reduces effectiveness.
                base * (1.0 - self.fear_level * 0.5)
            }
            TamingAction::Pet => {
                if self.trust_level < 0.2 {
                    // Too early to pet, mount may flee.
                    self.fear_level = (self.fear_level + 0.1).min(1.0);
                    return TamingResult::Failed {
                        reason: "Mount is not ready to be touched.".to_string(),
                    };
                }
                TAMING_PROGRESS_PER_ACTION * 0.5 * (1.0 + self.trust_level)
            }
            TamingAction::Approach => {
                TAMING_PROGRESS_PER_ACTION * 0.2
            }
            TamingAction::UseSpecialItem => {
                TAMING_PROGRESS_PER_ACTION * 5.0
            }
            TamingAction::Sing => {
                TAMING_PROGRESS_PER_ACTION * 0.3 * (1.0 + self.trust_level)
            }
            TamingAction::Wait => {
                // Patiently waiting reduces fear.
                self.fear_level = (self.fear_level - 0.05).max(0.0);
                TAMING_PROGRESS_PER_ACTION * 0.1
            }
        };

        let adjusted = progress_gained / self.temperament.difficulty_multiplier();
        self.progress += adjusted;
        self.trust_level = (self.progress / self.required_progress).min(1.0);

        if self.progress >= self.required_progress {
            self.tamed = true;
            self.trust_level = 1.0;
            self.bond_level = 10.0;
            self.fear_level = 0.0;
            TamingResult::Success
        } else {
            self.cooldown_remaining = self.cooldown;
            TamingResult::Progress {
                current: self.progress,
                required: self.required_progress,
                trust: self.trust_level,
            }
        }
    }

    /// Increase bond from positive interactions.
    pub fn increase_bond(&mut self, amount: f32) {
        if self.tamed {
            self.bond_level = (self.bond_level + amount).min(self.max_bond);
        }
    }

    /// Get bond level as a fraction (0-1).
    pub fn bond_fraction(&self) -> f32 {
        self.bond_level / self.max_bond
    }

    pub fn update(&mut self, dt: f32) {
        if self.cooldown_remaining > 0.0 {
            self.cooldown_remaining = (self.cooldown_remaining - dt).max(0.0);
        }
        // Fear decays slowly.
        self.fear_level = (self.fear_level - 0.01 * dt).max(0.0);
    }
}

#[derive(Debug, Clone)]
pub enum TamingAction {
    Feed { food_id: String },
    Pet,
    Approach,
    UseSpecialItem,
    Sing,
    Wait,
}

#[derive(Debug, Clone)]
pub enum TamingResult {
    Success,
    Progress { current: f32, required: f32, trust: f32 },
    AlreadyTamed,
    OnCooldown,
    Failed { reason: String },
}

// ---------------------------------------------------------------------------
// Mount Breeding
// ---------------------------------------------------------------------------

/// Genetic traits that can be inherited.
#[derive(Debug, Clone)]
pub struct MountGenetics {
    /// Speed gene (0-1, affects max speeds).
    pub speed_gene: f32,
    /// Stamina gene.
    pub stamina_gene: f32,
    /// Strength gene (affects carry weight, attack).
    pub strength_gene: f32,
    /// Temperament gene (lower = more docile).
    pub temperament_gene: f32,
    /// Special trait (rare abilities).
    pub special_trait: Option<SpecialTrait>,
    /// Coat color index.
    pub coat_color: u32,
    /// Size modifier.
    pub size_modifier: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialTrait {
    /// Faster stamina regeneration.
    Endurance,
    /// Higher top speed.
    Swiftness,
    /// More carry weight.
    PackMule,
    /// Better combat stats.
    Warhorse,
    /// Reduced fall damage.
    SureFooted,
    /// Better taming (offspring are easier to train).
    GentleNature,
    /// Rare coloring.
    Exotic,
}

impl MountGenetics {
    pub fn random(rng_state: &mut u64) -> Self {
        let rng = |state: &mut u64| -> f32 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*state >> 33) as f32) / (u32::MAX as f32)
        };

        Self {
            speed_gene: rng(rng_state) * 0.5 + 0.25,
            stamina_gene: rng(rng_state) * 0.5 + 0.25,
            strength_gene: rng(rng_state) * 0.5 + 0.25,
            temperament_gene: rng(rng_state),
            special_trait: if rng(rng_state) < 0.1 {
                Some(SpecialTrait::Endurance)
            } else {
                None
            },
            coat_color: (rng(rng_state) * 10.0) as u32,
            size_modifier: 0.9 + rng(rng_state) * 0.2,
        }
    }

    /// Breed two genetic profiles. Each gene is randomly picked from one parent
    /// with small mutation.
    pub fn breed(parent_a: &Self, parent_b: &Self, rng_state: &mut u64) -> Self {
        let rng = |state: &mut u64| -> f32 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*state >> 33) as f32) / (u32::MAX as f32)
        };

        let pick = |a: f32, b: f32, state: &mut u64| -> f32 {
            let base = if rng(state) < 0.5 { a } else { b };
            let mutation = (rng(state) - 0.5) * 0.1;
            (base + mutation).clamp(0.0, 1.0)
        };

        let special = match (&parent_a.special_trait, &parent_b.special_trait) {
            (Some(a), Some(b)) => {
                if rng(rng_state) < 0.5 { Some(*a) } else { Some(*b) }
            }
            (Some(a), None) => {
                if rng(rng_state) < 0.4 { Some(*a) } else { None }
            }
            (None, Some(b)) => {
                if rng(rng_state) < 0.4 { Some(*b) } else { None }
            }
            (None, None) => {
                // Small chance of spontaneous special trait.
                if rng(rng_state) < 0.05 { Some(SpecialTrait::Exotic) } else { None }
            }
        };

        Self {
            speed_gene: pick(parent_a.speed_gene, parent_b.speed_gene, rng_state),
            stamina_gene: pick(parent_a.stamina_gene, parent_b.stamina_gene, rng_state),
            strength_gene: pick(parent_a.strength_gene, parent_b.strength_gene, rng_state),
            temperament_gene: pick(parent_a.temperament_gene, parent_b.temperament_gene, rng_state),
            special_trait: special,
            coat_color: if rng(rng_state) < 0.5 {
                parent_a.coat_color
            } else {
                parent_b.coat_color
            },
            size_modifier: pick(parent_a.size_modifier, parent_b.size_modifier, rng_state),
        }
    }

    /// Apply genetics to mount stats.
    pub fn apply_to_stats(&self, stats: &mut MountStats) {
        stats.walk_speed *= 0.8 + self.speed_gene * 0.4;
        stats.run_speed *= 0.8 + self.speed_gene * 0.4;
        stats.gallop_speed *= 0.8 + self.speed_gene * 0.4;
        stats.max_stamina *= 0.8 + self.stamina_gene * 0.4;
        stats.stamina = stats.max_stamina;
        stats.carry_weight *= 0.8 + self.strength_gene * 0.4;
        stats.attack_damage *= 0.8 + self.strength_gene * 0.4;

        if let Some(trait_) = &self.special_trait {
            match trait_ {
                SpecialTrait::Endurance => stats.max_stamina *= 1.3,
                SpecialTrait::Swiftness => {
                    stats.gallop_speed *= 1.2;
                    stats.run_speed *= 1.15;
                }
                SpecialTrait::PackMule => stats.carry_weight *= 1.5,
                SpecialTrait::Warhorse => {
                    stats.attack_damage *= 1.3;
                    stats.defense *= 1.2;
                }
                SpecialTrait::SureFooted => {} // Handled elsewhere.
                SpecialTrait::GentleNature => {} // Affects offspring taming.
                SpecialTrait::Exotic => {} // Cosmetic.
            }
        }
    }
}

/// Breeding state for a mount pair.
#[derive(Debug, Clone)]
pub struct BreedingState {
    pub parent_a: MountId,
    pub parent_b: MountId,
    pub started_at: f64,
    pub gestation_time: f32,
    pub offspring_genetics: MountGenetics,
    pub complete: bool,
}

impl BreedingState {
    pub fn new(parent_a: MountId, parent_b: MountId, genetics: MountGenetics, game_time: f64) -> Self {
        Self {
            parent_a,
            parent_b,
            started_at: game_time,
            gestation_time: GESTATION_TIME,
            offspring_genetics: genetics,
            complete: false,
        }
    }

    pub fn update(&mut self, game_time: f64) {
        if (game_time - self.started_at) as f32 >= self.gestation_time {
            self.complete = true;
        }
    }

    pub fn progress(&self, game_time: f64) -> f32 {
        ((game_time - self.started_at) as f32 / self.gestation_time).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Mount Special Abilities
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MountAbilityType {
    Charge,
    Stomp,
    Fly,
    Swim,
    BreathAttack,
    Howl,
    Dive,
    Boost,
}

#[derive(Debug, Clone)]
pub struct MountAbility {
    pub ability_type: MountAbilityType,
    pub name: String,
    pub stamina_cost: f32,
    pub cooldown: f32,
    pub cooldown_remaining: f32,
    pub damage: f32,
    pub range: f32,
    pub duration: f32,
    pub active: bool,
    pub active_timer: f32,
}

impl MountAbility {
    pub fn new(ability_type: MountAbilityType) -> Self {
        let (name, stamina, cooldown, damage, range, duration) = match ability_type {
            MountAbilityType::Charge => ("Charge", CHARGE_STAMINA_COST, 5.0, 30.0, 15.0, 1.5),
            MountAbilityType::Stomp => ("Stomp", STOMP_STAMINA_COST, 8.0, 25.0, 5.0, 0.5),
            MountAbilityType::Fly => ("Fly", 0.0, 0.0, 0.0, 0.0, 0.0),
            MountAbilityType::Swim => ("Swim", 0.0, 0.0, 0.0, 0.0, 0.0),
            MountAbilityType::BreathAttack => ("Breath Attack", 40.0, 10.0, 50.0, 20.0, 2.0),
            MountAbilityType::Howl => ("Howl", 15.0, 15.0, 0.0, 30.0, 3.0),
            MountAbilityType::Dive => ("Dive", 20.0, 6.0, 35.0, 10.0, 1.0),
            MountAbilityType::Boost => ("Boost", 25.0, 10.0, 0.0, 0.0, 3.0),
        };

        Self {
            ability_type,
            name: name.to_string(),
            stamina_cost: stamina,
            cooldown,
            cooldown_remaining: 0.0,
            damage,
            range,
            duration,
            active: false,
            active_timer: 0.0,
        }
    }

    pub fn can_use(&self, stamina: f32) -> bool {
        self.cooldown_remaining <= 0.0 && stamina >= self.stamina_cost && !self.active
    }

    pub fn activate(&mut self) {
        self.active = true;
        self.active_timer = self.duration;
        self.cooldown_remaining = self.cooldown;
    }

    pub fn update(&mut self, dt: f32) {
        if self.cooldown_remaining > 0.0 {
            self.cooldown_remaining = (self.cooldown_remaining - dt).max(0.0);
        }
        if self.active {
            self.active_timer -= dt;
            if self.active_timer <= 0.0 {
                self.active = false;
                self.active_timer = 0.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Mount AI (when not ridden)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MountAIState {
    Idle,
    Wandering,
    Grazing,
    ReturningToStable,
    FleeingDanger,
    FollowingOwner,
    Sleeping,
}

#[derive(Debug, Clone)]
pub struct MountAI {
    pub state: MountAIState,
    pub home_position: Vec3,
    pub owner_position: Option<Vec3>,
    pub danger_position: Option<Vec3>,
    pub wander_target: Option<Vec3>,
    pub state_timer: f32,
    pub idle_duration: f32,
    pub wander_radius: f32,
    pub follow_distance: f32,
    rng_state: u64,
}

impl MountAI {
    pub fn new(home: Vec3) -> Self {
        Self {
            state: MountAIState::Idle,
            home_position: home,
            owner_position: None,
            danger_position: None,
            wander_target: None,
            state_timer: 0.0,
            idle_duration: 5.0,
            wander_radius: AI_WANDER_RADIUS,
            follow_distance: 10.0,
            rng_state: 12345,
        }
    }

    fn next_rng(&mut self) -> f32 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.rng_state >> 33) as f32) / (u32::MAX as f32)
    }

    /// Update AI state machine. Returns desired movement direction and speed.
    pub fn update(&mut self, dt: f32, mount_pos: Vec3, mount_tamed: bool) -> (Vec3, f32) {
        self.state_timer += dt;

        // Check for danger (highest priority).
        if let Some(danger_pos) = self.danger_position {
            let dist = mount_pos.distance(danger_pos);
            if dist < AI_FLEE_DISTANCE {
                self.state = MountAIState::FleeingDanger;
                let flee_dir = (mount_pos - danger_pos).normalize();
                return (flee_dir, DEFAULT_GALLOP_SPEED);
            } else {
                self.danger_position = None;
            }
        }

        // If tamed and owner is far, follow owner.
        if mount_tamed {
            if let Some(owner_pos) = self.owner_position {
                let dist = mount_pos.distance(owner_pos);
                if dist > self.follow_distance {
                    self.state = MountAIState::FollowingOwner;
                    let dir = (owner_pos - mount_pos).normalize();
                    let speed = if dist > self.follow_distance * 2.0 {
                        DEFAULT_RUN_SPEED
                    } else {
                        DEFAULT_WALK_SPEED
                    };
                    return (dir, speed);
                }
            }
        }

        // If too far from home, return.
        let dist_from_home = mount_pos.distance(self.home_position);
        if dist_from_home > AI_RETURN_DISTANCE {
            self.state = MountAIState::ReturningToStable;
            let dir = (self.home_position - mount_pos).normalize();
            return (dir, DEFAULT_RUN_SPEED);
        }

        // State machine for idle/wander/graze.
        match self.state {
            MountAIState::Idle => {
                if self.state_timer >= self.idle_duration {
                    self.state_timer = 0.0;
                    let r = self.next_rng();
                    if r < 0.4 {
                        self.state = MountAIState::Wandering;
                        let angle = self.next_rng() * std::f32::consts::PI * 2.0;
                        let dist = self.next_rng() * self.wander_radius;
                        self.wander_target = Some(Vec3::new(
                            self.home_position.x + angle.cos() * dist,
                            mount_pos.y,
                            self.home_position.z + angle.sin() * dist,
                        ));
                    } else if r < 0.7 {
                        self.state = MountAIState::Grazing;
                    } else {
                        // Stay idle a bit more.
                        self.idle_duration = 3.0 + self.next_rng() * 5.0;
                    }
                }
                (Vec3::ZERO, 0.0)
            }
            MountAIState::Wandering => {
                if let Some(target) = self.wander_target {
                    let dist = mount_pos.distance(target);
                    if dist < 1.0 || self.state_timer > 10.0 {
                        self.state = MountAIState::Idle;
                        self.state_timer = 0.0;
                        self.wander_target = None;
                        (Vec3::ZERO, 0.0)
                    } else {
                        let dir = (target - mount_pos).normalize();
                        (dir, DEFAULT_WALK_SPEED)
                    }
                } else {
                    self.state = MountAIState::Idle;
                    self.state_timer = 0.0;
                    (Vec3::ZERO, 0.0)
                }
            }
            MountAIState::Grazing => {
                if self.state_timer > 8.0 {
                    self.state = MountAIState::Idle;
                    self.state_timer = 0.0;
                }
                (Vec3::ZERO, 0.0)
            }
            MountAIState::FleeingDanger => {
                // Handled above, but if we get here, go idle.
                self.state = MountAIState::Idle;
                self.state_timer = 0.0;
                (Vec3::ZERO, 0.0)
            }
            MountAIState::FollowingOwner => {
                // Handled above.
                self.state = MountAIState::Idle;
                self.state_timer = 0.0;
                (Vec3::ZERO, 0.0)
            }
            MountAIState::ReturningToStable => {
                let dir = (self.home_position - mount_pos).normalize();
                if dist_from_home < 5.0 {
                    self.state = MountAIState::Idle;
                    self.state_timer = 0.0;
                    (Vec3::ZERO, 0.0)
                } else {
                    (dir, DEFAULT_WALK_SPEED)
                }
            }
            MountAIState::Sleeping => {
                if self.state_timer > 30.0 {
                    self.state = MountAIState::Idle;
                    self.state_timer = 0.0;
                }
                (Vec3::ZERO, 0.0)
            }
        }
    }

    pub fn set_danger(&mut self, pos: Vec3) {
        self.danger_position = Some(pos);
    }

    pub fn set_owner_position(&mut self, pos: Vec3) {
        self.owner_position = Some(pos);
    }
}

// ---------------------------------------------------------------------------
// Mount (main struct)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Mount {
    pub id: MountId,
    pub name: String,
    pub mount_type: MountType,
    pub stats: MountStats,
    pub base_stats: MountStats,
    pub mount_points: Vec<MountPoint>,
    pub anim_state: MountAnimState,
    pub position: Vec3,
    pub rotation: Quat,
    pub velocity: Vec3,
    pub current_speed: f32,
    pub target_speed: f32,
    pub taming: TamingState,
    pub is_mounted: bool,
    pub has_inventory: bool,
    pub inventory_slots: usize,
    pub owner: Option<RiderId>,
    pub tag: Option<String>,
    pub allows_combat: bool,
    pub status_effects: Vec<MountStatusEffect>,
    pub equipment: HashMap<MountEquipmentSlot, MountEquipment>,
    pub abilities: Vec<MountAbility>,
    pub genetics: MountGenetics,
    pub ai: MountAI,
    pub fatigue: f32,
    pub rng_state: u64,
}

#[derive(Debug, Clone)]
pub struct MountStatusEffect {
    pub name: String,
    pub duration: f32,
    pub speed_modifier: f32,
    pub stamina_modifier: f32,
}

impl Mount {
    pub fn new(id: MountId, name: impl Into<String>, mount_type: MountType) -> Self {
        let max_riders = mount_type.max_riders();
        let mut mount_points = vec![MountPoint::new(0, Vec3::new(0.0, 1.5, 0.0))];
        if max_riders >= 2 {
            mount_points.push(MountPoint::gunner(1, Vec3::new(0.0, 1.5, -1.0)));
        }
        for i in 2..max_riders {
            mount_points.push(MountPoint::new(i, Vec3::new(
                if i % 2 == 0 { -0.8 } else { 0.8 },
                1.5,
                -(i as f32) * 0.8,
            )));
        }

        let abilities: Vec<MountAbility> = mount_type.available_abilities()
            .iter()
            .map(|&a| MountAbility::new(a))
            .collect();

        let mut rng_state = id.0.wrapping_mul(12345) + 1;
        let genetics = MountGenetics::random(&mut rng_state);

        let mut stats = MountStats::default();
        genetics.apply_to_stats(&mut stats);
        let base_stats = stats.clone();

        Self {
            id,
            name: name.into(),
            mount_type,
            stats,
            base_stats,
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
            equipment: HashMap::new(),
            abilities,
            genetics,
            ai: MountAI::new(Vec3::ZERO),
            fatigue: 0.0,
            rng_state,
        }
    }

    /// Equip an item to the mount.
    pub fn equip(&mut self, item: MountEquipment) -> Option<MountEquipment> {
        let old = self.equipment.insert(item.slot, item);
        self.recalculate_stats();
        old
    }

    /// Unequip an item from a slot.
    pub fn unequip(&mut self, slot: MountEquipmentSlot) -> Option<MountEquipment> {
        let old = self.equipment.remove(&slot);
        self.recalculate_stats();
        old
    }

    /// Recalculate stats from base + equipment + genetics.
    fn recalculate_stats(&mut self) {
        self.stats = self.base_stats.clone();
        self.genetics.apply_to_stats(&mut self.stats);
        let equips: Vec<MountEquipment> = self.equipment.values().cloned().collect();
        self.stats.apply_equipment(&equips);

        // Update inventory slots from bag equipment.
        self.inventory_slots = 8;
        for equip in self.equipment.values() {
            if equip.slot == MountEquipmentSlot::Bags && !equip.is_broken() {
                self.inventory_slots += equip.inventory_slots_bonus;
            }
        }
    }

    pub fn can_mount(&self) -> bool {
        self.taming.tamed
            && self.stats.is_alive()
            && self.mount_points.iter().any(|p| !p.occupied)
    }

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

    /// Mount a rider at a specific point.
    pub fn mount_rider_at(&mut self, rider: RiderId, point_index: usize) -> bool {
        if !self.can_mount() {
            return false;
        }
        if let Some(point) = self.mount_points.iter_mut().find(|p| p.index == point_index) {
            if !point.occupied {
                point.assign_rider(rider);
                self.is_mounted = true;
                return true;
            }
        }
        false
    }

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

    pub fn primary_rider(&self) -> Option<RiderId> {
        self.mount_points.first().and_then(|p| p.rider_id)
    }

    pub fn all_riders(&self) -> Vec<RiderId> {
        self.mount_points
            .iter()
            .filter_map(|p| p.rider_id)
            .collect()
    }

    /// Get the gunner riders.
    pub fn gunners(&self) -> Vec<RiderId> {
        self.mount_points.iter()
            .filter(|p| p.role == MountPointRole::Gunner && p.occupied)
            .filter_map(|p| p.rider_id)
            .collect()
    }

    /// Use a special ability.
    pub fn use_ability(&mut self, ability_type: MountAbilityType) -> bool {
        let stamina = self.stats.stamina;
        if let Some(ability) = self.abilities.iter_mut().find(|a| a.ability_type == ability_type) {
            if ability.can_use(stamina) {
                ability.activate();
                self.stats.consume_stamina(ability.stamina_cost);
                self.anim_state = MountAnimState::UsingAbility;
                return true;
            }
        }
        false
    }

    pub fn update(&mut self, dt: f32) {
        // Update taming.
        self.taming.update(dt);

        // Update status effects.
        self.status_effects.retain_mut(|effect| {
            effect.duration -= dt;
            effect.duration > 0.0
        });

        // Update abilities.
        for ability in &mut self.abilities {
            ability.update(dt);
        }

        // Wear equipment.
        if self.is_mounted && self.current_speed > WALK_ANIM_THRESHOLD {
            for equip in self.equipment.values_mut() {
                equip.wear(0.001 * dt);
            }
        }

        // AI behavior when not mounted.
        if !self.is_mounted {
            let (ai_dir, ai_speed) = self.ai.update(dt, self.position, self.taming.tamed);
            if ai_speed > 0.0 {
                self.target_speed = ai_speed;
                let look_dir = ai_dir;
                if look_dir.length() > 0.1 {
                    let angle = look_dir.z.atan2(look_dir.x);
                    self.rotation = Quat::from_rotation_y(-angle + std::f32::consts::FRAC_PI_2);
                }
            } else {
                self.target_speed = 0.0;
            }
        }

        // Update speed.
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

        // Apply speed modifiers.
        let mut speed_mod = 1.0;
        for effect in &self.status_effects {
            speed_mod *= effect.speed_modifier;
        }

        // Apply fatigue penalty.
        speed_mod *= 1.0 - self.fatigue * FATIGUE_SPEED_PENALTY;

        let effective_speed = self.current_speed * speed_mod;

        // Stamina consumption.
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
                    self.target_speed = self.stats.run_speed;
                }
            } else {
                self.stats.regen_stamina(dt);
            }

            // Fatigue accumulation.
            if effective_speed >= GALLOP_ANIM_THRESHOLD {
                self.fatigue = (self.fatigue + FATIGUE_GALLOP_RATE * dt).min(FATIGUE_MAX);
            } else if effective_speed >= RUN_ANIM_THRESHOLD {
                self.fatigue = (self.fatigue + FATIGUE_RUN_RATE * dt).min(FATIGUE_MAX);
            } else if effective_speed < WALK_ANIM_THRESHOLD {
                self.fatigue = (self.fatigue - FATIGUE_RECOVERY_RATE * dt).max(0.0);
            }
        } else {
            self.stats.regen_stamina(dt);
            self.fatigue = (self.fatigue - FATIGUE_RECOVERY_RATE * dt * 0.5).max(0.0);
        }

        // Continuous ability costs (flying, swimming).
        for ability in &self.abilities {
            if ability.active {
                let cost = match ability.ability_type {
                    MountAbilityType::Fly => FLY_STAMINA_COST_PER_SEC * dt,
                    MountAbilityType::Swim => SWIM_STAMINA_COST_PER_SEC * dt,
                    _ => 0.0,
                };
                if cost > 0.0 {
                    self.stats.consume_stamina(cost);
                }
            }
        }

        // Update velocity and position.
        let forward = self.rotation * Vec3::new(0.0, 0.0, 1.0);
        self.velocity = forward * effective_speed;
        self.position += self.velocity * dt;

        // Update animation state.
        self.update_animation(effective_speed);

        // Bond increases while mounted.
        if self.is_mounted && self.taming.tamed {
            self.taming.increase_bond(0.01 * dt);
        }
    }

    fn update_animation(&mut self, speed: f32) {
        if !self.stats.is_alive() {
            self.anim_state = MountAnimState::Death;
            return;
        }

        // Check if an ability animation is playing.
        if self.abilities.iter().any(|a| a.active) {
            return; // Keep the ability animation.
        }

        let is_flying = self.abilities.iter().any(|a| a.ability_type == MountAbilityType::Fly && a.active);
        let is_swimming = self.abilities.iter().any(|a| a.ability_type == MountAbilityType::Swim && a.active);

        if is_flying {
            self.anim_state = MountAnimState::Flying;
        } else if is_swimming {
            self.anim_state = MountAnimState::Swimming;
        } else if !self.is_mounted && self.ai.state == MountAIState::Grazing {
            self.anim_state = MountAnimState::Eating;
        } else if speed < WALK_ANIM_THRESHOLD {
            self.anim_state = MountAnimState::Idle;
        } else if speed < RUN_ANIM_THRESHOLD {
            self.anim_state = MountAnimState::Walk;
        } else if speed < GALLOP_ANIM_THRESHOLD {
            self.anim_state = MountAnimState::Run;
        } else {
            self.anim_state = MountAnimState::Gallop;
        }
    }

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

    pub fn turn(&mut self, angle_degrees: f32, dt: f32) {
        let max_turn = self.stats.turn_rate * dt;
        let clamped = angle_degrees.clamp(-max_turn, max_turn);
        let turn_radians = clamped.to_radians();
        self.rotation = self.rotation * Quat::from_rotation_y(turn_radians);
    }
}

// ---------------------------------------------------------------------------
// Stable Management
// ---------------------------------------------------------------------------

/// A slot in a stable for housing mounts.
#[derive(Debug, Clone)]
pub struct StableSlot {
    pub mount_id: Option<MountId>,
    pub feed_timer: f32,
    pub last_fed: f64,
    pub clean: bool,
}

/// A stable that houses and manages mounts.
pub struct Stable {
    pub name: String,
    pub slots: Vec<StableSlot>,
    pub capacity: usize,
    pub position: Vec3,
    pub auto_feed: bool,
    pub auto_heal: bool,
}

impl Stable {
    pub fn new(name: &str, capacity: usize, position: Vec3) -> Self {
        Self {
            name: name.to_string(),
            slots: (0..capacity).map(|_| StableSlot {
                mount_id: None,
                feed_timer: 0.0,
                last_fed: 0.0,
                clean: true,
            }).collect(),
            capacity,
            position,
            auto_feed: false,
            auto_heal: false,
        }
    }

    /// Store a mount in the stable.
    pub fn store_mount(&mut self, mount_id: MountId) -> bool {
        for slot in &mut self.slots {
            if slot.mount_id.is_none() {
                slot.mount_id = Some(mount_id);
                return true;
            }
        }
        false
    }

    /// Remove a mount from the stable.
    pub fn retrieve_mount(&mut self, mount_id: MountId) -> bool {
        for slot in &mut self.slots {
            if slot.mount_id == Some(mount_id) {
                slot.mount_id = None;
                return true;
            }
        }
        false
    }

    /// Feed a specific mount in the stable (increases bond).
    pub fn feed_mount(&mut self, mount_id: MountId, game_time: f64) -> bool {
        for slot in &mut self.slots {
            if slot.mount_id == Some(mount_id) {
                slot.last_fed = game_time;
                slot.feed_timer = 0.0;
                return true;
            }
        }
        false
    }

    /// Update all mounts in the stable (heal, restore stamina, reduce fatigue).
    pub fn update(&mut self, dt: f32, mounts: &mut HashMap<MountId, Mount>) {
        for slot in &mut self.slots {
            if let Some(mount_id) = slot.mount_id {
                if let Some(mount) = mounts.get_mut(&mount_id) {
                    // Heal.
                    mount.stats.heal(STABLE_HEAL_RATE * dt);
                    // Restore stamina.
                    mount.stats.stamina = (mount.stats.stamina + STABLE_STAMINA_REGEN * dt)
                        .min(mount.stats.max_stamina);
                    // Reduce fatigue.
                    mount.fatigue = (mount.fatigue - STABLE_FATIGUE_RECOVERY * dt).max(0.0);
                }
            }
        }
    }

    /// Get stored mount IDs.
    pub fn stored_mounts(&self) -> Vec<MountId> {
        self.slots.iter().filter_map(|s| s.mount_id).collect()
    }

    /// Count of occupied slots.
    pub fn occupied_count(&self) -> usize {
        self.slots.iter().filter(|s| s.mount_id.is_some()).count()
    }

    /// Available slots.
    pub fn available_slots(&self) -> usize {
        self.capacity - self.occupied_count()
    }
}

// ---------------------------------------------------------------------------
// MountInput, MountComponent, RiderComponent
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MountInput {
    pub forward: f32,
    pub turn: f32,
    pub sprint: bool,
    pub jump: bool,
    pub dismount: bool,
    pub attack: bool,
    pub ability: Option<MountAbilityType>,
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
            ability: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MountComponent {
    pub mount: Mount,
    pub show_name: bool,
    pub interaction_prompt: String,
}

impl MountComponent {
    pub fn new(mount: Mount) -> Self {
        Self {
            mount,
            show_name: true,
            interaction_prompt: "Press E to mount".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RiderComponent {
    pub rider_id: RiderId,
    pub current_mount: Option<MountId>,
    pub mount_point_index: Option<usize>,
    pub is_controller: bool,
    pub position_offset: Vec3,
}

impl RiderComponent {
    pub fn new(rider_id: RiderId) -> Self {
        Self {
            rider_id,
            current_mount: None,
            mount_point_index: None,
            is_controller: false,
            position_offset: Vec3::ZERO,
        }
    }

    pub fn is_mounted(&self) -> bool {
        self.current_mount.is_some()
    }
}

// ---------------------------------------------------------------------------
// MountSystem
// ---------------------------------------------------------------------------

/// Events emitted by the mount system.
#[derive(Debug, Clone)]
pub enum MountEvent {
    Mounted { rider: RiderId, mount: MountId, point: usize },
    Dismounted { rider: RiderId, mount: MountId },
    Tamed { mount: MountId, owner: RiderId },
    MountDamaged { mount: MountId, damage: f32 },
    MountDied { mount: MountId },
    StaminaDepleted { mount: MountId },
    TamingProgress { mount: MountId, progress: f32, required: f32 },
    AbilityUsed { mount: MountId, ability: MountAbilityType },
    EquipmentBroken { mount: MountId, slot: MountEquipmentSlot },
    BreedingComplete { parent_a: MountId, parent_b: MountId, offspring: MountId },
    MountStabled { mount: MountId, stable: String },
    MountRetrieved { mount: MountId, stable: String },
    FatigueWarning { mount: MountId, fatigue: f32 },
    BondLevelUp { mount: MountId, new_level: f32 },
}

pub struct MountSystem {
    mounts: HashMap<MountId, Mount>,
    stables: Vec<Stable>,
    active_breedings: Vec<BreedingState>,
    next_id: u64,
    events: Vec<MountEvent>,
    rng_state: u64,
}

impl MountSystem {
    pub fn new() -> Self {
        Self {
            mounts: HashMap::new(),
            stables: Vec::new(),
            active_breedings: Vec::new(),
            next_id: 1,
            events: Vec::new(),
            rng_state: 42,
        }
    }

    pub fn register_mount(&mut self, mut mount: Mount) -> MountId {
        let id = MountId(self.next_id);
        self.next_id += 1;
        mount.id = id;
        self.mounts.insert(id, mount);
        id
    }

    pub fn get_mount(&self, id: MountId) -> Option<&Mount> {
        self.mounts.get(&id)
    }

    pub fn get_mount_mut(&mut self, id: MountId) -> Option<&mut Mount> {
        self.mounts.get_mut(&id)
    }

    pub fn add_stable(&mut self, stable: Stable) {
        self.stables.push(stable);
    }

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

    /// Start breeding two mounts.
    pub fn start_breeding(&mut self, parent_a: MountId, parent_b: MountId, game_time: f64) -> bool {
        let gen_a = self.mounts.get(&parent_a).map(|m| m.genetics.clone());
        let gen_b = self.mounts.get(&parent_b).map(|m| m.genetics.clone());

        if let (Some(ga), Some(gb)) = (gen_a, gen_b) {
            let offspring_genetics = MountGenetics::breed(&ga, &gb, &mut self.rng_state);
            self.active_breedings.push(BreedingState::new(
                parent_a,
                parent_b,
                offspring_genetics,
                game_time,
            ));
            true
        } else {
            false
        }
    }

    /// Store a mount in a stable.
    pub fn stable_mount(&mut self, mount_id: MountId, stable_index: usize) -> bool {
        if stable_index >= self.stables.len() {
            return false;
        }
        let stable_name = self.stables[stable_index].name.clone();
        if self.stables[stable_index].store_mount(mount_id) {
            self.events.push(MountEvent::MountStabled {
                mount: mount_id,
                stable: stable_name,
            });
            true
        } else {
            false
        }
    }

    /// Retrieve a mount from a stable.
    pub fn retrieve_from_stable(&mut self, mount_id: MountId) -> bool {
        for stable in &mut self.stables {
            if stable.retrieve_mount(mount_id) {
                self.events.push(MountEvent::MountRetrieved {
                    mount: mount_id,
                    stable: stable.name.clone(),
                });
                return true;
            }
        }
        false
    }

    pub fn update(&mut self, dt: f32, game_time: f64) {
        self.events.clear();

        // Update all mounts.
        let ids: Vec<MountId> = self.mounts.keys().copied().collect();
        for id in ids {
            if let Some(mount) = self.mounts.get_mut(&id) {
                mount.update(dt);

                // Check for fatigue warning.
                if mount.fatigue > 0.7 {
                    self.events.push(MountEvent::FatigueWarning {
                        mount: id,
                        fatigue: mount.fatigue,
                    });
                }

                // Check for broken equipment.
                let broken_slots: Vec<MountEquipmentSlot> = mount.equipment.iter()
                    .filter(|(_, e)| e.is_broken())
                    .map(|(&slot, _)| slot)
                    .collect();
                for slot in broken_slots {
                    self.events.push(MountEvent::EquipmentBroken {
                        mount: id,
                        slot,
                    });
                }
            }
        }

        // Update stables.
        for stable in &mut self.stables {
            stable.update(dt, &mut self.mounts);
        }

        // Update breedings.
        let mut completed_breedings = Vec::new();
        for breeding in &mut self.active_breedings {
            breeding.update(game_time);
            if breeding.complete {
                completed_breedings.push(breeding.clone());
            }
        }
        self.active_breedings.retain(|b| !b.complete);

        // Create offspring from completed breedings.
        for breeding in completed_breedings {
            let parent_type = self.mounts.get(&breeding.parent_a)
                .map(|m| m.mount_type)
                .unwrap_or(MountType::Horse);

            let offspring_id = MountId(self.next_id);
            self.next_id += 1;

            let mut offspring = Mount::new(offspring_id, "Foal", parent_type);
            offspring.genetics = breeding.offspring_genetics.clone();
            offspring.recalculate_stats();
            offspring.taming.tamed = true;
            offspring.taming.trust_level = 0.8;

            if let Some(parent_a) = self.mounts.get(&breeding.parent_a) {
                if parent_a.genetics.special_trait == Some(SpecialTrait::GentleNature) {
                    offspring.taming.bond_level = 30.0;
                }
            }

            self.mounts.insert(offspring_id, offspring);
            self.events.push(MountEvent::BreedingComplete {
                parent_a: breeding.parent_a,
                parent_b: breeding.parent_b,
                offspring: offspring_id,
            });
        }
    }

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

            // Handle ability usage.
            if let Some(ability_type) = input.ability {
                if mount.use_ability(ability_type) {
                    self.events.push(MountEvent::AbilityUsed {
                        mount: mount_id,
                        ability: ability_type,
                    });
                }
            }

            let walk = input.forward > 0.1;
            let run = input.forward > 0.5;
            let gallop = input.sprint && run;
            mount.set_input(walk, run, gallop);
            mount.turn(input.turn * mount.stats.turn_rate, dt);
        }
    }

    pub fn events(&self) -> &[MountEvent] {
        &self.events
    }

    pub fn mount_count(&self) -> usize {
        self.mounts.len()
    }

    pub fn stable_count(&self) -> usize {
        self.stables.len()
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

    #[test]
    fn test_multi_rider() {
        let mut mount = Mount::new(MountId(1), "Dragon", MountType::Dragon);
        mount.taming.tamed = true;
        assert_eq!(mount.mount_points.len(), 2);

        let rider1 = RiderId(1);
        let rider2 = RiderId(2);

        assert!(mount.mount_rider(rider1).is_some());
        assert!(mount.mount_rider(rider2).is_some());
        assert_eq!(mount.all_riders().len(), 2);
    }

    #[test]
    fn test_mount_equipment() {
        let mut mount = Mount::new(MountId(1), "Roach", MountType::Horse);
        let saddle = MountEquipment {
            item_id: "iron_saddle".to_string(),
            name: "Iron Saddle".to_string(),
            slot: MountEquipmentSlot::Saddle,
            speed_modifier: 2.0,
            defense_modifier: 5.0,
            stamina_modifier: 10.0,
            carry_weight_bonus: 20.0,
            inventory_slots_bonus: 0,
            durability: 100.0,
            max_durability: 100.0,
            quality: EquipmentQuality::Uncommon,
        };

        let old_speed = mount.stats.gallop_speed;
        mount.equip(saddle);
        assert!(mount.stats.gallop_speed > old_speed);
    }

    #[test]
    fn test_breeding() {
        let mut rng = 42u64;
        let gen_a = MountGenetics::random(&mut rng);
        let gen_b = MountGenetics::random(&mut rng);
        let offspring = MountGenetics::breed(&gen_a, &gen_b, &mut rng);

        assert!(offspring.speed_gene >= 0.0 && offspring.speed_gene <= 1.0);
        assert!(offspring.stamina_gene >= 0.0 && offspring.stamina_gene <= 1.0);
    }

    #[test]
    fn test_ability() {
        let mut mount = Mount::new(MountId(1), "Roach", MountType::Horse);
        mount.taming.tamed = true;
        mount.is_mounted = true;

        let has_charge = mount.abilities.iter().any(|a| a.ability_type == MountAbilityType::Charge);
        assert!(has_charge);

        let used = mount.use_ability(MountAbilityType::Charge);
        assert!(used);
        assert!(mount.stats.stamina < DEFAULT_MOUNT_STAMINA);
    }

    #[test]
    fn test_stable() {
        let mut stable = Stable::new("Main Stable", 5, Vec3::ZERO);
        assert_eq!(stable.available_slots(), 5);

        assert!(stable.store_mount(MountId(1)));
        assert_eq!(stable.available_slots(), 4);
        assert!(stable.retrieve_mount(MountId(1)));
        assert_eq!(stable.available_slots(), 5);
    }

    #[test]
    fn test_fatigue() {
        let mut mount = Mount::new(MountId(1), "Roach", MountType::Horse);
        mount.taming.tamed = true;
        mount.is_mounted = true;

        // Simulate galloping for a while.
        mount.target_speed = mount.stats.gallop_speed;
        mount.current_speed = mount.stats.gallop_speed;
        for _ in 0..100 {
            mount.update(0.1);
        }
        assert!(mount.fatigue > 0.0);
    }

    #[test]
    fn test_mount_ai() {
        let mut ai = MountAI::new(Vec3::ZERO);
        let (dir, speed) = ai.update(10.0, Vec3::ZERO, true);
        // After enough idle time, should start wandering.
        let (dir2, speed2) = ai.update(0.1, Vec3::ZERO, true);
        // AI should have transitioned to some state.
        assert!(ai.state != MountAIState::Idle || speed2 == 0.0);
    }
}
