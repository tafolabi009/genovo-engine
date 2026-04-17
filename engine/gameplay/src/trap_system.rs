//! Trap and hazard system for environmental dangers.
//!
//! Provides:
//! - **Spike trap**: floor spikes that extend when triggered
//! - **Arrow trap**: wall-mounted arrow launchers
//! - **Pit trap**: concealed pits that entities fall into
//! - **Fire trap**: floor vents that shoot flame jets
//! - **Poison cloud**: area-of-effect toxic gas
//! - **Tripwire**: wire that triggers connected traps
//! - **Pressure plate**: weight-activated trigger
//! - **Swinging blade**: pendulum blades on a timer or trigger
//! - **Falling rocks**: ceiling collapse hazard
//! - **Electric fence**: electrified barrier
//! - **Environmental hazards**: lava, quicksand, toxic water
//! - **Trap placement and triggering**: builder API and trigger logic
//! - **ECS integration**: `TrapComponent`, `TrapSystem`
//!
//! # Design
//!
//! Each trap has a [`TrapTrigger`] (how it activates), a [`TrapEffect`] (what
//! damage/status it applies), and a [`TrapState`] (armed, triggered, cooldown,
//! destroyed). The [`TrapManager`] updates all traps and generates
//! [`TrapEvent`]s for the game to handle.

use glam::Vec3;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of traps in the world.
pub const MAX_TRAPS: usize = 1024;
/// Default trap activation delay (seconds).
pub const DEFAULT_ACTIVATION_DELAY: f32 = 0.2;
/// Default trap cooldown after triggering (seconds).
pub const DEFAULT_COOLDOWN: f32 = 5.0;
/// Default pressure plate weight threshold (kg).
pub const DEFAULT_WEIGHT_THRESHOLD: f32 = 30.0;
/// Default tripwire break force.
pub const DEFAULT_TRIPWIRE_BREAK_FORCE: f32 = 10.0;
/// Default arrow trap projectile speed (m/s).
pub const ARROW_TRAP_SPEED: f32 = 30.0;
/// Default fire trap range (meters).
pub const FIRE_TRAP_RANGE: f32 = 5.0;
/// Default poison cloud radius (meters).
pub const POISON_CLOUD_RADIUS: f32 = 4.0;
/// Default poison cloud duration (seconds).
pub const POISON_CLOUD_DURATION: f32 = 8.0;
/// Default swinging blade period (seconds).
pub const BLADE_SWING_PERIOD: f32 = 2.0;
/// Default electric fence damage per second.
pub const ELECTRIC_FENCE_DPS: f32 = 15.0;
/// Default spike damage.
pub const SPIKE_DAMAGE: f32 = 40.0;
/// Default pit trap depth.
pub const PIT_DEPTH: f32 = 5.0;
/// Small epsilon.
const EPSILON: f32 = 1e-6;
/// Pi constant.
const PI: f32 = std::f32::consts::PI;

// ---------------------------------------------------------------------------
// TrapId
// ---------------------------------------------------------------------------

/// Unique identifier for a trap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TrapId(pub u64);

// ---------------------------------------------------------------------------
// TrapType
// ---------------------------------------------------------------------------

/// Type of trap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrapType {
    /// Floor spikes.
    SpikeTrap,
    /// Wall-mounted arrow launcher.
    ArrowTrap,
    /// Concealed pit.
    PitTrap,
    /// Flame jet from floor.
    FireTrap,
    /// Area-of-effect toxic gas.
    PoisonCloud,
    /// Wire that triggers connected traps.
    Tripwire,
    /// Weight-activated trigger plate.
    PressurePlate,
    /// Pendulum blade hazard.
    SwingingBlade,
    /// Ceiling collapse.
    FallingRocks,
    /// Electrified barrier.
    ElectricFence,
    /// Lava pool.
    Lava,
    /// Quicksand patch.
    Quicksand,
    /// Toxic water.
    ToxicWater,
    /// Bear trap.
    BearTrap,
    /// Explosive mine.
    ExplosiveMine,
    /// Custom trap (game-defined).
    Custom(u32),
}

impl TrapType {
    /// Get a display name for this trap type.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::SpikeTrap => "Spike Trap",
            Self::ArrowTrap => "Arrow Trap",
            Self::PitTrap => "Pit Trap",
            Self::FireTrap => "Fire Trap",
            Self::PoisonCloud => "Poison Cloud",
            Self::Tripwire => "Tripwire",
            Self::PressurePlate => "Pressure Plate",
            Self::SwingingBlade => "Swinging Blade",
            Self::FallingRocks => "Falling Rocks",
            Self::ElectricFence => "Electric Fence",
            Self::Lava => "Lava",
            Self::Quicksand => "Quicksand",
            Self::ToxicWater => "Toxic Water",
            Self::BearTrap => "Bear Trap",
            Self::ExplosiveMine => "Explosive Mine",
            Self::Custom(_) => "Custom Trap",
        }
    }

    /// Whether this is a one-shot trap (destroyed after triggering).
    pub fn is_one_shot(&self) -> bool {
        matches!(self, Self::FallingRocks | Self::ExplosiveMine | Self::PitTrap)
    }

    /// Whether this is a continuous hazard (always active).
    pub fn is_continuous(&self) -> bool {
        matches!(self, Self::Lava | Self::Quicksand | Self::ToxicWater | Self::ElectricFence)
    }
}

// ---------------------------------------------------------------------------
// TrapState
// ---------------------------------------------------------------------------

/// Current state of a trap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrapState {
    /// Trap is armed and waiting to trigger.
    Armed,
    /// Trap is in the process of activating (delay).
    Activating,
    /// Trap is actively dealing damage/effects.
    Active,
    /// Trap is on cooldown after triggering.
    Cooldown,
    /// Trap is disabled/deactivated.
    Disabled,
    /// Trap is destroyed (one-shot traps after use).
    Destroyed,
    /// Trap is being placed (not yet armed).
    Placing,
}

// ---------------------------------------------------------------------------
// TrapTrigger
// ---------------------------------------------------------------------------

/// How a trap is triggered.
#[derive(Debug, Clone)]
pub enum TrapTrigger {
    /// Triggered by proximity (entity enters radius).
    Proximity { radius: f32 },
    /// Triggered by weight on a pressure plate.
    PressurePlate { weight_threshold: f32, plate_size: Vec3 },
    /// Triggered by breaking a tripwire.
    Tripwire { start: Vec3, end: Vec3, break_force: f32 },
    /// Triggered on a timer (periodic).
    Timer { period: f32, duty_cycle: f32 },
    /// Triggered by a linked trap (chain reaction).
    Linked { source_trap: TrapId },
    /// Triggered manually (by game code or player).
    Manual,
    /// Triggered by line-of-sight (laser tripwire).
    LineOfSight { direction: Vec3, range: f32 },
    /// Triggered by sound (noise above threshold).
    SoundTrigger { threshold: f32, radius: f32 },
    /// Always active (environmental hazard).
    AlwaysActive,
}

// ---------------------------------------------------------------------------
// TrapEffect
// ---------------------------------------------------------------------------

/// The effect/damage a trap applies.
#[derive(Debug, Clone)]
pub struct TrapEffect {
    /// Damage type.
    pub damage_type: TrapDamageType,
    /// Base damage per hit or per second (for continuous).
    pub damage: f32,
    /// Area of effect radius (0 = single target).
    pub aoe_radius: f32,
    /// Duration of the effect (for fire, poison, etc.).
    pub duration: f32,
    /// Knockback force.
    pub knockback: f32,
    /// Knockback direction (None = away from trap center).
    pub knockback_direction: Option<Vec3>,
    /// Status effect to apply.
    pub status_effect: Option<TrapStatusEffect>,
    /// Whether the effect is damage-per-second or damage-per-hit.
    pub is_dps: bool,
    /// Root/stun duration.
    pub root_duration: f32,
    /// Slow multiplier (1.0 = no slow, 0.5 = half speed).
    pub slow_multiplier: f32,
    /// Slow duration.
    pub slow_duration: f32,
}

impl Default for TrapEffect {
    fn default() -> Self {
        Self {
            damage_type: TrapDamageType::Physical,
            damage: 20.0,
            aoe_radius: 0.0,
            duration: 0.0,
            knockback: 0.0,
            knockback_direction: None,
            status_effect: None,
            is_dps: false,
            root_duration: 0.0,
            slow_multiplier: 1.0,
            slow_duration: 0.0,
        }
    }
}

/// Damage type for traps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrapDamageType {
    Physical,
    Fire,
    Poison,
    Electric,
    Cold,
    Falling,
    Drowning,
    Explosive,
}

/// Status effect applied by a trap.
#[derive(Debug, Clone)]
pub struct TrapStatusEffect {
    /// Effect name.
    pub name: String,
    /// Duration in seconds.
    pub duration: f32,
    /// Damage per second (for DoTs).
    pub dps: f32,
    /// Effect type.
    pub effect_type: TrapDamageType,
}

// ---------------------------------------------------------------------------
// Trap
// ---------------------------------------------------------------------------

/// A trap or hazard in the world.
#[derive(Debug, Clone)]
pub struct Trap {
    /// Unique identifier.
    pub id: TrapId,
    /// Display name.
    pub name: String,
    /// Trap type.
    pub trap_type: TrapType,
    /// Current state.
    pub state: TrapState,
    /// Position in world space.
    pub position: Vec3,
    /// Rotation (for directional traps).
    pub forward: Vec3,
    /// Trigger mechanism.
    pub trigger: TrapTrigger,
    /// Effect when triggered.
    pub effect: TrapEffect,
    /// Activation delay (seconds after trigger before effect).
    pub activation_delay: f32,
    /// Active duration (seconds the trap stays active).
    pub active_duration: f32,
    /// Cooldown duration (seconds between activations).
    pub cooldown_duration: f32,
    /// Timer for current state transition.
    state_timer: f32,
    /// Whether this trap can be disarmed.
    pub disarmable: bool,
    /// Skill check difficulty to disarm (0..100).
    pub disarm_difficulty: u32,
    /// Whether the trap is visible (can be detected).
    pub visible: bool,
    /// Detection difficulty (lower = easier to spot).
    pub detection_difficulty: u32,
    /// Collision layer mask for filtering.
    pub collision_layers: u32,
    /// Entities currently inside the trap's effect area.
    pub entities_in_range: Vec<u64>,
    /// Linked traps (triggered when this trap triggers).
    pub linked_traps: Vec<TrapId>,
    /// Owner entity (for player-placed traps).
    pub owner: Option<u64>,
    /// Number of times this trap has been triggered.
    pub trigger_count: u32,
    /// Maximum number of triggers (0 = unlimited).
    pub max_triggers: u32,
    /// Tag for game code.
    pub tag: Option<String>,
    /// Visual model/mesh reference.
    pub model: String,
    /// Sound effect on trigger.
    pub trigger_sound: Option<String>,
    /// Particle effect on trigger.
    pub trigger_particles: Option<String>,
}

impl Trap {
    /// Create a new trap.
    pub fn new(id: TrapId, trap_type: TrapType, position: Vec3) -> Self {
        let (trigger, effect, active_dur) = match trap_type {
            TrapType::SpikeTrap => (
                TrapTrigger::Proximity { radius: 1.5 },
                TrapEffect {
                    damage_type: TrapDamageType::Physical,
                    damage: SPIKE_DAMAGE,
                    ..TrapEffect::default()
                },
                0.5,
            ),
            TrapType::ArrowTrap => (
                TrapTrigger::Proximity { radius: 2.0 },
                TrapEffect {
                    damage_type: TrapDamageType::Physical,
                    damage: 25.0,
                    ..TrapEffect::default()
                },
                0.1,
            ),
            TrapType::PitTrap => (
                TrapTrigger::PressurePlate {
                    weight_threshold: DEFAULT_WEIGHT_THRESHOLD,
                    plate_size: Vec3::new(2.0, 0.1, 2.0),
                },
                TrapEffect {
                    damage_type: TrapDamageType::Falling,
                    damage: 50.0,
                    root_duration: 3.0,
                    ..TrapEffect::default()
                },
                3.0,
            ),
            TrapType::FireTrap => (
                TrapTrigger::Proximity { radius: 2.0 },
                TrapEffect {
                    damage_type: TrapDamageType::Fire,
                    damage: 15.0,
                    aoe_radius: FIRE_TRAP_RANGE,
                    duration: 3.0,
                    is_dps: true,
                    status_effect: Some(TrapStatusEffect {
                        name: "Burning".to_string(),
                        duration: 5.0,
                        dps: 5.0,
                        effect_type: TrapDamageType::Fire,
                    }),
                    ..TrapEffect::default()
                },
                2.0,
            ),
            TrapType::PoisonCloud => (
                TrapTrigger::Proximity { radius: POISON_CLOUD_RADIUS },
                TrapEffect {
                    damage_type: TrapDamageType::Poison,
                    damage: 8.0,
                    aoe_radius: POISON_CLOUD_RADIUS,
                    duration: POISON_CLOUD_DURATION,
                    is_dps: true,
                    status_effect: Some(TrapStatusEffect {
                        name: "Poisoned".to_string(),
                        duration: 10.0,
                        dps: 3.0,
                        effect_type: TrapDamageType::Poison,
                    }),
                    slow_multiplier: 0.7,
                    slow_duration: 5.0,
                    ..TrapEffect::default()
                },
                POISON_CLOUD_DURATION,
            ),
            TrapType::Tripwire => (
                TrapTrigger::Tripwire {
                    start: position,
                    end: position + Vec3::new(3.0, 0.0, 0.0),
                    break_force: DEFAULT_TRIPWIRE_BREAK_FORCE,
                },
                TrapEffect::default(),
                0.0,
            ),
            TrapType::PressurePlate => (
                TrapTrigger::PressurePlate {
                    weight_threshold: DEFAULT_WEIGHT_THRESHOLD,
                    plate_size: Vec3::new(1.0, 0.05, 1.0),
                },
                TrapEffect::default(),
                0.0,
            ),
            TrapType::SwingingBlade => (
                TrapTrigger::Timer { period: BLADE_SWING_PERIOD, duty_cycle: 0.3 },
                TrapEffect {
                    damage_type: TrapDamageType::Physical,
                    damage: 35.0,
                    knockback: 5.0,
                    ..TrapEffect::default()
                },
                BLADE_SWING_PERIOD * 0.3,
            ),
            TrapType::FallingRocks => (
                TrapTrigger::Proximity { radius: 3.0 },
                TrapEffect {
                    damage_type: TrapDamageType::Physical,
                    damage: 60.0,
                    aoe_radius: 4.0,
                    knockback: 8.0,
                    root_duration: 2.0,
                    ..TrapEffect::default()
                },
                1.0,
            ),
            TrapType::ElectricFence => (
                TrapTrigger::AlwaysActive,
                TrapEffect {
                    damage_type: TrapDamageType::Electric,
                    damage: ELECTRIC_FENCE_DPS,
                    is_dps: true,
                    root_duration: 0.5,
                    ..TrapEffect::default()
                },
                f32::MAX,
            ),
            TrapType::Lava => (
                TrapTrigger::AlwaysActive,
                TrapEffect {
                    damage_type: TrapDamageType::Fire,
                    damage: 50.0,
                    is_dps: true,
                    status_effect: Some(TrapStatusEffect {
                        name: "Burning".to_string(),
                        duration: 10.0,
                        dps: 10.0,
                        effect_type: TrapDamageType::Fire,
                    }),
                    slow_multiplier: 0.3,
                    slow_duration: 2.0,
                    ..TrapEffect::default()
                },
                f32::MAX,
            ),
            TrapType::Quicksand => (
                TrapTrigger::AlwaysActive,
                TrapEffect {
                    damage_type: TrapDamageType::Drowning,
                    damage: 5.0,
                    is_dps: true,
                    slow_multiplier: 0.2,
                    slow_duration: 1.0,
                    root_duration: 0.0,
                    ..TrapEffect::default()
                },
                f32::MAX,
            ),
            TrapType::ToxicWater => (
                TrapTrigger::AlwaysActive,
                TrapEffect {
                    damage_type: TrapDamageType::Poison,
                    damage: 10.0,
                    is_dps: true,
                    status_effect: Some(TrapStatusEffect {
                        name: "Toxic".to_string(),
                        duration: 15.0,
                        dps: 3.0,
                        effect_type: TrapDamageType::Poison,
                    }),
                    ..TrapEffect::default()
                },
                f32::MAX,
            ),
            TrapType::BearTrap => (
                TrapTrigger::Proximity { radius: 0.5 },
                TrapEffect {
                    damage_type: TrapDamageType::Physical,
                    damage: 30.0,
                    root_duration: 5.0,
                    ..TrapEffect::default()
                },
                5.0,
            ),
            TrapType::ExplosiveMine => (
                TrapTrigger::Proximity { radius: 1.0 },
                TrapEffect {
                    damage_type: TrapDamageType::Explosive,
                    damage: 80.0,
                    aoe_radius: 5.0,
                    knockback: 15.0,
                    ..TrapEffect::default()
                },
                0.1,
            ),
            TrapType::Custom(_) => (
                TrapTrigger::Proximity { radius: 2.0 },
                TrapEffect::default(),
                1.0,
            ),
        };

        Self {
            id,
            name: trap_type.display_name().to_string(),
            trap_type,
            state: if trap_type.is_continuous() { TrapState::Active } else { TrapState::Armed },
            position,
            forward: Vec3::new(0.0, 0.0, 1.0),
            trigger,
            effect,
            activation_delay: DEFAULT_ACTIVATION_DELAY,
            active_duration: active_dur,
            cooldown_duration: DEFAULT_COOLDOWN,
            state_timer: 0.0,
            disarmable: !trap_type.is_continuous(),
            disarm_difficulty: 50,
            visible: trap_type.is_continuous(),
            detection_difficulty: 60,
            collision_layers: 0xFFFFFFFF,
            entities_in_range: Vec::new(),
            linked_traps: Vec::new(),
            owner: None,
            trigger_count: 0,
            max_triggers: if trap_type.is_one_shot() { 1 } else { 0 },
            tag: None,
            model: String::new(),
            trigger_sound: None,
            trigger_particles: None,
        }
    }

    /// Check if a position triggers this trap.
    pub fn check_trigger(&self, entity_pos: Vec3, entity_weight: f32) -> bool {
        if self.state != TrapState::Armed {
            return false;
        }
        match &self.trigger {
            TrapTrigger::Proximity { radius } => {
                (entity_pos - self.position).length() <= *radius
            }
            TrapTrigger::PressurePlate { weight_threshold, plate_size } => {
                let half = *plate_size * 0.5;
                let local = entity_pos - self.position;
                local.x.abs() <= half.x && local.z.abs() <= half.z
                    && local.y.abs() <= half.y
                    && entity_weight >= *weight_threshold
            }
            TrapTrigger::Tripwire { start, end, .. } => {
                let seg = *end - *start;
                let seg_len_sq = seg.length_squared();
                if seg_len_sq < EPSILON {
                    return false;
                }
                let t = ((entity_pos - *start).dot(seg) / seg_len_sq).clamp(0.0, 1.0);
                let closest = *start + seg * t;
                (entity_pos - closest).length() <= 0.3
            }
            TrapTrigger::Timer { .. } => false, // Handled by timer logic
            TrapTrigger::Linked { .. } => false, // Handled by link logic
            TrapTrigger::Manual => false,
            TrapTrigger::LineOfSight { direction, range } => {
                let to_entity = entity_pos - self.position;
                let dist = to_entity.length();
                if dist > *range {
                    return false;
                }
                let dot = to_entity.normalize_or_zero().dot(direction.normalize_or_zero());
                dot > 0.95
            }
            TrapTrigger::SoundTrigger { .. } => false,
            TrapTrigger::AlwaysActive => true,
        }
    }

    /// Activate (trigger) the trap.
    pub fn activate(&mut self) {
        if self.state != TrapState::Armed {
            return;
        }
        if self.max_triggers > 0 && self.trigger_count >= self.max_triggers {
            self.state = TrapState::Destroyed;
            return;
        }
        self.state = TrapState::Activating;
        self.state_timer = 0.0;
    }

    /// Manually trigger the trap.
    pub fn manual_trigger(&mut self) {
        if self.state == TrapState::Armed || self.state == TrapState::Disabled {
            self.state = TrapState::Armed;
            self.activate();
        }
    }

    /// Disarm the trap. Returns true if successful.
    pub fn disarm(&mut self) -> bool {
        if !self.disarmable || self.state == TrapState::Destroyed {
            return false;
        }
        self.state = TrapState::Disabled;
        true
    }

    /// Re-arm a disabled trap.
    pub fn rearm(&mut self) {
        if self.state == TrapState::Disabled {
            self.state = TrapState::Armed;
            self.state_timer = 0.0;
        }
    }

    /// Update the trap state.
    pub fn update(&mut self, dt: f32) -> Option<TrapEvent> {
        self.state_timer += dt;

        match self.state {
            TrapState::Activating => {
                if self.state_timer >= self.activation_delay {
                    self.state = TrapState::Active;
                    self.state_timer = 0.0;
                    self.trigger_count += 1;
                    return Some(TrapEvent::Triggered {
                        trap_id: self.id,
                        trap_type: self.trap_type,
                        position: self.position,
                    });
                }
            }
            TrapState::Active => {
                if self.state_timer >= self.active_duration {
                    if self.trap_type.is_one_shot() {
                        self.state = TrapState::Destroyed;
                        return Some(TrapEvent::Destroyed {
                            trap_id: self.id,
                            position: self.position,
                        });
                    } else if self.trap_type.is_continuous() {
                        // Stay active
                    } else {
                        self.state = TrapState::Cooldown;
                        self.state_timer = 0.0;
                    }
                }
            }
            TrapState::Cooldown => {
                if self.state_timer >= self.cooldown_duration {
                    self.state = TrapState::Armed;
                    self.state_timer = 0.0;
                    return Some(TrapEvent::Rearmed { trap_id: self.id });
                }
            }
            _ => {}
        }

        // Timer-based traps
        if let TrapTrigger::Timer { period, duty_cycle } = &self.trigger {
            if self.state == TrapState::Armed {
                let cycle = self.state_timer % period;
                if cycle < period * duty_cycle {
                    self.activate();
                }
            }
        }

        None
    }

    /// Check if this trap is dealing damage right now.
    pub fn is_dealing_damage(&self) -> bool {
        self.state == TrapState::Active
    }

    /// Get the damage this trap deals.
    pub fn get_damage(&self, dt: f32) -> f32 {
        if !self.is_dealing_damage() {
            return 0.0;
        }
        if self.effect.is_dps {
            self.effect.damage * dt
        } else {
            self.effect.damage
        }
    }
}

// ---------------------------------------------------------------------------
// TrapEvent
// ---------------------------------------------------------------------------

/// Events emitted by the trap system.
#[derive(Debug, Clone)]
pub enum TrapEvent {
    /// A trap was triggered.
    Triggered { trap_id: TrapId, trap_type: TrapType, position: Vec3 },
    /// An entity was damaged by a trap.
    EntityDamaged {
        trap_id: TrapId,
        entity_id: u64,
        damage: f32,
        damage_type: TrapDamageType,
    },
    /// An entity entered a trap's effect area.
    EntityEntered { trap_id: TrapId, entity_id: u64 },
    /// An entity left a trap's effect area.
    EntityExited { trap_id: TrapId, entity_id: u64 },
    /// A trap was disarmed.
    Disarmed { trap_id: TrapId },
    /// A trap was rearmed.
    Rearmed { trap_id: TrapId },
    /// A trap was destroyed.
    Destroyed { trap_id: TrapId, position: Vec3 },
    /// A chain reaction triggered another trap.
    ChainTriggered { source_trap: TrapId, target_trap: TrapId },
    /// A trap was detected by an entity.
    Detected { trap_id: TrapId, entity_id: u64 },
}

// ---------------------------------------------------------------------------
// TrapManager
// ---------------------------------------------------------------------------

/// Manages all traps in the world.
pub struct TrapManager {
    /// All traps indexed by ID.
    traps: HashMap<TrapId, Trap>,
    /// Next trap ID.
    next_id: u64,
    /// Events from last update.
    events: Vec<TrapEvent>,
}

impl TrapManager {
    /// Create a new trap manager.
    pub fn new() -> Self {
        Self {
            traps: HashMap::new(),
            next_id: 1,
            events: Vec::new(),
        }
    }

    /// Add a trap and return its ID.
    pub fn add_trap(&mut self, mut trap: Trap) -> TrapId {
        let id = TrapId(self.next_id);
        self.next_id += 1;
        trap.id = id;
        self.traps.insert(id, trap);
        id
    }

    /// Remove a trap.
    pub fn remove_trap(&mut self, id: TrapId) -> Option<Trap> {
        self.traps.remove(&id)
    }

    /// Get a trap by ID.
    pub fn get_trap(&self, id: TrapId) -> Option<&Trap> {
        self.traps.get(&id)
    }

    /// Get a trap mutably.
    pub fn get_trap_mut(&mut self, id: TrapId) -> Option<&mut Trap> {
        self.traps.get_mut(&id)
    }

    /// Link two traps (triggering one triggers the other).
    pub fn link_traps(&mut self, source: TrapId, target: TrapId) {
        if let Some(trap) = self.traps.get_mut(&source) {
            if !trap.linked_traps.contains(&target) {
                trap.linked_traps.push(target);
            }
        }
    }

    /// Update all traps and check for entity interactions.
    pub fn update(&mut self, dt: f32, entities: &[(u64, Vec3, f32)]) {
        self.events.clear();

        let trap_ids: Vec<TrapId> = self.traps.keys().copied().collect();
        let mut chain_triggers: Vec<TrapId> = Vec::new();

        for id in &trap_ids {
            if let Some(trap) = self.traps.get_mut(id) {
                // Check entity triggers
                for &(entity_id, entity_pos, entity_weight) in entities {
                    if trap.check_trigger(entity_pos, entity_weight) {
                        trap.activate();

                        // Track chain triggers
                        for linked in &trap.linked_traps {
                            chain_triggers.push(*linked);
                            self.events.push(TrapEvent::ChainTriggered {
                                source_trap: *id,
                                target_trap: *linked,
                            });
                        }
                    }

                    // Check for damage
                    if trap.is_dealing_damage() {
                        let in_range = match &trap.trigger {
                            TrapTrigger::Proximity { radius } => {
                                (entity_pos - trap.position).length() <= *radius
                            }
                            _ => {
                                if trap.effect.aoe_radius > 0.0 {
                                    (entity_pos - trap.position).length() <= trap.effect.aoe_radius
                                } else {
                                    (entity_pos - trap.position).length() <= 1.5
                                }
                            }
                        };

                        if in_range {
                            let damage = trap.get_damage(dt);
                            if damage > 0.0 {
                                self.events.push(TrapEvent::EntityDamaged {
                                    trap_id: *id,
                                    entity_id,
                                    damage,
                                    damage_type: trap.effect.damage_type,
                                });
                            }
                        }
                    }
                }

                // Update trap state
                if let Some(event) = trap.update(dt) {
                    self.events.push(event);
                }
            }
        }

        // Process chain triggers
        for chain_id in chain_triggers {
            if let Some(trap) = self.traps.get_mut(&chain_id) {
                trap.activate();
            }
        }
    }

    /// Get events from last update.
    pub fn events(&self) -> &[TrapEvent] {
        &self.events
    }

    /// Get the number of active traps.
    pub fn active_trap_count(&self) -> usize {
        self.traps.values()
            .filter(|t| !matches!(t.state, TrapState::Destroyed | TrapState::Disabled))
            .count()
    }

    /// Get all traps of a specific type.
    pub fn traps_of_type(&self, trap_type: TrapType) -> Vec<TrapId> {
        self.traps.values()
            .filter(|t| t.trap_type == trap_type)
            .map(|t| t.id)
            .collect()
    }

    /// Get trap statistics.
    pub fn stats(&self) -> TrapStats {
        TrapStats {
            total_traps: self.traps.len(),
            armed_traps: self.traps.values().filter(|t| t.state == TrapState::Armed).count(),
            active_traps: self.traps.values().filter(|t| t.state == TrapState::Active).count(),
            destroyed_traps: self.traps.values().filter(|t| t.state == TrapState::Destroyed).count(),
            disabled_traps: self.traps.values().filter(|t| t.state == TrapState::Disabled).count(),
        }
    }
}

/// Trap system statistics.
#[derive(Debug, Clone)]
pub struct TrapStats {
    pub total_traps: usize,
    pub armed_traps: usize,
    pub active_traps: usize,
    pub destroyed_traps: usize,
    pub disabled_traps: usize,
}

// ---------------------------------------------------------------------------
// TrapComponent (ECS)
// ---------------------------------------------------------------------------

/// ECS component for trap entities.
#[derive(Debug, Clone)]
pub struct TrapComponent {
    /// Trap ID in the trap manager.
    pub trap_id: TrapId,
    /// Whether the trap is player-placed.
    pub player_placed: bool,
    /// Whether this trap is detected by the player.
    pub detected: bool,
}

impl TrapComponent {
    /// Create a new trap component.
    pub fn new(trap_id: TrapId) -> Self {
        Self {
            trap_id,
            player_placed: false,
            detected: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_trap_creation() {
        let trap = Trap::new(TrapId(1), TrapType::SpikeTrap, Vec3::ZERO);
        assert_eq!(trap.state, TrapState::Armed);
        assert_eq!(trap.trap_type, TrapType::SpikeTrap);
    }

    #[test]
    fn test_trap_trigger() {
        let trap = Trap::new(TrapId(1), TrapType::SpikeTrap, Vec3::ZERO);
        assert!(trap.check_trigger(Vec3::new(0.5, 0.0, 0.0), 70.0));
        assert!(!trap.check_trigger(Vec3::new(10.0, 0.0, 0.0), 70.0));
    }

    #[test]
    fn test_trap_activation_flow() {
        let mut trap = Trap::new(TrapId(1), TrapType::SpikeTrap, Vec3::ZERO);
        trap.activate();
        assert_eq!(trap.state, TrapState::Activating);

        // Simulate time passing activation delay
        trap.state_timer = trap.activation_delay + EPSILON;
        let event = trap.update(0.0);
        assert!(event.is_some());
        assert_eq!(trap.state, TrapState::Active);
    }

    #[test]
    fn test_trap_disarm() {
        let mut trap = Trap::new(TrapId(1), TrapType::SpikeTrap, Vec3::ZERO);
        assert!(trap.disarm());
        assert_eq!(trap.state, TrapState::Disabled);
    }

    #[test]
    fn test_continuous_hazard() {
        let trap = Trap::new(TrapId(1), TrapType::Lava, Vec3::ZERO);
        assert_eq!(trap.state, TrapState::Active);
        assert!(trap.is_dealing_damage());
    }

    #[test]
    fn test_one_shot_trap() {
        assert!(TrapType::ExplosiveMine.is_one_shot());
        assert!(!TrapType::SpikeTrap.is_one_shot());
    }
}
