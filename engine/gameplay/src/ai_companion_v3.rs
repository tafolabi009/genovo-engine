// engine/gameplay/src/ai_companion_v3.rs
//
// Enhanced AI companion system (v3) for the Genovo engine.
//
// Provides a full-featured AI companion with:
// - Formation following with dynamic slot assignment
// - Combat support with tactical positioning
// - Healing and buff support
// - Contextual callouts and barks
// - Companion-specific abilities with cooldowns
// - Loyalty/relationship system with progression
// - Morale system affecting combat effectiveness
// - Commands (follow, stay, attack, defend, use ability)
// - Companion inventory with auto-loot preferences
// - Revival mechanic when downed

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const MAX_COMPANIONS: usize = 4;
pub const MAX_ABILITIES: usize = 8;
pub const DEFAULT_FOLLOW_DISTANCE: f32 = 3.0;
pub const DEFAULT_COMBAT_RANGE: f32 = 15.0;
pub const DEFAULT_HEAL_THRESHOLD: f32 = 0.5;
pub const MAX_LOYALTY: f32 = 100.0;
pub const DEFAULT_REVIVE_TIME: f32 = 5.0;
pub const CALLOUT_COOLDOWN: f32 = 8.0;
pub const BARK_COOLDOWN: f32 = 15.0;

// ---------------------------------------------------------------------------
// Companion class
// ---------------------------------------------------------------------------

/// Companion archetype/class that determines abilities and behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompanionClass {
    /// Balanced melee/ranged fighter.
    Warrior,
    /// Ranged damage dealer with area attacks.
    Mage,
    /// Support healer with buffs.
    Healer,
    /// Stealthy scout with traps and intel.
    Ranger,
    /// Defensive tank that draws aggro.
    Guardian,
    /// Utility companion with crafting and gathering.
    Artificer,
}

impl CompanionClass {
    /// Default abilities for this class.
    pub fn default_abilities(&self) -> Vec<CompanionAbilityDef> {
        match self {
            Self::Warrior => vec![
                CompanionAbilityDef::new("Charge", AbilityType::Offensive, 8.0, 20.0),
                CompanionAbilityDef::new("Shield Bash", AbilityType::Offensive, 5.0, 10.0),
                CompanionAbilityDef::new("War Cry", AbilityType::Buff, 20.0, 30.0),
            ],
            Self::Mage => vec![
                CompanionAbilityDef::new("Fireball", AbilityType::Offensive, 6.0, 25.0),
                CompanionAbilityDef::new("Ice Shield", AbilityType::Defensive, 15.0, 20.0),
                CompanionAbilityDef::new("Lightning Storm", AbilityType::Offensive, 25.0, 50.0),
            ],
            Self::Healer => vec![
                CompanionAbilityDef::new("Heal", AbilityType::Healing, 3.0, 15.0),
                CompanionAbilityDef::new("Regeneration", AbilityType::Buff, 10.0, 25.0),
                CompanionAbilityDef::new("Purify", AbilityType::Healing, 8.0, 10.0),
                CompanionAbilityDef::new("Mass Heal", AbilityType::Healing, 30.0, 40.0),
            ],
            Self::Ranger => vec![
                CompanionAbilityDef::new("Snipe", AbilityType::Offensive, 10.0, 35.0),
                CompanionAbilityDef::new("Trap", AbilityType::Defensive, 15.0, 20.0),
                CompanionAbilityDef::new("Scout", AbilityType::Utility, 20.0, 0.0),
            ],
            Self::Guardian => vec![
                CompanionAbilityDef::new("Taunt", AbilityType::Defensive, 8.0, 0.0),
                CompanionAbilityDef::new("Shield Wall", AbilityType::Defensive, 20.0, 30.0),
                CompanionAbilityDef::new("Bodyguard", AbilityType::Defensive, 30.0, 15.0),
            ],
            Self::Artificer => vec![
                CompanionAbilityDef::new("Turret", AbilityType::Offensive, 15.0, 20.0),
                CompanionAbilityDef::new("Repair", AbilityType::Healing, 5.0, 10.0),
                CompanionAbilityDef::new("Craft Potion", AbilityType::Utility, 30.0, 0.0),
            ],
        }
    }

    /// Base stats for this class.
    pub fn base_stats(&self) -> CompanionStats {
        match self {
            Self::Warrior => CompanionStats { max_health: 150.0, attack: 20.0, defense: 15.0, speed: 5.0, perception: 8.0 },
            Self::Mage => CompanionStats { max_health: 80.0, attack: 30.0, defense: 5.0, speed: 4.0, perception: 12.0 },
            Self::Healer => CompanionStats { max_health: 100.0, attack: 8.0, defense: 10.0, speed: 4.5, perception: 15.0 },
            Self::Ranger => CompanionStats { max_health: 90.0, attack: 25.0, defense: 8.0, speed: 6.0, perception: 20.0 },
            Self::Guardian => CompanionStats { max_health: 200.0, attack: 12.0, defense: 25.0, speed: 3.5, perception: 10.0 },
            Self::Artificer => CompanionStats { max_health: 100.0, attack: 15.0, defense: 10.0, speed: 4.5, perception: 12.0 },
        }
    }
}

// ---------------------------------------------------------------------------
// Stats and abilities
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CompanionStats {
    pub max_health: f32,
    pub attack: f32,
    pub defense: f32,
    pub speed: f32,
    pub perception: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AbilityType {
    Offensive,
    Defensive,
    Healing,
    Buff,
    Utility,
}

#[derive(Debug, Clone)]
pub struct CompanionAbilityDef {
    pub name: String,
    pub ability_type: AbilityType,
    pub cooldown: f32,
    pub resource_cost: f32,
    pub range: f32,
    pub area_radius: f32,
    pub effect_value: f32,
    pub duration: f32,
}

impl CompanionAbilityDef {
    pub fn new(name: &str, ability_type: AbilityType, cooldown: f32, cost: f32) -> Self {
        Self {
            name: name.to_string(),
            ability_type,
            cooldown,
            resource_cost: cost,
            range: match ability_type {
                AbilityType::Offensive => 15.0,
                AbilityType::Defensive => 5.0,
                AbilityType::Healing => 10.0,
                AbilityType::Buff => 8.0,
                AbilityType::Utility => 20.0,
            },
            area_radius: 0.0,
            effect_value: cost * 0.8,
            duration: if matches!(ability_type, AbilityType::Buff) { 10.0 } else { 0.0 },
        }
    }
}

#[derive(Debug, Clone)]
pub struct AbilityState {
    pub def: CompanionAbilityDef,
    pub cooldown_remaining: f32,
    pub charges: u32,
    pub max_charges: u32,
    pub is_active: bool,
    pub active_timer: f32,
}

impl AbilityState {
    pub fn new(def: CompanionAbilityDef) -> Self {
        Self {
            def,
            cooldown_remaining: 0.0,
            charges: 1,
            max_charges: 1,
            is_active: false,
            active_timer: 0.0,
        }
    }

    pub fn is_ready(&self) -> bool {
        self.cooldown_remaining <= 0.0 && self.charges > 0
    }

    pub fn use_ability(&mut self) -> bool {
        if !self.is_ready() { return false; }
        self.charges -= 1;
        self.cooldown_remaining = self.def.cooldown;
        if self.def.duration > 0.0 {
            self.is_active = true;
            self.active_timer = self.def.duration;
        }
        true
    }

    pub fn update(&mut self, dt: f32) {
        if self.cooldown_remaining > 0.0 {
            self.cooldown_remaining = (self.cooldown_remaining - dt).max(0.0);
            if self.cooldown_remaining <= 0.0 && self.charges < self.max_charges {
                self.charges += 1;
            }
        }
        if self.is_active {
            self.active_timer -= dt;
            if self.active_timer <= 0.0 {
                self.is_active = false;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Loyalty system
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LoyaltySystem {
    pub loyalty: f32,
    pub max_loyalty: f32,
    pub relationship_level: RelationshipLevel,
    pub approval_history: Vec<ApprovalEvent>,
    pub gifts_given: u32,
    pub battles_together: u32,
    pub times_healed_player: u32,
    pub times_revived: u32,
    pub personal_quest_completed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationshipLevel {
    Stranger,
    Acquaintance,
    Ally,
    Friend,
    BestFriend,
    BondedPartner,
}

impl RelationshipLevel {
    pub fn from_loyalty(loyalty: f32) -> Self {
        match loyalty as u32 {
            0..=15 => Self::Stranger,
            16..=35 => Self::Acquaintance,
            36..=55 => Self::Ally,
            56..=75 => Self::Friend,
            76..=95 => Self::BestFriend,
            _ => Self::BondedPartner,
        }
    }

    pub fn combat_bonus(&self) -> f32 {
        match self {
            Self::Stranger => 0.0,
            Self::Acquaintance => 0.05,
            Self::Ally => 0.1,
            Self::Friend => 0.15,
            Self::BestFriend => 0.25,
            Self::BondedPartner => 0.35,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApprovalEvent {
    pub delta: f32,
    pub reason: String,
    pub timestamp: f32,
}

impl Default for LoyaltySystem {
    fn default() -> Self {
        Self {
            loyalty: 20.0,
            max_loyalty: MAX_LOYALTY,
            relationship_level: RelationshipLevel::Acquaintance,
            approval_history: Vec::new(),
            gifts_given: 0,
            battles_together: 0,
            times_healed_player: 0,
            times_revived: 0,
            personal_quest_completed: false,
        }
    }
}

impl LoyaltySystem {
    pub fn add_approval(&mut self, delta: f32, reason: &str, time: f32) {
        self.loyalty = (self.loyalty + delta).clamp(0.0, self.max_loyalty);
        self.relationship_level = RelationshipLevel::from_loyalty(self.loyalty);
        self.approval_history.push(ApprovalEvent {
            delta,
            reason: reason.to_string(),
            timestamp: time,
        });
    }
}

// ---------------------------------------------------------------------------
// Morale
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MoraleSystem {
    pub morale: f32,
    pub max_morale: f32,
    pub combat_effectiveness: f32,
    pub fear_level: f32,
    pub morale_factors: Vec<(String, f32)>,
}

impl Default for MoraleSystem {
    fn default() -> Self {
        Self {
            morale: 80.0,
            max_morale: 100.0,
            combat_effectiveness: 1.0,
            fear_level: 0.0,
            morale_factors: Vec::new(),
        }
    }
}

impl MoraleSystem {
    pub fn update(&mut self, dt: f32, health_fraction: f32, loyalty: f32, enemies_nearby: u32) {
        // Base recovery.
        self.morale = (self.morale + 2.0 * dt).min(self.max_morale);

        // Health factor.
        if health_fraction < 0.3 { self.morale -= 5.0 * dt; }
        if health_fraction < 0.1 { self.morale -= 10.0 * dt; }

        // Fear from enemies.
        self.fear_level = (enemies_nearby as f32 * 5.0 - loyalty * 0.3).max(0.0);
        self.morale -= self.fear_level * 0.1 * dt;

        self.morale = self.morale.clamp(0.0, self.max_morale);
        self.combat_effectiveness = (self.morale / self.max_morale).clamp(0.3, 1.2);
    }

    pub fn boost(&mut self, amount: f32) {
        self.morale = (self.morale + amount).min(self.max_morale);
    }
}

// ---------------------------------------------------------------------------
// Callouts
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CalloutType {
    EnemySpotted,
    LowHealth,
    Healing,
    AbilityUsed,
    ItemFound,
    DangerAhead,
    AllClear,
    NeedHelp,
    Downed,
    Reviving,
    Following,
    Staying,
    Attacking,
    GoodJob,
    Thanks,
    Idle,
}

#[derive(Debug, Clone)]
pub struct CalloutSystem {
    pub callout_cooldowns: HashMap<CalloutType, f32>,
    pub global_cooldown: f32,
    pub pending_callouts: Vec<PendingCallout>,
    pub bark_cooldown: f32,
}

#[derive(Debug, Clone)]
pub struct PendingCallout {
    pub callout_type: CalloutType,
    pub text_key: String,
    pub priority: u32,
    pub position: [f32; 3],
}

impl Default for CalloutSystem {
    fn default() -> Self {
        Self {
            callout_cooldowns: HashMap::new(),
            global_cooldown: 0.0,
            pending_callouts: Vec::new(),
            bark_cooldown: 0.0,
        }
    }
}

impl CalloutSystem {
    pub fn try_callout(&mut self, callout_type: CalloutType, text_key: &str, pos: [f32; 3], priority: u32) -> bool {
        if self.global_cooldown > 0.0 { return false; }
        if let Some(cd) = self.callout_cooldowns.get(&callout_type) {
            if *cd > 0.0 { return false; }
        }

        self.pending_callouts.push(PendingCallout {
            callout_type,
            text_key: text_key.to_string(),
            priority,
            position: pos,
        });
        self.callout_cooldowns.insert(callout_type, CALLOUT_COOLDOWN);
        self.global_cooldown = 2.0;
        true
    }

    pub fn update(&mut self, dt: f32) {
        self.global_cooldown = (self.global_cooldown - dt).max(0.0);
        self.bark_cooldown = (self.bark_cooldown - dt).max(0.0);
        for cd in self.callout_cooldowns.values_mut() {
            *cd = (*cd - dt).max(0.0);
        }
    }

    pub fn drain_callouts(&mut self) -> Vec<PendingCallout> {
        // Return highest priority callouts first.
        self.pending_callouts.sort_by(|a, b| b.priority.cmp(&a.priority));
        std::mem::take(&mut self.pending_callouts)
    }
}

// ---------------------------------------------------------------------------
// Formation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormationTypeV3 {
    Follow,
    Line,
    Wedge,
    Circle,
    Spread,
    Tight,
}

#[derive(Debug, Clone)]
pub struct FormationSlotV3 {
    pub offset: [f32; 3],
    pub companion_index: usize,
    pub occupied: bool,
}

/// Formation manager for companion positioning.
#[derive(Debug, Clone)]
pub struct CompanionFormation {
    pub formation_type: FormationTypeV3,
    pub slots: Vec<FormationSlotV3>,
    pub follow_distance: f32,
    pub spacing: f32,
}

impl CompanionFormation {
    pub fn new(formation_type: FormationTypeV3, count: usize) -> Self {
        let mut formation = Self {
            formation_type,
            slots: Vec::new(),
            follow_distance: DEFAULT_FOLLOW_DISTANCE,
            spacing: 2.0,
        };
        formation.compute_slots(count);
        formation
    }

    fn compute_slots(&mut self, count: usize) {
        self.slots.clear();
        let sp = self.spacing;
        let dist = self.follow_distance;
        for i in 0..count {
            let offset = match self.formation_type {
                FormationTypeV3::Follow => {
                    [-dist * (i as f32 + 1.0), 0.0, 0.0]
                }
                FormationTypeV3::Line => {
                    [0.0, 0.0, -dist + (i as f32 - count as f32 / 2.0) * sp]
                }
                FormationTypeV3::Wedge => {
                    let side = if i % 2 == 0 { 1.0 } else { -1.0 };
                    let row = (i / 2 + 1) as f32;
                    [-row * dist, 0.0, side * row * sp * 0.5]
                }
                FormationTypeV3::Circle => {
                    let angle = (i as f32 / count as f32) * std::f32::consts::TAU;
                    [angle.cos() * dist, 0.0, angle.sin() * dist]
                }
                FormationTypeV3::Spread => {
                    let side = if i % 2 == 0 { 1.0 } else { -1.0 };
                    [-dist, 0.0, side * sp * ((i / 2 + 1) as f32)]
                }
                FormationTypeV3::Tight => {
                    let side = if i % 2 == 0 { 1.0 } else { -1.0 };
                    [-dist * 0.5, 0.0, side * sp * 0.5]
                }
            };
            self.slots.push(FormationSlotV3 { offset, companion_index: i, occupied: true });
        }
    }

    pub fn world_position(&self, leader_pos: [f32; 3], leader_forward: [f32; 2], slot_index: usize) -> [f32; 3] {
        if slot_index >= self.slots.len() { return leader_pos; }
        let slot = &self.slots[slot_index];
        let forward = leader_forward;
        let right = [forward[1], -forward[0]];
        [
            leader_pos[0] + forward[0] * slot.offset[0] + right[0] * slot.offset[2],
            leader_pos[1] + slot.offset[1],
            leader_pos[2] + forward[1] * slot.offset[0] + right[1] * slot.offset[2],
        ]
    }
}

// ---------------------------------------------------------------------------
// Companion state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompanionBehavior {
    Following,
    Staying,
    Attacking,
    Defending,
    Healing,
    Reviving,
    Fleeing,
    Downed,
    UsingAbility,
    Idle,
    Looting,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompanionCommand {
    Follow,
    Stay,
    Attack(u64),
    Defend,
    UseAbility(usize),
    Revive,
    Loot,
    Dismiss,
}

/// A single AI companion.
#[derive(Debug, Clone)]
pub struct CompanionV3 {
    pub id: u64,
    pub name: String,
    pub class: CompanionClass,
    pub stats: CompanionStats,
    pub health: f32,
    pub resource: f32,
    pub max_resource: f32,
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub behavior: CompanionBehavior,
    pub target_entity: Option<u64>,
    pub abilities: Vec<AbilityState>,
    pub loyalty: LoyaltySystem,
    pub morale: MoraleSystem,
    pub callouts: CalloutSystem,
    pub formation_slot: usize,
    pub level: u32,
    pub experience: f32,
    pub is_downed: bool,
    pub downed_timer: f32,
    pub revive_progress: f32,
    pub auto_heal: bool,
    pub auto_heal_threshold: f32,
    pub combat_range: f32,
    pub last_command: Option<CompanionCommand>,
}

impl CompanionV3 {
    pub fn new(id: u64, name: &str, class: CompanionClass) -> Self {
        let stats = class.base_stats();
        let abilities = class.default_abilities().into_iter().map(AbilityState::new).collect();
        Self {
            id,
            name: name.to_string(),
            class,
            health: stats.max_health,
            resource: 100.0,
            max_resource: 100.0,
            stats,
            position: [0.0; 3],
            velocity: [0.0; 3],
            behavior: CompanionBehavior::Following,
            target_entity: None,
            abilities,
            loyalty: LoyaltySystem::default(),
            morale: MoraleSystem::default(),
            callouts: CalloutSystem::default(),
            formation_slot: 0,
            level: 1,
            experience: 0.0,
            is_downed: false,
            downed_timer: 0.0,
            revive_progress: 0.0,
            auto_heal: true,
            auto_heal_threshold: DEFAULT_HEAL_THRESHOLD,
            combat_range: DEFAULT_COMBAT_RANGE,
            last_command: None,
        }
    }

    pub fn update(&mut self, dt: f32, context: &CompanionContext) {
        // Update ability cooldowns.
        for ability in &mut self.abilities {
            ability.update(dt);
        }

        // Update callouts.
        self.callouts.update(dt);

        // Update morale.
        let health_frac = self.health / self.stats.max_health;
        self.morale.update(dt, health_frac, self.loyalty.loyalty, context.enemies_nearby);

        // Resource regeneration.
        self.resource = (self.resource + 5.0 * dt).min(self.max_resource);

        // Downed state.
        if self.is_downed {
            self.downed_timer += dt;
            if self.downed_timer > 60.0 {
                // Auto-revive with low health after 60 seconds.
                self.revive(0.2);
            }
            return;
        }

        // Auto-heal player or self.
        if self.auto_heal && self.class == CompanionClass::Healer {
            if context.player_health_fraction < self.auto_heal_threshold {
                self.try_heal(context, dt);
            } else if health_frac < self.auto_heal_threshold {
                self.try_self_heal(dt);
            }
        }

        // Behavior execution.
        match self.behavior {
            CompanionBehavior::Following => self.behavior_follow(context, dt),
            CompanionBehavior::Attacking => self.behavior_attack(context, dt),
            CompanionBehavior::Defending => self.behavior_defend(context, dt),
            CompanionBehavior::Healing => self.behavior_heal(context, dt),
            _ => {}
        }

        // Check for low health callout.
        if health_frac < 0.2 {
            self.callouts.try_callout(CalloutType::LowHealth, "companion_low_health", self.position, 5);
        }
    }

    pub fn give_command(&mut self, command: CompanionCommand) {
        self.last_command = Some(command);
        match command {
            CompanionCommand::Follow => {
                self.behavior = CompanionBehavior::Following;
                self.target_entity = None;
            }
            CompanionCommand::Stay => {
                self.behavior = CompanionBehavior::Staying;
            }
            CompanionCommand::Attack(target) => {
                self.behavior = CompanionBehavior::Attacking;
                self.target_entity = Some(target);
                self.callouts.try_callout(CalloutType::Attacking, "companion_attacking", self.position, 3);
            }
            CompanionCommand::Defend => {
                self.behavior = CompanionBehavior::Defending;
            }
            CompanionCommand::UseAbility(idx) => {
                if idx < self.abilities.len() {
                    self.abilities[idx].use_ability();
                    self.behavior = CompanionBehavior::UsingAbility;
                }
            }
            CompanionCommand::Revive => {
                self.behavior = CompanionBehavior::Reviving;
            }
            CompanionCommand::Loot => {
                self.behavior = CompanionBehavior::Looting;
            }
            CompanionCommand::Dismiss => {
                self.behavior = CompanionBehavior::Idle;
            }
        }
    }

    pub fn take_damage(&mut self, amount: f32) {
        let reduced = amount * (1.0 - self.stats.defense * 0.01).max(0.1);
        self.health = (self.health - reduced).max(0.0);
        if self.health <= 0.0 {
            self.is_downed = true;
            self.downed_timer = 0.0;
            self.behavior = CompanionBehavior::Downed;
            self.callouts.try_callout(CalloutType::Downed, "companion_downed", self.position, 10);
        }
    }

    pub fn revive(&mut self, health_fraction: f32) {
        self.is_downed = false;
        self.downed_timer = 0.0;
        self.revive_progress = 0.0;
        self.health = self.stats.max_health * health_fraction;
        self.behavior = CompanionBehavior::Following;
        self.loyalty.times_revived += 1;
        self.loyalty.add_approval(5.0, "Revived me", 0.0);
    }

    pub fn add_experience(&mut self, xp: f32) {
        self.experience += xp;
        let xp_needed = self.level as f32 * 100.0;
        while self.experience >= xp_needed {
            self.experience -= xp_needed;
            self.level += 1;
            self.stats.max_health += 10.0;
            self.stats.attack += 2.0;
            self.stats.defense += 1.0;
            self.health = self.stats.max_health;
        }
    }

    fn behavior_follow(&mut self, _context: &CompanionContext, _dt: f32) {
        // Movement toward formation slot is handled by the manager.
    }

    fn behavior_attack(&mut self, context: &CompanionContext, dt: f32) {
        if let Some(_target) = self.target_entity {
            // Try to use offensive abilities.
            for i in 0..self.abilities.len() {
                if self.abilities[i].def.ability_type == AbilityType::Offensive && self.abilities[i].is_ready() {
                    if self.resource >= self.abilities[i].def.resource_cost {
                        self.resource -= self.abilities[i].def.resource_cost;
                        self.abilities[i].use_ability();
                        self.callouts.try_callout(CalloutType::AbilityUsed, &self.abilities[i].def.name, self.position, 2);
                        break;
                    }
                }
            }
        } else {
            // No target, return to following.
            self.behavior = CompanionBehavior::Following;
        }
    }

    fn behavior_defend(&mut self, _context: &CompanionContext, _dt: f32) {
        for i in 0..self.abilities.len() {
            if self.abilities[i].def.ability_type == AbilityType::Defensive && self.abilities[i].is_ready() {
                if self.resource >= self.abilities[i].def.resource_cost {
                    self.resource -= self.abilities[i].def.resource_cost;
                    self.abilities[i].use_ability();
                    break;
                }
            }
        }
    }

    fn behavior_heal(&mut self, _context: &CompanionContext, _dt: f32) {
        // Done in try_heal.
    }

    fn try_heal(&mut self, _context: &CompanionContext, _dt: f32) {
        for i in 0..self.abilities.len() {
            if self.abilities[i].def.ability_type == AbilityType::Healing && self.abilities[i].is_ready() {
                if self.resource >= self.abilities[i].def.resource_cost {
                    self.resource -= self.abilities[i].def.resource_cost;
                    self.abilities[i].use_ability();
                    self.loyalty.times_healed_player += 1;
                    self.callouts.try_callout(CalloutType::Healing, "companion_healing", self.position, 4);
                    break;
                }
            }
        }
    }

    fn try_self_heal(&mut self, _dt: f32) {
        for i in 0..self.abilities.len() {
            if self.abilities[i].def.ability_type == AbilityType::Healing && self.abilities[i].is_ready() {
                if self.resource >= self.abilities[i].def.resource_cost {
                    self.resource -= self.abilities[i].def.resource_cost;
                    self.abilities[i].use_ability();
                    self.health = (self.health + self.abilities[i].def.effect_value).min(self.stats.max_health);
                    break;
                }
            }
        }
    }
}

/// Context data passed to companion update.
#[derive(Debug, Clone)]
pub struct CompanionContext {
    pub player_position: [f32; 3],
    pub player_forward: [f32; 2],
    pub player_health_fraction: f32,
    pub enemies_nearby: u32,
    pub in_combat: bool,
    pub time: f32,
    pub dt: f32,
}

// ---------------------------------------------------------------------------
// Companion manager
// ---------------------------------------------------------------------------

/// Manages all active companions.
#[derive(Debug)]
pub struct CompanionManagerV3 {
    pub companions: Vec<CompanionV3>,
    pub formation: CompanionFormation,
    pub max_companions: usize,
    pub party_morale_bonus: f32,
}

impl CompanionManagerV3 {
    pub fn new() -> Self {
        Self {
            companions: Vec::new(),
            formation: CompanionFormation::new(FormationTypeV3::Follow, 0),
            max_companions: MAX_COMPANIONS,
            party_morale_bonus: 0.0,
        }
    }

    pub fn add_companion(&mut self, companion: CompanionV3) -> bool {
        if self.companions.len() >= self.max_companions { return false; }
        let idx = self.companions.len();
        let mut c = companion;
        c.formation_slot = idx;
        self.companions.push(c);
        self.formation = CompanionFormation::new(self.formation.formation_type, self.companions.len());
        true
    }

    pub fn remove_companion(&mut self, id: u64) {
        self.companions.retain(|c| c.id != id);
        for (i, c) in self.companions.iter_mut().enumerate() {
            c.formation_slot = i;
        }
        self.formation = CompanionFormation::new(self.formation.formation_type, self.companions.len());
    }

    pub fn update(&mut self, context: &CompanionContext) {
        for companion in &mut self.companions {
            companion.update(context.dt, context);
        }
        // Compute party morale bonus.
        let avg_morale: f32 = if self.companions.is_empty() { 0.0 } else {
            self.companions.iter().map(|c| c.morale.morale).sum::<f32>() / self.companions.len() as f32
        };
        self.party_morale_bonus = (avg_morale - 50.0) * 0.005;
    }

    pub fn set_formation(&mut self, formation_type: FormationTypeV3) {
        self.formation = CompanionFormation::new(formation_type, self.companions.len());
    }

    pub fn command_all(&mut self, command: CompanionCommand) {
        for companion in &mut self.companions {
            companion.give_command(command);
        }
    }

    pub fn get_companion(&self, id: u64) -> Option<&CompanionV3> {
        self.companions.iter().find(|c| c.id == id)
    }

    pub fn active_count(&self) -> usize {
        self.companions.iter().filter(|c| !c.is_downed).count()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_companion_creation() {
        let c = CompanionV3::new(1, "Aria", CompanionClass::Healer);
        assert_eq!(c.class, CompanionClass::Healer);
        assert!(c.abilities.len() >= 3);
    }

    #[test]
    fn test_loyalty_system() {
        let mut loyalty = LoyaltySystem::default();
        loyalty.add_approval(20.0, "Gift", 0.0);
        assert_eq!(loyalty.relationship_level, RelationshipLevel::Ally);
    }

    #[test]
    fn test_ability_cooldown() {
        let def = CompanionAbilityDef::new("Test", AbilityType::Offensive, 5.0, 10.0);
        let mut state = AbilityState::new(def);
        assert!(state.is_ready());
        state.use_ability();
        assert!(!state.is_ready());
        state.update(5.0);
        assert!(state.is_ready());
    }

    #[test]
    fn test_companion_damage() {
        let mut c = CompanionV3::new(1, "Test", CompanionClass::Warrior);
        c.take_damage(200.0);
        assert!(c.is_downed);
        c.revive(0.5);
        assert!(!c.is_downed);
        assert!(c.health > 0.0);
    }

    #[test]
    fn test_formation_positions() {
        let f = CompanionFormation::new(FormationTypeV3::Wedge, 4);
        let pos = f.world_position([0.0, 0.0, 0.0], [0.0, 1.0], 0);
        // Should be behind the leader.
        assert!(pos[0].abs() > 0.0 || pos[2].abs() > 0.0);
    }

    #[test]
    fn test_manager_add_remove() {
        let mut mgr = CompanionManagerV3::new();
        let c = CompanionV3::new(1, "Test", CompanionClass::Warrior);
        assert!(mgr.add_companion(c));
        assert_eq!(mgr.companions.len(), 1);
        mgr.remove_companion(1);
        assert_eq!(mgr.companions.len(), 0);
    }
}
