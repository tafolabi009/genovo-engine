//! AI companion system for NPC party members.
//!
//! Provides:
//! - **Follow player**: companions follow at configurable distance
//! - **Combat behavior**: aggressive, defensive, passive modes
//! - **Companion commands**: wait, follow, attack target, go to position
//! - **Companion inventory sharing**: shared loot and item transfer
//! - **Companion leveling**: experience and stat growth
//! - **Companion abilities**: special skills with cooldowns
//! - **Companion mood/morale**: affects combat effectiveness
//! - **Companion dialogue**: contextual bark lines
//! - **Formation**: companions arrange in formation around the player
//! - **ECS integration**: `CompanionComponent`, `CompanionSystem`
//!
//! # Design
//!
//! A [`Companion`] is an NPC entity that follows the player and participates
//! in combat. The [`CompanionManager`] coordinates all active companions,
//! manages their AI behaviors, and handles commands from the player.

use glam::Vec3;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of active companions.
pub const MAX_COMPANIONS: usize = 4;
/// Default follow distance (meters behind player).
pub const DEFAULT_FOLLOW_DISTANCE: f32 = 3.0;
/// Distance at which companions start sprinting to catch up.
pub const CATCH_UP_DISTANCE: f32 = 10.0;
/// Maximum leash distance before teleporting to player.
pub const MAX_LEASH_DISTANCE: f32 = 50.0;
/// Default combat engagement range.
pub const DEFAULT_COMBAT_RANGE: f32 = 15.0;
/// Default melee attack range.
pub const DEFAULT_MELEE_RANGE: f32 = 2.0;
/// Default ranged attack range.
pub const DEFAULT_RANGED_RANGE: f32 = 20.0;
/// Morale decay rate per second in combat.
pub const MORALE_DECAY_RATE: f32 = 0.5;
/// Morale recovery rate per second out of combat.
pub const MORALE_RECOVERY_RATE: f32 = 1.0;
/// Morale loss on companion being downed.
pub const MORALE_LOSS_ON_DOWN: f32 = 20.0;
/// Morale boost on enemy kill.
pub const MORALE_BOOST_ON_KILL: f32 = 5.0;
/// Maximum companion level.
pub const MAX_COMPANION_LEVEL: u32 = 50;
/// XP multiplier for companion (fraction of player XP).
pub const COMPANION_XP_FRACTION: f32 = 0.75;
/// Default companion inventory slots.
pub const DEFAULT_COMPANION_SLOTS: usize = 12;
/// Minimum movement speed.
pub const MIN_MOVE_SPEED: f32 = 1.0;
/// Small epsilon.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// CompanionId
// ---------------------------------------------------------------------------

/// Unique identifier for a companion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompanionId(pub u64);

// ---------------------------------------------------------------------------
// CombatBehavior
// ---------------------------------------------------------------------------

/// How the companion behaves in combat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombatBehavior {
    /// Actively seeks and attacks enemies.
    Aggressive,
    /// Only attacks enemies that attack the player or companion.
    Defensive,
    /// Does not attack, focuses on healing/support.
    Support,
    /// Does not engage in combat at all.
    Passive,
    /// Focuses on ranged attacks from a safe distance.
    Ranged,
    /// Tanks enemies and draws aggro.
    Tank,
}

impl CombatBehavior {
    /// Display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Aggressive => "Aggressive",
            Self::Defensive => "Defensive",
            Self::Support => "Support",
            Self::Passive => "Passive",
            Self::Ranged => "Ranged",
            Self::Tank => "Tank",
        }
    }

    /// Whether this behavior engages in combat.
    pub fn engages_combat(&self) -> bool {
        !matches!(self, Self::Passive)
    }

    /// Preferred distance from enemies.
    pub fn preferred_distance(&self) -> f32 {
        match self {
            Self::Aggressive => DEFAULT_MELEE_RANGE,
            Self::Defensive => DEFAULT_MELEE_RANGE * 1.5,
            Self::Support => DEFAULT_RANGED_RANGE,
            Self::Passive => DEFAULT_FOLLOW_DISTANCE,
            Self::Ranged => DEFAULT_RANGED_RANGE,
            Self::Tank => DEFAULT_MELEE_RANGE * 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// CompanionCommand
// ---------------------------------------------------------------------------

/// A command issued to a companion.
#[derive(Debug, Clone)]
pub enum CompanionCommand {
    /// Follow the player.
    Follow,
    /// Wait at current position.
    Wait,
    /// Move to a specific position.
    GoTo(Vec3),
    /// Attack a specific target.
    Attack(u64),
    /// Retreat/fall back to player.
    Retreat,
    /// Use a specific ability.
    UseAbility(String),
    /// Interact with an object.
    Interact(u64),
    /// Guard a position.
    Guard(Vec3),
    /// Dismiss (remove from party).
    Dismiss,
    /// Recall (teleport to player).
    Recall,
    /// Toggle combat behavior.
    SetCombatBehavior(CombatBehavior),
    /// Loot nearby items.
    Loot,
    /// Heal the player (if support).
    HealPlayer,
}

// ---------------------------------------------------------------------------
// CompanionState
// ---------------------------------------------------------------------------

/// Current AI state of a companion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompanionState {
    /// Following the player.
    Following,
    /// Waiting at a position.
    Waiting,
    /// Moving to a destination.
    MovingTo,
    /// Engaged in combat.
    Combat,
    /// Retreating from combat.
    Retreating,
    /// Guarding a position.
    Guarding,
    /// Using an ability.
    UsingAbility,
    /// Downed (needs revival).
    Downed,
    /// Looting nearby items.
    Looting,
    /// Interacting with an object.
    Interacting,
    /// Idle behavior (e.g., looking around).
    Idle,
}

// ---------------------------------------------------------------------------
// CompanionStats
// ---------------------------------------------------------------------------

/// Stats for a companion.
#[derive(Debug, Clone)]
pub struct CompanionStats {
    /// Current level.
    pub level: u32,
    /// Current XP.
    pub xp: u32,
    /// XP needed for next level.
    pub xp_for_next_level: u32,
    /// Maximum health.
    pub max_health: f32,
    /// Current health.
    pub health: f32,
    /// Maximum mana/energy.
    pub max_mana: f32,
    /// Current mana.
    pub mana: f32,
    /// Attack power.
    pub attack: f32,
    /// Defense.
    pub defense: f32,
    /// Movement speed (m/s).
    pub move_speed: f32,
    /// Sprint speed.
    pub sprint_speed: f32,
    /// Critical hit chance (0..1).
    pub crit_chance: f32,
    /// Critical hit damage multiplier.
    pub crit_multiplier: f32,
}

impl Default for CompanionStats {
    fn default() -> Self {
        Self {
            level: 1,
            xp: 0,
            xp_for_next_level: 100,
            max_health: 100.0,
            health: 100.0,
            max_mana: 50.0,
            mana: 50.0,
            attack: 10.0,
            defense: 5.0,
            move_speed: 5.0,
            sprint_speed: 10.0,
            crit_chance: 0.05,
            crit_multiplier: 1.5,
        }
    }
}

impl CompanionStats {
    /// Add XP and check for level up. Returns new level if leveled up.
    pub fn add_xp(&mut self, xp: u32) -> Option<u32> {
        self.xp += xp;
        if self.xp >= self.xp_for_next_level && self.level < MAX_COMPANION_LEVEL {
            self.xp -= self.xp_for_next_level;
            self.level += 1;
            self.xp_for_next_level = self.level * 100 + 50;

            // Stat growth per level
            self.max_health += 10.0;
            self.health = self.max_health;
            self.max_mana += 5.0;
            self.mana = self.max_mana;
            self.attack += 2.0;
            self.defense += 1.0;

            Some(self.level)
        } else {
            None
        }
    }

    /// Take damage. Returns actual damage dealt.
    pub fn take_damage(&mut self, amount: f32) -> f32 {
        let actual = (amount - self.defense).max(1.0);
        self.health = (self.health - actual).max(0.0);
        actual
    }

    /// Heal.
    pub fn heal(&mut self, amount: f32) {
        self.health = (self.health + amount).min(self.max_health);
    }

    /// Check if alive.
    pub fn is_alive(&self) -> bool {
        self.health > 0.0
    }

    /// Check if downed (health <= 0).
    pub fn is_downed(&self) -> bool {
        self.health <= 0.0
    }

    /// Revive the companion with a fraction of health.
    pub fn revive(&mut self, health_fraction: f32) {
        self.health = self.max_health * health_fraction.clamp(0.1, 1.0);
    }

    /// Consume mana. Returns false if not enough.
    pub fn consume_mana(&mut self, amount: f32) -> bool {
        if self.mana >= amount {
            self.mana -= amount;
            true
        } else {
            false
        }
    }

    /// Regenerate mana.
    pub fn regen_mana(&mut self, dt: f32) {
        self.mana = (self.mana + 2.0 * dt).min(self.max_mana);
    }

    /// Get health fraction (0..1).
    pub fn health_fraction(&self) -> f32 {
        self.health / self.max_health
    }
}

// ---------------------------------------------------------------------------
// CompanionAbility
// ---------------------------------------------------------------------------

/// An ability that a companion can use.
#[derive(Debug, Clone)]
pub struct CompanionAbility {
    /// Ability name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Cooldown duration (seconds).
    pub cooldown: f32,
    /// Current cooldown remaining.
    pub cooldown_remaining: f32,
    /// Mana cost.
    pub mana_cost: f32,
    /// Damage (if damaging ability).
    pub damage: f32,
    /// Heal amount (if healing ability).
    pub heal_amount: f32,
    /// Range.
    pub range: f32,
    /// Area of effect radius.
    pub aoe_radius: f32,
    /// Whether this is an auto-cast ability.
    pub auto_cast: bool,
    /// Level requirement.
    pub level_required: u32,
    /// Ability type.
    pub ability_type: CompanionAbilityType,
}

/// Type of companion ability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompanionAbilityType {
    /// Melee attack.
    MeleeAttack,
    /// Ranged attack.
    RangedAttack,
    /// Area attack.
    AreaAttack,
    /// Single-target heal.
    Heal,
    /// Area heal.
    AreaHeal,
    /// Buff ally.
    Buff,
    /// Debuff enemy.
    Debuff,
    /// Crowd control.
    CrowdControl,
    /// Defensive (shield, taunt).
    Defensive,
}

impl CompanionAbility {
    /// Create a new ability.
    pub fn new(name: impl Into<String>, ability_type: CompanionAbilityType) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            cooldown: 10.0,
            cooldown_remaining: 0.0,
            mana_cost: 10.0,
            damage: 0.0,
            heal_amount: 0.0,
            range: 5.0,
            aoe_radius: 0.0,
            auto_cast: true,
            level_required: 1,
            ability_type,
        }
    }

    /// Check if the ability is ready.
    pub fn is_ready(&self) -> bool {
        self.cooldown_remaining <= 0.0
    }

    /// Use the ability (start cooldown).
    pub fn use_ability(&mut self) {
        self.cooldown_remaining = self.cooldown;
    }

    /// Update cooldown.
    pub fn update(&mut self, dt: f32) {
        if self.cooldown_remaining > 0.0 {
            self.cooldown_remaining = (self.cooldown_remaining - dt).max(0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// CompanionMood
// ---------------------------------------------------------------------------

/// Mood/morale state of a companion.
#[derive(Debug, Clone)]
pub struct CompanionMood {
    /// Current morale (0..100).
    pub morale: f32,
    /// Base morale (returns to this over time).
    pub base_morale: f32,
    /// Current mood type.
    pub mood: MoodType,
    /// Relationship level with the player (0..100).
    pub relationship: f32,
    /// Loyalty level (affects chance of leaving party).
    pub loyalty: f32,
}

/// Type of mood.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoodType {
    /// Happy and enthusiastic.
    Happy,
    /// Normal/content.
    Content,
    /// Worried/anxious.
    Worried,
    /// Angry/frustrated.
    Angry,
    /// Sad/depressed.
    Sad,
    /// Fearful.
    Fearful,
    /// Determined/brave.
    Determined,
}

impl Default for CompanionMood {
    fn default() -> Self {
        Self {
            morale: 75.0,
            base_morale: 75.0,
            mood: MoodType::Content,
            relationship: 50.0,
            loyalty: 70.0,
        }
    }
}

impl CompanionMood {
    /// Update mood based on morale level.
    pub fn update_mood(&mut self) {
        self.mood = match self.morale {
            x if x >= 90.0 => MoodType::Happy,
            x if x >= 70.0 => MoodType::Content,
            x if x >= 50.0 => MoodType::Worried,
            x if x >= 30.0 => MoodType::Sad,
            x if x >= 15.0 => MoodType::Fearful,
            _ => MoodType::Angry,
        };
    }

    /// Apply a morale change.
    pub fn change_morale(&mut self, amount: f32) {
        self.morale = (self.morale + amount).clamp(0.0, 100.0);
        self.update_mood();
    }

    /// Recover morale toward base.
    pub fn recover(&mut self, dt: f32) {
        if self.morale < self.base_morale {
            self.morale = (self.morale + MORALE_RECOVERY_RATE * dt).min(self.base_morale);
        }
        self.update_mood();
    }

    /// Get the effectiveness multiplier from morale (affects damage and healing).
    pub fn effectiveness_multiplier(&self) -> f32 {
        if self.morale >= 80.0 {
            1.2
        } else if self.morale >= 50.0 {
            1.0
        } else if self.morale >= 20.0 {
            0.8
        } else {
            0.5
        }
    }

    /// Change relationship with player.
    pub fn change_relationship(&mut self, amount: f32) {
        self.relationship = (self.relationship + amount).clamp(0.0, 100.0);
    }
}

// ---------------------------------------------------------------------------
// CompanionInventory
// ---------------------------------------------------------------------------

/// Companion inventory.
#[derive(Debug, Clone)]
pub struct CompanionInventory {
    /// Items (item_id -> quantity).
    pub items: HashMap<String, u32>,
    /// Maximum number of unique item stacks.
    pub max_slots: usize,
    /// Whether the companion auto-loots items.
    pub auto_loot: bool,
    /// Item filter: item IDs that the companion won't pick up.
    pub loot_blacklist: Vec<String>,
}

impl CompanionInventory {
    /// Create a new inventory.
    pub fn new(max_slots: usize) -> Self {
        Self {
            items: HashMap::new(),
            max_slots,
            auto_loot: true,
            loot_blacklist: Vec::new(),
        }
    }

    /// Add an item. Returns false if inventory is full.
    pub fn add_item(&mut self, item_id: impl Into<String>, quantity: u32) -> bool {
        let item_id = item_id.into();
        if self.items.contains_key(&item_id) {
            *self.items.get_mut(&item_id).unwrap() += quantity;
            true
        } else if self.items.len() < self.max_slots {
            self.items.insert(item_id, quantity);
            true
        } else {
            false
        }
    }

    /// Remove an item. Returns false if not enough.
    pub fn remove_item(&mut self, item_id: &str, quantity: u32) -> bool {
        if let Some(count) = self.items.get_mut(item_id) {
            if *count >= quantity {
                *count -= quantity;
                if *count == 0 {
                    self.items.remove(item_id);
                }
                return true;
            }
        }
        false
    }

    /// Check if an item exists.
    pub fn has_item(&self, item_id: &str, quantity: u32) -> bool {
        self.items.get(item_id).copied().unwrap_or(0) >= quantity
    }

    /// Get the number of used slots.
    pub fn used_slots(&self) -> usize {
        self.items.len()
    }

    /// Check if full.
    pub fn is_full(&self) -> bool {
        self.items.len() >= self.max_slots
    }
}

// ---------------------------------------------------------------------------
// Companion
// ---------------------------------------------------------------------------

/// An AI companion NPC.
#[derive(Debug, Clone)]
pub struct Companion {
    /// Unique ID.
    pub id: CompanionId,
    /// Display name.
    pub name: String,
    /// Companion class/role.
    pub role: String,
    /// Current AI state.
    pub state: CompanionState,
    /// Combat behavior.
    pub combat_behavior: CombatBehavior,
    /// Stats.
    pub stats: CompanionStats,
    /// Abilities.
    pub abilities: Vec<CompanionAbility>,
    /// Mood/morale.
    pub mood: CompanionMood,
    /// Inventory.
    pub inventory: CompanionInventory,
    /// Position in world.
    pub position: Vec3,
    /// Target position (for MovingTo state).
    pub target_position: Option<Vec3>,
    /// Current combat target (entity ID).
    pub combat_target: Option<u64>,
    /// Follow distance.
    pub follow_distance: f32,
    /// Whether the companion is in the active party.
    pub in_party: bool,
    /// Whether the companion is controllable.
    pub controllable: bool,
    /// Portrait/icon reference.
    pub portrait: String,
    /// Command history (last N commands).
    command_history: Vec<CompanionCommand>,
    /// Time in current state (seconds).
    pub state_time: f32,
    /// Whether currently in combat.
    pub in_combat: bool,
    /// Bark lines for different situations.
    pub barks: HashMap<String, Vec<String>>,
    /// Formation offset from player.
    pub formation_offset: Vec3,
}

impl Companion {
    /// Create a new companion.
    pub fn new(id: CompanionId, name: impl Into<String>, role: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            role: role.into(),
            state: CompanionState::Following,
            combat_behavior: CombatBehavior::Defensive,
            stats: CompanionStats::default(),
            abilities: Vec::new(),
            mood: CompanionMood::default(),
            inventory: CompanionInventory::new(DEFAULT_COMPANION_SLOTS),
            position: Vec3::ZERO,
            target_position: None,
            combat_target: None,
            follow_distance: DEFAULT_FOLLOW_DISTANCE,
            in_party: true,
            controllable: true,
            portrait: String::new(),
            command_history: Vec::new(),
            state_time: 0.0,
            in_combat: false,
            barks: HashMap::new(),
            formation_offset: Vec3::new(-2.0, 0.0, -1.0),
        }
    }

    /// Issue a command to this companion.
    pub fn issue_command(&mut self, command: CompanionCommand) {
        match &command {
            CompanionCommand::Follow => {
                self.state = CompanionState::Following;
                self.combat_target = None;
            }
            CompanionCommand::Wait => {
                self.state = CompanionState::Waiting;
            }
            CompanionCommand::GoTo(pos) => {
                self.state = CompanionState::MovingTo;
                self.target_position = Some(*pos);
            }
            CompanionCommand::Attack(target) => {
                if self.combat_behavior.engages_combat() {
                    self.state = CompanionState::Combat;
                    self.combat_target = Some(*target);
                    self.in_combat = true;
                }
            }
            CompanionCommand::Retreat => {
                self.state = CompanionState::Retreating;
                self.combat_target = None;
                self.in_combat = false;
            }
            CompanionCommand::Guard(pos) => {
                self.state = CompanionState::Guarding;
                self.target_position = Some(*pos);
            }
            CompanionCommand::Recall => {
                self.state = CompanionState::Following;
                self.combat_target = None;
            }
            CompanionCommand::SetCombatBehavior(behavior) => {
                self.combat_behavior = *behavior;
            }
            CompanionCommand::Dismiss => {
                self.in_party = false;
                self.state = CompanionState::Idle;
            }
            _ => {}
        }

        self.command_history.push(command);
        if self.command_history.len() > 20 {
            self.command_history.remove(0);
        }
        self.state_time = 0.0;
    }

    /// Update the companion AI.
    pub fn update(&mut self, dt: f32, player_pos: Vec3) {
        self.state_time += dt;

        // Update ability cooldowns
        for ability in &mut self.abilities {
            ability.update(dt);
        }

        // Regenerate mana
        self.stats.regen_mana(dt);

        // Update mood
        if self.in_combat {
            self.mood.change_morale(-MORALE_DECAY_RATE * dt);
        } else {
            self.mood.recover(dt);
        }

        // State-specific behavior
        match self.state {
            CompanionState::Following => {
                self.update_follow(dt, player_pos);
            }
            CompanionState::Waiting => {
                // Stand still, look around
            }
            CompanionState::MovingTo => {
                if let Some(target) = self.target_position {
                    self.move_toward(target, self.stats.move_speed, dt);
                    if (self.position - target).length() < 1.0 {
                        self.state = CompanionState::Waiting;
                        self.target_position = None;
                    }
                }
            }
            CompanionState::Combat => {
                // Combat AI
                self.update_combat(dt, player_pos);
            }
            CompanionState::Retreating => {
                self.move_toward(player_pos, self.stats.sprint_speed, dt);
                if (self.position - player_pos).length() < DEFAULT_FOLLOW_DISTANCE {
                    self.state = CompanionState::Following;
                }
            }
            CompanionState::Guarding => {
                if let Some(guard_pos) = self.target_position {
                    if (self.position - guard_pos).length() > 2.0 {
                        self.move_toward(guard_pos, self.stats.move_speed, dt);
                    }
                }
            }
            CompanionState::Downed => {
                // Waiting for revival
            }
            _ => {}
        }

        // Check leash distance
        if self.state != CompanionState::Downed {
            let dist_to_player = (self.position - player_pos).length();
            if dist_to_player > MAX_LEASH_DISTANCE {
                // Teleport to player
                self.position = player_pos + self.formation_offset;
            }
        }
    }

    /// Follow behavior.
    fn update_follow(&mut self, dt: f32, player_pos: Vec3) {
        let target = player_pos + self.formation_offset;
        let dist = (self.position - target).length();

        if dist > CATCH_UP_DISTANCE {
            self.move_toward(target, self.stats.sprint_speed, dt);
        } else if dist > self.follow_distance {
            self.move_toward(target, self.stats.move_speed, dt);
        }
    }

    /// Combat behavior.
    fn update_combat(&mut self, _dt: f32, player_pos: Vec3) {
        if !self.combat_behavior.engages_combat() {
            self.state = CompanionState::Following;
            return;
        }

        // If no target, go back to following
        if self.combat_target.is_none() {
            self.in_combat = false;
            self.state = CompanionState::Following;
            return;
        }

        // Check if health is low -> retreat
        if self.stats.health_fraction() < 0.2 {
            self.state = CompanionState::Retreating;
            self.combat_target = None;
            self.in_combat = false;
        }
    }

    /// Move toward a position.
    fn move_toward(&mut self, target: Vec3, speed: f32, dt: f32) {
        let dir = target - self.position;
        let dist = dir.length();
        if dist < EPSILON {
            return;
        }
        let move_dist = (speed * dt).min(dist);
        self.position += (dir / dist) * move_dist;
    }

    /// Take damage.
    pub fn take_damage(&mut self, amount: f32) -> f32 {
        let actual = self.stats.take_damage(amount * self.mood.effectiveness_multiplier());
        if self.stats.is_downed() {
            self.state = CompanionState::Downed;
            self.in_combat = false;
            self.combat_target = None;
            self.mood.change_morale(-MORALE_LOSS_ON_DOWN);
        }
        actual
    }

    /// Revive the companion.
    pub fn revive(&mut self, health_fraction: f32) {
        self.stats.revive(health_fraction);
        self.state = CompanionState::Following;
    }

    /// Notify the companion of an enemy kill (morale boost).
    pub fn on_enemy_killed(&mut self) {
        self.mood.change_morale(MORALE_BOOST_ON_KILL);
    }

    /// Add an ability.
    pub fn add_ability(&mut self, ability: CompanionAbility) {
        self.abilities.push(ability);
    }

    /// Get a bark line for a situation.
    pub fn get_bark(&self, situation: &str) -> Option<&str> {
        self.barks.get(situation)
            .and_then(|lines| {
                if lines.is_empty() {
                    None
                } else {
                    // Simple round-robin (in a real implementation, use randomness)
                    Some(lines[0].as_str())
                }
            })
    }

    /// Add bark lines for a situation.
    pub fn add_barks(&mut self, situation: impl Into<String>, lines: Vec<String>) {
        self.barks.insert(situation.into(), lines);
    }
}

// ---------------------------------------------------------------------------
// CompanionEvent
// ---------------------------------------------------------------------------

/// Events from the companion system.
#[derive(Debug, Clone)]
pub enum CompanionEvent {
    /// Companion joined the party.
    Joined { companion_id: CompanionId, name: String },
    /// Companion left the party.
    Left { companion_id: CompanionId, name: String },
    /// Companion was downed.
    Downed { companion_id: CompanionId },
    /// Companion was revived.
    Revived { companion_id: CompanionId },
    /// Companion leveled up.
    LevelUp { companion_id: CompanionId, new_level: u32 },
    /// Companion used an ability.
    AbilityUsed { companion_id: CompanionId, ability_name: String },
    /// Companion mood changed.
    MoodChanged { companion_id: CompanionId, new_mood: MoodType },
    /// Companion said a bark.
    Bark { companion_id: CompanionId, text: String },
    /// Command issued to companion.
    CommandIssued { companion_id: CompanionId, command: String },
    /// Companion teleported to player (leash).
    Teleported { companion_id: CompanionId },
}

// ---------------------------------------------------------------------------
// CompanionManager
// ---------------------------------------------------------------------------

/// Manages all companions.
pub struct CompanionManager {
    /// All companions.
    companions: HashMap<CompanionId, Companion>,
    /// Next companion ID.
    next_id: u64,
    /// Events from last update.
    events: Vec<CompanionEvent>,
    /// Formation type.
    pub formation: FormationType,
}

/// Party formation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormationType {
    /// Follow in a line behind the player.
    Line,
    /// Spread out in a V shape.
    Wedge,
    /// Circle around the player.
    Circle,
    /// Two columns.
    Column,
    /// Free form (each companion uses their own offset).
    Free,
}

impl CompanionManager {
    /// Create a new companion manager.
    pub fn new() -> Self {
        Self {
            companions: HashMap::new(),
            next_id: 1,
            events: Vec::new(),
            formation: FormationType::Wedge,
        }
    }

    /// Add a companion to the party.
    pub fn add_companion(&mut self, mut companion: Companion) -> Option<CompanionId> {
        if self.active_count() >= MAX_COMPANIONS {
            return None;
        }
        let id = CompanionId(self.next_id);
        self.next_id += 1;
        companion.id = id;
        companion.in_party = true;

        // Assign formation position
        let index = self.companions.len();
        companion.formation_offset = self.calculate_formation_offset(index);

        self.events.push(CompanionEvent::Joined {
            companion_id: id,
            name: companion.name.clone(),
        });
        self.companions.insert(id, companion);
        Some(id)
    }

    /// Remove a companion from the party.
    pub fn remove_companion(&mut self, id: CompanionId) -> Option<Companion> {
        if let Some(mut companion) = self.companions.remove(&id) {
            companion.in_party = false;
            self.events.push(CompanionEvent::Left {
                companion_id: id,
                name: companion.name.clone(),
            });
            Some(companion)
        } else {
            None
        }
    }

    /// Get a companion by ID.
    pub fn get_companion(&self, id: CompanionId) -> Option<&Companion> {
        self.companions.get(&id)
    }

    /// Get a companion mutably.
    pub fn get_companion_mut(&mut self, id: CompanionId) -> Option<&mut Companion> {
        self.companions.get_mut(&id)
    }

    /// Issue a command to all companions.
    pub fn command_all(&mut self, command: CompanionCommand) {
        let ids: Vec<CompanionId> = self.companions.keys().copied().collect();
        for id in ids {
            if let Some(companion) = self.companions.get_mut(&id) {
                companion.issue_command(command.clone());
            }
        }
    }

    /// Issue a command to a specific companion.
    pub fn command(&mut self, id: CompanionId, command: CompanionCommand) {
        if let Some(companion) = self.companions.get_mut(&id) {
            companion.issue_command(command);
        }
    }

    /// Update all companions.
    pub fn update(&mut self, dt: f32, player_pos: Vec3) {
        self.events.clear();
        let ids: Vec<CompanionId> = self.companions.keys().copied().collect();
        for id in ids {
            if let Some(companion) = self.companions.get_mut(&id) {
                companion.update(dt, player_pos);
            }
        }
    }

    /// Calculate formation offset for a companion at a given index.
    fn calculate_formation_offset(&self, index: usize) -> Vec3 {
        let i = index as f32;
        match self.formation {
            FormationType::Line => {
                Vec3::new(0.0, 0.0, -(i + 1.0) * DEFAULT_FOLLOW_DISTANCE)
            }
            FormationType::Wedge => {
                let side = if index % 2 == 0 { -1.0 } else { 1.0 };
                let depth = ((index / 2) + 1) as f32;
                Vec3::new(side * depth * 2.0, 0.0, -depth * DEFAULT_FOLLOW_DISTANCE)
            }
            FormationType::Circle => {
                let angle = (index as f32 / MAX_COMPANIONS as f32) * std::f32::consts::TAU;
                Vec3::new(
                    angle.cos() * DEFAULT_FOLLOW_DISTANCE,
                    0.0,
                    angle.sin() * DEFAULT_FOLLOW_DISTANCE,
                )
            }
            FormationType::Column => {
                let col = if index % 2 == 0 { -1.5 } else { 1.5 };
                let row = (index / 2) as f32 + 1.0;
                Vec3::new(col, 0.0, -row * DEFAULT_FOLLOW_DISTANCE)
            }
            FormationType::Free => {
                Vec3::new(-(i + 1.0) * 2.0, 0.0, -(i + 1.0))
            }
        }
    }

    /// Get active companion count.
    pub fn active_count(&self) -> usize {
        self.companions.values().filter(|c| c.in_party).count()
    }

    /// Get all companion IDs.
    pub fn companion_ids(&self) -> Vec<CompanionId> {
        self.companions.keys().copied().collect()
    }

    /// Get events from last update.
    pub fn events(&self) -> &[CompanionEvent] {
        &self.events
    }

    /// Award XP to all companions.
    pub fn award_xp(&mut self, xp: u32) {
        let companion_xp = (xp as f32 * COMPANION_XP_FRACTION) as u32;
        for companion in self.companions.values_mut() {
            if companion.in_party && companion.stats.is_alive() {
                if let Some(new_level) = companion.stats.add_xp(companion_xp) {
                    self.events.push(CompanionEvent::LevelUp {
                        companion_id: companion.id,
                        new_level,
                    });
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ECS Component
// ---------------------------------------------------------------------------

/// ECS component for companion entities.
#[derive(Debug, Clone)]
pub struct CompanionComponent {
    /// Companion ID.
    pub companion_id: CompanionId,
    /// Whether to show the companion UI panel.
    pub show_ui: bool,
}

impl CompanionComponent {
    /// Create a new companion component.
    pub fn new(companion_id: CompanionId) -> Self {
        Self {
            companion_id,
            show_ui: true,
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
    fn test_companion_creation() {
        let companion = Companion::new(CompanionId(1), "Lydia", "Warrior");
        assert_eq!(companion.name, "Lydia");
        assert_eq!(companion.state, CompanionState::Following);
        assert!(companion.in_party);
    }

    #[test]
    fn test_companion_command() {
        let mut companion = Companion::new(CompanionId(1), "Lydia", "Warrior");
        companion.issue_command(CompanionCommand::Wait);
        assert_eq!(companion.state, CompanionState::Waiting);

        companion.issue_command(CompanionCommand::Follow);
        assert_eq!(companion.state, CompanionState::Following);
    }

    #[test]
    fn test_companion_damage_and_down() {
        let mut companion = Companion::new(CompanionId(1), "Lydia", "Warrior");
        companion.take_damage(200.0);
        assert!(companion.stats.is_downed());
        assert_eq!(companion.state, CompanionState::Downed);
    }

    #[test]
    fn test_companion_revive() {
        let mut companion = Companion::new(CompanionId(1), "Lydia", "Warrior");
        companion.take_damage(200.0);
        companion.revive(0.5);
        assert!(companion.stats.is_alive());
        assert_eq!(companion.state, CompanionState::Following);
    }

    #[test]
    fn test_companion_level_up() {
        let mut stats = CompanionStats::default();
        let result = stats.add_xp(100);
        assert!(result.is_some());
        assert_eq!(stats.level, 2);
    }

    #[test]
    fn test_morale_effectiveness() {
        let mut mood = CompanionMood::default();
        mood.morale = 90.0;
        mood.update_mood();
        assert!(mood.effectiveness_multiplier() > 1.0);

        mood.morale = 10.0;
        mood.update_mood();
        assert!(mood.effectiveness_multiplier() < 1.0);
    }

    #[test]
    fn test_companion_manager() {
        let mut manager = CompanionManager::new();
        let companion = Companion::new(CompanionId(0), "Lydia", "Warrior");
        let id = manager.add_companion(companion);
        assert!(id.is_some());
        assert_eq!(manager.active_count(), 1);
    }
}
