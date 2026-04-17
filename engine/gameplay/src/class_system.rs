// engine/gameplay/src/class_system.rs
//
// RPG class/job system for the Genovo engine.
//
// Provides a flexible class system for RPG-style games:
//
// - Class definitions with stats, abilities, and requirements.
// - Class progression with level-based unlocks.
// - Multi-class support with primary/secondary classes.
// - Class switching with cooldowns and restrictions.
// - Passive bonuses from class mastery.
// - Class-specific ability lists.
// - Stat scaling per class.
// - Class prerequisite chains.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of active classes a character can have.
const MAX_ACTIVE_CLASSES: usize = 3;

/// Maximum class level.
const MAX_CLASS_LEVEL: u32 = 100;

/// Default class switch cooldown in seconds.
const DEFAULT_SWITCH_COOLDOWN: f32 = 300.0;

/// Maximum passive bonuses per class.
const MAX_PASSIVE_BONUSES: usize = 16;

// ---------------------------------------------------------------------------
// Class ID
// ---------------------------------------------------------------------------

/// Unique identifier for a class definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClassId(pub u32);

impl ClassId {
    /// No class.
    pub const NONE: Self = Self(0);
}

/// Unique identifier for an ability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AbilityId(pub u32);

/// Unique identifier for a stat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StatId(pub u32);

impl StatId {
    pub const STRENGTH: Self = Self(1);
    pub const DEXTERITY: Self = Self(2);
    pub const INTELLIGENCE: Self = Self(3);
    pub const CONSTITUTION: Self = Self(4);
    pub const WISDOM: Self = Self(5);
    pub const CHARISMA: Self = Self(6);
    pub const LUCK: Self = Self(7);
}

// ---------------------------------------------------------------------------
// Stat Bonus
// ---------------------------------------------------------------------------

/// A stat bonus applied by a class or passive.
#[derive(Debug, Clone)]
pub struct StatBonus {
    /// Which stat is affected.
    pub stat: StatId,
    /// Flat bonus value.
    pub flat: f32,
    /// Percentage bonus (0.1 = +10%).
    pub percent: f32,
    /// Per-level scaling.
    pub per_level: f32,
}

impl StatBonus {
    /// Create a flat stat bonus.
    pub fn flat(stat: StatId, value: f32) -> Self {
        Self { stat, flat: value, percent: 0.0, per_level: 0.0 }
    }

    /// Create a percentage stat bonus.
    pub fn percent(stat: StatId, value: f32) -> Self {
        Self { stat, flat: 0.0, percent: value, per_level: 0.0 }
    }

    /// Create a per-level stat bonus.
    pub fn per_level(stat: StatId, value: f32) -> Self {
        Self { stat, flat: 0.0, percent: 0.0, per_level: value }
    }

    /// Compute the total bonus at a given level.
    pub fn total_at_level(&self, level: u32) -> f32 {
        self.flat + self.per_level * level as f32
    }
}

// ---------------------------------------------------------------------------
// Passive Bonus
// ---------------------------------------------------------------------------

/// A passive bonus granted by class mastery.
#[derive(Debug, Clone)]
pub struct PassiveBonus {
    /// Unique name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Localization key.
    pub loc_key: Option<String>,
    /// Level required to unlock this passive.
    pub required_level: u32,
    /// Stat bonuses granted.
    pub stat_bonuses: Vec<StatBonus>,
    /// Whether this passive persists when switching away from the class.
    pub permanent: bool,
    /// Icon identifier.
    pub icon: String,
}

// ---------------------------------------------------------------------------
// Class Ability
// ---------------------------------------------------------------------------

/// An ability that belongs to a class.
#[derive(Debug, Clone)]
pub struct ClassAbility {
    /// Ability identifier.
    pub ability_id: AbilityId,
    /// Name of the ability.
    pub name: String,
    /// Description.
    pub description: String,
    /// Level required to learn this ability.
    pub required_level: u32,
    /// Whether this ability can be used by other classes once learned.
    pub cross_class: bool,
    /// Cooldown in seconds.
    pub cooldown: f32,
    /// Resource cost (mana, stamina, etc.).
    pub resource_cost: f32,
    /// Resource type name.
    pub resource_type: String,
    /// Icon identifier.
    pub icon: String,
}

// ---------------------------------------------------------------------------
// Class Requirement
// ---------------------------------------------------------------------------

/// A requirement that must be met before a class can be selected.
#[derive(Debug, Clone)]
pub enum ClassRequirement {
    /// Minimum character level.
    CharacterLevel(u32),
    /// Must have another class at a minimum level.
    ClassLevel { class: ClassId, level: u32 },
    /// Must have completed a quest.
    QuestComplete(String),
    /// Must possess a specific item.
    HasItem(String),
    /// Minimum stat value.
    MinStat { stat: StatId, value: f32 },
    /// Multiple requirements (all must be met).
    All(Vec<ClassRequirement>),
    /// Any of the requirements must be met.
    Any(Vec<ClassRequirement>),
}

impl ClassRequirement {
    /// Check if this requirement is met by the given character state.
    pub fn is_met(&self, state: &CharacterClassState) -> bool {
        match self {
            Self::CharacterLevel(level) => state.character_level >= *level,
            Self::ClassLevel { class, level } => {
                state.class_levels.get(class).copied().unwrap_or(0) >= *level
            }
            Self::QuestComplete(quest_id) => state.completed_quests.contains(quest_id),
            Self::HasItem(item_id) => state.owned_items.contains(item_id),
            Self::MinStat { stat, value } => {
                state.base_stats.get(stat).copied().unwrap_or(0.0) >= *value
            }
            Self::All(reqs) => reqs.iter().all(|r| r.is_met(state)),
            Self::Any(reqs) => reqs.iter().any(|r| r.is_met(state)),
        }
    }
}

// ---------------------------------------------------------------------------
// Class Definition
// ---------------------------------------------------------------------------

/// A class/job definition.
#[derive(Debug, Clone)]
pub struct ClassDefinition {
    /// Unique class identifier.
    pub id: ClassId,
    /// Class name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Localization key for name.
    pub name_loc_key: Option<String>,
    /// Category (warrior, mage, rogue, etc.).
    pub category: String,
    /// Base stat bonuses at level 1.
    pub base_bonuses: Vec<StatBonus>,
    /// Abilities learned through leveling.
    pub abilities: Vec<ClassAbility>,
    /// Passive bonuses unlocked at various levels.
    pub passives: Vec<PassiveBonus>,
    /// Requirements to unlock this class.
    pub requirements: Vec<ClassRequirement>,
    /// XP curve type.
    pub xp_curve: XpCurve,
    /// Whether this is an advanced/prestige class.
    pub is_advanced: bool,
    /// Maximum level for this class.
    pub max_level: u32,
    /// Primary resource type (mana, rage, energy, etc.).
    pub resource_type: String,
    /// Base resource pool size at level 1.
    pub base_resource: f32,
    /// Resource per level.
    pub resource_per_level: f32,
    /// Icon identifier.
    pub icon: String,
    /// Color theme for UI.
    pub color: [f32; 4],
}

/// XP curve types for class leveling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XpCurve {
    /// Linear: xp_for_level = base + level * scale.
    Linear,
    /// Quadratic: xp_for_level = base + level^2 * scale.
    Quadratic,
    /// Exponential: xp_for_level = base * scale^level.
    Exponential,
    /// Custom (looked up from a table).
    Custom,
}

impl ClassDefinition {
    /// Create a new class definition.
    pub fn new(id: ClassId, name: &str, category: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            description: String::new(),
            name_loc_key: None,
            category: category.to_string(),
            base_bonuses: Vec::new(),
            abilities: Vec::new(),
            passives: Vec::new(),
            requirements: Vec::new(),
            xp_curve: XpCurve::Quadratic,
            is_advanced: false,
            max_level: MAX_CLASS_LEVEL,
            resource_type: "mana".to_string(),
            base_resource: 100.0,
            resource_per_level: 10.0,
            icon: String::new(),
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }

    /// XP required to reach a given level.
    pub fn xp_for_level(&self, level: u32) -> u64 {
        let base = 100.0f64;
        let scale = 50.0f64;
        let l = level as f64;

        match self.xp_curve {
            XpCurve::Linear => (base + l * scale) as u64,
            XpCurve::Quadratic => (base + l * l * scale) as u64,
            XpCurve::Exponential => (base * 1.15f64.powf(l)) as u64,
            XpCurve::Custom => (base + l * l * scale) as u64,
        }
    }

    /// Get abilities available at a given level.
    pub fn abilities_at_level(&self, level: u32) -> Vec<&ClassAbility> {
        self.abilities.iter().filter(|a| a.required_level <= level).collect()
    }

    /// Get passives unlocked at a given level.
    pub fn passives_at_level(&self, level: u32) -> Vec<&PassiveBonus> {
        self.passives.iter().filter(|p| p.required_level <= level).collect()
    }

    /// Compute total stat bonuses at a given level.
    pub fn total_bonuses_at_level(&self, level: u32) -> Vec<(StatId, f32)> {
        let mut totals: HashMap<StatId, f32> = HashMap::new();

        for bonus in &self.base_bonuses {
            *totals.entry(bonus.stat).or_insert(0.0) += bonus.total_at_level(level);
        }

        for passive in self.passives_at_level(level) {
            for bonus in &passive.stat_bonuses {
                *totals.entry(bonus.stat).or_insert(0.0) += bonus.total_at_level(level);
            }
        }

        totals.into_iter().collect()
    }

    /// Resource pool size at a given level.
    pub fn resource_at_level(&self, level: u32) -> f32 {
        self.base_resource + self.resource_per_level * level as f32
    }
}

// ---------------------------------------------------------------------------
// Character Class State
// ---------------------------------------------------------------------------

/// Character state relevant to the class system (for requirement checking).
#[derive(Debug, Clone)]
pub struct CharacterClassState {
    /// Character level.
    pub character_level: u32,
    /// Levels in each class.
    pub class_levels: HashMap<ClassId, u32>,
    /// Completed quests.
    pub completed_quests: Vec<String>,
    /// Owned items.
    pub owned_items: Vec<String>,
    /// Base stats.
    pub base_stats: HashMap<StatId, f32>,
}

impl CharacterClassState {
    /// Create a new default character state.
    pub fn new(character_level: u32) -> Self {
        Self {
            character_level,
            class_levels: HashMap::new(),
            completed_quests: Vec::new(),
            owned_items: Vec::new(),
            base_stats: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Active Class
// ---------------------------------------------------------------------------

/// An actively-equipped class on a character.
#[derive(Debug, Clone)]
pub struct ActiveClass {
    /// Class identifier.
    pub class_id: ClassId,
    /// Current level in this class.
    pub level: u32,
    /// Current XP in this class.
    pub xp: u64,
    /// Whether this is the primary class.
    pub is_primary: bool,
    /// Slot index (0 = primary, 1+ = secondary).
    pub slot: u32,
    /// Time since last switch away from this class.
    pub time_since_active: f32,
}

impl ActiveClass {
    /// Create a new active class.
    pub fn new(class_id: ClassId, slot: u32) -> Self {
        Self {
            class_id,
            level: 1,
            xp: 0,
            is_primary: slot == 0,
            slot,
            time_since_active: 0.0,
        }
    }

    /// Add XP and check for level up. Returns levels gained.
    pub fn add_xp(&mut self, amount: u64, definition: &ClassDefinition) -> u32 {
        self.xp += amount;
        let mut levels_gained = 0;

        while self.level < definition.max_level {
            let required = definition.xp_for_level(self.level + 1);
            if self.xp >= required {
                self.xp -= required;
                self.level += 1;
                levels_gained += 1;
            } else {
                break;
            }
        }

        levels_gained
    }

    /// XP progress toward next level (0.0 to 1.0).
    pub fn xp_progress(&self, definition: &ClassDefinition) -> f32 {
        if self.level >= definition.max_level {
            return 1.0;
        }
        let required = definition.xp_for_level(self.level + 1);
        if required == 0 {
            return 0.0;
        }
        self.xp as f32 / required as f32
    }
}

// ---------------------------------------------------------------------------
// Class System
// ---------------------------------------------------------------------------

/// Events generated by the class system.
#[derive(Debug, Clone)]
pub enum ClassEvent {
    /// A class was unlocked.
    ClassUnlocked { class_id: ClassId },
    /// A class was equipped.
    ClassEquipped { class_id: ClassId, slot: u32 },
    /// A class was unequipped.
    ClassUnequipped { class_id: ClassId, slot: u32 },
    /// A class leveled up.
    ClassLevelUp { class_id: ClassId, new_level: u32 },
    /// A new ability was learned.
    AbilityLearned { class_id: ClassId, ability_id: AbilityId },
    /// A passive bonus was unlocked.
    PassiveUnlocked { class_id: ClassId, passive_name: String },
    /// Class switch occurred.
    ClassSwitched { from: ClassId, to: ClassId, slot: u32 },
}

/// The class system manager.
#[derive(Debug)]
pub struct ClassSystem {
    /// All class definitions.
    pub definitions: HashMap<ClassId, ClassDefinition>,
    /// Active classes for the player.
    pub active_classes: Vec<ActiveClass>,
    /// Unlocked classes.
    pub unlocked_classes: Vec<ClassId>,
    /// Maximum simultaneous active classes.
    pub max_active: usize,
    /// Class switch cooldown in seconds.
    pub switch_cooldown: f32,
    /// Time since last class switch.
    pub time_since_switch: f32,
    /// Events generated this frame.
    pub events: Vec<ClassEvent>,
    /// Character state for requirement checking.
    pub character_state: CharacterClassState,
}

impl ClassSystem {
    /// Create a new class system.
    pub fn new() -> Self {
        Self {
            definitions: HashMap::new(),
            active_classes: Vec::new(),
            unlocked_classes: Vec::new(),
            max_active: MAX_ACTIVE_CLASSES,
            switch_cooldown: DEFAULT_SWITCH_COOLDOWN,
            time_since_switch: DEFAULT_SWITCH_COOLDOWN, // Allow immediate first switch.
            events: Vec::new(),
            character_state: CharacterClassState::new(1),
        }
    }

    /// Register a class definition.
    pub fn register_class(&mut self, definition: ClassDefinition) {
        self.definitions.insert(definition.id, definition);
    }

    /// Check if a class is unlocked.
    pub fn is_unlocked(&self, class_id: ClassId) -> bool {
        self.unlocked_classes.contains(&class_id)
    }

    /// Try to unlock a class.
    pub fn try_unlock(&mut self, class_id: ClassId) -> bool {
        if self.is_unlocked(class_id) {
            return true;
        }

        let definition = match self.definitions.get(&class_id) {
            Some(d) => d.clone(),
            None => return false,
        };

        let meets_reqs = definition.requirements.iter().all(|r| r.is_met(&self.character_state));
        if meets_reqs {
            self.unlocked_classes.push(class_id);
            self.events.push(ClassEvent::ClassUnlocked { class_id });
            true
        } else {
            false
        }
    }

    /// Equip a class in a slot.
    pub fn equip_class(&mut self, class_id: ClassId, slot: u32) -> bool {
        if !self.is_unlocked(class_id) {
            return false;
        }

        if slot as usize >= self.max_active {
            return false;
        }

        if self.time_since_switch < self.switch_cooldown && !self.active_classes.is_empty() {
            return false;
        }

        let old_class_id = self.active_classes.iter()
            .find(|c| c.slot == slot)
            .map(|c| c.class_id);

        // Remove existing class in this slot.
        self.active_classes.retain(|c| c.slot != slot);

        if let Some(old_id) = old_class_id {
            self.events.push(ClassEvent::ClassUnequipped { class_id: old_id, slot });
            self.events.push(ClassEvent::ClassSwitched { from: old_id, to: class_id, slot });
        }

        let active = ActiveClass::new(class_id, slot);
        self.active_classes.push(active);
        self.time_since_switch = 0.0;
        self.events.push(ClassEvent::ClassEquipped { class_id, slot });

        // Update character state.
        self.character_state.class_levels.insert(class_id, 1);

        true
    }

    /// Add XP to a class.
    pub fn add_class_xp(&mut self, class_id: ClassId, amount: u64) {
        let definition = match self.definitions.get(&class_id) {
            Some(d) => d.clone(),
            None => return,
        };

        if let Some(active) = self.active_classes.iter_mut().find(|c| c.class_id == class_id) {
            let old_level = active.level;
            let levels_gained = active.add_xp(amount, &definition);

            for new_lvl in (old_level + 1)..=(old_level + levels_gained) {
                self.events.push(ClassEvent::ClassLevelUp {
                    class_id,
                    new_level: new_lvl,
                });

                // Check for new abilities.
                for ability in &definition.abilities {
                    if ability.required_level == new_lvl {
                        self.events.push(ClassEvent::AbilityLearned {
                            class_id,
                            ability_id: ability.ability_id,
                        });
                    }
                }

                // Check for new passives.
                for passive in &definition.passives {
                    if passive.required_level == new_lvl {
                        self.events.push(ClassEvent::PassiveUnlocked {
                            class_id,
                            passive_name: passive.name.clone(),
                        });
                    }
                }
            }

            self.character_state.class_levels.insert(class_id, active.level);
        }
    }

    /// Get the primary class.
    pub fn primary_class(&self) -> Option<&ActiveClass> {
        self.active_classes.iter().find(|c| c.is_primary)
    }

    /// Get all active abilities across all equipped classes.
    pub fn all_abilities(&self) -> Vec<(&ClassAbility, ClassId)> {
        let mut abilities = Vec::new();
        for active in &self.active_classes {
            if let Some(def) = self.definitions.get(&active.class_id) {
                for ability in def.abilities_at_level(active.level) {
                    abilities.push((ability, active.class_id));
                }
            }
        }
        abilities
    }

    /// Compute total stat bonuses from all active classes.
    pub fn total_stat_bonuses(&self) -> HashMap<StatId, f32> {
        let mut totals = HashMap::new();
        for active in &self.active_classes {
            if let Some(def) = self.definitions.get(&active.class_id) {
                for (stat, value) in def.total_bonuses_at_level(active.level) {
                    *totals.entry(stat).or_insert(0.0) += value;
                }
            }
        }
        totals
    }

    /// Update the system (advance cooldowns, etc.).
    pub fn update(&mut self, dt: f32) {
        self.time_since_switch += dt;
        for active in &mut self.active_classes {
            active.time_since_active += dt;
        }
    }

    /// Can the player switch classes right now?
    pub fn can_switch(&self) -> bool {
        self.time_since_switch >= self.switch_cooldown
    }

    /// Drain pending events.
    pub fn drain_events(&mut self) -> Vec<ClassEvent> {
        std::mem::take(&mut self.events)
    }
}
