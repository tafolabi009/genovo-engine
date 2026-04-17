//! RPG progression systems: experience, leveling, stats, skill trees, achievements.
//!
//! This module provides a complete progression framework suitable for RPGs,
//! action-RPGs, and any game with character growth:
//!
//! - [`ExperienceSystem`] — XP tracking, level-up detection, configurable XP
//!   curves.
//! - [`StatBlock`] — named stats with a base + modifier (flat & percent)
//!   stacking system.
//! - [`SkillTree`] — prerequisite-gated skill nodes with unlock/query API.
//! - [`AchievementSystem`] — condition-based achievements with reward dispatch.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Experience & Leveling
// ============================================================================

/// The shape of the XP-per-level curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum XPCurveKind {
    /// `xp(n) = base * n * (n + 1) / 2` — gentle quadratic.
    Quadratic,
    /// `xp(n) = base * n^2` — steeper quadratic.
    QuadraticSteep,
    /// `xp(n) = base * n^exponent` — power curve.
    Power,
    /// `xp(n) = base * (ratio^n - 1) / (ratio - 1)` — exponential / geometric.
    Exponential,
    /// `xp(n) = base * n` — linear (equal XP per level).
    Linear,
    /// Custom table — the user supplies XP thresholds directly.
    Custom,
}

impl Default for XPCurveKind {
    fn default() -> Self {
        Self::Quadratic
    }
}

/// Configuration for the XP-to-level curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XPTable {
    /// The type of curve.
    pub kind: XPCurveKind,
    /// Base XP value used in the formula.
    pub base: u64,
    /// Exponent for [`XPCurveKind::Power`].
    pub exponent: f64,
    /// Ratio for [`XPCurveKind::Exponential`].
    pub ratio: f64,
    /// Maximum level. Levels beyond this grant no further progression.
    pub max_level: u32,
    /// Custom XP thresholds (only used with [`XPCurveKind::Custom`]).
    /// Index `i` is the *total* XP required to reach level `i + 1`.
    pub custom_thresholds: Vec<u64>,
}

impl Default for XPTable {
    fn default() -> Self {
        Self {
            kind: XPCurveKind::Quadratic,
            base: 100,
            exponent: 2.0,
            ratio: 1.5,
            max_level: 100,
            custom_thresholds: Vec::new(),
        }
    }
}

impl XPTable {
    /// Calculate the *total* XP required to reach a given level from level 0.
    ///
    /// Level 0 requires 0 XP. Level 1 requires `xp_for_level(1)`, etc.
    pub fn xp_for_level(&self, level: u32) -> u64 {
        if level == 0 {
            return 0;
        }

        match self.kind {
            XPCurveKind::Quadratic => {
                // sum from 1..=level of (base * k) = base * level*(level+1)/2
                let n = level as u64;
                self.base * n * (n + 1) / 2
            }
            XPCurveKind::QuadraticSteep => {
                let n = level as u64;
                self.base * n * n
            }
            XPCurveKind::Power => {
                let n = level as f64;
                (self.base as f64 * n.powf(self.exponent)) as u64
            }
            XPCurveKind::Exponential => {
                if (self.ratio - 1.0).abs() < f64::EPSILON {
                    // Degenerate: linear
                    self.base * level as u64
                } else {
                    let n = level as f64;
                    (self.base as f64 * (self.ratio.powf(n) - 1.0) / (self.ratio - 1.0)) as u64
                }
            }
            XPCurveKind::Linear => self.base * level as u64,
            XPCurveKind::Custom => {
                if let Some(&xp) = self.custom_thresholds.get(level as usize - 1) {
                    xp
                } else {
                    // Fallback to quadratic for levels beyond the table
                    let n = level as u64;
                    self.base * n * (n + 1) / 2
                }
            }
        }
    }

    /// XP needed to go from `current_level` to `current_level + 1`.
    pub fn xp_to_next_level(&self, current_level: u32) -> u64 {
        if current_level >= self.max_level {
            return 0;
        }
        self.xp_for_level(current_level + 1) - self.xp_for_level(current_level)
    }

    /// Given total XP, determine the current level.
    pub fn level_for_xp(&self, total_xp: u64) -> u32 {
        // Binary search for the highest level whose threshold is <= total_xp
        let mut low = 0u32;
        let mut high = self.max_level;
        while low < high {
            let mid = low + (high - low + 1) / 2;
            if self.xp_for_level(mid) <= total_xp {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        low
    }

    /// Progress fraction towards the next level [0, 1].
    pub fn progress_fraction(&self, total_xp: u64) -> f32 {
        let level = self.level_for_xp(total_xp);
        if level >= self.max_level {
            return 1.0;
        }
        let current_threshold = self.xp_for_level(level);
        let next_threshold = self.xp_for_level(level + 1);
        let range = next_threshold - current_threshold;
        if range == 0 {
            return 1.0;
        }
        (total_xp - current_threshold) as f32 / range as f32
    }
}

// ---------------------------------------------------------------------------
// Level-up event
// ---------------------------------------------------------------------------

/// Emitted when an entity reaches a new level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelUpEvent {
    /// The entity that leveled up.
    pub entity: u32,
    /// Previous level.
    pub old_level: u32,
    /// New level.
    pub new_level: u32,
    /// Total XP at the time of leveling.
    pub total_xp: u64,
}

// ---------------------------------------------------------------------------
// Experience system
// ---------------------------------------------------------------------------

/// Per-entity experience record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceRecord {
    /// Total accumulated XP.
    pub total_xp: u64,
    /// Current level (cached; always consistent with `total_xp`).
    pub level: u32,
}

impl Default for ExperienceRecord {
    fn default() -> Self {
        Self {
            total_xp: 0,
            level: 0,
        }
    }
}

/// Manages experience and leveling for all entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceSystem {
    /// XP table shared by all entities in this system.
    pub xp_table: XPTable,
    /// Per-entity XP records.
    records: HashMap<u32, ExperienceRecord>,
    /// Pending level-up events from the last `grant_xp` call.
    pending_events: Vec<LevelUpEvent>,
    /// Global XP multiplier (e.g. from double-XP events).
    pub xp_multiplier: f64,
}

impl ExperienceSystem {
    /// Create a new experience system with the given XP table.
    pub fn new(xp_table: XPTable) -> Self {
        Self {
            xp_table,
            records: HashMap::new(),
            pending_events: Vec::new(),
            xp_multiplier: 1.0,
        }
    }

    /// Register an entity (starts at level 0, 0 XP).
    pub fn register(&mut self, entity: u32) {
        self.records.entry(entity).or_default();
    }

    /// Grant XP to an entity. Returns the number of level-ups that occurred.
    /// Level-up events are stored in `pending_events`.
    pub fn grant_xp(&mut self, entity: u32, base_amount: u64) -> u32 {
        let amount = (base_amount as f64 * self.xp_multiplier) as u64;
        let record = self.records.entry(entity).or_default();
        let old_level = record.level;

        record.total_xp = record.total_xp.saturating_add(amount);
        record.level = self.xp_table.level_for_xp(record.total_xp);

        let levels_gained = record.level - old_level;
        if levels_gained > 0 {
            // Emit one event per level gained (so listeners can apply per-level rewards)
            for lvl in (old_level + 1)..=record.level {
                self.pending_events.push(LevelUpEvent {
                    entity,
                    old_level: lvl - 1,
                    new_level: lvl,
                    total_xp: record.total_xp,
                });
            }
        }

        levels_gained
    }

    /// Drain pending level-up events.
    pub fn drain_events(&mut self) -> Vec<LevelUpEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Get the current level of an entity.
    pub fn level(&self, entity: u32) -> Option<u32> {
        self.records.get(&entity).map(|r| r.level)
    }

    /// Get the total XP of an entity.
    pub fn total_xp(&self, entity: u32) -> Option<u64> {
        self.records.get(&entity).map(|r| r.total_xp)
    }

    /// Get the progress fraction towards the next level.
    pub fn progress(&self, entity: u32) -> Option<f32> {
        self.records
            .get(&entity)
            .map(|r| self.xp_table.progress_fraction(r.total_xp))
    }

    /// Check if the entity has reached the maximum level.
    pub fn is_max_level(&self, entity: u32) -> bool {
        self.records
            .get(&entity)
            .map_or(false, |r| r.level >= self.xp_table.max_level)
    }

    /// Set the entity's XP directly (e.g. for save/load). Updates level.
    pub fn set_xp(&mut self, entity: u32, total_xp: u64) {
        let record = self.records.entry(entity).or_default();
        record.total_xp = total_xp;
        record.level = self.xp_table.level_for_xp(total_xp);
    }

    /// Remove an entity from the system.
    pub fn remove(&mut self, entity: u32) {
        self.records.remove(&entity);
    }

    /// Number of tracked entities.
    pub fn entity_count(&self) -> usize {
        self.records.len()
    }
}

// ============================================================================
// Stat Block / Modifier System
// ============================================================================

/// Common RPG stat names. Games can also use arbitrary string names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CommonStat {
    Strength,
    Dexterity,
    Intelligence,
    Vitality,
    Luck,
    MaxHealth,
    MaxMana,
    Armor,
    MagicResist,
    AttackPower,
    SpellPower,
    CritChance,
    CritDamage,
    MovementSpeed,
    AttackSpeed,
    CooldownReduction,
}

/// Identifier for a stat — either a well-known enum variant or a custom string.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StatId {
    Common(CommonStat),
    Custom(String),
}

impl From<CommonStat> for StatId {
    fn from(s: CommonStat) -> Self {
        StatId::Common(s)
    }
}

impl From<&str> for StatId {
    fn from(s: &str) -> Self {
        StatId::Custom(s.to_string())
    }
}

/// The kind of modifier: flat additive or percentage multiplicative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModifierKind {
    /// Added directly to the base value.
    Flat,
    /// Multiplied: `final *= (1 + percent)`. A value of 0.25 = +25%.
    Percent,
}

/// Duration policy for a stat modifier.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModifierDuration {
    /// Lasts forever (equipment, talent, etc.).
    Permanent,
    /// Expires after `remaining` seconds.
    Timed { remaining: f32 },
}

/// A modifier applied to a stat. Identified by `source_id` so it can be
/// removed when the source (buff, equipment, etc.) is removed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatModifier {
    /// Unique identifier of the source that applied this modifier.
    pub source_id: u64,
    /// Which stat this modifier affects.
    pub stat: StatId,
    /// Flat or percent.
    pub kind: ModifierKind,
    /// The modifier value. For flat: absolute amount. For percent: fractional
    /// (0.1 = +10%).
    pub value: f32,
    /// Duration policy.
    pub duration: ModifierDuration,
    /// Priority for ordering (higher = applied later). Mainly cosmetic for
    /// tooltip ordering.
    pub priority: i32,
}

/// A single stat with base value and collected modifiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stat {
    /// The unmodified base value.
    pub base_value: f32,
    /// Flat modifiers (added to base).
    flat_modifiers: Vec<StatModifier>,
    /// Percent modifiers (multiplicative on top of base + flat).
    percent_modifiers: Vec<StatModifier>,
    /// Cached final value (invalidated on modifier change).
    cached_final: Option<f32>,
    /// Optional min/max clamp.
    pub min_value: Option<f32>,
    pub max_value: Option<f32>,
}

impl Stat {
    /// Create a new stat with a base value.
    pub fn new(base: f32) -> Self {
        Self {
            base_value: base,
            flat_modifiers: Vec::new(),
            percent_modifiers: Vec::new(),
            cached_final: None,
            min_value: None,
            max_value: None,
        }
    }

    /// Create a stat with min/max clamping.
    pub fn with_bounds(base: f32, min: f32, max: f32) -> Self {
        Self {
            base_value: base,
            flat_modifiers: Vec::new(),
            percent_modifiers: Vec::new(),
            cached_final: None,
            min_value: Some(min),
            max_value: Some(max),
        }
    }

    /// Add a modifier. Invalidates the cached final value.
    pub fn add_modifier(&mut self, modifier: StatModifier) {
        self.cached_final = None;
        match modifier.kind {
            ModifierKind::Flat => self.flat_modifiers.push(modifier),
            ModifierKind::Percent => self.percent_modifiers.push(modifier),
        }
    }

    /// Remove all modifiers from a specific source. Returns the number removed.
    pub fn remove_modifiers_by_source(&mut self, source_id: u64) -> usize {
        self.cached_final = None;
        let before = self.flat_modifiers.len() + self.percent_modifiers.len();
        self.flat_modifiers.retain(|m| m.source_id != source_id);
        self.percent_modifiers.retain(|m| m.source_id != source_id);
        let after = self.flat_modifiers.len() + self.percent_modifiers.len();
        before - after
    }

    /// Tick timed modifiers. Removes expired ones. Returns `true` if any
    /// modifier was removed.
    pub fn tick_modifiers(&mut self, dt: f32) -> bool {
        let mut changed = false;

        let drain_timed = |mods: &mut Vec<StatModifier>, dt: f32, changed: &mut bool| {
            mods.retain_mut(|m| {
                if let ModifierDuration::Timed { remaining } = &mut m.duration {
                    *remaining -= dt;
                    if *remaining <= 0.0 {
                        *changed = true;
                        return false;
                    }
                }
                true
            });
        };

        drain_timed(&mut self.flat_modifiers, dt, &mut changed);
        drain_timed(&mut self.percent_modifiers, dt, &mut changed);

        if changed {
            self.cached_final = None;
        }
        changed
    }

    /// Compute the final value: `(base + sum(flat)) * (1 + sum(percent))`.
    ///
    /// The result is cached until modifiers change.
    pub fn final_value(&mut self) -> f32 {
        if let Some(cached) = self.cached_final {
            return cached;
        }

        let flat_sum: f32 = self.flat_modifiers.iter().map(|m| m.value).sum();
        let pct_sum: f32 = self.percent_modifiers.iter().map(|m| m.value).sum();

        let mut val = (self.base_value + flat_sum) * (1.0 + pct_sum);

        if let Some(min) = self.min_value {
            val = val.max(min);
        }
        if let Some(max) = self.max_value {
            val = val.min(max);
        }

        self.cached_final = Some(val);
        val
    }

    /// Compute the final value immutably (no caching).
    pub fn compute_final(&self) -> f32 {
        let flat_sum: f32 = self.flat_modifiers.iter().map(|m| m.value).sum();
        let pct_sum: f32 = self.percent_modifiers.iter().map(|m| m.value).sum();

        let mut val = (self.base_value + flat_sum) * (1.0 + pct_sum);

        if let Some(min) = self.min_value {
            val = val.max(min);
        }
        if let Some(max) = self.max_value {
            val = val.min(max);
        }
        val
    }

    /// Total number of active modifiers.
    pub fn modifier_count(&self) -> usize {
        self.flat_modifiers.len() + self.percent_modifiers.len()
    }

    /// Clear all modifiers.
    pub fn clear_modifiers(&mut self) {
        self.flat_modifiers.clear();
        self.percent_modifiers.clear();
        self.cached_final = None;
    }
}

/// A collection of named stats for an entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatBlock {
    stats: HashMap<StatId, Stat>,
}

impl Default for StatBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StatBlock {
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
        }
    }

    /// Create a stat block pre-populated with common RPG stats.
    pub fn with_defaults(
        str_val: f32,
        dex_val: f32,
        int_val: f32,
        vit_val: f32,
        luck_val: f32,
    ) -> Self {
        let mut block = Self::new();
        block.set_base(CommonStat::Strength.into(), str_val);
        block.set_base(CommonStat::Dexterity.into(), dex_val);
        block.set_base(CommonStat::Intelligence.into(), int_val);
        block.set_base(CommonStat::Vitality.into(), vit_val);
        block.set_base(CommonStat::Luck.into(), luck_val);

        // Derived stats with sensible defaults
        block.set_base(CommonStat::MaxHealth.into(), 100.0 + vit_val * 10.0);
        block.set_base(CommonStat::MaxMana.into(), 50.0 + int_val * 8.0);
        block.set_base(CommonStat::Armor.into(), str_val * 0.5 + dex_val * 0.3);
        block.set_base(CommonStat::AttackPower.into(), str_val * 2.0);
        block.set_base(CommonStat::SpellPower.into(), int_val * 2.0);
        block.set_base(CommonStat::CritChance.into(), 0.05 + luck_val * 0.002);
        block.set_base(CommonStat::CritDamage.into(), 1.5);
        block.set_base(CommonStat::MovementSpeed.into(), 5.0);
        block.set_base(CommonStat::AttackSpeed.into(), 1.0);
        block.set_base(CommonStat::CooldownReduction.into(), 0.0);
        block
    }

    /// Set the base value of a stat (creates if not present).
    pub fn set_base(&mut self, id: StatId, value: f32) {
        self.stats.entry(id).or_insert_with(|| Stat::new(0.0)).base_value = value;
    }

    /// Get an immutable reference to a stat.
    pub fn get(&self, id: &StatId) -> Option<&Stat> {
        self.stats.get(id)
    }

    /// Get a mutable reference to a stat.
    pub fn get_mut(&mut self, id: &StatId) -> Option<&mut Stat> {
        self.stats.get_mut(id)
    }

    /// Get the final (computed) value of a stat. Returns `None` if the stat
    /// doesn't exist.
    pub fn final_value(&mut self, id: &StatId) -> Option<f32> {
        self.stats.get_mut(id).map(|s| s.final_value())
    }

    /// Add a modifier to a specific stat.
    pub fn add_modifier(&mut self, modifier: StatModifier) {
        if let Some(stat) = self.stats.get_mut(&modifier.stat) {
            stat.add_modifier(modifier);
        }
    }

    /// Remove all modifiers from a source across all stats. Returns total removed.
    pub fn remove_source(&mut self, source_id: u64) -> usize {
        self.stats
            .values_mut()
            .map(|s| s.remove_modifiers_by_source(source_id))
            .sum()
    }

    /// Tick all timed modifiers. Returns `true` if any expired.
    pub fn tick(&mut self, dt: f32) -> bool {
        self.stats.values_mut().any(|s| s.tick_modifiers(dt))
    }

    /// List all stat IDs.
    pub fn stat_ids(&self) -> impl Iterator<Item = &StatId> {
        self.stats.keys()
    }

    /// Number of stats in the block.
    pub fn stat_count(&self) -> usize {
        self.stats.len()
    }
}

// ============================================================================
// Skill Tree
// ============================================================================

/// An effect granted when a skill node is unlocked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SkillEffect {
    /// Permanently add a stat modifier.
    StatModifier {
        stat: StatId,
        kind: ModifierKind,
        value: f32,
    },
    /// Unlock an ability (by ID).
    UnlockAbility(String),
    /// Grant a passive perk (by ID).
    GrantPerk(String),
    /// Increase a resource cap (e.g. +20 max health).
    IncreaseResourceCap { resource: String, amount: f32 },
    /// Custom effect identified by a string tag.
    Custom(String),
}

/// A single node in a skill tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillNode {
    /// Unique identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Tooltip / description.
    pub description: String,
    /// Icon asset path.
    pub icon: String,
    /// Skill point cost to unlock.
    pub cost: u32,
    /// Minimum character level required.
    pub required_level: u32,
    /// IDs of prerequisite nodes (all must be unlocked).
    pub prerequisites: Vec<String>,
    /// Effects granted on unlock.
    pub effects: Vec<SkillEffect>,
    /// Maximum rank (for multi-rank skills). 1 = single unlock.
    pub max_rank: u32,
    /// Row/column for UI layout.
    pub row: u32,
    pub column: u32,
}

/// Error type for skill tree operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkillTreeError {
    /// The node ID was not found in the tree.
    NodeNotFound(String),
    /// One or more prerequisites are not met.
    PrerequisitesNotMet(Vec<String>),
    /// Not enough skill points.
    InsufficientSkillPoints { required: u32, available: u32 },
    /// The node is already at max rank.
    AlreadyMaxRank(String),
    /// The character level is too low.
    LevelTooLow { required: u32, current: u32 },
}

impl std::fmt::Display for SkillTreeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "Skill node '{id}' not found"),
            Self::PrerequisitesNotMet(ids) => {
                write!(f, "Prerequisites not met: {}", ids.join(", "))
            }
            Self::InsufficientSkillPoints { required, available } => {
                write!(f, "Need {required} skill points, have {available}")
            }
            Self::AlreadyMaxRank(id) => write!(f, "Skill '{id}' already at max rank"),
            Self::LevelTooLow { required, current } => {
                write!(f, "Requires level {required}, currently {current}")
            }
        }
    }
}

/// A complete skill tree with nodes and player progress.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillTree {
    /// Tree name (e.g. "Warrior", "Mage", "Rogue").
    pub name: String,
    /// All nodes indexed by ID.
    nodes: HashMap<String, SkillNode>,
    /// Current rank of each unlocked node.
    unlocked: HashMap<String, u32>,
    /// Available skill points.
    pub available_points: u32,
}

impl SkillTree {
    /// Create a new empty skill tree.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: HashMap::new(),
            unlocked: HashMap::new(),
            available_points: 0,
        }
    }

    /// Add a node to the tree.
    pub fn add_node(&mut self, node: SkillNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Grant skill points (e.g. on level up).
    pub fn grant_points(&mut self, points: u32) {
        self.available_points += points;
    }

    /// Attempt to unlock (or rank up) a skill node.
    pub fn unlock_skill(
        &mut self,
        node_id: &str,
        player_level: u32,
    ) -> Result<Vec<SkillEffect>, SkillTreeError> {
        // Validate node exists
        let node = self
            .nodes
            .get(node_id)
            .ok_or_else(|| SkillTreeError::NodeNotFound(node_id.to_string()))?
            .clone();

        // Check level requirement
        if player_level < node.required_level {
            return Err(SkillTreeError::LevelTooLow {
                required: node.required_level,
                current: player_level,
            });
        }

        // Check rank
        let current_rank = self.unlocked.get(node_id).copied().unwrap_or(0);
        if current_rank >= node.max_rank {
            return Err(SkillTreeError::AlreadyMaxRank(node_id.to_string()));
        }

        // Check prerequisites
        let unmet: Vec<String> = node
            .prerequisites
            .iter()
            .filter(|pre_id| !self.is_unlocked(pre_id))
            .cloned()
            .collect();
        if !unmet.is_empty() {
            return Err(SkillTreeError::PrerequisitesNotMet(unmet));
        }

        // Check cost
        if self.available_points < node.cost {
            return Err(SkillTreeError::InsufficientSkillPoints {
                required: node.cost,
                available: self.available_points,
            });
        }

        // Spend points and unlock
        self.available_points -= node.cost;
        *self.unlocked.entry(node_id.to_string()).or_insert(0) += 1;

        Ok(node.effects.clone())
    }

    /// Whether a node has been unlocked (at least rank 1).
    pub fn is_unlocked(&self, node_id: &str) -> bool {
        self.unlocked.get(node_id).copied().unwrap_or(0) > 0
    }

    /// Current rank of a node (0 if not unlocked).
    pub fn rank(&self, node_id: &str) -> u32 {
        self.unlocked.get(node_id).copied().unwrap_or(0)
    }

    /// Get all skills whose prerequisites are met (and not already at max rank).
    pub fn get_available_skills(&self, player_level: u32) -> Vec<&SkillNode> {
        self.nodes
            .values()
            .filter(|node| {
                let current_rank = self.unlocked.get(&node.id).copied().unwrap_or(0);
                if current_rank >= node.max_rank {
                    return false;
                }
                if player_level < node.required_level {
                    return false;
                }
                if self.available_points < node.cost {
                    return false;
                }
                node.prerequisites
                    .iter()
                    .all(|pre| self.is_unlocked(pre))
            })
            .collect()
    }

    /// Get all unlocked skill node IDs with their ranks.
    pub fn unlocked_skills(&self) -> &HashMap<String, u32> {
        &self.unlocked
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: &str) -> Option<&SkillNode> {
        self.nodes.get(id)
    }

    /// Total number of nodes in the tree.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total skill points spent.
    pub fn total_points_spent(&self) -> u32 {
        self.unlocked
            .iter()
            .map(|(id, &rank)| {
                self.nodes.get(id).map_or(0, |n| n.cost * rank)
            })
            .sum()
    }

    /// Reset the tree: refund all points, clear unlocked nodes.
    pub fn reset(&mut self) {
        let refund = self.total_points_spent();
        self.unlocked.clear();
        self.available_points += refund;
    }
}

// ============================================================================
// Achievement System
// ============================================================================

/// A condition that must be satisfied to unlock an achievement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AchievementCondition {
    /// Player has killed at least `n` enemies.
    KillCount(u32),
    /// Player has reached at least level `n`.
    ReachLevel(u32),
    /// Player has collected a specific item (by ID).
    CollectItem(String),
    /// Player has completed a specific quest (by ID).
    CompleteQuest(String),
    /// Player has accumulated at least `n` total XP.
    TotalXP(u64),
    /// Player has unlocked a specific skill.
    UnlockSkill(String),
    /// Player's score has reached at least `n`.
    ScoreReached(u64),
    /// Player has died fewer than `n` times.
    DeathsBelow(u32),
    /// Multiple conditions that ALL must be met.
    All(Vec<AchievementCondition>),
    /// At least one of the conditions must be met.
    Any(Vec<AchievementCondition>),
}

/// A reward granted when an achievement is unlocked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AchievementReward {
    /// Grant XP.
    XP(u64),
    /// Grant an item (by ID, with count).
    Item { item_id: String, count: u32 },
    /// Grant skill points.
    SkillPoints(u32),
    /// Grant a cosmetic/title.
    Title(String),
    /// Unlock a feature or area.
    UnlockFeature(String),
    /// No reward (just bragging rights).
    None,
}

/// A single achievement definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Achievement {
    /// Unique identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Description / hint.
    pub description: String,
    /// Icon asset path.
    pub icon: String,
    /// Condition to check.
    pub condition: AchievementCondition,
    /// Reward on unlock.
    pub reward: AchievementReward,
    /// Whether this achievement is hidden until unlocked.
    pub hidden: bool,
}

/// Event emitted when an achievement is unlocked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AchievementUnlockedEvent {
    pub achievement_id: String,
    pub achievement_name: String,
    pub reward: AchievementReward,
}

/// A snapshot of the player's state used for achievement evaluation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlayerProgressState {
    pub kills: u32,
    pub level: u32,
    pub total_xp: u64,
    pub score: u64,
    pub deaths: u32,
    pub collected_items: Vec<String>,
    pub completed_quests: Vec<String>,
    pub unlocked_skills: Vec<String>,
}

/// Manages all achievements and tracks which are unlocked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AchievementSystem {
    achievements: Vec<Achievement>,
    unlocked: HashMap<String, bool>,
    pending_events: Vec<AchievementUnlockedEvent>,
}

impl Default for AchievementSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl AchievementSystem {
    pub fn new() -> Self {
        Self {
            achievements: Vec::new(),
            unlocked: HashMap::new(),
            pending_events: Vec::new(),
        }
    }

    /// Register a new achievement.
    pub fn add_achievement(&mut self, achievement: Achievement) {
        self.unlocked.insert(achievement.id.clone(), false);
        self.achievements.push(achievement);
    }

    /// Check all achievements against the current player state. Returns the
    /// number of newly unlocked achievements.
    pub fn check_achievements(&mut self, state: &PlayerProgressState) -> u32 {
        let mut newly_unlocked = 0u32;

        for achievement in &self.achievements {
            if self.unlocked.get(&achievement.id).copied().unwrap_or(false) {
                continue;
            }

            if Self::evaluate_condition(&achievement.condition, state) {
                self.unlocked.insert(achievement.id.clone(), true);
                newly_unlocked += 1;

                self.pending_events.push(AchievementUnlockedEvent {
                    achievement_id: achievement.id.clone(),
                    achievement_name: achievement.name.clone(),
                    reward: achievement.reward.clone(),
                });
            }
        }

        // We need a separate mutable borrow for unlocked, so update after
        // the immutable iteration above. The code structure above uses
        // self.achievements (immutable) and self.unlocked (via get), then
        // inserts into pending_events. This is fine because we only read
        // `unlocked` in the guard above, and push to pending_events is
        // on a separate field.

        newly_unlocked
    }

    /// Evaluate a single condition against the player state.
    fn evaluate_condition(condition: &AchievementCondition, state: &PlayerProgressState) -> bool {
        match condition {
            AchievementCondition::KillCount(n) => state.kills >= *n,
            AchievementCondition::ReachLevel(n) => state.level >= *n,
            AchievementCondition::CollectItem(id) => state.collected_items.contains(id),
            AchievementCondition::CompleteQuest(id) => state.completed_quests.contains(id),
            AchievementCondition::TotalXP(n) => state.total_xp >= *n,
            AchievementCondition::UnlockSkill(id) => state.unlocked_skills.contains(id),
            AchievementCondition::ScoreReached(n) => state.score >= *n,
            AchievementCondition::DeathsBelow(n) => state.deaths < *n,
            AchievementCondition::All(conditions) => {
                conditions.iter().all(|c| Self::evaluate_condition(c, state))
            }
            AchievementCondition::Any(conditions) => {
                conditions.iter().any(|c| Self::evaluate_condition(c, state))
            }
        }
    }

    /// Drain pending unlock events.
    pub fn drain_events(&mut self) -> Vec<AchievementUnlockedEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Whether an achievement has been unlocked.
    pub fn is_unlocked(&self, id: &str) -> bool {
        self.unlocked.get(id).copied().unwrap_or(false)
    }

    /// Get all visible (non-hidden or unlocked) achievements.
    pub fn visible_achievements(&self) -> Vec<&Achievement> {
        self.achievements
            .iter()
            .filter(|a| !a.hidden || self.is_unlocked(&a.id))
            .collect()
    }

    /// Total number of achievements.
    pub fn total_count(&self) -> usize {
        self.achievements.len()
    }

    /// Number of unlocked achievements.
    pub fn unlocked_count(&self) -> usize {
        self.unlocked.values().filter(|&&v| v).count()
    }

    /// Completion percentage [0, 100].
    pub fn completion_percent(&self) -> f32 {
        if self.achievements.is_empty() {
            return 0.0;
        }
        self.unlocked_count() as f32 / self.achievements.len() as f32 * 100.0
    }

    /// Force-unlock an achievement (e.g. from console commands).
    pub fn force_unlock(&mut self, id: &str) {
        if let Some(achievement) = self.achievements.iter().find(|a| a.id == id) {
            self.unlocked.insert(id.to_string(), true);
            self.pending_events.push(AchievementUnlockedEvent {
                achievement_id: achievement.id.clone(),
                achievement_name: achievement.name.clone(),
                reward: achievement.reward.clone(),
            });
        }
    }

    /// Reset all achievements (for new game / testing).
    pub fn reset(&mut self) {
        for val in self.unlocked.values_mut() {
            *val = false;
        }
        self.pending_events.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- XP Table --

    #[test]
    fn test_xp_table_quadratic() {
        let table = XPTable::default(); // Quadratic, base=100
        assert_eq!(table.xp_for_level(0), 0);
        assert_eq!(table.xp_for_level(1), 100); // 100*1*2/2 = 100
        assert_eq!(table.xp_for_level(2), 300); // 100*2*3/2 = 300
        assert_eq!(table.xp_for_level(3), 600); // 100*3*4/2 = 600
        assert_eq!(table.xp_for_level(10), 5500); // 100*10*11/2 = 5500
    }

    #[test]
    fn test_xp_table_linear() {
        let table = XPTable {
            kind: XPCurveKind::Linear,
            base: 100,
            max_level: 50,
            ..Default::default()
        };
        assert_eq!(table.xp_for_level(1), 100);
        assert_eq!(table.xp_for_level(5), 500);
        assert_eq!(table.xp_for_level(10), 1000);
    }

    #[test]
    fn test_xp_table_steep_quadratic() {
        let table = XPTable {
            kind: XPCurveKind::QuadraticSteep,
            base: 100,
            max_level: 50,
            ..Default::default()
        };
        assert_eq!(table.xp_for_level(1), 100); // 100*1
        assert_eq!(table.xp_for_level(2), 400); // 100*4
        assert_eq!(table.xp_for_level(5), 2500); // 100*25
    }

    #[test]
    fn test_xp_table_exponential() {
        let table = XPTable {
            kind: XPCurveKind::Exponential,
            base: 100,
            ratio: 2.0,
            max_level: 20,
            ..Default::default()
        };
        // xp(1) = 100 * (2^1 - 1)/(2-1) = 100
        assert_eq!(table.xp_for_level(1), 100);
        // xp(2) = 100 * (4 - 1)/1 = 300
        assert_eq!(table.xp_for_level(2), 300);
        // xp(3) = 100 * (8 - 1)/1 = 700
        assert_eq!(table.xp_for_level(3), 700);
    }

    #[test]
    fn test_xp_table_custom() {
        let table = XPTable {
            kind: XPCurveKind::Custom,
            custom_thresholds: vec![100, 350, 800, 1500, 3000],
            max_level: 5,
            ..Default::default()
        };
        assert_eq!(table.xp_for_level(1), 100);
        assert_eq!(table.xp_for_level(3), 800);
        assert_eq!(table.xp_for_level(5), 3000);
    }

    #[test]
    fn test_level_for_xp() {
        let table = XPTable::default();
        assert_eq!(table.level_for_xp(0), 0);
        assert_eq!(table.level_for_xp(99), 0);
        assert_eq!(table.level_for_xp(100), 1);
        assert_eq!(table.level_for_xp(299), 1);
        assert_eq!(table.level_for_xp(300), 2);
        assert_eq!(table.level_for_xp(600), 3);
    }

    #[test]
    fn test_xp_to_next_level() {
        let table = XPTable::default();
        assert_eq!(table.xp_to_next_level(0), 100); // 100 - 0
        assert_eq!(table.xp_to_next_level(1), 200); // 300 - 100
        assert_eq!(table.xp_to_next_level(2), 300); // 600 - 300
    }

    #[test]
    fn test_progress_fraction() {
        let table = XPTable::default();
        assert!((table.progress_fraction(0) - 0.0).abs() < 0.01);
        assert!((table.progress_fraction(50) - 0.5).abs() < 0.01); // 50/100
        assert!((table.progress_fraction(100) - 0.0).abs() < 0.01); // Just hit lvl 1, 0% to lvl 2
    }

    // -- Experience System --

    #[test]
    fn test_grant_xp_and_level_up() {
        let mut sys = ExperienceSystem::new(XPTable::default());
        sys.register(1);

        let levels = sys.grant_xp(1, 100);
        assert_eq!(levels, 1);
        assert_eq!(sys.level(1), Some(1));

        let events = sys.drain_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].new_level, 1);
    }

    #[test]
    fn test_multi_level_up() {
        let mut sys = ExperienceSystem::new(XPTable::default());
        sys.register(1);

        // 600 XP → level 3 (quadratic: 0→100→300→600)
        let levels = sys.grant_xp(1, 600);
        assert_eq!(levels, 3);
        assert_eq!(sys.level(1), Some(3));

        let events = sys.drain_events();
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn test_xp_multiplier() {
        let mut sys = ExperienceSystem::new(XPTable::default());
        sys.register(1);
        sys.xp_multiplier = 2.0;

        sys.grant_xp(1, 50);
        // 50 * 2.0 = 100 XP → level 1
        assert_eq!(sys.level(1), Some(1));
    }

    // -- Stat Block --

    #[test]
    fn test_stat_final_value_no_modifiers() {
        let mut stat = Stat::new(100.0);
        assert!((stat.final_value() - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stat_flat_modifiers() {
        let mut stat = Stat::new(100.0);
        stat.add_modifier(StatModifier {
            source_id: 1,
            stat: StatId::Custom("test".into()),
            kind: ModifierKind::Flat,
            value: 20.0,
            duration: ModifierDuration::Permanent,
            priority: 0,
        });
        stat.add_modifier(StatModifier {
            source_id: 2,
            stat: StatId::Custom("test".into()),
            kind: ModifierKind::Flat,
            value: -5.0,
            duration: ModifierDuration::Permanent,
            priority: 0,
        });
        // (100 + 20 + (-5)) * (1 + 0) = 115
        assert!((stat.final_value() - 115.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stat_percent_modifiers() {
        let mut stat = Stat::new(100.0);
        stat.add_modifier(StatModifier {
            source_id: 1,
            stat: StatId::Custom("test".into()),
            kind: ModifierKind::Percent,
            value: 0.5, // +50%
            duration: ModifierDuration::Permanent,
            priority: 0,
        });
        // (100 + 0) * (1 + 0.5) = 150
        assert!((stat.final_value() - 150.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stat_mixed_modifiers() {
        let mut stat = Stat::new(100.0);
        stat.add_modifier(StatModifier {
            source_id: 1,
            stat: StatId::Custom("test".into()),
            kind: ModifierKind::Flat,
            value: 50.0,
            duration: ModifierDuration::Permanent,
            priority: 0,
        });
        stat.add_modifier(StatModifier {
            source_id: 2,
            stat: StatId::Custom("test".into()),
            kind: ModifierKind::Percent,
            value: 0.2, // +20%
            duration: ModifierDuration::Permanent,
            priority: 0,
        });
        // (100 + 50) * (1 + 0.2) = 180
        assert!((stat.final_value() - 180.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stat_remove_source() {
        let mut stat = Stat::new(100.0);
        stat.add_modifier(StatModifier {
            source_id: 1,
            stat: StatId::Custom("test".into()),
            kind: ModifierKind::Flat,
            value: 50.0,
            duration: ModifierDuration::Permanent,
            priority: 0,
        });
        assert!((stat.final_value() - 150.0).abs() < f32::EPSILON);

        stat.remove_modifiers_by_source(1);
        assert!((stat.final_value() - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stat_timed_modifier_expiry() {
        let mut stat = Stat::new(100.0);
        stat.add_modifier(StatModifier {
            source_id: 1,
            stat: StatId::Custom("test".into()),
            kind: ModifierKind::Flat,
            value: 50.0,
            duration: ModifierDuration::Timed { remaining: 1.0 },
            priority: 0,
        });
        assert!((stat.final_value() - 150.0).abs() < f32::EPSILON);

        // Tick 0.5s — still active
        let expired = stat.tick_modifiers(0.5);
        assert!(!expired);
        assert!((stat.final_value() - 150.0).abs() < f32::EPSILON);

        // Tick 0.6s — expired
        let expired = stat.tick_modifiers(0.6);
        assert!(expired);
        assert!((stat.final_value() - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stat_bounds() {
        let mut stat = Stat::with_bounds(100.0, 0.0, 200.0);
        stat.add_modifier(StatModifier {
            source_id: 1,
            stat: StatId::Custom("test".into()),
            kind: ModifierKind::Flat,
            value: 500.0,
            duration: ModifierDuration::Permanent,
            priority: 0,
        });
        assert!((stat.final_value() - 200.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stat_block_defaults() {
        let mut block = StatBlock::with_defaults(10.0, 8.0, 12.0, 15.0, 5.0);
        let hp = block.final_value(&StatId::Common(CommonStat::MaxHealth));
        // 100 + 15*10 = 250
        assert!((hp.unwrap() - 250.0).abs() < f32::EPSILON);
    }

    // -- Skill Tree --

    #[test]
    fn test_skill_tree_unlock() {
        let mut tree = SkillTree::new("Warrior");
        tree.add_node(SkillNode {
            id: "bash".into(),
            name: "Shield Bash".into(),
            description: "Stun an enemy".into(),
            icon: "icons/bash.png".into(),
            cost: 1,
            required_level: 1,
            prerequisites: vec![],
            effects: vec![SkillEffect::UnlockAbility("bash".into())],
            max_rank: 1,
            row: 0,
            column: 0,
        });
        tree.grant_points(3);

        let effects = tree.unlock_skill("bash", 1).unwrap();
        assert_eq!(effects.len(), 1);
        assert!(tree.is_unlocked("bash"));
        assert_eq!(tree.available_points, 2);
    }

    #[test]
    fn test_skill_tree_prerequisites() {
        let mut tree = SkillTree::new("Mage");
        tree.add_node(SkillNode {
            id: "fireball".into(),
            name: "Fireball".into(),
            description: "Hurl fire".into(),
            icon: "".into(),
            cost: 1,
            required_level: 1,
            prerequisites: vec![],
            effects: vec![],
            max_rank: 1,
            row: 0,
            column: 0,
        });
        tree.add_node(SkillNode {
            id: "meteor".into(),
            name: "Meteor".into(),
            description: "Call down a meteor".into(),
            icon: "".into(),
            cost: 2,
            required_level: 5,
            prerequisites: vec!["fireball".into()],
            effects: vec![],
            max_rank: 1,
            row: 1,
            column: 0,
        });
        tree.grant_points(5);

        // Should fail: prerequisite not met
        let result = tree.unlock_skill("meteor", 5);
        assert!(matches!(result, Err(SkillTreeError::PrerequisitesNotMet(_))));

        // Unlock prerequisite first
        tree.unlock_skill("fireball", 1).unwrap();
        tree.unlock_skill("meteor", 5).unwrap();
        assert!(tree.is_unlocked("meteor"));
    }

    #[test]
    fn test_skill_tree_level_requirement() {
        let mut tree = SkillTree::new("Rogue");
        tree.add_node(SkillNode {
            id: "stealth".into(),
            name: "Stealth".into(),
            description: "Become invisible".into(),
            icon: "".into(),
            cost: 1,
            required_level: 10,
            prerequisites: vec![],
            effects: vec![],
            max_rank: 1,
            row: 0,
            column: 0,
        });
        tree.grant_points(5);

        let result = tree.unlock_skill("stealth", 5);
        assert!(matches!(result, Err(SkillTreeError::LevelTooLow { .. })));

        tree.unlock_skill("stealth", 10).unwrap();
        assert!(tree.is_unlocked("stealth"));
    }

    #[test]
    fn test_skill_tree_max_rank() {
        let mut tree = SkillTree::new("General");
        tree.add_node(SkillNode {
            id: "toughness".into(),
            name: "Toughness".into(),
            description: "+10 HP per rank".into(),
            icon: "".into(),
            cost: 1,
            required_level: 1,
            prerequisites: vec![],
            effects: vec![SkillEffect::IncreaseResourceCap {
                resource: "health".into(),
                amount: 10.0,
            }],
            max_rank: 3,
            row: 0,
            column: 0,
        });
        tree.grant_points(10);

        tree.unlock_skill("toughness", 1).unwrap();
        assert_eq!(tree.rank("toughness"), 1);
        tree.unlock_skill("toughness", 1).unwrap();
        assert_eq!(tree.rank("toughness"), 2);
        tree.unlock_skill("toughness", 1).unwrap();
        assert_eq!(tree.rank("toughness"), 3);

        let result = tree.unlock_skill("toughness", 1);
        assert!(matches!(result, Err(SkillTreeError::AlreadyMaxRank(_))));
    }

    #[test]
    fn test_skill_tree_reset() {
        let mut tree = SkillTree::new("Test");
        tree.add_node(SkillNode {
            id: "a".into(),
            name: "A".into(),
            description: "".into(),
            icon: "".into(),
            cost: 2,
            required_level: 1,
            prerequisites: vec![],
            effects: vec![],
            max_rank: 1,
            row: 0,
            column: 0,
        });
        tree.grant_points(5);
        tree.unlock_skill("a", 1).unwrap();
        assert_eq!(tree.available_points, 3);

        tree.reset();
        assert!(!tree.is_unlocked("a"));
        assert_eq!(tree.available_points, 5);
    }

    // -- Achievement System --

    #[test]
    fn test_achievement_unlock() {
        let mut sys = AchievementSystem::new();
        sys.add_achievement(Achievement {
            id: "first_blood".into(),
            name: "First Blood".into(),
            description: "Kill your first enemy".into(),
            icon: "".into(),
            condition: AchievementCondition::KillCount(1),
            reward: AchievementReward::XP(100),
            hidden: false,
        });

        let state = PlayerProgressState {
            kills: 0,
            ..Default::default()
        };
        assert_eq!(sys.check_achievements(&state), 0);

        let state = PlayerProgressState {
            kills: 1,
            ..Default::default()
        };
        assert_eq!(sys.check_achievements(&state), 1);
        assert!(sys.is_unlocked("first_blood"));

        // Should not unlock again
        assert_eq!(sys.check_achievements(&state), 0);
    }

    #[test]
    fn test_achievement_compound_condition() {
        let mut sys = AchievementSystem::new();
        sys.add_achievement(Achievement {
            id: "perfectionist".into(),
            name: "Perfectionist".into(),
            description: "Reach level 10 with fewer than 3 deaths".into(),
            icon: "".into(),
            condition: AchievementCondition::All(vec![
                AchievementCondition::ReachLevel(10),
                AchievementCondition::DeathsBelow(3),
            ]),
            reward: AchievementReward::Title("Perfectionist".into()),
            hidden: true,
        });

        let state = PlayerProgressState {
            level: 10,
            deaths: 5,
            ..Default::default()
        };
        assert_eq!(sys.check_achievements(&state), 0);

        let state = PlayerProgressState {
            level: 10,
            deaths: 2,
            ..Default::default()
        };
        assert_eq!(sys.check_achievements(&state), 1);
    }

    #[test]
    fn test_achievement_completion_percent() {
        let mut sys = AchievementSystem::new();
        for i in 0..4 {
            sys.add_achievement(Achievement {
                id: format!("ach_{i}"),
                name: format!("Achievement {i}"),
                description: "".into(),
                icon: "".into(),
                condition: AchievementCondition::KillCount(i + 1),
                reward: AchievementReward::None,
                hidden: false,
            });
        }

        let state = PlayerProgressState {
            kills: 2,
            ..Default::default()
        };
        sys.check_achievements(&state);
        // 2 out of 4 unlocked (kill counts 1 and 2)
        assert!((sys.completion_percent() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_achievement_reset() {
        let mut sys = AchievementSystem::new();
        sys.add_achievement(Achievement {
            id: "test".into(),
            name: "Test".into(),
            description: "".into(),
            icon: "".into(),
            condition: AchievementCondition::KillCount(1),
            reward: AchievementReward::None,
            hidden: false,
        });
        sys.force_unlock("test");
        assert!(sys.is_unlocked("test"));

        sys.reset();
        assert!(!sys.is_unlocked("test"));
    }
}
