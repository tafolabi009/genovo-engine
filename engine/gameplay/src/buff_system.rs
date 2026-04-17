// engine/gameplay/src/buff_system.rs
//
// Buff/debuff framework for the Genovo engine.
//
// Provides a flexible buff/debuff system with complex stacking and interaction:
//
// - **Buff stacking rules** -- Stack count, refresh duration, strongest wins.
// - **Buff categories** -- Categorize buffs for interaction rules.
// - **Buff immunity** -- Entities can be immune to specific buff categories.
// - **Buff cleansing** -- Remove buffs by category or specific type.
// - **Buff icons** -- UI display data for buff/debuff icons.
// - **Buff tooltip data** -- Rich tooltip information for UI display.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_STACKS: u32 = 99;
const MAX_BUFFS_PER_ENTITY: usize = 64;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Buff type identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BuffTypeId(pub u32);

/// Buff instance identifier (unique per active buff).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BuffInstanceId(pub u64);

/// Entity identifier for the buff system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BuffEntityId(pub u32);

// ---------------------------------------------------------------------------
// Buff category
// ---------------------------------------------------------------------------

/// Category of a buff, used for interaction rules and cleansing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuffCategory {
    /// Positive effect (healing, speed boost, etc.).
    Buff,
    /// Negative effect (poison, slow, etc.).
    Debuff,
    /// Crowd control (stun, root, silence, etc.).
    CrowdControl,
    /// Healing over time.
    HealOverTime,
    /// Damage over time.
    DamageOverTime,
    /// Stat modifier.
    StatModifier,
    /// Aura (affects nearby allies/enemies).
    Aura,
    /// Shield/barrier.
    Shield,
    /// Transformation.
    Transform,
    /// Custom category.
    Custom(u32),
}

impl fmt::Display for BuffCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buff => write!(f, "Buff"),
            Self::Debuff => write!(f, "Debuff"),
            Self::CrowdControl => write!(f, "CC"),
            Self::HealOverTime => write!(f, "HoT"),
            Self::DamageOverTime => write!(f, "DoT"),
            Self::StatModifier => write!(f, "Stat"),
            Self::Aura => write!(f, "Aura"),
            Self::Shield => write!(f, "Shield"),
            Self::Transform => write!(f, "Transform"),
            Self::Custom(id) => write!(f, "Custom({})", id),
        }
    }
}

// ---------------------------------------------------------------------------
// Stacking behavior
// ---------------------------------------------------------------------------

/// How multiple applications of the same buff type interact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackBehavior {
    /// Each application adds a stack up to max_stacks.
    StackCount { max_stacks: u32 },
    /// Reapplication refreshes the duration to the original value.
    RefreshDuration,
    /// Reapplication extends the duration additively.
    ExtendDuration,
    /// Only the strongest (highest value) instance is kept.
    StrongestWins,
    /// Only the most recent application is kept.
    MostRecent,
    /// Multiple independent instances can coexist.
    Independent,
    /// Cannot be reapplied while active.
    NoStack,
}

impl Default for StackBehavior {
    fn default() -> Self {
        Self::RefreshDuration
    }
}

// ---------------------------------------------------------------------------
// Buff effect
// ---------------------------------------------------------------------------

/// A single stat or gameplay effect applied by a buff.
#[derive(Debug, Clone)]
pub enum BuffEffect {
    /// Modify a stat by a flat amount.
    FlatStat { stat: String, amount: f32 },
    /// Modify a stat by a percentage.
    PercentStat { stat: String, percent: f32 },
    /// Damage over time (per tick).
    DamagePerTick { damage: f32, damage_type: String },
    /// Heal over time (per tick).
    HealPerTick { amount: f32 },
    /// Shield (absorb damage).
    ShieldAmount { amount: f32 },
    /// Speed modifier.
    SpeedMultiplier(f32),
    /// Stun (prevents actions).
    Stun,
    /// Root (prevents movement).
    Root,
    /// Silence (prevents abilities).
    Silence,
    /// Invulnerability.
    Invulnerable,
    /// Stealth/invisibility.
    Stealth { detection_reduction: f32 },
    /// Custom effect with arbitrary data.
    Custom { name: String, value: f32 },
}

impl BuffEffect {
    /// Returns true if this effect is positive (beneficial).
    pub fn is_positive(&self) -> bool {
        matches!(
            self,
            Self::FlatStat { amount, .. } if *amount > 0.0
        ) || matches!(
            self,
            Self::PercentStat { percent, .. } if *percent > 0.0
        ) || matches!(
            self,
            Self::HealPerTick { .. }
                | Self::ShieldAmount { .. }
                | Self::Invulnerable
                | Self::Stealth { .. }
        ) || matches!(
            self,
            Self::SpeedMultiplier(m) if *m > 1.0
        )
    }

    /// Get the primary numeric value of the effect.
    pub fn value(&self) -> f32 {
        match self {
            Self::FlatStat { amount, .. } => *amount,
            Self::PercentStat { percent, .. } => *percent,
            Self::DamagePerTick { damage, .. } => *damage,
            Self::HealPerTick { amount } => *amount,
            Self::ShieldAmount { amount } => *amount,
            Self::SpeedMultiplier(m) => *m,
            Self::Stun | Self::Root | Self::Silence | Self::Invulnerable => 1.0,
            Self::Stealth { detection_reduction } => *detection_reduction,
            Self::Custom { value, .. } => *value,
        }
    }
}

// ---------------------------------------------------------------------------
// Buff definition
// ---------------------------------------------------------------------------

/// Defines a buff type's properties.
#[derive(Debug, Clone)]
pub struct BuffDefinition {
    /// Unique type ID.
    pub id: BuffTypeId,
    /// Display name.
    pub name: String,
    /// Description text.
    pub description: String,
    /// Category.
    pub category: BuffCategory,
    /// Default duration in seconds (0 = permanent until removed).
    pub duration: f32,
    /// Tick interval in seconds (for DoTs/HoTs; 0 = no tick).
    pub tick_interval: f32,
    /// Effects applied by this buff.
    pub effects: Vec<BuffEffect>,
    /// Stacking behavior.
    pub stack_behavior: StackBehavior,
    /// Icon identifier for UI.
    pub icon: String,
    /// Color tint for the icon (RGBA).
    pub icon_color: [f32; 4],
    /// Whether this is a hidden buff (not shown in UI).
    pub hidden: bool,
    /// Whether this buff persists through death.
    pub persists_through_death: bool,
    /// Whether this buff can be cleansed.
    pub cleansable: bool,
    /// Priority for "strongest wins" stacking.
    pub priority: i32,
    /// Per-stack effect multiplier (for StackCount behavior).
    pub per_stack_multiplier: f32,
}

impl BuffDefinition {
    /// Create a new buff definition.
    pub fn new(id: BuffTypeId, name: &str, category: BuffCategory) -> Self {
        Self {
            id,
            name: name.to_string(),
            description: String::new(),
            category,
            duration: 10.0,
            tick_interval: 0.0,
            effects: Vec::new(),
            stack_behavior: StackBehavior::RefreshDuration,
            icon: String::new(),
            icon_color: [1.0, 1.0, 1.0, 1.0],
            hidden: false,
            persists_through_death: false,
            cleansable: true,
            priority: 0,
            per_stack_multiplier: 1.0,
        }
    }

    /// Add an effect.
    pub fn with_effect(mut self, effect: BuffEffect) -> Self {
        self.effects.push(effect);
        self
    }
}

// ---------------------------------------------------------------------------
// Active buff instance
// ---------------------------------------------------------------------------

/// An active buff instance on an entity.
#[derive(Debug, Clone)]
pub struct ActiveBuff {
    /// Instance ID (unique).
    pub instance_id: BuffInstanceId,
    /// Buff type.
    pub buff_type: BuffTypeId,
    /// Source entity (who applied the buff).
    pub source: Option<BuffEntityId>,
    /// Remaining duration (-1 = permanent).
    pub remaining_duration: f32,
    /// Original duration.
    pub original_duration: f32,
    /// Current stack count.
    pub stacks: u32,
    /// Time since last tick.
    pub tick_timer: f32,
    /// Whether the buff is marked for removal.
    pub expired: bool,
    /// Custom data.
    pub custom_data: HashMap<String, f32>,
}

impl ActiveBuff {
    /// Create a new active buff.
    pub fn new(
        instance_id: BuffInstanceId,
        buff_type: BuffTypeId,
        duration: f32,
        source: Option<BuffEntityId>,
    ) -> Self {
        Self {
            instance_id,
            buff_type,
            source,
            remaining_duration: if duration <= 0.0 { -1.0 } else { duration },
            original_duration: duration,
            stacks: 1,
            tick_timer: 0.0,
            expired: false,
            custom_data: HashMap::new(),
        }
    }

    /// Whether this is a permanent (non-expiring) buff.
    pub fn is_permanent(&self) -> bool {
        self.remaining_duration < 0.0
    }

    /// Get the remaining duration fraction (0..1).
    pub fn duration_fraction(&self) -> f32 {
        if self.is_permanent() || self.original_duration <= 0.0 {
            1.0
        } else {
            (self.remaining_duration / self.original_duration).clamp(0.0, 1.0)
        }
    }

    /// Refresh the duration.
    pub fn refresh(&mut self) {
        self.remaining_duration = self.original_duration;
    }

    /// Extend the duration.
    pub fn extend(&mut self, additional: f32) {
        if !self.is_permanent() {
            self.remaining_duration += additional;
        }
    }

    /// Add stacks.
    pub fn add_stacks(&mut self, count: u32, max: u32) {
        self.stacks = (self.stacks + count).min(max);
    }
}

// ---------------------------------------------------------------------------
// Buff tooltip data
// ---------------------------------------------------------------------------

/// Data for rendering a buff tooltip in the UI.
#[derive(Debug, Clone)]
pub struct BuffTooltip {
    pub name: String,
    pub description: String,
    pub category: BuffCategory,
    pub icon: String,
    pub icon_color: [f32; 4],
    pub remaining_seconds: f32,
    pub total_seconds: f32,
    pub stacks: u32,
    pub effects_text: Vec<String>,
    pub source_name: Option<String>,
    pub is_positive: bool,
}

// ---------------------------------------------------------------------------
// Buff events
// ---------------------------------------------------------------------------

/// Events emitted by the buff system.
#[derive(Debug, Clone)]
pub enum BuffEvent {
    /// A buff was applied.
    Applied { entity: BuffEntityId, buff: BuffTypeId, instance: BuffInstanceId },
    /// A buff's stacks changed.
    StacksChanged { entity: BuffEntityId, buff: BuffTypeId, stacks: u32 },
    /// A buff was refreshed.
    Refreshed { entity: BuffEntityId, buff: BuffTypeId },
    /// A buff expired naturally.
    Expired { entity: BuffEntityId, buff: BuffTypeId },
    /// A buff was cleansed/removed.
    Removed { entity: BuffEntityId, buff: BuffTypeId },
    /// A buff tick occurred (DoT/HoT).
    Ticked { entity: BuffEntityId, buff: BuffTypeId, effects: Vec<BuffEffect> },
    /// An application was blocked by immunity.
    Immune { entity: BuffEntityId, buff: BuffTypeId },
}

// ---------------------------------------------------------------------------
// Entity buff state
// ---------------------------------------------------------------------------

/// All buff state for a single entity.
#[derive(Debug, Clone, Default)]
pub struct EntityBuffState {
    /// Active buffs.
    pub buffs: Vec<ActiveBuff>,
    /// Immunities (buff categories this entity is immune to).
    pub immunities: Vec<BuffCategory>,
    /// Temporary immunities with duration.
    pub temp_immunities: Vec<(BuffCategory, f32)>,
}

impl EntityBuffState {
    /// Check if the entity is immune to a category.
    pub fn is_immune(&self, category: BuffCategory) -> bool {
        self.immunities.contains(&category)
            || self.temp_immunities.iter().any(|(c, _)| *c == category)
    }

    /// Find an active buff by type.
    pub fn find_buff(&self, buff_type: BuffTypeId) -> Option<&ActiveBuff> {
        self.buffs.iter().find(|b| b.buff_type == buff_type && !b.expired)
    }

    /// Find a mutable active buff by type.
    pub fn find_buff_mut(&mut self, buff_type: BuffTypeId) -> Option<&mut ActiveBuff> {
        self.buffs.iter_mut().find(|b| b.buff_type == buff_type && !b.expired)
    }

    /// Get the total stack count for a buff type.
    pub fn stack_count(&self, buff_type: BuffTypeId) -> u32 {
        self.buffs
            .iter()
            .filter(|b| b.buff_type == buff_type && !b.expired)
            .map(|b| b.stacks)
            .sum()
    }

    /// Check if any buff applies a specific effect type.
    pub fn has_stun(&self) -> bool {
        // Would need to look up buff definitions; simplified here.
        false
    }
}

// ---------------------------------------------------------------------------
// Buff system
// ---------------------------------------------------------------------------

/// The buff system managing definitions, instances, and updates.
pub struct BuffSystem {
    /// Buff type definitions.
    definitions: HashMap<BuffTypeId, BuffDefinition>,
    /// Per-entity buff states.
    entity_states: HashMap<BuffEntityId, EntityBuffState>,
    /// Next instance ID.
    next_instance_id: u64,
    /// Event queue.
    events: Vec<BuffEvent>,
}

impl BuffSystem {
    /// Create a new buff system.
    pub fn new() -> Self {
        Self {
            definitions: HashMap::new(),
            entity_states: HashMap::new(),
            next_instance_id: 0,
            events: Vec::new(),
        }
    }

    /// Register a buff definition.
    pub fn register(&mut self, definition: BuffDefinition) {
        self.definitions.insert(definition.id, definition);
    }

    /// Get a buff definition.
    pub fn definition(&self, id: BuffTypeId) -> Option<&BuffDefinition> {
        self.definitions.get(&id)
    }

    /// Apply a buff to an entity.
    pub fn apply(
        &mut self,
        entity: BuffEntityId,
        buff_type: BuffTypeId,
        source: Option<BuffEntityId>,
    ) -> Option<BuffInstanceId> {
        let def = self.definitions.get(&buff_type)?.clone();
        let state = self.entity_states.entry(entity).or_default();

        // Check immunity.
        if state.is_immune(def.category) {
            self.events.push(BuffEvent::Immune { entity, buff: buff_type });
            return None;
        }

        // Check max buffs.
        if state.buffs.len() >= MAX_BUFFS_PER_ENTITY {
            return None;
        }

        // Handle stacking.
        match def.stack_behavior {
            StackBehavior::RefreshDuration => {
                if let Some(existing) = state.find_buff_mut(buff_type) {
                    existing.refresh();
                    self.events.push(BuffEvent::Refreshed { entity, buff: buff_type });
                    return Some(existing.instance_id);
                }
            }
            StackBehavior::ExtendDuration => {
                if let Some(existing) = state.find_buff_mut(buff_type) {
                    existing.extend(def.duration);
                    self.events.push(BuffEvent::Refreshed { entity, buff: buff_type });
                    return Some(existing.instance_id);
                }
            }
            StackBehavior::StackCount { max_stacks } => {
                if let Some(existing) = state.find_buff_mut(buff_type) {
                    existing.add_stacks(1, max_stacks.min(MAX_STACKS));
                    existing.refresh();
                    let stacks = existing.stacks;
                    let id = existing.instance_id;
                    self.events.push(BuffEvent::StacksChanged { entity, buff: buff_type, stacks });
                    return Some(id);
                }
            }
            StackBehavior::StrongestWins => {
                if let Some(existing) = state.find_buff(buff_type) {
                    if existing.stacks as i32 >= def.priority {
                        return None; // Existing is stronger.
                    }
                    // Remove existing, apply new below.
                    let instance_id = existing.instance_id;
                    state.buffs.retain(|b| b.instance_id != instance_id);
                }
            }
            StackBehavior::MostRecent => {
                state.buffs.retain(|b| b.buff_type != buff_type || b.expired);
            }
            StackBehavior::NoStack => {
                if state.find_buff(buff_type).is_some() {
                    return None; // Already active.
                }
            }
            StackBehavior::Independent => {
                // Always add a new instance.
            }
        }

        // Create new buff instance.
        let instance_id = BuffInstanceId(self.next_instance_id);
        self.next_instance_id += 1;

        let active = ActiveBuff::new(instance_id, buff_type, def.duration, source);
        state.buffs.push(active);

        self.events.push(BuffEvent::Applied { entity, buff: buff_type, instance: instance_id });

        Some(instance_id)
    }

    /// Remove a specific buff instance.
    pub fn remove(&mut self, entity: BuffEntityId, instance_id: BuffInstanceId) {
        if let Some(state) = self.entity_states.get_mut(&entity) {
            if let Some(buff) = state.buffs.iter_mut().find(|b| b.instance_id == instance_id) {
                buff.expired = true;
                let buff_type = buff.buff_type;
                self.events.push(BuffEvent::Removed { entity, buff: buff_type });
            }
        }
    }

    /// Remove all buffs of a type from an entity.
    pub fn remove_by_type(&mut self, entity: BuffEntityId, buff_type: BuffTypeId) {
        if let Some(state) = self.entity_states.get_mut(&entity) {
            for buff in &mut state.buffs {
                if buff.buff_type == buff_type && !buff.expired {
                    buff.expired = true;
                    self.events.push(BuffEvent::Removed { entity, buff: buff_type });
                }
            }
        }
    }

    /// Cleanse buffs by category.
    pub fn cleanse(&mut self, entity: BuffEntityId, category: BuffCategory, max_count: u32) {
        if let Some(state) = self.entity_states.get_mut(&entity) {
            let mut removed = 0;
            for buff in &mut state.buffs {
                if removed >= max_count {
                    break;
                }
                if buff.expired {
                    continue;
                }
                if let Some(def) = self.definitions.get(&buff.buff_type) {
                    if def.category == category && def.cleansable {
                        buff.expired = true;
                        removed += 1;
                        self.events.push(BuffEvent::Removed {
                            entity,
                            buff: buff.buff_type,
                        });
                    }
                }
            }
        }
    }

    /// Add immunity.
    pub fn add_immunity(&mut self, entity: BuffEntityId, category: BuffCategory) {
        let state = self.entity_states.entry(entity).or_default();
        if !state.immunities.contains(&category) {
            state.immunities.push(category);
        }
    }

    /// Add temporary immunity.
    pub fn add_temp_immunity(&mut self, entity: BuffEntityId, category: BuffCategory, duration: f32) {
        let state = self.entity_states.entry(entity).or_default();
        state.temp_immunities.push((category, duration));
    }

    /// Update all buffs (call each frame).
    pub fn update(&mut self, dt: f32) {
        let entity_ids: Vec<BuffEntityId> = self.entity_states.keys().copied().collect();

        for entity_id in entity_ids {
            if let Some(state) = self.entity_states.get_mut(&entity_id) {
                // Update temp immunities.
                state.temp_immunities.retain_mut(|(_, dur)| {
                    *dur -= dt;
                    *dur > 0.0
                });

                // Update buffs.
                for buff in &mut state.buffs {
                    if buff.expired {
                        continue;
                    }

                    // Duration tick.
                    if !buff.is_permanent() {
                        buff.remaining_duration -= dt;
                        if buff.remaining_duration <= 0.0 {
                            buff.expired = true;
                            self.events.push(BuffEvent::Expired {
                                entity: entity_id,
                                buff: buff.buff_type,
                            });
                            continue;
                        }
                    }

                    // Tick effects.
                    if let Some(def) = self.definitions.get(&buff.buff_type) {
                        if def.tick_interval > 0.0 {
                            buff.tick_timer += dt;
                            if buff.tick_timer >= def.tick_interval {
                                buff.tick_timer -= def.tick_interval;
                                self.events.push(BuffEvent::Ticked {
                                    entity: entity_id,
                                    buff: buff.buff_type,
                                    effects: def.effects.clone(),
                                });
                            }
                        }
                    }
                }

                // Remove expired buffs.
                state.buffs.retain(|b| !b.expired);
            }
        }
    }

    /// Get active buffs for an entity.
    pub fn active_buffs(&self, entity: BuffEntityId) -> &[ActiveBuff] {
        self.entity_states
            .get(&entity)
            .map(|s| s.buffs.as_slice())
            .unwrap_or(&[])
    }

    /// Generate tooltip data for a buff instance.
    pub fn tooltip(&self, entity: BuffEntityId, instance_id: BuffInstanceId) -> Option<BuffTooltip> {
        let state = self.entity_states.get(&entity)?;
        let buff = state.buffs.iter().find(|b| b.instance_id == instance_id)?;
        let def = self.definitions.get(&buff.buff_type)?;

        let effects_text: Vec<String> = def
            .effects
            .iter()
            .map(|e| format!("{:?}", e))
            .collect();

        Some(BuffTooltip {
            name: def.name.clone(),
            description: def.description.clone(),
            category: def.category,
            icon: def.icon.clone(),
            icon_color: def.icon_color,
            remaining_seconds: buff.remaining_duration.max(0.0),
            total_seconds: buff.original_duration,
            stacks: buff.stacks,
            effects_text,
            source_name: None,
            is_positive: def.effects.iter().any(|e| e.is_positive()),
        })
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<BuffEvent> {
        std::mem::take(&mut self.events)
    }

    /// Remove all buff state for an entity.
    pub fn remove_entity(&mut self, entity: BuffEntityId) {
        self.entity_states.remove(&entity);
    }
}

impl Default for BuffSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (BuffSystem, BuffTypeId) {
        let mut sys = BuffSystem::new();
        let id = BuffTypeId(1);
        sys.register(
            BuffDefinition::new(id, "Poison", BuffCategory::DamageOverTime)
                .with_effect(BuffEffect::DamagePerTick {
                    damage: 5.0,
                    damage_type: "poison".into(),
                }),
        );
        (sys, id)
    }

    #[test]
    fn test_apply_buff() {
        let (mut sys, id) = setup();
        let entity = BuffEntityId(1);
        let inst = sys.apply(entity, id, None);
        assert!(inst.is_some());
        assert_eq!(sys.active_buffs(entity).len(), 1);
    }

    #[test]
    fn test_immunity() {
        let (mut sys, id) = setup();
        let entity = BuffEntityId(1);
        sys.add_immunity(entity, BuffCategory::DamageOverTime);
        let inst = sys.apply(entity, id, None);
        assert!(inst.is_none());
    }

    #[test]
    fn test_cleanse() {
        let (mut sys, id) = setup();
        let entity = BuffEntityId(1);
        sys.apply(entity, id, None);
        assert_eq!(sys.active_buffs(entity).len(), 1);
        sys.cleanse(entity, BuffCategory::DamageOverTime, 10);
        sys.update(0.0);
        assert_eq!(sys.active_buffs(entity).len(), 0);
    }

    #[test]
    fn test_duration_expiry() {
        let mut sys = BuffSystem::new();
        let id = BuffTypeId(2);
        let mut def = BuffDefinition::new(id, "Speed", BuffCategory::Buff);
        def.duration = 1.0;
        sys.register(def);

        let entity = BuffEntityId(1);
        sys.apply(entity, id, None);
        assert_eq!(sys.active_buffs(entity).len(), 1);

        sys.update(1.5);
        assert_eq!(sys.active_buffs(entity).len(), 0);
    }

    #[test]
    fn test_stack_count() {
        let mut sys = BuffSystem::new();
        let id = BuffTypeId(3);
        let mut def = BuffDefinition::new(id, "Bleed", BuffCategory::DamageOverTime);
        def.stack_behavior = StackBehavior::StackCount { max_stacks: 5 };
        def.duration = 10.0;
        sys.register(def);

        let entity = BuffEntityId(1);
        sys.apply(entity, id, None);
        sys.apply(entity, id, None);
        sys.apply(entity, id, None);

        let buffs = sys.active_buffs(entity);
        assert_eq!(buffs.len(), 1);
        assert_eq!(buffs[0].stacks, 3);
    }
}
