//! Damage, health, status effects, and combat calculations.
//!
//! Provides a flexible damage system suitable for RPGs, action games, and
//! survival games. Features include:
//!
//! - Multiple damage types (Physical, Fire, Ice, Lightning, Poison, Magic, True)
//! - Health component with regeneration, shields, and invulnerability frames
//! - Critical hit calculations
//! - Damage-over-time (DoT) effects
//! - Status effects with duration, stacking, and tick-based processing
//! - Damage events for inter-system communication

use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Damage types
// ---------------------------------------------------------------------------

/// Types of damage that can be dealt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DamageType {
    /// Physical melee/ranged damage (reduced by armor).
    Physical,
    /// Fire damage (can ignite).
    Fire,
    /// Ice/cold damage (can slow/freeze).
    Ice,
    /// Lightning/electric damage (can stun).
    Lightning,
    /// Poison/toxin damage (typically DoT).
    Poison,
    /// Arcane/magical damage (reduced by magic resistance).
    Magic,
    /// True damage (ignores all resistances and shields).
    True,
}

impl DamageType {
    /// Whether this damage type typically applies a status effect.
    pub fn default_status_effect(&self) -> Option<StatusEffectType> {
        match self {
            Self::Fire => Some(StatusEffectType::Burning),
            Self::Ice => Some(StatusEffectType::Frozen),
            Self::Lightning => Some(StatusEffectType::Stunned),
            Self::Poison => Some(StatusEffectType::Poisoned),
            _ => None,
        }
    }

    /// All damage types.
    pub const ALL: &'static [DamageType] = &[
        Self::Physical,
        Self::Fire,
        Self::Ice,
        Self::Lightning,
        Self::Poison,
        Self::Magic,
        Self::True,
    ];
}

// ---------------------------------------------------------------------------
// Damage event
// ---------------------------------------------------------------------------

/// A damage event describing damage dealt from one entity to another.
#[derive(Debug, Clone)]
pub struct DamageEvent {
    /// The entity dealing damage.
    pub source: Option<u32>,
    /// The entity receiving damage.
    pub target: u32,
    /// Base damage amount (before resistances).
    pub base_damage: f32,
    /// Type of damage.
    pub damage_type: DamageType,
    /// Whether this was a critical hit.
    pub is_critical: bool,
    /// Critical damage multiplier (applied if is_critical is true).
    pub crit_multiplier: f32,
    /// World-space position where the damage occurred (for effects).
    pub hit_position: Vec3,
    /// Direction of the hit (for knockback).
    pub hit_direction: Vec3,
    /// Knockback force.
    pub knockback_force: f32,
    /// Whether this damage can be blocked/parried.
    pub blockable: bool,
    /// Whether to apply the default status effect for this damage type.
    pub apply_status: bool,
    /// Custom status effects to apply on hit.
    pub status_effects: Vec<StatusEffect>,
}

impl DamageEvent {
    /// Create a simple damage event.
    pub fn new(target: u32, damage: f32, damage_type: DamageType) -> Self {
        Self {
            source: None,
            target,
            base_damage: damage,
            damage_type,
            is_critical: false,
            crit_multiplier: 2.0,
            hit_position: Vec3::ZERO,
            hit_direction: Vec3::ZERO,
            knockback_force: 0.0,
            blockable: true,
            apply_status: false,
            status_effects: Vec::new(),
        }
    }

    /// Set the source entity.
    pub fn with_source(mut self, source: u32) -> Self {
        self.source = Some(source);
        self
    }

    /// Set as a critical hit.
    pub fn with_crit(mut self, multiplier: f32) -> Self {
        self.is_critical = true;
        self.crit_multiplier = multiplier;
        self
    }

    /// Set hit position and direction.
    pub fn with_hit_info(mut self, position: Vec3, direction: Vec3) -> Self {
        self.hit_position = position;
        self.hit_direction = direction.normalize_or_zero();
        self
    }

    /// Set knockback force.
    pub fn with_knockback(mut self, force: f32) -> Self {
        self.knockback_force = force;
        self
    }

    /// Enable default status effect for this damage type.
    pub fn with_status(mut self) -> Self {
        self.apply_status = true;
        self
    }

    /// Calculate the final damage after all modifiers.
    pub fn calculate_final_damage(&self, resistances: &DamageResistances) -> f32 {
        let mut damage = self.base_damage;

        // Apply crit multiplier.
        if self.is_critical {
            damage *= self.crit_multiplier;
        }

        // Apply resistance (True damage ignores resistances).
        if self.damage_type != DamageType::True {
            let resistance = resistances.get(self.damage_type);
            damage *= (1.0 - resistance).max(0.0);
        }

        damage.max(0.0)
    }
}

// ---------------------------------------------------------------------------
// Damage result
// ---------------------------------------------------------------------------

/// Result of processing a damage event, returned to the caller.
#[derive(Debug, Clone)]
pub struct DamageResult {
    /// Actual damage dealt (after resistances, shields, etc.).
    pub damage_dealt: f32,
    /// Damage absorbed by shields.
    pub shield_absorbed: f32,
    /// Whether the target died from this damage.
    pub killed: bool,
    /// Whether the damage was blocked.
    pub blocked: bool,
    /// Whether invulnerability prevented the damage.
    pub invulnerable: bool,
    /// Whether this was a critical hit.
    pub was_critical: bool,
    /// Overkill amount (damage beyond what was needed to kill).
    pub overkill: f32,
}

impl Default for DamageResult {
    fn default() -> Self {
        Self {
            damage_dealt: 0.0,
            shield_absorbed: 0.0,
            killed: false,
            blocked: false,
            invulnerable: false,
            was_critical: false,
            overkill: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Damage resistances
// ---------------------------------------------------------------------------

/// Per-damage-type resistance values (0.0 = no resistance, 1.0 = immune).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DamageResistances {
    resistances: HashMap<DamageType, f32>,
}

impl DamageResistances {
    /// Create default resistances (all zero).
    pub fn new() -> Self {
        Self {
            resistances: HashMap::new(),
        }
    }

    /// Get resistance for a damage type (0.0 if not set).
    pub fn get(&self, damage_type: DamageType) -> f32 {
        *self.resistances.get(&damage_type).unwrap_or(&0.0)
    }

    /// Set resistance for a damage type. Clamped to [-1.0, 1.0].
    /// Negative values represent vulnerability.
    pub fn set(&mut self, damage_type: DamageType, value: f32) {
        self.resistances.insert(damage_type, value.clamp(-1.0, 1.0));
    }

    /// Add to the existing resistance.
    pub fn add(&mut self, damage_type: DamageType, value: f32) {
        let current = self.get(damage_type);
        self.set(damage_type, current + value);
    }

    /// Reset all resistances to zero.
    pub fn clear(&mut self) {
        self.resistances.clear();
    }
}

impl Default for DamageResistances {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Status effect types
// ---------------------------------------------------------------------------

/// Built-in status effect types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StatusEffectType {
    /// Deals fire damage over time.
    Burning,
    /// Slowed movement and attack speed.
    Frozen,
    /// Cannot act.
    Stunned,
    /// Deals poison damage over time.
    Poisoned,
    /// Deals bleed damage over time.
    Bleeding,
    /// Increased damage taken.
    Weakened,
    /// Reduced damage dealt.
    Silenced,
    /// Increased movement speed.
    Haste,
    /// Increased damage dealt.
    Empowered,
    /// Healing over time.
    Regenerating,
    /// Damage immunity.
    Invulnerable,
    /// Reduced movement speed.
    Slowed,
    /// Cannot be targeted.
    Invisible,
    /// Custom effect with string identifier.
    Custom(u32),
}

impl StatusEffectType {
    /// Whether this effect is beneficial (buff) or harmful (debuff).
    pub fn is_buff(&self) -> bool {
        matches!(
            self,
            Self::Haste
                | Self::Empowered
                | Self::Regenerating
                | Self::Invulnerable
                | Self::Invisible
        )
    }

    /// Whether this effect deals damage over time.
    pub fn is_dot(&self) -> bool {
        matches!(
            self,
            Self::Burning | Self::Poisoned | Self::Bleeding
        )
    }
}

// ---------------------------------------------------------------------------
// Status effect
// ---------------------------------------------------------------------------

/// A status effect applied to an entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusEffect {
    /// Type of effect.
    pub effect_type: StatusEffectType,
    /// Remaining duration in seconds (f32::MAX for permanent effects).
    pub duration: f32,
    /// Maximum duration (for UI display).
    pub max_duration: f32,
    /// Number of stacks (for stackable effects).
    pub stacks: u32,
    /// Maximum stacks.
    pub max_stacks: u32,
    /// Damage per tick (for DoT effects).
    pub damage_per_tick: f32,
    /// Time between ticks (seconds).
    pub tick_interval: f32,
    /// Time since last tick.
    pub time_since_tick: f32,
    /// Damage type of the DoT (if applicable).
    pub dot_damage_type: DamageType,
    /// Source entity that applied this effect.
    pub source: Option<u32>,
    /// Magnitude/power of the effect (interpretation depends on type).
    pub magnitude: f32,
}

impl StatusEffect {
    /// Create a new status effect.
    pub fn new(effect_type: StatusEffectType, duration: f32) -> Self {
        Self {
            effect_type,
            duration,
            max_duration: duration,
            stacks: 1,
            max_stacks: 1,
            damage_per_tick: 0.0,
            tick_interval: 1.0,
            time_since_tick: 0.0,
            dot_damage_type: DamageType::True,
            source: None,
            magnitude: 1.0,
        }
    }

    /// Create a DoT (damage over time) effect.
    pub fn new_dot(
        effect_type: StatusEffectType,
        duration: f32,
        dps: f32,
        damage_type: DamageType,
    ) -> Self {
        Self {
            effect_type,
            duration,
            max_duration: duration,
            stacks: 1,
            max_stacks: 5,
            damage_per_tick: dps,
            tick_interval: 1.0,
            time_since_tick: 0.0,
            dot_damage_type: damage_type,
            source: None,
            magnitude: 1.0,
        }
    }

    /// Set the source entity.
    pub fn with_source(mut self, source: u32) -> Self {
        self.source = Some(source);
        self
    }

    /// Set magnitude.
    pub fn with_magnitude(mut self, magnitude: f32) -> Self {
        self.magnitude = magnitude;
        self
    }

    /// Set max stacks.
    pub fn with_max_stacks(mut self, max: u32) -> Self {
        self.max_stacks = max;
        self
    }

    /// Whether the effect has expired.
    #[inline]
    pub fn is_expired(&self) -> bool {
        self.duration <= 0.0
    }

    /// Whether the effect is permanent (infinite duration).
    #[inline]
    pub fn is_permanent(&self) -> bool {
        self.duration >= f32::MAX * 0.5
    }

    /// Remaining duration as a fraction (0..1).
    #[inline]
    pub fn fraction_remaining(&self) -> f32 {
        if self.max_duration <= 0.0 || self.is_permanent() {
            return 1.0;
        }
        (self.duration / self.max_duration).clamp(0.0, 1.0)
    }

    /// Refresh the effect duration (reset to max).
    pub fn refresh(&mut self) {
        self.duration = self.max_duration;
    }

    /// Add a stack (up to max), refreshing duration.
    pub fn add_stack(&mut self) {
        if self.stacks < self.max_stacks {
            self.stacks += 1;
        }
        self.refresh();
    }

    /// Tick the effect by `dt` seconds. Returns damage dealt this tick, if any.
    pub fn tick(&mut self, dt: f32) -> f32 {
        if self.is_expired() {
            return 0.0;
        }

        self.duration -= dt;
        let mut damage = 0.0;

        // DoT processing.
        if self.damage_per_tick > 0.0 {
            self.time_since_tick += dt;
            while self.time_since_tick >= self.tick_interval {
                self.time_since_tick -= self.tick_interval;
                damage += self.damage_per_tick * self.stacks as f32;
            }
        }

        damage
    }
}

// ---------------------------------------------------------------------------
// Health component
// ---------------------------------------------------------------------------

/// Health component for entities that can take damage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Health {
    /// Current health.
    pub current: f32,
    /// Maximum health.
    pub max: f32,
    /// Health regeneration per second.
    pub regen_rate: f32,
    /// Time since last damage taken (for regen delay).
    pub time_since_damaged: f32,
    /// Delay before regen starts after taking damage.
    pub regen_delay: f32,
    /// Whether the entity is alive.
    pub alive: bool,
    /// Current shield amount (absorbs damage before health).
    pub shield: f32,
    /// Maximum shield amount.
    pub max_shield: f32,
    /// Shield regeneration per second.
    pub shield_regen_rate: f32,
    /// Delay before shield regen starts.
    pub shield_regen_delay: f32,
    /// Invulnerability timer (seconds remaining).
    pub invulnerable_timer: f32,
    /// Duration of invulnerability after taking a hit (i-frames).
    pub invulnerable_after_hit: f32,
    /// Damage resistances.
    pub resistances: DamageResistances,
    /// Active status effects.
    pub status_effects: Vec<StatusEffect>,
    /// Total damage taken (lifetime stat).
    pub total_damage_taken: f32,
    /// Total damage dealt (lifetime stat).
    pub total_damage_dealt: f32,
    /// Number of kills (lifetime stat).
    pub kill_count: u32,
}

impl Health {
    /// Create a new health component.
    pub fn new(max_health: f32) -> Self {
        Self {
            current: max_health,
            max: max_health,
            regen_rate: 0.0,
            time_since_damaged: f32::MAX,
            regen_delay: 5.0,
            alive: true,
            shield: 0.0,
            max_shield: 0.0,
            shield_regen_rate: 0.0,
            shield_regen_delay: 3.0,
            invulnerable_timer: 0.0,
            invulnerable_after_hit: 0.0,
            resistances: DamageResistances::new(),
            status_effects: Vec::new(),
            total_damage_taken: 0.0,
            total_damage_dealt: 0.0,
            kill_count: 0,
        }
    }

    /// Create with shield.
    pub fn with_shield(mut self, max_shield: f32) -> Self {
        self.shield = max_shield;
        self.max_shield = max_shield;
        self
    }

    /// Create with regen.
    pub fn with_regen(mut self, rate: f32, delay: f32) -> Self {
        self.regen_rate = rate;
        self.regen_delay = delay;
        self
    }

    /// Create with i-frames.
    pub fn with_iframes(mut self, duration: f32) -> Self {
        self.invulnerable_after_hit = duration;
        self
    }

    /// Health as a fraction (0..1).
    #[inline]
    pub fn fraction(&self) -> f32 {
        if self.max <= 0.0 {
            return 0.0;
        }
        (self.current / self.max).clamp(0.0, 1.0)
    }

    /// Shield as a fraction (0..1).
    #[inline]
    pub fn shield_fraction(&self) -> f32 {
        if self.max_shield <= 0.0 {
            return 0.0;
        }
        (self.shield / self.max_shield).clamp(0.0, 1.0)
    }

    /// Whether the entity is currently invulnerable.
    #[inline]
    pub fn is_invulnerable(&self) -> bool {
        self.invulnerable_timer > 0.0
            || self
                .status_effects
                .iter()
                .any(|e| e.effect_type == StatusEffectType::Invulnerable && !e.is_expired())
    }

    /// Whether the entity is at full health.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.current >= self.max
    }

    /// Process a damage event and return the result.
    pub fn take_damage(&mut self, event: &DamageEvent) -> DamageResult {
        let mut result = DamageResult::default();
        result.was_critical = event.is_critical;

        // Check invulnerability.
        if self.is_invulnerable() {
            result.invulnerable = true;
            return result;
        }

        // Check if already dead.
        if !self.alive {
            return result;
        }

        // Calculate final damage.
        let mut damage = event.calculate_final_damage(&self.resistances);

        // Shield absorption.
        if self.shield > 0.0 && event.damage_type != DamageType::True {
            let absorbed = damage.min(self.shield);
            self.shield -= absorbed;
            damage -= absorbed;
            result.shield_absorbed = absorbed;
        }

        // Apply damage to health.
        let health_before = self.current;
        self.current = (self.current - damage).max(0.0);
        result.damage_dealt = health_before - self.current + result.shield_absorbed;
        self.total_damage_taken += result.damage_dealt;

        // Check death.
        if self.current <= 0.0 {
            self.alive = false;
            result.killed = true;
            result.overkill = -self.current + damage - (health_before - self.current).max(0.0);
            log::debug!(
                "Entity {} killed (overkill: {:.1})",
                event.target,
                result.overkill
            );
        }

        // Apply i-frames.
        if self.invulnerable_after_hit > 0.0 {
            self.invulnerable_timer = self.invulnerable_after_hit;
        }

        // Reset regen timer.
        self.time_since_damaged = 0.0;

        // Apply status effects from the damage event.
        if event.apply_status {
            if let Some(effect_type) = event.damage_type.default_status_effect() {
                let effect = StatusEffect::new(effect_type, 3.0)
                    .with_magnitude(damage * 0.1);
                self.apply_status_effect(effect);
            }
        }

        for effect in &event.status_effects {
            self.apply_status_effect(effect.clone());
        }

        result
    }

    /// Heal the entity by the given amount.
    pub fn heal(&mut self, amount: f32) -> f32 {
        if !self.alive || amount <= 0.0 {
            return 0.0;
        }
        let before = self.current;
        self.current = (self.current + amount).min(self.max);
        self.current - before
    }

    /// Revive from death with the given health amount.
    pub fn revive(&mut self, health: f32) {
        self.alive = true;
        self.current = health.clamp(1.0, self.max);
        self.invulnerable_timer = 0.0;
        self.status_effects.clear();
        log::debug!("Entity revived with {:.0} health", self.current);
    }

    /// Set max health (scales current proportionally).
    pub fn set_max_health(&mut self, new_max: f32) {
        let ratio = self.fraction();
        self.max = new_max.max(1.0);
        self.current = self.max * ratio;
    }

    /// Add shield.
    pub fn add_shield(&mut self, amount: f32) {
        self.shield = (self.shield + amount).min(self.max_shield);
    }

    /// Apply a status effect, handling stacking and refreshing.
    pub fn apply_status_effect(&mut self, effect: StatusEffect) {
        // Check for existing effect of the same type.
        if let Some(existing) = self
            .status_effects
            .iter_mut()
            .find(|e| e.effect_type == effect.effect_type)
        {
            // Stack or refresh.
            existing.add_stack();
            // Use the higher magnitude.
            if effect.magnitude > existing.magnitude {
                existing.magnitude = effect.magnitude;
                existing.damage_per_tick = effect.damage_per_tick;
            }
        } else {
            self.status_effects.push(effect);
        }
    }

    /// Remove all instances of a status effect type.
    pub fn remove_status_effect(&mut self, effect_type: StatusEffectType) {
        self.status_effects
            .retain(|e| e.effect_type != effect_type);
    }

    /// Check if the entity has a specific status effect.
    pub fn has_status_effect(&self, effect_type: StatusEffectType) -> bool {
        self.status_effects
            .iter()
            .any(|e| e.effect_type == effect_type && !e.is_expired())
    }

    /// Get the number of stacks for a status effect.
    pub fn status_stacks(&self, effect_type: StatusEffectType) -> u32 {
        self.status_effects
            .iter()
            .find(|e| e.effect_type == effect_type && !e.is_expired())
            .map(|e| e.stacks)
            .unwrap_or(0)
    }

    /// Update the health component: process regen, status effects, timers.
    /// Returns a list of DoT damage events to process.
    pub fn update(&mut self, dt: f32) -> Vec<DotDamage> {
        if !self.alive {
            return Vec::new();
        }

        // Update invulnerability timer.
        if self.invulnerable_timer > 0.0 {
            self.invulnerable_timer = (self.invulnerable_timer - dt).max(0.0);
        }

        // Update time since damaged.
        self.time_since_damaged += dt;

        // Health regeneration.
        if self.regen_rate > 0.0
            && self.time_since_damaged >= self.regen_delay
            && self.current < self.max
        {
            self.current = (self.current + self.regen_rate * dt).min(self.max);
        }

        // Shield regeneration.
        if self.shield_regen_rate > 0.0
            && self.time_since_damaged >= self.shield_regen_delay
            && self.shield < self.max_shield
        {
            self.shield = (self.shield + self.shield_regen_rate * dt).min(self.max_shield);
        }

        // Process status effects.
        let mut dot_damages = Vec::new();
        for effect in &mut self.status_effects {
            let tick_damage = effect.tick(dt);
            if tick_damage > 0.0 {
                dot_damages.push(DotDamage {
                    damage: tick_damage,
                    damage_type: effect.dot_damage_type,
                    source: effect.source,
                    effect_type: effect.effect_type,
                });
            }
        }

        // Apply DoT damage directly.
        for dot in &dot_damages {
            self.current = (self.current - dot.damage).max(0.0);
            self.total_damage_taken += dot.damage;
            if self.current <= 0.0 && self.alive {
                self.alive = false;
                log::debug!("Entity killed by {:?} DoT", dot.effect_type);
            }
        }

        // Remove expired effects.
        self.status_effects.retain(|e| !e.is_expired());

        dot_damages
    }

    /// Remove all debuffs.
    pub fn cleanse_debuffs(&mut self) {
        self.status_effects
            .retain(|e| e.effect_type.is_buff());
    }

    /// Remove all buffs.
    pub fn purge_buffs(&mut self) {
        self.status_effects
            .retain(|e| !e.effect_type.is_buff());
    }
}

impl genovo_ecs::Component for Health {}

/// Damage from a DoT (damage-over-time) tick.
#[derive(Debug, Clone)]
pub struct DotDamage {
    /// Damage amount.
    pub damage: f32,
    /// Damage type.
    pub damage_type: DamageType,
    /// Source entity.
    pub source: Option<u32>,
    /// Effect type that caused this damage.
    pub effect_type: StatusEffectType,
}

// ---------------------------------------------------------------------------
// Critical hit calculator
// ---------------------------------------------------------------------------

/// Utility for computing critical hits.
pub struct CritCalculator;

impl CritCalculator {
    /// Roll for a critical hit.
    ///
    /// * `crit_chance` -- probability of a crit (0..1).
    /// * `crit_damage` -- damage multiplier on crit (e.g., 2.0 = double damage).
    /// * `rng` -- random number generator.
    ///
    /// Returns `(is_crit, multiplier)`.
    pub fn roll(
        crit_chance: f32,
        crit_damage: f32,
        rng: &mut genovo_core::Rng,
    ) -> (bool, f32) {
        let is_crit = rng.next_f32() < crit_chance.clamp(0.0, 1.0);
        let multiplier = if is_crit { crit_damage.max(1.0) } else { 1.0 };
        (is_crit, multiplier)
    }

    /// Calculate effective DPS considering crit chance and multiplier.
    pub fn effective_dps(base_dps: f32, crit_chance: f32, crit_multiplier: f32) -> f32 {
        let effective_multiplier = 1.0 + crit_chance.clamp(0.0, 1.0) * (crit_multiplier - 1.0);
        base_dps * effective_multiplier
    }
}

// ---------------------------------------------------------------------------
// Damage number (for UI)
// ---------------------------------------------------------------------------

/// A floating damage number for UI display.
#[derive(Debug, Clone)]
pub struct DamageNumber {
    /// Position in world space.
    pub position: Vec3,
    /// Damage value to display.
    pub value: f32,
    /// Damage type (for coloring).
    pub damage_type: DamageType,
    /// Whether this was a critical hit.
    pub is_critical: bool,
    /// Whether this was healing.
    pub is_heal: bool,
    /// Lifetime remaining (seconds).
    pub lifetime: f32,
    /// Maximum lifetime.
    pub max_lifetime: f32,
    /// Upward velocity for floating animation.
    pub velocity: Vec3,
}

impl DamageNumber {
    /// Create a damage number.
    pub fn new(position: Vec3, value: f32, damage_type: DamageType, is_critical: bool) -> Self {
        Self {
            position,
            value,
            damage_type,
            is_critical,
            is_heal: false,
            lifetime: 1.5,
            max_lifetime: 1.5,
            velocity: Vec3::new(0.0, 2.0, 0.0),
        }
    }

    /// Create a heal number.
    pub fn heal(position: Vec3, value: f32) -> Self {
        Self {
            position,
            value,
            damage_type: DamageType::Magic,
            is_critical: false,
            is_heal: true,
            lifetime: 1.2,
            max_lifetime: 1.2,
            velocity: Vec3::new(0.0, 1.5, 0.0),
        }
    }

    /// Update the damage number. Returns `false` when it should be removed.
    pub fn update(&mut self, dt: f32) -> bool {
        self.lifetime -= dt;
        self.position += self.velocity * dt;
        self.velocity.y -= 3.0 * dt; // Slow down
        self.lifetime > 0.0
    }

    /// Opacity based on remaining lifetime (fades out).
    pub fn opacity(&self) -> f32 {
        if self.max_lifetime <= 0.0 {
            return 0.0;
        }
        (self.lifetime / self.max_lifetime).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_damage() {
        let mut health = Health::new(100.0);
        let event = DamageEvent::new(0, 30.0, DamageType::Physical);
        let result = health.take_damage(&event);

        assert!((result.damage_dealt - 30.0).abs() < 0.01);
        assert!(!result.killed);
        assert!((health.current - 70.0).abs() < 0.01);
    }

    #[test]
    fn lethal_damage() {
        let mut health = Health::new(50.0);
        let event = DamageEvent::new(0, 80.0, DamageType::Physical);
        let result = health.take_damage(&event);

        assert!(result.killed);
        assert!(!health.alive);
        assert!(result.overkill > 0.0);
    }

    #[test]
    fn critical_damage() {
        let mut health = Health::new(100.0);
        let event = DamageEvent::new(0, 20.0, DamageType::Physical).with_crit(2.5);
        let result = health.take_damage(&event);

        assert!(result.was_critical);
        assert!((result.damage_dealt - 50.0).abs() < 0.01); // 20 * 2.5
    }

    #[test]
    fn resistance_reduces_damage() {
        let mut health = Health::new(100.0);
        health.resistances.set(DamageType::Fire, 0.5);

        let event = DamageEvent::new(0, 40.0, DamageType::Fire);
        let result = health.take_damage(&event);

        assert!((result.damage_dealt - 20.0).abs() < 0.01); // 40 * (1 - 0.5)
    }

    #[test]
    fn true_damage_ignores_resistance() {
        let mut health = Health::new(100.0);
        health.resistances.set(DamageType::True, 0.9);

        let event = DamageEvent::new(0, 30.0, DamageType::True);
        let result = health.take_damage(&event);

        assert!((result.damage_dealt - 30.0).abs() < 0.01);
    }

    #[test]
    fn shield_absorbs_damage() {
        let mut health = Health::new(100.0).with_shield(50.0);
        let event = DamageEvent::new(0, 70.0, DamageType::Physical);
        let result = health.take_damage(&event);

        assert!((result.shield_absorbed - 50.0).abs() < 0.01);
        assert!((health.current - 80.0).abs() < 0.01); // 100 - (70 - 50)
        assert!((health.shield - 0.0).abs() < 0.01);
    }

    #[test]
    fn invulnerability_blocks_damage() {
        let mut health = Health::new(100.0).with_iframes(0.5);

        let event = DamageEvent::new(0, 50.0, DamageType::Physical);
        health.take_damage(&event);
        // Should now have i-frames.

        let event2 = DamageEvent::new(0, 50.0, DamageType::Physical);
        let result = health.take_damage(&event2);
        assert!(result.invulnerable);
        assert!((health.current - 50.0).abs() < 0.01);
    }

    #[test]
    fn health_regeneration() {
        let mut health = Health::new(100.0).with_regen(10.0, 2.0);
        health.current = 50.0;
        health.time_since_damaged = 3.0; // Past regen delay.

        health.update(1.0);
        assert!((health.current - 60.0).abs() < 0.01);
    }

    #[test]
    fn regen_delayed_after_damage() {
        let mut health = Health::new(100.0).with_regen(10.0, 2.0);
        health.current = 50.0;

        // Take damage (resets timer).
        let event = DamageEvent::new(0, 10.0, DamageType::Physical);
        health.take_damage(&event);

        // Update 1 second -- should NOT regen (delay is 2s).
        health.update(1.0);
        assert!(
            (health.current - 40.0).abs() < 0.01,
            "Should be 40 after 1s, got {}",
            health.current
        );

        // Update 0.5 more seconds -- still shouldn't regen (1.5s < 2.0s delay).
        health.update(0.5);
        assert!(
            (health.current - 40.0).abs() < 0.01,
            "Should be 40 after 1.5s, got {}",
            health.current
        );

        // Update 1 more second -- NOW should regen (2.5s >= 2.0s delay).
        health.update(1.0);
        assert!(
            health.current > 40.0,
            "Should have regenerated after delay, got {}",
            health.current
        );
    }

    #[test]
    fn heal() {
        let mut health = Health::new(100.0);
        health.current = 30.0;

        let healed = health.heal(50.0);
        assert!((healed - 50.0).abs() < 0.01);
        assert!((health.current - 80.0).abs() < 0.01);
    }

    #[test]
    fn heal_does_not_exceed_max() {
        let mut health = Health::new(100.0);
        health.current = 90.0;

        let healed = health.heal(50.0);
        assert!((healed - 10.0).abs() < 0.01);
        assert!((health.current - 100.0).abs() < 0.01);
    }

    #[test]
    fn revive() {
        let mut health = Health::new(100.0);
        health.current = 0.0;
        health.alive = false;

        health.revive(50.0);
        assert!(health.alive);
        assert!((health.current - 50.0).abs() < 0.01);
    }

    #[test]
    fn status_effect_dot() {
        let mut health = Health::new(100.0);
        let dot = StatusEffect::new_dot(
            StatusEffectType::Burning,
            5.0,
            10.0,
            DamageType::Fire,
        );
        health.apply_status_effect(dot);

        // Tick for 1 second (should trigger one tick at 10 damage).
        let dots = health.update(1.0);
        assert!(!dots.is_empty());
        assert!(health.current < 100.0);
    }

    #[test]
    fn status_effect_stacking() {
        let mut health = Health::new(100.0);
        let dot1 = StatusEffect::new_dot(
            StatusEffectType::Poisoned,
            5.0,
            5.0,
            DamageType::Poison,
        )
        .with_max_stacks(3);
        let dot2 = StatusEffect::new_dot(
            StatusEffectType::Poisoned,
            5.0,
            5.0,
            DamageType::Poison,
        )
        .with_max_stacks(3);

        health.apply_status_effect(dot1);
        health.apply_status_effect(dot2);

        assert_eq!(health.status_stacks(StatusEffectType::Poisoned), 2);
    }

    #[test]
    fn status_effect_expiry() {
        let mut health = Health::new(100.0);
        let effect = StatusEffect::new(StatusEffectType::Stunned, 1.0);
        health.apply_status_effect(effect);

        assert!(health.has_status_effect(StatusEffectType::Stunned));

        // Tick past duration.
        health.update(1.5);
        assert!(!health.has_status_effect(StatusEffectType::Stunned));
    }

    #[test]
    fn cleanse_debuffs() {
        let mut health = Health::new(100.0);
        health.apply_status_effect(StatusEffect::new(StatusEffectType::Stunned, 5.0));
        health.apply_status_effect(StatusEffect::new(StatusEffectType::Haste, 5.0));

        health.cleanse_debuffs();

        assert!(!health.has_status_effect(StatusEffectType::Stunned));
        assert!(health.has_status_effect(StatusEffectType::Haste));
    }

    #[test]
    fn crit_calculator_deterministic() {
        let mut rng = genovo_core::Rng::new(42);
        let (is_crit, mult) = CritCalculator::roll(1.0, 2.0, &mut rng);
        assert!(is_crit);
        assert!((mult - 2.0).abs() < 0.01);

        let (is_crit, mult) = CritCalculator::roll(0.0, 2.0, &mut rng);
        assert!(!is_crit);
        assert!((mult - 1.0).abs() < 0.01);
    }

    #[test]
    fn effective_dps() {
        let edps = CritCalculator::effective_dps(100.0, 0.5, 2.0);
        assert!((edps - 150.0).abs() < 0.01);
    }

    #[test]
    fn damage_number_fades() {
        let mut dn = DamageNumber::new(Vec3::ZERO, 50.0, DamageType::Physical, false);
        assert!((dn.opacity() - 1.0).abs() < 0.01);

        for _ in 0..100 {
            dn.update(1.0 / 60.0);
        }
        assert!(dn.opacity() < 0.5);
    }

    #[test]
    fn vulnerability_increases_damage() {
        let mut health = Health::new(100.0);
        health.resistances.set(DamageType::Fire, -0.5); // 50% vulnerability

        let event = DamageEvent::new(0, 20.0, DamageType::Fire);
        let result = health.take_damage(&event);

        assert!((result.damage_dealt - 30.0).abs() < 0.01); // 20 * 1.5
    }

    #[test]
    fn damage_event_builder() {
        let event = DamageEvent::new(0, 50.0, DamageType::Lightning)
            .with_source(1)
            .with_crit(3.0)
            .with_knockback(10.0)
            .with_status();

        assert_eq!(event.source, Some(1));
        assert!(event.is_critical);
        assert!((event.crit_multiplier - 3.0).abs() < 0.01);
        assert!((event.knockback_force - 10.0).abs() < 0.01);
        assert!(event.apply_status);
    }

    #[test]
    fn status_effect_is_buff() {
        assert!(StatusEffectType::Haste.is_buff());
        assert!(StatusEffectType::Empowered.is_buff());
        assert!(!StatusEffectType::Burning.is_buff());
        assert!(!StatusEffectType::Stunned.is_buff());
    }

    #[test]
    fn status_effect_is_dot() {
        assert!(StatusEffectType::Burning.is_dot());
        assert!(StatusEffectType::Poisoned.is_dot());
        assert!(!StatusEffectType::Stunned.is_dot());
        assert!(!StatusEffectType::Haste.is_dot());
    }
}
