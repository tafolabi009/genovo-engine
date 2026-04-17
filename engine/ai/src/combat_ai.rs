// engine/ai/src/combat_ai.rs
//
// Combat AI system for the Genovo engine.
//
// Provides intelligent combat behavior for AI-controlled entities:
//
// - Target selection based on threat, distance, health, and priority.
// - Attack pattern management with timing and combos.
// - Dodge and block decision-making.
// - Ability usage AI with cooldown awareness.
// - Retreat conditions based on health and ally status.
// - Group tactics: flanking, focus fire, suppression.
// - Combat role assignment (tank, dps, healer, support).
// - Difficulty scaling for combat parameters.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum targets tracked simultaneously.
const MAX_TRACKED_TARGETS: usize = 16;

/// Default threat decay per second.
const THREAT_DECAY_RATE: f32 = 0.1;

/// Health threshold for considering retreat.
const RETREAT_HEALTH_THRESHOLD: f32 = 0.2;

/// Minimum time between attack decisions.
const MIN_ATTACK_INTERVAL: f32 = 0.5;

/// Maximum simultaneous abilities tracked.
const MAX_ABILITIES: usize = 32;

/// Flanking angle threshold (degrees from behind).
const FLANK_ANGLE_THRESHOLD: f32 = 60.0;

// ---------------------------------------------------------------------------
// Combat Role
// ---------------------------------------------------------------------------

/// Combat role assigned to an AI combatant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CombatRole {
    /// Frontline fighter, absorbs damage.
    Tank,
    /// Damage dealer, prioritizes killing targets.
    Dps,
    /// Healer, keeps allies alive.
    Healer,
    /// Support, buffs/debuffs and crowd control.
    Support,
    /// Ranged attacker, stays at distance.
    Ranged,
    /// Assassin, targets low-health enemies.
    Assassin,
    /// Commander, coordinates group tactics.
    Commander,
}

impl CombatRole {
    /// Preferred engagement range for this role.
    pub fn preferred_range(self) -> f32 {
        match self {
            Self::Tank => 2.0,
            Self::Dps => 3.0,
            Self::Healer => 15.0,
            Self::Support => 10.0,
            Self::Ranged => 20.0,
            Self::Assassin => 2.5,
            Self::Commander => 12.0,
        }
    }

    /// Whether this role should avoid taking damage.
    pub fn is_squishy(self) -> bool {
        matches!(self, Self::Healer | Self::Ranged | Self::Support)
    }

    /// Whether this role can flank.
    pub fn can_flank(self) -> bool {
        matches!(self, Self::Dps | Self::Assassin)
    }
}

// ---------------------------------------------------------------------------
// Target Info
// ---------------------------------------------------------------------------

/// Information about a potential target.
#[derive(Debug, Clone)]
pub struct TargetInfo {
    /// Entity ID of the target.
    pub entity_id: u64,
    /// World position.
    pub position: [f32; 3],
    /// Target's current health (0.0 to 1.0).
    pub health: f32,
    /// Distance from the AI agent.
    pub distance: f32,
    /// Accumulated threat value.
    pub threat: f32,
    /// Whether the target is visible.
    pub visible: bool,
    /// Whether the target is in attack range.
    pub in_range: bool,
    /// Whether the target is currently attacking this agent.
    pub attacking_me: bool,
    /// Target's combat role (if known).
    pub role: Option<CombatRole>,
    /// Priority score (computed from threat, distance, etc.).
    pub priority: f32,
    /// Time since this target was last seen.
    pub time_since_seen: f32,
    /// Whether this target is currently the selected target.
    pub selected: bool,
    /// Whether the target is stunned or incapacitated.
    pub incapacitated: bool,
    /// The target's facing direction.
    pub facing: [f32; 3],
}

impl TargetInfo {
    /// Create a new target info.
    pub fn new(entity_id: u64, position: [f32; 3], health: f32) -> Self {
        Self {
            entity_id,
            position,
            health,
            distance: 0.0,
            threat: 0.0,
            visible: true,
            in_range: false,
            attacking_me: false,
            role: None,
            priority: 0.0,
            time_since_seen: 0.0,
            selected: false,
            incapacitated: false,
            facing: [0.0, 0.0, 1.0],
        }
    }

    /// Check if this is a valid target (visible and alive).
    pub fn is_valid(&self) -> bool {
        self.health > 0.0 && self.visible
    }

    /// Check if we are behind this target (for flanking).
    pub fn is_flanking(&self, attacker_position: [f32; 3]) -> bool {
        let to_attacker = [
            attacker_position[0] - self.position[0],
            attacker_position[1] - self.position[1],
            attacker_position[2] - self.position[2],
        ];
        let len = (to_attacker[0] * to_attacker[0] + to_attacker[1] * to_attacker[1]
            + to_attacker[2] * to_attacker[2]).sqrt();
        if len < 1e-6 { return false; }
        let normalized = [to_attacker[0] / len, to_attacker[1] / len, to_attacker[2] / len];

        let dot = normalized[0] * self.facing[0] + normalized[1] * self.facing[1]
            + normalized[2] * self.facing[2];
        let angle = dot.acos().to_degrees();
        angle > (180.0 - FLANK_ANGLE_THRESHOLD)
    }
}

// ---------------------------------------------------------------------------
// Target Selection
// ---------------------------------------------------------------------------

/// Strategy for selecting targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetSelectionStrategy {
    /// Highest threat.
    HighestThreat,
    /// Closest target.
    Closest,
    /// Lowest health.
    LowestHealth,
    /// Highest priority role (healer > dps > tank).
    RolePriority,
    /// Balanced (weighted combination).
    Balanced,
    /// Focus fire (attack same target as allies).
    FocusFire,
    /// Random.
    Random,
}

/// Configuration for target selection.
#[derive(Debug, Clone)]
pub struct TargetSelectionConfig {
    /// Selection strategy.
    pub strategy: TargetSelectionStrategy,
    /// Weight for threat in balanced mode.
    pub threat_weight: f32,
    /// Weight for distance in balanced mode.
    pub distance_weight: f32,
    /// Weight for health in balanced mode.
    pub health_weight: f32,
    /// Weight for role priority in balanced mode.
    pub role_weight: f32,
    /// Maximum target switch rate (times per second).
    pub max_switch_rate: f32,
    /// Hysteresis: current target gets a bonus to prevent constant switching.
    pub current_target_bonus: f32,
    /// Whether to prefer targets that are attacking this agent.
    pub prefer_attackers: bool,
    /// Whether to prefer incapacitated targets.
    pub prefer_incapacitated: bool,
}

impl Default for TargetSelectionConfig {
    fn default() -> Self {
        Self {
            strategy: TargetSelectionStrategy::Balanced,
            threat_weight: 0.3,
            distance_weight: 0.25,
            health_weight: 0.15,
            role_weight: 0.2,
            max_switch_rate: 2.0,
            current_target_bonus: 0.2,
            prefer_attackers: true,
            prefer_incapacitated: false,
        }
    }
}

/// Select the best target from available targets.
pub fn select_target(
    targets: &[TargetInfo],
    config: &TargetSelectionConfig,
    current_target: Option<u64>,
    agent_position: [f32; 3],
) -> Option<u64> {
    if targets.is_empty() {
        return None;
    }

    let valid_targets: Vec<&TargetInfo> = targets.iter().filter(|t| t.is_valid()).collect();
    if valid_targets.is_empty() {
        return None;
    }

    match config.strategy {
        TargetSelectionStrategy::HighestThreat => {
            valid_targets.iter().max_by(|a, b| a.threat.partial_cmp(&b.threat).unwrap_or(std::cmp::Ordering::Equal))
                .map(|t| t.entity_id)
        }
        TargetSelectionStrategy::Closest => {
            valid_targets.iter().min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal))
                .map(|t| t.entity_id)
        }
        TargetSelectionStrategy::LowestHealth => {
            valid_targets.iter().min_by(|a, b| a.health.partial_cmp(&b.health).unwrap_or(std::cmp::Ordering::Equal))
                .map(|t| t.entity_id)
        }
        TargetSelectionStrategy::Balanced | _ => {
            // Compute priority scores.
            let max_threat = valid_targets.iter().map(|t| t.threat).fold(0.0f32, f32::max).max(1.0);
            let max_dist = valid_targets.iter().map(|t| t.distance).fold(0.0f32, f32::max).max(1.0);

            let mut best_id = valid_targets[0].entity_id;
            let mut best_score = f32::MIN;

            for target in &valid_targets {
                let threat_score = target.threat / max_threat * config.threat_weight;
                let dist_score = (1.0 - target.distance / max_dist) * config.distance_weight;
                let health_score = (1.0 - target.health) * config.health_weight;

                let role_score = match target.role {
                    Some(CombatRole::Healer) => 1.0,
                    Some(CombatRole::Support) => 0.7,
                    Some(CombatRole::Ranged) => 0.5,
                    Some(CombatRole::Dps) => 0.4,
                    _ => 0.2,
                } * config.role_weight;

                let mut score = threat_score + dist_score + health_score + role_score;

                if config.prefer_attackers && target.attacking_me {
                    score += 0.3;
                }

                if current_target == Some(target.entity_id) {
                    score += config.current_target_bonus;
                }

                if score > best_score {
                    best_score = score;
                    best_id = target.entity_id;
                }
            }

            Some(best_id)
        }
    }
}

// ---------------------------------------------------------------------------
// Attack Patterns
// ---------------------------------------------------------------------------

/// A single attack in a pattern.
#[derive(Debug, Clone)]
pub struct AttackMove {
    /// Attack identifier.
    pub id: u32,
    /// Attack name.
    pub name: String,
    /// Damage multiplier.
    pub damage_multiplier: f32,
    /// Range required.
    pub range: f32,
    /// Wind-up time (before damage).
    pub windup: f32,
    /// Recovery time (after attack).
    pub recovery: f32,
    /// Cooldown.
    pub cooldown: f32,
    /// Current cooldown remaining.
    pub current_cooldown: f32,
    /// Whether this is a ranged attack.
    pub is_ranged: bool,
    /// Whether this can be chained from the previous attack.
    pub can_chain: bool,
    /// Animation trigger name.
    pub animation: String,
}

impl AttackMove {
    /// Whether this attack is available (off cooldown and in range).
    pub fn is_available(&self, distance: f32) -> bool {
        self.current_cooldown <= 0.0 && distance <= self.range
    }

    /// Update the cooldown.
    pub fn update_cooldown(&mut self, dt: f32) {
        if self.current_cooldown > 0.0 {
            self.current_cooldown -= dt;
        }
    }

    /// Trigger this attack (starts cooldown).
    pub fn trigger(&mut self) {
        self.current_cooldown = self.cooldown;
    }

    /// Total duration of this attack.
    pub fn total_duration(&self) -> f32 {
        self.windup + self.recovery
    }
}

/// Attack pattern with a sequence of moves.
#[derive(Debug, Clone)]
pub struct AttackPattern {
    /// Pattern name.
    pub name: String,
    /// Moves in sequence.
    pub moves: Vec<AttackMove>,
    /// Current move index.
    pub current_index: usize,
    /// Time in current move.
    pub move_timer: f32,
    /// Whether the pattern is currently executing.
    pub executing: bool,
}

impl AttackPattern {
    /// Create a new attack pattern.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            moves: Vec::new(),
            current_index: 0,
            move_timer: 0.0,
            executing: false,
        }
    }

    /// Add a move to the pattern.
    pub fn add_move(&mut self, attack: AttackMove) {
        self.moves.push(attack);
    }

    /// Start executing the pattern.
    pub fn start(&mut self) {
        self.current_index = 0;
        self.move_timer = 0.0;
        self.executing = true;
    }

    /// Get the current move.
    pub fn current_move(&self) -> Option<&AttackMove> {
        if self.executing {
            self.moves.get(self.current_index)
        } else {
            None
        }
    }

    /// Update the pattern execution.
    pub fn update(&mut self, dt: f32) {
        if !self.executing { return; }

        // Update all cooldowns.
        for m in &mut self.moves {
            m.update_cooldown(dt);
        }

        self.move_timer += dt;
        if let Some(current) = self.moves.get(self.current_index) {
            if self.move_timer >= current.total_duration() {
                self.current_index += 1;
                self.move_timer = 0.0;
                if self.current_index >= self.moves.len() {
                    self.executing = false;
                }
            }
        }
    }

    /// Select the best available attack from this pattern for a given distance.
    pub fn select_attack(&self, distance: f32) -> Option<usize> {
        self.moves.iter().enumerate()
            .filter(|(_, m)| m.is_available(distance))
            .max_by(|(_, a), (_, b)| a.damage_multiplier.partial_cmp(&b.damage_multiplier).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }
}

// ---------------------------------------------------------------------------
// Dodge / Block Decision
// ---------------------------------------------------------------------------

/// Defensive action decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefensiveAction {
    /// Do nothing.
    None,
    /// Block the incoming attack.
    Block,
    /// Dodge to the left.
    DodgeLeft,
    /// Dodge to the right.
    DodgeRight,
    /// Dodge backward.
    DodgeBack,
    /// Roll.
    Roll,
    /// Use a counter-attack.
    Counter,
    /// Use an ability (shield, heal, etc.).
    UseAbility,
}

/// Configuration for defensive decision-making.
#[derive(Debug, Clone)]
pub struct DefensiveConfig {
    /// Base probability of attempting a defensive action.
    pub reaction_chance: f32,
    /// Reaction time in seconds (delay before responding).
    pub reaction_time: f32,
    /// Probability of choosing block over dodge.
    pub block_preference: f32,
    /// Minimum stamina to attempt dodge.
    pub min_stamina_for_dodge: f32,
    /// Minimum stamina to attempt block.
    pub min_stamina_for_block: f32,
    /// Whether to prefer counter-attacks when available.
    pub prefer_counter: bool,
    /// Difficulty scaling (0.0 = easy, 1.0 = hard).
    pub difficulty: f32,
}

impl Default for DefensiveConfig {
    fn default() -> Self {
        Self {
            reaction_chance: 0.5,
            reaction_time: 0.3,
            block_preference: 0.5,
            min_stamina_for_dodge: 0.2,
            min_stamina_for_block: 0.1,
            prefer_counter: false,
            difficulty: 0.5,
        }
    }
}

/// Decide on a defensive action.
pub fn decide_defensive_action(
    config: &DefensiveConfig,
    stamina: f32,
    has_shield: bool,
    can_counter: bool,
    incoming_attack_direction: [f32; 3],
) -> DefensiveAction {
    // Scale reaction chance by difficulty.
    let effective_chance = config.reaction_chance * (0.5 + config.difficulty * 0.5);

    // Simple deterministic pseudo-check using attack direction.
    let hash = incoming_attack_direction[0].abs() + incoming_attack_direction[2].abs();
    let roll = (hash * 7.31).fract();

    if roll > effective_chance {
        return DefensiveAction::None;
    }

    if config.prefer_counter && can_counter && stamina >= config.min_stamina_for_dodge {
        return DefensiveAction::Counter;
    }

    if has_shield && stamina >= config.min_stamina_for_block {
        let block_roll = (hash * 3.17).fract();
        if block_roll < config.block_preference {
            return DefensiveAction::Block;
        }
    }

    if stamina >= config.min_stamina_for_dodge {
        let dir_roll = (hash * 11.23).fract();
        if dir_roll < 0.33 {
            DefensiveAction::DodgeLeft
        } else if dir_roll < 0.66 {
            DefensiveAction::DodgeRight
        } else {
            DefensiveAction::DodgeBack
        }
    } else {
        DefensiveAction::None
    }
}

// ---------------------------------------------------------------------------
// Retreat Logic
// ---------------------------------------------------------------------------

/// Retreat decision result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetreatDecision {
    /// Stay and fight.
    StandGround,
    /// Retreat to a safe position.
    Retreat,
    /// Call for reinforcements then retreat.
    CallAndRetreat,
    /// Seek healing.
    SeekHealing,
    /// Last stand (fight to the death).
    LastStand,
}

/// Retreat configuration.
#[derive(Debug, Clone)]
pub struct RetreatConfig {
    /// Health threshold below which retreat is considered.
    pub health_threshold: f32,
    /// Ally health threshold (retreat if allies are dying).
    pub ally_health_threshold: f32,
    /// Minimum allies required to keep fighting.
    pub min_allies: u32,
    /// Whether this entity can retreat.
    pub can_retreat: bool,
    /// Whether to prefer healing over retreating.
    pub prefer_healing: bool,
    /// Whether to do a last stand when cornered.
    pub last_stand_enabled: bool,
}

impl Default for RetreatConfig {
    fn default() -> Self {
        Self {
            health_threshold: RETREAT_HEALTH_THRESHOLD,
            ally_health_threshold: 0.3,
            min_allies: 0,
            can_retreat: true,
            prefer_healing: true,
            last_stand_enabled: false,
        }
    }
}

/// Evaluate whether to retreat.
pub fn evaluate_retreat(
    health: f32,
    ally_count: u32,
    enemy_count: u32,
    has_escape_route: bool,
    healer_nearby: bool,
    config: &RetreatConfig,
) -> RetreatDecision {
    if !config.can_retreat {
        return RetreatDecision::StandGround;
    }

    if health < config.health_threshold {
        if config.prefer_healing && healer_nearby {
            return RetreatDecision::SeekHealing;
        }

        if !has_escape_route && config.last_stand_enabled {
            return RetreatDecision::LastStand;
        }

        if has_escape_route {
            if ally_count > 0 {
                return RetreatDecision::CallAndRetreat;
            }
            return RetreatDecision::Retreat;
        }

        if config.last_stand_enabled {
            return RetreatDecision::LastStand;
        }

        return RetreatDecision::StandGround;
    }

    if ally_count < config.min_allies && enemy_count > ally_count + 1 {
        return RetreatDecision::Retreat;
    }

    RetreatDecision::StandGround
}

// ---------------------------------------------------------------------------
// Group Tactics
// ---------------------------------------------------------------------------

/// A tactical order for a group of combatants.
#[derive(Debug, Clone)]
pub enum GroupTactic {
    /// All attack the same target.
    FocusFire { target: u64 },
    /// Spread attacks across multiple targets.
    SpreadAttacks,
    /// Flank the target from multiple directions.
    Flank { target: u64, positions: Vec<[f32; 3]> },
    /// Suppress a position with ranged fire.
    Suppress { position: [f32; 3] },
    /// Hold position and defend.
    HoldPosition,
    /// Advance toward the enemy.
    Advance { direction: [f32; 3] },
    /// Retreat in formation.
    Retreat { rally_point: [f32; 3] },
    /// Surround a target.
    Surround { target: u64, radius: f32 },
}

/// Combat AI state for a single entity.
#[derive(Debug)]
pub struct CombatAiState {
    /// Entity ID.
    pub entity_id: u64,
    /// Combat role.
    pub role: CombatRole,
    /// Tracked targets.
    pub targets: Vec<TargetInfo>,
    /// Current selected target.
    pub current_target: Option<u64>,
    /// Target selection config.
    pub target_config: TargetSelectionConfig,
    /// Attack patterns.
    pub attack_patterns: Vec<AttackPattern>,
    /// Active attack pattern index.
    pub active_pattern: usize,
    /// Defensive configuration.
    pub defensive_config: DefensiveConfig,
    /// Retreat configuration.
    pub retreat_config: RetreatConfig,
    /// Current health.
    pub health: f32,
    /// Current stamina.
    pub stamina: f32,
    /// Time since last attack decision.
    pub time_since_attack: f32,
    /// Whether currently in combat.
    pub in_combat: bool,
    /// Current group tactic order.
    pub group_tactic: Option<GroupTactic>,
    /// Difficulty level (0.0 to 1.0).
    pub difficulty: f32,
}

impl CombatAiState {
    /// Create a new combat AI state.
    pub fn new(entity_id: u64, role: CombatRole) -> Self {
        Self {
            entity_id,
            role,
            targets: Vec::new(),
            current_target: None,
            target_config: TargetSelectionConfig::default(),
            attack_patterns: Vec::new(),
            active_pattern: 0,
            defensive_config: DefensiveConfig::default(),
            retreat_config: RetreatConfig::default(),
            health: 1.0,
            stamina: 1.0,
            time_since_attack: 0.0,
            in_combat: false,
            group_tactic: None,
            difficulty: 0.5,
        }
    }

    /// Update target tracking and selection.
    pub fn update_target_selection(&mut self, agent_position: [f32; 3]) {
        self.current_target = select_target(
            &self.targets,
            &self.target_config,
            self.current_target,
            agent_position,
        );
    }

    /// Add or update threat for an entity.
    pub fn add_threat(&mut self, entity_id: u64, threat_amount: f32) {
        if let Some(target) = self.targets.iter_mut().find(|t| t.entity_id == entity_id) {
            target.threat += threat_amount;
        }
    }

    /// Decay threat over time.
    pub fn decay_threat(&mut self, dt: f32) {
        for target in &mut self.targets {
            target.threat = (target.threat - THREAT_DECAY_RATE * dt).max(0.0);
        }
    }

    /// Update combat state.
    pub fn update(&mut self, dt: f32) {
        self.time_since_attack += dt;
        self.decay_threat(dt);

        for pattern in &mut self.attack_patterns {
            pattern.update(dt);
        }

        self.in_combat = self.targets.iter().any(|t| t.is_valid() && t.distance < 50.0);
    }

    /// Get the selected target info.
    pub fn selected_target(&self) -> Option<&TargetInfo> {
        self.current_target.and_then(|id| {
            self.targets.iter().find(|t| t.entity_id == id)
        })
    }
}
