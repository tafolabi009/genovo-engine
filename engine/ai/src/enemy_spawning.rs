//! Enemy spawner AI with difficulty scaling and wave composition.
//!
//! Provides:
//! - **Difficulty curves**: scale enemy count/strength with progression
//! - **Enemy type selection**: choose enemies based on player level
//! - **Spawn budgets**: point-based system for balanced encounters
//! - **Elite spawn conditions**: trigger elites when criteria are met
//! - **Miniboss/boss triggers**: spawn bosses at defined thresholds
//! - **Spawn wave composition**: multi-wave encounters with varied enemies
//! - **Rest periods**: configurable cooldowns between waves
//! - **Dynamic balancing**: adjust spawning based on player performance
//! - **Spawn zones**: spatial regions where enemies appear
//! - **ECS integration**: `SpawnerComponent`, `EnemySpawnSystem`
//!
//! # Design
//!
//! The [`EnemySpawnManager`] uses a budget system where each enemy type costs
//! a number of points. The budget available per wave is determined by the
//! [`DifficultyScaler`] based on player level, progress, and performance.
//! Waves are composed by the [`WaveComposer`] which selects enemy types to
//! fill the budget.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum enemy types in the database.
pub const MAX_ENEMY_TYPES: usize = 128;
/// Maximum waves per encounter.
pub const MAX_WAVES: usize = 20;
/// Maximum spawn zones.
pub const MAX_SPAWN_ZONES: usize = 32;
/// Default rest period between waves (seconds).
pub const DEFAULT_REST_PERIOD: f32 = 10.0;
/// Default budget per wave at level 1.
pub const BASE_BUDGET: u32 = 100;
/// Budget increase per player level.
pub const BUDGET_PER_LEVEL: u32 = 15;
/// Elite spawn chance at base difficulty.
pub const BASE_ELITE_CHANCE: f32 = 0.05;
/// Elite chance increase per wave.
pub const ELITE_CHANCE_PER_WAVE: f32 = 0.02;
/// Miniboss spawn interval (waves).
pub const MINIBOSS_INTERVAL: u32 = 5;
/// Boss spawn interval (waves).
pub const BOSS_INTERVAL: u32 = 10;
/// Maximum simultaneous enemies.
pub const MAX_SIMULTANEOUS_ENEMIES: usize = 30;
/// Performance window size (seconds).
pub const PERFORMANCE_WINDOW: f32 = 60.0;
/// Small epsilon.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// EnemyTypeId
// ---------------------------------------------------------------------------

/// Identifier for an enemy type definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnemyTypeId(pub u32);

// ---------------------------------------------------------------------------
// SpawnerEvent
// ---------------------------------------------------------------------------

/// Events emitted by the spawner system.
#[derive(Debug, Clone)]
pub enum SpawnerEvent {
    /// A wave started.
    WaveStarted { wave_number: u32, budget: u32 },
    /// A wave was completed.
    WaveCompleted { wave_number: u32, time_taken: f32 },
    /// An enemy was spawned.
    EnemySpawned { enemy_type: EnemyTypeId, position: glam::Vec3, is_elite: bool },
    /// A miniboss was spawned.
    MinibossSpawned { enemy_type: EnemyTypeId, position: glam::Vec3 },
    /// A boss was spawned.
    BossSpawned { enemy_type: EnemyTypeId, position: glam::Vec3 },
    /// Rest period started.
    RestStarted { duration: f32 },
    /// Rest period ended.
    RestEnded,
    /// All waves completed.
    EncounterComplete { total_waves: u32, total_time: f32 },
    /// Difficulty adjusted.
    DifficultyAdjusted { new_multiplier: f32 },
}

// ---------------------------------------------------------------------------
// EnemyTier
// ---------------------------------------------------------------------------

/// Tier/rank of an enemy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EnemyTier {
    /// Weak fodder enemies.
    Fodder,
    /// Standard enemies.
    Standard,
    /// Veteran enemies (tougher variants).
    Veteran,
    /// Elite enemies (significantly stronger).
    Elite,
    /// Miniboss (semi-boss encounter).
    Miniboss,
    /// Boss (major encounter).
    Boss,
}

impl EnemyTier {
    /// Budget cost multiplier for this tier.
    pub fn cost_multiplier(&self) -> f32 {
        match self {
            Self::Fodder => 0.5,
            Self::Standard => 1.0,
            Self::Veteran => 1.5,
            Self::Elite => 3.0,
            Self::Miniboss => 10.0,
            Self::Boss => 30.0,
        }
    }

    /// Display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Fodder => "Fodder",
            Self::Standard => "Standard",
            Self::Veteran => "Veteran",
            Self::Elite => "Elite",
            Self::Miniboss => "Miniboss",
            Self::Boss => "Boss",
        }
    }
}

// ---------------------------------------------------------------------------
// EnemyTypeDefinition
// ---------------------------------------------------------------------------

/// Definition of an enemy type for spawning.
#[derive(Debug, Clone)]
pub struct EnemyTypeDefinition {
    /// Type ID.
    pub id: EnemyTypeId,
    /// Name of this enemy type.
    pub name: String,
    /// Spawn cost in budget points.
    pub cost: u32,
    /// Base tier.
    pub tier: EnemyTier,
    /// Minimum player level to encounter.
    pub min_level: u32,
    /// Maximum player level (0 = no maximum).
    pub max_level: u32,
    /// Weight for random selection (higher = more likely).
    pub weight: f32,
    /// Tags for categorization (e.g., "undead", "flying", "ranged").
    pub tags: Vec<String>,
    /// Whether this enemy can be an elite variant.
    pub can_be_elite: bool,
    /// Elite cost modifier.
    pub elite_cost_multiplier: f32,
    /// Group size range (min, max).
    pub group_size: (u32, u32),
    /// Preferred spawn zone tags.
    pub preferred_zones: Vec<String>,
    /// Whether this type is a boss.
    pub is_boss: bool,
    /// Whether this type is a miniboss.
    pub is_miniboss: bool,
    /// Prefab/template reference for spawning.
    pub prefab: String,
}

impl EnemyTypeDefinition {
    /// Create a new enemy type.
    pub fn new(id: EnemyTypeId, name: impl Into<String>, cost: u32, tier: EnemyTier) -> Self {
        Self {
            id,
            name: name.into(),
            cost,
            tier,
            min_level: 1,
            max_level: 0,
            weight: 1.0,
            tags: Vec::new(),
            can_be_elite: tier == EnemyTier::Standard || tier == EnemyTier::Veteran,
            elite_cost_multiplier: 2.0,
            group_size: (1, 1),
            preferred_zones: Vec::new(),
            is_boss: tier == EnemyTier::Boss,
            is_miniboss: tier == EnemyTier::Miniboss,
            prefab: String::new(),
        }
    }

    /// Builder: set level range.
    pub fn with_level_range(mut self, min: u32, max: u32) -> Self {
        self.min_level = min;
        self.max_level = max;
        self
    }

    /// Builder: set weight.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Builder: add tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Builder: set group size.
    pub fn with_group_size(mut self, min: u32, max: u32) -> Self {
        self.group_size = (min, max);
        self
    }

    /// Check if this enemy type is valid for a given player level.
    pub fn valid_for_level(&self, level: u32) -> bool {
        level >= self.min_level && (self.max_level == 0 || level <= self.max_level)
    }

    /// Get the elite cost.
    pub fn elite_cost(&self) -> u32 {
        (self.cost as f32 * self.elite_cost_multiplier) as u32
    }
}

// ---------------------------------------------------------------------------
// DifficultyScaler
// ---------------------------------------------------------------------------

/// Calculates difficulty parameters based on player progression.
#[derive(Debug, Clone)]
pub struct DifficultyScaler {
    /// Base budget at level 1.
    pub base_budget: u32,
    /// Budget increase per level.
    pub budget_per_level: u32,
    /// Dynamic difficulty multiplier (adjusted by performance).
    pub dynamic_multiplier: f32,
    /// Minimum dynamic multiplier.
    pub min_multiplier: f32,
    /// Maximum dynamic multiplier.
    pub max_multiplier: f32,
    /// How quickly the multiplier adjusts.
    pub adjustment_rate: f32,
    /// Player's average kill speed (kills per second).
    pub player_kill_rate: f32,
    /// Target kill rate (considered "balanced").
    pub target_kill_rate: f32,
    /// Player damage taken per second (averaged).
    pub player_damage_rate: f32,
    /// Target damage rate.
    pub target_damage_rate: f32,
    /// Number of player deaths in the current session.
    pub player_deaths: u32,
    /// Wave budget scaling curve.
    pub wave_scaling: DifficultyWaveScaling,
}

/// How budget scales across waves within an encounter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifficultyWaveScaling {
    /// Constant budget each wave.
    Constant,
    /// Linear increase.
    Linear,
    /// Exponential increase.
    Exponential,
    /// Peaks in the middle (bell curve).
    BellCurve,
    /// Crescendo (slow start, big finish).
    Crescendo,
}

impl Default for DifficultyScaler {
    fn default() -> Self {
        Self {
            base_budget: BASE_BUDGET,
            budget_per_level: BUDGET_PER_LEVEL,
            dynamic_multiplier: 1.0,
            min_multiplier: 0.5,
            max_multiplier: 2.0,
            adjustment_rate: 0.1,
            player_kill_rate: 0.0,
            target_kill_rate: 0.5,
            player_damage_rate: 0.0,
            target_damage_rate: 10.0,
            player_deaths: 0,
            wave_scaling: DifficultyWaveScaling::Linear,
        }
    }
}

impl DifficultyScaler {
    /// Calculate the budget for a specific wave.
    pub fn wave_budget(&self, player_level: u32, wave_number: u32, total_waves: u32) -> u32 {
        let base = self.base_budget + self.budget_per_level * player_level;
        let wave_fraction = if total_waves > 1 {
            wave_number as f32 / (total_waves - 1) as f32
        } else {
            0.5
        };

        let wave_multiplier = match self.wave_scaling {
            DifficultyWaveScaling::Constant => 1.0,
            DifficultyWaveScaling::Linear => 0.5 + wave_fraction * 1.0,
            DifficultyWaveScaling::Exponential => (1.0 + wave_fraction).powf(2.0) / 2.0,
            DifficultyWaveScaling::BellCurve => {
                let x = wave_fraction * 2.0 - 1.0;
                (-x * x * 2.0).exp()
            }
            DifficultyWaveScaling::Crescendo => wave_fraction * wave_fraction * 2.0 + 0.3,
        };

        let budget = base as f32 * wave_multiplier * self.dynamic_multiplier;
        budget.max(10.0) as u32
    }

    /// Adjust difficulty based on player performance.
    pub fn adjust(&mut self) {
        // If player is killing too fast, increase difficulty
        if self.player_kill_rate > self.target_kill_rate * 1.5 {
            self.dynamic_multiplier += self.adjustment_rate;
        }
        // If player is dying, decrease difficulty
        if self.player_deaths > 0 {
            self.dynamic_multiplier -= self.adjustment_rate * 2.0;
            self.player_deaths = 0;
        }
        // If player is taking too much damage, ease off
        if self.player_damage_rate > self.target_damage_rate * 2.0 {
            self.dynamic_multiplier -= self.adjustment_rate * 0.5;
        }

        self.dynamic_multiplier = self.dynamic_multiplier.clamp(self.min_multiplier, self.max_multiplier);
    }

    /// Get the elite spawn chance for a given wave.
    pub fn elite_chance(&self, wave_number: u32) -> f32 {
        (BASE_ELITE_CHANCE + ELITE_CHANCE_PER_WAVE * wave_number as f32)
            .min(0.3) * self.dynamic_multiplier
    }

    /// Check if a miniboss should spawn on this wave.
    pub fn should_spawn_miniboss(&self, wave_number: u32) -> bool {
        wave_number > 0 && wave_number % MINIBOSS_INTERVAL == 0
    }

    /// Check if a boss should spawn on this wave.
    pub fn should_spawn_boss(&self, wave_number: u32) -> bool {
        wave_number > 0 && wave_number % BOSS_INTERVAL == 0
    }
}

// ---------------------------------------------------------------------------
// SpawnZone
// ---------------------------------------------------------------------------

/// A region where enemies can spawn.
#[derive(Debug, Clone)]
pub struct SpawnZone {
    /// Zone center.
    pub center: glam::Vec3,
    /// Zone radius.
    pub radius: f32,
    /// Tags for matching enemy types.
    pub tags: Vec<String>,
    /// Whether the zone is currently active.
    pub active: bool,
    /// Maximum enemies spawned from this zone.
    pub max_enemies: usize,
    /// Current number of living enemies from this zone.
    pub current_enemies: usize,
    /// Whether enemies spawn at the zone edge (true) or randomly within (false).
    pub spawn_at_edge: bool,
    /// Height offset for spawn position.
    pub height_offset: f32,
}

impl SpawnZone {
    /// Create a new spawn zone.
    pub fn new(center: glam::Vec3, radius: f32) -> Self {
        Self {
            center,
            radius,
            tags: Vec::new(),
            active: true,
            max_enemies: MAX_SIMULTANEOUS_ENEMIES,
            current_enemies: 0,
            spawn_at_edge: false,
            height_offset: 0.0,
        }
    }

    /// Generate a random spawn position within the zone.
    pub fn random_position(&self, rng_seed: u32) -> glam::Vec3 {
        // Simple pseudo-random using seed
        let angle = (rng_seed as f32 * 2.399963) % (std::f32::consts::TAU);
        let dist = if self.spawn_at_edge {
            self.radius
        } else {
            let r = ((rng_seed.wrapping_mul(1103515245).wrapping_add(12345) >> 16) & 0x7FFF) as f32 / 32767.0;
            r.sqrt() * self.radius
        };
        glam::Vec3::new(
            self.center.x + angle.cos() * dist,
            self.center.y + self.height_offset,
            self.center.z + angle.sin() * dist,
        )
    }

    /// Check if the zone can accept more enemies.
    pub fn can_spawn(&self) -> bool {
        self.active && self.current_enemies < self.max_enemies
    }
}

// ---------------------------------------------------------------------------
// SpawnRequest
// ---------------------------------------------------------------------------

/// A request to spawn an enemy.
#[derive(Debug, Clone)]
pub struct EnemySpawnRequest {
    /// Enemy type to spawn.
    pub enemy_type: EnemyTypeId,
    /// Spawn position.
    pub position: glam::Vec3,
    /// Whether this is an elite variant.
    pub is_elite: bool,
    /// Whether this is a boss/miniboss.
    pub is_boss: bool,
    /// Wave number this enemy belongs to.
    pub wave_number: u32,
    /// Group ID (enemies in the same group).
    pub group_id: u32,
}

// ---------------------------------------------------------------------------
// WaveDefinition
// ---------------------------------------------------------------------------

/// Definition of a single wave in an encounter.
#[derive(Debug, Clone)]
pub struct SpawnWaveDefinition {
    /// Wave number.
    pub wave_number: u32,
    /// Budget for this wave.
    pub budget: u32,
    /// Spawn requests (filled by the composer).
    pub spawn_requests: Vec<EnemySpawnRequest>,
    /// Rest period after this wave (seconds).
    pub rest_period: f32,
    /// Whether this wave has a miniboss.
    pub has_miniboss: bool,
    /// Whether this wave has a boss.
    pub has_boss: bool,
    /// Special conditions for this wave.
    pub conditions: Vec<String>,
}

// ---------------------------------------------------------------------------
// WaveComposer
// ---------------------------------------------------------------------------

/// Composes waves by selecting enemy types to fill a budget.
#[derive(Debug)]
pub struct WaveComposer {
    /// Enemy type database.
    enemy_types: HashMap<EnemyTypeId, EnemyTypeDefinition>,
    /// RNG state.
    rng_state: u32,
}

impl WaveComposer {
    /// Create a new wave composer.
    pub fn new() -> Self {
        Self {
            enemy_types: HashMap::new(),
            rng_state: 42,
        }
    }

    /// Register an enemy type.
    pub fn register_enemy_type(&mut self, def: EnemyTypeDefinition) {
        self.enemy_types.insert(def.id, def);
    }

    /// Get an enemy type definition.
    pub fn get_enemy_type(&self, id: EnemyTypeId) -> Option<&EnemyTypeDefinition> {
        self.enemy_types.get(&id)
    }

    /// Simple LCG random.
    fn random(&mut self) -> f32 {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.rng_state >> 16) & 0x7FFF) as f32 / 32767.0
    }

    /// Random u32.
    fn random_u32(&mut self) -> u32 {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        self.rng_state
    }

    /// Compose a wave given a budget, player level, and difficulty settings.
    pub fn compose_wave(
        &mut self,
        wave_number: u32,
        budget: u32,
        player_level: u32,
        elite_chance: f32,
        spawn_miniboss: bool,
        spawn_boss: bool,
        zones: &[SpawnZone],
    ) -> SpawnWaveDefinition {
        let mut remaining_budget = budget;
        let mut spawn_requests = Vec::new();
        let mut group_id = 0u32;

        // Get valid enemy types for this level
        let valid_types: Vec<&EnemyTypeDefinition> = self.enemy_types.values()
            .filter(|t| t.valid_for_level(player_level) && !t.is_boss && !t.is_miniboss)
            .collect();

        // Spawn boss if needed
        if spawn_boss {
            let boss_types: Vec<&EnemyTypeDefinition> = self.enemy_types.values()
                .filter(|t| t.is_boss && t.valid_for_level(player_level))
                .collect();
            if let Some(boss) = boss_types.first() {
                let zone_idx = (self.random_u32() as usize) % zones.len().max(1);
                let pos = if !zones.is_empty() {
                    zones[zone_idx].random_position(self.random_u32())
                } else {
                    glam::Vec3::ZERO
                };
                spawn_requests.push(EnemySpawnRequest {
                    enemy_type: boss.id,
                    position: pos,
                    is_elite: false,
                    is_boss: true,
                    wave_number,
                    group_id,
                });
                remaining_budget = remaining_budget.saturating_sub(boss.cost);
                group_id += 1;
            }
        }

        // Spawn miniboss if needed
        if spawn_miniboss && !spawn_boss {
            let miniboss_types: Vec<&EnemyTypeDefinition> = self.enemy_types.values()
                .filter(|t| t.is_miniboss && t.valid_for_level(player_level))
                .collect();
            if let Some(mb) = miniboss_types.first() {
                let zone_idx = (self.random_u32() as usize) % zones.len().max(1);
                let pos = if !zones.is_empty() {
                    zones[zone_idx].random_position(self.random_u32())
                } else {
                    glam::Vec3::ZERO
                };
                spawn_requests.push(EnemySpawnRequest {
                    enemy_type: mb.id,
                    position: pos,
                    is_elite: false,
                    is_boss: true,
                    wave_number,
                    group_id,
                });
                remaining_budget = remaining_budget.saturating_sub(mb.cost);
                group_id += 1;
            }
        }

        // Fill remaining budget with regular enemies
        let mut attempts = 0;
        while remaining_budget > 0 && attempts < 100 && spawn_requests.len() < MAX_SIMULTANEOUS_ENEMIES {
            attempts += 1;

            // Weighted random selection
            let affordable: Vec<&EnemyTypeDefinition> = valid_types.iter()
                .filter(|t| t.cost <= remaining_budget)
                .copied()
                .collect();
            if affordable.is_empty() {
                break;
            }

            let total_weight: f32 = affordable.iter().map(|t| t.weight).sum();
            let mut r = self.random() * total_weight;
            let mut selected = affordable[0];
            for t in &affordable {
                r -= t.weight;
                if r <= 0.0 {
                    selected = t;
                    break;
                }
            }

            // Check for elite
            let is_elite = selected.can_be_elite && self.random() < elite_chance;
            let cost = if is_elite { selected.elite_cost() } else { selected.cost };

            if cost > remaining_budget {
                continue;
            }

            // Determine group size
            let group_size = if selected.group_size.0 == selected.group_size.1 {
                selected.group_size.0
            } else {
                let range = selected.group_size.1 - selected.group_size.0;
                selected.group_size.0 + (self.random_u32() % (range + 1))
            };

            let actual_group = group_size.min((remaining_budget / cost).max(1));

            for _ in 0..actual_group {
                if remaining_budget < cost {
                    break;
                }
                let zone_idx = (self.random_u32() as usize) % zones.len().max(1);
                let pos = if !zones.is_empty() {
                    zones[zone_idx].random_position(self.random_u32())
                } else {
                    glam::Vec3::ZERO
                };

                spawn_requests.push(EnemySpawnRequest {
                    enemy_type: selected.id,
                    position: pos,
                    is_elite,
                    is_boss: false,
                    wave_number,
                    group_id,
                });
                remaining_budget = remaining_budget.saturating_sub(cost);
            }
            group_id += 1;
        }

        SpawnWaveDefinition {
            wave_number,
            budget,
            spawn_requests,
            rest_period: DEFAULT_REST_PERIOD,
            has_miniboss: spawn_miniboss,
            has_boss: spawn_boss,
            conditions: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// EncounterState
// ---------------------------------------------------------------------------

/// State of an ongoing encounter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncounterState {
    /// Encounter has not started.
    Idle,
    /// Spawning enemies for the current wave.
    Spawning,
    /// Waiting for the current wave to be cleared.
    WaveActive,
    /// Rest period between waves.
    Resting,
    /// All waves completed.
    Complete,
}

// ---------------------------------------------------------------------------
// EnemySpawnManager
// ---------------------------------------------------------------------------

/// Manages enemy spawning for encounters.
pub struct EnemySpawnManager {
    /// Wave composer.
    pub composer: WaveComposer,
    /// Difficulty scaler.
    pub difficulty: DifficultyScaler,
    /// Spawn zones.
    pub zones: Vec<SpawnZone>,
    /// Current encounter state.
    pub state: EncounterState,
    /// Current wave number.
    pub current_wave: u32,
    /// Total waves in current encounter.
    pub total_waves: u32,
    /// Current wave definition.
    current_wave_def: Option<SpawnWaveDefinition>,
    /// Spawn index within current wave.
    spawn_index: usize,
    /// Enemies alive from current wave.
    pub enemies_alive: u32,
    /// Total enemies spawned.
    pub total_spawned: u32,
    /// Total enemies killed.
    pub total_killed: u32,
    /// Rest timer.
    rest_timer: f32,
    /// Total encounter time.
    encounter_time: f32,
    /// Wave start time.
    wave_start_time: f32,
    /// Events.
    events: Vec<SpawnerEvent>,
    /// Player level (for difficulty calculation).
    pub player_level: u32,
    /// Spawn requests waiting to be processed.
    pending_spawns: Vec<EnemySpawnRequest>,
}

impl EnemySpawnManager {
    /// Create a new spawn manager.
    pub fn new() -> Self {
        Self {
            composer: WaveComposer::new(),
            difficulty: DifficultyScaler::default(),
            zones: Vec::new(),
            state: EncounterState::Idle,
            current_wave: 0,
            total_waves: 5,
            current_wave_def: None,
            spawn_index: 0,
            enemies_alive: 0,
            total_spawned: 0,
            total_killed: 0,
            rest_timer: 0.0,
            encounter_time: 0.0,
            wave_start_time: 0.0,
            events: Vec::new(),
            player_level: 1,
            pending_spawns: Vec::new(),
        }
    }

    /// Add a spawn zone.
    pub fn add_zone(&mut self, zone: SpawnZone) {
        self.zones.push(zone);
    }

    /// Start an encounter with N waves.
    pub fn start_encounter(&mut self, total_waves: u32) {
        self.total_waves = total_waves;
        self.current_wave = 0;
        self.state = EncounterState::Idle;
        self.enemies_alive = 0;
        self.total_spawned = 0;
        self.total_killed = 0;
        self.encounter_time = 0.0;

        self.start_next_wave();
    }

    /// Start the next wave.
    fn start_next_wave(&mut self) {
        if self.current_wave >= self.total_waves {
            self.state = EncounterState::Complete;
            self.events.push(SpawnerEvent::EncounterComplete {
                total_waves: self.total_waves,
                total_time: self.encounter_time,
            });
            return;
        }

        let budget = self.difficulty.wave_budget(
            self.player_level,
            self.current_wave,
            self.total_waves,
        );
        let elite_chance = self.difficulty.elite_chance(self.current_wave);
        let spawn_miniboss = self.difficulty.should_spawn_miniboss(self.current_wave);
        let spawn_boss = self.difficulty.should_spawn_boss(self.current_wave);

        let wave_def = self.composer.compose_wave(
            self.current_wave,
            budget,
            self.player_level,
            elite_chance,
            spawn_miniboss,
            spawn_boss,
            &self.zones,
        );

        self.events.push(SpawnerEvent::WaveStarted {
            wave_number: self.current_wave,
            budget,
        });

        self.pending_spawns = wave_def.spawn_requests.clone();
        self.current_wave_def = Some(wave_def);
        self.spawn_index = 0;
        self.wave_start_time = self.encounter_time;
        self.state = EncounterState::Spawning;
    }

    /// Update the spawn manager.
    pub fn update(&mut self, dt: f32) {
        self.events.clear();
        self.encounter_time += dt;

        match self.state {
            EncounterState::Idle => {}
            EncounterState::Spawning => {
                // Spawn enemies from pending list
                while !self.pending_spawns.is_empty() {
                    let request = self.pending_spawns.remove(0);
                    self.enemies_alive += 1;
                    self.total_spawned += 1;

                    if request.is_boss {
                        self.events.push(SpawnerEvent::BossSpawned {
                            enemy_type: request.enemy_type,
                            position: request.position,
                        });
                    } else {
                        self.events.push(SpawnerEvent::EnemySpawned {
                            enemy_type: request.enemy_type,
                            position: request.position,
                            is_elite: request.is_elite,
                        });
                    }
                }
                self.state = EncounterState::WaveActive;
            }
            EncounterState::WaveActive => {
                if self.enemies_alive == 0 {
                    let wave_time = self.encounter_time - self.wave_start_time;
                    self.events.push(SpawnerEvent::WaveCompleted {
                        wave_number: self.current_wave,
                        time_taken: wave_time,
                    });

                    self.current_wave += 1;

                    if self.current_wave >= self.total_waves {
                        self.state = EncounterState::Complete;
                        self.events.push(SpawnerEvent::EncounterComplete {
                            total_waves: self.total_waves,
                            total_time: self.encounter_time,
                        });
                    } else {
                        let rest = self.current_wave_def.as_ref()
                            .map(|w| w.rest_period)
                            .unwrap_or(DEFAULT_REST_PERIOD);
                        self.rest_timer = rest;
                        self.state = EncounterState::Resting;
                        self.events.push(SpawnerEvent::RestStarted { duration: rest });
                    }

                    // Adjust difficulty
                    self.difficulty.adjust();
                }
            }
            EncounterState::Resting => {
                self.rest_timer -= dt;
                if self.rest_timer <= 0.0 {
                    self.events.push(SpawnerEvent::RestEnded);
                    self.start_next_wave();
                }
            }
            EncounterState::Complete => {}
        }
    }

    /// Notify that an enemy was killed.
    pub fn on_enemy_killed(&mut self) {
        if self.enemies_alive > 0 {
            self.enemies_alive -= 1;
        }
        self.total_killed += 1;
    }

    /// Get events from last update.
    pub fn events(&self) -> &[SpawnerEvent] {
        &self.events
    }

    /// Get stats.
    pub fn stats(&self) -> SpawnerStats {
        SpawnerStats {
            state: self.state,
            current_wave: self.current_wave,
            total_waves: self.total_waves,
            enemies_alive: self.enemies_alive,
            total_spawned: self.total_spawned,
            total_killed: self.total_killed,
            encounter_time: self.encounter_time,
            difficulty_multiplier: self.difficulty.dynamic_multiplier,
        }
    }
}

/// Statistics for the spawner.
#[derive(Debug, Clone)]
pub struct SpawnerStats {
    pub state: EncounterState,
    pub current_wave: u32,
    pub total_waves: u32,
    pub enemies_alive: u32,
    pub total_spawned: u32,
    pub total_killed: u32,
    pub encounter_time: f32,
    pub difficulty_multiplier: f32,
}

// ---------------------------------------------------------------------------
// ECS Component
// ---------------------------------------------------------------------------

/// ECS component for spawn zone entities.
#[derive(Debug, Clone)]
pub struct SpawnerComponent {
    /// Zone index in the spawn manager.
    pub zone_index: usize,
    /// Whether auto-spawn is enabled.
    pub auto_spawn: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_difficulty_scaling() {
        let scaler = DifficultyScaler::default();
        let budget_1 = scaler.wave_budget(1, 0, 5);
        let budget_10 = scaler.wave_budget(10, 0, 5);
        assert!(budget_10 > budget_1);
    }

    #[test]
    fn test_wave_composition() {
        let mut composer = WaveComposer::new();
        composer.register_enemy_type(
            EnemyTypeDefinition::new(EnemyTypeId(1), "Goblin", 10, EnemyTier::Fodder)
                .with_weight(2.0)
        );
        composer.register_enemy_type(
            EnemyTypeDefinition::new(EnemyTypeId(2), "Orc", 25, EnemyTier::Standard)
                .with_weight(1.0)
        );

        let zone = SpawnZone::new(glam::Vec3::ZERO, 20.0);
        let wave = composer.compose_wave(0, 100, 1, 0.0, false, false, &[zone]);
        assert!(!wave.spawn_requests.is_empty());
    }

    #[test]
    fn test_enemy_type_level_filter() {
        let def = EnemyTypeDefinition::new(EnemyTypeId(1), "Dragon", 100, EnemyTier::Boss)
            .with_level_range(10, 0);
        assert!(!def.valid_for_level(5));
        assert!(def.valid_for_level(10));
        assert!(def.valid_for_level(50));
    }

    #[test]
    fn test_encounter_lifecycle() {
        let mut manager = EnemySpawnManager::new();
        manager.composer.register_enemy_type(
            EnemyTypeDefinition::new(EnemyTypeId(1), "Goblin", 10, EnemyTier::Fodder)
        );
        manager.add_zone(SpawnZone::new(glam::Vec3::ZERO, 20.0));
        manager.start_encounter(2);

        manager.update(0.016);
        assert!(manager.enemies_alive > 0 || manager.state == EncounterState::Spawning);
    }
}
