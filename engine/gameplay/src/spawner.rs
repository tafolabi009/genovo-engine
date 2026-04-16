//! Entity spawner systems for timed and wave-based enemy spawning.
//!
//! Provides two spawner types:
//!
//! - [`Spawner`] -- interval-based spawner that continuously produces entities
//!   up to a maximum alive count.
//! - [`WaveSpawner`] -- multi-wave enemy spawner with a state machine for
//!   wave transitions, rest periods, and difficulty scaling.

use glam::Vec3;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Spawn request
// ---------------------------------------------------------------------------

/// A request to spawn an entity, produced by spawners and consumed by
/// game code that creates actual entities.
#[derive(Debug, Clone)]
pub struct SpawnRequest {
    /// Entity type / prefab name to spawn.
    pub entity_type: String,
    /// World-space position to spawn at.
    pub position: Vec3,
    /// Optional rotation in radians (yaw).
    pub rotation: f32,
    /// Spawn point index (for spawners with multiple points).
    pub spawn_point_index: usize,
    /// Wave number (for wave spawners).
    pub wave: u32,
    /// Custom properties for the spawned entity.
    pub properties: Vec<(String, SpawnProperty)>,
}

/// A typed property value for spawned entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpawnProperty {
    Int(i32),
    Float(f32),
    String(String),
    Bool(bool),
}

// ---------------------------------------------------------------------------
// Spawn point
// ---------------------------------------------------------------------------

/// A position and rotation where entities can be spawned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnPoint {
    /// World-space position.
    pub position: Vec3,
    /// Yaw rotation in radians.
    pub rotation: f32,
    /// Random offset radius (spawns within this radius of position).
    pub random_radius: f32,
    /// Weight for random selection among multiple points.
    pub weight: f32,
}

impl SpawnPoint {
    /// Create a spawn point at the given position.
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            rotation: 0.0,
            random_radius: 0.0,
            weight: 1.0,
        }
    }

    /// Set random radius.
    pub fn with_radius(mut self, radius: f32) -> Self {
        self.random_radius = radius;
        self
    }

    /// Compute a randomized spawn position within the radius.
    pub fn randomized_position(&self, rng: &mut genovo_core::Rng) -> Vec3 {
        if self.random_radius <= 0.0 {
            return self.position;
        }
        let offset = rng.in_circle(self.random_radius);
        self.position + Vec3::new(offset.x, 0.0, offset.y)
    }
}

impl Default for SpawnPoint {
    fn default() -> Self {
        Self::new(Vec3::ZERO)
    }
}

// ---------------------------------------------------------------------------
// Spawner component
// ---------------------------------------------------------------------------

/// Interval-based entity spawner.
///
/// Spawns entities at a configurable interval, up to a maximum number of
/// simultaneously alive entities. When an entity is destroyed, a new one
/// can be spawned on the next interval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spawner {
    /// Entity type to spawn.
    pub entity_type: String,
    /// Spawn interval in seconds.
    pub interval: f32,
    /// Random jitter added to the interval (+/- this many seconds).
    pub interval_jitter: f32,
    /// Maximum number of alive entities from this spawner.
    pub max_alive: u32,
    /// Maximum total spawns (0 = unlimited).
    pub max_total: u32,
    /// Whether the spawner is active.
    pub active: bool,
    /// Spawn points.
    pub spawn_points: Vec<SpawnPoint>,
    /// Time accumulator since last spawn.
    time_since_spawn: f32,
    /// Current interval target (includes jitter).
    current_interval: f32,
    /// Number of currently alive entities from this spawner.
    alive_count: u32,
    /// Total number of entities spawned.
    total_spawned: u32,
    /// Whether to spawn one immediately on activation.
    pub spawn_on_activate: bool,
    /// Whether an initial spawn has been done.
    initial_spawned: bool,
}

impl Spawner {
    /// Create a new spawner.
    pub fn new(entity_type: impl Into<String>, interval: f32, max_alive: u32) -> Self {
        Self {
            entity_type: entity_type.into(),
            interval,
            interval_jitter: 0.0,
            max_alive,
            max_total: 0,
            active: true,
            spawn_points: vec![SpawnPoint::default()],
            time_since_spawn: 0.0,
            current_interval: interval,
            alive_count: 0,
            total_spawned: 0,
            spawn_on_activate: false,
            initial_spawned: false,
        }
    }

    /// Set spawn points.
    pub fn with_spawn_points(mut self, points: Vec<SpawnPoint>) -> Self {
        if !points.is_empty() {
            self.spawn_points = points;
        }
        self
    }

    /// Set interval jitter.
    pub fn with_jitter(mut self, jitter: f32) -> Self {
        self.interval_jitter = jitter;
        self
    }

    /// Set max total spawns.
    pub fn with_max_total(mut self, max: u32) -> Self {
        self.max_total = max;
        self
    }

    /// Enable immediate spawn on activation.
    pub fn with_immediate_spawn(mut self) -> Self {
        self.spawn_on_activate = true;
        self
    }

    /// Update the spawner. Returns spawn requests for entities that should
    /// be created this frame.
    pub fn update(&mut self, dt: f32, rng: &mut genovo_core::Rng) -> Vec<SpawnRequest> {
        let mut requests = Vec::new();

        if !self.active {
            return requests;
        }

        // Check total limit.
        if self.max_total > 0 && self.total_spawned >= self.max_total {
            return requests;
        }

        // Initial spawn.
        if self.spawn_on_activate && !self.initial_spawned {
            self.initial_spawned = true;
            if self.alive_count < self.max_alive {
                if let Some(req) = self.create_spawn_request(rng) {
                    requests.push(req);
                    self.alive_count += 1;
                    self.total_spawned += 1;
                }
            }
        }

        // Interval-based spawning.
        self.time_since_spawn += dt;
        while self.time_since_spawn >= self.current_interval {
            self.time_since_spawn -= self.current_interval;

            // Recalculate interval with jitter.
            self.current_interval = self.interval
                + if self.interval_jitter > 0.0 {
                    rng.range(-self.interval_jitter, self.interval_jitter)
                } else {
                    0.0
                };
            self.current_interval = self.current_interval.max(0.1);

            if self.alive_count >= self.max_alive {
                continue;
            }
            if self.max_total > 0 && self.total_spawned >= self.max_total {
                break;
            }

            if let Some(req) = self.create_spawn_request(rng) {
                requests.push(req);
                self.alive_count += 1;
                self.total_spawned += 1;
            }
        }

        requests
    }

    /// Create a spawn request at a random spawn point.
    fn create_spawn_request(&self, rng: &mut genovo_core::Rng) -> Option<SpawnRequest> {
        if self.spawn_points.is_empty() {
            return None;
        }

        let weights: Vec<f32> = self.spawn_points.iter().map(|p| p.weight).collect();
        let index = rng.weighted_pick(&weights);
        let point = &self.spawn_points[index];

        Some(SpawnRequest {
            entity_type: self.entity_type.clone(),
            position: point.randomized_position(rng),
            rotation: point.rotation,
            spawn_point_index: index,
            wave: 0,
            properties: Vec::new(),
        })
    }

    /// Notify the spawner that one of its entities was destroyed.
    pub fn on_entity_destroyed(&mut self) {
        self.alive_count = self.alive_count.saturating_sub(1);
    }

    /// Reset the spawner to its initial state.
    pub fn reset(&mut self) {
        self.alive_count = 0;
        self.total_spawned = 0;
        self.time_since_spawn = 0.0;
        self.current_interval = self.interval;
        self.initial_spawned = false;
    }

    /// Current number of alive entities.
    #[inline]
    pub fn alive_count(&self) -> u32 {
        self.alive_count
    }

    /// Total entities spawned.
    #[inline]
    pub fn total_spawned(&self) -> u32 {
        self.total_spawned
    }

    /// Whether the spawner has reached its total limit.
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        self.max_total > 0 && self.total_spawned >= self.max_total
    }

    /// Activate the spawner.
    pub fn activate(&mut self) {
        self.active = true;
    }

    /// Deactivate the spawner.
    pub fn deactivate(&mut self) {
        self.active = false;
    }
}

impl genovo_ecs::Component for Spawner {}

// ---------------------------------------------------------------------------
// Wave spawner
// ---------------------------------------------------------------------------

/// State of the wave spawner state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaveState {
    /// Waiting to start (not yet triggered).
    Idle,
    /// Pre-wave countdown.
    Countdown,
    /// Spawning enemies for the current wave.
    Spawning,
    /// Waiting for all enemies in the wave to be killed.
    WaitingForClear,
    /// Rest period between waves.
    Rest,
    /// All waves completed.
    Completed,
    /// The wave spawner was defeated (all enemies cleared on final wave).
    Victory,
}

/// Definition of a single wave of enemies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveDefinition {
    /// Enemy types and counts for this wave.
    pub enemy_groups: Vec<EnemyGroup>,
    /// Delay before the wave starts spawning (seconds).
    pub start_delay: f32,
    /// Interval between individual spawns within the wave (seconds).
    pub spawn_interval: f32,
    /// Rest time after this wave (before next wave starts).
    pub rest_time: f32,
    /// Display name for this wave.
    pub name: String,
}

/// A group of enemies of the same type within a wave.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnemyGroup {
    /// Entity type / prefab name.
    pub entity_type: String,
    /// Number of enemies in this group.
    pub count: u32,
    /// Difficulty scaling factor (multiplied with base stats).
    pub difficulty_scale: f32,
}

impl EnemyGroup {
    /// Create a new enemy group.
    pub fn new(entity_type: impl Into<String>, count: u32) -> Self {
        Self {
            entity_type: entity_type.into(),
            count,
            difficulty_scale: 1.0,
        }
    }

    /// Set difficulty scaling.
    pub fn with_difficulty(mut self, scale: f32) -> Self {
        self.difficulty_scale = scale;
        self
    }
}

impl WaveDefinition {
    /// Create a new wave.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            enemy_groups: Vec::new(),
            start_delay: 3.0,
            spawn_interval: 0.5,
            rest_time: 10.0,
            name: name.into(),
        }
    }

    /// Add an enemy group.
    pub fn with_group(mut self, group: EnemyGroup) -> Self {
        self.enemy_groups.push(group);
        self
    }

    /// Set the rest time.
    pub fn with_rest_time(mut self, seconds: f32) -> Self {
        self.rest_time = seconds;
        self
    }

    /// Total number of enemies in this wave.
    pub fn total_enemies(&self) -> u32 {
        self.enemy_groups.iter().map(|g| g.count).sum()
    }
}

/// Multi-wave enemy spawner with a state machine.
///
/// Manages a sequence of waves, each containing groups of enemies. Tracks
/// spawning, wave transitions, rest periods, and completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveSpawner {
    /// Wave definitions.
    pub waves: Vec<WaveDefinition>,
    /// Spawn points.
    pub spawn_points: Vec<SpawnPoint>,
    /// Current state.
    state: WaveState,
    /// Current wave index.
    current_wave: usize,
    /// Timer for state transitions.
    timer: f32,
    /// Spawn timer for individual enemy spawns within a wave.
    spawn_timer: f32,
    /// Index into the flattened enemy list for the current wave.
    spawn_index: usize,
    /// Number of alive enemies from the current wave.
    alive_count: u32,
    /// Total enemies spawned in the current wave.
    wave_spawned: u32,
    /// Total enemies killed across all waves.
    pub total_kills: u32,
    /// Whether to auto-start the next wave after rest.
    pub auto_start: bool,
    /// Countdown duration before the first wave.
    pub initial_countdown: f32,
    /// Difficulty multiplier that increases per wave.
    pub difficulty_per_wave: f32,
}

impl WaveSpawner {
    /// Create a new wave spawner.
    pub fn new() -> Self {
        Self {
            waves: Vec::new(),
            spawn_points: vec![SpawnPoint::default()],
            state: WaveState::Idle,
            current_wave: 0,
            timer: 0.0,
            spawn_timer: 0.0,
            spawn_index: 0,
            alive_count: 0,
            wave_spawned: 0,
            total_kills: 0,
            auto_start: true,
            initial_countdown: 5.0,
            difficulty_per_wave: 0.1,
        }
    }

    /// Add a wave.
    pub fn add_wave(&mut self, wave: WaveDefinition) {
        self.waves.push(wave);
    }

    /// Set spawn points.
    pub fn set_spawn_points(&mut self, points: Vec<SpawnPoint>) {
        if !points.is_empty() {
            self.spawn_points = points;
        }
    }

    /// Start the wave spawner (begins countdown to first wave).
    pub fn start(&mut self) {
        self.state = WaveState::Countdown;
        self.timer = self.initial_countdown;
        self.current_wave = 0;
        self.total_kills = 0;
        log::info!("Wave spawner started (countdown: {:.1}s)", self.timer);
    }

    /// Update the wave spawner. Returns spawn requests.
    pub fn update(&mut self, dt: f32, rng: &mut genovo_core::Rng) -> Vec<SpawnRequest> {
        let mut requests = Vec::new();

        match self.state {
            WaveState::Idle | WaveState::Completed | WaveState::Victory => {}

            WaveState::Countdown => {
                self.timer -= dt;
                if self.timer <= 0.0 {
                    self.begin_wave();
                }
            }

            WaveState::Spawning => {
                self.spawn_timer -= dt;
                if self.spawn_timer <= 0.0 {
                    if let Some(req) = self.spawn_next_enemy(rng) {
                        requests.push(req);
                    }

                    // Check if all enemies for this wave have been spawned.
                    let wave = &self.waves[self.current_wave];
                    if self.wave_spawned >= wave.total_enemies() {
                        self.state = WaveState::WaitingForClear;
                        log::info!(
                            "Wave {} '{}': all enemies spawned, waiting for clear",
                            self.current_wave + 1,
                            wave.name
                        );
                    } else {
                        self.spawn_timer = wave.spawn_interval;
                    }
                }
            }

            WaveState::WaitingForClear => {
                if self.alive_count == 0 {
                    self.on_wave_cleared();
                }
            }

            WaveState::Rest => {
                self.timer -= dt;
                if self.timer <= 0.0 && self.auto_start {
                    self.current_wave += 1;
                    if self.current_wave < self.waves.len() {
                        self.begin_wave();
                    } else {
                        self.state = WaveState::Completed;
                        log::info!("All waves completed!");
                    }
                }
            }
        }

        requests
    }

    /// Begin the current wave.
    fn begin_wave(&mut self) {
        if self.current_wave >= self.waves.len() {
            self.state = WaveState::Completed;
            return;
        }

        self.state = WaveState::Spawning;
        self.spawn_index = 0;
        self.wave_spawned = 0;
        self.alive_count = 0;
        self.spawn_timer = 0.0;

        let wave = &self.waves[self.current_wave];
        log::info!(
            "Wave {} '{}' starting ({} enemies)",
            self.current_wave + 1,
            wave.name,
            wave.total_enemies()
        );
    }

    /// Spawn the next enemy from the current wave.
    fn spawn_next_enemy(&mut self, rng: &mut genovo_core::Rng) -> Option<SpawnRequest> {
        let wave = &self.waves[self.current_wave];

        // Find which enemy group and index within it.
        let mut cumulative = 0u32;
        for group in &wave.enemy_groups {
            if self.spawn_index < (cumulative + group.count) as usize {
                let weights: Vec<f32> = self.spawn_points.iter().map(|p| p.weight).collect();
                let point_idx = rng.weighted_pick(&weights);
                let point = &self.spawn_points[point_idx];

                let difficulty =
                    group.difficulty_scale * (1.0 + self.current_wave as f32 * self.difficulty_per_wave);

                let req = SpawnRequest {
                    entity_type: group.entity_type.clone(),
                    position: point.randomized_position(rng),
                    rotation: point.rotation,
                    spawn_point_index: point_idx,
                    wave: self.current_wave as u32,
                    properties: vec![
                        (
                            "difficulty_scale".into(),
                            SpawnProperty::Float(difficulty),
                        ),
                        (
                            "wave".into(),
                            SpawnProperty::Int(self.current_wave as i32 + 1),
                        ),
                    ],
                };

                self.spawn_index += 1;
                self.wave_spawned += 1;
                self.alive_count += 1;

                return Some(req);
            }
            cumulative += group.count;
        }

        None
    }

    /// Handle wave clear.
    fn on_wave_cleared(&mut self) {
        let wave = &self.waves[self.current_wave];
        let rest_time = wave.rest_time;

        log::info!(
            "Wave {} '{}' cleared!",
            self.current_wave + 1,
            wave.name
        );

        if self.current_wave + 1 >= self.waves.len() {
            self.state = WaveState::Victory;
            log::info!("All waves cleared -- victory!");
        } else {
            self.state = WaveState::Rest;
            self.timer = rest_time;
        }
    }

    /// Notify the spawner that one of its enemies was killed.
    pub fn on_enemy_killed(&mut self) {
        self.alive_count = self.alive_count.saturating_sub(1);
        self.total_kills += 1;
    }

    /// Force-start the next wave (skip rest period).
    pub fn force_next_wave(&mut self) {
        if self.state == WaveState::Rest {
            self.current_wave += 1;
            if self.current_wave < self.waves.len() {
                self.begin_wave();
            } else {
                self.state = WaveState::Completed;
            }
        }
    }

    /// Current wave state.
    #[inline]
    pub fn state(&self) -> WaveState {
        self.state
    }

    /// Current wave number (1-indexed for display).
    #[inline]
    pub fn current_wave_number(&self) -> usize {
        self.current_wave + 1
    }

    /// Total number of waves.
    #[inline]
    pub fn total_waves(&self) -> usize {
        self.waves.len()
    }

    /// Number of alive enemies.
    #[inline]
    pub fn alive_count(&self) -> u32 {
        self.alive_count
    }

    /// Number of enemies spawned in the current wave.
    #[inline]
    pub fn wave_spawned(&self) -> u32 {
        self.wave_spawned
    }

    /// Timer value (countdown, rest, etc.).
    #[inline]
    pub fn timer(&self) -> f32 {
        self.timer
    }

    /// Whether all waves are complete.
    #[inline]
    pub fn is_complete(&self) -> bool {
        matches!(self.state, WaveState::Completed | WaveState::Victory)
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.state = WaveState::Idle;
        self.current_wave = 0;
        self.timer = 0.0;
        self.spawn_timer = 0.0;
        self.spawn_index = 0;
        self.alive_count = 0;
        self.wave_spawned = 0;
        self.total_kills = 0;
    }
}

impl Default for WaveSpawner {
    fn default() -> Self {
        Self::new()
    }
}

impl genovo_ecs::Component for WaveSpawner {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawner_basic() {
        let mut rng = genovo_core::Rng::new(42);
        let mut spawner = Spawner::new("goblin", 1.0, 3);

        // Should not spawn immediately (needs 1 second).
        let requests = spawner.update(0.5, &mut rng);
        assert!(requests.is_empty());

        // After 1 second total, should spawn.
        let requests = spawner.update(0.5, &mut rng);
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].entity_type, "goblin");
    }

    #[test]
    fn spawner_max_alive() {
        let mut rng = genovo_core::Rng::new(42);
        let mut spawner = Spawner::new("goblin", 0.1, 2);

        // Spawn several times.
        let mut total = 0;
        for _ in 0..20 {
            let requests = spawner.update(0.1, &mut rng);
            total += requests.len();
        }

        // Should cap at max_alive.
        assert!(total <= 2, "Should not exceed max_alive: {}", total);
        assert_eq!(spawner.alive_count(), 2);
    }

    #[test]
    fn spawner_entity_destroyed() {
        let mut rng = genovo_core::Rng::new(42);
        let mut spawner = Spawner::new("goblin", 0.1, 1);

        // Spawn one.
        let requests = spawner.update(1.0, &mut rng);
        assert_eq!(requests.len(), 1);

        // Should not spawn while alive.
        let requests = spawner.update(1.0, &mut rng);
        assert!(requests.is_empty());

        // Destroy it.
        spawner.on_entity_destroyed();

        // Should spawn again.
        let requests = spawner.update(1.0, &mut rng);
        assert_eq!(requests.len(), 1);
    }

    #[test]
    fn spawner_max_total() {
        let mut rng = genovo_core::Rng::new(42);
        let mut spawner = Spawner::new("goblin", 0.1, 10).with_max_total(3);

        for _ in 0..50 {
            spawner.update(0.1, &mut rng);
        }

        assert!(spawner.is_exhausted());
        assert_eq!(spawner.total_spawned(), 3);
    }

    #[test]
    fn spawner_immediate_spawn() {
        let mut rng = genovo_core::Rng::new(42);
        let mut spawner = Spawner::new("goblin", 10.0, 5).with_immediate_spawn();

        let requests = spawner.update(0.01, &mut rng);
        assert_eq!(requests.len(), 1, "Should spawn immediately");
    }

    #[test]
    fn spawner_reset() {
        let mut rng = genovo_core::Rng::new(42);
        let mut spawner = Spawner::new("goblin", 0.1, 3);

        for _ in 0..10 {
            spawner.update(0.1, &mut rng);
        }

        spawner.reset();
        assert_eq!(spawner.alive_count(), 0);
        assert_eq!(spawner.total_spawned(), 0);
    }

    #[test]
    fn wave_spawner_basic() {
        let mut rng = genovo_core::Rng::new(42);
        let mut ws = WaveSpawner::new();
        ws.initial_countdown = 0.0;

        ws.add_wave(
            WaveDefinition::new("Wave 1")
                .with_group(EnemyGroup::new("goblin", 3))
                .with_rest_time(1.0),
        );
        ws.add_wave(
            WaveDefinition::new("Wave 2")
                .with_group(EnemyGroup::new("orc", 2))
                .with_rest_time(0.0),
        );

        ws.start();

        // Spawn wave 1.
        let mut total_spawned = 0;
        for _ in 0..50 {
            let requests = ws.update(0.1, &mut rng);
            total_spawned += requests.len();
        }

        assert_eq!(total_spawned, 3, "Wave 1 should spawn 3 goblins");
        assert_eq!(ws.state(), WaveState::WaitingForClear);

        // Kill all enemies.
        for _ in 0..3 {
            ws.on_enemy_killed();
        }

        // Should advance to rest.
        ws.update(0.0, &mut rng);
        assert_eq!(ws.state(), WaveState::Rest);

        // Wait for rest to finish.
        for _ in 0..20 {
            ws.update(0.1, &mut rng);
        }

        // Should be in wave 2.
        assert!(
            ws.state() == WaveState::Spawning || ws.state() == WaveState::WaitingForClear,
            "Should be in wave 2, state: {:?}",
            ws.state()
        );
    }

    #[test]
    fn wave_spawner_victory() {
        let mut rng = genovo_core::Rng::new(42);
        let mut ws = WaveSpawner::new();
        ws.initial_countdown = 0.0;

        ws.add_wave(
            WaveDefinition::new("Final Wave")
                .with_group(EnemyGroup::new("boss", 1)),
        );

        ws.start();

        // Spawn.
        for _ in 0..10 {
            ws.update(0.1, &mut rng);
        }

        // Kill the boss.
        ws.on_enemy_killed();
        ws.update(0.0, &mut rng);

        assert_eq!(ws.state(), WaveState::Victory);
        assert!(ws.is_complete());
    }

    #[test]
    fn wave_spawner_reset() {
        let mut ws = WaveSpawner::new();
        ws.add_wave(WaveDefinition::new("W1"));
        ws.start();
        ws.reset();

        assert_eq!(ws.state(), WaveState::Idle);
        assert_eq!(ws.current_wave_number(), 1);
        assert_eq!(ws.total_kills, 0);
    }

    #[test]
    fn spawn_point_randomization() {
        let mut rng = genovo_core::Rng::new(42);
        let point = SpawnPoint::new(Vec3::new(10.0, 0.0, 10.0)).with_radius(2.0);

        let pos = point.randomized_position(&mut rng);
        let dist = (pos - point.position).length();
        assert!(dist <= 2.01, "Should be within radius: {}", dist);
    }

    #[test]
    fn enemy_group_difficulty() {
        let group = EnemyGroup::new("orc", 5).with_difficulty(1.5);
        assert_eq!(group.count, 5);
        assert!((group.difficulty_scale - 1.5).abs() < 0.01);
    }

    #[test]
    fn wave_total_enemies() {
        let wave = WaveDefinition::new("Test")
            .with_group(EnemyGroup::new("goblin", 5))
            .with_group(EnemyGroup::new("orc", 3));

        assert_eq!(wave.total_enemies(), 8);
    }

    #[test]
    fn force_next_wave() {
        let mut rng = genovo_core::Rng::new(42);
        let mut ws = WaveSpawner::new();
        ws.initial_countdown = 0.0;

        ws.add_wave(
            WaveDefinition::new("W1")
                .with_group(EnemyGroup::new("a", 1))
                .with_rest_time(999.0),
        );
        ws.add_wave(WaveDefinition::new("W2").with_group(EnemyGroup::new("b", 1)));

        ws.start();

        // Spawn and clear wave 1.
        for _ in 0..10 {
            ws.update(0.1, &mut rng);
        }
        ws.on_enemy_killed();
        ws.update(0.0, &mut rng);

        assert_eq!(ws.state(), WaveState::Rest);

        // Force next wave instead of waiting.
        ws.force_next_wave();
        assert_eq!(ws.state(), WaveState::Spawning);
        assert_eq!(ws.current_wave_number(), 2);
    }
}
