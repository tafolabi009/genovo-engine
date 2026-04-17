// engine/gameplay/src/level_progression.rs
//
// Level/map progression system for the Genovo engine.
//
// Manages level sequencing, unlocking, scoring, and world map navigation:
//
// - **Level sequence** -- Ordered list of levels with dependencies.
// - **Unlock conditions** -- Levels unlock based on stars, scores, or completion.
// - **Par time/score** -- Target times and scores for medal ratings.
// - **Star ratings** -- 1-3 star ratings based on performance criteria.
// - **Level statistics** -- Track attempts, completions, best times, scores.
// - **Level select data** -- Provide UI-ready data for level selection screens.
// - **World map nodes** -- Graph-based world map with paths between levels.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_STARS: u32 = 3;

// ---------------------------------------------------------------------------
// Level identifier
// ---------------------------------------------------------------------------

/// Unique identifier for a level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LevelId(pub u32);

impl fmt::Display for LevelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Level({})", self.0)
    }
}

/// World/chapter identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorldId(pub u32);

// ---------------------------------------------------------------------------
// Star criteria
// ---------------------------------------------------------------------------

/// Criteria for earning stars on a level.
#[derive(Debug, Clone)]
pub struct StarCriteria {
    /// Score thresholds for 1, 2, 3 stars.
    pub score_thresholds: [u32; 3],
    /// Time thresholds for 1, 2, 3 stars (seconds, lower is better).
    /// 0 means time is not a criterion for that star.
    pub time_thresholds: [f32; 3],
    /// Custom criteria names and whether they are met.
    pub custom_criteria: Vec<String>,
}

impl Default for StarCriteria {
    fn default() -> Self {
        Self {
            score_thresholds: [1000, 2000, 5000],
            time_thresholds: [0.0; 3],
            custom_criteria: Vec::new(),
        }
    }
}

impl StarCriteria {
    /// Calculate stars earned based on score and time.
    pub fn calculate_stars(&self, score: u32, time: f32, custom_met: &[bool]) -> u32 {
        let mut stars = 0;
        for i in 0..MAX_STARS as usize {
            let score_ok = score >= self.score_thresholds[i];
            let time_ok = self.time_thresholds[i] <= 0.0 || time <= self.time_thresholds[i];
            let custom_ok = i >= custom_met.len() || custom_met.get(i).copied().unwrap_or(true);
            if score_ok && time_ok && custom_ok {
                stars = (i + 1) as u32;
            } else {
                break;
            }
        }
        stars
    }
}

// ---------------------------------------------------------------------------
// Unlock condition
// ---------------------------------------------------------------------------

/// Condition that must be met to unlock a level.
#[derive(Debug, Clone)]
pub enum UnlockCondition {
    /// Always unlocked.
    Always,
    /// Requires another level to be completed.
    LevelCompleted(LevelId),
    /// Requires a minimum number of total stars.
    TotalStars(u32),
    /// Requires a specific star count on a specific level.
    StarsOnLevel(LevelId, u32),
    /// Requires a minimum total score across all levels.
    TotalScore(u32),
    /// Requires all levels in a world to be completed.
    WorldCompleted(WorldId),
    /// All conditions must be met.
    All(Vec<UnlockCondition>),
    /// Any condition must be met.
    Any(Vec<UnlockCondition>),
    /// Custom condition (checked externally).
    Custom(String),
}

// ---------------------------------------------------------------------------
// Level definition
// ---------------------------------------------------------------------------

/// Defines a level's metadata and progression data.
#[derive(Debug, Clone)]
pub struct LevelDefinition {
    /// Unique level ID.
    pub id: LevelId,
    /// Display name.
    pub name: String,
    /// Description.
    pub description: String,
    /// World/chapter this level belongs to.
    pub world: WorldId,
    /// Order within the world (for display).
    pub order: u32,
    /// Scene asset to load.
    pub scene_asset: String,
    /// Par time (target completion time in seconds).
    pub par_time: f32,
    /// Par score (target score).
    pub par_score: u32,
    /// Star rating criteria.
    pub star_criteria: StarCriteria,
    /// Unlock condition.
    pub unlock_condition: UnlockCondition,
    /// Whether this is a boss level.
    pub is_boss: bool,
    /// Whether this is a bonus/secret level.
    pub is_bonus: bool,
    /// Thumbnail/preview image asset path.
    pub thumbnail: String,
    /// Difficulty rating (1-5).
    pub difficulty: u8,
    /// Estimated play time in minutes.
    pub estimated_time_minutes: u32,
}

impl LevelDefinition {
    /// Create a new level definition.
    pub fn new(id: LevelId, name: &str, world: WorldId) -> Self {
        Self {
            id,
            name: name.to_string(),
            description: String::new(),
            world,
            order: 0,
            scene_asset: String::new(),
            par_time: 120.0,
            par_score: 1000,
            star_criteria: StarCriteria::default(),
            unlock_condition: UnlockCondition::Always,
            is_boss: false,
            is_bonus: false,
            thumbnail: String::new(),
            difficulty: 1,
            estimated_time_minutes: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Level statistics
// ---------------------------------------------------------------------------

/// Statistics for a specific level.
#[derive(Debug, Clone, Default)]
pub struct LevelStats {
    /// Number of times the level was attempted.
    pub attempts: u32,
    /// Number of times the level was completed.
    pub completions: u32,
    /// Best completion time (seconds).
    pub best_time: Option<f32>,
    /// Best score achieved.
    pub best_score: u32,
    /// Most stars earned.
    pub best_stars: u32,
    /// Total time spent in this level (seconds).
    pub total_time: f32,
    /// Number of deaths in this level.
    pub deaths: u32,
    /// Whether the level has been completed at least once.
    pub completed: bool,
    /// Last completion timestamp (as f64 for precision).
    pub last_completed: Option<f64>,
    /// Custom stats (e.g., "coins_collected", "enemies_defeated").
    pub custom: HashMap<String, u32>,
}

impl LevelStats {
    /// Record a level completion.
    pub fn record_completion(&mut self, time: f32, score: u32, stars: u32) {
        self.completions += 1;
        self.completed = true;
        self.best_time = Some(match self.best_time {
            Some(prev) => prev.min(time),
            None => time,
        });
        self.best_score = self.best_score.max(score);
        self.best_stars = self.best_stars.max(stars);
    }

    /// Record a level attempt (may or may not complete).
    pub fn record_attempt(&mut self, time_spent: f32, deaths: u32) {
        self.attempts += 1;
        self.total_time += time_spent;
        self.deaths += deaths;
    }

    /// Completion rate.
    pub fn completion_rate(&self) -> f32 {
        if self.attempts > 0 {
            self.completions as f32 / self.attempts as f32
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// World map node
// ---------------------------------------------------------------------------

/// A node on the world map representing a level.
#[derive(Debug, Clone)]
pub struct WorldMapNode {
    /// Level ID this node represents.
    pub level_id: LevelId,
    /// Position on the world map (normalized 0..1 coordinates).
    pub position: [f32; 2],
    /// Connected node IDs (levels reachable from this node).
    pub connections: Vec<LevelId>,
    /// Visual style for the node (normal, boss, bonus, locked).
    pub style: NodeStyle,
    /// Whether the node is currently visible on the map.
    pub visible: bool,
    /// Custom icon identifier.
    pub icon: Option<String>,
}

/// Visual style for a world map node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStyle {
    Normal,
    Boss,
    Bonus,
    Locked,
    Completed,
    Current,
}

// ---------------------------------------------------------------------------
// Level select data (UI-ready)
// ---------------------------------------------------------------------------

/// Data for rendering a level in the level selection screen.
#[derive(Debug, Clone)]
pub struct LevelSelectEntry {
    /// Level ID.
    pub id: LevelId,
    /// Display name.
    pub name: String,
    /// World name.
    pub world_name: String,
    /// Whether the level is unlocked.
    pub unlocked: bool,
    /// Whether the level has been completed.
    pub completed: bool,
    /// Stars earned (0-3).
    pub stars: u32,
    /// Best score.
    pub best_score: u32,
    /// Best time (None if not completed).
    pub best_time: Option<f32>,
    /// Par time.
    pub par_time: f32,
    /// Thumbnail path.
    pub thumbnail: String,
    /// Difficulty.
    pub difficulty: u8,
    /// Is boss level.
    pub is_boss: bool,
    /// Is bonus level.
    pub is_bonus: bool,
}

// ---------------------------------------------------------------------------
// World definition
// ---------------------------------------------------------------------------

/// A world/chapter containing multiple levels.
#[derive(Debug, Clone)]
pub struct WorldDefinition {
    /// World ID.
    pub id: WorldId,
    /// Display name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Theme/biome.
    pub theme: String,
    /// Order for display.
    pub order: u32,
    /// Unlock condition for the world itself.
    pub unlock_condition: UnlockCondition,
    /// Background image/color for level select.
    pub background: String,
}

impl WorldDefinition {
    pub fn new(id: WorldId, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            description: String::new(),
            theme: String::new(),
            order: 0,
            unlock_condition: UnlockCondition::Always,
            background: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Level progression system
// ---------------------------------------------------------------------------

/// Events emitted by the progression system.
#[derive(Debug, Clone)]
pub enum ProgressionEvent {
    /// A level was unlocked.
    LevelUnlocked(LevelId),
    /// A level was completed.
    LevelCompleted { level: LevelId, stars: u32, score: u32, time: f32 },
    /// A new best was achieved.
    NewBest { level: LevelId, category: String },
    /// A world was completed.
    WorldCompleted(WorldId),
    /// A star milestone was reached.
    StarMilestone(u32),
}

/// The level progression manager.
pub struct LevelProgressionSystem {
    /// World definitions.
    worlds: Vec<WorldDefinition>,
    /// Level definitions.
    levels: Vec<LevelDefinition>,
    /// Per-level statistics.
    stats: HashMap<LevelId, LevelStats>,
    /// Unlocked levels.
    unlocked: HashMap<LevelId, bool>,
    /// World map nodes.
    map_nodes: Vec<WorldMapNode>,
    /// Current level being played.
    current_level: Option<LevelId>,
    /// Events queue.
    events: Vec<ProgressionEvent>,
    /// Total stars earned.
    total_stars: u32,
    /// Total score across all levels.
    total_score: u32,
}

impl LevelProgressionSystem {
    /// Create a new progression system.
    pub fn new() -> Self {
        Self {
            worlds: Vec::new(),
            levels: Vec::new(),
            stats: HashMap::new(),
            unlocked: HashMap::new(),
            map_nodes: Vec::new(),
            current_level: None,
            events: Vec::new(),
            total_stars: 0,
            total_score: 0,
        }
    }

    /// Add a world definition.
    pub fn add_world(&mut self, world: WorldDefinition) {
        self.worlds.push(world);
    }

    /// Add a level definition.
    pub fn add_level(&mut self, level: LevelDefinition) {
        let id = level.id;
        let always_unlocked = matches!(level.unlock_condition, UnlockCondition::Always);
        self.levels.push(level);
        if always_unlocked {
            self.unlocked.insert(id, true);
        }
    }

    /// Get a level definition.
    pub fn level(&self, id: LevelId) -> Option<&LevelDefinition> {
        self.levels.iter().find(|l| l.id == id)
    }

    /// Get level statistics.
    pub fn level_stats(&self, id: LevelId) -> Option<&LevelStats> {
        self.stats.get(&id)
    }

    /// Check if a level is unlocked.
    pub fn is_unlocked(&self, id: LevelId) -> bool {
        self.unlocked.get(&id).copied().unwrap_or(false)
    }

    /// Set the current level.
    pub fn set_current_level(&mut self, id: LevelId) {
        self.current_level = Some(id);
    }

    /// Record a level attempt.
    pub fn record_attempt(&mut self, level_id: LevelId, time_spent: f32, deaths: u32) {
        let stats = self.stats.entry(level_id).or_default();
        stats.record_attempt(time_spent, deaths);
    }

    /// Record a level completion and check for unlocks.
    pub fn complete_level(&mut self, level_id: LevelId, time: f32, score: u32, custom_met: &[bool]) {
        // Calculate stars.
        let stars = if let Some(level) = self.level(level_id) {
            level.star_criteria.calculate_stars(score, time, custom_met)
        } else {
            0
        };

        // Update stats.
        let was_completed = self.stats.get(&level_id).map(|s| s.completed).unwrap_or(false);
        let old_best_score = self.stats.get(&level_id).map(|s| s.best_score).unwrap_or(0);
        let old_best_stars = self.stats.get(&level_id).map(|s| s.best_stars).unwrap_or(0);

        let stats = self.stats.entry(level_id).or_default();
        stats.record_completion(time, score, stars);

        // Check for new bests.
        if score > old_best_score {
            self.events.push(ProgressionEvent::NewBest {
                level: level_id,
                category: "score".into(),
            });
        }
        if stars > old_best_stars {
            self.events.push(ProgressionEvent::NewBest {
                level: level_id,
                category: "stars".into(),
            });
        }

        self.events.push(ProgressionEvent::LevelCompleted {
            level: level_id,
            stars,
            score,
            time,
        });

        // Recalculate totals.
        self.recalculate_totals();

        // Check for new unlocks.
        self.check_unlocks();

        // Check if world is completed.
        if !was_completed {
            self.check_world_completion(level_id);
        }
    }

    /// Recalculate total stars and score.
    fn recalculate_totals(&mut self) {
        self.total_stars = self.stats.values().map(|s| s.best_stars).sum();
        self.total_score = self.stats.values().map(|s| s.best_score).sum();
    }

    /// Check unlock conditions for all levels.
    fn check_unlocks(&mut self) {
        let level_conditions: Vec<(LevelId, UnlockCondition)> = self
            .levels
            .iter()
            .filter(|l| !self.is_unlocked(l.id))
            .map(|l| (l.id, l.unlock_condition.clone()))
            .collect();

        for (level_id, condition) in level_conditions {
            if self.evaluate_unlock(&condition) {
                self.unlocked.insert(level_id, true);
                self.events.push(ProgressionEvent::LevelUnlocked(level_id));
            }
        }
    }

    /// Evaluate an unlock condition.
    fn evaluate_unlock(&self, condition: &UnlockCondition) -> bool {
        match condition {
            UnlockCondition::Always => true,
            UnlockCondition::LevelCompleted(id) => {
                self.stats.get(id).map(|s| s.completed).unwrap_or(false)
            }
            UnlockCondition::TotalStars(required) => self.total_stars >= *required,
            UnlockCondition::StarsOnLevel(id, required) => {
                self.stats.get(id).map(|s| s.best_stars >= *required).unwrap_or(false)
            }
            UnlockCondition::TotalScore(required) => self.total_score >= *required,
            UnlockCondition::WorldCompleted(world_id) => self.is_world_completed(*world_id),
            UnlockCondition::All(conditions) => {
                conditions.iter().all(|c| self.evaluate_unlock(c))
            }
            UnlockCondition::Any(conditions) => {
                conditions.iter().any(|c| self.evaluate_unlock(c))
            }
            UnlockCondition::Custom(_) => false,
        }
    }

    /// Check if a world is completed.
    fn is_world_completed(&self, world_id: WorldId) -> bool {
        let world_levels: Vec<LevelId> = self
            .levels
            .iter()
            .filter(|l| l.world == world_id && !l.is_bonus)
            .map(|l| l.id)
            .collect();

        if world_levels.is_empty() {
            return false;
        }

        world_levels
            .iter()
            .all(|id| self.stats.get(id).map(|s| s.completed).unwrap_or(false))
    }

    /// Check world completion after a level is finished.
    fn check_world_completion(&mut self, level_id: LevelId) {
        if let Some(level) = self.level(level_id) {
            let world_id = level.world;
            if self.is_world_completed(world_id) {
                self.events.push(ProgressionEvent::WorldCompleted(world_id));
            }
        }
    }

    /// Generate level select data for the UI.
    pub fn level_select_data(&self) -> Vec<LevelSelectEntry> {
        let mut entries = Vec::new();
        for level in &self.levels {
            let stats = self.stats.get(&level.id);
            let world_name = self
                .worlds
                .iter()
                .find(|w| w.id == level.world)
                .map(|w| w.name.as_str())
                .unwrap_or("Unknown");

            entries.push(LevelSelectEntry {
                id: level.id,
                name: level.name.clone(),
                world_name: world_name.to_string(),
                unlocked: self.is_unlocked(level.id),
                completed: stats.map(|s| s.completed).unwrap_or(false),
                stars: stats.map(|s| s.best_stars).unwrap_or(0),
                best_score: stats.map(|s| s.best_score).unwrap_or(0),
                best_time: stats.and_then(|s| s.best_time),
                par_time: level.par_time,
                thumbnail: level.thumbnail.clone(),
                difficulty: level.difficulty,
                is_boss: level.is_boss,
                is_bonus: level.is_bonus,
            });
        }
        entries
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<ProgressionEvent> {
        std::mem::take(&mut self.events)
    }

    /// Get total stars.
    pub fn total_stars(&self) -> u32 {
        self.total_stars
    }

    /// Get total score.
    pub fn total_score(&self) -> u32 {
        self.total_score
    }

    /// Get the current level.
    pub fn current_level(&self) -> Option<LevelId> {
        self.current_level
    }

    /// Get all levels in a world.
    pub fn levels_in_world(&self, world_id: WorldId) -> Vec<&LevelDefinition> {
        self.levels.iter().filter(|l| l.world == world_id).collect()
    }
}

impl Default for LevelProgressionSystem {
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

    #[test]
    fn test_star_calculation() {
        let criteria = StarCriteria {
            score_thresholds: [100, 500, 1000],
            time_thresholds: [0.0; 3],
            custom_criteria: Vec::new(),
        };
        assert_eq!(criteria.calculate_stars(50, 0.0, &[]), 0);
        assert_eq!(criteria.calculate_stars(100, 0.0, &[]), 1);
        assert_eq!(criteria.calculate_stars(500, 0.0, &[]), 2);
        assert_eq!(criteria.calculate_stars(1000, 0.0, &[]), 3);
    }

    #[test]
    fn test_level_completion() {
        let mut sys = LevelProgressionSystem::new();
        let world = WorldId(0);
        sys.add_world(WorldDefinition::new(world, "World 1"));
        let level_1 = LevelId(0);
        sys.add_level(LevelDefinition::new(level_1, "Level 1", world));

        sys.complete_level(level_1, 60.0, 2000, &[]);
        let stats = sys.level_stats(level_1).unwrap();
        assert!(stats.completed);
        assert_eq!(stats.best_stars, 2); // Default criteria: 2000 >= threshold[1]
    }

    #[test]
    fn test_unlock_condition() {
        let mut sys = LevelProgressionSystem::new();
        let world = WorldId(0);
        sys.add_world(WorldDefinition::new(world, "World 1"));

        let l1 = LevelId(0);
        let l2 = LevelId(1);
        sys.add_level(LevelDefinition::new(l1, "Level 1", world));

        let mut level2 = LevelDefinition::new(l2, "Level 2", world);
        level2.unlock_condition = UnlockCondition::LevelCompleted(l1);
        sys.add_level(level2);

        assert!(!sys.is_unlocked(l2));
        sys.complete_level(l1, 60.0, 5000, &[]);
        assert!(sys.is_unlocked(l2));
    }
}
