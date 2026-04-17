//! Score and leaderboard system.
//!
//! Provides a point system, combo multipliers, kill streaks, score events,
//! a local leaderboard, stat tracking (kills, deaths, accuracy, playtime),
//! end-of-match stats, and MVP calculation.
//!
//! # Key concepts
//!
//! - **ScoreTracker**: Per-player score state with combo tracking.
//! - **ScoreEvent**: A discrete score-earning event (kill, objective, etc.).
//! - **ComboMultiplier**: Increasing multiplier for rapid successive scores.
//! - **KillStreak**: Tracking consecutive kills without dying.
//! - **Leaderboard**: Ranked list of players by score or stat.
//! - **MatchStats**: Aggregated statistics for a completed match.
//! - **MVPCalculation**: Weighted formula to determine the most valuable
//!   player of a match.

use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum players tracked by the leaderboard.
pub const MAX_LEADERBOARD_ENTRIES: usize = 100;

/// Maximum score events stored per player.
pub const MAX_SCORE_EVENTS: usize = 512;

/// Default combo timeout (seconds between actions to maintain combo).
pub const DEFAULT_COMBO_TIMEOUT: f32 = 5.0;

/// Maximum combo multiplier.
pub const MAX_COMBO_MULTIPLIER: f32 = 8.0;

/// Combo multiplier increment per consecutive event.
pub const COMBO_INCREMENT: f32 = 0.5;

/// Base combo multiplier (starting value).
pub const BASE_COMBO_MULTIPLIER: f32 = 1.0;

/// Kill streak thresholds and names.
pub const STREAK_NAMES: &[(u32, &str)] = &[
    (3, "Killing Spree"),
    (5, "Rampage"),
    (7, "Dominating"),
    (10, "Unstoppable"),
    (15, "Godlike"),
    (20, "Legendary"),
    (25, "Beyond Godlike"),
];

/// Points awarded for different kill streak milestones.
pub const STREAK_BONUS_POINTS: &[(u32, u32)] = &[
    (3, 50),
    (5, 100),
    (7, 200),
    (10, 500),
    (15, 1000),
    (20, 2000),
];

/// MVP weight for kills.
pub const MVP_KILL_WEIGHT: f32 = 1.0;

/// MVP weight for assists.
pub const MVP_ASSIST_WEIGHT: f32 = 0.5;

/// MVP weight for objectives.
pub const MVP_OBJECTIVE_WEIGHT: f32 = 2.0;

/// MVP weight for deaths (negative).
pub const MVP_DEATH_WEIGHT: f32 = -0.3;

/// MVP weight for accuracy.
pub const MVP_ACCURACY_WEIGHT: f32 = 0.2;

/// MVP weight for healing.
pub const MVP_HEALING_WEIGHT: f32 = 0.8;

/// MVP weight for damage dealt.
pub const MVP_DAMAGE_WEIGHT: f32 = 0.01;

/// Maximum tracked stats per player.
pub const MAX_TRACKED_STATS: usize = 64;

/// Default precision for accuracy calculation.
pub const ACCURACY_PRECISION: f32 = 100.0;

// ---------------------------------------------------------------------------
// ScoreEventType
// ---------------------------------------------------------------------------

/// Type of score-earning event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScoreEventType {
    /// Killed an enemy.
    Kill,
    /// Assisted in killing an enemy.
    Assist,
    /// Headshot kill.
    Headshot,
    /// Multi-kill (multiple kills in rapid succession).
    MultiKill,
    /// First blood (first kill of the match).
    FirstBlood,
    /// Captured an objective.
    ObjectiveCapture,
    /// Defended an objective.
    ObjectiveDefend,
    /// Completed a challenge or achievement.
    Challenge,
    /// Healed a teammate.
    Healing,
    /// Revived a teammate.
    Revive,
    /// Destroyed a vehicle or structure.
    Destruction,
    /// Flag capture (CTF).
    FlagCapture,
    /// Flag return (CTF).
    FlagReturn,
    /// Survived a round.
    Survival,
    /// Bonus points (events, pickups, etc.).
    Bonus,
    /// Streak milestone.
    StreakBonus,
    /// Combo bonus.
    ComboBonus,
    /// Custom game-specific event.
    Custom(u32),
}

impl ScoreEventType {
    /// Base points for this event type.
    pub fn base_points(&self) -> u32 {
        match self {
            Self::Kill => 100,
            Self::Assist => 50,
            Self::Headshot => 150,
            Self::MultiKill => 200,
            Self::FirstBlood => 250,
            Self::ObjectiveCapture => 300,
            Self::ObjectiveDefend => 200,
            Self::Challenge => 500,
            Self::Healing => 25,
            Self::Revive => 150,
            Self::Destruction => 200,
            Self::FlagCapture => 400,
            Self::FlagReturn => 150,
            Self::Survival => 100,
            Self::Bonus => 50,
            Self::StreakBonus => 0, // determined by streak level
            Self::ComboBonus => 0, // determined by combo
            Self::Custom(_) => 100,
        }
    }

    /// Display name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Kill => "Kill",
            Self::Assist => "Assist",
            Self::Headshot => "Headshot",
            Self::MultiKill => "Multi Kill",
            Self::FirstBlood => "First Blood",
            Self::ObjectiveCapture => "Objective Captured",
            Self::ObjectiveDefend => "Objective Defended",
            Self::Challenge => "Challenge Complete",
            Self::Healing => "Healing",
            Self::Revive => "Revive",
            Self::Destruction => "Destruction",
            Self::FlagCapture => "Flag Captured",
            Self::FlagReturn => "Flag Returned",
            Self::Survival => "Survived",
            Self::Bonus => "Bonus",
            Self::StreakBonus => "Streak Bonus",
            Self::ComboBonus => "Combo Bonus",
            Self::Custom(_) => "Custom",
        }
    }

    /// Whether this event type extends a combo.
    pub fn extends_combo(&self) -> bool {
        matches!(
            self,
            Self::Kill
                | Self::Assist
                | Self::Headshot
                | Self::MultiKill
                | Self::ObjectiveCapture
                | Self::ObjectiveDefend
                | Self::Destruction
        )
    }

    /// Whether this event type counts toward a kill streak.
    pub fn counts_for_streak(&self) -> bool {
        matches!(
            self,
            Self::Kill | Self::Headshot | Self::MultiKill
        )
    }
}

// ---------------------------------------------------------------------------
// ScoreEvent
// ---------------------------------------------------------------------------

/// A discrete score-earning event.
#[derive(Debug, Clone)]
pub struct ScoreEvent {
    /// Event type.
    pub event_type: ScoreEventType,
    /// Points awarded (after multipliers).
    pub points: u32,
    /// Base points (before multipliers).
    pub base_points: u32,
    /// Combo multiplier at time of event.
    pub combo_multiplier: f32,
    /// Timestamp (game time in seconds).
    pub timestamp: f64,
    /// Description.
    pub description: String,
    /// Related entity (e.g., victim ID for kills).
    pub related_entity: Option<u64>,
}

impl ScoreEvent {
    /// Create a new score event.
    pub fn new(event_type: ScoreEventType, points: u32, timestamp: f64) -> Self {
        Self {
            event_type,
            points,
            base_points: points,
            combo_multiplier: 1.0,
            timestamp,
            description: event_type.name().to_string(),
            related_entity: None,
        }
    }

    /// Set a custom description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set the related entity.
    pub fn with_entity(mut self, entity_id: u64) -> Self {
        self.related_entity = Some(entity_id);
        self
    }
}

// ---------------------------------------------------------------------------
// ComboTracker
// ---------------------------------------------------------------------------

/// Tracks combo multiplier state.
#[derive(Debug, Clone)]
pub struct ComboTracker {
    /// Current combo multiplier.
    pub multiplier: f32,
    /// Current combo count (number of consecutive events).
    pub count: u32,
    /// Time since last combo event.
    pub timer: f32,
    /// Timeout duration before combo resets.
    pub timeout: f32,
    /// Highest combo achieved.
    pub best_combo: u32,
    /// Whether the combo is active.
    pub active: bool,
}

impl ComboTracker {
    /// Create a new combo tracker.
    pub fn new(timeout: f32) -> Self {
        Self {
            multiplier: BASE_COMBO_MULTIPLIER,
            count: 0,
            timer: 0.0,
            timeout,
            best_combo: 0,
            active: false,
        }
    }

    /// Extend the combo with a new event.
    pub fn extend(&mut self) {
        self.count += 1;
        self.timer = 0.0;
        self.multiplier =
            (BASE_COMBO_MULTIPLIER + self.count as f32 * COMBO_INCREMENT)
                .min(MAX_COMBO_MULTIPLIER);
        self.active = true;

        if self.count > self.best_combo {
            self.best_combo = self.count;
        }
    }

    /// Update the combo timer. Resets if timeout exceeded.
    pub fn update(&mut self, dt: f32) {
        if self.active {
            self.timer += dt;
            if self.timer >= self.timeout {
                self.reset();
            }
        }
    }

    /// Reset the combo.
    pub fn reset(&mut self) {
        self.count = 0;
        self.multiplier = BASE_COMBO_MULTIPLIER;
        self.timer = 0.0;
        self.active = false;
    }

    /// Get the current multiplier.
    pub fn current_multiplier(&self) -> f32 {
        self.multiplier
    }

    /// Get remaining time before combo resets.
    pub fn remaining_time(&self) -> f32 {
        if self.active {
            (self.timeout - self.timer).max(0.0)
        } else {
            0.0
        }
    }
}

impl Default for ComboTracker {
    fn default() -> Self {
        Self::new(DEFAULT_COMBO_TIMEOUT)
    }
}

// ---------------------------------------------------------------------------
// KillStreak
// ---------------------------------------------------------------------------

/// Tracks consecutive kills without dying.
#[derive(Debug, Clone)]
pub struct KillStreak {
    /// Current streak count.
    pub current: u32,
    /// Best streak ever achieved.
    pub best: u32,
    /// Current streak name (if at a milestone).
    pub current_name: Option<&'static str>,
    /// Total streaks achieved (count of times reaching 3+).
    pub total_streaks: u32,
}

impl KillStreak {
    /// Create a new kill streak tracker.
    pub fn new() -> Self {
        Self {
            current: 0,
            best: 0,
            current_name: None,
            total_streaks: 0,
        }
    }

    /// Record a kill. Returns the streak bonus points if a milestone was hit.
    pub fn record_kill(&mut self) -> Option<(u32, &'static str)> {
        self.current += 1;
        if self.current > self.best {
            self.best = self.current;
        }

        // Check for milestone
        self.current_name = None;
        for &(threshold, name) in STREAK_NAMES.iter().rev() {
            if self.current >= threshold {
                self.current_name = Some(name);
                break;
            }
        }

        // Check for bonus points
        for &(threshold, bonus) in STREAK_BONUS_POINTS {
            if self.current == threshold {
                if threshold == 3 {
                    self.total_streaks += 1;
                }
                return Some((bonus, self.current_name.unwrap_or("Streak")));
            }
        }

        None
    }

    /// Record a death. Resets the streak.
    pub fn record_death(&mut self) {
        self.current = 0;
        self.current_name = None;
    }

    /// Get the current streak name.
    pub fn streak_name(&self) -> Option<&'static str> {
        for &(threshold, name) in STREAK_NAMES.iter().rev() {
            if self.current >= threshold {
                return Some(name);
            }
        }
        None
    }
}

impl Default for KillStreak {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StatTracker
// ---------------------------------------------------------------------------

/// Tracks various gameplay statistics for a player.
#[derive(Debug, Clone)]
pub struct StatTracker {
    /// Total kills.
    pub kills: u32,
    /// Total deaths.
    pub deaths: u32,
    /// Total assists.
    pub assists: u32,
    /// Shots fired.
    pub shots_fired: u32,
    /// Shots hit.
    pub shots_hit: u32,
    /// Headshots.
    pub headshots: u32,
    /// Total damage dealt.
    pub damage_dealt: f64,
    /// Total damage taken.
    pub damage_taken: f64,
    /// Total healing done.
    pub healing_done: f64,
    /// Objectives captured.
    pub objectives_captured: u32,
    /// Objectives defended.
    pub objectives_defended: u32,
    /// Revives performed.
    pub revives: u32,
    /// Total playtime in seconds.
    pub playtime: f64,
    /// Distance traveled (world units).
    pub distance_traveled: f64,
    /// Items collected.
    pub items_collected: u32,
    /// Resources gathered.
    pub resources_gathered: u64,
    /// Structures built.
    pub structures_built: u32,
    /// Structures destroyed.
    pub structures_destroyed: u32,
    /// Custom stats.
    pub custom_stats: HashMap<String, f64>,
}

impl StatTracker {
    /// Create a new stat tracker.
    pub fn new() -> Self {
        Self {
            kills: 0,
            deaths: 0,
            assists: 0,
            shots_fired: 0,
            shots_hit: 0,
            headshots: 0,
            damage_dealt: 0.0,
            damage_taken: 0.0,
            healing_done: 0.0,
            objectives_captured: 0,
            objectives_defended: 0,
            revives: 0,
            playtime: 0.0,
            distance_traveled: 0.0,
            items_collected: 0,
            resources_gathered: 0,
            structures_built: 0,
            structures_destroyed: 0,
            custom_stats: HashMap::new(),
        }
    }

    /// Calculate accuracy as a percentage.
    pub fn accuracy(&self) -> f32 {
        if self.shots_fired == 0 {
            return 0.0;
        }
        (self.shots_hit as f32 / self.shots_fired as f32 * ACCURACY_PRECISION)
            .min(ACCURACY_PRECISION)
    }

    /// Calculate kill/death ratio.
    pub fn kd_ratio(&self) -> f32 {
        if self.deaths == 0 {
            return self.kills as f32;
        }
        self.kills as f32 / self.deaths as f32
    }

    /// Calculate kill/death/assist ratio (KDA).
    pub fn kda_ratio(&self) -> f32 {
        let d = if self.deaths == 0 { 1 } else { self.deaths };
        (self.kills as f32 + self.assists as f32 * 0.5) / d as f32
    }

    /// Headshot percentage.
    pub fn headshot_percentage(&self) -> f32 {
        if self.kills == 0 {
            return 0.0;
        }
        self.headshots as f32 / self.kills as f32 * 100.0
    }

    /// Average damage per kill.
    pub fn damage_per_kill(&self) -> f64 {
        if self.kills == 0 {
            return 0.0;
        }
        self.damage_dealt / self.kills as f64
    }

    /// Set a custom stat.
    pub fn set_custom(&mut self, key: impl Into<String>, value: f64) {
        let key = key.into();
        if self.custom_stats.len() < MAX_TRACKED_STATS || self.custom_stats.contains_key(&key) {
            self.custom_stats.insert(key, value);
        }
    }

    /// Increment a custom stat.
    pub fn increment_custom(&mut self, key: impl Into<String>, amount: f64) {
        let key = key.into();
        *self.custom_stats.entry(key).or_insert(0.0) += amount;
    }

    /// Get a custom stat.
    pub fn get_custom(&self, key: &str) -> f64 {
        self.custom_stats.get(key).copied().unwrap_or(0.0)
    }

    /// Update playtime.
    pub fn update_playtime(&mut self, dt: f64) {
        self.playtime += dt;
    }

    /// Merge another tracker's stats into this one (for aggregation).
    pub fn merge(&mut self, other: &StatTracker) {
        self.kills += other.kills;
        self.deaths += other.deaths;
        self.assists += other.assists;
        self.shots_fired += other.shots_fired;
        self.shots_hit += other.shots_hit;
        self.headshots += other.headshots;
        self.damage_dealt += other.damage_dealt;
        self.damage_taken += other.damage_taken;
        self.healing_done += other.healing_done;
        self.objectives_captured += other.objectives_captured;
        self.objectives_defended += other.objectives_defended;
        self.revives += other.revives;
        self.playtime += other.playtime;
        self.distance_traveled += other.distance_traveled;
        self.items_collected += other.items_collected;
        self.resources_gathered += other.resources_gathered;
        self.structures_built += other.structures_built;
        self.structures_destroyed += other.structures_destroyed;

        for (key, value) in &other.custom_stats {
            *self.custom_stats.entry(key.clone()).or_insert(0.0) += value;
        }
    }
}

impl Default for StatTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ScoreTracker
// ---------------------------------------------------------------------------

/// Per-player score tracking with combos and streaks.
pub struct ScoreTracker {
    /// Player entity ID.
    pub player_id: u64,
    /// Player display name.
    pub player_name: String,
    /// Current total score.
    pub score: u64,
    /// Combo tracker.
    pub combo: ComboTracker,
    /// Kill streak tracker.
    pub streak: KillStreak,
    /// Score event history.
    events: Vec<ScoreEvent>,
    /// Stat tracker.
    pub stats: StatTracker,
    /// Team ID (for team-based games).
    pub team_id: Option<u32>,
    /// Whether the player is alive.
    pub alive: bool,
    /// Multi-kill tracker: kills within a short window.
    multi_kill_count: u32,
    multi_kill_timer: f32,
    /// Has first blood been awarded in this match?
    first_blood_available: bool,
}

impl ScoreTracker {
    /// Create a new score tracker.
    pub fn new(player_id: u64, player_name: impl Into<String>) -> Self {
        Self {
            player_id,
            player_name: player_name.into(),
            score: 0,
            combo: ComboTracker::default(),
            streak: KillStreak::default(),
            events: Vec::new(),
            stats: StatTracker::new(),
            team_id: None,
            alive: true,
            multi_kill_count: 0,
            multi_kill_timer: 0.0,
            first_blood_available: true,
        }
    }

    /// Set the team.
    pub fn with_team(mut self, team_id: u32) -> Self {
        self.team_id = Some(team_id);
        self
    }

    /// Award points for an event.
    pub fn award(
        &mut self,
        event_type: ScoreEventType,
        game_time: f64,
    ) -> ScoreEvent {
        let base = event_type.base_points();

        // Extend combo if applicable
        if event_type.extends_combo() {
            self.combo.extend();
        }

        let multiplier = self.combo.current_multiplier();
        let points = (base as f32 * multiplier) as u32;

        self.score += points as u64;

        // Update stats
        match event_type {
            ScoreEventType::Kill | ScoreEventType::Headshot => {
                self.stats.kills += 1;
                if matches!(event_type, ScoreEventType::Headshot) {
                    self.stats.headshots += 1;
                }
            }
            ScoreEventType::Assist => {
                self.stats.assists += 1;
            }
            ScoreEventType::ObjectiveCapture => {
                self.stats.objectives_captured += 1;
            }
            ScoreEventType::ObjectiveDefend => {
                self.stats.objectives_defended += 1;
            }
            ScoreEventType::Revive => {
                self.stats.revives += 1;
            }
            _ => {}
        }

        // Handle kill streak
        if event_type.counts_for_streak() {
            if let Some((bonus, _name)) = self.streak.record_kill() {
                self.score += bonus as u64;
            }

            // Multi-kill tracking
            self.multi_kill_count += 1;
            self.multi_kill_timer = 3.0; // 3 second window
        }

        let event = ScoreEvent {
            event_type,
            points,
            base_points: base,
            combo_multiplier: multiplier,
            timestamp: game_time,
            description: event_type.name().to_string(),
            related_entity: None,
        };

        if self.events.len() < MAX_SCORE_EVENTS {
            self.events.push(event.clone());
        }

        event
    }

    /// Record a kill with a specific victim.
    pub fn record_kill(&mut self, victim_id: u64, headshot: bool, game_time: f64) -> Vec<ScoreEvent> {
        let mut events = Vec::new();

        // First blood check
        if self.first_blood_available {
            events.push(self.award(ScoreEventType::FirstBlood, game_time));
            self.first_blood_available = false;
        }

        let event_type = if headshot {
            ScoreEventType::Headshot
        } else {
            ScoreEventType::Kill
        };

        let mut event = self.award(event_type, game_time);
        event.related_entity = Some(victim_id);
        events.push(event);

        // Multi-kill check
        if self.multi_kill_count >= 2 {
            let mk_event = self.award(ScoreEventType::MultiKill, game_time);
            events.push(mk_event);
        }

        events
    }

    /// Record a death.
    pub fn record_death(&mut self, _killer_id: Option<u64>, game_time: f64) {
        self.stats.deaths += 1;
        self.streak.record_death();
        self.combo.reset();
        self.multi_kill_count = 0;
        self.alive = false;

        // Record a zero-point event for history
        let event = ScoreEvent::new(ScoreEventType::Custom(0), 0, game_time)
            .with_description("Death");
        if self.events.len() < MAX_SCORE_EVENTS {
            self.events.push(event);
        }
    }

    /// Record a respawn.
    pub fn record_respawn(&mut self) {
        self.alive = true;
    }

    /// Record damage dealt.
    pub fn record_damage(&mut self, amount: f64) {
        self.stats.damage_dealt += amount;
    }

    /// Record damage taken.
    pub fn record_damage_taken(&mut self, amount: f64) {
        self.stats.damage_taken += amount;
    }

    /// Record a shot fired.
    pub fn record_shot(&mut self, hit: bool) {
        self.stats.shots_fired += 1;
        if hit {
            self.stats.shots_hit += 1;
        }
    }

    /// Record healing performed.
    pub fn record_healing(&mut self, amount: f64, game_time: f64) {
        self.stats.healing_done += amount;
        // Award small points for healing
        let heal_points = (amount / 10.0) as u32;
        if heal_points > 0 {
            self.award(ScoreEventType::Healing, game_time);
        }
    }

    /// Update per-frame state.
    pub fn update(&mut self, dt: f32) {
        self.combo.update(dt);

        // Multi-kill timer
        if self.multi_kill_timer > 0.0 {
            self.multi_kill_timer -= dt;
            if self.multi_kill_timer <= 0.0 {
                self.multi_kill_count = 0;
            }
        }

        self.stats.update_playtime(dt as f64);
    }

    /// Get the score event history.
    pub fn events(&self) -> &[ScoreEvent] {
        &self.events
    }

    /// Get recent events (last N).
    pub fn recent_events(&self, count: usize) -> &[ScoreEvent] {
        let start = if self.events.len() > count {
            self.events.len() - count
        } else {
            0
        };
        &self.events[start..]
    }

    /// Calculate MVP score for this player.
    pub fn mvp_score(&self) -> f32 {
        self.stats.kills as f32 * MVP_KILL_WEIGHT
            + self.stats.assists as f32 * MVP_ASSIST_WEIGHT
            + (self.stats.objectives_captured + self.stats.objectives_defended) as f32
                * MVP_OBJECTIVE_WEIGHT
            + self.stats.deaths as f32 * MVP_DEATH_WEIGHT
            + self.stats.accuracy() * MVP_ACCURACY_WEIGHT
            + self.stats.healing_done as f32 * MVP_HEALING_WEIGHT
            + self.stats.damage_dealt as f32 * MVP_DAMAGE_WEIGHT
    }

    /// Clear first blood availability (call after someone gets first blood).
    pub fn clear_first_blood(&mut self) {
        self.first_blood_available = false;
    }

    /// Reset for a new match.
    pub fn reset(&mut self) {
        self.score = 0;
        self.combo.reset();
        self.streak = KillStreak::new();
        self.events.clear();
        self.stats = StatTracker::new();
        self.alive = true;
        self.multi_kill_count = 0;
        self.multi_kill_timer = 0.0;
        self.first_blood_available = true;
    }
}

// ---------------------------------------------------------------------------
// LeaderboardEntry
// ---------------------------------------------------------------------------

/// A single entry in the leaderboard.
#[derive(Debug, Clone)]
pub struct LeaderboardEntry {
    /// Player entity ID.
    pub player_id: u64,
    /// Player name.
    pub name: String,
    /// Score.
    pub score: u64,
    /// Rank (1-based).
    pub rank: u32,
    /// Team ID.
    pub team_id: Option<u32>,
    /// K/D ratio.
    pub kd_ratio: f32,
    /// Kills.
    pub kills: u32,
    /// Deaths.
    pub deaths: u32,
    /// Assists.
    pub assists: u32,
}

impl Eq for LeaderboardEntry {}
impl PartialEq for LeaderboardEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.player_id == other.player_id
    }
}

impl PartialOrd for LeaderboardEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LeaderboardEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.cmp(&other.score)
    }
}

// ---------------------------------------------------------------------------
// Leaderboard
// ---------------------------------------------------------------------------

/// Ranked leaderboard of players.
pub struct Leaderboard {
    /// Entries sorted by score (descending).
    entries: Vec<LeaderboardEntry>,
    /// Maximum entries.
    max_entries: usize,
    /// Sort criteria.
    pub sort_by: LeaderboardSort,
}

/// Criteria for sorting the leaderboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaderboardSort {
    Score,
    Kills,
    KDRatio,
    Objectives,
    Accuracy,
}

impl Leaderboard {
    /// Create a new leaderboard.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries: max_entries.min(MAX_LEADERBOARD_ENTRIES),
            sort_by: LeaderboardSort::Score,
        }
    }

    /// Update the leaderboard from score trackers.
    pub fn update_from_trackers(&mut self, trackers: &[&ScoreTracker]) {
        self.entries.clear();

        for tracker in trackers {
            let entry = LeaderboardEntry {
                player_id: tracker.player_id,
                name: tracker.player_name.clone(),
                score: tracker.score,
                rank: 0,
                team_id: tracker.team_id,
                kd_ratio: tracker.stats.kd_ratio(),
                kills: tracker.stats.kills,
                deaths: tracker.stats.deaths,
                assists: tracker.stats.assists,
            };
            self.entries.push(entry);
        }

        self.sort();
    }

    /// Sort the leaderboard.
    pub fn sort(&mut self) {
        match self.sort_by {
            LeaderboardSort::Score => {
                self.entries.sort_by(|a, b| b.score.cmp(&a.score));
            }
            LeaderboardSort::Kills => {
                self.entries.sort_by(|a, b| b.kills.cmp(&a.kills));
            }
            LeaderboardSort::KDRatio => {
                self.entries.sort_by(|a, b| {
                    b.kd_ratio
                        .partial_cmp(&a.kd_ratio)
                        .unwrap_or(Ordering::Equal)
                });
            }
            LeaderboardSort::Objectives | LeaderboardSort::Accuracy => {
                self.entries.sort_by(|a, b| b.score.cmp(&a.score));
            }
        }

        // Assign ranks
        for (i, entry) in self.entries.iter_mut().enumerate() {
            entry.rank = (i + 1) as u32;
        }

        // Truncate
        self.entries.truncate(self.max_entries);
    }

    /// Get the top N entries.
    pub fn top(&self, n: usize) -> &[LeaderboardEntry] {
        &self.entries[..n.min(self.entries.len())]
    }

    /// Get all entries.
    pub fn entries(&self) -> &[LeaderboardEntry] {
        &self.entries
    }

    /// Find a player's entry.
    pub fn find_player(&self, player_id: u64) -> Option<&LeaderboardEntry> {
        self.entries.iter().find(|e| e.player_id == player_id)
    }

    /// Get the leader.
    pub fn leader(&self) -> Option<&LeaderboardEntry> {
        self.entries.first()
    }

    /// Entry count.
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Get entries for a specific team.
    pub fn team_entries(&self, team_id: u32) -> Vec<&LeaderboardEntry> {
        self.entries
            .iter()
            .filter(|e| e.team_id == Some(team_id))
            .collect()
    }

    /// Get the team total score.
    pub fn team_score(&self, team_id: u32) -> u64 {
        self.entries
            .iter()
            .filter(|e| e.team_id == Some(team_id))
            .map(|e| e.score)
            .sum()
    }
}

impl Default for Leaderboard {
    fn default() -> Self {
        Self::new(MAX_LEADERBOARD_ENTRIES)
    }
}

// ---------------------------------------------------------------------------
// MatchStats
// ---------------------------------------------------------------------------

/// End-of-match statistics summary.
#[derive(Debug, Clone)]
pub struct MatchStats {
    /// Match duration in seconds.
    pub duration: f64,
    /// Total kills across all players.
    pub total_kills: u32,
    /// Total deaths across all players.
    pub total_deaths: u32,
    /// Total score across all players.
    pub total_score: u64,
    /// MVP player ID.
    pub mvp_id: Option<u64>,
    /// MVP player name.
    pub mvp_name: Option<String>,
    /// MVP score.
    pub mvp_value: f32,
    /// Winning team (if team game).
    pub winning_team: Option<u32>,
    /// Per-team scores.
    pub team_scores: HashMap<u32, u64>,
    /// Per-player summary.
    pub player_summaries: Vec<PlayerMatchSummary>,
    /// Match highlights (notable events).
    pub highlights: Vec<MatchHighlight>,
}

/// Per-player summary for a match.
#[derive(Debug, Clone)]
pub struct PlayerMatchSummary {
    pub player_id: u64,
    pub player_name: String,
    pub score: u64,
    pub kills: u32,
    pub deaths: u32,
    pub assists: u32,
    pub accuracy: f32,
    pub damage_dealt: f64,
    pub healing_done: f64,
    pub objectives: u32,
    pub best_streak: u32,
    pub best_combo: u32,
    pub mvp_score: f32,
}

/// A notable event from the match.
#[derive(Debug, Clone)]
pub struct MatchHighlight {
    /// Description of the event.
    pub description: String,
    /// Player involved.
    pub player_id: u64,
    /// Timestamp.
    pub timestamp: f64,
    /// Type of highlight.
    pub highlight_type: HighlightType,
}

/// Type of match highlight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HighlightType {
    FirstBlood,
    LongestStreak,
    HighestCombo,
    MostKills,
    BestAccuracy,
    MostHealing,
    MostObjectives,
    ClutchPlay,
}

impl MatchStats {
    /// Generate match stats from player trackers.
    pub fn generate(trackers: &[&ScoreTracker], duration: f64) -> Self {
        let total_kills: u32 = trackers.iter().map(|t| t.stats.kills).sum();
        let total_deaths: u32 = trackers.iter().map(|t| t.stats.deaths).sum();
        let total_score: u64 = trackers.iter().map(|t| t.score).sum();

        // Calculate MVP
        let mut mvp_id = None;
        let mut mvp_name = None;
        let mut mvp_value = f32::MIN;

        let mut player_summaries = Vec::new();
        let mut team_scores: HashMap<u32, u64> = HashMap::new();

        for tracker in trackers {
            let mvp = tracker.mvp_score();
            if mvp > mvp_value {
                mvp_value = mvp;
                mvp_id = Some(tracker.player_id);
                mvp_name = Some(tracker.player_name.clone());
            }

            if let Some(team) = tracker.team_id {
                *team_scores.entry(team).or_insert(0) += tracker.score;
            }

            player_summaries.push(PlayerMatchSummary {
                player_id: tracker.player_id,
                player_name: tracker.player_name.clone(),
                score: tracker.score,
                kills: tracker.stats.kills,
                deaths: tracker.stats.deaths,
                assists: tracker.stats.assists,
                accuracy: tracker.stats.accuracy(),
                damage_dealt: tracker.stats.damage_dealt,
                healing_done: tracker.stats.healing_done,
                objectives: tracker.stats.objectives_captured
                    + tracker.stats.objectives_defended,
                best_streak: tracker.streak.best,
                best_combo: tracker.combo.best_combo,
                mvp_score: mvp,
            });
        }

        // Determine winning team
        let winning_team = team_scores
            .iter()
            .max_by_key(|(_, score)| *score)
            .map(|(team, _)| *team);

        // Generate highlights
        let mut highlights = Vec::new();

        // Most kills
        if let Some(best) = player_summaries.iter().max_by_key(|p| p.kills) {
            if best.kills > 0 {
                highlights.push(MatchHighlight {
                    description: format!("{} had the most kills ({}).", best.player_name, best.kills),
                    player_id: best.player_id,
                    timestamp: duration,
                    highlight_type: HighlightType::MostKills,
                });
            }
        }

        // Best accuracy
        if let Some(best) = player_summaries
            .iter()
            .filter(|p| p.kills > 0)
            .max_by(|a, b| a.accuracy.partial_cmp(&b.accuracy).unwrap_or(Ordering::Equal))
        {
            highlights.push(MatchHighlight {
                description: format!(
                    "{} had the best accuracy ({:.1}%).",
                    best.player_name, best.accuracy
                ),
                player_id: best.player_id,
                timestamp: duration,
                highlight_type: HighlightType::BestAccuracy,
            });
        }

        // Longest streak
        if let Some(best) = player_summaries.iter().max_by_key(|p| p.best_streak) {
            if best.best_streak >= 3 {
                highlights.push(MatchHighlight {
                    description: format!(
                        "{} had a {}-kill streak.",
                        best.player_name, best.best_streak
                    ),
                    player_id: best.player_id,
                    timestamp: duration,
                    highlight_type: HighlightType::LongestStreak,
                });
            }
        }

        // Sort summaries by score
        player_summaries.sort_by(|a, b| b.score.cmp(&a.score));

        Self {
            duration,
            total_kills,
            total_deaths,
            total_score,
            mvp_id,
            mvp_name,
            mvp_value,
            winning_team,
            team_scores,
            player_summaries,
            highlights,
        }
    }
}

// ---------------------------------------------------------------------------
// ScoringSystem — top-level manager
// ---------------------------------------------------------------------------

/// Top-level scoring system.
pub struct ScoringSystem {
    /// Per-player score trackers.
    trackers: HashMap<u64, ScoreTracker>,
    /// Global leaderboard.
    pub leaderboard: Leaderboard,
    /// Whether first blood has been claimed.
    first_blood_claimed: bool,
    /// Match start time.
    pub match_start_time: f64,
    /// Whether a match is in progress.
    pub match_active: bool,
}

impl ScoringSystem {
    /// Create a new scoring system.
    pub fn new() -> Self {
        Self {
            trackers: HashMap::new(),
            leaderboard: Leaderboard::default(),
            first_blood_claimed: false,
            match_start_time: 0.0,
            match_active: false,
        }
    }

    /// Start a new match.
    pub fn start_match(&mut self, game_time: f64) {
        self.match_start_time = game_time;
        self.match_active = true;
        self.first_blood_claimed = false;

        for tracker in self.trackers.values_mut() {
            tracker.reset();
        }
    }

    /// Register a player.
    pub fn register_player(
        &mut self,
        player_id: u64,
        name: impl Into<String>,
    ) {
        self.trackers
            .entry(player_id)
            .or_insert_with(|| ScoreTracker::new(player_id, name));
    }

    /// Unregister a player.
    pub fn unregister_player(&mut self, player_id: u64) {
        self.trackers.remove(&player_id);
    }

    /// Get a tracker.
    pub fn get_tracker(&self, player_id: u64) -> Option<&ScoreTracker> {
        self.trackers.get(&player_id)
    }

    /// Get a mutable tracker.
    pub fn get_tracker_mut(&mut self, player_id: u64) -> Option<&mut ScoreTracker> {
        self.trackers.get_mut(&player_id)
    }

    /// Award score to a player.
    pub fn award(
        &mut self,
        player_id: u64,
        event_type: ScoreEventType,
        game_time: f64,
    ) -> Option<ScoreEvent> {
        let tracker = self.trackers.get_mut(&player_id)?;

        // Handle first blood
        if event_type == ScoreEventType::FirstBlood {
            if self.first_blood_claimed {
                return None;
            }
            self.first_blood_claimed = true;
        }

        Some(tracker.award(event_type, game_time))
    }

    /// Record a kill.
    pub fn record_kill(
        &mut self,
        killer_id: u64,
        victim_id: u64,
        headshot: bool,
        game_time: f64,
    ) -> Vec<ScoreEvent> {
        // Handle first blood
        if !self.first_blood_claimed {
            self.first_blood_claimed = true;
        } else {
            if let Some(tracker) = self.trackers.get_mut(&killer_id) {
                tracker.clear_first_blood();
            }
        }

        let events = if let Some(tracker) = self.trackers.get_mut(&killer_id) {
            tracker.record_kill(victim_id, headshot, game_time)
        } else {
            Vec::new()
        };

        if let Some(victim) = self.trackers.get_mut(&victim_id) {
            victim.record_death(Some(killer_id), game_time);
        }

        events
    }

    /// Update all trackers and the leaderboard.
    pub fn update(&mut self, dt: f32) {
        for tracker in self.trackers.values_mut() {
            tracker.update(dt);
        }

        // Update leaderboard
        let tracker_refs: Vec<&ScoreTracker> = self.trackers.values().collect();
        self.leaderboard.update_from_trackers(&tracker_refs);
    }

    /// End the match and generate stats.
    pub fn end_match(&mut self, game_time: f64) -> MatchStats {
        self.match_active = false;
        let duration = game_time - self.match_start_time;
        let tracker_refs: Vec<&ScoreTracker> = self.trackers.values().collect();
        MatchStats::generate(&tracker_refs, duration)
    }

    /// Player count.
    pub fn player_count(&self) -> usize {
        self.trackers.len()
    }

    /// Get all player IDs.
    pub fn player_ids(&self) -> Vec<u64> {
        self.trackers.keys().copied().collect()
    }
}

impl Default for ScoringSystem {
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
    fn test_score_award() {
        let mut tracker = ScoreTracker::new(1, "Player1");
        let event = tracker.award(ScoreEventType::Kill, 1.0);
        assert_eq!(event.points, 100);
        assert_eq!(tracker.score, 100);
        assert_eq!(tracker.stats.kills, 1);
    }

    #[test]
    fn test_combo_multiplier() {
        let mut tracker = ScoreTracker::new(1, "Player1");

        // First kill
        tracker.award(ScoreEventType::Kill, 1.0);
        assert_eq!(tracker.combo.count, 1);

        // Second kill quickly
        tracker.award(ScoreEventType::Kill, 1.5);
        assert!(tracker.combo.multiplier > BASE_COMBO_MULTIPLIER);

        // Score should be higher due to combo
        assert!(tracker.score > 200);
    }

    #[test]
    fn test_kill_streak() {
        let mut streak = KillStreak::new();
        streak.record_kill();
        streak.record_kill();
        let result = streak.record_kill(); // 3rd kill = Killing Spree
        assert!(result.is_some());
        assert_eq!(streak.current, 3);
        assert_eq!(streak.streak_name(), Some("Killing Spree"));

        streak.record_death();
        assert_eq!(streak.current, 0);
    }

    #[test]
    fn test_stat_tracker() {
        let mut stats = StatTracker::new();
        stats.shots_fired = 100;
        stats.shots_hit = 75;
        assert!((stats.accuracy() - 75.0).abs() < 0.1);

        stats.kills = 10;
        stats.deaths = 5;
        assert!((stats.kd_ratio() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_leaderboard() {
        let t1 = ScoreTracker::new(1, "Alice");
        let mut t2 = ScoreTracker::new(2, "Bob");
        t2.award(ScoreEventType::Kill, 1.0);

        let mut lb = Leaderboard::new(10);
        lb.update_from_trackers(&[&t1, &t2]);

        let leader = lb.leader().unwrap();
        assert_eq!(leader.player_id, 2);
        assert_eq!(leader.rank, 1);
    }

    #[test]
    fn test_match_stats() {
        let mut t1 = ScoreTracker::new(1, "Alice");
        let mut t2 = ScoreTracker::new(2, "Bob");

        t1.award(ScoreEventType::Kill, 1.0);
        t1.award(ScoreEventType::Kill, 2.0);
        t2.award(ScoreEventType::Kill, 3.0);

        let stats = MatchStats::generate(&[&t1, &t2], 120.0);
        assert_eq!(stats.total_kills, 3);
        assert!(stats.mvp_id.is_some());
    }

    #[test]
    fn test_scoring_system() {
        let mut system = ScoringSystem::new();
        system.start_match(0.0);

        system.register_player(1, "Player1");
        system.register_player(2, "Player2");

        let events = system.record_kill(1, 2, false, 5.0);
        assert!(!events.is_empty());

        system.update(0.016);

        let match_stats = system.end_match(60.0);
        assert_eq!(match_stats.total_kills, 1);
        assert_eq!(match_stats.total_deaths, 1);
    }
}
