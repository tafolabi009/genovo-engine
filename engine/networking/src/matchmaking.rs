//! Matchmaking system for automatic player matching.
//!
//! Provides ELO/MMR-based skill rating, queue management, and automatic match
//! formation. Supports multiple matchmaking algorithms:
//!
//! - **ELO matchmaker**: pairs players by closest skill rating within a latency
//!   threshold.
//! - **Queue matchmaker**: FIFO with a skill tolerance that widens over time to
//!   prevent indefinite waits.
//!
//! ## ELO Rating Math
//!
//! Expected score:  E_a = 1 / (1 + 10^((R_b - R_a) / 400))
//! New rating:      R'_a = R_a + K * (S_a - E_a)
//!
//! Where:
//! - R_a, R_b: player ratings
//! - S_a: actual score (1.0 = win, 0.5 = draw, 0.0 = loss)
//! - K: development factor (higher for new players)

use std::collections::HashMap;
use std::time::{Duration, Instant};

use genovo_core::{EngineError, EngineResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default ELO rating for new players.
pub const DEFAULT_RATING: f64 = 1200.0;

/// Default K-factor for established players.
pub const DEFAULT_K_FACTOR: f64 = 32.0;

/// K-factor for new players (first 30 games).
pub const NEW_PLAYER_K_FACTOR: f64 = 40.0;

/// K-factor for high-rated players (above 2400).
pub const HIGH_RATING_K_FACTOR: f64 = 16.0;

/// Minimum rating floor.
pub const MIN_RATING: f64 = 100.0;

/// Maximum rating ceiling.
pub const MAX_RATING: f64 = 5000.0;

/// Default skill tolerance for matchmaking (initial window).
pub const DEFAULT_SKILL_TOLERANCE: f64 = 100.0;

/// Maximum skill tolerance (after widening).
pub const MAX_SKILL_TOLERANCE: f64 = 1000.0;

/// How fast the skill tolerance widens per second of waiting.
pub const TOLERANCE_WIDEN_RATE: f64 = 10.0;

/// Maximum queue wait time before auto-matching with anyone.
pub const MAX_QUEUE_WAIT: Duration = Duration::from_secs(300);

/// Default maximum ping for matchmaking.
pub const DEFAULT_MAX_PING: u32 = 200;

/// Minimum players to form a match.
pub const MIN_MATCH_PLAYERS: usize = 2;

// ---------------------------------------------------------------------------
// PlayerId (re-exported type alias)
// ---------------------------------------------------------------------------

/// Player identifier for the matchmaking system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatchPlayerId(pub u32);

impl MatchPlayerId {
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
    pub const fn raw(&self) -> u32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// ELO Rating System
// ---------------------------------------------------------------------------

/// ELO rating calculator.
///
/// Implements the standard ELO rating system with variable K-factor based on
/// the number of games played and current rating.
pub struct EloRating;

impl EloRating {
    /// Calculate the expected score of player A against player B.
    ///
    /// Returns a value between 0.0 and 1.0, where 1.0 means player A
    /// is expected to always win.
    pub fn expected_score(rating_a: f64, rating_b: f64) -> f64 {
        1.0 / (1.0 + 10.0_f64.powf((rating_b - rating_a) / 400.0))
    }

    /// Calculate the K-factor based on games played and current rating.
    pub fn k_factor(rating: f64, games_played: u32) -> f64 {
        if games_played < 30 {
            NEW_PLAYER_K_FACTOR
        } else if rating >= 2400.0 {
            HIGH_RATING_K_FACTOR
        } else {
            DEFAULT_K_FACTOR
        }
    }

    /// Calculate the new rating after a game.
    ///
    /// `actual_score` should be 1.0 for a win, 0.5 for a draw, 0.0 for a loss.
    pub fn new_rating(
        current_rating: f64,
        opponent_rating: f64,
        actual_score: f64,
        games_played: u32,
    ) -> f64 {
        let k = Self::k_factor(current_rating, games_played);
        let expected = Self::expected_score(current_rating, opponent_rating);
        let new_rating = current_rating + k * (actual_score - expected);
        new_rating.clamp(MIN_RATING, MAX_RATING)
    }

    /// Update ratings for a two-player match.
    ///
    /// Returns (new_rating_a, new_rating_b).
    /// `outcome`: 1.0 = A wins, 0.0 = B wins, 0.5 = draw.
    pub fn update_pair(
        rating_a: f64,
        games_a: u32,
        rating_b: f64,
        games_b: u32,
        outcome: f64,
    ) -> (f64, f64) {
        let new_a = Self::new_rating(rating_a, rating_b, outcome, games_a);
        let new_b = Self::new_rating(rating_b, rating_a, 1.0 - outcome, games_b);
        (new_a, new_b)
    }

    /// Update ratings for a team match.
    ///
    /// Each team has a list of (player_id, rating, games_played).
    /// The team rating is the average of all members.
    pub fn update_team_match(
        winners: &[(MatchPlayerId, f64, u32)],
        losers: &[(MatchPlayerId, f64, u32)],
    ) -> Vec<(MatchPlayerId, f64)> {
        let avg_winner: f64 = winners.iter().map(|(_, r, _)| r).sum::<f64>()
            / winners.len().max(1) as f64;
        let avg_loser: f64 = losers.iter().map(|(_, r, _)| r).sum::<f64>()
            / losers.len().max(1) as f64;

        let mut results = Vec::new();

        for &(id, rating, games) in winners {
            let new_rating = Self::new_rating(rating, avg_loser, 1.0, games);
            results.push((id, new_rating));
        }

        for &(id, rating, games) in losers {
            let new_rating = Self::new_rating(rating, avg_winner, 0.0, games);
            results.push((id, new_rating));
        }

        results
    }
}

// ---------------------------------------------------------------------------
// PlayerProfile
// ---------------------------------------------------------------------------

/// A player's matchmaking profile.
#[derive(Debug, Clone)]
pub struct PlayerProfile {
    /// Unique player identifier.
    pub id: MatchPlayerId,
    /// Display name.
    pub name: String,
    /// Skill rating (ELO/MMR).
    pub skill_rating: f64,
    /// Number of games played (for K-factor calculation).
    pub games_played: u32,
    /// Total wins.
    pub wins: u32,
    /// Total losses.
    pub losses: u32,
    /// Total draws.
    pub draws: u32,
    /// Player's geographic region (for latency matching).
    pub region: String,
    /// Estimated latency to the server, in milliseconds.
    pub latency_ms: u32,
    /// Win rate (derived).
    pub win_rate: f64,
}

impl PlayerProfile {
    /// Create a new player profile with default rating.
    pub fn new(id: MatchPlayerId, name: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            skill_rating: DEFAULT_RATING,
            games_played: 0,
            wins: 0,
            losses: 0,
            draws: 0,
            region: region.into(),
            latency_ms: 0,
            win_rate: 0.0,
        }
    }

    /// Create a profile with a specific rating.
    pub fn with_rating(
        id: MatchPlayerId,
        name: impl Into<String>,
        region: impl Into<String>,
        rating: f64,
    ) -> Self {
        let mut p = Self::new(id, name, region);
        p.skill_rating = rating.clamp(MIN_RATING, MAX_RATING);
        p
    }

    /// Record a win.
    pub fn record_win(&mut self, opponent_rating: f64) {
        self.skill_rating =
            EloRating::new_rating(self.skill_rating, opponent_rating, 1.0, self.games_played);
        self.games_played += 1;
        self.wins += 1;
        self.update_win_rate();
    }

    /// Record a loss.
    pub fn record_loss(&mut self, opponent_rating: f64) {
        self.skill_rating =
            EloRating::new_rating(self.skill_rating, opponent_rating, 0.0, self.games_played);
        self.games_played += 1;
        self.losses += 1;
        self.update_win_rate();
    }

    /// Record a draw.
    pub fn record_draw(&mut self, opponent_rating: f64) {
        self.skill_rating =
            EloRating::new_rating(self.skill_rating, opponent_rating, 0.5, self.games_played);
        self.games_played += 1;
        self.draws += 1;
        self.update_win_rate();
    }

    /// Update the win rate.
    fn update_win_rate(&mut self) {
        if self.games_played > 0 {
            self.win_rate = self.wins as f64 / self.games_played as f64;
        }
    }

    /// Returns the K-factor for this player.
    pub fn k_factor(&self) -> f64 {
        EloRating::k_factor(self.skill_rating, self.games_played)
    }
}

// ---------------------------------------------------------------------------
// MatchPreferences
// ---------------------------------------------------------------------------

/// Player preferences for matchmaking.
#[derive(Debug, Clone)]
pub struct MatchPreferences {
    /// Preferred game mode.
    pub game_mode: String,
    /// Maximum acceptable skill difference.
    pub skill_range: f64,
    /// Maximum acceptable ping in milliseconds.
    pub max_ping: u32,
    /// Preferred team size (0 = any).
    pub team_size: u32,
    /// Preferred regions (empty = any).
    pub preferred_regions: Vec<String>,
}

impl MatchPreferences {
    /// Create default match preferences.
    pub fn new(game_mode: impl Into<String>) -> Self {
        Self {
            game_mode: game_mode.into(),
            skill_range: DEFAULT_SKILL_TOLERANCE,
            max_ping: DEFAULT_MAX_PING,
            team_size: 0,
            preferred_regions: Vec::new(),
        }
    }

    /// Check if a given player is compatible with these preferences.
    pub fn is_compatible(&self, player: &PlayerProfile, my_rating: f64) -> bool {
        // Check ping.
        if player.latency_ms > self.max_ping {
            return false;
        }

        // Check skill range.
        let skill_diff = (player.skill_rating - my_rating).abs();
        if skill_diff > self.skill_range {
            return false;
        }

        // Check region.
        if !self.preferred_regions.is_empty()
            && !self.preferred_regions.contains(&player.region)
        {
            return false;
        }

        true
    }
}

impl Default for MatchPreferences {
    fn default() -> Self {
        Self::new("default")
    }
}

// ---------------------------------------------------------------------------
// QueueEntry
// ---------------------------------------------------------------------------

/// A player waiting in the matchmaking queue.
#[derive(Debug, Clone)]
pub struct QueueEntry {
    /// The player's profile.
    pub profile: PlayerProfile,
    /// The player's match preferences.
    pub preferences: MatchPreferences,
    /// When the player entered the queue.
    pub enqueued_at: Instant,
    /// Current effective skill tolerance (widens over time).
    pub effective_tolerance: f64,
}

impl QueueEntry {
    /// Create a new queue entry.
    pub fn new(profile: PlayerProfile, preferences: MatchPreferences) -> Self {
        let initial_tolerance = preferences.skill_range;
        Self {
            profile,
            preferences,
            enqueued_at: Instant::now(),
            effective_tolerance: initial_tolerance,
        }
    }

    /// Returns how long this entry has been waiting, in seconds.
    pub fn wait_time_secs(&self) -> f64 {
        Instant::now()
            .duration_since(self.enqueued_at)
            .as_secs_f64()
    }

    /// Update the effective tolerance based on wait time.
    ///
    /// The tolerance widens linearly over time to prevent indefinite waits.
    pub fn update_tolerance(&mut self) {
        let wait_secs = self.wait_time_secs();
        let base = self.preferences.skill_range;
        self.effective_tolerance =
            (base + wait_secs * TOLERANCE_WIDEN_RATE).min(MAX_SKILL_TOLERANCE);
    }

    /// Returns true if this entry has exceeded the maximum wait time.
    pub fn is_expired(&self) -> bool {
        Instant::now().duration_since(self.enqueued_at) > MAX_QUEUE_WAIT
    }

    /// Check if this entry is compatible with another entry.
    pub fn is_compatible(&self, other: &QueueEntry) -> bool {
        // Check game mode.
        if self.preferences.game_mode != other.preferences.game_mode {
            return false;
        }

        // Check skill tolerance (mutual).
        let skill_diff = (self.profile.skill_rating - other.profile.skill_rating).abs();
        if skill_diff > self.effective_tolerance || skill_diff > other.effective_tolerance {
            return false;
        }

        // Check ping.
        if self.profile.latency_ms > other.preferences.max_ping
            || other.profile.latency_ms > self.preferences.max_ping
        {
            return false;
        }

        true
    }
}

// ---------------------------------------------------------------------------
// Match
// ---------------------------------------------------------------------------

/// A formed match ready to be started.
#[derive(Debug, Clone)]
pub struct Match {
    /// Unique match identifier.
    pub id: u64,
    /// Players in the match.
    pub players: Vec<MatchPlayerId>,
    /// Team assignments (player_id -> team).
    pub teams: HashMap<MatchPlayerId, u32>,
    /// Average skill rating of all players.
    pub avg_skill: f64,
    /// Skill spread (max - min rating).
    pub skill_spread: f64,
    /// The game mode for this match.
    pub game_mode: String,
    /// When this match was created.
    pub created_at: Instant,
}

impl Match {
    /// Create a new match from a list of players.
    pub fn new(
        id: u64,
        players: Vec<(MatchPlayerId, f64)>,
        game_mode: impl Into<String>,
    ) -> Self {
        let player_ids: Vec<MatchPlayerId> = players.iter().map(|(id, _)| *id).collect();
        let ratings: Vec<f64> = players.iter().map(|(_, r)| *r).collect();

        let avg_skill = if ratings.is_empty() {
            0.0
        } else {
            ratings.iter().sum::<f64>() / ratings.len() as f64
        };

        let skill_spread = if ratings.is_empty() {
            0.0
        } else {
            let min = ratings.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = ratings.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            max - min
        };

        Self {
            id,
            players: player_ids,
            teams: HashMap::new(),
            avg_skill,
            skill_spread,
            game_mode: game_mode.into(),
            created_at: Instant::now(),
        }
    }

    /// Assign players to balanced teams.
    ///
    /// Uses a greedy algorithm: sort players by rating, then alternate assignment
    /// to minimize the total skill difference between teams.
    pub fn balance_teams(&mut self, team_count: u32, ratings: &HashMap<MatchPlayerId, f64>) {
        if team_count == 0 || self.players.is_empty() {
            return;
        }

        // Sort players by rating (descending).
        let mut sorted: Vec<(MatchPlayerId, f64)> = self
            .players
            .iter()
            .map(|&id| {
                let rating = ratings.get(&id).copied().unwrap_or(DEFAULT_RATING);
                (id, rating)
            })
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Track team totals for greedy assignment.
        let mut team_totals: Vec<f64> = vec![0.0; team_count as usize];
        let mut team_counts: Vec<usize> = vec![0; team_count as usize];
        let max_per_team = (self.players.len() + team_count as usize - 1) / team_count as usize;

        self.teams.clear();
        for (id, rating) in sorted {
            // Find the team with the lowest total rating that still has capacity.
            let best_team = (0..team_count as usize)
                .filter(|&t| team_counts[t] < max_per_team)
                .min_by(|&a, &b| {
                    team_totals[a]
                        .partial_cmp(&team_totals[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);

            self.teams.insert(id, best_team as u32 + 1);
            team_totals[best_team] += rating;
            team_counts[best_team] += 1;
        }
    }

    /// Calculate the team skill difference.
    ///
    /// Returns the difference between the highest and lowest team average ratings.
    pub fn team_skill_difference(&self, ratings: &HashMap<MatchPlayerId, f64>) -> f64 {
        if self.teams.is_empty() {
            return 0.0;
        }

        let mut team_sums: HashMap<u32, (f64, usize)> = HashMap::new();
        for (&pid, &team) in &self.teams {
            let rating = ratings.get(&pid).copied().unwrap_or(DEFAULT_RATING);
            let entry = team_sums.entry(team).or_insert((0.0, 0));
            entry.0 += rating;
            entry.1 += 1;
        }

        let averages: Vec<f64> = team_sums
            .values()
            .filter(|(_, count)| *count > 0)
            .map(|(sum, count)| sum / *count as f64)
            .collect();

        if averages.is_empty() {
            return 0.0;
        }

        let min = averages.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = averages.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        max - min
    }
}

// ---------------------------------------------------------------------------
// MatchmakingAlgorithm trait
// ---------------------------------------------------------------------------

/// Trait for matchmaking algorithms.
pub trait MatchmakingAlgorithm: Send + Sync {
    /// Try to form matches from the current queue.
    ///
    /// Returns a list of matches that were formed.
    fn find_matches(
        &self,
        queue: &[QueueEntry],
        team_size: u32,
    ) -> Vec<Vec<usize>>;

    /// Name of this algorithm.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// EloMatchmaker
// ---------------------------------------------------------------------------

/// Matchmaker that pairs players by closest skill rating.
///
/// Pairs the closest-rated compatible players first, forming matches of
/// the requested team size.
pub struct EloMatchmaker;

impl EloMatchmaker {
    /// Create a new ELO-based matchmaker.
    pub fn new() -> Self {
        Self
    }
}

impl Default for EloMatchmaker {
    fn default() -> Self {
        Self::new()
    }
}

impl MatchmakingAlgorithm for EloMatchmaker {
    fn find_matches(
        &self,
        queue: &[QueueEntry],
        team_size: u32,
    ) -> Vec<Vec<usize>> {
        let match_size = if team_size == 0 {
            MIN_MATCH_PLAYERS
        } else {
            (team_size * 2) as usize // 2 teams
        };

        if queue.len() < match_size {
            return Vec::new();
        }

        // Sort indices by skill rating.
        let mut indices: Vec<usize> = (0..queue.len()).collect();
        indices.sort_by(|&a, &b| {
            queue[a]
                .profile
                .skill_rating
                .partial_cmp(&queue[b].profile.skill_rating)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut matches = Vec::new();
        let mut used = vec![false; queue.len()];

        // Greedy: pick windows of adjacent-rated compatible players.
        let mut i = 0;
        while i + match_size <= indices.len() {
            // Find a window of match_size consecutive compatible players.
            let window: Vec<usize> = indices[i..i + match_size]
                .iter()
                .copied()
                .filter(|&idx| !used[idx])
                .collect();

            if window.len() < match_size {
                i += 1;
                continue;
            }

            // Check all-pairs compatibility.
            let all_compatible = window.iter().all(|&a| {
                window.iter().all(|&b| {
                    a == b || queue[a].is_compatible(&queue[b])
                })
            });

            if all_compatible {
                let match_group: Vec<usize> = window[..match_size].to_vec();
                for &idx in &match_group {
                    used[idx] = true;
                }
                matches.push(match_group);
                i += match_size;
            } else {
                i += 1;
            }
        }

        matches
    }

    fn name(&self) -> &str {
        "elo"
    }
}

// ---------------------------------------------------------------------------
// QueueMatchmaker
// ---------------------------------------------------------------------------

/// FIFO-based matchmaker with skill tolerance that widens over time.
///
/// Prioritizes wait time: players who have been waiting longer get matched
/// first, with progressively wider skill brackets.
pub struct QueueMatchmaker;

impl QueueMatchmaker {
    /// Create a new queue-based matchmaker.
    pub fn new() -> Self {
        Self
    }
}

impl Default for QueueMatchmaker {
    fn default() -> Self {
        Self::new()
    }
}

impl MatchmakingAlgorithm for QueueMatchmaker {
    fn find_matches(
        &self,
        queue: &[QueueEntry],
        team_size: u32,
    ) -> Vec<Vec<usize>> {
        let match_size = if team_size == 0 {
            MIN_MATCH_PLAYERS
        } else {
            (team_size * 2) as usize
        };

        if queue.len() < match_size {
            return Vec::new();
        }

        // Sort by wait time (longest first -- FIFO priority).
        let mut indices: Vec<usize> = (0..queue.len()).collect();
        indices.sort_by(|&a, &b| {
            queue[b]
                .wait_time_secs()
                .partial_cmp(&queue[a].wait_time_secs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut matches = Vec::new();
        let mut used = vec![false; queue.len()];

        for &anchor in &indices {
            if used[anchor] {
                continue;
            }

            // Try to build a match around this anchor.
            let mut group = vec![anchor];

            for &candidate in &indices {
                if group.len() >= match_size {
                    break;
                }
                if used[candidate] || candidate == anchor {
                    continue;
                }

                // Check compatibility with all current group members.
                let compatible = group.iter().all(|&g| {
                    queue[g].is_compatible(&queue[candidate])
                });

                if compatible {
                    group.push(candidate);
                }
            }

            if group.len() >= match_size {
                for &idx in &group[..match_size] {
                    used[idx] = true;
                }
                matches.push(group[..match_size].to_vec());
            }
        }

        matches
    }

    fn name(&self) -> &str {
        "queue"
    }
}

// ---------------------------------------------------------------------------
// MatchmakingQueue
// ---------------------------------------------------------------------------

/// The main matchmaking queue.
///
/// Manages player enqueue/dequeue, runs the matchmaking algorithm, and
/// produces formed matches.
pub struct MatchmakingQueue {
    /// Players waiting in the queue.
    queue: Vec<QueueEntry>,
    /// Player ID to queue index mapping.
    player_index: HashMap<MatchPlayerId, usize>,
    /// The matchmaking algorithm to use.
    algorithm: Box<dyn MatchmakingAlgorithm>,
    /// Default team size.
    team_size: u32,
    /// Formed matches.
    formed_matches: Vec<Match>,
    /// Next match ID.
    next_match_id: u64,
    /// Total matches formed.
    total_matches_formed: u64,
    /// Total players processed.
    total_players_processed: u64,
    /// Average wait time in seconds (EWMA).
    avg_wait_time_secs: f64,
}

impl MatchmakingQueue {
    /// Create a new matchmaking queue with the given algorithm.
    pub fn new(algorithm: Box<dyn MatchmakingAlgorithm>, team_size: u32) -> Self {
        Self {
            queue: Vec::new(),
            player_index: HashMap::new(),
            algorithm,
            team_size,
            formed_matches: Vec::new(),
            next_match_id: 1,
            total_matches_formed: 0,
            total_players_processed: 0,
            avg_wait_time_secs: 0.0,
        }
    }

    /// Create a queue with the ELO matchmaker.
    pub fn with_elo(team_size: u32) -> Self {
        Self::new(Box::new(EloMatchmaker::new()), team_size)
    }

    /// Create a queue with the queue-based matchmaker.
    pub fn with_queue(team_size: u32) -> Self {
        Self::new(Box::new(QueueMatchmaker::new()), team_size)
    }

    /// Add a player to the queue.
    pub fn enqueue(
        &mut self,
        profile: PlayerProfile,
        preferences: MatchPreferences,
    ) -> EngineResult<()> {
        if self.player_index.contains_key(&profile.id) {
            return Err(EngineError::InvalidArgument(
                "player is already in queue".into(),
            ));
        }

        let id = profile.id;
        let entry = QueueEntry::new(profile, preferences);
        let index = self.queue.len();
        self.queue.push(entry);
        self.player_index.insert(id, index);
        self.total_players_processed += 1;

        log::debug!("Player {} enqueued for matchmaking", id.raw());
        Ok(())
    }

    /// Remove a player from the queue.
    pub fn dequeue(&mut self, player_id: MatchPlayerId) -> EngineResult<()> {
        let index = self.player_index.remove(&player_id).ok_or_else(|| {
            EngineError::NotFound("player not in queue".into())
        })?;

        self.queue.swap_remove(index);

        // Update the index of the swapped element.
        if index < self.queue.len() {
            let swapped_id = self.queue[index].profile.id;
            self.player_index.insert(swapped_id, index);
        }

        log::debug!("Player {} dequeued from matchmaking", player_id.raw());
        Ok(())
    }

    /// Update the matchmaking queue: widen tolerances and try to form matches.
    ///
    /// Should be called periodically (e.g., every second).
    /// Returns the number of matches formed in this update.
    pub fn update(&mut self) -> usize {
        // Update tolerances.
        for entry in &mut self.queue {
            entry.update_tolerance();
        }

        // Remove expired entries.
        let expired: Vec<MatchPlayerId> = self
            .queue
            .iter()
            .filter(|e| e.is_expired())
            .map(|e| e.profile.id)
            .collect();
        for id in expired {
            let _ = self.dequeue(id);
            log::debug!("Player {} removed from queue (expired)", id.raw());
        }

        // Run the matching algorithm.
        let match_groups = self.algorithm.find_matches(&self.queue, self.team_size);

        if match_groups.is_empty() {
            return 0;
        }

        let matches_formed = match_groups.len();

        // Collect indices to remove (sorted descending to remove safely).
        let mut all_indices: Vec<usize> = match_groups
            .iter()
            .flat_map(|g| g.iter().copied())
            .collect();
        all_indices.sort_unstable();
        all_indices.dedup();

        // Build matches.
        for group in &match_groups {
            let players: Vec<(MatchPlayerId, f64)> = group
                .iter()
                .map(|&idx| {
                    let entry = &self.queue[idx];
                    (entry.profile.id, entry.profile.skill_rating)
                })
                .collect();

            let game_mode = self.queue[group[0]].preferences.game_mode.clone();

            // Track wait times.
            for &idx in group {
                let wait = self.queue[idx].wait_time_secs();
                self.avg_wait_time_secs =
                    0.9 * self.avg_wait_time_secs + 0.1 * wait;
            }

            let mut m = Match::new(self.next_match_id, players, game_mode);
            self.next_match_id += 1;

            // Balance teams.
            if self.team_size > 0 {
                let ratings: HashMap<MatchPlayerId, f64> = group
                    .iter()
                    .map(|&idx| {
                        let e = &self.queue[idx];
                        (e.profile.id, e.profile.skill_rating)
                    })
                    .collect();
                m.balance_teams(2, &ratings);
            }

            self.formed_matches.push(m);
            self.total_matches_formed += 1;
        }

        // Remove matched players from queue (descending order).
        all_indices.sort_unstable_by(|a, b| b.cmp(a));
        for idx in all_indices {
            if idx < self.queue.len() {
                let id = self.queue[idx].profile.id;
                self.player_index.remove(&id);
                self.queue.swap_remove(idx);
                // Update swapped element's index.
                if idx < self.queue.len() {
                    let swapped_id = self.queue[idx].profile.id;
                    self.player_index.insert(swapped_id, idx);
                }
            }
        }

        matches_formed
    }

    /// Take all formed matches.
    pub fn take_matches(&mut self) -> Vec<Match> {
        std::mem::take(&mut self.formed_matches)
    }

    /// Returns the number of players currently in the queue.
    pub fn queue_size(&self) -> usize {
        self.queue.len()
    }

    /// Returns the number of formed but untaken matches.
    pub fn pending_match_count(&self) -> usize {
        self.formed_matches.len()
    }

    /// Returns true if a player is in the queue.
    pub fn is_queued(&self, player_id: MatchPlayerId) -> bool {
        self.player_index.contains_key(&player_id)
    }

    /// Returns the total number of matches formed.
    pub fn total_matches(&self) -> u64 {
        self.total_matches_formed
    }

    /// Returns the average wait time in seconds.
    pub fn avg_wait_time(&self) -> f64 {
        self.avg_wait_time_secs
    }

    /// Returns the algorithm name.
    pub fn algorithm_name(&self) -> &str {
        self.algorithm.name()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_profile(id: u32, rating: f64) -> PlayerProfile {
        PlayerProfile::with_rating(
            MatchPlayerId::new(id),
            format!("Player{}", id),
            "us-east",
            rating,
        )
    }

    fn make_prefs(mode: &str) -> MatchPreferences {
        MatchPreferences::new(mode)
    }

    // -----------------------------------------------------------------------
    // ELO Rating
    // -----------------------------------------------------------------------

    #[test]
    fn test_elo_expected_score_equal() {
        let e = EloRating::expected_score(1200.0, 1200.0);
        assert!((e - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_elo_expected_score_higher() {
        let e = EloRating::expected_score(1400.0, 1200.0);
        assert!(e > 0.5);
        assert!(e < 1.0);
    }

    #[test]
    fn test_elo_expected_score_lower() {
        let e = EloRating::expected_score(1000.0, 1400.0);
        assert!(e < 0.5);
        assert!(e > 0.0);
    }

    #[test]
    fn test_elo_expected_score_symmetry() {
        let e_a = EloRating::expected_score(1500.0, 1200.0);
        let e_b = EloRating::expected_score(1200.0, 1500.0);
        assert!((e_a + e_b - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_elo_rating_win() {
        let new_rating = EloRating::new_rating(1200.0, 1200.0, 1.0, 50);
        assert!(new_rating > 1200.0);
        // With equal ratings and K=32, winning should give +16.
        assert!((new_rating - 1216.0).abs() < 1.0);
    }

    #[test]
    fn test_elo_rating_loss() {
        let new_rating = EloRating::new_rating(1200.0, 1200.0, 0.0, 50);
        assert!(new_rating < 1200.0);
        // With equal ratings and K=32, losing should give -16.
        assert!((new_rating - 1184.0).abs() < 1.0);
    }

    #[test]
    fn test_elo_rating_draw() {
        let new_rating = EloRating::new_rating(1200.0, 1200.0, 0.5, 50);
        assert!((new_rating - 1200.0).abs() < 0.1);
    }

    #[test]
    fn test_elo_rating_upset_win() {
        // Lower-rated player beats higher-rated: should gain a lot.
        let new_rating = EloRating::new_rating(1000.0, 1400.0, 1.0, 50);
        assert!(new_rating > 1020.0); // big gain
    }

    #[test]
    fn test_elo_rating_upset_loss() {
        // Higher-rated player loses to lower-rated: should lose a lot.
        let new_rating = EloRating::new_rating(1400.0, 1000.0, 0.0, 50);
        assert!(new_rating < 1380.0);
    }

    #[test]
    fn test_elo_rating_clamped() {
        // Rating should not go below MIN_RATING.
        let r = EloRating::new_rating(MIN_RATING, 3000.0, 0.0, 50);
        assert!(r >= MIN_RATING);

        // Rating should not go above MAX_RATING.
        let r = EloRating::new_rating(MAX_RATING, MIN_RATING, 1.0, 50);
        assert!(r <= MAX_RATING);
    }

    #[test]
    fn test_elo_k_factor() {
        assert_eq!(EloRating::k_factor(1200.0, 5), NEW_PLAYER_K_FACTOR);
        assert_eq!(EloRating::k_factor(1200.0, 50), DEFAULT_K_FACTOR);
        assert_eq!(EloRating::k_factor(2500.0, 100), HIGH_RATING_K_FACTOR);
    }

    #[test]
    fn test_elo_update_pair() {
        let (new_a, new_b) = EloRating::update_pair(1200.0, 50, 1200.0, 50, 1.0);
        assert!(new_a > 1200.0);
        assert!(new_b < 1200.0);
        // Sum should be approximately conserved (may differ slightly due to clamping).
        assert!((new_a + new_b - 2400.0).abs() < 1.0);
    }

    #[test]
    fn test_elo_team_match() {
        let winners = vec![
            (MatchPlayerId::new(1), 1200.0, 50u32),
            (MatchPlayerId::new(2), 1300.0, 50u32),
        ];
        let losers = vec![
            (MatchPlayerId::new(3), 1250.0, 50u32),
            (MatchPlayerId::new(4), 1150.0, 50u32),
        ];

        let results = EloRating::update_team_match(&winners, &losers);
        assert_eq!(results.len(), 4);

        // Winners should gain rating.
        for &(id, rating) in &results {
            if id == MatchPlayerId::new(1) || id == MatchPlayerId::new(2) {
                assert!(rating > 1150.0); // gained
            }
        }
    }

    // -----------------------------------------------------------------------
    // PlayerProfile
    // -----------------------------------------------------------------------

    #[test]
    fn test_player_profile_new() {
        let p = PlayerProfile::new(MatchPlayerId::new(1), "Alice", "us-east");
        assert_eq!(p.skill_rating, DEFAULT_RATING);
        assert_eq!(p.games_played, 0);
        assert_eq!(p.wins, 0);
    }

    #[test]
    fn test_player_profile_record() {
        let mut p = PlayerProfile::new(MatchPlayerId::new(1), "Alice", "us-east");

        p.record_win(1200.0);
        assert_eq!(p.games_played, 1);
        assert_eq!(p.wins, 1);
        assert!(p.skill_rating > DEFAULT_RATING);
        assert!((p.win_rate - 1.0).abs() < 0.001);

        p.record_loss(1200.0);
        assert_eq!(p.games_played, 2);
        assert_eq!(p.losses, 1);
        assert!((p.win_rate - 0.5).abs() < 0.001);

        p.record_draw(1200.0);
        assert_eq!(p.games_played, 3);
        assert_eq!(p.draws, 1);
    }

    // -----------------------------------------------------------------------
    // MatchPreferences
    // -----------------------------------------------------------------------

    #[test]
    fn test_preferences_compatibility() {
        let prefs = MatchPreferences {
            game_mode: "dm".into(),
            skill_range: 200.0,
            max_ping: 100,
            team_size: 0,
            preferred_regions: vec!["us-east".into()],
        };

        let mut player = make_profile(1, 1300.0);
        player.latency_ms = 50;
        assert!(prefs.is_compatible(&player, 1200.0));

        // Too high ping.
        player.latency_ms = 150;
        assert!(!prefs.is_compatible(&player, 1200.0));

        // Too far in skill.
        player.latency_ms = 50;
        assert!(!prefs.is_compatible(&player, 800.0));

        // Wrong region.
        player.region = "eu-west".into();
        assert!(!prefs.is_compatible(&player, 1200.0));
    }

    // -----------------------------------------------------------------------
    // QueueEntry
    // -----------------------------------------------------------------------

    #[test]
    fn test_queue_entry_tolerance_widens() {
        let profile = make_profile(1, 1200.0);
        let prefs = MatchPreferences::new("dm");
        let mut entry = QueueEntry::new(profile, prefs);

        let initial = entry.effective_tolerance;
        // Simulate time passing by updating tolerance.
        std::thread::sleep(std::time::Duration::from_millis(50));
        entry.update_tolerance();

        // Tolerance should have widened at least slightly.
        assert!(entry.effective_tolerance >= initial);
    }

    #[test]
    fn test_queue_entry_compatibility() {
        let entry_a = QueueEntry::new(make_profile(1, 1200.0), MatchPreferences::new("dm"));
        let entry_b = QueueEntry::new(make_profile(2, 1250.0), MatchPreferences::new("dm"));
        assert!(entry_a.is_compatible(&entry_b));

        // Different game mode.
        let entry_c = QueueEntry::new(make_profile(3, 1200.0), MatchPreferences::new("ctf"));
        assert!(!entry_a.is_compatible(&entry_c));
    }

    // -----------------------------------------------------------------------
    // Match
    // -----------------------------------------------------------------------

    #[test]
    fn test_match_creation() {
        let players = vec![
            (MatchPlayerId::new(1), 1200.0),
            (MatchPlayerId::new(2), 1300.0),
            (MatchPlayerId::new(3), 1100.0),
            (MatchPlayerId::new(4), 1400.0),
        ];

        let m = Match::new(1, players, "dm");
        assert_eq!(m.players.len(), 4);
        assert!((m.avg_skill - 1250.0).abs() < 0.1);
        assert!((m.skill_spread - 300.0).abs() < 0.1);
    }

    #[test]
    fn test_match_balance_teams() {
        let players = vec![
            (MatchPlayerId::new(1), 1400.0),
            (MatchPlayerId::new(2), 1300.0),
            (MatchPlayerId::new(3), 1200.0),
            (MatchPlayerId::new(4), 1100.0),
        ];

        let ratings: HashMap<MatchPlayerId, f64> = players.iter().copied().collect();
        let mut m = Match::new(1, players, "dm");
        m.balance_teams(2, &ratings);

        assert_eq!(m.teams.len(), 4);

        // Check that both teams have 2 players each.
        let team1_count = m.teams.values().filter(|&&t| t == 1).count();
        let team2_count = m.teams.values().filter(|&&t| t == 2).count();
        assert_eq!(team1_count, 2);
        assert_eq!(team2_count, 2);

        // The greedy algorithm should pair 1400+1100 and 1300+1200
        // or similar, so difference should be small.
        let diff = m.team_skill_difference(&ratings);
        assert!(diff < 200.0);
    }

    #[test]
    fn test_match_balance_teams_odd() {
        let players = vec![
            (MatchPlayerId::new(1), 1500.0),
            (MatchPlayerId::new(2), 1400.0),
            (MatchPlayerId::new(3), 1300.0),
        ];

        let ratings: HashMap<MatchPlayerId, f64> = players.iter().copied().collect();
        let mut m = Match::new(1, players, "dm");
        m.balance_teams(2, &ratings);

        assert_eq!(m.teams.len(), 3);
        // One team should have 2, other should have 1.
        let team1_count = m.teams.values().filter(|&&t| t == 1).count();
        let team2_count = m.teams.values().filter(|&&t| t == 2).count();
        assert!(team1_count + team2_count == 3);
    }

    // -----------------------------------------------------------------------
    // MatchmakingQueue
    // -----------------------------------------------------------------------

    #[test]
    fn test_queue_enqueue_dequeue() {
        let mut queue = MatchmakingQueue::with_elo(0);
        let profile = make_profile(1, 1200.0);
        queue.enqueue(profile, make_prefs("dm")).unwrap();

        assert_eq!(queue.queue_size(), 1);
        assert!(queue.is_queued(MatchPlayerId::new(1)));

        queue.dequeue(MatchPlayerId::new(1)).unwrap();
        assert_eq!(queue.queue_size(), 0);
        assert!(!queue.is_queued(MatchPlayerId::new(1)));
    }

    #[test]
    fn test_queue_duplicate_enqueue() {
        let mut queue = MatchmakingQueue::with_elo(0);
        let profile = make_profile(1, 1200.0);
        queue.enqueue(profile.clone(), make_prefs("dm")).unwrap();
        let result = queue.enqueue(profile, make_prefs("dm"));
        assert!(result.is_err());
    }

    #[test]
    fn test_queue_form_match_elo() {
        let mut queue = MatchmakingQueue::with_elo(0);

        // Add two players with similar ratings.
        queue
            .enqueue(make_profile(1, 1200.0), make_prefs("dm"))
            .unwrap();
        queue
            .enqueue(make_profile(2, 1250.0), make_prefs("dm"))
            .unwrap();

        let matches_formed = queue.update();
        assert_eq!(matches_formed, 1);

        let matches = queue.take_matches();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].players.len(), 2);
        assert_eq!(queue.queue_size(), 0);
    }

    #[test]
    fn test_queue_form_match_queue_algo() {
        let mut queue = MatchmakingQueue::with_queue(0);

        queue
            .enqueue(make_profile(1, 1200.0), make_prefs("dm"))
            .unwrap();
        queue
            .enqueue(make_profile(2, 1250.0), make_prefs("dm"))
            .unwrap();

        let matches_formed = queue.update();
        assert_eq!(matches_formed, 1);
    }

    #[test]
    fn test_queue_incompatible_game_mode() {
        let mut queue = MatchmakingQueue::with_elo(0);

        queue
            .enqueue(make_profile(1, 1200.0), make_prefs("dm"))
            .unwrap();
        queue
            .enqueue(make_profile(2, 1200.0), make_prefs("ctf"))
            .unwrap();

        let matches_formed = queue.update();
        assert_eq!(matches_formed, 0);
    }

    #[test]
    fn test_queue_too_few_players() {
        let mut queue = MatchmakingQueue::with_elo(0);

        queue
            .enqueue(make_profile(1, 1200.0), make_prefs("dm"))
            .unwrap();

        let matches_formed = queue.update();
        assert_eq!(matches_formed, 0);
    }

    #[test]
    fn test_queue_multiple_matches() {
        let mut queue = MatchmakingQueue::with_elo(0);

        for i in 0..6 {
            queue
                .enqueue(make_profile(i, 1200.0), make_prefs("dm"))
                .unwrap();
        }

        let matches_formed = queue.update();
        assert!(matches_formed >= 2); // 6 players, match size 2 => 3 matches possible
    }

    #[test]
    fn test_queue_with_teams() {
        let mut queue = MatchmakingQueue::with_elo(2); // 2v2

        for i in 0..4 {
            let mut prefs = make_prefs("dm");
            prefs.skill_range = 200.0; // wider tolerance for this test
            queue
                .enqueue(make_profile(i, 1200.0 + i as f64 * 50.0), prefs)
                .unwrap();
        }

        let matches_formed = queue.update();
        assert_eq!(matches_formed, 1);

        let matches = queue.take_matches();
        assert_eq!(matches[0].players.len(), 4);
        assert!(!matches[0].teams.is_empty());
    }

    #[test]
    fn test_queue_stats() {
        let mut queue = MatchmakingQueue::with_elo(0);
        assert_eq!(queue.total_matches(), 0);
        assert_eq!(queue.algorithm_name(), "elo");

        queue
            .enqueue(make_profile(1, 1200.0), make_prefs("dm"))
            .unwrap();
        queue
            .enqueue(make_profile(2, 1200.0), make_prefs("dm"))
            .unwrap();

        queue.update();
        assert_eq!(queue.total_matches(), 1);
    }
}
