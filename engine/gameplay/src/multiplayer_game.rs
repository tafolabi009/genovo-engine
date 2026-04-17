// engine/gameplay/src/multiplayer_game.rs
//
// Multiplayer game framework for the Genovo engine.
//
// Provides a complete multiplayer game mode system with lobby management,
// team setup, round-based gameplay, scoreboard, kill feed, match timer,
// and game mode rules.

use std::collections::HashMap;

// Constants
pub const MAX_PLAYERS: usize = 64;
pub const MAX_TEAMS: usize = 8;
pub const MAX_KILL_FEED_ENTRIES: usize = 8;
pub const DEFAULT_ROUND_TIME: f32 = 300.0;
pub const DEFAULT_WARMUP_TIME: f32 = 15.0;
pub const DEFAULT_HALFTIME_DURATION: f32 = 10.0;
pub const DEFAULT_OVERTIME_DURATION: f32 = 120.0;
pub const SCOREBOARD_SORT_INTERVAL: f32 = 1.0;

// ---------------------------------------------------------------------------
// Game mode type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GameModeType {
    FreeForAll,
    TeamDeathmatch,
    CaptureTheFlag,
    Domination,
    SearchAndDestroy,
    KingOfTheHill,
    BattleRoyale,
    Escort,
    Custom(u32),
}

impl GameModeType {
    pub fn is_team_based(&self) -> bool {
        !matches!(self, Self::FreeForAll | Self::BattleRoyale)
    }

    pub fn default_team_count(&self) -> usize {
        match self {
            Self::FreeForAll | Self::BattleRoyale => 0,
            Self::CaptureTheFlag | Self::TeamDeathmatch | Self::SearchAndDestroy | Self::Escort => 2,
            Self::Domination | Self::KingOfTheHill => 2,
            Self::Custom(_) => 2,
        }
    }

    pub fn has_rounds(&self) -> bool {
        matches!(self, Self::SearchAndDestroy | Self::CaptureTheFlag)
    }
}

// ---------------------------------------------------------------------------
// Match state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchState {
    WaitingForPlayers,
    Warmup,
    InProgress,
    Halftime,
    Overtime,
    RoundEnd,
    MatchEnd,
    PostMatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundState {
    Preparing,
    Active,
    Ending,
}

// ---------------------------------------------------------------------------
// Team
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Team {
    pub id: u32,
    pub name: String,
    pub color: [f32; 4],
    pub score: i32,
    pub round_wins: u32,
    pub players: Vec<u64>,
    pub spawn_points: Vec<[f32; 3]>,
    pub max_players: usize,
}

impl Team {
    pub fn new(id: u32, name: &str, color: [f32; 4]) -> Self {
        Self { id, name: name.to_string(), color, score: 0, round_wins: 0, players: Vec::new(), spawn_points: Vec::new(), max_players: MAX_PLAYERS / 2 }
    }

    pub fn player_count(&self) -> usize { self.players.len() }
    pub fn is_full(&self) -> bool { self.players.len() >= self.max_players }
    pub fn add_player(&mut self, id: u64) -> bool { if self.is_full() { false } else { self.players.push(id); true } }
    pub fn remove_player(&mut self, id: u64) { self.players.retain(|&p| p != id); }
    pub fn get_spawn_point(&self, index: usize) -> [f32; 3] { self.spawn_points.get(index % self.spawn_points.len().max(1)).copied().unwrap_or([0.0; 3]) }
}

// ---------------------------------------------------------------------------
// Player
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MultiplayerPlayer {
    pub id: u64,
    pub name: String,
    pub team_id: Option<u32>,
    pub kills: u32,
    pub deaths: u32,
    pub assists: u32,
    pub score: i32,
    pub ping: u32,
    pub is_alive: bool,
    pub is_ready: bool,
    pub is_spectating: bool,
    pub respawn_timer: f32,
    pub streak: u32,
    pub best_streak: u32,
    pub damage_dealt: f32,
    pub damage_taken: f32,
    pub healing_done: f32,
    pub objectives_completed: u32,
    pub time_alive: f32,
    pub join_time: f32,
    pub last_killer: Option<u64>,
    pub position: [f32; 3],
}

impl MultiplayerPlayer {
    pub fn new(id: u64, name: &str) -> Self {
        Self { id, name: name.to_string(), team_id: None, kills: 0, deaths: 0, assists: 0, score: 0, ping: 0, is_alive: true, is_ready: false, is_spectating: false, respawn_timer: 0.0, streak: 0, best_streak: 0, damage_dealt: 0.0, damage_taken: 0.0, healing_done: 0.0, objectives_completed: 0, time_alive: 0.0, join_time: 0.0, last_killer: None, position: [0.0; 3] }
    }

    pub fn kd_ratio(&self) -> f32 { if self.deaths == 0 { self.kills as f32 } else { self.kills as f32 / self.deaths as f32 } }
    pub fn register_kill(&mut self, points: i32) { self.kills += 1; self.streak += 1; self.best_streak = self.best_streak.max(self.streak); self.score += points; }
    pub fn register_death(&mut self, killer: Option<u64>) { self.deaths += 1; self.streak = 0; self.is_alive = false; self.last_killer = killer; }
    pub fn register_assist(&mut self, points: i32) { self.assists += 1; self.score += points; }
    pub fn respawn(&mut self) { self.is_alive = true; self.respawn_timer = 0.0; self.time_alive = 0.0; }
}

// ---------------------------------------------------------------------------
// Kill feed
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct KillFeedEntry {
    pub killer_name: String,
    pub killer_team: Option<u32>,
    pub victim_name: String,
    pub victim_team: Option<u32>,
    pub weapon: String,
    pub is_headshot: bool,
    pub is_revenge: bool,
    pub is_first_blood: bool,
    pub streak_count: u32,
    pub timestamp: f32,
    pub display_timer: f32,
}

#[derive(Debug, Clone)]
pub struct KillFeed {
    pub entries: Vec<KillFeedEntry>,
    pub display_duration: f32,
    pub first_blood_awarded: bool,
}

impl KillFeed {
    pub fn new() -> Self { Self { entries: Vec::new(), display_duration: 5.0, first_blood_awarded: false } }

    pub fn add_kill(&mut self, killer: &str, killer_team: Option<u32>, victim: &str, victim_team: Option<u32>, weapon: &str, headshot: bool, streak: u32, is_revenge: bool) {
        let first_blood = !self.first_blood_awarded;
        if first_blood { self.first_blood_awarded = true; }
        self.entries.push(KillFeedEntry {
            killer_name: killer.to_string(), killer_team, victim_name: victim.to_string(), victim_team, weapon: weapon.to_string(), is_headshot: headshot, is_revenge, is_first_blood: first_blood, streak_count: streak, timestamp: 0.0, display_timer: self.display_duration,
        });
        while self.entries.len() > MAX_KILL_FEED_ENTRIES { self.entries.remove(0); }
    }

    pub fn update(&mut self, dt: f32) {
        for entry in &mut self.entries { entry.display_timer -= dt; }
        self.entries.retain(|e| e.display_timer > 0.0);
    }
}

// ---------------------------------------------------------------------------
// Scoreboard
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ScoreboardEntry {
    pub player_id: u64,
    pub name: String,
    pub team_id: Option<u32>,
    pub kills: u32,
    pub deaths: u32,
    pub assists: u32,
    pub score: i32,
    pub ping: u32,
    pub kd_ratio: f32,
    pub is_alive: bool,
}

#[derive(Debug, Clone)]
pub struct Scoreboard {
    pub entries: Vec<ScoreboardEntry>,
    pub sort_by: ScoreboardSort,
    pub sort_timer: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreboardSort { Score, Kills, KdRatio, Name }

impl Scoreboard {
    pub fn new() -> Self { Self { entries: Vec::new(), sort_by: ScoreboardSort::Score, sort_timer: 0.0 } }

    pub fn rebuild(&mut self, players: &HashMap<u64, MultiplayerPlayer>) {
        self.entries.clear();
        for p in players.values() {
            self.entries.push(ScoreboardEntry {
                player_id: p.id, name: p.name.clone(), team_id: p.team_id, kills: p.kills, deaths: p.deaths, assists: p.assists, score: p.score, ping: p.ping, kd_ratio: p.kd_ratio(), is_alive: p.is_alive,
            });
        }
        self.sort();
    }

    pub fn sort(&mut self) {
        match self.sort_by {
            ScoreboardSort::Score => self.entries.sort_by(|a, b| b.score.cmp(&a.score)),
            ScoreboardSort::Kills => self.entries.sort_by(|a, b| b.kills.cmp(&a.kills)),
            ScoreboardSort::KdRatio => self.entries.sort_by(|a, b| b.kd_ratio.partial_cmp(&a.kd_ratio).unwrap_or(std::cmp::Ordering::Equal)),
            ScoreboardSort::Name => self.entries.sort_by(|a, b| a.name.cmp(&b.name)),
        }
    }

    pub fn update(&mut self, dt: f32, players: &HashMap<u64, MultiplayerPlayer>) {
        self.sort_timer += dt;
        if self.sort_timer >= SCOREBOARD_SORT_INTERVAL {
            self.sort_timer = 0.0;
            self.rebuild(players);
        }
    }
}

// ---------------------------------------------------------------------------
// Match config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MatchConfig {
    pub game_mode: GameModeType,
    pub map_name: String,
    pub max_players: usize,
    pub round_time: f32,
    pub warmup_time: f32,
    pub halftime_duration: f32,
    pub overtime_duration: f32,
    pub score_limit: i32,
    pub round_limit: u32,
    pub respawn_delay: f32,
    pub friendly_fire: bool,
    pub auto_balance: bool,
    pub allow_spectators: bool,
    pub min_players_to_start: usize,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self { game_mode: GameModeType::TeamDeathmatch, map_name: String::from("default"), max_players: 16, round_time: DEFAULT_ROUND_TIME, warmup_time: DEFAULT_WARMUP_TIME, halftime_duration: DEFAULT_HALFTIME_DURATION, overtime_duration: DEFAULT_OVERTIME_DURATION, score_limit: 75, round_limit: 0, respawn_delay: 5.0, friendly_fire: false, auto_balance: true, allow_spectators: true, min_players_to_start: 2 }
    }
}

// ---------------------------------------------------------------------------
// Match timer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MatchTimer {
    pub total_time: f32,
    pub remaining_time: f32,
    pub is_paused: bool,
    pub overtime: bool,
    pub overtime_remaining: f32,
}

impl MatchTimer {
    pub fn new(total: f32) -> Self { Self { total_time: total, remaining_time: total, is_paused: false, overtime: false, overtime_remaining: 0.0 } }
    pub fn update(&mut self, dt: f32) { if self.is_paused { return; } if self.overtime { self.overtime_remaining -= dt; } else { self.remaining_time -= dt; } }
    pub fn is_expired(&self) -> bool { if self.overtime { self.overtime_remaining <= 0.0 } else { self.remaining_time <= 0.0 } }
    pub fn start_overtime(&mut self, duration: f32) { self.overtime = true; self.overtime_remaining = duration; }
    pub fn formatted_time(&self) -> String { let t = if self.overtime { self.overtime_remaining } else { self.remaining_time }; let mins = (t / 60.0).floor() as u32; let secs = (t % 60.0).floor() as u32; format!("{:02}:{:02}", mins, secs) }
    pub fn fraction_remaining(&self) -> f32 { self.remaining_time / self.total_time }
    pub fn reset(&mut self) { self.remaining_time = self.total_time; self.overtime = false; }
}

// ---------------------------------------------------------------------------
// Match events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum MatchEvent {
    PlayerJoined { player_id: u64, name: String },
    PlayerLeft { player_id: u64 },
    PlayerKill { killer: u64, victim: u64, weapon: String, headshot: bool },
    PlayerDeath { player_id: u64 },
    PlayerRespawn { player_id: u64, position: [f32; 3] },
    ObjectiveCaptured { team_id: u32, objective_name: String },
    ObjectiveLost { team_id: u32, objective_name: String },
    RoundStart { round: u32 },
    RoundEnd { winning_team: Option<u32> },
    MatchStart,
    MatchEnd { winning_team: Option<u32> },
    TeamScoreChange { team_id: u32, new_score: i32 },
    OvertimeStart,
    FirstBlood { killer: u64, victim: u64 },
    KillStreak { player_id: u64, streak: u32 },
    MultiKill { player_id: u64, count: u32 },
}

// ---------------------------------------------------------------------------
// Multiplayer game
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct MultiplayerGame {
    pub config: MatchConfig,
    pub state: MatchState,
    pub round_state: RoundState,
    pub current_round: u32,
    pub timer: MatchTimer,
    pub teams: Vec<Team>,
    pub players: HashMap<u64, MultiplayerPlayer>,
    pub kill_feed: KillFeed,
    pub scoreboard: Scoreboard,
    pub events: Vec<MatchEvent>,
    pub match_time: f32,
    pub warmup_timer: f32,
}

impl MultiplayerGame {
    pub fn new(config: MatchConfig) -> Self {
        let team_count = config.game_mode.default_team_count();
        let mut teams = Vec::new();
        let colors = [[0.2, 0.4, 1.0, 1.0], [1.0, 0.3, 0.2, 1.0], [0.2, 0.8, 0.2, 1.0], [1.0, 0.8, 0.1, 1.0]];
        let names = ["Blue", "Red", "Green", "Yellow"];
        for i in 0..team_count.min(MAX_TEAMS) {
            teams.push(Team::new(i as u32, names[i % names.len()], colors[i % colors.len()]));
        }
        let timer = MatchTimer::new(config.round_time);
        Self { config, state: MatchState::WaitingForPlayers, round_state: RoundState::Preparing, current_round: 0, timer, teams, players: HashMap::new(), kill_feed: KillFeed::new(), scoreboard: Scoreboard::new(), events: Vec::new(), match_time: 0.0, warmup_timer: 0.0 }
    }

    pub fn update(&mut self, dt: f32) {
        self.match_time += dt;
        self.kill_feed.update(dt);
        self.scoreboard.update(dt, &self.players);

        match self.state {
            MatchState::WaitingForPlayers => {
                if self.players.len() >= self.config.min_players_to_start && self.all_ready() {
                    self.state = MatchState::Warmup;
                    self.warmup_timer = self.config.warmup_time;
                }
            }
            MatchState::Warmup => {
                self.warmup_timer -= dt;
                if self.warmup_timer <= 0.0 { self.start_match(); }
            }
            MatchState::InProgress => {
                self.timer.update(dt);
                self.update_respawns(dt);
                if self.timer.is_expired() || self.check_score_limit() {
                    self.end_round();
                }
            }
            MatchState::RoundEnd => {
                if self.config.game_mode.has_rounds() && self.current_round < self.config.round_limit {
                    self.start_round();
                } else {
                    self.end_match();
                }
            }
            MatchState::Overtime => {
                self.timer.update(dt);
                if self.timer.is_expired() { self.end_match(); }
            }
            _ => {}
        }
    }

    pub fn add_player(&mut self, id: u64, name: &str) {
        let player = MultiplayerPlayer::new(id, name);
        self.players.insert(id, player);
        self.events.push(MatchEvent::PlayerJoined { player_id: id, name: name.to_string() });
        if self.config.auto_balance && self.config.game_mode.is_team_based() {
            self.auto_assign_team(id);
        }
    }

    pub fn remove_player(&mut self, id: u64) {
        if let Some(player) = self.players.remove(&id) {
            if let Some(team_id) = player.team_id {
                if let Some(team) = self.teams.iter_mut().find(|t| t.id == team_id) {
                    team.remove_player(id);
                }
            }
        }
        self.events.push(MatchEvent::PlayerLeft { player_id: id });
    }

    fn auto_assign_team(&mut self, player_id: u64) {
        let min_team = self.teams.iter().min_by_key(|t| t.player_count()).map(|t| t.id);
        if let Some(team_id) = min_team {
            self.assign_team(player_id, team_id);
        }
    }

    pub fn assign_team(&mut self, player_id: u64, team_id: u32) {
        if let Some(player) = self.players.get_mut(&player_id) {
            if let Some(old_team) = player.team_id {
                if let Some(team) = self.teams.iter_mut().find(|t| t.id == old_team) { team.remove_player(player_id); }
            }
            player.team_id = Some(team_id);
            if let Some(team) = self.teams.iter_mut().find(|t| t.id == team_id) { team.add_player(player_id); }
        }
    }

    pub fn register_kill(&mut self, killer_id: u64, victim_id: u64, weapon: &str, headshot: bool) {
        let kill_points = if headshot { 150 } else { 100 };
        let assist_points = 50;

        let killer_name = self.players.get(&killer_id).map(|p| p.name.clone()).unwrap_or_default();
        let killer_team = self.players.get(&killer_id).and_then(|p| p.team_id);
        let victim_name = self.players.get(&victim_id).map(|p| p.name.clone()).unwrap_or_default();
        let victim_team = self.players.get(&victim_id).and_then(|p| p.team_id);
        let is_revenge = self.players.get(&killer_id).and_then(|p| p.last_killer).map_or(false, |lk| lk == victim_id);

        if let Some(killer) = self.players.get_mut(&killer_id) {
            killer.register_kill(kill_points);
            if killer.streak >= 3 {
                self.events.push(MatchEvent::KillStreak { player_id: killer_id, streak: killer.streak });
            }
        }
        if let Some(victim) = self.players.get_mut(&victim_id) {
            victim.register_death(Some(killer_id));
            victim.respawn_timer = self.config.respawn_delay;
        }

        // Team score.
        if let Some(team_id) = killer_team {
            if let Some(team) = self.teams.iter_mut().find(|t| t.id == team_id) {
                team.score += 1;
                self.events.push(MatchEvent::TeamScoreChange { team_id, new_score: team.score });
            }
        }

        let streak = self.players.get(&killer_id).map(|p| p.streak).unwrap_or(0);
        self.kill_feed.add_kill(&killer_name, killer_team, &victim_name, victim_team, weapon, headshot, streak, is_revenge);
        self.events.push(MatchEvent::PlayerKill { killer: killer_id, victim: victim_id, weapon: weapon.to_string(), headshot });
    }

    fn update_respawns(&mut self, dt: f32) {
        let respawn_ids: Vec<u64> = self.players.iter()
            .filter(|(_, p)| !p.is_alive && p.respawn_timer > 0.0)
            .map(|(id, _)| *id)
            .collect();

        for id in respawn_ids {
            if let Some(player) = self.players.get_mut(&id) {
                player.respawn_timer -= dt;
                if player.respawn_timer <= 0.0 {
                    player.respawn();
                    let spawn = self.get_spawn_point(player.team_id);
                    player.position = spawn;
                    self.events.push(MatchEvent::PlayerRespawn { player_id: id, position: spawn });
                }
            }
        }
    }

    fn get_spawn_point(&self, team_id: Option<u32>) -> [f32; 3] {
        if let Some(tid) = team_id {
            self.teams.iter().find(|t| t.id == tid).map(|t| t.get_spawn_point(0)).unwrap_or([0.0; 3])
        } else { [0.0; 3] }
    }

    fn all_ready(&self) -> bool { self.players.values().all(|p| p.is_ready || p.is_spectating) }
    fn check_score_limit(&self) -> bool { self.teams.iter().any(|t| t.score >= self.config.score_limit) }

    fn start_match(&mut self) {
        self.state = MatchState::InProgress;
        self.current_round = 1;
        self.timer.reset();
        self.events.push(MatchEvent::MatchStart);
        self.events.push(MatchEvent::RoundStart { round: 1 });
    }

    fn start_round(&mut self) {
        self.current_round += 1;
        self.state = MatchState::InProgress;
        self.round_state = RoundState::Active;
        self.timer.reset();
        for player in self.players.values_mut() { player.respawn(); }
        self.events.push(MatchEvent::RoundStart { round: self.current_round });
    }

    fn end_round(&mut self) {
        self.state = MatchState::RoundEnd;
        self.round_state = RoundState::Ending;
        let winner = self.teams.iter().max_by_key(|t| t.score).map(|t| t.id);
        if let Some(wid) = winner {
            if let Some(team) = self.teams.iter_mut().find(|t| t.id == wid) { team.round_wins += 1; }
        }
        self.events.push(MatchEvent::RoundEnd { winning_team: winner });
    }

    fn end_match(&mut self) {
        self.state = MatchState::MatchEnd;
        let winner = self.teams.iter().max_by_key(|t| t.score).map(|t| t.id);
        self.events.push(MatchEvent::MatchEnd { winning_team: winner });
    }

    pub fn drain_events(&mut self) -> Vec<MatchEvent> { std::mem::take(&mut self.events) }
    pub fn player_count(&self) -> usize { self.players.len() }
    pub fn team_scores(&self) -> Vec<(String, i32)> { self.teams.iter().map(|t| (t.name.clone(), t.score)).collect() }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_creation() {
        let game = MultiplayerGame::new(MatchConfig::default());
        assert_eq!(game.state, MatchState::WaitingForPlayers);
        assert_eq!(game.teams.len(), 2);
    }

    #[test]
    fn test_add_player() {
        let mut game = MultiplayerGame::new(MatchConfig::default());
        game.add_player(1, "Player1");
        assert_eq!(game.player_count(), 1);
    }

    #[test]
    fn test_register_kill() {
        let mut game = MultiplayerGame::new(MatchConfig::default());
        game.add_player(1, "Alice");
        game.add_player(2, "Bob");
        game.state = MatchState::InProgress;
        game.register_kill(1, 2, "Rifle", false);
        assert_eq!(game.players[&1].kills, 1);
        assert_eq!(game.players[&2].deaths, 1);
    }

    #[test]
    fn test_match_timer() {
        let mut timer = MatchTimer::new(300.0);
        timer.update(100.0);
        assert!((timer.remaining_time - 200.0).abs() < 0.01);
        assert!(!timer.is_expired());
        timer.update(201.0);
        assert!(timer.is_expired());
    }

    #[test]
    fn test_kill_feed() {
        let mut feed = KillFeed::new();
        feed.add_kill("Alice", Some(0), "Bob", Some(1), "Rifle", true, 3, false);
        assert_eq!(feed.entries.len(), 1);
        assert!(feed.entries[0].is_first_blood);
        assert!(feed.entries[0].is_headshot);
    }

    #[test]
    fn test_player_kd_ratio() {
        let mut p = MultiplayerPlayer::new(1, "Test");
        p.register_kill(100);
        p.register_kill(100);
        p.register_death(None);
        assert!((p.kd_ratio() - 2.0).abs() < 0.01);
    }
}
