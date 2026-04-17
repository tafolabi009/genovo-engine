// engine/gameplay/src/game_mode.rs
//
// Game mode framework: game mode state machine, game mode rules (team setup,
// scoring, win conditions), game mode events, player spawning rules, game
// mode transitions, match lifecycle.

use std::collections::HashMap;

pub type PlayerId = u64;
pub type TeamId = u8;

// --- Game mode state ---
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatchState { WaitingForPlayers, Countdown, InProgress, Overtime, Ending, Ended }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GameModeType { FreeForAll, TeamDeathmatch, CaptureTheFlag, Domination, Elimination, Custom(u32) }

// --- Team ---
#[derive(Debug, Clone)]
pub struct Team { pub id: TeamId, pub name: String, pub color: [f32; 4], pub score: i32, pub max_players: u32, pub players: Vec<PlayerId> }

impl Team {
    pub fn new(id: TeamId, name: &str, color: [f32; 4], max_players: u32) -> Self {
        Self { id, name: name.to_string(), color, score: 0, max_players, players: Vec::new() }
    }
    pub fn is_full(&self) -> bool { self.players.len() >= self.max_players as usize }
    pub fn player_count(&self) -> usize { self.players.len() }
    pub fn add_player(&mut self, player: PlayerId) -> bool {
        if self.is_full() { return false; }
        if !self.players.contains(&player) { self.players.push(player); }
        true
    }
    pub fn remove_player(&mut self, player: PlayerId) -> bool {
        if let Some(pos) = self.players.iter().position(|&p| p == player) { self.players.remove(pos); true } else { false }
    }
}

// --- Win condition ---
#[derive(Debug, Clone)]
pub enum WinCondition {
    ScoreLimit(i32),
    TimeLimit(f32),
    Elimination,
    ScoreAndTime { score_limit: i32, time_limit: f32 },
    Custom(String),
}

// --- Spawn rule ---
#[derive(Debug, Clone)]
pub struct SpawnRule {
    pub respawn_time: f32,
    pub invincibility_time: f32,
    pub spawn_selection: SpawnSelection,
    pub spawn_loadout: Option<String>,
    pub lives: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpawnSelection { Random, FarthestFromEnemy, NearTeammate, Fixed }

impl Default for SpawnRule {
    fn default() -> Self {
        Self { respawn_time: 3.0, invincibility_time: 2.0, spawn_selection: SpawnSelection::FarthestFromEnemy, spawn_loadout: None, lives: None }
    }
}

// --- Spawn point ---
#[derive(Debug, Clone)]
pub struct SpawnPoint { pub position: [f32; 3], pub rotation: f32, pub team: Option<TeamId>, pub enabled: bool, pub priority: i32 }

impl SpawnPoint {
    pub fn new(position: [f32; 3], rotation: f32) -> Self {
        Self { position, rotation, team: None, enabled: true, priority: 0 }
    }
    pub fn for_team(mut self, team: TeamId) -> Self { self.team = Some(team); self }
}

// --- Player state ---
#[derive(Debug, Clone)]
pub struct PlayerMatchState {
    pub player_id: PlayerId,
    pub team: Option<TeamId>,
    pub score: i32,
    pub kills: u32,
    pub deaths: u32,
    pub assists: u32,
    pub lives_remaining: Option<u32>,
    pub is_alive: bool,
    pub respawn_timer: f32,
    pub is_spectating: bool,
    pub connected: bool,
    pub ping_ms: u32,
}

impl PlayerMatchState {
    pub fn new(player_id: PlayerId) -> Self {
        Self {
            player_id, team: None, score: 0, kills: 0, deaths: 0, assists: 0,
            lives_remaining: None, is_alive: false, respawn_timer: 0.0,
            is_spectating: false, connected: true, ping_ms: 0,
        }
    }
    pub fn kd_ratio(&self) -> f32 { if self.deaths > 0 { self.kills as f32 / self.deaths as f32 } else { self.kills as f32 } }
    pub fn add_kill(&mut self) { self.kills += 1; self.score += 100; }
    pub fn add_death(&mut self) {
        self.deaths += 1;
        self.is_alive = false;
        if let Some(ref mut lives) = self.lives_remaining { *lives = lives.saturating_sub(1); }
    }
    pub fn can_respawn(&self) -> bool {
        if let Some(lives) = self.lives_remaining { lives > 0 } else { true }
    }
}

// --- Game mode events ---
#[derive(Debug, Clone)]
pub enum GameModeEvent {
    MatchStateChanged { from: MatchState, to: MatchState },
    PlayerJoined { player: PlayerId },
    PlayerLeft { player: PlayerId },
    PlayerKilled { killer: PlayerId, victim: PlayerId, weapon: String },
    PlayerSpawned { player: PlayerId, position: [f32; 3] },
    TeamScoreChanged { team: TeamId, score: i32 },
    PlayerScoreChanged { player: PlayerId, score: i32 },
    WinConditionMet { winner: WinnerInfo },
    RoundStarted { round: u32 },
    RoundEnded { round: u32 },
    OvertimeStarted,
    CountdownTick { seconds_remaining: u32 },
}

#[derive(Debug, Clone)]
pub enum WinnerInfo { Team(TeamId), Player(PlayerId), Draw }

// --- Game mode config ---
#[derive(Debug, Clone)]
pub struct GameModeConfig {
    pub mode_type: GameModeType,
    pub name: String,
    pub min_players: u32,
    pub max_players: u32,
    pub win_condition: WinCondition,
    pub spawn_rule: SpawnRule,
    pub teams: Vec<Team>,
    pub countdown_duration: f32,
    pub match_end_delay: f32,
    pub friendly_fire: bool,
    pub auto_balance_teams: bool,
    pub warmup_time: f32,
    pub round_based: bool,
    pub max_rounds: u32,
}

impl Default for GameModeConfig {
    fn default() -> Self {
        Self {
            mode_type: GameModeType::FreeForAll, name: "Free For All".into(),
            min_players: 2, max_players: 16,
            win_condition: WinCondition::ScoreLimit(50),
            spawn_rule: SpawnRule::default(), teams: Vec::new(),
            countdown_duration: 5.0, match_end_delay: 10.0,
            friendly_fire: false, auto_balance_teams: true,
            warmup_time: 30.0, round_based: false, max_rounds: 1,
        }
    }
}

// --- Game mode ---
pub struct GameMode {
    pub config: GameModeConfig,
    pub state: MatchState,
    pub players: HashMap<PlayerId, PlayerMatchState>,
    pub spawn_points: Vec<SpawnPoint>,
    pub events: Vec<GameModeEvent>,
    pub elapsed_time: f32,
    pub countdown_timer: f32,
    pub end_timer: f32,
    pub current_round: u32,
}

impl GameMode {
    pub fn new(config: GameModeConfig) -> Self {
        Self {
            state: MatchState::WaitingForPlayers, config,
            players: HashMap::new(), spawn_points: Vec::new(),
            events: Vec::new(), elapsed_time: 0.0,
            countdown_timer: 0.0, end_timer: 0.0, current_round: 1,
        }
    }

    pub fn add_player(&mut self, player_id: PlayerId) -> bool {
        if self.players.len() >= self.config.max_players as usize { return false; }
        let mut ps = PlayerMatchState::new(player_id);
        if !self.config.teams.is_empty() {
            let team_id = self.find_best_team();
            ps.team = Some(team_id);
            if let Some(team) = self.config.teams.iter_mut().find(|t| t.id == team_id) { team.add_player(player_id); }
        }
        self.players.insert(player_id, ps);
        self.events.push(GameModeEvent::PlayerJoined { player: player_id });
        true
    }

    pub fn remove_player(&mut self, player_id: PlayerId) -> bool {
        if let Some(ps) = self.players.remove(&player_id) {
            if let Some(team_id) = ps.team {
                if let Some(team) = self.config.teams.iter_mut().find(|t| t.id == team_id) { team.remove_player(player_id); }
            }
            self.events.push(GameModeEvent::PlayerLeft { player: player_id });
            true
        } else { false }
    }

    pub fn on_kill(&mut self, killer: PlayerId, victim: PlayerId, weapon: &str) {
        if let Some(k) = self.players.get_mut(&killer) { k.add_kill(); }
        if let Some(v) = self.players.get_mut(&victim) { v.add_death(); v.respawn_timer = self.config.spawn_rule.respawn_time; }
        if !self.config.teams.is_empty() {
            if let Some(killer_team) = self.players.get(&killer).and_then(|p| p.team) {
                if let Some(team) = self.config.teams.iter_mut().find(|t| t.id == killer_team) { team.score += 1; }
            }
        }
        self.events.push(GameModeEvent::PlayerKilled { killer, victim, weapon: weapon.to_string() });
    }

    pub fn update(&mut self, dt: f32) {
        match self.state {
            MatchState::WaitingForPlayers => {
                if self.players.len() >= self.config.min_players as usize {
                    self.transition_to(MatchState::Countdown);
                    self.countdown_timer = self.config.countdown_duration;
                }
            }
            MatchState::Countdown => {
                self.countdown_timer -= dt;
                if self.countdown_timer <= 0.0 { self.transition_to(MatchState::InProgress); }
            }
            MatchState::InProgress => {
                self.elapsed_time += dt;
                self.update_respawn_timers(dt);
                if self.check_win_condition() { self.transition_to(MatchState::Ending); self.end_timer = self.config.match_end_delay; }
            }
            MatchState::Ending => {
                self.end_timer -= dt;
                if self.end_timer <= 0.0 { self.transition_to(MatchState::Ended); }
            }
            _ => {}
        }
    }

    fn transition_to(&mut self, new_state: MatchState) {
        let old = self.state;
        self.state = new_state;
        self.events.push(GameModeEvent::MatchStateChanged { from: old, to: new_state });
    }

    fn update_respawn_timers(&mut self, dt: f32) {
        let spawn_points = &self.spawn_points;
        for ps in self.players.values_mut() {
            if !ps.is_alive && ps.respawn_timer > 0.0 {
                ps.respawn_timer -= dt;
                if ps.respawn_timer <= 0.0 && ps.can_respawn() {
                    ps.is_alive = true;
                    ps.respawn_timer = 0.0;
                }
            }
        }
    }

    fn check_win_condition(&self) -> bool {
        match &self.config.win_condition {
            WinCondition::ScoreLimit(limit) => {
                self.players.values().any(|p| p.score >= *limit) || self.config.teams.iter().any(|t| t.score >= *limit)
            }
            WinCondition::TimeLimit(limit) => self.elapsed_time >= *limit,
            WinCondition::Elimination => {
                let alive = self.players.values().filter(|p| p.is_alive || p.can_respawn()).count();
                alive <= 1
            }
            WinCondition::ScoreAndTime { score_limit, time_limit } => {
                self.elapsed_time >= *time_limit || self.players.values().any(|p| p.score >= *score_limit)
            }
            WinCondition::Custom(_) => false,
        }
    }

    pub fn determine_winner(&self) -> WinnerInfo {
        if !self.config.teams.is_empty() {
            if let Some(winner) = self.config.teams.iter().max_by_key(|t| t.score) { return WinnerInfo::Team(winner.id); }
        }
        if let Some(winner) = self.players.values().max_by_key(|p| p.score) { return WinnerInfo::Player(winner.player_id); }
        WinnerInfo::Draw
    }

    fn find_best_team(&self) -> TeamId {
        self.config.teams.iter().min_by_key(|t| t.players.len()).map(|t| t.id).unwrap_or(0)
    }

    pub fn add_spawn_point(&mut self, sp: SpawnPoint) { self.spawn_points.push(sp); }

    pub fn drain_events(&mut self) -> Vec<GameModeEvent> { std::mem::take(&mut self.events) }

    pub fn get_scoreboard(&self) -> Vec<&PlayerMatchState> {
        let mut scores: Vec<&PlayerMatchState> = self.players.values().collect();
        scores.sort_by(|a, b| b.score.cmp(&a.score));
        scores
    }
}

// --- Preset game modes ---
pub fn tdm_config(max_players: u32, score_limit: i32) -> GameModeConfig {
    GameModeConfig {
        mode_type: GameModeType::TeamDeathmatch, name: "Team Deathmatch".into(),
        max_players,
        win_condition: WinCondition::ScoreLimit(score_limit),
        teams: vec![
            Team::new(0, "Blue Team", [0.2, 0.4, 1.0, 1.0], max_players / 2),
            Team::new(1, "Red Team", [1.0, 0.2, 0.2, 1.0], max_players / 2),
        ],
        ..Default::default()
    }
}

pub fn ffa_config(max_players: u32, score_limit: i32) -> GameModeConfig {
    GameModeConfig {
        mode_type: GameModeType::FreeForAll, name: "Free For All".into(),
        max_players,
        win_condition: WinCondition::ScoreLimit(score_limit),
        ..Default::default()
    }
}

pub fn elimination_config(max_players: u32) -> GameModeConfig {
    let mut cfg = GameModeConfig {
        mode_type: GameModeType::Elimination, name: "Elimination".into(),
        max_players,
        win_condition: WinCondition::Elimination,
        spawn_rule: SpawnRule { lives: Some(1), respawn_time: 0.0, ..Default::default() },
        round_based: true, max_rounds: 5,
        ..Default::default()
    };
    cfg.teams = vec![
        Team::new(0, "Attackers", [1.0, 0.8, 0.2, 1.0], max_players / 2),
        Team::new(1, "Defenders", [0.2, 0.8, 1.0, 1.0], max_players / 2),
    ];
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffa_game_mode() {
        let config = ffa_config(8, 10);
        let mut mode = GameMode::new(config);
        mode.add_player(1);
        mode.add_player(2);
        mode.update(0.016);
        assert_eq!(mode.state, MatchState::Countdown);
    }

    #[test]
    fn test_kill_scoring() {
        let mut mode = GameMode::new(ffa_config(4, 100));
        mode.add_player(1);
        mode.add_player(2);
        mode.on_kill(1, 2, "rifle");
        assert_eq!(mode.players.get(&1).unwrap().kills, 1);
        assert_eq!(mode.players.get(&2).unwrap().deaths, 1);
    }

    #[test]
    fn test_tdm_team_assignment() {
        let mut mode = GameMode::new(tdm_config(8, 50));
        mode.add_player(1);
        mode.add_player(2);
        let teams: Vec<TeamId> = mode.players.values().filter_map(|p| p.team).collect();
        assert_eq!(teams.len(), 2);
    }

    #[test]
    fn test_win_condition() {
        let mut mode = GameMode::new(ffa_config(4, 1));
        mode.add_player(1);
        mode.add_player(2);
        mode.transition_to(MatchState::InProgress);
        mode.on_kill(1, 2, "pistol");
        mode.update(0.016);
        assert_eq!(mode.state, MatchState::Ending);
    }
}
