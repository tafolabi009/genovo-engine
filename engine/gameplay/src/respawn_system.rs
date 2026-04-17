// engine/gameplay/src/respawn_system.rs
//
// Death and respawn system for the Genovo engine.
//
// Manages player death, respawn logic, and associated penalties:
//
// - **Respawn points** -- Named locations where players can respawn.
// - **Respawn timer** -- Configurable delay before respawn.
// - **Respawn invulnerability** -- Temporary invulnerability after respawn.
// - **Death penalty** -- XP loss, item drops, currency loss on death.
// - **Spectator mode** -- Camera behavior during respawn wait.
// - **Respawn wave timing** -- Team-based wave respawn for multiplayer.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlayerId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RespawnPointId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TeamId(pub u32);

// ---------------------------------------------------------------------------
// Respawn point
// ---------------------------------------------------------------------------

/// A respawn point in the level.
#[derive(Debug, Clone)]
pub struct RespawnPoint {
    pub id: RespawnPointId,
    pub name: String,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub team: Option<TeamId>,
    pub active: bool,
    pub priority: i32,
    pub max_concurrent: u32,
    pub current_users: u32,
    pub safe_radius: f32,
    pub checkpoint: bool,
}

impl RespawnPoint {
    pub fn new(id: RespawnPointId, name: &str, position: [f32; 3]) -> Self {
        Self {
            id,
            name: name.to_string(),
            position,
            rotation: [0.0, 0.0, 0.0, 1.0],
            team: None,
            active: true,
            priority: 0,
            max_concurrent: 4,
            current_users: 0,
            safe_radius: 5.0,
            checkpoint: false,
        }
    }

    pub fn is_available(&self) -> bool {
        self.active && (self.max_concurrent == 0 || self.current_users < self.max_concurrent)
    }
}

// ---------------------------------------------------------------------------
// Death penalty
// ---------------------------------------------------------------------------

/// Configuration for death penalties.
#[derive(Debug, Clone)]
pub struct DeathPenalty {
    pub xp_loss_percent: f32,
    pub xp_loss_flat: u32,
    pub currency_loss_percent: f32,
    pub currency_loss_flat: u32,
    pub drop_equipped_items: bool,
    pub drop_inventory_items: bool,
    pub drop_item_chance: f32,
    pub max_items_dropped: u32,
    pub durability_loss_percent: f32,
    pub score_loss: u32,
    pub streak_reset: bool,
}

impl Default for DeathPenalty {
    fn default() -> Self {
        Self {
            xp_loss_percent: 0.0,
            xp_loss_flat: 0,
            currency_loss_percent: 0.0,
            currency_loss_flat: 0,
            drop_equipped_items: false,
            drop_inventory_items: false,
            drop_item_chance: 0.0,
            max_items_dropped: 0,
            durability_loss_percent: 10.0,
            score_loss: 0,
            streak_reset: true,
        }
    }
}

impl DeathPenalty {
    pub fn none() -> Self {
        Self {
            durability_loss_percent: 0.0,
            streak_reset: false,
            ..Default::default()
        }
    }

    pub fn hardcore() -> Self {
        Self {
            xp_loss_percent: 10.0,
            currency_loss_percent: 25.0,
            drop_inventory_items: true,
            drop_item_chance: 0.5,
            max_items_dropped: 3,
            durability_loss_percent: 20.0,
            streak_reset: true,
            ..Default::default()
        }
    }

    pub fn compute_xp_loss(&self, current_xp: u32) -> u32 {
        let percent_loss = (current_xp as f32 * self.xp_loss_percent / 100.0) as u32;
        percent_loss + self.xp_loss_flat
    }

    pub fn compute_currency_loss(&self, current_amount: u64) -> u64 {
        let percent_loss = (current_amount as f64 * self.currency_loss_percent as f64 / 100.0) as u64;
        percent_loss + self.currency_loss_flat as u64
    }
}

// ---------------------------------------------------------------------------
// Spectator mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectatorMode {
    /// Fixed camera at death location.
    DeathCamera,
    /// Follow killer.
    FollowKiller,
    /// Follow a teammate.
    FollowTeammate,
    /// Free-fly camera.
    FreeFly,
    /// Overhead view.
    Overhead,
    /// No spectator (black screen / respawn UI).
    None,
}

impl Default for SpectatorMode {
    fn default() -> Self {
        Self::DeathCamera
    }
}

// ---------------------------------------------------------------------------
// Player death state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerLifeState {
    Alive,
    Dying,
    Dead,
    WaitingToRespawn,
    Respawning,
    Invulnerable,
}

impl Default for PlayerLifeState {
    fn default() -> Self {
        Self::Alive
    }
}

#[derive(Debug, Clone)]
pub struct DeathInfo {
    pub death_position: [f32; 3],
    pub death_time: f64,
    pub killer_id: Option<PlayerId>,
    pub damage_type: String,
    pub death_count: u32,
}

#[derive(Debug, Clone)]
pub struct PlayerRespawnState {
    pub player_id: PlayerId,
    pub team: Option<TeamId>,
    pub life_state: PlayerLifeState,
    pub respawn_timer: f32,
    pub invulnerability_timer: f32,
    pub death_info: Option<DeathInfo>,
    pub chosen_respawn_point: Option<RespawnPointId>,
    pub spectator_mode: SpectatorMode,
    pub spectator_target: Option<PlayerId>,
    pub total_deaths: u32,
    pub total_respawns: u32,
    pub last_checkpoint: Option<RespawnPointId>,
}

impl PlayerRespawnState {
    pub fn new(player_id: PlayerId) -> Self {
        Self {
            player_id,
            team: None,
            life_state: PlayerLifeState::Alive,
            respawn_timer: 0.0,
            invulnerability_timer: 0.0,
            death_info: None,
            chosen_respawn_point: None,
            spectator_mode: SpectatorMode::default(),
            spectator_target: None,
            total_deaths: 0,
            total_respawns: 0,
            last_checkpoint: None,
        }
    }

    pub fn is_alive(&self) -> bool {
        matches!(self.life_state, PlayerLifeState::Alive | PlayerLifeState::Invulnerable)
    }

    pub fn is_dead(&self) -> bool {
        matches!(
            self.life_state,
            PlayerLifeState::Dead | PlayerLifeState::WaitingToRespawn | PlayerLifeState::Dying
        )
    }

    pub fn is_invulnerable(&self) -> bool {
        self.life_state == PlayerLifeState::Invulnerable
    }
}

// ---------------------------------------------------------------------------
// Wave respawn
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WaveRespawnConfig {
    pub wave_interval: f32,
    pub enabled: bool,
    pub per_team: bool,
}

impl Default for WaveRespawnConfig {
    fn default() -> Self {
        Self {
            wave_interval: 10.0,
            enabled: false,
            per_team: true,
        }
    }
}

#[derive(Debug, Clone)]
struct WaveTimer {
    timer: f32,
    waiting_players: Vec<PlayerId>,
}

// ---------------------------------------------------------------------------
// Respawn events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum RespawnEvent {
    PlayerDied { player: PlayerId, killer: Option<PlayerId>, position: [f32; 3] },
    RespawnTimerStarted { player: PlayerId, duration: f32 },
    PlayerRespawned { player: PlayerId, position: [f32; 3] },
    InvulnerabilityStarted { player: PlayerId, duration: f32 },
    InvulnerabilityEnded { player: PlayerId },
    WaveRespawn { team: Option<TeamId>, count: u32 },
    CheckpointReached { player: PlayerId, point: RespawnPointId },
    PenaltyApplied { player: PlayerId, xp_lost: u32, currency_lost: u64 },
}

// ---------------------------------------------------------------------------
// Respawn system
// ---------------------------------------------------------------------------

/// Configuration for the respawn system.
#[derive(Debug, Clone)]
pub struct RespawnConfig {
    pub respawn_delay: f32,
    pub invulnerability_duration: f32,
    pub death_penalty: DeathPenalty,
    pub default_spectator_mode: SpectatorMode,
    pub wave_config: WaveRespawnConfig,
    pub max_respawn_delay: f32,
    pub delay_increase_per_death: f32,
    pub auto_respawn: bool,
}

impl Default for RespawnConfig {
    fn default() -> Self {
        Self {
            respawn_delay: 3.0,
            invulnerability_duration: 3.0,
            death_penalty: DeathPenalty::default(),
            default_spectator_mode: SpectatorMode::DeathCamera,
            wave_config: WaveRespawnConfig::default(),
            max_respawn_delay: 30.0,
            delay_increase_per_death: 0.0,
            auto_respawn: true,
        }
    }
}

pub struct RespawnSystem {
    config: RespawnConfig,
    players: HashMap<PlayerId, PlayerRespawnState>,
    respawn_points: Vec<RespawnPoint>,
    next_point_id: u32,
    events: Vec<RespawnEvent>,
    wave_timers: HashMap<Option<TeamId>, WaveTimer>,
    game_time: f64,
}

impl RespawnSystem {
    pub fn new(config: RespawnConfig) -> Self {
        Self {
            config,
            players: HashMap::new(),
            respawn_points: Vec::new(),
            next_point_id: 0,
            events: Vec::new(),
            wave_timers: HashMap::new(),
            game_time: 0.0,
        }
    }

    pub fn config(&self) -> &RespawnConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: RespawnConfig) {
        self.config = config;
    }

    pub fn register_player(&mut self, player_id: PlayerId, team: Option<TeamId>) {
        let mut state = PlayerRespawnState::new(player_id);
        state.team = team;
        self.players.insert(player_id, state);
    }

    pub fn unregister_player(&mut self, player_id: PlayerId) {
        self.players.remove(&player_id);
    }

    pub fn add_respawn_point(&mut self, name: &str, position: [f32; 3]) -> RespawnPointId {
        let id = RespawnPointId(self.next_point_id);
        self.next_point_id += 1;
        self.respawn_points.push(RespawnPoint::new(id, name, position));
        id
    }

    pub fn set_checkpoint(&mut self, player_id: PlayerId, point_id: RespawnPointId) {
        if let Some(state) = self.players.get_mut(&player_id) {
            state.last_checkpoint = Some(point_id);
            self.events.push(RespawnEvent::CheckpointReached {
                player: player_id,
                point: point_id,
            });
        }
    }

    pub fn kill_player(
        &mut self,
        player_id: PlayerId,
        killer: Option<PlayerId>,
        position: [f32; 3],
        damage_type: &str,
    ) {
        if let Some(state) = self.players.get_mut(&player_id) {
            if !state.is_alive() {
                return;
            }

            state.total_deaths += 1;
            state.life_state = PlayerLifeState::Dead;
            state.death_info = Some(DeathInfo {
                death_position: position,
                death_time: self.game_time,
                killer_id: killer,
                damage_type: damage_type.to_string(),
                death_count: state.total_deaths,
            });

            let delay = (self.config.respawn_delay
                + self.config.delay_increase_per_death * state.total_deaths as f32)
                .min(self.config.max_respawn_delay);
            state.respawn_timer = delay;
            state.life_state = PlayerLifeState::WaitingToRespawn;
            state.spectator_mode = self.config.default_spectator_mode;

            self.events.push(RespawnEvent::PlayerDied {
                player: player_id,
                killer,
                position,
            });
            self.events.push(RespawnEvent::RespawnTimerStarted {
                player: player_id,
                duration: delay,
            });

            // Apply death penalty.
            let xp_loss = self.config.death_penalty.compute_xp_loss(0);
            let currency_loss = self.config.death_penalty.compute_currency_loss(0);
            if xp_loss > 0 || currency_loss > 0 {
                self.events.push(RespawnEvent::PenaltyApplied {
                    player: player_id,
                    xp_lost: xp_loss,
                    currency_lost: currency_loss,
                });
            }

            // Add to wave respawn if enabled.
            if self.config.wave_config.enabled {
                let team_key = if self.config.wave_config.per_team {
                    state.team
                } else {
                    None
                };
                self.wave_timers
                    .entry(team_key)
                    .or_insert_with(|| WaveTimer {
                        timer: self.config.wave_config.wave_interval,
                        waiting_players: Vec::new(),
                    })
                    .waiting_players
                    .push(player_id);
            }
        }
    }

    pub fn request_respawn(&mut self, player_id: PlayerId) {
        if let Some(state) = self.players.get_mut(&player_id) {
            if state.life_state == PlayerLifeState::WaitingToRespawn && state.respawn_timer <= 0.0 {
                self.perform_respawn(player_id);
            }
        }
    }

    fn perform_respawn(&mut self, player_id: PlayerId) {
        let point = self.find_best_respawn_point(player_id);
        let position = point.map(|p| p.position).unwrap_or([0.0, 0.0, 0.0]);

        if let Some(state) = self.players.get_mut(&player_id) {
            state.life_state = PlayerLifeState::Invulnerable;
            state.invulnerability_timer = self.config.invulnerability_duration;
            state.total_respawns += 1;
            state.death_info = None;

            self.events.push(RespawnEvent::PlayerRespawned {
                player: player_id,
                position,
            });
            self.events.push(RespawnEvent::InvulnerabilityStarted {
                player: player_id,
                duration: self.config.invulnerability_duration,
            });
        }
    }

    fn find_best_respawn_point(&self, player_id: PlayerId) -> Option<&RespawnPoint> {
        let state = self.players.get(&player_id)?;

        // Prefer checkpoint.
        if let Some(cp) = state.last_checkpoint {
            if let Some(point) = self.respawn_points.iter().find(|p| p.id == cp && p.is_available()) {
                return Some(point);
            }
        }

        // Prefer chosen point.
        if let Some(chosen) = state.chosen_respawn_point {
            if let Some(point) = self.respawn_points.iter().find(|p| p.id == chosen && p.is_available()) {
                return Some(point);
            }
        }

        // Find best available by priority.
        self.respawn_points
            .iter()
            .filter(|p| {
                p.is_available()
                    && (p.team.is_none() || p.team == state.team)
            })
            .max_by_key(|p| p.priority)
    }

    pub fn update(&mut self, dt: f32) {
        self.game_time += dt as f64;

        let player_ids: Vec<PlayerId> = self.players.keys().copied().collect();

        for pid in player_ids {
            if let Some(state) = self.players.get_mut(&pid) {
                match state.life_state {
                    PlayerLifeState::WaitingToRespawn => {
                        if !self.config.wave_config.enabled {
                            state.respawn_timer -= dt;
                            if state.respawn_timer <= 0.0 && self.config.auto_respawn {
                                drop(state);
                                self.perform_respawn(pid);
                            }
                        }
                    }
                    PlayerLifeState::Invulnerable => {
                        state.invulnerability_timer -= dt;
                        if state.invulnerability_timer <= 0.0 {
                            state.life_state = PlayerLifeState::Alive;
                            self.events.push(RespawnEvent::InvulnerabilityEnded { player: pid });
                        }
                    }
                    _ => {}
                }
            }
        }

        // Wave respawn.
        if self.config.wave_config.enabled {
            let wave_keys: Vec<Option<TeamId>> = self.wave_timers.keys().copied().collect();
            for key in wave_keys {
                if let Some(wave) = self.wave_timers.get_mut(&key) {
                    wave.timer -= dt;
                    if wave.timer <= 0.0 {
                        let players = std::mem::take(&mut wave.waiting_players);
                        wave.timer = self.config.wave_config.wave_interval;
                        let count = players.len() as u32;
                        for pid in players {
                            self.perform_respawn(pid);
                        }
                        if count > 0 {
                            self.events.push(RespawnEvent::WaveRespawn { team: key, count });
                        }
                    }
                }
            }
        }
    }

    pub fn player_state(&self, player_id: PlayerId) -> Option<&PlayerRespawnState> {
        self.players.get(&player_id)
    }

    pub fn drain_events(&mut self) -> Vec<RespawnEvent> {
        std::mem::take(&mut self.events)
    }

    pub fn respawn_points(&self) -> &[RespawnPoint] {
        &self.respawn_points
    }
}

impl Default for RespawnSystem {
    fn default() -> Self {
        Self::new(RespawnConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kill_and_respawn() {
        let mut sys = RespawnSystem::default();
        let p = PlayerId(0);
        sys.register_player(p, None);
        sys.add_respawn_point("spawn", [0.0, 0.0, 0.0]);

        sys.kill_player(p, None, [10.0, 0.0, 10.0], "fall");
        assert!(sys.player_state(p).unwrap().is_dead());

        // Wait for timer.
        sys.update(4.0);
        assert!(sys.player_state(p).unwrap().is_alive());
    }

    #[test]
    fn test_invulnerability() {
        let mut sys = RespawnSystem::new(RespawnConfig {
            respawn_delay: 0.0,
            invulnerability_duration: 3.0,
            auto_respawn: true,
            ..Default::default()
        });
        let p = PlayerId(0);
        sys.register_player(p, None);
        sys.add_respawn_point("spawn", [0.0, 0.0, 0.0]);

        sys.kill_player(p, None, [0.0; 3], "test");
        sys.update(0.1);
        assert!(sys.player_state(p).unwrap().is_invulnerable());

        sys.update(3.5);
        assert!(!sys.player_state(p).unwrap().is_invulnerable());
    }

    #[test]
    fn test_death_penalty() {
        let penalty = DeathPenalty {
            xp_loss_percent: 10.0,
            xp_loss_flat: 50,
            ..Default::default()
        };
        assert_eq!(penalty.compute_xp_loss(1000), 150);
    }
}
