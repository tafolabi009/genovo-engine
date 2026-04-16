//! Game session management.
//!
//! Manages the complete lifecycle of a multiplayer game session from
//! connection through gameplay to disconnection. Handles player state
//! tracking, session migration, reconnection windows, and graceful
//! disconnect handling.
//!
//! ## Session State Machine
//!
//! ```text
//! Connecting ──[handshake complete]──> Loading
//! Loading    ──[all loaded]──>         Syncing
//! Syncing    ──[state synced]──>       Playing
//! Playing    ──[pause request]──>      Paused
//! Paused     ──[resume request]──>     Playing
//! Playing    ──[end condition]──>      Ending
//! Ending     ──[cleanup done]──>       (destroyed)
//!
//! Any state  ──[timeout]──>            Ending
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use genovo_core::{EngineError, EngineResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default tick rate (ticks per second).
pub const DEFAULT_TICK_RATE: u32 = 60;

/// Default connection timeout.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Default reconnection window.
pub const DEFAULT_RECONNECT_WINDOW: Duration = Duration::from_secs(60);

/// Default loading timeout.
pub const DEFAULT_LOADING_TIMEOUT: Duration = Duration::from_secs(120);

/// Keep-alive interval during loading screens.
pub const LOADING_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(5);

/// Maximum players per session (hard limit).
pub const MAX_SESSION_PLAYERS: usize = 64;

/// Maximum ban list size.
pub const MAX_BAN_LIST: usize = 256;

// ---------------------------------------------------------------------------
// SessionState
// ---------------------------------------------------------------------------

/// The state of a game session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionState {
    /// Players are establishing connections.
    Connecting,
    /// Players are loading assets/map data.
    Loading,
    /// Initial state synchronization (full world state transfer).
    Syncing,
    /// Gameplay is active.
    Playing,
    /// Gameplay is paused.
    Paused,
    /// The session is ending (cleanup, final stats).
    Ending,
}

impl SessionState {
    /// Returns true if gameplay is active.
    pub fn is_active(&self) -> bool {
        matches!(self, SessionState::Playing | SessionState::Paused)
    }

    /// Returns true if new players can join.
    pub fn can_join(&self) -> bool {
        matches!(
            self,
            SessionState::Connecting | SessionState::Loading | SessionState::Playing
        )
    }

    /// Returns the valid next states from this state.
    pub fn valid_transitions(&self) -> &[SessionState] {
        match self {
            SessionState::Connecting => &[SessionState::Loading, SessionState::Ending],
            SessionState::Loading => &[SessionState::Syncing, SessionState::Ending],
            SessionState::Syncing => &[SessionState::Playing, SessionState::Ending],
            SessionState::Playing => &[SessionState::Paused, SessionState::Ending],
            SessionState::Paused => &[SessionState::Playing, SessionState::Ending],
            SessionState::Ending => &[],
        }
    }

    /// Returns true if transitioning to `next` is valid.
    pub fn can_transition_to(&self, next: SessionState) -> bool {
        self.valid_transitions().contains(&next)
    }
}

// ---------------------------------------------------------------------------
// PlayerState
// ---------------------------------------------------------------------------

/// The state of a player within a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlayerState {
    /// Player is establishing connection.
    Connecting,
    /// Player connection established.
    Connected,
    /// Player is loading assets.
    Loading,
    /// Player has finished loading and is ready.
    Ready,
    /// Player is actively playing.
    Playing,
    /// Player has disconnected but may reconnect.
    Disconnected,
    /// Player has been kicked or banned.
    Kicked,
}

impl PlayerState {
    /// Returns true if the player is considered active.
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            PlayerState::Connecting
                | PlayerState::Connected
                | PlayerState::Loading
                | PlayerState::Ready
                | PlayerState::Playing
        )
    }

    /// Returns true if the player can be interacted with.
    pub fn is_present(&self) -> bool {
        !matches!(self, PlayerState::Disconnected | PlayerState::Kicked)
    }
}

// ---------------------------------------------------------------------------
// SessionConfig
// ---------------------------------------------------------------------------

/// Configuration for a game session.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Maximum number of players.
    pub max_players: u32,
    /// Simulation tick rate.
    pub tick_rate: u32,
    /// Connection timeout.
    pub timeout: Duration,
    /// Map name.
    pub map: String,
    /// Game mode.
    pub game_mode: String,
    /// Whether to allow late joining.
    pub allow_late_join: bool,
    /// Whether to allow reconnection.
    pub allow_reconnect: bool,
    /// Reconnection window duration.
    pub reconnect_window: Duration,
    /// Loading screen timeout.
    pub loading_timeout: Duration,
    /// Whether to pause when a player disconnects.
    pub pause_on_disconnect: bool,
    /// Custom key-value settings.
    pub custom: HashMap<String, String>,
}

impl SessionConfig {
    /// Create a new session config with defaults.
    pub fn new(map: impl Into<String>, game_mode: impl Into<String>) -> Self {
        Self {
            max_players: 16,
            tick_rate: DEFAULT_TICK_RATE,
            timeout: DEFAULT_TIMEOUT,
            map: map.into(),
            game_mode: game_mode.into(),
            allow_late_join: true,
            allow_reconnect: true,
            reconnect_window: DEFAULT_RECONNECT_WINDOW,
            loading_timeout: DEFAULT_LOADING_TIMEOUT,
            pause_on_disconnect: false,
            custom: HashMap::new(),
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> EngineResult<()> {
        if self.max_players == 0 || self.max_players as usize > MAX_SESSION_PLAYERS {
            return Err(EngineError::InvalidArgument(format!(
                "max_players must be 1..{}",
                MAX_SESSION_PLAYERS
            )));
        }
        if self.tick_rate == 0 || self.tick_rate > 240 {
            return Err(EngineError::InvalidArgument(
                "tick_rate must be 1..240".into(),
            ));
        }
        if self.map.is_empty() {
            return Err(EngineError::InvalidArgument("map cannot be empty".into()));
        }
        Ok(())
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self::new("default_map", "default")
    }
}

// ---------------------------------------------------------------------------
// PlayerSession
// ---------------------------------------------------------------------------

/// Per-player session data.
#[derive(Debug)]
pub struct PlayerSession {
    /// Player ID.
    pub player_id: u32,
    /// Display name.
    pub name: String,
    /// Current player state.
    pub state: PlayerState,
    /// When the player joined the session.
    pub joined_at: Instant,
    /// When the last data was received from this player.
    pub last_activity: Instant,
    /// When the player disconnected (if applicable).
    pub disconnected_at: Option<Instant>,
    /// Number of reconnection attempts.
    pub reconnect_attempts: u32,
    /// Loading progress (0.0 - 1.0).
    pub loading_progress: f32,
    /// Current ping in milliseconds.
    pub ping_ms: u32,
    /// Total data sent to this player (bytes).
    pub bytes_sent: u64,
    /// Total data received from this player (bytes).
    pub bytes_received: u64,
    /// Custom player data.
    pub data: HashMap<String, String>,
}

impl PlayerSession {
    /// Create a new player session.
    pub fn new(player_id: u32, name: impl Into<String>) -> Self {
        let now = Instant::now();
        Self {
            player_id,
            name: name.into(),
            state: PlayerState::Connecting,
            joined_at: now,
            last_activity: now,
            disconnected_at: None,
            reconnect_attempts: 0,
            loading_progress: 0.0,
            ping_ms: 0,
            bytes_sent: 0,
            bytes_received: 0,
            data: HashMap::new(),
        }
    }

    /// Returns how long this player has been in the session.
    pub fn time_in_session(&self) -> Duration {
        Instant::now().duration_since(self.joined_at)
    }

    /// Returns how long since the last activity.
    pub fn time_since_activity(&self) -> Duration {
        Instant::now().duration_since(self.last_activity)
    }

    /// Returns how long the player has been disconnected.
    pub fn time_disconnected(&self) -> Option<Duration> {
        self.disconnected_at
            .map(|t| Instant::now().duration_since(t))
    }

    /// Record activity (data received).
    pub fn on_activity(&mut self) {
        self.last_activity = Instant::now();
    }

    /// Transition to a new state.
    pub fn set_state(&mut self, state: PlayerState) {
        self.state = state;
        if state == PlayerState::Disconnected {
            self.disconnected_at = Some(Instant::now());
        } else {
            self.disconnected_at = None;
        }
    }

    /// Returns true if this player has timed out.
    pub fn is_timed_out(&self, timeout: Duration) -> bool {
        self.time_since_activity() > timeout
    }

    /// Check if the reconnection window has expired.
    pub fn reconnect_expired(&self, window: Duration) -> bool {
        if let Some(disc_time) = self.disconnected_at {
            Instant::now().duration_since(disc_time) > window
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// SessionEvent
// ---------------------------------------------------------------------------

/// Events emitted by the session system.
#[derive(Debug, Clone)]
pub enum SessionEvent {
    /// Session state changed.
    StateChanged {
        old_state: SessionState,
        new_state: SessionState,
    },
    /// A player joined the session.
    PlayerJoined { player_id: u32, name: String },
    /// A player left the session.
    PlayerLeft { player_id: u32, reason: LeaveReason },
    /// A player changed state.
    PlayerStateChanged {
        player_id: u32,
        old_state: PlayerState,
        new_state: PlayerState,
    },
    /// A player reconnected.
    PlayerReconnected { player_id: u32 },
    /// A player was kicked.
    PlayerKicked { player_id: u32, reason: String },
    /// A player was banned.
    PlayerBanned { player_id: u32, reason: String },
    /// All players have finished loading.
    AllPlayersLoaded,
    /// All players are ready for gameplay.
    AllPlayersReady,
    /// Session is about to end.
    SessionEnding { reason: String },
    /// Keep-alive sent during loading.
    KeepAliveSent { player_id: u32 },
}

/// Reason a player left the session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeaveReason {
    /// Graceful disconnect.
    Voluntary,
    /// Connection timed out.
    Timeout,
    /// Kicked by host/admin.
    Kicked,
    /// Banned.
    Banned,
    /// Session ended.
    SessionEnded,
}

// ---------------------------------------------------------------------------
// GameSession
// ---------------------------------------------------------------------------

/// Manages a complete multiplayer game session.
pub struct GameSession {
    /// Current session state.
    state: SessionState,
    /// Session configuration.
    config: SessionConfig,
    /// Player sessions, keyed by player ID.
    players: HashMap<u32, PlayerSession>,
    /// Host player ID.
    host_id: Option<u32>,
    /// Banned player IDs.
    ban_list: Vec<u32>,
    /// When the session was created.
    created_at: Instant,
    /// When the session state last changed.
    state_changed_at: Instant,
    /// Current simulation tick.
    current_tick: u64,
    /// Event queue.
    events: Vec<SessionEvent>,
    /// When the last keep-alive was sent (during loading).
    last_keepalive: Instant,
    /// Whether the session has been explicitly started.
    started: bool,
}

impl GameSession {
    /// Create a new game session.
    pub fn new(config: SessionConfig) -> EngineResult<Self> {
        config.validate()?;

        let now = Instant::now();
        Ok(Self {
            state: SessionState::Connecting,
            config,
            players: HashMap::new(),
            host_id: None,
            ban_list: Vec::new(),
            created_at: now,
            state_changed_at: now,
            current_tick: 0,
            events: Vec::new(),
            last_keepalive: now,
            started: false,
        })
    }

    // -- Getters --

    /// Returns the current session state.
    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Returns the session configuration.
    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    /// Returns the current tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Returns the host player ID.
    pub fn host_id(&self) -> Option<u32> {
        self.host_id
    }

    /// Returns the number of connected players.
    pub fn player_count(&self) -> usize {
        self.players
            .values()
            .filter(|p| p.state.is_active())
            .count()
    }

    /// Returns the total number of players (including disconnected).
    pub fn total_player_count(&self) -> usize {
        self.players.len()
    }

    /// Returns a player session by ID.
    pub fn get_player(&self, player_id: u32) -> Option<&PlayerSession> {
        self.players.get(&player_id)
    }

    /// Returns a mutable reference to a player session.
    pub fn get_player_mut(&mut self, player_id: u32) -> Option<&mut PlayerSession> {
        self.players.get_mut(&player_id)
    }

    /// Returns all active players.
    pub fn active_players(&self) -> impl Iterator<Item = &PlayerSession> {
        self.players.values().filter(|p| p.state.is_active())
    }

    /// Returns all players regardless of state.
    pub fn all_players(&self) -> impl Iterator<Item = &PlayerSession> {
        self.players.values()
    }

    /// Returns the time since the session was created.
    pub fn session_age(&self) -> Duration {
        Instant::now().duration_since(self.created_at)
    }

    /// Drain pending events.
    pub fn drain_events(&mut self) -> Vec<SessionEvent> {
        std::mem::take(&mut self.events)
    }

    // -- State transitions --

    /// Transition to a new session state.
    fn transition_to(&mut self, new_state: SessionState) -> EngineResult<()> {
        if !self.state.can_transition_to(new_state) {
            return Err(EngineError::InvalidState(format!(
                "cannot transition from {:?} to {:?}",
                self.state, new_state
            )));
        }

        let old_state = self.state;
        self.state = new_state;
        self.state_changed_at = Instant::now();

        self.events.push(SessionEvent::StateChanged {
            old_state,
            new_state,
        });

        log::info!("Session state: {:?} -> {:?}", old_state, new_state);
        Ok(())
    }

    /// Start the session (move to Connecting -> Loading).
    pub fn start_session(&mut self) -> EngineResult<()> {
        if self.started {
            return Err(EngineError::InvalidState(
                "session already started".into(),
            ));
        }
        self.started = true;
        self.transition_to(SessionState::Loading)?;
        Ok(())
    }

    /// Signal that loading is complete and begin state sync.
    pub fn begin_sync(&mut self) -> EngineResult<()> {
        self.transition_to(SessionState::Syncing)
    }

    /// Signal that sync is complete and begin gameplay.
    pub fn begin_playing(&mut self) -> EngineResult<()> {
        // Set all ready players to Playing.
        let player_ids: Vec<u32> = self
            .players
            .values()
            .filter(|p| p.state == PlayerState::Ready)
            .map(|p| p.player_id)
            .collect();

        for pid in player_ids {
            self.set_player_state(pid, PlayerState::Playing);
        }

        self.transition_to(SessionState::Playing)
    }

    /// Pause the session.
    pub fn pause(&mut self) -> EngineResult<()> {
        self.transition_to(SessionState::Paused)
    }

    /// Resume the session.
    pub fn resume(&mut self) -> EngineResult<()> {
        self.transition_to(SessionState::Playing)
    }

    /// End the session.
    pub fn end_session(&mut self, reason: impl Into<String>) -> EngineResult<()> {
        let reason = reason.into();
        self.events.push(SessionEvent::SessionEnding {
            reason: reason.clone(),
        });
        self.transition_to(SessionState::Ending)?;

        // Mark all players as leaving.
        let player_ids: Vec<u32> = self.players.keys().copied().collect();
        for pid in player_ids {
            self.events.push(SessionEvent::PlayerLeft {
                player_id: pid,
                reason: LeaveReason::SessionEnded,
            });
        }

        Ok(())
    }

    // -- Player management --

    /// Add a player to the session.
    pub fn join_player(
        &mut self,
        player_id: u32,
        name: impl Into<String>,
    ) -> EngineResult<()> {
        if !self.state.can_join() {
            return Err(EngineError::InvalidState(
                "session is not accepting players".into(),
            ));
        }

        if !self.config.allow_late_join && self.state == SessionState::Playing {
            return Err(EngineError::InvalidState(
                "late joining is not allowed".into(),
            ));
        }

        if self.ban_list.contains(&player_id) {
            return Err(EngineError::InvalidArgument(
                "player is banned".into(),
            ));
        }

        if self.player_count() >= self.config.max_players as usize {
            return Err(EngineError::InvalidState("session is full".into()));
        }

        // Check for reconnection.
        if let Some(existing) = self.players.get_mut(&player_id) {
            if existing.state == PlayerState::Disconnected {
                if !self.config.allow_reconnect {
                    return Err(EngineError::InvalidState(
                        "reconnection is not allowed".into(),
                    ));
                }

                if existing.reconnect_expired(self.config.reconnect_window) {
                    return Err(EngineError::InvalidState(
                        "reconnection window expired".into(),
                    ));
                }

                existing.set_state(PlayerState::Connected);
                existing.on_activity();
                existing.reconnect_attempts += 1;

                self.events.push(SessionEvent::PlayerReconnected { player_id });

                log::info!("Player {} reconnected", player_id);
                return Ok(());
            }

            return Err(EngineError::InvalidArgument(
                "player already in session".into(),
            ));
        }

        let name = name.into();
        let player = PlayerSession::new(player_id, &name);
        self.players.insert(player_id, player);

        // First player becomes host.
        if self.host_id.is_none() {
            self.host_id = Some(player_id);
        }

        self.events.push(SessionEvent::PlayerJoined {
            player_id,
            name: name.clone(),
        });

        log::info!("Player {} ({}) joined session", name, player_id);
        Ok(())
    }

    /// Remove a player from the session.
    pub fn leave_player(&mut self, player_id: u32, reason: LeaveReason) -> EngineResult<()> {
        if !self.players.contains_key(&player_id) {
            return Err(EngineError::NotFound("player not in session".into()));
        }

        match reason {
            LeaveReason::Voluntary | LeaveReason::SessionEnded => {
                // Graceful leave: remove entirely.
                self.players.remove(&player_id);
            }
            LeaveReason::Timeout => {
                // Timeout: mark as disconnected (can reconnect).
                if let Some(player) = self.players.get_mut(&player_id) {
                    player.set_state(PlayerState::Disconnected);
                }
            }
            LeaveReason::Kicked => {
                self.players.remove(&player_id);
            }
            LeaveReason::Banned => {
                self.players.remove(&player_id);
                if self.ban_list.len() < MAX_BAN_LIST {
                    self.ban_list.push(player_id);
                }
            }
        }

        self.events.push(SessionEvent::PlayerLeft {
            player_id,
            reason: reason.clone(),
        });

        // Host migration.
        if self.host_id == Some(player_id) {
            self.migrate_host(player_id);
        }

        // Pause on disconnect if configured.
        if reason == LeaveReason::Timeout
            && self.config.pause_on_disconnect
            && self.state == SessionState::Playing
        {
            let _ = self.pause();
        }

        log::info!(
            "Player {} left session (reason: {:?})",
            player_id,
            reason
        );
        Ok(())
    }

    /// Kick a player from the session.
    pub fn kick_player(
        &mut self,
        player_id: u32,
        reason: impl Into<String>,
    ) -> EngineResult<()> {
        let reason_str = reason.into();
        self.events.push(SessionEvent::PlayerKicked {
            player_id,
            reason: reason_str.clone(),
        });
        self.leave_player(player_id, LeaveReason::Kicked)
    }

    /// Ban a player from the session.
    pub fn ban_player(
        &mut self,
        player_id: u32,
        reason: impl Into<String>,
    ) -> EngineResult<()> {
        let reason_str = reason.into();
        self.events.push(SessionEvent::PlayerBanned {
            player_id,
            reason: reason_str.clone(),
        });
        self.leave_player(player_id, LeaveReason::Banned)
    }

    /// Set a player's state.
    fn set_player_state(&mut self, player_id: u32, new_state: PlayerState) {
        if let Some(player) = self.players.get_mut(&player_id) {
            let old_state = player.state;
            player.set_state(new_state);
            self.events.push(SessionEvent::PlayerStateChanged {
                player_id,
                old_state,
                new_state,
            });
        }
    }

    /// Update a player's loading progress.
    pub fn set_loading_progress(&mut self, player_id: u32, progress: f32) {
        if let Some(player) = self.players.get_mut(&player_id) {
            player.loading_progress = progress.clamp(0.0, 1.0);
            if progress >= 1.0 && player.state == PlayerState::Loading {
                let old = player.state;
                player.set_state(PlayerState::Ready);
                self.events.push(SessionEvent::PlayerStateChanged {
                    player_id,
                    old_state: old,
                    new_state: PlayerState::Ready,
                });
            }
        }

        // Check if all players have loaded.
        if self.all_players_ready() {
            self.events.push(SessionEvent::AllPlayersLoaded);
        }
    }

    /// Mark a player as connected (handshake complete).
    pub fn mark_connected(&mut self, player_id: u32) {
        self.set_player_state(player_id, PlayerState::Connected);
    }

    /// Mark a player as loading.
    pub fn mark_loading(&mut self, player_id: u32) {
        self.set_player_state(player_id, PlayerState::Loading);
    }

    /// Mark a player as ready.
    pub fn mark_ready(&mut self, player_id: u32) {
        self.set_player_state(player_id, PlayerState::Ready);

        if self.all_players_ready() {
            self.events.push(SessionEvent::AllPlayersReady);
        }
    }

    // -- Queries --

    /// Returns true if all active players are ready.
    pub fn all_players_ready(&self) -> bool {
        let active: Vec<_> = self
            .players
            .values()
            .filter(|p| p.state.is_active())
            .collect();

        if active.is_empty() {
            return false;
        }

        active.iter().all(|p| p.state == PlayerState::Ready || p.state == PlayerState::Playing)
    }

    /// Returns true if all active players have finished loading.
    pub fn all_players_loaded(&self) -> bool {
        self.players
            .values()
            .filter(|p| p.state.is_active())
            .all(|p| p.loading_progress >= 1.0 || p.state == PlayerState::Ready || p.state == PlayerState::Playing)
    }

    /// Returns the overall loading progress (average).
    pub fn loading_progress(&self) -> f32 {
        let active: Vec<&PlayerSession> = self
            .players
            .values()
            .filter(|p| p.state.is_active())
            .collect();

        if active.is_empty() {
            return 0.0;
        }

        let total: f32 = active.iter().map(|p| p.loading_progress).sum();
        total / active.len() as f32
    }

    /// Returns true if a player is banned.
    pub fn is_banned(&self, player_id: u32) -> bool {
        self.ban_list.contains(&player_id)
    }

    // -- Host migration --

    /// Migrate the host role to another player.
    fn migrate_host(&mut self, old_host_id: u32) {
        // Select the active player who joined earliest.
        let new_host = self
            .players
            .values()
            .filter(|p| p.state.is_active() && p.player_id != old_host_id)
            .min_by_key(|p| p.joined_at)
            .map(|p| p.player_id);

        self.host_id = new_host;

        if let Some(new_id) = new_host {
            log::info!("Session host migrated to player {}", new_id);
        } else {
            log::warn!("Session has no remaining players for host");
        }
    }

    // -- Update --

    /// Update the session: check timeouts, send keep-alives, advance tick.
    ///
    /// Should be called once per frame/tick.
    pub fn update(&mut self) {
        let now = Instant::now();

        // Check for player timeouts.
        let timeout = self.config.timeout;
        let reconnect_window = self.config.reconnect_window;

        let timed_out: Vec<u32> = self
            .players
            .values()
            .filter(|p| {
                p.state.is_active() && p.is_timed_out(timeout)
            })
            .map(|p| p.player_id)
            .collect();

        for pid in timed_out {
            let _ = self.leave_player(pid, LeaveReason::Timeout);
        }

        // Check for expired reconnection windows.
        let expired: Vec<u32> = self
            .players
            .values()
            .filter(|p| {
                p.state == PlayerState::Disconnected
                    && p.reconnect_expired(reconnect_window)
            })
            .map(|p| p.player_id)
            .collect();

        for pid in expired {
            self.players.remove(&pid);
            log::info!(
                "Player {} removed (reconnection window expired)",
                pid
            );
        }

        // Loading screen keep-alives.
        if self.state == SessionState::Loading
            && now.duration_since(self.last_keepalive) >= LOADING_KEEPALIVE_INTERVAL
        {
            self.last_keepalive = now;
            for player in self.players.values() {
                if player.state == PlayerState::Loading {
                    self.events.push(SessionEvent::KeepAliveSent {
                        player_id: player.player_id,
                    });
                }
            }
        }

        // Check loading timeout.
        if self.state == SessionState::Loading {
            let loading_time = now.duration_since(self.state_changed_at);
            if loading_time > self.config.loading_timeout {
                log::warn!("Loading timeout exceeded, ending session");
                let _ = self.end_session("loading timeout");
            }
        }

        // Advance tick during gameplay.
        if self.state == SessionState::Playing {
            self.current_tick += 1;
        }

        // Auto-end if no active players.
        if self.state.is_active() && self.player_count() == 0 {
            let _ = self.end_session("all players left");
        }
    }

    /// Record activity for a player (keep-alive / data received).
    pub fn record_activity(&mut self, player_id: u32) {
        if let Some(player) = self.players.get_mut(&player_id) {
            player.on_activity();
        }
    }

    /// Update a player's ping.
    pub fn update_ping(&mut self, player_id: u32, ping_ms: u32) {
        if let Some(player) = self.players.get_mut(&player_id) {
            player.ping_ms = ping_ms;
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> SessionConfig {
        SessionConfig::new("test_map", "deathmatch")
    }

    // -----------------------------------------------------------------------
    // SessionConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_validation() {
        let config = default_config();
        assert!(config.validate().is_ok());

        let mut bad = default_config();
        bad.max_players = 0;
        assert!(bad.validate().is_err());

        let mut bad2 = default_config();
        bad2.tick_rate = 0;
        assert!(bad2.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // SessionState
    // -----------------------------------------------------------------------

    #[test]
    fn test_session_state_transitions() {
        assert!(SessionState::Connecting.can_transition_to(SessionState::Loading));
        assert!(!SessionState::Connecting.can_transition_to(SessionState::Playing));
        assert!(SessionState::Playing.can_transition_to(SessionState::Paused));
        assert!(SessionState::Paused.can_transition_to(SessionState::Playing));
        assert!(!SessionState::Ending.can_transition_to(SessionState::Playing));
    }

    #[test]
    fn test_session_state_can_join() {
        assert!(SessionState::Connecting.can_join());
        assert!(SessionState::Loading.can_join());
        assert!(SessionState::Playing.can_join());
        assert!(!SessionState::Ending.can_join());
    }

    // -----------------------------------------------------------------------
    // GameSession lifecycle
    // -----------------------------------------------------------------------

    #[test]
    fn test_session_creation() {
        let session = GameSession::new(default_config()).unwrap();
        assert_eq!(session.state(), SessionState::Connecting);
        assert_eq!(session.player_count(), 0);
        assert!(session.host_id().is_none());
    }

    #[test]
    fn test_session_full_lifecycle() {
        let mut session = GameSession::new(default_config()).unwrap();

        // Add players.
        session.join_player(1, "Alice").unwrap();
        session.join_player(2, "Bob").unwrap();
        assert_eq!(session.player_count(), 2);
        assert_eq!(session.host_id(), Some(1));

        // Start session.
        session.start_session().unwrap();
        assert_eq!(session.state(), SessionState::Loading);

        // Mark loading.
        session.mark_loading(1);
        session.mark_loading(2);

        // Complete loading.
        session.set_loading_progress(1, 1.0);
        session.set_loading_progress(2, 1.0);
        assert!(session.all_players_loaded());

        // Begin sync.
        session.begin_sync().unwrap();
        assert_eq!(session.state(), SessionState::Syncing);

        // Begin playing.
        session.begin_playing().unwrap();
        assert_eq!(session.state(), SessionState::Playing);

        // Pause and resume.
        session.pause().unwrap();
        assert_eq!(session.state(), SessionState::Paused);
        session.resume().unwrap();
        assert_eq!(session.state(), SessionState::Playing);

        // End session.
        session.end_session("game over").unwrap();
        assert_eq!(session.state(), SessionState::Ending);
    }

    // -----------------------------------------------------------------------
    // Player management
    // -----------------------------------------------------------------------

    #[test]
    fn test_player_join_leave() {
        let mut session = GameSession::new(default_config()).unwrap();

        session.join_player(1, "Alice").unwrap();
        assert_eq!(session.player_count(), 1);

        session
            .leave_player(1, LeaveReason::Voluntary)
            .unwrap();
        assert_eq!(session.player_count(), 0);
    }

    #[test]
    fn test_player_kick() {
        let mut session = GameSession::new(default_config()).unwrap();
        session.join_player(1, "Alice").unwrap();

        session.kick_player(1, "bad behavior").unwrap();
        assert_eq!(session.player_count(), 0);
    }

    #[test]
    fn test_player_ban() {
        let mut session = GameSession::new(default_config()).unwrap();
        session.join_player(1, "Alice").unwrap();

        session.ban_player(1, "cheating").unwrap();
        assert!(session.is_banned(1));

        // Banned player cannot rejoin.
        let result = session.join_player(1, "Alice");
        assert!(result.is_err());
    }

    #[test]
    fn test_session_full() {
        let mut config = default_config();
        config.max_players = 2;
        let mut session = GameSession::new(config).unwrap();

        session.join_player(1, "Alice").unwrap();
        session.join_player(2, "Bob").unwrap();
        assert_eq!(session.player_count(), 2);

        // Third player should be rejected because session is full.
        let result = session.join_player(3, "Charlie");
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Reconnection
    // -----------------------------------------------------------------------

    #[test]
    fn test_reconnection() {
        let mut session = GameSession::new(default_config()).unwrap();
        session.join_player(1, "Alice").unwrap();
        session.join_player(2, "Bob").unwrap();

        // Disconnect player 1.
        session
            .leave_player(1, LeaveReason::Timeout)
            .unwrap();

        // Player should still exist but be disconnected.
        let player = session.get_player(1).unwrap();
        assert_eq!(player.state, PlayerState::Disconnected);

        // Reconnect.
        session.join_player(1, "Alice").unwrap();
        let player = session.get_player(1).unwrap();
        assert_eq!(player.state, PlayerState::Connected);
        assert_eq!(player.reconnect_attempts, 1);
    }

    // -----------------------------------------------------------------------
    // Host migration
    // -----------------------------------------------------------------------

    #[test]
    fn test_host_migration() {
        let mut session = GameSession::new(default_config()).unwrap();
        session.join_player(1, "Alice").unwrap();
        session.join_player(2, "Bob").unwrap();

        assert_eq!(session.host_id(), Some(1));

        session
            .leave_player(1, LeaveReason::Voluntary)
            .unwrap();
        assert_eq!(session.host_id(), Some(2));
    }

    // -----------------------------------------------------------------------
    // Loading progress
    // -----------------------------------------------------------------------

    #[test]
    fn test_loading_progress() {
        let mut session = GameSession::new(default_config()).unwrap();
        session.join_player(1, "Alice").unwrap();
        session.join_player(2, "Bob").unwrap();

        session.mark_loading(1);
        session.mark_loading(2);

        session.set_loading_progress(1, 0.5);
        session.set_loading_progress(2, 0.3);

        let progress = session.loading_progress();
        assert!((progress - 0.4).abs() < 0.01);

        session.set_loading_progress(1, 1.0);
        session.set_loading_progress(2, 1.0);

        assert!(session.all_players_loaded());
    }

    // -----------------------------------------------------------------------
    // Events
    // -----------------------------------------------------------------------

    #[test]
    fn test_session_events() {
        let mut session = GameSession::new(default_config()).unwrap();
        session.join_player(1, "Alice").unwrap();

        let events = session.drain_events();
        assert!(!events.is_empty());

        let has_join = events
            .iter()
            .any(|e| matches!(e, SessionEvent::PlayerJoined { .. }));
        assert!(has_join);
    }

    #[test]
    fn test_state_change_event() {
        let mut session = GameSession::new(default_config()).unwrap();
        session.join_player(1, "Alice").unwrap();
        session.start_session().unwrap();

        let events = session.drain_events();
        let has_state_change = events.iter().any(|e| {
            matches!(
                e,
                SessionEvent::StateChanged {
                    new_state: SessionState::Loading,
                    ..
                }
            )
        });
        assert!(has_state_change);
    }

    // -----------------------------------------------------------------------
    // PlayerSession
    // -----------------------------------------------------------------------

    #[test]
    fn test_player_session() {
        let player = PlayerSession::new(1, "Alice");
        assert_eq!(player.state, PlayerState::Connecting);
        assert_eq!(player.loading_progress, 0.0);
        assert_eq!(player.reconnect_attempts, 0);
        assert!(player.state.is_active());
    }

    #[test]
    fn test_player_state_active() {
        assert!(PlayerState::Connected.is_active());
        assert!(PlayerState::Loading.is_active());
        assert!(PlayerState::Ready.is_active());
        assert!(PlayerState::Playing.is_active());
        assert!(!PlayerState::Disconnected.is_active());
        assert!(!PlayerState::Kicked.is_active());
    }

    #[test]
    fn test_player_activity() {
        let mut player = PlayerSession::new(1, "Alice");
        std::thread::sleep(Duration::from_millis(10));
        assert!(player.time_since_activity().as_millis() >= 10);

        player.on_activity();
        assert!(player.time_since_activity().as_millis() < 10);
    }

    // -----------------------------------------------------------------------
    // Update loop
    // -----------------------------------------------------------------------

    #[test]
    fn test_session_tick_advance() {
        let mut session = GameSession::new(default_config()).unwrap();
        session.join_player(1, "Alice").unwrap();
        session.start_session().unwrap();
        session.mark_connected(1);
        session.mark_loading(1);
        session.set_loading_progress(1, 1.0);
        session.begin_sync().unwrap();
        session.begin_playing().unwrap();

        assert_eq!(session.current_tick(), 0);
        session.update();
        assert_eq!(session.current_tick(), 1);
        session.update();
        assert_eq!(session.current_tick(), 2);
    }

    #[test]
    fn test_update_ping() {
        let mut session = GameSession::new(default_config()).unwrap();
        session.join_player(1, "Alice").unwrap();
        session.update_ping(1, 42);

        let player = session.get_player(1).unwrap();
        assert_eq!(player.ping_ms, 42);
    }

    #[test]
    fn test_late_join_disabled() {
        let mut config = default_config();
        config.allow_late_join = false;

        let mut session = GameSession::new(config).unwrap();
        session.join_player(1, "Alice").unwrap();
        session.start_session().unwrap();
        session.mark_connected(1);
        session.mark_ready(1);
        session.begin_sync().unwrap();
        session.begin_playing().unwrap();

        let result = session.join_player(2, "Bob");
        assert!(result.is_err());
    }

    #[test]
    fn test_double_start() {
        let mut session = GameSession::new(default_config()).unwrap();
        session.join_player(1, "Alice").unwrap();
        session.start_session().unwrap();
        let result = session.start_session();
        assert!(result.is_err());
    }
}
