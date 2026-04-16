//! Lobby system for pre-game player gathering.
//!
//! Provides a full lobby lifecycle: creation, joining, leaving, readying up,
//! chat, settings management, and host migration. Lobbies serve as the
//! waiting room before a match begins.
//!
//! ## State Machine
//!
//! ```text
//! Waiting ──[all ready + host starts]──> Starting
//! Starting ──[load complete]──> InGame
//! InGame ──[game ends]──> Finished
//! Finished ──[reset]──> Waiting
//!
//! Any state ──[host disconnects]──> (host migration) ──> same state
//! ```

use std::collections::HashMap;
use std::time::Instant;

use genovo_core::{EngineError, EngineResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum lobby name length.
pub const MAX_LOBBY_NAME_LEN: usize = 64;

/// Maximum chat message length.
pub const MAX_CHAT_MESSAGE_LEN: usize = 256;

/// Maximum number of teams.
pub const MAX_TEAMS: usize = 8;

/// Maximum lobbies per lobby manager.
pub const MAX_LOBBIES: usize = 1024;

/// Maximum players per lobby (hard limit).
pub const MAX_PLAYERS_PER_LOBBY: u32 = 64;

/// Maximum chat history per lobby.
pub const MAX_CHAT_HISTORY: usize = 200;

// ---------------------------------------------------------------------------
// LobbyId / PlayerId
// ---------------------------------------------------------------------------

/// Unique identifier for a lobby.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LobbyId(pub u64);

impl LobbyId {
    pub const fn new(id: u64) -> Self {
        Self(id)
    }
    pub const fn raw(&self) -> u64 {
        self.0
    }
}

/// Unique identifier for a player in the lobby system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlayerId(pub u32);

impl PlayerId {
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
    pub const fn raw(&self) -> u32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// LobbyState
// ---------------------------------------------------------------------------

/// The current state of a lobby.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LobbyState {
    /// Waiting for players to join and ready up.
    Waiting,
    /// All players are ready; the game is about to start.
    Starting,
    /// The game is in progress.
    InGame,
    /// The game has finished.
    Finished,
}

impl LobbyState {
    /// Returns true if players can join in this state.
    pub fn can_join(&self) -> bool {
        matches!(self, LobbyState::Waiting)
    }

    /// Returns true if the game can be started from this state.
    pub fn can_start(&self) -> bool {
        matches!(self, LobbyState::Waiting)
    }

    /// Returns true if settings can be changed in this state.
    pub fn can_change_settings(&self) -> bool {
        matches!(self, LobbyState::Waiting)
    }

    /// Returns the next valid state transition.
    pub fn next(&self) -> Option<LobbyState> {
        match self {
            LobbyState::Waiting => Some(LobbyState::Starting),
            LobbyState::Starting => Some(LobbyState::InGame),
            LobbyState::InGame => Some(LobbyState::Finished),
            LobbyState::Finished => Some(LobbyState::Waiting),
        }
    }
}

// ---------------------------------------------------------------------------
// LobbySettings
// ---------------------------------------------------------------------------

/// Configurable settings for a lobby.
#[derive(Debug, Clone)]
pub struct LobbySettings {
    /// The game mode (e.g., "deathmatch", "capture_flag", "coop").
    pub game_mode: String,
    /// The map name to play on.
    pub map: String,
    /// Maximum number of players allowed.
    pub max_players: u32,
    /// Optional password for private lobbies.
    pub password: Option<String>,
    /// Whether the lobby is publicly listed.
    pub is_public: bool,
    /// Number of teams (0 = free-for-all).
    pub team_count: u32,
    /// Maximum players per team (0 = no limit).
    pub max_per_team: u32,
    /// Custom key-value settings for game-specific configuration.
    pub custom: HashMap<String, String>,
}

impl LobbySettings {
    /// Creates default lobby settings.
    pub fn new(game_mode: impl Into<String>, map: impl Into<String>, max_players: u32) -> Self {
        Self {
            game_mode: game_mode.into(),
            map: map.into(),
            max_players: max_players.min(MAX_PLAYERS_PER_LOBBY),
            password: None,
            is_public: true,
            team_count: 0,
            max_per_team: 0,
            custom: HashMap::new(),
        }
    }

    /// Returns true if the lobby requires a password.
    pub fn is_password_protected(&self) -> bool {
        self.password.is_some()
    }

    /// Check if a password matches.
    pub fn check_password(&self, password: &str) -> bool {
        match &self.password {
            Some(pw) => pw == password,
            None => true, // no password required
        }
    }

    /// Validate the settings.
    pub fn validate(&self) -> EngineResult<()> {
        if self.max_players == 0 || self.max_players > MAX_PLAYERS_PER_LOBBY {
            return Err(EngineError::InvalidArgument(format!(
                "max_players must be 1..{}",
                MAX_PLAYERS_PER_LOBBY
            )));
        }
        if self.game_mode.is_empty() {
            return Err(EngineError::InvalidArgument(
                "game_mode cannot be empty".into(),
            ));
        }
        if self.map.is_empty() {
            return Err(EngineError::InvalidArgument(
                "map cannot be empty".into(),
            ));
        }
        if self.team_count as usize > MAX_TEAMS {
            return Err(EngineError::InvalidArgument(format!(
                "team_count cannot exceed {}",
                MAX_TEAMS
            )));
        }
        Ok(())
    }

    /// Set a custom setting.
    pub fn set_custom(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.custom.insert(key.into(), value.into());
    }

    /// Get a custom setting.
    pub fn get_custom(&self, key: &str) -> Option<&str> {
        self.custom.get(key).map(|s| s.as_str())
    }
}

impl Default for LobbySettings {
    fn default() -> Self {
        Self::new("default", "default_map", 8)
    }
}

// ---------------------------------------------------------------------------
// LobbyPlayer
// ---------------------------------------------------------------------------

/// A player in a lobby.
#[derive(Debug, Clone)]
pub struct LobbyPlayer {
    /// Player's unique ID.
    pub id: PlayerId,
    /// Display name.
    pub name: String,
    /// Assigned team (0 = unassigned / FFA).
    pub team: u32,
    /// Whether the player is ready to start.
    pub ready: bool,
    /// Current latency in milliseconds.
    pub ping_ms: u32,
    /// When this player joined the lobby.
    pub joined_at: Instant,
    /// Whether this player is the lobby host.
    pub is_host: bool,
    /// Custom player properties (e.g., selected character, loadout).
    pub properties: HashMap<String, String>,
}

impl LobbyPlayer {
    /// Create a new lobby player.
    pub fn new(id: PlayerId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            team: 0,
            ready: false,
            ping_ms: 0,
            joined_at: Instant::now(),
            is_host: false,
            properties: HashMap::new(),
        }
    }

    /// Create a host player.
    pub fn host(id: PlayerId, name: impl Into<String>) -> Self {
        let mut player = Self::new(id, name);
        player.is_host = true;
        player
    }

    /// Set a player property.
    pub fn set_property(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.properties.insert(key.into(), value.into());
    }

    /// Get a player property.
    pub fn get_property(&self, key: &str) -> Option<&str> {
        self.properties.get(key).map(|s| s.as_str())
    }

    /// Time since this player joined, in seconds.
    pub fn time_in_lobby_secs(&self) -> f64 {
        Instant::now().duration_since(self.joined_at).as_secs_f64()
    }
}

// ---------------------------------------------------------------------------
// ChatMessage
// ---------------------------------------------------------------------------

/// A chat message in a lobby.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Who sent the message (None = system message).
    pub sender: Option<PlayerId>,
    /// The sender's display name (cached for history).
    pub sender_name: String,
    /// The message text.
    pub text: String,
    /// When the message was sent.
    pub timestamp: Instant,
}

impl ChatMessage {
    /// Create a player chat message.
    pub fn player(sender: PlayerId, name: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            sender: Some(sender),
            sender_name: name.into(),
            text: text.into(),
            timestamp: Instant::now(),
        }
    }

    /// Create a system message.
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            sender: None,
            sender_name: "SYSTEM".to_string(),
            text: text.into(),
            timestamp: Instant::now(),
        }
    }

    /// Returns true if this is a system message.
    pub fn is_system(&self) -> bool {
        self.sender.is_none()
    }
}

// ---------------------------------------------------------------------------
// LobbyEvent
// ---------------------------------------------------------------------------

/// Events emitted by the lobby system for external handling.
#[derive(Debug, Clone)]
pub enum LobbyEvent {
    /// A new lobby was created.
    LobbyCreated { lobby_id: LobbyId },
    /// A lobby was destroyed.
    LobbyDestroyed { lobby_id: LobbyId },
    /// A player joined a lobby.
    PlayerJoined {
        lobby_id: LobbyId,
        player_id: PlayerId,
    },
    /// A player left a lobby.
    PlayerLeft {
        lobby_id: LobbyId,
        player_id: PlayerId,
    },
    /// A player changed their ready state.
    ReadyChanged {
        lobby_id: LobbyId,
        player_id: PlayerId,
        ready: bool,
    },
    /// The lobby host changed (host migration).
    HostChanged {
        lobby_id: LobbyId,
        old_host: PlayerId,
        new_host: PlayerId,
    },
    /// The lobby state changed.
    StateChanged {
        lobby_id: LobbyId,
        old_state: LobbyState,
        new_state: LobbyState,
    },
    /// Lobby settings were changed.
    SettingsChanged { lobby_id: LobbyId },
    /// A chat message was sent.
    ChatMessageSent {
        lobby_id: LobbyId,
        message: ChatMessage,
    },
    /// All players are ready.
    AllPlayersReady { lobby_id: LobbyId },
    /// A player's team was changed.
    TeamChanged {
        lobby_id: LobbyId,
        player_id: PlayerId,
        team: u32,
    },
}

// ---------------------------------------------------------------------------
// LobbyInfo
// ---------------------------------------------------------------------------

/// Summary information about a lobby (for listing).
#[derive(Debug, Clone)]
pub struct LobbyInfo {
    /// Lobby ID.
    pub id: LobbyId,
    /// Lobby name.
    pub name: String,
    /// Current state.
    pub state: LobbyState,
    /// Host player name.
    pub host_name: String,
    /// Current player count.
    pub player_count: u32,
    /// Maximum player count.
    pub max_players: u32,
    /// Game mode.
    pub game_mode: String,
    /// Map name.
    pub map: String,
    /// Whether the lobby is password protected.
    pub has_password: bool,
    /// Whether the lobby is public.
    pub is_public: bool,
    /// Average ping of players.
    pub avg_ping_ms: u32,
}

// ---------------------------------------------------------------------------
// Lobby
// ---------------------------------------------------------------------------

/// A game lobby where players gather before a match.
pub struct Lobby {
    /// Unique lobby identifier.
    pub id: LobbyId,
    /// Human-readable lobby name.
    pub name: String,
    /// Current lobby state.
    state: LobbyState,
    /// Lobby settings.
    settings: LobbySettings,
    /// Players in the lobby, keyed by player ID.
    players: HashMap<PlayerId, LobbyPlayer>,
    /// The current host player.
    host_id: Option<PlayerId>,
    /// Chat history.
    chat_history: Vec<ChatMessage>,
    /// When the lobby was created.
    created_at: Instant,
    /// Event queue for this lobby.
    events: Vec<LobbyEvent>,
}

impl Lobby {
    /// Create a new lobby.
    pub fn new(
        id: LobbyId,
        name: impl Into<String>,
        host: LobbyPlayer,
        settings: LobbySettings,
    ) -> EngineResult<Self> {
        settings.validate()?;
        let lobby_name = name.into();
        if lobby_name.is_empty() || lobby_name.len() > MAX_LOBBY_NAME_LEN {
            return Err(EngineError::InvalidArgument(format!(
                "lobby name must be 1..{} characters",
                MAX_LOBBY_NAME_LEN
            )));
        }

        let host_id = host.id;
        let mut players = HashMap::new();
        let mut host_player = host;
        host_player.is_host = true;
        players.insert(host_id, host_player);

        let mut lobby = Self {
            id,
            name: lobby_name,
            state: LobbyState::Waiting,
            settings,
            players,
            host_id: Some(host_id),
            chat_history: Vec::new(),
            created_at: Instant::now(),
            events: Vec::new(),
        };

        lobby.push_system_message("Lobby created.");
        Ok(lobby)
    }

    // -- Getters --

    /// Returns the current lobby state.
    pub fn state(&self) -> LobbyState {
        self.state
    }

    /// Returns a reference to the lobby settings.
    pub fn settings(&self) -> &LobbySettings {
        &self.settings
    }

    /// Returns the host player ID.
    pub fn host_id(&self) -> Option<PlayerId> {
        self.host_id
    }

    /// Returns the current number of players.
    pub fn player_count(&self) -> u32 {
        self.players.len() as u32
    }

    /// Returns the maximum number of players.
    pub fn max_players(&self) -> u32 {
        self.settings.max_players
    }

    /// Returns true if the lobby is full.
    pub fn is_full(&self) -> bool {
        self.player_count() >= self.settings.max_players
    }

    /// Returns true if all players are ready.
    pub fn all_ready(&self) -> bool {
        !self.players.is_empty() && self.players.values().all(|p| p.ready)
    }

    /// Returns a player by ID.
    pub fn get_player(&self, id: PlayerId) -> Option<&LobbyPlayer> {
        self.players.get(&id)
    }

    /// Returns a mutable reference to a player by ID.
    pub fn get_player_mut(&mut self, id: PlayerId) -> Option<&mut LobbyPlayer> {
        self.players.get_mut(&id)
    }

    /// Returns all players in the lobby.
    pub fn players(&self) -> impl Iterator<Item = &LobbyPlayer> {
        self.players.values()
    }

    /// Returns the chat history.
    pub fn chat_history(&self) -> &[ChatMessage] {
        &self.chat_history
    }

    /// Drain all pending events.
    pub fn drain_events(&mut self) -> Vec<LobbyEvent> {
        std::mem::take(&mut self.events)
    }

    /// Returns the time since the lobby was created.
    pub fn age_secs(&self) -> f64 {
        Instant::now().duration_since(self.created_at).as_secs_f64()
    }

    /// Returns summary info for lobby listings.
    pub fn info(&self) -> LobbyInfo {
        let host_name = self
            .host_id
            .and_then(|id| self.players.get(&id))
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "Unknown".to_string());

        let avg_ping = if self.players.is_empty() {
            0
        } else {
            let total: u32 = self.players.values().map(|p| p.ping_ms).sum();
            total / self.players.len() as u32
        };

        LobbyInfo {
            id: self.id,
            name: self.name.clone(),
            state: self.state,
            host_name,
            player_count: self.player_count(),
            max_players: self.settings.max_players,
            game_mode: self.settings.game_mode.clone(),
            map: self.settings.map.clone(),
            has_password: self.settings.is_password_protected(),
            is_public: self.settings.is_public,
            avg_ping_ms: avg_ping,
        }
    }

    // -- Player management --

    /// Add a player to the lobby.
    pub fn join(&mut self, player: LobbyPlayer, password: Option<&str>) -> EngineResult<()> {
        if !self.state.can_join() {
            return Err(EngineError::InvalidState(
                "lobby is not accepting players".into(),
            ));
        }

        if self.is_full() {
            return Err(EngineError::InvalidState("lobby is full".into()));
        }

        if self.players.contains_key(&player.id) {
            return Err(EngineError::InvalidArgument(
                "player already in lobby".into(),
            ));
        }

        // Check password.
        if self.settings.is_password_protected() {
            let pw = password.unwrap_or("");
            if !self.settings.check_password(pw) {
                return Err(EngineError::InvalidArgument("incorrect password".into()));
            }
        }

        let player_id = player.id;
        let player_name = player.name.clone();
        self.players.insert(player_id, player);

        self.push_system_message(&format!("{} joined the lobby.", player_name));
        self.events.push(LobbyEvent::PlayerJoined {
            lobby_id: self.id,
            player_id,
        });

        log::info!(
            "Player {} ({}) joined lobby {} ({})",
            player_name,
            player_id.raw(),
            self.name,
            self.id.raw()
        );

        Ok(())
    }

    /// Remove a player from the lobby.
    ///
    /// If the host leaves, triggers host migration.
    pub fn leave(&mut self, player_id: PlayerId) -> EngineResult<()> {
        let player = self.players.remove(&player_id).ok_or_else(|| {
            EngineError::NotFound("player not in lobby".into())
        })?;

        self.push_system_message(&format!("{} left the lobby.", player.name));
        self.events.push(LobbyEvent::PlayerLeft {
            lobby_id: self.id,
            player_id,
        });

        log::info!(
            "Player {} ({}) left lobby {}",
            player.name,
            player_id.raw(),
            self.name
        );

        // Host migration if the host left.
        if self.host_id == Some(player_id) {
            self.migrate_host(player_id);
        }

        Ok(())
    }

    /// Set a player's ready state.
    pub fn set_ready(&mut self, player_id: PlayerId, ready: bool) -> EngineResult<()> {
        let player = self.players.get_mut(&player_id).ok_or_else(|| {
            EngineError::NotFound("player not in lobby".into())
        })?;

        player.ready = ready;

        self.events.push(LobbyEvent::ReadyChanged {
            lobby_id: self.id,
            player_id,
            ready,
        });

        if self.all_ready() {
            self.events.push(LobbyEvent::AllPlayersReady {
                lobby_id: self.id,
            });
        }

        Ok(())
    }

    /// Set a player's team.
    pub fn set_team(&mut self, player_id: PlayerId, team: u32) -> EngineResult<()> {
        if self.settings.team_count > 0 && team > self.settings.team_count {
            return Err(EngineError::InvalidArgument(format!(
                "team must be 0..{}",
                self.settings.team_count
            )));
        }

        // Check team capacity.
        if self.settings.max_per_team > 0 {
            let team_count = self
                .players
                .values()
                .filter(|p| p.team == team && p.id != player_id)
                .count() as u32;
            if team_count >= self.settings.max_per_team {
                return Err(EngineError::InvalidState(format!(
                    "team {} is full ({}/{})",
                    team, team_count, self.settings.max_per_team
                )));
            }
        }

        let player = self.players.get_mut(&player_id).ok_or_else(|| {
            EngineError::NotFound("player not in lobby".into())
        })?;
        player.team = team;

        self.events.push(LobbyEvent::TeamChanged {
            lobby_id: self.id,
            player_id,
            team,
        });

        Ok(())
    }

    /// Update a player's ping.
    pub fn update_ping(&mut self, player_id: PlayerId, ping_ms: u32) {
        if let Some(player) = self.players.get_mut(&player_id) {
            player.ping_ms = ping_ms;
        }
    }

    // -- State transitions --

    /// Attempt to start the game.
    ///
    /// Requires the caller to be the host and all players to be ready.
    pub fn start_game(&mut self, requester: PlayerId) -> EngineResult<()> {
        if self.host_id != Some(requester) {
            return Err(EngineError::InvalidArgument(
                "only the host can start the game".into(),
            ));
        }

        if !self.state.can_start() {
            return Err(EngineError::InvalidState(
                "lobby is not in a state to start".into(),
            ));
        }

        if !self.all_ready() {
            return Err(EngineError::InvalidState(
                "not all players are ready".into(),
            ));
        }

        if self.players.len() < 1 {
            return Err(EngineError::InvalidState(
                "need at least 1 player to start".into(),
            ));
        }

        let old_state = self.state;
        self.state = LobbyState::Starting;

        self.events.push(LobbyEvent::StateChanged {
            lobby_id: self.id,
            old_state,
            new_state: self.state,
        });

        self.push_system_message("Game starting...");
        log::info!("Lobby {} starting game", self.name);

        Ok(())
    }

    /// Transition from Starting to InGame.
    pub fn begin_game(&mut self) -> EngineResult<()> {
        if self.state != LobbyState::Starting {
            return Err(EngineError::InvalidState(
                "lobby must be in Starting state".into(),
            ));
        }

        let old_state = self.state;
        self.state = LobbyState::InGame;

        self.events.push(LobbyEvent::StateChanged {
            lobby_id: self.id,
            old_state,
            new_state: self.state,
        });

        self.push_system_message("Game in progress.");
        Ok(())
    }

    /// End the game.
    pub fn end_game(&mut self) -> EngineResult<()> {
        if self.state != LobbyState::InGame {
            return Err(EngineError::InvalidState(
                "lobby must be in InGame state".into(),
            ));
        }

        let old_state = self.state;
        self.state = LobbyState::Finished;

        // Reset all players' ready status.
        for player in self.players.values_mut() {
            player.ready = false;
        }

        self.events.push(LobbyEvent::StateChanged {
            lobby_id: self.id,
            old_state,
            new_state: self.state,
        });

        self.push_system_message("Game finished.");
        Ok(())
    }

    /// Reset the lobby back to Waiting state (from Finished).
    pub fn reset(&mut self) -> EngineResult<()> {
        if self.state != LobbyState::Finished {
            return Err(EngineError::InvalidState(
                "lobby must be in Finished state to reset".into(),
            ));
        }

        let old_state = self.state;
        self.state = LobbyState::Waiting;

        for player in self.players.values_mut() {
            player.ready = false;
        }

        self.events.push(LobbyEvent::StateChanged {
            lobby_id: self.id,
            old_state,
            new_state: self.state,
        });

        self.push_system_message("Lobby reset. Waiting for players.");
        Ok(())
    }

    // -- Settings --

    /// Update lobby settings. Only the host can change settings.
    pub fn set_settings(
        &mut self,
        requester: PlayerId,
        settings: LobbySettings,
    ) -> EngineResult<()> {
        if self.host_id != Some(requester) {
            return Err(EngineError::InvalidArgument(
                "only the host can change settings".into(),
            ));
        }

        if !self.state.can_change_settings() {
            return Err(EngineError::InvalidState(
                "cannot change settings in this state".into(),
            ));
        }

        settings.validate()?;

        // Don't allow reducing max_players below current count.
        if settings.max_players < self.player_count() {
            return Err(EngineError::InvalidArgument(format!(
                "max_players ({}) cannot be less than current player count ({})",
                settings.max_players,
                self.player_count()
            )));
        }

        self.settings = settings;
        self.events.push(LobbyEvent::SettingsChanged {
            lobby_id: self.id,
        });

        self.push_system_message("Lobby settings changed.");
        Ok(())
    }

    // -- Chat --

    /// Send a chat message.
    pub fn send_message(
        &mut self,
        player_id: PlayerId,
        text: impl Into<String>,
    ) -> EngineResult<()> {
        let player = self.players.get(&player_id).ok_or_else(|| {
            EngineError::NotFound("player not in lobby".into())
        })?;

        let text = text.into();
        if text.is_empty() || text.len() > MAX_CHAT_MESSAGE_LEN {
            return Err(EngineError::InvalidArgument(format!(
                "message must be 1..{} characters",
                MAX_CHAT_MESSAGE_LEN
            )));
        }

        let msg = ChatMessage::player(player_id, &player.name, &text);
        let msg_clone = msg.clone();

        self.chat_history.push(msg);
        if self.chat_history.len() > MAX_CHAT_HISTORY {
            self.chat_history.remove(0);
        }

        self.events.push(LobbyEvent::ChatMessageSent {
            lobby_id: self.id,
            message: msg_clone,
        });

        Ok(())
    }

    /// Push a system message to chat.
    fn push_system_message(&mut self, text: &str) {
        let msg = ChatMessage::system(text);
        self.chat_history.push(msg);
        if self.chat_history.len() > MAX_CHAT_HISTORY {
            self.chat_history.remove(0);
        }
    }

    // -- Host migration --

    /// Migrate the host role to another player.
    fn migrate_host(&mut self, old_host_id: PlayerId) {
        // Select the player who has been in the lobby the longest.
        let new_host = self
            .players
            .values()
            .min_by_key(|p| p.joined_at)
            .map(|p| p.id);

        if let Some(new_host_id) = new_host {
            self.host_id = Some(new_host_id);

            if let Some(player) = self.players.get_mut(&new_host_id) {
                player.is_host = true;
                let name = player.name.clone();
                self.push_system_message(&format!("{} is now the host.", name));
            }

            self.events.push(LobbyEvent::HostChanged {
                lobby_id: self.id,
                old_host: old_host_id,
                new_host: new_host_id,
            });

            log::info!(
                "Lobby {}: host migrated from {} to {}",
                self.name,
                old_host_id.raw(),
                new_host_id.raw()
            );
        } else {
            // No players left.
            self.host_id = None;
            log::info!("Lobby {}: no players remain, host is None", self.name);
        }
    }

    /// Returns the players on a given team.
    pub fn players_on_team(&self, team: u32) -> Vec<&LobbyPlayer> {
        self.players.values().filter(|p| p.team == team).collect()
    }

    /// Returns team counts for all teams.
    pub fn team_counts(&self) -> HashMap<u32, usize> {
        let mut counts = HashMap::new();
        for player in self.players.values() {
            *counts.entry(player.team).or_insert(0) += 1;
        }
        counts
    }

    /// Auto-balance teams by distributing players evenly.
    pub fn auto_balance_teams(&mut self) {
        if self.settings.team_count == 0 {
            return;
        }

        let player_ids: Vec<PlayerId> = self.players.keys().copied().collect();
        for (i, pid) in player_ids.iter().enumerate() {
            let team = (i as u32 % self.settings.team_count) + 1;
            if let Some(player) = self.players.get_mut(pid) {
                player.team = team;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LobbyManager
// ---------------------------------------------------------------------------

/// Manages multiple lobbies.
pub struct LobbyManager {
    /// All active lobbies.
    lobbies: HashMap<LobbyId, Lobby>,
    /// Player to lobby mapping (which lobby each player is in).
    player_lobby: HashMap<PlayerId, LobbyId>,
    /// Next lobby ID to assign.
    next_lobby_id: u64,
    /// Global event queue.
    global_events: Vec<LobbyEvent>,
}

impl LobbyManager {
    /// Create a new lobby manager.
    pub fn new() -> Self {
        Self {
            lobbies: HashMap::new(),
            player_lobby: HashMap::new(),
            next_lobby_id: 1,
            global_events: Vec::new(),
        }
    }

    /// Create a new lobby.
    pub fn create_lobby(
        &mut self,
        name: impl Into<String>,
        host: LobbyPlayer,
        settings: LobbySettings,
    ) -> EngineResult<LobbyId> {
        if self.lobbies.len() >= MAX_LOBBIES {
            return Err(EngineError::InvalidState(
                "maximum number of lobbies reached".into(),
            ));
        }

        // Check if host is already in a lobby.
        if self.player_lobby.contains_key(&host.id) {
            return Err(EngineError::InvalidArgument(
                "player is already in a lobby".into(),
            ));
        }

        let lobby_id = LobbyId::new(self.next_lobby_id);
        self.next_lobby_id += 1;

        let host_id = host.id;
        let lobby = Lobby::new(lobby_id, name, host, settings)?;

        self.lobbies.insert(lobby_id, lobby);
        self.player_lobby.insert(host_id, lobby_id);

        self.global_events.push(LobbyEvent::LobbyCreated { lobby_id });

        log::info!("Lobby {} created", lobby_id.raw());
        Ok(lobby_id)
    }

    /// Destroy a lobby and remove all players.
    pub fn destroy_lobby(&mut self, lobby_id: LobbyId) -> EngineResult<()> {
        let lobby = self.lobbies.remove(&lobby_id).ok_or_else(|| {
            EngineError::NotFound(format!("lobby {} not found", lobby_id.raw()))
        })?;

        // Remove all players from the player_lobby mapping.
        for player in lobby.players() {
            self.player_lobby.remove(&player.id);
        }

        self.global_events
            .push(LobbyEvent::LobbyDestroyed { lobby_id });

        log::info!("Lobby {} destroyed", lobby_id.raw());
        Ok(())
    }

    /// Join a player to a lobby.
    pub fn join_lobby(
        &mut self,
        lobby_id: LobbyId,
        player: LobbyPlayer,
        password: Option<&str>,
    ) -> EngineResult<()> {
        // Check if player is already in a lobby.
        if self.player_lobby.contains_key(&player.id) {
            return Err(EngineError::InvalidArgument(
                "player is already in a lobby".into(),
            ));
        }

        let player_id = player.id;
        let lobby = self.lobbies.get_mut(&lobby_id).ok_or_else(|| {
            EngineError::NotFound(format!("lobby {} not found", lobby_id.raw()))
        })?;

        lobby.join(player, password)?;
        self.player_lobby.insert(player_id, lobby_id);
        Ok(())
    }

    /// Remove a player from their lobby.
    pub fn leave_lobby(&mut self, player_id: PlayerId) -> EngineResult<()> {
        let lobby_id = self.player_lobby.remove(&player_id).ok_or_else(|| {
            EngineError::NotFound("player is not in any lobby".into())
        })?;

        let lobby = self.lobbies.get_mut(&lobby_id).ok_or_else(|| {
            EngineError::NotFound(format!("lobby {} not found", lobby_id.raw()))
        })?;

        lobby.leave(player_id)?;

        // Destroy the lobby if empty.
        if lobby.player_count() == 0 {
            self.lobbies.remove(&lobby_id);
            self.global_events
                .push(LobbyEvent::LobbyDestroyed { lobby_id });
            log::info!("Lobby {} auto-destroyed (empty)", lobby_id.raw());
        }

        Ok(())
    }

    /// Set a player's ready state.
    pub fn set_ready(
        &mut self,
        lobby_id: LobbyId,
        player_id: PlayerId,
        ready: bool,
    ) -> EngineResult<()> {
        let lobby = self.lobbies.get_mut(&lobby_id).ok_or_else(|| {
            EngineError::NotFound(format!("lobby {} not found", lobby_id.raw()))
        })?;
        lobby.set_ready(player_id, ready)
    }

    /// Start the game in a lobby.
    pub fn start_game(&mut self, lobby_id: LobbyId, requester: PlayerId) -> EngineResult<()> {
        let lobby = self.lobbies.get_mut(&lobby_id).ok_or_else(|| {
            EngineError::NotFound(format!("lobby {} not found", lobby_id.raw()))
        })?;
        lobby.start_game(requester)
    }

    /// List all public lobbies.
    pub fn list_lobbies(&self) -> Vec<LobbyInfo> {
        self.lobbies
            .values()
            .filter(|l| l.settings().is_public)
            .map(|l| l.info())
            .collect()
    }

    /// List all lobbies (including private).
    pub fn list_all_lobbies(&self) -> Vec<LobbyInfo> {
        self.lobbies.values().map(|l| l.info()).collect()
    }

    /// Get a reference to a lobby.
    pub fn get_lobby(&self, lobby_id: LobbyId) -> Option<&Lobby> {
        self.lobbies.get(&lobby_id)
    }

    /// Get a mutable reference to a lobby.
    pub fn get_lobby_mut(&mut self, lobby_id: LobbyId) -> Option<&mut Lobby> {
        self.lobbies.get_mut(&lobby_id)
    }

    /// Find the lobby a player is in.
    pub fn player_lobby(&self, player_id: PlayerId) -> Option<LobbyId> {
        self.player_lobby.get(&player_id).copied()
    }

    /// Send a chat message in a lobby.
    pub fn send_message(
        &mut self,
        lobby_id: LobbyId,
        player_id: PlayerId,
        message: impl Into<String>,
    ) -> EngineResult<()> {
        let lobby = self.lobbies.get_mut(&lobby_id).ok_or_else(|| {
            EngineError::NotFound(format!("lobby {} not found", lobby_id.raw()))
        })?;
        lobby.send_message(player_id, message)
    }

    /// Update lobby settings.
    pub fn set_lobby_settings(
        &mut self,
        lobby_id: LobbyId,
        requester: PlayerId,
        settings: LobbySettings,
    ) -> EngineResult<()> {
        let lobby = self.lobbies.get_mut(&lobby_id).ok_or_else(|| {
            EngineError::NotFound(format!("lobby {} not found", lobby_id.raw()))
        })?;
        lobby.set_settings(requester, settings)
    }

    /// Returns the total number of active lobbies.
    pub fn lobby_count(&self) -> usize {
        self.lobbies.len()
    }

    /// Returns the total number of players across all lobbies.
    pub fn total_players(&self) -> usize {
        self.player_lobby.len()
    }

    /// Drain global events.
    pub fn drain_events(&mut self) -> Vec<LobbyEvent> {
        std::mem::take(&mut self.global_events)
    }
}

impl Default for LobbyManager {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_player(id: u32, name: &str) -> LobbyPlayer {
        LobbyPlayer::new(PlayerId::new(id), name)
    }

    fn make_host(id: u32, name: &str) -> LobbyPlayer {
        LobbyPlayer::host(PlayerId::new(id), name)
    }

    fn default_settings() -> LobbySettings {
        LobbySettings::new("deathmatch", "arena", 4)
    }

    // -----------------------------------------------------------------------
    // LobbySettings
    // -----------------------------------------------------------------------

    #[test]
    fn test_settings_validation() {
        let mut s = default_settings();
        assert!(s.validate().is_ok());

        s.max_players = 0;
        assert!(s.validate().is_err());

        s.max_players = MAX_PLAYERS_PER_LOBBY + 1;
        assert!(s.validate().is_err());

        let mut s2 = default_settings();
        s2.game_mode = String::new();
        assert!(s2.validate().is_err());
    }

    #[test]
    fn test_settings_password() {
        let mut s = default_settings();
        assert!(!s.is_password_protected());
        assert!(s.check_password("anything"));

        s.password = Some("secret".into());
        assert!(s.is_password_protected());
        assert!(s.check_password("secret"));
        assert!(!s.check_password("wrong"));
    }

    #[test]
    fn test_settings_custom() {
        let mut s = default_settings();
        s.set_custom("rounds", "5");
        assert_eq!(s.get_custom("rounds"), Some("5"));
        assert!(s.get_custom("nonexistent").is_none());
    }

    // -----------------------------------------------------------------------
    // LobbyState
    // -----------------------------------------------------------------------

    #[test]
    fn test_state_transitions() {
        assert!(LobbyState::Waiting.can_join());
        assert!(!LobbyState::Starting.can_join());
        assert!(!LobbyState::InGame.can_join());
        assert!(!LobbyState::Finished.can_join());

        assert!(LobbyState::Waiting.can_start());
        assert!(!LobbyState::InGame.can_start());

        assert_eq!(LobbyState::Waiting.next(), Some(LobbyState::Starting));
        assert_eq!(LobbyState::Starting.next(), Some(LobbyState::InGame));
        assert_eq!(LobbyState::InGame.next(), Some(LobbyState::Finished));
        assert_eq!(LobbyState::Finished.next(), Some(LobbyState::Waiting));
    }

    // -----------------------------------------------------------------------
    // Lobby lifecycle
    // -----------------------------------------------------------------------

    #[test]
    fn test_lobby_create() {
        let host = make_host(1, "Alice");
        let lobby = Lobby::new(LobbyId::new(1), "Test Lobby", host, default_settings()).unwrap();

        assert_eq!(lobby.state(), LobbyState::Waiting);
        assert_eq!(lobby.player_count(), 1);
        assert_eq!(lobby.host_id(), Some(PlayerId::new(1)));
        assert!(!lobby.is_full());
    }

    #[test]
    fn test_lobby_join_leave() {
        let host = make_host(1, "Alice");
        let mut lobby =
            Lobby::new(LobbyId::new(1), "Test", host, default_settings()).unwrap();

        let player = make_player(2, "Bob");
        lobby.join(player, None).unwrap();
        assert_eq!(lobby.player_count(), 2);

        lobby.leave(PlayerId::new(2)).unwrap();
        assert_eq!(lobby.player_count(), 1);
    }

    #[test]
    fn test_lobby_full() {
        let settings = LobbySettings::new("dm", "arena", 2);
        let host = make_host(1, "Alice");
        let mut lobby =
            Lobby::new(LobbyId::new(1), "Test", host, settings).unwrap();

        lobby.join(make_player(2, "Bob"), None).unwrap();
        assert!(lobby.is_full());

        let result = lobby.join(make_player(3, "Charlie"), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_lobby_password() {
        let mut settings = default_settings();
        settings.password = Some("pass123".into());

        let host = make_host(1, "Alice");
        let mut lobby =
            Lobby::new(LobbyId::new(1), "Private", host, settings).unwrap();

        // Wrong password.
        let result = lobby.join(make_player(2, "Bob"), Some("wrong"));
        assert!(result.is_err());

        // Correct password.
        let result = lobby.join(make_player(2, "Bob"), Some("pass123"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_lobby_ready_and_start() {
        let host = make_host(1, "Alice");
        let mut lobby =
            Lobby::new(LobbyId::new(1), "Test", host, default_settings()).unwrap();

        lobby.join(make_player(2, "Bob"), None).unwrap();

        // Not all ready yet.
        assert!(!lobby.all_ready());

        lobby.set_ready(PlayerId::new(1), true).unwrap();
        assert!(!lobby.all_ready());

        lobby.set_ready(PlayerId::new(2), true).unwrap();
        assert!(lobby.all_ready());

        // Non-host cannot start.
        let result = lobby.start_game(PlayerId::new(2));
        assert!(result.is_err());

        // Host can start.
        lobby.start_game(PlayerId::new(1)).unwrap();
        assert_eq!(lobby.state(), LobbyState::Starting);
    }

    #[test]
    fn test_lobby_full_lifecycle() {
        let host = make_host(1, "Alice");
        let mut lobby =
            Lobby::new(LobbyId::new(1), "Test", host, default_settings()).unwrap();

        lobby.set_ready(PlayerId::new(1), true).unwrap();
        lobby.start_game(PlayerId::new(1)).unwrap();
        assert_eq!(lobby.state(), LobbyState::Starting);

        lobby.begin_game().unwrap();
        assert_eq!(lobby.state(), LobbyState::InGame);

        lobby.end_game().unwrap();
        assert_eq!(lobby.state(), LobbyState::Finished);

        lobby.reset().unwrap();
        assert_eq!(lobby.state(), LobbyState::Waiting);
    }

    // -----------------------------------------------------------------------
    // Host migration
    // -----------------------------------------------------------------------

    #[test]
    fn test_host_migration() {
        let host = make_host(1, "Alice");
        let mut lobby =
            Lobby::new(LobbyId::new(1), "Test", host, default_settings()).unwrap();

        lobby.join(make_player(2, "Bob"), None).unwrap();
        lobby.join(make_player(3, "Charlie"), None).unwrap();

        // Host leaves.
        lobby.leave(PlayerId::new(1)).unwrap();

        // Host should have migrated.
        let new_host = lobby.host_id().unwrap();
        assert_ne!(new_host, PlayerId::new(1));

        let host_player = lobby.get_player(new_host).unwrap();
        assert!(host_player.is_host);
    }

    #[test]
    fn test_host_migration_empty() {
        let host = make_host(1, "Alice");
        let mut lobby =
            Lobby::new(LobbyId::new(1), "Test", host, default_settings()).unwrap();

        lobby.leave(PlayerId::new(1)).unwrap();
        assert!(lobby.host_id().is_none());
        assert_eq!(lobby.player_count(), 0);
    }

    // -----------------------------------------------------------------------
    // Teams
    // -----------------------------------------------------------------------

    #[test]
    fn test_lobby_teams() {
        let mut settings = default_settings();
        settings.team_count = 2;
        settings.max_per_team = 2;

        let host = make_host(1, "Alice");
        let mut lobby =
            Lobby::new(LobbyId::new(1), "Team Game", host, settings).unwrap();

        lobby.join(make_player(2, "Bob"), None).unwrap();
        lobby.join(make_player(3, "Charlie"), None).unwrap();
        lobby.join(make_player(4, "Diana"), None).unwrap();

        lobby.set_team(PlayerId::new(1), 1).unwrap();
        lobby.set_team(PlayerId::new(2), 1).unwrap();
        lobby.set_team(PlayerId::new(3), 2).unwrap();

        assert_eq!(lobby.players_on_team(1).len(), 2);
        assert_eq!(lobby.players_on_team(2).len(), 1);

        // Team 1 is full.
        let result = lobby.set_team(PlayerId::new(4), 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_auto_balance() {
        let mut settings = default_settings();
        settings.team_count = 2;

        let host = make_host(1, "Alice");
        let mut lobby =
            Lobby::new(LobbyId::new(1), "Balance Test", host, settings).unwrap();

        lobby.join(make_player(2, "Bob"), None).unwrap();
        lobby.join(make_player(3, "Charlie"), None).unwrap();
        lobby.join(make_player(4, "Diana"), None).unwrap();

        lobby.auto_balance_teams();

        let counts = lobby.team_counts();
        // Should be evenly distributed (2 on each team).
        assert_eq!(counts.values().max(), counts.values().min());
    }

    // -----------------------------------------------------------------------
    // Chat
    // -----------------------------------------------------------------------

    #[test]
    fn test_lobby_chat() {
        let host = make_host(1, "Alice");
        let mut lobby =
            Lobby::new(LobbyId::new(1), "Chat Test", host, default_settings()).unwrap();

        lobby
            .send_message(PlayerId::new(1), "Hello everyone!")
            .unwrap();

        let history = lobby.chat_history();
        // System message from creation + player message.
        assert!(history.len() >= 2);

        let last = history.last().unwrap();
        assert_eq!(last.text, "Hello everyone!");
        assert!(!last.is_system());
        assert_eq!(last.sender, Some(PlayerId::new(1)));
    }

    #[test]
    fn test_chat_invalid() {
        let host = make_host(1, "Alice");
        let mut lobby =
            Lobby::new(LobbyId::new(1), "Test", host, default_settings()).unwrap();

        // Empty message.
        let result = lobby.send_message(PlayerId::new(1), "");
        assert!(result.is_err());

        // Too long.
        let long_msg = "x".repeat(MAX_CHAT_MESSAGE_LEN + 1);
        let result = lobby.send_message(PlayerId::new(1), long_msg);
        assert!(result.is_err());

        // Non-existent player.
        let result = lobby.send_message(PlayerId::new(99), "hello");
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // LobbyManager
    // -----------------------------------------------------------------------

    #[test]
    fn test_manager_create_and_list() {
        let mut mgr = LobbyManager::new();
        let host = make_host(1, "Alice");
        let lobby_id = mgr
            .create_lobby("Game 1", host, default_settings())
            .unwrap();

        assert_eq!(mgr.lobby_count(), 1);
        assert_eq!(mgr.total_players(), 1);

        let lobbies = mgr.list_lobbies();
        assert_eq!(lobbies.len(), 1);
        assert_eq!(lobbies[0].id, lobby_id);
        assert_eq!(lobbies[0].name, "Game 1");
    }

    #[test]
    fn test_manager_join_leave() {
        let mut mgr = LobbyManager::new();
        let host = make_host(1, "Alice");
        let lobby_id = mgr
            .create_lobby("Game", host, default_settings())
            .unwrap();

        mgr.join_lobby(lobby_id, make_player(2, "Bob"), None)
            .unwrap();
        assert_eq!(mgr.total_players(), 2);

        mgr.leave_lobby(PlayerId::new(2)).unwrap();
        assert_eq!(mgr.total_players(), 1);
    }

    #[test]
    fn test_manager_auto_destroy() {
        let mut mgr = LobbyManager::new();
        let host = make_host(1, "Alice");
        let lobby_id = mgr
            .create_lobby("Game", host, default_settings())
            .unwrap();

        mgr.leave_lobby(PlayerId::new(1)).unwrap();
        // Lobby should be auto-destroyed.
        assert_eq!(mgr.lobby_count(), 0);
        assert!(mgr.get_lobby(lobby_id).is_none());
    }

    #[test]
    fn test_manager_player_already_in_lobby() {
        let mut mgr = LobbyManager::new();
        let host = make_host(1, "Alice");
        mgr.create_lobby("Game 1", host, default_settings()).unwrap();

        // Same player cannot join another lobby.
        let host2 = make_host(1, "Alice");
        let result = mgr.create_lobby("Game 2", host2, default_settings());
        assert!(result.is_err());
    }

    #[test]
    fn test_manager_find_player_lobby() {
        let mut mgr = LobbyManager::new();
        let host = make_host(1, "Alice");
        let lobby_id = mgr
            .create_lobby("Game", host, default_settings())
            .unwrap();

        assert_eq!(mgr.player_lobby(PlayerId::new(1)), Some(lobby_id));
        assert!(mgr.player_lobby(PlayerId::new(99)).is_none());
    }

    #[test]
    fn test_manager_private_lobby_not_listed() {
        let mut mgr = LobbyManager::new();
        let mut settings = default_settings();
        settings.is_public = false;

        let host = make_host(1, "Alice");
        mgr.create_lobby("Private", host, settings).unwrap();

        let public = mgr.list_lobbies();
        assert!(public.is_empty());

        let all = mgr.list_all_lobbies();
        assert_eq!(all.len(), 1);
    }

    #[test]
    fn test_manager_destroy_lobby() {
        let mut mgr = LobbyManager::new();
        let host = make_host(1, "Alice");
        let lobby_id = mgr
            .create_lobby("Game", host, default_settings())
            .unwrap();

        mgr.join_lobby(lobby_id, make_player(2, "Bob"), None)
            .unwrap();

        mgr.destroy_lobby(lobby_id).unwrap();
        assert_eq!(mgr.lobby_count(), 0);
        assert_eq!(mgr.total_players(), 0);
    }

    // -----------------------------------------------------------------------
    // LobbyInfo
    // -----------------------------------------------------------------------

    #[test]
    fn test_lobby_info() {
        let host = make_host(1, "Alice");
        let lobby =
            Lobby::new(LobbyId::new(1), "Info Test", host, default_settings()).unwrap();

        let info = lobby.info();
        assert_eq!(info.name, "Info Test");
        assert_eq!(info.host_name, "Alice");
        assert_eq!(info.player_count, 1);
        assert_eq!(info.max_players, 4);
        assert_eq!(info.game_mode, "deathmatch");
        assert!(!info.has_password);
        assert!(info.is_public);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_lobby_settings_change() {
        let host = make_host(1, "Alice");
        let mut lobby =
            Lobby::new(LobbyId::new(1), "Test", host, default_settings()).unwrap();

        let new_settings = LobbySettings::new("ctf", "castle", 8);
        lobby
            .set_settings(PlayerId::new(1), new_settings)
            .unwrap();
        assert_eq!(lobby.settings().game_mode, "ctf");
        assert_eq!(lobby.settings().map, "castle");
    }

    #[test]
    fn test_lobby_settings_change_non_host() {
        let host = make_host(1, "Alice");
        let mut lobby =
            Lobby::new(LobbyId::new(1), "Test", host, default_settings()).unwrap();

        lobby.join(make_player(2, "Bob"), None).unwrap();

        let new_settings = default_settings();
        let result = lobby.set_settings(PlayerId::new(2), new_settings);
        assert!(result.is_err());
    }

    #[test]
    fn test_lobby_cannot_start_not_ready() {
        let host = make_host(1, "Alice");
        let mut lobby =
            Lobby::new(LobbyId::new(1), "Test", host, default_settings()).unwrap();

        lobby.join(make_player(2, "Bob"), None).unwrap();

        // Only host is ready, not Bob.
        lobby.set_ready(PlayerId::new(1), true).unwrap();
        let result = lobby.start_game(PlayerId::new(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_lobby_events() {
        let host = make_host(1, "Alice");
        let mut lobby =
            Lobby::new(LobbyId::new(1), "Events", host, default_settings()).unwrap();

        lobby.join(make_player(2, "Bob"), None).unwrap();
        lobby.set_ready(PlayerId::new(1), true).unwrap();

        let events = lobby.drain_events();
        assert!(!events.is_empty());

        // Should contain PlayerJoined and ReadyChanged.
        let has_join = events
            .iter()
            .any(|e| matches!(e, LobbyEvent::PlayerJoined { .. }));
        let has_ready = events
            .iter()
            .any(|e| matches!(e, LobbyEvent::ReadyChanged { .. }));
        assert!(has_join);
        assert!(has_ready);
    }

    #[test]
    fn test_player_properties() {
        let mut player = make_player(1, "Alice");
        player.set_property("character", "warrior");
        assert_eq!(player.get_property("character"), Some("warrior"));
        assert!(player.get_property("nonexistent").is_none());
    }

    #[test]
    fn test_lobby_ping_update() {
        let host = make_host(1, "Alice");
        let mut lobby =
            Lobby::new(LobbyId::new(1), "Ping", host, default_settings()).unwrap();

        lobby.update_ping(PlayerId::new(1), 50);
        let player = lobby.get_player(PlayerId::new(1)).unwrap();
        assert_eq!(player.ping_ms, 50);
    }
}
