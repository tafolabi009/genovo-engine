//! Music system with crossfade, playlists, layered/adaptive music,
//! beat tracking, horizontal resequencing, and stingers.
//!
//! Provides:
//! - `MusicPlayer`: background music with crossfade, volume fading, playlists
//! - `MusicLayer`: layered adaptive music with per-layer volume, state transitions,
//!   beat synchronisation, and stingers
//! - `MusicComponent`, `MusicSystem` for ECS integration

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default crossfade duration in seconds.
const DEFAULT_CROSSFADE: f32 = 2.0;
/// Default BPM for beat tracking.
const DEFAULT_BPM: f32 = 120.0;

// ---------------------------------------------------------------------------
// Music state (for adaptive music)
// ---------------------------------------------------------------------------

/// Game state that drives adaptive music behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MusicState {
    /// Peaceful exploration.
    Explore,
    /// Active combat.
    Combat,
    /// Stealth / tension.
    Stealth,
    /// Boss encounter.
    Boss,
    /// Victory / celebration.
    Victory,
    /// Custom state identified by index.
    Custom(u16),
}

impl Default for MusicState {
    fn default() -> Self {
        Self::Explore
    }
}

// ---------------------------------------------------------------------------
// Transition type
// ---------------------------------------------------------------------------

/// How to transition between music states or tracks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransitionType {
    /// Immediate cut (no fade).
    Immediate,
    /// Simple crossfade over the given duration (seconds).
    Crossfade(f32),
    /// Fade out current, silence gap, then fade in next.
    FadeOutFadeIn { fade_out: f32, gap: f32, fade_in: f32 },
    /// Synchronise the transition to the next beat boundary.
    SyncToBeat,
}

impl Default for TransitionType {
    fn default() -> Self {
        Self::Crossfade(DEFAULT_CROSSFADE)
    }
}

// ---------------------------------------------------------------------------
// MusicTrack
// ---------------------------------------------------------------------------

/// Reference to a music audio clip.
#[derive(Debug, Clone)]
pub struct MusicTrack {
    /// Clip name (matches the name in the audio mixer's clip registry).
    pub clip_name: String,
    /// Track title (for UI display).
    pub title: String,
    /// Duration in seconds (0 = unknown / determine from clip).
    pub duration: f32,
    /// BPM of this track (for beat sync).
    pub bpm: f32,
    /// Volume multiplier for this track.
    pub volume: f32,
}

impl MusicTrack {
    /// Create a new music track.
    pub fn new(clip_name: &str) -> Self {
        Self {
            clip_name: clip_name.to_string(),
            title: clip_name.to_string(),
            duration: 0.0,
            bpm: DEFAULT_BPM,
            volume: 1.0,
        }
    }

    /// Set the title.
    pub fn with_title(mut self, title: &str) -> Self {
        self.title = title.to_string();
        self
    }

    /// Set BPM.
    pub fn with_bpm(mut self, bpm: f32) -> Self {
        self.bpm = bpm;
        self
    }

    /// Set volume.
    pub fn with_volume(mut self, volume: f32) -> Self {
        self.volume = volume;
        self
    }
}

// ---------------------------------------------------------------------------
// Playlist
// ---------------------------------------------------------------------------

/// An ordered or shuffled list of music tracks.
#[derive(Debug, Clone)]
pub struct Playlist {
    /// Tracks in the playlist.
    pub tracks: Vec<MusicTrack>,
    /// Current track index.
    pub current_index: usize,
    /// Whether to shuffle playback order.
    pub shuffle: bool,
    /// Whether to loop the playlist.
    pub loop_playlist: bool,
    /// Shuffled order indices (when shuffle is enabled).
    shuffled_order: Vec<usize>,
}

impl Playlist {
    /// Create a new empty playlist.
    pub fn new() -> Self {
        Self {
            tracks: Vec::new(),
            current_index: 0,
            shuffle: false,
            loop_playlist: true,
            shuffled_order: Vec::new(),
        }
    }

    /// Add a track to the playlist.
    pub fn add_track(&mut self, track: MusicTrack) {
        self.tracks.push(track);
        self.rebuild_shuffle_order();
    }

    /// Get the current track.
    pub fn current_track(&self) -> Option<&MusicTrack> {
        let idx = self.effective_index();
        if idx < self.tracks.len() {
            Some(&self.tracks[idx])
        } else {
            None
        }
    }

    /// Advance to the next track.
    pub fn next(&mut self) -> Option<&MusicTrack> {
        if self.tracks.is_empty() {
            return None;
        }
        self.current_index += 1;
        if self.current_index >= self.tracks.len() {
            if self.loop_playlist {
                self.current_index = 0;
                if self.shuffle {
                    self.rebuild_shuffle_order();
                }
            } else {
                self.current_index = self.tracks.len(); // Past end
                return None;
            }
        }
        self.current_track()
    }

    /// Go to the previous track.
    pub fn previous(&mut self) -> Option<&MusicTrack> {
        if self.tracks.is_empty() {
            return None;
        }
        if self.current_index > 0 {
            self.current_index -= 1;
        } else if self.loop_playlist {
            self.current_index = self.tracks.len() - 1;
        }
        self.current_track()
    }

    /// Get the number of tracks.
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    fn effective_index(&self) -> usize {
        if self.shuffle && self.current_index < self.shuffled_order.len() {
            self.shuffled_order[self.current_index]
        } else {
            self.current_index
        }
    }

    fn rebuild_shuffle_order(&mut self) {
        self.shuffled_order = (0..self.tracks.len()).collect();
        // Simple deterministic shuffle using index-based swap
        let n = self.shuffled_order.len();
        for i in (1..n).rev() {
            let j = (i * 7 + 3) % (i + 1); // Pseudo-random but deterministic
            self.shuffled_order.swap(i, j);
        }
    }
}

impl Default for Playlist {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Fade envelope
// ---------------------------------------------------------------------------

/// State of a volume fade.
#[derive(Debug, Clone)]
struct FadeState {
    /// Whether a fade is in progress.
    active: bool,
    /// Start volume.
    start_volume: f32,
    /// Target volume.
    target_volume: f32,
    /// Fade duration in seconds.
    duration: f32,
    /// Elapsed time in the fade.
    elapsed: f32,
}

impl FadeState {
    fn new() -> Self {
        Self {
            active: false,
            start_volume: 1.0,
            target_volume: 1.0,
            duration: 0.0,
            elapsed: 0.0,
        }
    }

    fn start(&mut self, from: f32, to: f32, duration: f32) {
        self.active = true;
        self.start_volume = from;
        self.target_volume = to;
        self.duration = duration.max(0.001);
        self.elapsed = 0.0;
    }

    fn update(&mut self, dt: f32) -> f32 {
        if !self.active {
            return self.target_volume;
        }
        self.elapsed += dt;
        let t = (self.elapsed / self.duration).clamp(0.0, 1.0);
        let volume = self.start_volume + (self.target_volume - self.start_volume) * t;
        if t >= 1.0 {
            self.active = false;
        }
        volume
    }

    fn current_volume(&self) -> f32 {
        if !self.active {
            return self.target_volume;
        }
        let t = (self.elapsed / self.duration).clamp(0.0, 1.0);
        self.start_volume + (self.target_volume - self.start_volume) * t
    }
}

// ---------------------------------------------------------------------------
// MusicPlayer
// ---------------------------------------------------------------------------

/// Playback state of the music player.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MusicPlaybackState {
    Stopped,
    Playing,
    Paused,
    CrossfadingOut,
    CrossfadingIn,
    FadingIn,
    FadingOut,
}

/// Music player for background music with crossfade, volume fading, and playlists.
#[derive(Debug, Clone)]
pub struct MusicPlayer {
    /// Currently playing track.
    pub current_track: Option<MusicTrack>,
    /// Next track (during crossfade).
    pub next_track: Option<MusicTrack>,
    /// Current playback state.
    pub state: MusicPlaybackState,
    /// Master music volume.
    pub volume: f32,
    /// Current effective volume (after fading).
    pub effective_volume: f32,
    /// Volume fade state for the current track.
    fade: FadeState,
    /// Crossfade state for the incoming track.
    crossfade_in: FadeState,
    /// Whether to loop the current single track.
    pub loop_track: bool,
    /// Active playlist (if any).
    pub playlist: Option<Playlist>,
    /// Whether to auto-advance the playlist when a track ends.
    pub auto_advance: bool,
    /// Playback position in seconds (approximate).
    pub playback_position: f32,
    /// Whether the player is muted.
    pub muted: bool,
}

impl MusicPlayer {
    /// Create a new music player.
    pub fn new() -> Self {
        Self {
            current_track: None,
            next_track: None,
            state: MusicPlaybackState::Stopped,
            volume: 1.0,
            effective_volume: 1.0,
            fade: FadeState::new(),
            crossfade_in: FadeState::new(),
            loop_track: false,
            playlist: None,
            auto_advance: true,
            playback_position: 0.0,
            muted: false,
        }
    }

    /// Play a track with an optional fade-in.
    pub fn play_track(&mut self, track: MusicTrack, fade_in: f32) {
        if fade_in > 0.0 {
            self.fade.start(0.0, track.volume, fade_in);
            self.state = MusicPlaybackState::FadingIn;
        } else {
            self.fade.active = false;
            self.effective_volume = track.volume;
            self.state = MusicPlaybackState::Playing;
        }
        self.current_track = Some(track);
        self.next_track = None;
        self.playback_position = 0.0;
    }

    /// Stop playback with an optional fade-out.
    pub fn stop(&mut self, fade_out: f32) {
        if fade_out > 0.0 && self.state != MusicPlaybackState::Stopped {
            self.fade.start(self.effective_volume, 0.0, fade_out);
            self.state = MusicPlaybackState::FadingOut;
        } else {
            self.state = MusicPlaybackState::Stopped;
            self.effective_volume = 0.0;
            self.current_track = None;
        }
    }

    /// Crossfade to a new track.
    pub fn crossfade_to(&mut self, next: MusicTrack, duration: f32) {
        let duration = duration.max(0.01);
        // Fade out current
        self.fade.start(self.effective_volume, 0.0, duration);
        // Fade in next
        self.crossfade_in.start(0.0, next.volume, duration);
        self.next_track = Some(next);
        self.state = MusicPlaybackState::CrossfadingOut;
    }

    /// Set the master music volume with an optional fade.
    pub fn set_volume(&mut self, volume: f32, fade: f32) {
        self.volume = volume.max(0.0);
        if fade > 0.0 {
            self.fade.start(self.effective_volume, volume, fade);
        } else {
            self.effective_volume = volume;
        }
    }

    /// Pause playback.
    pub fn pause(&mut self) {
        if self.state == MusicPlaybackState::Playing {
            self.state = MusicPlaybackState::Paused;
        }
    }

    /// Resume playback.
    pub fn resume(&mut self) {
        if self.state == MusicPlaybackState::Paused {
            self.state = MusicPlaybackState::Playing;
        }
    }

    /// Update the music player. Call once per frame.
    pub fn update(&mut self, dt: f32) {
        if self.state == MusicPlaybackState::Stopped || self.state == MusicPlaybackState::Paused {
            return;
        }

        self.playback_position += dt;

        match self.state {
            MusicPlaybackState::FadingIn => {
                self.effective_volume = self.fade.update(dt);
                if !self.fade.active {
                    self.state = MusicPlaybackState::Playing;
                }
            }
            MusicPlaybackState::FadingOut => {
                self.effective_volume = self.fade.update(dt);
                if !self.fade.active {
                    self.state = MusicPlaybackState::Stopped;
                    self.current_track = None;
                }
            }
            MusicPlaybackState::CrossfadingOut => {
                self.effective_volume = self.fade.update(dt);
                let _incoming_vol = self.crossfade_in.update(dt);

                if !self.fade.active {
                    // Crossfade complete: swap tracks
                    self.current_track = self.next_track.take();
                    self.effective_volume = self.crossfade_in.current_volume();
                    self.state = MusicPlaybackState::Playing;
                    self.playback_position = 0.0;
                }
            }
            MusicPlaybackState::Playing => {
                // Check if current track has ended (if duration is known)
                if let Some(track) = &self.current_track {
                    if track.duration > 0.0 && self.playback_position >= track.duration {
                        if self.loop_track {
                            self.playback_position = 0.0;
                        } else if self.auto_advance {
                            self.advance_playlist();
                        } else {
                            self.state = MusicPlaybackState::Stopped;
                        }
                    }
                }

                // Update ongoing volume fade
                if self.fade.active {
                    self.effective_volume = self.fade.update(dt);
                }
            }
            _ => {}
        }
    }

    /// Advance to the next track in the playlist.
    fn advance_playlist(&mut self) {
        if let Some(ref mut playlist) = self.playlist {
            if let Some(track) = playlist.next() {
                let track = track.clone();
                self.crossfade_to(track, DEFAULT_CROSSFADE);
            } else {
                self.state = MusicPlaybackState::Stopped;
            }
        }
    }

    /// Get the current effective volume (accounting for fades and mute).
    pub fn output_volume(&self) -> f32 {
        if self.muted {
            0.0
        } else {
            self.effective_volume * self.volume
        }
    }

    /// Whether music is currently audible.
    pub fn is_playing(&self) -> bool {
        self.state == MusicPlaybackState::Playing
            || self.state == MusicPlaybackState::FadingIn
            || self.state == MusicPlaybackState::CrossfadingOut
    }
}

impl Default for MusicPlayer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Beat tracker
// ---------------------------------------------------------------------------

/// Beat tracking for synchronising music transitions.
#[derive(Debug, Clone)]
pub struct BeatTracker {
    /// Beats per minute.
    pub bpm: f32,
    /// Current beat position (fractional).
    pub current_beat: f32,
    /// Beat duration in seconds.
    pub beat_duration: f32,
    /// Bar length in beats (typically 4).
    pub beats_per_bar: u32,
    /// Current bar number.
    pub current_bar: u32,
    /// Time accumulated since last beat.
    time_since_beat: f32,
}

impl BeatTracker {
    /// Create a new beat tracker.
    pub fn new(bpm: f32, beats_per_bar: u32) -> Self {
        let beat_duration = 60.0 / bpm.max(1.0);
        Self {
            bpm,
            current_beat: 0.0,
            beat_duration,
            beats_per_bar: beats_per_bar.max(1),
            current_bar: 0,
            time_since_beat: 0.0,
        }
    }

    /// Set a new BPM.
    pub fn set_bpm(&mut self, bpm: f32) {
        self.bpm = bpm.max(1.0);
        self.beat_duration = 60.0 / self.bpm;
    }

    /// Update the beat tracker by `dt` seconds. Returns true if a beat occurred.
    pub fn update(&mut self, dt: f32) -> bool {
        self.time_since_beat += dt;
        self.current_beat += dt / self.beat_duration;

        if self.time_since_beat >= self.beat_duration {
            self.time_since_beat -= self.beat_duration;
            let beat_in_bar = (self.current_beat as u32) % self.beats_per_bar;
            if beat_in_bar == 0 {
                self.current_bar += 1;
            }
            return true;
        }
        false
    }

    /// Time until the next beat boundary.
    pub fn time_to_next_beat(&self) -> f32 {
        (self.beat_duration - self.time_since_beat).max(0.0)
    }

    /// Time until the next bar boundary.
    pub fn time_to_next_bar(&self) -> f32 {
        let beat_in_bar = (self.current_beat as u32) % self.beats_per_bar;
        let beats_remaining = self.beats_per_bar - beat_in_bar;
        let time_remaining =
            beats_remaining as f32 * self.beat_duration - self.time_since_beat;
        time_remaining.max(0.0)
    }

    /// Reset the beat tracker.
    pub fn reset(&mut self) {
        self.current_beat = 0.0;
        self.current_bar = 0;
        self.time_since_beat = 0.0;
    }
}

impl Default for BeatTracker {
    fn default() -> Self {
        Self::new(DEFAULT_BPM, 4)
    }
}

// ---------------------------------------------------------------------------
// Stinger
// ---------------------------------------------------------------------------

/// A one-shot musical accent played on an event, overlaid on current music.
#[derive(Debug, Clone)]
pub struct Stinger {
    /// Clip name of the stinger sound.
    pub clip_name: String,
    /// Volume of the stinger.
    pub volume: f32,
    /// Whether this stinger is currently playing.
    pub playing: bool,
    /// Playback position.
    pub position: f32,
    /// Duration (0 = determine from clip).
    pub duration: f32,
}

impl Stinger {
    /// Create a new stinger.
    pub fn new(clip_name: &str, volume: f32) -> Self {
        Self {
            clip_name: clip_name.to_string(),
            volume,
            playing: false,
            position: 0.0,
            duration: 0.0,
        }
    }

    /// Trigger the stinger.
    pub fn trigger(&mut self) {
        self.playing = true;
        self.position = 0.0;
    }

    /// Update the stinger. Returns true if still playing.
    pub fn update(&mut self, dt: f32) -> bool {
        if !self.playing {
            return false;
        }
        self.position += dt;
        if self.duration > 0.0 && self.position >= self.duration {
            self.playing = false;
        }
        self.playing
    }
}

// ---------------------------------------------------------------------------
// MusicLayer
// ---------------------------------------------------------------------------

/// A single layer in a layered music system.
#[derive(Debug, Clone)]
pub struct LayerInfo {
    /// Layer name (e.g. "drums", "bass", "melody").
    pub name: String,
    /// Clip name.
    pub clip_name: String,
    /// Current volume [0, 1].
    pub volume: f32,
    /// Target volume (for fading).
    pub target_volume: f32,
    /// Fade speed (volume units per second).
    pub fade_speed: f32,
    /// Whether this layer is muted.
    pub muted: bool,
}

impl LayerInfo {
    /// Create a new layer.
    pub fn new(name: &str, clip_name: &str) -> Self {
        Self {
            name: name.to_string(),
            clip_name: clip_name.to_string(),
            volume: 0.0,
            target_volume: 0.0,
            fade_speed: 1.0,
            muted: false,
        }
    }

    /// Set volume with fade.
    pub fn set_volume(&mut self, volume: f32, fade_duration: f32) {
        self.target_volume = volume.clamp(0.0, 1.0);
        if fade_duration > 0.0 {
            self.fade_speed = (self.target_volume - self.volume).abs() / fade_duration;
        } else {
            self.volume = self.target_volume;
            self.fade_speed = 0.0;
        }
    }

    /// Update volume fade.
    pub fn update(&mut self, dt: f32) {
        if (self.volume - self.target_volume).abs() < 0.001 {
            self.volume = self.target_volume;
            return;
        }

        let direction = if self.target_volume > self.volume {
            1.0
        } else {
            -1.0
        };
        self.volume += direction * self.fade_speed * dt;

        if direction > 0.0 {
            self.volume = self.volume.min(self.target_volume);
        } else {
            self.volume = self.volume.max(self.target_volume);
        }
    }

    /// Effective volume (accounting for mute).
    pub fn effective_volume(&self) -> f32 {
        if self.muted {
            0.0
        } else {
            self.volume
        }
    }
}

/// Transition rule: maps from one state to another with a transition type.
#[derive(Debug, Clone)]
pub struct TransitionRule {
    pub from_state: MusicState,
    pub to_state: MusicState,
    pub transition: TransitionType,
}

/// Layered adaptive music system.
///
/// Multiple layers play simultaneously (drums, bass, melody, etc.) with
/// individual volume control. The system responds to game state changes
/// by adjusting layer volumes and transitioning between song sections.
#[derive(Debug, Clone)]
pub struct MusicLayerSystem {
    /// All music layers.
    pub layers: Vec<LayerInfo>,
    /// Layer configuration per state: maps state -> layer_index -> target_volume.
    pub state_configs: HashMap<MusicState, Vec<f32>>,
    /// Current music state.
    pub current_state: MusicState,
    /// Pending state (waiting for beat sync).
    pub pending_state: Option<MusicState>,
    /// Transition rules.
    pub transition_rules: Vec<TransitionRule>,
    /// Default transition type.
    pub default_transition: TransitionType,
    /// Beat tracker.
    pub beat_tracker: BeatTracker,
    /// Active stingers.
    pub stingers: Vec<Stinger>,
    /// Master volume for the layer system.
    pub master_volume: f32,
    /// Whether the system is playing.
    pub playing: bool,
}

impl MusicLayerSystem {
    /// Create a new layered music system.
    pub fn new(bpm: f32) -> Self {
        Self {
            layers: Vec::new(),
            state_configs: HashMap::new(),
            current_state: MusicState::Explore,
            pending_state: None,
            transition_rules: Vec::new(),
            default_transition: TransitionType::Crossfade(1.0),
            beat_tracker: BeatTracker::new(bpm, 4),
            stingers: Vec::new(),
            master_volume: 1.0,
            playing: false,
        }
    }

    /// Add a layer.
    pub fn add_layer(&mut self, layer: LayerInfo) -> usize {
        let idx = self.layers.len();
        self.layers.push(layer);
        idx
    }

    /// Set a layer's target volume with fade.
    pub fn set_layer_volume(&mut self, layer_index: usize, volume: f32, fade_duration: f32) {
        if layer_index < self.layers.len() {
            self.layers[layer_index].set_volume(volume, fade_duration);
        }
    }

    /// Configure layer volumes for a specific game state.
    pub fn configure_state(&mut self, state: MusicState, layer_volumes: Vec<f32>) {
        self.state_configs.insert(state, layer_volumes);
    }

    /// Add a transition rule.
    pub fn add_transition_rule(&mut self, rule: TransitionRule) {
        self.transition_rules.push(rule);
    }

    /// Transition to a new music state.
    pub fn set_state(&mut self, new_state: MusicState) {
        if new_state == self.current_state {
            return;
        }

        // Find transition rule
        let transition = self
            .transition_rules
            .iter()
            .find(|r| r.from_state == self.current_state && r.to_state == new_state)
            .map(|r| r.transition)
            .unwrap_or(self.default_transition);

        match transition {
            TransitionType::SyncToBeat => {
                self.pending_state = Some(new_state);
            }
            _ => {
                self.apply_state_transition(new_state, transition);
            }
        }
    }

    fn apply_state_transition(&mut self, new_state: MusicState, transition: TransitionType) {
        self.current_state = new_state;

        let fade_duration = match transition {
            TransitionType::Immediate => 0.0,
            TransitionType::Crossfade(d) => d,
            TransitionType::FadeOutFadeIn { fade_in, .. } => fade_in,
            TransitionType::SyncToBeat => 0.5,
        };

        // Apply layer volumes for the new state
        if let Some(volumes) = self.state_configs.get(&new_state) {
            for (i, &vol) in volumes.iter().enumerate() {
                if i < self.layers.len() {
                    self.layers[i].set_volume(vol, fade_duration);
                }
            }
        }
    }

    /// Play a stinger.
    pub fn play_stinger(&mut self, stinger: Stinger) {
        let mut s = stinger;
        s.trigger();
        self.stingers.push(s);
    }

    /// Start the layered music system.
    pub fn start(&mut self) {
        self.playing = true;
        self.beat_tracker.reset();
    }

    /// Stop the layered music system.
    pub fn stop(&mut self) {
        self.playing = false;
    }

    /// Update the layered music system. Call once per frame.
    pub fn update(&mut self, dt: f32) {
        if !self.playing {
            return;
        }

        // Update beat tracker
        let beat_hit = self.beat_tracker.update(dt);

        // Check for pending beat-synced transitions
        if beat_hit {
            if let Some(pending) = self.pending_state.take() {
                self.apply_state_transition(pending, TransitionType::Crossfade(0.5));
            }
        }

        // Update layer fades
        for layer in &mut self.layers {
            layer.update(dt);
        }

        // Update stingers
        self.stingers.retain_mut(|s| s.update(dt));
    }

    /// Get the number of layers.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Get a layer by name.
    pub fn get_layer(&self, name: &str) -> Option<&LayerInfo> {
        self.layers.iter().find(|l| l.name == name)
    }

    /// Get the total output volume for a specific layer.
    pub fn layer_output_volume(&self, index: usize) -> f32 {
        if index < self.layers.len() {
            self.layers[index].effective_volume() * self.master_volume
        } else {
            0.0
        }
    }
}

impl Default for MusicLayerSystem {
    fn default() -> Self {
        Self::new(DEFAULT_BPM)
    }
}

// ---------------------------------------------------------------------------
// ECS integration
// ---------------------------------------------------------------------------

/// ECS component for attaching a music system to an entity (typically a global entity).
#[derive(Clone)]
pub struct MusicComponent {
    /// The music player for simple track playback.
    pub player: MusicPlayer,
    /// The layered music system for adaptive music.
    pub layers: MusicLayerSystem,
    /// Whether to use the layered system (true) or simple player (false).
    pub use_layers: bool,
    /// Whether the component is active.
    pub active: bool,
}

impl MusicComponent {
    /// Create a new music component with a simple player.
    pub fn simple() -> Self {
        Self {
            player: MusicPlayer::new(),
            layers: MusicLayerSystem::default(),
            use_layers: false,
            active: true,
        }
    }

    /// Create a new music component with a layered system.
    pub fn layered(bpm: f32) -> Self {
        Self {
            player: MusicPlayer::new(),
            layers: MusicLayerSystem::new(bpm),
            use_layers: true,
            active: true,
        }
    }
}

/// System that updates all music components each frame.
pub struct MusicSystem {
    _placeholder: (),
}

impl Default for MusicSystem {
    fn default() -> Self {
        Self { _placeholder: () }
    }
}

impl MusicSystem {
    /// Create a new music system.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update all music components.
    pub fn update(&self, dt: f32, components: &mut [MusicComponent]) {
        for comp in components.iter_mut() {
            if !comp.active {
                continue;
            }
            if comp.use_layers {
                comp.layers.update(dt);
            } else {
                comp.player.update(dt);
            }
        }
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_music_player_play() {
        let mut player = MusicPlayer::new();
        let track = MusicTrack::new("bgm_01");
        player.play_track(track, 0.0);

        assert!(player.is_playing());
        assert!(player.current_track.is_some());
    }

    #[test]
    fn test_music_player_fade_in() {
        let mut player = MusicPlayer::new();
        let track = MusicTrack::new("bgm_01").with_volume(0.8);
        player.play_track(track, 1.0);

        assert_eq!(player.state, MusicPlaybackState::FadingIn);

        // Update through the fade
        for _ in 0..70 {
            player.update(1.0 / 60.0);
        }

        assert_eq!(player.state, MusicPlaybackState::Playing);
        assert!((player.effective_volume - 0.8).abs() < 0.05);
    }

    #[test]
    fn test_music_player_stop_with_fade() {
        let mut player = MusicPlayer::new();
        player.play_track(MusicTrack::new("bgm"), 0.0);
        player.stop(0.5);

        assert_eq!(player.state, MusicPlaybackState::FadingOut);

        for _ in 0..40 {
            player.update(1.0 / 60.0);
        }

        assert_eq!(player.state, MusicPlaybackState::Stopped);
    }

    #[test]
    fn test_music_player_crossfade() {
        let mut player = MusicPlayer::new();
        player.play_track(MusicTrack::new("track_a"), 0.0);
        player.crossfade_to(MusicTrack::new("track_b"), 1.0);

        assert_eq!(player.state, MusicPlaybackState::CrossfadingOut);

        for _ in 0..70 {
            player.update(1.0 / 60.0);
        }

        assert_eq!(player.state, MusicPlaybackState::Playing);
        assert_eq!(
            player.current_track.as_ref().unwrap().clip_name,
            "track_b"
        );
    }

    #[test]
    fn test_playlist() {
        let mut playlist = Playlist::new();
        playlist.add_track(MusicTrack::new("track_1"));
        playlist.add_track(MusicTrack::new("track_2"));
        playlist.add_track(MusicTrack::new("track_3"));

        assert_eq!(playlist.track_count(), 3);
        assert_eq!(playlist.current_track().unwrap().clip_name, "track_1");

        playlist.next();
        assert_eq!(playlist.current_track().unwrap().clip_name, "track_2");

        playlist.previous();
        assert_eq!(playlist.current_track().unwrap().clip_name, "track_1");
    }

    #[test]
    fn test_playlist_loop() {
        let mut playlist = Playlist::new();
        playlist.loop_playlist = true;
        playlist.add_track(MusicTrack::new("a"));
        playlist.add_track(MusicTrack::new("b"));

        playlist.next(); // b
        playlist.next(); // wraps to a
        assert_eq!(playlist.current_track().unwrap().clip_name, "a");
    }

    #[test]
    fn test_beat_tracker() {
        let mut tracker = BeatTracker::new(120.0, 4);

        // At 120 BPM, beat duration = 0.5s
        assert!((tracker.beat_duration - 0.5).abs() < 0.01);

        let mut beat_count = 0;
        for _ in 0..120 {
            if tracker.update(1.0 / 60.0) {
                beat_count += 1;
            }
        }

        // 2 seconds at 120 BPM = 4 beats
        assert!(
            beat_count >= 3 && beat_count <= 5,
            "Expected ~4 beats, got {}",
            beat_count
        );
    }

    #[test]
    fn test_stinger() {
        let mut stinger = Stinger::new("stinger_victory", 1.0);
        stinger.duration = 0.5;
        stinger.trigger();

        assert!(stinger.playing);

        for _ in 0..35 {
            stinger.update(1.0 / 60.0);
        }

        assert!(!stinger.playing);
    }

    #[test]
    fn test_music_layer_system() {
        let mut system = MusicLayerSystem::new(120.0);
        system.add_layer(LayerInfo::new("drums", "drums_clip"));
        system.add_layer(LayerInfo::new("bass", "bass_clip"));
        system.add_layer(LayerInfo::new("melody", "melody_clip"));

        system.configure_state(MusicState::Explore, vec![0.5, 0.3, 0.8]);
        system.configure_state(MusicState::Combat, vec![1.0, 1.0, 0.5]);

        system.start();
        system.set_state(MusicState::Explore);

        for _ in 0..60 {
            system.update(1.0 / 60.0);
        }

        // Drums should approach 0.5
        assert!(
            (system.layers[0].volume - 0.5).abs() < 0.1,
            "Drums volume: {}",
            system.layers[0].volume
        );

        // Transition to combat
        system.set_state(MusicState::Combat);
        for _ in 0..120 {
            system.update(1.0 / 60.0);
        }

        assert!(
            (system.layers[0].volume - 1.0).abs() < 0.1,
            "Drums should be at 1.0 in combat: {}",
            system.layers[0].volume
        );
    }

    #[test]
    fn test_beat_sync_transition() {
        let mut system = MusicLayerSystem::new(120.0);
        system.add_layer(LayerInfo::new("drums", "d"));
        system.configure_state(MusicState::Explore, vec![0.5]);
        system.configure_state(MusicState::Combat, vec![1.0]);

        system.add_transition_rule(TransitionRule {
            from_state: MusicState::Explore,
            to_state: MusicState::Combat,
            transition: TransitionType::SyncToBeat,
        });

        system.start();
        system.set_state(MusicState::Explore);
        system.set_state(MusicState::Combat); // Should be pending

        assert!(system.pending_state.is_some());

        // Simulate until a beat hits
        for _ in 0..60 {
            system.update(1.0 / 60.0);
        }

        // Should have transitioned after beat
        assert_eq!(system.current_state, MusicState::Combat);
        assert!(system.pending_state.is_none());
    }

    #[test]
    fn test_music_component() {
        let comp = MusicComponent::simple();
        assert!(comp.active);
        assert!(!comp.use_layers);
    }

    #[test]
    fn test_music_system_update() {
        let system = MusicSystem::new();
        let mut components = vec![MusicComponent::simple()];
        components[0].player.play_track(MusicTrack::new("test"), 0.0);

        system.update(1.0 / 60.0, &mut components);
        // Should not panic
    }

    #[test]
    fn test_beat_tracker_time_to_next() {
        let tracker = BeatTracker::new(120.0, 4);
        let time = tracker.time_to_next_beat();
        assert!(time > 0.0);
        assert!(time <= 0.5);
    }
}
