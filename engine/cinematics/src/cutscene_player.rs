//! Cutscene Playback System
//!
//! Provides a complete cutscene player that manages camera, animation, audio,
//! and subtitle tracks simultaneously during non-interactive cinematic
//! sequences.
//!
//! Features:
//! - Load and play cutscene assets
//! - Manage multiple synchronized tracks (camera, animation, audio, subtitles)
//! - Skip, pause, and seek within cutscenes
//! - Event triggers at specific timestamps
//! - Fade-in/fade-out transitions
//! - Letterbox aspect ratio control
//! - Cutscene chaining (play sequences back-to-back)
//!
//! # Architecture
//!
//! ```text
//! CutscenePlayer
//!   +-- CutsceneAssetHandle  (loaded cutscene data)
//!   +-- TrackPlayer[]        (per-track playback state)
//!   +-- EventTimeline        (sorted event triggers)
//!   +-- TransitionState      (fade-in/out)
//!   +-- LetterboxState       (aspect ratio bars)
//! ```

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// CutsceneId
// ---------------------------------------------------------------------------

/// Unique identifier for a cutscene asset.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CutsceneId(pub String);

impl CutsceneId {
    /// Creates a new cutscene ID from a string.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl fmt::Display for CutsceneId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cutscene({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// TrackType
// ---------------------------------------------------------------------------

/// The type of track in a cutscene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrackType {
    /// Camera position/rotation track.
    Camera,
    /// Skeletal or property animation track.
    Animation,
    /// Audio/music track.
    Audio,
    /// Subtitle/dialogue track.
    Subtitle,
    /// Event trigger track.
    Event,
    /// Particle/VFX track.
    VFX,
    /// Light intensity/color track.
    Lighting,
    /// Custom gameplay track.
    Custom,
}

impl fmt::Display for TrackType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Camera => write!(f, "Camera"),
            Self::Animation => write!(f, "Animation"),
            Self::Audio => write!(f, "Audio"),
            Self::Subtitle => write!(f, "Subtitle"),
            Self::Event => write!(f, "Event"),
            Self::VFX => write!(f, "VFX"),
            Self::Lighting => write!(f, "Lighting"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

// ---------------------------------------------------------------------------
// CutsceneTrack
// ---------------------------------------------------------------------------

/// A single track within a cutscene, containing typed keyframe data.
#[derive(Debug, Clone)]
pub struct CutsceneTrack {
    /// Human-readable name.
    pub name: String,
    /// Track type.
    pub track_type: TrackType,
    /// Whether this track is enabled.
    pub enabled: bool,
    /// Whether this track should loop.
    pub looping: bool,
    /// Time offset applied to all keyframes in this track.
    pub time_offset: f64,
    /// Playback rate multiplier for this track (1.0 = normal).
    pub playback_rate: f64,
    /// Keyframes sorted by time.
    pub keyframes: Vec<CutsceneKeyframe>,
    /// Target entity or object identifier.
    pub target: Option<String>,
    /// Weight/blend factor for this track (0.0 to 1.0).
    pub weight: f32,
    /// Mute flag (track is skipped during playback).
    pub muted: bool,
}

impl CutsceneTrack {
    /// Creates a new cutscene track.
    pub fn new(name: impl Into<String>, track_type: TrackType) -> Self {
        Self {
            name: name.into(),
            track_type,
            enabled: true,
            looping: false,
            time_offset: 0.0,
            playback_rate: 1.0,
            keyframes: Vec::new(),
            target: None,
            weight: 1.0,
            muted: false,
        }
    }

    /// Sets the target entity.
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Adds a keyframe.
    pub fn add_keyframe(&mut self, keyframe: CutsceneKeyframe) {
        let time = keyframe.time;
        self.keyframes.push(keyframe);
        self.keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    }

    /// Returns the duration of this track (time of last keyframe).
    pub fn duration(&self) -> f64 {
        self.keyframes
            .last()
            .map(|k| k.time + k.duration.unwrap_or(0.0))
            .unwrap_or(0.0)
    }

    /// Finds the keyframe(s) surrounding the given time for interpolation.
    pub fn find_keyframes_at(&self, time: f64) -> (Option<&CutsceneKeyframe>, Option<&CutsceneKeyframe>) {
        let adjusted_time = (time - self.time_offset) * self.playback_rate;

        let mut prev: Option<&CutsceneKeyframe> = None;
        let mut next: Option<&CutsceneKeyframe> = None;

        for kf in &self.keyframes {
            if kf.time <= adjusted_time {
                prev = Some(kf);
            }
            if kf.time > adjusted_time && next.is_none() {
                next = Some(kf);
            }
        }

        (prev, next)
    }

    /// Returns the number of keyframes.
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }
}

// ---------------------------------------------------------------------------
// CutsceneKeyframe
// ---------------------------------------------------------------------------

/// A single keyframe in a cutscene track.
#[derive(Debug, Clone)]
pub struct CutsceneKeyframe {
    /// Time of this keyframe in seconds.
    pub time: f64,
    /// Optional duration for sustained keyframes (e.g., audio clips, subtitles).
    pub duration: Option<f64>,
    /// The keyframe data.
    pub data: KeyframeData,
    /// Easing function for interpolation to the next keyframe.
    pub easing: EasingType,
}

impl CutsceneKeyframe {
    /// Creates a new keyframe at the given time.
    pub fn new(time: f64, data: KeyframeData) -> Self {
        Self {
            time,
            duration: None,
            data,
            easing: EasingType::Linear,
        }
    }

    /// Sets the duration.
    pub fn with_duration(mut self, duration: f64) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Sets the easing function.
    pub fn with_easing(mut self, easing: EasingType) -> Self {
        self.easing = easing;
        self
    }

    /// Returns the end time (time + duration, or just time if no duration).
    pub fn end_time(&self) -> f64 {
        self.time + self.duration.unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// KeyframeData
// ---------------------------------------------------------------------------

/// Data associated with a cutscene keyframe.
#[derive(Debug, Clone)]
pub enum KeyframeData {
    /// Camera position and rotation.
    Camera {
        position: [f32; 3],
        rotation: [f32; 4],
        fov: f32,
    },
    /// Animation playback control.
    Animation {
        /// Animation clip asset ID.
        clip: String,
        /// Playback speed.
        speed: f32,
        /// Blend weight.
        weight: f32,
    },
    /// Audio playback control.
    Audio {
        /// Audio asset ID.
        asset: String,
        /// Volume (0.0 to 1.0).
        volume: f32,
        /// Pitch multiplier.
        pitch: f32,
        /// Whether this is a 3D positional sound.
        spatial: bool,
        /// 3D position for spatial audio.
        position: Option<[f32; 3]>,
    },
    /// Subtitle display.
    Subtitle {
        /// The subtitle text.
        text: String,
        /// Speaker name.
        speaker: Option<String>,
        /// Speaker color.
        speaker_color: Option<[f32; 4]>,
        /// Localization key (if using localized text).
        localization_key: Option<String>,
    },
    /// Event trigger.
    Event {
        /// Event name.
        name: String,
        /// Event parameters.
        parameters: HashMap<String, String>,
    },
    /// VFX spawn.
    VFX {
        /// VFX asset ID.
        asset: String,
        /// Spawn position.
        position: [f32; 3],
        /// Spawn rotation.
        rotation: [f32; 4],
        /// Scale.
        scale: f32,
    },
    /// Light property change.
    Light {
        /// Target light entity.
        target: String,
        /// Light color.
        color: [f32; 3],
        /// Light intensity.
        intensity: f32,
    },
    /// Custom data as key-value pairs.
    Custom {
        /// Custom type identifier.
        type_name: String,
        /// Key-value parameters.
        parameters: HashMap<String, String>,
    },
}

// ---------------------------------------------------------------------------
// EasingType
// ---------------------------------------------------------------------------

/// Easing function types for keyframe interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EasingType {
    /// Linear interpolation.
    Linear,
    /// Ease in (slow start, fast end).
    EaseIn,
    /// Ease out (fast start, slow end).
    EaseOut,
    /// Ease in and out (slow start and end).
    EaseInOut,
    /// Cubic bezier ease in.
    CubicIn,
    /// Cubic bezier ease out.
    CubicOut,
    /// Cubic bezier ease in and out.
    CubicInOut,
    /// Step function (instant change at keyframe time).
    Step,
    /// Hold the previous value until the next keyframe.
    Hold,
}

impl Default for EasingType {
    fn default() -> Self {
        Self::Linear
    }
}

impl EasingType {
    /// Evaluates the easing function for a value `t` in [0, 1].
    pub fn evaluate(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseIn => t * t,
            Self::EaseOut => t * (2.0 - t),
            Self::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    -1.0 + (4.0 - 2.0 * t) * t
                }
            }
            Self::CubicIn => t * t * t,
            Self::CubicOut => {
                let t1 = t - 1.0;
                t1 * t1 * t1 + 1.0
            }
            Self::CubicInOut => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    let t1 = 2.0 * t - 2.0;
                    0.5 * t1 * t1 * t1 + 1.0
                }
            }
            Self::Step => {
                if t >= 1.0 { 1.0 } else { 0.0 }
            }
            Self::Hold => 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// CutsceneEvent
// ---------------------------------------------------------------------------

/// An event triggered during cutscene playback.
#[derive(Debug, Clone)]
pub struct CutsceneEvent {
    /// Time at which the event fires.
    pub time: f64,
    /// Event name.
    pub name: String,
    /// Event parameters.
    pub parameters: HashMap<String, String>,
    /// Whether this event has been fired during current playback.
    pub fired: bool,
}

impl CutsceneEvent {
    /// Creates a new cutscene event.
    pub fn new(time: f64, name: impl Into<String>) -> Self {
        Self {
            time,
            name: name.into(),
            parameters: HashMap::new(),
            fired: false,
        }
    }

    /// Adds a parameter.
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// TransitionState
// ---------------------------------------------------------------------------

/// State for fade-in/fade-out transitions during cutscenes.
#[derive(Debug, Clone)]
pub struct TransitionState {
    /// Whether a transition is currently active.
    pub active: bool,
    /// The type of transition.
    pub transition_type: TransitionType,
    /// Duration of the transition in seconds.
    pub duration: f64,
    /// Elapsed time in the current transition.
    pub elapsed: f64,
    /// The transition color (usually black).
    pub color: [f32; 4],
    /// Current opacity of the transition overlay.
    pub opacity: f32,
    /// Easing function for the transition.
    pub easing: EasingType,
}

/// Type of screen transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionType {
    /// Screen fades from black to clear.
    FadeIn,
    /// Screen fades from clear to black.
    FadeOut,
    /// Screen fades from one color to another.
    CrossFade,
    /// Instant cut (no transition).
    Cut,
    /// Wipe transition from left to right.
    WipeLeft,
    /// Wipe transition from right to left.
    WipeRight,
}

impl Default for TransitionType {
    fn default() -> Self {
        Self::FadeIn
    }
}

impl TransitionState {
    /// Creates a new inactive transition state.
    pub fn new() -> Self {
        Self {
            active: false,
            transition_type: TransitionType::FadeIn,
            duration: 1.0,
            elapsed: 0.0,
            color: [0.0, 0.0, 0.0, 1.0],
            opacity: 0.0,
            easing: EasingType::EaseInOut,
        }
    }

    /// Starts a fade-in transition.
    pub fn start_fade_in(&mut self, duration: f64) {
        self.active = true;
        self.transition_type = TransitionType::FadeIn;
        self.duration = duration;
        self.elapsed = 0.0;
        self.opacity = 1.0;
    }

    /// Starts a fade-out transition.
    pub fn start_fade_out(&mut self, duration: f64) {
        self.active = true;
        self.transition_type = TransitionType::FadeOut;
        self.duration = duration;
        self.elapsed = 0.0;
        self.opacity = 0.0;
    }

    /// Updates the transition state.
    pub fn update(&mut self, dt: f64) {
        if !self.active {
            return;
        }

        self.elapsed += dt;
        let t = (self.elapsed / self.duration).clamp(0.0, 1.0);
        let eased_t = self.easing.evaluate(t) as f32;

        match self.transition_type {
            TransitionType::FadeIn => {
                self.opacity = 1.0 - eased_t;
            }
            TransitionType::FadeOut => {
                self.opacity = eased_t;
            }
            TransitionType::CrossFade => {
                self.opacity = eased_t;
            }
            TransitionType::Cut => {
                self.opacity = if t >= 1.0 { 1.0 } else { 0.0 };
            }
            TransitionType::WipeLeft | TransitionType::WipeRight => {
                self.opacity = eased_t;
            }
        }

        if self.elapsed >= self.duration {
            self.active = false;
        }
    }

    /// Returns `true` if the transition is complete.
    pub fn is_complete(&self) -> bool {
        !self.active
    }

    /// Returns the current transition progress (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        if self.duration <= 0.0 {
            return 1.0;
        }
        (self.elapsed / self.duration).clamp(0.0, 1.0)
    }
}

impl Default for TransitionState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// LetterboxState
// ---------------------------------------------------------------------------

/// State for letterbox (cinematic bars) during cutscenes.
#[derive(Debug, Clone)]
pub struct LetterboxState {
    /// Whether letterboxing is active.
    pub active: bool,
    /// Target aspect ratio (e.g., 2.39 for cinema scope).
    pub target_aspect_ratio: f32,
    /// Current bar height (top and bottom) as a fraction of screen height.
    pub bar_height: f32,
    /// Target bar height.
    pub target_bar_height: f32,
    /// Animation speed (fraction per second).
    pub animation_speed: f32,
    /// Bar color (usually black).
    pub color: [f32; 4],
}

impl LetterboxState {
    /// Creates a new letterbox state.
    pub fn new() -> Self {
        Self {
            active: false,
            target_aspect_ratio: 2.39,
            bar_height: 0.0,
            target_bar_height: 0.0,
            animation_speed: 2.0,
            color: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Enables letterboxing with the given aspect ratio.
    pub fn enable(&mut self, aspect_ratio: f32, screen_aspect: f32) {
        self.active = true;
        self.target_aspect_ratio = aspect_ratio;

        if aspect_ratio > screen_aspect {
            // Need horizontal bars (top/bottom).
            let visible_height = screen_aspect / aspect_ratio;
            self.target_bar_height = (1.0 - visible_height) / 2.0;
        } else {
            self.target_bar_height = 0.0;
        }
    }

    /// Disables letterboxing (animates bars away).
    pub fn disable(&mut self) {
        self.active = false;
        self.target_bar_height = 0.0;
    }

    /// Updates the letterbox animation.
    pub fn update(&mut self, dt: f32) {
        let diff = self.target_bar_height - self.bar_height;
        if diff.abs() < 0.001 {
            self.bar_height = self.target_bar_height;
            return;
        }
        let speed = self.animation_speed * dt;
        if diff > 0.0 {
            self.bar_height = (self.bar_height + speed).min(self.target_bar_height);
        } else {
            self.bar_height = (self.bar_height - speed).max(self.target_bar_height);
        }
    }

    /// Returns `true` if the bars are fully animated in or out.
    pub fn is_settled(&self) -> bool {
        (self.bar_height - self.target_bar_height).abs() < 0.001
    }

    /// Returns the height of the top/bottom bars in pixels given screen height.
    pub fn bar_height_pixels(&self, screen_height: f32) -> f32 {
        self.bar_height * screen_height
    }
}

impl Default for LetterboxState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CutscenePlaybackState
// ---------------------------------------------------------------------------

/// The current state of cutscene playback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackState {
    /// No cutscene is loaded or playing.
    Idle,
    /// A cutscene is loading.
    Loading,
    /// Waiting for a transition before playback starts.
    WaitingForTransition,
    /// Cutscene is actively playing.
    Playing,
    /// Cutscene playback is paused.
    Paused,
    /// Cutscene is being skipped.
    Skipping,
    /// Cutscene has finished playing.
    Finished,
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self::Idle
    }
}

// ---------------------------------------------------------------------------
// CutsceneDescriptor
// ---------------------------------------------------------------------------

/// Description of a cutscene asset to be loaded and played.
#[derive(Debug, Clone)]
pub struct CutsceneDescriptor {
    /// Unique identifier.
    pub id: CutsceneId,
    /// Human-readable name.
    pub name: String,
    /// Tracks in this cutscene.
    pub tracks: Vec<CutsceneTrack>,
    /// Event triggers.
    pub events: Vec<CutsceneEvent>,
    /// Whether the cutscene can be skipped by the player.
    pub skippable: bool,
    /// Whether to apply letterboxing.
    pub letterbox: bool,
    /// Letterbox aspect ratio.
    pub letterbox_aspect: f32,
    /// Fade-in duration at the start.
    pub fade_in_duration: f64,
    /// Fade-out duration at the end.
    pub fade_out_duration: f64,
    /// Whether to pause game simulation during the cutscene.
    pub pause_gameplay: bool,
    /// Whether to hide the HUD during the cutscene.
    pub hide_hud: bool,
    /// Optional next cutscene to chain into.
    pub next_cutscene: Option<CutsceneId>,
    /// Total duration (calculated from tracks).
    pub duration: f64,
}

impl CutsceneDescriptor {
    /// Creates a new cutscene descriptor.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: CutsceneId::new(id),
            name: name.into(),
            tracks: Vec::new(),
            events: Vec::new(),
            skippable: true,
            letterbox: true,
            letterbox_aspect: 2.39,
            fade_in_duration: 0.5,
            fade_out_duration: 0.5,
            pause_gameplay: true,
            hide_hud: true,
            next_cutscene: None,
            duration: 0.0,
        }
    }

    /// Adds a track.
    pub fn add_track(&mut self, track: CutsceneTrack) {
        self.tracks.push(track);
        self.recalculate_duration();
    }

    /// Adds an event trigger.
    pub fn add_event(&mut self, event: CutsceneEvent) {
        self.events.push(event);
        self.events.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    }

    /// Sets the next cutscene in the chain.
    pub fn with_next(mut self, next_id: impl Into<String>) -> Self {
        self.next_cutscene = Some(CutsceneId::new(next_id));
        self
    }

    /// Recalculates the total duration from all tracks.
    fn recalculate_duration(&mut self) {
        self.duration = self
            .tracks
            .iter()
            .map(|t| t.duration())
            .fold(0.0f64, f64::max);
    }

    /// Returns the number of tracks.
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Returns the number of events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }
}

// ---------------------------------------------------------------------------
// CutscenePlayer
// ---------------------------------------------------------------------------

/// Plays back cutscenes, managing all tracks, transitions, and events.
///
/// The cutscene player is the main runtime component. It loads cutscene
/// descriptors, drives playback, fires events at the appropriate times,
/// and manages screen transitions and letterboxing.
pub struct CutscenePlayer {
    /// The current cutscene descriptor.
    current_cutscene: Option<CutsceneDescriptor>,
    /// Current playback state.
    state: PlaybackState,
    /// Current playback time in seconds.
    current_time: f64,
    /// Playback rate (1.0 = normal, 0.5 = half speed, 2.0 = double speed).
    playback_rate: f64,
    /// Screen transition state.
    pub transition: TransitionState,
    /// Letterbox state.
    pub letterbox: LetterboxState,
    /// Events that have been fired and not yet consumed.
    pending_fired_events: Vec<CutsceneEvent>,
    /// Active subtitle text (if any).
    active_subtitles: Vec<ActiveSubtitle>,
    /// Queue of cutscenes to play after the current one finishes.
    queue: Vec<CutsceneDescriptor>,
    /// Whether the player is waiting for user input to skip.
    skip_requested: bool,
    /// Callback events to notify the game about cutscene state changes.
    state_change_events: Vec<CutsceneStateEvent>,
    /// Screen aspect ratio for letterbox calculations.
    screen_aspect_ratio: f32,
}

/// An active subtitle being displayed.
#[derive(Debug, Clone)]
pub struct ActiveSubtitle {
    /// The subtitle text.
    pub text: String,
    /// Speaker name.
    pub speaker: Option<String>,
    /// Speaker color.
    pub speaker_color: Option<[f32; 4]>,
    /// Time when this subtitle started.
    pub start_time: f64,
    /// Duration of this subtitle.
    pub duration: f64,
    /// Whether this subtitle has been dismissed.
    pub dismissed: bool,
}

/// Events emitted by the cutscene player.
#[derive(Debug, Clone)]
pub enum CutsceneStateEvent {
    /// A cutscene has started playing.
    Started { id: CutsceneId },
    /// A cutscene has been paused.
    Paused { id: CutsceneId, time: f64 },
    /// A cutscene has been resumed.
    Resumed { id: CutsceneId, time: f64 },
    /// A cutscene has been skipped.
    Skipped { id: CutsceneId, time: f64 },
    /// A cutscene has finished naturally.
    Finished { id: CutsceneId },
    /// An event trigger fired.
    EventTriggered { event_name: String, parameters: HashMap<String, String> },
    /// A subtitle should be displayed.
    SubtitleStarted { text: String, speaker: Option<String> },
    /// A subtitle should be hidden.
    SubtitleEnded { text: String },
}

impl CutscenePlayer {
    /// Creates a new cutscene player.
    pub fn new() -> Self {
        Self {
            current_cutscene: None,
            state: PlaybackState::Idle,
            current_time: 0.0,
            playback_rate: 1.0,
            transition: TransitionState::new(),
            letterbox: LetterboxState::new(),
            pending_fired_events: Vec::new(),
            active_subtitles: Vec::new(),
            queue: Vec::new(),
            skip_requested: false,
            state_change_events: Vec::new(),
            screen_aspect_ratio: 16.0 / 9.0,
        }
    }

    /// Sets the screen aspect ratio for letterbox calculations.
    pub fn set_screen_aspect_ratio(&mut self, ratio: f32) {
        self.screen_aspect_ratio = ratio;
    }

    /// Loads and starts playing a cutscene.
    pub fn play(&mut self, descriptor: CutsceneDescriptor) {
        let id = descriptor.id.clone();
        let fade_in = descriptor.fade_in_duration;
        let do_letterbox = descriptor.letterbox;
        let letterbox_aspect = descriptor.letterbox_aspect;

        self.current_cutscene = Some(descriptor);
        self.current_time = 0.0;
        self.skip_requested = false;
        self.pending_fired_events.clear();
        self.active_subtitles.clear();

        // Start fade-in transition.
        if fade_in > 0.0 {
            self.transition.start_fade_in(fade_in);
            self.state = PlaybackState::WaitingForTransition;
        } else {
            self.state = PlaybackState::Playing;
        }

        // Enable letterboxing.
        if do_letterbox {
            self.letterbox.enable(letterbox_aspect, self.screen_aspect_ratio);
        }

        self.state_change_events
            .push(CutsceneStateEvent::Started { id });
    }

    /// Queues a cutscene to play after the current one finishes.
    pub fn queue_cutscene(&mut self, descriptor: CutsceneDescriptor) {
        self.queue.push(descriptor);
    }

    /// Pauses the current cutscene.
    pub fn pause(&mut self) {
        if self.state == PlaybackState::Playing {
            self.state = PlaybackState::Paused;
            if let Some(ref cutscene) = self.current_cutscene {
                self.state_change_events.push(CutsceneStateEvent::Paused {
                    id: cutscene.id.clone(),
                    time: self.current_time,
                });
            }
        }
    }

    /// Resumes the current cutscene.
    pub fn resume(&mut self) {
        if self.state == PlaybackState::Paused {
            self.state = PlaybackState::Playing;
            if let Some(ref cutscene) = self.current_cutscene {
                self.state_change_events.push(CutsceneStateEvent::Resumed {
                    id: cutscene.id.clone(),
                    time: self.current_time,
                });
            }
        }
    }

    /// Requests the cutscene to be skipped.
    pub fn skip(&mut self) {
        let skippable = self
            .current_cutscene
            .as_ref()
            .map_or(false, |c| c.skippable);
        if !skippable {
            return;
        }

        if self.state == PlaybackState::Playing || self.state == PlaybackState::Paused {
            self.skip_requested = true;
            self.state = PlaybackState::Skipping;

            if let Some(ref cutscene) = self.current_cutscene {
                let fade_out = cutscene.fade_out_duration;
                self.state_change_events.push(CutsceneStateEvent::Skipped {
                    id: cutscene.id.clone(),
                    time: self.current_time,
                });
                if fade_out > 0.0 {
                    self.transition.start_fade_out(fade_out);
                }
            }
        }
    }

    /// Seeks to a specific time in the cutscene.
    pub fn seek(&mut self, time: f64) {
        if self.current_cutscene.is_none() {
            return;
        }
        let duration = self.current_cutscene.as_ref().unwrap().duration;
        self.current_time = time.clamp(0.0, duration);

        // Reset event fired flags for events after the seek position.
        if let Some(ref mut cutscene) = self.current_cutscene {
            for event in &mut cutscene.events {
                if event.time > self.current_time {
                    event.fired = false;
                }
            }
        }
    }

    /// Sets the playback rate.
    pub fn set_playback_rate(&mut self, rate: f64) {
        self.playback_rate = rate.max(0.0);
    }

    /// Returns the current playback state.
    pub fn state(&self) -> PlaybackState {
        self.state
    }

    /// Returns the current playback time in seconds.
    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Returns the total duration of the current cutscene.
    pub fn duration(&self) -> f64 {
        self.current_cutscene.as_ref().map_or(0.0, |c| c.duration)
    }

    /// Returns the playback progress as a fraction (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        let duration = self.duration();
        if duration <= 0.0 {
            return 0.0;
        }
        (self.current_time / duration).clamp(0.0, 1.0)
    }

    /// Returns `true` if a cutscene is currently playing or paused.
    pub fn is_active(&self) -> bool {
        matches!(
            self.state,
            PlaybackState::Playing
                | PlaybackState::Paused
                | PlaybackState::WaitingForTransition
                | PlaybackState::Skipping
        )
    }

    /// Returns the active subtitles.
    pub fn active_subtitles(&self) -> &[ActiveSubtitle] {
        &self.active_subtitles
    }

    /// Takes and returns all pending state change events.
    pub fn drain_events(&mut self) -> Vec<CutsceneStateEvent> {
        std::mem::take(&mut self.state_change_events)
    }

    /// Returns a reference to the current cutscene descriptor, if any.
    pub fn current_cutscene(&self) -> Option<&CutsceneDescriptor> {
        self.current_cutscene.as_ref()
    }

    /// Updates the cutscene player (call once per frame).
    pub fn update(&mut self, dt: f64) {
        // Update transition.
        self.transition.update(dt);

        // Update letterbox.
        self.letterbox.update(dt as f32);

        match self.state {
            PlaybackState::Idle | PlaybackState::Loading | PlaybackState::Finished => {
                return;
            }
            PlaybackState::WaitingForTransition => {
                if self.transition.is_complete() {
                    self.state = PlaybackState::Playing;
                }
                return;
            }
            PlaybackState::Paused => {
                return;
            }
            PlaybackState::Skipping => {
                if self.transition.is_complete() || !self.transition.active {
                    self.finish_cutscene();
                }
                return;
            }
            PlaybackState::Playing => {}
        }

        // Advance time.
        let prev_time = self.current_time;
        self.current_time += dt * self.playback_rate;

        // Fire events.
        self.fire_events(prev_time, self.current_time);

        // Update subtitles.
        self.update_subtitles();

        // Check for end of cutscene.
        let duration = self.current_cutscene.as_ref().map_or(0.0, |c| c.duration);
        if self.current_time >= duration {
            let fade_out = self
                .current_cutscene
                .as_ref()
                .map_or(0.0, |c| c.fade_out_duration);
            if fade_out > 0.0 && !self.transition.active {
                self.transition.start_fade_out(fade_out);
                self.state = PlaybackState::Skipping; // Reuse skipping state for end fade.
            } else if !self.transition.active {
                self.finish_cutscene();
            }
        }
    }

    /// Fires events that fall within the time window.
    fn fire_events(&mut self, from_time: f64, to_time: f64) {
        if let Some(ref mut cutscene) = self.current_cutscene {
            for event in &mut cutscene.events {
                if !event.fired && event.time >= from_time && event.time < to_time {
                    event.fired = true;
                    self.state_change_events
                        .push(CutsceneStateEvent::EventTriggered {
                            event_name: event.name.clone(),
                            parameters: event.parameters.clone(),
                        });
                    self.pending_fired_events.push(event.clone());
                }
            }
        }

        // Check subtitle tracks for new subtitles.
        if let Some(ref cutscene) = self.current_cutscene {
            for track in &cutscene.tracks {
                if track.track_type != TrackType::Subtitle || track.muted || !track.enabled {
                    continue;
                }
                for kf in &track.keyframes {
                    if kf.time >= from_time && kf.time < to_time {
                        if let KeyframeData::Subtitle { ref text, ref speaker, ref speaker_color, .. } = kf.data {
                            let subtitle = ActiveSubtitle {
                                text: text.clone(),
                                speaker: speaker.clone(),
                                speaker_color: *speaker_color,
                                start_time: kf.time,
                                duration: kf.duration.unwrap_or(3.0),
                                dismissed: false,
                            };
                            self.state_change_events
                                .push(CutsceneStateEvent::SubtitleStarted {
                                    text: text.clone(),
                                    speaker: speaker.clone(),
                                });
                            self.active_subtitles.push(subtitle);
                        }
                    }
                }
            }
        }
    }

    /// Updates active subtitles, removing expired ones.
    fn update_subtitles(&mut self) {
        let current = self.current_time;
        self.active_subtitles.retain(|sub| {
            if sub.dismissed {
                return false;
            }
            let expired = current > sub.start_time + sub.duration;
            if expired {
                // The subtitle ended event would be pushed here in a full implementation.
            }
            !expired
        });
    }

    /// Finishes the current cutscene and potentially starts the next one.
    fn finish_cutscene(&mut self) {
        let id = self
            .current_cutscene
            .as_ref()
            .map(|c| c.id.clone())
            .unwrap_or_else(|| CutsceneId::new("unknown"));

        self.state_change_events
            .push(CutsceneStateEvent::Finished { id });

        // Disable letterboxing.
        self.letterbox.disable();

        // Check for chained or queued cutscene.
        let next = self
            .current_cutscene
            .as_ref()
            .and_then(|c| c.next_cutscene.clone());

        self.current_cutscene = None;
        self.active_subtitles.clear();

        // Play queued cutscene.
        if !self.queue.is_empty() {
            let next_cutscene = self.queue.remove(0);
            self.play(next_cutscene);
        } else {
            self.state = PlaybackState::Finished;
        }
    }
}

impl Default for CutscenePlayer {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for CutscenePlayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CutscenePlayer")
            .field("state", &self.state)
            .field("current_time", &self.current_time)
            .field("playback_rate", &self.playback_rate)
            .field("has_cutscene", &self.current_cutscene.is_some())
            .field("queue_size", &self.queue.len())
            .field("active_subtitles", &self.active_subtitles.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_cutscene() -> CutsceneDescriptor {
        let mut cutscene = CutsceneDescriptor::new("test_cutscene", "Test Cutscene");
        cutscene.fade_in_duration = 0.0;
        cutscene.fade_out_duration = 0.0;

        let mut camera_track = CutsceneTrack::new("Main Camera", TrackType::Camera);
        camera_track.add_keyframe(CutsceneKeyframe::new(0.0, KeyframeData::Camera {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            fov: 60.0,
        }));
        camera_track.add_keyframe(CutsceneKeyframe::new(5.0, KeyframeData::Camera {
            position: [10.0, 5.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            fov: 75.0,
        }));
        cutscene.add_track(camera_track);

        let mut subtitle_track = CutsceneTrack::new("Subtitles", TrackType::Subtitle);
        subtitle_track.add_keyframe(
            CutsceneKeyframe::new(1.0, KeyframeData::Subtitle {
                text: "Hello, welcome to the demo.".to_string(),
                speaker: Some("Narrator".to_string()),
                speaker_color: Some([1.0, 1.0, 1.0, 1.0]),
                localization_key: None,
            })
            .with_duration(3.0),
        );
        cutscene.add_track(subtitle_track);

        cutscene.add_event(CutsceneEvent::new(2.5, "explosion"));

        cutscene
    }

    #[test]
    fn test_cutscene_descriptor() {
        let cutscene = make_simple_cutscene();
        assert_eq!(cutscene.track_count(), 2);
        assert_eq!(cutscene.event_count(), 1);
        assert!(cutscene.duration > 0.0);
    }

    #[test]
    fn test_play_cutscene() {
        let mut player = CutscenePlayer::new();
        let cutscene = make_simple_cutscene();
        player.play(cutscene);
        assert!(player.is_active());
        assert_eq!(player.state(), PlaybackState::Playing);
    }

    #[test]
    fn test_pause_resume() {
        let mut player = CutscenePlayer::new();
        player.play(make_simple_cutscene());
        player.pause();
        assert_eq!(player.state(), PlaybackState::Paused);
        player.resume();
        assert_eq!(player.state(), PlaybackState::Playing);
    }

    #[test]
    fn test_skip() {
        let mut player = CutscenePlayer::new();
        player.play(make_simple_cutscene());
        player.skip();
        assert_eq!(player.state(), PlaybackState::Skipping);
    }

    #[test]
    fn test_update_advances_time() {
        let mut player = CutscenePlayer::new();
        player.play(make_simple_cutscene());
        player.update(1.0);
        assert!(player.current_time() > 0.0);
    }

    #[test]
    fn test_easing_linear() {
        assert!((EasingType::Linear.evaluate(0.5) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_easing_ease_in() {
        assert!((EasingType::EaseIn.evaluate(0.0) - 0.0).abs() < 0.001);
        assert!((EasingType::EaseIn.evaluate(1.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_transition_fade_in() {
        let mut transition = TransitionState::new();
        transition.start_fade_in(1.0);
        assert!(transition.active);
        transition.update(0.5);
        assert!(transition.opacity < 1.0);
        transition.update(0.5);
        assert!(!transition.active);
    }

    #[test]
    fn test_letterbox() {
        let mut lb = LetterboxState::new();
        lb.enable(2.39, 16.0 / 9.0);
        assert!(lb.target_bar_height > 0.0);
        for _ in 0..100 {
            lb.update(0.016);
        }
        assert!(lb.is_settled());
    }

    #[test]
    fn test_event_firing() {
        let mut player = CutscenePlayer::new();
        player.play(make_simple_cutscene());
        // Advance past the event at t=2.5
        player.update(3.0);
        let events = player.drain_events();
        assert!(events.iter().any(|e| matches!(e, CutsceneStateEvent::EventTriggered { .. })));
    }

    #[test]
    fn test_track_find_keyframes() {
        let mut track = CutsceneTrack::new("test", TrackType::Camera);
        track.add_keyframe(CutsceneKeyframe::new(0.0, KeyframeData::Camera {
            position: [0.0; 3], rotation: [0.0, 0.0, 0.0, 1.0], fov: 60.0,
        }));
        track.add_keyframe(CutsceneKeyframe::new(5.0, KeyframeData::Camera {
            position: [10.0; 3], rotation: [0.0, 0.0, 0.0, 1.0], fov: 60.0,
        }));
        let (prev, next) = track.find_keyframes_at(2.5);
        assert!(prev.is_some());
        assert!(next.is_some());
    }
}
