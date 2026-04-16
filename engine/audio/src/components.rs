//! ECS components and systems for audio integration.
//!
//! Bridges the audio mixer and spatial audio processing with the
//! entity-component-system.  The [`AudioSystem`] runs each frame to
//! synchronise component state with the mixer, update spatial parameters
//! from entity transforms, and drive playback lifecycle.

use glam::Vec3;

use crate::mixer::{AudioBus, AudioMixer, AudioSource, ChannelState, MixerHandle, SoftwareMixer};
use crate::spatial::{AudioListener, OcclusionParams, SpatialAudioSource};

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Component that makes an entity an audio source.
///
/// When attached to an entity, the [`AudioSystem`] will manage playback
/// and (if `spatial` is true) apply 3D spatialization based on the
/// entity's transform.
#[derive(Debug, Clone)]
pub struct AudioSourceComponent {
    /// Handle to the playing voice (set by AudioSystem; `None` if not playing).
    pub handle: Option<MixerHandle>,

    /// The clip asset name/path to play.
    pub clip: String,

    /// Volume multiplier [0.0, 1.0+].
    pub volume: f32,

    /// Pitch multiplier (1.0 = normal).
    pub pitch: f32,

    /// Whether to loop playback.
    pub looping: bool,

    /// Whether to play automatically when the component is added.
    pub play_on_awake: bool,

    /// Whether this source should be spatialized (3D).
    pub spatial: bool,

    /// Audio bus routing.
    pub bus: AudioBus,

    /// Voice priority [0, 255].
    pub priority: u8,

    /// Spatial audio parameters (only used when `spatial` is true).
    pub spatial_params: SpatialAudioSource,

    /// Occlusion parameters.
    pub occlusion: OcclusionParams,

    /// Whether the component state has been modified and needs to be pushed
    /// to the mixer.
    pub dirty: bool,

    /// Desired playback state.
    pub desired_state: PlaybackState,

    /// Whether `play_on_awake` has already been triggered.
    awake_triggered: bool,
}

/// Desired playback state for an audio source component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackState {
    Stopped,
    Playing,
    Paused,
}

impl Default for AudioSourceComponent {
    fn default() -> Self {
        Self {
            handle: None,
            clip: String::new(),
            volume: 1.0,
            pitch: 1.0,
            looping: false,
            play_on_awake: false,
            spatial: false,
            bus: AudioBus::SFX,
            priority: 128,
            spatial_params: SpatialAudioSource::default(),
            occlusion: OcclusionParams::default(),
            dirty: true,
            desired_state: PlaybackState::Stopped,
            awake_triggered: false,
        }
    }
}

impl AudioSourceComponent {
    /// Build an [`AudioSource`] from this component's current state.
    pub fn to_audio_source(&self) -> AudioSource {
        AudioSource {
            clip_name: self.clip.clone(),
            volume: self.volume,
            pitch: self.pitch,
            looping: self.looping,
            spatial: self.spatial,
            bus: self.bus,
            priority: self.priority,
            start_offset: 0.0,
            fade_in: 0.0,
            pan: 0.0,
        }
    }

    /// Request playback to start.
    pub fn play(&mut self) {
        self.desired_state = PlaybackState::Playing;
        self.dirty = true;
    }

    /// Request playback to stop.
    pub fn stop(&mut self) {
        self.desired_state = PlaybackState::Stopped;
        self.dirty = true;
    }

    /// Request playback to pause.
    pub fn pause(&mut self) {
        self.desired_state = PlaybackState::Paused;
        self.dirty = true;
    }

    /// Set volume and mark dirty.
    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume;
        self.dirty = true;
    }

    /// Set pitch and mark dirty.
    pub fn set_pitch(&mut self, pitch: f32) {
        self.pitch = pitch;
        self.dirty = true;
    }

    /// Whether the source is currently playing (has a live handle).
    pub fn is_playing(&self) -> bool {
        self.handle.is_some() && self.desired_state == PlaybackState::Playing
    }

    /// Update the spatial position (called by AudioSystem from entity transform).
    pub fn set_position(&mut self, position: Vec3) {
        self.spatial_params.position = position;
    }

    /// Update the spatial velocity (called by AudioSystem from entity physics).
    pub fn set_velocity(&mut self, velocity: Vec3) {
        self.spatial_params.velocity = velocity;
    }
}

/// Component that marks an entity as the audio listener (the "ears").
///
/// Typically attached to the player camera.  At most one listener should be
/// active at a time.
#[derive(Debug, Clone)]
pub struct AudioListenerComponent {
    /// The listener state (position, orientation, etc.).
    pub listener: AudioListener,

    /// Whether this listener is the active listener.
    pub active: bool,
}

impl Default for AudioListenerComponent {
    fn default() -> Self {
        Self {
            listener: AudioListener::default(),
            active: true,
        }
    }
}

impl AudioListenerComponent {
    /// Update listener position from a transform.
    pub fn set_position(&mut self, position: Vec3) {
        self.listener.position = position;
    }

    /// Update listener orientation.
    pub fn set_orientation(&mut self, forward: Vec3, up: Vec3) {
        self.listener.set_orientation(forward, up);
    }

    /// Update listener velocity.
    pub fn set_velocity(&mut self, velocity: Vec3) {
        self.listener.velocity = velocity;
    }
}

// ---------------------------------------------------------------------------
// Audio System
// ---------------------------------------------------------------------------

/// System that drives audio playback, spatial processing, and mixer updates.
///
/// Each frame the system:
/// 1. Handles `play_on_awake` components that haven't been triggered yet.
/// 2. Processes play/stop/pause state transitions from dirty components.
/// 3. Updates spatial parameters (attenuation, pan, Doppler) for spatial sources.
/// 4. Pushes volume/pitch changes to the mixer.
/// 5. Calls `AudioMixer::update` to mix and output audio.
pub struct AudioSystem {
    /// The active audio mixer (concrete type for direct access).
    mixer: Option<SoftwareMixer>,

    /// Master volume [0.0, 1.0].
    pub master_volume: f32,

    /// Whether the audio system is paused (e.g. during menu/pause screen).
    pub paused: bool,

    /// Speed of sound in m/s (used for Doppler calculations).  Default: 343.0.
    pub speed_of_sound: f32,

    /// The current listener snapshot (updated each frame from the active
    /// AudioListenerComponent).
    listener: AudioListener,
}

impl Default for AudioSystem {
    fn default() -> Self {
        Self {
            mixer: None,
            master_volume: 1.0,
            paused: false,
            speed_of_sound: 343.0,
            listener: AudioListener::default(),
        }
    }
}

impl AudioSystem {
    /// Create a new audio system.
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize the audio system with a software mixer.
    pub fn initialize(&mut self, mixer: SoftwareMixer) {
        self.mixer = Some(mixer);
        log::info!("Audio system initialized");
    }

    /// Initialize with a boxed trait object mixer (for use with other mixer
    /// implementations).
    pub fn initialize_boxed(&mut self, mixer: SoftwareMixer) {
        self.mixer = Some(mixer);
        log::info!("Audio system initialized (boxed)");
    }

    /// Update the listener from a listener component.
    pub fn update_listener(&mut self, component: &AudioListenerComponent) {
        if component.active {
            self.listener = component.listener.clone();
        }
    }

    /// Process a single audio source component, managing its lifecycle with
    /// the mixer.
    ///
    /// Call this once per source component per frame.
    pub fn process_source(&mut self, source: &mut AudioSourceComponent) {
        // Snapshot listener data before borrowing mixer mutably to avoid
        // conflicting borrows on `self`.
        let listener_snapshot = self.listener.clone();
        let speed_of_sound = self.speed_of_sound;

        let mixer = match self.mixer.as_mut() {
            Some(m) => m,
            None => return,
        };

        // Handle play_on_awake.
        if source.play_on_awake && !source.awake_triggered {
            source.awake_triggered = true;
            source.desired_state = PlaybackState::Playing;
            source.dirty = true;
        }

        // If the handle refers to a stopped/finished voice, clear it.
        if let Some(handle) = source.handle {
            match mixer.channel_state(handle) {
                Ok(ChannelState::Stopped) | Err(_) => {
                    source.handle = None;
                    if source.desired_state == PlaybackState::Playing && !source.looping {
                        source.desired_state = PlaybackState::Stopped;
                    }
                }
                _ => {}
            }
        }

        if !source.dirty {
            // Even if not dirty, still update spatial params for active spatial sources.
            if source.spatial {
                if let Some(handle) = source.handle {
                    Self::apply_spatial_to_mixer(
                        mixer,
                        &listener_snapshot,
                        speed_of_sound,
                        handle,
                        &source.spatial_params,
                        &source.occlusion,
                    );
                }
            }
            return;
        }
        source.dirty = false;

        match source.desired_state {
            PlaybackState::Playing => {
                if source.handle.is_none() {
                    // Start playback.
                    let audio_source = source.to_audio_source();
                    match mixer.play(audio_source) {
                        Ok(handle) => {
                            source.handle = Some(handle);
                        }
                        Err(e) => {
                            log::warn!("Failed to play audio source '{}': {}", source.clip, e);
                        }
                    }
                } else if let Some(handle) = source.handle {
                    // Update volume/pitch on existing voice.
                    let _ = mixer.set_volume(handle, source.volume);
                    let _ = mixer.set_pitch(handle, source.pitch);
                    // Resume if paused.
                    if let Ok(ChannelState::Paused) = mixer.channel_state(handle) {
                        let _ = mixer.resume(handle);
                    }
                }
            }
            PlaybackState::Paused => {
                if let Some(handle) = source.handle {
                    let _ = mixer.pause(handle);
                }
            }
            PlaybackState::Stopped => {
                if let Some(handle) = source.handle {
                    let _ = mixer.stop(handle, 0.0);
                    source.handle = None;
                }
            }
        }

        // Apply spatial parameters.
        if source.spatial {
            if let Some(handle) = source.handle {
                Self::apply_spatial_to_mixer(
                    mixer,
                    &listener_snapshot,
                    speed_of_sound,
                    handle,
                    &source.spatial_params,
                    &source.occlusion,
                );
            }
        }
    }

    /// Apply spatial audio parameters (attenuation, pan, Doppler) to a mixer
    /// voice.
    ///
    /// This is a static method to avoid borrow-checker issues: `process_source`
    /// holds a mutable borrow on `self.mixer` and needs to pass it here
    /// alongside immutable reads of `self.listener` / `self.speed_of_sound`.
    fn apply_spatial_to_mixer(
        mixer: &mut SoftwareMixer,
        listener: &AudioListener,
        speed_of_sound: f32,
        handle: MixerHandle,
        spatial: &SpatialAudioSource,
        occlusion: &OcclusionParams,
    ) {
        let result = spatial.compute_spatial_params(listener, speed_of_sound);

        // Combine spatial volume with occlusion.
        let occlusion_vol = occlusion.volume_multiplier();
        let final_volume = result.volume * occlusion_vol * listener.volume;

        let _ = mixer.set_volume(handle, final_volume);
        let _ = mixer.set_pitch(handle, result.doppler_pitch_ratio);

        // Update pan on the channel directly.
        if let Some(idx) = mixer.channels.iter().position(|c| c.handle == handle) {
            mixer.channels[idx].pan = result.pan;
        }
    }

    /// Per-frame update.  Call after `update_listener` and all `process_source`
    /// calls.
    pub fn update(&mut self, dt: f32) {
        if self.paused {
            return;
        }

        let mixer = match self.mixer.as_mut() {
            Some(m) => m,
            None => {
                log::warn!("AudioSystem::update called but no mixer is initialized");
                return;
            }
        };

        profiling::scope!("AudioSystem::update");

        mixer.set_master_volume(self.master_volume);
        mixer.update(dt);
    }

    /// Returns a reference to the underlying mixer, if initialized.
    pub fn mixer(&self) -> Option<&SoftwareMixer> {
        self.mixer.as_ref()
    }

    /// Returns a mutable reference to the underlying mixer, if initialized.
    pub fn mixer_mut(&mut self) -> Option<&mut SoftwareMixer> {
        self.mixer.as_mut()
    }

    /// Set the master volume.
    pub fn set_master_volume(&mut self, volume: f32) {
        self.master_volume = volume;
        if let Some(mixer) = self.mixer.as_mut() {
            mixer.set_master_volume(volume);
        }
    }

    /// Pause all audio.
    pub fn pause_all(&mut self) {
        self.paused = true;
        if let Some(mixer) = self.mixer.as_mut() {
            mixer.pause_all();
        }
    }

    /// Resume all audio.
    pub fn resume_all(&mut self) {
        self.paused = false;
        if let Some(mixer) = self.mixer.as_mut() {
            mixer.resume_all();
        }
    }

    /// Stop all audio.
    pub fn stop_all(&mut self) {
        if let Some(mixer) = self.mixer.as_mut() {
            mixer.stop_all();
        }
    }

    /// Set bus volume.
    pub fn set_bus_volume(&mut self, bus: AudioBus, volume: f32) {
        if let Some(mixer) = self.mixer.as_mut() {
            mixer.set_bus_volume(bus, volume);
        }
    }

    /// Mute/unmute a bus.
    pub fn set_bus_mute(&mut self, bus: AudioBus, muted: bool) {
        if let Some(mixer) = self.mixer.as_mut() {
            mixer.set_bus_mute(bus, muted);
        }
    }

    /// Get the current listener.
    pub fn listener(&self) -> &AudioListener {
        &self.listener
    }

    /// Active voice count.
    pub fn active_voice_count(&self) -> u32 {
        self.mixer.as_ref().map_or(0, |m| m.active_voice_count())
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mixer::{AudioClip, SoftwareMixer};
    use glam::Vec3;

    fn make_system() -> AudioSystem {
        let mut system = AudioSystem::new();
        let mut mixer = SoftwareMixer::new(44100, 2, 256, 32);
        mixer.load_clip(AudioClip::sine_wave(440.0, 2.0, 44100));
        system.initialize(mixer);
        system
    }

    #[test]
    fn system_play_source() {
        let mut system = make_system();

        let mut source = AudioSourceComponent {
            clip: "sine_440hz".into(),
            ..Default::default()
        };
        source.play();

        system.process_source(&mut source);
        assert!(source.handle.is_some());
        assert_eq!(system.active_voice_count(), 1);

        system.update(1.0 / 60.0);
    }

    #[test]
    fn system_stop_source() {
        let mut system = make_system();

        let mut source = AudioSourceComponent {
            clip: "sine_440hz".into(),
            ..Default::default()
        };
        source.play();
        system.process_source(&mut source);
        assert!(source.handle.is_some());

        source.stop();
        system.process_source(&mut source);
        assert!(source.handle.is_none());
    }

    #[test]
    fn system_pause_resume() {
        let mut system = make_system();

        let mut source = AudioSourceComponent {
            clip: "sine_440hz".into(),
            ..Default::default()
        };
        source.play();
        system.process_source(&mut source);
        let handle = source.handle.unwrap();

        source.pause();
        system.process_source(&mut source);

        let mixer = system.mixer().unwrap();
        assert_eq!(mixer.channel_state(handle).unwrap(), ChannelState::Paused);

        source.play();
        system.process_source(&mut source);
        assert_eq!(
            system.mixer().unwrap().channel_state(handle).unwrap(),
            ChannelState::Playing
        );
    }

    #[test]
    fn system_play_on_awake() {
        let mut system = make_system();

        let mut source = AudioSourceComponent {
            clip: "sine_440hz".into(),
            play_on_awake: true,
            ..Default::default()
        };

        system.process_source(&mut source);
        assert!(source.handle.is_some(), "play_on_awake should auto-play");
    }

    #[test]
    fn system_spatial_source() {
        let mut system = make_system();

        let listener = AudioListenerComponent {
            listener: AudioListener {
                position: Vec3::ZERO,
                forward: Vec3::NEG_Z,
                up: Vec3::Y,
                ..Default::default()
            },
            active: true,
        };
        system.update_listener(&listener);

        let mut source = AudioSourceComponent {
            clip: "sine_440hz".into(),
            spatial: true,
            ..Default::default()
        };
        source.spatial_params.position = Vec3::new(5.0, 0.0, 0.0);
        source.play();

        system.process_source(&mut source);
        assert!(source.handle.is_some());

        system.update(1.0 / 60.0);
    }

    #[test]
    fn system_bus_volume() {
        let mut system = make_system();
        system.set_bus_volume(AudioBus::SFX, 0.3);
        system.set_bus_mute(AudioBus::Music, true);
        // Should not panic.
    }

    #[test]
    fn system_master_volume() {
        let mut system = make_system();
        system.set_master_volume(0.5);
        assert!((system.master_volume - 0.5).abs() < 1e-6);
    }

    #[test]
    fn system_pause_resume_all() {
        let mut system = make_system();

        let mut s1 = AudioSourceComponent {
            clip: "sine_440hz".into(),
            ..Default::default()
        };
        s1.play();
        system.process_source(&mut s1);

        system.pause_all();
        assert!(system.paused);

        system.resume_all();
        assert!(!system.paused);
    }

    #[test]
    fn system_stop_all() {
        let mut system = make_system();

        let mut s1 = AudioSourceComponent {
            clip: "sine_440hz".into(),
            ..Default::default()
        };
        s1.play();
        system.process_source(&mut s1);

        system.stop_all();
        system.update(1.0 / 60.0);
        assert_eq!(system.active_voice_count(), 0);
    }

    #[test]
    fn source_set_volume_pitch() {
        let mut source = AudioSourceComponent::default();
        source.set_volume(0.7);
        source.set_pitch(1.5);
        assert!(source.dirty);
        assert!((source.volume - 0.7).abs() < 1e-6);
        assert!((source.pitch - 1.5).abs() < 1e-6);
    }

    #[test]
    fn source_set_position_velocity() {
        let mut source = AudioSourceComponent::default();
        source.set_position(Vec3::new(1.0, 2.0, 3.0));
        source.set_velocity(Vec3::new(0.0, -1.0, 0.0));
        assert!((source.spatial_params.position - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-6);
        assert!((source.spatial_params.velocity - Vec3::new(0.0, -1.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn listener_component_orientation() {
        let mut lc = AudioListenerComponent::default();
        lc.set_position(Vec3::new(5.0, 0.0, 0.0));
        lc.set_orientation(Vec3::X, Vec3::Y);
        lc.set_velocity(Vec3::new(1.0, 0.0, 0.0));
        assert!((lc.listener.position - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-6);
        assert!((lc.listener.forward - Vec3::X).length() < 1e-4);
    }
}
