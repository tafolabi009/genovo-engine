//! # Genovo Audio
//!
//! Audio playback, mixing, and spatial audio module for the Genovo game engine.
//!
//! This crate provides:
//!
//! - **Software mixer** -- CPU-based PCM audio mixing with multi-channel
//!   playback, pitch shifting, bus routing, priority-based voice stealing,
//!   and fade envelopes.
//! - **WAV parser** -- loads RIFF/WAVE files (8-bit, 16-bit, 24-bit PCM and
//!   32-bit IEEE float) into normalised f32 sample buffers.
//! - **Spatial audio** -- 3D positioned sound with distance attenuation
//!   (linear, logarithmic, inverse-square, custom curve), stereo panning,
//!   Doppler effect, and directional cone emitters.
//! - **Audio backends** -- pluggable output layer with a fully functional
//!   `NullBackend` for headless/testing use and interface stubs for WASAPI,
//!   CoreAudio, and ALSA.
//! - **ECS integration** -- `AudioSourceComponent`, `AudioListenerComponent`,
//!   and `AudioSystem` for driving audio from entity transforms.
//! - **Music system** (`music`) -- background music with crossfade, playlists,
//!   layered/adaptive music with beat synchronisation, and stingers.
//! - **Ambient sound** (`ambience`) -- zone-based ambient sounds with smooth
//!   crossfading, layered loops, and random one-shot triggers.
//! - **Audio graph** (`audio_graph`) -- node-based audio routing with
//!   topological sort evaluation, built-in gain/pan/filter/delay/mixer nodes.

pub mod ambience;
pub mod audio_graph;
pub mod backends;
pub mod components;
pub mod dsp;
pub mod mixer;
pub mod music;
pub mod spatial;

// Re-exports for ergonomic top-level access.
pub use backends::{AudioBackend, NullBackend};
pub use components::{AudioListenerComponent, AudioSourceComponent, AudioSystem, PlaybackState};
pub use mixer::{
    AudioBus, AudioChannel, AudioClip, AudioError, AudioFormat, AudioMixer, AudioResult,
    AudioSource, ChannelState, MixerHandle, SoftwareMixer,
};
pub use dsp::{
    BandPassFilter, Chorus, Compressor, Delay, Distortion, DspChain, DspEffect, Equalizer,
    HighPassFilter, Lfo, LfoWaveform, LowPassFilter, Phaser, Reverb, Tremolo,
};
pub use spatial::{
    AttenuationModel, AudioListener, HrtfData, HrtfProcessor, OcclusionParams,
    SpatialAudioSource, SpatialResult,
};

// Re-exports for expanded audio systems.
pub use ambience::{AmbienceComponent, AmbienceManager, AmbienceSystem, AmbienceZone};
pub use audio_graph::{AudioGraph, AudioGraphBuilder, AudioNode, NodeHandle};
pub use music::{
    MusicComponent, MusicLayerSystem, MusicPlayer, MusicState, MusicSystem, MusicTrack, Playlist,
};
