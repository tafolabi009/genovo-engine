//! # Genovo Cinematics
//!
//! Cinematic sequencer, camera control, and timeline editing for the Genovo
//! game engine. This crate provides everything needed to author and play back
//! non-interactive (or semi-interactive) cinematic sequences:
//!
//! - **Sequencer** -- typed keyframe tracks with multiple interpolation modes,
//!   a sequence player, and runtime property binding.
//! - **Camera** -- cinematic camera movements (dolly, truck, pan, shake) with
//!   spline paths and constraint blending.
//! - **Timeline** -- editor-facing data model for multi-track editing,
//!   curve editing, and cutscene asset serialization.

pub mod camera;
pub mod sequencer;
pub mod timeline;

// Re-exports for ergonomic access.
pub use camera::{
    CameraConstraint, CameraKeypoint, CameraPath, CameraShake, CinematicCamera, ImpulseShake,
    PerlinNoise, PerlinShake, ShakeInstance,
};
pub use sequencer::{
    AudioTrack, BoolTrack, CameraTrack, ColorTrack, EventTrack, FloatTrack, Interpolation,
    Keyframe, LoopMode, Sequence, SequencePlayer, SequenceTrack, SubSequenceTrack, TrackKind,
    TransformTrack,
};
pub use timeline::{
    CurveEditor, CutsceneAsset, Marker, TangentMode, Timeline, TimelineGroup, TimelineSelection,
    TimelineTrackMeta,
};
