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
pub mod cutscene_player;
pub mod sequencer;
pub mod subtitle_system;
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
pub use cutscene_player::{
    CutsceneDescriptor, CutsceneEvent, CutsceneId, CutsceneKeyframe, CutscenePlayer,
    CutsceneStateEvent, CutsceneTrack, EasingType, KeyframeData, LetterboxState, PlaybackState,
    TrackType, TransitionState, TransitionType,
};
// Cinematic camera: Bezier path following, focus tracking, dolly zoom,
// handheld shake, rack focus, letterbox transitions.
pub mod camera_system;

// Cutscene management: cutscene loading, cutscene state machine, skip handling,
// cutscene events, cutscene blending with gameplay.
pub mod cutscene_manager;

pub use subtitle_system::{
    ActiveSubtitleSlot, PortraitAlignment, SpeakerPortrait, SubtitleEntry, SubtitleId,
    SubtitleManager, SubtitlePosition, SubtitlePreset, SubtitleStyle, SubtitleTextAlignment,
    WordRevealState,
};
