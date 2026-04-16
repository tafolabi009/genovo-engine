//! # genovo-replay
//!
//! Replay recording and playback system for the Genovo game engine.
//!
//! Provides frame-accurate recording of input events, entity state
//! snapshots, and periodic world-state checkpoints. Replays can be
//! played back at variable speed with seeking, and support features
//! like kill-cam slow-motion sequences.

pub mod replay;

pub use replay::{
    Checkpoint, CustomEvent, EntityEvent, EntityEventKind, InputEvent, InputFrame, KillCam,
    KillCamState, PlaybackState, ReplayData, ReplayError, ReplayHeader, ReplayPlayer,
    ReplayRecorder, ReplaySettings, RecordingState,
};
