//! # genovo-replay
//!
//! Replay recording and playback system for the Genovo game engine.
//!
//! Provides frame-accurate recording of input events, entity state
//! snapshots, and periodic world-state checkpoints. Replays can be
//! played back at variable speed with seeking, and support features
//! like kill-cam slow-motion sequences.

pub mod replay;
pub mod replay_camera;
pub mod replay_compression;

pub use replay::{
    Checkpoint, CustomEvent, EntityEvent, EntityEventKind, InputEvent, InputFrame, KillCam,
    KillCamState, PlaybackState, ReplayData, ReplayError, ReplayHeader, ReplayPlayer,
    ReplayRecorder, ReplaySettings, RecordingState,
};
pub use replay_compression::{
    BitReader, BitWriter, CompressedReplayStream, CompressionError, FrameDelta,
    InputChangeTracker, InputDelta, QuantizationPrecision, ReplaySizeAnalysis,
};
pub use replay_camera::{
    CameraMode, CameraTransform, DirectedCameraSequence, EasingFunction, FollowCamera,
    FreeCamera, OrbitCamera, PipViewport, PlaybackSpeedController, ReplayCameraController,
    SplitLayout,
};
