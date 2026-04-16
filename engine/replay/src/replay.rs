//! Replay Recording and Playback
//!
//! # Architecture
//!
//! Recording works by capturing two streams of data each frame:
//!
//! 1. **Input events**: Every input action (key press, mouse movement, button
//!    press) is recorded with its exact frame number and timestamp.
//!
//! 2. **Entity events**: Spawns, despawns, and component snapshots are
//!    recorded as delta events between checkpoints.
//!
//! 3. **Checkpoints**: At configurable intervals (e.g., every 300 frames) a
//!    full snapshot of the world state is captured. This enables fast seeking:
//!    to reach frame N, find the nearest checkpoint before N and replay
//!    inputs forward from there.
//!
//! # Playback
//!
//! The [`ReplayPlayer`] reconstructs the world state by loading the nearest
//! checkpoint and replaying input frames forward. Variable-speed playback
//! and reverse are supported through the same seek mechanism.
//!
//! # Compression
//!
//! Between checkpoints, entity state is delta-encoded: only components that
//! changed since the last frame are stored. Checkpoints contain full state
//! and serve as keyframes.

use std::collections::HashMap;
use std::path::Path;

use glam::Vec3;
use serde::{Deserialize, Serialize};

use genovo_ecs::Entity;

// ---------------------------------------------------------------------------
// ReplayError
// ---------------------------------------------------------------------------

/// Errors that can occur during replay operations.
#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    #[error("replay is not recording")]
    NotRecording,

    #[error("replay is not playing")]
    NotPlaying,

    #[error("invalid replay data: {0}")]
    InvalidData(String),

    #[error("seek target frame {0} is out of range (0..{1})")]
    SeekOutOfRange(u64, u64),

    #[error("no checkpoint available before frame {0}")]
    NoCheckpoint(u64),

    #[error("version mismatch: replay version {replay} != engine version {engine}")]
    VersionMismatch { replay: u32, engine: u32 },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("{0}")]
    Other(String),
}

// ---------------------------------------------------------------------------
// InputEvent
// ---------------------------------------------------------------------------

/// A single input event (key press, mouse movement, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputEvent {
    /// A keyboard key was pressed.
    KeyDown {
        /// Key code (platform-specific scancode).
        key_code: u32,
        /// Whether this is a repeated press.
        is_repeat: bool,
    },
    /// A keyboard key was released.
    KeyUp {
        key_code: u32,
    },
    /// Mouse button pressed.
    MouseButtonDown {
        button: u8,
        position: [f32; 2],
    },
    /// Mouse button released.
    MouseButtonUp {
        button: u8,
        position: [f32; 2],
    },
    /// Mouse moved.
    MouseMove {
        position: [f32; 2],
        delta: [f32; 2],
    },
    /// Mouse wheel scrolled.
    MouseScroll {
        delta: f32,
    },
    /// Gamepad axis moved.
    GamepadAxis {
        gamepad_id: u32,
        axis: u8,
        value: f32,
    },
    /// Gamepad button pressed.
    GamepadButtonDown {
        gamepad_id: u32,
        button: u8,
    },
    /// Gamepad button released.
    GamepadButtonUp {
        gamepad_id: u32,
        button: u8,
    },
    /// Touch event started.
    TouchStart {
        touch_id: u32,
        position: [f32; 2],
    },
    /// Touch moved.
    TouchMove {
        touch_id: u32,
        position: [f32; 2],
    },
    /// Touch ended.
    TouchEnd {
        touch_id: u32,
        position: [f32; 2],
    },
    /// Custom input action (game-specific).
    CustomAction {
        name: String,
        value: f32,
    },
}

impl InputEvent {
    /// Get a human-readable name for this event type.
    pub fn event_type(&self) -> &str {
        match self {
            InputEvent::KeyDown { .. } => "key_down",
            InputEvent::KeyUp { .. } => "key_up",
            InputEvent::MouseButtonDown { .. } => "mouse_down",
            InputEvent::MouseButtonUp { .. } => "mouse_up",
            InputEvent::MouseMove { .. } => "mouse_move",
            InputEvent::MouseScroll { .. } => "mouse_scroll",
            InputEvent::GamepadAxis { .. } => "gamepad_axis",
            InputEvent::GamepadButtonDown { .. } => "gamepad_button_down",
            InputEvent::GamepadButtonUp { .. } => "gamepad_button_up",
            InputEvent::TouchStart { .. } => "touch_start",
            InputEvent::TouchMove { .. } => "touch_move",
            InputEvent::TouchEnd { .. } => "touch_end",
            InputEvent::CustomAction { .. } => "custom_action",
        }
    }

    /// Compute a compact size estimate for this event.
    pub fn estimated_size(&self) -> usize {
        match self {
            InputEvent::CustomAction { name, .. } => 16 + name.len(),
            _ => 16,
        }
    }
}

// ---------------------------------------------------------------------------
// InputFrame
// ---------------------------------------------------------------------------

/// All input events that occurred during a single frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputFrame {
    /// Frame number.
    pub frame: u64,
    /// Timestamp in seconds since recording started.
    pub timestamp: f64,
    /// Delta time for this frame.
    pub dt: f32,
    /// Input events that occurred this frame.
    pub events: Vec<InputEvent>,
}

impl InputFrame {
    /// Create a new input frame.
    pub fn new(frame: u64, timestamp: f64, dt: f32) -> Self {
        Self {
            frame,
            timestamp,
            dt,
            events: Vec::new(),
        }
    }

    /// Add an input event to this frame.
    pub fn add_event(&mut self, event: InputEvent) {
        self.events.push(event);
    }

    /// Returns `true` if no input events occurred this frame.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Count of events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }
}

// ---------------------------------------------------------------------------
// EntityEventKind
// ---------------------------------------------------------------------------

/// Types of entity lifecycle events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntityEventKind {
    /// An entity was spawned.
    Spawned {
        /// Serialized component data for the new entity.
        components: HashMap<String, Vec<u8>>,
        /// World-space position at spawn time.
        position: [f32; 3],
        /// Rotation as quaternion [x, y, z, w].
        rotation: [f32; 4],
    },
    /// An entity was despawned.
    Despawned,
    /// A component was added to an entity.
    ComponentAdded {
        component_type: String,
        data: Vec<u8>,
    },
    /// A component was removed from an entity.
    ComponentRemoved {
        component_type: String,
    },
    /// A component was modified (delta from last state).
    ComponentChanged {
        component_type: String,
        /// Full new component data (in a real engine, this would be delta-encoded).
        data: Vec<u8>,
    },
    /// Entity position changed.
    PositionChanged {
        position: [f32; 3],
    },
    /// Entity rotation changed.
    RotationChanged {
        rotation: [f32; 4],
    },
}

// ---------------------------------------------------------------------------
// EntityEvent
// ---------------------------------------------------------------------------

/// An event affecting a specific entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityEvent {
    /// Frame number when this event occurred.
    pub frame: u64,
    /// Entity id (index only -- generation is reconstructed from checkpoints).
    pub entity_id: u32,
    /// The event.
    pub kind: EntityEventKind,
}

impl EntityEvent {
    /// Create a new entity event.
    pub fn new(frame: u64, entity_id: u32, kind: EntityEventKind) -> Self {
        Self {
            frame,
            entity_id,
            kind,
        }
    }

    /// Estimated memory size.
    pub fn estimated_size(&self) -> usize {
        let base = 24;
        match &self.kind {
            EntityEventKind::Spawned { components, .. } => {
                base + components.values().map(|v| v.len()).sum::<usize>()
            }
            EntityEventKind::ComponentAdded { data, .. } => base + data.len(),
            EntityEventKind::ComponentChanged { data, .. } => base + data.len(),
            _ => base,
        }
    }
}

// ---------------------------------------------------------------------------
// CustomEvent
// ---------------------------------------------------------------------------

/// A game-specific event recorded in the replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomEvent {
    /// Frame number.
    pub frame: u64,
    /// Timestamp.
    pub timestamp: f64,
    /// Event name.
    pub name: String,
    /// Event properties.
    pub properties: HashMap<String, String>,
    /// Associated entity (if any).
    pub entity_id: Option<u32>,
    /// World-space position (if relevant).
    pub position: Option<[f32; 3]>,
}

impl CustomEvent {
    /// Create a new custom event.
    pub fn new(frame: u64, timestamp: f64, name: impl Into<String>) -> Self {
        Self {
            frame,
            timestamp,
            name: name.into(),
            properties: HashMap::new(),
            entity_id: None,
            position: None,
        }
    }

    /// Add a property.
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Associate with an entity.
    pub fn with_entity(mut self, entity_id: u32) -> Self {
        self.entity_id = Some(entity_id);
        self
    }

    /// Set position.
    pub fn with_position(mut self, pos: [f32; 3]) -> Self {
        self.position = Some(pos);
        self
    }
}

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

/// A full world-state snapshot at a specific frame.
///
/// Checkpoints are the keyframes of the replay. They contain enough
/// information to reconstruct the entire world state without replaying
/// from the beginning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Frame number of this checkpoint.
    pub frame: u64,
    /// Timestamp.
    pub timestamp: f64,
    /// Full entity state: entity_id -> (components, position, rotation).
    pub entities: HashMap<u32, CheckpointEntity>,
    /// Custom game state.
    pub game_state: HashMap<String, Vec<u8>>,
    /// Camera position at this checkpoint.
    pub camera_position: [f32; 3],
    /// Camera rotation at this checkpoint.
    pub camera_rotation: [f32; 4],
    /// Compressed size in bytes (0 if not compressed).
    pub compressed_size: usize,
}

/// Entity state within a checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointEntity {
    /// Entity index.
    pub entity_id: u32,
    /// Entity generation.
    pub generation: u32,
    /// Component data, keyed by type name.
    pub components: HashMap<String, Vec<u8>>,
    /// World position.
    pub position: [f32; 3],
    /// Rotation quaternion.
    pub rotation: [f32; 4],
    /// Scale.
    pub scale: [f32; 3],
    /// Entity name / tag.
    pub name: String,
}

impl Checkpoint {
    /// Create a new empty checkpoint.
    pub fn new(frame: u64, timestamp: f64) -> Self {
        Self {
            frame,
            timestamp,
            entities: HashMap::new(),
            game_state: HashMap::new(),
            camera_position: [0.0, 0.0, 0.0],
            camera_rotation: [0.0, 0.0, 0.0, 1.0],
            compressed_size: 0,
        }
    }

    /// Add an entity to this checkpoint.
    pub fn add_entity(&mut self, entity: CheckpointEntity) {
        self.entities.insert(entity.entity_id, entity);
    }

    /// Estimated memory size of this checkpoint.
    pub fn estimated_size(&self) -> usize {
        let mut size = std::mem::size_of::<Self>();
        for (_, e) in &self.entities {
            size += 32 + e.name.len();
            for (k, v) in &e.components {
                size += k.len() + v.len();
            }
        }
        for (k, v) in &self.game_state {
            size += k.len() + v.len();
        }
        size
    }
}

// ---------------------------------------------------------------------------
// ReplayHeader
// ---------------------------------------------------------------------------

/// Metadata header for a replay file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayHeader {
    /// Magic bytes for file format identification.
    pub magic: u32,
    /// Version of the replay format.
    pub version: u32,
    /// Total duration in seconds.
    pub duration: f64,
    /// Total number of frames.
    pub frame_count: u64,
    /// Game mode / ruleset name.
    pub game_mode: String,
    /// Map / level name.
    pub map_name: String,
    /// Date and time when the recording started (ISO 8601).
    pub recorded_at: String,
    /// Player name.
    pub player_name: String,
    /// Engine version string.
    pub engine_version: String,
    /// Checkpoint interval in frames.
    pub checkpoint_interval: u64,
    /// Number of checkpoints in the replay.
    pub checkpoint_count: u32,
    /// Custom metadata.
    pub metadata: HashMap<String, String>,
}

impl ReplayHeader {
    /// Current replay format version.
    pub const FORMAT_VERSION: u32 = 1;
    /// Magic bytes: "GRVR" (Genovo Replay).
    pub const MAGIC: u32 = 0x47525652;

    /// Create a new header with defaults.
    pub fn new() -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::FORMAT_VERSION,
            duration: 0.0,
            frame_count: 0,
            game_mode: String::new(),
            map_name: String::new(),
            recorded_at: String::new(),
            player_name: String::new(),
            engine_version: String::new(),
            checkpoint_interval: 300,
            checkpoint_count: 0,
            metadata: HashMap::new(),
        }
    }

    /// Validate the header.
    pub fn validate(&self) -> Result<(), ReplayError> {
        if self.magic != Self::MAGIC {
            return Err(ReplayError::InvalidData(
                "Invalid magic bytes".to_owned(),
            ));
        }
        if self.version > Self::FORMAT_VERSION {
            return Err(ReplayError::VersionMismatch {
                replay: self.version,
                engine: Self::FORMAT_VERSION,
            });
        }
        Ok(())
    }
}

impl Default for ReplayHeader {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ReplayData
// ---------------------------------------------------------------------------

/// Complete replay data, including header, input frames, entity events,
/// checkpoints, and custom events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayData {
    /// Header with metadata.
    pub header: ReplayHeader,
    /// Input frames (one per game frame, may be empty if no input occurred).
    pub input_frames: Vec<InputFrame>,
    /// Entity lifecycle events (spawns, despawns, component changes).
    pub entity_events: Vec<EntityEvent>,
    /// World-state checkpoints at regular intervals.
    pub checkpoints: Vec<Checkpoint>,
    /// Custom game events.
    pub custom_events: Vec<CustomEvent>,
}

impl ReplayData {
    /// Create new empty replay data.
    pub fn new() -> Self {
        Self {
            header: ReplayHeader::new(),
            input_frames: Vec::new(),
            entity_events: Vec::new(),
            checkpoints: Vec::new(),
            custom_events: Vec::new(),
        }
    }

    /// Total estimated memory usage.
    pub fn estimated_memory(&self) -> usize {
        let mut total = std::mem::size_of::<Self>();
        total += self.input_frames.len() * 64; // rough estimate per frame
        for e in &self.entity_events {
            total += e.estimated_size();
        }
        for c in &self.checkpoints {
            total += c.estimated_size();
        }
        total += self.custom_events.len() * 128;
        total
    }

    /// Get the checkpoint nearest to (but not after) the given frame.
    pub fn nearest_checkpoint(&self, frame: u64) -> Option<&Checkpoint> {
        self.checkpoints
            .iter()
            .rev()
            .find(|c| c.frame <= frame)
    }

    /// Get all entity events between two frames (inclusive).
    pub fn events_in_range(&self, start_frame: u64, end_frame: u64) -> Vec<&EntityEvent> {
        self.entity_events
            .iter()
            .filter(|e| e.frame >= start_frame && e.frame <= end_frame)
            .collect()
    }

    /// Get all input frames between two frames (inclusive).
    pub fn inputs_in_range(&self, start_frame: u64, end_frame: u64) -> Vec<&InputFrame> {
        self.input_frames
            .iter()
            .filter(|f| f.frame >= start_frame && f.frame <= end_frame)
            .collect()
    }

    /// Get all custom events between two frames (inclusive).
    pub fn custom_events_in_range(&self, start_frame: u64, end_frame: u64) -> Vec<&CustomEvent> {
        self.custom_events
            .iter()
            .filter(|e| e.frame >= start_frame && e.frame <= end_frame)
            .collect()
    }

    /// Save replay data to a JSON file.
    pub fn save_to_file(&self, path: &Path) -> Result<(), ReplayError> {
        let json = serde_json::to_string(self)
            .map_err(|e| ReplayError::Serialization(e.to_string()))?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load replay data from a JSON file.
    pub fn load_from_file(path: &Path) -> Result<Self, ReplayError> {
        let data = std::fs::read_to_string(path)?;
        let replay: Self = serde_json::from_str(&data)
            .map_err(|e| ReplayError::Serialization(e.to_string()))?;
        replay.header.validate()?;
        Ok(replay)
    }

    /// Save replay data in a compact binary format.
    ///
    /// Format: [header_json_length: u32][header_json][data_json_length: u32][data_json]
    pub fn save_binary(&self, path: &Path) -> Result<(), ReplayError> {
        let header_json = serde_json::to_string(&self.header)
            .map_err(|e| ReplayError::Serialization(e.to_string()))?;

        // Pack frames, events, checkpoints separately.
        let frames_json = serde_json::to_string(&self.input_frames)
            .map_err(|e| ReplayError::Serialization(e.to_string()))?;

        let events_json = serde_json::to_string(&self.entity_events)
            .map_err(|e| ReplayError::Serialization(e.to_string()))?;

        let checkpoints_json = serde_json::to_string(&self.checkpoints)
            .map_err(|e| ReplayError::Serialization(e.to_string()))?;

        let custom_json = serde_json::to_string(&self.custom_events)
            .map_err(|e| ReplayError::Serialization(e.to_string()))?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut output = Vec::new();

        // Magic + version.
        output.extend_from_slice(&ReplayHeader::MAGIC.to_le_bytes());
        output.extend_from_slice(&ReplayHeader::FORMAT_VERSION.to_le_bytes());

        // Write sections.
        write_section(&mut output, header_json.as_bytes());
        write_section(&mut output, frames_json.as_bytes());
        write_section(&mut output, events_json.as_bytes());
        write_section(&mut output, checkpoints_json.as_bytes());
        write_section(&mut output, custom_json.as_bytes());

        std::fs::write(path, output)?;
        Ok(())
    }

    /// Load replay data from the compact binary format.
    pub fn load_binary(path: &Path) -> Result<Self, ReplayError> {
        let data = std::fs::read(path)?;

        if data.len() < 8 {
            return Err(ReplayError::InvalidData("File too short".to_owned()));
        }

        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

        if magic != ReplayHeader::MAGIC {
            return Err(ReplayError::InvalidData("Invalid magic bytes".to_owned()));
        }
        if version > ReplayHeader::FORMAT_VERSION {
            return Err(ReplayError::VersionMismatch {
                replay: version,
                engine: ReplayHeader::FORMAT_VERSION,
            });
        }

        let mut offset = 8;
        let header_bytes = read_section(&data, &mut offset)?;
        let frames_bytes = read_section(&data, &mut offset)?;
        let events_bytes = read_section(&data, &mut offset)?;
        let checkpoints_bytes = read_section(&data, &mut offset)?;
        let custom_bytes = read_section(&data, &mut offset)?;

        let header: ReplayHeader = serde_json::from_slice(header_bytes)
            .map_err(|e| ReplayError::Serialization(e.to_string()))?;
        let input_frames: Vec<InputFrame> = serde_json::from_slice(frames_bytes)
            .map_err(|e| ReplayError::Serialization(e.to_string()))?;
        let entity_events: Vec<EntityEvent> = serde_json::from_slice(events_bytes)
            .map_err(|e| ReplayError::Serialization(e.to_string()))?;
        let checkpoints: Vec<Checkpoint> = serde_json::from_slice(checkpoints_bytes)
            .map_err(|e| ReplayError::Serialization(e.to_string()))?;
        let custom_events: Vec<CustomEvent> = serde_json::from_slice(custom_bytes)
            .map_err(|e| ReplayError::Serialization(e.to_string()))?;

        Ok(Self {
            header,
            input_frames,
            entity_events,
            checkpoints,
            custom_events,
        })
    }
}

impl Default for ReplayData {
    fn default() -> Self {
        Self::new()
    }
}

/// Write a length-prefixed section to the output buffer.
fn write_section(output: &mut Vec<u8>, data: &[u8]) {
    let len = data.len() as u32;
    output.extend_from_slice(&len.to_le_bytes());
    output.extend_from_slice(data);
}

/// Read a length-prefixed section from the input buffer.
fn read_section<'a>(data: &'a [u8], offset: &mut usize) -> Result<&'a [u8], ReplayError> {
    if *offset + 4 > data.len() {
        return Err(ReplayError::InvalidData(
            "Unexpected end of file reading section length".to_owned(),
        ));
    }

    let len = u32::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
    ]) as usize;
    *offset += 4;

    if *offset + len > data.len() {
        return Err(ReplayError::InvalidData(
            "Unexpected end of file reading section data".to_owned(),
        ));
    }

    let section = &data[*offset..*offset + len];
    *offset += len;
    Ok(section)
}

// ---------------------------------------------------------------------------
// RecordingState
// ---------------------------------------------------------------------------

/// State of the replay recorder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordingState {
    /// Not recording.
    Idle,
    /// Currently recording.
    Recording,
    /// Recording paused.
    Paused,
    /// Finalizing the recording (writing checkpoints, etc.).
    Finalizing,
}

// ---------------------------------------------------------------------------
// ReplaySettings
// ---------------------------------------------------------------------------

/// Configuration for replay recording.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaySettings {
    /// Interval in frames between full checkpoints.
    pub checkpoint_interval: u64,
    /// Whether to record input events.
    pub record_inputs: bool,
    /// Whether to record entity events (spawns, despawns).
    pub record_entities: bool,
    /// Whether to record component changes.
    pub record_component_changes: bool,
    /// Maximum replay duration in seconds (0 = unlimited).
    pub max_duration: f64,
    /// Maximum replay memory in bytes (0 = unlimited).
    pub max_memory: usize,
    /// Whether to record custom game events.
    pub record_custom_events: bool,
    /// Game mode string stored in the header.
    pub game_mode: String,
    /// Map name stored in the header.
    pub map_name: String,
    /// Player name stored in the header.
    pub player_name: String,
}

impl Default for ReplaySettings {
    fn default() -> Self {
        Self {
            checkpoint_interval: 300,
            record_inputs: true,
            record_entities: true,
            record_component_changes: true,
            max_duration: 0.0,
            max_memory: 0,
            record_custom_events: true,
            game_mode: String::new(),
            map_name: String::new(),
            player_name: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// ReplayRecorder
// ---------------------------------------------------------------------------

/// Records gameplay into a [`ReplayData`] structure.
///
/// # Usage
///
/// ```ignore
/// let mut recorder = ReplayRecorder::new(ReplaySettings::default());
/// recorder.start_recording();
///
/// // Each frame:
/// recorder.begin_frame(frame_number, timestamp, dt);
/// recorder.record_input(InputEvent::KeyDown { key_code: 32, is_repeat: false });
/// recorder.end_frame();
///
/// // When done:
/// let replay = recorder.stop_recording();
/// replay.save_to_file("replay.json")?;
/// ```
pub struct ReplayRecorder {
    /// Recording settings.
    settings: ReplaySettings,
    /// Current state.
    state: RecordingState,
    /// Accumulated replay data.
    data: ReplayData,
    /// Current frame being recorded.
    current_frame: u64,
    /// Current timestamp.
    current_timestamp: f64,
    /// Current frame's input events (accumulated during the frame).
    current_input_frame: Option<InputFrame>,
    /// Frame of the last checkpoint.
    last_checkpoint_frame: u64,
    /// Estimated memory usage.
    estimated_memory: usize,
    /// Start time of recording (for elapsed calculation).
    start_time: f64,
}

impl ReplayRecorder {
    /// Create a new replay recorder with the given settings.
    pub fn new(settings: ReplaySettings) -> Self {
        Self {
            settings,
            state: RecordingState::Idle,
            data: ReplayData::new(),
            current_frame: 0,
            current_timestamp: 0.0,
            current_input_frame: None,
            last_checkpoint_frame: 0,
            estimated_memory: 0,
            start_time: 0.0,
        }
    }

    /// Start recording.
    pub fn start_recording(&mut self) {
        if self.state != RecordingState::Idle {
            log::warn!("Recorder already active (state={:?})", self.state);
            return;
        }

        self.data = ReplayData::new();
        self.data.header.game_mode = self.settings.game_mode.clone();
        self.data.header.map_name = self.settings.map_name.clone();
        self.data.header.player_name = self.settings.player_name.clone();
        self.data.header.checkpoint_interval = self.settings.checkpoint_interval;
        self.data.header.engine_version = "genovo-0.1.0".to_owned();

        self.current_frame = 0;
        self.current_timestamp = 0.0;
        self.last_checkpoint_frame = 0;
        self.estimated_memory = 0;
        self.state = RecordingState::Recording;

        log::info!("Replay recording started");
    }

    /// Pause recording.
    pub fn pause(&mut self) {
        if self.state == RecordingState::Recording {
            self.state = RecordingState::Paused;
            log::info!("Replay recording paused at frame {}", self.current_frame);
        }
    }

    /// Resume recording.
    pub fn resume(&mut self) {
        if self.state == RecordingState::Paused {
            self.state = RecordingState::Recording;
            log::info!("Replay recording resumed at frame {}", self.current_frame);
        }
    }

    /// Stop recording and return the completed replay data.
    pub fn stop_recording(&mut self) -> ReplayData {
        self.state = RecordingState::Finalizing;

        // Finalize header.
        self.data.header.frame_count = self.current_frame;
        self.data.header.duration = self.current_timestamp;
        self.data.header.checkpoint_count = self.data.checkpoints.len() as u32;

        log::info!(
            "Replay recording stopped: {} frames, {:.1}s, {} checkpoints, {:.1} KiB",
            self.current_frame,
            self.current_timestamp,
            self.data.checkpoints.len(),
            self.estimated_memory as f64 / 1024.0,
        );

        self.state = RecordingState::Idle;
        std::mem::take(&mut self.data)
    }

    /// Get the current recording state.
    pub fn state(&self) -> RecordingState {
        self.state
    }

    /// Get the current frame number.
    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Get the current timestamp.
    pub fn current_timestamp(&self) -> f64 {
        self.current_timestamp
    }

    /// Get estimated memory usage of the recording so far.
    pub fn estimated_memory(&self) -> usize {
        self.estimated_memory
    }

    // -- Per-frame recording -----------------------------------------------

    /// Begin recording a new frame. Call at the start of each game frame.
    pub fn begin_frame(&mut self, frame: u64, timestamp: f64, dt: f32) {
        if self.state != RecordingState::Recording {
            return;
        }

        self.current_frame = frame;
        self.current_timestamp = timestamp;

        if self.settings.record_inputs {
            self.current_input_frame = Some(InputFrame::new(frame, timestamp, dt));
        }

        // Check if we should create a checkpoint.
        if self.settings.checkpoint_interval > 0
            && (frame - self.last_checkpoint_frame) >= self.settings.checkpoint_interval
        {
            self.create_checkpoint();
        }

        // Check duration limit.
        if self.settings.max_duration > 0.0 && timestamp >= self.settings.max_duration {
            log::info!("Replay max duration reached ({:.1}s)", timestamp);
            self.pause();
        }

        // Check memory limit.
        if self.settings.max_memory > 0 && self.estimated_memory >= self.settings.max_memory {
            log::info!(
                "Replay max memory reached ({:.1} KiB)",
                self.estimated_memory as f64 / 1024.0,
            );
            self.pause();
        }
    }

    /// Record an input event for the current frame.
    pub fn record_input(&mut self, event: InputEvent) {
        if self.state != RecordingState::Recording || !self.settings.record_inputs {
            return;
        }

        self.estimated_memory += event.estimated_size();

        if let Some(ref mut frame) = self.current_input_frame {
            frame.add_event(event);
        }
    }

    /// Record an entity event (spawn, despawn, component change).
    pub fn record_entity_event(&mut self, entity_id: u32, kind: EntityEventKind) {
        if self.state != RecordingState::Recording || !self.settings.record_entities {
            return;
        }

        let event = EntityEvent::new(self.current_frame, entity_id, kind);
        self.estimated_memory += event.estimated_size();
        self.data.entity_events.push(event);
    }

    /// Record a custom game event.
    pub fn record_custom_event(&mut self, name: impl Into<String>) -> Option<&mut CustomEvent> {
        if self.state != RecordingState::Recording || !self.settings.record_custom_events {
            return None;
        }

        let event = CustomEvent::new(self.current_frame, self.current_timestamp, name);
        self.estimated_memory += 128;
        self.data.custom_events.push(event);
        self.data.custom_events.last_mut()
    }

    /// End the current frame. Call at the end of each game frame.
    pub fn end_frame(&mut self) {
        if self.state != RecordingState::Recording {
            return;
        }

        // Store the input frame (even if empty, to maintain frame indices).
        if let Some(frame) = self.current_input_frame.take() {
            // Only store non-empty frames to save memory.
            if !frame.is_empty() {
                self.data.input_frames.push(frame);
            }
        }
    }

    /// Create a checkpoint at the current frame.
    pub fn create_checkpoint(&mut self) {
        let checkpoint = Checkpoint::new(self.current_frame, self.current_timestamp);
        self.estimated_memory += checkpoint.estimated_size();
        self.data.checkpoints.push(checkpoint);
        self.last_checkpoint_frame = self.current_frame;

        log::trace!("Checkpoint created at frame {}", self.current_frame);
    }

    /// Add entity data to the most recent checkpoint.
    pub fn add_entity_to_checkpoint(&mut self, entity: CheckpointEntity) {
        if let Some(checkpoint) = self.data.checkpoints.last_mut() {
            self.estimated_memory += 32 + entity.name.len();
            for (k, v) in &entity.components {
                self.estimated_memory += k.len() + v.len();
            }
            checkpoint.add_entity(entity);
        }
    }

    /// Set camera state on the most recent checkpoint.
    pub fn set_checkpoint_camera(&mut self, position: [f32; 3], rotation: [f32; 4]) {
        if let Some(checkpoint) = self.data.checkpoints.last_mut() {
            checkpoint.camera_position = position;
            checkpoint.camera_rotation = rotation;
        }
    }
}

// ---------------------------------------------------------------------------
// PlaybackState
// ---------------------------------------------------------------------------

/// State of the replay player.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackState {
    /// Not loaded / inactive.
    Idle,
    /// Loaded but not playing.
    Loaded,
    /// Currently playing.
    Playing,
    /// Paused.
    Paused,
    /// Seeking to a specific frame.
    Seeking,
    /// Playback has reached the end.
    Finished,
}

// ---------------------------------------------------------------------------
// ReplayPlayer
// ---------------------------------------------------------------------------

/// Plays back a recorded replay.
///
/// The player reconstructs the world state by loading the nearest checkpoint
/// before the target frame and then replaying inputs/events forward to the
/// desired frame.
///
/// # Seeking
///
/// Seeking works in two phases:
/// 1. Find the nearest checkpoint at or before the target frame.
/// 2. Replay all input frames and entity events from the checkpoint to the
///    target frame.
///
/// Reverse playback is implemented by seeking to a frame that is
/// `speed * dt` frames before the current frame each tick.
pub struct ReplayPlayer {
    /// The replay data being played back.
    data: Option<ReplayData>,
    /// Current playback state.
    state: PlaybackState,
    /// Current frame index.
    current_frame: u64,
    /// Playback speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double).
    speed: f64,
    /// Target frame rate for time computation.
    target_fps: f64,
    /// Accumulated fractional frame (for sub-frame speed control).
    frame_accumulator: f64,
    /// Whether to loop back to the start when reaching the end.
    looping: bool,
    /// Active kill-cam sequence, if any.
    active_kill_cam: Option<KillCam>,
    /// Callbacks: list of (frame, callback_id) triggers.
    frame_triggers: Vec<(u64, u32)>,
    /// Index of the current checkpoint being used as the base state.
    current_checkpoint_index: Option<usize>,
    /// Events that have been applied since the current checkpoint.
    applied_event_count: usize,
}

impl ReplayPlayer {
    /// Create a new replay player.
    pub fn new() -> Self {
        Self {
            data: None,
            state: PlaybackState::Idle,
            current_frame: 0,
            speed: 1.0,
            target_fps: 60.0,
            frame_accumulator: 0.0,
            looping: false,
            active_kill_cam: None,
            frame_triggers: Vec::new(),
            current_checkpoint_index: None,
            applied_event_count: 0,
        }
    }

    /// Load replay data for playback.
    pub fn load(&mut self, data: ReplayData) -> Result<(), ReplayError> {
        data.header.validate()?;

        log::info!(
            "Loaded replay: {} frames, {:.1}s, {} checkpoints",
            data.header.frame_count,
            data.header.duration,
            data.checkpoints.len(),
        );

        self.data = Some(data);
        self.state = PlaybackState::Loaded;
        self.current_frame = 0;
        self.frame_accumulator = 0.0;
        self.current_checkpoint_index = None;
        self.applied_event_count = 0;

        Ok(())
    }

    /// Unload the current replay.
    pub fn unload(&mut self) {
        self.data = None;
        self.state = PlaybackState::Idle;
        self.current_frame = 0;
        self.active_kill_cam = None;
    }

    /// Start playback from the current frame.
    pub fn play(&mut self) -> Result<(), ReplayError> {
        if self.data.is_none() {
            return Err(ReplayError::NotPlaying);
        }
        self.state = PlaybackState::Playing;
        log::info!("Replay playback started at frame {}", self.current_frame);
        Ok(())
    }

    /// Pause playback.
    pub fn pause(&mut self) {
        if self.state == PlaybackState::Playing {
            self.state = PlaybackState::Paused;
        }
    }

    /// Resume playback.
    pub fn resume(&mut self) {
        if self.state == PlaybackState::Paused {
            self.state = PlaybackState::Playing;
        }
    }

    /// Stop playback entirely.
    pub fn stop(&mut self) {
        self.state = PlaybackState::Loaded;
        self.current_frame = 0;
        self.frame_accumulator = 0.0;
    }

    /// Set playback speed.
    pub fn set_speed(&mut self, speed: f64) {
        self.speed = speed;
    }

    /// Get current playback speed.
    pub fn speed(&self) -> f64 {
        self.speed
    }

    /// Set looping mode.
    pub fn set_looping(&mut self, looping: bool) {
        self.looping = looping;
    }

    /// Get current playback state.
    pub fn playback_state(&self) -> PlaybackState {
        self.state
    }

    /// Get the current frame number.
    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Get the total frame count.
    pub fn frame_count(&self) -> u64 {
        self.data
            .as_ref()
            .map(|d| d.header.frame_count)
            .unwrap_or(0)
    }

    /// Get the total duration in seconds.
    pub fn duration(&self) -> f64 {
        self.data
            .as_ref()
            .map(|d| d.header.duration)
            .unwrap_or(0.0)
    }

    /// Get playback progress (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        let total = self.frame_count();
        if total == 0 {
            return 0.0;
        }
        self.current_frame as f64 / total as f64
    }

    /// Get the current timestamp in seconds.
    pub fn current_timestamp(&self) -> f64 {
        self.progress() * self.duration()
    }

    // -- Seeking -----------------------------------------------------------

    /// Seek to a specific frame.
    ///
    /// Finds the nearest checkpoint before the target frame, loads it,
    /// and returns the events/inputs that need to be replayed to reach
    /// the target frame.
    pub fn seek(&mut self, target_frame: u64) -> Result<SeekResult, ReplayError> {
        let data = self
            .data
            .as_ref()
            .ok_or(ReplayError::NotPlaying)?;

        let max_frame = data.header.frame_count;
        if target_frame > max_frame {
            return Err(ReplayError::SeekOutOfRange(target_frame, max_frame));
        }

        self.state = PlaybackState::Seeking;

        // Find the nearest checkpoint.
        let checkpoint = data.nearest_checkpoint(target_frame);
        let checkpoint_frame = checkpoint
            .map(|c| c.frame)
            .unwrap_or(0);

        // Collect events between checkpoint and target.
        let entity_events = data.events_in_range(checkpoint_frame, target_frame);
        let input_frames = data.inputs_in_range(checkpoint_frame, target_frame);

        let result = SeekResult {
            target_frame,
            checkpoint_frame,
            checkpoint: checkpoint.cloned(),
            entity_events_count: entity_events.len(),
            input_frames_count: input_frames.len(),
        };

        self.current_frame = target_frame;
        self.frame_accumulator = 0.0;
        self.state = PlaybackState::Paused;

        log::debug!(
            "Seeked to frame {}: checkpoint at {}, {} events, {} inputs to replay",
            target_frame,
            checkpoint_frame,
            result.entity_events_count,
            result.input_frames_count,
        );

        Ok(result)
    }

    /// Seek to a specific time in seconds.
    pub fn seek_to_time(&mut self, time: f64) -> Result<SeekResult, ReplayError> {
        let duration = self.duration();
        if duration <= 0.0 {
            return self.seek(0);
        }
        let progress = (time / duration).clamp(0.0, 1.0);
        let frame = (progress * self.frame_count() as f64) as u64;
        self.seek(frame)
    }

    // -- Per-frame update --------------------------------------------------

    /// Advance playback by one real-time frame.
    ///
    /// Returns the list of input events and entity events for the current
    /// replay frame, if any.
    pub fn update(&mut self, dt: f64) -> PlaybackFrame {
        if self.state != PlaybackState::Playing {
            return PlaybackFrame::empty(self.current_frame);
        }

        // Handle kill-cam speed override.
        let effective_speed = if let Some(ref kill_cam) = self.active_kill_cam {
            kill_cam.current_speed()
        } else {
            self.speed
        };

        // Advance frame accumulator.
        self.frame_accumulator += effective_speed * dt * self.target_fps;

        let mut result = PlaybackFrame::empty(self.current_frame);

        // Process whole frames.
        while self.frame_accumulator >= 1.0 {
            self.frame_accumulator -= 1.0;

            if effective_speed >= 0.0 {
                self.current_frame += 1;
            } else {
                if self.current_frame > 0 {
                    self.current_frame -= 1;
                }
            }

            // Collect events for this frame.
            if let Some(ref data) = self.data {
                // Input events.
                for input_frame in &data.input_frames {
                    if input_frame.frame == self.current_frame {
                        result.input_events.extend(input_frame.events.clone());
                        result.dt = input_frame.dt;
                        break;
                    }
                }

                // Entity events.
                for event in &data.entity_events {
                    if event.frame == self.current_frame {
                        result.entity_events.push(event.clone());
                    }
                }

                // Custom events.
                for event in &data.custom_events {
                    if event.frame == self.current_frame {
                        result.custom_events.push(event.clone());
                    }
                }
            }

            // Check triggers.
            for &(trigger_frame, trigger_id) in &self.frame_triggers {
                if trigger_frame == self.current_frame {
                    result.triggered.push(trigger_id);
                }
            }

            // Check end of replay.
            if self.current_frame >= self.frame_count() {
                if self.looping {
                    self.current_frame = 0;
                    self.frame_accumulator = 0.0;
                } else {
                    self.state = PlaybackState::Finished;
                    break;
                }
            }
        }

        // Update kill-cam.
        if let Some(ref mut kill_cam) = self.active_kill_cam {
            kill_cam.update(dt as f32);
            if kill_cam.is_complete() {
                self.active_kill_cam = None;
                self.speed = 1.0;
            }
        }

        result.frame = self.current_frame;
        result
    }

    /// Add a frame trigger: the trigger id will appear in
    /// [`PlaybackFrame::triggered`] when playback reaches the given frame.
    pub fn add_trigger(&mut self, frame: u64, trigger_id: u32) {
        self.frame_triggers.push((frame, trigger_id));
    }

    /// Remove all triggers.
    pub fn clear_triggers(&mut self) {
        self.frame_triggers.clear();
    }

    // -- Kill-cam ----------------------------------------------------------

    /// Start a kill-cam sequence.
    pub fn start_kill_cam(&mut self, kill_cam: KillCam) {
        log::info!(
            "Starting kill-cam at frame {}: {:.1}s duration, {:.1}x -> {:.1}x -> {:.1}x",
            self.current_frame,
            kill_cam.duration,
            kill_cam.initial_speed,
            kill_cam.slow_speed,
            kill_cam.final_speed,
        );
        self.active_kill_cam = Some(kill_cam);
    }

    /// Stop the active kill-cam.
    pub fn stop_kill_cam(&mut self) {
        self.active_kill_cam = None;
    }

    /// Check if a kill-cam is active.
    pub fn is_kill_cam_active(&self) -> bool {
        self.active_kill_cam.is_some()
    }

    /// Get the active kill-cam.
    pub fn kill_cam(&self) -> Option<&KillCam> {
        self.active_kill_cam.as_ref()
    }
}

impl Default for ReplayPlayer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SeekResult
// ---------------------------------------------------------------------------

/// Result of a seek operation.
#[derive(Debug, Clone)]
pub struct SeekResult {
    /// Frame that was sought to.
    pub target_frame: u64,
    /// Frame of the checkpoint that was loaded as the base state.
    pub checkpoint_frame: u64,
    /// The checkpoint data (if any).
    pub checkpoint: Option<Checkpoint>,
    /// Number of entity events between checkpoint and target.
    pub entity_events_count: usize,
    /// Number of input frames between checkpoint and target.
    pub input_frames_count: usize,
}

// ---------------------------------------------------------------------------
// PlaybackFrame
// ---------------------------------------------------------------------------

/// Data for a single playback frame.
#[derive(Debug, Clone, Default)]
pub struct PlaybackFrame {
    /// Current frame number.
    pub frame: u64,
    /// Delta time for this frame (from the recording).
    pub dt: f32,
    /// Input events that occurred on this frame.
    pub input_events: Vec<InputEvent>,
    /// Entity events that occurred on this frame.
    pub entity_events: Vec<EntityEvent>,
    /// Custom events that occurred on this frame.
    pub custom_events: Vec<CustomEvent>,
    /// Trigger IDs that fired on this frame.
    pub triggered: Vec<u32>,
}

impl PlaybackFrame {
    /// Create an empty playback frame.
    pub fn empty(frame: u64) -> Self {
        Self {
            frame,
            dt: 1.0 / 60.0,
            input_events: Vec::new(),
            entity_events: Vec::new(),
            custom_events: Vec::new(),
            triggered: Vec::new(),
        }
    }

    /// Returns `true` if there are no events this frame.
    pub fn is_empty(&self) -> bool {
        self.input_events.is_empty()
            && self.entity_events.is_empty()
            && self.custom_events.is_empty()
    }
}

// ---------------------------------------------------------------------------
// KillCam
// ---------------------------------------------------------------------------

/// A slow-motion replay sequence for dramatic moments (kills, goals, etc.).
///
/// The kill-cam controls the playback speed over time using a curve:
///
/// ```text
/// Speed ^
///       |  initial_speed
///       |──────┐
///       |      │
///       |      └── slow_speed (ramp down)
///       |          ─────────── (hold at slow speed)
///       |                    ┌── final_speed (ramp up)
///       |                    │
///       └────────────────────┘──────> Time
/// ```
///
/// The camera can also be interpolated from the player's perspective to a
/// cinematic angle.
#[derive(Debug, Clone)]
pub struct KillCam {
    /// Total duration of the kill-cam in seconds.
    pub duration: f32,
    /// Current elapsed time.
    pub elapsed: f32,
    /// Speed during the initial phase.
    pub initial_speed: f64,
    /// Speed during the slow-motion phase.
    pub slow_speed: f64,
    /// Speed during the final phase (often 1.0 for normal speed).
    pub final_speed: f64,
    /// Duration of the initial phase (seconds).
    pub initial_duration: f32,
    /// Duration of the slow-motion phase (seconds).
    pub slow_duration: f32,
    /// Duration of the ramp-down transition (seconds).
    pub ramp_down_duration: f32,
    /// Duration of the ramp-up transition (seconds).
    pub ramp_up_duration: f32,
    /// Current state of the kill-cam.
    pub state: KillCamState,
    /// Camera start position (attacker/player perspective).
    pub camera_start: Vec3,
    /// Camera end position (cinematic angle).
    pub camera_end: Vec3,
    /// Camera look-at target (typically the victim).
    pub camera_target: Vec3,
    /// Entity ID of the attacker (for camera tracking).
    pub attacker_entity: Option<u32>,
    /// Entity ID of the victim.
    pub victim_entity: Option<u32>,
}

/// Phase of the kill-cam sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KillCamState {
    /// Playing at initial speed.
    Initial,
    /// Ramping down to slow motion.
    RampDown,
    /// Slow-motion phase.
    SlowMotion,
    /// Ramping back up to normal speed.
    RampUp,
    /// Playing at final speed.
    Final,
    /// Kill-cam complete.
    Complete,
}

impl KillCam {
    /// Create a new kill-cam with default settings.
    pub fn new(duration: f32) -> Self {
        let initial = duration * 0.15;
        let slow = duration * 0.5;
        let ramp_down = duration * 0.1;
        let ramp_up = duration * 0.1;

        Self {
            duration,
            elapsed: 0.0,
            initial_speed: 1.0,
            slow_speed: 0.2,
            final_speed: 1.0,
            initial_duration: initial,
            slow_duration: slow,
            ramp_down_duration: ramp_down,
            ramp_up_duration: ramp_up,
            state: KillCamState::Initial,
            camera_start: Vec3::ZERO,
            camera_end: Vec3::ZERO,
            camera_target: Vec3::ZERO,
            attacker_entity: None,
            victim_entity: None,
        }
    }

    /// Set the camera positions.
    pub fn with_camera(
        mut self,
        start: Vec3,
        end: Vec3,
        target: Vec3,
    ) -> Self {
        self.camera_start = start;
        self.camera_end = end;
        self.camera_target = target;
        self
    }

    /// Set the attacker and victim entities.
    pub fn with_entities(mut self, attacker: u32, victim: u32) -> Self {
        self.attacker_entity = Some(attacker);
        self.victim_entity = Some(victim);
        self
    }

    /// Set the slow-motion speed.
    pub fn with_slow_speed(mut self, speed: f64) -> Self {
        self.slow_speed = speed;
        self
    }

    /// Get the current playback speed based on the kill-cam phase.
    pub fn current_speed(&self) -> f64 {
        match self.state {
            KillCamState::Initial => self.initial_speed,
            KillCamState::RampDown => {
                let progress = self.phase_progress();
                let t = smooth_step(progress);
                lerp_f64(self.initial_speed, self.slow_speed, t)
            }
            KillCamState::SlowMotion => self.slow_speed,
            KillCamState::RampUp => {
                let progress = self.phase_progress();
                let t = smooth_step(progress);
                lerp_f64(self.slow_speed, self.final_speed, t)
            }
            KillCamState::Final => self.final_speed,
            KillCamState::Complete => 1.0,
        }
    }

    /// Get the current camera position (interpolated).
    pub fn camera_position(&self) -> Vec3 {
        let t = self.elapsed / self.duration.max(0.001);
        let t = smooth_step(t.clamp(0.0, 1.0));
        Vec3::lerp(self.camera_start, self.camera_end, t)
    }

    /// Update the kill-cam by `dt` seconds. Returns `true` if complete.
    pub fn update(&mut self, dt: f32) -> bool {
        self.elapsed += dt;

        // Determine phase boundaries.
        let ramp_down_start = self.initial_duration;
        let slow_start = ramp_down_start + self.ramp_down_duration;
        let ramp_up_start = slow_start + self.slow_duration;
        let final_start = ramp_up_start + self.ramp_up_duration;

        self.state = if self.elapsed < ramp_down_start {
            KillCamState::Initial
        } else if self.elapsed < slow_start {
            KillCamState::RampDown
        } else if self.elapsed < ramp_up_start {
            KillCamState::SlowMotion
        } else if self.elapsed < final_start {
            KillCamState::RampUp
        } else if self.elapsed < self.duration {
            KillCamState::Final
        } else {
            KillCamState::Complete
        };

        self.state == KillCamState::Complete
    }

    /// Check if the kill-cam is complete.
    pub fn is_complete(&self) -> bool {
        self.state == KillCamState::Complete
    }

    /// Progress within the current phase (0.0 to 1.0).
    fn phase_progress(&self) -> f32 {
        let ramp_down_start = self.initial_duration;
        let slow_start = ramp_down_start + self.ramp_down_duration;
        let ramp_up_start = slow_start + self.slow_duration;
        let final_start = ramp_up_start + self.ramp_up_duration;

        match self.state {
            KillCamState::Initial => {
                if self.initial_duration > 0.0 {
                    self.elapsed / self.initial_duration
                } else {
                    1.0
                }
            }
            KillCamState::RampDown => {
                if self.ramp_down_duration > 0.0 {
                    (self.elapsed - ramp_down_start) / self.ramp_down_duration
                } else {
                    1.0
                }
            }
            KillCamState::SlowMotion => {
                if self.slow_duration > 0.0 {
                    (self.elapsed - slow_start) / self.slow_duration
                } else {
                    1.0
                }
            }
            KillCamState::RampUp => {
                if self.ramp_up_duration > 0.0 {
                    (self.elapsed - ramp_up_start) / self.ramp_up_duration
                } else {
                    1.0
                }
            }
            KillCamState::Final => {
                let final_duration = self.duration - final_start;
                if final_duration > 0.0 {
                    (self.elapsed - final_start) / final_duration
                } else {
                    1.0
                }
            }
            KillCamState::Complete => 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Smooth step interpolation (cubic Hermite).
fn smooth_step(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Linear interpolation for f64.
fn lerp_f64(a: f64, b: f64, t: f32) -> f64 {
    a + (b - a) * t as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorder_lifecycle() {
        let settings = ReplaySettings {
            checkpoint_interval: 10,
            ..Default::default()
        };
        let mut recorder = ReplayRecorder::new(settings);

        assert_eq!(recorder.state(), RecordingState::Idle);

        recorder.start_recording();
        assert_eq!(recorder.state(), RecordingState::Recording);

        // Record a few frames.
        for frame in 0..20 {
            recorder.begin_frame(frame, frame as f64 / 60.0, 1.0 / 60.0);

            if frame == 5 {
                recorder.record_input(InputEvent::KeyDown {
                    key_code: 32,
                    is_repeat: false,
                });
            }

            if frame == 10 {
                recorder.record_entity_event(
                    1,
                    EntityEventKind::Spawned {
                        components: HashMap::new(),
                        position: [100.0, 0.0, 50.0],
                        rotation: [0.0, 0.0, 0.0, 1.0],
                    },
                );
            }

            recorder.end_frame();
        }

        let replay = recorder.stop_recording();
        assert_eq!(replay.header.frame_count, 19);
        assert!(replay.input_frames.len() >= 1);
        assert_eq!(replay.entity_events.len(), 1);
        assert!(replay.checkpoints.len() >= 1); // At least one checkpoint
    }

    #[test]
    fn player_load_and_seek() {
        let mut data = ReplayData::new();
        data.header.frame_count = 100;
        data.header.duration = 100.0 / 60.0;

        // Add a checkpoint at frame 0.
        data.checkpoints.push(Checkpoint::new(0, 0.0));
        // Add a checkpoint at frame 50.
        data.checkpoints.push(Checkpoint::new(50, 50.0 / 60.0));

        let mut player = ReplayPlayer::new();
        player.load(data).unwrap();

        assert_eq!(player.playback_state(), PlaybackState::Loaded);
        assert_eq!(player.frame_count(), 100);

        // Seek to frame 75: should use checkpoint at 50.
        let result = player.seek(75).unwrap();
        assert_eq!(result.target_frame, 75);
        assert_eq!(result.checkpoint_frame, 50);
    }

    #[test]
    fn player_playback() {
        let mut data = ReplayData::new();
        data.header.frame_count = 10;
        data.header.duration = 10.0 / 60.0;
        data.input_frames.push({
            let mut f = InputFrame::new(5, 5.0 / 60.0, 1.0 / 60.0);
            f.add_event(InputEvent::KeyDown {
                key_code: 65,
                is_repeat: false,
            });
            f
        });

        let mut player = ReplayPlayer::new();
        player.load(data).unwrap();
        player.play().unwrap();

        // Advance 10 frames worth of time.
        for _ in 0..10 {
            let frame = player.update(1.0 / 60.0);
            // At frame 5 we should get the key event.
            if frame.frame == 5 && !frame.input_events.is_empty() {
                assert_eq!(frame.input_events.len(), 1);
            }
        }
    }

    #[test]
    fn kill_cam_speed_curve() {
        let mut kc = KillCam::new(3.0).with_slow_speed(0.1);

        // Initial phase.
        kc.update(0.1);
        assert!(kc.current_speed() >= 0.9);

        // Ramp down.
        kc.update(0.5);
        let speed = kc.current_speed();
        assert!(speed < 1.0);

        // Advance to completion.
        kc.update(3.0);
        assert!(kc.is_complete());
    }

    #[test]
    fn replay_data_round_trip_json() {
        let mut data = ReplayData::new();
        data.header.game_mode = "deathmatch".to_owned();
        data.header.frame_count = 5;
        data.checkpoints.push(Checkpoint::new(0, 0.0));

        let json = serde_json::to_string(&data).unwrap();
        let loaded: ReplayData = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.header.game_mode, "deathmatch");
        assert_eq!(loaded.checkpoints.len(), 1);
    }

    #[test]
    fn seek_out_of_range() {
        let mut data = ReplayData::new();
        data.header.frame_count = 50;

        let mut player = ReplayPlayer::new();
        player.load(data).unwrap();

        let result = player.seek(100);
        assert!(result.is_err());
    }

    #[test]
    fn custom_event_recording() {
        let mut recorder = ReplayRecorder::new(ReplaySettings {
            checkpoint_interval: 100,
            ..Default::default()
        });
        recorder.start_recording();

        recorder.begin_frame(0, 0.0, 1.0 / 60.0);
        if let Some(event) = recorder.record_custom_event("player_kill") {
            *event = event
                .clone()
                .with_property("weapon", "rifle")
                .with_entity(42);
        }
        recorder.end_frame();

        let replay = recorder.stop_recording();
        assert_eq!(replay.custom_events.len(), 1);
        assert_eq!(replay.custom_events[0].name, "player_kill");
    }

    #[test]
    fn checkpoint_entity_snapshot() {
        let mut cp = Checkpoint::new(0, 0.0);
        cp.add_entity(CheckpointEntity {
            entity_id: 1,
            generation: 0,
            components: {
                let mut m = HashMap::new();
                m.insert("Position".to_owned(), vec![0, 0, 0, 0]);
                m
            },
            position: [10.0, 0.0, 20.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            name: "player".to_owned(),
        });

        assert!(cp.entities.contains_key(&1));
    }
}
