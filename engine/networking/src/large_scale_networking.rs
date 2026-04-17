//! # Large-Scale Networking
//!
//! Fortnite-scale authoritative server networking supporting 100+ concurrent
//! players with high-tick-rate simulation, full client-side prediction with
//! server reconciliation, interest management, property replication, bandwidth
//! management, anti-cheat validation, and visual smoothing.
//!
//! ## Architecture
//!
//! ```text
//! Client                          Server
//! ------                          ------
//! Input -> InputBuffer            <-- ServerTick loop (60Hz)
//!   |                                  |
//!   v                                  v
//! ClientPrediction               GameServer
//!   |  (apply input locally)       |  (authoritative simulation)
//!   |                              v
//!   |                          InterestManagement
//!   |                              |
//!   |                          PropertyReplication
//!   |                              |  (serialize changed props)
//!   |                              v
//!   |                          BandwidthManager
//!   |                              |  (prioritize packets)
//!   |                              v
//!   |   <---- state snapshot ------+
//!   v
//! ServerReconciliation
//!   |  (compare predicted vs actual)
//!   v
//! NetworkPredictionSmoothing
//!   |  (visual interpolation)
//!   v
//! Rendered position
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ===========================================================================
// Constants
// ===========================================================================

/// Default server tick rate (Hz).
pub const DEFAULT_TICK_RATE: u32 = 60;

/// Minimum tick rate.
pub const MIN_TICK_RATE: u32 = 20;

/// Maximum tick rate.
pub const MAX_TICK_RATE: u32 = 128;

/// Default maximum connected clients.
pub const DEFAULT_MAX_CLIENTS: usize = 100;

/// Default input buffer size (ticks of buffered input).
pub const DEFAULT_INPUT_BUFFER_SIZE: usize = 128;

/// Small correction threshold (units) -- below this, ignore mismatch.
pub const CORRECTION_IGNORE_THRESHOLD: f32 = 0.1;

/// Medium correction threshold -- smooth correction.
pub const CORRECTION_SMOOTH_THRESHOLD: f32 = 1.0;

/// Large correction threshold -- snap to correct position.
pub const CORRECTION_SNAP_THRESHOLD: f32 = 5.0;

/// Default target bandwidth per client (bytes/s).
pub const DEFAULT_BANDWIDTH_TARGET: u32 = 50_000; // 50 KB/s

/// Default relevancy grid cell size.
pub const DEFAULT_RELEVANCY_CELL_SIZE: f32 = 100.0;

/// Default maximum relevancy distance.
pub const DEFAULT_MAX_RELEVANCY_DISTANCE: f32 = 500.0;

/// Default smooth rate for visual position correction.
pub const DEFAULT_SMOOTH_RATE: f32 = 10.0;

/// Speed hack detection: max units per tick.
pub const SPEED_HACK_MAX_UNITS_PER_TICK: f32 = 50.0;

/// Aimbot detection: max suspicious accurate shots in window.
pub const AIMBOT_MAX_ACCURATE_IN_WINDOW: u32 = 20;

/// Aimbot detection window size in ticks.
pub const AIMBOT_WINDOW_TICKS: u32 = 600; // 10 seconds at 60Hz

/// Maximum input rate (inputs per second) before rate limiting.
pub const MAX_INPUT_RATE: u32 = 128;

/// Default property quantization precision for positions.
pub const POSITION_QUANTIZE_PRECISION: f32 = 0.01;

/// Default rotation quantization precision (degrees).
pub const ROTATION_QUANTIZE_PRECISION: f32 = 0.1;

/// Maximum sequence number before wrapping.
pub const MAX_SEQUENCE_NUMBER: u32 = u32::MAX;

/// Maximum replicated properties per entity.
pub const MAX_PROPERTIES_PER_ENTITY: usize = 64;

/// Default dormancy timeout (ticks without change).
pub const DEFAULT_DORMANCY_TIMEOUT: u32 = 300; // 5 seconds at 60Hz

// ===========================================================================
// Vec3Net — lightweight 3D vector for networking
// ===========================================================================

/// Lightweight 3D vector used in networking (avoids glam dependency in net layer).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3Net {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3Net {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn distance_sq(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }

    #[inline]
    pub fn distance(&self, other: &Self) -> f32 {
        self.distance_sq(other).sqrt()
    }

    #[inline]
    pub fn length_sq(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    #[inline]
    pub fn length(&self) -> f32 {
        self.length_sq().sqrt()
    }

    #[inline]
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        Self {
            x: a.x + (b.x - a.x) * t,
            y: a.y + (b.y - a.y) * t,
            z: a.z + (b.z - a.z) * t,
        }
    }

    /// Quantize to the given precision (snap to grid).
    #[inline]
    pub fn quantize(&self, precision: f32) -> Self {
        let inv = 1.0 / precision;
        Self {
            x: (self.x * inv).round() * precision,
            y: (self.y * inv).round() * precision,
            z: (self.z * inv).round() * precision,
        }
    }

    /// Add.
    #[inline]
    pub fn add(&self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    /// Subtract.
    #[inline]
    pub fn sub(&self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    /// Scale.
    #[inline]
    pub fn scale(&self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}

impl Default for Vec3Net {
    fn default() -> Self {
        Self::ZERO
    }
}

impl fmt::Display for Vec3Net {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.2}, {:.2}, {:.2})", self.x, self.y, self.z)
    }
}

// ===========================================================================
// ClientId / EntityId types
// ===========================================================================

/// Unique client identifier assigned by the server.
pub type NetClientId = u32;

/// Network entity identifier.
pub type NetEntityId = u64;

/// Server tick number.
pub type TickNumber = u32;

// ===========================================================================
// ServerTick — deterministic simulation step
// ===========================================================================

/// Represents a single deterministic server simulation tick.
#[derive(Debug, Clone)]
pub struct ServerTick {
    /// Tick number (monotonically increasing).
    pub tick: TickNumber,
    /// Fixed delta time for this tick (1/tick_rate).
    pub dt: f32,
    /// Wall-clock time when this tick was processed.
    pub wall_time: f64,
    /// Number of clients connected during this tick.
    pub client_count: u32,
    /// Number of entities simulated.
    pub entity_count: u32,
}

impl ServerTick {
    /// Create a new server tick.
    pub fn new(tick: TickNumber, tick_rate: u32) -> Self {
        Self {
            tick,
            dt: 1.0 / tick_rate as f32,
            wall_time: 0.0,
            client_count: 0,
            entity_count: 0,
        }
    }
}

// ===========================================================================
// PlayerInput — client input data
// ===========================================================================

/// Represents a single frame of client input.
#[derive(Debug, Clone)]
pub struct PlayerInput {
    /// Input sequence number (client-assigned, monotonically increasing).
    pub sequence: u32,
    /// Server tick this input is targeting.
    pub target_tick: TickNumber,
    /// Movement direction (normalized).
    pub move_direction: Vec3Net,
    /// Look direction / aim direction.
    pub look_direction: Vec3Net,
    /// Button state as a bitfield.
    pub buttons: u64,
    /// Timestamp when the input was created (client clock).
    pub client_timestamp: f64,
    /// Whether this input has been acknowledged by the server.
    pub acknowledged: bool,
}

impl PlayerInput {
    /// Create a new input.
    pub fn new(sequence: u32, target_tick: TickNumber) -> Self {
        Self {
            sequence,
            target_tick,
            move_direction: Vec3Net::ZERO,
            look_direction: Vec3Net::new(0.0, 0.0, 1.0),
            buttons: 0,
            client_timestamp: 0.0,
            acknowledged: false,
        }
    }

    /// Check if a specific button is pressed.
    #[inline]
    pub fn is_button_pressed(&self, button_index: u8) -> bool {
        (self.buttons >> button_index) & 1 == 1
    }

    /// Set a button state.
    #[inline]
    pub fn set_button(&mut self, button_index: u8, pressed: bool) {
        if pressed {
            self.buttons |= 1 << button_index;
        } else {
            self.buttons &= !(1 << button_index);
        }
    }

    /// Approximate movement speed from the input direction.
    pub fn movement_speed(&self) -> f32 {
        self.move_direction.length()
    }
}

// ===========================================================================
// InputRingBuffer — circular buffer for client inputs
// ===========================================================================

/// Ring buffer for storing client inputs with sequence numbers.
#[derive(Debug)]
pub struct InputRingBuffer {
    /// Buffer storage.
    buffer: VecDeque<PlayerInput>,
    /// Maximum capacity.
    capacity: usize,
    /// Last acknowledged sequence number.
    last_ack_sequence: u32,
    /// Total inputs received.
    total_received: u64,
    /// Total inputs dropped (buffer overflow).
    total_dropped: u64,
}

impl InputRingBuffer {
    /// Create a new input ring buffer.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            last_ack_sequence: 0,
            total_received: 0,
            total_dropped: 0,
        }
    }

    /// Push a new input into the buffer.
    pub fn push(&mut self, input: PlayerInput) {
        self.total_received += 1;
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
            self.total_dropped += 1;
        }
        self.buffer.push_back(input);
    }

    /// Get all unacknowledged inputs (sequence > last_ack_sequence).
    pub fn unacknowledged_inputs(&self) -> Vec<&PlayerInput> {
        self.buffer
            .iter()
            .filter(|i| i.sequence > self.last_ack_sequence && !i.acknowledged)
            .collect()
    }

    /// Mark all inputs up to and including `sequence` as acknowledged.
    pub fn acknowledge_up_to(&mut self, sequence: u32) {
        self.last_ack_sequence = sequence;
        for input in self.buffer.iter_mut() {
            if input.sequence <= sequence {
                input.acknowledged = true;
            }
        }
        // Remove old acknowledged inputs to save memory.
        while let Some(front) = self.buffer.front() {
            if front.acknowledged {
                self.buffer.pop_front();
            } else {
                break;
            }
        }
    }

    /// Get the most recent input.
    pub fn latest(&self) -> Option<&PlayerInput> {
        self.buffer.back()
    }

    /// Get an input by sequence number.
    pub fn get_by_sequence(&self, sequence: u32) -> Option<&PlayerInput> {
        self.buffer.iter().find(|i| i.sequence == sequence)
    }

    /// Current buffer occupancy.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.last_ack_sequence = 0;
    }

    /// Get all inputs for replay after a reconciliation.
    pub fn inputs_after_tick(&self, tick: TickNumber) -> Vec<&PlayerInput> {
        self.buffer
            .iter()
            .filter(|i| i.target_tick > tick)
            .collect()
    }
}

// ===========================================================================
// EntityState — snapshot of entity state at a tick
// ===========================================================================

/// Snapshot of an entity's replicated state at a specific tick.
#[derive(Debug, Clone)]
pub struct NetEntityState {
    /// Entity network ID.
    pub entity_id: NetEntityId,
    /// Server tick this state corresponds to.
    pub tick: TickNumber,
    /// Position.
    pub position: Vec3Net,
    /// Velocity.
    pub velocity: Vec3Net,
    /// Rotation (yaw, pitch, roll in degrees).
    pub rotation: Vec3Net,
    /// Angular velocity.
    pub angular_velocity: Vec3Net,
    /// Health (game-specific but commonly needed).
    pub health: f32,
    /// Custom state flags.
    pub state_flags: u64,
    /// Custom replicated properties (opaque bytes).
    pub custom_data: Vec<u8>,
}

impl NetEntityState {
    /// Create a new entity state.
    pub fn new(entity_id: NetEntityId, tick: TickNumber) -> Self {
        Self {
            entity_id,
            tick,
            position: Vec3Net::ZERO,
            velocity: Vec3Net::ZERO,
            rotation: Vec3Net::ZERO,
            angular_velocity: Vec3Net::ZERO,
            health: 100.0,
            state_flags: 0,
            custom_data: Vec::new(),
        }
    }

    /// Compute position error between this state and another.
    pub fn position_error(&self, other: &NetEntityState) -> f32 {
        self.position.distance(&other.position)
    }

    /// Compute velocity error.
    pub fn velocity_error(&self, other: &NetEntityState) -> f32 {
        self.velocity.distance(&other.velocity)
    }

    /// Interpolate between two entity states.
    pub fn interpolate(a: &NetEntityState, b: &NetEntityState, t: f32) -> NetEntityState {
        NetEntityState {
            entity_id: a.entity_id,
            tick: if t < 0.5 { a.tick } else { b.tick },
            position: Vec3Net::lerp(a.position, b.position, t),
            velocity: Vec3Net::lerp(a.velocity, b.velocity, t),
            rotation: Vec3Net::lerp(a.rotation, b.rotation, t),
            angular_velocity: Vec3Net::lerp(a.angular_velocity, b.angular_velocity, t),
            health: a.health + (b.health - a.health) * t,
            state_flags: if t < 0.5 { a.state_flags } else { b.state_flags },
            custom_data: if t < 0.5 {
                a.custom_data.clone()
            } else {
                b.custom_data.clone()
            },
        }
    }
}

// ===========================================================================
// ReconciliationResult — outcome of comparing predicted vs actual state
// ===========================================================================

/// Result of comparing the client's predicted state with the server's authoritative state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReconciliationResult {
    /// States match (within tolerance). No correction needed.
    Match,
    /// Small difference. Smoothly interpolate toward correct position.
    SmallCorrection {
        position_error: f32,
        velocity_error: f32,
    },
    /// Large difference. Snap partway toward correct position.
    LargeCorrection {
        position_error: f32,
        velocity_error: f32,
    },
    /// Impossibly large difference (teleport/desync). Snap instantly.
    Teleport {
        position_error: f32,
    },
}

impl ReconciliationResult {
    /// Whether any correction is needed.
    pub fn needs_correction(&self) -> bool {
        !matches!(self, ReconciliationResult::Match)
    }

    /// Whether the correction requires replaying inputs.
    pub fn needs_replay(&self) -> bool {
        matches!(
            self,
            ReconciliationResult::LargeCorrection { .. } | ReconciliationResult::Teleport { .. }
        )
    }
}

// ===========================================================================
// ClientPrediction — full client-side prediction system
// ===========================================================================

/// Manages client-side prediction for a single player entity.
///
/// The client predicts its own movement locally by applying inputs immediately,
/// then reconciles with authoritative server state when it arrives.
#[derive(Debug)]
pub struct ClientPrediction {
    /// The entity being predicted.
    pub entity_id: NetEntityId,
    /// Input ring buffer.
    pub input_buffer: InputRingBuffer,
    /// History of predicted states (one per tick, for reconciliation).
    predicted_states: VecDeque<NetEntityState>,
    /// Maximum predicted state history size.
    max_history: usize,
    /// Current predicted state.
    pub current_state: NetEntityState,
    /// Last acknowledged server state.
    pub last_server_state: Option<NetEntityState>,
    /// Prediction error accumulator (for smooth correction).
    pub prediction_error: Vec3Net,
    /// Maximum tolerable prediction error before forcing a snap.
    pub max_error_budget: f32,
    /// Whether prediction is enabled.
    pub enabled: bool,
    /// Statistics.
    pub stats: PredictionStats,
}

/// Prediction statistics.
#[derive(Debug, Clone, Default)]
pub struct PredictionStats {
    /// Total predictions made.
    pub total_predictions: u64,
    /// Total reconciliations performed.
    pub total_reconciliations: u64,
    /// Matches (no correction needed).
    pub matches: u64,
    /// Small corrections.
    pub small_corrections: u64,
    /// Large corrections.
    pub large_corrections: u64,
    /// Teleports.
    pub teleports: u64,
    /// Average prediction error.
    pub avg_error: f32,
    /// Maximum prediction error observed.
    pub max_error: f32,
    /// Input replays performed.
    pub input_replays: u64,
}

impl ClientPrediction {
    /// Create a new client prediction system.
    pub fn new(entity_id: NetEntityId) -> Self {
        Self {
            entity_id,
            input_buffer: InputRingBuffer::new(DEFAULT_INPUT_BUFFER_SIZE),
            predicted_states: VecDeque::with_capacity(128),
            max_history: 128,
            current_state: NetEntityState::new(entity_id, 0),
            last_server_state: None,
            prediction_error: Vec3Net::ZERO,
            max_error_budget: CORRECTION_SNAP_THRESHOLD,
            enabled: true,
            stats: PredictionStats::default(),
        }
    }

    /// Record a new input and predict the next state.
    ///
    /// `simulate_fn` takes the current state and input, and returns the predicted
    /// next state. This is the client's local simulation function.
    pub fn predict(
        &mut self,
        input: PlayerInput,
        simulate_fn: impl FnOnce(&NetEntityState, &PlayerInput) -> NetEntityState,
    ) {
        if !self.enabled {
            return;
        }

        self.stats.total_predictions += 1;

        // Apply the input to produce the predicted next state.
        let predicted = simulate_fn(&self.current_state, &input);

        // Store the predicted state for later reconciliation.
        if self.predicted_states.len() >= self.max_history {
            self.predicted_states.pop_front();
        }
        self.predicted_states.push_back(predicted.clone());

        // Buffer the input.
        self.input_buffer.push(input);

        // Update current state to the predicted one.
        self.current_state = predicted;
    }

    /// Reconcile with an authoritative server state.
    ///
    /// Compares the server state with what we predicted for that tick, and
    /// determines the correction needed. If correction requires it, replays
    /// all unacknowledged inputs from the server state.
    pub fn reconcile(
        &mut self,
        server_state: &NetEntityState,
        simulate_fn: impl Fn(&NetEntityState, &PlayerInput) -> NetEntityState,
    ) -> ReconciliationResult {
        self.stats.total_reconciliations += 1;

        // Acknowledge inputs up to the server's tick.
        self.input_buffer.acknowledge_up_to(server_state.tick);

        // Find our predicted state for the same tick.
        let predicted_at_tick = self
            .predicted_states
            .iter()
            .find(|s| s.tick == server_state.tick)
            .cloned();

        let result = if let Some(predicted) = predicted_at_tick {
            let pos_error = predicted.position_error(server_state);
            let vel_error = predicted.velocity_error(server_state);

            if pos_error < CORRECTION_IGNORE_THRESHOLD {
                self.stats.matches += 1;
                ReconciliationResult::Match
            } else if pos_error < CORRECTION_SMOOTH_THRESHOLD {
                self.stats.small_corrections += 1;
                ReconciliationResult::SmallCorrection {
                    position_error: pos_error,
                    velocity_error: vel_error,
                }
            } else if pos_error < CORRECTION_SNAP_THRESHOLD {
                self.stats.large_corrections += 1;
                ReconciliationResult::LargeCorrection {
                    position_error: pos_error,
                    velocity_error: vel_error,
                }
            } else {
                self.stats.teleports += 1;
                ReconciliationResult::Teleport {
                    position_error: pos_error,
                }
            }
        } else {
            // No predicted state found for this tick -- treat as teleport.
            self.stats.teleports += 1;
            ReconciliationResult::Teleport {
                position_error: server_state.position.distance(&self.current_state.position),
            }
        };

        // Store the server state.
        self.last_server_state = Some(server_state.clone());

        // Apply correction based on result.
        match result {
            ReconciliationResult::Match => {
                // Nothing to do.
            }
            ReconciliationResult::SmallCorrection {
                position_error, ..
            } => {
                // Accumulate the error for smooth visual correction.
                self.prediction_error = server_state.position.sub(self.current_state.position);
                self.stats.avg_error =
                    self.stats.avg_error * 0.95 + position_error * 0.05;
                if position_error > self.stats.max_error {
                    self.stats.max_error = position_error;
                }
            }
            ReconciliationResult::LargeCorrection { .. }
            | ReconciliationResult::Teleport { .. } => {
                // Rollback to server state and replay unacknowledged inputs.
                self.replay_from_server_state(server_state, &simulate_fn);
            }
        }

        result
    }

    /// Replay all unacknowledged inputs from a known-good server state.
    fn replay_from_server_state(
        &mut self,
        server_state: &NetEntityState,
        simulate_fn: &impl Fn(&NetEntityState, &PlayerInput) -> NetEntityState,
    ) {
        self.stats.input_replays += 1;

        // Start from the server's authoritative state.
        let mut state = server_state.clone();

        // Get all inputs that haven't been acknowledged (inputs after server tick).
        let replay_inputs: Vec<PlayerInput> = self
            .input_buffer
            .inputs_after_tick(server_state.tick)
            .into_iter()
            .cloned()
            .collect();

        // Clear predicted history.
        self.predicted_states.clear();

        // Replay each input.
        for input in &replay_inputs {
            state = simulate_fn(&state, input);
            self.predicted_states.push_back(state.clone());
        }

        // Set current state to the result of replay.
        self.prediction_error = state.position.sub(self.current_state.position);
        self.current_state = state;
    }

    /// Get the visual position (with smooth correction applied).
    pub fn visual_position(&self) -> Vec3Net {
        self.current_state.position.add(self.prediction_error)
    }

    /// Reset prediction state.
    pub fn reset(&mut self) {
        self.input_buffer.clear();
        self.predicted_states.clear();
        self.prediction_error = Vec3Net::ZERO;
        self.last_server_state = None;
    }
}

// ===========================================================================
// ServerReconciliation — server-side reconciliation tracking
// ===========================================================================

/// Tracks reconciliation state for all connected clients on the server side.
#[derive(Debug)]
pub struct ServerReconciliation {
    /// Per-client latest acknowledged tick.
    client_acks: HashMap<NetClientId, TickNumber>,
    /// Per-client latest input sequence.
    client_input_sequences: HashMap<NetClientId, u32>,
    /// Reconciliation thresholds.
    pub ignore_threshold: f32,
    pub smooth_threshold: f32,
    pub snap_threshold: f32,
}

impl ServerReconciliation {
    /// Create a new server reconciliation tracker.
    pub fn new() -> Self {
        Self {
            client_acks: HashMap::new(),
            client_input_sequences: HashMap::new(),
            ignore_threshold: CORRECTION_IGNORE_THRESHOLD,
            smooth_threshold: CORRECTION_SMOOTH_THRESHOLD,
            snap_threshold: CORRECTION_SNAP_THRESHOLD,
        }
    }

    /// Record that a client has acknowledged up to a given tick.
    pub fn record_ack(&mut self, client_id: NetClientId, tick: TickNumber) {
        let entry = self.client_acks.entry(client_id).or_insert(0);
        if tick > *entry {
            *entry = tick;
        }
    }

    /// Record latest input sequence from a client.
    pub fn record_input_sequence(&mut self, client_id: NetClientId, sequence: u32) {
        let entry = self.client_input_sequences.entry(client_id).or_insert(0);
        if sequence > *entry {
            *entry = sequence;
        }
    }

    /// Get the last acknowledged tick for a client.
    pub fn last_ack(&self, client_id: NetClientId) -> TickNumber {
        self.client_acks.get(&client_id).copied().unwrap_or(0)
    }

    /// Classify a position error.
    pub fn classify_error(&self, position_error: f32) -> ReconciliationResult {
        if position_error < self.ignore_threshold {
            ReconciliationResult::Match
        } else if position_error < self.smooth_threshold {
            ReconciliationResult::SmallCorrection {
                position_error,
                velocity_error: 0.0,
            }
        } else if position_error < self.snap_threshold {
            ReconciliationResult::LargeCorrection {
                position_error,
                velocity_error: 0.0,
            }
        } else {
            ReconciliationResult::Teleport { position_error }
        }
    }

    /// Remove a client's tracking data.
    pub fn remove_client(&mut self, client_id: NetClientId) {
        self.client_acks.remove(&client_id);
        self.client_input_sequences.remove(&client_id);
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.client_acks.clear();
        self.client_input_sequences.clear();
    }
}

impl Default for ServerReconciliation {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// RelevancyGrid — spatial grid for proximity-based relevance
// ===========================================================================

/// An entity's position in the relevancy grid.
#[derive(Debug, Clone)]
struct RelevancyEntry {
    entity_id: NetEntityId,
    position: Vec3Net,
    owner_client: Option<NetClientId>,
    relevancy_override: Option<f32>,
}

/// Cell coordinate in the relevancy grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RelevancyCell {
    x: i32,
    z: i32,
}

/// Spatial grid for determining which entities are relevant to each client.
#[derive(Debug)]
pub struct RelevancyGrid {
    /// Cell size.
    pub cell_size: f32,
    /// Maximum relevancy distance.
    pub max_distance: f32,
    /// Grid cells -> entity lists.
    cells: HashMap<RelevancyCell, Vec<RelevancyEntry>>,
    /// Entity -> cell mapping for fast removal.
    entity_cells: HashMap<NetEntityId, RelevancyCell>,
}

impl RelevancyGrid {
    /// Create a new relevancy grid.
    pub fn new(cell_size: f32, max_distance: f32) -> Self {
        Self {
            cell_size,
            max_distance,
            cells: HashMap::new(),
            entity_cells: HashMap::new(),
        }
    }

    /// Create with default settings.
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_RELEVANCY_CELL_SIZE, DEFAULT_MAX_RELEVANCY_DISTANCE)
    }

    /// Convert a position to a cell coordinate.
    fn pos_to_cell(&self, pos: &Vec3Net) -> RelevancyCell {
        RelevancyCell {
            x: (pos.x / self.cell_size).floor() as i32,
            z: (pos.z / self.cell_size).floor() as i32,
        }
    }

    /// Insert or update an entity in the grid.
    pub fn update_entity(
        &mut self,
        entity_id: NetEntityId,
        position: Vec3Net,
        owner_client: Option<NetClientId>,
    ) {
        // Remove from old cell if it moved.
        if let Some(&old_cell) = self.entity_cells.get(&entity_id) {
            let new_cell = self.pos_to_cell(&position);
            if old_cell != new_cell {
                if let Some(entries) = self.cells.get_mut(&old_cell) {
                    entries.retain(|e| e.entity_id != entity_id);
                }
                self.entity_cells.insert(entity_id, new_cell);
                let entry = RelevancyEntry {
                    entity_id,
                    position,
                    owner_client,
                    relevancy_override: None,
                };
                self.cells.entry(new_cell).or_default().push(entry);
            } else {
                // Same cell, just update position.
                if let Some(entries) = self.cells.get_mut(&old_cell) {
                    if let Some(e) = entries.iter_mut().find(|e| e.entity_id == entity_id) {
                        e.position = position;
                        e.owner_client = owner_client;
                    }
                }
            }
        } else {
            // New entity.
            let cell = self.pos_to_cell(&position);
            self.entity_cells.insert(entity_id, cell);
            let entry = RelevancyEntry {
                entity_id,
                position,
                owner_client,
                relevancy_override: None,
            };
            self.cells.entry(cell).or_default().push(entry);
        }
    }

    /// Remove an entity from the grid.
    pub fn remove_entity(&mut self, entity_id: NetEntityId) {
        if let Some(cell) = self.entity_cells.remove(&entity_id) {
            if let Some(entries) = self.cells.get_mut(&cell) {
                entries.retain(|e| e.entity_id != entity_id);
            }
        }
    }

    /// Query all entities relevant to a given position within max_distance.
    ///
    /// Returns entities sorted by distance (nearest first).
    pub fn query_relevant(
        &self,
        position: &Vec3Net,
        max_results: usize,
    ) -> Vec<(NetEntityId, f32)> {
        let center_cell = self.pos_to_cell(position);
        let cell_radius = (self.max_distance / self.cell_size).ceil() as i32;
        let max_dist_sq = self.max_distance * self.max_distance;

        let mut results: Vec<(NetEntityId, f32)> = Vec::new();

        for dx in -cell_radius..=cell_radius {
            for dz in -cell_radius..=cell_radius {
                let cell = RelevancyCell {
                    x: center_cell.x + dx,
                    z: center_cell.z + dz,
                };
                if let Some(entries) = self.cells.get(&cell) {
                    for entry in entries {
                        let dist_sq = position.distance_sq(&entry.position);
                        if dist_sq <= max_dist_sq {
                            results.push((entry.entity_id, dist_sq.sqrt()));
                        }
                    }
                }
            }
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(max_results);
        results
    }

    /// Clear the grid.
    pub fn clear(&mut self) {
        self.cells.clear();
        self.entity_cells.clear();
    }

    /// Total entities tracked.
    pub fn entity_count(&self) -> usize {
        self.entity_cells.len()
    }
}

// ===========================================================================
// InterestManagement — controls what data each client receives
// ===========================================================================

/// Priority tier for entity relevance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RelevancyTier {
    /// Critical: always replicated (player's own entity).
    Critical = 0,
    /// High: replicated at maximum frequency (nearby enemies).
    High = 1,
    /// Medium: moderate update frequency.
    Medium = 2,
    /// Low: infrequent updates.
    Low = 3,
    /// Irrelevant: not replicated.
    Irrelevant = 4,
}

/// Bandwidth allocation per relevancy tier.
#[derive(Debug, Clone)]
pub struct RelevancyBudget {
    /// Fraction of bandwidth allocated to Critical tier.
    pub critical_fraction: f32,
    /// Fraction for High tier.
    pub high_fraction: f32,
    /// Fraction for Medium tier.
    pub medium_fraction: f32,
    /// Fraction for Low tier.
    pub low_fraction: f32,
}

impl Default for RelevancyBudget {
    fn default() -> Self {
        Self {
            critical_fraction: 0.3,
            high_fraction: 0.35,
            medium_fraction: 0.25,
            low_fraction: 0.1,
        }
    }
}

impl RelevancyBudget {
    /// Get the fraction for a tier.
    pub fn fraction_for_tier(&self, tier: RelevancyTier) -> f32 {
        match tier {
            RelevancyTier::Critical => self.critical_fraction,
            RelevancyTier::High => self.high_fraction,
            RelevancyTier::Medium => self.medium_fraction,
            RelevancyTier::Low => self.low_fraction,
            RelevancyTier::Irrelevant => 0.0,
        }
    }

    /// Get the byte budget for a tier given total bandwidth.
    pub fn bytes_for_tier(&self, tier: RelevancyTier, total_bytes: u32) -> u32 {
        (total_bytes as f32 * self.fraction_for_tier(tier)) as u32
    }
}

/// Custom relevancy rule callback type.
pub type RelevancyRuleFn = fn(entity_id: NetEntityId, client_id: NetClientId) -> Option<RelevancyTier>;

/// Manages per-client interest: decides which entities each client should
/// receive updates about and at what priority.
#[derive(Debug)]
pub struct InterestManager {
    /// Spatial relevancy grid.
    pub grid: RelevancyGrid,
    /// Per-client view position.
    client_positions: HashMap<NetClientId, Vec3Net>,
    /// Per-entity dormancy tracking: ticks since last change.
    dormancy_counters: HashMap<NetEntityId, u32>,
    /// Dormant entities (not replicated until they change).
    dormant_entities: HashSet<NetEntityId>,
    /// Dormancy timeout threshold.
    pub dormancy_timeout: u32,
    /// Custom relevancy rules per entity type.
    custom_rules: Vec<RelevancyRuleFn>,
    /// Bandwidth budget.
    pub budget: RelevancyBudget,
    /// Per-client relevant entity sets (cached).
    client_relevant: HashMap<NetClientId, Vec<(NetEntityId, RelevancyTier, f32)>>,
    /// Distance thresholds for relevancy tiers.
    pub tier_distances: [f32; 4], // Critical, High, Medium, Low
}

impl InterestManager {
    /// Create a new interest manager.
    pub fn new() -> Self {
        Self {
            grid: RelevancyGrid::with_defaults(),
            client_positions: HashMap::new(),
            dormancy_counters: HashMap::new(),
            dormant_entities: HashSet::new(),
            dormancy_timeout: DEFAULT_DORMANCY_TIMEOUT,
            custom_rules: Vec::new(),
            budget: RelevancyBudget::default(),
            client_relevant: HashMap::new(),
            tier_distances: [0.0, 50.0, 150.0, 350.0],
        }
    }

    /// Set a client's view position.
    pub fn set_client_position(&mut self, client_id: NetClientId, position: Vec3Net) {
        self.client_positions.insert(client_id, position);
    }

    /// Remove a client.
    pub fn remove_client(&mut self, client_id: NetClientId) {
        self.client_positions.remove(&client_id);
        self.client_relevant.remove(&client_id);
    }

    /// Update entity position in the spatial grid.
    pub fn update_entity(
        &mut self,
        entity_id: NetEntityId,
        position: Vec3Net,
        owner_client: Option<NetClientId>,
    ) {
        self.grid.update_entity(entity_id, position, owner_client);
    }

    /// Remove an entity.
    pub fn remove_entity(&mut self, entity_id: NetEntityId) {
        self.grid.remove_entity(entity_id);
        self.dormancy_counters.remove(&entity_id);
        self.dormant_entities.remove(&entity_id);
    }

    /// Mark an entity as changed (resets dormancy counter).
    pub fn mark_changed(&mut self, entity_id: NetEntityId) {
        self.dormancy_counters.insert(entity_id, 0);
        self.dormant_entities.remove(&entity_id);
    }

    /// Tick dormancy counters. Call once per server tick.
    pub fn tick_dormancy(&mut self) {
        let timeout = self.dormancy_timeout;
        let mut newly_dormant = Vec::new();

        for (entity_id, counter) in &mut self.dormancy_counters {
            *counter += 1;
            if *counter >= timeout && !self.dormant_entities.contains(entity_id) {
                newly_dormant.push(*entity_id);
            }
        }

        for entity_id in newly_dormant {
            self.dormant_entities.insert(entity_id);
        }
    }

    /// Determine the relevancy tier based on distance.
    fn tier_for_distance(&self, distance: f32) -> RelevancyTier {
        if distance <= self.tier_distances[1] {
            RelevancyTier::High
        } else if distance <= self.tier_distances[2] {
            RelevancyTier::Medium
        } else if distance <= self.tier_distances[3] {
            RelevancyTier::Low
        } else {
            RelevancyTier::Irrelevant
        }
    }

    /// Compute the relevant entity set for a client.
    ///
    /// Returns a list of (entity_id, tier, distance) sorted by priority.
    pub fn compute_relevancy(
        &mut self,
        client_id: NetClientId,
    ) -> Vec<(NetEntityId, RelevancyTier, f32)> {
        let client_pos = match self.client_positions.get(&client_id) {
            Some(pos) => *pos,
            None => return Vec::new(),
        };

        let nearby = self.grid.query_relevant(&client_pos, 500);
        let mut result: Vec<(NetEntityId, RelevancyTier, f32)> = Vec::new();

        for (entity_id, distance) in nearby {
            // Skip dormant entities.
            if self.dormant_entities.contains(&entity_id) {
                continue;
            }

            let mut tier = self.tier_for_distance(distance);

            // Check custom rules.
            for rule in &self.custom_rules {
                if let Some(override_tier) = rule(entity_id, client_id) {
                    tier = override_tier;
                    break;
                }
            }

            if tier != RelevancyTier::Irrelevant {
                result.push((entity_id, tier, distance));
            }
        }

        // Sort by tier then distance.
        result.sort_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
        });

        // Cache the result.
        self.client_relevant.insert(client_id, result.clone());

        result
    }

    /// Get the cached relevant entities for a client.
    pub fn cached_relevancy(
        &self,
        client_id: NetClientId,
    ) -> Option<&Vec<(NetEntityId, RelevancyTier, f32)>> {
        self.client_relevant.get(&client_id)
    }

    /// Add a custom relevancy rule.
    pub fn add_rule(&mut self, rule: RelevancyRuleFn) {
        self.custom_rules.push(rule);
    }

    /// Get dormant entity count.
    pub fn dormant_count(&self) -> usize {
        self.dormant_entities.len()
    }
}

impl Default for InterestManager {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// PropertyReplication — fine-grained state sync
// ===========================================================================

/// Replication condition for a property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicationCondition {
    /// Always replicate every tick.
    Always,
    /// Replicate only when changed.
    OnChange,
    /// Replicate only on initial spawn.
    InitialOnly,
    /// Never replicate (local only).
    Never,
    /// Replicate to owner only.
    OwnerOnly,
    /// Replicate to non-owners only.
    SkipOwner,
}

/// Reliability mode for a replicated property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropertyReliability {
    /// Reliable ordered delivery (health, inventory).
    Reliable,
    /// Unreliable delivery (position, velocity).
    Unreliable,
}

/// Definition of a replicated property.
#[derive(Debug, Clone)]
pub struct PropertyDef {
    /// Property index (0-63).
    pub index: u8,
    /// Human-readable name.
    pub name: String,
    /// Replication condition.
    pub condition: ReplicationCondition,
    /// Reliability mode.
    pub reliability: PropertyReliability,
    /// Quantization precision (0.0 = no quantization).
    pub quantize_precision: f32,
    /// Size in bytes of the serialized property.
    pub byte_size: u16,
}

/// Property dirty flags as a 64-bit bitfield.
#[derive(Debug, Clone, Copy, Default)]
pub struct DirtyFlags(pub u64);

impl DirtyFlags {
    /// Set a property as dirty.
    #[inline]
    pub fn set(&mut self, index: u8) {
        self.0 |= 1u64 << index;
    }

    /// Clear a property's dirty flag.
    #[inline]
    pub fn clear(&mut self, index: u8) {
        self.0 &= !(1u64 << index);
    }

    /// Check if a property is dirty.
    #[inline]
    pub fn is_dirty(&self, index: u8) -> bool {
        (self.0 >> index) & 1 == 1
    }

    /// Clear all flags.
    #[inline]
    pub fn clear_all(&mut self) {
        self.0 = 0;
    }

    /// Whether any flag is set.
    #[inline]
    pub fn any_dirty(&self) -> bool {
        self.0 != 0
    }

    /// Count of dirty properties.
    #[inline]
    pub fn dirty_count(&self) -> u32 {
        self.0.count_ones()
    }

    /// Bitwise OR with another set of dirty flags.
    #[inline]
    pub fn merge(&mut self, other: DirtyFlags) {
        self.0 |= other.0;
    }
}

/// Manages property replication for a single entity.
#[derive(Debug)]
pub struct EntityReplicator {
    /// Entity ID.
    pub entity_id: NetEntityId,
    /// Property definitions.
    pub properties: Vec<PropertyDef>,
    /// Current dirty flags.
    pub dirty: DirtyFlags,
    /// Last known property values (serialized bytes).
    pub last_values: Vec<Vec<u8>>,
    /// Owning client (for OwnerOnly/SkipOwner conditions).
    pub owner_client: Option<NetClientId>,
    /// Whether this entity has ever been replicated.
    pub has_initial_replicated: bool,
    /// Last tick this entity was replicated.
    pub last_replication_tick: TickNumber,
}

impl EntityReplicator {
    /// Create a new entity replicator.
    pub fn new(entity_id: NetEntityId) -> Self {
        Self {
            entity_id,
            properties: Vec::new(),
            dirty: DirtyFlags::default(),
            last_values: Vec::new(),
            owner_client: None,
            has_initial_replicated: false,
            last_replication_tick: 0,
        }
    }

    /// Register a property.
    pub fn register_property(&mut self, def: PropertyDef) {
        let index = def.index as usize;
        while self.last_values.len() <= index {
            self.last_values.push(Vec::new());
        }
        self.properties.push(def);
    }

    /// Mark a property as dirty.
    pub fn mark_dirty(&mut self, index: u8) {
        self.dirty.set(index);
    }

    /// Set a property value and mark dirty if it changed.
    pub fn set_property_value(&mut self, index: u8, value: Vec<u8>) -> bool {
        let idx = index as usize;
        if idx >= self.last_values.len() {
            return false;
        }
        if self.last_values[idx] != value {
            self.last_values[idx] = value;
            self.dirty.set(index);
            true
        } else {
            false
        }
    }

    /// Get the serialized properties that need replication for a given client.
    pub fn get_replication_data(
        &self,
        target_client: NetClientId,
        is_initial: bool,
    ) -> Vec<(u8, &[u8])> {
        let mut result = Vec::new();

        for prop in &self.properties {
            let idx = prop.index as usize;
            let should_send = match prop.condition {
                ReplicationCondition::Always => true,
                ReplicationCondition::OnChange => {
                    is_initial || self.dirty.is_dirty(prop.index)
                }
                ReplicationCondition::InitialOnly => is_initial,
                ReplicationCondition::Never => false,
                ReplicationCondition::OwnerOnly => {
                    self.owner_client == Some(target_client)
                }
                ReplicationCondition::SkipOwner => {
                    self.owner_client != Some(target_client)
                }
            };

            if should_send && idx < self.last_values.len() {
                result.push((prop.index, self.last_values[idx].as_slice()));
            }
        }

        result
    }

    /// Clear all dirty flags after replication.
    pub fn clear_dirty(&mut self) {
        self.dirty.clear_all();
    }

    /// Estimate the byte size of all dirty properties.
    pub fn dirty_byte_size(&self) -> u32 {
        let mut total = 0u32;
        for prop in &self.properties {
            if self.dirty.is_dirty(prop.index) {
                total += prop.byte_size as u32;
            }
        }
        total
    }
}

/// Manages property replication for all entities.
#[derive(Debug)]
pub struct PropertyReplicationManager {
    /// Per-entity replicators.
    replicators: HashMap<NetEntityId, EntityReplicator>,
}

impl PropertyReplicationManager {
    /// Create a new manager.
    pub fn new() -> Self {
        Self {
            replicators: HashMap::new(),
        }
    }

    /// Register an entity for replication.
    pub fn register_entity(&mut self, entity_id: NetEntityId) -> &mut EntityReplicator {
        self.replicators
            .entry(entity_id)
            .or_insert_with(|| EntityReplicator::new(entity_id))
    }

    /// Unregister an entity.
    pub fn unregister_entity(&mut self, entity_id: NetEntityId) {
        self.replicators.remove(&entity_id);
    }

    /// Get a replicator.
    pub fn get(&self, entity_id: NetEntityId) -> Option<&EntityReplicator> {
        self.replicators.get(&entity_id)
    }

    /// Get a mutable replicator.
    pub fn get_mut(&mut self, entity_id: NetEntityId) -> Option<&mut EntityReplicator> {
        self.replicators.get_mut(&entity_id)
    }

    /// Count of registered entities.
    pub fn entity_count(&self) -> usize {
        self.replicators.len()
    }

    /// Get all entities with dirty properties.
    pub fn dirty_entities(&self) -> Vec<NetEntityId> {
        self.replicators
            .iter()
            .filter(|(_, r)| r.dirty.any_dirty())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Clear all dirty flags.
    pub fn clear_all_dirty(&mut self) {
        for r in self.replicators.values_mut() {
            r.clear_dirty();
        }
    }
}

impl Default for PropertyReplicationManager {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// BandwidthManager — priority-based bandwidth allocation
// ===========================================================================

/// Per-client bandwidth tracking and allocation.
#[derive(Debug, Clone)]
pub struct ClientBandwidth {
    /// Client ID.
    pub client_id: NetClientId,
    /// Target bandwidth (bytes per second).
    pub target_bps: u32,
    /// Current estimated used bandwidth (bytes per second).
    pub used_bps: u32,
    /// Bytes sent this tick.
    pub bytes_this_tick: u32,
    /// Bytes budget per tick.
    pub bytes_per_tick: u32,
    /// Whether bandwidth is saturated.
    pub saturated: bool,
    /// Adaptive quality level (1.0 = full, 0.0 = minimal).
    pub quality: f32,
    /// Rolling average of bytes sent per tick.
    rolling_average: f32,
    /// Alpha for exponential moving average.
    ema_alpha: f32,
}

impl ClientBandwidth {
    /// Create bandwidth tracker for a client.
    pub fn new(client_id: NetClientId, target_bps: u32, tick_rate: u32) -> Self {
        Self {
            client_id,
            target_bps,
            used_bps: 0,
            bytes_this_tick: 0,
            bytes_per_tick: target_bps / tick_rate,
            saturated: false,
            quality: 1.0,
            rolling_average: 0.0,
            ema_alpha: 0.1,
        }
    }

    /// Record bytes sent this tick.
    pub fn record_sent(&mut self, bytes: u32) {
        self.bytes_this_tick += bytes;
    }

    /// End the tick: update statistics and check saturation.
    pub fn end_tick(&mut self) {
        self.rolling_average = self.rolling_average * (1.0 - self.ema_alpha)
            + self.bytes_this_tick as f32 * self.ema_alpha;

        self.saturated = self.bytes_this_tick >= self.bytes_per_tick;

        // Adapt quality based on saturation.
        if self.saturated {
            self.quality = (self.quality - 0.05).max(0.2);
        } else if self.quality < 1.0 {
            self.quality = (self.quality + 0.02).min(1.0);
        }

        self.used_bps = (self.rolling_average * 60.0) as u32; // approx
        self.bytes_this_tick = 0;
    }

    /// Remaining bytes budget for this tick.
    pub fn remaining_budget(&self) -> u32 {
        self.bytes_per_tick.saturating_sub(self.bytes_this_tick)
    }

    /// Whether there is budget remaining.
    pub fn has_budget(&self) -> bool {
        self.bytes_this_tick < self.bytes_per_tick
    }
}

/// Manages bandwidth for all connected clients.
#[derive(Debug)]
pub struct BandwidthController {
    /// Per-client bandwidth trackers.
    clients: HashMap<NetClientId, ClientBandwidth>,
    /// Default target bandwidth.
    pub default_target_bps: u32,
    /// Tick rate for computing per-tick budgets.
    pub tick_rate: u32,
    /// Global bandwidth multiplier (for server-wide throttling).
    pub global_multiplier: f32,
    /// Statistics.
    pub stats: BandwidthStats,
}

/// Bandwidth statistics.
#[derive(Debug, Clone, Default)]
pub struct BandwidthStats {
    /// Total bytes sent this tick (all clients).
    pub total_bytes_this_tick: u32,
    /// Number of saturated clients.
    pub saturated_clients: u32,
    /// Average quality across clients.
    pub average_quality: f32,
    /// Peak bytes in a single tick.
    pub peak_bytes: u32,
}

impl BandwidthController {
    /// Create a new bandwidth controller.
    pub fn new(tick_rate: u32) -> Self {
        Self {
            clients: HashMap::new(),
            default_target_bps: DEFAULT_BANDWIDTH_TARGET,
            tick_rate,
            global_multiplier: 1.0,
            stats: BandwidthStats::default(),
        }
    }

    /// Add a client.
    pub fn add_client(&mut self, client_id: NetClientId) {
        let adjusted = (self.default_target_bps as f32 * self.global_multiplier) as u32;
        let tracker = ClientBandwidth::new(client_id, adjusted, self.tick_rate);
        self.clients.insert(client_id, tracker);
    }

    /// Remove a client.
    pub fn remove_client(&mut self, client_id: NetClientId) {
        self.clients.remove(&client_id);
    }

    /// Get bandwidth tracker for a client.
    pub fn get_client(&self, client_id: NetClientId) -> Option<&ClientBandwidth> {
        self.clients.get(&client_id)
    }

    /// Get mutable bandwidth tracker.
    pub fn get_client_mut(&mut self, client_id: NetClientId) -> Option<&mut ClientBandwidth> {
        self.clients.get_mut(&client_id)
    }

    /// Record bytes sent to a client.
    pub fn record_sent(&mut self, client_id: NetClientId, bytes: u32) {
        if let Some(tracker) = self.clients.get_mut(&client_id) {
            tracker.record_sent(bytes);
        }
    }

    /// Check if a client has bandwidth budget remaining.
    pub fn has_budget(&self, client_id: NetClientId) -> bool {
        self.clients
            .get(&client_id)
            .map(|t| t.has_budget())
            .unwrap_or(false)
    }

    /// Allocate bandwidth for an entity update to a client.
    ///
    /// Returns true if the update fits within budget, consuming the bytes.
    pub fn try_allocate(&mut self, client_id: NetClientId, bytes: u32) -> bool {
        if let Some(tracker) = self.clients.get_mut(&client_id) {
            if tracker.remaining_budget() >= bytes {
                tracker.record_sent(bytes);
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// End the tick for all clients.
    pub fn end_tick(&mut self) {
        self.stats.total_bytes_this_tick = 0;
        self.stats.saturated_clients = 0;
        let mut total_quality = 0.0f32;

        for tracker in self.clients.values_mut() {
            self.stats.total_bytes_this_tick += tracker.bytes_this_tick;
            tracker.end_tick();
            if tracker.saturated {
                self.stats.saturated_clients += 1;
            }
            total_quality += tracker.quality;
        }

        if !self.clients.is_empty() {
            self.stats.average_quality = total_quality / self.clients.len() as f32;
        }

        if self.stats.total_bytes_this_tick > self.stats.peak_bytes {
            self.stats.peak_bytes = self.stats.total_bytes_this_tick;
        }
    }

    /// Get the quality level for a client (for adaptive replication).
    pub fn client_quality(&self, client_id: NetClientId) -> f32 {
        self.clients
            .get(&client_id)
            .map(|t| t.quality)
            .unwrap_or(1.0)
    }

    /// Number of connected clients.
    pub fn client_count(&self) -> usize {
        self.clients.len()
    }
}

// ===========================================================================
// AntiCheat — server-side validation
// ===========================================================================

/// Anti-cheat violation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheatViolation {
    /// Movement speed exceeds maximum.
    SpeedHack,
    /// Position changed impossibly between ticks.
    Teleport,
    /// Aim accuracy is statistically impossible.
    Aimbot,
    /// Input rate exceeds maximum.
    InputFlood,
    /// Action rate exceeds maximum.
    ActionRateLimit,
    /// Invalid input data.
    InvalidInput,
    /// Generic suspicious behavior.
    Suspicious,
}

/// A recorded anti-cheat violation.
#[derive(Debug, Clone)]
pub struct Violation {
    /// The type of violation.
    pub violation_type: CheatViolation,
    /// The offending client.
    pub client_id: NetClientId,
    /// Server tick when detected.
    pub tick: TickNumber,
    /// Severity score (0.0 = minor, 1.0 = definite cheat).
    pub severity: f32,
    /// Human-readable details.
    pub details: String,
}

/// Per-client anti-cheat tracking.
#[derive(Debug)]
struct ClientCheatTracker {
    /// Client ID.
    client_id: NetClientId,
    /// Last known position.
    last_position: Vec3Net,
    /// Last position update tick.
    last_position_tick: TickNumber,
    /// Cumulative violation score.
    violation_score: f32,
    /// Input count this second.
    input_count_this_second: u32,
    /// Last input rate reset tick.
    input_rate_reset_tick: TickNumber,
    /// Accurate shot count in the current window.
    accurate_shots: u32,
    /// Total shots in the current window.
    total_shots: u32,
    /// Window start tick for aimbot detection.
    aimbot_window_start: TickNumber,
    /// Action rate counters: action_type -> (count, reset_tick).
    action_rates: HashMap<u32, (u32, TickNumber)>,
    /// Maximum violations before auto-kick.
    max_violations: f32,
}

impl ClientCheatTracker {
    fn new(client_id: NetClientId) -> Self {
        Self {
            client_id,
            last_position: Vec3Net::ZERO,
            last_position_tick: 0,
            violation_score: 0.0,
            input_count_this_second: 0,
            input_rate_reset_tick: 0,
            accurate_shots: 0,
            total_shots: 0,
            aimbot_window_start: 0,
            action_rates: HashMap::new(),
            max_violations: 10.0,
        }
    }
}

/// Server-side anti-cheat system.
#[derive(Debug)]
pub struct AntiCheatSystem {
    /// Per-client trackers.
    trackers: HashMap<NetClientId, ClientCheatTracker>,
    /// Pending violations to report.
    pub pending_violations: Vec<Violation>,
    /// Configuration.
    pub config: AntiCheatConfig,
    /// Clients to kick.
    pub kick_queue: Vec<(NetClientId, String)>,
}

/// Anti-cheat configuration.
#[derive(Debug, Clone)]
pub struct AntiCheatConfig {
    /// Maximum units a player can move per tick.
    pub max_speed_per_tick: f32,
    /// Maximum teleport distance (anything beyond = violation).
    pub max_teleport_distance: f32,
    /// Maximum inputs per second.
    pub max_input_rate: u32,
    /// Aimbot detection: maximum ratio of accurate shots.
    pub aimbot_accuracy_threshold: f32,
    /// Aimbot detection window in ticks.
    pub aimbot_window_ticks: u32,
    /// Minimum shots in window before evaluating aimbot.
    pub aimbot_min_shots: u32,
    /// Score decay per tick.
    pub score_decay_per_tick: f32,
    /// Score threshold for auto-kick.
    pub kick_threshold: f32,
    /// Whether the system is enabled.
    pub enabled: bool,
}

impl Default for AntiCheatConfig {
    fn default() -> Self {
        Self {
            max_speed_per_tick: SPEED_HACK_MAX_UNITS_PER_TICK,
            max_teleport_distance: 100.0,
            max_input_rate: MAX_INPUT_RATE,
            aimbot_accuracy_threshold: 0.95,
            aimbot_window_ticks: AIMBOT_WINDOW_TICKS,
            aimbot_min_shots: 10,
            score_decay_per_tick: 0.001,
            kick_threshold: 10.0,
            enabled: true,
        }
    }
}

impl AntiCheatSystem {
    /// Create a new anti-cheat system.
    pub fn new(config: AntiCheatConfig) -> Self {
        Self {
            trackers: HashMap::new(),
            pending_violations: Vec::new(),
            config,
            kick_queue: Vec::new(),
        }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(AntiCheatConfig::default())
    }

    /// Register a client.
    pub fn add_client(&mut self, client_id: NetClientId) {
        self.trackers
            .insert(client_id, ClientCheatTracker::new(client_id));
    }

    /// Remove a client.
    pub fn remove_client(&mut self, client_id: NetClientId) {
        self.trackers.remove(&client_id);
    }

    /// Validate a client's input.
    pub fn validate_input(
        &mut self,
        client_id: NetClientId,
        input: &PlayerInput,
        current_tick: TickNumber,
    ) -> bool {
        if !self.config.enabled {
            return true;
        }

        let tracker = match self.trackers.get_mut(&client_id) {
            Some(t) => t,
            None => return true,
        };

        let mut valid = true;

        // Check input rate.
        let ticks_per_second = 60u32; // approximate
        if current_tick - tracker.input_rate_reset_tick >= ticks_per_second {
            tracker.input_count_this_second = 0;
            tracker.input_rate_reset_tick = current_tick;
        }
        tracker.input_count_this_second += 1;
        if tracker.input_count_this_second > self.config.max_input_rate {
            self.record_violation(
                client_id,
                CheatViolation::InputFlood,
                current_tick,
                0.5,
                format!(
                    "Input rate {} exceeds max {}",
                    tracker.input_count_this_second, self.config.max_input_rate
                ),
            );
            valid = false;
        }

        // Check movement speed.
        let move_speed = input.movement_speed();
        if move_speed > 1.5 {
            // Movement direction should be normalized.
            self.record_violation(
                client_id,
                CheatViolation::InvalidInput,
                current_tick,
                0.3,
                format!("Move direction length {:.2} exceeds 1.0", move_speed),
            );
            valid = false;
        }

        valid
    }

    /// Validate a position update from a client.
    pub fn validate_position(
        &mut self,
        client_id: NetClientId,
        new_position: Vec3Net,
        current_tick: TickNumber,
    ) -> bool {
        if !self.config.enabled {
            return true;
        }

        let tracker = match self.trackers.get_mut(&client_id) {
            Some(t) => t,
            None => return true,
        };

        let mut valid = true;

        if tracker.last_position_tick > 0 {
            let tick_delta = current_tick.saturating_sub(tracker.last_position_tick);
            let distance = tracker.last_position.distance(&new_position);
            let max_allowed = self.config.max_speed_per_tick * tick_delta as f32;

            if distance > max_allowed && tick_delta < 60 {
                let severity = ((distance / max_allowed) - 1.0).min(1.0);
                self.record_violation(
                    client_id,
                    CheatViolation::SpeedHack,
                    current_tick,
                    severity,
                    format!(
                        "Moved {:.2} units in {} ticks (max {:.2})",
                        distance, tick_delta, max_allowed
                    ),
                );
                valid = false;
            }

            if distance > self.config.max_teleport_distance && tick_delta < 10 {
                self.record_violation(
                    client_id,
                    CheatViolation::Teleport,
                    current_tick,
                    1.0,
                    format!(
                        "Teleported {:.2} units in {} ticks",
                        distance, tick_delta
                    ),
                );
                valid = false;
            }
        }

        tracker.last_position = new_position;
        tracker.last_position_tick = current_tick;

        valid
    }

    /// Record a shot result for aimbot detection.
    pub fn record_shot(
        &mut self,
        client_id: NetClientId,
        hit: bool,
        current_tick: TickNumber,
    ) {
        if !self.config.enabled {
            return;
        }

        let tracker = match self.trackers.get_mut(&client_id) {
            Some(t) => t,
            None => return,
        };

        // Reset window if expired.
        if current_tick - tracker.aimbot_window_start >= self.config.aimbot_window_ticks {
            tracker.accurate_shots = 0;
            tracker.total_shots = 0;
            tracker.aimbot_window_start = current_tick;
        }

        tracker.total_shots += 1;
        if hit {
            tracker.accurate_shots += 1;
        }

        // Evaluate aimbot detection.
        if tracker.total_shots >= self.config.aimbot_min_shots {
            let accuracy = tracker.accurate_shots as f32 / tracker.total_shots as f32;
            if accuracy > self.config.aimbot_accuracy_threshold {
                self.record_violation(
                    client_id,
                    CheatViolation::Aimbot,
                    current_tick,
                    accuracy,
                    format!(
                        "Accuracy {:.1}% over {} shots",
                        accuracy * 100.0,
                        tracker.total_shots
                    ),
                );
            }
        }
    }

    /// Check an action rate limit.
    pub fn check_action_rate(
        &mut self,
        client_id: NetClientId,
        action_type: u32,
        max_per_second: u32,
        current_tick: TickNumber,
    ) -> bool {
        if !self.config.enabled {
            return true;
        }

        let tracker = match self.trackers.get_mut(&client_id) {
            Some(t) => t,
            None => return true,
        };

        let entry = tracker
            .action_rates
            .entry(action_type)
            .or_insert((0, current_tick));

        // Reset if a second has passed.
        if current_tick - entry.1 >= 60 {
            entry.0 = 0;
            entry.1 = current_tick;
        }

        entry.0 += 1;
        if entry.0 > max_per_second {
            self.record_violation(
                client_id,
                CheatViolation::ActionRateLimit,
                current_tick,
                0.3,
                format!(
                    "Action type {} rate {} exceeds max {}",
                    action_type, entry.0, max_per_second
                ),
            );
            false
        } else {
            true
        }
    }

    /// Tick the anti-cheat system: decay scores, check for kicks.
    pub fn tick(&mut self, current_tick: TickNumber) {
        self.kick_queue.clear();

        for tracker in self.trackers.values_mut() {
            // Decay violation score.
            tracker.violation_score =
                (tracker.violation_score - self.config.score_decay_per_tick).max(0.0);

            // Check for auto-kick.
            if tracker.violation_score >= self.config.kick_threshold {
                // Will be processed by the caller.
            }
        }

        // Collect clients to kick.
        let kick_threshold = self.config.kick_threshold;
        let to_kick: Vec<(NetClientId, f32)> = self
            .trackers
            .iter()
            .filter(|(_, t)| t.violation_score >= kick_threshold)
            .map(|(&id, t)| (id, t.violation_score))
            .collect();

        for (client_id, score) in to_kick {
            self.kick_queue.push((
                client_id,
                format!("Anti-cheat violation score {:.2} exceeded threshold", score),
            ));
        }
    }

    /// Record a violation internally.
    fn record_violation(
        &mut self,
        client_id: NetClientId,
        violation_type: CheatViolation,
        tick: TickNumber,
        severity: f32,
        details: String,
    ) {
        if let Some(tracker) = self.trackers.get_mut(&client_id) {
            tracker.violation_score += severity;
        }

        self.pending_violations.push(Violation {
            violation_type,
            client_id,
            tick,
            severity,
            details,
        });
    }

    /// Get the violation score for a client.
    pub fn violation_score(&self, client_id: NetClientId) -> f32 {
        self.trackers
            .get(&client_id)
            .map(|t| t.violation_score)
            .unwrap_or(0.0)
    }

    /// Clear pending violations.
    pub fn clear_violations(&mut self) {
        self.pending_violations.clear();
    }
}

// ===========================================================================
// NetworkPredictionSmoothing — visual correction smoothing
// ===========================================================================

/// Smoothing state for a single entity's visual position.
#[derive(Debug, Clone)]
pub struct SmoothingState {
    /// Entity ID.
    pub entity_id: NetEntityId,
    /// Visual (rendered) position — what the player sees.
    pub visual_position: Vec3Net,
    /// Actual (simulated) position — where the entity really is.
    pub actual_position: Vec3Net,
    /// Visual rotation.
    pub visual_rotation: Vec3Net,
    /// Actual rotation.
    pub actual_rotation: Vec3Net,
    /// Smoothing rate (higher = faster correction).
    pub smooth_rate: f32,
    /// Snap threshold: if error exceeds this, snap instantly.
    pub snap_threshold: f32,
    /// Current position error magnitude.
    pub current_error: f32,
    /// Whether smoothing is active.
    pub active: bool,
}

impl SmoothingState {
    /// Create a new smoothing state.
    pub fn new(entity_id: NetEntityId, initial_position: Vec3Net) -> Self {
        Self {
            entity_id,
            visual_position: initial_position,
            actual_position: initial_position,
            visual_rotation: Vec3Net::ZERO,
            actual_rotation: Vec3Net::ZERO,
            smooth_rate: DEFAULT_SMOOTH_RATE,
            snap_threshold: 10.0,
            current_error: 0.0,
            active: true,
        }
    }

    /// Update the actual position (called when a new server state arrives).
    pub fn set_actual_position(&mut self, position: Vec3Net) {
        self.actual_position = position;
    }

    /// Update the actual rotation.
    pub fn set_actual_rotation(&mut self, rotation: Vec3Net) {
        self.actual_rotation = rotation;
    }

    /// Tick the smoothing: interpolate visual toward actual.
    ///
    /// Uses exponential smoothing: `visual = lerp(visual, actual, rate * dt)`.
    pub fn update(&mut self, dt: f32) {
        if !self.active {
            self.visual_position = self.actual_position;
            self.visual_rotation = self.actual_rotation;
            self.current_error = 0.0;
            return;
        }

        self.current_error = self.visual_position.distance(&self.actual_position);

        // Snap if error is too large.
        if self.current_error > self.snap_threshold {
            self.visual_position = self.actual_position;
            self.visual_rotation = self.actual_rotation;
            self.current_error = 0.0;
            return;
        }

        // Exponential smoothing.
        let t = (self.smooth_rate * dt).min(1.0);

        self.visual_position = Vec3Net::lerp(
            self.visual_position,
            self.actual_position,
            t,
        );

        self.visual_rotation = Vec3Net::lerp(
            self.visual_rotation,
            self.actual_rotation,
            t,
        );

        // Recalculate error after smoothing.
        self.current_error = self.visual_position.distance(&self.actual_position);
    }
}

/// Manages visual smoothing for all entities.
#[derive(Debug)]
pub struct NetworkSmoothingManager {
    /// Per-entity smoothing states.
    states: HashMap<NetEntityId, SmoothingState>,
    /// Default smooth rate.
    pub default_smooth_rate: f32,
    /// Default snap threshold.
    pub default_snap_threshold: f32,
}

impl NetworkSmoothingManager {
    /// Create a new smoothing manager.
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
            default_smooth_rate: DEFAULT_SMOOTH_RATE,
            default_snap_threshold: 10.0,
        }
    }

    /// Register an entity for smoothing.
    pub fn add_entity(&mut self, entity_id: NetEntityId, initial_position: Vec3Net) {
        let mut state = SmoothingState::new(entity_id, initial_position);
        state.smooth_rate = self.default_smooth_rate;
        state.snap_threshold = self.default_snap_threshold;
        self.states.insert(entity_id, state);
    }

    /// Remove an entity.
    pub fn remove_entity(&mut self, entity_id: NetEntityId) {
        self.states.remove(&entity_id);
    }

    /// Set the actual position for an entity (from server state).
    pub fn set_actual_position(&mut self, entity_id: NetEntityId, position: Vec3Net) {
        if let Some(state) = self.states.get_mut(&entity_id) {
            state.set_actual_position(position);
        }
    }

    /// Set actual rotation.
    pub fn set_actual_rotation(&mut self, entity_id: NetEntityId, rotation: Vec3Net) {
        if let Some(state) = self.states.get_mut(&entity_id) {
            state.set_actual_rotation(rotation);
        }
    }

    /// Get the visual (smoothed) position for rendering.
    pub fn visual_position(&self, entity_id: NetEntityId) -> Option<Vec3Net> {
        self.states.get(&entity_id).map(|s| s.visual_position)
    }

    /// Get the visual rotation.
    pub fn visual_rotation(&self, entity_id: NetEntityId) -> Option<Vec3Net> {
        self.states.get(&entity_id).map(|s| s.visual_rotation)
    }

    /// Update all smoothing states.
    pub fn update(&mut self, dt: f32) {
        for state in self.states.values_mut() {
            state.update(dt);
        }
    }

    /// Get the smoothing state for an entity.
    pub fn get_state(&self, entity_id: NetEntityId) -> Option<&SmoothingState> {
        self.states.get(&entity_id)
    }

    /// Entity count.
    pub fn entity_count(&self) -> usize {
        self.states.len()
    }
}

impl Default for NetworkSmoothingManager {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// GameServer — authoritative server
// ===========================================================================

/// Connection state for a client on the server.
#[derive(Debug, Clone)]
pub struct ConnectedClient {
    /// Client ID.
    pub client_id: NetClientId,
    /// Player entity in the simulation.
    pub player_entity: Option<NetEntityId>,
    /// Connection time (server time).
    pub connected_at: f64,
    /// Last input received tick.
    pub last_input_tick: TickNumber,
    /// Round-trip time estimate (seconds).
    pub rtt: f32,
    /// Packet loss percentage.
    pub packet_loss: f32,
    /// Whether the client is fully loaded and ready.
    pub ready: bool,
    /// Ping (ms).
    pub ping_ms: u32,
    /// Custom metadata.
    pub metadata: HashMap<String, String>,
}

impl ConnectedClient {
    pub fn new(client_id: NetClientId) -> Self {
        Self {
            client_id,
            player_entity: None,
            connected_at: 0.0,
            last_input_tick: 0,
            rtt: 0.0,
            packet_loss: 0.0,
            ready: false,
            ping_ms: 0,
            metadata: HashMap::new(),
        }
    }
}

/// Server-side load balancing hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerLoad {
    /// Low load: accepting new connections freely.
    Low,
    /// Medium load: normal operation.
    Medium,
    /// High load: consider redirecting new clients.
    High,
    /// Overloaded: reject new connections.
    Overloaded,
}

/// The authoritative game server.
#[derive(Debug)]
pub struct GameServer {
    /// Server tick rate (Hz).
    pub tick_rate: u32,
    /// Current tick number.
    pub current_tick: TickNumber,
    /// Fixed delta time per tick.
    pub dt: f32,
    /// Maximum connected clients.
    pub max_clients: usize,
    /// Connected clients.
    clients: HashMap<NetClientId, ConnectedClient>,
    /// Next client ID to assign.
    next_client_id: NetClientId,
    /// Interest management system.
    pub interest: InterestManager,
    /// Property replication manager.
    pub replication: PropertyReplicationManager,
    /// Bandwidth controller.
    pub bandwidth: BandwidthController,
    /// Anti-cheat system.
    pub anti_cheat: AntiCheatSystem,
    /// Server reconciliation tracker.
    pub reconciliation: ServerReconciliation,
    /// Network smoothing (used for remote entity interpolation on clients).
    pub smoothing: NetworkSmoothingManager,
    /// Server uptime in seconds.
    pub uptime: f64,
    /// Current load level.
    pub load: ServerLoad,
    /// Server name.
    pub name: String,
    /// Statistics.
    pub stats: GameServerStats,
}

/// Game server statistics.
#[derive(Debug, Clone, Default)]
pub struct GameServerStats {
    /// Total ticks processed.
    pub total_ticks: u64,
    /// Connected client count.
    pub client_count: usize,
    /// Total entities.
    pub entity_count: usize,
    /// Average tick processing time (ms).
    pub avg_tick_time_ms: f32,
    /// Peak tick processing time (ms).
    pub peak_tick_time_ms: f32,
    /// Total bytes sent.
    pub total_bytes_sent: u64,
    /// Total bytes received.
    pub total_bytes_received: u64,
    /// Inputs processed this tick.
    pub inputs_this_tick: u32,
    /// Entities replicated this tick.
    pub entities_replicated: u32,
}

impl GameServer {
    /// Create a new game server.
    pub fn new(name: impl Into<String>, tick_rate: u32, max_clients: usize) -> Self {
        let clamped_rate = tick_rate.clamp(MIN_TICK_RATE, MAX_TICK_RATE);
        Self {
            tick_rate: clamped_rate,
            current_tick: 0,
            dt: 1.0 / clamped_rate as f32,
            max_clients,
            clients: HashMap::new(),
            next_client_id: 1,
            interest: InterestManager::new(),
            replication: PropertyReplicationManager::new(),
            bandwidth: BandwidthController::new(clamped_rate),
            anti_cheat: AntiCheatSystem::with_defaults(),
            reconciliation: ServerReconciliation::new(),
            smoothing: NetworkSmoothingManager::new(),
            uptime: 0.0,
            load: ServerLoad::Low,
            name: name.into(),
            stats: GameServerStats::default(),
        }
    }

    /// Accept a new client connection. Returns the assigned client ID, or
    /// None if the server is full.
    pub fn accept_client(&mut self) -> Option<NetClientId> {
        if self.clients.len() >= self.max_clients {
            return None;
        }

        let client_id = self.next_client_id;
        self.next_client_id += 1;

        let mut client = ConnectedClient::new(client_id);
        client.connected_at = self.uptime;

        self.clients.insert(client_id, client);
        self.bandwidth.add_client(client_id);
        self.anti_cheat.add_client(client_id);

        self.update_load();
        Some(client_id)
    }

    /// Disconnect a client.
    pub fn disconnect_client(&mut self, client_id: NetClientId) {
        self.clients.remove(&client_id);
        self.interest.remove_client(client_id);
        self.bandwidth.remove_client(client_id);
        self.anti_cheat.remove_client(client_id);
        self.reconciliation.remove_client(client_id);
        self.update_load();
    }

    /// Process a client's input.
    pub fn process_input(
        &mut self,
        client_id: NetClientId,
        input: &PlayerInput,
    ) -> bool {
        // Anti-cheat validation.
        if !self.anti_cheat.validate_input(client_id, input, self.current_tick) {
            return false;
        }

        // Update last input tick.
        if let Some(client) = self.clients.get_mut(&client_id) {
            client.last_input_tick = self.current_tick;
        }

        // Record input sequence for reconciliation.
        self.reconciliation
            .record_input_sequence(client_id, input.sequence);

        self.stats.inputs_this_tick += 1;
        true
    }

    /// Execute a server tick.
    ///
    /// This is the main simulation loop step. The caller should:
    /// 1. Collect and process all client inputs
    /// 2. Call `tick()` to advance the simulation
    /// 3. Send state updates to clients
    pub fn tick(&mut self) {
        self.current_tick += 1;
        self.uptime += self.dt as f64;
        self.stats.total_ticks += 1;
        self.stats.inputs_this_tick = 0;
        self.stats.entities_replicated = 0;

        // Tick interest management dormancy.
        self.interest.tick_dormancy();

        // Tick anti-cheat.
        self.anti_cheat.tick(self.current_tick);

        // End bandwidth tick.
        self.bandwidth.end_tick();

        // Update statistics.
        self.stats.client_count = self.clients.len();
        self.stats.entity_count = self.replication.entity_count();

        // Update load level.
        self.update_load();
    }

    /// Replicate state to a specific client based on interest management.
    ///
    /// Returns the number of entities replicated and bytes used.
    pub fn replicate_to_client(
        &mut self,
        client_id: NetClientId,
    ) -> (u32, u32) {
        let relevant = self.interest.compute_relevancy(client_id);
        let budget = &self.interest.budget;
        let total_budget = self
            .bandwidth
            .get_client(client_id)
            .map(|b| b.remaining_budget())
            .unwrap_or(0);

        let mut entities_sent = 0u32;
        let mut bytes_sent = 0u32;

        for (entity_id, tier, _distance) in &relevant {
            if *tier == RelevancyTier::Irrelevant {
                continue;
            }

            let tier_budget = budget.bytes_for_tier(*tier, total_budget);
            if bytes_sent >= tier_budget {
                continue;
            }

            if let Some(replicator) = self.replication.get(entity_id) {
                let data_size = replicator.dirty_byte_size();
                if data_size > 0 && self.bandwidth.try_allocate(client_id, data_size) {
                    entities_sent += 1;
                    bytes_sent += data_size;
                }
            }
        }

        self.stats.entities_replicated += entities_sent;
        (entities_sent, bytes_sent)
    }

    /// Get a connected client.
    pub fn get_client(&self, client_id: NetClientId) -> Option<&ConnectedClient> {
        self.clients.get(&client_id)
    }

    /// Get all connected client IDs.
    pub fn client_ids(&self) -> Vec<NetClientId> {
        self.clients.keys().copied().collect()
    }

    /// Connected client count.
    pub fn client_count(&self) -> usize {
        self.clients.len()
    }

    /// Whether the server can accept new connections.
    pub fn can_accept(&self) -> bool {
        self.clients.len() < self.max_clients && self.load != ServerLoad::Overloaded
    }

    /// Update the server load level.
    fn update_load(&mut self) {
        let ratio = self.clients.len() as f32 / self.max_clients as f32;
        self.load = if ratio < 0.5 {
            ServerLoad::Low
        } else if ratio < 0.75 {
            ServerLoad::Medium
        } else if ratio < 0.95 {
            ServerLoad::High
        } else {
            ServerLoad::Overloaded
        };
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3_net_distance() {
        let a = Vec3Net::new(0.0, 0.0, 0.0);
        let b = Vec3Net::new(3.0, 4.0, 0.0);
        assert!((a.distance(&b) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_vec3_net_quantize() {
        let v = Vec3Net::new(1.234, 5.678, 9.012);
        let q = v.quantize(0.01);
        assert!((q.x - 1.23).abs() < 0.005);
        assert!((q.y - 5.68).abs() < 0.005);
    }

    #[test]
    fn test_input_ring_buffer() {
        let mut buf = InputRingBuffer::new(4);
        for i in 1..=5 {
            buf.push(PlayerInput::new(i, i));
        }
        assert_eq!(buf.len(), 4); // oldest dropped
        assert_eq!(buf.latest().unwrap().sequence, 5);

        buf.acknowledge_up_to(3);
        let unacked = buf.unacknowledged_inputs();
        assert!(unacked.iter().all(|i| i.sequence > 3));
    }

    #[test]
    fn test_reconciliation_classification() {
        let recon = ServerReconciliation::new();
        assert_eq!(recon.classify_error(0.05), ReconciliationResult::Match);
        match recon.classify_error(0.5) {
            ReconciliationResult::SmallCorrection { .. } => {}
            _ => panic!("Expected SmallCorrection"),
        }
        match recon.classify_error(3.0) {
            ReconciliationResult::LargeCorrection { .. } => {}
            _ => panic!("Expected LargeCorrection"),
        }
        match recon.classify_error(10.0) {
            ReconciliationResult::Teleport { .. } => {}
            _ => panic!("Expected Teleport"),
        }
    }

    #[test]
    fn test_dirty_flags() {
        let mut flags = DirtyFlags::default();
        assert!(!flags.any_dirty());
        flags.set(0);
        flags.set(5);
        flags.set(63);
        assert!(flags.any_dirty());
        assert_eq!(flags.dirty_count(), 3);
        assert!(flags.is_dirty(0));
        assert!(flags.is_dirty(5));
        assert!(!flags.is_dirty(1));
        flags.clear(5);
        assert!(!flags.is_dirty(5));
        assert_eq!(flags.dirty_count(), 2);
    }

    #[test]
    fn test_bandwidth_allocation() {
        let mut controller = BandwidthController::new(60);
        controller.add_client(1);

        // Default 50KB/s at 60Hz = ~833 bytes per tick.
        assert!(controller.has_budget(1));
        assert!(controller.try_allocate(1, 500));
        assert!(controller.try_allocate(1, 300));
        // Should be close to budget now.
        assert!(!controller.try_allocate(1, 500)); // exceeds remaining
    }

    #[test]
    fn test_relevancy_budget() {
        let budget = RelevancyBudget::default();
        let total = 1000u32;
        assert_eq!(budget.bytes_for_tier(RelevancyTier::Critical, total), 300);
        assert_eq!(budget.bytes_for_tier(RelevancyTier::High, total), 350);
        assert_eq!(budget.bytes_for_tier(RelevancyTier::Irrelevant, total), 0);
    }

    #[test]
    fn test_smoothing() {
        let mut state = SmoothingState::new(1, Vec3Net::ZERO);
        state.set_actual_position(Vec3Net::new(10.0, 0.0, 0.0));

        // After several ticks the visual should approach the actual.
        for _ in 0..100 {
            state.update(1.0 / 60.0);
        }
        assert!(state.visual_position.distance(&state.actual_position) < 0.1);
    }

    #[test]
    fn test_game_server_client_management() {
        let mut server = GameServer::new("TestServer", 60, 4);
        let c1 = server.accept_client().unwrap();
        let c2 = server.accept_client().unwrap();
        let c3 = server.accept_client().unwrap();
        let c4 = server.accept_client().unwrap();
        assert_eq!(server.client_count(), 4);
        assert!(server.accept_client().is_none()); // full
        server.disconnect_client(c2);
        assert_eq!(server.client_count(), 3);
        assert!(server.accept_client().is_some()); // room again
    }

    #[test]
    fn test_anti_cheat_speed_hack() {
        let mut ac = AntiCheatSystem::with_defaults();
        ac.add_client(1);

        // First position update.
        assert!(ac.validate_position(1, Vec3Net::new(0.0, 0.0, 0.0), 1));
        // Normal movement.
        assert!(ac.validate_position(1, Vec3Net::new(10.0, 0.0, 0.0), 2));
        // Impossible movement (1000 units in 1 tick).
        let valid = ac.validate_position(1, Vec3Net::new(1010.0, 0.0, 0.0), 3);
        assert!(!valid);
        assert!(!ac.pending_violations.is_empty());
    }

    #[test]
    fn test_entity_state_interpolation() {
        let a = NetEntityState {
            entity_id: 1,
            tick: 0,
            position: Vec3Net::new(0.0, 0.0, 0.0),
            velocity: Vec3Net::ZERO,
            rotation: Vec3Net::ZERO,
            angular_velocity: Vec3Net::ZERO,
            health: 100.0,
            state_flags: 0,
            custom_data: Vec::new(),
        };
        let b = NetEntityState {
            entity_id: 1,
            tick: 1,
            position: Vec3Net::new(10.0, 0.0, 0.0),
            velocity: Vec3Net::ZERO,
            rotation: Vec3Net::ZERO,
            angular_velocity: Vec3Net::ZERO,
            health: 80.0,
            state_flags: 0,
            custom_data: Vec::new(),
        };
        let mid = NetEntityState::interpolate(&a, &b, 0.5);
        assert!((mid.position.x - 5.0).abs() < 1e-5);
        assert!((mid.health - 90.0).abs() < 1e-5);
    }
}
