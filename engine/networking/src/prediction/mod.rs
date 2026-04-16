//! Client-side prediction and reconciliation.
//!
//! Implements techniques for responsive multiplayer gameplay:
//! - **Prediction**: clients simulate ahead using local inputs
//! - **Reconciliation**: server-authoritative state corrects mispredictions
//! - **Interpolation**: smooths remote entity positions between network updates
//! - **Lag compensation**: rewinds server state for fair hit detection
//!
//! ## Architecture
//!
//! The prediction system revolves around three key ideas:
//!
//! 1. The client applies player inputs immediately (prediction) so the game
//!    feels responsive.
//! 2. The server is authoritative. When the server sends back the canonical
//!    state for a tick, the client compares it with its prediction. If they
//!    differ, the client rolls back to the server state and replays all
//!    un-acknowledged inputs (reconciliation).
//! 3. Remote entities (other players) are rendered behind real-time using
//!    interpolation between received snapshots, hiding jitter.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use genovo_core::{EngineResult, Ray, AABB};

// ---------------------------------------------------------------------------
// TimestampedInput
// ---------------------------------------------------------------------------

/// A timestamped player input sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedInput {
    /// The simulation tick this input was generated on.
    pub tick: u64,
    /// Serialized input data (game-specific).
    pub data: Vec<u8>,
    /// Whether this input has been acknowledged by the server.
    pub acknowledged: bool,
}

impl TimestampedInput {
    /// Create a new input sample.
    pub fn new(tick: u64, data: Vec<u8>) -> Self {
        Self {
            tick,
            data,
            acknowledged: false,
        }
    }

    /// Encode the input to bytes for network transmission.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 + 2 + self.data.len());
        buf.extend_from_slice(&self.tick.to_be_bytes());
        buf.extend_from_slice(&(self.data.len() as u16).to_be_bytes());
        buf.extend_from_slice(&self.data);
        buf
    }

    /// Decode an input from bytes.
    pub fn decode(data: &[u8]) -> Option<(Self, usize)> {
        if data.len() < 10 {
            return None;
        }
        let tick = u64::from_be_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);
        let payload_len =
            u16::from_be_bytes([data[8], data[9]]) as usize;
        let total = 10 + payload_len;
        if data.len() < total {
            return None;
        }
        let payload = data[10..total].to_vec();
        Some((
            Self {
                tick,
                data: payload,
                acknowledged: false,
            },
            total,
        ))
    }
}

// ---------------------------------------------------------------------------
// InputBuffer
// ---------------------------------------------------------------------------

/// Ring buffer storing recent player inputs for prediction and reconciliation.
///
/// Inputs are kept until acknowledged by the server so they can be replayed
/// during reconciliation if the server disagrees with the client's prediction.
pub struct InputBuffer {
    /// The input history, ordered by tick (oldest first).
    inputs: VecDeque<TimestampedInput>,
    /// Maximum number of inputs to retain.
    capacity: usize,
    /// The last tick acknowledged by the server.
    last_acknowledged_tick: u64,
}

impl InputBuffer {
    /// Creates a new input buffer with the given capacity (default 64).
    pub fn new(capacity: usize) -> Self {
        Self {
            inputs: VecDeque::with_capacity(capacity),
            capacity,
            last_acknowledged_tick: 0,
        }
    }

    /// Records a new input at the given tick.
    pub fn push(&mut self, tick: u64, data: Vec<u8>) {
        if self.inputs.len() >= self.capacity {
            self.inputs.pop_front();
        }
        self.inputs.push_back(TimestampedInput::new(tick, data));
    }

    /// Marks all inputs up to and including `tick` as acknowledged and removes them.
    pub fn acknowledge_up_to(&mut self, tick: u64) {
        self.last_acknowledged_tick = tick;
        for input in &mut self.inputs {
            if input.tick <= tick {
                input.acknowledged = true;
            }
        }
        // Remove acknowledged inputs.
        while self
            .inputs
            .front()
            .is_some_and(|i| i.acknowledged)
        {
            self.inputs.pop_front();
        }
    }

    /// Returns all unacknowledged inputs (for replay during reconciliation).
    pub fn unacknowledged(&self) -> impl Iterator<Item = &TimestampedInput> {
        self.inputs.iter().filter(|i| !i.acknowledged)
    }

    /// Returns all unacknowledged inputs as a collected Vec (for serialization).
    pub fn unacknowledged_vec(&self) -> Vec<&TimestampedInput> {
        self.unacknowledged().collect()
    }

    /// Returns the input for a specific tick, if it exists.
    pub fn get(&self, tick: u64) -> Option<&TimestampedInput> {
        self.inputs.iter().find(|i| i.tick == tick)
    }

    /// Returns inputs in the range [start_tick, end_tick], inclusive.
    pub fn range(&self, start_tick: u64, end_tick: u64) -> Vec<&TimestampedInput> {
        self.inputs
            .iter()
            .filter(|i| i.tick >= start_tick && i.tick <= end_tick)
            .collect()
    }

    /// Returns the number of stored inputs.
    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    /// Returns `true` if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }

    /// Returns the last acknowledged tick.
    pub fn last_acknowledged_tick(&self) -> u64 {
        self.last_acknowledged_tick
    }

    /// Returns the newest tick in the buffer.
    pub fn newest_tick(&self) -> Option<u64> {
        self.inputs.back().map(|i| i.tick)
    }

    /// Returns the oldest tick in the buffer.
    pub fn oldest_tick(&self) -> Option<u64> {
        self.inputs.front().map(|i| i.tick)
    }

    /// Clears all inputs.
    pub fn clear(&mut self) {
        self.inputs.clear();
    }

    /// Encode all unacknowledged inputs to bytes (for sending to server).
    pub fn encode_unacknowledged(&self) -> Vec<u8> {
        let unacked: Vec<&TimestampedInput> = self.unacknowledged().collect();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(unacked.len() as u16).to_be_bytes());
        for input in &unacked {
            buf.extend_from_slice(&input.encode());
        }
        buf
    }

    /// Decode inputs from bytes (server receiving client inputs).
    pub fn decode_inputs(data: &[u8]) -> Option<Vec<TimestampedInput>> {
        if data.len() < 2 {
            return None;
        }
        let count = u16::from_be_bytes([data[0], data[1]]) as usize;
        let mut offset = 2;
        let mut inputs = Vec::with_capacity(count);
        for _ in 0..count {
            let (input, consumed) = TimestampedInput::decode(&data[offset..])?;
            offset += consumed;
            inputs.push(input);
        }
        Some(inputs)
    }
}

// ---------------------------------------------------------------------------
// StateSnapshot
// ---------------------------------------------------------------------------

/// A serialized snapshot of an entity's state at a specific simulation tick.
///
/// Used for rollback: when the server sends an authoritative state that
/// disagrees with a client prediction, the client rolls back to this snapshot
/// and replays inputs from that point forward.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    /// The simulation tick this snapshot was captured at.
    pub tick: u64,
    /// The entity this snapshot belongs to (network ID or local entity ID).
    pub entity_id: u32,
    /// Serialized component state.
    pub state: Vec<u8>,
    /// CRC32 checksum for fast mismatch detection.
    pub checksum: u32,
}

impl StateSnapshot {
    /// Creates a new state snapshot with CRC32 checksum.
    pub fn new(tick: u64, entity_id: u32, state: Vec<u8>) -> Self {
        let checksum = Self::compute_crc32(&state);
        Self {
            tick,
            entity_id,
            state,
            checksum,
        }
    }

    /// CRC32 checksum computation.
    fn compute_crc32(data: &[u8]) -> u32 {
        let mut crc: u32 = 0xFFFFFFFF;
        for &byte in data {
            crc ^= byte as u32;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB88320;
                } else {
                    crc >>= 1;
                }
            }
        }
        !crc
    }

    /// Returns `true` if two snapshots have matching checksums.
    pub fn matches(&self, other: &StateSnapshot) -> bool {
        self.entity_id == other.entity_id && self.checksum == other.checksum
    }

    /// Encode the snapshot to bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 + 4 + 4 + 2 + self.state.len());
        buf.extend_from_slice(&self.tick.to_be_bytes());
        buf.extend_from_slice(&self.entity_id.to_be_bytes());
        buf.extend_from_slice(&self.checksum.to_be_bytes());
        buf.extend_from_slice(&(self.state.len() as u16).to_be_bytes());
        buf.extend_from_slice(&self.state);
        buf
    }

    /// Decode a snapshot from bytes.
    pub fn decode(data: &[u8]) -> Option<(Self, usize)> {
        if data.len() < 18 {
            return None;
        }
        let tick = u64::from_be_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);
        let entity_id =
            u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
        let checksum =
            u32::from_be_bytes([data[12], data[13], data[14], data[15]]);
        let state_len =
            u16::from_be_bytes([data[16], data[17]]) as usize;
        let total = 18 + state_len;
        if data.len() < total {
            return None;
        }
        let state = data[18..total].to_vec();
        Some((
            Self {
                tick,
                entity_id,
                state,
                checksum,
            },
            total,
        ))
    }
}

// ---------------------------------------------------------------------------
// InterpolationBuffer
// ---------------------------------------------------------------------------

/// Stores recent state snapshots for smooth interpolation of remote entities.
///
/// Remote entities arrive in discrete network updates. The interpolation
/// buffer holds a short history and blends between samples to produce
/// smooth visual positions even with high latency or packet loss.
pub struct InterpolationBuffer {
    /// Ring buffer of snapshots (oldest first).
    snapshots: VecDeque<StateSnapshot>,
    /// Maximum number of snapshots to retain.
    capacity: usize,
    /// Interpolation delay in seconds (how far behind "real time" we render).
    delay_seconds: f64,
    /// Tick rate (ticks per second) for converting between ticks and seconds.
    tick_rate: f64,
}

impl InterpolationBuffer {
    /// Creates a new interpolation buffer.
    ///
    /// `delay_seconds` controls the interpolation latency (typically ~100ms).
    /// `tick_rate` is the simulation tick rate in Hz.
    pub fn new(capacity: usize, delay_seconds: f64, tick_rate: f64) -> Self {
        Self {
            snapshots: VecDeque::with_capacity(capacity),
            capacity,
            delay_seconds,
            tick_rate,
        }
    }

    /// Creates an interpolation buffer with tick-based delay.
    pub fn with_delay_ticks(capacity: usize, delay_ticks: u64, tick_rate: f64) -> Self {
        let delay_seconds = delay_ticks as f64 / tick_rate;
        Self::new(capacity, delay_seconds, tick_rate)
    }

    /// Adds a new snapshot to the buffer.
    pub fn push(&mut self, snapshot: StateSnapshot) {
        // Insert in order (snapshots might arrive out of order).
        let tick = snapshot.tick;

        // Find the correct insertion position.
        let pos = self
            .snapshots
            .iter()
            .position(|s| s.tick > tick)
            .unwrap_or(self.snapshots.len());

        // Don't insert duplicates.
        if pos > 0 && self.snapshots[pos - 1].tick == tick {
            // Replace existing.
            self.snapshots[pos - 1] = snapshot;
            return;
        }

        self.snapshots.insert(pos, snapshot);

        // Trim to capacity.
        while self.snapshots.len() > self.capacity {
            self.snapshots.pop_front();
        }
    }

    /// Computes the interpolated state for the given render time.
    ///
    /// `render_time` is the current time in seconds. The interpolation
    /// target is `render_time - delay_seconds`, converted to a tick.
    ///
    /// Returns the two bounding snapshots and the interpolation factor `t`
    /// in `[0.0, 1.0]`, or `None` if insufficient data is available.
    pub fn sample(
        &self,
        render_time: f64,
    ) -> Option<(&StateSnapshot, &StateSnapshot, f32)> {
        let target_time = render_time - self.delay_seconds;
        let target_tick = (target_time * self.tick_rate).max(0.0);

        self.interpolate_at_tick(target_tick)
    }

    /// Interpolate at a specific fractional tick.
    pub fn interpolate_at_tick(
        &self,
        target_tick: f64,
    ) -> Option<(&StateSnapshot, &StateSnapshot, f32)> {
        if self.snapshots.len() < 2 {
            return None;
        }

        // Find the two snapshots bracketing the target tick.
        let mut before = None;
        let mut after = None;

        for snapshot in &self.snapshots {
            if (snapshot.tick as f64) <= target_tick {
                before = Some(snapshot);
            } else if after.is_none() {
                after = Some(snapshot);
            }
        }

        match (before, after) {
            (Some(a), Some(b)) => {
                let range = (b.tick as f64) - (a.tick as f64);
                let t = if range > 0.0 {
                    ((target_tick - a.tick as f64) / range) as f32
                } else {
                    0.0
                };
                Some((a, b, t.clamp(0.0, 1.0)))
            }
            // Extrapolation: if we're past all snapshots, return the last two.
            (Some(_), None) => {
                if self.snapshots.len() >= 2 {
                    let len = self.snapshots.len();
                    let a = &self.snapshots[len - 2];
                    let b = &self.snapshots[len - 1];
                    // t > 1.0 indicates extrapolation; we clamp.
                    Some((a, b, 1.0))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Returns the interpolation delay in seconds.
    pub fn delay_seconds(&self) -> f64 {
        self.delay_seconds
    }

    /// Sets the interpolation delay in seconds.
    pub fn set_delay_seconds(&mut self, delay: f64) {
        self.delay_seconds = delay;
    }

    /// Returns the interpolation delay in ticks.
    pub fn delay_ticks(&self) -> u64 {
        (self.delay_seconds * self.tick_rate) as u64
    }

    /// Sets the interpolation delay in ticks.
    pub fn set_delay_ticks(&mut self, ticks: u64) {
        self.delay_seconds = ticks as f64 / self.tick_rate;
    }

    /// Returns the number of snapshots in the buffer.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Returns the newest snapshot tick.
    pub fn newest_tick(&self) -> Option<u64> {
        self.snapshots.back().map(|s| s.tick)
    }

    /// Returns the oldest snapshot tick.
    pub fn oldest_tick(&self) -> Option<u64> {
        self.snapshots.front().map(|s| s.tick)
    }

    /// Clears all snapshots.
    pub fn clear(&mut self) {
        self.snapshots.clear();
    }
}

// ---------------------------------------------------------------------------
// LagCompensation
// ---------------------------------------------------------------------------

/// A hit result from a lag-compensated raycast.
#[derive(Debug, Clone)]
pub struct Hit {
    /// Entity ID of the hit entity.
    pub entity_id: u32,
    /// Distance along the ray to the hit point.
    pub distance: f32,
    /// The point of intersection.
    pub point: [f32; 3],
}

/// Per-entity hitbox state for lag compensation.
#[derive(Debug, Clone)]
pub struct EntityHitbox {
    /// Entity ID.
    pub entity_id: u32,
    /// Position (center of the AABB).
    pub position: [f32; 3],
    /// Half-extents of the axis-aligned bounding box.
    pub half_extents: [f32; 3],
}

impl EntityHitbox {
    /// Create a new entity hitbox.
    pub fn new(entity_id: u32, position: [f32; 3], half_extents: [f32; 3]) -> Self {
        Self {
            entity_id,
            position,
            half_extents,
        }
    }

    /// Convert to an AABB for intersection testing.
    pub fn to_aabb(&self) -> AABB {
        AABB::new(
            glam::Vec3::new(
                self.position[0] - self.half_extents[0],
                self.position[1] - self.half_extents[1],
                self.position[2] - self.half_extents[2],
            ),
            glam::Vec3::new(
                self.position[0] + self.half_extents[0],
                self.position[1] + self.half_extents[1],
                self.position[2] + self.half_extents[2],
            ),
        )
    }

    /// Encode to bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + 12 + 12);
        buf.extend_from_slice(&self.entity_id.to_be_bytes());
        for &v in &self.position {
            buf.extend_from_slice(&v.to_be_bytes());
        }
        for &v in &self.half_extents {
            buf.extend_from_slice(&v.to_be_bytes());
        }
        buf
    }

    /// Decode from bytes.
    pub fn decode(data: &[u8]) -> Option<(Self, usize)> {
        if data.len() < 28 {
            return None;
        }
        let entity_id =
            u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let px = f32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        let py = f32::from_be_bytes([data[8], data[9], data[10], data[11]]);
        let pz = f32::from_be_bytes([data[12], data[13], data[14], data[15]]);
        let hx = f32::from_be_bytes([data[16], data[17], data[18], data[19]]);
        let hy = f32::from_be_bytes([data[20], data[21], data[22], data[23]]);
        let hz = f32::from_be_bytes([data[24], data[25], data[26], data[27]]);
        Some((
            Self {
                entity_id,
                position: [px, py, pz],
                half_extents: [hx, hy, hz],
            },
            28,
        ))
    }
}

/// The complete relevant state for one simulation tick (lag compensation).
#[derive(Debug, Clone)]
pub struct TickState {
    /// The simulation tick.
    pub tick: u64,
    /// Per-entity hitbox states.
    pub entities: Vec<EntityHitbox>,
}

impl TickState {
    /// Create a new tick state.
    pub fn new(tick: u64) -> Self {
        Self {
            tick,
            entities: Vec::new(),
        }
    }

    /// Add an entity hitbox.
    pub fn add_entity(&mut self, hitbox: EntityHitbox) {
        self.entities.push(hitbox);
    }

    /// Find an entity by ID.
    pub fn get_entity(&self, entity_id: u32) -> Option<&EntityHitbox> {
        self.entities.iter().find(|e| e.entity_id == entity_id)
    }

    /// Encode to bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.tick.to_be_bytes());
        buf.extend_from_slice(&(self.entities.len() as u16).to_be_bytes());
        for entity in &self.entities {
            buf.extend_from_slice(&entity.encode());
        }
        buf
    }

    /// Decode from bytes.
    pub fn decode(data: &[u8]) -> Option<Self> {
        if data.len() < 10 {
            return None;
        }
        let tick = u64::from_be_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);
        let count = u16::from_be_bytes([data[8], data[9]]) as usize;
        let mut offset = 10;
        let mut entities = Vec::with_capacity(count);
        for _ in 0..count {
            let (entity, consumed) = EntityHitbox::decode(&data[offset..])?;
            offset += consumed;
            entities.push(entity);
        }
        Some(Self { tick, entities })
    }
}

/// Manages server-side lag compensation for fair hit detection.
///
/// When a client fires a weapon, they see the world as it was `RTT/2` ago.
/// The lag compensator stores recent world-state history so the server can
/// rewind and verify hits against what the client actually saw.
pub struct LagCompensation {
    /// History of state snapshots, keyed by tick.
    history: VecDeque<TickState>,
    /// Maximum history length (in ticks).
    max_history: usize,
}

impl LagCompensation {
    /// Creates a new lag compensator with the given history depth.
    pub fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    /// Records the current tick state for future rewind queries.
    pub fn record_tick(&mut self, tick_state: TickState) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(tick_state);
    }

    /// Retrieves the state at a specific tick for hit verification.
    ///
    /// Returns `None` if the tick is no longer in history.
    pub fn get_state_at_tick(&self, tick: u64) -> Option<&TickState> {
        self.history.iter().find(|s| s.tick == tick)
    }

    /// Finds the two tick states bracketing the given tick for interpolation.
    pub fn get_bracketing_states(
        &self,
        tick: u64,
    ) -> Option<(&TickState, &TickState, f32)> {
        let mut before = None;
        let mut after = None;

        for state in &self.history {
            if state.tick <= tick {
                before = Some(state);
            } else if after.is_none() {
                after = Some(state);
            }
        }

        match (before, after) {
            (Some(a), Some(b)) => {
                let range = (b.tick - a.tick) as f32;
                let t = if range > 0.0 {
                    (tick - a.tick) as f32 / range
                } else {
                    0.0
                };
                Some((a, b, t.clamp(0.0, 1.0)))
            }
            _ => None,
        }
    }

    /// Performs a lag-compensated raycast at a specific tick.
    ///
    /// Rewinds the world to the given tick and tests the ray against all
    /// entity hitboxes that existed at that time.
    pub fn raycast_at_tick(
        &self,
        tick: u64,
        ray: Ray,
    ) -> Vec<Hit> {
        let state = match self.get_state_at_tick(tick) {
            Some(s) => s,
            None => return Vec::new(),
        };

        let mut hits = Vec::new();

        for entity in &state.entities {
            let aabb = entity.to_aabb();
            if let Some(t) = aabb.ray_intersect(&ray) {
                if t >= 0.0 {
                    let point = ray.point_at(t);
                    hits.push(Hit {
                        entity_id: entity.entity_id,
                        distance: t,
                        point: [point.x, point.y, point.z],
                    });
                }
            }
        }

        // Sort by distance (closest first).
        hits.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        hits
    }

    /// Performs a lag-compensated hit verification.
    ///
    /// Checks whether a ray fired by `shooter_entity` at `client_tick` would
    /// hit `target_entity`.
    pub fn verify_hit(
        &self,
        client_tick: u64,
        _shooter_entity: u32,
        target_entity: u32,
        ray_origin: [f32; 3],
        ray_direction: [f32; 3],
    ) -> EngineResult<bool> {
        let state = self.get_state_at_tick(client_tick).ok_or_else(|| {
            genovo_core::EngineError::NotFound(format!(
                "tick {} not in lag compensation history",
                client_tick
            ))
        })?;

        let target = state.get_entity(target_entity).ok_or_else(|| {
            genovo_core::EngineError::NotFound(format!(
                "entity {} not found at tick {}",
                target_entity, client_tick
            ))
        })?;

        let ray = Ray::new(
            glam::Vec3::new(ray_origin[0], ray_origin[1], ray_origin[2]),
            glam::Vec3::new(
                ray_direction[0],
                ray_direction[1],
                ray_direction[2],
            )
            .normalize(),
        );

        let aabb = target.to_aabb();
        Ok(aabb.ray_intersect(&ray).is_some_and(|t| t >= 0.0))
    }

    /// Returns the number of ticks currently stored.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Returns the tick range currently stored.
    pub fn tick_range(&self) -> Option<(u64, u64)> {
        let first = self.history.front()?.tick;
        let last = self.history.back()?.tick;
        Some((first, last))
    }

    /// Clears all history.
    pub fn clear(&mut self) {
        self.history.clear();
    }
}

// ---------------------------------------------------------------------------
// PredictionManager
// ---------------------------------------------------------------------------

/// Coordinates client-side prediction, reconciliation, and interpolation.
///
/// Ties together the [`InputBuffer`], [`StateSnapshot`] history, and
/// reconciliation logic to provide responsive gameplay with authoritative
/// server corrections.
pub struct PredictionManager {
    /// Buffer of local player inputs.
    input_buffer: InputBuffer,
    /// History of predicted states for reconciliation (tick -> snapshot).
    predicted_states: VecDeque<StateSnapshot>,
    /// Maximum number of predicted states to retain.
    max_prediction_history: usize,
    /// The last authoritative tick received from the server.
    last_server_tick: u64,
    /// Current client simulation tick.
    client_tick: u64,
    /// Number of mispredictions detected (for diagnostics).
    misprediction_count: u64,
    /// The entity ID this prediction manager is tracking.
    entity_id: u32,
    /// Callback-style: stores the last reconciliation result.
    last_reconciliation_result: ReconciliationResult,
}

/// Result of a reconciliation attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconciliationResult {
    /// No reconciliation was needed (prediction matched server).
    Match,
    /// Reconciliation was performed (misprediction detected).
    Misprediction {
        /// The tick at which the divergence was detected.
        divergence_tick: u64,
        /// The number of inputs that need to be replayed.
        inputs_to_replay: usize,
    },
    /// No reconciliation attempted yet.
    None,
}

impl PredictionManager {
    /// Creates a new prediction manager for the given entity.
    pub fn new(
        entity_id: u32,
        input_buffer_capacity: usize,
        max_prediction_history: usize,
    ) -> Self {
        Self {
            input_buffer: InputBuffer::new(input_buffer_capacity),
            predicted_states: VecDeque::with_capacity(max_prediction_history),
            max_prediction_history,
            last_server_tick: 0,
            client_tick: 0,
            misprediction_count: 0,
            entity_id,
            last_reconciliation_result: ReconciliationResult::None,
        }
    }

    /// Records a player input for the current tick and saves the predicted state.
    ///
    /// `input_data` is the serialized input (game-specific).
    /// `predicted_state` is the resulting state after applying the input locally.
    pub fn predict(&mut self, input_data: Vec<u8>, predicted_state: Vec<u8>) {
        let tick = self.client_tick;

        // Store the input.
        self.input_buffer.push(tick, input_data);

        // Store the predicted state.
        let snapshot = StateSnapshot::new(tick, self.entity_id, predicted_state);
        self.predicted_states.push_back(snapshot);

        // Trim old predictions.
        while self.predicted_states.len() > self.max_prediction_history {
            self.predicted_states.pop_front();
        }

        self.client_tick += 1;
    }

    /// Reconciles local prediction against an authoritative server state.
    ///
    /// If the server state disagrees with the local prediction at the given
    /// tick, the caller should:
    /// 1. Roll back local state to `server_snapshot.state`.
    /// 2. Replay all unacknowledged inputs returned by `inputs_to_replay()`.
    ///
    /// Returns the reconciliation result.
    pub fn reconcile(
        &mut self,
        server_snapshot: StateSnapshot,
    ) -> ReconciliationResult {
        let server_tick = server_snapshot.tick;
        self.last_server_tick = server_tick;

        // Find our predicted state at the server's tick.
        let prediction_at_tick = self
            .predicted_states
            .iter()
            .find(|s| s.tick == server_tick);

        let result = match prediction_at_tick {
            Some(predicted) => {
                if predicted.matches(&server_snapshot) {
                    // Prediction was correct.
                    ReconciliationResult::Match
                } else {
                    // Misprediction! Need to rollback and replay.
                    self.misprediction_count += 1;

                    let inputs_to_replay = self
                        .input_buffer
                        .unacknowledged()
                        .filter(|i| i.tick > server_tick)
                        .count();

                    log::debug!(
                        "Misprediction at tick {}: replaying {} inputs",
                        server_tick,
                        inputs_to_replay
                    );

                    ReconciliationResult::Misprediction {
                        divergence_tick: server_tick,
                        inputs_to_replay,
                    }
                }
            }
            None => {
                // We don't have a prediction for this tick (too old or too new).
                // Treat as a misprediction and apply the server state.
                let inputs_to_replay =
                    self.input_buffer.unacknowledged().count();

                ReconciliationResult::Misprediction {
                    divergence_tick: server_tick,
                    inputs_to_replay,
                }
            }
        };

        // Acknowledge inputs up to the server tick.
        self.input_buffer.acknowledge_up_to(server_tick);

        // Remove predicted states up to and including the server tick.
        while self
            .predicted_states
            .front()
            .is_some_and(|s| s.tick <= server_tick)
        {
            self.predicted_states.pop_front();
        }

        self.last_reconciliation_result = result.clone();
        result
    }

    /// Returns inputs that need to be replayed after a misprediction.
    /// These are all inputs after the server's acknowledged tick.
    pub fn inputs_to_replay(&self) -> Vec<&TimestampedInput> {
        self.input_buffer.unacknowledged_vec()
    }

    /// After reconciliation with misprediction, call this to store the
    /// replayed predicted states (replacing the old ones).
    pub fn store_replayed_states(&mut self, states: Vec<StateSnapshot>) {
        // Clear old predicted states and replace with new replayed ones.
        self.predicted_states.clear();
        for state in states {
            self.predicted_states.push_back(state);
        }
    }

    /// Returns a reference to the input buffer.
    pub fn input_buffer(&self) -> &InputBuffer {
        &self.input_buffer
    }

    /// Returns a mutable reference to the input buffer.
    pub fn input_buffer_mut(&mut self) -> &mut InputBuffer {
        &mut self.input_buffer
    }

    /// Returns the current client tick.
    pub fn client_tick(&self) -> u64 {
        self.client_tick
    }

    /// Returns the last server tick received.
    pub fn last_server_tick(&self) -> u64 {
        self.last_server_tick
    }

    /// Returns the prediction lead (how many ticks the client is ahead).
    pub fn prediction_lead(&self) -> u64 {
        self.client_tick.saturating_sub(self.last_server_tick)
    }

    /// Returns the total number of mispredictions detected.
    pub fn misprediction_count(&self) -> u64 {
        self.misprediction_count
    }

    /// Returns the last reconciliation result.
    pub fn last_reconciliation_result(&self) -> &ReconciliationResult {
        &self.last_reconciliation_result
    }

    /// Returns the number of predicted states currently stored.
    pub fn prediction_history_len(&self) -> usize {
        self.predicted_states.len()
    }

    /// Returns the entity ID being tracked.
    pub fn entity_id(&self) -> u32 {
        self.entity_id
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // TimestampedInput
    // -----------------------------------------------------------------------

    #[test]
    fn test_timestamped_input_roundtrip() {
        let input = TimestampedInput::new(42, vec![1, 2, 3, 4]);
        let encoded = input.encode();
        let (decoded, consumed) = TimestampedInput::decode(&encoded).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.tick, 42);
        assert_eq!(decoded.data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_timestamped_input_empty_data() {
        let input = TimestampedInput::new(0, vec![]);
        let encoded = input.encode();
        let (decoded, _) = TimestampedInput::decode(&encoded).unwrap();
        assert_eq!(decoded.tick, 0);
        assert!(decoded.data.is_empty());
    }

    #[test]
    fn test_timestamped_input_decode_too_short() {
        assert!(TimestampedInput::decode(&[0; 5]).is_none());
    }

    // -----------------------------------------------------------------------
    // InputBuffer
    // -----------------------------------------------------------------------

    #[test]
    fn test_input_buffer_push_and_get() {
        let mut buf = InputBuffer::new(64);
        buf.push(0, vec![1]);
        buf.push(1, vec![2]);
        buf.push(2, vec![3]);

        assert_eq!(buf.len(), 3);
        assert_eq!(buf.get(0).unwrap().data, vec![1]);
        assert_eq!(buf.get(1).unwrap().data, vec![2]);
        assert_eq!(buf.get(2).unwrap().data, vec![3]);
        assert!(buf.get(99).is_none());
    }

    #[test]
    fn test_input_buffer_capacity() {
        let mut buf = InputBuffer::new(3);
        buf.push(0, vec![1]);
        buf.push(1, vec![2]);
        buf.push(2, vec![3]);
        buf.push(3, vec![4]); // should evict oldest (tick 0)

        assert_eq!(buf.len(), 3);
        assert!(buf.get(0).is_none()); // evicted
        assert!(buf.get(1).is_some());
        assert!(buf.get(3).is_some());
    }

    #[test]
    fn test_input_buffer_acknowledge() {
        let mut buf = InputBuffer::new(64);
        buf.push(0, vec![1]);
        buf.push(1, vec![2]);
        buf.push(2, vec![3]);
        buf.push(3, vec![4]);

        buf.acknowledge_up_to(2);
        assert_eq!(buf.last_acknowledged_tick(), 2);
        // Inputs 0, 1, 2 should be removed.
        assert_eq!(buf.len(), 1);
        assert!(buf.get(0).is_none());
        assert!(buf.get(1).is_none());
        assert!(buf.get(2).is_none());
        assert!(buf.get(3).is_some());
    }

    #[test]
    fn test_input_buffer_unacknowledged() {
        let mut buf = InputBuffer::new(64);
        buf.push(0, vec![1]);
        buf.push(1, vec![2]);
        buf.push(2, vec![3]);

        buf.acknowledge_up_to(1);
        let unacked: Vec<_> = buf.unacknowledged().collect();
        assert_eq!(unacked.len(), 1);
        assert_eq!(unacked[0].tick, 2);
    }

    #[test]
    fn test_input_buffer_range() {
        let mut buf = InputBuffer::new(64);
        for i in 0..10 {
            buf.push(i, vec![i as u8]);
        }

        let range = buf.range(3, 7);
        assert_eq!(range.len(), 5);
        assert_eq!(range[0].tick, 3);
        assert_eq!(range[4].tick, 7);
    }

    #[test]
    fn test_input_buffer_newest_oldest() {
        let mut buf = InputBuffer::new(64);
        assert!(buf.newest_tick().is_none());
        assert!(buf.oldest_tick().is_none());

        buf.push(5, vec![1]);
        buf.push(10, vec![2]);

        assert_eq!(buf.oldest_tick(), Some(5));
        assert_eq!(buf.newest_tick(), Some(10));
    }

    #[test]
    fn test_input_buffer_encode_decode() {
        let mut buf = InputBuffer::new(64);
        buf.push(0, vec![10, 20]);
        buf.push(1, vec![30, 40]);
        buf.push(2, vec![50, 60]);

        let encoded = buf.encode_unacknowledged();
        let decoded = InputBuffer::decode_inputs(&encoded).unwrap();
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].tick, 0);
        assert_eq!(decoded[0].data, vec![10, 20]);
        assert_eq!(decoded[2].tick, 2);
        assert_eq!(decoded[2].data, vec![50, 60]);
    }

    #[test]
    fn test_input_buffer_clear() {
        let mut buf = InputBuffer::new(64);
        buf.push(0, vec![1]);
        buf.push(1, vec![2]);
        buf.clear();
        assert!(buf.is_empty());
    }

    // -----------------------------------------------------------------------
    // StateSnapshot
    // -----------------------------------------------------------------------

    #[test]
    fn test_state_snapshot_roundtrip() {
        let snap = StateSnapshot::new(42, 7, vec![1, 2, 3, 4, 5]);
        let encoded = snap.encode();
        let (decoded, consumed) = StateSnapshot::decode(&encoded).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.tick, 42);
        assert_eq!(decoded.entity_id, 7);
        assert_eq!(decoded.state, vec![1, 2, 3, 4, 5]);
        assert_eq!(decoded.checksum, snap.checksum);
    }

    #[test]
    fn test_state_snapshot_matches() {
        let s1 = StateSnapshot::new(1, 1, vec![10, 20, 30]);
        let s2 = StateSnapshot::new(2, 1, vec![10, 20, 30]); // same state, different tick
        assert!(s1.matches(&s2));

        let s3 = StateSnapshot::new(1, 1, vec![10, 20, 31]); // different state
        assert!(!s1.matches(&s3));
    }

    #[test]
    fn test_state_snapshot_different_entity() {
        let s1 = StateSnapshot::new(1, 1, vec![10, 20, 30]);
        let s2 = StateSnapshot::new(1, 2, vec![10, 20, 30]); // different entity
        assert!(!s1.matches(&s2)); // entity_id differs
    }

    #[test]
    fn test_state_snapshot_decode_too_short() {
        assert!(StateSnapshot::decode(&[0; 10]).is_none());
    }

    #[test]
    fn test_state_snapshot_crc32_determinism() {
        let s1 = StateSnapshot::new(0, 0, vec![1, 2, 3]);
        let s2 = StateSnapshot::new(0, 0, vec![1, 2, 3]);
        assert_eq!(s1.checksum, s2.checksum);
    }

    // -----------------------------------------------------------------------
    // InterpolationBuffer
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpolation_buffer_push() {
        let mut buf = InterpolationBuffer::new(10, 0.1, 60.0);
        buf.push(StateSnapshot::new(0, 1, vec![0]));
        buf.push(StateSnapshot::new(1, 1, vec![1]));
        buf.push(StateSnapshot::new(2, 1, vec![2]));

        assert_eq!(buf.len(), 3);
        assert_eq!(buf.oldest_tick(), Some(0));
        assert_eq!(buf.newest_tick(), Some(2));
    }

    #[test]
    fn test_interpolation_buffer_ordering() {
        let mut buf = InterpolationBuffer::new(10, 0.1, 60.0);
        // Insert out of order.
        buf.push(StateSnapshot::new(5, 1, vec![5]));
        buf.push(StateSnapshot::new(2, 1, vec![2]));
        buf.push(StateSnapshot::new(8, 1, vec![8]));

        assert_eq!(buf.oldest_tick(), Some(2));
        assert_eq!(buf.newest_tick(), Some(8));
    }

    #[test]
    fn test_interpolation_buffer_duplicate_replace() {
        let mut buf = InterpolationBuffer::new(10, 0.1, 60.0);
        buf.push(StateSnapshot::new(5, 1, vec![1]));
        buf.push(StateSnapshot::new(5, 1, vec![2])); // replace

        assert_eq!(buf.len(), 1); // not duplicated
    }

    #[test]
    fn test_interpolation_buffer_capacity() {
        let mut buf = InterpolationBuffer::new(3, 0.1, 60.0);
        buf.push(StateSnapshot::new(0, 1, vec![0]));
        buf.push(StateSnapshot::new(1, 1, vec![1]));
        buf.push(StateSnapshot::new(2, 1, vec![2]));
        buf.push(StateSnapshot::new(3, 1, vec![3]));

        assert_eq!(buf.len(), 3);
        assert_eq!(buf.oldest_tick(), Some(1)); // tick 0 evicted
    }

    #[test]
    fn test_interpolation_buffer_interpolate() {
        let mut buf = InterpolationBuffer::new(10, 0.0, 60.0);
        buf.push(StateSnapshot::new(0, 1, vec![0]));
        buf.push(StateSnapshot::new(10, 1, vec![100]));

        // Interpolate at tick 5 (midpoint).
        let result = buf.interpolate_at_tick(5.0);
        assert!(result.is_some());
        let (before, after, t) = result.unwrap();
        assert_eq!(before.tick, 0);
        assert_eq!(after.tick, 10);
        assert!((t - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_interpolation_buffer_interpolate_at_exact() {
        let mut buf = InterpolationBuffer::new(10, 0.0, 60.0);
        buf.push(StateSnapshot::new(0, 1, vec![0]));
        buf.push(StateSnapshot::new(10, 1, vec![100]));

        // At tick 0 exactly.
        let result = buf.interpolate_at_tick(0.0);
        assert!(result.is_some());
        let (_, _, t) = result.unwrap();
        assert!((t - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_interpolation_buffer_extrapolation_clamp() {
        let mut buf = InterpolationBuffer::new(10, 0.0, 60.0);
        buf.push(StateSnapshot::new(0, 1, vec![0]));
        buf.push(StateSnapshot::new(10, 1, vec![100]));

        // Beyond the last snapshot.
        let result = buf.interpolate_at_tick(20.0);
        assert!(result.is_some());
        let (_, _, t) = result.unwrap();
        assert!((t - 1.0).abs() < 0.001); // clamped
    }

    #[test]
    fn test_interpolation_buffer_insufficient_data() {
        let buf = InterpolationBuffer::new(10, 0.0, 60.0);
        assert!(buf.interpolate_at_tick(5.0).is_none());

        let mut buf2 = InterpolationBuffer::new(10, 0.0, 60.0);
        buf2.push(StateSnapshot::new(0, 1, vec![0]));
        // Only one snapshot; need two.
        assert!(buf2.interpolate_at_tick(5.0).is_none());
    }

    #[test]
    fn test_interpolation_delay_settings() {
        let mut buf = InterpolationBuffer::new(10, 0.1, 60.0);
        assert!((buf.delay_seconds() - 0.1).abs() < 0.001);
        assert_eq!(buf.delay_ticks(), 6); // 0.1 * 60

        buf.set_delay_seconds(0.2);
        assert_eq!(buf.delay_ticks(), 12);

        buf.set_delay_ticks(3);
        assert!((buf.delay_seconds() - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_interpolation_sample_with_delay() {
        let mut buf = InterpolationBuffer::new(10, 0.0, 10.0);
        buf.set_delay_seconds(0.5); // 5 ticks at 10Hz

        buf.push(StateSnapshot::new(0, 1, vec![0]));
        buf.push(StateSnapshot::new(10, 1, vec![100]));

        // At render_time = 1.0s (tick 10), with 0.5s delay, target = tick 5.
        let result = buf.sample(1.0);
        assert!(result.is_some());
        let (before, after, t) = result.unwrap();
        assert_eq!(before.tick, 0);
        assert_eq!(after.tick, 10);
        assert!((t - 0.5).abs() < 0.001);
    }

    // -----------------------------------------------------------------------
    // EntityHitbox
    // -----------------------------------------------------------------------

    #[test]
    fn test_entity_hitbox_roundtrip() {
        let hitbox = EntityHitbox::new(42, [1.0, 2.0, 3.0], [0.5, 0.5, 0.5]);
        let encoded = hitbox.encode();
        let (decoded, consumed) = EntityHitbox::decode(&encoded).unwrap();
        assert_eq!(consumed, 28);
        assert_eq!(decoded.entity_id, 42);
        assert_eq!(decoded.position, [1.0, 2.0, 3.0]);
        assert_eq!(decoded.half_extents, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_entity_hitbox_to_aabb() {
        let hitbox = EntityHitbox::new(0, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let aabb = hitbox.to_aabb();
        assert_eq!(aabb.min, glam::Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(aabb.max, glam::Vec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn test_entity_hitbox_decode_too_short() {
        assert!(EntityHitbox::decode(&[0; 10]).is_none());
    }

    // -----------------------------------------------------------------------
    // TickState
    // -----------------------------------------------------------------------

    #[test]
    fn test_tick_state_roundtrip() {
        let mut ts = TickState::new(100);
        ts.add_entity(EntityHitbox::new(1, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
        ts.add_entity(EntityHitbox::new(2, [5.0, 0.0, 0.0], [0.5, 0.5, 0.5]));

        let encoded = ts.encode();
        let decoded = TickState::decode(&encoded).unwrap();
        assert_eq!(decoded.tick, 100);
        assert_eq!(decoded.entities.len(), 2);
        assert_eq!(decoded.entities[0].entity_id, 1);
        assert_eq!(decoded.entities[1].entity_id, 2);
    }

    #[test]
    fn test_tick_state_get_entity() {
        let mut ts = TickState::new(0);
        ts.add_entity(EntityHitbox::new(1, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
        ts.add_entity(EntityHitbox::new(2, [5.0, 0.0, 0.0], [0.5, 0.5, 0.5]));

        assert!(ts.get_entity(1).is_some());
        assert!(ts.get_entity(2).is_some());
        assert!(ts.get_entity(3).is_none());
    }

    // -----------------------------------------------------------------------
    // LagCompensation
    // -----------------------------------------------------------------------

    #[test]
    fn test_lag_compensation_record_and_retrieve() {
        let mut lc = LagCompensation::new(64);

        let mut ts = TickState::new(10);
        ts.add_entity(EntityHitbox::new(1, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
        lc.record_tick(ts);

        assert_eq!(lc.history_len(), 1);
        assert!(lc.get_state_at_tick(10).is_some());
        assert!(lc.get_state_at_tick(11).is_none());
    }

    #[test]
    fn test_lag_compensation_history_limit() {
        let mut lc = LagCompensation::new(3);

        for i in 0..5 {
            let ts = TickState::new(i);
            lc.record_tick(ts);
        }

        assert_eq!(lc.history_len(), 3);
        assert!(lc.get_state_at_tick(0).is_none()); // evicted
        assert!(lc.get_state_at_tick(1).is_none()); // evicted
        assert!(lc.get_state_at_tick(2).is_some());
        assert!(lc.get_state_at_tick(3).is_some());
        assert!(lc.get_state_at_tick(4).is_some());
    }

    #[test]
    fn test_lag_compensation_tick_range() {
        let mut lc = LagCompensation::new(64);
        assert!(lc.tick_range().is_none());

        lc.record_tick(TickState::new(5));
        lc.record_tick(TickState::new(10));
        lc.record_tick(TickState::new(15));

        assert_eq!(lc.tick_range(), Some((5, 15)));
    }

    #[test]
    fn test_lag_compensation_raycast() {
        let mut lc = LagCompensation::new(64);

        let mut ts = TickState::new(10);
        // Place entity at (0, 0, 5) with unit half-extents.
        ts.add_entity(EntityHitbox::new(1, [0.0, 0.0, 5.0], [1.0, 1.0, 1.0]));
        // Place another entity at (0, 0, 20).
        ts.add_entity(EntityHitbox::new(2, [0.0, 0.0, 20.0], [1.0, 1.0, 1.0]));
        lc.record_tick(ts);

        // Ray from origin looking down +Z.
        let ray = Ray::new(
            glam::Vec3::ZERO,
            glam::Vec3::new(0.0, 0.0, 1.0),
        );

        let hits = lc.raycast_at_tick(10, ray);
        assert_eq!(hits.len(), 2);
        // Entity 1 should be hit first (closer).
        assert_eq!(hits[0].entity_id, 1);
        assert_eq!(hits[1].entity_id, 2);
        assert!(hits[0].distance < hits[1].distance);
    }

    #[test]
    fn test_lag_compensation_raycast_miss() {
        let mut lc = LagCompensation::new(64);

        let mut ts = TickState::new(10);
        ts.add_entity(EntityHitbox::new(1, [10.0, 10.0, 10.0], [1.0, 1.0, 1.0]));
        lc.record_tick(ts);

        // Ray from origin pointing in +Z (entity is at 10,10,10).
        let ray = Ray::new(
            glam::Vec3::ZERO,
            glam::Vec3::new(0.0, 0.0, 1.0),
        );

        let hits = lc.raycast_at_tick(10, ray);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_lag_compensation_raycast_nonexistent_tick() {
        let lc = LagCompensation::new(64);
        let ray = Ray::new(glam::Vec3::ZERO, glam::Vec3::new(0.0, 0.0, 1.0));
        let hits = lc.raycast_at_tick(99, ray);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_lag_compensation_verify_hit() {
        let mut lc = LagCompensation::new(64);

        let mut ts = TickState::new(10);
        ts.add_entity(EntityHitbox::new(1, [0.0, 0.0, 5.0], [1.0, 1.0, 1.0]));
        lc.record_tick(ts);

        // Should hit: ray from origin toward entity.
        let result = lc.verify_hit(10, 0, 1, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(result.unwrap());

        // Should miss: ray pointing away.
        let result = lc.verify_hit(10, 0, 1, [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(!result.unwrap());
    }

    #[test]
    fn test_lag_compensation_verify_hit_missing_tick() {
        let lc = LagCompensation::new(64);
        let result = lc.verify_hit(99, 0, 1, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_lag_compensation_verify_hit_missing_entity() {
        let mut lc = LagCompensation::new(64);
        let ts = TickState::new(10);
        lc.record_tick(ts);

        let result = lc.verify_hit(10, 0, 99, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_lag_compensation_clear() {
        let mut lc = LagCompensation::new(64);
        lc.record_tick(TickState::new(1));
        lc.record_tick(TickState::new(2));
        assert_eq!(lc.history_len(), 2);

        lc.clear();
        assert_eq!(lc.history_len(), 0);
    }

    // -----------------------------------------------------------------------
    // PredictionManager
    // -----------------------------------------------------------------------

    #[test]
    fn test_prediction_manager_predict() {
        let mut pm = PredictionManager::new(1, 64, 64);

        pm.predict(vec![1, 0], vec![10, 0, 0]);
        assert_eq!(pm.client_tick(), 1);
        assert_eq!(pm.prediction_history_len(), 1);

        pm.predict(vec![0, 1], vec![10, 5, 0]);
        assert_eq!(pm.client_tick(), 2);
        assert_eq!(pm.prediction_history_len(), 2);
    }

    #[test]
    fn test_prediction_manager_reconcile_match() {
        let mut pm = PredictionManager::new(1, 64, 64);

        // Predict tick 0 with state [10, 0, 0].
        pm.predict(vec![1, 0], vec![10, 0, 0]);

        // Server says tick 0 state is [10, 0, 0] (matches).
        let server = StateSnapshot::new(0, 1, vec![10, 0, 0]);
        let result = pm.reconcile(server);

        assert_eq!(result, ReconciliationResult::Match);
        assert_eq!(pm.misprediction_count(), 0);
    }

    #[test]
    fn test_prediction_manager_reconcile_misprediction() {
        let mut pm = PredictionManager::new(1, 64, 64);

        // Predict several ticks.
        pm.predict(vec![1, 0], vec![10, 0, 0]);
        pm.predict(vec![1, 0], vec![20, 0, 0]);
        pm.predict(vec![1, 0], vec![30, 0, 0]);

        // Server says tick 0 state is [15, 0, 0] (mismatch!).
        let server = StateSnapshot::new(0, 1, vec![15, 0, 0]);
        let result = pm.reconcile(server);

        match result {
            ReconciliationResult::Misprediction {
                divergence_tick,
                inputs_to_replay,
            } => {
                assert_eq!(divergence_tick, 0);
                assert_eq!(inputs_to_replay, 2); // ticks 1 and 2
            }
            _ => panic!("Expected misprediction"),
        }
        assert_eq!(pm.misprediction_count(), 1);
    }

    #[test]
    fn test_prediction_manager_inputs_to_replay() {
        let mut pm = PredictionManager::new(1, 64, 64);

        pm.predict(vec![1], vec![10]);
        pm.predict(vec![2], vec![20]);
        pm.predict(vec![3], vec![30]);
        pm.predict(vec![4], vec![40]);

        // Ack tick 1.
        let server = StateSnapshot::new(1, 1, vec![20]);
        pm.reconcile(server);

        // Ticks 2 and 3 should need replay.
        let to_replay = pm.inputs_to_replay();
        assert_eq!(to_replay.len(), 2);
        assert_eq!(to_replay[0].tick, 2);
        assert_eq!(to_replay[1].tick, 3);
    }

    #[test]
    fn test_prediction_manager_prediction_lead() {
        let mut pm = PredictionManager::new(1, 64, 64);

        pm.predict(vec![1], vec![10]);
        pm.predict(vec![2], vec![20]);
        pm.predict(vec![3], vec![30]);

        assert_eq!(pm.prediction_lead(), 3); // 3 ticks ahead, no server ack

        let server = StateSnapshot::new(1, 1, vec![20]);
        pm.reconcile(server);

        assert_eq!(pm.prediction_lead(), 2); // 3 - 1 = 2
    }

    #[test]
    fn test_prediction_manager_store_replayed() {
        let mut pm = PredictionManager::new(1, 64, 64);

        pm.predict(vec![1], vec![10]);
        pm.predict(vec![2], vec![20]);

        // After misprediction, store corrected states.
        let corrected = vec![
            StateSnapshot::new(0, 1, vec![15]),
            StateSnapshot::new(1, 1, vec![25]),
        ];
        pm.store_replayed_states(corrected);

        assert_eq!(pm.prediction_history_len(), 2);
    }

    #[test]
    fn test_prediction_manager_max_history() {
        let mut pm = PredictionManager::new(1, 64, 3);

        pm.predict(vec![1], vec![10]);
        pm.predict(vec![2], vec![20]);
        pm.predict(vec![3], vec![30]);
        pm.predict(vec![4], vec![40]); // should evict oldest prediction

        assert_eq!(pm.prediction_history_len(), 3);
    }

    #[test]
    fn test_prediction_manager_entity_id() {
        let pm = PredictionManager::new(42, 64, 64);
        assert_eq!(pm.entity_id(), 42);
    }

    // -----------------------------------------------------------------------
    // Integration: multiple reconciliations
    // -----------------------------------------------------------------------

    #[test]
    fn test_prediction_multi_reconcile() {
        let mut pm = PredictionManager::new(1, 64, 64);

        // Simulate 10 ticks of prediction.
        for i in 0..10 {
            pm.predict(vec![i as u8], vec![i as u8 * 10]);
        }
        assert_eq!(pm.client_tick(), 10);

        // Server acks tick 3 (matching).
        let server = StateSnapshot::new(3, 1, vec![30]);
        let result = pm.reconcile(server);
        assert_eq!(result, ReconciliationResult::Match);
        assert_eq!(pm.last_server_tick(), 3);

        // Server acks tick 7 (mismatch).
        let server = StateSnapshot::new(7, 1, vec![99]);
        let result = pm.reconcile(server);
        match result {
            ReconciliationResult::Misprediction {
                divergence_tick,
                inputs_to_replay,
            } => {
                assert_eq!(divergence_tick, 7);
                assert_eq!(inputs_to_replay, 2); // ticks 8, 9
            }
            _ => panic!("Expected misprediction"),
        }
        assert_eq!(pm.misprediction_count(), 1);
    }

    // -----------------------------------------------------------------------
    // InterpolationBuffer with sample
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpolation_three_snapshots() {
        let mut buf = InterpolationBuffer::new(10, 0.0, 10.0);

        buf.push(StateSnapshot::new(0, 1, vec![0]));
        buf.push(StateSnapshot::new(5, 1, vec![50]));
        buf.push(StateSnapshot::new(10, 1, vec![100]));

        // At tick 2.5 (between 0 and 5).
        let (before, after, t) = buf.interpolate_at_tick(2.5).unwrap();
        assert_eq!(before.tick, 0);
        assert_eq!(after.tick, 5);
        assert!((t - 0.5).abs() < 0.001);

        // At tick 7.5 (between 5 and 10).
        let (before, after, t) = buf.interpolate_at_tick(7.5).unwrap();
        assert_eq!(before.tick, 5);
        assert_eq!(after.tick, 10);
        assert!((t - 0.5).abs() < 0.001);
    }
}
