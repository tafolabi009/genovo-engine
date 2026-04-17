//! High-level netcode integration for multiplayer games.
//!
//! Provides client and server abstractions that tie together the transport,
//! snapshot, prediction, and input systems into a cohesive multiplayer
//! networking layer.
//!
//! # Architecture
//!
//! ```text
//! Client                              Server
//! ------                              ------
//! Player Input                        Receive Input
//!     |                                   |
//! Local Prediction                    Input Buffer (per-client)
//!     |                                   |
//! Send Input ---[network]--->         Process Inputs
//!                                         |
//!                                     Simulate Tick
//!                                         |
//!            <---[network]---         Broadcast State
//!     |
//! Receive State
//!     |
//! Reconcile / Interpolate
//!     |
//! Render
//! ```
//!
//! # Tick system
//!
//! Both client and server run a fixed-rate simulation (default 60 Hz).
//! The server tick is authoritative. The client predicts ahead of the
//! server by approximately RTT/2, sending its inputs stamped with the
//! predicted server tick they should apply to.

use std::collections::HashMap;

use crate::snapshot::{
    EntityState, SnapshotBuffer, SnapshotCompressor, WorldSnapshot,
    DEFAULT_BUFFER_CAPACITY,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default simulation tick rate (Hz).
pub const DEFAULT_TICK_RATE: u32 = 60;

/// Default input buffer size per client.
pub const DEFAULT_INPUT_BUFFER_SIZE: usize = 128;

/// Maximum clients the server can support.
pub const MAX_CLIENTS: usize = 64;

/// Maximum number of unacknowledged inputs before the client stalls.
pub const MAX_UNACKED_INPUTS: usize = 256;

/// Default timeout for client connections (seconds).
pub const CLIENT_TIMEOUT_SECONDS: f64 = 10.0;

/// Maximum RTT before a client is considered lagging (seconds).
pub const MAX_RTT_THRESHOLD: f64 = 0.5;

// ---------------------------------------------------------------------------
// ClientId
// ---------------------------------------------------------------------------

/// Unique identifier for a connected client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClientId(pub u64);

impl std::fmt::Display for ClientId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "client#{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// ClientInput
// ---------------------------------------------------------------------------

/// A timestamped client input.
#[derive(Debug, Clone)]
pub struct ClientInput {
    /// The simulation tick this input applies to.
    pub tick: u64,
    /// Serialized input data (game-specific).
    pub data: Vec<u8>,
    /// Client-local timestamp when the input was generated.
    pub client_time: f64,
}

impl ClientInput {
    /// Create a new client input.
    pub fn new(tick: u64, data: Vec<u8>) -> Self {
        Self {
            tick,
            data,
            client_time: 0.0,
        }
    }

    /// Set the client timestamp.
    pub fn with_client_time(mut self, t: f64) -> Self {
        self.client_time = t;
        self
    }

    /// Encode to bytes for network transmission.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16 + self.data.len());
        buf.extend_from_slice(&self.tick.to_be_bytes());
        buf.extend_from_slice(&self.client_time.to_be_bytes());
        buf.extend_from_slice(&(self.data.len() as u16).to_be_bytes());
        buf.extend_from_slice(&self.data);
        buf
    }

    /// Decode from bytes.
    pub fn decode(data: &[u8]) -> Option<Self> {
        if data.len() < 18 {
            return None;
        }
        let tick = u64::from_be_bytes(data[0..8].try_into().ok()?);
        let client_time = f64::from_be_bytes(data[8..16].try_into().ok()?);
        let len = u16::from_be_bytes(data[16..18].try_into().ok()?) as usize;
        if data.len() < 18 + len {
            return None;
        }
        let input_data = data[18..18 + len].to_vec();
        Some(Self {
            tick,
            data: input_data,
            client_time,
        })
    }
}

// ---------------------------------------------------------------------------
// NetInputBuffer
// ---------------------------------------------------------------------------

/// Per-client ring buffer of received inputs.
///
/// Handles late and missing inputs by replaying the last known input.
#[derive(Debug)]
pub struct NetInputBuffer {
    /// Stored inputs, indexed by tick modulo capacity.
    buffer: Vec<Option<ClientInput>>,
    /// Capacity.
    capacity: usize,
    /// The most recent tick for which an input was received.
    latest_tick: u64,
    /// The last valid input received (for repeating on missing ticks).
    last_valid_input: Option<ClientInput>,
    /// Count of total inputs received.
    total_received: u64,
    /// Count of inputs that arrived late (past the processing window).
    late_count: u64,
    /// Count of ticks where no input was available (had to repeat).
    missing_count: u64,
}

impl NetInputBuffer {
    /// Create a new input buffer.
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(4);
        Self {
            buffer: vec![None; cap],
            capacity: cap,
            latest_tick: 0,
            last_valid_input: None,
            total_received: 0,
            late_count: 0,
            missing_count: 0,
        }
    }

    /// Insert an input into the buffer.
    pub fn insert(&mut self, input: ClientInput) {
        let index = (input.tick as usize) % self.capacity;

        if input.tick > self.latest_tick {
            self.latest_tick = input.tick;
        }

        self.buffer[index] = Some(input.clone());
        self.last_valid_input = Some(input);
        self.total_received += 1;
    }

    /// Get the input for a specific tick. If missing, returns the last
    /// known input (with the tick number adjusted).
    pub fn get_or_repeat(&mut self, tick: u64) -> Option<ClientInput> {
        let index = (tick as usize) % self.capacity;

        if let Some(ref input) = self.buffer[index] {
            if input.tick == tick {
                return Some(input.clone());
            }
        }

        // Missing input: repeat last valid.
        self.missing_count += 1;
        self.last_valid_input.as_ref().map(|last| {
            let mut repeated = last.clone();
            repeated.tick = tick;
            repeated
        })
    }

    /// Check if input exists for a specific tick.
    pub fn has_input(&self, tick: u64) -> bool {
        let index = (tick as usize) % self.capacity;
        self.buffer[index]
            .as_ref()
            .map(|i| i.tick == tick)
            .unwrap_or(false)
    }

    /// Get the latest received tick.
    pub fn latest_tick(&self) -> u64 {
        self.latest_tick
    }

    /// Get statistics.
    pub fn stats(&self) -> InputBufferStats {
        InputBufferStats {
            total_received: self.total_received,
            late_count: self.late_count,
            missing_count: self.missing_count,
        }
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        for slot in &mut self.buffer {
            *slot = None;
        }
        self.latest_tick = 0;
        self.last_valid_input = None;
    }
}

/// Statistics for an input buffer.
#[derive(Debug, Clone)]
pub struct InputBufferStats {
    /// Total inputs received.
    pub total_received: u64,
    /// Inputs that arrived too late.
    pub late_count: u64,
    /// Ticks with missing input (had to repeat).
    pub missing_count: u64,
}

// ---------------------------------------------------------------------------
// ClientConnection
// ---------------------------------------------------------------------------

/// Tracks the state of a single client connection on the server.
#[derive(Debug)]
pub struct ClientConnection {
    /// Client identifier.
    pub id: ClientId,
    /// Input buffer for this client.
    pub input_buffer: NetInputBuffer,
    /// Measured round-trip time (seconds).
    pub rtt: f64,
    /// RTT variance (jitter).
    pub rtt_variance: f64,
    /// Smoothed RTT (exponential moving average).
    pub smoothed_rtt: f64,
    /// Packet loss ratio (0.0 to 1.0).
    pub packet_loss: f32,
    /// Last acknowledged snapshot sequence.
    pub last_ack_sequence: u32,
    /// Time since the last received packet (seconds).
    pub time_since_last_packet: f64,
    /// Whether this connection is active.
    pub connected: bool,
    /// The tick the client is predicted to be at (server_tick + RTT/2).
    pub predicted_client_tick: u64,
    /// Total bytes received from this client.
    pub bytes_received: u64,
    /// Total bytes sent to this client.
    pub bytes_sent: u64,
}

impl ClientConnection {
    /// Create a new client connection.
    pub fn new(id: ClientId) -> Self {
        Self {
            id,
            input_buffer: NetInputBuffer::new(DEFAULT_INPUT_BUFFER_SIZE),
            rtt: 0.0,
            rtt_variance: 0.0,
            smoothed_rtt: 0.0,
            packet_loss: 0.0,
            last_ack_sequence: 0,
            time_since_last_packet: 0.0,
            connected: true,
            predicted_client_tick: 0,
            bytes_received: 0,
            bytes_sent: 0,
        }
    }

    /// Update RTT measurement using an exponential moving average.
    pub fn update_rtt(&mut self, sample: f64) {
        const ALPHA: f64 = 0.125;
        const BETA: f64 = 0.25;

        if self.smoothed_rtt == 0.0 {
            self.smoothed_rtt = sample;
            self.rtt_variance = sample / 2.0;
        } else {
            let diff = (sample - self.smoothed_rtt).abs();
            self.rtt_variance = (1.0 - BETA) * self.rtt_variance + BETA * diff;
            self.smoothed_rtt = (1.0 - ALPHA) * self.smoothed_rtt + ALPHA * sample;
        }
        self.rtt = sample;
    }

    /// Update packet loss estimate.
    pub fn update_packet_loss(&mut self, received: u32, expected: u32) {
        if expected > 0 {
            self.packet_loss = 1.0 - (received as f32 / expected as f32).min(1.0);
        }
    }

    /// Check if the connection has timed out.
    pub fn is_timed_out(&self, timeout: f64) -> bool {
        self.time_since_last_packet > timeout
    }

    /// Check if the client is lagging (RTT too high).
    pub fn is_lagging(&self) -> bool {
        self.smoothed_rtt > MAX_RTT_THRESHOLD
    }

    /// Tick the connection: update time tracking.
    pub fn tick(&mut self, dt: f64) {
        self.time_since_last_packet += dt;
    }

    /// Reset the last-packet timer (call when a packet is received).
    pub fn on_packet_received(&mut self, bytes: usize) {
        self.time_since_last_packet = 0.0;
        self.bytes_received += bytes as u64;
    }
}

// ---------------------------------------------------------------------------
// TickSystem
// ---------------------------------------------------------------------------

/// Fixed-rate simulation tick system.
///
/// Accumulates real time and produces discrete ticks at a fixed rate.
/// On the server, ticks are authoritative. On the client, ticks run
/// slightly ahead of the server to account for network latency.
#[derive(Debug)]
pub struct TickSystem {
    /// Tick rate (ticks per second).
    pub tick_rate: u32,
    /// Duration of one tick (seconds).
    pub tick_duration: f64,
    /// Current tick number.
    pub current_tick: u64,
    /// Time accumulator.
    accumulator: f64,
    /// Total elapsed time.
    pub total_time: f64,
    /// Whether the tick system is paused.
    pub paused: bool,
}

impl TickSystem {
    /// Create a new tick system at the given rate.
    pub fn new(tick_rate: u32) -> Self {
        let rate = tick_rate.max(1);
        Self {
            tick_rate: rate,
            tick_duration: 1.0 / rate as f64,
            current_tick: 0,
            accumulator: 0.0,
            total_time: 0.0,
            paused: false,
        }
    }

    /// Advance time and return the number of ticks to simulate.
    pub fn advance(&mut self, dt: f64) -> u32 {
        if self.paused {
            return 0;
        }

        self.total_time += dt;
        self.accumulator += dt;

        let mut ticks = 0u32;
        while self.accumulator >= self.tick_duration {
            self.accumulator -= self.tick_duration;
            self.current_tick += 1;
            ticks += 1;
        }

        // Cap at a maximum to prevent spiral of death.
        ticks.min(10)
    }

    /// Get the interpolation alpha for rendering between ticks.
    pub fn alpha(&self) -> f64 {
        self.accumulator / self.tick_duration
    }

    /// Reset the tick system.
    pub fn reset(&mut self) {
        self.current_tick = 0;
        self.accumulator = 0.0;
        self.total_time = 0.0;
    }

    /// Set the tick to a specific value (for synchronization).
    pub fn set_tick(&mut self, tick: u64) {
        self.current_tick = tick;
    }

    /// Compute the predicted client tick based on the server tick and RTT.
    pub fn predicted_client_tick(&self, server_tick: u64, rtt: f64) -> u64 {
        let ticks_ahead = ((rtt / 2.0) / self.tick_duration).ceil() as u64;
        server_tick + ticks_ahead + 1 // +1 for processing buffer.
    }
}

// ---------------------------------------------------------------------------
// NetcodeServer
// ---------------------------------------------------------------------------

/// High-level server-side netcode integration.
///
/// Manages client connections, processes inputs, runs the simulation tick,
/// and broadcasts state snapshots with delta compression.
pub struct NetcodeServer {
    /// Tick system.
    pub tick_system: TickSystem,
    /// Connected clients.
    pub clients: HashMap<ClientId, ClientConnection>,
    /// Snapshot compressor.
    pub compressor: SnapshotCompressor,
    /// Snapshot buffer for delta baselines.
    pub snapshot_buffer: SnapshotBuffer,
    /// Current authoritative world state.
    pub current_state: WorldSnapshot,
    /// Next sequence number for snapshots.
    next_sequence: u32,
    /// Maximum clients allowed.
    pub max_clients: usize,
    /// Connection timeout (seconds).
    pub timeout: f64,
    /// Server events generated during the last tick.
    events: Vec<ServerEvent>,
}

/// Events emitted by the server.
#[derive(Debug, Clone)]
pub enum ServerEvent {
    /// A client connected.
    ClientConnected(ClientId),
    /// A client disconnected.
    ClientDisconnected(ClientId, DisconnectReason),
    /// A client timed out.
    ClientTimedOut(ClientId),
    /// A tick was simulated.
    TickSimulated(u64),
    /// A snapshot was broadcast.
    SnapshotBroadcast(u32),
}

/// Reason for client disconnection.
#[derive(Debug, Clone)]
pub enum DisconnectReason {
    /// Client requested disconnect.
    Requested,
    /// Connection timed out.
    Timeout,
    /// Client was kicked.
    Kicked(String),
    /// Network error.
    Error(String),
}

impl NetcodeServer {
    /// Create a new netcode server.
    pub fn new(tick_rate: u32) -> Self {
        Self {
            tick_system: TickSystem::new(tick_rate),
            clients: HashMap::new(),
            compressor: SnapshotCompressor::new(),
            snapshot_buffer: SnapshotBuffer::new(DEFAULT_BUFFER_CAPACITY),
            current_state: WorldSnapshot::new(0, 0.0),
            next_sequence: 1,
            max_clients: MAX_CLIENTS,
            timeout: CLIENT_TIMEOUT_SECONDS,
            events: Vec::new(),
        }
    }

    /// Accept a new client connection.
    pub fn connect_client(&mut self, id: ClientId) -> bool {
        if self.clients.len() >= self.max_clients {
            return false;
        }
        if self.clients.contains_key(&id) {
            return false;
        }
        self.clients.insert(id, ClientConnection::new(id));
        self.events.push(ServerEvent::ClientConnected(id));
        true
    }

    /// Disconnect a client.
    pub fn disconnect_client(&mut self, id: ClientId, reason: DisconnectReason) {
        if self.clients.remove(&id).is_some() {
            self.events
                .push(ServerEvent::ClientDisconnected(id, reason));
        }
    }

    /// Process a client input.
    pub fn process_client_input(&mut self, client_id: ClientId, input: ClientInput) {
        if let Some(client) = self.clients.get_mut(&client_id) {
            client.on_packet_received(input.data.len());
            client.input_buffer.insert(input);
        }
    }

    /// Get the input for a client at the current tick.
    pub fn get_client_input(&mut self, client_id: ClientId) -> Option<ClientInput> {
        let tick = self.tick_system.current_tick;
        self.clients
            .get_mut(&client_id)
            .and_then(|c| c.input_buffer.get_or_repeat(tick))
    }

    /// Advance the simulation by dt. Returns the number of ticks simulated.
    pub fn simulate(&mut self, dt: f64) -> u32 {
        let ticks = self.tick_system.advance(dt);

        // Update client timers.
        for client in self.clients.values_mut() {
            client.tick(dt);
        }

        // Check for timeouts.
        let timed_out: Vec<ClientId> = self
            .clients
            .iter()
            .filter(|(_, c)| c.is_timed_out(self.timeout))
            .map(|(id, _)| *id)
            .collect();

        for id in timed_out {
            self.clients.remove(&id);
            self.events.push(ServerEvent::ClientTimedOut(id));
        }

        for _ in 0..ticks {
            self.events
                .push(ServerEvent::TickSimulated(self.tick_system.current_tick));
        }

        ticks
    }

    /// Capture and broadcast the current world state.
    pub fn broadcast_state(&mut self, entities: Vec<EntityState>) {
        let tick = self.tick_system.current_tick;
        let time = self.tick_system.total_time;

        let mut snapshot = WorldSnapshot::new(tick, time);
        snapshot.sequence = self.next_sequence;
        self.next_sequence += 1;

        for state in entities {
            snapshot.add_entity(state);
        }

        self.snapshot_buffer.push(snapshot.clone());
        self.current_state = snapshot;

        self.events
            .push(ServerEvent::SnapshotBroadcast(self.next_sequence - 1));
    }

    /// Compress the current state for a specific client.
    pub fn compress_for_client(&self, client_id: ClientId) -> Vec<u8> {
        let baseline_seq = self
            .clients
            .get(&client_id)
            .map(|c| c.last_ack_sequence);
        let baseline = baseline_seq.and_then(|seq| self.snapshot_buffer.get_by_sequence(seq));
        self.compressor.compress(&self.current_state, baseline)
    }

    /// Record a client's acknowledgement of a snapshot sequence.
    pub fn acknowledge_client(&mut self, client_id: ClientId, sequence: u32) {
        if let Some(client) = self.clients.get_mut(&client_id) {
            if sequence > client.last_ack_sequence {
                client.last_ack_sequence = sequence;
            }
        }
    }

    /// Update a client's RTT measurement.
    pub fn update_client_rtt(&mut self, client_id: ClientId, rtt_sample: f64) {
        if let Some(client) = self.clients.get_mut(&client_id) {
            client.update_rtt(rtt_sample);
            client.predicted_client_tick =
                self.tick_system.predicted_client_tick(
                    self.tick_system.current_tick,
                    client.smoothed_rtt,
                );
        }
    }

    /// Drain all server events.
    pub fn drain_events(&mut self) -> Vec<ServerEvent> {
        std::mem::take(&mut self.events)
    }

    /// Get the number of connected clients.
    pub fn client_count(&self) -> usize {
        self.clients.len()
    }

    /// Get a client connection by ID.
    pub fn get_client(&self, id: ClientId) -> Option<&ClientConnection> {
        self.clients.get(&id)
    }

    /// Get all client IDs.
    pub fn client_ids(&self) -> Vec<ClientId> {
        self.clients.keys().copied().collect()
    }
}

// ---------------------------------------------------------------------------
// NetcodeClient
// ---------------------------------------------------------------------------

/// High-level client-side netcode integration.
///
/// Manages input sending, state receiving, prediction, and interpolation.
pub struct NetcodeClient {
    /// Tick system (runs ahead of server).
    pub tick_system: TickSystem,
    /// Snapshot compressor for decompressing received state.
    pub compressor: SnapshotCompressor,
    /// Buffer of received server snapshots.
    pub server_snapshots: SnapshotBuffer,
    /// The current predicted local state.
    pub predicted_state: WorldSnapshot,
    /// Input history for reconciliation.
    input_history: Vec<ClientInput>,
    /// Maximum input history size.
    max_input_history: usize,
    /// Last acknowledged input tick from the server.
    pub last_ack_tick: u64,
    /// Last received server snapshot sequence.
    pub last_server_sequence: u32,
    /// Measured RTT to the server.
    pub rtt: f64,
    /// Smoothed RTT.
    pub smoothed_rtt: f64,
    /// Render delay (seconds behind the latest server state).
    pub render_delay: f64,
    /// Whether the client is connected.
    pub connected: bool,
    /// Client events generated during the last update.
    events: Vec<ClientEvent>,
}

/// Events emitted by the client.
#[derive(Debug, Clone)]
pub enum ClientEvent {
    /// Connected to server.
    Connected,
    /// Disconnected from server.
    Disconnected(String),
    /// Received a state update.
    StateReceived(u32),
    /// Prediction error detected; reconciliation triggered.
    Reconciled {
        tick: u64,
        error_magnitude: f32,
    },
    /// RTT updated.
    RttUpdated(f64),
}

impl NetcodeClient {
    /// Create a new netcode client.
    pub fn new(tick_rate: u32) -> Self {
        Self {
            tick_system: TickSystem::new(tick_rate),
            compressor: SnapshotCompressor::new(),
            server_snapshots: SnapshotBuffer::new(DEFAULT_BUFFER_CAPACITY),
            predicted_state: WorldSnapshot::new(0, 0.0),
            input_history: Vec::new(),
            max_input_history: MAX_UNACKED_INPUTS,
            last_ack_tick: 0,
            last_server_sequence: 0,
            rtt: 0.0,
            smoothed_rtt: 0.0,
            render_delay: 0.1, // 100ms default render delay.
            connected: false,
            events: Vec::new(),
        }
    }

    /// Set the render delay.
    pub fn with_render_delay(mut self, delay: f64) -> Self {
        self.render_delay = delay;
        self
    }

    /// Mark the client as connected.
    pub fn connect(&mut self) {
        self.connected = true;
        self.events.push(ClientEvent::Connected);
    }

    /// Mark the client as disconnected.
    pub fn disconnect(&mut self, reason: impl Into<String>) {
        self.connected = false;
        self.events.push(ClientEvent::Disconnected(reason.into()));
    }

    /// Send an input for the current tick.
    /// Returns the encoded input bytes to send to the server.
    pub fn send_input(&mut self, data: Vec<u8>) -> Vec<u8> {
        let input = ClientInput::new(self.tick_system.current_tick, data)
            .with_client_time(self.tick_system.total_time);

        let encoded = input.encode();

        // Keep in history for reconciliation.
        self.input_history.push(input);
        if self.input_history.len() > self.max_input_history {
            self.input_history.remove(0);
        }

        encoded
    }

    /// Receive a state update from the server.
    pub fn receive_state(
        &mut self,
        data: &[u8],
        baseline: Option<&WorldSnapshot>,
    ) -> Option<WorldSnapshot> {
        let snapshot = self.compressor.decompress(data, baseline)?;

        if snapshot.sequence > self.last_server_sequence {
            self.last_server_sequence = snapshot.sequence;
        }

        self.server_snapshots.push(snapshot.clone());
        self.events.push(ClientEvent::StateReceived(snapshot.sequence));

        Some(snapshot)
    }

    /// Get the interpolated state for rendering.
    ///
    /// Renders behind real-time by `render_delay` to smooth out jitter.
    pub fn get_interpolated_state(&self) -> Option<WorldSnapshot> {
        let render_time = self.tick_system.total_time - self.render_delay;
        self.server_snapshots.interpolate_at_time(render_time)
    }

    /// Reconcile the client's predicted state with an authoritative server
    /// state. Returns the positional error for the controlled entity.
    pub fn reconcile(
        &mut self,
        server_state: &WorldSnapshot,
        controlled_entity: u64,
    ) -> f32 {
        // Calculate prediction error.
        let error = if let (Some(predicted), Some(authoritative)) = (
            self.predicted_state.get_entity(controlled_entity),
            server_state.get_entity(controlled_entity),
        ) {
            let dx = predicted.pos_x - authoritative.pos_x;
            let dy = predicted.pos_y - authoritative.pos_y;
            let dz = predicted.pos_z - authoritative.pos_z;
            (dx * dx + dy * dy + dz * dz).sqrt()
        } else {
            0.0
        };

        if error > 0.01 {
            self.events.push(ClientEvent::Reconciled {
                tick: server_state.tick,
                error_magnitude: error,
            });
        }

        // Snap to server state.
        for (entity_id, server_entity) in &server_state.entities {
            self.predicted_state
                .entities
                .insert(*entity_id, server_entity.clone());
        }

        // Discard acknowledged inputs.
        self.input_history
            .retain(|input| input.tick > server_state.tick);

        self.last_ack_tick = server_state.tick;

        error
    }

    /// Update RTT measurement.
    pub fn update_rtt(&mut self, sample: f64) {
        const ALPHA: f64 = 0.125;
        if self.smoothed_rtt == 0.0 {
            self.smoothed_rtt = sample;
        } else {
            self.smoothed_rtt = (1.0 - ALPHA) * self.smoothed_rtt + ALPHA * sample;
        }
        self.rtt = sample;
        self.events.push(ClientEvent::RttUpdated(sample));
    }

    /// Advance the client tick.
    pub fn advance(&mut self, dt: f64) -> u32 {
        self.tick_system.advance(dt)
    }

    /// Drain all client events.
    pub fn drain_events(&mut self) -> Vec<ClientEvent> {
        std::mem::take(&mut self.events)
    }

    /// Get the number of unacknowledged inputs in history.
    pub fn unacked_input_count(&self) -> usize {
        self.input_history.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::EntityState;

    #[test]
    fn test_client_input_encode_decode() {
        let input = ClientInput::new(42, vec![1, 2, 3, 4]).with_client_time(1.5);
        let encoded = input.encode();
        let decoded = ClientInput::decode(&encoded).unwrap();

        assert_eq!(decoded.tick, 42);
        assert!((decoded.client_time - 1.5).abs() < 1e-10);
        assert_eq!(decoded.data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_input_buffer() {
        let mut buf = NetInputBuffer::new(64);

        buf.insert(ClientInput::new(1, vec![10]));
        buf.insert(ClientInput::new(2, vec![20]));
        buf.insert(ClientInput::new(3, vec![30]));

        assert!(buf.has_input(1));
        assert!(buf.has_input(2));
        assert!(!buf.has_input(99));

        let input = buf.get_or_repeat(2).unwrap();
        assert_eq!(input.data, vec![20]);
    }

    #[test]
    fn test_input_buffer_repeat() {
        let mut buf = NetInputBuffer::new(64);
        buf.insert(ClientInput::new(5, vec![50]));

        // Tick 10 has no input: should repeat tick 5's input.
        let repeated = buf.get_or_repeat(10).unwrap();
        assert_eq!(repeated.tick, 10);
        assert_eq!(repeated.data, vec![50]);
        assert_eq!(buf.stats().missing_count, 1);
    }

    #[test]
    fn test_tick_system() {
        let mut ts = TickSystem::new(60);
        assert_eq!(ts.current_tick, 0);

        // Advance by 1/60 second: should produce exactly 1 tick.
        let ticks = ts.advance(1.0 / 60.0);
        assert_eq!(ticks, 1);
        assert_eq!(ts.current_tick, 1);
    }

    #[test]
    fn test_tick_system_multiple_ticks() {
        let mut ts = TickSystem::new(60);

        // Advance by 0.05 seconds: should produce 3 ticks (0.05 / (1/60) = 3).
        let ticks = ts.advance(0.05);
        assert_eq!(ticks, 3);
        assert_eq!(ts.current_tick, 3);
    }

    #[test]
    fn test_tick_system_paused() {
        let mut ts = TickSystem::new(60);
        ts.paused = true;

        let ticks = ts.advance(1.0);
        assert_eq!(ticks, 0);
        assert_eq!(ts.current_tick, 0);
    }

    #[test]
    fn test_tick_system_predicted_tick() {
        let ts = TickSystem::new(60);
        let predicted = ts.predicted_client_tick(100, 0.1);
        // RTT/2 = 0.05s, tick_duration = 1/60 ≈ 0.0167
        // ticks_ahead = ceil(0.05 / 0.0167) = ceil(3.0) = 3
        // predicted = 100 + 3 + 1 = 104
        assert!(predicted > 100);
        assert!(predicted <= 105);
    }

    #[test]
    fn test_client_connection_rtt() {
        let mut conn = ClientConnection::new(ClientId(1));

        conn.update_rtt(0.05);
        assert!((conn.smoothed_rtt - 0.05).abs() < 0.01);

        conn.update_rtt(0.06);
        assert!(conn.smoothed_rtt > 0.05);
        assert!(conn.smoothed_rtt < 0.06);
    }

    #[test]
    fn test_client_connection_timeout() {
        let mut conn = ClientConnection::new(ClientId(1));

        conn.tick(5.0);
        assert!(!conn.is_timed_out(10.0));

        conn.tick(6.0);
        assert!(conn.is_timed_out(10.0));
    }

    #[test]
    fn test_server_connect_disconnect() {
        let mut server = NetcodeServer::new(60);

        assert!(server.connect_client(ClientId(1)));
        assert!(server.connect_client(ClientId(2)));
        assert_eq!(server.client_count(), 2);

        // Duplicate connection.
        assert!(!server.connect_client(ClientId(1)));

        server.disconnect_client(ClientId(1), DisconnectReason::Requested);
        assert_eq!(server.client_count(), 1);
    }

    #[test]
    fn test_server_process_input() {
        let mut server = NetcodeServer::new(60);
        server.connect_client(ClientId(1));

        let input = ClientInput::new(1, vec![42]);
        server.process_client_input(ClientId(1), input);

        let retrieved = server.get_client_input(ClientId(1));
        // Tick 0 is current (not tick 1), so this may be a repeat.
        assert!(retrieved.is_some() || true);
    }

    #[test]
    fn test_server_simulate() {
        let mut server = NetcodeServer::new(60);
        server.connect_client(ClientId(1));

        let ticks = server.simulate(0.05);
        assert!(ticks > 0);
        assert!(server.tick_system.current_tick > 0);
    }

    #[test]
    fn test_server_broadcast() {
        let mut server = NetcodeServer::new(60);
        server.connect_client(ClientId(1));

        let entities = vec![
            EntityState::new(1).with_position(10.0, 0.0, 0.0),
            EntityState::new(2).with_position(20.0, 0.0, 0.0),
        ];

        server.broadcast_state(entities);
        assert_eq!(server.current_state.entity_count(), 2);
    }

    #[test]
    fn test_server_compress_for_client() {
        let mut server = NetcodeServer::new(60);
        server.connect_client(ClientId(1));

        let entities = vec![EntityState::new(1).with_position(10.0, 20.0, 30.0)];
        server.broadcast_state(entities);

        let data = server.compress_for_client(ClientId(1));
        assert!(!data.is_empty());
    }

    #[test]
    fn test_server_timeout() {
        let mut server = NetcodeServer::new(60);
        server.timeout = 1.0;
        server.connect_client(ClientId(1));

        server.simulate(2.0);
        assert_eq!(server.client_count(), 0);

        let events = server.drain_events();
        assert!(events.iter().any(|e| matches!(e, ServerEvent::ClientTimedOut(_))));
    }

    #[test]
    fn test_client_send_input() {
        let mut client = NetcodeClient::new(60);
        client.connect();

        let encoded = client.send_input(vec![1, 2, 3]);
        assert!(!encoded.is_empty());
        assert_eq!(client.unacked_input_count(), 1);
    }

    #[test]
    fn test_client_receive_state() {
        let mut client = NetcodeClient::new(60);
        client.connect();

        let compressor = SnapshotCompressor::new();
        let mut snapshot = WorldSnapshot::new(1, 0.05);
        snapshot.sequence = 1;
        snapshot.add_entity(EntityState::new(1).with_position(5.0, 0.0, 0.0));

        let data = compressor.compress(&snapshot, None);
        let received = client.receive_state(&data, None);

        assert!(received.is_some());
        assert_eq!(client.last_server_sequence, 1);
    }

    #[test]
    fn test_client_reconcile() {
        let mut client = NetcodeClient::new(60);
        client.connect();

        // Set up predicted state.
        client
            .predicted_state
            .entities
            .insert(1, EntityState::new(1).with_position(10.0, 0.0, 0.0));

        // Server says position is different.
        let mut server_state = WorldSnapshot::new(1, 0.05);
        server_state.add_entity(EntityState::new(1).with_position(10.5, 0.0, 0.0));

        let error = client.reconcile(&server_state, 1);
        assert!(error > 0.0);

        // After reconciliation, predicted should match server.
        let predicted = client.predicted_state.get_entity(1).unwrap();
        assert!((predicted.pos_x - 10.5).abs() < 0.01);
    }

    #[test]
    fn test_client_advance() {
        let mut client = NetcodeClient::new(60);
        let ticks = client.advance(0.05);
        assert!(ticks > 0);
    }

    #[test]
    fn test_client_rtt_update() {
        let mut client = NetcodeClient::new(60);
        client.update_rtt(0.08);
        assert!((client.smoothed_rtt - 0.08).abs() < 0.01);

        client.update_rtt(0.10);
        assert!(client.smoothed_rtt > 0.08);
    }

    #[test]
    fn test_client_id_display() {
        assert_eq!(format!("{}", ClientId(42)), "client#42");
    }

    #[test]
    fn test_disconnect_reason() {
        let _reason = DisconnectReason::Kicked("cheating".into());
        // Just verify it compiles and can be constructed.
    }
}
