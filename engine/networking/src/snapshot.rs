//! Network snapshot system for competitive multiplayer.
//!
//! Captures, transmits, and interpolates authoritative world state snapshots
//! at a fixed rate. Provides delta compression to minimize bandwidth by only
//! sending what changed since the last acknowledged snapshot.
//!
//! # Architecture
//!
//! ```text
//! Server:
//!   World --> SnapshotSystem::capture() --> WorldSnapshot
//!                                               |
//!                        SnapshotCompressor::compress(current, baseline)
//!                                               |
//!                                          DeltaPacket -----> Network
//!
//! Client:
//!   Network -----> DeltaPacket
//!                      |
//!       SnapshotCompressor::decompress(data, baseline)
//!                      |
//!                 WorldSnapshot
//!                      |
//!        SnapshotBuffer::interpolate_between(a, b, alpha)
//!                      |
//!                 Rendered State
//! ```
//!
//! # Delta compression
//!
//! The compressor encodes only the differences between the current snapshot
//! and a baseline (the last snapshot acknowledged by the client). For each
//! entity, a bitmask indicates which fields changed. Floating-point values
//! are quantized to fixed-point integers for compact encoding.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default snapshot rate (snapshots per second).
pub const DEFAULT_SNAPSHOT_RATE: f32 = 20.0;

/// Default snapshot buffer capacity.
pub const DEFAULT_BUFFER_CAPACITY: usize = 128;

/// Quantization precision for position values (units).
pub const POSITION_QUANTIZE_PRECISION: f32 = 0.01;

/// Quantization precision for rotation values (radians).
pub const ROTATION_QUANTIZE_PRECISION: f32 = 0.001;

/// Quantization precision for velocity values.
pub const VELOCITY_QUANTIZE_PRECISION: f32 = 0.05;

/// Maximum entities per snapshot.
pub const MAX_ENTITIES_PER_SNAPSHOT: usize = 1024;

/// Maximum snapshot size in bytes (for safety).
pub const MAX_SNAPSHOT_BYTES: usize = 65536;

/// Entity state field flags.
pub const FIELD_POSITION_X: u16 = 1 << 0;
pub const FIELD_POSITION_Y: u16 = 1 << 1;
pub const FIELD_POSITION_Z: u16 = 1 << 2;
pub const FIELD_ROTATION_X: u16 = 1 << 3;
pub const FIELD_ROTATION_Y: u16 = 1 << 4;
pub const FIELD_ROTATION_Z: u16 = 1 << 5;
pub const FIELD_ROTATION_W: u16 = 1 << 6;
pub const FIELD_VELOCITY_X: u16 = 1 << 7;
pub const FIELD_VELOCITY_Y: u16 = 1 << 8;
pub const FIELD_VELOCITY_Z: u16 = 1 << 9;
pub const FIELD_HEALTH: u16 = 1 << 10;
pub const FIELD_FLAGS: u16 = 1 << 11;
pub const FIELD_ANIM_STATE: u16 = 1 << 12;
pub const FIELD_CUSTOM_A: u16 = 1 << 13;
pub const FIELD_CUSTOM_B: u16 = 1 << 14;
pub const FIELD_ALL: u16 = 0x7FFF;

/// Epsilon for float comparison during delta detection.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// EntityState
// ---------------------------------------------------------------------------

/// The replicated state of a single entity in a snapshot.
#[derive(Debug, Clone)]
pub struct EntityState {
    /// Network entity identifier.
    pub entity_id: u64,
    /// Position X.
    pub pos_x: f32,
    /// Position Y.
    pub pos_y: f32,
    /// Position Z.
    pub pos_z: f32,
    /// Rotation quaternion X.
    pub rot_x: f32,
    /// Rotation quaternion Y.
    pub rot_y: f32,
    /// Rotation quaternion Z.
    pub rot_z: f32,
    /// Rotation quaternion W.
    pub rot_w: f32,
    /// Velocity X.
    pub vel_x: f32,
    /// Velocity Y.
    pub vel_y: f32,
    /// Velocity Z.
    pub vel_z: f32,
    /// Health (or similar bounded stat).
    pub health: f32,
    /// Bitfield of boolean flags (crouching, sprinting, etc.).
    pub flags: u32,
    /// Animation state identifier.
    pub anim_state: u16,
    /// Custom data slot A.
    pub custom_a: f32,
    /// Custom data slot B.
    pub custom_b: f32,
}

impl EntityState {
    /// Create a new entity state with default values.
    pub fn new(entity_id: u64) -> Self {
        Self {
            entity_id,
            pos_x: 0.0,
            pos_y: 0.0,
            pos_z: 0.0,
            rot_x: 0.0,
            rot_y: 0.0,
            rot_z: 0.0,
            rot_w: 1.0,
            vel_x: 0.0,
            vel_y: 0.0,
            vel_z: 0.0,
            health: 100.0,
            flags: 0,
            anim_state: 0,
            custom_a: 0.0,
            custom_b: 0.0,
        }
    }

    /// Set position.
    pub fn with_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.pos_x = x;
        self.pos_y = y;
        self.pos_z = z;
        self
    }

    /// Set rotation quaternion.
    pub fn with_rotation(mut self, x: f32, y: f32, z: f32, w: f32) -> Self {
        self.rot_x = x;
        self.rot_y = y;
        self.rot_z = z;
        self.rot_w = w;
        self
    }

    /// Set velocity.
    pub fn with_velocity(mut self, x: f32, y: f32, z: f32) -> Self {
        self.vel_x = x;
        self.vel_y = y;
        self.vel_z = z;
        self
    }

    /// Set health.
    pub fn with_health(mut self, h: f32) -> Self {
        self.health = h;
        self
    }

    /// Compute a bitmask of fields that differ between `self` and `other`.
    pub fn diff_mask(&self, other: &EntityState) -> u16 {
        let mut mask: u16 = 0;

        if (self.pos_x - other.pos_x).abs() > POSITION_QUANTIZE_PRECISION {
            mask |= FIELD_POSITION_X;
        }
        if (self.pos_y - other.pos_y).abs() > POSITION_QUANTIZE_PRECISION {
            mask |= FIELD_POSITION_Y;
        }
        if (self.pos_z - other.pos_z).abs() > POSITION_QUANTIZE_PRECISION {
            mask |= FIELD_POSITION_Z;
        }
        if (self.rot_x - other.rot_x).abs() > ROTATION_QUANTIZE_PRECISION {
            mask |= FIELD_ROTATION_X;
        }
        if (self.rot_y - other.rot_y).abs() > ROTATION_QUANTIZE_PRECISION {
            mask |= FIELD_ROTATION_Y;
        }
        if (self.rot_z - other.rot_z).abs() > ROTATION_QUANTIZE_PRECISION {
            mask |= FIELD_ROTATION_Z;
        }
        if (self.rot_w - other.rot_w).abs() > ROTATION_QUANTIZE_PRECISION {
            mask |= FIELD_ROTATION_W;
        }
        if (self.vel_x - other.vel_x).abs() > VELOCITY_QUANTIZE_PRECISION {
            mask |= FIELD_VELOCITY_X;
        }
        if (self.vel_y - other.vel_y).abs() > VELOCITY_QUANTIZE_PRECISION {
            mask |= FIELD_VELOCITY_Y;
        }
        if (self.vel_z - other.vel_z).abs() > VELOCITY_QUANTIZE_PRECISION {
            mask |= FIELD_VELOCITY_Z;
        }
        if (self.health - other.health).abs() > EPSILON {
            mask |= FIELD_HEALTH;
        }
        if self.flags != other.flags {
            mask |= FIELD_FLAGS;
        }
        if self.anim_state != other.anim_state {
            mask |= FIELD_ANIM_STATE;
        }
        if (self.custom_a - other.custom_a).abs() > EPSILON {
            mask |= FIELD_CUSTOM_A;
        }
        if (self.custom_b - other.custom_b).abs() > EPSILON {
            mask |= FIELD_CUSTOM_B;
        }

        mask
    }

    /// Linearly interpolate between two entity states.
    pub fn interpolate(a: &EntityState, b: &EntityState, alpha: f32) -> EntityState {
        let lerp = |x: f32, y: f32| x + (y - x) * alpha;
        let inv = 1.0 - alpha;

        EntityState {
            entity_id: b.entity_id,
            pos_x: lerp(a.pos_x, b.pos_x),
            pos_y: lerp(a.pos_y, b.pos_y),
            pos_z: lerp(a.pos_z, b.pos_z),
            // Quaternion interpolation: simple nlerp for efficiency.
            rot_x: lerp(a.rot_x, b.rot_x),
            rot_y: lerp(a.rot_y, b.rot_y),
            rot_z: lerp(a.rot_z, b.rot_z),
            rot_w: lerp(a.rot_w, b.rot_w),
            vel_x: lerp(a.vel_x, b.vel_x),
            vel_y: lerp(a.vel_y, b.vel_y),
            vel_z: lerp(a.vel_z, b.vel_z),
            health: lerp(a.health, b.health),
            // Non-interpolatable fields: use latest.
            flags: if alpha >= 0.5 { b.flags } else { a.flags },
            anim_state: if alpha >= 0.5 { b.anim_state } else { a.anim_state },
            custom_a: lerp(a.custom_a, b.custom_a),
            custom_b: lerp(a.custom_b, b.custom_b),
        }
    }

    /// Normalize the rotation quaternion.
    pub fn normalize_rotation(&mut self) {
        let len = (self.rot_x * self.rot_x
            + self.rot_y * self.rot_y
            + self.rot_z * self.rot_z
            + self.rot_w * self.rot_w)
            .sqrt();
        if len > EPSILON {
            let inv = 1.0 / len;
            self.rot_x *= inv;
            self.rot_y *= inv;
            self.rot_z *= inv;
            self.rot_w *= inv;
        }
    }
}

// ---------------------------------------------------------------------------
// WorldSnapshot
// ---------------------------------------------------------------------------

/// A complete snapshot of the world state at a specific simulation tick.
#[derive(Debug, Clone)]
pub struct WorldSnapshot {
    /// The simulation tick this snapshot represents.
    pub tick: u64,
    /// Server timestamp when this snapshot was created.
    pub server_time: f64,
    /// All entity states in this snapshot.
    pub entities: HashMap<u64, EntityState>,
    /// Sequence number (for ordering and acknowledgement).
    pub sequence: u32,
}

impl WorldSnapshot {
    /// Create an empty snapshot at the given tick.
    pub fn new(tick: u64, server_time: f64) -> Self {
        Self {
            tick,
            server_time,
            entities: HashMap::new(),
            sequence: 0,
        }
    }

    /// Add an entity state to the snapshot.
    pub fn add_entity(&mut self, state: EntityState) {
        if self.entities.len() < MAX_ENTITIES_PER_SNAPSHOT {
            self.entities.insert(state.entity_id, state);
        }
    }

    /// Remove an entity from the snapshot.
    pub fn remove_entity(&mut self, entity_id: u64) -> Option<EntityState> {
        self.entities.remove(&entity_id)
    }

    /// Get an entity state by ID.
    pub fn get_entity(&self, entity_id: u64) -> Option<&EntityState> {
        self.entities.get(&entity_id)
    }

    /// Get a mutable entity state.
    pub fn get_entity_mut(&mut self, entity_id: u64) -> Option<&mut EntityState> {
        self.entities.get_mut(&entity_id)
    }

    /// Returns the number of entities in this snapshot.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Returns all entity IDs.
    pub fn entity_ids(&self) -> Vec<u64> {
        self.entities.keys().copied().collect()
    }

    /// Interpolate between two snapshots at the given alpha (0..1).
    pub fn interpolate(a: &WorldSnapshot, b: &WorldSnapshot, alpha: f32) -> WorldSnapshot {
        let mut result = WorldSnapshot::new(
            b.tick,
            a.server_time + (b.server_time - a.server_time) * alpha as f64,
        );

        // Entities present in both snapshots are interpolated.
        for (id, state_b) in &b.entities {
            if let Some(state_a) = a.entities.get(id) {
                let mut interpolated = EntityState::interpolate(state_a, state_b, alpha);
                interpolated.normalize_rotation();
                result.entities.insert(*id, interpolated);
            } else {
                // Entity only in b: snap to b.
                result.entities.insert(*id, state_b.clone());
            }
        }

        // Entities only in a: include if alpha < 0.5 (they're disappearing).
        for (id, state_a) in &a.entities {
            if !b.entities.contains_key(id) && alpha < 0.5 {
                result.entities.insert(*id, state_a.clone());
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// SnapshotBuffer
// ---------------------------------------------------------------------------

/// Circular buffer of recent world snapshots.
///
/// Used on both server (history for delta baselines) and client (for
/// interpolation between received snapshots).
#[derive(Debug)]
pub struct SnapshotBuffer {
    /// Stored snapshots in order.
    buffer: Vec<WorldSnapshot>,
    /// Maximum number of snapshots to retain.
    capacity: usize,
    /// Write cursor (next insertion index).
    write_pos: usize,
    /// Number of snapshots currently stored.
    count: usize,
}

impl SnapshotBuffer {
    /// Create a new buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(2);
        Self {
            buffer: Vec::with_capacity(cap),
            capacity: cap,
            write_pos: 0,
            count: 0,
        }
    }

    /// Push a new snapshot into the buffer.
    pub fn push(&mut self, snapshot: WorldSnapshot) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(snapshot);
        } else {
            self.buffer[self.write_pos] = snapshot;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
        self.count = (self.count + 1).min(self.capacity);
    }

    /// Get a snapshot by tick number.
    pub fn get_snapshot_at_tick(&self, tick: u64) -> Option<&WorldSnapshot> {
        self.buffer.iter().find(|s| s.tick == tick)
    }

    /// Get a snapshot by sequence number.
    pub fn get_by_sequence(&self, seq: u32) -> Option<&WorldSnapshot> {
        self.buffer.iter().find(|s| s.sequence == seq)
    }

    /// Get the most recent snapshot.
    pub fn latest(&self) -> Option<&WorldSnapshot> {
        if self.buffer.is_empty() {
            return None;
        }
        self.buffer.iter().max_by_key(|s| s.tick)
    }

    /// Get the two snapshots bracketing the given tick for interpolation.
    /// Returns (before, after) where before.tick <= tick <= after.tick.
    pub fn get_interpolation_pair(&self, tick: u64) -> Option<(&WorldSnapshot, &WorldSnapshot)> {
        let mut before: Option<&WorldSnapshot> = None;
        let mut after: Option<&WorldSnapshot> = None;

        for snapshot in &self.buffer {
            if snapshot.tick <= tick {
                if before.is_none() || snapshot.tick > before.unwrap().tick {
                    before = Some(snapshot);
                }
            }
            if snapshot.tick >= tick {
                if after.is_none() || snapshot.tick < after.unwrap().tick {
                    after = Some(snapshot);
                }
            }
        }

        match (before, after) {
            (Some(b), Some(a)) => Some((b, a)),
            _ => None,
        }
    }

    /// Interpolate between the two snapshots closest to the given tick.
    pub fn interpolate_at_tick(&self, tick: u64) -> Option<WorldSnapshot> {
        let (before, after) = self.get_interpolation_pair(tick)?;

        if before.tick == after.tick {
            return Some(before.clone());
        }

        let range = (after.tick - before.tick) as f32;
        let alpha = (tick - before.tick) as f32 / range;

        Some(WorldSnapshot::interpolate(before, after, alpha.clamp(0.0, 1.0)))
    }

    /// Interpolate at a fractional tick.
    pub fn interpolate_at_time(&self, server_time: f64) -> Option<WorldSnapshot> {
        let mut before: Option<&WorldSnapshot> = None;
        let mut after: Option<&WorldSnapshot> = None;

        for snapshot in &self.buffer {
            if snapshot.server_time <= server_time {
                if before.is_none() || snapshot.server_time > before.unwrap().server_time {
                    before = Some(snapshot);
                }
            }
            if snapshot.server_time >= server_time {
                if after.is_none() || snapshot.server_time < after.unwrap().server_time {
                    after = Some(snapshot);
                }
            }
        }

        let b = before?;
        let a = after?;

        if (b.server_time - a.server_time).abs() < 1e-9 {
            return Some(b.clone());
        }

        let range = a.server_time - b.server_time;
        let alpha = ((server_time - b.server_time) / range) as f32;

        Some(WorldSnapshot::interpolate(b, a, alpha.clamp(0.0, 1.0)))
    }

    /// Returns the number of stored snapshots.
    pub fn len(&self) -> usize {
        self.count.min(self.buffer.len())
    }

    /// Returns `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear all stored snapshots.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.write_pos = 0;
        self.count = 0;
    }
}

// ---------------------------------------------------------------------------
// SnapshotCompressor
// ---------------------------------------------------------------------------

/// Delta compressor for world snapshots.
///
/// Compresses a snapshot relative to a baseline by encoding only changed
/// fields. Uses fixed-point quantization for floating-point values.
///
/// # Wire format
///
/// ```text
/// [tick: u64] [entity_count: u16]
/// For each entity:
///   [entity_id: u64] [delta_flags: u16]
///   For each set flag:
///     [quantized value: i32] (for floats)
///     [raw value] (for ints/flags)
/// ```
pub struct SnapshotCompressor {
    /// Quantization precision for positions.
    pub position_precision: f32,
    /// Quantization precision for rotations.
    pub rotation_precision: f32,
    /// Quantization precision for velocities.
    pub velocity_precision: f32,
}

impl SnapshotCompressor {
    /// Create a compressor with default precision.
    pub fn new() -> Self {
        Self {
            position_precision: POSITION_QUANTIZE_PRECISION,
            rotation_precision: ROTATION_QUANTIZE_PRECISION,
            velocity_precision: VELOCITY_QUANTIZE_PRECISION,
        }
    }

    /// Set custom position precision.
    pub fn with_position_precision(mut self, p: f32) -> Self {
        self.position_precision = p.max(0.0001);
        self
    }

    /// Set custom rotation precision.
    pub fn with_rotation_precision(mut self, p: f32) -> Self {
        self.rotation_precision = p.max(0.00001);
        self
    }

    /// Quantize a float to a fixed-point i32 with the given precision.
    fn quantize(&self, value: f32, precision: f32) -> i32 {
        (value / precision).round() as i32
    }

    /// Dequantize a fixed-point i32 back to float.
    fn dequantize(&self, value: i32, precision: f32) -> f32 {
        value as f32 * precision
    }

    /// Write an i32 to the buffer in big-endian.
    fn write_i32(buf: &mut Vec<u8>, value: i32) {
        buf.extend_from_slice(&value.to_be_bytes());
    }

    /// Write a u16 to the buffer in big-endian.
    fn write_u16(buf: &mut Vec<u8>, value: u16) {
        buf.extend_from_slice(&value.to_be_bytes());
    }

    /// Write a u32 to the buffer in big-endian.
    fn write_u32(buf: &mut Vec<u8>, value: u32) {
        buf.extend_from_slice(&value.to_be_bytes());
    }

    /// Write a u64 to the buffer in big-endian.
    fn write_u64(buf: &mut Vec<u8>, value: u64) {
        buf.extend_from_slice(&value.to_be_bytes());
    }

    /// Read an i32 from a byte slice at the given offset.
    fn read_i32(data: &[u8], offset: &mut usize) -> Option<i32> {
        if *offset + 4 > data.len() {
            return None;
        }
        let bytes: [u8; 4] = data[*offset..*offset + 4].try_into().ok()?;
        *offset += 4;
        Some(i32::from_be_bytes(bytes))
    }

    /// Read a u16 from a byte slice.
    fn read_u16(data: &[u8], offset: &mut usize) -> Option<u16> {
        if *offset + 2 > data.len() {
            return None;
        }
        let bytes: [u8; 2] = data[*offset..*offset + 2].try_into().ok()?;
        *offset += 2;
        Some(u16::from_be_bytes(bytes))
    }

    /// Read a u32 from a byte slice.
    fn read_u32(data: &[u8], offset: &mut usize) -> Option<u32> {
        if *offset + 4 > data.len() {
            return None;
        }
        let bytes: [u8; 4] = data[*offset..*offset + 4].try_into().ok()?;
        *offset += 4;
        Some(u32::from_be_bytes(bytes))
    }

    /// Read a u64 from a byte slice.
    fn read_u64(data: &[u8], offset: &mut usize) -> Option<u64> {
        if *offset + 8 > data.len() {
            return None;
        }
        let bytes: [u8; 8] = data[*offset..*offset + 8].try_into().ok()?;
        *offset += 8;
        Some(u64::from_be_bytes(bytes))
    }

    /// Compress a snapshot relative to a baseline.
    ///
    /// If `baseline` is `None`, encodes all entities fully (keyframe).
    /// Otherwise, only encodes fields that differ from the baseline.
    ///
    /// Returns the compressed byte buffer.
    pub fn compress(
        &self,
        current: &WorldSnapshot,
        baseline: Option<&WorldSnapshot>,
    ) -> Vec<u8> {
        let mut buf = Vec::with_capacity(1024);

        // Header: tick + entity count.
        Self::write_u64(&mut buf, current.tick);
        Self::write_u16(&mut buf, current.entities.len() as u16);

        for (entity_id, state) in &current.entities {
            Self::write_u64(&mut buf, *entity_id);

            let baseline_state = baseline.and_then(|b| b.entities.get(entity_id));

            let flags = match baseline_state {
                Some(base) => state.diff_mask(base),
                None => FIELD_ALL, // Full encode for new entities.
            };

            Self::write_u16(&mut buf, flags);

            // Encode each changed field.
            if flags & FIELD_POSITION_X != 0 {
                Self::write_i32(&mut buf, self.quantize(state.pos_x, self.position_precision));
            }
            if flags & FIELD_POSITION_Y != 0 {
                Self::write_i32(&mut buf, self.quantize(state.pos_y, self.position_precision));
            }
            if flags & FIELD_POSITION_Z != 0 {
                Self::write_i32(&mut buf, self.quantize(state.pos_z, self.position_precision));
            }
            if flags & FIELD_ROTATION_X != 0 {
                Self::write_i32(&mut buf, self.quantize(state.rot_x, self.rotation_precision));
            }
            if flags & FIELD_ROTATION_Y != 0 {
                Self::write_i32(&mut buf, self.quantize(state.rot_y, self.rotation_precision));
            }
            if flags & FIELD_ROTATION_Z != 0 {
                Self::write_i32(&mut buf, self.quantize(state.rot_z, self.rotation_precision));
            }
            if flags & FIELD_ROTATION_W != 0 {
                Self::write_i32(&mut buf, self.quantize(state.rot_w, self.rotation_precision));
            }
            if flags & FIELD_VELOCITY_X != 0 {
                Self::write_i32(&mut buf, self.quantize(state.vel_x, self.velocity_precision));
            }
            if flags & FIELD_VELOCITY_Y != 0 {
                Self::write_i32(&mut buf, self.quantize(state.vel_y, self.velocity_precision));
            }
            if flags & FIELD_VELOCITY_Z != 0 {
                Self::write_i32(&mut buf, self.quantize(state.vel_z, self.velocity_precision));
            }
            if flags & FIELD_HEALTH != 0 {
                Self::write_i32(&mut buf, self.quantize(state.health, 0.1));
            }
            if flags & FIELD_FLAGS != 0 {
                Self::write_u32(&mut buf, state.flags);
            }
            if flags & FIELD_ANIM_STATE != 0 {
                Self::write_u16(&mut buf, state.anim_state);
            }
            if flags & FIELD_CUSTOM_A != 0 {
                Self::write_i32(&mut buf, self.quantize(state.custom_a, 0.01));
            }
            if flags & FIELD_CUSTOM_B != 0 {
                Self::write_i32(&mut buf, self.quantize(state.custom_b, 0.01));
            }
        }

        buf
    }

    /// Decompress a delta-compressed snapshot against a baseline.
    ///
    /// If `baseline` is `None`, the data must be a keyframe (all fields).
    pub fn decompress(
        &self,
        data: &[u8],
        baseline: Option<&WorldSnapshot>,
    ) -> Option<WorldSnapshot> {
        let mut offset = 0;

        let tick = Self::read_u64(data, &mut offset)?;
        let entity_count = Self::read_u16(data, &mut offset)? as usize;

        let mut snapshot = WorldSnapshot::new(tick, 0.0);

        for _ in 0..entity_count {
            let entity_id = Self::read_u64(data, &mut offset)?;
            let flags = Self::read_u16(data, &mut offset)?;

            // Start from baseline if available, otherwise fresh.
            let mut state = baseline
                .and_then(|b| b.entities.get(&entity_id))
                .cloned()
                .unwrap_or_else(|| EntityState::new(entity_id));

            state.entity_id = entity_id;

            if flags & FIELD_POSITION_X != 0 {
                let q = Self::read_i32(data, &mut offset)?;
                state.pos_x = self.dequantize(q, self.position_precision);
            }
            if flags & FIELD_POSITION_Y != 0 {
                let q = Self::read_i32(data, &mut offset)?;
                state.pos_y = self.dequantize(q, self.position_precision);
            }
            if flags & FIELD_POSITION_Z != 0 {
                let q = Self::read_i32(data, &mut offset)?;
                state.pos_z = self.dequantize(q, self.position_precision);
            }
            if flags & FIELD_ROTATION_X != 0 {
                let q = Self::read_i32(data, &mut offset)?;
                state.rot_x = self.dequantize(q, self.rotation_precision);
            }
            if flags & FIELD_ROTATION_Y != 0 {
                let q = Self::read_i32(data, &mut offset)?;
                state.rot_y = self.dequantize(q, self.rotation_precision);
            }
            if flags & FIELD_ROTATION_Z != 0 {
                let q = Self::read_i32(data, &mut offset)?;
                state.rot_z = self.dequantize(q, self.rotation_precision);
            }
            if flags & FIELD_ROTATION_W != 0 {
                let q = Self::read_i32(data, &mut offset)?;
                state.rot_w = self.dequantize(q, self.rotation_precision);
            }
            if flags & FIELD_VELOCITY_X != 0 {
                let q = Self::read_i32(data, &mut offset)?;
                state.vel_x = self.dequantize(q, self.velocity_precision);
            }
            if flags & FIELD_VELOCITY_Y != 0 {
                let q = Self::read_i32(data, &mut offset)?;
                state.vel_y = self.dequantize(q, self.velocity_precision);
            }
            if flags & FIELD_VELOCITY_Z != 0 {
                let q = Self::read_i32(data, &mut offset)?;
                state.vel_z = self.dequantize(q, self.velocity_precision);
            }
            if flags & FIELD_HEALTH != 0 {
                let q = Self::read_i32(data, &mut offset)?;
                state.health = self.dequantize(q, 0.1);
            }
            if flags & FIELD_FLAGS != 0 {
                state.flags = Self::read_u32(data, &mut offset)?;
            }
            if flags & FIELD_ANIM_STATE != 0 {
                state.anim_state = Self::read_u16(data, &mut offset)?;
            }
            if flags & FIELD_CUSTOM_A != 0 {
                let q = Self::read_i32(data, &mut offset)?;
                state.custom_a = self.dequantize(q, 0.01);
            }
            if flags & FIELD_CUSTOM_B != 0 {
                let q = Self::read_i32(data, &mut offset)?;
                state.custom_b = self.dequantize(q, 0.01);
            }

            snapshot.entities.insert(entity_id, state);
        }

        Some(snapshot)
    }

    /// Compute the compression ratio: compressed_size / full_size.
    pub fn compression_ratio(compressed_bytes: usize, entity_count: usize) -> f32 {
        if entity_count == 0 {
            return 1.0;
        }
        // Full entity is roughly: 8 (id) + 15 fields * 4 bytes = 68 bytes.
        let full_size = entity_count * 68 + 10;
        compressed_bytes as f32 / full_size as f32
    }
}

impl Default for SnapshotCompressor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ServerSnapshot
// ---------------------------------------------------------------------------

/// Server-authoritative snapshot state.
///
/// Wraps a `WorldSnapshot` with server-specific metadata: the last
/// acknowledged sequence per client and the baseline used for delta
/// encoding.
#[derive(Debug)]
pub struct ServerSnapshot {
    /// The authoritative world snapshot.
    pub snapshot: WorldSnapshot,
    /// Per-client: last acknowledged sequence number.
    pub client_acks: HashMap<u64, u32>,
    /// The next sequence number to assign.
    next_sequence: u32,
}

impl ServerSnapshot {
    /// Create a new server snapshot container.
    pub fn new() -> Self {
        Self {
            snapshot: WorldSnapshot::new(0, 0.0),
            client_acks: HashMap::new(),
            next_sequence: 1,
        }
    }

    /// Capture a new snapshot from entity states.
    pub fn capture(
        &mut self,
        tick: u64,
        server_time: f64,
        entities: Vec<EntityState>,
    ) -> &WorldSnapshot {
        let mut snapshot = WorldSnapshot::new(tick, server_time);
        snapshot.sequence = self.next_sequence;
        self.next_sequence += 1;

        for state in entities {
            snapshot.add_entity(state);
        }

        self.snapshot = snapshot;
        &self.snapshot
    }

    /// Record that a client acknowledged a sequence number.
    pub fn acknowledge(&mut self, client_id: u64, sequence: u32) {
        self.client_acks
            .entry(client_id)
            .and_modify(|ack| {
                if sequence > *ack {
                    *ack = sequence;
                }
            })
            .or_insert(sequence);
    }

    /// Get the last acknowledged sequence for a client.
    pub fn last_ack(&self, client_id: u64) -> Option<u32> {
        self.client_acks.get(&client_id).copied()
    }
}

impl Default for ServerSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ClientSnapshot
// ---------------------------------------------------------------------------

/// Client-side predicted snapshot state.
///
/// Stores the client's predicted entity states for reconciliation with
/// the server's authoritative snapshots.
#[derive(Debug)]
pub struct ClientSnapshot {
    /// The client's current predicted state.
    pub predicted: WorldSnapshot,
    /// Buffer of received server snapshots for interpolation.
    pub server_buffer: SnapshotBuffer,
    /// The last server sequence the client has acknowledged.
    pub last_server_sequence: u32,
    /// Client-side tick prediction offset.
    pub tick_offset: i64,
    /// Render delay (in ticks) behind the latest server snapshot.
    pub render_delay: u32,
}

impl ClientSnapshot {
    /// Create a new client snapshot state.
    pub fn new() -> Self {
        Self {
            predicted: WorldSnapshot::new(0, 0.0),
            server_buffer: SnapshotBuffer::new(DEFAULT_BUFFER_CAPACITY),
            last_server_sequence: 0,
            tick_offset: 0,
            render_delay: 2,
        }
    }

    /// Set the render delay (in ticks).
    pub fn with_render_delay(mut self, delay: u32) -> Self {
        self.render_delay = delay;
        self
    }

    /// Receive a server snapshot (after decompression).
    pub fn receive_server_snapshot(&mut self, snapshot: WorldSnapshot) {
        if snapshot.sequence > self.last_server_sequence {
            self.last_server_sequence = snapshot.sequence;
        }
        self.server_buffer.push(snapshot);
    }

    /// Get the interpolated render state at the given render time.
    pub fn get_render_state(&self, render_time: f64) -> Option<WorldSnapshot> {
        self.server_buffer.interpolate_at_time(render_time)
    }

    /// Compare the predicted state with the server state for a specific
    /// entity and return the positional error squared.
    pub fn prediction_error(&self, entity_id: u64, server: &WorldSnapshot) -> f32 {
        let predicted = match self.predicted.get_entity(entity_id) {
            Some(p) => p,
            None => return 0.0,
        };
        let authoritative = match server.get_entity(entity_id) {
            Some(a) => a,
            None => return 0.0,
        };

        let dx = predicted.pos_x - authoritative.pos_x;
        let dy = predicted.pos_y - authoritative.pos_y;
        let dz = predicted.pos_z - authoritative.pos_z;

        dx * dx + dy * dy + dz * dz
    }

    /// Apply the server snapshot, correcting prediction errors.
    pub fn reconcile(&mut self, server_snapshot: &WorldSnapshot) {
        // For each entity in the server snapshot, snap to server state.
        for (entity_id, server_state) in &server_snapshot.entities {
            self.predicted
                .entities
                .insert(*entity_id, server_state.clone());
        }

        // Remove entities not present in server snapshot.
        let server_ids: Vec<u64> = server_snapshot.entity_ids();
        self.predicted
            .entities
            .retain(|id, _| server_ids.contains(id));

        self.predicted.tick = server_snapshot.tick;
    }
}

impl Default for ClientSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SnapshotSystem
// ---------------------------------------------------------------------------

/// Top-level snapshot system that coordinates capture, compression,
/// buffering, and interpolation.
///
/// # Usage
///
/// ## Server side
///
/// ```ignore
/// let mut system = SnapshotSystem::new(SnapshotConfig::default());
///
/// // Each simulation tick:
/// let entities = gather_entity_states(&world);
/// system.server_capture(tick, time, entities);
///
/// // When sending to a client:
/// let baseline_seq = system.server.last_ack(client_id).unwrap_or(0);
/// let baseline = system.buffer.get_by_sequence(baseline_seq);
/// let data = system.compressor.compress(&system.server.snapshot, baseline);
/// send_to_client(client_id, data);
/// ```
///
/// ## Client side
///
/// ```ignore
/// // When receiving snapshot data:
/// let snapshot = system.compressor.decompress(&data, baseline);
/// system.client.receive_server_snapshot(snapshot);
///
/// // Each render frame:
/// let render_time = current_time - render_delay;
/// let state = system.client.get_render_state(render_time);
/// apply_to_rendering(state);
/// ```
pub struct SnapshotSystem {
    /// Configuration.
    pub config: SnapshotConfig,
    /// Server-side snapshot state.
    pub server: ServerSnapshot,
    /// Client-side snapshot state.
    pub client: ClientSnapshot,
    /// Snapshot buffer (server uses for baselines, client for interpolation).
    pub buffer: SnapshotBuffer,
    /// Compressor for delta encoding.
    pub compressor: SnapshotCompressor,
    /// Time accumulator for snapshot rate limiting.
    snapshot_timer: f64,
}

/// Configuration for the snapshot system.
#[derive(Debug, Clone)]
pub struct SnapshotConfig {
    /// Snapshots per second.
    pub snapshot_rate: f32,
    /// Buffer capacity (number of snapshots to retain).
    pub buffer_capacity: usize,
    /// Position quantization precision.
    pub position_precision: f32,
    /// Rotation quantization precision.
    pub rotation_precision: f32,
    /// Velocity quantization precision.
    pub velocity_precision: f32,
    /// Client-side render delay in ticks.
    pub render_delay: u32,
}

impl SnapshotConfig {
    /// Create default configuration.
    pub fn new() -> Self {
        Self {
            snapshot_rate: DEFAULT_SNAPSHOT_RATE,
            buffer_capacity: DEFAULT_BUFFER_CAPACITY,
            position_precision: POSITION_QUANTIZE_PRECISION,
            rotation_precision: ROTATION_QUANTIZE_PRECISION,
            velocity_precision: VELOCITY_QUANTIZE_PRECISION,
            render_delay: 2,
        }
    }

    /// Interval between snapshots in seconds.
    pub fn snapshot_interval(&self) -> f64 {
        1.0 / self.snapshot_rate as f64
    }
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl SnapshotSystem {
    /// Create a new snapshot system.
    pub fn new(config: SnapshotConfig) -> Self {
        let buffer = SnapshotBuffer::new(config.buffer_capacity);
        let compressor = SnapshotCompressor::new()
            .with_position_precision(config.position_precision)
            .with_rotation_precision(config.rotation_precision);
        let client = ClientSnapshot::new().with_render_delay(config.render_delay);

        Self {
            config,
            server: ServerSnapshot::new(),
            client,
            buffer,
            compressor,
            snapshot_timer: 0.0,
        }
    }

    /// Server: capture a snapshot if enough time has elapsed.
    /// Returns `true` if a snapshot was captured.
    pub fn server_tick(
        &mut self,
        dt: f64,
        tick: u64,
        server_time: f64,
        entities: Vec<EntityState>,
    ) -> bool {
        self.snapshot_timer += dt;
        let interval = self.config.snapshot_interval();

        if self.snapshot_timer >= interval {
            self.snapshot_timer -= interval;
            self.server_capture(tick, server_time, entities);
            true
        } else {
            false
        }
    }

    /// Server: force-capture a snapshot immediately.
    pub fn server_capture(
        &mut self,
        tick: u64,
        server_time: f64,
        entities: Vec<EntityState>,
    ) {
        self.server.capture(tick, server_time, entities);
        self.buffer.push(self.server.snapshot.clone());
    }

    /// Server: compress the current snapshot for a specific client using
    /// delta encoding relative to the client's last acknowledged snapshot.
    pub fn compress_for_client(&self, client_id: u64) -> Vec<u8> {
        let baseline_seq = self.server.last_ack(client_id);
        let baseline = baseline_seq.and_then(|seq| self.buffer.get_by_sequence(seq));
        self.compressor.compress(&self.server.snapshot, baseline)
    }

    /// Client: receive and decompress a snapshot from the server.
    pub fn client_receive(
        &mut self,
        data: &[u8],
        baseline: Option<&WorldSnapshot>,
    ) -> Option<WorldSnapshot> {
        let snapshot = self.compressor.decompress(data, baseline)?;
        self.client.receive_server_snapshot(snapshot.clone());
        Some(snapshot)
    }

    /// Client: get the interpolated state for rendering.
    pub fn client_render_state(&self, render_time: f64) -> Option<WorldSnapshot> {
        self.client.get_render_state(render_time)
    }

    /// Reset the snapshot system.
    pub fn reset(&mut self) {
        self.server = ServerSnapshot::new();
        self.client = ClientSnapshot::new().with_render_delay(self.config.render_delay);
        self.buffer.clear();
        self.snapshot_timer = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_state(id: u64, x: f32, y: f32, z: f32) -> EntityState {
        EntityState::new(id).with_position(x, y, z).with_health(100.0)
    }

    #[test]
    fn test_entity_state_diff_mask_no_changes() {
        let a = make_test_state(1, 10.0, 20.0, 30.0);
        let b = a.clone();
        assert_eq!(a.diff_mask(&b), 0);
    }

    #[test]
    fn test_entity_state_diff_mask_position_changed() {
        let a = make_test_state(1, 10.0, 20.0, 30.0);
        let b = make_test_state(1, 10.5, 20.0, 30.0);
        let mask = a.diff_mask(&b);
        assert!(mask & FIELD_POSITION_X != 0);
        assert!(mask & FIELD_POSITION_Y == 0);
        assert!(mask & FIELD_POSITION_Z == 0);
    }

    #[test]
    fn test_entity_state_diff_mask_all_changed() {
        let a = EntityState::new(1);
        let mut b = EntityState::new(1).with_position(99.0, 99.0, 99.0).with_velocity(5.0, 5.0, 5.0);
        b.health = 50.0;
        b.flags = 0xFF;
        b.anim_state = 5;
        b.custom_a = 10.0;
        b.custom_b = 20.0;

        let mask = b.diff_mask(&a);
        assert!(mask & FIELD_POSITION_X != 0);
        assert!(mask & FIELD_HEALTH != 0);
        assert!(mask & FIELD_FLAGS != 0);
    }

    #[test]
    fn test_entity_state_interpolation() {
        let a = make_test_state(1, 0.0, 0.0, 0.0);
        let b = make_test_state(1, 10.0, 20.0, 30.0);

        let mid = EntityState::interpolate(&a, &b, 0.5);
        assert!((mid.pos_x - 5.0).abs() < 0.01);
        assert!((mid.pos_y - 10.0).abs() < 0.01);
        assert!((mid.pos_z - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_world_snapshot_add_get() {
        let mut snapshot = WorldSnapshot::new(1, 0.0);
        snapshot.add_entity(make_test_state(42, 1.0, 2.0, 3.0));

        assert_eq!(snapshot.entity_count(), 1);
        let entity = snapshot.get_entity(42).unwrap();
        assert!((entity.pos_x - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_world_snapshot_interpolation() {
        let mut a = WorldSnapshot::new(10, 0.0);
        a.add_entity(make_test_state(1, 0.0, 0.0, 0.0));

        let mut b = WorldSnapshot::new(20, 1.0);
        b.add_entity(make_test_state(1, 100.0, 0.0, 0.0));

        let mid = WorldSnapshot::interpolate(&a, &b, 0.5);
        let entity = mid.get_entity(1).unwrap();
        assert!((entity.pos_x - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_snapshot_buffer_push_and_get() {
        let mut buffer = SnapshotBuffer::new(4);

        for i in 0..4 {
            let snapshot = WorldSnapshot::new(i, i as f64);
            buffer.push(snapshot);
        }

        assert_eq!(buffer.len(), 4);
        assert!(buffer.get_snapshot_at_tick(0).is_some());
        assert!(buffer.get_snapshot_at_tick(3).is_some());
        assert!(buffer.get_snapshot_at_tick(99).is_none());
    }

    #[test]
    fn test_snapshot_buffer_wraps() {
        let mut buffer = SnapshotBuffer::new(3);

        for i in 0..5 {
            buffer.push(WorldSnapshot::new(i, i as f64));
        }

        assert_eq!(buffer.len(), 3);
        // Oldest snapshots should have been overwritten.
        assert!(buffer.get_snapshot_at_tick(0).is_none());
        assert!(buffer.get_snapshot_at_tick(4).is_some());
    }

    #[test]
    fn test_snapshot_buffer_interpolation_pair() {
        let mut buffer = SnapshotBuffer::new(10);

        let mut s1 = WorldSnapshot::new(10, 1.0);
        s1.add_entity(make_test_state(1, 0.0, 0.0, 0.0));
        buffer.push(s1);

        let mut s2 = WorldSnapshot::new(20, 2.0);
        s2.add_entity(make_test_state(1, 100.0, 0.0, 0.0));
        buffer.push(s2);

        let (before, after) = buffer.get_interpolation_pair(15).unwrap();
        assert_eq!(before.tick, 10);
        assert_eq!(after.tick, 20);
    }

    #[test]
    fn test_snapshot_buffer_interpolate_at_tick() {
        let mut buffer = SnapshotBuffer::new(10);

        let mut s1 = WorldSnapshot::new(10, 1.0);
        s1.add_entity(make_test_state(1, 0.0, 0.0, 0.0));
        buffer.push(s1);

        let mut s2 = WorldSnapshot::new(20, 2.0);
        s2.add_entity(make_test_state(1, 100.0, 0.0, 0.0));
        buffer.push(s2);

        let interp = buffer.interpolate_at_tick(15).unwrap();
        let entity = interp.get_entity(1).unwrap();
        assert!((entity.pos_x - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_compressor_full_snapshot() {
        let compressor = SnapshotCompressor::new();

        let mut snapshot = WorldSnapshot::new(1, 0.0);
        snapshot.add_entity(make_test_state(42, 10.0, 20.0, 30.0));

        let data = compressor.compress(&snapshot, None);
        assert!(!data.is_empty());

        let decompressed = compressor.decompress(&data, None).unwrap();
        assert_eq!(decompressed.tick, 1);
        assert_eq!(decompressed.entity_count(), 1);

        let entity = decompressed.get_entity(42).unwrap();
        // Quantization means values are approximate.
        assert!((entity.pos_x - 10.0).abs() < 0.02);
        assert!((entity.pos_y - 20.0).abs() < 0.02);
        assert!((entity.pos_z - 30.0).abs() < 0.02);
    }

    #[test]
    fn test_compressor_delta_compression() {
        let compressor = SnapshotCompressor::new();

        let mut baseline = WorldSnapshot::new(1, 0.0);
        baseline.add_entity(make_test_state(42, 10.0, 20.0, 30.0));

        let mut current = WorldSnapshot::new(2, 0.05);
        // Only X position changed.
        current.add_entity(make_test_state(42, 15.0, 20.0, 30.0));

        let full_data = compressor.compress(&current, None);
        let delta_data = compressor.compress(&current, Some(&baseline));

        // Delta should be smaller than full (only one field changed).
        assert!(delta_data.len() < full_data.len());

        // Decompress delta and verify.
        let decompressed = compressor.decompress(&delta_data, Some(&baseline)).unwrap();
        let entity = decompressed.get_entity(42).unwrap();
        assert!((entity.pos_x - 15.0).abs() < 0.02);
        assert!((entity.pos_y - 20.0).abs() < 0.02);
        assert!((entity.pos_z - 30.0).abs() < 0.02);
    }

    #[test]
    fn test_compressor_new_entity_in_delta() {
        let compressor = SnapshotCompressor::new();

        let mut baseline = WorldSnapshot::new(1, 0.0);
        baseline.add_entity(make_test_state(1, 0.0, 0.0, 0.0));

        let mut current = WorldSnapshot::new(2, 0.05);
        current.add_entity(make_test_state(1, 0.0, 0.0, 0.0));
        current.add_entity(make_test_state(2, 50.0, 60.0, 70.0));

        let data = compressor.compress(&current, Some(&baseline));
        let decompressed = compressor.decompress(&data, Some(&baseline)).unwrap();

        assert_eq!(decompressed.entity_count(), 2);
        let new_entity = decompressed.get_entity(2).unwrap();
        assert!((new_entity.pos_x - 50.0).abs() < 0.02);
    }

    #[test]
    fn test_compressor_quantization_precision() {
        let compressor = SnapshotCompressor::new();

        let q = compressor.quantize(12.345, 0.01);
        let dq = compressor.dequantize(q, 0.01);
        assert!((dq - 12.35).abs() < 0.02);
    }

    #[test]
    fn test_server_snapshot_capture() {
        let mut server = ServerSnapshot::new();
        let entities = vec![
            make_test_state(1, 10.0, 0.0, 0.0),
            make_test_state(2, 20.0, 0.0, 0.0),
        ];

        let snapshot = server.capture(1, 0.0, entities);
        assert_eq!(snapshot.entity_count(), 2);
        assert_eq!(snapshot.sequence, 1);
    }

    #[test]
    fn test_server_snapshot_ack() {
        let mut server = ServerSnapshot::new();
        server.acknowledge(100, 5);
        assert_eq!(server.last_ack(100), Some(5));

        server.acknowledge(100, 3); // Older ack should not overwrite.
        assert_eq!(server.last_ack(100), Some(5));

        server.acknowledge(100, 10);
        assert_eq!(server.last_ack(100), Some(10));
    }

    #[test]
    fn test_client_snapshot_receive_and_render() {
        let mut client = ClientSnapshot::new();

        let mut s1 = WorldSnapshot::new(10, 1.0);
        s1.add_entity(make_test_state(1, 0.0, 0.0, 0.0));
        s1.sequence = 1;
        client.receive_server_snapshot(s1);

        let mut s2 = WorldSnapshot::new(20, 2.0);
        s2.add_entity(make_test_state(1, 100.0, 0.0, 0.0));
        s2.sequence = 2;
        client.receive_server_snapshot(s2);

        assert_eq!(client.last_server_sequence, 2);

        let render = client.get_render_state(1.5).unwrap();
        let entity = render.get_entity(1).unwrap();
        assert!((entity.pos_x - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_client_prediction_error() {
        let mut client = ClientSnapshot::new();
        client
            .predicted
            .entities
            .insert(1, make_test_state(1, 10.0, 0.0, 0.0));

        let mut server = WorldSnapshot::new(1, 0.0);
        server.add_entity(make_test_state(1, 10.5, 0.0, 0.0));

        let err = client.prediction_error(1, &server);
        assert!((err - 0.25).abs() < 0.01); // 0.5^2 = 0.25
    }

    #[test]
    fn test_client_reconcile() {
        let mut client = ClientSnapshot::new();
        client
            .predicted
            .entities
            .insert(1, make_test_state(1, 10.0, 0.0, 0.0));
        client
            .predicted
            .entities
            .insert(2, make_test_state(2, 20.0, 0.0, 0.0));

        let mut server = WorldSnapshot::new(5, 0.5);
        server.add_entity(make_test_state(1, 11.0, 0.0, 0.0));
        // Entity 2 not in server snapshot (despawned).

        client.reconcile(&server);

        assert_eq!(client.predicted.entity_count(), 1);
        let e = client.predicted.get_entity(1).unwrap();
        assert!((e.pos_x - 11.0).abs() < 0.01);
    }

    #[test]
    fn test_snapshot_system_server_tick() {
        let config = SnapshotConfig {
            snapshot_rate: 20.0,
            ..Default::default()
        };
        let mut system = SnapshotSystem::new(config);

        let entities = vec![make_test_state(1, 5.0, 0.0, 0.0)];

        // First tick: not enough time elapsed.
        let captured = system.server_tick(0.01, 1, 0.01, entities.clone());
        assert!(!captured);

        // After 50ms (>= 1/20 = 50ms): should capture.
        let captured = system.server_tick(0.04, 2, 0.05, entities.clone());
        assert!(captured);
    }

    #[test]
    fn test_snapshot_system_compress_for_client() {
        let mut system = SnapshotSystem::new(SnapshotConfig::default());

        let entities = vec![make_test_state(1, 10.0, 20.0, 30.0)];
        system.server_capture(1, 0.0, entities);

        // No baseline for this client: full snapshot.
        let data = system.compress_for_client(100);
        assert!(!data.is_empty());
    }

    #[test]
    fn test_compression_ratio() {
        let ratio = SnapshotCompressor::compression_ratio(50, 10);
        assert!(ratio < 1.0);
    }
}
