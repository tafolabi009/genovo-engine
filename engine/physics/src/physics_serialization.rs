// engine/physics/src/physics_serialization.rs
//
// Physics state serialization for the Genovo engine.
//
// Provides save/load functionality for the entire physics world state:
//
// - **Full world state** -- Serialize/deserialize all body positions, velocities,
//   and orientations.
// - **Joint states** -- Save/restore joint configurations and motor states.
// - **Constraint states** -- Preserve warm-starting data and constraint impulses.
// - **Deterministic replay** -- Support for frame-by-frame replay with
//   deterministic input sequences.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PHYSICS_SAVE_MAGIC: u32 = 0x50485953; // "PHYS"
const PHYSICS_SAVE_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Serialized types
// ---------------------------------------------------------------------------

/// Serialized rigid body state.
#[derive(Debug, Clone)]
pub struct SerializedBody {
    /// Body identifier.
    pub id: u32,
    /// Body type (0=static, 1=dynamic, 2=kinematic).
    pub body_type: u8,
    /// Position (x, y, z).
    pub position: [f32; 3],
    /// Rotation quaternion (x, y, z, w).
    pub rotation: [f32; 4],
    /// Linear velocity.
    pub linear_velocity: [f32; 3],
    /// Angular velocity.
    pub angular_velocity: [f32; 3],
    /// Mass.
    pub mass: f32,
    /// Inverse mass.
    pub inv_mass: f32,
    /// Inertia tensor (diagonal, local space).
    pub inertia: [f32; 3],
    /// Linear damping.
    pub linear_damping: f32,
    /// Angular damping.
    pub angular_damping: f32,
    /// Gravity scale.
    pub gravity_scale: f32,
    /// Whether the body is sleeping.
    pub sleeping: bool,
    /// Sleep timer.
    pub sleep_timer: f32,
    /// Collision layer.
    pub collision_layer: u32,
    /// Collision mask.
    pub collision_mask: u32,
    /// Whether CCD is enabled.
    pub ccd_enabled: bool,
    /// User data tag.
    pub user_tag: u64,
}

impl SerializedBody {
    /// Create a default serialized body.
    pub fn new(id: u32) -> Self {
        Self {
            id,
            body_type: 1,
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            mass: 1.0,
            inv_mass: 1.0,
            inertia: [1.0; 3],
            linear_damping: 0.0,
            angular_damping: 0.05,
            gravity_scale: 1.0,
            sleeping: false,
            sleep_timer: 0.0,
            collision_layer: 1,
            collision_mask: u32::MAX,
            ccd_enabled: false,
            user_tag: 0,
        }
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(128);
        bytes.extend_from_slice(&self.id.to_le_bytes());
        bytes.push(self.body_type);
        for v in &self.position { bytes.extend_from_slice(&v.to_le_bytes()); }
        for v in &self.rotation { bytes.extend_from_slice(&v.to_le_bytes()); }
        for v in &self.linear_velocity { bytes.extend_from_slice(&v.to_le_bytes()); }
        for v in &self.angular_velocity { bytes.extend_from_slice(&v.to_le_bytes()); }
        bytes.extend_from_slice(&self.mass.to_le_bytes());
        bytes.extend_from_slice(&self.inv_mass.to_le_bytes());
        for v in &self.inertia { bytes.extend_from_slice(&v.to_le_bytes()); }
        bytes.extend_from_slice(&self.linear_damping.to_le_bytes());
        bytes.extend_from_slice(&self.angular_damping.to_le_bytes());
        bytes.extend_from_slice(&self.gravity_scale.to_le_bytes());
        bytes.push(self.sleeping as u8);
        bytes.extend_from_slice(&self.sleep_timer.to_le_bytes());
        bytes.extend_from_slice(&self.collision_layer.to_le_bytes());
        bytes.extend_from_slice(&self.collision_mask.to_le_bytes());
        bytes.push(self.ccd_enabled as u8);
        bytes.extend_from_slice(&self.user_tag.to_le_bytes());
        bytes
    }

    /// Deserialize from bytes. Returns (body, bytes_consumed).
    pub fn from_bytes(data: &[u8]) -> Option<(Self, usize)> {
        if data.len() < 128 {
            return None;
        }
        let mut offset = 0;

        let id = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
        offset += 4;
        let body_type = data[offset];
        offset += 1;

        let mut position = [0.0_f32; 3];
        for p in &mut position {
            *p = f32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
            offset += 4;
        }

        let mut rotation = [0.0_f32; 4];
        for r in &mut rotation {
            *r = f32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
            offset += 4;
        }

        let mut linear_velocity = [0.0_f32; 3];
        for v in &mut linear_velocity {
            *v = f32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
            offset += 4;
        }

        let mut angular_velocity = [0.0_f32; 3];
        for v in &mut angular_velocity {
            *v = f32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
            offset += 4;
        }

        let mass = f32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
        offset += 4;
        let inv_mass = f32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
        offset += 4;

        let mut inertia = [0.0_f32; 3];
        for i in &mut inertia {
            *i = f32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
            offset += 4;
        }

        let linear_damping = f32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
        offset += 4;
        let angular_damping = f32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
        offset += 4;
        let gravity_scale = f32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
        offset += 4;
        let sleeping = data[offset] != 0;
        offset += 1;
        let sleep_timer = f32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
        offset += 4;
        let collision_layer = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
        offset += 4;
        let collision_mask = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
        offset += 4;
        let ccd_enabled = data[offset] != 0;
        offset += 1;
        let user_tag = u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?);
        offset += 8;

        Some((
            Self {
                id,
                body_type,
                position,
                rotation,
                linear_velocity,
                angular_velocity,
                mass,
                inv_mass,
                inertia,
                linear_damping,
                angular_damping,
                gravity_scale,
                sleeping,
                sleep_timer,
                collision_layer,
                collision_mask,
                ccd_enabled,
                user_tag,
            },
            offset,
        ))
    }
}

/// Serialized joint state.
#[derive(Debug, Clone)]
pub struct SerializedJoint {
    /// Joint identifier.
    pub id: u32,
    /// Joint type (0=fixed, 1=hinge, 2=ball, 3=slider, 4=spring, etc.).
    pub joint_type: u8,
    /// Body A identifier.
    pub body_a: u32,
    /// Body B identifier.
    pub body_b: u32,
    /// Anchor point on body A (local space).
    pub anchor_a: [f32; 3],
    /// Anchor point on body B (local space).
    pub anchor_b: [f32; 3],
    /// Joint axis (for hinge/slider joints).
    pub axis: [f32; 3],
    /// Whether the joint is breakable.
    pub breakable: bool,
    /// Break force threshold.
    pub break_force: f32,
    /// Motor enabled.
    pub motor_enabled: bool,
    /// Motor target velocity.
    pub motor_target_velocity: f32,
    /// Motor max force.
    pub motor_max_force: f32,
    /// Angular limits (min, max) in radians.
    pub angular_limits: Option<[f32; 2]>,
    /// Linear limits (min, max).
    pub linear_limits: Option<[f32; 2]>,
    /// Accumulated impulse (warm starting).
    pub accumulated_impulse: [f32; 3],
}

impl SerializedJoint {
    pub fn new(id: u32, joint_type: u8, body_a: u32, body_b: u32) -> Self {
        Self {
            id,
            joint_type,
            body_a,
            body_b,
            anchor_a: [0.0; 3],
            anchor_b: [0.0; 3],
            axis: [0.0, 1.0, 0.0],
            breakable: false,
            break_force: f32::MAX,
            motor_enabled: false,
            motor_target_velocity: 0.0,
            motor_max_force: 0.0,
            angular_limits: None,
            linear_limits: None,
            accumulated_impulse: [0.0; 3],
        }
    }
}

/// Serialized constraint solver state.
#[derive(Debug, Clone)]
pub struct SerializedConstraintState {
    /// Constraint identifier.
    pub id: u32,
    /// Constraint type.
    pub constraint_type: u8,
    /// Related body IDs.
    pub body_ids: Vec<u32>,
    /// Accumulated lambda (warm starting).
    pub accumulated_lambda: Vec<f32>,
    /// Error correction term.
    pub error: f32,
}

// ---------------------------------------------------------------------------
// Replay input frame
// ---------------------------------------------------------------------------

/// A single frame of replay input data.
#[derive(Debug, Clone)]
pub struct ReplayInputFrame {
    /// Frame number.
    pub frame: u64,
    /// Fixed timestep used for this frame.
    pub dt: f32,
    /// External forces applied this frame (body_id -> force).
    pub forces: HashMap<u32, [f32; 3]>,
    /// External torques applied this frame.
    pub torques: HashMap<u32, [f32; 3]>,
    /// Impulses applied this frame.
    pub impulses: Vec<ReplayImpulse>,
    /// Body spawns this frame.
    pub spawned_bodies: Vec<SerializedBody>,
    /// Body removals this frame.
    pub removed_bodies: Vec<u32>,
    /// Joint additions this frame.
    pub added_joints: Vec<SerializedJoint>,
    /// Joint removals this frame.
    pub removed_joints: Vec<u32>,
}

impl ReplayInputFrame {
    pub fn new(frame: u64, dt: f32) -> Self {
        Self {
            frame,
            dt,
            forces: HashMap::new(),
            torques: HashMap::new(),
            impulses: Vec::new(),
            spawned_bodies: Vec::new(),
            removed_bodies: Vec::new(),
            added_joints: Vec::new(),
            removed_joints: Vec::new(),
        }
    }
}

/// An impulse record for replay.
#[derive(Debug, Clone)]
pub struct ReplayImpulse {
    pub body_id: u32,
    pub impulse: [f32; 3],
    pub point: [f32; 3],
}

// ---------------------------------------------------------------------------
// Physics world snapshot
// ---------------------------------------------------------------------------

/// A complete snapshot of the physics world state.
#[derive(Debug, Clone)]
pub struct PhysicsWorldSnapshot {
    /// Magic number for validation.
    pub magic: u32,
    /// Version number.
    pub version: u32,
    /// Frame number when the snapshot was taken.
    pub frame: u64,
    /// Simulation time.
    pub simulation_time: f64,
    /// Fixed timestep.
    pub fixed_dt: f32,
    /// Gravity vector.
    pub gravity: [f32; 3],
    /// All body states.
    pub bodies: Vec<SerializedBody>,
    /// All joint states.
    pub joints: Vec<SerializedJoint>,
    /// Constraint solver states.
    pub constraints: Vec<SerializedConstraintState>,
    /// Number of solver iterations.
    pub solver_iterations: u32,
    /// Whether sub-stepping is enabled.
    pub sub_stepping: bool,
    /// Number of sub-steps.
    pub sub_steps: u32,
}

impl PhysicsWorldSnapshot {
    /// Create a new empty snapshot.
    pub fn new() -> Self {
        Self {
            magic: PHYSICS_SAVE_MAGIC,
            version: PHYSICS_SAVE_VERSION,
            frame: 0,
            simulation_time: 0.0,
            fixed_dt: 1.0 / 60.0,
            gravity: [0.0, -9.81, 0.0],
            bodies: Vec::new(),
            joints: Vec::new(),
            constraints: Vec::new(),
            solver_iterations: 8,
            sub_stepping: false,
            sub_steps: 1,
        }
    }

    /// Validate the snapshot.
    pub fn validate(&self) -> Result<(), String> {
        if self.magic != PHYSICS_SAVE_MAGIC {
            return Err("Invalid magic number".into());
        }
        if self.version > PHYSICS_SAVE_VERSION {
            return Err(format!(
                "Unsupported version {} (max {})",
                self.version, PHYSICS_SAVE_VERSION
            ));
        }
        // Check for duplicate body IDs.
        let mut seen_ids = std::collections::HashSet::new();
        for body in &self.bodies {
            if !seen_ids.insert(body.id) {
                return Err(format!("Duplicate body ID: {}", body.id));
            }
        }
        // Check joint body references.
        for joint in &self.joints {
            if !seen_ids.contains(&joint.body_a) {
                return Err(format!(
                    "Joint {} references missing body {}",
                    joint.id, joint.body_a
                ));
            }
            if !seen_ids.contains(&joint.body_b) {
                return Err(format!(
                    "Joint {} references missing body {}",
                    joint.id, joint.body_b
                ));
            }
        }
        Ok(())
    }

    /// Serialize the snapshot to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(4096);

        // Header.
        bytes.extend_from_slice(&self.magic.to_le_bytes());
        bytes.extend_from_slice(&self.version.to_le_bytes());
        bytes.extend_from_slice(&self.frame.to_le_bytes());
        bytes.extend_from_slice(&self.simulation_time.to_le_bytes());
        bytes.extend_from_slice(&self.fixed_dt.to_le_bytes());
        for g in &self.gravity {
            bytes.extend_from_slice(&g.to_le_bytes());
        }
        bytes.extend_from_slice(&self.solver_iterations.to_le_bytes());
        bytes.push(self.sub_stepping as u8);
        bytes.extend_from_slice(&self.sub_steps.to_le_bytes());

        // Body count + bodies.
        bytes.extend_from_slice(&(self.bodies.len() as u32).to_le_bytes());
        for body in &self.bodies {
            let body_bytes = body.to_bytes();
            bytes.extend_from_slice(&(body_bytes.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&body_bytes);
        }

        // Joint count.
        bytes.extend_from_slice(&(self.joints.len() as u32).to_le_bytes());

        // Constraint count.
        bytes.extend_from_slice(&(self.constraints.len() as u32).to_le_bytes());

        bytes
    }

    /// Get the total number of dynamic bodies.
    pub fn dynamic_body_count(&self) -> usize {
        self.bodies.iter().filter(|b| b.body_type == 1).count()
    }

    /// Get the total number of sleeping bodies.
    pub fn sleeping_body_count(&self) -> usize {
        self.bodies.iter().filter(|b| b.sleeping).count()
    }

    /// Compute a checksum for determinism verification.
    pub fn checksum(&self) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for body in &self.bodies {
            for &v in &body.position {
                hash ^= v.to_bits() as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            for &v in &body.linear_velocity {
                hash ^= v.to_bits() as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        hash
    }
}

impl Default for PhysicsWorldSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Deterministic replay
// ---------------------------------------------------------------------------

/// Replay recording and playback for deterministic physics replay.
pub struct PhysicsReplay {
    /// Initial world state.
    initial_state: Option<PhysicsWorldSnapshot>,
    /// Recorded input frames.
    frames: Vec<ReplayInputFrame>,
    /// Current playback frame index.
    playback_index: usize,
    /// Whether recording is active.
    recording: bool,
    /// Whether playback is active.
    playing: bool,
    /// Playback speed multiplier.
    playback_speed: f32,
    /// Checksums per frame for determinism verification.
    checksums: HashMap<u64, u64>,
}

impl PhysicsReplay {
    /// Create a new replay system.
    pub fn new() -> Self {
        Self {
            initial_state: None,
            frames: Vec::new(),
            playback_index: 0,
            recording: false,
            playing: false,
            playback_speed: 1.0,
            checksums: HashMap::new(),
        }
    }

    /// Start recording from a given world state.
    pub fn start_recording(&mut self, initial_state: PhysicsWorldSnapshot) {
        self.initial_state = Some(initial_state);
        self.frames.clear();
        self.checksums.clear();
        self.recording = true;
        self.playing = false;
    }

    /// Stop recording.
    pub fn stop_recording(&mut self) {
        self.recording = false;
    }

    /// Record a frame of input.
    pub fn record_frame(&mut self, frame: ReplayInputFrame) {
        if self.recording {
            self.frames.push(frame);
        }
    }

    /// Record a checksum for a frame (for determinism verification).
    pub fn record_checksum(&mut self, frame: u64, checksum: u64) {
        if self.recording {
            self.checksums.insert(frame, checksum);
        }
    }

    /// Start playback from the beginning.
    pub fn start_playback(&mut self) -> Option<&PhysicsWorldSnapshot> {
        self.playback_index = 0;
        self.playing = true;
        self.recording = false;
        self.initial_state.as_ref()
    }

    /// Get the next frame of input during playback.
    pub fn next_frame(&mut self) -> Option<&ReplayInputFrame> {
        if !self.playing || self.playback_index >= self.frames.len() {
            self.playing = false;
            return None;
        }
        let frame = &self.frames[self.playback_index];
        self.playback_index += 1;
        Some(frame)
    }

    /// Verify determinism by comparing a checksum.
    pub fn verify_checksum(&self, frame: u64, checksum: u64) -> DeterminismResult {
        match self.checksums.get(&frame) {
            None => DeterminismResult::NoReference,
            Some(&expected) if expected == checksum => DeterminismResult::Match,
            Some(&expected) => DeterminismResult::Mismatch {
                expected,
                actual: checksum,
                frame,
            },
        }
    }

    /// Get the total number of recorded frames.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Whether recording is active.
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Whether playback is active.
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Set playback speed.
    pub fn set_playback_speed(&mut self, speed: f32) {
        self.playback_speed = speed.max(0.0);
    }

    /// Get the current playback progress (0.0..1.0).
    pub fn playback_progress(&self) -> f32 {
        if self.frames.is_empty() {
            return 0.0;
        }
        self.playback_index as f32 / self.frames.len() as f32
    }

    /// Seek to a specific frame index.
    pub fn seek(&mut self, frame_index: usize) {
        self.playback_index = frame_index.min(self.frames.len());
    }

    /// Get the initial state snapshot.
    pub fn initial_state(&self) -> Option<&PhysicsWorldSnapshot> {
        self.initial_state.as_ref()
    }

    /// Estimate the memory usage of the replay data.
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total = 0;
        if let Some(state) = &self.initial_state {
            total += state.bodies.len() * std::mem::size_of::<SerializedBody>();
            total += state.joints.len() * std::mem::size_of::<SerializedJoint>();
        }
        for frame in &self.frames {
            total += std::mem::size_of::<ReplayInputFrame>();
            total += frame.forces.len() * (4 + 12);
            total += frame.impulses.len() * std::mem::size_of::<ReplayImpulse>();
            total += frame.spawned_bodies.len() * std::mem::size_of::<SerializedBody>();
        }
        total += self.checksums.len() * 16;
        total
    }
}

impl Default for PhysicsReplay {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a determinism verification check.
#[derive(Debug, Clone)]
pub enum DeterminismResult {
    /// Checksums match -- deterministic.
    Match,
    /// Checksums do not match -- non-deterministic.
    Mismatch {
        expected: u64,
        actual: u64,
        frame: u64,
    },
    /// No reference checksum available for this frame.
    NoReference,
}

impl fmt::Display for DeterminismResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Match => write!(f, "Deterministic (match)"),
            Self::Mismatch {
                expected,
                actual,
                frame,
            } => write!(
                f,
                "NON-DETERMINISTIC at frame {}: expected {:#x}, got {:#x}",
                frame, expected, actual
            ),
            Self::NoReference => write!(f, "No reference checksum"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_body_serialization() {
        let body = SerializedBody {
            id: 42,
            body_type: 1,
            position: [1.0, 2.0, 3.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.5, -1.0, 0.0],
            angular_velocity: [0.0; 3],
            mass: 10.0,
            inv_mass: 0.1,
            inertia: [1.0, 2.0, 3.0],
            linear_damping: 0.01,
            angular_damping: 0.05,
            gravity_scale: 1.0,
            sleeping: false,
            sleep_timer: 0.0,
            collision_layer: 1,
            collision_mask: u32::MAX,
            ccd_enabled: false,
            user_tag: 12345,
        };

        let bytes = body.to_bytes();
        let (restored, _) = SerializedBody::from_bytes(&bytes).unwrap();

        assert_eq!(restored.id, 42);
        assert_eq!(restored.position, [1.0, 2.0, 3.0]);
        assert_eq!(restored.user_tag, 12345);
    }

    #[test]
    fn test_snapshot_validation() {
        let mut snapshot = PhysicsWorldSnapshot::new();
        assert!(snapshot.validate().is_ok());

        snapshot.bodies.push(SerializedBody::new(0));
        snapshot.bodies.push(SerializedBody::new(0)); // Duplicate!
        assert!(snapshot.validate().is_err());
    }

    #[test]
    fn test_snapshot_checksum() {
        let mut s1 = PhysicsWorldSnapshot::new();
        let mut s2 = PhysicsWorldSnapshot::new();

        let body1 = SerializedBody::new(0);
        s1.bodies.push(body1.clone());
        s2.bodies.push(body1);

        assert_eq!(s1.checksum(), s2.checksum());

        // Modify and verify different checksum.
        s2.bodies[0].position[0] = 1.0;
        assert_ne!(s1.checksum(), s2.checksum());
    }

    #[test]
    fn test_replay_recording() {
        let mut replay = PhysicsReplay::new();
        let state = PhysicsWorldSnapshot::new();
        replay.start_recording(state);

        replay.record_frame(ReplayInputFrame::new(0, 1.0 / 60.0));
        replay.record_frame(ReplayInputFrame::new(1, 1.0 / 60.0));
        replay.stop_recording();

        assert_eq!(replay.frame_count(), 2);

        replay.start_playback();
        assert!(replay.next_frame().is_some());
        assert!(replay.next_frame().is_some());
        assert!(replay.next_frame().is_none());
    }

    #[test]
    fn test_determinism_check() {
        let mut replay = PhysicsReplay::new();
        replay.start_recording(PhysicsWorldSnapshot::new());
        replay.record_checksum(0, 12345);
        replay.stop_recording();

        match replay.verify_checksum(0, 12345) {
            DeterminismResult::Match => {}
            other => panic!("Expected Match, got {:?}", other),
        }

        match replay.verify_checksum(0, 99999) {
            DeterminismResult::Mismatch { .. } => {}
            other => panic!("Expected Mismatch, got {:?}", other),
        }
    }
}
