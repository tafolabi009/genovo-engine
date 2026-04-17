// engine/physics/src/joint_system.rs
//
// Joint lifecycle management system for the Genovo engine.
//
// Manages the complete lifecycle of physics joints:
//
// - Create, destroy, enable, and disable joints.
// - Joint iteration for solver traversal.
// - Joint queries by body (find all joints attached to a body).
// - Joint type registry and factory.
// - Joint warm-starting data persistence.
// - Joint breaking detection (delegates to breakable_joints).
// - Joint event notifications (created, destroyed, broken).
// - Debug visualization data generation.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum joints in the system.
const MAX_JOINTS: usize = 16384;

/// Maximum joints per body.
const MAX_JOINTS_PER_BODY: usize = 64;

/// Joint handle generation bits.
const GENERATION_BITS: u32 = 8;
/// Joint handle index bits.
const INDEX_BITS: u32 = 24;
/// Generation mask.
const GENERATION_MASK: u32 = (1 << GENERATION_BITS) - 1;
/// Index mask.
const INDEX_MASK: u32 = (1 << INDEX_BITS) - 1;

// ---------------------------------------------------------------------------
// Joint Handle
// ---------------------------------------------------------------------------

/// Generational handle for a joint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JointHandle {
    /// Packed index and generation.
    pub raw: u32,
}

impl JointHandle {
    /// Invalid/null handle.
    pub const INVALID: Self = Self { raw: u32::MAX };

    /// Create a handle from index and generation.
    pub fn new(index: u32, generation: u8) -> Self {
        Self {
            raw: (index & INDEX_MASK) | ((generation as u32 & GENERATION_MASK) << INDEX_BITS),
        }
    }

    /// Extract the index.
    pub fn index(self) -> u32 {
        self.raw & INDEX_MASK
    }

    /// Extract the generation.
    pub fn generation(self) -> u8 {
        ((self.raw >> INDEX_BITS) & GENERATION_MASK) as u8
    }

    /// Check if this handle is valid (not the null sentinel).
    pub fn is_valid(self) -> bool {
        self.raw != u32::MAX
    }
}

// ---------------------------------------------------------------------------
// Joint Type
// ---------------------------------------------------------------------------

/// Type of physics joint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JointType {
    /// Fixed joint (welds two bodies together).
    Fixed,
    /// Hinge joint (single axis of rotation).
    Hinge,
    /// Ball joint (spherical, three rotational DOFs).
    Ball,
    /// Spring joint (soft constraint with stiffness and damping).
    Spring,
    /// Slider joint (single axis of translation).
    Slider,
    /// Cone-twist joint (constrained rotation with cone limit).
    ConeTwist,
    /// Distance joint (maintains distance between anchors).
    Distance,
    /// Gear joint (couples rotation of two joints).
    Gear,
    /// Pulley joint (connected via virtual rope and pulleys).
    Pulley,
    /// Weld joint (rigid connection).
    Weld,
    /// Motor joint (applies force/torque to reach target).
    Motor,
    /// Custom joint type.
    Custom(u32),
}

// ---------------------------------------------------------------------------
// Joint Configuration
// ---------------------------------------------------------------------------

/// Configuration for creating a joint.
#[derive(Debug, Clone)]
pub struct JointConfig {
    /// Type of joint.
    pub joint_type: JointType,
    /// First body.
    pub body_a: u64,
    /// Second body (0 for world anchor).
    pub body_b: u64,
    /// Anchor point on body A (local space).
    pub anchor_a: [f32; 3],
    /// Anchor point on body B (local space).
    pub anchor_b: [f32; 3],
    /// Primary axis (local to body A).
    pub axis_a: [f32; 3],
    /// Secondary axis (local to body B).
    pub axis_b: [f32; 3],
    /// Whether to allow collision between connected bodies.
    pub collide_connected: bool,
    /// Breaking force threshold (0 = unbreakable).
    pub break_force: f32,
    /// Breaking torque threshold (0 = unbreakable).
    pub break_torque: f32,
    /// Whether this joint starts enabled.
    pub enabled: bool,
    /// Joint-specific parameters.
    pub params: JointParams,
}

impl Default for JointConfig {
    fn default() -> Self {
        Self {
            joint_type: JointType::Fixed,
            body_a: 0,
            body_b: 0,
            anchor_a: [0.0; 3],
            anchor_b: [0.0; 3],
            axis_a: [0.0, 1.0, 0.0],
            axis_b: [0.0, 1.0, 0.0],
            collide_connected: false,
            break_force: 0.0,
            break_torque: 0.0,
            enabled: true,
            params: JointParams::None,
        }
    }
}

/// Type-specific joint parameters.
#[derive(Debug, Clone)]
pub enum JointParams {
    /// No extra parameters.
    None,
    /// Hinge joint limits.
    Hinge {
        lower_limit: f32,
        upper_limit: f32,
        motor_speed: f32,
        max_motor_torque: f32,
        enable_limit: bool,
        enable_motor: bool,
    },
    /// Spring parameters.
    Spring {
        stiffness: f32,
        damping: f32,
        rest_length: f32,
    },
    /// Slider parameters.
    Slider {
        lower_limit: f32,
        upper_limit: f32,
        motor_speed: f32,
        max_motor_force: f32,
        enable_limit: bool,
        enable_motor: bool,
    },
    /// Cone-twist limits.
    ConeTwist {
        swing_limit_1: f32,
        swing_limit_2: f32,
        twist_limit: f32,
        softness: f32,
    },
    /// Distance limits.
    Distance {
        min_distance: f32,
        max_distance: f32,
        stiffness: f32,
        damping: f32,
    },
    /// Motor target.
    Motor {
        target_position: [f32; 3],
        target_rotation: [f32; 4],
        max_force: f32,
        max_torque: f32,
    },
}

// ---------------------------------------------------------------------------
// Joint State
// ---------------------------------------------------------------------------

/// Runtime state of a joint.
#[derive(Debug, Clone)]
pub struct JointState {
    /// Joint handle.
    pub handle: JointHandle,
    /// Joint configuration.
    pub config: JointConfig,
    /// Whether the joint is currently enabled.
    pub enabled: bool,
    /// Whether the joint has been broken.
    pub broken: bool,
    /// Current constraint error (for solver convergence).
    pub constraint_error: f32,
    /// Accumulated impulse (for warm starting).
    pub accumulated_impulse: [f32; 3],
    /// Accumulated angular impulse.
    pub accumulated_angular_impulse: [f32; 3],
    /// Current reaction force.
    pub reaction_force: [f32; 3],
    /// Current reaction torque.
    pub reaction_torque: [f32; 3],
    /// Age of the joint in seconds.
    pub age: f32,
    /// User data (opaque).
    pub user_data: u64,
}

impl JointState {
    /// Create a new joint state from a config.
    pub fn new(handle: JointHandle, config: JointConfig) -> Self {
        let enabled = config.enabled;
        Self {
            handle,
            config,
            enabled,
            broken: false,
            constraint_error: 0.0,
            accumulated_impulse: [0.0; 3],
            accumulated_angular_impulse: [0.0; 3],
            reaction_force: [0.0; 3],
            reaction_torque: [0.0; 3],
            age: 0.0,
            user_data: 0,
        }
    }

    /// Check if the joint should break based on current forces.
    pub fn check_break(&self) -> bool {
        if self.broken {
            return true;
        }

        let force_mag = (self.reaction_force[0] * self.reaction_force[0]
            + self.reaction_force[1] * self.reaction_force[1]
            + self.reaction_force[2] * self.reaction_force[2])
            .sqrt();

        let torque_mag = (self.reaction_torque[0] * self.reaction_torque[0]
            + self.reaction_torque[1] * self.reaction_torque[1]
            + self.reaction_torque[2] * self.reaction_torque[2])
            .sqrt();

        (self.config.break_force > 0.0 && force_mag > self.config.break_force)
            || (self.config.break_torque > 0.0 && torque_mag > self.config.break_torque)
    }

    /// Get the reaction force magnitude.
    pub fn reaction_force_magnitude(&self) -> f32 {
        (self.reaction_force[0] * self.reaction_force[0]
            + self.reaction_force[1] * self.reaction_force[1]
            + self.reaction_force[2] * self.reaction_force[2])
            .sqrt()
    }

    /// Get the reaction torque magnitude.
    pub fn reaction_torque_magnitude(&self) -> f32 {
        (self.reaction_torque[0] * self.reaction_torque[0]
            + self.reaction_torque[1] * self.reaction_torque[1]
            + self.reaction_torque[2] * self.reaction_torque[2])
            .sqrt()
    }
}

// ---------------------------------------------------------------------------
// Joint Events
// ---------------------------------------------------------------------------

/// Events emitted by the joint system.
#[derive(Debug, Clone)]
pub enum JointEvent {
    /// A joint was created.
    Created {
        handle: JointHandle,
        body_a: u64,
        body_b: u64,
        joint_type: JointType,
    },
    /// A joint was destroyed.
    Destroyed {
        handle: JointHandle,
        body_a: u64,
        body_b: u64,
    },
    /// A joint was broken by force/torque.
    Broken {
        handle: JointHandle,
        body_a: u64,
        body_b: u64,
        force: f32,
        torque: f32,
    },
    /// A joint was enabled.
    Enabled { handle: JointHandle },
    /// A joint was disabled.
    Disabled { handle: JointHandle },
}

// ---------------------------------------------------------------------------
// Joint System
// ---------------------------------------------------------------------------

/// Manages the lifecycle of all physics joints.
#[derive(Debug)]
pub struct JointSystem {
    /// All joint states.
    joints: Vec<Option<JointState>>,
    /// Generation counters for handle validation.
    generations: Vec<u8>,
    /// Free list of available slot indices.
    free_list: Vec<u32>,
    /// Mapping from body ID to joint handles attached to it.
    body_joints: HashMap<u64, Vec<JointHandle>>,
    /// Pending events.
    pub events: Vec<JointEvent>,
    /// Total active joints.
    pub active_count: usize,
    /// Statistics.
    pub stats: JointSystemStats,
}

impl JointSystem {
    /// Create a new joint system.
    pub fn new() -> Self {
        let capacity = MAX_JOINTS;
        let mut free_list = Vec::with_capacity(capacity);
        for i in (0..capacity as u32).rev() {
            free_list.push(i);
        }

        Self {
            joints: (0..capacity).map(|_| None).collect(),
            generations: vec![0u8; capacity],
            free_list,
            body_joints: HashMap::new(),
            events: Vec::new(),
            active_count: 0,
            stats: JointSystemStats::default(),
        }
    }

    /// Create a new joint. Returns the joint handle.
    pub fn create_joint(&mut self, config: JointConfig) -> Option<JointHandle> {
        let index = self.free_list.pop()?;
        let generation = self.generations[index as usize];
        let handle = JointHandle::new(index, generation);

        let body_a = config.body_a;
        let body_b = config.body_b;
        let joint_type = config.joint_type;

        let state = JointState::new(handle, config);
        self.joints[index as usize] = Some(state);
        self.active_count += 1;

        // Register with body mapping.
        self.body_joints
            .entry(body_a)
            .or_insert_with(Vec::new)
            .push(handle);
        if body_b != 0 {
            self.body_joints
                .entry(body_b)
                .or_insert_with(Vec::new)
                .push(handle);
        }

        self.events.push(JointEvent::Created {
            handle,
            body_a,
            body_b,
            joint_type,
        });

        self.stats.joints_created += 1;
        Some(handle)
    }

    /// Destroy a joint by handle.
    pub fn destroy_joint(&mut self, handle: JointHandle) -> bool {
        let index = handle.index() as usize;
        if index >= self.joints.len() {
            return false;
        }

        if let Some(state) = self.joints[index].take() {
            if state.handle.generation() != handle.generation() {
                self.joints[index] = Some(state);
                return false;
            }

            // Remove from body mapping.
            Self::remove_from_body_map(&mut self.body_joints, state.config.body_a, handle);
            if state.config.body_b != 0 {
                Self::remove_from_body_map(&mut self.body_joints, state.config.body_b, handle);
            }

            self.events.push(JointEvent::Destroyed {
                handle,
                body_a: state.config.body_a,
                body_b: state.config.body_b,
            });

            // Increment generation and return to free list.
            self.generations[index] = self.generations[index].wrapping_add(1);
            self.free_list.push(index as u32);
            self.active_count -= 1;
            self.stats.joints_destroyed += 1;

            true
        } else {
            false
        }
    }

    /// Enable a joint.
    pub fn enable_joint(&mut self, handle: JointHandle) -> bool {
        if let Some(state) = self.get_mut(handle) {
            state.enabled = true;
            self.events.push(JointEvent::Enabled { handle });
            true
        } else {
            false
        }
    }

    /// Disable a joint.
    pub fn disable_joint(&mut self, handle: JointHandle) -> bool {
        if let Some(state) = self.get_mut(handle) {
            state.enabled = false;
            self.events.push(JointEvent::Disabled { handle });
            true
        } else {
            false
        }
    }

    /// Get a reference to a joint state.
    pub fn get(&self, handle: JointHandle) -> Option<&JointState> {
        let index = handle.index() as usize;
        if index >= self.joints.len() {
            return None;
        }
        self.joints[index].as_ref().filter(|s| s.handle.generation() == handle.generation())
    }

    /// Get a mutable reference to a joint state.
    pub fn get_mut(&mut self, handle: JointHandle) -> Option<&mut JointState> {
        let index = handle.index() as usize;
        if index >= self.joints.len() {
            return None;
        }
        let generation = handle.generation();
        self.joints[index].as_mut().filter(|s| s.handle.generation() == generation)
    }

    /// Find all joints attached to a body.
    pub fn joints_for_body(&self, body_id: u64) -> Vec<JointHandle> {
        self.body_joints
            .get(&body_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Find all joints connecting two specific bodies.
    pub fn joints_between(&self, body_a: u64, body_b: u64) -> Vec<JointHandle> {
        let joints_a = self.joints_for_body(body_a);
        joints_a
            .into_iter()
            .filter(|&h| {
                self.get(h).map_or(false, |s| {
                    (s.config.body_a == body_a && s.config.body_b == body_b)
                        || (s.config.body_a == body_b && s.config.body_b == body_a)
                })
            })
            .collect()
    }

    /// Iterate all active, enabled joints.
    pub fn iter_active(&self) -> impl Iterator<Item = &JointState> {
        self.joints.iter().filter_map(|slot| {
            slot.as_ref().filter(|s| s.enabled && !s.broken)
        })
    }

    /// Iterate all joints (including disabled and broken).
    pub fn iter_all(&self) -> impl Iterator<Item = &JointState> {
        self.joints.iter().filter_map(|slot| slot.as_ref())
    }

    /// Update joints: check for breaking, advance age.
    pub fn update(&mut self, dt: f32) {
        let mut broken_handles = Vec::new();

        for slot in &mut self.joints {
            if let Some(state) = slot {
                state.age += dt;

                if !state.broken && state.check_break() {
                    state.broken = true;
                    state.enabled = false;

                    broken_handles.push((
                        state.handle,
                        state.config.body_a,
                        state.config.body_b,
                        state.reaction_force_magnitude(),
                        state.reaction_torque_magnitude(),
                    ));
                }
            }
        }

        for (handle, body_a, body_b, force, torque) in broken_handles {
            self.events.push(JointEvent::Broken {
                handle,
                body_a,
                body_b,
                force,
                torque,
            });
            self.stats.joints_broken += 1;
        }
    }

    /// Remove all joints attached to a body (e.g., when body is destroyed).
    pub fn remove_body_joints(&mut self, body_id: u64) {
        let handles = self.joints_for_body(body_id);
        for handle in handles {
            self.destroy_joint(handle);
        }
    }

    /// Drain pending events.
    pub fn drain_events(&mut self) -> Vec<JointEvent> {
        std::mem::take(&mut self.events)
    }

    /// Returns the number of active joints.
    pub fn count(&self) -> usize {
        self.active_count
    }

    /// Returns the number of enabled (active, non-broken) joints.
    pub fn enabled_count(&self) -> usize {
        self.joints.iter()
            .filter_map(|s| s.as_ref())
            .filter(|s| s.enabled && !s.broken)
            .count()
    }

    /// Remove a handle from the body-joint mapping.
    fn remove_from_body_map(
        map: &mut HashMap<u64, Vec<JointHandle>>,
        body_id: u64,
        handle: JointHandle,
    ) {
        if let Some(handles) = map.get_mut(&body_id) {
            handles.retain(|&h| h != handle);
            if handles.is_empty() {
                map.remove(&body_id);
            }
        }
    }
}

/// Statistics for the joint system.
#[derive(Debug, Clone, Default)]
pub struct JointSystemStats {
    /// Total joints created over lifetime.
    pub joints_created: u64,
    /// Total joints destroyed over lifetime.
    pub joints_destroyed: u64,
    /// Total joints broken.
    pub joints_broken: u64,
    /// Peak active joint count.
    pub peak_active: usize,
}
