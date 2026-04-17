//! Joints that break under sufficient force or torque.
//!
//! Provides:
//! - **Break force threshold**: joints snap when linear force exceeds a limit
//! - **Break torque threshold**: joints snap when angular force exceeds a limit
//! - **Post-break callbacks**: notification system for game code when breaks occur
//! - **Partial break**: degrade joint properties (stiffness, damping) before full break
//! - **Chain breaking**: propagate break events through connected joints
//! - **Break effects**: configurable debris, sound, and particle hints on break
//! - **Stress tracking**: accumulate micro-damage over time (fatigue)
//!
//! # Design
//!
//! A [`BreakableJoint`] wraps an underlying joint (by handle) and adds breakage
//! semantics. The [`BreakableJointSystem`] evaluates all joints each frame,
//! computes forces/torques, and triggers breaks when thresholds are exceeded.

use glam::Vec3;
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default break force threshold (Newtons).
pub const DEFAULT_BREAK_FORCE: f32 = 1000.0;
/// Default break torque threshold (Newton-meters).
pub const DEFAULT_BREAK_TORQUE: f32 = 500.0;
/// Maximum number of breakable joints in the system.
pub const MAX_BREAKABLE_JOINTS: usize = 4096;
/// Maximum chain propagation depth per frame.
pub const MAX_CHAIN_DEPTH: usize = 16;
/// Stress accumulation rate per frame (for fatigue).
pub const DEFAULT_FATIGUE_RATE: f32 = 0.001;
/// Minimum stiffness after degradation (fraction of original).
pub const MIN_DEGRADED_STIFFNESS: f32 = 0.05;
/// Small epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-7;
/// Maximum number of break events to process per frame.
pub const MAX_BREAK_EVENTS_PER_FRAME: usize = 64;
/// Default partial break threshold (fraction of full break force).
pub const DEFAULT_PARTIAL_BREAK_RATIO: f32 = 0.7;

// ---------------------------------------------------------------------------
// BreakableJointId
// ---------------------------------------------------------------------------

/// Unique identifier for a breakable joint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BreakableJointId(pub u64);

impl BreakableJointId {
    /// Create a new breakable joint ID.
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

// ---------------------------------------------------------------------------
// JointHandle (placeholder for the underlying physics joint)
// ---------------------------------------------------------------------------

/// Handle to the underlying physics joint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JointHandle(pub u64);

/// Handle to a rigid body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BodyHandle(pub u64);

// ---------------------------------------------------------------------------
// BreakMode
// ---------------------------------------------------------------------------

/// How a joint breaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakMode {
    /// Instant break: joint is removed immediately.
    Instant,
    /// Gradual: joint degrades over time before breaking.
    Gradual,
    /// Delayed: joint breaks after a delay once threshold is first exceeded.
    Delayed,
}

// ---------------------------------------------------------------------------
// BreakCondition
// ---------------------------------------------------------------------------

/// Condition that must be met for a joint to break.
#[derive(Debug, Clone)]
pub enum BreakCondition {
    /// Break when linear force exceeds threshold.
    Force(f32),
    /// Break when torque exceeds threshold.
    Torque(f32),
    /// Break when either force or torque exceeds their respective thresholds.
    ForceOrTorque { force: f32, torque: f32 },
    /// Break when both force and torque exceed thresholds simultaneously.
    ForceAndTorque { force: f32, torque: f32 },
    /// Break based on accumulated stress (fatigue).
    Fatigue { max_stress: f32 },
    /// Break based on impulse (single-frame force spike).
    Impulse(f32),
    /// Never break (unbreakable, but can still degrade).
    Unbreakable,
}

impl BreakCondition {
    /// Check if the break condition is met given current forces.
    pub fn is_met(&self, force: f32, torque: f32, stress: f32, impulse: f32) -> bool {
        match self {
            BreakCondition::Force(threshold) => force >= *threshold,
            BreakCondition::Torque(threshold) => torque >= *threshold,
            BreakCondition::ForceOrTorque { force: f_thresh, torque: t_thresh } => {
                force >= *f_thresh || torque >= *t_thresh
            }
            BreakCondition::ForceAndTorque { force: f_thresh, torque: t_thresh } => {
                force >= *f_thresh && torque >= *t_thresh
            }
            BreakCondition::Fatigue { max_stress } => stress >= *max_stress,
            BreakCondition::Impulse(threshold) => impulse >= *threshold,
            BreakCondition::Unbreakable => false,
        }
    }

    /// Check if partial degradation should occur.
    pub fn should_degrade(&self, force: f32, torque: f32, partial_ratio: f32) -> bool {
        match self {
            BreakCondition::Force(threshold) => force >= *threshold * partial_ratio,
            BreakCondition::Torque(threshold) => torque >= *threshold * partial_ratio,
            BreakCondition::ForceOrTorque { force: f_thresh, torque: t_thresh } => {
                force >= *f_thresh * partial_ratio || torque >= *t_thresh * partial_ratio
            }
            BreakCondition::ForceAndTorque { force: f_thresh, torque: t_thresh } => {
                force >= *f_thresh * partial_ratio || torque >= *t_thresh * partial_ratio
            }
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// BreakEffectHint
// ---------------------------------------------------------------------------

/// Hints for visual/audio effects when a joint breaks.
#[derive(Debug, Clone)]
pub struct BreakEffectHint {
    /// Particle effect to spawn on break.
    pub particle_effect: Option<String>,
    /// Sound effect to play on break.
    pub sound_effect: Option<String>,
    /// Camera shake intensity on break (0..1).
    pub camera_shake: f32,
    /// Whether to spawn debris fragments.
    pub spawn_debris: bool,
    /// Number of debris pieces.
    pub debris_count: u32,
    /// Debris launch velocity multiplier.
    pub debris_velocity: f32,
    /// Whether to apply an impulse to connected bodies on break.
    pub apply_break_impulse: bool,
    /// Break impulse magnitude.
    pub break_impulse: f32,
}

impl Default for BreakEffectHint {
    fn default() -> Self {
        Self {
            particle_effect: None,
            sound_effect: None,
            camera_shake: 0.0,
            spawn_debris: false,
            debris_count: 0,
            debris_velocity: 1.0,
            apply_break_impulse: false,
            break_impulse: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// DegradationState
// ---------------------------------------------------------------------------

/// Current degradation state of a joint.
#[derive(Debug, Clone)]
pub struct DegradationState {
    /// Current stiffness multiplier (1.0 = original, 0.0 = fully degraded).
    pub stiffness_multiplier: f32,
    /// Current damping multiplier.
    pub damping_multiplier: f32,
    /// Accumulated stress (for fatigue calculations).
    pub accumulated_stress: f32,
    /// Number of times the joint has been partially broken.
    pub partial_break_count: u32,
    /// Maximum number of partial breaks before full break.
    pub max_partial_breaks: u32,
    /// Rate at which stiffness degrades per partial break (fraction).
    pub degradation_rate: f32,
    /// Whether the joint is currently in degraded state.
    pub is_degraded: bool,
    /// Time spent in degraded state.
    pub degraded_time: f32,
}

impl Default for DegradationState {
    fn default() -> Self {
        Self {
            stiffness_multiplier: 1.0,
            damping_multiplier: 1.0,
            accumulated_stress: 0.0,
            partial_break_count: 0,
            max_partial_breaks: 3,
            degradation_rate: 0.3,
            is_degraded: false,
            degraded_time: 0.0,
        }
    }
}

impl DegradationState {
    /// Apply a partial break (degrade properties).
    pub fn apply_partial_break(&mut self) {
        self.partial_break_count += 1;
        self.stiffness_multiplier = (self.stiffness_multiplier * (1.0 - self.degradation_rate))
            .max(MIN_DEGRADED_STIFFNESS);
        self.damping_multiplier = (self.damping_multiplier * (1.0 - self.degradation_rate * 0.5))
            .max(MIN_DEGRADED_STIFFNESS);
        self.is_degraded = true;
    }

    /// Check if the joint has reached maximum degradation.
    pub fn is_fully_degraded(&self) -> bool {
        self.partial_break_count >= self.max_partial_breaks
            || self.stiffness_multiplier <= MIN_DEGRADED_STIFFNESS + EPSILON
    }

    /// Accumulate stress.
    pub fn accumulate_stress(&mut self, force_ratio: f32, dt: f32) {
        // Stress accumulates faster as force approaches threshold
        let rate = force_ratio * force_ratio * DEFAULT_FATIGUE_RATE;
        self.accumulated_stress += rate * dt;
    }

    /// Reset degradation state.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ---------------------------------------------------------------------------
// ChainBreakConfig
// ---------------------------------------------------------------------------

/// Configuration for chain-breaking behavior.
#[derive(Debug, Clone)]
pub struct ChainBreakConfig {
    /// Whether this joint can trigger chain breaks.
    pub enabled: bool,
    /// How much force is propagated to connected joints (0..1).
    pub propagation_factor: f32,
    /// Maximum chain depth from this joint.
    pub max_depth: usize,
    /// Delay in seconds before propagated break occurs.
    pub propagation_delay: f32,
    /// Whether to propagate force or use a fixed impulse.
    pub propagate_force: bool,
    /// Fixed impulse to apply to neighbors on break.
    pub fixed_impulse: f32,
    /// Which groups this chain break propagates to.
    pub propagation_groups: HashSet<u32>,
}

impl Default for ChainBreakConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            propagation_factor: 0.5,
            max_depth: MAX_CHAIN_DEPTH,
            propagation_delay: 0.0,
            propagate_force: true,
            fixed_impulse: 100.0,
            propagation_groups: HashSet::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// BreakableJoint
// ---------------------------------------------------------------------------

/// A joint that can break under sufficient force or torque.
#[derive(Debug, Clone)]
pub struct BreakableJoint {
    /// Unique identifier.
    pub id: BreakableJointId,
    /// Handle to the underlying physics joint.
    pub joint_handle: JointHandle,
    /// Bodies connected by this joint.
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    /// Name for debugging.
    pub name: String,
    /// Break condition.
    pub break_condition: BreakCondition,
    /// Break mode.
    pub break_mode: BreakMode,
    /// Partial break ratio (fraction of break threshold that triggers degradation).
    pub partial_break_ratio: f32,
    /// Current degradation state.
    pub degradation: DegradationState,
    /// Chain break configuration.
    pub chain_config: ChainBreakConfig,
    /// Effect hints for when the joint breaks.
    pub break_effects: BreakEffectHint,
    /// Group ID for chain propagation.
    pub group_id: u32,
    /// Whether the joint is currently broken.
    pub is_broken: bool,
    /// Whether the joint is currently active.
    pub active: bool,
    /// Current force on the joint (from last evaluation).
    pub current_force: f32,
    /// Current torque on the joint.
    pub current_torque: f32,
    /// Current impulse (single-frame force spike).
    pub current_impulse: f32,
    /// Delayed break timer (for Delayed mode).
    delayed_break_timer: f32,
    /// Delay duration.
    pub break_delay: f32,
    /// Original stiffness value (for degradation reference).
    pub original_stiffness: f32,
    /// Original damping value.
    pub original_damping: f32,
    /// Position of the joint anchor in world space.
    pub anchor_position: Vec3,
    /// Tag for game code.
    pub tag: Option<String>,
}

impl BreakableJoint {
    /// Create a new breakable joint.
    pub fn new(
        id: BreakableJointId,
        joint_handle: JointHandle,
        body_a: BodyHandle,
        body_b: BodyHandle,
        break_condition: BreakCondition,
    ) -> Self {
        Self {
            id,
            joint_handle,
            body_a,
            body_b,
            name: String::new(),
            break_condition,
            break_mode: BreakMode::Instant,
            partial_break_ratio: DEFAULT_PARTIAL_BREAK_RATIO,
            degradation: DegradationState::default(),
            chain_config: ChainBreakConfig::default(),
            break_effects: BreakEffectHint::default(),
            group_id: 0,
            is_broken: false,
            active: true,
            current_force: 0.0,
            current_torque: 0.0,
            current_impulse: 0.0,
            delayed_break_timer: 0.0,
            break_delay: 0.5,
            original_stiffness: 1.0,
            original_damping: 1.0,
            anchor_position: Vec3::ZERO,
            tag: None,
        }
    }

    /// Builder: set name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Builder: set break mode.
    pub fn with_break_mode(mut self, mode: BreakMode) -> Self {
        self.break_mode = mode;
        self
    }

    /// Builder: set partial break ratio.
    pub fn with_partial_break_ratio(mut self, ratio: f32) -> Self {
        self.partial_break_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Builder: set chain break config.
    pub fn with_chain_config(mut self, config: ChainBreakConfig) -> Self {
        self.chain_config = config;
        self
    }

    /// Builder: set break effects.
    pub fn with_break_effects(mut self, effects: BreakEffectHint) -> Self {
        self.break_effects = effects;
        self
    }

    /// Builder: set group ID.
    pub fn with_group(mut self, group_id: u32) -> Self {
        self.group_id = group_id;
        self
    }

    /// Builder: set tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    /// Builder: set original stiffness and damping.
    pub fn with_original_properties(mut self, stiffness: f32, damping: f32) -> Self {
        self.original_stiffness = stiffness;
        self.original_damping = damping;
        self
    }

    /// Get the current effective stiffness.
    pub fn effective_stiffness(&self) -> f32 {
        self.original_stiffness * self.degradation.stiffness_multiplier
    }

    /// Get the current effective damping.
    pub fn effective_damping(&self) -> f32 {
        self.original_damping * self.degradation.damping_multiplier
    }

    /// Update forces on this joint (called by the system).
    pub fn set_forces(&mut self, force: f32, torque: f32, impulse: f32) {
        self.current_force = force;
        self.current_torque = torque;
        self.current_impulse = impulse;
    }

    /// Evaluate whether this joint should break or degrade.
    /// Returns a `BreakEvaluation` describing what should happen.
    pub fn evaluate(&mut self, dt: f32) -> BreakEvaluation {
        if self.is_broken || !self.active {
            return BreakEvaluation::None;
        }

        // Check for full break condition
        let should_break = self.break_condition.is_met(
            self.current_force,
            self.current_torque,
            self.degradation.accumulated_stress,
            self.current_impulse,
        );

        if should_break {
            match self.break_mode {
                BreakMode::Instant => {
                    return BreakEvaluation::Break;
                }
                BreakMode::Gradual => {
                    // Degrade first, break when fully degraded
                    self.degradation.apply_partial_break();
                    if self.degradation.is_fully_degraded() {
                        return BreakEvaluation::Break;
                    }
                    return BreakEvaluation::Degraded;
                }
                BreakMode::Delayed => {
                    self.delayed_break_timer += dt;
                    if self.delayed_break_timer >= self.break_delay {
                        return BreakEvaluation::Break;
                    }
                    return BreakEvaluation::DelayedBreakPending {
                        progress: self.delayed_break_timer / self.break_delay,
                    };
                }
            }
        } else {
            // Reset delayed break timer if condition is no longer met
            if self.break_mode == BreakMode::Delayed {
                self.delayed_break_timer = (self.delayed_break_timer - dt * 2.0).max(0.0);
            }
        }

        // Check for partial degradation
        if self.break_condition.should_degrade(
            self.current_force,
            self.current_torque,
            self.partial_break_ratio,
        ) {
            // Accumulate stress
            let force_ratio = match &self.break_condition {
                BreakCondition::Force(thresh) => self.current_force / thresh,
                BreakCondition::Torque(thresh) => self.current_torque / thresh,
                BreakCondition::ForceOrTorque { force, torque } => {
                    (self.current_force / force).max(self.current_torque / torque)
                }
                _ => 0.0,
            };
            self.degradation.accumulate_stress(force_ratio, dt);
        }

        BreakEvaluation::None
    }

    /// Force-break this joint.
    pub fn force_break(&mut self) {
        self.is_broken = true;
        self.active = false;
    }

    /// Repair this joint (reset degradation, reactivate).
    pub fn repair(&mut self) {
        self.is_broken = false;
        self.active = true;
        self.degradation.reset();
        self.delayed_break_timer = 0.0;
    }
}

/// Result of evaluating a breakable joint.
#[derive(Debug, Clone)]
pub enum BreakEvaluation {
    /// No change.
    None,
    /// Joint should fully break.
    Break,
    /// Joint was degraded (partial break).
    Degraded,
    /// Delayed break is pending.
    DelayedBreakPending { progress: f32 },
}

// ---------------------------------------------------------------------------
// BreakEvent
// ---------------------------------------------------------------------------

/// Event fired when a joint breaks.
#[derive(Debug, Clone)]
pub struct BreakEvent {
    /// ID of the broken joint.
    pub joint_id: BreakableJointId,
    /// Joint handle that was removed.
    pub joint_handle: JointHandle,
    /// Bodies that were connected.
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    /// Position where the break occurred.
    pub position: Vec3,
    /// Force that caused the break.
    pub breaking_force: f32,
    /// Torque that caused the break.
    pub breaking_torque: f32,
    /// Whether this break was caused by chain propagation.
    pub is_chain_break: bool,
    /// Depth in the chain propagation (0 = original break).
    pub chain_depth: u32,
    /// Effect hints.
    pub effects: BreakEffectHint,
    /// Group ID of the broken joint.
    pub group_id: u32,
    /// Tag of the broken joint.
    pub tag: Option<String>,
}

/// Event fired when a joint degrades.
#[derive(Debug, Clone)]
pub struct DegradeEvent {
    /// ID of the degraded joint.
    pub joint_id: BreakableJointId,
    /// Current stiffness multiplier after degradation.
    pub stiffness_multiplier: f32,
    /// Current damping multiplier after degradation.
    pub damping_multiplier: f32,
    /// Number of partial breaks so far.
    pub partial_break_count: u32,
    /// Accumulated stress.
    pub accumulated_stress: f32,
}

// ---------------------------------------------------------------------------
// PendingChainBreak
// ---------------------------------------------------------------------------

/// A chain break waiting to be processed.
#[derive(Debug, Clone)]
struct PendingChainBreak {
    /// Joint to apply force to.
    joint_id: BreakableJointId,
    /// Force to apply.
    force: f32,
    /// Remaining delay before break.
    delay: f32,
    /// Chain depth.
    depth: u32,
    /// Source joint that caused this chain break.
    source_id: BreakableJointId,
}

// ---------------------------------------------------------------------------
// BreakableJointSystem
// ---------------------------------------------------------------------------

/// System that manages breakable joints, evaluates break conditions, and
/// processes chain-breaking propagation.
pub struct BreakableJointSystem {
    /// All breakable joints.
    joints: HashMap<BreakableJointId, BreakableJoint>,
    /// Adjacency map for chain breaking (joint -> neighbors).
    adjacency: HashMap<BreakableJointId, Vec<BreakableJointId>>,
    /// Pending chain breaks.
    pending_chain_breaks: VecDeque<PendingChainBreak>,
    /// Break events from the last frame.
    break_events: Vec<BreakEvent>,
    /// Degrade events from the last frame.
    degrade_events: Vec<DegradeEvent>,
    /// Next joint ID.
    next_id: u64,
    /// Whether chain breaking is globally enabled.
    pub chain_breaking_enabled: bool,
    /// Global break force multiplier.
    pub break_force_multiplier: f32,
    /// Callback IDs for break events.
    break_callbacks: Vec<BreakCallbackEntry>,
    /// Next callback ID.
    next_callback_id: u64,
}

/// A registered callback for break events.
#[derive(Debug)]
struct BreakCallbackEntry {
    id: u64,
    /// Filter: only fire for joints with this tag.
    tag_filter: Option<String>,
    /// Filter: only fire for joints in this group.
    group_filter: Option<u32>,
    /// Callback function (stored as a trait object placeholder; real implementation
    /// would use function pointers or channels).
    _active: bool,
}

impl BreakableJointSystem {
    /// Create a new breakable joint system.
    pub fn new() -> Self {
        Self {
            joints: HashMap::new(),
            adjacency: HashMap::new(),
            pending_chain_breaks: VecDeque::new(),
            break_events: Vec::new(),
            degrade_events: Vec::new(),
            next_id: 1,
            chain_breaking_enabled: true,
            break_force_multiplier: 1.0,
            break_callbacks: Vec::new(),
            next_callback_id: 1,
        }
    }

    /// Add a breakable joint and return its ID.
    pub fn add_joint(&mut self, mut joint: BreakableJoint) -> BreakableJointId {
        let id = BreakableJointId::new(self.next_id);
        self.next_id += 1;
        joint.id = id;
        self.joints.insert(id, joint);
        id
    }

    /// Remove a breakable joint.
    pub fn remove_joint(&mut self, id: BreakableJointId) -> Option<BreakableJoint> {
        // Remove from adjacency
        self.adjacency.remove(&id);
        for neighbors in self.adjacency.values_mut() {
            neighbors.retain(|n| *n != id);
        }
        self.joints.remove(&id)
    }

    /// Get a joint by ID.
    pub fn joint(&self, id: BreakableJointId) -> Option<&BreakableJoint> {
        self.joints.get(&id)
    }

    /// Get a mutable joint by ID.
    pub fn joint_mut(&mut self, id: BreakableJointId) -> Option<&mut BreakableJoint> {
        self.joints.get_mut(&id)
    }

    /// Connect two breakable joints for chain breaking.
    pub fn connect_chain(&mut self, a: BreakableJointId, b: BreakableJointId) {
        self.adjacency.entry(a).or_default().push(b);
        self.adjacency.entry(b).or_default().push(a);
    }

    /// Disconnect two joints from chain breaking.
    pub fn disconnect_chain(&mut self, a: BreakableJointId, b: BreakableJointId) {
        if let Some(neighbors) = self.adjacency.get_mut(&a) {
            neighbors.retain(|n| *n != b);
        }
        if let Some(neighbors) = self.adjacency.get_mut(&b) {
            neighbors.retain(|n| *n != a);
        }
    }

    /// Register a break callback. Returns a callback ID for later removal.
    pub fn register_break_callback(
        &mut self,
        tag_filter: Option<String>,
        group_filter: Option<u32>,
    ) -> u64 {
        let id = self.next_callback_id;
        self.next_callback_id += 1;
        self.break_callbacks.push(BreakCallbackEntry {
            id,
            tag_filter,
            group_filter,
            _active: true,
        });
        id
    }

    /// Unregister a break callback.
    pub fn unregister_break_callback(&mut self, callback_id: u64) {
        self.break_callbacks.retain(|cb| cb.id != callback_id);
    }

    /// Update forces on a joint (called by physics evaluation).
    pub fn set_joint_forces(
        &mut self,
        id: BreakableJointId,
        force: f32,
        torque: f32,
        impulse: f32,
    ) {
        if let Some(joint) = self.joints.get_mut(&id) {
            joint.set_forces(
                force * self.break_force_multiplier,
                torque * self.break_force_multiplier,
                impulse * self.break_force_multiplier,
            );
        }
    }

    /// Main update: evaluate all joints and process breaks.
    pub fn update(&mut self, dt: f32) {
        self.break_events.clear();
        self.degrade_events.clear();

        // Collect joints that need evaluation
        let joint_ids: Vec<BreakableJointId> = self.joints.keys().copied().collect();
        let mut joints_to_break: Vec<BreakableJointId> = Vec::new();

        for id in &joint_ids {
            if let Some(joint) = self.joints.get_mut(id) {
                match joint.evaluate(dt) {
                    BreakEvaluation::Break => {
                        joints_to_break.push(*id);
                    }
                    BreakEvaluation::Degraded => {
                        self.degrade_events.push(DegradeEvent {
                            joint_id: *id,
                            stiffness_multiplier: joint.degradation.stiffness_multiplier,
                            damping_multiplier: joint.degradation.damping_multiplier,
                            partial_break_count: joint.degradation.partial_break_count,
                            accumulated_stress: joint.degradation.accumulated_stress,
                        });
                    }
                    BreakEvaluation::DelayedBreakPending { .. } => {
                        // Still pending
                    }
                    BreakEvaluation::None => {}
                }
            }
        }

        // Process breaks
        for id in joints_to_break {
            self.break_joint(id, false, 0);
        }

        // Process pending chain breaks
        self.process_chain_breaks(dt);
    }

    /// Break a specific joint.
    fn break_joint(&mut self, id: BreakableJointId, is_chain: bool, chain_depth: u32) {
        let event = {
            let joint = match self.joints.get_mut(&id) {
                Some(j) if !j.is_broken => j,
                _ => return,
            };

            joint.force_break();

            BreakEvent {
                joint_id: id,
                joint_handle: joint.joint_handle,
                body_a: joint.body_a,
                body_b: joint.body_b,
                position: joint.anchor_position,
                breaking_force: joint.current_force,
                breaking_torque: joint.current_torque,
                is_chain_break: is_chain,
                chain_depth,
                effects: joint.break_effects.clone(),
                group_id: joint.group_id,
                tag: joint.tag.clone(),
            }
        };

        // Queue chain breaks if enabled
        if self.chain_breaking_enabled {
            let joint = self.joints.get(&id).unwrap();
            if joint.chain_config.enabled && chain_depth < joint.chain_config.max_depth as u32 {
                let propagation_factor = joint.chain_config.propagation_factor;
                let propagation_delay = joint.chain_config.propagation_delay;
                let propagate_force = joint.chain_config.propagate_force;
                let fixed_impulse = joint.chain_config.fixed_impulse;
                let current_force = joint.current_force;

                if let Some(neighbors) = self.adjacency.get(&id).cloned() {
                    for neighbor_id in neighbors {
                        if let Some(neighbor) = self.joints.get(&neighbor_id) {
                            if neighbor.is_broken {
                                continue;
                            }
                            let force = if propagate_force {
                                current_force * propagation_factor
                            } else {
                                fixed_impulse
                            };

                            self.pending_chain_breaks.push_back(PendingChainBreak {
                                joint_id: neighbor_id,
                                force,
                                delay: propagation_delay,
                                depth: chain_depth + 1,
                                source_id: id,
                            });
                        }
                    }
                }
            }
        }

        self.break_events.push(event);
    }

    /// Process pending chain breaks.
    fn process_chain_breaks(&mut self, dt: f32) {
        let mut events_this_frame = 0;
        let mut ready_breaks = Vec::new();

        // Decrement delays and collect ready breaks
        for pending in self.pending_chain_breaks.iter_mut() {
            pending.delay -= dt;
        }

        while let Some(pending) = self.pending_chain_breaks.front() {
            if pending.delay <= 0.0 && events_this_frame < MAX_BREAK_EVENTS_PER_FRAME {
                let pending = self.pending_chain_breaks.pop_front().unwrap();
                ready_breaks.push(pending);
                events_this_frame += 1;
            } else {
                break;
            }
        }

        for pending in ready_breaks {
            // Apply the chain force to the neighbor
            if let Some(joint) = self.joints.get_mut(&pending.joint_id) {
                joint.current_impulse = pending.force;
            }
            self.break_joint(pending.joint_id, true, pending.depth);
        }
    }

    /// Get break events from the last frame.
    pub fn break_events(&self) -> &[BreakEvent] {
        &self.break_events
    }

    /// Get degrade events from the last frame.
    pub fn degrade_events(&self) -> &[DegradeEvent] {
        &self.degrade_events
    }

    /// Get all joint IDs.
    pub fn joint_ids(&self) -> Vec<BreakableJointId> {
        self.joints.keys().copied().collect()
    }

    /// Get the number of active (non-broken) joints.
    pub fn active_joint_count(&self) -> usize {
        self.joints.values().filter(|j| !j.is_broken && j.active).count()
    }

    /// Get the number of broken joints.
    pub fn broken_joint_count(&self) -> usize {
        self.joints.values().filter(|j| j.is_broken).count()
    }

    /// Get all joints in a group.
    pub fn joints_in_group(&self, group_id: u32) -> Vec<BreakableJointId> {
        self.joints.values()
            .filter(|j| j.group_id == group_id)
            .map(|j| j.id)
            .collect()
    }

    /// Break all joints in a group.
    pub fn break_group(&mut self, group_id: u32) {
        let ids: Vec<_> = self.joints_in_group(group_id);
        for id in ids {
            self.break_joint(id, false, 0);
        }
    }

    /// Repair all joints in a group.
    pub fn repair_group(&mut self, group_id: u32) {
        let ids: Vec<_> = self.joints_in_group(group_id);
        for id in ids {
            if let Some(joint) = self.joints.get_mut(&id) {
                joint.repair();
            }
        }
    }

    /// Get system statistics.
    pub fn stats(&self) -> BreakableJointStats {
        let total = self.joints.len();
        let active = self.active_joint_count();
        let broken = self.broken_joint_count();
        let degraded = self.joints.values().filter(|j| j.degradation.is_degraded).count();
        BreakableJointStats {
            total_joints: total,
            active_joints: active,
            broken_joints: broken,
            degraded_joints: degraded,
            pending_chain_breaks: self.pending_chain_breaks.len(),
            break_events_last_frame: self.break_events.len(),
            degrade_events_last_frame: self.degrade_events.len(),
        }
    }

    /// Clear all joints and reset state.
    pub fn clear(&mut self) {
        self.joints.clear();
        self.adjacency.clear();
        self.pending_chain_breaks.clear();
        self.break_events.clear();
        self.degrade_events.clear();
    }
}

/// Statistics for the breakable joint system.
#[derive(Debug, Clone)]
pub struct BreakableJointStats {
    /// Total number of joints.
    pub total_joints: usize,
    /// Number of active (non-broken) joints.
    pub active_joints: usize,
    /// Number of broken joints.
    pub broken_joints: usize,
    /// Number of degraded joints.
    pub degraded_joints: usize,
    /// Number of pending chain breaks.
    pub pending_chain_breaks: usize,
    /// Number of break events generated last frame.
    pub break_events_last_frame: usize,
    /// Number of degrade events generated last frame.
    pub degrade_events_last_frame: usize,
}

// ---------------------------------------------------------------------------
// ECS Component
// ---------------------------------------------------------------------------

/// ECS component that marks an entity as having breakable joints.
#[derive(Debug, Clone)]
pub struct BreakableJointComponent {
    /// IDs of breakable joints owned by this entity.
    pub joint_ids: Vec<BreakableJointId>,
    /// Whether to auto-remove the entity when all joints are broken.
    pub remove_on_all_broken: bool,
    /// Whether any joints are currently broken.
    pub has_broken_joints: bool,
    /// Total number of active joints.
    pub active_count: usize,
}

impl BreakableJointComponent {
    /// Create a new breakable joint component.
    pub fn new() -> Self {
        Self {
            joint_ids: Vec::new(),
            remove_on_all_broken: false,
            has_broken_joints: false,
            active_count: 0,
        }
    }

    /// Add a joint ID.
    pub fn add_joint(&mut self, id: BreakableJointId) {
        self.joint_ids.push(id);
        self.active_count += 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_break_condition_force() {
        let cond = BreakCondition::Force(100.0);
        assert!(!cond.is_met(50.0, 0.0, 0.0, 0.0));
        assert!(cond.is_met(100.0, 0.0, 0.0, 0.0));
        assert!(cond.is_met(200.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_break_condition_torque() {
        let cond = BreakCondition::Torque(50.0);
        assert!(!cond.is_met(0.0, 25.0, 0.0, 0.0));
        assert!(cond.is_met(0.0, 50.0, 0.0, 0.0));
    }

    #[test]
    fn test_degradation() {
        let mut deg = DegradationState::default();
        assert!(!deg.is_degraded);
        deg.apply_partial_break();
        assert!(deg.is_degraded);
        assert!(deg.stiffness_multiplier < 1.0);
        assert_eq!(deg.partial_break_count, 1);
    }

    #[test]
    fn test_instant_break() {
        let mut joint = BreakableJoint::new(
            BreakableJointId::new(1),
            JointHandle(1),
            BodyHandle(1),
            BodyHandle(2),
            BreakCondition::Force(100.0),
        ).with_break_mode(BreakMode::Instant);

        joint.set_forces(200.0, 0.0, 0.0);
        let result = joint.evaluate(0.016);
        assert!(matches!(result, BreakEvaluation::Break));
    }

    #[test]
    fn test_gradual_break() {
        let mut joint = BreakableJoint::new(
            BreakableJointId::new(1),
            JointHandle(1),
            BodyHandle(1),
            BodyHandle(2),
            BreakCondition::Force(100.0),
        ).with_break_mode(BreakMode::Gradual);

        joint.set_forces(200.0, 0.0, 0.0);

        // First evaluation should degrade, not break
        let result = joint.evaluate(0.016);
        assert!(matches!(result, BreakEvaluation::Degraded));
        assert!(!joint.is_broken);
    }

    #[test]
    fn test_system_basic() {
        let mut system = BreakableJointSystem::new();
        let joint = BreakableJoint::new(
            BreakableJointId::new(0),
            JointHandle(1),
            BodyHandle(1),
            BodyHandle(2),
            BreakCondition::Force(100.0),
        );
        let id = system.add_joint(joint);
        assert_eq!(system.active_joint_count(), 1);

        system.set_joint_forces(id, 200.0, 0.0, 0.0);
        system.update(0.016);

        assert_eq!(system.broken_joint_count(), 1);
        assert_eq!(system.break_events().len(), 1);
    }
}
