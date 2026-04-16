//! Animation blending and state machine.
//!
//! Provides blend trees for combining multiple animation clips (1D/2D blending,
//! additive layers, overrides) and a state machine for managing animation states
//! and transitions.
//!
//! # Blend Trees
//!
//! A blend tree is a directed acyclic graph of [`BlendNode`] instances that
//! produces a final animation pose by recursively evaluating and blending
//! child nodes. Leaf nodes sample animation clips; interior nodes combine
//! child outputs using various blending strategies.
//!
//! # State Machine
//!
//! The [`AnimationStateMachine`] manages discrete animation states (each
//! containing a blend tree) and transitions between them based on parameter
//! conditions. Transitions can be crossfaded with configurable easing curves.

use genovo_core::Transform;

use crate::skeleton::{self, AnimationClip};

// ---------------------------------------------------------------------------
// Blend parameter
// ---------------------------------------------------------------------------

/// A named parameter that drives blend tree evaluation.
///
/// Parameters are set by gameplay code (e.g. movement speed, direction) and
/// used by blend nodes to determine weights.
#[derive(Debug, Clone)]
pub struct BlendParameter {
    /// Parameter name (e.g. "Speed", "Direction").
    pub name: String,
    /// Current value.
    pub value: f32,
    /// Valid range (min, max).
    pub range: (f32, f32),
    /// Default value.
    pub default: f32,
}

impl BlendParameter {
    /// Create a new blend parameter.
    pub fn new(name: impl Into<String>, min: f32, max: f32, default: f32) -> Self {
        Self {
            name: name.into(),
            value: default,
            range: (min, max),
            default,
        }
    }

    /// Set the parameter value, clamping to the valid range.
    pub fn set(&mut self, value: f32) {
        self.value = value.clamp(self.range.0, self.range.1);
    }

    /// Reset to the default value.
    pub fn reset(&mut self) {
        self.value = self.default;
    }

    /// Get the normalized value [0.0, 1.0] within the range.
    pub fn normalized(&self) -> f32 {
        let range = self.range.1 - self.range.0;
        if range.abs() < f32::EPSILON {
            return 0.0;
        }
        (self.value - self.range.0) / range
    }
}

// ---------------------------------------------------------------------------
// Blend parameters collection
// ---------------------------------------------------------------------------

/// Collection of named blend parameters used by blend trees and state machines.
#[derive(Debug, Clone, Default)]
pub struct BlendParams {
    /// All parameters, stored by name.
    pub params: Vec<BlendParameter>,
}

impl BlendParams {
    /// Create an empty parameter set.
    pub fn new() -> Self {
        Self {
            params: Vec::new(),
        }
    }

    /// Add a parameter.
    pub fn add(&mut self, param: BlendParameter) {
        self.params.push(param);
    }

    /// Add a float parameter with the given range.
    pub fn add_float(&mut self, name: impl Into<String>, min: f32, max: f32, default: f32) {
        self.params.push(BlendParameter::new(name, min, max, default));
    }

    /// Add a bool parameter (stored as 0.0/1.0).
    pub fn add_bool(&mut self, name: impl Into<String>, default: bool) {
        self.params.push(BlendParameter::new(
            name,
            0.0,
            1.0,
            if default { 1.0 } else { 0.0 },
        ));
    }

    /// Get a parameter value by name.
    pub fn get(&self, name: &str) -> f32 {
        self.params
            .iter()
            .find(|p| p.name == name)
            .map(|p| p.value)
            .unwrap_or(0.0)
    }

    /// Get a bool parameter by name.
    pub fn get_bool(&self, name: &str) -> bool {
        self.get(name) > 0.5
    }

    /// Set a parameter value by name.
    pub fn set(&mut self, name: &str, value: f32) {
        if let Some(param) = self.params.iter_mut().find(|p| p.name == name) {
            param.set(value);
        }
    }

    /// Set a bool parameter.
    pub fn set_bool(&mut self, name: &str, value: bool) {
        self.set(name, if value { 1.0 } else { 0.0 });
    }

    /// Find a parameter by name (mutable).
    pub fn find_mut(&mut self, name: &str) -> Option<&mut BlendParameter> {
        self.params.iter_mut().find(|p| p.name == name)
    }

    /// Find a parameter by name.
    pub fn find(&self, name: &str) -> Option<&BlendParameter> {
        self.params.iter().find(|p| p.name == name)
    }
}

// ---------------------------------------------------------------------------
// Blend node
// ---------------------------------------------------------------------------

/// A node in a blend tree that produces an animation pose.
///
/// Blend nodes form a tree structure where leaf nodes sample clips and
/// interior nodes blend child outputs together.
#[derive(Debug, Clone)]
pub enum BlendNode {
    /// Leaf node: plays a single animation clip.
    Clip {
        /// Index into the clip collection.
        clip_index: usize,
        /// Playback speed multiplier.
        speed: f32,
    },

    /// 1D blend between multiple clips based on a single parameter.
    ///
    /// Example: Idle -> Walk -> Run blended by "Speed" parameter.
    /// Children are sorted by threshold. The parameter value selects
    /// the two nearest thresholds and linearly interpolates between them.
    Blend1D {
        /// Name of the driving parameter.
        parameter: String,
        /// Child nodes with their threshold values, sorted by threshold.
        children: Vec<(f32, Box<BlendNode>)>,
    },

    /// 2D blend between clips based on two parameters (e.g. direction + speed).
    ///
    /// Uses a 2D blend space with sample points. The two closest points
    /// are found using inverse-distance weighting.
    Blend2D {
        /// Name of the X-axis parameter.
        parameter_x: String,
        /// Name of the Y-axis parameter.
        parameter_y: String,
        /// Child nodes with their (x, y) positions in the blend space.
        children: Vec<(f32, f32, Box<BlendNode>)>,
    },

    /// Additive blend: applies one animation on top of another.
    Additive {
        /// The base pose.
        base: Box<BlendNode>,
        /// The additive layer.
        additive: Box<BlendNode>,
        /// Blend weight for the additive layer [0.0, 1.0].
        weight: f32,
    },

    /// Override: replaces specific bones from the overlay node,
    /// leaving other bones from the base unchanged.
    Override {
        /// The base pose (full body).
        base: Box<BlendNode>,
        /// The override pose (partial body).
        overlay: Box<BlendNode>,
        /// Bone mask: indices of bones affected by the override.
        bone_mask: Vec<usize>,
        /// Blend weight [0.0, 1.0].
        weight: f32,
    },
}

impl BlendNode {
    /// Create a clip node.
    pub fn clip(clip_index: usize) -> Self {
        Self::Clip {
            clip_index,
            speed: 1.0,
        }
    }

    /// Create a clip node with custom speed.
    pub fn clip_with_speed(clip_index: usize, speed: f32) -> Self {
        Self::Clip { clip_index, speed }
    }

    /// Create a 1D blend node.
    pub fn blend_1d(parameter: impl Into<String>, children: Vec<(f32, BlendNode)>) -> Self {
        let mut sorted: Vec<(f32, Box<BlendNode>)> = children
            .into_iter()
            .map(|(t, n)| (t, Box::new(n)))
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Self::Blend1D {
            parameter: parameter.into(),
            children: sorted,
        }
    }

    /// Create a 2D blend node.
    pub fn blend_2d(
        param_x: impl Into<String>,
        param_y: impl Into<String>,
        children: Vec<(f32, f32, BlendNode)>,
    ) -> Self {
        Self::Blend2D {
            parameter_x: param_x.into(),
            parameter_y: param_y.into(),
            children: children
                .into_iter()
                .map(|(x, y, n)| (x, y, Box::new(n)))
                .collect(),
        }
    }

    /// Create an additive blend node.
    pub fn additive(base: BlendNode, additive: BlendNode, weight: f32) -> Self {
        Self::Additive {
            base: Box::new(base),
            additive: Box::new(additive),
            weight,
        }
    }

    /// Create an override blend node.
    pub fn override_blend(
        base: BlendNode,
        overlay: BlendNode,
        bone_mask: Vec<usize>,
        weight: f32,
    ) -> Self {
        Self::Override {
            base: Box::new(base),
            overlay: Box::new(overlay),
            bone_mask,
            weight,
        }
    }

    /// Recursively evaluate this node and produce a pose.
    ///
    /// `clips` is the collection of animation clips. `params` provides
    /// parameter values for blend nodes. `time` is the current playback
    /// time. `bone_count` is the number of bones in the skeleton.
    pub fn evaluate(
        &self,
        clips: &[AnimationClip],
        params: &BlendParams,
        time: f32,
        bone_count: usize,
    ) -> Vec<Transform> {
        match self {
            BlendNode::Clip { clip_index, speed } => {
                if *clip_index < clips.len() {
                    clips[*clip_index].sample_pose(time * speed, bone_count)
                } else {
                    vec![Transform::IDENTITY; bone_count]
                }
            }

            BlendNode::Blend1D {
                parameter,
                children,
            } => {
                Self::evaluate_blend_1d(children, parameter, clips, params, time, bone_count)
            }

            BlendNode::Blend2D {
                parameter_x,
                parameter_y,
                children,
            } => Self::evaluate_blend_2d(
                children,
                parameter_x,
                parameter_y,
                clips,
                params,
                time,
                bone_count,
            ),

            BlendNode::Additive {
                base,
                additive,
                weight,
            } => {
                let base_pose = base.evaluate(clips, params, time, bone_count);
                let additive_pose = additive.evaluate(clips, params, time, bone_count);
                skeleton::additive_blend(&base_pose, &additive_pose, *weight)
            }

            BlendNode::Override {
                base,
                overlay,
                bone_mask,
                weight,
            } => {
                let base_pose = base.evaluate(clips, params, time, bone_count);
                let overlay_pose = overlay.evaluate(clips, params, time, bone_count);
                skeleton::masked_blend(&base_pose, &overlay_pose, bone_mask, *weight)
            }
        }
    }

    /// Evaluate a 1D blend between children based on a parameter value.
    ///
    /// Finds the two children whose thresholds bracket the parameter value
    /// and linearly interpolates between their poses.
    fn evaluate_blend_1d(
        children: &[(f32, Box<BlendNode>)],
        parameter: &str,
        clips: &[AnimationClip],
        params: &BlendParams,
        time: f32,
        bone_count: usize,
    ) -> Vec<Transform> {
        if children.is_empty() {
            return vec![Transform::IDENTITY; bone_count];
        }
        if children.len() == 1 {
            return children[0].1.evaluate(clips, params, time, bone_count);
        }

        let value = params.get(parameter);

        // Clamp to the threshold range.
        let first_threshold = children[0].0;
        let last_threshold = children[children.len() - 1].0;

        if value <= first_threshold {
            return children[0].1.evaluate(clips, params, time, bone_count);
        }
        if value >= last_threshold {
            return children[children.len() - 1]
                .1
                .evaluate(clips, params, time, bone_count);
        }

        // Find the two bracketing children.
        for i in 0..children.len() - 1 {
            let t0 = children[i].0;
            let t1 = children[i + 1].0;

            if value >= t0 && value <= t1 {
                let range = t1 - t0;
                let t = if range.abs() < f32::EPSILON {
                    0.0
                } else {
                    (value - t0) / range
                };

                let pose_a = children[i].1.evaluate(clips, params, time, bone_count);
                let pose_b = children[i + 1].1.evaluate(clips, params, time, bone_count);
                return skeleton::blend_poses(&pose_a, &pose_b, t);
            }
        }

        children[0].1.evaluate(clips, params, time, bone_count)
    }

    /// Evaluate a 2D blend between children using inverse-distance weighting.
    ///
    /// Each child has an (x, y) position in the blend space. The parameter
    /// values determine the sample point. Children closer to the sample
    /// point get higher weights.
    fn evaluate_blend_2d(
        children: &[(f32, f32, Box<BlendNode>)],
        parameter_x: &str,
        parameter_y: &str,
        clips: &[AnimationClip],
        params: &BlendParams,
        time: f32,
        bone_count: usize,
    ) -> Vec<Transform> {
        if children.is_empty() {
            return vec![Transform::IDENTITY; bone_count];
        }
        if children.len() == 1 {
            return children[0].2.evaluate(clips, params, time, bone_count);
        }

        let x = params.get(parameter_x);
        let y = params.get(parameter_y);

        // Compute inverse-distance weights.
        let mut weights: Vec<f32> = Vec::with_capacity(children.len());
        let mut has_exact_match = false;
        let mut exact_idx = 0;

        for (i, (cx, cy, _)) in children.iter().enumerate() {
            let dx = x - cx;
            let dy = y - cy;
            let dist_sq = dx * dx + dy * dy;

            if dist_sq < f32::EPSILON {
                has_exact_match = true;
                exact_idx = i;
                break;
            }

            // Inverse distance weighting with power p=2.
            weights.push(1.0 / dist_sq);
        }

        if has_exact_match {
            return children[exact_idx]
                .2
                .evaluate(clips, params, time, bone_count);
        }

        // Normalize weights.
        let total: f32 = weights.iter().sum();
        if total < f32::EPSILON {
            return children[0].2.evaluate(clips, params, time, bone_count);
        }
        for w in &mut weights {
            *w /= total;
        }

        // Evaluate all children and blend with computed weights.
        let mut result = vec![Transform::IDENTITY; bone_count];
        let mut first = true;

        for (i, (_, _, node)) in children.iter().enumerate() {
            let w = weights[i];
            if w < 0.001 {
                continue; // Skip negligible contributions.
            }

            let pose = node.evaluate(clips, params, time, bone_count);

            if first {
                // Initialize result with the first significant pose.
                for (r, p) in result.iter_mut().zip(pose.iter()) {
                    r.position = p.position * w;
                    r.rotation = p.rotation;
                    r.scale = p.scale * w;
                }
                first = false;
            } else {
                // Accumulate blended pose.
                for (r, p) in result.iter_mut().zip(pose.iter()) {
                    r.position += p.position * w;
                    r.rotation = r.rotation.slerp(p.rotation, w);
                    r.scale += p.scale * w;
                }
            }
        }

        result
    }

    /// Collect all clip indices referenced by this node and its children.
    pub fn collect_clip_indices(&self) -> Vec<usize> {
        let mut indices = Vec::new();
        self.collect_clip_indices_recursive(&mut indices);
        indices.sort();
        indices.dedup();
        indices
    }

    fn collect_clip_indices_recursive(&self, out: &mut Vec<usize>) {
        match self {
            BlendNode::Clip { clip_index, .. } => {
                out.push(*clip_index);
            }
            BlendNode::Blend1D { children, .. } => {
                for (_, child) in children {
                    child.collect_clip_indices_recursive(out);
                }
            }
            BlendNode::Blend2D { children, .. } => {
                for (_, _, child) in children {
                    child.collect_clip_indices_recursive(out);
                }
            }
            BlendNode::Additive { base, additive, .. } => {
                base.collect_clip_indices_recursive(out);
                additive.collect_clip_indices_recursive(out);
            }
            BlendNode::Override { base, overlay, .. } => {
                base.collect_clip_indices_recursive(out);
                overlay.collect_clip_indices_recursive(out);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Blend tree output
// ---------------------------------------------------------------------------

/// Output of blend tree evaluation: a clip and its computed weight.
#[derive(Debug, Clone)]
pub struct BlendTreeOutput {
    /// Index of the clip to sample.
    pub clip_index: usize,
    /// Blend weight for this clip [0.0, 1.0].
    pub weight: f32,
    /// Playback speed for this clip.
    pub speed: f32,
}

// ---------------------------------------------------------------------------
// Blend tree
// ---------------------------------------------------------------------------

/// A complete blend tree that produces a final animation pose from parameters.
///
/// The tree consists of a root [`BlendNode`] and a set of [`BlendParameter`]s
/// that drive the blending decisions. Call [`evaluate`](BlendTree::evaluate)
/// to recursively evaluate the tree and produce a pose.
#[derive(Debug, Clone)]
pub struct BlendTree {
    /// The root blend node.
    pub root: BlendNode,
    /// Named parameters that drive the tree.
    pub parameters: Vec<BlendParameter>,
}

impl BlendTree {
    /// Create a new blend tree with a single clip as root.
    pub fn from_clip(clip_index: usize) -> Self {
        Self {
            root: BlendNode::Clip {
                clip_index,
                speed: 1.0,
            },
            parameters: Vec::new(),
        }
    }

    /// Create a blend tree with a custom root node and parameters.
    pub fn new(root: BlendNode, parameters: Vec<BlendParameter>) -> Self {
        Self { root, parameters }
    }

    /// Find a parameter by name. Returns a mutable reference for updating.
    pub fn parameter_mut(&mut self, name: &str) -> Option<&mut BlendParameter> {
        self.parameters.iter_mut().find(|p| p.name == name)
    }

    /// Find a parameter by name (immutable).
    pub fn parameter(&self, name: &str) -> Option<&BlendParameter> {
        self.parameters.iter().find(|p| p.name == name)
    }

    /// Set a parameter value by name.
    pub fn set_parameter(&mut self, name: &str, value: f32) {
        if let Some(param) = self.parameter_mut(name) {
            param.set(value);
        }
    }

    /// Build a BlendParams from this tree's parameters.
    pub fn build_params(&self) -> BlendParams {
        BlendParams {
            params: self.parameters.clone(),
        }
    }

    /// Evaluate the blend tree and return the final pose.
    ///
    /// `clips` is the animation clip collection. `time` is the current
    /// playback time. `bone_count` is the number of bones in the skeleton.
    pub fn evaluate(
        &self,
        clips: &[AnimationClip],
        time: f32,
        bone_count: usize,
    ) -> Vec<Transform> {
        let params = self.build_params();
        self.root.evaluate(clips, &params, time, bone_count)
    }

    /// Evaluate with externally-provided parameters.
    pub fn evaluate_with_params(
        &self,
        clips: &[AnimationClip],
        params: &BlendParams,
        time: f32,
        bone_count: usize,
    ) -> Vec<Transform> {
        self.root.evaluate(clips, params, time, bone_count)
    }

    /// Collect all clip indices used by this tree.
    pub fn clip_indices(&self) -> Vec<usize> {
        self.root.collect_clip_indices()
    }
}

// ---------------------------------------------------------------------------
// Animation state machine
// ---------------------------------------------------------------------------

/// A state machine managing animation states and transitions.
///
/// Each state contains a blend tree (or single clip) and transitions define
/// how to move between states based on conditions. The state machine evaluates
/// transition conditions every frame and smoothly crossfades between states.
#[derive(Debug, Clone)]
pub struct AnimationStateMachine {
    /// All states in the machine.
    pub states: Vec<AnimationState>,

    /// All transitions between states.
    pub transitions: Vec<StateTransition>,

    /// Index of the currently active state.
    pub current_state: usize,

    /// Named parameters shared with blend trees and transition conditions.
    pub parameters: Vec<BlendParameter>,

    /// Index of the entry/default state.
    pub entry_state: usize,

    /// Time spent in the current state (seconds).
    pub state_time: f32,

    /// Whether a transition is currently in progress.
    pub transitioning: bool,

    /// Index of the target state during an active transition.
    pub transition_target: Option<usize>,

    /// Elapsed time of the current transition.
    pub transition_elapsed: f32,

    /// Duration of the current transition.
    pub transition_duration: f32,

    /// Curve of the current transition.
    pub transition_curve: TransitionCurve,

    /// Triggers that have been consumed and need to be reset.
    consumed_triggers: Vec<String>,
}

/// A single state in the animation state machine.
#[derive(Debug, Clone)]
pub struct AnimationState {
    /// Human-readable name (e.g. "Idle", "Locomotion", "Jump").
    pub name: String,

    /// Blend tree for this state.
    pub blend_tree: BlendTree,

    /// Playback speed multiplier.
    pub speed: f32,

    /// Whether to loop the animation in this state.
    pub looping: bool,

    /// Optional tag for grouping states (e.g. "grounded", "airborne").
    pub tag: Option<String>,
}

impl AnimationState {
    /// Create a new state with a single clip.
    pub fn from_clip(name: impl Into<String>, clip_index: usize) -> Self {
        Self {
            name: name.into(),
            blend_tree: BlendTree::from_clip(clip_index),
            speed: 1.0,
            looping: true,
            tag: None,
        }
    }

    /// Create a new state with a blend tree.
    pub fn new(name: impl Into<String>, blend_tree: BlendTree) -> Self {
        Self {
            name: name.into(),
            blend_tree,
            speed: 1.0,
            looping: true,
            tag: None,
        }
    }

    /// Set the tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    /// Set whether to loop.
    pub fn with_looping(mut self, looping: bool) -> Self {
        self.looping = looping;
        self
    }

    /// Set the playback speed.
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }
}

/// Defines a transition between two animation states.
#[derive(Debug, Clone)]
pub struct StateTransition {
    /// Index of the source state.
    pub from_state: usize,

    /// Index of the destination state.
    pub to_state: usize,

    /// Crossfade duration in seconds.
    pub duration: f32,

    /// Easing curve for the crossfade.
    pub curve: TransitionCurve,

    /// Conditions that must all be true for the transition to trigger.
    pub conditions: Vec<TransitionCondition>,

    /// Whether this transition can be interrupted by another.
    pub can_be_interrupted: bool,

    /// Whether the transition can fire from any state (ignores `from_state`).
    pub from_any_state: bool,

    /// Optional exit time [0.0, 1.0] as a fraction of the source clip duration.
    /// The transition only triggers after the source has played this far.
    pub exit_time: Option<f32>,
}

impl StateTransition {
    /// Create a simple transition with conditions.
    pub fn new(
        from_state: usize,
        to_state: usize,
        duration: f32,
        conditions: Vec<TransitionCondition>,
    ) -> Self {
        Self {
            from_state,
            to_state,
            duration,
            curve: TransitionCurve::Linear,
            conditions,
            can_be_interrupted: false,
            from_any_state: false,
            exit_time: None,
        }
    }

    /// Create a transition that fires from any state.
    pub fn from_any(
        to_state: usize,
        duration: f32,
        conditions: Vec<TransitionCondition>,
    ) -> Self {
        Self {
            from_state: 0,
            to_state,
            duration,
            curve: TransitionCurve::Linear,
            conditions,
            can_be_interrupted: true,
            from_any_state: true,
            exit_time: None,
        }
    }

    /// Set the easing curve.
    pub fn with_curve(mut self, curve: TransitionCurve) -> Self {
        self.curve = curve;
        self
    }

    /// Set the exit time.
    pub fn with_exit_time(mut self, exit_time: f32) -> Self {
        self.exit_time = Some(exit_time.clamp(0.0, 1.0));
        self
    }

    /// Allow this transition to be interrupted.
    pub fn interruptible(mut self) -> Self {
        self.can_be_interrupted = true;
        self
    }
}

/// Easing curve for crossfade transitions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransitionCurve {
    /// Linear interpolation.
    Linear,
    /// Smooth ease-in.
    EaseIn,
    /// Smooth ease-out.
    EaseOut,
    /// Smooth ease-in-out (smoothstep).
    EaseInOut,
}

impl Default for TransitionCurve {
    fn default() -> Self {
        Self::Linear
    }
}

impl TransitionCurve {
    /// Evaluate the curve at parameter `t` in [0.0, 1.0].
    pub fn evaluate(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseIn => t * t,
            Self::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
            Self::EaseInOut => t * t * (3.0 - 2.0 * t), // smoothstep
        }
    }
}

/// Condition that must be satisfied for a state transition to trigger.
#[derive(Debug, Clone)]
pub enum TransitionCondition {
    /// A float parameter compared against a threshold.
    Float {
        parameter: String,
        comparison: FloatComparison,
        threshold: f32,
    },
    /// A boolean parameter that must be true or false.
    Bool { parameter: String, value: bool },
    /// A trigger parameter that fires once and auto-resets.
    Trigger { parameter: String },
}

impl TransitionCondition {
    /// Create a "float greater than" condition.
    pub fn float_gt(parameter: impl Into<String>, threshold: f32) -> Self {
        Self::Float {
            parameter: parameter.into(),
            comparison: FloatComparison::Greater,
            threshold,
        }
    }

    /// Create a "float less than" condition.
    pub fn float_lt(parameter: impl Into<String>, threshold: f32) -> Self {
        Self::Float {
            parameter: parameter.into(),
            comparison: FloatComparison::Less,
            threshold,
        }
    }

    /// Create a "bool equals" condition.
    pub fn bool_eq(parameter: impl Into<String>, value: bool) -> Self {
        Self::Bool {
            parameter: parameter.into(),
            value,
        }
    }

    /// Create a trigger condition.
    pub fn trigger(parameter: impl Into<String>) -> Self {
        Self::Trigger {
            parameter: parameter.into(),
        }
    }
}

/// Comparison operator for float conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatComparison {
    Greater,
    Less,
    GreaterOrEqual,
    LessOrEqual,
}

// ---------------------------------------------------------------------------
// AnimationStateMachine implementation
// ---------------------------------------------------------------------------

impl AnimationStateMachine {
    /// Create a new state machine with a single default state.
    pub fn new(default_state: AnimationState) -> Self {
        Self {
            states: vec![default_state],
            transitions: Vec::new(),
            current_state: 0,
            parameters: Vec::new(),
            entry_state: 0,
            state_time: 0.0,
            transitioning: false,
            transition_target: None,
            transition_elapsed: 0.0,
            transition_duration: 0.0,
            transition_curve: TransitionCurve::Linear,
            consumed_triggers: Vec::new(),
        }
    }

    /// Add a state and return its index.
    pub fn add_state(&mut self, state: AnimationState) -> usize {
        self.states.push(state);
        self.states.len() - 1
    }

    /// Add a transition between states.
    pub fn add_transition(&mut self, transition: StateTransition) {
        self.transitions.push(transition);
    }

    /// Add a parameter.
    pub fn add_parameter(&mut self, param: BlendParameter) {
        self.parameters.push(param);
    }

    /// Add a float parameter.
    pub fn add_float_parameter(
        &mut self,
        name: impl Into<String>,
        min: f32,
        max: f32,
        default: f32,
    ) {
        self.parameters
            .push(BlendParameter::new(name, min, max, default));
    }

    /// Add a bool parameter.
    pub fn add_bool_parameter(&mut self, name: impl Into<String>, default: bool) {
        self.parameters.push(BlendParameter::new(
            name,
            0.0,
            1.0,
            if default { 1.0 } else { 0.0 },
        ));
    }

    /// Add a trigger parameter.
    pub fn add_trigger_parameter(&mut self, name: impl Into<String>) {
        self.parameters
            .push(BlendParameter::new(name, 0.0, 1.0, 0.0));
    }

    /// Set a float parameter value.
    pub fn set_float(&mut self, name: &str, value: f32) {
        if let Some(param) = self.parameters.iter_mut().find(|p| p.name == name) {
            param.set(value);
        }
    }

    /// Set a bool parameter (stored as 0.0 or 1.0).
    pub fn set_bool(&mut self, name: &str, value: bool) {
        if let Some(param) = self.parameters.iter_mut().find(|p| p.name == name) {
            param.value = if value { 1.0 } else { 0.0 };
        }
    }

    /// Set a trigger parameter (set to 1.0; will be auto-reset after consumption).
    pub fn set_trigger(&mut self, name: &str) {
        if let Some(param) = self.parameters.iter_mut().find(|p| p.name == name) {
            param.value = 1.0;
        }
    }

    /// Get the name of the current state.
    pub fn current_state_name(&self) -> &str {
        &self.states[self.current_state].name
    }

    /// Get the current state index.
    pub fn current_state_index(&self) -> usize {
        self.current_state
    }

    /// Get the transition target state name, if transitioning.
    pub fn transition_target_name(&self) -> Option<&str> {
        self.transition_target
            .map(|idx| self.states[idx].name.as_str())
    }

    /// Get the current transition progress [0.0, 1.0].
    pub fn transition_progress(&self) -> f32 {
        if !self.transitioning || self.transition_duration <= 0.0 {
            return 0.0;
        }
        (self.transition_elapsed / self.transition_duration).clamp(0.0, 1.0)
    }

    /// Force an immediate transition to a state (no crossfade).
    pub fn force_state(&mut self, state_index: usize) {
        if state_index < self.states.len() {
            self.current_state = state_index;
            self.state_time = 0.0;
            self.transitioning = false;
            self.transition_target = None;
        }
    }

    /// Build a BlendParams from this state machine's parameters.
    pub fn build_params(&self) -> BlendParams {
        BlendParams {
            params: self.parameters.clone(),
        }
    }

    /// Update the state machine: check transition conditions, advance transitions,
    /// and update the current state's blend tree.
    ///
    /// This is the main per-frame update method.
    pub fn update(&mut self, dt: f32) {
        self.state_time += dt;

        if self.transitioning {
            // Advance the transition.
            self.transition_elapsed += dt;

            if self.transition_elapsed >= self.transition_duration {
                // Transition complete: switch to target state.
                if let Some(target) = self.transition_target {
                    self.current_state = target;
                    self.state_time = self.transition_elapsed; // carry over time
                }
                self.transitioning = false;
                self.transition_target = None;
                self.transition_elapsed = 0.0;
            } else {
                // Check if the current transition can be interrupted.
                let can_interrupt = self.can_current_transition_be_interrupted();
                if can_interrupt {
                    // Check for higher-priority transitions.
                    if let Some(new_target) = self.find_valid_transition() {
                        if self.transition_target != Some(new_target.to_state) {
                            self.begin_transition(&new_target);
                        }
                    }
                }
            }
        } else {
            // Not transitioning: check for new transitions.
            if let Some(transition) = self.find_valid_transition() {
                self.begin_transition(&transition);
            }
        }

        // Reset consumed triggers.
        for trigger_name in self.consumed_triggers.drain(..) {
            if let Some(param) = self.parameters.iter_mut().find(|p| p.name == trigger_name) {
                param.value = 0.0;
            }
        }
    }

    /// Evaluate the current state's blend tree and produce a pose.
    ///
    /// If transitioning, blends between the source and target state poses.
    pub fn evaluate(
        &self,
        clips: &[AnimationClip],
        bone_count: usize,
    ) -> Vec<Transform> {
        let params = self.build_params();

        let current_time = self.state_time * self.states[self.current_state].speed;
        let current_pose = self.states[self.current_state]
            .blend_tree
            .evaluate_with_params(clips, &params, current_time, bone_count);

        if self.transitioning {
            if let Some(target_idx) = self.transition_target {
                let target_time = self.transition_elapsed * self.states[target_idx].speed;
                let target_pose = self.states[target_idx]
                    .blend_tree
                    .evaluate_with_params(clips, &params, target_time, bone_count);

                let raw_t = if self.transition_duration > 0.0 {
                    (self.transition_elapsed / self.transition_duration).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let t = self.transition_curve.evaluate(raw_t);

                return skeleton::blend_poses(&current_pose, &target_pose, t);
            }
        }

        current_pose
    }

    /// Find the first valid transition from the current state.
    fn find_valid_transition(&self) -> Option<StateTransition> {
        let mut best: Option<&StateTransition> = None;

        for transition in &self.transitions {
            // Check if this transition applies to the current state.
            if !transition.from_any_state && transition.from_state != self.current_state {
                continue;
            }

            // Don't transition to the current state.
            if transition.to_state == self.current_state {
                continue;
            }

            // Check exit time condition.
            if let Some(exit_time) = transition.exit_time {
                let state = &self.states[self.current_state];
                // Get the clip duration from the state's blend tree.
                let clip_indices = state.blend_tree.clip_indices();
                if !clip_indices.is_empty() {
                    // Use the first clip's duration as reference (simplified).
                    // A real implementation would track the actual playback progress.
                    let normalized_time = self.state_time * state.speed;
                    // Assume a 1-second duration if we don't know the actual duration.
                    let _ = normalized_time;
                    // For now, just check if state_time has passed exit_time seconds.
                    if self.state_time < exit_time {
                        continue;
                    }
                }
            }

            // Check all conditions.
            let all_met = transition
                .conditions
                .iter()
                .all(|cond| self.check_condition(cond));

            if all_met {
                best = Some(transition);
                break; // Use first matching transition.
            }
        }

        best.cloned()
    }

    /// Begin a transition to the target state.
    fn begin_transition(&mut self, transition: &StateTransition) {
        self.transitioning = true;
        self.transition_target = Some(transition.to_state);
        self.transition_elapsed = 0.0;
        self.transition_duration = transition.duration.max(0.001);
        self.transition_curve = transition.curve;

        // Mark any trigger conditions for reset.
        for condition in &transition.conditions {
            if let TransitionCondition::Trigger { parameter } = condition {
                self.consumed_triggers.push(parameter.clone());
            }
        }
    }

    /// Check if the currently active transition can be interrupted.
    fn can_current_transition_be_interrupted(&self) -> bool {
        // Find the transition that started the current crossfade.
        if let Some(target) = self.transition_target {
            for transition in &self.transitions {
                if transition.to_state == target
                    && (transition.from_any_state
                        || transition.from_state == self.current_state)
                {
                    return transition.can_be_interrupted;
                }
            }
        }
        false
    }

    /// Check if a single transition condition is satisfied.
    fn check_condition(&self, condition: &TransitionCondition) -> bool {
        match condition {
            TransitionCondition::Float {
                parameter,
                comparison,
                threshold,
            } => {
                let value = self
                    .parameters
                    .iter()
                    .find(|p| p.name == *parameter)
                    .map(|p| p.value)
                    .unwrap_or(0.0);
                match comparison {
                    FloatComparison::Greater => value > *threshold,
                    FloatComparison::Less => value < *threshold,
                    FloatComparison::GreaterOrEqual => value >= *threshold,
                    FloatComparison::LessOrEqual => value <= *threshold,
                }
            }
            TransitionCondition::Bool { parameter, value } => {
                let param_value = self
                    .parameters
                    .iter()
                    .find(|p| p.name == *parameter)
                    .map(|p| p.value)
                    .unwrap_or(0.0);
                let is_true = param_value > 0.5;
                is_true == *value
            }
            TransitionCondition::Trigger { parameter } => self
                .parameters
                .iter()
                .find(|p| p.name == *parameter)
                .map(|p| p.value > 0.5)
                .unwrap_or(false),
        }
    }
}

// ---------------------------------------------------------------------------
// Pose blending utilities (re-exported from skeleton module)
// ---------------------------------------------------------------------------

/// Blend two poses together using per-bone lerp/slerp.
///
/// `t = 0.0` returns `a`, `t = 1.0` returns `b`.
pub fn blend_poses(a: &[Transform], b: &[Transform], t: f32) -> Vec<Transform> {
    skeleton::blend_poses(a, b, t)
}

/// Apply an additive pose on top of a base pose.
pub fn additive_blend(
    base: &[Transform],
    additive: &[Transform],
    weight: f32,
) -> Vec<Transform> {
    skeleton::additive_blend(base, additive, weight)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{AnimationClip, BoneTrack, Keyframe};
    use glam::{Quat, Vec3};

    /// Helper: make a simple clip with the given bone count.
    fn make_clip(name: &str, duration: f32, bone_count: usize) -> AnimationClip {
        let mut clip = AnimationClip::new(name, duration);
        clip.looping = true;
        for i in 0..bone_count {
            let mut track = BoneTrack::new(i);
            track.position_keys = vec![
                Keyframe::new(0.0, Vec3::new(i as f32 * 0.1, 0.0, 0.0)),
                Keyframe::new(duration, Vec3::new(i as f32 * 0.1, 1.0, 0.0)),
            ];
            track.rotation_keys = vec![
                Keyframe::new(0.0, Quat::IDENTITY),
                Keyframe::new(duration, Quat::IDENTITY),
            ];
            track.scale_keys = vec![
                Keyframe::new(0.0, Vec3::ONE),
                Keyframe::new(duration, Vec3::ONE),
            ];
            clip.add_track(track);
        }
        clip
    }

    // -- BlendParameter tests --

    #[test]
    fn test_blend_parameter_creation() {
        let param = BlendParameter::new("Speed", 0.0, 10.0, 5.0);
        assert_eq!(param.name, "Speed");
        assert_eq!(param.value, 5.0);
        assert_eq!(param.range, (0.0, 10.0));
    }

    #[test]
    fn test_blend_parameter_clamping() {
        let mut param = BlendParameter::new("Speed", 0.0, 10.0, 5.0);
        param.set(15.0);
        assert_eq!(param.value, 10.0);
        param.set(-5.0);
        assert_eq!(param.value, 0.0);
    }

    #[test]
    fn test_blend_parameter_normalized() {
        let param = BlendParameter::new("Speed", 0.0, 10.0, 5.0);
        assert!((param.normalized() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_blend_parameter_reset() {
        let mut param = BlendParameter::new("Speed", 0.0, 10.0, 5.0);
        param.set(8.0);
        param.reset();
        assert_eq!(param.value, 5.0);
    }

    // -- BlendParams tests --

    #[test]
    fn test_blend_params() {
        let mut params = BlendParams::new();
        params.add_float("Speed", 0.0, 10.0, 3.0);
        params.add_bool("IsGrounded", true);

        assert!((params.get("Speed") - 3.0).abs() < f32::EPSILON);
        assert!(params.get_bool("IsGrounded"));

        params.set("Speed", 7.0);
        assert!((params.get("Speed") - 7.0).abs() < f32::EPSILON);

        params.set_bool("IsGrounded", false);
        assert!(!params.get_bool("IsGrounded"));
    }

    // -- BlendNode evaluation tests --

    #[test]
    fn test_clip_node_evaluation() {
        let clips = vec![make_clip("Walk", 1.0, 3)];
        let params = BlendParams::new();

        let node = BlendNode::clip(0);
        let pose = node.evaluate(&clips, &params, 0.5, 3);
        assert_eq!(pose.len(), 3);
        // At t=0.5, position.y should be about 0.5
        assert!((pose[0].position.y - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_blend_1d_evaluation() {
        let clips = vec![
            make_clip("Idle", 1.0, 3),
            make_clip("Walk", 1.0, 3),
            make_clip("Run", 1.0, 3),
        ];

        let node = BlendNode::blend_1d(
            "Speed",
            vec![
                (0.0, BlendNode::clip(0)),
                (1.0, BlendNode::clip(1)),
                (2.0, BlendNode::clip(2)),
            ],
        );

        let mut params = BlendParams::new();
        params.add_float("Speed", 0.0, 2.0, 0.5);

        let pose = node.evaluate(&clips, &params, 0.5, 3);
        assert_eq!(pose.len(), 3);
    }

    #[test]
    fn test_blend_1d_at_boundaries() {
        let clips = vec![
            make_clip("A", 1.0, 2),
            make_clip("B", 1.0, 2),
        ];

        let node = BlendNode::blend_1d(
            "Param",
            vec![
                (0.0, BlendNode::clip(0)),
                (1.0, BlendNode::clip(1)),
            ],
        );

        // At param=0, should be fully clip A
        let mut params = BlendParams::new();
        params.add_float("Param", 0.0, 1.0, 0.0);
        let pose_a = node.evaluate(&clips, &params, 0.0, 2);

        // At param=1, should be fully clip B
        params.set("Param", 1.0);
        let pose_b = node.evaluate(&clips, &params, 0.0, 2);

        // clip A bone 0 has position.x = 0.0, clip B bone 0 also has position.x = 0.0
        // But bone 1: clip A has 0.1, clip B also has 0.1 (same formula)
        // Positions should match clip outputs at their respective thresholds.
        assert_eq!(pose_a.len(), 2);
        assert_eq!(pose_b.len(), 2);
    }

    #[test]
    fn test_blend_1d_midpoint() {
        let clips = vec![
            {
                let mut c = AnimationClip::new("A", 1.0);
                c.looping = true;
                let mut t = BoneTrack::new(0);
                t.position_keys = vec![
                    Keyframe::new(0.0, Vec3::ZERO),
                    Keyframe::new(1.0, Vec3::ZERO),
                ];
                t.rotation_keys = vec![Keyframe::new(0.0, Quat::IDENTITY)];
                t.scale_keys = vec![Keyframe::new(0.0, Vec3::ONE)];
                c.add_track(t);
                c
            },
            {
                let mut c = AnimationClip::new("B", 1.0);
                c.looping = true;
                let mut t = BoneTrack::new(0);
                t.position_keys = vec![
                    Keyframe::new(0.0, Vec3::new(10.0, 0.0, 0.0)),
                    Keyframe::new(1.0, Vec3::new(10.0, 0.0, 0.0)),
                ];
                t.rotation_keys = vec![Keyframe::new(0.0, Quat::IDENTITY)];
                t.scale_keys = vec![Keyframe::new(0.0, Vec3::ONE)];
                c.add_track(t);
                c
            },
        ];

        let node = BlendNode::blend_1d(
            "Blend",
            vec![
                (0.0, BlendNode::clip(0)),
                (1.0, BlendNode::clip(1)),
            ],
        );

        let mut params = BlendParams::new();
        params.add_float("Blend", 0.0, 1.0, 0.5);

        let pose = node.evaluate(&clips, &params, 0.0, 1);
        // At midpoint, position.x should be ~5.0
        assert!((pose[0].position.x - 5.0).abs() < 0.1, "Blend 1D midpoint: {}", pose[0].position.x);
    }

    #[test]
    fn test_blend_2d_evaluation() {
        let clips = vec![
            make_clip("Forward", 1.0, 2),
            make_clip("Right", 1.0, 2),
            make_clip("Backward", 1.0, 2),
            make_clip("Left", 1.0, 2),
        ];

        let node = BlendNode::blend_2d(
            "DirX",
            "DirY",
            vec![
                (0.0, 1.0, BlendNode::clip(0)),   // Forward
                (1.0, 0.0, BlendNode::clip(1)),   // Right
                (0.0, -1.0, BlendNode::clip(2)),  // Backward
                (-1.0, 0.0, BlendNode::clip(3)),  // Left
            ],
        );

        let mut params = BlendParams::new();
        params.add_float("DirX", -1.0, 1.0, 0.0);
        params.add_float("DirY", -1.0, 1.0, 1.0); // Forward

        let pose = node.evaluate(&clips, &params, 0.5, 2);
        assert_eq!(pose.len(), 2);
    }

    #[test]
    fn test_additive_node_evaluation() {
        let clips = vec![
            make_clip("Base", 1.0, 2),
            {
                let mut c = AnimationClip::new("Hit", 1.0);
                c.is_additive = true;
                let mut t = BoneTrack::new(0);
                t.position_keys = vec![
                    Keyframe::new(0.0, Vec3::new(0.0, 0.0, 0.5)),
                    Keyframe::new(1.0, Vec3::new(0.0, 0.0, 0.5)),
                ];
                t.rotation_keys = vec![Keyframe::new(0.0, Quat::IDENTITY)];
                t.scale_keys = vec![Keyframe::new(0.0, Vec3::ONE)];
                c.add_track(t);
                let mut t1 = BoneTrack::new(1);
                t1.position_keys = vec![Keyframe::new(0.0, Vec3::ZERO)];
                t1.rotation_keys = vec![Keyframe::new(0.0, Quat::IDENTITY)];
                t1.scale_keys = vec![Keyframe::new(0.0, Vec3::ONE)];
                c.add_track(t1);
                c
            },
        ];

        let node = BlendNode::additive(
            BlendNode::clip(0),
            BlendNode::clip(1),
            1.0,
        );

        let params = BlendParams::new();
        let pose = node.evaluate(&clips, &params, 0.0, 2);
        assert_eq!(pose.len(), 2);
        // Base position + additive offset
        assert!(pose[0].position.z > 0.1, "Additive should add Z offset");
    }

    #[test]
    fn test_override_node_evaluation() {
        let clips = vec![
            {
                let mut c = AnimationClip::new("FullBody", 1.0);
                c.looping = true;
                for i in 0..3 {
                    let mut t = BoneTrack::new(i);
                    t.position_keys = vec![
                        Keyframe::new(0.0, Vec3::new(1.0, 0.0, 0.0)),
                        Keyframe::new(1.0, Vec3::new(1.0, 0.0, 0.0)),
                    ];
                    t.rotation_keys = vec![Keyframe::new(0.0, Quat::IDENTITY)];
                    t.scale_keys = vec![Keyframe::new(0.0, Vec3::ONE)];
                    c.add_track(t);
                }
                c
            },
            {
                let mut c = AnimationClip::new("UpperBody", 1.0);
                c.looping = true;
                for i in 0..3 {
                    let mut t = BoneTrack::new(i);
                    t.position_keys = vec![
                        Keyframe::new(0.0, Vec3::new(5.0, 0.0, 0.0)),
                        Keyframe::new(1.0, Vec3::new(5.0, 0.0, 0.0)),
                    ];
                    t.rotation_keys = vec![Keyframe::new(0.0, Quat::IDENTITY)];
                    t.scale_keys = vec![Keyframe::new(0.0, Vec3::ONE)];
                    c.add_track(t);
                }
                c
            },
        ];

        // Override only bone 1.
        let node = BlendNode::override_blend(
            BlendNode::clip(0),
            BlendNode::clip(1),
            vec![1],
            1.0,
        );

        let params = BlendParams::new();
        let pose = node.evaluate(&clips, &params, 0.0, 3);

        // Bone 0: from base (1.0)
        assert!((pose[0].position.x - 1.0).abs() < 0.01);
        // Bone 1: from overlay (5.0)
        assert!((pose[1].position.x - 5.0).abs() < 0.01);
        // Bone 2: from base (1.0)
        assert!((pose[2].position.x - 1.0).abs() < 0.01);
    }

    // -- BlendTree tests --

    #[test]
    fn test_blend_tree_from_clip() {
        let tree = BlendTree::from_clip(0);
        let indices = tree.clip_indices();
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn test_blend_tree_evaluate() {
        let clips = vec![make_clip("Walk", 1.0, 2)];
        let tree = BlendTree::from_clip(0);
        let pose = tree.evaluate(&clips, 0.5, 2);
        assert_eq!(pose.len(), 2);
    }

    #[test]
    fn test_blend_tree_parameters() {
        let mut tree = BlendTree::new(
            BlendNode::blend_1d(
                "Speed",
                vec![
                    (0.0, BlendNode::clip(0)),
                    (1.0, BlendNode::clip(1)),
                ],
            ),
            vec![BlendParameter::new("Speed", 0.0, 1.0, 0.5)],
        );

        tree.set_parameter("Speed", 0.8);
        assert!((tree.parameter("Speed").unwrap().value - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_blend_tree_collect_indices() {
        let tree = BlendTree::new(
            BlendNode::additive(
                BlendNode::blend_1d(
                    "Speed",
                    vec![
                        (0.0, BlendNode::clip(0)),
                        (1.0, BlendNode::clip(1)),
                    ],
                ),
                BlendNode::clip(2),
                1.0,
            ),
            Vec::new(),
        );
        let indices = tree.clip_indices();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    // -- TransitionCurve tests --

    #[test]
    fn test_transition_curve_linear() {
        let curve = TransitionCurve::Linear;
        assert!((curve.evaluate(0.0)).abs() < f32::EPSILON);
        assert!((curve.evaluate(0.5) - 0.5).abs() < f32::EPSILON);
        assert!((curve.evaluate(1.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_transition_curve_ease_in() {
        let curve = TransitionCurve::EaseIn;
        assert!((curve.evaluate(0.0)).abs() < f32::EPSILON);
        assert!(curve.evaluate(0.5) < 0.5); // Ease in is slower at start
        assert!((curve.evaluate(1.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_transition_curve_ease_out() {
        let curve = TransitionCurve::EaseOut;
        assert!((curve.evaluate(0.0)).abs() < f32::EPSILON);
        assert!(curve.evaluate(0.5) > 0.5); // Ease out is faster at start
        assert!((curve.evaluate(1.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_transition_curve_ease_in_out() {
        let curve = TransitionCurve::EaseInOut;
        assert!((curve.evaluate(0.0)).abs() < f32::EPSILON);
        assert!((curve.evaluate(0.5) - 0.5).abs() < f32::EPSILON); // Smoothstep midpoint
        assert!((curve.evaluate(1.0) - 1.0).abs() < f32::EPSILON);
    }

    // -- TransitionCondition tests --

    #[test]
    fn test_condition_float_gt() {
        let sm = {
            let mut sm = AnimationStateMachine::new(AnimationState::from_clip("Idle", 0));
            sm.add_float_parameter("Speed", 0.0, 10.0, 5.0);
            sm
        };

        let cond = TransitionCondition::float_gt("Speed", 3.0);
        assert!(sm.check_condition(&cond));

        let cond = TransitionCondition::float_gt("Speed", 7.0);
        assert!(!sm.check_condition(&cond));
    }

    #[test]
    fn test_condition_bool() {
        let sm = {
            let mut sm = AnimationStateMachine::new(AnimationState::from_clip("Idle", 0));
            sm.add_bool_parameter("IsGrounded", true);
            sm
        };

        let cond = TransitionCondition::bool_eq("IsGrounded", true);
        assert!(sm.check_condition(&cond));

        let cond = TransitionCondition::bool_eq("IsGrounded", false);
        assert!(!sm.check_condition(&cond));
    }

    #[test]
    fn test_condition_trigger() {
        let mut sm = AnimationStateMachine::new(AnimationState::from_clip("Idle", 0));
        sm.add_trigger_parameter("Jump");

        let cond = TransitionCondition::trigger("Jump");
        assert!(!sm.check_condition(&cond));

        sm.set_trigger("Jump");
        assert!(sm.check_condition(&cond));
    }

    // -- AnimationStateMachine tests --

    #[test]
    fn test_state_machine_creation() {
        let sm = AnimationStateMachine::new(AnimationState::from_clip("Idle", 0));
        assert_eq!(sm.current_state_name(), "Idle");
        assert_eq!(sm.states.len(), 1);
    }

    #[test]
    fn test_state_machine_add_state() {
        let mut sm = AnimationStateMachine::new(AnimationState::from_clip("Idle", 0));
        let idx = sm.add_state(AnimationState::from_clip("Walk", 1));
        assert_eq!(idx, 1);
        assert_eq!(sm.states.len(), 2);
    }

    #[test]
    fn test_state_machine_transition() {
        let mut sm = AnimationStateMachine::new(AnimationState::from_clip("Idle", 0));
        sm.add_state(AnimationState::from_clip("Walk", 1));
        sm.add_float_parameter("Speed", 0.0, 10.0, 0.0);

        sm.add_transition(StateTransition::new(
            0,
            1,
            0.2,
            vec![TransitionCondition::float_gt("Speed", 0.5)],
        ));

        // Speed is 0, no transition.
        sm.update(0.016);
        assert_eq!(sm.current_state_name(), "Idle");
        assert!(!sm.transitioning);

        // Set speed above threshold.
        sm.set_float("Speed", 1.0);
        sm.update(0.016);
        assert!(sm.transitioning);
        assert_eq!(sm.transition_target, Some(1));

        // Advance past transition duration.
        sm.update(0.3);
        assert!(!sm.transitioning);
        assert_eq!(sm.current_state_name(), "Walk");
    }

    #[test]
    fn test_state_machine_trigger_transition() {
        let mut sm = AnimationStateMachine::new(AnimationState::from_clip("Idle", 0));
        sm.add_state(AnimationState::from_clip("Jump", 1).with_looping(false));
        sm.add_trigger_parameter("Jump");

        sm.add_transition(StateTransition::new(
            0,
            1,
            0.1,
            vec![TransitionCondition::trigger("Jump")],
        ));

        // No trigger, no transition.
        sm.update(0.016);
        assert_eq!(sm.current_state_name(), "Idle");

        // Fire trigger.
        sm.set_trigger("Jump");
        sm.update(0.016);
        assert!(sm.transitioning);

        // Complete transition.
        sm.update(0.2);
        assert_eq!(sm.current_state_name(), "Jump");

        // Trigger should be auto-reset.
        let trigger_value = sm
            .parameters
            .iter()
            .find(|p| p.name == "Jump")
            .unwrap()
            .value;
        assert!(trigger_value < 0.5, "Trigger should be auto-reset");
    }

    #[test]
    fn test_state_machine_force_state() {
        let mut sm = AnimationStateMachine::new(AnimationState::from_clip("Idle", 0));
        sm.add_state(AnimationState::from_clip("Walk", 1));
        sm.add_state(AnimationState::from_clip("Run", 2));

        sm.force_state(2);
        assert_eq!(sm.current_state_name(), "Run");
        assert!(!sm.transitioning);
    }

    #[test]
    fn test_state_machine_evaluate() {
        let clips = vec![
            make_clip("Idle", 1.0, 2),
            make_clip("Walk", 1.0, 2),
        ];

        let mut sm = AnimationStateMachine::new(AnimationState::from_clip("Idle", 0));
        sm.add_state(AnimationState::from_clip("Walk", 1));

        let pose = sm.evaluate(&clips, 2);
        assert_eq!(pose.len(), 2);
    }

    #[test]
    fn test_state_machine_evaluate_during_transition() {
        let clips = vec![
            {
                let mut c = AnimationClip::new("Idle", 1.0);
                c.looping = true;
                let mut t = BoneTrack::new(0);
                t.position_keys = vec![
                    Keyframe::new(0.0, Vec3::ZERO),
                    Keyframe::new(1.0, Vec3::ZERO),
                ];
                t.rotation_keys = vec![Keyframe::new(0.0, Quat::IDENTITY)];
                t.scale_keys = vec![Keyframe::new(0.0, Vec3::ONE)];
                c.add_track(t);
                c
            },
            {
                let mut c = AnimationClip::new("Walk", 1.0);
                c.looping = true;
                let mut t = BoneTrack::new(0);
                t.position_keys = vec![
                    Keyframe::new(0.0, Vec3::new(10.0, 0.0, 0.0)),
                    Keyframe::new(1.0, Vec3::new(10.0, 0.0, 0.0)),
                ];
                t.rotation_keys = vec![Keyframe::new(0.0, Quat::IDENTITY)];
                t.scale_keys = vec![Keyframe::new(0.0, Vec3::ONE)];
                c.add_track(t);
                c
            },
        ];

        let mut sm = AnimationStateMachine::new(AnimationState::from_clip("Idle", 0));
        sm.add_state(AnimationState::from_clip("Walk", 1));
        sm.add_float_parameter("Speed", 0.0, 10.0, 0.0);
        sm.add_transition(StateTransition::new(
            0,
            1,
            1.0, // 1 second transition
            vec![TransitionCondition::float_gt("Speed", 0.5)],
        ));

        sm.set_float("Speed", 1.0);
        sm.update(0.016);
        assert!(sm.transitioning);

        // Advance halfway through the transition.
        sm.update(0.484); // total elapsed ~0.5
        let pose = sm.evaluate(&clips, 1);

        // Should be blended: between 0.0 and 10.0
        assert!(pose[0].position.x > 0.0 && pose[0].position.x < 10.0,
            "During transition, pose should be blended: {}", pose[0].position.x);
    }

    #[test]
    fn test_state_machine_from_any_state() {
        let mut sm = AnimationStateMachine::new(AnimationState::from_clip("Idle", 0));
        sm.add_state(AnimationState::from_clip("Walk", 1));
        sm.add_state(AnimationState::from_clip("Death", 2));
        sm.add_trigger_parameter("Die");

        // From any state to Death.
        sm.add_transition(StateTransition::from_any(
            2,
            0.1,
            vec![TransitionCondition::trigger("Die")],
        ));

        // Start in Walk.
        sm.force_state(1);
        assert_eq!(sm.current_state_name(), "Walk");

        // Trigger death.
        sm.set_trigger("Die");
        sm.update(0.016);
        assert!(sm.transitioning);
        assert_eq!(sm.transition_target, Some(2));

        sm.update(0.2);
        assert_eq!(sm.current_state_name(), "Death");
    }

    #[test]
    fn test_state_machine_transition_progress() {
        let mut sm = AnimationStateMachine::new(AnimationState::from_clip("Idle", 0));
        sm.add_state(AnimationState::from_clip("Walk", 1));
        sm.add_float_parameter("Speed", 0.0, 10.0, 0.0);
        sm.add_transition(StateTransition::new(
            0,
            1,
            1.0,
            vec![TransitionCondition::float_gt("Speed", 0.5)],
        ));

        sm.set_float("Speed", 1.0);
        sm.update(0.0);
        sm.update(0.5);
        let progress = sm.transition_progress();
        assert!(progress > 0.3 && progress < 0.7, "Progress should be ~0.5: {}", progress);
    }

    #[test]
    fn test_state_with_tag() {
        let state = AnimationState::from_clip("Idle", 0)
            .with_tag("grounded")
            .with_looping(true)
            .with_speed(1.0);
        assert_eq!(state.tag.as_deref(), Some("grounded"));
        assert!(state.looping);
    }

    #[test]
    fn test_transition_with_curve() {
        let transition = StateTransition::new(0, 1, 0.5, vec![])
            .with_curve(TransitionCurve::EaseInOut)
            .with_exit_time(0.8)
            .interruptible();

        assert_eq!(transition.curve, TransitionCurve::EaseInOut);
        assert_eq!(transition.exit_time, Some(0.8));
        assert!(transition.can_be_interrupted);
    }

    // -- Integration test --

    #[test]
    fn test_full_state_machine_workflow() {
        let clips = vec![
            make_clip("Idle", 1.0, 3),
            make_clip("Walk", 1.0, 3),
            make_clip("Run", 1.0, 3),
        ];

        // Build a locomotion state with 1D blending.
        let locomotion_tree = BlendTree::new(
            BlendNode::blend_1d(
                "Speed",
                vec![
                    (0.0, BlendNode::clip(1)),
                    (1.0, BlendNode::clip(2)),
                ],
            ),
            vec![BlendParameter::new("Speed", 0.0, 1.0, 0.0)],
        );

        let mut sm = AnimationStateMachine::new(AnimationState::from_clip("Idle", 0));
        let _loco_idx = sm.add_state(AnimationState::new("Locomotion", locomotion_tree));
        sm.add_float_parameter("Speed", 0.0, 10.0, 0.0);

        sm.add_transition(StateTransition::new(
            0,
            1,
            0.3,
            vec![TransitionCondition::float_gt("Speed", 0.1)],
        ));
        sm.add_transition(StateTransition::new(
            1,
            0,
            0.3,
            vec![TransitionCondition::float_lt("Speed", 0.1)],
        ));

        // Start idle.
        assert_eq!(sm.current_state_name(), "Idle");

        // Start moving.
        sm.set_float("Speed", 2.0);
        for _ in 0..30 {
            sm.update(0.016);
        }
        assert_eq!(sm.current_state_name(), "Locomotion");

        // Evaluate pose.
        let pose = sm.evaluate(&clips, 3);
        assert_eq!(pose.len(), 3);

        // Stop moving.
        sm.set_float("Speed", 0.0);
        for _ in 0..30 {
            sm.update(0.016);
        }
        assert_eq!(sm.current_state_name(), "Idle");
    }
}
