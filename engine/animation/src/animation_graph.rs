//! # Animation State Graph
//!
//! A visual animation graph system with nodes and edges for driving character
//! animation in the Genovo engine. Supports multiple node types (state, blend,
//! select, layer, IK), transition conditions, parameter-driven blending,
//! sub-graph nesting, any-state transitions, and animation events emitted
//! from graph evaluation.
//!
//! ## Architecture
//!
//! An [`AnimationGraph`] is a directed graph where each node implements
//! [`GraphNode`]. Edges are typed transitions that fire when their
//! [`TransitionCondition`] is met. The graph evaluator walks the active
//! nodes each frame, evaluates blend weights, and produces a final
//! [`GraphOutput`] consumed by the skeleton.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Opaque identifier for a node in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// Opaque identifier for an edge (transition) in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(pub u32);

/// Opaque identifier for a graph parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamId(pub u32);

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// The value of a graph parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    /// Boolean flag.
    Bool(bool),
    /// Floating-point scalar.
    Float(f32),
    /// Integer scalar.
    Int(i32),
    /// Trigger (auto-resets after being consumed).
    Trigger(bool),
}

impl ParamValue {
    /// Try to read as a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            Self::Trigger(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to read as a float.
    pub fn as_float(&self) -> Option<f32> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Try to read as an integer.
    pub fn as_int(&self) -> Option<i32> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Float(v) => Some(*v as i32),
            _ => None,
        }
    }
}

impl Default for ParamValue {
    fn default() -> Self {
        Self::Float(0.0)
    }
}

/// Descriptor for a graph parameter.
#[derive(Debug, Clone)]
pub struct ParamDescriptor {
    /// Unique name.
    pub name: String,
    /// Unique identifier.
    pub id: ParamId,
    /// Default value.
    pub default: ParamValue,
    /// Current value.
    pub value: ParamValue,
}

// ---------------------------------------------------------------------------
// Transition conditions
// ---------------------------------------------------------------------------

/// A condition that must be met for a transition to fire.
#[derive(Debug, Clone)]
pub enum TransitionCondition {
    /// Always true (immediate transition).
    Always,
    /// Parameter equals a specific value.
    ParamEquals { param: ParamId, value: ParamValue },
    /// Float parameter is greater than a threshold.
    ParamGreaterThan { param: ParamId, threshold: f32 },
    /// Float parameter is less than a threshold.
    ParamLessThan { param: ParamId, threshold: f32 },
    /// Boolean/trigger parameter is true.
    ParamIsTrue(ParamId),
    /// Boolean/trigger parameter is false.
    ParamIsFalse(ParamId),
    /// The current state's animation has finished playing.
    AnimationFinished,
    /// A specified time has elapsed in the current state (seconds).
    TimeElapsed(f32),
    /// Multiple conditions must all be true (AND).
    All(Vec<TransitionCondition>),
    /// At least one condition must be true (OR).
    Any(Vec<TransitionCondition>),
    /// Negation of another condition.
    Not(Box<TransitionCondition>),
}

impl TransitionCondition {
    /// Evaluate the condition against the current parameter set and state.
    pub fn evaluate(&self, params: &HashMap<ParamId, ParamValue>, state: &NodeState) -> bool {
        match self {
            Self::Always => true,
            Self::ParamEquals { param, value } => {
                params.get(param).map_or(false, |v| v == value)
            }
            Self::ParamGreaterThan { param, threshold } => {
                params
                    .get(param)
                    .and_then(|v| v.as_float())
                    .map_or(false, |f| f > *threshold)
            }
            Self::ParamLessThan { param, threshold } => {
                params
                    .get(param)
                    .and_then(|v| v.as_float())
                    .map_or(false, |f| f < *threshold)
            }
            Self::ParamIsTrue(param) => {
                params
                    .get(param)
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            }
            Self::ParamIsFalse(param) => {
                params
                    .get(param)
                    .and_then(|v| v.as_bool())
                    .map_or(false, |b| !b)
            }
            Self::AnimationFinished => state.animation_finished,
            Self::TimeElapsed(t) => state.time_in_state >= *t,
            Self::All(conditions) => conditions.iter().all(|c| c.evaluate(params, state)),
            Self::Any(conditions) => conditions.iter().any(|c| c.evaluate(params, state)),
            Self::Not(inner) => !inner.evaluate(params, state),
        }
    }
}

// ---------------------------------------------------------------------------
// Node types
// ---------------------------------------------------------------------------

/// The type of a graph node.
#[derive(Debug, Clone)]
pub enum NodeType {
    /// A single animation state (plays a clip).
    State(StateNode),
    /// Blends between two child nodes.
    Blend(BlendNode),
    /// Selects one of N child nodes based on a parameter.
    Select(SelectNode),
    /// An animation layer (additive or override).
    Layer(LayerNode),
    /// An inverse kinematics node.
    Ik(IkNode),
    /// A sub-graph reference.
    SubGraph(SubGraphNode),
    /// An entry point marker (does not produce animation).
    Entry,
    /// An exit point marker.
    Exit,
}

/// A node that plays a single animation clip.
#[derive(Debug, Clone)]
pub struct StateNode {
    /// Name of the animation clip to play.
    pub clip_name: String,
    /// Playback speed multiplier.
    pub speed: f32,
    /// Whether the clip should loop.
    pub looping: bool,
    /// Whether to mirror the animation.
    pub mirror: bool,
    /// Start time offset in seconds.
    pub start_offset: f32,
    /// Bone mask name (None = all bones).
    pub bone_mask: Option<String>,
    /// Events emitted at specific times in the clip.
    pub events: Vec<AnimGraphEvent>,
}

impl Default for StateNode {
    fn default() -> Self {
        Self {
            clip_name: String::new(),
            speed: 1.0,
            looping: true,
            mirror: false,
            start_offset: 0.0,
            bone_mask: None,
            events: Vec::new(),
        }
    }
}

/// A node that blends between two children.
#[derive(Debug, Clone)]
pub struct BlendNode {
    /// The parameter controlling the blend weight.
    pub blend_param: ParamId,
    /// Child A (weight = 1 - blend).
    pub child_a: NodeId,
    /// Child B (weight = blend).
    pub child_b: NodeId,
    /// Blend mode.
    pub blend_mode: BlendMode,
}

/// How two animations are blended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Linear interpolation of transforms.
    Linear,
    /// Additive blending (B is added on top of A).
    Additive,
    /// Override (B replaces A where weight > 0).
    Override,
}

impl Default for BlendMode {
    fn default() -> Self {
        Self::Linear
    }
}

/// A node that selects one of N children based on an integer parameter.
#[derive(Debug, Clone)]
pub struct SelectNode {
    /// Parameter used to select the active child.
    pub select_param: ParamId,
    /// Ordered list of child nodes.
    pub children: Vec<NodeId>,
    /// Transition time when switching between children (seconds).
    pub cross_fade: f32,
}

/// A layer node for additive/override blending.
#[derive(Debug, Clone)]
pub struct LayerNode {
    /// Base layer (underneath).
    pub base: NodeId,
    /// Overlay layer (on top).
    pub overlay: NodeId,
    /// Weight of the overlay (0 = base only, 1 = full overlay).
    pub weight: f32,
    /// Weight parameter (if driven by a parameter).
    pub weight_param: Option<ParamId>,
    /// Blend mode for the overlay.
    pub blend_mode: BlendMode,
    /// Bone mask for the overlay (None = all bones).
    pub bone_mask: Option<String>,
}

/// An IK node that adjusts the output pose.
#[derive(Debug, Clone)]
pub struct IkNode {
    /// Child node providing the input pose.
    pub child: NodeId,
    /// IK chain name (matches skeleton definition).
    pub chain_name: String,
    /// Target position (world or local space).
    pub target: IkTarget,
    /// IK solver weight (0 = no IK, 1 = full IK).
    pub weight: f32,
    /// Weight parameter.
    pub weight_param: Option<ParamId>,
}

/// Target specification for an IK node.
#[derive(Debug, Clone)]
pub enum IkTarget {
    /// Fixed position in local space.
    LocalPosition([f32; 3]),
    /// Driven by a parameter (x, y, z parameter IDs).
    ParameterDriven(ParamId, ParamId, ParamId),
    /// Follows another bone.
    Bone(String),
}

/// A reference to a nested sub-graph.
#[derive(Debug, Clone)]
pub struct SubGraphNode {
    /// The name of the sub-graph.
    pub graph_name: String,
    /// Parameter mappings: parent param -> sub-graph param name.
    pub param_bindings: HashMap<ParamId, String>,
}

// ---------------------------------------------------------------------------
// Graph edge (transition)
// ---------------------------------------------------------------------------

/// A transition between two nodes in the graph.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// Unique edge identifier.
    pub id: EdgeId,
    /// Source node.
    pub from: NodeId,
    /// Destination node.
    pub to: NodeId,
    /// Condition that must be met to fire.
    pub condition: TransitionCondition,
    /// Transition duration in seconds (cross-fade time).
    pub duration: f32,
    /// Priority (higher = evaluated first when multiple transitions are valid).
    pub priority: i32,
    /// Whether this transition can be interrupted by higher-priority ones.
    pub interruptible: bool,
    /// Whether this is an "any state" transition (can fire from any node).
    pub any_state: bool,
    /// Optional offset into the destination animation (0.0 - 1.0).
    pub target_offset: f32,
}

impl GraphEdge {
    /// Create a simple transition.
    pub fn new(id: EdgeId, from: NodeId, to: NodeId, duration: f32) -> Self {
        Self {
            id,
            from,
            to,
            condition: TransitionCondition::Always,
            duration,
            priority: 0,
            interruptible: true,
            any_state: false,
            target_offset: 0.0,
        }
    }

    /// Create an any-state transition.
    pub fn any_state(id: EdgeId, to: NodeId, condition: TransitionCondition, duration: f32) -> Self {
        Self {
            id,
            from: NodeId(u32::MAX), // sentinel
            to,
            condition,
            duration,
            priority: 100,
            interruptible: true,
            any_state: true,
            target_offset: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Animation events
// ---------------------------------------------------------------------------

/// An event emitted from the animation graph during evaluation.
#[derive(Debug, Clone)]
pub struct AnimGraphEvent {
    /// Name of the event.
    pub name: String,
    /// Time within the clip when the event fires (seconds).
    pub time: f32,
    /// Optional string payload.
    pub string_data: Option<String>,
    /// Optional float payload.
    pub float_data: Option<f32>,
    /// Optional integer payload.
    pub int_data: Option<i32>,
}

// ---------------------------------------------------------------------------
// Node runtime state
// ---------------------------------------------------------------------------

/// Runtime state for a single graph node.
#[derive(Debug, Clone)]
pub struct NodeState {
    /// Elapsed time in this state (seconds).
    pub time_in_state: f32,
    /// Current playback time within the animation clip.
    pub clip_time: f32,
    /// Whether the clip has finished (non-looping clips only).
    pub animation_finished: bool,
    /// Current blend weight of this node in the graph.
    pub weight: f32,
    /// Whether this node is currently active.
    pub active: bool,
}

impl Default for NodeState {
    fn default() -> Self {
        Self {
            time_in_state: 0.0,
            clip_time: 0.0,
            animation_finished: false,
            weight: 1.0,
            active: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Transition state
// ---------------------------------------------------------------------------

/// Runtime state for an in-progress transition.
#[derive(Debug, Clone)]
pub struct TransitionState {
    /// The edge being traversed.
    pub edge_id: EdgeId,
    /// Source node.
    pub from: NodeId,
    /// Destination node.
    pub to: NodeId,
    /// Total transition duration.
    pub duration: f32,
    /// Elapsed time in the transition.
    pub elapsed: f32,
    /// Progress [0, 1].
    pub progress: f32,
}

impl TransitionState {
    /// Whether the transition is complete.
    pub fn is_complete(&self) -> bool {
        self.elapsed >= self.duration
    }
}

// ---------------------------------------------------------------------------
// Graph output
// ---------------------------------------------------------------------------

/// Output of evaluating the animation graph for a single frame.
#[derive(Debug, Clone)]
pub struct GraphOutput {
    /// Active clip names and their weights.
    pub active_clips: Vec<(String, f32)>,
    /// Events fired this frame.
    pub events: Vec<AnimGraphEvent>,
    /// The current state name (for debugging).
    pub current_state: String,
    /// Whether a transition is in progress.
    pub transitioning: bool,
    /// Transition progress if transitioning.
    pub transition_progress: f32,
}

impl Default for GraphOutput {
    fn default() -> Self {
        Self {
            active_clips: Vec::new(),
            events: Vec::new(),
            current_state: String::new(),
            transitioning: false,
            transition_progress: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Graph node container
// ---------------------------------------------------------------------------

/// A node in the animation graph.
#[derive(Debug, Clone)]
pub struct GraphNodeData {
    /// Unique identifier.
    pub id: NodeId,
    /// Human-readable name.
    pub name: String,
    /// The node type and its data.
    pub node_type: NodeType,
    /// Visual position in the editor (x, y).
    pub editor_position: [f32; 2],
    /// Runtime state.
    pub state: NodeState,
}

impl GraphNodeData {
    /// Create a new node.
    pub fn new(id: NodeId, name: impl Into<String>, node_type: NodeType) -> Self {
        Self {
            id,
            name: name.into(),
            node_type,
            editor_position: [0.0, 0.0],
            state: NodeState::default(),
        }
    }

    /// Reset the runtime state.
    pub fn reset_state(&mut self) {
        self.state = NodeState::default();
    }

    /// Whether this node is a leaf (State node with a clip).
    pub fn is_leaf(&self) -> bool {
        matches!(self.node_type, NodeType::State(_))
    }

    /// Get the clip name if this is a state node.
    pub fn clip_name(&self) -> Option<&str> {
        match &self.node_type {
            NodeType::State(s) => Some(&s.clip_name),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Animation Graph
// ---------------------------------------------------------------------------

/// A complete animation state graph.
pub struct AnimationGraph {
    /// Human-readable name.
    pub name: String,
    /// All nodes, indexed by NodeId.
    nodes: HashMap<NodeId, GraphNodeData>,
    /// All edges (transitions).
    edges: Vec<GraphEdge>,
    /// Any-state transitions (can fire from any node).
    any_state_edges: Vec<GraphEdge>,
    /// Parameters.
    params: HashMap<ParamId, ParamDescriptor>,
    /// Parameter lookup by name.
    param_names: HashMap<String, ParamId>,
    /// Current active node.
    active_node: Option<NodeId>,
    /// Entry node (default starting state).
    entry_node: Option<NodeId>,
    /// Current transition (if any).
    current_transition: Option<TransitionState>,
    /// Events accumulated during the current evaluation.
    pending_events: Vec<AnimGraphEvent>,
    /// Next node ID counter.
    next_node_id: u32,
    /// Next edge ID counter.
    next_edge_id: u32,
    /// Next parameter ID counter.
    next_param_id: u32,
    /// Sub-graphs referenced by SubGraphNodes.
    sub_graphs: HashMap<String, AnimationGraph>,
}

impl AnimationGraph {
    /// Create a new empty graph.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: HashMap::new(),
            edges: Vec::new(),
            any_state_edges: Vec::new(),
            params: HashMap::new(),
            param_names: HashMap::new(),
            active_node: None,
            entry_node: None,
            current_transition: None,
            pending_events: Vec::new(),
            next_node_id: 0,
            next_edge_id: 0,
            next_param_id: 0,
            sub_graphs: HashMap::new(),
        }
    }

    // -- Node management ---

    /// Add a node to the graph and return its ID.
    pub fn add_node(&mut self, name: impl Into<String>, node_type: NodeType) -> NodeId {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        let node = GraphNodeData::new(id, name, node_type);
        self.nodes.insert(id, node);
        id
    }

    /// Set the entry node (starting state).
    pub fn set_entry(&mut self, id: NodeId) {
        self.entry_node = Some(id);
    }

    /// Remove a node and all its edges.
    pub fn remove_node(&mut self, id: NodeId) {
        self.nodes.remove(&id);
        self.edges.retain(|e| e.from != id && e.to != id);
        self.any_state_edges.retain(|e| e.to != id);
        if self.active_node == Some(id) {
            self.active_node = self.entry_node;
        }
        if self.entry_node == Some(id) {
            self.entry_node = None;
        }
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: NodeId) -> Option<&GraphNodeData> {
        self.nodes.get(&id)
    }

    /// Get a mutable reference to a node.
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut GraphNodeData> {
        self.nodes.get_mut(&id)
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// All node IDs.
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.nodes.keys().copied().collect()
    }

    // -- Edge management ---

    /// Add a transition edge between two nodes.
    pub fn add_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        condition: TransitionCondition,
        duration: f32,
    ) -> EdgeId {
        let id = EdgeId(self.next_edge_id);
        self.next_edge_id += 1;
        let mut edge = GraphEdge::new(id, from, to, duration);
        edge.condition = condition;
        self.edges.push(edge);
        id
    }

    /// Add an any-state transition.
    pub fn add_any_state_edge(
        &mut self,
        to: NodeId,
        condition: TransitionCondition,
        duration: f32,
    ) -> EdgeId {
        let id = EdgeId(self.next_edge_id);
        self.next_edge_id += 1;
        let edge = GraphEdge::any_state(id, to, condition, duration);
        self.any_state_edges.push(edge);
        id
    }

    /// Remove an edge by ID.
    pub fn remove_edge(&mut self, id: EdgeId) {
        self.edges.retain(|e| e.id != id);
        self.any_state_edges.retain(|e| e.id != id);
    }

    /// Get all edges from a specific node.
    pub fn edges_from(&self, id: NodeId) -> Vec<&GraphEdge> {
        self.edges.iter().filter(|e| e.from == id).collect()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len() + self.any_state_edges.len()
    }

    // -- Parameter management ---

    /// Add a parameter to the graph.
    pub fn add_param(&mut self, name: impl Into<String>, default: ParamValue) -> ParamId {
        let id = ParamId(self.next_param_id);
        self.next_param_id += 1;
        let name = name.into();
        let desc = ParamDescriptor {
            name: name.clone(),
            id,
            default: default.clone(),
            value: default,
        };
        self.params.insert(id, desc);
        self.param_names.insert(name, id);
        id
    }

    /// Set a parameter value.
    pub fn set_param(&mut self, id: ParamId, value: ParamValue) {
        if let Some(desc) = self.params.get_mut(&id) {
            desc.value = value;
        }
    }

    /// Set a parameter by name.
    pub fn set_param_by_name(&mut self, name: &str, value: ParamValue) {
        if let Some(&id) = self.param_names.get(name) {
            self.set_param(id, value);
        }
    }

    /// Get a parameter value.
    pub fn get_param(&self, id: ParamId) -> Option<&ParamValue> {
        self.params.get(&id).map(|d| &d.value)
    }

    /// Get a parameter ID by name.
    pub fn param_id(&self, name: &str) -> Option<ParamId> {
        self.param_names.get(name).copied()
    }

    /// Collect current parameter values into a HashMap for condition evaluation.
    fn current_params(&self) -> HashMap<ParamId, ParamValue> {
        self.params
            .iter()
            .map(|(&id, desc)| (id, desc.value.clone()))
            .collect()
    }

    // -- Sub-graph management ---

    /// Register a sub-graph.
    pub fn add_sub_graph(&mut self, name: impl Into<String>, graph: AnimationGraph) {
        self.sub_graphs.insert(name.into(), graph);
    }

    /// Get a sub-graph by name.
    pub fn get_sub_graph(&self, name: &str) -> Option<&AnimationGraph> {
        self.sub_graphs.get(name)
    }

    // -- Evaluation ---

    /// Start the graph evaluation at the entry node.
    pub fn start(&mut self) {
        if let Some(entry) = self.entry_node {
            self.active_node = Some(entry);
            if let Some(node) = self.nodes.get_mut(&entry) {
                node.state.active = true;
            }
        }
        self.current_transition = None;
        self.pending_events.clear();
    }

    /// Reset the graph to its initial state.
    pub fn reset(&mut self) {
        for node in self.nodes.values_mut() {
            node.reset_state();
        }
        self.active_node = self.entry_node;
        self.current_transition = None;
        self.pending_events.clear();
        // Reset triggers
        for desc in self.params.values_mut() {
            if matches!(desc.value, ParamValue::Trigger(_)) {
                desc.value = ParamValue::Trigger(false);
            }
        }
    }

    /// Evaluate the graph for one frame.
    pub fn evaluate(&mut self, dt: f32) -> GraphOutput {
        self.pending_events.clear();
        let params = self.current_params();

        // Update active node state
        if let Some(active_id) = self.active_node {
            if let Some(node) = self.nodes.get_mut(&active_id) {
                node.state.time_in_state += dt;
                node.state.active = true;
                // Update clip time for state nodes
                if let NodeType::State(ref state_data) = node.node_type {
                    let speed = state_data.speed;
                    node.state.clip_time += dt * speed;
                    // Check for events
                    let prev_time = node.state.clip_time - dt * speed;
                    for event in &state_data.events {
                        if prev_time < event.time && node.state.clip_time >= event.time {
                            self.pending_events.push(event.clone());
                        }
                    }
                }
            }
        }

        // Update transition if one is in progress
        if let Some(ref mut transition) = self.current_transition {
            transition.elapsed += dt;
            transition.progress = if transition.duration > 0.0 {
                (transition.elapsed / transition.duration).min(1.0)
            } else {
                1.0
            };
            if transition.is_complete() {
                // Transition complete: switch to destination
                let dest = transition.to;
                if let Some(old_id) = self.active_node {
                    if let Some(old_node) = self.nodes.get_mut(&old_id) {
                        old_node.state.active = false;
                    }
                }
                self.active_node = Some(dest);
                if let Some(new_node) = self.nodes.get_mut(&dest) {
                    new_node.state.active = true;
                    new_node.state.time_in_state = 0.0;
                    new_node.state.clip_time = 0.0;
                    new_node.state.animation_finished = false;
                }
                self.current_transition = None;
            }
        }

        // Check for new transitions (only if not currently transitioning)
        if self.current_transition.is_none() {
            if let Some(active_id) = self.active_node {
                let node_state = self
                    .nodes
                    .get(&active_id)
                    .map(|n| n.state.clone())
                    .unwrap_or_default();

                // Check any-state transitions first (higher priority)
                let mut best_edge: Option<&GraphEdge> = None;
                for edge in &self.any_state_edges {
                    if edge.to == active_id {
                        continue; // Don't transition to ourselves
                    }
                    if edge.condition.evaluate(&params, &node_state) {
                        if best_edge.map_or(true, |b| edge.priority > b.priority) {
                            best_edge = Some(edge);
                        }
                    }
                }

                // Check normal edges from the active node
                for edge in &self.edges {
                    if edge.from != active_id {
                        continue;
                    }
                    if edge.condition.evaluate(&params, &node_state) {
                        if best_edge.map_or(true, |b| edge.priority > b.priority) {
                            best_edge = Some(edge);
                        }
                    }
                }

                if let Some(edge) = best_edge {
                    self.current_transition = Some(TransitionState {
                        edge_id: edge.id,
                        from: active_id,
                        to: edge.to,
                        duration: edge.duration,
                        elapsed: 0.0,
                        progress: 0.0,
                    });
                }
            }
        }

        // Reset consumed triggers
        for desc in self.params.values_mut() {
            if matches!(desc.value, ParamValue::Trigger(true)) {
                desc.value = ParamValue::Trigger(false);
            }
        }

        // Build output
        let mut output = GraphOutput::default();

        if let Some(active_id) = self.active_node {
            if let Some(node) = self.nodes.get(&active_id) {
                output.current_state = node.name.clone();
                if let NodeType::State(ref state_data) = node.node_type {
                    let weight = if let Some(ref trans) = self.current_transition {
                        if trans.from == active_id {
                            1.0 - trans.progress
                        } else {
                            trans.progress
                        }
                    } else {
                        1.0
                    };
                    output
                        .active_clips
                        .push((state_data.clip_name.clone(), weight));
                }
            }
        }

        // Add destination state clips during transition
        if let Some(ref trans) = self.current_transition {
            output.transitioning = true;
            output.transition_progress = trans.progress;
            if let Some(dest_node) = self.nodes.get(&trans.to) {
                if let NodeType::State(ref state_data) = dest_node.node_type {
                    output
                        .active_clips
                        .push((state_data.clip_name.clone(), trans.progress));
                }
            }
        }

        output.events = self.pending_events.clone();
        output
    }

    /// Get the current active node ID.
    pub fn active_node(&self) -> Option<NodeId> {
        self.active_node
    }

    /// Get the current active node name.
    pub fn active_node_name(&self) -> Option<&str> {
        self.active_node
            .and_then(|id| self.nodes.get(&id))
            .map(|n| n.name.as_str())
    }

    /// Whether a transition is currently in progress.
    pub fn is_transitioning(&self) -> bool {
        self.current_transition.is_some()
    }

    // -- Validation ---

    /// Validate the graph for common issues.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.entry_node.is_none() {
            issues.push("no entry node set".into());
        }

        if let Some(entry) = self.entry_node {
            if !self.nodes.contains_key(&entry) {
                issues.push("entry node does not exist".into());
            }
        }

        // Check for orphan nodes (no incoming or outgoing edges)
        for (&id, node) in &self.nodes {
            if matches!(node.node_type, NodeType::Entry | NodeType::Exit) {
                continue;
            }
            let has_incoming = self.edges.iter().any(|e| e.to == id)
                || self.any_state_edges.iter().any(|e| e.to == id)
                || self.entry_node == Some(id);
            let has_outgoing = self.edges.iter().any(|e| e.from == id);
            if !has_incoming && !has_outgoing {
                issues.push(format!("node '{}' is disconnected", node.name));
            }
        }

        // Check edges reference valid nodes
        for edge in &self.edges {
            if !self.nodes.contains_key(&edge.from) {
                issues.push(format!("edge {:?} references missing source node", edge.id));
            }
            if !self.nodes.contains_key(&edge.to) {
                issues.push(format!("edge {:?} references missing dest node", edge.id));
            }
        }

        // Check for sub-graph references
        for node in self.nodes.values() {
            if let NodeType::SubGraph(ref sg) = node.node_type {
                if !self.sub_graphs.contains_key(&sg.graph_name) {
                    issues.push(format!(
                        "node '{}' references missing sub-graph '{}'",
                        node.name, sg.graph_name
                    ));
                }
            }
        }

        issues
    }

    /// Find all reachable nodes from the entry point.
    pub fn reachable_nodes(&self) -> HashSet<NodeId> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        if let Some(entry) = self.entry_node {
            queue.push_back(entry);
        }
        while let Some(id) = queue.pop_front() {
            if visited.contains(&id) {
                continue;
            }
            visited.insert(id);
            for edge in &self.edges {
                if edge.from == id && !visited.contains(&edge.to) {
                    queue.push_back(edge.to);
                }
            }
        }
        // Any-state transitions can reach any node
        for edge in &self.any_state_edges {
            visited.insert(edge.to);
        }
        visited
    }
}

impl Default for AnimationGraph {
    fn default() -> Self {
        Self::new("default")
    }
}

impl fmt::Debug for AnimationGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnimationGraph")
            .field("name", &self.name)
            .field("nodes", &self.nodes.len())
            .field("edges", &self.edges.len())
            .field("params", &self.params.len())
            .field("active_node", &self.active_node)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_graph() -> AnimationGraph {
        let mut graph = AnimationGraph::new("test_graph");

        // Parameters
        let is_running = graph.add_param("is_running", ParamValue::Bool(false));
        let speed = graph.add_param("speed", ParamValue::Float(0.0));

        // Nodes
        let idle = graph.add_node(
            "idle",
            NodeType::State(StateNode {
                clip_name: "idle_anim".into(),
                ..Default::default()
            }),
        );
        let run = graph.add_node(
            "run",
            NodeType::State(StateNode {
                clip_name: "run_anim".into(),
                ..Default::default()
            }),
        );

        graph.set_entry(idle);

        // Transitions
        graph.add_edge(
            idle,
            run,
            TransitionCondition::ParamIsTrue(is_running),
            0.2,
        );
        graph.add_edge(
            run,
            idle,
            TransitionCondition::ParamIsFalse(is_running),
            0.3,
        );

        graph
    }

    #[test]
    fn test_graph_creation() {
        let graph = make_simple_graph();
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_graph_start() {
        let mut graph = make_simple_graph();
        graph.start();
        assert!(graph.active_node().is_some());
        assert_eq!(graph.active_node_name(), Some("idle"));
    }

    #[test]
    fn test_graph_transition() {
        let mut graph = make_simple_graph();
        graph.start();

        // Should start in idle
        let output = graph.evaluate(0.016);
        assert_eq!(output.current_state, "idle");
        assert!(!output.transitioning);

        // Set is_running = true
        let param_id = graph.param_id("is_running").unwrap();
        graph.set_param(param_id, ParamValue::Bool(true));

        // Evaluate: should start transitioning to run
        let output = graph.evaluate(0.016);
        assert!(output.transitioning);

        // Advance past transition duration (0.2s)
        for _ in 0..20 {
            graph.evaluate(0.016);
        }

        // Should be in run now
        assert_eq!(graph.active_node_name(), Some("run"));
        assert!(!graph.is_transitioning());
    }

    #[test]
    fn test_graph_any_state_transition() {
        let mut graph = AnimationGraph::new("any_state_test");
        let trigger = graph.add_param("hit", ParamValue::Trigger(false));

        let idle = graph.add_node(
            "idle",
            NodeType::State(StateNode {
                clip_name: "idle".into(),
                ..Default::default()
            }),
        );
        let hit_react = graph.add_node(
            "hit_react",
            NodeType::State(StateNode {
                clip_name: "hit_react".into(),
                looping: false,
                ..Default::default()
            }),
        );
        graph.set_entry(idle);
        graph.add_any_state_edge(
            hit_react,
            TransitionCondition::ParamIsTrue(trigger),
            0.1,
        );

        graph.start();
        graph.evaluate(0.016);
        assert_eq!(graph.active_node_name(), Some("idle"));

        // Fire trigger
        graph.set_param(trigger, ParamValue::Trigger(true));
        graph.evaluate(0.016);
        assert!(graph.is_transitioning());
    }

    #[test]
    fn test_graph_events() {
        let mut graph = AnimationGraph::new("events_test");
        let idle = graph.add_node(
            "idle",
            NodeType::State(StateNode {
                clip_name: "idle".into(),
                events: vec![AnimGraphEvent {
                    name: "footstep".into(),
                    time: 0.01,
                    string_data: None,
                    float_data: None,
                    int_data: None,
                }],
                ..Default::default()
            }),
        );
        graph.set_entry(idle);
        graph.start();
        let output = graph.evaluate(0.02);
        assert_eq!(output.events.len(), 1);
        assert_eq!(output.events[0].name, "footstep");
    }

    #[test]
    fn test_graph_validation() {
        let mut graph = AnimationGraph::new("validation_test");
        let issues = graph.validate();
        assert!(issues.iter().any(|i| i.contains("no entry node")));

        let node = graph.add_node(
            "orphan",
            NodeType::State(StateNode {
                clip_name: "clip".into(),
                ..Default::default()
            }),
        );
        graph.set_entry(node);
        let issues = graph.validate();
        assert!(issues.is_empty() || issues.iter().all(|i| !i.contains("no entry")));
    }

    #[test]
    fn test_graph_reset() {
        let mut graph = make_simple_graph();
        graph.start();
        graph.evaluate(0.5);
        graph.reset();
        assert_eq!(graph.active_node_name(), Some("idle"));
        let node = graph.get_node(graph.active_node().unwrap()).unwrap();
        assert_eq!(node.state.time_in_state, 0.0);
    }

    #[test]
    fn test_param_value_conversions() {
        let b = ParamValue::Bool(true);
        assert_eq!(b.as_bool(), Some(true));
        assert_eq!(b.as_float(), None);

        let f = ParamValue::Float(3.14);
        assert_eq!(f.as_float(), Some(3.14));
        assert_eq!(f.as_int(), Some(3));

        let i = ParamValue::Int(42);
        assert_eq!(i.as_int(), Some(42));
        assert_eq!(i.as_float(), Some(42.0));
    }

    #[test]
    fn test_condition_all_any() {
        let p1 = ParamId(0);
        let p2 = ParamId(1);
        let mut params = HashMap::new();
        params.insert(p1, ParamValue::Bool(true));
        params.insert(p2, ParamValue::Bool(false));
        let state = NodeState::default();

        let all_cond = TransitionCondition::All(vec![
            TransitionCondition::ParamIsTrue(p1),
            TransitionCondition::ParamIsTrue(p2),
        ]);
        assert!(!all_cond.evaluate(&params, &state));

        let any_cond = TransitionCondition::Any(vec![
            TransitionCondition::ParamIsTrue(p1),
            TransitionCondition::ParamIsTrue(p2),
        ]);
        assert!(any_cond.evaluate(&params, &state));
    }

    #[test]
    fn test_condition_not() {
        let p = ParamId(0);
        let mut params = HashMap::new();
        params.insert(p, ParamValue::Bool(true));
        let state = NodeState::default();

        let cond = TransitionCondition::Not(Box::new(TransitionCondition::ParamIsTrue(p)));
        assert!(!cond.evaluate(&params, &state));
    }

    #[test]
    fn test_graph_remove_node() {
        let mut graph = make_simple_graph();
        let ids = graph.node_ids();
        assert_eq!(ids.len(), 2);
        graph.remove_node(ids[1]);
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn test_graph_reachable_nodes() {
        let graph = make_simple_graph();
        let reachable = graph.reachable_nodes();
        assert_eq!(reachable.len(), 2);
    }

    #[test]
    fn test_select_node_creation() {
        let mut graph = AnimationGraph::new("select_test");
        let param = graph.add_param("weapon_type", ParamValue::Int(0));
        let a = graph.add_node("pistol", NodeType::State(StateNode {
            clip_name: "pistol_idle".into(),
            ..Default::default()
        }));
        let b = graph.add_node("rifle", NodeType::State(StateNode {
            clip_name: "rifle_idle".into(),
            ..Default::default()
        }));
        let _select = graph.add_node("weapon_select", NodeType::Select(SelectNode {
            select_param: param,
            children: vec![a, b],
            cross_fade: 0.3,
        }));
        assert_eq!(graph.node_count(), 3);
    }

    #[test]
    fn test_blend_node_creation() {
        let mut graph = AnimationGraph::new("blend_test");
        let blend_param = graph.add_param("walk_run", ParamValue::Float(0.0));
        let walk = graph.add_node("walk", NodeType::State(StateNode {
            clip_name: "walk".into(),
            ..Default::default()
        }));
        let run = graph.add_node("run", NodeType::State(StateNode {
            clip_name: "run".into(),
            ..Default::default()
        }));
        let _blend = graph.add_node("locomotion", NodeType::Blend(BlendNode {
            blend_param,
            child_a: walk,
            child_b: run,
            blend_mode: BlendMode::Linear,
        }));
        assert_eq!(graph.node_count(), 3);
    }

    #[test]
    fn test_layer_node_creation() {
        let mut graph = AnimationGraph::new("layer_test");
        let weight = graph.add_param("upper_weight", ParamValue::Float(1.0));
        let body = graph.add_node("body", NodeType::State(StateNode {
            clip_name: "body_idle".into(),
            ..Default::default()
        }));
        let upper = graph.add_node("upper", NodeType::State(StateNode {
            clip_name: "aim".into(),
            ..Default::default()
        }));
        let _layer = graph.add_node("layered", NodeType::Layer(LayerNode {
            base: body,
            overlay: upper,
            weight: 1.0,
            weight_param: Some(weight),
            blend_mode: BlendMode::Override,
            bone_mask: Some("upper_body".into()),
        }));
        assert_eq!(graph.node_count(), 3);
    }

    #[test]
    fn test_sub_graph() {
        let mut sub = AnimationGraph::new("sub_locomotion");
        let idle = sub.add_node("idle", NodeType::State(StateNode {
            clip_name: "idle".into(),
            ..Default::default()
        }));
        sub.set_entry(idle);

        let mut main = AnimationGraph::new("main");
        main.add_sub_graph("sub_locomotion", sub);
        assert!(main.get_sub_graph("sub_locomotion").is_some());
    }

    #[test]
    fn test_edge_creation() {
        let edge = GraphEdge::new(EdgeId(0), NodeId(0), NodeId(1), 0.25);
        assert!(!edge.any_state);
        assert_eq!(edge.duration, 0.25);
    }

    #[test]
    fn test_any_state_edge() {
        let edge = GraphEdge::any_state(
            EdgeId(0),
            NodeId(1),
            TransitionCondition::Always,
            0.1,
        );
        assert!(edge.any_state);
        assert_eq!(edge.priority, 100);
    }

    #[test]
    fn test_param_set_by_name() {
        let mut graph = AnimationGraph::new("param_test");
        graph.add_param("speed", ParamValue::Float(0.0));
        graph.set_param_by_name("speed", ParamValue::Float(5.0));
        let id = graph.param_id("speed").unwrap();
        assert_eq!(graph.get_param(id), Some(&ParamValue::Float(5.0)));
    }

    #[test]
    fn test_graph_output_clips_during_transition() {
        let mut graph = make_simple_graph();
        graph.start();

        let param_id = graph.param_id("is_running").unwrap();
        graph.set_param(param_id, ParamValue::Bool(true));
        graph.evaluate(0.016); // Start transition

        // During transition, both clips should be active
        let output = graph.evaluate(0.016);
        assert!(output.transitioning);
        // We should have the source clip at least
        assert!(!output.active_clips.is_empty());
    }
}
