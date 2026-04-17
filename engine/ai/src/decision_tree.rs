//! Decision tree system for AI reasoning.
//!
//! Provides:
//! - **Binary decision nodes**: yes/no branches based on conditions
//! - **Leaf action nodes**: terminal nodes that return an action
//! - **Weighted random selection**: probabilistic branch selection
//! - **Decision tree builder**: fluent API for constructing trees
//! - **Tree serialization**: export/import as data
//! - **Tree visualization data**: generate data for debug rendering
//! - **Runtime statistics**: per-node execution counts and timings
//!
//! # Design
//!
//! A [`DecisionTree`] is a binary tree of [`DecisionNode`]s. Each internal
//! node evaluates a [`Condition`] against a [`DecisionContext`] and branches
//! left (true) or right (false). Leaf nodes contain [`Action`]s to execute.
//! The [`DecisionTreeEvaluator`] traverses the tree and returns the
//! selected action.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum tree depth.
pub const MAX_TREE_DEPTH: usize = 32;
/// Maximum number of nodes in a tree.
pub const MAX_TREE_NODES: usize = 1024;
/// Maximum number of condition parameters.
pub const MAX_CONDITION_PARAMS: usize = 8;
/// Small epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// NodeId
// ---------------------------------------------------------------------------

/// Unique identifier for a node within a tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    pub fn new(id: u32) -> Self { Self(id) }
    /// The null/invalid node ID.
    pub const NONE: NodeId = NodeId(u32::MAX);
}

// ---------------------------------------------------------------------------
// TreeId
// ---------------------------------------------------------------------------

/// Unique identifier for a decision tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TreeId(pub u64);

// ---------------------------------------------------------------------------
// ConditionValue
// ---------------------------------------------------------------------------

/// A value used in condition evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum ConditionValue {
    /// Boolean value.
    Bool(bool),
    /// Integer value.
    Int(i64),
    /// Float value.
    Float(f64),
    /// String value.
    String(String),
}

impl ConditionValue {
    /// Try to get as bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            Self::Int(v) => Some(*v != 0),
            _ => None,
        }
    }

    /// Try to get as f64.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Try to get as i64.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Float(v) => Some(*v as i64),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// DecisionContext
// ---------------------------------------------------------------------------

/// Context providing data for decision evaluation.
#[derive(Debug, Clone)]
pub struct DecisionContext {
    /// Named values accessible to conditions.
    values: HashMap<String, ConditionValue>,
}

impl DecisionContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Set a value.
    pub fn set(&mut self, key: impl Into<String>, value: ConditionValue) {
        self.values.insert(key.into(), value);
    }

    /// Set a bool value.
    pub fn set_bool(&mut self, key: impl Into<String>, value: bool) {
        self.set(key, ConditionValue::Bool(value));
    }

    /// Set an int value.
    pub fn set_int(&mut self, key: impl Into<String>, value: i64) {
        self.set(key, ConditionValue::Int(value));
    }

    /// Set a float value.
    pub fn set_float(&mut self, key: impl Into<String>, value: f64) {
        self.set(key, ConditionValue::Float(value));
    }

    /// Set a string value.
    pub fn set_string(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.set(key, ConditionValue::String(value.into()));
    }

    /// Get a value.
    pub fn get(&self, key: &str) -> Option<&ConditionValue> {
        self.values.get(key)
    }

    /// Get a bool value.
    pub fn get_bool(&self, key: &str) -> bool {
        self.get(key).and_then(|v| v.as_bool()).unwrap_or(false)
    }

    /// Get a float value.
    pub fn get_float(&self, key: &str) -> f64 {
        self.get(key).and_then(|v| v.as_float()).unwrap_or(0.0)
    }

    /// Get an int value.
    pub fn get_int(&self, key: &str) -> i64 {
        self.get(key).and_then(|v| v.as_int()).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Condition
// ---------------------------------------------------------------------------

/// A condition that evaluates against a context.
#[derive(Debug, Clone)]
pub enum Condition {
    /// Check if a boolean value is true.
    IsTrue(String),
    /// Check if a boolean value is false.
    IsFalse(String),
    /// Compare a float value against a threshold.
    GreaterThan { key: String, threshold: f64 },
    /// Compare a float value against a threshold.
    LessThan { key: String, threshold: f64 },
    /// Compare a float value against a threshold.
    GreaterOrEqual { key: String, threshold: f64 },
    /// Compare a float value against a threshold.
    LessOrEqual { key: String, threshold: f64 },
    /// Check equality.
    Equals { key: String, value: ConditionValue },
    /// Check inequality.
    NotEquals { key: String, value: ConditionValue },
    /// Check if value is in a range.
    InRange { key: String, min: f64, max: f64 },
    /// Boolean AND of sub-conditions.
    And(Vec<Condition>),
    /// Boolean OR of sub-conditions.
    Or(Vec<Condition>),
    /// Negate a condition.
    Not(Box<Condition>),
    /// Random chance (0..1).
    RandomChance(f32),
    /// Always true.
    Always,
    /// Always false.
    Never,
    /// Check if a key exists.
    Exists(String),
    /// Custom condition (name + parameters, evaluated externally).
    Custom { name: String, params: Vec<String> },
}

impl Condition {
    /// Evaluate this condition against a context.
    pub fn evaluate(&self, ctx: &DecisionContext, rng: &mut SimpleRng) -> bool {
        match self {
            Condition::IsTrue(key) => ctx.get_bool(key),
            Condition::IsFalse(key) => !ctx.get_bool(key),
            Condition::GreaterThan { key, threshold } => ctx.get_float(key) > *threshold,
            Condition::LessThan { key, threshold } => ctx.get_float(key) < *threshold,
            Condition::GreaterOrEqual { key, threshold } => ctx.get_float(key) >= *threshold,
            Condition::LessOrEqual { key, threshold } => ctx.get_float(key) <= *threshold,
            Condition::Equals { key, value } => {
                ctx.get(key).map_or(false, |v| v == value)
            }
            Condition::NotEquals { key, value } => {
                ctx.get(key).map_or(true, |v| v != value)
            }
            Condition::InRange { key, min, max } => {
                let v = ctx.get_float(key);
                v >= *min && v <= *max
            }
            Condition::And(conditions) => {
                conditions.iter().all(|c| c.evaluate(ctx, rng))
            }
            Condition::Or(conditions) => {
                conditions.iter().any(|c| c.evaluate(ctx, rng))
            }
            Condition::Not(condition) => !condition.evaluate(ctx, rng),
            Condition::RandomChance(chance) => rng.next_f32() < *chance,
            Condition::Always => true,
            Condition::Never => false,
            Condition::Exists(key) => ctx.get(key).is_some(),
            Condition::Custom { .. } => true, // External evaluation
        }
    }
}

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

/// An action that can be selected by the decision tree.
#[derive(Debug, Clone)]
pub struct Action {
    /// Action name/identifier.
    pub name: String,
    /// Priority of this action (for tie-breaking).
    pub priority: i32,
    /// Parameters for the action.
    pub params: HashMap<String, String>,
    /// Cooldown in seconds (if applicable).
    pub cooldown: f32,
    /// Weight for weighted random selection.
    pub weight: f32,
}

impl Action {
    /// Create a new action.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            priority: 0,
            params: HashMap::new(),
            cooldown: 0.0,
            weight: 1.0,
        }
    }

    /// Builder: set a parameter.
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }

    /// Builder: set priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Builder: set weight.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Builder: set cooldown.
    pub fn with_cooldown(mut self, cooldown: f32) -> Self {
        self.cooldown = cooldown;
        self
    }
}

// ---------------------------------------------------------------------------
// DecisionNode
// ---------------------------------------------------------------------------

/// A node in the decision tree.
#[derive(Debug, Clone)]
pub enum DecisionNode {
    /// Binary decision: evaluate condition, branch true (left) or false (right).
    Decision {
        /// Node ID.
        id: NodeId,
        /// Condition to evaluate.
        condition: Condition,
        /// Child for true branch.
        true_child: NodeId,
        /// Child for false branch.
        false_child: NodeId,
        /// Debug label.
        label: String,
    },
    /// Weighted random selection from multiple children.
    WeightedRandom {
        /// Node ID.
        id: NodeId,
        /// Children with weights: (weight, child_id).
        children: Vec<(f32, NodeId)>,
        /// Debug label.
        label: String,
    },
    /// Sequence: evaluate children in order, use first that succeeds.
    Selector {
        /// Node ID.
        id: NodeId,
        /// Condition-action pairs: (condition, child_id).
        options: Vec<(Condition, NodeId)>,
        /// Default child if none match.
        default_child: Option<NodeId>,
        /// Debug label.
        label: String,
    },
    /// Leaf: return an action.
    Leaf {
        /// Node ID.
        id: NodeId,
        /// The action to take.
        action: Action,
        /// Debug label.
        label: String,
    },
}

impl DecisionNode {
    /// Get the node ID.
    pub fn id(&self) -> NodeId {
        match self {
            DecisionNode::Decision { id, .. } => *id,
            DecisionNode::WeightedRandom { id, .. } => *id,
            DecisionNode::Selector { id, .. } => *id,
            DecisionNode::Leaf { id, .. } => *id,
        }
    }

    /// Get the debug label.
    pub fn label(&self) -> &str {
        match self {
            DecisionNode::Decision { label, .. } => label,
            DecisionNode::WeightedRandom { label, .. } => label,
            DecisionNode::Selector { label, .. } => label,
            DecisionNode::Leaf { label, .. } => label,
        }
    }

    /// Get the node type name.
    pub fn type_name(&self) -> &'static str {
        match self {
            DecisionNode::Decision { .. } => "Decision",
            DecisionNode::WeightedRandom { .. } => "WeightedRandom",
            DecisionNode::Selector { .. } => "Selector",
            DecisionNode::Leaf { .. } => "Leaf",
        }
    }
}

// ---------------------------------------------------------------------------
// NodeStats
// ---------------------------------------------------------------------------

/// Runtime statistics for a single node.
#[derive(Debug, Clone, Default)]
pub struct NodeStats {
    /// Number of times this node was evaluated.
    pub evaluation_count: u64,
    /// Number of times the true branch was taken (Decision nodes).
    pub true_count: u64,
    /// Number of times the false branch was taken.
    pub false_count: u64,
    /// Total time spent evaluating this node (microseconds).
    pub total_time_us: u64,
    /// Average evaluation time (microseconds).
    pub avg_time_us: f64,
    /// Number of times this node's action was selected (Leaf nodes).
    pub selection_count: u64,
}

// ---------------------------------------------------------------------------
// DecisionTree
// ---------------------------------------------------------------------------

/// A decision tree with nodes and a root.
#[derive(Debug, Clone)]
pub struct DecisionTree {
    /// Tree identifier.
    pub id: TreeId,
    /// Human-readable name.
    pub name: String,
    /// All nodes in the tree.
    nodes: HashMap<NodeId, DecisionNode>,
    /// Root node ID.
    pub root: NodeId,
    /// Next node ID.
    next_node_id: u32,
    /// Runtime statistics per node.
    stats: HashMap<NodeId, NodeStats>,
    /// Whether statistics collection is enabled.
    pub stats_enabled: bool,
    /// Description.
    pub description: String,
    /// Tags for categorization.
    pub tags: Vec<String>,
}

impl DecisionTree {
    /// Create a new empty decision tree.
    pub fn new(id: TreeId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            nodes: HashMap::new(),
            root: NodeId::NONE,
            next_node_id: 0,
            stats: HashMap::new(),
            stats_enabled: true,
            description: String::new(),
            tags: Vec::new(),
        }
    }

    /// Allocate a new node ID.
    fn alloc_node_id(&mut self) -> NodeId {
        let id = NodeId::new(self.next_node_id);
        self.next_node_id += 1;
        id
    }

    /// Add a decision node.
    pub fn add_decision(
        &mut self,
        condition: Condition,
        true_child: NodeId,
        false_child: NodeId,
        label: impl Into<String>,
    ) -> NodeId {
        let id = self.alloc_node_id();
        self.nodes.insert(id, DecisionNode::Decision {
            id,
            condition,
            true_child,
            false_child,
            label: label.into(),
        });
        if self.root == NodeId::NONE {
            self.root = id;
        }
        id
    }

    /// Add a leaf (action) node.
    pub fn add_leaf(&mut self, action: Action, label: impl Into<String>) -> NodeId {
        let id = self.alloc_node_id();
        self.nodes.insert(id, DecisionNode::Leaf {
            id,
            action,
            label: label.into(),
        });
        if self.root == NodeId::NONE {
            self.root = id;
        }
        id
    }

    /// Add a weighted random node.
    pub fn add_weighted_random(
        &mut self,
        children: Vec<(f32, NodeId)>,
        label: impl Into<String>,
    ) -> NodeId {
        let id = self.alloc_node_id();
        self.nodes.insert(id, DecisionNode::WeightedRandom {
            id,
            children,
            label: label.into(),
        });
        if self.root == NodeId::NONE {
            self.root = id;
        }
        id
    }

    /// Add a selector node.
    pub fn add_selector(
        &mut self,
        options: Vec<(Condition, NodeId)>,
        default_child: Option<NodeId>,
        label: impl Into<String>,
    ) -> NodeId {
        let id = self.alloc_node_id();
        self.nodes.insert(id, DecisionNode::Selector {
            id,
            options,
            default_child,
            label: label.into(),
        });
        if self.root == NodeId::NONE {
            self.root = id;
        }
        id
    }

    /// Set the root node.
    pub fn set_root(&mut self, root: NodeId) {
        self.root = root;
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: NodeId) -> Option<&DecisionNode> {
        self.nodes.get(&id)
    }

    /// Get the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get tree depth.
    pub fn depth(&self) -> usize {
        self.compute_depth(self.root, 0)
    }

    fn compute_depth(&self, node_id: NodeId, current: usize) -> usize {
        if current >= MAX_TREE_DEPTH {
            return current;
        }
        match self.nodes.get(&node_id) {
            Some(DecisionNode::Decision { true_child, false_child, .. }) => {
                let true_depth = self.compute_depth(*true_child, current + 1);
                let false_depth = self.compute_depth(*false_child, current + 1);
                true_depth.max(false_depth)
            }
            Some(DecisionNode::WeightedRandom { children, .. }) => {
                children.iter()
                    .map(|(_, child)| self.compute_depth(*child, current + 1))
                    .max()
                    .unwrap_or(current)
            }
            Some(DecisionNode::Selector { options, default_child, .. }) => {
                let mut max_depth = current;
                for (_, child) in options {
                    max_depth = max_depth.max(self.compute_depth(*child, current + 1));
                }
                if let Some(default) = default_child {
                    max_depth = max_depth.max(self.compute_depth(*default, current + 1));
                }
                max_depth
            }
            Some(DecisionNode::Leaf { .. }) => current,
            None => current,
        }
    }

    /// Get statistics for a node.
    pub fn node_stats(&self, id: NodeId) -> Option<&NodeStats> {
        self.stats.get(&id)
    }

    /// Reset all statistics.
    pub fn reset_stats(&mut self) {
        self.stats.clear();
    }

    /// Record a node evaluation.
    fn record_evaluation(&mut self, id: NodeId, took_true: Option<bool>) {
        if !self.stats_enabled {
            return;
        }
        let stats = self.stats.entry(id).or_default();
        stats.evaluation_count += 1;
        if let Some(true_branch) = took_true {
            if true_branch {
                stats.true_count += 1;
            } else {
                stats.false_count += 1;
            }
        }
    }

    /// Record an action selection.
    fn record_selection(&mut self, id: NodeId) {
        if !self.stats_enabled {
            return;
        }
        let stats = self.stats.entry(id).or_default();
        stats.selection_count += 1;
    }

    /// Get all node IDs.
    pub fn all_node_ids(&self) -> Vec<NodeId> {
        self.nodes.keys().copied().collect()
    }
}

// ---------------------------------------------------------------------------
// SimpleRng
// ---------------------------------------------------------------------------

/// Simple pseudo-random number generator (LCG).
#[derive(Debug, Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Create a new RNG with a seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate the next random u64.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    /// Generate a random f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Generate a random index weighted by the given weights.
    pub fn weighted_index(&mut self, weights: &[f32]) -> usize {
        let total: f32 = weights.iter().sum();
        if total <= 0.0 {
            return 0;
        }
        let mut r = self.next_f32() * total;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return i;
            }
        }
        weights.len().saturating_sub(1)
    }
}

// ---------------------------------------------------------------------------
// DecisionTreeEvaluator
// ---------------------------------------------------------------------------

/// Evaluates a decision tree against a context and returns the selected action.
pub struct DecisionTreeEvaluator {
    /// RNG for random decisions.
    pub rng: SimpleRng,
    /// Path taken during last evaluation (node IDs).
    last_path: Vec<NodeId>,
    /// Maximum depth for safety.
    pub max_depth: usize,
}

impl DecisionTreeEvaluator {
    /// Create a new evaluator.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SimpleRng::new(seed),
            last_path: Vec::new(),
            max_depth: MAX_TREE_DEPTH,
        }
    }

    /// Evaluate a decision tree and return the selected action (if any).
    pub fn evaluate(&mut self, tree: &mut DecisionTree, ctx: &DecisionContext) -> Option<Action> {
        self.last_path.clear();
        self.evaluate_node(tree, tree.root, ctx, 0)
    }

    /// Recursively evaluate a node.
    fn evaluate_node(
        &mut self,
        tree: &mut DecisionTree,
        node_id: NodeId,
        ctx: &DecisionContext,
        depth: usize,
    ) -> Option<Action> {
        if depth >= self.max_depth {
            return None;
        }

        self.last_path.push(node_id);

        // We need to clone the node to avoid borrow conflicts with tree.record_*
        let node = tree.nodes.get(&node_id)?.clone();

        match &node {
            DecisionNode::Decision { condition, true_child, false_child, .. } => {
                let result = condition.evaluate(ctx, &mut self.rng);
                tree.record_evaluation(node_id, Some(result));
                let next = if result { *true_child } else { *false_child };
                self.evaluate_node(tree, next, ctx, depth + 1)
            }
            DecisionNode::WeightedRandom { children, .. } => {
                tree.record_evaluation(node_id, None);
                if children.is_empty() {
                    return None;
                }
                let weights: Vec<f32> = children.iter().map(|(w, _)| *w).collect();
                let index = self.rng.weighted_index(&weights);
                let (_, child_id) = children[index];
                self.evaluate_node(tree, child_id, ctx, depth + 1)
            }
            DecisionNode::Selector { options, default_child, .. } => {
                tree.record_evaluation(node_id, None);
                for (condition, child_id) in options {
                    if condition.evaluate(ctx, &mut self.rng) {
                        return self.evaluate_node(tree, *child_id, ctx, depth + 1);
                    }
                }
                if let Some(default) = default_child {
                    self.evaluate_node(tree, *default, ctx, depth + 1)
                } else {
                    None
                }
            }
            DecisionNode::Leaf { action, .. } => {
                tree.record_evaluation(node_id, None);
                tree.record_selection(node_id);
                Some(action.clone())
            }
        }
    }

    /// Get the path taken during the last evaluation.
    pub fn last_path(&self) -> &[NodeId] {
        &self.last_path
    }
}

// ---------------------------------------------------------------------------
// DecisionTreeBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing decision trees.
pub struct DecisionTreeBuilder {
    tree: DecisionTree,
}

impl DecisionTreeBuilder {
    /// Start building a new tree.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            tree: DecisionTree::new(TreeId(0), name),
        }
    }

    /// Set the tree ID.
    pub fn with_id(mut self, id: TreeId) -> Self {
        self.tree.id = id;
        self
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.tree.description = desc.into();
        self
    }

    /// Add a decision node.
    pub fn decision(
        &mut self,
        condition: Condition,
        true_child: NodeId,
        false_child: NodeId,
        label: impl Into<String>,
    ) -> NodeId {
        self.tree.add_decision(condition, true_child, false_child, label)
    }

    /// Add a leaf node.
    pub fn leaf(&mut self, action: Action, label: impl Into<String>) -> NodeId {
        self.tree.add_leaf(action, label)
    }

    /// Add a weighted random node.
    pub fn weighted_random(
        &mut self,
        children: Vec<(f32, NodeId)>,
        label: impl Into<String>,
    ) -> NodeId {
        self.tree.add_weighted_random(children, label)
    }

    /// Add a selector node.
    pub fn selector(
        &mut self,
        options: Vec<(Condition, NodeId)>,
        default: Option<NodeId>,
        label: impl Into<String>,
    ) -> NodeId {
        self.tree.add_selector(options, default, label)
    }

    /// Set the root node.
    pub fn set_root(&mut self, root: NodeId) {
        self.tree.set_root(root);
    }

    /// Build the tree.
    pub fn build(self) -> DecisionTree {
        self.tree
    }
}

// ---------------------------------------------------------------------------
// VisualizationNode
// ---------------------------------------------------------------------------

/// Data for visualizing a decision tree node.
#[derive(Debug, Clone)]
pub struct VisualizationNode {
    /// Node ID.
    pub id: NodeId,
    /// Node type name.
    pub type_name: String,
    /// Debug label.
    pub label: String,
    /// Child node IDs.
    pub children: Vec<NodeId>,
    /// Whether this node was visited in the last evaluation.
    pub was_visited: bool,
    /// Depth in the tree.
    pub depth: u32,
    /// Evaluation count.
    pub eval_count: u64,
}

/// Generate visualization data for a decision tree.
pub fn generate_visualization(tree: &DecisionTree, last_path: &[NodeId]) -> Vec<VisualizationNode> {
    let visited_set: std::collections::HashSet<NodeId> = last_path.iter().copied().collect();
    let mut result = Vec::new();

    fn visit(
        tree: &DecisionTree,
        node_id: NodeId,
        depth: u32,
        visited: &std::collections::HashSet<NodeId>,
        result: &mut Vec<VisualizationNode>,
    ) {
        if let Some(node) = tree.get_node(node_id) {
            let children = match node {
                DecisionNode::Decision { true_child, false_child, .. } => {
                    vec![*true_child, *false_child]
                }
                DecisionNode::WeightedRandom { children, .. } => {
                    children.iter().map(|(_, id)| *id).collect()
                }
                DecisionNode::Selector { options, default_child, .. } => {
                    let mut c: Vec<_> = options.iter().map(|(_, id)| *id).collect();
                    if let Some(d) = default_child {
                        c.push(*d);
                    }
                    c
                }
                DecisionNode::Leaf { .. } => vec![],
            };

            result.push(VisualizationNode {
                id: node_id,
                type_name: node.type_name().to_string(),
                label: node.label().to_string(),
                children: children.clone(),
                was_visited: visited.contains(&node_id),
                depth,
                eval_count: tree.node_stats(node_id).map_or(0, |s| s.evaluation_count),
            });

            for child in &children {
                visit(tree, *child, depth + 1, visited, result);
            }
        }
    }

    visit(tree, tree.root, 0, &visited_set, &mut result);
    result
}

// ---------------------------------------------------------------------------
// DecisionTreeComponent (ECS)
// ---------------------------------------------------------------------------

/// ECS component for entities using decision trees.
#[derive(Debug, Clone)]
pub struct DecisionTreeComponent {
    /// Tree ID.
    pub tree_id: TreeId,
    /// Last selected action name.
    pub last_action: Option<String>,
    /// Cooldown timers per action.
    pub cooldowns: HashMap<String, f32>,
    /// Whether the tree should be re-evaluated this frame.
    pub needs_evaluation: bool,
    /// Evaluation interval (seconds between evaluations).
    pub eval_interval: f32,
    /// Time since last evaluation.
    pub time_since_eval: f32,
}

impl DecisionTreeComponent {
    /// Create a new component.
    pub fn new(tree_id: TreeId) -> Self {
        Self {
            tree_id,
            last_action: None,
            cooldowns: HashMap::new(),
            needs_evaluation: true,
            eval_interval: 0.5,
            time_since_eval: 0.0,
        }
    }

    /// Update cooldowns and check if re-evaluation is needed.
    pub fn update(&mut self, dt: f32) {
        self.time_since_eval += dt;
        if self.time_since_eval >= self.eval_interval {
            self.needs_evaluation = true;
            self.time_since_eval = 0.0;
        }
        for cooldown in self.cooldowns.values_mut() {
            *cooldown = (*cooldown - dt).max(0.0);
        }
        self.cooldowns.retain(|_, v| *v > 0.0);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_decision_tree() {
        let mut builder = DecisionTreeBuilder::new("test_tree");

        let attack = builder.leaf(Action::new("attack"), "Attack");
        let flee = builder.leaf(Action::new("flee"), "Flee");
        let root = builder.decision(
            Condition::GreaterThan { key: "health".to_string(), threshold: 30.0 },
            attack,
            flee,
            "Health Check",
        );
        builder.set_root(root);

        let mut tree = builder.build();
        let mut evaluator = DecisionTreeEvaluator::new(42);

        // Health > 30 -> attack
        let mut ctx = DecisionContext::new();
        ctx.set_float("health", 50.0);
        let action = evaluator.evaluate(&mut tree, &ctx);
        assert_eq!(action.as_ref().map(|a| a.name.as_str()), Some("attack"));

        // Health <= 30 -> flee
        ctx.set_float("health", 20.0);
        let action = evaluator.evaluate(&mut tree, &ctx);
        assert_eq!(action.as_ref().map(|a| a.name.as_str()), Some("flee"));
    }

    #[test]
    fn test_weighted_random() {
        let mut builder = DecisionTreeBuilder::new("random_tree");
        let a = builder.leaf(Action::new("A"), "A");
        let b = builder.leaf(Action::new("B"), "B");
        let root = builder.weighted_random(vec![(0.5, a), (0.5, b)], "Random");
        builder.set_root(root);

        let mut tree = builder.build();
        let mut evaluator = DecisionTreeEvaluator::new(42);
        let ctx = DecisionContext::new();

        let action = evaluator.evaluate(&mut tree, &ctx);
        assert!(action.is_some());
    }

    #[test]
    fn test_selector_node() {
        let mut builder = DecisionTreeBuilder::new("selector_tree");
        let heal = builder.leaf(Action::new("heal"), "Heal");
        let attack = builder.leaf(Action::new("attack"), "Attack");
        let idle = builder.leaf(Action::new("idle"), "Idle");

        let root = builder.selector(
            vec![
                (Condition::LessThan { key: "health".to_string(), threshold: 30.0 }, heal),
                (Condition::IsTrue("enemy_visible".to_string()), attack),
            ],
            Some(idle),
            "Behavior Selector",
        );
        builder.set_root(root);

        let mut tree = builder.build();
        let mut evaluator = DecisionTreeEvaluator::new(42);

        let mut ctx = DecisionContext::new();
        ctx.set_float("health", 20.0);
        let action = evaluator.evaluate(&mut tree, &ctx);
        assert_eq!(action.as_ref().map(|a| a.name.as_str()), Some("heal"));
    }

    #[test]
    fn test_condition_evaluation() {
        let mut rng = SimpleRng::new(42);
        let ctx = {
            let mut c = DecisionContext::new();
            c.set_bool("is_armed", true);
            c.set_float("distance", 15.0);
            c
        };

        assert!(Condition::IsTrue("is_armed".to_string()).evaluate(&ctx, &mut rng));
        assert!(!Condition::IsFalse("is_armed".to_string()).evaluate(&ctx, &mut rng));
        assert!(Condition::InRange {
            key: "distance".to_string(),
            min: 10.0,
            max: 20.0,
        }.evaluate(&ctx, &mut rng));
    }

    #[test]
    fn test_tree_depth() {
        let mut builder = DecisionTreeBuilder::new("depth_test");
        let a = builder.leaf(Action::new("a"), "a");
        let b = builder.leaf(Action::new("b"), "b");
        let root = builder.decision(Condition::Always, a, b, "root");
        builder.set_root(root);
        let tree = builder.build();
        assert_eq!(tree.depth(), 1);
    }
}
