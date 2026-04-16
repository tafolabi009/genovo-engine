//! Behavior tree system.
//!
//! Provides a composable behavior tree framework for AI decision-making.
//! Trees are built from composite nodes (Selector, Sequence, Parallel,
//! RandomSelector, WeightedSelector), decorator nodes (Inverter, Repeater,
//! RepeatUntilFail, Timeout, Cooldown, Condition), and leaf nodes (Action,
//! Condition, Wait, Log). A shared [`Blackboard`] provides data exchange
//! between nodes, with support for scoped/layered data.

use std::any::Any;
use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use genovo_core::EngineResult;

// ---------------------------------------------------------------------------
// NodeStatus
// ---------------------------------------------------------------------------

/// Result of ticking a behavior tree node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeStatus {
    /// The node is still executing (will be resumed next tick).
    Running,
    /// The node completed successfully.
    Success,
    /// The node failed.
    Failure,
}

impl NodeStatus {
    /// Returns `true` if the status is `Success`.
    pub fn is_success(self) -> bool {
        self == NodeStatus::Success
    }

    /// Returns `true` if the status is `Failure`.
    pub fn is_failure(self) -> bool {
        self == NodeStatus::Failure
    }

    /// Returns `true` if the status is `Running`.
    pub fn is_running(self) -> bool {
        self == NodeStatus::Running
    }

    /// Returns the inverted status (Success <-> Failure, Running unchanged).
    pub fn invert(self) -> NodeStatus {
        match self {
            NodeStatus::Success => NodeStatus::Failure,
            NodeStatus::Failure => NodeStatus::Success,
            NodeStatus::Running => NodeStatus::Running,
        }
    }
}

// ---------------------------------------------------------------------------
// Blackboard
// ---------------------------------------------------------------------------

/// Key-value data store shared among behavior tree nodes.
///
/// The blackboard supports type-erased storage via `Box<dyn Any>`, as well as
/// convenience accessors for common types. Layered blackboards allow entity-
/// level data to shadow global-level defaults.
#[derive(Default)]
pub struct Blackboard {
    /// Type-erased key-value storage.
    data: HashMap<String, Box<dyn Any + Send + Sync>>,
    /// Optional parent blackboard for layered/scoped lookup.
    parent: Option<Box<Blackboard>>,
}

impl Blackboard {
    /// Creates a new empty blackboard.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            parent: None,
        }
    }

    /// Creates a new blackboard with the given parent for scoped lookups.
    ///
    /// Values set on this blackboard shadow the parent. Gets fall through
    /// to the parent if the key is not found locally.
    pub fn with_parent(parent: Blackboard) -> Self {
        Self {
            data: HashMap::new(),
            parent: Some(Box::new(parent)),
        }
    }

    /// Sets a typed value on the blackboard, overwriting any previous value.
    pub fn set<T: Any + Send + Sync>(&mut self, key: &str, value: T) {
        self.data.insert(key.to_string(), Box::new(value));
    }

    /// Gets a typed reference to a value, checking the local scope first, then
    /// falling through to the parent.
    pub fn get<T: Any + Send + Sync>(&self, key: &str) -> Option<&T> {
        if let Some(val) = self.data.get(key) {
            val.downcast_ref::<T>()
        } else if let Some(ref parent) = self.parent {
            parent.get::<T>(key)
        } else {
            None
        }
    }

    /// Gets a mutable typed reference (local scope only; does not mutate parent).
    pub fn get_mut<T: Any + Send + Sync>(&mut self, key: &str) -> Option<&mut T> {
        self.data.get_mut(key).and_then(|v| v.downcast_mut::<T>())
    }

    /// Returns `true` if the key exists in the local scope or parent.
    pub fn has(&self, key: &str) -> bool {
        if self.data.contains_key(key) {
            true
        } else if let Some(ref parent) = self.parent {
            parent.has(key)
        } else {
            false
        }
    }

    /// Removes a value from the local scope.
    pub fn remove(&mut self, key: &str) -> bool {
        self.data.remove(key).is_some()
    }

    /// Clears all local entries.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Returns the number of local entries.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the local scope is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    // -- Convenience accessors for common types --

    /// Sets a boolean value.
    pub fn set_bool(&mut self, key: &str, value: bool) {
        self.set(key, value);
    }

    /// Gets a boolean value.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get::<bool>(key).copied()
    }

    /// Sets an i64 value.
    pub fn set_int(&mut self, key: &str, value: i64) {
        self.set(key, value);
    }

    /// Gets an i64 value.
    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.get::<i64>(key).copied()
    }

    /// Sets an f64 value.
    pub fn set_float(&mut self, key: &str, value: f64) {
        self.set(key, value);
    }

    /// Gets an f64 value.
    pub fn get_float(&self, key: &str) -> Option<f64> {
        self.get::<f64>(key).copied()
    }

    /// Sets a String value.
    pub fn set_string(&mut self, key: &str, value: String) {
        self.set(key, value);
    }

    /// Gets a string reference.
    pub fn get_string(&self, key: &str) -> Option<&String> {
        self.get::<String>(key)
    }

    /// Sets a Vec3 value (as [f32; 3]).
    pub fn set_vec3(&mut self, key: &str, value: [f32; 3]) {
        self.set(key, value);
    }

    /// Gets a Vec3 value.
    pub fn get_vec3(&self, key: &str) -> Option<[f32; 3]> {
        self.get::<[f32; 3]>(key).copied()
    }

    /// Sets an entity id (u32).
    pub fn set_entity(&mut self, key: &str, entity_id: u32) {
        self.set(key, entity_id);
    }

    /// Gets an entity id.
    pub fn get_entity(&self, key: &str) -> Option<u32> {
        self.get::<u32>(key).copied()
    }

    /// Takes the parent blackboard, if any, and returns it.
    pub fn take_parent(&mut self) -> Option<Blackboard> {
        self.parent.take().map(|b| *b)
    }

    /// Returns a reference to the parent blackboard, if any.
    pub fn parent(&self) -> Option<&Blackboard> {
        self.parent.as_deref()
    }

    /// Returns all keys in the local scope.
    pub fn keys(&self) -> Vec<&str> {
        self.data.keys().map(|s| s.as_str()).collect()
    }
}

impl std::fmt::Debug for Blackboard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Blackboard")
            .field("keys", &self.keys())
            .field("has_parent", &self.parent.is_some())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// BehaviorContext
// ---------------------------------------------------------------------------

/// Context passed to behavior tree nodes during a tick.
///
/// Contains the blackboard and timing information.
pub struct BehaviorContext<'a> {
    /// Shared blackboard for inter-node communication.
    pub blackboard: &'a mut Blackboard,
    /// Delta time since the last tick in seconds.
    pub dt: f32,
    /// Total elapsed time since the tree started.
    pub elapsed: f32,
}

// ---------------------------------------------------------------------------
// BehaviorNode trait
// ---------------------------------------------------------------------------

/// Trait implemented by all behavior tree nodes.
///
/// Nodes are ticked each frame (or at a fixed rate). They return a
/// [`NodeStatus`] indicating whether they are still running, succeeded, or
/// failed. The shared [`Blackboard`] provides inter-node communication.
pub trait BehaviorNode: Send + Sync {
    /// Returns a human-readable name for debugging.
    fn name(&self) -> &str;

    /// Ticks this node once, returning the resulting status.
    fn tick(&mut self, dt: f32, blackboard: &mut Blackboard) -> NodeStatus;

    /// Resets this node (and any children) to their initial state.
    ///
    /// Called when the tree is interrupted or restarted.
    fn reset(&mut self);
}

// ---------------------------------------------------------------------------
// Composite Nodes
// ---------------------------------------------------------------------------

/// Selector (OR) node: tries children in order until one succeeds.
///
/// - Returns `Success` as soon as any child succeeds.
/// - Returns `Failure` if all children fail.
/// - Returns `Running` if the current child is still running.
pub struct Selector {
    /// Node name.
    pub name: String,
    /// Child nodes, tried in order.
    pub children: Vec<Box<dyn BehaviorNode>>,
    /// Index of the currently running child.
    current: usize,
}

impl Selector {
    /// Creates a new selector with the given children.
    pub fn new(name: impl Into<String>, children: Vec<Box<dyn BehaviorNode>>) -> Self {
        Self {
            name: name.into(),
            children,
            current: 0,
        }
    }
}

impl BehaviorNode for Selector {
    fn name(&self) -> &str {
        &self.name
    }

    fn tick(&mut self, dt: f32, blackboard: &mut Blackboard) -> NodeStatus {
        while self.current < self.children.len() {
            let status = self.children[self.current].tick(dt, blackboard);
            match status {
                NodeStatus::Success => {
                    self.current = 0;
                    return NodeStatus::Success;
                }
                NodeStatus::Running => return NodeStatus::Running,
                NodeStatus::Failure => {
                    self.current += 1;
                }
            }
        }
        self.current = 0;
        NodeStatus::Failure
    }

    fn reset(&mut self) {
        self.current = 0;
        for child in &mut self.children {
            child.reset();
        }
    }
}

/// Sequence (AND) node: runs children in order until one fails.
///
/// - Returns `Success` if all children succeed.
/// - Returns `Failure` as soon as any child fails.
/// - Returns `Running` if the current child is still running.
pub struct Sequence {
    /// Node name.
    pub name: String,
    /// Child nodes, run in order.
    pub children: Vec<Box<dyn BehaviorNode>>,
    /// Index of the currently running child.
    current: usize,
}

impl Sequence {
    /// Creates a new sequence with the given children.
    pub fn new(name: impl Into<String>, children: Vec<Box<dyn BehaviorNode>>) -> Self {
        Self {
            name: name.into(),
            children,
            current: 0,
        }
    }
}

impl BehaviorNode for Sequence {
    fn name(&self) -> &str {
        &self.name
    }

    fn tick(&mut self, dt: f32, blackboard: &mut Blackboard) -> NodeStatus {
        while self.current < self.children.len() {
            let status = self.children[self.current].tick(dt, blackboard);
            match status {
                NodeStatus::Failure => {
                    self.current = 0;
                    return NodeStatus::Failure;
                }
                NodeStatus::Running => return NodeStatus::Running,
                NodeStatus::Success => {
                    self.current += 1;
                }
            }
        }
        self.current = 0;
        NodeStatus::Success
    }

    fn reset(&mut self) {
        self.current = 0;
        for child in &mut self.children {
            child.reset();
        }
    }
}

/// Parallel execution policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelPolicy {
    /// Succeed when the first child succeeds, fail when all fail.
    RequireOne,
    /// Succeed when all children succeed, fail when any fails.
    RequireAll,
}

/// Parallel node: ticks all children simultaneously.
///
/// The completion policy determines when the parallel node reports success
/// or failure. Children that have already completed are not re-ticked.
pub struct Parallel {
    /// Node name.
    pub name: String,
    /// Child nodes.
    pub children: Vec<Box<dyn BehaviorNode>>,
    /// Success/failure policy.
    pub policy: ParallelPolicy,
    /// Cached statuses from the current execution.
    statuses: Vec<Option<NodeStatus>>,
}

impl Parallel {
    /// Creates a new parallel node.
    pub fn new(
        name: impl Into<String>,
        policy: ParallelPolicy,
        children: Vec<Box<dyn BehaviorNode>>,
    ) -> Self {
        let count = children.len();
        Self {
            name: name.into(),
            children,
            policy,
            statuses: vec![None; count],
        }
    }
}

impl BehaviorNode for Parallel {
    fn name(&self) -> &str {
        &self.name
    }

    fn tick(&mut self, dt: f32, blackboard: &mut Blackboard) -> NodeStatus {
        let mut success_count = 0usize;
        let mut failure_count = 0usize;

        for (i, child) in self.children.iter_mut().enumerate() {
            // Skip children that have already completed.
            if let Some(status) = self.statuses[i] {
                match status {
                    NodeStatus::Success => success_count += 1,
                    NodeStatus::Failure => failure_count += 1,
                    NodeStatus::Running => {
                        // Should not happen; re-tick.
                        let s = child.tick(dt, blackboard);
                        self.statuses[i] = Some(s);
                        match s {
                            NodeStatus::Success => success_count += 1,
                            NodeStatus::Failure => failure_count += 1,
                            NodeStatus::Running => {}
                        }
                    }
                }
                continue;
            }

            let status = child.tick(dt, blackboard);
            self.statuses[i] = Some(status);
            match status {
                NodeStatus::Success => success_count += 1,
                NodeStatus::Failure => failure_count += 1,
                NodeStatus::Running => {}
            }
        }

        let total = self.children.len();
        match self.policy {
            ParallelPolicy::RequireOne => {
                if success_count > 0 {
                    self.reset_statuses();
                    NodeStatus::Success
                } else if failure_count == total {
                    self.reset_statuses();
                    NodeStatus::Failure
                } else {
                    NodeStatus::Running
                }
            }
            ParallelPolicy::RequireAll => {
                if success_count == total {
                    self.reset_statuses();
                    NodeStatus::Success
                } else if failure_count > 0 {
                    self.reset_statuses();
                    NodeStatus::Failure
                } else {
                    NodeStatus::Running
                }
            }
        }
    }

    fn reset(&mut self) {
        self.reset_statuses();
        for child in &mut self.children {
            child.reset();
        }
    }
}

impl Parallel {
    fn reset_statuses(&mut self) {
        for s in &mut self.statuses {
            *s = None;
        }
    }
}

/// RandomSelector: picks a child using a deterministic "random" selection
/// based on a counter (no external RNG crate needed). Each tick, it selects
/// the next child in round-robin fashion from those that haven't been tried yet.
pub struct RandomSelector {
    /// Node name.
    pub name: String,
    /// Child nodes.
    pub children: Vec<Box<dyn BehaviorNode>>,
    /// A simple counter used for pseudo-random selection.
    counter: u32,
    /// Permutation order for this round.
    order: Vec<usize>,
    /// Current index in the permutation.
    current: usize,
    /// Whether the order has been shuffled for the current attempt.
    shuffled: bool,
}

impl RandomSelector {
    /// Creates a new random selector.
    pub fn new(name: impl Into<String>, children: Vec<Box<dyn BehaviorNode>>) -> Self {
        let len = children.len();
        Self {
            name: name.into(),
            children,
            counter: 0,
            order: (0..len).collect(),
            current: 0,
            shuffled: false,
        }
    }

    /// Simple deterministic shuffle using a counter-based permutation.
    fn shuffle_order(&mut self) {
        let n = self.order.len();
        if n <= 1 {
            return;
        }
        self.counter = self.counter.wrapping_add(1);
        let mut seed = self.counter;
        for i in (1..n).rev() {
            // Simple LCG-style step.
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let j = (seed as usize) % (i + 1);
            self.order.swap(i, j);
        }
    }
}

impl BehaviorNode for RandomSelector {
    fn name(&self) -> &str {
        &self.name
    }

    fn tick(&mut self, dt: f32, blackboard: &mut Blackboard) -> NodeStatus {
        if !self.shuffled {
            self.shuffle_order();
            self.shuffled = true;
            self.current = 0;
        }

        while self.current < self.order.len() {
            let child_idx = self.order[self.current];
            let status = self.children[child_idx].tick(dt, blackboard);
            match status {
                NodeStatus::Success => {
                    self.shuffled = false;
                    self.current = 0;
                    return NodeStatus::Success;
                }
                NodeStatus::Running => return NodeStatus::Running,
                NodeStatus::Failure => {
                    self.current += 1;
                }
            }
        }
        self.shuffled = false;
        self.current = 0;
        NodeStatus::Failure
    }

    fn reset(&mut self) {
        self.current = 0;
        self.shuffled = false;
        for child in &mut self.children {
            child.reset();
        }
    }
}

/// WeightedSelector: picks a child based on weights.
///
/// Children with higher weights are more likely to be selected first.
/// Uses a deterministic round-robin weighted scheme.
pub struct WeightedSelector {
    /// Node name.
    pub name: String,
    /// Child nodes with their associated weights.
    pub children: Vec<(f32, Box<dyn BehaviorNode>)>,
    /// Sorted order by weight (descending), indices into children.
    sorted_order: Vec<usize>,
    /// Current position in sorted order.
    current: usize,
    /// Counter for round-robin variation.
    counter: u32,
    /// Whether the order has been computed for this attempt.
    initialized: bool,
}

impl WeightedSelector {
    /// Creates a new weighted selector. Each child has an associated weight.
    pub fn new(
        name: impl Into<String>,
        children: Vec<(f32, Box<dyn BehaviorNode>)>,
    ) -> Self {
        let len = children.len();
        Self {
            name: name.into(),
            children,
            sorted_order: (0..len).collect(),
            current: 0,
            counter: 0,
            initialized: false,
        }
    }

    fn compute_order(&mut self) {
        self.counter = self.counter.wrapping_add(1);

        // Sort children by weight descending. For ties, use the counter to vary.
        let children = &self.children;
        let counter = self.counter;
        self.sorted_order.sort_by(|&a, &b| {
            let wa = children[a].0;
            let wb = children[b].0;
            wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal).then_with(|| {
                // Break ties using counter-based variation.
                let ha = a as u32 ^ counter;
                let hb = b as u32 ^ counter;
                ha.cmp(&hb)
            })
        });
    }
}

impl BehaviorNode for WeightedSelector {
    fn name(&self) -> &str {
        &self.name
    }

    fn tick(&mut self, dt: f32, blackboard: &mut Blackboard) -> NodeStatus {
        if !self.initialized {
            self.compute_order();
            self.initialized = true;
            self.current = 0;
        }

        while self.current < self.sorted_order.len() {
            let child_idx = self.sorted_order[self.current];
            let status = self.children[child_idx].1.tick(dt, blackboard);
            match status {
                NodeStatus::Success => {
                    self.initialized = false;
                    self.current = 0;
                    return NodeStatus::Success;
                }
                NodeStatus::Running => return NodeStatus::Running,
                NodeStatus::Failure => {
                    self.current += 1;
                }
            }
        }
        self.initialized = false;
        self.current = 0;
        NodeStatus::Failure
    }

    fn reset(&mut self) {
        self.current = 0;
        self.initialized = false;
        for (_, child) in &mut self.children {
            child.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// Decorator Nodes
// ---------------------------------------------------------------------------

/// Inverter decorator: flips Success <-> Failure, passes Running through.
pub struct Inverter {
    /// Node name.
    pub name: String,
    /// The child node to invert.
    pub child: Box<dyn BehaviorNode>,
}

impl Inverter {
    /// Creates a new inverter wrapping the given child.
    pub fn new(name: impl Into<String>, child: Box<dyn BehaviorNode>) -> Self {
        Self {
            name: name.into(),
            child,
        }
    }
}

impl BehaviorNode for Inverter {
    fn name(&self) -> &str {
        &self.name
    }

    fn tick(&mut self, dt: f32, blackboard: &mut Blackboard) -> NodeStatus {
        self.child.tick(dt, blackboard).invert()
    }

    fn reset(&mut self) {
        self.child.reset();
    }
}

/// Repeater decorator: re-runs its child a specified number of times.
///
/// When `repeat_count` is 0, repeats forever (until externally reset).
pub struct Repeater {
    /// Node name.
    pub name: String,
    /// The child node to repeat.
    pub child: Box<dyn BehaviorNode>,
    /// Number of times to repeat (0 = infinite).
    pub repeat_count: u32,
    /// Current iteration.
    current_iteration: u32,
    /// Whether to abort on child failure (default: false = keep repeating).
    pub abort_on_failure: bool,
}

impl Repeater {
    /// Creates a new repeater. `count` of 0 means repeat forever.
    pub fn new(name: impl Into<String>, child: Box<dyn BehaviorNode>, count: u32) -> Self {
        Self {
            name: name.into(),
            child,
            repeat_count: count,
            current_iteration: 0,
            abort_on_failure: false,
        }
    }

    /// If set, the repeater will stop and return Failure when the child fails.
    pub fn with_abort_on_failure(mut self, abort: bool) -> Self {
        self.abort_on_failure = abort;
        self
    }
}

impl BehaviorNode for Repeater {
    fn name(&self) -> &str {
        &self.name
    }

    fn tick(&mut self, dt: f32, blackboard: &mut Blackboard) -> NodeStatus {
        let status = self.child.tick(dt, blackboard);
        match status {
            NodeStatus::Running => NodeStatus::Running,
            NodeStatus::Failure if self.abort_on_failure => {
                self.current_iteration = 0;
                NodeStatus::Failure
            }
            NodeStatus::Success | NodeStatus::Failure => {
                self.current_iteration += 1;
                if self.repeat_count > 0 && self.current_iteration >= self.repeat_count {
                    let final_status = status;
                    self.current_iteration = 0;
                    final_status
                } else {
                    self.child.reset();
                    NodeStatus::Running
                }
            }
        }
    }

    fn reset(&mut self) {
        self.current_iteration = 0;
        self.child.reset();
    }
}

/// RepeatUntilFail decorator: keeps running the child until it fails, then
/// returns Success.
pub struct RepeatUntilFail {
    /// Node name.
    pub name: String,
    /// The child node.
    pub child: Box<dyn BehaviorNode>,
}

impl RepeatUntilFail {
    /// Creates a new repeat-until-fail decorator.
    pub fn new(name: impl Into<String>, child: Box<dyn BehaviorNode>) -> Self {
        Self {
            name: name.into(),
            child,
        }
    }
}

impl BehaviorNode for RepeatUntilFail {
    fn name(&self) -> &str {
        &self.name
    }

    fn tick(&mut self, dt: f32, blackboard: &mut Blackboard) -> NodeStatus {
        let status = self.child.tick(dt, blackboard);
        match status {
            NodeStatus::Failure => {
                self.child.reset();
                NodeStatus::Success
            }
            NodeStatus::Running => NodeStatus::Running,
            NodeStatus::Success => {
                self.child.reset();
                NodeStatus::Running
            }
        }
    }

    fn reset(&mut self) {
        self.child.reset();
    }
}

/// Timeout decorator: fails if the child takes longer than a specified
/// duration.
pub struct Timeout {
    /// Node name.
    pub name: String,
    /// The child node.
    pub child: Box<dyn BehaviorNode>,
    /// Maximum duration before timeout.
    pub duration: Duration,
    /// Elapsed time since the child started running.
    elapsed: f32,
    /// Whether the child is currently running.
    running: bool,
}

impl Timeout {
    /// Creates a new timeout decorator.
    pub fn new(name: impl Into<String>, child: Box<dyn BehaviorNode>, duration: Duration) -> Self {
        Self {
            name: name.into(),
            child,
            duration,
            elapsed: 0.0,
            running: false,
        }
    }
}

impl BehaviorNode for Timeout {
    fn name(&self) -> &str {
        &self.name
    }

    fn tick(&mut self, dt: f32, blackboard: &mut Blackboard) -> NodeStatus {
        if !self.running {
            self.elapsed = 0.0;
            self.running = true;
        }

        self.elapsed += dt;
        if self.elapsed >= self.duration.as_secs_f32() {
            self.running = false;
            self.child.reset();
            return NodeStatus::Failure;
        }

        let status = self.child.tick(dt, blackboard);
        if status != NodeStatus::Running {
            self.running = false;
        }
        status
    }

    fn reset(&mut self) {
        self.elapsed = 0.0;
        self.running = false;
        self.child.reset();
    }
}

/// Cooldown decorator: after the child completes (success or failure),
/// prevents re-execution for a specified duration.
///
/// During the cooldown period, the node returns `Failure`.
pub struct Cooldown {
    /// Node name.
    pub name: String,
    /// The child node.
    pub child: Box<dyn BehaviorNode>,
    /// Cooldown duration in seconds.
    pub cooldown_secs: f32,
    /// Time remaining in the current cooldown.
    remaining: f32,
    /// Whether the child is currently executing.
    child_running: bool,
}

impl Cooldown {
    /// Creates a new cooldown decorator.
    pub fn new(name: impl Into<String>, child: Box<dyn BehaviorNode>, cooldown_secs: f32) -> Self {
        Self {
            name: name.into(),
            child,
            cooldown_secs,
            remaining: 0.0,
            child_running: false,
        }
    }
}

impl BehaviorNode for Cooldown {
    fn name(&self) -> &str {
        &self.name
    }

    fn tick(&mut self, dt: f32, blackboard: &mut Blackboard) -> NodeStatus {
        if self.remaining > 0.0 && !self.child_running {
            self.remaining -= dt;
            if self.remaining > 0.0 {
                return NodeStatus::Failure;
            }
            self.remaining = 0.0;
        }

        self.child_running = true;
        let status = self.child.tick(dt, blackboard);

        if status != NodeStatus::Running {
            self.child_running = false;
            self.remaining = self.cooldown_secs;
            self.child.reset();
        }

        status
    }

    fn reset(&mut self) {
        self.remaining = 0.0;
        self.child_running = false;
        self.child.reset();
    }
}

/// Condition decorator: only runs the child if a predicate returns true.
/// If the condition is false, the node returns `Failure` immediately.
pub struct ConditionDecorator {
    /// Node name.
    pub name: String,
    /// The child node.
    pub child: Box<dyn BehaviorNode>,
    /// The condition predicate.
    condition: Box<dyn Fn(&Blackboard) -> bool + Send + Sync>,
}

impl ConditionDecorator {
    /// Creates a new condition decorator.
    pub fn new(
        name: impl Into<String>,
        child: Box<dyn BehaviorNode>,
        condition: impl Fn(&Blackboard) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            child,
            condition: Box::new(condition),
        }
    }
}

impl BehaviorNode for ConditionDecorator {
    fn name(&self) -> &str {
        &self.name
    }

    fn tick(&mut self, dt: f32, blackboard: &mut Blackboard) -> NodeStatus {
        if (self.condition)(blackboard) {
            self.child.tick(dt, blackboard)
        } else {
            self.child.reset();
            NodeStatus::Failure
        }
    }

    fn reset(&mut self) {
        self.child.reset();
    }
}

// ---------------------------------------------------------------------------
// Leaf Nodes
// ---------------------------------------------------------------------------

/// A leaf node that executes a game-specific action via a closure.
pub struct ActionNode {
    /// Node name.
    pub name: String,
    /// The action callback.
    action: Box<dyn FnMut(f32, &mut Blackboard) -> NodeStatus + Send + Sync>,
}

impl ActionNode {
    /// Creates a new action node with the given callback.
    pub fn new(
        name: impl Into<String>,
        action: impl FnMut(f32, &mut Blackboard) -> NodeStatus + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            action: Box::new(action),
        }
    }
}

impl BehaviorNode for ActionNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn tick(&mut self, dt: f32, blackboard: &mut Blackboard) -> NodeStatus {
        (self.action)(dt, blackboard)
    }

    fn reset(&mut self) {
        // Actions manage state through their closure captures / blackboard.
    }
}

/// A leaf node that checks a condition (returns Success or Failure, never Running).
pub struct ConditionNode {
    /// Node name.
    pub name: String,
    /// The condition predicate.
    condition: Box<dyn Fn(&Blackboard) -> bool + Send + Sync>,
}

impl ConditionNode {
    /// Creates a new condition node.
    pub fn new(
        name: impl Into<String>,
        condition: impl Fn(&Blackboard) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            condition: Box::new(condition),
        }
    }
}

impl BehaviorNode for ConditionNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn tick(&mut self, _dt: f32, blackboard: &mut Blackboard) -> NodeStatus {
        if (self.condition)(blackboard) {
            NodeStatus::Success
        } else {
            NodeStatus::Failure
        }
    }

    fn reset(&mut self) {
        // Conditions are stateless.
    }
}

/// A leaf node that waits for a specified duration then returns `Success`.
///
/// Returns `Running` while the wait is in progress.
pub struct WaitNode {
    /// Node name.
    pub name: String,
    /// Duration to wait.
    pub duration: f32,
    /// Elapsed time.
    elapsed: f32,
}

impl WaitNode {
    /// Creates a new wait node.
    pub fn new(name: impl Into<String>, duration: f32) -> Self {
        Self {
            name: name.into(),
            duration,
            elapsed: 0.0,
        }
    }
}

impl BehaviorNode for WaitNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn tick(&mut self, dt: f32, _blackboard: &mut Blackboard) -> NodeStatus {
        self.elapsed += dt;
        if self.elapsed >= self.duration {
            self.elapsed = 0.0;
            NodeStatus::Success
        } else {
            NodeStatus::Running
        }
    }

    fn reset(&mut self) {
        self.elapsed = 0.0;
    }
}

/// A leaf node that logs a message and returns `Success`.
///
/// Useful for debugging behavior tree execution flow.
pub struct LogNode {
    /// Node name.
    pub name: String,
    /// Message to log.
    pub message: String,
    /// Log level.
    pub level: LogLevel,
}

/// Log level for [`LogNode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Trace level.
    Trace,
    /// Debug level.
    Debug,
    /// Info level.
    Info,
    /// Warning level.
    Warn,
}

impl LogNode {
    /// Creates a new log node.
    pub fn new(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            message: message.into(),
            level: LogLevel::Debug,
        }
    }

    /// Sets the log level.
    pub fn with_level(mut self, level: LogLevel) -> Self {
        self.level = level;
        self
    }
}

impl BehaviorNode for LogNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn tick(&mut self, _dt: f32, _blackboard: &mut Blackboard) -> NodeStatus {
        match self.level {
            LogLevel::Trace => log::trace!("[BT:{}] {}", self.name, self.message),
            LogLevel::Debug => log::debug!("[BT:{}] {}", self.name, self.message),
            LogLevel::Info => log::info!("[BT:{}] {}", self.name, self.message),
            LogLevel::Warn => log::warn!("[BT:{}] {}", self.name, self.message),
        }
        NodeStatus::Success
    }

    fn reset(&mut self) {
        // Logging is stateless.
    }
}

// ---------------------------------------------------------------------------
// BehaviorTree
// ---------------------------------------------------------------------------

/// A complete behavior tree with a root node and associated blackboard.
pub struct BehaviorTree {
    /// Human-readable name for this tree.
    pub name: String,
    /// The root node of the tree.
    root: Box<dyn BehaviorNode>,
    /// Shared data store for all nodes.
    pub blackboard: Blackboard,
    /// Status from the most recent tick.
    last_status: NodeStatus,
    /// Total elapsed time.
    elapsed: f32,
    /// Number of ticks performed.
    tick_count: u64,
}

impl BehaviorTree {
    /// Creates a new behavior tree with the given root node.
    pub fn new(name: impl Into<String>, root: Box<dyn BehaviorNode>) -> Self {
        Self {
            name: name.into(),
            root,
            blackboard: Blackboard::new(),
            last_status: NodeStatus::Failure,
            elapsed: 0.0,
            tick_count: 0,
        }
    }

    /// Creates a new behavior tree with a pre-populated blackboard.
    pub fn with_blackboard(
        name: impl Into<String>,
        root: Box<dyn BehaviorNode>,
        blackboard: Blackboard,
    ) -> Self {
        Self {
            name: name.into(),
            root,
            blackboard,
            last_status: NodeStatus::Failure,
            elapsed: 0.0,
            tick_count: 0,
        }
    }

    /// Ticks the tree once, advancing all active nodes.
    pub fn tick(&mut self, dt: f32) -> NodeStatus {
        profiling::scope!("BehaviorTree::tick");
        self.elapsed += dt;
        self.tick_count += 1;
        self.last_status = self.root.tick(dt, &mut self.blackboard);
        self.last_status
    }

    /// Resets the entire tree to its initial state.
    pub fn reset(&mut self) {
        self.root.reset();
        self.last_status = NodeStatus::Failure;
        self.elapsed = 0.0;
        self.tick_count = 0;
    }

    /// Returns the status from the most recent tick.
    pub fn last_status(&self) -> NodeStatus {
        self.last_status
    }

    /// Returns the total elapsed time.
    pub fn elapsed(&self) -> f32 {
        self.elapsed
    }

    /// Returns the number of ticks performed.
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }
}

// ---------------------------------------------------------------------------
// BehaviorTreeBuilder
// ---------------------------------------------------------------------------

/// Builder API for constructing behavior trees in code using a fluent,
/// nested syntax.
///
/// # Example
/// ```rust,ignore
/// let tree = BehaviorTreeBuilder::new("combat_ai")
///     .selector("root")
///         .sequence("attack")
///             .condition("enemy_visible", |bb| bb.get_bool("enemy_visible").unwrap_or(false))
///             .action("do_attack", |_dt, bb| {
///                 bb.set_bool("attacking", true);
///                 NodeStatus::Success
///             })
///         .end()
///         .action("patrol", |_dt, _bb| NodeStatus::Running)
///     .end()
///     .build();
/// ```
pub struct BehaviorTreeBuilder {
    /// Tree name.
    name: String,
    /// Stack of in-progress composite nodes.
    stack: Vec<BuilderFrame>,
}

/// A frame on the builder stack representing an in-progress composite node.
enum BuilderFrame {
    Selector {
        name: String,
        children: Vec<Box<dyn BehaviorNode>>,
    },
    Sequence {
        name: String,
        children: Vec<Box<dyn BehaviorNode>>,
    },
    Parallel {
        name: String,
        policy: ParallelPolicy,
        children: Vec<Box<dyn BehaviorNode>>,
    },
}

impl BuilderFrame {
    fn add_child(&mut self, child: Box<dyn BehaviorNode>) {
        match self {
            BuilderFrame::Selector { children, .. } => children.push(child),
            BuilderFrame::Sequence { children, .. } => children.push(child),
            BuilderFrame::Parallel { children, .. } => children.push(child),
        }
    }

    fn into_node(self) -> Box<dyn BehaviorNode> {
        match self {
            BuilderFrame::Selector { name, children } => {
                Box::new(Selector::new(name, children))
            }
            BuilderFrame::Sequence { name, children } => {
                Box::new(Sequence::new(name, children))
            }
            BuilderFrame::Parallel {
                name,
                policy,
                children,
            } => Box::new(Parallel::new(name, policy, children)),
        }
    }
}

impl BehaviorTreeBuilder {
    /// Creates a new builder with the given tree name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            stack: Vec::new(),
        }
    }

    /// Pushes a Selector composite node.
    pub fn selector(mut self, name: impl Into<String>) -> Self {
        self.stack.push(BuilderFrame::Selector {
            name: name.into(),
            children: Vec::new(),
        });
        self
    }

    /// Pushes a Sequence composite node.
    pub fn sequence(mut self, name: impl Into<String>) -> Self {
        self.stack.push(BuilderFrame::Sequence {
            name: name.into(),
            children: Vec::new(),
        });
        self
    }

    /// Pushes a Parallel composite node.
    pub fn parallel(
        mut self,
        name: impl Into<String>,
        policy: ParallelPolicy,
    ) -> Self {
        self.stack.push(BuilderFrame::Parallel {
            name: name.into(),
            policy,
            children: Vec::new(),
        });
        self
    }

    /// Adds an action leaf node to the current composite.
    pub fn action(
        mut self,
        name: impl Into<String>,
        action: impl FnMut(f32, &mut Blackboard) -> NodeStatus + Send + Sync + 'static,
    ) -> Self {
        let node = Box::new(ActionNode::new(name, action));
        if let Some(frame) = self.stack.last_mut() {
            frame.add_child(node);
        }
        self
    }

    /// Adds a condition leaf node to the current composite.
    pub fn condition(
        mut self,
        name: impl Into<String>,
        condition: impl Fn(&Blackboard) -> bool + Send + Sync + 'static,
    ) -> Self {
        let node = Box::new(ConditionNode::new(name, condition));
        if let Some(frame) = self.stack.last_mut() {
            frame.add_child(node);
        }
        self
    }

    /// Adds a wait leaf node to the current composite.
    pub fn wait(mut self, name: impl Into<String>, duration: f32) -> Self {
        let node = Box::new(WaitNode::new(name, duration));
        if let Some(frame) = self.stack.last_mut() {
            frame.add_child(node);
        }
        self
    }

    /// Adds a log leaf node to the current composite.
    pub fn log(mut self, name: impl Into<String>, message: impl Into<String>) -> Self {
        let node = Box::new(LogNode::new(name, message));
        if let Some(frame) = self.stack.last_mut() {
            frame.add_child(node);
        }
        self
    }

    /// Adds an inverter decorator wrapping the most recently added child.
    /// The child must already be on the current composite. This pops the last
    /// child, wraps it, and pushes it back.
    pub fn invert_last(mut self, name: impl Into<String>) -> Self {
        if let Some(frame) = self.stack.last_mut() {
            let child = match frame {
                BuilderFrame::Selector { children, .. }
                | BuilderFrame::Sequence { children, .. }
                | BuilderFrame::Parallel { children, .. } => children.pop(),
            };
            if let Some(child) = child {
                let inverted = Box::new(Inverter::new(name, child));
                frame.add_child(inverted);
            }
        }
        self
    }

    /// Adds a custom node to the current composite.
    pub fn custom(mut self, node: Box<dyn BehaviorNode>) -> Self {
        if let Some(frame) = self.stack.last_mut() {
            frame.add_child(node);
        }
        self
    }

    /// Ends the current composite node and adds it as a child of the parent,
    /// or keeps it as the root if it's the only one.
    pub fn end(mut self) -> Self {
        if let Some(frame) = self.stack.pop() {
            let node = frame.into_node();
            if let Some(parent) = self.stack.last_mut() {
                parent.add_child(node);
            } else {
                // This is the root node. Push a temporary frame to hold it.
                self.stack.push(BuilderFrame::Selector {
                    name: "__root_holder__".into(),
                    children: vec![node],
                });
            }
        }
        self
    }

    /// Builds the behavior tree from the accumulated nodes.
    ///
    /// The last composite on the stack (or its single child if it's the root
    /// holder) becomes the root of the tree.
    pub fn build(mut self) -> BehaviorTree {
        // Collapse remaining stack into a single root.
        while self.stack.len() > 1 {
            let frame = self.stack.pop().unwrap();
            let node = frame.into_node();
            if let Some(parent) = self.stack.last_mut() {
                parent.add_child(node);
            }
        }

        let root = if let Some(frame) = self.stack.pop() {
            match frame {
                BuilderFrame::Selector {
                    name,
                    mut children,
                } if name == "__root_holder__" && children.len() == 1 => {
                    children.pop().unwrap()
                }
                other => other.into_node(),
            }
        } else {
            // Empty tree: use a no-op action.
            Box::new(ActionNode::new("empty", |_, _| NodeStatus::Failure))
        };

        BehaviorTree::new(self.name, root)
    }
}

// ---------------------------------------------------------------------------
// BehaviorTreeAsset
// ---------------------------------------------------------------------------

/// Serializable definition of a behavior tree for loading from asset files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorTreeAsset {
    /// Name of this behavior tree definition.
    pub name: String,
    /// The root node definition.
    pub root: NodeDefinition,
}

/// Serializable definition of a single behavior tree node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDefinition {
    /// The type of node (e.g., `"selector"`, `"sequence"`, `"action"`).
    pub node_type: String,
    /// Human-readable name.
    pub name: String,
    /// Child node definitions (for composite/decorator nodes).
    #[serde(default)]
    pub children: Vec<NodeDefinition>,
    /// Node-specific parameters (e.g., timeout duration, repeat count).
    #[serde(default)]
    pub params: HashMap<String, serde_json::Value>,
}

/// Registry entry for action/condition factories.
pub type NodeFactory = Box<dyn Fn() -> Box<dyn BehaviorNode> + Send + Sync>;

impl BehaviorTreeAsset {
    /// Instantiates this asset into a live [`BehaviorTree`].
    ///
    /// Action and condition nodes are resolved against a registry of
    /// named callbacks.
    pub fn instantiate(
        &self,
        action_registry: &HashMap<String, NodeFactory>,
    ) -> EngineResult<BehaviorTree> {
        let root = self.build_node(&self.root, action_registry)?;
        Ok(BehaviorTree::new(&self.name, root))
    }

    fn build_node(
        &self,
        def: &NodeDefinition,
        registry: &HashMap<String, NodeFactory>,
    ) -> EngineResult<Box<dyn BehaviorNode>> {
        match def.node_type.as_str() {
            "selector" => {
                let children = self.build_children(&def.children, registry)?;
                Ok(Box::new(Selector::new(&def.name, children)))
            }
            "sequence" => {
                let children = self.build_children(&def.children, registry)?;
                Ok(Box::new(Sequence::new(&def.name, children)))
            }
            "parallel" => {
                let children = self.build_children(&def.children, registry)?;
                let policy = match def
                    .params
                    .get("policy")
                    .and_then(|v| v.as_str())
                    .unwrap_or("require_all")
                {
                    "require_one" => ParallelPolicy::RequireOne,
                    _ => ParallelPolicy::RequireAll,
                };
                Ok(Box::new(Parallel::new(&def.name, policy, children)))
            }
            "inverter" => {
                let child = if !def.children.is_empty() {
                    self.build_node(&def.children[0], registry)?
                } else {
                    return Err(genovo_core::EngineError::InvalidArgument(
                        "Inverter requires a child node".into(),
                    ));
                };
                Ok(Box::new(Inverter::new(&def.name, child)))
            }
            "repeater" => {
                let child = if !def.children.is_empty() {
                    self.build_node(&def.children[0], registry)?
                } else {
                    return Err(genovo_core::EngineError::InvalidArgument(
                        "Repeater requires a child node".into(),
                    ));
                };
                let count = def
                    .params
                    .get("count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;
                Ok(Box::new(Repeater::new(&def.name, child, count)))
            }
            "timeout" => {
                let child = if !def.children.is_empty() {
                    self.build_node(&def.children[0], registry)?
                } else {
                    return Err(genovo_core::EngineError::InvalidArgument(
                        "Timeout requires a child node".into(),
                    ));
                };
                let secs = def
                    .params
                    .get("duration")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(5.0);
                Ok(Box::new(Timeout::new(
                    &def.name,
                    child,
                    Duration::from_secs_f64(secs),
                )))
            }
            "wait" => {
                let duration = def
                    .params
                    .get("duration")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0) as f32;
                Ok(Box::new(WaitNode::new(&def.name, duration)))
            }
            "log" => {
                let message = def
                    .params
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                Ok(Box::new(LogNode::new(&def.name, message)))
            }
            "action" | "condition" => {
                // Look up in the registry by name.
                if let Some(factory) = registry.get(&def.name) {
                    Ok(factory())
                } else {
                    Err(genovo_core::EngineError::NotFound(format!(
                        "No registered action/condition named '{}'",
                        def.name
                    )))
                }
            }
            other => Err(genovo_core::EngineError::InvalidArgument(format!(
                "Unknown node type: '{}'",
                other
            ))),
        }
    }

    fn build_children(
        &self,
        defs: &[NodeDefinition],
        registry: &HashMap<String, NodeFactory>,
    ) -> EngineResult<Vec<Box<dyn BehaviorNode>>> {
        defs.iter()
            .map(|d| self.build_node(d, registry))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_blackboard_typed() {
        let mut bb = Blackboard::new();
        bb.set("health", 100.0f64);
        bb.set("name", "enemy".to_string());
        bb.set("alive", true);

        assert_eq!(bb.get::<f64>("health"), Some(&100.0));
        assert_eq!(bb.get::<String>("name"), Some(&"enemy".to_string()));
        assert_eq!(bb.get::<bool>("alive"), Some(&true));
        assert!(bb.has("health"));
        assert!(!bb.has("missing"));
    }

    #[test]
    fn test_blackboard_scoped() {
        let mut global = Blackboard::new();
        global.set("difficulty", 3i64);
        global.set("global_flag", true);

        let mut local = Blackboard::with_parent(global);
        local.set("entity_hp", 50i64);

        // Local lookup.
        assert_eq!(local.get::<i64>("entity_hp"), Some(&50));
        // Falls through to parent.
        assert_eq!(local.get::<i64>("difficulty"), Some(&3));
        assert_eq!(local.get::<bool>("global_flag"), Some(&true));

        // Shadow parent value.
        local.set("difficulty", 5i64);
        assert_eq!(local.get::<i64>("difficulty"), Some(&5));
    }

    #[test]
    fn test_blackboard_convenience() {
        let mut bb = Blackboard::new();
        bb.set_bool("flag", true);
        bb.set_int("count", 42);
        bb.set_float("speed", 3.14);
        bb.set_string("name", "test".to_string());
        bb.set_vec3("pos", [1.0, 2.0, 3.0]);
        bb.set_entity("target", 99);

        assert_eq!(bb.get_bool("flag"), Some(true));
        assert_eq!(bb.get_int("count"), Some(42));
        assert_eq!(bb.get_float("speed"), Some(3.14));
        assert_eq!(bb.get_string("name"), Some(&"test".to_string()));
        assert_eq!(bb.get_vec3("pos"), Some([1.0, 2.0, 3.0]));
        assert_eq!(bb.get_entity("target"), Some(99));
    }

    #[test]
    fn test_selector_success() {
        let mut tree = BehaviorTree::new(
            "test",
            Box::new(Selector::new(
                "sel",
                vec![
                    Box::new(ActionNode::new("fail", |_, _| NodeStatus::Failure)),
                    Box::new(ActionNode::new("succeed", |_, _| NodeStatus::Success)),
                    Box::new(ActionNode::new("never", |_, _| {
                        panic!("should not be reached")
                    })),
                ],
            )),
        );
        assert_eq!(tree.tick(0.016), NodeStatus::Success);
    }

    #[test]
    fn test_selector_all_fail() {
        let mut tree = BehaviorTree::new(
            "test",
            Box::new(Selector::new(
                "sel",
                vec![
                    Box::new(ActionNode::new("f1", |_, _| NodeStatus::Failure)),
                    Box::new(ActionNode::new("f2", |_, _| NodeStatus::Failure)),
                ],
            )),
        );
        assert_eq!(tree.tick(0.016), NodeStatus::Failure);
    }

    #[test]
    fn test_selector_running() {
        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();
        let mut tree = BehaviorTree::new(
            "test",
            Box::new(Selector::new(
                "sel",
                vec![
                    Box::new(ActionNode::new("running", move |_, _| {
                        let n = c.fetch_add(1, Ordering::SeqCst);
                        if n < 2 {
                            NodeStatus::Running
                        } else {
                            NodeStatus::Success
                        }
                    })),
                ],
            )),
        );
        assert_eq!(tree.tick(0.016), NodeStatus::Running);
        assert_eq!(tree.tick(0.016), NodeStatus::Running);
        assert_eq!(tree.tick(0.016), NodeStatus::Success);
    }

    #[test]
    fn test_sequence_success() {
        let mut tree = BehaviorTree::new(
            "test",
            Box::new(Sequence::new(
                "seq",
                vec![
                    Box::new(ActionNode::new("s1", |_, _| NodeStatus::Success)),
                    Box::new(ActionNode::new("s2", |_, _| NodeStatus::Success)),
                    Box::new(ActionNode::new("s3", |_, _| NodeStatus::Success)),
                ],
            )),
        );
        assert_eq!(tree.tick(0.016), NodeStatus::Success);
    }

    #[test]
    fn test_sequence_failure() {
        let mut tree = BehaviorTree::new(
            "test",
            Box::new(Sequence::new(
                "seq",
                vec![
                    Box::new(ActionNode::new("s1", |_, _| NodeStatus::Success)),
                    Box::new(ActionNode::new("f1", |_, _| NodeStatus::Failure)),
                    Box::new(ActionNode::new("never", |_, _| {
                        panic!("should not be reached")
                    })),
                ],
            )),
        );
        assert_eq!(tree.tick(0.016), NodeStatus::Failure);
    }

    #[test]
    fn test_sequence_running() {
        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();
        let mut tree = BehaviorTree::new(
            "test",
            Box::new(Sequence::new(
                "seq",
                vec![
                    Box::new(ActionNode::new("s1", |_, _| NodeStatus::Success)),
                    Box::new(ActionNode::new("running", move |_, _| {
                        let n = c.fetch_add(1, Ordering::SeqCst);
                        if n < 1 {
                            NodeStatus::Running
                        } else {
                            NodeStatus::Success
                        }
                    })),
                ],
            )),
        );
        assert_eq!(tree.tick(0.016), NodeStatus::Running);
        assert_eq!(tree.tick(0.016), NodeStatus::Success);
    }

    #[test]
    fn test_parallel_require_all() {
        let mut tree = BehaviorTree::new(
            "test",
            Box::new(Parallel::new(
                "par",
                ParallelPolicy::RequireAll,
                vec![
                    Box::new(ActionNode::new("s1", |_, _| NodeStatus::Success)),
                    Box::new(ActionNode::new("s2", |_, _| NodeStatus::Success)),
                ],
            )),
        );
        assert_eq!(tree.tick(0.016), NodeStatus::Success);
    }

    #[test]
    fn test_parallel_require_all_failure() {
        let mut tree = BehaviorTree::new(
            "test",
            Box::new(Parallel::new(
                "par",
                ParallelPolicy::RequireAll,
                vec![
                    Box::new(ActionNode::new("s1", |_, _| NodeStatus::Success)),
                    Box::new(ActionNode::new("f1", |_, _| NodeStatus::Failure)),
                ],
            )),
        );
        assert_eq!(tree.tick(0.016), NodeStatus::Failure);
    }

    #[test]
    fn test_parallel_require_one() {
        let mut tree = BehaviorTree::new(
            "test",
            Box::new(Parallel::new(
                "par",
                ParallelPolicy::RequireOne,
                vec![
                    Box::new(ActionNode::new("f1", |_, _| NodeStatus::Failure)),
                    Box::new(ActionNode::new("s1", |_, _| NodeStatus::Success)),
                ],
            )),
        );
        assert_eq!(tree.tick(0.016), NodeStatus::Success);
    }

    #[test]
    fn test_inverter() {
        let mut bb = Blackboard::new();
        let mut inv = Inverter::new(
            "inv",
            Box::new(ActionNode::new("succeed", |_, _| NodeStatus::Success)),
        );
        assert_eq!(inv.tick(0.016, &mut bb), NodeStatus::Failure);

        let mut inv2 = Inverter::new(
            "inv2",
            Box::new(ActionNode::new("fail", |_, _| NodeStatus::Failure)),
        );
        assert_eq!(inv2.tick(0.016, &mut bb), NodeStatus::Success);

        let mut inv3 = Inverter::new(
            "inv3",
            Box::new(ActionNode::new("run", |_, _| NodeStatus::Running)),
        );
        assert_eq!(inv3.tick(0.016, &mut bb), NodeStatus::Running);
    }

    #[test]
    fn test_repeater() {
        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();
        let mut rep = Repeater::new(
            "rep",
            Box::new(ActionNode::new("count", move |_, _| {
                c.fetch_add(1, Ordering::SeqCst);
                NodeStatus::Success
            })),
            3,
        );
        let mut bb = Blackboard::new();

        // First two ticks should return Running (child completes but we haven't
        // hit 3 repetitions yet).
        assert_eq!(rep.tick(0.016, &mut bb), NodeStatus::Running);
        assert_eq!(rep.tick(0.016, &mut bb), NodeStatus::Running);
        // Third tick completes the repeater.
        assert_eq!(rep.tick(0.016, &mut bb), NodeStatus::Success);
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_repeat_until_fail() {
        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();
        let mut rep = RepeatUntilFail::new(
            "ruf",
            Box::new(ActionNode::new("count", move |_, _| {
                let n = c.fetch_add(1, Ordering::SeqCst);
                if n < 3 {
                    NodeStatus::Success
                } else {
                    NodeStatus::Failure
                }
            })),
        );
        let mut bb = Blackboard::new();

        assert_eq!(rep.tick(0.016, &mut bb), NodeStatus::Running);
        assert_eq!(rep.tick(0.016, &mut bb), NodeStatus::Running);
        assert_eq!(rep.tick(0.016, &mut bb), NodeStatus::Running);
        assert_eq!(rep.tick(0.016, &mut bb), NodeStatus::Success);
        assert_eq!(counter.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn test_timeout() {
        let mut timeout = Timeout::new(
            "to",
            Box::new(ActionNode::new("slow", |_, _| NodeStatus::Running)),
            Duration::from_millis(100),
        );
        let mut bb = Blackboard::new();

        // 50ms: still running.
        assert_eq!(timeout.tick(0.050, &mut bb), NodeStatus::Running);
        // Another 60ms: total 110ms > 100ms timeout.
        assert_eq!(timeout.tick(0.060, &mut bb), NodeStatus::Failure);
    }

    #[test]
    fn test_timeout_child_completes() {
        let mut timeout = Timeout::new(
            "to",
            Box::new(ActionNode::new("fast", |_, _| NodeStatus::Success)),
            Duration::from_secs(10),
        );
        let mut bb = Blackboard::new();
        assert_eq!(timeout.tick(0.016, &mut bb), NodeStatus::Success);
    }

    #[test]
    fn test_cooldown() {
        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();
        let mut cd = Cooldown::new(
            "cd",
            Box::new(ActionNode::new("act", move |_, _| {
                c.fetch_add(1, Ordering::SeqCst);
                NodeStatus::Success
            })),
            1.0,
        );
        let mut bb = Blackboard::new();

        // First tick: child executes.
        assert_eq!(cd.tick(0.016, &mut bb), NodeStatus::Success);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Immediately: cooldown blocks.
        assert_eq!(cd.tick(0.016, &mut bb), NodeStatus::Failure);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // After 1.0 seconds: cooldown expires.
        assert_eq!(cd.tick(1.0, &mut bb), NodeStatus::Success);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_condition_decorator() {
        let mut bb = Blackboard::new();
        bb.set("allowed", true);

        let mut cd = ConditionDecorator::new(
            "guard",
            Box::new(ActionNode::new("act", |_, _| NodeStatus::Success)),
            |bb| bb.get::<bool>("allowed").copied().unwrap_or(false),
        );

        assert_eq!(cd.tick(0.016, &mut bb), NodeStatus::Success);

        bb.set("allowed", false);
        assert_eq!(cd.tick(0.016, &mut bb), NodeStatus::Failure);
    }

    #[test]
    fn test_condition_node() {
        let mut bb = Blackboard::new();
        bb.set("has_ammo", true);

        let mut cond = ConditionNode::new("ammo_check", |bb| {
            bb.get::<bool>("has_ammo").copied().unwrap_or(false)
        });

        assert_eq!(cond.tick(0.016, &mut bb), NodeStatus::Success);

        bb.set("has_ammo", false);
        assert_eq!(cond.tick(0.016, &mut bb), NodeStatus::Failure);
    }

    #[test]
    fn test_wait_node() {
        let mut wait = WaitNode::new("wait", 0.5);
        let mut bb = Blackboard::new();

        assert_eq!(wait.tick(0.2, &mut bb), NodeStatus::Running);
        assert_eq!(wait.tick(0.2, &mut bb), NodeStatus::Running);
        assert_eq!(wait.tick(0.2, &mut bb), NodeStatus::Success);
    }

    #[test]
    fn test_random_selector() {
        let counter = Arc::new(AtomicU32::new(0));
        let c1 = counter.clone();
        let c2 = counter.clone();
        let c3 = counter.clone();

        let mut rs = RandomSelector::new(
            "rand",
            vec![
                Box::new(ActionNode::new("a", move |_, _| {
                    c1.fetch_add(1, Ordering::SeqCst);
                    NodeStatus::Failure
                })),
                Box::new(ActionNode::new("b", move |_, _| {
                    c2.fetch_add(10, Ordering::SeqCst);
                    NodeStatus::Failure
                })),
                Box::new(ActionNode::new("c", move |_, _| {
                    c3.fetch_add(100, Ordering::SeqCst);
                    NodeStatus::Failure
                })),
            ],
        );
        let mut bb = Blackboard::new();

        // All fail, so result is Failure. But all 3 children were tried.
        assert_eq!(rs.tick(0.016, &mut bb), NodeStatus::Failure);
        assert_eq!(counter.load(Ordering::SeqCst), 111);
    }

    #[test]
    fn test_weighted_selector() {
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let o1 = order.clone();
        let o2 = order.clone();
        let o3 = order.clone();

        let mut ws = WeightedSelector::new(
            "weighted",
            vec![
                (1.0, Box::new(ActionNode::new("low", move |_, _| {
                    o1.lock().unwrap().push("low");
                    NodeStatus::Failure
                })) as Box<dyn BehaviorNode>),
                (10.0, Box::new(ActionNode::new("high", move |_, _| {
                    o2.lock().unwrap().push("high");
                    NodeStatus::Failure
                })) as Box<dyn BehaviorNode>),
                (5.0, Box::new(ActionNode::new("mid", move |_, _| {
                    o3.lock().unwrap().push("mid");
                    NodeStatus::Failure
                })) as Box<dyn BehaviorNode>),
            ],
        );
        let mut bb = Blackboard::new();

        assert_eq!(ws.tick(0.016, &mut bb), NodeStatus::Failure);

        let executed = order.lock().unwrap();
        // High weight should be tried first.
        assert_eq!(executed[0], "high");
        assert_eq!(executed[1], "mid");
        assert_eq!(executed[2], "low");
    }

    #[test]
    fn test_builder_api() {
        let mut tree = BehaviorTreeBuilder::new("test_tree")
            .selector("root")
                .sequence("attack_seq")
                    .condition("has_target", |bb| {
                        bb.get::<bool>("has_target").copied().unwrap_or(false)
                    })
                    .action("attack", |_, bb| {
                        bb.set("attacked", true);
                        NodeStatus::Success
                    })
                .end()
                .action("idle", |_, _| NodeStatus::Success)
            .end()
            .build();

        // No target: selector tries attack_seq (fails at condition), then idle (succeeds).
        assert_eq!(tree.tick(0.016), NodeStatus::Success);
        assert_eq!(tree.blackboard.get::<bool>("attacked"), None);

        // With target: attack succeeds.
        tree.blackboard.set("has_target", true);
        tree.reset();
        assert_eq!(tree.tick(0.016), NodeStatus::Success);
        assert_eq!(tree.blackboard.get::<bool>("attacked"), Some(&true));
    }

    #[test]
    fn test_behavior_tree_reset() {
        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();
        let mut tree = BehaviorTree::new(
            "test",
            Box::new(ActionNode::new("count", move |_, _| {
                c.fetch_add(1, Ordering::SeqCst);
                NodeStatus::Running
            })),
        );

        tree.tick(0.016);
        tree.tick(0.016);
        assert_eq!(tree.tick_count(), 2);
        assert_eq!(tree.last_status(), NodeStatus::Running);

        tree.reset();
        assert_eq!(tree.tick_count(), 0);
        assert_eq!(tree.last_status(), NodeStatus::Failure);
    }

    #[test]
    fn test_blackboard_with_action() {
        let mut tree = BehaviorTree::new(
            "test",
            Box::new(Sequence::new(
                "seq",
                vec![
                    Box::new(ActionNode::new("set_health", |_, bb| {
                        bb.set("health", 100i64);
                        NodeStatus::Success
                    })),
                    Box::new(ConditionNode::new("check_health", |bb| {
                        bb.get::<i64>("health").copied().unwrap_or(0) > 50
                    })),
                    Box::new(ActionNode::new("decrement", |_, bb| {
                        if let Some(h) = bb.get::<i64>("health").copied() {
                            bb.set("health", h - 30);
                        }
                        NodeStatus::Success
                    })),
                ],
            )),
        );

        assert_eq!(tree.tick(0.016), NodeStatus::Success);
        assert_eq!(tree.blackboard.get::<i64>("health"), Some(&70));
    }

    #[test]
    fn test_asset_instantiation() {
        let asset = BehaviorTreeAsset {
            name: "test_asset".into(),
            root: NodeDefinition {
                node_type: "selector".into(),
                name: "root".into(),
                children: vec![
                    NodeDefinition {
                        node_type: "action".into(),
                        name: "greet".into(),
                        children: vec![],
                        params: HashMap::new(),
                    },
                    NodeDefinition {
                        node_type: "wait".into(),
                        name: "pause".into(),
                        children: vec![],
                        params: {
                            let mut m = HashMap::new();
                            m.insert(
                                "duration".into(),
                                serde_json::Value::Number(serde_json::Number::from_f64(2.0).unwrap()),
                            );
                            m
                        },
                    },
                ],
                params: HashMap::new(),
            },
        };

        let mut registry: HashMap<String, NodeFactory> = HashMap::new();
        registry.insert(
            "greet".into(),
            Box::new(|| {
                Box::new(ActionNode::new("greet", |_, _| NodeStatus::Failure))
            }),
        );

        let result = asset.instantiate(&registry);
        assert!(result.is_ok());
        let mut tree = result.unwrap();
        // greet fails, wait returns Running.
        assert_eq!(tree.tick(0.016), NodeStatus::Running);
    }

    #[test]
    fn test_node_status_helpers() {
        assert!(NodeStatus::Success.is_success());
        assert!(!NodeStatus::Success.is_failure());
        assert!(!NodeStatus::Success.is_running());
        assert_eq!(NodeStatus::Success.invert(), NodeStatus::Failure);
        assert_eq!(NodeStatus::Failure.invert(), NodeStatus::Success);
        assert_eq!(NodeStatus::Running.invert(), NodeStatus::Running);
    }

    #[test]
    fn test_complex_tree() {
        // Build a more complex tree to test deep nesting.
        let mut tree = BehaviorTreeBuilder::new("complex")
            .selector("root")
                .sequence("combat")
                    .condition("enemy_near", |bb| {
                        bb.get::<bool>("enemy_near").copied().unwrap_or(false)
                    })
                    .selector("attack_or_flee")
                        .sequence("attack_if_strong")
                            .condition("is_strong", |bb| {
                                bb.get::<i64>("strength").copied().unwrap_or(0) > 5
                            })
                            .action("attack", |_, bb| {
                                bb.set("action", "attack".to_string());
                                NodeStatus::Success
                            })
                        .end()
                        .action("flee", |_, bb| {
                            bb.set("action", "flee".to_string());
                            NodeStatus::Success
                        })
                    .end()
                .end()
                .action("patrol", |_, bb| {
                    bb.set("action", "patrol".to_string());
                    NodeStatus::Success
                })
            .end()
            .build();

        // No enemy: patrol.
        assert_eq!(tree.tick(0.016), NodeStatus::Success);
        assert_eq!(
            tree.blackboard.get::<String>("action"),
            Some(&"patrol".to_string())
        );

        // Enemy near, weak: flee.
        tree.reset();
        tree.blackboard.set("enemy_near", true);
        tree.blackboard.set("strength", 3i64);
        assert_eq!(tree.tick(0.016), NodeStatus::Success);
        assert_eq!(
            tree.blackboard.get::<String>("action"),
            Some(&"flee".to_string())
        );

        // Enemy near, strong: attack.
        tree.reset();
        tree.blackboard.set("enemy_near", true);
        tree.blackboard.set("strength", 10i64);
        assert_eq!(tree.tick(0.016), NodeStatus::Success);
        assert_eq!(
            tree.blackboard.get::<String>("action"),
            Some(&"attack".to_string())
        );
    }
}
