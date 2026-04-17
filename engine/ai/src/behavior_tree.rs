// engine/ai/src/behavior_tree_v2.rs
//
// Enhanced behavior trees with utility-scored selectors, advanced decorators,
// subtree references, shared blackboards, and runtime debugging data.

use std::collections::HashMap;

pub const MAX_BT_DEPTH: usize = 32;
pub const MAX_CHILDREN: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BtStatus { Running, Success, Failure, Invalid }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BtNodeId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BtNodeType {
    Sequence, Selector, Parallel, UtilitySelector, RandomSelector,
    Action, Condition, Decorator, SubtreeRef, Root,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoratorType {
    Inverter, Repeater, RepeatUntilFail, RepeatUntilSuccess,
    Cooldown, Limit, Timeout, Probability, ForceSuccess, ForceFailure,
    Delay, Guard, Semaphore, ObserverAbort,
}

#[derive(Debug, Clone)]
pub struct BtDecorator {
    pub decorator_type: DecoratorType,
    pub int_param: i32,
    pub float_param: f32,
    pub cooldown_remaining: f32,
    pub execution_count: u32,
    pub max_executions: u32,
    pub timeout_elapsed: f32,
    pub probability: f32,
    pub delay_time: f32,
    pub delay_elapsed: f32,
}

impl BtDecorator {
    pub fn inverter() -> Self { Self { decorator_type: DecoratorType::Inverter, ..Default::default() } }
    pub fn cooldown(seconds: f32) -> Self { Self { decorator_type: DecoratorType::Cooldown, float_param: seconds, ..Default::default() } }
    pub fn limit(max: u32) -> Self { Self { decorator_type: DecoratorType::Limit, max_executions: max, ..Default::default() } }
    pub fn timeout(seconds: f32) -> Self { Self { decorator_type: DecoratorType::Timeout, float_param: seconds, ..Default::default() } }
    pub fn probability(p: f32) -> Self { Self { decorator_type: DecoratorType::Probability, probability: p, ..Default::default() } }
    pub fn force_success() -> Self { Self { decorator_type: DecoratorType::ForceSuccess, ..Default::default() } }
    pub fn force_failure() -> Self { Self { decorator_type: DecoratorType::ForceFailure, ..Default::default() } }
    pub fn repeater(times: i32) -> Self { Self { decorator_type: DecoratorType::Repeater, int_param: times, ..Default::default() } }
    pub fn delay(seconds: f32) -> Self { Self { decorator_type: DecoratorType::Delay, delay_time: seconds, ..Default::default() } }

    pub fn can_execute(&self) -> bool {
        match self.decorator_type {
            DecoratorType::Cooldown => self.cooldown_remaining <= 0.0,
            DecoratorType::Limit => self.execution_count < self.max_executions,
            DecoratorType::Probability => { let r: f32 = (self.execution_count as f32 * 0.1234).fract(); r < self.probability }
            DecoratorType::Delay => self.delay_elapsed >= self.delay_time,
            _ => true,
        }
    }

    pub fn apply_result(&self, child_status: BtStatus) -> BtStatus {
        match self.decorator_type {
            DecoratorType::Inverter => match child_status {
                BtStatus::Success => BtStatus::Failure,
                BtStatus::Failure => BtStatus::Success,
                other => other,
            },
            DecoratorType::ForceSuccess => if child_status != BtStatus::Running { BtStatus::Success } else { BtStatus::Running },
            DecoratorType::ForceFailure => if child_status != BtStatus::Running { BtStatus::Failure } else { BtStatus::Running },
            _ => child_status,
        }
    }

    pub fn update(&mut self, dt: f32) {
        if self.cooldown_remaining > 0.0 { self.cooldown_remaining -= dt; }
        if self.delay_elapsed < self.delay_time { self.delay_elapsed += dt; }
        if self.timeout_elapsed < self.float_param { self.timeout_elapsed += dt; }
    }

    pub fn reset(&mut self) {
        self.cooldown_remaining = 0.0;
        self.execution_count = 0;
        self.timeout_elapsed = 0.0;
        self.delay_elapsed = 0.0;
    }
}

impl Default for BtDecorator {
    fn default() -> Self {
        Self {
            decorator_type: DecoratorType::Inverter, int_param: 0, float_param: 0.0,
            cooldown_remaining: 0.0, execution_count: 0, max_executions: u32::MAX,
            timeout_elapsed: 0.0, probability: 1.0, delay_time: 0.0, delay_elapsed: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct UtilityScore {
    pub base_score: f32,
    pub considerations: Vec<UtilityConsideration>,
}

#[derive(Debug, Clone)]
pub struct UtilityConsideration {
    pub name: String,
    pub blackboard_key: String,
    pub curve_type: UtilityCurve,
    pub weight: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UtilityCurve { Linear, Quadratic, InverseLinear, Sigmoid, Constant(f32), Step(f32) }

impl UtilityConsideration {
    pub fn evaluate(&self, input: f32) -> f32 {
        let raw = match self.curve_type {
            UtilityCurve::Linear => input.clamp(0.0, 1.0),
            UtilityCurve::Quadratic => { let v = input.clamp(0.0, 1.0); v * v }
            UtilityCurve::InverseLinear => (1.0 - input).clamp(0.0, 1.0),
            UtilityCurve::Sigmoid => 1.0 / (1.0 + (-10.0 * (input - 0.5)).exp()),
            UtilityCurve::Constant(v) => v,
            UtilityCurve::Step(threshold) => if input >= threshold { 1.0 } else { 0.0 },
        };
        raw * self.weight
    }
}

impl UtilityScore {
    pub fn new(base: f32) -> Self { Self { base_score: base, considerations: Vec::new() } }
    pub fn evaluate(&self, blackboard: &SharedBlackboard) -> f32 {
        let mut score = self.base_score;
        for c in &self.considerations {
            let input = blackboard.get_float(&c.blackboard_key).unwrap_or(0.0);
            score *= c.evaluate(input);
        }
        score
    }
}

#[derive(Debug, Clone)]
pub struct BtNodeV2 {
    pub id: BtNodeId,
    pub name: String,
    pub node_type: BtNodeType,
    pub children: Vec<BtNodeId>,
    pub decorator: Option<BtDecorator>,
    pub utility: Option<UtilityScore>,
    pub subtree_ref: Option<String>,
    pub status: BtStatus,
    pub action_fn_id: u32,
    pub condition_fn_id: u32,
    pub running_child: usize,
    pub debug_info: BtDebugInfo,
}

#[derive(Debug, Clone, Default)]
pub struct BtDebugInfo {
    pub total_ticks: u64,
    pub success_count: u64,
    pub failure_count: u64,
    pub running_count: u64,
    pub last_status: BtStatus,
    pub last_tick_time: f64,
    pub avg_tick_duration_us: f64,
    pub active: bool,
}

impl BtNodeV2 {
    pub fn sequence(id: BtNodeId, name: &str) -> Self {
        Self::new(id, name, BtNodeType::Sequence)
    }
    pub fn selector(id: BtNodeId, name: &str) -> Self {
        Self::new(id, name, BtNodeType::Selector)
    }
    pub fn utility_selector(id: BtNodeId, name: &str) -> Self {
        Self::new(id, name, BtNodeType::UtilitySelector)
    }
    pub fn action(id: BtNodeId, name: &str, fn_id: u32) -> Self {
        let mut n = Self::new(id, name, BtNodeType::Action);
        n.action_fn_id = fn_id;
        n
    }
    pub fn condition(id: BtNodeId, name: &str, fn_id: u32) -> Self {
        let mut n = Self::new(id, name, BtNodeType::Condition);
        n.condition_fn_id = fn_id;
        n
    }
    pub fn subtree(id: BtNodeId, name: &str, tree_name: &str) -> Self {
        let mut n = Self::new(id, name, BtNodeType::SubtreeRef);
        n.subtree_ref = Some(tree_name.to_string());
        n
    }

    fn new(id: BtNodeId, name: &str, node_type: BtNodeType) -> Self {
        Self {
            id, name: name.to_string(), node_type, children: Vec::new(),
            decorator: None, utility: None, subtree_ref: None,
            status: BtStatus::Invalid, action_fn_id: 0, condition_fn_id: 0,
            running_child: 0, debug_info: BtDebugInfo::default(),
        }
    }

    pub fn with_decorator(mut self, dec: BtDecorator) -> Self { self.decorator = Some(dec); self }
    pub fn with_utility(mut self, score: UtilityScore) -> Self { self.utility = Some(score); self }
    pub fn add_child(&mut self, child_id: BtNodeId) { self.children.push(child_id); }

    pub fn record_tick(&mut self, status: BtStatus) {
        self.debug_info.total_ticks += 1;
        self.debug_info.last_status = status;
        self.debug_info.active = status == BtStatus::Running;
        match status {
            BtStatus::Success => self.debug_info.success_count += 1,
            BtStatus::Failure => self.debug_info.failure_count += 1,
            BtStatus::Running => self.debug_info.running_count += 1,
            _ => {}
        }
    }
}

#[derive(Debug, Clone)]
pub struct SharedBlackboard {
    pub ints: HashMap<String, i32>,
    pub floats: HashMap<String, f32>,
    pub bools: HashMap<String, bool>,
    pub strings: HashMap<String, String>,
    pub vectors: HashMap<String, [f32; 3]>,
}

impl SharedBlackboard {
    pub fn new() -> Self {
        Self {
            ints: HashMap::new(), floats: HashMap::new(), bools: HashMap::new(),
            strings: HashMap::new(), vectors: HashMap::new(),
        }
    }
    pub fn set_int(&mut self, key: &str, val: i32) { self.ints.insert(key.to_string(), val); }
    pub fn set_float(&mut self, key: &str, val: f32) { self.floats.insert(key.to_string(), val); }
    pub fn set_bool(&mut self, key: &str, val: bool) { self.bools.insert(key.to_string(), val); }
    pub fn set_string(&mut self, key: &str, val: &str) { self.strings.insert(key.to_string(), val.to_string()); }
    pub fn set_vector(&mut self, key: &str, val: [f32; 3]) { self.vectors.insert(key.to_string(), val); }
    pub fn get_int(&self, key: &str) -> Option<i32> { self.ints.get(key).copied() }
    pub fn get_float(&self, key: &str) -> Option<f32> { self.floats.get(key).copied() }
    pub fn get_bool(&self, key: &str) -> Option<bool> { self.bools.get(key).copied() }
    pub fn get_string(&self, key: &str) -> Option<&str> { self.strings.get(key).map(|s| s.as_str()) }
    pub fn get_vector(&self, key: &str) -> Option<[f32; 3]> { self.vectors.get(key).copied() }
    pub fn clear(&mut self) { self.ints.clear(); self.floats.clear(); self.bools.clear(); self.strings.clear(); self.vectors.clear(); }
}

impl Default for SharedBlackboard { fn default() -> Self { Self::new() } }

#[derive(Debug)]
pub struct BehaviorTreeV2 {
    pub name: String,
    pub nodes: HashMap<BtNodeId, BtNodeV2>,
    pub root_id: BtNodeId,
    pub blackboard: SharedBlackboard,
    next_id: u32,
}

impl BehaviorTreeV2 {
    pub fn new(name: &str) -> Self {
        let root_id = BtNodeId(0);
        let mut nodes = HashMap::new();
        nodes.insert(root_id, BtNodeV2::new(root_id, "root", BtNodeType::Root));
        Self { name: name.to_string(), nodes, root_id, blackboard: SharedBlackboard::new(), next_id: 1 }
    }
    pub fn add_node(&mut self, node: BtNodeV2) -> BtNodeId {
        let id = node.id;
        self.nodes.insert(id, node);
        id
    }
    pub fn next_id(&mut self) -> BtNodeId { let id = BtNodeId(self.next_id); self.next_id += 1; id }
    pub fn set_root_child(&mut self, child_id: BtNodeId) {
        if let Some(root) = self.nodes.get_mut(&self.root_id) { root.children = vec![child_id]; }
    }
    pub fn node_count(&self) -> usize { self.nodes.len() }
    pub fn get_debug_data(&self) -> Vec<(BtNodeId, &str, BtStatus, u64)> {
        self.nodes.iter().map(|(id, n)| (*id, n.name.as_str(), n.debug_info.last_status, n.debug_info.total_ticks)).collect()
    }
}

#[derive(Debug)]
pub struct BtLibrary {
    pub trees: HashMap<String, BehaviorTreeV2>,
}

impl BtLibrary {
    pub fn new() -> Self { Self { trees: HashMap::new() } }
    pub fn register(&mut self, tree: BehaviorTreeV2) { self.trees.insert(tree.name.clone(), tree); }
    pub fn get(&self, name: &str) -> Option<&BehaviorTreeV2> { self.trees.get(name) }
    pub fn get_mut(&mut self, name: &str) -> Option<&mut BehaviorTreeV2> { self.trees.get_mut(name) }
    pub fn tree_count(&self) -> usize { self.trees.len() }
}

impl Default for BtLibrary { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_decorator_inverter() {
        let dec = BtDecorator::inverter();
        assert_eq!(dec.apply_result(BtStatus::Success), BtStatus::Failure);
        assert_eq!(dec.apply_result(BtStatus::Failure), BtStatus::Success);
    }
    #[test]
    fn test_blackboard() {
        let mut bb = SharedBlackboard::new();
        bb.set_float("health", 0.75);
        bb.set_bool("alive", true);
        assert_eq!(bb.get_float("health"), Some(0.75));
        assert_eq!(bb.get_bool("alive"), Some(true));
    }
    #[test]
    fn test_utility_score() {
        let bb = SharedBlackboard::new();
        let score = UtilityScore::new(1.0);
        assert!((score.evaluate(&bb) - 1.0).abs() < 0.01);
    }
    #[test]
    fn test_behavior_tree() {
        let mut bt = BehaviorTreeV2::new("test");
        let id = bt.next_id();
        let node = BtNodeV2::selector(id, "main_selector");
        bt.add_node(node);
        bt.set_root_child(id);
        assert_eq!(bt.node_count(), 2);
    }
}
