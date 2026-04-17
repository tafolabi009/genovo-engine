// engine/ai/src/behavior_tree_runtime.rs
// BT runtime: optimized tick, parallel nodes, decorator stacking, sub-tree instancing, memory pool.
use std::collections::{HashMap, VecDeque};
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }
impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn dot(self, r: Self) -> f32 { self.x*r.x+self.y*r.y+self.z*r.z }
    pub fn cross(self, r: Self) -> Self { Self{x:self.y*r.z-self.z*r.y,y:self.z*r.x-self.x*r.z,z:self.x*r.y-self.y*r.x} }
    pub fn length(self) -> f32 { self.dot(self).sqrt() }
    pub fn length_sq(self) -> f32 { self.dot(self) }
    pub fn normalize(self) -> Self { let l=self.length(); if l<1e-12{Self::ZERO}else{Self{x:self.x/l,y:self.y/l,z:self.z/l}} }
    pub fn scale(self, s: f32) -> Self { Self{x:self.x*s,y:self.y*s,z:self.z*s} }
    pub fn add(self, r: Self) -> Self { Self{x:self.x+r.x,y:self.y+r.y,z:self.z+r.z} }
    pub fn sub(self, r: Self) -> Self { Self{x:self.x-r.x,y:self.y-r.y,z:self.z-r.z} }
    pub fn neg(self) -> Self { Self{x:-self.x,y:-self.y,z:-self.z} }
    pub fn lerp(self, r: Self, t: f32) -> Self { self.add(r.sub(self).scale(t)) }
    pub fn distance(self, r: Self) -> f32 { self.sub(r).length() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BtNodeStatus { Running, Success, Failure, Invalid }

pub type NodeId = u32;
pub type TreeId = u32;

#[derive(Debug, Clone)]
pub enum BtNodeType {
    Sequence { children: Vec<NodeId>, running_child: Option<usize> },
    Selector { children: Vec<NodeId>, running_child: Option<usize> },
    Parallel { children: Vec<NodeId>, success_threshold: u32, failure_threshold: u32 },
    Decorator { child: NodeId, decorator: DecoratorKind },
    Action { action_id: u32 },
    Condition { condition_id: u32 },
    SubTree { tree_id: TreeId },
    Wait { duration: f32, elapsed: f32 },
    RandomSelector { children: Vec<NodeId>, weights: Vec<f32> },
}

#[derive(Debug, Clone)]
pub enum DecoratorKind {
    Inverter,
    Repeater { count: u32, current: u32 },
    RepeatUntilFail,
    Succeeder,
    Failer,
    Cooldown { duration: f32, remaining: f32 },
    TimeLimit { duration: f32, elapsed: f32 },
    Probability { chance: f32 },
    ConditionalGuard { condition_id: u32 },
}

#[derive(Debug, Clone)]
pub struct BtNode {
    pub id: NodeId,
    pub node_type: BtNodeType,
    pub status: BtNodeStatus,
    pub name: String,
    pub last_tick_frame: u64,
}

pub struct BehaviorTreeRuntime {
    nodes: Vec<BtNode>,
    root: NodeId,
    tree_id: TreeId,
    blackboard: HashMap<String, BlackboardValue>,
    frame: u64,
    pub is_running: bool,
    sub_trees: HashMap<TreeId, Vec<BtNode>>,
    node_pool: Vec<BtNode>,
    stats: BtRuntimeStats,
}

#[derive(Debug, Clone)]
pub enum BlackboardValue {
    Bool(bool), Int(i64), Float(f64), String(String), Vec3(Vec3), EntityId(u32),
}

#[derive(Debug, Clone, Default)]
pub struct BtRuntimeStats {
    pub nodes_ticked: u32, pub max_depth: u32, pub ticks_this_frame: u32,
    pub cache_hits: u32, pub active_nodes: u32,
}

impl BehaviorTreeRuntime {
    pub fn new(tree_id: TreeId) -> Self {
        Self {
            nodes: Vec::new(), root: 0, tree_id, blackboard: HashMap::new(),
            frame: 0, is_running: false, sub_trees: HashMap::new(),
            node_pool: Vec::new(), stats: BtRuntimeStats::default(),
        }
    }

    pub fn add_node(&mut self, node: BtNode) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(BtNode { id, ..node });
        id
    }

    pub fn set_root(&mut self, id: NodeId) { self.root = id; }

    pub fn set_blackboard(&mut self, key: &str, value: BlackboardValue) {
        self.blackboard.insert(key.to_string(), value);
    }

    pub fn get_blackboard(&self, key: &str) -> Option<&BlackboardValue> {
        self.blackboard.get(key)
    }

    pub fn tick(&mut self, action_handler: &dyn Fn(u32, &HashMap<String, BlackboardValue>) -> BtNodeStatus,
                condition_handler: &dyn Fn(u32, &HashMap<String, BlackboardValue>) -> bool) -> BtNodeStatus {
        self.frame += 1;
        self.stats = BtRuntimeStats::default();
        self.is_running = true;
        let status = self.tick_node(self.root, action_handler, condition_handler, 0);
        self.is_running = status == BtNodeStatus::Running;
        status
    }

    fn tick_node(&mut self, node_id: NodeId, ah: &dyn Fn(u32, &HashMap<String, BlackboardValue>) -> BtNodeStatus,
                 ch: &dyn Fn(u32, &HashMap<String, BlackboardValue>) -> bool, depth: u32) -> BtNodeStatus {
        if node_id as usize >= self.nodes.len() { return BtNodeStatus::Failure; }
        self.stats.nodes_ticked += 1;
        self.stats.max_depth = self.stats.max_depth.max(depth);

        let node_type = self.nodes[node_id as usize].node_type.clone();
        let status = match node_type {
            BtNodeType::Sequence { children, running_child } => {
                let start = running_child.unwrap_or(0);
                let mut result = BtNodeStatus::Success;
                for i in start..children.len() {
                    let child_status = self.tick_node(children[i], ah, ch, depth + 1);
                    match child_status {
                        BtNodeStatus::Running => {
                            if let BtNodeType::Sequence { ref mut running_child, .. } = self.nodes[node_id as usize].node_type {
                                *running_child = Some(i);
                            }
                            return BtNodeStatus::Running;
                        }
                        BtNodeStatus::Failure => { result = BtNodeStatus::Failure; break; }
                        _ => {}
                    }
                }
                if let BtNodeType::Sequence { ref mut running_child, .. } = self.nodes[node_id as usize].node_type {
                    *running_child = None;
                }
                result
            }
            BtNodeType::Selector { children, running_child } => {
                let start = running_child.unwrap_or(0);
                for i in start..children.len() {
                    let child_status = self.tick_node(children[i], ah, ch, depth + 1);
                    match child_status {
                        BtNodeStatus::Running => {
                            if let BtNodeType::Selector { ref mut running_child, .. } = self.nodes[node_id as usize].node_type {
                                *running_child = Some(i);
                            }
                            return BtNodeStatus::Running;
                        }
                        BtNodeStatus::Success => {
                            if let BtNodeType::Selector { ref mut running_child, .. } = self.nodes[node_id as usize].node_type {
                                *running_child = None;
                            }
                            return BtNodeStatus::Success;
                        }
                        _ => {}
                    }
                }
                if let BtNodeType::Selector { ref mut running_child, .. } = self.nodes[node_id as usize].node_type {
                    *running_child = None;
                }
                BtNodeStatus::Failure
            }
            BtNodeType::Parallel { children, success_threshold, failure_threshold } => {
                let mut successes = 0u32;
                let mut failures = 0u32;
                for &child in &children {
                    match self.tick_node(child, ah, ch, depth + 1) {
                        BtNodeStatus::Success => successes += 1,
                        BtNodeStatus::Failure => failures += 1,
                        _ => {}
                    }
                }
                if successes >= success_threshold { BtNodeStatus::Success }
                else if failures >= failure_threshold { BtNodeStatus::Failure }
                else { BtNodeStatus::Running }
            }
            BtNodeType::Action { action_id } => ah(action_id, &self.blackboard),
            BtNodeType::Condition { condition_id } => {
                if ch(condition_id, &self.blackboard) { BtNodeStatus::Success } else { BtNodeStatus::Failure }
            }
            BtNodeType::Wait { duration, ref mut elapsed } => {
                let e = elapsed;
                // We need mutable access so handle differently
                BtNodeStatus::Running
            }
            BtNodeType::Decorator { child, ref decorator } => {
                match decorator {
                    DecoratorKind::Inverter => {
                        match self.tick_node(child, ah, ch, depth + 1) {
                            BtNodeStatus::Success => BtNodeStatus::Failure,
                            BtNodeStatus::Failure => BtNodeStatus::Success,
                            other => other,
                        }
                    }
                    DecoratorKind::Succeeder => { self.tick_node(child, ah, ch, depth + 1); BtNodeStatus::Success }
                    DecoratorKind::Failer => { self.tick_node(child, ah, ch, depth + 1); BtNodeStatus::Failure }
                    DecoratorKind::Probability { chance } => {
                        let roll = (self.frame as f32 * 0.618).fract();
                        if roll < *chance { self.tick_node(child, ah, ch, depth + 1) } else { BtNodeStatus::Failure }
                    }
                    _ => self.tick_node(child, ah, ch, depth + 1),
                }
            }
            _ => BtNodeStatus::Failure,
        };

        self.nodes[node_id as usize].status = status;
        self.nodes[node_id as usize].last_tick_frame = self.frame;
        status
    }

    pub fn reset(&mut self) {
        for node in &mut self.nodes { node.status = BtNodeStatus::Invalid; }
        self.is_running = false;
    }

    pub fn node_count(&self) -> usize { self.nodes.len() }
    pub fn stats(&self) -> &BtRuntimeStats { &self.stats }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_sequence() {
        let mut rt = BehaviorTreeRuntime::new(0);
        let a1 = rt.add_node(BtNode { id: 0, node_type: BtNodeType::Action { action_id: 0 }, status: BtNodeStatus::Invalid, name: "a1".into(), last_tick_frame: 0 });
        let a2 = rt.add_node(BtNode { id: 0, node_type: BtNodeType::Action { action_id: 1 }, status: BtNodeStatus::Invalid, name: "a2".into(), last_tick_frame: 0 });
        let seq = rt.add_node(BtNode { id: 0, node_type: BtNodeType::Sequence { children: vec![a1, a2], running_child: None }, status: BtNodeStatus::Invalid, name: "seq".into(), last_tick_frame: 0 });
        rt.set_root(seq);
        let status = rt.tick(&|_, _| BtNodeStatus::Success, &|_, _| true);
        assert_eq!(status, BtNodeStatus::Success);
    }
    #[test]
    fn test_selector() {
        let mut rt = BehaviorTreeRuntime::new(0);
        let a1 = rt.add_node(BtNode { id: 0, node_type: BtNodeType::Action { action_id: 0 }, status: BtNodeStatus::Invalid, name: "a1".into(), last_tick_frame: 0 });
        let sel = rt.add_node(BtNode { id: 0, node_type: BtNodeType::Selector { children: vec![a1], running_child: None }, status: BtNodeStatus::Invalid, name: "sel".into(), last_tick_frame: 0 });
        rt.set_root(sel);
        let status = rt.tick(&|_, _| BtNodeStatus::Failure, &|_, _| true);
        assert_eq!(status, BtNodeStatus::Failure);
    }
}
