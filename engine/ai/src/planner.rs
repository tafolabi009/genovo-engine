// engine/ai/src/planner_v2.rs
//
// Hierarchical Task Network (HTN) planner for the Genovo AI module.
//
// Compound tasks decompose into primitive tasks via methods with
// preconditions. Plan search uses backtracking. Supports partial plan
// execution, plan monitoring, and replanning on failure.

use std::collections::{HashMap, VecDeque};
use std::fmt;

pub type TaskId = u32;
pub type MethodId = u32;
pub type PropertyId = u32;

pub const MAX_PLAN_DEPTH: usize = 64;
pub const MAX_PLAN_LENGTH: usize = 256;
pub const MAX_BACKTRACK_STEPS: usize = 1000;

#[derive(Debug, Clone, PartialEq)]
pub enum PropertyValue { Bool(bool), Int(i64), Float(f64), String(String), None }

impl PropertyValue {
    pub fn as_bool(&self) -> bool { match self { Self::Bool(b) => *b, _ => false } }
    pub fn as_int(&self) -> i64 { match self { Self::Int(i) => *i, _ => 0 } }
    pub fn as_float(&self) -> f64 { match self { Self::Float(f) => *f, _ => 0.0 } }
}

#[derive(Debug, Clone, Default)]
pub struct WorldState {
    pub properties: HashMap<PropertyId, PropertyValue>,
}

impl WorldState {
    pub fn new() -> Self { Self::default() }
    pub fn set(&mut self, id: PropertyId, val: PropertyValue) { self.properties.insert(id, val); }
    pub fn get(&self, id: PropertyId) -> &PropertyValue { self.properties.get(&id).unwrap_or(&PropertyValue::None) }
    pub fn set_bool(&mut self, id: PropertyId, v: bool) { self.set(id, PropertyValue::Bool(v)); }
    pub fn set_int(&mut self, id: PropertyId, v: i64) { self.set(id, PropertyValue::Int(v)); }
    pub fn get_bool(&self, id: PropertyId) -> bool { self.get(id).as_bool() }
    pub fn get_int(&self, id: PropertyId) -> i64 { self.get(id).as_int() }
    pub fn clone_state(&self) -> Self { Self { properties: self.properties.clone() } }
}

#[derive(Debug, Clone)]
pub enum Condition {
    IsTrue(PropertyId),
    IsFalse(PropertyId),
    Equals(PropertyId, PropertyValue),
    GreaterThan(PropertyId, i64),
    LessThan(PropertyId, i64),
    And(Vec<Condition>),
    Or(Vec<Condition>),
    Not(Box<Condition>),
    Always,
}

impl Condition {
    pub fn evaluate(&self, state: &WorldState) -> bool {
        match self {
            Self::IsTrue(id) => state.get_bool(*id),
            Self::IsFalse(id) => !state.get_bool(*id),
            Self::Equals(id, val) => state.get(*id) == val,
            Self::GreaterThan(id, val) => state.get_int(*id) > *val,
            Self::LessThan(id, val) => state.get_int(*id) < *val,
            Self::And(cs) => cs.iter().all(|c| c.evaluate(state)),
            Self::Or(cs) => cs.iter().any(|c| c.evaluate(state)),
            Self::Not(c) => !c.evaluate(state),
            Self::Always => true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Effect {
    pub property: PropertyId,
    pub value: PropertyValue,
}

impl Effect {
    pub fn set_bool(id: PropertyId, v: bool) -> Self { Self { property: id, value: PropertyValue::Bool(v) } }
    pub fn set_int(id: PropertyId, v: i64) -> Self { Self { property: id, value: PropertyValue::Int(v) } }
    pub fn apply(&self, state: &mut WorldState) { state.set(self.property, self.value.clone()); }
}

#[derive(Debug, Clone)]
pub struct PrimitiveTask {
    pub id: TaskId,
    pub name: String,
    pub preconditions: Condition,
    pub effects: Vec<Effect>,
    pub cost: f32,
    pub duration: f32,
    pub interruptible: bool,
}

impl PrimitiveTask {
    pub fn new(id: TaskId, name: &str) -> Self {
        Self { id, name: name.to_string(), preconditions: Condition::Always, effects: Vec::new(), cost: 1.0, duration: 1.0, interruptible: true }
    }
    pub fn with_precondition(mut self, c: Condition) -> Self { self.preconditions = c; self }
    pub fn with_effect(mut self, e: Effect) -> Self { self.effects.push(e); self }
    pub fn with_cost(mut self, c: f32) -> Self { self.cost = c; self }
    pub fn can_execute(&self, state: &WorldState) -> bool { self.preconditions.evaluate(state) }
    pub fn apply_effects(&self, state: &mut WorldState) { for e in &self.effects { e.apply(state); } }
}

#[derive(Debug, Clone)]
pub struct Method {
    pub id: MethodId,
    pub name: String,
    pub preconditions: Condition,
    pub subtasks: Vec<TaskId>,
    pub priority: u32,
}

impl Method {
    pub fn new(id: MethodId, name: &str, precondition: Condition, subtasks: Vec<TaskId>) -> Self {
        Self { id, name: name.to_string(), preconditions: precondition, subtasks, priority: 0 }
    }
    pub fn is_applicable(&self, state: &WorldState) -> bool { self.preconditions.evaluate(state) }
}

#[derive(Debug, Clone)]
pub struct CompoundTask {
    pub id: TaskId,
    pub name: String,
    pub methods: Vec<Method>,
}

impl CompoundTask {
    pub fn new(id: TaskId, name: &str) -> Self { Self { id, name: name.to_string(), methods: Vec::new() } }
    pub fn add_method(&mut self, m: Method) { self.methods.push(m); }
    pub fn find_applicable_method(&self, state: &WorldState) -> Option<&Method> {
        let mut sorted: Vec<&Method> = self.methods.iter().collect();
        sorted.sort_by(|a, b| b.priority.cmp(&a.priority));
        sorted.into_iter().find(|m| m.is_applicable(state))
    }
}

#[derive(Debug, Clone)]
pub enum Task {
    Primitive(PrimitiveTask),
    Compound(CompoundTask),
}

impl Task {
    pub fn id(&self) -> TaskId { match self { Self::Primitive(t) => t.id, Self::Compound(t) => t.id } }
    pub fn name(&self) -> &str { match self { Self::Primitive(t) => &t.name, Self::Compound(t) => &t.name } }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanTaskStatus { Pending, Running, Completed, Failed }

#[derive(Debug, Clone)]
pub struct PlanStep {
    pub task_id: TaskId,
    pub task_name: String,
    pub status: PlanTaskStatus,
    pub cost: f32,
    pub duration: f32,
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub steps: Vec<PlanStep>,
    pub total_cost: f32,
    pub total_duration: f32,
    pub current_step: usize,
    pub valid: bool,
    pub backtrack_count: u32,
}

impl Plan {
    pub fn empty() -> Self { Self { steps: Vec::new(), total_cost: 0.0, total_duration: 0.0, current_step: 0, valid: false, backtrack_count: 0 } }
    pub fn is_empty(&self) -> bool { self.steps.is_empty() }
    pub fn is_complete(&self) -> bool { self.current_step >= self.steps.len() }
    pub fn current(&self) -> Option<&PlanStep> { self.steps.get(self.current_step) }
    pub fn advance(&mut self) { if self.current_step < self.steps.len() { self.steps[self.current_step].status = PlanTaskStatus::Completed; self.current_step += 1; } }
    pub fn fail_current(&mut self) { if self.current_step < self.steps.len() { self.steps[self.current_step].status = PlanTaskStatus::Failed; self.valid = false; } }
    pub fn remaining_steps(&self) -> usize { self.steps.len().saturating_sub(self.current_step) }
    pub fn progress(&self) -> f32 { if self.steps.is_empty() { 0.0 } else { self.current_step as f32 / self.steps.len() as f32 } }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlannerState { Idle, Planning, Executing, Replanning, Failed }

#[derive(Debug, Clone, Copy, Default)]
pub struct PlannerStats {
    pub plans_generated: u32,
    pub plans_completed: u32,
    pub plans_failed: u32,
    pub replans: u32,
    pub total_backtrack_steps: u32,
    pub avg_plan_length: f32,
}

pub struct HtnPlanner {
    tasks: HashMap<TaskId, Task>,
    current_plan: Plan,
    world_state: WorldState,
    state: PlannerState,
    root_task: Option<TaskId>,
    stats: PlannerStats,
    replan_on_failure: bool,
    max_replan_attempts: u32,
    replan_attempts: u32,
}

impl HtnPlanner {
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(), current_plan: Plan::empty(),
            world_state: WorldState::new(), state: PlannerState::Idle,
            root_task: None, stats: PlannerStats::default(),
            replan_on_failure: true, max_replan_attempts: 3, replan_attempts: 0,
        }
    }

    pub fn register_task(&mut self, task: Task) { self.tasks.insert(task.id(), task); }
    pub fn set_root_task(&mut self, id: TaskId) { self.root_task = Some(id); }
    pub fn set_world_state(&mut self, state: WorldState) { self.world_state = state; }
    pub fn world_state_mut(&mut self) -> &mut WorldState { &mut self.world_state }

    pub fn generate_plan(&mut self) -> bool {
        let root = match self.root_task { Some(id) => id, None => return false };
        let mut plan_steps = Vec::new();
        let mut working_state = self.world_state.clone_state();
        let mut backtrack_count = 0u32;

        let success = self.decompose(root, &mut working_state, &mut plan_steps, 0, &mut backtrack_count);

        if success && !plan_steps.is_empty() {
            let total_cost: f32 = plan_steps.iter().map(|s| s.cost).sum();
            let total_dur: f32 = plan_steps.iter().map(|s| s.duration).sum();
            self.current_plan = Plan { steps: plan_steps, total_cost, total_duration: total_dur, current_step: 0, valid: true, backtrack_count };
            self.state = PlannerState::Executing;
            self.stats.plans_generated += 1;
            self.stats.total_backtrack_steps += backtrack_count;
            self.replan_attempts = 0;
            true
        } else {
            self.state = PlannerState::Failed;
            false
        }
    }

    fn decompose(&self, task_id: TaskId, state: &mut WorldState, plan: &mut Vec<PlanStep>, depth: usize, backtracks: &mut u32) -> bool {
        if depth >= MAX_PLAN_DEPTH || plan.len() >= MAX_PLAN_LENGTH { return false; }

        match self.tasks.get(&task_id) {
            Some(Task::Primitive(pt)) => {
                if pt.can_execute(state) {
                    plan.push(PlanStep { task_id: pt.id, task_name: pt.name.clone(), status: PlanTaskStatus::Pending, cost: pt.cost, duration: pt.duration });
                    pt.apply_effects(state);
                    true
                } else { false }
            }
            Some(Task::Compound(ct)) => {
                let methods: Vec<Method> = {
                    let mut ms: Vec<&Method> = ct.methods.iter().collect();
                    ms.sort_by(|a, b| b.priority.cmp(&a.priority));
                    ms.into_iter().cloned().collect()
                };
                for method in methods {
                    if !method.is_applicable(state) { continue; }
                    let saved_state = state.clone_state();
                    let saved_plan_len = plan.len();
                    let mut success = true;
                    for &subtask_id in &method.subtasks {
                        if !self.decompose(subtask_id, state, plan, depth + 1, backtracks) { success = false; break; }
                    }
                    if success { return true; }
                    // Backtrack.
                    *backtracks += 1;
                    if *backtracks > MAX_BACKTRACK_STEPS as u32 { return false; }
                    *state = saved_state;
                    plan.truncate(saved_plan_len);
                }
                false
            }
            None => false,
        }
    }

    pub fn update(&mut self) {
        match self.state {
            PlannerState::Executing => {
                if self.current_plan.is_complete() {
                    self.state = PlannerState::Idle;
                    self.stats.plans_completed += 1;
                } else if !self.current_plan.valid {
                    if self.replan_on_failure && self.replan_attempts < self.max_replan_attempts {
                        self.state = PlannerState::Replanning;
                        self.replan_attempts += 1;
                        self.stats.replans += 1;
                        self.generate_plan();
                    } else {
                        self.state = PlannerState::Failed;
                        self.stats.plans_failed += 1;
                    }
                }
            }
            PlannerState::Replanning => { self.generate_plan(); }
            _ => {}
        }
    }

    pub fn advance_plan(&mut self) { self.current_plan.advance(); }
    pub fn fail_current_step(&mut self) { self.current_plan.fail_current(); }
    pub fn current_plan(&self) -> &Plan { &self.current_plan }
    pub fn planner_state(&self) -> PlannerState { self.state }
    pub fn stats(&self) -> &PlannerStats { &self.stats }
    pub fn force_replan(&mut self) { self.state = PlannerState::Replanning; }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_task() {
        let task = PrimitiveTask::new(1, "Attack").with_precondition(Condition::IsTrue(0)).with_effect(Effect::set_bool(1, true));
        let mut state = WorldState::new();
        state.set_bool(0, true);
        assert!(task.can_execute(&state));
        task.apply_effects(&mut state);
        assert!(state.get_bool(1));
    }

    #[test]
    fn test_plan_generation() {
        let mut planner = HtnPlanner::new();
        planner.register_task(Task::Primitive(PrimitiveTask::new(1, "Move").with_precondition(Condition::Always).with_effect(Effect::set_bool(10, true))));
        planner.register_task(Task::Primitive(PrimitiveTask::new(2, "Attack").with_precondition(Condition::IsTrue(10)).with_effect(Effect::set_bool(11, true))));
        let mut compound = CompoundTask::new(3, "KillEnemy");
        compound.add_method(Method::new(0, "approach_and_attack", Condition::Always, vec![1, 2]));
        planner.register_task(Task::Compound(compound));
        planner.set_root_task(3);
        assert!(planner.generate_plan());
        assert_eq!(planner.current_plan().steps.len(), 2);
    }

    #[test]
    fn test_condition_evaluation() {
        let mut state = WorldState::new();
        state.set_bool(0, true);
        state.set_int(1, 5);
        assert!(Condition::IsTrue(0).evaluate(&state));
        assert!(Condition::GreaterThan(1, 3).evaluate(&state));
        assert!(!Condition::LessThan(1, 3).evaluate(&state));
        assert!(Condition::And(vec![Condition::IsTrue(0), Condition::GreaterThan(1, 3)]).evaluate(&state));
    }
}
