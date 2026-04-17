// engine/ai/src/goal_system.rs
//
// Goal-driven AI: goal priorities, goal selection, goal decomposition
// into tasks, goal completion tracking, concurrent goals, and dynamic
// re-prioritization based on world state.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Unique identifier for a goal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GoalId(pub u32);

/// Unique identifier for a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(pub u32);

/// Unique identifier for an agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AgentId(pub u64);

/// The status of a goal or task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalStatus {
    /// Not yet started.
    Inactive,
    /// Currently being pursued.
    Active,
    /// Temporarily suspended for a higher-priority goal.
    Suspended,
    /// Successfully completed.
    Completed,
    /// Failed and cannot be retried.
    Failed,
    /// Cancelled by the system or player.
    Cancelled,
}

impl GoalStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active)
    }
}

/// Priority category for goals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GoalPriority {
    /// Background/optional goals.
    Low = 0,
    /// Normal gameplay goals.
    Medium = 1,
    /// Important objectives.
    High = 2,
    /// Survival-critical goals (e.g., flee, heal).
    Critical = 3,
}

// ---------------------------------------------------------------------------
// Task
// ---------------------------------------------------------------------------

/// A concrete action that contributes toward a goal.
#[derive(Debug, Clone)]
pub struct Task {
    pub id: TaskId,
    pub name: String,
    pub status: GoalStatus,
    pub progress: f32, // 0.0 to 1.0
    pub priority: i32,
    /// Estimated time to complete (seconds).
    pub estimated_duration: f32,
    /// Actual time elapsed on this task.
    pub elapsed: f32,
    /// Maximum time before timeout.
    pub timeout: f32,
    /// Dependencies: task IDs that must be completed first.
    pub dependencies: Vec<TaskId>,
    /// Requirements: key-value pairs that must be satisfied.
    pub requirements: HashMap<String, TaskRequirement>,
    /// Effects: changes to the world state when complete.
    pub effects: HashMap<String, TaskEffect>,
    /// Whether this task can be interrupted.
    pub interruptible: bool,
    /// Custom action identifier for the behavior system to execute.
    pub action_id: String,
    /// Custom parameters for the action.
    pub action_params: HashMap<String, String>,
}

impl Task {
    pub fn new(name: &str, action_id: &str) -> Self {
        Self {
            id: TaskId(0),
            name: name.to_string(),
            status: GoalStatus::Inactive,
            progress: 0.0,
            priority: 0,
            estimated_duration: 1.0,
            elapsed: 0.0,
            timeout: 60.0,
            dependencies: Vec::new(),
            requirements: HashMap::new(),
            effects: HashMap::new(),
            interruptible: true,
            action_id: action_id.to_string(),
            action_params: HashMap::new(),
        }
    }

    pub fn with_duration(mut self, duration: f32) -> Self {
        self.estimated_duration = duration;
        self
    }

    pub fn with_requirement(mut self, key: &str, req: TaskRequirement) -> Self {
        self.requirements.insert(key.to_string(), req);
        self
    }

    pub fn with_effect(mut self, key: &str, effect: TaskEffect) -> Self {
        self.effects.insert(key.to_string(), effect);
        self
    }

    pub fn with_dependency(mut self, dep: TaskId) -> Self {
        self.dependencies.push(dep);
        self
    }

    pub fn with_param(mut self, key: &str, value: &str) -> Self {
        self.action_params.insert(key.to_string(), value.to_string());
        self
    }

    pub fn is_complete(&self) -> bool { self.status == GoalStatus::Completed }
    pub fn is_failed(&self) -> bool { self.status == GoalStatus::Failed }
    pub fn is_timed_out(&self) -> bool { self.elapsed >= self.timeout }

    /// Check if all dependencies are completed.
    pub fn dependencies_met(&self, tasks: &HashMap<u32, Task>) -> bool {
        self.dependencies.iter().all(|dep_id| {
            tasks.get(&dep_id.0).map_or(false, |t| t.is_complete())
        })
    }
}

/// A requirement for a task to begin.
#[derive(Debug, Clone)]
pub enum TaskRequirement {
    /// A boolean condition that must be true.
    BoolTrue(String),
    BoolFalse(String),
    /// A float value must be above a threshold.
    FloatAbove(String, f32),
    /// A float value must be below a threshold.
    FloatBelow(String, f32),
    /// An entity must exist/be alive.
    EntityExists(u64),
    /// Must be within range of a location.
    InRange { target: [f32; 3], distance: f32 },
    /// Must have an item.
    HasItem(String),
}

/// An effect of completing a task.
#[derive(Debug, Clone)]
pub enum TaskEffect {
    SetBool(String, bool),
    SetFloat(String, f32),
    IncrementFloat(String, f32),
    RemoveKey(String),
    SpawnEntity(String),
    DestroyEntity(u64),
}

// ---------------------------------------------------------------------------
// Goal
// ---------------------------------------------------------------------------

/// A high-level goal that decomposes into tasks.
#[derive(Debug, Clone)]
pub struct Goal {
    pub id: GoalId,
    pub name: String,
    pub description: String,
    pub priority: GoalPriority,
    pub dynamic_priority: f32,
    pub status: GoalStatus,
    pub progress: f32,
    pub tasks: Vec<TaskId>,
    pub parallel_tasks: bool,
    /// Conditions under which this goal becomes relevant.
    pub activation_conditions: Vec<GoalCondition>,
    /// Conditions that cause the goal to be abandoned.
    pub abort_conditions: Vec<GoalCondition>,
    /// How many times this goal can be retried on failure.
    pub max_retries: u32,
    pub retry_count: u32,
    /// Cooldown after completion before the goal can activate again.
    pub cooldown: f32,
    pub cooldown_remaining: f32,
    /// Whether this goal can run concurrently with other goals.
    pub concurrent: bool,
    /// Time the goal has been active.
    pub active_time: f32,
    /// User data for custom logic.
    pub tags: Vec<String>,
}

impl Goal {
    pub fn new(name: &str, priority: GoalPriority) -> Self {
        Self {
            id: GoalId(0),
            name: name.to_string(),
            description: String::new(),
            priority,
            dynamic_priority: priority as u8 as f32,
            status: GoalStatus::Inactive,
            progress: 0.0,
            tasks: Vec::new(),
            parallel_tasks: false,
            activation_conditions: Vec::new(),
            abort_conditions: Vec::new(),
            max_retries: 0,
            retry_count: 0,
            cooldown: 0.0,
            cooldown_remaining: 0.0,
            concurrent: false,
            active_time: 0.0,
            tags: Vec::new(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn with_activation(mut self, cond: GoalCondition) -> Self {
        self.activation_conditions.push(cond);
        self
    }

    pub fn with_abort(mut self, cond: GoalCondition) -> Self {
        self.abort_conditions.push(cond);
        self
    }

    pub fn with_cooldown(mut self, seconds: f32) -> Self {
        self.cooldown = seconds;
        self
    }

    pub fn with_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    pub fn with_concurrent(mut self, concurrent: bool) -> Self {
        self.concurrent = concurrent;
        self
    }

    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    pub fn effective_priority(&self) -> f32 {
        self.dynamic_priority + self.priority as u8 as f32 * 10.0
    }
}

/// A condition for goal activation or abort.
#[derive(Debug, Clone)]
pub enum GoalCondition {
    BoolTrue(String),
    BoolFalse(String),
    FloatAbove(String, f32),
    FloatBelow(String, f32),
    HasTag(String),
    TimeElapsed(f32),
    Always,
}

// ---------------------------------------------------------------------------
// World state (for goal evaluation)
// ---------------------------------------------------------------------------

/// Simplified world state for goal condition evaluation.
#[derive(Debug, Clone, Default)]
pub struct GoalWorldState {
    pub bools: HashMap<String, bool>,
    pub floats: HashMap<String, f32>,
    pub tags: Vec<String>,
    pub agent_position: [f32; 3],
    pub time: f64,
}

impl GoalWorldState {
    pub fn set_bool(&mut self, key: &str, value: bool) {
        self.bools.insert(key.to_string(), value);
    }
    pub fn set_float(&mut self, key: &str, value: f32) {
        self.floats.insert(key.to_string(), value);
    }
    pub fn get_bool(&self, key: &str) -> bool {
        self.bools.get(key).copied().unwrap_or(false)
    }
    pub fn get_float(&self, key: &str) -> f32 {
        self.floats.get(key).copied().unwrap_or(0.0)
    }
}

fn evaluate_condition(cond: &GoalCondition, state: &GoalWorldState, active_time: f32) -> bool {
    match cond {
        GoalCondition::BoolTrue(key) => state.get_bool(key),
        GoalCondition::BoolFalse(key) => !state.get_bool(key),
        GoalCondition::FloatAbove(key, threshold) => state.get_float(key) > *threshold,
        GoalCondition::FloatBelow(key, threshold) => state.get_float(key) < *threshold,
        GoalCondition::HasTag(tag) => state.tags.contains(tag),
        GoalCondition::TimeElapsed(t) => active_time >= *t,
        GoalCondition::Always => true,
    }
}

// ---------------------------------------------------------------------------
// Goal manager (per-agent)
// ---------------------------------------------------------------------------

/// Events generated by the goal system.
#[derive(Debug, Clone)]
pub enum GoalEvent {
    GoalActivated { agent: AgentId, goal: GoalId },
    GoalCompleted { agent: AgentId, goal: GoalId },
    GoalFailed { agent: AgentId, goal: GoalId },
    GoalCancelled { agent: AgentId, goal: GoalId },
    GoalSuspended { agent: AgentId, goal: GoalId },
    TaskStarted { agent: AgentId, goal: GoalId, task: TaskId },
    TaskCompleted { agent: AgentId, goal: GoalId, task: TaskId },
    TaskFailed { agent: AgentId, goal: GoalId, task: TaskId },
}

/// Manages goals for a single AI agent.
pub struct GoalManager {
    agent_id: AgentId,
    goals: HashMap<u32, Goal>,
    tasks: HashMap<u32, Task>,
    next_goal_id: u32,
    next_task_id: u32,
    active_goals: Vec<GoalId>,
    events: Vec<GoalEvent>,
    max_concurrent_goals: usize,
}

impl GoalManager {
    pub fn new(agent_id: AgentId) -> Self {
        Self {
            agent_id,
            goals: HashMap::new(),
            tasks: HashMap::new(),
            next_goal_id: 1,
            next_task_id: 1,
            active_goals: Vec::new(),
            events: Vec::new(),
            max_concurrent_goals: 3,
        }
    }

    /// Add a goal to the manager.
    pub fn add_goal(&mut self, goal: Goal) -> GoalId {
        let id = GoalId(self.next_goal_id);
        self.next_goal_id += 1;
        let mut g = goal;
        g.id = id;
        self.goals.insert(id.0, g);
        id
    }

    /// Add a task to a goal.
    pub fn add_task(&mut self, goal_id: GoalId, task: Task) -> TaskId {
        let id = TaskId(self.next_task_id);
        self.next_task_id += 1;
        let mut t = task;
        t.id = id;
        self.tasks.insert(id.0, t);
        if let Some(goal) = self.goals.get_mut(&goal_id.0) {
            goal.tasks.push(id);
        }
        id
    }

    /// Remove a goal and its tasks.
    pub fn remove_goal(&mut self, goal_id: GoalId) {
        if let Some(goal) = self.goals.remove(&goal_id.0) {
            for task_id in &goal.tasks {
                self.tasks.remove(&task_id.0);
            }
            self.active_goals.retain(|id| *id != goal_id);
        }
    }

    /// Get a goal by ID.
    pub fn get_goal(&self, id: GoalId) -> Option<&Goal> {
        self.goals.get(&id.0)
    }

    /// Get a mutable goal by ID.
    pub fn get_goal_mut(&mut self, id: GoalId) -> Option<&mut Goal> {
        self.goals.get_mut(&id.0)
    }

    /// Get a task by ID.
    pub fn get_task(&self, id: TaskId) -> Option<&Task> {
        self.tasks.get(&id.0)
    }

    /// Complete a task.
    pub fn complete_task(&mut self, task_id: TaskId) {
        if let Some(task) = self.tasks.get_mut(&task_id.0) {
            task.status = GoalStatus::Completed;
            task.progress = 1.0;
        }
    }

    /// Fail a task.
    pub fn fail_task(&mut self, task_id: TaskId) {
        if let Some(task) = self.tasks.get_mut(&task_id.0) {
            task.status = GoalStatus::Failed;
        }
    }

    /// Set task progress.
    pub fn set_task_progress(&mut self, task_id: TaskId, progress: f32) {
        if let Some(task) = self.tasks.get_mut(&task_id.0) {
            task.progress = progress.clamp(0.0, 1.0);
        }
    }

    /// Get the current task to execute (highest priority active task).
    pub fn current_task(&self) -> Option<&Task> {
        for &goal_id in &self.active_goals {
            if let Some(goal) = self.goals.get(&goal_id.0) {
                for task_id in &goal.tasks {
                    if let Some(task) = self.tasks.get(&task_id.0) {
                        if task.status == GoalStatus::Active {
                            return Some(task);
                        }
                    }
                }
            }
        }
        None
    }

    /// Get all active goal IDs.
    pub fn active_goals(&self) -> &[GoalId] {
        &self.active_goals
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<GoalEvent> {
        std::mem::take(&mut self.events)
    }

    /// Update the goal system. Call once per tick.
    pub fn update(&mut self, dt: f32, world_state: &GoalWorldState) {
        // Update cooldowns.
        for goal in self.goals.values_mut() {
            if goal.cooldown_remaining > 0.0 {
                goal.cooldown_remaining -= dt;
            }
        }

        // Check abort conditions on active goals.
        let agent = self.agent_id;
        let mut to_abort = Vec::new();
        for &goal_id in &self.active_goals {
            if let Some(goal) = self.goals.get(&goal_id.0) {
                for cond in &goal.abort_conditions {
                    if evaluate_condition(cond, world_state, goal.active_time) {
                        to_abort.push(goal_id);
                        break;
                    }
                }
            }
        }
        for goal_id in to_abort {
            self.cancel_goal(goal_id);
        }

        // Evaluate activation conditions for inactive goals.
        let inactive_goals: Vec<GoalId> = self.goals.values()
            .filter(|g| g.status == GoalStatus::Inactive && g.cooldown_remaining <= 0.0)
            .filter(|g| {
                g.activation_conditions.is_empty()
                    || g.activation_conditions.iter().all(|c| evaluate_condition(c, world_state, 0.0))
            })
            .map(|g| g.id)
            .collect();

        // Select goals to activate based on priority.
        let mut candidates: Vec<(GoalId, f32)> = inactive_goals.iter()
            .filter_map(|&id| {
                self.goals.get(&id.0).map(|g| (id, g.effective_priority()))
            })
            .collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (goal_id, _) in candidates {
            if self.active_goals.len() >= self.max_concurrent_goals {
                // Check if we should preempt a lower-priority goal.
                let new_prio = self.goals.get(&goal_id.0).map(|g| g.effective_priority()).unwrap_or(0.0);
                if let Some(lowest) = self.active_goals.iter()
                    .filter_map(|&id| self.goals.get(&id.0).map(|g| (id, g.effective_priority())))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                {
                    if new_prio > lowest.1 {
                        self.suspend_goal(lowest.0);
                    } else {
                        continue;
                    }
                }
            }

            self.activate_goal(goal_id);
        }

        // Update active goals.
        let active_ids: Vec<GoalId> = self.active_goals.clone();
        for goal_id in active_ids {
            self.update_goal(goal_id, dt, world_state);
        }

        // Update task timeouts.
        for task in self.tasks.values_mut() {
            if task.status == GoalStatus::Active {
                task.elapsed += dt;
                if task.is_timed_out() {
                    task.status = GoalStatus::Failed;
                }
            }
        }
    }

    fn activate_goal(&mut self, goal_id: GoalId) {
        if let Some(goal) = self.goals.get_mut(&goal_id.0) {
            goal.status = GoalStatus::Active;
            goal.active_time = 0.0;
            self.active_goals.push(goal_id);
            self.events.push(GoalEvent::GoalActivated { agent: self.agent_id, goal: goal_id });

            // Activate the first task.
            if let Some(&first_task) = goal.tasks.first() {
                if let Some(task) = self.tasks.get_mut(&first_task.0) {
                    task.status = GoalStatus::Active;
                    self.events.push(GoalEvent::TaskStarted {
                        agent: self.agent_id, goal: goal_id, task: first_task,
                    });
                }
            }
        }
    }

    fn suspend_goal(&mut self, goal_id: GoalId) {
        if let Some(goal) = self.goals.get_mut(&goal_id.0) {
            goal.status = GoalStatus::Suspended;
            self.events.push(GoalEvent::GoalSuspended { agent: self.agent_id, goal: goal_id });

            // Suspend active tasks.
            for &task_id in &goal.tasks {
                if let Some(task) = self.tasks.get_mut(&task_id.0) {
                    if task.status == GoalStatus::Active {
                        task.status = GoalStatus::Suspended;
                    }
                }
            }
        }
        self.active_goals.retain(|&id| id != goal_id);
    }

    fn cancel_goal(&mut self, goal_id: GoalId) {
        if let Some(goal) = self.goals.get_mut(&goal_id.0) {
            goal.status = GoalStatus::Cancelled;
            self.events.push(GoalEvent::GoalCancelled { agent: self.agent_id, goal: goal_id });

            for &task_id in &goal.tasks {
                if let Some(task) = self.tasks.get_mut(&task_id.0) {
                    if !task.status.is_terminal() {
                        task.status = GoalStatus::Cancelled;
                    }
                }
            }
        }
        self.active_goals.retain(|&id| id != goal_id);
    }

    fn update_goal(&mut self, goal_id: GoalId, dt: f32, _world: &GoalWorldState) {
        let task_ids: Vec<TaskId> = self.goals.get(&goal_id.0)
            .map(|g| g.tasks.clone())
            .unwrap_or_default();

        if let Some(goal) = self.goals.get_mut(&goal_id.0) {
            goal.active_time += dt;
        }

        // Check if all tasks are complete.
        let all_complete = task_ids.iter().all(|tid| {
            self.tasks.get(&tid.0).map_or(true, |t| t.is_complete())
        });
        let any_failed = task_ids.iter().any(|tid| {
            self.tasks.get(&tid.0).map_or(false, |t| t.is_failed())
        });

        if all_complete {
            if let Some(goal) = self.goals.get_mut(&goal_id.0) {
                goal.status = GoalStatus::Completed;
                goal.progress = 1.0;
                goal.cooldown_remaining = goal.cooldown;
                self.events.push(GoalEvent::GoalCompleted { agent: self.agent_id, goal: goal_id });
            }
            self.active_goals.retain(|&id| id != goal_id);
            return;
        }

        if any_failed {
            let can_retry = self.goals.get(&goal_id.0)
                .map(|g| g.retry_count < g.max_retries)
                .unwrap_or(false);

            if can_retry {
                if let Some(goal) = self.goals.get_mut(&goal_id.0) {
                    goal.retry_count += 1;
                    // Reset failed tasks.
                    for &tid in &goal.tasks {
                        if let Some(task) = self.tasks.get_mut(&tid.0) {
                            if task.is_failed() {
                                task.status = GoalStatus::Inactive;
                                task.progress = 0.0;
                                task.elapsed = 0.0;
                            }
                        }
                    }
                }
            } else {
                if let Some(goal) = self.goals.get_mut(&goal_id.0) {
                    goal.status = GoalStatus::Failed;
                    self.events.push(GoalEvent::GoalFailed { agent: self.agent_id, goal: goal_id });
                }
                self.active_goals.retain(|&id| id != goal_id);
            }
            return;
        }

        // Advance to the next unstarted task if current is complete.
        let parallel = self.goals.get(&goal_id.0).map_or(false, |g| g.parallel_tasks);

        if parallel {
            // Activate all tasks that have met dependencies.
            for &tid in &task_ids {
                if let Some(task) = self.tasks.get(&tid.0) {
                    if task.status == GoalStatus::Inactive && task.dependencies_met(&self.tasks) {
                        let task = self.tasks.get_mut(&tid.0).unwrap();
                        task.status = GoalStatus::Active;
                        self.events.push(GoalEvent::TaskStarted {
                            agent: self.agent_id, goal: goal_id, task: tid,
                        });
                    }
                }
            }
        } else {
            // Sequential: find first non-complete task.
            for &tid in &task_ids {
                let status = self.tasks.get(&tid.0).map(|t| t.status);
                if status == Some(GoalStatus::Inactive) {
                    if let Some(task) = self.tasks.get_mut(&tid.0) {
                        task.status = GoalStatus::Active;
                        self.events.push(GoalEvent::TaskStarted {
                            agent: self.agent_id, goal: goal_id, task: tid,
                        });
                    }
                    break;
                }
                if status == Some(GoalStatus::Active) {
                    break;
                }
            }
        }

        // Update progress.
        if !task_ids.is_empty() {
            let completed = task_ids.iter()
                .filter(|tid| self.tasks.get(&tid.0).map_or(false, |t| t.is_complete()))
                .count();
            if let Some(goal) = self.goals.get_mut(&goal_id.0) {
                goal.progress = completed as f32 / task_ids.len() as f32;
            }
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
    fn test_basic_goal() {
        let mut mgr = GoalManager::new(AgentId(1));
        let goal_id = mgr.add_goal(
            Goal::new("find_food", GoalPriority::Medium)
                .with_activation(GoalCondition::FloatBelow("hunger".to_string(), 50.0))
        );
        let t1 = mgr.add_task(goal_id, Task::new("go_to_kitchen", "navigate"));
        let t2 = mgr.add_task(goal_id, Task::new("eat_food", "eat"));

        let mut world = GoalWorldState::default();
        world.set_float("hunger", 30.0);

        mgr.update(0.1, &world);
        assert!(!mgr.active_goals().is_empty());

        // Complete first task.
        mgr.complete_task(t1);
        mgr.update(0.1, &world);

        // Second task should now be active.
        let task2 = mgr.get_task(t2).unwrap();
        assert_eq!(task2.status, GoalStatus::Active);

        // Complete second task.
        mgr.complete_task(t2);
        mgr.update(0.1, &world);

        let goal = mgr.get_goal(goal_id).unwrap();
        assert_eq!(goal.status, GoalStatus::Completed);
    }

    #[test]
    fn test_priority_preemption() {
        let mut mgr = GoalManager::new(AgentId(1));
        mgr.max_concurrent_goals = 1;

        let low = mgr.add_goal(
            Goal::new("patrol", GoalPriority::Low)
                .with_activation(GoalCondition::Always)
        );
        mgr.add_task(low, Task::new("walk", "walk"));

        let high = mgr.add_goal(
            Goal::new("flee", GoalPriority::Critical)
                .with_activation(GoalCondition::BoolTrue("danger".to_string()))
        );
        mgr.add_task(high, Task::new("run", "run"));

        let mut world = GoalWorldState::default();
        mgr.update(0.1, &world);
        assert_eq!(mgr.active_goals().len(), 1);
        assert_eq!(mgr.active_goals()[0], low);

        // Trigger danger.
        world.set_bool("danger", true);
        mgr.update(0.1, &world);
        assert_eq!(mgr.active_goals()[0], high);
    }

    #[test]
    fn test_task_timeout() {
        let mut mgr = GoalManager::new(AgentId(1));
        let goal_id = mgr.add_goal(
            Goal::new("test", GoalPriority::Medium).with_activation(GoalCondition::Always)
        );
        let mut task = Task::new("wait", "idle");
        task.timeout = 1.0;
        let tid = mgr.add_task(goal_id, task);

        let world = GoalWorldState::default();
        mgr.update(0.1, &world);

        // Advance time past timeout.
        for _ in 0..20 {
            mgr.update(0.1, &world);
        }

        let t = mgr.get_task(tid).unwrap();
        assert_eq!(t.status, GoalStatus::Failed);
    }

    #[test]
    fn test_cooldown() {
        let mut mgr = GoalManager::new(AgentId(1));
        let goal_id = mgr.add_goal(
            Goal::new("greet", GoalPriority::Low)
                .with_activation(GoalCondition::Always)
                .with_cooldown(5.0)
        );
        let tid = mgr.add_task(goal_id, Task::new("say_hello", "talk"));

        let world = GoalWorldState::default();
        mgr.update(0.1, &world);
        mgr.complete_task(tid);
        mgr.update(0.1, &world);

        // Goal should be completed and on cooldown.
        let goal = mgr.get_goal(goal_id).unwrap();
        assert_eq!(goal.status, GoalStatus::Completed);
        assert!(goal.cooldown_remaining > 0.0);
    }
}
