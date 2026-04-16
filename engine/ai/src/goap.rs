//! Goal-Oriented Action Planning (GOAP).
//!
//! Implements a GOAP planner that uses A* search over world states to find
//! optimal sequences of actions that achieve a desired goal. This is the same
//! approach used in games like F.E.A.R. and Shadow of Mordor.
//!
//! # Overview
//!
//! - **WorldState**: a set of key-value conditions describing the game world.
//! - **GOAPAction**: a possible action with preconditions, effects, and a cost.
//! - **GOAPGoal**: a desired world state with a priority.
//! - **GOAPPlanner**: uses A* to find the cheapest sequence of actions that
//!   transforms the current world state into the goal state.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

// ---------------------------------------------------------------------------
// WorldState
// ---------------------------------------------------------------------------

/// A world state is a collection of named properties, each either a boolean
/// or an integer value.
///
/// World states represent the AI's understanding of the game world. They are
/// compared to determine goal satisfaction and used by the A* planner as
/// nodes in the search graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldState {
    /// Boolean conditions (e.g., "has_weapon", "enemy_visible").
    pub booleans: HashMap<String, bool>,
    /// Integer conditions (e.g., "ammo_count", "health").
    pub integers: HashMap<String, i32>,
}

impl WorldState {
    /// Creates a new empty world state.
    pub fn new() -> Self {
        Self {
            booleans: HashMap::new(),
            integers: HashMap::new(),
        }
    }

    /// Sets a boolean property.
    pub fn set_bool(&mut self, key: impl Into<String>, value: bool) {
        self.booleans.insert(key.into(), value);
    }

    /// Gets a boolean property.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.booleans.get(key).copied()
    }

    /// Sets an integer property.
    pub fn set_int(&mut self, key: impl Into<String>, value: i32) {
        self.integers.insert(key.into(), value);
    }

    /// Gets an integer property.
    pub fn get_int(&self, key: &str) -> Option<i32> {
        self.integers.get(key).copied()
    }

    /// Check if this state satisfies all conditions in `goal`.
    ///
    /// A state satisfies a goal if, for every property in the goal, the
    /// state contains the same property with the same value.
    pub fn satisfies(&self, goal: &WorldState) -> bool {
        for (key, goal_val) in &goal.booleans {
            match self.booleans.get(key) {
                Some(val) if val == goal_val => {}
                _ => return false,
            }
        }
        for (key, goal_val) in &goal.integers {
            match self.integers.get(key) {
                Some(val) if val == goal_val => {}
                _ => return false,
            }
        }
        true
    }

    /// Count the number of unsatisfied conditions relative to the goal.
    ///
    /// This serves as the heuristic for A* search — the number of goal
    /// conditions not yet met.
    pub fn unsatisfied_count(&self, goal: &WorldState) -> u32 {
        let mut count = 0u32;
        for (key, goal_val) in &goal.booleans {
            match self.booleans.get(key) {
                Some(val) if val == goal_val => {}
                _ => count += 1,
            }
        }
        for (key, goal_val) in &goal.integers {
            match self.integers.get(key) {
                Some(val) if val == goal_val => {}
                _ => count += 1,
            }
        }
        count
    }

    /// Apply effects to this world state, producing a new state.
    ///
    /// Each effect overwrites the corresponding property in the state.
    pub fn apply_effects(&self, effects: &WorldState) -> WorldState {
        let mut result = self.clone();
        for (key, val) in &effects.booleans {
            result.booleans.insert(key.clone(), *val);
        }
        for (key, val) in &effects.integers {
            result.integers.insert(key.clone(), *val);
        }
        result
    }

    /// Check if all preconditions are met in this state.
    pub fn meets_preconditions(&self, preconditions: &WorldState) -> bool {
        self.satisfies(preconditions)
    }

    /// Returns a hash key for use in closed-set lookups.
    ///
    /// Since `HashMap` itself doesn't implement `Hash`, we produce a
    /// deterministic string representation.
    pub fn hash_key(&self) -> String {
        let mut parts = Vec::new();

        let mut bool_keys: Vec<&String> = self.booleans.keys().collect();
        bool_keys.sort();
        for key in bool_keys {
            parts.push(format!("b:{}={}", key, self.booleans[key]));
        }

        let mut int_keys: Vec<&String> = self.integers.keys().collect();
        int_keys.sort();
        for key in int_keys {
            parts.push(format!("i:{}={}", key, self.integers[key]));
        }

        parts.join("|")
    }

    /// Merge another world state into this one (other's values take precedence).
    pub fn merge(&mut self, other: &WorldState) {
        for (k, v) in &other.booleans {
            self.booleans.insert(k.clone(), *v);
        }
        for (k, v) in &other.integers {
            self.integers.insert(k.clone(), *v);
        }
    }

    /// Returns true if the state has no properties.
    pub fn is_empty(&self) -> bool {
        self.booleans.is_empty() && self.integers.is_empty()
    }

    /// Returns the total number of properties.
    pub fn len(&self) -> usize {
        self.booleans.len() + self.integers.len()
    }
}

impl Default for WorldState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WorldState Builder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing world states.
pub struct WorldStateBuilder {
    state: WorldState,
}

impl WorldStateBuilder {
    /// Start building a new world state.
    pub fn new() -> Self {
        Self {
            state: WorldState::new(),
        }
    }

    /// Add a boolean condition.
    pub fn with_bool(mut self, key: impl Into<String>, value: bool) -> Self {
        self.state.set_bool(key, value);
        self
    }

    /// Add an integer condition.
    pub fn with_int(mut self, key: impl Into<String>, value: i32) -> Self {
        self.state.set_int(key, value);
        self
    }

    /// Build the world state.
    pub fn build(self) -> WorldState {
        self.state
    }
}

impl Default for WorldStateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GOAPAction
// ---------------------------------------------------------------------------

/// An action in the GOAP system.
///
/// Each action has preconditions that must be met before it can be used,
/// effects that describe how it changes the world state, and a cost that
/// the planner uses to find optimal plans.
#[derive(Debug, Clone)]
pub struct GOAPAction {
    /// Human-readable name of this action.
    pub name: String,
    /// Preconditions that must be true for this action to be usable.
    pub preconditions: WorldState,
    /// Effects that this action has on the world state when executed.
    pub effects: WorldState,
    /// Cost of executing this action (lower = preferred by planner).
    pub cost: f32,
    /// Whether this action is currently available (can be disabled at runtime).
    pub enabled: bool,
}

impl GOAPAction {
    /// Creates a new GOAP action.
    pub fn new(name: impl Into<String>, cost: f32) -> Self {
        Self {
            name: name.into(),
            preconditions: WorldState::new(),
            effects: WorldState::new(),
            cost,
            enabled: true,
        }
    }

    /// Sets the preconditions for this action.
    pub fn with_preconditions(mut self, preconditions: WorldState) -> Self {
        self.preconditions = preconditions;
        self
    }

    /// Sets the effects of this action.
    pub fn with_effects(mut self, effects: WorldState) -> Self {
        self.effects = effects;
        self
    }

    /// Adds a boolean precondition.
    pub fn with_precondition_bool(mut self, key: impl Into<String>, value: bool) -> Self {
        self.preconditions.set_bool(key, value);
        self
    }

    /// Adds a boolean effect.
    pub fn with_effect_bool(mut self, key: impl Into<String>, value: bool) -> Self {
        self.effects.set_bool(key, value);
        self
    }

    /// Adds an integer precondition.
    pub fn with_precondition_int(mut self, key: impl Into<String>, value: i32) -> Self {
        self.preconditions.set_int(key, value);
        self
    }

    /// Adds an integer effect.
    pub fn with_effect_int(mut self, key: impl Into<String>, value: i32) -> Self {
        self.effects.set_int(key, value);
        self
    }

    /// Check if this action can be applied in the given world state.
    pub fn is_applicable(&self, state: &WorldState) -> bool {
        self.enabled && state.meets_preconditions(&self.preconditions)
    }

    /// Apply this action's effects to a world state, returning the new state.
    pub fn apply(&self, state: &WorldState) -> WorldState {
        state.apply_effects(&self.effects)
    }
}

// ---------------------------------------------------------------------------
// GOAPGoal
// ---------------------------------------------------------------------------

/// A goal for the GOAP planner.
///
/// Goals describe a desired world state and have a priority that determines
/// which goal the agent should pursue when multiple goals are available.
#[derive(Debug, Clone)]
pub struct GOAPGoal {
    /// Human-readable name.
    pub name: String,
    /// The target world state conditions that must all be satisfied.
    pub target_state: WorldState,
    /// Priority of this goal (higher = more important).
    pub priority: f32,
    /// Whether this goal is currently active.
    pub active: bool,
}

impl GOAPGoal {
    /// Creates a new goal.
    pub fn new(name: impl Into<String>, target_state: WorldState, priority: f32) -> Self {
        Self {
            name: name.into(),
            target_state,
            priority,
            active: true,
        }
    }

    /// Check if the given state satisfies this goal.
    pub fn is_satisfied(&self, state: &WorldState) -> bool {
        state.satisfies(&self.target_state)
    }
}

// ---------------------------------------------------------------------------
// A* Search Node
// ---------------------------------------------------------------------------

/// A node in the GOAP A* search graph.
#[derive(Clone)]
struct PlanNode {
    /// The world state at this point in the plan.
    state: WorldState,
    /// Hash key for the state (for closed-set lookups).
    state_key: String,
    /// The sequence of action indices taken to reach this state.
    actions: Vec<usize>,
    /// Cost so far (g-cost).
    g_cost: f32,
    /// Estimated remaining cost (h-cost).
    h_cost: f32,
}

impl PlanNode {
    fn f_cost(&self) -> f32 {
        self.g_cost + self.h_cost
    }
}

impl PartialEq for PlanNode {
    fn eq(&self, other: &Self) -> bool {
        self.state_key == other.state_key
    }
}

impl Eq for PlanNode {}

impl PartialOrd for PlanNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PlanNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (lower f-cost = higher priority).
        other
            .f_cost()
            .partial_cmp(&self.f_cost())
            .unwrap_or(Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// GOAPPlan
// ---------------------------------------------------------------------------

/// A completed plan: a sequence of actions to achieve a goal.
#[derive(Debug, Clone)]
pub struct GOAPPlan {
    /// The ordered list of actions to execute.
    pub actions: Vec<GOAPAction>,
    /// Total cost of the plan.
    pub total_cost: f32,
    /// The goal this plan achieves.
    pub goal_name: String,
    /// The expected final world state after executing all actions.
    pub expected_final_state: WorldState,
}

impl GOAPPlan {
    /// Returns true if the plan has no actions (goal already satisfied).
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// Returns the number of actions in the plan.
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    /// Check if this plan is still valid given the current world state
    /// and available actions.
    ///
    /// Replays the plan's actions from the current state, verifying each
    /// action's preconditions are met.
    pub fn is_valid(&self, current_state: &WorldState) -> bool {
        let mut state = current_state.clone();
        for action in &self.actions {
            if !action.is_applicable(&state) {
                return false;
            }
            state = action.apply(&state);
        }
        true
    }

    /// Returns the names of actions in execution order.
    pub fn action_names(&self) -> Vec<&str> {
        self.actions.iter().map(|a| a.name.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// GOAPPlanner
// ---------------------------------------------------------------------------

/// The GOAP planner uses A* search to find optimal action sequences.
///
/// The search space is the set of all possible world states. Each edge
/// corresponds to applying an action (if its preconditions are met),
/// transforming the current state into a new state via the action's effects.
/// The heuristic is the count of unsatisfied goal conditions.
pub struct GOAPPlanner {
    /// Maximum number of nodes to expand before giving up.
    pub max_iterations: usize,
    /// Maximum plan depth (max number of actions in a plan).
    pub max_depth: usize,
    /// Cache of previously computed plans: (state_key + goal_name) -> plan.
    plan_cache: HashMap<String, GOAPPlan>,
    /// Whether to use plan caching.
    pub cache_enabled: bool,
}

impl GOAPPlanner {
    /// Creates a new GOAP planner with default settings.
    pub fn new() -> Self {
        Self {
            max_iterations: 10_000,
            max_depth: 20,
            plan_cache: HashMap::new(),
            cache_enabled: true,
        }
    }

    /// Sets the maximum iteration count.
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Sets the maximum plan depth.
    pub fn with_max_depth(mut self, max: usize) -> Self {
        self.max_depth = max;
        self
    }

    /// Enable or disable plan caching.
    pub fn with_cache(mut self, enabled: bool) -> Self {
        self.cache_enabled = enabled;
        self
    }

    /// Clear the plan cache.
    pub fn clear_cache(&mut self) {
        self.plan_cache.clear();
    }

    /// Plan a sequence of actions to achieve the given goal from the
    /// current world state.
    ///
    /// Uses A* search where:
    /// - Nodes are world states
    /// - Edges are actions (applied when preconditions are met)
    /// - Heuristic: count of unsatisfied goal conditions
    /// - Cost: sum of action costs along the path
    ///
    /// Returns `Some(GOAPPlan)` if a plan is found, `None` otherwise.
    pub fn plan(
        &mut self,
        current_state: &WorldState,
        goal: &GOAPGoal,
        available_actions: &[GOAPAction],
    ) -> Option<GOAPPlan> {
        profiling::scope!("GOAPPlanner::plan");

        // Check if the goal is already satisfied.
        if goal.is_satisfied(current_state) {
            return Some(GOAPPlan {
                actions: Vec::new(),
                total_cost: 0.0,
                goal_name: goal.name.clone(),
                expected_final_state: current_state.clone(),
            });
        }

        // Check cache.
        if self.cache_enabled {
            let cache_key = format!("{}|{}", current_state.hash_key(), goal.name);
            if let Some(cached_plan) = self.plan_cache.get(&cache_key) {
                if cached_plan.is_valid(current_state) {
                    return Some(cached_plan.clone());
                }
            }
        }

        // Filter to enabled actions only.
        let enabled_actions: Vec<&GOAPAction> = available_actions
            .iter()
            .filter(|a| a.enabled)
            .collect();

        if enabled_actions.is_empty() {
            return None;
        }

        // A* search.
        let start_key = current_state.hash_key();
        let h = current_state.unsatisfied_count(&goal.target_state) as f32;

        let mut open: BinaryHeap<PlanNode> = BinaryHeap::new();
        let mut closed: HashSet<String> = HashSet::new();

        open.push(PlanNode {
            state: current_state.clone(),
            state_key: start_key,
            actions: Vec::new(),
            g_cost: 0.0,
            h_cost: h,
        });

        let mut iterations = 0;

        while let Some(current) = open.pop() {
            iterations += 1;
            if iterations > self.max_iterations {
                log::warn!(
                    "GOAP planner reached iteration limit ({}) for goal '{}'",
                    self.max_iterations,
                    goal.name
                );
                return None;
            }

            // Check if the goal is satisfied.
            if goal.is_satisfied(&current.state) {
                let plan_actions: Vec<GOAPAction> = current
                    .actions
                    .iter()
                    .map(|&idx| enabled_actions[idx].clone())
                    .collect();

                let plan = GOAPPlan {
                    total_cost: current.g_cost,
                    goal_name: goal.name.clone(),
                    expected_final_state: current.state.clone(),
                    actions: plan_actions,
                };

                // Cache the plan.
                if self.cache_enabled {
                    let cache_key =
                        format!("{}|{}", current_state.hash_key(), goal.name);
                    self.plan_cache.insert(cache_key, plan.clone());
                }

                log::debug!(
                    "GOAP plan found for '{}' in {} iterations: {} actions, cost {:.2}",
                    goal.name,
                    iterations,
                    plan.len(),
                    plan.total_cost
                );

                return Some(plan);
            }

            if closed.contains(&current.state_key) {
                continue;
            }
            closed.insert(current.state_key.clone());

            // Enforce max depth.
            if current.actions.len() >= self.max_depth {
                continue;
            }

            // Expand: try each available action.
            for (action_idx, action) in enabled_actions.iter().enumerate() {
                if !action.is_applicable(&current.state) {
                    continue;
                }

                let new_state = action.apply(&current.state);
                let new_key = new_state.hash_key();

                if closed.contains(&new_key) {
                    continue;
                }

                let new_g = current.g_cost + action.cost;
                let new_h = new_state.unsatisfied_count(&goal.target_state) as f32;

                let mut new_actions = current.actions.clone();
                new_actions.push(action_idx);

                open.push(PlanNode {
                    state: new_state,
                    state_key: new_key,
                    actions: new_actions,
                    g_cost: new_g,
                    h_cost: new_h,
                });
            }
        }

        log::debug!(
            "GOAP planner: no plan found for goal '{}' after {} iterations",
            goal.name,
            iterations
        );
        None
    }

    /// Select the highest-priority goal and plan for it.
    pub fn plan_best_goal(
        &mut self,
        current_state: &WorldState,
        goals: &[GOAPGoal],
        available_actions: &[GOAPAction],
    ) -> Option<GOAPPlan> {
        // Sort goals by priority (descending).
        let mut sorted_goals: Vec<&GOAPGoal> = goals
            .iter()
            .filter(|g| g.active && !g.is_satisfied(current_state))
            .collect();
        sorted_goals.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(Ordering::Equal)
        });

        // Try to plan for each goal in priority order.
        for goal in sorted_goals {
            if let Some(plan) = self.plan(current_state, goal, available_actions) {
                return Some(plan);
            }
        }

        None
    }
}

impl Default for GOAPPlanner {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GOAPAgent
// ---------------------------------------------------------------------------

/// An agent that uses GOAP for decision making.
///
/// Maintains a current plan, tracks plan validity, and triggers replanning
/// when the world state changes enough to invalidate the current plan.
pub struct GOAPAgent {
    /// The agent's current world state knowledge.
    pub world_state: WorldState,
    /// Available actions for this agent.
    pub actions: Vec<GOAPAction>,
    /// Goals for this agent.
    pub goals: Vec<GOAPGoal>,
    /// The current plan being executed.
    pub current_plan: Option<GOAPPlan>,
    /// Index of the current action being executed in the plan.
    pub current_action_index: usize,
    /// Time since the last replan.
    pub time_since_replan: f32,
    /// How often to check plan validity and potentially replan (seconds).
    pub replan_interval: f32,
    /// The planner instance.
    planner: GOAPPlanner,
    /// Whether the agent needs to replan on the next update.
    needs_replan: bool,
}

impl GOAPAgent {
    /// Creates a new GOAP agent.
    pub fn new() -> Self {
        Self {
            world_state: WorldState::new(),
            actions: Vec::new(),
            goals: Vec::new(),
            current_plan: None,
            current_action_index: 0,
            time_since_replan: 0.0,
            replan_interval: 1.0,
            planner: GOAPPlanner::new(),
            needs_replan: true,
        }
    }

    /// Set the replan interval.
    pub fn with_replan_interval(mut self, interval: f32) -> Self {
        self.replan_interval = interval;
        self
    }

    /// Add an action to the agent's repertoire.
    pub fn add_action(&mut self, action: GOAPAction) {
        self.actions.push(action);
    }

    /// Add a goal to the agent.
    pub fn add_goal(&mut self, goal: GOAPGoal) {
        self.goals.push(goal);
    }

    /// Set a boolean value in the agent's world state.
    pub fn set_world_bool(&mut self, key: impl Into<String>, value: bool) {
        self.world_state.set_bool(key, value);
    }

    /// Set an integer value in the agent's world state.
    pub fn set_world_int(&mut self, key: impl Into<String>, value: i32) {
        self.world_state.set_int(key, value);
    }

    /// Force the agent to replan on the next update.
    pub fn request_replan(&mut self) {
        self.needs_replan = true;
    }

    /// Update the agent. Call once per frame.
    ///
    /// Checks if the current plan is still valid, replans if necessary,
    /// and returns the name of the current action to execute.
    pub fn update(&mut self, dt: f32) -> Option<&str> {
        self.time_since_replan += dt;

        // Check if we need to replan.
        let should_replan = self.needs_replan
            || self.current_plan.is_none()
            || self.time_since_replan >= self.replan_interval;

        if should_replan {
            // Check if current plan is still valid.
            let plan_invalid = match &self.current_plan {
                Some(plan) => {
                    // Check remaining actions from current index onward.
                    let remaining_actions: Vec<GOAPAction> = plan.actions
                        [self.current_action_index..]
                        .to_vec();
                    let remaining_plan = GOAPPlan {
                        actions: remaining_actions,
                        total_cost: 0.0,
                        goal_name: plan.goal_name.clone(),
                        expected_final_state: plan.expected_final_state.clone(),
                    };
                    !remaining_plan.is_valid(&self.world_state)
                }
                None => true,
            };

            if plan_invalid {
                self.replan();
            }

            self.time_since_replan = 0.0;
            self.needs_replan = false;
        }

        // Return the current action name.
        self.current_action_name()
    }

    /// Execute a replan using the current world state and goals.
    fn replan(&mut self) {
        let plan = self.planner.plan_best_goal(
            &self.world_state,
            &self.goals,
            &self.actions,
        );

        self.current_plan = plan;
        self.current_action_index = 0;
    }

    /// Returns the name of the current action being executed.
    pub fn current_action_name(&self) -> Option<&str> {
        self.current_plan.as_ref().and_then(|plan| {
            plan.actions
                .get(self.current_action_index)
                .map(|a| a.name.as_str())
        })
    }

    /// Signal that the current action has completed successfully.
    ///
    /// The agent's world state is updated with the action's effects,
    /// and the agent moves to the next action in the plan.
    pub fn action_completed(&mut self) {
        if let Some(ref plan) = self.current_plan {
            if self.current_action_index < plan.actions.len() {
                let action = &plan.actions[self.current_action_index];
                self.world_state = action.apply(&self.world_state);
                self.current_action_index += 1;
            }
        }
    }

    /// Signal that the current action has failed.
    ///
    /// The current plan is invalidated and a replan will occur on the next update.
    pub fn action_failed(&mut self) {
        self.current_plan = None;
        self.current_action_index = 0;
        self.needs_replan = true;
    }

    /// Check if the current plan is complete (all actions executed).
    pub fn is_plan_complete(&self) -> bool {
        match &self.current_plan {
            Some(plan) => self.current_action_index >= plan.actions.len(),
            None => true,
        }
    }

    /// Check if the agent has a valid plan.
    pub fn has_plan(&self) -> bool {
        self.current_plan.is_some() && !self.is_plan_complete()
    }

    /// Get the full current plan.
    pub fn plan(&self) -> Option<&GOAPPlan> {
        self.current_plan.as_ref()
    }

    /// Access the underlying planner.
    pub fn planner(&self) -> &GOAPPlanner {
        &self.planner
    }

    /// Mutable access to the underlying planner.
    pub fn planner_mut(&mut self) -> &mut GOAPPlanner {
        &mut self.planner
    }
}

impl Default for GOAPAgent {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_state_creation() {
        let state = WorldState::new();
        assert!(state.is_empty());
        assert_eq!(state.len(), 0);
    }

    #[test]
    fn test_world_state_set_get() {
        let mut state = WorldState::new();
        state.set_bool("has_weapon", true);
        state.set_int("health", 100);

        assert_eq!(state.get_bool("has_weapon"), Some(true));
        assert_eq!(state.get_int("health"), Some(100));
        assert_eq!(state.get_bool("missing"), None);
    }

    #[test]
    fn test_world_state_satisfies() {
        let mut current = WorldState::new();
        current.set_bool("has_weapon", true);
        current.set_bool("enemy_visible", true);
        current.set_int("ammo", 10);

        let mut goal = WorldState::new();
        goal.set_bool("has_weapon", true);
        goal.set_bool("enemy_visible", true);

        assert!(current.satisfies(&goal));

        goal.set_bool("has_armor", true);
        assert!(!current.satisfies(&goal));
    }

    #[test]
    fn test_world_state_unsatisfied_count() {
        let mut current = WorldState::new();
        current.set_bool("a", true);
        current.set_bool("b", false);

        let mut goal = WorldState::new();
        goal.set_bool("a", true);
        goal.set_bool("b", true);
        goal.set_bool("c", true);

        // 'a' is satisfied, 'b' is not (false != true), 'c' is missing.
        assert_eq!(current.unsatisfied_count(&goal), 2);
    }

    #[test]
    fn test_world_state_apply_effects() {
        let mut state = WorldState::new();
        state.set_bool("has_weapon", false);
        state.set_int("ammo", 0);

        let mut effects = WorldState::new();
        effects.set_bool("has_weapon", true);
        effects.set_int("ammo", 10);

        let new_state = state.apply_effects(&effects);
        assert_eq!(new_state.get_bool("has_weapon"), Some(true));
        assert_eq!(new_state.get_int("ammo"), Some(10));
    }

    #[test]
    fn test_world_state_hash_key() {
        let mut s1 = WorldState::new();
        s1.set_bool("a", true);
        s1.set_int("x", 5);

        let mut s2 = WorldState::new();
        s2.set_bool("a", true);
        s2.set_int("x", 5);

        assert_eq!(s1.hash_key(), s2.hash_key());

        s2.set_int("x", 6);
        assert_ne!(s1.hash_key(), s2.hash_key());
    }

    #[test]
    fn test_world_state_builder() {
        let state = WorldStateBuilder::new()
            .with_bool("armed", true)
            .with_int("health", 50)
            .build();

        assert_eq!(state.get_bool("armed"), Some(true));
        assert_eq!(state.get_int("health"), Some(50));
    }

    #[test]
    fn test_goap_action_creation() {
        let action = GOAPAction::new("attack", 1.0)
            .with_precondition_bool("has_weapon", true)
            .with_precondition_bool("enemy_visible", true)
            .with_effect_bool("enemy_dead", true);

        assert_eq!(action.name, "attack");
        assert_eq!(action.cost, 1.0);
        assert!(action.enabled);
    }

    #[test]
    fn test_goap_action_applicable() {
        let action = GOAPAction::new("attack", 1.0)
            .with_precondition_bool("has_weapon", true);

        let mut state = WorldState::new();
        state.set_bool("has_weapon", true);
        assert!(action.is_applicable(&state));

        state.set_bool("has_weapon", false);
        assert!(!action.is_applicable(&state));
    }

    #[test]
    fn test_goap_action_apply() {
        let action = GOAPAction::new("pickup_weapon", 1.0)
            .with_effect_bool("has_weapon", true);

        let mut state = WorldState::new();
        state.set_bool("has_weapon", false);

        let new_state = action.apply(&state);
        assert_eq!(new_state.get_bool("has_weapon"), Some(true));
    }

    #[test]
    fn test_goap_goal_creation() {
        let goal_state = WorldStateBuilder::new()
            .with_bool("enemy_dead", true)
            .build();

        let goal = GOAPGoal::new("kill_enemy", goal_state, 1.0);
        assert_eq!(goal.name, "kill_enemy");
        assert_eq!(goal.priority, 1.0);
        assert!(goal.active);
    }

    #[test]
    fn test_goap_goal_satisfied() {
        let goal_state = WorldStateBuilder::new()
            .with_bool("enemy_dead", true)
            .build();
        let goal = GOAPGoal::new("kill_enemy", goal_state, 1.0);

        let mut state = WorldState::new();
        state.set_bool("enemy_dead", false);
        assert!(!goal.is_satisfied(&state));

        state.set_bool("enemy_dead", true);
        assert!(goal.is_satisfied(&state));
    }

    #[test]
    fn test_goap_planner_simple() {
        let mut planner = GOAPPlanner::new();

        // Current state: no weapon, enemy alive.
        let current = WorldStateBuilder::new()
            .with_bool("has_weapon", false)
            .with_bool("enemy_dead", false)
            .build();

        // Goal: enemy dead.
        let goal_state = WorldStateBuilder::new()
            .with_bool("enemy_dead", true)
            .build();
        let goal = GOAPGoal::new("kill_enemy", goal_state, 1.0);

        // Actions.
        let actions = vec![
            GOAPAction::new("pickup_weapon", 1.0)
                .with_effect_bool("has_weapon", true),
            GOAPAction::new("attack", 2.0)
                .with_precondition_bool("has_weapon", true)
                .with_effect_bool("enemy_dead", true),
        ];

        let plan = planner.plan(&current, &goal, &actions);
        assert!(plan.is_some());

        let plan = plan.unwrap();
        assert_eq!(plan.len(), 2);
        assert_eq!(plan.actions[0].name, "pickup_weapon");
        assert_eq!(plan.actions[1].name, "attack");
        assert!((plan.total_cost - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_goap_planner_already_satisfied() {
        let mut planner = GOAPPlanner::new();

        let current = WorldStateBuilder::new()
            .with_bool("enemy_dead", true)
            .build();

        let goal_state = WorldStateBuilder::new()
            .with_bool("enemy_dead", true)
            .build();
        let goal = GOAPGoal::new("kill_enemy", goal_state, 1.0);

        let plan = planner.plan(&current, &goal, &[]);
        assert!(plan.is_some());
        assert!(plan.unwrap().is_empty());
    }

    #[test]
    fn test_goap_planner_no_plan() {
        let mut planner = GOAPPlanner::new();

        let current = WorldStateBuilder::new()
            .with_bool("has_weapon", false)
            .build();

        let goal_state = WorldStateBuilder::new()
            .with_bool("enemy_dead", true)
            .build();
        let goal = GOAPGoal::new("kill_enemy", goal_state, 1.0);

        // Only action requires a weapon we can't get.
        let actions = vec![
            GOAPAction::new("attack", 1.0)
                .with_precondition_bool("has_weapon", true)
                .with_effect_bool("enemy_dead", true),
        ];

        let plan = planner.plan(&current, &goal, &actions);
        assert!(plan.is_none());
    }

    #[test]
    fn test_goap_planner_chooses_cheapest_plan() {
        let mut planner = GOAPPlanner::new();

        let current = WorldStateBuilder::new()
            .with_bool("enemy_dead", false)
            .with_bool("has_weapon", true)
            .with_bool("has_grenade", true)
            .build();

        let goal_state = WorldStateBuilder::new()
            .with_bool("enemy_dead", true)
            .build();
        let goal = GOAPGoal::new("kill_enemy", goal_state, 1.0);

        // Two ways to kill: attack (cost 5) or throw grenade (cost 2).
        let actions = vec![
            GOAPAction::new("attack", 5.0)
                .with_precondition_bool("has_weapon", true)
                .with_effect_bool("enemy_dead", true),
            GOAPAction::new("throw_grenade", 2.0)
                .with_precondition_bool("has_grenade", true)
                .with_effect_bool("enemy_dead", true),
        ];

        let plan = planner.plan(&current, &goal, &actions).unwrap();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan.actions[0].name, "throw_grenade");
        assert!((plan.total_cost - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_goap_planner_multi_step() {
        let mut planner = GOAPPlanner::new();

        // Start: nothing.
        let current = WorldStateBuilder::new()
            .with_bool("has_wood", false)
            .with_bool("has_axe", false)
            .with_bool("fire_lit", false)
            .build();

        // Goal: fire lit.
        let goal_state = WorldStateBuilder::new()
            .with_bool("fire_lit", true)
            .build();
        let goal = GOAPGoal::new("make_fire", goal_state, 1.0);

        let actions = vec![
            GOAPAction::new("find_axe", 1.0)
                .with_effect_bool("has_axe", true),
            GOAPAction::new("chop_wood", 2.0)
                .with_precondition_bool("has_axe", true)
                .with_effect_bool("has_wood", true),
            GOAPAction::new("light_fire", 1.0)
                .with_precondition_bool("has_wood", true)
                .with_effect_bool("fire_lit", true),
            GOAPAction::new("gather_sticks", 3.0)
                .with_effect_bool("has_wood", true),
        ];

        let plan = planner.plan(&current, &goal, &actions).unwrap();

        // Should choose the cheapest path:
        // find_axe(1) + chop_wood(2) + light_fire(1) = 4, which is cheaper
        // than gather_sticks(3) + light_fire(1) = 4. Both are equal cost;
        // A* should find one of them.
        assert!(plan.total_cost <= 4.0 + 0.01);
        assert!(plan.len() >= 2);

        // Verify the final state has fire_lit = true.
        assert_eq!(
            plan.expected_final_state.get_bool("fire_lit"),
            Some(true)
        );
    }

    #[test]
    fn test_goap_planner_disabled_action() {
        let mut planner = GOAPPlanner::new();

        let current = WorldStateBuilder::new()
            .with_bool("enemy_dead", false)
            .build();

        let goal_state = WorldStateBuilder::new()
            .with_bool("enemy_dead", true)
            .build();
        let goal = GOAPGoal::new("kill_enemy", goal_state, 1.0);

        let mut cheap_action = GOAPAction::new("instant_kill", 0.1)
            .with_effect_bool("enemy_dead", true);
        cheap_action.enabled = false;

        let actions = vec![
            cheap_action,
            GOAPAction::new("fight", 5.0)
                .with_effect_bool("enemy_dead", true),
        ];

        let plan = planner.plan(&current, &goal, &actions).unwrap();
        // Should skip the disabled instant_kill.
        assert_eq!(plan.actions[0].name, "fight");
    }

    #[test]
    fn test_goap_plan_validity() {
        let current = WorldStateBuilder::new()
            .with_bool("has_weapon", true)
            .build();

        let plan = GOAPPlan {
            actions: vec![
                GOAPAction::new("attack", 1.0)
                    .with_precondition_bool("has_weapon", true)
                    .with_effect_bool("enemy_dead", true),
            ],
            total_cost: 1.0,
            goal_name: "kill".into(),
            expected_final_state: WorldState::new(),
        };

        assert!(plan.is_valid(&current));

        let no_weapon = WorldState::new();
        assert!(!plan.is_valid(&no_weapon));
    }

    #[test]
    fn test_goap_plan_best_goal() {
        let mut planner = GOAPPlanner::new();

        let current = WorldStateBuilder::new()
            .with_bool("hungry", true)
            .with_bool("tired", true)
            .build();

        let goals = vec![
            GOAPGoal::new(
                "eat",
                WorldStateBuilder::new().with_bool("hungry", false).build(),
                2.0, // Higher priority
            ),
            GOAPGoal::new(
                "sleep",
                WorldStateBuilder::new().with_bool("tired", false).build(),
                1.0,
            ),
        ];

        let actions = vec![
            GOAPAction::new("find_food", 1.0)
                .with_effect_bool("hungry", false),
            GOAPAction::new("go_to_bed", 1.0)
                .with_effect_bool("tired", false),
        ];

        let plan = planner.plan_best_goal(&current, &goals, &actions).unwrap();
        // Should plan for the higher priority goal (eat).
        assert_eq!(plan.goal_name, "eat");
    }

    #[test]
    fn test_goap_plan_cache() {
        let mut planner = GOAPPlanner::new();
        planner.cache_enabled = true;

        let current = WorldStateBuilder::new()
            .with_bool("a", false)
            .build();

        let goal_state = WorldStateBuilder::new()
            .with_bool("a", true)
            .build();
        let goal = GOAPGoal::new("set_a", goal_state, 1.0);

        let actions = vec![
            GOAPAction::new("do_a", 1.0).with_effect_bool("a", true),
        ];

        // First plan.
        let plan1 = planner.plan(&current, &goal, &actions).unwrap();
        assert_eq!(plan1.len(), 1);

        // Second plan should use cache.
        let plan2 = planner.plan(&current, &goal, &actions).unwrap();
        assert_eq!(plan2.len(), 1);
        assert_eq!(plan2.actions[0].name, plan1.actions[0].name);
    }

    #[test]
    fn test_goap_planner_max_depth() {
        let mut planner = GOAPPlanner::new().with_max_depth(2);

        let current = WorldStateBuilder::new()
            .with_bool("a", false)
            .with_bool("b", false)
            .with_bool("c", false)
            .build();

        let goal_state = WorldStateBuilder::new()
            .with_bool("c", true)
            .build();
        let goal = GOAPGoal::new("get_c", goal_state, 1.0);

        // Requires 3 steps: a -> b -> c, but max depth is 2.
        let actions = vec![
            GOAPAction::new("get_a", 1.0)
                .with_effect_bool("a", true),
            GOAPAction::new("get_b", 1.0)
                .with_precondition_bool("a", true)
                .with_effect_bool("b", true),
            GOAPAction::new("get_c", 1.0)
                .with_precondition_bool("b", true)
                .with_effect_bool("c", true),
        ];

        let plan = planner.plan(&current, &goal, &actions);
        // Should fail because the plan requires 3 steps but max depth is 2.
        assert!(plan.is_none());
    }

    #[test]
    fn test_goap_agent_basic() {
        let mut agent = GOAPAgent::new();

        agent.set_world_bool("has_weapon", false);
        agent.set_world_bool("enemy_dead", false);

        agent.add_action(
            GOAPAction::new("pickup_weapon", 1.0)
                .with_effect_bool("has_weapon", true),
        );
        agent.add_action(
            GOAPAction::new("attack", 2.0)
                .with_precondition_bool("has_weapon", true)
                .with_effect_bool("enemy_dead", true),
        );

        agent.add_goal(GOAPGoal::new(
            "kill_enemy",
            WorldStateBuilder::new()
                .with_bool("enemy_dead", true)
                .build(),
            1.0,
        ));

        // First update should trigger planning.
        let action = agent.update(0.016);
        assert!(action.is_some());
        assert_eq!(action.unwrap(), "pickup_weapon");
        assert!(agent.has_plan());

        // Complete the first action.
        agent.action_completed();
        let action = agent.current_action_name();
        assert_eq!(action, Some("attack"));

        // Complete the second action.
        agent.action_completed();
        assert!(agent.is_plan_complete());
    }

    #[test]
    fn test_goap_agent_replan_on_failure() {
        let mut agent = GOAPAgent::new();

        agent.set_world_bool("has_weapon", true);
        agent.set_world_bool("enemy_dead", false);

        agent.add_action(
            GOAPAction::new("attack", 1.0)
                .with_precondition_bool("has_weapon", true)
                .with_effect_bool("enemy_dead", true),
        );

        agent.add_goal(GOAPGoal::new(
            "kill_enemy",
            WorldStateBuilder::new()
                .with_bool("enemy_dead", true)
                .build(),
            1.0,
        ));

        agent.update(0.016);
        assert_eq!(agent.current_action_name(), Some("attack"));

        // Simulate failure.
        agent.action_failed();
        assert!(!agent.has_plan());

        // Next update should replan.
        agent.update(0.016);
        assert_eq!(agent.current_action_name(), Some("attack"));
    }

    #[test]
    fn test_goap_action_names() {
        let plan = GOAPPlan {
            actions: vec![
                GOAPAction::new("step1", 1.0),
                GOAPAction::new("step2", 2.0),
                GOAPAction::new("step3", 3.0),
            ],
            total_cost: 6.0,
            goal_name: "test".into(),
            expected_final_state: WorldState::new(),
        };

        assert_eq!(plan.action_names(), vec!["step1", "step2", "step3"]);
    }

    #[test]
    fn test_world_state_merge() {
        let mut a = WorldStateBuilder::new()
            .with_bool("x", true)
            .with_int("n", 1)
            .build();

        let b = WorldStateBuilder::new()
            .with_bool("y", true)
            .with_int("n", 2)
            .build();

        a.merge(&b);
        assert_eq!(a.get_bool("x"), Some(true));
        assert_eq!(a.get_bool("y"), Some(true));
        assert_eq!(a.get_int("n"), Some(2)); // b's value takes precedence
    }
}
