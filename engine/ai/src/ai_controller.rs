// engine/ai/src/ai_controller.rs
//
// AI controller framework for the Genovo engine.
//
// Provides the core sense-think-act loop for AI agents:
//
// - Sense: gather perception data from the environment.
// - Think: evaluate behaviors and select actions via behavior selection.
// - Act: execute the chosen action and update the agent state.
// - Per-agent blackboard for working memory.
// - Behavior selection from multiple strategies (BT, utility, GOAP).
// - Action execution with interruption and preemption support.
// - Perception integration with configurable sense modalities.
// - Debug visualization hooks for AI state inspection.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of managed AI agents.
const MAX_AGENTS: usize = 1024;

/// Default think rate (ticks per second).
const DEFAULT_THINK_RATE: f32 = 10.0;

/// Maximum blackboard entries per agent.
const MAX_BLACKBOARD_ENTRIES: usize = 128;

/// Default perception range.
const DEFAULT_PERCEPTION_RANGE: f32 = 30.0;

/// Maximum concurrent actions per agent.
const MAX_CONCURRENT_ACTIONS: usize = 4;

// ---------------------------------------------------------------------------
// Agent ID
// ---------------------------------------------------------------------------

/// Unique identifier for an AI agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AgentId(pub u64);

/// Unique identifier for an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActionId(pub u32);

// ---------------------------------------------------------------------------
// Blackboard
// ---------------------------------------------------------------------------

/// Value types stored in the blackboard.
#[derive(Debug, Clone)]
pub enum BlackboardValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Vector3([f32; 3]),
    EntityId(u64),
    List(Vec<BlackboardValue>),
}

impl BlackboardValue {
    pub fn as_bool(&self) -> bool {
        match self {
            Self::Bool(b) => *b,
            Self::Int(i) => *i != 0,
            Self::Float(f) => *f != 0.0,
            _ => false,
        }
    }

    pub fn as_float(&self) -> f64 {
        match self {
            Self::Bool(b) => if *b { 1.0 } else { 0.0 },
            Self::Int(i) => *i as f64,
            Self::Float(f) => *f,
            _ => 0.0,
        }
    }

    pub fn as_vector3(&self) -> [f32; 3] {
        match self {
            Self::Vector3(v) => *v,
            _ => [0.0; 3],
        }
    }

    pub fn as_entity_id(&self) -> u64 {
        match self {
            Self::EntityId(id) => *id,
            Self::Int(i) => *i as u64,
            _ => 0,
        }
    }
}

/// Per-agent blackboard for working memory.
#[derive(Debug, Clone)]
pub struct Blackboard {
    /// Key-value entries.
    pub entries: HashMap<String, BlackboardValue>,
    /// Expiry times for entries (key -> remaining seconds).
    pub expiry: HashMap<String, f32>,
    /// Whether any entry was modified since last check.
    pub dirty: bool,
}

impl Blackboard {
    /// Create a new empty blackboard.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            expiry: HashMap::new(),
            dirty: false,
        }
    }

    /// Set a value.
    pub fn set(&mut self, key: &str, value: BlackboardValue) {
        self.entries.insert(key.to_string(), value);
        self.dirty = true;
    }

    /// Set a value with an expiry time.
    pub fn set_with_expiry(&mut self, key: &str, value: BlackboardValue, ttl: f32) {
        self.set(key, value);
        self.expiry.insert(key.to_string(), ttl);
    }

    /// Get a value.
    pub fn get(&self, key: &str) -> Option<&BlackboardValue> {
        self.entries.get(key)
    }

    /// Get a boolean with default.
    pub fn get_bool(&self, key: &str, default: bool) -> bool {
        self.entries.get(key).map(|v| v.as_bool()).unwrap_or(default)
    }

    /// Get a float with default.
    pub fn get_float(&self, key: &str, default: f64) -> f64 {
        self.entries.get(key).map(|v| v.as_float()).unwrap_or(default)
    }

    /// Get a vector3 with default.
    pub fn get_vector3(&self, key: &str) -> Option<[f32; 3]> {
        self.entries.get(key).map(|v| v.as_vector3())
    }

    /// Remove a value.
    pub fn remove(&mut self, key: &str) {
        self.entries.remove(key);
        self.expiry.remove(key);
        self.dirty = true;
    }

    /// Check if a key exists.
    pub fn has(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.expiry.clear();
        self.dirty = true;
    }

    /// Update expiry timers and remove expired entries.
    pub fn update_expiry(&mut self, dt: f32) {
        let mut expired = Vec::new();
        for (key, ttl) in &mut self.expiry {
            *ttl -= dt;
            if *ttl <= 0.0 {
                expired.push(key.clone());
            }
        }
        for key in expired {
            self.entries.remove(&key);
            self.expiry.remove(&key);
            self.dirty = true;
        }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Perception
// ---------------------------------------------------------------------------

/// A perceived entity in the environment.
#[derive(Debug, Clone)]
pub struct PerceivedEntity {
    /// Entity ID.
    pub entity_id: u64,
    /// World position.
    pub position: [f32; 3],
    /// Velocity.
    pub velocity: [f32; 3],
    /// Distance from the agent.
    pub distance: f32,
    /// Awareness level (0.0 = unaware, 1.0 = fully aware).
    pub awareness: f32,
    /// Time since last seen.
    pub time_since_seen: f32,
    /// Whether currently visible.
    pub visible: bool,
    /// Whether currently audible.
    pub audible: bool,
    /// Threat level (0.0 = none, 1.0 = maximum threat).
    pub threat_level: f32,
    /// Faction relationship.
    pub relationship: Relationship,
    /// Last known position.
    pub last_known_position: [f32; 3],
}

/// Relationship between the agent and a perceived entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relationship {
    Friendly,
    Neutral,
    Hostile,
    Unknown,
}

/// Perception state for an agent.
#[derive(Debug, Clone)]
pub struct PerceptionState {
    /// All perceived entities.
    pub entities: Vec<PerceivedEntity>,
    /// Perception range.
    pub range: f32,
    /// Field of view (in degrees, for cone-of-vision).
    pub fov: f32,
    /// Whether hearing is enabled.
    pub hearing_enabled: bool,
    /// Hearing range.
    pub hearing_range: f32,
    /// Current alert level.
    pub alert_level: AlertLevel,
    /// Closest hostile entity.
    pub closest_hostile: Option<u64>,
    /// Number of visible hostiles.
    pub visible_hostile_count: u32,
}

impl Default for PerceptionState {
    fn default() -> Self {
        Self {
            entities: Vec::new(),
            range: DEFAULT_PERCEPTION_RANGE,
            fov: 120.0,
            hearing_enabled: true,
            hearing_range: 20.0,
            alert_level: AlertLevel::Unaware,
            closest_hostile: None,
            visible_hostile_count: 0,
        }
    }
}

/// Agent alert level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlertLevel {
    /// No threats detected.
    Unaware,
    /// Something suspicious noticed.
    Suspicious,
    /// Actively investigating.
    Investigating,
    /// Threat confirmed, engaged.
    Alert,
    /// In combat.
    Combat,
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/// Status of an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionStatus {
    /// Action has not started.
    Idle,
    /// Action is currently running.
    Running,
    /// Action completed successfully.
    Success,
    /// Action failed.
    Failed,
    /// Action was interrupted/cancelled.
    Cancelled,
}

/// An AI action being executed.
#[derive(Debug, Clone)]
pub struct AiAction {
    /// Action identifier.
    pub id: ActionId,
    /// Action name.
    pub name: String,
    /// Current status.
    pub status: ActionStatus,
    /// Elapsed time.
    pub elapsed: f32,
    /// Maximum duration (0 = unlimited).
    pub max_duration: f32,
    /// Priority (higher = more important).
    pub priority: i32,
    /// Whether this action can be interrupted.
    pub interruptible: bool,
    /// Action-specific parameters.
    pub params: HashMap<String, BlackboardValue>,
}

impl AiAction {
    /// Create a new action.
    pub fn new(id: ActionId, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            status: ActionStatus::Idle,
            elapsed: 0.0,
            max_duration: 0.0,
            priority: 0,
            interruptible: true,
            params: HashMap::new(),
        }
    }

    /// Set a parameter.
    pub fn with_param(mut self, key: &str, value: BlackboardValue) -> Self {
        self.params.insert(key.to_string(), value);
        self
    }

    /// Start the action.
    pub fn start(&mut self) {
        self.status = ActionStatus::Running;
        self.elapsed = 0.0;
    }

    /// Update the action.
    pub fn update(&mut self, dt: f32) {
        if self.status == ActionStatus::Running {
            self.elapsed += dt;
            if self.max_duration > 0.0 && self.elapsed >= self.max_duration {
                self.status = ActionStatus::Failed;
            }
        }
    }

    /// Complete the action.
    pub fn complete(&mut self) {
        self.status = ActionStatus::Success;
    }

    /// Fail the action.
    pub fn fail(&mut self) {
        self.status = ActionStatus::Failed;
    }

    /// Cancel the action.
    pub fn cancel(&mut self) {
        if self.interruptible {
            self.status = ActionStatus::Cancelled;
        }
    }

    /// Whether the action is still running.
    pub fn is_running(&self) -> bool {
        self.status == ActionStatus::Running
    }

    /// Whether the action is finished (success, failure, or cancelled).
    pub fn is_finished(&self) -> bool {
        matches!(self.status, ActionStatus::Success | ActionStatus::Failed | ActionStatus::Cancelled)
    }
}

// ---------------------------------------------------------------------------
// Behavior Selection Strategy
// ---------------------------------------------------------------------------

/// Strategy for selecting behaviors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BehaviorStrategy {
    /// Use behavior trees.
    BehaviorTree,
    /// Use utility AI.
    UtilityAI,
    /// Use GOAP.
    Goap,
    /// Use finite state machine.
    StateMachine,
    /// Hybrid: utility AI selects high-level behavior, BT executes.
    Hybrid,
}

// ---------------------------------------------------------------------------
// AI Agent
// ---------------------------------------------------------------------------

/// A complete AI agent with sense-think-act capabilities.
#[derive(Debug)]
pub struct AiAgent {
    /// Agent identifier.
    pub id: AgentId,
    /// Entity ID this agent controls.
    pub entity_id: u64,
    /// Agent name (for debugging).
    pub name: String,
    /// World position.
    pub position: [f32; 3],
    /// Forward direction.
    pub forward: [f32; 3],
    /// Blackboard (working memory).
    pub blackboard: Blackboard,
    /// Perception state.
    pub perception: PerceptionState,
    /// Current action stack.
    pub actions: Vec<AiAction>,
    /// Behavior selection strategy.
    pub strategy: BehaviorStrategy,
    /// Think rate (evaluations per second).
    pub think_rate: f32,
    /// Time since last think.
    pub time_since_think: f32,
    /// Whether this agent is active.
    pub active: bool,
    /// Whether this agent is alive.
    pub alive: bool,
    /// Health (0.0 to 1.0).
    pub health: f32,
    /// Team/faction.
    pub team: u32,
    /// Debug visualization enabled.
    pub debug_enabled: bool,
    /// Agent state for debugging.
    pub debug_state: String,
}

impl AiAgent {
    /// Create a new AI agent.
    pub fn new(id: AgentId, entity_id: u64, name: &str) -> Self {
        Self {
            id,
            entity_id,
            name: name.to_string(),
            position: [0.0; 3],
            forward: [0.0, 0.0, 1.0],
            blackboard: Blackboard::new(),
            perception: PerceptionState::default(),
            actions: Vec::new(),
            strategy: BehaviorStrategy::BehaviorTree,
            think_rate: DEFAULT_THINK_RATE,
            time_since_think: 0.0,
            active: true,
            alive: true,
            health: 1.0,
            team: 0,
            debug_enabled: false,
            debug_state: String::new(),
        }
    }

    /// Check if the agent should think this frame.
    pub fn should_think(&self, dt: f32) -> bool {
        if !self.active || !self.alive {
            return false;
        }
        self.time_since_think + dt >= 1.0 / self.think_rate
    }

    /// Get the current action (if any).
    pub fn current_action(&self) -> Option<&AiAction> {
        self.actions.last()
    }

    /// Get the current action mutably.
    pub fn current_action_mut(&mut self) -> Option<&mut AiAction> {
        self.actions.last_mut()
    }

    /// Push a new action onto the action stack.
    pub fn push_action(&mut self, action: AiAction) {
        // Interrupt current action if lower priority.
        if let Some(current) = self.actions.last_mut() {
            if current.is_running() && action.priority > current.priority {
                current.cancel();
            }
        }
        self.actions.push(action);
    }

    /// Pop the current action from the stack.
    pub fn pop_action(&mut self) -> Option<AiAction> {
        self.actions.pop()
    }

    /// Clean up finished actions.
    pub fn clean_finished_actions(&mut self) {
        self.actions.retain(|a| !a.is_finished());
    }

    /// Set the target entity on the blackboard.
    pub fn set_target(&mut self, entity_id: u64, position: [f32; 3]) {
        self.blackboard.set("target_entity", BlackboardValue::EntityId(entity_id));
        self.blackboard.set("target_position", BlackboardValue::Vector3(position));
    }

    /// Clear the target.
    pub fn clear_target(&mut self) {
        self.blackboard.remove("target_entity");
        self.blackboard.remove("target_position");
    }

    /// Get the target entity ID.
    pub fn target_entity(&self) -> Option<u64> {
        self.blackboard.get("target_entity").map(|v| v.as_entity_id())
    }

    /// Get the target position.
    pub fn target_position(&self) -> Option<[f32; 3]> {
        self.blackboard.get_vector3("target_position")
    }

    /// Distance to a point.
    pub fn distance_to(&self, point: [f32; 3]) -> f32 {
        let dx = point[0] - self.position[0];
        let dy = point[1] - self.position[1];
        let dz = point[2] - self.position[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Debug Visualization
// ---------------------------------------------------------------------------

/// Debug visualization data for an AI agent.
#[derive(Debug, Clone)]
pub struct AiDebugData {
    /// Agent ID.
    pub agent_id: AgentId,
    /// Agent position.
    pub position: [f32; 3],
    /// Forward direction.
    pub forward: [f32; 3],
    /// Perception cone (position, direction, angle, range).
    pub perception_cone: Option<([f32; 3], [f32; 3], f32, f32)>,
    /// Current action name.
    pub current_action: String,
    /// Alert level.
    pub alert_level: AlertLevel,
    /// Path points (if navigating).
    pub path_points: Vec<[f32; 3]>,
    /// Target line (agent -> target).
    pub target_line: Option<([f32; 3], [f32; 3])>,
    /// Blackboard summary.
    pub blackboard_summary: Vec<(String, String)>,
    /// Agent state text.
    pub state_text: String,
}

impl AiDebugData {
    /// Create debug data from an agent.
    pub fn from_agent(agent: &AiAgent) -> Self {
        let target_line = agent.target_position().map(|tp| (agent.position, tp));

        let bb_summary: Vec<(String, String)> = agent.blackboard.entries.iter()
            .take(10)
            .map(|(k, v)| (k.clone(), format!("{:?}", v)))
            .collect();

        Self {
            agent_id: agent.id,
            position: agent.position,
            forward: agent.forward,
            perception_cone: Some((
                agent.position,
                agent.forward,
                agent.perception.fov.to_radians(),
                agent.perception.range,
            )),
            current_action: agent.current_action().map(|a| a.name.clone()).unwrap_or_default(),
            alert_level: agent.perception.alert_level,
            path_points: Vec::new(),
            target_line,
            blackboard_summary: bb_summary,
            state_text: agent.debug_state.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// AI Controller System
// ---------------------------------------------------------------------------

/// The main AI controller that manages all agents.
#[derive(Debug)]
pub struct AiControllerSystem {
    /// All managed agents.
    pub agents: HashMap<AgentId, AiAgent>,
    /// Next agent ID.
    next_agent_id: u64,
    /// Global think rate multiplier.
    pub think_rate_multiplier: f32,
    /// Whether AI processing is paused.
    pub paused: bool,
    /// Debug mode.
    pub debug_mode: bool,
    /// Statistics.
    pub stats: AiControllerStats,
    /// Events generated.
    pub events: Vec<AiControllerEvent>,
}

/// Events from the AI controller.
#[derive(Debug, Clone)]
pub enum AiControllerEvent {
    /// Agent created.
    AgentCreated(AgentId),
    /// Agent destroyed.
    AgentDestroyed(AgentId),
    /// Agent started an action.
    ActionStarted { agent_id: AgentId, action_name: String },
    /// Agent completed an action.
    ActionCompleted { agent_id: AgentId, action_name: String },
    /// Agent alert level changed.
    AlertChanged { agent_id: AgentId, level: AlertLevel },
    /// Agent died.
    AgentDied(AgentId),
}

impl AiControllerSystem {
    /// Create a new AI controller system.
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            next_agent_id: 1,
            think_rate_multiplier: 1.0,
            paused: false,
            debug_mode: false,
            stats: AiControllerStats::default(),
            events: Vec::new(),
        }
    }

    /// Create a new agent.
    pub fn create_agent(&mut self, entity_id: u64, name: &str) -> AgentId {
        let id = AgentId(self.next_agent_id);
        self.next_agent_id += 1;

        let agent = AiAgent::new(id, entity_id, name);
        self.agents.insert(id, agent);
        self.events.push(AiControllerEvent::AgentCreated(id));
        id
    }

    /// Destroy an agent.
    pub fn destroy_agent(&mut self, id: AgentId) {
        if self.agents.remove(&id).is_some() {
            self.events.push(AiControllerEvent::AgentDestroyed(id));
        }
    }

    /// Get an agent by ID.
    pub fn get_agent(&self, id: AgentId) -> Option<&AiAgent> {
        self.agents.get(&id)
    }

    /// Get a mutable agent by ID.
    pub fn get_agent_mut(&mut self, id: AgentId) -> Option<&mut AiAgent> {
        self.agents.get_mut(&id)
    }

    /// Update all agents.
    pub fn update(&mut self, dt: f32) {
        if self.paused {
            return;
        }

        let effective_dt = dt * self.think_rate_multiplier;
        let mut agents_thought = 0u32;

        let agent_ids: Vec<AgentId> = self.agents.keys().copied().collect();
        for id in agent_ids {
            if let Some(agent) = self.agents.get_mut(&id) {
                agent.blackboard.update_expiry(dt);

                if agent.should_think(effective_dt) {
                    agent.time_since_think = 0.0;
                    agents_thought += 1;
                } else {
                    agent.time_since_think += effective_dt;
                }

                // Update current action.
                if let Some(action) = agent.actions.last_mut() {
                    action.update(dt);
                }
                agent.clean_finished_actions();
            }
        }

        self.stats.agents_active = self.agents.values().filter(|a| a.active).count() as u32;
        self.stats.agents_thinking = agents_thought;
        self.stats.total_agents = self.agents.len() as u32;
    }

    /// Get debug data for all agents.
    pub fn debug_data(&self) -> Vec<AiDebugData> {
        self.agents.values()
            .filter(|a| a.debug_enabled || self.debug_mode)
            .map(AiDebugData::from_agent)
            .collect()
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<AiControllerEvent> {
        std::mem::take(&mut self.events)
    }

    /// Find agents within a radius of a point.
    pub fn agents_in_range(&self, center: [f32; 3], radius: f32) -> Vec<AgentId> {
        let r_sq = radius * radius;
        self.agents.values()
            .filter(|a| {
                let dx = a.position[0] - center[0];
                let dy = a.position[1] - center[1];
                let dz = a.position[2] - center[2];
                dx * dx + dy * dy + dz * dz <= r_sq
            })
            .map(|a| a.id)
            .collect()
    }

    /// Find agents on a specific team.
    pub fn agents_on_team(&self, team: u32) -> Vec<AgentId> {
        self.agents.values()
            .filter(|a| a.team == team)
            .map(|a| a.id)
            .collect()
    }
}

/// Statistics for the AI controller.
#[derive(Debug, Clone, Default)]
pub struct AiControllerStats {
    /// Total registered agents.
    pub total_agents: u32,
    /// Active agents.
    pub agents_active: u32,
    /// Agents that performed a think this frame.
    pub agents_thinking: u32,
    /// Total actions in progress.
    pub actions_running: u32,
}
