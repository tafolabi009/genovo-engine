// engine/ai/src/behavior_coordinator.rs
//
// Multi-agent behavior coordination for the Genovo engine.
//
// Enables groups of AI agents to coordinate their actions:
//
// - **Group goals** -- Shared objectives that multiple agents work toward.
// - **Task assignment** -- Distribute tasks optimally among group members.
// - **Role rotation** -- Agents swap roles periodically for varied behavior.
// - **Synchronized actions** -- Coordinated breach, ambush, flanking maneuvers.
// - **Radio communication simulation** -- Agents share information with delays.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AgentId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GoalId(pub u32);

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Agent({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Agent role
// ---------------------------------------------------------------------------

/// Roles agents can assume within a group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AgentRole {
    Leader,
    Pointman,
    Flanker,
    Support,
    Sniper,
    Medic,
    Assault,
    Scout,
    Rearguard,
    Custom(u32),
}

impl fmt::Display for AgentRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Leader => write!(f, "Leader"),
            Self::Pointman => write!(f, "Pointman"),
            Self::Flanker => write!(f, "Flanker"),
            Self::Support => write!(f, "Support"),
            Self::Sniper => write!(f, "Sniper"),
            Self::Medic => write!(f, "Medic"),
            Self::Assault => write!(f, "Assault"),
            Self::Scout => write!(f, "Scout"),
            Self::Rearguard => write!(f, "Rearguard"),
            Self::Custom(id) => write!(f, "Custom({})", id),
        }
    }
}

// ---------------------------------------------------------------------------
// Group goal
// ---------------------------------------------------------------------------

/// Status of a group goal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

/// A shared goal for a group of agents.
#[derive(Debug, Clone)]
pub struct GroupGoal {
    pub id: GoalId,
    pub name: String,
    pub priority: i32,
    pub status: GoalStatus,
    pub target_position: Option<[f32; 3]>,
    pub target_entity: Option<u32>,
    pub required_agents: u32,
    pub assigned_agents: Vec<AgentId>,
    pub progress: f32,
    pub timeout: f32,
    pub elapsed: f32,
}

impl GroupGoal {
    pub fn new(id: GoalId, name: &str, priority: i32) -> Self {
        Self {
            id,
            name: name.to_string(),
            priority,
            status: GoalStatus::Pending,
            target_position: None,
            target_entity: None,
            required_agents: 1,
            assigned_agents: Vec::new(),
            progress: 0.0,
            timeout: 60.0,
            elapsed: 0.0,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.status == GoalStatus::Completed
    }

    pub fn is_active(&self) -> bool {
        matches!(self.status, GoalStatus::Pending | GoalStatus::InProgress)
    }

    pub fn has_enough_agents(&self) -> bool {
        self.assigned_agents.len() as u32 >= self.required_agents
    }
}

// ---------------------------------------------------------------------------
// Task
// ---------------------------------------------------------------------------

/// A task that can be assigned to an agent.
#[derive(Debug, Clone)]
pub struct CoordinationTask {
    pub id: TaskId,
    pub name: String,
    pub goal_id: GoalId,
    pub required_role: Option<AgentRole>,
    pub assigned_agent: Option<AgentId>,
    pub priority: i32,
    pub position: Option<[f32; 3]>,
    pub status: TaskStatus,
    pub dependencies: Vec<TaskId>,
    pub estimated_duration: f32,
    pub elapsed: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    Unassigned,
    Assigned,
    InProgress,
    WaitingForDependency,
    Completed,
    Failed,
}

impl CoordinationTask {
    pub fn new(id: TaskId, name: &str, goal_id: GoalId) -> Self {
        Self {
            id,
            name: name.to_string(),
            goal_id,
            required_role: None,
            assigned_agent: None,
            priority: 0,
            position: None,
            status: TaskStatus::Unassigned,
            dependencies: Vec::new(),
            estimated_duration: 5.0,
            elapsed: 0.0,
        }
    }

    pub fn is_ready(&self, completed_tasks: &[TaskId]) -> bool {
        self.dependencies.iter().all(|dep| completed_tasks.contains(dep))
    }
}

// ---------------------------------------------------------------------------
// Synchronized action
// ---------------------------------------------------------------------------

/// Type of synchronized group action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncActionType {
    CoordinatedBreach,
    PincerAttack,
    SimultaneousFire,
    FlankAndDistract,
    CoordinatedRetreat,
    SurroundTarget,
    StackAndClear,
    AmbushTrigger,
}

/// A synchronized action requiring multiple agents to act simultaneously.
#[derive(Debug, Clone)]
pub struct SyncAction {
    pub action_type: SyncActionType,
    pub participants: Vec<AgentId>,
    pub required_count: u32,
    pub ready_agents: Vec<AgentId>,
    pub trigger_position: [f32; 3],
    pub countdown: f32,
    pub countdown_active: bool,
    pub executed: bool,
    pub abort_on_detection: bool,
}

impl SyncAction {
    pub fn new(action_type: SyncActionType, trigger_pos: [f32; 3], required: u32) -> Self {
        Self {
            action_type,
            participants: Vec::new(),
            required_count: required,
            ready_agents: Vec::new(),
            trigger_position: trigger_pos,
            countdown: 3.0,
            countdown_active: false,
            executed: false,
            abort_on_detection: true,
        }
    }

    pub fn all_ready(&self) -> bool {
        self.ready_agents.len() as u32 >= self.required_count
    }

    pub fn mark_ready(&mut self, agent: AgentId) {
        if self.participants.contains(&agent) && !self.ready_agents.contains(&agent) {
            self.ready_agents.push(agent);
        }
    }

    pub fn start_countdown(&mut self) {
        if self.all_ready() && !self.countdown_active {
            self.countdown_active = true;
        }
    }

    pub fn update(&mut self, dt: f32) -> bool {
        if self.countdown_active && !self.executed {
            self.countdown -= dt;
            if self.countdown <= 0.0 {
                self.executed = true;
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Radio message
// ---------------------------------------------------------------------------

/// Simulated radio communication between agents.
#[derive(Debug, Clone)]
pub struct RadioMessage {
    pub sender: AgentId,
    pub recipients: Vec<AgentId>,
    pub message_type: RadioMessageType,
    pub data: RadioData,
    pub send_time: f64,
    pub delay: f32,
    pub delivered: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadioMessageType {
    EnemySpotted,
    RequestBackup,
    InPosition,
    Moving,
    Engaging,
    Retreating,
    TargetDown,
    AllClear,
    RegroupAt,
    CoverMe,
}

#[derive(Debug, Clone)]
pub struct RadioData {
    pub position: Option<[f32; 3]>,
    pub target_id: Option<u32>,
    pub urgency: f32,
    pub custom: HashMap<String, f32>,
}

impl RadioData {
    pub fn empty() -> Self {
        Self {
            position: None,
            target_id: None,
            urgency: 0.5,
            custom: HashMap::new(),
        }
    }

    pub fn at_position(pos: [f32; 3]) -> Self {
        Self {
            position: Some(pos),
            ..Self::empty()
        }
    }
}

// ---------------------------------------------------------------------------
// Agent state
// ---------------------------------------------------------------------------

/// Per-agent coordination state.
#[derive(Debug, Clone)]
pub struct AgentCoordState {
    pub id: AgentId,
    pub group: Option<GroupId>,
    pub role: AgentRole,
    pub position: [f32; 3],
    pub current_task: Option<TaskId>,
    pub ready_for_sync: bool,
    pub alive: bool,
    pub combat_effectiveness: f32,
    pub role_duration: f32,
    pub inbox: Vec<RadioMessage>,
}

impl AgentCoordState {
    pub fn new(id: AgentId) -> Self {
        Self {
            id,
            group: None,
            role: AgentRole::Assault,
            position: [0.0; 3],
            current_task: None,
            ready_for_sync: false,
            alive: true,
            combat_effectiveness: 1.0,
            role_duration: 0.0,
            inbox: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Role rotation
// ---------------------------------------------------------------------------

/// Configuration for automatic role rotation.
#[derive(Debug, Clone)]
pub struct RoleRotationConfig {
    pub enabled: bool,
    pub rotation_interval: f32,
    pub roles: Vec<AgentRole>,
    pub rotation_timer: f32,
}

impl Default for RoleRotationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            rotation_interval: 30.0,
            roles: vec![AgentRole::Assault, AgentRole::Flanker, AgentRole::Support],
            rotation_timer: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Agent group
// ---------------------------------------------------------------------------

/// A group of coordinated agents.
#[derive(Debug, Clone)]
pub struct AgentGroup {
    pub id: GroupId,
    pub name: String,
    pub members: Vec<AgentId>,
    pub leader: Option<AgentId>,
    pub goals: Vec<GroupGoal>,
    pub tasks: Vec<CoordinationTask>,
    pub sync_actions: Vec<SyncAction>,
    pub role_rotation: RoleRotationConfig,
    pub formation_center: [f32; 3],
    pub alert_level: f32,
}

impl AgentGroup {
    pub fn new(id: GroupId, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            members: Vec::new(),
            leader: None,
            goals: Vec::new(),
            tasks: Vec::new(),
            sync_actions: Vec::new(),
            role_rotation: RoleRotationConfig::default(),
            formation_center: [0.0; 3],
            alert_level: 0.0,
        }
    }

    pub fn add_member(&mut self, agent: AgentId) {
        if !self.members.contains(&agent) {
            self.members.push(agent);
            if self.leader.is_none() {
                self.leader = Some(agent);
            }
        }
    }

    pub fn remove_member(&mut self, agent: AgentId) {
        self.members.retain(|&m| m != agent);
        if self.leader == Some(agent) {
            self.leader = self.members.first().copied();
        }
    }

    pub fn active_goal(&self) -> Option<&GroupGoal> {
        self.goals.iter().filter(|g| g.is_active()).max_by_key(|g| g.priority)
    }

    pub fn completed_task_ids(&self) -> Vec<TaskId> {
        self.tasks
            .iter()
            .filter(|t| t.status == TaskStatus::Completed)
            .map(|t| t.id)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Coordinator events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum CoordinatorEvent {
    GoalAssigned { group: GroupId, goal: GoalId },
    GoalCompleted { group: GroupId, goal: GoalId },
    GoalFailed { group: GroupId, goal: GoalId },
    TaskAssigned { agent: AgentId, task: TaskId },
    TaskCompleted { agent: AgentId, task: TaskId },
    SyncActionTriggered { group: GroupId, action_type: SyncActionType },
    RoleChanged { agent: AgentId, old_role: AgentRole, new_role: AgentRole },
    RadioSent { sender: AgentId, message_type: RadioMessageType },
    LeaderChanged { group: GroupId, new_leader: AgentId },
}

// ---------------------------------------------------------------------------
// Behavior coordinator
// ---------------------------------------------------------------------------

/// The behavior coordinator managing multi-agent coordination.
pub struct BehaviorCoordinator {
    groups: HashMap<GroupId, AgentGroup>,
    agents: HashMap<AgentId, AgentCoordState>,
    next_group_id: u32,
    next_task_id: u32,
    next_goal_id: u32,
    radio_queue: Vec<RadioMessage>,
    events: Vec<CoordinatorEvent>,
    game_time: f64,
    radio_delay: f32,
}

impl BehaviorCoordinator {
    pub fn new() -> Self {
        Self {
            groups: HashMap::new(),
            agents: HashMap::new(),
            next_group_id: 0,
            next_task_id: 0,
            next_goal_id: 0,
            radio_queue: Vec::new(),
            events: Vec::new(),
            game_time: 0.0,
            radio_delay: 0.5,
        }
    }

    /// Register an agent.
    pub fn register_agent(&mut self, id: AgentId) {
        self.agents.insert(id, AgentCoordState::new(id));
    }

    /// Unregister an agent.
    pub fn unregister_agent(&mut self, id: AgentId) {
        if let Some(state) = self.agents.remove(&id) {
            if let Some(gid) = state.group {
                if let Some(group) = self.groups.get_mut(&gid) {
                    group.remove_member(id);
                }
            }
        }
    }

    /// Create a group.
    pub fn create_group(&mut self, name: &str) -> GroupId {
        let id = GroupId(self.next_group_id);
        self.next_group_id += 1;
        self.groups.insert(id, AgentGroup::new(id, name));
        id
    }

    /// Add an agent to a group.
    pub fn add_to_group(&mut self, agent: AgentId, group: GroupId) {
        if let Some(g) = self.groups.get_mut(&group) {
            g.add_member(agent);
        }
        if let Some(a) = self.agents.get_mut(&agent) {
            a.group = Some(group);
        }
    }

    /// Set an agent's role.
    pub fn set_role(&mut self, agent: AgentId, role: AgentRole) {
        if let Some(a) = self.agents.get_mut(&agent) {
            let old = a.role;
            a.role = role;
            a.role_duration = 0.0;
            self.events.push(CoordinatorEvent::RoleChanged {
                agent,
                old_role: old,
                new_role: role,
            });
        }
    }

    /// Add a goal to a group.
    pub fn add_goal(&mut self, group: GroupId, name: &str, priority: i32) -> GoalId {
        let id = GoalId(self.next_goal_id);
        self.next_goal_id += 1;
        if let Some(g) = self.groups.get_mut(&group) {
            g.goals.push(GroupGoal::new(id, name, priority));
            self.events.push(CoordinatorEvent::GoalAssigned { group, goal: id });
        }
        id
    }

    /// Add a task to a group's goal.
    pub fn add_task(&mut self, group: GroupId, goal: GoalId, name: &str) -> TaskId {
        let id = TaskId(self.next_task_id);
        self.next_task_id += 1;
        if let Some(g) = self.groups.get_mut(&group) {
            g.tasks.push(CoordinationTask::new(id, name, goal));
        }
        id
    }

    /// Send a radio message.
    pub fn send_radio(
        &mut self,
        sender: AgentId,
        msg_type: RadioMessageType,
        data: RadioData,
    ) {
        let recipients: Vec<AgentId> = if let Some(agent) = self.agents.get(&sender) {
            if let Some(gid) = agent.group {
                if let Some(group) = self.groups.get(&gid) {
                    group.members.iter().filter(|&&m| m != sender).copied().collect()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let msg = RadioMessage {
            sender,
            recipients: recipients.clone(),
            message_type: msg_type,
            data,
            send_time: self.game_time,
            delay: self.radio_delay,
            delivered: false,
        };

        self.radio_queue.push(msg);
        self.events.push(CoordinatorEvent::RadioSent {
            sender,
            message_type: msg_type,
        });
    }

    /// Create a synchronized action for a group.
    pub fn create_sync_action(
        &mut self,
        group: GroupId,
        action_type: SyncActionType,
        position: [f32; 3],
        required: u32,
    ) {
        if let Some(g) = self.groups.get_mut(&group) {
            let mut action = SyncAction::new(action_type, position, required);
            action.participants = g.members.clone();
            g.sync_actions.push(action);
        }
    }

    /// Mark an agent as ready for sync.
    pub fn mark_sync_ready(&mut self, agent: AgentId) {
        if let Some(a) = self.agents.get_mut(&agent) {
            a.ready_for_sync = true;
            if let Some(gid) = a.group {
                if let Some(group) = self.groups.get_mut(&gid) {
                    for action in &mut group.sync_actions {
                        if !action.executed {
                            action.mark_ready(agent);
                            if action.all_ready() {
                                action.start_countdown();
                            }
                        }
                    }
                }
            }
        }
    }

    /// Assign tasks to agents based on role and proximity.
    pub fn assign_tasks(&mut self, group_id: GroupId) {
        let completed = {
            let group = match self.groups.get(&group_id) {
                Some(g) => g,
                None => return,
            };
            group.completed_task_ids()
        };

        let group = match self.groups.get_mut(&group_id) {
            Some(g) => g,
            None => return,
        };

        for task in &mut group.tasks {
            if task.status != TaskStatus::Unassigned {
                continue;
            }
            if !task.is_ready(&completed) {
                task.status = TaskStatus::WaitingForDependency;
                continue;
            }

            // Find best agent for task.
            let mut best_agent: Option<AgentId> = None;
            let mut best_score = f32::MIN;

            for &member in &group.members {
                if let Some(agent) = self.agents.get(&member) {
                    if !agent.alive || agent.current_task.is_some() {
                        continue;
                    }
                    let mut score = agent.combat_effectiveness;
                    if let Some(req_role) = task.required_role {
                        if agent.role == req_role {
                            score += 10.0;
                        }
                    }
                    if let Some(task_pos) = task.position {
                        let dx = agent.position[0] - task_pos[0];
                        let dy = agent.position[1] - task_pos[1];
                        let dz = agent.position[2] - task_pos[2];
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                        score -= dist * 0.1;
                    }
                    if score > best_score {
                        best_score = score;
                        best_agent = Some(member);
                    }
                }
            }

            if let Some(agent_id) = best_agent {
                task.assigned_agent = Some(agent_id);
                task.status = TaskStatus::Assigned;
                if let Some(agent) = self.agents.get_mut(&agent_id) {
                    agent.current_task = Some(task.id);
                }
                self.events.push(CoordinatorEvent::TaskAssigned {
                    agent: agent_id,
                    task: task.id,
                });
            }
        }
    }

    /// Update the coordinator.
    pub fn update(&mut self, dt: f32) {
        self.game_time += dt as f64;

        // Deliver radio messages.
        for msg in &mut self.radio_queue {
            if !msg.delivered && (self.game_time - msg.send_time) as f32 >= msg.delay {
                msg.delivered = true;
                for &recipient in &msg.recipients {
                    if let Some(agent) = self.agents.get_mut(&recipient) {
                        agent.inbox.push(msg.clone());
                    }
                }
            }
        }
        self.radio_queue.retain(|m| !m.delivered);

        // Update groups.
        let group_ids: Vec<GroupId> = self.groups.keys().copied().collect();
        for gid in group_ids {
            // Update sync actions.
            if let Some(group) = self.groups.get_mut(&gid) {
                for action in &mut group.sync_actions {
                    if action.update(dt) {
                        self.events.push(CoordinatorEvent::SyncActionTriggered {
                            group: gid,
                            action_type: action.action_type,
                        });
                    }
                }
                group.sync_actions.retain(|a| !a.executed);

                // Update goals.
                for goal in &mut group.goals {
                    if goal.is_active() {
                        goal.elapsed += dt;
                        if goal.elapsed >= goal.timeout {
                            goal.status = GoalStatus::Failed;
                            self.events.push(CoordinatorEvent::GoalFailed { group: gid, goal: goal.id });
                        }
                    }
                }

                // Role rotation.
                if group.role_rotation.enabled {
                    group.role_rotation.rotation_timer += dt;
                    if group.role_rotation.rotation_timer >= group.role_rotation.rotation_interval {
                        group.role_rotation.rotation_timer = 0.0;
                        // Rotate roles.
                        let roles = &group.role_rotation.roles;
                        if !roles.is_empty() {
                            for (i, &member) in group.members.iter().enumerate() {
                                let role = roles[i % roles.len()];
                                if let Some(agent) = self.agents.get_mut(&member) {
                                    let old = agent.role;
                                    agent.role = role;
                                    self.events.push(CoordinatorEvent::RoleChanged {
                                        agent: member,
                                        old_role: old,
                                        new_role: role,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        // Update agent role durations.
        for agent in self.agents.values_mut() {
            agent.role_duration += dt;
        }
    }

    /// Get group.
    pub fn group(&self, id: GroupId) -> Option<&AgentGroup> {
        self.groups.get(&id)
    }

    /// Get agent state.
    pub fn agent(&self, id: AgentId) -> Option<&AgentCoordState> {
        self.agents.get(&id)
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<CoordinatorEvent> {
        std::mem::take(&mut self.events)
    }

    /// Update an agent's position.
    pub fn set_agent_position(&mut self, agent: AgentId, position: [f32; 3]) {
        if let Some(a) = self.agents.get_mut(&agent) {
            a.position = position;
        }
    }

    /// Complete a task.
    pub fn complete_task(&mut self, agent: AgentId, task: TaskId) {
        if let Some(a) = self.agents.get_mut(&agent) {
            a.current_task = None;
        }
        for group in self.groups.values_mut() {
            if let Some(task_entry) = group.tasks.iter_mut().find(|t| t.id == task) {
                task_entry.status = TaskStatus::Completed;
            }
        }
        self.events.push(CoordinatorEvent::TaskCompleted { agent, task });
    }
}

impl Default for BehaviorCoordinator {
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
    fn test_group_creation() {
        let mut coord = BehaviorCoordinator::new();
        let a1 = AgentId(0);
        let a2 = AgentId(1);
        coord.register_agent(a1);
        coord.register_agent(a2);

        let gid = coord.create_group("Alpha");
        coord.add_to_group(a1, gid);
        coord.add_to_group(a2, gid);

        let group = coord.group(gid).unwrap();
        assert_eq!(group.members.len(), 2);
        assert_eq!(group.leader, Some(a1));
    }

    #[test]
    fn test_sync_action() {
        let mut coord = BehaviorCoordinator::new();
        let a1 = AgentId(0);
        let a2 = AgentId(1);
        coord.register_agent(a1);
        coord.register_agent(a2);

        let gid = coord.create_group("Breach");
        coord.add_to_group(a1, gid);
        coord.add_to_group(a2, gid);

        coord.create_sync_action(gid, SyncActionType::CoordinatedBreach, [0.0; 3], 2);
        coord.mark_sync_ready(a1);
        coord.mark_sync_ready(a2);

        // Countdown should start, then trigger.
        coord.update(4.0);
        let events = coord.drain_events();
        assert!(events.iter().any(|e| matches!(e, CoordinatorEvent::SyncActionTriggered { .. })));
    }

    #[test]
    fn test_radio() {
        let mut coord = BehaviorCoordinator::new();
        let a1 = AgentId(0);
        let a2 = AgentId(1);
        coord.register_agent(a1);
        coord.register_agent(a2);

        let gid = coord.create_group("Team");
        coord.add_to_group(a1, gid);
        coord.add_to_group(a2, gid);

        coord.send_radio(a1, RadioMessageType::EnemySpotted, RadioData::at_position([10.0, 0.0, 5.0]));
        coord.update(1.0);

        let agent2 = coord.agent(a2).unwrap();
        assert!(!agent2.inbox.is_empty());
    }
}
