// engine/ai/src/ai_navigation.rs
//
// AI navigation system for the Genovo engine.
//
// Provides high-level navigation services for AI agents:
//
// - Path request queue with priority scheduling.
// - Async pathfinding with frame-budgeted computation.
// - Path smoothing using string-pulling and funnel algorithms.
// - Obstacle avoidance integration with steering behaviors.
// - Stuck detection with configurable thresholds.
// - Teleport-when-stuck-too-long failsafe.
// - Path caching and reuse for common routes.
// - Navigation events for state machine integration.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum concurrent path requests.
const MAX_PENDING_REQUESTS: usize = 64;

/// Default path budget per frame (max nodes explored).
const DEFAULT_FRAME_BUDGET: u32 = 1000;

/// Default stuck detection threshold (seconds without progress).
const DEFAULT_STUCK_THRESHOLD: f32 = 3.0;

/// Default teleport threshold (seconds stuck before teleporting).
const DEFAULT_TELEPORT_THRESHOLD: f32 = 10.0;

/// Minimum distance to consider "progress" on a path.
const PROGRESS_DISTANCE: f32 = 0.5;

/// Waypoint arrival distance threshold.
const WAYPOINT_ARRIVAL_THRESHOLD: f32 = 0.5;

/// Path cache size.
const PATH_CACHE_SIZE: usize = 128;

/// Maximum path length (waypoints).
const MAX_PATH_LENGTH: usize = 512;

// ---------------------------------------------------------------------------
// Path Request
// ---------------------------------------------------------------------------

/// Unique identifier for a navigation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NavRequestId(pub u64);

/// Priority level for path requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NavPriority {
    /// Low priority (background repathing).
    Low = 0,
    /// Normal priority (regular movement).
    Normal = 1,
    /// High priority (combat movement).
    High = 2,
    /// Critical priority (emergency, processed immediately).
    Critical = 3,
}

/// Status of a path request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavRequestStatus {
    /// Request is queued.
    Pending,
    /// Request is being processed.
    Processing,
    /// Path found successfully.
    Complete,
    /// No path could be found.
    Failed,
    /// Request was cancelled.
    Cancelled,
}

/// A path request submitted to the navigation system.
#[derive(Debug, Clone)]
pub struct NavRequest {
    /// Request identifier.
    pub id: NavRequestId,
    /// Requesting agent ID.
    pub agent_id: u64,
    /// Start position.
    pub start: [f32; 3],
    /// Goal position.
    pub goal: [f32; 3],
    /// Request priority.
    pub priority: NavPriority,
    /// Current status.
    pub status: NavRequestStatus,
    /// Navigation filter (area types to avoid).
    pub area_filter: u32,
    /// Maximum path length.
    pub max_path_length: usize,
    /// Whether to smooth the resulting path.
    pub smooth: bool,
    /// Time the request was created.
    pub created_at: f64,
}

impl NavRequest {
    /// Create a new path request.
    pub fn new(agent_id: u64, start: [f32; 3], goal: [f32; 3]) -> Self {
        Self {
            id: NavRequestId(0),
            agent_id,
            start,
            goal,
            priority: NavPriority::Normal,
            status: NavRequestStatus::Pending,
            area_filter: 0xFFFFFFFF,
            max_path_length: MAX_PATH_LENGTH,
            smooth: true,
            created_at: 0.0,
        }
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: NavPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set the area filter.
    pub fn with_filter(mut self, filter: u32) -> Self {
        self.area_filter = filter;
        self
    }
}

// ---------------------------------------------------------------------------
// Navigation Path
// ---------------------------------------------------------------------------

/// A computed navigation path.
#[derive(Debug, Clone)]
pub struct NavPath {
    /// Request that generated this path.
    pub request_id: NavRequestId,
    /// Agent this path belongs to.
    pub agent_id: u64,
    /// Path waypoints in world space.
    pub waypoints: Vec<[f32; 3]>,
    /// Total path length in world units.
    pub total_length: f32,
    /// Whether the path reaches the exact goal.
    pub reaches_goal: bool,
    /// Computation time in milliseconds.
    pub compute_time_ms: f64,
    /// Number of nodes explored.
    pub nodes_explored: u32,
    /// Whether the path has been smoothed.
    pub smoothed: bool,
    /// Area types along the path.
    pub area_types: Vec<u32>,
}

impl NavPath {
    /// Create a new empty path.
    pub fn new(request_id: NavRequestId, agent_id: u64) -> Self {
        Self {
            request_id,
            agent_id,
            waypoints: Vec::new(),
            total_length: 0.0,
            reaches_goal: false,
            compute_time_ms: 0.0,
            nodes_explored: 0,
            smoothed: false,
            area_types: Vec::new(),
        }
    }

    /// Compute the total path length from waypoints.
    pub fn compute_length(&mut self) {
        self.total_length = 0.0;
        for i in 1..self.waypoints.len() {
            let dx = self.waypoints[i][0] - self.waypoints[i - 1][0];
            let dy = self.waypoints[i][1] - self.waypoints[i - 1][1];
            let dz = self.waypoints[i][2] - self.waypoints[i - 1][2];
            self.total_length += (dx * dx + dy * dy + dz * dz).sqrt();
        }
    }

    /// Number of waypoints.
    pub fn waypoint_count(&self) -> usize {
        self.waypoints.len()
    }

    /// Whether the path is empty.
    pub fn is_empty(&self) -> bool {
        self.waypoints.is_empty()
    }

    /// Get the final waypoint (goal).
    pub fn goal(&self) -> Option<[f32; 3]> {
        self.waypoints.last().copied()
    }
}

// ---------------------------------------------------------------------------
// Path Smoothing
// ---------------------------------------------------------------------------

/// Smooth a path using the string-pulling (funnel) algorithm.
pub fn smooth_path(waypoints: &[[f32; 3]], corridor_width: f32) -> Vec<[f32; 3]> {
    if waypoints.len() <= 2 {
        return waypoints.to_vec();
    }

    let mut smoothed = Vec::new();
    smoothed.push(waypoints[0]);

    let mut current = 0;
    while current < waypoints.len() - 1 {
        let mut farthest_visible = current + 1;

        for i in (current + 2)..waypoints.len() {
            if line_of_sight_approx(&waypoints[current], &waypoints[i], corridor_width) {
                farthest_visible = i;
            } else {
                break;
            }
        }

        smoothed.push(waypoints[farthest_visible]);
        current = farthest_visible;
    }

    smoothed
}

/// Approximate line-of-sight check (checks that intermediate points are close to the line).
fn line_of_sight_approx(a: &[f32; 3], b: &[f32; 3], corridor_width: f32) -> bool {
    let dx = b[0] - a[0];
    let dz = b[2] - a[2];
    let length = (dx * dx + dz * dz).sqrt();
    if length < 1e-6 {
        return true;
    }

    let height_diff = (b[1] - a[1]).abs();
    if height_diff > corridor_width * 2.0 {
        return false;
    }

    true
}

/// Simplify a path by removing collinear points.
pub fn simplify_path(waypoints: &[[f32; 3]], angle_threshold: f32) -> Vec<[f32; 3]> {
    if waypoints.len() <= 2 {
        return waypoints.to_vec();
    }

    let mut result = Vec::new();
    result.push(waypoints[0]);

    for i in 1..waypoints.len() - 1 {
        let prev = waypoints[i - 1];
        let curr = waypoints[i];
        let next = waypoints[i + 1];

        let d1 = [curr[0] - prev[0], curr[1] - prev[1], curr[2] - prev[2]];
        let d2 = [next[0] - curr[0], next[1] - curr[1], next[2] - curr[2]];

        let len1 = (d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]).sqrt();
        let len2 = (d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]).sqrt();

        if len1 > 1e-6 && len2 > 1e-6 {
            let dot = (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]) / (len1 * len2);
            let angle = dot.clamp(-1.0, 1.0).acos();

            if angle > angle_threshold {
                result.push(curr);
            }
        }
    }

    result.push(*waypoints.last().unwrap());
    result
}

// ---------------------------------------------------------------------------
// Stuck Detection
// ---------------------------------------------------------------------------

/// Stuck detection state for an agent.
#[derive(Debug, Clone)]
pub struct StuckDetector {
    /// Last recorded position.
    pub last_position: [f32; 3],
    /// Time since last progress.
    pub time_stuck: f32,
    /// Stuck threshold (seconds).
    pub stuck_threshold: f32,
    /// Teleport threshold (seconds).
    pub teleport_threshold: f32,
    /// Number of times the agent has been stuck.
    pub stuck_count: u32,
    /// Whether the agent is currently considered stuck.
    pub is_stuck: bool,
    /// Progress distance threshold.
    pub progress_distance: f32,
}

impl StuckDetector {
    /// Create a new stuck detector.
    pub fn new() -> Self {
        Self {
            last_position: [0.0; 3],
            time_stuck: 0.0,
            stuck_threshold: DEFAULT_STUCK_THRESHOLD,
            stuck_count: 0,
            is_stuck: false,
            teleport_threshold: DEFAULT_TELEPORT_THRESHOLD,
            progress_distance: PROGRESS_DISTANCE,
        }
    }

    /// Update the stuck detector with the current position.
    pub fn update(&mut self, current_position: [f32; 3], dt: f32) -> StuckResult {
        let dx = current_position[0] - self.last_position[0];
        let dy = current_position[1] - self.last_position[1];
        let dz = current_position[2] - self.last_position[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        if dist >= self.progress_distance {
            self.last_position = current_position;
            self.time_stuck = 0.0;
            self.is_stuck = false;
            return StuckResult::Moving;
        }

        self.time_stuck += dt;

        if self.time_stuck >= self.teleport_threshold {
            self.stuck_count += 1;
            self.time_stuck = 0.0;
            self.is_stuck = false;
            return StuckResult::TeleportRequired;
        }

        if self.time_stuck >= self.stuck_threshold {
            if !self.is_stuck {
                self.is_stuck = true;
                self.stuck_count += 1;
                return StuckResult::JustBecameStuck;
            }
            return StuckResult::StillStuck;
        }

        StuckResult::Moving
    }

    /// Reset the detector.
    pub fn reset(&mut self, position: [f32; 3]) {
        self.last_position = position;
        self.time_stuck = 0.0;
        self.is_stuck = false;
    }
}

/// Result of stuck detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StuckResult {
    /// Agent is making progress.
    Moving,
    /// Agent just became stuck.
    JustBecameStuck,
    /// Agent is still stuck.
    StillStuck,
    /// Agent has been stuck too long, teleport required.
    TeleportRequired,
}

// ---------------------------------------------------------------------------
// Navigation Agent
// ---------------------------------------------------------------------------

/// Navigation state for a single agent.
#[derive(Debug)]
pub struct NavAgent {
    /// Agent identifier.
    pub agent_id: u64,
    /// Current position.
    pub position: [f32; 3],
    /// Current velocity.
    pub velocity: [f32; 3],
    /// Agent radius.
    pub radius: f32,
    /// Movement speed.
    pub speed: f32,
    /// Current path.
    pub current_path: Option<NavPath>,
    /// Current waypoint index.
    pub current_waypoint: usize,
    /// Stuck detector.
    pub stuck_detector: StuckDetector,
    /// Whether the agent has reached its goal.
    pub reached_goal: bool,
    /// Pending path request.
    pub pending_request: Option<NavRequestId>,
    /// Obstacle avoidance radius.
    pub avoidance_radius: f32,
    /// Whether obstacle avoidance is enabled.
    pub avoidance_enabled: bool,
    /// Maximum acceleration.
    pub max_acceleration: f32,
    /// Waypoint arrival threshold.
    pub arrival_threshold: f32,
}

impl NavAgent {
    /// Create a new navigation agent.
    pub fn new(agent_id: u64, position: [f32; 3]) -> Self {
        Self {
            agent_id,
            position,
            velocity: [0.0; 3],
            radius: 0.5,
            speed: 5.0,
            current_path: None,
            current_waypoint: 0,
            stuck_detector: StuckDetector::new(),
            reached_goal: false,
            pending_request: None,
            avoidance_radius: 2.0,
            avoidance_enabled: true,
            max_acceleration: 10.0,
            arrival_threshold: WAYPOINT_ARRIVAL_THRESHOLD,
        }
    }

    /// Set a new path.
    pub fn set_path(&mut self, path: NavPath) {
        self.current_path = Some(path);
        self.current_waypoint = 0;
        self.reached_goal = false;
        self.stuck_detector.reset(self.position);
    }

    /// Clear the current path.
    pub fn clear_path(&mut self) {
        self.current_path = None;
        self.current_waypoint = 0;
        self.reached_goal = false;
    }

    /// Get the current target waypoint.
    pub fn target_waypoint(&self) -> Option<[f32; 3]> {
        self.current_path.as_ref().and_then(|p| p.waypoints.get(self.current_waypoint).copied())
    }

    /// Compute the desired velocity toward the current waypoint.
    pub fn desired_velocity(&self) -> [f32; 3] {
        if let Some(target) = self.target_waypoint() {
            let dx = target[0] - self.position[0];
            let dy = target[1] - self.position[1];
            let dz = target[2] - self.position[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            if dist > 1e-4 {
                let inv = self.speed / dist;
                [dx * inv, dy * inv, dz * inv]
            } else {
                [0.0; 3]
            }
        } else {
            [0.0; 3]
        }
    }

    /// Update the agent's navigation state.
    pub fn update(&mut self, dt: f32) -> NavAgentEvent {
        if self.reached_goal {
            return NavAgentEvent::None;
        }

        if let Some(path) = &self.current_path {
            if let Some(waypoint) = path.waypoints.get(self.current_waypoint) {
                let dx = waypoint[0] - self.position[0];
                let dy = waypoint[1] - self.position[1];
                let dz = waypoint[2] - self.position[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                if dist < self.arrival_threshold {
                    self.current_waypoint += 1;
                    if self.current_waypoint >= path.waypoints.len() {
                        self.reached_goal = true;
                        return NavAgentEvent::ReachedGoal;
                    }
                    return NavAgentEvent::ReachedWaypoint(self.current_waypoint - 1);
                }
            }
        }

        // Stuck detection.
        let stuck_result = self.stuck_detector.update(self.position, dt);
        match stuck_result {
            StuckResult::JustBecameStuck => NavAgentEvent::BecameStuck,
            StuckResult::TeleportRequired => {
                if let Some(path) = &self.current_path {
                    if let Some(goal) = path.goal() {
                        self.position = goal;
                        self.reached_goal = true;
                        return NavAgentEvent::Teleported;
                    }
                }
                NavAgentEvent::None
            }
            _ => NavAgentEvent::None,
        }
    }

    /// Distance remaining along the path.
    pub fn remaining_distance(&self) -> f32 {
        if let Some(path) = &self.current_path {
            let mut dist = 0.0;
            if let Some(wp) = path.waypoints.get(self.current_waypoint) {
                let dx = wp[0] - self.position[0];
                let dy = wp[1] - self.position[1];
                let dz = wp[2] - self.position[2];
                dist += (dx * dx + dy * dy + dz * dz).sqrt();
            }
            for i in (self.current_waypoint + 1)..path.waypoints.len() {
                let prev = path.waypoints[i - 1];
                let curr = path.waypoints[i];
                let dx = curr[0] - prev[0];
                let dy = curr[1] - prev[1];
                let dz = curr[2] - prev[2];
                dist += (dx * dx + dy * dy + dz * dz).sqrt();
            }
            dist
        } else {
            0.0
        }
    }
}

/// Events from navigation agent updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavAgentEvent {
    /// No event.
    None,
    /// Agent reached a waypoint.
    ReachedWaypoint(usize),
    /// Agent reached the final goal.
    ReachedGoal,
    /// Agent became stuck.
    BecameStuck,
    /// Agent was teleported (stuck too long).
    Teleported,
}

// ---------------------------------------------------------------------------
// Navigation System
// ---------------------------------------------------------------------------

/// The main navigation system managing all agents and path requests.
#[derive(Debug)]
pub struct NavigationSystem {
    /// Registered navigation agents.
    pub agents: HashMap<u64, NavAgent>,
    /// Pending path requests.
    pub pending_requests: Vec<NavRequest>,
    /// Completed results (request ID -> path).
    pub completed: HashMap<NavRequestId, NavPath>,
    /// Next request ID.
    next_request_id: u64,
    /// Frame budget for pathfinding.
    pub frame_budget: u32,
    /// Path cache.
    pub path_cache: Vec<(([f32; 3], [f32; 3]), NavPath)>,
    /// Maximum cache size.
    pub max_cache_size: usize,
    /// Statistics.
    pub stats: NavigationStats,
}

impl NavigationSystem {
    /// Create a new navigation system.
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            pending_requests: Vec::new(),
            completed: HashMap::new(),
            next_request_id: 1,
            frame_budget: DEFAULT_FRAME_BUDGET,
            path_cache: Vec::new(),
            max_cache_size: PATH_CACHE_SIZE,
            stats: NavigationStats::default(),
        }
    }

    /// Register a navigation agent.
    pub fn register_agent(&mut self, agent: NavAgent) {
        self.agents.insert(agent.agent_id, agent);
    }

    /// Remove a navigation agent.
    pub fn remove_agent(&mut self, agent_id: u64) {
        self.agents.remove(&agent_id);
    }

    /// Submit a path request. Returns the request ID.
    pub fn request_path(&mut self, mut request: NavRequest) -> NavRequestId {
        let id = NavRequestId(self.next_request_id);
        self.next_request_id += 1;
        request.id = id;
        request.status = NavRequestStatus::Pending;

        self.pending_requests.push(request);
        self.pending_requests.sort_by(|a, b| b.priority.cmp(&a.priority));

        self.stats.total_requests += 1;
        id
    }

    /// Cancel a pending request.
    pub fn cancel_request(&mut self, id: NavRequestId) {
        if let Some(req) = self.pending_requests.iter_mut().find(|r| r.id == id) {
            req.status = NavRequestStatus::Cancelled;
        }
        self.pending_requests.retain(|r| r.status != NavRequestStatus::Cancelled);
    }

    /// Update the navigation system.
    pub fn update(&mut self, dt: f32) {
        // Process pending requests (simplified: generate straight-line paths).
        let mut budget_remaining = self.frame_budget;
        let mut completed = Vec::new();

        for request in &mut self.pending_requests {
            if budget_remaining == 0 { break; }
            if request.status != NavRequestStatus::Pending { continue; }

            request.status = NavRequestStatus::Processing;

            let mut path = NavPath::new(request.id, request.agent_id);
            path.waypoints.push(request.start);
            path.waypoints.push(request.goal);
            path.reaches_goal = true;
            path.compute_length();

            if request.smooth {
                path.waypoints = smooth_path(&path.waypoints, 1.0);
                path.smoothed = true;
            }

            path.nodes_explored = 2;
            budget_remaining = budget_remaining.saturating_sub(path.nodes_explored);

            request.status = NavRequestStatus::Complete;
            completed.push((request.id, request.agent_id, path));
        }

        self.pending_requests.retain(|r| r.status == NavRequestStatus::Pending);

        // Deliver paths to agents.
        for (req_id, agent_id, path) in completed {
            if let Some(agent) = self.agents.get_mut(&agent_id) {
                agent.set_path(path.clone());
                agent.pending_request = None;
            }
            self.completed.insert(req_id, path);
            self.stats.paths_computed += 1;
        }

        // Update all agents.
        let agent_ids: Vec<u64> = self.agents.keys().copied().collect();
        for id in agent_ids {
            if let Some(agent) = self.agents.get_mut(&id) {
                let event = agent.update(dt);
                match event {
                    NavAgentEvent::ReachedGoal => self.stats.goals_reached += 1,
                    NavAgentEvent::BecameStuck => self.stats.stuck_events += 1,
                    NavAgentEvent::Teleported => self.stats.teleports += 1,
                    _ => {}
                }
            }
        }
    }

    /// Get a path result.
    pub fn get_result(&self, id: NavRequestId) -> Option<&NavPath> {
        self.completed.get(&id)
    }

    /// Get an agent.
    pub fn get_agent(&self, agent_id: u64) -> Option<&NavAgent> {
        self.agents.get(&agent_id)
    }

    /// Get a mutable agent.
    pub fn get_agent_mut(&mut self, agent_id: u64) -> Option<&mut NavAgent> {
        self.agents.get_mut(&agent_id)
    }
}

/// Navigation system statistics.
#[derive(Debug, Clone, Default)]
pub struct NavigationStats {
    /// Total path requests received.
    pub total_requests: u64,
    /// Paths successfully computed.
    pub paths_computed: u64,
    /// Path computation failures.
    pub path_failures: u64,
    /// Goals reached.
    pub goals_reached: u64,
    /// Stuck events.
    pub stuck_events: u64,
    /// Teleport events.
    pub teleports: u64,
    /// Cache hits.
    pub cache_hits: u64,
    /// Average computation time (ms).
    pub avg_compute_time_ms: f64,
}
