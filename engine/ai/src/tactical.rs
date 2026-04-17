//! Tactical AI system.
//!
//! Provides cover evaluation, firing position selection, patrol routes,
//! ambush setups, retreat path finding, and a tactical grid map for AI
//! agents that need to reason about spatial combat.
//!
//! # Key concepts
//!
//! - **TacticalMap**: A grid of cells each holding tactical values such as
//!   exposure, cover quality, threat level, and desirability.
//! - **CoverPoint**: A position that provides protection from one or more
//!   directions, scored by exposure, distance to threat, and flanking risk.
//! - **FiringPosition**: A position from which an agent can engage a target
//!   while minimizing their own exposure.
//! - **PatrolRoute**: An ordered set of waypoints an agent visits in sequence
//!   or randomly for area control.
//! - **AmbushSetup**: A coordinated arrangement of agents at positions with
//!   good concealment and overlapping fields of fire.
//! - **RetreatPath**: A route from the current position to a safe zone,
//!   preferring cover along the way.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

use glam::Vec3;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default tactical grid cell size in world units.
pub const DEFAULT_CELL_SIZE: f32 = 2.0;

/// Maximum grid dimensions (cells per axis).
pub const MAX_GRID_SIZE: usize = 256;

/// Maximum number of cover points tracked per tactical map.
pub const MAX_COVER_POINTS: usize = 2048;

/// Maximum number of threats considered simultaneously.
pub const MAX_THREATS: usize = 64;

/// Maximum patrol waypoints per route.
pub const MAX_PATROL_WAYPOINTS: usize = 64;

/// Maximum agents in a single ambush setup.
pub const MAX_AMBUSH_AGENTS: usize = 16;

/// Minimum cover quality score to be considered valid cover (0-1).
pub const MIN_COVER_QUALITY: f32 = 0.3;

/// Weight for distance in cover scoring.
pub const COVER_DISTANCE_WEIGHT: f32 = 0.3;

/// Weight for exposure in cover scoring.
pub const COVER_EXPOSURE_WEIGHT: f32 = 0.4;

/// Weight for flanking safety in cover scoring.
pub const COVER_FLANK_WEIGHT: f32 = 0.2;

/// Weight for retreat accessibility in cover scoring.
pub const COVER_RETREAT_WEIGHT: f32 = 0.1;

/// Maximum distance to consider for firing positions.
pub const MAX_FIRING_RANGE: f32 = 100.0;

/// Ideal engagement distance for most weapons.
pub const DEFAULT_IDEAL_RANGE: f32 = 20.0;

/// Distance at which a threat is considered "close".
pub const CLOSE_THREAT_DISTANCE: f32 = 8.0;

/// Angular tolerance for flanking detection (radians).
pub const FLANK_ANGLE_THRESHOLD: f32 = 1.2;

/// Number of directional samples for exposure calculation.
pub const EXPOSURE_SAMPLE_COUNT: usize = 16;

/// Height offset for line-of-sight checks (crouch vs stand).
pub const CROUCH_HEIGHT: f32 = 0.8;

/// Height offset for standing.
pub const STAND_HEIGHT: f32 = 1.6;

/// Epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// CoverType
// ---------------------------------------------------------------------------

/// Type of cover a point provides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoverType {
    /// Full cover — agent is fully hidden (e.g., behind a wall).
    Full,
    /// Half cover — agent is partially exposed (e.g., behind a low wall).
    Half,
    /// Soft cover — concealment only, no ballistic protection (e.g., bushes).
    Soft,
    /// Elevated cover — height advantage (e.g., rooftop, balcony).
    Elevated,
    /// Corner cover — cover at the edge of a wall, allowing lean-peek.
    Corner,
}

impl CoverType {
    /// Base protection factor (0 = no protection, 1 = full).
    pub fn protection_factor(&self) -> f32 {
        match self {
            Self::Full => 1.0,
            Self::Half => 0.6,
            Self::Soft => 0.2,
            Self::Elevated => 0.7,
            Self::Corner => 0.8,
        }
    }

    /// Whether the agent can fire from this cover without leaving it.
    pub fn can_fire_from(&self) -> bool {
        match self {
            Self::Full => false,
            Self::Half => true,
            Self::Soft => true,
            Self::Elevated => true,
            Self::Corner => true,
        }
    }
}

// ---------------------------------------------------------------------------
// ThreatInfo
// ---------------------------------------------------------------------------

/// Information about a known threat for tactical reasoning.
#[derive(Debug, Clone)]
pub struct ThreatInfo {
    /// Entity ID of the threat.
    pub entity_id: u64,
    /// Current position of the threat.
    pub position: Vec3,
    /// Direction the threat is facing.
    pub facing: Vec3,
    /// Threat level (higher = more dangerous).
    pub threat_level: f32,
    /// Maximum effective range of the threat's weapon.
    pub weapon_range: f32,
    /// Whether the threat has line of sight to a given position.
    pub has_los: bool,
    /// Last known velocity.
    pub velocity: Vec3,
    /// Timestamp of last observation.
    pub last_seen: f64,
    /// Whether the threat is currently suppressed.
    pub suppressed: bool,
}

impl ThreatInfo {
    /// Create a new threat info.
    pub fn new(entity_id: u64, position: Vec3, threat_level: f32) -> Self {
        Self {
            entity_id,
            position,
            facing: Vec3::X,
            threat_level,
            weapon_range: MAX_FIRING_RANGE,
            has_los: true,
            velocity: Vec3::ZERO,
            last_seen: 0.0,
            suppressed: false,
        }
    }

    /// Distance from this threat to a point.
    pub fn distance_to(&self, point: Vec3) -> f32 {
        self.position.distance(point)
    }

    /// Check if a position is within this threat's weapon range.
    pub fn in_range(&self, point: Vec3) -> bool {
        self.distance_to(point) <= self.weapon_range
    }

    /// Check if a position is behind this threat (relative to facing).
    pub fn is_behind(&self, point: Vec3) -> bool {
        let to_point = (point - self.position).normalize_or_zero();
        self.facing.dot(to_point) < -0.3
    }

    /// Predict the threat's position after a given time.
    pub fn predicted_position(&self, dt: f32) -> Vec3 {
        self.position + self.velocity * dt
    }
}

// ---------------------------------------------------------------------------
// CoverPoint
// ---------------------------------------------------------------------------

/// A position that provides cover from threats.
#[derive(Debug, Clone)]
pub struct CoverPoint {
    /// World position of the cover point.
    pub position: Vec3,
    /// Normal direction of the cover (direction the cover faces — away from
    /// the protected side).
    pub normal: Vec3,
    /// Type of cover.
    pub cover_type: CoverType,
    /// Overall quality score (0..1).
    pub quality: f32,
    /// Exposure score — how exposed the agent is (0 = hidden, 1 = fully exposed).
    pub exposure: f32,
    /// Whether this point is currently occupied by another agent.
    pub occupied: bool,
    /// Entity ID of the occupying agent, if any.
    pub occupant: Option<u64>,
    /// Directions from which this cover protects.
    pub protected_directions: Vec<Vec3>,
    /// Flanking vulnerability score (0 = safe from flanks, 1 = easily flanked).
    pub flank_vulnerability: f32,
    /// Distance to the nearest retreat route.
    pub retreat_distance: f32,
    /// Whether line of sight to at least one target is available.
    pub has_firing_line: bool,
    /// Timestamp when this cover was last evaluated.
    pub last_evaluated: f64,
}

impl CoverPoint {
    /// Create a new cover point.
    pub fn new(position: Vec3, normal: Vec3, cover_type: CoverType) -> Self {
        Self {
            position,
            normal,
            cover_type,
            quality: 0.0,
            exposure: 1.0,
            occupied: false,
            occupant: None,
            protected_directions: vec![normal],
            flank_vulnerability: 0.5,
            retreat_distance: f32::MAX,
            has_firing_line: false,
            last_evaluated: 0.0,
        }
    }

    /// Evaluate the quality of this cover point against given threats.
    pub fn evaluate(&mut self, agent_pos: Vec3, threats: &[ThreatInfo]) {
        if threats.is_empty() {
            self.quality = 0.0;
            self.exposure = 1.0;
            return;
        }

        // Calculate exposure from all threats
        let mut total_exposure = 0.0;
        let mut threat_count = 0.0;
        let mut best_firing_line = false;

        for threat in threats {
            let to_threat = (threat.position - self.position).normalize_or_zero();

            // Check if cover normal faces the threat (cover is useful)
            let cover_alignment = self.normal.dot(-to_threat);
            if cover_alignment > 0.0 {
                // Cover is between agent and threat
                let protection = self.cover_type.protection_factor() * cover_alignment;
                total_exposure += 1.0 - protection;
            } else {
                // Threat is behind the cover — fully exposed from that direction
                total_exposure += 1.0;
            }
            threat_count += 1.0;

            // Check firing line
            if self.cover_type.can_fire_from() && threat.in_range(self.position) {
                best_firing_line = true;
            }
        }

        self.exposure = if threat_count > 0.0 {
            (total_exposure / threat_count).clamp(0.0, 1.0)
        } else {
            1.0
        };

        self.has_firing_line = best_firing_line;

        // Calculate flank vulnerability
        self.flank_vulnerability = self.calculate_flank_vulnerability(threats);

        // Calculate overall quality
        let distance_to_agent = agent_pos.distance(self.position);
        let distance_score = 1.0 - (distance_to_agent / MAX_FIRING_RANGE).clamp(0.0, 1.0);
        let exposure_score = 1.0 - self.exposure;
        let flank_score = 1.0 - self.flank_vulnerability;
        let retreat_score = if self.retreat_distance < f32::MAX {
            1.0 - (self.retreat_distance / 50.0).clamp(0.0, 1.0)
        } else {
            0.0
        };

        self.quality = distance_score * COVER_DISTANCE_WEIGHT
            + exposure_score * COVER_EXPOSURE_WEIGHT
            + flank_score * COVER_FLANK_WEIGHT
            + retreat_score * COVER_RETREAT_WEIGHT;

        // Penalty for being occupied
        if self.occupied {
            self.quality *= 0.1;
        }
    }

    /// Calculate how vulnerable this cover is to flanking.
    fn calculate_flank_vulnerability(&self, threats: &[ThreatInfo]) -> f32 {
        if threats.len() < 2 {
            return 0.2; // Low flank risk with single threat
        }

        let mut max_angle_diff = 0.0f32;

        for i in 0..threats.len() {
            for j in (i + 1)..threats.len() {
                let dir_a = (threats[i].position - self.position).normalize_or_zero();
                let dir_b = (threats[j].position - self.position).normalize_or_zero();
                let angle = dir_a.dot(dir_b).acos();
                max_angle_diff = max_angle_diff.max(angle);
            }
        }

        // If threats span a wide angle, flanking is likely
        (max_angle_diff / std::f32::consts::PI).clamp(0.0, 1.0)
    }

    /// Check if this cover is suitable (quality exceeds minimum threshold).
    pub fn is_suitable(&self) -> bool {
        self.quality >= MIN_COVER_QUALITY && !self.occupied
    }
}

// ---------------------------------------------------------------------------
// FiringPosition
// ---------------------------------------------------------------------------

/// A position suitable for engaging a target.
#[derive(Debug, Clone)]
pub struct FiringPosition {
    /// World position.
    pub position: Vec3,
    /// Target entity ID.
    pub target_id: u64,
    /// Distance to target.
    pub distance_to_target: f32,
    /// Angle quality (1.0 = optimal engagement angle).
    pub angle_quality: f32,
    /// Cover available at this position.
    pub cover_available: Option<CoverType>,
    /// Overall score (0..1).
    pub score: f32,
    /// Whether the position has line of sight to the target.
    pub has_los: bool,
    /// Height advantage over target (positive = higher).
    pub height_advantage: f32,
}

impl FiringPosition {
    /// Create a new firing position.
    pub fn new(position: Vec3, target_id: u64, target_pos: Vec3) -> Self {
        let distance = position.distance(target_pos);
        let height_advantage = position.y - target_pos.y;

        Self {
            position,
            target_id,
            distance_to_target: distance,
            angle_quality: 0.5,
            cover_available: None,
            score: 0.0,
            has_los: true,
            height_advantage,
        }
    }

    /// Evaluate this firing position.
    pub fn evaluate(&mut self, ideal_range: f32, weapon_range: f32) {
        // Distance score: how close to ideal range
        let range_ratio = self.distance_to_target / ideal_range;
        let distance_score = if range_ratio <= 1.0 {
            range_ratio
        } else {
            (2.0 - range_ratio).max(0.0)
        };

        // Range penalty if beyond weapon range
        let in_range = if self.distance_to_target <= weapon_range {
            1.0
        } else {
            0.0
        };

        // Height advantage bonus
        let height_bonus = (self.height_advantage / 5.0).clamp(0.0, 0.2);

        // Cover bonus
        let cover_bonus = self
            .cover_available
            .map_or(0.0, |c| c.protection_factor() * 0.3);

        // LOS requirement
        let los_factor = if self.has_los { 1.0 } else { 0.0 };

        self.score = (distance_score * 0.4
            + self.angle_quality * 0.2
            + cover_bonus
            + height_bonus)
            * in_range
            * los_factor;
    }
}

// ---------------------------------------------------------------------------
// PatrolRoute
// ---------------------------------------------------------------------------

/// A patrol route consisting of ordered waypoints.
#[derive(Debug, Clone)]
pub struct PatrolRoute {
    /// Unique route identifier.
    pub id: u32,
    /// Human-readable name.
    pub name: String,
    /// Ordered waypoints.
    pub waypoints: Vec<PatrolWaypoint>,
    /// Whether the route loops (last waypoint connects to first).
    pub loops: bool,
    /// Current waypoint index for an agent following this route.
    current_index: usize,
    /// Direction of traversal (1 = forward, -1 = backward for ping-pong).
    direction: i32,
    /// Patrol mode.
    pub mode: PatrolMode,
    /// Total route distance (computed).
    pub total_distance: f32,
    /// Whether the route is active.
    pub active: bool,
}

/// A single waypoint in a patrol route.
#[derive(Debug, Clone)]
pub struct PatrolWaypoint {
    /// World position.
    pub position: Vec3,
    /// Time to wait at this waypoint (seconds).
    pub wait_time: f32,
    /// Direction to face while waiting.
    pub facing: Option<Vec3>,
    /// Alert level at this waypoint (0 = relaxed, 1 = fully alert).
    pub alert_level: f32,
    /// Optional action to perform at this waypoint (e.g., "look_around").
    pub action: Option<String>,
    /// Radius within which the waypoint is considered reached.
    pub arrival_radius: f32,
}

/// How the patrol route is traversed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatrolMode {
    /// Traverse waypoints in order, then loop back to start.
    Loop,
    /// Traverse forward then backward (ping-pong).
    PingPong,
    /// Choose the next waypoint randomly.
    Random,
    /// Visit once and stop at the last waypoint.
    OneShot,
}

impl PatrolRoute {
    /// Create a new patrol route.
    pub fn new(id: u32, name: impl Into<String>, mode: PatrolMode) -> Self {
        Self {
            id,
            name: name.into(),
            waypoints: Vec::new(),
            loops: mode == PatrolMode::Loop,
            current_index: 0,
            direction: 1,
            mode,
            total_distance: 0.0,
            active: true,
        }
    }

    /// Add a waypoint to the route.
    pub fn add_waypoint(&mut self, waypoint: PatrolWaypoint) {
        if self.waypoints.len() < MAX_PATROL_WAYPOINTS {
            self.waypoints.push(waypoint);
            self.recalculate_distance();
        }
    }

    /// Add a simple waypoint (position only).
    pub fn add_point(&mut self, position: Vec3, wait_time: f32) {
        self.add_waypoint(PatrolWaypoint {
            position,
            wait_time,
            facing: None,
            alert_level: 0.0,
            action: None,
            arrival_radius: 1.0,
        });
    }

    /// Get the current waypoint.
    pub fn current_waypoint(&self) -> Option<&PatrolWaypoint> {
        self.waypoints.get(self.current_index)
    }

    /// Advance to the next waypoint. Returns the new waypoint index.
    pub fn advance(&mut self) -> usize {
        if self.waypoints.is_empty() {
            return 0;
        }

        match self.mode {
            PatrolMode::Loop => {
                self.current_index = (self.current_index + 1) % self.waypoints.len();
            }
            PatrolMode::PingPong => {
                let next = self.current_index as i32 + self.direction;
                if next < 0 {
                    self.direction = 1;
                    self.current_index = 1.min(self.waypoints.len() - 1);
                } else if next >= self.waypoints.len() as i32 {
                    self.direction = -1;
                    self.current_index = if self.waypoints.len() >= 2 {
                        self.waypoints.len() - 2
                    } else {
                        0
                    };
                } else {
                    self.current_index = next as usize;
                }
            }
            PatrolMode::Random => {
                if self.waypoints.len() > 1 {
                    // Simple deterministic "random" — pick next or skip one
                    let skip = (self.current_index * 7 + 3) % self.waypoints.len();
                    self.current_index = if skip == self.current_index {
                        (self.current_index + 1) % self.waypoints.len()
                    } else {
                        skip
                    };
                }
            }
            PatrolMode::OneShot => {
                if self.current_index < self.waypoints.len() - 1 {
                    self.current_index += 1;
                } else {
                    self.active = false;
                }
            }
        }

        self.current_index
    }

    /// Reset to the first waypoint.
    pub fn reset(&mut self) {
        self.current_index = 0;
        self.direction = 1;
        self.active = true;
    }

    /// Check if an agent at a given position has reached the current waypoint.
    pub fn has_reached_current(&self, agent_pos: Vec3) -> bool {
        self.current_waypoint()
            .map_or(false, |wp| agent_pos.distance(wp.position) <= wp.arrival_radius)
    }

    /// Recalculate the total route distance.
    fn recalculate_distance(&mut self) {
        self.total_distance = 0.0;
        for i in 1..self.waypoints.len() {
            self.total_distance += self.waypoints[i - 1]
                .position
                .distance(self.waypoints[i].position);
        }
        if self.loops && self.waypoints.len() > 1 {
            self.total_distance += self.waypoints.last().unwrap().position.distance(
                self.waypoints[0].position,
            );
        }
    }

    /// Get the number of waypoints.
    pub fn waypoint_count(&self) -> usize {
        self.waypoints.len()
    }

    /// Get estimated time to complete one full patrol cycle (excluding wait times).
    pub fn estimated_cycle_time(&self, move_speed: f32) -> f32 {
        if move_speed < EPSILON {
            return f32::MAX;
        }
        let move_time = self.total_distance / move_speed;
        let wait_time: f32 = self.waypoints.iter().map(|w| w.wait_time).sum();
        move_time + wait_time
    }
}

// ---------------------------------------------------------------------------
// AmbushSetup
// ---------------------------------------------------------------------------

/// A coordinated ambush plan for a group of agents.
#[derive(Debug, Clone)]
pub struct AmbushSetup {
    /// Unique setup identifier.
    pub id: u32,
    /// Kill zone center — where the ambush targets are expected to be.
    pub kill_zone_center: Vec3,
    /// Kill zone radius.
    pub kill_zone_radius: f32,
    /// Agent positions in the ambush.
    pub positions: Vec<AmbushPosition>,
    /// Trigger condition for the ambush.
    pub trigger: AmbushTrigger,
    /// Whether the ambush has been sprung.
    pub sprung: bool,
    /// Time the ambush was set up.
    pub setup_time: f64,
    /// Quality score of the ambush (0..1).
    pub quality: f32,
}

/// A single agent's position in an ambush.
#[derive(Debug, Clone)]
pub struct AmbushPosition {
    /// Agent entity ID.
    pub agent_id: u64,
    /// Position to wait at.
    pub position: Vec3,
    /// Direction to face.
    pub facing: Vec3,
    /// Cover type available.
    pub cover: Option<CoverType>,
    /// Field of fire arc (half-angle in radians).
    pub fire_arc: f32,
    /// Role in the ambush.
    pub role: AmbushRole,
    /// Whether this agent is in position.
    pub in_position: bool,
}

/// Role of an agent in an ambush.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmbushRole {
    /// Primary shooter — fires on trigger.
    Shooter,
    /// Blocks escape routes.
    Blocker,
    /// Watches for early detection and alerts the team.
    Spotter,
    /// Draws the target into the kill zone.
    Bait,
    /// Reserves, joins after ambush is sprung.
    Reserve,
}

/// When the ambush is triggered.
#[derive(Debug, Clone)]
pub enum AmbushTrigger {
    /// Trigger when a target enters the kill zone.
    ProximityTrigger { radius: f32 },
    /// Trigger on command.
    Manual,
    /// Trigger after a timer expires.
    Timed { delay: f32 },
    /// Trigger when all agents are in position.
    AllInPosition,
}

impl AmbushSetup {
    /// Create a new ambush setup.
    pub fn new(id: u32, kill_zone_center: Vec3, kill_zone_radius: f32) -> Self {
        Self {
            id,
            kill_zone_center,
            kill_zone_radius,
            positions: Vec::new(),
            trigger: AmbushTrigger::ProximityTrigger {
                radius: kill_zone_radius,
            },
            sprung: false,
            setup_time: 0.0,
            quality: 0.0,
        }
    }

    /// Add an agent position to the ambush.
    pub fn add_position(&mut self, position: AmbushPosition) -> bool {
        if self.positions.len() >= MAX_AMBUSH_AGENTS {
            return false;
        }
        self.positions.push(position);
        true
    }

    /// Check if all agents are in position.
    pub fn all_in_position(&self) -> bool {
        !self.positions.is_empty() && self.positions.iter().all(|p| p.in_position)
    }

    /// Set an agent as being in position.
    pub fn mark_in_position(&mut self, agent_id: u64) {
        if let Some(pos) = self.positions.iter_mut().find(|p| p.agent_id == agent_id) {
            pos.in_position = true;
        }
    }

    /// Check if the ambush should trigger based on a target position.
    pub fn should_trigger(&self, target_pos: Vec3) -> bool {
        if self.sprung {
            return false;
        }

        match &self.trigger {
            AmbushTrigger::ProximityTrigger { radius } => {
                target_pos.distance(self.kill_zone_center) <= *radius
            }
            AmbushTrigger::Manual => false,
            AmbushTrigger::Timed { .. } => false, // handled by timer
            AmbushTrigger::AllInPosition => self.all_in_position(),
        }
    }

    /// Spring the ambush.
    pub fn spring(&mut self) {
        self.sprung = true;
    }

    /// Evaluate the quality of this ambush setup.
    pub fn evaluate_quality(&mut self) {
        if self.positions.is_empty() {
            self.quality = 0.0;
            return;
        }

        let mut score = 0.0;

        // Coverage: how many angles are covered
        let mut covered_angles = HashSet::new();
        for pos in &self.positions {
            let dir = (self.kill_zone_center - pos.position).normalize_or_zero();
            let angle_bucket = (dir.x.atan2(dir.z) * 4.0 / std::f32::consts::PI) as i32;
            covered_angles.insert(angle_bucket);
        }
        let coverage = covered_angles.len() as f32 / 8.0;
        score += coverage * 0.3;

        // Cover quality
        let cover_ratio = self
            .positions
            .iter()
            .filter(|p| p.cover.is_some())
            .count() as f32
            / self.positions.len() as f32;
        score += cover_ratio * 0.3;

        // Role balance
        let has_shooter = self
            .positions
            .iter()
            .any(|p| p.role == AmbushRole::Shooter);
        let has_blocker = self
            .positions
            .iter()
            .any(|p| p.role == AmbushRole::Blocker);
        if has_shooter {
            score += 0.2;
        }
        if has_blocker {
            score += 0.2;
        }

        self.quality = score.clamp(0.0, 1.0);
    }

    /// Get agents by role.
    pub fn agents_with_role(&self, role: AmbushRole) -> Vec<u64> {
        self.positions
            .iter()
            .filter(|p| p.role == role)
            .map(|p| p.agent_id)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// RetreatPath
// ---------------------------------------------------------------------------

/// A retreat path from a combat position to a safe zone.
#[derive(Debug, Clone)]
pub struct RetreatPath {
    /// Ordered positions along the retreat path.
    pub waypoints: Vec<Vec3>,
    /// Total distance of the path.
    pub total_distance: f32,
    /// Average cover quality along the path (0..1).
    pub average_cover: f32,
    /// Maximum exposure along the path (0..1).
    pub max_exposure: f32,
    /// Destination safety score (0..1).
    pub destination_safety: f32,
    /// Overall path score (0..1).
    pub score: f32,
}

impl RetreatPath {
    /// Create a new retreat path.
    pub fn new(waypoints: Vec<Vec3>) -> Self {
        let total_distance = waypoints
            .windows(2)
            .map(|w| w[0].distance(w[1]))
            .sum();

        Self {
            waypoints,
            total_distance,
            average_cover: 0.0,
            max_exposure: 1.0,
            destination_safety: 0.0,
            score: 0.0,
        }
    }

    /// Evaluate this retreat path against known threats.
    pub fn evaluate(&mut self, threats: &[ThreatInfo], safe_zones: &[Vec3]) {
        if self.waypoints.is_empty() {
            self.score = 0.0;
            return;
        }

        // Calculate average exposure along path
        let mut total_exposure = 0.0;
        let mut max_exposure = 0.0f32;
        let sample_count = self.waypoints.len();

        for wp in &self.waypoints {
            let exposure = calculate_point_exposure(*wp, threats);
            total_exposure += exposure;
            max_exposure = max_exposure.max(exposure);
        }

        self.max_exposure = max_exposure;
        self.average_cover = 1.0 - total_exposure / sample_count as f32;

        // Destination safety: distance to nearest safe zone
        if let Some(dest) = self.waypoints.last() {
            let min_safe_dist = safe_zones
                .iter()
                .map(|sz| dest.distance(*sz))
                .fold(f32::MAX, f32::min);
            self.destination_safety = (1.0 - min_safe_dist / 100.0).clamp(0.0, 1.0);
        }

        // Overall score
        let distance_penalty = (self.total_distance / 50.0).clamp(0.0, 0.5);
        self.score = self.average_cover * 0.3
            + (1.0 - self.max_exposure) * 0.3
            + self.destination_safety * 0.3
            - distance_penalty * 0.1;
        self.score = self.score.clamp(0.0, 1.0);
    }

    /// Get the first waypoint (starting position).
    pub fn start(&self) -> Option<Vec3> {
        self.waypoints.first().copied()
    }

    /// Get the last waypoint (destination).
    pub fn destination(&self) -> Option<Vec3> {
        self.waypoints.last().copied()
    }
}

// ---------------------------------------------------------------------------
// TacticalCell
// ---------------------------------------------------------------------------

/// A single cell in the tactical grid map.
#[derive(Debug, Clone)]
pub struct TacticalCell {
    /// Grid coordinates.
    pub x: usize,
    pub z: usize,
    /// World position of the cell center.
    pub world_position: Vec3,
    /// Threat level at this cell (0..1).
    pub threat_level: f32,
    /// Cover quality available (0..1).
    pub cover_quality: f32,
    /// Visibility score (0 = hidden, 1 = visible from many directions).
    pub visibility: f32,
    /// Elevation relative to neighbors.
    pub elevation: f32,
    /// Whether this cell is traversable.
    pub walkable: bool,
    /// Tactical desirability (0 = avoid, 1 = highly desirable).
    pub desirability: f32,
    /// Cover points in this cell.
    pub cover_points: Vec<usize>,
    /// Last time this cell was updated.
    pub last_update: f64,
    /// Custom tags for game-specific tactical marking.
    pub tags: u32,
}

impl TacticalCell {
    /// Create a new empty tactical cell.
    pub fn new(x: usize, z: usize, world_position: Vec3) -> Self {
        Self {
            x,
            z,
            world_position,
            threat_level: 0.0,
            cover_quality: 0.0,
            visibility: 0.5,
            elevation: 0.0,
            walkable: true,
            desirability: 0.5,
            cover_points: Vec::new(),
            last_update: 0.0,
            tags: 0,
        }
    }

    /// Update the desirability based on current values.
    pub fn update_desirability(&mut self) {
        if !self.walkable {
            self.desirability = 0.0;
            return;
        }

        // High cover, low threat, low visibility = desirable for defense
        self.desirability = self.cover_quality * 0.4
            + (1.0 - self.threat_level) * 0.4
            + (1.0 - self.visibility) * 0.2;
        self.desirability = self.desirability.clamp(0.0, 1.0);
    }

    /// Check if the cell has a specific tag.
    pub fn has_tag(&self, tag: u32) -> bool {
        (self.tags & tag) != 0
    }

    /// Set a tag.
    pub fn set_tag(&mut self, tag: u32) {
        self.tags |= tag;
    }

    /// Clear a tag.
    pub fn clear_tag(&mut self, tag: u32) {
        self.tags &= !tag;
    }
}

// Tactical tag constants
pub const TAG_CHOKEPOINT: u32 = 1 << 0;
pub const TAG_SNIPER_NEST: u32 = 1 << 1;
pub const TAG_OBJECTIVE: u32 = 1 << 2;
pub const TAG_SPAWN_AREA: u32 = 1 << 3;
pub const TAG_DANGER_ZONE: u32 = 1 << 4;
pub const TAG_SAFE_ZONE: u32 = 1 << 5;
pub const TAG_FLANK_ROUTE: u32 = 1 << 6;
pub const TAG_HIGH_GROUND: u32 = 1 << 7;

// ---------------------------------------------------------------------------
// TacticalMap
// ---------------------------------------------------------------------------

/// Grid-based tactical analysis map.
pub struct TacticalMap {
    /// Grid width (number of cells on the X axis).
    pub width: usize,
    /// Grid height (number of cells on the Z axis).
    pub height: usize,
    /// Cell size in world units.
    pub cell_size: f32,
    /// World-space origin of the grid (bottom-left corner).
    pub origin: Vec3,
    /// Grid cells.
    cells: Vec<TacticalCell>,
    /// All cover points registered on the map.
    cover_points: Vec<CoverPoint>,
    /// Known threats.
    threats: Vec<ThreatInfo>,
    /// Safe zones for retreat calculations.
    safe_zones: Vec<Vec3>,
    /// Whether the map needs a full refresh.
    dirty: bool,
}

impl TacticalMap {
    /// Create a new tactical map.
    pub fn new(width: usize, height: usize, cell_size: f32, origin: Vec3) -> Self {
        let w = width.min(MAX_GRID_SIZE);
        let h = height.min(MAX_GRID_SIZE);

        let mut cells = Vec::with_capacity(w * h);
        for z in 0..h {
            for x in 0..w {
                let world_pos = Vec3::new(
                    origin.x + x as f32 * cell_size + cell_size * 0.5,
                    origin.y,
                    origin.z + z as f32 * cell_size + cell_size * 0.5,
                );
                cells.push(TacticalCell::new(x, z, world_pos));
            }
        }

        Self {
            width: w,
            height: h,
            cell_size,
            origin,
            cells,
            cover_points: Vec::new(),
            threats: Vec::new(),
            safe_zones: Vec::new(),
            dirty: true,
        }
    }

    /// Get a cell at grid coordinates.
    pub fn get_cell(&self, x: usize, z: usize) -> Option<&TacticalCell> {
        if x < self.width && z < self.height {
            Some(&self.cells[z * self.width + x])
        } else {
            None
        }
    }

    /// Get a mutable cell at grid coordinates.
    pub fn get_cell_mut(&mut self, x: usize, z: usize) -> Option<&mut TacticalCell> {
        if x < self.width && z < self.height {
            Some(&mut self.cells[z * self.width + x])
        } else {
            None
        }
    }

    /// Convert a world position to grid coordinates.
    pub fn world_to_grid(&self, world_pos: Vec3) -> Option<(usize, usize)> {
        let local = world_pos - self.origin;
        let x = (local.x / self.cell_size) as i32;
        let z = (local.z / self.cell_size) as i32;

        if x >= 0 && x < self.width as i32 && z >= 0 && z < self.height as i32 {
            Some((x as usize, z as usize))
        } else {
            None
        }
    }

    /// Convert grid coordinates to a world position (cell center).
    pub fn grid_to_world(&self, x: usize, z: usize) -> Vec3 {
        Vec3::new(
            self.origin.x + x as f32 * self.cell_size + self.cell_size * 0.5,
            self.origin.y,
            self.origin.z + z as f32 * self.cell_size + self.cell_size * 0.5,
        )
    }

    /// Get the cell at a world position.
    pub fn cell_at_world(&self, world_pos: Vec3) -> Option<&TacticalCell> {
        let (x, z) = self.world_to_grid(world_pos)?;
        self.get_cell(x, z)
    }

    // -----------------------------------------------------------------------
    // Cover points
    // -----------------------------------------------------------------------

    /// Register a cover point on the map.
    pub fn add_cover_point(&mut self, point: CoverPoint) -> usize {
        if self.cover_points.len() >= MAX_COVER_POINTS {
            return usize::MAX;
        }
        let idx = self.cover_points.len();
        // Link to grid cell
        if let Some((gx, gz)) = self.world_to_grid(point.position) {
            if let Some(cell) = self.get_cell_mut(gx, gz) {
                cell.cover_points.push(idx);
            }
        }
        self.cover_points.push(point);
        self.dirty = true;
        idx
    }

    /// Get all cover points.
    pub fn cover_points(&self) -> &[CoverPoint] {
        &self.cover_points
    }

    /// Find the best cover point for an agent, considering their position
    /// and current threats.
    pub fn find_best_cover(
        &mut self,
        agent_pos: Vec3,
        max_distance: f32,
    ) -> Option<&CoverPoint> {
        // Re-evaluate cover points against current threats
        let threats = self.threats.clone();
        for point in &mut self.cover_points {
            if agent_pos.distance(point.position) <= max_distance {
                point.evaluate(agent_pos, &threats);
            }
        }

        self.cover_points
            .iter()
            .filter(|cp| {
                cp.is_suitable() && agent_pos.distance(cp.position) <= max_distance
            })
            .max_by(|a, b| {
                a.quality
                    .partial_cmp(&b.quality)
                    .unwrap_or(Ordering::Equal)
            })
    }

    /// Find the N best cover points within range.
    pub fn find_best_cover_n(
        &mut self,
        agent_pos: Vec3,
        max_distance: f32,
        n: usize,
    ) -> Vec<&CoverPoint> {
        let threats = self.threats.clone();
        for point in &mut self.cover_points {
            if agent_pos.distance(point.position) <= max_distance {
                point.evaluate(agent_pos, &threats);
            }
        }

        let mut candidates: Vec<&CoverPoint> = self
            .cover_points
            .iter()
            .filter(|cp| {
                cp.is_suitable() && agent_pos.distance(cp.position) <= max_distance
            })
            .collect();

        candidates.sort_by(|a, b| {
            b.quality
                .partial_cmp(&a.quality)
                .unwrap_or(Ordering::Equal)
        });

        candidates.truncate(n);
        candidates
    }

    // -----------------------------------------------------------------------
    // Threats
    // -----------------------------------------------------------------------

    /// Update the threat list.
    pub fn set_threats(&mut self, threats: Vec<ThreatInfo>) {
        self.threats = threats;
        self.dirty = true;
    }

    /// Add a single threat.
    pub fn add_threat(&mut self, threat: ThreatInfo) {
        if self.threats.len() < MAX_THREATS {
            self.threats.push(threat);
            self.dirty = true;
        }
    }

    /// Remove a threat by entity ID.
    pub fn remove_threat(&mut self, entity_id: u64) {
        self.threats.retain(|t| t.entity_id != entity_id);
        self.dirty = true;
    }

    /// Get current threats.
    pub fn threats(&self) -> &[ThreatInfo] {
        &self.threats
    }

    // -----------------------------------------------------------------------
    // Safe zones
    // -----------------------------------------------------------------------

    /// Set safe zones for retreat path calculation.
    pub fn set_safe_zones(&mut self, zones: Vec<Vec3>) {
        self.safe_zones = zones;
    }

    /// Add a safe zone.
    pub fn add_safe_zone(&mut self, position: Vec3) {
        self.safe_zones.push(position);
    }

    // -----------------------------------------------------------------------
    // Firing positions
    // -----------------------------------------------------------------------

    /// Find the best firing position to engage a target.
    pub fn find_firing_position(
        &self,
        agent_pos: Vec3,
        target: &ThreatInfo,
        ideal_range: f32,
        weapon_range: f32,
        search_radius: f32,
    ) -> Option<FiringPosition> {
        let mut best: Option<FiringPosition> = None;

        // Sample cells within search radius
        if let Some((ax, az)) = self.world_to_grid(agent_pos) {
            let cell_radius = (search_radius / self.cell_size).ceil() as i32;

            for dz in -cell_radius..=cell_radius {
                for dx in -cell_radius..=cell_radius {
                    let gx = ax as i32 + dx;
                    let gz = az as i32 + dz;

                    if gx < 0 || gx >= self.width as i32 || gz < 0 || gz >= self.height as i32 {
                        continue;
                    }

                    let cell = &self.cells[gz as usize * self.width + gx as usize];
                    if !cell.walkable {
                        continue;
                    }

                    let pos = cell.world_position;
                    if agent_pos.distance(pos) > search_radius {
                        continue;
                    }

                    let mut fp = FiringPosition::new(pos, target.entity_id, target.position);

                    // Determine cover at this position
                    for &cp_idx in &cell.cover_points {
                        if cp_idx < self.cover_points.len() {
                            let cp = &self.cover_points[cp_idx];
                            if !cp.occupied {
                                fp.cover_available = Some(cp.cover_type);
                                break;
                            }
                        }
                    }

                    // Angle quality: prefer positions that face the target's flank or rear
                    let to_target = (target.position - pos).normalize_or_zero();
                    let facing_dot = target.facing.dot(to_target);
                    fp.angle_quality = ((1.0 - facing_dot) / 2.0).clamp(0.0, 1.0);

                    fp.evaluate(ideal_range, weapon_range);

                    if best
                        .as_ref()
                        .map_or(true, |b| fp.score > b.score)
                    {
                        best = Some(fp);
                    }
                }
            }
        }

        best
    }

    // -----------------------------------------------------------------------
    // Retreat paths
    // -----------------------------------------------------------------------

    /// Find a retreat path from the current position to the nearest safe zone.
    pub fn find_retreat_path(&self, from: Vec3, max_path_length: usize) -> Option<RetreatPath> {
        if self.safe_zones.is_empty() {
            return None;
        }

        // Find nearest safe zone
        let target = self
            .safe_zones
            .iter()
            .min_by(|a, b| {
                from.distance(**a)
                    .partial_cmp(&from.distance(**b))
                    .unwrap_or(Ordering::Equal)
            })?;

        // Simple A*-like path toward safe zone, preferring low-threat cells
        let path = self.find_tactical_path(from, *target, max_path_length)?;

        let mut retreat = RetreatPath::new(path);
        retreat.evaluate(&self.threats, &self.safe_zones);
        Some(retreat)
    }

    /// Find a path that considers tactical values (threat, cover).
    fn find_tactical_path(
        &self,
        from: Vec3,
        to: Vec3,
        max_steps: usize,
    ) -> Option<Vec<Vec3>> {
        let start = self.world_to_grid(from)?;
        let goal = self.world_to_grid(to)?;

        #[derive(Clone, PartialEq)]
        struct Node {
            x: usize,
            z: usize,
            g: f32,
            f: f32,
        }

        impl Eq for Node {}

        impl PartialOrd for Node {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for Node {
            fn cmp(&self, other: &Self) -> Ordering {
                other
                    .f
                    .partial_cmp(&self.f)
                    .unwrap_or(Ordering::Equal)
            }
        }

        let mut open = BinaryHeap::new();
        let mut came_from: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        let mut g_score: HashMap<(usize, usize), f32> = HashMap::new();

        open.push(Node {
            x: start.0,
            z: start.1,
            g: 0.0,
            f: 0.0,
        });
        g_score.insert(start, 0.0);

        let neighbors = [(0i32, 1i32), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)];

        let mut iterations = 0;

        while let Some(current) = open.pop() {
            iterations += 1;
            if iterations > max_steps * 10 {
                break;
            }

            if (current.x, current.z) == goal {
                // Reconstruct path
                let mut path = Vec::new();
                let mut pos = goal;
                path.push(self.grid_to_world(pos.0, pos.1));
                while let Some(&prev) = came_from.get(&pos) {
                    path.push(self.grid_to_world(prev.0, prev.1));
                    pos = prev;
                    if path.len() > max_steps {
                        break;
                    }
                }
                path.reverse();
                return Some(path);
            }

            for &(dx, dz) in &neighbors {
                let nx = current.x as i32 + dx;
                let nz = current.z as i32 + dz;

                if nx < 0 || nx >= self.width as i32 || nz < 0 || nz >= self.height as i32 {
                    continue;
                }

                let nx = nx as usize;
                let nz = nz as usize;
                let cell = &self.cells[nz * self.width + nx];

                if !cell.walkable {
                    continue;
                }

                // Movement cost includes tactical penalty for high-threat cells
                let base_cost = if dx.abs() + dz.abs() == 2 {
                    1.414
                } else {
                    1.0
                };
                let threat_penalty = cell.threat_level * 3.0;
                let move_cost = base_cost + threat_penalty as f64;

                let tentative_g = current.g + move_cost as f32;
                let prev_g = g_score.get(&(nx, nz)).copied().unwrap_or(f32::MAX);

                if tentative_g < prev_g {
                    came_from.insert((nx, nz), (current.x, current.z));
                    g_score.insert((nx, nz), tentative_g);

                    let h = ((nx as f32 - goal.0 as f32).powi(2)
                        + (nz as f32 - goal.1 as f32).powi(2))
                    .sqrt();

                    open.push(Node {
                        x: nx,
                        z: nz,
                        g: tentative_g,
                        f: tentative_g + h,
                    });
                }
            }
        }

        None
    }

    // -----------------------------------------------------------------------
    // Threat map update
    // -----------------------------------------------------------------------

    /// Recalculate threat levels on all cells based on current threats.
    pub fn update_threat_map(&mut self, game_time: f64) {
        for cell in &mut self.cells {
            cell.threat_level = 0.0;
        }

        for threat in &self.threats {
            if let Some((tx, tz)) = self.world_to_grid(threat.position) {
                let radius_cells =
                    (threat.weapon_range / self.cell_size).ceil() as i32;

                for dz in -radius_cells..=radius_cells {
                    for dx in -radius_cells..=radius_cells {
                        let gx = tx as i32 + dx;
                        let gz = tz as i32 + dz;

                        if gx < 0
                            || gx >= self.width as i32
                            || gz < 0
                            || gz >= self.height as i32
                        {
                            continue;
                        }

                        let cell = &mut self.cells[gz as usize * self.width + gx as usize];
                        let dist = cell.world_position.distance(threat.position);

                        if dist <= threat.weapon_range {
                            let falloff = 1.0 - (dist / threat.weapon_range);
                            let contribution = threat.threat_level * falloff * falloff;
                            cell.threat_level =
                                (cell.threat_level + contribution).clamp(0.0, 1.0);
                        }

                        cell.last_update = game_time;
                    }
                }
            }
        }

        // Update desirability
        for cell in &mut self.cells {
            cell.update_desirability();
        }

        self.dirty = false;
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Find the cell with the highest desirability within a radius.
    pub fn most_desirable_cell(
        &self,
        center: Vec3,
        radius: f32,
    ) -> Option<&TacticalCell> {
        self.cells
            .iter()
            .filter(|c| c.walkable && c.world_position.distance(center) <= radius)
            .max_by(|a, b| {
                a.desirability
                    .partial_cmp(&b.desirability)
                    .unwrap_or(Ordering::Equal)
            })
    }

    /// Find the cell with the lowest threat within a radius.
    pub fn safest_cell(
        &self,
        center: Vec3,
        radius: f32,
    ) -> Option<&TacticalCell> {
        self.cells
            .iter()
            .filter(|c| c.walkable && c.world_position.distance(center) <= radius)
            .min_by(|a, b| {
                a.threat_level
                    .partial_cmp(&b.threat_level)
                    .unwrap_or(Ordering::Equal)
            })
    }

    /// Find cells matching a tag within a radius.
    pub fn cells_with_tag(
        &self,
        center: Vec3,
        radius: f32,
        tag: u32,
    ) -> Vec<&TacticalCell> {
        self.cells
            .iter()
            .filter(|c| {
                c.walkable
                    && c.has_tag(tag)
                    && c.world_position.distance(center) <= radius
            })
            .collect()
    }

    /// Get the threat level at a world position.
    pub fn threat_at(&self, pos: Vec3) -> f32 {
        self.cell_at_world(pos)
            .map_or(0.0, |c| c.threat_level)
    }

    /// Get the cover quality at a world position.
    pub fn cover_at(&self, pos: Vec3) -> f32 {
        self.cell_at_world(pos)
            .map_or(0.0, |c| c.cover_quality)
    }

    /// Total number of cells.
    pub fn cell_count(&self) -> usize {
        self.width * self.height
    }

    /// Check if the map needs a refresh.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Calculate the exposure of a point given a set of threats.
fn calculate_point_exposure(point: Vec3, threats: &[ThreatInfo]) -> f32 {
    if threats.is_empty() {
        return 0.0;
    }

    let mut total = 0.0;
    for threat in threats {
        let dist = point.distance(threat.position);
        if dist <= threat.weapon_range {
            let normalized = dist / threat.weapon_range;
            total += (1.0 - normalized) * threat.threat_level;
        }
    }

    (total / threats.len() as f32).clamp(0.0, 1.0)
}

/// Calculate a flanking score for a position relative to a threat.
/// Returns 0 if directly in front, 1 if directly behind.
pub fn flanking_score(position: Vec3, threat: &ThreatInfo) -> f32 {
    let to_pos = (position - threat.position).normalize_or_zero();
    let dot = threat.facing.dot(to_pos);
    // dot = 1.0 means directly in front, -1.0 means directly behind
    ((1.0 - dot) / 2.0).clamp(0.0, 1.0)
}

/// Check if a position provides a crossfire angle with an ally against a target.
pub fn has_crossfire(
    position_a: Vec3,
    position_b: Vec3,
    target: Vec3,
) -> bool {
    let dir_a = (target - position_a).normalize_or_zero();
    let dir_b = (target - position_b).normalize_or_zero();
    let angle = dir_a.dot(dir_b).acos();
    // Good crossfire requires at least 45 degrees separation
    angle > std::f32::consts::FRAC_PI_4
}

/// Evaluate how suitable a position is for overwatch (covering an area).
pub fn overwatch_score(
    position: Vec3,
    watch_area_center: Vec3,
    watch_area_radius: f32,
) -> f32 {
    let dist = position.distance(watch_area_center);
    let height_advantage = (position.y - watch_area_center.y).max(0.0);

    // Ideal overwatch: elevated, at medium range
    let range_score = if dist < watch_area_radius {
        0.3 // Too close
    } else if dist < watch_area_radius * 3.0 {
        1.0 // Ideal range
    } else {
        (1.0 - (dist - watch_area_radius * 3.0) / (watch_area_radius * 3.0)).max(0.0)
    };

    let height_score = (height_advantage / 10.0).clamp(0.0, 0.3);

    range_score * 0.7 + height_score + 0.0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cover_point_evaluation() {
        let mut cover = CoverPoint::new(
            Vec3::new(5.0, 0.0, 5.0),
            Vec3::new(-1.0, 0.0, 0.0),
            CoverType::Half,
        );

        let threats = vec![ThreatInfo::new(
            1,
            Vec3::new(20.0, 0.0, 5.0),
            0.8,
        )];

        cover.evaluate(Vec3::ZERO, &threats);
        assert!(cover.quality > 0.0);
        assert!(cover.exposure < 1.0);
    }

    #[test]
    fn test_tactical_map_grid_conversion() {
        let map = TacticalMap::new(10, 10, 2.0, Vec3::ZERO);
        let world_pos = Vec3::new(5.0, 0.0, 7.0);
        let grid = map.world_to_grid(world_pos).unwrap();
        assert_eq!(grid, (2, 3));

        let back = map.grid_to_world(2, 3);
        assert!((back.x - 5.0).abs() < 2.0);
        assert!((back.z - 7.0).abs() < 2.0);
    }

    #[test]
    fn test_patrol_route() {
        let mut route = PatrolRoute::new(1, "test_patrol", PatrolMode::Loop);
        route.add_point(Vec3::new(0.0, 0.0, 0.0), 2.0);
        route.add_point(Vec3::new(10.0, 0.0, 0.0), 2.0);
        route.add_point(Vec3::new(10.0, 0.0, 10.0), 2.0);

        assert_eq!(route.waypoint_count(), 3);
        assert_eq!(route.advance(), 1);
        assert_eq!(route.advance(), 2);
        assert_eq!(route.advance(), 0); // loops back
    }

    #[test]
    fn test_flanking_score() {
        let threat = ThreatInfo {
            entity_id: 1,
            position: Vec3::new(0.0, 0.0, 0.0),
            facing: Vec3::new(1.0, 0.0, 0.0),
            threat_level: 1.0,
            weapon_range: 50.0,
            has_los: true,
            velocity: Vec3::ZERO,
            last_seen: 0.0,
            suppressed: false,
        };

        // Directly in front
        let front_score = flanking_score(Vec3::new(10.0, 0.0, 0.0), &threat);
        assert!(front_score < 0.1);

        // Directly behind
        let behind_score = flanking_score(Vec3::new(-10.0, 0.0, 0.0), &threat);
        assert!(behind_score > 0.9);
    }

    #[test]
    fn test_ambush_setup() {
        let mut ambush = AmbushSetup::new(1, Vec3::new(50.0, 0.0, 50.0), 10.0);

        ambush.add_position(AmbushPosition {
            agent_id: 1,
            position: Vec3::new(40.0, 0.0, 50.0),
            facing: Vec3::new(1.0, 0.0, 0.0),
            cover: Some(CoverType::Full),
            fire_arc: 0.5,
            role: AmbushRole::Shooter,
            in_position: false,
        });

        ambush.add_position(AmbushPosition {
            agent_id: 2,
            position: Vec3::new(60.0, 0.0, 50.0),
            facing: Vec3::new(-1.0, 0.0, 0.0),
            cover: Some(CoverType::Half),
            fire_arc: 0.5,
            role: AmbushRole::Blocker,
            in_position: false,
        });

        assert!(!ambush.all_in_position());
        ambush.mark_in_position(1);
        ambush.mark_in_position(2);
        assert!(ambush.all_in_position());

        ambush.evaluate_quality();
        assert!(ambush.quality > 0.5);
    }
}
