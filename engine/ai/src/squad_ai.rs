//! Squad-level AI coordination for group tactics.
//!
//! Provides:
//! - **Squad composition**: named squads with member management
//! - **Squad formation manager**: dynamic formations for movement
//! - **Cover assignment**: distribute cover points among squad members
//! - **Fire-and-move tactics**: alternating fire and movement teams
//! - **Suppressive fire**: coordinated suppression of enemy positions
//! - **Flanking maneuvers**: detect and execute flanking opportunities
//! - **Rally points**: fallback positions for regrouping
//! - **Squad health tracking**: aggregate health and casualty awareness
//! - **Medic behavior**: prioritize healing wounded squad members
//! - **Squad orders**: advance, hold, retreat, regroup
//! - **ECS integration**: `SquadComponent`, `SquadAISystem`
//!
//! # Design
//!
//! A [`Squad`] is a group of AI agents that coordinate their behavior.
//! The [`SquadAI`] issues [`SquadOrder`]s and coordinates tactics such as
//! flanking, suppression, and fire-and-move. Individual members receive
//! [`MemberOrder`]s derived from the squad-level strategy.

use glam::Vec3;
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum squad size.
pub const MAX_SQUAD_SIZE: usize = 12;
/// Maximum number of squads.
pub const MAX_SQUADS: usize = 32;
/// Default squad formation spacing.
pub const DEFAULT_FORMATION_SPACING: f32 = 3.0;
/// Maximum rally points per squad.
pub const MAX_RALLY_POINTS: usize = 4;
/// Suppression fire duration (seconds).
pub const DEFAULT_SUPPRESSION_DURATION: f32 = 5.0;
/// Flanking angle threshold (radians).
pub const FLANKING_ANGLE_THRESHOLD: f32 = 1.2;
/// Maximum engagement range.
pub const MAX_ENGAGEMENT_RANGE: f32 = 50.0;
/// Cover assignment update interval (seconds).
pub const COVER_UPDATE_INTERVAL: f32 = 2.0;
/// Low health threshold for medic behavior.
pub const LOW_HEALTH_THRESHOLD: f32 = 0.3;
/// Critical health threshold.
pub const CRITICAL_HEALTH_THRESHOLD: f32 = 0.15;
/// Small epsilon.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// SquadId, MemberId
// ---------------------------------------------------------------------------

/// Unique squad identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SquadId(pub u64);

/// A squad member is identified by their entity ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemberId(pub u64);

// ---------------------------------------------------------------------------
// SquadRole
// ---------------------------------------------------------------------------

/// Role of a squad member.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SquadRole {
    /// Squad leader (makes tactical decisions).
    Leader,
    /// Assault: close-range combat.
    Assault,
    /// Rifleman: general-purpose.
    Rifleman,
    /// Support: suppressive fire, heavy weapons.
    Support,
    /// Sniper: long-range precision.
    Sniper,
    /// Medic: heals teammates.
    Medic,
    /// Engineer: utility, traps, repairs.
    Engineer,
    /// Scout: reconnaissance, flanking.
    Scout,
    /// Heavy: armor, close-range devastation.
    Heavy,
}

impl SquadRole {
    /// Whether this role should be in the front line.
    pub fn is_front_line(&self) -> bool {
        matches!(self, Self::Assault | Self::Heavy | Self::Leader)
    }

    /// Whether this role should stay in the rear.
    pub fn is_rear(&self) -> bool {
        matches!(self, Self::Sniper | Self::Medic | Self::Support)
    }

    /// Priority for cover assignment (lower = gets cover first).
    pub fn cover_priority(&self) -> u32 {
        match self {
            Self::Sniper => 0,
            Self::Support => 1,
            Self::Leader => 2,
            Self::Rifleman => 3,
            Self::Medic => 4,
            Self::Assault => 5,
            Self::Scout => 6,
            Self::Engineer => 7,
            Self::Heavy => 8,
        }
    }

    /// Display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Leader => "Leader",
            Self::Assault => "Assault",
            Self::Rifleman => "Rifleman",
            Self::Support => "Support",
            Self::Sniper => "Sniper",
            Self::Medic => "Medic",
            Self::Engineer => "Engineer",
            Self::Scout => "Scout",
            Self::Heavy => "Heavy",
        }
    }
}

// ---------------------------------------------------------------------------
// MemberState
// ---------------------------------------------------------------------------

/// Current state of a squad member.
#[derive(Debug, Clone)]
pub struct MemberState {
    /// Entity ID.
    pub id: MemberId,
    /// Squad role.
    pub role: SquadRole,
    /// Current position.
    pub position: Vec3,
    /// Current health fraction (0..1).
    pub health: f32,
    /// Whether the member is alive.
    pub alive: bool,
    /// Whether the member is in cover.
    pub in_cover: bool,
    /// Assigned cover point (if any).
    pub cover_point: Option<Vec3>,
    /// Current individual order.
    pub current_order: MemberOrder,
    /// Whether the member is suppressed (under heavy fire).
    pub suppressed: bool,
    /// Ammo fraction (0..1).
    pub ammo: f32,
    /// Whether this member can see an enemy.
    pub has_line_of_sight: bool,
    /// Current target entity ID.
    pub target: Option<MemberId>,
    /// Whether the member needs healing.
    pub needs_healing: bool,
    /// Team assignment (for fire-and-move: 0 = Alpha, 1 = Bravo).
    pub team: u8,
    /// Time since last fired.
    pub time_since_fire: f32,
}

impl MemberState {
    /// Create a new member state.
    pub fn new(id: MemberId, role: SquadRole, position: Vec3) -> Self {
        Self {
            id,
            role,
            position,
            health: 1.0,
            alive: true,
            in_cover: false,
            cover_point: None,
            current_order: MemberOrder::Hold,
            suppressed: false,
            ammo: 1.0,
            has_line_of_sight: false,
            target: None,
            needs_healing: false,
            team: 0,
            time_since_fire: 0.0,
        }
    }

    /// Check if this member is combat-effective.
    pub fn is_effective(&self) -> bool {
        self.alive && self.health > CRITICAL_HEALTH_THRESHOLD && self.ammo > 0.05
    }

    /// Check if this member needs medical attention.
    pub fn needs_medic(&self) -> bool {
        self.alive && self.health < LOW_HEALTH_THRESHOLD
    }
}

// ---------------------------------------------------------------------------
// MemberOrder
// ---------------------------------------------------------------------------

/// An order issued to an individual squad member.
#[derive(Debug, Clone)]
pub enum MemberOrder {
    /// Hold current position.
    Hold,
    /// Move to a specific position.
    MoveTo(Vec3),
    /// Move to cover.
    MoveToCover(Vec3),
    /// Fire at a target.
    FireAt(MemberId),
    /// Suppress a position.
    Suppress(Vec3),
    /// Flank to a position and engage.
    Flank { move_to: Vec3, target: Vec3 },
    /// Retreat to a rally point.
    Retreat(Vec3),
    /// Regroup at squad leader's position.
    Regroup,
    /// Heal a teammate.
    Heal(MemberId),
    /// Scout/reconnoiter an area.
    Scout(Vec3),
    /// Go prone/take cover.
    TakeCover,
    /// Follow another member.
    Follow(MemberId),
    /// Overwatch (watch a direction for enemies).
    Overwatch { position: Vec3, direction: Vec3 },
}

// ---------------------------------------------------------------------------
// SquadOrder
// ---------------------------------------------------------------------------

/// A high-level order issued to the entire squad.
#[derive(Debug, Clone)]
pub enum SquadOrder {
    /// Advance toward objective.
    Advance { target: Vec3 },
    /// Hold current position.
    Hold,
    /// Retreat to rally point.
    Retreat { rally_point: Vec3 },
    /// Regroup at leader.
    Regroup,
    /// Execute fire-and-move toward objective.
    FireAndMove { target: Vec3 },
    /// Flank the enemy position.
    Flank { enemy_position: Vec3 },
    /// Suppress enemy position.
    Suppress { position: Vec3, duration: f32 },
    /// Ambush at designated positions.
    Ambush { positions: Vec<Vec3> },
    /// Free engagement (members act independently).
    FreeEngagement,
    /// Fall back and defend a position.
    Defend { position: Vec3 },
    /// Search an area.
    Search { center: Vec3, radius: f32 },
}

// ---------------------------------------------------------------------------
// SquadFormation
// ---------------------------------------------------------------------------

/// Formation shape for squad movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SquadFormationShape {
    /// Line abreast (horizontal line).
    Line,
    /// Column (single file).
    Column,
    /// Wedge (V-shape, leader at front).
    Wedge,
    /// Diamond.
    Diamond,
    /// Staggered column (two offset columns).
    StaggeredColumn,
    /// Circle/perimeter defense.
    Circle,
    /// Echelon (diagonal line).
    EchelonLeft,
    EchelonRight,
}

/// Manages squad formation positions.
#[derive(Debug, Clone)]
pub struct SquadFormation {
    /// Formation shape.
    pub shape: SquadFormationShape,
    /// Spacing between members.
    pub spacing: f32,
    /// Formation center position.
    pub center: Vec3,
    /// Formation facing direction.
    pub facing: Vec3,
}

impl SquadFormation {
    /// Create a new formation.
    pub fn new(shape: SquadFormationShape) -> Self {
        Self {
            shape,
            spacing: DEFAULT_FORMATION_SPACING,
            center: Vec3::ZERO,
            facing: Vec3::new(0.0, 0.0, 1.0),
        }
    }

    /// Calculate formation positions for N members.
    pub fn calculate_positions(&self, member_count: usize) -> Vec<Vec3> {
        let right = Vec3::new(self.facing.z, 0.0, -self.facing.x).normalize_or_zero();
        let forward = Vec3::new(self.facing.x, 0.0, self.facing.z).normalize_or_zero();

        let mut positions = Vec::with_capacity(member_count);

        match self.shape {
            SquadFormationShape::Line => {
                let half = (member_count as f32 - 1.0) / 2.0;
                for i in 0..member_count {
                    let offset = (i as f32 - half) * self.spacing;
                    positions.push(self.center + right * offset);
                }
            }
            SquadFormationShape::Column => {
                for i in 0..member_count {
                    let offset = -(i as f32) * self.spacing;
                    positions.push(self.center + forward * offset);
                }
            }
            SquadFormationShape::Wedge => {
                // Leader at front, members spread behind
                positions.push(self.center); // Leader
                for i in 1..member_count {
                    let side = if i % 2 == 1 { 1.0 } else { -1.0 };
                    let depth = ((i + 1) / 2) as f32;
                    let pos = self.center
                        - forward * depth * self.spacing
                        + right * side * depth * self.spacing * 0.7;
                    positions.push(pos);
                }
            }
            SquadFormationShape::Diamond => {
                positions.push(self.center + forward * self.spacing); // Front
                if member_count > 1 {
                    positions.push(self.center - right * self.spacing); // Left
                }
                if member_count > 2 {
                    positions.push(self.center + right * self.spacing); // Right
                }
                if member_count > 3 {
                    positions.push(self.center - forward * self.spacing); // Rear
                }
                for i in 4..member_count {
                    let angle = (i as f32 / member_count as f32) * std::f32::consts::TAU;
                    positions.push(self.center + Vec3::new(
                        angle.cos() * self.spacing,
                        0.0,
                        angle.sin() * self.spacing,
                    ));
                }
            }
            SquadFormationShape::StaggeredColumn => {
                for i in 0..member_count {
                    let depth = -(i as f32 / 2.0).floor() * self.spacing;
                    let side = if i % 2 == 0 { -1.0 } else { 1.0 };
                    positions.push(self.center + forward * depth + right * side * self.spacing * 0.5);
                }
            }
            SquadFormationShape::Circle => {
                for i in 0..member_count {
                    let angle = (i as f32 / member_count as f32) * std::f32::consts::TAU;
                    positions.push(self.center + Vec3::new(
                        angle.cos() * self.spacing * 2.0,
                        0.0,
                        angle.sin() * self.spacing * 2.0,
                    ));
                }
            }
            SquadFormationShape::EchelonLeft => {
                for i in 0..member_count {
                    let offset = i as f32 * self.spacing;
                    positions.push(self.center - forward * offset - right * offset * 0.5);
                }
            }
            SquadFormationShape::EchelonRight => {
                for i in 0..member_count {
                    let offset = i as f32 * self.spacing;
                    positions.push(self.center - forward * offset + right * offset * 0.5);
                }
            }
        }

        positions
    }
}

// ---------------------------------------------------------------------------
// RallyPoint
// ---------------------------------------------------------------------------

/// A designated fallback/regroup position.
#[derive(Debug, Clone)]
pub struct RallyPoint {
    /// Position.
    pub position: Vec3,
    /// Whether this is the current active rally point.
    pub active: bool,
    /// Priority (lower = preferred).
    pub priority: u32,
    /// Label.
    pub label: String,
    /// Whether the rally point has been compromised.
    pub compromised: bool,
}

// ---------------------------------------------------------------------------
// SquadHealthStatus
// ---------------------------------------------------------------------------

/// Aggregate health information for the squad.
#[derive(Debug, Clone)]
pub struct SquadHealthStatus {
    /// Total members.
    pub total_members: usize,
    /// Alive members.
    pub alive_members: usize,
    /// Wounded members (below low health threshold).
    pub wounded_members: usize,
    /// Average health of alive members.
    pub average_health: f32,
    /// Whether the squad is combat effective (>50% alive and effective).
    pub combat_effective: bool,
    /// Number of casualties this engagement.
    pub casualties: usize,
    /// Whether any member has critical health.
    pub has_critical: bool,
}

// ---------------------------------------------------------------------------
// Squad
// ---------------------------------------------------------------------------

/// A squad of AI agents.
#[derive(Debug, Clone)]
pub struct Squad {
    /// Squad identifier.
    pub id: SquadId,
    /// Squad name.
    pub name: String,
    /// Members.
    pub members: HashMap<MemberId, MemberState>,
    /// Current squad order.
    pub current_order: SquadOrder,
    /// Formation.
    pub formation: SquadFormation,
    /// Rally points.
    pub rally_points: Vec<RallyPoint>,
    /// Known enemy positions.
    pub known_enemies: Vec<(MemberId, Vec3)>,
    /// Squad leader ID.
    pub leader: Option<MemberId>,
    /// Fire-and-move state.
    pub fire_move_state: FireAndMoveState,
    /// Suppression targets.
    pub suppression_targets: Vec<Vec3>,
    /// Time in current order.
    pub order_time: f32,
    /// Engagement active.
    pub in_combat: bool,
    /// Combat start time.
    pub combat_start_time: f32,
    /// Morale (0..1).
    pub morale: f32,
    /// Communication range.
    pub comms_range: f32,
    /// Order history.
    order_history: VecDeque<SquadOrder>,
}

/// Fire-and-move coordination state.
#[derive(Debug, Clone)]
pub struct FireAndMoveState {
    /// Which team is currently moving (0 = Alpha, 1 = Bravo).
    pub moving_team: u8,
    /// Duration each team moves before switching.
    pub phase_duration: f32,
    /// Current phase timer.
    pub phase_timer: f32,
    /// Whether fire-and-move is active.
    pub active: bool,
}

impl Default for FireAndMoveState {
    fn default() -> Self {
        Self {
            moving_team: 0,
            phase_duration: 3.0,
            phase_timer: 0.0,
            active: false,
        }
    }
}

impl Squad {
    /// Create a new squad.
    pub fn new(id: SquadId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            members: HashMap::new(),
            current_order: SquadOrder::Hold,
            formation: SquadFormation::new(SquadFormationShape::Wedge),
            rally_points: Vec::new(),
            known_enemies: Vec::new(),
            leader: None,
            fire_move_state: FireAndMoveState::default(),
            suppression_targets: Vec::new(),
            order_time: 0.0,
            in_combat: false,
            combat_start_time: 0.0,
            morale: 1.0,
            comms_range: 50.0,
            order_history: VecDeque::new(),
        }
    }

    /// Add a member to the squad.
    pub fn add_member(&mut self, member: MemberState) -> bool {
        if self.members.len() >= MAX_SQUAD_SIZE {
            return false;
        }
        let id = member.id;
        // Auto-assign team for fire-and-move
        let team = if self.members.len() % 2 == 0 { 0 } else { 1 };
        let mut member = member;
        member.team = team;

        // Auto-assign leader
        if member.role == SquadRole::Leader || self.leader.is_none() {
            self.leader = Some(id);
        }

        self.members.insert(id, member);
        true
    }

    /// Remove a member from the squad.
    pub fn remove_member(&mut self, id: MemberId) -> Option<MemberState> {
        let removed = self.members.remove(&id);
        if self.leader == Some(id) {
            // Assign new leader
            self.leader = self.members.keys().next().copied();
        }
        removed
    }

    /// Issue a squad order.
    pub fn issue_order(&mut self, order: SquadOrder) {
        self.order_history.push_back(self.current_order.clone());
        if self.order_history.len() > 10 {
            self.order_history.pop_front();
        }
        self.current_order = order;
        self.order_time = 0.0;
    }

    /// Get the leader's position.
    pub fn leader_position(&self) -> Option<Vec3> {
        self.leader.and_then(|id| self.members.get(&id).map(|m| m.position))
    }

    /// Get the squad center of mass.
    pub fn center_of_mass(&self) -> Vec3 {
        let alive: Vec<Vec3> = self.members.values()
            .filter(|m| m.alive)
            .map(|m| m.position)
            .collect();
        if alive.is_empty() {
            return Vec3::ZERO;
        }
        let sum: Vec3 = alive.iter().copied().sum();
        sum / alive.len() as f32
    }

    /// Get health status.
    pub fn health_status(&self) -> SquadHealthStatus {
        let total = self.members.len();
        let alive: Vec<&MemberState> = self.members.values().filter(|m| m.alive).collect();
        let alive_count = alive.len();
        let wounded = alive.iter().filter(|m| m.needs_medic()).count();
        let has_critical = alive.iter().any(|m| m.health < CRITICAL_HEALTH_THRESHOLD);
        let avg_health = if alive_count > 0 {
            alive.iter().map(|m| m.health).sum::<f32>() / alive_count as f32
        } else {
            0.0
        };
        let effective_count = alive.iter().filter(|m| m.is_effective()).count();
        let combat_effective = effective_count as f32 / total.max(1) as f32 > 0.5;

        SquadHealthStatus {
            total_members: total,
            alive_members: alive_count,
            wounded_members: wounded,
            average_health: avg_health,
            combat_effective,
            casualties: total - alive_count,
            has_critical,
        }
    }

    /// Find the best flanking position for the given enemy position.
    pub fn find_flanking_position(&self, enemy_pos: Vec3) -> Option<Vec3> {
        let center = self.center_of_mass();
        let to_enemy = enemy_pos - center;
        let to_enemy_flat = Vec3::new(to_enemy.x, 0.0, to_enemy.z);
        let dist = to_enemy_flat.length();
        if dist < EPSILON {
            return None;
        }

        let perpendicular = Vec3::new(-to_enemy_flat.z, 0.0, to_enemy_flat.x).normalize_or_zero();

        // Try left flank and right flank, pick the one further from known enemies
        let left_flank = enemy_pos + perpendicular * dist * 0.5 - to_enemy_flat.normalize_or_zero() * 5.0;
        let right_flank = enemy_pos - perpendicular * dist * 0.5 - to_enemy_flat.normalize_or_zero() * 5.0;

        // Simple heuristic: pick the side with fewer enemies
        let left_threats = self.known_enemies.iter()
            .filter(|(_, pos)| (*pos - left_flank).length() < 15.0)
            .count();
        let right_threats = self.known_enemies.iter()
            .filter(|(_, pos)| (*pos - right_flank).length() < 15.0)
            .count();

        if left_threats <= right_threats {
            Some(left_flank)
        } else {
            Some(right_flank)
        }
    }

    /// Assign cover points to members based on role priority.
    pub fn assign_cover(&mut self, available_cover: &[Vec3]) {
        if available_cover.is_empty() {
            return;
        }

        // Sort members by cover priority
        let mut members_by_priority: Vec<MemberId> = self.members.keys().copied().collect();
        members_by_priority.sort_by_key(|id| {
            self.members.get(id).map_or(u32::MAX, |m| m.role.cover_priority())
        });

        let mut used_cover: HashSet<usize> = HashSet::new();

        for member_id in members_by_priority {
            if let Some(member) = self.members.get_mut(&member_id) {
                if !member.alive {
                    continue;
                }

                // Find nearest unused cover point
                let mut best_idx = None;
                let mut best_dist = f32::MAX;

                for (i, cover_pos) in available_cover.iter().enumerate() {
                    if used_cover.contains(&i) {
                        continue;
                    }
                    let dist = (member.position - *cover_pos).length();
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = Some(i);
                    }
                }

                if let Some(idx) = best_idx {
                    member.cover_point = Some(available_cover[idx]);
                    member.current_order = MemberOrder::MoveToCover(available_cover[idx]);
                    used_cover.insert(idx);
                }
            }
        }
    }

    /// Update fire-and-move state.
    pub fn update_fire_and_move(&mut self, dt: f32, target: Vec3) {
        if !self.fire_move_state.active {
            return;
        }

        self.fire_move_state.phase_timer += dt;
        if self.fire_move_state.phase_timer >= self.fire_move_state.phase_duration {
            self.fire_move_state.phase_timer = 0.0;
            self.fire_move_state.moving_team = 1 - self.fire_move_state.moving_team;
        }

        let moving_team = self.fire_move_state.moving_team;

        for member in self.members.values_mut() {
            if !member.alive {
                continue;
            }
            if member.team == moving_team {
                // This team moves
                member.current_order = MemberOrder::MoveTo(target);
            } else {
                // This team provides covering fire
                member.current_order = MemberOrder::Suppress(target);
            }
        }
    }

    /// Get the medic member (if any).
    pub fn get_medic(&self) -> Option<MemberId> {
        self.members.iter()
            .find(|(_, m)| m.role == SquadRole::Medic && m.alive)
            .map(|(id, _)| *id)
    }

    /// Find the most wounded member that needs healing.
    pub fn most_wounded(&self) -> Option<MemberId> {
        self.members.iter()
            .filter(|(_, m)| m.alive && m.needs_medic())
            .min_by(|a, b| a.1.health.partial_cmp(&b.1.health).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| *id)
    }

    /// Update the squad (main tick).
    pub fn update(&mut self, dt: f32) -> Vec<SquadEvent> {
        let mut events = Vec::new();
        self.order_time += dt;

        // Update morale based on casualties
        let status = self.health_status();
        if !status.combat_effective && self.morale > 0.3 {
            self.morale = (self.morale - 0.01 * dt).max(0.0);
        }

        // Medic behavior
        if let Some(medic_id) = self.get_medic() {
            if let Some(wounded_id) = self.most_wounded() {
                if medic_id != wounded_id {
                    if let Some(medic) = self.members.get_mut(&medic_id) {
                        medic.current_order = MemberOrder::Heal(wounded_id);
                    }
                    events.push(SquadEvent::MedicDispatched {
                        squad_id: self.id,
                        medic: medic_id,
                        patient: wounded_id,
                    });
                }
            }
        }

        // Process squad order
        match &self.current_order {
            SquadOrder::FireAndMove { target } => {
                let target = *target;
                self.fire_move_state.active = true;
                self.update_fire_and_move(dt, target);
            }
            SquadOrder::Retreat { rally_point } => {
                let rp = *rally_point;
                for member in self.members.values_mut() {
                    if member.alive {
                        member.current_order = MemberOrder::Retreat(rp);
                    }
                }
            }
            SquadOrder::Regroup => {
                if let Some(leader_pos) = self.leader_position() {
                    for member in self.members.values_mut() {
                        if member.alive {
                            member.current_order = MemberOrder::Regroup;
                        }
                    }
                }
            }
            SquadOrder::Suppress { position, .. } => {
                let pos = *position;
                for member in self.members.values_mut() {
                    if member.alive && member.role != SquadRole::Medic {
                        member.current_order = MemberOrder::Suppress(pos);
                    }
                }
            }
            SquadOrder::Hold => {
                for member in self.members.values_mut() {
                    if member.alive {
                        member.current_order = MemberOrder::Hold;
                    }
                }
            }
            _ => {}
        }

        events
    }

    /// Get the number of alive members.
    pub fn alive_count(&self) -> usize {
        self.members.values().filter(|m| m.alive).count()
    }

    /// Get all member IDs.
    pub fn member_ids(&self) -> Vec<MemberId> {
        self.members.keys().copied().collect()
    }

    /// Add a rally point.
    pub fn add_rally_point(&mut self, position: Vec3, label: impl Into<String>) {
        if self.rally_points.len() < MAX_RALLY_POINTS {
            self.rally_points.push(RallyPoint {
                position,
                active: self.rally_points.is_empty(),
                priority: self.rally_points.len() as u32,
                label: label.into(),
                compromised: false,
            });
        }
    }

    /// Get the nearest active rally point.
    pub fn nearest_rally_point(&self, from: Vec3) -> Option<Vec3> {
        self.rally_points.iter()
            .filter(|rp| rp.active && !rp.compromised)
            .min_by(|a, b| {
                let da = (a.position - from).length();
                let db = (b.position - from).length();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|rp| rp.position)
    }
}

// ---------------------------------------------------------------------------
// SquadEvent
// ---------------------------------------------------------------------------

/// Events from the squad AI system.
#[derive(Debug, Clone)]
pub enum SquadEvent {
    /// Squad order was issued.
    OrderIssued { squad_id: SquadId, order: String },
    /// Member was assigned to cover.
    CoverAssigned { squad_id: SquadId, member: MemberId, position: Vec3 },
    /// Medic dispatched to heal a teammate.
    MedicDispatched { squad_id: SquadId, medic: MemberId, patient: MemberId },
    /// Flanking maneuver initiated.
    FlankInitiated { squad_id: SquadId, flankers: Vec<MemberId>, position: Vec3 },
    /// Squad morale changed.
    MoraleChanged { squad_id: SquadId, morale: f32 },
    /// Squad is no longer combat effective.
    CombatIneffective { squad_id: SquadId },
    /// Member was killed.
    MemberKilled { squad_id: SquadId, member: MemberId },
    /// Fire-and-move phase changed.
    FireMovePhaseChanged { squad_id: SquadId, moving_team: u8 },
    /// Suppression started.
    SuppressionStarted { squad_id: SquadId, target: Vec3 },
}

// ---------------------------------------------------------------------------
// SquadAISystem
// ---------------------------------------------------------------------------

/// System managing all squads.
pub struct SquadAISystem {
    /// All squads.
    squads: HashMap<SquadId, Squad>,
    /// Next squad ID.
    next_id: u64,
    /// Events from last update.
    events: Vec<SquadEvent>,
    /// Cover update timer.
    cover_timer: f32,
}

impl SquadAISystem {
    /// Create a new squad AI system.
    pub fn new() -> Self {
        Self {
            squads: HashMap::new(),
            next_id: 1,
            events: Vec::new(),
            cover_timer: 0.0,
        }
    }

    /// Create a new squad.
    pub fn create_squad(&mut self, name: impl Into<String>) -> SquadId {
        let id = SquadId(self.next_id);
        self.next_id += 1;
        self.squads.insert(id, Squad::new(id, name));
        id
    }

    /// Get a squad by ID.
    pub fn get_squad(&self, id: SquadId) -> Option<&Squad> {
        self.squads.get(&id)
    }

    /// Get a squad mutably.
    pub fn get_squad_mut(&mut self, id: SquadId) -> Option<&mut Squad> {
        self.squads.get_mut(&id)
    }

    /// Issue an order to a squad.
    pub fn issue_order(&mut self, squad_id: SquadId, order: SquadOrder) {
        if let Some(squad) = self.squads.get_mut(&squad_id) {
            let order_name = format!("{:?}", &order).chars().take(30).collect::<String>();
            squad.issue_order(order);
            self.events.push(SquadEvent::OrderIssued {
                squad_id,
                order: order_name,
            });
        }
    }

    /// Update all squads.
    pub fn update(&mut self, dt: f32) {
        self.events.clear();
        self.cover_timer += dt;

        let squad_ids: Vec<SquadId> = self.squads.keys().copied().collect();
        for id in squad_ids {
            if let Some(squad) = self.squads.get_mut(&id) {
                let events = squad.update(dt);
                self.events.extend(events);
            }
        }
    }

    /// Get events from last update.
    pub fn events(&self) -> &[SquadEvent] {
        &self.events
    }

    /// Get the number of squads.
    pub fn squad_count(&self) -> usize {
        self.squads.len()
    }
}

// ---------------------------------------------------------------------------
// SquadComponent (ECS)
// ---------------------------------------------------------------------------

/// ECS component for entities that belong to a squad.
#[derive(Debug, Clone)]
pub struct SquadComponent {
    /// Squad ID.
    pub squad_id: SquadId,
    /// Member ID (entity ID).
    pub member_id: MemberId,
    /// Current individual order.
    pub current_order: MemberOrder,
}

impl SquadComponent {
    /// Create a new squad component.
    pub fn new(squad_id: SquadId, member_id: MemberId) -> Self {
        Self {
            squad_id,
            member_id,
            current_order: MemberOrder::Hold,
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
    fn test_squad_creation() {
        let squad = Squad::new(SquadId(1), "Alpha");
        assert_eq!(squad.name, "Alpha");
        assert_eq!(squad.members.len(), 0);
    }

    #[test]
    fn test_add_members() {
        let mut squad = Squad::new(SquadId(1), "Alpha");
        let m1 = MemberState::new(MemberId(1), SquadRole::Leader, Vec3::ZERO);
        let m2 = MemberState::new(MemberId(2), SquadRole::Rifleman, Vec3::new(2.0, 0.0, 0.0));
        squad.add_member(m1);
        squad.add_member(m2);
        assert_eq!(squad.members.len(), 2);
        assert_eq!(squad.leader, Some(MemberId(1)));
    }

    #[test]
    fn test_formation_positions() {
        let formation = SquadFormation::new(SquadFormationShape::Wedge);
        let positions = formation.calculate_positions(5);
        assert_eq!(positions.len(), 5);
    }

    #[test]
    fn test_health_status() {
        let mut squad = Squad::new(SquadId(1), "Alpha");
        squad.add_member(MemberState::new(MemberId(1), SquadRole::Leader, Vec3::ZERO));
        squad.add_member(MemberState::new(MemberId(2), SquadRole::Rifleman, Vec3::ZERO));

        let status = squad.health_status();
        assert_eq!(status.alive_members, 2);
        assert!(status.combat_effective);
    }

    #[test]
    fn test_rally_point() {
        let mut squad = Squad::new(SquadId(1), "Alpha");
        squad.add_rally_point(Vec3::new(10.0, 0.0, 0.0), "RP-1");
        squad.add_rally_point(Vec3::new(20.0, 0.0, 0.0), "RP-2");

        let nearest = squad.nearest_rally_point(Vec3::new(12.0, 0.0, 0.0));
        assert!(nearest.is_some());
    }

    #[test]
    fn test_squad_ai_system() {
        let mut system = SquadAISystem::new();
        let id = system.create_squad("Alpha");
        assert_eq!(system.squad_count(), 1);

        system.issue_order(id, SquadOrder::Hold);
        system.update(0.016);
    }
}
