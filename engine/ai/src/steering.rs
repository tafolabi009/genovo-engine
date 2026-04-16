//! Steering behaviors for autonomous agent movement.
//!
//! Implements Craig Reynolds' steering behaviors for real-time agent navigation.
//! Each behavior produces a steering force vector that, when applied to an agent,
//! produces natural-looking movement patterns.
//!
//! Supported behaviors:
//! - **Seek/Flee**: Move toward/away from a target
//! - **Arrive**: Seek with deceleration near the target
//! - **Pursue/Evade**: Intercept or evade a moving target
//! - **Wander**: Smooth random wandering
//! - **Obstacle/Wall Avoidance**: Steer around obstacles
//! - **Flocking**: Separation + Alignment + Cohesion
//! - **Path Following**: Follow a sequence of waypoints
//! - **Leader Following**: Follow behind a leader agent
//! - **Hide**: Find concealment behind obstacles

use glam::Vec3;

// ---------------------------------------------------------------------------
// SteeringAgent
// ---------------------------------------------------------------------------

/// Represents a steerable agent with position, velocity, and physical parameters.
#[derive(Debug, Clone)]
pub struct SteeringAgent {
    /// Current world-space position.
    pub position: Vec3,
    /// Current velocity vector.
    pub velocity: Vec3,
    /// Maximum speed (units per second).
    pub max_speed: f32,
    /// Maximum steering force magnitude.
    pub max_force: f32,
    /// Agent mass (affects force application).
    pub mass: f32,
    /// Agent orientation angle (radians, around Y axis).
    pub orientation: f32,
    /// Agent collision radius.
    pub radius: f32,
}

impl SteeringAgent {
    /// Creates a new steering agent with default values.
    pub fn new(position: Vec3, max_speed: f32, max_force: f32) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            max_speed,
            max_force,
            mass: 1.0,
            orientation: 0.0,
            radius: 0.5,
        }
    }

    /// Sets the mass.
    pub fn with_mass(mut self, mass: f32) -> Self {
        self.mass = mass;
        self
    }

    /// Sets the collision radius.
    pub fn with_radius(mut self, radius: f32) -> Self {
        self.radius = radius;
        self
    }

    /// Returns the current speed (magnitude of velocity).
    pub fn speed(&self) -> f32 {
        self.velocity.length()
    }

    /// Returns the normalized heading (direction of velocity), or forward.
    pub fn heading(&self) -> Vec3 {
        let s = self.velocity.length();
        if s > 1e-6 {
            self.velocity / s
        } else {
            Vec3::new(self.orientation.sin(), 0.0, self.orientation.cos())
        }
    }

    /// Returns the side vector (perpendicular to heading, on XZ plane).
    pub fn side(&self) -> Vec3 {
        let h = self.heading();
        Vec3::new(-h.z, 0.0, h.x)
    }

    /// Apply a steering force and update position for one time step.
    pub fn apply_force(&mut self, force: Vec3, dt: f32) {
        // Truncate force to max_force.
        let truncated_force = truncate(force, self.max_force);

        // F = ma => a = F / m
        let acceleration = truncated_force / self.mass;

        // Update velocity.
        self.velocity += acceleration * dt;
        self.velocity = truncate(self.velocity, self.max_speed);

        // Update position.
        self.position += self.velocity * dt;

        // Update orientation.
        if self.velocity.length_squared() > 1e-6 {
            self.orientation = self.velocity.x.atan2(self.velocity.z);
        }
    }

    /// Predict this agent's position in the future.
    pub fn predict_position(&self, time: f32) -> Vec3 {
        self.position + self.velocity * time
    }
}

// ---------------------------------------------------------------------------
// Obstacle / Wall types
// ---------------------------------------------------------------------------

/// A circular obstacle in the world.
#[derive(Debug, Clone)]
pub struct Obstacle {
    /// Center position.
    pub position: Vec3,
    /// Radius.
    pub radius: f32,
}

/// A wall segment defined by two endpoints.
#[derive(Debug, Clone)]
pub struct Wall {
    /// Start point.
    pub start: Vec3,
    /// End point.
    pub end: Vec3,
    /// Normal vector (pointing into the navigable space).
    pub normal: Vec3,
}

impl Wall {
    /// Creates a wall between two points, computing the normal.
    pub fn new(start: Vec3, end: Vec3) -> Self {
        let dir = (end - start).normalize_or_zero();
        // Normal is perpendicular on XZ plane (pointing left of the direction).
        let normal = Vec3::new(-dir.z, 0.0, dir.x);
        Self { start, end, normal }
    }

    /// Creates a wall with a specific normal.
    pub fn with_normal(start: Vec3, end: Vec3, normal: Vec3) -> Self {
        Self { start, end, normal }
    }
}

/// A waypoint path.
#[derive(Debug, Clone)]
pub struct WaypointPath {
    /// Ordered waypoints.
    pub points: Vec<Vec3>,
    /// Whether the path loops.
    pub looped: bool,
}

impl WaypointPath {
    /// Creates a new path.
    pub fn new(points: Vec<Vec3>) -> Self {
        Self {
            points,
            looped: false,
        }
    }

    /// Makes this path loop back to the start.
    pub fn with_loop(mut self, looped: bool) -> Self {
        self.looped = looped;
        self
    }

    /// Returns the total length of the path.
    pub fn total_length(&self) -> f32 {
        if self.points.len() < 2 {
            return 0.0;
        }
        let mut length = 0.0f32;
        for i in 0..self.points.len() - 1 {
            length += (self.points[i + 1] - self.points[i]).length();
        }
        if self.looped && self.points.len() >= 2 {
            length += (self.points[0] - self.points[self.points.len() - 1]).length();
        }
        length
    }

    /// Find the nearest point on the path to the given position.
    /// Returns (nearest_point, segment_index, distance_along_segment).
    pub fn nearest_point(&self, position: Vec3) -> (Vec3, usize, f32) {
        if self.points.is_empty() {
            return (position, 0, 0.0);
        }
        if self.points.len() == 1 {
            return (self.points[0], 0, 0.0);
        }

        let mut best_point = self.points[0];
        let mut best_dist_sq = f32::INFINITY;
        let mut best_segment = 0;
        let mut best_t = 0.0;

        let segment_count = if self.looped {
            self.points.len()
        } else {
            self.points.len() - 1
        };

        for i in 0..segment_count {
            let a = self.points[i];
            let b = self.points[(i + 1) % self.points.len()];

            let ab = b - a;
            let len_sq = ab.length_squared();
            let t = if len_sq < 1e-8 {
                0.0
            } else {
                ((position - a).dot(ab) / len_sq).clamp(0.0, 1.0)
            };

            let closest = a + ab * t;
            let dist_sq = (closest - position).length_squared();

            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_point = closest;
                best_segment = i;
                best_t = t;
            }
        }

        (best_point, best_segment, best_t)
    }

    /// Get a point ahead on the path from a given position.
    pub fn point_ahead(&self, position: Vec3, distance: f32) -> Vec3 {
        let (_, segment, t) = self.nearest_point(position);
        let mut remaining = distance;
        let mut seg = segment;

        // Start from current position on the segment.
        let a = self.points[seg];
        let b = self.points[(seg + 1) % self.points.len()];
        let seg_length = (b - a).length();
        let pos_on_seg = t * seg_length;
        remaining -= (seg_length - pos_on_seg).max(0.0);

        if remaining <= 0.0 {
            // The target is on the current segment.
            let target_t = t + distance / seg_length.max(1e-6);
            return a + (b - a) * target_t.clamp(0.0, 1.0);
        }

        // Walk forward through segments.
        let max_segments = if self.looped {
            self.points.len()
        } else {
            self.points.len() - 1
        };

        for _ in 0..max_segments {
            seg = (seg + 1) % max_segments;
            if !self.looped && seg >= self.points.len() - 1 {
                return *self.points.last().unwrap();
            }

            let sa = self.points[seg];
            let sb = self.points[(seg + 1) % self.points.len()];
            let sl = (sb - sa).length();

            if remaining <= sl {
                let frac = remaining / sl.max(1e-6);
                return sa + (sb - sa) * frac;
            }
            remaining -= sl;
        }

        // Past the end of the path.
        if self.looped {
            self.points[0]
        } else {
            *self.points.last().unwrap_or(&Vec3::ZERO)
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Truncate a vector to a maximum length.
fn truncate(v: Vec3, max_length: f32) -> Vec3 {
    let len = v.length();
    if len > max_length && len > 1e-8 {
        v * (max_length / len)
    } else {
        v
    }
}

/// Set the Y component to zero (project onto XZ plane).
fn flatten(v: Vec3) -> Vec3 {
    Vec3::new(v.x, 0.0, v.z)
}

// ---------------------------------------------------------------------------
// Individual Steering Behaviors
// ---------------------------------------------------------------------------

/// Seek: steer toward a target position at maximum speed.
pub fn seek(agent: &SteeringAgent, target: Vec3) -> Vec3 {
    let desired = (target - agent.position).normalize_or_zero() * agent.max_speed;
    desired - agent.velocity
}

/// Flee: steer away from a target position at maximum speed.
pub fn flee(agent: &SteeringAgent, target: Vec3) -> Vec3 {
    let desired = (agent.position - target).normalize_or_zero() * agent.max_speed;
    desired - agent.velocity
}

/// Arrive: seek with deceleration near the target.
///
/// When the agent is within `slow_radius`, it decelerates to arrive smoothly
/// at the target position.
pub fn arrive(agent: &SteeringAgent, target: Vec3, slow_radius: f32) -> Vec3 {
    let to_target = target - agent.position;
    let distance = to_target.length();

    if distance < 1e-4 {
        return -agent.velocity;
    }

    let speed = if distance < slow_radius {
        agent.max_speed * (distance / slow_radius)
    } else {
        agent.max_speed
    };

    let desired = (to_target / distance) * speed;
    desired - agent.velocity
}

/// Pursue: predict a moving target's future position and seek it.
pub fn pursue(agent: &SteeringAgent, target: &SteeringAgent) -> Vec3 {
    let to_target = target.position - agent.position;
    let distance = to_target.length();

    // Estimate the time to reach the target.
    let prediction_time = distance / (agent.max_speed + target.speed()).max(1e-4);

    // Seek the predicted position.
    let predicted_pos = target.predict_position(prediction_time);
    seek(agent, predicted_pos)
}

/// Evade: predict a moving target's future position and flee from it.
pub fn evade(agent: &SteeringAgent, target: &SteeringAgent) -> Vec3 {
    let to_target = target.position - agent.position;
    let distance = to_target.length();

    let prediction_time = distance / (agent.max_speed + target.speed()).max(1e-4);
    let predicted_pos = target.predict_position(prediction_time);
    flee(agent, predicted_pos)
}

/// Wander: produce smooth random steering using a projected circle.
///
/// A circle is projected in front of the agent, and a target point on the
/// circle is perturbed slightly each frame, producing smooth wandering.
///
/// # Arguments
/// - `circle_radius`: Radius of the wander circle.
/// - `circle_distance`: Distance of the circle center from the agent.
/// - `jitter`: Amount of random displacement per frame.
/// - `wander_angle`: Mutable reference to the current wander angle (state).
pub fn wander(
    agent: &SteeringAgent,
    circle_radius: f32,
    circle_distance: f32,
    jitter: f32,
    wander_angle: &mut f32,
) -> Vec3 {
    // Use a simple deterministic approach based on agent position.
    let noise = {
        let x = agent.position.x * 13.37 + agent.position.z * 7.13;
        let s = (x.sin() * 43758.5453).fract();
        s * 2.0 - 1.0
    };

    *wander_angle += noise * jitter;

    // Point on the wander circle.
    let circle_center = agent.heading() * circle_distance;
    let displacement = Vec3::new(
        circle_radius * wander_angle.cos(),
        0.0,
        circle_radius * wander_angle.sin(),
    );

    let wander_target = agent.position + circle_center + displacement;
    seek(agent, wander_target)
}

/// Obstacle avoidance: steer around circular obstacles.
///
/// Casts a detection box ahead of the agent and steers away from the
/// closest intersecting obstacle.
pub fn obstacle_avoidance(
    agent: &SteeringAgent,
    obstacles: &[Obstacle],
    look_ahead: f32,
) -> Vec3 {
    let heading = agent.heading();
    let side = agent.side();

    // Detection box length scales with speed.
    let box_length = look_ahead * (agent.speed() / agent.max_speed).max(0.3);

    let mut closest_obstacle: Option<(usize, f32)> = None;

    for (i, obstacle) in obstacles.iter().enumerate() {
        // Transform obstacle position to agent's local space.
        let to_obstacle = obstacle.position - agent.position;
        let local_x = to_obstacle.dot(side);
        let local_z = to_obstacle.dot(heading);

        // Skip if behind the agent.
        if local_z < 0.0 {
            continue;
        }

        // Skip if beyond the detection box.
        if local_z > box_length + obstacle.radius {
            continue;
        }

        // Check lateral overlap.
        let expanded_radius = obstacle.radius + agent.radius;
        if local_x.abs() > expanded_radius {
            continue;
        }

        // Compute the closest intersection distance.
        let sqrt_part = (expanded_radius * expanded_radius - local_x * local_x).max(0.0).sqrt();
        let intersection_z = local_z - sqrt_part;

        match closest_obstacle {
            Some((_, closest_z)) if intersection_z < closest_z => {
                closest_obstacle = Some((i, intersection_z));
            }
            None => {
                closest_obstacle = Some((i, intersection_z));
            }
            _ => {}
        }
    }

    if let Some((obs_idx, _)) = closest_obstacle {
        let obstacle = &obstacles[obs_idx];
        let to_obstacle = obstacle.position - agent.position;
        let local_x = to_obstacle.dot(side);
        let local_z = to_obstacle.dot(heading);

        // Steer laterally away from the obstacle.
        let multiplier = 1.0 + (box_length - local_z) / box_length;
        let lateral_force = -local_x.signum() * multiplier * agent.max_force;

        // Also brake slightly to avoid rear collisions.
        let braking_weight = 0.2;
        let braking_force = (obstacle.radius - local_z) * braking_weight;

        side * lateral_force + heading * braking_force
    } else {
        Vec3::ZERO
    }
}

/// Wall avoidance: use feeler rays to detect and avoid walls.
///
/// Three feelers (forward, 45 degrees left, 45 degrees right) are cast from
/// the agent. When a feeler intersects a wall, the agent steers away.
pub fn wall_avoidance(
    agent: &SteeringAgent,
    walls: &[Wall],
    whisker_length: f32,
) -> Vec3 {
    let heading = agent.heading();
    let side_vec = agent.side();

    // Create three feeler rays.
    let feelers = [
        // Forward.
        agent.position + heading * whisker_length,
        // 45 degrees left.
        agent.position + (heading + side_vec * 0.5).normalize_or_zero() * whisker_length * 0.7,
        // 45 degrees right.
        agent.position + (heading - side_vec * 0.5).normalize_or_zero() * whisker_length * 0.7,
    ];

    let mut steering = Vec3::ZERO;
    let mut closest_dist = f32::INFINITY;

    for feeler in &feelers {
        for wall in walls {
            if let Some((intersection, dist)) =
                line_segment_intersection(agent.position, *feeler, wall.start, wall.end)
            {
                if dist < closest_dist {
                    closest_dist = dist;
                    // Steer in the direction of the wall normal.
                    let overshoot = *feeler - intersection;
                    steering = wall.normal * overshoot.length();
                }
            }
        }
    }

    steering
}

/// Separation: maintain distance from nearby agents.
pub fn separation(
    agent: &SteeringAgent,
    neighbors: &[&SteeringAgent],
    radius: f32,
) -> Vec3 {
    let mut force = Vec3::ZERO;

    for neighbor in neighbors {
        let to_agent = agent.position - neighbor.position;
        let distance = to_agent.length();

        if distance < 1e-6 || distance > radius {
            continue;
        }

        // Force is inversely proportional to distance.
        let strength = (radius - distance) / radius;
        force += (to_agent / distance) * strength;
    }

    force
}

/// Alignment: match heading of nearby agents.
pub fn alignment(
    agent: &SteeringAgent,
    neighbors: &[&SteeringAgent],
    radius: f32,
) -> Vec3 {
    let mut avg_heading = Vec3::ZERO;
    let mut count = 0;

    for neighbor in neighbors {
        let distance = (neighbor.position - agent.position).length();
        if distance < 1e-6 || distance > radius {
            continue;
        }

        avg_heading += neighbor.heading();
        count += 1;
    }

    if count > 0 {
        avg_heading /= count as f32;
        avg_heading -= agent.heading();
    }

    avg_heading
}

/// Cohesion: move toward the center of mass of nearby agents.
pub fn cohesion(
    agent: &SteeringAgent,
    neighbors: &[&SteeringAgent],
    radius: f32,
) -> Vec3 {
    let mut center = Vec3::ZERO;
    let mut count = 0;

    for neighbor in neighbors {
        let distance = (neighbor.position - agent.position).length();
        if distance < 1e-6 || distance > radius {
            continue;
        }

        center += neighbor.position;
        count += 1;
    }

    if count > 0 {
        center /= count as f32;
        seek(agent, center)
    } else {
        Vec3::ZERO
    }
}

/// Flocking: combines separation, alignment, and cohesion.
pub fn flocking(
    agent: &SteeringAgent,
    neighbors: &[&SteeringAgent],
    separation_radius: f32,
    alignment_radius: f32,
    cohesion_radius: f32,
    separation_weight: f32,
    alignment_weight: f32,
    cohesion_weight: f32,
) -> Vec3 {
    let sep = separation(agent, neighbors, separation_radius) * separation_weight;
    let ali = alignment(agent, neighbors, alignment_radius) * alignment_weight;
    let coh = cohesion(agent, neighbors, cohesion_radius) * cohesion_weight;

    sep + ali + coh
}

/// Path following: follow a waypoint path.
///
/// Projects the agent's predicted position onto the path and seeks a point
/// slightly ahead on the path.
pub fn path_following(
    agent: &SteeringAgent,
    path: &WaypointPath,
    predict_time: f32,
) -> Vec3 {
    if path.points.is_empty() {
        return Vec3::ZERO;
    }

    // Predict future position.
    let future_pos = agent.predict_position(predict_time);

    // Find the nearest point on the path.
    let (nearest, _segment, _t) = path.nearest_point(future_pos);

    // If we're close enough to the path, just follow ahead.
    let distance_to_path = (future_pos - nearest).length();
    let path_ahead_dist = agent.max_speed * predict_time;

    // Seek a point ahead on the path.
    let target = path.point_ahead(agent.position, path_ahead_dist);

    if distance_to_path > agent.radius * 2.0 {
        // We're off the path: seek the nearest point first.
        seek(agent, nearest)
    } else {
        // On the path: seek ahead.
        seek(agent, target)
    }
}

/// Leader following: follow behind a leader agent with an offset.
pub fn leader_following(
    agent: &SteeringAgent,
    leader: &SteeringAgent,
    behind_distance: f32,
    slow_radius: f32,
) -> Vec3 {
    // Compute the position behind the leader.
    let leader_heading = leader.heading();
    let behind_pos = leader.position - leader_heading * behind_distance;

    // If we're in the leader's path (in front of them), evade.
    let to_leader = leader.position - agent.position;
    let dot = to_leader.normalize_or_zero().dot(leader_heading);

    if dot > 0.7 && to_leader.length() < behind_distance {
        // We're in front of the leader; evade.
        let evade_force = evade(agent, leader);
        let arrive_force = arrive(agent, behind_pos, slow_radius);
        evade_force + arrive_force
    } else {
        arrive(agent, behind_pos, slow_radius)
    }
}

/// Hide: find a hiding spot behind an obstacle from a target.
///
/// For each obstacle, computes a hiding position on the far side from the
/// target, then arrives at the best (closest) one.
pub fn hide(
    agent: &SteeringAgent,
    target: Vec3,
    obstacles: &[Obstacle],
) -> Vec3 {
    if obstacles.is_empty() {
        return flee(agent, target);
    }

    let mut best_hiding_spot = Vec3::ZERO;
    let mut best_dist_sq = f32::INFINITY;

    for obstacle in obstacles {
        // Compute the hiding spot behind this obstacle from the target.
        let to_obstacle = (obstacle.position - target).normalize_or_zero();
        let hide_distance = obstacle.radius + agent.radius * 2.0;
        let hiding_spot = obstacle.position + to_obstacle * hide_distance;

        let dist_sq = (hiding_spot - agent.position).length_squared();
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best_hiding_spot = hiding_spot;
        }
    }

    arrive(agent, best_hiding_spot, 2.0)
}

// ---------------------------------------------------------------------------
// SteeringCombinator
// ---------------------------------------------------------------------------

/// Method for combining multiple steering forces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombineMethod {
    /// Weighted sum of all forces (all contribute proportionally).
    WeightedSum,
    /// Priority-based: first non-zero force wins.
    Priority,
    /// Dithered: randomly select one behavior per frame.
    Dithered,
}

/// A named, weighted steering behavior entry.
pub struct SteeringEntry {
    /// Name for debugging.
    pub name: String,
    /// Weight (for weighted sum) or priority (for priority-based).
    pub weight: f32,
    /// The computed force.
    pub force: Vec3,
}

/// Combines multiple steering forces into a single force vector.
pub struct SteeringCombinator {
    /// The entries (behaviors with computed forces).
    entries: Vec<SteeringEntry>,
    /// The combination method.
    method: CombineMethod,
    /// Max total force.
    max_force: f32,
    /// Counter for dithered selection.
    dither_counter: u32,
}

impl SteeringCombinator {
    /// Creates a new combinator.
    pub fn new(method: CombineMethod, max_force: f32) -> Self {
        Self {
            entries: Vec::new(),
            method,
            max_force,
            dither_counter: 0,
        }
    }

    /// Add a steering force entry.
    pub fn add(&mut self, name: impl Into<String>, weight: f32, force: Vec3) {
        self.entries.push(SteeringEntry {
            name: name.into(),
            weight,
            force,
        });
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Calculate the combined steering force.
    pub fn calculate(&mut self) -> Vec3 {
        match self.method {
            CombineMethod::WeightedSum => {
                let mut total = Vec3::ZERO;
                for entry in &self.entries {
                    total += entry.force * entry.weight;
                }
                truncate(total, self.max_force)
            }
            CombineMethod::Priority => {
                // Sort by weight descending (higher = higher priority).
                self.entries
                    .sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));

                let mut remaining_force = self.max_force;
                let mut total = Vec3::ZERO;

                for entry in &self.entries {
                    let force_mag = entry.force.length();
                    if force_mag < 1e-6 {
                        continue;
                    }

                    if force_mag <= remaining_force {
                        total += entry.force;
                        remaining_force -= force_mag;
                    } else {
                        // Partially add this force.
                        total += entry.force.normalize_or_zero() * remaining_force;
                        break;
                    }
                }

                total
            }
            CombineMethod::Dithered => {
                // Deterministic selection based on counter.
                self.dither_counter = self
                    .dither_counter
                    .wrapping_mul(1664525)
                    .wrapping_add(1013904223);

                let non_zero: Vec<&SteeringEntry> =
                    self.entries.iter().filter(|e| e.force.length() > 1e-6).collect();

                if non_zero.is_empty() {
                    return Vec3::ZERO;
                }

                // Weighted random selection.
                let total_weight: f32 = non_zero.iter().map(|e| e.weight).sum();
                let r = (self.dither_counter as f32 / u32::MAX as f32) * total_weight;
                let mut accumulated = 0.0;

                for entry in &non_zero {
                    accumulated += entry.weight;
                    if accumulated >= r {
                        return truncate(entry.force, self.max_force);
                    }
                }

                truncate(non_zero.last().unwrap().force, self.max_force)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Test if two line segments intersect. Returns intersection point and
/// distance along the first segment.
fn line_segment_intersection(
    a1: Vec3,
    a2: Vec3,
    b1: Vec3,
    b2: Vec3,
) -> Option<(Vec3, f32)> {
    let d1 = a2 - a1;
    let d2 = b2 - b1;

    let denom = d1.x * d2.z - d1.z * d2.x;
    if denom.abs() < 1e-8 {
        return None;
    }

    let t = ((b1.x - a1.x) * d2.z - (b1.z - a1.z) * d2.x) / denom;
    let u = ((b1.x - a1.x) * d1.z - (b1.z - a1.z) * d1.x) / denom;

    if t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 {
        let point = a1 + d1 * t;
        let dist = t * d1.length();
        Some((point, dist))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_agent(x: f32, z: f32) -> SteeringAgent {
        SteeringAgent::new(Vec3::new(x, 0.0, z), 5.0, 10.0)
    }

    fn make_agent_with_vel(x: f32, z: f32, vx: f32, vz: f32) -> SteeringAgent {
        let mut a = make_agent(x, z);
        a.velocity = Vec3::new(vx, 0.0, vz);
        a
    }

    #[test]
    fn test_steering_agent_creation() {
        let agent = make_agent(0.0, 0.0);
        assert_eq!(agent.position, Vec3::ZERO);
        assert_eq!(agent.max_speed, 5.0);
        assert_eq!(agent.max_force, 10.0);
        assert_eq!(agent.speed(), 0.0);
    }

    #[test]
    fn test_steering_agent_apply_force() {
        let mut agent = make_agent(0.0, 0.0);
        agent.apply_force(Vec3::new(1.0, 0.0, 0.0), 1.0);

        assert!(agent.velocity.x > 0.0);
        assert!(agent.position.x > 0.0);
    }

    #[test]
    fn test_steering_agent_max_speed() {
        let mut agent = make_agent(0.0, 0.0);
        // Apply a huge force repeatedly.
        for _ in 0..100 {
            agent.apply_force(Vec3::new(100.0, 0.0, 0.0), 0.1);
        }
        // Speed should be clamped to max_speed.
        assert!(agent.speed() <= agent.max_speed + 0.01);
    }

    #[test]
    fn test_seek() {
        let agent = make_agent(0.0, 0.0);
        let target = Vec3::new(10.0, 0.0, 0.0);
        let force = seek(&agent, target);

        // Force should point toward the target.
        assert!(force.x > 0.0);
    }

    #[test]
    fn test_flee() {
        let agent = make_agent(0.0, 0.0);
        let target = Vec3::new(10.0, 0.0, 0.0);
        let force = flee(&agent, target);

        // Force should point away from the target.
        assert!(force.x < 0.0);
    }

    #[test]
    fn test_arrive() {
        let agent = make_agent(0.0, 0.0);
        let target = Vec3::new(1.0, 0.0, 0.0);

        // Close to target: force should be smaller.
        let force_close = arrive(&agent, target, 5.0);

        let agent_far = make_agent(-20.0, 0.0);
        let force_far = arrive(&agent_far, target, 5.0);

        // Far force should be larger than close force.
        assert!(force_far.length() > force_close.length());
    }

    #[test]
    fn test_pursue() {
        let agent = make_agent(0.0, 0.0);
        let target = make_agent_with_vel(10.0, 0.0, 1.0, 0.0);

        let force = pursue(&agent, &target);
        // Should steer ahead of the target.
        assert!(force.x > 0.0);
    }

    #[test]
    fn test_evade() {
        let agent = make_agent(0.0, 0.0);
        let target = make_agent_with_vel(5.0, 0.0, -1.0, 0.0);

        let force = evade(&agent, &target);
        // Should steer away.
        assert!(force.x < 0.0);
    }

    #[test]
    fn test_wander() {
        let agent = make_agent_with_vel(0.0, 0.0, 0.0, 1.0);
        let mut angle = 0.0;
        let force = wander(&agent, 1.0, 2.0, 0.3, &mut angle);

        // Wander should produce a non-zero force.
        assert!(force.length() > 0.0);
    }

    #[test]
    fn test_separation() {
        let agent = make_agent(5.0, 5.0);
        let neighbor1 = make_agent(5.5, 5.0);
        let neighbor2 = make_agent(5.0, 5.5);
        let neighbors: Vec<&SteeringAgent> = vec![&neighbor1, &neighbor2];

        let force = separation(&agent, &neighbors, 3.0);
        // Should push away from neighbors.
        assert!(force.x < 0.0 || force.z < 0.0);
    }

    #[test]
    fn test_alignment() {
        let agent = make_agent_with_vel(0.0, 0.0, 1.0, 0.0);
        let neighbor = make_agent_with_vel(2.0, 0.0, 0.0, 1.0);
        let neighbors: Vec<&SteeringAgent> = vec![&neighbor];

        let force = alignment(&agent, &neighbors, 5.0);
        // Should try to align with the neighbor's heading.
        assert!(force.length() > 0.0);
    }

    #[test]
    fn test_cohesion() {
        let agent = make_agent(0.0, 0.0);
        let n1 = make_agent(3.0, 0.0);
        let n2 = make_agent(0.0, 3.0);
        let neighbors: Vec<&SteeringAgent> = vec![&n1, &n2];

        let force = cohesion(&agent, &neighbors, 5.0);
        // Should steer toward center of neighbors.
        assert!(force.x > 0.0);
        assert!(force.z > 0.0);
    }

    #[test]
    fn test_flocking() {
        let agent = make_agent_with_vel(5.0, 5.0, 1.0, 0.0);
        let n1 = make_agent_with_vel(5.5, 5.0, 0.8, 0.2);
        let n2 = make_agent_with_vel(4.5, 5.0, 1.2, -0.1);
        let neighbors: Vec<&SteeringAgent> = vec![&n1, &n2];

        let force = flocking(&agent, &neighbors, 2.0, 5.0, 5.0, 1.5, 1.0, 1.0);
        // Should produce a combined force.
        assert!(force.length() > 0.0);
    }

    #[test]
    fn test_obstacle_avoidance() {
        let agent = make_agent_with_vel(0.0, 0.0, 0.0, 5.0);
        let obstacles = vec![Obstacle {
            position: Vec3::new(0.0, 0.0, 5.0),
            radius: 1.0,
        }];

        let force = obstacle_avoidance(&agent, &obstacles, 10.0);
        // Should steer laterally to avoid the obstacle.
        assert!(force.length() > 0.0);
    }

    #[test]
    fn test_obstacle_avoidance_no_obstacles() {
        let agent = make_agent_with_vel(0.0, 0.0, 0.0, 5.0);
        let force = obstacle_avoidance(&agent, &[], 10.0);
        assert_eq!(force, Vec3::ZERO);
    }

    #[test]
    fn test_wall_avoidance() {
        let agent = make_agent_with_vel(0.0, 0.0, 0.0, 5.0);
        let walls = vec![Wall::new(
            Vec3::new(-5.0, 0.0, 3.0),
            Vec3::new(5.0, 0.0, 3.0),
        )];

        let force = wall_avoidance(&agent, &walls, 5.0);
        // Should steer away from the wall ahead.
        assert!(force.length() > 0.0);
    }

    #[test]
    fn test_path_following() {
        let agent = make_agent_with_vel(0.0, 0.0, 0.0, 1.0);
        let path = WaypointPath::new(vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 10.0),
        ]);

        let force = path_following(&agent, &path, 0.5);
        // Should steer toward the path.
        assert!(force.length() > 0.0);
    }

    #[test]
    fn test_leader_following() {
        let agent = make_agent(0.0, 0.0);
        let leader = make_agent_with_vel(5.0, 0.0, 1.0, 0.0);

        let force = leader_following(&agent, &leader, 3.0, 5.0);
        // Should steer toward a position behind the leader.
        assert!(force.length() > 0.0);
    }

    #[test]
    fn test_hide() {
        let agent = make_agent(0.0, 0.0);
        let target = Vec3::new(10.0, 0.0, 0.0);
        let obstacles = vec![Obstacle {
            position: Vec3::new(5.0, 0.0, 3.0),
            radius: 1.0,
        }];

        let force = hide(&agent, target, &obstacles);
        // Should steer toward a hiding spot.
        assert!(force.length() > 0.0);
    }

    #[test]
    fn test_hide_no_obstacles() {
        let agent = make_agent(0.0, 0.0);
        let target = Vec3::new(10.0, 0.0, 0.0);

        let force = hide(&agent, target, &[]);
        // Should flee when no obstacles to hide behind.
        assert!(force.x < 0.0);
    }

    #[test]
    fn test_combinator_weighted_sum() {
        let mut combinator = SteeringCombinator::new(CombineMethod::WeightedSum, 20.0);
        combinator.add("seek", 1.0, Vec3::new(5.0, 0.0, 0.0));
        combinator.add("avoid", 2.0, Vec3::new(0.0, 0.0, 3.0));

        let force = combinator.calculate();
        // 5*1 + 0*2 = 5 on X, 0*1 + 3*2 = 6 on Z.
        assert!((force.x - 5.0).abs() < 0.01);
        assert!((force.z - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_combinator_priority() {
        let mut combinator = SteeringCombinator::new(CombineMethod::Priority, 10.0);
        combinator.add("high_priority", 10.0, Vec3::new(5.0, 0.0, 0.0));
        combinator.add("low_priority", 1.0, Vec3::new(0.0, 0.0, 100.0));

        let force = combinator.calculate();
        // High priority force should dominate.
        assert!(force.x > 0.0);
    }

    #[test]
    fn test_waypoint_path() {
        let path = WaypointPath::new(vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 10.0),
        ]);

        assert!((path.total_length() - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_waypoint_path_nearest() {
        let path = WaypointPath::new(vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
        ]);

        let (nearest, segment, _t) = path.nearest_point(Vec3::new(5.0, 0.0, 3.0));
        assert_eq!(segment, 0);
        assert!((nearest.x - 5.0).abs() < 0.01);
        assert!((nearest.z - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_waypoint_path_ahead() {
        let path = WaypointPath::new(vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 10.0),
        ]);

        let ahead = path.point_ahead(Vec3::new(0.0, 0.0, 0.0), 5.0);
        assert!((ahead.x - 5.0).abs() < 0.5);
    }

    #[test]
    fn test_wall_creation() {
        let wall = Wall::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
        );
        // Wall along X axis: normal should point in Z direction.
        assert!(wall.normal.z.abs() > 0.9);
    }

    #[test]
    fn test_line_segment_intersection() {
        let result = line_segment_intersection(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, -5.0),
            Vec3::new(5.0, 0.0, 5.0),
        );
        assert!(result.is_some());
        let (point, _dist) = result.unwrap();
        assert!((point.x - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_line_segment_no_intersection() {
        let result = line_segment_intersection(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 5.0),
            Vec3::new(5.0, 0.0, 10.0),
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_truncate() {
        let v = Vec3::new(10.0, 0.0, 0.0);
        let t = truncate(v, 5.0);
        assert!((t.length() - 5.0).abs() < 0.01);

        let short = Vec3::new(2.0, 0.0, 0.0);
        let t2 = truncate(short, 5.0);
        assert!((t2.x - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_predict_position() {
        let agent = make_agent_with_vel(0.0, 0.0, 5.0, 0.0);
        let predicted = agent.predict_position(2.0);
        assert!((predicted.x - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_looped_path() {
        let path = WaypointPath::new(vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 10.0),
            Vec3::new(0.0, 0.0, 10.0),
        ])
        .with_loop(true);

        // Total length should include closing edge.
        assert!((path.total_length() - 40.0).abs() < 0.01);
    }
}
