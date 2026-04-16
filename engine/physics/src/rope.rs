//! Rope and chain physics simulation using Verlet integration.
//!
//! Provides:
//! - `RopeSimulation`: particle-based rope with distance constraints (Jakobsen),
//!   Verlet integration, wind force, self-collision, and Catmull-Rom rendering data
//! - `ChainSimulation`: rigid-link chain with capsule shapes and hinge joints
//! - `RopeComponent`, `RopeSystem` for ECS integration
//! - Factory functions: `create_rope`, endpoint pinning/attachment

use std::collections::HashMap;

use glam::{Quat, Vec3};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default number of constraint solver iterations per step.
const DEFAULT_SOLVER_ITERATIONS: usize = 12;
/// Default damping factor (0 = no damping, 1 = full).
const DEFAULT_DAMPING: f32 = 0.01;
/// Default gravity.
const DEFAULT_GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);
/// Self-collision spatial hash cell size.
const SELF_COLLISION_CELL_SIZE: f32 = 0.3;
/// Default particle radius for self-collision.
const DEFAULT_PARTICLE_RADIUS: f32 = 0.02;

// ---------------------------------------------------------------------------
// Rope endpoint attachment
// ---------------------------------------------------------------------------

/// How a rope endpoint is attached.
#[derive(Debug, Clone)]
pub enum RopeAttachment {
    /// Pinned to a fixed world position.
    WorldPin(Vec3),
    /// Attached to a rigid body (body handle placeholder, local offset).
    BodyAttach {
        body_id: u64,
        local_offset: Vec3,
        /// World-space position updated each frame from the body's transform.
        world_position: Vec3,
    },
    /// Free (not attached to anything).
    Free,
}

// ---------------------------------------------------------------------------
// Rope particle
// ---------------------------------------------------------------------------

/// A single mass point in the rope.
#[derive(Debug, Clone)]
pub struct RopeParticle {
    /// Current world-space position.
    pub position: Vec3,
    /// Previous position (for Verlet integration).
    pub prev_position: Vec3,
    /// Accumulated external force for the current step.
    pub accumulated_force: Vec3,
    /// Particle mass in kg.
    pub mass: f32,
    /// Inverse mass (0.0 for pinned particles).
    pub inv_mass: f32,
    /// Whether this particle is pinned.
    pub pinned: bool,
    /// Particle radius for self-collision.
    pub radius: f32,
}

impl RopeParticle {
    /// Create a new rope particle at the given position.
    pub fn new(position: Vec3, mass: f32) -> Self {
        let inv_mass = if mass > 1e-8 { 1.0 / mass } else { 0.0 };
        Self {
            position,
            prev_position: position,
            accumulated_force: Vec3::ZERO,
            mass,
            inv_mass,
            pinned: false,
            radius: DEFAULT_PARTICLE_RADIUS,
        }
    }

    /// Pin this particle so it cannot move.
    pub fn pin(&mut self) {
        self.pinned = true;
        self.inv_mass = 0.0;
    }

    /// Unpin this particle.
    pub fn unpin(&mut self) {
        self.pinned = false;
        if self.mass > 1e-8 {
            self.inv_mass = 1.0 / self.mass;
        }
    }

    /// Apply a force (accumulated over the step).
    pub fn apply_force(&mut self, force: Vec3) {
        if !self.pinned {
            self.accumulated_force += force;
        }
    }

    /// Compute current velocity from position delta.
    pub fn velocity(&self) -> Vec3 {
        self.position - self.prev_position
    }
}

// ---------------------------------------------------------------------------
// Rope distance constraint
// ---------------------------------------------------------------------------

/// Distance constraint between two adjacent rope particles.
#[derive(Debug, Clone)]
pub struct RopeConstraint {
    /// Index of particle A.
    pub particle_a: usize,
    /// Index of particle B.
    pub particle_b: usize,
    /// Rest length between the two particles.
    pub rest_length: f32,
    /// Stiffness [0, 1]. Higher = stiffer.
    pub stiffness: f32,
}

impl RopeConstraint {
    pub fn new(a: usize, b: usize, rest_length: f32, stiffness: f32) -> Self {
        Self {
            particle_a: a,
            particle_b: b,
            rest_length,
            stiffness,
        }
    }
}

// ---------------------------------------------------------------------------
// Rope settings
// ---------------------------------------------------------------------------

/// Configuration for rope simulation behaviour.
#[derive(Debug, Clone)]
pub struct RopeSettings {
    /// Number of constraint solver iterations per step.
    pub solver_iterations: usize,
    /// Damping factor for Verlet integration.
    pub damping: f32,
    /// Gravity vector.
    pub gravity: Vec3,
    /// Constraint stiffness [0, 1].
    pub stiffness: f32,
    /// Whether self-collision is enabled.
    pub self_collision_enabled: bool,
    /// Self-collision particle radius.
    pub self_collision_radius: f32,
    /// Simulation time step.
    pub time_step: f32,
}

impl Default for RopeSettings {
    fn default() -> Self {
        Self {
            solver_iterations: DEFAULT_SOLVER_ITERATIONS,
            damping: DEFAULT_DAMPING,
            gravity: DEFAULT_GRAVITY,
            stiffness: 1.0,
            self_collision_enabled: false,
            self_collision_radius: DEFAULT_PARTICLE_RADIUS,
            time_step: 1.0 / 60.0,
        }
    }
}

// ---------------------------------------------------------------------------
// RopeSimulation
// ---------------------------------------------------------------------------

/// A rope simulation using Verlet integration and Jakobsen distance constraints.
///
/// Particles are laid out along the rope, each connected to its neighbor by a
/// distance constraint. The rope acts as a one-dimensional cloth strip.
#[derive(Debug, Clone)]
pub struct RopeSimulation {
    /// All particles along the rope.
    pub particles: Vec<RopeParticle>,
    /// Distance constraints between adjacent particles.
    pub constraints: Vec<RopeConstraint>,
    /// Simulation settings.
    pub settings: RopeSettings,
    /// Start endpoint attachment.
    pub start_attachment: RopeAttachment,
    /// End endpoint attachment.
    pub end_attachment: RopeAttachment,
    /// Current wind direction and strength.
    pub wind: Vec3,
    /// Wind turbulence factor.
    pub wind_turbulence: f32,
    /// Running simulation time (for turbulence noise).
    sim_time: f32,
    /// Total rest length of the rope.
    pub total_rest_length: f32,
}

impl RopeSimulation {
    /// Create a new empty rope simulation.
    pub fn new(settings: RopeSettings) -> Self {
        Self {
            particles: Vec::new(),
            constraints: Vec::new(),
            settings,
            start_attachment: RopeAttachment::Free,
            end_attachment: RopeAttachment::Free,
            wind: Vec3::ZERO,
            wind_turbulence: 0.0,
            sim_time: 0.0,
            total_rest_length: 0.0,
        }
    }

    /// Get the number of particles.
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Get the total length of the rope at rest.
    pub fn rest_length(&self) -> f32 {
        self.total_rest_length
    }

    /// Compute the current stretched length of the rope.
    pub fn current_length(&self) -> f32 {
        let mut length = 0.0;
        for i in 1..self.particles.len() {
            length += (self.particles[i].position - self.particles[i - 1].position).length();
        }
        length
    }

    /// Pin the start particle to its current position.
    pub fn pin_start(&mut self) {
        if !self.particles.is_empty() {
            let pos = self.particles[0].position;
            self.particles[0].pin();
            self.start_attachment = RopeAttachment::WorldPin(pos);
        }
    }

    /// Pin the end particle to its current position.
    pub fn pin_end(&mut self) {
        if let Some(last) = self.particles.last_mut() {
            let pos = last.position;
            last.pin();
            self.end_attachment = RopeAttachment::WorldPin(pos);
        }
    }

    /// Pin the start to a specific world position.
    pub fn pin_start_to(&mut self, position: Vec3) {
        if !self.particles.is_empty() {
            self.particles[0].pin();
            self.particles[0].position = position;
            self.particles[0].prev_position = position;
            self.start_attachment = RopeAttachment::WorldPin(position);
        }
    }

    /// Pin the end to a specific world position.
    pub fn pin_end_to(&mut self, position: Vec3) {
        if let Some(last) = self.particles.last_mut() {
            last.pin();
            last.position = position;
            last.prev_position = position;
            self.end_attachment = RopeAttachment::WorldPin(position);
        }
    }

    /// Attach the start to a rigid body.
    pub fn attach_start_to_body(&mut self, body_id: u64, local_offset: Vec3, world_pos: Vec3) {
        if !self.particles.is_empty() {
            self.particles[0].pin();
            self.particles[0].position = world_pos;
            self.particles[0].prev_position = world_pos;
            self.start_attachment = RopeAttachment::BodyAttach {
                body_id,
                local_offset,
                world_position: world_pos,
            };
        }
    }

    /// Attach the end to a rigid body.
    pub fn attach_end_to_body(&mut self, body_id: u64, local_offset: Vec3, world_pos: Vec3) {
        if let Some(last) = self.particles.last_mut() {
            last.pin();
            last.position = world_pos;
            last.prev_position = world_pos;
            self.end_attachment = RopeAttachment::BodyAttach {
                body_id,
                local_offset,
                world_position: world_pos,
            };
        }
    }

    /// Update body attachment positions (called before step when bodies have moved).
    pub fn update_attachments(&mut self) {
        if let RopeAttachment::WorldPin(pos) = &self.start_attachment {
            if !self.particles.is_empty() {
                self.particles[0].position = *pos;
                self.particles[0].prev_position = *pos;
            }
        }
        if let RopeAttachment::BodyAttach { world_position, .. } = &self.start_attachment {
            if !self.particles.is_empty() {
                self.particles[0].position = *world_position;
                self.particles[0].prev_position = *world_position;
            }
        }

        if let RopeAttachment::WorldPin(pos) = &self.end_attachment {
            if let Some(last) = self.particles.last_mut() {
                last.position = *pos;
                last.prev_position = *pos;
            }
        }
        if let RopeAttachment::BodyAttach { world_position, .. } = &self.end_attachment {
            if let Some(last) = self.particles.last_mut() {
                last.position = *world_position;
                last.prev_position = *world_position;
            }
        }
    }

    /// Set the wind vector and turbulence.
    pub fn set_wind(&mut self, wind: Vec3, turbulence: f32) {
        self.wind = wind;
        self.wind_turbulence = turbulence;
    }

    /// Step the rope simulation forward by one time step.
    ///
    /// Pipeline:
    /// 1. Update attachment positions
    /// 2. Apply external forces (gravity, wind)
    /// 3. Verlet integration
    /// 4. Constraint relaxation (Jakobsen method)
    /// 5. Self-collision (optional)
    /// 6. Pin constraint enforcement
    pub fn step(&mut self, dt: f32) {
        let dt = if dt > 0.0 { dt } else { self.settings.time_step };
        self.sim_time += dt;

        // 1. Update attachments
        self.update_attachments();

        // 2. Apply gravity
        self.apply_gravity();

        // 2b. Apply wind
        self.apply_wind();

        // 3. Verlet integration
        self.verlet_integrate(dt);

        // 4. Constraint relaxation
        for _ in 0..self.settings.solver_iterations {
            self.solve_constraints();
        }

        // 5. Self-collision
        if self.settings.self_collision_enabled {
            self.solve_self_collisions();
        }

        // 6. Pin constraints
        self.enforce_pins();

        // Clear forces
        for p in &mut self.particles {
            p.accumulated_force = Vec3::ZERO;
        }
    }

    // -----------------------------------------------------------------------
    // Force application
    // -----------------------------------------------------------------------

    fn apply_gravity(&mut self) {
        let gravity = self.settings.gravity;
        for p in &mut self.particles {
            if !p.pinned {
                p.apply_force(gravity * p.mass);
            }
        }
    }

    fn apply_wind(&mut self) {
        if self.wind.length_squared() < 1e-8 {
            return;
        }

        let turbulence_offset = if self.wind_turbulence > 0.0 {
            let t = self.sim_time;
            Vec3::new(
                (t * 3.7).sin() * self.wind_turbulence,
                (t * 2.3).cos() * self.wind_turbulence * 0.5,
                (t * 5.1).sin() * self.wind_turbulence,
            )
        } else {
            Vec3::ZERO
        };
        let effective_wind = self.wind + turbulence_offset;

        // Apply wind as a drag-like force on each segment
        for i in 0..self.particles.len().saturating_sub(1) {
            let p0 = self.particles[i].position;
            let p1 = self.particles[i + 1].position;
            let segment = p1 - p0;
            let seg_len = segment.length();
            if seg_len < 1e-8 {
                continue;
            }
            let seg_dir = segment / seg_len;

            // Normal to the segment in the wind plane
            let wind_component = effective_wind - seg_dir * effective_wind.dot(seg_dir);
            let force = wind_component * seg_len * 0.5;

            self.particles[i].apply_force(force * 0.5);
            self.particles[i + 1].apply_force(force * 0.5);
        }
    }

    // -----------------------------------------------------------------------
    // Verlet integration
    // -----------------------------------------------------------------------

    fn verlet_integrate(&mut self, dt: f32) {
        let damping = 1.0 - self.settings.damping;
        let dt2 = dt * dt;

        for p in &mut self.particles {
            if p.pinned {
                p.prev_position = p.position;
                continue;
            }

            let acceleration = p.accumulated_force * p.inv_mass;
            let new_pos =
                p.position + (p.position - p.prev_position) * damping + acceleration * dt2;
            p.prev_position = p.position;
            p.position = new_pos;
        }
    }

    // -----------------------------------------------------------------------
    // Jakobsen constraint solver
    // -----------------------------------------------------------------------

    fn solve_constraints(&mut self) {
        for c_idx in 0..self.constraints.len() {
            let c = &self.constraints[c_idx];
            let idx_a = c.particle_a;
            let idx_b = c.particle_b;
            let rest_length = c.rest_length;
            let stiffness = c.stiffness;

            let pos_a = self.particles[idx_a].position;
            let pos_b = self.particles[idx_b].position;
            let inv_mass_a = self.particles[idx_a].inv_mass;
            let inv_mass_b = self.particles[idx_b].inv_mass;

            let w_sum = inv_mass_a + inv_mass_b;
            if w_sum < 1e-12 {
                continue;
            }

            let delta = pos_b - pos_a;
            let current_length = delta.length();
            if current_length < 1e-10 {
                continue;
            }

            let diff = (current_length - rest_length) / current_length;
            let correction = delta * diff * stiffness;

            let w_a = inv_mass_a / w_sum;
            let w_b = inv_mass_b / w_sum;

            self.particles[idx_a].position += correction * w_a;
            self.particles[idx_b].position -= correction * w_b;
        }
    }

    // -----------------------------------------------------------------------
    // Self-collision
    // -----------------------------------------------------------------------

    fn solve_self_collisions(&mut self) {
        let radius = self.settings.self_collision_radius;
        let min_dist = radius * 2.0;
        let min_dist_sq = min_dist * min_dist;
        let cell_size = SELF_COLLISION_CELL_SIZE;
        let inv_cell = 1.0 / cell_size;

        // Build spatial hash
        let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (i, p) in self.particles.iter().enumerate() {
            let cx = (p.position.x * inv_cell).floor() as i32;
            let cy = (p.position.y * inv_cell).floor() as i32;
            let cz = (p.position.z * inv_cell).floor() as i32;
            grid.entry((cx, cy, cz)).or_default().push(i);
        }

        let mut corrections: Vec<(usize, Vec3)> = Vec::new();

        // Check within same cell and neighbors
        let cell_keys: Vec<(i32, i32, i32)> = grid.keys().copied().collect();
        let offsets: [(i32, i32, i32); 14] = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
            (-1, 1, 0),
            (-1, 0, 1),
            (0, -1, 1),
            (-1, 1, 1),
            (1, -1, 1),
            (-1, -1, 1),
        ];

        for key in &cell_keys {
            for offset in &offsets {
                let neighbor_key = (key.0 + offset.0, key.1 + offset.1, key.2 + offset.2);
                let cell_a = match grid.get(key) {
                    Some(v) => v,
                    None => continue,
                };
                let cell_b = if *offset == (0, 0, 0) {
                    cell_a
                } else {
                    match grid.get(&neighbor_key) {
                        Some(v) => v,
                        None => continue,
                    }
                };

                for &idx_a in cell_a {
                    for &idx_b in cell_b {
                        if idx_a >= idx_b {
                            continue;
                        }
                        // Skip adjacent particles (they are connected by constraints)
                        if (idx_a as isize - idx_b as isize).unsigned_abs() <= 2 {
                            continue;
                        }

                        let pa = self.particles[idx_a].position;
                        let pb = self.particles[idx_b].position;
                        let diff = pb - pa;
                        let dist_sq = diff.length_squared();

                        if dist_sq < min_dist_sq && dist_sq > 1e-12 {
                            let dist = dist_sq.sqrt();
                            let penetration = min_dist - dist;
                            let normal = diff / dist;
                            let half_pen = penetration * 0.5;

                            let inv_a = self.particles[idx_a].inv_mass;
                            let inv_b = self.particles[idx_b].inv_mass;
                            let w = inv_a + inv_b;
                            if w > 1e-12 {
                                corrections.push((idx_a, -normal * half_pen * (inv_a / w)));
                                corrections.push((idx_b, normal * half_pen * (inv_b / w)));
                            }
                        }
                    }
                }
            }
        }

        for (idx, correction) in corrections {
            if !self.particles[idx].pinned {
                self.particles[idx].position += correction;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Pin enforcement
    // -----------------------------------------------------------------------

    fn enforce_pins(&mut self) {
        if let RopeAttachment::WorldPin(pos) = self.start_attachment {
            if !self.particles.is_empty() {
                self.particles[0].position = pos;
                self.particles[0].prev_position = pos;
            }
        }
        if let RopeAttachment::BodyAttach { world_position, .. } = self.start_attachment {
            if !self.particles.is_empty() {
                self.particles[0].position = world_position;
                self.particles[0].prev_position = world_position;
            }
        }

        if let RopeAttachment::WorldPin(pos) = self.end_attachment {
            if let Some(last) = self.particles.last_mut() {
                last.position = pos;
                last.prev_position = pos;
            }
        }
        if let RopeAttachment::BodyAttach { world_position, .. } = self.end_attachment {
            if let Some(last) = self.particles.last_mut() {
                last.position = world_position;
                last.prev_position = world_position;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Rendering data: Catmull-Rom spline
    // -----------------------------------------------------------------------

    /// Generate a smooth curve through the rope particles using Catmull-Rom
    /// interpolation.
    ///
    /// `subdivisions` is the number of interpolated points between each pair
    /// of particles. Returns a list of world-space positions forming the curve.
    pub fn generate_smooth_curve(&self, subdivisions: usize) -> Vec<Vec3> {
        let n = self.particles.len();
        if n < 2 {
            return self.particles.iter().map(|p| p.position).collect();
        }

        let subs = subdivisions.max(1);
        let mut curve = Vec::with_capacity((n - 1) * subs + 1);

        for i in 0..n - 1 {
            // Catmull-Rom needs 4 control points: P_{i-1}, P_i, P_{i+1}, P_{i+2}
            let p0 = if i > 0 {
                self.particles[i - 1].position
            } else {
                // Extrapolate
                self.particles[0].position * 2.0 - self.particles[1].position
            };
            let p1 = self.particles[i].position;
            let p2 = self.particles[i + 1].position;
            let p3 = if i + 2 < n {
                self.particles[i + 2].position
            } else {
                // Extrapolate
                self.particles[n - 1].position * 2.0
                    - self.particles[n - 2].position
            };

            for s in 0..subs {
                let t = s as f32 / subs as f32;
                let point = catmull_rom(p0, p1, p2, p3, t);
                curve.push(point);
            }
        }

        // Add the last point
        curve.push(self.particles[n - 1].position);

        curve
    }

    /// Get particle positions as a Vec (for simple rendering).
    pub fn positions(&self) -> Vec<Vec3> {
        self.particles.iter().map(|p| p.position).collect()
    }
}

/// Catmull-Rom spline interpolation between p1 and p2.
///
/// `t` is in [0, 1], where 0 returns p1 and 1 returns p2.
/// p0 and p3 are the control points before and after the segment.
fn catmull_rom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;

    // Catmull-Rom matrix coefficients (tau = 0.5)
    let a = p1 * 2.0;
    let b = p2 - p0;
    let c = p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3;
    let d = -p0 + p1 * 3.0 - p2 * 3.0 + p3;

    (a + b * t + c * t2 + d * t3) * 0.5
}

// ---------------------------------------------------------------------------
// Rope creation helper
// ---------------------------------------------------------------------------

/// Create a rope between two world-space points.
///
/// # Arguments
/// * `start` - Start position in world space.
/// * `end` - End position in world space.
/// * `segments` - Number of segments (particles = segments + 1).
/// * `mass` - Total mass of the rope in kg.
///
/// # Returns
/// A fully initialised `RopeSimulation`.
pub fn create_rope(start: Vec3, end: Vec3, segments: usize, mass: f32) -> RopeSimulation {
    create_rope_with_settings(start, end, segments, mass, RopeSettings::default())
}

/// Create a rope with custom settings.
pub fn create_rope_with_settings(
    start: Vec3,
    end: Vec3,
    segments: usize,
    mass: f32,
    settings: RopeSettings,
) -> RopeSimulation {
    let seg = segments.max(1);
    let num_particles = seg + 1;
    let particle_mass = mass / num_particles as f32;

    let mut particles = Vec::with_capacity(num_particles);
    for i in 0..num_particles {
        let t = i as f32 / seg as f32;
        let pos = start + (end - start) * t;
        particles.push(RopeParticle::new(pos, particle_mass));
    }

    let stiffness = settings.stiffness;
    let mut constraints = Vec::with_capacity(seg);
    let mut total_rest_length = 0.0;
    for i in 0..seg {
        let rest = (particles[i + 1].position - particles[i].position).length();
        total_rest_length += rest;
        constraints.push(RopeConstraint::new(i, i + 1, rest, stiffness));
    }

    RopeSimulation {
        particles,
        constraints,
        settings,
        start_attachment: RopeAttachment::Free,
        end_attachment: RopeAttachment::Free,
        wind: Vec3::ZERO,
        wind_turbulence: 0.0,
        sim_time: 0.0,
        total_rest_length,
    }
}

/// Create a hanging rope pinned at the start.
pub fn create_hanging_rope(
    start: Vec3,
    end: Vec3,
    segments: usize,
    mass: f32,
) -> RopeSimulation {
    let mut rope = create_rope(start, end, segments, mass);
    rope.pin_start();
    rope
}

// ---------------------------------------------------------------------------
// ChainLink
// ---------------------------------------------------------------------------

/// A single rigid link in a chain.
#[derive(Debug, Clone)]
pub struct ChainLink {
    /// World-space position of the link center.
    pub position: Vec3,
    /// World-space rotation.
    pub rotation: Quat,
    /// Previous position (for Verlet-like integration).
    pub prev_position: Vec3,
    /// Link length (half-extent along the local Y axis).
    pub half_length: f32,
    /// Link capsule radius.
    pub radius: f32,
    /// Mass of the link.
    pub mass: f32,
    /// Inverse mass (0 for pinned links).
    pub inv_mass: f32,
    /// Whether this link is pinned.
    pub pinned: bool,
    /// Angular velocity (simplified).
    pub angular_velocity: Vec3,
}

impl ChainLink {
    /// Create a new chain link.
    pub fn new(position: Vec3, half_length: f32, radius: f32, mass: f32) -> Self {
        let inv_mass = if mass > 1e-8 { 1.0 / mass } else { 0.0 };
        Self {
            position,
            rotation: Quat::IDENTITY,
            prev_position: position,
            half_length,
            radius,
            mass,
            inv_mass,
            pinned: false,
            angular_velocity: Vec3::ZERO,
        }
    }

    /// Get the top attachment point (in world space).
    pub fn top_point(&self) -> Vec3 {
        self.position + self.rotation * Vec3::new(0.0, self.half_length, 0.0)
    }

    /// Get the bottom attachment point (in world space).
    pub fn bottom_point(&self) -> Vec3 {
        self.position - self.rotation * Vec3::new(0.0, self.half_length, 0.0)
    }

    /// Pin this link.
    pub fn pin(&mut self) {
        self.pinned = true;
        self.inv_mass = 0.0;
    }
}

// ---------------------------------------------------------------------------
// ChainSimulation
// ---------------------------------------------------------------------------

/// A chain simulation with rigid capsule links connected by hinge constraints.
#[derive(Debug, Clone)]
pub struct ChainSimulation {
    /// All links in the chain.
    pub links: Vec<ChainLink>,
    /// Gravity vector.
    pub gravity: Vec3,
    /// Damping factor.
    pub damping: f32,
    /// Number of constraint solver iterations.
    pub solver_iterations: usize,
    /// Running simulation time.
    sim_time: f32,
    /// Hinge axis for joints (in local space of each link).
    pub hinge_axis: Vec3,
    /// Angular limit for each hinge joint (radians, symmetric).
    pub hinge_limit: f32,
}

impl ChainSimulation {
    /// Create a new chain simulation.
    pub fn new() -> Self {
        Self {
            links: Vec::new(),
            gravity: DEFAULT_GRAVITY,
            damping: 0.02,
            solver_iterations: 8,
            sim_time: 0.0,
            hinge_axis: Vec3::Z,
            hinge_limit: std::f32::consts::FRAC_PI_2,
        }
    }

    /// Create a chain from a start point downward.
    pub fn create_chain(
        start: Vec3,
        num_links: usize,
        link_length: f32,
        link_radius: f32,
        link_mass: f32,
    ) -> Self {
        let half_len = link_length * 0.5;
        let mut chain = Self::new();

        for i in 0..num_links {
            let y_offset = -(i as f32) * link_length;
            let pos = start + Vec3::new(0.0, y_offset - half_len, 0.0);
            chain.links.push(ChainLink::new(pos, half_len, link_radius, link_mass));
        }

        chain
    }

    /// Get the number of links.
    pub fn link_count(&self) -> usize {
        self.links.len()
    }

    /// Pin the first link.
    pub fn pin_start(&mut self) {
        if !self.links.is_empty() {
            self.links[0].pin();
        }
    }

    /// Step the chain simulation forward.
    pub fn step(&mut self, dt: f32) {
        if dt <= 0.0 || self.links.is_empty() {
            return;
        }
        self.sim_time += dt;

        let damping_factor = 1.0 - self.damping;
        let gravity = self.gravity;

        // Verlet-like position integration for each link
        for link in &mut self.links {
            if link.pinned {
                link.prev_position = link.position;
                continue;
            }

            let velocity = (link.position - link.prev_position) * damping_factor;
            let acceleration = gravity * link.mass * link.inv_mass;
            let new_pos = link.position + velocity + acceleration * dt * dt;
            link.prev_position = link.position;
            link.position = new_pos;

            // Damp angular velocity
            link.angular_velocity *= damping_factor;
        }

        // Constraint solving: keep links connected at their endpoints
        for _ in 0..self.solver_iterations {
            self.solve_link_constraints();
        }

        // Update link rotations from their positions
        self.update_rotations();
    }

    fn solve_link_constraints(&mut self) {
        for i in 0..self.links.len().saturating_sub(1) {
            // The bottom of link i should coincide with the top of link i+1
            let bottom_i = self.links[i].bottom_point();
            let top_next = self.links[i + 1].top_point();

            let error = bottom_i - top_next;
            if error.length_squared() < 1e-12 {
                continue;
            }

            let inv_a = self.links[i].inv_mass;
            let inv_b = self.links[i + 1].inv_mass;
            let w = inv_a + inv_b;
            if w < 1e-12 {
                continue;
            }

            let correction_a = -error * (inv_a / w);
            let correction_b = error * (inv_b / w);

            if !self.links[i].pinned {
                self.links[i].position += correction_a;
            }
            if !self.links[i + 1].pinned {
                self.links[i + 1].position += correction_b;
            }
        }
    }

    fn update_rotations(&mut self) {
        for i in 0..self.links.len() {
            let link = &self.links[i];
            // Compute direction from bottom to top
            let dir = if link.half_length > 1e-6 {
                let top = link.position + Vec3::new(0.0, link.half_length, 0.0);
                let bottom = link.position - Vec3::new(0.0, link.half_length, 0.0);
                // For now, rotation is based on the link's position relative to neighbors
                if i > 0 {
                    let to_prev = self.links[i - 1].position - link.position;
                    if to_prev.length_squared() > 1e-8 {
                        to_prev.normalize()
                    } else {
                        Vec3::Y
                    }
                } else {
                    Vec3::Y
                }
            } else {
                Vec3::Y
            };

            // Simple rotation: align Y axis with direction to parent
            let up = Vec3::Y;
            if dir.cross(up).length_squared() > 1e-8 {
                let rotation_axis = up.cross(dir).normalize();
                let angle = up.dot(dir).clamp(-1.0, 1.0).acos();
                self.links[i].rotation =
                    Quat::from_axis_angle(rotation_axis, angle);
            }
        }
    }

    /// Get all link positions.
    pub fn positions(&self) -> Vec<Vec3> {
        self.links.iter().map(|l| l.position).collect()
    }
}

// ---------------------------------------------------------------------------
// ECS integration
// ---------------------------------------------------------------------------

/// ECS component for attaching a rope simulation to an entity.
#[derive(Debug, Clone)]
pub struct RopeComponent {
    /// The rope simulation.
    pub rope: RopeSimulation,
    /// Whether the simulation is currently active.
    pub active: bool,
}

impl RopeComponent {
    /// Create a new rope component.
    pub fn new(rope: RopeSimulation) -> Self {
        Self { rope, active: true }
    }

    /// Create a simple hanging rope component.
    pub fn hanging(start: Vec3, end: Vec3, segments: usize, mass: f32) -> Self {
        let rope = create_hanging_rope(start, end, segments, mass);
        Self::new(rope)
    }
}

/// System that steps all rope simulations each frame.
pub struct RopeSystem {
    /// Global wind affecting all ropes.
    pub global_wind: Vec3,
    /// Global wind turbulence.
    pub global_wind_turbulence: f32,
    /// Fixed time step for the rope simulation.
    pub fixed_timestep: f32,
    /// Accumulated time for fixed-step sub-stepping.
    time_accumulator: f32,
}

impl Default for RopeSystem {
    fn default() -> Self {
        Self {
            global_wind: Vec3::ZERO,
            global_wind_turbulence: 0.0,
            fixed_timestep: 1.0 / 60.0,
            time_accumulator: 0.0,
        }
    }
}

impl RopeSystem {
    /// Create a new rope system.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set global wind parameters.
    pub fn set_wind(&mut self, wind: Vec3, turbulence: f32) {
        self.global_wind = wind;
        self.global_wind_turbulence = turbulence;
    }

    /// Update all rope simulations.
    pub fn update(&mut self, dt: f32, ropes: &mut [RopeComponent]) {
        self.time_accumulator += dt;
        let mut steps = 0u32;
        let max_steps = 4u32;

        while self.time_accumulator >= self.fixed_timestep && steps < max_steps {
            for rope in ropes.iter_mut() {
                if !rope.active {
                    continue;
                }
                rope.rope.set_wind(self.global_wind, self.global_wind_turbulence);
                rope.rope.step(self.fixed_timestep);
            }
            self.time_accumulator -= self.fixed_timestep;
            steps += 1;
        }

        if self.time_accumulator > self.fixed_timestep {
            self.time_accumulator = 0.0;
        }
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_rope() {
        let rope = create_rope(Vec3::ZERO, Vec3::new(0.0, -5.0, 0.0), 10, 1.0);
        assert_eq!(rope.particle_count(), 11);
        assert_eq!(rope.constraints.len(), 10);
        assert!(rope.total_rest_length > 0.0);
    }

    #[test]
    fn test_hanging_rope_falls() {
        let mut rope = create_hanging_rope(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(5.0, 5.0, 0.0),
            10,
            1.0,
        );

        let initial_y = rope.particles[5].position.y;

        for _ in 0..60 {
            rope.step(1.0 / 60.0);
        }

        // Middle particles should sag under gravity
        assert!(
            rope.particles[5].position.y < initial_y,
            "Middle should sag: {} vs {}",
            rope.particles[5].position.y,
            initial_y
        );

        // Start should remain pinned
        assert!(
            (rope.particles[0].position - Vec3::new(0.0, 5.0, 0.0)).length() < 1e-4,
            "Start should stay pinned"
        );
    }

    #[test]
    fn test_rope_constraints_maintain_distance() {
        let mut rope = create_rope(Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0), 5, 1.0);
        rope.pin_start();
        rope.pin_end();

        // Displace middle particle
        rope.particles[2].position.y += 10.0;

        for _ in 0..120 {
            rope.step(1.0 / 60.0);
        }

        // Check that consecutive segment lengths are reasonable
        for i in 0..rope.constraints.len() {
            let c = &rope.constraints[i];
            let dist = (rope.particles[c.particle_b].position
                - rope.particles[c.particle_a].position)
                .length();
            // Should be within 20% of rest length after settling
            assert!(
                dist < c.rest_length * 1.5,
                "Segment {} too stretched: {} vs rest {}",
                i,
                dist,
                c.rest_length
            );
        }
    }

    #[test]
    fn test_catmull_rom() {
        let p0 = Vec3::new(-1.0, 0.0, 0.0);
        let p1 = Vec3::new(0.0, 0.0, 0.0);
        let p2 = Vec3::new(1.0, 0.0, 0.0);
        let p3 = Vec3::new(2.0, 0.0, 0.0);

        // At t=0, should return p1
        let at_0 = catmull_rom(p0, p1, p2, p3, 0.0);
        assert!((at_0 - p1).length() < 1e-4);

        // At t=1, should return p2
        let at_1 = catmull_rom(p0, p1, p2, p3, 1.0);
        assert!((at_1 - p2).length() < 1e-4);
    }

    #[test]
    fn test_smooth_curve_generation() {
        let rope = create_rope(Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0), 4, 1.0);
        let curve = rope.generate_smooth_curve(4);

        // 4 segments * 4 subdivisions + 1 = 17 points
        assert_eq!(curve.len(), 17);
    }

    #[test]
    fn test_rope_wind() {
        let mut rope = create_hanging_rope(Vec3::ZERO, Vec3::new(0.0, -3.0, 0.0), 5, 0.5);
        rope.set_wind(Vec3::new(5.0, 0.0, 0.0), 0.0);

        for _ in 0..60 {
            rope.step(1.0 / 60.0);
        }

        // Wind should push particles in the +X direction
        let has_moved_x = rope.particles.iter().any(|p| p.position.x > 0.01);
        assert!(has_moved_x, "Wind should push rope in +X");
    }

    #[test]
    fn test_chain_creation() {
        let chain = ChainSimulation::create_chain(Vec3::ZERO, 5, 0.5, 0.05, 0.2);
        assert_eq!(chain.link_count(), 5);
    }

    #[test]
    fn test_chain_falls() {
        let mut chain = ChainSimulation::create_chain(Vec3::new(0.0, 5.0, 0.0), 5, 0.5, 0.05, 0.2);
        chain.pin_start();

        let initial_y = chain.links[4].position.y;

        for _ in 0..120 {
            chain.step(1.0 / 60.0);
        }

        assert!(
            chain.links[4].position.y < initial_y,
            "Chain end should fall"
        );
        assert!(chain.links[0].pinned, "First link should stay pinned");
    }

    #[test]
    fn test_rope_component() {
        let component = RopeComponent::hanging(Vec3::ZERO, Vec3::new(3.0, 0.0, 0.0), 5, 0.5);
        assert!(component.active);
        assert_eq!(component.rope.particle_count(), 6);
    }

    #[test]
    fn test_rope_system_update() {
        let mut system = RopeSystem::new();
        let mut ropes = vec![RopeComponent::hanging(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(3.0, 5.0, 0.0),
            5,
            0.5,
        )];

        system.update(1.0 / 60.0, &mut ropes);

        // Middle particle should have moved (fallen slightly)
        assert!(ropes[0].rope.particles[3].position.y < 5.0);
    }

    #[test]
    fn test_rope_current_length() {
        let rope = create_rope(Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0), 5, 1.0);
        let length = rope.current_length();
        assert!((length - 5.0).abs() < 1e-3);
    }

    #[test]
    fn test_rope_pin_to_position() {
        let mut rope = create_rope(Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0), 5, 1.0);
        let target = Vec3::new(10.0, 10.0, 10.0);
        rope.pin_start_to(target);

        assert!(rope.particles[0].pinned);
        assert!((rope.particles[0].position - target).length() < 1e-4);
    }
}
