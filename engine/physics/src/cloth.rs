//! Cloth simulation using mass-spring systems with Verlet integration.
//!
//! Provides a complete cloth simulation pipeline:
//! - Verlet integration for position-based dynamics
//! - Jakobsen constraint relaxation (structural, shear, bend)
//! - Self-collision via spatial hashing
//! - External collision (sphere, plane)
//! - Wind force (per-triangle area-weighted)
//! - Pin constraints (fix particles to world positions or bones)
//! - Tearing (break constraints exceeding max stretch)
//! - ECS integration via `ClothComponent` and `ClothSystem`

use std::collections::HashMap;

use glam::{Vec2, Vec3};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default number of constraint solver iterations per step.
const DEFAULT_SOLVER_ITERATIONS: usize = 15;
/// Default damping factor for Verlet integration (0 = no damping, 1 = full damping).
const DEFAULT_DAMPING: f32 = 0.01;
/// Default gravity vector.
const DEFAULT_GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);
/// Spatial hash cell size for self-collision detection.
const SELF_COLLISION_CELL_SIZE: f32 = 0.5;
/// Minimum particle separation distance for self-collision.
const SELF_COLLISION_RADIUS: f32 = 0.05;
/// Maximum stretch ratio before a constraint tears.
const DEFAULT_MAX_STRETCH_RATIO: f32 = 2.5;

// ---------------------------------------------------------------------------
// Constraint type classification
// ---------------------------------------------------------------------------

/// The type of cloth constraint, determining its structural role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClothConstraintType {
    /// Direct neighbor connections (horizontal/vertical in grid).
    Structural,
    /// Diagonal connections for shear resistance.
    Shear,
    /// Skip-one connections for bending resistance.
    Bend,
}

// ---------------------------------------------------------------------------
// Cloth particle
// ---------------------------------------------------------------------------

/// A single mass point in the cloth mesh.
#[derive(Debug, Clone)]
pub struct ClothParticle {
    /// Current world-space position.
    pub position: Vec3,
    /// Previous position (used for Verlet integration).
    pub prev_position: Vec3,
    /// Current velocity (derived from position delta, used for external queries).
    pub velocity: Vec3,
    /// Accumulated external force for the current step.
    pub accumulated_force: Vec3,
    /// Particle mass in kg.
    pub mass: f32,
    /// Inverse mass (0.0 for pinned particles).
    pub inv_mass: f32,
    /// Whether this particle is fixed in world space.
    pub pinned: bool,
    /// UV coordinates for rendering.
    pub uv: Vec2,
    /// Normal vector (computed from surrounding triangles).
    pub normal: Vec3,
}

impl ClothParticle {
    /// Create a new particle at the given position with the specified mass.
    pub fn new(position: Vec3, mass: f32, uv: Vec2) -> Self {
        let inv_mass = if mass > 1e-8 { 1.0 / mass } else { 0.0 };
        Self {
            position,
            prev_position: position,
            velocity: Vec3::ZERO,
            accumulated_force: Vec3::ZERO,
            mass,
            inv_mass,
            pinned: false,
            uv,
            normal: Vec3::Y,
        }
    }

    /// Pin this particle so it cannot move.
    pub fn pin(&mut self) {
        self.pinned = true;
        self.inv_mass = 0.0;
    }

    /// Unpin this particle, restoring its original inverse mass.
    pub fn unpin(&mut self) {
        self.pinned = false;
        if self.mass > 1e-8 {
            self.inv_mass = 1.0 / self.mass;
        }
    }

    /// Apply a force to this particle (accumulated over the step).
    pub fn apply_force(&mut self, force: Vec3) {
        if !self.pinned {
            self.accumulated_force += force;
        }
    }
}

// ---------------------------------------------------------------------------
// Cloth constraint (distance constraint between two particles)
// ---------------------------------------------------------------------------

/// A distance constraint between two particles in the cloth mesh.
#[derive(Debug, Clone)]
pub struct ClothConstraint {
    /// Index of the first particle.
    pub particle_a: usize,
    /// Index of the second particle.
    pub particle_b: usize,
    /// Rest length (initial distance when the constraint was created).
    pub rest_length: f32,
    /// Stiffness coefficient [0, 1]. Higher = stiffer.
    pub stiffness: f32,
    /// The structural role of this constraint.
    pub constraint_type: ClothConstraintType,
    /// Whether this constraint has been torn.
    pub torn: bool,
}

impl ClothConstraint {
    /// Create a new constraint between two particles.
    pub fn new(
        particle_a: usize,
        particle_b: usize,
        rest_length: f32,
        stiffness: f32,
        constraint_type: ClothConstraintType,
    ) -> Self {
        Self {
            particle_a,
            particle_b,
            rest_length,
            stiffness,
            constraint_type,
            torn: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Pin constraint (attach a particle to a world position or bone)
// ---------------------------------------------------------------------------

/// A pin constraint that fixes a particle to a target position.
#[derive(Debug, Clone)]
pub struct PinConstraint {
    /// Index of the pinned particle.
    pub particle_index: usize,
    /// The world-space target position.
    pub target_position: Vec3,
    /// Optional bone index for skeletal attachment.
    pub bone_index: Option<usize>,
    /// Local offset from the bone's transform.
    pub bone_local_offset: Vec3,
    /// Stiffness of the pin [0, 1]. 1.0 = hard pin, < 1.0 = soft pin.
    pub stiffness: f32,
}

impl PinConstraint {
    /// Create a hard pin to a world position.
    pub fn world(particle_index: usize, position: Vec3) -> Self {
        Self {
            particle_index,
            target_position: position,
            bone_index: None,
            bone_local_offset: Vec3::ZERO,
            stiffness: 1.0,
        }
    }

    /// Create a pin attached to a bone.
    pub fn bone(particle_index: usize, bone_index: usize, local_offset: Vec3) -> Self {
        Self {
            particle_index,
            target_position: Vec3::ZERO,
            bone_index: Some(bone_index),
            bone_local_offset: local_offset,
            stiffness: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Collision primitives for cloth
// ---------------------------------------------------------------------------

/// Sphere collider for cloth-vs-sphere collision.
#[derive(Debug, Clone)]
pub struct ClothSphereCollider {
    /// Center of the sphere in world space.
    pub center: Vec3,
    /// Radius of the sphere.
    pub radius: f32,
    /// Friction coefficient for sliding along the surface.
    pub friction: f32,
}

/// Infinite plane collider for cloth-vs-plane collision.
#[derive(Debug, Clone)]
pub struct ClothPlaneCollider {
    /// A point on the plane.
    pub point: Vec3,
    /// The plane normal (particles below the plane are pushed out).
    pub normal: Vec3,
    /// Friction coefficient.
    pub friction: f32,
}

// ---------------------------------------------------------------------------
// Triangle (for rendering and wind force)
// ---------------------------------------------------------------------------

/// A triangle in the cloth mesh, referencing three particle indices.
#[derive(Debug, Clone, Copy)]
pub struct ClothTriangle {
    pub indices: [usize; 3],
}

// ---------------------------------------------------------------------------
// Cloth settings
// ---------------------------------------------------------------------------

/// Configuration for cloth simulation behavior.
#[derive(Debug, Clone)]
pub struct ClothSettings {
    /// Number of solver iterations per step.
    pub solver_iterations: usize,
    /// Damping factor for Verlet integration.
    pub damping: f32,
    /// Gravity vector.
    pub gravity: Vec3,
    /// Stiffness for structural constraints [0, 1].
    pub structural_stiffness: f32,
    /// Stiffness for shear constraints [0, 1].
    pub shear_stiffness: f32,
    /// Stiffness for bend constraints [0, 1].
    pub bend_stiffness: f32,
    /// Maximum stretch ratio before tearing (ratio of current length to rest length).
    pub max_stretch_ratio: f32,
    /// Whether tearing is enabled.
    pub tearing_enabled: bool,
    /// Whether self-collision is enabled.
    pub self_collision_enabled: bool,
    /// Self-collision particle radius.
    pub self_collision_radius: f32,
    /// Simulation time step.
    pub time_step: f32,
}

impl Default for ClothSettings {
    fn default() -> Self {
        Self {
            solver_iterations: DEFAULT_SOLVER_ITERATIONS,
            damping: DEFAULT_DAMPING,
            gravity: DEFAULT_GRAVITY,
            structural_stiffness: 1.0,
            shear_stiffness: 0.8,
            bend_stiffness: 0.3,
            max_stretch_ratio: DEFAULT_MAX_STRETCH_RATIO,
            tearing_enabled: false,
            self_collision_enabled: false,
            self_collision_radius: SELF_COLLISION_RADIUS,
            time_step: 1.0 / 60.0,
        }
    }
}

// ---------------------------------------------------------------------------
// ClothMesh — the main simulation structure
// ---------------------------------------------------------------------------

/// A cloth simulation mesh containing particles, constraints, and triangles.
///
/// Uses Verlet integration for time-stepping and Jakobsen constraint relaxation
/// for maintaining structural integrity.
#[derive(Debug, Clone)]
pub struct ClothMesh {
    /// All particles in the cloth.
    pub particles: Vec<ClothParticle>,
    /// Distance constraints (structural, shear, bend).
    pub constraints: Vec<ClothConstraint>,
    /// Triangles for rendering and wind force computation.
    pub triangles: Vec<ClothTriangle>,
    /// Pin constraints fixing particles to world positions or bones.
    pub pin_constraints: Vec<PinConstraint>,
    /// Grid resolution (width in particles).
    pub grid_width: usize,
    /// Grid resolution (height in particles).
    pub grid_height: usize,
    /// Simulation settings.
    pub settings: ClothSettings,
    /// Sphere colliders in the scene.
    pub sphere_colliders: Vec<ClothSphereCollider>,
    /// Plane colliders in the scene.
    pub plane_colliders: Vec<ClothPlaneCollider>,
    /// Current wind direction and strength.
    pub wind: Vec3,
    /// Wind turbulence (randomization factor).
    pub wind_turbulence: f32,
    /// Running simulation time for turbulence noise.
    sim_time: f32,
}

impl ClothMesh {
    /// Create a new empty cloth mesh with default settings.
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            constraints: Vec::new(),
            triangles: Vec::new(),
            pin_constraints: Vec::new(),
            grid_width: 0,
            grid_height: 0,
            settings: ClothSettings::default(),
            sphere_colliders: Vec::new(),
            plane_colliders: Vec::new(),
            wind: Vec3::ZERO,
            wind_turbulence: 0.0,
            sim_time: 0.0,
        }
    }

    /// Get the total number of active (non-torn) constraints.
    pub fn active_constraint_count(&self) -> usize {
        self.constraints.iter().filter(|c| !c.torn).count()
    }

    /// Get the total number of particles.
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Add a sphere collider to the cloth simulation.
    pub fn add_sphere_collider(&mut self, collider: ClothSphereCollider) {
        self.sphere_colliders.push(collider);
    }

    /// Add a plane collider to the cloth simulation.
    pub fn add_plane_collider(&mut self, collider: ClothPlaneCollider) {
        self.plane_colliders.push(collider);
    }

    /// Pin a particle at the given index.
    pub fn pin_particle(&mut self, index: usize) {
        if index < self.particles.len() {
            let pos = self.particles[index].position;
            self.particles[index].pin();
            self.pin_constraints.push(PinConstraint::world(index, pos));
        }
    }

    /// Unpin a particle at the given index.
    pub fn unpin_particle(&mut self, index: usize) {
        if index < self.particles.len() {
            self.particles[index].unpin();
            self.pin_constraints.retain(|p| p.particle_index != index);
        }
    }

    /// Set the wind vector (direction * strength).
    pub fn set_wind(&mut self, wind: Vec3, turbulence: f32) {
        self.wind = wind;
        self.wind_turbulence = turbulence;
    }

    /// Step the cloth simulation forward by one time step.
    ///
    /// Pipeline:
    /// 1. Apply external forces (gravity, wind)
    /// 2. Verlet integration
    /// 3. Constraint relaxation (Jakobsen method)
    /// 4. Collision response (sphere, plane, self)
    /// 5. Pin constraints
    /// 6. Tearing check
    /// 7. Compute normals
    pub fn step(&mut self, dt: f32) {
        let dt = if dt > 0.0 { dt } else { self.settings.time_step };
        self.sim_time += dt;

        // 1. Apply gravity and wind forces
        self.apply_gravity();
        self.apply_wind_forces();

        // 2. Verlet integration
        self.verlet_integrate(dt);

        // 3. Constraint relaxation (multiple iterations)
        for _ in 0..self.settings.solver_iterations {
            self.solve_constraints();
        }

        // 4. Collision response
        self.solve_sphere_collisions();
        self.solve_plane_collisions();
        if self.settings.self_collision_enabled {
            self.solve_self_collisions();
        }

        // 5. Pin constraints
        self.solve_pin_constraints();

        // 6. Tearing
        if self.settings.tearing_enabled {
            self.check_tearing();
        }

        // 7. Compute normals for rendering
        self.compute_normals();

        // Update velocity approximation
        let inv_dt = if dt > 1e-8 { 1.0 / dt } else { 0.0 };
        for p in &mut self.particles {
            p.velocity = (p.position - p.prev_position) * inv_dt;
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

    /// Apply wind force using per-triangle area-weighted normals.
    ///
    /// For each triangle, the wind force is:
    ///   F = dot(wind, normal) * normal * area
    /// distributed equally among the three vertices.
    fn apply_wind_forces(&mut self) {
        if self.wind.length_squared() < 1e-8 {
            return;
        }

        // Simple turbulence: vary wind direction slightly using time-based noise
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

        // Collect per-triangle forces, then apply
        let mut forces = vec![Vec3::ZERO; self.particles.len()];

        for tri in &self.triangles {
            let p0 = self.particles[tri.indices[0]].position;
            let p1 = self.particles[tri.indices[1]].position;
            let p2 = self.particles[tri.indices[2]].position;

            let e1 = p1 - p0;
            let e2 = p2 - p0;
            let cross = e1.cross(e2);
            let area = cross.length() * 0.5;
            if area < 1e-8 {
                continue;
            }
            let normal = cross / (area * 2.0);

            // Wind force proportional to the projected area
            let wind_dot = effective_wind.dot(normal);
            let force = normal * wind_dot * area;
            let per_vertex = force / 3.0;

            for &idx in &tri.indices {
                forces[idx] += per_vertex;
            }
        }

        for (i, p) in self.particles.iter_mut().enumerate() {
            p.apply_force(forces[i]);
        }
    }

    // -----------------------------------------------------------------------
    // Verlet integration
    // -----------------------------------------------------------------------

    /// Verlet integration:
    ///   new_pos = pos + (pos - prev_pos) * (1 - damping) + acceleration * dt^2
    fn verlet_integrate(&mut self, dt: f32) {
        let damping = 1.0 - self.settings.damping;
        let dt2 = dt * dt;

        for p in &mut self.particles {
            if p.pinned {
                p.prev_position = p.position;
                continue;
            }

            let acceleration = p.accumulated_force * p.inv_mass;
            let new_pos = p.position + (p.position - p.prev_position) * damping + acceleration * dt2;
            p.prev_position = p.position;
            p.position = new_pos;
        }
    }

    // -----------------------------------------------------------------------
    // Jakobsen constraint solver
    // -----------------------------------------------------------------------

    /// Solve all distance constraints using Jakobsen's method.
    ///
    /// For each constraint:
    ///   delta = p2 - p1
    ///   diff = (|delta| - rest_length) / |delta|
    ///   p1 += delta * diff * 0.5 * stiffness * w1/(w1+w2)
    ///   p2 -= delta * diff * 0.5 * stiffness * w2/(w1+w2)
    fn solve_constraints(&mut self) {
        for c_idx in 0..self.constraints.len() {
            if self.constraints[c_idx].torn {
                continue;
            }

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
                continue; // Both pinned
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
    // Collision response
    // -----------------------------------------------------------------------

    /// Push particles out of sphere colliders.
    fn solve_sphere_collisions(&mut self) {
        for collider in &self.sphere_colliders {
            for p in &mut self.particles {
                if p.pinned {
                    continue;
                }

                let diff = p.position - collider.center;
                let dist = diff.length();
                let min_dist = collider.radius + SELF_COLLISION_RADIUS;

                if dist < min_dist && dist > 1e-8 {
                    let normal = diff / dist;
                    let penetration = min_dist - dist;
                    p.position += normal * penetration;

                    // Friction: remove tangential velocity component
                    if collider.friction > 0.0 {
                        let vel = p.position - p.prev_position;
                        let vel_n = vel.dot(normal) * normal;
                        let vel_t = vel - vel_n;
                        p.prev_position = p.position - vel_n - vel_t * (1.0 - collider.friction);
                    }
                }
            }
        }
    }

    /// Push particles above plane colliders.
    fn solve_plane_collisions(&mut self) {
        for collider in &self.plane_colliders {
            let normal = collider.normal.normalize();
            for p in &mut self.particles {
                if p.pinned {
                    continue;
                }

                let dist = (p.position - collider.point).dot(normal);
                if dist < SELF_COLLISION_RADIUS {
                    let penetration = SELF_COLLISION_RADIUS - dist;
                    p.position += normal * penetration;

                    // Friction
                    if collider.friction > 0.0 {
                        let vel = p.position - p.prev_position;
                        let vel_n = vel.dot(normal) * normal;
                        let vel_t = vel - vel_n;
                        p.prev_position = p.position - vel_n - vel_t * (1.0 - collider.friction);
                    }
                }
            }
        }
    }

    /// Self-collision detection and response using a spatial hash grid.
    ///
    /// Particles that are too close to each other (but not connected by a constraint)
    /// are pushed apart symmetrically.
    fn solve_self_collisions(&mut self) {
        let radius = self.settings.self_collision_radius;
        let radius_sq = radius * radius;
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

        // Build a set of connected pairs for exclusion
        let mut connected: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        for c in &self.constraints {
            if !c.torn {
                let a = c.particle_a.min(c.particle_b);
                let b = c.particle_a.max(c.particle_b);
                connected.insert((a, b));
            }
        }

        // Collect corrections
        let mut corrections: Vec<(usize, Vec3)> = Vec::new();

        for (_, cell_particles) in &grid {
            for i in 0..cell_particles.len() {
                for j in (i + 1)..cell_particles.len() {
                    let idx_a = cell_particles[i];
                    let idx_b = cell_particles[j];
                    let a = idx_a.min(idx_b);
                    let b = idx_a.max(idx_b);

                    if connected.contains(&(a, b)) {
                        continue;
                    }

                    let pa = self.particles[idx_a].position;
                    let pb = self.particles[idx_b].position;
                    let diff = pb - pa;
                    let dist_sq = diff.length_squared();

                    if dist_sq < radius_sq * 4.0 && dist_sq > 1e-12 {
                        let dist = dist_sq.sqrt();
                        let min_dist = radius * 2.0;
                        if dist < min_dist {
                            let normal = diff / dist;
                            let penetration = min_dist - dist;
                            let half_pen = penetration * 0.5;

                            let inv_a = self.particles[idx_a].inv_mass;
                            let inv_b = self.particles[idx_b].inv_mass;
                            let w_sum = inv_a + inv_b;
                            if w_sum > 1e-12 {
                                corrections.push((idx_a, -normal * half_pen * (inv_a / w_sum)));
                                corrections.push((idx_b, normal * half_pen * (inv_b / w_sum)));
                            }
                        }
                    }
                }
            }
        }

        // Also check neighboring cells
        let offsets: [(i32, i32, i32); 13] = [
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

        let cell_keys: Vec<(i32, i32, i32)> = grid.keys().copied().collect();
        for key in &cell_keys {
            let cell_a = match grid.get(key) {
                Some(v) => v.clone(),
                None => continue,
            };
            for offset in &offsets {
                let neighbor_key = (key.0 + offset.0, key.1 + offset.1, key.2 + offset.2);
                let cell_b = match grid.get(&neighbor_key) {
                    Some(v) => v,
                    None => continue,
                };
                for &idx_a in &cell_a {
                    for &idx_b in cell_b {
                        let a = idx_a.min(idx_b);
                        let b = idx_a.max(idx_b);
                        if connected.contains(&(a, b)) {
                            continue;
                        }

                        let pa = self.particles[idx_a].position;
                        let pb = self.particles[idx_b].position;
                        let diff = pb - pa;
                        let dist_sq = diff.length_squared();

                        if dist_sq < radius_sq * 4.0 && dist_sq > 1e-12 {
                            let dist = dist_sq.sqrt();
                            let min_dist = radius * 2.0;
                            if dist < min_dist {
                                let normal = diff / dist;
                                let penetration = min_dist - dist;
                                let half_pen = penetration * 0.5;

                                let inv_a = self.particles[idx_a].inv_mass;
                                let inv_b = self.particles[idx_b].inv_mass;
                                let w_sum = inv_a + inv_b;
                                if w_sum > 1e-12 {
                                    corrections.push((idx_a, -normal * half_pen * (inv_a / w_sum)));
                                    corrections.push((idx_b, normal * half_pen * (inv_b / w_sum)));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Apply corrections
        for (idx, correction) in corrections {
            if !self.particles[idx].pinned {
                self.particles[idx].position += correction;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Pin constraints
    // -----------------------------------------------------------------------

    fn solve_pin_constraints(&mut self) {
        for pin in &self.pin_constraints {
            let idx = pin.particle_index;
            if idx >= self.particles.len() {
                continue;
            }

            let target = pin.target_position;
            let stiffness = pin.stiffness;

            if stiffness >= 1.0 - 1e-6 {
                // Hard pin: set position directly
                self.particles[idx].position = target;
                self.particles[idx].prev_position = target;
            } else {
                // Soft pin: interpolate toward target
                let current = self.particles[idx].position;
                self.particles[idx].position = current + (target - current) * stiffness;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Tearing
    // -----------------------------------------------------------------------

    /// Check constraints for tearing. If the stretch ratio exceeds the maximum,
    /// mark the constraint as torn.
    fn check_tearing(&mut self) {
        let max_ratio = self.settings.max_stretch_ratio;
        for c in &mut self.constraints {
            if c.torn {
                continue;
            }

            let pos_a = self.particles[c.particle_a].position;
            let pos_b = self.particles[c.particle_b].position;
            let current_length = (pos_b - pos_a).length();
            let ratio = current_length / c.rest_length.max(1e-8);

            if ratio > max_ratio {
                c.torn = true;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Normal computation
    // -----------------------------------------------------------------------

    /// Compute smooth vertex normals by averaging face normals of adjacent triangles.
    fn compute_normals(&mut self) {
        // Reset normals
        for p in &mut self.particles {
            p.normal = Vec3::ZERO;
        }

        // Accumulate face normals
        for tri in &self.triangles {
            let p0 = self.particles[tri.indices[0]].position;
            let p1 = self.particles[tri.indices[1]].position;
            let p2 = self.particles[tri.indices[2]].position;

            let e1 = p1 - p0;
            let e2 = p2 - p0;
            let face_normal = e1.cross(e2);

            for &idx in &tri.indices {
                self.particles[idx].normal += face_normal;
            }
        }

        // Normalize
        for p in &mut self.particles {
            let len = p.normal.length();
            if len > 1e-8 {
                p.normal /= len;
            } else {
                p.normal = Vec3::Y;
            }
        }
    }

    /// Get particle positions as a flat Vec for GPU upload.
    pub fn positions(&self) -> Vec<Vec3> {
        self.particles.iter().map(|p| p.position).collect()
    }

    /// Get particle normals as a flat Vec for GPU upload.
    pub fn normals(&self) -> Vec<Vec3> {
        self.particles.iter().map(|p| p.normal).collect()
    }

    /// Get triangle indices as a flat Vec for GPU upload.
    pub fn indices(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.triangles.len() * 3);
        for tri in &self.triangles {
            out.push(tri.indices[0] as u32);
            out.push(tri.indices[1] as u32);
            out.push(tri.indices[2] as u32);
        }
        out
    }

    /// Get UVs as a flat Vec for GPU upload.
    pub fn uvs(&self) -> Vec<Vec2> {
        self.particles.iter().map(|p| p.uv).collect()
    }
}

// ---------------------------------------------------------------------------
// Cloth creation helper
// ---------------------------------------------------------------------------

/// Create a rectangular cloth mesh with the given dimensions and resolution.
///
/// The cloth is created in the XZ plane at y=0, with particles spaced evenly.
/// Structural, shear, and bend constraints are automatically generated.
///
/// # Arguments
/// * `width` - Total width of the cloth in world units.
/// * `height` - Total height (depth) of the cloth in world units.
/// * `resolution_x` - Number of particles along the width.
/// * `resolution_y` - Number of particles along the height.
/// * `particle_mass` - Mass of each particle in kg.
/// * `settings` - Simulation settings.
///
/// # Returns
/// A fully initialized `ClothMesh` ready for simulation.
pub fn create_cloth(
    width: f32,
    height: f32,
    resolution_x: usize,
    resolution_y: usize,
    particle_mass: f32,
    settings: ClothSettings,
) -> ClothMesh {
    let res_x = resolution_x.max(2);
    let res_y = resolution_y.max(2);
    let total_particles = res_x * res_y;
    let spacing_x = width / (res_x - 1) as f32;
    let spacing_y = height / (res_y - 1) as f32;

    // Create particles in a grid
    let mut particles = Vec::with_capacity(total_particles);
    for y in 0..res_y {
        for x in 0..res_x {
            let pos = Vec3::new(
                x as f32 * spacing_x - width * 0.5,
                0.0,
                y as f32 * spacing_y - height * 0.5,
            );
            let uv = Vec2::new(x as f32 / (res_x - 1) as f32, y as f32 / (res_y - 1) as f32);
            particles.push(ClothParticle::new(pos, particle_mass, uv));
        }
    }

    let mut constraints = Vec::new();

    // Helper to compute index in the grid
    let idx = |x: usize, y: usize| -> usize { y * res_x + x };

    // Structural constraints (horizontal + vertical)
    for y in 0..res_y {
        for x in 0..res_x {
            let i = idx(x, y);

            // Right neighbor
            if x + 1 < res_x {
                let j = idx(x + 1, y);
                let rest = (particles[i].position - particles[j].position).length();
                constraints.push(ClothConstraint::new(
                    i,
                    j,
                    rest,
                    settings.structural_stiffness,
                    ClothConstraintType::Structural,
                ));
            }

            // Bottom neighbor
            if y + 1 < res_y {
                let j = idx(x, y + 1);
                let rest = (particles[i].position - particles[j].position).length();
                constraints.push(ClothConstraint::new(
                    i,
                    j,
                    rest,
                    settings.structural_stiffness,
                    ClothConstraintType::Structural,
                ));
            }
        }
    }

    // Shear constraints (diagonals)
    for y in 0..res_y {
        for x in 0..res_x {
            let i = idx(x, y);

            // Bottom-right diagonal
            if x + 1 < res_x && y + 1 < res_y {
                let j = idx(x + 1, y + 1);
                let rest = (particles[i].position - particles[j].position).length();
                constraints.push(ClothConstraint::new(
                    i,
                    j,
                    rest,
                    settings.shear_stiffness,
                    ClothConstraintType::Shear,
                ));
            }

            // Bottom-left diagonal
            if x > 0 && y + 1 < res_y {
                let j = idx(x - 1, y + 1);
                let rest = (particles[i].position - particles[j].position).length();
                constraints.push(ClothConstraint::new(
                    i,
                    j,
                    rest,
                    settings.shear_stiffness,
                    ClothConstraintType::Shear,
                ));
            }
        }
    }

    // Bend constraints (skip-one neighbor)
    for y in 0..res_y {
        for x in 0..res_x {
            let i = idx(x, y);

            // Two-right
            if x + 2 < res_x {
                let j = idx(x + 2, y);
                let rest = (particles[i].position - particles[j].position).length();
                constraints.push(ClothConstraint::new(
                    i,
                    j,
                    rest,
                    settings.bend_stiffness,
                    ClothConstraintType::Bend,
                ));
            }

            // Two-down
            if y + 2 < res_y {
                let j = idx(x, y + 2);
                let rest = (particles[i].position - particles[j].position).length();
                constraints.push(ClothConstraint::new(
                    i,
                    j,
                    rest,
                    settings.bend_stiffness,
                    ClothConstraintType::Bend,
                ));
            }
        }
    }

    // Generate triangles (two per quad cell)
    let mut triangles = Vec::new();
    for y in 0..(res_y - 1) {
        for x in 0..(res_x - 1) {
            let tl = idx(x, y);
            let tr = idx(x + 1, y);
            let bl = idx(x, y + 1);
            let br = idx(x + 1, y + 1);

            triangles.push(ClothTriangle {
                indices: [tl, bl, tr],
            });
            triangles.push(ClothTriangle {
                indices: [tr, bl, br],
            });
        }
    }

    ClothMesh {
        particles,
        constraints,
        triangles,
        pin_constraints: Vec::new(),
        grid_width: res_x,
        grid_height: res_y,
        settings,
        sphere_colliders: Vec::new(),
        plane_colliders: Vec::new(),
        wind: Vec3::ZERO,
        wind_turbulence: 0.0,
        sim_time: 0.0,
    }
}

/// Create a cloth mesh and pin the top row of particles.
pub fn create_hanging_cloth(
    width: f32,
    height: f32,
    resolution_x: usize,
    resolution_y: usize,
    particle_mass: f32,
    settings: ClothSettings,
) -> ClothMesh {
    let mut cloth = create_cloth(width, height, resolution_x, resolution_y, particle_mass, settings);

    // Pin the top row (y=0 in the grid, which is at z = -height/2)
    let res_x = resolution_x.max(2);
    for x in 0..res_x {
        cloth.pin_particle(x);
    }

    cloth
}

// ---------------------------------------------------------------------------
// ECS integration
// ---------------------------------------------------------------------------

/// ECS component for attaching a cloth simulation to an entity.
#[derive(Debug, Clone)]
pub struct ClothComponent {
    /// The cloth simulation mesh.
    pub mesh: ClothMesh,
    /// Whether the simulation is currently active.
    pub active: bool,
    /// Whether the cloth should respond to collisions with the physics world.
    pub interact_with_physics: bool,
}

impl ClothComponent {
    /// Create a new cloth component from a cloth mesh.
    pub fn new(mesh: ClothMesh) -> Self {
        Self {
            mesh,
            active: true,
            interact_with_physics: true,
        }
    }

    /// Create a simple hanging cloth component.
    pub fn hanging(width: f32, height: f32, resolution: usize) -> Self {
        let settings = ClothSettings::default();
        let mesh = create_hanging_cloth(width, height, resolution, resolution, 0.1, settings);
        Self::new(mesh)
    }
}

/// System that steps all cloth simulations each frame.
pub struct ClothSystem {
    /// Global wind affecting all cloths.
    pub global_wind: Vec3,
    /// Global wind turbulence.
    pub global_wind_turbulence: f32,
    /// Fixed time step for the cloth simulation.
    pub fixed_timestep: f32,
    /// Accumulated time for fixed-step sub-stepping.
    time_accumulator: f32,
}

impl Default for ClothSystem {
    fn default() -> Self {
        Self {
            global_wind: Vec3::ZERO,
            global_wind_turbulence: 0.0,
            fixed_timestep: 1.0 / 60.0,
            time_accumulator: 0.0,
        }
    }
}

impl ClothSystem {
    /// Create a new cloth system.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set global wind parameters.
    pub fn set_wind(&mut self, wind: Vec3, turbulence: f32) {
        self.global_wind = wind;
        self.global_wind_turbulence = turbulence;
    }

    /// Update all cloth simulations by the given frame delta time.
    pub fn update(&mut self, dt: f32, cloths: &mut [ClothComponent]) {
        self.time_accumulator += dt;
        let mut steps = 0u32;
        let max_steps = 4u32;

        while self.time_accumulator >= self.fixed_timestep && steps < max_steps {
            for cloth in cloths.iter_mut() {
                if !cloth.active {
                    continue;
                }
                cloth.mesh.set_wind(self.global_wind, self.global_wind_turbulence);
                cloth.mesh.step(self.fixed_timestep);
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
    fn test_create_cloth() {
        let settings = ClothSettings::default();
        let cloth = create_cloth(2.0, 2.0, 10, 10, 0.1, settings);

        assert_eq!(cloth.particles.len(), 100);
        assert!(cloth.constraints.len() > 0);
        assert_eq!(cloth.triangles.len(), 9 * 9 * 2);
    }

    #[test]
    fn test_hanging_cloth() {
        let settings = ClothSettings::default();
        let cloth = create_hanging_cloth(2.0, 2.0, 5, 5, 0.1, settings);

        // Top row should be pinned
        for x in 0..5 {
            assert!(cloth.particles[x].pinned);
        }
        // Other particles should not be pinned
        assert!(!cloth.particles[5].pinned);
    }

    #[test]
    fn test_verlet_integration() {
        let settings = ClothSettings {
            solver_iterations: 1,
            ..Default::default()
        };
        let mut cloth = create_cloth(1.0, 1.0, 2, 2, 1.0, settings);

        let initial_y = cloth.particles[0].position.y;
        cloth.step(1.0 / 60.0);

        // Particles should fall under gravity
        assert!(cloth.particles[0].position.y < initial_y);
    }

    #[test]
    fn test_constraint_maintains_distance() {
        let settings = ClothSettings {
            solver_iterations: 50,
            gravity: Vec3::ZERO,
            ..Default::default()
        };
        let mut cloth = create_cloth(1.0, 1.0, 2, 2, 1.0, settings);

        // Move a particle far away
        cloth.particles[0].position = Vec3::new(100.0, 0.0, 0.0);

        // After many solver iterations, constraints should pull it back
        cloth.step(1.0 / 60.0);

        // The distance between connected particles should be closer to rest length
        let rest = cloth.constraints[0].rest_length;
        let p0 = cloth.particles[cloth.constraints[0].particle_a].position;
        let p1 = cloth.particles[cloth.constraints[0].particle_b].position;
        let dist = (p1 - p0).length();
        // Should be much closer to rest length than 100
        assert!(dist < 50.0);
    }

    #[test]
    fn test_pin_constraint() {
        let settings = ClothSettings::default();
        let mut cloth = create_cloth(1.0, 1.0, 3, 3, 0.1, settings);

        let pin_pos = Vec3::new(5.0, 5.0, 5.0);
        cloth.pin_particle(0);
        cloth.pin_constraints[0].target_position = pin_pos;

        cloth.step(1.0 / 60.0);

        assert!((cloth.particles[0].position - pin_pos).length() < 1e-4);
    }

    #[test]
    fn test_plane_collision() {
        let settings = ClothSettings {
            solver_iterations: 5,
            ..Default::default()
        };
        let mut cloth = create_cloth(1.0, 1.0, 3, 3, 0.1, settings);

        // Add a ground plane at y = -1
        cloth.add_plane_collider(ClothPlaneCollider {
            point: Vec3::new(0.0, -1.0, 0.0),
            normal: Vec3::Y,
            friction: 0.5,
        });

        // Simulate falling
        for _ in 0..120 {
            cloth.step(1.0 / 60.0);
        }

        // All particles should be above or at the plane
        for p in &cloth.particles {
            assert!(
                p.position.y >= -1.0 - SELF_COLLISION_RADIUS - 0.1,
                "Particle below plane: y = {}",
                p.position.y
            );
        }
    }

    #[test]
    fn test_sphere_collision() {
        let settings = ClothSettings {
            solver_iterations: 5,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            ..Default::default()
        };
        let mut cloth = create_cloth(2.0, 2.0, 5, 5, 0.1, settings);

        // Place a sphere at the origin
        cloth.add_sphere_collider(ClothSphereCollider {
            center: Vec3::new(0.0, -0.5, 0.0),
            radius: 0.5,
            friction: 0.3,
        });

        for _ in 0..60 {
            cloth.step(1.0 / 60.0);
        }

        // Particles near the center should be pushed away from sphere center
        for p in &cloth.particles {
            let dist = (p.position - Vec3::new(0.0, -0.5, 0.0)).length();
            // All particles should be outside or at the sphere surface
            assert!(
                dist >= 0.5 - 0.15,
                "Particle inside sphere: dist = {}",
                dist
            );
        }
    }

    #[test]
    fn test_tearing() {
        let settings = ClothSettings {
            solver_iterations: 1,
            gravity: Vec3::ZERO,
            tearing_enabled: true,
            max_stretch_ratio: 1.5,
            ..Default::default()
        };
        let mut cloth = create_cloth(1.0, 1.0, 3, 3, 0.1, settings);

        // Pin one particle and move another very far
        cloth.pin_particle(0);
        cloth.particles[1].position = Vec3::new(100.0, 0.0, 0.0);
        cloth.particles[1].prev_position = cloth.particles[1].position;

        // Directly check tearing (the step method checks after constraint solving)
        cloth.check_tearing();

        // At least one constraint should be torn
        assert!(cloth.constraints.iter().any(|c| c.torn));
    }

    #[test]
    fn test_wind_force() {
        let settings = ClothSettings {
            solver_iterations: 5,
            gravity: Vec3::ZERO,
            ..Default::default()
        };
        // Create cloth in XZ plane; wind along Y will push it since the face normals
        // are along Y initially
        let mut cloth = create_cloth(1.0, 1.0, 3, 3, 0.1, settings);
        cloth.set_wind(Vec3::new(0.0, 10.0, 0.0), 0.0);

        let initial_positions: Vec<Vec3> = cloth.particles.iter().map(|p| p.position).collect();

        for _ in 0..30 {
            cloth.step(1.0 / 60.0);
        }

        // Particles should have moved (pushed by wind in Y direction)
        let moved = cloth
            .particles
            .iter()
            .enumerate()
            .any(|(i, p)| (p.position - initial_positions[i]).length() > 1e-6);
        assert!(moved, "Wind should move particles");
    }

    #[test]
    fn test_cloth_component() {
        let component = ClothComponent::hanging(2.0, 2.0, 5);
        assert!(component.active);
        assert_eq!(component.mesh.particles.len(), 25);
    }

    #[test]
    fn test_cloth_system_update() {
        let mut system = ClothSystem::new();
        let mut cloths = vec![ClothComponent::hanging(1.0, 1.0, 3)];

        system.update(1.0 / 60.0, &mut cloths);

        // Non-pinned particles should have moved
        assert!(cloths[0].mesh.particles[4].position.y < 0.0);
    }

    #[test]
    fn test_cloth_indices_output() {
        let settings = ClothSettings::default();
        let cloth = create_cloth(1.0, 1.0, 3, 3, 0.1, settings);
        let indices = cloth.indices();
        assert_eq!(indices.len(), 2 * 2 * 2 * 3); // 4 quads * 2 tris * 3 indices
    }

    #[test]
    fn test_normals_computed() {
        let settings = ClothSettings::default();
        let mut cloth = create_cloth(1.0, 1.0, 3, 3, 0.1, settings);
        cloth.compute_normals();

        // Interior particles should have non-zero normals
        for p in &cloth.particles {
            assert!(p.normal.length() > 0.9);
        }
    }
}
