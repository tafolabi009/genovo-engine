// engine/physics/src/soft_body_v2.rs
//
// Enhanced soft body physics: position-based dynamics (PBD/XPBD), shape matching,
// volume preservation, attachment constraints to rigid bodies, cutting/tearing
// simulation, and GPU simulation data preparation.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn dot(self, r: Self) -> f32 { self.x*r.x + self.y*r.y + self.z*r.z }
    pub fn cross(self, r: Self) -> Self { Self { x: self.y*r.z-self.z*r.y, y: self.z*r.x-self.x*r.z, z: self.x*r.y-self.y*r.x } }
    pub fn length(self) -> f32 { self.dot(self).sqrt() }
    pub fn length_sq(self) -> f32 { self.dot(self) }
    pub fn normalize(self) -> Self { let l=self.length(); if l<1e-12{Self::ZERO}else{Self{x:self.x/l,y:self.y/l,z:self.z/l}} }
    pub fn scale(self, s: f32) -> Self { Self { x:self.x*s, y:self.y*s, z:self.z*s } }
    pub fn add(self, r: Self) -> Self { Self { x:self.x+r.x, y:self.y+r.y, z:self.z+r.z } }
    pub fn sub(self, r: Self) -> Self { Self { x:self.x-r.x, y:self.y-r.y, z:self.z-r.z } }
    pub fn neg(self) -> Self { Self { x:-self.x, y:-self.y, z:-self.z } }
    pub fn lerp(self, r: Self, t: f32) -> Self { self.add(r.sub(self).scale(t)) }
    pub fn distance(self, r: Self) -> f32 { self.sub(r).length() }
}

/// 3x3 matrix for shape matching.
#[derive(Debug, Clone, Copy)]
pub struct Mat3 { pub data: [f32; 9] }
impl Mat3 {
    pub const IDENTITY: Self = Self { data: [1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0] };
    pub fn from_outer(a: Vec3, b: Vec3) -> Self {
        Self { data: [a.x*b.x, a.x*b.y, a.x*b.z, a.y*b.x, a.y*b.y, a.y*b.z, a.z*b.x, a.z*b.y, a.z*b.z] }
    }
    pub fn add(self, r: Self) -> Self {
        let mut d = [0.0f32; 9];
        for i in 0..9 { d[i] = self.data[i] + r.data[i]; }
        Self { data: d }
    }
    pub fn scale(self, s: f32) -> Self {
        let mut d = self.data;
        for v in &mut d { *v *= s; }
        Self { data: d }
    }
    pub fn mul_vec(self, v: Vec3) -> Vec3 {
        let d = &self.data;
        Vec3::new(d[0]*v.x+d[1]*v.y+d[2]*v.z, d[3]*v.x+d[4]*v.y+d[5]*v.z, d[6]*v.x+d[7]*v.y+d[8]*v.z)
    }
    pub fn determinant(&self) -> f32 {
        let d = &self.data;
        d[0]*(d[4]*d[8]-d[5]*d[7]) - d[1]*(d[3]*d[8]-d[5]*d[6]) + d[2]*(d[3]*d[7]-d[4]*d[6])
    }
    pub fn transpose(self) -> Self {
        let d = &self.data;
        Self { data: [d[0],d[3],d[6], d[1],d[4],d[7], d[2],d[5],d[8]] }
    }
    /// Extract rotation via polar decomposition (iterative method).
    pub fn extract_rotation(self) -> Self {
        let mut r = self;
        for _ in 0..10 {
            let rt = r.transpose();
            let det = r.determinant();
            if det.abs() < 1e-12 { return Mat3::IDENTITY; }
            // Average R with (R^-T) = (R^T)^-1, using the approximation R_new = 0.5 * (R + R^-T)
            // For well-conditioned R, R^-T ≈ R (if R is already orthogonal)
            r = r.add(cofactor_transpose(&r).scale(1.0 / det)).scale(0.5);
        }
        r
    }
}

fn cofactor_transpose(m: &Mat3) -> Mat3 {
    let d = &m.data;
    Mat3 { data: [
        d[4]*d[8]-d[5]*d[7], -(d[3]*d[8]-d[5]*d[6]), d[3]*d[7]-d[4]*d[6],
        -(d[1]*d[8]-d[2]*d[7]), d[0]*d[8]-d[2]*d[6], -(d[0]*d[7]-d[1]*d[6]),
        d[1]*d[5]-d[2]*d[4], -(d[0]*d[5]-d[2]*d[3]), d[0]*d[4]-d[1]*d[3],
    ]}
}

// ---------------------------------------------------------------------------
// Particle
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SoftParticle {
    pub position: Vec3,
    pub predicted: Vec3,
    pub velocity: Vec3,
    pub inv_mass: f32,
    pub rest_position: Vec3, // for shape matching
    pub pinned: bool,
    pub attachment: Option<AttachmentInfo>,
}

#[derive(Debug, Clone)]
pub struct AttachmentInfo {
    pub rigid_body_id: u32,
    pub local_offset: Vec3,
    pub stiffness: f32,
}

impl SoftParticle {
    pub fn new(position: Vec3, mass: f32) -> Self {
        Self {
            position,
            predicted: position,
            velocity: Vec3::ZERO,
            inv_mass: if mass > 0.0 { 1.0 / mass } else { 0.0 },
            rest_position: position,
            pinned: false,
            attachment: None,
        }
    }

    pub fn mass(&self) -> f32 {
        if self.inv_mass > 0.0 { 1.0 / self.inv_mass } else { f32::MAX }
    }
}

// ---------------------------------------------------------------------------
// Constraints
// ---------------------------------------------------------------------------

/// Distance constraint between two particles.
#[derive(Debug, Clone)]
pub struct DistanceConstraint {
    pub particle_a: usize,
    pub particle_b: usize,
    pub rest_length: f32,
    pub compliance: f32, // inverse stiffness (XPBD)
    pub max_strain: f32, // maximum stretch ratio before tearing
    pub lambda: f32,     // Lagrange multiplier (XPBD)
    pub broken: bool,
}

impl DistanceConstraint {
    pub fn new(a: usize, b: usize, rest_length: f32, compliance: f32) -> Self {
        Self { particle_a: a, particle_b: b, rest_length, compliance, max_strain: 2.0, lambda: 0.0, broken: false }
    }

    /// Solve this constraint using XPBD.
    pub fn solve(&mut self, particles: &mut [SoftParticle], dt: f32) {
        if self.broken { return; }
        let pa = &particles[self.particle_a];
        let pb = &particles[self.particle_b];
        if pa.pinned && pb.pinned { return; }

        let diff = pb.predicted.sub(pa.predicted);
        let dist = diff.length();
        if dist < 1e-9 { return; }

        // Check for tearing
        if dist / self.rest_length > self.max_strain {
            self.broken = true;
            return;
        }

        let c = dist - self.rest_length;
        let gradient = diff.scale(1.0 / dist);

        let w_a = pa.inv_mass;
        let w_b = pb.inv_mass;
        let w_sum = w_a + w_b;
        if w_sum < 1e-12 { return; }

        // XPBD: compliance-aware correction
        let alpha = self.compliance / (dt * dt);
        let delta_lambda = (-c - alpha * self.lambda) / (w_sum + alpha);
        self.lambda += delta_lambda;

        let correction = gradient.scale(delta_lambda);

        if !particles[self.particle_a].pinned {
            particles[self.particle_a].predicted = particles[self.particle_a].predicted.add(correction.scale(w_a));
        }
        if !particles[self.particle_b].pinned {
            particles[self.particle_b].predicted = particles[self.particle_b].predicted.sub(correction.scale(w_b));
        }
    }
}

/// Volume preservation constraint for a tetrahedron.
#[derive(Debug, Clone)]
pub struct VolumeConstraint {
    pub indices: [usize; 4],
    pub rest_volume: f32,
    pub compliance: f32,
    pub lambda: f32,
}

impl VolumeConstraint {
    pub fn new(indices: [usize; 4], particles: &[SoftParticle], compliance: f32) -> Self {
        let rest_volume = compute_tet_volume(
            particles[indices[0]].position,
            particles[indices[1]].position,
            particles[indices[2]].position,
            particles[indices[3]].position,
        );
        Self { indices, rest_volume, compliance, lambda: 0.0 }
    }

    pub fn solve(&mut self, particles: &mut [SoftParticle], dt: f32) {
        let p0 = particles[self.indices[0]].predicted;
        let p1 = particles[self.indices[1]].predicted;
        let p2 = particles[self.indices[2]].predicted;
        let p3 = particles[self.indices[3]].predicted;

        let current_volume = compute_tet_volume(p0, p1, p2, p3);
        let c = current_volume - self.rest_volume;
        if c.abs() < 1e-12 { return; }

        // Gradients of volume w.r.t. each vertex
        let g0 = p1.sub(p3).cross(p2.sub(p3)).scale(1.0 / 6.0);
        let g1 = p2.sub(p3).cross(p0.sub(p3)).scale(1.0 / 6.0);
        let g2 = p0.sub(p3).cross(p1.sub(p3)).scale(1.0 / 6.0);
        let g3 = g0.add(g1).add(g2).neg();

        let grads = [g0, g1, g2, g3];
        let mut w_sum = 0.0_f32;
        for i in 0..4 {
            w_sum += particles[self.indices[i]].inv_mass * grads[i].length_sq();
        }
        if w_sum < 1e-12 { return; }

        let alpha = self.compliance / (dt * dt);
        let delta_lambda = (-c - alpha * self.lambda) / (w_sum + alpha);
        self.lambda += delta_lambda;

        for i in 0..4 {
            if !particles[self.indices[i]].pinned {
                let correction = grads[i].scale(delta_lambda * particles[self.indices[i]].inv_mass);
                particles[self.indices[i]].predicted = particles[self.indices[i]].predicted.add(correction);
            }
        }
    }
}

fn compute_tet_volume(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> f32 {
    let d1 = p1.sub(p0);
    let d2 = p2.sub(p0);
    let d3 = p3.sub(p0);
    d1.cross(d2).dot(d3) / 6.0
}

/// Bending constraint (dihedral angle between two triangles sharing an edge).
#[derive(Debug, Clone)]
pub struct BendingConstraint {
    pub particles: [usize; 4], // p0, p1 = shared edge; p2, p3 = opposite vertices
    pub rest_angle: f32,
    pub compliance: f32,
    pub lambda: f32,
}

impl BendingConstraint {
    pub fn new(particles: [usize; 4], positions: &[SoftParticle], compliance: f32) -> Self {
        let p0 = positions[particles[0]].position;
        let p1 = positions[particles[1]].position;
        let p2 = positions[particles[2]].position;
        let p3 = positions[particles[3]].position;
        let rest_angle = compute_dihedral_angle(p0, p1, p2, p3);
        Self { particles, rest_angle, compliance, lambda: 0.0 }
    }

    pub fn solve(&mut self, particles: &mut [SoftParticle], dt: f32) {
        let p0 = particles[self.particles[0]].predicted;
        let p1 = particles[self.particles[1]].predicted;
        let p2 = particles[self.particles[2]].predicted;
        let p3 = particles[self.particles[3]].predicted;

        let current_angle = compute_dihedral_angle(p0, p1, p2, p3);
        let c = current_angle - self.rest_angle;
        if c.abs() < 1e-8 { return; }

        // Approximate gradients
        let edge = p1.sub(p0);
        let edge_len = edge.length();
        if edge_len < 1e-9 { return; }
        let edge_dir = edge.scale(1.0 / edge_len);

        let n1 = edge.cross(p2.sub(p0)).normalize();
        let n2 = edge.cross(p3.sub(p0)).normalize();

        let h2 = p2.sub(p0).sub(edge_dir.scale(p2.sub(p0).dot(edge_dir)));
        let h3 = p3.sub(p0).sub(edge_dir.scale(p3.sub(p0).dot(edge_dir)));
        let h2_len = h2.length();
        let h3_len = h3.length();
        if h2_len < 1e-9 || h3_len < 1e-9 { return; }

        let g2 = n1.scale(1.0 / h2_len);
        let g3 = n2.scale(-1.0 / h3_len);

        let e_dot_p2 = edge_dir.dot(p2.sub(p0));
        let e_dot_p3 = edge_dir.dot(p3.sub(p0));
        let t2 = e_dot_p2 / edge_len;
        let t3 = e_dot_p3 / edge_len;

        let g0 = g2.scale(-(1.0 - t2)).add(g3.scale(-(1.0 - t3)));
        let g1 = g2.scale(-t2).add(g3.scale(-t3));

        let grads = [g0, g1, g2, g3];
        let mut w_sum = 0.0_f32;
        for i in 0..4 { w_sum += particles[self.particles[i]].inv_mass * grads[i].length_sq(); }
        if w_sum < 1e-12 { return; }

        let alpha = self.compliance / (dt * dt);
        let delta_lambda = (-c - alpha * self.lambda) / (w_sum + alpha);
        self.lambda += delta_lambda;

        for i in 0..4 {
            if !particles[self.particles[i]].pinned {
                let corr = grads[i].scale(delta_lambda * particles[self.particles[i]].inv_mass);
                particles[self.particles[i]].predicted = particles[self.particles[i]].predicted.add(corr);
            }
        }
    }
}

fn compute_dihedral_angle(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> f32 {
    let edge = p1.sub(p0);
    let n1 = edge.cross(p2.sub(p0)).normalize();
    let n2 = edge.cross(p3.sub(p0)).normalize();
    let cos_angle = n1.dot(n2).clamp(-1.0, 1.0);
    cos_angle.acos()
}

// ---------------------------------------------------------------------------
// Shape matching
// ---------------------------------------------------------------------------

/// Shape matching for rigid/deformable body approximation (Mueller et al.).
pub struct ShapeMatchingRegion {
    pub particle_indices: Vec<usize>,
    pub rest_center: Vec3,
    pub rest_positions: Vec<Vec3>, // relative to rest_center
    pub masses: Vec<f32>,
    pub total_mass: f32,
    pub stiffness: f32, // 0..1, how much to enforce shape
    pub allow_stretch: bool,
}

impl ShapeMatchingRegion {
    pub fn new(indices: Vec<usize>, particles: &[SoftParticle], stiffness: f32) -> Self {
        let total_mass: f32 = indices.iter().map(|&i| particles[i].mass()).sum();
        let rest_center = if total_mass > 0.0 {
            let sum: Vec3 = indices.iter().map(|&i| particles[i].rest_position.scale(particles[i].mass())).fold(Vec3::ZERO, Vec3::add);
            sum.scale(1.0 / total_mass)
        } else { Vec3::ZERO };

        let rest_positions: Vec<Vec3> = indices.iter().map(|&i| particles[i].rest_position.sub(rest_center)).collect();
        let masses: Vec<f32> = indices.iter().map(|&i| particles[i].mass()).collect();

        Self { particle_indices: indices, rest_center, rest_positions, masses, total_mass, stiffness, allow_stretch: false }
    }

    /// Apply shape matching constraint.
    pub fn apply(&self, particles: &mut [SoftParticle]) {
        if self.particle_indices.is_empty() || self.total_mass < 1e-12 { return; }

        // Compute current center of mass
        let mut com = Vec3::ZERO;
        for (i, &pi) in self.particle_indices.iter().enumerate() {
            com = com.add(particles[pi].predicted.scale(self.masses[i]));
        }
        com = com.scale(1.0 / self.total_mass);

        // Compute Apq matrix (deformation gradient)
        let mut apq = Mat3 { data: [0.0; 9] };
        for (i, &pi) in self.particle_indices.iter().enumerate() {
            let q = self.rest_positions[i];
            let p = particles[pi].predicted.sub(com);
            apq = apq.add(Mat3::from_outer(p, q).scale(self.masses[i]));
        }

        // Extract rotation
        let rotation = apq.extract_rotation();

        // Goal positions
        for (i, &pi) in self.particle_indices.iter().enumerate() {
            if particles[pi].pinned { continue; }
            let goal = com.add(rotation.mul_vec(self.rest_positions[i]));
            particles[pi].predicted = particles[pi].predicted.lerp(goal, self.stiffness);
        }
    }
}

// ---------------------------------------------------------------------------
// Soft body simulation
// ---------------------------------------------------------------------------

/// Configuration for soft body simulation.
#[derive(Debug, Clone)]
pub struct SoftBodyConfig {
    pub solver_iterations: u32,
    pub sub_steps: u32,
    pub gravity: Vec3,
    pub damping: f32,
    pub distance_compliance: f32,
    pub volume_compliance: f32,
    pub bending_compliance: f32,
    pub collision_margin: f32,
    pub friction: f32,
    pub enable_self_collision: bool,
    pub self_collision_radius: f32,
    pub air_drag: f32,
    pub wind: Vec3,
}

impl Default for SoftBodyConfig {
    fn default() -> Self {
        Self {
            solver_iterations: 10,
            sub_steps: 4,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.99,
            distance_compliance: 0.0,
            volume_compliance: 0.0001,
            bending_compliance: 0.01,
            collision_margin: 0.01,
            friction: 0.3,
            enable_self_collision: false,
            self_collision_radius: 0.05,
            air_drag: 0.01,
            wind: Vec3::ZERO,
        }
    }
}

/// A soft body simulation instance.
pub struct SoftBodyV2 {
    pub particles: Vec<SoftParticle>,
    pub distance_constraints: Vec<DistanceConstraint>,
    pub volume_constraints: Vec<VolumeConstraint>,
    pub bending_constraints: Vec<BendingConstraint>,
    pub shape_regions: Vec<ShapeMatchingRegion>,
    pub config: SoftBodyConfig,
    pub surface_triangles: Vec<[usize; 3]>,
    pub collision_spheres: Vec<CollisionSphere>,
    pub stats: SoftBodyStats,
}

#[derive(Debug, Clone, Default)]
pub struct SoftBodyStats {
    pub particle_count: u32,
    pub constraint_count: u32,
    pub broken_constraints: u32,
    pub max_velocity: f32,
    pub avg_strain: f32,
    pub volume_ratio: f32,
}

#[derive(Debug, Clone)]
pub struct CollisionSphere {
    pub center: Vec3,
    pub radius: f32,
    pub is_static: bool,
}

impl SoftBodyV2 {
    pub fn new(config: SoftBodyConfig) -> Self {
        Self {
            particles: Vec::new(),
            distance_constraints: Vec::new(),
            volume_constraints: Vec::new(),
            bending_constraints: Vec::new(),
            shape_regions: Vec::new(),
            config,
            surface_triangles: Vec::new(),
            collision_spheres: Vec::new(),
            stats: SoftBodyStats::default(),
        }
    }

    /// Create a soft body from a tetrahedral mesh.
    pub fn from_tet_mesh(vertices: &[Vec3], tets: &[[usize; 4]], mass: f32, config: SoftBodyConfig) -> Self {
        let mut body = Self::new(config);
        let particle_mass = mass / vertices.len() as f32;

        // Create particles
        for &v in vertices {
            body.particles.push(SoftParticle::new(v, particle_mass));
        }

        // Create distance constraints for all tet edges
        let mut edge_set: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        for tet in tets {
            let edges = [(tet[0],tet[1]),(tet[0],tet[2]),(tet[0],tet[3]),(tet[1],tet[2]),(tet[1],tet[3]),(tet[2],tet[3])];
            for (a, b) in edges {
                let key = if a < b { (a, b) } else { (b, a) };
                if edge_set.insert(key) {
                    let rest_len = vertices[a].distance(vertices[b]);
                    body.distance_constraints.push(DistanceConstraint::new(a, b, rest_len, body.config.distance_compliance));
                }
            }
        }

        // Create volume constraints
        for tet in tets {
            body.volume_constraints.push(VolumeConstraint::new(*tet, &body.particles, body.config.volume_compliance));
        }

        body
    }

    /// Create a soft body from a surface mesh (cloth-like).
    pub fn from_surface_mesh(vertices: &[Vec3], triangles: &[[usize; 3]], mass: f32, config: SoftBodyConfig) -> Self {
        let mut body = Self::new(config);
        let particle_mass = mass / vertices.len() as f32;

        for &v in vertices {
            body.particles.push(SoftParticle::new(v, particle_mass));
        }

        // Distance constraints from triangle edges
        let mut edge_set: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        for tri in triangles {
            let edges = [(tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])];
            for (a, b) in edges {
                let key = if a < b { (a, b) } else { (b, a) };
                if edge_set.insert(key) {
                    let rest_len = vertices[a].distance(vertices[b]);
                    body.distance_constraints.push(DistanceConstraint::new(a, b, rest_len, body.config.distance_compliance));
                }
            }
        }

        body.surface_triangles = triangles.to_vec();
        body
    }

    /// Step the simulation.
    pub fn step(&mut self, dt: f32) {
        let sub_dt = dt / self.config.sub_steps as f32;

        for _ in 0..self.config.sub_steps {
            self.predict(sub_dt);
            self.reset_lambdas();
            for _ in 0..self.config.solver_iterations {
                self.solve_constraints(sub_dt);
            }
            self.apply_shape_matching();
            self.handle_collisions();
            self.update_velocities(sub_dt);
        }

        self.update_stats();
    }

    fn predict(&mut self, dt: f32) {
        for particle in &mut self.particles {
            if particle.pinned { particle.predicted = particle.position; continue; }
            // Apply external forces
            particle.velocity = particle.velocity.add(self.config.gravity.scale(dt));
            // Air drag
            let drag_force = particle.velocity.scale(-self.config.air_drag);
            particle.velocity = particle.velocity.add(drag_force.scale(dt));
            // Wind
            particle.velocity = particle.velocity.add(self.config.wind.scale(dt * self.config.air_drag));
            // Damping
            particle.velocity = particle.velocity.scale(self.config.damping);
            // Predict position
            particle.predicted = particle.position.add(particle.velocity.scale(dt));
            // Handle attachments
            if let Some(ref attach) = particle.attachment {
                let target = attach.local_offset; // In real engine, would transform by rigid body
                particle.predicted = particle.predicted.lerp(target, attach.stiffness);
            }
        }
    }

    fn reset_lambdas(&mut self) {
        for c in &mut self.distance_constraints { c.lambda = 0.0; }
        for c in &mut self.volume_constraints { c.lambda = 0.0; }
        for c in &mut self.bending_constraints { c.lambda = 0.0; }
    }

    fn solve_constraints(&mut self, dt: f32) {
        // Distance constraints
        for i in 0..self.distance_constraints.len() {
            let mut constraint = self.distance_constraints[i].clone();
            constraint.solve(&mut self.particles, dt);
            self.distance_constraints[i] = constraint;
        }

        // Volume constraints
        for i in 0..self.volume_constraints.len() {
            let mut constraint = self.volume_constraints[i].clone();
            constraint.solve(&mut self.particles, dt);
            self.volume_constraints[i] = constraint;
        }

        // Bending constraints
        for i in 0..self.bending_constraints.len() {
            let mut constraint = self.bending_constraints[i].clone();
            constraint.solve(&mut self.particles, dt);
            self.bending_constraints[i] = constraint;
        }
    }

    fn apply_shape_matching(&mut self) {
        for region in &self.shape_regions {
            region.apply(&mut self.particles);
        }
    }

    fn handle_collisions(&mut self) {
        let margin = self.config.collision_margin;
        let friction = self.config.friction;

        for particle in &mut self.particles {
            if particle.pinned { continue; }

            // Ground plane collision
            if particle.predicted.y < margin {
                particle.predicted.y = margin;
                // Friction
                let vel_tangent = Vec3::new(particle.velocity.x, 0.0, particle.velocity.z);
                let correction = vel_tangent.scale(-friction);
                particle.predicted = particle.predicted.add(correction.scale(0.016)); // approximate dt
            }

            // Sphere collisions
            for sphere in &self.collision_spheres {
                let diff = particle.predicted.sub(sphere.center);
                let dist = diff.length();
                let min_dist = sphere.radius + margin;
                if dist < min_dist && dist > 1e-9 {
                    let normal = diff.scale(1.0 / dist);
                    particle.predicted = sphere.center.add(normal.scale(min_dist));
                }
            }
        }

        // Self-collision (spatial hashing)
        if self.config.enable_self_collision {
            self.solve_self_collision();
        }
    }

    fn solve_self_collision(&mut self) {
        let radius = self.config.self_collision_radius;
        let n = self.particles.len();
        // Simple O(n^2) for small particle counts
        if n > 1000 { return; } // Would use spatial hashing for large counts

        for i in 0..n {
            if self.particles[i].pinned { continue; }
            for j in (i+1)..n {
                if self.particles[j].pinned { continue; }
                let diff = self.particles[j].predicted.sub(self.particles[i].predicted);
                let dist = diff.length();
                let min_dist = radius * 2.0;
                if dist < min_dist && dist > 1e-9 {
                    let normal = diff.scale(1.0 / dist);
                    let overlap = (min_dist - dist) * 0.5;
                    self.particles[i].predicted = self.particles[i].predicted.sub(normal.scale(overlap));
                    self.particles[j].predicted = self.particles[j].predicted.add(normal.scale(overlap));
                }
            }
        }
    }

    fn update_velocities(&mut self, dt: f32) {
        let inv_dt = 1.0 / dt;
        for particle in &mut self.particles {
            if particle.pinned { continue; }
            particle.velocity = particle.predicted.sub(particle.position).scale(inv_dt);
            particle.position = particle.predicted;
        }
    }

    fn update_stats(&mut self) {
        self.stats.particle_count = self.particles.len() as u32;
        self.stats.constraint_count = (self.distance_constraints.len() + self.volume_constraints.len() + self.bending_constraints.len()) as u32;
        self.stats.broken_constraints = self.distance_constraints.iter().filter(|c| c.broken).count() as u32;
        self.stats.max_velocity = self.particles.iter().map(|p| p.velocity.length()).fold(0.0_f32, f32::max);

        if !self.distance_constraints.is_empty() {
            let total_strain: f32 = self.distance_constraints.iter()
                .filter(|c| !c.broken)
                .map(|c| {
                    let dist = self.particles[c.particle_a].position.distance(self.particles[c.particle_b].position);
                    (dist / c.rest_length - 1.0).abs()
                })
                .sum();
            self.stats.avg_strain = total_strain / self.distance_constraints.len() as f32;
        }
    }

    /// Pin a particle (make it immovable).
    pub fn pin_particle(&mut self, index: usize) {
        if index < self.particles.len() {
            self.particles[index].pinned = true;
            self.particles[index].inv_mass = 0.0;
        }
    }

    /// Unpin a particle.
    pub fn unpin_particle(&mut self, index: usize, mass: f32) {
        if index < self.particles.len() {
            self.particles[index].pinned = false;
            self.particles[index].inv_mass = 1.0 / mass;
        }
    }

    /// Apply a force to a particle.
    pub fn apply_force(&mut self, index: usize, force: Vec3) {
        if index < self.particles.len() && !self.particles[index].pinned {
            let impulse = force.scale(self.particles[index].inv_mass);
            self.particles[index].velocity = self.particles[index].velocity.add(impulse);
        }
    }

    /// Cut/tear along a plane.
    pub fn cut(&mut self, plane_point: Vec3, plane_normal: Vec3, cut_width: f32) {
        for constraint in &mut self.distance_constraints {
            let pa = self.particles[constraint.particle_a].position;
            let pb = self.particles[constraint.particle_b].position;
            let mid = pa.lerp(pb, 0.5);

            let dist_to_plane = mid.sub(plane_point).dot(plane_normal).abs();
            if dist_to_plane < cut_width {
                constraint.broken = true;
            }
        }
    }

    /// Prepare data for GPU simulation.
    pub fn prepare_gpu_data(&self) -> GpuSoftBodyData {
        let positions: Vec<[f32; 4]> = self.particles.iter()
            .map(|p| [p.position.x, p.position.y, p.position.z, p.inv_mass])
            .collect();
        let velocities: Vec<[f32; 4]> = self.particles.iter()
            .map(|p| [p.velocity.x, p.velocity.y, p.velocity.z, if p.pinned { 0.0 } else { 1.0 }])
            .collect();
        let constraints: Vec<[u32; 4]> = self.distance_constraints.iter()
            .filter(|c| !c.broken)
            .map(|c| [c.particle_a as u32, c.particle_b as u32, c.rest_length.to_bits(), c.compliance.to_bits()])
            .collect();

        GpuSoftBodyData { positions, velocities, constraints }
    }

    /// Compute surface normals for rendering.
    pub fn compute_normals(&self) -> Vec<Vec3> {
        let mut normals = vec![Vec3::ZERO; self.particles.len()];
        for tri in &self.surface_triangles {
            let v0 = self.particles[tri[0]].position;
            let v1 = self.particles[tri[1]].position;
            let v2 = self.particles[tri[2]].position;
            let n = v1.sub(v0).cross(v2.sub(v0));
            normals[tri[0]] = normals[tri[0]].add(n);
            normals[tri[1]] = normals[tri[1]].add(n);
            normals[tri[2]] = normals[tri[2]].add(n);
        }
        for n in &mut normals { *n = n.normalize(); }
        normals
    }
}

#[derive(Debug, Clone)]
pub struct GpuSoftBodyData {
    pub positions: Vec<[f32; 4]>,
    pub velocities: Vec<[f32; 4]>,
    pub constraints: Vec<[u32; 4]>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_creation() {
        let p = SoftParticle::new(Vec3::ZERO, 1.0);
        assert!((p.inv_mass - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_constraint() {
        let mut particles = vec![
            SoftParticle::new(Vec3::new(0.0, 0.0, 0.0), 1.0),
            SoftParticle::new(Vec3::new(2.0, 0.0, 0.0), 1.0),
        ];
        particles[0].predicted = particles[0].position;
        particles[1].predicted = particles[1].position;

        let mut constraint = DistanceConstraint::new(0, 1, 1.0, 0.0);
        constraint.solve(&mut particles, 0.016);

        let dist = particles[0].predicted.distance(particles[1].predicted);
        assert!(dist < 2.0); // should have pulled closer
    }

    #[test]
    fn test_tet_volume() {
        let v = compute_tet_volume(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        assert!((v - 1.0/6.0).abs() < 1e-5);
    }

    #[test]
    fn test_shape_matching() {
        let particles = vec![
            SoftParticle::new(Vec3::new(0.0, 0.0, 0.0), 1.0),
            SoftParticle::new(Vec3::new(1.0, 0.0, 0.0), 1.0),
            SoftParticle::new(Vec3::new(0.0, 1.0, 0.0), 1.0),
        ];
        let region = ShapeMatchingRegion::new(vec![0, 1, 2], &particles, 0.5);
        assert!(region.total_mass > 0.0);
    }

    #[test]
    fn test_soft_body_step() {
        let verts = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.5, 1.0, 1.0),
        ];
        let tris = vec![[0, 1, 2]];
        let mut body = SoftBodyV2::from_surface_mesh(&verts, &tris, 3.0, SoftBodyConfig::default());
        body.step(0.016);
        // Particles should have moved down due to gravity
        assert!(body.particles[0].position.y < 1.0);
    }

    #[test]
    fn test_cut() {
        let verts = vec![
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        ];
        let tris = vec![[0, 1, 2]];
        let mut body = SoftBodyV2::from_surface_mesh(&verts, &tris, 3.0, SoftBodyConfig::default());
        body.cut(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.5);
        let broken = body.distance_constraints.iter().filter(|c| c.broken).count();
        assert!(broken > 0);
    }
}
