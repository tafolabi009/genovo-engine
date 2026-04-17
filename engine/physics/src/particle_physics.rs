//! Physics particle system for position-based dynamics simulation.
//!
//! Provides a complete particle physics pipeline independent from rendering:
//! - Mass-spring networks with configurable stiffness and damping
//! - Position-based dynamics (PBD) solver
//! - Distance constraints with strain limiting
//! - Bending constraints (dihedral angle preservation)
//! - Volume preservation constraints
//! - Collision response against spheres, planes, and boxes
//! - Self-collision detection via spatial hashing
//! - Particle groups, emitters, and lifetime management

use std::collections::HashMap;

use glam::{Mat3, Vec3};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default gravity vector (Y-down).
const DEFAULT_GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);
/// Default damping factor applied to velocities each step.
const DEFAULT_DAMPING: f32 = 0.01;
/// Default number of PBD solver iterations per substep.
const DEFAULT_SOLVER_ITERATIONS: usize = 10;
/// Default number of substeps per physics frame.
const DEFAULT_SUBSTEPS: usize = 4;
/// Small epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-7;
/// Default spatial hash cell size for self-collision.
const DEFAULT_CELL_SIZE: f32 = 0.25;
/// Default particle collision radius.
const DEFAULT_COLLISION_RADIUS: f32 = 0.05;
/// Default strain limit (max ratio of current / rest length).
const DEFAULT_STRAIN_LIMIT: f32 = 1.5;
/// Minimum inverse mass to consider a particle dynamic.
const MIN_INV_MASS: f32 = 1e-10;
/// Maximum number of particles per spatial hash cell for self-collision queries.
const MAX_PARTICLES_PER_CELL: usize = 32;

// ---------------------------------------------------------------------------
// Particle
// ---------------------------------------------------------------------------

/// Unique identifier for a particle within a simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParticleId(pub u32);

/// A single physics particle with mass, position, velocity, and constraints.
#[derive(Debug, Clone)]
pub struct Particle {
    /// Unique identifier.
    pub id: ParticleId,
    /// Current world-space position.
    pub position: Vec3,
    /// Predicted position (used during PBD solve).
    pub predicted: Vec3,
    /// Previous-frame position for Verlet integration fallback.
    pub prev_position: Vec3,
    /// Current velocity.
    pub velocity: Vec3,
    /// Accumulated external forces for the current step.
    pub force_accumulator: Vec3,
    /// Particle mass in kilograms.
    pub mass: f32,
    /// Inverse mass (0.0 for static/pinned particles).
    pub inv_mass: f32,
    /// Whether the particle is pinned (immovable).
    pub pinned: bool,
    /// Remaining lifetime in seconds (<0 means infinite).
    pub lifetime: f32,
    /// Collision radius for self-collision and environment collision.
    pub radius: f32,
    /// Phase identifier for collision filtering (same-phase particles can skip self-collision).
    pub phase: i32,
    /// User data tag for application-specific identification.
    pub user_tag: u64,
}

impl Particle {
    /// Create a new dynamic particle with the given position and mass.
    pub fn new(id: ParticleId, position: Vec3, mass: f32) -> Self {
        let inv_mass = if mass > EPSILON { 1.0 / mass } else { 0.0 };
        Self {
            id,
            position,
            predicted: position,
            prev_position: position,
            velocity: Vec3::ZERO,
            force_accumulator: Vec3::ZERO,
            mass,
            inv_mass,
            pinned: false,
            lifetime: -1.0,
            radius: DEFAULT_COLLISION_RADIUS,
            phase: 0,
            user_tag: 0,
        }
    }

    /// Create a static (pinned) particle at the given position.
    pub fn new_static(id: ParticleId, position: Vec3) -> Self {
        Self {
            id,
            position,
            predicted: position,
            prev_position: position,
            velocity: Vec3::ZERO,
            force_accumulator: Vec3::ZERO,
            mass: 0.0,
            inv_mass: 0.0,
            pinned: true,
            lifetime: -1.0,
            radius: DEFAULT_COLLISION_RADIUS,
            phase: 0,
            user_tag: 0,
        }
    }

    /// Pin this particle so it becomes immovable.
    pub fn pin(&mut self) {
        self.pinned = true;
        self.inv_mass = 0.0;
    }

    /// Unpin this particle, restoring its original inverse mass from its mass.
    pub fn unpin(&mut self) {
        self.pinned = false;
        if self.mass > EPSILON {
            self.inv_mass = 1.0 / self.mass;
        }
    }

    /// Set the mass and recompute inverse mass (respects pinned state).
    pub fn set_mass(&mut self, mass: f32) {
        self.mass = mass;
        if !self.pinned && mass > EPSILON {
            self.inv_mass = 1.0 / mass;
        }
    }

    /// Apply an external force to this particle for the current step.
    pub fn apply_force(&mut self, force: Vec3) {
        self.force_accumulator += force;
    }

    /// Apply an impulse directly to velocity.
    pub fn apply_impulse(&mut self, impulse: Vec3) {
        if self.inv_mass > MIN_INV_MASS {
            self.velocity += impulse * self.inv_mass;
        }
    }

    /// Returns true if this particle has expired (lifetime reached zero).
    pub fn is_expired(&self) -> bool {
        self.lifetime >= 0.0 && self.lifetime <= 0.0
    }

    /// Returns the kinetic energy of this particle.
    pub fn kinetic_energy(&self) -> f32 {
        0.5 * self.mass * self.velocity.length_squared()
    }
}

// ---------------------------------------------------------------------------
// Constraints
// ---------------------------------------------------------------------------

/// Unique handle for a constraint within the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstraintHandle(pub u32);

/// A distance constraint between two particles, maintaining a rest length.
#[derive(Debug, Clone)]
pub struct DistanceConstraint {
    /// Handle for this constraint.
    pub handle: ConstraintHandle,
    /// Index of the first particle.
    pub particle_a: usize,
    /// Index of the second particle.
    pub particle_b: usize,
    /// Rest length of the constraint.
    pub rest_length: f32,
    /// Stiffness in [0, 1] range. Higher means stiffer.
    pub stiffness: f32,
    /// Compliance (inverse stiffness for XPBD). 0 = infinitely stiff.
    pub compliance: f32,
    /// Whether this constraint is breakable.
    pub breakable: bool,
    /// Maximum stretch ratio before breaking (if breakable).
    pub max_stretch: f32,
    /// Whether this constraint has been broken.
    pub broken: bool,
}

impl DistanceConstraint {
    /// Create a new distance constraint with the given rest length and stiffness.
    pub fn new(
        handle: ConstraintHandle,
        particle_a: usize,
        particle_b: usize,
        rest_length: f32,
        stiffness: f32,
    ) -> Self {
        Self {
            handle,
            particle_a,
            particle_b,
            rest_length,
            stiffness,
            compliance: 0.0,
            breakable: false,
            max_stretch: DEFAULT_STRAIN_LIMIT,
            broken: false,
        }
    }

    /// Solve this distance constraint using PBD projection.
    /// Returns the constraint error magnitude.
    pub fn solve_pbd(&self, particles: &mut [Particle], dt: f32) -> f32 {
        if self.broken {
            return 0.0;
        }

        let pa = &particles[self.particle_a];
        let pb = &particles[self.particle_b];

        let w_a = pa.inv_mass;
        let w_b = pb.inv_mass;
        let w_sum = w_a + w_b;
        if w_sum < MIN_INV_MASS {
            return 0.0;
        }

        let diff = pb.predicted - pa.predicted;
        let dist = diff.length();
        if dist < EPSILON {
            return 0.0;
        }

        let c = dist - self.rest_length;
        let n = diff / dist;

        // XPBD compliance term
        let alpha = self.compliance / (dt * dt);
        let delta_lambda = -c / (w_sum + alpha);
        let correction = n * delta_lambda;

        let stiffness_factor = self.stiffness;
        particles[self.particle_a].predicted -= correction * w_a * stiffness_factor;
        particles[self.particle_b].predicted += correction * w_b * stiffness_factor;

        c.abs()
    }

    /// Check if this constraint should break due to exceeding strain limit.
    pub fn check_break(&mut self, particles: &[Particle]) -> bool {
        if !self.breakable || self.broken {
            return false;
        }
        let diff = particles[self.particle_b].predicted - particles[self.particle_a].predicted;
        let ratio = diff.length() / self.rest_length.max(EPSILON);
        if ratio > self.max_stretch {
            self.broken = true;
            true
        } else {
            false
        }
    }
}

/// A bending constraint that preserves the dihedral angle between two triangles
/// sharing an edge. Uses four particles: the shared edge (p1, p2) and the
/// opposing vertices (p0, p3).
#[derive(Debug, Clone)]
pub struct BendingConstraint {
    /// Handle for this constraint.
    pub handle: ConstraintHandle,
    /// The four particle indices: p0, p1 (edge), p2 (edge), p3.
    pub particles: [usize; 4],
    /// Rest dihedral angle in radians.
    pub rest_angle: f32,
    /// Bending stiffness in [0, 1].
    pub stiffness: f32,
    /// Compliance for XPBD.
    pub compliance: f32,
}

impl BendingConstraint {
    /// Create a new bending constraint. Computes the rest angle from current positions.
    pub fn new(
        handle: ConstraintHandle,
        particles: [usize; 4],
        stiffness: f32,
        all_particles: &[Particle],
    ) -> Self {
        let rest_angle = Self::compute_dihedral_angle(particles, all_particles);
        Self {
            handle,
            particles,
            rest_angle,
            stiffness,
            compliance: 0.0,
        }
    }

    /// Compute the dihedral angle between two triangles sharing an edge.
    fn compute_dihedral_angle(indices: [usize; 4], particles: &[Particle]) -> f32 {
        let p0 = particles[indices[0]].predicted;
        let p1 = particles[indices[1]].predicted;
        let p2 = particles[indices[2]].predicted;
        let p3 = particles[indices[3]].predicted;

        let e = p2 - p1; // shared edge
        let n0 = (p1 - p0).cross(p2 - p0);
        let n1 = (p2 - p3).cross(p1 - p3);

        let n0_len = n0.length();
        let n1_len = n1.length();
        if n0_len < EPSILON || n1_len < EPSILON {
            return 0.0;
        }

        let n0 = n0 / n0_len;
        let n1 = n1 / n1_len;

        let cos_angle = n0.dot(n1).clamp(-1.0, 1.0);
        let sin_angle = n0.cross(n1).dot(e.normalize_or_zero());

        sin_angle.atan2(cos_angle)
    }

    /// Solve this bending constraint using PBD.
    /// Returns the constraint error.
    pub fn solve_pbd(&self, particles: &mut [Particle], dt: f32) -> f32 {
        let current_angle = Self::compute_dihedral_angle(self.particles, particles);
        let c = current_angle - self.rest_angle;

        if c.abs() < EPSILON {
            return 0.0;
        }

        // Compute gradients using finite differences
        let h = 1e-4_f32;
        let mut gradients = [Vec3::ZERO; 4];

        for i in 0..4 {
            let idx = self.particles[i];
            for axis in 0..3 {
                let mut displaced = particles.to_vec();
                match axis {
                    0 => displaced[idx].predicted.x += h,
                    1 => displaced[idx].predicted.y += h,
                    2 => displaced[idx].predicted.z += h,
                    _ => unreachable!(),
                }
                let angle_plus = Self::compute_dihedral_angle(self.particles, &displaced);
                let grad = (angle_plus - current_angle) / h;
                match axis {
                    0 => gradients[i].x = grad,
                    1 => gradients[i].y = grad,
                    2 => gradients[i].z = grad,
                    _ => unreachable!(),
                }
            }
        }

        let mut denom = 0.0_f32;
        for i in 0..4 {
            let w = particles[self.particles[i]].inv_mass;
            denom += w * gradients[i].length_squared();
        }

        let alpha = self.compliance / (dt * dt);
        denom += alpha;

        if denom < EPSILON {
            return c.abs();
        }

        let s = -c / denom * self.stiffness;

        for i in 0..4 {
            let idx = self.particles[i];
            let w = particles[idx].inv_mass;
            if w > MIN_INV_MASS {
                particles[idx].predicted += gradients[i] * s * w;
            }
        }

        c.abs()
    }
}

/// A volume preservation constraint that attempts to maintain the volume of a
/// tetrahedron defined by four particles.
#[derive(Debug, Clone)]
pub struct VolumeConstraint {
    /// Handle for this constraint.
    pub handle: ConstraintHandle,
    /// The four particle indices forming the tetrahedron.
    pub particles: [usize; 4],
    /// Rest volume of the tetrahedron.
    pub rest_volume: f32,
    /// Stiffness in [0, 1].
    pub stiffness: f32,
    /// Compliance for XPBD.
    pub compliance: f32,
}

impl VolumeConstraint {
    /// Create a new volume constraint. Computes rest volume from current positions.
    pub fn new(
        handle: ConstraintHandle,
        particles: [usize; 4],
        stiffness: f32,
        all_particles: &[Particle],
    ) -> Self {
        let rest_volume = Self::compute_volume(particles, all_particles);
        Self {
            handle,
            particles,
            rest_volume,
            stiffness,
            compliance: 0.0,
        }
    }

    /// Compute the signed volume of the tetrahedron.
    fn compute_volume(indices: [usize; 4], particles: &[Particle]) -> f32 {
        let p0 = particles[indices[0]].predicted;
        let p1 = particles[indices[1]].predicted;
        let p2 = particles[indices[2]].predicted;
        let p3 = particles[indices[3]].predicted;

        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let e3 = p3 - p0;

        e1.dot(e2.cross(e3)) / 6.0
    }

    /// Compute gradients of the volume with respect to each particle position.
    fn compute_gradients(indices: [usize; 4], particles: &[Particle]) -> [Vec3; 4] {
        let p0 = particles[indices[0]].predicted;
        let p1 = particles[indices[1]].predicted;
        let p2 = particles[indices[2]].predicted;
        let p3 = particles[indices[3]].predicted;

        let factor = 1.0 / 6.0;
        let g1 = (p2 - p0).cross(p3 - p0) * factor;
        let g2 = (p3 - p0).cross(p1 - p0) * factor;
        let g3 = (p1 - p0).cross(p2 - p0) * factor;
        let g0 = -(g1 + g2 + g3);

        [g0, g1, g2, g3]
    }

    /// Solve this volume constraint using PBD projection.
    /// Returns the constraint error.
    pub fn solve_pbd(&self, particles: &mut [Particle], dt: f32) -> f32 {
        let current_volume = Self::compute_volume(self.particles, particles);
        let c = current_volume - self.rest_volume;

        if c.abs() < EPSILON {
            return 0.0;
        }

        let gradients = Self::compute_gradients(self.particles, particles);

        let mut denom = 0.0_f32;
        for i in 0..4 {
            let w = particles[self.particles[i]].inv_mass;
            denom += w * gradients[i].length_squared();
        }

        let alpha = self.compliance / (dt * dt);
        denom += alpha;

        if denom < EPSILON {
            return c.abs();
        }

        let s = -c / denom * self.stiffness;

        for i in 0..4 {
            let idx = self.particles[i];
            let w = particles[idx].inv_mass;
            if w > MIN_INV_MASS {
                particles[idx].predicted += gradients[i] * s * w;
            }
        }

        c.abs()
    }
}

// ---------------------------------------------------------------------------
// Strain limiter
// ---------------------------------------------------------------------------

/// Limits the strain (stretch) of distance constraints to prevent explosion.
#[derive(Debug, Clone)]
pub struct StrainLimiter {
    /// Maximum compression ratio (e.g., 0.9 means cannot compress below 90% rest length).
    pub min_ratio: f32,
    /// Maximum stretch ratio (e.g., 1.1 means cannot stretch beyond 110% rest length).
    pub max_ratio: f32,
}

impl Default for StrainLimiter {
    fn default() -> Self {
        Self {
            min_ratio: 0.9,
            max_ratio: 1.1,
        }
    }
}

impl StrainLimiter {
    /// Create a new strain limiter with the given compression/stretch limits.
    pub fn new(min_ratio: f32, max_ratio: f32) -> Self {
        Self { min_ratio, max_ratio }
    }

    /// Apply strain limiting to all distance constraints in the system.
    pub fn apply(
        &self,
        particles: &mut [Particle],
        constraints: &[DistanceConstraint],
    ) {
        for c in constraints {
            if c.broken {
                continue;
            }
            let pa = &particles[c.particle_a];
            let pb = &particles[c.particle_b];
            let w_a = pa.inv_mass;
            let w_b = pb.inv_mass;
            let w_sum = w_a + w_b;
            if w_sum < MIN_INV_MASS {
                continue;
            }

            let diff = pb.predicted - pa.predicted;
            let dist = diff.length();
            if dist < EPSILON {
                continue;
            }

            let ratio = dist / c.rest_length.max(EPSILON);
            let clamped_ratio = ratio.clamp(self.min_ratio, self.max_ratio);

            if (ratio - clamped_ratio).abs() < EPSILON {
                continue;
            }

            let target_dist = c.rest_length * clamped_ratio;
            let correction_mag = dist - target_dist;
            let n = diff / dist;
            let correction = n * correction_mag;

            particles[c.particle_a].predicted += correction * (w_a / w_sum);
            particles[c.particle_b].predicted -= correction * (w_b / w_sum);
        }
    }
}

// ---------------------------------------------------------------------------
// Collision shapes (environment)
// ---------------------------------------------------------------------------

/// An environment collision shape that particles collide against.
#[derive(Debug, Clone)]
pub enum CollisionShape {
    /// Infinite plane defined by a point and normal.
    Plane { point: Vec3, normal: Vec3 },
    /// Sphere defined by center and radius.
    Sphere { center: Vec3, radius: f32 },
    /// Axis-aligned box defined by min and max corners.
    Box { min: Vec3, max: Vec3 },
    /// Capsule defined by two endpoints and a radius.
    Capsule { start: Vec3, end: Vec3, radius: f32 },
}

impl CollisionShape {
    /// Compute the signed distance from a point to this shape.
    /// Negative values indicate penetration.
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        match self {
            CollisionShape::Plane { point: p, normal } => {
                (point - *p).dot(*normal)
            }
            CollisionShape::Sphere { center, radius } => {
                (point - *center).length() - radius
            }
            CollisionShape::Box { min, max } => {
                let center = (*min + *max) * 0.5;
                let half = (*max - *min) * 0.5;
                let local = point - center;
                let d = Vec3::new(
                    local.x.abs() - half.x,
                    local.y.abs() - half.y,
                    local.z.abs() - half.z,
                );
                let outside = Vec3::new(d.x.max(0.0), d.y.max(0.0), d.z.max(0.0)).length();
                let inside = d.x.max(d.y).max(d.z).min(0.0);
                outside + inside
            }
            CollisionShape::Capsule { start, end, radius } => {
                let ab = *end - *start;
                let ap = point - *start;
                let t = ap.dot(ab) / ab.dot(ab).max(EPSILON);
                let t = t.clamp(0.0, 1.0);
                let closest = *start + ab * t;
                (point - closest).length() - radius
            }
        }
    }

    /// Compute the surface normal at the closest point from the given position.
    pub fn normal_at(&self, point: Vec3) -> Vec3 {
        match self {
            CollisionShape::Plane { normal, .. } => *normal,
            CollisionShape::Sphere { center, .. } => {
                (point - *center).normalize_or_zero()
            }
            CollisionShape::Box { min, max } => {
                let center = (*min + *max) * 0.5;
                let half = (*max - *min) * 0.5;
                let local = point - center;
                // Find the closest face
                let mut min_dist = f32::INFINITY;
                let mut normal = Vec3::Y;
                for axis in 0..3 {
                    let d_pos = half[axis] - local[axis];
                    let d_neg = half[axis] + local[axis];
                    if d_pos < min_dist {
                        min_dist = d_pos;
                        normal = Vec3::ZERO;
                        normal[axis] = 1.0;
                    }
                    if d_neg < min_dist {
                        min_dist = d_neg;
                        normal = Vec3::ZERO;
                        normal[axis] = -1.0;
                    }
                }
                normal
            }
            CollisionShape::Capsule { start, end, .. } => {
                let ab = *end - *start;
                let ap = point - *start;
                let t = (ap.dot(ab) / ab.dot(ab).max(EPSILON)).clamp(0.0, 1.0);
                let closest = *start + ab * t;
                (point - closest).normalize_or_zero()
            }
        }
    }

    /// Project a penetrating point out of this shape. Returns the corrected position.
    pub fn project_out(&self, point: Vec3, particle_radius: f32) -> Vec3 {
        let sd = self.signed_distance(point);
        if sd < particle_radius {
            let n = self.normal_at(point);
            point + n * (particle_radius - sd)
        } else {
            point
        }
    }
}

// ---------------------------------------------------------------------------
// Spatial hash for self-collision
// ---------------------------------------------------------------------------

/// Cell key for the spatial hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CellKey {
    x: i32,
    y: i32,
    z: i32,
}

/// Spatial hash for efficient neighbor queries during self-collision detection.
#[derive(Debug)]
pub struct SpatialHash {
    /// Cell size.
    cell_size: f32,
    /// Inverse cell size for fast computation.
    inv_cell_size: f32,
    /// Map from cell keys to lists of particle indices.
    cells: HashMap<CellKey, Vec<usize>>,
    /// Number of particles inserted.
    particle_count: usize,
}

impl SpatialHash {
    /// Create a new spatial hash with the given cell size.
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size: cell_size.max(EPSILON),
            inv_cell_size: 1.0 / cell_size.max(EPSILON),
            cells: HashMap::new(),
            particle_count: 0,
        }
    }

    /// Clear all entries from the spatial hash.
    pub fn clear(&mut self) {
        self.cells.clear();
        self.particle_count = 0;
    }

    /// Compute the cell key for a given position.
    fn cell_key(&self, position: Vec3) -> CellKey {
        CellKey {
            x: (position.x * self.inv_cell_size).floor() as i32,
            y: (position.y * self.inv_cell_size).floor() as i32,
            z: (position.z * self.inv_cell_size).floor() as i32,
        }
    }

    /// Insert a particle index at the given position.
    pub fn insert(&mut self, index: usize, position: Vec3) {
        let key = self.cell_key(position);
        let cell = self.cells.entry(key).or_insert_with(Vec::new);
        if cell.len() < MAX_PARTICLES_PER_CELL {
            cell.push(index);
            self.particle_count += 1;
        }
    }

    /// Build the spatial hash from a set of particles.
    pub fn build(&mut self, particles: &[Particle]) {
        self.clear();
        for (i, p) in particles.iter().enumerate() {
            self.insert(i, p.predicted);
        }
    }

    /// Query all particle indices within a radius of the given position.
    /// The callback receives each nearby particle index.
    pub fn query_radius(&self, position: Vec3, radius: f32, results: &mut Vec<usize>) {
        let min_key = self.cell_key(position - Vec3::splat(radius));
        let max_key = self.cell_key(position + Vec3::splat(radius));

        for z in min_key.z..=max_key.z {
            for y in min_key.y..=max_key.y {
                for x in min_key.x..=max_key.x {
                    let key = CellKey { x, y, z };
                    if let Some(cell) = self.cells.get(&key) {
                        results.extend(cell.iter().copied());
                    }
                }
            }
        }
    }

    /// Get the number of non-empty cells.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Get the total number of particle entries.
    pub fn entry_count(&self) -> usize {
        self.particle_count
    }
}

// ---------------------------------------------------------------------------
// Collision pair (self-collision result)
// ---------------------------------------------------------------------------

/// A detected self-collision pair between two particles.
#[derive(Debug, Clone)]
pub struct CollisionPair {
    /// Index of the first particle.
    pub a: usize,
    /// Index of the second particle.
    pub b: usize,
    /// Penetration depth.
    pub depth: f32,
    /// Contact normal (from a to b).
    pub normal: Vec3,
}

// ---------------------------------------------------------------------------
// Self-collision detector
// ---------------------------------------------------------------------------

/// Detects self-collisions between particles using spatial hashing.
#[derive(Debug)]
pub struct SelfCollisionDetector {
    /// Spatial hash for broad-phase.
    spatial_hash: SpatialHash,
    /// Collision radius for particles.
    collision_radius: f32,
    /// Scratch buffer for query results.
    query_buffer: Vec<usize>,
    /// Detected collision pairs.
    pairs: Vec<CollisionPair>,
}

impl SelfCollisionDetector {
    /// Create a new self-collision detector with the given cell size and collision radius.
    pub fn new(cell_size: f32, collision_radius: f32) -> Self {
        Self {
            spatial_hash: SpatialHash::new(cell_size),
            collision_radius,
            query_buffer: Vec::with_capacity(64),
            pairs: Vec::new(),
        }
    }

    /// Detect all self-collision pairs among the given particles.
    pub fn detect(&mut self, particles: &[Particle]) -> &[CollisionPair] {
        self.pairs.clear();
        self.spatial_hash.build(particles);

        for i in 0..particles.len() {
            let p = &particles[i];
            if p.inv_mass < MIN_INV_MASS && p.pinned {
                continue;
            }

            self.query_buffer.clear();
            let r = self.collision_radius + p.radius;
            self.spatial_hash.query_radius(p.predicted, r, &mut self.query_buffer);

            for &j in &self.query_buffer {
                if j <= i {
                    continue; // avoid duplicates
                }

                let q = &particles[j];

                // Skip same-phase particles (e.g., particles in the same triangle)
                if p.phase == q.phase && p.phase != 0 {
                    continue;
                }

                let diff = q.predicted - p.predicted;
                let dist = diff.length();
                let min_dist = p.radius + q.radius;

                if dist < min_dist && dist > EPSILON {
                    self.pairs.push(CollisionPair {
                        a: i,
                        b: j,
                        depth: min_dist - dist,
                        normal: diff / dist,
                    });
                }
            }
        }

        &self.pairs
    }

    /// Resolve detected self-collision pairs by projecting particles apart.
    pub fn resolve(&self, particles: &mut [Particle], friction: f32) {
        for pair in &self.pairs {
            let w_a = particles[pair.a].inv_mass;
            let w_b = particles[pair.b].inv_mass;
            let w_sum = w_a + w_b;
            if w_sum < MIN_INV_MASS {
                continue;
            }

            let correction = pair.normal * pair.depth;
            particles[pair.a].predicted -= correction * (w_a / w_sum);
            particles[pair.b].predicted += correction * (w_b / w_sum);

            // Friction: dampen relative tangential velocity
            if friction > 0.0 {
                let rel_vel = particles[pair.b].velocity - particles[pair.a].velocity;
                let vn = rel_vel.dot(pair.normal);
                let vt = rel_vel - pair.normal * vn;
                let vt_len = vt.length();
                if vt_len > EPSILON {
                    let friction_impulse = vt * friction.min(1.0) * 0.5;
                    if w_a > MIN_INV_MASS {
                        particles[pair.a].velocity += friction_impulse * w_a;
                    }
                    if w_b > MIN_INV_MASS {
                        particles[pair.b].velocity -= friction_impulse * w_b;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Mass-Spring Network builder
// ---------------------------------------------------------------------------

/// Topology for a mass-spring network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkTopology {
    /// Regular grid (rows x cols).
    Grid,
    /// Triangulated grid.
    TriangulatedGrid,
    /// Random point cloud with nearest-neighbor connections.
    PointCloud,
}

/// Builder for creating mass-spring networks.
#[derive(Debug)]
pub struct MassSpringNetworkBuilder {
    /// Starting position (top-left corner for grids).
    pub origin: Vec3,
    /// Spacing between particles for grid topologies.
    pub spacing: f32,
    /// Number of rows (Y direction for grids).
    pub rows: usize,
    /// Number of columns (X direction for grids).
    pub cols: usize,
    /// Mass per particle.
    pub particle_mass: f32,
    /// Structural constraint stiffness.
    pub structural_stiffness: f32,
    /// Shear constraint stiffness.
    pub shear_stiffness: f32,
    /// Bend constraint stiffness.
    pub bend_stiffness: f32,
    /// Topology type.
    pub topology: NetworkTopology,
    /// Whether to add bending constraints.
    pub enable_bending: bool,
    /// Whether to add shear constraints (for grid topologies).
    pub enable_shear: bool,
}

impl Default for MassSpringNetworkBuilder {
    fn default() -> Self {
        Self {
            origin: Vec3::ZERO,
            spacing: 0.1,
            rows: 10,
            cols: 10,
            particle_mass: 0.1,
            structural_stiffness: 1.0,
            shear_stiffness: 0.5,
            bend_stiffness: 0.2,
            topology: NetworkTopology::Grid,
            enable_bending: true,
            enable_shear: true,
        }
    }
}

impl MassSpringNetworkBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the origin position.
    pub fn with_origin(mut self, origin: Vec3) -> Self {
        self.origin = origin;
        self
    }

    /// Set the grid dimensions.
    pub fn with_dimensions(mut self, rows: usize, cols: usize) -> Self {
        self.rows = rows;
        self.cols = cols;
        self
    }

    /// Set the particle spacing.
    pub fn with_spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing;
        self
    }

    /// Set the particle mass.
    pub fn with_mass(mut self, mass: f32) -> Self {
        self.particle_mass = mass;
        self
    }

    /// Set all stiffness values at once.
    pub fn with_stiffness(mut self, structural: f32, shear: f32, bend: f32) -> Self {
        self.structural_stiffness = structural;
        self.shear_stiffness = shear;
        self.bend_stiffness = bend;
        self
    }

    /// Build the network, producing particles and constraints.
    pub fn build(
        &self,
        next_particle_id: &mut u32,
        next_constraint_id: &mut u32,
    ) -> (Vec<Particle>, Vec<DistanceConstraint>, Vec<BendingConstraint>) {
        let mut particles = Vec::new();
        let mut distance_constraints = Vec::new();
        let mut bending_constraints = Vec::new();

        match self.topology {
            NetworkTopology::Grid | NetworkTopology::TriangulatedGrid => {
                // Create grid particles
                for row in 0..self.rows {
                    for col in 0..self.cols {
                        let pos = self.origin
                            + Vec3::new(col as f32 * self.spacing, 0.0, row as f32 * self.spacing);
                        let id = ParticleId(*next_particle_id);
                        *next_particle_id += 1;
                        particles.push(Particle::new(id, pos, self.particle_mass));
                    }
                }

                let idx = |r: usize, c: usize| -> usize { r * self.cols + c };

                // Structural constraints (horizontal + vertical)
                for row in 0..self.rows {
                    for col in 0..self.cols {
                        // Right neighbor
                        if col + 1 < self.cols {
                            let handle = ConstraintHandle(*next_constraint_id);
                            *next_constraint_id += 1;
                            distance_constraints.push(DistanceConstraint::new(
                                handle,
                                idx(row, col),
                                idx(row, col + 1),
                                self.spacing,
                                self.structural_stiffness,
                            ));
                        }
                        // Down neighbor
                        if row + 1 < self.rows {
                            let handle = ConstraintHandle(*next_constraint_id);
                            *next_constraint_id += 1;
                            distance_constraints.push(DistanceConstraint::new(
                                handle,
                                idx(row, col),
                                idx(row + 1, col),
                                self.spacing,
                                self.structural_stiffness,
                            ));
                        }
                    }
                }

                // Shear constraints (diagonals)
                if self.enable_shear {
                    let diag_len = self.spacing * std::f32::consts::SQRT_2;
                    for row in 0..self.rows - 1 {
                        for col in 0..self.cols - 1 {
                            // Top-left to bottom-right
                            let handle = ConstraintHandle(*next_constraint_id);
                            *next_constraint_id += 1;
                            distance_constraints.push(DistanceConstraint::new(
                                handle,
                                idx(row, col),
                                idx(row + 1, col + 1),
                                diag_len,
                                self.shear_stiffness,
                            ));
                            // Top-right to bottom-left
                            let handle = ConstraintHandle(*next_constraint_id);
                            *next_constraint_id += 1;
                            distance_constraints.push(DistanceConstraint::new(
                                handle,
                                idx(row, col + 1),
                                idx(row + 1, col),
                                diag_len,
                                self.shear_stiffness,
                            ));
                        }
                    }
                }

                // Bending constraints (skip-one)
                if self.enable_bending {
                    let bend_len = self.spacing * 2.0;
                    for row in 0..self.rows {
                        for col in 0..self.cols {
                            // Skip-one horizontal
                            if col + 2 < self.cols {
                                let handle = ConstraintHandle(*next_constraint_id);
                                *next_constraint_id += 1;
                                distance_constraints.push(DistanceConstraint::new(
                                    handle,
                                    idx(row, col),
                                    idx(row, col + 2),
                                    bend_len,
                                    self.bend_stiffness,
                                ));
                            }
                            // Skip-one vertical
                            if row + 2 < self.rows {
                                let handle = ConstraintHandle(*next_constraint_id);
                                *next_constraint_id += 1;
                                distance_constraints.push(DistanceConstraint::new(
                                    handle,
                                    idx(row, col),
                                    idx(row + 2, col),
                                    bend_len,
                                    self.bend_stiffness,
                                ));
                            }
                        }
                    }

                    // Dihedral bending constraints for triangulated grids
                    if self.topology == NetworkTopology::TriangulatedGrid {
                        for row in 0..self.rows - 1 {
                            for col in 0..self.cols - 1 {
                                if col + 1 < self.cols && row + 1 < self.rows {
                                    let handle = ConstraintHandle(*next_constraint_id);
                                    *next_constraint_id += 1;
                                    let indices = [
                                        idx(row, col),
                                        idx(row, col + 1),
                                        idx(row + 1, col),
                                        idx(row + 1, col + 1),
                                    ];
                                    bending_constraints.push(BendingConstraint::new(
                                        handle,
                                        indices,
                                        self.bend_stiffness,
                                        &particles,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            NetworkTopology::PointCloud => {
                // For point clouds, we just create the particles; the user adds connections
                for _ in 0..(self.rows * self.cols) {
                    let id = ParticleId(*next_particle_id);
                    *next_particle_id += 1;
                    particles.push(Particle::new(id, self.origin, self.particle_mass));
                }
            }
        }

        (particles, distance_constraints, bending_constraints)
    }
}

// ---------------------------------------------------------------------------
// Particle group
// ---------------------------------------------------------------------------

/// A named group of particles that can be operated on as a unit.
#[derive(Debug, Clone)]
pub struct ParticleGroup {
    /// Name of this group.
    pub name: String,
    /// Indices of particles belonging to this group.
    pub particle_indices: Vec<usize>,
    /// Whether this group is active (participates in simulation).
    pub active: bool,
    /// Center of mass of the group (cached).
    pub center_of_mass: Vec3,
    /// Total mass of the group (cached).
    pub total_mass: f32,
}

impl ParticleGroup {
    /// Create a new empty particle group.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            particle_indices: Vec::new(),
            active: true,
            center_of_mass: Vec3::ZERO,
            total_mass: 0.0,
        }
    }

    /// Add a particle index to this group.
    pub fn add_particle(&mut self, index: usize) {
        if !self.particle_indices.contains(&index) {
            self.particle_indices.push(index);
        }
    }

    /// Remove a particle index from this group.
    pub fn remove_particle(&mut self, index: usize) {
        self.particle_indices.retain(|&i| i != index);
    }

    /// Recompute center of mass and total mass from current particle data.
    pub fn update_cached_properties(&mut self, particles: &[Particle]) {
        self.total_mass = 0.0;
        self.center_of_mass = Vec3::ZERO;

        for &idx in &self.particle_indices {
            if idx < particles.len() {
                let p = &particles[idx];
                self.center_of_mass += p.position * p.mass;
                self.total_mass += p.mass;
            }
        }

        if self.total_mass > EPSILON {
            self.center_of_mass /= self.total_mass;
        }
    }

    /// Apply a force to all particles in this group.
    pub fn apply_force(&self, particles: &mut [Particle], force: Vec3) {
        for &idx in &self.particle_indices {
            if idx < particles.len() {
                particles[idx].apply_force(force);
            }
        }
    }

    /// Pin all particles in this group.
    pub fn pin_all(&self, particles: &mut [Particle]) {
        for &idx in &self.particle_indices {
            if idx < particles.len() {
                particles[idx].pin();
            }
        }
    }

    /// Unpin all particles in this group.
    pub fn unpin_all(&self, particles: &mut [Particle]) {
        for &idx in &self.particle_indices {
            if idx < particles.len() {
                particles[idx].unpin();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Particle emitter
// ---------------------------------------------------------------------------

/// Configuration for a particle emitter.
#[derive(Debug, Clone)]
pub struct ParticleEmitterConfig {
    /// World-space position of the emitter.
    pub position: Vec3,
    /// Direction of emission.
    pub direction: Vec3,
    /// Spread cone half-angle in radians.
    pub spread_angle: f32,
    /// Emission rate in particles per second.
    pub rate: f32,
    /// Initial speed of emitted particles.
    pub initial_speed: f32,
    /// Speed variation (random +/- range).
    pub speed_variation: f32,
    /// Mass of emitted particles.
    pub particle_mass: f32,
    /// Lifetime of emitted particles in seconds.
    pub particle_lifetime: f32,
    /// Collision radius of emitted particles.
    pub particle_radius: f32,
    /// Maximum number of particles this emitter can have alive.
    pub max_particles: usize,
    /// Whether the emitter is currently active.
    pub active: bool,
    /// Phase assigned to emitted particles.
    pub phase: i32,
}

impl Default for ParticleEmitterConfig {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            direction: Vec3::Y,
            spread_angle: 0.3,
            rate: 100.0,
            initial_speed: 5.0,
            speed_variation: 1.0,
            particle_mass: 0.01,
            particle_lifetime: 5.0,
            particle_radius: 0.02,
            max_particles: 10000,
            active: true,
            phase: 0,
        }
    }
}

/// A particle emitter that spawns particles over time.
#[derive(Debug)]
pub struct ParticleEmitter {
    /// Emitter configuration.
    pub config: ParticleEmitterConfig,
    /// Accumulated emission time remainder.
    pub emission_accumulator: f32,
    /// Number of currently alive particles from this emitter.
    pub alive_count: usize,
    /// Simple pseudo-random state for deterministic emission.
    rng_state: u32,
}

impl ParticleEmitter {
    /// Create a new particle emitter with the given config.
    pub fn new(config: ParticleEmitterConfig) -> Self {
        Self {
            config,
            emission_accumulator: 0.0,
            alive_count: 0,
            rng_state: 42,
        }
    }

    /// Simple xorshift32 pseudo-random number generator.
    fn next_random(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 17;
        self.rng_state ^= self.rng_state << 5;
        (self.rng_state as f32) / (u32::MAX as f32)
    }

    /// Generate a random direction within the spread cone.
    fn random_direction(&mut self) -> Vec3 {
        let r1 = self.next_random();
        let r2 = self.next_random();
        let cos_theta = 1.0 - r1 * (1.0 - self.config.spread_angle.cos());
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi = 2.0 * std::f32::consts::PI * r2;

        // Build an orthonormal basis from the emission direction
        let w = self.config.direction.normalize_or_zero();
        let u = if w.y.abs() < 0.999 {
            Vec3::Y.cross(w).normalize_or_zero()
        } else {
            Vec3::X.cross(w).normalize_or_zero()
        };
        let v = w.cross(u);

        u * (sin_theta * phi.cos()) + v * (sin_theta * phi.sin()) + w * cos_theta
    }

    /// Emit particles for the given timestep. Returns new particles to add.
    pub fn emit(&mut self, dt: f32, next_id: &mut u32) -> Vec<Particle> {
        let mut new_particles = Vec::new();
        if !self.config.active {
            return new_particles;
        }

        self.emission_accumulator += dt * self.config.rate;

        while self.emission_accumulator >= 1.0 && self.alive_count < self.config.max_particles {
            self.emission_accumulator -= 1.0;

            let direction = self.random_direction();
            let speed_var = (self.next_random() - 0.5) * 2.0 * self.config.speed_variation;
            let speed = (self.config.initial_speed + speed_var).max(0.0);

            let id = ParticleId(*next_id);
            *next_id += 1;

            let mut particle = Particle::new(id, self.config.position, self.config.particle_mass);
            particle.velocity = direction * speed;
            particle.lifetime = self.config.particle_lifetime;
            particle.radius = self.config.particle_radius;
            particle.phase = self.config.phase;

            new_particles.push(particle);
            self.alive_count += 1;
        }

        new_particles
    }
}

// ---------------------------------------------------------------------------
// Simulation settings
// ---------------------------------------------------------------------------

/// Configuration for the particle physics simulation.
#[derive(Debug, Clone)]
pub struct ParticleSimulationSettings {
    /// Gravity vector.
    pub gravity: Vec3,
    /// Velocity damping factor per step.
    pub damping: f32,
    /// Number of PBD constraint solver iterations.
    pub solver_iterations: usize,
    /// Number of substeps per frame.
    pub substeps: usize,
    /// Spatial hash cell size for self-collision.
    pub cell_size: f32,
    /// Whether self-collision is enabled.
    pub enable_self_collision: bool,
    /// Friction coefficient for self-collision.
    pub self_collision_friction: f32,
    /// Whether strain limiting is enabled.
    pub enable_strain_limiting: bool,
    /// Strain limiter configuration.
    pub strain_limiter: StrainLimiter,
    /// Whether to remove expired particles automatically.
    pub auto_remove_expired: bool,
    /// Maximum number of particles in the simulation.
    pub max_particles: usize,
    /// Global time scale (1.0 = real-time).
    pub time_scale: f32,
    /// Whether sleeping is enabled for low-energy particles.
    pub enable_sleeping: bool,
    /// Velocity threshold below which particles are candidates for sleeping.
    pub sleep_velocity_threshold: f32,
    /// Number of frames a particle must be below threshold to sleep.
    pub sleep_frames_threshold: usize,
}

impl Default for ParticleSimulationSettings {
    fn default() -> Self {
        Self {
            gravity: DEFAULT_GRAVITY,
            damping: DEFAULT_DAMPING,
            solver_iterations: DEFAULT_SOLVER_ITERATIONS,
            substeps: DEFAULT_SUBSTEPS,
            cell_size: DEFAULT_CELL_SIZE,
            enable_self_collision: true,
            self_collision_friction: 0.3,
            enable_strain_limiting: true,
            strain_limiter: StrainLimiter::default(),
            auto_remove_expired: true,
            max_particles: 100_000,
            time_scale: 1.0,
            enable_sleeping: false,
            sleep_velocity_threshold: 0.01,
            sleep_frames_threshold: 60,
        }
    }
}

// ---------------------------------------------------------------------------
// Simulation statistics
// ---------------------------------------------------------------------------

/// Runtime statistics for the particle simulation.
#[derive(Debug, Clone, Default)]
pub struct ParticleSimulationStats {
    /// Total number of alive particles.
    pub particle_count: usize,
    /// Number of active (non-sleeping) particles.
    pub active_particle_count: usize,
    /// Number of distance constraints.
    pub distance_constraint_count: usize,
    /// Number of bending constraints.
    pub bending_constraint_count: usize,
    /// Number of volume constraints.
    pub volume_constraint_count: usize,
    /// Number of self-collision pairs detected.
    pub self_collision_pairs: usize,
    /// Number of environment collisions resolved.
    pub environment_collisions: usize,
    /// Number of broken constraints.
    pub broken_constraints: usize,
    /// Number of spatial hash cells used.
    pub spatial_hash_cells: usize,
    /// Average constraint error after solving.
    pub avg_constraint_error: f32,
    /// Maximum constraint error after solving.
    pub max_constraint_error: f32,
    /// Number of particles removed this frame (expired).
    pub particles_removed: usize,
    /// Number of particles emitted this frame.
    pub particles_emitted: usize,
}

// ---------------------------------------------------------------------------
// Main simulation
// ---------------------------------------------------------------------------

/// The main particle physics simulation, managing particles, constraints,
/// colliders, emitters, and groups.
#[derive(Debug)]
pub struct ParticleSimulation {
    /// All particles in the simulation.
    pub particles: Vec<Particle>,
    /// Distance constraints.
    pub distance_constraints: Vec<DistanceConstraint>,
    /// Bending constraints.
    pub bending_constraints: Vec<BendingConstraint>,
    /// Volume constraints.
    pub volume_constraints: Vec<VolumeConstraint>,
    /// Environment collision shapes.
    pub collision_shapes: Vec<CollisionShape>,
    /// Self-collision detector.
    self_collision: SelfCollisionDetector,
    /// Named particle groups.
    pub groups: Vec<ParticleGroup>,
    /// Particle emitters.
    pub emitters: Vec<ParticleEmitter>,
    /// Simulation settings.
    pub settings: ParticleSimulationSettings,
    /// Statistics from the last simulation step.
    pub stats: ParticleSimulationStats,
    /// Next particle ID counter.
    next_particle_id: u32,
    /// Next constraint handle counter.
    next_constraint_id: u32,
    /// Per-particle sleep counters (frames below threshold).
    sleep_counters: Vec<usize>,
    /// Per-particle sleeping state.
    sleeping: Vec<bool>,
}

impl ParticleSimulation {
    /// Create a new empty particle simulation with default settings.
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            distance_constraints: Vec::new(),
            bending_constraints: Vec::new(),
            volume_constraints: Vec::new(),
            collision_shapes: Vec::new(),
            self_collision: SelfCollisionDetector::new(DEFAULT_CELL_SIZE, DEFAULT_COLLISION_RADIUS),
            groups: Vec::new(),
            emitters: Vec::new(),
            settings: ParticleSimulationSettings::default(),
            stats: ParticleSimulationStats::default(),
            next_particle_id: 0,
            next_constraint_id: 0,
            sleep_counters: Vec::new(),
            sleeping: Vec::new(),
        }
    }

    /// Create a new simulation with custom settings.
    pub fn with_settings(settings: ParticleSimulationSettings) -> Self {
        let cell_size = settings.cell_size;
        Self {
            particles: Vec::new(),
            distance_constraints: Vec::new(),
            bending_constraints: Vec::new(),
            volume_constraints: Vec::new(),
            collision_shapes: Vec::new(),
            self_collision: SelfCollisionDetector::new(cell_size, DEFAULT_COLLISION_RADIUS),
            groups: Vec::new(),
            emitters: Vec::new(),
            settings,
            stats: ParticleSimulationStats::default(),
            next_particle_id: 0,
            next_constraint_id: 0,
            sleep_counters: Vec::new(),
            sleeping: Vec::new(),
        }
    }

    /// Add a particle to the simulation. Returns its index.
    pub fn add_particle(&mut self, position: Vec3, mass: f32) -> usize {
        let id = ParticleId(self.next_particle_id);
        self.next_particle_id += 1;
        let particle = Particle::new(id, position, mass);
        let index = self.particles.len();
        self.particles.push(particle);
        self.sleep_counters.push(0);
        self.sleeping.push(false);
        index
    }

    /// Add a static (pinned) particle. Returns its index.
    pub fn add_static_particle(&mut self, position: Vec3) -> usize {
        let id = ParticleId(self.next_particle_id);
        self.next_particle_id += 1;
        let particle = Particle::new_static(id, position);
        let index = self.particles.len();
        self.particles.push(particle);
        self.sleep_counters.push(0);
        self.sleeping.push(false);
        index
    }

    /// Add a distance constraint between two particles. Returns the constraint handle.
    pub fn add_distance_constraint(
        &mut self,
        a: usize,
        b: usize,
        stiffness: f32,
    ) -> ConstraintHandle {
        let rest_length = (self.particles[b].position - self.particles[a].position).length();
        let handle = ConstraintHandle(self.next_constraint_id);
        self.next_constraint_id += 1;
        self.distance_constraints.push(DistanceConstraint::new(
            handle, a, b, rest_length, stiffness,
        ));
        handle
    }

    /// Add a distance constraint with a specific rest length.
    pub fn add_distance_constraint_with_length(
        &mut self,
        a: usize,
        b: usize,
        rest_length: f32,
        stiffness: f32,
    ) -> ConstraintHandle {
        let handle = ConstraintHandle(self.next_constraint_id);
        self.next_constraint_id += 1;
        self.distance_constraints.push(DistanceConstraint::new(
            handle, a, b, rest_length, stiffness,
        ));
        handle
    }

    /// Add a bending constraint between four particles.
    pub fn add_bending_constraint(
        &mut self,
        particles: [usize; 4],
        stiffness: f32,
    ) -> ConstraintHandle {
        let handle = ConstraintHandle(self.next_constraint_id);
        self.next_constraint_id += 1;
        self.bending_constraints.push(BendingConstraint::new(
            handle, particles, stiffness, &self.particles,
        ));
        handle
    }

    /// Add a volume preservation constraint between four particles.
    pub fn add_volume_constraint(
        &mut self,
        particles: [usize; 4],
        stiffness: f32,
    ) -> ConstraintHandle {
        let handle = ConstraintHandle(self.next_constraint_id);
        self.next_constraint_id += 1;
        self.volume_constraints.push(VolumeConstraint::new(
            handle, particles, stiffness, &self.particles,
        ));
        handle
    }

    /// Add an environment collision shape.
    pub fn add_collision_shape(&mut self, shape: CollisionShape) {
        self.collision_shapes.push(shape);
    }

    /// Add a particle emitter.
    pub fn add_emitter(&mut self, config: ParticleEmitterConfig) {
        self.emitters.push(ParticleEmitter::new(config));
    }

    /// Create and add a named particle group. Returns its index.
    pub fn add_group(&mut self, name: &str) -> usize {
        let idx = self.groups.len();
        self.groups.push(ParticleGroup::new(name));
        idx
    }

    /// Add a particle to a named group.
    pub fn add_to_group(&mut self, group_index: usize, particle_index: usize) {
        if group_index < self.groups.len() {
            self.groups[group_index].add_particle(particle_index);
        }
    }

    /// Build a mass-spring network and add it to the simulation.
    /// Returns the range of particle indices added.
    pub fn add_mass_spring_network(
        &mut self,
        builder: &MassSpringNetworkBuilder,
    ) -> std::ops::Range<usize> {
        let start = self.particles.len();
        let (particles, dist_constraints, bend_constraints) =
            builder.build(&mut self.next_particle_id, &mut self.next_constraint_id);

        let count = particles.len();
        self.particles.extend(particles);
        self.distance_constraints.extend(dist_constraints);
        self.bending_constraints.extend(bend_constraints);
        self.sleep_counters.resize(self.particles.len(), 0);
        self.sleeping.resize(self.particles.len(), false);

        start..(start + count)
    }

    /// Remove expired particles and their associated constraints.
    fn remove_expired_particles(&mut self) -> usize {
        let mut removed = 0usize;
        let mut expired_indices: Vec<usize> = Vec::new();

        for (i, p) in self.particles.iter().enumerate() {
            if p.is_expired() && p.lifetime >= 0.0 {
                expired_indices.push(i);
            }
        }

        if expired_indices.is_empty() {
            return 0;
        }

        // Remove in reverse order to maintain index validity
        for &idx in expired_indices.iter().rev() {
            self.particles.swap_remove(idx);
            if idx < self.sleep_counters.len() {
                self.sleep_counters.swap_remove(idx);
            }
            if idx < self.sleeping.len() {
                self.sleeping.swap_remove(idx);
            }
            removed += 1;
        }

        // Fix up constraint indices (remove broken ones referencing removed particles)
        let particle_count = self.particles.len();
        self.distance_constraints.retain(|c| {
            c.particle_a < particle_count && c.particle_b < particle_count
        });
        self.bending_constraints.retain(|c| {
            c.particles.iter().all(|&i| i < particle_count)
        });
        self.volume_constraints.retain(|c| {
            c.particles.iter().all(|&i| i < particle_count)
        });

        // Update emitter alive counts
        for emitter in &mut self.emitters {
            emitter.alive_count = emitter.alive_count.saturating_sub(removed);
        }

        removed
    }

    /// Emit particles from all active emitters.
    fn emit_particles(&mut self, dt: f32) -> usize {
        let mut total_emitted = 0;
        let mut new_particles = Vec::new();

        for emitter in &mut self.emitters {
            if self.particles.len() + new_particles.len() >= self.settings.max_particles {
                break;
            }
            let emitted = emitter.emit(dt, &mut self.next_particle_id);
            total_emitted += emitted.len();
            new_particles.extend(emitted);
        }

        let new_count = new_particles.len();
        self.particles.extend(new_particles);
        self.sleep_counters.resize(self.particles.len(), 0);
        self.sleeping.resize(self.particles.len(), false);

        new_count
    }

    /// Update particle lifetimes.
    fn update_lifetimes(&mut self, dt: f32) {
        for p in &mut self.particles {
            if p.lifetime > 0.0 {
                p.lifetime = (p.lifetime - dt).max(0.0);
            }
        }
    }

    /// Update sleeping state for particles.
    fn update_sleeping(&mut self) {
        if !self.settings.enable_sleeping {
            return;
        }

        let threshold_sq = self.settings.sleep_velocity_threshold
            * self.settings.sleep_velocity_threshold;
        let frames_threshold = self.settings.sleep_frames_threshold;

        for i in 0..self.particles.len() {
            if self.particles[i].pinned {
                continue;
            }

            let vel_sq = self.particles[i].velocity.length_squared();
            if vel_sq < threshold_sq {
                self.sleep_counters[i] += 1;
                if self.sleep_counters[i] >= frames_threshold {
                    self.sleeping[i] = true;
                }
            } else {
                self.sleep_counters[i] = 0;
                self.sleeping[i] = false;
            }
        }
    }

    /// Wake up a particle and its neighbors.
    pub fn wake_particle(&mut self, index: usize) {
        if index < self.sleeping.len() {
            self.sleeping[index] = false;
            self.sleep_counters[index] = 0;
        }
    }

    /// Perform a single simulation step with the given delta time.
    pub fn step(&mut self, dt: f32) {
        let dt = dt * self.settings.time_scale;
        if dt <= 0.0 {
            return;
        }

        // Reset stats
        self.stats = ParticleSimulationStats::default();

        // Emit new particles
        self.stats.particles_emitted = self.emit_particles(dt);

        // Update lifetimes
        self.update_lifetimes(dt);

        // Remove expired particles
        if self.settings.auto_remove_expired {
            self.stats.particles_removed = self.remove_expired_particles();
        }

        let substep_dt = dt / self.settings.substeps as f32;

        for _substep in 0..self.settings.substeps {
            self.substep(substep_dt);
        }

        // Update sleeping
        self.update_sleeping();

        // Gather stats
        self.stats.particle_count = self.particles.len();
        self.stats.active_particle_count = self
            .particles
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.sleeping.get(*i).copied().unwrap_or(false))
            .count();
        self.stats.distance_constraint_count = self.distance_constraints.len();
        self.stats.bending_constraint_count = self.bending_constraints.len();
        self.stats.volume_constraint_count = self.volume_constraints.len();
        self.stats.broken_constraints = self
            .distance_constraints
            .iter()
            .filter(|c| c.broken)
            .count();

        // Update group cached properties
        for group in &mut self.groups {
            group.update_cached_properties(&self.particles);
        }
    }

    /// Perform a single substep of the PBD simulation.
    fn substep(&mut self, dt: f32) {
        // Phase 1: Apply external forces and predict positions
        for (i, p) in self.particles.iter_mut().enumerate() {
            if p.pinned || self.sleeping.get(i).copied().unwrap_or(false) {
                p.predicted = p.position;
                continue;
            }

            // Apply gravity
            let gravity_force = self.settings.gravity * p.mass;
            p.force_accumulator += gravity_force;

            // Semi-implicit Euler: update velocity, then predict position
            let acceleration = p.force_accumulator * p.inv_mass;
            p.velocity += acceleration * dt;
            p.velocity *= 1.0 - self.settings.damping;
            p.predicted = p.position + p.velocity * dt;

            // Clear force accumulator
            p.force_accumulator = Vec3::ZERO;
        }

        // Phase 2: Solve constraints
        let mut total_error = 0.0_f32;
        let mut max_error = 0.0_f32;
        let mut constraint_count = 0usize;

        for _iter in 0..self.settings.solver_iterations {
            // Distance constraints
            for c in &self.distance_constraints {
                if !c.broken {
                    let err = c.solve_pbd(&mut self.particles, dt);
                    total_error += err;
                    max_error = max_error.max(err);
                    constraint_count += 1;
                }
            }

            // Bending constraints
            for c in &self.bending_constraints {
                let err = c.solve_pbd(&mut self.particles, dt);
                total_error += err;
                max_error = max_error.max(err);
                constraint_count += 1;
            }

            // Volume constraints
            for c in &self.volume_constraints {
                let err = c.solve_pbd(&mut self.particles, dt);
                total_error += err;
                max_error = max_error.max(err);
                constraint_count += 1;
            }
        }

        if constraint_count > 0 {
            self.stats.avg_constraint_error = total_error / constraint_count as f32;
            self.stats.max_constraint_error = max_error;
        }

        // Phase 3: Strain limiting
        if self.settings.enable_strain_limiting {
            self.settings
                .strain_limiter
                .apply(&mut self.particles, &self.distance_constraints);
        }

        // Phase 4: Self-collision detection and resolution
        if self.settings.enable_self_collision {
            let pairs = self.self_collision.detect(&self.particles);
            self.stats.self_collision_pairs = pairs.len();
            self.self_collision
                .resolve(&mut self.particles, self.settings.self_collision_friction);
        }

        // Phase 5: Environment collision
        let mut env_collisions = 0usize;
        for i in 0..self.particles.len() {
            if self.particles[i].pinned {
                continue;
            }
            for shape in &self.collision_shapes {
                let sd = shape.signed_distance(self.particles[i].predicted);
                if sd < self.particles[i].radius {
                    let corrected =
                        shape.project_out(self.particles[i].predicted, self.particles[i].radius);
                    self.particles[i].predicted = corrected;
                    env_collisions += 1;

                    // Velocity reflection with restitution
                    let normal = shape.normal_at(self.particles[i].predicted);
                    let vn = self.particles[i].velocity.dot(normal);
                    if vn < 0.0 {
                        self.particles[i].velocity -= normal * vn * 1.5; // 0.5 restitution
                    }
                }
            }
        }
        self.stats.environment_collisions = env_collisions;

        // Phase 6: Update velocities and positions from predicted positions
        let inv_dt = if dt > EPSILON { 1.0 / dt } else { 0.0 };
        for p in &mut self.particles {
            if p.pinned {
                continue;
            }
            p.prev_position = p.position;
            p.velocity = (p.predicted - p.position) * inv_dt;
            p.position = p.predicted;
        }

        // Phase 7: Check for breakable constraints
        for c in &mut self.distance_constraints {
            c.check_break(&self.particles);
        }
    }

    /// Get the total kinetic energy of the system.
    pub fn total_kinetic_energy(&self) -> f32 {
        self.particles.iter().map(|p| p.kinetic_energy()).sum()
    }

    /// Get the center of mass of all particles.
    pub fn center_of_mass(&self) -> Vec3 {
        let mut total_mass = 0.0_f32;
        let mut com = Vec3::ZERO;
        for p in &self.particles {
            com += p.position * p.mass;
            total_mass += p.mass;
        }
        if total_mass > EPSILON {
            com / total_mass
        } else {
            Vec3::ZERO
        }
    }

    /// Apply an explosion force at the given origin with the given strength and radius.
    pub fn apply_explosion(&mut self, origin: Vec3, strength: f32, radius: f32) {
        for p in &mut self.particles {
            if p.pinned {
                continue;
            }
            let diff = p.position - origin;
            let dist = diff.length();
            if dist < radius && dist > EPSILON {
                let falloff = 1.0 - (dist / radius);
                let force = diff.normalize() * strength * falloff;
                p.apply_force(force);
            }
        }
    }

    /// Apply a vortex force around the given axis at the given origin.
    pub fn apply_vortex(&mut self, origin: Vec3, axis: Vec3, strength: f32, radius: f32) {
        let axis = axis.normalize_or_zero();
        for p in &mut self.particles {
            if p.pinned {
                continue;
            }
            let to_particle = p.position - origin;
            let projected = to_particle - axis * to_particle.dot(axis);
            let dist = projected.length();
            if dist < radius && dist > EPSILON {
                let falloff = 1.0 - (dist / radius);
                let tangent = axis.cross(projected.normalize());
                let force = tangent * strength * falloff;
                p.apply_force(force);
            }
        }
    }

    /// Compute the inertia tensor of a group of particles around their center of mass.
    pub fn compute_group_inertia(&self, group_index: usize) -> Mat3 {
        if group_index >= self.groups.len() {
            return Mat3::ZERO;
        }

        let group = &self.groups[group_index];
        let com = group.center_of_mass;
        let mut ixx = 0.0_f32;
        let mut iyy = 0.0_f32;
        let mut izz = 0.0_f32;
        let mut ixy = 0.0_f32;
        let mut ixz = 0.0_f32;
        let mut iyz = 0.0_f32;

        for &idx in &group.particle_indices {
            if idx >= self.particles.len() {
                continue;
            }
            let p = &self.particles[idx];
            let r = p.position - com;
            let m = p.mass;

            ixx += m * (r.y * r.y + r.z * r.z);
            iyy += m * (r.x * r.x + r.z * r.z);
            izz += m * (r.x * r.x + r.y * r.y);
            ixy -= m * r.x * r.y;
            ixz -= m * r.x * r.z;
            iyz -= m * r.y * r.z;
        }

        Mat3::from_cols(
            Vec3::new(ixx, ixy, ixz),
            Vec3::new(ixy, iyy, iyz),
            Vec3::new(ixz, iyz, izz),
        )
    }

    /// Clear all particles, constraints, and emitters.
    pub fn clear(&mut self) {
        self.particles.clear();
        self.distance_constraints.clear();
        self.bending_constraints.clear();
        self.volume_constraints.clear();
        self.collision_shapes.clear();
        self.groups.clear();
        self.emitters.clear();
        self.sleep_counters.clear();
        self.sleeping.clear();
        self.next_particle_id = 0;
        self.next_constraint_id = 0;
        self.stats = ParticleSimulationStats::default();
    }

    /// Get the number of alive particles.
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Get the number of active (non-sleeping) particles.
    pub fn active_particle_count(&self) -> usize {
        self.particles
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.sleeping.get(*i).copied().unwrap_or(false))
            .count()
    }

    /// Get a read-only reference to a particle by index.
    pub fn get_particle(&self, index: usize) -> Option<&Particle> {
        self.particles.get(index)
    }

    /// Get a mutable reference to a particle by index.
    pub fn get_particle_mut(&mut self, index: usize) -> Option<&mut Particle> {
        self.particles.get_mut(index)
    }

    /// Apply a force field (function of position) to all dynamic particles.
    pub fn apply_force_field(&mut self, field: &dyn Fn(Vec3) -> Vec3) {
        for p in &mut self.particles {
            if !p.pinned {
                let force = field(p.position);
                p.apply_force(force);
            }
        }
    }

    /// Get positions of all particles as a contiguous slice (for rendering).
    pub fn positions(&self) -> Vec<Vec3> {
        self.particles.iter().map(|p| p.position).collect()
    }

    /// Get velocities of all particles.
    pub fn velocities(&self) -> Vec<Vec3> {
        self.particles.iter().map(|p| p.velocity).collect()
    }
}

// ---------------------------------------------------------------------------
// ECS Components
// ---------------------------------------------------------------------------

/// ECS component wrapping a particle simulation.
#[derive(Debug)]
pub struct ParticlePhysicsComponent {
    /// The underlying simulation.
    pub simulation: ParticleSimulation,
    /// Whether this component is enabled.
    pub enabled: bool,
    /// Entity ID that owns this component (for reference).
    pub entity_id: u64,
}

impl ParticlePhysicsComponent {
    /// Create a new component with a fresh simulation.
    pub fn new(entity_id: u64) -> Self {
        Self {
            simulation: ParticleSimulation::new(),
            enabled: true,
            entity_id,
        }
    }

    /// Create a new component with custom settings.
    pub fn with_settings(entity_id: u64, settings: ParticleSimulationSettings) -> Self {
        Self {
            simulation: ParticleSimulation::with_settings(settings),
            enabled: true,
            entity_id,
        }
    }
}

/// ECS system that updates all particle physics components.
pub struct ParticlePhysicsSystem {
    /// Components managed by this system.
    pub components: Vec<ParticlePhysicsComponent>,
}

impl ParticlePhysicsSystem {
    /// Create a new particle physics system.
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
        }
    }

    /// Add a component and return its index.
    pub fn add_component(&mut self, component: ParticlePhysicsComponent) -> usize {
        let idx = self.components.len();
        self.components.push(component);
        idx
    }

    /// Update all enabled components with the given delta time.
    pub fn update(&mut self, dt: f32) {
        for component in &mut self.components {
            if component.enabled {
                component.simulation.step(dt);
            }
        }
    }

    /// Get combined statistics from all components.
    pub fn combined_stats(&self) -> ParticleSimulationStats {
        let mut combined = ParticleSimulationStats::default();
        for component in &self.components {
            let s = &component.simulation.stats;
            combined.particle_count += s.particle_count;
            combined.active_particle_count += s.active_particle_count;
            combined.distance_constraint_count += s.distance_constraint_count;
            combined.bending_constraint_count += s.bending_constraint_count;
            combined.volume_constraint_count += s.volume_constraint_count;
            combined.self_collision_pairs += s.self_collision_pairs;
            combined.environment_collisions += s.environment_collisions;
            combined.broken_constraints += s.broken_constraints;
            combined.particles_removed += s.particles_removed;
            combined.particles_emitted += s.particles_emitted;
        }
        combined
    }
}
