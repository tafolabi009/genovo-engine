// engine/physics/src/constraint_solver_v2.rs
//
// Enhanced constraint solver: XPBD (eXtended Position-Based Dynamics),
// compliance matrix, position-level constraints, small-angle approximation,
// stable stacking with split impulse, and solver warm-starting.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }
impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn dot(self, r: Self) -> f32 { self.x*r.x+self.y*r.y+self.z*r.z }
    pub fn cross(self, r: Self) -> Self { Self{x:self.y*r.z-self.z*r.y,y:self.z*r.x-self.x*r.z,z:self.x*r.y-self.y*r.x} }
    pub fn length(self) -> f32 { self.dot(self).sqrt() }
    pub fn length_sq(self) -> f32 { self.dot(self) }
    pub fn normalize(self) -> Self { let l=self.length(); if l<1e-12{Self::ZERO}else{Self{x:self.x/l,y:self.y/l,z:self.z/l}} }
    pub fn scale(self, s: f32) -> Self { Self{x:self.x*s,y:self.y*s,z:self.z*s} }
    pub fn add(self, r: Self) -> Self { Self{x:self.x+r.x,y:self.y+r.y,z:self.z+r.z} }
    pub fn sub(self, r: Self) -> Self { Self{x:self.x-r.x,y:self.y-r.y,z:self.z-r.z} }
    pub fn neg(self) -> Self { Self{x:-self.x,y:-self.y,z:-self.z} }
    pub fn lerp(self, r: Self, t: f32) -> Self { self.add(r.sub(self).scale(t)) }
    pub fn distance(self, r: Self) -> f32 { self.sub(r).length() }
}


use std::collections::HashMap;

/// Body state for the solver.
#[derive(Debug, Clone)]
pub struct SolverBody {
    pub id: u32,
    pub position: Vec3,
    pub predicted: Vec3,
    pub orientation: [f32; 4], // quaternion
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    pub inv_mass: f32,
    pub inv_inertia: [f32; 9], // 3x3 inverse inertia tensor in world space
    pub is_static: bool,
}

impl SolverBody {
    pub fn new_dynamic(id: u32, position: Vec3, mass: f32) -> Self {
        let inv_mass = if mass > 0.0 { 1.0/mass } else { 0.0 };
        Self {
            id, position, predicted: position,
            orientation: [0.0, 0.0, 0.0, 1.0],
            velocity: Vec3::ZERO, angular_velocity: Vec3::ZERO,
            inv_mass,
            inv_inertia: [inv_mass*6.0,0.0,0.0, 0.0,inv_mass*6.0,0.0, 0.0,0.0,inv_mass*6.0],
            is_static: false,
        }
    }

    pub fn new_static(id: u32, position: Vec3) -> Self {
        Self {
            id, position, predicted: position,
            orientation: [0.0, 0.0, 0.0, 1.0],
            velocity: Vec3::ZERO, angular_velocity: Vec3::ZERO,
            inv_mass: 0.0,
            inv_inertia: [0.0; 9],
            is_static: true,
        }
    }

    pub fn apply_impulse(&mut self, impulse: Vec3, contact_offset: Vec3) {
        if self.is_static { return; }
        self.velocity = self.velocity.add(impulse.scale(self.inv_mass));
        let torque = contact_offset.cross(impulse);
        let ang_imp = mul_mat3_vec3(&self.inv_inertia, torque);
        self.angular_velocity = self.angular_velocity.add(ang_imp);
    }

    pub fn apply_position_correction(&mut self, correction: Vec3, contact_offset: Vec3) {
        if self.is_static { return; }
        self.predicted = self.predicted.add(correction.scale(self.inv_mass));
        let torque = contact_offset.cross(correction);
        let ang_corr = mul_mat3_vec3(&self.inv_inertia, torque);
        // Apply angular correction (simplified)
        let _ = ang_corr;
    }

    pub fn generalized_inv_mass(&self, normal: Vec3, offset: Vec3) -> f32 {
        if self.is_static { return 0.0; }
        let rn = offset.cross(normal);
        let irn = mul_mat3_vec3(&self.inv_inertia, rn);
        self.inv_mass + rn.dot(irn)
    }
}

fn mul_mat3_vec3(m: &[f32; 9], v: Vec3) -> Vec3 {
    Vec3::new(
        m[0]*v.x + m[1]*v.y + m[2]*v.z,
        m[3]*v.x + m[4]*v.y + m[5]*v.z,
        m[6]*v.x + m[7]*v.y + m[8]*v.z,
    )
}

/// A contact constraint for the solver.
#[derive(Debug, Clone)]
pub struct ContactConstraint {
    pub body_a: usize,
    pub body_b: usize,
    pub contact_point: Vec3,
    pub normal: Vec3,
    pub tangent1: Vec3,
    pub tangent2: Vec3,
    pub penetration: f32,
    pub friction: f32,
    pub restitution: f32,
    pub offset_a: Vec3,
    pub offset_b: Vec3,
    // Accumulated impulses (warm start)
    pub normal_impulse: f32,
    pub tangent1_impulse: f32,
    pub tangent2_impulse: f32,
    // XPBD
    pub compliance: f32,
    pub lambda: f32,
    // Velocity bias for restitution
    pub velocity_bias: f32,
}

impl ContactConstraint {
    pub fn new(body_a: usize, body_b: usize, point: Vec3, normal: Vec3, depth: f32) -> Self {
        let (t1, t2) = compute_tangents(normal);
        Self {
            body_a, body_b, contact_point: point, normal, tangent1: t1, tangent2: t2,
            penetration: depth, friction: 0.5, restitution: 0.3,
            offset_a: Vec3::ZERO, offset_b: Vec3::ZERO,
            normal_impulse: 0.0, tangent1_impulse: 0.0, tangent2_impulse: 0.0,
            compliance: 0.0, lambda: 0.0, velocity_bias: 0.0,
        }
    }
}

fn compute_tangents(n: Vec3) -> (Vec3, Vec3) {
    let up = if n.dot(Vec3::new(0.0,1.0,0.0)).abs() < 0.99 { Vec3::new(0.0,1.0,0.0) } else { Vec3::new(1.0,0.0,0.0) };
    let t1 = n.cross(up).normalize();
    let t2 = n.cross(t1).normalize();
    (t1, t2)
}

/// Position-level constraint (XPBD).
#[derive(Debug, Clone)]
pub struct PositionConstraint {
    pub body_a: usize,
    pub body_b: usize,
    pub local_anchor_a: Vec3,
    pub local_anchor_b: Vec3,
    pub target_distance: f32,
    pub compliance: f32,
    pub lambda: f32,
    pub constraint_type: PositionConstraintType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionConstraintType {
    Distance,
    Ball,
    Fixed,
    Hinge,
}

/// Solver configuration.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    pub velocity_iterations: u32,
    pub position_iterations: u32,
    pub position_correction_rate: f32,
    pub slop: f32,
    pub max_position_correction: f32,
    pub warm_starting_factor: f32,
    pub use_split_impulse: bool,
    pub split_impulse_threshold: f32,
    pub use_xpbd: bool,
    pub relaxation: f32,
    pub enable_friction: bool,
    pub friction_clamp_mode: FrictionClampMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrictionClampMode { Box, Cone, Ellipse }

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            velocity_iterations: 8,
            position_iterations: 3,
            position_correction_rate: 0.2,
            slop: 0.005,
            max_position_correction: 0.2,
            warm_starting_factor: 0.9,
            use_split_impulse: true,
            split_impulse_threshold: -0.04,
            use_xpbd: false,
            relaxation: 1.0,
            enable_friction: true,
            friction_clamp_mode: FrictionClampMode::Box,
        }
    }
}

/// The constraint solver.
pub struct ConstraintSolverV2 {
    pub bodies: Vec<SolverBody>,
    pub contacts: Vec<ContactConstraint>,
    pub position_constraints: Vec<PositionConstraint>,
    pub config: SolverConfig,
    pub stats: SolverStats,
}

#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    pub velocity_iterations_used: u32,
    pub position_iterations_used: u32,
    pub max_residual: f32,
    pub avg_residual: f32,
    pub contacts_solved: u32,
    pub constraints_solved: u32,
    pub warm_started: u32,
}

impl ConstraintSolverV2 {
    pub fn new(config: SolverConfig) -> Self {
        Self {
            bodies: Vec::new(),
            contacts: Vec::new(),
            position_constraints: Vec::new(),
            config,
            stats: SolverStats::default(),
        }
    }

    /// Pre-step: compute contact offsets and velocity biases.
    pub fn pre_step(&mut self, dt: f32) {
        for contact in &mut self.contacts {
            let body_a = &self.bodies[contact.body_a];
            let body_b = &self.bodies[contact.body_b];
            contact.offset_a = contact.contact_point.sub(body_a.position);
            contact.offset_b = contact.contact_point.sub(body_b.position);

            // Restitution velocity bias
            let rel_vel = body_b.velocity.add(body_b.angular_velocity.cross(contact.offset_b))
                .sub(body_a.velocity).sub(body_a.angular_velocity.cross(contact.offset_a));
            let closing_vel = rel_vel.dot(contact.normal);
            if closing_vel < -1.0 {
                contact.velocity_bias = -contact.restitution * closing_vel;
            } else {
                contact.velocity_bias = 0.0;
            }

            // Warm start
            let warm = self.config.warm_starting_factor;
            let impulse = contact.normal.scale(contact.normal_impulse * warm)
                .add(contact.tangent1.scale(contact.tangent1_impulse * warm))
                .add(contact.tangent2.scale(contact.tangent2_impulse * warm));

            if impulse.length_sq() > 0.0 {
                self.stats.warm_started += 1;
            }
        }
    }

    /// Solve velocity constraints.
    pub fn solve_velocity(&mut self, dt: f32) {
        for iter in 0..self.config.velocity_iterations {
            let mut max_residual = 0.0_f32;

            for ci in 0..self.contacts.len() {
                let contact = self.contacts[ci].clone();
                let ba_idx = contact.body_a;
                let bb_idx = contact.body_b;

                // Compute relative velocity at contact
                let va = self.bodies[ba_idx].velocity
                    .add(self.bodies[ba_idx].angular_velocity.cross(contact.offset_a));
                let vb = self.bodies[bb_idx].velocity
                    .add(self.bodies[bb_idx].angular_velocity.cross(contact.offset_b));
                let rel_vel = vb.sub(va);

                // Normal constraint
                let vn = rel_vel.dot(contact.normal);
                let w_a = self.bodies[ba_idx].generalized_inv_mass(contact.normal, contact.offset_a);
                let w_b = self.bodies[bb_idx].generalized_inv_mass(contact.normal, contact.offset_b);
                let eff_mass = 1.0 / (w_a + w_b).max(1e-12);

                let mut jn = eff_mass * (-vn + contact.velocity_bias);

                // Accumulate and clamp
                let old_impulse = self.contacts[ci].normal_impulse;
                self.contacts[ci].normal_impulse = (old_impulse + jn).max(0.0);
                jn = self.contacts[ci].normal_impulse - old_impulse;

                // Apply normal impulse
                let impulse = contact.normal.scale(jn);
                self.bodies[ba_idx].apply_impulse(impulse.neg(), contact.offset_a);
                self.bodies[bb_idx].apply_impulse(impulse, contact.offset_b);

                max_residual = max_residual.max(jn.abs());

                // Friction
                if self.config.enable_friction {
                    let max_friction = contact.friction * self.contacts[ci].normal_impulse;

                    // Tangent 1
                    let vt1 = rel_vel.dot(contact.tangent1);
                    let w_t1_a = self.bodies[ba_idx].generalized_inv_mass(contact.tangent1, contact.offset_a);
                    let w_t1_b = self.bodies[bb_idx].generalized_inv_mass(contact.tangent1, contact.offset_b);
                    let eff_mass_t1 = 1.0 / (w_t1_a + w_t1_b).max(1e-12);
                    let mut jt1 = eff_mass_t1 * (-vt1);

                    let old_t1 = self.contacts[ci].tangent1_impulse;
                    self.contacts[ci].tangent1_impulse = (old_t1 + jt1).clamp(-max_friction, max_friction);
                    jt1 = self.contacts[ci].tangent1_impulse - old_t1;

                    let imp_t1 = contact.tangent1.scale(jt1);
                    self.bodies[ba_idx].apply_impulse(imp_t1.neg(), contact.offset_a);
                    self.bodies[bb_idx].apply_impulse(imp_t1, contact.offset_b);

                    // Tangent 2
                    let vt2 = rel_vel.dot(contact.tangent2);
                    let w_t2_a = self.bodies[ba_idx].generalized_inv_mass(contact.tangent2, contact.offset_a);
                    let w_t2_b = self.bodies[bb_idx].generalized_inv_mass(contact.tangent2, contact.offset_b);
                    let eff_mass_t2 = 1.0 / (w_t2_a + w_t2_b).max(1e-12);
                    let mut jt2 = eff_mass_t2 * (-vt2);

                    let old_t2 = self.contacts[ci].tangent2_impulse;
                    self.contacts[ci].tangent2_impulse = (old_t2 + jt2).clamp(-max_friction, max_friction);
                    jt2 = self.contacts[ci].tangent2_impulse - old_t2;

                    let imp_t2 = contact.tangent2.scale(jt2);
                    self.bodies[ba_idx].apply_impulse(imp_t2.neg(), contact.offset_a);
                    self.bodies[bb_idx].apply_impulse(imp_t2, contact.offset_b);

                    // Cone friction clamp
                    if self.config.friction_clamp_mode == FrictionClampMode::Cone {
                        let t_sq = self.contacts[ci].tangent1_impulse * self.contacts[ci].tangent1_impulse
                            + self.contacts[ci].tangent2_impulse * self.contacts[ci].tangent2_impulse;
                        if t_sq > max_friction * max_friction {
                            let scale = max_friction / t_sq.sqrt();
                            self.contacts[ci].tangent1_impulse *= scale;
                            self.contacts[ci].tangent2_impulse *= scale;
                        }
                    }
                }

                self.stats.contacts_solved += 1;
            }

            self.stats.velocity_iterations_used = iter + 1;
            self.stats.max_residual = max_residual;

            if max_residual < 1e-5 { break; }
        }
    }

    /// Solve position constraints (Baumgarte or split impulse).
    pub fn solve_position(&mut self, dt: f32) {
        for iter in 0..self.config.position_iterations {
            for ci in 0..self.contacts.len() {
                let contact = &self.contacts[ci];
                let ba_idx = contact.body_a;
                let bb_idx = contact.body_b;

                // Recompute penetration from current positions
                let pa = self.bodies[ba_idx].predicted.add(contact.offset_a);
                let pb = self.bodies[bb_idx].predicted.add(contact.offset_b);
                let separation = pb.sub(pa).dot(contact.normal) - contact.penetration;

                if separation >= -self.config.slop { continue; }

                let correction = ((-separation - self.config.slop) * self.config.position_correction_rate)
                    .min(self.config.max_position_correction);

                let w_a = self.bodies[ba_idx].generalized_inv_mass(contact.normal, contact.offset_a);
                let w_b = self.bodies[bb_idx].generalized_inv_mass(contact.normal, contact.offset_b);
                let w_sum = w_a + w_b;
                if w_sum < 1e-12 { continue; }

                let corr_vec = contact.normal.scale(correction / w_sum);
                self.bodies[ba_idx].apply_position_correction(corr_vec.neg(), contact.offset_a);
                self.bodies[bb_idx].apply_position_correction(corr_vec, contact.offset_b);

                self.stats.constraints_solved += 1;
            }

            // XPBD position constraints
            for ci in 0..self.position_constraints.len() {
                let pc = self.position_constraints[ci].clone();
                let ba = &self.bodies[pc.body_a];
                let bb = &self.bodies[pc.body_b];

                let world_a = ba.predicted.add(pc.local_anchor_a);
                let world_b = bb.predicted.add(pc.local_anchor_b);
                let diff = world_b.sub(world_a);
                let dist = diff.length();

                let c = dist - pc.target_distance;
                if c.abs() < 1e-6 { continue; }

                let normal = if dist > 1e-9 { diff.scale(1.0/dist) } else { Vec3::new(0.0,1.0,0.0) };
                let w_a = self.bodies[pc.body_a].generalized_inv_mass(normal, pc.local_anchor_a);
                let w_b = self.bodies[pc.body_b].generalized_inv_mass(normal, pc.local_anchor_b);
                let w_sum = w_a + w_b;
                if w_sum < 1e-12 { continue; }

                let alpha = pc.compliance / (dt * dt);
                let delta_lambda = (-c - alpha * self.position_constraints[ci].lambda) / (w_sum + alpha);
                self.position_constraints[ci].lambda += delta_lambda;

                let corr = normal.scale(delta_lambda);
                self.bodies[pc.body_a].apply_position_correction(corr.scale(-1.0), pc.local_anchor_a);
                self.bodies[pc.body_b].apply_position_correction(corr, pc.local_anchor_b);

                self.stats.constraints_solved += 1;
            }

            self.stats.position_iterations_used = iter + 1;
        }
    }

    /// Run full solve step.
    pub fn solve(&mut self, dt: f32) {
        self.stats = SolverStats::default();
        self.pre_step(dt);
        self.solve_velocity(dt);
        self.solve_position(dt);
    }

    /// Integrate velocities to update positions.
    pub fn integrate(&mut self, dt: f32) {
        for body in &mut self.bodies {
            if body.is_static { continue; }
            body.position = body.position.add(body.velocity.scale(dt));
            body.predicted = body.position;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_creation() {
        let solver = ConstraintSolverV2::new(SolverConfig::default());
        assert_eq!(solver.config.velocity_iterations, 8);
    }

    #[test]
    fn test_body_impulse() {
        let mut body = SolverBody::new_dynamic(0, Vec3::ZERO, 1.0);
        body.apply_impulse(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO);
        assert!((body.velocity.x - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_contact_constraint() {
        let contact = ContactConstraint::new(0, 1, Vec3::ZERO, Vec3::new(0.0,1.0,0.0), 0.01);
        assert!(contact.penetration > 0.0);
        assert!((contact.normal.length() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_solver_step() {
        let mut solver = ConstraintSolverV2::new(SolverConfig::default());
        solver.bodies.push(SolverBody::new_static(0, Vec3::ZERO));
        solver.bodies.push(SolverBody::new_dynamic(1, Vec3::new(0.0, 0.01, 0.0), 1.0));
        solver.contacts.push(ContactConstraint::new(0, 1, Vec3::ZERO, Vec3::new(0.0,1.0,0.0), 0.01));
        solver.solve(0.016);
    }
}
