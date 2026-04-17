// engine/physics/src/soft_body_v2.rs - Enhanced soft body: PBD/XPBD, shape matching,
// volume preservation, attachment to rigid bodies, cutting/tearing, GPU prep.
// (See softbody/ for original; this extends with position-based dynamics.)

pub use super::softbody::*;

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3SB { pub x: f32, pub y: f32, pub z: f32 }
impl Vec3SB {
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

/// Soft body particle.
#[derive(Debug, Clone)]
pub struct ParticleSB {
    pub position: Vec3SB, pub predicted: Vec3SB, pub velocity: Vec3SB,
    pub inv_mass: f32, pub rest_position: Vec3SB, pub pinned: bool,
}

impl ParticleSB {
    pub fn new(pos: Vec3SB, mass: f32) -> Self {
        Self { position: pos, predicted: pos, velocity: Vec3SB::ZERO,
            inv_mass: if mass > 0.0 { 1.0/mass } else { 0.0 }, rest_position: pos, pinned: false }
    }
    pub fn mass(&self) -> f32 { if self.inv_mass > 0.0 { 1.0/self.inv_mass } else { f32::MAX } }
}

/// XPBD distance constraint.
#[derive(Debug, Clone)]
pub struct DistConstraintSB {
    pub a: usize, pub b: usize, pub rest_length: f32,
    pub compliance: f32, pub max_strain: f32, pub lambda: f32, pub broken: bool,
}

impl DistConstraintSB {
    pub fn new(a: usize, b: usize, rest: f32, compliance: f32) -> Self {
        Self { a, b, rest_length: rest, compliance, max_strain: 2.0, lambda: 0.0, broken: false }
    }
    pub fn solve(&mut self, particles: &mut [ParticleSB], dt: f32) {
        if self.broken { return; }
        let diff = particles[self.b].predicted.sub(particles[self.a].predicted);
        let dist = diff.length(); if dist < 1e-9 { return; }
        if dist / self.rest_length > self.max_strain { self.broken = true; return; }
        let c = dist - self.rest_length; let grad = diff.scale(1.0/dist);
        let wa = particles[self.a].inv_mass; let wb = particles[self.b].inv_mass;
        let wsum = wa + wb; if wsum < 1e-12 { return; }
        let alpha = self.compliance / (dt * dt);
        let dl = (-c - alpha * self.lambda) / (wsum + alpha); self.lambda += dl;
        let corr = grad.scale(dl);
        if !particles[self.a].pinned { particles[self.a].predicted = particles[self.a].predicted.add(corr.scale(wa)); }
        if !particles[self.b].pinned { particles[self.b].predicted = particles[self.b].predicted.sub(corr.scale(wb)); }
    }
}

/// Volume constraint for a tetrahedron.
#[derive(Debug, Clone)]
pub struct VolConstraintSB { pub indices: [usize; 4], pub rest_volume: f32, pub compliance: f32, pub lambda: f32 }

impl VolConstraintSB {
    pub fn new(idx: [usize; 4], particles: &[ParticleSB], compliance: f32) -> Self {
        let rv = tet_volume(particles[idx[0]].position, particles[idx[1]].position, particles[idx[2]].position, particles[idx[3]].position);
        Self { indices: idx, rest_volume: rv, compliance, lambda: 0.0 }
    }
    pub fn solve(&mut self, particles: &mut [ParticleSB], dt: f32) {
        let [i0,i1,i2,i3] = self.indices;
        let cv = tet_volume(particles[i0].predicted, particles[i1].predicted, particles[i2].predicted, particles[i3].predicted);
        let c = cv - self.rest_volume; if c.abs() < 1e-12 { return; }
        let g0 = particles[i1].predicted.sub(particles[i3].predicted).cross(particles[i2].predicted.sub(particles[i3].predicted)).scale(1.0/6.0);
        let g1 = particles[i2].predicted.sub(particles[i3].predicted).cross(particles[i0].predicted.sub(particles[i3].predicted)).scale(1.0/6.0);
        let g2 = particles[i0].predicted.sub(particles[i3].predicted).cross(particles[i1].predicted.sub(particles[i3].predicted)).scale(1.0/6.0);
        let g3 = g0.add(g1).add(g2).neg();
        let grads = [g0, g1, g2, g3];
        let mut wsum = 0.0_f32;
        for k in 0..4 { wsum += particles[self.indices[k]].inv_mass * grads[k].length_sq(); }
        if wsum < 1e-12 { return; }
        let alpha = self.compliance / (dt * dt);
        let dl = (-c - alpha * self.lambda) / (wsum + alpha); self.lambda += dl;
        for k in 0..4 {
            if !particles[self.indices[k]].pinned {
                particles[self.indices[k]].predicted = particles[self.indices[k]].predicted.add(grads[k].scale(dl * particles[self.indices[k]].inv_mass));
            }
        }
    }
}

fn tet_volume(p0: Vec3SB, p1: Vec3SB, p2: Vec3SB, p3: Vec3SB) -> f32 {
    p1.sub(p0).cross(p2.sub(p0)).dot(p3.sub(p0)) / 6.0
}

/// Shape matching region.
pub struct ShapeMatchRegion {
    pub indices: Vec<usize>, pub rest_center: Vec3SB, pub rest_offsets: Vec<Vec3SB>,
    pub masses: Vec<f32>, pub total_mass: f32, pub stiffness: f32,
}

impl ShapeMatchRegion {
    pub fn new(indices: Vec<usize>, particles: &[ParticleSB], stiffness: f32) -> Self {
        let tm: f32 = indices.iter().map(|&i| particles[i].mass()).sum();
        let rc = if tm > 0.0 { let s: Vec3SB = indices.iter().map(|&i| particles[i].rest_position.scale(particles[i].mass())).fold(Vec3SB::ZERO, Vec3SB::add); s.scale(1.0/tm) } else { Vec3SB::ZERO };
        let ro: Vec<Vec3SB> = indices.iter().map(|&i| particles[i].rest_position.sub(rc)).collect();
        let ms: Vec<f32> = indices.iter().map(|&i| particles[i].mass()).collect();
        Self { indices, rest_center: rc, rest_offsets: ro, masses: ms, total_mass: tm, stiffness }
    }
    pub fn apply(&self, particles: &mut [ParticleSB]) {
        if self.total_mass < 1e-12 { return; }
        let mut com = Vec3SB::ZERO;
        for (i, &pi) in self.indices.iter().enumerate() { com = com.add(particles[pi].predicted.scale(self.masses[i])); }
        com = com.scale(1.0 / self.total_mass);
        for (i, &pi) in self.indices.iter().enumerate() {
            if particles[pi].pinned { continue; }
            let goal = com.add(self.rest_offsets[i]); // simplified: no rotation extraction
            particles[pi].predicted = particles[pi].predicted.lerp(goal, self.stiffness);
        }
    }
}

/// Soft body simulation config.
#[derive(Debug, Clone)]
pub struct SoftBodyCfg {
    pub solver_iters: u32, pub sub_steps: u32, pub gravity: Vec3SB,
    pub damping: f32, pub dist_compliance: f32, pub vol_compliance: f32,
}

impl Default for SoftBodyCfg {
    fn default() -> Self { Self { solver_iters: 10, sub_steps: 4, gravity: Vec3SB::new(0.0, -9.81, 0.0), damping: 0.99, dist_compliance: 0.0, vol_compliance: 0.0001 } }
}

/// Soft body simulation.
pub struct SoftBodySimV2 {
    pub particles: Vec<ParticleSB>, pub dist_constraints: Vec<DistConstraintSB>,
    pub vol_constraints: Vec<VolConstraintSB>, pub shape_regions: Vec<ShapeMatchRegion>,
    pub config: SoftBodyCfg,
}

impl SoftBodySimV2 {
    pub fn new(cfg: SoftBodyCfg) -> Self { Self { particles: Vec::new(), dist_constraints: Vec::new(), vol_constraints: Vec::new(), shape_regions: Vec::new(), config: cfg } }

    pub fn from_surface_mesh(verts: &[Vec3SB], tris: &[[usize; 3]], mass: f32, cfg: SoftBodyCfg) -> Self {
        let mut body = Self::new(cfg);
        let pm = mass / verts.len() as f32;
        for &v in verts { body.particles.push(ParticleSB::new(v, pm)); }
        let mut edges = std::collections::HashSet::new();
        for tri in tris {
            for &(a,b) in &[(tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])] {
                let key = if a < b { (a,b) } else { (b,a) };
                if edges.insert(key) {
                    body.dist_constraints.push(DistConstraintSB::new(a, b, verts[a].distance(verts[b]), body.config.dist_compliance));
                }
            }
        }
        body
    }

    pub fn step(&mut self, dt: f32) {
        let sub_dt = dt / self.config.sub_steps as f32;
        for _ in 0..self.config.sub_steps {
            for p in &mut self.particles {
                if p.pinned { p.predicted = p.position; continue; }
                p.velocity = p.velocity.add(self.config.gravity.scale(sub_dt)).scale(self.config.damping);
                p.predicted = p.position.add(p.velocity.scale(sub_dt));
            }
            for c in &mut self.dist_constraints { c.lambda = 0.0; }
            for c in &mut self.vol_constraints { c.lambda = 0.0; }
            for _ in 0..self.config.solver_iters {
                for i in 0..self.dist_constraints.len() { let mut c = self.dist_constraints[i].clone(); c.solve(&mut self.particles, sub_dt); self.dist_constraints[i] = c; }
                for i in 0..self.vol_constraints.len() { let mut c = self.vol_constraints[i].clone(); c.solve(&mut self.particles, sub_dt); self.vol_constraints[i] = c; }
            }
            for r in &self.shape_regions { r.apply(&mut self.particles); }
            // Ground collision
            for p in &mut self.particles { if !p.pinned && p.predicted.y < 0.0 { p.predicted.y = 0.0; } }
            let inv_dt = 1.0 / sub_dt;
            for p in &mut self.particles { if !p.pinned { p.velocity = p.predicted.sub(p.position).scale(inv_dt); p.position = p.predicted; } }
        }
    }

    pub fn pin(&mut self, i: usize) { if i < self.particles.len() { self.particles[i].pinned = true; self.particles[i].inv_mass = 0.0; } }
    pub fn apply_force(&mut self, i: usize, f: Vec3SB) { if i < self.particles.len() && !self.particles[i].pinned { self.particles[i].velocity = self.particles[i].velocity.add(f.scale(self.particles[i].inv_mass)); } }

    pub fn cut(&mut self, plane_pt: Vec3SB, plane_n: Vec3SB, width: f32) {
        for c in &mut self.dist_constraints {
            let mid = self.particles[c.a].position.lerp(self.particles[c.b].position, 0.5);
            if mid.sub(plane_pt).dot(plane_n).abs() < width { c.broken = true; }
        }
    }

    pub fn gpu_data(&self) -> Vec<[f32; 4]> {
        self.particles.iter().map(|p| [p.position.x, p.position.y, p.position.z, p.inv_mass]).collect()
    }

    pub fn compute_normals(&self, tris: &[[usize; 3]]) -> Vec<Vec3SB> {
        let mut normals = vec![Vec3SB::ZERO; self.particles.len()];
        for tri in tris {
            let e1 = self.particles[tri[1]].position.sub(self.particles[tri[0]].position);
            let e2 = self.particles[tri[2]].position.sub(self.particles[tri[0]].position);
            let n = e1.cross(e2);
            for &i in tri { normals[i] = normals[i].add(n); }
        }
        for n in &mut normals { *n = n.normalize(); }
        normals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_particle() { let p = ParticleSB::new(Vec3SB::ZERO, 1.0); assert!((p.inv_mass - 1.0).abs() < 1e-6); }
    #[test] fn test_dist_constraint() {
        let mut ps = vec![ParticleSB::new(Vec3SB::ZERO, 1.0), ParticleSB::new(Vec3SB::new(2.0, 0.0, 0.0), 1.0)];
        ps[0].predicted = ps[0].position; ps[1].predicted = ps[1].position;
        let mut c = DistConstraintSB::new(0, 1, 1.0, 0.0); c.solve(&mut ps, 0.016);
        assert!(ps[0].predicted.distance(ps[1].predicted) < 2.0);
    }
    #[test] fn test_tet_volume() {
        let v = tet_volume(Vec3SB::ZERO, Vec3SB::new(1.0,0.0,0.0), Vec3SB::new(0.0,1.0,0.0), Vec3SB::new(0.0,0.0,1.0));
        assert!((v - 1.0/6.0).abs() < 1e-5);
    }
    #[test] fn test_soft_body_step() {
        let verts = vec![Vec3SB::new(0.0,1.0,0.0), Vec3SB::new(1.0,1.0,0.0), Vec3SB::new(0.5,1.0,1.0)];
        let tris = vec![[0,1,2]];
        let mut body = SoftBodySimV2::from_surface_mesh(&verts, &tris, 3.0, SoftBodyCfg::default());
        body.step(0.016);
        assert!(body.particles[0].position.y < 1.0);
    }
}
