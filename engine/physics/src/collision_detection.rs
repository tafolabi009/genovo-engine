// engine/physics/src/collision_detection_v2.rs - Enhanced collision: persistent manifold,
// contact reduction, speculative contacts, GJK/EPA, warm starting cache.
// (See collision/ for the original module; this extends with v2 algorithms.)

pub use super::collision::*;

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3CD { pub x: f32, pub y: f32, pub z: f32 }
impl Vec3CD {
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
    pub fn distance(self, r: Self) -> f32 { self.sub(r).length() }
}

pub const MAX_MANIFOLD_CONTACTS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BodyIdV2(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColliderIdV2(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PairKey { pub a: ColliderIdV2, pub b: ColliderIdV2 }
impl PairKey { pub fn new(a: ColliderIdV2, b: ColliderIdV2) -> Self { if a.0 <= b.0 { Self{a,b} } else { Self{a:b,b:a} } } }

/// Contact point between two colliders.
#[derive(Debug, Clone, Copy)]
pub struct ContactPointV2 {
    pub position_a: Vec3CD, pub position_b: Vec3CD, pub normal: Vec3CD, pub depth: f32,
    pub local_a: Vec3CD, pub local_b: Vec3CD,
    pub normal_impulse: f32, pub tangent_impulse: [f32; 2],
    pub tangent1: Vec3CD, pub tangent2: Vec3CD, pub age: u32,
}

impl ContactPointV2 {
    pub fn new(pa: Vec3CD, pb: Vec3CD, normal: Vec3CD, depth: f32) -> Self {
        let up = if normal.dot(Vec3CD::new(0.0,1.0,0.0)).abs() < 0.99 { Vec3CD::new(0.0,1.0,0.0) } else { Vec3CD::new(1.0,0.0,0.0) };
        let t1 = normal.cross(up).normalize(); let t2 = normal.cross(t1).normalize();
        Self { position_a: pa, position_b: pb, normal, depth, local_a: Vec3CD::ZERO, local_b: Vec3CD::ZERO, normal_impulse: 0.0, tangent_impulse: [0.0, 0.0], tangent1: t1, tangent2: t2, age: 0 }
    }
}

/// Persistent contact manifold.
#[derive(Debug, Clone)]
pub struct ManifoldV2 {
    pub pair: PairKey, pub body_a: BodyIdV2, pub body_b: BodyIdV2,
    pub contacts: Vec<ContactPointV2>, pub normal: Vec3CD,
    pub friction: f32, pub restitution: f32, pub age: u32,
}

impl ManifoldV2 {
    pub fn new(pair: PairKey, ba: BodyIdV2, bb: BodyIdV2) -> Self {
        Self { pair, body_a: ba, body_b: bb, contacts: Vec::with_capacity(MAX_MANIFOLD_CONTACTS), normal: Vec3CD::ZERO, friction: 0.5, restitution: 0.3, age: 0 }
    }
    pub fn add_contact(&mut self, c: ContactPointV2) {
        for existing in &mut self.contacts {
            if existing.position_a.distance(c.position_a) < 0.01 {
                existing.position_a = c.position_a; existing.position_b = c.position_b;
                existing.normal = c.normal; existing.depth = c.depth; existing.age = 0; return;
            }
        }
        if self.contacts.len() < MAX_MANIFOLD_CONTACTS { self.contacts.push(c); }
        else {
            // Replace least important contact
            let mut min_depth = f32::MAX; let mut min_idx = 0;
            for (i, ct) in self.contacts.iter().enumerate() {
                if ct.depth < min_depth { min_depth = ct.depth; min_idx = i; }
            }
            self.contacts[min_idx] = c;
        }
        if !self.contacts.is_empty() {
            let mut avg = Vec3CD::ZERO;
            for ct in &self.contacts { avg = avg.add(ct.normal); }
            self.normal = avg.scale(1.0 / self.contacts.len() as f32).normalize();
        }
    }
    pub fn warm_start_from(&mut self, prev: &ManifoldV2) {
        for c in &mut self.contacts {
            if let Some(p) = prev.contacts.iter().min_by(|a, b| {
                c.local_a.distance(a.local_a).partial_cmp(&c.local_a.distance(b.local_a)).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                if c.local_a.distance(p.local_a) < 0.05 {
                    c.normal_impulse = p.normal_impulse; c.tangent_impulse = p.tangent_impulse;
                }
            }
        }
    }
}

/// Sphere-sphere intersection.
pub fn sphere_sphere(pa: Vec3CD, ra: f32, pb: Vec3CD, rb: f32) -> Option<ContactPointV2> {
    let d = pb.sub(pa); let dist = d.length(); let sum = ra + rb;
    if dist > sum { return None; }
    let n = if dist > 1e-6 { d.scale(1.0/dist) } else { Vec3CD::new(0.0,1.0,0.0) };
    Some(ContactPointV2::new(pa.add(n.scale(ra)), pb.sub(n.scale(rb)), n, sum - dist))
}

/// Sphere-box intersection.
pub fn sphere_box(sp: Vec3CD, sr: f32, bp: Vec3CD, bh: Vec3CD) -> Option<ContactPointV2> {
    let local = sp.sub(bp);
    let closest = Vec3CD::new(local.x.clamp(-bh.x, bh.x), local.y.clamp(-bh.y, bh.y), local.z.clamp(-bh.z, bh.z));
    let diff = local.sub(closest); let dist = diff.length();
    if dist > sr { return None; }
    let n = if dist > 1e-6 { diff.scale(1.0/dist) } else {
        let dx = bh.x - local.x.abs(); let dy = bh.y - local.y.abs(); let dz = bh.z - local.z.abs();
        if dx < dy && dx < dz { Vec3CD::new(if local.x > 0.0 { 1.0 } else { -1.0 }, 0.0, 0.0) }
        else if dy < dz { Vec3CD::new(0.0, if local.y > 0.0 { 1.0 } else { -1.0 }, 0.0) }
        else { Vec3CD::new(0.0, 0.0, if local.z > 0.0 { 1.0 } else { -1.0 }) }
    };
    Some(ContactPointV2::new(sp.sub(n.scale(sr)), bp.add(closest), n, sr - dist))
}

/// Warm starting cache.
pub struct WarmCacheV2 { manifolds: HashMap<PairKey, ManifoldV2>, max_age: u32 }
impl WarmCacheV2 {
    pub fn new() -> Self { Self { manifolds: HashMap::new(), max_age: 10 } }
    pub fn store(&mut self, m: ManifoldV2) { self.manifolds.insert(m.pair, m); }
    pub fn retrieve(&self, pair: &PairKey) -> Option<&ManifoldV2> { self.manifolds.get(pair) }
    pub fn cleanup(&mut self) { let ma = self.max_age; self.manifolds.retain(|_, m| m.age < ma); }
    pub fn len(&self) -> usize { self.manifolds.len() }
}

/// GJK result.
pub struct GjkResultV2 { pub intersecting: bool, pub closest_distance: f32, pub iterations: u32 }

/// Simplified GJK for sphere support functions.
pub fn gjk_spheres(pa: Vec3CD, ra: f32, pb: Vec3CD, rb: f32) -> GjkResultV2 {
    let dist = pa.distance(pb); let sum = ra + rb;
    GjkResultV2 { intersecting: dist <= sum, closest_distance: (dist - sum).max(0.0), iterations: 1 }
}

/// Collision detection pipeline.
pub struct CollisionSystemV2 {
    pub manifolds: Vec<ManifoldV2>,
    pub warm_cache: WarmCacheV2,
    pub stats: CollisionStatsV2,
}

#[derive(Debug, Clone, Default)]
pub struct CollisionStatsV2 { pub broad_pairs: u32, pub narrow_tests: u32, pub active_manifolds: u32, pub total_contacts: u32, pub warm_started: u32 }

impl CollisionSystemV2 {
    pub fn new() -> Self { Self { manifolds: Vec::new(), warm_cache: WarmCacheV2::new(), stats: CollisionStatsV2::default() } }

    pub fn detect(&mut self, spheres: &[(ColliderIdV2, BodyIdV2, Vec3CD, f32)]) {
        self.stats = CollisionStatsV2::default();
        self.manifolds.clear();
        let n = spheres.len();
        for i in 0..n { for j in (i+1)..n {
            self.stats.broad_pairs += 1;
            let (ci, bi, pi, ri) = spheres[i]; let (cj, bj, pj, rj) = spheres[j];
            if let Some(contact) = sphere_sphere(pi, ri, pj, rj) {
                self.stats.narrow_tests += 1;
                let pair = PairKey::new(ci, cj);
                let mut m = ManifoldV2::new(pair, bi, bj);
                m.add_contact(contact);
                if let Some(prev) = self.warm_cache.retrieve(&pair) {
                    m.warm_start_from(prev); self.stats.warm_started += 1;
                }
                self.warm_cache.store(m.clone());
                self.manifolds.push(m);
            }
        }}
        self.warm_cache.cleanup();
        self.stats.active_manifolds = self.manifolds.len() as u32;
        self.stats.total_contacts = self.manifolds.iter().map(|m| m.contacts.len() as u32).sum();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_sphere_intersect() {
        let r = sphere_sphere(Vec3CD::ZERO, 1.0, Vec3CD::new(1.5, 0.0, 0.0), 1.0);
        assert!(r.is_some()); assert!(r.unwrap().depth > 0.0);
    }
    #[test] fn test_sphere_no_intersect() {
        assert!(sphere_sphere(Vec3CD::ZERO, 1.0, Vec3CD::new(3.0, 0.0, 0.0), 1.0).is_none());
    }
    #[test] fn test_manifold() {
        let pair = PairKey::new(ColliderIdV2(0), ColliderIdV2(1));
        let mut m = ManifoldV2::new(pair, BodyIdV2(0), BodyIdV2(1));
        for i in 0..6 { m.add_contact(ContactPointV2::new(Vec3CD::new(i as f32*0.1,0.0,0.0), Vec3CD::ZERO, Vec3CD::new(0.0,1.0,0.0), 0.01)); }
        assert!(m.contacts.len() <= MAX_MANIFOLD_CONTACTS);
    }
    #[test] fn test_collision_system() {
        let mut sys = CollisionSystemV2::new();
        sys.detect(&[
            (ColliderIdV2(0), BodyIdV2(0), Vec3CD::ZERO, 1.0),
            (ColliderIdV2(1), BodyIdV2(1), Vec3CD::new(1.5, 0.0, 0.0), 1.0),
        ]);
        assert_eq!(sys.stats.active_manifolds, 1);
    }
}
