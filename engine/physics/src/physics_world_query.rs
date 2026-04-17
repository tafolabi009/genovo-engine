// engine/physics/src/physics_world_query.rs
//
// World queries: closest body, bodies in AABB, ray vs specific body,
// shape vs shape distance, time of impact computation.

use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;

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


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BodyId(pub u32);

#[derive(Debug, Clone, Copy)]
pub struct AABB { pub min: Vec3, pub max: Vec3 }

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }
    pub fn contains(&self, p: Vec3) -> bool {
        p.x >= self.min.x && p.x <= self.max.x &&
        p.y >= self.min.y && p.y <= self.max.y &&
        p.z >= self.min.z && p.z <= self.max.z
    }
    pub fn overlaps(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x &&
        self.min.y <= other.max.y && self.max.y >= other.min.y &&
        self.min.z <= other.max.z && self.max.z >= other.min.z
    }
    pub fn center(&self) -> Vec3 { self.min.add(self.max).scale(0.5) }
    pub fn expand(&self, margin: f32) -> AABB {
        AABB {
            min: Vec3::new(self.min.x - margin, self.min.y - margin, self.min.z - margin),
            max: Vec3::new(self.max.x + margin, self.max.y + margin, self.max.z + margin),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeKind { Sphere, Box, Capsule }

#[derive(Debug, Clone)]
pub struct WorldBody {
    pub id: BodyId,
    pub position: Vec3,
    pub velocity: Vec3,
    pub shape: ShapeKind,
    pub radius: f32,
    pub half_extents: Vec3,
    pub half_height: f32,
    pub aabb: AABB,
    pub layer: u32,
    pub is_static: bool,
}

/// Query filter for world queries.
#[derive(Debug, Clone)]
pub struct QueryFilter {
    pub layer_mask: u32,
    pub exclude_body: Option<BodyId>,
    pub include_static: bool,
    pub include_dynamic: bool,
    pub max_distance: f32,
}

impl Default for QueryFilter {
    fn default() -> Self {
        Self {
            layer_mask: 0xFFFFFFFF,
            exclude_body: None,
            include_static: true,
            include_dynamic: true,
            max_distance: f32::MAX,
        }
    }
}

impl QueryFilter {
    pub fn accepts(&self, body: &WorldBody) -> bool {
        if body.layer & self.layer_mask == 0 { return false; }
        if let Some(exc) = self.exclude_body { if body.id == exc { return false; } }
        if body.is_static && !self.include_static { return false; }
        if !body.is_static && !self.include_dynamic { return false; }
        true
    }
}

/// Ray definition.
#[derive(Debug, Clone, Copy)]
pub struct Ray { pub origin: Vec3, pub direction: Vec3 }

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self { Self { origin, direction: direction.normalize() } }
    pub fn point_at(&self, t: f32) -> Vec3 { self.origin.add(self.direction.scale(t)) }
}

/// Raycast hit result.
#[derive(Debug, Clone)]
pub struct RayHitResult {
    pub body_id: BodyId,
    pub distance: f32,
    pub point: Vec3,
    pub normal: Vec3,
}

/// Closest body result.
#[derive(Debug, Clone)]
pub struct ClosestBodyResult {
    pub body_id: BodyId,
    pub distance: f32,
    pub closest_point: Vec3,
}

/// Shape-shape distance result.
#[derive(Debug, Clone)]
pub struct ShapeDistanceResult {
    pub distance: f32,
    pub closest_on_a: Vec3,
    pub closest_on_b: Vec3,
    pub normal: Vec3,
}

/// Time of impact result.
#[derive(Debug, Clone)]
pub struct TimeOfImpactResult {
    pub toi: f32,
    pub contact_point: Vec3,
    pub contact_normal: Vec3,
    pub body_a: BodyId,
    pub body_b: BodyId,
}

/// Physics world query system.
pub struct PhysicsWorldQuery {
    bodies: Vec<WorldBody>,
}

impl PhysicsWorldQuery {
    pub fn new() -> Self { Self { bodies: Vec::new() } }

    pub fn add_body(&mut self, body: WorldBody) { self.bodies.push(body); }

    pub fn remove_body(&mut self, id: BodyId) { self.bodies.retain(|b| b.id != id); }

    pub fn update_body(&mut self, id: BodyId, position: Vec3, velocity: Vec3) {
        if let Some(b) = self.bodies.iter_mut().find(|b| b.id == id) {
            b.position = position;
            b.velocity = velocity;
            // Update AABB
            let r = b.radius.max(b.half_extents.x).max(b.half_extents.y).max(b.half_extents.z).max(b.half_height + b.radius);
            b.aabb = AABB::new(
                Vec3::new(position.x - r, position.y - r, position.z - r),
                Vec3::new(position.x + r, position.y + r, position.z + r),
            );
        }
    }

    /// Find the closest body to a point.
    pub fn closest_body(&self, point: Vec3, filter: &QueryFilter) -> Option<ClosestBodyResult> {
        let mut best: Option<ClosestBodyResult> = None;
        for body in &self.bodies {
            if !filter.accepts(body) { continue; }
            let (dist, closest) = self.point_to_body_distance(point, body);
            if dist > filter.max_distance { continue; }
            if best.as_ref().map_or(true, |b| dist < b.distance) {
                best = Some(ClosestBodyResult { body_id: body.id, distance: dist, closest_point: closest });
            }
        }
        best
    }

    /// Find all bodies whose AABB overlaps the given AABB.
    pub fn bodies_in_aabb(&self, query_aabb: &AABB, filter: &QueryFilter) -> Vec<BodyId> {
        self.bodies.iter()
            .filter(|b| filter.accepts(b) && b.aabb.overlaps(query_aabb))
            .map(|b| b.id)
            .collect()
    }

    /// Find all bodies within a sphere.
    pub fn bodies_in_sphere(&self, center: Vec3, radius: f32, filter: &QueryFilter) -> Vec<BodyId> {
        let aabb = AABB::new(
            Vec3::new(center.x-radius, center.y-radius, center.z-radius),
            Vec3::new(center.x+radius, center.y+radius, center.z+radius),
        );
        self.bodies.iter()
            .filter(|b| {
                if !filter.accepts(b) || !b.aabb.overlaps(&aabb) { return false; }
                let (dist, _) = self.point_to_body_distance(center, b);
                dist <= radius
            })
            .map(|b| b.id)
            .collect()
    }

    /// Cast a ray against all bodies.
    pub fn raycast(&self, ray: &Ray, max_dist: f32, filter: &QueryFilter) -> Option<RayHitResult> {
        let mut best: Option<RayHitResult> = None;
        for body in &self.bodies {
            if !filter.accepts(body) { continue; }
            if let Some(hit) = self.ray_vs_body(ray, body) {
                if hit.distance <= max_dist {
                    if best.as_ref().map_or(true, |b| hit.distance < b.distance) {
                        best = Some(hit);
                    }
                }
            }
        }
        best
    }

    /// Cast a ray against all bodies, returning all hits sorted by distance.
    pub fn raycast_all(&self, ray: &Ray, max_dist: f32, filter: &QueryFilter) -> Vec<RayHitResult> {
        let mut hits = Vec::new();
        for body in &self.bodies {
            if !filter.accepts(body) { continue; }
            if let Some(hit) = self.ray_vs_body(ray, body) {
                if hit.distance <= max_dist {
                    hits.push(hit);
                }
            }
        }
        hits.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        hits
    }

    /// Ray vs specific body.
    fn ray_vs_body(&self, ray: &Ray, body: &WorldBody) -> Option<RayHitResult> {
        match body.shape {
            ShapeKind::Sphere => self.ray_vs_sphere(ray, body.position, body.radius, body.id),
            ShapeKind::Box => self.ray_vs_aabb(ray, &body.aabb, body.id),
            ShapeKind::Capsule => self.ray_vs_capsule(ray, body.position, body.radius, body.half_height, body.id),
        }
    }

    fn ray_vs_sphere(&self, ray: &Ray, center: Vec3, radius: f32, id: BodyId) -> Option<RayHitResult> {
        let oc = ray.origin.sub(center);
        let b = oc.dot(ray.direction);
        let c = oc.dot(oc) - radius * radius;
        let discriminant = b * b - c;
        if discriminant < 0.0 { return None; }
        let t = -b - discriminant.sqrt();
        if t < 0.0 { return None; }
        let point = ray.point_at(t);
        let normal = point.sub(center).normalize();
        Some(RayHitResult { body_id: id, distance: t, point, normal })
    }

    fn ray_vs_aabb(&self, ray: &Ray, aabb: &AABB, id: BodyId) -> Option<RayHitResult> {
        let inv = Vec3::new(
            if ray.direction.x.abs() > 1e-12 { 1.0/ray.direction.x } else { f32::MAX },
            if ray.direction.y.abs() > 1e-12 { 1.0/ray.direction.y } else { f32::MAX },
            if ray.direction.z.abs() > 1e-12 { 1.0/ray.direction.z } else { f32::MAX },
        );
        let t1 = (aabb.min.x - ray.origin.x) * inv.x;
        let t2 = (aabb.max.x - ray.origin.x) * inv.x;
        let t3 = (aabb.min.y - ray.origin.y) * inv.y;
        let t4 = (aabb.max.y - ray.origin.y) * inv.y;
        let t5 = (aabb.min.z - ray.origin.z) * inv.z;
        let t6 = (aabb.max.z - ray.origin.z) * inv.z;
        let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));
        if tmax < 0.0 || tmin > tmax { return None; }
        let t = if tmin >= 0.0 { tmin } else { tmax };
        let point = ray.point_at(t);
        // Determine face normal
        let center = aabb.center();
        let d = point.sub(center);
        let half = aabb.max.sub(center);
        let bias = Vec3::new(d.x / half.x, d.y / half.y, d.z / half.z);
        let normal = if bias.x.abs() > bias.y.abs() && bias.x.abs() > bias.z.abs() {
            Vec3::new(if bias.x > 0.0 { 1.0 } else { -1.0 }, 0.0, 0.0)
        } else if bias.y.abs() > bias.z.abs() {
            Vec3::new(0.0, if bias.y > 0.0 { 1.0 } else { -1.0 }, 0.0)
        } else {
            Vec3::new(0.0, 0.0, if bias.z > 0.0 { 1.0 } else { -1.0 })
        };
        Some(RayHitResult { body_id: id, distance: t, point, normal })
    }

    fn ray_vs_capsule(&self, ray: &Ray, center: Vec3, radius: f32, half_h: f32, id: BodyId) -> Option<RayHitResult> {
        // Test against the two spheres and the cylinder
        let top = center.add(Vec3::new(0.0, half_h, 0.0));
        let bot = center.sub(Vec3::new(0.0, half_h, 0.0));
        let mut best: Option<RayHitResult> = None;
        if let Some(h) = self.ray_vs_sphere(ray, top, radius, id) {
            if best.as_ref().map_or(true, |b| h.distance < b.distance) { best = Some(h); }
        }
        if let Some(h) = self.ray_vs_sphere(ray, bot, radius, id) {
            if best.as_ref().map_or(true, |b| h.distance < b.distance) { best = Some(h); }
        }
        // Simplified: test against AABB as cylinder approximation
        let aabb = AABB::new(
            Vec3::new(center.x-radius, center.y-half_h-radius, center.z-radius),
            Vec3::new(center.x+radius, center.y+half_h+radius, center.z+radius),
        );
        if let Some(h) = self.ray_vs_aabb(ray, &aabb, id) {
            if best.as_ref().map_or(true, |b| h.distance < b.distance) { best = Some(h); }
        }
        best
    }

    /// Point to body distance.
    fn point_to_body_distance(&self, point: Vec3, body: &WorldBody) -> (f32, Vec3) {
        match body.shape {
            ShapeKind::Sphere => {
                let d = point.sub(body.position);
                let dist = d.length();
                if dist < 1e-9 { return (0.0, body.position); }
                let closest = body.position.add(d.scale(body.radius / dist));
                ((dist - body.radius).max(0.0), closest)
            }
            ShapeKind::Box => {
                let closest = Vec3::new(
                    point.x.clamp(body.aabb.min.x, body.aabb.max.x),
                    point.y.clamp(body.aabb.min.y, body.aabb.max.y),
                    point.z.clamp(body.aabb.min.z, body.aabb.max.z),
                );
                (point.distance(closest), closest)
            }
            ShapeKind::Capsule => {
                let top = body.position.add(Vec3::new(0.0, body.half_height, 0.0));
                let bot = body.position.sub(Vec3::new(0.0, body.half_height, 0.0));
                let seg = top.sub(bot);
                let seg_len_sq = seg.length_sq();
                let t = if seg_len_sq < 1e-12 { 0.5 } else {
                    point.sub(bot).dot(seg) / seg_len_sq
                }.clamp(0.0, 1.0);
                let closest_on_axis = bot.lerp(top, t);
                let d = point.sub(closest_on_axis);
                let dist = d.length();
                if dist < 1e-9 { return (0.0, closest_on_axis); }
                let closest = closest_on_axis.add(d.scale(body.radius / dist));
                ((dist - body.radius).max(0.0), closest)
            }
        }
    }

    /// Compute shape-to-shape distance.
    pub fn shape_distance(&self, body_a: BodyId, body_b: BodyId) -> Option<ShapeDistanceResult> {
        let a = self.bodies.iter().find(|b| b.id == body_a)?;
        let b = self.bodies.iter().find(|b| b.id == body_b)?;
        // Simplified: use closest point from A center to B, then B center to A
        let (dist_a, closest_b) = self.point_to_body_distance(a.position, b);
        let (dist_b, closest_a) = self.point_to_body_distance(b.position, a);
        let diff = closest_b.sub(closest_a);
        let dist = diff.length();
        let normal = if dist > 1e-9 { diff.scale(1.0/dist) } else { Vec3::new(0.0,1.0,0.0) };
        Some(ShapeDistanceResult { distance: dist, closest_on_a: closest_a, closest_on_b: closest_b, normal })
    }

    /// Compute time of impact between two moving bodies using conservative advancement.
    pub fn time_of_impact(&self, body_a: BodyId, body_b: BodyId, dt: f32) -> Option<TimeOfImpactResult> {
        let a = self.bodies.iter().find(|b| b.id == body_a)?;
        let b = self.bodies.iter().find(|b| b.id == body_b)?;
        let rel_vel = a.velocity.sub(b.velocity);
        let closing_speed = rel_vel.length();
        if closing_speed < 1e-6 { return None; }

        let max_steps = 20;
        let mut t = 0.0_f32;
        let sum_radii = a.radius + b.radius;

        for _ in 0..max_steps {
            let pa = a.position.add(a.velocity.scale(t));
            let pb = b.position.add(b.velocity.scale(t));
            let dist = pa.distance(pb) - sum_radii;
            if dist <= 0.01 {
                let normal = pb.sub(pa).normalize();
                let contact = pa.add(normal.scale(a.radius));
                return Some(TimeOfImpactResult { toi: t, contact_point: contact, contact_normal: normal, body_a: body_a, body_b: body_b });
            }
            let advance = (dist / closing_speed).max(0.001);
            t += advance;
            if t > dt { return None; }
        }
        None
    }

    /// Get body count.
    pub fn body_count(&self) -> usize { self.bodies.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raycast_sphere() {
        let mut q = PhysicsWorldQuery::new();
        q.add_body(WorldBody {
            id: BodyId(0), position: Vec3::new(5.0,0.0,0.0), velocity: Vec3::ZERO,
            shape: ShapeKind::Sphere, radius: 1.0, half_extents: Vec3::ZERO, half_height: 0.0,
            aabb: AABB::new(Vec3::new(4.0,-1.0,-1.0), Vec3::new(6.0,1.0,1.0)),
            layer: 1, is_static: true,
        });
        let ray = Ray::new(Vec3::ZERO, Vec3::new(1.0,0.0,0.0));
        let hit = q.raycast(&ray, 100.0, &QueryFilter::default());
        assert!(hit.is_some());
        assert!((hit.unwrap().distance - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_closest_body() {
        let mut q = PhysicsWorldQuery::new();
        q.add_body(WorldBody {
            id: BodyId(0), position: Vec3::new(3.0,0.0,0.0), velocity: Vec3::ZERO,
            shape: ShapeKind::Sphere, radius: 1.0, half_extents: Vec3::ZERO, half_height: 0.0,
            aabb: AABB::new(Vec3::new(2.0,-1.0,-1.0), Vec3::new(4.0,1.0,1.0)),
            layer: 1, is_static: true,
        });
        let result = q.closest_body(Vec3::ZERO, &QueryFilter::default());
        assert!(result.is_some());
        assert!((result.unwrap().distance - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_bodies_in_aabb() {
        let mut q = PhysicsWorldQuery::new();
        q.add_body(WorldBody {
            id: BodyId(0), position: Vec3::ZERO, velocity: Vec3::ZERO,
            shape: ShapeKind::Sphere, radius: 1.0, half_extents: Vec3::ZERO, half_height: 0.0,
            aabb: AABB::new(Vec3::new(-1.0,-1.0,-1.0), Vec3::new(1.0,1.0,1.0)),
            layer: 1, is_static: true,
        });
        let query_aabb = AABB::new(Vec3::new(-2.0,-2.0,-2.0), Vec3::new(2.0,2.0,2.0));
        let results = q.bodies_in_aabb(&query_aabb, &QueryFilter::default());
        assert_eq!(results.len(), 1);
    }
}
