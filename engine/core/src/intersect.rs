//! Intersection tests and closest-point queries for the Genovo engine.
//!
//! Provides a comprehensive set of geometric intersection tests between rays,
//! spheres, AABBs, OBBs, planes, triangles, capsules, cylinders, discs, and
//! frustums. Also includes closest-point and distance utilities.

use glam::{Vec2, Vec3};
use crate::math::{AABB, Frustum, Plane, Ray};
use crate::geometry::OBB;

const EPSILON: f32 = 1e-7;

// ===========================================================================
// Hit result
// ===========================================================================

/// Result of a ray intersection test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    /// Distance along the ray to the hit point.
    pub t: f32,
    /// World-space position of the hit.
    pub point: Vec3,
    /// Surface normal at the hit point (points away from the surface).
    pub normal: Vec3,
}

// ===========================================================================
// Ray vs Sphere
// ===========================================================================

/// Tests a ray against a sphere. Returns the nearest hit, or `None`.
///
/// Uses the geometric method: project the sphere center onto the ray, then
/// apply the Pythagorean theorem to find the intersection distance.
pub fn ray_sphere(ray: &Ray, center: Vec3, radius: f32) -> Option<RayHit> {
    let oc = ray.origin - center;
    let b = oc.dot(ray.direction);
    let c = oc.dot(oc) - radius * radius;

    let discriminant = b * b - c;
    if discriminant < 0.0 {
        return None;
    }

    let sqrt_disc = discriminant.sqrt();
    let mut t = -b - sqrt_disc;
    if t < 0.0 {
        t = -b + sqrt_disc;
        if t < 0.0 {
            return None;
        }
    }

    let point = ray.point_at(t);
    let normal = (point - center).normalize();
    Some(RayHit { t, point, normal })
}

// ===========================================================================
// Ray vs AABB
// ===========================================================================

/// Tests a ray against an AABB. Returns the nearest hit with the face normal.
pub fn ray_aabb(ray: &Ray, aabb: &AABB) -> Option<RayHit> {
    let inv_dir = Vec3::ONE / ray.direction;
    let t1 = (aabb.min - ray.origin) * inv_dir;
    let t2 = (aabb.max - ray.origin) * inv_dir;
    let t_min = t1.min(t2);
    let t_max = t1.max(t2);

    let t_enter = t_min.x.max(t_min.y).max(t_min.z);
    let t_exit = t_max.x.min(t_max.y).min(t_max.z);

    if t_enter > t_exit || t_exit < 0.0 {
        return None;
    }

    let t = if t_enter >= 0.0 { t_enter } else { t_exit };
    let point = ray.point_at(t);

    // Determine which face was hit by finding which axis entered last.
    let normal = if t == t_enter {
        if t_enter == t_min.x {
            Vec3::new(-inv_dir.x.signum(), 0.0, 0.0)
        } else if t_enter == t_min.y {
            Vec3::new(0.0, -inv_dir.y.signum(), 0.0)
        } else {
            Vec3::new(0.0, 0.0, -inv_dir.z.signum())
        }
    } else {
        // Hit from inside.
        if t_exit == t_max.x {
            Vec3::new(inv_dir.x.signum(), 0.0, 0.0)
        } else if t_exit == t_max.y {
            Vec3::new(0.0, inv_dir.y.signum(), 0.0)
        } else {
            Vec3::new(0.0, 0.0, inv_dir.z.signum())
        }
    };

    Some(RayHit { t, point, normal })
}

// ===========================================================================
// Ray vs OBB
// ===========================================================================

/// Tests a ray against an oriented bounding box.
///
/// Transforms the ray into the OBB's local frame, then performs a slab test.
pub fn ray_obb(ray: &Ray, obb: &OBB) -> Option<RayHit> {
    let d = ray.origin - obb.center;
    let origin_local = Vec3::new(
        d.dot(obb.axes[0]),
        d.dot(obb.axes[1]),
        d.dot(obb.axes[2]),
    );
    let dir_local = Vec3::new(
        ray.direction.dot(obb.axes[0]),
        ray.direction.dot(obb.axes[1]),
        ray.direction.dot(obb.axes[2]),
    );

    let half = obb.half_extents;
    let local_aabb = AABB::new(-half, half);
    let local_ray = Ray::new(origin_local, dir_local);

    let hit = ray_aabb(&local_ray, &local_aabb)?;

    // Transform hit back to world space.
    let world_point = obb.center
        + obb.axes[0] * (origin_local.x + dir_local.x * hit.t)
        + obb.axes[1] * (origin_local.y + dir_local.y * hit.t)
        + obb.axes[2] * (origin_local.z + dir_local.z * hit.t);
    let world_normal =
        obb.axes[0] * hit.normal.x + obb.axes[1] * hit.normal.y + obb.axes[2] * hit.normal.z;

    Some(RayHit {
        t: hit.t,
        point: world_point,
        normal: world_normal.normalize(),
    })
}

// ===========================================================================
// Ray vs Plane
// ===========================================================================

/// Tests a ray against a plane. Returns the hit if the ray is not parallel.
pub fn ray_plane(ray: &Ray, plane: &Plane) -> Option<RayHit> {
    let denom = ray.direction.dot(plane.normal);
    if denom.abs() < EPSILON {
        return None;
    }

    let t = (plane.distance - ray.origin.dot(plane.normal)) / denom;
    if t < 0.0 {
        return None;
    }

    let point = ray.point_at(t);
    let normal = if denom < 0.0 {
        plane.normal
    } else {
        -plane.normal
    };

    Some(RayHit { t, point, normal })
}

// ===========================================================================
// Ray vs Triangle (Moller-Trumbore)
// ===========================================================================

/// Tests a ray against a triangle using the Moller-Trumbore algorithm.
///
/// Returns the hit with barycentric coordinates encoded in the normal
/// (the actual geometric normal of the triangle face).
pub fn ray_triangle(ray: &Ray, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<RayHit> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = ray.direction.cross(e2);
    let a = e1.dot(h);

    if a.abs() < EPSILON {
        return None;
    }

    let f = 1.0 / a;
    let s = ray.origin - v0;
    let u = f * s.dot(h);
    if u < 0.0 || u > 1.0 {
        return None;
    }

    let q = s.cross(e1);
    let v = f * ray.direction.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = f * e2.dot(q);
    if t < EPSILON {
        return None;
    }

    let point = ray.point_at(t);
    let normal = e1.cross(e2).normalize();

    Some(RayHit { t, point, normal })
}

// ===========================================================================
// Ray vs Capsule
// ===========================================================================

/// A capsule defined by a line segment (a, b) and a radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Capsule {
    pub a: Vec3,
    pub b: Vec3,
    pub radius: f32,
}

/// Tests a ray against a capsule.
///
/// A capsule is the Minkowski sum of a line segment and a sphere. The test
/// finds the closest point on the segment to the ray, then tests the ray
/// against a sphere at that point.
pub fn ray_capsule(ray: &Ray, capsule: &Capsule) -> Option<RayHit> {
    // Project the problem: find the closest point on the capsule segment
    // to the ray, then test a sphere there.
    let segment_dir = capsule.b - capsule.a;
    let segment_len_sq = segment_dir.length_squared();

    if segment_len_sq < EPSILON * EPSILON {
        // Degenerate capsule (point sphere).
        return ray_sphere(ray, capsule.a, capsule.radius);
    }

    // Test against the cylinder body using the infinite cylinder formula,
    // then clamp and test hemispheres.
    let d = ray.direction;
    let m = ray.origin - capsule.a;
    let n = segment_dir / segment_len_sq.sqrt();

    let md = m.dot(n);
    let nd = d.dot(n);
    let _mm = m.dot(m);

    // Project out the segment axis.
    let d_perp = d - n * nd;
    let m_perp = m - n * md;

    let a_coeff = d_perp.dot(d_perp);
    let b_coeff = 2.0 * m_perp.dot(d_perp);
    let c_coeff = m_perp.dot(m_perp) - capsule.radius * capsule.radius;

    let disc = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff;

    let mut best_hit: Option<RayHit> = None;

    if disc >= 0.0 && a_coeff > EPSILON {
        let sqrt_disc = disc.sqrt();
        for sign in &[-1.0f32, 1.0f32] {
            let t = (-b_coeff + sign * sqrt_disc) / (2.0 * a_coeff);
            if t < 0.0 {
                continue;
            }
            // Check if the hit is within the cylinder body.
            let hit_along = md + nd * t;
            let seg_len = segment_len_sq.sqrt();
            if hit_along >= 0.0 && hit_along <= seg_len {
                let point = ray.point_at(t);
                let proj = capsule.a + n * hit_along;
                let normal = (point - proj).normalize();
                if best_hit.is_none() || t < best_hit.unwrap().t {
                    best_hit = Some(RayHit { t, point, normal });
                }
                break; // First valid t is the nearest.
            }
        }
    }

    // Test hemisphere at a.
    if let Some(hit) = ray_sphere(ray, capsule.a, capsule.radius) {
        // Ensure the hit is on the correct hemisphere.
        let local = hit.point - capsule.a;
        if local.dot(segment_dir) <= 0.0 {
            if best_hit.is_none() || hit.t < best_hit.unwrap().t {
                best_hit = Some(hit);
            }
        }
    }

    // Test hemisphere at b.
    if let Some(hit) = ray_sphere(ray, capsule.b, capsule.radius) {
        let local = hit.point - capsule.b;
        if local.dot(-segment_dir) <= 0.0 {
            if best_hit.is_none() || hit.t < best_hit.unwrap().t {
                best_hit = Some(hit);
            }
        }
    }

    best_hit
}

// ===========================================================================
// Ray vs Cylinder
// ===========================================================================

/// Tests a ray against a finite cylinder defined by endpoints (a, b) and radius.
///
/// Tests the infinite cylinder first, clamps to the finite extent, then tests
/// the two disc caps.
pub fn ray_cylinder(ray: &Ray, a: Vec3, b: Vec3, radius: f32) -> Option<RayHit> {
    let axis = b - a;
    let axis_len = axis.length();
    if axis_len < EPSILON {
        return ray_sphere(ray, a, radius);
    }
    let axis_n = axis / axis_len;

    let d = ray.direction;
    let m = ray.origin - a;

    let md = m.dot(axis_n);
    let nd = d.dot(axis_n);

    let d_perp = d - axis_n * nd;
    let m_perp = m - axis_n * md;

    let a_coeff = d_perp.dot(d_perp);
    let b_coeff = 2.0 * m_perp.dot(d_perp);
    let c_coeff = m_perp.dot(m_perp) - radius * radius;

    let mut best: Option<RayHit> = None;

    let disc = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff;
    if disc >= 0.0 && a_coeff > EPSILON {
        let sqrt_disc = disc.sqrt();
        for sign in &[-1.0f32, 1.0f32] {
            let t = (-b_coeff + sign * sqrt_disc) / (2.0 * a_coeff);
            if t < 0.0 {
                continue;
            }
            let h = md + nd * t;
            if h >= 0.0 && h <= axis_len {
                let point = ray.point_at(t);
                let proj = a + axis_n * h;
                let normal = (point - proj).normalize();
                if best.is_none() || t < best.unwrap().t {
                    best = Some(RayHit { t, point, normal });
                }
                break;
            }
        }
    }

    // Test bottom cap (disc at a).
    if let Some(hit) = ray_disc(ray, a, -axis_n, radius) {
        if best.is_none() || hit.t < best.unwrap().t {
            best = Some(hit);
        }
    }

    // Test top cap (disc at b).
    if let Some(hit) = ray_disc(ray, b, axis_n, radius) {
        if best.is_none() || hit.t < best.unwrap().t {
            best = Some(hit);
        }
    }

    best
}

// ===========================================================================
// Ray vs Disc
// ===========================================================================

/// Tests a ray against a disc (filled circle) at `center` with `normal` and `radius`.
pub fn ray_disc(ray: &Ray, center: Vec3, normal: Vec3, radius: f32) -> Option<RayHit> {
    let plane = Plane::from_point_normal(center, normal);
    let hit = ray_plane(ray, &plane)?;

    let d = hit.point - center;
    if d.dot(d) <= radius * radius {
        Some(hit)
    } else {
        None
    }
}

// ===========================================================================
// Sphere vs Sphere
// ===========================================================================

/// Contact information for sphere-sphere intersection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphereContact {
    /// Contact point (midpoint of the overlap on the line between centers).
    pub point: Vec3,
    /// Contact normal (from sphere A to sphere B).
    pub normal: Vec3,
    /// Penetration depth (positive means overlap).
    pub depth: f32,
}

/// Tests two spheres for intersection. Returns contact information or `None`.
pub fn sphere_sphere(
    center_a: Vec3,
    radius_a: f32,
    center_b: Vec3,
    radius_b: f32,
) -> Option<SphereContact> {
    let d = center_b - center_a;
    let dist_sq = d.length_squared();
    let r_sum = radius_a + radius_b;

    if dist_sq > r_sum * r_sum {
        return None;
    }

    let dist = dist_sq.sqrt();
    let normal = if dist > EPSILON {
        d / dist
    } else {
        Vec3::Y
    };

    let depth = r_sum - dist;
    let point = center_a + normal * (radius_a - depth * 0.5);

    Some(SphereContact {
        point,
        normal,
        depth,
    })
}

// ===========================================================================
// Sphere vs AABB
// ===========================================================================

/// Tests a sphere against an AABB. Returns `true` if they overlap.
pub fn sphere_aabb(center: Vec3, radius: f32, aabb: &AABB) -> bool {
    let closest = closest_point_on_aabb(center, aabb);
    (closest - center).length_squared() <= radius * radius
}

// ===========================================================================
// Sphere vs Plane
// ===========================================================================

/// Tests a sphere against a plane.
///
/// Returns the signed distance from the sphere center to the plane. Negative
/// means the sphere is (at least partially) on the back side.
pub fn sphere_plane_distance(center: Vec3, _radius: f32, plane: &Plane) -> f32 {
    plane.signed_distance(center)
}

/// Returns `true` if the sphere intersects the plane.
pub fn sphere_plane(center: Vec3, radius: f32, plane: &Plane) -> bool {
    plane.signed_distance(center).abs() <= radius
}

// ===========================================================================
// AABB vs AABB
// ===========================================================================

/// Tests two AABBs for intersection. This is just `a.intersects(b)`, provided
/// here for consistency with the rest of the intersection API.
#[inline]
pub fn aabb_aabb(a: &AABB, b: &AABB) -> bool {
    a.intersects(b)
}

// ===========================================================================
// OBB vs OBB (SAT, 15-axis)
// ===========================================================================

/// Tests two OBBs for intersection using the Separating Axis Theorem.
///
/// This is a re-export of `geometry::obb_obb_sat` for convenience.
#[inline]
pub fn obb_obb(a: &OBB, b: &OBB) -> bool {
    crate::geometry::obb_obb_sat(a, b)
}

// ===========================================================================
// Capsule vs Capsule
// ===========================================================================

/// Tests two capsules for intersection.
///
/// Computes the closest points between the two internal segments, then checks
/// if the distance is less than the sum of the radii.
pub fn capsule_capsule(c1: &Capsule, c2: &Capsule) -> bool {
    let (_, _, dist_sq) = closest_points_segments(c1.a, c1.b, c2.a, c2.b);
    let r_sum = c1.radius + c2.radius;
    dist_sq <= r_sum * r_sum
}

// ===========================================================================
// Frustum vs Sphere
// ===========================================================================

/// Tests a sphere against a frustum. Returns `true` if at least partially inside.
///
/// This delegates to `Frustum::contains_sphere`.
#[inline]
pub fn frustum_sphere(frustum: &Frustum, center: Vec3, radius: f32) -> bool {
    frustum.contains_sphere(center, radius)
}

// ===========================================================================
// Frustum vs AABB
// ===========================================================================

/// Tests an AABB against a frustum. Returns `true` if at least partially inside.
///
/// This delegates to `Frustum::contains_aabb`.
#[inline]
pub fn frustum_aabb(frustum: &Frustum, aabb: &AABB) -> bool {
    frustum.contains_aabb(aabb)
}

// ===========================================================================
// Point-in-triangle (3-D)
// ===========================================================================

/// Tests whether a point lies inside a triangle (3-D) using barycentric
/// coordinates.
pub fn point_in_triangle_3d(p: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> bool {
    let e0 = v1 - v0;
    let e1 = v2 - v0;
    let v = p - v0;

    let d00 = e0.dot(e0);
    let d01 = e0.dot(e1);
    let d11 = e1.dot(e1);
    let d20 = v.dot(e0);
    let d21 = v.dot(e1);

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < EPSILON {
        return false;
    }

    let inv_denom = 1.0 / denom;
    let u = (d11 * d20 - d01 * d21) * inv_denom;
    let w = (d00 * d21 - d01 * d20) * inv_denom;

    u >= 0.0 && w >= 0.0 && (u + w) <= 1.0
}

// ===========================================================================
// Point in polygon (2-D) -- delegates to geometry module
// ===========================================================================

/// Tests whether a point lies inside a 2-D polygon (ray-casting algorithm).
///
/// Re-export of `geometry::point_in_polygon`.
#[inline]
pub fn point_in_polygon(point: Vec2, vertices: &[Vec2]) -> bool {
    crate::geometry::point_in_polygon(point, vertices)
}

// ===========================================================================
// Closest point on segment
// ===========================================================================

/// Returns the closest point on segment `(a, b)` to `point`.
pub fn closest_point_on_segment(point: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < EPSILON * EPSILON {
        return a;
    }
    let t = ((point - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    a + ab * t
}

// ===========================================================================
// Closest point on triangle
// ===========================================================================

/// Returns the closest point on triangle (v0, v1, v2) to `point`.
///
/// Uses the Voronoi region method for robust projection.
pub fn closest_point_on_triangle(point: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Vec3 {
    let ab = v1 - v0;
    let ac = v2 - v0;
    let ap = point - v0;

    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return v0; // Vertex region A.
    }

    let bp = point - v1;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return v1; // Vertex region B.
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return v0 + ab * v; // Edge AB.
    }

    let cp = point - v2;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return v2; // Vertex region C.
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return v0 + ac * w; // Edge AC.
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return v1 + (v2 - v1) * w; // Edge BC.
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    v0 + ab * v + ac * w // Inside the triangle.
}

// ===========================================================================
// Closest point on AABB
// ===========================================================================

/// Returns the closest point on an AABB to `point`.
#[inline]
pub fn closest_point_on_aabb(point: Vec3, aabb: &AABB) -> Vec3 {
    Vec3::new(
        point.x.clamp(aabb.min.x, aabb.max.x),
        point.y.clamp(aabb.min.y, aabb.max.y),
        point.z.clamp(aabb.min.z, aabb.max.z),
    )
}

// ===========================================================================
// Distance: segment to segment
// ===========================================================================

/// Computes the closest points between two line segments (p1-q1) and (p2-q2).
///
/// Returns `(closest_on_segment1, closest_on_segment2, distance_squared)`.
///
/// Uses the robust algorithm from "Real-Time Collision Detection" (Ericson).
pub fn closest_points_segments(
    p1: Vec3,
    q1: Vec3,
    p2: Vec3,
    q2: Vec3,
) -> (Vec3, Vec3, f32) {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;

    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    // Both segments degenerate to points.
    if a <= EPSILON && e <= EPSILON {
        return (p1, p2, r.length_squared());
    }

    let (mut s, mut t);

    if a <= EPSILON {
        // First segment degenerates.
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= EPSILON {
            // Second segment degenerates.
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            // General non-degenerate case.
            let b = d1.dot(d2);
            let denom = a * e - b * b;

            if denom.abs() > EPSILON {
                s = ((b * f - c * e) / denom).clamp(0.0, 1.0);
            } else {
                s = 0.0;
            }

            t = (b * s + f) / e;

            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
        }
    }

    let c1 = p1 + d1 * s;
    let c2 = p2 + d2 * t;
    let diff = c1 - c2;

    (c1, c2, diff.length_squared())
}

/// Returns the squared distance between two line segments.
#[inline]
pub fn distance_segment_segment_sq(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> f32 {
    let (_, _, d2) = closest_points_segments(p1, q1, p2, q2);
    d2
}

/// Returns the distance between two line segments.
#[inline]
pub fn distance_segment_segment(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> f32 {
    distance_segment_segment_sq(p1, q1, p2, q2).sqrt()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_ray_sphere_hit() {
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::Z);
        let hit = ray_sphere(&ray, Vec3::ZERO, 1.0);
        assert!(hit.is_some());
        let h = hit.unwrap();
        assert!((h.t - 4.0).abs() < 0.01);
        assert!((h.point.z - -1.0).abs() < 0.01);
    }

    #[test]
    fn test_ray_sphere_miss() {
        let ray = Ray::new(Vec3::new(0.0, 5.0, -5.0), Vec3::Z);
        assert!(ray_sphere(&ray, Vec3::ZERO, 1.0).is_none());
    }

    #[test]
    fn test_ray_aabb_hit() {
        let ray = Ray::new(Vec3::new(0.5, 0.5, -5.0), Vec3::Z);
        let aabb = AABB::new(Vec3::ZERO, Vec3::ONE);
        let hit = ray_aabb(&ray, &aabb);
        assert!(hit.is_some());
        let h = hit.unwrap();
        assert!((h.t - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_ray_aabb_miss() {
        let ray = Ray::new(Vec3::new(5.0, 5.0, -5.0), Vec3::Z);
        let aabb = AABB::new(Vec3::ZERO, Vec3::ONE);
        assert!(ray_aabb(&ray, &aabb).is_none());
    }

    #[test]
    fn test_ray_obb_hit() {
        let obb = OBB {
            center: Vec3::ZERO,
            axes: [Vec3::X, Vec3::Y, Vec3::Z],
            half_extents: Vec3::new(1.0, 1.0, 1.0),
        };
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::Z);
        let hit = ray_obb(&ray, &obb);
        assert!(hit.is_some());
    }

    #[test]
    fn test_ray_plane_hit() {
        let plane = Plane::new(Vec3::Y, 0.0);
        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y);
        let hit = ray_plane(&ray, &plane);
        assert!(hit.is_some());
        assert!((hit.unwrap().t - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_ray_plane_parallel() {
        let plane = Plane::new(Vec3::Y, 0.0);
        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::X);
        assert!(ray_plane(&ray, &plane).is_none());
    }

    #[test]
    fn test_ray_triangle_hit() {
        let v0 = Vec3::new(-1.0, -1.0, 0.0);
        let v1 = Vec3::new(1.0, -1.0, 0.0);
        let v2 = Vec3::new(0.0, 1.0, 0.0);
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::Z);
        let hit = ray_triangle(&ray, v0, v1, v2);
        assert!(hit.is_some());
        assert!((hit.unwrap().t - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_ray_triangle_miss() {
        let v0 = Vec3::new(-1.0, -1.0, 0.0);
        let v1 = Vec3::new(1.0, -1.0, 0.0);
        let v2 = Vec3::new(0.0, 1.0, 0.0);
        let ray = Ray::new(Vec3::new(5.0, 5.0, -5.0), Vec3::Z);
        assert!(ray_triangle(&ray, v0, v1, v2).is_none());
    }

    #[test]
    fn test_ray_capsule() {
        let capsule = Capsule {
            a: Vec3::new(0.0, 0.0, 0.0),
            b: Vec3::new(0.0, 5.0, 0.0),
            radius: 1.0,
        };
        let ray = Ray::new(Vec3::new(5.0, 2.5, 0.0), Vec3::NEG_X);
        let hit = ray_capsule(&ray, &capsule);
        assert!(hit.is_some());
    }

    #[test]
    fn test_ray_cylinder() {
        let ray = Ray::new(Vec3::new(5.0, 0.5, 0.0), Vec3::NEG_X);
        let hit = ray_cylinder(&ray, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 1.0);
        assert!(hit.is_some());
    }

    #[test]
    fn test_ray_disc() {
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::Z);
        let hit = ray_disc(&ray, Vec3::ZERO, Vec3::NEG_Z, 2.0);
        assert!(hit.is_some());
    }

    #[test]
    fn test_ray_disc_miss() {
        let ray = Ray::new(Vec3::new(5.0, 5.0, -5.0), Vec3::Z);
        let hit = ray_disc(&ray, Vec3::ZERO, Vec3::NEG_Z, 1.0);
        assert!(hit.is_none());
    }

    #[test]
    fn test_sphere_sphere_overlap() {
        let contact = sphere_sphere(Vec3::ZERO, 1.0, Vec3::new(1.5, 0.0, 0.0), 1.0);
        assert!(contact.is_some());
        let c = contact.unwrap();
        assert!(c.depth > 0.0);
    }

    #[test]
    fn test_sphere_sphere_separated() {
        let contact = sphere_sphere(Vec3::ZERO, 1.0, Vec3::new(5.0, 0.0, 0.0), 1.0);
        assert!(contact.is_none());
    }

    #[test]
    fn test_sphere_aabb_overlap() {
        let aabb = AABB::new(Vec3::ZERO, Vec3::ONE);
        assert!(sphere_aabb(Vec3::new(0.5, 0.5, 0.5), 0.1, &aabb));
        assert!(sphere_aabb(Vec3::new(-0.5, 0.5, 0.5), 1.0, &aabb));
    }

    #[test]
    fn test_sphere_aabb_miss() {
        let aabb = AABB::new(Vec3::ZERO, Vec3::ONE);
        assert!(!sphere_aabb(Vec3::new(5.0, 5.0, 5.0), 0.1, &aabb));
    }

    #[test]
    fn test_sphere_plane() {
        let plane = Plane::new(Vec3::Y, 0.0);
        assert!(sphere_plane(Vec3::new(0.0, 0.5, 0.0), 1.0, &plane));
        assert!(!sphere_plane(Vec3::new(0.0, 5.0, 0.0), 1.0, &plane));
    }

    #[test]
    fn test_aabb_aabb_overlap() {
        let a = AABB::new(Vec3::ZERO, Vec3::ONE);
        let b = AABB::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.5, 1.5, 1.5));
        assert!(aabb_aabb(&a, &b));
    }

    #[test]
    fn test_aabb_aabb_separated() {
        let a = AABB::new(Vec3::ZERO, Vec3::ONE);
        let b = AABB::new(Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0));
        assert!(!aabb_aabb(&a, &b));
    }

    #[test]
    fn test_obb_obb_intersection() {
        let a = OBB {
            center: Vec3::ZERO,
            axes: [Vec3::X, Vec3::Y, Vec3::Z],
            half_extents: Vec3::ONE,
        };
        let b = OBB {
            center: Vec3::new(1.0, 0.0, 0.0),
            axes: [Vec3::X, Vec3::Y, Vec3::Z],
            half_extents: Vec3::ONE,
        };
        assert!(obb_obb(&a, &b));
    }

    #[test]
    fn test_capsule_capsule_overlap() {
        let c1 = Capsule {
            a: Vec3::ZERO,
            b: Vec3::new(0.0, 2.0, 0.0),
            radius: 0.5,
        };
        let c2 = Capsule {
            a: Vec3::new(0.8, 1.0, 0.0),
            b: Vec3::new(0.8, 3.0, 0.0),
            radius: 0.5,
        };
        assert!(capsule_capsule(&c1, &c2));
    }

    #[test]
    fn test_capsule_capsule_separated() {
        let c1 = Capsule {
            a: Vec3::ZERO,
            b: Vec3::new(0.0, 1.0, 0.0),
            radius: 0.1,
        };
        let c2 = Capsule {
            a: Vec3::new(5.0, 0.0, 0.0),
            b: Vec3::new(5.0, 1.0, 0.0),
            radius: 0.1,
        };
        assert!(!capsule_capsule(&c1, &c2));
    }

    #[test]
    fn test_point_in_triangle_3d() {
        let v0 = Vec3::new(0.0, 0.0, 0.0);
        let v1 = Vec3::new(2.0, 0.0, 0.0);
        let v2 = Vec3::new(1.0, 2.0, 0.0);
        assert!(point_in_triangle_3d(Vec3::new(1.0, 0.5, 0.0), v0, v1, v2));
        assert!(!point_in_triangle_3d(Vec3::new(5.0, 5.0, 0.0), v0, v1, v2));
    }

    #[test]
    fn test_closest_point_on_segment() {
        let a = Vec3::ZERO;
        let b = Vec3::new(10.0, 0.0, 0.0);
        let p = Vec3::new(5.0, 3.0, 0.0);
        let cp = closest_point_on_segment(p, a, b);
        assert!((cp - Vec3::new(5.0, 0.0, 0.0)).length() < 0.01);
    }

    #[test]
    fn test_closest_point_on_segment_clamped() {
        let a = Vec3::ZERO;
        let b = Vec3::new(10.0, 0.0, 0.0);
        let p = Vec3::new(-5.0, 3.0, 0.0);
        let cp = closest_point_on_segment(p, a, b);
        assert!((cp - Vec3::ZERO).length() < 0.01);
    }

    #[test]
    fn test_closest_point_on_triangle() {
        let v0 = Vec3::ZERO;
        let v1 = Vec3::new(4.0, 0.0, 0.0);
        let v2 = Vec3::new(2.0, 4.0, 0.0);

        // Point inside.
        let inside = Vec3::new(2.0, 1.0, 0.0);
        let cp = closest_point_on_triangle(inside, v0, v1, v2);
        assert!((cp - inside).length() < 0.01);

        // Point above.
        let above = Vec3::new(2.0, 1.0, 5.0);
        let cp = closest_point_on_triangle(above, v0, v1, v2);
        assert!((cp - Vec3::new(2.0, 1.0, 0.0)).length() < 0.01);
    }

    #[test]
    fn test_closest_point_on_aabb() {
        let aabb = AABB::new(Vec3::ZERO, Vec3::ONE);
        let p = Vec3::new(0.5, 5.0, 0.5);
        let cp = closest_point_on_aabb(p, &aabb);
        assert!((cp - Vec3::new(0.5, 1.0, 0.5)).length() < 0.01);
    }

    #[test]
    fn test_distance_segment_segment() {
        // Parallel segments.
        let d = distance_segment_segment(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
        );
        assert!((d - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_distance_segment_segment_crossing() {
        // Segments that cross at closest approach.
        let d = distance_segment_segment(
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
        );
        assert!((d - 1.0).abs() < 0.01);
    }
}
