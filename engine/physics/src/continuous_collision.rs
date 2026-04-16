//! Continuous Collision Detection (CCD) for the physics engine.
//!
//! Prevents fast-moving objects from tunnelling through thin geometry by
//! computing the exact time of impact (TOI) along their swept trajectory.
//!
//! Provides:
//! - `sweep_sphere_sphere`: time of first contact between two moving spheres
//! - `sweep_aabb_aabb`: swept AABB test with time and contact normal
//! - `sweep_sphere_plane`: sphere vs infinite plane sweep
//! - `sweep_sphere_triangle`: sphere vs triangle sweep
//! - Conservative advancement algorithm for general convex pairs
//! - `CCDSettings`: configuration for the CCD pipeline
//! - Integration helpers for `PhysicsWorld`

use glam::Vec3;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Small epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-7;
/// Maximum iterations for conservative advancement.
const MAX_CONSERVATIVE_ITERATIONS: usize = 32;
/// Convergence threshold for conservative advancement (distance).
const CONSERVATIVE_THRESHOLD: f32 = 0.001;

// ---------------------------------------------------------------------------
// CCD Settings
// ---------------------------------------------------------------------------

/// Configuration for the continuous collision detection pipeline.
#[derive(Debug, Clone)]
pub struct CCDSettings {
    /// Whether CCD is enabled globally.
    pub enabled: bool,
    /// Maximum number of substeps when resolving CCD.
    pub max_substeps: usize,
    /// Velocity threshold: bodies moving faster than this (m/s) trigger CCD.
    pub velocity_threshold: f32,
    /// Maximum time of impact search iterations.
    pub max_toi_iterations: usize,
    /// TOI convergence tolerance.
    pub toi_tolerance: f32,
    /// Minimum separation distance to maintain after TOI resolution.
    pub contact_offset: f32,
}

impl Default for CCDSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            max_substeps: 4,
            velocity_threshold: 5.0,
            max_toi_iterations: MAX_CONSERVATIVE_ITERATIONS,
            toi_tolerance: CONSERVATIVE_THRESHOLD,
            contact_offset: 0.01,
        }
    }
}

impl CCDSettings {
    /// Check whether a body with the given speed should use CCD.
    pub fn should_use_ccd(&self, speed: f32) -> bool {
        self.enabled && speed > self.velocity_threshold
    }
}

// ---------------------------------------------------------------------------
// Sweep result
// ---------------------------------------------------------------------------

/// Result of a sweep/time-of-impact test.
#[derive(Debug, Clone, Copy)]
pub struct SweepResult {
    /// Time of first contact in [0, dt]. None if no contact.
    pub toi: f32,
    /// Contact normal at the point of impact (from A to B).
    pub normal: Vec3,
    /// Contact point at the time of impact.
    pub contact_point: Vec3,
}

// ===========================================================================
// Sphere vs Sphere sweep
// ===========================================================================

/// Compute the time of first contact between two moving spheres.
///
/// Solves the quadratic equation for when the distance between the sphere
/// centers equals the sum of their radii:
///
///   |((pos_a + vel_a * t) - (pos_b + vel_b * t))| = radius_a + radius_b
///
/// # Arguments
/// * `a_pos` - Initial position of sphere A's center
/// * `a_vel` - Velocity of sphere A
/// * `a_radius` - Radius of sphere A
/// * `b_pos` - Initial position of sphere B's center
/// * `b_vel` - Velocity of sphere B
/// * `b_radius` - Radius of sphere B
/// * `dt` - Time window to search within
///
/// # Returns
/// `Some(toi)` if the spheres collide within [0, dt], `None` otherwise.
pub fn sweep_sphere_sphere(
    a_pos: Vec3,
    a_vel: Vec3,
    a_radius: f32,
    b_pos: Vec3,
    b_vel: Vec3,
    b_radius: f32,
    dt: f32,
) -> Option<f32> {
    // Relative displacement
    let d = a_pos - b_pos; // initial separation
    let v = a_vel - b_vel; // relative velocity

    let sum_radii = a_radius + b_radius;

    // Quadratic: |d + v*t|^2 = sum_radii^2
    // a*t^2 + b*t + c = 0
    let a = v.dot(v);
    let b = 2.0 * d.dot(v);
    let c = d.dot(d) - sum_radii * sum_radii;

    // If c < 0, spheres are already overlapping
    if c < 0.0 {
        // Already overlapping at t=0
        return Some(0.0);
    }

    // If a ~ 0, relative velocity is zero (or very small)
    if a < EPSILON {
        return None; // Not approaching
    }

    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None; // No intersection
    }

    let sqrt_disc = discriminant.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);

    // We want the earliest positive t
    let toi = if t1 >= 0.0 && t1 <= dt {
        t1
    } else if t2 >= 0.0 && t2 <= dt {
        t2
    } else if t1 < 0.0 && t2 > dt {
        // Spheres pass through each other during dt but first contact was before t=0
        return None;
    } else {
        return None;
    };

    Some(toi)
}

/// Compute the full sweep result for sphere vs sphere.
pub fn sweep_sphere_sphere_full(
    a_pos: Vec3,
    a_vel: Vec3,
    a_radius: f32,
    b_pos: Vec3,
    b_vel: Vec3,
    b_radius: f32,
    dt: f32,
) -> Option<SweepResult> {
    let toi = sweep_sphere_sphere(a_pos, a_vel, a_radius, b_pos, b_vel, b_radius, dt)?;

    // Compute positions at TOI
    let pos_a = a_pos + a_vel * toi;
    let pos_b = b_pos + b_vel * toi;
    let diff = pos_a - pos_b;
    let dist = diff.length();

    let normal = if dist > EPSILON {
        diff / dist
    } else {
        Vec3::Y // Degenerate: coincident centers
    };

    let contact_point = pos_b + normal * b_radius;

    Some(SweepResult {
        toi,
        normal,
        contact_point,
    })
}

// ===========================================================================
// AABB vs AABB sweep
// ===========================================================================

/// Axis-aligned bounding box for sweep tests.
#[derive(Debug, Clone, Copy)]
pub struct SweepAABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl SweepAABB {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Create from center and half-extents.
    pub fn from_center(center: Vec3, half_extents: Vec3) -> Self {
        Self {
            min: center - half_extents,
            max: center + half_extents,
        }
    }

    /// Get the center.
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Get half-extents.
    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Translate the AABB by a displacement.
    pub fn translated(&self, displacement: Vec3) -> Self {
        Self {
            min: self.min + displacement,
            max: self.max + displacement,
        }
    }
}

/// Compute the time of first contact between two moving AABBs.
///
/// Uses the slab method on the relative AABB (Minkowski difference).
///
/// # Returns
/// `Some((toi, normal))` if the AABBs collide within [0, dt].
pub fn sweep_aabb_aabb(
    a: &SweepAABB,
    a_vel: Vec3,
    b: &SweepAABB,
    b_vel: Vec3,
    dt: f32,
) -> Option<(f32, Vec3)> {
    // Compute relative velocity (A relative to B)
    let rel_vel = a_vel - b_vel;

    // Minkowski sum: expand B by A's extents
    let expanded_min = b.min - (a.max - a.min); // b.min - a_size
    let expanded_max = b.max;

    // The problem reduces to: does a point (a.min) moving with rel_vel
    // hit the expanded AABB?
    let origin = a.min;

    // Slab intersection
    let mut t_enter = 0.0f32;
    let mut t_exit = dt;
    let mut hit_normal = Vec3::ZERO;
    let mut hit_axis = -1i32;

    let axes = [
        (origin.x, rel_vel.x, expanded_min.x, expanded_max.x, Vec3::NEG_X, Vec3::X),
        (origin.y, rel_vel.y, expanded_min.y, expanded_max.y, Vec3::NEG_Y, Vec3::Y),
        (origin.z, rel_vel.z, expanded_min.z, expanded_max.z, Vec3::NEG_Z, Vec3::Z),
    ];

    for (i, &(o, v, slab_min, slab_max, neg_normal, pos_normal)) in axes.iter().enumerate() {
        if v.abs() < EPSILON {
            // Parallel to slab
            if o < slab_min || o > slab_max {
                return None; // Outside slab, no intersection
            }
        } else {
            let inv_v = 1.0 / v;
            let mut t1 = (slab_min - o) * inv_v;
            let mut t2 = (slab_max - o) * inv_v;
            let mut normal_near = neg_normal;

            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
                normal_near = pos_normal;
            }

            if t1 > t_enter {
                t_enter = t1;
                hit_normal = normal_near;
                hit_axis = i as i32;
            }
            if t2 < t_exit {
                t_exit = t2;
            }

            if t_enter > t_exit {
                return None;
            }
        }
    }

    if t_enter < 0.0 || t_enter > dt {
        // Check if already overlapping
        if t_enter <= 0.0 && t_exit >= 0.0 {
            return Some((0.0, hit_normal));
        }
        return None;
    }

    Some((t_enter, hit_normal))
}

// ===========================================================================
// Sphere vs Plane sweep
// ===========================================================================

/// A plane defined by a point and normal.
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub point: Vec3,
    pub normal: Vec3,
}

/// Compute the time of first contact between a moving sphere and an infinite plane.
///
/// The sphere sweeps along `vel * t`. Contact occurs when:
///   dot(pos + vel * t - plane_point, plane_normal) = radius
///
/// # Returns
/// `Some(toi)` if contact within [0, dt].
pub fn sweep_sphere_plane(
    pos: Vec3,
    vel: Vec3,
    radius: f32,
    plane: &Plane,
    dt: f32,
) -> Option<f32> {
    let normal = plane.normal.normalize();

    // Initial signed distance from sphere center to plane
    let initial_dist = (pos - plane.point).dot(normal);

    // If already penetrating
    if initial_dist.abs() <= radius {
        return Some(0.0);
    }

    // Rate of approach
    let vel_toward = -vel.dot(normal); // positive = approaching

    if vel_toward.abs() < EPSILON {
        return None; // Moving parallel to plane
    }

    // Time when distance equals radius
    let toi = if initial_dist > 0.0 {
        // Above the plane
        (initial_dist - radius) / vel_toward
    } else {
        // Below the plane
        (initial_dist + radius) / vel_toward
    };

    if toi >= 0.0 && toi <= dt {
        Some(toi)
    } else {
        None
    }
}

/// Full sweep result for sphere vs plane.
pub fn sweep_sphere_plane_full(
    pos: Vec3,
    vel: Vec3,
    radius: f32,
    plane: &Plane,
    dt: f32,
) -> Option<SweepResult> {
    let toi = sweep_sphere_plane(pos, vel, radius, plane, dt)?;
    let normal = plane.normal.normalize();
    let hit_pos = pos + vel * toi;
    let contact = hit_pos - normal * radius;

    Some(SweepResult {
        toi,
        normal,
        contact_point: contact,
    })
}

// ===========================================================================
// Sphere vs Triangle sweep
// ===========================================================================

/// A triangle defined by three vertices.
#[derive(Debug, Clone, Copy)]
pub struct Triangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
}

impl Triangle {
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3) -> Self {
        Self { v0, v1, v2 }
    }

    /// Compute the face normal (not normalized).
    pub fn normal(&self) -> Vec3 {
        (self.v1 - self.v0).cross(self.v2 - self.v0)
    }

    /// Compute the normalized face normal.
    pub fn unit_normal(&self) -> Vec3 {
        let n = self.normal();
        let len = n.length();
        if len > EPSILON {
            n / len
        } else {
            Vec3::Y
        }
    }

    /// Compute the closest point on the triangle to a given point.
    pub fn closest_point(&self, p: Vec3) -> Vec3 {
        let ab = self.v1 - self.v0;
        let ac = self.v2 - self.v0;
        let ap = p - self.v0;

        let d1 = ab.dot(ap);
        let d2 = ac.dot(ap);
        if d1 <= 0.0 && d2 <= 0.0 {
            return self.v0;
        }

        let bp = p - self.v1;
        let d3 = ab.dot(bp);
        let d4 = ac.dot(bp);
        if d3 >= 0.0 && d4 <= d3 {
            return self.v1;
        }

        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            let v = d1 / (d1 - d3);
            return self.v0 + ab * v;
        }

        let cp = p - self.v2;
        let d5 = ab.dot(cp);
        let d6 = ac.dot(cp);
        if d6 >= 0.0 && d5 <= d6 {
            return self.v2;
        }

        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            let w = d2 / (d2 - d6);
            return self.v0 + ac * w;
        }

        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return self.v1 + (self.v2 - self.v1) * w;
        }

        let denom = 1.0 / (va + vb + vc);
        let v = vb * denom;
        let w = vc * denom;
        self.v0 + ab * v + ac * w
    }
}

/// Compute the time of first contact between a moving sphere and a triangle.
///
/// Uses a two-phase approach:
/// 1. Test sphere sweep against the triangle plane
/// 2. Verify the contact point is inside the triangle (barycentric test)
/// 3. If not, test against triangle edges and vertices
///
/// # Returns
/// `Some((toi, normal))` if contact within [0, dt].
pub fn sweep_sphere_triangle(
    pos: Vec3,
    vel: Vec3,
    radius: f32,
    tri: &Triangle,
    dt: f32,
) -> Option<(f32, Vec3)> {
    let tri_normal = tri.unit_normal();

    // Phase 1: Sweep against the triangle plane
    let plane = Plane {
        point: tri.v0,
        normal: tri_normal,
    };

    if let Some(toi) = sweep_sphere_plane(pos, vel, radius, &plane, dt) {
        // Check if the contact point is inside the triangle
        let hit_center = pos + vel * toi;
        let contact_on_plane = hit_center - tri_normal * radius;

        if point_in_triangle(contact_on_plane, tri) {
            return Some((toi, tri_normal));
        }
    }

    // Phase 2: Test against edges and vertices using closest-point approach
    // Conservative advancement: iteratively advance until closest distance = radius
    let mut t = 0.0;
    let mut best_toi: Option<f32> = None;
    let mut best_normal = Vec3::ZERO;

    for _ in 0..MAX_CONSERVATIVE_ITERATIONS {
        let current_pos = pos + vel * t;
        let closest = tri.closest_point(current_pos);
        let diff = current_pos - closest;
        let dist = diff.length();

        if dist <= radius + CONSERVATIVE_THRESHOLD {
            // Contact found
            let normal = if dist > EPSILON {
                diff / dist
            } else {
                tri_normal
            };
            best_toi = Some(t);
            best_normal = normal;
            break;
        }

        // Advance by the distance to contact minus radius
        let approach_speed = vel.dot(-diff.normalize_or_zero()).max(EPSILON);
        let advance = (dist - radius) / approach_speed;

        if advance < EPSILON {
            break;
        }

        t += advance;
        if t > dt {
            break;
        }
    }

    best_toi.map(|toi| (toi, best_normal))
}

/// Check if a point lies inside a triangle (barycentric coordinates).
fn point_in_triangle(p: Vec3, tri: &Triangle) -> bool {
    let v0 = tri.v2 - tri.v0;
    let v1 = tri.v1 - tri.v0;
    let v2 = p - tri.v0;

    let dot00 = v0.dot(v0);
    let dot01 = v0.dot(v1);
    let dot02 = v0.dot(v2);
    let dot11 = v1.dot(v1);
    let dot12 = v1.dot(v2);

    let inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    u >= -EPSILON && v >= -EPSILON && (u + v) <= 1.0 + EPSILON
}

// ===========================================================================
// Conservative Advancement
// ===========================================================================

/// Conservative advancement for general convex pair CCD.
///
/// Iteratively advances time until the closest distance between two
/// convex shapes equals zero (or a tolerance threshold).
///
/// # Arguments
/// * `closest_distance_fn` - Returns the closest distance between shapes at time `t`
/// * `approach_speed_fn` - Returns the maximum approach speed at time `t`
/// * `dt` - Time window
///
/// # Returns
/// `Some(toi)` if contact is found within [0, dt].
pub fn conservative_advancement(
    closest_distance_fn: &dyn Fn(f32) -> f32,
    approach_speed_fn: &dyn Fn(f32) -> f32,
    dt: f32,
    tolerance: f32,
    max_iterations: usize,
) -> Option<f32> {
    let mut t = 0.0;

    for _ in 0..max_iterations {
        let dist = closest_distance_fn(t);

        if dist <= tolerance {
            return Some(t);
        }

        let speed = approach_speed_fn(t).max(EPSILON);
        let advance = (dist - tolerance).max(0.0) / speed;

        if advance < EPSILON * 0.1 {
            // Not making progress
            return if dist < tolerance * 2.0 { Some(t) } else { None };
        }

        t += advance;

        if t > dt {
            return None;
        }
    }

    None
}

// ===========================================================================
// CCD Integration helpers
// ===========================================================================

/// Determine whether a body should use CCD based on its velocity and size.
///
/// A body should use CCD when its velocity is high enough that it could
/// travel more than a fraction of its size in one timestep.
pub fn should_use_ccd(velocity: Vec3, size: f32, dt: f32, fraction: f32) -> bool {
    let travel_distance = velocity.length() * dt;
    travel_distance > size * fraction
}

/// Resolve a TOI collision by advancing the body to the contact point
/// and reflecting the velocity component along the contact normal.
///
/// # Returns
/// The remaining time after the collision.
pub fn resolve_toi(
    position: &mut Vec3,
    velocity: &mut Vec3,
    toi: f32,
    normal: Vec3,
    dt: f32,
    restitution: f32,
) -> f32 {
    // Advance to TOI
    *position += *velocity * toi;

    // Reflect velocity
    let vel_normal = velocity.dot(normal) * normal;
    let vel_tangent = *velocity - vel_normal;

    // Apply restitution to normal component
    *velocity = vel_tangent - vel_normal * restitution;

    // Return remaining time
    dt - toi
}

/// Perform CCD for a sphere moving through a set of triangles.
///
/// Returns the earliest contact, if any.
pub fn ccd_sphere_vs_triangles(
    pos: Vec3,
    vel: Vec3,
    radius: f32,
    triangles: &[Triangle],
    dt: f32,
) -> Option<SweepResult> {
    let mut earliest_toi = dt + 1.0;
    let mut earliest_normal = Vec3::ZERO;
    let mut earliest_contact = Vec3::ZERO;
    let mut found = false;

    for tri in triangles {
        if let Some((toi, normal)) = sweep_sphere_triangle(pos, vel, radius, tri, dt) {
            if toi < earliest_toi {
                earliest_toi = toi;
                earliest_normal = normal;
                earliest_contact = pos + vel * toi - normal * radius;
                found = true;
            }
        }
    }

    if found {
        Some(SweepResult {
            toi: earliest_toi,
            normal: earliest_normal,
            contact_point: earliest_contact,
        })
    } else {
        None
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_sphere_head_on() {
        // Two spheres moving toward each other
        let toi = sweep_sphere_sphere(
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            1.0,
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(-10.0, 0.0, 0.0),
            1.0,
            1.0,
        );

        assert!(toi.is_some());
        let t = toi.unwrap();
        // Distance = 10, sum_radii = 2, relative speed = 20
        // TOI = (10 - 2) / 20 = 0.4
        assert!((t - 0.4).abs() < 0.01, "TOI = {}", t);
    }

    #[test]
    fn test_sphere_sphere_miss() {
        // Spheres moving parallel, never collide
        let toi = sweep_sphere_sphere(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            0.5,
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            0.5,
            10.0,
        );

        assert!(toi.is_none());
    }

    #[test]
    fn test_sphere_sphere_already_overlapping() {
        let toi = sweep_sphere_sphere(
            Vec3::ZERO,
            Vec3::X,
            1.0,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::NEG_X,
            1.0,
            1.0,
        );

        assert!(toi.is_some());
        assert_eq!(toi.unwrap(), 0.0); // Already overlapping
    }

    #[test]
    fn test_sphere_sphere_stationary() {
        let toi = sweep_sphere_sphere(
            Vec3::ZERO,
            Vec3::ZERO,
            1.0,
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::ZERO,
            1.0,
            1.0,
        );

        assert!(toi.is_none()); // Not moving, not touching
    }

    #[test]
    fn test_sweep_aabb_aabb_collision() {
        let a = SweepAABB::from_center(Vec3::new(-5.0, 0.0, 0.0), Vec3::splat(0.5));
        let b = SweepAABB::from_center(Vec3::new(5.0, 0.0, 0.0), Vec3::splat(0.5));

        let result = sweep_aabb_aabb(
            &a,
            Vec3::new(20.0, 0.0, 0.0),
            &b,
            Vec3::ZERO,
            1.0,
        );

        assert!(result.is_some());
        let (toi, normal) = result.unwrap();
        // Distance between surfaces = 10 - 1 = 9, speed = 20
        // TOI ~ 9/20 = 0.45
        assert!(toi > 0.0 && toi < 1.0, "TOI = {}", toi);
    }

    #[test]
    fn test_sweep_aabb_aabb_miss() {
        let a = SweepAABB::from_center(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(0.5));
        let b = SweepAABB::from_center(Vec3::new(0.0, 10.0, 0.0), Vec3::splat(0.5));

        let result = sweep_aabb_aabb(
            &a,
            Vec3::new(1.0, 0.0, 0.0), // Moving in X, target is in Y
            &b,
            Vec3::ZERO,
            1.0,
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_sphere_plane_collision() {
        let toi = sweep_sphere_plane(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, -10.0, 0.0),
            0.5,
            &Plane {
                point: Vec3::ZERO,
                normal: Vec3::Y,
            },
            1.0,
        );

        assert!(toi.is_some());
        let t = toi.unwrap();
        // Distance = 5 - 0.5 = 4.5, speed = 10
        assert!((t - 0.45).abs() < 0.01, "TOI = {}", t);
    }

    #[test]
    fn test_sphere_plane_miss() {
        let toi = sweep_sphere_plane(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0), // Moving away
            0.5,
            &Plane {
                point: Vec3::ZERO,
                normal: Vec3::Y,
            },
            1.0,
        );

        assert!(toi.is_none());
    }

    #[test]
    fn test_sphere_plane_already_penetrating() {
        let toi = sweep_sphere_plane(
            Vec3::new(0.0, 0.3, 0.0), // Within radius of plane
            Vec3::new(0.0, -1.0, 0.0),
            0.5,
            &Plane {
                point: Vec3::ZERO,
                normal: Vec3::Y,
            },
            1.0,
        );

        assert!(toi.is_some());
        assert_eq!(toi.unwrap(), 0.0);
    }

    #[test]
    fn test_point_in_triangle() {
        let tri = Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );

        assert!(point_in_triangle(Vec3::new(0.2, 0.0, 0.2), &tri));
        assert!(!point_in_triangle(Vec3::new(2.0, 0.0, 2.0), &tri));
    }

    #[test]
    fn test_sphere_triangle_direct_hit() {
        let tri = Triangle::new(
            Vec3::new(-5.0, 0.0, -5.0),
            Vec3::new(5.0, 0.0, -5.0),
            Vec3::new(0.0, 0.0, 5.0),
        );

        let result = sweep_sphere_triangle(
            Vec3::new(0.0, 5.0, 0.0), // Above triangle
            Vec3::new(0.0, -10.0, 0.0), // Moving down
            0.5,
            &tri,
            1.0,
        );

        assert!(result.is_some());
        let (toi, normal) = result.unwrap();
        assert!(toi > 0.0 && toi < 1.0);
        // Normal should be roughly +Y (triangle face normal)
        assert!(normal.y > 0.5, "Normal should point up: {:?}", normal);
    }

    #[test]
    fn test_closest_point_on_triangle() {
        let tri = Triangle::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0));

        // Point directly above the center of the triangle
        let closest = tri.closest_point(Vec3::new(0.25, 5.0, 0.25));
        assert!(
            (closest - Vec3::new(0.25, 0.0, 0.25)).length() < 0.01,
            "Closest = {:?}",
            closest
        );
    }

    #[test]
    fn test_ccd_settings() {
        let settings = CCDSettings::default();
        assert!(settings.should_use_ccd(10.0));
        assert!(!settings.should_use_ccd(1.0));
    }

    #[test]
    fn test_should_use_ccd_fn() {
        assert!(should_use_ccd(Vec3::new(100.0, 0.0, 0.0), 1.0, 1.0 / 60.0, 0.5));
        assert!(!should_use_ccd(Vec3::new(0.1, 0.0, 0.0), 1.0, 1.0 / 60.0, 0.5));
    }

    #[test]
    fn test_resolve_toi() {
        let mut pos = Vec3::ZERO;
        let mut vel = Vec3::new(0.0, -10.0, 0.0);
        let remaining = resolve_toi(&mut pos, &mut vel, 0.5, Vec3::Y, 1.0, 0.5);

        assert!((remaining - 0.5).abs() < 1e-4);
        assert!(vel.y > 0.0, "Velocity should be reflected upward");
    }

    #[test]
    fn test_conservative_advancement() {
        // Simple test: distance decreases linearly at speed 10, starting from 5
        let result = conservative_advancement(
            &|t| 5.0 - 10.0 * t, // distance function
            &|_t| 10.0,           // approach speed
            1.0,
            0.01,
            32,
        );

        assert!(result.is_some());
        let toi = result.unwrap();
        assert!((toi - 0.499).abs() < 0.01, "TOI = {}", toi);
    }

    #[test]
    fn test_sweep_sphere_sphere_full() {
        let result = sweep_sphere_sphere_full(
            Vec3::new(-3.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            0.5,
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::ZERO,
            0.5,
            1.0,
        );

        assert!(result.is_some());
        let sr = result.unwrap();
        assert!(sr.toi > 0.0);
        assert!(sr.normal.x < -0.9, "Normal should point in -X");
    }

    #[test]
    fn test_ccd_sphere_vs_triangles() {
        let triangles = vec![Triangle::new(
            Vec3::new(-10.0, 0.0, -10.0),
            Vec3::new(10.0, 0.0, -10.0),
            Vec3::new(0.0, 0.0, 10.0),
        )];

        let result = ccd_sphere_vs_triangles(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, -100.0, 0.0), // Very fast
            0.5,
            &triangles,
            1.0,
        );

        assert!(result.is_some(), "Fast sphere should still hit triangle via CCD");
    }
}
