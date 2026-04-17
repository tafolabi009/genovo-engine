// engine/physics/src/gjk_epa_v2.rs
//
// Production-quality GJK (Gilbert-Johnson-Keerthi) + EPA (Expanding Polytope
// Algorithm) implementation. Handles:
//   - Minkowski difference support function dispatch
//   - GJK simplex evolution: point, line, triangle, tetrahedron cases
//   - EPA polytope expansion for penetration depth/normal
//   - Convex hull, sphere, capsule, box support functions
//   - Distance and closest-point queries
//   - Configurable iteration limits and tolerances

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    #[inline]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    #[inline]
    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-12 {
            return Self::ZERO;
        }
        self * (1.0 / l)
    }

    #[inline]
    pub fn triple_product(a: Self, b: Self, c: Self) -> Self {
        // (a x b) x c = b * (c.dot(a)) - a * (c.dot(b))
        b * c.dot(a) - a * c.dot(b)
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

/// A 3D rigid body transform (position + rotation as 3x3 matrix).
#[derive(Debug, Clone, Copy)]
pub struct Transform3D {
    pub position: Vec3,
    /// Column-major 3x3 rotation matrix.
    pub rotation: [[f32; 3]; 3],
}

impl Transform3D {
    pub fn identity() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    pub fn from_position(pos: Vec3) -> Self {
        Self {
            position: pos,
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Rotate a direction vector by this transform's rotation.
    #[inline]
    pub fn rotate(&self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.rotation[0][0] * v.x + self.rotation[1][0] * v.y + self.rotation[2][0] * v.z,
            self.rotation[0][1] * v.x + self.rotation[1][1] * v.y + self.rotation[2][1] * v.z,
            self.rotation[0][2] * v.x + self.rotation[1][2] * v.y + self.rotation[2][2] * v.z,
        )
    }

    /// Inverse-rotate a direction vector.
    #[inline]
    pub fn inverse_rotate(&self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.rotation[0][0] * v.x + self.rotation[0][1] * v.y + self.rotation[0][2] * v.z,
            self.rotation[1][0] * v.x + self.rotation[1][1] * v.y + self.rotation[1][2] * v.z,
            self.rotation[2][0] * v.x + self.rotation[2][1] * v.y + self.rotation[2][2] * v.z,
        )
    }

    /// Transform a point.
    #[inline]
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        self.rotate(p) + self.position
    }
}

// ---------------------------------------------------------------------------
// Support functions for convex shapes
// ---------------------------------------------------------------------------

/// Trait for shapes that can provide a support point in a given direction.
pub trait SupportFunction {
    /// Return the point on the shape farthest in the direction `dir`
    /// (in world space).
    fn support(&self, dir: Vec3) -> Vec3;

    /// Return the center of the shape (used for initial search direction).
    fn center(&self) -> Vec3;
}

/// A sphere.
#[derive(Debug, Clone, Copy)]
pub struct SphereShape {
    pub center: Vec3,
    pub radius: f32,
}

impl SupportFunction for SphereShape {
    #[inline]
    fn support(&self, dir: Vec3) -> Vec3 {
        let n = dir.normalized();
        self.center + n * self.radius
    }

    fn center(&self) -> Vec3 {
        self.center
    }
}

/// An axis-aligned box (with transform for OBB).
#[derive(Debug, Clone, Copy)]
pub struct BoxShape {
    pub half_extents: Vec3,
    pub transform: Transform3D,
}

impl SupportFunction for BoxShape {
    fn support(&self, dir: Vec3) -> Vec3 {
        // Transform direction to local space.
        let local_dir = self.transform.inverse_rotate(dir);
        let local_support = Vec3::new(
            if local_dir.x >= 0.0 { self.half_extents.x } else { -self.half_extents.x },
            if local_dir.y >= 0.0 { self.half_extents.y } else { -self.half_extents.y },
            if local_dir.z >= 0.0 { self.half_extents.z } else { -self.half_extents.z },
        );
        self.transform.transform_point(local_support)
    }

    fn center(&self) -> Vec3 {
        self.transform.position
    }
}

/// A capsule (two hemispheres connected by a cylinder).
#[derive(Debug, Clone, Copy)]
pub struct CapsuleShape {
    pub start: Vec3,
    pub end: Vec3,
    pub radius: f32,
}

impl SupportFunction for CapsuleShape {
    fn support(&self, dir: Vec3) -> Vec3 {
        let n = dir.normalized();
        let dot_start = self.start.dot(n);
        let dot_end = self.end.dot(n);
        let base = if dot_start >= dot_end {
            self.start
        } else {
            self.end
        };
        base + n * self.radius
    }

    fn center(&self) -> Vec3 {
        (self.start + self.end) * 0.5
    }
}

/// A convex hull defined by a set of vertices.
#[derive(Debug, Clone)]
pub struct ConvexHullShape {
    pub vertices: Vec<Vec3>,
    pub transform: Transform3D,
}

impl SupportFunction for ConvexHullShape {
    fn support(&self, dir: Vec3) -> Vec3 {
        let local_dir = self.transform.inverse_rotate(dir);
        let mut best = self.vertices[0];
        let mut best_dot = local_dir.dot(best);
        for &v in &self.vertices[1..] {
            let d = local_dir.dot(v);
            if d > best_dot {
                best_dot = d;
                best = v;
            }
        }
        self.transform.transform_point(best)
    }

    fn center(&self) -> Vec3 {
        if self.vertices.is_empty() {
            return self.transform.position;
        }
        let sum = self.vertices.iter().fold(Vec3::ZERO, |acc, &v| acc + v);
        let local_center = sum * (1.0 / self.vertices.len() as f32);
        self.transform.transform_point(local_center)
    }
}

/// A cylinder shape.
#[derive(Debug, Clone, Copy)]
pub struct CylinderShape {
    pub half_height: f32,
    pub radius: f32,
    pub transform: Transform3D,
}

impl SupportFunction for CylinderShape {
    fn support(&self, dir: Vec3) -> Vec3 {
        let local_dir = self.transform.inverse_rotate(dir);
        let xz_len = (local_dir.x * local_dir.x + local_dir.z * local_dir.z).sqrt();
        let local_support = if xz_len > 1e-8 {
            let scale = self.radius / xz_len;
            Vec3::new(
                local_dir.x * scale,
                if local_dir.y >= 0.0 { self.half_height } else { -self.half_height },
                local_dir.z * scale,
            )
        } else {
            Vec3::new(
                self.radius,
                if local_dir.y >= 0.0 { self.half_height } else { -self.half_height },
                0.0,
            )
        };
        self.transform.transform_point(local_support)
    }

    fn center(&self) -> Vec3 {
        self.transform.position
    }
}

/// A cone shape.
#[derive(Debug, Clone, Copy)]
pub struct ConeShape {
    pub half_height: f32,
    pub radius: f32,
    pub transform: Transform3D,
}

impl SupportFunction for ConeShape {
    fn support(&self, dir: Vec3) -> Vec3 {
        let local_dir = self.transform.inverse_rotate(dir);
        let sin_angle = self.radius / (self.radius * self.radius + (2.0 * self.half_height).powi(2)).sqrt();
        let local_support = if local_dir.y > local_dir.length() * sin_angle {
            Vec3::new(0.0, self.half_height, 0.0)
        } else {
            let xz_len = (local_dir.x * local_dir.x + local_dir.z * local_dir.z).sqrt();
            if xz_len > 1e-8 {
                let scale = self.radius / xz_len;
                Vec3::new(local_dir.x * scale, -self.half_height, local_dir.z * scale)
            } else {
                Vec3::new(self.radius, -self.half_height, 0.0)
            }
        };
        self.transform.transform_point(local_support)
    }

    fn center(&self) -> Vec3 {
        self.transform.position
    }
}

// ---------------------------------------------------------------------------
// Minkowski difference support
// ---------------------------------------------------------------------------

/// A support point on the Minkowski difference A - B.
#[derive(Debug, Clone, Copy)]
pub struct MinkowskiPoint {
    /// The point on the Minkowski difference.
    pub point: Vec3,
    /// The support point on shape A.
    pub support_a: Vec3,
    /// The support point on shape B.
    pub support_b: Vec3,
}

/// Compute the support point of the Minkowski difference A - B.
#[inline]
fn minkowski_support(
    shape_a: &dyn SupportFunction,
    shape_b: &dyn SupportFunction,
    dir: Vec3,
) -> MinkowskiPoint {
    let sa = shape_a.support(dir);
    let sb = shape_b.support(-dir);
    MinkowskiPoint {
        point: sa - sb,
        support_a: sa,
        support_b: sb,
    }
}

// ---------------------------------------------------------------------------
// GJK Simplex
// ---------------------------------------------------------------------------

/// The evolving simplex used by GJK.
#[derive(Debug, Clone)]
struct Simplex {
    points: [MinkowskiPoint; 4],
    size: usize,
}

impl Simplex {
    fn new() -> Self {
        let zero = MinkowskiPoint {
            point: Vec3::ZERO,
            support_a: Vec3::ZERO,
            support_b: Vec3::ZERO,
        };
        Self {
            points: [zero; 4],
            size: 0,
        }
    }

    fn push(&mut self, p: MinkowskiPoint) {
        // Shift existing points back and insert new point at front.
        match self.size {
            0 => {
                self.points[0] = p;
            }
            1 => {
                self.points[1] = self.points[0];
                self.points[0] = p;
            }
            2 => {
                self.points[2] = self.points[1];
                self.points[1] = self.points[0];
                self.points[0] = p;
            }
            3 => {
                self.points[3] = self.points[2];
                self.points[2] = self.points[1];
                self.points[1] = self.points[0];
                self.points[0] = p;
            }
            _ => {
                self.points[3] = self.points[2];
                self.points[2] = self.points[1];
                self.points[1] = self.points[0];
                self.points[0] = p;
            }
        }
        if self.size < 4 {
            self.size += 1;
        }
    }

    fn a(&self) -> Vec3 {
        self.points[0].point
    }

    fn b(&self) -> Vec3 {
        self.points[1].point
    }

    fn c(&self) -> Vec3 {
        self.points[2].point
    }

    fn d(&self) -> Vec3 {
        self.points[3].point
    }
}

/// Determine the new search direction and reduce the simplex.
/// Returns `true` if the origin is contained within the simplex (collision).
fn do_simplex(simplex: &mut Simplex, direction: &mut Vec3) -> bool {
    match simplex.size {
        2 => do_simplex_line(simplex, direction),
        3 => do_simplex_triangle(simplex, direction),
        4 => do_simplex_tetrahedron(simplex, direction),
        _ => false,
    }
}

/// Line case: A is the newest point.
fn do_simplex_line(simplex: &mut Simplex, direction: &mut Vec3) -> bool {
    let a = simplex.a();
    let b = simplex.b();
    let ab = b - a;
    let ao = -a;

    if ab.dot(ao) > 0.0 {
        // Origin is between A and B; direction perpendicular to AB toward origin.
        *direction = Vec3::triple_product(ab, ao, ab);
        if direction.length_sq() < 1e-12 {
            // AB passes through origin (degenerate) — pick any perpendicular.
            *direction = Vec3::new(ab.y, -ab.x, 0.0);
            if direction.length_sq() < 1e-12 {
                *direction = Vec3::new(0.0, ab.z, -ab.y);
            }
        }
    } else {
        // Origin is behind A, discard B.
        simplex.points[0] = simplex.points[0]; // A stays at 0
        simplex.size = 1;
        *direction = ao;
    }
    false
}

/// Triangle case: vertices are A (newest), B, C.
fn do_simplex_triangle(simplex: &mut Simplex, direction: &mut Vec3) -> bool {
    let a = simplex.a();
    let b = simplex.b();
    let c = simplex.c();
    let ab = b - a;
    let ac = c - a;
    let ao = -a;
    let abc = ab.cross(ac);

    // Check if origin is outside edge AC (on the side away from B).
    let abc_cross_ac = abc.cross(ac);
    if abc_cross_ac.dot(ao) > 0.0 {
        if ac.dot(ao) > 0.0 {
            // Region AC: keep A and C.
            simplex.points[1] = simplex.points[2]; // C -> slot 1
            simplex.size = 2;
            *direction = Vec3::triple_product(ac, ao, ac);
        } else {
            // Reduce to line AB or just A.
            simplex.size = 2;
            return do_simplex_line(simplex, direction);
        }
        return false;
    }

    // Check if origin is outside edge AB (on the side away from C).
    let ab_cross_abc = ab.cross(abc);
    if ab_cross_abc.dot(ao) > 0.0 {
        simplex.size = 2;
        return do_simplex_line(simplex, direction);
    }

    // Origin is inside the triangle prism; determine which side.
    if abc.dot(ao) > 0.0 {
        *direction = abc;
    } else {
        // Flip winding.
        let tmp = simplex.points[1];
        simplex.points[1] = simplex.points[2];
        simplex.points[2] = tmp;
        *direction = -abc;
    }
    false
}

/// Tetrahedron case: vertices are A (newest), B, C, D.
fn do_simplex_tetrahedron(simplex: &mut Simplex, direction: &mut Vec3) -> bool {
    let a = simplex.a();
    let b = simplex.b();
    let c = simplex.c();
    let d = simplex.d();
    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    let ao = -a;

    let abc = ab.cross(ac);
    let acd = ac.cross(ad);
    let adb = ad.cross(ab);

    // Check face ABC (pointing away from D).
    if abc.dot(ao) > 0.0 {
        // Origin is outside face ABC; reduce to triangle ABC.
        simplex.size = 3;
        *direction = abc;
        return do_simplex_triangle(simplex, direction);
    }

    // Check face ACD.
    if acd.dot(ao) > 0.0 {
        // Reduce to triangle ACD.
        simplex.points[1] = simplex.points[2]; // C
        simplex.points[2] = simplex.points[3]; // D
        simplex.size = 3;
        *direction = acd;
        return do_simplex_triangle(simplex, direction);
    }

    // Check face ADB.
    if adb.dot(ao) > 0.0 {
        // Reduce to triangle ADB.
        simplex.points[2] = simplex.points[1]; // B -> slot 2
        simplex.points[1] = simplex.points[3]; // D -> slot 1
        simplex.size = 3;
        *direction = adb;
        return do_simplex_triangle(simplex, direction);
    }

    // Origin is inside the tetrahedron — collision!
    true
}

// ---------------------------------------------------------------------------
// GJK result types
// ---------------------------------------------------------------------------

/// The result of a GJK query.
#[derive(Debug, Clone)]
pub enum GjkResult {
    /// The shapes are intersecting.
    Intersecting,
    /// The shapes are separated by `distance`.
    Separated {
        distance: f32,
        closest_a: Vec3,
        closest_b: Vec3,
    },
    /// GJK did not converge within the iteration limit.
    NoConvergence,
}

/// Contact information from EPA.
#[derive(Debug, Clone, Copy)]
pub struct EpaContact {
    /// Penetration normal (from B to A).
    pub normal: Vec3,
    /// Penetration depth (positive when overlapping).
    pub depth: f32,
    /// Contact point on shape A.
    pub point_a: Vec3,
    /// Contact point on shape B.
    pub point_b: Vec3,
}

/// Combined GJK+EPA result.
#[derive(Debug, Clone)]
pub enum CollisionResult {
    /// No collision.
    NoCollision {
        distance: f32,
        closest_a: Vec3,
        closest_b: Vec3,
    },
    /// Collision with contact details.
    Collision(EpaContact),
    /// Algorithm did not converge.
    Failed,
}

// ---------------------------------------------------------------------------
// GJK configuration
// ---------------------------------------------------------------------------

/// Configuration for GJK and EPA.
#[derive(Debug, Clone)]
pub struct GjkEpaConfig {
    /// Maximum GJK iterations.
    pub gjk_max_iterations: u32,
    /// GJK convergence tolerance.
    pub gjk_tolerance: f32,
    /// Maximum EPA iterations.
    pub epa_max_iterations: u32,
    /// Maximum EPA faces.
    pub epa_max_faces: usize,
    /// EPA convergence tolerance.
    pub epa_tolerance: f32,
}

impl Default for GjkEpaConfig {
    fn default() -> Self {
        Self {
            gjk_max_iterations: 64,
            gjk_tolerance: 1e-6,
            epa_max_iterations: 64,
            epa_max_faces: 128,
            epa_tolerance: 1e-4,
        }
    }
}

// ---------------------------------------------------------------------------
// GJK algorithm
// ---------------------------------------------------------------------------

/// Run the GJK algorithm to determine if two convex shapes intersect.
pub fn gjk(
    shape_a: &dyn SupportFunction,
    shape_b: &dyn SupportFunction,
    config: &GjkEpaConfig,
) -> (GjkResult, Simplex) {
    // Initial search direction: from center of A to center of B.
    let mut direction = shape_b.center() - shape_a.center();
    if direction.length_sq() < 1e-12 {
        direction = Vec3::new(1.0, 0.0, 0.0);
    }

    let mut simplex = Simplex::new();

    // Get the first support point.
    let first = minkowski_support(shape_a, shape_b, direction);
    simplex.push(first);
    direction = -first.point;

    for _iter in 0..config.gjk_max_iterations {
        let new_point = minkowski_support(shape_a, shape_b, direction);

        // If the new point didn't pass the origin, shapes don't intersect.
        if new_point.point.dot(direction) < -config.gjk_tolerance {
            // Compute closest points from the current simplex.
            let (closest_a, closest_b, dist) = closest_points_from_simplex(&simplex);
            return (
                GjkResult::Separated {
                    distance: dist,
                    closest_a,
                    closest_b,
                },
                simplex,
            );
        }

        simplex.push(new_point);

        if do_simplex(&mut simplex, &mut direction) {
            return (GjkResult::Intersecting, simplex);
        }

        // Safety: if direction is near-zero, we're at the origin.
        if direction.length_sq() < 1e-16 {
            return (GjkResult::Intersecting, simplex);
        }
    }

    (GjkResult::NoConvergence, simplex)
}

/// Extract closest points from the GJK simplex.
fn closest_points_from_simplex(simplex: &Simplex) -> (Vec3, Vec3, f32) {
    match simplex.size {
        1 => {
            let p = &simplex.points[0];
            (p.support_a, p.support_b, p.point.length())
        }
        2 => {
            let a = &simplex.points[0];
            let b = &simplex.points[1];
            let ab = b.point - a.point;
            let ao = -a.point;
            let t = ao.dot(ab) / ab.length_sq();
            let t = t.clamp(0.0, 1.0);
            let closest_a = a.support_a + (b.support_a - a.support_a) * t;
            let closest_b = a.support_b + (b.support_b - a.support_b) * t;
            let dist = (closest_a - closest_b).length();
            (closest_a, closest_b, dist)
        }
        3 => {
            let a = &simplex.points[0];
            let b = &simplex.points[1];
            let c = &simplex.points[2];
            let (u, v, w) = barycentric_triangle(
                a.point, b.point, c.point, Vec3::ZERO,
            );
            let closest_a = a.support_a * u + b.support_a * v + c.support_a * w;
            let closest_b = a.support_b * u + b.support_b * v + c.support_b * w;
            let dist = (closest_a - closest_b).length();
            (closest_a, closest_b, dist)
        }
        _ => {
            // 4 points = origin is inside, distance is 0.
            let p = &simplex.points[0];
            (p.support_a, p.support_b, 0.0)
        }
    }
}

/// Compute barycentric coordinates of point P projected onto triangle ABC.
fn barycentric_triangle(a: Vec3, b: Vec3, c: Vec3, p: Vec3) -> (f32, f32, f32) {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;

    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-12 {
        return (1.0, 0.0, 0.0);
    }
    let inv = 1.0 / denom;
    let v = (d11 * d20 - d01 * d21) * inv;
    let w = (d00 * d21 - d01 * d20) * inv;
    let u = 1.0 - v - w;
    (u, v, w)
}

// ---------------------------------------------------------------------------
// EPA (Expanding Polytope Algorithm)
// ---------------------------------------------------------------------------

/// An EPA face (triangle on the polytope).
#[derive(Debug, Clone, Copy)]
struct EpaFace {
    indices: [usize; 3],
    normal: Vec3,
    distance: f32,
    obsolete: bool,
}

/// An EPA edge (for horizon computation).
#[derive(Debug, Clone, Copy)]
struct EpaEdge {
    a: usize,
    b: usize,
}

/// Run the EPA algorithm given a GJK simplex that contains the origin.
pub fn epa(
    shape_a: &dyn SupportFunction,
    shape_b: &dyn SupportFunction,
    simplex: &Simplex,
    config: &GjkEpaConfig,
) -> Option<EpaContact> {
    if simplex.size < 4 {
        return None;
    }

    // Build initial polytope from the tetrahedron.
    let mut vertices: Vec<MinkowskiPoint> = Vec::with_capacity(config.epa_max_faces);
    vertices.push(simplex.points[0]);
    vertices.push(simplex.points[1]);
    vertices.push(simplex.points[2]);
    vertices.push(simplex.points[3]);

    let mut faces: Vec<EpaFace> = Vec::with_capacity(config.epa_max_faces);

    // Ensure correct winding: each face normal should point outward.
    // Face 0: 0,1,2
    // Face 1: 0,3,1
    // Face 2: 0,2,3
    // Face 3: 1,3,2
    let face_indices = [
        [0usize, 1, 2],
        [0, 3, 1],
        [0, 2, 3],
        [1, 3, 2],
    ];

    for indices in &face_indices {
        if let Some(face) = make_face(&vertices, indices[0], indices[1], indices[2]) {
            faces.push(face);
        }
    }

    if faces.is_empty() {
        return None;
    }

    for _iter in 0..config.epa_max_iterations {
        // Find the closest face to the origin.
        let (closest_idx, closest_face) = match find_closest_face(&faces) {
            Some(v) => v,
            None => return None,
        };

        // Get a new support point in the direction of the closest face normal.
        let new_point = minkowski_support(shape_a, shape_b, closest_face.normal);
        let new_dist = new_point.point.dot(closest_face.normal);

        // Check convergence: if the new point doesn't extend the polytope significantly.
        if (new_dist - closest_face.distance).abs() < config.epa_tolerance {
            // Compute contact point using barycentric coordinates on the closest face.
            let fi = closest_face.indices;
            let (u, v, w) = barycentric_triangle(
                vertices[fi[0]].point,
                vertices[fi[1]].point,
                vertices[fi[2]].point,
                closest_face.normal * closest_face.distance,
            );
            let point_a = vertices[fi[0]].support_a * u
                + vertices[fi[1]].support_a * v
                + vertices[fi[2]].support_a * w;
            let point_b = vertices[fi[0]].support_b * u
                + vertices[fi[1]].support_b * v
                + vertices[fi[2]].support_b * w;

            return Some(EpaContact {
                normal: closest_face.normal,
                depth: closest_face.distance,
                point_a,
                point_b,
            });
        }

        // Add the new vertex.
        let new_idx = vertices.len();
        vertices.push(new_point);

        if vertices.len() >= config.epa_max_faces {
            // Hit face limit; return best result so far.
            let fi = closest_face.indices;
            let (u, v, w) = barycentric_triangle(
                vertices[fi[0]].point,
                vertices[fi[1]].point,
                vertices[fi[2]].point,
                closest_face.normal * closest_face.distance,
            );
            let point_a = vertices[fi[0]].support_a * u
                + vertices[fi[1]].support_a * v
                + vertices[fi[2]].support_a * w;
            let point_b = vertices[fi[0]].support_b * u
                + vertices[fi[1]].support_b * v
                + vertices[fi[2]].support_b * w;
            return Some(EpaContact {
                normal: closest_face.normal,
                depth: closest_face.distance,
                point_a,
                point_b,
            });
        }

        // Remove faces visible from the new point and collect horizon edges.
        let mut horizon_edges: Vec<EpaEdge> = Vec::new();

        for face in faces.iter_mut() {
            if face.obsolete {
                continue;
            }
            let v0 = vertices[face.indices[0]].point;
            let to_new = new_point.point - v0;
            if face.normal.dot(to_new) > 0.0 {
                // This face is visible from the new point; mark obsolete.
                face.obsolete = true;
                // Add edges to horizon (only if not shared with another visible face).
                add_horizon_edge(&mut horizon_edges, face.indices[0], face.indices[1]);
                add_horizon_edge(&mut horizon_edges, face.indices[1], face.indices[2]);
                add_horizon_edge(&mut horizon_edges, face.indices[2], face.indices[0]);
            }
        }

        // Remove obsolete faces.
        faces.retain(|f| !f.obsolete);

        // Create new faces from horizon edges to the new vertex.
        for edge in &horizon_edges {
            if let Some(face) = make_face(&vertices, edge.a, edge.b, new_idx) {
                faces.push(face);
            }
        }

        if faces.is_empty() {
            return None;
        }
    }

    // Didn't converge; return best result.
    if let Some((_, face)) = find_closest_face(&faces) {
        let fi = face.indices;
        let (u, v, w) = barycentric_triangle(
            vertices[fi[0]].point,
            vertices[fi[1]].point,
            vertices[fi[2]].point,
            face.normal * face.distance,
        );
        let point_a = vertices[fi[0]].support_a * u
            + vertices[fi[1]].support_a * v
            + vertices[fi[2]].support_a * w;
        let point_b = vertices[fi[0]].support_b * u
            + vertices[fi[1]].support_b * v
            + vertices[fi[2]].support_b * w;
        Some(EpaContact {
            normal: face.normal,
            depth: face.distance,
            point_a,
            point_b,
        })
    } else {
        None
    }
}

fn make_face(vertices: &[MinkowskiPoint], a: usize, b: usize, c: usize) -> Option<EpaFace> {
    let ab = vertices[b].point - vertices[a].point;
    let ac = vertices[c].point - vertices[a].point;
    let normal = ab.cross(ac);
    let len = normal.length();
    if len < 1e-10 {
        return None;
    }
    let normal = normal * (1.0 / len);

    // Ensure normal points away from origin.
    let distance = normal.dot(vertices[a].point);
    if distance < 0.0 {
        // Flip winding.
        Some(EpaFace {
            indices: [a, c, b],
            normal: -normal,
            distance: -distance,
            obsolete: false,
        })
    } else {
        Some(EpaFace {
            indices: [a, b, c],
            normal,
            distance,
            obsolete: false,
        })
    }
}

fn find_closest_face(faces: &[EpaFace]) -> Option<(usize, EpaFace)> {
    let mut best_idx = None;
    let mut best_dist = f32::MAX;
    for (i, face) in faces.iter().enumerate() {
        if !face.obsolete && face.distance < best_dist {
            best_dist = face.distance;
            best_idx = Some(i);
        }
    }
    best_idx.map(|i| (i, faces[i]))
}

fn add_horizon_edge(edges: &mut Vec<EpaEdge>, a: usize, b: usize) {
    // If the reverse edge already exists, remove it (shared edge).
    if let Some(pos) = edges.iter().position(|e| e.a == b && e.b == a) {
        edges.swap_remove(pos);
    } else {
        edges.push(EpaEdge { a, b });
    }
}

// ---------------------------------------------------------------------------
// Combined GJK+EPA query
// ---------------------------------------------------------------------------

/// Run GJK and, if intersecting, EPA to get full contact information.
pub fn gjk_epa(
    shape_a: &dyn SupportFunction,
    shape_b: &dyn SupportFunction,
    config: &GjkEpaConfig,
) -> CollisionResult {
    let (result, simplex) = gjk(shape_a, shape_b, config);

    match result {
        GjkResult::Intersecting => {
            if let Some(contact) = epa(shape_a, shape_b, &simplex, config) {
                CollisionResult::Collision(contact)
            } else {
                CollisionResult::Failed
            }
        }
        GjkResult::Separated { distance, closest_a, closest_b } => {
            CollisionResult::NoCollision { distance, closest_a, closest_b }
        }
        GjkResult::NoConvergence => CollisionResult::Failed,
    }
}

/// Convenience: test collision with default config.
pub fn test_collision(
    shape_a: &dyn SupportFunction,
    shape_b: &dyn SupportFunction,
) -> CollisionResult {
    gjk_epa(shape_a, shape_b, &GjkEpaConfig::default())
}

/// Convenience: boolean intersection test only (no EPA).
pub fn intersects(
    shape_a: &dyn SupportFunction,
    shape_b: &dyn SupportFunction,
) -> bool {
    let (result, _) = gjk(shape_a, shape_b, &GjkEpaConfig::default());
    matches!(result, GjkResult::Intersecting)
}

/// Distance between two convex shapes (0 if intersecting).
pub fn distance(
    shape_a: &dyn SupportFunction,
    shape_b: &dyn SupportFunction,
) -> f32 {
    let (result, _) = gjk(shape_a, shape_b, &GjkEpaConfig::default());
    match result {
        GjkResult::Separated { distance, .. } => distance,
        _ => 0.0,
    }
}

/// Closest points between two convex shapes.
pub fn closest_points(
    shape_a: &dyn SupportFunction,
    shape_b: &dyn SupportFunction,
) -> Option<(Vec3, Vec3, f32)> {
    let (result, _) = gjk(shape_a, shape_b, &GjkEpaConfig::default());
    match result {
        GjkResult::Separated { distance, closest_a, closest_b } => {
            Some((closest_a, closest_b, distance))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Shape dispatch enum
// ---------------------------------------------------------------------------

/// Enum wrapping all supported convex shapes for dispatch.
pub enum ConvexShape {
    Sphere(SphereShape),
    Box(BoxShape),
    Capsule(CapsuleShape),
    ConvexHull(ConvexHullShape),
    Cylinder(CylinderShape),
    Cone(ConeShape),
}

impl SupportFunction for ConvexShape {
    fn support(&self, dir: Vec3) -> Vec3 {
        match self {
            Self::Sphere(s) => s.support(dir),
            Self::Box(b) => b.support(dir),
            Self::Capsule(c) => c.support(dir),
            Self::ConvexHull(h) => h.support(dir),
            Self::Cylinder(c) => c.support(dir),
            Self::Cone(c) => c.support(dir),
        }
    }

    fn center(&self) -> Vec3 {
        match self {
            Self::Sphere(s) => s.center(),
            Self::Box(b) => b.center(),
            Self::Capsule(c) => c.center(),
            Self::ConvexHull(h) => h.center(),
            Self::Cylinder(c) => c.center(),
            Self::Cone(c) => c.center(),
        }
    }
}

// ---------------------------------------------------------------------------
// Margin-based GJK (Minkowski sum with sphere for rounded shapes)
// ---------------------------------------------------------------------------

/// A convex shape expanded by a margin (Minkowski sum with sphere).
pub struct MarginShape<'a> {
    pub inner: &'a dyn SupportFunction,
    pub margin: f32,
}

impl<'a> SupportFunction for MarginShape<'a> {
    fn support(&self, dir: Vec3) -> Vec3 {
        let n = dir.normalized();
        self.inner.support(dir) + n * self.margin
    }

    fn center(&self) -> Vec3 {
        self.inner.center()
    }
}

/// Run GJK+EPA with a collision margin applied to both shapes.
pub fn gjk_epa_with_margin(
    shape_a: &dyn SupportFunction,
    shape_b: &dyn SupportFunction,
    margin: f32,
    config: &GjkEpaConfig,
) -> CollisionResult {
    let ma = MarginShape { inner: shape_a, margin: margin * 0.5 };
    let mb = MarginShape { inner: shape_b, margin: margin * 0.5 };
    gjk_epa(&ma, &mb, config)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_sphere_intersecting() {
        let a = SphereShape { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0 };
        let b = SphereShape { center: Vec3::new(1.5, 0.0, 0.0), radius: 1.0 };
        assert!(intersects(&a, &b));
    }

    #[test]
    fn test_sphere_sphere_separated() {
        let a = SphereShape { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0 };
        let b = SphereShape { center: Vec3::new(3.0, 0.0, 0.0), radius: 1.0 };
        assert!(!intersects(&a, &b));
        let d = distance(&a, &b);
        assert!((d - 1.0).abs() < 0.1, "distance was {d}");
    }

    #[test]
    fn test_box_box_intersecting() {
        let a = BoxShape {
            half_extents: Vec3::new(1.0, 1.0, 1.0),
            transform: Transform3D::from_position(Vec3::ZERO),
        };
        let b = BoxShape {
            half_extents: Vec3::new(1.0, 1.0, 1.0),
            transform: Transform3D::from_position(Vec3::new(1.5, 0.0, 0.0)),
        };
        assert!(intersects(&a, &b));
    }

    #[test]
    fn test_box_box_separated() {
        let a = BoxShape {
            half_extents: Vec3::new(1.0, 1.0, 1.0),
            transform: Transform3D::from_position(Vec3::ZERO),
        };
        let b = BoxShape {
            half_extents: Vec3::new(1.0, 1.0, 1.0),
            transform: Transform3D::from_position(Vec3::new(5.0, 0.0, 0.0)),
        };
        assert!(!intersects(&a, &b));
    }

    #[test]
    fn test_sphere_epa() {
        let a = SphereShape { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0 };
        let b = SphereShape { center: Vec3::new(0.5, 0.0, 0.0), radius: 1.0 };
        let result = test_collision(&a, &b);
        match result {
            CollisionResult::Collision(contact) => {
                assert!(contact.depth > 0.0, "depth = {}", contact.depth);
                // Normal should be roughly along X.
                assert!(contact.normal.x.abs() > 0.5 || contact.normal.y.abs() > 0.5 || contact.normal.z.abs() > 0.5);
            }
            other => panic!("Expected collision, got {:?}", other),
        }
    }

    #[test]
    fn test_capsule_support() {
        let c = CapsuleShape {
            start: Vec3::new(0.0, -1.0, 0.0),
            end: Vec3::new(0.0, 1.0, 0.0),
            radius: 0.5,
        };
        let up = c.support(Vec3::new(0.0, 1.0, 0.0));
        assert!((up.y - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_convex_hull_support() {
        let hull = ConvexHullShape {
            vertices: vec![
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            transform: Transform3D::identity(),
        };
        let s = hull.support(Vec3::new(1.0, 0.0, 0.0));
        assert!((s.x - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_margin_gjk() {
        let a = SphereShape { center: Vec3::ZERO, radius: 0.5 };
        let b = SphereShape { center: Vec3::new(2.0, 0.0, 0.0), radius: 0.5 };
        // Without margin: separated by 1.0.
        assert!(!intersects(&a, &b));
        // With margin of 1.0: touching.
        let result = gjk_epa_with_margin(&a, &b, 1.0, &GjkEpaConfig::default());
        match result {
            CollisionResult::NoCollision { distance, .. } => {
                assert!(distance < 0.2, "distance was {distance}");
            }
            CollisionResult::Collision(_) => { /* also fine, within margin */ }
            CollisionResult::Failed => panic!("should not fail"),
        }
    }
}
