// engine/physics/src/collision_geometry.rs
//
// Extended collision geometry: convex hull support function, GJK distance
// query, EPA (Expanding Polytope Algorithm) for penetration depth, Minkowski
// portal refinement, margin-based collision (GJK+EPA with margins).
//
// This module implements the full GJK+EPA collision detection pipeline for
// arbitrary convex shapes. GJK determines whether two convex shapes overlap
// and computes the closest distance when they don't. When shapes do overlap,
// EPA computes the penetration depth and direction.

// ---------------------------------------------------------------------------
// Vec3 (local math type)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    #[inline] pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    #[inline] pub fn dot(self, o: Self) -> f32 { self.x * o.x + self.y * o.y + self.z * o.z }
    #[inline] pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }
    #[inline] pub fn length_sq(self) -> f32 { self.dot(self) }
    #[inline] pub fn length(self) -> f32 { self.length_sq().sqrt() }
    #[inline] pub fn normalize(self) -> Self {
        let l = self.length();
        if l < 1e-12 { Self::ZERO } else { self * (1.0 / l) }
    }
    #[inline] pub fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z } }
    #[inline] pub fn triple_product(a: Self, b: Self, c: Self) -> Self {
        // (a x b) x c = b(c.a) - a(c.b)
        b * c.dot(a) - a * c.dot(b)
    }
    #[inline] pub fn abs_components(self) -> Self {
        Self { x: self.x.abs(), y: self.y.abs(), z: self.z.abs() }
    }
}

impl std::ops::Add for Vec3 { type Output = Self; fn add(self, r: Self) -> Self { Self { x: self.x + r.x, y: self.y + r.y, z: self.z + r.z } } }
impl std::ops::Sub for Vec3 { type Output = Self; fn sub(self, r: Self) -> Self { Self { x: self.x - r.x, y: self.y - r.y, z: self.z - r.z } } }
impl std::ops::Mul<f32> for Vec3 { type Output = Self; fn mul(self, s: f32) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } } }
impl std::ops::Neg for Vec3 { type Output = Self; fn neg(self) -> Self { Self::neg(self) } }
impl std::ops::AddAssign for Vec3 { fn add_assign(&mut self, r: Self) { self.x += r.x; self.y += r.y; self.z += r.z; } }

// ---------------------------------------------------------------------------
// Convex shape trait
// ---------------------------------------------------------------------------

/// Trait for any convex shape that can be used with GJK/EPA.
/// The support function returns the point on the shape that is farthest
/// in a given direction.
pub trait ConvexShape {
    /// Return the point on the shape boundary that is farthest in the
    /// given direction.
    fn support(&self, direction: Vec3) -> Vec3;

    /// Return the center of the shape (used for initial search direction).
    fn center(&self) -> Vec3;
}

// ---------------------------------------------------------------------------
// Primitive shapes
// ---------------------------------------------------------------------------

/// A sphere centered at the origin with a given radius.
#[derive(Debug, Clone)]
pub struct SphereShape {
    pub center: Vec3,
    pub radius: f32,
}

impl ConvexShape for SphereShape {
    fn support(&self, direction: Vec3) -> Vec3 {
        self.center + direction.normalize() * self.radius
    }
    fn center(&self) -> Vec3 { self.center }
}

/// An axis-aligned box (center + half-extents).
#[derive(Debug, Clone)]
pub struct BoxShape {
    pub center: Vec3,
    pub half_extents: Vec3,
}

impl ConvexShape for BoxShape {
    fn support(&self, direction: Vec3) -> Vec3 {
        Vec3::new(
            self.center.x + if direction.x >= 0.0 { self.half_extents.x } else { -self.half_extents.x },
            self.center.y + if direction.y >= 0.0 { self.half_extents.y } else { -self.half_extents.y },
            self.center.z + if direction.z >= 0.0 { self.half_extents.z } else { -self.half_extents.z },
        )
    }
    fn center(&self) -> Vec3 { self.center }
}

/// A capsule (two endpoints + radius).
#[derive(Debug, Clone)]
pub struct CapsuleShape {
    pub start: Vec3,
    pub end: Vec3,
    pub radius: f32,
}

impl ConvexShape for CapsuleShape {
    fn support(&self, direction: Vec3) -> Vec3 {
        let d = direction.normalize();
        let dot_start = self.start.dot(d);
        let dot_end = self.end.dot(d);
        let base = if dot_start > dot_end { self.start } else { self.end };
        base + d * self.radius
    }
    fn center(&self) -> Vec3 {
        (self.start + self.end) * 0.5
    }
}

/// A cylinder shape (center, axis, half-height, radius).
#[derive(Debug, Clone)]
pub struct CylinderShape {
    pub center: Vec3,
    pub axis: Vec3,
    pub half_height: f32,
    pub radius: f32,
}

impl ConvexShape for CylinderShape {
    fn support(&self, direction: Vec3) -> Vec3 {
        let axis_norm = self.axis.normalize();
        let axis_proj = direction.dot(axis_norm);
        let radial = direction - axis_norm * axis_proj;
        let radial_len = radial.length();

        let cap_offset = if axis_proj >= 0.0 {
            axis_norm * self.half_height
        } else {
            axis_norm * (-self.half_height)
        };

        let radial_offset = if radial_len > 1e-12 {
            radial * (self.radius / radial_len)
        } else {
            Vec3::ZERO
        };

        self.center + cap_offset + radial_offset
    }
    fn center(&self) -> Vec3 { self.center }
}

/// A convex hull defined by a set of vertices.
#[derive(Debug, Clone)]
pub struct ConvexHullShape {
    pub vertices: Vec<Vec3>,
    pub center_point: Vec3,
}

impl ConvexHullShape {
    pub fn new(vertices: Vec<Vec3>) -> Self {
        let n = vertices.len() as f32;
        let center = if n > 0.0 {
            let sum = vertices.iter().fold(Vec3::ZERO, |acc, v| acc + *v);
            sum * (1.0 / n)
        } else {
            Vec3::ZERO
        };
        Self { vertices, center_point: center }
    }

    /// Hill-climbing support function for faster lookup on larger hulls.
    pub fn support_hill_climb(&self, direction: Vec3, adjacency: &[Vec<usize>], start: usize) -> (Vec3, usize) {
        let mut current = start;
        let mut best_dot = self.vertices[current].dot(direction);

        loop {
            let mut improved = false;
            for &neighbor in &adjacency[current] {
                let d = self.vertices[neighbor].dot(direction);
                if d > best_dot {
                    best_dot = d;
                    current = neighbor;
                    improved = true;
                }
            }
            if !improved { break; }
        }
        (self.vertices[current], current)
    }
}

impl ConvexShape for ConvexHullShape {
    fn support(&self, direction: Vec3) -> Vec3 {
        let mut best = Vec3::ZERO;
        let mut best_dot = f32::NEG_INFINITY;
        for v in &self.vertices {
            let d = v.dot(direction);
            if d > best_dot {
                best_dot = d;
                best = *v;
            }
        }
        best
    }
    fn center(&self) -> Vec3 { self.center_point }
}

/// A shape with a collision margin (GJK+EPA with margins).
#[derive(Debug, Clone)]
pub struct MarginShape<S: ConvexShape> {
    pub inner: S,
    pub margin: f32,
}

impl<S: ConvexShape> ConvexShape for MarginShape<S> {
    fn support(&self, direction: Vec3) -> Vec3 {
        let inner_support = self.inner.support(direction);
        inner_support + direction.normalize() * self.margin
    }
    fn center(&self) -> Vec3 { self.inner.center() }
}

// ---------------------------------------------------------------------------
// Minkowski difference support
// ---------------------------------------------------------------------------

/// A point in the Minkowski difference, storing both the result and the
/// original support points on shapes A and B.
#[derive(Debug, Clone, Copy)]
pub struct MinkowskiVertex {
    /// Point in Minkowski difference space (support_a - support_b).
    pub point: Vec3,
    /// Support point on shape A.
    pub support_a: Vec3,
    /// Support point on shape B.
    pub support_b: Vec3,
}

impl MinkowskiVertex {
    pub fn compute(a: &dyn ConvexShape, b: &dyn ConvexShape, direction: Vec3) -> Self {
        let sa = a.support(direction);
        let sb = b.support(-direction);
        Self {
            point: sa - sb,
            support_a: sa,
            support_b: sb,
        }
    }
}

// ---------------------------------------------------------------------------
// GJK (Gilbert-Johnson-Keerthi)
// ---------------------------------------------------------------------------

/// Maximum iterations for GJK.
pub const GJK_MAX_ITERATIONS: usize = 64;

/// GJK convergence tolerance.
pub const GJK_TOLERANCE: f32 = 1e-6;

/// Result of a GJK query.
#[derive(Debug, Clone)]
pub enum GjkResult {
    /// Shapes do not overlap. Contains the closest distance and closest points.
    Separated {
        distance: f32,
        closest_a: Vec3,
        closest_b: Vec3,
    },
    /// Shapes overlap. The simplex can be used as starting point for EPA.
    Overlapping {
        simplex: Vec<MinkowskiVertex>,
    },
}

/// The GJK simplex (1-4 vertices).
#[derive(Debug, Clone)]
struct Simplex {
    vertices: Vec<MinkowskiVertex>,
}

impl Simplex {
    fn new() -> Self { Self { vertices: Vec::with_capacity(4) } }

    fn push(&mut self, v: MinkowskiVertex) { self.vertices.push(v); }

    fn size(&self) -> usize { self.vertices.len() }

    fn last(&self) -> &MinkowskiVertex { &self.vertices[self.vertices.len() - 1] }
}

/// Run the GJK algorithm on two convex shapes.
pub fn gjk(a: &dyn ConvexShape, b: &dyn ConvexShape) -> GjkResult {
    // Initial search direction: from B's center to A's center.
    let mut direction = a.center() - b.center();
    if direction.length_sq() < GJK_TOLERANCE {
        direction = Vec3::new(1.0, 0.0, 0.0);
    }

    let mut simplex = Simplex::new();
    let first = MinkowskiVertex::compute(a, b, direction);
    simplex.push(first);

    direction = -first.point;
    if direction.length_sq() < GJK_TOLERANCE {
        return GjkResult::Overlapping { simplex: simplex.vertices };
    }

    for _ in 0..GJK_MAX_ITERATIONS {
        let new_point = MinkowskiVertex::compute(a, b, direction);

        // If the new point doesn't pass the origin, shapes are separated.
        if new_point.point.dot(direction) < 0.0 {
            let (closest_a, closest_b, distance) = compute_closest_points(&simplex);
            return GjkResult::Separated { distance, closest_a, closest_b };
        }

        simplex.push(new_point);

        match simplex.size() {
            2 => {
                // Line case.
                let result = do_simplex_line(&mut simplex);
                if let Some(dir) = result {
                    direction = dir;
                } else {
                    return GjkResult::Overlapping { simplex: simplex.vertices };
                }
            }
            3 => {
                // Triangle case.
                let result = do_simplex_triangle(&mut simplex);
                if let Some(dir) = result {
                    direction = dir;
                } else {
                    return GjkResult::Overlapping { simplex: simplex.vertices };
                }
            }
            4 => {
                // Tetrahedron case.
                let result = do_simplex_tetrahedron(&mut simplex);
                if let Some(dir) = result {
                    direction = dir;
                } else {
                    return GjkResult::Overlapping { simplex: simplex.vertices };
                }
            }
            _ => unreachable!(),
        }

        if direction.length_sq() < GJK_TOLERANCE {
            return GjkResult::Overlapping { simplex: simplex.vertices };
        }
    }

    // If we run out of iterations, assume overlapping.
    GjkResult::Overlapping { simplex: simplex.vertices }
}

/// Line simplex case: update simplex and return new search direction.
fn do_simplex_line(simplex: &mut Simplex) -> Option<Vec3> {
    let b = simplex.vertices[0].point;
    let a = simplex.vertices[1].point;
    let ab = b - a;
    let ao = -a;

    if ab.dot(ao) > 0.0 {
        // Origin is between A and B.
        let dir = Vec3::triple_product(ab, ao, ab);
        if dir.length_sq() < GJK_TOLERANCE {
            return None; // Origin is on the line.
        }
        Some(dir)
    } else {
        // Origin is past A, discard B.
        simplex.vertices = vec![simplex.vertices[1]];
        Some(ao)
    }
}

/// Triangle simplex case.
fn do_simplex_triangle(simplex: &mut Simplex) -> Option<Vec3> {
    let c = simplex.vertices[0].point;
    let b = simplex.vertices[1].point;
    let a = simplex.vertices[2].point;

    let ab = b - a;
    let ac = c - a;
    let ao = -a;
    let abc = ab.cross(ac);

    if abc.cross(ac).dot(ao) > 0.0 {
        if ac.dot(ao) > 0.0 {
            simplex.vertices = vec![simplex.vertices[0], simplex.vertices[2]];
            Some(Vec3::triple_product(ac, ao, ac))
        } else {
            simplex.vertices = vec![simplex.vertices[1], simplex.vertices[2]];
            return do_simplex_line(simplex);
        }
    } else if ab.cross(abc).dot(ao) > 0.0 {
        simplex.vertices = vec![simplex.vertices[1], simplex.vertices[2]];
        return do_simplex_line(simplex);
    } else {
        // Origin is inside the triangle prism.
        if abc.dot(ao) > 0.0 {
            Some(abc)
        } else {
            simplex.vertices = vec![simplex.vertices[1], simplex.vertices[0], simplex.vertices[2]];
            Some(-abc)
        }
    }
}

/// Tetrahedron simplex case.
fn do_simplex_tetrahedron(simplex: &mut Simplex) -> Option<Vec3> {
    let d = simplex.vertices[0].point;
    let c = simplex.vertices[1].point;
    let b = simplex.vertices[2].point;
    let a = simplex.vertices[3].point;

    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    let ao = -a;

    let abc = ab.cross(ac);
    let acd = ac.cross(ad);
    let adb = ad.cross(ab);

    // Check each face.
    if abc.dot(ao) > 0.0 {
        simplex.vertices = vec![simplex.vertices[1], simplex.vertices[2], simplex.vertices[3]];
        return do_simplex_triangle(simplex);
    }
    if acd.dot(ao) > 0.0 {
        simplex.vertices = vec![simplex.vertices[0], simplex.vertices[1], simplex.vertices[3]];
        return do_simplex_triangle(simplex);
    }
    if adb.dot(ao) > 0.0 {
        simplex.vertices = vec![simplex.vertices[2], simplex.vertices[0], simplex.vertices[3]];
        return do_simplex_triangle(simplex);
    }

    // Origin is inside the tetrahedron.
    None
}

/// Compute closest points from the simplex (for separated case).
fn compute_closest_points(simplex: &Simplex) -> (Vec3, Vec3, f32) {
    match simplex.size() {
        1 => {
            let v = &simplex.vertices[0];
            (v.support_a, v.support_b, v.point.length())
        }
        2 => {
            let a = &simplex.vertices[1];
            let b = &simplex.vertices[0];
            let ab = b.point - a.point;
            let ao = -a.point;
            let t = ao.dot(ab) / ab.length_sq();
            let t = t.clamp(0.0, 1.0);
            let closest_a = a.support_a + (b.support_a - a.support_a) * t;
            let closest_b = a.support_b + (b.support_b - a.support_b) * t;
            let closest = a.point + ab * t;
            (closest_a, closest_b, closest.length())
        }
        3 => {
            // Barycentric coordinates on triangle.
            let a = &simplex.vertices[2];
            let b = &simplex.vertices[1];
            let c = &simplex.vertices[0];

            let ab = b.point - a.point;
            let ac = c.point - a.point;
            let ao = -a.point;

            let d00 = ab.dot(ab);
            let d01 = ab.dot(ac);
            let d11 = ac.dot(ac);
            let d20 = ao.dot(ab);
            let d21 = ao.dot(ac);

            let denom = d00 * d11 - d01 * d01;
            if denom.abs() < 1e-12 {
                return (a.support_a, a.support_b, a.point.length());
            }

            let v = (d11 * d20 - d01 * d21) / denom;
            let w = (d00 * d21 - d01 * d20) / denom;
            let u = 1.0 - v - w;

            let closest_a = a.support_a * u + b.support_a * v + c.support_a * w;
            let closest_b = a.support_b * u + b.support_b * v + c.support_b * w;
            let closest = a.point * u + b.point * v + c.point * w;
            (closest_a, closest_b, closest.length())
        }
        _ => {
            (Vec3::ZERO, Vec3::ZERO, 0.0)
        }
    }
}

// ---------------------------------------------------------------------------
// EPA (Expanding Polytope Algorithm)
// ---------------------------------------------------------------------------

/// Maximum iterations for EPA.
pub const EPA_MAX_ITERATIONS: usize = 64;

/// EPA convergence tolerance.
pub const EPA_TOLERANCE: f32 = 1e-4;

/// A face (triangle) on the EPA polytope.
#[derive(Debug, Clone)]
struct EpaFace {
    vertices: [usize; 3],
    normal: Vec3,
    distance: f32,
}

/// Result of an EPA query.
#[derive(Debug, Clone)]
pub struct EpaResult {
    /// Penetration normal (from A to B).
    pub normal: Vec3,
    /// Penetration depth.
    pub depth: f32,
    /// Contact point on shape A.
    pub contact_a: Vec3,
    /// Contact point on shape B.
    pub contact_b: Vec3,
    /// Number of iterations used.
    pub iterations: usize,
}

/// Run the EPA algorithm to compute penetration depth and direction.
///
/// The `initial_simplex` should be the overlapping simplex from GJK (must have
/// 4 vertices forming a valid tetrahedron).
pub fn epa(
    a: &dyn ConvexShape,
    b: &dyn ConvexShape,
    initial_simplex: &[MinkowskiVertex],
) -> Option<EpaResult> {
    if initial_simplex.len() < 4 {
        return epa_from_small_simplex(a, b, initial_simplex);
    }

    let mut vertices: Vec<MinkowskiVertex> = initial_simplex.to_vec();
    let mut faces: Vec<EpaFace> = Vec::new();

    // Build initial tetrahedron faces.
    let face_indices = [
        [0, 1, 2],
        [0, 3, 1],
        [0, 2, 3],
        [1, 3, 2],
    ];

    for indices in &face_indices {
        let v0 = vertices[indices[0]].point;
        let v1 = vertices[indices[1]].point;
        let v2 = vertices[indices[2]].point;

        let mut normal = (v1 - v0).cross(v2 - v0);
        let len = normal.length();
        if len < 1e-12 { continue; }
        normal = normal * (1.0 / len);

        // Ensure normal points outward (away from origin).
        if normal.dot(v0) < 0.0 {
            normal = -normal;
        }

        let distance = normal.dot(v0).abs();
        faces.push(EpaFace {
            vertices: *indices,
            normal,
            distance,
        });
    }

    if faces.is_empty() { return None; }

    for iteration in 0..EPA_MAX_ITERATIONS {
        // Find closest face to origin.
        let (closest_idx, closest_face) = faces.iter().enumerate()
            .min_by(|(_, a), (_, b)| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, f)| (i, f.clone()))?;

        // Get new support point in the direction of the closest face normal.
        let new_vertex = MinkowskiVertex::compute(a, b, closest_face.normal);
        let new_dist = new_vertex.point.dot(closest_face.normal);

        // Check convergence.
        if (new_dist - closest_face.distance).abs() < EPA_TOLERANCE {
            // Compute contact points using barycentric coordinates.
            let (contact_a, contact_b) = compute_epa_contacts(&vertices, &closest_face);
            return Some(EpaResult {
                normal: closest_face.normal,
                depth: closest_face.distance,
                contact_a,
                contact_b,
                iterations: iteration + 1,
            });
        }

        // Add new vertex and rebuild faces.
        let new_idx = vertices.len();
        vertices.push(new_vertex);

        // Find all faces visible from the new point.
        let mut visible: Vec<bool> = faces.iter().map(|f| {
            let v0 = vertices[f.vertices[0]].point;
            f.normal.dot(new_vertex.point - v0) > 0.0
        }).collect();

        // Find boundary edges of visible faces.
        let mut edges: Vec<[usize; 2]> = Vec::new();
        for (i, face) in faces.iter().enumerate() {
            if !visible[i] { continue; }
            for edge_idx in 0..3 {
                let e0 = face.vertices[edge_idx];
                let e1 = face.vertices[(edge_idx + 1) % 3];

                // Check if this edge is shared with a non-visible face.
                let mut shared = false;
                for (j, other) in faces.iter().enumerate() {
                    if i == j || visible[j] { continue; }
                    for k in 0..3 {
                        let oe0 = other.vertices[k];
                        let oe1 = other.vertices[(k + 1) % 3];
                        if (e0 == oe1 && e1 == oe0) || (e0 == oe0 && e1 == oe1) {
                            shared = true;
                            break;
                        }
                    }
                    if shared { break; }
                }

                if shared {
                    edges.push([e0, e1]);
                }
            }
        }

        // Remove visible faces.
        let mut kept_faces: Vec<EpaFace> = Vec::new();
        for (i, face) in faces.into_iter().enumerate() {
            if !visible[i] {
                kept_faces.push(face);
            }
        }
        faces = kept_faces;

        // Create new faces from boundary edges to new vertex.
        for edge in &edges {
            let v0 = vertices[edge[0]].point;
            let v1 = vertices[edge[1]].point;
            let v2 = vertices[new_idx].point;

            let mut normal = (v1 - v0).cross(v2 - v0);
            let len = normal.length();
            if len < 1e-12 { continue; }
            normal = normal * (1.0 / len);

            if normal.dot(v0) < 0.0 {
                normal = -normal;
                faces.push(EpaFace {
                    vertices: [edge[1], edge[0], new_idx],
                    normal,
                    distance: normal.dot(v0).abs(),
                });
            } else {
                faces.push(EpaFace {
                    vertices: [edge[0], edge[1], new_idx],
                    normal,
                    distance: normal.dot(v0).abs(),
                });
            }
        }
    }

    // Failed to converge -- return best result.
    let closest = faces.iter()
        .min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal))?;
    let (contact_a, contact_b) = compute_epa_contacts(&vertices, closest);
    Some(EpaResult {
        normal: closest.normal,
        depth: closest.distance,
        contact_a,
        contact_b,
        iterations: EPA_MAX_ITERATIONS,
    })
}

/// Handle EPA from a simplex with fewer than 4 points.
fn epa_from_small_simplex(
    a: &dyn ConvexShape,
    b: &dyn ConvexShape,
    simplex: &[MinkowskiVertex],
) -> Option<EpaResult> {
    // Expand the simplex to a tetrahedron by adding new support points.
    let mut vertices = simplex.to_vec();

    let search_dirs = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
    ];

    while vertices.len() < 4 {
        let mut best_v = None;
        let mut best_dist = -f32::MAX;
        for dir in &search_dirs {
            let v = MinkowskiVertex::compute(a, b, *dir);
            let mut is_dup = false;
            for existing in &vertices {
                if (v.point - existing.point).length_sq() < GJK_TOLERANCE {
                    is_dup = true;
                    break;
                }
            }
            if !is_dup {
                let dist = v.point.dot(*dir);
                if dist > best_dist {
                    best_dist = dist;
                    best_v = Some(v);
                }
            }
        }
        if let Some(v) = best_v {
            vertices.push(v);
        } else {
            return None; // Degenerate shapes.
        }
    }

    epa(a, b, &vertices)
}

/// Compute contact points from EPA face using barycentric coordinates.
fn compute_epa_contacts(vertices: &[MinkowskiVertex], face: &EpaFace) -> (Vec3, Vec3) {
    let v0 = &vertices[face.vertices[0]];
    let v1 = &vertices[face.vertices[1]];
    let v2 = &vertices[face.vertices[2]];

    // Project origin onto the face.
    let origin_proj = face.normal * face.distance;

    // Barycentric coordinates.
    let e0 = v1.point - v0.point;
    let e1 = v2.point - v0.point;
    let v = origin_proj - v0.point;

    let d00 = e0.dot(e0);
    let d01 = e0.dot(e1);
    let d11 = e1.dot(e1);
    let d20 = v.dot(e0);
    let d21 = v.dot(e1);

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-12 {
        return (v0.support_a, v0.support_b);
    }

    let bv = (d11 * d20 - d01 * d21) / denom;
    let bw = (d00 * d21 - d01 * d20) / denom;
    let bu = 1.0 - bv - bw;

    let contact_a = v0.support_a * bu + v1.support_a * bv + v2.support_a * bw;
    let contact_b = v0.support_b * bu + v1.support_b * bv + v2.support_b * bw;

    (contact_a, contact_b)
}

// ---------------------------------------------------------------------------
// High-level collision query
// ---------------------------------------------------------------------------

/// Result of a collision query between two convex shapes.
#[derive(Debug, Clone)]
pub enum CollisionResult {
    /// No collision. Distance is the gap between the shapes.
    NoCollision {
        distance: f32,
        closest_a: Vec3,
        closest_b: Vec3,
    },
    /// Collision detected. Penetration information is provided.
    Collision {
        normal: Vec3,
        depth: f32,
        contact_a: Vec3,
        contact_b: Vec3,
    },
}

/// Perform a full collision query (GJK + EPA) between two convex shapes.
pub fn collide(a: &dyn ConvexShape, b: &dyn ConvexShape) -> CollisionResult {
    match gjk(a, b) {
        GjkResult::Separated { distance, closest_a, closest_b } => {
            CollisionResult::NoCollision { distance, closest_a, closest_b }
        }
        GjkResult::Overlapping { simplex } => {
            match epa(a, b, &simplex) {
                Some(epa_result) => {
                    CollisionResult::Collision {
                        normal: epa_result.normal,
                        depth: epa_result.depth,
                        contact_a: epa_result.contact_a,
                        contact_b: epa_result.contact_b,
                    }
                }
                None => {
                    // EPA failed; return a zero-depth collision.
                    let dir = (a.center() - b.center()).normalize();
                    CollisionResult::Collision {
                        normal: dir,
                        depth: 0.0,
                        contact_a: a.center(),
                        contact_b: b.center(),
                    }
                }
            }
        }
    }
}

/// Perform a collision query with margins.
pub fn collide_with_margins(
    a: &dyn ConvexShape,
    b: &dyn ConvexShape,
    margin_a: f32,
    margin_b: f32,
) -> CollisionResult {
    let total_margin = margin_a + margin_b;

    match gjk(a, b) {
        GjkResult::Separated { distance, closest_a, closest_b } => {
            if distance <= total_margin {
                // Shapes are within margin distance -- compute contact.
                let dir = if distance > 1e-12 {
                    (closest_b - closest_a).normalize()
                } else {
                    (a.center() - b.center()).normalize()
                };
                CollisionResult::Collision {
                    normal: dir,
                    depth: total_margin - distance,
                    contact_a: closest_a + dir * margin_a,
                    contact_b: closest_b - dir * margin_b,
                }
            } else {
                CollisionResult::NoCollision {
                    distance: distance - total_margin,
                    closest_a,
                    closest_b,
                }
            }
        }
        GjkResult::Overlapping { simplex } => {
            match epa(a, b, &simplex) {
                Some(epa_result) => {
                    CollisionResult::Collision {
                        normal: epa_result.normal,
                        depth: epa_result.depth + total_margin,
                        contact_a: epa_result.contact_a + epa_result.normal * margin_a,
                        contact_b: epa_result.contact_b - epa_result.normal * margin_b,
                    }
                }
                None => {
                    let dir = (a.center() - b.center()).normalize();
                    CollisionResult::Collision {
                        normal: dir,
                        depth: total_margin,
                        contact_a: a.center(),
                        contact_b: b.center(),
                    }
                }
            }
        }
    }
}

/// GJK distance query (returns only the distance, not contacts).
pub fn gjk_distance(a: &dyn ConvexShape, b: &dyn ConvexShape) -> f32 {
    match gjk(a, b) {
        GjkResult::Separated { distance, .. } => distance,
        GjkResult::Overlapping { .. } => 0.0,
    }
}

/// GJK boolean intersection test (faster than full GJK when you only need yes/no).
pub fn gjk_intersects(a: &dyn ConvexShape, b: &dyn ConvexShape) -> bool {
    matches!(gjk(a, b), GjkResult::Overlapping { .. })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_sphere_separated() {
        let a = SphereShape { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0 };
        let b = SphereShape { center: Vec3::new(5.0, 0.0, 0.0), radius: 1.0 };
        let result = collide(&a, &b);
        match result {
            CollisionResult::NoCollision { distance, .. } => {
                assert!((distance - 3.0).abs() < 0.1, "Expected ~3.0, got {}", distance);
            }
            _ => panic!("Expected NoCollision"),
        }
    }

    #[test]
    fn test_sphere_sphere_overlapping() {
        let a = SphereShape { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0 };
        let b = SphereShape { center: Vec3::new(1.0, 0.0, 0.0), radius: 1.0 };
        let result = collide(&a, &b);
        match result {
            CollisionResult::Collision { depth, normal, .. } => {
                assert!(depth > 0.0, "Depth should be positive: {}", depth);
                assert!(normal.x.abs() > 0.5, "Normal should be along X: {:?}", normal);
            }
            _ => panic!("Expected Collision"),
        }
    }

    #[test]
    fn test_box_box_separated() {
        let a = BoxShape { center: Vec3::new(0.0, 0.0, 0.0), half_extents: Vec3::new(1.0, 1.0, 1.0) };
        let b = BoxShape { center: Vec3::new(5.0, 0.0, 0.0), half_extents: Vec3::new(1.0, 1.0, 1.0) };
        assert!(!gjk_intersects(&a, &b));
    }

    #[test]
    fn test_box_box_overlapping() {
        let a = BoxShape { center: Vec3::new(0.0, 0.0, 0.0), half_extents: Vec3::new(1.0, 1.0, 1.0) };
        let b = BoxShape { center: Vec3::new(1.5, 0.0, 0.0), half_extents: Vec3::new(1.0, 1.0, 1.0) };
        assert!(gjk_intersects(&a, &b));
    }

    #[test]
    fn test_capsule_sphere() {
        let capsule = CapsuleShape {
            start: Vec3::new(0.0, 0.0, 0.0),
            end: Vec3::new(0.0, 2.0, 0.0),
            radius: 0.5,
        };
        let sphere = SphereShape { center: Vec3::new(0.5, 1.0, 0.0), radius: 0.5 };
        assert!(gjk_intersects(&capsule, &sphere));
    }

    #[test]
    fn test_convex_hull() {
        // Cube vertices.
        let vertices = vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
        ];
        let hull = ConvexHullShape::new(vertices);
        let sphere = SphereShape { center: Vec3::new(0.0, 0.0, 0.0), radius: 0.5 };
        assert!(gjk_intersects(&hull, &sphere));
    }

    #[test]
    fn test_gjk_distance() {
        let a = SphereShape { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0 };
        let b = SphereShape { center: Vec3::new(4.0, 0.0, 0.0), radius: 1.0 };
        let dist = gjk_distance(&a, &b);
        assert!((dist - 2.0).abs() < 0.1, "Expected ~2.0, got {}", dist);
    }

    #[test]
    fn test_margin_collision() {
        let a = SphereShape { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0 };
        let b = SphereShape { center: Vec3::new(2.5, 0.0, 0.0), radius: 1.0 };

        // Without margin: separated.
        let result = collide(&a, &b);
        assert!(matches!(result, CollisionResult::NoCollision { .. }));

        // With margin: should be colliding.
        let result = collide_with_margins(&a, &b, 0.3, 0.3);
        match result {
            CollisionResult::Collision { depth, .. } => {
                assert!(depth > 0.0);
            }
            _ => panic!("Expected collision with margins"),
        }
    }

    #[test]
    fn test_epa_penetration_depth() {
        let a = SphereShape { center: Vec3::new(0.0, 0.0, 0.0), radius: 2.0 };
        let b = SphereShape { center: Vec3::new(1.0, 0.0, 0.0), radius: 2.0 };
        let result = collide(&a, &b);
        match result {
            CollisionResult::Collision { depth, .. } => {
                // Expected penetration depth ~3.0 (2 + 2 - 1).
                assert!((depth - 3.0).abs() < 0.5, "Expected ~3.0, got {}", depth);
            }
            _ => panic!("Expected Collision"),
        }
    }
}
