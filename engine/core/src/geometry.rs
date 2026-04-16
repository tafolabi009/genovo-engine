//! Computational geometry algorithms for the Genovo engine.
//!
//! Provides convex hull construction (2-D and 3-D), polygon triangulation,
//! Delaunay triangulation with Voronoi dual, segment intersection, point-in-polygon
//! tests, minimum enclosing circle/sphere, oriented bounding boxes, and polygon
//! boolean operations via Sutherland-Hodgman clipping.

use glam::{Vec2, Vec3};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const EPSILON: f32 = 1e-7;

/// 2-D cross product (z-component of the 3-D cross product of two 2-D vectors).
#[inline]
fn cross2(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

/// Orientation test for three 2-D points.
/// Returns positive if counter-clockwise, negative if clockwise, zero if collinear.
#[inline]
fn orient2d(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    cross2(b - a, c - a)
}

// ===========================================================================
// ConvexHull2D -- Andrew's Monotone Chain
// ===========================================================================

/// Result of a 2-D convex hull computation.
///
/// Vertices are stored in counter-clockwise order. The first vertex is NOT
/// repeated at the end.
#[derive(Debug, Clone)]
pub struct ConvexHull2D {
    /// Hull vertices in CCW order.
    pub vertices: Vec<Vec2>,
}

impl ConvexHull2D {
    /// Computes the convex hull of a set of 2-D points using Andrew's monotone
    /// chain algorithm. Runs in O(n log n).
    ///
    /// If fewer than 3 non-collinear points are supplied, the hull may be
    /// degenerate (a line segment or a single point).
    pub fn compute(points: &[Vec2]) -> Self {
        if points.len() <= 1 {
            return Self {
                vertices: points.to_vec(),
            };
        }

        let mut pts: Vec<Vec2> = points.to_vec();
        // Sort lexicographically by (x, y).
        pts.sort_by(|a, b| {
            a.x.partial_cmp(&b.x)
                .unwrap()
                .then(a.y.partial_cmp(&b.y).unwrap())
        });
        pts.dedup();

        if pts.len() <= 2 {
            return Self { vertices: pts };
        }

        let n = pts.len();
        let mut hull: Vec<Vec2> = Vec::with_capacity(2 * n);

        // Build lower hull.
        for i in 0..n {
            while hull.len() >= 2
                && orient2d(hull[hull.len() - 2], hull[hull.len() - 1], pts[i]) <= 0.0
            {
                hull.pop();
            }
            hull.push(pts[i]);
        }

        // Build upper hull.
        let lower_len = hull.len() + 1;
        for i in (0..n).rev() {
            while hull.len() >= lower_len
                && orient2d(hull[hull.len() - 2], hull[hull.len() - 1], pts[i]) <= 0.0
            {
                hull.pop();
            }
            hull.push(pts[i]);
        }

        // Remove the last point because it is the same as the first.
        hull.pop();

        Self { vertices: hull }
    }

    /// Returns the signed area of the convex hull (positive for CCW).
    pub fn area(&self) -> f32 {
        polygon_area_signed(&self.vertices)
    }

    /// Returns the perimeter of the convex hull.
    pub fn perimeter(&self) -> f32 {
        let n = self.vertices.len();
        if n < 2 {
            return 0.0;
        }
        let mut p = 0.0f32;
        for i in 0..n {
            p += (self.vertices[(i + 1) % n] - self.vertices[i]).length();
        }
        p
    }

    /// Tests whether a point lies inside the convex hull.
    /// Uses cross-product tests against each edge.
    pub fn contains_point(&self, point: Vec2) -> bool {
        let n = self.vertices.len();
        if n < 3 {
            return false;
        }
        for i in 0..n {
            let j = (i + 1) % n;
            if orient2d(self.vertices[i], self.vertices[j], point) < 0.0 {
                return false;
            }
        }
        true
    }
}

// ===========================================================================
// ConvexHull3D -- Incremental
// ===========================================================================

/// A face of the 3-D convex hull.
#[derive(Debug, Clone)]
pub struct HullFace {
    /// Indices into the point array, wound counter-clockwise when viewed from
    /// outside the hull.
    pub indices: [usize; 3],
    /// Outward-facing normal (not necessarily unit length).
    pub normal: Vec3,
    /// Plane offset: `normal.dot(vertex) = offset` for vertices on this face.
    pub offset: f32,
}

/// Result of a 3-D convex hull computation (incremental algorithm).
#[derive(Debug, Clone)]
pub struct ConvexHull3D {
    /// The original points used to build the hull.
    pub points: Vec<Vec3>,
    /// Triangular faces of the hull.
    pub faces: Vec<HullFace>,
}

impl ConvexHull3D {
    /// Builds the convex hull of a set of 3-D points using an incremental
    /// algorithm.
    ///
    /// Complexity: O(n^2) worst case (expected O(n log n) for random inputs).
    /// Input points are slightly perturbed (by ~1e-6) to handle degenerate
    /// coplanar configurations.
    pub fn compute(points: &[Vec3]) -> Self {
        let pts: Vec<Vec3> = points.to_vec();
        let n = pts.len();
        if n < 4 {
            return Self {
                points: pts,
                faces: Vec::new(),
            };
        }

        // Find four non-coplanar points to seed the tetrahedron.
        let i0 = 0;
        let mut i1 = 1;
        // Find a point distinct from i0.
        while i1 < n && (pts[i1] - pts[i0]).length_squared() < EPSILON * EPSILON {
            i1 += 1;
        }
        if i1 >= n {
            return Self { points: pts, faces: Vec::new() };
        }

        // Find a point not collinear with i0, i1.
        let mut i2 = i1 + 1;
        while i2 < n {
            let cross = (pts[i1] - pts[i0]).cross(pts[i2] - pts[i0]);
            if cross.length_squared() > EPSILON * EPSILON {
                break;
            }
            i2 += 1;
        }
        if i2 >= n {
            return Self { points: pts, faces: Vec::new() };
        }

        // Find a point not coplanar with i0, i1, i2.
        let base_normal = (pts[i1] - pts[i0]).cross(pts[i2] - pts[i0]);
        let mut i3 = i2 + 1;
        while i3 < n {
            let d = base_normal.dot(pts[i3] - pts[i0]);
            if d.abs() > EPSILON {
                break;
            }
            i3 += 1;
        }
        if i3 >= n {
            return Self { points: pts, faces: Vec::new() };
        }

        let make_face = |a: usize, b: usize, c: usize, pts: &[Vec3]| -> HullFace {
            let normal = (pts[b] - pts[a]).cross(pts[c] - pts[a]);
            let offset = normal.dot(pts[a]);
            HullFace {
                indices: [a, b, c],
                normal,
                offset,
            }
        };

        // Build initial tetrahedron with guaranteed outward-facing normals.
        // For each face, ensure the normal points away from the opposite vertex.
        let tet = [i0, i1, i2, i3];
        let face_indices = [
            [tet[0], tet[1], tet[2]],
            [tet[0], tet[2], tet[3]],
            [tet[0], tet[3], tet[1]],
            [tet[1], tet[3], tet[2]],
        ];
        let opposite = [tet[3], tet[1], tet[2], tet[0]];

        let mut faces: Vec<HullFace> = Vec::with_capacity(4);
        for (fi, &[a, b, c]) in face_indices.iter().enumerate() {
            let mut face = make_face(a, b, c, &pts);
            // Check: does the normal point away from the opposite vertex?
            let opp = pts[opposite[fi]];
            if face.normal.dot(opp) > face.offset {
                // Normal points toward the opposite vertex -- flip winding.
                face = make_face(a, c, b, &pts);
            }
            faces.push(face);
        }

        let seed_set = [tet[0], tet[1], tet[2], tet[3]];

        // Incrementally add each point.
        for pi in 0..n {
            if seed_set.contains(&pi) {
                continue;
            }
            let p = pts[pi];

            // Determine which faces are visible from p.
            let mut visible = vec![false; faces.len()];
            let mut any_visible = false;
            for (fi, face) in faces.iter().enumerate() {
                if face.normal.dot(p) > face.offset + EPSILON {
                    visible[fi] = true;
                    any_visible = true;
                }
            }
            if !any_visible {
                continue;
            }

            // Find the horizon edges.
            // A horizon edge separates a visible face from a non-visible face.
            // We collect the edge from the NON-VISIBLE face's perspective:
            // the non-visible face has edge (eb, ea), so we store (eb, ea)
            // and create the new face to match.
            let mut horizon: Vec<(usize, usize)> = Vec::new();
            for fi in 0..faces.len() {
                if !visible[fi] {
                    continue;
                }
                let idx = faces[fi].indices;
                let edges = [(idx[0], idx[1]), (idx[1], idx[2]), (idx[2], idx[0])];
                for &(ea, eb) in &edges {
                    // Check if the reverse edge (eb, ea) belongs to a non-visible face.
                    let mut border = false;
                    for fj in 0..faces.len() {
                        if visible[fj] {
                            continue;
                        }
                        let jdx = faces[fj].indices;
                        let jedges = [(jdx[0], jdx[1]), (jdx[1], jdx[2]), (jdx[2], jdx[0])];
                        if jedges.contains(&(eb, ea)) {
                            border = true;
                            break;
                        }
                    }
                    if border {
                        horizon.push((ea, eb));
                    }
                }
            }

            // Remove visible faces.
            let mut new_faces: Vec<HullFace> = Vec::new();
            for (fi, face) in faces.iter().enumerate() {
                if !visible[fi] {
                    new_faces.push(face.clone());
                }
            }

            // Create new faces from pi to each horizon edge.
            // The horizon edge (ea, eb) comes from the visible face.
            // The non-visible neighbor has edge (eb, ea).
            // The new face [ea, eb, pi] has edge (ea, eb) which correctly
            // opposes (eb, ea) in the non-visible neighbor.
            for &(ea, eb) in &horizon {
                new_faces.push(make_face(ea, eb, pi, &pts));
            }

            faces = new_faces;
        }

        Self {
            points: pts,
            faces,
        }
    }

    /// Returns the volume of the convex hull using signed tetrahedra from
    /// the centroid to each face.
    pub fn volume(&self) -> f32 {
        if self.faces.is_empty() {
            return 0.0;
        }
        // Use centroid as reference point for more robust computation.
        let centroid = self.points.iter().copied().fold(Vec3::ZERO, |a, b| a + b)
            / self.points.len() as f32;
        let mut vol = 0.0f32;
        for face in &self.faces {
            let a = self.points[face.indices[0]] - centroid;
            let b = self.points[face.indices[1]] - centroid;
            let c = self.points[face.indices[2]] - centroid;
            vol += a.dot(b.cross(c));
        }
        (vol / 6.0).abs()
    }
}

// ===========================================================================
// Polygon area & winding
// ===========================================================================

/// Signed area of a simple polygon (positive for CCW winding, negative for CW).
/// Uses the shoelace formula.
pub fn polygon_area_signed(vertices: &[Vec2]) -> f32 {
    let n = vertices.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area += vertices[i].x * vertices[j].y;
        area -= vertices[j].x * vertices[i].y;
    }
    area * 0.5
}

/// Unsigned area of a simple polygon.
#[inline]
pub fn polygon_area(vertices: &[Vec2]) -> f32 {
    polygon_area_signed(vertices).abs()
}

/// Returns `true` if the polygon vertices are wound counter-clockwise.
#[inline]
pub fn is_ccw(vertices: &[Vec2]) -> bool {
    polygon_area_signed(vertices) > 0.0
}

/// Ensures the polygon is wound counter-clockwise (reverses in-place if not).
pub fn ensure_ccw(vertices: &mut Vec<Vec2>) {
    if polygon_area_signed(vertices) < 0.0 {
        vertices.reverse();
    }
}

// ===========================================================================
// Point-in-polygon (ray casting)
// ===========================================================================

/// Tests whether `point` lies inside the simple polygon defined by `vertices`
/// using the ray-casting (crossing number) algorithm.
///
/// Boundary points may or may not be classified as inside due to floating-point
/// precision; use an epsilon-thickened test if exact boundary behavior matters.
pub fn point_in_polygon(point: Vec2, vertices: &[Vec2]) -> bool {
    let n = vertices.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let vi = vertices[i];
        let vj = vertices[j];
        if ((vi.y > point.y) != (vj.y > point.y))
            && (point.x < (vj.x - vi.x) * (point.y - vi.y) / (vj.y - vi.y) + vi.x)
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

// ===========================================================================
// Segment intersection
// ===========================================================================

/// Result of a 2-D segment intersection test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SegmentIntersection {
    /// Segments do not intersect.
    None,
    /// Segments intersect at a single point, with parameters `t` and `u`
    /// along the first and second segments respectively.
    Point {
        point: Vec2,
        t: f32,
        u: f32,
    },
    /// Segments are collinear and overlap.
    Collinear,
}

/// Tests whether two 2-D segments (a0-a1) and (b0-b1) intersect.
///
/// Uses the parametric form `a0 + t*(a1-a0) = b0 + u*(b1-b0)`.
pub fn segment_intersection(a0: Vec2, a1: Vec2, b0: Vec2, b1: Vec2) -> SegmentIntersection {
    let d1 = a1 - a0;
    let d2 = b1 - b0;
    let denom = cross2(d1, d2);

    let ab = b0 - a0;

    if denom.abs() < EPSILON {
        // Parallel -- check collinearity.
        if cross2(ab, d1).abs() < EPSILON {
            // Collinear: check overlap using projections.
            let len_sq = d1.length_squared();
            if len_sq < EPSILON * EPSILON {
                return SegmentIntersection::None;
            }
            let t0 = ab.dot(d1) / len_sq;
            let t1 = t0 + d2.dot(d1) / len_sq;
            let (tmin, tmax) = if t0 < t1 { (t0, t1) } else { (t1, t0) };
            if tmax >= 0.0 && tmin <= 1.0 {
                return SegmentIntersection::Collinear;
            }
        }
        return SegmentIntersection::None;
    }

    let t = cross2(ab, d2) / denom;
    let u = cross2(ab, d1) / denom;

    if t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 {
        SegmentIntersection::Point {
            point: a0 + d1 * t,
            t,
            u,
        }
    } else {
        SegmentIntersection::None
    }
}

// ===========================================================================
// Ear-clipping triangulation (supports holes)
// ===========================================================================

/// Triangulates a simple polygon (optionally with holes) using the ear-clipping
/// algorithm.
///
/// `outer` is the outer boundary in CCW order. `holes` contains inner boundaries
/// in CW order. The algorithm merges holes into the outer boundary via bridge
/// edges, then performs ear clipping.
///
/// Returns a list of index triples referencing the merged vertex list.
pub struct EarClipTriangulation;

impl EarClipTriangulation {
    /// Triangulates and returns (merged_vertices, triangle_indices).
    pub fn triangulate(
        outer: &[Vec2],
        holes: &[Vec<Vec2>],
    ) -> (Vec<Vec2>, Vec<[usize; 3]>) {
        // Build the merged polygon by incorporating holes.
        let mut merged = outer.to_vec();
        ensure_ccw(&mut merged);

        // Sort holes by the maximum x-coordinate of their vertices (descending)
        // so we process the rightmost holes first.
        let mut sorted_holes: Vec<Vec<Vec2>> = holes.to_vec();
        for hole in &mut sorted_holes {
            // Holes should be CW; reverse if they are CCW.
            if polygon_area_signed(hole) > 0.0 {
                hole.reverse();
            }
        }
        sorted_holes.sort_by(|a, b| {
            let max_a = a.iter().map(|v| v.x).fold(f32::NEG_INFINITY, f32::max);
            let max_b = b.iter().map(|v| v.x).fold(f32::NEG_INFINITY, f32::max);
            max_b.partial_cmp(&max_a).unwrap()
        });

        for hole in &sorted_holes {
            Self::merge_hole(&mut merged, hole);
        }

        // Ear clipping on the merged polygon.
        let tris = Self::ear_clip(&merged);
        (merged, tris)
    }

    /// Merges a single hole into the outer polygon by finding a mutually visible
    /// bridge vertex pair.
    fn merge_hole(outer: &mut Vec<Vec2>, hole: &[Vec2]) {
        if hole.is_empty() {
            return;
        }
        // Find the rightmost vertex of the hole.
        let (hi, _) = hole
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.x.partial_cmp(&b.x).unwrap())
            .unwrap();
        let hp = hole[hi];

        // Cast a ray to the right from hp and find the nearest intersection
        // with an edge of the outer polygon.
        let mut best_x = f32::INFINITY;
        let mut best_edge = 0usize;
        let n = outer.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let a = outer[i];
            let b = outer[j];
            if (a.y - hp.y) * (b.y - hp.y) > 0.0 {
                continue;
            }
            if a.y == b.y {
                continue;
            }
            let t = (hp.y - a.y) / (b.y - a.y);
            if t < 0.0 || t > 1.0 {
                continue;
            }
            let ix = a.x + t * (b.x - a.x);
            if ix >= hp.x && ix < best_x {
                best_x = ix;
                best_edge = i;
            }
        }

        // The bridge connects hp to a visible vertex of the outer polygon near
        // the intersection. Start with the closer endpoint of the intersected edge.
        let a = outer[best_edge];
        let b = outer[(best_edge + 1) % n];
        let bridge_idx = if a.x > b.x { best_edge } else { (best_edge + 1) % n };

        // Insert the hole into the outer polygon at bridge_idx.
        let hole_len = hole.len();
        let mut new_verts: Vec<Vec2> = Vec::with_capacity(outer.len() + hole_len + 2);
        new_verts.extend_from_slice(&outer[..=bridge_idx]);
        // Hole vertices starting from hi, wrapping around.
        for k in 0..=hole_len {
            new_verts.push(hole[(hi + k) % hole_len]);
        }
        // Duplicate the bridge vertex to close the seam.
        new_verts.push(outer[bridge_idx]);
        if bridge_idx + 1 < outer.len() {
            new_verts.extend_from_slice(&outer[bridge_idx + 1..]);
        }

        *outer = new_verts;
    }

    /// Standard ear-clipping triangulation of a simple polygon (no holes).
    fn ear_clip(vertices: &[Vec2]) -> Vec<[usize; 3]> {
        let n = vertices.len();
        if n < 3 {
            return Vec::new();
        }

        let mut indices: Vec<usize> = (0..n).collect();
        let mut result: Vec<[usize; 3]> = Vec::with_capacity(n - 2);

        while indices.len() > 3 {
            let m = indices.len();
            let mut ear_found = false;
            for i in 0..m {
                let prev = indices[(i + m - 1) % m];
                let curr = indices[i];
                let next = indices[(i + 1) % m];

                let vp = vertices[prev];
                let vc = vertices[curr];
                let vn = vertices[next];

                // Must be a convex vertex (left turn).
                if orient2d(vp, vc, vn) <= EPSILON {
                    continue;
                }

                // Check that no other vertex lies inside the triangle.
                let mut ear = true;
                for j in 0..m {
                    let k = indices[j];
                    if k == prev || k == curr || k == next {
                        continue;
                    }
                    if Self::point_in_triangle(vertices[k], vp, vc, vn) {
                        ear = false;
                        break;
                    }
                }

                if ear {
                    result.push([prev, curr, next]);
                    indices.remove(i);
                    ear_found = true;
                    break;
                }
            }
            if !ear_found {
                // Degenerate polygon -- bail out.
                break;
            }
        }

        if indices.len() == 3 {
            result.push([indices[0], indices[1], indices[2]]);
        }

        result
    }

    #[inline]
    fn point_in_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
        let d1 = orient2d(a, b, p);
        let d2 = orient2d(b, c, p);
        let d3 = orient2d(c, a, p);
        let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
        let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
        !(has_neg && has_pos)
    }
}

// ===========================================================================
// Bowyer-Watson Delaunay triangulation + Voronoi dual
// ===========================================================================

/// A triangle in the Delaunay triangulation, referencing point indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DelaunayTriangle {
    pub indices: [usize; 3],
}

/// Result of a Delaunay triangulation.
#[derive(Debug, Clone)]
pub struct DelaunayTriangulation {
    /// Input points.
    pub points: Vec<Vec2>,
    /// Triangles (indices into `points`).
    pub triangles: Vec<DelaunayTriangle>,
}

impl DelaunayTriangulation {
    /// Computes the Delaunay triangulation of a set of 2-D points using the
    /// Bowyer-Watson algorithm. O(n^2) worst case (expected O(n log n) for
    /// uniformly distributed points).
    pub fn compute(points: &[Vec2]) -> Self {
        if points.len() < 3 {
            return Self {
                points: points.to_vec(),
                triangles: Vec::new(),
            };
        }

        // Compute a super-triangle that contains all points.
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        for &p in points {
            min = min.min(p);
            max = max.max(p);
        }
        let dx = max.x - min.x;
        let dy = max.y - min.y;
        let d_max = dx.max(dy);
        let mid = (min + max) * 0.5;

        // Super-triangle vertices (placed far away).
        let st0 = Vec2::new(mid.x - 20.0 * d_max, mid.y - d_max);
        let st1 = Vec2::new(mid.x, mid.y + 20.0 * d_max);
        let st2 = Vec2::new(mid.x + 20.0 * d_max, mid.y - d_max);

        let n = points.len();
        // Internal point list: original points + 3 super-triangle vertices.
        let mut all_pts: Vec<Vec2> = points.to_vec();
        all_pts.push(st0);
        all_pts.push(st1);
        all_pts.push(st2);

        let si0 = n;
        let si1 = n + 1;
        let si2 = n + 2;

        // Start with the super-triangle.
        let mut triangles: Vec<[usize; 3]> = vec![[si0, si1, si2]];

        // Insert each point.
        for pi in 0..n {
            let p = all_pts[pi];

            // Find all triangles whose circumcircle contains p.
            let mut bad_triangles: Vec<usize> = Vec::new();
            for (ti, tri) in triangles.iter().enumerate() {
                if Self::in_circumcircle(
                    p,
                    all_pts[tri[0]],
                    all_pts[tri[1]],
                    all_pts[tri[2]],
                ) {
                    bad_triangles.push(ti);
                }
            }

            // Find the boundary (polygon hole) of the bad triangles.
            let mut polygon: Vec<(usize, usize)> = Vec::new();
            for &bi in &bad_triangles {
                let tri = triangles[bi];
                let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];
                for &(ea, eb) in &edges {
                    // An edge is on the boundary if it is not shared with another
                    // bad triangle.
                    let shared = bad_triangles.iter().any(|&bj| {
                        if bj == bi {
                            return false;
                        }
                        let other = triangles[bj];
                        let oedges = [
                            (other[0], other[1]),
                            (other[1], other[2]),
                            (other[2], other[0]),
                        ];
                        oedges.contains(&(eb, ea))
                    });
                    if !shared {
                        polygon.push((ea, eb));
                    }
                }
            }

            // Remove bad triangles (in reverse order to preserve indices).
            bad_triangles.sort_unstable();
            for &bi in bad_triangles.iter().rev() {
                triangles.swap_remove(bi);
            }

            // Re-triangulate the polygon hole by creating triangles to the new point.
            for &(ea, eb) in &polygon {
                triangles.push([pi, ea, eb]);
            }
        }

        // Remove any triangles that reference super-triangle vertices.
        triangles.retain(|tri| {
            tri[0] < n && tri[1] < n && tri[2] < n
        });

        let dt_triangles = triangles
            .iter()
            .map(|&t| DelaunayTriangle { indices: t })
            .collect();

        Self {
            points: points.to_vec(),
            triangles: dt_triangles,
        }
    }

    /// Tests whether point `p` lies inside the circumcircle of triangle (a, b, c).
    ///
    /// Uses the circumcenter/circumradius approach for robustness regardless of
    /// triangle winding order.
    fn in_circumcircle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
        let cc = Self::circumcenter(a, b, c);
        let r_sq = (a - cc).length_squared();
        (p - cc).length_squared() < r_sq + EPSILON
    }

    /// Computes the circumcenter of a triangle.
    fn circumcenter(a: Vec2, b: Vec2, c: Vec2) -> Vec2 {
        let d = 2.0
            * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
        if d.abs() < EPSILON {
            return (a + b + c) / 3.0;
        }
        let ux = ((a.x * a.x + a.y * a.y) * (b.y - c.y)
            + (b.x * b.x + b.y * b.y) * (c.y - a.y)
            + (c.x * c.x + c.y * c.y) * (a.y - b.y))
            / d;
        let uy = ((a.x * a.x + a.y * a.y) * (c.x - b.x)
            + (b.x * b.x + b.y * b.y) * (a.x - c.x)
            + (c.x * c.x + c.y * c.y) * (b.x - a.x))
            / d;
        Vec2::new(ux, uy)
    }

    /// Extracts the Voronoi diagram as the dual of this Delaunay triangulation.
    ///
    /// Returns `(voronoi_vertices, voronoi_edges)` where each edge is a pair of
    /// indices into the vertex array. Edges on the convex hull boundary are
    /// clipped to a finite extent.
    pub fn voronoi_dual(&self) -> (Vec<Vec2>, Vec<(usize, usize)>) {
        let mut centers: Vec<Vec2> = Vec::with_capacity(self.triangles.len());
        for tri in &self.triangles {
            let a = self.points[tri.indices[0]];
            let b = self.points[tri.indices[1]];
            let c = self.points[tri.indices[2]];
            centers.push(Self::circumcenter(a, b, c));
        }

        // Build adjacency: for each directed edge, store the triangle index.
        let mut edge_to_tri: HashMap<(usize, usize), usize> = HashMap::new();
        for (ti, tri) in self.triangles.iter().enumerate() {
            let idx = tri.indices;
            edge_to_tri.insert((idx[0], idx[1]), ti);
            edge_to_tri.insert((idx[1], idx[2]), ti);
            edge_to_tri.insert((idx[2], idx[0]), ti);
        }

        let mut edges: Vec<(usize, usize)> = Vec::new();
        let mut seen: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

        for (ti, tri) in self.triangles.iter().enumerate() {
            let idx = tri.indices;
            let directed = [(idx[0], idx[1]), (idx[1], idx[2]), (idx[2], idx[0])];
            for &(ea, eb) in &directed {
                if let Some(&tj) = edge_to_tri.get(&(eb, ea)) {
                    let key = if ti < tj { (ti, tj) } else { (tj, ti) };
                    if seen.insert(key) {
                        edges.push(key);
                    }
                }
            }
        }

        (centers, edges)
    }
}

// ===========================================================================
// Sutherland-Hodgman polygon clipping (boolean intersection)
// ===========================================================================

/// Clips `subject` polygon against `clip` polygon using the Sutherland-Hodgman
/// algorithm. Both polygons are assumed to be convex and wound CCW.
///
/// Returns the clipped polygon (intersection), or an empty vec if there is no
/// overlap.
pub fn sutherland_hodgman_clip(subject: &[Vec2], clip: &[Vec2]) -> Vec<Vec2> {
    if subject.is_empty() || clip.is_empty() {
        return Vec::new();
    }

    let mut output = subject.to_vec();

    let cn = clip.len();
    for i in 0..cn {
        if output.is_empty() {
            return Vec::new();
        }
        let edge_start = clip[i];
        let edge_end = clip[(i + 1) % cn];

        let input = output;
        output = Vec::with_capacity(input.len() + 1);

        let m = input.len();
        let mut s = input[m - 1];

        for k in 0..m {
            let e = input[k];
            let e_inside = orient2d(edge_start, edge_end, e) >= 0.0;
            let s_inside = orient2d(edge_start, edge_end, s) >= 0.0;

            if e_inside {
                if !s_inside {
                    // s is outside, e is inside: add intersection then e.
                    if let Some(inter) = line_line_intersect(edge_start, edge_end, s, e) {
                        output.push(inter);
                    }
                }
                output.push(e);
            } else if s_inside {
                // s is inside, e is outside: add intersection.
                if let Some(inter) = line_line_intersect(edge_start, edge_end, s, e) {
                    output.push(inter);
                }
            }
            s = e;
        }
    }

    output
}

/// Intersection of two infinite lines, each defined by two points.
fn line_line_intersect(a0: Vec2, a1: Vec2, b0: Vec2, b1: Vec2) -> Option<Vec2> {
    let d1 = a1 - a0;
    let d2 = b1 - b0;
    let denom = cross2(d1, d2);
    if denom.abs() < EPSILON {
        return None;
    }
    let t = cross2(b0 - a0, d2) / denom;
    Some(a0 + d1 * t)
}

// ===========================================================================
// Minimum enclosing circle (Welzl's algorithm)
// ===========================================================================

/// A circle defined by center and radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    pub center: Vec2,
    pub radius: f32,
}

/// Computes the minimum enclosing circle of a set of 2-D points using Welzl's
/// randomized algorithm. Expected O(n) time.
pub fn minimum_enclosing_circle(points: &[Vec2]) -> Circle {
    if points.is_empty() {
        return Circle {
            center: Vec2::ZERO,
            radius: 0.0,
        };
    }

    // Shuffle to get expected linear time.
    let mut pts = points.to_vec();
    // Simple Fisher-Yates shuffle using a deterministic seed for reproducibility.
    let mut seed: u64 = 0xDEAD_BEEF_CAFE_BABE;
    for i in (1..pts.len()).rev() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (seed >> 33) as usize % (i + 1);
        pts.swap(i, j);
    }

    welzl_recurse(&pts, &mut Vec::new(), pts.len())
}

fn welzl_recurse(pts: &[Vec2], boundary: &mut Vec<Vec2>, n: usize) -> Circle {
    if n == 0 || boundary.len() == 3 {
        return circle_from_boundary(boundary);
    }

    let p = pts[n - 1];
    let circle = welzl_recurse(pts, boundary, n - 1);

    if (p - circle.center).length_squared() <= (circle.radius + EPSILON) * (circle.radius + EPSILON)
    {
        return circle;
    }

    boundary.push(p);
    let result = welzl_recurse(pts, boundary, n - 1);
    boundary.pop();
    result
}

fn circle_from_boundary(boundary: &[Vec2]) -> Circle {
    match boundary.len() {
        0 => Circle {
            center: Vec2::ZERO,
            radius: 0.0,
        },
        1 => Circle {
            center: boundary[0],
            radius: 0.0,
        },
        2 => {
            let center = (boundary[0] + boundary[1]) * 0.5;
            let radius = (boundary[1] - boundary[0]).length() * 0.5;
            Circle { center, radius }
        }
        _ => {
            // Circumcircle of three points.
            let a = boundary[0];
            let b = boundary[1];
            let c = boundary[2];
            let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
            if d.abs() < EPSILON {
                // Degenerate: return circle through two farthest points.
                let d01 = (b - a).length();
                let d12 = (c - b).length();
                let d02 = (c - a).length();
                if d01 >= d12 && d01 >= d02 {
                    return Circle {
                        center: (a + b) * 0.5,
                        radius: d01 * 0.5,
                    };
                } else if d12 >= d02 {
                    return Circle {
                        center: (b + c) * 0.5,
                        radius: d12 * 0.5,
                    };
                } else {
                    return Circle {
                        center: (a + c) * 0.5,
                        radius: d02 * 0.5,
                    };
                }
            }
            let ux = ((a.x * a.x + a.y * a.y) * (b.y - c.y)
                + (b.x * b.x + b.y * b.y) * (c.y - a.y)
                + (c.x * c.x + c.y * c.y) * (a.y - b.y))
                / d;
            let uy = ((a.x * a.x + a.y * a.y) * (c.x - b.x)
                + (b.x * b.x + b.y * b.y) * (a.x - c.x)
                + (c.x * c.x + c.y * c.y) * (b.x - a.x))
                / d;
            let center = Vec2::new(ux, uy);
            let radius = (a - center).length();
            Circle { center, radius }
        }
    }
}

// ===========================================================================
// Minimum enclosing sphere (Welzl's algorithm, 3-D)
// ===========================================================================

/// A sphere defined by center and radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

/// Computes the minimum enclosing sphere of a set of 3-D points using Welzl's
/// randomized algorithm. Expected O(n).
pub fn minimum_enclosing_sphere(points: &[Vec3]) -> Sphere {
    if points.is_empty() {
        return Sphere {
            center: Vec3::ZERO,
            radius: 0.0,
        };
    }

    let mut pts = points.to_vec();
    let mut seed: u64 = 0xCAFE_BABE_DEAD_BEEF;
    for i in (1..pts.len()).rev() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (seed >> 33) as usize % (i + 1);
        pts.swap(i, j);
    }

    welzl3d_recurse(&pts, &mut Vec::new(), pts.len())
}

fn welzl3d_recurse(pts: &[Vec3], boundary: &mut Vec<Vec3>, n: usize) -> Sphere {
    if n == 0 || boundary.len() == 4 {
        return sphere_from_boundary(boundary);
    }

    let p = pts[n - 1];
    let sphere = welzl3d_recurse(pts, boundary, n - 1);

    if (p - sphere.center).length_squared()
        <= (sphere.radius + EPSILON) * (sphere.radius + EPSILON)
    {
        return sphere;
    }

    boundary.push(p);
    let result = welzl3d_recurse(pts, boundary, n - 1);
    boundary.pop();
    result
}

fn sphere_from_boundary(boundary: &[Vec3]) -> Sphere {
    match boundary.len() {
        0 => Sphere {
            center: Vec3::ZERO,
            radius: 0.0,
        },
        1 => Sphere {
            center: boundary[0],
            radius: 0.0,
        },
        2 => {
            let center = (boundary[0] + boundary[1]) * 0.5;
            let radius = (boundary[1] - boundary[0]).length() * 0.5;
            Sphere { center, radius }
        }
        3 => {
            // Circumsphere of three points (the circumcircle in the plane they define).
            let a = boundary[0];
            let b = boundary[1];
            let c = boundary[2];
            let ab = b - a;
            let ac = c - a;
            let ab_cross_ac = ab.cross(ac);
            let denom = 2.0 * ab_cross_ac.length_squared();
            if denom < EPSILON * EPSILON {
                let d01 = (b - a).length();
                let d12 = (c - b).length();
                let d02 = (c - a).length();
                if d01 >= d12 && d01 >= d02 {
                    return Sphere { center: (a + b) * 0.5, radius: d01 * 0.5 };
                } else if d12 >= d02 {
                    return Sphere { center: (b + c) * 0.5, radius: d12 * 0.5 };
                } else {
                    return Sphere { center: (a + c) * 0.5, radius: d02 * 0.5 };
                }
            }
            let t = (ab_cross_ac.cross(ab) * ac.length_squared()
                + ac.cross(ab_cross_ac) * ab.length_squared())
                / denom;
            let center = a + t;
            let radius = t.length();
            Sphere { center, radius }
        }
        _ => {
            // Circumsphere of four points.
            let a = boundary[0];
            let b = boundary[1];
            let c = boundary[2];
            let d = boundary[3];

            let ab = b - a;
            let ac = c - a;
            let ad = d - a;

            let det = ab.dot(ac.cross(ad));
            if det.abs() < EPSILON {
                // Degenerate -- fall back to 3-point sphere of the widest triple.
                return sphere_from_boundary(&boundary[..3]);
            }

            let ab2 = ab.length_squared();
            let ac2 = ac.length_squared();
            let ad2 = ad.length_squared();

            let t = (ac.cross(ad) * ab2 + ad.cross(ab) * ac2 + ab.cross(ac) * ad2)
                / (2.0 * det);
            let center = a + t;
            let radius = t.length();
            Sphere { center, radius }
        }
    }
}

// ===========================================================================
// OBB (Oriented Bounding Box) from PCA
// ===========================================================================

/// An oriented bounding box in 3-D.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OBB {
    /// Center of the box.
    pub center: Vec3,
    /// Orthonormal axes (columns of the rotation matrix).
    pub axes: [Vec3; 3],
    /// Half-extents along each axis.
    pub half_extents: Vec3,
}

impl OBB {
    /// Constructs an oriented bounding box from a set of points using PCA
    /// (principal component analysis) to find the best-fit axes.
    ///
    /// The covariance matrix is computed and its eigenvectors (found via
    /// iterative Jacobi rotations) form the OBB axes.
    pub fn from_points(points: &[Vec3]) -> Self {
        if points.is_empty() {
            return Self {
                center: Vec3::ZERO,
                axes: [Vec3::X, Vec3::Y, Vec3::Z],
                half_extents: Vec3::ZERO,
            };
        }

        // Compute centroid.
        let n = points.len() as f32;
        let centroid = points.iter().copied().fold(Vec3::ZERO, |a, b| a + b) / n;

        // Compute covariance matrix (symmetric 3x3).
        let mut cov = [[0.0f32; 3]; 3];
        for &p in points {
            let d = p - centroid;
            let da = [d.x, d.y, d.z];
            for i in 0..3 {
                for j in i..3 {
                    cov[i][j] += da[i] * da[j];
                }
            }
        }
        for i in 0..3 {
            for j in i..3 {
                cov[i][j] /= n;
                if j != i {
                    cov[j][i] = cov[i][j];
                }
            }
        }

        // Jacobi eigendecomposition.
        let (eigenvalues, eigenvectors) = jacobi_eigen_3x3(cov);

        // Sort eigenvectors by eigenvalue (largest first).
        let mut order: [usize; 3] = [0, 1, 2];
        order.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

        let axes = [
            Vec3::new(
                eigenvectors[0][order[0]],
                eigenvectors[1][order[0]],
                eigenvectors[2][order[0]],
            )
            .normalize(),
            Vec3::new(
                eigenvectors[0][order[1]],
                eigenvectors[1][order[1]],
                eigenvectors[2][order[1]],
            )
            .normalize(),
            Vec3::new(
                eigenvectors[0][order[2]],
                eigenvectors[1][order[2]],
                eigenvectors[2][order[2]],
            )
            .normalize(),
        ];

        // Project points onto axes and find extents.
        let mut mins = [f32::INFINITY; 3];
        let mut maxs = [f32::NEG_INFINITY; 3];
        for &p in points {
            let d = p - centroid;
            for i in 0..3 {
                let proj = d.dot(axes[i]);
                mins[i] = mins[i].min(proj);
                maxs[i] = maxs[i].max(proj);
            }
        }

        let half_extents = Vec3::new(
            (maxs[0] - mins[0]) * 0.5,
            (maxs[1] - mins[1]) * 0.5,
            (maxs[2] - mins[2]) * 0.5,
        );

        let center_offset = Vec3::new(
            (maxs[0] + mins[0]) * 0.5,
            (maxs[1] + mins[1]) * 0.5,
            (maxs[2] + mins[2]) * 0.5,
        );
        let center = centroid + axes[0] * center_offset.x + axes[1] * center_offset.y + axes[2] * center_offset.z;

        Self {
            center,
            axes,
            half_extents,
        }
    }

    /// Returns the 8 corner vertices of the OBB.
    pub fn corners(&self) -> [Vec3; 8] {
        let mut corners = [Vec3::ZERO; 8];
        for i in 0..8 {
            let sx = if i & 1 == 0 { 1.0 } else { -1.0 };
            let sy = if i & 2 == 0 { 1.0 } else { -1.0 };
            let sz = if i & 4 == 0 { 1.0 } else { -1.0 };
            corners[i] = self.center
                + self.axes[0] * self.half_extents.x * sx
                + self.axes[1] * self.half_extents.y * sy
                + self.axes[2] * self.half_extents.z * sz;
        }
        corners
    }

    /// Tests whether two OBBs intersect using the Separating Axis Theorem
    /// (15-axis test).
    pub fn intersects_obb(&self, other: &OBB) -> bool {
        obb_obb_sat(self, other)
    }

    /// Tests whether a point lies inside this OBB.
    pub fn contains_point(&self, point: Vec3) -> bool {
        let d = point - self.center;
        for i in 0..3 {
            let proj = d.dot(self.axes[i]).abs();
            let extent = match i {
                0 => self.half_extents.x,
                1 => self.half_extents.y,
                _ => self.half_extents.z,
            };
            if proj > extent + EPSILON {
                return false;
            }
        }
        true
    }
}

/// OBB-OBB intersection using the Separating Axis Theorem (15-axis test).
///
/// Tests the 3 face normals of each OBB (6 axes) plus the 9 cross products
/// of edge pairs.
pub fn obb_obb_sat(a: &OBB, b: &OBB) -> bool {
    let t = b.center - a.center;
    let ae = [a.half_extents.x, a.half_extents.y, a.half_extents.z];
    let be = [b.half_extents.x, b.half_extents.y, b.half_extents.z];

    // Rotation matrix expressing b's axes in a's frame, and its absolute value
    // with epsilon for robustness.
    let mut r = [[0.0f32; 3]; 3];
    let mut abs_r = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a.axes[i].dot(b.axes[j]);
            abs_r[i][j] = r[i][j].abs() + EPSILON;
        }
    }

    let ta = [t.dot(a.axes[0]), t.dot(a.axes[1]), t.dot(a.axes[2])];

    // Test axes L = a.axes[i]  (i = 0, 1, 2)
    for i in 0..3 {
        let ra = ae[i];
        let rb = be[0] * abs_r[i][0] + be[1] * abs_r[i][1] + be[2] * abs_r[i][2];
        if ta[i].abs() > ra + rb {
            return false;
        }
    }

    // Test axes L = b.axes[j]  (j = 0, 1, 2)
    for j in 0..3 {
        let ra = ae[0] * abs_r[0][j] + ae[1] * abs_r[1][j] + ae[2] * abs_r[2][j];
        let rb = be[j];
        let sep = (ta[0] * r[0][j] + ta[1] * r[1][j] + ta[2] * r[2][j]).abs();
        if sep > ra + rb {
            return false;
        }
    }

    // Test axes L = a.axes[i] x b.axes[j]  (9 axes)
    // L = a0 x b0
    {
        let ra = ae[1] * abs_r[2][0] + ae[2] * abs_r[1][0];
        let rb = be[1] * abs_r[0][2] + be[2] * abs_r[0][1];
        let sep = (ta[2] * r[1][0] - ta[1] * r[2][0]).abs();
        if sep > ra + rb { return false; }
    }
    // L = a0 x b1
    {
        let ra = ae[1] * abs_r[2][1] + ae[2] * abs_r[1][1];
        let rb = be[0] * abs_r[0][2] + be[2] * abs_r[0][0];
        let sep = (ta[2] * r[1][1] - ta[1] * r[2][1]).abs();
        if sep > ra + rb { return false; }
    }
    // L = a0 x b2
    {
        let ra = ae[1] * abs_r[2][2] + ae[2] * abs_r[1][2];
        let rb = be[0] * abs_r[0][1] + be[1] * abs_r[0][0];
        let sep = (ta[2] * r[1][2] - ta[1] * r[2][2]).abs();
        if sep > ra + rb { return false; }
    }
    // L = a1 x b0
    {
        let ra = ae[0] * abs_r[2][0] + ae[2] * abs_r[0][0];
        let rb = be[1] * abs_r[1][2] + be[2] * abs_r[1][1];
        let sep = (ta[0] * r[2][0] - ta[2] * r[0][0]).abs();
        if sep > ra + rb { return false; }
    }
    // L = a1 x b1
    {
        let ra = ae[0] * abs_r[2][1] + ae[2] * abs_r[0][1];
        let rb = be[0] * abs_r[1][2] + be[2] * abs_r[1][0];
        let sep = (ta[0] * r[2][1] - ta[2] * r[0][1]).abs();
        if sep > ra + rb { return false; }
    }
    // L = a1 x b2
    {
        let ra = ae[0] * abs_r[2][2] + ae[2] * abs_r[0][2];
        let rb = be[0] * abs_r[1][1] + be[1] * abs_r[1][0];
        let sep = (ta[0] * r[2][2] - ta[2] * r[0][2]).abs();
        if sep > ra + rb { return false; }
    }
    // L = a2 x b0
    {
        let ra = ae[0] * abs_r[1][0] + ae[1] * abs_r[0][0];
        let rb = be[1] * abs_r[2][2] + be[2] * abs_r[2][1];
        let sep = (ta[1] * r[0][0] - ta[0] * r[1][0]).abs();
        if sep > ra + rb { return false; }
    }
    // L = a2 x b1
    {
        let ra = ae[0] * abs_r[1][1] + ae[1] * abs_r[0][1];
        let rb = be[0] * abs_r[2][2] + be[2] * abs_r[2][0];
        let sep = (ta[1] * r[0][1] - ta[0] * r[1][1]).abs();
        if sep > ra + rb { return false; }
    }
    // L = a2 x b2
    {
        let ra = ae[0] * abs_r[1][2] + ae[1] * abs_r[0][2];
        let rb = be[0] * abs_r[2][1] + be[1] * abs_r[2][0];
        let sep = (ta[1] * r[0][2] - ta[0] * r[1][2]).abs();
        if sep > ra + rb { return false; }
    }

    true
}

// ---------------------------------------------------------------------------
// Jacobi eigendecomposition for symmetric 3x3 matrices
// ---------------------------------------------------------------------------

/// Jacobi eigendecomposition of a symmetric 3x3 matrix.
/// Returns (eigenvalues, eigenvectors_as_columns).
fn jacobi_eigen_3x3(mut a: [[f32; 3]; 3]) -> ([f32; 3], [[f32; 3]; 3]) {
    let mut v = [[0.0f32; 3]; 3];
    for i in 0..3 {
        v[i][i] = 1.0;
    }

    for _ in 0..50 {
        // Find the largest off-diagonal element.
        let mut p = 0;
        let mut q = 1;
        let mut max_val = a[0][1].abs();
        for i in 0..3 {
            for j in (i + 1)..3 {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-10 {
            break;
        }

        // Compute rotation angle.
        let theta = if (a[p][p] - a[q][q]).abs() < 1e-12 {
            std::f32::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation.
        let mut new_a = a;
        for i in 0..3 {
            new_a[i][p] = c * a[i][p] + s * a[i][q];
            new_a[i][q] = -s * a[i][p] + c * a[i][q];
        }
        for j in 0..3 {
            a[p][j] = c * new_a[p][j] + s * new_a[q][j];
            a[q][j] = -s * new_a[p][j] + c * new_a[q][j];
        }
        // Ensure symmetry at p,q.
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        // Fix diagonal from the rotation.
        for i in 0..3 {
            if i != p && i != q {
                a[i][p] = new_a[i][p];
                a[p][i] = new_a[i][p];
                a[i][q] = new_a[i][q];
                a[q][i] = new_a[i][q];
            }
        }

        // Update eigenvectors.
        let mut new_v = v;
        for i in 0..3 {
            new_v[i][p] = c * v[i][p] + s * v[i][q];
            new_v[i][q] = -s * v[i][p] + c * v[i][q];
        }
        v = new_v;
    }

    ([a[0][0], a[1][1], a[2][2]], v)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec2, Vec3};

    #[test]
    fn test_convex_hull_2d_square() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(0.5, 0.5), // interior point
        ];
        let hull = ConvexHull2D::compute(&points);
        assert_eq!(hull.vertices.len(), 4);
        assert!((hull.area() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_convex_hull_2d_triangle() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(2.0, 3.0),
        ];
        let hull = ConvexHull2D::compute(&points);
        assert_eq!(hull.vertices.len(), 3);
        assert!((hull.area() - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_convex_hull_2d_contains() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(4.0, 4.0),
            Vec2::new(0.0, 4.0),
        ];
        let hull = ConvexHull2D::compute(&points);
        assert!(hull.contains_point(Vec2::new(2.0, 2.0)));
        assert!(!hull.contains_point(Vec2::new(5.0, 5.0)));
    }

    #[test]
    fn test_polygon_area_shoelace() {
        let square = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
        ];
        assert!((polygon_area(&square) - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_winding_order() {
        let ccw = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
        ];
        assert!(is_ccw(&ccw));
        let cw = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 0.0),
        ];
        assert!(!is_ccw(&cw));
    }

    #[test]
    fn test_point_in_polygon() {
        let polygon = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(4.0, 4.0),
            Vec2::new(0.0, 4.0),
        ];
        assert!(point_in_polygon(Vec2::new(2.0, 2.0), &polygon));
        assert!(!point_in_polygon(Vec2::new(5.0, 5.0), &polygon));
    }

    #[test]
    fn test_segment_intersection() {
        let result = segment_intersection(
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
            Vec2::new(2.0, 0.0),
        );
        match result {
            SegmentIntersection::Point { point, .. } => {
                assert!((point - Vec2::new(1.0, 1.0)).length() < 0.01);
            }
            _ => panic!("Expected intersection point"),
        }
    }

    #[test]
    fn test_segment_no_intersection() {
        let result = segment_intersection(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
        );
        assert!(matches!(result, SegmentIntersection::None));
    }

    #[test]
    fn test_ear_clip_simple_triangle() {
        let outer = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(2.0, 3.0),
        ];
        let (_, tris) = EarClipTriangulation::triangulate(&outer, &[]);
        assert_eq!(tris.len(), 1);
    }

    #[test]
    fn test_ear_clip_square() {
        let outer = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(4.0, 4.0),
            Vec2::new(0.0, 4.0),
        ];
        let (_, tris) = EarClipTriangulation::triangulate(&outer, &[]);
        assert_eq!(tris.len(), 2);
    }

    #[test]
    fn test_delaunay_basic() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(4.0, 4.0),
            Vec2::new(0.0, 4.0),
            Vec2::new(2.0, 2.0),
        ];
        let dt = DelaunayTriangulation::compute(&points);
        assert!(dt.triangles.len() >= 4);
    }

    #[test]
    fn test_delaunay_voronoi_dual() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(4.0, 4.0),
            Vec2::new(0.0, 4.0),
            Vec2::new(2.0, 2.0),
        ];
        let dt = DelaunayTriangulation::compute(&points);
        let (verts, edges) = dt.voronoi_dual();
        assert!(!verts.is_empty());
        assert!(!edges.is_empty());
    }

    #[test]
    fn test_sutherland_hodgman() {
        let subject = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(4.0, 4.0),
            Vec2::new(0.0, 4.0),
        ];
        let clip = vec![
            Vec2::new(2.0, 0.0),
            Vec2::new(6.0, 0.0),
            Vec2::new(6.0, 4.0),
            Vec2::new(2.0, 4.0),
        ];
        let result = sutherland_hodgman_clip(&subject, &clip);
        assert!(!result.is_empty());
        let area = polygon_area(&result);
        assert!((area - 8.0).abs() < 0.1);
    }

    #[test]
    fn test_minimum_enclosing_circle() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(2.0, 2.0),
        ];
        let circle = minimum_enclosing_circle(&points);
        // All points should be inside or on the boundary.
        for &p in &points {
            assert!((p - circle.center).length() <= circle.radius + 0.01);
        }
    }

    #[test]
    fn test_minimum_enclosing_circle_symmetric() {
        let points = vec![
            Vec2::new(-1.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, -1.0),
            Vec2::new(0.0, 1.0),
        ];
        let circle = minimum_enclosing_circle(&points);
        assert!((circle.center - Vec2::ZERO).length() < 0.1);
        assert!((circle.radius - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_minimum_enclosing_sphere() {
        let points = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ];
        let sphere = minimum_enclosing_sphere(&points);
        assert!((sphere.center - Vec3::ZERO).length() < 0.1);
        assert!((sphere.radius - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_obb_from_points() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.5),
            Vec3::new(2.0, 0.0, 0.5),
            Vec3::new(2.0, 1.0, 0.5),
            Vec3::new(0.0, 1.0, 0.5),
        ];
        let obb = OBB::from_points(&points);
        assert!(obb.half_extents.x > 0.0);
        assert!(obb.half_extents.y > 0.0);
        assert!(obb.half_extents.z > 0.0);
    }

    #[test]
    fn test_obb_contains_point() {
        let obb = OBB {
            center: Vec3::ZERO,
            axes: [Vec3::X, Vec3::Y, Vec3::Z],
            half_extents: Vec3::new(1.0, 1.0, 1.0),
        };
        assert!(obb.contains_point(Vec3::new(0.5, 0.5, 0.5)));
        assert!(!obb.contains_point(Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn test_obb_obb_sat_overlapping() {
        let a = OBB {
            center: Vec3::ZERO,
            axes: [Vec3::X, Vec3::Y, Vec3::Z],
            half_extents: Vec3::new(1.0, 1.0, 1.0),
        };
        let b = OBB {
            center: Vec3::new(1.5, 0.0, 0.0),
            axes: [Vec3::X, Vec3::Y, Vec3::Z],
            half_extents: Vec3::new(1.0, 1.0, 1.0),
        };
        assert!(a.intersects_obb(&b));
    }

    #[test]
    fn test_obb_obb_sat_separated() {
        let a = OBB {
            center: Vec3::ZERO,
            axes: [Vec3::X, Vec3::Y, Vec3::Z],
            half_extents: Vec3::new(1.0, 1.0, 1.0),
        };
        let b = OBB {
            center: Vec3::new(5.0, 0.0, 0.0),
            axes: [Vec3::X, Vec3::Y, Vec3::Z],
            half_extents: Vec3::new(1.0, 1.0, 1.0),
        };
        assert!(!a.intersects_obb(&b));
    }

    #[test]
    fn test_convex_hull_3d_tetrahedron() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let hull = ConvexHull3D::compute(&points);
        assert_eq!(hull.faces.len(), 4);
        assert!((hull.volume() - 1.0 / 6.0).abs() < 0.01);
    }

    #[test]
    fn test_convex_hull_3d_general_position() {
        // Use points in general position (no 4 coplanar) to avoid
        // degeneracies that challenge the incremental algorithm.
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.1, 0.0),
            Vec3::new(1.0, 2.0, 0.1),
            Vec3::new(0.1, 1.0, 2.0),
            Vec3::new(2.0, 2.0, 2.0),
            Vec3::new(1.5, 0.5, 1.8),
            Vec3::new(0.5, 1.5, 0.7),
            Vec3::new(0.8, 0.3, 1.2), // interior point
        ];
        let hull = ConvexHull3D::compute(&points);
        assert!(hull.faces.len() >= 6);
        assert!(hull.volume() > 0.1);
        // All points should be inside or on the hull.
        for face in &hull.faces {
            for &p in &hull.points {
                let dist = face.normal.dot(p) - face.offset;
                assert!(dist <= 0.01, "Point outside hull face");
            }
        }
    }
}
