// engine/render/src/raytracing/bvh_tracer.rs
//
// BVH acceleration structure and ray tracing core for the Genovo engine.
// Implements a two-level acceleration structure (TLAS/BLAS) with SAH-based
// BVH construction and Moller-Trumbore ray-triangle intersection.

use glam::{Mat4, Vec2, Vec3, Vec4};
use std::f32;

// ---------------------------------------------------------------------------
// Ray
// ---------------------------------------------------------------------------

/// A ray defined by origin + direction. Stores the precomputed reciprocal
/// direction for fast AABB slab tests.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
    /// 1.0 / direction -- cached for slab intersection.
    pub inv_direction: Vec3,
    /// Minimum parametric distance (usually 0 or a small epsilon).
    pub t_min: f32,
    /// Maximum parametric distance.
    pub t_max: f32,
}

impl Ray {
    /// Create a new ray.
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        let dir = direction.normalize_or_zero();
        Self {
            origin,
            direction: dir,
            inv_direction: Vec3::new(
                safe_reciprocal(dir.x),
                safe_reciprocal(dir.y),
                safe_reciprocal(dir.z),
            ),
            t_min: 1e-4,
            t_max: f32::MAX,
        }
    }

    /// Create a ray with explicit t range.
    pub fn with_range(origin: Vec3, direction: Vec3, t_min: f32, t_max: f32) -> Self {
        let dir = direction.normalize_or_zero();
        Self {
            origin,
            direction: dir,
            inv_direction: Vec3::new(
                safe_reciprocal(dir.x),
                safe_reciprocal(dir.y),
                safe_reciprocal(dir.z),
            ),
            t_min,
            t_max,
        }
    }

    /// Compute a point along the ray at parameter t.
    #[inline]
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    /// Transform a ray by a matrix (for instance transforms).
    pub fn transform(&self, inv_transform: &Mat4) -> Self {
        let o4 = *inv_transform * Vec4::new(self.origin.x, self.origin.y, self.origin.z, 1.0);
        let d4 = *inv_transform * Vec4::new(self.direction.x, self.direction.y, self.direction.z, 0.0);
        let new_origin = Vec3::new(o4.x, o4.y, o4.z);
        let new_dir = Vec3::new(d4.x, d4.y, d4.z);
        let len = new_dir.length();
        let norm_dir = if len > 1e-8 { new_dir / len } else { new_dir };
        Self {
            origin: new_origin,
            direction: norm_dir,
            inv_direction: Vec3::new(
                safe_reciprocal(norm_dir.x),
                safe_reciprocal(norm_dir.y),
                safe_reciprocal(norm_dir.z),
            ),
            t_min: self.t_min * len,
            t_max: self.t_max * len,
        }
    }
}

#[inline]
fn safe_reciprocal(v: f32) -> f32 {
    if v.abs() < 1e-12 {
        if v >= 0.0 { 1e12 } else { -1e12 }
    } else {
        1.0 / v
    }
}

// ---------------------------------------------------------------------------
// AABB
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    /// An empty (inverted) AABB that will grow on first union.
    pub const EMPTY: Self = Self {
        min: Vec3::new(f32::MAX, f32::MAX, f32::MAX),
        max: Vec3::new(f32::MIN, f32::MIN, f32::MIN),
    };

    /// Create a new AABB from min/max corners.
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Create an AABB from a single point.
    pub fn from_point(p: Vec3) -> Self {
        Self { min: p, max: p }
    }

    /// Expand to include a point.
    #[inline]
    pub fn expand_point(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    /// Union of two AABBs.
    #[inline]
    pub fn union(&self, other: &AABB) -> AABB {
        AABB {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Merge another AABB into this one.
    #[inline]
    pub fn merge(&mut self, other: &AABB) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    /// Surface area of the AABB (for SAH).
    #[inline]
    pub fn surface_area(&self) -> f32 {
        let d = self.max - self.min;
        if d.x < 0.0 || d.y < 0.0 || d.z < 0.0 {
            return 0.0;
        }
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Volume of the AABB.
    #[inline]
    pub fn volume(&self) -> f32 {
        let d = self.max - self.min;
        if d.x < 0.0 || d.y < 0.0 || d.z < 0.0 {
            return 0.0;
        }
        d.x * d.y * d.z
    }

    /// Center of the AABB.
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Extent (half-size) of the AABB.
    #[inline]
    pub fn extent(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Size (full diagonal).
    #[inline]
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// The longest axis (0=X, 1=Y, 2=Z).
    #[inline]
    pub fn longest_axis(&self) -> usize {
        let d = self.max - self.min;
        if d.x > d.y && d.x > d.z {
            0
        } else if d.y > d.z {
            1
        } else {
            2
        }
    }

    /// Axis component by index.
    #[inline]
    fn axis_min(&self, axis: usize) -> f32 {
        match axis {
            0 => self.min.x,
            1 => self.min.y,
            _ => self.min.z,
        }
    }

    #[inline]
    fn axis_max(&self, axis: usize) -> f32 {
        match axis {
            0 => self.max.x,
            1 => self.max.y,
            _ => self.max.z,
        }
    }

    /// Ray-AABB intersection using the slab method.
    /// Returns (t_enter, t_exit). The intersection is valid if
    /// t_enter <= t_exit AND t_exit >= t_min AND t_enter <= t_max.
    #[inline]
    pub fn intersect_ray(&self, ray: &Ray) -> (f32, f32) {
        let t1 = (self.min - ray.origin) * ray.inv_direction;
        let t2 = (self.max - ray.origin) * ray.inv_direction;

        let t_min_v = t1.min(t2);
        let t_max_v = t1.max(t2);

        let t_enter = t_min_v.x.max(t_min_v.y).max(t_min_v.z);
        let t_exit = t_max_v.x.min(t_max_v.y).min(t_max_v.z);

        (t_enter, t_exit)
    }

    /// Fast ray-AABB hit test (boolean).
    #[inline]
    pub fn hit_ray(&self, ray: &Ray) -> bool {
        let (t_enter, t_exit) = self.intersect_ray(ray);
        t_enter <= t_exit && t_exit >= ray.t_min && t_enter <= ray.t_max
    }

    /// Transform an AABB by a 4x4 matrix (produces a conservative AABB).
    pub fn transform(&self, mat: &Mat4) -> AABB {
        let center = self.center();
        let ext = self.extent();

        let new_center_4 = *mat * Vec4::new(center.x, center.y, center.z, 1.0);
        let new_center = Vec3::new(new_center_4.x, new_center_4.y, new_center_4.z);

        // Absolute values of the rotation columns scaled by extent.
        let ax = Vec3::new(
            mat.x_axis.x.abs() * ext.x + mat.y_axis.x.abs() * ext.y + mat.z_axis.x.abs() * ext.z,
            mat.x_axis.y.abs() * ext.x + mat.y_axis.y.abs() * ext.y + mat.z_axis.y.abs() * ext.z,
            mat.x_axis.z.abs() * ext.x + mat.y_axis.z.abs() * ext.y + mat.z_axis.z.abs() * ext.z,
        );

        AABB {
            min: new_center - ax,
            max: new_center + ax,
        }
    }

    /// Check if two AABBs overlap.
    #[inline]
    pub fn overlaps(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Check if a point is inside the AABB.
    #[inline]
    pub fn contains_point(&self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }
}

impl Default for AABB {
    fn default() -> Self {
        Self::EMPTY
    }
}

// ---------------------------------------------------------------------------
// Triangle
// ---------------------------------------------------------------------------

/// A single triangle with vertices, normals, and UVs.
#[derive(Debug, Clone, Copy)]
pub struct Triangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
    pub n0: Vec3,
    pub n1: Vec3,
    pub n2: Vec3,
    pub uv0: Vec2,
    pub uv1: Vec2,
    pub uv2: Vec2,
}

impl Triangle {
    /// Construct a triangle from three positions (flat-shaded normal, zero UVs).
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3) -> Self {
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let n = edge1.cross(edge2).normalize_or_zero();
        Self {
            v0, v1, v2,
            n0: n, n1: n, n2: n,
            uv0: Vec2::ZERO, uv1: Vec2::ZERO, uv2: Vec2::ZERO,
        }
    }

    /// Construct a triangle with all attributes.
    pub fn with_attributes(
        v0: Vec3, v1: Vec3, v2: Vec3,
        n0: Vec3, n1: Vec3, n2: Vec3,
        uv0: Vec2, uv1: Vec2, uv2: Vec2,
    ) -> Self {
        Self { v0, v1, v2, n0, n1, n2, uv0, uv1, uv2 }
    }

    /// Compute the geometric (face) normal.
    #[inline]
    pub fn face_normal(&self) -> Vec3 {
        let e1 = self.v1 - self.v0;
        let e2 = self.v2 - self.v0;
        e1.cross(e2).normalize_or_zero()
    }

    /// Compute the AABB of this triangle.
    pub fn aabb(&self) -> AABB {
        AABB {
            min: self.v0.min(self.v1).min(self.v2),
            max: self.v0.max(self.v1).max(self.v2),
        }
    }

    /// Compute the centroid.
    #[inline]
    pub fn centroid(&self) -> Vec3 {
        (self.v0 + self.v1 + self.v2) * (1.0 / 3.0)
    }

    /// Area of the triangle.
    #[inline]
    pub fn area(&self) -> f32 {
        let e1 = self.v1 - self.v0;
        let e2 = self.v2 - self.v0;
        e1.cross(e2).length() * 0.5
    }

    /// Interpolate attributes at barycentric coordinates (u, v).
    /// The point is at: (1 - u - v) * v0 + u * v1 + v * v2.
    pub fn interpolate_position(&self, u: f32, v: f32) -> Vec3 {
        let w = 1.0 - u - v;
        self.v0 * w + self.v1 * u + self.v2 * v
    }

    pub fn interpolate_normal(&self, u: f32, v: f32) -> Vec3 {
        let w = 1.0 - u - v;
        (self.n0 * w + self.n1 * u + self.n2 * v).normalize_or_zero()
    }

    pub fn interpolate_uv(&self, u: f32, v: f32) -> Vec2 {
        let w = 1.0 - u - v;
        self.uv0 * w + self.uv1 * u + self.uv2 * v
    }
}

// ---------------------------------------------------------------------------
// Moller-Trumbore Ray-Triangle Intersection
// ---------------------------------------------------------------------------

/// Result of a Moller-Trumbore ray-triangle intersection test.
#[derive(Debug, Clone, Copy)]
pub struct TriangleHit {
    /// Parametric distance along the ray.
    pub t: f32,
    /// Barycentric u coordinate.
    pub u: f32,
    /// Barycentric v coordinate.
    pub v: f32,
    /// Whether the hit was on the back face.
    pub back_face: bool,
}

/// Moller-Trumbore ray-triangle intersection.
///
/// Returns `Some(TriangleHit)` if the ray intersects the triangle within
/// `[t_min, t_max]`. This is the standard algorithm from:
///   "Fast, Minimum Storage Ray/Triangle Intersection" (1997).
///
/// The barycentric coordinates satisfy:
///   hit_point = (1 - u - v) * v0 + u * v1 + v * v2
#[inline]
pub fn intersect_ray_triangle(
    ray: &Ray,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    t_min: f32,
    t_max: f32,
    cull_backface: bool,
) -> Option<TriangleHit> {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;

    // Begin calculating determinant -- also used to calculate u parameter.
    let pvec = ray.direction.cross(edge2);
    let det = edge1.dot(pvec);

    // If determinant is near zero, the ray is parallel to the triangle plane.
    if cull_backface {
        if det < 1e-8 {
            return None;
        }
    } else if det.abs() < 1e-8 {
        return None;
    }

    let inv_det = 1.0 / det;

    // Calculate distance from v0 to ray origin.
    let tvec = ray.origin - v0;

    // Calculate u parameter and test bounds.
    let u = tvec.dot(pvec) * inv_det;
    if u < 0.0 || u > 1.0 {
        return None;
    }

    // Prepare to test v parameter.
    let qvec = tvec.cross(edge1);

    // Calculate v parameter and test bounds.
    let v = ray.direction.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    // Calculate t -- the parametric distance along the ray.
    let t = edge2.dot(qvec) * inv_det;

    if t >= t_min && t <= t_max {
        Some(TriangleHit {
            t,
            u,
            v,
            back_face: det < 0.0,
        })
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// HitInfo -- full hit record
// ---------------------------------------------------------------------------

/// Complete information about a ray-scene intersection.
#[derive(Debug, Clone, Copy)]
pub struct HitInfo {
    /// World-space hit position.
    pub position: Vec3,
    /// Interpolated shading normal at the hit point.
    pub normal: Vec3,
    /// Geometric (face) normal.
    pub geometric_normal: Vec3,
    /// Interpolated UV coordinates.
    pub uv: Vec2,
    /// Index of the triangle within the BLAS.
    pub triangle_id: u32,
    /// Index of the mesh instance (TLAS instance).
    pub instance_id: u32,
    /// Material index for the hit instance.
    pub material_id: u32,
    /// Parametric distance from the ray origin.
    pub distance: f32,
    /// Whether the ray hit the back face of the triangle.
    pub back_face: bool,
}

// ---------------------------------------------------------------------------
// BVH Node
// ---------------------------------------------------------------------------

/// A node in the BVH tree. Uses a flat array representation for cache
/// efficiency. Each node is either an interior node (with two children)
/// or a leaf (with a range of primitives).
#[derive(Debug, Clone, Copy)]
pub struct BVHNode {
    /// Bounding box of this node.
    pub bounds: AABB,
    /// For interior nodes: index of the left child. Right child is at left_child + 1
    /// when using the "adjacent pair" layout, or stored in `right_or_count`.
    /// For leaf nodes: first primitive index.
    pub left_first: u32,
    /// For interior nodes: index of the right child.
    /// For leaf nodes: number of primitives.
    pub right_or_count: u32,
    /// The split axis used at this node (0=X, 1=Y, 2=Z). 3 means leaf.
    pub axis: u8,
    /// Whether this is a leaf node.
    pub is_leaf: bool,
}

impl BVHNode {
    /// Create a leaf node.
    pub fn leaf(bounds: AABB, first_prim: u32, prim_count: u32) -> Self {
        Self {
            bounds,
            left_first: first_prim,
            right_or_count: prim_count,
            axis: 3,
            is_leaf: true,
        }
    }

    /// Create an interior node.
    pub fn interior(bounds: AABB, left: u32, right: u32, axis: u8) -> Self {
        Self {
            bounds,
            left_first: left,
            right_or_count: right,
            axis,
            is_leaf: false,
        }
    }
}

// ---------------------------------------------------------------------------
// SAH BVH Builder
// ---------------------------------------------------------------------------

/// Surface Area Heuristic cost parameters.
const SAH_TRAVERSAL_COST: f32 = 1.0;
const SAH_INTERSECT_COST: f32 = 1.5;
/// Number of SAH buckets for binned partitioning.
const SAH_NUM_BUCKETS: usize = 12;
/// Maximum number of primitives in a leaf node.
const MAX_LEAF_SIZE: usize = 4;

/// A SAH bucket for binned BVH construction.
#[derive(Debug, Clone, Copy)]
struct SAHBucket {
    bounds: AABB,
    count: u32,
}

impl SAHBucket {
    fn empty() -> Self {
        Self {
            bounds: AABB::EMPTY,
            count: 0,
        }
    }
}

/// Build a BVH from a set of primitive AABBs and centroids using SAH.
///
/// Returns (nodes, reordered_indices) where `reordered_indices` maps each
/// position in the leaf ranges back to the original primitive index.
pub fn build_bvh_sah(
    prim_bounds: &[AABB],
    prim_centroids: &[Vec3],
) -> (Vec<BVHNode>, Vec<u32>) {
    let n = prim_bounds.len();
    if n == 0 {
        return (vec![BVHNode::leaf(AABB::EMPTY, 0, 0)], Vec::new());
    }

    let mut indices: Vec<u32> = (0..n as u32).collect();
    let mut nodes = Vec::with_capacity(2 * n);

    // Recursively build.
    build_bvh_recursive(
        &mut nodes,
        &mut indices,
        prim_bounds,
        prim_centroids,
        0,
        n,
    );

    (nodes, indices)
}

fn build_bvh_recursive(
    nodes: &mut Vec<BVHNode>,
    indices: &mut [u32],
    prim_bounds: &[AABB],
    prim_centroids: &[Vec3],
    start: usize,
    end: usize,
) -> u32 {
    let count = end - start;

    // Compute bounds of all primitives in this range.
    let mut node_bounds = AABB::EMPTY;
    for i in start..end {
        node_bounds.merge(&prim_bounds[indices[i] as usize]);
    }

    // If few enough primitives, make a leaf.
    if count <= MAX_LEAF_SIZE {
        let node_idx = nodes.len() as u32;
        nodes.push(BVHNode::leaf(node_bounds, start as u32, count as u32));
        return node_idx;
    }

    // Compute centroid bounds for splitting.
    let mut centroid_bounds = AABB::EMPTY;
    for i in start..end {
        centroid_bounds.expand_point(prim_centroids[indices[i] as usize]);
    }

    // Find the best split using SAH with binned partitioning.
    let mut best_cost = f32::MAX;
    let mut best_axis = 0usize;
    let mut best_split_bucket = 0usize;

    let parent_sa = node_bounds.surface_area();
    if parent_sa < 1e-12 {
        // Degenerate bounds -- make a leaf.
        let node_idx = nodes.len() as u32;
        nodes.push(BVHNode::leaf(node_bounds, start as u32, count as u32));
        return node_idx;
    }
    let inv_parent_sa = 1.0 / parent_sa;

    for axis in 0..3 {
        let axis_min = centroid_bounds.axis_min(axis);
        let axis_max = centroid_bounds.axis_max(axis);
        let axis_range = axis_max - axis_min;

        if axis_range < 1e-8 {
            continue; // All centroids coincide on this axis.
        }

        let inv_range = SAH_NUM_BUCKETS as f32 / axis_range;

        // Initialize buckets.
        let mut buckets = [SAHBucket::empty(); SAH_NUM_BUCKETS];

        // Assign primitives to buckets.
        for i in start..end {
            let c = centroid_axis(prim_centroids[indices[i] as usize], axis);
            let mut b = ((c - axis_min) * inv_range) as usize;
            if b >= SAH_NUM_BUCKETS {
                b = SAH_NUM_BUCKETS - 1;
            }
            buckets[b].count += 1;
            buckets[b].bounds.merge(&prim_bounds[indices[i] as usize]);
        }

        // Sweep from left to right to compute prefix areas/counts.
        let mut left_area = [0.0f32; SAH_NUM_BUCKETS - 1];
        let mut left_count = [0u32; SAH_NUM_BUCKETS - 1];
        let mut running_bounds = AABB::EMPTY;
        let mut running_count = 0u32;
        for i in 0..(SAH_NUM_BUCKETS - 1) {
            running_bounds.merge(&buckets[i].bounds);
            running_count += buckets[i].count;
            left_area[i] = running_bounds.surface_area();
            left_count[i] = running_count;
        }

        // Sweep from right to left.
        let mut right_area = [0.0f32; SAH_NUM_BUCKETS - 1];
        let mut right_count = [0u32; SAH_NUM_BUCKETS - 1];
        running_bounds = AABB::EMPTY;
        running_count = 0;
        for i in (0..(SAH_NUM_BUCKETS - 1)).rev() {
            running_bounds.merge(&buckets[i + 1].bounds);
            running_count += buckets[i + 1].count;
            right_area[i] = running_bounds.surface_area();
            right_count[i] = running_count;
        }

        // Evaluate SAH cost at each split position.
        for i in 0..(SAH_NUM_BUCKETS - 1) {
            if left_count[i] == 0 || right_count[i] == 0 {
                continue;
            }
            let cost = SAH_TRAVERSAL_COST
                + SAH_INTERSECT_COST
                    * (left_count[i] as f32 * left_area[i]
                        + right_count[i] as f32 * right_area[i])
                    * inv_parent_sa;
            if cost < best_cost {
                best_cost = cost;
                best_axis = axis;
                best_split_bucket = i;
            }
        }
    }

    // Compare best SAH cost to the cost of making a leaf.
    let leaf_cost = SAH_INTERSECT_COST * count as f32;
    if best_cost >= leaf_cost && count <= MAX_LEAF_SIZE * 2 {
        // Leaf is cheaper -- make a leaf node.
        let node_idx = nodes.len() as u32;
        nodes.push(BVHNode::leaf(node_bounds, start as u32, count as u32));
        return node_idx;
    }

    // If no valid split was found (all centroids on same axis), fallback to
    // midpoint split on the longest axis.
    if best_cost == f32::MAX {
        best_axis = centroid_bounds.longest_axis();

        // Partition at the midpoint of centroids.
        let mid = (centroid_bounds.axis_min(best_axis) + centroid_bounds.axis_max(best_axis)) * 0.5;
        let mut i = start;
        let mut j = end - 1;
        while i <= j && j > start {
            if centroid_axis(prim_centroids[indices[i] as usize], best_axis) <= mid {
                i += 1;
            } else {
                indices.swap(i, j);
                if j == 0 {
                    break;
                }
                j -= 1;
            }
        }

        let split = if i == start || i == end {
            (start + end) / 2
        } else {
            i
        };

        // Reserve this node's index.
        let node_idx = nodes.len() as u32;
        nodes.push(BVHNode::leaf(AABB::EMPTY, 0, 0)); // placeholder

        let left = build_bvh_recursive(nodes, indices, prim_bounds, prim_centroids, start, split);
        let right = build_bvh_recursive(nodes, indices, prim_bounds, prim_centroids, split, end);

        nodes[node_idx as usize] = BVHNode::interior(node_bounds, left, right, best_axis as u8);
        return node_idx;
    }

    // Partition primitives according to the best split.
    let axis_min = centroid_bounds.axis_min(best_axis);
    let axis_max = centroid_bounds.axis_max(best_axis);
    let inv_range = SAH_NUM_BUCKETS as f32 / (axis_max - axis_min).max(1e-8);

    let mut i = start;
    let mut j = end - 1;
    while i <= j && j >= start {
        let c = centroid_axis(prim_centroids[indices[i] as usize], best_axis);
        let mut b = ((c - axis_min) * inv_range) as usize;
        if b >= SAH_NUM_BUCKETS {
            b = SAH_NUM_BUCKETS - 1;
        }
        if b <= best_split_bucket {
            i += 1;
        } else {
            indices.swap(i, j);
            if j == 0 {
                break;
            }
            j -= 1;
        }
    }

    let split = if i == start || i == end {
        (start + end) / 2
    } else {
        i
    };

    // Reserve this node's index.
    let node_idx = nodes.len() as u32;
    nodes.push(BVHNode::leaf(AABB::EMPTY, 0, 0)); // placeholder

    let left = build_bvh_recursive(nodes, indices, prim_bounds, prim_centroids, start, split);
    let right = build_bvh_recursive(nodes, indices, prim_bounds, prim_centroids, split, end);

    nodes[node_idx as usize] = BVHNode::interior(node_bounds, left, right, best_axis as u8);
    node_idx
}

#[inline]
fn centroid_axis(c: Vec3, axis: usize) -> f32 {
    match axis {
        0 => c.x,
        1 => c.y,
        _ => c.z,
    }
}

// ---------------------------------------------------------------------------
// BLAS -- Bottom Level Acceleration Structure
// ---------------------------------------------------------------------------

/// Bottom Level Acceleration Structure -- BVH over the triangles of a single
/// mesh. The triangles are stored in a reordered array matching the BVH leaf
/// order for coherent traversal.
pub struct BLAS {
    /// Flat array of BVH nodes.
    pub nodes: Vec<BVHNode>,
    /// Triangles in BVH leaf order.
    pub triangles: Vec<Triangle>,
    /// Overall bounds of the mesh.
    pub bounds: AABB,
    /// Number of triangles.
    pub triangle_count: u32,
}

impl BLAS {
    /// Build a BLAS from a list of triangles using SAH.
    pub fn build(triangles: &[Triangle]) -> Self {
        if triangles.is_empty() {
            return Self {
                nodes: vec![BVHNode::leaf(AABB::EMPTY, 0, 0)],
                triangles: Vec::new(),
                bounds: AABB::EMPTY,
                triangle_count: 0,
            };
        }

        let n = triangles.len();
        let mut prim_bounds = Vec::with_capacity(n);
        let mut prim_centroids = Vec::with_capacity(n);

        for tri in triangles {
            prim_bounds.push(tri.aabb());
            prim_centroids.push(tri.centroid());
        }

        let (nodes, indices) = build_bvh_sah(&prim_bounds, &prim_centroids);

        // Reorder triangles to match BVH leaf order.
        let reordered: Vec<Triangle> = indices.iter().map(|&i| triangles[i as usize]).collect();

        let mut total_bounds = AABB::EMPTY;
        for b in &prim_bounds {
            total_bounds.merge(b);
        }

        Self {
            nodes,
            triangles: reordered,
            bounds: total_bounds,
            triangle_count: n as u32,
        }
    }

    /// Build a BLAS from raw vertex/index data (position + normal + uv).
    pub fn build_from_mesh(
        positions: &[Vec3],
        normals: &[Vec3],
        uvs: &[Vec2],
        indices: &[u32],
    ) -> Self {
        let tri_count = indices.len() / 3;
        let mut triangles = Vec::with_capacity(tri_count);

        for i in 0..tri_count {
            let i0 = indices[i * 3] as usize;
            let i1 = indices[i * 3 + 1] as usize;
            let i2 = indices[i * 3 + 2] as usize;

            let v0 = positions.get(i0).copied().unwrap_or(Vec3::ZERO);
            let v1 = positions.get(i1).copied().unwrap_or(Vec3::ZERO);
            let v2 = positions.get(i2).copied().unwrap_or(Vec3::ZERO);

            let n0 = normals.get(i0).copied().unwrap_or(Vec3::Y);
            let n1 = normals.get(i1).copied().unwrap_or(Vec3::Y);
            let n2 = normals.get(i2).copied().unwrap_or(Vec3::Y);

            let uv0 = uvs.get(i0).copied().unwrap_or(Vec2::ZERO);
            let uv1 = uvs.get(i1).copied().unwrap_or(Vec2::ZERO);
            let uv2 = uvs.get(i2).copied().unwrap_or(Vec2::ZERO);

            triangles.push(Triangle::with_attributes(v0, v1, v2, n0, n1, n2, uv0, uv1, uv2));
        }

        Self::build(&triangles)
    }

    /// Trace a ray against this BLAS. Returns the closest hit.
    pub fn trace_ray(&self, ray: &Ray) -> Option<(TriangleHit, u32)> {
        if self.nodes.is_empty() || self.triangles.is_empty() {
            return None;
        }

        let mut closest_t = ray.t_max;
        let mut closest_hit: Option<(TriangleHit, u32)> = None;

        // Stack-based iterative traversal.
        let mut stack = [0u32; 64];
        let mut stack_ptr = 0usize;
        stack[0] = 0; // root node
        stack_ptr = 1;

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let node_idx = stack[stack_ptr] as usize;
            let node = &self.nodes[node_idx];

            // Test AABB.
            let (t_enter, t_exit) = node.bounds.intersect_ray(ray);
            if t_enter > t_exit || t_exit < ray.t_min || t_enter > closest_t {
                continue;
            }

            if node.is_leaf {
                // Test each triangle in the leaf.
                let first = node.left_first as usize;
                let count = node.right_or_count as usize;
                for i in first..(first + count) {
                    if i >= self.triangles.len() {
                        break;
                    }
                    let tri = &self.triangles[i];
                    if let Some(hit) = intersect_ray_triangle(
                        ray, tri.v0, tri.v1, tri.v2, ray.t_min, closest_t, false,
                    ) {
                        if hit.t < closest_t {
                            closest_t = hit.t;
                            closest_hit = Some((hit, i as u32));
                        }
                    }
                }
            } else {
                // Push children. Visit the closer child first for better early termination.
                let left_idx = node.left_first;
                let right_idx = node.right_or_count;

                if stack_ptr + 2 > 64 {
                    continue; // stack overflow guard
                }

                // Determine traversal order based on ray direction and split axis.
                let dir_neg = match node.axis {
                    0 => ray.direction.x < 0.0,
                    1 => ray.direction.y < 0.0,
                    _ => ray.direction.z < 0.0,
                };

                if dir_neg {
                    stack[stack_ptr] = left_idx;
                    stack_ptr += 1;
                    stack[stack_ptr] = right_idx;
                    stack_ptr += 1;
                } else {
                    stack[stack_ptr] = right_idx;
                    stack_ptr += 1;
                    stack[stack_ptr] = left_idx;
                    stack_ptr += 1;
                }
            }
        }

        closest_hit
    }

    /// Any-hit query: returns true if any triangle is hit within [t_min, t_max].
    /// Used for shadow rays with early termination.
    pub fn any_hit(&self, ray: &Ray) -> bool {
        if self.nodes.is_empty() || self.triangles.is_empty() {
            return false;
        }

        let mut stack = [0u32; 64];
        let mut stack_ptr = 0usize;
        stack[0] = 0;
        stack_ptr = 1;

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let node_idx = stack[stack_ptr] as usize;
            let node = &self.nodes[node_idx];

            let (t_enter, t_exit) = node.bounds.intersect_ray(ray);
            if t_enter > t_exit || t_exit < ray.t_min || t_enter > ray.t_max {
                continue;
            }

            if node.is_leaf {
                let first = node.left_first as usize;
                let count = node.right_or_count as usize;
                for i in first..(first + count) {
                    if i >= self.triangles.len() {
                        break;
                    }
                    let tri = &self.triangles[i];
                    if intersect_ray_triangle(
                        ray, tri.v0, tri.v1, tri.v2, ray.t_min, ray.t_max, false,
                    )
                    .is_some()
                    {
                        return true;
                    }
                }
            } else {
                if stack_ptr + 2 > 64 {
                    continue;
                }
                stack[stack_ptr] = node.left_first;
                stack_ptr += 1;
                stack[stack_ptr] = node.right_or_count;
                stack_ptr += 1;
            }
        }

        false
    }

    /// Return BVH statistics for debugging.
    pub fn stats(&self) -> BVHStats {
        let mut stats = BVHStats::default();
        stats.node_count = self.nodes.len() as u32;
        stats.triangle_count = self.triangle_count;
        for node in &self.nodes {
            if node.is_leaf {
                stats.leaf_count += 1;
                stats.max_leaf_size =
                    stats.max_leaf_size.max(node.right_or_count);
                stats.total_leaf_prims += node.right_or_count;
            } else {
                stats.interior_count += 1;
            }
        }
        if stats.leaf_count > 0 {
            stats.avg_leaf_size = stats.total_leaf_prims as f32 / stats.leaf_count as f32;
        }
        stats
    }
}

/// BVH statistics for profiling and debugging.
#[derive(Debug, Default, Clone, Copy)]
pub struct BVHStats {
    pub node_count: u32,
    pub interior_count: u32,
    pub leaf_count: u32,
    pub triangle_count: u32,
    pub max_leaf_size: u32,
    pub avg_leaf_size: f32,
    pub total_leaf_prims: u32,
}

// ---------------------------------------------------------------------------
// MeshInstance -- per-instance data for TLAS
// ---------------------------------------------------------------------------

/// Describes a mesh instance in the scene for the top-level AS.
#[derive(Debug, Clone)]
pub struct MeshInstance {
    /// Index into the array of BLAS structures.
    pub blas_index: u32,
    /// Instance ID (user-defined, returned in HitInfo).
    pub instance_id: u32,
    /// Material ID for shading.
    pub material_id: u32,
    /// Object-to-world transform.
    pub transform: Mat4,
    /// World-to-object (inverse) transform.
    pub inv_transform: Mat4,
    /// Whether this instance should be included in tracing.
    pub visible: bool,
    /// Mask bits for ray filtering.
    pub mask: u32,
}

impl MeshInstance {
    /// Create a new mesh instance.
    pub fn new(blas_index: u32, instance_id: u32, transform: Mat4) -> Self {
        let inv = transform.inverse();
        Self {
            blas_index,
            instance_id,
            material_id: 0,
            transform,
            inv_transform: inv,
            visible: true,
            mask: 0xFFFF_FFFF,
        }
    }

    /// Create with a material ID.
    pub fn with_material(mut self, material_id: u32) -> Self {
        self.material_id = material_id;
        self
    }

    /// Set the instance mask for ray filtering.
    pub fn with_mask(mut self, mask: u32) -> Self {
        self.mask = mask;
        self
    }

    /// Compute the world-space AABB of this instance given its BLAS bounds.
    pub fn world_bounds(&self, blas_bounds: &AABB) -> AABB {
        blas_bounds.transform(&self.transform)
    }
}

// ---------------------------------------------------------------------------
// TLAS -- Top Level Acceleration Structure
// ---------------------------------------------------------------------------

/// Top Level Acceleration Structure -- BVH over mesh instances. Each leaf
/// references one or more instances whose BLAS is then traversed.
pub struct TLAS {
    /// BVH nodes for instance-level traversal.
    pub nodes: Vec<BVHNode>,
    /// Instances in BVH leaf order.
    pub instances: Vec<u32>,
    /// World-space bounds of the entire scene.
    pub bounds: AABB,
    /// Number of instances.
    pub instance_count: u32,
}

impl TLAS {
    /// Build a TLAS from instances and their BLAS structures.
    pub fn build(
        mesh_instances: &[MeshInstance],
        blas_list: &[BLAS],
    ) -> Self {
        let visible: Vec<usize> = mesh_instances
            .iter()
            .enumerate()
            .filter(|(_, inst)| inst.visible)
            .map(|(i, _)| i)
            .collect();

        if visible.is_empty() {
            return Self {
                nodes: vec![BVHNode::leaf(AABB::EMPTY, 0, 0)],
                instances: Vec::new(),
                bounds: AABB::EMPTY,
                instance_count: 0,
            };
        }

        let mut prim_bounds = Vec::with_capacity(visible.len());
        let mut prim_centroids = Vec::with_capacity(visible.len());

        for &idx in &visible {
            let inst = &mesh_instances[idx];
            let blas = &blas_list[inst.blas_index as usize];
            let wb = inst.world_bounds(&blas.bounds);
            prim_centroids.push(wb.center());
            prim_bounds.push(wb);
        }

        let (nodes, reordered) = build_bvh_sah(&prim_bounds, &prim_centroids);

        // Map reordered indices back to instance indices.
        let instance_indices: Vec<u32> = reordered
            .iter()
            .map(|&ri| visible[ri as usize] as u32)
            .collect();

        let mut total_bounds = AABB::EMPTY;
        for b in &prim_bounds {
            total_bounds.merge(b);
        }

        Self {
            nodes,
            instances: instance_indices,
            bounds: total_bounds,
            instance_count: visible.len() as u32,
        }
    }
}

// ---------------------------------------------------------------------------
// BVHAccelerationStructure -- complete two-level structure
// ---------------------------------------------------------------------------

/// The complete two-level acceleration structure containing all BLAS
/// meshes and the TLAS instance hierarchy.
pub struct BVHAccelerationStructure {
    /// Per-mesh bottom-level acceleration structures.
    pub blas_list: Vec<BLAS>,
    /// All mesh instances in the scene.
    pub instances: Vec<MeshInstance>,
    /// Top-level acceleration structure (rebuilt per frame or on change).
    pub tlas: TLAS,
}

impl BVHAccelerationStructure {
    /// Create a new empty acceleration structure.
    pub fn new() -> Self {
        Self {
            blas_list: Vec::new(),
            instances: Vec::new(),
            tlas: TLAS {
                nodes: vec![BVHNode::leaf(AABB::EMPTY, 0, 0)],
                instances: Vec::new(),
                bounds: AABB::EMPTY,
                instance_count: 0,
            },
        }
    }

    /// Add a mesh and build its BLAS. Returns the BLAS index.
    pub fn add_mesh(&mut self, triangles: &[Triangle]) -> u32 {
        let idx = self.blas_list.len() as u32;
        self.blas_list.push(BLAS::build(triangles));
        idx
    }

    /// Add a mesh instance. Returns the instance index.
    pub fn add_instance(&mut self, instance: MeshInstance) -> u32 {
        let idx = self.instances.len() as u32;
        self.instances.push(instance);
        idx
    }

    /// Rebuild the TLAS from all current instances.
    pub fn rebuild_tlas(&mut self) {
        self.tlas = TLAS::build(&self.instances, &self.blas_list);
    }

    /// Trace a single ray through the two-level structure.
    /// Returns the closest `HitInfo` if any intersection is found.
    pub fn trace_ray(&self, ray: &Ray) -> Option<HitInfo> {
        if self.tlas.nodes.is_empty() {
            return None;
        }

        let mut closest_t = ray.t_max;
        let mut closest_hit: Option<HitInfo> = None;

        // Traverse the TLAS with a stack.
        let mut stack = [0u32; 64];
        let mut stack_ptr = 0usize;
        stack[0] = 0;
        stack_ptr = 1;

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let node_idx = stack[stack_ptr] as usize;

            if node_idx >= self.tlas.nodes.len() {
                continue;
            }

            let node = &self.tlas.nodes[node_idx];

            // Test TLAS node AABB.
            let (t_enter, t_exit) = node.bounds.intersect_ray(ray);
            if t_enter > t_exit || t_exit < ray.t_min || t_enter > closest_t {
                continue;
            }

            if node.is_leaf {
                // Test each instance in this leaf.
                let first = node.left_first as usize;
                let count = node.right_or_count as usize;
                for i in first..(first + count) {
                    if i >= self.tlas.instances.len() {
                        break;
                    }

                    let instance_idx = self.tlas.instances[i];
                    let instance = &self.instances[instance_idx as usize];

                    if !instance.visible {
                        continue;
                    }

                    let blas = &self.blas_list[instance.blas_index as usize];

                    // Transform ray to object space.
                    let local_ray = ray.transform(&instance.inv_transform);
                    let local_ray = Ray {
                        t_max: closest_t, // Propagate current closest
                        ..local_ray
                    };

                    if let Some((tri_hit, tri_idx)) = blas.trace_ray(&local_ray) {
                        if tri_hit.t < closest_t {
                            let tri = &blas.triangles[tri_idx as usize];
                            let local_pos = tri.interpolate_position(tri_hit.u, tri_hit.v);
                            let local_normal = tri.interpolate_normal(tri_hit.u, tri_hit.v);
                            let local_geo_normal = tri.face_normal();
                            let uv = tri.interpolate_uv(tri_hit.u, tri_hit.v);

                            // Transform hit back to world space.
                            let world_pos = instance.transform.transform_point3(local_pos);
                            let world_normal = instance
                                .inv_transform
                                .transpose()
                                .transform_vector3(local_normal)
                                .normalize_or_zero();
                            let world_geo_normal = instance
                                .inv_transform
                                .transpose()
                                .transform_vector3(local_geo_normal)
                                .normalize_or_zero();

                            closest_t = tri_hit.t;
                            closest_hit = Some(HitInfo {
                                position: world_pos,
                                normal: world_normal,
                                geometric_normal: world_geo_normal,
                                uv,
                                triangle_id: tri_idx,
                                instance_id: instance.instance_id,
                                material_id: instance.material_id,
                                distance: (world_pos - ray.origin).length(),
                                back_face: tri_hit.back_face,
                            });
                        }
                    }
                }
            } else {
                if stack_ptr + 2 > 64 {
                    continue;
                }

                let left_idx = node.left_first;
                let right_idx = node.right_or_count;

                let dir_neg = match node.axis {
                    0 => ray.direction.x < 0.0,
                    1 => ray.direction.y < 0.0,
                    _ => ray.direction.z < 0.0,
                };

                if dir_neg {
                    stack[stack_ptr] = left_idx;
                    stack_ptr += 1;
                    stack[stack_ptr] = right_idx;
                    stack_ptr += 1;
                } else {
                    stack[stack_ptr] = right_idx;
                    stack_ptr += 1;
                    stack[stack_ptr] = left_idx;
                    stack_ptr += 1;
                }
            }
        }

        closest_hit
    }

    /// Trace a batch of rays (coherent tracing opportunity).
    pub fn trace_rays_batch(
        &self,
        rays: &[Ray],
        results: &mut [Option<HitInfo>],
    ) {
        assert_eq!(rays.len(), results.len());
        for (i, ray) in rays.iter().enumerate() {
            results[i] = self.trace_ray(ray);
        }
    }

    /// Any-hit query for shadow rays. Returns true if any geometry is hit
    /// between ray origin and `max_dist`.
    pub fn any_hit(&self, ray: &Ray) -> bool {
        if self.tlas.nodes.is_empty() {
            return false;
        }

        let mut stack = [0u32; 64];
        let mut stack_ptr = 0usize;
        stack[0] = 0;
        stack_ptr = 1;

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let node_idx = stack[stack_ptr] as usize;

            if node_idx >= self.tlas.nodes.len() {
                continue;
            }

            let node = &self.tlas.nodes[node_idx];

            let (t_enter, t_exit) = node.bounds.intersect_ray(ray);
            if t_enter > t_exit || t_exit < ray.t_min || t_enter > ray.t_max {
                continue;
            }

            if node.is_leaf {
                let first = node.left_first as usize;
                let count = node.right_or_count as usize;
                for i in first..(first + count) {
                    if i >= self.tlas.instances.len() {
                        break;
                    }

                    let instance_idx = self.tlas.instances[i];
                    let instance = &self.instances[instance_idx as usize];

                    if !instance.visible {
                        continue;
                    }

                    let blas = &self.blas_list[instance.blas_index as usize];
                    let local_ray = ray.transform(&instance.inv_transform);
                    if blas.any_hit(&local_ray) {
                        return true;
                    }
                }
            } else {
                if stack_ptr + 2 > 64 {
                    continue;
                }
                stack[stack_ptr] = node.left_first;
                stack_ptr += 1;
                stack[stack_ptr] = node.right_or_count;
                stack_ptr += 1;
            }
        }

        false
    }

    /// Return combined stats for all BLAS structures.
    pub fn stats(&self) -> Vec<BVHStats> {
        self.blas_list.iter().map(|b| b.stats()).collect()
    }

    /// Total triangle count across all BLAS.
    pub fn total_triangles(&self) -> u32 {
        self.blas_list.iter().map(|b| b.triangle_count).sum()
    }

    /// Total instance count.
    pub fn total_instances(&self) -> u32 {
        self.instances.len() as u32
    }
}

impl Default for BVHAccelerationStructure {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AnyHitQuery -- configurable any-hit for shadow/occlusion rays
// ---------------------------------------------------------------------------

/// Configuration for any-hit (shadow) queries.
#[derive(Debug, Clone, Copy)]
pub struct AnyHitQuery {
    /// Ray to test.
    pub ray: Ray,
    /// Mask bits: only test instances whose mask & this is non-zero.
    pub mask: u32,
    /// Whether to ignore back-face hits.
    pub cull_backface: bool,
}

impl AnyHitQuery {
    /// Create a shadow ray query between two points.
    pub fn shadow_ray(from: Vec3, to: Vec3) -> Self {
        let dir = to - from;
        let dist = dir.length();
        Self {
            ray: Ray::with_range(from, dir, 1e-3, dist - 1e-3),
            mask: 0xFFFF_FFFF,
            cull_backface: false,
        }
    }

    /// Execute the query against an acceleration structure.
    pub fn execute(&self, accel: &BVHAccelerationStructure) -> bool {
        accel.any_hit(&self.ray)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triangle_floor() -> Vec<Triangle> {
        vec![
            Triangle::new(
                Vec3::new(-5.0, 0.0, -5.0),
                Vec3::new(5.0, 0.0, -5.0),
                Vec3::new(5.0, 0.0, 5.0),
            ),
            Triangle::new(
                Vec3::new(-5.0, 0.0, -5.0),
                Vec3::new(5.0, 0.0, 5.0),
                Vec3::new(-5.0, 0.0, 5.0),
            ),
        ]
    }

    #[test]
    fn moller_trumbore_hit() {
        let v0 = Vec3::new(-1.0, 0.0, 0.0);
        let v1 = Vec3::new(1.0, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 1.0, 0.0);

        let ray = Ray::new(Vec3::new(0.0, 0.3, 1.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = intersect_ray_triangle(&ray, v0, v1, v2, 0.0, f32::MAX, false);
        assert!(hit.is_some());
        let h = hit.unwrap();
        assert!((h.t - 1.0).abs() < 1e-4);
    }

    #[test]
    fn moller_trumbore_miss() {
        let v0 = Vec3::new(-1.0, 0.0, 0.0);
        let v1 = Vec3::new(1.0, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 1.0, 0.0);

        let ray = Ray::new(Vec3::new(5.0, 5.0, 1.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = intersect_ray_triangle(&ray, v0, v1, v2, 0.0, f32::MAX, false);
        assert!(hit.is_none());
    }

    #[test]
    fn aabb_surface_area() {
        let aabb = AABB::new(Vec3::ZERO, Vec3::ONE);
        assert!((aabb.surface_area() - 6.0).abs() < 1e-5);
    }

    #[test]
    fn aabb_ray_hit() {
        let aabb = AABB::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let ray = Ray::new(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(aabb.hit_ray(&ray));
    }

    #[test]
    fn aabb_ray_miss() {
        let aabb = AABB::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let ray = Ray::new(Vec3::new(5.0, 5.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(!aabb.hit_ray(&ray));
    }

    #[test]
    fn blas_build_and_trace() {
        let tris = make_triangle_floor();
        let blas = BLAS::build(&tris);
        assert_eq!(blas.triangle_count, 2);

        // Ray pointing down should hit the floor.
        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = blas.trace_ray(&ray);
        assert!(hit.is_some());
        let (h, _) = hit.unwrap();
        assert!((h.t - 5.0).abs() < 1e-3);
    }

    #[test]
    fn blas_any_hit() {
        let tris = make_triangle_floor();
        let blas = BLAS::build(&tris);

        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        assert!(blas.any_hit(&ray));

        let ray_miss = Ray::new(Vec3::new(100.0, 5.0, 100.0), Vec3::new(0.0, -1.0, 0.0));
        assert!(!blas.any_hit(&ray_miss));
    }

    #[test]
    fn two_level_trace() {
        let tris = make_triangle_floor();
        let mut accel = BVHAccelerationStructure::new();
        let blas_idx = accel.add_mesh(&tris);
        accel.add_instance(MeshInstance::new(blas_idx, 0, Mat4::IDENTITY));
        accel.rebuild_tlas();

        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = accel.trace_ray(&ray);
        assert!(hit.is_some());
        let info = hit.unwrap();
        assert!((info.distance - 5.0).abs() < 0.1);
        assert!(info.normal.y.abs() > 0.9);
    }

    #[test]
    fn shadow_ray_query() {
        let tris = make_triangle_floor();
        let mut accel = BVHAccelerationStructure::new();
        let blas_idx = accel.add_mesh(&tris);
        accel.add_instance(MeshInstance::new(blas_idx, 0, Mat4::IDENTITY));
        accel.rebuild_tlas();

        let query = AnyHitQuery::shadow_ray(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, -5.0, 0.0),
        );
        assert!(query.execute(&accel));

        let query_miss = AnyHitQuery::shadow_ray(
            Vec3::new(100.0, 5.0, 100.0),
            Vec3::new(100.0, -5.0, 100.0),
        );
        assert!(!query_miss.execute(&accel));
    }

    #[test]
    fn bvh_stats() {
        let mut tris = Vec::new();
        for i in 0..20 {
            let x = (i % 5) as f32 * 2.0;
            let z = (i / 5) as f32 * 2.0;
            tris.push(Triangle::new(
                Vec3::new(x, 0.0, z),
                Vec3::new(x + 1.0, 0.0, z),
                Vec3::new(x, 0.0, z + 1.0),
            ));
        }
        let blas = BLAS::build(&tris);
        let stats = blas.stats();
        assert_eq!(stats.triangle_count, 20);
        assert!(stats.leaf_count > 0);
        assert!(stats.interior_count > 0);
    }

    #[test]
    fn batch_trace() {
        let tris = make_triangle_floor();
        let mut accel = BVHAccelerationStructure::new();
        let blas_idx = accel.add_mesh(&tris);
        accel.add_instance(MeshInstance::new(blas_idx, 0, Mat4::IDENTITY));
        accel.rebuild_tlas();

        let rays = vec![
            Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0)),
            Ray::new(Vec3::new(100.0, 5.0, 100.0), Vec3::new(0.0, -1.0, 0.0)),
        ];
        let mut results = vec![None; 2];
        accel.trace_rays_batch(&rays, &mut results);

        assert!(results[0].is_some());
        assert!(results[1].is_none());
    }
}
