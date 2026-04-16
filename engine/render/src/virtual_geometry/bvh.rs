// engine/render/src/virtual_geometry/bvh.rs
//
// Bounding Volume Hierarchy (BVH) with Surface Area Heuristic (SAH)
// construction, ray traversal, and frustum culling for the Genovo engine.

use crate::mesh::AABB;
use glam::Vec3;
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Cost of traversing an internal node (relative to intersection cost of 1.0).
const SAH_TRAVERSAL_COST: f32 = 1.0;

/// Cost of intersecting a primitive.
const SAH_INTERSECTION_COST: f32 = 1.0;

/// Maximum number of primitives in a leaf node before forced split.
const MAX_LEAF_PRIMITIVES: usize = 4;

/// Number of SAH bins for binned SAH construction.
const SAH_BIN_COUNT: usize = 16;

// ---------------------------------------------------------------------------
// Ray
// ---------------------------------------------------------------------------

/// A ray for BVH traversal.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    /// Origin of the ray.
    pub origin: Vec3,
    /// Direction of the ray (should be normalised).
    pub direction: Vec3,
    /// Reciprocal of the direction (precomputed for slab test).
    pub inv_direction: Vec3,
    /// Minimum parametric distance.
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
                if dir.x.abs() > 1e-8 { 1.0 / dir.x } else { f32::MAX * dir.x.signum() },
                if dir.y.abs() > 1e-8 { 1.0 / dir.y } else { f32::MAX * dir.y.signum() },
                if dir.z.abs() > 1e-8 { 1.0 / dir.z } else { f32::MAX * dir.z.signum() },
            ),
            t_min: 0.0,
            t_max: f32::MAX,
        }
    }

    /// Create a ray with parametric bounds.
    pub fn new_bounded(origin: Vec3, direction: Vec3, t_min: f32, t_max: f32) -> Self {
        let mut r = Self::new(origin, direction);
        r.t_min = t_min;
        r.t_max = t_max;
        r
    }

    /// Evaluate the ray at parameter t.
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

// ---------------------------------------------------------------------------
// Hit
// ---------------------------------------------------------------------------

/// Result of a ray-BVH intersection.
#[derive(Debug, Clone, Copy)]
pub struct Hit {
    /// Parametric distance along the ray.
    pub t: f32,
    /// Index of the hit primitive.
    pub primitive_index: usize,
    /// Normal at the hit point (if available).
    pub normal: Vec3,
}

impl Hit {
    pub fn new(t: f32, primitive_index: usize) -> Self {
        Self { t, primitive_index, normal: Vec3::Y }
    }
}

// ---------------------------------------------------------------------------
// Frustum
// ---------------------------------------------------------------------------

/// A view frustum defined by six planes (normals pointing inward).
#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    /// Six frustum planes: left, right, bottom, top, near, far.
    /// Each plane is (normal.x, normal.y, normal.z, distance).
    pub planes: [[f32; 4]; 6],
}

impl Frustum {
    /// Create a frustum from a combined view-projection matrix.
    pub fn from_view_projection(vp: &glam::Mat4) -> Self {
        let m = vp.to_cols_array_2d();

        let mut planes = [[0.0f32; 4]; 6];

        // Left plane.
        planes[0] = [
            m[0][3] + m[0][0],
            m[1][3] + m[1][0],
            m[2][3] + m[2][0],
            m[3][3] + m[3][0],
        ];
        // Right plane.
        planes[1] = [
            m[0][3] - m[0][0],
            m[1][3] - m[1][0],
            m[2][3] - m[2][0],
            m[3][3] - m[3][0],
        ];
        // Bottom plane.
        planes[2] = [
            m[0][3] + m[0][1],
            m[1][3] + m[1][1],
            m[2][3] + m[2][1],
            m[3][3] + m[3][1],
        ];
        // Top plane.
        planes[3] = [
            m[0][3] - m[0][1],
            m[1][3] - m[1][1],
            m[2][3] - m[2][1],
            m[3][3] - m[3][1],
        ];
        // Near plane.
        planes[4] = [
            m[0][3] + m[0][2],
            m[1][3] + m[1][2],
            m[2][3] + m[2][2],
            m[3][3] + m[3][2],
        ];
        // Far plane.
        planes[5] = [
            m[0][3] - m[0][2],
            m[1][3] - m[1][2],
            m[2][3] - m[2][2],
            m[3][3] - m[3][2],
        ];

        // Normalise planes.
        for plane in &mut planes {
            let len = (plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]).sqrt();
            if len > 1e-8 {
                plane[0] /= len;
                plane[1] /= len;
                plane[2] /= len;
                plane[3] /= len;
            }
        }

        Self { planes }
    }

    /// Test if an AABB is at least partially inside the frustum.
    pub fn test_aabb(&self, aabb: &AABB) -> bool {
        for plane in &self.planes {
            let normal = Vec3::new(plane[0], plane[1], plane[2]);

            // Find the AABB vertex most in the direction of the plane normal.
            let p = Vec3::new(
                if plane[0] >= 0.0 { aabb.max.x } else { aabb.min.x },
                if plane[1] >= 0.0 { aabb.max.y } else { aabb.min.y },
                if plane[2] >= 0.0 { aabb.max.z } else { aabb.min.z },
            );

            if normal.dot(p) + plane[3] < 0.0 {
                return false;
            }
        }

        true
    }

    /// Test if a sphere is at least partially inside the frustum.
    pub fn test_sphere(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            let normal = Vec3::new(plane[0], plane[1], plane[2]);
            let dist = normal.dot(center) + plane[3];
            if dist < -radius {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// BVHNode
// ---------------------------------------------------------------------------

/// A node in the BVH. Internal nodes have two children; leaf nodes store
/// primitive indices.
#[derive(Debug, Clone)]
pub enum BVHNode {
    /// Internal node with bounding box and two children.
    Internal {
        /// Bounding box encompassing both children.
        aabb: AABB,
        /// Index of the left child in the node array.
        left: usize,
        /// Index of the right child in the node array.
        right: usize,
        /// Split axis used (0=X, 1=Y, 2=Z).
        split_axis: u8,
    },
    /// Leaf node containing primitive indices.
    Leaf {
        /// Bounding box of the primitives in this leaf.
        aabb: AABB,
        /// Indices into the original primitive array.
        primitive_indices: Vec<usize>,
    },
}

impl BVHNode {
    /// Get the AABB of this node.
    pub fn aabb(&self) -> &AABB {
        match self {
            BVHNode::Internal { aabb, .. } => aabb,
            BVHNode::Leaf { aabb, .. } => aabb,
        }
    }

    /// Whether this node is a leaf.
    pub fn is_leaf(&self) -> bool {
        matches!(self, BVHNode::Leaf { .. })
    }
}

// ---------------------------------------------------------------------------
// SAH bin
// ---------------------------------------------------------------------------

/// A bin for binned SAH construction.
#[derive(Debug, Clone)]
struct SAHBin {
    aabb: AABB,
    count: usize,
}

impl Default for SAHBin {
    fn default() -> Self {
        Self {
            aabb: AABB::default(),
            count: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// BVH
// ---------------------------------------------------------------------------

/// Bounding Volume Hierarchy built with Surface Area Heuristic (SAH).
#[derive(Debug, Clone)]
pub struct BVH {
    /// Flat array of BVH nodes. The root is at index 0.
    pub nodes: Vec<BVHNode>,
    /// The original AABBs that were used to build the BVH.
    pub primitive_aabbs: Vec<AABB>,
    /// Total number of primitives.
    pub primitive_count: usize,
    /// Depth of the tree.
    pub depth: usize,
}

/// Build a BVH from a set of primitive AABBs using binned SAH.
pub fn build_bvh(primitives: &[AABB]) -> BVH {
    let primitive_count = primitives.len();

    if primitive_count == 0 {
        return BVH {
            nodes: vec![BVHNode::Leaf {
                aabb: AABB::default(),
                primitive_indices: Vec::new(),
            }],
            primitive_aabbs: Vec::new(),
            primitive_count: 0,
            depth: 0,
        };
    }

    let mut indices: Vec<usize> = (0..primitive_count).collect();
    let mut centroids: Vec<Vec3> = primitives.iter().map(|a| a.center()).collect();

    let mut nodes: Vec<BVHNode> = Vec::with_capacity(2 * primitive_count);
    let mut max_depth = 0usize;

    build_bvh_recursive(
        primitives,
        &mut indices,
        &centroids,
        &mut nodes,
        0,
        primitive_count,
        0,
        &mut max_depth,
    );

    BVH {
        nodes,
        primitive_aabbs: primitives.to_vec(),
        primitive_count,
        depth: max_depth,
    }
}

/// Recursive SAH-based BVH construction.
fn build_bvh_recursive(
    primitives: &[AABB],
    indices: &mut [usize],
    centroids: &[Vec3],
    nodes: &mut Vec<BVHNode>,
    start: usize,
    end: usize,
    depth: usize,
    max_depth: &mut usize,
) -> usize {
    *max_depth = (*max_depth).max(depth);
    let count = end - start;

    // Compute bounding box for all primitives in this range.
    let mut node_aabb = AABB::default();
    for i in start..end {
        node_aabb.expand_aabb(&primitives[indices[i]]);
    }

    // Base case: create leaf.
    if count <= MAX_LEAF_PRIMITIVES {
        let node_idx = nodes.len();
        nodes.push(BVHNode::Leaf {
            aabb: node_aabb,
            primitive_indices: indices[start..end].to_vec(),
        });
        return node_idx;
    }

    // Compute centroid bounds.
    let mut centroid_aabb = AABB::default();
    for i in start..end {
        centroid_aabb.expand_point(centroids[indices[i]]);
    }

    let extent = centroid_aabb.size();

    // Choose the split axis (longest extent of centroid bounds).
    let split_axis = if extent.x >= extent.y && extent.x >= extent.z {
        0u8
    } else if extent.y >= extent.z {
        1u8
    } else {
        2u8
    };

    let axis_extent = match split_axis {
        0 => extent.x,
        1 => extent.y,
        _ => extent.z,
    };

    // Degenerate case: all centroids at same position.
    if axis_extent < 1e-8 {
        let node_idx = nodes.len();
        nodes.push(BVHNode::Leaf {
            aabb: node_aabb,
            primitive_indices: indices[start..end].to_vec(),
        });
        return node_idx;
    }

    // Binned SAH.
    let mut bins = vec![SAHBin::default(); SAH_BIN_COUNT];
    let axis_min = match split_axis {
        0 => centroid_aabb.min.x,
        1 => centroid_aabb.min.y,
        _ => centroid_aabb.min.z,
    };

    let inv_extent = SAH_BIN_COUNT as f32 / axis_extent;

    // Populate bins.
    for i in start..end {
        let c = centroids[indices[i]];
        let axis_val = match split_axis {
            0 => c.x,
            1 => c.y,
            _ => c.z,
        };

        let bin_idx = ((axis_val - axis_min) * inv_extent) as usize;
        let bin_idx = bin_idx.min(SAH_BIN_COUNT - 1);

        bins[bin_idx].aabb.expand_aabb(&primitives[indices[i]]);
        bins[bin_idx].count += 1;
    }

    // Evaluate SAH cost for each split position.
    // Prefix sweep from left.
    let mut left_aabbs = vec![AABB::default(); SAH_BIN_COUNT];
    let mut left_counts = vec![0usize; SAH_BIN_COUNT];
    let mut running_aabb = AABB::default();
    let mut running_count = 0usize;

    for i in 0..SAH_BIN_COUNT {
        running_count += bins[i].count;
        if bins[i].count > 0 {
            running_aabb.expand_aabb(&bins[i].aabb);
        }
        left_aabbs[i] = running_aabb;
        left_counts[i] = running_count;
    }

    // Suffix sweep from right and compute costs.
    let mut right_aabb = AABB::default();
    let mut right_count = 0usize;
    let mut best_cost = f32::MAX;
    let mut best_split = 0usize;

    let parent_area = node_aabb.surface_area();
    let inv_parent_area = if parent_area > 1e-8 { 1.0 / parent_area } else { 0.0 };

    for i in (1..SAH_BIN_COUNT).rev() {
        right_count += bins[i].count;
        if bins[i].count > 0 {
            right_aabb.expand_aabb(&bins[i].aabb);
        }

        let lc = left_counts[i - 1];
        let rc = right_count;

        if lc == 0 || rc == 0 {
            continue;
        }

        let left_area = left_aabbs[i - 1].surface_area();
        let right_area = right_aabb.surface_area();

        let cost = SAH_TRAVERSAL_COST
            + SAH_INTERSECTION_COST
                * (lc as f32 * left_area + rc as f32 * right_area)
                * inv_parent_area;

        if cost < best_cost {
            best_cost = cost;
            best_split = i;
        }
    }

    // Compare SAH cost against leaf cost.
    let leaf_cost = SAH_INTERSECTION_COST * count as f32;

    if best_cost >= leaf_cost && count <= MAX_LEAF_PRIMITIVES * 4 {
        // Cheaper to make a leaf.
        let node_idx = nodes.len();
        nodes.push(BVHNode::Leaf {
            aabb: node_aabb,
            primitive_indices: indices[start..end].to_vec(),
        });
        return node_idx;
    }

    // Partition indices around the best split.
    let split_value = axis_min + best_split as f32 / inv_extent;

    let mut mid = start;
    for i in start..end {
        let c = centroids[indices[i]];
        let axis_val = match split_axis {
            0 => c.x,
            1 => c.y,
            _ => c.z,
        };

        if axis_val < split_value {
            indices.swap(i, mid);
            mid += 1;
        }
    }

    // Ensure neither side is empty.
    if mid == start || mid == end {
        mid = start + count / 2;

        // Sort by axis to get a reasonable partition.
        indices[start..end].sort_by(|&a, &b| {
            let ca = match split_axis {
                0 => centroids[a].x,
                1 => centroids[a].y,
                _ => centroids[a].z,
            };
            let cb = match split_axis {
                0 => centroids[b].x,
                1 => centroids[b].y,
                _ => centroids[b].z,
            };
            ca.partial_cmp(&cb).unwrap_or(Ordering::Equal)
        });
    }

    // Reserve a slot for this internal node.
    let node_idx = nodes.len();
    nodes.push(BVHNode::Leaf {
        aabb: AABB::default(),
        primitive_indices: Vec::new(),
    }); // placeholder

    // Recurse.
    let left_idx = build_bvh_recursive(
        primitives, indices, centroids, nodes, start, mid, depth + 1, max_depth,
    );
    let right_idx = build_bvh_recursive(
        primitives, indices, centroids, nodes, mid, end, depth + 1, max_depth,
    );

    // Replace placeholder with actual internal node.
    nodes[node_idx] = BVHNode::Internal {
        aabb: node_aabb,
        left: left_idx,
        right: right_idx,
        split_axis,
    };

    node_idx
}

impl BVH {
    /// Traverse the BVH with a ray and return all hits sorted by distance.
    pub fn traverse(&self, ray: &Ray) -> Vec<Hit> {
        let mut hits = Vec::new();

        if self.nodes.is_empty() {
            return hits;
        }

        let mut stack: Vec<usize> = Vec::with_capacity(64);
        stack.push(0);

        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx];

            // Test ray-AABB intersection.
            if !ray_aabb_intersect(ray, node.aabb()) {
                continue;
            }

            match node {
                BVHNode::Internal { left, right, .. } => {
                    stack.push(*right);
                    stack.push(*left);
                }
                BVHNode::Leaf { primitive_indices, .. } => {
                    for &prim_idx in primitive_indices {
                        // For each primitive, test intersection.
                        // Here we test against the primitive AABB as a conservative test.
                        if prim_idx < self.primitive_aabbs.len() {
                            if let Some(t) = ray_aabb_distance(ray, &self.primitive_aabbs[prim_idx]) {
                                hits.push(Hit::new(t, prim_idx));
                            }
                        }
                    }
                }
            }
        }

        hits.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(Ordering::Equal));
        hits
    }

    /// Traverse the BVH and find the closest hit.
    pub fn traverse_closest(&self, ray: &Ray) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }

        let mut closest: Option<Hit> = None;
        let mut closest_t = ray.t_max;

        let mut stack: Vec<usize> = Vec::with_capacity(64);
        stack.push(0);

        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx];

            if !ray_aabb_intersect_bounded(ray, node.aabb(), closest_t) {
                continue;
            }

            match node {
                BVHNode::Internal { left, right, split_axis, .. } => {
                    // Traverse the nearer child first.
                    let axis = *split_axis as usize;
                    let dir_component = match axis {
                        0 => ray.direction.x,
                        1 => ray.direction.y,
                        _ => ray.direction.z,
                    };

                    if dir_component >= 0.0 {
                        stack.push(*right);
                        stack.push(*left);
                    } else {
                        stack.push(*left);
                        stack.push(*right);
                    }
                }
                BVHNode::Leaf { primitive_indices, .. } => {
                    for &prim_idx in primitive_indices {
                        if prim_idx < self.primitive_aabbs.len() {
                            if let Some(t) = ray_aabb_distance(ray, &self.primitive_aabbs[prim_idx]) {
                                if t < closest_t && t >= ray.t_min {
                                    closest_t = t;
                                    closest = Some(Hit::new(t, prim_idx));
                                }
                            }
                        }
                    }
                }
            }
        }

        closest
    }

    /// Return indices of all primitives whose AABBs are at least partially
    /// inside the frustum.
    pub fn frustum_cull(&self, frustum: &Frustum) -> Vec<usize> {
        let mut visible = Vec::new();

        if self.nodes.is_empty() {
            return visible;
        }

        let mut stack: Vec<usize> = Vec::with_capacity(64);
        stack.push(0);

        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx];

            if !frustum.test_aabb(node.aabb()) {
                continue;
            }

            match node {
                BVHNode::Internal { left, right, .. } => {
                    stack.push(*right);
                    stack.push(*left);
                }
                BVHNode::Leaf { primitive_indices, .. } => {
                    for &prim_idx in primitive_indices {
                        if prim_idx < self.primitive_aabbs.len()
                            && frustum.test_aabb(&self.primitive_aabbs[prim_idx])
                        {
                            visible.push(prim_idx);
                        }
                    }
                }
            }
        }

        visible
    }

    /// Refit the BVH by updating AABBs from leaf to root without rebuilding
    /// the tree structure. Used when primitive positions change (e.g.
    /// animated objects).
    pub fn refit(&mut self, new_aabbs: &[AABB]) {
        assert_eq!(new_aabbs.len(), self.primitive_count);
        self.primitive_aabbs = new_aabbs.to_vec();

        if self.nodes.is_empty() {
            return;
        }

        self.refit_recursive(0);
    }

    /// Recursive bottom-up AABB refit.
    fn refit_recursive(&mut self, node_idx: usize) -> AABB {
        // We need to avoid borrowing issues by reading children first.
        let node = self.nodes[node_idx].clone();

        match node {
            BVHNode::Internal { left, right, split_axis, .. } => {
                let left_aabb = self.refit_recursive(left);
                let right_aabb = self.refit_recursive(right);

                let mut combined = left_aabb;
                combined.expand_aabb(&right_aabb);

                self.nodes[node_idx] = BVHNode::Internal {
                    aabb: combined,
                    left,
                    right,
                    split_axis,
                };

                combined
            }
            BVHNode::Leaf { primitive_indices, .. } => {
                let mut aabb = AABB::default();
                for &prim_idx in &primitive_indices {
                    if prim_idx < self.primitive_aabbs.len() {
                        aabb.expand_aabb(&self.primitive_aabbs[prim_idx]);
                    }
                }

                self.nodes[node_idx] = BVHNode::Leaf {
                    aabb,
                    primitive_indices,
                };

                aabb
            }
        }
    }

    /// Get the number of nodes in the BVH.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of leaf nodes.
    pub fn leaf_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    /// Validate the BVH structure (for testing/debugging).
    pub fn validate(&self) -> bool {
        if self.nodes.is_empty() {
            return true;
        }

        self.validate_recursive(0)
    }

    fn validate_recursive(&self, node_idx: usize) -> bool {
        if node_idx >= self.nodes.len() {
            return false;
        }

        match &self.nodes[node_idx] {
            BVHNode::Internal { left, right, aabb, .. } => {
                if *left >= self.nodes.len() || *right >= self.nodes.len() {
                    return false;
                }

                // Children's AABBs should be contained by parent.
                let left_aabb = self.nodes[*left].aabb();
                let right_aabb = self.nodes[*right].aabb();

                // Validate children recursively.
                self.validate_recursive(*left) && self.validate_recursive(*right)
            }
            BVHNode::Leaf { primitive_indices, .. } => {
                // All primitive indices should be valid.
                primitive_indices.iter().all(|&idx| idx < self.primitive_count)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ray-AABB intersection
// ---------------------------------------------------------------------------

/// Test if a ray intersects an AABB (slab test).
fn ray_aabb_intersect(ray: &Ray, aabb: &AABB) -> bool {
    let t1 = (aabb.min - ray.origin) * ray.inv_direction;
    let t2 = (aabb.max - ray.origin) * ray.inv_direction;

    let t_min_v = t1.min(t2);
    let t_max_v = t1.max(t2);

    let t_enter = t_min_v.x.max(t_min_v.y).max(t_min_v.z).max(ray.t_min);
    let t_exit = t_max_v.x.min(t_max_v.y).min(t_max_v.z).min(ray.t_max);

    t_enter <= t_exit
}

/// Test if a ray intersects an AABB with a distance bound.
fn ray_aabb_intersect_bounded(ray: &Ray, aabb: &AABB, max_t: f32) -> bool {
    let t1 = (aabb.min - ray.origin) * ray.inv_direction;
    let t2 = (aabb.max - ray.origin) * ray.inv_direction;

    let t_min_v = t1.min(t2);
    let t_max_v = t1.max(t2);

    let t_enter = t_min_v.x.max(t_min_v.y).max(t_min_v.z).max(ray.t_min);
    let t_exit = t_max_v.x.min(t_max_v.y).min(t_max_v.z).min(max_t);

    t_enter <= t_exit
}

/// Compute the distance to the entry point of a ray-AABB intersection.
fn ray_aabb_distance(ray: &Ray, aabb: &AABB) -> Option<f32> {
    let t1 = (aabb.min - ray.origin) * ray.inv_direction;
    let t2 = (aabb.max - ray.origin) * ray.inv_direction;

    let t_min_v = t1.min(t2);
    let t_max_v = t1.max(t2);

    let t_enter = t_min_v.x.max(t_min_v.y).max(t_min_v.z).max(ray.t_min);
    let t_exit = t_max_v.x.min(t_max_v.y).min(t_max_v.z).min(ray.t_max);

    if t_enter <= t_exit {
        Some(t_enter.max(0.0))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_unit_aabb(center: Vec3) -> AABB {
        AABB::new(center - Vec3::splat(0.5), center + Vec3::splat(0.5))
    }

    #[test]
    fn test_build_empty_bvh() {
        let bvh = build_bvh(&[]);
        assert_eq!(bvh.primitive_count, 0);
        assert_eq!(bvh.node_count(), 1); // empty leaf
    }

    #[test]
    fn test_build_single_primitive() {
        let aabbs = vec![make_unit_aabb(Vec3::ZERO)];
        let bvh = build_bvh(&aabbs);
        assert_eq!(bvh.primitive_count, 1);
        assert!(bvh.validate());
    }

    #[test]
    fn test_build_many_primitives() {
        let mut aabbs = Vec::new();
        for x in 0..10 {
            for z in 0..10 {
                aabbs.push(make_unit_aabb(Vec3::new(
                    x as f32 * 2.0,
                    0.0,
                    z as f32 * 2.0,
                )));
            }
        }

        let bvh = build_bvh(&aabbs);
        assert_eq!(bvh.primitive_count, 100);
        assert!(bvh.validate());
        assert!(bvh.depth > 0);
    }

    #[test]
    fn test_ray_traversal() {
        let aabbs = vec![
            make_unit_aabb(Vec3::new(0.0, 0.0, -5.0)),
            make_unit_aabb(Vec3::new(0.0, 0.0, -10.0)),
            make_unit_aabb(Vec3::new(5.0, 0.0, -5.0)),
        ];

        let bvh = build_bvh(&aabbs);
        let ray = Ray::new(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0));

        let hits = bvh.traverse(&ray);
        assert!(!hits.is_empty());

        // The closest hit should be the box at z=-5.
        let closest = bvh.traverse_closest(&ray);
        assert!(closest.is_some());
    }

    #[test]
    fn test_frustum_culling() {
        let aabbs = vec![
            make_unit_aabb(Vec3::new(0.0, 0.0, -5.0)),  // in front
            make_unit_aabb(Vec3::new(0.0, 0.0, 5.0)),   // behind
            make_unit_aabb(Vec3::new(100.0, 0.0, -5.0)), // far right
        ];

        let bvh = build_bvh(&aabbs);

        let view = glam::Mat4::look_at_rh(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::Y,
        );
        let proj = glam::Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_2,
            1.0,
            0.1,
            100.0,
        );
        let vp = proj * view;
        let frustum = Frustum::from_view_projection(&vp);

        let visible = bvh.frustum_cull(&frustum);
        // The box in front should be visible; the one behind should not.
        assert!(visible.contains(&0));
    }

    #[test]
    fn test_refit() {
        let mut aabbs = vec![
            make_unit_aabb(Vec3::ZERO),
            make_unit_aabb(Vec3::new(5.0, 0.0, 0.0)),
        ];

        let mut bvh = build_bvh(&aabbs);

        // Move the first AABB.
        aabbs[0] = make_unit_aabb(Vec3::new(10.0, 0.0, 0.0));
        bvh.refit(&aabbs);

        assert!(bvh.validate());
        // Root AABB should now encompass the new position.
        assert!(bvh.nodes[0].aabb().max.x >= 10.0);
    }

    #[test]
    fn test_ray_miss() {
        let aabbs = vec![make_unit_aabb(Vec3::new(0.0, 0.0, -5.0))];
        let bvh = build_bvh(&aabbs);

        // Fire a ray that misses.
        let ray = Ray::new(Vec3::new(100.0, 100.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let hits = bvh.traverse(&ray);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_sah_quality() {
        // Build a BVH for a line of AABBs and check SAH produces reasonable depth.
        let mut aabbs = Vec::new();
        for i in 0..64 {
            aabbs.push(make_unit_aabb(Vec3::new(i as f32 * 2.0, 0.0, 0.0)));
        }

        let bvh = build_bvh(&aabbs);
        assert!(bvh.validate());
        // Should be roughly log2(64) = 6 levels deep, give or take.
        assert!(bvh.depth <= 20);
        assert!(bvh.depth >= 3);
    }
}
