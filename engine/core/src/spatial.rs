//! Spatial data structures for the Genovo engine.
//!
//! Provides acceleration structures for spatial queries: k-d trees, octrees,
//! bounding volume hierarchies (BVH), spatial hash grids, and R-trees.

use glam::{Vec2, Vec3};
use std::collections::HashMap;

use crate::math::{AABB, Frustum, Ray};

const EPSILON: f32 = 1e-7;

// ===========================================================================
// KDTree -- k-dimensional tree (3-D)
// ===========================================================================

/// A node in a 3-D k-d tree.
#[derive(Debug)]
enum KDNode {
    Leaf {
        point: Vec3,
        data: u64,
    },
    Internal {
        split_axis: usize,
        split_value: f32,
        left: Box<KDNode>,
        right: Box<KDNode>,
    },
}

/// A k-d tree for efficient nearest-neighbor and range queries in 3-D space.
///
/// Points are split along the axis of maximum spread using the median, giving
/// a balanced tree of depth O(log n). Nearest-neighbor queries use
/// backtracking pruning for O(log n) expected time.
#[derive(Debug)]
pub struct KDTree {
    root: Option<Box<KDNode>>,
    size: usize,
}

/// An entry returned from k-d tree queries.
#[derive(Debug, Clone, Copy)]
pub struct KDEntry {
    pub point: Vec3,
    pub data: u64,
    pub distance_sq: f32,
}

impl KDTree {
    /// Builds a k-d tree from a set of (point, data) pairs using median-split.
    pub fn build(points: &[(Vec3, u64)]) -> Self {
        if points.is_empty() {
            return Self { root: None, size: 0 };
        }
        let mut entries: Vec<(Vec3, u64)> = points.to_vec();
        let size = entries.len();
        let root = Self::build_recursive(&mut entries, 0);
        Self {
            root: Some(root),
            size,
        }
    }

    fn build_recursive(entries: &mut [(Vec3, u64)], depth: usize) -> Box<KDNode> {
        let n = entries.len();
        if n == 1 {
            return Box::new(KDNode::Leaf {
                point: entries[0].0,
                data: entries[0].1,
            });
        }

        // Choose axis: cycle through x, y, z based on depth.
        // Better: choose axis with maximum spread.
        let axis = Self::axis_of_max_spread(entries);

        // Sort by the chosen axis and find median.
        entries.sort_by(|a, b| {
            let va = Self::axis_value(a.0, axis);
            let vb = Self::axis_value(b.0, axis);
            va.partial_cmp(&vb).unwrap()
        });

        let mid = n / 2;
        let split_value = Self::axis_value(entries[mid].0, axis);

        // Handle the case where mid == 0 (only 2 entries, left gets 1).
        if mid == 0 {
            // Put first entry in left, second in right.
            let left = Box::new(KDNode::Leaf {
                point: entries[0].0,
                data: entries[0].1,
            });
            let right = Box::new(KDNode::Leaf {
                point: entries[1].0,
                data: entries[1].1,
            });
            return Box::new(KDNode::Internal {
                split_axis: axis,
                split_value,
                left,
                right,
            });
        }

        let (left_slice, right_slice) = entries.split_at_mut(mid);
        let left = Self::build_recursive(left_slice, depth + 1);
        let right = Self::build_recursive(right_slice, depth + 1);

        Box::new(KDNode::Internal {
            split_axis: axis,
            split_value,
            left,
            right,
        })
    }

    fn axis_of_max_spread(entries: &[(Vec3, u64)]) -> usize {
        let mut best_axis = 0;
        let mut best_spread = f32::NEG_INFINITY;
        for axis in 0..3 {
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            for &(p, _) in entries {
                let v = Self::axis_value(p, axis);
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
            let spread = max_val - min_val;
            if spread > best_spread {
                best_spread = spread;
                best_axis = axis;
            }
        }
        best_axis
    }

    #[inline]
    fn axis_value(p: Vec3, axis: usize) -> f32 {
        match axis {
            0 => p.x,
            1 => p.y,
            _ => p.z,
        }
    }

    /// Returns the number of points in the tree.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns true if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Finds the nearest neighbor to `query`. Returns `None` if the tree is empty.
    ///
    /// Uses backtracking pruning: after finding a candidate in the nearer subtree,
    /// the farther subtree is only explored if the splitting plane is closer than
    /// the current best distance.
    pub fn nearest(&self, query: Vec3) -> Option<KDEntry> {
        match &self.root {
            None => None,
            Some(root) => {
                let mut best = KDEntry {
                    point: Vec3::ZERO,
                    data: 0,
                    distance_sq: f32::INFINITY,
                };
                Self::nearest_recursive(root, query, &mut best);
                if best.distance_sq < f32::INFINITY {
                    Some(best)
                } else {
                    None
                }
            }
        }
    }

    fn nearest_recursive(node: &KDNode, query: Vec3, best: &mut KDEntry) {
        match node {
            KDNode::Leaf { point, data } => {
                let dist_sq = (*point - query).length_squared();
                if dist_sq < best.distance_sq {
                    best.point = *point;
                    best.data = *data;
                    best.distance_sq = dist_sq;
                }
            }
            KDNode::Internal {
                split_axis,
                split_value,
                left,
                right,
            } => {
                let query_val = Self::axis_value(query, *split_axis);
                let diff = query_val - *split_value;

                // Determine near and far subtrees.
                let (near, far) = if diff <= 0.0 {
                    (left.as_ref(), right.as_ref())
                } else {
                    (right.as_ref(), left.as_ref())
                };

                // Search the near subtree first.
                Self::nearest_recursive(near, query, best);

                // Only search the far subtree if the splitting plane is closer
                // than our current best.
                if diff * diff < best.distance_sq {
                    Self::nearest_recursive(far, query, best);
                }
            }
        }
    }

    /// Finds the k nearest neighbors to `query`.
    ///
    /// Returns up to `k` entries sorted by distance (nearest first).
    pub fn k_nearest(&self, query: Vec3, k: usize) -> Vec<KDEntry> {
        if k == 0 || self.root.is_none() {
            return Vec::new();
        }

        let mut heap: Vec<KDEntry> = Vec::with_capacity(k + 1);

        if let Some(root) = &self.root {
            Self::knn_recursive(root, query, k, &mut heap);
        }

        heap.sort_by(|a, b| a.distance_sq.partial_cmp(&b.distance_sq).unwrap());
        heap
    }

    fn knn_recursive(node: &KDNode, query: Vec3, k: usize, heap: &mut Vec<KDEntry>) {
        match node {
            KDNode::Leaf { point, data } => {
                let dist_sq = (*point - query).length_squared();
                if heap.len() < k || dist_sq < heap.last().map_or(f32::INFINITY, |e| e.distance_sq)
                {
                    heap.push(KDEntry {
                        point: *point,
                        data: *data,
                        distance_sq: dist_sq,
                    });
                    heap.sort_by(|a, b| a.distance_sq.partial_cmp(&b.distance_sq).unwrap());
                    if heap.len() > k {
                        heap.pop();
                    }
                }
            }
            KDNode::Internal {
                split_axis,
                split_value,
                left,
                right,
            } => {
                let query_val = Self::axis_value(query, *split_axis);
                let diff = query_val - *split_value;

                let (near, far) = if diff <= 0.0 {
                    (left.as_ref(), right.as_ref())
                } else {
                    (right.as_ref(), left.as_ref())
                };

                Self::knn_recursive(near, query, k, heap);

                let worst_dist =
                    heap.last().map_or(f32::INFINITY, |e| e.distance_sq);
                if heap.len() < k || diff * diff < worst_dist {
                    Self::knn_recursive(far, query, k, heap);
                }
            }
        }
    }

    /// Range query: returns all points within `radius` of `center`.
    pub fn range_query(&self, center: Vec3, radius: f32) -> Vec<KDEntry> {
        let mut results = Vec::new();
        let radius_sq = radius * radius;
        if let Some(root) = &self.root {
            Self::range_recursive(root, center, radius_sq, &mut results);
        }
        results
    }

    fn range_recursive(
        node: &KDNode,
        center: Vec3,
        radius_sq: f32,
        results: &mut Vec<KDEntry>,
    ) {
        match node {
            KDNode::Leaf { point, data } => {
                let dist_sq = (*point - center).length_squared();
                if dist_sq <= radius_sq {
                    results.push(KDEntry {
                        point: *point,
                        data: *data,
                        distance_sq: dist_sq,
                    });
                }
            }
            KDNode::Internal {
                split_axis,
                split_value,
                left,
                right,
            } => {
                let center_val = Self::axis_value(center, *split_axis);
                let diff = center_val - *split_value;

                let (near, far) = if diff <= 0.0 {
                    (left.as_ref(), right.as_ref())
                } else {
                    (right.as_ref(), left.as_ref())
                };

                Self::range_recursive(near, center, radius_sq, results);

                if diff * diff <= radius_sq {
                    Self::range_recursive(far, center, radius_sq, results);
                }
            }
        }
    }
}

// ===========================================================================
// Octree
// ===========================================================================

/// Maximum number of items in a leaf node before splitting.
const OCTREE_MAX_LEAF_SIZE: usize = 16;
/// Maximum depth of the octree.
const OCTREE_MAX_DEPTH: usize = 20;

/// An item stored in the octree.
#[derive(Debug, Clone)]
struct OctreeItem {
    aabb: AABB,
    data: u64,
}

/// A node in the octree.
#[derive(Debug)]
enum OctreeNode {
    Leaf {
        items: Vec<OctreeItem>,
    },
    Internal {
        children: [Option<Box<OctreeNode>>; 8],
    },
}

/// A loose octree for spatial partitioning of 3-D objects.
///
/// Supports dynamic insertion, removal, and queries against AABBs, spheres,
/// and frustums. Internal nodes split when their item count exceeds a
/// threshold, and merge when items drop below it.
#[derive(Debug)]
pub struct Octree {
    root: OctreeNode,
    bounds: AABB,
    depth: usize,
}

impl Octree {
    /// Creates a new octree covering the given bounds.
    pub fn new(bounds: AABB) -> Self {
        Self {
            root: OctreeNode::Leaf { items: Vec::new() },
            bounds,
            depth: 0,
        }
    }

    /// Inserts an object with the given AABB and associated data.
    pub fn insert(&mut self, aabb: AABB, data: u64) {
        let item = OctreeItem { aabb, data };
        Self::insert_into_node(&mut self.root, &self.bounds, item, 0);
    }

    fn insert_into_node(
        node: &mut OctreeNode,
        node_bounds: &AABB,
        item: OctreeItem,
        depth: usize,
    ) {
        match node {
            OctreeNode::Leaf { items } => {
                items.push(item);
                // Split if over capacity and not at max depth.
                if items.len() > OCTREE_MAX_LEAF_SIZE && depth < OCTREE_MAX_DEPTH {
                    let center = node_bounds.center();
                    let mut children: [Option<Box<OctreeNode>>; 8] = Default::default();
                    for i in 0..8 {
                        children[i] = Some(Box::new(OctreeNode::Leaf { items: Vec::new() }));
                    }

                    let old_items = std::mem::take(items);
                    let mut new_node = OctreeNode::Internal { children };

                    for old_item in old_items {
                        Self::insert_into_node(&mut new_node, node_bounds, old_item, depth);
                    }

                    *node = new_node;
                }
            }
            OctreeNode::Internal { children } => {
                let octant = Self::find_octant(&item.aabb, node_bounds.center());
                match octant {
                    Some(idx) => {
                        let child_bounds = Self::child_bounds(node_bounds, idx);
                        if let Some(child) = &mut children[idx] {
                            Self::insert_into_node(child, &child_bounds, item, depth + 1);
                        }
                    }
                    None => {
                        // Item straddles multiple octants; insert into the first
                        // child that intersects (or store at this level by
                        // converting to a mixed node -- for simplicity, push into
                        // the first overlapping child).
                        for i in 0..8 {
                            let child_bounds = Self::child_bounds(node_bounds, i);
                            if child_bounds.intersects(&item.aabb) {
                                if let Some(child) = &mut children[i] {
                                    Self::insert_into_node(child, &child_bounds, item, depth + 1);
                                }
                                return;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Determines which octant an AABB falls entirely within, or `None` if it
    /// straddles the split planes.
    fn find_octant(aabb: &AABB, center: Vec3) -> Option<usize> {
        let min_gt = aabb.min.x >= center.x;
        let min_lt = aabb.max.x <= center.x;
        let xbit = if min_gt {
            Some(1)
        } else if min_lt {
            Some(0)
        } else {
            None
        };

        let min_gt_y = aabb.min.y >= center.y;
        let min_lt_y = aabb.max.y <= center.y;
        let ybit = if min_gt_y {
            Some(1)
        } else if min_lt_y {
            Some(0)
        } else {
            None
        };

        let min_gt_z = aabb.min.z >= center.z;
        let min_lt_z = aabb.max.z <= center.z;
        let zbit = if min_gt_z {
            Some(1)
        } else if min_lt_z {
            Some(0)
        } else {
            None
        };

        match (xbit, ybit, zbit) {
            (Some(x), Some(y), Some(z)) => Some(x | (y << 1) | (z << 2)),
            _ => None,
        }
    }

    /// Computes the AABB of a child octant.
    fn child_bounds(parent: &AABB, index: usize) -> AABB {
        let center = parent.center();
        let min = Vec3::new(
            if index & 1 != 0 { center.x } else { parent.min.x },
            if index & 2 != 0 { center.y } else { parent.min.y },
            if index & 4 != 0 { center.z } else { parent.min.z },
        );
        let max = Vec3::new(
            if index & 1 != 0 { parent.max.x } else { center.x },
            if index & 2 != 0 { parent.max.y } else { center.y },
            if index & 4 != 0 { parent.max.z } else { center.z },
        );
        AABB::new(min, max)
    }

    /// Removes an item by its data value. Returns `true` if found and removed.
    pub fn remove(&mut self, data: u64) -> bool {
        let removed = Self::remove_from_node(&mut self.root, data);
        if removed {
            Self::try_merge(&mut self.root);
        }
        removed
    }

    fn remove_from_node(node: &mut OctreeNode, data: u64) -> bool {
        match node {
            OctreeNode::Leaf { items } => {
                if let Some(pos) = items.iter().position(|i| i.data == data) {
                    items.swap_remove(pos);
                    return true;
                }
                false
            }
            OctreeNode::Internal { children } => {
                for child in children.iter_mut() {
                    if let Some(c) = child {
                        if Self::remove_from_node(c, data) {
                            return true;
                        }
                    }
                }
                false
            }
        }
    }

    /// Tries to merge an internal node back to a leaf if total items are few.
    fn try_merge(node: &mut OctreeNode) {
        if let OctreeNode::Internal { children } = node {
            let mut total = 0usize;
            let mut all_leaves = true;
            for child in children.iter() {
                if let Some(c) = child {
                    match c.as_ref() {
                        OctreeNode::Leaf { items } => total += items.len(),
                        OctreeNode::Internal { .. } => {
                            all_leaves = false;
                            break;
                        }
                    }
                }
            }
            if all_leaves && total <= OCTREE_MAX_LEAF_SIZE / 2 {
                let mut merged_items = Vec::with_capacity(total);
                for child in children.iter_mut() {
                    if let Some(c) = child {
                        if let OctreeNode::Leaf { items } = c.as_mut() {
                            merged_items.append(items);
                        }
                    }
                }
                *node = OctreeNode::Leaf { items: merged_items };
            }
        }
    }

    /// Returns all items whose AABB intersects the query AABB.
    pub fn query_aabb(&self, query: &AABB) -> Vec<u64> {
        let mut results = Vec::new();
        Self::query_aabb_node(&self.root, &self.bounds, query, &mut results);
        results
    }

    fn query_aabb_node(
        node: &OctreeNode,
        node_bounds: &AABB,
        query: &AABB,
        results: &mut Vec<u64>,
    ) {
        if !node_bounds.intersects(query) {
            return;
        }
        match node {
            OctreeNode::Leaf { items } => {
                for item in items {
                    if item.aabb.intersects(query) {
                        results.push(item.data);
                    }
                }
            }
            OctreeNode::Internal { children } => {
                for (i, child) in children.iter().enumerate() {
                    if let Some(c) = child {
                        let child_bounds = Self::child_bounds(node_bounds, i);
                        Self::query_aabb_node(c, &child_bounds, query, results);
                    }
                }
            }
        }
    }

    /// Returns all items whose AABB intersects a sphere.
    pub fn query_sphere(&self, center: Vec3, radius: f32) -> Vec<u64> {
        let mut results = Vec::new();
        Self::query_sphere_node(&self.root, &self.bounds, center, radius, &mut results);
        results
    }

    fn query_sphere_node(
        node: &OctreeNode,
        node_bounds: &AABB,
        center: Vec3,
        radius: f32,
        results: &mut Vec<u64>,
    ) {
        // Quick rejection: sphere vs AABB of the node.
        if !Self::sphere_aabb_intersect(center, radius, node_bounds) {
            return;
        }
        match node {
            OctreeNode::Leaf { items } => {
                for item in items {
                    if Self::sphere_aabb_intersect(center, radius, &item.aabb) {
                        results.push(item.data);
                    }
                }
            }
            OctreeNode::Internal { children } => {
                for (i, child) in children.iter().enumerate() {
                    if let Some(c) = child {
                        let child_bounds = Self::child_bounds(node_bounds, i);
                        Self::query_sphere_node(c, &child_bounds, center, radius, results);
                    }
                }
            }
        }
    }

    fn sphere_aabb_intersect(center: Vec3, radius: f32, aabb: &AABB) -> bool {
        // Closest point on the AABB to the sphere center.
        let closest = Vec3::new(
            center.x.clamp(aabb.min.x, aabb.max.x),
            center.y.clamp(aabb.min.y, aabb.max.y),
            center.z.clamp(aabb.min.z, aabb.max.z),
        );
        (closest - center).length_squared() <= radius * radius
    }

    /// Returns all items whose AABB is at least partially inside the frustum.
    pub fn query_frustum(&self, frustum: &Frustum) -> Vec<u64> {
        let mut results = Vec::new();
        Self::query_frustum_node(&self.root, &self.bounds, frustum, &mut results);
        results
    }

    fn query_frustum_node(
        node: &OctreeNode,
        node_bounds: &AABB,
        frustum: &Frustum,
        results: &mut Vec<u64>,
    ) {
        if !frustum.contains_aabb(node_bounds) {
            return;
        }
        match node {
            OctreeNode::Leaf { items } => {
                for item in items {
                    if frustum.contains_aabb(&item.aabb) {
                        results.push(item.data);
                    }
                }
            }
            OctreeNode::Internal { children } => {
                for (i, child) in children.iter().enumerate() {
                    if let Some(c) = child {
                        let child_bounds = Self::child_bounds(node_bounds, i);
                        Self::query_frustum_node(c, &child_bounds, frustum, results);
                    }
                }
            }
        }
    }
}

// ===========================================================================
// BVH -- Bounding Volume Hierarchy (SAH build)
// ===========================================================================

/// A node of the bounding volume hierarchy.
#[derive(Debug)]
enum BVHNode {
    Leaf {
        bounds: AABB,
        items: Vec<BVHLeafItem>,
    },
    Internal {
        bounds: AABB,
        left: Box<BVHNode>,
        right: Box<BVHNode>,
    },
}

/// An item stored in a BVH leaf, retaining the individual AABB for precise queries.
#[derive(Debug, Clone)]
struct BVHLeafItem {
    aabb: AABB,
    data: u64,
}

/// A bounding volume hierarchy built using the Surface Area Heuristic (SAH).
///
/// The SAH evaluates candidate splits along each axis and selects the split
/// that minimizes the expected cost of ray traversal (proportional to the
/// surface area of child nodes weighted by their item counts).
#[derive(Debug)]
pub struct BVH {
    root: Option<Box<BVHNode>>,
}

/// An item to insert into the BVH.
#[derive(Debug, Clone)]
pub struct BVHItem {
    pub aabb: AABB,
    pub data: u64,
}

/// Number of SAH buckets.
const SAH_BUCKET_COUNT: usize = 12;
/// Cost of traversing a node (relative to intersecting a primitive).
const SAH_TRAVERSAL_COST: f32 = 1.0;
/// Cost of intersecting a primitive.
const SAH_INTERSECT_COST: f32 = 1.0;

impl BVH {
    /// Builds a BVH from a list of items using the Surface Area Heuristic.
    pub fn build(items: &[BVHItem]) -> Self {
        if items.is_empty() {
            return Self { root: None };
        }
        let mut work: Vec<BVHItem> = items.to_vec();
        let root = Self::build_recursive(&mut work);
        Self {
            root: Some(root),
        }
    }

    fn build_recursive(items: &mut [BVHItem]) -> Box<BVHNode> {
        let n = items.len();

        // Compute overall bounds.
        let mut total_bounds = AABB::INVALID;
        for item in items.iter() {
            total_bounds = total_bounds.union(&item.aabb);
        }

        // Leaf threshold.
        if n <= 4 {
            return Box::new(BVHNode::Leaf {
                bounds: total_bounds,
                items: items.iter().map(|i| BVHLeafItem { aabb: i.aabb, data: i.data }).collect(),
            });
        }

        // Try SAH split on each axis.
        let mut best_cost = f32::INFINITY;
        let mut best_axis = 0usize;
        let mut best_split = 0usize;

        let total_sa = Self::surface_area(&total_bounds);
        let leaf_cost = n as f32 * SAH_INTERSECT_COST;

        for axis in 0..3 {
            // Sort items along axis by centroid.
            items.sort_by(|a, b| {
                let ca = Self::centroid_axis(&a.aabb, axis);
                let cb = Self::centroid_axis(&b.aabb, axis);
                ca.partial_cmp(&cb).unwrap()
            });

            // Build buckets.
            let mut buckets_bounds = vec![AABB::INVALID; SAH_BUCKET_COUNT];
            let mut buckets_count = vec![0usize; SAH_BUCKET_COUNT];

            let centroid_min = Self::centroid_axis(&items[0].aabb, axis);
            let centroid_max = Self::centroid_axis(&items[n - 1].aabb, axis);
            let centroid_range = centroid_max - centroid_min;

            if centroid_range < EPSILON {
                continue;
            }

            for item in items.iter() {
                let c = Self::centroid_axis(&item.aabb, axis);
                let mut bi =
                    ((c - centroid_min) / centroid_range * SAH_BUCKET_COUNT as f32) as usize;
                if bi >= SAH_BUCKET_COUNT {
                    bi = SAH_BUCKET_COUNT - 1;
                }
                buckets_bounds[bi] = buckets_bounds[bi].union(&item.aabb);
                buckets_count[bi] += 1;
            }

            // Sweep from left to find costs for each partition.
            let mut left_bounds = AABB::INVALID;
            let mut left_count = 0usize;
            for i in 0..(SAH_BUCKET_COUNT - 1) {
                left_bounds = left_bounds.union(&buckets_bounds[i]);
                left_count += buckets_count[i];

                let mut right_bounds = AABB::INVALID;
                let mut right_count = 0usize;
                for j in (i + 1)..SAH_BUCKET_COUNT {
                    right_bounds = right_bounds.union(&buckets_bounds[j]);
                    right_count += buckets_count[j];
                }

                if left_count == 0 || right_count == 0 {
                    continue;
                }

                let cost = SAH_TRAVERSAL_COST
                    + (Self::surface_area(&left_bounds) * left_count as f32
                        + Self::surface_area(&right_bounds) * right_count as f32)
                        * SAH_INTERSECT_COST
                        / total_sa;

                if cost < best_cost {
                    best_cost = cost;
                    best_axis = axis;
                    // Find the actual split index: left_count items go to the left.
                    best_split = left_count;
                }
            }
        }

        // If no good split found, or the leaf cost is lower, make a leaf.
        if best_cost >= leaf_cost || best_split == 0 || best_split >= n {
            return Box::new(BVHNode::Leaf {
                bounds: total_bounds,
                items: items.iter().map(|i| BVHLeafItem { aabb: i.aabb, data: i.data }).collect(),
            });
        }

        // Partition along best axis.
        items.sort_by(|a, b| {
            let ca = Self::centroid_axis(&a.aabb, best_axis);
            let cb = Self::centroid_axis(&b.aabb, best_axis);
            ca.partial_cmp(&cb).unwrap()
        });

        let (left_items, right_items) = items.split_at_mut(best_split);
        let left = Self::build_recursive(left_items);
        let right = Self::build_recursive(right_items);

        Box::new(BVHNode::Internal {
            bounds: total_bounds,
            left,
            right,
        })
    }

    fn centroid_axis(aabb: &AABB, axis: usize) -> f32 {
        let c = aabb.center();
        match axis {
            0 => c.x,
            1 => c.y,
            _ => c.z,
        }
    }

    fn surface_area(aabb: &AABB) -> f32 {
        let d = aabb.max - aabb.min;
        if d.x < 0.0 || d.y < 0.0 || d.z < 0.0 {
            return 0.0;
        }
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Tests whether a point is inside any item's AABB. Returns matching data values.
    pub fn query_point(&self, point: Vec3) -> Vec<u64> {
        let mut results = Vec::new();
        if let Some(root) = &self.root {
            Self::query_point_node(root, point, &mut results);
        }
        results
    }

    fn query_point_node(node: &BVHNode, point: Vec3, results: &mut Vec<u64>) {
        match node {
            BVHNode::Leaf { bounds, items } => {
                if bounds.contains_point(point) {
                    for item in items {
                        if item.aabb.contains_point(point) {
                            results.push(item.data);
                        }
                    }
                }
            }
            BVHNode::Internal { bounds, left, right } => {
                if bounds.contains_point(point) {
                    Self::query_point_node(left, point, results);
                    Self::query_point_node(right, point, results);
                }
            }
        }
    }

    /// Returns all items whose AABB intersects the query AABB.
    pub fn query_aabb(&self, query: &AABB) -> Vec<u64> {
        let mut results = Vec::new();
        if let Some(root) = &self.root {
            Self::query_aabb_node(root, query, &mut results);
        }
        results
    }

    fn query_aabb_node(node: &BVHNode, query: &AABB, results: &mut Vec<u64>) {
        match node {
            BVHNode::Leaf { bounds, items } => {
                if bounds.intersects(query) {
                    for item in items {
                        if item.aabb.intersects(query) {
                            results.push(item.data);
                        }
                    }
                }
            }
            BVHNode::Internal { bounds, left, right } => {
                if bounds.intersects(query) {
                    Self::query_aabb_node(left, query, results);
                    Self::query_aabb_node(right, query, results);
                }
            }
        }
    }

    /// Casts a ray into the BVH and returns the data values of all items whose
    /// AABB the ray intersects, along with the distance.
    pub fn query_ray(&self, ray: &Ray) -> Vec<(u64, f32)> {
        let mut results = Vec::new();
        if let Some(root) = &self.root {
            Self::query_ray_node(root, ray, &mut results);
        }
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results
    }

    fn query_ray_node(node: &BVHNode, ray: &Ray, results: &mut Vec<(u64, f32)>) {
        match node {
            BVHNode::Leaf { bounds, items } => {
                if bounds.ray_intersect(ray).is_some() {
                    for item in items {
                        if let Some(t) = item.aabb.ray_intersect(ray) {
                            results.push((item.data, t));
                        }
                    }
                }
            }
            BVHNode::Internal { bounds, left, right } => {
                if bounds.ray_intersect(ray).is_some() {
                    Self::query_ray_node(left, ray, results);
                    Self::query_ray_node(right, ray, results);
                }
            }
        }
    }

    /// Refits the BVH after items have moved. This recomputes bounds bottom-up
    /// without changing the tree topology.
    ///
    /// `get_aabb` is called with each item's data value and should return the
    /// updated AABB.
    pub fn refit(&mut self, get_aabb: &dyn Fn(u64) -> AABB) {
        if let Some(root) = &mut self.root {
            Self::refit_node(root, get_aabb);
        }
    }

    fn refit_node(node: &mut BVHNode, get_aabb: &dyn Fn(u64) -> AABB) -> AABB {
        match node {
            BVHNode::Leaf { bounds, items } => {
                let mut new_bounds = AABB::INVALID;
                for item in items.iter_mut() {
                    let new_aabb = get_aabb(item.data);
                    item.aabb = new_aabb;
                    new_bounds = new_bounds.union(&new_aabb);
                }
                *bounds = new_bounds;
                new_bounds
            }
            BVHNode::Internal { bounds, left, right } => {
                let lb = Self::refit_node(left, get_aabb);
                let rb = Self::refit_node(right, get_aabb);
                *bounds = lb.union(&rb);
                *bounds
            }
        }
    }
}

// ===========================================================================
// SpatialHashGrid
// ===========================================================================

/// An unbounded spatial hash grid backed by a `HashMap`.
///
/// Objects are hashed into grid cells based on their position. Suitable for
/// uniform-density scenarios like particle systems or entity proximity queries.
#[derive(Debug)]
pub struct SpatialHashGrid {
    cell_size: f32,
    inv_cell_size: f32,
    cells: HashMap<(i32, i32, i32), Vec<SpatialHashEntry>>,
}

#[derive(Debug, Clone)]
struct SpatialHashEntry {
    position: Vec3,
    radius: f32,
    data: u64,
}

impl SpatialHashGrid {
    /// Creates a new spatial hash grid with the given cell size.
    ///
    /// `cell_size` should be approximately the diameter of the largest object
    /// or the query radius for best performance.
    pub fn new(cell_size: f32) -> Self {
        assert!(cell_size > 0.0, "Cell size must be positive");
        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            cells: HashMap::new(),
        }
    }

    /// Converts a world-space coordinate to a cell coordinate.
    #[inline]
    fn cell_coord(&self, pos: Vec3) -> (i32, i32, i32) {
        (
            (pos.x * self.inv_cell_size).floor() as i32,
            (pos.y * self.inv_cell_size).floor() as i32,
            (pos.z * self.inv_cell_size).floor() as i32,
        )
    }

    /// Inserts an object at the given position with a bounding radius.
    pub fn insert(&mut self, position: Vec3, radius: f32, data: u64) {
        let entry = SpatialHashEntry {
            position,
            radius,
            data,
        };

        // Insert into all cells that the object's bounding sphere overlaps.
        let min = self.cell_coord(position - Vec3::splat(radius));
        let max = self.cell_coord(position + Vec3::splat(radius));
        for x in min.0..=max.0 {
            for y in min.1..=max.1 {
                for z in min.2..=max.2 {
                    self.cells
                        .entry((x, y, z))
                        .or_insert_with(Vec::new)
                        .push(entry.clone());
                }
            }
        }
    }

    /// Removes an object by its data value. Returns `true` if found.
    pub fn remove(&mut self, data: u64) -> bool {
        let mut found = false;
        for (_, entries) in self.cells.iter_mut() {
            if let Some(pos) = entries.iter().position(|e| e.data == data) {
                entries.swap_remove(pos);
                found = true;
            }
        }
        // Clean up empty cells.
        self.cells.retain(|_, v| !v.is_empty());
        found
    }

    /// Updates an object's position. Equivalent to remove + insert.
    pub fn update(&mut self, data: u64, new_position: Vec3, new_radius: f32) {
        self.remove(data);
        self.insert(new_position, new_radius, data);
    }

    /// Returns all objects within `radius` of `center`.
    ///
    /// The query considers the object's own bounding radius: an object is
    /// returned if the distance between centers minus the object's radius is
    /// less than the query radius.
    pub fn query_radius(&self, center: Vec3, radius: f32) -> Vec<u64> {
        let mut results = Vec::new();
        let mut seen = std::collections::HashSet::new();

        let min = self.cell_coord(center - Vec3::splat(radius));
        let max = self.cell_coord(center + Vec3::splat(radius));

        for x in min.0..=max.0 {
            for y in min.1..=max.1 {
                for z in min.2..=max.2 {
                    if let Some(entries) = self.cells.get(&(x, y, z)) {
                        for entry in entries {
                            if seen.contains(&entry.data) {
                                continue;
                            }
                            let dist = (entry.position - center).length();
                            if dist - entry.radius <= radius {
                                results.push(entry.data);
                                seen.insert(entry.data);
                            }
                        }
                    }
                }
            }
        }

        results
    }

    /// Returns the number of non-empty cells.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Clears all entries from the grid.
    pub fn clear(&mut self) {
        self.cells.clear();
    }
}

// ===========================================================================
// RTree (Sort-Tile-Recursive bulk load)
// ===========================================================================

/// A 2-D rectangle for the R-tree, defined by min and max corners.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RTreeRect {
    pub min: Vec2,
    pub max: Vec2,
}

impl RTreeRect {
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    pub fn area(&self) -> f32 {
        let d = self.max - self.min;
        if d.x < 0.0 || d.y < 0.0 {
            return 0.0;
        }
        d.x * d.y
    }

    pub fn perimeter(&self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x.max(0.0) + d.y.max(0.0))
    }

    pub fn center(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    pub fn contains_point(&self, p: Vec2) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }

    pub fn intersects(&self, other: &RTreeRect) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }

    pub fn union(&self, other: &RTreeRect) -> RTreeRect {
        RTreeRect {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn expand_to_include(&mut self, other: &RTreeRect) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    pub const INVALID: Self = Self {
        min: Vec2::splat(f32::INFINITY),
        max: Vec2::splat(f32::NEG_INFINITY),
    };
}

/// A node of the R-tree.
#[derive(Debug)]
enum RTreeNode {
    Leaf {
        bounds: RTreeRect,
        items: Vec<RTreeLeafEntry>,
    },
    Internal {
        bounds: RTreeRect,
        children: Vec<Box<RTreeNode>>,
    },
}

#[derive(Debug, Clone)]
struct RTreeLeafEntry {
    rect: RTreeRect,
    data: u64,
}

/// Maximum number of entries per R-tree node.
const RTREE_MAX_ENTRIES: usize = 16;

/// An R-tree for 2-D rectangle queries, built using the Sort-Tile-Recursive
/// (STR) bulk-loading algorithm.
///
/// STR sorts items along each dimension in a round-robin fashion, then packs
/// them into nodes of `RTREE_MAX_ENTRIES` entries, recursing until a single
/// root node remains.
#[derive(Debug)]
pub struct RTree {
    root: Option<Box<RTreeNode>>,
    count: usize,
}

impl RTree {
    /// Bulk-loads an R-tree from a set of (rect, data) pairs using the
    /// Sort-Tile-Recursive algorithm.
    pub fn bulk_load(entries: &[(RTreeRect, u64)]) -> Self {
        if entries.is_empty() {
            return Self {
                root: None,
                count: 0,
            };
        }

        let count = entries.len();
        let mut leaf_entries: Vec<RTreeLeafEntry> = entries
            .iter()
            .map(|&(rect, data)| RTreeLeafEntry { rect, data })
            .collect();

        let root = Self::str_build(&mut leaf_entries);
        Self {
            root: Some(root),
            count,
        }
    }

    fn str_build(entries: &mut [RTreeLeafEntry]) -> Box<RTreeNode> {
        let n = entries.len();
        if n <= RTREE_MAX_ENTRIES {
            let mut bounds = RTreeRect::INVALID;
            for e in entries.iter() {
                bounds.expand_to_include(&e.rect);
            }
            return Box::new(RTreeNode::Leaf {
                bounds,
                items: entries.to_vec(),
            });
        }

        // Sort-Tile-Recursive: compute the number of slices.
        let num_leaves = (n + RTREE_MAX_ENTRIES - 1) / RTREE_MAX_ENTRIES;
        let num_slices_x = (num_leaves as f32).sqrt().ceil() as usize;
        let slice_size = num_slices_x * RTREE_MAX_ENTRIES;

        // Sort by x-center.
        entries.sort_by(|a, b| {
            a.rect
                .center()
                .x
                .partial_cmp(&b.rect.center().x)
                .unwrap()
        });

        let mut nodes: Vec<Box<RTreeNode>> = Vec::new();

        for slice_start in (0..n).step_by(slice_size) {
            let slice_end = (slice_start + slice_size).min(n);
            let slice = &mut entries[slice_start..slice_end];

            // Sort each x-slice by y-center.
            slice.sort_by(|a, b| {
                a.rect
                    .center()
                    .y
                    .partial_cmp(&b.rect.center().y)
                    .unwrap()
            });

            // Pack into leaf nodes.
            for chunk_start in (0..slice.len()).step_by(RTREE_MAX_ENTRIES) {
                let chunk_end = (chunk_start + RTREE_MAX_ENTRIES).min(slice.len());
                let chunk = &slice[chunk_start..chunk_end];
                let mut bounds = RTreeRect::INVALID;
                for e in chunk {
                    bounds.expand_to_include(&e.rect);
                }
                nodes.push(Box::new(RTreeNode::Leaf {
                    bounds,
                    items: chunk.to_vec(),
                }));
            }
        }

        // Recursively pack internal nodes.
        Self::pack_internal(nodes)
    }

    fn pack_internal(mut nodes: Vec<Box<RTreeNode>>) -> Box<RTreeNode> {
        while nodes.len() > RTREE_MAX_ENTRIES {
            let n = nodes.len();
            let mut new_nodes: Vec<Box<RTreeNode>> = Vec::new();

            // Sort nodes by their bounds center x.
            nodes.sort_by(|a, b| {
                let ca = Self::node_bounds(a).center().x;
                let cb = Self::node_bounds(b).center().x;
                ca.partial_cmp(&cb).unwrap()
            });

            let _num_groups = (n + RTREE_MAX_ENTRIES - 1) / RTREE_MAX_ENTRIES;
            let group_size = RTREE_MAX_ENTRIES;

            for chunk in nodes.chunks_mut(group_size) {
                // Sort each chunk by y.
                chunk.sort_by(|a, b| {
                    let ca = Self::node_bounds(a).center().y;
                    let cb = Self::node_bounds(b).center().y;
                    ca.partial_cmp(&cb).unwrap()
                });
            }

            for start in (0..n).step_by(RTREE_MAX_ENTRIES) {
                let end = (start + RTREE_MAX_ENTRIES).min(n);
                // We need to drain, but since we're iterating, build from a slice.
                let children: Vec<Box<RTreeNode>> = (start..end)
                    .map(|_| {
                        // Placeholder -- will be replaced below.
                        Box::new(RTreeNode::Leaf {
                            bounds: RTreeRect::INVALID,
                            items: Vec::new(),
                        })
                    })
                    .collect();
                // Actually we need a different approach since we can't easily
                // drain chunks. Collect all nodes into a new set.
                let _ = children;
            }

            // Simpler approach: chunk and create internal nodes.
            let all_nodes = std::mem::take(&mut nodes);
            let mut iter = all_nodes.into_iter();
            loop {
                let mut children: Vec<Box<RTreeNode>> = Vec::new();
                for _ in 0..RTREE_MAX_ENTRIES {
                    if let Some(node) = iter.next() {
                        children.push(node);
                    } else {
                        break;
                    }
                }
                if children.is_empty() {
                    break;
                }
                let mut bounds = RTreeRect::INVALID;
                for child in &children {
                    bounds.expand_to_include(&Self::node_bounds(child));
                }
                new_nodes.push(Box::new(RTreeNode::Internal { bounds, children }));
            }

            nodes = new_nodes;
        }

        // Final root node.
        if nodes.len() == 1 {
            return nodes.into_iter().next().unwrap();
        }

        let mut bounds = RTreeRect::INVALID;
        for node in &nodes {
            bounds.expand_to_include(&Self::node_bounds(node));
        }
        Box::new(RTreeNode::Internal {
            bounds,
            children: nodes,
        })
    }

    fn node_bounds(node: &RTreeNode) -> RTreeRect {
        match node {
            RTreeNode::Leaf { bounds, .. } => *bounds,
            RTreeNode::Internal { bounds, .. } => *bounds,
        }
    }

    /// Returns the total number of entries.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns true if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Finds all entries whose rectangle intersects the query rectangle.
    pub fn query_rect(&self, query: &RTreeRect) -> Vec<u64> {
        let mut results = Vec::new();
        if let Some(root) = &self.root {
            Self::query_rect_node(root, query, &mut results);
        }
        results
    }

    fn query_rect_node(node: &RTreeNode, query: &RTreeRect, results: &mut Vec<u64>) {
        match node {
            RTreeNode::Leaf { bounds, items } => {
                if !bounds.intersects(query) {
                    return;
                }
                for item in items {
                    if item.rect.intersects(query) {
                        results.push(item.data);
                    }
                }
            }
            RTreeNode::Internal { bounds, children } => {
                if !bounds.intersects(query) {
                    return;
                }
                for child in children {
                    Self::query_rect_node(child, query, results);
                }
            }
        }
    }

    /// Finds all entries whose rectangle contains the query point.
    pub fn query_point(&self, point: Vec2) -> Vec<u64> {
        let mut results = Vec::new();
        if let Some(root) = &self.root {
            Self::query_point_node(root, point, &mut results);
        }
        results
    }

    fn query_point_node(node: &RTreeNode, point: Vec2, results: &mut Vec<u64>) {
        match node {
            RTreeNode::Leaf { bounds, items } => {
                if !bounds.contains_point(point) {
                    return;
                }
                for item in items {
                    if item.rect.contains_point(point) {
                        results.push(item.data);
                    }
                }
            }
            RTreeNode::Internal { bounds, children } => {
                if !bounds.contains_point(point) {
                    return;
                }
                for child in children {
                    Self::query_point_node(child, point, results);
                }
            }
        }
    }

    /// Finds the nearest entry to a query point using branch-and-bound.
    pub fn nearest(&self, point: Vec2) -> Option<(u64, f32)> {
        let mut best_dist_sq = f32::INFINITY;
        let mut best_data: Option<u64> = None;

        if let Some(root) = &self.root {
            Self::nearest_node(root, point, &mut best_dist_sq, &mut best_data);
        }

        best_data.map(|d| (d, best_dist_sq.sqrt()))
    }

    fn nearest_node(
        node: &RTreeNode,
        point: Vec2,
        best_dist_sq: &mut f32,
        best_data: &mut Option<u64>,
    ) {
        match node {
            RTreeNode::Leaf { bounds, items } => {
                if Self::min_dist_sq_to_rect(point, bounds) > *best_dist_sq {
                    return;
                }
                for item in items {
                    let dist_sq = Self::min_dist_sq_to_rect(point, &item.rect);
                    if dist_sq < *best_dist_sq {
                        *best_dist_sq = dist_sq;
                        *best_data = Some(item.data);
                    }
                }
            }
            RTreeNode::Internal { bounds, children } => {
                if Self::min_dist_sq_to_rect(point, bounds) > *best_dist_sq {
                    return;
                }
                // Sort children by minimum distance to prioritize closer ones.
                let mut child_dists: Vec<(usize, f32)> = children
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, Self::min_dist_sq_to_rect(point, &Self::node_bounds(c))))
                    .collect();
                child_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                for (i, _) in child_dists {
                    Self::nearest_node(&children[i], point, best_dist_sq, best_data);
                }
            }
        }
    }

    fn min_dist_sq_to_rect(point: Vec2, rect: &RTreeRect) -> f32 {
        let cx = point.x.clamp(rect.min.x, rect.max.x);
        let cy = point.y.clamp(rect.min.y, rect.max.y);
        let dx = point.x - cx;
        let dy = point.y - cy;
        dx * dx + dy * dy
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec2, Vec3};

    // --- KDTree tests -------------------------------------------------------

    #[test]
    fn test_kdtree_build_and_nearest() {
        let points: Vec<(Vec3, u64)> = vec![
            (Vec3::new(1.0, 0.0, 0.0), 1),
            (Vec3::new(0.0, 1.0, 0.0), 2),
            (Vec3::new(0.0, 0.0, 1.0), 3),
            (Vec3::new(1.0, 1.0, 1.0), 4),
        ];
        let tree = KDTree::build(&points);
        assert_eq!(tree.len(), 4);

        let nearest = tree.nearest(Vec3::new(0.9, 0.1, 0.0)).unwrap();
        assert_eq!(nearest.data, 1);
    }

    #[test]
    fn test_kdtree_nearest_backtracking() {
        // Place points so the nearest neighbor is across the split plane,
        // forcing the backtracking logic to engage.
        let points: Vec<(Vec3, u64)> = vec![
            (Vec3::new(0.0, 0.0, 0.0), 1),
            (Vec3::new(10.0, 0.0, 0.0), 2),
            (Vec3::new(5.1, 0.0, 0.0), 3), // close to query
            (Vec3::new(4.9, 0.0, 0.0), 4), // closest
        ];
        let tree = KDTree::build(&points);
        let nearest = tree.nearest(Vec3::new(5.0, 0.0, 0.0)).unwrap();
        // Should be either 3 or 4 (both are 0.1 away).
        assert!(nearest.data == 3 || nearest.data == 4);
        assert!(nearest.distance_sq < 0.02);
    }

    #[test]
    fn test_kdtree_knn() {
        let points: Vec<(Vec3, u64)> = vec![
            (Vec3::new(0.0, 0.0, 0.0), 1),
            (Vec3::new(1.0, 0.0, 0.0), 2),
            (Vec3::new(2.0, 0.0, 0.0), 3),
            (Vec3::new(3.0, 0.0, 0.0), 4),
            (Vec3::new(4.0, 0.0, 0.0), 5),
        ];
        let tree = KDTree::build(&points);
        let results = tree.k_nearest(Vec3::new(0.5, 0.0, 0.0), 3);
        assert_eq!(results.len(), 3);
        // Nearest should be 1 or 2.
        assert!(results[0].data == 1 || results[0].data == 2);
    }

    #[test]
    fn test_kdtree_range_query() {
        let points: Vec<(Vec3, u64)> = vec![
            (Vec3::new(0.0, 0.0, 0.0), 1),
            (Vec3::new(1.0, 0.0, 0.0), 2),
            (Vec3::new(5.0, 0.0, 0.0), 3),
            (Vec3::new(10.0, 0.0, 0.0), 4),
        ];
        let tree = KDTree::build(&points);
        let results = tree.range_query(Vec3::ZERO, 2.0);
        assert_eq!(results.len(), 2);
    }

    // --- Octree tests -------------------------------------------------------

    #[test]
    fn test_octree_insert_and_query() {
        let bounds = AABB::new(Vec3::splat(-100.0), Vec3::splat(100.0));
        let mut octree = Octree::new(bounds);

        for i in 0..20 {
            let pos = Vec3::new(i as f32, 0.0, 0.0);
            let aabb = AABB::new(pos - Vec3::splat(0.5), pos + Vec3::splat(0.5));
            octree.insert(aabb, i as u64);
        }

        let query = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(3.0, 1.0, 1.0));
        let results = octree.query_aabb(&query);
        assert!(results.len() >= 3); // Should find items 0, 1, 2, 3
    }

    #[test]
    fn test_octree_remove() {
        let bounds = AABB::new(Vec3::splat(-10.0), Vec3::splat(10.0));
        let mut octree = Octree::new(bounds);

        let aabb = AABB::new(Vec3::ZERO, Vec3::ONE);
        octree.insert(aabb, 42);
        assert!(octree.remove(42));
        assert!(!octree.remove(42));
    }

    #[test]
    fn test_octree_sphere_query() {
        let bounds = AABB::new(Vec3::splat(-50.0), Vec3::splat(50.0));
        let mut octree = Octree::new(bounds);

        for i in 0..10 {
            let pos = Vec3::new(i as f32 * 2.0, 0.0, 0.0);
            let aabb = AABB::new(pos - Vec3::splat(0.5), pos + Vec3::splat(0.5));
            octree.insert(aabb, i as u64);
        }

        let results = octree.query_sphere(Vec3::ZERO, 3.0);
        assert!(!results.is_empty());
    }

    // --- BVH tests ----------------------------------------------------------

    #[test]
    fn test_bvh_build_and_query() {
        let items: Vec<BVHItem> = (0..100)
            .map(|i| {
                let x = (i % 10) as f32 * 2.0;
                let y = (i / 10) as f32 * 2.0;
                BVHItem {
                    aabb: AABB::new(
                        Vec3::new(x, y, 0.0),
                        Vec3::new(x + 1.0, y + 1.0, 1.0),
                    ),
                    data: i as u64,
                }
            })
            .collect();

        let bvh = BVH::build(&items);
        let query = AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(3.0, 3.0, 1.0));
        let results = bvh.query_aabb(&query);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_bvh_point_query() {
        let items = vec![
            BVHItem {
                aabb: AABB::new(Vec3::ZERO, Vec3::ONE),
                data: 1,
            },
            BVHItem {
                aabb: AABB::new(Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0)),
                data: 2,
            },
        ];
        let bvh = BVH::build(&items);
        let results = bvh.query_point(Vec3::new(0.5, 0.5, 0.5));
        assert!(results.contains(&1));
        assert!(!results.contains(&2));
    }

    #[test]
    fn test_bvh_ray_query() {
        let items = vec![
            BVHItem {
                aabb: AABB::new(Vec3::new(5.0, -1.0, -1.0), Vec3::new(6.0, 1.0, 1.0)),
                data: 1,
            },
            BVHItem {
                aabb: AABB::new(Vec3::new(10.0, -1.0, -1.0), Vec3::new(11.0, 1.0, 1.0)),
                data: 2,
            },
        ];
        let bvh = BVH::build(&items);
        let ray = Ray::new(Vec3::ZERO, Vec3::X);
        let results = bvh.query_ray(&ray);
        assert_eq!(results.len(), 2);
        // First hit should be data=1 (closer).
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_bvh_refit() {
        let items = vec![
            BVHItem {
                aabb: AABB::new(Vec3::ZERO, Vec3::ONE),
                data: 1,
            },
        ];
        let mut bvh = BVH::build(&items);

        // Refit with a moved AABB.
        bvh.refit(&|_data| AABB::new(Vec3::new(10.0, 10.0, 10.0), Vec3::new(11.0, 11.0, 11.0)));

        // Old position should not find anything.
        let results = bvh.query_point(Vec3::new(0.5, 0.5, 0.5));
        assert!(results.is_empty());

        // New position should.
        let results = bvh.query_point(Vec3::new(10.5, 10.5, 10.5));
        assert!(results.contains(&1));
    }

    // --- SpatialHashGrid tests ----------------------------------------------

    #[test]
    fn test_spatial_hash_insert_query() {
        let mut grid = SpatialHashGrid::new(2.0);
        grid.insert(Vec3::new(0.0, 0.0, 0.0), 0.5, 1);
        grid.insert(Vec3::new(1.0, 0.0, 0.0), 0.5, 2);
        grid.insert(Vec3::new(10.0, 0.0, 0.0), 0.5, 3);

        let results = grid.query_radius(Vec3::ZERO, 2.0);
        assert!(results.contains(&1));
        assert!(results.contains(&2));
        assert!(!results.contains(&3));
    }

    #[test]
    fn test_spatial_hash_remove() {
        let mut grid = SpatialHashGrid::new(2.0);
        grid.insert(Vec3::ZERO, 0.5, 42);
        assert!(grid.remove(42));

        let results = grid.query_radius(Vec3::ZERO, 1.0);
        assert!(!results.contains(&42));
    }

    #[test]
    fn test_spatial_hash_update() {
        let mut grid = SpatialHashGrid::new(2.0);
        grid.insert(Vec3::ZERO, 0.5, 1);

        // Move far away.
        grid.update(1, Vec3::new(100.0, 0.0, 0.0), 0.5);

        let results = grid.query_radius(Vec3::ZERO, 2.0);
        assert!(!results.contains(&1));

        let results = grid.query_radius(Vec3::new(100.0, 0.0, 0.0), 2.0);
        assert!(results.contains(&1));
    }

    // --- RTree tests --------------------------------------------------------

    #[test]
    fn test_rtree_bulk_load_and_query() {
        let entries: Vec<(RTreeRect, u64)> = (0..100)
            .map(|i| {
                let x = (i % 10) as f32 * 2.0;
                let y = (i / 10) as f32 * 2.0;
                (
                    RTreeRect::new(Vec2::new(x, y), Vec2::new(x + 1.0, y + 1.0)),
                    i as u64,
                )
            })
            .collect();

        let tree = RTree::bulk_load(&entries);
        assert_eq!(tree.len(), 100);

        let query = RTreeRect::new(Vec2::new(0.0, 0.0), Vec2::new(3.0, 3.0));
        let results = tree.query_rect(&query);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_rtree_point_query() {
        let entries = vec![
            (
                RTreeRect::new(Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0)),
                1u64,
            ),
            (
                RTreeRect::new(Vec2::new(5.0, 5.0), Vec2::new(7.0, 7.0)),
                2,
            ),
        ];
        let tree = RTree::bulk_load(&entries);
        let results = tree.query_point(Vec2::new(1.0, 1.0));
        assert!(results.contains(&1));
        assert!(!results.contains(&2));
    }

    #[test]
    fn test_rtree_nearest() {
        let entries = vec![
            (
                RTreeRect::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0)),
                1u64,
            ),
            (
                RTreeRect::new(Vec2::new(10.0, 10.0), Vec2::new(11.0, 11.0)),
                2,
            ),
        ];
        let tree = RTree::bulk_load(&entries);
        let result = tree.nearest(Vec2::new(0.5, 0.5));
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, 1);
    }

    #[test]
    fn test_rtree_empty() {
        let tree = RTree::bulk_load(&[]);
        assert!(tree.is_empty());
        assert_eq!(tree.query_rect(&RTreeRect::new(Vec2::ZERO, Vec2::ONE)).len(), 0);
    }
}
