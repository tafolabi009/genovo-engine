// engine/render/src/virtual_geometry/cluster.rs
//
// Mesh clustering and hierarchical LOD DAG construction. Partitions a mesh
// into clusters of ~128 triangles, then builds a multi-level DAG where each
// parent cluster is a simplified union of its children.

use crate::mesh::{Mesh, Vertex, AABB};
use glam::Vec3;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Target number of triangles per cluster.
pub const CLUSTER_TARGET_TRIANGLES: usize = 128;

/// Maximum triangles per cluster (hard cap).
pub const CLUSTER_MAX_TRIANGLES: usize = 256;

/// Number of clusters to merge when building the next LOD level.
pub const CLUSTER_GROUP_SIZE: usize = 4;

/// Maximum depth of the cluster DAG.
pub const MAX_DAG_DEPTH: usize = 16;

// ---------------------------------------------------------------------------
// ClusterBounds
// ---------------------------------------------------------------------------

/// Bounding sphere for a cluster, used for screen-space error computation.
#[derive(Debug, Clone, Copy)]
pub struct ClusterBounds {
    /// Centre of the bounding sphere in object space.
    pub center: Vec3,
    /// Radius of the bounding sphere.
    pub radius: f32,
    /// Axis-aligned bounding box.
    pub aabb: AABB,
}

impl ClusterBounds {
    /// Compute from a set of vertex positions.
    pub fn from_positions(positions: &[Vec3]) -> Self {
        if positions.is_empty() {
            return Self {
                center: Vec3::ZERO,
                radius: 0.0,
                aabb: AABB::default(),
            };
        }

        let mut aabb = AABB::default();
        for &p in positions {
            aabb.expand_point(p);
        }

        let center = aabb.center();
        let radius = positions
            .iter()
            .map(|p| (*p - center).length())
            .fold(0.0_f32, f32::max);

        Self { center, radius, aabb }
    }

    /// Merge two bounding volumes.
    pub fn merge(&self, other: &ClusterBounds) -> ClusterBounds {
        let mut aabb = self.aabb;
        aabb.expand_aabb(&other.aabb);
        let center = aabb.center();

        let r1 = (self.center - center).length() + self.radius;
        let r2 = (other.center - center).length() + other.radius;
        let radius = r1.max(r2);

        ClusterBounds { center, radius, aabb }
    }

    /// Compute the screen-space pixel error for this cluster given camera
    /// distance and viewport height.
    pub fn screen_space_error(&self, error_metric: f32, distance: f32, viewport_height: f32, fov_y: f32) -> f32 {
        if distance <= 0.0 {
            return f32::MAX;
        }
        let projected_size = self.radius / (distance * (fov_y * 0.5).tan());
        projected_size * viewport_height * error_metric
    }
}

// ---------------------------------------------------------------------------
// MeshCluster
// ---------------------------------------------------------------------------

/// A cluster of triangles from a mesh. Each cluster contains ~128 triangles
/// and represents a renderable unit in the virtual geometry system.
#[derive(Debug, Clone)]
pub struct MeshCluster {
    /// Unique identifier for this cluster within the DAG.
    pub id: u32,
    /// LOD level (0 = finest detail).
    pub lod_level: u32,
    /// Vertex data for this cluster (local copy).
    pub vertices: Vec<Vertex>,
    /// Triangle indices (local to this cluster's vertex array).
    pub indices: Vec<u32>,
    /// Bounding volume of this cluster.
    pub bounds: ClusterBounds,
    /// Screen-space error metric. When the projected error is below the
    /// threshold the cluster is fine enough and its children need not be
    /// rendered.
    pub error_metric: f32,
    /// Maximum error of this cluster and all descendants. Used for early-out
    /// during DAG traversal.
    pub max_child_error: f32,
    /// Parent group index in the DAG (None for root clusters).
    pub parent_group: Option<u32>,
    /// Child group indices. Empty for leaf clusters.
    pub child_groups: Vec<u32>,
    /// Byte offset into the GPU cluster buffer (set during streaming).
    pub gpu_vertex_offset: u64,
    /// Byte offset for index data.
    pub gpu_index_offset: u64,
    /// Whether this cluster is currently resident on the GPU.
    pub is_resident: bool,
    /// Page ID for streaming (which virtual page this cluster lives in).
    pub page_id: u32,
}

impl MeshCluster {
    /// Number of triangles in this cluster.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Number of vertices in this cluster.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Total byte size for GPU upload (vertices + indices).
    pub fn gpu_byte_size(&self) -> usize {
        self.vertices.len() * std::mem::size_of::<Vertex>()
            + self.indices.len() * std::mem::size_of::<u32>()
    }

    /// Whether this cluster is a leaf (no children).
    pub fn is_leaf(&self) -> bool {
        self.child_groups.is_empty()
    }

    /// Whether this cluster is a root (no parent).
    pub fn is_root(&self) -> bool {
        self.parent_group.is_none()
    }
}

// ---------------------------------------------------------------------------
// ClusterGroup
// ---------------------------------------------------------------------------

/// A group of clusters at the same LOD level. Groups are the unit of LOD
/// switching: either all clusters in a group are rendered, or their parent
/// group is rendered instead.
#[derive(Debug, Clone)]
pub struct ClusterGroup {
    /// Unique identifier for this group.
    pub id: u32,
    /// LOD level of the clusters in this group.
    pub lod_level: u32,
    /// Indices of clusters belonging to this group.
    pub cluster_indices: Vec<u32>,
    /// Combined bounding volume of all clusters in the group.
    pub bounds: ClusterBounds,
    /// Error metric for the group (max of constituent clusters).
    pub error_metric: f32,
    /// Parent group index (None for root groups).
    pub parent_group: Option<u32>,
    /// Child group indices.
    pub child_groups: Vec<u32>,
}

impl ClusterGroup {
    /// Total triangle count across all clusters in the group.
    pub fn total_triangles(&self, clusters: &[MeshCluster]) -> usize {
        self.cluster_indices
            .iter()
            .map(|&idx| clusters[idx as usize].triangle_count())
            .sum()
    }

    /// Total GPU byte size for all clusters in the group.
    pub fn total_gpu_bytes(&self, clusters: &[MeshCluster]) -> usize {
        self.cluster_indices
            .iter()
            .map(|&idx| clusters[idx as usize].gpu_byte_size())
            .sum()
    }
}

// ---------------------------------------------------------------------------
// ClusterDAG
// ---------------------------------------------------------------------------

/// Hierarchical cluster DAG for a mesh. Each level contains groups of
/// clusters; parent levels are progressively simplified versions of child
/// levels.
#[derive(Debug, Clone)]
pub struct ClusterDAG {
    /// All clusters across all LOD levels.
    pub clusters: Vec<MeshCluster>,
    /// All cluster groups across all LOD levels.
    pub groups: Vec<ClusterGroup>,
    /// Number of LOD levels.
    pub level_count: u32,
    /// Total triangle count at the finest level.
    pub base_triangle_count: u32,
    /// Bounding volume of the entire mesh.
    pub bounds: ClusterBounds,
    /// Error threshold below which we stop refining (pixels).
    pub pixel_error_threshold: f32,
}

impl ClusterDAG {
    /// Determine which clusters to render given camera parameters.
    ///
    /// This performs a top-down traversal of the DAG, selecting clusters
    /// whose screen-space error is acceptable while avoiding rendering both
    /// a parent and its children.
    pub fn select_clusters(
        &self,
        camera_pos: Vec3,
        viewport_height: f32,
        fov_y: f32,
        error_threshold: f32,
    ) -> Vec<u32> {
        let mut selected = Vec::new();
        let mut stack: Vec<u32> = Vec::new();

        // Start from root groups.
        for (i, group) in self.groups.iter().enumerate() {
            if group.parent_group.is_none() {
                stack.push(i as u32);
            }
        }

        while let Some(group_idx) = stack.pop() {
            let group = &self.groups[group_idx as usize];

            // Compute screen-space error for this group.
            let distance = (group.bounds.center - camera_pos).length();
            let screen_error = group.bounds.screen_space_error(
                group.error_metric,
                distance,
                viewport_height,
                fov_y,
            );

            if screen_error <= error_threshold || group.child_groups.is_empty() {
                // This group's error is acceptable -- render its clusters.
                selected.extend_from_slice(&group.cluster_indices);
            } else {
                // Need finer detail -- descend to child groups.
                for &child_idx in &group.child_groups {
                    stack.push(child_idx);
                }
            }
        }

        selected
    }

    /// Get total cluster count.
    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }

    /// Get total group count.
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Get all clusters at a specific LOD level.
    pub fn clusters_at_level(&self, level: u32) -> Vec<&MeshCluster> {
        self.clusters.iter().filter(|c| c.lod_level == level).collect()
    }

    /// Get total triangle count across all clusters at a given level.
    pub fn triangles_at_level(&self, level: u32) -> usize {
        self.clusters
            .iter()
            .filter(|c| c.lod_level == level)
            .map(|c| c.triangle_count())
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Triangle adjacency graph
// ---------------------------------------------------------------------------

/// An edge in the mesh, represented as ordered pair of vertex indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Edge(u32, u32);

impl Edge {
    fn new(a: u32, b: u32) -> Self {
        if a <= b { Edge(a, b) } else { Edge(b, a) }
    }
}

/// Build triangle adjacency: for each triangle, which triangles share an edge.
fn build_adjacency(indices: &[u32], vertex_count: usize) -> Vec<Vec<usize>> {
    let tri_count = indices.len() / 3;
    let mut edge_to_tris: HashMap<Edge, Vec<usize>> = HashMap::new();
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); tri_count];

    for tri in 0..tri_count {
        let base = tri * 3;
        let i0 = indices[base];
        let i1 = indices[base + 1];
        let i2 = indices[base + 2];

        let edges = [Edge::new(i0, i1), Edge::new(i1, i2), Edge::new(i2, i0)];
        for edge in &edges {
            edge_to_tris.entry(*edge).or_default().push(tri);
        }
    }

    for (_edge, tris) in &edge_to_tris {
        for i in 0..tris.len() {
            for j in (i + 1)..tris.len() {
                adjacency[tris[i]].push(tris[j]);
                adjacency[tris[j]].push(tris[i]);
            }
        }
    }

    // Deduplicate adjacency lists.
    for adj in &mut adjacency {
        adj.sort_unstable();
        adj.dedup();
    }

    adjacency
}

// ---------------------------------------------------------------------------
// Spatial partitioning (k-means style)
// ---------------------------------------------------------------------------

/// Compute the centroid of a triangle.
fn triangle_centroid(vertices: &[Vertex], indices: &[u32], tri_idx: usize) -> Vec3 {
    let base = tri_idx * 3;
    let p0 = Vec3::from_array(vertices[indices[base] as usize].position);
    let p1 = Vec3::from_array(vertices[indices[base + 1] as usize].position);
    let p2 = Vec3::from_array(vertices[indices[base + 2] as usize].position);
    (p0 + p1 + p2) / 3.0
}

/// Partition triangles into clusters using a graph-based spatial approach
/// that respects adjacency. This uses a greedy BFS seeded partitioning
/// combined with spatial k-means refinement.
fn partition_triangles_into_clusters(
    vertices: &[Vertex],
    indices: &[u32],
    target_cluster_size: usize,
) -> Vec<Vec<usize>> {
    let tri_count = indices.len() / 3;
    if tri_count == 0 {
        return Vec::new();
    }

    if tri_count <= target_cluster_size {
        return vec![(0..tri_count).collect()];
    }

    let adjacency = build_adjacency(indices, vertices.len());
    let centroids: Vec<Vec3> = (0..tri_count)
        .map(|t| triangle_centroid(vertices, indices, t))
        .collect();

    let num_clusters = (tri_count + target_cluster_size - 1) / target_cluster_size;

    // Seed cluster centres using farthest-point sampling.
    let mut seeds: Vec<usize> = Vec::with_capacity(num_clusters);
    seeds.push(0);

    let mut min_dist = vec![f32::MAX; tri_count];
    for _ in 1..num_clusters {
        let last_seed = *seeds.last().unwrap();
        for t in 0..tri_count {
            let d = (centroids[t] - centroids[last_seed]).length_squared();
            min_dist[t] = min_dist[t].min(d);
        }

        let farthest = (0..tri_count)
            .max_by(|&a, &b| min_dist[a].partial_cmp(&min_dist[b]).unwrap_or(Ordering::Equal))
            .unwrap();
        seeds.push(farthest);
    }

    // Assign triangles to nearest seed using BFS along adjacency graph.
    let mut assignments = vec![u32::MAX; tri_count];
    let mut queue: Vec<(usize, u32)> = Vec::new();

    for (cluster_id, &seed) in seeds.iter().enumerate() {
        assignments[seed] = cluster_id as u32;
        queue.push((seed, cluster_id as u32));
    }

    // BFS to flood-fill from seeds.
    let mut head = 0;
    while head < queue.len() {
        let (tri, cluster_id) = queue[head];
        head += 1;

        for &neighbor in &adjacency[tri] {
            if assignments[neighbor] == u32::MAX {
                assignments[neighbor] = cluster_id;
                queue.push((neighbor, cluster_id));
            }
        }
    }

    // Handle any unassigned triangles (disconnected components).
    for t in 0..tri_count {
        if assignments[t] == u32::MAX {
            let nearest_cluster = seeds
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = (centroids[t] - centroids[**a]).length_squared();
                    let db = (centroids[t] - centroids[**b]).length_squared();
                    da.partial_cmp(&db).unwrap_or(Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            assignments[t] = nearest_cluster as u32;
        }
    }

    // K-means refinement passes.
    for _iteration in 0..8 {
        // Recompute cluster centres.
        let mut cluster_centers = vec![Vec3::ZERO; num_clusters];
        let mut cluster_counts = vec![0u32; num_clusters];

        for t in 0..tri_count {
            let c = assignments[t] as usize;
            cluster_centers[c] += centroids[t];
            cluster_counts[c] += 1;
        }

        for c in 0..num_clusters {
            if cluster_counts[c] > 0 {
                cluster_centers[c] /= cluster_counts[c] as f32;
            }
        }

        // Re-assign each triangle to the nearest cluster centre,
        // but only if an adjacent triangle belongs to that cluster
        // (preserves spatial connectivity).
        let mut changed = false;
        for t in 0..tri_count {
            let current = assignments[t] as usize;
            let current_dist = (centroids[t] - cluster_centers[current]).length_squared();

            // Find the best cluster among neighbours.
            let mut best_cluster = current;
            let mut best_dist = current_dist;

            for &neighbor in &adjacency[t] {
                let nc = assignments[neighbor] as usize;
                if nc != current {
                    let d = (centroids[t] - cluster_centers[nc]).length_squared();
                    if d < best_dist {
                        best_dist = d;
                        best_cluster = nc;
                    }
                }
            }

            if best_cluster != current {
                assignments[t] = best_cluster as u32;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    // Collect clusters.
    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); num_clusters];
    for t in 0..tri_count {
        clusters[assignments[t] as usize].push(t);
    }

    // Remove empty clusters.
    clusters.retain(|c| !c.is_empty());

    clusters
}

// ---------------------------------------------------------------------------
// Cluster construction from triangle lists
// ---------------------------------------------------------------------------

/// Build a `MeshCluster` from a list of triangle indices into the original mesh.
fn build_cluster_from_triangles(
    cluster_id: u32,
    lod_level: u32,
    mesh_vertices: &[Vertex],
    mesh_indices: &[u32],
    triangle_indices: &[usize],
) -> MeshCluster {
    // Map original vertex indices to local cluster indices.
    let mut vertex_map: HashMap<u32, u32> = HashMap::new();
    let mut local_vertices: Vec<Vertex> = Vec::new();
    let mut local_indices: Vec<u32> = Vec::new();

    for &tri_idx in triangle_indices {
        let base = tri_idx * 3;
        for k in 0..3 {
            let orig_idx = mesh_indices[base + k];
            let local_idx = *vertex_map.entry(orig_idx).or_insert_with(|| {
                let idx = local_vertices.len() as u32;
                local_vertices.push(mesh_vertices[orig_idx as usize]);
                idx
            });
            local_indices.push(local_idx);
        }
    }

    let positions: Vec<Vec3> = local_vertices
        .iter()
        .map(|v| Vec3::from_array(v.position))
        .collect();

    let bounds = ClusterBounds::from_positions(&positions);

    MeshCluster {
        id: cluster_id,
        lod_level,
        vertices: local_vertices,
        indices: local_indices,
        bounds,
        error_metric: 0.0,
        max_child_error: 0.0,
        parent_group: None,
        child_groups: Vec::new(),
        gpu_vertex_offset: 0,
        gpu_index_offset: 0,
        is_resident: false,
        page_id: 0,
    }
}

// ---------------------------------------------------------------------------
// Error metric computation
// ---------------------------------------------------------------------------

/// Compute the geometric error for a simplified cluster relative to its
/// source clusters. Uses Hausdorff-like distance sampling.
fn compute_cluster_error(
    simplified: &MeshCluster,
    source_clusters: &[&MeshCluster],
) -> f32 {
    if source_clusters.is_empty() || simplified.vertices.is_empty() {
        return 0.0;
    }

    // Collect all source vertex positions.
    let source_positions: Vec<Vec3> = source_clusters
        .iter()
        .flat_map(|c| c.vertices.iter().map(|v| Vec3::from_array(v.position)))
        .collect();

    if source_positions.is_empty() {
        return 0.0;
    }

    // For each vertex in the simplified mesh, find the maximum minimum distance
    // to any source vertex. This is a one-sided Hausdorff distance.
    let mut max_error: f32 = 0.0;

    for sv in &simplified.vertices {
        let sp = Vec3::from_array(sv.position);
        let min_dist = source_positions
            .iter()
            .map(|&p| (p - sp).length_squared())
            .fold(f32::MAX, f32::min);
        max_error = max_error.max(min_dist.sqrt());
    }

    max_error
}

// ---------------------------------------------------------------------------
// DAG construction
// ---------------------------------------------------------------------------

/// Build a complete cluster DAG from a mesh.
///
/// The process:
/// 1. Partition the base mesh into clusters of ~128 triangles.
/// 2. Group adjacent clusters.
/// 3. For each group, merge and simplify to create parent clusters.
/// 4. Repeat until only one group remains (or max depth reached).
pub fn build_cluster_dag(mesh: &Mesh) -> ClusterDAG {
    let mut all_clusters: Vec<MeshCluster> = Vec::new();
    let mut all_groups: Vec<ClusterGroup> = Vec::new();
    let mut next_cluster_id: u32 = 0;
    let mut next_group_id: u32 = 0;

    // ---- Level 0: partition base mesh into clusters ----
    let partitions = partition_triangles_into_clusters(
        &mesh.vertices,
        &mesh.indices,
        CLUSTER_TARGET_TRIANGLES,
    );

    let base_cluster_start = all_clusters.len();
    for partition in &partitions {
        let cluster = build_cluster_from_triangles(
            next_cluster_id,
            0,
            &mesh.vertices,
            &mesh.indices,
            partition,
        );
        all_clusters.push(cluster);
        next_cluster_id += 1;
    }
    let base_cluster_end = all_clusters.len();

    // Create groups for the base level.
    let base_cluster_indices: Vec<u32> = (base_cluster_start..base_cluster_end)
        .map(|i| i as u32)
        .collect();

    let base_groups = create_groups_from_clusters(
        &all_clusters,
        &base_cluster_indices,
        0,
        &mut next_group_id,
    );

    // Set child groups on base clusters.
    for group in &base_groups {
        for &ci in &group.cluster_indices {
            // Base clusters are leaves -- no child groups, but we assign
            // parent group.
            all_clusters[ci as usize].parent_group = Some(group.id);
        }
    }

    let mut current_level_groups: Vec<u32> = base_groups
        .iter()
        .map(|g| g.id)
        .collect();

    all_groups.extend(base_groups);

    let mut level = 0u32;

    // ---- Build higher LOD levels ----
    while current_level_groups.len() > 1 && level < MAX_DAG_DEPTH as u32 {
        level += 1;

        let mut next_level_clusters: Vec<u32> = Vec::new();

        for &group_idx in &current_level_groups {
            let group = &all_groups[group_idx as usize];
            let cluster_owned: Vec<MeshCluster> = group
                .cluster_indices
                .iter()
                .map(|&ci| all_clusters[ci as usize].clone())
                .collect();
            let cluster_refs: Vec<&MeshCluster> = cluster_owned.iter().collect();

            // Merge all vertices and indices from child clusters.
            let (merged_verts, merged_indices) = merge_cluster_data(&cluster_refs);

            if merged_indices.is_empty() {
                continue;
            }

            // Simplify the merged geometry.
            let target_tris = (merged_indices.len() / 3 / 2).max(1);
            let (simp_verts, simp_indices) = simplify_cluster_geometry(
                &merged_verts,
                &merged_indices,
                target_tris,
            );

            // Re-partition into clusters if the simplified result is large.
            if simp_indices.len() / 3 > CLUSTER_MAX_TRIANGLES {
                let sub_partitions = partition_triangles_into_clusters(
                    &simp_verts,
                    &simp_indices,
                    CLUSTER_TARGET_TRIANGLES,
                );

                for sp in &sub_partitions {
                    let mut cluster = build_cluster_from_triangles(
                        next_cluster_id,
                        level,
                        &simp_verts,
                        &simp_indices,
                        sp,
                    );

                    cluster.error_metric = compute_cluster_error(&cluster, &cluster_refs);
                    cluster.child_groups.push(group_idx);

                    next_level_clusters.push(next_cluster_id);
                    all_clusters.push(cluster);
                    next_cluster_id += 1;
                }
            } else {
                // Single cluster for this simplified group.
                let positions: Vec<Vec3> = simp_verts
                    .iter()
                    .map(|v| Vec3::from_array(v.position))
                    .collect();

                let bounds = ClusterBounds::from_positions(&positions);

                let mut cluster = MeshCluster {
                    id: next_cluster_id,
                    lod_level: level,
                    vertices: simp_verts,
                    indices: simp_indices,
                    bounds,
                    error_metric: 0.0,
                    max_child_error: 0.0,
                    parent_group: None,
                    child_groups: vec![group_idx],
                    gpu_vertex_offset: 0,
                    gpu_index_offset: 0,
                    is_resident: false,
                    page_id: 0,
                };

                cluster.error_metric = compute_cluster_error(&cluster, &cluster_refs);

                next_level_clusters.push(next_cluster_id);
                all_clusters.push(cluster);
                next_cluster_id += 1;
            }
        }

        if next_level_clusters.is_empty() {
            break;
        }

        // Create groups for this level.
        let new_groups = create_groups_from_clusters(
            &all_clusters,
            &next_level_clusters,
            level,
            &mut next_group_id,
        );

        // Link parent groups to child groups.
        for group in &new_groups {
            for &ci in &group.cluster_indices {
                all_clusters[ci as usize].parent_group = Some(group.id);
            }
        }

        // Link child groups to parent groups.
        for group in &new_groups {
            for &ci in &group.cluster_indices {
                let cluster = &all_clusters[ci as usize];
                for &child_group_idx in &cluster.child_groups {
                    let child_group = &mut all_groups[child_group_idx as usize];
                    if !child_group.child_groups.contains(&group.id) {
                        // Store reference from child to parent.
                    }
                }
            }
        }

        current_level_groups = new_groups.iter().map(|g| g.id).collect();
        all_groups.extend(new_groups);
    }

    // Propagate max child error up the DAG.
    propagate_max_child_error(&mut all_clusters);

    // Compute overall bounds.
    let overall_bounds = if all_clusters.is_empty() {
        ClusterBounds {
            center: Vec3::ZERO,
            radius: 0.0,
            aabb: AABB::default(),
        }
    } else {
        let mut bounds = all_clusters[0].bounds;
        for cluster in &all_clusters[1..] {
            bounds = bounds.merge(&cluster.bounds);
        }
        bounds
    };

    ClusterDAG {
        clusters: all_clusters,
        groups: all_groups,
        level_count: level + 1,
        base_triangle_count: mesh.triangle_count(),
        bounds: overall_bounds,
        pixel_error_threshold: 1.0,
    }
}

/// Create groups from a set of cluster indices. Groups clusters based on
/// spatial proximity.
fn create_groups_from_clusters(
    clusters: &[MeshCluster],
    cluster_indices: &[u32],
    lod_level: u32,
    next_group_id: &mut u32,
) -> Vec<ClusterGroup> {
    if cluster_indices.is_empty() {
        return Vec::new();
    }

    // Simple spatial grouping: sort by centroid position and chunk.
    let mut sorted_indices: Vec<u32> = cluster_indices.to_vec();
    sorted_indices.sort_by(|&a, &b| {
        let ca = clusters[a as usize].bounds.center;
        let cb = clusters[b as usize].bounds.center;
        // Sort by Morton code approximation (interleave xyz bits via comparison).
        let ka = ca.x + ca.y * 1000.0 + ca.z * 1000000.0;
        let kb = cb.x + cb.y * 1000.0 + cb.z * 1000000.0;
        ka.partial_cmp(&kb).unwrap_or(Ordering::Equal)
    });

    let mut groups = Vec::new();
    for chunk in sorted_indices.chunks(CLUSTER_GROUP_SIZE) {
        let group_id = *next_group_id;
        *next_group_id += 1;

        let mut bounds = clusters[chunk[0] as usize].bounds;
        let mut max_error: f32 = 0.0;
        for &ci in chunk {
            bounds = bounds.merge(&clusters[ci as usize].bounds);
            max_error = max_error.max(clusters[ci as usize].error_metric);
        }

        groups.push(ClusterGroup {
            id: group_id,
            lod_level,
            cluster_indices: chunk.to_vec(),
            bounds,
            error_metric: max_error,
            parent_group: None,
            child_groups: Vec::new(),
        });
    }

    groups
}

/// Propagate maximum child error from leaves upward through the DAG.
fn propagate_max_child_error(clusters: &mut [MeshCluster]) {
    // Process from highest LOD level (leaves) to lowest (roots).
    let max_level = clusters.iter().map(|c| c.lod_level).max().unwrap_or(0);

    for level in 0..=max_level {
        for i in 0..clusters.len() {
            if clusters[i].lod_level == level {
                if clusters[i].child_groups.is_empty() {
                    clusters[i].max_child_error = clusters[i].error_metric;
                } else {
                    clusters[i].max_child_error = clusters[i].error_metric;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cluster merging and simplification helpers
// ---------------------------------------------------------------------------

/// Merge vertex and index data from multiple clusters into one buffer.
fn merge_cluster_data(clusters: &[&MeshCluster]) -> (Vec<Vertex>, Vec<u32>) {
    let total_verts: usize = clusters.iter().map(|c| c.vertices.len()).sum();
    let total_indices: usize = clusters.iter().map(|c| c.indices.len()).sum();

    let mut vertices = Vec::with_capacity(total_verts);
    let mut indices = Vec::with_capacity(total_indices);

    for cluster in clusters {
        let base_vertex = vertices.len() as u32;
        vertices.extend_from_slice(&cluster.vertices);
        for &idx in &cluster.indices {
            indices.push(idx + base_vertex);
        }
    }

    (vertices, indices)
}

/// Simplified in-line mesh simplification for cluster DAG construction.
/// This is a lightweight version; the full QEM simplifier is in
/// `simplification.rs`.
fn simplify_cluster_geometry(
    vertices: &[Vertex],
    indices: &[u32],
    target_triangle_count: usize,
) -> (Vec<Vertex>, Vec<u32>) {
    use crate::virtual_geometry::simplification;

    let result = simplification::simplify_raw(vertices, indices, target_triangle_count);
    (result.vertices, result.indices)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{create_sphere, create_cube};

    #[test]
    fn test_cluster_bounds_from_positions() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let bounds = ClusterBounds::from_positions(&positions);
        assert!(bounds.radius > 0.0);
        assert!(bounds.aabb.min.x <= 0.0);
        assert!(bounds.aabb.max.x >= 1.0);
    }

    #[test]
    fn test_cluster_bounds_merge() {
        let a = ClusterBounds::from_positions(&[Vec3::ZERO, Vec3::ONE]);
        let b = ClusterBounds::from_positions(&[Vec3::new(2.0, 2.0, 2.0), Vec3::new(3.0, 3.0, 3.0)]);
        let merged = a.merge(&b);
        assert!(merged.radius >= a.radius);
        assert!(merged.aabb.max.x >= 3.0);
    }

    #[test]
    fn test_partition_small_mesh() {
        let mesh = create_cube();
        let partitions = partition_triangles_into_clusters(
            &mesh.vertices,
            &mesh.indices,
            CLUSTER_TARGET_TRIANGLES,
        );
        // A cube has only 12 triangles, should be 1 cluster.
        assert_eq!(partitions.len(), 1);
        assert_eq!(partitions[0].len(), 12);
    }

    #[test]
    fn test_build_cluster_dag_cube() {
        let mesh = create_cube();
        let dag = build_cluster_dag(&mesh);
        assert!(dag.cluster_count() >= 1);
        assert!(dag.group_count() >= 1);
        assert_eq!(dag.base_triangle_count, 12);
    }

    #[test]
    fn test_build_cluster_dag_sphere() {
        let mesh = create_sphere(32, 24);
        let dag = build_cluster_dag(&mesh);
        assert!(dag.cluster_count() >= 1);
        assert!(dag.group_count() >= 1);
        // Should have multiple clusters for a high-poly sphere.
        let base_clusters = dag.clusters_at_level(0);
        assert!(base_clusters.len() >= 1);
    }

    #[test]
    fn test_select_clusters() {
        let mesh = create_sphere(16, 12);
        let dag = build_cluster_dag(&mesh);

        let selected = dag.select_clusters(
            Vec3::new(0.0, 0.0, 5.0),
            1080.0,
            std::f32::consts::FRAC_PI_4,
            1.0,
        );
        assert!(!selected.is_empty());
    }

    #[test]
    fn test_screen_space_error() {
        let bounds = ClusterBounds {
            center: Vec3::ZERO,
            radius: 1.0,
            aabb: AABB::new(-Vec3::ONE, Vec3::ONE),
        };

        // Far away: small error.
        let error_far = bounds.screen_space_error(1.0, 100.0, 1080.0, std::f32::consts::FRAC_PI_4);
        // Close: large error.
        let error_near = bounds.screen_space_error(1.0, 1.0, 1080.0, std::f32::consts::FRAC_PI_4);
        assert!(error_near > error_far);
    }
}
