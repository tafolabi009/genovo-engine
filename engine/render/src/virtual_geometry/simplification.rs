// engine/render/src/virtual_geometry/simplification.rs
//
// Mesh simplification via Quadric Error Metrics (QEM). Implements edge-collapse
// based mesh decimation following the Garland-Heckbert algorithm, with support
// for boundary preservation, UV seam locking, and cluster boundary constraints.

use crate::mesh::Vertex;
use glam::{Mat4, Vec2, Vec3, Vec4};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum cost for an edge collapse before it is considered invalid.
const MAX_COLLAPSE_COST: f64 = 1e18;

/// Weight multiplier for boundary edges to discourage collapsing them.
const BOUNDARY_PENALTY_WEIGHT: f64 = 100.0;

/// Weight multiplier for UV seam edges.
const UV_SEAM_PENALTY_WEIGHT: f64 = 50.0;

/// Minimum triangle quality (aspect ratio) after a collapse, below which
/// the collapse is rejected to prevent degenerate triangles.
const MIN_TRIANGLE_QUALITY: f64 = 0.05;

// ---------------------------------------------------------------------------
// QuadricErrorMetric
// ---------------------------------------------------------------------------

/// A 4x4 symmetric matrix representing the quadric error at a vertex.
/// Stored as 10 unique elements (upper triangle of a symmetric 4x4 matrix).
///
/// The quadric Q for a plane (a, b, c, d) is the outer product:
///   Q = [a b c d]^T * [a b c d]
///
/// The error at a point v = (x, y, z, 1) is:
///   error = v^T * Q * v
#[derive(Debug, Clone, Copy)]
pub struct QuadricErrorMetric {
    /// Elements of the upper triangle: a11, a12, a13, a14, a22, a23, a24, a33, a34, a44
    pub data: [f64; 10],
}

impl QuadricErrorMetric {
    /// Zero quadric.
    pub fn zero() -> Self {
        Self { data: [0.0; 10] }
    }

    /// Build a quadric from a plane equation (a, b, c, d) where
    /// ax + by + cz + d = 0.
    pub fn from_plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            data: [
                a * a, a * b, a * c, a * d, // row 0: a11, a12, a13, a14
                b * b, b * c, b * d,         // row 1: a22, a23, a24
                c * c, c * d,                 // row 2: a33, a34
                d * d,                         // row 3: a44
            ],
        }
    }

    /// Build a quadric from a triangle (three vertex positions).
    pub fn from_triangle(p0: Vec3, p1: Vec3, p2: Vec3) -> Self {
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let normal = e1.cross(e2);

        let len = normal.length();
        if len < 1e-10 {
            return Self::zero();
        }

        let n = normal / len;
        let a = n.x as f64;
        let b = n.y as f64;
        let c = n.z as f64;
        let d = -(a * p0.x as f64 + b * p0.y as f64 + c * p0.z as f64);

        // Weight by triangle area (half the cross product magnitude).
        let area = len as f64 * 0.5;
        let mut q = Self::from_plane(a, b, c, d);
        q.scale(area);
        q
    }

    /// Add another quadric to this one.
    pub fn add(&mut self, other: &QuadricErrorMetric) {
        for i in 0..10 {
            self.data[i] += other.data[i];
        }
    }

    /// Scale all elements by a factor.
    pub fn scale(&mut self, factor: f64) {
        for d in &mut self.data {
            *d *= factor;
        }
    }

    /// Sum of two quadrics.
    pub fn sum(&self, other: &QuadricErrorMetric) -> QuadricErrorMetric {
        let mut result = *self;
        result.add(other);
        result
    }

    /// Evaluate the error at a point v = (x, y, z).
    ///
    /// error = v^T * Q * v where v = (x, y, z, 1)
    pub fn evaluate(&self, v: Vec3) -> f64 {
        let x = v.x as f64;
        let y = v.y as f64;
        let z = v.z as f64;

        let a11 = self.data[0];
        let a12 = self.data[1];
        let a13 = self.data[2];
        let a14 = self.data[3];
        let a22 = self.data[4];
        let a23 = self.data[5];
        let a24 = self.data[6];
        let a33 = self.data[7];
        let a34 = self.data[8];
        let a44 = self.data[9];

        a11 * x * x + 2.0 * a12 * x * y + 2.0 * a13 * x * z + 2.0 * a14 * x
            + a22 * y * y + 2.0 * a23 * y * z + 2.0 * a24 * y
            + a33 * z * z + 2.0 * a34 * z
            + a44
    }

    /// Find the optimal contraction point by solving the linear system.
    /// Returns None if the system is singular (degenerate case).
    pub fn optimal_point(&self) -> Option<Vec3> {
        let a11 = self.data[0];
        let a12 = self.data[1];
        let a13 = self.data[2];
        let a14 = self.data[3];
        let a22 = self.data[4];
        let a23 = self.data[5];
        let a24 = self.data[6];
        let a33 = self.data[7];
        let a34 = self.data[8];

        // Solve the 3x3 system:
        // [a11 a12 a13] [x]   [-a14]
        // [a12 a22 a23] [y] = [-a24]
        // [a13 a23 a33] [z]   [-a34]

        let det = a11 * (a22 * a33 - a23 * a23)
            - a12 * (a12 * a33 - a23 * a13)
            + a13 * (a12 * a23 - a22 * a13);

        if det.abs() < 1e-15 {
            return None;
        }

        let inv_det = 1.0 / det;

        let x = inv_det
            * (-a14 * (a22 * a33 - a23 * a23)
                + a24 * (a12 * a33 - a13 * a23)
                - a34 * (a12 * a23 - a13 * a22));

        let y = inv_det
            * (a14 * (a12 * a33 - a13 * a23)
                - a24 * (a11 * a33 - a13 * a13)
                + a34 * (a11 * a23 - a12 * a13));

        let z = inv_det
            * (-a14 * (a12 * a23 - a13 * a22)
                + a24 * (a11 * a23 - a12 * a13)
                - a34 * (a11 * a22 - a12 * a12));

        // Sanity check: reject if the point is too far from origin.
        if x.abs() > 1e6 || y.abs() > 1e6 || z.abs() > 1e6 {
            return None;
        }

        Some(Vec3::new(x as f32, y as f32, z as f32))
    }
}

impl Default for QuadricErrorMetric {
    fn default() -> Self {
        Self::zero()
    }
}

impl std::ops::Add for QuadricErrorMetric {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        self.sum(&rhs)
    }
}

impl std::ops::AddAssign for QuadricErrorMetric {
    fn add_assign(&mut self, rhs: Self) {
        self.add(&rhs);
    }
}

// ---------------------------------------------------------------------------
// EdgeCollapse
// ---------------------------------------------------------------------------

/// An ordered edge in the mesh identified by two vertex indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Edge {
    v0: u32,
    v1: u32,
}

impl Edge {
    fn new(a: u32, b: u32) -> Self {
        if a <= b { Edge { v0: a, v1: b } } else { Edge { v0: b, v1: a } }
    }
}

/// A candidate edge collapse with computed cost and optimal position.
#[derive(Debug, Clone)]
pub struct EdgeCollapse {
    /// The edge to collapse.
    pub v0: u32,
    pub v1: u32,
    /// Cost of this collapse (lower = better).
    pub cost: f64,
    /// Optimal position for the merged vertex.
    pub optimal_position: Vec3,
    /// Whether this edge is on the mesh boundary.
    pub is_boundary: bool,
    /// Whether this edge is on a UV seam.
    pub is_uv_seam: bool,
    /// Sequence number for tie-breaking in the priority queue.
    sequence: u64,
}

impl PartialEq for EdgeCollapse {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for EdgeCollapse {}

impl PartialOrd for EdgeCollapse {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EdgeCollapse {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (lowest cost first).
        other.cost.partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

// ---------------------------------------------------------------------------
// SimplifiedMesh
// ---------------------------------------------------------------------------

/// Result of mesh simplification.
#[derive(Debug, Clone)]
pub struct SimplifiedMesh {
    /// Simplified vertex data.
    pub vertices: Vec<Vertex>,
    /// Simplified index data.
    pub indices: Vec<u32>,
    /// Final triangle count.
    pub triangle_count: usize,
    /// Final vertex count.
    pub vertex_count: usize,
    /// Total error introduced by simplification.
    pub total_error: f64,
    /// Number of edge collapses performed.
    pub collapses_performed: usize,
}

// ---------------------------------------------------------------------------
// Internal simplification state
// ---------------------------------------------------------------------------

/// Per-vertex data during simplification.
#[derive(Debug, Clone)]
struct VertexData {
    /// Position.
    position: Vec3,
    /// Normal.
    normal: Vec3,
    /// UV coordinate.
    uv: Vec2,
    /// Quadric error metric.
    quadric: QuadricErrorMetric,
    /// Set of adjacent vertex indices.
    neighbors: HashSet<u32>,
    /// Set of adjacent triangle indices.
    triangles: HashSet<u32>,
    /// Whether this vertex has been collapsed (removed).
    collapsed: bool,
    /// If collapsed, the vertex it was merged into.
    collapse_target: u32,
    /// Whether this vertex is on a boundary edge.
    is_boundary: bool,
    /// Whether this vertex is locked (cannot be collapsed).
    is_locked: bool,
}

/// Per-triangle data during simplification.
#[derive(Debug, Clone)]
struct TriangleData {
    /// Vertex indices.
    indices: [u32; 3],
    /// Whether this triangle has been removed.
    removed: bool,
}

impl TriangleData {
    fn has_vertex(&self, v: u32) -> bool {
        self.indices[0] == v || self.indices[1] == v || self.indices[2] == v
    }

    fn replace_vertex(&mut self, old: u32, new: u32) {
        for idx in &mut self.indices {
            if *idx == old {
                *idx = new;
            }
        }
    }

    fn is_degenerate(&self) -> bool {
        self.indices[0] == self.indices[1]
            || self.indices[1] == self.indices[2]
            || self.indices[2] == self.indices[0]
    }
}

// ---------------------------------------------------------------------------
// Simplification implementation
// ---------------------------------------------------------------------------

/// Simplify a mesh to approximately the target triangle count using QEM.
pub fn simplify(mesh: &crate::mesh::Mesh, target_triangle_count: usize) -> SimplifiedMesh {
    simplify_raw(&mesh.vertices, &mesh.indices, target_triangle_count)
}

/// Simplify with boundary preservation. Vertices on mesh boundaries and
/// UV seams are penalised to discourage their collapse.
pub fn simplify_preserve_boundaries(
    mesh: &crate::mesh::Mesh,
    target_triangle_count: usize,
) -> SimplifiedMesh {
    simplify_internal(
        &mesh.vertices,
        &mesh.indices,
        target_triangle_count,
        &HashSet::new(),
        true,
    )
}

/// Simplify raw vertex/index data (used by the cluster DAG builder).
pub fn simplify_raw(
    vertices: &[Vertex],
    indices: &[u32],
    target_triangle_count: usize,
) -> SimplifiedMesh {
    simplify_internal(vertices, indices, target_triangle_count, &HashSet::new(), true)
}

/// Simplify with locked vertices (vertices at cluster boundaries that must
/// not be moved).
pub fn simplify_with_locked_vertices(
    vertices: &[Vertex],
    indices: &[u32],
    target_triangle_count: usize,
    locked_vertices: &HashSet<u32>,
) -> SimplifiedMesh {
    simplify_internal(vertices, indices, target_triangle_count, locked_vertices, true)
}

/// Core simplification implementation.
fn simplify_internal(
    vertices: &[Vertex],
    indices: &[u32],
    target_triangle_count: usize,
    locked_vertices: &HashSet<u32>,
    preserve_boundaries: bool,
) -> SimplifiedMesh {
    let tri_count = indices.len() / 3;

    if tri_count <= target_triangle_count || tri_count == 0 {
        return SimplifiedMesh {
            vertices: vertices.to_vec(),
            indices: indices.to_vec(),
            triangle_count: tri_count,
            vertex_count: vertices.len(),
            total_error: 0.0,
            collapses_performed: 0,
        };
    }

    // Initialize vertex data.
    let mut vertex_data: Vec<VertexData> = vertices
        .iter()
        .enumerate()
        .map(|(i, v)| VertexData {
            position: Vec3::from_array(v.position),
            normal: Vec3::from_array(v.normal),
            uv: Vec2::from_array(v.uv),
            quadric: QuadricErrorMetric::zero(),
            neighbors: HashSet::new(),
            triangles: HashSet::new(),
            collapsed: false,
            collapse_target: i as u32,
            is_boundary: false,
            is_locked: locked_vertices.contains(&(i as u32)),
        })
        .collect();

    // Initialize triangle data.
    let mut triangle_data: Vec<TriangleData> = (0..tri_count)
        .map(|t| {
            let base = t * 3;
            TriangleData {
                indices: [indices[base], indices[base + 1], indices[base + 2]],
                removed: false,
            }
        })
        .collect();

    // Build adjacency and compute initial quadrics.
    let mut edge_count_map: HashMap<(u32, u32), usize> = HashMap::new();

    for (tri_idx, tri) in triangle_data.iter().enumerate() {
        let [i0, i1, i2] = tri.indices;

        // Add triangle to vertex adjacency.
        vertex_data[i0 as usize].triangles.insert(tri_idx as u32);
        vertex_data[i1 as usize].triangles.insert(tri_idx as u32);
        vertex_data[i2 as usize].triangles.insert(tri_idx as u32);

        // Add vertex neighbors.
        vertex_data[i0 as usize].neighbors.insert(i1);
        vertex_data[i0 as usize].neighbors.insert(i2);
        vertex_data[i1 as usize].neighbors.insert(i0);
        vertex_data[i1 as usize].neighbors.insert(i2);
        vertex_data[i2 as usize].neighbors.insert(i0);
        vertex_data[i2 as usize].neighbors.insert(i1);

        // Count edge usage for boundary detection.
        let edges = [
            Edge::new(i0, i1),
            Edge::new(i1, i2),
            Edge::new(i2, i0),
        ];
        for e in &edges {
            *edge_count_map.entry((e.v0, e.v1)).or_insert(0) += 1;
        }

        // Compute quadric from triangle.
        let p0 = Vec3::from_array(vertices[i0 as usize].position);
        let p1 = Vec3::from_array(vertices[i1 as usize].position);
        let p2 = Vec3::from_array(vertices[i2 as usize].position);

        let q = QuadricErrorMetric::from_triangle(p0, p1, p2);
        vertex_data[i0 as usize].quadric.add(&q);
        vertex_data[i1 as usize].quadric.add(&q);
        vertex_data[i2 as usize].quadric.add(&q);
    }

    // Detect boundary edges and vertices.
    if preserve_boundaries {
        for (&(v0, v1), &count) in &edge_count_map {
            if count == 1 {
                // This is a boundary edge.
                vertex_data[v0 as usize].is_boundary = true;
                vertex_data[v1 as usize].is_boundary = true;

                // Add boundary constraint planes to quadrics.
                let p0 = vertex_data[v0 as usize].position;
                let p1 = vertex_data[v1 as usize].position;
                let edge_dir = (p1 - p0).normalize_or_zero();

                // Create a plane perpendicular to the boundary edge.
                // Use a large weight to strongly discourage boundary collapses.
                let arbitrary_up = if edge_dir.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
                let boundary_normal = edge_dir.cross(arbitrary_up).normalize_or_zero();

                if boundary_normal.length() > 0.5 {
                    let a = boundary_normal.x as f64;
                    let b = boundary_normal.y as f64;
                    let c = boundary_normal.z as f64;
                    let d = -(a * p0.x as f64 + b * p0.y as f64 + c * p0.z as f64);

                    let mut boundary_q = QuadricErrorMetric::from_plane(a, b, c, d);
                    boundary_q.scale(BOUNDARY_PENALTY_WEIGHT);

                    vertex_data[v0 as usize].quadric.add(&boundary_q);
                    vertex_data[v1 as usize].quadric.add(&boundary_q);
                }
            }
        }
    }

    // Detect UV seams and add penalties.
    detect_uv_seams_and_penalize(&mut vertex_data, &triangle_data);

    // Build initial priority queue of edge collapses.
    let mut heap: BinaryHeap<EdgeCollapse> = BinaryHeap::new();
    let mut sequence: u64 = 0;
    let mut processed_edges: HashSet<(u32, u32)> = HashSet::new();

    for v_idx in 0..vertex_data.len() {
        if vertex_data[v_idx].collapsed || vertex_data[v_idx].is_locked {
            continue;
        }

        let neighbors: Vec<u32> = vertex_data[v_idx].neighbors.iter().copied().collect();
        for &neighbor in &neighbors {
            let edge = Edge::new(v_idx as u32, neighbor);
            let key = (edge.v0, edge.v1);
            if processed_edges.contains(&key) {
                continue;
            }
            processed_edges.insert(key);

            if let Some(collapse) = compute_edge_collapse(
                &vertex_data, edge.v0, edge.v1, sequence,
            ) {
                heap.push(collapse);
                sequence += 1;
            }
        }
    }

    // Perform edge collapses.
    let mut current_tri_count = tri_count;
    let mut total_error = 0.0;
    let mut collapses = 0;

    while current_tri_count > target_triangle_count {
        let collapse = match heap.pop() {
            Some(c) => c,
            None => break,
        };

        // Validate the collapse is still valid.
        let v0 = collapse.v0;
        let v1 = collapse.v1;

        if vertex_data[v0 as usize].collapsed || vertex_data[v1 as usize].collapsed {
            continue;
        }

        if vertex_data[v0 as usize].is_locked && vertex_data[v1 as usize].is_locked {
            continue;
        }

        // Check topology: prevent non-manifold results.
        if !is_collapse_valid(&vertex_data, &triangle_data, v0, v1, collapse.optimal_position) {
            continue;
        }

        // Perform the collapse: merge v1 into v0.
        let (keep, remove) = if vertex_data[v1 as usize].is_locked {
            (v1, v0)
        } else {
            (v0, v1)
        };

        // Update the kept vertex.
        vertex_data[keep as usize].position = collapse.optimal_position;
        vertex_data[keep as usize].quadric = vertex_data[keep as usize]
            .quadric
            .sum(&vertex_data[remove as usize].quadric);

        // Interpolate normal.
        let n0 = vertex_data[keep as usize].normal;
        let n1 = vertex_data[remove as usize].normal;
        vertex_data[keep as usize].normal = ((n0 + n1) * 0.5).normalize_or_zero();

        // Interpolate UV.
        let uv0 = vertex_data[keep as usize].uv;
        let uv1 = vertex_data[remove as usize].uv;
        vertex_data[keep as usize].uv = (uv0 + uv1) * 0.5;

        // Mark the removed vertex.
        vertex_data[remove as usize].collapsed = true;
        vertex_data[remove as usize].collapse_target = keep;

        // Update triangles: replace `remove` with `keep`.
        let affected_tris: Vec<u32> = vertex_data[remove as usize]
            .triangles
            .iter()
            .copied()
            .collect();

        for &tri_idx in &affected_tris {
            let tri = &mut triangle_data[tri_idx as usize];
            if tri.removed {
                continue;
            }

            tri.replace_vertex(remove, keep);

            if tri.is_degenerate() {
                tri.removed = true;
                current_tri_count -= 1;

                // Remove triangle from vertex adjacency.
                for &v in &tri.indices {
                    vertex_data[v as usize].triangles.remove(&tri_idx);
                }
            } else {
                vertex_data[keep as usize].triangles.insert(tri_idx);
            }
        }

        // Transfer neighbors.
        let remove_neighbors: Vec<u32> = vertex_data[remove as usize]
            .neighbors
            .iter()
            .copied()
            .collect();

        for &n in &remove_neighbors {
            if n != keep && !vertex_data[n as usize].collapsed {
                vertex_data[n as usize].neighbors.remove(&remove);
                vertex_data[n as usize].neighbors.insert(keep);
                vertex_data[keep as usize].neighbors.insert(n);
            }
        }
        vertex_data[keep as usize].neighbors.remove(&remove);
        vertex_data[remove as usize].neighbors.clear();
        vertex_data[remove as usize].triangles.clear();

        total_error += collapse.cost;
        collapses += 1;

        // Re-evaluate edges incident to the kept vertex.
        let keep_neighbors: Vec<u32> = vertex_data[keep as usize]
            .neighbors
            .iter()
            .copied()
            .collect();

        for &n in &keep_neighbors {
            if vertex_data[n as usize].collapsed || vertex_data[n as usize].is_locked {
                continue;
            }
            if let Some(new_collapse) = compute_edge_collapse(
                &vertex_data, keep, n, sequence,
            ) {
                heap.push(new_collapse);
                sequence += 1;
            }
        }
    }

    // Build output mesh.
    let mut vertex_remap: Vec<u32> = vec![u32::MAX; vertex_data.len()];
    let mut out_vertices: Vec<Vertex> = Vec::new();

    for (i, vd) in vertex_data.iter().enumerate() {
        if !vd.collapsed {
            vertex_remap[i] = out_vertices.len() as u32;
            out_vertices.push(Vertex {
                position: vd.position.to_array(),
                normal: vd.normal.to_array(),
                tangent: [0.0, 0.0, 1.0, 1.0],
                uv: vd.uv.to_array(),
            });
        }
    }

    let mut out_indices: Vec<u32> = Vec::new();
    for tri in &triangle_data {
        if tri.removed {
            continue;
        }

        let mut valid = true;
        let mut mapped = [0u32; 3];
        for k in 0..3 {
            let mut v = tri.indices[k];
            // Follow collapse chain.
            let mut depth = 0;
            while vertex_data[v as usize].collapsed && depth < 100 {
                v = vertex_data[v as usize].collapse_target;
                depth += 1;
            }

            if vertex_remap[v as usize] == u32::MAX {
                valid = false;
                break;
            }
            mapped[k] = vertex_remap[v as usize];
        }

        if valid && mapped[0] != mapped[1] && mapped[1] != mapped[2] && mapped[2] != mapped[0] {
            out_indices.push(mapped[0]);
            out_indices.push(mapped[1]);
            out_indices.push(mapped[2]);
        }
    }

    let final_tri_count = out_indices.len() / 3;

    SimplifiedMesh {
        vertex_count: out_vertices.len(),
        vertices: out_vertices,
        indices: out_indices,
        triangle_count: final_tri_count,
        total_error,
        collapses_performed: collapses,
    }
}

/// Compute the collapse cost and optimal position for an edge.
fn compute_edge_collapse(
    vertex_data: &[VertexData],
    v0: u32,
    v1: u32,
    sequence: u64,
) -> Option<EdgeCollapse> {
    let vd0 = &vertex_data[v0 as usize];
    let vd1 = &vertex_data[v1 as usize];

    if vd0.collapsed || vd1.collapsed {
        return None;
    }

    if vd0.is_locked && vd1.is_locked {
        return None;
    }

    let combined_q = vd0.quadric.sum(&vd1.quadric);

    // Try to find the optimal contraction point.
    let optimal_position = combined_q
        .optimal_point()
        .unwrap_or_else(|| {
            // Fallback: use midpoint or the vertex with lower error.
            let mid = (vd0.position + vd1.position) * 0.5;
            let e_mid = combined_q.evaluate(mid);
            let e0 = combined_q.evaluate(vd0.position);
            let e1 = combined_q.evaluate(vd1.position);

            if e0 <= e1 && e0 <= e_mid {
                vd0.position
            } else if e1 <= e_mid {
                vd1.position
            } else {
                mid
            }
        });

    let mut cost = combined_q.evaluate(optimal_position);

    // Apply penalties.
    let is_boundary = vd0.is_boundary || vd1.is_boundary;
    let is_uv_seam = is_uv_seam_edge(vd0, vd1);

    if is_boundary {
        cost += BOUNDARY_PENALTY_WEIGHT;
    }
    if is_uv_seam {
        cost += UV_SEAM_PENALTY_WEIGHT;
    }

    if cost > MAX_COLLAPSE_COST {
        return None;
    }

    Some(EdgeCollapse {
        v0,
        v1,
        cost,
        optimal_position,
        is_boundary,
        is_uv_seam,
        sequence,
    })
}

/// Check if an edge lies on a UV seam (vertices with same position but
/// different UVs).
fn is_uv_seam_edge(v0: &VertexData, v1: &VertexData) -> bool {
    let pos_dist = (v0.position - v1.position).length();
    if pos_dist < 1e-6 {
        let uv_dist = (v0.uv - v1.uv).length();
        uv_dist > 1e-4
    } else {
        false
    }
}

/// Detect UV seams across the mesh and add quadric penalties.
fn detect_uv_seams_and_penalize(
    vertex_data: &mut [VertexData],
    triangle_data: &[TriangleData],
) {
    // Group vertices by position to find UV seam vertices.
    let mut position_map: HashMap<[i32; 3], Vec<usize>> = HashMap::new();
    let scale = 10000.0_f32;

    for (i, vd) in vertex_data.iter().enumerate() {
        let key = [
            (vd.position.x * scale) as i32,
            (vd.position.y * scale) as i32,
            (vd.position.z * scale) as i32,
        ];
        position_map.entry(key).or_default().push(i);
    }

    for (_pos, indices) in &position_map {
        if indices.len() <= 1 {
            continue;
        }

        // Check if any pair has different UVs.
        let mut has_seam = false;
        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                let uv_dist = (vertex_data[indices[i]].uv - vertex_data[indices[j]].uv).length();
                if uv_dist > 1e-4 {
                    has_seam = true;
                    break;
                }
            }
            if has_seam {
                break;
            }
        }

        if has_seam {
            for &idx in indices {
                let pos = vertex_data[idx].position;
                let n = vertex_data[idx].normal;

                let a = n.x as f64;
                let b = n.y as f64;
                let c = n.z as f64;
                let d = -(a * pos.x as f64 + b * pos.y as f64 + c * pos.z as f64);

                let mut seam_q = QuadricErrorMetric::from_plane(a, b, c, d);
                seam_q.scale(UV_SEAM_PENALTY_WEIGHT);
                vertex_data[idx].quadric.add(&seam_q);
            }
        }
    }
}

/// Validate that a collapse would not create degenerate or non-manifold geometry.
fn is_collapse_valid(
    vertex_data: &[VertexData],
    triangle_data: &[TriangleData],
    v0: u32,
    v1: u32,
    new_position: Vec3,
) -> bool {
    // Check link condition: the intersection of v0's and v1's neighborhoods
    // should equal the vertices shared by both.
    let shared_neighbors: HashSet<u32> = vertex_data[v0 as usize]
        .neighbors
        .intersection(&vertex_data[v1 as usize].neighbors)
        .copied()
        .collect();

    // For manifold meshes, exactly 2 shared neighbors for interior edges,
    // 1 for boundary edges.
    if shared_neighbors.len() > 2 {
        return false;
    }

    // Check for normal flipping: ensure triangles don't invert.
    let affected_tris: Vec<u32> = vertex_data[v0 as usize]
        .triangles
        .union(&vertex_data[v1 as usize].triangles)
        .copied()
        .collect();

    for &tri_idx in &affected_tris {
        let tri = &triangle_data[tri_idx as usize];
        if tri.removed {
            continue;
        }

        // Skip triangles that will be degenerated by the collapse.
        if tri.has_vertex(v0) && tri.has_vertex(v1) {
            continue;
        }

        // Compute normal before and after.
        let mut new_indices = tri.indices;
        for idx in &mut new_indices {
            if *idx == v0 || *idx == v1 {
                *idx = v0; // both map to v0
            }
        }

        if new_indices[0] == new_indices[1]
            || new_indices[1] == new_indices[2]
            || new_indices[2] == new_indices[0]
        {
            continue; // degenerate
        }

        let p0 = if tri.indices[0] == v0 || tri.indices[0] == v1 {
            new_position
        } else {
            vertex_data[tri.indices[0] as usize].position
        };
        let p1 = if tri.indices[1] == v0 || tri.indices[1] == v1 {
            new_position
        } else {
            vertex_data[tri.indices[1] as usize].position
        };
        let p2 = if tri.indices[2] == v0 || tri.indices[2] == v1 {
            new_position
        } else {
            vertex_data[tri.indices[2] as usize].position
        };

        // Original normal.
        let orig_p0 = vertex_data[tri.indices[0] as usize].position;
        let orig_p1 = vertex_data[tri.indices[1] as usize].position;
        let orig_p2 = vertex_data[tri.indices[2] as usize].position;

        let orig_normal = (orig_p1 - orig_p0).cross(orig_p2 - orig_p0);
        let new_normal = (p1 - p0).cross(p2 - p0);

        // Check for normal flip.
        if orig_normal.dot(new_normal) < 0.0 {
            return false;
        }

        // Check triangle quality.
        let new_area = new_normal.length();
        if new_area < 1e-10 {
            continue; // degenerate, will be removed
        }

        let edge_lengths = [
            (p1 - p0).length(),
            (p2 - p1).length(),
            (p0 - p2).length(),
        ];
        let max_edge = edge_lengths.iter().fold(0.0_f32, |a, &b| a.max(b));
        let quality = (new_area * 0.5) / (max_edge * max_edge + 1e-10);

        if (quality as f64) < MIN_TRIANGLE_QUALITY {
            return false;
        }
    }

    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{create_sphere, create_cube, create_plane, Mesh, Vertex};
    use glam::Vec3;

    #[test]
    fn test_quadric_from_plane() {
        let q = QuadricErrorMetric::from_plane(0.0, 1.0, 0.0, -1.0);
        // A point on the plane y=1 should have zero error.
        let error = q.evaluate(Vec3::new(5.0, 1.0, 3.0));
        assert!(error.abs() < 1e-10);
    }

    #[test]
    fn test_quadric_from_triangle() {
        let p0 = Vec3::new(0.0, 0.0, 0.0);
        let p1 = Vec3::new(1.0, 0.0, 0.0);
        let p2 = Vec3::new(0.0, 0.0, 1.0);

        let q = QuadricErrorMetric::from_triangle(p0, p1, p2);
        // Points on the y=0 plane should have zero error.
        let error = q.evaluate(Vec3::new(0.5, 0.0, 0.5));
        assert!(error.abs() < 1e-6);

        // Points above the plane should have nonzero error.
        let error_above = q.evaluate(Vec3::new(0.5, 1.0, 0.5));
        assert!(error_above > 0.0);
    }

    #[test]
    fn test_quadric_add() {
        let q1 = QuadricErrorMetric::from_plane(1.0, 0.0, 0.0, -1.0);
        let q2 = QuadricErrorMetric::from_plane(0.0, 1.0, 0.0, -1.0);
        let sum = q1.sum(&q2);

        // The intersection of x=1 and y=1 is the line (1,1,z).
        let error = sum.evaluate(Vec3::new(1.0, 1.0, 0.0));
        assert!(error.abs() < 1e-10);
    }

    #[test]
    fn test_quadric_optimal_point() {
        // Three orthogonal planes intersecting at (1, 2, 3).
        let q1 = QuadricErrorMetric::from_plane(1.0, 0.0, 0.0, -1.0);
        let q2 = QuadricErrorMetric::from_plane(0.0, 1.0, 0.0, -2.0);
        let q3 = QuadricErrorMetric::from_plane(0.0, 0.0, 1.0, -3.0);
        let sum = q1.sum(&q2).sum(&q3);

        let opt = sum.optimal_point().unwrap();
        assert!((opt.x - 1.0).abs() < 1e-4);
        assert!((opt.y - 2.0).abs() < 1e-4);
        assert!((opt.z - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_simplify_cube() {
        let mesh = create_cube();
        let result = simplify(&mesh, 6);
        // Should have reduced triangle count.
        assert!(result.triangle_count <= 12);
        assert!(result.triangle_count >= 4); // can't go below a tetrahedron
    }

    #[test]
    fn test_simplify_sphere() {
        let mesh = create_sphere(16, 12);
        let original_tris = mesh.triangle_count() as usize;
        let target = original_tris / 4;

        let result = simplify(&mesh, target);
        assert!(result.triangle_count <= target + target / 2); // allow some slack
        assert!(result.triangle_count > 0);
        assert!(result.vertex_count > 0);
        assert!(result.collapses_performed > 0);
    }

    #[test]
    fn test_simplify_preserves_boundaries() {
        let mesh = create_plane(10.0, 10.0, 10, 10);
        let result = simplify_preserve_boundaries(&mesh, 50);
        assert!(result.triangle_count > 0);
    }

    #[test]
    fn test_simplify_no_reduction_needed() {
        let mesh = create_cube();
        let result = simplify(&mesh, 100);
        // Should return unchanged since cube has only 12 triangles.
        assert_eq!(result.triangle_count, 12);
        assert_eq!(result.collapses_performed, 0);
    }

    #[test]
    fn test_simplify_aggressive() {
        let mesh = create_sphere(32, 24);
        let result = simplify(&mesh, 4);
        // Even aggressive simplification should produce valid geometry.
        assert!(result.triangle_count >= 4);
        assert!(result.indices.len() == result.triangle_count * 3);
    }

    #[test]
    fn test_locked_vertices() {
        let mesh = create_sphere(8, 6);
        // Lock the first 10 vertices.
        let locked: HashSet<u32> = (0..10).collect();
        let result = simplify_with_locked_vertices(
            &mesh.vertices,
            &mesh.indices,
            10,
            &locked,
        );
        assert!(result.triangle_count > 0);
    }
}
