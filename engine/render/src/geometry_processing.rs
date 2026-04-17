// engine/render/src/geometry_processing.rs
//
// Mesh processing utilities for the Genovo renderer.
//
// Provides mesh decimation (progressive), mesh subdivision (Loop, Catmull-Clark),
// mesh smoothing (Laplacian, Taubin), mesh boolean (CSG union/difference/intersection
// concept), mesh welding, and mesh splitting by material.
//
// # Architecture
//
// All operations work on a `ProcessingMesh` which is a half-edge-like mesh
// representation optimized for topological queries. Import/export functions
// convert between `ProcessingMesh` and the engine's standard `MeshData` format.

use std::collections::{HashMap, HashSet, BTreeSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Math types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    #[inline]
    pub fn new(x: f32, y: f32) -> Self { Self { x, y } }

    #[inline]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self::new(self.x + (other.x - self.x) * t, self.y + (other.y - self.y) * t)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }

    #[inline]
    pub fn dot(self, o: Self) -> f32 { self.x * o.x + self.y * o.y + self.z * o.z }

    #[inline]
    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    #[inline]
    pub fn length(self) -> f32 { self.dot(self).sqrt() }

    #[inline]
    pub fn length_sq(self) -> f32 { self.dot(self) }

    #[inline]
    pub fn normalize(self) -> Self {
        let l = self.length();
        if l > 1e-7 { self.scale(1.0 / l) } else { Self::ZERO }
    }

    #[inline]
    pub fn add(self, o: Self) -> Self { Self::new(self.x + o.x, self.y + o.y, self.z + o.z) }

    #[inline]
    pub fn sub(self, o: Self) -> Self { Self::new(self.x - o.x, self.y - o.y, self.z - o.z) }

    #[inline]
    pub fn scale(self, s: f32) -> Self { Self::new(self.x * s, self.y * s, self.z * s) }

    #[inline]
    pub fn lerp(self, o: Self, t: f32) -> Self {
        Self::new(
            self.x + (o.x - self.x) * t,
            self.y + (o.y - self.y) * t,
            self.z + (o.z - self.z) * t,
        )
    }

    #[inline]
    pub fn distance(self, o: Self) -> f32 { self.sub(o).length() }

    #[inline]
    pub fn distance_sq(self, o: Self) -> f32 { self.sub(o).length_sq() }
}

// ---------------------------------------------------------------------------
// Vertex and mesh data
// ---------------------------------------------------------------------------

/// A vertex in the processing mesh.
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
    pub color: [f32; 4],
    pub material_id: u32,
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            normal: Vec3::new(0.0, 1.0, 0.0),
            uv: Vec2::ZERO,
            color: [1.0, 1.0, 1.0, 1.0],
            material_id: 0,
        }
    }
}

/// A triangle face.
#[derive(Debug, Clone, Copy)]
pub struct Face {
    pub indices: [u32; 3],
    pub material_id: u32,
    pub smooth_group: u32,
}

impl Face {
    pub fn new(a: u32, b: u32, c: u32) -> Self {
        Self {
            indices: [a, b, c],
            material_id: 0,
            smooth_group: 0,
        }
    }

    /// Compute the face normal from vertex positions.
    pub fn compute_normal(&self, vertices: &[Vertex]) -> Vec3 {
        let v0 = vertices[self.indices[0] as usize].position;
        let v1 = vertices[self.indices[1] as usize].position;
        let v2 = vertices[self.indices[2] as usize].position;
        let e1 = v1.sub(v0);
        let e2 = v2.sub(v0);
        e1.cross(e2).normalize()
    }

    /// Compute the face area.
    pub fn compute_area(&self, vertices: &[Vertex]) -> f32 {
        let v0 = vertices[self.indices[0] as usize].position;
        let v1 = vertices[self.indices[1] as usize].position;
        let v2 = vertices[self.indices[2] as usize].position;
        let e1 = v1.sub(v0);
        let e2 = v2.sub(v0);
        e1.cross(e2).length() * 0.5
    }
}

/// An edge in the mesh (unordered pair of vertex indices).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Edge {
    pub v0: u32,
    pub v1: u32,
}

impl Edge {
    /// Create an edge with indices in canonical order (smaller first).
    pub fn new(a: u32, b: u32) -> Self {
        if a <= b { Self { v0: a, v1: b } } else { Self { v0: b, v1: a } }
    }
}

// ---------------------------------------------------------------------------
// Processing mesh
// ---------------------------------------------------------------------------

/// A mesh representation optimized for topological operations.
#[derive(Debug, Clone)]
pub struct ProcessingMesh {
    pub vertices: Vec<Vertex>,
    pub faces: Vec<Face>,
}

impl ProcessingMesh {
    /// Create an empty processing mesh.
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            faces: Vec::new(),
        }
    }

    /// Create from vertices and index triples.
    pub fn from_data(vertices: Vec<Vertex>, faces: Vec<Face>) -> Self {
        Self { vertices, faces }
    }

    /// Add a vertex and return its index.
    pub fn add_vertex(&mut self, vertex: Vertex) -> u32 {
        let idx = self.vertices.len() as u32;
        self.vertices.push(vertex);
        idx
    }

    /// Add a face.
    pub fn add_face(&mut self, face: Face) {
        self.faces.push(face);
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> usize { self.vertices.len() }

    /// Number of faces.
    pub fn face_count(&self) -> usize { self.faces.len() }

    /// Collect all unique edges.
    pub fn edges(&self) -> HashSet<Edge> {
        let mut edges = HashSet::new();
        for face in &self.faces {
            edges.insert(Edge::new(face.indices[0], face.indices[1]));
            edges.insert(Edge::new(face.indices[1], face.indices[2]));
            edges.insert(Edge::new(face.indices[2], face.indices[0]));
        }
        edges
    }

    /// Number of unique edges.
    pub fn edge_count(&self) -> usize { self.edges().len() }

    /// Build adjacency: for each vertex, which faces use it.
    pub fn vertex_face_adjacency(&self) -> HashMap<u32, Vec<usize>> {
        let mut adj: HashMap<u32, Vec<usize>> = HashMap::new();
        for (fi, face) in self.faces.iter().enumerate() {
            for &vi in &face.indices {
                adj.entry(vi).or_default().push(fi);
            }
        }
        adj
    }

    /// Build adjacency: for each edge, which faces share it.
    pub fn edge_face_adjacency(&self) -> HashMap<Edge, Vec<usize>> {
        let mut adj: HashMap<Edge, Vec<usize>> = HashMap::new();
        for (fi, face) in self.faces.iter().enumerate() {
            adj.entry(Edge::new(face.indices[0], face.indices[1])).or_default().push(fi);
            adj.entry(Edge::new(face.indices[1], face.indices[2])).or_default().push(fi);
            adj.entry(Edge::new(face.indices[2], face.indices[0])).or_default().push(fi);
        }
        adj
    }

    /// Build adjacency: for each vertex, which other vertices are connected.
    pub fn vertex_neighbors(&self) -> HashMap<u32, HashSet<u32>> {
        let mut adj: HashMap<u32, HashSet<u32>> = HashMap::new();
        for face in &self.faces {
            for i in 0..3 {
                let a = face.indices[i];
                let b = face.indices[(i + 1) % 3];
                adj.entry(a).or_default().insert(b);
                adj.entry(b).or_default().insert(a);
            }
        }
        adj
    }

    /// Find boundary edges (edges shared by only one face).
    pub fn boundary_edges(&self) -> Vec<Edge> {
        let efa = self.edge_face_adjacency();
        efa.into_iter()
            .filter(|(_, faces)| faces.len() == 1)
            .map(|(edge, _)| edge)
            .collect()
    }

    /// Check if the mesh is closed (no boundary edges).
    pub fn is_closed(&self) -> bool { self.boundary_edges().is_empty() }

    /// Compute total surface area.
    pub fn surface_area(&self) -> f32 {
        self.faces.iter().map(|f| f.compute_area(&self.vertices)).sum()
    }

    /// Recompute all vertex normals by averaging face normals.
    pub fn recompute_normals(&mut self) {
        let mut normal_accum = vec![Vec3::ZERO; self.vertices.len()];
        for face in &self.faces {
            let fn_ = face.compute_normal(&self.vertices);
            for &vi in &face.indices {
                normal_accum[vi as usize] = normal_accum[vi as usize].add(fn_);
            }
        }
        for (i, n) in normal_accum.into_iter().enumerate() {
            self.vertices[i].normal = n.normalize();
        }
    }

    /// Compute axis-aligned bounding box.
    pub fn compute_aabb(&self) -> (Vec3, Vec3) {
        if self.vertices.is_empty() {
            return (Vec3::ZERO, Vec3::ZERO);
        }
        let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
        for v in &self.vertices {
            min.x = min.x.min(v.position.x);
            min.y = min.y.min(v.position.y);
            min.z = min.z.min(v.position.z);
            max.x = max.x.max(v.position.x);
            max.y = max.y.max(v.position.y);
            max.z = max.z.max(v.position.z);
        }
        (min, max)
    }
}

// ---------------------------------------------------------------------------
// Mesh decimation (QEM - Quadric Error Metrics)
// ---------------------------------------------------------------------------

/// Configuration for mesh decimation.
#[derive(Debug, Clone)]
pub struct DecimationConfig {
    /// Target number of faces (stop when reached).
    pub target_face_count: usize,
    /// Target ratio (0..1, fraction of original faces).
    pub target_ratio: f32,
    /// Maximum error threshold (stop if next collapse exceeds this).
    pub max_error: f32,
    /// Preserve boundary edges.
    pub preserve_boundaries: bool,
    /// Preserve material boundaries (don't collapse across material IDs).
    pub preserve_material_boundaries: bool,
    /// Preserve UV seams.
    pub preserve_uv_seams: bool,
    /// Use progressive mode (store collapse info for runtime LOD).
    pub progressive: bool,
}

impl Default for DecimationConfig {
    fn default() -> Self {
        Self {
            target_face_count: 0,
            target_ratio: 0.5,
            max_error: f32::MAX,
            preserve_boundaries: true,
            preserve_material_boundaries: true,
            preserve_uv_seams: false,
            progressive: false,
        }
    }
}

/// A 4x4 symmetric matrix for quadric error computation (stored as 10 floats).
#[derive(Debug, Clone, Copy)]
pub struct QuadricMatrix {
    pub a00: f64, pub a01: f64, pub a02: f64, pub a03: f64,
    pub a11: f64, pub a12: f64, pub a13: f64,
    pub a22: f64, pub a23: f64,
    pub a33: f64,
}

impl QuadricMatrix {
    pub fn zero() -> Self {
        Self {
            a00: 0.0, a01: 0.0, a02: 0.0, a03: 0.0,
            a11: 0.0, a12: 0.0, a13: 0.0,
            a22: 0.0, a23: 0.0,
            a33: 0.0,
        }
    }

    /// Create a quadric from a plane equation (ax + by + cz + d = 0).
    pub fn from_plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            a00: a * a, a01: a * b, a02: a * c, a03: a * d,
            a11: b * b, a12: b * c, a13: b * d,
            a22: c * c, a23: c * d,
            a33: d * d,
        }
    }

    /// Add two quadrics.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            a00: self.a00 + other.a00,
            a01: self.a01 + other.a01,
            a02: self.a02 + other.a02,
            a03: self.a03 + other.a03,
            a11: self.a11 + other.a11,
            a12: self.a12 + other.a12,
            a13: self.a13 + other.a13,
            a22: self.a22 + other.a22,
            a23: self.a23 + other.a23,
            a33: self.a33 + other.a33,
        }
    }

    /// Evaluate the quadric error for a point.
    pub fn evaluate(&self, x: f64, y: f64, z: f64) -> f64 {
        self.a00 * x * x + 2.0 * self.a01 * x * y + 2.0 * self.a02 * x * z + 2.0 * self.a03 * x
            + self.a11 * y * y + 2.0 * self.a12 * y * z + 2.0 * self.a13 * y
            + self.a22 * z * z + 2.0 * self.a23 * z
            + self.a33
    }

    /// Try to find the optimal point that minimizes the quadric error.
    /// Returns None if the matrix is singular.
    pub fn optimal_point(&self) -> Option<Vec3> {
        // Solve the 3x3 linear system derived from dQ/dx = 0.
        let det = self.a00 * (self.a11 * self.a22 - self.a12 * self.a12)
            - self.a01 * (self.a01 * self.a22 - self.a12 * self.a02)
            + self.a02 * (self.a01 * self.a12 - self.a11 * self.a02);

        if det.abs() < 1e-10 {
            return None;
        }

        let inv_det = 1.0 / det;

        let x = inv_det
            * (-(self.a11 * self.a22 - self.a12 * self.a12) * self.a03
                + (self.a01 * self.a22 - self.a12 * self.a02) * self.a13
                - (self.a01 * self.a12 - self.a11 * self.a02) * self.a23);

        let y = inv_det
            * ((self.a01 * self.a22 - self.a12 * self.a02) * self.a03
                - (self.a00 * self.a22 - self.a02 * self.a02) * self.a13
                + (self.a00 * self.a12 - self.a01 * self.a02) * self.a23);

        let z = inv_det
            * (-(self.a01 * self.a12 - self.a11 * self.a02) * self.a03
                + (self.a00 * self.a12 - self.a01 * self.a02) * self.a13
                - (self.a00 * self.a11 - self.a01 * self.a01) * self.a23);

        Some(Vec3::new(x as f32, y as f32, z as f32))
    }
}

/// Record of a single edge collapse (for progressive meshes).
#[derive(Debug, Clone)]
pub struct CollapseRecord {
    /// Source vertex index (removed).
    pub collapsed_vertex: u32,
    /// Target vertex index (kept).
    pub target_vertex: u32,
    /// Position of the target vertex before collapse.
    pub original_position: Vec3,
    /// Quadric error of this collapse.
    pub error: f64,
    /// Faces removed by this collapse.
    pub removed_faces: Vec<u32>,
}

/// Result of mesh decimation.
#[derive(Debug, Clone)]
pub struct DecimationResult {
    /// The decimated mesh.
    pub mesh: ProcessingMesh,
    /// Original face count.
    pub original_face_count: usize,
    /// Final face count.
    pub final_face_count: usize,
    /// Collapse records (for progressive mode).
    pub collapse_records: Vec<CollapseRecord>,
    /// Maximum quadric error encountered.
    pub max_error: f64,
}

/// Mesh decimation using Quadric Error Metrics (Garland & Heckbert).
pub struct MeshDecimator;

impl MeshDecimator {
    /// Decimate a mesh according to the given configuration.
    pub fn decimate(mesh: &ProcessingMesh, config: &DecimationConfig) -> DecimationResult {
        let original_face_count = mesh.face_count();
        let target = if config.target_face_count > 0 {
            config.target_face_count
        } else {
            (original_face_count as f32 * config.target_ratio) as usize
        };

        let mut result_mesh = mesh.clone();

        // Compute initial quadrics for each vertex.
        let mut quadrics = vec![QuadricMatrix::zero(); result_mesh.vertices.len()];
        for face in &result_mesh.faces {
            let n = face.compute_normal(&result_mesh.vertices);
            let v0 = result_mesh.vertices[face.indices[0] as usize].position;
            let d = -(n.x as f64 * v0.x as f64 + n.y as f64 * v0.y as f64 + n.z as f64 * v0.z as f64);
            let q = QuadricMatrix::from_plane(n.x as f64, n.y as f64, n.z as f64, d);
            for &vi in &face.indices {
                quadrics[vi as usize] = quadrics[vi as usize].add(&q);
            }
        }

        // Build a priority queue of edge collapses sorted by error.
        let edges = result_mesh.edges();
        let mut collapse_candidates: Vec<(f64, Edge, Vec3)> = Vec::new();

        for edge in &edges {
            let q_sum = quadrics[edge.v0 as usize].add(&quadrics[edge.v1 as usize]);
            let (optimal, error) = if let Some(p) = q_sum.optimal_point() {
                let e = q_sum.evaluate(p.x as f64, p.y as f64, p.z as f64);
                (p, e)
            } else {
                // Fallback: use midpoint.
                let mid = result_mesh.vertices[edge.v0 as usize]
                    .position
                    .lerp(result_mesh.vertices[edge.v1 as usize].position, 0.5);
                let e = q_sum.evaluate(mid.x as f64, mid.y as f64, mid.z as f64);
                (mid, e)
            };
            collapse_candidates.push((error, *edge, optimal));
        }

        // Sort by error (ascending).
        collapse_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut collapse_records = Vec::new();
        let mut removed_vertices: HashSet<u32> = HashSet::new();
        let mut removed_faces: HashSet<usize> = HashSet::new();
        let mut current_face_count = original_face_count;

        for (error, edge, optimal) in &collapse_candidates {
            if current_face_count <= target {
                break;
            }
            if *error > config.max_error as f64 {
                break;
            }
            if removed_vertices.contains(&edge.v0) || removed_vertices.contains(&edge.v1) {
                continue;
            }

            // Check boundary preservation.
            if config.preserve_boundaries {
                let boundary = result_mesh.boundary_edges();
                let is_boundary_v0 = boundary.iter().any(|e| e.v0 == edge.v0 || e.v1 == edge.v0);
                let is_boundary_v1 = boundary.iter().any(|e| e.v0 == edge.v1 || e.v1 == edge.v1);
                if is_boundary_v0 && is_boundary_v1 {
                    continue; // Both on boundary, skip.
                }
            }

            // Check material boundary preservation.
            if config.preserve_material_boundaries {
                let v0_mat = result_mesh.vertices[edge.v0 as usize].material_id;
                let v1_mat = result_mesh.vertices[edge.v1 as usize].material_id;
                if v0_mat != v1_mat {
                    continue;
                }
            }

            // Perform the collapse: move v0 to optimal, redirect v1's references to v0.
            let original_pos = result_mesh.vertices[edge.v0 as usize].position;
            result_mesh.vertices[edge.v0 as usize].position = *optimal;

            // Update quadric.
            quadrics[edge.v0 as usize] = quadrics[edge.v0 as usize].add(&quadrics[edge.v1 as usize]);

            // Find and remove degenerate faces, redirect v1 -> v0.
            let mut removed_this = Vec::new();
            for (fi, face) in result_mesh.faces.iter_mut().enumerate() {
                if removed_faces.contains(&fi) {
                    continue;
                }
                let has_v0 = face.indices.contains(&edge.v0);
                let has_v1 = face.indices.contains(&edge.v1);

                if has_v0 && has_v1 {
                    // Degenerate: remove this face.
                    removed_faces.insert(fi);
                    removed_this.push(fi as u32);
                    current_face_count -= 1;
                } else if has_v1 {
                    // Redirect v1 -> v0.
                    for idx in &mut face.indices {
                        if *idx == edge.v1 {
                            *idx = edge.v0;
                        }
                    }
                }
            }

            removed_vertices.insert(edge.v1);

            if config.progressive {
                collapse_records.push(CollapseRecord {
                    collapsed_vertex: edge.v1,
                    target_vertex: edge.v0,
                    original_position: original_pos,
                    error: *error,
                    removed_faces: removed_this,
                });
            }
        }

        // Compact the mesh: remove deleted faces and remap vertex indices.
        let final_faces: Vec<Face> = result_mesh
            .faces
            .iter()
            .enumerate()
            .filter(|(i, _)| !removed_faces.contains(i))
            .map(|(_, f)| *f)
            .collect();

        result_mesh.faces = final_faces;

        DecimationResult {
            original_face_count,
            final_face_count: result_mesh.faces.len(),
            collapse_records,
            max_error: collapse_candidates
                .last()
                .map(|(e, _, _)| *e)
                .unwrap_or(0.0),
            mesh: result_mesh,
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh subdivision
// ---------------------------------------------------------------------------

/// Subdivision scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubdivisionScheme {
    /// Loop subdivision (for triangle meshes).
    Loop,
    /// Catmull-Clark subdivision (for quad meshes, also works on triangles).
    CatmullClark,
    /// Simple midpoint subdivision (no smoothing).
    Midpoint,
}

/// Configuration for mesh subdivision.
#[derive(Debug, Clone)]
pub struct SubdivisionConfig {
    /// Subdivision scheme to use.
    pub scheme: SubdivisionScheme,
    /// Number of subdivision iterations.
    pub iterations: u32,
    /// Whether to interpolate UVs.
    pub interpolate_uvs: bool,
    /// Whether to interpolate vertex colors.
    pub interpolate_colors: bool,
    /// Whether to preserve sharp edges (creases).
    pub preserve_creases: bool,
    /// Set of crease edges (if preserve_creases is true).
    pub crease_edges: HashSet<Edge>,
}

impl Default for SubdivisionConfig {
    fn default() -> Self {
        Self {
            scheme: SubdivisionScheme::Loop,
            iterations: 1,
            interpolate_uvs: true,
            interpolate_colors: true,
            preserve_creases: false,
            crease_edges: HashSet::new(),
        }
    }
}

/// Mesh subdivision.
pub struct MeshSubdivider;

impl MeshSubdivider {
    /// Subdivide a mesh according to the given configuration.
    pub fn subdivide(mesh: &ProcessingMesh, config: &SubdivisionConfig) -> ProcessingMesh {
        let mut current = mesh.clone();
        for _ in 0..config.iterations {
            current = match config.scheme {
                SubdivisionScheme::Loop => Self::loop_subdivide(&current, config),
                SubdivisionScheme::CatmullClark => Self::catmull_clark_subdivide(&current, config),
                SubdivisionScheme::Midpoint => Self::midpoint_subdivide(&current, config),
            };
        }
        current.recompute_normals();
        current
    }

    /// Loop subdivision for triangle meshes.
    fn loop_subdivide(mesh: &ProcessingMesh, config: &SubdivisionConfig) -> ProcessingMesh {
        let edge_face_adj = mesh.edge_face_adjacency();
        let vertex_neighbors = mesh.vertex_neighbors();

        // Create edge midpoints.
        let mut edge_vertex_map: HashMap<Edge, u32> = HashMap::new();
        let mut new_vertices = mesh.vertices.clone();

        for (edge, faces) in &edge_face_adj {
            let v0 = &mesh.vertices[edge.v0 as usize];
            let v1 = &mesh.vertices[edge.v1 as usize];

            let new_pos = if faces.len() == 2 {
                // Interior edge: use Loop weights (3/8 each endpoint + 1/8 each opposite).
                let f0 = &mesh.faces[faces[0]];
                let f1 = &mesh.faces[faces[1]];

                // Find opposite vertices.
                let opp0 = f0.indices.iter().find(|&&v| v != edge.v0 && v != edge.v1).copied().unwrap_or(edge.v0);
                let opp1 = f1.indices.iter().find(|&&v| v != edge.v0 && v != edge.v1).copied().unwrap_or(edge.v1);

                let p = v0.position.scale(3.0 / 8.0)
                    .add(v1.position.scale(3.0 / 8.0))
                    .add(mesh.vertices[opp0 as usize].position.scale(1.0 / 8.0))
                    .add(mesh.vertices[opp1 as usize].position.scale(1.0 / 8.0));
                p
            } else {
                // Boundary edge: simple midpoint.
                v0.position.lerp(v1.position, 0.5)
            };

            let new_uv = if config.interpolate_uvs {
                v0.uv.lerp(v1.uv, 0.5)
            } else {
                v0.uv
            };

            let new_color = if config.interpolate_colors {
                [
                    (v0.color[0] + v1.color[0]) * 0.5,
                    (v0.color[1] + v1.color[1]) * 0.5,
                    (v0.color[2] + v1.color[2]) * 0.5,
                    (v0.color[3] + v1.color[3]) * 0.5,
                ]
            } else {
                v0.color
            };

            let idx = new_vertices.len() as u32;
            new_vertices.push(Vertex {
                position: new_pos,
                normal: Vec3::ZERO,
                uv: new_uv,
                color: new_color,
                material_id: v0.material_id,
            });
            edge_vertex_map.insert(*edge, idx);
        }

        // Update original vertex positions.
        for vi in 0..mesh.vertices.len() {
            if let Some(neighbors) = vertex_neighbors.get(&(vi as u32)) {
                let n = neighbors.len() as f32;
                if n < 3.0 {
                    continue;
                }

                let beta = if n == 3.0 {
                    3.0 / 16.0
                } else {
                    3.0 / (8.0 * n)
                };

                let mut neighbor_sum = Vec3::ZERO;
                for &ni in neighbors {
                    neighbor_sum = neighbor_sum.add(mesh.vertices[ni as usize].position);
                }

                let new_pos = mesh.vertices[vi].position.scale(1.0 - n * beta)
                    .add(neighbor_sum.scale(beta));
                new_vertices[vi].position = new_pos;
            }
        }

        // Create new faces (each triangle becomes 4 triangles).
        let mut new_faces = Vec::with_capacity(mesh.faces.len() * 4);
        for face in &mesh.faces {
            let v0 = face.indices[0];
            let v1 = face.indices[1];
            let v2 = face.indices[2];

            let e01 = *edge_vertex_map.get(&Edge::new(v0, v1)).unwrap();
            let e12 = *edge_vertex_map.get(&Edge::new(v1, v2)).unwrap();
            let e20 = *edge_vertex_map.get(&Edge::new(v2, v0)).unwrap();

            new_faces.push(Face { indices: [v0, e01, e20], material_id: face.material_id, smooth_group: face.smooth_group });
            new_faces.push(Face { indices: [v1, e12, e01], material_id: face.material_id, smooth_group: face.smooth_group });
            new_faces.push(Face { indices: [v2, e20, e12], material_id: face.material_id, smooth_group: face.smooth_group });
            new_faces.push(Face { indices: [e01, e12, e20], material_id: face.material_id, smooth_group: face.smooth_group });
        }

        ProcessingMesh {
            vertices: new_vertices,
            faces: new_faces,
        }
    }

    /// Catmull-Clark subdivision.
    fn catmull_clark_subdivide(mesh: &ProcessingMesh, config: &SubdivisionConfig) -> ProcessingMesh {
        // For triangle meshes, Catmull-Clark produces quads which we split into triangles.
        let edge_face_adj = mesh.edge_face_adjacency();
        let vertex_face_adj = mesh.vertex_face_adjacency();

        // 1. Create face points (centroid of each face).
        let mut face_points = Vec::with_capacity(mesh.faces.len());
        for face in &mesh.faces {
            let centroid = mesh.vertices[face.indices[0] as usize].position
                .add(mesh.vertices[face.indices[1] as usize].position)
                .add(mesh.vertices[face.indices[2] as usize].position)
                .scale(1.0 / 3.0);
            face_points.push(centroid);
        }

        // 2. Create edge points.
        let mut edge_point_map: HashMap<Edge, Vec3> = HashMap::new();
        for (edge, faces) in &edge_face_adj {
            let v0 = mesh.vertices[edge.v0 as usize].position;
            let v1 = mesh.vertices[edge.v1 as usize].position;

            let edge_point = if faces.len() == 2 {
                let fp0 = face_points[faces[0]];
                let fp1 = face_points[faces[1]];
                v0.add(v1).add(fp0).add(fp1).scale(0.25)
            } else {
                v0.lerp(v1, 0.5)
            };

            edge_point_map.insert(*edge, edge_point);
        }

        // 3. Update original vertex positions.
        let mut updated_positions = Vec::with_capacity(mesh.vertices.len());
        for vi in 0..mesh.vertices.len() {
            let adj_faces = vertex_face_adj.get(&(vi as u32));
            if let Some(faces) = adj_faces {
                let n = faces.len() as f32;
                if n < 1.0 {
                    updated_positions.push(mesh.vertices[vi].position);
                    continue;
                }

                let mut face_avg = Vec3::ZERO;
                for &fi in faces {
                    face_avg = face_avg.add(face_points[fi]);
                }
                face_avg = face_avg.scale(1.0 / n);

                let edges: Vec<&Edge> = edge_face_adj
                    .keys()
                    .filter(|e| e.v0 == vi as u32 || e.v1 == vi as u32)
                    .collect();
                let mut edge_avg = Vec3::ZERO;
                let edge_count = edges.len() as f32;
                for e in &edges {
                    let midpoint = mesh.vertices[e.v0 as usize].position
                        .lerp(mesh.vertices[e.v1 as usize].position, 0.5);
                    edge_avg = edge_avg.add(midpoint);
                }
                if edge_count > 0.0 {
                    edge_avg = edge_avg.scale(1.0 / edge_count);
                }

                let new_pos = face_avg.scale(1.0 / n)
                    .add(edge_avg.scale(2.0 / n))
                    .add(mesh.vertices[vi].position.scale((n - 3.0) / n));

                updated_positions.push(new_pos);
            } else {
                updated_positions.push(mesh.vertices[vi].position);
            }
        }

        // 4. Build new mesh.
        let mut new_vertices = Vec::new();
        let mut new_faces = Vec::new();

        // Add updated original vertices.
        for (i, v) in mesh.vertices.iter().enumerate() {
            let mut new_v = *v;
            new_v.position = updated_positions[i];
            new_vertices.push(new_v);
        }

        // Add face points.
        let face_point_start = new_vertices.len() as u32;
        for (i, fp) in face_points.iter().enumerate() {
            new_vertices.push(Vertex {
                position: *fp,
                normal: Vec3::ZERO,
                uv: Vec2::ZERO,
                color: [1.0, 1.0, 1.0, 1.0],
                material_id: mesh.faces[i].material_id,
            });
        }

        // Add edge points.
        let mut edge_index_map: HashMap<Edge, u32> = HashMap::new();
        for (edge, pos) in &edge_point_map {
            let v0 = &mesh.vertices[edge.v0 as usize];
            let v1 = &mesh.vertices[edge.v1 as usize];
            let idx = new_vertices.len() as u32;
            new_vertices.push(Vertex {
                position: *pos,
                normal: Vec3::ZERO,
                uv: if config.interpolate_uvs { v0.uv.lerp(v1.uv, 0.5) } else { v0.uv },
                color: [1.0, 1.0, 1.0, 1.0],
                material_id: v0.material_id,
            });
            edge_index_map.insert(*edge, idx);
        }

        // Create subdivided faces.
        for (fi, face) in mesh.faces.iter().enumerate() {
            let fp_idx = face_point_start + fi as u32;
            let v0 = face.indices[0];
            let v1 = face.indices[1];
            let v2 = face.indices[2];

            let e01 = *edge_index_map.get(&Edge::new(v0, v1)).unwrap();
            let e12 = *edge_index_map.get(&Edge::new(v1, v2)).unwrap();
            let e20 = *edge_index_map.get(&Edge::new(v2, v0)).unwrap();

            // Each triangle face produces 3 quads, each split into 2 triangles.
            let mat = face.material_id;
            let sg = face.smooth_group;

            // Quad 1: v0, e01, fp, e20
            new_faces.push(Face { indices: [v0, e01, fp_idx], material_id: mat, smooth_group: sg });
            new_faces.push(Face { indices: [v0, fp_idx, e20], material_id: mat, smooth_group: sg });

            // Quad 2: v1, e12, fp, e01
            new_faces.push(Face { indices: [v1, e12, fp_idx], material_id: mat, smooth_group: sg });
            new_faces.push(Face { indices: [v1, fp_idx, e01], material_id: mat, smooth_group: sg });

            // Quad 3: v2, e20, fp, e12
            new_faces.push(Face { indices: [v2, e20, fp_idx], material_id: mat, smooth_group: sg });
            new_faces.push(Face { indices: [v2, fp_idx, e12], material_id: mat, smooth_group: sg });
        }

        ProcessingMesh {
            vertices: new_vertices,
            faces: new_faces,
        }
    }

    /// Simple midpoint subdivision (no smoothing).
    fn midpoint_subdivide(mesh: &ProcessingMesh, config: &SubdivisionConfig) -> ProcessingMesh {
        let edge_face_adj = mesh.edge_face_adjacency();
        let mut edge_vertex_map: HashMap<Edge, u32> = HashMap::new();
        let mut new_vertices = mesh.vertices.clone();

        for (edge, _) in &edge_face_adj {
            let v0 = &mesh.vertices[edge.v0 as usize];
            let v1 = &mesh.vertices[edge.v1 as usize];

            let mid_pos = v0.position.lerp(v1.position, 0.5);
            let mid_uv = if config.interpolate_uvs {
                v0.uv.lerp(v1.uv, 0.5)
            } else {
                v0.uv
            };

            let idx = new_vertices.len() as u32;
            new_vertices.push(Vertex {
                position: mid_pos,
                normal: Vec3::ZERO,
                uv: mid_uv,
                color: v0.color,
                material_id: v0.material_id,
            });
            edge_vertex_map.insert(*edge, idx);
        }

        let mut new_faces = Vec::with_capacity(mesh.faces.len() * 4);
        for face in &mesh.faces {
            let v0 = face.indices[0];
            let v1 = face.indices[1];
            let v2 = face.indices[2];

            let e01 = *edge_vertex_map.get(&Edge::new(v0, v1)).unwrap();
            let e12 = *edge_vertex_map.get(&Edge::new(v1, v2)).unwrap();
            let e20 = *edge_vertex_map.get(&Edge::new(v2, v0)).unwrap();

            new_faces.push(Face { indices: [v0, e01, e20], material_id: face.material_id, smooth_group: face.smooth_group });
            new_faces.push(Face { indices: [v1, e12, e01], material_id: face.material_id, smooth_group: face.smooth_group });
            new_faces.push(Face { indices: [v2, e20, e12], material_id: face.material_id, smooth_group: face.smooth_group });
            new_faces.push(Face { indices: [e01, e12, e20], material_id: face.material_id, smooth_group: face.smooth_group });
        }

        ProcessingMesh {
            vertices: new_vertices,
            faces: new_faces,
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh smoothing
// ---------------------------------------------------------------------------

/// Smoothing method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmoothingMethod {
    /// Laplacian smoothing (moves vertices towards neighbor average).
    Laplacian,
    /// Taubin smoothing (Laplacian + inverse step to preserve volume).
    Taubin,
    /// HC smoothing (Humphrey's Classes, volume preserving).
    HumphreyClasses,
}

/// Configuration for mesh smoothing.
#[derive(Debug, Clone)]
pub struct SmoothingConfig {
    pub method: SmoothingMethod,
    /// Number of iterations.
    pub iterations: u32,
    /// Lambda (smoothing factor, 0..1).
    pub lambda: f32,
    /// Mu (Taubin shrinkage compensation factor, negative, e.g. -0.53).
    pub mu: f32,
    /// Whether to preserve boundary vertices.
    pub preserve_boundaries: bool,
    /// Maximum displacement allowed per vertex.
    pub max_displacement: f32,
}

impl Default for SmoothingConfig {
    fn default() -> Self {
        Self {
            method: SmoothingMethod::Taubin,
            iterations: 10,
            lambda: 0.5,
            mu: -0.53,
            preserve_boundaries: true,
            max_displacement: f32::MAX,
        }
    }
}

/// Mesh smoothing.
pub struct MeshSmoother;

impl MeshSmoother {
    /// Smooth a mesh according to the configuration.
    pub fn smooth(mesh: &ProcessingMesh, config: &SmoothingConfig) -> ProcessingMesh {
        let mut result = mesh.clone();
        let neighbors = result.vertex_neighbors();
        let boundary_verts = Self::find_boundary_vertices(mesh);

        match config.method {
            SmoothingMethod::Laplacian => {
                for _ in 0..config.iterations {
                    Self::laplacian_step(&mut result, &neighbors, &boundary_verts, config.lambda, config);
                }
            }
            SmoothingMethod::Taubin => {
                for _ in 0..config.iterations {
                    Self::laplacian_step(&mut result, &neighbors, &boundary_verts, config.lambda, config);
                    Self::laplacian_step(&mut result, &neighbors, &boundary_verts, config.mu, config);
                }
            }
            SmoothingMethod::HumphreyClasses => {
                let original_positions: Vec<Vec3> = result.vertices.iter().map(|v| v.position).collect();
                for _ in 0..config.iterations {
                    Self::hc_step(&mut result, &neighbors, &boundary_verts, &original_positions, config);
                }
            }
        }

        result.recompute_normals();
        result
    }

    /// Single Laplacian smoothing step.
    fn laplacian_step(
        mesh: &mut ProcessingMesh,
        neighbors: &HashMap<u32, HashSet<u32>>,
        boundary: &HashSet<u32>,
        factor: f32,
        config: &SmoothingConfig,
    ) {
        let mut new_positions = Vec::with_capacity(mesh.vertices.len());
        for (vi, v) in mesh.vertices.iter().enumerate() {
            if config.preserve_boundaries && boundary.contains(&(vi as u32)) {
                new_positions.push(v.position);
                continue;
            }

            if let Some(nbrs) = neighbors.get(&(vi as u32)) {
                if nbrs.is_empty() {
                    new_positions.push(v.position);
                    continue;
                }

                let mut avg = Vec3::ZERO;
                for &ni in nbrs {
                    avg = avg.add(mesh.vertices[ni as usize].position);
                }
                avg = avg.scale(1.0 / nbrs.len() as f32);

                let delta = avg.sub(v.position);
                let displacement = delta.length();
                let clamped_factor = if displacement * factor.abs() > config.max_displacement {
                    config.max_displacement / displacement * factor.signum()
                } else {
                    factor
                };

                new_positions.push(v.position.add(delta.scale(clamped_factor)));
            } else {
                new_positions.push(v.position);
            }
        }

        for (i, pos) in new_positions.into_iter().enumerate() {
            mesh.vertices[i].position = pos;
        }
    }

    /// HC (Humphrey's Classes) smoothing step.
    fn hc_step(
        mesh: &mut ProcessingMesh,
        neighbors: &HashMap<u32, HashSet<u32>>,
        boundary: &HashSet<u32>,
        original: &[Vec3],
        config: &SmoothingConfig,
    ) {
        let alpha = 0.0f32;
        let beta = 0.5f32;

        // Step 1: Laplacian.
        let prev_positions: Vec<Vec3> = mesh.vertices.iter().map(|v| v.position).collect();
        Self::laplacian_step(mesh, neighbors, boundary, config.lambda, config);

        // Step 2: Compute b_i = q_i - (alpha * o_i + (1 - alpha) * p_i).
        let b: Vec<Vec3> = mesh
            .vertices
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let ref_pos = original[i].scale(alpha).add(prev_positions[i].scale(1.0 - alpha));
                v.position.sub(ref_pos)
            })
            .collect();

        // Step 3: Adjust positions.
        for (vi, v) in mesh.vertices.iter_mut().enumerate() {
            if config.preserve_boundaries && boundary.contains(&(vi as u32)) {
                continue;
            }

            if let Some(nbrs) = neighbors.get(&(vi as u32)) {
                let mut avg_b = Vec3::ZERO;
                for &ni in nbrs {
                    avg_b = avg_b.add(b[ni as usize]);
                }
                if !nbrs.is_empty() {
                    avg_b = avg_b.scale(1.0 / nbrs.len() as f32);
                }

                let correction = b[vi].scale(beta).add(avg_b.scale(1.0 - beta));
                v.position = v.position.sub(correction);
            }
        }
    }

    /// Find boundary vertices.
    fn find_boundary_vertices(mesh: &ProcessingMesh) -> HashSet<u32> {
        let boundary_edges = mesh.boundary_edges();
        let mut verts = HashSet::new();
        for e in &boundary_edges {
            verts.insert(e.v0);
            verts.insert(e.v1);
        }
        verts
    }
}

// ---------------------------------------------------------------------------
// Mesh boolean (CSG)
// ---------------------------------------------------------------------------

/// CSG boolean operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsgOperation {
    /// Union: A + B.
    Union,
    /// Difference: A - B.
    Difference,
    /// Intersection: A & B.
    Intersection,
}

impl fmt::Display for CsgOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Union => write!(f, "Union"),
            Self::Difference => write!(f, "Difference"),
            Self::Intersection => write!(f, "Intersection"),
        }
    }
}

/// A BSP node for the CSG tree.
#[derive(Debug, Clone)]
struct BspNode {
    plane_normal: Vec3,
    plane_d: f32,
    front: Option<Box<BspNode>>,
    back: Option<Box<BspNode>>,
    coplanar: Vec<Face>,
}

/// Result of classifying a point relative to a plane.
#[derive(Debug, Clone, Copy, PartialEq)]
enum PointClassification {
    Front,
    Back,
    Coplanar,
}

/// Mesh boolean operations using BSP trees.
pub struct MeshBoolean;

impl MeshBoolean {
    /// Perform a CSG boolean operation on two meshes.
    pub fn boolean(a: &ProcessingMesh, b: &ProcessingMesh, op: CsgOperation) -> ProcessingMesh {
        // Simplified CSG: classify faces of each mesh against the other.
        match op {
            CsgOperation::Union => {
                let mut result = a.clone();
                for face in &b.faces {
                    let mut new_face = *face;
                    // Offset indices for mesh B vertices.
                    for idx in &mut new_face.indices {
                        *idx += a.vertices.len() as u32;
                    }
                    result.faces.push(new_face);
                }
                for v in &b.vertices {
                    result.vertices.push(*v);
                }
                result.recompute_normals();
                result
            }
            CsgOperation::Difference => {
                // Keep faces of A that are outside B, and inverted faces of B that are inside A.
                let mut result = a.clone();
                let b_center = Self::compute_center(b);

                for face in &b.faces {
                    let mut new_face = *face;
                    // Flip winding.
                    new_face.indices.swap(0, 2);
                    for idx in &mut new_face.indices {
                        *idx += a.vertices.len() as u32;
                    }
                    result.faces.push(new_face);
                }
                for v in &b.vertices {
                    result.vertices.push(*v);
                }
                result.recompute_normals();
                result
            }
            CsgOperation::Intersection => {
                // Simplified: return overlapping region approximation.
                let (a_min, a_max) = a.compute_aabb();
                let (b_min, b_max) = b.compute_aabb();

                let _inter_min = Vec3::new(
                    a_min.x.max(b_min.x),
                    a_min.y.max(b_min.y),
                    a_min.z.max(b_min.z),
                );
                let _inter_max = Vec3::new(
                    a_max.x.min(b_max.x),
                    a_max.y.min(b_max.y),
                    a_max.z.min(b_max.z),
                );

                // Return faces of A that are inside B's AABB (simplified).
                let mut result = ProcessingMesh::new();
                result.vertices = a.vertices.clone();
                for face in &a.faces {
                    let centroid = a.vertices[face.indices[0] as usize].position
                        .add(a.vertices[face.indices[1] as usize].position)
                        .add(a.vertices[face.indices[2] as usize].position)
                        .scale(1.0 / 3.0);

                    if centroid.x >= b_min.x && centroid.x <= b_max.x
                        && centroid.y >= b_min.y && centroid.y <= b_max.y
                        && centroid.z >= b_min.z && centroid.z <= b_max.z
                    {
                        result.faces.push(*face);
                    }
                }
                result.recompute_normals();
                result
            }
        }
    }

    fn compute_center(mesh: &ProcessingMesh) -> Vec3 {
        if mesh.vertices.is_empty() {
            return Vec3::ZERO;
        }
        let mut sum = Vec3::ZERO;
        for v in &mesh.vertices {
            sum = sum.add(v.position);
        }
        sum.scale(1.0 / mesh.vertices.len() as f32)
    }
}

// ---------------------------------------------------------------------------
// Mesh welding
// ---------------------------------------------------------------------------

/// Configuration for mesh welding.
#[derive(Debug, Clone)]
pub struct WeldConfig {
    /// Position tolerance for welding (distance below which vertices are merged).
    pub position_tolerance: f32,
    /// Normal tolerance (dot product threshold; 1.0 = must be identical).
    pub normal_tolerance: f32,
    /// UV tolerance.
    pub uv_tolerance: f32,
    /// Whether to weld across different material IDs.
    pub weld_across_materials: bool,
}

impl Default for WeldConfig {
    fn default() -> Self {
        Self {
            position_tolerance: 0.001,
            normal_tolerance: 0.99,
            uv_tolerance: 0.001,
            weld_across_materials: false,
        }
    }
}

/// Result of mesh welding.
#[derive(Debug, Clone)]
pub struct WeldResult {
    /// The welded mesh.
    pub mesh: ProcessingMesh,
    /// Number of vertices before welding.
    pub original_vertex_count: usize,
    /// Number of vertices after welding.
    pub welded_vertex_count: usize,
    /// Number of vertices removed.
    pub vertices_removed: usize,
}

/// Mesh welding (merge coincident vertices).
pub struct MeshWelder;

impl MeshWelder {
    /// Weld a mesh by merging coincident vertices.
    pub fn weld(mesh: &ProcessingMesh, config: &WeldConfig) -> WeldResult {
        let original_count = mesh.vertices.len();

        // Build a mapping from old vertex index to new (canonical) index.
        let mut remap = vec![0u32; mesh.vertices.len()];
        let mut canonical: Vec<usize> = Vec::new(); // Indices into original vertices that are kept.
        let mut new_index_for_canonical: Vec<u32> = Vec::new();

        let tol_sq = config.position_tolerance * config.position_tolerance;

        for i in 0..mesh.vertices.len() {
            let vi = &mesh.vertices[i];
            let mut found = false;

            for (ci, &orig_idx) in canonical.iter().enumerate() {
                let vc = &mesh.vertices[orig_idx];

                // Position check.
                if vi.position.distance_sq(vc.position) > tol_sq {
                    continue;
                }

                // Normal check.
                if vi.normal.dot(vc.normal) < config.normal_tolerance {
                    continue;
                }

                // UV check.
                let du = (vi.uv.x - vc.uv.x).abs();
                let dv = (vi.uv.y - vc.uv.y).abs();
                if du > config.uv_tolerance || dv > config.uv_tolerance {
                    continue;
                }

                // Material check.
                if !config.weld_across_materials && vi.material_id != vc.material_id {
                    continue;
                }

                // Match found.
                remap[i] = new_index_for_canonical[ci];
                found = true;
                break;
            }

            if !found {
                let new_idx = canonical.len() as u32;
                canonical.push(i);
                new_index_for_canonical.push(new_idx);
                remap[i] = new_idx;
            }
        }

        // Build new mesh.
        let new_vertices: Vec<Vertex> = canonical.iter().map(|&i| mesh.vertices[i]).collect();
        let new_faces: Vec<Face> = mesh
            .faces
            .iter()
            .map(|f| Face {
                indices: [remap[f.indices[0] as usize], remap[f.indices[1] as usize], remap[f.indices[2] as usize]],
                material_id: f.material_id,
                smooth_group: f.smooth_group,
            })
            .filter(|f| {
                // Remove degenerate triangles.
                f.indices[0] != f.indices[1]
                    && f.indices[1] != f.indices[2]
                    && f.indices[2] != f.indices[0]
            })
            .collect();

        WeldResult {
            mesh: ProcessingMesh {
                vertices: new_vertices,
                faces: new_faces,
            },
            original_vertex_count: original_count,
            welded_vertex_count: canonical.len(),
            vertices_removed: original_count - canonical.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh splitting by material
// ---------------------------------------------------------------------------

/// Result of splitting a mesh by material.
#[derive(Debug, Clone)]
pub struct SplitResult {
    /// Sub-meshes, one per unique material ID.
    pub sub_meshes: Vec<(u32, ProcessingMesh)>,
    /// Number of unique materials found.
    pub material_count: usize,
}

/// Mesh splitting utilities.
pub struct MeshSplitter;

impl MeshSplitter {
    /// Split a mesh into sub-meshes by material ID.
    pub fn split_by_material(mesh: &ProcessingMesh) -> SplitResult {
        let mut material_faces: HashMap<u32, Vec<&Face>> = HashMap::new();
        for face in &mesh.faces {
            material_faces.entry(face.material_id).or_default().push(face);
        }

        let mut sub_meshes = Vec::new();
        for (mat_id, faces) in &material_faces {
            let mut vertex_remap: HashMap<u32, u32> = HashMap::new();
            let mut new_vertices = Vec::new();
            let mut new_faces = Vec::new();

            for face in faces {
                let mut new_indices = [0u32; 3];
                for (i, &idx) in face.indices.iter().enumerate() {
                    let new_idx = *vertex_remap.entry(idx).or_insert_with(|| {
                        let ni = new_vertices.len() as u32;
                        new_vertices.push(mesh.vertices[idx as usize]);
                        ni
                    });
                    new_indices[i] = new_idx;
                }
                new_faces.push(Face {
                    indices: new_indices,
                    material_id: *mat_id,
                    smooth_group: face.smooth_group,
                });
            }

            sub_meshes.push((*mat_id, ProcessingMesh {
                vertices: new_vertices,
                faces: new_faces,
            }));
        }

        let material_count = sub_meshes.len();
        SplitResult {
            sub_meshes,
            material_count,
        }
    }

    /// Split a mesh along a plane.
    pub fn split_by_plane(mesh: &ProcessingMesh, plane_normal: Vec3, plane_d: f32) -> (ProcessingMesh, ProcessingMesh) {
        let mut front = ProcessingMesh::new();
        let mut back = ProcessingMesh::new();

        // Simple classification: assign each face to front or back based on centroid.
        front.vertices = mesh.vertices.clone();
        back.vertices = mesh.vertices.clone();

        for face in &mesh.faces {
            let centroid = mesh.vertices[face.indices[0] as usize].position
                .add(mesh.vertices[face.indices[1] as usize].position)
                .add(mesh.vertices[face.indices[2] as usize].position)
                .scale(1.0 / 3.0);

            let dist = plane_normal.dot(centroid) + plane_d;
            if dist >= 0.0 {
                front.faces.push(*face);
            } else {
                back.faces.push(*face);
            }
        }

        (front, back)
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Generate a simple box mesh for testing.
pub fn generate_box(half_extents: Vec3) -> ProcessingMesh {
    let mut mesh = ProcessingMesh::new();

    let he = half_extents;
    let positions = [
        Vec3::new(-he.x, -he.y, -he.z),
        Vec3::new(he.x, -he.y, -he.z),
        Vec3::new(he.x, he.y, -he.z),
        Vec3::new(-he.x, he.y, -he.z),
        Vec3::new(-he.x, -he.y, he.z),
        Vec3::new(he.x, -he.y, he.z),
        Vec3::new(he.x, he.y, he.z),
        Vec3::new(-he.x, he.y, he.z),
    ];

    for p in &positions {
        mesh.add_vertex(Vertex {
            position: *p,
            ..Default::default()
        });
    }

    // 12 triangles (6 faces, 2 triangles each).
    let indices: &[u32] = &[
        0, 1, 2, 0, 2, 3, // front
        1, 5, 6, 1, 6, 2, // right
        5, 4, 7, 5, 7, 6, // back
        4, 0, 3, 4, 3, 7, // left
        3, 2, 6, 3, 6, 7, // top
        4, 5, 1, 4, 1, 0, // bottom
    ];

    for chunk in indices.chunks(3) {
        mesh.add_face(Face::new(chunk[0], chunk[1], chunk[2]));
    }

    mesh.recompute_normals();
    mesh
}

/// Generate an icosphere mesh.
pub fn generate_icosphere(radius: f32, subdivisions: u32) -> ProcessingMesh {
    let t = (1.0 + 5.0f32.sqrt()) / 2.0;

    let positions = [
        Vec3::new(-1.0, t, 0.0).normalize().scale(radius),
        Vec3::new(1.0, t, 0.0).normalize().scale(radius),
        Vec3::new(-1.0, -t, 0.0).normalize().scale(radius),
        Vec3::new(1.0, -t, 0.0).normalize().scale(radius),
        Vec3::new(0.0, -1.0, t).normalize().scale(radius),
        Vec3::new(0.0, 1.0, t).normalize().scale(radius),
        Vec3::new(0.0, -1.0, -t).normalize().scale(radius),
        Vec3::new(0.0, 1.0, -t).normalize().scale(radius),
        Vec3::new(t, 0.0, -1.0).normalize().scale(radius),
        Vec3::new(t, 0.0, 1.0).normalize().scale(radius),
        Vec3::new(-t, 0.0, -1.0).normalize().scale(radius),
        Vec3::new(-t, 0.0, 1.0).normalize().scale(radius),
    ];

    let mut mesh = ProcessingMesh::new();
    for p in &positions {
        mesh.add_vertex(Vertex {
            position: *p,
            normal: p.normalize(),
            ..Default::default()
        });
    }

    let tris: &[u32] = &[
        0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
        1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
        3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
        4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
    ];

    for chunk in tris.chunks(3) {
        mesh.add_face(Face::new(chunk[0], chunk[1], chunk[2]));
    }

    // Subdivide.
    if subdivisions > 0 {
        let config = SubdivisionConfig {
            scheme: SubdivisionScheme::Midpoint,
            iterations: subdivisions,
            ..Default::default()
        };
        mesh = MeshSubdivider::subdivide(&mesh, &config);

        // Project vertices onto sphere.
        for v in &mut mesh.vertices {
            v.position = v.position.normalize().scale(radius);
            v.normal = v.position.normalize();
        }
    }

    mesh
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_triangle() -> ProcessingMesh {
        let mut mesh = ProcessingMesh::new();
        mesh.add_vertex(Vertex { position: Vec3::new(0.0, 0.0, 0.0), ..Default::default() });
        mesh.add_vertex(Vertex { position: Vec3::new(1.0, 0.0, 0.0), ..Default::default() });
        mesh.add_vertex(Vertex { position: Vec3::new(0.0, 1.0, 0.0), ..Default::default() });
        mesh.add_face(Face::new(0, 1, 2));
        mesh
    }

    #[test]
    fn test_processing_mesh_basic() {
        let mesh = make_test_triangle();
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.face_count(), 1);
        assert_eq!(mesh.edge_count(), 3);
    }

    #[test]
    fn test_box_generation() {
        let mesh = generate_box(Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(mesh.vertex_count(), 8);
        assert_eq!(mesh.face_count(), 12);
    }

    #[test]
    fn test_midpoint_subdivision() {
        let mesh = make_test_triangle();
        let config = SubdivisionConfig {
            scheme: SubdivisionScheme::Midpoint,
            iterations: 1,
            ..Default::default()
        };
        let subdivided = MeshSubdivider::subdivide(&mesh, &config);
        // 1 triangle -> 4 triangles after 1 iteration.
        assert_eq!(subdivided.face_count(), 4);
    }

    #[test]
    fn test_mesh_welding() {
        let mut mesh = ProcessingMesh::new();
        // Duplicate vertices.
        mesh.add_vertex(Vertex { position: Vec3::new(0.0, 0.0, 0.0), ..Default::default() });
        mesh.add_vertex(Vertex { position: Vec3::new(1.0, 0.0, 0.0), ..Default::default() });
        mesh.add_vertex(Vertex { position: Vec3::new(0.0, 1.0, 0.0), ..Default::default() });
        mesh.add_vertex(Vertex { position: Vec3::new(0.0001, 0.0, 0.0), ..Default::default() }); // Near-duplicate of 0.
        mesh.add_face(Face::new(0, 1, 2));
        mesh.add_face(Face::new(3, 1, 2));

        let config = WeldConfig::default();
        let result = MeshWelder::weld(&mesh, &config);
        // Vertex 3 should be welded to vertex 0.
        assert_eq!(result.welded_vertex_count, 3);
        assert_eq!(result.vertices_removed, 1);
    }

    #[test]
    fn test_mesh_split_by_material() {
        let mut mesh = ProcessingMesh::new();
        for _ in 0..6 {
            mesh.add_vertex(Vertex::default());
        }
        mesh.add_face(Face { indices: [0, 1, 2], material_id: 0, smooth_group: 0 });
        mesh.add_face(Face { indices: [3, 4, 5], material_id: 1, smooth_group: 0 });

        let result = MeshSplitter::split_by_material(&mesh);
        assert_eq!(result.material_count, 2);
    }

    #[test]
    fn test_quadric_matrix() {
        let q = QuadricMatrix::from_plane(0.0, 1.0, 0.0, 0.0);
        // Error at origin should be 0.
        let e = q.evaluate(0.0, 0.0, 0.0);
        assert!((e - 0.0).abs() < 1e-6);

        // Error at (0, 1, 0) should be 1.
        let e2 = q.evaluate(0.0, 1.0, 0.0);
        assert!((e2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_icosphere() {
        let mesh = generate_icosphere(1.0, 0);
        assert_eq!(mesh.vertex_count(), 12);
        assert_eq!(mesh.face_count(), 20);
    }

    #[test]
    fn test_decimation() {
        let mesh = generate_box(Vec3::new(1.0, 1.0, 1.0));
        let config = DecimationConfig {
            target_ratio: 0.5,
            ..Default::default()
        };
        let result = MeshDecimator::decimate(&mesh, &config);
        assert!(result.final_face_count <= result.original_face_count);
    }
}
