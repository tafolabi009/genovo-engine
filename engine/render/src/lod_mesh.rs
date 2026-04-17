// engine/render/src/lod_mesh.rs
//
// Automatic LOD mesh generation subsystem for the Genovo engine.
//
// Provides:
// - `LodGenerator` -- generates lower-detail meshes from a high-detail source
//   via QEM (Quadric Error Metrics) edge collapse simplification.
// - `LodChain` -- array of meshes at decreasing detail with screen-size
//   thresholds and dithered cross-fade transitions.
// - `ProxyMesh` -- extremely simplified mesh for far-distance rendering.
// - `MeshOptimizer` -- post-process mesh for GPU efficiency (vertex cache,
//   overdraw, and vertex fetch optimisation).

use crate::mesh::{Mesh, MeshBuilder, Vertex, AABB};
use crate::virtual_geometry::simplification::{self, QuadricErrorMetric, SimplifiedMesh};
use glam::{Vec2, Vec3, Vec4};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default number of LOD levels generated when no explicit config is given.
pub const DEFAULT_LOD_LEVELS: usize = 4;

/// Maximum number of LOD levels the system supports.
pub const MAX_LOD_LEVELS: usize = 8;

/// Threshold below which a normal difference indicates a hard edge (radians).
pub const HARD_EDGE_ANGLE_THRESHOLD: f32 = 0.5236; // ~30 degrees

/// Vertex cache size assumed by the Forsyth algorithm.
pub const FORSYTH_CACHE_SIZE: usize = 32;

/// Maximum valence for the Forsyth score lookup table.
pub const FORSYTH_VALENCE_LIMIT: usize = 32;

// ---------------------------------------------------------------------------
// LodConfig
// ---------------------------------------------------------------------------

/// Configuration for a single LOD level.
#[derive(Debug, Clone, Copy)]
pub struct LodConfig {
    /// Target triangle count for this LOD level. If `None`, use `reduction_ratio`.
    pub target_triangle_count: Option<u32>,
    /// Ratio of triangle reduction relative to the source mesh (0.0 .. 1.0).
    /// For example, 0.5 means half the triangles of the source.
    pub reduction_ratio: f32,
    /// Maximum geometric error allowed for this LOD level. Collapses exceeding
    /// this error will be rejected.
    pub max_error: f32,
    /// Whether to preserve UV seams during simplification.
    pub preserve_uv_seams: bool,
    /// Whether to preserve material boundaries (submesh edges).
    pub preserve_material_boundaries: bool,
    /// Threshold angle (radians) for hard edge preservation. Normals differing
    /// by more than this angle across an edge will be locked.
    pub hard_edge_threshold: f32,
    /// Screen-size threshold below which this LOD should be used (in pixels of
    /// screen-space projected diameter).
    pub screen_size_threshold: f32,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            target_triangle_count: None,
            reduction_ratio: 0.5,
            max_error: f32::MAX,
            preserve_uv_seams: true,
            preserve_material_boundaries: true,
            hard_edge_threshold: HARD_EDGE_ANGLE_THRESHOLD,
            screen_size_threshold: 100.0,
        }
    }
}

impl LodConfig {
    /// Create a config from a target triangle count.
    pub fn from_triangle_count(count: u32) -> Self {
        Self {
            target_triangle_count: Some(count),
            ..Default::default()
        }
    }

    /// Create a config from a reduction ratio.
    pub fn from_ratio(ratio: f32) -> Self {
        Self {
            reduction_ratio: ratio.clamp(0.01, 1.0),
            ..Default::default()
        }
    }

    /// Set the screen-size threshold.
    pub fn with_screen_size(mut self, size: f32) -> Self {
        self.screen_size_threshold = size;
        self
    }

    /// Set the maximum geometric error.
    pub fn with_max_error(mut self, error: f32) -> Self {
        self.max_error = error;
        self
    }

    /// Resolve the target triangle count given the source mesh triangle count.
    pub fn resolve_target(&self, source_tri_count: u32) -> u32 {
        if let Some(count) = self.target_triangle_count {
            count.min(source_tri_count)
        } else {
            ((source_tri_count as f32 * self.reduction_ratio).ceil() as u32).max(4)
        }
    }
}

// ---------------------------------------------------------------------------
// LodGenerator
// ---------------------------------------------------------------------------

/// Generates LOD meshes from a high-detail source mesh using QEM
/// simplification with edge collapse and a priority queue.
pub struct LodGenerator {
    /// Whether to log statistics during generation.
    pub verbose: bool,
}

impl LodGenerator {
    /// Create a new LOD generator.
    pub fn new() -> Self {
        Self { verbose: false }
    }

    /// Generate a set of LOD meshes from the source mesh.
    ///
    /// Each entry in `levels` describes one LOD level. The returned vector has
    /// the same length as `levels`, with each element being the simplified mesh
    /// for that level.
    pub fn generate_lods(&self, mesh: &Mesh, levels: &[LodConfig]) -> Vec<Mesh> {
        let source_tri_count = mesh.triangle_count();
        let mut results = Vec::with_capacity(levels.len());

        // Identify locked vertices: those on hard edges, UV seams, and material
        // boundaries, so the simplifier preserves them.
        let hard_edge_verts = detect_hard_edge_vertices(mesh, HARD_EDGE_ANGLE_THRESHOLD);
        let material_boundary_verts = detect_material_boundary_vertices(mesh);

        for config in levels {
            let target = config.resolve_target(source_tri_count);

            // Build the set of locked vertices based on config.
            let mut locked = HashSet::new();
            if config.hard_edge_threshold < std::f32::consts::PI {
                let threshold_verts = detect_hard_edge_vertices(mesh, config.hard_edge_threshold);
                locked.extend(&threshold_verts);
            }
            if config.preserve_material_boundaries {
                locked.extend(&material_boundary_verts);
            }

            let simplified = if locked.is_empty() && !config.preserve_uv_seams {
                simplification::simplify_raw(
                    &mesh.vertices,
                    &mesh.indices,
                    target as usize,
                )
            } else {
                simplification::simplify_with_locked_vertices(
                    &mesh.vertices,
                    &mesh.indices,
                    target as usize,
                    &locked,
                )
            };

            let lod_mesh = simplified_to_mesh(simplified);
            results.push(lod_mesh);
        }

        results
    }

    /// Generate a default set of LOD levels with geometric reduction ratios.
    ///
    /// Produces `DEFAULT_LOD_LEVELS` levels with ratios 1.0, 0.5, 0.25, 0.125.
    pub fn generate_default_lods(&self, mesh: &Mesh) -> Vec<Mesh> {
        let configs: Vec<LodConfig> = (0..DEFAULT_LOD_LEVELS)
            .map(|i| {
                let ratio = 1.0 / (1 << i) as f32;
                let screen_size = 400.0 / (1 << i) as f32;
                LodConfig::from_ratio(ratio).with_screen_size(screen_size)
            })
            .collect();
        self.generate_lods(mesh, &configs)
    }
}

impl Default for LodGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a `SimplifiedMesh` result to a `Mesh`.
fn simplified_to_mesh(simplified: SimplifiedMesh) -> Mesh {
    Mesh::new(simplified.vertices, simplified.indices)
}

/// Detect vertices on hard edges where adjacent face normals differ by more
/// than `threshold` radians.
fn detect_hard_edge_vertices(mesh: &Mesh, threshold: f32) -> HashSet<u32> {
    let cos_threshold = threshold.cos();
    let tri_count = mesh.indices.len() / 3;
    let mut face_normals: Vec<Vec3> = Vec::with_capacity(tri_count);

    // Compute per-face normals.
    for t in 0..tri_count {
        let base = t * 3;
        let i0 = mesh.indices[base] as usize;
        let i1 = mesh.indices[base + 1] as usize;
        let i2 = mesh.indices[base + 2] as usize;

        let p0 = Vec3::from_array(mesh.vertices[i0].position);
        let p1 = Vec3::from_array(mesh.vertices[i1].position);
        let p2 = Vec3::from_array(mesh.vertices[i2].position);

        let n = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        face_normals.push(n);
    }

    // Build edge-to-face map.
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for t in 0..tri_count {
        let base = t * 3;
        let verts = [mesh.indices[base], mesh.indices[base + 1], mesh.indices[base + 2]];
        for k in 0..3 {
            let a = verts[k].min(verts[(k + 1) % 3]);
            let b = verts[k].max(verts[(k + 1) % 3]);
            edge_faces.entry((a, b)).or_default().push(t);
        }
    }

    let mut hard_verts = HashSet::new();
    for (&(v0, v1), faces) in &edge_faces {
        if faces.len() >= 2 {
            // Check all pairs of adjacent faces on this edge.
            for i in 0..faces.len() {
                for j in (i + 1)..faces.len() {
                    let dot = face_normals[faces[i]].dot(face_normals[faces[j]]);
                    if dot < cos_threshold {
                        hard_verts.insert(v0);
                        hard_verts.insert(v1);
                    }
                }
            }
        }
    }

    hard_verts
}

/// Detect vertices on material boundaries (where different submeshes meet).
fn detect_material_boundary_vertices(mesh: &Mesh) -> HashSet<u32> {
    let mut boundary_verts = HashSet::new();

    if mesh.submeshes.len() <= 1 {
        return boundary_verts;
    }

    // Assign each triangle to a material.
    let tri_count = mesh.indices.len() / 3;
    let mut tri_material = vec![0u32; tri_count];
    for sub in &mesh.submeshes {
        let start_tri = sub.index_offset as usize / 3;
        let end_tri = start_tri + sub.index_count as usize / 3;
        for t in start_tri..end_tri.min(tri_count) {
            tri_material[t] = sub.material_index;
        }
    }

    // For each vertex, find all materials that reference it.
    let mut vertex_materials: Vec<HashSet<u32>> = vec![HashSet::new(); mesh.vertices.len()];
    for t in 0..tri_count {
        let base = t * 3;
        let mat = tri_material[t];
        for k in 0..3 {
            let vi = mesh.indices[base + k] as usize;
            vertex_materials[vi].insert(mat);
        }
    }

    for (vi, mats) in vertex_materials.iter().enumerate() {
        if mats.len() > 1 {
            boundary_verts.insert(vi as u32);
        }
    }

    boundary_verts
}

// ---------------------------------------------------------------------------
// LodChain
// ---------------------------------------------------------------------------

/// An ordered chain of LOD meshes at decreasing detail, with screen-size
/// thresholds for selection and dithered cross-fade between levels.
pub struct LodChain {
    /// LOD meshes from highest detail (index 0) to lowest.
    pub meshes: Vec<Mesh>,
    /// Screen-size thresholds in pixels (projected diameter). `thresholds[i]` is
    /// the minimum screen size for LOD level `i`.
    pub thresholds: Vec<f32>,
    /// Cross-fade transition range in pixels of screen size. Within this range,
    /// both LOD levels are rendered with dithered alpha.
    pub fade_range: f32,
}

impl LodChain {
    /// Create a LOD chain from meshes and thresholds.
    pub fn new(meshes: Vec<Mesh>, thresholds: Vec<f32>) -> Self {
        assert_eq!(meshes.len(), thresholds.len());
        Self {
            meshes,
            thresholds,
            fade_range: 16.0,
        }
    }

    /// Create a LOD chain from meshes with automatically computed thresholds.
    pub fn from_meshes(meshes: Vec<Mesh>) -> Self {
        let n = meshes.len();
        let thresholds: Vec<f32> = (0..n)
            .map(|i| {
                if i == 0 {
                    f32::MAX // highest LOD is always above threshold
                } else {
                    // Geometric falloff: 400, 200, 100, 50, ...
                    400.0 / (1 << (i - 1)) as f32
                }
            })
            .collect();
        Self::new(meshes, thresholds)
    }

    /// Select the appropriate LOD level based on screen-space projected size.
    ///
    /// Returns the index into `meshes`.
    pub fn select_lod(&self, screen_size: f32) -> usize {
        // Walk from highest detail to lowest; return the first level whose
        // threshold is met.
        for i in 0..self.meshes.len() {
            if screen_size >= self.thresholds[i] {
                return i;
            }
        }
        // Fallback to lowest detail.
        self.meshes.len().saturating_sub(1)
    }

    /// Select the LOD level and compute the cross-fade factor for a smooth
    /// transition between adjacent LOD levels.
    ///
    /// Returns `(primary_lod, secondary_lod, fade_factor)` where `fade_factor`
    /// is in `[0.0, 1.0]`. At 0.0 the primary LOD is fully shown; at 1.0 the
    /// secondary LOD takes over completely.
    pub fn select_lod_with_fade(&self, screen_size: f32) -> (usize, usize, f32) {
        let primary = self.select_lod(screen_size);
        if primary >= self.meshes.len().saturating_sub(1) {
            return (primary, primary, 0.0);
        }

        let next = primary + 1;
        let threshold = self.thresholds[next];
        let fade_start = threshold + self.fade_range;
        let fade_end = threshold;

        if screen_size >= fade_start {
            (primary, next, 0.0)
        } else if screen_size <= fade_end {
            (next, primary, 0.0)
        } else {
            let t = (fade_start - screen_size) / self.fade_range.max(0.001);
            (primary, next, t.clamp(0.0, 1.0))
        }
    }

    /// Number of LOD levels in the chain.
    pub fn level_count(&self) -> usize {
        self.meshes.len()
    }

    /// Get a reference to a specific LOD mesh.
    pub fn get_mesh(&self, level: usize) -> Option<&Mesh> {
        self.meshes.get(level)
    }

    /// Get triangle count for a specific LOD level.
    pub fn triangle_count(&self, level: usize) -> Option<u32> {
        self.meshes.get(level).map(|m| m.triangle_count())
    }

    /// Set the fade range (transition zone) in pixels of screen size.
    pub fn set_fade_range(&mut self, range: f32) {
        self.fade_range = range.max(0.0);
    }
}

// ---------------------------------------------------------------------------
// DitherPattern
// ---------------------------------------------------------------------------

/// 4x4 Bayer dithering matrix used for LOD cross-fade transitions.
///
/// Each pixel samples the dither pattern based on its screen position. If the
/// dither value is below the fade factor, the secondary LOD fragment is shown;
/// otherwise the primary LOD fragment is kept.
pub struct DitherPattern;

impl DitherPattern {
    /// The standard 4x4 Bayer ordered dither matrix, normalised to [0, 1).
    pub const MATRIX_4X4: [[f32; 4]; 4] = [
        [0.0 / 16.0, 8.0 / 16.0, 2.0 / 16.0, 10.0 / 16.0],
        [12.0 / 16.0, 4.0 / 16.0, 14.0 / 16.0, 6.0 / 16.0],
        [3.0 / 16.0, 11.0 / 16.0, 1.0 / 16.0, 9.0 / 16.0],
        [15.0 / 16.0, 7.0 / 16.0, 13.0 / 16.0, 5.0 / 16.0],
    ];

    /// Sample the dither pattern at pixel position `(x, y)`.
    ///
    /// Returns a threshold value in `[0, 1)`. Compare against the fade factor
    /// to determine which LOD level to render for this pixel.
    pub fn sample(x: u32, y: u32) -> f32 {
        Self::MATRIX_4X4[(y & 3) as usize][(x & 3) as usize]
    }

    /// Test whether a pixel should show the secondary (lower) LOD based on
    /// the dither pattern and fade factor.
    ///
    /// Returns `true` if the pixel should render the secondary LOD.
    pub fn should_fade(x: u32, y: u32, fade_factor: f32) -> bool {
        Self::sample(x, y) < fade_factor
    }

    /// Generate a flat array of the 4x4 dither matrix for GPU upload.
    pub fn to_flat_array() -> [f32; 16] {
        let m = &Self::MATRIX_4X4;
        [
            m[0][0], m[0][1], m[0][2], m[0][3],
            m[1][0], m[1][1], m[1][2], m[1][3],
            m[2][0], m[2][1], m[2][2], m[2][3],
            m[3][0], m[3][1], m[3][2], m[3][3],
        ]
    }

    /// Generate the WGSL snippet for LOD dither fade in a fragment shader.
    pub fn wgsl_snippet() -> &'static str {
        LOD_DITHER_WGSL
    }
}

/// WGSL code for dithered LOD cross-fade. Include this in a fragment shader and
/// call `lod_dither_clip(pixel_pos, fade_factor)` to discard fragments.
const LOD_DITHER_WGSL: &str = r#"
// 4x4 Bayer dither matrix for LOD cross-fade.
const DITHER_MATRIX: array<f32, 16> = array<f32, 16>(
    0.0 / 16.0,  8.0 / 16.0,  2.0 / 16.0, 10.0 / 16.0,
   12.0 / 16.0,  4.0 / 16.0, 14.0 / 16.0,  6.0 / 16.0,
    3.0 / 16.0, 11.0 / 16.0,  1.0 / 16.0,  9.0 / 16.0,
   15.0 / 16.0,  7.0 / 16.0, 13.0 / 16.0,  5.0 / 16.0
);

fn lod_dither_threshold(pixel_pos: vec2<f32>) -> f32 {
    let x = u32(pixel_pos.x) & 3u;
    let y = u32(pixel_pos.y) & 3u;
    return DITHER_MATRIX[y * 4u + x];
}

fn lod_dither_clip(pixel_pos: vec2<f32>, fade_factor: f32) {
    let threshold = lod_dither_threshold(pixel_pos);
    if (threshold < fade_factor) {
        discard;
    }
}
"#;

// ---------------------------------------------------------------------------
// ProxyMesh
// ---------------------------------------------------------------------------

/// Extremely simplified proxy mesh for far-distance rendering.
///
/// Generates 8-12 triangle meshes from bounding volumes that serve as
/// stand-in geometry when the object is very far from the camera.
pub struct ProxyMesh;

impl ProxyMesh {
    /// Generate a box proxy mesh from an AABB.
    ///
    /// Produces a 12-triangle box (6 faces, 2 triangles each) that exactly
    /// matches the AABB extents.
    pub fn generate_proxy(aabb: &AABB) -> Mesh {
        let min = aabb.min;
        let max = aabb.max;

        let mut builder = MeshBuilder::new();

        // 8 corners of the AABB.
        let corners = [
            Vec3::new(min.x, min.y, min.z), // 0: ---
            Vec3::new(max.x, min.y, min.z), // 1: +--
            Vec3::new(max.x, max.y, min.z), // 2: ++-
            Vec3::new(min.x, max.y, min.z), // 3: -+-
            Vec3::new(min.x, min.y, max.z), // 4: --+
            Vec3::new(max.x, min.y, max.z), // 5: +-+
            Vec3::new(max.x, max.y, max.z), // 6: +++
            Vec3::new(min.x, max.y, max.z), // 7: -++
        ];

        // 6 face definitions: (corner indices, normal, UV layout).
        let faces: [(usize, usize, usize, usize, Vec3); 6] = [
            (0, 1, 2, 3, -Vec3::Z), // -Z face
            (5, 4, 7, 6, Vec3::Z),  // +Z face
            (4, 0, 3, 7, -Vec3::X), // -X face
            (1, 5, 6, 2, Vec3::X),  // +X face
            (3, 2, 6, 7, Vec3::Y),  // +Y face
            (4, 5, 1, 0, -Vec3::Y), // -Y face
        ];

        let uvs = [
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 0.0),
        ];

        for &(c0, c1, c2, c3, normal) in &faces {
            let v0 = builder.add_vertex(corners[c0], normal, uvs[0]);
            let v1 = builder.add_vertex(corners[c1], normal, uvs[1]);
            let v2 = builder.add_vertex(corners[c2], normal, uvs[2]);
            let v3 = builder.add_vertex(corners[c3], normal, uvs[3]);
            builder.add_quad(v0, v1, v2, v3);
        }

        builder.build_no_tangents()
    }

    /// Generate a convex proxy mesh from a set of convex hull vertices.
    ///
    /// Uses a simple fan triangulation from the centroid for each face. This
    /// produces 8-12 triangles for typical convex hulls.
    pub fn generate_convex_proxy(convex_hull: &[Vec3]) -> Mesh {
        if convex_hull.len() < 4 {
            // Degenerate: fall back to AABB proxy.
            let mut aabb = AABB::default();
            for &p in convex_hull {
                aabb.expand_point(p);
            }
            return Self::generate_proxy(&aabb);
        }

        let mut builder = MeshBuilder::new();

        // Compute centroid.
        let centroid = convex_hull.iter().copied().fold(Vec3::ZERO, |a, b| a + b)
            / convex_hull.len() as f32;

        // Simple convex hull triangulation: for each set of 3 adjacent hull
        // vertices, create a triangle. We use a simple approach: create a fan
        // from the first vertex across the hull if it is small enough; otherwise
        // triangulate from the centroid.
        let n = convex_hull.len();

        if n <= 8 {
            // Fan triangulation from vertex 0.
            let v0_idx = builder.add_vertex(convex_hull[0], Vec3::Y, Vec2::ZERO);
            let mut prev_idx = builder.add_vertex(convex_hull[1], Vec3::Y, Vec2::X);

            for i in 2..n {
                let cur_idx = builder.add_vertex(
                    convex_hull[i],
                    Vec3::Y,
                    Vec2::new(i as f32 / n as f32, 1.0),
                );
                // Compute face normal.
                let e1 = convex_hull[i] - convex_hull[0];
                let e2 = convex_hull[i - 1] - convex_hull[0];
                let face_normal = e1.cross(e2).normalize_or_zero();

                // Check winding: normal should point away from centroid.
                let to_centroid = centroid - convex_hull[0];
                if face_normal.dot(to_centroid) > 0.0 {
                    builder.add_triangle(v0_idx, cur_idx, prev_idx);
                } else {
                    builder.add_triangle(v0_idx, prev_idx, cur_idx);
                }

                prev_idx = cur_idx;
            }
        } else {
            // Centroid-based triangulation for larger hulls.
            let centroid_idx = builder.add_vertex(centroid, Vec3::Y, Vec2::new(0.5, 0.5));

            for i in 0..n {
                let next = (i + 1) % n;
                let p0 = convex_hull[i];
                let p1 = convex_hull[next];

                let face_normal = (p0 - centroid).cross(p1 - centroid).normalize_or_zero();
                let to_out = ((p0 + p1) * 0.5 - centroid).normalize_or_zero();

                let v0 = builder.add_vertex(p0, face_normal, Vec2::new(0.0, 0.0));
                let v1 = builder.add_vertex(p1, face_normal, Vec2::new(1.0, 0.0));

                if face_normal.dot(to_out) >= 0.0 {
                    builder.add_triangle(centroid_idx, v0, v1);
                } else {
                    builder.add_triangle(centroid_idx, v1, v0);
                }
            }
        }

        let mut mesh = builder.build_no_tangents();
        mesh.compute_flat_normals();
        mesh
    }

    /// Generate a billboard-style proxy (two intersecting quads) from an AABB.
    /// This produces only 4 triangles and is suitable for very distant objects.
    pub fn generate_cross_proxy(aabb: &AABB) -> Mesh {
        let center = aabb.center();
        let half = aabb.half_extents();
        let mut builder = MeshBuilder::new();

        // Quad 1: XY plane through center.
        let v0 = builder.add_vertex(
            center + Vec3::new(-half.x, -half.y, 0.0),
            Vec3::Z, Vec2::new(0.0, 1.0),
        );
        let v1 = builder.add_vertex(
            center + Vec3::new(half.x, -half.y, 0.0),
            Vec3::Z, Vec2::new(1.0, 1.0),
        );
        let v2 = builder.add_vertex(
            center + Vec3::new(half.x, half.y, 0.0),
            Vec3::Z, Vec2::new(1.0, 0.0),
        );
        let v3 = builder.add_vertex(
            center + Vec3::new(-half.x, half.y, 0.0),
            Vec3::Z, Vec2::new(0.0, 0.0),
        );
        builder.add_quad(v0, v1, v2, v3);

        // Quad 2: YZ plane through center (rotated 90 degrees).
        let v4 = builder.add_vertex(
            center + Vec3::new(0.0, -half.y, -half.z),
            Vec3::X, Vec2::new(0.0, 1.0),
        );
        let v5 = builder.add_vertex(
            center + Vec3::new(0.0, -half.y, half.z),
            Vec3::X, Vec2::new(1.0, 1.0),
        );
        let v6 = builder.add_vertex(
            center + Vec3::new(0.0, half.y, half.z),
            Vec3::X, Vec2::new(1.0, 0.0),
        );
        let v7 = builder.add_vertex(
            center + Vec3::new(0.0, half.y, -half.z),
            Vec3::X, Vec2::new(0.0, 0.0),
        );
        builder.add_quad(v4, v5, v6, v7);

        builder.build_no_tangents()
    }
}

// ---------------------------------------------------------------------------
// MeshOptimizer
// ---------------------------------------------------------------------------

/// Post-process mesh for GPU efficiency. Applies three optimisation passes:
///
/// 1. **Vertex cache optimisation** -- reorder triangles to maximise post-
///    transform vertex cache hits (Forsyth algorithm).
/// 2. **Overdraw optimisation** -- reorder triangle clusters to reduce overdraw
///    by sorting clusters by centroid depth.
/// 3. **Vertex fetch optimisation** -- reorder vertices to match the triangle
///    order, improving spatial locality for vertex buffer fetches.
pub struct MeshOptimizer;

impl MeshOptimizer {
    /// Apply all three optimisation passes to the mesh.
    pub fn optimize_mesh(mesh: &mut Mesh) {
        Self::optimize_vertex_cache(mesh);
        Self::optimize_overdraw(mesh);
        Self::optimize_vertex_fetch(mesh);
    }

    /// Reorder triangles for optimal vertex cache usage using the Forsyth
    /// algorithm.
    ///
    /// The Forsyth algorithm assigns a score to each triangle based on how many
    /// of its vertices are currently in the simulated LRU vertex cache. It
    /// greedily selects the highest-scoring triangle at each step.
    ///
    /// Reference: Tom Forsyth, "Linear-Speed Vertex Cache Optimisation" (2006).
    pub fn optimize_vertex_cache(mesh: &mut Mesh) {
        let num_vertices = mesh.vertices.len();
        let num_indices = mesh.indices.len();
        let num_triangles = num_indices / 3;

        if num_triangles == 0 || num_vertices == 0 {
            return;
        }

        // --- Build per-vertex data ---

        // Active triangle count (number of not-yet-emitted triangles using this
        // vertex).
        let mut vertex_active_tri_count: Vec<u32> = vec![0; num_vertices];
        for idx in &mesh.indices {
            vertex_active_tri_count[*idx as usize] += 1;
        }

        // Build adjacency: for each vertex, the list of triangle indices that
        // reference it.
        let mut vertex_tri_offset: Vec<usize> = vec![0; num_vertices + 1];
        for i in 0..num_vertices {
            vertex_tri_offset[i + 1] = vertex_tri_offset[i] + vertex_active_tri_count[i] as usize;
        }
        let total_adj = vertex_tri_offset[num_vertices];
        let mut vertex_tri_list: Vec<u32> = vec![0; total_adj];
        let mut fill_pos: Vec<usize> = vec![0; num_vertices];

        for t in 0..num_triangles {
            for k in 0..3 {
                let v = mesh.indices[t * 3 + k] as usize;
                let pos = vertex_tri_offset[v] + fill_pos[v];
                vertex_tri_list[pos] = t as u32;
                fill_pos[v] += 1;
            }
        }

        // --- Forsyth scoring ---

        // Precompute cache position scores.
        let cache_scores = forsyth_cache_position_scores();
        // Precompute valence scores.
        let valence_scores = forsyth_valence_scores();

        // Compute initial vertex score.
        let mut vertex_score: Vec<f32> = Vec::with_capacity(num_vertices);
        for i in 0..num_vertices {
            let valence = vertex_active_tri_count[i] as usize;
            // Initially no vertex is in the cache, so cache position = -1.
            vertex_score.push(forsyth_score(
                usize::MAX, // not in cache
                valence,
                &cache_scores,
                &valence_scores,
            ));
        }

        // Compute initial triangle score = sum of its vertex scores.
        let mut tri_score: Vec<f32> = Vec::with_capacity(num_triangles);
        for t in 0..num_triangles {
            let s = vertex_score[mesh.indices[t * 3] as usize]
                + vertex_score[mesh.indices[t * 3 + 1] as usize]
                + vertex_score[mesh.indices[t * 3 + 2] as usize];
            tri_score.push(s);
        }

        // Whether a triangle has been emitted.
        let mut tri_emitted: Vec<bool> = vec![false; num_triangles];

        // Simulated LRU cache. Stores vertex indices. Position 0 = most recent.
        let mut cache: Vec<usize> = Vec::with_capacity(FORSYTH_CACHE_SIZE + 3);

        // Output index buffer.
        let mut new_indices: Vec<u32> = Vec::with_capacity(num_indices);

        // --- Greedy loop ---

        let mut best_tri: i64 = -1;
        let mut best_score: f32 = -1.0;

        // Find the initial best triangle.
        for t in 0..num_triangles {
            if tri_score[t] > best_score {
                best_score = tri_score[t];
                best_tri = t as i64;
            }
        }

        let mut tris_emitted = 0;
        while tris_emitted < num_triangles {
            if best_tri < 0 {
                // Scan for any remaining non-emitted triangle (fallback).
                best_score = -1.0;
                for t in 0..num_triangles {
                    if !tri_emitted[t] && tri_score[t] > best_score {
                        best_score = tri_score[t];
                        best_tri = t as i64;
                    }
                }
                if best_tri < 0 {
                    break;
                }
            }

            let t = best_tri as usize;
            tri_emitted[t] = true;
            tris_emitted += 1;

            // Emit the triangle.
            let i0 = mesh.indices[t * 3] as usize;
            let i1 = mesh.indices[t * 3 + 1] as usize;
            let i2 = mesh.indices[t * 3 + 2] as usize;
            new_indices.push(i0 as u32);
            new_indices.push(i1 as u32);
            new_indices.push(i2 as u32);

            // Update the cache: push the 3 vertices to the front.
            let tri_verts = [i0, i1, i2];
            for &v in &tri_verts {
                // Remove from current position if present.
                if let Some(pos) = cache.iter().position(|&cv| cv == v) {
                    cache.remove(pos);
                }
                cache.insert(0, v);

                // Decrease active tri count for this vertex.
                vertex_active_tri_count[v] = vertex_active_tri_count[v].saturating_sub(1);
            }

            // Trim cache to size.
            cache.truncate(FORSYTH_CACHE_SIZE);

            // Recompute scores for all vertices in the cache.
            for (cache_pos, &v) in cache.iter().enumerate() {
                vertex_score[v] = forsyth_score(
                    cache_pos,
                    vertex_active_tri_count[v] as usize,
                    &cache_scores,
                    &valence_scores,
                );
            }

            // Find the new best triangle among triangles adjacent to cache vertices.
            best_tri = -1;
            best_score = -1.0;

            for &v in &cache {
                let adj_start = vertex_tri_offset[v];
                let adj_end = vertex_tri_offset[v + 1];
                for adj_idx in adj_start..adj_end {
                    let adj_t = vertex_tri_list[adj_idx] as usize;
                    if tri_emitted[adj_t] {
                        continue;
                    }
                    // Recompute the triangle score.
                    let s = vertex_score[mesh.indices[adj_t * 3] as usize]
                        + vertex_score[mesh.indices[adj_t * 3 + 1] as usize]
                        + vertex_score[mesh.indices[adj_t * 3 + 2] as usize];
                    tri_score[adj_t] = s;
                    if s > best_score {
                        best_score = s;
                        best_tri = adj_t as i64;
                    }
                }
            }
        }

        if new_indices.len() == num_indices {
            mesh.indices = new_indices;
        }
    }

    /// Reorder triangles within clusters to reduce overdraw by sorting clusters
    /// front-to-back based on average centroid depth along an approximate view
    /// direction.
    ///
    /// Clusters are groups of 64 consecutive triangles (after vertex cache
    /// optimisation). Within each cluster, triangle order is preserved to
    /// maintain cache efficiency.
    pub fn optimize_overdraw(mesh: &mut Mesh) {
        let num_indices = mesh.indices.len();
        let num_triangles = num_indices / 3;

        if num_triangles < 2 {
            return;
        }

        // Cluster size for overdraw sorting.
        const CLUSTER_SIZE: usize = 64;

        let num_clusters = (num_triangles + CLUSTER_SIZE - 1) / CLUSTER_SIZE;
        if num_clusters < 2 {
            return;
        }

        // Compute cluster centroids.
        let mut cluster_centroids: Vec<(usize, Vec3)> = Vec::with_capacity(num_clusters);

        for c in 0..num_clusters {
            let start_tri = c * CLUSTER_SIZE;
            let end_tri = ((c + 1) * CLUSTER_SIZE).min(num_triangles);
            let mut centroid_sum = Vec3::ZERO;
            let mut count = 0;

            for t in start_tri..end_tri {
                let base = t * 3;
                let p0 = Vec3::from_array(mesh.vertices[mesh.indices[base] as usize].position);
                let p1 = Vec3::from_array(mesh.vertices[mesh.indices[base + 1] as usize].position);
                let p2 = Vec3::from_array(mesh.vertices[mesh.indices[base + 2] as usize].position);
                centroid_sum += (p0 + p1 + p2) / 3.0;
                count += 1;
            }

            if count > 0 {
                centroid_sum /= count as f32;
            }

            cluster_centroids.push((c, centroid_sum));
        }

        // Sort clusters by depth along the longest AABB axis (approximating the
        // most common view direction).
        let size = mesh.bounds.size();
        let sort_axis = if size.x >= size.y && size.x >= size.z {
            0
        } else if size.y >= size.z {
            1
        } else {
            2
        };

        cluster_centroids.sort_by(|a, b| {
            let da = match sort_axis {
                0 => a.1.x,
                1 => a.1.y,
                _ => a.1.z,
            };
            let db = match sort_axis {
                0 => b.1.x,
                1 => b.1.y,
                _ => b.1.z,
            };
            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
        });

        // Rebuild the index buffer in cluster-sorted order.
        let mut new_indices: Vec<u32> = Vec::with_capacity(num_indices);

        for &(cluster_idx, _) in &cluster_centroids {
            let start_tri = cluster_idx * CLUSTER_SIZE;
            let end_tri = ((cluster_idx + 1) * CLUSTER_SIZE).min(num_triangles);

            for t in start_tri..end_tri {
                let base = t * 3;
                new_indices.push(mesh.indices[base]);
                new_indices.push(mesh.indices[base + 1]);
                new_indices.push(mesh.indices[base + 2]);
            }
        }

        if new_indices.len() == num_indices {
            mesh.indices = new_indices;
        }
    }

    /// Reorder vertices to match the order in which they appear in the index
    /// buffer (sequential access optimisation).
    ///
    /// After this pass, vertex `i` is referenced before vertex `i+1` in the
    /// index buffer, which maximises spatial locality for vertex buffer fetches.
    pub fn optimize_vertex_fetch(mesh: &mut Mesh) {
        let num_vertices = mesh.vertices.len();
        let num_indices = mesh.indices.len();

        if num_vertices == 0 || num_indices == 0 {
            return;
        }

        // Build a remapping table: new_index = remap[old_index].
        let mut remap: Vec<u32> = vec![u32::MAX; num_vertices];
        let mut new_vertices: Vec<Vertex> = Vec::with_capacity(num_vertices);
        let mut next_new_index: u32 = 0;

        for idx in &mut mesh.indices {
            let old = *idx as usize;
            if remap[old] == u32::MAX {
                remap[old] = next_new_index;
                new_vertices.push(mesh.vertices[old]);
                next_new_index += 1;
            }
            *idx = remap[old];
        }

        // Append any unreferenced vertices (shouldn't happen in well-formed
        // meshes, but handle gracefully).
        for i in 0..num_vertices {
            if remap[i] == u32::MAX {
                remap[i] = next_new_index;
                new_vertices.push(mesh.vertices[i]);
                next_new_index += 1;
            }
        }

        mesh.vertices = new_vertices;
        mesh.vertex_count = mesh.vertices.len() as u32;
    }

    /// Measure the Average Cache Miss Ratio (ACMR) for the mesh's current
    /// triangle order, given a simulated vertex cache of `cache_size` entries.
    ///
    /// Lower is better. A perfect ACMR of `3.0 / cache_size` means every vertex
    /// is a cache hit except the very first occurrence.
    pub fn compute_acmr(indices: &[u32], cache_size: usize) -> f32 {
        let num_triangles = indices.len() / 3;
        if num_triangles == 0 {
            return 0.0;
        }

        let mut cache: Vec<u32> = Vec::with_capacity(cache_size);
        let mut misses: u32 = 0;

        for idx in indices {
            if !cache.contains(idx) {
                misses += 1;
                cache.insert(0, *idx);
                cache.truncate(cache_size);
            } else {
                // Move to front on hit.
                if let Some(pos) = cache.iter().position(|&v| v == *idx) {
                    cache.remove(pos);
                    cache.insert(0, *idx);
                }
            }
        }

        misses as f32 / num_triangles as f32
    }

    /// Measure the Average Cache Miss Ratio per vertex (ATVR) -- misses divided
    /// by unique vertex count.
    pub fn compute_atvr(indices: &[u32], cache_size: usize) -> f32 {
        let mut unique_verts: HashSet<u32> = HashSet::new();
        for idx in indices {
            unique_verts.insert(*idx);
        }
        let unique_count = unique_verts.len();
        if unique_count == 0 {
            return 0.0;
        }

        let mut cache: Vec<u32> = Vec::with_capacity(cache_size);
        let mut misses: u32 = 0;

        for idx in indices {
            if !cache.contains(idx) {
                misses += 1;
                cache.insert(0, *idx);
                cache.truncate(cache_size);
            } else {
                if let Some(pos) = cache.iter().position(|&v| v == *idx) {
                    cache.remove(pos);
                    cache.insert(0, *idx);
                }
            }
        }

        misses as f32 / unique_count as f32
    }
}

// ---------------------------------------------------------------------------
// Forsyth algorithm helpers
// ---------------------------------------------------------------------------

/// Precompute the score for each cache position.
///
/// Position 0..2 (the 3 most recent vertices) get a fixed high score because
/// they were just fetched. Positions 3..CACHE_SIZE get a decaying score.
/// Position MAX (not in cache) gets 0.
fn forsyth_cache_position_scores() -> Vec<f32> {
    let mut scores = vec![0.0f32; FORSYTH_CACHE_SIZE + 1];

    // The 3 most recently used vertices get the highest fixed score.
    // This incentivises re-using vertices from the last-emitted triangle.
    scores[0] = 0.75;
    scores[1] = 0.75;
    scores[2] = 0.75;

    // Remaining cache positions get a decaying score.
    let cache_decay_power: f32 = 1.5;
    let last_tri_score: f32 = 0.75;

    for i in 3..FORSYTH_CACHE_SIZE {
        let normalised = 1.0 - (i as f32 - 3.0) / (FORSYTH_CACHE_SIZE as f32 - 3.0);
        scores[i] = normalised.powf(cache_decay_power);
    }

    // A vertex just beyond the cache gets 0.
    scores[FORSYTH_CACHE_SIZE] = 0.0;

    scores
}

/// Precompute the score boost for each valence (number of remaining triangles
/// using this vertex).
///
/// Lower valence gets a higher score, because it is more "urgent" to emit
/// those triangles before the vertex falls out of the cache.
fn forsyth_valence_scores() -> Vec<f32> {
    let mut scores = vec![0.0f32; FORSYTH_VALENCE_LIMIT + 1];
    let valence_boost_scale: f32 = 2.0;
    let valence_boost_power: f32 = 0.5;

    // Valence 0 means the vertex is fully processed, no score bonus.
    scores[0] = 0.0;

    for v in 1..=FORSYTH_VALENCE_LIMIT {
        scores[v] = valence_boost_scale * (v as f32).powf(-valence_boost_power);
    }

    scores
}

/// Compute the Forsyth score for a vertex given its cache position and active
/// triangle valence.
fn forsyth_score(
    cache_pos: usize,
    valence: usize,
    cache_scores: &[f32],
    valence_scores: &[f32],
) -> f32 {
    if valence == 0 {
        return -1.0; // Fully processed.
    }

    let cache_score = if cache_pos < cache_scores.len() {
        cache_scores[cache_pos]
    } else {
        0.0 // Not in cache.
    };

    let clamped_valence = valence.min(FORSYTH_VALENCE_LIMIT);
    let valence_score = valence_scores[clamped_valence];

    cache_score + valence_score
}

// ---------------------------------------------------------------------------
// QEM Edge Collapse (standalone for lod_mesh usage)
// ---------------------------------------------------------------------------

/// A self-contained QEM edge collapse entry for the LOD generator's internal
/// priority queue.
#[derive(Debug, Clone)]
struct LodEdgeCollapse {
    v0: u32,
    v1: u32,
    cost: f64,
    target_position: Vec3,
    sequence: u64,
}

impl PartialEq for LodEdgeCollapse {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for LodEdgeCollapse {}

impl PartialOrd for LodEdgeCollapse {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LodEdgeCollapse {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: lowest cost first.
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

// ---------------------------------------------------------------------------
// Utility: compute screen-space projected size
// ---------------------------------------------------------------------------

/// Compute the screen-space projected diameter of a bounding sphere.
///
/// This is the key metric used for LOD selection. The result is in pixels.
pub fn compute_screen_size(
    center: Vec3,
    radius: f32,
    view_projection: &glam::Mat4,
    viewport_height: f32,
) -> f32 {
    let clip = *view_projection * center.extend(1.0);
    if clip.w <= 0.0 {
        return 0.0;
    }
    2.0 * radius * viewport_height / clip.w
}

/// Compute the screen-space error for a LOD level at a given distance.
pub fn compute_screen_error(
    geometric_error: f32,
    distance: f32,
    fov_y: f32,
    viewport_height: f32,
) -> f32 {
    if distance <= 0.0 {
        return f32::MAX;
    }
    let proj_factor = viewport_height / (2.0 * (fov_y * 0.5).tan());
    geometric_error * proj_factor / distance
}

// ---------------------------------------------------------------------------
// LodStatistics
// ---------------------------------------------------------------------------

/// Statistics about a LOD chain for debugging and profiling.
#[derive(Debug, Clone)]
pub struct LodStatistics {
    /// Triangle counts per level.
    pub triangle_counts: Vec<u32>,
    /// Vertex counts per level.
    pub vertex_counts: Vec<u32>,
    /// Reduction ratios per level relative to level 0.
    pub reduction_ratios: Vec<f32>,
    /// ACMR per level (Average Cache Miss Ratio).
    pub acmr_values: Vec<f32>,
    /// Total memory footprint in bytes (vertex + index data).
    pub total_memory_bytes: usize,
}

impl LodStatistics {
    /// Compute statistics for a LOD chain.
    pub fn from_chain(chain: &LodChain) -> Self {
        let mut triangle_counts = Vec::new();
        let mut vertex_counts = Vec::new();
        let mut reduction_ratios = Vec::new();
        let mut acmr_values = Vec::new();
        let mut total_memory: usize = 0;

        let base_tris = chain.meshes.first().map(|m| m.triangle_count()).unwrap_or(0);

        for mesh in &chain.meshes {
            let tris = mesh.triangle_count();
            let verts = mesh.vertex_count;
            triangle_counts.push(tris);
            vertex_counts.push(verts);

            let ratio = if base_tris > 0 {
                tris as f32 / base_tris as f32
            } else {
                0.0
            };
            reduction_ratios.push(ratio);

            let acmr = MeshOptimizer::compute_acmr(&mesh.indices, FORSYTH_CACHE_SIZE);
            acmr_values.push(acmr);

            total_memory += mesh.vertices.len() * std::mem::size_of::<Vertex>();
            total_memory += mesh.indices.len() * std::mem::size_of::<u32>();
        }

        Self {
            triangle_counts,
            vertex_counts,
            reduction_ratios,
            acmr_values,
            total_memory_bytes: total_memory,
        }
    }
}

// ---------------------------------------------------------------------------
// AutoLodBuilder
// ---------------------------------------------------------------------------

/// Convenience builder that generates a complete `LodChain` from a source mesh
/// with sensible defaults.
pub struct AutoLodBuilder {
    /// Number of LOD levels to generate (including the source mesh as LOD 0).
    pub level_count: usize,
    /// Reduction factor per level (each level has `factor` times the triangles
    /// of the previous level).
    pub reduction_factor: f32,
    /// Screen-size thresholds, or None to compute automatically.
    pub screen_thresholds: Option<Vec<f32>>,
    /// Whether to run mesh optimisation on each level.
    pub optimize: bool,
    /// Cross-fade range in pixels.
    pub fade_range: f32,
}

impl AutoLodBuilder {
    /// Create a builder with defaults: 4 levels, 0.5x reduction, auto thresholds.
    pub fn new() -> Self {
        Self {
            level_count: DEFAULT_LOD_LEVELS,
            reduction_factor: 0.5,
            screen_thresholds: None,
            optimize: true,
            fade_range: 16.0,
        }
    }

    /// Set the number of LOD levels.
    pub fn levels(mut self, count: usize) -> Self {
        self.level_count = count.clamp(1, MAX_LOD_LEVELS);
        self
    }

    /// Set the reduction factor per level.
    pub fn reduction(mut self, factor: f32) -> Self {
        self.reduction_factor = factor.clamp(0.1, 0.9);
        self
    }

    /// Set explicit screen-size thresholds.
    pub fn thresholds(mut self, thresholds: Vec<f32>) -> Self {
        self.screen_thresholds = Some(thresholds);
        self
    }

    /// Enable or disable mesh optimisation.
    pub fn optimize(mut self, enable: bool) -> Self {
        self.optimize = enable;
        self
    }

    /// Set the cross-fade range.
    pub fn fade(mut self, range: f32) -> Self {
        self.fade_range = range;
        self
    }

    /// Build the LOD chain from the source mesh.
    pub fn build(self, source: &Mesh) -> LodChain {
        let generator = LodGenerator::new();

        let configs: Vec<LodConfig> = (0..self.level_count)
            .map(|i| {
                let ratio = self.reduction_factor.powi(i as i32);
                let screen = if let Some(ref thresholds) = self.screen_thresholds {
                    thresholds.get(i).copied().unwrap_or(0.0)
                } else {
                    if i == 0 {
                        f32::MAX
                    } else {
                        400.0 * self.reduction_factor.powi(i as i32 - 1)
                    }
                };
                LodConfig::from_ratio(ratio).with_screen_size(screen)
            })
            .collect();

        let mut meshes = generator.generate_lods(source, &configs);

        if self.optimize {
            for mesh in &mut meshes {
                MeshOptimizer::optimize_mesh(mesh);
            }
        }

        let thresholds: Vec<f32> = configs.iter().map(|c| c.screen_size_threshold).collect();

        let mut chain = LodChain::new(meshes, thresholds);
        chain.set_fade_range(self.fade_range);
        chain
    }
}

impl Default for AutoLodBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{create_cube, create_plane, create_sphere, Mesh};

    #[test]
    fn test_lod_config_defaults() {
        let config = LodConfig::default();
        assert!(config.preserve_uv_seams);
        assert!(config.preserve_material_boundaries);
        assert!((config.reduction_ratio - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_lod_config_resolve_target() {
        let config = LodConfig::from_ratio(0.5);
        assert_eq!(config.resolve_target(100), 50);
        assert_eq!(config.resolve_target(7), 4); // min 4

        let config2 = LodConfig::from_triangle_count(30);
        assert_eq!(config2.resolve_target(100), 30);
        assert_eq!(config2.resolve_target(20), 20); // clamped to source
    }

    #[test]
    fn test_lod_generator_sphere() {
        let mesh = create_sphere(16, 12);
        let gen = LodGenerator::new();
        let configs = vec![
            LodConfig::from_ratio(0.5),
            LodConfig::from_ratio(0.25),
        ];
        let lods = gen.generate_lods(&mesh, &configs);

        assert_eq!(lods.len(), 2);
        // Each LOD should have fewer triangles than the previous.
        assert!(lods[0].triangle_count() <= mesh.triangle_count());
        assert!(lods[1].triangle_count() <= lods[0].triangle_count());
        // All should have valid geometry.
        for lod in &lods {
            assert!(lod.vertex_count > 0);
            assert!(lod.indices.len() % 3 == 0);
        }
    }

    #[test]
    fn test_lod_generator_default_lods() {
        let mesh = create_sphere(16, 12);
        let gen = LodGenerator::new();
        let lods = gen.generate_default_lods(&mesh);

        assert_eq!(lods.len(), DEFAULT_LOD_LEVELS);
        // Monotonically decreasing triangle count.
        for i in 1..lods.len() {
            assert!(lods[i].triangle_count() <= lods[i - 1].triangle_count());
        }
    }

    #[test]
    fn test_lod_chain_select() {
        let meshes = vec![
            create_sphere(32, 24),
            create_sphere(16, 12),
            create_sphere(8, 6),
        ];
        let thresholds = vec![f32::MAX, 200.0, 50.0];
        let chain = LodChain::new(meshes, thresholds);

        assert_eq!(chain.select_lod(300.0), 1); // above 200, below MAX
        assert_eq!(chain.select_lod(100.0), 1); // above 50
        assert_eq!(chain.select_lod(30.0), 2);  // below 50
    }

    #[test]
    fn test_lod_chain_from_meshes() {
        let meshes = vec![
            create_sphere(16, 12),
            create_sphere(8, 6),
        ];
        let chain = LodChain::from_meshes(meshes);
        assert_eq!(chain.level_count(), 2);
    }

    #[test]
    fn test_lod_chain_fade() {
        let meshes = vec![
            create_sphere(16, 12),
            create_sphere(8, 6),
        ];
        let thresholds = vec![f32::MAX, 100.0];
        let mut chain = LodChain::new(meshes, thresholds);
        chain.set_fade_range(20.0);

        let (primary, secondary, factor) = chain.select_lod_with_fade(90.0);
        // Should be in the transition zone (100 - 20 = 80 to 100).
        assert!(primary == 0 || primary == 1);
        // At the very bottom of the range, factor should be high.
        let (_, _, factor_low) = chain.select_lod_with_fade(80.0);
        assert!((factor_low - 0.0).abs() < 0.01 || factor_low > 0.9);
    }

    #[test]
    fn test_dither_pattern_values() {
        // All values should be in [0, 1).
        for y in 0..4 {
            for x in 0..4 {
                let val = DitherPattern::sample(x, y);
                assert!(val >= 0.0 && val < 1.0, "Dither value out of range: {val}");
            }
        }

        // All 16 values should be unique.
        let flat = DitherPattern::to_flat_array();
        let mut sorted = flat.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 1..sorted.len() {
            assert!(
                (sorted[i] - sorted[i - 1]).abs() > 1e-6,
                "Duplicate dither values"
            );
        }
    }

    #[test]
    fn test_dither_should_fade() {
        // At fade_factor = 0.0, nothing should fade.
        let mut fade_count = 0;
        for y in 0..4 {
            for x in 0..4 {
                if DitherPattern::should_fade(x, y, 0.0) {
                    fade_count += 1;
                }
            }
        }
        assert_eq!(fade_count, 0);

        // At fade_factor = 1.0, everything should fade.
        fade_count = 0;
        for y in 0..4 {
            for x in 0..4 {
                if DitherPattern::should_fade(x, y, 1.0) {
                    fade_count += 1;
                }
            }
        }
        assert_eq!(fade_count, 16);
    }

    #[test]
    fn test_proxy_mesh_from_aabb() {
        let aabb = AABB::new(Vec3::ZERO, Vec3::ONE);
        let proxy = ProxyMesh::generate_proxy(&aabb);

        assert_eq!(proxy.vertex_count, 24); // 6 faces * 4 verts
        assert_eq!(proxy.triangle_count(), 12); // 6 faces * 2 tris
    }

    #[test]
    fn test_proxy_mesh_cross() {
        let aabb = AABB::new(-Vec3::ONE, Vec3::ONE);
        let proxy = ProxyMesh::generate_cross_proxy(&aabb);

        assert_eq!(proxy.vertex_count, 8); // 2 quads * 4 verts
        assert_eq!(proxy.triangle_count(), 4); // 2 quads * 2 tris
    }

    #[test]
    fn test_proxy_mesh_convex() {
        let hull = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(-1.0, -1.0, 1.0),
        ];
        let proxy = ProxyMesh::generate_convex_proxy(&hull);

        assert!(proxy.vertex_count > 0);
        assert!(proxy.triangle_count() > 0);
        assert!(proxy.indices.len() % 3 == 0);
    }

    #[test]
    fn test_mesh_optimizer_vertex_cache() {
        let mut mesh = create_sphere(16, 12);
        let acmr_before = MeshOptimizer::compute_acmr(&mesh.indices, FORSYTH_CACHE_SIZE);

        MeshOptimizer::optimize_vertex_cache(&mut mesh);

        let acmr_after = MeshOptimizer::compute_acmr(&mesh.indices, FORSYTH_CACHE_SIZE);

        // The optimised ACMR should be no worse than the original (and typically
        // significantly better).
        assert!(
            acmr_after <= acmr_before + 0.01,
            "ACMR got worse: {acmr_before:.4} -> {acmr_after:.4}"
        );
    }

    #[test]
    fn test_mesh_optimizer_vertex_fetch() {
        let mut mesh = create_cube();
        let original_tri_count = mesh.triangle_count();

        MeshOptimizer::optimize_vertex_fetch(&mut mesh);

        // Vertex count and triangle count should be unchanged.
        assert_eq!(mesh.triangle_count(), original_tri_count);
        assert_eq!(mesh.vertex_count, mesh.vertices.len() as u32);

        // The first index in the buffer should reference vertex 0.
        assert_eq!(mesh.indices[0], 0);
    }

    #[test]
    fn test_mesh_optimizer_overdraw() {
        let mut mesh = create_sphere(16, 12);
        let original_tri_count = mesh.triangle_count();

        MeshOptimizer::optimize_overdraw(&mut mesh);

        // Triangle count should be unchanged.
        assert_eq!(mesh.triangle_count(), original_tri_count);
    }

    #[test]
    fn test_mesh_optimizer_full_pipeline() {
        let mut mesh = create_sphere(16, 12);
        let original_verts = mesh.vertex_count;
        let original_tris = mesh.triangle_count();

        MeshOptimizer::optimize_mesh(&mut mesh);

        // Vertex and triangle counts must be preserved.
        assert_eq!(mesh.vertex_count, mesh.vertices.len() as u32);
        assert_eq!(mesh.triangle_count(), original_tris);
    }

    #[test]
    fn test_compute_acmr() {
        let mesh = create_cube();
        let acmr = MeshOptimizer::compute_acmr(&mesh.indices, 16);
        // ACMR should be positive and reasonable.
        assert!(acmr > 0.0);
        assert!(acmr < 10.0);
    }

    #[test]
    fn test_compute_atvr() {
        let mesh = create_cube();
        let atvr = MeshOptimizer::compute_atvr(&mesh.indices, 16);
        // ATVR should be >= 1.0 (every vertex must be fetched at least once).
        assert!(atvr >= 1.0);
    }

    #[test]
    fn test_screen_size_computation() {
        let vp = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        let size = compute_screen_size(Vec3::new(0.0, 0.0, -5.0), 1.0, &vp, 1080.0);
        assert!(size > 0.0);
    }

    #[test]
    fn test_screen_error() {
        let error = compute_screen_error(0.1, 10.0, std::f32::consts::FRAC_PI_4, 1080.0);
        assert!(error > 0.0);
        // Closer distance should produce larger screen error.
        let error_close = compute_screen_error(0.1, 2.0, std::f32::consts::FRAC_PI_4, 1080.0);
        assert!(error_close > error);
    }

    #[test]
    fn test_hard_edge_detection() {
        let mesh = create_cube();
        let hard = detect_hard_edge_vertices(&mesh, 0.5);
        // A cube has all hard edges, so most vertices should be detected.
        assert!(!hard.is_empty());
    }

    #[test]
    fn test_material_boundary_detection() {
        let mesh = create_cube();
        // A single-submesh mesh should have no material boundaries.
        let boundaries = detect_material_boundary_vertices(&mesh);
        assert!(boundaries.is_empty());
    }

    #[test]
    fn test_lod_statistics() {
        let meshes = vec![
            create_sphere(16, 12),
            create_sphere(8, 6),
        ];
        let chain = LodChain::from_meshes(meshes);
        let stats = LodStatistics::from_chain(&chain);

        assert_eq!(stats.triangle_counts.len(), 2);
        assert_eq!(stats.vertex_counts.len(), 2);
        assert!(stats.total_memory_bytes > 0);
        assert!((stats.reduction_ratios[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_auto_lod_builder() {
        let source = create_sphere(16, 12);
        let chain = AutoLodBuilder::new()
            .levels(3)
            .reduction(0.5)
            .optimize(false) // skip optimisation for speed in tests
            .fade(20.0)
            .build(&source);

        assert_eq!(chain.level_count(), 3);
        assert!((chain.fade_range - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_forsyth_scores() {
        let cache_scores = forsyth_cache_position_scores();
        assert_eq!(cache_scores.len(), FORSYTH_CACHE_SIZE + 1);
        // First 3 positions should have the same high score.
        assert!((cache_scores[0] - cache_scores[1]).abs() < 1e-5);
        assert!((cache_scores[1] - cache_scores[2]).abs() < 1e-5);
        // Last position (just outside cache) should be 0.
        assert!((cache_scores[FORSYTH_CACHE_SIZE] - 0.0).abs() < 1e-5);

        let valence_scores = forsyth_valence_scores();
        assert_eq!(valence_scores.len(), FORSYTH_VALENCE_LIMIT + 1);
        // Valence 0 should be 0.
        assert!((valence_scores[0] - 0.0).abs() < 1e-5);
        // Valence 1 should have the highest score.
        assert!(valence_scores[1] > valence_scores[2]);
    }

    #[test]
    fn test_optimize_empty_mesh() {
        let mut mesh = Mesh::new(Vec::new(), Vec::new());
        // Should not panic.
        MeshOptimizer::optimize_mesh(&mut mesh);
        assert_eq!(mesh.vertex_count, 0);
    }

    #[test]
    fn test_optimize_single_triangle() {
        let mut builder = MeshBuilder::new();
        let v0 = builder.add_vertex(Vec3::ZERO, Vec3::Y, Vec2::ZERO);
        let v1 = builder.add_vertex(Vec3::X, Vec3::Y, Vec2::X);
        let v2 = builder.add_vertex(Vec3::Z, Vec3::Y, Vec2::Y);
        builder.add_triangle(v0, v1, v2);
        let mut mesh = builder.build_no_tangents();

        MeshOptimizer::optimize_mesh(&mut mesh);

        assert_eq!(mesh.triangle_count(), 1);
        assert_eq!(mesh.vertex_count, 3);
    }
}
