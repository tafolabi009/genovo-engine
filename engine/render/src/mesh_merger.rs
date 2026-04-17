// engine/render/src/mesh_merger.rs
//
// Static mesh merging for the Genovo engine.
//
// Combines multiple static meshes into single draw calls to reduce
// per-object overhead:
//
// - **Mesh combining** -- Merges vertex and index data from multiple static
//   meshes into a single vertex/index buffer.
// - **Per-material sub-meshes** -- Groups merged geometry by material,
//   producing one draw call per material.
// - **Bounding volume update** -- Automatically computes merged bounding
//   boxes and spheres for frustum culling.
// - **Automatic merging** -- Identifies candidates for merging based on
//   spatial proximity, material compatibility, and static flag.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EPSILON: f32 = 1e-6;

/// Maximum vertices in a single merged mesh (65535 for u16 index).
const DEFAULT_MAX_VERTICES: u32 = 65535;

/// Maximum indices in a single merged mesh.
const DEFAULT_MAX_INDICES: u32 = 1_000_000;

/// Default spatial cell size for grouping nearby meshes.
const DEFAULT_CELL_SIZE: f32 = 32.0;

// ---------------------------------------------------------------------------
// Vertex data
// ---------------------------------------------------------------------------

/// A vertex with position, normal, UV, and tangent.
#[derive(Debug, Clone, Copy)]
pub struct MergedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub tangent: [f32; 4],
    pub color: [f32; 4],
}

impl MergedVertex {
    pub fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> Self {
        Self {
            position,
            normal,
            uv,
            tangent: [1.0, 0.0, 0.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }

    /// Transform this vertex by a 4x4 matrix (column-major).
    pub fn transform(&self, matrix: &[f32; 16]) -> Self {
        let p = self.position;
        let n = self.normal;

        // Transform position.
        let px = matrix[0] * p[0] + matrix[4] * p[1] + matrix[8] * p[2] + matrix[12];
        let py = matrix[1] * p[0] + matrix[5] * p[1] + matrix[9] * p[2] + matrix[13];
        let pz = matrix[2] * p[0] + matrix[6] * p[1] + matrix[10] * p[2] + matrix[14];

        // Transform normal (using upper 3x3 of the inverse transpose).
        // For uniform scale, this is the same as the upper 3x3.
        let nx = matrix[0] * n[0] + matrix[4] * n[1] + matrix[8] * n[2];
        let ny = matrix[1] * n[0] + matrix[5] * n[1] + matrix[9] * n[2];
        let nz = matrix[2] * n[0] + matrix[6] * n[1] + matrix[10] * n[2];
        let nlen = (nx * nx + ny * ny + nz * nz).sqrt().max(EPSILON);

        Self {
            position: [px, py, pz],
            normal: [nx / nlen, ny / nlen, nz / nlen],
            uv: self.uv,
            tangent: self.tangent,
            color: self.color,
        }
    }
}

// ---------------------------------------------------------------------------
// Bounding volume
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct MergedAABB {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl MergedAABB {
    /// Create an empty (inverted) AABB.
    pub fn empty() -> Self {
        Self {
            min: [f32::MAX; 3],
            max: [f32::MIN; 3],
        }
    }

    /// Expand to include a point.
    pub fn expand_point(&mut self, p: [f32; 3]) {
        for i in 0..3 {
            self.min[i] = self.min[i].min(p[i]);
            self.max[i] = self.max[i].max(p[i]);
        }
    }

    /// Expand to include another AABB.
    pub fn expand_aabb(&mut self, other: &MergedAABB) {
        for i in 0..3 {
            self.min[i] = self.min[i].min(other.min[i]);
            self.max[i] = self.max[i].max(other.max[i]);
        }
    }

    /// Center of the AABB.
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Half-extents.
    pub fn half_extents(&self) -> [f32; 3] {
        [
            (self.max[0] - self.min[0]) * 0.5,
            (self.max[1] - self.min[1]) * 0.5,
            (self.max[2] - self.min[2]) * 0.5,
        ]
    }

    /// Bounding sphere radius.
    pub fn bounding_radius(&self) -> f32 {
        let he = self.half_extents();
        (he[0] * he[0] + he[1] * he[1] + he[2] * he[2]).sqrt()
    }

    /// Volume.
    pub fn volume(&self) -> f32 {
        let e = self.half_extents();
        8.0 * e[0] * e[1] * e[2]
    }

    /// Check if this AABB is valid (non-empty).
    pub fn is_valid(&self) -> bool {
        self.min[0] <= self.max[0] && self.min[1] <= self.max[1] && self.min[2] <= self.max[2]
    }
}

// ---------------------------------------------------------------------------
// Material ID
// ---------------------------------------------------------------------------

/// Material identifier for grouping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MergeMaterialId(pub u32);

// ---------------------------------------------------------------------------
// Sub-mesh
// ---------------------------------------------------------------------------

/// A sub-mesh within a merged mesh, representing one material group.
#[derive(Debug, Clone)]
pub struct SubMesh {
    /// Material ID for this sub-mesh.
    pub material: MergeMaterialId,
    /// Starting index in the merged index buffer.
    pub index_offset: u32,
    /// Number of indices.
    pub index_count: u32,
    /// Starting vertex in the merged vertex buffer.
    pub vertex_offset: u32,
    /// Number of vertices.
    pub vertex_count: u32,
    /// Bounding box of this sub-mesh.
    pub bounds: MergedAABB,
}

// ---------------------------------------------------------------------------
// Mesh source (input to merging)
// ---------------------------------------------------------------------------

/// A source mesh to be merged.
#[derive(Debug, Clone)]
pub struct MeshSource {
    /// Unique identifier for the source mesh.
    pub id: u32,
    /// Vertices.
    pub vertices: Vec<MergedVertex>,
    /// Indices (triangles).
    pub indices: Vec<u32>,
    /// Material ID.
    pub material: MergeMaterialId,
    /// World transform (column-major 4x4).
    pub transform: [f32; 16],
    /// Bounding box in local space.
    pub local_bounds: MergedAABB,
    /// Whether this mesh is static (eligible for merging).
    pub is_static: bool,
}

impl MeshSource {
    /// Compute the bounding box in world space.
    pub fn world_bounds(&self) -> MergedAABB {
        let mut aabb = MergedAABB::empty();
        for v in &self.vertices {
            let tv = v.transform(&self.transform);
            aabb.expand_point(tv.position);
        }
        aabb
    }
}

// ---------------------------------------------------------------------------
// Merged mesh (output)
// ---------------------------------------------------------------------------

/// A merged mesh containing combined geometry from multiple sources.
#[derive(Debug, Clone)]
pub struct MergedMesh {
    /// Unique identifier.
    pub id: u32,
    /// Combined vertices (in world space).
    pub vertices: Vec<MergedVertex>,
    /// Combined indices.
    pub indices: Vec<u32>,
    /// Sub-meshes (one per material).
    pub sub_meshes: Vec<SubMesh>,
    /// Overall bounding box.
    pub bounds: MergedAABB,
    /// IDs of source meshes that were merged.
    pub source_ids: Vec<u32>,
    /// Total triangle count.
    pub triangle_count: u32,
}

// ---------------------------------------------------------------------------
// Merge statistics
// ---------------------------------------------------------------------------

/// Statistics from a merge operation.
#[derive(Debug, Clone, Default)]
pub struct MergeStats {
    /// Number of input meshes.
    pub input_meshes: u32,
    /// Number of output merged meshes.
    pub output_meshes: u32,
    /// Number of draw calls saved.
    pub draw_calls_saved: u32,
    /// Total input vertices.
    pub input_vertices: u64,
    /// Total output vertices.
    pub output_vertices: u64,
    /// Total input indices.
    pub input_indices: u64,
    /// Total output indices.
    pub output_indices: u64,
    /// Merge time in microseconds.
    pub merge_time_us: u64,
    /// Number of merge groups (by material).
    pub material_groups: u32,
    /// Average meshes per merged mesh.
    pub avg_meshes_per_group: f32,
}

impl fmt::Display for MergeStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MeshMerge: {} -> {} meshes ({} draw calls saved), {:.1} avg/group",
            self.input_meshes,
            self.output_meshes,
            self.draw_calls_saved,
            self.avg_meshes_per_group,
        )
    }
}

// ---------------------------------------------------------------------------
// Merge configuration
// ---------------------------------------------------------------------------

/// Configuration for the mesh merger.
#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// Maximum vertices in a single merged mesh.
    pub max_vertices: u32,
    /// Maximum indices in a single merged mesh.
    pub max_indices: u32,
    /// Spatial cell size for grouping nearby meshes (world units).
    pub cell_size: f32,
    /// Only merge meshes flagged as static.
    pub static_only: bool,
    /// Only merge meshes with the same material.
    pub same_material_only: bool,
    /// Minimum number of meshes to trigger merging.
    pub min_meshes_to_merge: u32,
    /// Maximum world-space distance between merged mesh centers.
    pub max_merge_distance: f32,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            max_vertices: DEFAULT_MAX_VERTICES,
            max_indices: DEFAULT_MAX_INDICES,
            cell_size: DEFAULT_CELL_SIZE,
            static_only: true,
            same_material_only: true,
            min_meshes_to_merge: 2,
            max_merge_distance: 100.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh merger
// ---------------------------------------------------------------------------

/// Merges static meshes to reduce draw calls.
pub struct MeshMerger {
    /// Configuration.
    config: MergeConfig,
    /// Next merged mesh ID.
    next_id: u32,
    /// Statistics from the last merge operation.
    stats: MergeStats,
}

impl MeshMerger {
    /// Create a new mesh merger.
    pub fn new(config: MergeConfig) -> Self {
        Self {
            config,
            next_id: 0,
            stats: MergeStats::default(),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &MergeConfig {
        &self.config
    }

    /// Set the configuration.
    pub fn set_config(&mut self, config: MergeConfig) {
        self.config = config;
    }

    /// Get the statistics from the last merge.
    pub fn stats(&self) -> &MergeStats {
        &self.stats
    }

    /// Merge a set of source meshes into merged meshes.
    pub fn merge(&mut self, sources: &[MeshSource]) -> Vec<MergedMesh> {
        let start = std::time::Instant::now();

        self.stats = MergeStats::default();
        self.stats.input_meshes = sources.len() as u32;

        // Filter eligible meshes.
        let eligible: Vec<&MeshSource> = sources
            .iter()
            .filter(|s| !self.config.static_only || s.is_static)
            .collect();

        // Count input geometry.
        self.stats.input_vertices = eligible.iter().map(|s| s.vertices.len() as u64).sum();
        self.stats.input_indices = eligible.iter().map(|s| s.indices.len() as u64).sum();

        // Group by material.
        let mut material_groups: HashMap<MergeMaterialId, Vec<&MeshSource>> = HashMap::new();
        for source in &eligible {
            material_groups
                .entry(source.material)
                .or_default()
                .push(source);
        }
        self.stats.material_groups = material_groups.len() as u32;

        let mut merged_meshes = Vec::new();

        for (material, group) in &material_groups {
            if group.len() < self.config.min_meshes_to_merge as usize {
                // Not enough meshes; create individual merged meshes.
                for source in group {
                    merged_meshes.push(self.create_single_merged(source, *material));
                }
                continue;
            }

            // Further group by spatial proximity.
            let spatial_groups = self.spatial_group(group);

            for spatial_group in spatial_groups {
                // Split into batches that fit within vertex/index limits.
                let batches = self.split_into_batches(&spatial_group);

                for batch in batches {
                    if batch.len() < self.config.min_meshes_to_merge as usize {
                        for source in &batch {
                            merged_meshes.push(self.create_single_merged(source, *material));
                        }
                    } else {
                        merged_meshes.push(self.merge_batch(&batch, *material));
                    }
                }
            }
        }

        // Compute statistics.
        self.stats.output_meshes = merged_meshes.len() as u32;
        self.stats.draw_calls_saved =
            self.stats.input_meshes.saturating_sub(self.stats.output_meshes);
        self.stats.output_vertices = merged_meshes.iter().map(|m| m.vertices.len() as u64).sum();
        self.stats.output_indices = merged_meshes.iter().map(|m| m.indices.len() as u64).sum();
        self.stats.avg_meshes_per_group = if !merged_meshes.is_empty() {
            self.stats.input_meshes as f32 / merged_meshes.len() as f32
        } else {
            0.0
        };
        self.stats.merge_time_us = start.elapsed().as_micros() as u64;

        merged_meshes
    }

    /// Create a merged mesh from a single source.
    fn create_single_merged(&mut self, source: &MeshSource, material: MergeMaterialId) -> MergedMesh {
        let id = self.next_id;
        self.next_id += 1;

        let vertices: Vec<MergedVertex> = source
            .vertices
            .iter()
            .map(|v| v.transform(&source.transform))
            .collect();

        let mut bounds = MergedAABB::empty();
        for v in &vertices {
            bounds.expand_point(v.position);
        }

        MergedMesh {
            id,
            vertices: vertices.clone(),
            indices: source.indices.clone(),
            sub_meshes: vec![SubMesh {
                material,
                index_offset: 0,
                index_count: source.indices.len() as u32,
                vertex_offset: 0,
                vertex_count: vertices.len() as u32,
                bounds,
            }],
            bounds,
            source_ids: vec![source.id],
            triangle_count: source.indices.len() as u32 / 3,
        }
    }

    /// Merge a batch of sources into one merged mesh.
    fn merge_batch(&mut self, sources: &[&MeshSource], material: MergeMaterialId) -> MergedMesh {
        let id = self.next_id;
        self.next_id += 1;

        let total_verts: usize = sources.iter().map(|s| s.vertices.len()).sum();
        let total_indices: usize = sources.iter().map(|s| s.indices.len()).sum();

        let mut vertices = Vec::with_capacity(total_verts);
        let mut indices = Vec::with_capacity(total_indices);
        let mut source_ids = Vec::with_capacity(sources.len());
        let mut bounds = MergedAABB::empty();

        let mut vertex_offset: u32 = 0;

        for source in sources {
            // Transform vertices to world space.
            for v in &source.vertices {
                let tv = v.transform(&source.transform);
                bounds.expand_point(tv.position);
                vertices.push(tv);
            }

            // Offset indices.
            for &idx in &source.indices {
                indices.push(idx + vertex_offset);
            }

            vertex_offset += source.vertices.len() as u32;
            source_ids.push(source.id);
        }

        MergedMesh {
            id,
            vertices,
            indices: indices.clone(),
            sub_meshes: vec![SubMesh {
                material,
                index_offset: 0,
                index_count: indices.len() as u32,
                vertex_offset: 0,
                vertex_count: total_verts as u32,
                bounds,
            }],
            bounds,
            source_ids,
            triangle_count: indices.len() as u32 / 3,
        }
    }

    /// Group meshes by spatial proximity using a grid.
    fn spatial_group<'a>(&self, meshes: &[&'a MeshSource]) -> Vec<Vec<&'a MeshSource>> {
        let cell = self.config.cell_size;
        if cell <= EPSILON {
            return vec![meshes.to_vec()];
        }

        let mut grid: HashMap<(i32, i32, i32), Vec<&'a MeshSource>> = HashMap::new();

        for &mesh in meshes {
            let wb = mesh.world_bounds();
            let center = wb.center();
            let cx = (center[0] / cell).floor() as i32;
            let cy = (center[1] / cell).floor() as i32;
            let cz = (center[2] / cell).floor() as i32;
            grid.entry((cx, cy, cz)).or_default().push(mesh);
        }

        grid.into_values().collect()
    }

    /// Split a group into batches that fit within vertex/index limits.
    fn split_into_batches<'a>(&self, meshes: &[&'a MeshSource]) -> Vec<Vec<&'a MeshSource>> {
        let max_v = self.config.max_vertices as usize;
        let max_i = self.config.max_indices as usize;

        let mut batches: Vec<Vec<&'a MeshSource>> = Vec::new();
        let mut current_batch: Vec<&'a MeshSource> = Vec::new();
        let mut current_verts: usize = 0;
        let mut current_indices: usize = 0;

        for &mesh in meshes {
            let v = mesh.vertices.len();
            let i = mesh.indices.len();

            if current_verts + v > max_v || current_indices + i > max_i {
                if !current_batch.is_empty() {
                    batches.push(std::mem::take(&mut current_batch));
                    current_verts = 0;
                    current_indices = 0;
                }
            }

            current_batch.push(mesh);
            current_verts += v;
            current_indices += i;
        }

        if !current_batch.is_empty() {
            batches.push(current_batch);
        }

        batches
    }
}

impl Default for MeshMerger {
    fn default() -> Self {
        Self::new(MergeConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    fn make_quad(id: u32, material: u32) -> MeshSource {
        MeshSource {
            id,
            vertices: vec![
                MergedVertex::new([-1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0]),
                MergedVertex::new([1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0]),
                MergedVertex::new([1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0]),
                MergedVertex::new([-1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0]),
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
            material: MergeMaterialId(material),
            transform: identity(),
            local_bounds: MergedAABB {
                min: [-1.0, 0.0, -1.0],
                max: [1.0, 0.0, 1.0],
            },
            is_static: true,
        }
    }

    #[test]
    fn test_merge_two_quads() {
        let mut merger = MeshMerger::default();
        let sources = vec![make_quad(0, 0), make_quad(1, 0)];
        let merged = merger.merge(&sources);

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].vertices.len(), 8);
        assert_eq!(merged[0].indices.len(), 12);
        assert_eq!(merged[0].source_ids.len(), 2);
    }

    #[test]
    fn test_different_materials_not_merged() {
        let mut merger = MeshMerger::default();
        let sources = vec![make_quad(0, 0), make_quad(1, 1)];
        let merged = merger.merge(&sources);

        // Each material gets its own merged mesh.
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_non_static_excluded() {
        let mut merger = MeshMerger::default();
        let mut dynamic = make_quad(1, 0);
        dynamic.is_static = false;
        let sources = vec![make_quad(0, 0), dynamic];
        let merged = merger.merge(&sources);

        // Only one static mesh, below min_meshes_to_merge.
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn test_aabb() {
        let mut aabb = MergedAABB::empty();
        aabb.expand_point([1.0, 2.0, 3.0]);
        aabb.expand_point([-1.0, -2.0, -3.0]);
        assert!(aabb.is_valid());
        let c = aabb.center();
        assert!((c[0]).abs() < EPSILON);
    }

    #[test]
    fn test_stats() {
        let mut merger = MeshMerger::default();
        let sources = vec![make_quad(0, 0), make_quad(1, 0), make_quad(2, 0)];
        merger.merge(&sources);
        let stats = merger.stats();
        assert_eq!(stats.input_meshes, 3);
        assert_eq!(stats.output_meshes, 1);
        assert_eq!(stats.draw_calls_saved, 2);
    }
}
