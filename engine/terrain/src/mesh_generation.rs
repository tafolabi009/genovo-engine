//! Terrain mesh generation from heightmaps.
//!
//! Converts [`Heightmap`](crate::Heightmap) data into renderable vertex and
//! index buffers. Supports chunk-based subdivision, skirt generation for
//! hiding LOD seams, and tangent-space computation for normal mapping.

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::heightmap::Heightmap;
use crate::{TerrainError, TerrainResult};

// ---------------------------------------------------------------------------
// Vertex format
// ---------------------------------------------------------------------------

/// A single terrain vertex with all attributes needed for rendering.
///
/// Layout matches a tightly packed GPU vertex buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct TerrainVertex {
    /// World-space position (x, y, z).
    pub position: [f32; 3],
    /// Surface normal (x, y, z).
    pub normal: [f32; 3],
    /// Texture coordinates (u, v).
    pub uv: [f32; 2],
    /// Tangent vector with bitangent sign in w (x, y, z, sign).
    pub tangent: [f32; 4],
}

impl TerrainVertex {
    /// Creates a new terrain vertex.
    #[inline]
    pub fn new(position: Vec3, normal: Vec3, uv: Vec2, tangent: Vec4) -> Self {
        Self {
            position: position.into(),
            normal: normal.into(),
            uv: uv.into(),
            tangent: tangent.into(),
        }
    }

    /// Returns the position as a `Vec3`.
    #[inline]
    pub fn pos(&self) -> Vec3 {
        Vec3::from(self.position)
    }

    /// Returns the normal as a `Vec3`.
    #[inline]
    pub fn norm(&self) -> Vec3 {
        Vec3::from(self.normal)
    }
}

// ---------------------------------------------------------------------------
// MeshData
// ---------------------------------------------------------------------------

/// Generated mesh data ready for upload to the GPU.
#[derive(Debug, Clone)]
pub struct MeshData {
    /// Vertex buffer data.
    pub vertices: Vec<TerrainVertex>,
    /// Index buffer data (triangle list, CCW winding).
    pub indices: Vec<u32>,
    /// Axis-aligned bounding box minimum corner.
    pub aabb_min: Vec3,
    /// Axis-aligned bounding box maximum corner.
    pub aabb_max: Vec3,
}

impl MeshData {
    /// Returns the number of triangles.
    #[inline]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Returns the vertex count.
    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Returns the raw vertex bytes for GPU upload.
    pub fn vertex_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.vertices)
    }

    /// Returns the raw index bytes for GPU upload.
    pub fn index_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.indices)
    }
}

// ---------------------------------------------------------------------------
// TerrainMesh — generation settings
// ---------------------------------------------------------------------------

/// Configuration for terrain mesh generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainMeshSettings {
    /// Horizontal cell size in world units.
    pub cell_size: f32,
    /// Vertical scale applied to heightmap values.
    pub height_scale: f32,
    /// UV tiling factor (how many times the texture repeats).
    pub uv_scale: f32,
    /// Whether to generate skirt vertices to hide LOD seams.
    pub generate_skirts: bool,
    /// Depth of the skirt below the edge vertices (world units).
    pub skirt_depth: f32,
}

impl Default for TerrainMeshSettings {
    fn default() -> Self {
        Self {
            cell_size: 1.0,
            height_scale: 100.0,
            uv_scale: 1.0,
            generate_skirts: true,
            skirt_depth: 10.0,
        }
    }
}

// ---------------------------------------------------------------------------
// TerrainMesh
// ---------------------------------------------------------------------------

/// Generates renderable mesh data from heightmaps.
pub struct TerrainMesh;

impl TerrainMesh {
    /// Generates a complete mesh from the entire heightmap.
    ///
    /// `resolution` is the step size between sampled height values (1 = every
    /// cell, 2 = every other cell, etc.).
    #[profiling::function]
    pub fn generate_mesh(
        heightmap: &Heightmap,
        resolution: u32,
        settings: &TerrainMeshSettings,
    ) -> TerrainResult<MeshData> {
        let resolution = resolution.max(1);
        let hm_w = heightmap.width();
        let hm_h = heightmap.height();

        if hm_w < 2 || hm_h < 2 {
            return Err(TerrainError::MeshGeneration(
                "Heightmap must be at least 2x2".into(),
            ));
        }

        // Number of vertices along each axis at the chosen resolution
        let verts_x = ((hm_w - 1) / resolution + 1) as usize;
        let verts_z = ((hm_h - 1) / resolution + 1) as usize;
        let num_verts = verts_x * verts_z;
        let num_quads = (verts_x - 1) * (verts_z - 1);

        let mut vertices = Vec::with_capacity(num_verts);
        let mut indices = Vec::with_capacity(num_quads * 6);

        let mut aabb_min = Vec3::splat(f32::INFINITY);
        let mut aabb_max = Vec3::splat(f32::NEG_INFINITY);

        // Generate vertices
        for vz in 0..verts_z {
            for vx in 0..verts_x {
                let hx = (vx as u32 * resolution).min(hm_w - 1);
                let hz = (vz as u32 * resolution).min(hm_h - 1);

                let height = heightmap.get(hx, hz) * settings.height_scale;
                let pos = Vec3::new(
                    hx as f32 * settings.cell_size,
                    height,
                    hz as f32 * settings.cell_size,
                );

                let normal = heightmap.normal_at_scaled(
                    hx as f32,
                    hz as f32,
                    settings.cell_size,
                    settings.height_scale,
                );

                let uv = Vec2::new(
                    hx as f32 / (hm_w - 1) as f32 * settings.uv_scale,
                    hz as f32 / (hm_h - 1) as f32 * settings.uv_scale,
                );

                aabb_min = aabb_min.min(pos);
                aabb_max = aabb_max.max(pos);

                vertices.push(TerrainVertex::new(
                    pos,
                    normal,
                    uv,
                    Vec4::new(1.0, 0.0, 0.0, 1.0), // placeholder tangent
                ));
            }
        }

        // Generate indices (two triangles per quad, CCW winding)
        for vz in 0..(verts_z - 1) {
            for vx in 0..(verts_x - 1) {
                let tl = (vz * verts_x + vx) as u32;
                let tr = tl + 1;
                let bl = ((vz + 1) * verts_x + vx) as u32;
                let br = bl + 1;

                // Triangle 1: tl -> bl -> tr
                indices.push(tl);
                indices.push(bl);
                indices.push(tr);

                // Triangle 2: tr -> bl -> br
                indices.push(tr);
                indices.push(bl);
                indices.push(br);
            }
        }

        // Compute proper normals from triangle cross-products
        Self::compute_normals_from_triangles(&mut vertices, &indices);

        // Compute tangents for normal mapping
        Self::compute_tangents(&mut vertices, &indices);

        // Generate skirts if requested
        if settings.generate_skirts {
            Self::generate_skirts(
                &mut vertices,
                &mut indices,
                verts_x,
                verts_z,
                settings.skirt_depth,
                &mut aabb_min,
                &mut aabb_max,
            );
        }

        Ok(MeshData {
            vertices,
            indices,
            aabb_min,
            aabb_max,
        })
    }

    /// Generates chunk-based meshes by splitting the terrain into an NxN grid
    /// of chunks.
    ///
    /// Returns a 2-D array (row-major) of mesh data, one per chunk.
    #[profiling::function]
    pub fn generate_chunked(
        heightmap: &Heightmap,
        chunks_x: u32,
        chunks_z: u32,
        resolution: u32,
        settings: &TerrainMeshSettings,
    ) -> TerrainResult<Vec<ChunkMeshData>> {
        let resolution = resolution.max(1);
        let hm_w = heightmap.width();
        let hm_h = heightmap.height();

        if hm_w < 2 || hm_h < 2 {
            return Err(TerrainError::MeshGeneration(
                "Heightmap must be at least 2x2".into(),
            ));
        }

        if chunks_x == 0 || chunks_z == 0 {
            return Err(TerrainError::MeshGeneration(
                "Chunk count must be > 0".into(),
            ));
        }

        let chunk_cells_x = (hm_w - 1) / chunks_x;
        let chunk_cells_z = (hm_h - 1) / chunks_z;

        let mut chunks = Vec::with_capacity((chunks_x * chunks_z) as usize);

        for cz in 0..chunks_z {
            for cx in 0..chunks_x {
                let start_x = cx * chunk_cells_x;
                let start_z = cz * chunk_cells_z;
                let end_x = if cx == chunks_x - 1 {
                    hm_w - 1
                } else {
                    start_x + chunk_cells_x
                };
                let end_z = if cz == chunks_z - 1 {
                    hm_h - 1
                } else {
                    start_z + chunk_cells_z
                };

                let mesh = Self::generate_chunk_mesh(
                    heightmap,
                    start_x,
                    start_z,
                    end_x,
                    end_z,
                    resolution,
                    settings,
                )?;

                chunks.push(ChunkMeshData {
                    chunk_x: cx,
                    chunk_z: cz,
                    mesh,
                });
            }
        }

        Ok(chunks)
    }

    /// Generates mesh data for a single chunk region of the heightmap.
    pub fn generate_chunk_mesh(
        heightmap: &Heightmap,
        start_x: u32,
        start_z: u32,
        end_x: u32,
        end_z: u32,
        resolution: u32,
        settings: &TerrainMeshSettings,
    ) -> TerrainResult<MeshData> {
        let resolution = resolution.max(1);
        let range_x = end_x - start_x;
        let range_z = end_z - start_z;

        let verts_x = (range_x / resolution + 1) as usize;
        let verts_z = (range_z / resolution + 1) as usize;

        if verts_x < 2 || verts_z < 2 {
            return Err(TerrainError::MeshGeneration(
                "Chunk too small for the given resolution".into(),
            ));
        }

        let num_verts = verts_x * verts_z;
        let num_quads = (verts_x - 1) * (verts_z - 1);

        let mut vertices = Vec::with_capacity(num_verts);
        let mut indices = Vec::with_capacity(num_quads * 6);

        let mut aabb_min = Vec3::splat(f32::INFINITY);
        let mut aabb_max = Vec3::splat(f32::NEG_INFINITY);

        let hm_w = heightmap.width();
        let hm_h = heightmap.height();

        // Generate vertices
        for vz in 0..verts_z {
            for vx in 0..verts_x {
                let hx = (start_x + vx as u32 * resolution).min(hm_w - 1);
                let hz = (start_z + vz as u32 * resolution).min(hm_h - 1);

                let height = heightmap.get(hx, hz) * settings.height_scale;
                let pos = Vec3::new(
                    hx as f32 * settings.cell_size,
                    height,
                    hz as f32 * settings.cell_size,
                );

                let normal = heightmap.normal_at_scaled(
                    hx as f32,
                    hz as f32,
                    settings.cell_size,
                    settings.height_scale,
                );

                let uv = Vec2::new(
                    hx as f32 / (hm_w - 1) as f32 * settings.uv_scale,
                    hz as f32 / (hm_h - 1) as f32 * settings.uv_scale,
                );

                aabb_min = aabb_min.min(pos);
                aabb_max = aabb_max.max(pos);

                vertices.push(TerrainVertex::new(
                    pos,
                    normal,
                    uv,
                    Vec4::new(1.0, 0.0, 0.0, 1.0),
                ));
            }
        }

        // Generate indices
        for vz in 0..(verts_z - 1) {
            for vx in 0..(verts_x - 1) {
                let tl = (vz * verts_x + vx) as u32;
                let tr = tl + 1;
                let bl = ((vz + 1) * verts_x + vx) as u32;
                let br = bl + 1;

                indices.push(tl);
                indices.push(bl);
                indices.push(tr);

                indices.push(tr);
                indices.push(bl);
                indices.push(br);
            }
        }

        Self::compute_normals_from_triangles(&mut vertices, &indices);
        Self::compute_tangents(&mut vertices, &indices);

        if settings.generate_skirts {
            Self::generate_skirts(
                &mut vertices,
                &mut indices,
                verts_x,
                verts_z,
                settings.skirt_depth,
                &mut aabb_min,
                &mut aabb_max,
            );
        }

        Ok(MeshData {
            vertices,
            indices,
            aabb_min,
            aabb_max,
        })
    }

    // -- Normal computation -------------------------------------------------

    /// Computes smooth normals by accumulating face normals at each vertex.
    fn compute_normals_from_triangles(vertices: &mut [TerrainVertex], indices: &[u32]) {
        // Zero out normals
        for v in vertices.iter_mut() {
            v.normal = [0.0, 0.0, 0.0];
        }

        // Accumulate face normals
        for tri in indices.chunks_exact(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;

            if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
                continue;
            }

            let p0 = vertices[i0].pos();
            let p1 = vertices[i1].pos();
            let p2 = vertices[i2].pos();

            let e1 = p1 - p0;
            let e2 = p2 - p0;
            let face_normal = e1.cross(e2);

            // Weight by face area (proportional to cross product magnitude)
            for &idx in &[i0, i1, i2] {
                let n = Vec3::from(vertices[idx].normal);
                vertices[idx].normal = (n + face_normal).into();
            }
        }

        // Normalize
        for v in vertices.iter_mut() {
            let n = Vec3::from(v.normal);
            let len = n.length();
            if len > 1e-8 {
                v.normal = (n / len).into();
            } else {
                v.normal = [0.0, 1.0, 0.0];
            }
        }
    }

    // -- Tangent computation ------------------------------------------------

    /// Computes tangent vectors using the MikkTSpace-compatible algorithm.
    ///
    /// For each triangle, the tangent is derived from the UV gradient in
    /// the triangle plane. Tangents are accumulated per-vertex and then
    /// orthogonalized against the vertex normal.
    fn compute_tangents(vertices: &mut [TerrainVertex], indices: &[u32]) {
        let num_verts = vertices.len();
        let mut tan1 = vec![Vec3::ZERO; num_verts];
        let mut tan2 = vec![Vec3::ZERO; num_verts];

        for tri in indices.chunks_exact(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;

            if i0 >= num_verts || i1 >= num_verts || i2 >= num_verts {
                continue;
            }

            let p0 = vertices[i0].pos();
            let p1 = vertices[i1].pos();
            let p2 = vertices[i2].pos();

            let uv0 = Vec2::from(vertices[i0].uv);
            let uv1 = Vec2::from(vertices[i1].uv);
            let uv2 = Vec2::from(vertices[i2].uv);

            let dp1 = p1 - p0;
            let dp2 = p2 - p0;
            let duv1 = uv1 - uv0;
            let duv2 = uv2 - uv0;

            let denom = duv1.x * duv2.y - duv1.y * duv2.x;
            if denom.abs() < 1e-8 {
                continue;
            }
            let r = 1.0 / denom;

            let sdir = Vec3::new(
                (duv2.y * dp1.x - duv1.y * dp2.x) * r,
                (duv2.y * dp1.y - duv1.y * dp2.y) * r,
                (duv2.y * dp1.z - duv1.y * dp2.z) * r,
            );
            let tdir = Vec3::new(
                (duv1.x * dp2.x - duv2.x * dp1.x) * r,
                (duv1.x * dp2.y - duv2.x * dp1.y) * r,
                (duv1.x * dp2.z - duv2.x * dp1.z) * r,
            );

            tan1[i0] += sdir;
            tan1[i1] += sdir;
            tan1[i2] += sdir;
            tan2[i0] += tdir;
            tan2[i1] += tdir;
            tan2[i2] += tdir;
        }

        // Orthogonalize and compute handedness
        for i in 0..num_verts {
            let n = vertices[i].norm();
            let t = tan1[i];

            // Gram-Schmidt orthogonalize
            let tangent = (t - n * n.dot(t)).normalize_or_zero();
            // Handedness
            let w = if n.cross(t).dot(tan2[i]) < 0.0 {
                -1.0
            } else {
                1.0
            };

            vertices[i].tangent = Vec4::new(tangent.x, tangent.y, tangent.z, w).into();
        }
    }

    // -- Skirt generation ---------------------------------------------------

    /// Generates skirt vertices and indices around the mesh edges.
    ///
    /// Skirts are extra geometry that extends downward from edge vertices,
    /// preventing cracks between chunks at different LOD levels.
    fn generate_skirts(
        vertices: &mut Vec<TerrainVertex>,
        indices: &mut Vec<u32>,
        verts_x: usize,
        verts_z: usize,
        skirt_depth: f32,
        aabb_min: &mut Vec3,
        aabb_max: &mut Vec3,
    ) {
        let base_count = vertices.len() as u32;

        // For each edge vertex, create a copy displaced downward
        let add_skirt_vertex = |verts: &mut Vec<TerrainVertex>,
                                 src_idx: usize,
                                 depth: f32,
                                 aabb_min: &mut Vec3,
                                 aabb_max: &mut Vec3| {
            let mut v = verts[src_idx];
            v.position[1] -= depth;
            let pos = Vec3::from(v.position);
            *aabb_min = aabb_min.min(pos);
            *aabb_max = aabb_max.max(pos);
            verts.push(v);
        };

        // Bottom edge (z = 0)
        let edge_start = vertices.len() as u32;
        for x in 0..verts_x {
            let src = x;
            add_skirt_vertex(vertices, src, skirt_depth, aabb_min, aabb_max);
        }
        for x in 0..(verts_x - 1) {
            let top = x as u32;
            let top_next = top + 1;
            let bot = edge_start + x as u32;
            let bot_next = bot + 1;
            indices.push(top);
            indices.push(bot);
            indices.push(top_next);
            indices.push(top_next);
            indices.push(bot);
            indices.push(bot_next);
        }

        // Top edge (z = max)
        let edge_start = vertices.len() as u32;
        for x in 0..verts_x {
            let src = (verts_z - 1) * verts_x + x;
            add_skirt_vertex(vertices, src, skirt_depth, aabb_min, aabb_max);
        }
        for x in 0..(verts_x - 1) {
            let top = ((verts_z - 1) * verts_x + x) as u32;
            let top_next = top + 1;
            let bot = edge_start + x as u32;
            let bot_next = bot + 1;
            // Reversed winding for back face
            indices.push(top);
            indices.push(top_next);
            indices.push(bot);
            indices.push(top_next);
            indices.push(bot_next);
            indices.push(bot);
        }

        // Left edge (x = 0)
        let edge_start = vertices.len() as u32;
        for z in 0..verts_z {
            let src = z * verts_x;
            add_skirt_vertex(vertices, src, skirt_depth, aabb_min, aabb_max);
        }
        for z in 0..(verts_z - 1) {
            let top = (z * verts_x) as u32;
            let top_next = ((z + 1) * verts_x) as u32;
            let bot = edge_start + z as u32;
            let bot_next = bot + 1;
            indices.push(top);
            indices.push(top_next);
            indices.push(bot);
            indices.push(top_next);
            indices.push(bot_next);
            indices.push(bot);
        }

        // Right edge (x = max)
        let edge_start = vertices.len() as u32;
        for z in 0..verts_z {
            let src = z * verts_x + (verts_x - 1);
            add_skirt_vertex(vertices, src, skirt_depth, aabb_min, aabb_max);
        }
        for z in 0..(verts_z - 1) {
            let top = (z * verts_x + (verts_x - 1)) as u32;
            let top_next = ((z + 1) * verts_x + (verts_x - 1)) as u32;
            let bot = edge_start + z as u32;
            let bot_next = bot + 1;
            indices.push(top);
            indices.push(bot);
            indices.push(top_next);
            indices.push(top_next);
            indices.push(bot);
            indices.push(bot_next);
        }

        let _ = base_count; // suppress unused warning
    }

    /// Generates a wireframe index buffer from the mesh data (for debug
    /// visualization).
    pub fn generate_wireframe_indices(mesh: &MeshData) -> Vec<u32> {
        let mut lines = Vec::with_capacity(mesh.indices.len() * 2);
        for tri in mesh.indices.chunks_exact(3) {
            lines.push(tri[0]);
            lines.push(tri[1]);
            lines.push(tri[1]);
            lines.push(tri[2]);
            lines.push(tri[2]);
            lines.push(tri[0]);
        }
        lines
    }

    /// Computes the morph-adjusted position for LOD transitions.
    ///
    /// `morph_factor` in `[0, 1]` blends between the current LOD vertex
    /// position and the coarser LOD position. Used to prevent popping
    /// artifacts when transitioning between LOD levels.
    pub fn compute_morph_position(
        fine_pos: Vec3,
        coarse_pos: Vec3,
        morph_factor: f32,
    ) -> Vec3 {
        fine_pos.lerp(coarse_pos, morph_factor)
    }

    /// Computes the morph factor for a vertex based on its distance from the
    /// camera and the LOD transition range.
    ///
    /// Returns a value in `[0, 1]` where 0 = fully fine LOD and 1 = fully
    /// coarse LOD.
    pub fn compute_morph_factor(
        distance: f32,
        lod_start: f32,
        lod_end: f32,
    ) -> f32 {
        if distance <= lod_start {
            return 0.0;
        }
        if distance >= lod_end {
            return 1.0;
        }
        let range = lod_end - lod_start;
        if range < f32::EPSILON {
            return 1.0;
        }
        (distance - lod_start) / range
    }
}

// ---------------------------------------------------------------------------
// ChunkMeshData
// ---------------------------------------------------------------------------

/// Mesh data for a single terrain chunk, with its grid position.
#[derive(Debug, Clone)]
pub struct ChunkMeshData {
    /// Column index of this chunk in the terrain grid.
    pub chunk_x: u32,
    /// Row index of this chunk in the terrain grid.
    pub chunk_z: u32,
    /// The generated mesh data.
    pub mesh: MeshData,
}

// ---------------------------------------------------------------------------
// LOD mesh strip generation
// ---------------------------------------------------------------------------

/// Generates index buffers at multiple resolutions for GeoMipMap-style LOD.
///
/// Returns a vector of index buffers, one per LOD level. Level 0 is the
/// finest (full resolution). Each subsequent level halves the resolution.
pub fn generate_lod_index_buffers(
    verts_x: usize,
    verts_z: usize,
    num_lod_levels: usize,
) -> Vec<Vec<u32>> {
    let mut lod_indices = Vec::with_capacity(num_lod_levels);

    for lod in 0..num_lod_levels {
        let step = 1usize << lod;
        let mut indices = Vec::new();

        let mut z = 0;
        while z + step < verts_z {
            let mut x = 0;
            while x + step < verts_x {
                let tl = (z * verts_x + x) as u32;
                let tr = (z * verts_x + (x + step)) as u32;
                let bl = ((z + step) * verts_x + x) as u32;
                let br = ((z + step) * verts_x + (x + step)) as u32;

                indices.push(tl);
                indices.push(bl);
                indices.push(tr);

                indices.push(tr);
                indices.push(bl);
                indices.push(br);

                x += step;
            }
            z += step;
        }

        lod_indices.push(indices);
    }

    lod_indices
}

/// Generates stitching indices between two adjacent chunks at different LOD
/// levels. Eliminates T-junctions along the shared edge.
///
/// `edge` specifies which edge of the higher-detail chunk borders the
/// lower-detail chunk.
pub fn generate_stitch_indices(
    verts_per_side: usize,
    fine_lod: usize,
    coarse_lod: usize,
    edge: ChunkEdge,
) -> Vec<u32> {
    let fine_step = 1usize << fine_lod;
    let coarse_step = 1usize << coarse_lod;

    if coarse_step <= fine_step {
        return Vec::new(); // no stitching needed
    }

    let mut indices = Vec::new();
    let n = verts_per_side;

    match edge {
        ChunkEdge::Bottom => {
            // z = 0 edge
            let mut x = 0;
            while x + coarse_step < n {
                let coarse_left = x as u32;
                let coarse_right = (x + coarse_step) as u32;
                let inner_row = fine_step;

                // Fan from coarse vertices to fine vertices on inner row
                let mut fx = x;
                while fx + fine_step <= x + coarse_step {
                    let fine_vert = (inner_row * n + fx) as u32;
                    let fine_next = (inner_row * n + (fx + fine_step).min(n - 1)) as u32;

                    if fx == x {
                        indices.push(coarse_left);
                        indices.push(fine_vert);
                        indices.push(fine_next);
                    } else if fx + fine_step >= x + coarse_step {
                        indices.push(fine_vert);
                        indices.push(coarse_right);
                        indices.push(coarse_left);
                    } else {
                        indices.push(coarse_left);
                        indices.push(fine_vert);
                        indices.push(fine_next);
                    }

                    fx += fine_step;
                }

                x += coarse_step;
            }
        }
        ChunkEdge::Top => {
            let mut x = 0;
            let z_edge = n - 1;
            while x + coarse_step < n {
                let coarse_left = (z_edge * n + x) as u32;
                let coarse_right = (z_edge * n + x + coarse_step) as u32;
                let inner_row = z_edge - fine_step;

                let mut fx = x;
                while fx + fine_step <= x + coarse_step {
                    let fine_vert = (inner_row * n + fx) as u32;
                    let fine_next = (inner_row * n + (fx + fine_step).min(n - 1)) as u32;

                    if fx == x {
                        indices.push(coarse_left);
                        indices.push(fine_next);
                        indices.push(fine_vert);
                    } else if fx + fine_step >= x + coarse_step {
                        indices.push(fine_vert);
                        indices.push(coarse_left);
                        indices.push(coarse_right);
                    } else {
                        indices.push(coarse_left);
                        indices.push(fine_next);
                        indices.push(fine_vert);
                    }

                    fx += fine_step;
                }

                x += coarse_step;
            }
        }
        ChunkEdge::Left => {
            let mut z = 0;
            while z + coarse_step < n {
                let coarse_top = (z * n) as u32;
                let coarse_bot = ((z + coarse_step) * n) as u32;
                let inner_col = fine_step;

                let mut fz = z;
                while fz + fine_step <= z + coarse_step {
                    let fine_vert = (fz * n + inner_col) as u32;
                    let fine_next = (((fz + fine_step).min(n - 1)) * n + inner_col) as u32;

                    if fz == z {
                        indices.push(coarse_top);
                        indices.push(fine_next);
                        indices.push(fine_vert);
                    } else if fz + fine_step >= z + coarse_step {
                        indices.push(fine_vert);
                        indices.push(coarse_top);
                        indices.push(coarse_bot);
                    } else {
                        indices.push(coarse_top);
                        indices.push(fine_next);
                        indices.push(fine_vert);
                    }

                    fz += fine_step;
                }

                z += coarse_step;
            }
        }
        ChunkEdge::Right => {
            let mut z = 0;
            let x_edge = n - 1;
            while z + coarse_step < n {
                let coarse_top = (z * n + x_edge) as u32;
                let coarse_bot = ((z + coarse_step) * n + x_edge) as u32;
                let inner_col = x_edge - fine_step;

                let mut fz = z;
                while fz + fine_step <= z + coarse_step {
                    let fine_vert = (fz * n + inner_col) as u32;
                    let fine_next = (((fz + fine_step).min(n - 1)) * n + inner_col) as u32;

                    if fz == z {
                        indices.push(coarse_top);
                        indices.push(fine_vert);
                        indices.push(fine_next);
                    } else if fz + fine_step >= z + coarse_step {
                        indices.push(fine_vert);
                        indices.push(coarse_bot);
                        indices.push(coarse_top);
                    } else {
                        indices.push(coarse_top);
                        indices.push(fine_vert);
                        indices.push(fine_next);
                    }

                    fz += fine_step;
                }

                z += coarse_step;
            }
        }
    }

    indices
}

/// Identifies which edge of a chunk borders another chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkEdge {
    /// Z = 0 edge (south).
    Bottom,
    /// Z = max edge (north).
    Top,
    /// X = 0 edge (west).
    Left,
    /// X = max edge (east).
    Right,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_mesh_generation() {
        let hm = crate::Heightmap::new_flat(17, 17, 0.5).unwrap();
        let settings = TerrainMeshSettings {
            generate_skirts: false,
            ..Default::default()
        };
        let mesh = TerrainMesh::generate_mesh(&hm, 1, &settings).unwrap();
        assert_eq!(mesh.vertex_count(), 17 * 17);
        assert_eq!(mesh.triangle_count(), 16 * 16 * 2);
    }

    #[test]
    fn mesh_with_skirts() {
        let hm = crate::Heightmap::new_flat(9, 9, 0.0).unwrap();
        let settings = TerrainMeshSettings::default();
        let mesh = TerrainMesh::generate_mesh(&hm, 1, &settings).unwrap();
        // With skirts, vertex count > 9*9
        assert!(mesh.vertex_count() > 81);
    }

    #[test]
    fn chunked_generation() {
        let hm = crate::Heightmap::new_flat(33, 33, 1.0).unwrap();
        let settings = TerrainMeshSettings {
            generate_skirts: false,
            ..Default::default()
        };
        let chunks = TerrainMesh::generate_chunked(&hm, 4, 4, 1, &settings).unwrap();
        assert_eq!(chunks.len(), 16);
    }

    #[test]
    fn lod_index_buffers() {
        let buffers = generate_lod_index_buffers(17, 17, 4);
        assert_eq!(buffers.len(), 4);
        // Each level should have fewer indices
        for i in 1..buffers.len() {
            assert!(buffers[i].len() < buffers[i - 1].len());
        }
    }

    #[test]
    fn morph_factor_range() {
        assert_eq!(TerrainMesh::compute_morph_factor(10.0, 50.0, 100.0), 0.0);
        assert_eq!(TerrainMesh::compute_morph_factor(200.0, 50.0, 100.0), 1.0);
        let mid = TerrainMesh::compute_morph_factor(75.0, 50.0, 100.0);
        assert!((mid - 0.5).abs() < 1e-5);
    }
}
