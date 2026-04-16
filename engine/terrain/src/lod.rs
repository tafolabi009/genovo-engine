//! Terrain Level-of-Detail (LOD) systems.
//!
//! Provides two complementary LOD approaches:
//!
//! - **CDLOD** (Continuous Distance-Dependent LOD): A quadtree-based system
//!   that selects visible terrain chunks at appropriate detail levels based on
//!   camera distance. Supports morphing between LOD levels for seamless
//!   transitions.
//!
//! - **GeoMipMap**: Pre-computed meshes at multiple resolutions with stitching
//!   strips to eliminate T-junctions between chunks at different LOD levels.

use genovo_core::{AABB, Frustum};
use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::heightmap::Heightmap;
use crate::TerrainResult;

// ---------------------------------------------------------------------------
// LOD settings
// ---------------------------------------------------------------------------

/// Configuration for the terrain LOD system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LODSettings {
    /// Distance thresholds for each LOD level (in world units).
    ///
    /// `lod_distances[0]` is the distance at which LOD 0 (finest) ends and
    /// LOD 1 begins, and so on.
    pub lod_distances: Vec<f32>,

    /// The fraction of each LOD range over which morphing occurs.
    ///
    /// A value of 0.3 means the last 30% of each LOD band transitions
    /// smoothly to the next coarser level.
    pub morph_range: f32,

    /// Minimum LOD level (0 = finest detail).
    pub min_lod: u32,

    /// Maximum LOD level.
    pub max_lod: u32,

    /// The base size of a terrain chunk at the finest LOD level (world units).
    pub chunk_size: f32,

    /// Error threshold in screen-space pixels. Nodes with projected error
    /// below this threshold are not subdivided further.
    pub pixel_error_threshold: f32,
}

impl Default for LODSettings {
    fn default() -> Self {
        Self {
            lod_distances: vec![100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0, 12800.0],
            morph_range: 0.3,
            min_lod: 0,
            max_lod: 7,
            chunk_size: 64.0,
            pixel_error_threshold: 2.0,
        }
    }
}

impl LODSettings {
    /// Returns the distance threshold for a given LOD level.
    pub fn distance_for_lod(&self, lod: u32) -> f32 {
        let idx = lod as usize;
        if idx < self.lod_distances.len() {
            self.lod_distances[idx]
        } else {
            *self.lod_distances.last().unwrap_or(&f32::MAX)
        }
    }

    /// Returns the morph start distance for a given LOD level.
    ///
    /// Morphing begins at `distance * (1 - morph_range)` and ends at
    /// `distance`.
    pub fn morph_start_for_lod(&self, lod: u32) -> f32 {
        let dist = self.distance_for_lod(lod);
        dist * (1.0 - self.morph_range)
    }

    /// Computes the LOD level for a given distance from the camera.
    pub fn lod_for_distance(&self, distance: f32) -> u32 {
        for (i, &threshold) in self.lod_distances.iter().enumerate() {
            if distance < threshold {
                return (i as u32).max(self.min_lod);
            }
        }
        self.max_lod
    }

    /// Computes the morph factor for a given distance and LOD level.
    ///
    /// Returns `0.0` when fully at the current LOD and `1.0` when fully
    /// transitioned to the next coarser LOD.
    pub fn morph_factor(&self, distance: f32, lod: u32) -> f32 {
        let lod_end = self.distance_for_lod(lod);
        let lod_start = self.morph_start_for_lod(lod);

        if distance <= lod_start {
            0.0
        } else if distance >= lod_end {
            1.0
        } else {
            (distance - lod_start) / (lod_end - lod_start)
        }
    }
}

// ---------------------------------------------------------------------------
// TerrainChunkInfo — selected chunk data
// ---------------------------------------------------------------------------

/// Information about a terrain chunk selected by the LOD system for rendering.
#[derive(Debug, Clone)]
pub struct TerrainChunkInfo {
    /// LOD level of this chunk (0 = finest).
    pub lod_level: u32,
    /// World-space AABB of this chunk.
    pub aabb: AABB,
    /// World-space center of this chunk.
    pub center: Vec3,
    /// Size of this chunk in world units.
    pub size: f32,
    /// The column index in the quadtree grid at this LOD level.
    pub grid_x: u32,
    /// The row index in the quadtree grid at this LOD level.
    pub grid_z: u32,
    /// Morph factor for smooth LOD transitions (0.0 = current, 1.0 = coarser).
    pub morph_factor: f32,
    /// Distance from the camera to the chunk center.
    pub distance: f32,
    /// LOD level of the neighbor in -X direction (for stitching).
    pub neighbor_lod_neg_x: u32,
    /// LOD level of the neighbor in +X direction.
    pub neighbor_lod_pos_x: u32,
    /// LOD level of the neighbor in -Z direction.
    pub neighbor_lod_neg_z: u32,
    /// LOD level of the neighbor in +Z direction.
    pub neighbor_lod_pos_z: u32,
}

// ---------------------------------------------------------------------------
// QuadtreeNode
// ---------------------------------------------------------------------------

/// A node in the terrain quadtree. Each node represents a square region of
/// terrain and may have four children for finer subdivision.
#[derive(Debug, Clone)]
struct QuadtreeNode {
    /// Axis-aligned bounding box of this node.
    aabb: AABB,
    /// The LOD level (0 = root / coarsest in the tree).
    level: u32,
    /// Grid position at this level.
    grid_x: u32,
    grid_z: u32,
    /// Size of this node in world units.
    size: f32,
    /// Indices of the four children in the node array, or `None` if leaf.
    children: Option<[u32; 4]>,
    /// Minimum height in this node's region.
    min_height: f32,
    /// Maximum height in this node's region.
    max_height: f32,
}

// ---------------------------------------------------------------------------
// TerrainQuadtree — CDLOD implementation
// ---------------------------------------------------------------------------

/// A quadtree for Continuous Distance-Dependent Level-of-Detail (CDLOD)
/// terrain rendering.
///
/// The quadtree subdivides the terrain into a hierarchy of square regions.
/// At runtime, the tree is traversed from the root, and nodes are selected
/// or further subdivided based on camera distance and frustum visibility.
///
/// # Algorithm
///
/// 1. Start at the root node (covering the entire terrain).
/// 2. For each node, compute the distance from the camera to the node center.
/// 3. If the distance is greater than the LOD threshold for this level, render
///    this node (do not subdivide further).
/// 4. If the node has children and the distance warrants finer detail,
///    recurse into the four children.
/// 5. Compute morph factors for nodes near LOD boundaries to enable smooth
///    transitions.
pub struct TerrainQuadtree {
    /// All nodes in the quadtree, stored in a flat array.
    nodes: Vec<QuadtreeNode>,
    /// Index of the root node.
    root: u32,
    /// The maximum depth of the tree (number of subdivision levels).
    max_depth: u32,
    /// Total terrain size in world units.
    terrain_size: f32,
    /// LOD settings.
    settings: LODSettings,
}

impl TerrainQuadtree {
    /// Builds a new terrain quadtree from a heightmap.
    ///
    /// `max_depth` controls the number of subdivision levels (6-8 is typical).
    /// The heightmap is sampled to determine AABB height bounds for each node.
    #[profiling::function]
    pub fn build(
        heightmap: &Heightmap,
        terrain_size: f32,
        max_depth: u32,
        settings: LODSettings,
    ) -> Self {
        let max_depth = max_depth.max(1).min(12);

        // Estimate node count: sum of 4^i for i in 0..max_depth
        let estimated_nodes: usize = (0..max_depth).map(|i| 4usize.pow(i)).sum();
        let mut nodes = Vec::with_capacity(estimated_nodes);

        // Build recursively
        let root = Self::build_node(
            &mut nodes,
            heightmap,
            terrain_size,
            0,
            0,
            terrain_size,
            0,
            max_depth,
        );

        Self {
            nodes,
            root,
            max_depth,
            terrain_size,
            settings,
        }
    }

    fn build_node(
        nodes: &mut Vec<QuadtreeNode>,
        heightmap: &Heightmap,
        terrain_size: f32,
        grid_x: u32,
        grid_z: u32,
        node_size: f32,
        depth: u32,
        max_depth: u32,
    ) -> u32 {
        let hm_w = heightmap.width() as f32;
        let hm_h = heightmap.height() as f32;

        // Compute AABB for this node
        let world_x = grid_x as f32 * node_size;
        let world_z = grid_z as f32 * node_size;

        // Sample heightmap to find min/max in this region
        let hm_start_x = (world_x / terrain_size * (hm_w - 1.0)) as u32;
        let hm_start_z = (world_z / terrain_size * (hm_h - 1.0)) as u32;
        let hm_end_x =
            ((world_x + node_size) / terrain_size * (hm_w - 1.0)).ceil() as u32;
        let hm_end_z =
            ((world_z + node_size) / terrain_size * (hm_h - 1.0)).ceil() as u32;

        let hm_end_x = hm_end_x.min(heightmap.width() - 1);
        let hm_end_z = hm_end_z.min(heightmap.height() - 1);

        let mut min_h = f32::INFINITY;
        let mut max_h = f32::NEG_INFINITY;

        // Sample at a reasonable stride to avoid O(n^2) for large nodes
        let stride = ((hm_end_x - hm_start_x) / 16).max(1);
        let stride_z = ((hm_end_z - hm_start_z) / 16).max(1);

        let mut sz = hm_start_z;
        while sz <= hm_end_z {
            let mut sx = hm_start_x;
            while sx <= hm_end_x {
                let h = heightmap.get(sx, sz);
                if h < min_h {
                    min_h = h;
                }
                if h > max_h {
                    max_h = h;
                }
                sx += stride;
            }
            sz += stride_z;
        }

        // Also check corners and edges exactly
        for &cz in &[hm_start_z, hm_end_z] {
            for &cx in &[hm_start_x, hm_end_x] {
                let h = heightmap.get(cx, cz);
                if h < min_h {
                    min_h = h;
                }
                if h > max_h {
                    max_h = h;
                }
            }
        }

        if min_h == f32::INFINITY {
            min_h = 0.0;
            max_h = 0.0;
        }

        let aabb = AABB::new(
            Vec3::new(world_x, min_h, world_z),
            Vec3::new(world_x + node_size, max_h, world_z + node_size),
        );

        // The LOD level maps from tree depth: deepest = finest = LOD 0
        let lod_level = max_depth - 1 - depth;

        let node_idx = nodes.len() as u32;
        nodes.push(QuadtreeNode {
            aabb,
            level: lod_level,
            grid_x,
            grid_z,
            size: node_size,
            children: None,
            min_height: min_h,
            max_height: max_h,
        });

        // Subdivide if not at max depth
        if depth + 1 < max_depth {
            let child_size = node_size * 0.5;
            let child_grid_x = grid_x * 2;
            let child_grid_z = grid_z * 2;

            let c0 = Self::build_node(
                nodes,
                heightmap,
                terrain_size,
                child_grid_x,
                child_grid_z,
                child_size,
                depth + 1,
                max_depth,
            );
            let c1 = Self::build_node(
                nodes,
                heightmap,
                terrain_size,
                child_grid_x + 1,
                child_grid_z,
                child_size,
                depth + 1,
                max_depth,
            );
            let c2 = Self::build_node(
                nodes,
                heightmap,
                terrain_size,
                child_grid_x,
                child_grid_z + 1,
                child_size,
                depth + 1,
                max_depth,
            );
            let c3 = Self::build_node(
                nodes,
                heightmap,
                terrain_size,
                child_grid_x + 1,
                child_grid_z + 1,
                child_size,
                depth + 1,
                max_depth,
            );

            nodes[node_idx as usize].children = Some([c0, c1, c2, c3]);
        }

        node_idx
    }

    /// Selects visible terrain chunks based on camera position and frustum.
    ///
    /// Traverses the quadtree and returns a list of chunks to render, each
    /// with the appropriate LOD level and morph factor.
    #[profiling::function]
    pub fn select_visible_nodes(
        &self,
        camera_pos: Vec3,
        frustum: &Frustum,
    ) -> Vec<TerrainChunkInfo> {
        let mut result = Vec::with_capacity(256);
        self.select_node(self.root, camera_pos, frustum, &mut result);

        // Compute neighbor LOD levels for stitching
        self.compute_neighbor_lods(&mut result);

        result
    }

    fn select_node(
        &self,
        node_idx: u32,
        camera_pos: Vec3,
        frustum: &Frustum,
        result: &mut Vec<TerrainChunkInfo>,
    ) {
        let node = &self.nodes[node_idx as usize];

        // Frustum culling
        if !frustum.contains_aabb(&node.aabb) {
            return;
        }

        // Distance from camera to node center
        let center = node.aabb.center();
        let distance = camera_pos.distance(center);

        // Determine if we should subdivide or render at this level
        let lod_threshold = self.settings.distance_for_lod(node.level);
        let should_subdivide = distance < lod_threshold
            && node.children.is_some()
            && node.level > self.settings.min_lod;

        if should_subdivide {
            // Recurse into children
            if let Some(children) = &node.children {
                for &child_idx in children {
                    self.select_node(child_idx, camera_pos, frustum, result);
                }
            }
        } else {
            // Render this node
            let morph = self.settings.morph_factor(distance, node.level);

            result.push(TerrainChunkInfo {
                lod_level: node.level,
                aabb: node.aabb,
                center,
                size: node.size,
                grid_x: node.grid_x,
                grid_z: node.grid_z,
                morph_factor: morph,
                distance,
                neighbor_lod_neg_x: node.level,
                neighbor_lod_pos_x: node.level,
                neighbor_lod_neg_z: node.level,
                neighbor_lod_pos_z: node.level,
            });
        }
    }

    /// Computes neighbor LOD levels for each selected chunk.
    ///
    /// This information is needed for generating stitching strips to
    /// eliminate T-junctions between adjacent chunks at different LODs.
    fn compute_neighbor_lods(&self, chunks: &mut Vec<TerrainChunkInfo>) {
        // Build a spatial map of selected chunks for O(1) neighbor lookup
        let mut lod_map: std::collections::HashMap<(i32, i32, u32), u32> =
            std::collections::HashMap::new();

        for chunk in chunks.iter() {
            lod_map.insert(
                (chunk.grid_x as i32, chunk.grid_z as i32, chunk.lod_level),
                chunk.lod_level,
            );
        }

        // For each chunk, find the LOD of its neighbors
        for chunk in chunks.iter_mut() {
            let gx = chunk.grid_x as i32;
            let gz = chunk.grid_z as i32;
            let lod = chunk.lod_level;

            // Check neighbors at same LOD level first, then coarser levels
            chunk.neighbor_lod_neg_x = Self::find_neighbor_lod(&lod_map, gx - 1, gz, lod);
            chunk.neighbor_lod_pos_x = Self::find_neighbor_lod(&lod_map, gx + 1, gz, lod);
            chunk.neighbor_lod_neg_z = Self::find_neighbor_lod(&lod_map, gx, gz - 1, lod);
            chunk.neighbor_lod_pos_z = Self::find_neighbor_lod(&lod_map, gx, gz + 1, lod);
        }
    }

    fn find_neighbor_lod(
        lod_map: &std::collections::HashMap<(i32, i32, u32), u32>,
        gx: i32,
        gz: i32,
        current_lod: u32,
    ) -> u32 {
        // Look for neighbor at same LOD
        if let Some(&lod) = lod_map.get(&(gx, gz, current_lod)) {
            return lod;
        }
        // Look for neighbor at coarser LOD (parent node covers this region)
        for lod_check in (current_lod + 1)..=(current_lod + 4) {
            let scale = 1i32 << (lod_check - current_lod);
            let parent_gx = if gx >= 0 { gx / scale } else { (gx - scale + 1) / scale };
            let parent_gz = if gz >= 0 { gz / scale } else { (gz - scale + 1) / scale };
            if let Some(&lod) = lod_map.get(&(parent_gx, parent_gz, lod_check)) {
                return lod;
            }
        }
        current_lod
    }

    /// Returns the total number of nodes in the quadtree.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the maximum depth of the quadtree.
    #[inline]
    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }

    /// Returns the terrain size in world units.
    #[inline]
    pub fn terrain_size(&self) -> f32 {
        self.terrain_size
    }

    /// Returns a reference to the LOD settings.
    #[inline]
    pub fn settings(&self) -> &LODSettings {
        &self.settings
    }

    /// Updates the LOD settings.
    #[inline]
    pub fn set_settings(&mut self, settings: LODSettings) {
        self.settings = settings;
    }

    /// Computes the geometric error for a given LOD level.
    ///
    /// The error is the maximum height difference that is "lost" when
    /// rendering at a coarser resolution. This is used for screen-space
    /// error metric LOD selection.
    pub fn geometric_error(&self, lod_level: u32) -> f32 {
        // At LOD 0 (finest), error is zero. At coarser levels,
        // the error doubles with each level.
        let base_error = self.terrain_size / (1u32 << self.max_depth) as f32;
        base_error * (1u32 << lod_level) as f32
    }

    /// Computes the screen-space error for a given geometric error and
    /// distance from the camera.
    ///
    /// `fov_y` is the vertical field of view in radians.
    /// `viewport_height` is the viewport height in pixels.
    pub fn screen_space_error(
        geometric_error: f32,
        distance: f32,
        fov_y: f32,
        viewport_height: f32,
    ) -> f32 {
        if distance < f32::EPSILON {
            return f32::MAX;
        }
        let projection_factor = viewport_height / (2.0 * (fov_y * 0.5).tan());
        geometric_error * projection_factor / distance
    }

    /// Selects visible nodes using screen-space error metric instead of
    /// fixed distance thresholds.
    ///
    /// This produces more consistent visual quality across different
    /// viewport sizes and FOV settings.
    #[profiling::function]
    pub fn select_visible_nodes_sse(
        &self,
        camera_pos: Vec3,
        frustum: &Frustum,
        fov_y: f32,
        viewport_height: f32,
    ) -> Vec<TerrainChunkInfo> {
        let mut result = Vec::with_capacity(256);
        self.select_node_sse(
            self.root,
            camera_pos,
            frustum,
            fov_y,
            viewport_height,
            &mut result,
        );
        self.compute_neighbor_lods(&mut result);
        result
    }

    fn select_node_sse(
        &self,
        node_idx: u32,
        camera_pos: Vec3,
        frustum: &Frustum,
        fov_y: f32,
        viewport_height: f32,
        result: &mut Vec<TerrainChunkInfo>,
    ) {
        let node = &self.nodes[node_idx as usize];

        if !frustum.contains_aabb(&node.aabb) {
            return;
        }

        let center = node.aabb.center();
        let distance = camera_pos.distance(center);
        let geo_error = self.geometric_error(node.level);
        let sse = Self::screen_space_error(geo_error, distance, fov_y, viewport_height);

        let should_subdivide = sse > self.settings.pixel_error_threshold
            && node.children.is_some()
            && node.level > self.settings.min_lod;

        if should_subdivide {
            if let Some(children) = &node.children {
                for &child_idx in children {
                    self.select_node_sse(
                        child_idx,
                        camera_pos,
                        frustum,
                        fov_y,
                        viewport_height,
                        result,
                    );
                }
            }
        } else {
            let morph = self.settings.morph_factor(distance, node.level);
            result.push(TerrainChunkInfo {
                lod_level: node.level,
                aabb: node.aabb,
                center,
                size: node.size,
                grid_x: node.grid_x,
                grid_z: node.grid_z,
                morph_factor: morph,
                distance,
                neighbor_lod_neg_x: node.level,
                neighbor_lod_pos_x: node.level,
                neighbor_lod_neg_z: node.level,
                neighbor_lod_pos_z: node.level,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// GeoMipMap — alternative LOD strategy
// ---------------------------------------------------------------------------

/// Pre-computed LOD meshes using the GeoMipMapping approach.
///
/// Each chunk has a set of pre-built index buffers at different resolutions.
/// At runtime, the appropriate resolution is selected based on distance.
/// Stitching strips handle T-junction elimination between adjacent chunks
/// at different LOD levels.
pub struct GeoMipMap {
    /// Number of chunks along the X axis.
    chunks_x: u32,
    /// Number of chunks along the Z axis.
    chunks_z: u32,
    /// Number of vertices per chunk side.
    chunk_verts: u32,
    /// Number of LOD levels per chunk.
    num_lod_levels: u32,
    /// Pre-computed index buffers per chunk per LOD level.
    /// Indexed as [chunk_z * chunks_x + chunk_x][lod_level].
    chunk_index_buffers: Vec<Vec<Vec<u32>>>,
    /// Pre-computed stitching index buffers.
    /// Indexed by (fine_lod, coarse_lod, edge).
    stitch_buffers: std::collections::HashMap<(u32, u32, u32), Vec<u32>>,
    /// Current LOD level per chunk.
    chunk_lod_levels: Vec<u32>,
}

impl GeoMipMap {
    /// Creates a new GeoMipMap LOD system.
    ///
    /// `chunks_x` and `chunks_z` are the number of chunks along each axis.
    /// `chunk_verts` is the number of vertices per chunk side (should be
    /// 2^n + 1, e.g. 17, 33, 65).
    /// `num_lod_levels` is the number of LOD levels to pre-compute.
    pub fn new(
        chunks_x: u32,
        chunks_z: u32,
        chunk_verts: u32,
        num_lod_levels: u32,
    ) -> Self {
        let num_chunks = (chunks_x * chunks_z) as usize;
        let cv = chunk_verts as usize;

        // Pre-compute index buffers for each LOD level
        let base_indices = crate::mesh_generation::generate_lod_index_buffers(
            cv,
            cv,
            num_lod_levels as usize,
        );

        let mut chunk_index_buffers = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            chunk_index_buffers.push(base_indices.clone());
        }

        // Pre-compute stitching buffers for all LOD level combinations
        let mut stitch_buffers = std::collections::HashMap::new();
        for fine in 0..num_lod_levels {
            for coarse in (fine + 1)..num_lod_levels {
                for edge_idx in 0..4u32 {
                    let edge = match edge_idx {
                        0 => crate::mesh_generation::ChunkEdge::Bottom,
                        1 => crate::mesh_generation::ChunkEdge::Top,
                        2 => crate::mesh_generation::ChunkEdge::Left,
                        _ => crate::mesh_generation::ChunkEdge::Right,
                    };
                    let stitches = crate::mesh_generation::generate_stitch_indices(
                        cv,
                        fine as usize,
                        coarse as usize,
                        edge,
                    );
                    stitch_buffers.insert((fine, coarse, edge_idx), stitches);
                }
            }
        }

        let chunk_lod_levels = vec![0u32; num_chunks];

        Self {
            chunks_x,
            chunks_z,
            chunk_verts,
            num_lod_levels,
            chunk_index_buffers,
            stitch_buffers,
            chunk_lod_levels,
        }
    }

    /// Updates LOD levels for all chunks based on camera position.
    pub fn update(
        &mut self,
        camera_pos: Vec3,
        chunk_size: f32,
        settings: &LODSettings,
    ) {
        for cz in 0..self.chunks_z {
            for cx in 0..self.chunks_x {
                let center_x = (cx as f32 + 0.5) * chunk_size;
                let center_z = (cz as f32 + 0.5) * chunk_size;
                let center = Vec3::new(center_x, 0.0, center_z);
                let dist = camera_pos.distance(center);

                let lod = settings.lod_for_distance(dist);
                let lod = lod.min(self.num_lod_levels - 1);
                let idx = (cz * self.chunks_x + cx) as usize;
                self.chunk_lod_levels[idx] = lod;
            }
        }
    }

    /// Returns the current LOD level for a chunk.
    #[inline]
    pub fn chunk_lod(&self, chunk_x: u32, chunk_z: u32) -> u32 {
        let idx = (chunk_z * self.chunks_x + chunk_x) as usize;
        self.chunk_lod_levels[idx]
    }

    /// Returns the index buffer for a chunk at its current LOD level.
    pub fn chunk_indices(&self, chunk_x: u32, chunk_z: u32) -> &[u32] {
        let chunk_idx = (chunk_z * self.chunks_x + chunk_x) as usize;
        let lod = self.chunk_lod_levels[chunk_idx] as usize;
        &self.chunk_index_buffers[chunk_idx][lod]
    }

    /// Returns the stitching indices for the edge between two adjacent chunks.
    ///
    /// Returns `None` if the chunks are at the same LOD level (no stitching
    /// needed).
    pub fn stitch_indices(
        &self,
        chunk_x: u32,
        chunk_z: u32,
        edge: crate::mesh_generation::ChunkEdge,
    ) -> Option<&[u32]> {
        let fine_lod = self.chunk_lod(chunk_x, chunk_z);

        // Find neighbor
        let (nx, nz) = match edge {
            crate::mesh_generation::ChunkEdge::Left => {
                if chunk_x == 0 {
                    return None;
                }
                (chunk_x - 1, chunk_z)
            }
            crate::mesh_generation::ChunkEdge::Right => {
                if chunk_x + 1 >= self.chunks_x {
                    return None;
                }
                (chunk_x + 1, chunk_z)
            }
            crate::mesh_generation::ChunkEdge::Bottom => {
                if chunk_z == 0 {
                    return None;
                }
                (chunk_x, chunk_z - 1)
            }
            crate::mesh_generation::ChunkEdge::Top => {
                if chunk_z + 1 >= self.chunks_z {
                    return None;
                }
                (chunk_x, chunk_z + 1)
            }
        };

        let coarse_lod = self.chunk_lod(nx, nz);

        if coarse_lod <= fine_lod {
            return None;
        }

        let edge_idx = match edge {
            crate::mesh_generation::ChunkEdge::Bottom => 0,
            crate::mesh_generation::ChunkEdge::Top => 1,
            crate::mesh_generation::ChunkEdge::Left => 2,
            crate::mesh_generation::ChunkEdge::Right => 3,
        };

        self.stitch_buffers
            .get(&(fine_lod, coarse_lod, edge_idx))
            .map(|v| v.as_slice())
    }

    /// Returns the number of chunks along the X axis.
    #[inline]
    pub fn chunks_x(&self) -> u32 {
        self.chunks_x
    }

    /// Returns the number of chunks along the Z axis.
    #[inline]
    pub fn chunks_z(&self) -> u32 {
        self.chunks_z
    }

    /// Returns the number of LOD levels.
    #[inline]
    pub fn num_lod_levels(&self) -> u32 {
        self.num_lod_levels
    }

    /// Returns the total number of chunks.
    #[inline]
    pub fn total_chunks(&self) -> usize {
        (self.chunks_x * self.chunks_z) as usize
    }

    /// Returns per-chunk statistics for debugging.
    pub fn debug_stats(&self) -> GeoMipMapStats {
        let mut lod_counts = vec![0u32; self.num_lod_levels as usize];
        for &lod in &self.chunk_lod_levels {
            if (lod as usize) < lod_counts.len() {
                lod_counts[lod as usize] += 1;
            }
        }
        GeoMipMapStats {
            total_chunks: self.total_chunks(),
            chunks_per_lod: lod_counts,
        }
    }
}

/// Debugging statistics for the GeoMipMap system.
#[derive(Debug, Clone)]
pub struct GeoMipMapStats {
    /// Total number of chunks.
    pub total_chunks: usize,
    /// Number of chunks at each LOD level.
    pub chunks_per_lod: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Terrain streaming
// ---------------------------------------------------------------------------

/// Manages streaming of terrain chunks in and out of memory.
///
/// As the camera moves, chunks beyond a certain distance are unloaded and
/// new chunks entering the view range are loaded.
pub struct TerrainStreamer {
    /// Chunks currently loaded in memory.
    loaded_chunks: std::collections::HashSet<(u32, u32)>,
    /// Maximum distance (in chunks) from the camera to keep loaded.
    load_radius: u32,
    /// Distance (in chunks) at which to unload.
    unload_radius: u32,
    /// The chunk size in world units.
    chunk_size: f32,
    /// Queue of chunks to load.
    load_queue: Vec<(u32, u32)>,
    /// Queue of chunks to unload.
    unload_queue: Vec<(u32, u32)>,
}

impl TerrainStreamer {
    /// Creates a new terrain streamer.
    pub fn new(chunk_size: f32, load_radius: u32, unload_radius: u32) -> Self {
        Self {
            loaded_chunks: std::collections::HashSet::new(),
            load_radius,
            unload_radius: unload_radius.max(load_radius + 1),
            chunk_size,
            load_queue: Vec::new(),
            unload_queue: Vec::new(),
        }
    }

    /// Updates the streamer based on the current camera position.
    ///
    /// Populates the load and unload queues. The caller should process
    /// these queues each frame (loading/unloading a limited number of
    /// chunks to avoid frame spikes).
    pub fn update(&mut self, camera_pos: Vec3, max_chunk_x: u32, max_chunk_z: u32) {
        let cam_cx = (camera_pos.x / self.chunk_size).floor() as i32;
        let cam_cz = (camera_pos.z / self.chunk_size).floor() as i32;

        self.load_queue.clear();
        self.unload_queue.clear();

        // Find chunks that need loading
        let lr = self.load_radius as i32;
        for dz in -lr..=lr {
            for dx in -lr..=lr {
                let cx = cam_cx + dx;
                let cz = cam_cz + dz;

                if cx < 0 || cz < 0 || cx >= max_chunk_x as i32 || cz >= max_chunk_z as i32
                {
                    continue;
                }

                let dist_sq = dx * dx + dz * dz;
                if dist_sq <= lr * lr {
                    let key = (cx as u32, cz as u32);
                    if !self.loaded_chunks.contains(&key) {
                        self.load_queue.push(key);
                    }
                }
            }
        }

        // Sort load queue by distance (closest first)
        self.load_queue.sort_by(|a, b| {
            let da = (a.0 as i32 - cam_cx).pow(2) + (a.1 as i32 - cam_cz).pow(2);
            let db = (b.0 as i32 - cam_cx).pow(2) + (b.1 as i32 - cam_cz).pow(2);
            da.cmp(&db)
        });

        // Find chunks that need unloading
        let ur = self.unload_radius as i32;
        let ur_sq = ur * ur;
        let to_unload: Vec<_> = self
            .loaded_chunks
            .iter()
            .filter(|&&(cx, cz)| {
                let dx = cx as i32 - cam_cx;
                let dz = cz as i32 - cam_cz;
                dx * dx + dz * dz > ur_sq
            })
            .copied()
            .collect();

        self.unload_queue = to_unload;
    }

    /// Marks a chunk as loaded.
    pub fn mark_loaded(&mut self, chunk_x: u32, chunk_z: u32) {
        self.loaded_chunks.insert((chunk_x, chunk_z));
    }

    /// Marks a chunk as unloaded.
    pub fn mark_unloaded(&mut self, chunk_x: u32, chunk_z: u32) {
        self.loaded_chunks.remove(&(chunk_x, chunk_z));
    }

    /// Returns the queue of chunks waiting to be loaded.
    pub fn load_queue(&self) -> &[(u32, u32)] {
        &self.load_queue
    }

    /// Returns the queue of chunks waiting to be unloaded.
    pub fn unload_queue(&self) -> &[(u32, u32)] {
        &self.unload_queue
    }

    /// Returns whether a chunk is currently loaded.
    pub fn is_loaded(&self, chunk_x: u32, chunk_z: u32) -> bool {
        self.loaded_chunks.contains(&(chunk_x, chunk_z))
    }

    /// Returns the number of currently loaded chunks.
    pub fn loaded_count(&self) -> usize {
        self.loaded_chunks.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lod_settings_distance() {
        let settings = LODSettings::default();
        assert_eq!(settings.lod_for_distance(50.0), 0);
        assert_eq!(settings.lod_for_distance(150.0), 1);
        assert_eq!(settings.lod_for_distance(300.0), 2);
    }

    #[test]
    fn lod_morph_factor() {
        let settings = LODSettings::default();
        let morph = settings.morph_factor(90.0, 0);
        assert!(morph > 0.0 && morph < 1.0);

        assert_eq!(settings.morph_factor(50.0, 0), 0.0);
        assert_eq!(settings.morph_factor(100.0, 0), 1.0);
    }

    #[test]
    fn quadtree_build() {
        let hm = crate::Heightmap::new_flat(65, 65, 0.5).unwrap();
        let settings = LODSettings::default();
        let qt = TerrainQuadtree::build(&hm, 1024.0, 4, settings);
        assert!(qt.node_count() > 0);
        assert_eq!(qt.max_depth(), 4);
    }

    #[test]
    fn quadtree_selection() {
        let hm = crate::Heightmap::generate_procedural(65, 0.5, 42).unwrap();
        let settings = LODSettings::default();
        let qt = TerrainQuadtree::build(&hm, 1024.0, 4, settings);

        let camera = Vec3::new(512.0, 50.0, 512.0);
        let vp = glam::Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4,
            1.6,
            0.1,
            10000.0,
        ) * glam::Mat4::look_at_rh(camera, Vec3::new(512.0, 0.0, 512.0), Vec3::Y);
        let frustum = Frustum::from_view_projection(&vp);

        let chunks = qt.select_visible_nodes(camera, &frustum);
        assert!(!chunks.is_empty());

        // All chunks should have non-negative morph factors
        for chunk in &chunks {
            assert!(chunk.morph_factor >= 0.0);
            assert!(chunk.morph_factor <= 1.0);
        }
    }

    #[test]
    fn geomipmap_creation() {
        let gmm = GeoMipMap::new(4, 4, 17, 4);
        assert_eq!(gmm.total_chunks(), 16);
        assert_eq!(gmm.num_lod_levels(), 4);
    }

    #[test]
    fn geomipmap_update() {
        let mut gmm = GeoMipMap::new(4, 4, 17, 4);
        let settings = LODSettings::default();
        let camera = Vec3::new(32.0, 10.0, 32.0);
        gmm.update(camera, 64.0, &settings);

        // Chunk at camera position should have low LOD (fine detail)
        let lod = gmm.chunk_lod(0, 0);
        assert!(lod <= 1);
    }

    #[test]
    fn terrain_streamer() {
        let mut streamer = TerrainStreamer::new(64.0, 3, 5);
        let camera = Vec3::new(128.0, 10.0, 128.0);
        streamer.update(camera, 16, 16);
        assert!(!streamer.load_queue().is_empty());
    }

    #[test]
    fn screen_space_error_calculation() {
        let sse = TerrainQuadtree::screen_space_error(
            1.0,
            100.0,
            std::f32::consts::FRAC_PI_4,
            1080.0,
        );
        assert!(sse > 0.0);
        assert!(sse < 100.0);

        // Closer = larger SSE
        let sse_close = TerrainQuadtree::screen_space_error(
            1.0,
            10.0,
            std::f32::consts::FRAC_PI_4,
            1080.0,
        );
        assert!(sse_close > sse);
    }
}
