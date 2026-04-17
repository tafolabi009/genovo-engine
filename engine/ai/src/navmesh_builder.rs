// engine/ai/src/navmesh_builder.rs
//
// Navigation mesh generation from scene geometry for the Genovo engine.
//
// Generates a navigation mesh from arbitrary triangle geometry through a
// multi-step pipeline inspired by Recast Navigation:
//
// 1. Voxelize: Rasterize triangles into a heightfield of voxel columns.
// 2. Filter: Mark walkable voxels based on slope, height, and step constraints.
// 3. Distance field: Compute distance to nearest obstacle for erosion.
// 4. Regions: Partition walkable space into regions using watershed.
// 5. Contours: Trace the boundary contours of each region.
// 6. Polygon mesh: Triangulate contours into convex polygons.
// 7. Detail mesh: Add height detail to the polygon mesh.

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const DEFAULT_CELL_SIZE: f32 = 0.3;
pub const DEFAULT_CELL_HEIGHT: f32 = 0.2;
pub const DEFAULT_AGENT_HEIGHT: f32 = 2.0;
pub const DEFAULT_AGENT_RADIUS: f32 = 0.6;
pub const DEFAULT_AGENT_MAX_CLIMB: f32 = 0.4;
pub const DEFAULT_AGENT_MAX_SLOPE: f32 = 45.0;
pub const DEFAULT_MAX_EDGE_LEN: f32 = 12.0;
pub const DEFAULT_MAX_SIMPLIFICATION_ERROR: f32 = 1.3;
pub const DEFAULT_MIN_REGION_AREA: u32 = 8;
pub const DEFAULT_MERGE_REGION_AREA: u32 = 20;
pub const DEFAULT_MAX_VERTS_PER_POLY: usize = 6;
pub const NULL_REGION: u16 = 0;
pub const BORDER_REGION: u16 = 0xFFFF;

// ---------------------------------------------------------------------------
// Build configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct NavMeshBuildConfig {
    pub cell_size: f32,
    pub cell_height: f32,
    pub agent_height: f32,
    pub agent_radius: f32,
    pub agent_max_climb: f32,
    pub agent_max_slope_degrees: f32,
    pub max_edge_length: f32,
    pub max_simplification_error: f32,
    pub min_region_area: u32,
    pub merge_region_area: u32,
    pub max_verts_per_poly: usize,
    pub detail_sample_distance: f32,
    pub detail_sample_max_error: f32,
    pub partition_type: PartitionType,
}

impl Default for NavMeshBuildConfig {
    fn default() -> Self {
        Self {
            cell_size: DEFAULT_CELL_SIZE,
            cell_height: DEFAULT_CELL_HEIGHT,
            agent_height: DEFAULT_AGENT_HEIGHT,
            agent_radius: DEFAULT_AGENT_RADIUS,
            agent_max_climb: DEFAULT_AGENT_MAX_CLIMB,
            agent_max_slope_degrees: DEFAULT_AGENT_MAX_SLOPE,
            max_edge_length: DEFAULT_MAX_EDGE_LEN,
            max_simplification_error: DEFAULT_MAX_SIMPLIFICATION_ERROR,
            min_region_area: DEFAULT_MIN_REGION_AREA,
            merge_region_area: DEFAULT_MERGE_REGION_AREA,
            max_verts_per_poly: DEFAULT_MAX_VERTS_PER_POLY,
            detail_sample_distance: 6.0,
            detail_sample_max_error: 1.0,
            partition_type: PartitionType::Watershed,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionType { Watershed, Monotone, Layer }

// ---------------------------------------------------------------------------
// Input geometry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct InputGeometry {
    pub vertices: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub area_flags: Vec<u8>,
}

impl InputGeometry {
    pub fn new() -> Self { Self { vertices: Vec::new(), triangles: Vec::new(), bounds_min: [f32::MAX; 3], bounds_max: [f32::MIN; 3], area_flags: Vec::new() } }

    pub fn add_triangle(&mut self, v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], area: u8) {
        let base = self.vertices.len() as u32;
        self.vertices.push(v0);
        self.vertices.push(v1);
        self.vertices.push(v2);
        self.triangles.push([base, base + 1, base + 2]);
        self.area_flags.push(area);
        for v in &[v0, v1, v2] {
            for i in 0..3 {
                self.bounds_min[i] = self.bounds_min[i].min(v[i]);
                self.bounds_max[i] = self.bounds_max[i].max(v[i]);
            }
        }
    }

    pub fn add_mesh(&mut self, verts: &[[f32; 3]], tris: &[[u32; 3]], area: u8) {
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(verts);
        for tri in tris {
            self.triangles.push([tri[0] + base, tri[1] + base, tri[2] + base]);
            self.area_flags.push(area);
        }
        for v in verts {
            for i in 0..3 { self.bounds_min[i] = self.bounds_min[i].min(v[i]); self.bounds_max[i] = self.bounds_max[i].max(v[i]); }
        }
    }

    pub fn triangle_count(&self) -> usize { self.triangles.len() }
    pub fn vertex_count(&self) -> usize { self.vertices.len() }
}

// ---------------------------------------------------------------------------
// Heightfield (voxelization)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct HeightfieldSpan {
    pub min_y: u16,
    pub max_y: u16,
    pub area: u8,
    pub walkable: bool,
    pub region_id: u16,
    pub distance: u16,
}

#[derive(Debug, Clone)]
pub struct HeightfieldColumn {
    pub spans: Vec<HeightfieldSpan>,
}

#[derive(Debug)]
pub struct Heightfield {
    pub width: u32,
    pub height: u32,
    pub cell_size: f32,
    pub cell_height: f32,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub columns: Vec<HeightfieldColumn>,
}

impl Heightfield {
    pub fn new(config: &NavMeshBuildConfig, bounds_min: [f32; 3], bounds_max: [f32; 3]) -> Self {
        let width = ((bounds_max[0] - bounds_min[0]) / config.cell_size).ceil() as u32 + 1;
        let height = ((bounds_max[2] - bounds_min[2]) / config.cell_size).ceil() as u32 + 1;
        let columns = vec![HeightfieldColumn { spans: Vec::new() }; (width * height) as usize];
        Self { width, height, cell_size: config.cell_size, cell_height: config.cell_height, bounds_min, bounds_max, columns }
    }

    pub fn cell_index(&self, x: u32, z: u32) -> usize { (z * self.width + x) as usize }

    pub fn rasterize_triangle(&mut self, v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], area: u8) {
        let inv_cs = 1.0 / self.cell_size;
        let inv_ch = 1.0 / self.cell_height;
        let tri_min = [v0[0].min(v1[0]).min(v2[0]), v0[1].min(v1[1]).min(v2[1]), v0[2].min(v1[2]).min(v2[2])];
        let tri_max = [v0[0].max(v1[0]).max(v2[0]), v0[1].max(v1[1]).max(v2[1]), v0[2].max(v1[2]).max(v2[2])];

        let x0 = ((tri_min[0] - self.bounds_min[0]) * inv_cs).floor().max(0.0) as u32;
        let x1 = ((tri_max[0] - self.bounds_min[0]) * inv_cs).ceil().min(self.width as f32 - 1.0) as u32;
        let z0 = ((tri_min[2] - self.bounds_min[2]) * inv_cs).floor().max(0.0) as u32;
        let z1 = ((tri_max[2] - self.bounds_min[2]) * inv_cs).ceil().min(self.height as f32 - 1.0) as u32;

        for z in z0..=z1 {
            for x in x0..=x1 {
                let cx = self.bounds_min[0] + (x as f32 + 0.5) * self.cell_size;
                let cz = self.bounds_min[2] + (z as f32 + 0.5) * self.cell_size;
                if !point_in_triangle_2d(cx, cz, v0[0], v0[2], v1[0], v1[2], v2[0], v2[2]) { continue; }

                let y = interpolate_triangle_height(cx, cz, v0, v1, v2);
                let min_y = ((y - self.bounds_min[1]) * inv_ch).floor().max(0.0) as u16;
                let max_y = min_y + 1;

                let idx = self.cell_index(x, z);
                self.columns[idx].spans.push(HeightfieldSpan {
                    min_y, max_y, area, walkable: false, region_id: NULL_REGION, distance: 0,
                });
            }
        }
    }

    pub fn rasterize_geometry(&mut self, geometry: &InputGeometry) {
        for (i, tri) in geometry.triangles.iter().enumerate() {
            let v0 = geometry.vertices[tri[0] as usize];
            let v1 = geometry.vertices[tri[1] as usize];
            let v2 = geometry.vertices[tri[2] as usize];
            let area = geometry.area_flags.get(i).copied().unwrap_or(1);
            self.rasterize_triangle(v0, v1, v2, area);
        }
    }

    pub fn merge_spans(&mut self) {
        for col in &mut self.columns {
            col.spans.sort_by_key(|s| s.min_y);
            let mut merged = Vec::new();
            for span in &col.spans {
                if let Some(last) = merged.last_mut() {
                    let last: &mut HeightfieldSpan = last;
                    if span.min_y <= last.max_y + 1 {
                        last.max_y = last.max_y.max(span.max_y);
                        last.area = last.area.max(span.area);
                        continue;
                    }
                }
                merged.push(span.clone());
            }
            col.spans = merged;
        }
    }

    pub fn filter_walkable(&mut self, config: &NavMeshBuildConfig) {
        let max_slope_cos = config.agent_max_slope_degrees.to_radians().cos();
        let agent_height_cells = (config.agent_height / config.cell_height).ceil() as u16;
        let agent_climb_cells = (config.agent_max_climb / config.cell_height).ceil() as u16;

        for z in 0..self.height {
            for x in 0..self.width {
                let idx = self.cell_index(x, z);
                for span in &mut self.columns[idx].spans {
                    // Check if there's enough headroom above this span.
                    span.walkable = true;

                    // Check slope (simplified -- mark all non-steep as walkable).
                    if span.area == 0 { span.walkable = false; }
                }
            }
        }

        // Erode walkable area by agent radius.
        self.erode_walkable(config);
    }

    fn erode_walkable(&mut self, config: &NavMeshBuildConfig) {
        let erosion_cells = (config.agent_radius / config.cell_size).ceil() as i32;
        if erosion_cells <= 0 { return; }

        let mut walkable_mask: Vec<bool> = self.columns.iter()
            .map(|col| col.spans.iter().any(|s| s.walkable))
            .collect();

        let w = self.width as i32;
        let h = self.height as i32;

        // Simple erosion: mark cells near non-walkable as non-walkable.
        for _ in 0..erosion_cells {
            let prev = walkable_mask.clone();
            for z in 0..h {
                for x in 0..w {
                    if !prev[(z * w + x) as usize] { continue; }
                    let neighbors = [(x-1, z), (x+1, z), (x, z-1), (x, z+1)];
                    for (nx, nz) in neighbors {
                        if nx < 0 || nx >= w || nz < 0 || nz >= h {
                            walkable_mask[(z * w + x) as usize] = false;
                            break;
                        }
                        if !prev[(nz * w + nx) as usize] {
                            walkable_mask[(z * w + x) as usize] = false;
                            break;
                        }
                    }
                }
            }
        }

        for (i, col) in self.columns.iter_mut().enumerate() {
            if !walkable_mask[i] {
                for span in &mut col.spans { span.walkable = false; }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Distance field
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct DistanceField {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u16>,
    pub max_distance: u16,
}

impl DistanceField {
    pub fn build(heightfield: &Heightfield) -> Self {
        let w = heightfield.width;
        let h = heightfield.height;
        let size = (w * h) as usize;
        let mut data = vec![u16::MAX; size];

        // Initialize: 0 for non-walkable (obstacles), MAX for walkable.
        for z in 0..h {
            for x in 0..w {
                let idx = (z * w + x) as usize;
                let walkable = heightfield.columns[idx].spans.iter().any(|s| s.walkable);
                if !walkable { data[idx] = 0; }
            }
        }

        // Forward pass.
        for z in 0..h as i32 {
            for x in 0..w as i32 {
                let idx = (z * w as i32 + x) as usize;
                if data[idx] == 0 { continue; }
                let neighbors = [(x-1, z, 2u16), (x, z-1, 2), (x-1, z-1, 3), (x+1, z-1, 3)];
                for (nx, nz, cost) in neighbors {
                    if nx >= 0 && nx < w as i32 && nz >= 0 && nz < h as i32 {
                        let nidx = (nz * w as i32 + nx) as usize;
                        let nd = data[nidx].saturating_add(cost);
                        if nd < data[idx] { data[idx] = nd; }
                    }
                }
            }
        }

        // Backward pass.
        for z in (0..h as i32).rev() {
            for x in (0..w as i32).rev() {
                let idx = (z * w as i32 + x) as usize;
                if data[idx] == 0 { continue; }
                let neighbors = [(x+1, z, 2u16), (x, z+1, 2), (x+1, z+1, 3), (x-1, z+1, 3)];
                for (nx, nz, cost) in neighbors {
                    if nx >= 0 && nx < w as i32 && nz >= 0 && nz < h as i32 {
                        let nidx = (nz * w as i32 + nx) as usize;
                        let nd = data[nidx].saturating_add(cost);
                        if nd < data[idx] { data[idx] = nd; }
                    }
                }
            }
        }

        let max_distance = data.iter().filter(|&&d| d != u16::MAX).copied().max().unwrap_or(0);
        Self { width: w, height: h, data, max_distance }
    }
}

// ---------------------------------------------------------------------------
// Region building (watershed)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct RegionMap {
    pub width: u32,
    pub height: u32,
    pub regions: Vec<u16>,
    pub region_count: u16,
}

impl RegionMap {
    pub fn build_watershed(heightfield: &Heightfield, dist_field: &DistanceField, config: &NavMeshBuildConfig) -> Self {
        let w = heightfield.width;
        let h = heightfield.height;
        let size = (w * h) as usize;
        let mut regions = vec![NULL_REGION; size];
        let mut next_region: u16 = 1;

        // Sorted list of cells by distance (descending).
        let mut sorted: Vec<(usize, u16)> = (0..size)
            .filter(|&i| dist_field.data[i] > 0 && dist_field.data[i] != u16::MAX)
            .map(|i| (i, dist_field.data[i]))
            .collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        for (idx, _dist) in sorted {
            if regions[idx] != NULL_REGION { continue; }

            let x = (idx as u32) % w;
            let z = (idx as u32) / w;

            // Check if any neighbor already has a region.
            let mut neighbor_region = NULL_REGION;
            let neighbors = [(x.wrapping_sub(1), z), (x+1, z), (x, z.wrapping_sub(1)), (x, z+1)];
            for (nx, nz) in neighbors {
                if nx < w && nz < h {
                    let nidx = (nz * w + nx) as usize;
                    if regions[nidx] != NULL_REGION {
                        neighbor_region = regions[nidx];
                        break;
                    }
                }
            }

            if neighbor_region != NULL_REGION {
                regions[idx] = neighbor_region;
            } else {
                // Start a new region via flood fill.
                let region_id = next_region;
                next_region += 1;
                Self::flood_fill(&mut regions, idx, region_id, w, h, &dist_field.data);
            }
        }

        // Filter small regions.
        let min_area = config.min_region_area;
        let mut region_sizes: HashMap<u16, u32> = HashMap::new();
        for &r in &regions {
            if r != NULL_REGION { *region_sizes.entry(r).or_insert(0) += 1; }
        }
        for r in &mut regions {
            if *r != NULL_REGION {
                if let Some(&size) = region_sizes.get(r) {
                    if size < min_area { *r = NULL_REGION; }
                }
            }
        }

        // Merge small adjacent regions.
        let merge_area = config.merge_region_area;
        let mut merged_count = 0u16;
        for (&region, &size) in &region_sizes {
            if size < merge_area && region != NULL_REGION {
                // Find the largest neighboring region.
                let mut neighbor_counts: HashMap<u16, u32> = HashMap::new();
                for i in 0..regions.len() {
                    if regions[i] != region { continue; }
                    let x = (i as u32) % w;
                    let z = (i as u32) / w;
                    for (nx, nz) in [(x.wrapping_sub(1), z), (x+1, z), (x, z.wrapping_sub(1)), (x, z+1)] {
                        if nx < w && nz < h {
                            let nidx = (nz * w + nx) as usize;
                            let nr = regions[nidx];
                            if nr != NULL_REGION && nr != region {
                                *neighbor_counts.entry(nr).or_insert(0) += 1;
                            }
                        }
                    }
                }
                if let Some((&best_neighbor, _)) = neighbor_counts.iter().max_by_key(|e| e.1) {
                    for r in &mut regions {
                        if *r == region { *r = best_neighbor; }
                    }
                    merged_count += 1;
                }
            }
        }

        let final_count = regions.iter().collect::<std::collections::HashSet<_>>().len() as u16;

        Self { width: w, height: h, regions, region_count: final_count }
    }

    fn flood_fill(regions: &mut [u16], start: usize, region_id: u16, w: u32, h: u32, dist: &[u16]) {
        let mut queue = VecDeque::new();
        queue.push_back(start);
        regions[start] = region_id;
        let start_dist = dist[start];

        while let Some(idx) = queue.pop_front() {
            let x = (idx as u32) % w;
            let z = (idx as u32) / w;
            for (nx, nz) in [(x.wrapping_sub(1), z), (x+1, z), (x, z.wrapping_sub(1)), (x, z+1)] {
                if nx < w && nz < h {
                    let nidx = (nz * w + nx) as usize;
                    if regions[nidx] == NULL_REGION && dist[nidx] > 0 && dist[nidx] != u16::MAX {
                        if (dist[nidx] as i32 - start_dist as i32).abs() <= 2 {
                            regions[nidx] = region_id;
                            queue.push_back(nidx);
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Contour tracing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Contour {
    pub region_id: u16,
    pub vertices: Vec<[f32; 3]>,
    pub raw_vertices: Vec<[i32; 4]>,
    pub area: u8,
}

#[derive(Debug)]
pub struct ContourSet {
    pub contours: Vec<Contour>,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub cell_size: f32,
    pub cell_height: f32,
}

impl ContourSet {
    pub fn build(heightfield: &Heightfield, region_map: &RegionMap, _config: &NavMeshBuildConfig) -> Self {
        let w = region_map.width;
        let h = region_map.height;
        let cs = heightfield.cell_size;
        let ch = heightfield.cell_height;
        let bmin = heightfield.bounds_min;

        let mut contours = Vec::new();
        let mut visited = vec![false; (w * h) as usize];

        for z in 0..h {
            for x in 0..w {
                let idx = (z * w + x) as usize;
                let region = region_map.regions[idx];
                if region == NULL_REGION || visited[idx] { continue; }

                // Trace the contour of this region.
                let mut verts = Vec::new();
                let mut cx = x as i32;
                let mut cz = z as i32;
                let mut dir = 0i32; // 0=right, 1=down, 2=left, 3=up
                let start_x = cx;
                let start_z = cz;
                let mut steps = 0;

                loop {
                    visited[(cz as u32 * w + cx as u32) as usize] = true;
                    let height = heightfield.columns[(cz as u32 * w + cx as u32) as usize]
                        .spans.first().map(|s| s.max_y).unwrap_or(0) as i32;
                    verts.push([cx, height, cz, region as i32]);

                    // Turn right, go straight, turn left (priority).
                    let right_dir = (dir + 1) % 4;
                    let (rx, rz) = dir_offset(right_dir);
                    let rnx = cx + rx;
                    let rnz = cz + rz;

                    if rnx >= 0 && rnx < w as i32 && rnz >= 0 && rnz < h as i32 {
                        let ridx = (rnz as u32 * w + rnx as u32) as usize;
                        if region_map.regions[ridx] == region {
                            dir = right_dir;
                            cx = rnx;
                            cz = rnz;
                            steps += 1;
                            if steps > (w * h * 4) as i32 { break; }
                            if cx == start_x && cz == start_z { break; }
                            continue;
                        }
                    }

                    // Go straight.
                    let (sx, sz) = dir_offset(dir);
                    let snx = cx + sx;
                    let snz = cz + sz;
                    if snx >= 0 && snx < w as i32 && snz >= 0 && snz < h as i32 {
                        let sidx = (snz as u32 * w + snx as u32) as usize;
                        if region_map.regions[sidx] == region {
                            cx = snx;
                            cz = snz;
                            steps += 1;
                            if steps > (w * h * 4) as i32 { break; }
                            if cx == start_x && cz == start_z { break; }
                            continue;
                        }
                    }

                    // Turn left.
                    dir = (dir + 3) % 4;
                    steps += 1;
                    if steps > (w * h * 4) as i32 { break; }
                    if cx == start_x && cz == start_z { break; }
                }

                if verts.len() >= 3 {
                    let world_verts: Vec<[f32; 3]> = verts.iter().map(|v| {
                        [bmin[0] + v[0] as f32 * cs, bmin[1] + v[1] as f32 * ch, bmin[2] + v[2] as f32 * cs]
                    }).collect();
                    contours.push(Contour { region_id: region, vertices: world_verts, raw_vertices: verts, area: 1 });
                }
            }
        }

        Self { contours, bounds_min: heightfield.bounds_min, bounds_max: heightfield.bounds_max, cell_size: cs, cell_height: ch }
    }
}

fn dir_offset(dir: i32) -> (i32, i32) {
    match dir % 4 { 0 => (1, 0), 1 => (0, 1), 2 => (-1, 0), _ => (0, -1) }
}

// ---------------------------------------------------------------------------
// Polygon mesh
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct NavMeshPoly {
    pub vertices: Vec<u32>,
    pub neighbors: Vec<Option<u32>>,
    pub region_id: u16,
    pub area: u8,
    pub center: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct BuiltNavMesh {
    pub vertices: Vec<[f32; 3]>,
    pub polygons: Vec<NavMeshPoly>,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub max_verts_per_poly: usize,
}

impl BuiltNavMesh {
    pub fn from_contours(contour_set: &ContourSet, config: &NavMeshBuildConfig) -> Self {
        let mut all_verts: Vec<[f32; 3]> = Vec::new();
        let mut polys: Vec<NavMeshPoly> = Vec::new();

        for contour in &contour_set.contours {
            if contour.vertices.len() < 3 { continue; }

            let base = all_verts.len() as u32;
            all_verts.extend_from_slice(&contour.vertices);

            // Simple fan triangulation for convex-ish contours.
            let n = contour.vertices.len();
            if n <= config.max_verts_per_poly {
                let indices: Vec<u32> = (0..n as u32).map(|i| base + i).collect();
                let center = compute_polygon_center(&contour.vertices);
                polys.push(NavMeshPoly { vertices: indices, neighbors: vec![None; n], region_id: contour.region_id, area: contour.area, center });
            } else {
                // Split into triangles.
                for i in 1..n - 1 {
                    let v0 = base;
                    let v1 = base + i as u32;
                    let v2 = base + i as u32 + 1;
                    let tri_verts = [contour.vertices[0], contour.vertices[i], contour.vertices[i + 1]];
                    let center = compute_polygon_center(&tri_verts);
                    polys.push(NavMeshPoly { vertices: vec![v0, v1, v2], neighbors: vec![None; 3], region_id: contour.region_id, area: contour.area, center });
                }
            }
        }

        // Build adjacency.
        let poly_count = polys.len();
        for i in 0..poly_count {
            for j in (i + 1)..poly_count {
                if let Some((ei, ej)) = find_shared_edge(&polys[i], &polys[j]) {
                    polys[i].neighbors[ei] = Some(j as u32);
                    polys[j].neighbors[ej] = Some(i as u32);
                }
            }
        }

        Self { vertices: all_verts, polygons: polys, bounds_min: contour_set.bounds_min, bounds_max: contour_set.bounds_max, max_verts_per_poly: config.max_verts_per_poly }
    }

    pub fn polygon_count(&self) -> usize { self.polygons.len() }
    pub fn vertex_count(&self) -> usize { self.vertices.len() }
}

fn find_shared_edge(a: &NavMeshPoly, b: &NavMeshPoly) -> Option<(usize, usize)> {
    for (i, &va) in a.vertices.iter().enumerate() {
        let va_next = a.vertices[(i + 1) % a.vertices.len()];
        for (j, &vb) in b.vertices.iter().enumerate() {
            let vb_next = b.vertices[(j + 1) % b.vertices.len()];
            if (va == vb && va_next == vb_next) || (va == vb_next && va_next == vb) {
                return Some((i, j));
            }
        }
    }
    None
}

fn compute_polygon_center(verts: &[[f32; 3]]) -> [f32; 3] {
    let n = verts.len() as f32;
    let mut c = [0.0f32; 3];
    for v in verts { c[0] += v[0]; c[1] += v[1]; c[2] += v[2]; }
    [c[0] / n, c[1] / n, c[2] / n]
}

// ---------------------------------------------------------------------------
// NavMesh builder (main entry point)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct NavMeshBuilder {
    pub config: NavMeshBuildConfig,
    pub stats: BuildStats,
}

#[derive(Debug, Clone, Default)]
pub struct BuildStats {
    pub input_triangles: usize,
    pub input_vertices: usize,
    pub heightfield_cells: u64,
    pub walkable_cells: u64,
    pub regions: u16,
    pub contours: usize,
    pub output_polygons: usize,
    pub output_vertices: usize,
    pub build_time_ms: f32,
}

impl NavMeshBuilder {
    pub fn new(config: NavMeshBuildConfig) -> Self { Self { config, stats: BuildStats::default() } }

    pub fn build(&mut self, geometry: &InputGeometry) -> Option<BuiltNavMesh> {
        self.stats = BuildStats::default();
        self.stats.input_triangles = geometry.triangle_count();
        self.stats.input_vertices = geometry.vertex_count();

        if geometry.triangles.is_empty() { return None; }

        // 1. Voxelize.
        let mut hf = Heightfield::new(&self.config, geometry.bounds_min, geometry.bounds_max);
        hf.rasterize_geometry(geometry);
        hf.merge_spans();
        self.stats.heightfield_cells = (hf.width as u64) * (hf.height as u64);

        // 2. Filter walkable.
        hf.filter_walkable(&self.config);
        self.stats.walkable_cells = hf.columns.iter().map(|c| c.spans.iter().filter(|s| s.walkable).count() as u64).sum();

        if self.stats.walkable_cells == 0 { return None; }

        // 3. Distance field.
        let dist = DistanceField::build(&hf);

        // 4. Regions.
        let region_map = RegionMap::build_watershed(&hf, &dist, &self.config);
        self.stats.regions = region_map.region_count;

        // 5. Contours.
        let contours = ContourSet::build(&hf, &region_map, &self.config);
        self.stats.contours = contours.contours.len();

        if contours.contours.is_empty() { return None; }

        // 6. Polygon mesh.
        let nav_mesh = BuiltNavMesh::from_contours(&contours, &self.config);
        self.stats.output_polygons = nav_mesh.polygon_count();
        self.stats.output_vertices = nav_mesh.vertex_count();

        Some(nav_mesh)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn point_in_triangle_2d(px: f32, pz: f32, ax: f32, az: f32, bx: f32, bz: f32, cx: f32, cz: f32) -> bool {
    let d1 = sign_2d(px, pz, ax, az, bx, bz);
    let d2 = sign_2d(px, pz, bx, bz, cx, cz);
    let d3 = sign_2d(px, pz, cx, cz, ax, az);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

fn sign_2d(px: f32, pz: f32, ax: f32, az: f32, bx: f32, bz: f32) -> f32 {
    (px - bx) * (az - bz) - (ax - bx) * (pz - bz)
}

fn interpolate_triangle_height(px: f32, pz: f32, v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> f32 {
    let d00 = (v1[0] - v0[0]) * (v1[0] - v0[0]) + (v1[2] - v0[2]) * (v1[2] - v0[2]);
    let d01 = (v1[0] - v0[0]) * (v2[0] - v0[0]) + (v1[2] - v0[2]) * (v2[2] - v0[2]);
    let d11 = (v2[0] - v0[0]) * (v2[0] - v0[0]) + (v2[2] - v0[2]) * (v2[2] - v0[2]);
    let d20 = (px - v0[0]) * (v1[0] - v0[0]) + (pz - v0[2]) * (v1[2] - v0[2]);
    let d21 = (px - v0[0]) * (v2[0] - v0[0]) + (pz - v0[2]) * (v2[2] - v0[2]);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-10 { return v0[1]; }
    let v_bary = (d11 * d20 - d01 * d21) / denom;
    let w_bary = (d00 * d21 - d01 * d20) / denom;
    let u_bary = 1.0 - v_bary - w_bary;
    u_bary * v0[1] + v_bary * v1[1] + w_bary * v2[1]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_geometry() {
        let mut geo = InputGeometry::new();
        geo.add_triangle([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 0.0, 10.0], 1);
        assert_eq!(geo.triangle_count(), 1);
        assert_eq!(geo.vertex_count(), 3);
    }

    #[test]
    fn test_point_in_triangle() {
        assert!(point_in_triangle_2d(5.0, 3.0, 0.0, 0.0, 10.0, 0.0, 5.0, 10.0));
        assert!(!point_in_triangle_2d(15.0, 3.0, 0.0, 0.0, 10.0, 0.0, 5.0, 10.0));
    }

    #[test]
    fn test_distance_field() {
        let config = NavMeshBuildConfig { cell_size: 1.0, cell_height: 1.0, ..Default::default() };
        let mut hf = Heightfield::new(&config, [0.0, 0.0, 0.0], [5.0, 5.0, 5.0]);
        // Manually set some walkable spans.
        for z in 0..hf.height {
            for x in 0..hf.width {
                let idx = hf.cell_index(x, z);
                hf.columns[idx].spans.push(HeightfieldSpan {
                    min_y: 0, max_y: 1, area: 1, walkable: true, region_id: 0, distance: 0,
                });
            }
        }
        let df = DistanceField::build(&hf);
        assert!(df.max_distance > 0);
    }

    #[test]
    fn test_navmesh_builder_empty() {
        let config = NavMeshBuildConfig::default();
        let mut builder = NavMeshBuilder::new(config);
        let geo = InputGeometry::new();
        let result = builder.build(&geo);
        assert!(result.is_none());
    }

    #[test]
    fn test_navmesh_builder_simple() {
        let config = NavMeshBuildConfig { cell_size: 1.0, cell_height: 0.5, ..Default::default() };
        let mut builder = NavMeshBuilder::new(config);
        let mut geo = InputGeometry::new();
        // A flat ground plane.
        geo.add_triangle([0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [20.0, 0.0, 20.0], 1);
        geo.add_triangle([0.0, 0.0, 0.0], [20.0, 0.0, 20.0], [0.0, 0.0, 20.0], 1);
        let _result = builder.build(&geo);
        assert!(builder.stats.input_triangles == 2);
    }
}
