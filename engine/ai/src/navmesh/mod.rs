//! Navigation mesh system.
//!
//! Provides navmesh construction from triangle soup, spatial queries
//! (point-on-mesh, nearest polygon, path corridors), the Simple Stupid Funnel
//! Algorithm for path smoothing, agent navigation components, crowd simulation,
//! and obstacle avoidance using ORCA (Optimal Reciprocal Collision Avoidance).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use glam::Vec3;
use smallvec::SmallVec;

use genovo_core::EngineResult;
use genovo_ecs::Component;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Index into the navmesh vertex array.
pub type VertexIndex = u32;

/// Index into the navmesh polygon array.
pub type PolyIndex = u32;

// ---------------------------------------------------------------------------
// NavPolygon
// ---------------------------------------------------------------------------

/// A single convex polygon in the navigation mesh.
#[derive(Debug, Clone)]
pub struct NavPoly {
    /// Indices into the navmesh vertex array (counter-clockwise winding).
    pub vertices: SmallVec<[VertexIndex; 6]>,
    /// Indices of adjacent polygons per edge. `u32::MAX` means no neighbor.
    /// Edge i connects vertex[i] to vertex[(i+1) % len].
    pub adjacency: SmallVec<[PolyIndex; 6]>,
    /// Area type / traversal cost modifier.
    pub area_type: u16,
    /// Precomputed polygon centroid (for heuristics).
    pub centroid: Vec3,
    /// Precomputed polygon normal (for slope checks).
    pub normal: Vec3,
}

impl NavPoly {
    /// Number of edges (and vertices) in this polygon.
    pub fn edge_count(&self) -> usize {
        self.vertices.len()
    }

    /// Returns the shared edge between this polygon and `other_poly` if they
    /// are adjacent. Returns the pair of vertex indices forming the shared edge.
    pub fn shared_edge_with(&self, other_poly: PolyIndex) -> Option<(VertexIndex, VertexIndex)> {
        for (i, &adj) in self.adjacency.iter().enumerate() {
            if adj == other_poly {
                let a = self.vertices[i];
                let b = self.vertices[(i + 1) % self.vertices.len()];
                return Some((a, b));
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// NavMesh
// ---------------------------------------------------------------------------

/// A navigation mesh composed of convex polygons.
///
/// The navmesh defines the walkable surface for AI agents. It is typically
/// generated from level geometry using [`NavMeshBuilder`] and then queried
/// at runtime via [`NavMeshQuery`].
#[derive(Debug, Clone)]
pub struct NavMesh {
    /// All vertices in the mesh (world-space positions).
    pub vertices: Vec<Vec3>,
    /// All convex polygons.
    pub polygons: Vec<NavPoly>,
    /// Axis-aligned bounding box minimum.
    pub bounds_min: Vec3,
    /// Axis-aligned bounding box maximum.
    pub bounds_max: Vec3,
}

impl NavMesh {
    /// Creates a new empty navmesh.
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            polygons: Vec::new(),
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::ZERO,
        }
    }

    /// Returns the number of polygons.
    pub fn poly_count(&self) -> usize {
        self.polygons.len()
    }

    /// Returns the number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Recomputes bounding box from vertices.
    pub fn recompute_bounds(&mut self) {
        if self.vertices.is_empty() {
            self.bounds_min = Vec3::ZERO;
            self.bounds_max = Vec3::ZERO;
            return;
        }
        let mut min = self.vertices[0];
        let mut max = self.vertices[0];
        for v in &self.vertices[1..] {
            min = min.min(*v);
            max = max.max(*v);
        }
        self.bounds_min = min;
        self.bounds_max = max;
    }

    /// Returns `true` if the point is within the navmesh bounding box (XZ).
    pub fn contains_point_2d(&self, point: Vec3) -> bool {
        point.x >= self.bounds_min.x
            && point.x <= self.bounds_max.x
            && point.z >= self.bounds_min.z
            && point.z <= self.bounds_max.z
    }

    /// Test if a point lies inside a specific polygon (2D projection on XZ).
    pub fn contains_point(&self, poly_idx: usize, point: Vec3) -> bool {
        if poly_idx >= self.polygons.len() {
            return false;
        }
        let poly = &self.polygons[poly_idx];
        point_in_polygon_xz(point, &poly.vertices, &self.vertices)
    }

    /// Find the nearest polygon to a point. Returns `(poly_index, nearest_point)`.
    pub fn find_nearest_polygon(&self, point: Vec3) -> Option<(usize, Vec3)> {
        if self.polygons.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_point = Vec3::ZERO;
        let mut best_dist_sq = f32::INFINITY;

        for (i, poly) in self.polygons.iter().enumerate() {
            let projected = closest_point_on_polygon(point, &poly.vertices, &self.vertices);
            let dist_sq = (projected - point).length_squared();
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_idx = i;
                best_point = projected;
            }
        }

        Some((best_idx, best_point))
    }

    /// Find the nearest point on the navmesh surface to the given world position.
    pub fn find_nearest_point(&self, point: Vec3) -> Vec3 {
        self.find_nearest_polygon(point)
            .map(|(_, p)| p)
            .unwrap_or(point)
    }
}

impl Default for NavMesh {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// 2D cross product of (b - a) x (c - a) on the XZ plane.
fn cross_2d(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    (b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x)
}

/// Test if a point is inside a convex polygon on the XZ plane.
/// Handles both CW and CCW winding orders by detecting the winding first.
fn point_in_polygon_xz(point: Vec3, indices: &[VertexIndex], vertices: &[Vec3]) -> bool {
    let n = indices.len();
    if n < 3 {
        return false;
    }

    // Determine winding order from the signed area of the polygon.
    let mut signed_area = 0.0f32;
    for i in 0..n {
        let a = vertices[indices[i] as usize];
        let b = vertices[indices[(i + 1) % n] as usize];
        signed_area += (b.x - a.x) * (b.z + a.z);
    }
    // If signed_area > 0 => CW, < 0 => CCW on XZ plane.
    let expect_positive = signed_area < 0.0; // CCW: cross products should be >= 0.

    for i in 0..n {
        let a = vertices[indices[i] as usize];
        let b = vertices[indices[(i + 1) % n] as usize];
        let cross = cross_2d(a, b, point);
        if expect_positive {
            if cross < -1e-5 {
                return false;
            }
        } else {
            if cross > 1e-5 {
                return false;
            }
        }
    }
    true
}

/// Find the closest point on a convex polygon to a given point (3D, projected to XZ for containment).
fn closest_point_on_polygon(point: Vec3, indices: &[VertexIndex], vertices: &[Vec3]) -> Vec3 {
    // If point is inside the polygon (XZ), project vertically.
    if point_in_polygon_xz(point, indices, vertices) {
        // Compute the height at this XZ position using barycentric interpolation
        // on the first triangle.
        if indices.len() >= 3 {
            let a = vertices[indices[0] as usize];
            let b = vertices[indices[1] as usize];
            let c = vertices[indices[2] as usize];
            let y = barycentric_height(point, a, b, c);
            return Vec3::new(point.x, y, point.z);
        }
        return point;
    }

    // Otherwise find closest point on the polygon edges.
    let n = indices.len();
    let mut best = vertices[indices[0] as usize];
    let mut best_dist_sq = f32::INFINITY;

    for i in 0..n {
        let a = vertices[indices[i] as usize];
        let b = vertices[indices[(i + 1) % n] as usize];
        let closest = closest_point_on_segment(point, a, b);
        let dist_sq = (closest - point).length_squared();
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best = closest;
        }
    }

    best
}

/// Compute height at XZ position using barycentric coordinates of triangle (a, b, c).
fn barycentric_height(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let v0 = Vec3::new(c.x - a.x, 0.0, c.z - a.z);
    let v1 = Vec3::new(b.x - a.x, 0.0, b.z - a.z);
    let v2 = Vec3::new(p.x - a.x, 0.0, p.z - a.z);

    let dot00 = v0.x * v0.x + v0.z * v0.z;
    let dot01 = v0.x * v1.x + v0.z * v1.z;
    let dot02 = v0.x * v2.x + v0.z * v2.z;
    let dot11 = v1.x * v1.x + v1.z * v1.z;
    let dot12 = v1.x * v2.x + v1.z * v2.z;

    let inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    a.y + u * (c.y - a.y) + v * (b.y - a.y)
}

/// Closest point on line segment [a, b] to point p.
fn closest_point_on_segment(p: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-12 {
        return a;
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    a + ab * t
}

/// Compute the normal of a triangle.
fn triangle_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    (b - a).cross(c - a).normalize_or_zero()
}

/// Compute the centroid of a polygon.
fn polygon_centroid(indices: &[VertexIndex], vertices: &[Vec3]) -> Vec3 {
    if indices.is_empty() {
        return Vec3::ZERO;
    }
    let sum: Vec3 = indices.iter().map(|&i| vertices[i as usize]).sum();
    sum / indices.len() as f32
}

// ---------------------------------------------------------------------------
// NavMeshQuery
// ---------------------------------------------------------------------------

/// Min-heap entry for A* on navmesh polygon graph.
#[derive(Debug, Clone)]
struct NavOpenEntry {
    poly_idx: u32,
    f_cost: f32,
}

impl PartialEq for NavOpenEntry {
    fn eq(&self, other: &Self) -> bool {
        self.poly_idx == other.poly_idx
    }
}

impl Eq for NavOpenEntry {}

impl PartialOrd for NavOpenEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NavOpenEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f_cost
            .partial_cmp(&self.f_cost)
            .unwrap_or(Ordering::Equal)
    }
}

/// Spatial queries over a [`NavMesh`].
pub struct NavMeshQuery<'a> {
    /// Reference to the navmesh being queried.
    navmesh: &'a NavMesh,
}

impl<'a> NavMeshQuery<'a> {
    /// Creates a new query context for the given navmesh.
    pub fn new(navmesh: &'a NavMesh) -> Self {
        Self { navmesh }
    }

    /// Returns a reference to the underlying navmesh.
    pub fn navmesh(&self) -> &NavMesh {
        self.navmesh
    }

    /// Finds the nearest point on the navmesh surface to the given world position.
    pub fn find_nearest_point(&self, position: Vec3) -> Option<(PolyIndex, Vec3)> {
        self.navmesh
            .find_nearest_polygon(position)
            .map(|(idx, pt)| (idx as u32, pt))
    }

    /// Finds the polygon that contains the given point (projected onto XZ).
    pub fn find_containing_polygon(&self, position: Vec3) -> Option<PolyIndex> {
        for (i, poly) in self.navmesh.polygons.iter().enumerate() {
            if point_in_polygon_xz(position, &poly.vertices, &self.navmesh.vertices) {
                return Some(i as u32);
            }
        }
        None
    }

    /// Computes a path corridor (sequence of polygon indices) from start to end
    /// using A* on the polygon adjacency graph.
    pub fn find_path_corridor(
        &self,
        start_poly: PolyIndex,
        end_poly: PolyIndex,
    ) -> EngineResult<Vec<PolyIndex>> {
        profiling::scope!("NavMeshQuery::find_path_corridor");

        if start_poly == end_poly {
            return Ok(vec![start_poly]);
        }

        let polys = &self.navmesh.polygons;
        let max_polys = polys.len();
        if start_poly as usize >= max_polys || end_poly as usize >= max_polys {
            return Err(genovo_core::EngineError::InvalidArgument(
                "Polygon index out of range".into(),
            ));
        }

        let goal_centroid = polys[end_poly as usize].centroid;

        let mut g_costs: HashMap<u32, f32> = HashMap::new();
        let mut parents: HashMap<u32, u32> = HashMap::new();
        let mut closed: HashSet<u32> = HashSet::new();
        let mut open: BinaryHeap<NavOpenEntry> = BinaryHeap::new();

        let h = (polys[start_poly as usize].centroid - goal_centroid).length();
        g_costs.insert(start_poly, 0.0);
        open.push(NavOpenEntry {
            poly_idx: start_poly,
            f_cost: h,
        });

        let max_iterations = max_polys * 2;
        let mut iterations = 0;

        while let Some(entry) = open.pop() {
            let current = entry.poly_idx;

            if current == end_poly {
                // Reconstruct corridor.
                let mut corridor = Vec::new();
                let mut c = end_poly;
                loop {
                    corridor.push(c);
                    if c == start_poly {
                        break;
                    }
                    match parents.get(&c) {
                        Some(&p) => c = p,
                        None => break,
                    }
                }
                corridor.reverse();
                return Ok(corridor);
            }

            if closed.contains(&current) {
                continue;
            }
            closed.insert(current);

            iterations += 1;
            if iterations > max_iterations {
                break;
            }

            let current_g = g_costs[&current];
            let current_poly = &polys[current as usize];

            for &neighbor in &current_poly.adjacency {
                if neighbor == u32::MAX {
                    continue;
                }
                if closed.contains(&neighbor) {
                    continue;
                }

                let edge_cost =
                    (current_poly.centroid - polys[neighbor as usize].centroid).length();
                let tentative_g = current_g + edge_cost;
                let existing_g = g_costs.get(&neighbor).copied().unwrap_or(f32::INFINITY);

                if tentative_g < existing_g {
                    g_costs.insert(neighbor, tentative_g);
                    parents.insert(neighbor, current);
                    let h = (polys[neighbor as usize].centroid - goal_centroid).length();
                    open.push(NavOpenEntry {
                        poly_idx: neighbor,
                        f_cost: tentative_g + h,
                    });
                }
            }
        }

        Err(genovo_core::EngineError::NotFound(
            "No path corridor found between polygons".into(),
        ))
    }

    /// Computes a smoothed path from start to end using A* on the polygon graph
    /// followed by the funnel algorithm.
    pub fn find_path(&self, start: Vec3, end: Vec3) -> Option<Vec<Vec3>> {
        profiling::scope!("NavMeshQuery::find_path");

        let start_poly = self.find_containing_polygon(start).or_else(|| {
            self.navmesh
                .find_nearest_polygon(start)
                .map(|(i, _)| i as u32)
        })?;
        let end_poly = self.find_containing_polygon(end).or_else(|| {
            self.navmesh
                .find_nearest_polygon(end)
                .map(|(i, _)| i as u32)
        })?;

        let corridor = self.find_path_corridor(start_poly, end_poly).ok()?;

        if corridor.len() == 1 {
            return Some(vec![start, end]);
        }

        // Build portal edges from corridor.
        let portals = self.build_portals(&corridor, start, end);

        // Run funnel algorithm.
        Some(funnel_algorithm(&portals, start, end))
    }

    /// Performs a raycast against the navmesh surface.
    pub fn raycast(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
    ) -> Option<(f32, PolyIndex)> {
        let dir_norm = direction.normalize_or_zero();
        if dir_norm.length_squared() < 0.5 {
            return None;
        }

        let end = origin + dir_norm * max_distance;

        // Walk the navmesh polygons the ray crosses.
        let start_poly = self.find_containing_polygon(origin)?;
        let mut current_poly = start_poly;
        let mut visited: HashSet<u32> = HashSet::new();
        visited.insert(current_poly);

        // Walk polygon-by-polygon.
        let max_steps = self.navmesh.polygons.len();
        for _ in 0..max_steps {
            let poly = &self.navmesh.polygons[current_poly as usize];
            let n = poly.vertices.len();

            // Check if end point is in current polygon.
            if point_in_polygon_xz(end, &poly.vertices, &self.navmesh.vertices) {
                let t = (end - origin).length() / max_distance;
                return Some((t.min(1.0), current_poly));
            }

            // Find which edge the ray exits through.
            let mut found_exit = false;
            for i in 0..n {
                let a = self.navmesh.vertices[poly.vertices[i] as usize];
                let b = self.navmesh.vertices[poly.vertices[(i + 1) % n] as usize];

                if let Some(_t) = ray_segment_intersection_2d(origin, end, a, b) {
                    let neighbor = poly.adjacency[i];
                    if neighbor == u32::MAX {
                        // Hit a boundary edge: the ray exits the navmesh.
                        let hit_point = closest_point_on_segment(origin, a, b);
                        let dist = (hit_point - origin).length();
                        return Some((dist / max_distance, current_poly));
                    }
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        current_poly = neighbor;
                        found_exit = true;
                        break;
                    }
                }
            }

            if !found_exit {
                break;
            }
        }

        None
    }

    /// Build portal edges from a corridor of polygon indices.
    fn build_portals(&self, corridor: &[PolyIndex], start: Vec3, end: Vec3) -> Vec<Portal> {
        let mut portals = Vec::with_capacity(corridor.len() + 1);

        // First portal: start point.
        portals.push(Portal {
            left: start,
            right: start,
        });

        // Interior portals from shared edges.
        for i in 0..corridor.len() - 1 {
            let current = &self.navmesh.polygons[corridor[i] as usize];
            let next_idx = corridor[i + 1];

            if let Some((vi_a, vi_b)) = current.shared_edge_with(next_idx) {
                let a = self.navmesh.vertices[vi_a as usize];
                let b = self.navmesh.vertices[vi_b as usize];

                // Determine left/right ordering relative to the path direction.
                // We use the cross product with the general direction.
                let dir = end - start;
                let cross = (b - a).cross(dir);

                if cross.y >= 0.0 {
                    portals.push(Portal { left: a, right: b });
                } else {
                    portals.push(Portal { left: b, right: a });
                }
            }
        }

        // Last portal: end point.
        portals.push(Portal {
            left: end,
            right: end,
        });

        portals
    }
}

/// 2D ray-segment intersection test (XZ plane).
/// Returns the parameter t along the ray [origin -> end] where it crosses [a, b].
fn ray_segment_intersection_2d(origin: Vec3, end: Vec3, a: Vec3, b: Vec3) -> Option<f32> {
    let d = end - origin;
    let e = b - a;

    let denom = d.x * e.z - d.z * e.x;
    if denom.abs() < 1e-8 {
        return None; // Parallel.
    }

    let t = ((a.x - origin.x) * e.z - (a.z - origin.z) * e.x) / denom;
    let u = ((a.x - origin.x) * d.z - (a.z - origin.z) * d.x) / denom;

    if t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 {
        Some(t)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Funnel Algorithm (Simple Stupid Funnel Algorithm)
// ---------------------------------------------------------------------------

/// A portal edge between two adjacent polygons.
#[derive(Debug, Clone)]
struct Portal {
    left: Vec3,
    right: Vec3,
}

/// The Simple Stupid Funnel Algorithm for path string-pulling.
///
/// Given a sequence of portal edges (from the corridor), this produces
/// the shortest path that stays within the corridor.
fn funnel_algorithm(portals: &[Portal], start: Vec3, end: Vec3) -> Vec<Vec3> {
    if portals.len() < 2 {
        return vec![start, end];
    }

    let mut path = Vec::new();
    path.push(start);

    let mut apex = start;
    #[allow(unused_assignments)]
    let mut apex_idx = 0;
    let mut left = portals[0].left;
    let mut right = portals[0].right;
    let mut left_idx = 0usize;
    let mut right_idx = 0usize;

    for i in 1..portals.len() {
        let new_left = portals[i].left;
        let new_right = portals[i].right;

        // Update right vertex.
        if tri_area_2d(apex, right, new_right) <= 0.0 {
            if apex == right || tri_area_2d(apex, left, new_right) > 0.0 {
                // Tighten the funnel.
                right = new_right;
                right_idx = i;
            } else {
                // Right over left: add left to path and restart.
                path.push(left);
                apex = left;
                apex_idx = left_idx;
                // Reset funnel.
                left = apex;
                right = apex;
                left_idx = apex_idx;
                right_idx = apex_idx;
                // Re-scan from apex_idx + 1.
                continue;
            }
        }

        // Update left vertex.
        if tri_area_2d(apex, left, new_left) >= 0.0 {
            if apex == left || tri_area_2d(apex, right, new_left) < 0.0 {
                left = new_left;
                left_idx = i;
            } else {
                path.push(right);
                apex = right;
                apex_idx = right_idx;
                left = apex;
                right = apex;
                left_idx = apex_idx;
                right_idx = apex_idx;
                continue;
            }
        }
    }

    // Add the end point.
    if path.last() != Some(&end) {
        path.push(end);
    }

    // Deduplicate consecutive identical points.
    path.dedup_by(|a, b| (*a - *b).length_squared() < 1e-8);

    path
}

/// Signed 2D triangle area on XZ plane (positive = CCW).
fn tri_area_2d(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    (b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x)
}

// ---------------------------------------------------------------------------
// NavMeshBuilder
// ---------------------------------------------------------------------------

/// Build configuration for navmesh generation.
#[derive(Debug, Clone)]
pub struct NavMeshBuildConfig {
    /// Voxel cell size on XZ plane (smaller = more detail, slower).
    pub cell_size: f32,
    /// Voxel cell height (Y axis).
    pub cell_height: f32,
    /// Maximum walkable slope angle (radians).
    pub max_slope: f32,
    /// Agent height for clearance checks.
    pub agent_height: f32,
    /// Agent radius for erosion.
    pub agent_radius: f32,
    /// Maximum ledge height the agent can step over.
    pub max_step_height: f32,
    /// Minimum region area (in cells) to keep after filtering.
    pub min_region_area: u32,
    /// Maximum vertices per polygon.
    pub max_verts_per_poly: u32,
}

impl Default for NavMeshBuildConfig {
    fn default() -> Self {
        Self {
            cell_size: 0.3,
            cell_height: 0.2,
            max_slope: std::f32::consts::FRAC_PI_4,
            agent_height: 2.0,
            agent_radius: 0.6,
            max_step_height: 0.4,
            min_region_area: 8,
            max_verts_per_poly: 6,
        }
    }
}

/// Builds a [`NavMesh`] from input triangle geometry.
///
/// Supports two construction modes:
/// 1. `from_triangles` - Direct triangle-to-polygon construction (fast, for simple geometry).
/// 2. `build` - Full pipeline with merging of coplanar triangles.
pub struct NavMeshBuilder {
    /// Build configuration.
    config: NavMeshBuildConfig,
    /// Input triangles (groups of 3 vertices).
    input_vertices: Vec<Vec3>,
    /// Triangle indices (groups of 3).
    input_indices: Vec<u32>,
}

impl NavMeshBuilder {
    /// Creates a new builder with the given configuration.
    pub fn new(config: NavMeshBuildConfig) -> Self {
        Self {
            config,
            input_vertices: Vec::new(),
            input_indices: Vec::new(),
        }
    }

    /// Creates a new builder with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(NavMeshBuildConfig::default())
    }

    /// Adds triangle soup geometry to the builder.
    pub fn add_geometry(&mut self, vertices: &[Vec3], indices: &[u32]) {
        let base = self.input_vertices.len() as u32;
        self.input_vertices.extend_from_slice(vertices);
        self.input_indices.extend(indices.iter().map(|i| i + base));
    }

    /// Returns the build configuration.
    pub fn config(&self) -> &NavMeshBuildConfig {
        &self.config
    }

    /// Build a navmesh directly from triangles. Each triangle becomes a polygon,
    /// then coplanar adjacent triangles are merged into larger convex polygons.
    pub fn from_triangles(vertices: &[Vec3], indices: &[u32]) -> NavMesh {
        profiling::scope!("NavMeshBuilder::from_triangles");

        if indices.len() < 3 || vertices.is_empty() {
            return NavMesh::new();
        }

        let num_tris = indices.len() / 3;

        // Step 1: Build initial polygons (one per triangle).
        let mut polygons: Vec<NavPoly> = Vec::with_capacity(num_tris);
        for t in 0..num_tris {
            let i0 = indices[t * 3];
            let i1 = indices[t * 3 + 1];
            let i2 = indices[t * 3 + 2];

            let a = vertices[i0 as usize];
            let b = vertices[i1 as usize];
            let c = vertices[i2 as usize];

            let normal = triangle_normal(a, b, c);
            let centroid = (a + b + c) / 3.0;

            let mut verts = SmallVec::new();
            verts.push(i0);
            verts.push(i1);
            verts.push(i2);

            let adjacency = SmallVec::from_elem(u32::MAX, 3);

            polygons.push(NavPoly {
                vertices: verts,
                adjacency,
                area_type: 0,
                centroid,
                normal,
            });
        }

        // Step 2: Build adjacency by finding shared edges.
        Self::build_adjacency(&mut polygons, vertices);

        // Step 3: Merge coplanar adjacent triangles into larger convex polygons.
        let merged = Self::merge_coplanar_polygons(polygons, vertices, 6);

        let mut navmesh = NavMesh {
            vertices: vertices.to_vec(),
            polygons: merged,
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::ZERO,
        };
        navmesh.recompute_bounds();
        navmesh
    }

    /// Build the navmesh from accumulated geometry.
    pub fn build(&self) -> EngineResult<NavMesh> {
        profiling::scope!("NavMeshBuilder::build");

        if self.input_indices.len() < 3 {
            return Ok(NavMesh::new());
        }

        // Filter walkable triangles based on slope.
        let mut walkable_indices = Vec::new();
        let max_slope_cos = self.config.max_slope.cos();

        let num_tris = self.input_indices.len() / 3;
        for t in 0..num_tris {
            let i0 = self.input_indices[t * 3] as usize;
            let i1 = self.input_indices[t * 3 + 1] as usize;
            let i2 = self.input_indices[t * 3 + 2] as usize;

            if i0 >= self.input_vertices.len()
                || i1 >= self.input_vertices.len()
                || i2 >= self.input_vertices.len()
            {
                continue;
            }

            let a = self.input_vertices[i0];
            let b = self.input_vertices[i1];
            let c = self.input_vertices[i2];

            let normal = triangle_normal(a, b, c);
            // Check slope: |normal.y| >= cos(max_slope) means the surface is walkable.
            // We use abs to handle both CW and CCW winding orders.
            if normal.y.abs() >= max_slope_cos {
                walkable_indices.push(self.input_indices[t * 3]);
                walkable_indices.push(self.input_indices[t * 3 + 1]);
                walkable_indices.push(self.input_indices[t * 3 + 2]);
            }
        }

        if walkable_indices.is_empty() {
            log::info!("NavMesh build: no walkable triangles found");
            return Ok(NavMesh::new());
        }

        let navmesh = Self::from_triangles(&self.input_vertices, &walkable_indices);

        log::info!(
            "NavMesh built: {} vertices, {} polygons",
            navmesh.vertex_count(),
            navmesh.poly_count()
        );

        Ok(navmesh)
    }

    /// Build adjacency graph by finding shared edges between polygons.
    fn build_adjacency(polygons: &mut [NavPoly], _vertices: &[Vec3]) {
        // Edge map: (min_vertex, max_vertex) -> (polygon_index, edge_index)
        let mut edge_map: HashMap<(u32, u32), Vec<(usize, usize)>> = HashMap::new();

        for (poly_idx, poly) in polygons.iter().enumerate() {
            let n = poly.vertices.len();
            for edge_idx in 0..n {
                let v0 = poly.vertices[edge_idx];
                let v1 = poly.vertices[(edge_idx + 1) % n];
                let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                edge_map
                    .entry(key)
                    .or_insert_with(Vec::new)
                    .push((poly_idx, edge_idx));
            }
        }

        for entries in edge_map.values() {
            if entries.len() == 2 {
                let (poly_a, edge_a) = entries[0];
                let (poly_b, edge_b) = entries[1];
                polygons[poly_a].adjacency[edge_a] = poly_b as u32;
                polygons[poly_b].adjacency[edge_b] = poly_a as u32;
            }
        }
    }

    /// Merge coplanar adjacent polygons into larger convex polygons.
    fn merge_coplanar_polygons(
        mut polygons: Vec<NavPoly>,
        vertices: &[Vec3],
        max_verts: usize,
    ) -> Vec<NavPoly> {
        let normal_threshold = 0.998; // ~3.6 degrees
        let mut merged = true;

        while merged {
            merged = false;

            for i in 0..polygons.len() {
                if polygons[i].vertices.is_empty() {
                    continue;
                }

                let adj_list: SmallVec<[PolyIndex; 6]> = polygons[i].adjacency.clone();
                for &neighbor_idx in &adj_list {
                    if neighbor_idx == u32::MAX {
                        continue;
                    }
                    let j = neighbor_idx as usize;
                    if j >= polygons.len() || polygons[j].vertices.is_empty() {
                        continue;
                    }
                    if i == j {
                        continue;
                    }

                    // Check coplanarity.
                    let dot = polygons[i].normal.dot(polygons[j].normal);
                    if dot < normal_threshold {
                        continue;
                    }

                    // Check if combined polygon would exceed max vertices.
                    let combined_verts =
                        polygons[i].vertices.len() + polygons[j].vertices.len() - 2;
                    if combined_verts > max_verts {
                        continue;
                    }

                    // Find the shared edge.
                    let shared = Self::find_shared_edge(&polygons[i], &polygons[j]);
                    if shared.is_none() {
                        continue;
                    }
                    let (edge_a, edge_b) = shared.unwrap();

                    // Merge: combine vertex lists.
                    let merged_verts =
                        Self::merge_vertex_lists(&polygons[i], &polygons[j], edge_a, edge_b);

                    // Verify the merged polygon is convex.
                    if !Self::is_convex_xz(&merged_verts, vertices) {
                        continue;
                    }

                    // Apply merge.
                    let normal = polygons[i].normal;
                    let area_type = polygons[i].area_type;
                    let centroid = polygon_centroid(&merged_verts, vertices);

                    // Mark polygon j as empty.
                    polygons[j].vertices.clear();
                    polygons[j].adjacency.clear();

                    // Update polygon i.
                    polygons[i].vertices = merged_verts;
                    polygons[i].centroid = centroid;
                    polygons[i].normal = normal;
                    polygons[i].area_type = area_type;
                    polygons[i].adjacency =
                        SmallVec::from_elem(u32::MAX, polygons[i].vertices.len());

                    merged = true;
                    break;
                }
            }

            if merged {
                // Remove empty polygons and rebuild adjacency.
                let old_to_new: Vec<Option<usize>> = {
                    let mut map = Vec::with_capacity(polygons.len());
                    let mut new_idx = 0;
                    for p in &polygons {
                        if p.vertices.is_empty() {
                            map.push(None);
                        } else {
                            map.push(Some(new_idx));
                            new_idx += 1;
                        }
                    }
                    map
                };

                polygons.retain(|p| !p.vertices.is_empty());

                // Remap adjacency indices.
                for poly in &mut polygons {
                    for adj in &mut poly.adjacency {
                        if *adj != u32::MAX {
                            let old = *adj as usize;
                            if old < old_to_new.len() {
                                if let Some(new) = old_to_new[old] {
                                    *adj = new as u32;
                                } else {
                                    *adj = u32::MAX;
                                }
                            } else {
                                *adj = u32::MAX;
                            }
                        }
                    }
                }

                // Rebuild adjacency from scratch.
                Self::build_adjacency(&mut polygons, vertices);
                merged = true;
            }
        }

        polygons
    }

    fn find_shared_edge(a: &NavPoly, b: &NavPoly) -> Option<(usize, usize)> {
        let na = a.vertices.len();
        let nb = b.vertices.len();
        for i in 0..na {
            let av0 = a.vertices[i];
            let av1 = a.vertices[(i + 1) % na];
            for j in 0..nb {
                let bv0 = b.vertices[j];
                let bv1 = b.vertices[(j + 1) % nb];
                // Shared edges have reversed winding.
                if (av0 == bv1 && av1 == bv0) || (av0 == bv0 && av1 == bv1) {
                    return Some((i, j));
                }
            }
        }
        None
    }

    fn merge_vertex_lists(
        a: &NavPoly,
        b: &NavPoly,
        edge_a: usize,
        edge_b: usize,
    ) -> SmallVec<[VertexIndex; 6]> {
        let na = a.vertices.len();
        let nb = b.vertices.len();

        let mut result = SmallVec::new();

        // Add vertices from polygon a, skipping the second vertex of the shared edge.
        let skip_a = (edge_a + 1) % na;
        for i in 0..na {
            let idx = (edge_a + 1 + i) % na;
            if idx == skip_a && i != 0 {
                continue;
            }
            if i == na - 1 {
                continue; // Skip the first vertex of the shared edge from a.
            }
            result.push(a.vertices[idx]);
        }

        // Add vertices from polygon b, skipping the shared edge vertices.
        let skip_b0 = edge_b;
        let skip_b1 = (edge_b + 1) % nb;
        for i in 0..nb {
            let idx = (edge_b + 2 + i) % nb;
            if idx == skip_b0 || idx == skip_b1 {
                continue;
            }
            result.push(b.vertices[idx]);
        }

        // Deduplicate while preserving order.
        let mut seen = HashSet::new();
        result.retain(|v| seen.insert(*v));

        result
    }

    fn is_convex_xz(indices: &[VertexIndex], vertices: &[Vec3]) -> bool {
        let n = indices.len();
        if n < 3 {
            return false;
        }

        let mut sign = 0i32;
        for i in 0..n {
            let a = vertices[indices[i] as usize];
            let b = vertices[indices[(i + 1) % n] as usize];
            let c = vertices[indices[(i + 2) % n] as usize];
            let cross = cross_2d(a, b, c);
            if cross.abs() > 1e-6 {
                let s = if cross > 0.0 { 1 } else { -1 };
                if sign == 0 {
                    sign = s;
                } else if sign != s {
                    return false;
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// NavMeshAgent
// ---------------------------------------------------------------------------

/// ECS component for entities that navigate using a navmesh.
#[derive(Debug, Clone)]
pub struct NavMeshAgent {
    /// Maximum movement speed (units per second).
    pub speed: f32,
    /// Maximum acceleration (units per second squared).
    pub acceleration: f32,
    /// Agent collision radius.
    pub radius: f32,
    /// Agent height (for clearance checks).
    pub height: f32,
    /// Current target position (if navigating).
    pub target: Option<Vec3>,
    /// Current velocity.
    pub velocity: Vec3,
    /// Current world position.
    pub position: Vec3,
    /// Index of the polygon the agent is currently on.
    pub current_poly: Option<PolyIndex>,
    /// Whether the agent is actively navigating toward a target.
    pub is_navigating: bool,
    /// Remaining path waypoints.
    pub path_waypoints: Vec<Vec3>,
    /// Index of the current waypoint being pursued.
    pub current_waypoint_index: usize,
}

impl Component for NavMeshAgent {}

impl NavMeshAgent {
    /// Creates a new navmesh agent with default settings.
    pub fn new(speed: f32, radius: f32, height: f32) -> Self {
        Self {
            speed,
            acceleration: speed * 8.0,
            radius,
            height,
            target: None,
            velocity: Vec3::ZERO,
            position: Vec3::ZERO,
            current_poly: None,
            is_navigating: false,
            path_waypoints: Vec::new(),
            current_waypoint_index: 0,
        }
    }

    /// Sets a navigation target. The agent will request a path on the next update.
    pub fn set_target(&mut self, target: Vec3) {
        self.target = Some(target);
        self.is_navigating = true;
        self.current_waypoint_index = 0;
    }

    /// Stops navigation and clears the current path.
    pub fn stop(&mut self) {
        self.target = None;
        self.is_navigating = false;
        self.path_waypoints.clear();
        self.current_waypoint_index = 0;
        self.velocity = Vec3::ZERO;
    }

    /// Returns `true` if the agent has reached its target.
    pub fn has_reached_target(&self) -> bool {
        !self.is_navigating && self.target.is_some()
    }

    /// Move the agent along its path for the given delta time.
    ///
    /// Returns the new position after movement.
    pub fn move_along_path(&mut self, dt: f32) -> Vec3 {
        if !self.is_navigating || self.path_waypoints.is_empty() {
            return self.position;
        }

        if self.current_waypoint_index >= self.path_waypoints.len() {
            self.is_navigating = false;
            self.velocity = Vec3::ZERO;
            return self.position;
        }

        let target_wp = self.path_waypoints[self.current_waypoint_index];
        let to_target = target_wp - self.position;
        let dist = to_target.length();

        let arrival_dist = self.radius * 0.5;
        if dist < arrival_dist {
            // Arrived at waypoint.
            self.position = target_wp;
            self.current_waypoint_index += 1;

            if self.current_waypoint_index >= self.path_waypoints.len() {
                // Reached final waypoint.
                self.is_navigating = false;
                self.velocity = Vec3::ZERO;
                return self.position;
            }
            return self.move_along_path(dt);
        }

        // Compute desired velocity toward the current waypoint.
        let desired_dir = to_target / dist;
        let desired_speed = if self.current_waypoint_index == self.path_waypoints.len() - 1 {
            // Slow down near the final waypoint (arrival behavior).
            let slow_radius = self.speed * 0.5;
            if dist < slow_radius {
                self.speed * (dist / slow_radius)
            } else {
                self.speed
            }
        } else {
            self.speed
        };

        let desired_velocity = desired_dir * desired_speed;

        // Apply acceleration.
        let vel_diff = desired_velocity - self.velocity;
        let accel_mag = vel_diff.length();
        if accel_mag > 0.0 {
            let max_accel = self.acceleration * dt;
            if accel_mag <= max_accel {
                self.velocity = desired_velocity;
            } else {
                self.velocity += vel_diff * (max_accel / accel_mag);
            }
        }

        // Clamp speed.
        let current_speed = self.velocity.length();
        if current_speed > self.speed {
            self.velocity = self.velocity * (self.speed / current_speed);
        }

        // Update position.
        self.position += self.velocity * dt;

        self.position
    }
}

// ---------------------------------------------------------------------------
// CrowdManager
// ---------------------------------------------------------------------------

/// Manages group movement and local avoidance for multiple navmesh agents.
pub struct CrowdManager {
    /// Maximum number of agents this crowd can manage.
    max_agents: usize,
    /// Active agents with their indices.
    agents: Vec<CrowdAgent>,
    /// Configuration for obstacle avoidance.
    avoidance_config: ObstacleAvoidanceConfig,
    /// Obstacle avoidance solver.
    avoidance: ObstacleAvoidance,
}

/// Internal crowd agent data.
#[derive(Debug, Clone)]
struct CrowdAgent {
    /// Agent id.
    pub id: u32,
    /// Position.
    pub position: Vec3,
    /// Velocity.
    pub velocity: Vec3,
    /// Desired velocity (before avoidance).
    pub desired_velocity: Vec3,
    /// Collision radius.
    pub radius: f32,
    /// Max speed.
    pub max_speed: f32,
    /// Active flag.
    pub active: bool,
}

impl CrowdManager {
    /// Creates a new crowd manager.
    pub fn new(max_agents: usize) -> Self {
        let config = ObstacleAvoidanceConfig::default();
        Self {
            max_agents,
            agents: Vec::with_capacity(max_agents),
            avoidance_config: config.clone(),
            avoidance: ObstacleAvoidance::new(config),
        }
    }

    /// Add an agent to the crowd. Returns the agent index, or None if full.
    pub fn add_agent(
        &mut self,
        id: u32,
        position: Vec3,
        radius: f32,
        max_speed: f32,
    ) -> Option<usize> {
        if self.agents.len() >= self.max_agents {
            return None;
        }
        let idx = self.agents.len();
        self.agents.push(CrowdAgent {
            id,
            position,
            velocity: Vec3::ZERO,
            desired_velocity: Vec3::ZERO,
            radius,
            max_speed,
            active: true,
        });
        Some(idx)
    }

    /// Remove an agent by index.
    pub fn remove_agent(&mut self, idx: usize) {
        if idx < self.agents.len() {
            self.agents[idx].active = false;
        }
    }

    /// Set the desired velocity for an agent.
    pub fn set_desired_velocity(&mut self, idx: usize, velocity: Vec3) {
        if idx < self.agents.len() {
            self.agents[idx].desired_velocity = velocity;
        }
    }

    /// Get an agent's position.
    pub fn agent_position(&self, idx: usize) -> Option<Vec3> {
        self.agents.get(idx).filter(|a| a.active).map(|a| a.position)
    }

    /// Get an agent's velocity.
    pub fn agent_velocity(&self, idx: usize) -> Option<Vec3> {
        self.agents.get(idx).filter(|a| a.active).map(|a| a.velocity)
    }

    /// Updates all managed agents for the current frame.
    ///
    /// 1. Collect neighbor data for each agent.
    /// 2. Compute collision-free velocities using ORCA.
    /// 3. Apply velocities and update positions.
    pub fn update(&mut self, dt: f32, _navmesh: &NavMesh) {
        profiling::scope!("CrowdManager::update");

        if dt <= 0.0 {
            return;
        }

        let num_agents = self.agents.len();

        // Collect new velocities without borrowing self mutably.
        let mut new_velocities = Vec::with_capacity(num_agents);

        for i in 0..num_agents {
            if !self.agents[i].active {
                new_velocities.push(Vec3::ZERO);
                continue;
            }

            // Build neighbor list.
            let mut neighbors = Vec::new();
            let query_radius = self.avoidance_config.agent_time_horizon * self.agents[i].max_speed
                + self.agents[i].radius * 2.0;

            for j in 0..num_agents {
                if i == j || !self.agents[j].active {
                    continue;
                }
                let dist = (self.agents[j].position - self.agents[i].position).length();
                if dist < query_radius {
                    neighbors.push((
                        self.agents[j].position,
                        self.agents[j].velocity,
                        self.agents[j].radius,
                    ));
                }
            }

            let new_vel = self.avoidance.compute_velocity(
                self.agents[i].position,
                self.agents[i].velocity,
                self.agents[i].desired_velocity,
                self.agents[i].radius,
                &neighbors,
                &[], // No static obstacles for now.
            );

            new_velocities.push(new_vel);
        }

        // Apply velocities.
        for (i, new_vel) in new_velocities.into_iter().enumerate() {
            if !self.agents[i].active {
                continue;
            }
            let agent = &mut self.agents[i];
            agent.velocity = new_vel;

            // Clamp to max speed.
            let speed = agent.velocity.length();
            if speed > agent.max_speed {
                let ratio = agent.max_speed / speed;
                agent.velocity *= ratio;
            }

            // Update position.
            let vel = agent.velocity;
            agent.position += vel * dt;
        }
    }

    /// Sets the obstacle avoidance configuration.
    pub fn set_avoidance_config(&mut self, config: ObstacleAvoidanceConfig) {
        self.avoidance_config = config.clone();
        self.avoidance = ObstacleAvoidance::new(config);
    }

    /// Returns the maximum number of agents.
    pub fn max_agents(&self) -> usize {
        self.max_agents
    }

    /// Returns the current number of active agents.
    pub fn active_agent_count(&self) -> usize {
        self.agents.iter().filter(|a| a.active).count()
    }
}

// ---------------------------------------------------------------------------
// ObstacleAvoidance (ORCA)
// ---------------------------------------------------------------------------

/// Configuration for the obstacle avoidance algorithm.
#[derive(Debug, Clone)]
pub struct ObstacleAvoidanceConfig {
    /// Number of velocity samples to evaluate.
    pub sample_count: u32,
    /// Time horizon for agent-agent avoidance (seconds).
    pub agent_time_horizon: f32,
    /// Time horizon for static obstacle avoidance (seconds).
    pub obstacle_time_horizon: f32,
    /// Weight for preferring the desired velocity direction.
    pub velocity_bias: f32,
}

impl Default for ObstacleAvoidanceConfig {
    fn default() -> Self {
        Self {
            sample_count: 32,
            agent_time_horizon: 2.0,
            obstacle_time_horizon: 1.0,
            velocity_bias: 0.4,
        }
    }
}

/// An ORCA half-plane constraint.
#[derive(Debug, Clone)]
struct OrcaLine {
    /// A point on the line.
    point: Vec3,
    /// Direction of the line (the valid half-plane is to the left of this direction).
    direction: Vec3,
}

/// Obstacle avoidance using ORCA (Optimal Reciprocal Collision Avoidance).
///
/// Computes a collision-free velocity for an agent given its desired velocity
/// and the velocities/positions of nearby agents and static obstacles.
pub struct ObstacleAvoidance {
    /// Algorithm configuration.
    config: ObstacleAvoidanceConfig,
}

impl ObstacleAvoidance {
    /// Creates a new obstacle avoidance solver.
    pub fn new(config: ObstacleAvoidanceConfig) -> Self {
        Self { config }
    }

    /// Creates a solver with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ObstacleAvoidanceConfig::default())
    }

    /// Computes a collision-free velocity for the given agent using ORCA.
    ///
    /// # Arguments
    /// * `agent_position` - Current agent position.
    /// * `agent_velocity` - Current agent velocity.
    /// * `desired_velocity` - The velocity the agent wants to move at.
    /// * `agent_radius` - The agent's collision radius.
    /// * `neighbors` - (position, velocity, radius) of nearby agents.
    /// * `obstacles` - Static obstacle line segments (start, end).
    pub fn compute_velocity(
        &self,
        agent_position: Vec3,
        agent_velocity: Vec3,
        desired_velocity: Vec3,
        agent_radius: f32,
        neighbors: &[(Vec3, Vec3, f32)],
        obstacles: &[(Vec3, Vec3)],
    ) -> Vec3 {
        profiling::scope!("ObstacleAvoidance::compute_velocity");

        let mut orca_lines: Vec<OrcaLine> = Vec::new();

        // Generate ORCA lines for static obstacles.
        for &(obs_start, obs_end) in obstacles {
            self.add_obstacle_orca_line(
                agent_position,
                agent_radius,
                obs_start,
                obs_end,
                &mut orca_lines,
            );
        }

        let num_obstacle_lines = orca_lines.len();

        // Generate ORCA lines for agent-agent avoidance.
        for &(neighbor_pos, neighbor_vel, neighbor_radius) in neighbors {
            self.add_agent_orca_line(
                agent_position,
                agent_velocity,
                agent_radius,
                neighbor_pos,
                neighbor_vel,
                neighbor_radius,
                &mut orca_lines,
            );
        }

        // Solve: find the velocity closest to desired_velocity that satisfies
        // all ORCA constraints.
        let result = self.linear_program_2d(&orca_lines, desired_velocity, num_obstacle_lines);

        result
    }

    /// Add ORCA half-plane for agent-agent avoidance.
    fn add_agent_orca_line(
        &self,
        pos_a: Vec3,
        vel_a: Vec3,
        radius_a: f32,
        pos_b: Vec3,
        vel_b: Vec3,
        radius_b: f32,
        lines: &mut Vec<OrcaLine>,
    ) {
        let relative_pos = pos_b - pos_a;
        let relative_vel = vel_a - vel_b;
        let combined_radius = radius_a + radius_b;
        let dist_sq = relative_pos.x * relative_pos.x + relative_pos.z * relative_pos.z;
        let combined_radius_sq = combined_radius * combined_radius;

        let tau = self.config.agent_time_horizon;
        let inv_tau = 1.0 / tau;

        // Vector from cutoff circle center to relative velocity.
        let w = relative_vel - relative_pos * inv_tau;
        let w_length_sq = w.x * w.x + w.z * w.z;

        let dot_product = w.x * relative_pos.x + w.z * relative_pos.z;

        if dist_sq > combined_radius_sq {
            // No collision. Project onto the velocity obstacle boundary.
            if dot_product < 0.0 && dot_product * dot_product > combined_radius_sq * w_length_sq {
                // Project on cut-off circle.
                let w_length = w_length_sq.sqrt();
                if w_length < 1e-6 {
                    return;
                }
                let unit_w = Vec3::new(w.x / w_length, 0.0, w.z / w_length);
                let direction = Vec3::new(unit_w.z, 0.0, -unit_w.x);
                let point = vel_a + unit_w * (combined_radius * inv_tau - w_length) * 0.5;
                lines.push(OrcaLine { point, direction });
            } else {
                // Project on legs.
                let leg = (dist_sq - combined_radius_sq).max(0.0).sqrt();

                if relative_pos.x * w.z - relative_pos.z * w.x > 0.0 {
                    // Left leg.
                    let direction = Vec3::new(
                        relative_pos.x * leg - relative_pos.z * combined_radius,
                        0.0,
                        relative_pos.x * combined_radius + relative_pos.z * leg,
                    ) / dist_sq;
                    let point = vel_a
                        + Vec3::new(
                            -(w.x * direction.x + w.z * direction.z) * direction.x,
                            0.0,
                            -(w.x * direction.x + w.z * direction.z) * direction.z,
                        ) * 0.5;
                    lines.push(OrcaLine { point, direction });
                } else {
                    // Right leg.
                    let direction = -Vec3::new(
                        relative_pos.x * leg + relative_pos.z * combined_radius,
                        0.0,
                        -relative_pos.x * combined_radius + relative_pos.z * leg,
                    ) / dist_sq;
                    let point = vel_a
                        + Vec3::new(
                            -(w.x * direction.x + w.z * direction.z) * direction.x,
                            0.0,
                            -(w.x * direction.x + w.z * direction.z) * direction.z,
                        ) * 0.5;
                    lines.push(OrcaLine { point, direction });
                }
            }
        } else {
            // Collision! Project on cut-off circle at time dt.
            let inv_dt = 1.0 / 0.016; // Assume ~60fps
            let w2 = relative_vel - relative_pos * inv_dt;
            let w2_len = (w2.x * w2.x + w2.z * w2.z).sqrt();
            if w2_len < 1e-6 {
                return;
            }
            let unit_w = Vec3::new(w2.x / w2_len, 0.0, w2.z / w2_len);
            let direction = Vec3::new(unit_w.z, 0.0, -unit_w.x);
            let point = vel_a + unit_w * (combined_radius * inv_dt - w2_len) * 0.5;
            lines.push(OrcaLine { point, direction });
        }
    }

    /// Add ORCA half-plane for a static obstacle segment.
    fn add_obstacle_orca_line(
        &self,
        agent_pos: Vec3,
        agent_radius: f32,
        obs_start: Vec3,
        obs_end: Vec3,
        lines: &mut Vec<OrcaLine>,
    ) {
        let closest = closest_point_on_segment(agent_pos, obs_start, obs_end);
        let diff = agent_pos - closest;
        let dist = (diff.x * diff.x + diff.z * diff.z).sqrt();

        if dist < 1e-6 {
            return;
        }

        let normal = Vec3::new(diff.x / dist, 0.0, diff.z / dist);
        let direction = Vec3::new(-normal.z, 0.0, normal.x);

        let penetration = agent_radius - dist;
        let push = if penetration > 0.0 {
            penetration / self.config.obstacle_time_horizon
        } else {
            0.0
        };

        let point = normal * push;
        lines.push(OrcaLine { point, direction });
    }

    /// Solve the 2D linear program: find velocity closest to `preferred`
    /// that satisfies all half-plane constraints.
    fn linear_program_2d(
        &self,
        lines: &[OrcaLine],
        preferred: Vec3,
        _num_obstacle_lines: usize,
    ) -> Vec3 {
        let max_speed = preferred.length().max(0.01);
        let mut result = preferred;

        for i in 0..lines.len() {
            // Check if current result violates constraint i.
            let det = det_2d(lines[i].direction, lines[i].point - result);
            if det > 0.0 {
                // Violated. Project onto line i.
                let _temp = result;
                result = self.project_onto_line(
                    &lines[i],
                    preferred,
                    max_speed,
                    i > 0,
                );

                // Verify against previous constraints.
                let mut valid = true;
                for j in 0..i {
                    let det_j = det_2d(lines[j].direction, lines[j].point - result);
                    if det_j > 1e-4 {
                        valid = false;
                        break;
                    }
                }

                if !valid {
                    // Fall back to finding a safe velocity via iterative projection.
                    result = self.linear_program_3d(lines, i, preferred, max_speed);
                }
            }
        }

        result
    }

    /// Project a velocity onto an ORCA line, clamped by max speed.
    fn project_onto_line(
        &self,
        line: &OrcaLine,
        preferred: Vec3,
        max_speed: f32,
        _clamp: bool,
    ) -> Vec3 {
        let t = dot_2d(line.direction, preferred - line.point);
        let result = line.point + line.direction * t;

        // Clamp to max speed circle.
        let speed = (result.x * result.x + result.z * result.z).sqrt();
        if speed > max_speed {
            result * (max_speed / speed)
        } else {
            result
        }
    }

    /// Fallback solver when 2D LP fails due to infeasibility.
    /// Projects velocity to satisfy the most violated constraints.
    fn linear_program_3d(
        &self,
        lines: &[OrcaLine],
        begin_line: usize,
        preferred: Vec3,
        max_speed: f32,
    ) -> Vec3 {
        let mut result = preferred;

        // Iteratively project to satisfy each constraint.
        for i in begin_line..lines.len() {
            let det = det_2d(lines[i].direction, lines[i].point - result);
            if det > 0.0 {
                // Project result onto this line.
                let t = dot_2d(lines[i].direction, result - lines[i].point);
                result = lines[i].point + lines[i].direction * t;
            }
        }

        // Clamp speed.
        let speed = (result.x * result.x + result.z * result.z).sqrt();
        if speed > max_speed {
            result * (max_speed / speed)
        } else {
            result
        }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &ObstacleAvoidanceConfig {
        &self.config
    }
}

/// 2D cross product (determinant) on XZ plane.
fn det_2d(a: Vec3, b: Vec3) -> f32 {
    a.x * b.z - a.z * b.x
}

/// 2D dot product on XZ plane.
fn dot_2d(a: Vec3, b: Vec3) -> f32 {
    a.x * b.x + a.z * b.z
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple quad navmesh (two triangles forming a square).
    fn make_simple_navmesh() -> NavMesh {
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 10.0),
            Vec3::new(0.0, 0.0, 10.0),
        ];
        let indices = vec![
            0, 1, 2, // Triangle 1
            0, 2, 3, // Triangle 2
        ];
        NavMeshBuilder::from_triangles(&vertices, &indices)
    }

    /// Create a larger navmesh with multiple quads in a row.
    fn make_corridor_navmesh() -> NavMesh {
        // Build the navmesh manually to ensure correct adjacency.
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),  // 0
            Vec3::new(5.0, 0.0, 0.0),  // 1
            Vec3::new(5.0, 0.0, 5.0),  // 2
            Vec3::new(0.0, 0.0, 5.0),  // 3
            Vec3::new(10.0, 0.0, 0.0), // 4
            Vec3::new(10.0, 0.0, 5.0), // 5
            Vec3::new(15.0, 0.0, 0.0), // 6
            Vec3::new(15.0, 0.0, 5.0), // 7
        ];

        // Build 3 quads as polygons directly (skip triangle merging).
        let poly0 = NavPoly {
            vertices: SmallVec::from_slice(&[0, 3, 2, 1]), // CCW on XZ
            adjacency: SmallVec::from_slice(&[u32::MAX, u32::MAX, 1, u32::MAX]),
            area_type: 0,
            centroid: Vec3::new(2.5, 0.0, 2.5),
            normal: Vec3::Y,
        };
        let poly1 = NavPoly {
            vertices: SmallVec::from_slice(&[1, 2, 5, 4]), // CCW on XZ
            adjacency: SmallVec::from_slice(&[0, u32::MAX, 2, u32::MAX]),
            area_type: 0,
            centroid: Vec3::new(7.5, 0.0, 2.5),
            normal: Vec3::Y,
        };
        let poly2 = NavPoly {
            vertices: SmallVec::from_slice(&[4, 5, 7, 6]), // CCW on XZ
            adjacency: SmallVec::from_slice(&[1, u32::MAX, u32::MAX, u32::MAX]),
            area_type: 0,
            centroid: Vec3::new(12.5, 0.0, 2.5),
            normal: Vec3::Y,
        };

        let mut nm = NavMesh {
            vertices,
            polygons: vec![poly0, poly1, poly2],
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::ZERO,
        };
        nm.recompute_bounds();
        nm
    }

    #[test]
    fn test_navmesh_creation() {
        let nm = make_simple_navmesh();
        assert!(nm.poly_count() > 0);
        assert_eq!(nm.vertex_count(), 4);
    }

    #[test]
    fn test_navmesh_bounds() {
        let nm = make_simple_navmesh();
        assert!((nm.bounds_min.x - 0.0).abs() < 0.01);
        assert!((nm.bounds_max.x - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_contains_point_2d() {
        let nm = make_simple_navmesh();
        assert!(nm.contains_point_2d(Vec3::new(5.0, 0.0, 5.0)));
        assert!(!nm.contains_point_2d(Vec3::new(15.0, 0.0, 5.0)));
    }

    #[test]
    fn test_find_nearest_polygon() {
        let nm = make_simple_navmesh();
        let result = nm.find_nearest_polygon(Vec3::new(5.0, 0.0, 5.0));
        assert!(result.is_some());
        let (_, nearest_point) = result.unwrap();
        assert!((nearest_point.x - 5.0).abs() < 0.1);
        assert!((nearest_point.z - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_find_nearest_point_outside() {
        let nm = make_simple_navmesh();
        let nearest = nm.find_nearest_point(Vec3::new(15.0, 0.0, 5.0));
        // Should clamp to the navmesh boundary.
        assert!(nearest.x <= 10.1);
    }

    #[test]
    fn test_navmesh_query_containing_polygon() {
        let nm = make_simple_navmesh();
        let query = NavMeshQuery::new(&nm);
        let poly = query.find_containing_polygon(Vec3::new(5.0, 0.0, 5.0));
        assert!(poly.is_some());
    }

    #[test]
    fn test_navmesh_query_path_corridor() {
        let nm = make_corridor_navmesh();
        let query = NavMeshQuery::new(&nm);

        // Find start and end polygons.
        let start_poly = query
            .find_containing_polygon(Vec3::new(2.0, 0.0, 2.0))
            .unwrap();
        let end_poly = query
            .find_containing_polygon(Vec3::new(12.0, 0.0, 2.0))
            .unwrap();

        let corridor = query.find_path_corridor(start_poly, end_poly);
        assert!(corridor.is_ok());
        let c = corridor.unwrap();
        assert!(c.len() >= 2); // At least start and end polygons.
    }

    #[test]
    fn test_navmesh_query_find_path() {
        let nm = make_corridor_navmesh();
        let query = NavMeshQuery::new(&nm);

        let path = query.find_path(Vec3::new(2.0, 0.0, 2.0), Vec3::new(12.0, 0.0, 2.0));
        assert!(path.is_some());
        let p = path.unwrap();
        assert!(p.len() >= 2);
        // Start should be close to (2, 0, 2).
        assert!((p[0].x - 2.0).abs() < 0.1);
        // End should be close to (12, 0, 2).
        assert!((p.last().unwrap().x - 12.0).abs() < 0.1);
    }

    #[test]
    fn test_navmesh_builder_from_triangles() {
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 10.0),
        ];
        let indices = vec![0, 1, 2];
        let nm = NavMeshBuilder::from_triangles(&vertices, &indices);
        assert_eq!(nm.poly_count(), 1);
        assert_eq!(nm.vertex_count(), 3);
    }

    #[test]
    fn test_navmesh_builder_slope_filter() {
        let mut builder = NavMeshBuilder::with_defaults();

        // Flat triangle (walkable).
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 10.0),
            // Steep triangle (not walkable at 45 degree limit).
            Vec3::new(20.0, 0.0, 0.0),
            Vec3::new(30.0, 20.0, 0.0), // Very steep
            Vec3::new(25.0, 0.0, 10.0),
        ];
        let indices = vec![0, 1, 2, 3, 4, 5];
        builder.add_geometry(&vertices, &indices);

        let nm = builder.build().unwrap();
        // Only the flat triangle should be included.
        assert_eq!(nm.poly_count(), 1);
    }

    #[test]
    fn test_navmesh_agent_basic() {
        let mut agent = NavMeshAgent::new(5.0, 0.5, 2.0);
        assert_eq!(agent.speed, 5.0);
        assert_eq!(agent.radius, 0.5);
        assert_eq!(agent.height, 2.0);
        assert!(!agent.is_navigating);

        agent.set_target(Vec3::new(10.0, 0.0, 0.0));
        assert!(agent.is_navigating);

        agent.stop();
        assert!(!agent.is_navigating);
    }

    #[test]
    fn test_navmesh_agent_movement() {
        let mut agent = NavMeshAgent::new(10.0, 0.5, 2.0);
        agent.position = Vec3::new(0.0, 0.0, 0.0);
        agent.path_waypoints = vec![Vec3::new(10.0, 0.0, 0.0)];
        agent.is_navigating = true;
        agent.current_waypoint_index = 0;

        // Move for 0.5 seconds. Agent should move toward the waypoint.
        let new_pos = agent.move_along_path(0.5);
        assert!(new_pos.x > 0.0);
        assert!(new_pos.x < 10.0);
    }

    #[test]
    fn test_navmesh_agent_arrival() {
        let mut agent = NavMeshAgent::new(100.0, 0.5, 2.0);
        agent.position = Vec3::new(9.9, 0.0, 0.0);
        agent.path_waypoints = vec![Vec3::new(10.0, 0.0, 0.0)];
        agent.is_navigating = true;
        agent.current_waypoint_index = 0;

        // Move a lot; should arrive.
        agent.move_along_path(1.0);
        assert!(!agent.is_navigating);
    }

    #[test]
    fn test_crowd_manager_basic() {
        let mut crowd = CrowdManager::new(100);
        let idx = crowd
            .add_agent(0, Vec3::new(0.0, 0.0, 0.0), 0.5, 5.0)
            .unwrap();
        assert_eq!(crowd.active_agent_count(), 1);

        crowd.set_desired_velocity(idx, Vec3::new(1.0, 0.0, 0.0));

        let nm = NavMesh::new();
        crowd.update(0.016, &nm);

        let pos = crowd.agent_position(idx).unwrap();
        assert!(pos.x > 0.0);
    }

    #[test]
    fn test_crowd_manager_two_agents() {
        let mut crowd = CrowdManager::new(100);

        // Two agents moving toward each other.
        let a = crowd
            .add_agent(0, Vec3::new(0.0, 0.0, 0.0), 1.0, 5.0)
            .unwrap();
        let b = crowd
            .add_agent(1, Vec3::new(10.0, 0.0, 0.0), 1.0, 5.0)
            .unwrap();

        crowd.set_desired_velocity(a, Vec3::new(5.0, 0.0, 0.0));
        crowd.set_desired_velocity(b, Vec3::new(-5.0, 0.0, 0.0));

        let nm = NavMesh::new();

        // Simulate several frames.
        for _ in 0..60 {
            crowd.update(0.016, &nm);
        }

        // Agents should not have passed through each other.
        let pos_a = crowd.agent_position(a).unwrap();
        let pos_b = crowd.agent_position(b).unwrap();
        let dist = (pos_b - pos_a).length();
        // They should maintain some distance due to ORCA.
        assert!(dist > 0.5, "Agents are too close: {}", dist);
    }

    #[test]
    fn test_obstacle_avoidance_no_neighbors() {
        let oa = ObstacleAvoidance::with_defaults();
        let result = oa.compute_velocity(
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            0.5,
            &[],
            &[],
        );
        // With no obstacles, should return desired velocity.
        assert!((result.x - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_closest_point_on_segment() {
        let p = Vec3::new(5.0, 0.0, 5.0);
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(10.0, 0.0, 0.0);
        let closest = closest_point_on_segment(p, a, b);
        assert!((closest.x - 5.0).abs() < 0.01);
        assert!((closest.z - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_closest_point_on_segment_end() {
        let p = Vec3::new(15.0, 0.0, 0.0);
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(10.0, 0.0, 0.0);
        let closest = closest_point_on_segment(p, a, b);
        assert!((closest.x - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_point_in_polygon() {
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 10.0),
            Vec3::new(0.0, 0.0, 10.0),
        ];
        let indices: Vec<u32> = vec![0, 1, 2, 3];
        assert!(point_in_polygon_xz(
            Vec3::new(5.0, 0.0, 5.0),
            &indices,
            &vertices
        ));
        assert!(!point_in_polygon_xz(
            Vec3::new(15.0, 0.0, 5.0),
            &indices,
            &vertices
        ));
    }

    #[test]
    fn test_triangle_normal() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 0.0, 1.0);
        let n = triangle_normal(a, b, c);
        // For a flat triangle on the XZ plane, normal should point up (Y).
        assert!((n.y - 1.0).abs() < 0.01 || (n.y + 1.0).abs() < 0.01);
    }

    #[test]
    fn test_polygon_centroid() {
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 4.0),
            Vec3::new(0.0, 0.0, 4.0),
        ];
        let indices: Vec<u32> = vec![0, 1, 2, 3];
        let c = polygon_centroid(&indices, &vertices);
        assert!((c.x - 2.0).abs() < 0.01);
        assert!((c.z - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_navmesh_same_polygon_path() {
        let nm = make_simple_navmesh();
        let query = NavMeshQuery::new(&nm);
        let path = query.find_path(Vec3::new(2.0, 0.0, 2.0), Vec3::new(8.0, 0.0, 8.0));
        assert!(path.is_some());
        let p = path.unwrap();
        assert_eq!(p.len(), 2); // Direct path, same polygon.
    }

    #[test]
    fn test_navmesh_empty() {
        let nm = NavMesh::new();
        let query = NavMeshQuery::new(&nm);
        assert!(query.find_containing_polygon(Vec3::ZERO).is_none());
        assert!(query.find_nearest_point(Vec3::ZERO).is_none());
    }

    #[test]
    fn test_navmesh_builder_empty_input() {
        let builder = NavMeshBuilder::with_defaults();
        let nm = builder.build().unwrap();
        assert_eq!(nm.poly_count(), 0);
    }

    #[test]
    fn test_crowd_remove_agent() {
        let mut crowd = CrowdManager::new(10);
        let idx = crowd
            .add_agent(0, Vec3::ZERO, 0.5, 5.0)
            .unwrap();
        assert_eq!(crowd.active_agent_count(), 1);
        crowd.remove_agent(idx);
        assert_eq!(crowd.active_agent_count(), 0);
    }

    #[test]
    fn test_funnel_algorithm_simple() {
        let portals = vec![
            Portal {
                left: Vec3::new(0.0, 0.0, 0.0),
                right: Vec3::new(0.0, 0.0, 0.0),
            },
            Portal {
                left: Vec3::new(3.0, 0.0, 1.0),
                right: Vec3::new(3.0, 0.0, -1.0),
            },
            Portal {
                left: Vec3::new(6.0, 0.0, 0.0),
                right: Vec3::new(6.0, 0.0, 0.0),
            },
        ];
        let path =
            funnel_algorithm(&portals, Vec3::new(0.0, 0.0, 0.0), Vec3::new(6.0, 0.0, 0.0));
        assert!(path.len() >= 2);
        assert!((path[0].x - 0.0).abs() < 0.01);
        assert!((path.last().unwrap().x - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_shared_edge() {
        let poly = NavPoly {
            vertices: SmallVec::from_slice(&[0, 1, 2]),
            adjacency: SmallVec::from_slice(&[1, u32::MAX, u32::MAX]),
            area_type: 0,
            centroid: Vec3::ZERO,
            normal: Vec3::Y,
        };
        let edge = poly.shared_edge_with(1);
        assert!(edge.is_some());
        assert_eq!(edge.unwrap(), (0, 1));

        let edge_none = poly.shared_edge_with(5);
        assert!(edge_none.is_none());
    }
}
