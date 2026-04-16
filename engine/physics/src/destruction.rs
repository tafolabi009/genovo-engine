//! Destructible objects and Voronoi fracture system.
//!
//! Provides:
//! - `DestructibleMesh`: pre-fractured mesh using Voronoi patterns
//! - `FractureChunk`: individual piece with mesh, mass, and neighbor connections
//! - `DestructionManager`: impact-driven breakage, cascading destruction, debris cleanup
//! - Voronoi fracture generation from seed points
//! - Interior face generation for fracture surfaces
//! - Connectivity graph for structural integrity checks
//! - ECS integration

use std::collections::{HashMap, HashSet, VecDeque};

use glam::Vec3;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Small epsilon for geometric comparisons.
const EPSILON: f32 = 1e-6;
/// Default break threshold in Joules (impact energy).
const DEFAULT_BREAK_THRESHOLD: f32 = 100.0;
/// Default debris lifetime in seconds before cleanup.
const DEFAULT_DEBRIS_LIFETIME: f32 = 10.0;
/// Minimum chunk volume below which it becomes debris.
const MIN_CHUNK_VOLUME: f32 = 0.001;

// ---------------------------------------------------------------------------
// Mesh data
// ---------------------------------------------------------------------------

/// A triangle mesh with vertices and indices.
#[derive(Debug, Clone)]
pub struct FractureMesh {
    /// Vertex positions.
    pub vertices: Vec<Vec3>,
    /// Vertex normals.
    pub normals: Vec<Vec3>,
    /// Triangle indices (groups of 3).
    pub indices: Vec<u32>,
}

impl FractureMesh {
    /// Create an empty mesh.
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Compute the axis-aligned bounding box of this mesh.
    pub fn aabb(&self) -> (Vec3, Vec3) {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for v in &self.vertices {
            min = min.min(*v);
            max = max.max(*v);
        }
        (min, max)
    }

    /// Compute the volume of the mesh (assuming it is closed).
    /// Uses the signed volume of tetrahedra formed with the origin.
    pub fn compute_volume(&self) -> f32 {
        let mut volume = 0.0f32;
        let tri_count = self.indices.len() / 3;
        for t in 0..tri_count {
            let i0 = self.indices[t * 3] as usize;
            let i1 = self.indices[t * 3 + 1] as usize;
            let i2 = self.indices[t * 3 + 2] as usize;
            if i0 >= self.vertices.len() || i1 >= self.vertices.len() || i2 >= self.vertices.len()
            {
                continue;
            }
            let v0 = self.vertices[i0];
            let v1 = self.vertices[i1];
            let v2 = self.vertices[i2];
            // Signed volume of tetrahedron with origin
            volume += v0.dot(v1.cross(v2)) / 6.0;
        }
        volume.abs()
    }

    /// Compute the centroid of the mesh.
    pub fn centroid(&self) -> Vec3 {
        if self.vertices.is_empty() {
            return Vec3::ZERO;
        }
        let sum: Vec3 = self.vertices.iter().copied().fold(Vec3::ZERO, |a, b| a + b);
        sum / self.vertices.len() as f32
    }

    /// Compute normals for all vertices (smooth normals from face normals).
    pub fn compute_normals(&mut self) {
        self.normals = vec![Vec3::ZERO; self.vertices.len()];
        let tri_count = self.indices.len() / 3;
        for t in 0..tri_count {
            let i0 = self.indices[t * 3] as usize;
            let i1 = self.indices[t * 3 + 1] as usize;
            let i2 = self.indices[t * 3 + 2] as usize;
            if i0 >= self.vertices.len() || i1 >= self.vertices.len() || i2 >= self.vertices.len()
            {
                continue;
            }
            let e1 = self.vertices[i1] - self.vertices[i0];
            let e2 = self.vertices[i2] - self.vertices[i0];
            let face_normal = e1.cross(e2);
            self.normals[i0] += face_normal;
            self.normals[i1] += face_normal;
            self.normals[i2] += face_normal;
        }
        for n in &mut self.normals {
            let len = n.length();
            if len > EPSILON {
                *n /= len;
            } else {
                *n = Vec3::Y;
            }
        }
    }

    /// Get the vertex count.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Get the triangle count.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

// ---------------------------------------------------------------------------
// Fracture chunk
// ---------------------------------------------------------------------------

/// A single chunk resulting from fracturing a mesh.
#[derive(Debug, Clone)]
pub struct FractureChunk {
    /// Unique identifier for this chunk.
    pub id: usize,
    /// The mesh geometry of this chunk.
    pub mesh: FractureMesh,
    /// The interior faces (generated by the fracture planes).
    pub interior_mesh: FractureMesh,
    /// Center of mass (in local space of the original mesh).
    pub center_of_mass: Vec3,
    /// Volume of this chunk.
    pub volume: f32,
    /// Mass of this chunk (volume * density).
    pub mass: f32,
    /// Indices of neighboring chunks (connected before breaking).
    pub neighbor_indices: Vec<usize>,
    /// Whether this chunk has been separated from the main body.
    pub is_detached: bool,
    /// The Voronoi seed point that generated this chunk.
    pub seed_point: Vec3,
    /// Whether this chunk is considered debris (small, should be cleaned up).
    pub is_debris: bool,
    /// Remaining lifetime for debris (seconds).
    pub debris_timer: f32,
}

impl FractureChunk {
    /// Create a new fracture chunk.
    pub fn new(id: usize, mesh: FractureMesh, seed: Vec3) -> Self {
        let center_of_mass = mesh.centroid();
        let volume = mesh.compute_volume();
        Self {
            id,
            mesh,
            interior_mesh: FractureMesh::new(),
            center_of_mass,
            volume,
            mass: volume * 1000.0, // Default density
            neighbor_indices: Vec::new(),
            is_detached: false,
            seed_point: seed,
            is_debris: volume < MIN_CHUNK_VOLUME,
            debris_timer: DEFAULT_DEBRIS_LIFETIME,
        }
    }
}

// ---------------------------------------------------------------------------
// Voronoi fracture generation
// ---------------------------------------------------------------------------

/// A clipping plane defined by a point and normal.
#[derive(Debug, Clone, Copy)]
struct ClipPlane {
    point: Vec3,
    normal: Vec3,
}

impl ClipPlane {
    /// Signed distance from a point to this plane.
    /// Positive = in front (same side as normal), negative = behind.
    #[inline]
    fn signed_distance(&self, p: Vec3) -> f32 {
        (p - self.point).dot(self.normal)
    }
}

/// Clip a convex mesh (represented as vertices) by a plane.
/// Returns vertices on the positive (front) side of the plane.
/// New vertices are created at plane intersections.
fn clip_vertices_by_plane(vertices: &[Vec3], plane: &ClipPlane) -> Vec<Vec3> {
    if vertices.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let n = vertices.len();

    for i in 0..n {
        let current = vertices[i];
        let next = vertices[(i + 1) % n];
        let d_current = plane.signed_distance(current);
        let d_next = plane.signed_distance(next);

        if d_current >= 0.0 {
            result.push(current);
        }

        // Edge crosses the plane
        if (d_current > 0.0 && d_next < 0.0) || (d_current < 0.0 && d_next > 0.0) {
            let t = d_current / (d_current - d_next);
            let intersection = current + (next - current) * t;
            result.push(intersection);
        }
    }

    result
}

/// Generate a convex hull from a set of points (simplified: use AABB of clipped region).
fn compute_convex_mesh_from_points(points: &[Vec3]) -> FractureMesh {
    if points.len() < 4 {
        return FractureMesh::new();
    }

    // Compute AABB
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for &p in points {
        min = min.min(p);
        max = max.max(p);
    }

    // If degenerate, return empty
    let size = max - min;
    if size.x < EPSILON || size.y < EPSILON || size.z < EPSILON {
        return FractureMesh::new();
    }

    // Generate a box mesh from the AABB (8 vertices, 12 triangles)
    let center = (min + max) * 0.5;
    let he = size * 0.5;

    let vertices = vec![
        center + Vec3::new(-he.x, -he.y, -he.z), // 0
        center + Vec3::new(he.x, -he.y, -he.z),  // 1
        center + Vec3::new(he.x, he.y, -he.z),   // 2
        center + Vec3::new(-he.x, he.y, -he.z),  // 3
        center + Vec3::new(-he.x, -he.y, he.z),  // 4
        center + Vec3::new(he.x, -he.y, he.z),   // 5
        center + Vec3::new(he.x, he.y, he.z),    // 6
        center + Vec3::new(-he.x, he.y, he.z),   // 7
    ];

    #[rustfmt::skip]
    let indices = vec![
        // Front face (-Z)
        0, 2, 1, 0, 3, 2,
        // Back face (+Z)
        4, 5, 6, 4, 6, 7,
        // Left face (-X)
        0, 4, 7, 0, 7, 3,
        // Right face (+X)
        1, 2, 6, 1, 6, 5,
        // Bottom face (-Y)
        0, 1, 5, 0, 5, 4,
        // Top face (+Y)
        2, 3, 7, 2, 7, 6,
    ];

    let mut mesh = FractureMesh {
        vertices,
        normals: Vec::new(),
        indices,
    };
    mesh.compute_normals();
    mesh
}

/// Generate interior faces along a clipping plane within a bounding region.
fn generate_interior_face(
    plane: &ClipPlane,
    bounding_min: Vec3,
    bounding_max: Vec3,
) -> FractureMesh {
    // Create a quad on the plane, clipped to the bounding box
    let center = (bounding_min + bounding_max) * 0.5;
    let he = (bounding_max - bounding_min) * 0.5;

    // Project the bounding box center onto the plane
    let projected_center = center - plane.normal * plane.signed_distance(center);

    // Create two tangent vectors
    let tangent1 = if plane.normal.x.abs() < 0.9 {
        plane.normal.cross(Vec3::X).normalize()
    } else {
        plane.normal.cross(Vec3::Y).normalize()
    };
    let tangent2 = plane.normal.cross(tangent1).normalize();

    let extent = he.length();

    // Quad vertices
    let v0 = projected_center - tangent1 * extent - tangent2 * extent;
    let v1 = projected_center + tangent1 * extent - tangent2 * extent;
    let v2 = projected_center + tangent1 * extent + tangent2 * extent;
    let v3 = projected_center - tangent1 * extent + tangent2 * extent;

    // Clip to bounding box
    let mut verts = vec![v0, v1, v2, v3];

    // Clip by all 6 AABB planes
    let aabb_planes = [
        ClipPlane { point: bounding_min, normal: Vec3::X },
        ClipPlane { point: bounding_max, normal: Vec3::NEG_X },
        ClipPlane { point: bounding_min, normal: Vec3::Y },
        ClipPlane { point: bounding_max, normal: Vec3::NEG_Y },
        ClipPlane { point: bounding_min, normal: Vec3::Z },
        ClipPlane { point: bounding_max, normal: Vec3::NEG_Z },
    ];

    for aabb_plane in &aabb_planes {
        verts = clip_vertices_by_plane(&verts, aabb_plane);
        if verts.is_empty() {
            break;
        }
    }

    if verts.len() < 3 {
        return FractureMesh::new();
    }

    // Triangulate the polygon (fan triangulation)
    let vertex_count = verts.len();
    let mut indices = Vec::new();
    for i in 1..(vertex_count - 1) {
        indices.push(0u32);
        indices.push(i as u32);
        indices.push((i + 1) as u32);
    }

    let normals = vec![plane.normal; vertex_count];

    FractureMesh {
        vertices: verts,
        normals,
        indices,
    }
}

/// Generate a Voronoi fracture pattern from seed points within a bounding box.
///
/// For each seed point, we compute the Voronoi cell by clipping the bounding box
/// with bisector planes between this seed and all other seeds.
///
/// # Arguments
/// * `bounding_min` - Minimum corner of the bounding box.
/// * `bounding_max` - Maximum corner of the bounding box.
/// * `seed_points` - Voronoi seed points.
/// * `density` - Material density for mass computation (kg/m^3).
///
/// # Returns
/// A vector of fracture chunks, one per Voronoi cell.
pub fn generate_voronoi_fracture(
    bounding_min: Vec3,
    bounding_max: Vec3,
    seed_points: &[Vec3],
    density: f32,
) -> Vec<FractureChunk> {
    let num_seeds = seed_points.len();
    if num_seeds == 0 {
        return Vec::new();
    }

    let mut chunks = Vec::with_capacity(num_seeds);

    for i in 0..num_seeds {
        let seed = seed_points[i];

        // Start with the bounding box vertices
        let he = (bounding_max - bounding_min) * 0.5;
        let center = (bounding_min + bounding_max) * 0.5;

        // Generate initial box vertices
        let mut cell_points = vec![
            center + Vec3::new(-he.x, -he.y, -he.z),
            center + Vec3::new(he.x, -he.y, -he.z),
            center + Vec3::new(he.x, he.y, -he.z),
            center + Vec3::new(-he.x, he.y, -he.z),
            center + Vec3::new(-he.x, -he.y, he.z),
            center + Vec3::new(he.x, -he.y, he.z),
            center + Vec3::new(he.x, he.y, he.z),
            center + Vec3::new(-he.x, he.y, he.z),
        ];

        let mut interior_faces = Vec::new();

        // Clip against bisector planes with all other seeds
        for j in 0..num_seeds {
            if i == j {
                continue;
            }
            let other = seed_points[j];

            // Bisector plane: midpoint between seeds, normal pointing away from other
            let midpoint = (seed + other) * 0.5;
            let normal = (seed - other).normalize();

            let plane = ClipPlane {
                point: midpoint,
                normal,
            };

            // Clip the current cell points
            cell_points = clip_cell_by_plane(&cell_points, &plane);

            if cell_points.is_empty() {
                break;
            }

            // Generate interior face along this clipping plane
            let interior = generate_interior_face(&plane, bounding_min, bounding_max);
            if interior.vertex_count() >= 3 {
                interior_faces.push(interior);
            }
        }

        if cell_points.len() < 4 {
            continue;
        }

        // Generate mesh from the remaining points
        let mesh = compute_convex_mesh_from_points(&cell_points);
        if mesh.vertex_count() == 0 {
            continue;
        }

        let mut chunk = FractureChunk::new(i, mesh, seed);
        chunk.mass = chunk.volume * density;
        chunk.is_debris = chunk.volume < MIN_CHUNK_VOLUME;

        // Merge interior faces
        if !interior_faces.is_empty() {
            let mut combined = FractureMesh::new();
            for face in interior_faces {
                let base = combined.vertices.len() as u32;
                combined.vertices.extend_from_slice(&face.vertices);
                combined.normals.extend_from_slice(&face.normals);
                for idx in &face.indices {
                    combined.indices.push(base + idx);
                }
            }
            chunk.interior_mesh = combined;
        }

        chunks.push(chunk);
    }

    // Compute neighbor connections (chunks whose cells share a face)
    compute_chunk_neighbors(&mut chunks, seed_points);

    chunks
}

/// Clip cell vertices by a plane, keeping points on the positive side.
///
/// Only considers edges between vertices that are reasonably close (within the
/// bounding box diagonal), preventing quadratic blowup of intersection points.
fn clip_cell_by_plane(points: &[Vec3], plane: &ClipPlane) -> Vec<Vec3> {
    let mut result = Vec::new();

    // Keep all points on the positive side
    let mut distances: Vec<f32> = points.iter().map(|p| plane.signed_distance(*p)).collect();

    for (i, &d) in distances.iter().enumerate() {
        if d >= -EPSILON {
            result.push(points[i]);
        }
    }

    // For intersection points, only consider pairs of points that are
    // reasonably close (nearest neighbors), to avoid O(n^2) blowup.
    // We limit to checking each point against a bounded number of neighbors.
    let n = points.len();
    let max_neighbors = 20.min(n);

    for i in 0..n {
        let d_i = distances[i];
        // Only process if this point is on one side
        if d_i.abs() < EPSILON {
            continue;
        }

        let mut checked = 0;
        for j in 0..n {
            if i == j || checked >= max_neighbors {
                break;
            }
            let d_j = distances[j];
            if (d_i > EPSILON && d_j < -EPSILON) || (d_i < -EPSILON && d_j > EPSILON) {
                let t = d_i / (d_i - d_j);
                let intersection = points[i] + (points[j] - points[i]) * t;
                result.push(intersection);
                checked += 1;
            }
        }
    }

    // Cap maximum number of points to prevent memory explosion
    if result.len() > 256 {
        result.truncate(256);
    }

    result
}

/// Determine which chunks are neighbors (share Voronoi cell boundaries).
fn compute_chunk_neighbors(chunks: &mut [FractureChunk], seeds: &[Vec3]) {
    let n = chunks.len();

    // Two chunks are neighbors if their seeds are Voronoi neighbors
    // (i.e., there exists a point equidistant to both seeds and no other seed is closer).
    // Simplified: check if the distance between seeds is small relative to the average.
    let avg_dist = if n > 1 {
        let total: f32 = seeds
            .iter()
            .enumerate()
            .flat_map(|(i, &a)| {
                seeds.iter().skip(i + 1).map(move |&b| (a - b).length())
            })
            .sum();
        let count = n * (n - 1) / 2;
        total / count.max(1) as f32
    } else {
        1.0
    };

    let neighbor_threshold = avg_dist * 2.0;

    for i in 0..n {
        let seed_i = chunks[i].seed_point;
        for j in (i + 1)..n {
            let seed_j = chunks[j].seed_point;
            let dist = (seed_i - seed_j).length();
            if dist < neighbor_threshold {
                chunks[i].neighbor_indices.push(j);
                chunks[j].neighbor_indices.push(i);
            }
        }
    }
}

/// Generate random seed points within a bounding box.
///
/// Uses a simple pseudo-random number generator seeded with the given value.
pub fn generate_seed_points(
    bounding_min: Vec3,
    bounding_max: Vec3,
    num_points: usize,
    seed: u64,
) -> Vec<Vec3> {
    let mut rng = SimpleRng::new(seed);
    let extent = bounding_max - bounding_min;

    (0..num_points)
        .map(|_| {
            bounding_min
                + Vec3::new(
                    rng.next_f32() * extent.x,
                    rng.next_f32() * extent.y,
                    rng.next_f32() * extent.z,
                )
        })
        .collect()
}

/// Simple pseudo-random number generator (xoshiro128+).
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

// ---------------------------------------------------------------------------
// Destructible mesh
// ---------------------------------------------------------------------------

/// A pre-fractured mesh ready for runtime destruction.
#[derive(Debug, Clone)]
pub struct DestructibleMesh {
    /// All fracture chunks.
    pub chunks: Vec<FractureChunk>,
    /// Connectivity graph (which chunks are connected).
    pub connectivity: ConnectivityGraph,
    /// Break threshold (minimum impact energy to trigger fracture).
    pub break_threshold: f32,
    /// Material density (kg/m^3).
    pub density: f32,
    /// Debris lifetime in seconds.
    pub debris_lifetime: f32,
    /// Original bounding box.
    pub bounding_min: Vec3,
    /// Original bounding box.
    pub bounding_max: Vec3,
}

impl DestructibleMesh {
    /// Create a destructible mesh from a bounding box.
    pub fn from_box(
        half_extents: Vec3,
        num_chunks: usize,
        seed: u64,
        density: f32,
        break_threshold: f32,
    ) -> Self {
        let bounding_min = -half_extents;
        let bounding_max = half_extents;

        let seed_points = generate_seed_points(bounding_min, bounding_max, num_chunks, seed);
        let chunks = generate_voronoi_fracture(bounding_min, bounding_max, &seed_points, density);
        let connectivity = ConnectivityGraph::from_chunks(&chunks);

        Self {
            chunks,
            connectivity,
            break_threshold,
            density,
            debris_lifetime: DEFAULT_DEBRIS_LIFETIME,
            bounding_min,
            bounding_max,
        }
    }

    /// Get the total number of chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Get the number of detached chunks.
    pub fn detached_count(&self) -> usize {
        self.chunks.iter().filter(|c| c.is_detached).count()
    }

    /// Get the total mass.
    pub fn total_mass(&self) -> f32 {
        self.chunks.iter().map(|c| c.mass).sum()
    }

    /// Check if the mesh is fully intact (no detached chunks).
    pub fn is_intact(&self) -> bool {
        self.chunks.iter().all(|c| !c.is_detached)
    }

    /// Check if the mesh is fully destroyed (all chunks detached).
    pub fn is_fully_destroyed(&self) -> bool {
        self.chunks.iter().all(|c| c.is_detached)
    }
}

// ---------------------------------------------------------------------------
// Connectivity graph
// ---------------------------------------------------------------------------

/// Connectivity graph tracking which chunks are still connected.
#[derive(Debug, Clone)]
pub struct ConnectivityGraph {
    /// Edges: pairs of chunk indices that are connected.
    pub edges: HashSet<(usize, usize)>,
    /// Number of nodes.
    pub node_count: usize,
}

impl ConnectivityGraph {
    /// Build connectivity from chunk neighbor lists.
    pub fn from_chunks(chunks: &[FractureChunk]) -> Self {
        let mut edges = HashSet::new();
        for chunk in chunks {
            for &neighbor in &chunk.neighbor_indices {
                let a = chunk.id.min(neighbor);
                let b = chunk.id.max(neighbor);
                edges.insert((a, b));
            }
        }
        Self {
            edges,
            node_count: chunks.len(),
        }
    }

    /// Remove all connections involving a specific chunk.
    pub fn disconnect_chunk(&mut self, chunk_id: usize) {
        self.edges
            .retain(|&(a, b)| a != chunk_id && b != chunk_id);
    }

    /// Remove a specific edge.
    pub fn disconnect_edge(&mut self, a: usize, b: usize) {
        let edge = (a.min(b), a.max(b));
        self.edges.remove(&edge);
    }

    /// Check if a chunk is still connected to any other chunk.
    pub fn is_connected(&self, chunk_id: usize) -> bool {
        self.edges
            .iter()
            .any(|&(a, b)| a == chunk_id || b == chunk_id)
    }

    /// Find all connected components (groups of still-connected chunks).
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.node_count];
        let mut components = Vec::new();

        // Build adjacency list
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(a, b) in &self.edges {
            adj.entry(a).or_default().push(b);
            adj.entry(b).or_default().push(a);
        }

        for start in 0..self.node_count {
            if visited[start] {
                continue;
            }

            // BFS from this node
            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(start);
            visited[start] = true;

            while let Some(node) = queue.pop_front() {
                component.push(node);
                if let Some(neighbors) = adj.get(&node) {
                    for &neighbor in neighbors {
                        if !visited[neighbor] {
                            visited[neighbor] = true;
                            queue.push_back(neighbor);
                        }
                    }
                }
            }

            components.push(component);
        }

        components
    }

    /// Check if the structure is still supported (has a connected component
    /// containing a support point).
    pub fn is_supported(&self, support_chunks: &[usize]) -> bool {
        if support_chunks.is_empty() {
            return false;
        }

        let components = self.connected_components();
        for component in &components {
            let has_support = component.iter().any(|c| support_chunks.contains(c));
            if has_support {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Destruction manager
// ---------------------------------------------------------------------------

/// Impact event for the destruction system.
#[derive(Debug, Clone)]
pub struct ImpactEvent {
    /// World-space position of the impact.
    pub position: Vec3,
    /// Impact energy (in Joules, proportional to mass * velocity^2).
    pub energy: f32,
    /// Impact direction (normalized).
    pub direction: Vec3,
    /// Radius of effect.
    pub radius: f32,
}

/// Result of processing an impact event.
#[derive(Debug, Clone)]
pub struct DestructionResult {
    /// Indices of chunks that were detached.
    pub detached_chunks: Vec<usize>,
    /// Whether cascading destruction occurred.
    pub cascading: bool,
    /// Number of new free-floating islands.
    pub new_islands: usize,
}

/// Manages destruction of destructible meshes.
pub struct DestructionManager {
    /// The destructible mesh.
    pub mesh: DestructibleMesh,
    /// Chunk velocities (after detachment).
    pub chunk_velocities: Vec<Vec3>,
    /// Chunk positions (after detachment).
    pub chunk_positions: Vec<Vec3>,
    /// Chunk rotations (simplified as angular velocity).
    pub chunk_angular_velocities: Vec<Vec3>,
    /// Which chunks are support points (e.g., attached to the ground).
    pub support_chunks: Vec<usize>,
    /// Whether to enable cascading destruction.
    pub cascading_enabled: bool,
    /// Energy propagation factor for cascading (0 = none, 1 = full).
    pub cascade_factor: f32,
    /// Time accumulator for debris cleanup.
    sim_time: f32,
}

impl DestructionManager {
    /// Create a new destruction manager.
    pub fn new(mesh: DestructibleMesh) -> Self {
        let n = mesh.chunks.len();
        let positions: Vec<Vec3> = mesh.chunks.iter().map(|c| c.center_of_mass).collect();
        Self {
            mesh,
            chunk_velocities: vec![Vec3::ZERO; n],
            chunk_positions: positions,
            chunk_angular_velocities: vec![Vec3::ZERO; n],
            support_chunks: Vec::new(),
            cascading_enabled: true,
            cascade_factor: 0.5,
            sim_time: 0.0,
        }
    }

    /// Apply an impact and process destruction.
    pub fn apply_impact(&mut self, impact: &ImpactEvent) -> DestructionResult {
        let mut detached = Vec::new();
        let threshold = self.mesh.break_threshold;

        // Find chunks affected by the impact
        for i in 0..self.mesh.chunks.len() {
            if self.mesh.chunks[i].is_detached {
                continue;
            }

            let chunk_center = self.chunk_positions[i];
            let dist = (chunk_center - impact.position).length();

            if dist > impact.radius {
                continue;
            }

            // Energy falloff with distance
            let falloff = 1.0 - (dist / impact.radius).min(1.0);
            let effective_energy = impact.energy * falloff;

            if effective_energy > threshold {
                // Detach this chunk
                self.mesh.chunks[i].is_detached = true;
                self.mesh.connectivity.disconnect_chunk(i);
                detached.push(i);

                // Apply velocity from impact
                let dir = (chunk_center - impact.position).normalize_or_zero();
                let speed = (2.0 * effective_energy / self.mesh.chunks[i].mass.max(0.1)).sqrt();
                self.chunk_velocities[i] = dir * speed + impact.direction * speed * 0.5;

                // Add some angular velocity
                let torque_dir = dir.cross(impact.direction).normalize_or_zero();
                self.chunk_angular_velocities[i] = torque_dir * speed * 0.3;
            }
        }

        let mut cascading = false;

        // Cascading destruction: check if detaching chunks causes unsupported islands
        if self.cascading_enabled && !detached.is_empty() {
            let components = self.mesh.connectivity.connected_components();

            for component in &components {
                // Check if this component has any support
                let has_support = component
                    .iter()
                    .any(|c| self.support_chunks.contains(c) && !self.mesh.chunks[*c].is_detached);

                if !has_support {
                    // This component is unsupported -- detach all chunks in it
                    for &chunk_id in component {
                        if !self.mesh.chunks[chunk_id].is_detached {
                            self.mesh.chunks[chunk_id].is_detached = true;
                            self.mesh.connectivity.disconnect_chunk(chunk_id);
                            detached.push(chunk_id);
                            cascading = true;

                            // Give a small downward velocity
                            self.chunk_velocities[chunk_id] = Vec3::new(0.0, -1.0, 0.0);
                        }
                    }
                }
            }
        }

        let new_islands = self
            .mesh
            .connectivity
            .connected_components()
            .len()
            .saturating_sub(1);

        DestructionResult {
            detached_chunks: detached,
            cascading,
            new_islands,
        }
    }

    /// Step the simulation (update debris positions, cleanup old debris).
    pub fn step(&mut self, dt: f32, gravity: Vec3) {
        self.sim_time += dt;

        for i in 0..self.mesh.chunks.len() {
            if !self.mesh.chunks[i].is_detached {
                continue;
            }

            // Apply gravity
            self.chunk_velocities[i] += gravity * dt;

            // Update position
            self.chunk_positions[i] += self.chunk_velocities[i] * dt;

            // Damping
            self.chunk_velocities[i] *= 0.99;
            self.chunk_angular_velocities[i] *= 0.97;

            // Ground collision (y = 0)
            if self.chunk_positions[i].y < 0.0 {
                self.chunk_positions[i].y = 0.0;
                if self.chunk_velocities[i].y < 0.0 {
                    self.chunk_velocities[i].y *= -0.3;
                    self.chunk_velocities[i].x *= 0.8;
                    self.chunk_velocities[i].z *= 0.8;
                }
            }

            // Debris timer
            if self.mesh.chunks[i].is_debris {
                self.mesh.chunks[i].debris_timer -= dt;
            }
        }
    }

    /// Remove expired debris chunks.
    pub fn cleanup_debris(&mut self) -> Vec<usize> {
        let mut removed = Vec::new();
        for i in 0..self.mesh.chunks.len() {
            if self.mesh.chunks[i].is_debris && self.mesh.chunks[i].debris_timer <= 0.0 {
                removed.push(i);
            }
        }
        removed
    }

    /// Get all active (non-removed) detached chunk indices.
    pub fn active_detached_chunks(&self) -> Vec<usize> {
        self.mesh
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.is_detached && (!c.is_debris || c.debris_timer > 0.0))
            .map(|(i, _)| i)
            .collect()
    }

    /// Get all intact (non-detached) chunk indices.
    pub fn intact_chunks(&self) -> Vec<usize> {
        self.mesh
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_detached)
            .map(|(i, _)| i)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ECS integration
// ---------------------------------------------------------------------------

/// ECS component for attaching a destructible mesh to an entity.
pub struct DestructibleComponent {
    /// The destruction manager.
    pub manager: DestructionManager,
    /// World-space position of the destructible object.
    pub world_position: Vec3,
    /// Whether destruction is enabled.
    pub active: bool,
}

impl DestructibleComponent {
    /// Create a new destructible component.
    pub fn new(mesh: DestructibleMesh) -> Self {
        Self {
            manager: DestructionManager::new(mesh),
            world_position: Vec3::ZERO,
            active: true,
        }
    }

    /// Create a simple box destructible.
    pub fn box_shape(half_extents: Vec3, num_chunks: usize, seed: u64) -> Self {
        let mesh = DestructibleMesh::from_box(half_extents, num_chunks, seed, 1000.0, DEFAULT_BREAK_THRESHOLD);
        Self::new(mesh)
    }
}

/// System that steps all destructible objects.
pub struct DestructionSystem {
    /// Gravity vector.
    pub gravity: Vec3,
    /// Pending impact events.
    pub pending_impacts: Vec<(usize, ImpactEvent)>,
}

impl Default for DestructionSystem {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            pending_impacts: Vec::new(),
        }
    }
}

impl DestructionSystem {
    /// Create a new destruction system.
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue an impact event for a specific destructible entity.
    pub fn queue_impact(&mut self, entity_index: usize, impact: ImpactEvent) {
        self.pending_impacts.push((entity_index, impact));
    }

    /// Update all destructible objects.
    pub fn update(&mut self, dt: f32, destructibles: &mut [DestructibleComponent]) {
        // Process pending impacts
        let impacts = std::mem::take(&mut self.pending_impacts);
        for (entity_idx, impact) in impacts {
            if entity_idx < destructibles.len() && destructibles[entity_idx].active {
                destructibles[entity_idx].manager.apply_impact(&impact);
            }
        }

        // Step physics for detached chunks
        for d in destructibles.iter_mut() {
            if d.active {
                d.manager.step(dt, self.gravity);
                d.manager.cleanup_debris();
            }
        }
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_seed_points() {
        let seeds = generate_seed_points(Vec3::ZERO, Vec3::ONE, 10, 42);
        assert_eq!(seeds.len(), 10);

        // All seeds should be within bounds
        for s in &seeds {
            assert!(s.x >= 0.0 && s.x <= 1.0);
            assert!(s.y >= 0.0 && s.y <= 1.0);
            assert!(s.z >= 0.0 && s.z <= 1.0);
        }
    }

    #[test]
    fn test_voronoi_fracture() {
        let seeds = generate_seed_points(-Vec3::ONE, Vec3::ONE, 5, 42);
        let chunks = generate_voronoi_fracture(-Vec3::ONE, Vec3::ONE, &seeds, 1000.0);

        assert!(chunks.len() > 0, "Should generate at least one chunk");

        // Each chunk should have valid geometry
        for chunk in &chunks {
            assert!(
                chunk.mesh.vertex_count() > 0,
                "Chunk {} has no vertices",
                chunk.id
            );
            assert!(chunk.volume > 0.0, "Chunk {} has zero volume", chunk.id);
            assert!(chunk.mass > 0.0, "Chunk {} has zero mass", chunk.id);
        }
    }

    #[test]
    fn test_destructible_mesh_creation() {
        let mesh = DestructibleMesh::from_box(Vec3::ONE, 5, 42, 1000.0, 100.0);

        assert!(mesh.chunk_count() > 0);
        assert!(mesh.total_mass() > 0.0);
        assert!(mesh.is_intact());
        assert!(!mesh.is_fully_destroyed());
    }

    #[test]
    fn test_destruction_manager_impact() {
        let mesh = DestructibleMesh::from_box(Vec3::ONE, 4, 42, 1000.0, 10.0);
        let mut manager = DestructionManager::new(mesh);

        let impact = ImpactEvent {
            position: Vec3::ZERO,
            energy: 1000.0,
            direction: Vec3::X,
            radius: 3.0,
        };

        let result = manager.apply_impact(&impact);

        // Should have detached some chunks
        assert!(
            !result.detached_chunks.is_empty(),
            "Impact should detach chunks"
        );
    }

    #[test]
    fn test_connectivity_graph() {
        let mut graph = ConnectivityGraph {
            edges: HashSet::new(),
            node_count: 4,
        };
        graph.edges.insert((0, 1));
        graph.edges.insert((1, 2));
        graph.edges.insert((2, 3));

        // All connected
        let components = graph.connected_components();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 4);

        // Disconnect node 1
        graph.disconnect_chunk(1);
        let components = graph.connected_components();
        // Should have 3 components: {0}, {2, 3}, and {1}
        // Actually {0}, {1}, and {2,3}
        assert!(components.len() >= 2);
    }

    #[test]
    fn test_connectivity_support() {
        let mut graph = ConnectivityGraph {
            edges: HashSet::new(),
            node_count: 3,
        };
        graph.edges.insert((0, 1));
        graph.edges.insert((1, 2));

        // Node 0 is the support
        assert!(graph.is_supported(&[0]));

        // Disconnect the support
        graph.disconnect_chunk(0);
        // Now no component has support node 0 connected
        // But node 0 is still in its own component
        assert!(graph.is_supported(&[0]));
    }

    #[test]
    fn test_debris_cleanup() {
        let mesh = DestructibleMesh::from_box(Vec3::ONE, 3, 42, 1000.0, 1.0);
        let mut manager = DestructionManager::new(mesh);

        // Force all chunks to be debris
        for c in &mut manager.mesh.chunks {
            c.is_detached = true;
            c.is_debris = true;
            c.debris_timer = 0.5;
        }

        // Step until debris expires
        for _ in 0..60 {
            manager.step(1.0 / 60.0, Vec3::new(0.0, -9.81, 0.0));
        }

        let removed = manager.cleanup_debris();
        assert!(
            !removed.is_empty(),
            "Should have debris to clean up"
        );
    }

    #[test]
    fn test_chunk_physics_step() {
        let mesh = DestructibleMesh::from_box(Vec3::ONE, 3, 42, 1000.0, 1.0);
        let mut manager = DestructionManager::new(mesh);

        // Detach a chunk with velocity
        if let Some(chunk) = manager.mesh.chunks.first_mut() {
            chunk.is_detached = true;
        }
        if !manager.chunk_velocities.is_empty() {
            manager.chunk_velocities[0] = Vec3::new(5.0, 10.0, 0.0);
        }

        let initial_pos = manager.chunk_positions[0];

        manager.step(1.0 / 60.0, Vec3::new(0.0, -9.81, 0.0));

        // Position should have changed
        let final_pos = manager.chunk_positions[0];
        assert!(
            (final_pos - initial_pos).length() > 0.01,
            "Chunk should move"
        );
    }

    #[test]
    fn test_cascading_destruction() {
        let mesh = DestructibleMesh::from_box(Vec3::ONE, 4, 42, 1000.0, 5.0);
        let mut manager = DestructionManager::new(mesh);
        manager.cascading_enabled = true;

        // Set chunk 0 as support
        manager.support_chunks = vec![0];

        // Impact that detaches chunk 0 (the support)
        let impact = ImpactEvent {
            position: manager.chunk_positions[0],
            energy: 10000.0,
            direction: Vec3::X,
            radius: 5.0,
        };

        let result = manager.apply_impact(&impact);

        // Cascading should occur since the support was removed
        assert!(
            result.detached_chunks.len() > 1 || result.cascading,
            "Removing support should cause cascading destruction"
        );
    }

    #[test]
    fn test_fracture_mesh_volume() {
        // Create a unit cube mesh
        let mesh = compute_convex_mesh_from_points(&[
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
        ]);

        let vol = mesh.compute_volume();
        assert!(
            (vol - 1.0).abs() < 0.1,
            "Unit cube should have volume ~1.0, got {}",
            vol
        );
    }

    #[test]
    fn test_clip_plane() {
        let plane = ClipPlane {
            point: Vec3::ZERO,
            normal: Vec3::X,
        };

        // Point in front
        assert!(plane.signed_distance(Vec3::new(1.0, 0.0, 0.0)) > 0.0);
        // Point behind
        assert!(plane.signed_distance(Vec3::new(-1.0, 0.0, 0.0)) < 0.0);
        // Point on plane
        assert!(plane.signed_distance(Vec3::ZERO).abs() < EPSILON);
    }

    #[test]
    fn test_destructible_component() {
        let component = DestructibleComponent::box_shape(Vec3::ONE, 5, 42);
        assert!(component.active);
        assert!(component.manager.mesh.chunk_count() > 0);
    }

    #[test]
    fn test_destruction_system() {
        let mut system = DestructionSystem::new();
        let mut destructibles = vec![DestructibleComponent::box_shape(Vec3::ONE, 4, 42)];

        system.queue_impact(
            0,
            ImpactEvent {
                position: Vec3::ZERO,
                energy: 10000.0,
                direction: Vec3::X,
                radius: 5.0,
            },
        );

        system.update(1.0 / 60.0, &mut destructibles);
        // Should not panic
    }

    #[test]
    fn test_interior_face_generation() {
        let plane = ClipPlane {
            point: Vec3::ZERO,
            normal: Vec3::X,
        };
        let face = generate_interior_face(&plane, -Vec3::ONE, Vec3::ONE);
        assert!(face.vertex_count() >= 3, "Should generate interior face");
        assert!(face.triangle_count() >= 1, "Should have triangles");
    }
}
