// engine/physics/src/heightfield_collision.rs
//
// Heightfield collision detection for the Genovo engine.
//
// Provides efficient collision queries against heightfield terrain data:
//
// - Ray vs heightfield intersection with early-out traversal.
// - Sphere vs heightfield collision with contact generation.
// - AABB vs heightfield overlap detection.
// - Per-cell material assignment for surface property queries.
// - Hole support (cells marked as non-collidable).
// - Grid-based traversal for efficient large-terrain queries.
// - Configurable cell resolution and height scale.
// - Contact point generation with surface normals.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of contacts generated per query.
const MAX_CONTACTS_PER_QUERY: usize = 64;

/// Small epsilon for floating point comparisons.
const EPSILON: f32 = 1e-6;

/// Default height scale factor.
const DEFAULT_HEIGHT_SCALE: f32 = 1.0;

/// Maximum ray marching steps for heightfield ray traversal.
const MAX_RAY_STEPS: usize = 2048;

// ---------------------------------------------------------------------------
// Heightfield Data
// ---------------------------------------------------------------------------

/// Material identifier for heightfield cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HeightfieldMaterialId(pub u16);

impl HeightfieldMaterialId {
    /// Default material.
    pub const DEFAULT: Self = Self(0);
    /// Grass material.
    pub const GRASS: Self = Self(1);
    /// Rock material.
    pub const ROCK: Self = Self(2);
    /// Sand material.
    pub const SAND: Self = Self(3);
    /// Dirt material.
    pub const DIRT: Self = Self(4);
    /// Snow material.
    pub const SNOW: Self = Self(5);
    /// Water (non-solid surface indicator).
    pub const WATER: Self = Self(6);
}

/// Cell flags for heightfield cells.
#[derive(Debug, Clone, Copy)]
pub struct CellFlags(pub u8);

impl CellFlags {
    /// Normal collidable cell.
    pub const SOLID: Self = Self(0x01);
    /// Hole (non-collidable).
    pub const HOLE: Self = Self(0x00);
    /// Walkable surface.
    pub const WALKABLE: Self = Self(0x02);
    /// Climbable surface.
    pub const CLIMBABLE: Self = Self(0x04);
    /// Slippery surface.
    pub const SLIPPERY: Self = Self(0x08);

    /// Check if this cell is solid (collidable).
    pub fn is_solid(self) -> bool {
        (self.0 & Self::SOLID.0) != 0
    }

    /// Check if this cell is a hole.
    pub fn is_hole(self) -> bool {
        !self.is_solid()
    }

    /// Check if this cell is walkable.
    pub fn is_walkable(self) -> bool {
        (self.0 & Self::WALKABLE.0) != 0
    }
}

/// Heightfield terrain data structure.
#[derive(Debug, Clone)]
pub struct Heightfield {
    /// Height samples in row-major order (row = z, column = x).
    pub heights: Vec<f32>,
    /// Per-cell material IDs (one per cell, not per vertex).
    pub materials: Vec<HeightfieldMaterialId>,
    /// Per-cell flags.
    pub flags: Vec<CellFlags>,
    /// Number of samples along the X axis.
    pub width: u32,
    /// Number of samples along the Z axis.
    pub depth: u32,
    /// World-space origin (min corner).
    pub origin: [f32; 3],
    /// Cell size in world units (X and Z spacing).
    pub cell_size: f32,
    /// Height scale multiplier.
    pub height_scale: f32,
    /// Minimum height in the field.
    pub min_height: f32,
    /// Maximum height in the field.
    pub max_height: f32,
    /// Cached AABB for the entire heightfield.
    pub aabb_min: [f32; 3],
    /// Cached AABB max.
    pub aabb_max: [f32; 3],
}

impl Heightfield {
    /// Create a new heightfield from height data.
    pub fn new(
        heights: Vec<f32>,
        width: u32,
        depth: u32,
        origin: [f32; 3],
        cell_size: f32,
    ) -> Self {
        assert_eq!(heights.len(), (width * depth) as usize);

        let mut min_h = f32::MAX;
        let mut max_h = f32::MIN;
        for &h in &heights {
            min_h = min_h.min(h);
            max_h = max_h.max(h);
        }

        let cell_count = ((width - 1) * (depth - 1)) as usize;
        let materials = vec![HeightfieldMaterialId::DEFAULT; cell_count];
        let flags = vec![CellFlags(CellFlags::SOLID.0 | CellFlags::WALKABLE.0); cell_count];

        let aabb_min = [origin[0], origin[1] + min_h, origin[2]];
        let aabb_max = [
            origin[0] + (width - 1) as f32 * cell_size,
            origin[1] + max_h,
            origin[2] + (depth - 1) as f32 * cell_size,
        ];

        Self {
            heights,
            materials,
            flags,
            width,
            depth,
            origin,
            cell_size,
            height_scale: DEFAULT_HEIGHT_SCALE,
            min_height: min_h,
            max_height: max_h,
            aabb_min,
            aabb_max,
        }
    }

    /// Create a flat heightfield at the given height.
    pub fn flat(width: u32, depth: u32, height: f32, origin: [f32; 3], cell_size: f32) -> Self {
        let heights = vec![height; (width * depth) as usize];
        Self::new(heights, width, depth, origin, cell_size)
    }

    /// Get the height at grid coordinates (ix, iz).
    pub fn height_at_grid(&self, ix: u32, iz: u32) -> f32 {
        if ix >= self.width || iz >= self.depth {
            return 0.0;
        }
        self.heights[(iz * self.width + ix) as usize] * self.height_scale
    }

    /// Get the height at a world-space (x, z) position using bilinear interpolation.
    pub fn height_at_world(&self, x: f32, z: f32) -> f32 {
        let local_x = (x - self.origin[0]) / self.cell_size;
        let local_z = (z - self.origin[2]) / self.cell_size;

        let ix = local_x.floor() as i32;
        let iz = local_z.floor() as i32;

        if ix < 0 || iz < 0 || ix >= (self.width - 1) as i32 || iz >= (self.depth - 1) as i32 {
            return 0.0;
        }

        let fx = local_x - ix as f32;
        let fz = local_z - iz as f32;

        let ix = ix as u32;
        let iz = iz as u32;

        let h00 = self.height_at_grid(ix, iz);
        let h10 = self.height_at_grid(ix + 1, iz);
        let h01 = self.height_at_grid(ix, iz + 1);
        let h11 = self.height_at_grid(ix + 1, iz + 1);

        // Bilinear interpolation.
        let h0 = h00 + (h10 - h00) * fx;
        let h1 = h01 + (h11 - h01) * fx;
        self.origin[1] + h0 + (h1 - h0) * fz
    }

    /// Get the surface normal at a world-space (x, z) position.
    pub fn normal_at_world(&self, x: f32, z: f32) -> [f32; 3] {
        let dx = self.cell_size * 0.5;
        let hl = self.height_at_world(x - dx, z);
        let hr = self.height_at_world(x + dx, z);
        let hb = self.height_at_world(x, z - dx);
        let hf = self.height_at_world(x, z + dx);

        let nx = hl - hr;
        let nz = hb - hf;
        let ny = 2.0 * dx;

        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        if len < EPSILON {
            return [0.0, 1.0, 0.0];
        }
        [nx / len, ny / len, nz / len]
    }

    /// Get the cell index from grid coordinates.
    pub fn cell_index(&self, cx: u32, cz: u32) -> Option<usize> {
        if cx >= self.width - 1 || cz >= self.depth - 1 {
            return None;
        }
        Some((cz * (self.width - 1) + cx) as usize)
    }

    /// Get the material for a cell.
    pub fn cell_material(&self, cx: u32, cz: u32) -> HeightfieldMaterialId {
        self.cell_index(cx, cz)
            .and_then(|i| self.materials.get(i))
            .copied()
            .unwrap_or(HeightfieldMaterialId::DEFAULT)
    }

    /// Set the material for a cell.
    pub fn set_cell_material(&mut self, cx: u32, cz: u32, material: HeightfieldMaterialId) {
        if let Some(i) = self.cell_index(cx, cz) {
            if i < self.materials.len() {
                self.materials[i] = material;
            }
        }
    }

    /// Get the flags for a cell.
    pub fn cell_flags(&self, cx: u32, cz: u32) -> CellFlags {
        self.cell_index(cx, cz)
            .and_then(|i| self.flags.get(i))
            .copied()
            .unwrap_or(CellFlags::HOLE)
    }

    /// Mark a cell as a hole.
    pub fn set_hole(&mut self, cx: u32, cz: u32) {
        if let Some(i) = self.cell_index(cx, cz) {
            if i < self.flags.len() {
                self.flags[i] = CellFlags::HOLE;
            }
        }
    }

    /// Mark a cell as solid.
    pub fn set_solid(&mut self, cx: u32, cz: u32) {
        if let Some(i) = self.cell_index(cx, cz) {
            if i < self.flags.len() {
                self.flags[i] = CellFlags(CellFlags::SOLID.0 | CellFlags::WALKABLE.0);
            }
        }
    }

    /// Check if a cell is a hole.
    pub fn is_hole(&self, cx: u32, cz: u32) -> bool {
        self.cell_flags(cx, cz).is_hole()
    }

    /// World-space position of a grid vertex.
    pub fn vertex_world_pos(&self, ix: u32, iz: u32) -> [f32; 3] {
        [
            self.origin[0] + ix as f32 * self.cell_size,
            self.origin[1] + self.height_at_grid(ix, iz),
            self.origin[2] + iz as f32 * self.cell_size,
        ]
    }

    /// Recalculate the AABB bounds.
    pub fn recalculate_bounds(&mut self) {
        self.min_height = f32::MAX;
        self.max_height = f32::MIN;
        for &h in &self.heights {
            let scaled = h * self.height_scale;
            self.min_height = self.min_height.min(scaled);
            self.max_height = self.max_height.max(scaled);
        }
        self.aabb_min = [self.origin[0], self.origin[1] + self.min_height, self.origin[2]];
        self.aabb_max = [
            self.origin[0] + (self.width - 1) as f32 * self.cell_size,
            self.origin[1] + self.max_height,
            self.origin[2] + (self.depth - 1) as f32 * self.cell_size,
        ];
    }

    /// Total number of cells.
    pub fn cell_count(&self) -> usize {
        ((self.width - 1) * (self.depth - 1)) as usize
    }

    /// Total number of vertices.
    pub fn vertex_count(&self) -> usize {
        (self.width * self.depth) as usize
    }
}

// ---------------------------------------------------------------------------
// Ray vs Heightfield
// ---------------------------------------------------------------------------

/// Result of a ray-heightfield intersection.
#[derive(Debug, Clone)]
pub struct HeightfieldRayHit {
    /// Hit point in world space.
    pub point: [f32; 3],
    /// Surface normal at hit point.
    pub normal: [f32; 3],
    /// Distance along the ray to the hit.
    pub distance: f32,
    /// Cell coordinates of the hit.
    pub cell_x: u32,
    /// Cell Z coordinate.
    pub cell_z: u32,
    /// Material at the hit cell.
    pub material: HeightfieldMaterialId,
}

/// Cast a ray against a heightfield.
///
/// Uses a grid traversal algorithm (DDA-like) for efficient early-out.
pub fn raycast_heightfield(
    heightfield: &Heightfield,
    ray_origin: [f32; 3],
    ray_direction: [f32; 3],
    max_distance: f32,
) -> Option<HeightfieldRayHit> {
    // Normalize direction.
    let dir_len = (ray_direction[0] * ray_direction[0]
        + ray_direction[1] * ray_direction[1]
        + ray_direction[2] * ray_direction[2])
        .sqrt();
    if dir_len < EPSILON {
        return None;
    }
    let dir = [
        ray_direction[0] / dir_len,
        ray_direction[1] / dir_len,
        ray_direction[2] / dir_len,
    ];

    // Check AABB first.
    if !ray_aabb_test(ray_origin, dir, heightfield.aabb_min, heightfield.aabb_max, max_distance) {
        return None;
    }

    // Step through the heightfield using uniform steps.
    let step_size = heightfield.cell_size * 0.5;
    let steps = ((max_distance / step_size) as usize).min(MAX_RAY_STEPS);

    let mut prev_y = ray_origin[1];
    for i in 0..steps {
        let t = (i as f32 + 0.5) * step_size;
        if t > max_distance {
            break;
        }

        let px = ray_origin[0] + dir[0] * t;
        let py = ray_origin[1] + dir[1] * t;
        let pz = ray_origin[2] + dir[2] * t;

        let terrain_y = heightfield.height_at_world(px, pz);

        if py <= terrain_y && prev_y > terrain_y - heightfield.cell_size {
            // Refine the intersection with binary search.
            let mut t_lo = ((i as f32 - 1.0).max(0.0)) * step_size;
            let mut t_hi = t;

            for _ in 0..16 {
                let t_mid = (t_lo + t_hi) * 0.5;
                let mx = ray_origin[0] + dir[0] * t_mid;
                let my = ray_origin[1] + dir[1] * t_mid;
                let mz = ray_origin[2] + dir[2] * t_mid;
                let mh = heightfield.height_at_world(mx, mz);

                if my <= mh {
                    t_hi = t_mid;
                } else {
                    t_lo = t_mid;
                }
            }

            let hit_t = (t_lo + t_hi) * 0.5;
            let hit_point = [
                ray_origin[0] + dir[0] * hit_t,
                ray_origin[1] + dir[1] * hit_t,
                ray_origin[2] + dir[2] * hit_t,
            ];
            let normal = heightfield.normal_at_world(hit_point[0], hit_point[2]);

            let local_x = (hit_point[0] - heightfield.origin[0]) / heightfield.cell_size;
            let local_z = (hit_point[2] - heightfield.origin[2]) / heightfield.cell_size;
            let cx = (local_x.floor() as u32).min(heightfield.width.saturating_sub(2));
            let cz = (local_z.floor() as u32).min(heightfield.depth.saturating_sub(2));

            if heightfield.is_hole(cx, cz) {
                prev_y = py;
                continue;
            }

            return Some(HeightfieldRayHit {
                point: hit_point,
                normal,
                distance: hit_t,
                cell_x: cx,
                cell_z: cz,
                material: heightfield.cell_material(cx, cz),
            });
        }

        prev_y = py;
    }

    None
}

/// Quick ray-AABB intersection test.
fn ray_aabb_test(
    origin: [f32; 3],
    dir: [f32; 3],
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    max_dist: f32,
) -> bool {
    let mut tmin = 0.0f32;
    let mut tmax = max_dist;

    for i in 0..3 {
        if dir[i].abs() < EPSILON {
            if origin[i] < aabb_min[i] || origin[i] > aabb_max[i] {
                return false;
            }
        } else {
            let inv_d = 1.0 / dir[i];
            let mut t1 = (aabb_min[i] - origin[i]) * inv_d;
            let mut t2 = (aabb_max[i] - origin[i]) * inv_d;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            tmin = tmin.max(t1);
            tmax = tmax.min(t2);
            if tmin > tmax {
                return false;
            }
        }
    }

    true
}

// ---------------------------------------------------------------------------
// Sphere vs Heightfield
// ---------------------------------------------------------------------------

/// Contact point from a sphere-heightfield collision.
#[derive(Debug, Clone)]
pub struct HeightfieldContact {
    /// Contact point in world space.
    pub point: [f32; 3],
    /// Contact normal (pointing away from the heightfield surface).
    pub normal: [f32; 3],
    /// Penetration depth (positive = penetrating).
    pub depth: f32,
    /// Cell coordinates.
    pub cell_x: u32,
    /// Cell Z coordinate.
    pub cell_z: u32,
    /// Material at contact.
    pub material: HeightfieldMaterialId,
}

/// Test sphere vs heightfield collision.
pub fn sphere_vs_heightfield(
    heightfield: &Heightfield,
    sphere_center: [f32; 3],
    sphere_radius: f32,
) -> Vec<HeightfieldContact> {
    let mut contacts = Vec::new();

    // Determine which cells the sphere overlaps.
    let min_x = ((sphere_center[0] - sphere_radius - heightfield.origin[0]) / heightfield.cell_size)
        .floor()
        .max(0.0) as u32;
    let max_x = ((sphere_center[0] + sphere_radius - heightfield.origin[0]) / heightfield.cell_size)
        .ceil()
        .min((heightfield.width - 2) as f32) as u32;
    let min_z = ((sphere_center[2] - sphere_radius - heightfield.origin[2]) / heightfield.cell_size)
        .floor()
        .max(0.0) as u32;
    let max_z = ((sphere_center[2] + sphere_radius - heightfield.origin[2]) / heightfield.cell_size)
        .ceil()
        .min((heightfield.depth - 2) as f32) as u32;

    for cz in min_z..=max_z {
        for cx in min_x..=max_x {
            if heightfield.is_hole(cx, cz) {
                continue;
            }

            // Sample height at the closest point to sphere center in this cell.
            let cell_x_min = heightfield.origin[0] + cx as f32 * heightfield.cell_size;
            let cell_z_min = heightfield.origin[2] + cz as f32 * heightfield.cell_size;
            let cell_x_max = cell_x_min + heightfield.cell_size;
            let cell_z_max = cell_z_min + heightfield.cell_size;

            let closest_x = sphere_center[0].clamp(cell_x_min, cell_x_max);
            let closest_z = sphere_center[2].clamp(cell_z_min, cell_z_max);

            let terrain_y = heightfield.height_at_world(closest_x, closest_z);
            let normal = heightfield.normal_at_world(closest_x, closest_z);

            // Distance from sphere center to terrain surface along the normal.
            let to_sphere = [
                sphere_center[0] - closest_x,
                sphere_center[1] - terrain_y,
                sphere_center[2] - closest_z,
            ];

            let dist_along_normal =
                to_sphere[0] * normal[0] + to_sphere[1] * normal[1] + to_sphere[2] * normal[2];

            if dist_along_normal < sphere_radius {
                let depth = sphere_radius - dist_along_normal;
                let contact_point = [
                    sphere_center[0] - normal[0] * (dist_along_normal + depth * 0.5),
                    sphere_center[1] - normal[1] * (dist_along_normal + depth * 0.5),
                    sphere_center[2] - normal[2] * (dist_along_normal + depth * 0.5),
                ];

                contacts.push(HeightfieldContact {
                    point: contact_point,
                    normal,
                    depth,
                    cell_x: cx,
                    cell_z: cz,
                    material: heightfield.cell_material(cx, cz),
                });

                if contacts.len() >= MAX_CONTACTS_PER_QUERY {
                    return contacts;
                }
            }
        }
    }

    contacts
}

// ---------------------------------------------------------------------------
// AABB vs Heightfield
// ---------------------------------------------------------------------------

/// Result of an AABB-heightfield overlap test.
#[derive(Debug, Clone)]
pub struct HeightfieldOverlap {
    /// Whether the AABB overlaps the heightfield.
    pub overlapping: bool,
    /// Overlapping cells.
    pub cells: Vec<(u32, u32)>,
    /// Minimum penetration depth.
    pub min_depth: f32,
    /// Maximum penetration depth.
    pub max_depth: f32,
    /// Average surface normal in the overlap region.
    pub average_normal: [f32; 3],
}

/// Test AABB vs heightfield overlap.
pub fn aabb_vs_heightfield(
    heightfield: &Heightfield,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> HeightfieldOverlap {
    let mut result = HeightfieldOverlap {
        overlapping: false,
        cells: Vec::new(),
        min_depth: f32::MAX,
        max_depth: 0.0,
        average_normal: [0.0, 0.0, 0.0],
    };

    let cx_min = ((aabb_min[0] - heightfield.origin[0]) / heightfield.cell_size)
        .floor()
        .max(0.0) as u32;
    let cx_max = ((aabb_max[0] - heightfield.origin[0]) / heightfield.cell_size)
        .ceil()
        .min((heightfield.width - 2) as f32) as u32;
    let cz_min = ((aabb_min[2] - heightfield.origin[2]) / heightfield.cell_size)
        .floor()
        .max(0.0) as u32;
    let cz_max = ((aabb_max[2] - heightfield.origin[2]) / heightfield.cell_size)
        .ceil()
        .min((heightfield.depth - 2) as f32) as u32;

    let mut normal_sum = [0.0f32; 3];
    let mut normal_count = 0u32;

    for cz in cz_min..=cz_max {
        for cx in cx_min..=cx_max {
            if heightfield.is_hole(cx, cz) {
                continue;
            }

            // Check the four corners of this cell.
            let corners = [
                heightfield.vertex_world_pos(cx, cz),
                heightfield.vertex_world_pos(cx + 1, cz),
                heightfield.vertex_world_pos(cx, cz + 1),
                heightfield.vertex_world_pos(cx + 1, cz + 1),
            ];

            let cell_max_y = corners.iter().map(|c| c[1]).fold(f32::MIN, f32::max);

            if aabb_min[1] <= cell_max_y {
                let depth = cell_max_y - aabb_min[1];
                result.overlapping = true;
                result.cells.push((cx, cz));
                result.min_depth = result.min_depth.min(depth);
                result.max_depth = result.max_depth.max(depth);

                let center_x = heightfield.origin[0] + (cx as f32 + 0.5) * heightfield.cell_size;
                let center_z = heightfield.origin[2] + (cz as f32 + 0.5) * heightfield.cell_size;
                let n = heightfield.normal_at_world(center_x, center_z);
                normal_sum[0] += n[0];
                normal_sum[1] += n[1];
                normal_sum[2] += n[2];
                normal_count += 1;
            }
        }
    }

    if normal_count > 0 {
        let inv = 1.0 / normal_count as f32;
        let n = [normal_sum[0] * inv, normal_sum[1] * inv, normal_sum[2] * inv];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > EPSILON {
            result.average_normal = [n[0] / len, n[1] / len, n[2] / len];
        } else {
            result.average_normal = [0.0, 1.0, 0.0];
        }
    }

    if !result.overlapping {
        result.min_depth = 0.0;
    }

    result
}

// ---------------------------------------------------------------------------
// Heightfield Query System
// ---------------------------------------------------------------------------

/// Options for heightfield collision queries.
#[derive(Debug, Clone)]
pub struct HeightfieldQueryOptions {
    /// Maximum distance for raycasts.
    pub max_ray_distance: f32,
    /// Whether to check cell holes.
    pub respect_holes: bool,
    /// Material filter (None = accept all).
    pub material_filter: Option<Vec<HeightfieldMaterialId>>,
    /// Whether to generate contact normals.
    pub compute_normals: bool,
    /// Maximum contacts to generate.
    pub max_contacts: usize,
}

impl Default for HeightfieldQueryOptions {
    fn default() -> Self {
        Self {
            max_ray_distance: 1000.0,
            respect_holes: true,
            material_filter: None,
            compute_normals: true,
            max_contacts: MAX_CONTACTS_PER_QUERY,
        }
    }
}

/// Unified query system for heightfield collision.
#[derive(Debug)]
pub struct HeightfieldCollisionSystem {
    /// Registered heightfields.
    pub heightfields: Vec<Heightfield>,
    /// Default query options.
    pub default_options: HeightfieldQueryOptions,
    /// Statistics.
    pub stats: HeightfieldCollisionStats,
}

impl HeightfieldCollisionSystem {
    /// Create a new collision system.
    pub fn new() -> Self {
        Self {
            heightfields: Vec::new(),
            default_options: HeightfieldQueryOptions::default(),
            stats: HeightfieldCollisionStats::default(),
        }
    }

    /// Add a heightfield to the system.
    pub fn add_heightfield(&mut self, heightfield: Heightfield) -> usize {
        let idx = self.heightfields.len();
        self.heightfields.push(heightfield);
        idx
    }

    /// Raycast against all heightfields.
    pub fn raycast(
        &self,
        origin: [f32; 3],
        direction: [f32; 3],
        max_distance: f32,
    ) -> Option<(usize, HeightfieldRayHit)> {
        let mut closest: Option<(usize, HeightfieldRayHit)> = None;

        for (i, hf) in self.heightfields.iter().enumerate() {
            if let Some(hit) = raycast_heightfield(hf, origin, direction, max_distance) {
                let is_closer = closest.as_ref().map_or(true, |(_, h)| hit.distance < h.distance);
                if is_closer {
                    closest = Some((i, hit));
                }
            }
        }

        closest
    }

    /// Sphere collision against all heightfields.
    pub fn sphere_query(
        &self,
        center: [f32; 3],
        radius: f32,
    ) -> Vec<(usize, HeightfieldContact)> {
        let mut all_contacts = Vec::new();

        for (i, hf) in self.heightfields.iter().enumerate() {
            let contacts = sphere_vs_heightfield(hf, center, radius);
            for contact in contacts {
                all_contacts.push((i, contact));
            }
        }

        all_contacts
    }

    /// AABB overlap against all heightfields.
    pub fn aabb_query(
        &self,
        min: [f32; 3],
        max: [f32; 3],
    ) -> Vec<(usize, HeightfieldOverlap)> {
        let mut results = Vec::new();

        for (i, hf) in self.heightfields.iter().enumerate() {
            let overlap = aabb_vs_heightfield(hf, min, max);
            if overlap.overlapping {
                results.push((i, overlap));
            }
        }

        results
    }

    /// Get the heightfield at the given index.
    pub fn get(&self, index: usize) -> Option<&Heightfield> {
        self.heightfields.get(index)
    }

    /// Get a mutable reference to a heightfield.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Heightfield> {
        self.heightfields.get_mut(index)
    }
}

/// Statistics for heightfield collision queries.
#[derive(Debug, Clone, Default)]
pub struct HeightfieldCollisionStats {
    /// Total raycasts performed.
    pub raycasts: u64,
    /// Total raycast hits.
    pub raycast_hits: u64,
    /// Total sphere queries.
    pub sphere_queries: u64,
    /// Total sphere contacts generated.
    pub sphere_contacts: u64,
    /// Total AABB queries.
    pub aabb_queries: u64,
    /// Total AABB overlaps found.
    pub aabb_overlaps: u64,
}
