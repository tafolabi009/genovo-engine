//! Terrain collision and physics integration for the Genovo engine.
//!
//! Provides a rich physics interface for terrain surfaces, including:
//!
//! - **Heightfield collider** — efficient collision detection against the
//!   terrain surface using the heightmap directly.
//! - **Per-cell physics material** — each terrain cell can have a different
//!   physics material (friction, restitution, sound effect).
//! - **Slope analysis** — compute slopes for AI navigation, vehicle physics,
//!   and character movement.
//! - **Terrain holes** — cells marked as holes are excluded from collision
//!   (used for caves, tunnels, building interiors).
//! - **Contact queries** — sphere, capsule, AABB, and ray tests against the
//!   terrain surface.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Vec3 (self-contained)
// ---------------------------------------------------------------------------

/// Minimal 3D vector for self-contained compilation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn length_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len < 1e-9 { return Self::ZERO; }
        Self { x: self.x / len, y: self.y / len, z: self.z / len }
    }

    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }

    pub fn scale(&self, s: f32) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }
}

impl Default for Vec3 {
    fn default() -> Self { Self::ZERO }
}

// ---------------------------------------------------------------------------
// Physics material
// ---------------------------------------------------------------------------

/// Surface physics material properties.
#[derive(Debug, Clone)]
pub struct PhysicsMaterial {
    /// Material identifier name.
    pub name: String,
    /// Static friction coefficient (0..1+).
    pub static_friction: f32,
    /// Dynamic friction coefficient (0..1+).
    pub dynamic_friction: f32,
    /// Restitution / bounciness (0..1).
    pub restitution: f32,
    /// Sound effect identifier for footsteps / impacts.
    pub sound_id: Option<String>,
    /// Particle effect identifier for impacts.
    pub particle_id: Option<String>,
    /// Speed multiplier for characters/vehicles on this surface.
    pub speed_multiplier: f32,
    /// Whether this surface is slippery (ice, oil).
    pub slippery: bool,
    /// Softness (0 = hard, 1 = very soft). Affects deformation / sinking.
    pub softness: f32,
}

impl PhysicsMaterial {
    /// Creates a default material.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            static_friction: 0.6,
            dynamic_friction: 0.4,
            restitution: 0.1,
            sound_id: None,
            particle_id: None,
            speed_multiplier: 1.0,
            slippery: false,
            softness: 0.0,
        }
    }

    /// Standard dirt/soil material.
    pub fn dirt() -> Self {
        Self {
            name: "Dirt".to_string(),
            static_friction: 0.7,
            dynamic_friction: 0.5,
            restitution: 0.05,
            sound_id: Some("sfx_footstep_dirt".to_string()),
            particle_id: Some("fx_dust".to_string()),
            speed_multiplier: 0.9,
            slippery: false,
            softness: 0.3,
        }
    }

    /// Grass material.
    pub fn grass() -> Self {
        Self {
            name: "Grass".to_string(),
            static_friction: 0.65,
            dynamic_friction: 0.45,
            restitution: 0.05,
            sound_id: Some("sfx_footstep_grass".to_string()),
            particle_id: None,
            speed_multiplier: 0.95,
            slippery: false,
            softness: 0.2,
        }
    }

    /// Rock/stone material.
    pub fn rock() -> Self {
        Self {
            name: "Rock".to_string(),
            static_friction: 0.8,
            dynamic_friction: 0.6,
            restitution: 0.3,
            sound_id: Some("sfx_footstep_rock".to_string()),
            particle_id: Some("fx_sparks".to_string()),
            speed_multiplier: 1.0,
            slippery: false,
            softness: 0.0,
        }
    }

    /// Sand material.
    pub fn sand() -> Self {
        Self {
            name: "Sand".to_string(),
            static_friction: 0.5,
            dynamic_friction: 0.35,
            restitution: 0.02,
            sound_id: Some("sfx_footstep_sand".to_string()),
            particle_id: Some("fx_sand_puff".to_string()),
            speed_multiplier: 0.7,
            slippery: false,
            softness: 0.5,
        }
    }

    /// Snow material.
    pub fn snow() -> Self {
        Self {
            name: "Snow".to_string(),
            static_friction: 0.35,
            dynamic_friction: 0.2,
            restitution: 0.01,
            sound_id: Some("sfx_footstep_snow".to_string()),
            particle_id: Some("fx_snow_puff".to_string()),
            speed_multiplier: 0.6,
            slippery: true,
            softness: 0.7,
        }
    }

    /// Ice material.
    pub fn ice() -> Self {
        Self {
            name: "Ice".to_string(),
            static_friction: 0.1,
            dynamic_friction: 0.05,
            restitution: 0.2,
            sound_id: Some("sfx_footstep_ice".to_string()),
            particle_id: None,
            speed_multiplier: 0.8,
            slippery: true,
            softness: 0.0,
        }
    }

    /// Mud material.
    pub fn mud() -> Self {
        Self {
            name: "Mud".to_string(),
            static_friction: 0.4,
            dynamic_friction: 0.25,
            restitution: 0.0,
            sound_id: Some("sfx_footstep_mud".to_string()),
            particle_id: Some("fx_mud_splash".to_string()),
            speed_multiplier: 0.5,
            slippery: true,
            softness: 0.8,
        }
    }

    /// Blends two materials based on a weight (0 = self, 1 = other).
    pub fn blend(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            name: if t < 0.5 { self.name.clone() } else { other.name.clone() },
            static_friction: self.static_friction + (other.static_friction - self.static_friction) * t,
            dynamic_friction: self.dynamic_friction + (other.dynamic_friction - self.dynamic_friction) * t,
            restitution: self.restitution + (other.restitution - self.restitution) * t,
            sound_id: if t < 0.5 { self.sound_id.clone() } else { other.sound_id.clone() },
            particle_id: if t < 0.5 { self.particle_id.clone() } else { other.particle_id.clone() },
            speed_multiplier: self.speed_multiplier + (other.speed_multiplier - self.speed_multiplier) * t,
            slippery: if t < 0.5 { self.slippery } else { other.slippery },
            softness: self.softness + (other.softness - self.softness) * t,
        }
    }
}

impl Default for PhysicsMaterial {
    fn default() -> Self {
        Self::new("Default")
    }
}

impl fmt::Display for PhysicsMaterial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PhysicsMaterial({}, sf={:.2}, df={:.2}, rest={:.2})",
            self.name, self.static_friction, self.dynamic_friction, self.restitution
        )
    }
}

// ---------------------------------------------------------------------------
// Slope analysis
// ---------------------------------------------------------------------------

/// Result of slope analysis at a terrain point.
#[derive(Debug, Clone, Copy)]
pub struct SlopeInfo {
    /// Slope angle in radians (0 = flat, PI/2 = vertical).
    pub angle_rad: f32,
    /// Slope angle in degrees.
    pub angle_deg: f32,
    /// Slope gradient (rise/run).
    pub gradient: f32,
    /// Surface normal at the point.
    pub normal: Vec3,
    /// Downhill direction (tangent to surface, pointing downhill).
    pub downhill_direction: Vec3,
    /// Whether the slope is walkable (below the walkable threshold).
    pub walkable: bool,
    /// Whether the slope exceeds the maximum vehicle slope.
    pub vehicle_traversable: bool,
}

impl SlopeInfo {
    /// Default walkable slope threshold in degrees.
    pub const DEFAULT_WALKABLE_ANGLE: f32 = 45.0;
    /// Default vehicle traversable slope threshold in degrees.
    pub const DEFAULT_VEHICLE_ANGLE: f32 = 35.0;
}

/// Configuration for slope analysis thresholds.
#[derive(Debug, Clone, Copy)]
pub struct SlopeConfig {
    /// Maximum walkable slope in degrees.
    pub max_walkable_angle: f32,
    /// Maximum vehicle slope in degrees.
    pub max_vehicle_angle: f32,
    /// Whether to use the steepest direction or average.
    pub use_steepest: bool,
}

impl Default for SlopeConfig {
    fn default() -> Self {
        Self {
            max_walkable_angle: SlopeInfo::DEFAULT_WALKABLE_ANGLE,
            max_vehicle_angle: SlopeInfo::DEFAULT_VEHICLE_ANGLE,
            use_steepest: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Ray / contact results
// ---------------------------------------------------------------------------

/// Result of a ray cast against the terrain.
#[derive(Debug, Clone)]
pub struct TerrainRaycastHit {
    /// Whether the ray hit the terrain.
    pub hit: bool,
    /// World-space hit point.
    pub point: Vec3,
    /// Surface normal at the hit point.
    pub normal: Vec3,
    /// Distance from the ray origin to the hit point.
    pub distance: f32,
    /// Heightmap cell coordinates (column, row).
    pub cell: (u32, u32),
    /// The physics material at the hit point.
    pub material_index: u8,
    /// Whether the hit is on a hole cell.
    pub is_hole: bool,
}

impl TerrainRaycastHit {
    /// Creates a "no hit" result.
    pub fn miss() -> Self {
        Self {
            hit: false,
            point: Vec3::ZERO,
            normal: Vec3::UP,
            distance: f32::MAX,
            cell: (0, 0),
            material_index: 0,
            is_hole: false,
        }
    }
}

/// Contact information for a collision with the terrain.
#[derive(Debug, Clone)]
pub struct TerrainContact {
    /// World-space contact point.
    pub point: Vec3,
    /// Surface normal at the contact point.
    pub normal: Vec3,
    /// Penetration depth (positive = overlapping).
    pub penetration: f32,
    /// The physics material at the contact point.
    pub material: PhysicsMaterial,
    /// Friction to use for this contact.
    pub friction: f32,
    /// Restitution to use for this contact.
    pub restitution: f32,
    /// Cell coordinates.
    pub cell: (u32, u32),
}

// ---------------------------------------------------------------------------
// Heightfield collider
// ---------------------------------------------------------------------------

/// A physics collider built from a terrain heightfield.
///
/// Provides efficient collision queries against the terrain surface.
/// Each cell can have a different physics material and can be flagged
/// as a "hole" (non-collidable).
pub struct HeightfieldCollider {
    /// Width of the heightfield in samples.
    width: u32,
    /// Height (depth) of the heightfield in samples.
    height: u32,
    /// Heightmap data (row-major, width * height floats).
    heights: Vec<f32>,
    /// Per-cell physics material index (width-1 * height-1 entries).
    cell_materials: Vec<u8>,
    /// Per-cell hole flags (true = hole, no collision).
    cell_holes: Vec<bool>,
    /// Registered physics materials.
    materials: Vec<PhysicsMaterial>,
    /// Material name -> index lookup.
    material_index: HashMap<String, u8>,
    /// World-space size of the terrain in the X direction.
    world_size_x: f32,
    /// World-space size of the terrain in the Z direction.
    world_size_z: f32,
    /// Vertical scale applied to height values.
    height_scale: f32,
    /// Collision margin (skin thickness).
    margin: f32,
    /// World-space offset of the terrain origin.
    origin: Vec3,
    /// Whether the collider is enabled.
    enabled: bool,
    /// Slope analysis configuration.
    slope_config: SlopeConfig,
}

impl HeightfieldCollider {
    /// Creates a new heightfield collider.
    pub fn new(
        width: u32,
        height: u32,
        heights: Vec<f32>,
        world_size_x: f32,
        world_size_z: f32,
        height_scale: f32,
    ) -> Self {
        assert_eq!(
            heights.len(),
            (width * height) as usize,
            "Heights length must match width * height"
        );

        let cell_count = ((width - 1) * (height - 1)) as usize;
        let mut materials = Vec::new();
        materials.push(PhysicsMaterial::default());

        let mut material_index = HashMap::new();
        material_index.insert("Default".to_string(), 0);

        Self {
            width,
            height,
            heights,
            cell_materials: vec![0; cell_count],
            cell_holes: vec![false; cell_count],
            materials,
            material_index,
            world_size_x,
            world_size_z,
            height_scale,
            margin: 0.02,
            origin: Vec3::ZERO,
            enabled: true,
            slope_config: SlopeConfig::default(),
        }
    }

    /// Creates a flat collider.
    pub fn new_flat(
        width: u32,
        height: u32,
        flat_height: f32,
        world_size_x: f32,
        world_size_z: f32,
    ) -> Self {
        let heights = vec![flat_height; (width * height) as usize];
        Self::new(width, height, heights, world_size_x, world_size_z, 1.0)
    }

    // -- Configuration --------------------------------------------------------

    /// Sets the collider origin in world space.
    pub fn set_origin(&mut self, origin: Vec3) {
        self.origin = origin;
    }

    /// Returns the collider origin.
    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    /// Sets the collision margin.
    pub fn set_margin(&mut self, margin: f32) {
        self.margin = margin.max(0.0);
    }

    /// Enables or disables the collider.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Returns whether the collider is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Sets the slope analysis configuration.
    pub fn set_slope_config(&mut self, config: SlopeConfig) {
        self.slope_config = config;
    }

    /// Returns the heightfield dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Returns the world size.
    pub fn world_size(&self) -> (f32, f32) {
        (self.world_size_x, self.world_size_z)
    }

    /// Returns the cell size in world units.
    pub fn cell_size(&self) -> (f32, f32) {
        (
            self.world_size_x / (self.width - 1) as f32,
            self.world_size_z / (self.height - 1) as f32,
        )
    }

    // -- Material management --------------------------------------------------

    /// Registers a physics material. Returns its index.
    pub fn register_material(&mut self, material: PhysicsMaterial) -> u8 {
        let idx = self.materials.len() as u8;
        self.material_index.insert(material.name.clone(), idx);
        self.materials.push(material);
        idx
    }

    /// Returns a material by index.
    pub fn get_material(&self, index: u8) -> Option<&PhysicsMaterial> {
        self.materials.get(index as usize)
    }

    /// Returns a material index by name.
    pub fn material_index_by_name(&self, name: &str) -> Option<u8> {
        self.material_index.get(name).copied()
    }

    /// Returns the number of registered materials.
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    /// Sets the physics material for a cell.
    pub fn set_cell_material(&mut self, col: u32, row: u32, material_index: u8) {
        if col < self.width - 1 && row < self.height - 1 {
            let idx = (row * (self.width - 1) + col) as usize;
            self.cell_materials[idx] = material_index;
        }
    }

    /// Gets the physics material index for a cell.
    pub fn get_cell_material_index(&self, col: u32, row: u32) -> u8 {
        if col < self.width - 1 && row < self.height - 1 {
            let idx = (row * (self.width - 1) + col) as usize;
            self.cell_materials[idx]
        } else {
            0
        }
    }

    /// Gets the physics material for a cell.
    pub fn get_cell_material(&self, col: u32, row: u32) -> &PhysicsMaterial {
        let idx = self.get_cell_material_index(col, row);
        &self.materials[idx as usize]
    }

    /// Paints a physics material over a rectangular region of cells.
    pub fn paint_material_rect(
        &mut self,
        col_min: u32,
        row_min: u32,
        col_max: u32,
        row_max: u32,
        material_index: u8,
    ) {
        let col_max = col_max.min(self.width - 2);
        let row_max = row_max.min(self.height - 2);
        for row in row_min..=row_max {
            for col in col_min..=col_max {
                self.set_cell_material(col, row, material_index);
            }
        }
    }

    // -- Hole management ------------------------------------------------------

    /// Marks a cell as a hole (no collision).
    pub fn set_hole(&mut self, col: u32, row: u32, is_hole: bool) {
        if col < self.width - 1 && row < self.height - 1 {
            let idx = (row * (self.width - 1) + col) as usize;
            self.cell_holes[idx] = is_hole;
        }
    }

    /// Returns whether a cell is a hole.
    pub fn is_hole(&self, col: u32, row: u32) -> bool {
        if col < self.width - 1 && row < self.height - 1 {
            let idx = (row * (self.width - 1) + col) as usize;
            self.cell_holes[idx]
        } else {
            false
        }
    }

    /// Sets a rectangular region as holes.
    pub fn set_hole_rect(
        &mut self,
        col_min: u32,
        row_min: u32,
        col_max: u32,
        row_max: u32,
        is_hole: bool,
    ) {
        let col_max = col_max.min(self.width - 2);
        let row_max = row_max.min(self.height - 2);
        for row in row_min..=row_max {
            for col in col_min..=col_max {
                self.set_hole(col, row, is_hole);
            }
        }
    }

    /// Returns the total number of hole cells.
    pub fn hole_count(&self) -> usize {
        self.cell_holes.iter().filter(|&&h| h).count()
    }

    // -- Height queries -------------------------------------------------------

    /// Converts a world-space XZ position to heightmap cell coordinates.
    fn world_to_cell(&self, world_x: f32, world_z: f32) -> Option<(f32, f32)> {
        let local_x = world_x - self.origin.x;
        let local_z = world_z - self.origin.z;

        if local_x < 0.0 || local_z < 0.0
            || local_x > self.world_size_x
            || local_z > self.world_size_z
        {
            return None;
        }

        let cell_x = local_x / self.world_size_x * (self.width - 1) as f32;
        let cell_z = local_z / self.world_size_z * (self.height - 1) as f32;

        Some((cell_x, cell_z))
    }

    /// Returns the raw height value at integer cell coordinates.
    fn height_at_cell(&self, col: u32, row: u32) -> f32 {
        let col = col.min(self.width - 1);
        let row = row.min(self.height - 1);
        self.heights[(row * self.width + col) as usize] * self.height_scale
    }

    /// Bilinear interpolation of height at fractional cell coordinates.
    fn sample_height(&self, cell_x: f32, cell_z: f32) -> f32 {
        let x0 = (cell_x as u32).min(self.width - 2);
        let z0 = (cell_z as u32).min(self.height - 2);
        let x1 = x0 + 1;
        let z1 = z0 + 1;

        let fx = cell_x - x0 as f32;
        let fz = cell_z - z0 as f32;

        let h00 = self.height_at_cell(x0, z0);
        let h10 = self.height_at_cell(x1, z0);
        let h01 = self.height_at_cell(x0, z1);
        let h11 = self.height_at_cell(x1, z1);

        let h0 = h00 + (h10 - h00) * fx;
        let h1 = h01 + (h11 - h01) * fx;

        h0 + (h1 - h0) * fz
    }

    /// Returns the terrain height at a world-space XZ position.
    pub fn height_at_world(&self, world_x: f32, world_z: f32) -> Option<f32> {
        let (cx, cz) = self.world_to_cell(world_x, world_z)?;

        // Check for holes.
        let col = cx as u32;
        let row = cz as u32;
        if self.is_hole(col, row) {
            return None;
        }

        Some(self.sample_height(cx, cz) + self.origin.y)
    }

    /// Returns the surface normal at a world-space XZ position.
    pub fn normal_at_world(&self, world_x: f32, world_z: f32) -> Option<Vec3> {
        let (cx, cz) = self.world_to_cell(world_x, world_z)?;

        let col = cx as u32;
        let row = cz as u32;
        if self.is_hole(col, row) {
            return None;
        }

        let (dx, dz) = self.cell_size();
        let epsilon = 0.5;

        let h_l = self.sample_height((cx - epsilon).max(0.0), cz);
        let h_r = self.sample_height((cx + epsilon).min((self.width - 1) as f32), cz);
        let h_d = self.sample_height(cx, (cz - epsilon).max(0.0));
        let h_u = self.sample_height(cx, (cz + epsilon).min((self.height - 1) as f32));

        let normal = Vec3::new(
            (h_l - h_r) / (2.0 * epsilon * dx / (self.width - 1) as f32 * self.world_size_x / (self.width - 1) as f32),
            2.0 * dx,
            (h_d - h_u) / (2.0 * epsilon * dz / (self.height - 1) as f32 * self.world_size_z / (self.height - 1) as f32),
        );

        Some(normal.normalized())
    }

    /// Returns the physics material at a world-space XZ position.
    pub fn material_at_world(&self, world_x: f32, world_z: f32) -> Option<&PhysicsMaterial> {
        let (cx, cz) = self.world_to_cell(world_x, world_z)?;
        let col = cx as u32;
        let row = cz as u32;
        Some(self.get_cell_material(col, row))
    }

    // -- Slope analysis -------------------------------------------------------

    /// Computes slope information at a world-space XZ position.
    pub fn slope_at_world(&self, world_x: f32, world_z: f32) -> Option<SlopeInfo> {
        let normal = self.normal_at_world(world_x, world_z)?;

        // Angle between surface normal and vertical (UP).
        let cos_angle = normal.dot(&Vec3::UP).clamp(-1.0, 1.0);
        let angle_rad = cos_angle.acos();
        let angle_deg = angle_rad.to_degrees();

        // Gradient = tan(angle).
        let gradient = if cos_angle.abs() > 1e-6 {
            angle_rad.tan()
        } else {
            f32::INFINITY
        };

        // Downhill direction: project gravity onto the surface plane.
        let gravity = Vec3::new(0.0, -1.0, 0.0);
        let proj_scale = gravity.dot(&normal);
        let projected = gravity.sub(&normal.scale(proj_scale));
        let downhill = projected.normalized();

        let walkable = angle_deg <= self.slope_config.max_walkable_angle;
        let vehicle_traversable = angle_deg <= self.slope_config.max_vehicle_angle;

        Some(SlopeInfo {
            angle_rad,
            angle_deg,
            gradient,
            normal,
            downhill_direction: downhill,
            walkable,
            vehicle_traversable,
        })
    }

    /// Analyses slopes over an entire grid region and returns a walkability map.
    ///
    /// The result is a 2D boolean grid where `true` = walkable.
    pub fn compute_walkability_map(&self) -> Vec<Vec<bool>> {
        let rows = (self.height - 1) as usize;
        let cols = (self.width - 1) as usize;
        let (cell_w, cell_h) = self.cell_size();

        let mut map = vec![vec![false; cols]; rows];

        for row in 0..rows {
            for col in 0..cols {
                if self.is_hole(col as u32, row as u32) {
                    map[row][col] = false;
                    continue;
                }

                let world_x = self.origin.x + (col as f32 + 0.5) * cell_w;
                let world_z = self.origin.z + (row as f32 + 0.5) * cell_h;

                if let Some(slope) = self.slope_at_world(world_x, world_z) {
                    map[row][col] = slope.walkable;
                }
            }
        }

        map
    }

    /// Computes the average slope in a circular area.
    pub fn average_slope_in_radius(
        &self,
        center_x: f32,
        center_z: f32,
        radius: f32,
        sample_count: u32,
    ) -> Option<f32> {
        let mut total_angle = 0.0f32;
        let mut count = 0u32;

        // Sample in a grid pattern within the radius.
        let steps = (sample_count as f32).sqrt() as u32 + 1;
        let step_size = 2.0 * radius / steps as f32;

        for zi in 0..steps {
            for xi in 0..steps {
                let x = center_x - radius + xi as f32 * step_size;
                let z = center_z - radius + zi as f32 * step_size;

                let dx = x - center_x;
                let dz = z - center_z;
                if dx * dx + dz * dz > radius * radius {
                    continue;
                }

                if let Some(slope) = self.slope_at_world(x, z) {
                    total_angle += slope.angle_deg;
                    count += 1;
                }
            }
        }

        if count > 0 {
            Some(total_angle / count as f32)
        } else {
            None
        }
    }

    // -- Collision queries ----------------------------------------------------

    /// Performs a vertical raycast (from above) at position (x, z).
    pub fn raycast_vertical(&self, world_x: f32, world_z: f32) -> TerrainRaycastHit {
        if !self.enabled {
            return TerrainRaycastHit::miss();
        }

        match self.world_to_cell(world_x, world_z) {
            None => TerrainRaycastHit::miss(),
            Some((cx, cz)) => {
                let col = cx as u32;
                let row = cz as u32;
                let is_hole = self.is_hole(col, row);

                if is_hole {
                    return TerrainRaycastHit {
                        hit: false,
                        is_hole: true,
                        ..TerrainRaycastHit::miss()
                    };
                }

                let height = self.sample_height(cx, cz) + self.origin.y;
                let normal = self.normal_at_world(world_x, world_z).unwrap_or(Vec3::UP);
                let material_index = self.get_cell_material_index(col, row);

                TerrainRaycastHit {
                    hit: true,
                    point: Vec3::new(world_x, height, world_z),
                    normal,
                    distance: 0.0, // Vertical cast has no meaningful distance here.
                    cell: (col, row),
                    material_index,
                    is_hole: false,
                }
            }
        }
    }

    /// Performs a ray cast against the terrain using ray marching.
    ///
    /// Marches along the ray in steps, testing against the heightfield
    /// at each step. Uses binary refinement for accuracy.
    pub fn raycast(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
    ) -> TerrainRaycastHit {
        if !self.enabled {
            return TerrainRaycastHit::miss();
        }

        let dir = direction.normalized();
        let step_size = self.cell_size().0.min(self.cell_size().1) * 0.5;
        let max_steps = (max_distance / step_size) as u32 + 1;

        let mut prev_pos = origin;
        let mut prev_height = f32::MAX;
        let mut prev_above = true;

        for i in 1..=max_steps {
            let t = i as f32 * step_size;
            if t > max_distance {
                break;
            }

            let pos = origin.add(&dir.scale(t));

            if let Some(terrain_h) = self.height_at_world(pos.x, pos.z) {
                let above = pos.y >= terrain_h;

                if !above && prev_above {
                    // Crossed the terrain -- binary search for exact intersection.
                    let hit_point = self.binary_refine_raycast(
                        prev_pos, pos, 8,
                    );

                    let h = self.height_at_world(hit_point.x, hit_point.z)
                        .unwrap_or(hit_point.y);
                    let normal = self.normal_at_world(hit_point.x, hit_point.z)
                        .unwrap_or(Vec3::UP);
                    let (cx, cz) = self.world_to_cell(hit_point.x, hit_point.z)
                        .unwrap_or((0.0, 0.0));
                    let col = cx as u32;
                    let row = cz as u32;

                    return TerrainRaycastHit {
                        hit: true,
                        point: Vec3::new(hit_point.x, h, hit_point.z),
                        normal,
                        distance: origin.sub(&hit_point).length(),
                        cell: (col, row),
                        material_index: self.get_cell_material_index(col, row),
                        is_hole: false,
                    };
                }

                prev_above = above;
                prev_height = terrain_h;
            }

            prev_pos = pos;
        }

        let _ = prev_height; // suppress unused variable warning
        TerrainRaycastHit::miss()
    }

    /// Binary refinement between two positions to find the terrain intersection.
    fn binary_refine_raycast(&self, above: Vec3, below: Vec3, iterations: u32) -> Vec3 {
        let mut a = above;
        let mut b = below;

        for _ in 0..iterations {
            let mid = a.lerp(&b, 0.5);
            if let Some(h) = self.height_at_world(mid.x, mid.z) {
                if mid.y >= h {
                    a = mid;
                } else {
                    b = mid;
                }
            } else {
                break;
            }
        }

        a.lerp(&b, 0.5)
    }

    /// Tests whether a sphere intersects the terrain.
    ///
    /// Returns contact information if there is an intersection.
    pub fn sphere_test(&self, center: Vec3, radius: f32) -> Option<TerrainContact> {
        if !self.enabled {
            return None;
        }

        let terrain_h = self.height_at_world(center.x, center.z)?;
        let bottom = center.y - radius;

        if bottom >= terrain_h + self.margin {
            return None;
        }

        let penetration = terrain_h + self.margin - bottom;
        let normal = self.normal_at_world(center.x, center.z).unwrap_or(Vec3::UP);
        let (cx, cz) = self.world_to_cell(center.x, center.z)?;
        let col = cx as u32;
        let row = cz as u32;
        let material = self.get_cell_material(col, row).clone();

        Some(TerrainContact {
            point: Vec3::new(center.x, terrain_h, center.z),
            normal,
            penetration,
            friction: material.dynamic_friction,
            restitution: material.restitution,
            material,
            cell: (col, row),
        })
    }

    /// Tests whether a capsule (vertical) intersects the terrain.
    ///
    /// Samples multiple points along the capsule's base circle.
    pub fn capsule_test(
        &self,
        base_center: Vec3,
        capsule_radius: f32,
        capsule_height: f32,
    ) -> Option<TerrainContact> {
        if !self.enabled {
            return None;
        }

        // Test the center.
        let center_contact = self.sphere_test(
            Vec3::new(base_center.x, base_center.y + capsule_radius, base_center.z),
            capsule_radius,
        );

        // Test 4 points around the perimeter.
        let offsets = [
            (capsule_radius, 0.0),
            (-capsule_radius, 0.0),
            (0.0, capsule_radius),
            (0.0, -capsule_radius),
        ];

        let mut deepest: Option<TerrainContact> = center_contact;

        for (dx, dz) in &offsets {
            let test_pos = Vec3::new(
                base_center.x + dx,
                base_center.y + capsule_radius,
                base_center.z + dz,
            );

            if let Some(contact) = self.sphere_test(test_pos, capsule_radius) {
                match &deepest {
                    None => deepest = Some(contact),
                    Some(existing) => {
                        if contact.penetration > existing.penetration {
                            deepest = Some(contact);
                        }
                    }
                }
            }
        }

        let _ = capsule_height; // Available for future use
        deepest
    }

    /// Tests whether an axis-aligned bounding box intersects the terrain.
    pub fn aabb_test(
        &self,
        min: Vec3,
        max: Vec3,
    ) -> bool {
        if !self.enabled {
            return false;
        }

        let test_points = [
            (min.x, min.z),
            (max.x, min.z),
            (min.x, max.z),
            (max.x, max.z),
            ((min.x + max.x) * 0.5, (min.z + max.z) * 0.5),
        ];

        for (x, z) in &test_points {
            if let Some(h) = self.height_at_world(*x, *z) {
                if h + self.margin > min.y {
                    return true;
                }
            }
        }

        false
    }

    /// Returns the material at a contact point, blending based on the
    /// contact's position within the cell using the splatmap weights.
    pub fn material_at_contact(&self, contact: &TerrainContact) -> PhysicsMaterial {
        // For now, return the cell's primary material.
        // A full implementation would blend based on splatmap weights.
        let (col, row) = contact.cell;
        self.get_cell_material(col, row).clone()
    }

    // -- Batch queries --------------------------------------------------------

    /// Performs multiple vertical raycasts in batch.
    pub fn batch_raycast_vertical(&self, positions: &[(f32, f32)]) -> Vec<TerrainRaycastHit> {
        positions
            .iter()
            .map(|(x, z)| self.raycast_vertical(*x, *z))
            .collect()
    }

    /// Samples heights at multiple world positions.
    pub fn batch_height_query(&self, positions: &[(f32, f32)]) -> Vec<Option<f32>> {
        positions
            .iter()
            .map(|(x, z)| self.height_at_world(*x, *z))
            .collect()
    }

    // -- Statistics -----------------------------------------------------------

    /// Returns the minimum height in the heightfield.
    pub fn min_height(&self) -> f32 {
        self.heights
            .iter()
            .copied()
            .fold(f32::MAX, f32::min)
            * self.height_scale
            + self.origin.y
    }

    /// Returns the maximum height in the heightfield.
    pub fn max_height(&self) -> f32 {
        self.heights
            .iter()
            .copied()
            .fold(f32::MIN, f32::max)
            * self.height_scale
            + self.origin.y
    }

    /// Returns the height range.
    pub fn height_range(&self) -> f32 {
        self.max_height() - self.min_height()
    }

    /// Returns the total number of cells.
    pub fn cell_count(&self) -> u32 {
        (self.width - 1) * (self.height - 1)
    }

    /// Returns the total number of non-hole cells.
    pub fn solid_cell_count(&self) -> u32 {
        self.cell_count() - self.hole_count() as u32
    }
}

impl fmt::Debug for HeightfieldCollider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HeightfieldCollider")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("world_size", &(self.world_size_x, self.world_size_z))
            .field("height_scale", &self.height_scale)
            .field("materials", &self.materials.len())
            .field("holes", &self.hole_count())
            .field("enabled", &self.enabled)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_flat_collider() -> HeightfieldCollider {
        HeightfieldCollider::new_flat(17, 17, 0.0, 100.0, 100.0)
    }

    fn make_sloped_collider() -> HeightfieldCollider {
        let w = 17u32;
        let h = 17u32;
        let mut heights = vec![0.0f32; (w * h) as usize];
        for row in 0..h {
            for col in 0..w {
                // Slope in the Z direction.
                heights[(row * w + col) as usize] = row as f32 / (h - 1) as f32 * 10.0;
            }
        }
        HeightfieldCollider::new(w, h, heights, 100.0, 100.0, 1.0)
    }

    #[test]
    fn flat_height_query() {
        let collider = make_flat_collider();
        let h = collider.height_at_world(50.0, 50.0);
        assert!(h.is_some());
        assert!((h.unwrap()).abs() < 0.01);
    }

    #[test]
    fn out_of_bounds_query() {
        let collider = make_flat_collider();
        assert!(collider.height_at_world(-1.0, 50.0).is_none());
        assert!(collider.height_at_world(50.0, 101.0).is_none());
    }

    #[test]
    fn sphere_test_flat() {
        let collider = make_flat_collider();

        // Sphere above terrain -- no contact.
        let contact = collider.sphere_test(Vec3::new(50.0, 5.0, 50.0), 1.0);
        assert!(contact.is_none());

        // Sphere touching terrain.
        let contact = collider.sphere_test(Vec3::new(50.0, 0.5, 50.0), 1.0);
        assert!(contact.is_some());
        assert!(contact.unwrap().penetration > 0.0);
    }

    #[test]
    fn aabb_test_flat() {
        let collider = make_flat_collider();

        // AABB above terrain.
        assert!(!collider.aabb_test(
            Vec3::new(40.0, 5.0, 40.0),
            Vec3::new(60.0, 10.0, 60.0),
        ));

        // AABB overlapping terrain.
        assert!(collider.aabb_test(
            Vec3::new(40.0, -1.0, 40.0),
            Vec3::new(60.0, 1.0, 60.0),
        ));
    }

    #[test]
    fn physics_materials() {
        let mut collider = make_flat_collider();
        let rock_idx = collider.register_material(PhysicsMaterial::rock());
        let sand_idx = collider.register_material(PhysicsMaterial::sand());

        collider.set_cell_material(5, 5, rock_idx);
        assert_eq!(collider.get_cell_material_index(5, 5), rock_idx);

        collider.paint_material_rect(0, 0, 3, 3, sand_idx);
        assert_eq!(collider.get_cell_material_index(2, 2), sand_idx);
    }

    #[test]
    fn terrain_holes() {
        let mut collider = make_flat_collider();
        collider.set_hole(5, 5, true);
        assert!(collider.is_hole(5, 5));
        assert_eq!(collider.hole_count(), 1);

        // Height query through a hole should return None.
        let (cell_w, cell_h) = collider.cell_size();
        let x = 5.0 * cell_w + cell_w * 0.5;
        let z = 5.0 * cell_h + cell_h * 0.5;
        assert!(collider.height_at_world(x, z).is_none());
    }

    #[test]
    fn hole_rect() {
        let mut collider = make_flat_collider();
        collider.set_hole_rect(2, 2, 5, 5, true);
        assert!(collider.is_hole(3, 3));
        assert!(collider.is_hole(5, 5));
        assert!(!collider.is_hole(6, 6));
    }

    #[test]
    fn slope_analysis_flat() {
        let collider = make_flat_collider();
        let slope = collider.slope_at_world(50.0, 50.0);
        assert!(slope.is_some());
        let slope = slope.unwrap();
        assert!(slope.angle_deg < 5.0); // Near-flat
        assert!(slope.walkable);
        assert!(slope.vehicle_traversable);
    }

    #[test]
    fn slope_analysis_sloped() {
        let collider = make_sloped_collider();
        let slope = collider.slope_at_world(50.0, 50.0);
        assert!(slope.is_some());
        let slope = slope.unwrap();
        assert!(slope.angle_deg > 0.0);
    }

    #[test]
    fn material_blend() {
        let rock = PhysicsMaterial::rock();
        let sand = PhysicsMaterial::sand();
        let blend = rock.blend(&sand, 0.5);
        assert!((blend.static_friction - 0.65).abs() < 0.01);
    }

    #[test]
    fn vertical_raycast() {
        let collider = make_flat_collider();
        let hit = collider.raycast_vertical(50.0, 50.0);
        assert!(hit.hit);
        assert!((hit.point.y).abs() < 0.01);
    }

    #[test]
    fn vertical_raycast_hole() {
        let mut collider = make_flat_collider();
        let (cell_w, cell_h) = collider.cell_size();
        collider.set_hole(5, 5, true);
        let x = 5.0 * cell_w + cell_w * 0.5;
        let z = 5.0 * cell_h + cell_h * 0.5;
        let hit = collider.raycast_vertical(x, z);
        assert!(!hit.hit);
    }

    #[test]
    fn capsule_test() {
        let collider = make_flat_collider();
        // Capsule above terrain.
        let contact = collider.capsule_test(Vec3::new(50.0, 5.0, 50.0), 0.5, 1.8);
        assert!(contact.is_none());

        // Capsule on terrain.
        let contact = collider.capsule_test(Vec3::new(50.0, -0.2, 50.0), 0.5, 1.8);
        assert!(contact.is_some());
    }

    #[test]
    fn batch_height_query() {
        let collider = make_flat_collider();
        let positions = vec![(50.0, 50.0), (25.0, 25.0), (-1.0, 50.0)];
        let results = collider.batch_height_query(&positions);
        assert!(results[0].is_some());
        assert!(results[1].is_some());
        assert!(results[2].is_none());
    }

    #[test]
    fn collider_statistics() {
        let collider = make_flat_collider();
        assert_eq!(collider.cell_count(), 16 * 16);
        assert_eq!(collider.solid_cell_count(), 16 * 16);
        assert_eq!(collider.dimensions(), (17, 17));
    }

    #[test]
    fn walkability_map() {
        let collider = make_flat_collider();
        let map = collider.compute_walkability_map();
        assert_eq!(map.len(), 16);
        assert_eq!(map[0].len(), 16);
        // Flat terrain should be all walkable.
        assert!(map[8][8]);
    }

    #[test]
    fn collider_origin() {
        let mut collider = make_flat_collider();
        collider.set_origin(Vec3::new(100.0, 50.0, 100.0));
        let h = collider.height_at_world(150.0, 150.0);
        assert!(h.is_some());
        assert!((h.unwrap() - 50.0).abs() < 0.01);
    }

    #[test]
    fn material_presets() {
        let dirt = PhysicsMaterial::dirt();
        assert!(dirt.static_friction > 0.5);
        assert!(dirt.sound_id.is_some());

        let ice = PhysicsMaterial::ice();
        assert!(ice.slippery);
        assert!(ice.static_friction < 0.2);
    }
}
