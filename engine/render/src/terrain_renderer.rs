// engine/render/src/terrain_renderer.rs
//
// GPU terrain rendering system for the Genovo engine.
//
// Implements a high-performance terrain renderer using:
//
// - Clipmap-based continuous level-of-detail (LOD) for efficient GPU rendering.
// - Hardware tessellation parameters for adaptive detail based on camera distance.
// - Height-based material blending with smooth transitions.
// - Triplanar texture mapping for steep cliffs and overhangs.
// - Detail normal maps at multiple scales for close-up surface quality.
// - Distance fog integration for large terrain views.
// - Terrain self-shadowing using horizon-based ambient occlusion.
//
// # Pipeline overview
//
// 1. **Clipmap update** — Select the appropriate LOD rings centred on the camera.
// 2. **Tessellation** — Use hull/domain shaders (or compute-based tessellation)
//    to adaptively subdivide terrain patches based on screen-space error.
// 3. **Material blending** — Sample the splatmap and blend terrain materials
//    (grass, rock, sand, snow) based on height, slope, and painted weights.
// 4. **Shading** — Apply PBR lighting, normal mapping, triplanar projection,
//    and fog.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Clipmap LOD
// ---------------------------------------------------------------------------

/// Clipmap LOD level configuration.
///
/// A clipmap is a set of concentric grid rings around the camera, each at
/// a different resolution. Closer rings have higher resolution.
#[derive(Debug, Clone)]
pub struct ClipmapLevel {
    /// LOD level (0 = highest detail, nearest to camera).
    pub level: u32,
    /// Grid cell size at this LOD level (world units).
    pub cell_size: f32,
    /// Number of grid cells along one axis of the ring.
    pub grid_size: u32,
    /// Inner radius (start of this ring) in world units.
    pub inner_radius: f32,
    /// Outer radius (end of this ring) in world units.
    pub outer_radius: f32,
    /// Whether this level is currently active.
    pub active: bool,
    /// Snapped grid origin (world-space XZ, snapped to cell_size).
    pub origin: [f32; 2],
    /// Transition blend width (for seamless LOD transitions).
    pub blend_width: f32,
}

impl ClipmapLevel {
    /// Creates a new clipmap level.
    pub fn new(level: u32, base_cell_size: f32, grid_size: u32) -> Self {
        let scale = (1u32 << level) as f32;
        let cell_size = base_cell_size * scale;
        let half_extent = cell_size * grid_size as f32 * 0.5;
        let inner_radius = if level == 0 { 0.0 } else { half_extent * 0.5 };

        Self {
            level,
            cell_size,
            grid_size,
            inner_radius,
            outer_radius: half_extent,
            active: true,
            origin: [0.0, 0.0],
            blend_width: cell_size * 2.0,
        }
    }

    /// Updates the origin to snap to the grid based on camera position.
    pub fn update_origin(&mut self, camera_x: f32, camera_z: f32) {
        self.origin[0] = (camera_x / self.cell_size).floor() * self.cell_size;
        self.origin[1] = (camera_z / self.cell_size).floor() * self.cell_size;
    }

    /// Returns the world-space position of a grid vertex.
    pub fn vertex_position(&self, grid_x: u32, grid_z: u32) -> [f32; 2] {
        let half = self.grid_size as f32 * 0.5;
        let x = self.origin[0] + (grid_x as f32 - half) * self.cell_size;
        let z = self.origin[1] + (grid_z as f32 - half) * self.cell_size;
        [x, z]
    }

    /// Computes the LOD transition blend factor at a given distance.
    ///
    /// Returns 1.0 at the inner boundary (full current LOD) and 0.0 at the
    /// outer boundary (transitioning to the next LOD).
    pub fn transition_factor(&self, distance: f32) -> f32 {
        if self.blend_width < 1e-6 {
            return 1.0;
        }
        let blend_start = self.outer_radius - self.blend_width;
        if distance <= blend_start {
            1.0
        } else if distance >= self.outer_radius {
            0.0
        } else {
            1.0 - (distance - blend_start) / self.blend_width
        }
    }

    /// Returns the total number of grid vertices at this level.
    pub fn vertex_count(&self) -> u32 {
        (self.grid_size + 1) * (self.grid_size + 1)
    }

    /// Returns the number of triangles in this level's mesh.
    pub fn triangle_count(&self) -> u32 {
        self.grid_size * self.grid_size * 2
    }
}

/// Manages the complete clipmap LOD hierarchy.
#[derive(Debug)]
pub struct TerrainClipmap {
    /// LOD levels, ordered from finest (0) to coarsest.
    pub levels: Vec<ClipmapLevel>,
    /// Base cell size (finest LOD).
    pub base_cell_size: f32,
    /// Grid size per level.
    pub grid_size: u32,
    /// Number of LOD levels.
    pub num_levels: u32,
    /// Camera position (for LOD selection).
    pub camera_position: [f32; 3],
}

impl TerrainClipmap {
    /// Creates a new terrain clipmap.
    ///
    /// # Arguments
    /// * `base_cell_size` — Cell size at the finest LOD (e.g. 1.0 metre).
    /// * `grid_size` — Number of cells per axis per ring (e.g. 64).
    /// * `num_levels` — Number of LOD levels (e.g. 6).
    pub fn new(base_cell_size: f32, grid_size: u32, num_levels: u32) -> Self {
        let levels = (0..num_levels)
            .map(|i| ClipmapLevel::new(i, base_cell_size, grid_size))
            .collect();

        Self {
            levels,
            base_cell_size,
            grid_size,
            num_levels,
            camera_position: [0.0, 0.0, 0.0],
        }
    }

    /// Updates all clipmap levels based on the current camera position.
    pub fn update(&mut self, camera_x: f32, camera_y: f32, camera_z: f32) {
        self.camera_position = [camera_x, camera_y, camera_z];
        for level in &mut self.levels {
            level.update_origin(camera_x, camera_z);
        }
    }

    /// Returns the LOD level for a given world-space distance from the camera.
    pub fn lod_for_distance(&self, distance: f32) -> u32 {
        for level in &self.levels {
            if distance <= level.outer_radius {
                return level.level;
            }
        }
        self.num_levels.saturating_sub(1)
    }

    /// Returns the total number of vertices across all active levels.
    pub fn total_vertex_count(&self) -> u32 {
        self.levels
            .iter()
            .filter(|l| l.active)
            .map(|l| l.vertex_count())
            .sum()
    }

    /// Returns the total number of triangles across all active levels.
    pub fn total_triangle_count(&self) -> u32 {
        self.levels
            .iter()
            .filter(|l| l.active)
            .map(|l| l.triangle_count())
            .sum()
    }

    /// Returns the maximum render distance.
    pub fn max_render_distance(&self) -> f32 {
        self.levels
            .last()
            .map(|l| l.outer_radius)
            .unwrap_or(0.0)
    }
}

impl Default for TerrainClipmap {
    fn default() -> Self {
        Self::new(1.0, 64, 6)
    }
}

// ---------------------------------------------------------------------------
// Tessellation
// ---------------------------------------------------------------------------

/// Hardware tessellation configuration for terrain patches.
#[derive(Debug, Clone)]
pub struct TerrainTessellation {
    /// Whether hardware tessellation is enabled.
    pub enabled: bool,
    /// Minimum tessellation factor (at maximum distance).
    pub min_factor: f32,
    /// Maximum tessellation factor (at minimum distance).
    pub max_factor: f32,
    /// Distance at which minimum tessellation begins.
    pub min_distance: f32,
    /// Distance at which maximum tessellation is used.
    pub max_distance: f32,
    /// Target edge length in screen-space pixels.
    pub target_edge_pixels: f32,
    /// Screen height (for screen-space error computation).
    pub screen_height: f32,
    /// Vertical FOV in radians.
    pub fov_y: f32,
    /// Whether to use height displacement in tessellation.
    pub displacement_enabled: bool,
    /// Height displacement scale.
    pub displacement_scale: f32,
}

impl TerrainTessellation {
    /// Creates default tessellation settings.
    pub fn new() -> Self {
        Self {
            enabled: true,
            min_factor: 1.0,
            max_factor: 64.0,
            min_distance: 10.0,
            max_distance: 500.0,
            target_edge_pixels: 16.0,
            screen_height: 1080.0,
            fov_y: PI / 3.0,
            displacement_enabled: true,
            displacement_scale: 1.0,
        }
    }

    /// Computes the tessellation factor for a given distance from the camera.
    pub fn factor_for_distance(&self, distance: f32) -> f32 {
        if !self.enabled {
            return 1.0;
        }

        if distance <= self.min_distance {
            return self.max_factor;
        }
        if distance >= self.max_distance {
            return self.min_factor;
        }

        // Linear interpolation in log space for smoother transitions.
        let t = ((distance - self.min_distance) / (self.max_distance - self.min_distance))
            .clamp(0.0, 1.0);
        let log_min = self.min_factor.ln();
        let log_max = self.max_factor.ln();
        (log_max + t * (log_min - log_max)).exp()
    }

    /// Computes the tessellation factor based on screen-space edge length.
    ///
    /// # Arguments
    /// * `edge_length` — World-space edge length of the patch.
    /// * `distance` — Distance from camera to patch centre.
    pub fn factor_from_screen_error(&self, edge_length: f32, distance: f32) -> f32 {
        if !self.enabled || distance < 1e-6 {
            return self.max_factor;
        }

        let projected_length =
            edge_length * self.screen_height / (2.0 * distance * (self.fov_y * 0.5).tan());
        let factor = projected_length / self.target_edge_pixels;

        factor.clamp(self.min_factor, self.max_factor)
    }

    /// Computes tessellation factors for a quad patch (4 edges + 2 inner).
    ///
    /// # Arguments
    /// * `edge_distances` — Distance from camera to the centre of each of the 4 edges.
    /// * `edge_lengths` — World-space length of each of the 4 edges.
    ///
    /// # Returns
    /// `[edge0, edge1, edge2, edge3, inner_u, inner_v]`
    pub fn compute_patch_factors(
        &self,
        edge_distances: &[f32; 4],
        edge_lengths: &[f32; 4],
    ) -> [f32; 6] {
        let mut factors = [1.0f32; 6];

        for i in 0..4 {
            factors[i] = self.factor_from_screen_error(edge_lengths[i], edge_distances[i]);
        }

        // Inner factors: average of edges.
        factors[4] = (factors[0] + factors[2]) * 0.5;
        factors[5] = (factors[1] + factors[3]) * 0.5;

        factors
    }
}

impl Default for TerrainTessellation {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Terrain material
// ---------------------------------------------------------------------------

/// A single terrain material layer (e.g. grass, rock, sand).
#[derive(Debug, Clone)]
pub struct TerrainMaterial {
    /// Material name.
    pub name: String,
    /// Albedo (diffuse) colour tint.
    pub albedo_tint: [f32; 3],
    /// Roughness value [0, 1].
    pub roughness: f32,
    /// Metallic value [0, 1].
    pub metallic: f32,
    /// Normal map strength [0, 1].
    pub normal_strength: f32,
    /// UV tiling scale for this material.
    pub uv_scale: f32,
    /// Height-based blend: minimum altitude for this material.
    pub height_min: f32,
    /// Height-based blend: maximum altitude for this material.
    pub height_max: f32,
    /// Slope-based blend: minimum slope angle (radians) for this material.
    /// Slope = 0 means flat, PI/2 means vertical.
    pub slope_min: f32,
    /// Slope-based blend: maximum slope angle (radians).
    pub slope_max: f32,
    /// Blend sharpness (higher = sharper transitions).
    pub blend_sharpness: f32,
    /// Whether to use triplanar mapping for this material.
    pub use_triplanar: bool,
    /// Triplanar blend sharpness.
    pub triplanar_sharpness: f32,
    /// Displacement/height map scale (for tessellation displacement).
    pub displacement_scale: f32,
    /// Detail normal scale (UV multiplier for the detail normal map).
    pub detail_normal_scale: f32,
    /// Detail normal strength.
    pub detail_normal_strength: f32,
    /// Detail distance: beyond this, detail normals fade out.
    pub detail_fade_distance: f32,
}

impl TerrainMaterial {
    /// Creates a new terrain material with default settings.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            albedo_tint: [1.0, 1.0, 1.0],
            roughness: 0.7,
            metallic: 0.0,
            normal_strength: 1.0,
            uv_scale: 1.0,
            height_min: f32::NEG_INFINITY,
            height_max: f32::INFINITY,
            slope_min: 0.0,
            slope_max: PI / 2.0,
            blend_sharpness: 4.0,
            use_triplanar: false,
            triplanar_sharpness: 8.0,
            displacement_scale: 0.0,
            detail_normal_scale: 4.0,
            detail_normal_strength: 0.5,
            detail_fade_distance: 50.0,
        }
    }

    /// Creates a grass material preset.
    pub fn grass() -> Self {
        Self {
            name: "grass".into(),
            albedo_tint: [0.35, 0.55, 0.15],
            roughness: 0.85,
            metallic: 0.0,
            normal_strength: 0.8,
            uv_scale: 4.0,
            height_min: 0.0,
            height_max: 500.0,
            slope_min: 0.0,
            slope_max: 0.7,
            blend_sharpness: 4.0,
            use_triplanar: false,
            triplanar_sharpness: 8.0,
            displacement_scale: 0.02,
            detail_normal_scale: 8.0,
            detail_normal_strength: 0.3,
            detail_fade_distance: 30.0,
        }
    }

    /// Creates a rock/cliff material preset.
    pub fn rock() -> Self {
        Self {
            name: "rock".into(),
            albedo_tint: [0.45, 0.42, 0.38],
            roughness: 0.9,
            metallic: 0.0,
            normal_strength: 1.0,
            uv_scale: 2.0,
            height_min: 0.0,
            height_max: f32::INFINITY,
            slope_min: 0.6,
            slope_max: PI / 2.0,
            blend_sharpness: 6.0,
            use_triplanar: true,
            triplanar_sharpness: 8.0,
            displacement_scale: 0.05,
            detail_normal_scale: 4.0,
            detail_normal_strength: 0.6,
            detail_fade_distance: 40.0,
        }
    }

    /// Creates a sand material preset.
    pub fn sand() -> Self {
        Self {
            name: "sand".into(),
            albedo_tint: [0.85, 0.75, 0.55],
            roughness: 0.75,
            metallic: 0.0,
            normal_strength: 0.5,
            uv_scale: 6.0,
            height_min: -10.0,
            height_max: 50.0,
            slope_min: 0.0,
            slope_max: 0.5,
            blend_sharpness: 3.0,
            use_triplanar: false,
            triplanar_sharpness: 4.0,
            displacement_scale: 0.01,
            detail_normal_scale: 12.0,
            detail_normal_strength: 0.2,
            detail_fade_distance: 20.0,
        }
    }

    /// Creates a snow material preset.
    pub fn snow() -> Self {
        Self {
            name: "snow".into(),
            albedo_tint: [0.95, 0.95, 0.98],
            roughness: 0.3,
            metallic: 0.0,
            normal_strength: 0.3,
            uv_scale: 3.0,
            height_min: 400.0,
            height_max: f32::INFINITY,
            slope_min: 0.0,
            slope_max: 0.8,
            blend_sharpness: 5.0,
            use_triplanar: false,
            triplanar_sharpness: 4.0,
            displacement_scale: 0.03,
            detail_normal_scale: 6.0,
            detail_normal_strength: 0.4,
            detail_fade_distance: 40.0,
        }
    }

    /// Computes the weight of this material at a given height and slope.
    ///
    /// # Arguments
    /// * `height` — World-space height (Y).
    /// * `slope` — Surface slope angle in radians.
    ///
    /// # Returns
    /// Weight in [0, 1].
    pub fn weight_at(&self, height: f32, slope: f32) -> f32 {
        // Height weight.
        let height_weight = if height < self.height_min || height > self.height_max {
            0.0
        } else {
            let blend_range = 1.0 / self.blend_sharpness.max(0.01);
            let low = smooth_step(self.height_min, self.height_min + blend_range, height);
            let high = 1.0 - smooth_step(self.height_max - blend_range, self.height_max, height);
            low * high
        };

        // Slope weight.
        let slope_weight = {
            let blend_range = 0.1 / self.blend_sharpness.max(0.01);
            let low = smooth_step(self.slope_min - blend_range, self.slope_min + blend_range, slope);
            let high = 1.0
                - smooth_step(self.slope_max - blend_range, self.slope_max + blend_range, slope);
            low * high
        };

        (height_weight * slope_weight).clamp(0.0, 1.0)
    }
}

impl Default for TerrainMaterial {
    fn default() -> Self {
        Self::new("default")
    }
}

/// Smoothstep function.
#[inline]
fn smooth_step(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Triplanar mapping
// ---------------------------------------------------------------------------

/// Triplanar texture mapping weights.
///
/// Used for steep terrain (cliffs) where standard UV mapping would stretch.
/// Projects the texture along all three axes and blends based on the surface
/// normal.
#[derive(Debug, Clone, Copy)]
pub struct TriplanarWeights {
    /// Weight for the X-axis projection.
    pub x: f32,
    /// Weight for the Y-axis projection.
    pub y: f32,
    /// Weight for the Z-axis projection.
    pub z: f32,
}

impl TriplanarWeights {
    /// Computes triplanar blending weights from a world-space normal.
    ///
    /// # Arguments
    /// * `normal` — World-space surface normal (normalised).
    /// * `sharpness` — Blend sharpness (higher = sharper transitions).
    pub fn from_normal(normal: [f32; 3], sharpness: f32) -> Self {
        let nx = normal[0].abs();
        let ny = normal[1].abs();
        let nz = normal[2].abs();

        // Power-of-N blending.
        let mut x = nx.powf(sharpness);
        let mut y = ny.powf(sharpness);
        let mut z = nz.powf(sharpness);

        // Normalise.
        let total = x + y + z;
        if total > 1e-6 {
            let inv = 1.0 / total;
            x *= inv;
            y *= inv;
            z *= inv;
        } else {
            x = 0.0;
            y = 1.0;
            z = 0.0;
        }

        Self { x, y, z }
    }

    /// Computes triplanar UV coordinates for a given world position.
    ///
    /// # Returns
    /// Three UV pairs: `(uv_yz, uv_xz, uv_xy)` — one for each projection axis.
    pub fn compute_uvs(
        world_pos: [f32; 3],
        uv_scale: f32,
    ) -> ([f32; 2], [f32; 2], [f32; 2]) {
        let uv_yz = [world_pos[1] * uv_scale, world_pos[2] * uv_scale]; // X-axis projection
        let uv_xz = [world_pos[0] * uv_scale, world_pos[2] * uv_scale]; // Y-axis projection
        let uv_xy = [world_pos[0] * uv_scale, world_pos[1] * uv_scale]; // Z-axis projection
        (uv_yz, uv_xz, uv_xy)
    }

    /// Blends three colour samples using the triplanar weights.
    pub fn blend_colors(
        &self,
        color_x: [f32; 3],
        color_y: [f32; 3],
        color_z: [f32; 3],
    ) -> [f32; 3] {
        [
            color_x[0] * self.x + color_y[0] * self.y + color_z[0] * self.z,
            color_x[1] * self.x + color_y[1] * self.y + color_z[1] * self.z,
            color_x[2] * self.x + color_y[2] * self.y + color_z[2] * self.z,
        ]
    }

    /// Blends three normal vectors using the triplanar weights.
    ///
    /// Uses UDN (Unreal-style) normal blending for correct results.
    pub fn blend_normals(
        &self,
        normal_x: [f32; 3],
        normal_y: [f32; 3],
        normal_z: [f32; 3],
        world_normal: [f32; 3],
    ) -> [f32; 3] {
        // Swizzle normals to world-space axes and blend.
        let nx = [world_normal[0].signum() * normal_x[2], normal_x[1], normal_x[0]];
        let ny = [normal_y[0], world_normal[1].signum() * normal_y[2], normal_y[1]];
        let nz = [normal_z[0], normal_z[1], world_normal[2].signum() * normal_z[2]];

        let blended = [
            nx[0] * self.x + ny[0] * self.y + nz[0] * self.z,
            nx[1] * self.x + ny[1] * self.y + nz[1] * self.z,
            nx[2] * self.x + ny[2] * self.y + nz[2] * self.z,
        ];

        // Normalise.
        let len = (blended[0] * blended[0] + blended[1] * blended[1] + blended[2] * blended[2])
            .sqrt();
        if len > 1e-6 {
            [blended[0] / len, blended[1] / len, blended[2] / len]
        } else {
            world_normal
        }
    }
}

// ---------------------------------------------------------------------------
// Terrain heightmap
// ---------------------------------------------------------------------------

/// A terrain heightmap with bilinear sampling.
#[derive(Debug)]
pub struct TerrainHeightmap {
    /// Heightmap resolution (width).
    pub width: u32,
    /// Heightmap resolution (height).
    pub height: u32,
    /// World-space size of the terrain.
    pub world_size: [f32; 2],
    /// World-space origin (min XZ corner).
    pub origin: [f32; 2],
    /// Minimum height.
    pub min_height: f32,
    /// Maximum height.
    pub max_height: f32,
    /// Height data (row-major, bottom-to-top).
    data: Vec<f32>,
}

impl TerrainHeightmap {
    /// Creates a heightmap from raw data.
    pub fn new(
        width: u32,
        height: u32,
        world_size: [f32; 2],
        origin: [f32; 2],
        data: Vec<f32>,
    ) -> Self {
        let min_height = data.iter().cloned().fold(f32::MAX, f32::min);
        let max_height = data.iter().cloned().fold(f32::MIN, f32::max);
        Self {
            width,
            height,
            world_size,
            origin,
            min_height,
            max_height,
            data,
        }
    }

    /// Creates a flat heightmap at a given elevation.
    pub fn flat(width: u32, height: u32, world_size: [f32; 2], elevation: f32) -> Self {
        let total = (width * height) as usize;
        Self {
            width,
            height,
            world_size,
            origin: [0.0, 0.0],
            min_height: elevation,
            max_height: elevation,
            data: vec![elevation; total],
        }
    }

    /// Creates a procedural heightmap using fractal noise.
    pub fn procedural(
        width: u32,
        height: u32,
        world_size: [f32; 2],
        frequency: f32,
        amplitude: f32,
        octaves: u32,
    ) -> Self {
        let mut data = Vec::with_capacity((width * height) as usize);
        let inv_w = 1.0 / width as f32;
        let inv_h = 1.0 / height as f32;

        for z in 0..height {
            for x in 0..width {
                let nx = x as f32 * inv_w * frequency;
                let nz = z as f32 * inv_h * frequency;
                let h = fbm_2d(nx, nz, octaves, 2.0, 0.5) * amplitude;
                data.push(h);
            }
        }

        let min_height = data.iter().cloned().fold(f32::MAX, f32::min);
        let max_height = data.iter().cloned().fold(f32::MIN, f32::max);

        Self {
            width,
            height,
            world_size,
            origin: [0.0, 0.0],
            min_height,
            max_height,
            data,
        }
    }

    /// Samples the heightmap with bilinear interpolation.
    ///
    /// # Arguments
    /// * `world_x` — World-space X coordinate.
    /// * `world_z` — World-space Z coordinate.
    ///
    /// # Returns
    /// Interpolated height value.
    pub fn sample(&self, world_x: f32, world_z: f32) -> f32 {
        let u = (world_x - self.origin[0]) / self.world_size[0];
        let v = (world_z - self.origin[1]) / self.world_size[1];

        let u = u.clamp(0.0, 1.0);
        let v = v.clamp(0.0, 1.0);

        let fx = u * (self.width - 1) as f32;
        let fz = v * (self.height - 1) as f32;

        let x0 = (fx as u32).min(self.width - 2);
        let z0 = (fz as u32).min(self.height - 2);
        let x1 = x0 + 1;
        let z1 = z0 + 1;

        let tx = fx - x0 as f32;
        let tz = fz - z0 as f32;

        let h00 = self.data[(z0 * self.width + x0) as usize];
        let h10 = self.data[(z0 * self.width + x1) as usize];
        let h01 = self.data[(z1 * self.width + x0) as usize];
        let h11 = self.data[(z1 * self.width + x1) as usize];

        let h0 = h00 + (h10 - h00) * tx;
        let h1 = h01 + (h11 - h01) * tx;

        h0 + (h1 - h0) * tz
    }

    /// Computes the surface normal at a given world position using finite differences.
    pub fn normal_at(&self, world_x: f32, world_z: f32) -> [f32; 3] {
        let dx = self.world_size[0] / self.width as f32;
        let dz = self.world_size[1] / self.height as f32;

        let hl = self.sample(world_x - dx, world_z);
        let hr = self.sample(world_x + dx, world_z);
        let hd = self.sample(world_x, world_z - dz);
        let hu = self.sample(world_x, world_z + dz);

        let nx = hl - hr;
        let nz = hd - hu;
        let ny = 2.0 * dx;

        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        if len > 1e-6 {
            [nx / len, ny / len, nz / len]
        } else {
            [0.0, 1.0, 0.0]
        }
    }

    /// Computes the slope angle (in radians) at a given position.
    pub fn slope_at(&self, world_x: f32, world_z: f32) -> f32 {
        let n = self.normal_at(world_x, world_z);
        n[1].clamp(-1.0, 1.0).acos()
    }

    /// Returns the raw height data.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Returns the height range.
    pub fn height_range(&self) -> (f32, f32) {
        (self.min_height, self.max_height)
    }

    /// Returns memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

/// 2D fBm noise for heightmap generation.
fn fbm_2d(x: f32, z: f32, octaves: u32, lacunarity: f32, persistence: f32) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amp = 0.0;

    for _ in 0..octaves {
        // Use the 3D noise with y=0 for 2D.
        let n = noise_2d(x * frequency, z * frequency);
        value += n * amplitude;
        max_amp += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    if max_amp > 0.0 {
        value / max_amp
    } else {
        0.0
    }
}

/// Simple 2D value noise.
fn noise_2d(x: f32, z: f32) -> f32 {
    let xi = x.floor() as i32;
    let zi = z.floor() as i32;
    let xf = x - x.floor();
    let zf = z - z.floor();
    let u = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0);
    let w = zf * zf * zf * (zf * (zf * 6.0 - 15.0) + 10.0);

    let hash = |x: i32, z: i32| -> f32 {
        let n = x.wrapping_mul(374761393).wrapping_add(z.wrapping_mul(668265263));
        let n = n ^ (n >> 13);
        let n = n.wrapping_mul(n.wrapping_mul(n.wrapping_mul(60493).wrapping_add(19990303)).wrapping_add(1376312589));
        (n & 0x7FFF_FFFF) as f32 / 0x7FFF_FFFF as f32 * 2.0 - 1.0
    };

    let c00 = hash(xi, zi);
    let c10 = hash(xi + 1, zi);
    let c01 = hash(xi, zi + 1);
    let c11 = hash(xi + 1, zi + 1);

    let x0 = c00 + (c10 - c00) * u;
    let x1 = c01 + (c11 - c01) * u;

    x0 + (x1 - x0) * w
}

// ---------------------------------------------------------------------------
// Material blending
// ---------------------------------------------------------------------------

/// Blended material result from the splatmap.
#[derive(Debug, Clone)]
pub struct BlendedMaterial {
    /// Blended albedo colour.
    pub albedo: [f32; 3],
    /// Blended roughness.
    pub roughness: f32,
    /// Blended metallic.
    pub metallic: f32,
    /// Blended normal (world-space).
    pub normal: [f32; 3],
}

/// Blends terrain materials at a given position based on height, slope, and splatmap.
///
/// # Arguments
/// * `materials` — Available terrain materials.
/// * `height` — World-space height (Y).
/// * `slope` — Surface slope angle in radians.
/// * `splatmap_weights` — Optional per-material weights from a painted splatmap.
/// * `world_normal` — Surface normal (world-space).
///
/// # Returns
/// Blended material properties.
pub fn blend_terrain_materials(
    materials: &[TerrainMaterial],
    height: f32,
    slope: f32,
    splatmap_weights: Option<&[f32]>,
    world_normal: [f32; 3],
) -> BlendedMaterial {
    if materials.is_empty() {
        return BlendedMaterial {
            albedo: [0.5, 0.5, 0.5],
            roughness: 0.5,
            metallic: 0.0,
            normal: world_normal,
        };
    }

    let mut total_weight = 0.0;
    let mut albedo = [0.0f32; 3];
    let mut roughness = 0.0f32;
    let mut metallic = 0.0f32;

    for (i, mat) in materials.iter().enumerate() {
        // Combine automatic (height/slope) weight with splatmap weight.
        let auto_weight = mat.weight_at(height, slope);
        let splat_weight = splatmap_weights
            .and_then(|w| w.get(i))
            .copied()
            .unwrap_or(1.0);
        let weight = auto_weight * splat_weight;

        if weight < 1e-6 {
            continue;
        }

        albedo[0] += mat.albedo_tint[0] * weight;
        albedo[1] += mat.albedo_tint[1] * weight;
        albedo[2] += mat.albedo_tint[2] * weight;
        roughness += mat.roughness * weight;
        metallic += mat.metallic * weight;
        total_weight += weight;
    }

    if total_weight > 1e-6 {
        let inv = 1.0 / total_weight;
        albedo[0] *= inv;
        albedo[1] *= inv;
        albedo[2] *= inv;
        roughness *= inv;
        metallic *= inv;
    } else {
        albedo = [0.5, 0.5, 0.5];
        roughness = 0.5;
    }

    BlendedMaterial {
        albedo,
        roughness,
        metallic,
        normal: world_normal,
    }
}

// ---------------------------------------------------------------------------
// Distance fog
// ---------------------------------------------------------------------------

/// Terrain-specific fog settings.
#[derive(Debug, Clone)]
pub struct TerrainFog {
    /// Whether fog is enabled.
    pub enabled: bool,
    /// Fog colour (linear RGB).
    pub color: [f32; 3],
    /// Fog density (exponential coefficient).
    pub density: f32,
    /// Height fog density.
    pub height_density: f32,
    /// Height fog falloff.
    pub height_falloff: f32,
    /// Height fog base altitude.
    pub height_base: f32,
    /// Start distance for fog.
    pub start_distance: f32,
    /// Maximum fog opacity.
    pub max_opacity: f32,
}

impl TerrainFog {
    /// Creates default terrain fog.
    pub fn new() -> Self {
        Self {
            enabled: true,
            color: [0.7, 0.75, 0.8],
            density: 0.001,
            height_density: 0.05,
            height_falloff: 0.1,
            height_base: 0.0,
            start_distance: 100.0,
            max_opacity: 0.95,
        }
    }

    /// Computes the fog factor for a given distance and height.
    pub fn compute(&self, distance: f32, height: f32) -> f32 {
        if !self.enabled {
            return 0.0;
        }

        let effective_dist = (distance - self.start_distance).max(0.0);

        // Global exponential fog.
        let global_fog = 1.0 - (-self.density * effective_dist).exp();

        // Height-based fog.
        let height_diff = self.height_base - height;
        let height_fog = if height_diff > 0.0 {
            let factor = 1.0 - (-height_diff * self.height_falloff).exp();
            self.height_density * factor
        } else {
            0.0
        };

        let combined = global_fog + height_fog * (1.0 - global_fog);
        combined.clamp(0.0, self.max_opacity)
    }

    /// Applies fog to a colour.
    pub fn apply(&self, color: [f32; 3], distance: f32, height: f32) -> [f32; 3] {
        let fog = self.compute(distance, height);
        [
            color[0] * (1.0 - fog) + self.color[0] * fog,
            color[1] * (1.0 - fog) + self.color[1] * fog,
            color[2] * (1.0 - fog) + self.color[2] * fog,
        ]
    }
}

impl Default for TerrainFog {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Terrain shadow self-occlusion
// ---------------------------------------------------------------------------

/// Horizon-based terrain self-shadowing.
///
/// Precomputes the horizon angle for each point on the terrain from the
/// sun's direction. Points below the horizon are in self-shadow.
#[derive(Debug)]
pub struct TerrainSelfShadow {
    /// Shadow map resolution.
    pub resolution: u32,
    /// Computed horizon angles (per texel).
    horizon_angles: Vec<f32>,
    /// Shadow intensity [0, 1].
    pub intensity: f32,
    /// Bias angle (radians) to avoid self-shadowing artefacts.
    pub bias: f32,
}

impl TerrainSelfShadow {
    /// Creates a new self-shadow calculator.
    pub fn new(resolution: u32) -> Self {
        let total = (resolution * resolution) as usize;
        Self {
            resolution,
            horizon_angles: vec![0.0; total],
            intensity: 0.7,
            bias: 0.02,
        }
    }

    /// Computes the horizon map from a heightmap in the given light direction.
    ///
    /// # Arguments
    /// * `heightmap` — The terrain heightmap.
    /// * `light_dir` — Normalised direction TO the light source.
    pub fn compute(&mut self, heightmap: &TerrainHeightmap, light_dir: [f32; 3]) {
        let res = self.resolution;
        let dx = heightmap.world_size[0] / res as f32;
        let dz = heightmap.world_size[1] / res as f32;

        // Project light direction onto XZ plane.
        let light_xz_len = (light_dir[0] * light_dir[0] + light_dir[2] * light_dir[2]).sqrt();
        let light_elevation = if light_xz_len > 1e-6 {
            (light_dir[1] / light_xz_len).atan()
        } else {
            PI / 2.0
        };

        // Trace rays from each texel toward the light.
        for tz in 0..res {
            for tx in 0..res {
                let world_x = heightmap.origin[0] + (tx as f32 + 0.5) * dx;
                let world_z = heightmap.origin[1] + (tz as f32 + 0.5) * dz;
                let base_height = heightmap.sample(world_x, world_z);

                let mut max_angle = f32::NEG_INFINITY;
                let num_samples = 16;
                let step = dx * 2.0;

                for i in 1..=num_samples {
                    let t = i as f32 * step;
                    let sx = world_x + light_dir[0] * t / light_xz_len.max(0.01);
                    let sz = world_z + light_dir[2] * t / light_xz_len.max(0.01);
                    let sh = heightmap.sample(sx, sz);
                    let dh = sh - base_height;
                    let angle = (dh / t).atan();
                    if angle > max_angle {
                        max_angle = angle;
                    }
                }

                let idx = (tz * res + tx) as usize;
                self.horizon_angles[idx] = max_angle;
            }
        }
    }

    /// Queries the shadow factor at a given texel.
    ///
    /// Returns: 1.0 = fully lit, 0.0 = fully shadowed.
    pub fn shadow_factor(&self, tx: u32, tz: u32, light_elevation: f32) -> f32 {
        let idx = (tz * self.resolution + tx) as usize;
        let horizon = self.horizon_angles.get(idx).copied().unwrap_or(0.0);

        if light_elevation > horizon + self.bias {
            1.0
        } else {
            1.0 - self.intensity
        }
    }

    /// Returns the horizon angle data for GPU upload.
    pub fn horizon_data(&self) -> &[f32] {
        &self.horizon_angles
    }
}

// ---------------------------------------------------------------------------
// TerrainRenderer (top-level)
// ---------------------------------------------------------------------------

/// Top-level terrain rendering system.
pub struct TerrainRenderer {
    /// Clipmap LOD hierarchy.
    pub clipmap: TerrainClipmap,
    /// Tessellation settings.
    pub tessellation: TerrainTessellation,
    /// Terrain materials.
    pub materials: Vec<TerrainMaterial>,
    /// Terrain fog.
    pub fog: TerrainFog,
    /// Self-shadow calculator.
    pub self_shadow: TerrainSelfShadow,
    /// Whether the terrain is visible.
    pub enabled: bool,
    /// Whether wireframe mode is on (for debugging).
    pub wireframe: bool,
    /// Whether to show the clipmap LOD colouring (for debugging).
    pub debug_lod: bool,
}

impl TerrainRenderer {
    /// Creates a new terrain renderer with default settings.
    pub fn new() -> Self {
        Self {
            clipmap: TerrainClipmap::default(),
            tessellation: TerrainTessellation::default(),
            materials: vec![
                TerrainMaterial::grass(),
                TerrainMaterial::rock(),
                TerrainMaterial::sand(),
                TerrainMaterial::snow(),
            ],
            fog: TerrainFog::default(),
            self_shadow: TerrainSelfShadow::new(256),
            enabled: true,
            wireframe: false,
            debug_lod: false,
        }
    }

    /// Creates with custom clipmap settings.
    pub fn with_clipmap(mut self, base_cell_size: f32, grid_size: u32, num_levels: u32) -> Self {
        self.clipmap = TerrainClipmap::new(base_cell_size, grid_size, num_levels);
        self
    }

    /// Updates the terrain for the current frame.
    pub fn update(&mut self, camera_pos: [f32; 3]) {
        self.clipmap
            .update(camera_pos[0], camera_pos[1], camera_pos[2]);
    }

    /// Returns the total triangle count across all LOD levels.
    pub fn triangle_count(&self) -> u32 {
        self.clipmap.total_triangle_count()
    }

    /// Returns the number of materials.
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    /// Adds a material.
    pub fn add_material(&mut self, material: TerrainMaterial) {
        self.materials.push(material);
    }
}

impl Default for TerrainRenderer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WGSL terrain shader
// ---------------------------------------------------------------------------

/// WGSL vertex + fragment shader for terrain rendering with material blending.
pub const TERRAIN_RENDER_WGSL: &str = r#"
// -----------------------------------------------------------------------
// Terrain rendering shader (Genovo Engine)
// -----------------------------------------------------------------------

struct TerrainUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    sun_direction: vec3<f32>,
    sun_intensity: f32,
    sun_color: vec3<f32>,
    fog_density: f32,
    fog_color: vec3<f32>,
    fog_start: f32,
    terrain_size: vec2<f32>,
    terrain_origin: vec2<f32>,
    tess_factor: f32,
    _pad: vec3<f32>,
};

@group(0) @binding(0) var<uniform> terrain: TerrainUniforms;
@group(0) @binding(1) var heightmap_tex: texture_2d<f32>;
@group(0) @binding(2) var normalmap_tex: texture_2d<f32>;
@group(0) @binding(3) var splatmap_tex: texture_2d<f32>;
@group(0) @binding(4) var terrain_sampler: sampler;

@group(1) @binding(0) var albedo_array: texture_2d_array<f32>;
@group(1) @binding(1) var normal_array: texture_2d_array<f32>;
@group(1) @binding(2) var material_sampler: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) distance: f32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let terrain_uv = (input.position - terrain.terrain_origin) / terrain.terrain_size;
    let height = textureSampleLevel(heightmap_tex, terrain_sampler, terrain_uv, 0.0).r;

    let world_pos = vec3<f32>(input.position.x, height, input.position.y);
    out.clip_pos = terrain.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.uv = terrain_uv;

    let normal = textureSampleLevel(normalmap_tex, terrain_sampler, terrain_uv, 0.0).rgb;
    out.normal = normalize(normal * 2.0 - 1.0);

    out.distance = length(world_pos - terrain.camera_pos);

    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let splat = textureSample(splatmap_tex, terrain_sampler, input.uv);
    let uv_scale = 4.0;
    let tiled_uv = input.uv * uv_scale * terrain.terrain_size;

    // Sample material layers.
    let grass = textureSample(albedo_array, material_sampler, tiled_uv, 0).rgb;
    let rock = textureSample(albedo_array, material_sampler, tiled_uv, 1).rgb;
    let sand = textureSample(albedo_array, material_sampler, tiled_uv, 2).rgb;
    let snow = textureSample(albedo_array, material_sampler, tiled_uv, 3).rgb;

    // Blend based on splatmap.
    var albedo = grass * splat.r + rock * splat.g + sand * splat.b + snow * splat.a;

    // Simple diffuse lighting.
    let n_dot_l = max(dot(input.normal, terrain.sun_direction), 0.0);
    let diffuse = terrain.sun_color * terrain.sun_intensity * n_dot_l;
    let ambient = vec3<f32>(0.15, 0.18, 0.22);

    var color = albedo * (diffuse + ambient);

    // Distance fog.
    let fog_dist = max(input.distance - terrain.fog_start, 0.0);
    let fog_factor = 1.0 - exp(-terrain.fog_density * fog_dist);
    let fog = clamp(fog_factor, 0.0, 0.95);
    color = mix(color, terrain.fog_color, fog);

    return vec4<f32>(color, 1.0);
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clipmap_level_creation() {
        let level = ClipmapLevel::new(0, 1.0, 64);
        assert_eq!(level.level, 0);
        assert!((level.cell_size - 1.0).abs() < 0.01);
        assert_eq!(level.grid_size, 64);

        let level2 = ClipmapLevel::new(2, 1.0, 64);
        assert!((level2.cell_size - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_clipmap_update() {
        let mut clipmap = TerrainClipmap::new(1.0, 64, 4);
        clipmap.update(100.0, 50.0, 200.0);
        assert!((clipmap.camera_position[0] - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_tessellation_factor() {
        let tess = TerrainTessellation::new();
        let near = tess.factor_for_distance(5.0);
        let far = tess.factor_for_distance(1000.0);
        assert!(near > far, "Near should have higher tessellation");
        assert!((near - tess.max_factor).abs() < 0.01);
        assert!((far - tess.min_factor).abs() < 0.01);
    }

    #[test]
    fn test_tessellation_screen_error() {
        let tess = TerrainTessellation::new();
        let f1 = tess.factor_from_screen_error(10.0, 50.0);
        let f2 = tess.factor_from_screen_error(10.0, 200.0);
        assert!(f1 > f2, "Closer should have higher tessellation");
    }

    #[test]
    fn test_terrain_material_weights() {
        let grass = TerrainMaterial::grass();
        let w_flat_low = grass.weight_at(100.0, 0.0);
        let w_steep = grass.weight_at(100.0, 1.2);
        assert!(w_flat_low > w_steep, "Grass should prefer flat terrain");

        let rock = TerrainMaterial::rock();
        let w_rock_steep = rock.weight_at(100.0, 1.2);
        assert!(w_rock_steep > 0.0, "Rock should appear on steep slopes");
    }

    #[test]
    fn test_triplanar_weights() {
        let w = TriplanarWeights::from_normal([0.0, 1.0, 0.0], 8.0);
        assert!(w.y > 0.9, "Y weight should dominate for flat surface");

        let w2 = TriplanarWeights::from_normal([1.0, 0.0, 0.0], 8.0);
        assert!(w2.x > 0.9, "X weight should dominate for X-facing surface");
    }

    #[test]
    fn test_triplanar_colour_blend() {
        let w = TriplanarWeights { x: 0.5, y: 0.0, z: 0.5 };
        let blended = w.blend_colors([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        assert!((blended[0] - 0.5).abs() < 0.01);
        assert!((blended[2] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_heightmap_flat() {
        let hm = TerrainHeightmap::flat(16, 16, [100.0, 100.0], 10.0);
        let h = hm.sample(50.0, 50.0);
        assert!((h - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_heightmap_procedural() {
        let hm = TerrainHeightmap::procedural(32, 32, [100.0, 100.0], 1.0, 50.0, 4);
        assert!(hm.min_height < hm.max_height, "Should have height variation");
        let h = hm.sample(50.0, 50.0);
        assert!(h >= hm.min_height && h <= hm.max_height);
    }

    #[test]
    fn test_heightmap_normal() {
        let hm = TerrainHeightmap::flat(16, 16, [100.0, 100.0], 0.0);
        let n = hm.normal_at(50.0, 50.0);
        assert!(n[1] > 0.9, "Flat terrain should have upward normal");
    }

    #[test]
    fn test_terrain_fog() {
        let fog = TerrainFog::new();
        let f0 = fog.compute(0.0, 100.0);
        assert!(f0 < 0.01, "No fog at zero distance");

        let f_far = fog.compute(10000.0, 100.0);
        assert!(f_far > 0.5, "Should have significant fog at distance");
    }

    #[test]
    fn test_terrain_fog_apply() {
        let fog = TerrainFog::new();
        let color = fog.apply([1.0, 0.0, 0.0], 10000.0, 0.0);
        // Should shift toward fog colour.
        assert!(color[0] < 1.0, "Red should decrease with fog");
    }

    #[test]
    fn test_material_blending() {
        let materials = vec![TerrainMaterial::grass(), TerrainMaterial::rock()];
        let blended = blend_terrain_materials(
            &materials,
            100.0,
            0.1,
            None,
            [0.0, 1.0, 0.0],
        );
        // On flat low ground, grass should dominate.
        assert!(blended.albedo[1] > blended.albedo[0], "Should be green-ish");
    }

    #[test]
    fn test_terrain_renderer_creation() {
        let renderer = TerrainRenderer::new();
        assert!(renderer.enabled);
        assert_eq!(renderer.materials.len(), 4);
        assert!(renderer.triangle_count() > 0);
    }

    #[test]
    fn test_clipmap_transition() {
        let level = ClipmapLevel::new(0, 1.0, 64);
        let inside = level.transition_factor(10.0);
        assert!((inside - 1.0).abs() < 0.01, "Should be 1.0 inside");
        let outside = level.transition_factor(level.outer_radius + 10.0);
        assert!((outside - 0.0).abs() < 0.01, "Should be 0.0 outside");
    }

    #[test]
    fn test_self_shadow_creation() {
        let shadow = TerrainSelfShadow::new(64);
        assert_eq!(shadow.horizon_data().len(), 64 * 64);
    }

    #[test]
    fn test_smooth_step() {
        assert!((smooth_step(0.0, 1.0, 0.0) - 0.0).abs() < 0.01);
        assert!((smooth_step(0.0, 1.0, 1.0) - 1.0).abs() < 0.01);
        assert!((smooth_step(0.0, 1.0, 0.5) - 0.5).abs() < 0.01);
    }
}
