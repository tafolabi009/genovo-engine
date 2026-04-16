//! Vegetation and foliage scattering for terrain.
//!
//! Provides a system for distributing vegetation instances (trees, bushes,
//! grass) across the terrain surface. Uses Poisson disk sampling for natural
//! placement, supports density maps, slope/altitude filtering, instanced
//! rendering preparation, and LOD transitions from mesh to billboard.

use glam::{Quat, Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::heightmap::{Heightmap, SimpleRng};
use crate::texturing::SplatMap;

// ---------------------------------------------------------------------------
// VegetationLayer
// ---------------------------------------------------------------------------

/// Describes a class of vegetation (e.g. "Oak Trees", "Grass Patch").
///
/// Each layer controls what mesh/billboard to use, where it can spawn
/// (altitude, slope, splatmap channel constraints), and rendering parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VegetationLayer {
    /// Human-readable name.
    pub name: String,

    /// Path to the mesh asset for close-range rendering.
    pub mesh_path: String,

    /// Path to the billboard/impostor texture for distant rendering.
    pub billboard_path: String,

    /// Average instances per square world unit.
    pub density: f32,

    /// Minimum scale factor.
    pub scale_min: f32,

    /// Maximum scale factor.
    pub scale_max: f32,

    /// Minimum terrain slope (radians) where this vegetation can grow.
    pub slope_min: f32,

    /// Maximum terrain slope where this vegetation can grow.
    pub slope_max: f32,

    /// Minimum altitude (heightmap value, 0..1) where this vegetation can grow.
    pub altitude_min: f32,

    /// Maximum altitude.
    pub altitude_max: f32,

    /// Color variation: random hue/saturation shift range.
    pub color_variation: f32,

    /// Which splatmap channel(s) this layer prefers (0-3). -1 = no constraint.
    pub preferred_splatmap_channel: i32,

    /// Minimum splatmap weight required for this channel.
    pub min_splatmap_weight: f32,

    /// Wind animation parameters.
    pub wind: WindParams,

    /// LOD distances for mesh -> billboard -> cull transitions.
    pub lod: VegetationLOD,

    /// Whether instances should be aligned to the terrain normal (true)
    /// or always upright (false).
    pub align_to_normal: bool,

    /// Random rotation range in radians around Y axis.
    pub random_rotation: f32,

    /// Minimum distance between instances of this layer.
    pub min_spacing: f32,
}

impl Default for VegetationLayer {
    fn default() -> Self {
        Self {
            name: "Default Vegetation".into(),
            mesh_path: String::new(),
            billboard_path: String::new(),
            density: 0.05,
            scale_min: 0.8,
            scale_max: 1.2,
            slope_min: 0.0,
            slope_max: 0.7,
            altitude_min: 0.0,
            altitude_max: 1.0,
            color_variation: 0.1,
            preferred_splatmap_channel: -1,
            min_splatmap_weight: 0.3,
            wind: WindParams::default(),
            lod: VegetationLOD::default(),
            align_to_normal: false,
            random_rotation: std::f32::consts::TAU,
            min_spacing: 2.0,
        }
    }
}

/// Wind animation parameters for vegetation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WindParams {
    /// How much the vegetation bends in the wind.
    pub bend_strength: f32,
    /// Frequency of flutter animation.
    pub flutter_frequency: f32,
    /// Phase offset multiplier for variation between instances.
    pub phase_variation: f32,
    /// Direction of the wind (XZ plane).
    pub wind_direction: Vec2,
    /// Wind speed multiplier.
    pub wind_speed: f32,
}

impl Default for WindParams {
    fn default() -> Self {
        Self {
            bend_strength: 0.3,
            flutter_frequency: 2.0,
            phase_variation: 1.0,
            wind_direction: Vec2::new(1.0, 0.0),
            wind_speed: 1.0,
        }
    }
}

/// LOD transition distances for vegetation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VegetationLOD {
    /// Distance at which to switch from full mesh to simplified mesh.
    pub mesh_lod1_distance: f32,
    /// Distance at which to switch from simplified mesh to billboard.
    pub billboard_distance: f32,
    /// Distance at which to completely cull (remove) the instance.
    pub cull_distance: f32,
    /// Crossfade range for smooth transitions.
    pub crossfade_range: f32,
}

impl Default for VegetationLOD {
    fn default() -> Self {
        Self {
            mesh_lod1_distance: 50.0,
            billboard_distance: 150.0,
            cull_distance: 500.0,
            crossfade_range: 10.0,
        }
    }
}

// ---------------------------------------------------------------------------
// VegetationInstance
// ---------------------------------------------------------------------------

/// A single placed vegetation instance.
#[derive(Debug, Clone)]
pub struct VegetationInstance {
    /// World-space position (on the terrain surface).
    pub position: Vec3,
    /// Rotation quaternion.
    pub rotation: Quat,
    /// Uniform scale factor.
    pub scale: f32,
    /// Index of the [`VegetationLayer`] this instance belongs to.
    pub layer_index: usize,
    /// Color tint variation (RGB multiplier).
    pub color_tint: Vec3,
    /// Wind phase offset for animation variation.
    pub wind_phase: f32,
}

/// Data for GPU instanced rendering of vegetation.
///
/// Packed per-instance data suitable for an instance buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VegetationInstanceData {
    /// Model matrix row 0.
    pub model_row0: [f32; 4],
    /// Model matrix row 1.
    pub model_row1: [f32; 4],
    /// Model matrix row 2: [f32; 4].
    pub model_row2: [f32; 4],
    /// Color tint (RGB) + wind phase (A).
    pub color_and_phase: [f32; 4],
}

impl VegetationInstance {
    /// Converts to packed instance data for GPU upload.
    pub fn to_instance_data(&self) -> VegetationInstanceData {
        let matrix = glam::Mat4::from_scale_rotation_translation(
            Vec3::splat(self.scale),
            self.rotation,
            self.position,
        );

        let cols = matrix.to_cols_array_2d();

        VegetationInstanceData {
            model_row0: [cols[0][0], cols[1][0], cols[2][0], cols[3][0]],
            model_row1: [cols[0][1], cols[1][1], cols[2][1], cols[3][1]],
            model_row2: [cols[0][2], cols[1][2], cols[2][2], cols[3][2]],
            color_and_phase: [
                self.color_tint.x,
                self.color_tint.y,
                self.color_tint.z,
                self.wind_phase,
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Vegetation scattering
// ---------------------------------------------------------------------------

/// Scatters vegetation instances across the terrain.
///
/// Uses Poisson disk sampling for natural-looking placement. Instances are
/// filtered by altitude, slope, and splatmap weight constraints.
///
/// Returns a `Vec<VegetationInstance>` containing all placed instances across
/// all layers.
pub fn scatter_vegetation(
    heightmap: &Heightmap,
    splatmap: Option<&SplatMap>,
    density_map: Option<&DensityMap>,
    layers: &[VegetationLayer],
    terrain_size: f32,
    height_scale: f32,
    seed: u64,
) -> Vec<VegetationInstance> {
    let mut all_instances = Vec::new();
    let mut rng = SimpleRng::new(seed);

    for (layer_idx, layer) in layers.iter().enumerate() {
        let instances = scatter_layer(
            heightmap,
            splatmap,
            density_map,
            layer,
            layer_idx,
            terrain_size,
            height_scale,
            &mut rng,
        );
        all_instances.extend(instances);
    }

    log::info!(
        "Scattered {} vegetation instances across {} layers",
        all_instances.len(),
        layers.len()
    );

    all_instances
}

/// Scatters instances for a single vegetation layer.
fn scatter_layer(
    heightmap: &Heightmap,
    splatmap: Option<&SplatMap>,
    density_map: Option<&DensityMap>,
    layer: &VegetationLayer,
    layer_idx: usize,
    terrain_size: f32,
    height_scale: f32,
    rng: &mut SimpleRng,
) -> Vec<VegetationInstance> {
    let hm_w = heightmap.width() as f32;
    let hm_h = heightmap.height() as f32;

    // Use Poisson disk sampling for natural distribution
    let min_dist = layer.min_spacing;
    let points = poisson_disk_sample(
        terrain_size,
        terrain_size,
        min_dist,
        layer.density,
        rng,
    );

    let mut instances = Vec::with_capacity(points.len());

    for point in &points {
        // Convert world position to heightmap coordinates
        let hx = point.x / terrain_size * (hm_w - 1.0);
        let hz = point.y / terrain_size * (hm_h - 1.0);

        // Altitude check
        let altitude = heightmap.sample(hx, hz);
        if altitude < layer.altitude_min || altitude > layer.altitude_max {
            continue;
        }

        // Slope check
        let slope = heightmap.slope_at(hx, hz);
        if slope < layer.slope_min || slope > layer.slope_max {
            continue;
        }

        // Splatmap weight check
        if let Some(sm) = splatmap {
            if layer.preferred_splatmap_channel >= 0 {
                let u = hx / (hm_w - 1.0);
                let v = hz / (hm_h - 1.0);
                let weights = sm.sample(u, v);
                let channel_weight = match layer.preferred_splatmap_channel {
                    0 => weights.x,
                    1 => weights.y,
                    2 => weights.z,
                    3 => weights.w,
                    _ => 0.0,
                };
                if channel_weight < layer.min_splatmap_weight {
                    continue;
                }
            }
        }

        // Density map check
        if let Some(dm) = density_map {
            let u = point.x / terrain_size;
            let v = point.y / terrain_size;
            let density = dm.sample(u, v);
            if rng.next_f32() > density {
                continue;
            }
        }

        // Compute world position
        let world_y = altitude * height_scale;
        let position = Vec3::new(point.x, world_y, point.y);

        // Random rotation around Y axis
        let yaw = rng.next_f32() * layer.random_rotation;
        let rotation = if layer.align_to_normal {
            let normal = heightmap.normal_at(hx, hz);
            align_to_normal(normal, yaw)
        } else {
            Quat::from_rotation_y(yaw)
        };

        // Random scale
        let scale_range = layer.scale_max - layer.scale_min;
        let scale = layer.scale_min + rng.next_f32() * scale_range;

        // Color variation
        let color_var = layer.color_variation;
        let color_tint = Vec3::new(
            1.0 + (rng.next_f32() - 0.5) * color_var * 2.0,
            1.0 + (rng.next_f32() - 0.5) * color_var * 2.0,
            1.0 + (rng.next_f32() - 0.5) * color_var * 2.0,
        );

        let wind_phase = rng.next_f32() * std::f32::consts::TAU;

        instances.push(VegetationInstance {
            position,
            rotation,
            scale,
            layer_index: layer_idx,
            color_tint,
            wind_phase,
        });
    }

    instances
}

/// Computes a rotation quaternion that aligns an object to a terrain normal
/// with a yaw rotation around the normal axis.
fn align_to_normal(normal: Vec3, yaw: f32) -> Quat {
    let up = Vec3::Y;
    let axis = up.cross(normal);
    let len = axis.length();

    if len < 1e-6 {
        return Quat::from_rotation_y(yaw);
    }

    let angle = up.dot(normal).acos();
    let align_rot = Quat::from_axis_angle(axis.normalize(), angle);
    let yaw_rot = Quat::from_rotation_y(yaw);
    align_rot * yaw_rot
}

// ---------------------------------------------------------------------------
// Poisson disk sampling
// ---------------------------------------------------------------------------

/// Generates points distributed via Poisson disk sampling.
///
/// This produces a set of points where no two points are closer than
/// `min_distance`, creating a natural-looking distribution without the
/// regularity of grid-based placement.
///
/// Uses the fast Bridson algorithm with background grid acceleration.
fn poisson_disk_sample(
    width: f32,
    height: f32,
    min_distance: f32,
    density: f32,
    rng: &mut SimpleRng,
) -> Vec<Vec2> {
    // Target point count based on density
    let area = width * height;
    let target_count = (area * density) as usize;

    if target_count == 0 || min_distance <= 0.0 {
        return Vec::new();
    }

    let cell_size = min_distance / std::f32::consts::SQRT_2;
    let grid_w = (width / cell_size).ceil() as usize + 1;
    let grid_h = (height / cell_size).ceil() as usize + 1;

    // Background grid for fast spatial lookup (-1 = empty)
    let mut grid = vec![-1i32; grid_w * grid_h];
    let mut points: Vec<Vec2> = Vec::with_capacity(target_count);
    let mut active: Vec<usize> = Vec::new();

    let max_attempts = 30;

    // Helper to convert world position to grid cell
    let to_grid = |p: Vec2| -> (usize, usize) {
        let gx = (p.x / cell_size) as usize;
        let gz = (p.y / cell_size) as usize;
        (gx.min(grid_w - 1), gz.min(grid_h - 1))
    };

    // Seed with initial random point
    let first = Vec2::new(rng.next_f32() * width, rng.next_f32() * height);
    let (gx, gz) = to_grid(first);
    grid[gz * grid_w + gx] = 0;
    points.push(first);
    active.push(0);

    while !active.is_empty() && points.len() < target_count {
        // Pick a random active point
        let active_idx = rng.next_u32_range(active.len() as u32) as usize;
        let point_idx = active[active_idx];
        let center = points[point_idx];

        let mut found = false;

        for _ in 0..max_attempts {
            // Generate a random point in the annulus [min_distance, 2*min_distance]
            let angle = rng.next_f32() * std::f32::consts::TAU;
            let radius = min_distance + rng.next_f32() * min_distance;
            let candidate = Vec2::new(
                center.x + angle.cos() * radius,
                center.y + angle.sin() * radius,
            );

            // Bounds check
            if candidate.x < 0.0
                || candidate.x >= width
                || candidate.y < 0.0
                || candidate.y >= height
            {
                continue;
            }

            let (cgx, cgz) = to_grid(candidate);

            // Check neighboring grid cells for conflicts
            let mut too_close = false;
            let search_radius = 2i32;

            'outer: for dz in -search_radius..=search_radius {
                for dx in -search_radius..=search_radius {
                    let nx = cgx as i32 + dx;
                    let nz = cgz as i32 + dz;

                    if nx < 0 || nz < 0 || nx >= grid_w as i32 || nz >= grid_h as i32 {
                        continue;
                    }

                    let neighbor_idx = grid[nz as usize * grid_w + nx as usize];
                    if neighbor_idx >= 0 {
                        let neighbor = points[neighbor_idx as usize];
                        if candidate.distance(neighbor) < min_distance {
                            too_close = true;
                            break 'outer;
                        }
                    }
                }
            }

            if !too_close {
                let new_idx = points.len();
                grid[cgz * grid_w + cgx] = new_idx as i32;
                points.push(candidate);
                active.push(new_idx);
                found = true;
                break;
            }
        }

        if !found {
            // Remove from active list
            active.swap_remove(active_idx);
        }
    }

    points
}

// ---------------------------------------------------------------------------
// DensityMap
// ---------------------------------------------------------------------------

/// A grayscale density map that controls local vegetation density.
///
/// Values range from 0.0 (no vegetation) to 1.0 (full density).
/// Can be painted by artists in the editor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityMap {
    /// Width in texels.
    width: u32,
    /// Height in texels.
    height: u32,
    /// Grayscale density values (0..1).
    data: Vec<f32>,
}

impl DensityMap {
    /// Creates a new density map filled with `initial_density`.
    pub fn new(width: u32, height: u32, initial_density: f32) -> Self {
        let count = (width as usize) * (height as usize);
        Self {
            width,
            height,
            data: vec![initial_density.clamp(0.0, 1.0); count],
        }
    }

    /// Creates a density map from raw data.
    pub fn from_raw(width: u32, height: u32, data: Vec<f32>) -> Option<Self> {
        if data.len() != (width as usize) * (height as usize) {
            return None;
        }
        Some(Self {
            width,
            height,
            data,
        })
    }

    /// Bilinear sampling at fractional UV coordinates.
    pub fn sample(&self, u: f32, v: f32) -> f32 {
        let max_x = (self.width - 1) as f32;
        let max_z = (self.height - 1) as f32;
        let x = (u * max_x).clamp(0.0, max_x);
        let z = (v * max_z).clamp(0.0, max_z);

        let x0 = x.floor() as usize;
        let z0 = z.floor() as usize;
        let x1 = (x0 + 1).min((self.width - 1) as usize);
        let z1 = (z0 + 1).min((self.height - 1) as usize);

        let fx = x - x0 as f32;
        let fz = z - z0 as f32;

        let w = self.width as usize;
        let d00 = self.data[z0 * w + x0];
        let d10 = self.data[z0 * w + x1];
        let d01 = self.data[z1 * w + x0];
        let d11 = self.data[z1 * w + x1];

        let top = d00 + (d10 - d00) * fx;
        let bot = d01 + (d11 - d01) * fx;
        top + (bot - top) * fz
    }

    /// Sets the density at a pixel.
    pub fn set(&mut self, x: u32, z: u32, density: f32) {
        if x < self.width && z < self.height {
            let idx = (z as usize) * (self.width as usize) + (x as usize);
            self.data[idx] = density.clamp(0.0, 1.0);
        }
    }

    /// Paints a circular brush on the density map.
    pub fn paint_brush(
        &mut self,
        center_u: f32,
        center_v: f32,
        radius: f32,
        density: f32,
        falloff: crate::texturing::BrushFalloff,
    ) {
        let cx = center_u * (self.width - 1) as f32;
        let cz = center_v * (self.height - 1) as f32;
        let r2 = radius * radius;

        let min_x = ((cx - radius).floor() as i32).max(0) as u32;
        let max_x = ((cx + radius).ceil() as i32).min(self.width as i32 - 1) as u32;
        let min_z = ((cz - radius).floor() as i32).max(0) as u32;
        let max_z = ((cz + radius).ceil() as i32).min(self.height as i32 - 1) as u32;

        for z in min_z..=max_z {
            for x in min_x..=max_x {
                let dx = x as f32 - cx;
                let dz = z as f32 - cz;
                let dist_sq = dx * dx + dz * dz;

                if dist_sq > r2 {
                    continue;
                }

                let dist = dist_sq.sqrt();
                let t = dist / radius;

                let factor = match falloff {
                    crate::texturing::BrushFalloff::Linear => 1.0 - t,
                    crate::texturing::BrushFalloff::Smooth => {
                        let s = 1.0 - t;
                        s * s * (3.0 - 2.0 * s)
                    }
                    crate::texturing::BrushFalloff::Constant => 1.0,
                    crate::texturing::BrushFalloff::Sharp => (1.0 - t * t).max(0.0),
                };

                let idx = (z as usize) * (self.width as usize) + (x as usize);
                let current = self.data[idx];
                self.data[idx] = (current + density * factor).clamp(0.0, 1.0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GrassRenderer
// ---------------------------------------------------------------------------

/// Specialized renderer for dense grass coverage.
///
/// Instead of individual mesh instances, grass is rendered as camera-facing
/// quads with vertex displacement for wind animation. This is much more
/// efficient for the very high instance counts needed for grass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrassRenderer {
    /// Configuration for the grass rendering.
    pub settings: GrassSettings,
    /// Generated grass blade instances.
    instances: Vec<GrassBlade>,
}

/// Configuration for grass rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrassSettings {
    /// Density of grass blades per square world unit.
    pub density: f32,
    /// Minimum blade height.
    pub height_min: f32,
    /// Maximum blade height.
    pub height_max: f32,
    /// Blade width.
    pub width: f32,
    /// Maximum distance at which grass is rendered.
    pub render_distance: f32,
    /// Distance over which grass fades out.
    pub fade_distance: f32,
    /// Wind bend strength.
    pub wind_strength: f32,
    /// Wind flutter frequency.
    pub wind_frequency: f32,
    /// Base color of the grass.
    pub base_color: Vec3,
    /// Tip color of the grass.
    pub tip_color: Vec3,
    /// Color variation amount.
    pub color_variation: f32,
    /// Maximum slope where grass can grow (radians).
    pub max_slope: f32,
    /// Segments per blade (more = smoother curves, more expensive).
    pub segments_per_blade: u32,
}

impl Default for GrassSettings {
    fn default() -> Self {
        Self {
            density: 10.0,
            height_min: 0.2,
            height_max: 0.5,
            width: 0.05,
            render_distance: 50.0,
            fade_distance: 10.0,
            wind_strength: 0.3,
            wind_frequency: 1.5,
            base_color: Vec3::new(0.1, 0.35, 0.05),
            tip_color: Vec3::new(0.3, 0.55, 0.15),
            color_variation: 0.15,
            max_slope: 0.6,
            segments_per_blade: 3,
        }
    }
}

/// A single grass blade instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrassBlade {
    /// World position of the blade base.
    pub position: Vec3,
    /// Rotation angle around Y axis.
    pub rotation: f32,
    /// Height of the blade.
    pub height: f32,
    /// Width of the blade.
    pub width: f32,
    /// Color tint.
    pub color: Vec3,
    /// Wind phase offset.
    pub wind_phase: f32,
    /// Bend direction (random per blade).
    pub bend_direction: f32,
}

/// Packed grass blade data for GPU instanced rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GrassBladeGpuData {
    /// Position (XYZ) + rotation (W).
    pub position_rotation: [f32; 4],
    /// Height, width, wind_phase, bend_direction.
    pub params: [f32; 4],
    /// Color (RGB) + unused (A).
    pub color: [f32; 4],
}

impl GrassRenderer {
    /// Creates a new grass renderer with the given settings.
    pub fn new(settings: GrassSettings) -> Self {
        Self {
            settings,
            instances: Vec::new(),
        }
    }

    /// Generates grass blade instances across the terrain.
    pub fn generate(
        &mut self,
        heightmap: &Heightmap,
        terrain_size: f32,
        height_scale: f32,
        density_map: Option<&DensityMap>,
        seed: u64,
    ) {
        let mut rng = SimpleRng::new(seed);
        let hm_w = heightmap.width() as f32;
        let hm_h = heightmap.height() as f32;

        let area = terrain_size * terrain_size;
        let target_count = (area * self.settings.density) as usize;
        self.instances.clear();
        self.instances.reserve(target_count);

        for _ in 0..target_count {
            let wx = rng.next_f32() * terrain_size;
            let wz = rng.next_f32() * terrain_size;

            let hx = wx / terrain_size * (hm_w - 1.0);
            let hz = wz / terrain_size * (hm_h - 1.0);

            // Slope check
            let slope = heightmap.slope_at(hx, hz);
            if slope > self.settings.max_slope {
                continue;
            }

            // Density map check
            if let Some(dm) = density_map {
                let u = wx / terrain_size;
                let v = wz / terrain_size;
                let d = dm.sample(u, v);
                if rng.next_f32() > d {
                    continue;
                }
            }

            let altitude = heightmap.sample(hx, hz);
            let world_y = altitude * height_scale;

            let height_range = self.settings.height_max - self.settings.height_min;
            let blade_height = self.settings.height_min + rng.next_f32() * height_range;

            let color_var = self.settings.color_variation;
            let color = Vec3::new(
                self.settings.base_color.x
                    + (rng.next_f32() - 0.5) * color_var * 2.0,
                self.settings.base_color.y
                    + (rng.next_f32() - 0.5) * color_var * 2.0,
                self.settings.base_color.z
                    + (rng.next_f32() - 0.5) * color_var * 2.0,
            );

            self.instances.push(GrassBlade {
                position: Vec3::new(wx, world_y, wz),
                rotation: rng.next_f32() * std::f32::consts::TAU,
                height: blade_height,
                width: self.settings.width,
                color,
                wind_phase: rng.next_f32() * std::f32::consts::TAU,
                bend_direction: rng.next_f32() * std::f32::consts::TAU,
            });
        }

        log::info!("Generated {} grass blades", self.instances.len());
    }

    /// Returns the generated grass blade instances.
    pub fn instances(&self) -> &[GrassBlade] {
        &self.instances
    }

    /// Returns the number of grass blade instances.
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Converts instances to packed GPU data.
    pub fn to_gpu_data(&self) -> Vec<GrassBladeGpuData> {
        self.instances
            .iter()
            .map(|blade| GrassBladeGpuData {
                position_rotation: [
                    blade.position.x,
                    blade.position.y,
                    blade.position.z,
                    blade.rotation,
                ],
                params: [
                    blade.height,
                    blade.width,
                    blade.wind_phase,
                    blade.bend_direction,
                ],
                color: [blade.color.x, blade.color.y, blade.color.z, 1.0],
            })
            .collect()
    }

    /// Returns only the instances within `max_distance` of `camera_pos`,
    /// sorted by distance for back-to-front rendering of transparent blades.
    pub fn visible_instances(
        &self,
        camera_pos: Vec3,
        max_distance: f32,
    ) -> Vec<&GrassBlade> {
        let max_dist_sq = max_distance * max_distance;

        let mut visible: Vec<(&GrassBlade, f32)> = self
            .instances
            .iter()
            .filter_map(|blade| {
                let dist_sq = camera_pos.distance_squared(blade.position);
                if dist_sq <= max_dist_sq {
                    Some((blade, dist_sq))
                } else {
                    None
                }
            })
            .collect();

        // Sort back-to-front
        visible.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        visible.into_iter().map(|(blade, _)| blade).collect()
    }

    /// Generates vertices for a single grass blade as a set of quads.
    ///
    /// Used for CPU-side mesh building when instanced rendering is not
    /// available.
    pub fn generate_blade_vertices(
        blade: &GrassBlade,
        segments: u32,
        time: f32,
        wind_dir: Vec2,
        wind_speed: f32,
        bend_strength: f32,
    ) -> Vec<[f32; 3]> {
        let segments = segments.max(1);
        let mut vertices = Vec::with_capacity((segments as usize + 1) * 2);

        let cos_r = blade.rotation.cos();
        let sin_r = blade.rotation.sin();

        for seg in 0..=segments {
            let t = seg as f32 / segments as f32;
            let y = blade.position.y + t * blade.height;

            // Wind displacement increases with height (cubic)
            let wind_t = t * t * t;
            let wind_phase = blade.wind_phase + time * wind_speed;
            let wind_x = wind_dir.x * wind_phase.sin() * wind_t * bend_strength;
            let wind_z = wind_dir.y * (wind_phase + blade.bend_direction).sin()
                * wind_t
                * bend_strength;

            // Blade width decreases towards the tip
            let w = blade.width * (1.0 - t * 0.5);

            // Two vertices per segment (left and right of the blade)
            let base_x = blade.position.x + wind_x;
            let base_z = blade.position.z + wind_z;

            let left_x = base_x + cos_r * w * 0.5;
            let left_z = base_z + sin_r * w * 0.5;
            let right_x = base_x - cos_r * w * 0.5;
            let right_z = base_z - sin_r * w * 0.5;

            vertices.push([left_x, y, left_z]);
            vertices.push([right_x, y, right_z]);
        }

        vertices
    }
}

// ---------------------------------------------------------------------------
// Vegetation LOD culling
// ---------------------------------------------------------------------------

/// Performs LOD selection and culling for vegetation instances.
///
/// Returns indices into the instance array, grouped by render mode
/// (mesh, billboard, or culled).
pub fn cull_vegetation(
    instances: &[VegetationInstance],
    layers: &[VegetationLayer],
    camera_pos: Vec3,
) -> VegetationCullResult {
    let mut result = VegetationCullResult {
        mesh_instances: Vec::new(),
        billboard_instances: Vec::new(),
        culled_count: 0,
    };

    for (idx, instance) in instances.iter().enumerate() {
        if instance.layer_index >= layers.len() {
            continue;
        }
        let layer = &layers[instance.layer_index];
        let distance = camera_pos.distance(instance.position);

        if distance > layer.lod.cull_distance {
            result.culled_count += 1;
        } else if distance > layer.lod.billboard_distance {
            // Compute crossfade alpha
            let fade_start = layer.lod.cull_distance - layer.lod.crossfade_range;
            let alpha = if distance > fade_start {
                1.0 - (distance - fade_start) / layer.lod.crossfade_range
            } else {
                1.0
            };
            result.billboard_instances.push(CulledInstance {
                index: idx,
                alpha,
                distance,
            });
        } else {
            let fade_start = layer.lod.billboard_distance - layer.lod.crossfade_range;
            let alpha = if distance > fade_start {
                1.0 - (distance - fade_start) / layer.lod.crossfade_range
            } else {
                1.0
            };
            result.mesh_instances.push(CulledInstance {
                index: idx,
                alpha,
                distance,
            });
        }
    }

    // Sort by distance (front to back for opaque, back to front for alpha)
    result
        .mesh_instances
        .sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
    result.billboard_instances.sort_by(|a, b| {
        b.distance
            .partial_cmp(&a.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    result
}

/// Result of vegetation LOD culling.
#[derive(Debug)]
pub struct VegetationCullResult {
    /// Instances to render as full meshes.
    pub mesh_instances: Vec<CulledInstance>,
    /// Instances to render as billboards.
    pub billboard_instances: Vec<CulledInstance>,
    /// Number of instances culled entirely.
    pub culled_count: usize,
}

/// A culled vegetation instance with LOD information.
#[derive(Debug, Clone)]
pub struct CulledInstance {
    /// Index into the original instance array.
    pub index: usize,
    /// Alpha/fade value for crossfade transitions.
    pub alpha: f32,
    /// Distance from the camera.
    pub distance: f32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson_disk_basic() {
        let mut rng = SimpleRng::new(42);
        let points = poisson_disk_sample(100.0, 100.0, 5.0, 0.05, &mut rng);
        assert!(!points.is_empty());

        // Verify minimum distance constraint
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let dist = points[i].distance(points[j]);
                assert!(
                    dist >= 4.9, // small epsilon for floating point
                    "Points too close: {} (dist = {})",
                    i,
                    dist
                );
            }
        }
    }

    #[test]
    fn scatter_basic() {
        let hm = crate::Heightmap::new_flat(33, 33, 0.5).unwrap();
        let layer = VegetationLayer {
            density: 0.01,
            min_spacing: 3.0,
            ..Default::default()
        };
        let instances =
            scatter_vegetation(&hm, None, None, &[layer], 100.0, 50.0, 42);
        assert!(!instances.is_empty());
    }

    #[test]
    fn scatter_respects_slope() {
        // Create a very steep terrain
        let mut data = vec![0.0f32; 33 * 33];
        for z in 0..33 {
            for x in 0..33 {
                data[z * 33 + x] = x as f32 / 32.0; // steep slope in X
            }
        }
        let hm = crate::Heightmap::from_raw(33, 33, data).unwrap();

        let layer = VegetationLayer {
            density: 0.1,
            slope_max: 0.1, // very restrictive slope
            min_spacing: 1.0,
            ..Default::default()
        };
        let instances =
            scatter_vegetation(&hm, None, None, &[layer], 32.0, 100.0, 42);
        // Most/all should be filtered due to steep slope
        // The count should be much smaller than on flat terrain
        let flat_hm = crate::Heightmap::new_flat(33, 33, 0.5).unwrap();
        let flat_instances =
            scatter_vegetation(&flat_hm, None, None, &[layer.clone()], 32.0, 100.0, 42);
        assert!(instances.len() <= flat_instances.len());
    }

    #[test]
    fn vegetation_cull() {
        let layers = vec![VegetationLayer::default()];
        let instances = vec![
            VegetationInstance {
                position: Vec3::new(10.0, 0.0, 0.0),
                rotation: Quat::IDENTITY,
                scale: 1.0,
                layer_index: 0,
                color_tint: Vec3::ONE,
                wind_phase: 0.0,
            },
            VegetationInstance {
                position: Vec3::new(200.0, 0.0, 0.0),
                rotation: Quat::IDENTITY,
                scale: 1.0,
                layer_index: 0,
                color_tint: Vec3::ONE,
                wind_phase: 0.0,
            },
            VegetationInstance {
                position: Vec3::new(600.0, 0.0, 0.0),
                rotation: Quat::IDENTITY,
                scale: 1.0,
                layer_index: 0,
                color_tint: Vec3::ONE,
                wind_phase: 0.0,
            },
        ];

        let result = cull_vegetation(&instances, &layers, Vec3::ZERO);
        assert_eq!(result.mesh_instances.len(), 1);
        assert_eq!(result.billboard_instances.len(), 1);
        assert_eq!(result.culled_count, 1);
    }

    #[test]
    fn grass_renderer_generate() {
        let hm = crate::Heightmap::new_flat(17, 17, 0.5).unwrap();
        let settings = GrassSettings {
            density: 1.0,
            ..Default::default()
        };
        let mut grass = GrassRenderer::new(settings);
        grass.generate(&hm, 16.0, 10.0, None, 42);
        assert!(grass.instance_count() > 0);
    }

    #[test]
    fn grass_gpu_data() {
        let hm = crate::Heightmap::new_flat(9, 9, 0.3).unwrap();
        let settings = GrassSettings {
            density: 2.0,
            ..Default::default()
        };
        let mut grass = GrassRenderer::new(settings);
        grass.generate(&hm, 8.0, 5.0, None, 99);
        let gpu_data = grass.to_gpu_data();
        assert_eq!(gpu_data.len(), grass.instance_count());
    }

    #[test]
    fn density_map_sampling() {
        let dm = DensityMap::new(4, 4, 0.5);
        let v = dm.sample(0.5, 0.5);
        assert!((v - 0.5).abs() < 0.01);
    }

    #[test]
    fn instance_to_gpu_data() {
        let inst = VegetationInstance {
            position: Vec3::new(10.0, 5.0, 20.0),
            rotation: Quat::IDENTITY,
            scale: 2.0,
            layer_index: 0,
            color_tint: Vec3::ONE,
            wind_phase: 0.5,
        };
        let gpu = inst.to_instance_data();
        // The translation should appear in the model matrix
        assert!((gpu.model_row0[3] - 10.0).abs() < 0.01);
        assert!((gpu.model_row1[3] - 5.0).abs() < 0.01);
    }
}
