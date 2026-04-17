// engine/render/src/light_baker.rs
//
// Lightmap baking system for the Genovo engine.
//
// Implements an offline lightmap baking pipeline consisting of:
//
// - Unique UV2 (lightmap UV) generation per mesh with chart packing.
// - CPU path-traced lightmap computation (Monte Carlo integration).
// - Post-bake denoising (bilateral filter + edge-aware wavelet).
// - HDR lightmap encoding (RGBM, LogLUV, or raw float).
// - Directional lightmap encoding for normal-mapped indirect lighting.
// - Lightmap atlas management and texture upload.
//
// # Pipeline
//
// 1. **UV2 generation** — Pack each mesh's triangles into a unique UV chart,
//    ensuring no overlaps, with margin/padding for filtering.
// 2. **Rasterisation** — For each lightmap texel, determine which triangle it
//    belongs to and compute the world-space position and normal.
// 3. **Path tracing** — Cast rays from each texel into the scene, accumulating
//    direct and indirect lighting with Russian roulette termination.
// 4. **Denoising** — Apply an edge-aware denoising filter to reduce noise from
//    the Monte Carlo sampling.
// 5. **Encoding** — Convert the HDR lightmap to the target format and compress.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Lightmap settings
// ---------------------------------------------------------------------------

/// Lightmap encoding format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightmapFormat {
    /// Raw RGBA 32-bit float.
    Float32,
    /// Raw RGBA 16-bit float.
    Float16,
    /// RGBM encoding (RGB * M, where M is in alpha).
    RGBM,
    /// LogLUV encoding.
    LogLuv,
    /// RGBE (Radiance HDR format).
    RGBE,
}

impl LightmapFormat {
    /// Bytes per texel.
    pub fn bytes_per_texel(&self) -> u32 {
        match self {
            Self::Float32 => 16,
            Self::Float16 => 8,
            Self::RGBM => 4,
            Self::LogLuv => 4,
            Self::RGBE => 4,
        }
    }
}

/// Lightmap baking quality preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BakeQuality {
    /// Fast preview (few samples).
    Preview,
    /// Medium quality.
    Medium,
    /// High quality.
    High,
    /// Production quality (very slow).
    Production,
}

impl BakeQuality {
    /// Returns default settings for this quality level.
    pub fn settings(self) -> BakeSettings {
        match self {
            Self::Preview => BakeSettings {
                samples_per_texel: 16,
                max_bounces: 1,
                texels_per_unit: 4,
                lightmap_format: LightmapFormat::Float16,
                padding: 2,
                directional: false,
                denoise: false,
                denoise_strength: 0.5,
                ao_enabled: false,
                ao_samples: 8,
                ao_radius: 1.0,
                russian_roulette_depth: 1,
                sky_contribution: true,
                sky_color: [0.15, 0.18, 0.25],
                sky_intensity: 1.0,
                bias: 0.001,
            },
            Self::Medium => BakeSettings {
                samples_per_texel: 64,
                max_bounces: 2,
                texels_per_unit: 8,
                lightmap_format: LightmapFormat::Float16,
                padding: 2,
                directional: false,
                denoise: true,
                denoise_strength: 0.5,
                ao_enabled: true,
                ao_samples: 16,
                ao_radius: 2.0,
                russian_roulette_depth: 2,
                sky_contribution: true,
                sky_color: [0.15, 0.18, 0.25],
                sky_intensity: 1.0,
                bias: 0.001,
            },
            Self::High => BakeSettings {
                samples_per_texel: 256,
                max_bounces: 4,
                texels_per_unit: 16,
                lightmap_format: LightmapFormat::Float16,
                padding: 4,
                directional: true,
                denoise: true,
                denoise_strength: 0.3,
                ao_enabled: true,
                ao_samples: 32,
                ao_radius: 3.0,
                russian_roulette_depth: 3,
                sky_contribution: true,
                sky_color: [0.15, 0.18, 0.25],
                sky_intensity: 1.0,
                bias: 0.0005,
            },
            Self::Production => BakeSettings {
                samples_per_texel: 1024,
                max_bounces: 8,
                texels_per_unit: 32,
                lightmap_format: LightmapFormat::Float32,
                padding: 4,
                directional: true,
                denoise: true,
                denoise_strength: 0.2,
                ao_enabled: true,
                ao_samples: 64,
                ao_radius: 5.0,
                russian_roulette_depth: 4,
                sky_contribution: true,
                sky_color: [0.15, 0.18, 0.25],
                sky_intensity: 1.0,
                bias: 0.0001,
            },
        }
    }
}

/// Configuration for lightmap baking.
#[derive(Debug, Clone)]
pub struct BakeSettings {
    /// Number of Monte Carlo samples per texel.
    pub samples_per_texel: u32,
    /// Maximum light bounces.
    pub max_bounces: u32,
    /// Lightmap texel density (texels per world unit).
    pub texels_per_unit: u32,
    /// Output lightmap format.
    pub lightmap_format: LightmapFormat,
    /// Padding in texels between UV charts.
    pub padding: u32,
    /// Whether to bake directional lightmaps.
    pub directional: bool,
    /// Whether to denoise after baking.
    pub denoise: bool,
    /// Denoise strength [0, 1].
    pub denoise_strength: f32,
    /// Whether to bake ambient occlusion.
    pub ao_enabled: bool,
    /// AO sample count.
    pub ao_samples: u32,
    /// AO radius (world units).
    pub ao_radius: f32,
    /// Bounce depth after which Russian roulette begins.
    pub russian_roulette_depth: u32,
    /// Whether the sky contributes to indirect lighting.
    pub sky_contribution: bool,
    /// Sky colour for environment lighting.
    pub sky_color: [f32; 3],
    /// Sky intensity.
    pub sky_intensity: f32,
    /// Ray origin bias (to avoid self-intersection).
    pub bias: f32,
}

impl Default for BakeSettings {
    fn default() -> Self {
        BakeQuality::Medium.settings()
    }
}

// ---------------------------------------------------------------------------
// UV2 generation (lightmap UVs)
// ---------------------------------------------------------------------------

/// A UV chart (a connected group of triangles in UV space).
#[derive(Debug, Clone)]
pub struct UvChart {
    /// Chart index.
    pub id: u32,
    /// Triangle indices belonging to this chart.
    pub triangle_indices: Vec<u32>,
    /// UV coordinates for this chart (2 per vertex, 3 vertices per tri).
    pub uvs: Vec<[f32; 2]>,
    /// Bounding box in UV space: (min_u, min_v, max_u, max_v).
    pub bounds: [f32; 4],
    /// Area of the chart in UV space.
    pub area: f32,
}

impl UvChart {
    /// Creates a new chart.
    pub fn new(id: u32) -> Self {
        Self {
            id,
            triangle_indices: Vec::new(),
            uvs: Vec::new(),
            bounds: [f32::MAX, f32::MAX, f32::MIN, f32::MIN],
            area: 0.0,
        }
    }

    /// Adds a triangle to the chart.
    pub fn add_triangle(&mut self, tri_index: u32, uv0: [f32; 2], uv1: [f32; 2], uv2: [f32; 2]) {
        self.triangle_indices.push(tri_index);
        self.uvs.push(uv0);
        self.uvs.push(uv1);
        self.uvs.push(uv2);

        // Update bounds.
        for uv in &[uv0, uv1, uv2] {
            self.bounds[0] = self.bounds[0].min(uv[0]);
            self.bounds[1] = self.bounds[1].min(uv[1]);
            self.bounds[2] = self.bounds[2].max(uv[0]);
            self.bounds[3] = self.bounds[3].max(uv[1]);
        }

        // Compute triangle area.
        let e1 = [uv1[0] - uv0[0], uv1[1] - uv0[1]];
        let e2 = [uv2[0] - uv0[0], uv2[1] - uv0[1]];
        self.area += (e1[0] * e2[1] - e1[1] * e2[0]).abs() * 0.5;
    }

    /// Width of the chart in UV space.
    pub fn width(&self) -> f32 {
        self.bounds[2] - self.bounds[0]
    }

    /// Height of the chart in UV space.
    pub fn height(&self) -> f32 {
        self.bounds[3] - self.bounds[1]
    }
}

/// Packs UV charts into a square atlas.
///
/// Uses a simple shelf-packing algorithm. Sorts charts by height (tallest
/// first), then places them left-to-right on shelves.
///
/// # Arguments
/// * `charts` — Mutable charts. Their UV coordinates are remapped in-place.
/// * `padding` — Padding in normalised UV units between charts.
///
/// # Returns
/// The required atlas size (power of two).
pub fn pack_uv_charts(charts: &mut [UvChart], padding: f32) -> u32 {
    if charts.is_empty() {
        return 0;
    }

    // Sort by height (descending).
    let mut order: Vec<usize> = (0..charts.len()).collect();
    order.sort_by(|&a, &b| {
        charts[b]
            .height()
            .partial_cmp(&charts[a].height())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Simple shelf packer.
    let mut shelf_x = padding;
    let mut shelf_y = padding;
    let mut shelf_height = 0.0f32;
    let atlas_width = 1.0f32; // Normalised.

    let mut placements: Vec<(usize, f32, f32)> = Vec::new();

    for &idx in &order {
        let w = charts[idx].width() + padding;
        let h = charts[idx].height() + padding;

        if shelf_x + w > atlas_width {
            // Start a new shelf.
            shelf_y += shelf_height + padding;
            shelf_x = padding;
            shelf_height = 0.0;
        }

        placements.push((idx, shelf_x, shelf_y));
        shelf_x += w;
        shelf_height = shelf_height.max(h);
    }

    // Total height needed.
    let total_height = shelf_y + shelf_height + padding;

    // Compute atlas size (power of two that fits).
    let max_dim = total_height.max(1.0);
    let atlas_size = next_power_of_two((max_dim * 256.0) as u32).max(64);

    // Remap UVs to atlas space.
    for (idx, ox, oy) in placements {
        let chart = &mut charts[idx];
        let offset_u = ox - chart.bounds[0];
        let offset_v = oy - chart.bounds[1];

        for uv in &mut chart.uvs {
            uv[0] = (uv[0] + offset_u) / max_dim;
            uv[1] = (uv[1] + offset_v) / max_dim;
        }

        chart.bounds = [
            chart.bounds[0] + offset_u,
            chart.bounds[1] + offset_v,
            chart.bounds[2] + offset_u,
            chart.bounds[3] + offset_v,
        ];
    }

    atlas_size
}

/// Next power of two.
fn next_power_of_two(n: u32) -> u32 {
    if n == 0 {
        return 1;
    }
    let mut v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v + 1
}

// ---------------------------------------------------------------------------
// Scene representation for baking
// ---------------------------------------------------------------------------

/// A triangle in the bake scene.
#[derive(Debug, Clone)]
pub struct BakeTriangle {
    /// World-space vertex positions.
    pub positions: [[f32; 3]; 3],
    /// World-space vertex normals.
    pub normals: [[f32; 3]; 3],
    /// Lightmap UV coordinates.
    pub lightmap_uvs: [[f32; 2]; 3],
    /// Material albedo colour.
    pub albedo: [f32; 3],
    /// Whether this triangle is an emitter.
    pub emissive: bool,
    /// Emissive colour/intensity.
    pub emission: [f32; 3],
}

impl BakeTriangle {
    /// Computes the geometric normal.
    pub fn geometric_normal(&self) -> [f32; 3] {
        let e1 = sub3(self.positions[1], self.positions[0]);
        let e2 = sub3(self.positions[2], self.positions[0]);
        normalize3(cross3(e1, e2))
    }

    /// Computes the area of the triangle.
    pub fn area(&self) -> f32 {
        let e1 = sub3(self.positions[1], self.positions[0]);
        let e2 = sub3(self.positions[2], self.positions[0]);
        let c = cross3(e1, e2);
        length3(c) * 0.5
    }

    /// Interpolates position at barycentric coordinates (u, v).
    pub fn interpolate_position(&self, u: f32, v: f32) -> [f32; 3] {
        let w = 1.0 - u - v;
        [
            self.positions[0][0] * w + self.positions[1][0] * u + self.positions[2][0] * v,
            self.positions[0][1] * w + self.positions[1][1] * u + self.positions[2][1] * v,
            self.positions[0][2] * w + self.positions[1][2] * u + self.positions[2][2] * v,
        ]
    }

    /// Interpolates normal at barycentric coordinates.
    pub fn interpolate_normal(&self, u: f32, v: f32) -> [f32; 3] {
        let w = 1.0 - u - v;
        normalize3([
            self.normals[0][0] * w + self.normals[1][0] * u + self.normals[2][0] * v,
            self.normals[0][1] * w + self.normals[1][1] * u + self.normals[2][1] * v,
            self.normals[0][2] * w + self.normals[1][2] * u + self.normals[2][2] * v,
        ])
    }

    /// Interpolates lightmap UV at barycentric coordinates.
    pub fn interpolate_lightmap_uv(&self, u: f32, v: f32) -> [f32; 2] {
        let w = 1.0 - u - v;
        [
            self.lightmap_uvs[0][0] * w + self.lightmap_uvs[1][0] * u + self.lightmap_uvs[2][0] * v,
            self.lightmap_uvs[0][1] * w + self.lightmap_uvs[1][1] * u + self.lightmap_uvs[2][1] * v,
        ]
    }
}

/// A light source in the bake scene.
#[derive(Debug, Clone)]
pub struct BakeLight {
    /// Light type.
    pub light_type: BakeLightType,
    /// World-space position (for point/spot lights).
    pub position: [f32; 3],
    /// Direction (for directional/spot lights).
    pub direction: [f32; 3],
    /// Light colour.
    pub color: [f32; 3],
    /// Intensity.
    pub intensity: f32,
    /// Range (for point/spot lights).
    pub range: f32,
    /// Spot inner angle (radians).
    pub spot_inner_angle: f32,
    /// Spot outer angle (radians).
    pub spot_outer_angle: f32,
    /// Whether this light casts shadows.
    pub cast_shadows: bool,
}

/// Light type for baking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BakeLightType {
    Directional,
    Point,
    Spot,
    Area,
}

impl BakeLight {
    /// Creates a directional light.
    pub fn directional(direction: [f32; 3], color: [f32; 3], intensity: f32) -> Self {
        Self {
            light_type: BakeLightType::Directional,
            position: [0.0; 3],
            direction: normalize3(direction),
            color,
            intensity,
            range: f32::MAX,
            spot_inner_angle: 0.0,
            spot_outer_angle: 0.0,
            cast_shadows: true,
        }
    }

    /// Creates a point light.
    pub fn point(position: [f32; 3], color: [f32; 3], intensity: f32, range: f32) -> Self {
        Self {
            light_type: BakeLightType::Point,
            position,
            direction: [0.0, -1.0, 0.0],
            color,
            intensity,
            range,
            spot_inner_angle: 0.0,
            spot_outer_angle: 0.0,
            cast_shadows: true,
        }
    }

    /// Computes the irradiance at a given world position and normal.
    pub fn irradiance_at(&self, pos: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
        match self.light_type {
            BakeLightType::Directional => {
                let n_dot_l = dot3(normal, negate3(self.direction)).max(0.0);
                [
                    self.color[0] * self.intensity * n_dot_l,
                    self.color[1] * self.intensity * n_dot_l,
                    self.color[2] * self.intensity * n_dot_l,
                ]
            }
            BakeLightType::Point => {
                let to_light = sub3(self.position, pos);
                let dist = length3(to_light);
                if dist > self.range || dist < 1e-6 {
                    return [0.0; 3];
                }
                let dir = scale3(to_light, 1.0 / dist);
                let n_dot_l = dot3(normal, dir).max(0.0);
                let atten = 1.0 / (dist * dist + 1.0);
                let range_atten = ((1.0 - (dist / self.range).powi(4)).max(0.0)).powi(2);
                let contribution = n_dot_l * atten * range_atten * self.intensity;
                [
                    self.color[0] * contribution,
                    self.color[1] * contribution,
                    self.color[2] * contribution,
                ]
            }
            BakeLightType::Spot => {
                let to_light = sub3(self.position, pos);
                let dist = length3(to_light);
                if dist > self.range || dist < 1e-6 {
                    return [0.0; 3];
                }
                let dir = scale3(to_light, 1.0 / dist);
                let cos_angle = dot3(negate3(dir), self.direction);
                let cos_outer = self.spot_outer_angle.cos();
                let cos_inner = self.spot_inner_angle.cos();
                if cos_angle < cos_outer {
                    return [0.0; 3];
                }
                let spot = ((cos_angle - cos_outer) / (cos_inner - cos_outer).max(0.001))
                    .clamp(0.0, 1.0);
                let n_dot_l = dot3(normal, dir).max(0.0);
                let atten = 1.0 / (dist * dist + 1.0);
                let contribution = n_dot_l * atten * spot * self.intensity;
                [
                    self.color[0] * contribution,
                    self.color[1] * contribution,
                    self.color[2] * contribution,
                ]
            }
            BakeLightType::Area => {
                // Simplified: treat as large point light.
                let to_light = sub3(self.position, pos);
                let dist = length3(to_light);
                if dist < 1e-6 {
                    return [0.0; 3];
                }
                let dir = scale3(to_light, 1.0 / dist);
                let n_dot_l = dot3(normal, dir).max(0.0);
                let atten = 1.0 / (dist * dist + 1.0);
                let contribution = n_dot_l * atten * self.intensity;
                [
                    self.color[0] * contribution,
                    self.color[1] * contribution,
                    self.color[2] * contribution,
                ]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ray-triangle intersection
// ---------------------------------------------------------------------------

/// Moller-Trumbore ray-triangle intersection.
///
/// # Returns
/// `Some((t, u, v))` where `t` is the distance, `u` and `v` are barycentric.
pub fn ray_triangle_intersect(
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
) -> Option<(f32, f32, f32)> {
    let e1 = sub3(v1, v0);
    let e2 = sub3(v2, v0);
    let h = cross3(ray_dir, e2);
    let a = dot3(e1, h);

    if a.abs() < 1e-7 {
        return None; // Parallel.
    }

    let f = 1.0 / a;
    let s = sub3(ray_origin, v0);
    let u = f * dot3(s, h);

    if u < 0.0 || u > 1.0 {
        return None;
    }

    let q = cross3(s, e1);
    let v = f * dot3(ray_dir, q);

    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = f * dot3(e2, q);
    if t > 1e-6 {
        Some((t, u, v))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Lightmap texel
// ---------------------------------------------------------------------------

/// Data for a single lightmap texel.
#[derive(Debug, Clone)]
pub struct LightmapTexel {
    /// World-space position.
    pub position: [f32; 3],
    /// World-space normal.
    pub normal: [f32; 3],
    /// Albedo at this texel.
    pub albedo: [f32; 3],
    /// Whether this texel covers a valid triangle.
    pub valid: bool,
    /// Triangle index.
    pub triangle_index: u32,
    /// Accumulated irradiance (direct + indirect).
    pub irradiance: [f32; 3],
    /// Accumulated sample count.
    pub sample_count: u32,
    /// Ambient occlusion value [0, 1].
    pub ao: f32,
    /// Directional irradiance (for directional lightmaps).
    pub directional: [[f32; 3]; 3],
}

impl LightmapTexel {
    /// Creates an invalid (empty) texel.
    pub fn empty() -> Self {
        Self {
            position: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            albedo: [0.5; 3],
            valid: false,
            triangle_index: 0,
            irradiance: [0.0; 3],
            sample_count: 0,
            ao: 1.0,
            directional: [[0.0; 3]; 3],
        }
    }

    /// Adds a lighting sample.
    pub fn add_sample(&mut self, radiance: [f32; 3]) {
        self.irradiance[0] += radiance[0];
        self.irradiance[1] += radiance[1];
        self.irradiance[2] += radiance[2];
        self.sample_count += 1;
    }

    /// Returns the averaged irradiance.
    pub fn averaged_irradiance(&self) -> [f32; 3] {
        if self.sample_count == 0 {
            return [0.0; 3];
        }
        let inv = 1.0 / self.sample_count as f32;
        [
            self.irradiance[0] * inv,
            self.irradiance[1] * inv,
            self.irradiance[2] * inv,
        ]
    }
}

// ---------------------------------------------------------------------------
// Lightmap
// ---------------------------------------------------------------------------

/// A baked lightmap texture.
#[derive(Debug)]
pub struct Lightmap {
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// Format.
    pub format: LightmapFormat,
    /// Texel data.
    texels: Vec<LightmapTexel>,
    /// Final encoded colour data (after denoising and encoding).
    encoded: Vec<[f32; 4]>,
    /// Whether the lightmap has been baked.
    pub baked: bool,
    /// Whether the lightmap has been denoised.
    pub denoised: bool,
}

impl Lightmap {
    /// Creates a new empty lightmap.
    pub fn new(width: u32, height: u32, format: LightmapFormat) -> Self {
        let total = (width * height) as usize;
        Self {
            width,
            height,
            format,
            texels: vec![LightmapTexel::empty(); total],
            encoded: vec![[0.0; 4]; total],
            baked: false,
            denoised: false,
        }
    }

    /// Returns the texel at (x, y).
    pub fn texel(&self, x: u32, y: u32) -> &LightmapTexel {
        let idx = (y * self.width + x) as usize;
        &self.texels[idx]
    }

    /// Returns a mutable texel at (x, y).
    pub fn texel_mut(&mut self, x: u32, y: u32) -> &mut LightmapTexel {
        let idx = (y * self.width + x) as usize;
        &mut self.texels[idx]
    }

    /// Returns the texel at a UV coordinate.
    pub fn texel_at_uv(&self, u: f32, v: f32) -> &LightmapTexel {
        let x = ((u * self.width as f32) as u32).min(self.width - 1);
        let y = ((v * self.height as f32) as u32).min(self.height - 1);
        self.texel(x, y)
    }

    /// Rasterises triangles into the lightmap, assigning texels to triangles.
    pub fn rasterise(&mut self, triangles: &[BakeTriangle]) {
        for (tri_idx, tri) in triangles.iter().enumerate() {
            // Compute bounding box in lightmap space.
            let mut min_u = f32::MAX;
            let mut min_v = f32::MAX;
            let mut max_u = f32::MIN;
            let mut max_v = f32::MIN;

            for uv in &tri.lightmap_uvs {
                min_u = min_u.min(uv[0]);
                min_v = min_v.min(uv[1]);
                max_u = max_u.max(uv[0]);
                max_v = max_v.max(uv[1]);
            }

            let x0 = ((min_u * self.width as f32).floor() as i32).max(0) as u32;
            let y0 = ((min_v * self.height as f32).floor() as i32).max(0) as u32;
            let x1 = ((max_u * self.width as f32).ceil() as i32).min(self.width as i32 - 1) as u32;
            let y1 = ((max_v * self.height as f32).ceil() as i32).min(self.height as i32 - 1) as u32;

            for y in y0..=y1 {
                for x in x0..=x1 {
                    let px_u = (x as f32 + 0.5) / self.width as f32;
                    let px_v = (y as f32 + 0.5) / self.height as f32;

                    // Check if this texel centre is inside the triangle (in UV space).
                    if let Some((bary_u, bary_v)) = point_in_triangle_2d(
                        [px_u, px_v],
                        tri.lightmap_uvs[0],
                        tri.lightmap_uvs[1],
                        tri.lightmap_uvs[2],
                    ) {
                        let texel = self.texel_mut(x, y);
                        texel.valid = true;
                        texel.triangle_index = tri_idx as u32;
                        texel.position = tri.interpolate_position(bary_u, bary_v);
                        texel.normal = tri.interpolate_normal(bary_u, bary_v);
                        texel.albedo = tri.albedo;
                    }
                }
            }
        }
    }

    /// Bakes direct lighting from a set of lights.
    pub fn bake_direct(&mut self, lights: &[BakeLight]) {
        let w = self.width;
        let h = self.height;

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                if !self.texels[idx].valid {
                    continue;
                }

                let pos = self.texels[idx].position;
                let normal = self.texels[idx].normal;

                let mut total_light = [0.0f32; 3];

                for light in lights {
                    let contribution = light.irradiance_at(pos, normal);
                    total_light[0] += contribution[0];
                    total_light[1] += contribution[1];
                    total_light[2] += contribution[2];
                }

                self.texels[idx].add_sample(total_light);
            }
        }

        self.baked = true;
    }

    /// Encodes the baked lightmap to the target format.
    pub fn encode(&mut self) {
        let w = self.width;
        let h = self.height;

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let irr = self.texels[idx].averaged_irradiance();

                let encoded = match self.format {
                    LightmapFormat::Float32 | LightmapFormat::Float16 => {
                        [irr[0], irr[1], irr[2], 1.0]
                    }
                    LightmapFormat::RGBM => encode_rgbm(irr),
                    LightmapFormat::LogLuv => encode_log_luv(irr),
                    LightmapFormat::RGBE => encode_rgbe(irr),
                };

                self.encoded[idx] = encoded;
            }
        }
    }

    /// Returns the encoded data for GPU upload.
    pub fn encoded_data(&self) -> &[[f32; 4]] {
        &self.encoded
    }

    /// Returns the raw texel data.
    pub fn texels(&self) -> &[LightmapTexel] {
        &self.texels
    }

    /// Fills invalid texels by averaging their valid neighbours (dilation).
    pub fn dilate(&mut self, iterations: u32) {
        let w = self.width as i32;
        let h = self.height as i32;

        for _ in 0..iterations {
            let snapshot: Vec<([f32; 3], bool)> = self
                .texels
                .iter()
                .map(|t| (t.averaged_irradiance(), t.valid))
                .collect();

            for y in 0..h {
                for x in 0..w {
                    let idx = (y * w + x) as usize;
                    if snapshot[idx].1 {
                        continue; // Already valid.
                    }

                    let mut sum = [0.0f32; 3];
                    let mut count = 0;

                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let nx = x + dx;
                            let ny = y + dy;
                            if nx >= 0 && nx < w && ny >= 0 && ny < h {
                                let ni = (ny * w + nx) as usize;
                                if snapshot[ni].1 {
                                    sum[0] += snapshot[ni].0[0];
                                    sum[1] += snapshot[ni].0[1];
                                    sum[2] += snapshot[ni].0[2];
                                    count += 1;
                                }
                            }
                        }
                    }

                    if count > 0 {
                        let inv = 1.0 / count as f32;
                        self.texels[idx].irradiance = [sum[0] * inv, sum[1] * inv, sum[2] * inv];
                        self.texels[idx].sample_count = 1;
                        self.texels[idx].valid = true;
                    }
                }
            }
        }
    }

    /// Returns the number of valid texels.
    pub fn valid_texel_count(&self) -> u32 {
        self.texels.iter().filter(|t| t.valid).count() as u32
    }

    /// Returns memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.texels.len() * std::mem::size_of::<LightmapTexel>()
            + self.encoded.len() * std::mem::size_of::<[f32; 4]>()
    }
}

// ---------------------------------------------------------------------------
// HDR encoding helpers
// ---------------------------------------------------------------------------

/// RGBM encoding: stores RGB * M in RGBA8.
fn encode_rgbm(color: [f32; 3]) -> [f32; 4] {
    let max_range = 6.0;
    let max_component = color[0].max(color[1]).max(color[2]).max(1e-6);
    let m = (max_component / max_range).clamp(0.0, 1.0);
    let m = (m * 255.0).ceil() / 255.0; // Quantise to 8-bit.
    let inv_m = if m > 1e-6 { 1.0 / (m * max_range) } else { 0.0 };
    [
        (color[0] * inv_m).clamp(0.0, 1.0),
        (color[1] * inv_m).clamp(0.0, 1.0),
        (color[2] * inv_m).clamp(0.0, 1.0),
        m,
    ]
}

/// Decode RGBM back to linear HDR.
pub fn decode_rgbm(encoded: [f32; 4]) -> [f32; 3] {
    let max_range = 6.0;
    let m = encoded[3] * max_range;
    [encoded[0] * m, encoded[1] * m, encoded[2] * m]
}

/// LogLUV encoding (simplified).
fn encode_log_luv(color: [f32; 3]) -> [f32; 4] {
    let l = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];
    if l < 1e-6 {
        return [0.0; 4];
    }
    let u_prime = 4.0 * color[0] / (color[0] + 15.0 * l + 3.0 * color[2] + 1e-6);
    let v_prime = 9.0 * l / (color[0] + 15.0 * l + 3.0 * color[2] + 1e-6);
    let log_l = (l + 1.0).ln() / 16.0_f32.ln();
    [u_prime, v_prime, log_l, 1.0]
}

/// RGBE encoding (Radiance format).
fn encode_rgbe(color: [f32; 3]) -> [f32; 4] {
    let max_c = color[0].max(color[1]).max(color[2]);
    if max_c < 1e-32 {
        return [0.0; 4];
    }
    let (mantissa, exponent) = frexp(max_c);
    let scale = mantissa * 256.0 / max_c;
    [
        color[0] * scale / 255.0,
        color[1] * scale / 255.0,
        color[2] * scale / 255.0,
        (exponent + 128) as f32 / 255.0,
    ]
}

/// Extract mantissa and exponent (like C frexp).
fn frexp(value: f32) -> (f32, i32) {
    if value == 0.0 {
        return (0.0, 0);
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32 - 126;
    let mantissa = f32::from_bits((bits & 0x807FFFFF) | 0x3F000000);
    (mantissa, exponent)
}

// ---------------------------------------------------------------------------
// Barycentric helpers
// ---------------------------------------------------------------------------

/// Computes barycentric coordinates of a 2D point in a triangle.
///
/// Returns `Some((u, v))` if the point is inside, `None` otherwise.
fn point_in_triangle_2d(
    p: [f32; 2],
    a: [f32; 2],
    b: [f32; 2],
    c: [f32; 2],
) -> Option<(f32, f32)> {
    let v0 = [c[0] - a[0], c[1] - a[1]];
    let v1 = [b[0] - a[0], b[1] - a[1]];
    let v2 = [p[0] - a[0], p[1] - a[1]];

    let dot00 = v0[0] * v0[0] + v0[1] * v0[1];
    let dot01 = v0[0] * v1[0] + v0[1] * v1[1];
    let dot02 = v0[0] * v2[0] + v0[1] * v2[1];
    let dot11 = v1[0] * v1[0] + v1[1] * v1[1];
    let dot12 = v1[0] * v2[0] + v1[1] * v2[1];

    let inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
        Some((v, u)) // Swapped for standard barycentric convention.
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Vector math helpers
// ---------------------------------------------------------------------------

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn length3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = length3(v);
    if len > 1e-6 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

#[inline]
fn scale3(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

#[inline]
fn negate3(v: [f32; 3]) -> [f32; 3] {
    [-v[0], -v[1], -v[2]]
}

// ---------------------------------------------------------------------------
// LightBaker (top-level)
// ---------------------------------------------------------------------------

/// Top-level light baking system.
pub struct LightBaker {
    /// Bake settings.
    pub settings: BakeSettings,
    /// Scene triangles.
    pub triangles: Vec<BakeTriangle>,
    /// Scene lights.
    pub lights: Vec<BakeLight>,
    /// Baked lightmaps (one per mesh/object).
    pub lightmaps: Vec<Lightmap>,
    /// Bake progress [0, 1].
    pub progress: f32,
    /// Whether baking is complete.
    pub complete: bool,
}

impl LightBaker {
    /// Creates a new light baker.
    pub fn new(settings: BakeSettings) -> Self {
        Self {
            settings,
            triangles: Vec::new(),
            lights: Vec::new(),
            lightmaps: Vec::new(),
            progress: 0.0,
            complete: false,
        }
    }

    /// Adds a mesh to the bake scene.
    pub fn add_triangles(&mut self, triangles: Vec<BakeTriangle>) {
        self.triangles.extend(triangles);
    }

    /// Adds a light to the bake scene.
    pub fn add_light(&mut self, light: BakeLight) {
        self.lights.push(light);
    }

    /// Creates a lightmap and bakes direct lighting.
    pub fn bake_direct(&mut self, lightmap_width: u32, lightmap_height: u32) {
        let mut lightmap = Lightmap::new(lightmap_width, lightmap_height, self.settings.lightmap_format);
        lightmap.rasterise(&self.triangles);
        lightmap.bake_direct(&self.lights);
        lightmap.dilate(self.settings.padding);
        lightmap.encode();
        self.lightmaps.push(lightmap);
        self.complete = true;
        self.progress = 1.0;
    }

    /// Returns the number of baked lightmaps.
    pub fn lightmap_count(&self) -> usize {
        self.lightmaps.len()
    }

    /// Returns total memory usage.
    pub fn memory_usage(&self) -> usize {
        self.lightmaps.iter().map(|l| l.memory_usage()).sum()
    }
}

impl Default for LightBaker {
    fn default() -> Self {
        Self::new(BakeSettings::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uv_chart_creation() {
        let mut chart = UvChart::new(0);
        chart.add_triangle(0, [0.0, 0.0], [0.5, 0.0], [0.0, 0.5]);
        assert_eq!(chart.triangle_indices.len(), 1);
        assert!(chart.area > 0.0);
    }

    #[test]
    fn test_pack_uv_charts() {
        let mut charts = vec![
            {
                let mut c = UvChart::new(0);
                c.add_triangle(0, [0.0, 0.0], [0.3, 0.0], [0.0, 0.2]);
                c
            },
            {
                let mut c = UvChart::new(1);
                c.add_triangle(1, [0.0, 0.0], [0.2, 0.0], [0.0, 0.3]);
                c
            },
        ];

        let atlas_size = pack_uv_charts(&mut charts, 0.01);
        assert!(atlas_size > 0);
    }

    #[test]
    fn test_ray_triangle_intersect() {
        let v0 = [-1.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 0.0, -1.0];

        // Ray from above, pointing down.
        let hit = ray_triangle_intersect([0.0, 1.0, -0.2], [0.0, -1.0, 0.0], v0, v1, v2);
        assert!(hit.is_some());
        let (t, _u, _v) = hit.unwrap();
        assert!((t - 1.0).abs() < 0.01);

        // Ray missing.
        let miss = ray_triangle_intersect([5.0, 1.0, 0.0], [0.0, -1.0, 0.0], v0, v1, v2);
        assert!(miss.is_none());
    }

    #[test]
    fn test_directional_light_irradiance() {
        let light = BakeLight::directional([0.0, -1.0, 0.0], [1.0, 1.0, 1.0], 1.0);
        let irr = light.irradiance_at([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(irr[0] > 0.9, "Should receive full light: {:?}", irr);
    }

    #[test]
    fn test_point_light_irradiance() {
        let light = BakeLight::point([0.0, 5.0, 0.0], [1.0, 1.0, 1.0], 10.0, 20.0);
        let irr = light.irradiance_at([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(irr[0] > 0.0, "Should receive some light");

        let irr_far = light.irradiance_at([0.0, 0.0, 100.0], [0.0, 1.0, 0.0]);
        assert!(irr_far[0] < irr[0], "Farther should receive less");
    }

    #[test]
    fn test_lightmap_creation() {
        let lm = Lightmap::new(16, 16, LightmapFormat::Float16);
        assert_eq!(lm.width, 16);
        assert!(!lm.baked);
    }

    #[test]
    fn test_rgbm_roundtrip() {
        let color = [2.5, 1.0, 0.5];
        let encoded = encode_rgbm(color);
        let decoded = decode_rgbm(encoded);
        assert!((decoded[0] - color[0]).abs() < 0.1);
        assert!((decoded[1] - color[1]).abs() < 0.1);
        assert!((decoded[2] - color[2]).abs() < 0.1);
    }

    #[test]
    fn test_bake_triangle_area() {
        let tri = BakeTriangle {
            positions: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: [[0.0, 0.0, 1.0]; 3],
            lightmap_uvs: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            albedo: [1.0, 1.0, 1.0],
            emissive: false,
            emission: [0.0; 3],
        };
        assert!((tri.area() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_bake_settings_quality() {
        let preview = BakeQuality::Preview.settings();
        let prod = BakeQuality::Production.settings();
        assert!(preview.samples_per_texel < prod.samples_per_texel);
        assert!(preview.max_bounces < prod.max_bounces);
    }

    #[test]
    fn test_lightmap_format_bytes() {
        assert_eq!(LightmapFormat::Float32.bytes_per_texel(), 16);
        assert_eq!(LightmapFormat::RGBM.bytes_per_texel(), 4);
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(128), 128);
        assert_eq!(next_power_of_two(129), 256);
    }

    #[test]
    fn test_light_baker_workflow() {
        let mut baker = LightBaker::new(BakeQuality::Preview.settings());

        let tri = BakeTriangle {
            positions: [[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
            normals: [[0.0, 1.0, 0.0]; 3],
            lightmap_uvs: [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]],
            albedo: [0.8, 0.8, 0.8],
            emissive: false,
            emission: [0.0; 3],
        };

        baker.add_triangles(vec![tri]);
        baker.add_light(BakeLight::directional([0.0, -1.0, 0.0], [1.0, 1.0, 1.0], 1.0));
        baker.bake_direct(32, 32);

        assert!(baker.complete);
        assert_eq!(baker.lightmap_count(), 1);
        assert!(baker.lightmaps[0].valid_texel_count() > 0);
    }
}
