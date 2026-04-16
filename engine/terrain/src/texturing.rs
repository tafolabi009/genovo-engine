//! Terrain texturing — splatmap blending, materials, and auto-texturing.
//!
//! The terrain texturing system uses splatmaps (multi-channel weight textures)
//! to blend between multiple terrain material layers. Supports height-based
//! blending, slope-based auto-texturing, altitude-based layer assignment, and
//! triplanar mapping for steep surfaces.

use glam::{Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::heightmap::Heightmap;

// ---------------------------------------------------------------------------
// SplatMap
// ---------------------------------------------------------------------------

/// A multi-channel weight texture that controls blending between terrain
/// material layers.
///
/// Each splatmap has four channels (RGBA), each controlling the weight of one
/// terrain layer. Multiple splatmaps can be used to support more than four
/// layers (e.g. 2 splatmaps = 8 layers, 4 splatmaps = 16 layers).
///
/// Weights at each texel should sum to approximately 1.0, though the
/// system normalizes them during sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplatMap {
    /// Width of the splatmap in texels.
    width: u32,
    /// Height of the splatmap in texels.
    height: u32,
    /// RGBA weight data in row-major order. Length = width * height * 4.
    data: Vec<f32>,
}

impl SplatMap {
    /// Creates a new splatmap with the given dimensions, initialized to zero.
    pub fn new(width: u32, height: u32) -> Self {
        let count = (width as usize) * (height as usize) * 4;
        Self {
            width,
            height,
            data: vec![0.0; count],
        }
    }

    /// Creates a splatmap where the first channel has full weight everywhere.
    pub fn new_default(width: u32, height: u32) -> Self {
        let count = (width as usize) * (height as usize) * 4;
        let mut data = vec![0.0f32; count];
        // Set channel 0 (R) to 1.0
        for i in (0..count).step_by(4) {
            data[i] = 1.0;
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Creates a splatmap from raw RGBA data.
    pub fn from_raw(width: u32, height: u32, data: Vec<f32>) -> Option<Self> {
        let expected = (width as usize) * (height as usize) * 4;
        if data.len() != expected {
            return None;
        }
        Some(Self {
            width,
            height,
            data,
        })
    }

    /// Returns the width of the splatmap.
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the height of the splatmap.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns a reference to the raw data.
    #[inline]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Returns a mutable reference to the raw data.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Sets the weight for a specific channel at the given texel.
    pub fn set_weight(&mut self, x: u32, z: u32, channel: usize, weight: f32) {
        if x >= self.width || z >= self.height || channel >= 4 {
            return;
        }
        let idx = ((z as usize) * (self.width as usize) + (x as usize)) * 4 + channel;
        self.data[idx] = weight;
    }

    /// Returns the weight for a specific channel at the given texel.
    pub fn get_weight(&self, x: u32, z: u32, channel: usize) -> f32 {
        if x >= self.width || z >= self.height || channel >= 4 {
            return 0.0;
        }
        let idx = ((z as usize) * (self.width as usize) + (x as usize)) * 4 + channel;
        self.data[idx]
    }

    /// Returns all four weights at a texel as a Vec4.
    pub fn get_weights(&self, x: u32, z: u32) -> Vec4 {
        let x = x.min(self.width - 1);
        let z = z.min(self.height - 1);
        let base = ((z as usize) * (self.width as usize) + (x as usize)) * 4;
        Vec4::new(
            self.data[base],
            self.data[base + 1],
            self.data[base + 2],
            self.data[base + 3],
        )
    }

    /// Sets all four weights at a texel.
    pub fn set_weights(&mut self, x: u32, z: u32, weights: Vec4) {
        if x >= self.width || z >= self.height {
            return;
        }
        let base = ((z as usize) * (self.width as usize) + (x as usize)) * 4;
        self.data[base] = weights.x;
        self.data[base + 1] = weights.y;
        self.data[base + 2] = weights.z;
        self.data[base + 3] = weights.w;
    }

    /// Bilinear sampling of weights at fractional coordinates.
    pub fn sample(&self, u: f32, v: f32) -> Vec4 {
        let max_x = (self.width - 1) as f32;
        let max_z = (self.height - 1) as f32;
        let x = (u * max_x).clamp(0.0, max_x);
        let z = (v * max_z).clamp(0.0, max_z);

        let x0 = x.floor() as u32;
        let z0 = z.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let z1 = (z0 + 1).min(self.height - 1);

        let fx = x - x0 as f32;
        let fz = z - z0 as f32;

        let w00 = self.get_weights(x0, z0);
        let w10 = self.get_weights(x1, z0);
        let w01 = self.get_weights(x0, z1);
        let w11 = self.get_weights(x1, z1);

        let top = w00.lerp(w10, fx);
        let bottom = w01.lerp(w11, fx);
        top.lerp(bottom, fz)
    }

    /// Normalizes weights at every texel so they sum to 1.0.
    pub fn normalize(&mut self) {
        let count = (self.width as usize) * (self.height as usize);
        for i in 0..count {
            let base = i * 4;
            let sum = self.data[base]
                + self.data[base + 1]
                + self.data[base + 2]
                + self.data[base + 3];
            if sum > f32::EPSILON {
                let inv = 1.0 / sum;
                self.data[base] *= inv;
                self.data[base + 1] *= inv;
                self.data[base + 2] *= inv;
                self.data[base + 3] *= inv;
            }
        }
    }

    /// Paints a circular brush onto the splatmap.
    ///
    /// `center_u`, `center_v` are in [0..1] texture space. `radius` is in
    /// texels. `channel` selects which layer to paint (0-3). `strength`
    /// controls the paint intensity.
    pub fn paint_brush(
        &mut self,
        center_u: f32,
        center_v: f32,
        radius: f32,
        channel: usize,
        strength: f32,
        falloff: BrushFalloff,
    ) {
        if channel >= 4 {
            return;
        }

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

                let falloff_factor = match falloff {
                    BrushFalloff::Linear => 1.0 - t,
                    BrushFalloff::Smooth => {
                        let s = 1.0 - t;
                        s * s * (3.0 - 2.0 * s)
                    }
                    BrushFalloff::Constant => 1.0,
                    BrushFalloff::Sharp => (1.0 - t * t).max(0.0),
                };

                let paint = strength * falloff_factor;

                // Increase target channel, decrease others proportionally
                let base =
                    ((z as usize) * (self.width as usize) + (x as usize)) * 4;
                let mut weights = [
                    self.data[base],
                    self.data[base + 1],
                    self.data[base + 2],
                    self.data[base + 3],
                ];

                weights[channel] += paint;

                // Normalize
                let sum: f32 = weights.iter().sum();
                if sum > f32::EPSILON {
                    let inv = 1.0 / sum;
                    for w in &mut weights {
                        *w *= inv;
                    }
                }

                self.data[base] = weights[0];
                self.data[base + 1] = weights[1];
                self.data[base + 2] = weights[2];
                self.data[base + 3] = weights[3];
            }
        }
    }

    /// Converts the splatmap to 8-bit RGBA data for GPU upload.
    pub fn to_rgba8(&self) -> Vec<u8> {
        let count = (self.width as usize) * (self.height as usize) * 4;
        let mut out = Vec::with_capacity(count);
        for &v in &self.data {
            out.push((v.clamp(0.0, 1.0) * 255.0) as u8);
        }
        out
    }
}

/// Brush falloff curve for splatmap painting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrushFalloff {
    /// Constant weight across the entire brush.
    Constant,
    /// Linear falloff from center to edge.
    Linear,
    /// Smooth (Hermite) falloff.
    Smooth,
    /// Sharp falloff (quadratic).
    Sharp,
}

// ---------------------------------------------------------------------------
// TerrainLayer
// ---------------------------------------------------------------------------

/// A single terrain material layer.
///
/// Each layer can have its own albedo (base color), normal map, roughness,
/// and height map for parallax / height-blended transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainLayer {
    /// Human-readable name (e.g. "Grass", "Rock", "Sand").
    pub name: String,
    /// Albedo (base color) texture path.
    pub albedo_path: String,
    /// Normal map texture path.
    pub normal_path: String,
    /// Roughness map texture path (optional).
    pub roughness_path: String,
    /// Height map texture path for height-based blending (optional).
    pub height_path: String,
    /// UV tiling scale — how many times the texture repeats per terrain unit.
    pub tiling_scale: f32,
    /// Sharpness of height-based blending transitions. Higher = sharper.
    pub height_blend_sharpness: f32,
    /// Metallic value (for PBR rendering).
    pub metallic: f32,
    /// Roughness value (used when no roughness map is provided).
    pub roughness: f32,
    /// Color tint applied to the albedo.
    pub tint: Vec3,
}

impl Default for TerrainLayer {
    fn default() -> Self {
        Self {
            name: "Default".into(),
            albedo_path: String::new(),
            normal_path: String::new(),
            roughness_path: String::new(),
            height_path: String::new(),
            tiling_scale: 10.0,
            height_blend_sharpness: 8.0,
            metallic: 0.0,
            roughness: 0.8,
            tint: Vec3::ONE,
        }
    }
}

// ---------------------------------------------------------------------------
// TerrainMaterial
// ---------------------------------------------------------------------------

/// The complete terrain material definition.
///
/// Combines multiple [`TerrainLayer`]s with splatmaps to control blending.
/// Supports up to 16 layers through multiple splatmaps (4 channels each).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainMaterial {
    /// Material layers (up to 16 with 4 splatmaps).
    pub layers: Vec<TerrainLayer>,
    /// Splatmaps controlling layer blend weights.
    /// Each splatmap controls 4 layers.
    pub splatmaps: Vec<SplatMap>,
    /// Enable triplanar mapping for steep surfaces.
    pub triplanar_enabled: bool,
    /// Slope threshold (in radians) above which triplanar mapping kicks in.
    pub triplanar_slope_threshold: f32,
    /// Sharpness of the triplanar blending (higher = sharper seams).
    pub triplanar_sharpness: f32,
    /// Detail texture layer (overlaid at close range for extra detail).
    pub detail_layer: Option<DetailLayer>,
    /// Macro variation map for large-scale color variation.
    pub macro_variation: Option<MacroVariation>,
}

impl Default for TerrainMaterial {
    fn default() -> Self {
        Self {
            layers: vec![TerrainLayer::default()],
            splatmaps: vec![],
            triplanar_enabled: true,
            triplanar_slope_threshold: 0.8, // ~46 degrees
            triplanar_sharpness: 8.0,
            detail_layer: None,
            macro_variation: None,
        }
    }
}

impl TerrainMaterial {
    /// Creates a new terrain material with the given layers.
    pub fn new(layers: Vec<TerrainLayer>) -> Self {
        Self {
            layers,
            ..Default::default()
        }
    }

    /// Returns the number of splatmaps needed for the current layer count.
    pub fn required_splatmap_count(&self) -> usize {
        (self.layers.len() + 3) / 4
    }

    /// Computes the blended material properties at a given UV coordinate.
    ///
    /// Returns the weighted combination of all layer properties based on
    /// splatmap weights at that position.
    pub fn sample_blended(
        &self,
        u: f32,
        v: f32,
        height_values: &[f32],
    ) -> BlendedMaterial {
        let mut result = BlendedMaterial::default();
        let mut total_weight = 0.0f32;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let splatmap_idx = layer_idx / 4;
            let channel = layer_idx % 4;

            let base_weight = if splatmap_idx < self.splatmaps.len() {
                let weights = self.splatmaps[splatmap_idx].sample(u, v);
                match channel {
                    0 => weights.x,
                    1 => weights.y,
                    2 => weights.z,
                    3 => weights.w,
                    _ => 0.0,
                }
            } else {
                if layer_idx == 0 {
                    1.0
                } else {
                    0.0
                }
            };

            // Apply height-based blending
            let height_value = if layer_idx < height_values.len() {
                height_values[layer_idx]
            } else {
                0.5
            };

            let weight = height_blend(base_weight, height_value, layer.height_blend_sharpness);

            result.roughness += layer.roughness * weight;
            result.metallic += layer.metallic * weight;
            result.tint = result.tint + layer.tint * weight;
            result.tiling_scale += layer.tiling_scale * weight;
            total_weight += weight;
        }

        if total_weight > f32::EPSILON {
            let inv = 1.0 / total_weight;
            result.roughness *= inv;
            result.metallic *= inv;
            result.tint = result.tint * inv;
            result.tiling_scale *= inv;
        }

        result
    }

    /// Computes triplanar blend weights for a given surface normal.
    ///
    /// Returns weights for the X, Y, and Z projections. Y-projection
    /// dominates on flat surfaces; X/Z projections take over on steep
    /// surfaces.
    pub fn triplanar_weights(&self, normal: Vec3) -> Vec3 {
        let abs_normal = normal.abs();
        let mut weights = Vec3::new(
            abs_normal.x.powf(self.triplanar_sharpness),
            abs_normal.y.powf(self.triplanar_sharpness),
            abs_normal.z.powf(self.triplanar_sharpness),
        );

        let sum = weights.x + weights.y + weights.z;
        if sum > f32::EPSILON {
            weights /= sum;
        }

        weights
    }

    /// Determines whether triplanar mapping should be used at a given
    /// surface normal.
    pub fn should_use_triplanar(&self, normal: Vec3) -> bool {
        if !self.triplanar_enabled {
            return false;
        }
        let slope = normal.y.acos();
        slope > self.triplanar_slope_threshold
    }

    /// Computes triplanar UVs for a given world position and normal.
    pub fn triplanar_uvs(
        &self,
        world_pos: Vec3,
        normal: Vec3,
        tiling: f32,
    ) -> TriplanarUVs {
        let weights = self.triplanar_weights(normal);

        TriplanarUVs {
            uv_x: Vec2::new(world_pos.z * tiling, world_pos.y * tiling),
            uv_y: Vec2::new(world_pos.x * tiling, world_pos.z * tiling),
            uv_z: Vec2::new(world_pos.x * tiling, world_pos.y * tiling),
            weights,
        }
    }
}

/// Blended material properties at a single point.
#[derive(Debug, Clone, Default)]
pub struct BlendedMaterial {
    /// Blended roughness.
    pub roughness: f32,
    /// Blended metallic.
    pub metallic: f32,
    /// Blended color tint.
    pub tint: Vec3,
    /// Blended tiling scale.
    pub tiling_scale: f32,
}

/// Triplanar UV coordinates and blend weights.
#[derive(Debug, Clone)]
pub struct TriplanarUVs {
    /// UVs for X-axis projection (sampling YZ plane).
    pub uv_x: Vec2,
    /// UVs for Y-axis projection (sampling XZ plane — flat surfaces).
    pub uv_y: Vec2,
    /// UVs for Z-axis projection (sampling XY plane).
    pub uv_z: Vec2,
    /// Blend weights for each projection axis.
    pub weights: Vec3,
}

// ---------------------------------------------------------------------------
// Detail layer
// ---------------------------------------------------------------------------

/// A detail texture overlaid at close range for additional surface detail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailLayer {
    /// Detail albedo/normal texture path.
    pub texture_path: String,
    /// Tiling scale (much higher than base layers for fine detail).
    pub tiling_scale: f32,
    /// Maximum distance at which the detail layer is visible.
    pub fade_distance: f32,
    /// Blend strength of the detail overlay.
    pub strength: f32,
}

impl Default for DetailLayer {
    fn default() -> Self {
        Self {
            texture_path: String::new(),
            tiling_scale: 50.0,
            fade_distance: 100.0,
            strength: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Macro variation
// ---------------------------------------------------------------------------

/// Large-scale color variation applied across the entire terrain.
///
/// This prevents the terrain from looking uniformly tiled at a distance
/// by modulating the color with a low-frequency pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroVariation {
    /// Macro variation texture path (grayscale).
    pub texture_path: String,
    /// UV scale (very low — typically 0.01..0.1 so the pattern covers
    /// the entire terrain).
    pub uv_scale: f32,
    /// Intensity of the variation (0 = none, 1 = full).
    pub intensity: f32,
}

impl Default for MacroVariation {
    fn default() -> Self {
        Self {
            texture_path: String::new(),
            uv_scale: 0.02,
            intensity: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// Auto-texturing rules
// ---------------------------------------------------------------------------

/// A rule for automatic splatmap generation based on terrain properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TexturingRule {
    /// The splatmap channel this rule writes to (0-3).
    pub channel: usize,
    /// The splatmap index (for > 4 layers).
    pub splatmap_index: usize,
    /// Minimum altitude (heightmap value) for this layer.
    pub altitude_min: f32,
    /// Maximum altitude for this layer.
    pub altitude_max: f32,
    /// Transition width at the altitude boundaries (for soft blending).
    pub altitude_blend: f32,
    /// Minimum slope angle (radians) for this layer.
    pub slope_min: f32,
    /// Maximum slope angle for this layer.
    pub slope_max: f32,
    /// Transition width at the slope boundaries.
    pub slope_blend: f32,
    /// Optional noise amplitude to break up uniform boundaries.
    pub noise_amplitude: f32,
    /// Noise frequency.
    pub noise_frequency: f32,
}

impl Default for TexturingRule {
    fn default() -> Self {
        Self {
            channel: 0,
            splatmap_index: 0,
            altitude_min: 0.0,
            altitude_max: 1.0,
            altitude_blend: 0.05,
            slope_min: 0.0,
            slope_max: std::f32::consts::FRAC_PI_2,
            slope_blend: 0.1,
            noise_amplitude: 0.0,
            noise_frequency: 1.0,
        }
    }
}

/// Generates splatmaps automatically from terrain rules.
///
/// Each rule defines altitude and slope ranges for a specific material layer.
/// The function evaluates all rules at every texel and writes the appropriate
/// weights to the splatmaps.
pub fn generate_splatmap_from_rules(
    heightmap: &Heightmap,
    rules: &[TexturingRule],
    splatmap_width: u32,
    splatmap_height: u32,
) -> Vec<SplatMap> {
    // Determine how many splatmaps we need
    let max_splatmap_idx = rules
        .iter()
        .map(|r| r.splatmap_index)
        .max()
        .unwrap_or(0);

    let mut splatmaps: Vec<SplatMap> = (0..=max_splatmap_idx)
        .map(|_| SplatMap::new(splatmap_width, splatmap_height))
        .collect();

    let hm_w = heightmap.width() as f32;
    let hm_h = heightmap.height() as f32;

    // Simple hash-based noise for boundary variation
    let noise = |x: f32, z: f32, freq: f32| -> f32 {
        let ix = (x * freq * 1000.0) as i32;
        let iz = (z * freq * 1000.0) as i32;
        let hash = ((ix.wrapping_mul(374761393))
            .wrapping_add(iz.wrapping_mul(668265263)))
        .wrapping_mul(1274126177);
        (hash & 0x7FFF) as f32 / 0x7FFF as f32
    };

    for sz in 0..splatmap_height {
        for sx in 0..splatmap_width {
            // Map splatmap texel to heightmap coordinates
            let hx = sx as f32 / (splatmap_width - 1) as f32 * (hm_w - 1.0);
            let hz = sz as f32 / (splatmap_height - 1) as f32 * (hm_h - 1.0);

            let altitude = heightmap.sample(hx, hz);
            let slope = heightmap.slope_at(hx, hz);

            for rule in rules {
                // Compute altitude weight
                let alt_weight = soft_range(
                    altitude,
                    rule.altitude_min,
                    rule.altitude_max,
                    rule.altitude_blend,
                );

                // Compute slope weight
                let slope_weight = soft_range(
                    slope,
                    rule.slope_min,
                    rule.slope_max,
                    rule.slope_blend,
                );

                // Combine with optional noise
                let noise_offset = if rule.noise_amplitude > 0.0 {
                    (noise(hx, hz, rule.noise_frequency) - 0.5) * rule.noise_amplitude
                } else {
                    0.0
                };

                let weight = (alt_weight * slope_weight + noise_offset).clamp(0.0, 1.0);

                if rule.splatmap_index < splatmaps.len() && rule.channel < 4 {
                    let current = splatmaps[rule.splatmap_index].get_weight(sx, sz, rule.channel);
                    splatmaps[rule.splatmap_index].set_weight(
                        sx,
                        sz,
                        rule.channel,
                        (current + weight).min(1.0),
                    );
                }
            }
        }
    }

    // Normalize all splatmaps
    for sm in &mut splatmaps {
        sm.normalize();
    }

    splatmaps
}

/// Creates a default set of texturing rules for a natural-looking terrain.
///
/// - Channel 0: Sand/dirt at low altitudes and flat areas
/// - Channel 1: Grass at mid altitudes
/// - Channel 2: Rock on steep slopes
/// - Channel 3: Snow at high altitudes
pub fn default_natural_rules() -> Vec<TexturingRule> {
    vec![
        // Sand/dirt — low altitude, flat
        TexturingRule {
            channel: 0,
            splatmap_index: 0,
            altitude_min: 0.0,
            altitude_max: 0.3,
            altitude_blend: 0.05,
            slope_min: 0.0,
            slope_max: 0.5,
            slope_blend: 0.1,
            noise_amplitude: 0.05,
            noise_frequency: 2.0,
        },
        // Grass — mid altitude, moderate slope
        TexturingRule {
            channel: 1,
            splatmap_index: 0,
            altitude_min: 0.15,
            altitude_max: 0.65,
            altitude_blend: 0.08,
            slope_min: 0.0,
            slope_max: 0.7,
            slope_blend: 0.1,
            noise_amplitude: 0.08,
            noise_frequency: 3.0,
        },
        // Rock — steep slopes at any altitude
        TexturingRule {
            channel: 2,
            splatmap_index: 0,
            altitude_min: 0.0,
            altitude_max: 1.0,
            altitude_blend: 0.0,
            slope_min: 0.5,
            slope_max: std::f32::consts::FRAC_PI_2,
            slope_blend: 0.15,
            noise_amplitude: 0.03,
            noise_frequency: 5.0,
        },
        // Snow — high altitude, any slope
        TexturingRule {
            channel: 3,
            splatmap_index: 0,
            altitude_min: 0.6,
            altitude_max: 1.0,
            altitude_blend: 0.1,
            slope_min: 0.0,
            slope_max: 0.8,
            slope_blend: 0.15,
            noise_amplitude: 0.06,
            noise_frequency: 4.0,
        },
    ]
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Height-based blending: sharper transitions using the height map value.
///
/// `base_weight` is the splatmap weight. `height_value` is sampled from the
/// layer's height map. `sharpness` controls transition sharpness.
fn height_blend(base_weight: f32, height_value: f32, sharpness: f32) -> f32 {
    let adjusted = base_weight + height_value;
    let threshold = 1.0; // midpoint
    let range = 1.0 / sharpness.max(0.001);
    smoothstep(threshold - range, threshold + range, adjusted) * base_weight
}

/// Hermite smoothstep.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Computes a weight for a value within a range with soft edges.
fn soft_range(value: f32, min: f32, max: f32, blend: f32) -> f32 {
    let lower = smoothstep(min - blend, min + blend, value);
    let upper = 1.0 - smoothstep(max - blend, max + blend, value);
    lower * upper
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splatmap_default() {
        let sm = SplatMap::new_default(16, 16);
        let w = sm.get_weights(0, 0);
        assert_eq!(w.x, 1.0);
        assert_eq!(w.y, 0.0);
    }

    #[test]
    fn splatmap_paint() {
        let mut sm = SplatMap::new_default(32, 32);
        sm.paint_brush(0.5, 0.5, 5.0, 1, 1.0, BrushFalloff::Smooth);

        let center = sm.get_weights(15, 15);
        // Channel 1 should have gained weight
        assert!(center.y > 0.0);
    }

    #[test]
    fn splatmap_normalize() {
        let mut sm = SplatMap::new(4, 4);
        sm.set_weights(0, 0, Vec4::new(2.0, 3.0, 1.0, 4.0));
        sm.normalize();
        let w = sm.get_weights(0, 0);
        let sum = w.x + w.y + w.z + w.w;
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn auto_texturing_rules() {
        let hm = crate::Heightmap::generate_procedural(65, 0.5, 42).unwrap();
        let rules = default_natural_rules();
        let splatmaps = generate_splatmap_from_rules(&hm, &rules, 64, 64);
        assert_eq!(splatmaps.len(), 1);

        // Verify normalization
        let w = splatmaps[0].get_weights(32, 32);
        let sum = w.x + w.y + w.z + w.w;
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn triplanar_weights_flat() {
        let mat = TerrainMaterial::default();
        let w = mat.triplanar_weights(Vec3::Y);
        // Y-projection should dominate on flat surfaces
        assert!(w.y > w.x);
        assert!(w.y > w.z);
    }

    #[test]
    fn triplanar_weights_steep() {
        let mat = TerrainMaterial::default();
        let w = mat.triplanar_weights(Vec3::X);
        // X-projection should dominate on an X-facing wall
        assert!(w.x > w.y);
        assert!(w.x > w.z);
    }

    #[test]
    fn splatmap_bilinear_sampling() {
        let mut sm = SplatMap::new(4, 4);
        sm.set_weights(0, 0, Vec4::new(1.0, 0.0, 0.0, 0.0));
        sm.set_weights(1, 0, Vec4::new(0.0, 1.0, 0.0, 0.0));
        let mid = sm.sample(0.5 / 3.0, 0.0);
        // Should be an interpolation between (1,0,0,0) and (0,1,0,0)
        assert!(mid.x > 0.0 && mid.x < 1.0);
        assert!(mid.y > 0.0 && mid.y < 1.0);
    }

    #[test]
    fn splatmap_to_rgba8() {
        let sm = SplatMap::new_default(2, 2);
        let bytes = sm.to_rgba8();
        assert_eq!(bytes.len(), 2 * 2 * 4);
        assert_eq!(bytes[0], 255); // channel 0 = 1.0 -> 255
        assert_eq!(bytes[1], 0);   // channel 1 = 0.0 -> 0
    }
}
