// engine/render/src/indirect_lighting.rs
//
// Indirect lighting collection and blending system for the Genovo renderer.
//
// Gathers indirect illumination from multiple sources -- light probes,
// reflection probes, lightmaps, screen-space global illumination (SSGI),
// volumetric probes -- and blends/weights them into a unified per-pixel
// indirect diffuse + specular result.
//
// # Architecture
//
// The `IndirectLightingSystem` manages all probe grids, lightmap references,
// and SSGI buffers. Each frame, it evaluates a fallback chain per pixel:
//
// 1. SSGI (if available and quality permits)
// 2. Volumetric probe grid (if the pixel lies within a volume)
// 3. Light probe tetrahedralization / SH interpolation
// 4. Reflection probe blending with parallax correction
// 5. Lightmap lookup (for static geometry)
// 6. Flat ambient fallback
//
// Sources are weighted by quality, proximity, and confidence. The final
// indirect term is split into diffuse and specular and passed to the
// compositing stage.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Math types
// ---------------------------------------------------------------------------

/// 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0, z: 1.0 };

    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }

    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn length(self) -> f32 { self.dot(self).sqrt() }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len > 1e-7 { self.scale(1.0 / len) } else { Self::ZERO }
    }

    #[inline]
    pub fn add(self, o: Self) -> Self { Self::new(self.x + o.x, self.y + o.y, self.z + o.z) }

    #[inline]
    pub fn sub(self, o: Self) -> Self { Self::new(self.x - o.x, self.y - o.y, self.z - o.z) }

    #[inline]
    pub fn scale(self, s: f32) -> Self { Self::new(self.x * s, self.y * s, self.z * s) }

    #[inline]
    pub fn mul_comp(self, o: Self) -> Self { Self::new(self.x * o.x, self.y * o.y, self.z * o.z) }

    #[inline]
    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    #[inline]
    pub fn distance(self, o: Self) -> f32 { self.sub(o).length() }

    #[inline]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        self.scale(1.0 - t).add(other.scale(t))
    }

    #[inline]
    pub fn max_component(self) -> f32 { self.x.max(self.y).max(self.z) }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of light probes that can influence a single pixel.
pub const MAX_LIGHT_PROBES_PER_PIXEL: usize = 4;

/// Maximum number of reflection probes per pixel.
pub const MAX_REFLECTION_PROBES_PER_PIXEL: usize = 2;

/// Maximum SH band order supported (L2 = 9 coefficients).
pub const MAX_SH_ORDER: usize = 3;

/// Number of SH coefficients for L2 (3 bands: L0, L1, L2).
pub const SH_COEFF_COUNT: usize = 9;

/// Default indirect intensity multiplier.
pub const DEFAULT_INDIRECT_INTENSITY: f32 = 1.0;

/// Default ambient occlusion power.
pub const DEFAULT_AO_POWER: f32 = 1.0;

/// Minimum roughness for specular indirect.
pub const MIN_SPECULAR_ROUGHNESS: f32 = 0.04;

/// Maximum number of volumetric probe cells per axis.
pub const MAX_VOLUME_PROBE_RESOLUTION: u32 = 64;

/// Lightmap max resolution.
pub const MAX_LIGHTMAP_RESOLUTION: u32 = 4096;

/// Default SSGI influence radius in world units.
pub const DEFAULT_SSGI_RADIUS: f32 = 5.0;

/// Fallback ambient color when no other source is available.
pub const FALLBACK_AMBIENT: Vec3 = Vec3 { x: 0.03, y: 0.03, z: 0.03 };

// ---------------------------------------------------------------------------
// Spherical harmonics (SH L2)
// ---------------------------------------------------------------------------

/// L2 spherical harmonics coefficients (9 RGB coefficients).
#[derive(Debug, Clone)]
pub struct SphericalHarmonicsL2 {
    /// SH coefficients: 9 Vec3 values (R, G, B per coefficient).
    pub coefficients: [Vec3; SH_COEFF_COUNT],
}

impl Default for SphericalHarmonicsL2 {
    fn default() -> Self {
        Self {
            coefficients: [Vec3::ZERO; SH_COEFF_COUNT],
        }
    }
}

impl SphericalHarmonicsL2 {
    /// Create SH from a uniform color.
    pub fn from_ambient(color: Vec3) -> Self {
        let mut sh = Self::default();
        // L0 band (DC term) encodes the average irradiance.
        let sqrt_4pi = (4.0 * std::f32::consts::PI).sqrt();
        sh.coefficients[0] = color.scale(sqrt_4pi);
        sh
    }

    /// Evaluate the SH for a given direction.
    pub fn evaluate(&self, direction: Vec3) -> Vec3 {
        let d = direction.normalize();

        // SH basis functions (real, L2).
        let y0 = 0.282095; // Y_0^0 = 1 / (2 * sqrt(pi))
        let y1 = 0.488603; // Y_1^{-1,0,1}
        let y2_0 = 1.092548; // Y_2^{-2,-1}
        let y2_1 = 0.315392; // Y_2^0
        let y2_2 = 0.546274; // Y_2^{1,2}

        let basis = [
            y0,
            y1 * d.y,
            y1 * d.z,
            y1 * d.x,
            y2_0 * d.x * d.y,
            y2_0 * d.y * d.z,
            y2_1 * (3.0 * d.z * d.z - 1.0),
            y2_0 * d.x * d.z,
            y2_2 * (d.x * d.x - d.y * d.y),
        ];

        let mut result = Vec3::ZERO;
        for i in 0..SH_COEFF_COUNT {
            result = result.add(self.coefficients[i].scale(basis[i]));
        }

        // Clamp to non-negative.
        Vec3::new(result.x.max(0.0), result.y.max(0.0), result.z.max(0.0))
    }

    /// Linearly interpolate between two SH.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let mut result = Self::default();
        for i in 0..SH_COEFF_COUNT {
            result.coefficients[i] = self.coefficients[i].lerp(other.coefficients[i], t);
        }
        result
    }

    /// Add another SH (for accumulation).
    pub fn add(&self, other: &Self) -> Self {
        let mut result = Self::default();
        for i in 0..SH_COEFF_COUNT {
            result.coefficients[i] = self.coefficients[i].add(other.coefficients[i]);
        }
        result
    }

    /// Scale all coefficients.
    pub fn scale(&self, factor: f32) -> Self {
        let mut result = Self::default();
        for i in 0..SH_COEFF_COUNT {
            result.coefficients[i] = self.coefficients[i].scale(factor);
        }
        result
    }

    /// Compute the dominant direction (direction of maximum irradiance).
    pub fn dominant_direction(&self) -> Vec3 {
        // The L1 band coefficients encode the directional component.
        let dir = Vec3::new(
            self.coefficients[3].max_component(),
            self.coefficients[1].max_component(),
            self.coefficients[2].max_component(),
        );
        dir.normalize()
    }

    /// Compute average irradiance (L0 term).
    pub fn average_irradiance(&self) -> Vec3 {
        let sqrt_4pi_inv = 1.0 / (4.0 * std::f32::consts::PI).sqrt();
        self.coefficients[0].scale(sqrt_4pi_inv)
    }
}

// ---------------------------------------------------------------------------
// Light probe
// ---------------------------------------------------------------------------

/// Unique identifier for a light probe.
pub type LightProbeId = u32;

/// A single light probe that stores SH irradiance at a position in space.
#[derive(Debug, Clone)]
pub struct LightProbe {
    /// Unique identifier.
    pub id: LightProbeId,
    /// World-space position.
    pub position: Vec3,
    /// SH irradiance at this position.
    pub irradiance: SphericalHarmonicsL2,
    /// Influence radius.
    pub radius: f32,
    /// Priority (higher overrides lower in overlap regions).
    pub priority: u32,
    /// Whether this probe is valid (has been baked).
    pub valid: bool,
    /// Whether this probe is interior (indoor lighting).
    pub interior: bool,
}

impl LightProbe {
    /// Create a new light probe.
    pub fn new(id: LightProbeId, position: Vec3, radius: f32) -> Self {
        Self {
            id,
            position,
            irradiance: SphericalHarmonicsL2::default(),
            radius,
            priority: 0,
            valid: false,
            interior: false,
        }
    }

    /// Set the irradiance from baked data.
    pub fn set_irradiance(&mut self, sh: SphericalHarmonicsL2) {
        self.irradiance = sh;
        self.valid = true;
    }

    /// Compute the weight of this probe for a given world position.
    pub fn weight_at(&self, world_pos: Vec3) -> f32 {
        if !self.valid {
            return 0.0;
        }
        let dist = self.position.distance(world_pos);
        if dist >= self.radius {
            return 0.0;
        }
        // Smooth falloff.
        let t = dist / self.radius;
        let falloff = 1.0 - t * t * (3.0 - 2.0 * t); // smoothstep
        falloff
    }
}

// ---------------------------------------------------------------------------
// Light probe grid
// ---------------------------------------------------------------------------

/// A grid of light probes arranged in 3D space.
#[derive(Debug, Clone)]
pub struct LightProbeGrid {
    /// All probes in the grid.
    probes: Vec<LightProbe>,
    /// Grid origin (world space).
    origin: Vec3,
    /// Grid cell size.
    cell_size: Vec3,
    /// Grid dimensions (number of probes per axis).
    dimensions: [u32; 3],
    /// Next probe ID.
    next_id: LightProbeId,
}

impl LightProbeGrid {
    /// Create a new probe grid.
    pub fn new(origin: Vec3, cell_size: Vec3, dimensions: [u32; 3]) -> Self {
        let total = (dimensions[0] * dimensions[1] * dimensions[2]) as usize;
        let mut probes = Vec::with_capacity(total);
        let mut next_id: LightProbeId = 0;

        for z in 0..dimensions[2] {
            for y in 0..dimensions[1] {
                for x in 0..dimensions[0] {
                    let pos = Vec3::new(
                        origin.x + x as f32 * cell_size.x,
                        origin.y + y as f32 * cell_size.y,
                        origin.z + z as f32 * cell_size.z,
                    );
                    let radius = cell_size.x.max(cell_size.y).max(cell_size.z) * 1.5;
                    probes.push(LightProbe::new(next_id, pos, radius));
                    next_id += 1;
                }
            }
        }

        Self {
            probes,
            origin,
            cell_size,
            dimensions,
            next_id,
        }
    }

    /// Get the probe at grid coordinates.
    pub fn get_probe(&self, x: u32, y: u32, z: u32) -> Option<&LightProbe> {
        if x < self.dimensions[0] && y < self.dimensions[1] && z < self.dimensions[2] {
            let idx = (z * self.dimensions[1] * self.dimensions[0]
                + y * self.dimensions[0]
                + x) as usize;
            self.probes.get(idx)
        } else {
            None
        }
    }

    /// Get a mutable probe at grid coordinates.
    pub fn get_probe_mut(&mut self, x: u32, y: u32, z: u32) -> Option<&mut LightProbe> {
        if x < self.dimensions[0] && y < self.dimensions[1] && z < self.dimensions[2] {
            let idx = (z * self.dimensions[1] * self.dimensions[0]
                + y * self.dimensions[0]
                + x) as usize;
            self.probes.get_mut(idx)
        } else {
            None
        }
    }

    /// Find the grid cell containing a world position.
    pub fn world_to_grid(&self, world_pos: Vec3) -> Option<[u32; 3]> {
        let local = world_pos.sub(self.origin);
        let gx = (local.x / self.cell_size.x).floor() as i32;
        let gy = (local.y / self.cell_size.y).floor() as i32;
        let gz = (local.z / self.cell_size.z).floor() as i32;

        if gx >= 0
            && gy >= 0
            && gz >= 0
            && (gx as u32) < self.dimensions[0]
            && (gy as u32) < self.dimensions[1]
            && (gz as u32) < self.dimensions[2]
        {
            Some([gx as u32, gy as u32, gz as u32])
        } else {
            None
        }
    }

    /// Sample the probe grid at a world position using trilinear interpolation.
    pub fn sample(&self, world_pos: Vec3) -> SphericalHarmonicsL2 {
        let local = world_pos.sub(self.origin);
        let fx = local.x / self.cell_size.x;
        let fy = local.y / self.cell_size.y;
        let fz = local.z / self.cell_size.z;

        let x0 = (fx.floor() as u32).min(self.dimensions[0].saturating_sub(2));
        let y0 = (fy.floor() as u32).min(self.dimensions[1].saturating_sub(2));
        let z0 = (fz.floor() as u32).min(self.dimensions[2].saturating_sub(2));

        let tx = (fx - x0 as f32).clamp(0.0, 1.0);
        let ty = (fy - y0 as f32).clamp(0.0, 1.0);
        let tz = (fz - z0 as f32).clamp(0.0, 1.0);

        // Fetch 8 corner probes.
        let get = |x: u32, y: u32, z: u32| -> &SphericalHarmonicsL2 {
            &self.probes[(z * self.dimensions[1] * self.dimensions[0]
                + y * self.dimensions[0]
                + x) as usize]
                .irradiance
        };

        let c000 = get(x0, y0, z0);
        let c100 = get(x0 + 1, y0, z0);
        let c010 = get(x0, y0 + 1, z0);
        let c110 = get(x0 + 1, y0 + 1, z0);
        let c001 = get(x0, y0, z0 + 1);
        let c101 = get(x0 + 1, y0, z0 + 1);
        let c011 = get(x0, y0 + 1, z0 + 1);
        let c111 = get(x0 + 1, y0 + 1, z0 + 1);

        // Trilinear interpolation.
        let c00 = c000.lerp(c100, tx);
        let c10 = c010.lerp(c110, tx);
        let c01 = c001.lerp(c101, tx);
        let c11 = c011.lerp(c111, tx);

        let c0 = c00.lerp(&c10, ty);
        let c1 = c01.lerp(&c11, ty);

        c0.lerp(&c1, tz)
    }

    /// Get all probes.
    pub fn probes(&self) -> &[LightProbe] {
        &self.probes
    }

    /// Get mutable access to all probes.
    pub fn probes_mut(&mut self) -> &mut [LightProbe] {
        &mut self.probes
    }

    /// Get grid dimensions.
    pub fn dimensions(&self) -> [u32; 3] {
        self.dimensions
    }

    /// Get total number of probes.
    pub fn probe_count(&self) -> usize {
        self.probes.len()
    }

    /// Mark all probes as valid (for testing).
    pub fn validate_all(&mut self) {
        for probe in &mut self.probes {
            probe.valid = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Reflection probe
// ---------------------------------------------------------------------------

/// Unique identifier for a reflection probe.
pub type ReflectionProbeId = u32;

/// Reflection probe types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflectionProbeType {
    /// Cubemap captured at probe position.
    Cubemap,
    /// Planar reflection (for flat reflective surfaces).
    Planar,
    /// Screen-space reflections (runtime, not a placed probe).
    ScreenSpace,
}

/// A reflection probe capturing specular reflections at a point in space.
#[derive(Debug, Clone)]
pub struct ReflectionProbe {
    /// Unique identifier.
    pub id: ReflectionProbeId,
    /// World-space position.
    pub position: Vec3,
    /// Probe type.
    pub probe_type: ReflectionProbeType,
    /// Axis-aligned influence box minimum.
    pub influence_min: Vec3,
    /// Axis-aligned influence box maximum.
    pub influence_max: Vec3,
    /// Inner box for smooth blending (normalized within influence).
    pub blend_distance: f32,
    /// Cubemap resolution.
    pub resolution: u32,
    /// Number of mip levels (for roughness-based sampling).
    pub mip_count: u32,
    /// Whether parallax correction is enabled.
    pub parallax_correction: bool,
    /// Priority for blending order.
    pub priority: u32,
    /// Whether this probe has been baked/captured.
    pub valid: bool,
    /// HDR intensity multiplier.
    pub intensity: f32,
    /// Average color of the cubemap (for quick fallback).
    pub average_color: Vec3,
    /// Dominant reflection direction.
    pub dominant_direction: Vec3,
}

impl ReflectionProbe {
    /// Create a new reflection probe.
    pub fn new(id: ReflectionProbeId, position: Vec3, half_extents: Vec3) -> Self {
        Self {
            id,
            position,
            probe_type: ReflectionProbeType::Cubemap,
            influence_min: position.sub(half_extents),
            influence_max: position.add(half_extents),
            blend_distance: 1.0,
            resolution: 256,
            mip_count: 8,
            parallax_correction: true,
            priority: 0,
            valid: false,
            intensity: 1.0,
            average_color: Vec3::ZERO,
            dominant_direction: Vec3::new(0.0, 1.0, 0.0),
        }
    }

    /// Compute the influence weight at a world position.
    pub fn weight_at(&self, world_pos: Vec3) -> f32 {
        if !self.valid {
            return 0.0;
        }

        // Check if inside influence box.
        if world_pos.x < self.influence_min.x || world_pos.x > self.influence_max.x
            || world_pos.y < self.influence_min.y || world_pos.y > self.influence_max.y
            || world_pos.z < self.influence_min.z || world_pos.z > self.influence_max.z
        {
            return 0.0;
        }

        // Compute distance to box edges for smooth blending.
        let dist_to_min = Vec3::new(
            world_pos.x - self.influence_min.x,
            world_pos.y - self.influence_min.y,
            world_pos.z - self.influence_min.z,
        );
        let dist_to_max = Vec3::new(
            self.influence_max.x - world_pos.x,
            self.influence_max.y - world_pos.y,
            self.influence_max.z - world_pos.z,
        );

        let min_dist = dist_to_min.x.min(dist_to_min.y).min(dist_to_min.z)
            .min(dist_to_max.x).min(dist_to_max.y).min(dist_to_max.z);

        if self.blend_distance <= 0.0 {
            return 1.0;
        }

        (min_dist / self.blend_distance).clamp(0.0, 1.0)
    }

    /// Apply parallax correction to a reflection direction.
    pub fn correct_reflection(&self, world_pos: Vec3, reflection_dir: Vec3) -> Vec3 {
        if !self.parallax_correction {
            return reflection_dir;
        }

        // Intersect the reflection ray with the influence box.
        let rd = reflection_dir.normalize();

        let mut t_min = f32::MIN;
        let mut t_max = f32::MAX;

        // X axis.
        if rd.x.abs() > 1e-7 {
            let t1 = (self.influence_min.x - world_pos.x) / rd.x;
            let t2 = (self.influence_max.x - world_pos.x) / rd.x;
            let (t_near, t_far) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
            t_min = t_min.max(t_near);
            t_max = t_max.min(t_far);
        }

        // Y axis.
        if rd.y.abs() > 1e-7 {
            let t1 = (self.influence_min.y - world_pos.y) / rd.y;
            let t2 = (self.influence_max.y - world_pos.y) / rd.y;
            let (t_near, t_far) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
            t_min = t_min.max(t_near);
            t_max = t_max.min(t_far);
        }

        // Z axis.
        if rd.z.abs() > 1e-7 {
            let t1 = (self.influence_min.z - world_pos.z) / rd.z;
            let t2 = (self.influence_max.z - world_pos.z) / rd.z;
            let (t_near, t_far) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
            t_min = t_min.max(t_near);
            t_max = t_max.min(t_far);
        }

        let t = if t_max > 0.0 { t_max } else { 0.0 };
        let intersection = Vec3::new(
            world_pos.x + rd.x * t,
            world_pos.y + rd.y * t,
            world_pos.z + rd.z * t,
        );

        // The corrected direction is from the probe center to the intersection.
        intersection.sub(self.position).normalize()
    }

    /// Select the mip level based on roughness.
    pub fn roughness_to_mip(&self, roughness: f32) -> f32 {
        roughness * (self.mip_count.saturating_sub(1)) as f32
    }
}

// ---------------------------------------------------------------------------
// Lightmap
// ---------------------------------------------------------------------------

/// Unique identifier for a lightmap.
pub type LightmapId = u32;

/// A baked lightmap texture storing indirect irradiance.
#[derive(Debug, Clone)]
pub struct Lightmap {
    /// Unique identifier.
    pub id: LightmapId,
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// HDR pixel data (RGB per texel).
    pub pixels: Vec<Vec3>,
    /// Whether the lightmap has directional data (SH or dominant direction).
    pub directional: bool,
    /// Optional directional lightmap data (dominant direction per texel).
    pub directions: Vec<Vec3>,
    /// Encoding format.
    pub encoding: LightmapEncoding,
    /// Scale factor for the stored values.
    pub intensity_scale: f32,
}

/// Lightmap encoding format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightmapEncoding {
    /// Linear HDR (full float).
    LinearHdr,
    /// RGBM encoding (RGB * M).
    Rgbm,
    /// LogLuv encoding.
    LogLuv,
    /// RGBE (Radiance HDR).
    Rgbe,
}

impl Lightmap {
    /// Create a new empty lightmap.
    pub fn new(id: LightmapId, width: u32, height: u32) -> Self {
        let len = (width * height) as usize;
        Self {
            id,
            width,
            height,
            pixels: vec![Vec3::ZERO; len],
            directional: false,
            directions: Vec::new(),
            encoding: LightmapEncoding::LinearHdr,
            intensity_scale: 1.0,
        }
    }

    /// Sample the lightmap at UV coordinates with bilinear filtering.
    pub fn sample(&self, u: f32, v: f32) -> Vec3 {
        let fx = (u * (self.width as f32 - 1.0)).clamp(0.0, (self.width - 1) as f32);
        let fy = (v * (self.height as f32 - 1.0)).clamp(0.0, (self.height - 1) as f32);

        let x0 = fx.floor() as u32;
        let y0 = fy.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);

        let s = fx - x0 as f32;
        let t = fy - y0 as f32;

        let c00 = self.get_pixel(x0, y0);
        let c10 = self.get_pixel(x1, y0);
        let c01 = self.get_pixel(x0, y1);
        let c11 = self.get_pixel(x1, y1);

        let top = c00.lerp(c10, s);
        let bot = c01.lerp(c11, s);
        top.lerp(bot, t).scale(self.intensity_scale)
    }

    /// Get a pixel by coordinates.
    #[inline]
    pub fn get_pixel(&self, x: u32, y: u32) -> Vec3 {
        self.pixels[(y * self.width + x) as usize]
    }

    /// Set a pixel.
    #[inline]
    pub fn set_pixel(&mut self, x: u32, y: u32, color: Vec3) {
        self.pixels[(y * self.width + x) as usize] = color;
    }

    /// Get the average color of the lightmap.
    pub fn average_color(&self) -> Vec3 {
        if self.pixels.is_empty() {
            return Vec3::ZERO;
        }
        let mut sum = Vec3::ZERO;
        for p in &self.pixels {
            sum = sum.add(*p);
        }
        sum.scale(1.0 / self.pixels.len() as f32)
    }
}

// ---------------------------------------------------------------------------
// SSGI buffer
// ---------------------------------------------------------------------------

/// Screen-space global illumination buffer.
#[derive(Debug, Clone)]
pub struct SsgiBuffer {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Per-pixel indirect diffuse irradiance.
    pub diffuse: Vec<Vec3>,
    /// Per-pixel indirect specular radiance.
    pub specular: Vec<Vec3>,
    /// Per-pixel confidence (0 = no data, 1 = fully converged).
    pub confidence: Vec<f32>,
    /// Whether the buffer has valid data.
    pub valid: bool,
    /// Quality level used to generate this buffer.
    pub quality: SsgiQuality,
}

/// SSGI quality levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsgiQuality {
    /// Low quality: fewer samples, smaller radius.
    Low,
    /// Medium quality.
    Medium,
    /// High quality: more samples, full radius.
    High,
    /// Ultra quality: maximum samples with temporal accumulation.
    Ultra,
    /// Disabled.
    Off,
}

impl SsgiBuffer {
    /// Create a new SSGI buffer.
    pub fn new(width: u32, height: u32, quality: SsgiQuality) -> Self {
        let len = (width * height) as usize;
        Self {
            width,
            height,
            diffuse: vec![Vec3::ZERO; len],
            specular: vec![Vec3::ZERO; len],
            confidence: vec![0.0; len],
            valid: false,
            quality,
        }
    }

    /// Sample the SSGI buffer at a pixel location.
    pub fn sample_diffuse(&self, x: u32, y: u32) -> Vec3 {
        if !self.valid || x >= self.width || y >= self.height {
            return Vec3::ZERO;
        }
        self.diffuse[(y * self.width + x) as usize]
    }

    /// Sample specular SSGI.
    pub fn sample_specular(&self, x: u32, y: u32) -> Vec3 {
        if !self.valid || x >= self.width || y >= self.height {
            return Vec3::ZERO;
        }
        self.specular[(y * self.width + x) as usize]
    }

    /// Get confidence at a pixel.
    pub fn get_confidence(&self, x: u32, y: u32) -> f32 {
        if !self.valid || x >= self.width || y >= self.height {
            return 0.0;
        }
        self.confidence[(y * self.width + x) as usize]
    }

    /// Set diffuse and specular at a pixel.
    pub fn set_pixel(&mut self, x: u32, y: u32, diffuse: Vec3, specular: Vec3, confidence: f32) {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) as usize;
            self.diffuse[idx] = diffuse;
            self.specular[idx] = specular;
            self.confidence[idx] = confidence;
        }
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        for d in &mut self.diffuse {
            *d = Vec3::ZERO;
        }
        for s in &mut self.specular {
            *s = Vec3::ZERO;
        }
        for c in &mut self.confidence {
            *c = 0.0;
        }
        self.valid = false;
    }

    /// Resize the buffer.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        let len = (width * height) as usize;
        self.diffuse = vec![Vec3::ZERO; len];
        self.specular = vec![Vec3::ZERO; len];
        self.confidence = vec![0.0; len];
        self.valid = false;
    }
}

// ---------------------------------------------------------------------------
// Volumetric probe
// ---------------------------------------------------------------------------

/// Volumetric probe grid for 3D indirect lighting.
#[derive(Debug, Clone)]
pub struct VolumetricProbeGrid {
    /// Grid origin (world space).
    pub origin: Vec3,
    /// Cell size.
    pub cell_size: Vec3,
    /// Resolution per axis.
    pub resolution: [u32; 3],
    /// SH per cell.
    pub cells: Vec<SphericalHarmonicsL2>,
    /// Validity mask per cell.
    pub validity: Vec<bool>,
    /// Whether the grid has been baked.
    pub baked: bool,
}

impl VolumetricProbeGrid {
    /// Create a new volumetric probe grid.
    pub fn new(origin: Vec3, cell_size: Vec3, resolution: [u32; 3]) -> Self {
        let total = (resolution[0] * resolution[1] * resolution[2]) as usize;
        Self {
            origin,
            cell_size,
            resolution,
            cells: vec![SphericalHarmonicsL2::default(); total],
            validity: vec![false; total],
            baked: false,
        }
    }

    /// Get the cell index for grid coordinates.
    fn cell_index(&self, x: u32, y: u32, z: u32) -> usize {
        (z * self.resolution[1] * self.resolution[0]
            + y * self.resolution[0]
            + x) as usize
    }

    /// Check if a world position is within the grid.
    pub fn contains(&self, world_pos: Vec3) -> bool {
        let extent = Vec3::new(
            self.resolution[0] as f32 * self.cell_size.x,
            self.resolution[1] as f32 * self.cell_size.y,
            self.resolution[2] as f32 * self.cell_size.z,
        );
        let max = self.origin.add(extent);

        world_pos.x >= self.origin.x
            && world_pos.y >= self.origin.y
            && world_pos.z >= self.origin.z
            && world_pos.x <= max.x
            && world_pos.y <= max.y
            && world_pos.z <= max.z
    }

    /// Sample the volumetric grid at a world position.
    pub fn sample(&self, world_pos: Vec3) -> Option<SphericalHarmonicsL2> {
        if !self.baked || !self.contains(world_pos) {
            return None;
        }

        let local = world_pos.sub(self.origin);
        let fx = local.x / self.cell_size.x;
        let fy = local.y / self.cell_size.y;
        let fz = local.z / self.cell_size.z;

        let x0 = (fx.floor() as u32).min(self.resolution[0].saturating_sub(2));
        let y0 = (fy.floor() as u32).min(self.resolution[1].saturating_sub(2));
        let z0 = (fz.floor() as u32).min(self.resolution[2].saturating_sub(2));

        let tx = (fx - x0 as f32).clamp(0.0, 1.0);
        let ty = (fy - y0 as f32).clamp(0.0, 1.0);
        let tz = (fz - z0 as f32).clamp(0.0, 1.0);

        let get_sh = |x: u32, y: u32, z: u32| -> &SphericalHarmonicsL2 {
            &self.cells[self.cell_index(x, y, z)]
        };

        let c00 = get_sh(x0, y0, z0).lerp(get_sh(x0 + 1, y0, z0), tx);
        let c10 = get_sh(x0, y0 + 1, z0).lerp(get_sh(x0 + 1, y0 + 1, z0), tx);
        let c01 = get_sh(x0, y0, z0 + 1).lerp(get_sh(x0 + 1, y0, z0 + 1), tx);
        let c11 = get_sh(x0, y0 + 1, z0 + 1).lerp(get_sh(x0 + 1, y0 + 1, z0 + 1), tx);

        let c0 = c00.lerp(&c10, ty);
        let c1 = c01.lerp(&c11, ty);

        Some(c0.lerp(&c1, tz))
    }

    /// Set SH data for a cell.
    pub fn set_cell(&mut self, x: u32, y: u32, z: u32, sh: SphericalHarmonicsL2) {
        let idx = self.cell_index(x, y, z);
        self.cells[idx] = sh;
        self.validity[idx] = true;
    }

    /// Total number of cells.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }
}

// ---------------------------------------------------------------------------
// Indirect lighting source weights
// ---------------------------------------------------------------------------

/// Weight configuration for blending different indirect lighting sources.
#[derive(Debug, Clone, Copy)]
pub struct IndirectSourceWeights {
    /// Weight for SSGI contribution.
    pub ssgi_weight: f32,
    /// Weight for volumetric probe contribution.
    pub volumetric_weight: f32,
    /// Weight for light probe grid contribution.
    pub light_probe_weight: f32,
    /// Weight for reflection probe contribution (specular only).
    pub reflection_probe_weight: f32,
    /// Weight for lightmap contribution.
    pub lightmap_weight: f32,
    /// Weight for ambient fallback.
    pub ambient_weight: f32,
}

impl Default for IndirectSourceWeights {
    fn default() -> Self {
        Self {
            ssgi_weight: 1.0,
            volumetric_weight: 1.0,
            light_probe_weight: 1.0,
            reflection_probe_weight: 1.0,
            lightmap_weight: 1.0,
            ambient_weight: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Surface data (input to indirect lighting)
// ---------------------------------------------------------------------------

/// Per-pixel surface data needed for indirect lighting evaluation.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceData {
    /// World-space position.
    pub world_pos: Vec3,
    /// World-space normal (normalized).
    pub normal: Vec3,
    /// View direction (world to camera, normalized).
    pub view_dir: Vec3,
    /// Base color (albedo).
    pub albedo: Vec3,
    /// Roughness (0 = mirror, 1 = diffuse).
    pub roughness: f32,
    /// Metallic factor (0 = dielectric, 1 = metal).
    pub metallic: f32,
    /// Ambient occlusion (0 = fully occluded, 1 = no occlusion).
    pub ao: f32,
    /// Lightmap UV (if available).
    pub lightmap_uv: Option<(f32, f32)>,
    /// Screen-space pixel position.
    pub screen_x: u32,
    pub screen_y: u32,
}

// ---------------------------------------------------------------------------
// Indirect lighting result
// ---------------------------------------------------------------------------

/// Per-pixel indirect lighting result.
#[derive(Debug, Clone, Copy)]
pub struct IndirectLightingResult {
    /// Indirect diffuse irradiance (to be multiplied by albedo in compositing).
    pub diffuse: Vec3,
    /// Indirect specular radiance (already weighted by Fresnel/roughness).
    pub specular: Vec3,
    /// Which sources contributed to this result.
    pub source_mask: IndirectSourceMask,
    /// Total confidence (0 = all fallback, 1 = all high-quality sources).
    pub confidence: f32,
}

impl Default for IndirectLightingResult {
    fn default() -> Self {
        Self {
            diffuse: Vec3::ZERO,
            specular: Vec3::ZERO,
            source_mask: IndirectSourceMask::empty(),
            confidence: 0.0,
        }
    }
}

/// Bitmask indicating which indirect lighting sources contributed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndirectSourceMask(u8);

impl IndirectSourceMask {
    pub const SSGI: u8 = 1 << 0;
    pub const VOLUMETRIC: u8 = 1 << 1;
    pub const LIGHT_PROBE: u8 = 1 << 2;
    pub const REFLECTION_PROBE: u8 = 1 << 3;
    pub const LIGHTMAP: u8 = 1 << 4;
    pub const AMBIENT: u8 = 1 << 5;

    pub fn empty() -> Self { Self(0) }
    pub fn has(&self, flag: u8) -> bool { self.0 & flag != 0 }
    pub fn set(&mut self, flag: u8) { self.0 |= flag; }
    pub fn bits(&self) -> u8 { self.0 }
}

// ---------------------------------------------------------------------------
// Indirect lighting configuration
// ---------------------------------------------------------------------------

/// Configuration for the indirect lighting system.
#[derive(Debug, Clone)]
pub struct IndirectLightingConfig {
    /// Overall indirect intensity multiplier.
    pub intensity: f32,
    /// Source weights.
    pub weights: IndirectSourceWeights,
    /// Whether to apply AO to indirect diffuse.
    pub apply_ao: bool,
    /// AO power (exponent applied to AO factor).
    pub ao_power: f32,
    /// SSGI quality level.
    pub ssgi_quality: SsgiQuality,
    /// Whether to enable specular indirect.
    pub specular_enabled: bool,
    /// Maximum roughness for specular probes (higher roughness uses diffuse only).
    pub max_specular_roughness: f32,
    /// Fallback ambient color.
    pub fallback_ambient: Vec3,
    /// Whether to apply energy conservation (metallic surfaces don't receive diffuse).
    pub energy_conservation: bool,
}

impl Default for IndirectLightingConfig {
    fn default() -> Self {
        Self {
            intensity: DEFAULT_INDIRECT_INTENSITY,
            weights: IndirectSourceWeights::default(),
            apply_ao: true,
            ao_power: DEFAULT_AO_POWER,
            ssgi_quality: SsgiQuality::Medium,
            specular_enabled: true,
            max_specular_roughness: 0.95,
            fallback_ambient: FALLBACK_AMBIENT,
            energy_conservation: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Indirect lighting system
// ---------------------------------------------------------------------------

/// Statistics for the indirect lighting system.
#[derive(Debug, Clone, Copy, Default)]
pub struct IndirectLightingStats {
    /// Number of pixels evaluated.
    pub pixels_evaluated: u32,
    /// Number of pixels using SSGI.
    pub pixels_ssgi: u32,
    /// Number of pixels using volumetric probes.
    pub pixels_volumetric: u32,
    /// Number of pixels using light probes.
    pub pixels_light_probe: u32,
    /// Number of pixels using reflection probes.
    pub pixels_reflection_probe: u32,
    /// Number of pixels using lightmaps.
    pub pixels_lightmap: u32,
    /// Number of pixels falling back to ambient.
    pub pixels_ambient_fallback: u32,
    /// Average confidence.
    pub avg_confidence: f32,
}

/// Main indirect lighting collection system.
pub struct IndirectLightingSystem {
    config: IndirectLightingConfig,
    light_probe_grid: Option<LightProbeGrid>,
    reflection_probes: Vec<ReflectionProbe>,
    lightmaps: HashMap<LightmapId, Lightmap>,
    ssgi: Option<SsgiBuffer>,
    volumetric_grid: Option<VolumetricProbeGrid>,
    stats: IndirectLightingStats,
}

impl IndirectLightingSystem {
    /// Create a new indirect lighting system.
    pub fn new(config: IndirectLightingConfig) -> Self {
        Self {
            config,
            light_probe_grid: None,
            reflection_probes: Vec::new(),
            lightmaps: HashMap::new(),
            ssgi: None,
            volumetric_grid: None,
            stats: IndirectLightingStats::default(),
        }
    }

    /// Set the light probe grid.
    pub fn set_light_probe_grid(&mut self, grid: LightProbeGrid) {
        self.light_probe_grid = Some(grid);
    }

    /// Add a reflection probe.
    pub fn add_reflection_probe(&mut self, probe: ReflectionProbe) {
        self.reflection_probes.push(probe);
    }

    /// Remove a reflection probe by ID.
    pub fn remove_reflection_probe(&mut self, id: ReflectionProbeId) {
        self.reflection_probes.retain(|p| p.id != id);
    }

    /// Add a lightmap.
    pub fn add_lightmap(&mut self, lightmap: Lightmap) {
        self.lightmaps.insert(lightmap.id, lightmap);
    }

    /// Remove a lightmap.
    pub fn remove_lightmap(&mut self, id: LightmapId) {
        self.lightmaps.remove(&id);
    }

    /// Set the SSGI buffer for this frame.
    pub fn set_ssgi_buffer(&mut self, buffer: SsgiBuffer) {
        self.ssgi = Some(buffer);
    }

    /// Set the volumetric probe grid.
    pub fn set_volumetric_grid(&mut self, grid: VolumetricProbeGrid) {
        self.volumetric_grid = Some(grid);
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: IndirectLightingConfig) {
        self.config = config;
    }

    /// Get current configuration.
    pub fn config(&self) -> &IndirectLightingConfig {
        &self.config
    }

    /// Begin a new frame (reset stats).
    pub fn begin_frame(&mut self) {
        self.stats = IndirectLightingStats::default();
    }

    /// Evaluate indirect lighting for a single pixel.
    pub fn evaluate(&mut self, surface: &SurfaceData) -> IndirectLightingResult {
        self.stats.pixels_evaluated += 1;

        let mut result = IndirectLightingResult::default();
        let mut diffuse_accum = Vec3::ZERO;
        let mut specular_accum = Vec3::ZERO;
        let mut total_diffuse_weight = 0.0f32;
        let mut total_specular_weight = 0.0f32;

        // 1. SSGI (highest priority for diffuse and specular).
        if let Some(ssgi) = &self.ssgi {
            if ssgi.valid && self.config.ssgi_quality != SsgiQuality::Off {
                let confidence = ssgi.get_confidence(surface.screen_x, surface.screen_y);
                if confidence > 0.01 {
                    let ssgi_diffuse = ssgi.sample_diffuse(surface.screen_x, surface.screen_y);
                    let weight = confidence * self.config.weights.ssgi_weight;
                    diffuse_accum = diffuse_accum.add(ssgi_diffuse.scale(weight));
                    total_diffuse_weight += weight;

                    if self.config.specular_enabled {
                        let ssgi_spec = ssgi.sample_specular(surface.screen_x, surface.screen_y);
                        specular_accum = specular_accum.add(ssgi_spec.scale(weight));
                        total_specular_weight += weight;
                    }

                    result.source_mask.set(IndirectSourceMask::SSGI);
                    self.stats.pixels_ssgi += 1;
                }
            }
        }

        // 2. Volumetric probe grid.
        if let Some(vol) = &self.volumetric_grid {
            if let Some(sh) = vol.sample(surface.world_pos) {
                let irradiance = sh.evaluate(surface.normal);
                let weight = self.config.weights.volumetric_weight;
                diffuse_accum = diffuse_accum.add(irradiance.scale(weight));
                total_diffuse_weight += weight;

                result.source_mask.set(IndirectSourceMask::VOLUMETRIC);
                self.stats.pixels_volumetric += 1;
            }
        }

        // 3. Light probe grid.
        if let Some(grid) = &self.light_probe_grid {
            if grid.world_to_grid(surface.world_pos).is_some() {
                let sh = grid.sample(surface.world_pos);
                let irradiance = sh.evaluate(surface.normal);
                let weight = self.config.weights.light_probe_weight;
                diffuse_accum = diffuse_accum.add(irradiance.scale(weight));
                total_diffuse_weight += weight;

                result.source_mask.set(IndirectSourceMask::LIGHT_PROBE);
                self.stats.pixels_light_probe += 1;
            }
        }

        // 4. Reflection probes (specular only).
        if self.config.specular_enabled && surface.roughness < self.config.max_specular_roughness {
            let reflect_dir = self.reflect(surface.view_dir, surface.normal);

            // Find the best reflection probes.
            let mut best_probes: Vec<(usize, f32)> = Vec::new();
            for (i, probe) in self.reflection_probes.iter().enumerate() {
                let w = probe.weight_at(surface.world_pos);
                if w > 0.0 {
                    best_probes.push((i, w));
                }
            }
            // Sort by weight descending.
            best_probes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            best_probes.truncate(MAX_REFLECTION_PROBES_PER_PIXEL);

            let total_probe_weight: f32 = best_probes.iter().map(|(_, w)| *w).sum();
            if total_probe_weight > 0.0 {
                for &(idx, w) in &best_probes {
                    let probe = &self.reflection_probes[idx];
                    let corrected_dir = probe.correct_reflection(surface.world_pos, reflect_dir);
                    let _mip = probe.roughness_to_mip(surface.roughness);

                    // Approximate specular from average color and direction alignment.
                    let ndotl = corrected_dir.dot(surface.normal).max(0.0);
                    let spec = probe.average_color.scale(ndotl * probe.intensity);

                    let normalized_weight = w / total_probe_weight;
                    specular_accum = specular_accum.add(
                        spec.scale(normalized_weight * self.config.weights.reflection_probe_weight),
                    );
                }
                total_specular_weight += self.config.weights.reflection_probe_weight;
                result.source_mask.set(IndirectSourceMask::REFLECTION_PROBE);
                self.stats.pixels_reflection_probe += 1;
            }
        }

        // 5. Lightmap.
        if let Some((lm_u, lm_v)) = surface.lightmap_uv {
            // Use the first available lightmap for simplicity.
            if let Some(lm) = self.lightmaps.values().next() {
                let lm_color = lm.sample(lm_u, lm_v);
                let weight = self.config.weights.lightmap_weight;
                diffuse_accum = diffuse_accum.add(lm_color.scale(weight));
                total_diffuse_weight += weight;

                result.source_mask.set(IndirectSourceMask::LIGHTMAP);
                self.stats.pixels_lightmap += 1;
            }
        }

        // 6. Fallback ambient.
        if total_diffuse_weight < 0.01 {
            diffuse_accum = self.config.fallback_ambient.scale(self.config.weights.ambient_weight);
            total_diffuse_weight = self.config.weights.ambient_weight;
            result.source_mask.set(IndirectSourceMask::AMBIENT);
            self.stats.pixels_ambient_fallback += 1;
        }

        // Normalize accumulated lighting.
        if total_diffuse_weight > 0.0 {
            result.diffuse = diffuse_accum.scale(1.0 / total_diffuse_weight);
        }
        if total_specular_weight > 0.0 {
            result.specular = specular_accum.scale(1.0 / total_specular_weight);
        }

        // Apply AO.
        if self.config.apply_ao {
            let ao_factor = surface.ao.powf(self.config.ao_power);
            result.diffuse = result.diffuse.scale(ao_factor);
            // Specular AO is typically softer.
            let spec_ao = (surface.ao * 0.5 + 0.5).powf(self.config.ao_power);
            result.specular = result.specular.scale(spec_ao);
        }

        // Energy conservation: metals don't receive diffuse indirect.
        if self.config.energy_conservation {
            let diffuse_factor = 1.0 - surface.metallic;
            result.diffuse = result.diffuse.scale(diffuse_factor);
        }

        // Apply global intensity.
        result.diffuse = result.diffuse.scale(self.config.intensity);
        result.specular = result.specular.scale(self.config.intensity);

        // Compute confidence.
        let source_count = [
            result.source_mask.has(IndirectSourceMask::SSGI),
            result.source_mask.has(IndirectSourceMask::VOLUMETRIC),
            result.source_mask.has(IndirectSourceMask::LIGHT_PROBE),
            result.source_mask.has(IndirectSourceMask::LIGHTMAP),
        ]
        .iter()
        .filter(|&&x| x)
        .count();

        result.confidence = match source_count {
            0 => 0.0,
            1 => 0.5,
            2 => 0.75,
            _ => 1.0,
        };

        result
    }

    /// Reflect a direction around a normal.
    fn reflect(&self, incident: Vec3, normal: Vec3) -> Vec3 {
        let d = incident.scale(-1.0);
        let dot_dn = d.dot(normal);
        normal.scale(2.0 * dot_dn).sub(d)
    }

    /// Finalize frame and compute aggregate stats.
    pub fn end_frame(&mut self) {
        if self.stats.pixels_evaluated > 0 {
            self.stats.avg_confidence = (self.stats.pixels_ssgi as f32 * 1.0
                + self.stats.pixels_volumetric as f32 * 0.8
                + self.stats.pixels_light_probe as f32 * 0.6
                + self.stats.pixels_lightmap as f32 * 0.7
                + self.stats.pixels_ambient_fallback as f32 * 0.1)
                / self.stats.pixels_evaluated as f32;
        }
    }

    /// Get the stats from the last frame.
    pub fn stats(&self) -> &IndirectLightingStats {
        &self.stats
    }

    /// Get the number of reflection probes.
    pub fn reflection_probe_count(&self) -> usize {
        self.reflection_probes.len()
    }

    /// Get all reflection probes.
    pub fn reflection_probes(&self) -> &[ReflectionProbe] {
        &self.reflection_probes
    }

    /// Get the number of lightmaps.
    pub fn lightmap_count(&self) -> usize {
        self.lightmaps.len()
    }

    /// Check if the light probe grid is set.
    pub fn has_light_probe_grid(&self) -> bool {
        self.light_probe_grid.is_some()
    }

    /// Check if the volumetric grid is set.
    pub fn has_volumetric_grid(&self) -> bool {
        self.volumetric_grid.is_some()
    }

    /// Check if SSGI is available.
    pub fn has_ssgi(&self) -> bool {
        self.ssgi.as_ref().map_or(false, |s| s.valid)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sh_ambient() {
        let sh = SphericalHarmonicsL2::from_ambient(Vec3::new(0.5, 0.5, 0.5));
        let eval = sh.evaluate(Vec3::new(0.0, 1.0, 0.0));
        // Should be approximately the ambient color.
        assert!(eval.x > 0.0);
        assert!(eval.y > 0.0);
    }

    #[test]
    fn test_sh_lerp() {
        let a = SphericalHarmonicsL2::from_ambient(Vec3::new(1.0, 0.0, 0.0));
        let b = SphericalHarmonicsL2::from_ambient(Vec3::new(0.0, 1.0, 0.0));
        let mid = a.lerp(&b, 0.5);
        let eval = mid.evaluate(Vec3::new(0.0, 1.0, 0.0));
        // Should have both R and G contributions.
        assert!(eval.x > 0.0);
        assert!(eval.y > 0.0);
    }

    #[test]
    fn test_reflection_probe_weight() {
        let probe = ReflectionProbe::new(0, Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0));
        // Inside: should have weight.
        let w = probe.weight_at(Vec3::ZERO);
        // Probe not valid, so weight should be 0.
        assert_eq!(w, 0.0);

        let mut valid_probe = probe;
        valid_probe.valid = true;
        let w2 = valid_probe.weight_at(Vec3::ZERO);
        assert!(w2 > 0.0);

        // Outside: should have zero weight.
        let w3 = valid_probe.weight_at(Vec3::new(20.0, 0.0, 0.0));
        assert_eq!(w3, 0.0);
    }

    #[test]
    fn test_lightmap_sample() {
        let mut lm = Lightmap::new(0, 4, 4);
        lm.set_pixel(0, 0, Vec3::new(1.0, 0.0, 0.0));
        lm.set_pixel(1, 0, Vec3::new(0.0, 1.0, 0.0));
        let sample = lm.sample(0.0, 0.0);
        assert!((sample.x - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_indirect_lighting_fallback() {
        let config = IndirectLightingConfig::default();
        let mut system = IndirectLightingSystem::new(config);
        system.begin_frame();

        let surface = SurfaceData {
            world_pos: Vec3::ZERO,
            normal: Vec3::new(0.0, 1.0, 0.0),
            view_dir: Vec3::new(0.0, 0.0, 1.0),
            albedo: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            ao: 1.0,
            lightmap_uv: None,
            screen_x: 0,
            screen_y: 0,
        };

        let result = system.evaluate(&surface);
        // Should have fallen back to ambient.
        assert!(result.source_mask.has(IndirectSourceMask::AMBIENT));
        assert!(result.diffuse.x > 0.0);
    }

    #[test]
    fn test_source_mask() {
        let mut mask = IndirectSourceMask::empty();
        assert!(!mask.has(IndirectSourceMask::SSGI));
        mask.set(IndirectSourceMask::SSGI);
        assert!(mask.has(IndirectSourceMask::SSGI));
        mask.set(IndirectSourceMask::LIGHTMAP);
        assert!(mask.has(IndirectSourceMask::LIGHTMAP));
        assert!(!mask.has(IndirectSourceMask::VOLUMETRIC));
    }
}
