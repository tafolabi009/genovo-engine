// engine/render/src/ambient_system.rs
//
// Ambient lighting system for the Genovo renderer.
//
// Provides multiple ambient lighting modes: flat ambient, gradient (sky + ground),
// hemisphere (tri-directional), spherical harmonics (SH) ambient, ambient from
// cubemap, ambient probe grid, and ambient occlusion integration.
//
// The ambient system produces a per-pixel ambient contribution that feeds into
// the indirect lighting pipeline or can be used standalone.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Math types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0, z: 1.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };

    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }

    #[inline]
    pub fn dot(self, o: Self) -> f32 { self.x * o.x + self.y * o.y + self.z * o.z }

    #[inline]
    pub fn length(self) -> f32 { self.dot(self).sqrt() }

    #[inline]
    pub fn normalize(self) -> Self {
        let l = self.length();
        if l > 1e-7 { self.scale(1.0 / l) } else { Self::ZERO }
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
    pub fn lerp(self, o: Self, t: f32) -> Self {
        self.scale(1.0 - t).add(o.scale(t))
    }

    #[inline]
    pub fn max_component(self) -> f32 { self.x.max(self.y).max(self.z) }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of SH L2 coefficients.
pub const SH_L2_COEFFS: usize = 9;

/// Default ambient intensity.
pub const DEFAULT_AMBIENT_INTENSITY: f32 = 0.1;

/// Maximum number of ambient probes in a grid.
pub const MAX_AMBIENT_PROBES: usize = 4096;

/// Cubemap face count.
pub const CUBEMAP_FACES: usize = 6;

/// Default AO strength.
pub const DEFAULT_AO_STRENGTH: f32 = 1.0;

/// Default AO radius for SSAO.
pub const DEFAULT_AO_RADIUS: f32 = 0.5;

/// Maximum AO samples.
pub const MAX_AO_SAMPLES: u32 = 64;

// ---------------------------------------------------------------------------
// Ambient mode
// ---------------------------------------------------------------------------

/// Mode of ambient lighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmbientMode {
    /// Single flat color for all directions.
    Flat,
    /// Two-color gradient (sky color for up, ground color for down).
    Gradient,
    /// Three-axis hemisphere (positive X/Y/Z colors blended by normal).
    TriDirectional,
    /// Spherical harmonics L2 (9 coefficients for full directional ambient).
    SphericalHarmonics,
    /// Sampled from a cubemap texture.
    Cubemap,
    /// Grid of ambient probes with trilinear interpolation.
    ProbeGrid,
    /// No ambient (fully dark unless lit by direct or indirect sources).
    None,
}

impl fmt::Display for AmbientMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Flat => write!(f, "Flat"),
            Self::Gradient => write!(f, "Gradient"),
            Self::TriDirectional => write!(f, "Tri-Directional"),
            Self::SphericalHarmonics => write!(f, "Spherical Harmonics L2"),
            Self::Cubemap => write!(f, "Cubemap"),
            Self::ProbeGrid => write!(f, "Probe Grid"),
            Self::None => write!(f, "None"),
        }
    }
}

// ---------------------------------------------------------------------------
// Flat ambient
// ---------------------------------------------------------------------------

/// Flat ambient configuration.
#[derive(Debug, Clone, Copy)]
pub struct FlatAmbient {
    /// Ambient color (linear HDR).
    pub color: Vec3,
    /// Intensity multiplier.
    pub intensity: f32,
}

impl Default for FlatAmbient {
    fn default() -> Self {
        Self {
            color: Vec3::new(0.1, 0.1, 0.12),
            intensity: 1.0,
        }
    }
}

impl FlatAmbient {
    /// Evaluate the ambient for any normal direction.
    pub fn evaluate(&self, _normal: Vec3) -> Vec3 {
        self.color.scale(self.intensity)
    }
}

// ---------------------------------------------------------------------------
// Gradient ambient
// ---------------------------------------------------------------------------

/// Two-color gradient ambient (sky + ground).
#[derive(Debug, Clone, Copy)]
pub struct GradientAmbient {
    /// Sky color (for normals pointing up).
    pub sky_color: Vec3,
    /// Ground color (for normals pointing down).
    pub ground_color: Vec3,
    /// Equator color (optional, for normals pointing sideways).
    pub equator_color: Vec3,
    /// Intensity multiplier.
    pub intensity: f32,
    /// Whether to use equator color (tri-color gradient).
    pub use_equator: bool,
}

impl Default for GradientAmbient {
    fn default() -> Self {
        Self {
            sky_color: Vec3::new(0.2, 0.3, 0.5),
            ground_color: Vec3::new(0.1, 0.05, 0.02),
            equator_color: Vec3::new(0.15, 0.15, 0.15),
            intensity: 1.0,
            use_equator: false,
        }
    }
}

impl GradientAmbient {
    /// Evaluate the ambient for a given normal direction.
    pub fn evaluate(&self, normal: Vec3) -> Vec3 {
        let n = normal.normalize();
        let up_factor = n.y; // -1 to 1.

        let color = if self.use_equator {
            if up_factor >= 0.0 {
                // Sky to equator.
                self.equator_color.lerp(self.sky_color, up_factor)
            } else {
                // Equator to ground.
                self.ground_color.lerp(self.equator_color, up_factor + 1.0)
            }
        } else {
            // Simple sky-ground blend.
            let t = up_factor * 0.5 + 0.5; // Remap to 0..1.
            self.ground_color.lerp(self.sky_color, t)
        };

        color.scale(self.intensity)
    }
}

// ---------------------------------------------------------------------------
// Tri-directional (hemisphere) ambient
// ---------------------------------------------------------------------------

/// Three-axis hemisphere ambient.
#[derive(Debug, Clone, Copy)]
pub struct TriDirectionalAmbient {
    /// Color for +X direction.
    pub positive_x: Vec3,
    /// Color for -X direction.
    pub negative_x: Vec3,
    /// Color for +Y direction (up).
    pub positive_y: Vec3,
    /// Color for -Y direction (down).
    pub negative_y: Vec3,
    /// Color for +Z direction.
    pub positive_z: Vec3,
    /// Color for -Z direction.
    pub negative_z: Vec3,
    /// Intensity multiplier.
    pub intensity: f32,
}

impl Default for TriDirectionalAmbient {
    fn default() -> Self {
        Self {
            positive_x: Vec3::new(0.15, 0.1, 0.1),
            negative_x: Vec3::new(0.1, 0.1, 0.15),
            positive_y: Vec3::new(0.2, 0.3, 0.5),
            negative_y: Vec3::new(0.1, 0.05, 0.02),
            positive_z: Vec3::new(0.12, 0.12, 0.12),
            negative_z: Vec3::new(0.08, 0.08, 0.08),
            intensity: 1.0,
        }
    }
}

impl TriDirectionalAmbient {
    /// Evaluate the ambient for a given normal direction.
    pub fn evaluate(&self, normal: Vec3) -> Vec3 {
        let n = normal.normalize();

        // Blend based on absolute normal components.
        let x_color = if n.x >= 0.0 {
            self.positive_x.scale(n.x)
        } else {
            self.negative_x.scale(-n.x)
        };

        let y_color = if n.y >= 0.0 {
            self.positive_y.scale(n.y)
        } else {
            self.negative_y.scale(-n.y)
        };

        let z_color = if n.z >= 0.0 {
            self.positive_z.scale(n.z)
        } else {
            self.negative_z.scale(-n.z)
        };

        x_color.add(y_color).add(z_color).scale(self.intensity)
    }
}

// ---------------------------------------------------------------------------
// Spherical harmonics ambient
// ---------------------------------------------------------------------------

/// SH L2 ambient (9 RGB coefficients).
#[derive(Debug, Clone)]
pub struct ShAmbient {
    /// SH coefficients (9 Vec3 values).
    pub coefficients: [Vec3; SH_L2_COEFFS],
    /// Intensity multiplier.
    pub intensity: f32,
}

impl Default for ShAmbient {
    fn default() -> Self {
        let mut sh = Self {
            coefficients: [Vec3::ZERO; SH_L2_COEFFS],
            intensity: 1.0,
        };
        // Default: uniform low ambient.
        let sqrt_4pi = (4.0 * std::f32::consts::PI).sqrt();
        sh.coefficients[0] = Vec3::new(0.1, 0.1, 0.12).scale(sqrt_4pi);
        sh
    }
}

impl ShAmbient {
    /// Create SH ambient from a uniform color.
    pub fn from_uniform(color: Vec3) -> Self {
        let mut sh = Self {
            coefficients: [Vec3::ZERO; SH_L2_COEFFS],
            intensity: 1.0,
        };
        let sqrt_4pi = (4.0 * std::f32::consts::PI).sqrt();
        sh.coefficients[0] = color.scale(sqrt_4pi);
        sh
    }

    /// Create SH ambient from sky/ground colors (gradient).
    pub fn from_gradient(sky: Vec3, ground: Vec3) -> Self {
        let mut sh = Self {
            coefficients: [Vec3::ZERO; SH_L2_COEFFS],
            intensity: 1.0,
        };

        let avg = sky.add(ground).scale(0.5);
        let diff = sky.sub(ground).scale(0.5);

        let sqrt_4pi = (4.0 * std::f32::consts::PI).sqrt();
        let sqrt_4pi_3 = (4.0 * std::f32::consts::PI / 3.0).sqrt();

        sh.coefficients[0] = avg.scale(sqrt_4pi);
        sh.coefficients[2] = diff.scale(sqrt_4pi_3); // Y1_0 (z component = up in SH convention)

        sh
    }

    /// Evaluate the SH for a given direction.
    pub fn evaluate(&self, direction: Vec3) -> Vec3 {
        let d = direction.normalize();

        let basis = [
            0.282095,                          // Y0_0
            0.488603 * d.y,                    // Y1_{-1}
            0.488603 * d.z,                    // Y1_0
            0.488603 * d.x,                    // Y1_1
            1.092548 * d.x * d.y,              // Y2_{-2}
            1.092548 * d.y * d.z,              // Y2_{-1}
            0.315392 * (3.0 * d.z * d.z - 1.0), // Y2_0
            1.092548 * d.x * d.z,              // Y2_1
            0.546274 * (d.x * d.x - d.y * d.y), // Y2_2
        ];

        let mut result = Vec3::ZERO;
        for i in 0..SH_L2_COEFFS {
            result = result.add(self.coefficients[i].scale(basis[i]));
        }

        Vec3::new(
            result.x.max(0.0) * self.intensity,
            result.y.max(0.0) * self.intensity,
            result.z.max(0.0) * self.intensity,
        )
    }

    /// Accumulate another SH into this one.
    pub fn accumulate(&mut self, other: &ShAmbient, weight: f32) {
        for i in 0..SH_L2_COEFFS {
            self.coefficients[i] = self.coefficients[i].add(other.coefficients[i].scale(weight));
        }
    }

    /// Lerp between two SH ambients.
    pub fn lerp(&self, other: &ShAmbient, t: f32) -> ShAmbient {
        let mut result = ShAmbient {
            coefficients: [Vec3::ZERO; SH_L2_COEFFS],
            intensity: self.intensity + (other.intensity - self.intensity) * t,
        };
        for i in 0..SH_L2_COEFFS {
            result.coefficients[i] = self.coefficients[i].lerp(other.coefficients[i], t);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Cubemap ambient
// ---------------------------------------------------------------------------

/// Ambient from a cubemap (6 faces).
#[derive(Debug, Clone)]
pub struct CubemapAmbient {
    /// Cubemap face data (6 faces, each face is a grid of Vec3 colors).
    pub faces: [Vec<Vec3>; CUBEMAP_FACES],
    /// Resolution per face.
    pub resolution: u32,
    /// Intensity multiplier.
    pub intensity: f32,
    /// Number of mip levels for diffuse convolution.
    pub mip_count: u32,
    /// Precomputed SH from the cubemap (for fast evaluation).
    pub precomputed_sh: Option<ShAmbient>,
}

impl CubemapAmbient {
    /// Create a new empty cubemap ambient.
    pub fn new(resolution: u32) -> Self {
        let face_size = (resolution * resolution) as usize;
        Self {
            faces: [
                vec![Vec3::ZERO; face_size],
                vec![Vec3::ZERO; face_size],
                vec![Vec3::ZERO; face_size],
                vec![Vec3::ZERO; face_size],
                vec![Vec3::ZERO; face_size],
                vec![Vec3::ZERO; face_size],
            ],
            resolution,
            intensity: 1.0,
            mip_count: 1,
            precomputed_sh: None,
        }
    }

    /// Set a face pixel.
    pub fn set_pixel(&mut self, face: usize, x: u32, y: u32, color: Vec3) {
        if face < CUBEMAP_FACES && x < self.resolution && y < self.resolution {
            self.faces[face][(y * self.resolution + x) as usize] = color;
        }
    }

    /// Sample a face at UV coordinates.
    pub fn sample_face(&self, face: usize, u: f32, v: f32) -> Vec3 {
        if face >= CUBEMAP_FACES {
            return Vec3::ZERO;
        }

        let fx = (u * (self.resolution as f32 - 1.0)).clamp(0.0, (self.resolution - 1) as f32);
        let fy = (v * (self.resolution as f32 - 1.0)).clamp(0.0, (self.resolution - 1) as f32);

        let x0 = fx.floor() as u32;
        let y0 = fy.floor() as u32;
        let x1 = (x0 + 1).min(self.resolution - 1);
        let y1 = (y0 + 1).min(self.resolution - 1);

        let s = fx - x0 as f32;
        let t = fy - y0 as f32;

        let c00 = self.faces[face][(y0 * self.resolution + x0) as usize];
        let c10 = self.faces[face][(y0 * self.resolution + x1) as usize];
        let c01 = self.faces[face][(y1 * self.resolution + x0) as usize];
        let c11 = self.faces[face][(y1 * self.resolution + x1) as usize];

        let top = c00.lerp(c10, s);
        let bot = c01.lerp(c11, s);
        top.lerp(bot, t).scale(self.intensity)
    }

    /// Map a direction to cubemap face and UV coordinates.
    pub fn direction_to_face_uv(dir: Vec3) -> (usize, f32, f32) {
        let abs_x = dir.x.abs();
        let abs_y = dir.y.abs();
        let abs_z = dir.z.abs();

        let (face, u, v) = if abs_x >= abs_y && abs_x >= abs_z {
            if dir.x > 0.0 {
                (0, -dir.z / abs_x, -dir.y / abs_x) // +X
            } else {
                (1, dir.z / abs_x, -dir.y / abs_x) // -X
            }
        } else if abs_y >= abs_x && abs_y >= abs_z {
            if dir.y > 0.0 {
                (2, dir.x / abs_y, dir.z / abs_y) // +Y
            } else {
                (3, dir.x / abs_y, -dir.z / abs_y) // -Y
            }
        } else {
            if dir.z > 0.0 {
                (4, dir.x / abs_z, -dir.y / abs_z) // +Z
            } else {
                (5, -dir.x / abs_z, -dir.y / abs_z) // -Z
            }
        };

        // Remap from [-1, 1] to [0, 1].
        ((face), (u * 0.5 + 0.5), (v * 0.5 + 0.5))
    }

    /// Evaluate ambient for a direction by sampling the cubemap.
    pub fn evaluate(&self, direction: Vec3) -> Vec3 {
        // If we have precomputed SH, use it for speed.
        if let Some(sh) = &self.precomputed_sh {
            return sh.evaluate(direction);
        }

        let (face, u, v) = Self::direction_to_face_uv(direction);
        self.sample_face(face, u, v)
    }

    /// Precompute SH from the cubemap for fast evaluation.
    pub fn precompute_sh(&mut self) {
        let mut sh = ShAmbient {
            coefficients: [Vec3::ZERO; SH_L2_COEFFS],
            intensity: self.intensity,
        };

        let sample_count = 256u32;
        let inv_samples = 1.0 / (sample_count * sample_count) as f32;

        for sy in 0..sample_count {
            for sx in 0..sample_count {
                let u = (sx as f32 + 0.5) / sample_count as f32;
                let v = (sy as f32 + 0.5) / sample_count as f32;

                // Map to sphere direction (equirectangular).
                let theta = v * std::f32::consts::PI;
                let phi = u * std::f32::consts::TAU;
                let dir = Vec3::new(
                    theta.sin() * phi.cos(),
                    theta.cos(),
                    theta.sin() * phi.sin(),
                );

                let color = self.evaluate(dir);
                let solid_angle = theta.sin() * std::f32::consts::PI * std::f32::consts::TAU * inv_samples;

                let basis = [
                    0.282095,
                    0.488603 * dir.y,
                    0.488603 * dir.z,
                    0.488603 * dir.x,
                    1.092548 * dir.x * dir.y,
                    1.092548 * dir.y * dir.z,
                    0.315392 * (3.0 * dir.z * dir.z - 1.0),
                    1.092548 * dir.x * dir.z,
                    0.546274 * (dir.x * dir.x - dir.y * dir.y),
                ];

                for i in 0..SH_L2_COEFFS {
                    sh.coefficients[i] = sh.coefficients[i].add(color.scale(basis[i] * solid_angle));
                }
            }
        }

        self.precomputed_sh = Some(sh);
    }
}

// ---------------------------------------------------------------------------
// Ambient probe grid
// ---------------------------------------------------------------------------

/// Unique identifier for an ambient probe.
pub type AmbientProbeId = u32;

/// A single ambient probe storing SH data.
#[derive(Debug, Clone)]
pub struct AmbientProbe {
    pub id: AmbientProbeId,
    pub position: Vec3,
    pub sh: ShAmbient,
    pub radius: f32,
    pub valid: bool,
}

impl AmbientProbe {
    /// Create a new ambient probe.
    pub fn new(id: AmbientProbeId, position: Vec3, radius: f32) -> Self {
        Self {
            id,
            position,
            sh: ShAmbient::default(),
            radius,
            valid: false,
        }
    }

    /// Compute weight at a world position.
    pub fn weight_at(&self, world_pos: Vec3) -> f32 {
        if !self.valid {
            return 0.0;
        }
        let dist = self.position.sub(world_pos).length();
        if dist >= self.radius {
            return 0.0;
        }
        let t = dist / self.radius;
        1.0 - t * t * (3.0 - 2.0 * t)
    }

    /// Evaluate the probe for a direction.
    pub fn evaluate(&self, direction: Vec3) -> Vec3 {
        self.sh.evaluate(direction)
    }
}

/// Grid of ambient probes.
#[derive(Debug, Clone)]
pub struct AmbientProbeGrid {
    probes: Vec<AmbientProbe>,
    origin: Vec3,
    cell_size: Vec3,
    dimensions: [u32; 3],
    next_id: AmbientProbeId,
}

impl AmbientProbeGrid {
    /// Create a new probe grid.
    pub fn new(origin: Vec3, cell_size: Vec3, dimensions: [u32; 3]) -> Self {
        let total = (dimensions[0] * dimensions[1] * dimensions[2]) as usize;
        let mut probes = Vec::with_capacity(total);
        let mut next_id = 0u32;

        for z in 0..dimensions[2] {
            for y in 0..dimensions[1] {
                for x in 0..dimensions[0] {
                    let pos = Vec3::new(
                        origin.x + x as f32 * cell_size.x + cell_size.x * 0.5,
                        origin.y + y as f32 * cell_size.y + cell_size.y * 0.5,
                        origin.z + z as f32 * cell_size.z + cell_size.z * 0.5,
                    );
                    let radius = cell_size.x.max(cell_size.y).max(cell_size.z) * 1.5;
                    probes.push(AmbientProbe::new(next_id, pos, radius));
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

    /// Sample the grid at a world position.
    pub fn sample(&self, world_pos: Vec3, direction: Vec3) -> Vec3 {
        let local = world_pos.sub(self.origin);
        let fx = local.x / self.cell_size.x;
        let fy = local.y / self.cell_size.y;
        let fz = local.z / self.cell_size.z;

        // Find the 8 surrounding probes and trilinear interpolate.
        let x0 = (fx.floor() as u32).min(self.dimensions[0].saturating_sub(2));
        let y0 = (fy.floor() as u32).min(self.dimensions[1].saturating_sub(2));
        let z0 = (fz.floor() as u32).min(self.dimensions[2].saturating_sub(2));

        let tx = (fx - x0 as f32).clamp(0.0, 1.0);
        let ty = (fy - y0 as f32).clamp(0.0, 1.0);
        let tz = (fz - z0 as f32).clamp(0.0, 1.0);

        let get = |x: u32, y: u32, z: u32| -> Vec3 {
            let idx = (z * self.dimensions[1] * self.dimensions[0]
                + y * self.dimensions[0]
                + x) as usize;
            if idx < self.probes.len() {
                self.probes[idx].evaluate(direction)
            } else {
                Vec3::ZERO
            }
        };

        let c000 = get(x0, y0, z0);
        let c100 = get(x0 + 1, y0, z0);
        let c010 = get(x0, y0 + 1, z0);
        let c110 = get(x0 + 1, y0 + 1, z0);
        let c001 = get(x0, y0, z0 + 1);
        let c101 = get(x0 + 1, y0, z0 + 1);
        let c011 = get(x0, y0 + 1, z0 + 1);
        let c111 = get(x0 + 1, y0 + 1, z0 + 1);

        let c00 = c000.lerp(c100, tx);
        let c10 = c010.lerp(c110, tx);
        let c01 = c001.lerp(c101, tx);
        let c11 = c011.lerp(c111, tx);

        let c0 = c00.lerp(c10, ty);
        let c1 = c01.lerp(c11, ty);

        c0.lerp(c1, tz)
    }

    /// Get all probes.
    pub fn probes(&self) -> &[AmbientProbe] {
        &self.probes
    }

    /// Get mutable probes.
    pub fn probes_mut(&mut self) -> &mut [AmbientProbe] {
        &mut self.probes
    }

    /// Get grid dimensions.
    pub fn dimensions(&self) -> [u32; 3] {
        self.dimensions
    }

    /// Total probe count.
    pub fn probe_count(&self) -> usize {
        self.probes.len()
    }

    /// Mark all probes as valid.
    pub fn validate_all(&mut self) {
        for p in &mut self.probes {
            p.valid = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Ambient occlusion integration
// ---------------------------------------------------------------------------

/// AO technique.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AoTechnique {
    /// No AO.
    None,
    /// Screen-space ambient occlusion (SSAO).
    Ssao,
    /// Ground truth ambient occlusion (GTAO).
    Gtao,
    /// Horizon-based ambient occlusion (HBAO).
    Hbao,
    /// Ray-traced ambient occlusion (RTAO).
    Rtao,
    /// Baked AO from lightmaps or vertex data.
    Baked,
}

/// AO configuration.
#[derive(Debug, Clone, Copy)]
pub struct AoConfig {
    pub technique: AoTechnique,
    /// AO strength (0 = no effect, 1 = full).
    pub strength: f32,
    /// AO radius in world units.
    pub radius: f32,
    /// Bias to prevent self-occlusion.
    pub bias: f32,
    /// Number of samples for SSAO/HBAO.
    pub sample_count: u32,
    /// Power exponent applied to the AO term.
    pub power: f32,
    /// Whether to apply AO to specular (usually softer).
    pub apply_to_specular: bool,
    /// Specular AO power (softer than diffuse).
    pub specular_power: f32,
    /// Whether to apply temporal filtering to AO.
    pub temporal_filter: bool,
    /// Blur radius for spatial denoising.
    pub blur_radius: u32,
}

impl Default for AoConfig {
    fn default() -> Self {
        Self {
            technique: AoTechnique::Ssao,
            strength: DEFAULT_AO_STRENGTH,
            radius: DEFAULT_AO_RADIUS,
            bias: 0.025,
            sample_count: 16,
            power: DEFAULT_AO_POWER,
            apply_to_specular: true,
            specular_power: 0.5,
            temporal_filter: true,
            blur_radius: 2,
        }
    }
}

/// AO buffer.
#[derive(Debug, Clone)]
pub struct AoBuffer {
    pub width: u32,
    pub height: u32,
    /// Per-pixel AO value (0 = fully occluded, 1 = no occlusion).
    pub data: Vec<f32>,
    pub config: AoConfig,
}

impl AoBuffer {
    /// Create a new AO buffer.
    pub fn new(width: u32, height: u32, config: AoConfig) -> Self {
        Self {
            width,
            height,
            data: vec![1.0; (width * height) as usize],
            config,
        }
    }

    /// Sample AO at a pixel.
    pub fn sample(&self, x: u32, y: u32) -> f32 {
        if x < self.width && y < self.height {
            let raw = self.data[(y * self.width + x) as usize];
            raw.powf(self.config.power) * self.config.strength
                + (1.0 - self.config.strength)
        } else {
            1.0
        }
    }

    /// Sample specular AO at a pixel (softer).
    pub fn sample_specular(&self, x: u32, y: u32) -> f32 {
        if !self.config.apply_to_specular {
            return 1.0;
        }
        if x < self.width && y < self.height {
            let raw = self.data[(y * self.width + x) as usize];
            (raw * 0.5 + 0.5).powf(self.config.specular_power)
        } else {
            1.0
        }
    }

    /// Set AO at a pixel.
    pub fn set(&mut self, x: u32, y: u32, value: f32) {
        if x < self.width && y < self.height {
            self.data[(y * self.width + x) as usize] = value.clamp(0.0, 1.0);
        }
    }

    /// Clear to no occlusion.
    pub fn clear(&mut self) {
        for v in &mut self.data {
            *v = 1.0;
        }
    }

    /// Resize the buffer.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.data = vec![1.0; (width * height) as usize];
    }
}

// ---------------------------------------------------------------------------
// Ambient lighting system
// ---------------------------------------------------------------------------

/// Complete ambient lighting configuration.
#[derive(Debug, Clone)]
pub struct AmbientConfig {
    /// Active ambient mode.
    pub mode: AmbientMode,
    /// Flat ambient settings.
    pub flat: FlatAmbient,
    /// Gradient ambient settings.
    pub gradient: GradientAmbient,
    /// Tri-directional ambient settings.
    pub tri_directional: TriDirectionalAmbient,
    /// SH ambient data.
    pub sh: ShAmbient,
    /// AO configuration.
    pub ao: AoConfig,
    /// Global intensity multiplier.
    pub intensity: f32,
    /// Minimum ambient (never goes below this, even with AO).
    pub minimum_ambient: Vec3,
    /// Whether ambient affects the sky/background.
    pub affect_background: bool,
}

impl Default for AmbientConfig {
    fn default() -> Self {
        Self {
            mode: AmbientMode::Gradient,
            flat: FlatAmbient::default(),
            gradient: GradientAmbient::default(),
            tri_directional: TriDirectionalAmbient::default(),
            sh: ShAmbient::default(),
            ao: AoConfig::default(),
            intensity: 1.0,
            minimum_ambient: Vec3::new(0.01, 0.01, 0.01),
            affect_background: false,
        }
    }
}

/// Statistics for the ambient system.
#[derive(Debug, Clone, Copy, Default)]
pub struct AmbientStats {
    pub pixels_evaluated: u32,
    pub avg_ao: f32,
    pub min_ao: f32,
    pub max_ao: f32,
    pub avg_ambient_luminance: f32,
}

/// Main ambient lighting system.
pub struct AmbientSystem {
    config: AmbientConfig,
    cubemap: Option<CubemapAmbient>,
    probe_grid: Option<AmbientProbeGrid>,
    ao_buffer: Option<AoBuffer>,
    stats: AmbientStats,
}

impl AmbientSystem {
    /// Create a new ambient system.
    pub fn new(config: AmbientConfig) -> Self {
        Self {
            config,
            cubemap: None,
            probe_grid: None,
            ao_buffer: None,
            stats: AmbientStats::default(),
        }
    }

    /// Set the ambient configuration.
    pub fn set_config(&mut self, config: AmbientConfig) {
        self.config = config;
    }

    /// Get the current configuration.
    pub fn config(&self) -> &AmbientConfig {
        &self.config
    }

    /// Set the cubemap ambient source.
    pub fn set_cubemap(&mut self, cubemap: CubemapAmbient) {
        self.cubemap = Some(cubemap);
    }

    /// Set the probe grid.
    pub fn set_probe_grid(&mut self, grid: AmbientProbeGrid) {
        self.probe_grid = Some(grid);
    }

    /// Set the AO buffer for this frame.
    pub fn set_ao_buffer(&mut self, buffer: AoBuffer) {
        self.ao_buffer = Some(buffer);
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self) {
        self.stats = AmbientStats {
            min_ao: f32::MAX,
            max_ao: f32::MIN,
            ..Default::default()
        };
    }

    /// Evaluate ambient lighting for a single pixel.
    pub fn evaluate(
        &mut self,
        normal: Vec3,
        world_pos: Vec3,
        screen_x: u32,
        screen_y: u32,
    ) -> Vec3 {
        self.stats.pixels_evaluated += 1;

        // Compute raw ambient based on mode.
        let raw_ambient = match self.config.mode {
            AmbientMode::Flat => self.config.flat.evaluate(normal),
            AmbientMode::Gradient => self.config.gradient.evaluate(normal),
            AmbientMode::TriDirectional => self.config.tri_directional.evaluate(normal),
            AmbientMode::SphericalHarmonics => self.config.sh.evaluate(normal),
            AmbientMode::Cubemap => {
                if let Some(cm) = &self.cubemap {
                    cm.evaluate(normal)
                } else {
                    self.config.flat.evaluate(normal)
                }
            }
            AmbientMode::ProbeGrid => {
                if let Some(grid) = &self.probe_grid {
                    grid.sample(world_pos, normal)
                } else {
                    self.config.flat.evaluate(normal)
                }
            }
            AmbientMode::None => Vec3::ZERO,
        };

        // Apply intensity.
        let mut ambient = raw_ambient.scale(self.config.intensity);

        // Apply AO.
        if let Some(ao_buf) = &self.ao_buffer {
            let ao = ao_buf.sample(screen_x, screen_y);
            ambient = ambient.scale(ao);

            // Track AO stats.
            let raw_ao = if screen_x < ao_buf.width && screen_y < ao_buf.height {
                ao_buf.data[(screen_y * ao_buf.width + screen_x) as usize]
            } else {
                1.0
            };
            self.stats.min_ao = self.stats.min_ao.min(raw_ao);
            self.stats.max_ao = self.stats.max_ao.max(raw_ao);
            self.stats.avg_ao += raw_ao;
        }

        // Apply minimum ambient.
        ambient = Vec3::new(
            ambient.x.max(self.config.minimum_ambient.x),
            ambient.y.max(self.config.minimum_ambient.y),
            ambient.z.max(self.config.minimum_ambient.z),
        );

        // Track luminance.
        let lum = 0.2126 * ambient.x + 0.7152 * ambient.y + 0.0722 * ambient.z;
        self.stats.avg_ambient_luminance += lum;

        ambient
    }

    /// End frame and finalize statistics.
    pub fn end_frame(&mut self) {
        if self.stats.pixels_evaluated > 0 {
            let n = self.stats.pixels_evaluated as f32;
            self.stats.avg_ao /= n;
            self.stats.avg_ambient_luminance /= n;
        }
        if self.stats.min_ao == f32::MAX {
            self.stats.min_ao = 1.0;
        }
        if self.stats.max_ao == f32::MIN {
            self.stats.max_ao = 1.0;
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &AmbientStats {
        &self.stats
    }

    /// Whether a cubemap is loaded.
    pub fn has_cubemap(&self) -> bool {
        self.cubemap.is_some()
    }

    /// Whether a probe grid is loaded.
    pub fn has_probe_grid(&self) -> bool {
        self.probe_grid.is_some()
    }

    /// Whether an AO buffer is set.
    pub fn has_ao_buffer(&self) -> bool {
        self.ao_buffer.is_some()
    }

    /// Get the active ambient mode.
    pub fn mode(&self) -> AmbientMode {
        self.config.mode
    }
}

// ---------------------------------------------------------------------------
// Ambient presets
// ---------------------------------------------------------------------------

/// Predefined ambient lighting presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmbientPreset {
    /// Outdoor daylight.
    OutdoorDay,
    /// Outdoor sunset/golden hour.
    OutdoorSunset,
    /// Outdoor night.
    OutdoorNight,
    /// Indoor office lighting.
    IndoorOffice,
    /// Indoor warm (fireplace, candle).
    IndoorWarm,
    /// Underground/dungeon.
    Underground,
    /// Underwater.
    Underwater,
    /// Space.
    Space,
}

impl AmbientPreset {
    /// Convert preset to an ambient configuration.
    pub fn to_config(self) -> AmbientConfig {
        match self {
            Self::OutdoorDay => AmbientConfig {
                mode: AmbientMode::Gradient,
                gradient: GradientAmbient {
                    sky_color: Vec3::new(0.4, 0.6, 1.0),
                    ground_color: Vec3::new(0.15, 0.1, 0.05),
                    equator_color: Vec3::new(0.3, 0.3, 0.35),
                    use_equator: true,
                    intensity: 0.3,
                },
                intensity: 1.0,
                ..Default::default()
            },
            Self::OutdoorSunset => AmbientConfig {
                mode: AmbientMode::Gradient,
                gradient: GradientAmbient {
                    sky_color: Vec3::new(0.6, 0.3, 0.1),
                    ground_color: Vec3::new(0.1, 0.05, 0.03),
                    equator_color: Vec3::new(0.5, 0.2, 0.05),
                    use_equator: true,
                    intensity: 0.25,
                },
                intensity: 1.0,
                ..Default::default()
            },
            Self::OutdoorNight => AmbientConfig {
                mode: AmbientMode::Gradient,
                gradient: GradientAmbient {
                    sky_color: Vec3::new(0.02, 0.02, 0.05),
                    ground_color: Vec3::new(0.005, 0.005, 0.01),
                    intensity: 0.5,
                    ..Default::default()
                },
                intensity: 1.0,
                ..Default::default()
            },
            Self::IndoorOffice => AmbientConfig {
                mode: AmbientMode::Flat,
                flat: FlatAmbient {
                    color: Vec3::new(0.2, 0.2, 0.22),
                    intensity: 0.5,
                },
                intensity: 1.0,
                ..Default::default()
            },
            Self::IndoorWarm => AmbientConfig {
                mode: AmbientMode::Gradient,
                gradient: GradientAmbient {
                    sky_color: Vec3::new(0.15, 0.08, 0.03),
                    ground_color: Vec3::new(0.05, 0.02, 0.01),
                    intensity: 0.4,
                    ..Default::default()
                },
                intensity: 1.0,
                ..Default::default()
            },
            Self::Underground => AmbientConfig {
                mode: AmbientMode::Flat,
                flat: FlatAmbient {
                    color: Vec3::new(0.02, 0.02, 0.03),
                    intensity: 1.0,
                },
                intensity: 1.0,
                ..Default::default()
            },
            Self::Underwater => AmbientConfig {
                mode: AmbientMode::Gradient,
                gradient: GradientAmbient {
                    sky_color: Vec3::new(0.05, 0.15, 0.25),
                    ground_color: Vec3::new(0.01, 0.03, 0.05),
                    intensity: 0.6,
                    ..Default::default()
                },
                intensity: 1.0,
                ..Default::default()
            },
            Self::Space => AmbientConfig {
                mode: AmbientMode::Flat,
                flat: FlatAmbient {
                    color: Vec3::new(0.001, 0.001, 0.002),
                    intensity: 1.0,
                },
                intensity: 1.0,
                minimum_ambient: Vec3::new(0.001, 0.001, 0.001),
                ..Default::default()
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_ambient() {
        let flat = FlatAmbient {
            color: Vec3::new(0.5, 0.5, 0.5),
            intensity: 1.0,
        };
        let result = flat.evaluate(Vec3::UP);
        assert!((result.x - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_ambient() {
        let grad = GradientAmbient {
            sky_color: Vec3::new(0.0, 0.0, 1.0),
            ground_color: Vec3::new(1.0, 0.0, 0.0),
            intensity: 1.0,
            ..Default::default()
        };
        let up = grad.evaluate(Vec3::UP);
        let down = grad.evaluate(Vec3::new(0.0, -1.0, 0.0));

        // Up should be blueish.
        assert!(up.z > up.x);
        // Down should be reddish.
        assert!(down.x > down.z);
    }

    #[test]
    fn test_tri_directional() {
        let tri = TriDirectionalAmbient::default();
        let up = tri.evaluate(Vec3::UP);
        let down = tri.evaluate(Vec3::new(0.0, -1.0, 0.0));
        // Up should be brighter (sky).
        assert!(up.y > down.y);
    }

    #[test]
    fn test_sh_ambient_uniform() {
        let sh = ShAmbient::from_uniform(Vec3::new(0.5, 0.5, 0.5));
        let val = sh.evaluate(Vec3::UP);
        // Should be approximately 0.5 in all channels.
        assert!(val.x > 0.0);
    }

    #[test]
    fn test_ao_buffer() {
        let config = AoConfig::default();
        let mut buf = AoBuffer::new(4, 4, config);
        buf.set(1, 1, 0.5);
        let ao = buf.sample(1, 1);
        assert!(ao < 1.0);
        assert!(ao > 0.0);
    }

    #[test]
    fn test_ambient_system_evaluate() {
        let config = AmbientConfig::default();
        let mut system = AmbientSystem::new(config);
        system.begin_frame();
        let ambient = system.evaluate(Vec3::UP, Vec3::ZERO, 0, 0);
        assert!(ambient.x > 0.0);
        system.end_frame();
    }

    #[test]
    fn test_preset_outdoor_day() {
        let config = AmbientPreset::OutdoorDay.to_config();
        assert_eq!(config.mode, AmbientMode::Gradient);
        assert!(config.gradient.sky_color.z > config.gradient.ground_color.z);
    }
}
