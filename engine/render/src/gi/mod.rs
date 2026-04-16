// engine/render/src/gi/mod.rs
//
// Global Illumination (GI) system for the Genovo renderer. Provides light
// probes with spherical harmonics, reflection probes with cubemap textures,
// irradiance and prefilter map generation, BRDF integration LUT, ambient
// cubes, and a 3D probe grid with trilinear interpolation for smooth
// indirect lighting.

use crate::lighting::light_types::Light;
use glam::{Vec3, Vec4};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Spherical Harmonics (L2, order 2)
// ---------------------------------------------------------------------------

/// Number of SH coefficients for order 2 (L2): (L+1)^2 = 9.
pub const SH_COEFF_COUNT: usize = 9;

/// Total floats for RGB SH: 9 coefficients * 3 channels = 27.
pub const SH_FLOAT_COUNT: usize = SH_COEFF_COUNT * 3;

/// L2 Spherical Harmonics coefficients for RGB irradiance.
///
/// Stores 9 coefficients per colour channel (27 floats total).
/// The basis functions are:
///   Y_0^0  = 0.282095                                (constant / DC)
///   Y_1^{-1} = 0.488603 * y                          (linear band 1)
///   Y_1^0  = 0.488603 * z
///   Y_1^1  = 0.488603 * x
///   Y_2^{-2} = 1.092548 * x*y                        (quadratic band 2)
///   Y_2^{-1} = 1.092548 * y*z
///   Y_2^0  = 0.315392 * (3*z*z - 1)
///   Y_2^1  = 1.092548 * x*z
///   Y_2^2  = 0.546274 * (x*x - y*y)
#[derive(Debug, Clone)]
pub struct SphericalHarmonics {
    /// Coefficients per channel: `coeffs[channel][basis_index]`.
    /// channel: 0=R, 1=G, 2=B.
    pub coeffs: [[f32; SH_COEFF_COUNT]; 3],
}

impl Default for SphericalHarmonics {
    fn default() -> Self {
        Self {
            coeffs: [[0.0; SH_COEFF_COUNT]; 3],
        }
    }
}

// SH basis function constants.
const SH_Y00: f32 = 0.282_094_8;  // 1 / (2 * sqrt(pi))
const SH_Y1M1: f32 = 0.488_602_5; // sqrt(3) / (2 * sqrt(pi))
const SH_Y10: f32 = 0.488_602_5;
const SH_Y11: f32 = 0.488_602_5;
const SH_Y2M2: f32 = 1.092_548_4; // sqrt(15) / (2 * sqrt(pi))
const SH_Y2M1: f32 = 1.092_548_4;
const SH_Y20: f32 = 0.315_391_6;  // sqrt(5) / (4 * sqrt(pi))
const SH_Y21: f32 = 1.092_548_4;
const SH_Y22: f32 = 0.546_274_2;  // sqrt(15) / (4 * sqrt(pi))

// Cosine-lobe SH convolution constants (A_hat coefficients for diffuse
// irradiance). These pre-multiply the SH basis to convert radiance to
// irradiance via a cosine kernel convolution.
const SH_A0: f32 = 3.141_592_7; // pi
const SH_A1: f32 = 2.094_395_1; // 2*pi/3
const SH_A2: f32 = 0.785_398_2; // pi/4

impl SphericalHarmonics {
    /// Create a new zero-initialised SH.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create SH representing a uniform ambient colour.
    pub fn from_ambient(color: Vec3) -> Self {
        let mut sh = Self::new();
        // The DC coefficient for a uniform environment.
        let scale = SH_Y00;
        sh.coeffs[0][0] = color.x / scale;
        sh.coeffs[1][0] = color.y / scale;
        sh.coeffs[2][0] = color.z / scale;
        sh
    }

    /// Evaluate all 9 SH basis functions for a given direction.
    ///
    /// Returns [Y00, Y1m1, Y10, Y11, Y2m2, Y2m1, Y20, Y21, Y22].
    pub fn basis(direction: Vec3) -> [f32; SH_COEFF_COUNT] {
        let x = direction.x;
        let y = direction.y;
        let z = direction.z;

        [
            SH_Y00,                             // Y_0^0: constant band
            SH_Y1M1 * y,                        // Y_1^{-1}: linear y
            SH_Y10 * z,                         // Y_1^0: linear z
            SH_Y11 * x,                         // Y_1^1: linear x
            SH_Y2M2 * x * y,                    // Y_2^{-2}: xy product
            SH_Y2M1 * y * z,                    // Y_2^{-1}: yz product
            SH_Y20 * (3.0 * z * z - 1.0),       // Y_2^0: axial quadratic
            SH_Y21 * x * z,                      // Y_2^1: xz product
            SH_Y22 * (x * x - y * y),            // Y_2^2: planar quadratic
        ]
    }

    /// Individual SH basis functions for direct access.
    #[inline] pub fn y00() -> f32 { SH_Y00 }
    #[inline] pub fn y1m1(y: f32) -> f32 { SH_Y1M1 * y }
    #[inline] pub fn y10(z: f32) -> f32 { SH_Y10 * z }
    #[inline] pub fn y11(x: f32) -> f32 { SH_Y11 * x }
    #[inline] pub fn y2m2(x: f32, y: f32) -> f32 { SH_Y2M2 * x * y }
    #[inline] pub fn y2m1(y: f32, z: f32) -> f32 { SH_Y2M1 * y * z }
    #[inline] pub fn y20(z: f32) -> f32 { SH_Y20 * (3.0 * z * z - 1.0) }
    #[inline] pub fn y21(x: f32, z: f32) -> f32 { SH_Y21 * x * z }
    #[inline] pub fn y22(x: f32, y: f32) -> f32 { SH_Y22 * (x * x - y * y) }

    /// Evaluate the SH for a given direction. Returns the radiance (RGB)
    /// reconstructed from the stored coefficients.
    pub fn evaluate(&self, direction: Vec3) -> Vec3 {
        let basis = Self::basis(direction);
        let mut color = Vec3::ZERO;
        for i in 0..SH_COEFF_COUNT {
            color.x += self.coeffs[0][i] * basis[i];
            color.y += self.coeffs[1][i] * basis[i];
            color.z += self.coeffs[2][i] * basis[i];
        }
        // Clamp to non-negative (SH reconstruction can go negative).
        Vec3::new(color.x.max(0.0), color.y.max(0.0), color.z.max(0.0))
    }

    /// Evaluate the *irradiance* (cosine-lobe convolved) for a given normal.
    ///
    /// This applies the ZH (zonal harmonic) convolution factors A0, A1, A2
    /// to convert stored radiance SH into irradiance.
    pub fn evaluate_irradiance(&self, normal: Vec3) -> Vec3 {
        let basis = Self::basis(normal);
        let zonal = [SH_A0, SH_A1, SH_A1, SH_A1, SH_A2, SH_A2, SH_A2, SH_A2, SH_A2];
        let mut color = Vec3::ZERO;
        for i in 0..SH_COEFF_COUNT {
            let w = basis[i] * zonal[i];
            color.x += self.coeffs[0][i] * w;
            color.y += self.coeffs[1][i] * w;
            color.z += self.coeffs[2][i] * w;
        }
        Vec3::new(color.x.max(0.0), color.y.max(0.0), color.z.max(0.0))
    }

    /// Encode (project) a single directional sample onto SH.
    ///
    /// Adds the contribution of a single radiance sample to the accumulator.
    pub fn encode(&mut self, direction: Vec3, color: Vec3) {
        let basis = Self::basis(direction);
        for i in 0..SH_COEFF_COUNT {
            self.coeffs[0][i] += color.x * basis[i];
            self.coeffs[1][i] += color.y * basis[i];
            self.coeffs[2][i] += color.z * basis[i];
        }
    }

    /// Encode a weighted sample (explicit solid angle / pdf weight).
    pub fn encode_weighted(&mut self, direction: Vec3, color: Vec3, weight: f32) {
        let basis = Self::basis(direction);
        for i in 0..SH_COEFF_COUNT {
            self.coeffs[0][i] += color.x * basis[i] * weight;
            self.coeffs[1][i] += color.y * basis[i] * weight;
            self.coeffs[2][i] += color.z * basis[i] * weight;
        }
    }

    /// Add another SH to this one (accumulate).
    pub fn add(&mut self, other: &SphericalHarmonics) {
        for ch in 0..3 {
            for i in 0..SH_COEFF_COUNT {
                self.coeffs[ch][i] += other.coeffs[ch][i];
            }
        }
    }

    /// Subtract another SH from this one.
    pub fn subtract(&mut self, other: &SphericalHarmonics) {
        for ch in 0..3 {
            for i in 0..SH_COEFF_COUNT {
                self.coeffs[ch][i] -= other.coeffs[ch][i];
            }
        }
    }

    /// Scale all coefficients by a uniform factor.
    pub fn scale(&mut self, factor: f32) {
        for ch in 0..3 {
            for i in 0..SH_COEFF_COUNT {
                self.coeffs[ch][i] *= factor;
            }
        }
    }

    /// Scale each channel independently.
    pub fn scale_rgb(&mut self, r: f32, g: f32, b: f32) {
        for i in 0..SH_COEFF_COUNT {
            self.coeffs[0][i] *= r;
            self.coeffs[1][i] *= g;
            self.coeffs[2][i] *= b;
        }
    }

    /// Linearly interpolate between two SH sets.
    pub fn lerp(a: &SphericalHarmonics, b: &SphericalHarmonics, t: f32) -> Self {
        let mut result = Self::new();
        let one_minus_t = 1.0 - t;
        for ch in 0..3 {
            for i in 0..SH_COEFF_COUNT {
                result.coeffs[ch][i] = a.coeffs[ch][i] * one_minus_t + b.coeffs[ch][i] * t;
            }
        }
        result
    }

    /// Compute the dominant light direction from L1 (linear) band.
    pub fn dominant_direction(&self) -> Vec3 {
        let r = Vec3::new(self.coeffs[0][3], self.coeffs[0][1], self.coeffs[0][2]);
        let g = Vec3::new(self.coeffs[1][3], self.coeffs[1][1], self.coeffs[1][2]);
        let b = Vec3::new(self.coeffs[2][3], self.coeffs[2][1], self.coeffs[2][2]);
        // Luminance-weighted direction.
        let dir = r * 0.2126 + g * 0.7152 + b * 0.0722;
        if dir.length_squared() < 1e-8 {
            Vec3::Y
        } else {
            dir.normalize()
        }
    }

    /// Convert to a flat array of 27 floats (R0..R8, G0..G8, B0..B8).
    pub fn to_flat_array(&self) -> [f32; SH_FLOAT_COUNT] {
        let mut arr = [0.0f32; SH_FLOAT_COUNT];
        for ch in 0..3 {
            for i in 0..SH_COEFF_COUNT {
                arr[ch * SH_COEFF_COUNT + i] = self.coeffs[ch][i];
            }
        }
        arr
    }

    /// Create from a flat array of 27 floats.
    pub fn from_flat_array(arr: &[f32; SH_FLOAT_COUNT]) -> Self {
        let mut sh = Self::new();
        for ch in 0..3 {
            for i in 0..SH_COEFF_COUNT {
                sh.coeffs[ch][i] = arr[ch * SH_COEFF_COUNT + i];
            }
        }
        sh
    }

    /// Project a cubemap into SH coefficients.
    ///
    /// # Arguments
    /// - `faces` -- 6 face arrays of Vec3 pixel data, each `size * size`.
    ///   Face order: +X, -X, +Y, -Y, +Z, -Z.
    /// - `size` -- resolution of each face in pixels.
    pub fn from_cubemap(faces: &[&[Vec3]; 6], size: u32) -> SphericalHarmonics {
        let mut sh = SphericalHarmonics::new();
        let mut total_weight = 0.0f32;

        for face in 0..6u32 {
            for y in 0..size {
                for x in 0..size {
                    let u = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                    let v = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;

                    let dir = cubemap_direction(face, u, v).normalize_or_zero();
                    let texel_idx = (y * size + x) as usize;
                    let color = faces[face as usize][texel_idx];

                    // Solid angle of the cubemap texel (differential solid angle
                    // correction for cubemap projection distortion).
                    let texel_size = 2.0 / size as f32;
                    let solid_angle =
                        texel_size * texel_size / (1.0 + u * u + v * v).powf(1.5);

                    sh.encode_weighted(dir, color, solid_angle);
                    total_weight += solid_angle;
                }
            }
        }

        // Normalise by the total solid angle (should be approx 4*pi).
        if total_weight > 0.0 {
            let norm = 4.0 * PI / total_weight;
            sh.scale(norm);
        }

        sh
    }

    /// Project from a cubemap sampling function (closure interface).
    pub fn from_cubemap_fn(
        sample_cubemap: impl Fn(Vec3) -> Vec3,
        samples_per_face: u32,
    ) -> Self {
        let mut sh = Self::new();

        for face in 0..6u32 {
            for y in 0..samples_per_face {
                for x in 0..samples_per_face {
                    let u = (x as f32 + 0.5) / samples_per_face as f32 * 2.0 - 1.0;
                    let v = (y as f32 + 0.5) / samples_per_face as f32 * 2.0 - 1.0;

                    let dir = cubemap_direction(face, u, v).normalize_or_zero();
                    let color = sample_cubemap(dir);

                    let texel_size = 2.0 / samples_per_face as f32;
                    let solid_angle =
                        texel_size * texel_size / (1.0 + u * u + v * v).powf(1.5);

                    sh.encode_weighted(dir, color, solid_angle);
                }
            }
        }

        sh
    }

    /// Windowed sinc SH (Hanning window) to reduce ringing.
    pub fn apply_hanning_window(&mut self) {
        let w1 = (1.0 + (PI / 2.0).cos()) / 2.0;
        let w2 = (1.0 + PI.cos()) / 2.0;

        for ch in 0..3 {
            for i in 1..4 {
                self.coeffs[ch][i] *= w1;
            }
            for i in 4..9 {
                self.coeffs[ch][i] *= w2.max(0.0);
            }
        }
    }

    /// Compute the mean squared error between two SH sets.
    pub fn mse(&self, other: &SphericalHarmonics) -> f32 {
        let mut sum = 0.0f32;
        for ch in 0..3 {
            for i in 0..SH_COEFF_COUNT {
                let diff = self.coeffs[ch][i] - other.coeffs[ch][i];
                sum += diff * diff;
            }
        }
        sum / SH_FLOAT_COUNT as f32
    }
}

/// GPU-compatible SH data (padded to 16-byte alignment).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SphericalHarmonicsGpu {
    /// Coefficients packed as 7 Vec4s (28 floats, 27 used + 1 padding).
    pub data: [[f32; 4]; 7],
}

impl SphericalHarmonicsGpu {
    /// Build from a `SphericalHarmonics`.
    pub fn from_sh(sh: &SphericalHarmonics) -> Self {
        let flat = sh.to_flat_array();
        let mut data = [[0.0f32; 4]; 7];
        for i in 0..27 {
            data[i / 4][i % 4] = flat[i];
        }
        Self { data }
    }

    /// Convert back to a `SphericalHarmonics`.
    pub fn to_sh(&self) -> SphericalHarmonics {
        let mut arr = [0.0f32; SH_FLOAT_COUNT];
        for i in 0..27 {
            arr[i] = self.data[i / 4][i % 4];
        }
        SphericalHarmonics::from_flat_array(&arr)
    }
}

// ---------------------------------------------------------------------------
// Cubemap direction helpers
// ---------------------------------------------------------------------------

/// Convert cubemap face + UV to a direction vector.
///
/// Face indices: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z.
pub fn cubemap_direction(face: u32, u: f32, v: f32) -> Vec3 {
    match face {
        0 => Vec3::new(1.0, -v, -u),   // +X
        1 => Vec3::new(-1.0, -v, u),   // -X
        2 => Vec3::new(u, 1.0, v),     // +Y
        3 => Vec3::new(u, -1.0, -v),   // -Y
        4 => Vec3::new(u, -v, 1.0),    // +Z
        5 => Vec3::new(-u, -v, -1.0),  // -Z
        _ => Vec3::ZERO,
    }
}

/// Convert a direction vector to cubemap face + UV.
pub fn direction_to_cubemap(dir: Vec3) -> (u32, f32, f32) {
    let abs_dir = Vec3::new(dir.x.abs(), dir.y.abs(), dir.z.abs());
    let (face, u, v, ma) = if abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z {
        if dir.x > 0.0 {
            (0u32, -dir.z, -dir.y, abs_dir.x)
        } else {
            (1, dir.z, -dir.y, abs_dir.x)
        }
    } else if abs_dir.y >= abs_dir.x && abs_dir.y >= abs_dir.z {
        if dir.y > 0.0 {
            (2, dir.x, dir.z, abs_dir.y)
        } else {
            (3, dir.x, -dir.z, abs_dir.y)
        }
    } else if dir.z > 0.0 {
        (4, dir.x, -dir.y, abs_dir.z)
    } else {
        (5, -dir.x, -dir.y, abs_dir.z)
    };

    let u_coord = (u / ma + 1.0) * 0.5;
    let v_coord = (v / ma + 1.0) * 0.5;
    (face, u_coord, v_coord)
}

/// Sample a cubemap stored as 6 face arrays by direction.
pub fn sample_cubemap_faces(faces: &[&[Vec3]; 6], size: u32, direction: Vec3) -> Vec3 {
    let (face, u, v) = direction_to_cubemap(direction);
    let x = (u * size as f32).clamp(0.0, (size - 1) as f32) as usize;
    let y = (v * size as f32).clamp(0.0, (size - 1) as f32) as usize;
    let idx = y * size as usize + x;
    if idx < faces[face as usize].len() {
        faces[face as usize][idx]
    } else {
        Vec3::ZERO
    }
}

// ---------------------------------------------------------------------------
// LightProbe
// ---------------------------------------------------------------------------

/// A light probe capturing indirect illumination at a point in space.
#[derive(Debug, Clone)]
pub struct LightProbe {
    /// World-space position of the probe.
    pub position: Vec3,
    /// SH coefficients encoding the irradiance.
    pub irradiance_sh: SphericalHarmonics,
    /// Whether the probe data is valid (has been baked).
    pub valid: bool,
    /// Influence radius.
    pub radius: f32,
    /// Priority (higher = more important when overlapping).
    pub priority: u32,
}

impl LightProbe {
    /// Create a new probe at the given position.
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            irradiance_sh: SphericalHarmonics::new(),
            valid: false,
            radius: 10.0,
            priority: 0,
        }
    }

    /// Create a probe with uniform ambient lighting.
    pub fn with_ambient(position: Vec3, ambient: Vec3) -> Self {
        Self {
            position,
            irradiance_sh: SphericalHarmonics::from_ambient(ambient),
            valid: true,
            radius: 10.0,
            priority: 0,
        }
    }

    /// Evaluate the irradiance in a given direction.
    pub fn evaluate(&self, direction: Vec3) -> Vec3 {
        self.irradiance_sh.evaluate(direction)
    }

    /// Evaluate the cosine-convolved irradiance for a surface normal.
    pub fn evaluate_irradiance(&self, normal: Vec3) -> Vec3 {
        self.irradiance_sh.evaluate_irradiance(normal)
    }

    /// Bake the probe by sampling a cubemap.
    pub fn bake_from_cubemap(
        &mut self,
        sample_cubemap: impl Fn(Vec3) -> Vec3,
        samples_per_face: u32,
    ) {
        self.irradiance_sh =
            SphericalHarmonics::from_cubemap_fn(sample_cubemap, samples_per_face);
        self.valid = true;
    }

    /// Bake the probe from a set of scene lights.
    pub fn bake_from_lights(&mut self, lights: &[Light]) {
        self.irradiance_sh = SphericalHarmonics::new();

        for light in lights {
            match light {
                Light::Directional(d) => {
                    let dir = d.direction.normalize_or_zero();
                    let radiance = d.color * d.intensity;
                    let basis = SphericalHarmonics::basis(dir);
                    for i in 0..SH_COEFF_COUNT {
                        self.irradiance_sh.coeffs[0][i] += radiance.x * basis[i];
                        self.irradiance_sh.coeffs[1][i] += radiance.y * basis[i];
                        self.irradiance_sh.coeffs[2][i] += radiance.z * basis[i];
                    }
                }
                Light::Point(p) => {
                    let to_light = p.position - self.position;
                    let dist = to_light.length();
                    if dist < 1e-4 || dist > p.radius {
                        continue;
                    }
                    let dir = to_light / dist;
                    let atten = p.attenuation(dist);
                    let radiance = p.color * p.intensity * atten;
                    let basis = SphericalHarmonics::basis(dir);
                    for i in 0..SH_COEFF_COUNT {
                        self.irradiance_sh.coeffs[0][i] += radiance.x * basis[i];
                        self.irradiance_sh.coeffs[1][i] += radiance.y * basis[i];
                        self.irradiance_sh.coeffs[2][i] += radiance.z * basis[i];
                    }
                }
                Light::Spot(s) => {
                    let to_light = s.position - self.position;
                    let dist = to_light.length();
                    if dist < 1e-4 || dist > s.range {
                        continue;
                    }
                    let dir = to_light / dist;
                    let atten = s.total_attenuation(self.position);
                    let radiance = s.color * s.intensity * atten;
                    let basis = SphericalHarmonics::basis(dir);
                    for i in 0..SH_COEFF_COUNT {
                        self.irradiance_sh.coeffs[0][i] += radiance.x * basis[i];
                        self.irradiance_sh.coeffs[1][i] += radiance.y * basis[i];
                        self.irradiance_sh.coeffs[2][i] += radiance.z * basis[i];
                    }
                }
                Light::Area(a) => {
                    let to_light = a.position - self.position;
                    let dist = to_light.length();
                    if dist < 1e-4 || dist > a.range {
                        continue;
                    }
                    let dir = to_light / dist;
                    let atten_dist = 1.0 / (dist * dist).max(0.001);
                    let radiance = a.color * a.intensity * atten_dist;
                    let basis = SphericalHarmonics::basis(dir);
                    for i in 0..SH_COEFF_COUNT {
                        self.irradiance_sh.coeffs[0][i] += radiance.x * basis[i];
                        self.irradiance_sh.coeffs[1][i] += radiance.y * basis[i];
                        self.irradiance_sh.coeffs[2][i] += radiance.z * basis[i];
                    }
                }
            }
        }

        self.valid = true;
    }

    /// Compute the blend weight for this probe at a given position.
    pub fn blend_weight(&self, world_pos: Vec3) -> f32 {
        let dist = (world_pos - self.position).length();
        if dist >= self.radius { 0.0 } else { 1.0 - (dist / self.radius) }
    }
}

// ---------------------------------------------------------------------------
// LightProbeGrid
// ---------------------------------------------------------------------------

/// A 3D grid of light probes for volumetric irradiance lookup.
pub struct LightProbeGrid {
    /// Number of probes along each axis.
    pub resolution: [u32; 3],
    /// World-space origin (minimum corner of the grid).
    pub origin: Vec3,
    /// Cell size (spacing between adjacent probes).
    pub cell_size: Vec3,
    /// Total extent of the grid.
    pub extent: Vec3,
    /// The probes, stored in a flat array indexed by
    /// `z * res_y * res_x + y * res_x + x`.
    pub probes: Vec<LightProbe>,
}

impl LightProbeGrid {
    /// Create a new probe grid.
    pub fn new(origin: Vec3, extent: Vec3, resolution: [u32; 3]) -> Self {
        let res_x = resolution[0].max(2);
        let res_y = resolution[1].max(2);
        let res_z = resolution[2].max(2);
        let resolution = [res_x, res_y, res_z];

        let cell_size = Vec3::new(
            extent.x / (res_x - 1) as f32,
            extent.y / (res_y - 1) as f32,
            extent.z / (res_z - 1) as f32,
        );

        let total = (res_x * res_y * res_z) as usize;
        let mut probes = Vec::with_capacity(total);

        for z in 0..res_z {
            for y in 0..res_y {
                for x in 0..res_x {
                    let pos = origin
                        + Vec3::new(
                            x as f32 * cell_size.x,
                            y as f32 * cell_size.y,
                            z as f32 * cell_size.z,
                        );
                    probes.push(LightProbe::new(pos));
                }
            }
        }

        Self { resolution, origin, cell_size, extent, probes }
    }

    /// Total number of probes.
    pub fn probe_count(&self) -> usize { self.probes.len() }

    /// Convert a 3D grid index to a linear index.
    #[inline]
    pub fn to_linear_index(&self, x: u32, y: u32, z: u32) -> usize {
        (z * self.resolution[1] * self.resolution[0] + y * self.resolution[0] + x) as usize
    }

    /// Get a probe by grid coordinates.
    pub fn get_probe(&self, x: u32, y: u32, z: u32) -> &LightProbe {
        &self.probes[self.to_linear_index(x, y, z)]
    }

    /// Get a mutable probe by grid coordinates.
    pub fn get_probe_mut(&mut self, x: u32, y: u32, z: u32) -> &mut LightProbe {
        let idx = self.to_linear_index(x, y, z);
        &mut self.probes[idx]
    }

    /// Sample the irradiance at a position using trilinear interpolation.
    pub fn sample(&self, world_pos: Vec3, normal: Vec3) -> Vec3 {
        let sh = self.sample_sh(world_pos);
        sh.evaluate(normal)
    }

    /// Sample raw SH at a position via trilinear interpolation.
    pub fn sample_sh(&self, world_pos: Vec3) -> SphericalHarmonics {
        let local = world_pos - self.origin;
        let gx = (local.x / self.cell_size.x).clamp(0.0, (self.resolution[0] - 1) as f32);
        let gy = (local.y / self.cell_size.y).clamp(0.0, (self.resolution[1] - 1) as f32);
        let gz = (local.z / self.cell_size.z).clamp(0.0, (self.resolution[2] - 1) as f32);

        let x0 = gx.floor() as u32;
        let y0 = gy.floor() as u32;
        let z0 = gz.floor() as u32;
        let x1 = (x0 + 1).min(self.resolution[0] - 1);
        let y1 = (y0 + 1).min(self.resolution[1] - 1);
        let z1 = (z0 + 1).min(self.resolution[2] - 1);

        let fx = gx - gx.floor();
        let fy = gy - gy.floor();
        let fz = gz - gz.floor();

        let p000 = &self.get_probe(x0, y0, z0).irradiance_sh;
        let p100 = &self.get_probe(x1, y0, z0).irradiance_sh;
        let p010 = &self.get_probe(x0, y1, z0).irradiance_sh;
        let p110 = &self.get_probe(x1, y1, z0).irradiance_sh;
        let p001 = &self.get_probe(x0, y0, z1).irradiance_sh;
        let p101 = &self.get_probe(x1, y0, z1).irradiance_sh;
        let p011 = &self.get_probe(x0, y1, z1).irradiance_sh;
        let p111 = &self.get_probe(x1, y1, z1).irradiance_sh;

        let mut result = SphericalHarmonics::new();
        for ch in 0..3 {
            for i in 0..SH_COEFF_COUNT {
                let c00 = p000.coeffs[ch][i] * (1.0 - fx) + p100.coeffs[ch][i] * fx;
                let c10 = p010.coeffs[ch][i] * (1.0 - fx) + p110.coeffs[ch][i] * fx;
                let c01 = p001.coeffs[ch][i] * (1.0 - fx) + p101.coeffs[ch][i] * fx;
                let c11 = p011.coeffs[ch][i] * (1.0 - fx) + p111.coeffs[ch][i] * fx;
                let c0 = c00 * (1.0 - fy) + c10 * fy;
                let c1 = c01 * (1.0 - fy) + c11 * fy;
                result.coeffs[ch][i] = c0 * (1.0 - fz) + c1 * fz;
            }
        }
        result
    }

    /// Check if a position is inside the grid volume.
    pub fn contains(&self, world_pos: Vec3) -> bool {
        let local = world_pos - self.origin;
        local.x >= 0.0 && local.y >= 0.0 && local.z >= 0.0
            && local.x <= self.extent.x
            && local.y <= self.extent.y
            && local.z <= self.extent.z
    }

    /// Set all probes to a uniform ambient colour.
    pub fn fill_ambient(&mut self, color: Vec3) {
        for probe in &mut self.probes {
            probe.irradiance_sh = SphericalHarmonics::from_ambient(color);
            probe.valid = true;
        }
    }

    /// Probe position by grid coordinates.
    pub fn probe_position(&self, x: u32, y: u32, z: u32) -> Vec3 {
        self.origin + Vec3::new(
            x as f32 * self.cell_size.x,
            y as f32 * self.cell_size.y,
            z as f32 * self.cell_size.z,
        )
    }

    /// Bake all probes from scene lights.
    pub fn bake(&mut self, scene_lights: &[Light]) {
        for probe in &mut self.probes {
            probe.bake_from_lights(scene_lights);
        }
    }

    /// Bake from cubemap sampling (per-probe position).
    pub fn bake_from_cubemap(
        &mut self,
        sample_env: impl Fn(Vec3, Vec3) -> Vec3,
        samples_per_face: u32,
    ) {
        for probe in &mut self.probes {
            let pos = probe.position;
            probe.irradiance_sh = SphericalHarmonics::from_cubemap_fn(
                |dir| sample_env(pos, dir),
                samples_per_face,
            );
            probe.valid = true;
        }
    }

    /// Invalidate all probes.
    pub fn invalidate_all(&mut self) {
        for probe in &mut self.probes { probe.valid = false; }
    }

    /// Count valid probes.
    pub fn valid_probe_count(&self) -> usize {
        self.probes.iter().filter(|p| p.valid).count()
    }
}

// ---------------------------------------------------------------------------
// ReflectionProbe
// ---------------------------------------------------------------------------

/// A reflection probe for specular IBL with parallax-correct box projection.
#[derive(Debug, Clone)]
pub struct ReflectionProbe {
    pub position: Vec3,
    pub cubemap_handle: u64,
    pub mip_count: u32,
    pub blend_distance: f32,
    pub influence_radius: f32,
    pub box_min: Vec3,
    pub box_max: Vec3,
    pub use_box_projection: bool,
    pub priority: u32,
    pub valid: bool,
}

impl Default for ReflectionProbe {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO, cubemap_handle: 0, mip_count: 0,
            blend_distance: 1.0, influence_radius: 10.0,
            box_min: Vec3::splat(-5.0), box_max: Vec3::splat(5.0),
            use_box_projection: false, priority: 0, valid: false,
        }
    }
}

impl ReflectionProbe {
    pub fn new(position: Vec3, influence_radius: f32) -> Self {
        Self { position, influence_radius, ..Default::default() }
    }

    /// Create a box-projected reflection probe.
    pub fn new_box(position: Vec3, box_min: Vec3, box_max: Vec3) -> Self {
        let half_extents = (box_max - box_min) * 0.5;
        Self {
            position, influence_radius: half_extents.length(),
            box_min, box_max, use_box_projection: true, ..Default::default()
        }
    }

    /// Parallax-correct box projection for reflection directions.
    ///
    /// Intersects the reflection ray with the probe's AABB and uses the
    /// intersection point relative to probe center as sampling direction.
    pub fn box_project_direction(&self, world_pos: Vec3, reflection_dir: Vec3) -> Vec3 {
        if !self.use_box_projection {
            return reflection_dir;
        }

        let inv_dir = Vec3::new(
            if reflection_dir.x.abs() > 1e-8 { 1.0 / reflection_dir.x } else { f32::MAX },
            if reflection_dir.y.abs() > 1e-8 { 1.0 / reflection_dir.y } else { f32::MAX },
            if reflection_dir.z.abs() > 1e-8 { 1.0 / reflection_dir.z } else { f32::MAX },
        );

        let first_plane = (self.box_max - world_pos) * inv_dir;
        let second_plane = (self.box_min - world_pos) * inv_dir;

        let furthest = Vec3::new(
            first_plane.x.max(second_plane.x),
            first_plane.y.max(second_plane.y),
            first_plane.z.max(second_plane.z),
        );

        let dist = furthest.x.min(furthest.y).min(furthest.z);
        let intersection = world_pos + reflection_dir * dist;
        (intersection - self.position).normalize_or_zero()
    }

    /// Blend weight (1.0 inside, fading to 0.0 at edge).
    pub fn blend_weight(&self, world_pos: Vec3) -> f32 {
        if self.use_box_projection {
            let half = (self.box_max - self.box_min) * 0.5;
            let center = (self.box_max + self.box_min) * 0.5;
            let offset = world_pos - center;
            if half.x < 1e-6 || half.y < 1e-6 || half.z < 1e-6 { return 0.0; }
            let nx = offset.x.abs() / half.x;
            let ny = offset.y.abs() / half.y;
            let nz = offset.z.abs() / half.z;
            let max_dist = nx.max(ny).max(nz);
            if max_dist > 1.0 { return 0.0; }
            let blend_start = 1.0 - self.blend_distance / half.x.min(half.y).min(half.z).max(1e-4);
            if max_dist <= blend_start { 1.0 }
            else { 1.0 - (max_dist - blend_start) / (1.0 - blend_start).max(1e-6) }
        } else {
            let dist = (world_pos - self.position).length();
            if dist >= self.influence_radius { return 0.0; }
            let inner = self.influence_radius - self.blend_distance;
            if dist <= inner { return 1.0; }
            (1.0 - (dist - inner) / self.blend_distance.max(1e-6)).max(0.0)
        }
    }

    /// Mip level for a given roughness.
    pub fn mip_for_roughness(&self, roughness: f32) -> f32 {
        roughness * (self.mip_count.saturating_sub(1)) as f32
    }
}

/// GPU-compatible reflection probe data.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ReflectionProbeGpu {
    pub position_radius: [f32; 4],
    pub box_min_blend: [f32; 4],
    pub box_max_mips: [f32; 4],
    pub flags: [f32; 4],
}

impl ReflectionProbeGpu {
    pub fn from_probe(probe: &ReflectionProbe) -> Self {
        Self {
            position_radius: [probe.position.x, probe.position.y, probe.position.z, probe.influence_radius],
            box_min_blend: [probe.box_min.x, probe.box_min.y, probe.box_min.z, probe.blend_distance],
            box_max_mips: [probe.box_max.x, probe.box_max.y, probe.box_max.z, probe.mip_count as f32],
            flags: [if probe.use_box_projection { 1.0 } else { 0.0 }, probe.priority as f32, 0.0, 0.0],
        }
    }
}

// ---------------------------------------------------------------------------
// ReflectionProbeManager
// ---------------------------------------------------------------------------

/// Manages reflection probes: sorting by distance, blending overlapping probes.
pub struct ReflectionProbeManager {
    probes: Vec<ReflectionProbe>,
    pub max_blend_probes: usize,
}

impl ReflectionProbeManager {
    pub fn new() -> Self { Self { probes: Vec::new(), max_blend_probes: 2 } }

    pub fn add_probe(&mut self, probe: ReflectionProbe) { self.probes.push(probe); }

    pub fn remove_probe(&mut self, index: usize) -> Option<ReflectionProbe> {
        if index < self.probes.len() { Some(self.probes.remove(index)) } else { None }
    }

    pub fn probes(&self) -> &[ReflectionProbe] { &self.probes }
    pub fn probe_mut(&mut self, index: usize) -> Option<&mut ReflectionProbe> { self.probes.get_mut(index) }
    pub fn probe_count(&self) -> usize { self.probes.len() }

    /// Find probes affecting a position, sorted by priority desc then distance asc.
    pub fn find_affecting_probes(&self, world_pos: Vec3, max_count: usize) -> Vec<(usize, f32)> {
        let mut candidates: Vec<(usize, f32, u32, f32)> = Vec::new();
        for (i, probe) in self.probes.iter().enumerate() {
            if !probe.valid { continue; }
            let weight = probe.blend_weight(world_pos);
            if weight > 0.0 {
                let dist = (world_pos - probe.position).length();
                candidates.push((i, weight, probe.priority, dist));
            }
        }
        candidates.sort_by(|a, b| {
            b.2.cmp(&a.2).then(a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal))
        });
        candidates.into_iter().take(max_count).map(|(idx, w, _, _)| (idx, w)).collect()
    }

    /// Sample blended reflections: returns (probe_index, corrected_dir, weight).
    pub fn sample_blended(&self, world_pos: Vec3, reflection_dir: Vec3) -> Vec<(usize, Vec3, f32)> {
        let affecting = self.find_affecting_probes(world_pos, self.max_blend_probes);
        let total_weight: f32 = affecting.iter().map(|(_, w)| w).sum();
        if total_weight < 1e-6 { return Vec::new(); }
        affecting.into_iter().map(|(idx, weight)| {
            let probe = &self.probes[idx];
            let corrected = probe.box_project_direction(world_pos, reflection_dir);
            (idx, corrected, weight / total_weight)
        }).collect()
    }

    pub fn build_gpu_data(&self) -> Vec<ReflectionProbeGpu> {
        self.probes.iter().filter(|p| p.valid).map(|p| ReflectionProbeGpu::from_probe(p)).collect()
    }

    pub fn clear(&mut self) { self.probes.clear(); }
}

impl Default for ReflectionProbeManager {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// CubemapData
// ---------------------------------------------------------------------------

/// CPU-side cubemap data with 6 faces of Vec3 pixel data.
pub struct CubemapData {
    pub face_size: u32,
    pub faces: [Vec<Vec3>; 6],
}

impl CubemapData {
    pub fn new(face_size: u32) -> Self {
        let n = (face_size * face_size) as usize;
        Self {
            face_size,
            faces: [
                vec![Vec3::ZERO; n], vec![Vec3::ZERO; n], vec![Vec3::ZERO; n],
                vec![Vec3::ZERO; n], vec![Vec3::ZERO; n], vec![Vec3::ZERO; n],
            ],
        }
    }

    pub fn sample(&self, direction: Vec3) -> Vec3 {
        let (face, u, v) = direction_to_cubemap(direction);
        let x = (u * self.face_size as f32).clamp(0.0, (self.face_size - 1) as f32) as usize;
        let y = (v * self.face_size as f32).clamp(0.0, (self.face_size - 1) as f32) as usize;
        let idx = y * self.face_size as usize + x;
        self.faces[face as usize].get(idx).copied().unwrap_or(Vec3::ZERO)
    }

    pub fn write(&mut self, face: u32, x: u32, y: u32, color: Vec3) {
        let idx = (y * self.face_size + x) as usize;
        if let Some(texel) = self.faces[face as usize].get_mut(idx) { *texel = color; }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for face in &self.faces {
            for texel in face {
                bytes.extend_from_slice(bytemuck::bytes_of(&texel.x));
                bytes.extend_from_slice(bytemuck::bytes_of(&texel.y));
                bytes.extend_from_slice(bytemuck::bytes_of(&texel.z));
            }
        }
        bytes
    }

    pub fn total_texels(&self) -> usize { (self.face_size * self.face_size * 6) as usize }
}

// ---------------------------------------------------------------------------
// IrradianceMap -- diffuse environment convolution
// ---------------------------------------------------------------------------

/// Generates diffuse irradiance cubemaps via Monte Carlo hemisphere convolution.
pub struct IrradianceMap;

impl IrradianceMap {
    /// Convolve an environment cubemap to produce a diffuse irradiance cubemap.
    pub fn convolve_irradiance(env_cubemap: &CubemapData, output_size: u32) -> CubemapData {
        let mut output = CubemapData::new(output_size);
        let sample_sqrt = 16u32;

        for face in 0..6u32 {
            for y in 0..output_size {
                for x in 0..output_size {
                    let u = (x as f32 + 0.5) / output_size as f32 * 2.0 - 1.0;
                    let v = (y as f32 + 0.5) / output_size as f32 * 2.0 - 1.0;
                    let normal = cubemap_direction(face, u, v).normalize_or_zero();
                    let irradiance = convolve_hemisphere_mc(
                        normal, sample_sqrt, |dir| env_cubemap.sample(dir),
                    );
                    output.write(face, x, y, irradiance);
                }
            }
        }
        output
    }
}

fn convolve_hemisphere_mc(
    normal: Vec3,
    sample_sqrt: u32,
    sample_env: impl Fn(Vec3) -> Vec3,
) -> Vec3 {
    let mut irradiance = Vec3::ZERO;
    let mut total_weight = 0.0f32;

    let up = if normal.y.abs() < 0.999 { Vec3::Y } else { Vec3::X };
    let tangent = up.cross(normal).normalize_or_zero();
    let bitangent = normal.cross(tangent);

    for i in 0..sample_sqrt {
        for j in 0..sample_sqrt {
            let phi = 2.0 * PI * (i as f32 + 0.5) / sample_sqrt as f32;
            let theta = 0.5 * PI * (j as f32 + 0.5) / sample_sqrt as f32;
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();
            let ts_dir = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);
            let world_dir = (tangent * ts_dir.x + bitangent * ts_dir.y + normal * ts_dir.z).normalize_or_zero();
            let color = sample_env(world_dir);
            let weight = cos_theta * sin_theta;
            irradiance += color * weight;
            total_weight += weight;
        }
    }

    if total_weight > 0.0 { irradiance * (PI / total_weight) } else { Vec3::ZERO }
}

// ---------------------------------------------------------------------------
// PrefilterMap -- specular prefiltered environment map
// ---------------------------------------------------------------------------

/// Generates pre-filtered environment maps for specular IBL using GGX
/// importance sampling. Each mip level corresponds to increasing roughness.
pub struct PrefilterMap;

impl PrefilterMap {
    pub fn prefilter_environment(env_cubemap: &CubemapData, mip_levels: u32) -> Vec<CubemapData> {
        let sample_count = 512u32;
        let base_size = env_cubemap.face_size;
        let mut mips = Vec::with_capacity(mip_levels as usize);

        for mip in 0..mip_levels {
            let face_size = (base_size >> mip).max(1);
            let roughness = mip as f32 / (mip_levels - 1).max(1) as f32;
            let mut cubemap = CubemapData::new(face_size);

            for face in 0..6u32 {
                for y in 0..face_size {
                    for x in 0..face_size {
                        let u = (x as f32 + 0.5) / face_size as f32 * 2.0 - 1.0;
                        let v = (y as f32 + 0.5) / face_size as f32 * 2.0 - 1.0;
                        let normal = cubemap_direction(face, u, v).normalize_or_zero();
                        let color = prefilter_direction(
                            normal, roughness, sample_count,
                            |dir| env_cubemap.sample(dir),
                        );
                        cubemap.write(face, x, y, color);
                    }
                }
            }
            mips.push(cubemap);
        }
        mips
    }
}

/// GGX importance-sampled prefiltering for a single direction.
fn prefilter_direction(
    normal: Vec3, roughness: f32, sample_count: u32,
    sample_env: impl Fn(Vec3) -> Vec3,
) -> Vec3 {
    let n = normal;
    let v = normal;
    let mut total_color = Vec3::ZERO;
    let mut total_weight = 0.0f32;
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;

    for i in 0..sample_count {
        let xi = hammersley(i, sample_count);
        let phi = 2.0 * PI * xi.0;
        let cos_theta = ((1.0 - xi.1) / (1.0 + (alpha2 - 1.0) * xi.1)).sqrt();
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let h_ts = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);
        let h = tangent_to_world(h_ts, n);
        let l = (2.0 * v.dot(h) * h - v).normalize_or_zero();
        let n_dot_l = n.dot(l).max(0.0);
        if n_dot_l > 0.0 {
            total_color += sample_env(l) * n_dot_l;
            total_weight += n_dot_l;
        }
    }

    if total_weight > 0.0 { total_color / total_weight } else { Vec3::ZERO }
}

// ---------------------------------------------------------------------------
// BRDFIntegrationLUT -- split-sum BRDF LUT
// ---------------------------------------------------------------------------

/// Generates the 2D LUT for the split-sum IBL approximation.
/// U = NdotV, V = roughness. Output is (F0_scale, bias).
pub struct BRDFIntegrationLUT;

impl BRDFIntegrationLUT {
    /// Generate the BRDF integration LUT.
    pub fn generate_brdf_lut(size: u32) -> Vec<[f32; 2]> {
        let sample_count = 1024u32;
        let total = (size * size) as usize;
        let mut data = Vec::with_capacity(total);

        for y in 0..size {
            let roughness = (y as f32 + 0.5) / size as f32;
            for x in 0..size {
                let n_dot_v = ((x as f32 + 0.5) / size as f32).max(1e-4);
                let result = integrate_brdf(n_dot_v, roughness, sample_count);
                data.push([result.0, result.1]);
            }
        }
        data
    }

    pub fn generate_default() -> Vec<[f32; 2]> { Self::generate_brdf_lut(512) }

    pub fn to_bytes(data: &[[f32; 2]]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(data.len() * 8);
        for pixel in data {
            bytes.extend_from_slice(bytemuck::bytes_of(&pixel[0]));
            bytes.extend_from_slice(bytemuck::bytes_of(&pixel[1]));
        }
        bytes
    }
}

/// BRDF integration using GGX importance sampling. Returns (scale, bias).
fn integrate_brdf(n_dot_v: f32, roughness: f32, sample_count: u32) -> (f32, f32) {
    let v = Vec3::new((1.0 - n_dot_v * n_dot_v).max(0.0).sqrt(), 0.0, n_dot_v);
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let k = alpha / 2.0;
    let mut a = 0.0f32;
    let mut b = 0.0f32;

    for i in 0..sample_count {
        let xi = hammersley(i, sample_count);
        let phi = 2.0 * PI * xi.0;
        let cos_theta = ((1.0 - xi.1) / (1.0 + (alpha2 - 1.0) * xi.1)).sqrt();
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let h = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);
        let l = (2.0 * v.dot(h) * h - v).normalize_or_zero();
        let n_dot_l = l.z.max(0.0);
        let n_dot_h = h.z.max(0.0);
        let v_dot_h = v.dot(h).max(0.0);

        if n_dot_l > 0.0 {
            let g1_v = n_dot_v / (n_dot_v * (1.0 - k) + k).max(1e-7);
            let g1_l = n_dot_l / (n_dot_l * (1.0 - k) + k).max(1e-7);
            let g = g1_v * g1_l;
            let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v).max(1e-7);
            let fc = (1.0 - v_dot_h).max(0.0).powi(5);
            a += (1.0 - fc) * g_vis;
            b += fc * g_vis;
        }
    }

    let inv = 1.0 / sample_count as f32;
    (a * inv, b * inv)
}

// ---------------------------------------------------------------------------
// AmbientCube -- 6-direction ambient (simpler alternative to SH)
// ---------------------------------------------------------------------------

/// 6-direction ambient cube (Valve Source Engine style).
///
/// Stores one colour per axis direction (+X, -X, +Y, -Y, +Z, -Z).
/// Cheaper than SH but less accurate for complex lighting.
#[derive(Debug, Clone)]
pub struct AmbientCube {
    pub colors: [Vec3; 6],
}

impl Default for AmbientCube {
    fn default() -> Self { Self { colors: [Vec3::ZERO; 6] } }
}

impl AmbientCube {
    pub fn new() -> Self { Self::default() }

    pub fn from_ambient(color: Vec3) -> Self { Self { colors: [color; 6] } }

    /// Evaluate using squared-component weighting.
    pub fn evaluate(&self, direction: Vec3) -> Vec3 {
        let dir = direction.normalize_or_zero();
        let x_c = if dir.x >= 0.0 { self.colors[0] } else { self.colors[1] };
        let y_c = if dir.y >= 0.0 { self.colors[2] } else { self.colors[3] };
        let z_c = if dir.z >= 0.0 { self.colors[4] } else { self.colors[5] };
        x_c * (dir.x * dir.x) + y_c * (dir.y * dir.y) + z_c * (dir.z * dir.z)
    }

    pub fn encode(&mut self, direction: Vec3, color: Vec3) {
        let dir = direction.normalize_or_zero();
        if dir.x >= 0.0 { self.colors[0] += color * dir.x * dir.x; }
        else { self.colors[1] += color * dir.x * dir.x; }
        if dir.y >= 0.0 { self.colors[2] += color * dir.y * dir.y; }
        else { self.colors[3] += color * dir.y * dir.y; }
        if dir.z >= 0.0 { self.colors[4] += color * dir.z * dir.z; }
        else { self.colors[5] += color * dir.z * dir.z; }
    }

    pub fn from_lights(position: Vec3, lights: &[Light]) -> Self {
        let mut cube = Self::new();
        for light in lights {
            match light {
                Light::Directional(d) => {
                    cube.encode(d.direction, d.color * d.intensity);
                }
                Light::Point(p) => {
                    let to_light = p.position - position;
                    let dist = to_light.length();
                    if dist < 1e-4 || dist > p.radius { continue; }
                    let dir = to_light / dist;
                    cube.encode(dir, p.color * p.intensity * p.attenuation(dist));
                }
                Light::Spot(s) => {
                    let atten = s.total_attenuation(position);
                    if atten < 1e-6 { continue; }
                    let dir = (s.position - position).normalize_or_zero();
                    cube.encode(dir, s.color * s.intensity * atten);
                }
                Light::Area(a) => {
                    let to_light = a.position - position;
                    let dist = to_light.length();
                    if dist < 1e-4 || dist > a.range { continue; }
                    let dir = to_light / dist;
                    let atten = 1.0 / (dist * dist).max(0.001);
                    cube.encode(dir, a.color * a.intensity * atten);
                }
            }
        }
        cube
    }

    pub fn to_sh(&self) -> SphericalHarmonics {
        let mut sh = SphericalHarmonics::new();
        let dirs = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
        for (i, &dir) in dirs.iter().enumerate() {
            sh.encode_weighted(dir, self.colors[i], 1.0 / 6.0);
        }
        sh
    }

    pub fn scale(&mut self, factor: f32) { for c in &mut self.colors { *c *= factor; } }

    pub fn lerp(a: &AmbientCube, b: &AmbientCube, t: f32) -> Self {
        let mut result = AmbientCube::new();
        for i in 0..6 { result.colors[i] = a.colors[i] * (1.0 - t) + b.colors[i] * t; }
        result
    }
}

/// GPU-compatible ambient cube data.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AmbientCubeGpu {
    pub colors: [[f32; 4]; 6],
}

impl AmbientCubeGpu {
    pub fn from_cube(cube: &AmbientCube) -> Self {
        let mut colors = [[0.0f32; 4]; 6];
        for i in 0..6 { colors[i] = [cube.colors[i].x, cube.colors[i].y, cube.colors[i].z, 0.0]; }
        Self { colors }
    }
}

// ---------------------------------------------------------------------------
// AmbientOcclusion (CPU reference)
// ---------------------------------------------------------------------------

/// SSAO kernel and evaluation (CPU reference implementation).
pub struct AmbientOcclusion {
    pub radius: f32,
    pub intensity: f32,
    pub bias: f32,
    pub sample_count: u32,
    pub kernel: Vec<Vec3>,
    pub noise: Vec<Vec3>,
}

impl AmbientOcclusion {
    pub fn new(radius: f32, intensity: f32, sample_count: u32) -> Self {
        let kernel = generate_ao_kernel(sample_count);
        let noise = generate_ao_noise(16);
        Self { radius, intensity, bias: 0.025, sample_count, kernel, noise }
    }

    pub fn default_ssao() -> Self { Self::new(0.5, 1.0, 64) }

    pub fn evaluate(&self, position: Vec3, normal: Vec3, sample_depth: impl Fn(Vec3) -> f32) -> f32 {
        let mut occlusion = 0.0f32;
        for sample in &self.kernel {
            let oriented = if sample.dot(normal) < 0.0 { -*sample } else { *sample };
            let sample_pos = position + oriented * self.radius;
            let stored_depth = sample_depth(sample_pos);
            let range_check = ((position.z - stored_depth).abs() < self.radius) as u32 as f32;
            if stored_depth >= sample_pos.z + self.bias { occlusion += range_check; }
        }
        (1.0 - (occlusion / self.sample_count as f32) * self.intensity).clamp(0.0, 1.0)
    }
}

fn generate_ao_kernel(count: u32) -> Vec<Vec3> {
    let mut kernel = Vec::with_capacity(count as usize);
    for i in 0..count {
        let xi = hammersley(i, count);
        let phi = 2.0 * PI * xi.0;
        let cos_theta = 1.0 - xi.1;
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let mut dir = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);
        let mut scale = i as f32 / count as f32;
        scale = 0.1 + scale * scale * 0.9;
        dir *= scale;
        kernel.push(dir);
    }
    kernel
}

fn generate_ao_noise(count: u32) -> Vec<Vec3> {
    let mut noise = Vec::with_capacity(count as usize);
    for i in 0..count {
        let xi = hammersley(i, count);
        noise.push(Vec3::new(xi.0 * 2.0 - 1.0, xi.1 * 2.0 - 1.0, 0.0).normalize_or_zero());
    }
    noise
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

#[inline]
fn hammersley(i: u32, n: u32) -> (f32, f32) {
    (i as f32 / n as f32, radical_inverse_vdc(i))
}

#[inline]
fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    bits as f32 * 2.328_306_4e-10
}

fn tangent_to_world(tangent_vec: Vec3, normal: Vec3) -> Vec3 {
    let up = if normal.y.abs() < 0.999 { Vec3::Y } else { Vec3::X };
    let tangent = up.cross(normal).normalize_or_zero();
    let bitangent = normal.cross(tangent);
    (tangent * tangent_vec.x + bitangent * tangent_vec.y + normal * tangent_vec.z).normalize_or_zero()
}

/// Compute probe positions with vertical density adjustment.
pub fn compute_probe_positions(
    origin: Vec3, extent: Vec3, base_resolution: [u32; 3], density_y: f32,
) -> Vec<Vec3> {
    let res_x = base_resolution[0].max(2);
    let res_z = base_resolution[2].max(2);
    let res_y = base_resolution[1].max(2);
    let mut y_positions = Vec::with_capacity(res_y as usize);
    for i in 0..res_y {
        let t = i as f32 / (res_y - 1) as f32;
        y_positions.push(origin.y + t.powf(1.0 / density_y.max(0.1)) * extent.y);
    }
    let cell_x = extent.x / (res_x - 1) as f32;
    let cell_z = extent.z / (res_z - 1) as f32;
    let mut positions = Vec::with_capacity((res_x * res_y * res_z) as usize);
    for z in 0..res_z {
        for y_pos in &y_positions {
            for x in 0..res_x {
                positions.push(Vec3::new(origin.x + x as f32 * cell_x, *y_pos, origin.z + z as f32 * cell_z));
            }
        }
    }
    positions
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lighting::light_types::{DirectionalLight, PointLight};

    #[test]
    fn sh_basis_at_z_up() {
        let basis = SphericalHarmonics::basis(Vec3::Z);
        assert!((basis[0] - SH_Y00).abs() < 1e-5);
        assert!((basis[2] - SH_Y10).abs() < 1e-5);
    }

    #[test]
    fn sh_basis_functions_individual() {
        assert!((SphericalHarmonics::y00() - SH_Y00).abs() < 1e-6);
        assert!((SphericalHarmonics::y1m1(0.5) - SH_Y1M1 * 0.5).abs() < 1e-6);
        assert!((SphericalHarmonics::y10(0.8) - SH_Y10 * 0.8).abs() < 1e-6);
        assert!((SphericalHarmonics::y11(0.3) - SH_Y11 * 0.3).abs() < 1e-6);
    }

    #[test]
    fn sh_encode_decode_roundtrip() {
        let mut sh = SphericalHarmonics::new();
        sh.encode(Vec3::Z, Vec3::new(1.0, 0.5, 0.25));
        let result = sh.evaluate(Vec3::Z);
        assert!(result.x > 0.0 && result.y > 0.0 && result.z > 0.0);
    }

    #[test]
    fn sh_ambient_is_uniform() {
        let sh = SphericalHarmonics::from_ambient(Vec3::splat(0.3));
        let up = sh.evaluate(Vec3::Y);
        let down = sh.evaluate(-Vec3::Y);
        assert!((up.x - down.x).abs() < 0.05);
    }

    #[test]
    fn sh_add_and_scale() {
        let a = SphericalHarmonics::from_ambient(Vec3::splat(0.5));
        let mut c = a.clone();
        c.add(&a);
        let expected = SphericalHarmonics::from_ambient(Vec3::splat(1.0));
        assert!(c.mse(&expected) < 0.01);
        c.scale(0.5);
        assert!(c.mse(&a) < 0.01);
    }

    #[test]
    fn sh_flat_array_roundtrip() {
        let sh = SphericalHarmonics::from_ambient(Vec3::new(0.5, 0.3, 0.1));
        let arr = sh.to_flat_array();
        let sh2 = SphericalHarmonics::from_flat_array(&arr);
        assert!(sh.mse(&sh2) < 1e-10);
    }

    #[test]
    fn sh_from_cubemap() {
        let size = 4u32;
        let white = vec![Vec3::ONE; (size * size) as usize];
        let faces: [&[Vec3]; 6] = [&white, &white, &white, &white, &white, &white];
        let sh = SphericalHarmonics::from_cubemap(&faces, size);
        let up = sh.evaluate(Vec3::Y);
        let down = sh.evaluate(-Vec3::Y);
        assert!((up.x - down.x).abs() < 0.5);
    }

    #[test]
    fn sh_gpu_roundtrip() {
        let sh = SphericalHarmonics::from_ambient(Vec3::new(0.5, 0.3, 0.1));
        let gpu = SphericalHarmonicsGpu::from_sh(&sh);
        let sh2 = gpu.to_sh();
        assert!(sh.mse(&sh2) < 1e-10);
    }

    #[test]
    fn cubemap_direction_faces() {
        assert!(cubemap_direction(0, 0.0, 0.0).x > 0.0);
        assert!(cubemap_direction(1, 0.0, 0.0).x < 0.0);
    }

    #[test]
    fn light_probe_bake_from_lights() {
        let mut probe = LightProbe::new(Vec3::ZERO);
        let lights = vec![
            DirectionalLight::sun().to_light(),
            PointLight::new(Vec3::new(5.0, 5.0, 0.0), Vec3::ONE, 10.0, 20.0).to_light(),
        ];
        probe.bake_from_lights(&lights);
        assert!(probe.valid);
        assert!(probe.evaluate(Vec3::Y).x > 0.0);
    }

    #[test]
    fn light_probe_grid_creation() {
        let grid = LightProbeGrid::new(Vec3::ZERO, Vec3::new(10.0, 5.0, 10.0), [4, 3, 4]);
        assert_eq!(grid.probe_count(), 4 * 3 * 4);
    }

    #[test]
    fn light_probe_grid_bake() {
        let mut grid = LightProbeGrid::new(Vec3::ZERO, Vec3::new(10.0, 5.0, 10.0), [3, 2, 3]);
        grid.bake(&[DirectionalLight::sun().to_light()]);
        assert_eq!(grid.valid_probe_count(), grid.probe_count());
    }

    #[test]
    fn light_probe_grid_sample() {
        let mut grid = LightProbeGrid::new(Vec3::ZERO, Vec3::new(10.0, 5.0, 10.0), [3, 3, 3]);
        grid.fill_ambient(Vec3::splat(0.5));
        assert!(grid.sample(Vec3::new(5.0, 2.5, 5.0), Vec3::Y).x > 0.0);
    }

    #[test]
    fn reflection_probe_blend_weight() {
        let probe = ReflectionProbe::new(Vec3::ZERO, 10.0);
        assert!((probe.blend_weight(Vec3::ZERO) - 1.0).abs() < 0.01);
        assert!(probe.blend_weight(Vec3::new(15.0, 0.0, 0.0)).abs() < 0.01);
    }

    #[test]
    fn reflection_probe_box_projection() {
        let probe = ReflectionProbe::new_box(Vec3::ZERO, Vec3::splat(-5.0), Vec3::splat(5.0));
        let corrected = probe.box_project_direction(Vec3::new(2.0, 0.0, 0.0), Vec3::X);
        assert!(corrected.x > 0.0);
    }

    #[test]
    fn reflection_probe_manager_blending() {
        let mut mgr = ReflectionProbeManager::new();
        let mut p1 = ReflectionProbe::new(Vec3::ZERO, 10.0); p1.valid = true;
        let mut p2 = ReflectionProbe::new(Vec3::new(5.0, 0.0, 0.0), 10.0); p2.valid = true;
        mgr.add_probe(p1); mgr.add_probe(p2);
        let blended = mgr.sample_blended(Vec3::new(2.5, 0.0, 0.0), Vec3::X);
        assert_eq!(blended.len(), 2);
        let total_w: f32 = blended.iter().map(|(_, _, w)| w).sum();
        assert!((total_w - 1.0).abs() < 0.01);
    }

    #[test]
    fn ambient_cube_uniform() {
        let cube = AmbientCube::from_ambient(Vec3::splat(0.5));
        assert!((cube.evaluate(Vec3::Y).x - 0.5).abs() < 0.01);
    }

    #[test]
    fn ambient_cube_from_lights() {
        let cube = AmbientCube::from_lights(Vec3::ZERO, &[DirectionalLight::sun().to_light()]);
        let total: f32 = cube.colors.iter().map(|c| c.x + c.y + c.z).sum();
        assert!(total > 0.0);
    }

    #[test]
    fn brdf_lut_integration_range() {
        let (a, b) = integrate_brdf(0.5, 0.5, 256);
        assert!(a >= 0.0 && a <= 1.0);
        assert!(b >= 0.0 && b <= 1.0);
    }

    #[test]
    fn probe_placement_density() {
        let positions = compute_probe_positions(Vec3::ZERO, Vec3::splat(10.0), [3, 5, 3], 2.0);
        assert_eq!(positions.len(), 3 * 5 * 3);
    }

    #[test]
    fn ao_kernel_in_hemisphere() {
        let kernel = generate_ao_kernel(64);
        assert_eq!(kernel.len(), 64);
        for k in &kernel { assert!(k.z >= 0.0); }
    }
}
