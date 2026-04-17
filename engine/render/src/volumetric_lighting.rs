// engine/render/src/volumetric_lighting.rs
//
// Froxel-based volumetric lighting: ray-march through a frustum-aligned 3D
// volume (froxels) to accumulate in-scattered light from participating media.
// Supports per-light volumetric contribution, analytical/noise-driven density,
// temporal reprojection for stability, and phase-function evaluation.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Vec / Mat helpers
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

    #[inline] pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    #[inline] pub fn dot(self, r: Self) -> f32 { self.x*r.x + self.y*r.y + self.z*r.z }
    #[inline] pub fn length(self) -> f32 { self.dot(self).sqrt() }
    #[inline] pub fn normalize(self) -> Self { let l = self.length(); if l < 1e-12 { Self::ZERO } else { Self { x:self.x/l, y:self.y/l, z:self.z/l } } }
    #[inline] pub fn scale(self, s: f32) -> Self { Self { x:self.x*s, y:self.y*s, z:self.z*s } }
    #[inline] pub fn add(self, r: Self) -> Self { Self { x:self.x+r.x, y:self.y+r.y, z:self.z+r.z } }
    #[inline] pub fn sub(self, r: Self) -> Self { Self { x:self.x-r.x, y:self.y-r.y, z:self.z-r.z } }
    #[inline] pub fn mul_elem(self, r: Self) -> Self { Self { x:self.x*r.x, y:self.y*r.y, z:self.z*r.z } }
    #[inline] pub fn lerp(self, r: Self, t: f32) -> Self { self.add(r.sub(self).scale(t)) }
    #[inline] pub fn exp(self) -> Self { Self { x:self.x.exp(), y:self.y.exp(), z:self.z.exp() } }
    #[inline] pub fn neg(self) -> Self { Self { x:-self.x, y:-self.y, z:-self.z } }
}

/// 4x4 matrix for reprojection.
#[derive(Debug, Clone, Copy)]
pub struct Mat4 {
    pub cols: [[f32; 4]; 4],
}

impl Mat4 {
    pub const IDENTITY: Self = Self {
        cols: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    /// Transform a point (w=1) by this matrix.
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        let c = &self.cols;
        let w = c[0][3]*p.x + c[1][3]*p.y + c[2][3]*p.z + c[3][3];
        let inv_w = if w.abs() > 1e-12 { 1.0 / w } else { 1.0 };
        Vec3::new(
            (c[0][0]*p.x + c[1][0]*p.y + c[2][0]*p.z + c[3][0]) * inv_w,
            (c[0][1]*p.x + c[1][1]*p.y + c[2][1]*p.z + c[3][1]) * inv_w,
            (c[0][2]*p.x + c[1][2]*p.y + c[2][2]*p.z + c[3][2]) * inv_w,
        )
    }

    /// Multiply two matrices.
    pub fn mul(&self, rhs: &Mat4) -> Mat4 {
        let mut result = Mat4 { cols: [[0.0; 4]; 4] };
        for col in 0..4 {
            for row in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.cols[k][row] * rhs.cols[col][k];
                }
                result.cols[col][row] = sum;
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Froxel grid configuration
// ---------------------------------------------------------------------------

/// Configuration for the froxel (frustum-voxel) grid.
#[derive(Debug, Clone)]
pub struct FroxelGridConfig {
    /// Number of tiles along X (screen width / tile_size).
    pub tiles_x: u32,
    /// Number of tiles along Y (screen height / tile_size).
    pub tiles_y: u32,
    /// Number of depth slices.
    pub depth_slices: u32,
    /// Near plane distance.
    pub near: f32,
    /// Maximum volumetric distance (far plane for volumetrics).
    pub max_distance: f32,
    /// Depth slice distribution: 0 = linear, 1 = exponential.
    pub depth_distribution: f32,
    /// Temporal reprojection blend factor (0 = no history, 1 = all history).
    pub temporal_blend: f32,
    /// Jitter pattern for temporal anti-aliasing (index into Halton sequence).
    pub jitter_index: u32,
}

impl Default for FroxelGridConfig {
    fn default() -> Self {
        Self {
            tiles_x: 160,
            tiles_y: 90,
            depth_slices: 64,
            near: 0.1,
            max_distance: 200.0,
            depth_distribution: 0.7,
            temporal_blend: 0.9,
            jitter_index: 0,
        }
    }
}

impl FroxelGridConfig {
    /// Total number of froxels in the grid.
    pub fn total_froxels(&self) -> usize {
        self.tiles_x as usize * self.tiles_y as usize * self.depth_slices as usize
    }

    /// Convert a depth slice index to a linear depth value.
    pub fn slice_to_depth(&self, slice: u32) -> f32 {
        let t = slice as f32 / self.depth_slices as f32;
        let linear = self.near + t * (self.max_distance - self.near);
        let exponential = self.near * (self.max_distance / self.near).powf(t);
        linear * (1.0 - self.depth_distribution) + exponential * self.depth_distribution
    }

    /// Convert a linear depth to the nearest slice index.
    pub fn depth_to_slice(&self, depth: f32) -> u32 {
        let clamped = depth.clamp(self.near, self.max_distance);

        // Newton's method to invert the depth distribution
        let mut t = (clamped - self.near) / (self.max_distance - self.near);

        for _ in 0..4 {
            let linear = self.near + t * (self.max_distance - self.near);
            let exponential = self.near * (self.max_distance / self.near).powf(t);
            let f = linear * (1.0 - self.depth_distribution) + exponential * self.depth_distribution - clamped;

            let d_linear = self.max_distance - self.near;
            let d_exp = exponential * (self.max_distance / self.near).ln();
            let df = d_linear * (1.0 - self.depth_distribution) + d_exp * self.depth_distribution;

            if df.abs() > 1e-7 {
                t -= f / df;
            }
            t = t.clamp(0.0, 1.0);
        }

        (t * self.depth_slices as f32).round().clamp(0.0, (self.depth_slices - 1) as f32) as u32
    }

    /// Compute the 3D index of a froxel from a flat index.
    pub fn flat_to_3d(&self, idx: u32) -> (u32, u32, u32) {
        let z = idx / (self.tiles_x * self.tiles_y);
        let rem = idx % (self.tiles_x * self.tiles_y);
        let y = rem / self.tiles_x;
        let x = rem % self.tiles_x;
        (x, y, z)
    }

    /// Compute a flat index from 3D froxel coordinates.
    pub fn index_3d(&self, x: u32, y: u32, z: u32) -> u32 {
        z * self.tiles_x * self.tiles_y + y * self.tiles_x + x
    }

    /// Generate jitter offset for temporal anti-aliasing.
    pub fn jitter_offset(&self) -> f32 {
        halton(self.jitter_index, 2)
    }
}

fn halton(index: u32, base: u32) -> f32 {
    let mut result = 0.0_f32;
    let mut f = 1.0_f32;
    let mut i = index;
    while i > 0 {
        f /= base as f32;
        result += f * (i % base) as f32;
        i /= base;
    }
    result
}

// ---------------------------------------------------------------------------
// Phase functions
// ---------------------------------------------------------------------------

/// Phase function types for light scattering in participating media.
#[derive(Debug, Clone, Copy)]
pub enum PhaseFunction {
    /// Isotropic scattering (uniform in all directions).
    Isotropic,
    /// Henyey-Greenstein phase function. g in (-1, 1).
    /// g > 0 = forward scattering, g < 0 = back scattering.
    HenyeyGreenstein { g: f32 },
    /// Schlick approximation of Henyey-Greenstein.
    Schlick { k: f32 },
    /// Double Henyey-Greenstein (blend of forward and back scattering).
    DoubleHG { g_forward: f32, g_back: f32, blend: f32 },
    /// Rayleigh scattering (small particles, wavelength-dependent).
    Rayleigh,
    /// Mie scattering approximation.
    Mie { g: f32 },
    /// Cornette-Shanks (improved Henyey-Greenstein).
    CornetteShanks { g: f32 },
}

impl PhaseFunction {
    /// Evaluate the phase function given cos(theta) between view and light.
    pub fn evaluate(&self, cos_theta: f32) -> f32 {
        match *self {
            PhaseFunction::Isotropic => 1.0 / (4.0 * PI),

            PhaseFunction::HenyeyGreenstein { g } => {
                let g2 = g * g;
                let denom = 1.0 + g2 - 2.0 * g * cos_theta;
                (1.0 - g2) / (4.0 * PI * denom * denom.sqrt()).max(1e-7)
            }

            PhaseFunction::Schlick { k } => {
                let k2 = k * k;
                let denom = 1.0 + k * cos_theta;
                (1.0 - k2) / (4.0 * PI * denom * denom).max(1e-7)
            }

            PhaseFunction::DoubleHG { g_forward, g_back, blend } => {
                let fwd = PhaseFunction::HenyeyGreenstein { g: g_forward }.evaluate(cos_theta);
                let bck = PhaseFunction::HenyeyGreenstein { g: g_back }.evaluate(cos_theta);
                blend * fwd + (1.0 - blend) * bck
            }

            PhaseFunction::Rayleigh => {
                (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta)
            }

            PhaseFunction::Mie { g } => {
                let g2 = g * g;
                let numerator = 3.0 * (1.0 - g2) * (1.0 + cos_theta * cos_theta);
                let denominator = 8.0 * PI * (2.0 + g2) * (1.0 + g2 - 2.0 * g * cos_theta).powf(1.5);
                numerator / denominator.max(1e-7)
            }

            PhaseFunction::CornetteShanks { g } => {
                let g2 = g * g;
                let numerator = 3.0 * (1.0 - g2) * (1.0 + cos_theta * cos_theta);
                let denominator = 2.0 * (2.0 + g2) * (1.0 + g2 - 2.0 * g * cos_theta).powf(1.5);
                numerator / (4.0 * PI * denominator).max(1e-7)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Participating media description
// ---------------------------------------------------------------------------

/// Describes participating media properties for volumetric rendering.
#[derive(Debug, Clone)]
pub struct ParticipatingMedia {
    /// Scattering coefficient (how much light is scattered per unit distance).
    pub scattering: Vec3,
    /// Absorption coefficient (how much light is absorbed per unit distance).
    pub absorption: Vec3,
    /// Phase function for angular scattering distribution.
    pub phase: PhaseFunction,
    /// Density multiplier (uniform density scale).
    pub density: f32,
    /// Height fog parameters.
    pub height_fog: Option<HeightFogParams>,
    /// Noise parameters for volumetric density variation.
    pub noise: Option<VolumetricNoise>,
}

impl Default for ParticipatingMedia {
    fn default() -> Self {
        Self {
            scattering: Vec3::new(0.01, 0.01, 0.01),
            absorption: Vec3::new(0.001, 0.001, 0.001),
            phase: PhaseFunction::HenyeyGreenstein { g: 0.5 },
            density: 1.0,
            height_fog: None,
            noise: None,
        }
    }
}

impl ParticipatingMedia {
    /// Extinction = scattering + absorption.
    pub fn extinction(&self) -> Vec3 {
        self.scattering.add(self.absorption)
    }

    /// Single-scattering albedo = scattering / extinction.
    pub fn albedo(&self) -> Vec3 {
        let ext = self.extinction();
        Vec3::new(
            self.scattering.x / ext.x.max(1e-7),
            self.scattering.y / ext.y.max(1e-7),
            self.scattering.z / ext.z.max(1e-7),
        )
    }

    /// Sample density at a world-space position.
    pub fn density_at(&self, position: Vec3, time: f32) -> f32 {
        let mut d = self.density;

        // Apply height fog
        if let Some(ref hf) = self.height_fog {
            d *= hf.density_at_height(position.y);
        }

        // Apply noise
        if let Some(ref noise) = self.noise {
            d *= noise.sample(position, time);
        }

        d.max(0.0)
    }
}

/// Height-based fog density.
#[derive(Debug, Clone)]
pub struct HeightFogParams {
    /// Base height (full density below this).
    pub base_height: f32,
    /// Fog falloff rate (higher = fog falls off faster above base).
    pub falloff: f32,
    /// Maximum density at base.
    pub max_density: f32,
}

impl HeightFogParams {
    pub fn density_at_height(&self, height: f32) -> f32 {
        if height <= self.base_height {
            self.max_density
        } else {
            self.max_density * (-self.falloff * (height - self.base_height)).exp()
        }
    }
}

/// 3D noise for volumetric density modulation.
#[derive(Debug, Clone)]
pub struct VolumetricNoise {
    pub frequency: f32,
    pub amplitude: f32,
    pub octaves: u32,
    pub lacunarity: f32,
    pub persistence: f32,
    pub wind_direction: Vec3,
    pub wind_speed: f32,
    pub noise_type: NoiseType,
}

#[derive(Debug, Clone, Copy)]
pub enum NoiseType {
    Perlin,
    Worley,
    PerlinWorley, // blend
}

impl VolumetricNoise {
    pub fn sample(&self, position: Vec3, time: f32) -> f32 {
        let wind_offset = self.wind_direction.scale(self.wind_speed * time);
        let p = position.add(wind_offset).scale(self.frequency);

        let mut value = 0.0_f32;
        let mut freq = 1.0_f32;
        let mut amp = 1.0_f32;
        let mut total_amp = 0.0_f32;

        for _ in 0..self.octaves {
            let noise_val = match self.noise_type {
                NoiseType::Perlin => pseudo_perlin_3d(p.x * freq, p.y * freq, p.z * freq),
                NoiseType::Worley => pseudo_worley_3d(p.x * freq, p.y * freq, p.z * freq),
                NoiseType::PerlinWorley => {
                    let pe = pseudo_perlin_3d(p.x * freq, p.y * freq, p.z * freq);
                    let wo = pseudo_worley_3d(p.x * freq, p.y * freq, p.z * freq);
                    pe * 0.6 + wo * 0.4
                }
            };
            value += noise_val * amp;
            total_amp += amp;
            freq *= self.lacunarity;
            amp *= self.persistence;
        }

        let normalized = value / total_amp.max(1e-7);
        (0.5 + normalized * self.amplitude).clamp(0.0, 1.0)
    }
}

/// Simple hash-based pseudo-Perlin noise for volumetric density.
fn pseudo_perlin_3d(x: f32, y: f32, z: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;
    let xf = x - x.floor();
    let yf = y - y.floor();
    let zf = z - z.floor();

    let u = xf * xf * (3.0 - 2.0 * xf);
    let v = yf * yf * (3.0 - 2.0 * yf);
    let w = zf * zf * (3.0 - 2.0 * zf);

    let n000 = hash_grad_3d(xi, yi, zi, xf, yf, zf);
    let n100 = hash_grad_3d(xi+1, yi, zi, xf-1.0, yf, zf);
    let n010 = hash_grad_3d(xi, yi+1, zi, xf, yf-1.0, zf);
    let n110 = hash_grad_3d(xi+1, yi+1, zi, xf-1.0, yf-1.0, zf);
    let n001 = hash_grad_3d(xi, yi, zi+1, xf, yf, zf-1.0);
    let n101 = hash_grad_3d(xi+1, yi, zi+1, xf-1.0, yf, zf-1.0);
    let n011 = hash_grad_3d(xi, yi+1, zi+1, xf, yf-1.0, zf-1.0);
    let n111 = hash_grad_3d(xi+1, yi+1, zi+1, xf-1.0, yf-1.0, zf-1.0);

    let x1 = n000 + u * (n100 - n000);
    let x2 = n010 + u * (n110 - n010);
    let x3 = n001 + u * (n101 - n001);
    let x4 = n011 + u * (n111 - n011);

    let y1 = x1 + v * (x2 - x1);
    let y2 = x3 + v * (x4 - x3);

    y1 + w * (y2 - y1)
}

fn hash_grad_3d(xi: i32, yi: i32, zi: i32, xf: f32, yf: f32, zf: f32) -> f32 {
    let h = hash_i32(xi.wrapping_mul(374761393)
        .wrapping_add(yi.wrapping_mul(668265263))
        .wrapping_add(zi.wrapping_mul(1274126177)));
    let grad_x = if h & 1 != 0 { 1.0_f32 } else { -1.0 };
    let grad_y = if h & 2 != 0 { 1.0_f32 } else { -1.0 };
    let grad_z = if h & 4 != 0 { 1.0_f32 } else { -1.0 };
    grad_x * xf + grad_y * yf + grad_z * zf
}

fn hash_i32(mut x: i32) -> i32 {
    let x = x as u32;
    let x = x.wrapping_mul(0x9E3779B9);
    let x = (x ^ (x >> 16)).wrapping_mul(0x45D9F3B);
    let x = x ^ (x >> 16);
    x as i32
}

fn pseudo_worley_3d(x: f32, y: f32, z: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;
    let xf = x - x.floor();
    let yf = y - y.floor();
    let zf = z - z.floor();

    let mut min_dist = f32::MAX;

    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let hash = hash_i32(
                    (xi + dx).wrapping_mul(374761393)
                    .wrapping_add((yi + dy).wrapping_mul(668265263))
                    .wrapping_add((zi + dz).wrapping_mul(1274126177))
                );
                let px = dx as f32 + (hash & 0xFF) as f32 / 255.0 - xf;
                let py = dy as f32 + ((hash >> 8) & 0xFF) as f32 / 255.0 - yf;
                let pz = dz as f32 + ((hash >> 16) & 0xFF) as f32 / 255.0 - zf;
                let dist = px * px + py * py + pz * pz;
                min_dist = min_dist.min(dist);
            }
        }
    }

    1.0 - min_dist.sqrt().min(1.0)
}

// ---------------------------------------------------------------------------
// Light source types for volumetric contribution
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum VolumetricLight {
    Directional {
        direction: Vec3,     // normalized direction towards the light
        color: Vec3,
        intensity: f32,
        shadow_map_index: Option<u32>,
    },
    Point {
        position: Vec3,
        color: Vec3,
        intensity: f32,
        range: f32,
        shadow_cubemap_index: Option<u32>,
    },
    Spot {
        position: Vec3,
        direction: Vec3,
        color: Vec3,
        intensity: f32,
        range: f32,
        inner_angle: f32, // radians
        outer_angle: f32, // radians
        shadow_map_index: Option<u32>,
    },
}

impl VolumetricLight {
    /// Evaluate light contribution at a world-space position.
    pub fn evaluate_at(&self, position: Vec3) -> LightSample {
        match self {
            VolumetricLight::Directional { direction, color, intensity, .. } => {
                LightSample {
                    direction: *direction,
                    color: color.scale(*intensity),
                    attenuation: 1.0,
                    shadow: 1.0,
                }
            }

            VolumetricLight::Point { position: light_pos, color, intensity, range, .. } => {
                let to_light = light_pos.sub(position);
                let dist = to_light.length();
                let dir = if dist > 1e-6 { to_light.scale(1.0 / dist) } else { Vec3::new(0.0, 1.0, 0.0) };

                // Smooth distance attenuation
                let dist_ratio = (dist / range).min(1.0);
                let dist_ratio2 = dist_ratio * dist_ratio;
                let atten = ((1.0 - dist_ratio2) * (1.0 - dist_ratio2)).max(0.0) / (dist * dist + 1.0);

                LightSample {
                    direction: dir,
                    color: color.scale(*intensity),
                    attenuation: atten,
                    shadow: 1.0,
                }
            }

            VolumetricLight::Spot { position: light_pos, direction: light_dir, color, intensity, range, inner_angle, outer_angle, .. } => {
                let to_light = light_pos.sub(position);
                let dist = to_light.length();
                let dir = if dist > 1e-6 { to_light.scale(1.0 / dist) } else { Vec3::new(0.0, 1.0, 0.0) };

                // Distance attenuation
                let dist_ratio = (dist / range).min(1.0);
                let dist_ratio2 = dist_ratio * dist_ratio;
                let dist_atten = ((1.0 - dist_ratio2) * (1.0 - dist_ratio2)).max(0.0) / (dist * dist + 1.0);

                // Angular attenuation (cone)
                let cos_theta = dir.neg().dot(*light_dir);
                let cos_inner = inner_angle.cos();
                let cos_outer = outer_angle.cos();
                let angle_atten = ((cos_theta - cos_outer) / (cos_inner - cos_outer).max(1e-7))
                    .clamp(0.0, 1.0);
                let angle_atten = angle_atten * angle_atten;

                LightSample {
                    direction: dir,
                    color: color.scale(*intensity),
                    attenuation: dist_atten * angle_atten,
                    shadow: 1.0,
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LightSample {
    pub direction: Vec3,
    pub color: Vec3,
    pub attenuation: f32,
    pub shadow: f32,
}

// ---------------------------------------------------------------------------
// Froxel data
// ---------------------------------------------------------------------------

/// Per-froxel scattering and transmittance data.
#[derive(Debug, Clone, Copy)]
pub struct FroxelData {
    /// Accumulated in-scattered luminance.
    pub scattering: Vec3,
    /// Transmittance through this froxel.
    pub transmittance: Vec3,
    /// Density at froxel center.
    pub density: f32,
}

impl Default for FroxelData {
    fn default() -> Self {
        Self {
            scattering: Vec3::ZERO,
            transmittance: Vec3::ONE,
            density: 0.0,
        }
    }
}

/// Complete volumetric lighting system.
pub struct VolumetricLightingSystem {
    pub config: FroxelGridConfig,
    pub media: ParticipatingMedia,
    pub froxels: Vec<FroxelData>,
    pub integrated: Vec<FroxelData>, // front-to-back integrated
    pub history: Vec<FroxelData>,    // previous frame for temporal reprojection
    pub lights: Vec<VolumetricLight>,
    pub camera_position: Vec3,
    pub view_projection: Mat4,
    pub prev_view_projection: Mat4,
    pub inv_view_projection: Mat4,
    pub frame_count: u32,
}

impl VolumetricLightingSystem {
    /// Create a new volumetric lighting system with the given configuration.
    pub fn new(config: FroxelGridConfig) -> Self {
        let total = config.total_froxels();
        Self {
            config,
            media: ParticipatingMedia::default(),
            froxels: vec![FroxelData::default(); total],
            integrated: vec![FroxelData::default(); total],
            history: vec![FroxelData::default(); total],
            lights: Vec::new(),
            camera_position: Vec3::ZERO,
            view_projection: Mat4::IDENTITY,
            prev_view_projection: Mat4::IDENTITY,
            inv_view_projection: Mat4::IDENTITY,
            frame_count: 0,
        }
    }

    /// Resize the froxel grid.
    pub fn resize(&mut self, tiles_x: u32, tiles_y: u32) {
        self.config.tiles_x = tiles_x;
        self.config.tiles_y = tiles_y;
        let total = self.config.total_froxels();
        self.froxels.resize(total, FroxelData::default());
        self.integrated.resize(total, FroxelData::default());
        self.history.resize(total, FroxelData::default());
    }

    /// Run the complete volumetric lighting pipeline for one frame.
    pub fn update(&mut self, time: f32, shadow_samples: &dyn Fn(Vec3, u32) -> f32) {
        // Step 1: Compute per-froxel density and in-scattering
        self.compute_froxels(time, shadow_samples);

        // Step 2: Temporal reprojection
        self.temporal_reproject();

        // Step 3: Front-to-back integration (ray march accumulation)
        self.integrate();

        // Save history
        self.history.copy_from_slice(&self.froxels);
        self.frame_count += 1;
    }

    /// Step 1: Compute density and in-scattering for each froxel.
    fn compute_froxels(&mut self, time: f32, shadow_samples: &dyn Fn(Vec3, u32) -> f32) {
        let tiles_x = self.config.tiles_x;
        let tiles_y = self.config.tiles_y;
        let depth_slices = self.config.depth_slices;
        let jitter = self.config.jitter_offset();

        for z in 0..depth_slices {
            let depth_front = self.config.slice_to_depth(z);
            let depth_back = self.config.slice_to_depth(z + 1);
            let depth_center = (depth_front + depth_back) * 0.5;
            let slice_thickness = depth_back - depth_front;

            // Apply temporal jitter to depth
            let jittered_depth = depth_center + jitter * slice_thickness * 0.5;

            for y in 0..tiles_y {
                for x in 0..tiles_x {
                    let idx = self.config.index_3d(x, y, z) as usize;

                    // Compute world position of froxel center
                    let ndc_x = (x as f32 + 0.5) / tiles_x as f32 * 2.0 - 1.0;
                    let ndc_y = 1.0 - (y as f32 + 0.5) / tiles_y as f32 * 2.0;
                    let world_pos = self.ndc_to_world(ndc_x, ndc_y, jittered_depth);

                    // Sample density
                    let density = self.media.density_at(world_pos, time);
                    self.froxels[idx].density = density;

                    if density < 1e-6 {
                        self.froxels[idx].scattering = Vec3::ZERO;
                        self.froxels[idx].transmittance = Vec3::ONE;
                        continue;
                    }

                    // Compute extinction for this froxel
                    let extinction = self.media.extinction().scale(density);

                    // Transmittance through this froxel (Beer-Lambert)
                    self.froxels[idx].transmittance = Vec3::new(
                        (-extinction.x * slice_thickness).exp(),
                        (-extinction.y * slice_thickness).exp(),
                        (-extinction.z * slice_thickness).exp(),
                    );

                    // In-scattering: accumulate contribution from each light
                    let mut in_scatter = Vec3::ZERO;
                    let view_dir = self.camera_position.sub(world_pos).normalize();

                    for (light_idx, light) in self.lights.iter().enumerate() {
                        let sample = light.evaluate_at(world_pos);

                        // Shadow sampling
                        let shadow = match light {
                            VolumetricLight::Directional { shadow_map_index: Some(idx), .. } |
                            VolumetricLight::Spot { shadow_map_index: Some(idx), .. } => {
                                shadow_samples(world_pos, *idx)
                            }
                            VolumetricLight::Point { shadow_cubemap_index: Some(idx), .. } => {
                                shadow_samples(world_pos, *idx)
                            }
                            _ => 1.0,
                        };

                        // Phase function evaluation
                        let cos_theta = view_dir.dot(sample.direction);
                        let phase = self.media.phase.evaluate(cos_theta);

                        // In-scattered contribution from this light
                        let scattering_coeff = self.media.scattering.scale(density);
                        let light_contrib = sample.color.scale(
                            sample.attenuation * shadow * phase * slice_thickness
                        );
                        in_scatter = in_scatter.add(scattering_coeff.mul_elem(light_contrib));
                    }

                    self.froxels[idx].scattering = in_scatter;
                }
            }
        }
    }

    /// Step 2: Temporal reprojection - blend with previous frame's data.
    fn temporal_reproject(&mut self) {
        if self.frame_count == 0 || self.config.temporal_blend <= 0.0 {
            return;
        }

        let tiles_x = self.config.tiles_x;
        let tiles_y = self.config.tiles_y;
        let depth_slices = self.config.depth_slices;
        let blend = self.config.temporal_blend;

        for z in 0..depth_slices {
            let depth = self.config.slice_to_depth(z);
            for y in 0..tiles_y {
                for x in 0..tiles_x {
                    let idx = self.config.index_3d(x, y, z) as usize;

                    // Compute world position
                    let ndc_x = (x as f32 + 0.5) / tiles_x as f32 * 2.0 - 1.0;
                    let ndc_y = 1.0 - (y as f32 + 0.5) / tiles_y as f32 * 2.0;
                    let world_pos = self.ndc_to_world(ndc_x, ndc_y, depth);

                    // Reproject to previous frame
                    let prev_ndc = self.prev_view_projection.transform_point(world_pos);
                    let prev_x = ((prev_ndc.x * 0.5 + 0.5) * tiles_x as f32) as i32;
                    let prev_y = (((1.0 - prev_ndc.y) * 0.5) * tiles_y as f32) as i32;
                    let prev_z = self.config.depth_to_slice(depth) as i32;

                    // Check if reprojected position is valid
                    if prev_x >= 0 && prev_x < tiles_x as i32
                        && prev_y >= 0 && prev_y < tiles_y as i32
                        && prev_z >= 0 && prev_z < depth_slices as i32
                    {
                        let prev_idx = self.config.index_3d(
                            prev_x as u32, prev_y as u32, prev_z as u32
                        ) as usize;

                        if prev_idx < self.history.len() {
                            let history = &self.history[prev_idx];
                            let current = &self.froxels[idx];

                            // Neighborhood clamp to prevent ghosting
                            let clamped_scatter = clamp_to_neighborhood(
                                history.scattering,
                                current.scattering,
                                0.5, // tolerance
                            );

                            self.froxels[idx].scattering = current.scattering.lerp(
                                clamped_scatter, blend
                            );
                        }
                    }
                }
            }
        }
    }

    /// Step 3: Front-to-back integration along view rays.
    fn integrate(&mut self) {
        let tiles_x = self.config.tiles_x;
        let tiles_y = self.config.tiles_y;
        let depth_slices = self.config.depth_slices;

        for y in 0..tiles_y {
            for x in 0..tiles_x {
                let mut accumulated_scatter = Vec3::ZERO;
                let mut accumulated_transmittance = Vec3::ONE;

                for z in 0..depth_slices {
                    let idx = self.config.index_3d(x, y, z) as usize;
                    let froxel = &self.froxels[idx];

                    // Accumulate: L_out = L_in * T + S
                    accumulated_scatter = accumulated_scatter.add(
                        froxel.scattering.mul_elem(accumulated_transmittance)
                    );
                    accumulated_transmittance = accumulated_transmittance.mul_elem(
                        froxel.transmittance
                    );

                    self.integrated[idx] = FroxelData {
                        scattering: accumulated_scatter,
                        transmittance: accumulated_transmittance,
                        density: froxel.density,
                    };
                }
            }
        }
    }

    /// Convert NDC coordinates + linear depth to world space.
    fn ndc_to_world(&self, ndc_x: f32, ndc_y: f32, linear_depth: f32) -> Vec3 {
        // Approximate: use inverse VP and a pseudo-NDC z
        let ndc_z = linear_depth / self.config.max_distance * 2.0 - 1.0;
        self.inv_view_projection.transform_point(Vec3::new(ndc_x, ndc_y, ndc_z))
    }

    /// Sample the integrated volumetric fog at a screen position and depth.
    pub fn sample_at(&self, screen_uv_x: f32, screen_uv_y: f32, linear_depth: f32) -> FroxelData {
        let x = (screen_uv_x * self.config.tiles_x as f32) as u32;
        let y = (screen_uv_y * self.config.tiles_y as f32) as u32;
        let z = self.config.depth_to_slice(linear_depth);

        let x = x.min(self.config.tiles_x - 1);
        let y = y.min(self.config.tiles_y - 1);
        let z = z.min(self.config.depth_slices - 1);

        let idx = self.config.index_3d(x, y, z) as usize;
        if idx < self.integrated.len() {
            self.integrated[idx]
        } else {
            FroxelData::default()
        }
    }

    /// Sample with trilinear interpolation.
    pub fn sample_trilinear(&self, screen_uv_x: f32, screen_uv_y: f32, linear_depth: f32) -> FroxelData {
        let fx = screen_uv_x * self.config.tiles_x as f32 - 0.5;
        let fy = screen_uv_y * self.config.tiles_y as f32 - 0.5;

        // Compute fractional slice
        let fz = self.fractional_slice(linear_depth);

        let x0 = (fx.floor() as i32).max(0) as u32;
        let y0 = (fy.floor() as i32).max(0) as u32;
        let z0 = (fz.floor() as i32).max(0) as u32;
        let x1 = (x0 + 1).min(self.config.tiles_x - 1);
        let y1 = (y0 + 1).min(self.config.tiles_y - 1);
        let z1 = (z0 + 1).min(self.config.depth_slices - 1);

        let tx = fx - fx.floor();
        let ty = fy - fy.floor();
        let tz = fz - fz.floor();

        // 8 corner samples
        let s000 = self.fetch(x0, y0, z0);
        let s100 = self.fetch(x1, y0, z0);
        let s010 = self.fetch(x0, y1, z0);
        let s110 = self.fetch(x1, y1, z0);
        let s001 = self.fetch(x0, y0, z1);
        let s101 = self.fetch(x1, y0, z1);
        let s011 = self.fetch(x0, y1, z1);
        let s111 = self.fetch(x1, y1, z1);

        // Trilinear interpolation
        let lerp_scatter = |a: Vec3, b: Vec3, t: f32| a.lerp(b, t);
        let lerp_trans = |a: Vec3, b: Vec3, t: f32| a.lerp(b, t);

        let sx00 = lerp_scatter(s000.scattering, s100.scattering, tx);
        let sx10 = lerp_scatter(s010.scattering, s110.scattering, tx);
        let sx01 = lerp_scatter(s001.scattering, s101.scattering, tx);
        let sx11 = lerp_scatter(s011.scattering, s111.scattering, tx);

        let sxy0 = lerp_scatter(sx00, sx10, ty);
        let sxy1 = lerp_scatter(sx01, sx11, ty);
        let scatter = lerp_scatter(sxy0, sxy1, tz);

        let tx00 = lerp_trans(s000.transmittance, s100.transmittance, tx);
        let tx10 = lerp_trans(s010.transmittance, s110.transmittance, tx);
        let tx01 = lerp_trans(s001.transmittance, s101.transmittance, tx);
        let tx11 = lerp_trans(s011.transmittance, s111.transmittance, tx);

        let txy0 = lerp_trans(tx00, tx10, ty);
        let txy1 = lerp_trans(tx01, tx11, ty);
        let transmittance = lerp_trans(txy0, txy1, tz);

        FroxelData {
            scattering: scatter,
            transmittance,
            density: 0.0,
        }
    }

    fn fetch(&self, x: u32, y: u32, z: u32) -> FroxelData {
        let idx = self.config.index_3d(x, y, z) as usize;
        if idx < self.integrated.len() {
            self.integrated[idx]
        } else {
            FroxelData::default()
        }
    }

    fn fractional_slice(&self, depth: f32) -> f32 {
        let clamped = depth.clamp(self.config.near, self.config.max_distance);
        let mut t = (clamped - self.config.near) / (self.config.max_distance - self.config.near);
        for _ in 0..4 {
            let linear = self.config.near + t * (self.config.max_distance - self.config.near);
            let exp_val = self.config.near * (self.config.max_distance / self.config.near).powf(t);
            let val = linear * (1.0 - self.config.depth_distribution) + exp_val * self.config.depth_distribution;
            let d_linear = self.config.max_distance - self.config.near;
            let d_exp = exp_val * (self.config.max_distance / self.config.near).ln();
            let df = d_linear * (1.0 - self.config.depth_distribution) + d_exp * self.config.depth_distribution;
            if df.abs() > 1e-7 { t -= (val - clamped) / df; }
            t = t.clamp(0.0, 1.0);
        }
        t * self.config.depth_slices as f32
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let froxel_size = std::mem::size_of::<FroxelData>();
        let total_froxels = self.config.total_froxels();
        total_froxels * froxel_size * 3 // froxels + integrated + history
    }

    /// Get statistics about the volumetric lighting.
    pub fn stats(&self) -> VolumetricStats {
        let total = self.config.total_froxels();
        let mut non_empty = 0;
        let mut max_density = 0.0_f32;
        let mut total_density = 0.0_f32;

        for f in &self.froxels {
            if f.density > 1e-6 {
                non_empty += 1;
                max_density = max_density.max(f.density);
                total_density += f.density;
            }
        }

        VolumetricStats {
            total_froxels: total,
            non_empty_froxels: non_empty,
            max_density,
            avg_density: if non_empty > 0 { total_density / non_empty as f32 } else { 0.0 },
            num_lights: self.lights.len(),
            memory_bytes: self.memory_usage(),
        }
    }
}

/// Clamp a value to a neighborhood of the target to prevent temporal ghosting.
fn clamp_to_neighborhood(history: Vec3, current: Vec3, tolerance: f32) -> Vec3 {
    Vec3::new(
        history.x.clamp(current.x - tolerance, current.x + tolerance),
        history.y.clamp(current.y - tolerance, current.y + tolerance),
        history.z.clamp(current.z - tolerance, current.z + tolerance),
    )
}

#[derive(Debug, Clone)]
pub struct VolumetricStats {
    pub total_froxels: usize,
    pub non_empty_froxels: usize,
    pub max_density: f32,
    pub avg_density: f32,
    pub num_lights: usize,
    pub memory_bytes: usize,
}

// ---------------------------------------------------------------------------
// God rays (screen-space light shafts as an alternative / complement)
// ---------------------------------------------------------------------------

/// Screen-space god rays using radial blur from a light source.
pub struct GodRayRenderer {
    pub light_screen_pos: [f32; 2],
    pub num_samples: u32,
    pub density: f32,
    pub weight: f32,
    pub decay: f32,
    pub exposure: f32,
}

impl Default for GodRayRenderer {
    fn default() -> Self {
        Self {
            light_screen_pos: [0.5, 0.3],
            num_samples: 64,
            density: 1.0,
            weight: 0.01,
            decay: 0.97,
            exposure: 1.0,
        }
    }
}

impl GodRayRenderer {
    /// Generate god ray samples for a pixel at the given UV coordinate.
    pub fn sample(&self, pixel_uv: [f32; 2], occlusion_sampler: &dyn Fn(f32, f32) -> f32) -> f32 {
        let delta_x = (pixel_uv[0] - self.light_screen_pos[0]) * self.density / self.num_samples as f32;
        let delta_y = (pixel_uv[1] - self.light_screen_pos[1]) * self.density / self.num_samples as f32;

        let mut uv_x = pixel_uv[0];
        let mut uv_y = pixel_uv[1];
        let mut illumination_decay = 1.0_f32;
        let mut result = 0.0_f32;

        for _ in 0..self.num_samples {
            uv_x -= delta_x;
            uv_y -= delta_y;

            // Sample occlusion (0 = occluded, 1 = light visible)
            let occlusion = occlusion_sampler(uv_x.clamp(0.0, 1.0), uv_y.clamp(0.0, 1.0));

            result += occlusion * illumination_decay * self.weight;
            illumination_decay *= self.decay;
        }

        result * self.exposure
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_isotropic() {
        let p = PhaseFunction::Isotropic;
        let val = p.evaluate(0.0);
        assert!((val - 1.0 / (4.0 * PI)).abs() < 1e-6);
    }

    #[test]
    fn test_phase_hg_forward() {
        let p = PhaseFunction::HenyeyGreenstein { g: 0.8 };
        // Forward scattering should be strongest at cos_theta = 1
        let fwd = p.evaluate(1.0);
        let back = p.evaluate(-1.0);
        assert!(fwd > back);
    }

    #[test]
    fn test_froxel_config() {
        let config = FroxelGridConfig::default();
        assert!(config.total_froxels() > 0);

        // Near depth should map to slice 0
        let s = config.depth_to_slice(config.near);
        assert_eq!(s, 0);
    }

    #[test]
    fn test_slice_depth_roundtrip() {
        let config = FroxelGridConfig::default();
        for i in 0..config.depth_slices {
            let depth = config.slice_to_depth(i);
            let slice = config.depth_to_slice(depth);
            assert!((slice as i32 - i as i32).abs() <= 1);
        }
    }

    #[test]
    fn test_volume_system_creation() {
        let config = FroxelGridConfig {
            tiles_x: 4,
            tiles_y: 4,
            depth_slices: 4,
            ..Default::default()
        };
        let sys = VolumetricLightingSystem::new(config);
        assert_eq!(sys.froxels.len(), 64);
    }

    #[test]
    fn test_height_fog() {
        let hf = HeightFogParams {
            base_height: 0.0,
            falloff: 1.0,
            max_density: 1.0,
        };
        assert!((hf.density_at_height(-1.0) - 1.0).abs() < 1e-6);
        assert!(hf.density_at_height(5.0) < 0.01);
    }

    #[test]
    fn test_god_rays() {
        let gr = GodRayRenderer::default();
        let result = gr.sample([0.5, 0.5], &|_, _| 1.0);
        assert!(result > 0.0);
    }
}
