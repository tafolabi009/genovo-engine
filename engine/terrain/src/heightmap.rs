//! Heightmap storage, sampling, and procedural generation.
//!
//! The [`Heightmap`] struct is the core data representation for terrain
//! elevation. It stores a 2-D grid of `f32` heights and provides bilinear
//! sampling, normal computation, smoothing, and several procedural generation
//! algorithms including diamond-square, fault-line, and fractal noise.

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::{TerrainError, TerrainResult};

// ---------------------------------------------------------------------------
// Heightmap
// ---------------------------------------------------------------------------

/// A 2-D grid of floating-point height values representing terrain elevation.
///
/// Heights are stored in row-major order where `index = z * width + x`.
/// The coordinate system follows the engine convention:
/// - X axis = east
/// - Y axis = up (height)
/// - Z axis = north (into the screen)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heightmap {
    /// Number of samples along the X axis.
    width: u32,
    /// Number of samples along the Z axis.
    height: u32,
    /// Row-major height data. Length = width * height.
    heights: Vec<f32>,
    /// Cached minimum height value.
    min_height: f32,
    /// Cached maximum height value.
    max_height: f32,
}

impl Heightmap {
    // -- Constructors -------------------------------------------------------

    /// Creates a heightmap from raw float data.
    ///
    /// # Errors
    ///
    /// Returns [`TerrainError::InvalidDimensions`] if either dimension is zero,
    /// or [`TerrainError::DataLengthMismatch`] if `data.len() != width * height`.
    pub fn from_raw(width: u32, height: u32, data: Vec<f32>) -> TerrainResult<Self> {
        if width == 0 || height == 0 {
            return Err(TerrainError::InvalidDimensions { width, height });
        }
        let expected = (width as usize) * (height as usize);
        if data.len() != expected {
            return Err(TerrainError::DataLengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        let (min_height, max_height) = Self::compute_min_max(&data);
        Ok(Self {
            width,
            height,
            heights: data,
            min_height,
            max_height,
        })
    }

    /// Creates a flat heightmap filled with `initial_height`.
    pub fn new_flat(width: u32, height: u32, initial_height: f32) -> TerrainResult<Self> {
        if width == 0 || height == 0 {
            return Err(TerrainError::InvalidDimensions { width, height });
        }
        let count = (width as usize) * (height as usize);
        Ok(Self {
            width,
            height,
            heights: vec![initial_height; count],
            min_height: initial_height,
            max_height: initial_height,
        })
    }

    /// Parses a heightmap from raw 16-bit unsigned integer image bytes.
    ///
    /// The `bytes` slice is interpreted as a flat array of big-endian `u16`
    /// values (the common RAW heightmap format). The caller must provide the
    /// dimensions because RAW files do not contain header information.
    ///
    /// Heights are normalized to `[0.0, 1.0]` by dividing by `u16::MAX`.
    pub fn from_image_bytes(
        width: u32,
        height: u32,
        bytes: &[u8],
    ) -> TerrainResult<Self> {
        if width == 0 || height == 0 {
            return Err(TerrainError::InvalidDimensions { width, height });
        }
        let expected_len = (width as usize) * (height as usize) * 2;
        if bytes.len() < expected_len {
            return Err(TerrainError::DataLengthMismatch {
                expected: expected_len,
                actual: bytes.len(),
            });
        }

        let count = (width as usize) * (height as usize);
        let mut data = Vec::with_capacity(count);
        let inv_max = 1.0 / (u16::MAX as f32);

        for i in 0..count {
            let hi = bytes[i * 2] as u16;
            let lo = bytes[i * 2 + 1] as u16;
            let value = (hi << 8) | lo;
            data.push(value as f32 * inv_max);
        }

        let (min_height, max_height) = Self::compute_min_max(&data);
        Ok(Self {
            width,
            height,
            heights: data,
            min_height,
            max_height,
        })
    }

    // -- Accessors ----------------------------------------------------------

    /// Returns the width (number of samples along X).
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the height (number of samples along Z).
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the cached minimum height.
    #[inline]
    pub fn min_height(&self) -> f32 {
        self.min_height
    }

    /// Returns the cached maximum height.
    #[inline]
    pub fn max_height(&self) -> f32 {
        self.max_height
    }

    /// Returns a reference to the raw height data.
    #[inline]
    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    /// Returns a mutable reference to the raw height data.
    ///
    /// The caller **must** call [`recalculate_bounds`](Self::recalculate_bounds)
    /// after modifying data directly.
    #[inline]
    pub fn heights_mut(&mut self) -> &mut [f32] {
        &mut self.heights
    }

    /// Returns the height at integer grid coordinates.
    ///
    /// Out-of-bounds coordinates are clamped to the nearest edge.
    #[inline]
    pub fn get(&self, x: u32, z: u32) -> f32 {
        let x = x.min(self.width - 1);
        let z = z.min(self.height - 1);
        self.heights[(z as usize) * (self.width as usize) + (x as usize)]
    }

    /// Sets the height at integer grid coordinates.
    ///
    /// Clamped to bounds. Updates cached min/max.
    #[inline]
    pub fn set_height(&mut self, x: u32, z: u32, value: f32) {
        if x >= self.width || z >= self.height {
            return;
        }
        let idx = (z as usize) * (self.width as usize) + (x as usize);
        self.heights[idx] = value;
        if value < self.min_height {
            self.min_height = value;
        }
        if value > self.max_height {
            self.max_height = value;
        }
    }

    /// Recalculates cached min/max bounds from the height data.
    pub fn recalculate_bounds(&mut self) {
        let (min_h, max_h) = Self::compute_min_max(&self.heights);
        self.min_height = min_h;
        self.max_height = max_h;
    }

    // -- Sampling -----------------------------------------------------------

    /// Bilinear interpolation at a fractional position `(x, z)`.
    ///
    /// Coordinates are in heightmap space (i.e. `0..width-1` and `0..height-1`).
    /// Out-of-bounds coordinates are clamped.
    pub fn sample(&self, x: f32, z: f32) -> f32 {
        let max_x = (self.width - 1) as f32;
        let max_z = (self.height - 1) as f32;
        let x = x.clamp(0.0, max_x);
        let z = z.clamp(0.0, max_z);

        let x0 = x.floor() as u32;
        let z0 = z.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let z1 = (z0 + 1).min(self.height - 1);

        let fx = x - x0 as f32;
        let fz = z - z0 as f32;

        let h00 = self.get(x0, z0);
        let h10 = self.get(x1, z0);
        let h01 = self.get(x0, z1);
        let h11 = self.get(x1, z1);

        // Bilinear blend
        let h0 = h00 + (h10 - h00) * fx;
        let h1 = h01 + (h11 - h01) * fx;
        h0 + (h1 - h0) * fz
    }

    /// Computes the surface normal at fractional position `(x, z)` using
    /// finite differences.
    ///
    /// The `scale` parameter controls the horizontal spacing between samples
    /// (world units per heightmap cell). `height_scale` controls vertical
    /// exaggeration.
    pub fn normal_at(&self, x: f32, z: f32) -> Vec3 {
        self.normal_at_scaled(x, z, 1.0, 1.0)
    }

    /// Computes the surface normal with explicit horizontal and vertical scale.
    pub fn normal_at_scaled(
        &self,
        x: f32,
        z: f32,
        cell_size: f32,
        height_scale: f32,
    ) -> Vec3 {
        let eps = 1.0;
        let hx0 = self.sample(x - eps, z) * height_scale;
        let hx1 = self.sample(x + eps, z) * height_scale;
        let hz0 = self.sample(x, z - eps) * height_scale;
        let hz1 = self.sample(x, z + eps) * height_scale;

        let dx = (hx1 - hx0) / (2.0 * eps * cell_size);
        let dz = (hz1 - hz0) / (2.0 * eps * cell_size);

        Vec3::new(-dx, 1.0, -dz).normalize()
    }

    /// Returns the slope angle in radians at position `(x, z)`.
    pub fn slope_at(&self, x: f32, z: f32) -> f32 {
        let n = self.normal_at(x, z);
        n.y.acos()
    }

    // -- Modification -------------------------------------------------------

    /// Gaussian smoothing of the heightmap.
    ///
    /// `radius` is the kernel half-size (in cells). `iterations` controls
    /// how many smoothing passes are applied.
    pub fn smooth(&mut self, radius: usize, iterations: usize) {
        if radius == 0 || iterations == 0 {
            return;
        }

        let w = self.width as usize;
        let h = self.height as usize;

        // Build 1-D Gaussian kernel
        let kernel = Self::build_gaussian_kernel(radius);
        let klen = kernel.len();

        for _iter in 0..iterations {
            // Horizontal pass
            let mut temp = vec![0.0f32; w * h];
            for z in 0..h {
                for x in 0..w {
                    let mut sum = 0.0f32;
                    let mut weight_sum = 0.0f32;
                    for k in 0..klen {
                        let offset = k as isize - radius as isize;
                        let sx = (x as isize + offset).clamp(0, (w - 1) as isize) as usize;
                        let weight = kernel[k];
                        sum += self.heights[z * w + sx] * weight;
                        weight_sum += weight;
                    }
                    temp[z * w + x] = sum / weight_sum;
                }
            }

            // Vertical pass
            for z in 0..h {
                for x in 0..w {
                    let mut sum = 0.0f32;
                    let mut weight_sum = 0.0f32;
                    for k in 0..klen {
                        let offset = k as isize - radius as isize;
                        let sz = (z as isize + offset).clamp(0, (h - 1) as isize) as usize;
                        let weight = kernel[k];
                        sum += temp[sz * w + x] * weight;
                        weight_sum += weight;
                    }
                    self.heights[z * w + x] = sum / weight_sum;
                }
            }
        }

        self.recalculate_bounds();
    }

    /// Normalizes all heights to the range `[0.0, 1.0]`.
    pub fn normalize(&mut self) {
        let range = self.max_height - self.min_height;
        if range.abs() < f32::EPSILON {
            return;
        }
        let inv_range = 1.0 / range;
        let min = self.min_height;
        for h in &mut self.heights {
            *h = (*h - min) * inv_range;
        }
        self.min_height = 0.0;
        self.max_height = 1.0;
    }

    /// Rescales heights to a given range `[new_min, new_max]`.
    pub fn rescale(&mut self, new_min: f32, new_max: f32) {
        let old_range = self.max_height - self.min_height;
        if old_range.abs() < f32::EPSILON {
            for h in &mut self.heights {
                *h = new_min;
            }
            self.min_height = new_min;
            self.max_height = new_min;
            return;
        }
        let new_range = new_max - new_min;
        let old_min = self.min_height;
        let inv_old = 1.0 / old_range;
        for h in &mut self.heights {
            *h = ((*h - old_min) * inv_old) * new_range + new_min;
        }
        self.min_height = new_min;
        self.max_height = new_max;
    }

    // -- Procedural generation ----------------------------------------------

    /// Generates a procedural heightmap using the diamond-square algorithm.
    ///
    /// `size` must be a power of two plus one (e.g. 129, 257, 513, 1025).
    /// `roughness` controls how quickly detail amplitude decreases (0.0..1.0).
    /// `seed` is used to initialize the PRNG.
    pub fn generate_procedural(size: u32, roughness: f32, seed: u64) -> TerrainResult<Self> {
        // Validate that size is (2^n + 1)
        let inner = size - 1;
        if size < 3 || (inner & (inner - 1)) != 0 {
            return Err(TerrainError::InvalidDimensions {
                width: size,
                height: size,
            });
        }

        let n = size as usize;
        let mut data = vec![0.0f32; n * n];
        let mut rng = SimpleRng::new(seed);

        // Seed corners
        data[0] = rng.next_f32() * 2.0 - 1.0;
        data[n - 1] = rng.next_f32() * 2.0 - 1.0;
        data[(n - 1) * n] = rng.next_f32() * 2.0 - 1.0;
        data[(n - 1) * n + (n - 1)] = rng.next_f32() * 2.0 - 1.0;

        let mut step_size = n - 1;
        let mut scale = 1.0f32;

        while step_size > 1 {
            let half = step_size / 2;

            // Diamond step: compute center of each square
            {
                let mut y = 0;
                while y < n - 1 {
                    let mut x = 0;
                    while x < n - 1 {
                        let tl = data[y * n + x];
                        let tr = data[y * n + x + step_size];
                        let bl = data[(y + step_size) * n + x];
                        let br = data[(y + step_size) * n + x + step_size];
                        let avg = (tl + tr + bl + br) * 0.25;
                        data[(y + half) * n + (x + half)] =
                            avg + (rng.next_f32() * 2.0 - 1.0) * scale;
                        x += step_size;
                    }
                    y += step_size;
                }
            }

            // Square step: compute midpoints of each diamond edge
            {
                let mut y = 0;
                while y < n {
                    // Offset every other row
                    let x_start = if (y / half) % 2 == 0 { half } else { 0 };
                    let mut x = x_start;
                    while x < n {
                        let mut sum = 0.0f32;
                        let mut count = 0u32;

                        if y >= half {
                            sum += data[(y - half) * n + x];
                            count += 1;
                        }
                        if y + half < n {
                            sum += data[(y + half) * n + x];
                            count += 1;
                        }
                        if x >= half {
                            sum += data[y * n + (x - half)];
                            count += 1;
                        }
                        if x + half < n {
                            sum += data[y * n + (x + half)];
                            count += 1;
                        }

                        if count > 0 {
                            let avg = sum / count as f32;
                            data[y * n + x] = avg + (rng.next_f32() * 2.0 - 1.0) * scale;
                        }

                        x += step_size;
                    }
                    y += half;
                }
            }

            step_size = half;
            scale *= roughness;
        }

        let (min_h, max_h) = Self::compute_min_max(&data);
        let mut hm = Self {
            width: size,
            height: size,
            heights: data,
            min_height: min_h,
            max_height: max_h,
        };
        hm.normalize();
        Ok(hm)
    }

    /// Generates terrain using the fault-line algorithm.
    ///
    /// Creates `num_faults` random lines across the terrain. Points on one side
    /// are raised, points on the other side are lowered. The displacement
    /// decreases with each iteration to produce finer detail.
    pub fn generate_fault_line(
        width: u32,
        height: u32,
        num_faults: u32,
        displacement: f32,
        seed: u64,
    ) -> TerrainResult<Self> {
        if width == 0 || height == 0 {
            return Err(TerrainError::InvalidDimensions { width, height });
        }

        let w = width as usize;
        let h = height as usize;
        let mut data = vec![0.0f32; w * h];
        let mut rng = SimpleRng::new(seed);

        for fault_idx in 0..num_faults {
            // Random point on the map
            let px = rng.next_f32() * w as f32;
            let pz = rng.next_f32() * h as f32;

            // Random direction for the fault line
            let angle = rng.next_f32() * std::f32::consts::TAU;
            let dx = angle.cos();
            let dz = angle.sin();

            // Displacement decreases over iterations
            let current_disp = displacement * (1.0 - fault_idx as f32 / num_faults as f32);

            for z in 0..h {
                for x in 0..w {
                    // Determine which side of the line this point is on
                    let vx = x as f32 - px;
                    let vz = z as f32 - pz;
                    let cross = vx * dz - vz * dx;

                    if cross > 0.0 {
                        data[z * w + x] += current_disp;
                    } else {
                        data[z * w + x] -= current_disp;
                    }
                }
            }
        }

        let mut hm = Self {
            width,
            height,
            heights: data,
            min_height: 0.0,
            max_height: 0.0,
        };
        hm.recalculate_bounds();
        hm.normalize();
        Ok(hm)
    }

    /// Generates terrain using value noise with multiple octaves.
    ///
    /// Produces smoother results than diamond-square with controllable
    /// frequency and amplitude per octave.
    pub fn generate_noise(
        width: u32,
        height: u32,
        params: &NoiseParams,
        seed: u64,
    ) -> TerrainResult<Self> {
        if width == 0 || height == 0 {
            return Err(TerrainError::InvalidDimensions { width, height });
        }

        let w = width as usize;
        let h = height as usize;
        let mut data = vec![0.0f32; w * h];

        // Generate a permutation table for coherent noise
        let perm = Self::generate_permutation_table(seed);

        for z in 0..h {
            for x in 0..w {
                let mut amplitude = params.amplitude;
                let mut frequency = params.frequency;
                let mut value = 0.0f32;
                let mut max_amp = 0.0f32;

                for _octave in 0..params.octaves {
                    let nx = x as f32 * frequency / w as f32;
                    let nz = z as f32 * frequency / h as f32;

                    let noise_val = match params.noise_type {
                        NoiseType::Value => Self::value_noise_2d(nx, nz, &perm),
                        NoiseType::Ridged => {
                            let v = Self::value_noise_2d(nx, nz, &perm);
                            1.0 - (v * 2.0 - 1.0).abs()
                        }
                        NoiseType::Billow => {
                            let v = Self::value_noise_2d(nx, nz, &perm);
                            (v * 2.0 - 1.0).abs()
                        }
                        NoiseType::Terraced => {
                            let v = Self::value_noise_2d(nx, nz, &perm);
                            let steps = params.terrace_steps.max(2) as f32;
                            (v * steps).floor() / steps
                        }
                    };

                    value += noise_val * amplitude;
                    max_amp += amplitude;
                    amplitude *= params.persistence;
                    frequency *= params.lacunarity;
                }

                data[z * w + x] = value / max_amp;
            }
        }

        let mut hm = Self {
            width,
            height,
            heights: data,
            min_height: 0.0,
            max_height: 0.0,
        };
        hm.recalculate_bounds();
        hm.normalize();
        Ok(hm)
    }

    // -- Thermal erosion (basic, in-heightmap) ------------------------------

    /// Simple thermal erosion: material flows downhill when slope exceeds
    /// the talus angle.
    ///
    /// `talus` is the maximum stable slope angle (tangent). `strength`
    /// controls how much material moves per iteration.
    pub fn thermal_erode(
        &mut self,
        iterations: usize,
        talus: f32,
        strength: f32,
    ) {
        let w = self.width as usize;
        let h = self.height as usize;

        let offsets: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

        for _iter in 0..iterations {
            let snapshot = self.heights.clone();

            for z in 1..(h - 1) {
                for x in 1..(w - 1) {
                    let idx = z * w + x;
                    let center = snapshot[idx];

                    let mut max_diff = 0.0f32;
                    let mut total_diff = 0.0f32;
                    let mut diffs = [0.0f32; 4];

                    for (i, &(dx, dz)) in offsets.iter().enumerate() {
                        let nx = (x as isize + dx) as usize;
                        let nz = (z as isize + dz) as usize;
                        let neighbor = snapshot[nz * w + nx];
                        let diff = center - neighbor;
                        if diff > talus {
                            diffs[i] = diff;
                            total_diff += diff;
                            if diff > max_diff {
                                max_diff = diff;
                            }
                        }
                    }

                    if total_diff > 0.0 {
                        let amount = strength * (max_diff - talus) * 0.5;
                        self.heights[idx] -= amount;

                        for (i, &(dx, dz)) in offsets.iter().enumerate() {
                            if diffs[i] > 0.0 {
                                let nx = (x as isize + dx) as usize;
                                let nz = (z as isize + dz) as usize;
                                let proportion = diffs[i] / total_diff;
                                self.heights[nz * w + nx] += amount * proportion;
                            }
                        }
                    }
                }
            }
        }

        self.recalculate_bounds();
    }

    /// Simple hydraulic erosion on the heightmap: spawns raindrops that flow
    /// downhill, picking up and depositing sediment.
    ///
    /// This is a simplified version; for the full simulation with configurable
    /// parameters, see [`crate::erosion::HydraulicErosion`].
    pub fn hydraulic_erode(
        &mut self,
        iterations: usize,
        erosion_rate: f32,
        deposition_rate: f32,
        evaporation_rate: f32,
        seed: u64,
    ) {
        let w = self.width as usize;
        let h = self.height as usize;
        let mut rng = SimpleRng::new(seed);

        for _drop in 0..iterations {
            // Spawn droplet at random position
            let mut px = rng.next_f32() * (w - 2) as f32 + 1.0;
            let mut pz = rng.next_f32() * (h - 2) as f32 + 1.0;
            let mut vel_x = 0.0f32;
            let mut vel_z = 0.0f32;
            let mut speed = 0.0f32;
            let mut water = 1.0f32;
            let mut sediment = 0.0f32;

            let max_steps = 64;

            for _step in 0..max_steps {
                let ix = px.floor() as usize;
                let iz = pz.floor() as usize;

                if ix < 1 || ix >= w - 1 || iz < 1 || iz >= h - 1 {
                    break;
                }

                // Compute gradient using finite differences
                let idx = iz * w + ix;
                let grad_x = self.heights[idx + 1] - self.heights[idx.saturating_sub(1)];
                let grad_z = self.heights[idx + w] - self.heights[idx.saturating_sub(w)];

                // Update velocity with gradient
                let inertia = 0.3;
                vel_x = vel_x * inertia - grad_x * (1.0 - inertia);
                vel_z = vel_z * inertia - grad_z * (1.0 - inertia);

                let vel_len = (vel_x * vel_x + vel_z * vel_z).sqrt();
                if vel_len < 1e-6 {
                    break;
                }

                vel_x /= vel_len;
                vel_z /= vel_len;
                speed = vel_len;

                // Move droplet
                let new_px = px + vel_x;
                let new_pz = pz + vel_z;

                if new_px < 1.0
                    || new_px >= (w - 2) as f32
                    || new_pz < 1.0
                    || new_pz >= (h - 2) as f32
                {
                    break;
                }

                let old_height = self.sample(px, pz);
                let new_height = self.sample(new_px, new_pz);
                let height_diff = new_height - old_height;

                // Carrying capacity depends on speed and slope
                let capacity = (-height_diff).max(0.01) * speed * water * 4.0;

                if sediment > capacity || height_diff > 0.0 {
                    // Deposit sediment
                    let deposit_amount = if height_diff > 0.0 {
                        sediment.min(height_diff)
                    } else {
                        (sediment - capacity) * deposition_rate
                    };
                    sediment -= deposit_amount;
                    self.heights[idx] += deposit_amount;
                } else {
                    // Erode terrain
                    let erode_amount =
                        ((capacity - sediment) * erosion_rate).min(-height_diff);
                    sediment += erode_amount;
                    self.heights[idx] -= erode_amount;
                }

                px = new_px;
                pz = new_pz;
                water *= 1.0 - evaporation_rate;

                if water < 0.01 {
                    break;
                }
            }
        }

        self.recalculate_bounds();
    }

    /// Returns the area (in cells squared) of the heightmap.
    #[inline]
    pub fn area(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    /// Extracts a sub-region of the heightmap.
    ///
    /// `start_x`, `start_z` are the top-left corner. `sub_w`, `sub_h` are
    /// the dimensions of the extracted region (clamped to bounds).
    pub fn extract_region(
        &self,
        start_x: u32,
        start_z: u32,
        sub_w: u32,
        sub_h: u32,
    ) -> TerrainResult<Self> {
        let clamped_w = sub_w.min(self.width.saturating_sub(start_x));
        let clamped_h = sub_h.min(self.height.saturating_sub(start_z));

        if clamped_w == 0 || clamped_h == 0 {
            return Err(TerrainError::InvalidDimensions {
                width: clamped_w,
                height: clamped_h,
            });
        }

        let mut data = Vec::with_capacity((clamped_w as usize) * (clamped_h as usize));
        for z in 0..clamped_h {
            for x in 0..clamped_w {
                data.push(self.get(start_x + x, start_z + z));
            }
        }

        Self::from_raw(clamped_w, clamped_h, data)
    }

    /// Downsamples the heightmap by the given factor (2 = half resolution).
    pub fn downsample(&self, factor: u32) -> TerrainResult<Self> {
        let factor = factor.max(1);
        let new_w = (self.width + factor - 1) / factor;
        let new_h = (self.height + factor - 1) / factor;

        if new_w == 0 || new_h == 0 {
            return Err(TerrainError::InvalidDimensions {
                width: new_w,
                height: new_h,
            });
        }

        let mut data = Vec::with_capacity((new_w as usize) * (new_h as usize));
        for z in 0..new_h {
            for x in 0..new_w {
                let sx = (x * factor) as f32;
                let sz = (z * factor) as f32;
                data.push(self.sample(sx, sz));
            }
        }

        Self::from_raw(new_w, new_h, data)
    }

    // -- Private helpers ----------------------------------------------------

    fn compute_min_max(data: &[f32]) -> (f32, f32) {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in data {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        (min, max)
    }

    fn build_gaussian_kernel(radius: usize) -> Vec<f32> {
        let sigma = radius as f32 / 3.0;
        let sigma2 = 2.0 * sigma * sigma;
        let mut kernel = Vec::with_capacity(2 * radius + 1);
        for i in 0..=(2 * radius) {
            let x = i as f32 - radius as f32;
            kernel.push((-x * x / sigma2).exp());
        }
        kernel
    }

    fn generate_permutation_table(seed: u64) -> Vec<u8> {
        let mut rng = SimpleRng::new(seed);
        let mut table: Vec<u8> = (0..=255).collect();
        // Fisher-Yates shuffle
        for i in (1..256).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            table.swap(i, j);
        }
        // Double the table for overflow avoidance
        let mut doubled = table.clone();
        doubled.extend_from_slice(&table);
        doubled
    }

    fn value_noise_2d(x: f32, z: f32, perm: &[u8]) -> f32 {
        let x0 = x.floor() as i32;
        let z0 = z.floor() as i32;
        let x1 = x0 + 1;
        let z1 = z0 + 1;

        let fx = x - x0 as f32;
        let fz = z - z0 as f32;

        // Smoothstep for interpolation
        let sx = fx * fx * (3.0 - 2.0 * fx);
        let sz = fz * fz * (3.0 - 2.0 * fz);

        let hash = |ix: i32, iz: i32| -> f32 {
            let xi = (ix & 255) as usize;
            let zi = (iz & 255) as usize;
            let idx = (perm[xi] as usize + zi) & 511;
            perm[idx] as f32 / 255.0
        };

        let v00 = hash(x0, z0);
        let v10 = hash(x1, z0);
        let v01 = hash(x0, z1);
        let v11 = hash(x1, z1);

        let a = v00 + (v10 - v00) * sx;
        let b = v01 + (v11 - v01) * sx;
        a + (b - a) * sz
    }
}

// ---------------------------------------------------------------------------
// Noise parameters
// ---------------------------------------------------------------------------

/// Configuration for noise-based terrain generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseParams {
    /// Number of noise octaves to layer.
    pub octaves: u32,
    /// Initial amplitude of the first octave.
    pub amplitude: f32,
    /// Initial frequency of the first octave.
    pub frequency: f32,
    /// Amplitude multiplier per octave (typically 0.4..0.6).
    pub persistence: f32,
    /// Frequency multiplier per octave (typically 2.0).
    pub lacunarity: f32,
    /// Type of noise function to use.
    pub noise_type: NoiseType,
    /// Number of terracing steps (only used with `NoiseType::Terraced`).
    pub terrace_steps: u32,
}

impl Default for NoiseParams {
    fn default() -> Self {
        Self {
            octaves: 6,
            amplitude: 1.0,
            frequency: 4.0,
            persistence: 0.5,
            lacunarity: 2.0,
            noise_type: NoiseType::Value,
            terrace_steps: 8,
        }
    }
}

/// The type of noise function used for terrain generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseType {
    /// Standard value noise.
    Value,
    /// Ridged multifractal (absolute value inverted). Creates sharp ridges.
    Ridged,
    /// Billow noise (absolute value). Creates puffy, cloud-like terrain.
    Billow,
    /// Terraced noise (quantized to discrete levels). Creates mesa/plateau effects.
    Terraced,
}

// ---------------------------------------------------------------------------
// Simple deterministic PRNG
// ---------------------------------------------------------------------------

/// A simple xoshiro256** PRNG for deterministic terrain generation.
///
/// We avoid pulling in the `rand` crate to keep dependency count low.
/// This is not cryptographically secure but provides good distribution
/// for terrain algorithms.
#[derive(Debug, Clone)]
pub(crate) struct SimpleRng {
    state: [u64; 4],
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        // SplitMix64 seeding to fill state from a single seed
        let mut s = seed;
        let mut state = [0u64; 4];
        for slot in &mut state {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            *slot = z ^ (z >> 31);
        }
        Self { state }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = (self.state[1].wrapping_mul(5))
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];

        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    /// Returns a float in `[0.0, 1.0)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Returns a float in `[-1.0, 1.0)`.
    pub fn next_f32_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }

    /// Returns a u32 in `[0, max)`.
    pub fn next_u32_range(&mut self, max: u32) -> u32 {
        (self.next_u64() % max as u64) as u32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_heightmap() {
        let hm = Heightmap::new_flat(16, 16, 5.0).unwrap();
        assert_eq!(hm.width(), 16);
        assert_eq!(hm.height(), 16);
        assert_eq!(hm.get(0, 0), 5.0);
        assert_eq!(hm.sample(7.5, 7.5), 5.0);
    }

    #[test]
    fn bilinear_sampling() {
        let data = vec![0.0, 1.0, 0.0, 1.0];
        let hm = Heightmap::from_raw(2, 2, data).unwrap();
        let mid = hm.sample(0.5, 0.5);
        assert!((mid - 0.5).abs() < 1e-5);
    }

    #[test]
    fn diamond_square() {
        let hm = Heightmap::generate_procedural(65, 0.5, 42).unwrap();
        assert_eq!(hm.width(), 65);
        assert_eq!(hm.height(), 65);
        assert!(hm.min_height() >= 0.0);
        assert!(hm.max_height() <= 1.0);
    }

    #[test]
    fn fault_line() {
        let hm = Heightmap::generate_fault_line(64, 64, 100, 0.5, 123).unwrap();
        assert_eq!(hm.width(), 64);
        assert!((hm.min_height() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn smoothing() {
        let mut hm = Heightmap::generate_procedural(33, 0.6, 99).unwrap();
        let before_range = hm.max_height() - hm.min_height();
        hm.smooth(2, 3);
        let after_range = hm.max_height() - hm.min_height();
        // Smoothing should reduce the height range
        assert!(after_range <= before_range + 0.01);
    }

    #[test]
    fn normals_point_up_on_flat() {
        let hm = Heightmap::new_flat(16, 16, 0.0).unwrap();
        let n = hm.normal_at(8.0, 8.0);
        assert!((n.y - 1.0).abs() < 1e-4);
    }

    #[test]
    fn extract_region_basic() {
        let hm = Heightmap::generate_procedural(33, 0.5, 1).unwrap();
        let sub = hm.extract_region(4, 4, 8, 8).unwrap();
        assert_eq!(sub.width(), 8);
        assert_eq!(sub.height(), 8);
    }
}
