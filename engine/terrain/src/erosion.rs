//! Terrain erosion and brush operations.
//!
//! Provides physically-inspired erosion simulations for creating realistic
//! terrain features:
//!
//! - **Hydraulic erosion** — water-based erosion using raindrop simulation.
//! - **Thermal erosion** — material slides down slopes exceeding the talus angle.
//! - **Wind erosion** — directional sand/soil movement.
//! - **Terrain brushes** — interactive painting operations.

use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::heightmap::{Heightmap, SimpleRng};

// ---------------------------------------------------------------------------
// HydraulicErosion
// ---------------------------------------------------------------------------

/// Full hydraulic erosion simulation using raindrop particle tracking.
///
/// Each droplet is spawned at a random position, flows downhill following
/// the terrain gradient, picks up sediment where the flow is fast and the
/// terrain is steep, and deposits sediment when it slows down or the carrying
/// capacity is exceeded.
///
/// # Algorithm
///
/// 1. Spawn a water droplet at a random position with zero velocity.
/// 2. Compute the terrain gradient at the droplet position.
/// 3. Update the velocity using the gradient (steepest descent) and inertia.
/// 4. Move the droplet along its velocity vector.
/// 5. Compute the carrying capacity based on speed, slope, and water volume.
/// 6. If sediment exceeds capacity, deposit the excess.
/// 7. If sediment is below capacity, erode the terrain (using an erosion
///    brush for spatial spread).
/// 8. Evaporate a fraction of the water.
/// 9. Repeat from step 2 until the droplet evaporates or exits the map.
///
/// For realistic results, 100K-200K+ iterations are recommended.
pub struct HydraulicErosion {
    /// Erosion parameters.
    pub settings: ErosionSettings,
    /// The erosion brush weights (pre-computed for each brush radius).
    brush_weights: Vec<Vec<f32>>,
    /// Brush index offsets (x, z) for each weight.
    brush_offsets: Vec<Vec<(i32, i32)>>,
}

/// Configurable parameters for hydraulic erosion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErosionSettings {
    /// Number of droplet iterations. Higher = more erosion. 200K+ for quality.
    pub iterations: usize,
    /// Maximum steps per droplet before it is killed.
    pub max_droplet_lifetime: u32,
    /// Inertia of the droplet (0..1). Higher = more momentum, less direction change.
    pub inertia: f32,
    /// Base sediment capacity multiplier.
    pub sediment_capacity_factor: f32,
    /// Minimum sediment capacity (prevents zero capacity on flat terrain).
    pub min_sediment_capacity: f32,
    /// Rate at which sediment is picked up.
    pub erosion_rate: f32,
    /// Rate at which sediment is deposited.
    pub deposition_rate: f32,
    /// Fraction of water that evaporates per step.
    pub evaporation_rate: f32,
    /// Gravity constant (affects droplet acceleration).
    pub gravity: f32,
    /// Starting water volume per droplet.
    pub initial_water: f32,
    /// Starting speed of each droplet.
    pub initial_speed: f32,
    /// Erosion brush radius in cells. Larger = smoother but slower.
    pub erosion_radius: u32,
    /// PRNG seed for reproducible results.
    pub seed: u64,
}

impl Default for ErosionSettings {
    fn default() -> Self {
        Self {
            iterations: 100_000,
            max_droplet_lifetime: 64,
            inertia: 0.05,
            sediment_capacity_factor: 4.0,
            min_sediment_capacity: 0.01,
            erosion_rate: 0.3,
            deposition_rate: 0.3,
            evaporation_rate: 0.01,
            gravity: 4.0,
            initial_water: 1.0,
            initial_speed: 1.0,
            erosion_radius: 3,
            seed: 42,
        }
    }
}

impl HydraulicErosion {
    /// Creates a new hydraulic erosion simulator with the given settings.
    pub fn new(settings: ErosionSettings) -> Self {
        let mut erosion = Self {
            settings,
            brush_weights: Vec::new(),
            brush_offsets: Vec::new(),
        };
        erosion.precompute_brush();
        erosion
    }

    /// Pre-computes the erosion brush: a set of weights defining how erosion
    /// is distributed spatially around the droplet position.
    fn precompute_brush(&mut self) {
        let radius = self.settings.erosion_radius as i32;
        let mut weights = Vec::new();
        let mut offsets = Vec::new();
        let mut weight_sum = 0.0f32;

        for dz in -radius..=radius {
            for dx in -radius..=radius {
                let dist = ((dx * dx + dz * dz) as f32).sqrt();
                if dist <= radius as f32 {
                    let weight = (1.0 - dist / radius as f32).max(0.0);
                    weights.push(weight);
                    offsets.push((dx, dz));
                    weight_sum += weight;
                }
            }
        }

        // Normalize weights
        if weight_sum > 0.0 {
            let inv = 1.0 / weight_sum;
            for w in &mut weights {
                *w *= inv;
            }
        }

        self.brush_weights = vec![weights];
        self.brush_offsets = vec![offsets];
    }

    /// Runs the hydraulic erosion simulation on the given heightmap.
    ///
    /// This modifies the heightmap in-place. For 200K+ iterations on a
    /// 1024x1024 heightmap, expect several seconds of processing time.
    #[profiling::function]
    pub fn erode(&self, heightmap: &mut Heightmap) {
        let w = heightmap.width() as usize;
        let h = heightmap.height() as usize;
        let heights = heightmap.heights_mut();

        let mut rng = SimpleRng::new(self.settings.seed);

        let brush_weights = &self.brush_weights[0];
        let brush_offsets = &self.brush_offsets[0];

        for _iter in 0..self.settings.iterations {
            // Spawn droplet at random position
            let mut pos_x = rng.next_f32() * (w - 2) as f32 + 1.0;
            let mut pos_z = rng.next_f32() * (h - 2) as f32 + 1.0;
            let mut dir_x = 0.0f32;
            let mut dir_z = 0.0f32;
            let mut speed = self.settings.initial_speed;
            let mut water = self.settings.initial_water;
            let mut sediment = 0.0f32;

            for _step in 0..self.settings.max_droplet_lifetime {
                let node_x = pos_x.floor() as usize;
                let node_z = pos_z.floor() as usize;

                // Bounds check with margin for gradient computation
                if node_x < 1 || node_x >= w - 2 || node_z < 1 || node_z >= h - 2 {
                    break;
                }

                // Bilinear interpolation offsets
                let offset_x = pos_x - node_x as f32;
                let offset_z = pos_z - node_z as f32;

                // Height at the four surrounding grid points
                let idx = node_z * w + node_x;
                let h_nw = heights[idx];
                let h_ne = heights[idx + 1];
                let h_sw = heights[idx + w];
                let h_se = heights[idx + w + 1];

                // Compute gradient using bilinear interpolation derivatives
                let grad_x = (h_ne - h_nw) * (1.0 - offset_z) + (h_se - h_sw) * offset_z;
                let grad_z = (h_sw - h_nw) * (1.0 - offset_x) + (h_se - h_ne) * offset_x;

                // Update direction with inertia
                dir_x = dir_x * self.settings.inertia - grad_x * (1.0 - self.settings.inertia);
                dir_z = dir_z * self.settings.inertia - grad_z * (1.0 - self.settings.inertia);

                // Normalize direction
                let dir_len = (dir_x * dir_x + dir_z * dir_z).sqrt();
                if dir_len < 1e-6 {
                    // Random direction if stuck on flat terrain
                    let angle = rng.next_f32() * std::f32::consts::TAU;
                    dir_x = angle.cos();
                    dir_z = angle.sin();
                } else {
                    dir_x /= dir_len;
                    dir_z /= dir_len;
                }

                // Move droplet
                let new_pos_x = pos_x + dir_x;
                let new_pos_z = pos_z + dir_z;

                // Bounds check for new position
                if new_pos_x < 1.0
                    || new_pos_x >= (w - 2) as f32
                    || new_pos_z < 1.0
                    || new_pos_z >= (h - 2) as f32
                {
                    break;
                }

                // Compute height at old and new positions
                let old_height = Self::interpolate_height(heights, w, pos_x, pos_z);
                let new_height = Self::interpolate_height(heights, w, new_pos_x, new_pos_z);
                let height_diff = new_height - old_height;

                // Compute carrying capacity
                let capacity = ((-height_diff).max(0.0)
                    * speed
                    * water
                    * self.settings.sediment_capacity_factor)
                    .max(self.settings.min_sediment_capacity);

                if sediment > capacity || height_diff > 0.0 {
                    // Deposit sediment
                    let deposit_amount = if height_diff > 0.0 {
                        // Deposit enough to fill the pit (but not more than
                        // what we carry)
                        sediment.min(height_diff)
                    } else {
                        (sediment - capacity) * self.settings.deposition_rate
                    };

                    sediment -= deposit_amount;

                    // Distribute deposit using bilinear weights at old position
                    Self::deposit_at(
                        heights,
                        w,
                        h,
                        pos_x,
                        pos_z,
                        deposit_amount,
                    );
                } else {
                    // Erode terrain
                    let erode_amount = ((capacity - sediment) * self.settings.erosion_rate)
                        .min(-height_diff);

                    // Apply erosion using brush pattern
                    Self::erode_at(
                        heights,
                        w,
                        h,
                        node_x,
                        node_z,
                        erode_amount,
                        brush_weights,
                        brush_offsets,
                    );

                    sediment += erode_amount;
                }

                // Update speed
                speed = (speed * speed + height_diff * self.settings.gravity)
                    .abs()
                    .sqrt();

                // Evaporate water
                water *= 1.0 - self.settings.evaporation_rate;

                // Move to new position
                pos_x = new_pos_x;
                pos_z = new_pos_z;

                if water < 0.001 {
                    break;
                }
            }
        }

        heightmap.recalculate_bounds();
    }

    /// Bilinear interpolation of height at a fractional position.
    #[inline]
    fn interpolate_height(heights: &[f32], w: usize, x: f32, z: f32) -> f32 {
        let ix = x.floor() as usize;
        let iz = z.floor() as usize;
        let fx = x - ix as f32;
        let fz = z - iz as f32;

        let idx = iz * w + ix;
        let h00 = heights[idx];
        let h10 = heights[idx + 1];
        let h01 = heights[idx + w];
        let h11 = heights[idx + w + 1];

        let a = h00 + (h10 - h00) * fx;
        let b = h01 + (h11 - h01) * fx;
        a + (b - a) * fz
    }

    /// Deposits sediment at a fractional position using bilinear weights.
    fn deposit_at(
        heights: &mut [f32],
        w: usize,
        h: usize,
        x: f32,
        z: f32,
        amount: f32,
    ) {
        let ix = x.floor() as usize;
        let iz = z.floor() as usize;

        if ix >= w - 1 || iz >= h - 1 {
            return;
        }

        let fx = x - ix as f32;
        let fz = z - iz as f32;

        let idx = iz * w + ix;
        heights[idx] += amount * (1.0 - fx) * (1.0 - fz);
        heights[idx + 1] += amount * fx * (1.0 - fz);
        heights[idx + w] += amount * (1.0 - fx) * fz;
        heights[idx + w + 1] += amount * fx * fz;
    }

    /// Erodes terrain at a grid position using the erosion brush.
    fn erode_at(
        heights: &mut [f32],
        w: usize,
        h: usize,
        cx: usize,
        cz: usize,
        amount: f32,
        weights: &[f32],
        offsets: &[(i32, i32)],
    ) {
        for (i, &(dx, dz)) in offsets.iter().enumerate() {
            let nx = cx as i32 + dx;
            let nz = cz as i32 + dz;

            if nx >= 0 && (nx as usize) < w && nz >= 0 && (nz as usize) < h {
                let idx = nz as usize * w + nx as usize;
                heights[idx] -= amount * weights[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ThermalErosion
// ---------------------------------------------------------------------------

/// Thermal erosion simulation: material slides downhill when the slope
/// exceeds the talus (angle of repose) threshold.
///
/// This produces scree fields and smoothed slopes.
pub struct ThermalErosion {
    /// Erosion parameters.
    pub settings: ThermalSettings,
}

/// Configuration for thermal erosion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalSettings {
    /// Number of erosion iterations.
    pub iterations: usize,
    /// Talus angle — maximum stable slope (as height difference per cell).
    /// Material above this threshold slides downhill.
    pub talus: f32,
    /// Fraction of excess material that moves per iteration (0..1).
    pub strength: f32,
    /// Border margin (cells) to leave untouched.
    pub border: usize,
}

impl Default for ThermalSettings {
    fn default() -> Self {
        Self {
            iterations: 50,
            talus: 0.01,
            strength: 0.5,
            border: 1,
        }
    }
}

impl ThermalErosion {
    /// Creates a new thermal erosion simulator.
    pub fn new(settings: ThermalSettings) -> Self {
        Self { settings }
    }

    /// Runs thermal erosion on the heightmap.
    #[profiling::function]
    pub fn erode(&self, heightmap: &mut Heightmap) {
        let w = heightmap.width() as usize;
        let h = heightmap.height() as usize;
        let border = self.settings.border;

        // 8-connected neighbors
        let offsets: [(isize, isize); 8] = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ];

        // Diagonal neighbors have sqrt(2) distance
        let distances: [f32; 8] = [
            std::f32::consts::SQRT_2,
            1.0,
            std::f32::consts::SQRT_2,
            1.0,
            1.0,
            std::f32::consts::SQRT_2,
            1.0,
            std::f32::consts::SQRT_2,
        ];

        for _iter in 0..self.settings.iterations {
            let snapshot = heightmap.heights().to_vec();

            for z in border..(h - border) {
                for x in border..(w - border) {
                    let idx = z * w + x;
                    let center_h = snapshot[idx];

                    let mut total_diff = 0.0f32;
                    let mut max_diff = 0.0f32;
                    let mut diffs = [0.0f32; 8];

                    for (i, &(dx, dz)) in offsets.iter().enumerate() {
                        let nx = (x as isize + dx) as usize;
                        let nz = (z as isize + dz) as usize;
                        let neighbor_h = snapshot[nz * w + nx];

                        // Slope = height difference / horizontal distance
                        let diff = (center_h - neighbor_h) / distances[i];

                        if diff > self.settings.talus {
                            diffs[i] = diff - self.settings.talus;
                            total_diff += diffs[i];
                            if diffs[i] > max_diff {
                                max_diff = diffs[i];
                            }
                        }
                    }

                    if total_diff > f32::EPSILON {
                        // Amount of material to move
                        let amount = self.settings.strength * max_diff * 0.5;
                        heightmap.heights_mut()[idx] -= amount;

                        // Distribute proportionally to slope excess
                        for (i, &(dx, dz)) in offsets.iter().enumerate() {
                            if diffs[i] > 0.0 {
                                let nx = (x as isize + dx) as usize;
                                let nz = (z as isize + dz) as usize;
                                let proportion = diffs[i] / total_diff;
                                heightmap.heights_mut()[nz * w + nx] +=
                                    amount * proportion;
                            }
                        }
                    }
                }
            }
        }

        heightmap.recalculate_bounds();
    }
}

// ---------------------------------------------------------------------------
// WindErosion
// ---------------------------------------------------------------------------

/// Wind erosion simulation: directional erosion that picks up loose material
/// and deposits it downwind, creating dune-like formations.
pub struct WindErosion {
    /// Erosion parameters.
    pub settings: WindSettings,
}

/// Configuration for wind erosion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindSettings {
    /// Number of iterations.
    pub iterations: usize,
    /// Wind direction (unit vector on XZ plane).
    pub direction: Vec2,
    /// Wind strength (affects how much material is picked up).
    pub strength: f32,
    /// Maximum amount of material the wind can carry.
    pub capacity: f32,
    /// Rate at which suspended material settles (deposition).
    pub deposition_rate: f32,
    /// Rate at which material is picked up (erosion).
    pub pickup_rate: f32,
    /// Random perturbation to wind direction.
    pub turbulence: f32,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for WindSettings {
    fn default() -> Self {
        Self {
            iterations: 50,
            direction: Vec2::new(1.0, 0.0),
            strength: 0.5,
            capacity: 0.1,
            deposition_rate: 0.3,
            pickup_rate: 0.2,
            turbulence: 0.1,
            seed: 42,
        }
    }
}

impl WindErosion {
    /// Creates a new wind erosion simulator.
    pub fn new(settings: WindSettings) -> Self {
        Self { settings }
    }

    /// Runs wind erosion on the heightmap.
    #[profiling::function]
    pub fn erode(&self, heightmap: &mut Heightmap) {
        let w = heightmap.width() as usize;
        let h = heightmap.height() as usize;
        let mut rng = SimpleRng::new(self.settings.seed);

        // Suspension map: how much material is currently suspended at each cell
        let mut suspension = vec![0.0f32; w * h];

        let dir = self.settings.direction.normalize();

        for _iter in 0..self.settings.iterations {
            let heights_snap = heightmap.heights().to_vec();

            // Process cells in wind direction order
            // Determine iteration order based on wind direction
            let x_range: Vec<usize> = if dir.x >= 0.0 {
                (1..w - 1).collect()
            } else {
                (1..w - 1).rev().collect()
            };
            let z_range: Vec<usize> = if dir.y >= 0.0 {
                (1..h - 1).collect()
            } else {
                (1..h - 1).rev().collect()
            };

            for &z in &z_range {
                for &x in &x_range {
                    let idx = z * w + x;
                    let center_h = heights_snap[idx];

                    // Apply turbulence
                    let turb_x = dir.x + (rng.next_f32() - 0.5) * self.settings.turbulence;
                    let turb_z = dir.y + (rng.next_f32() - 0.5) * self.settings.turbulence;
                    let turb_len = (turb_x * turb_x + turb_z * turb_z).sqrt();
                    let wd_x = if turb_len > 0.0 { turb_x / turb_len } else { dir.x };
                    let wd_z = if turb_len > 0.0 { turb_z / turb_len } else { dir.y };

                    // Upwind cell (where wind comes from)
                    let upwind_x = (x as f32 - wd_x).round() as i32;
                    let upwind_z = (z as f32 - wd_z).round() as i32;

                    if upwind_x < 0
                        || upwind_x >= w as i32
                        || upwind_z < 0
                        || upwind_z >= h as i32
                    {
                        continue;
                    }

                    let upwind_idx = upwind_z as usize * w + upwind_x as usize;
                    let upwind_h = heights_snap[upwind_idx];

                    // Sheltering: cells behind higher terrain get less wind
                    let shelter = (upwind_h - center_h).max(0.0);
                    let effective_strength = (self.settings.strength - shelter * 2.0).max(0.0);

                    // Pickup: exposed terrain loses material
                    let pickup = effective_strength * self.settings.pickup_rate;
                    let actual_pickup = pickup.min(heightmap.heights()[idx]);
                    heightmap.heights_mut()[idx] -= actual_pickup;

                    // Add to suspension at downwind cell
                    let downwind_x = (x as f32 + wd_x).round() as usize;
                    let downwind_z = (z as f32 + wd_z).round() as usize;

                    if downwind_x < w && downwind_z < h {
                        let dw_idx = downwind_z * w + downwind_x;
                        suspension[dw_idx] += actual_pickup + suspension[idx] * 0.5;
                    }

                    // Deposit suspended material
                    let deposit = suspension[idx] * self.settings.deposition_rate;
                    heightmap.heights_mut()[idx] += deposit;
                    suspension[idx] -= deposit;

                    // Capacity limit
                    if suspension[idx] > self.settings.capacity {
                        let excess = suspension[idx] - self.settings.capacity;
                        heightmap.heights_mut()[idx] += excess;
                        suspension[idx] = self.settings.capacity;
                    }
                }
            }
        }

        // Deposit any remaining suspended material
        for i in 0..w * h {
            heightmap.heights_mut()[i] += suspension[i];
        }

        heightmap.recalculate_bounds();
    }
}

// ---------------------------------------------------------------------------
// TerrainBrush
// ---------------------------------------------------------------------------

/// Interactive terrain brush for real-time editing operations.
///
/// Supports raise, lower, smooth, flatten, and noise operations applied
/// through a circular brush with configurable radius, strength, and falloff.
pub struct TerrainBrush {
    /// Brush configuration.
    pub settings: BrushSettings,
}

/// Configuration for terrain brush operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrushSettings {
    /// Brush radius in heightmap cells.
    pub radius: f32,
    /// Brush strength (0..1).
    pub strength: f32,
    /// Brush falloff curve.
    pub falloff: BrushFalloff,
    /// Target height for flatten operations.
    pub target_height: f32,
    /// Noise frequency for noise brush.
    pub noise_frequency: f32,
    /// Noise amplitude for noise brush.
    pub noise_amplitude: f32,
}

impl Default for BrushSettings {
    fn default() -> Self {
        Self {
            radius: 5.0,
            strength: 0.5,
            falloff: BrushFalloff::Smooth,
            target_height: 0.0,
            noise_frequency: 10.0,
            noise_amplitude: 0.1,
        }
    }
}

/// Brush falloff curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrushFalloff {
    /// Constant strength across the brush.
    Constant,
    /// Linear falloff from center to edge.
    Linear,
    /// Smooth (Hermite) falloff.
    Smooth,
    /// Sharp quadratic falloff.
    Sharp,
}

/// The type of brush operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrushOperation {
    /// Raise terrain.
    Raise,
    /// Lower terrain.
    Lower,
    /// Smooth terrain (average with neighbors).
    Smooth,
    /// Flatten terrain to a target height.
    Flatten,
    /// Add noise to terrain.
    Noise,
}

impl TerrainBrush {
    /// Creates a new terrain brush.
    pub fn new(settings: BrushSettings) -> Self {
        Self { settings }
    }

    /// Applies the brush operation at the given heightmap position.
    ///
    /// `center_x` and `center_z` are in heightmap coordinates.
    /// `delta_time` is used to make the operation frame-rate independent.
    pub fn apply(
        &self,
        heightmap: &mut Heightmap,
        center_x: f32,
        center_z: f32,
        operation: BrushOperation,
        delta_time: f32,
    ) {
        let w = heightmap.width() as i32;
        let h = heightmap.height() as i32;
        let radius = self.settings.radius;
        let r2 = radius * radius;
        let strength = self.settings.strength * delta_time;

        let min_x = ((center_x - radius).floor() as i32).max(0);
        let max_x = ((center_x + radius).ceil() as i32).min(w - 1);
        let min_z = ((center_z - radius).floor() as i32).max(0);
        let max_z = ((center_z + radius).ceil() as i32).min(h - 1);

        match operation {
            BrushOperation::Raise => {
                for z in min_z..=max_z {
                    for x in min_x..=max_x {
                        let dx = x as f32 - center_x;
                        let dz = z as f32 - center_z;
                        let dist_sq = dx * dx + dz * dz;
                        if dist_sq > r2 {
                            continue;
                        }
                        let factor = self.falloff_at(dist_sq.sqrt(), radius);
                        let current = heightmap.get(x as u32, z as u32);
                        heightmap.set_height(x as u32, z as u32, current + strength * factor);
                    }
                }
            }

            BrushOperation::Lower => {
                for z in min_z..=max_z {
                    for x in min_x..=max_x {
                        let dx = x as f32 - center_x;
                        let dz = z as f32 - center_z;
                        let dist_sq = dx * dx + dz * dz;
                        if dist_sq > r2 {
                            continue;
                        }
                        let factor = self.falloff_at(dist_sq.sqrt(), radius);
                        let current = heightmap.get(x as u32, z as u32);
                        heightmap.set_height(x as u32, z as u32, current - strength * factor);
                    }
                }
            }

            BrushOperation::Smooth => {
                // First pass: compute smoothed values
                let mut smoothed = Vec::new();
                for z in min_z..=max_z {
                    for x in min_x..=max_x {
                        let dx = x as f32 - center_x;
                        let dz = z as f32 - center_z;
                        let dist_sq = dx * dx + dz * dz;
                        if dist_sq > r2 {
                            continue;
                        }

                        // Average of neighbors
                        let mut sum = 0.0f32;
                        let mut count = 0u32;
                        for nz in (z - 1).max(0)..=(z + 1).min(h - 1) {
                            for nx in (x - 1).max(0)..=(x + 1).min(w - 1) {
                                sum += heightmap.get(nx as u32, nz as u32);
                                count += 1;
                            }
                        }
                        let avg = sum / count as f32;
                        let factor = self.falloff_at(dist_sq.sqrt(), radius);
                        let current = heightmap.get(x as u32, z as u32);
                        let new_val = current + (avg - current) * strength * factor;
                        smoothed.push((x as u32, z as u32, new_val));
                    }
                }
                // Apply smoothed values
                for (x, z, val) in smoothed {
                    heightmap.set_height(x, z, val);
                }
            }

            BrushOperation::Flatten => {
                let target = self.settings.target_height;
                for z in min_z..=max_z {
                    for x in min_x..=max_x {
                        let dx = x as f32 - center_x;
                        let dz = z as f32 - center_z;
                        let dist_sq = dx * dx + dz * dz;
                        if dist_sq > r2 {
                            continue;
                        }
                        let factor = self.falloff_at(dist_sq.sqrt(), radius);
                        let current = heightmap.get(x as u32, z as u32);
                        let new_val = current + (target - current) * strength * factor;
                        heightmap.set_height(x as u32, z as u32, new_val);
                    }
                }
            }

            BrushOperation::Noise => {
                let freq = self.settings.noise_frequency;
                let amp = self.settings.noise_amplitude;
                for z in min_z..=max_z {
                    for x in min_x..=max_x {
                        let dx = x as f32 - center_x;
                        let dz = z as f32 - center_z;
                        let dist_sq = dx * dx + dz * dz;
                        if dist_sq > r2 {
                            continue;
                        }
                        let factor = self.falloff_at(dist_sq.sqrt(), radius);

                        // Simple hash-based noise
                        let hash = ((x.wrapping_mul(374761393))
                            .wrapping_add(z.wrapping_mul(668265263)))
                        .wrapping_mul(1274126177);
                        let noise_val = (hash & 0x7FFF) as f32 / 0x7FFF as f32 - 0.5;

                        let current = heightmap.get(x as u32, z as u32);
                        heightmap.set_height(
                            x as u32,
                            z as u32,
                            current + noise_val * amp * strength * factor,
                        );
                    }
                }
            }
        }
    }

    /// Computes the falloff factor at a given distance.
    fn falloff_at(&self, distance: f32, radius: f32) -> f32 {
        let t = (distance / radius).clamp(0.0, 1.0);
        match self.settings.falloff {
            BrushFalloff::Constant => 1.0,
            BrushFalloff::Linear => 1.0 - t,
            BrushFalloff::Smooth => {
                let s = 1.0 - t;
                s * s * (3.0 - 2.0 * s)
            }
            BrushFalloff::Sharp => (1.0 - t * t).max(0.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Noise-based terrain features
// ---------------------------------------------------------------------------

/// Applies ridged multifractal noise to a heightmap.
///
/// This creates sharp mountain ridge features by taking the absolute value
/// of noise and inverting it.
pub fn apply_ridged_multifractal(
    heightmap: &mut Heightmap,
    octaves: u32,
    frequency: f32,
    lacunarity: f32,
    gain: f32,
    offset: f32,
    strength: f32,
    seed: u64,
) {
    let w = heightmap.width() as usize;
    let h = heightmap.height() as usize;
    let perm = generate_perm_table(seed);

    for z in 0..h {
        for x in 0..w {
            let mut freq = frequency;
            let mut amp = 1.0f32;
            let mut value = 0.0f32;
            let mut weight = 1.0f32;

            for _oct in 0..octaves {
                let nx = x as f32 * freq / w as f32;
                let nz = z as f32 * freq / h as f32;
                let noise = value_noise(&perm, nx, nz);

                // Ridge: invert absolute value
                let signal = offset - (noise * 2.0 - 1.0).abs();
                let signal = signal * signal;

                // Weight successive octaves by previous
                let signal = signal * weight;
                weight = (signal * gain).clamp(0.0, 1.0);

                value += signal * amp;
                freq *= lacunarity;
                amp *= 0.5;
            }

            let idx = z * w + x;
            heightmap.heights_mut()[idx] += value * strength;
        }
    }

    heightmap.recalculate_bounds();
}

/// Applies terraced (stepped) noise to create mesa/plateau features.
pub fn apply_terraced_noise(
    heightmap: &mut Heightmap,
    steps: u32,
    sharpness: f32,
    strength: f32,
) {
    let steps_f = steps.max(2) as f32;

    for h in heightmap.heights_mut().iter_mut() {
        let normalized = *h; // assume 0..1
        let stepped = (normalized * steps_f).floor() / steps_f;
        let smooth = (normalized * steps_f).fract();
        let transition = if smooth < (1.0 / sharpness) {
            smooth * sharpness
        } else {
            1.0
        };
        let terraced = stepped + transition / steps_f;
        *h = *h * (1.0 - strength) + terraced * strength;
    }

    heightmap.recalculate_bounds();
}

/// Applies billow noise to create puffy, cloud-like terrain features.
pub fn apply_billow_noise(
    heightmap: &mut Heightmap,
    octaves: u32,
    frequency: f32,
    persistence: f32,
    strength: f32,
    seed: u64,
) {
    let w = heightmap.width() as usize;
    let h = heightmap.height() as usize;
    let perm = generate_perm_table(seed);

    for z in 0..h {
        for x in 0..w {
            let mut freq = frequency;
            let mut amp = 1.0f32;
            let mut value = 0.0f32;
            let mut max_amp = 0.0f32;

            for _oct in 0..octaves {
                let nx = x as f32 * freq / w as f32;
                let nz = z as f32 * freq / h as f32;
                let noise = value_noise(&perm, nx, nz);

                // Billow: absolute value of centered noise
                let signal = (noise * 2.0 - 1.0).abs();

                value += signal * amp;
                max_amp += amp;
                freq *= 2.0;
                amp *= persistence;
            }

            let idx = z * w + x;
            heightmap.heights_mut()[idx] += (value / max_amp) * strength;
        }
    }

    heightmap.recalculate_bounds();
}

// -- Helpers ----------------------------------------------------------------

fn generate_perm_table(seed: u64) -> Vec<u8> {
    let mut rng = SimpleRng::new(seed);
    let mut table: Vec<u8> = (0..=255).collect();
    for i in (1..256).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        table.swap(i, j);
    }
    let mut doubled = table.clone();
    doubled.extend_from_slice(&table);
    doubled
}

fn value_noise(perm: &[u8], x: f32, z: f32) -> f32 {
    let x0 = x.floor() as i32;
    let z0 = z.floor() as i32;
    let fx = x - x0 as f32;
    let fz = z - z0 as f32;
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sz = fz * fz * (3.0 - 2.0 * fz);

    let hash = |ix: i32, iz: i32| -> f32 {
        let xi = (ix & 255) as usize;
        let zi = (iz & 255) as usize;
        let idx = (perm[xi] as usize + zi) & 511;
        perm[idx] as f32 / 255.0
    };

    let v00 = hash(x0, z0);
    let v10 = hash(x0 + 1, z0);
    let v01 = hash(x0, z0 + 1);
    let v11 = hash(x0 + 1, z0 + 1);

    let a = v00 + (v10 - v00) * sx;
    let b = v01 + (v11 - v01) * sx;
    a + (b - a) * sz
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hydraulic_erosion_basic() {
        let mut hm = crate::Heightmap::generate_procedural(65, 0.5, 42).unwrap();
        let before = hm.heights().to_vec();

        let settings = ErosionSettings {
            iterations: 1000,
            ..Default::default()
        };
        let erosion = HydraulicErosion::new(settings);
        erosion.erode(&mut hm);

        // Terrain should have changed
        let after = hm.heights();
        let mut changed = false;
        for (a, b) in before.iter().zip(after.iter()) {
            if (a - b).abs() > 1e-6 {
                changed = true;
                break;
            }
        }
        assert!(changed, "Erosion should modify the heightmap");
    }

    #[test]
    fn thermal_erosion_reduces_peaks() {
        let mut hm = crate::Heightmap::generate_procedural(33, 0.7, 99).unwrap();
        let range_before = hm.max_height() - hm.min_height();

        let settings = ThermalSettings {
            iterations: 100,
            talus: 0.001,
            strength: 0.5,
            border: 1,
        };
        let erosion = ThermalErosion::new(settings);
        erosion.erode(&mut hm);

        let range_after = hm.max_height() - hm.min_height();
        // Thermal erosion should reduce height variation
        assert!(
            range_after <= range_before + 0.01,
            "Thermal erosion should reduce height range"
        );
    }

    #[test]
    fn wind_erosion_runs() {
        let mut hm = crate::Heightmap::generate_procedural(33, 0.5, 42).unwrap();
        let settings = WindSettings {
            iterations: 10,
            ..Default::default()
        };
        let erosion = WindErosion::new(settings);
        erosion.erode(&mut hm);
        // Just verify it runs without panic
    }

    #[test]
    fn brush_raise() {
        let mut hm = crate::Heightmap::new_flat(32, 32, 0.5).unwrap();
        let settings = BrushSettings {
            radius: 3.0,
            strength: 1.0,
            falloff: BrushFalloff::Constant,
            ..Default::default()
        };
        let brush = TerrainBrush::new(settings);
        brush.apply(&mut hm, 16.0, 16.0, BrushOperation::Raise, 1.0);

        let center_h = hm.get(16, 16);
        assert!(center_h > 0.5, "Brush should raise terrain at center");
    }

    #[test]
    fn brush_smooth() {
        // Create terrain with a spike
        let mut data = vec![0.0f32; 17 * 17];
        data[8 * 17 + 8] = 1.0; // spike at center
        let mut hm = crate::Heightmap::from_raw(17, 17, data).unwrap();

        let settings = BrushSettings {
            radius: 3.0,
            strength: 1.0,
            falloff: BrushFalloff::Constant,
            ..Default::default()
        };
        let brush = TerrainBrush::new(settings);
        brush.apply(&mut hm, 8.0, 8.0, BrushOperation::Smooth, 1.0);

        let center_h = hm.get(8, 8);
        assert!(center_h < 1.0, "Smoothing should reduce the spike");
    }

    #[test]
    fn brush_flatten() {
        let mut hm = crate::Heightmap::generate_procedural(33, 0.5, 42).unwrap();
        let settings = BrushSettings {
            radius: 5.0,
            strength: 1.0,
            target_height: 0.5,
            falloff: BrushFalloff::Constant,
            ..Default::default()
        };
        let brush = TerrainBrush::new(settings);

        for _ in 0..10 {
            brush.apply(&mut hm, 16.0, 16.0, BrushOperation::Flatten, 1.0);
        }

        let center_h = hm.get(16, 16);
        assert!(
            (center_h - 0.5).abs() < 0.1,
            "Flatten should bring terrain close to target"
        );
    }

    #[test]
    fn ridged_multifractal() {
        let mut hm = crate::Heightmap::new_flat(33, 33, 0.5).unwrap();
        apply_ridged_multifractal(&mut hm, 4, 4.0, 2.0, 0.5, 1.0, 0.5, 42);
        assert!(hm.max_height() > 0.5, "Ridged noise should add features");
    }

    #[test]
    fn terraced_noise() {
        let mut hm = crate::Heightmap::generate_procedural(33, 0.5, 42).unwrap();
        apply_terraced_noise(&mut hm, 5, 4.0, 0.8);
        // Should still be in valid range
        assert!(hm.min_height() >= -0.5);
    }

    #[test]
    fn billow_noise() {
        let mut hm = crate::Heightmap::new_flat(33, 33, 0.0).unwrap();
        apply_billow_noise(&mut hm, 4, 4.0, 0.5, 0.5, 42);
        assert!(hm.max_height() > 0.0, "Billow noise should add features");
    }
}
