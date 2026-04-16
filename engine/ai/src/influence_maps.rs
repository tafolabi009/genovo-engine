//! Influence map system.
//!
//! Provides 2D grid-based influence maps for strategic AI decision-making.
//! Influence maps track spatial "influence" values that represent concepts
//! like threat, resource density, or exploration priority. The AI can then
//! query these maps to make spatial decisions such as finding the safest
//! position, the best attack position, or the nearest resource.
//!
//! Features:
//! - Multiple falloff types (Linear, Quadratic, Constant)
//! - Bilinear interpolation for smooth sampling
//! - Influence propagation with configurable decay
//! - Gaussian blur for smoothing
//! - Map combination operations (Add, Max, Min, Multiply)
//! - Strategic queries (maximum, minimum, threshold regions)
//! - Temporal decay for values that fade over time
//! - Named map management via `InfluenceMapManager`

use glam::Vec3;

// ---------------------------------------------------------------------------
// Falloff
// ---------------------------------------------------------------------------

/// Controls how influence diminishes with distance from the source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Falloff {
    /// Influence decreases linearly: `value * (1 - distance / radius)`.
    Linear,
    /// Influence decreases quadratically: `value * (1 - (distance / radius)^2)`.
    Quadratic,
    /// Influence is constant within the radius.
    Constant,
}

// ---------------------------------------------------------------------------
// CombineOp
// ---------------------------------------------------------------------------

/// Operation used when combining two influence maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombineOp {
    /// Add values from both maps.
    Add,
    /// Take the maximum value from either map.
    Max,
    /// Take the minimum value from either map.
    Min,
    /// Multiply values from both maps.
    Multiply,
}

// ---------------------------------------------------------------------------
// InfluenceMap
// ---------------------------------------------------------------------------

/// A 2D grid of influence values.
///
/// Each cell stores a floating-point value representing the influence at
/// that grid position. The map covers a rectangular region of the XZ plane,
/// with configurable cell size for resolution control.
#[derive(Debug, Clone)]
pub struct InfluenceMap {
    /// Width of the grid in cells.
    pub width: usize,
    /// Height of the grid in cells.
    pub height: usize,
    /// World-space size of each cell.
    pub cell_size: f32,
    /// Origin of the map in world space (bottom-left corner).
    pub origin: Vec3,
    /// The influence values, stored row-major: values[y * width + x].
    pub values: Vec<f32>,
}

impl InfluenceMap {
    /// Creates a new influence map initialized to zero.
    pub fn new(width: usize, height: usize, cell_size: f32) -> Self {
        Self {
            width,
            height,
            cell_size,
            origin: Vec3::ZERO,
            values: vec![0.0; width * height],
        }
    }

    /// Creates a new influence map with a specified origin.
    pub fn with_origin(mut self, origin: Vec3) -> Self {
        self.origin = origin;
        self
    }

    /// Returns the total number of cells.
    pub fn cell_count(&self) -> usize {
        self.width * self.height
    }

    /// Get the value at grid coordinates (x, y).
    pub fn get(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height {
            self.values[y * self.width + x]
        } else {
            0.0
        }
    }

    /// Set the value at grid coordinates (x, y).
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        if x < self.width && y < self.height {
            self.values[y * self.width + x] = value;
        }
    }

    /// Clear all values to zero.
    pub fn clear(&mut self) {
        for v in &mut self.values {
            *v = 0.0;
        }
    }

    /// Fill all values with the given constant.
    pub fn fill(&mut self, value: f32) {
        for v in &mut self.values {
            *v = value;
        }
    }

    /// Convert world-space position to grid coordinates.
    pub fn world_to_grid(&self, position: Vec3) -> (i32, i32) {
        let local = position - self.origin;
        let x = (local.x / self.cell_size).floor() as i32;
        let y = (local.z / self.cell_size).floor() as i32;
        (x, y)
    }

    /// Convert grid coordinates to world-space position (center of cell).
    pub fn grid_to_world(&self, x: usize, y: usize) -> Vec3 {
        Vec3::new(
            (x as f32 + 0.5) * self.cell_size + self.origin.x,
            self.origin.y,
            (y as f32 + 0.5) * self.cell_size + self.origin.z,
        )
    }

    /// Check if grid coordinates are in bounds.
    pub fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height
    }

    // -----------------------------------------------------------------------
    // Influence stamping
    // -----------------------------------------------------------------------

    /// Add influence at a world-space position with a given radius and falloff.
    ///
    /// The influence is added (not set) to existing values, so multiple
    /// sources accumulate.
    pub fn add_influence(
        &mut self,
        position: Vec3,
        radius: f32,
        value: f32,
        falloff: Falloff,
    ) {
        let (cx, cy) = self.world_to_grid(position);
        let cell_radius = (radius / self.cell_size).ceil() as i32;

        let min_x = (cx - cell_radius).max(0) as usize;
        let max_x = ((cx + cell_radius) as usize).min(self.width - 1);
        let min_y = (cy - cell_radius).max(0) as usize;
        let max_y = ((cy + cell_radius) as usize).min(self.height - 1);

        for gy in min_y..=max_y {
            for gx in min_x..=max_x {
                let world_pos = self.grid_to_world(gx, gy);
                let dx = world_pos.x - position.x;
                let dz = world_pos.z - position.z;
                let distance = (dx * dx + dz * dz).sqrt();

                if distance > radius {
                    continue;
                }

                let influence = match falloff {
                    Falloff::Linear => {
                        value * (1.0 - distance / radius)
                    }
                    Falloff::Quadratic => {
                        let t = distance / radius;
                        value * (1.0 - t * t)
                    }
                    Falloff::Constant => value,
                };

                let idx = gy * self.width + gx;
                self.values[idx] += influence;
            }
        }
    }

    /// Set influence at a world-space position (replaces existing values).
    pub fn set_influence(
        &mut self,
        position: Vec3,
        radius: f32,
        value: f32,
        falloff: Falloff,
    ) {
        let (cx, cy) = self.world_to_grid(position);
        let cell_radius = (radius / self.cell_size).ceil() as i32;

        let min_x = (cx - cell_radius).max(0) as usize;
        let max_x = ((cx + cell_radius) as usize).min(self.width - 1);
        let min_y = (cy - cell_radius).max(0) as usize;
        let max_y = ((cy + cell_radius) as usize).min(self.height - 1);

        for gy in min_y..=max_y {
            for gx in min_x..=max_x {
                let world_pos = self.grid_to_world(gx, gy);
                let dx = world_pos.x - position.x;
                let dz = world_pos.z - position.z;
                let distance = (dx * dx + dz * dz).sqrt();

                if distance > radius {
                    continue;
                }

                let influence = match falloff {
                    Falloff::Linear => value * (1.0 - distance / radius),
                    Falloff::Quadratic => {
                        let t = distance / radius;
                        value * (1.0 - t * t)
                    }
                    Falloff::Constant => value,
                };

                let idx = gy * self.width + gx;
                self.values[idx] = influence;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Sampling
    // -----------------------------------------------------------------------

    /// Sample the influence at a world-space position using bilinear interpolation.
    pub fn sample(&self, position: Vec3) -> f32 {
        let local = position - self.origin;
        let fx = local.x / self.cell_size - 0.5;
        let fy = local.z / self.cell_size - 0.5;

        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let v00 = self.safe_get(x0, y0);
        let v10 = self.safe_get(x1, y0);
        let v01 = self.safe_get(x0, y1);
        let v11 = self.safe_get(x1, y1);

        let top = v00 * (1.0 - tx) + v10 * tx;
        let bottom = v01 * (1.0 - tx) + v11 * tx;
        top * (1.0 - ty) + bottom * ty
    }

    /// Safe grid access that returns 0 for out-of-bounds coordinates.
    fn safe_get(&self, x: i32, y: i32) -> f32 {
        if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
            self.values[y as usize * self.width + x as usize]
        } else {
            0.0
        }
    }

    // -----------------------------------------------------------------------
    // Propagation and smoothing
    // -----------------------------------------------------------------------

    /// Propagate influence to neighboring cells with decay.
    ///
    /// Each cell's value spreads to its 4-neighbors, reduced by the decay
    /// factor. This simulates influence diffusing outward over time.
    pub fn propagate(&mut self, decay: f32) {
        let mut new_values = vec![0.0f32; self.values.len()];

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;
                let current = self.values[idx];

                // Add decayed influence from neighbors.
                let mut neighbor_sum = 0.0f32;
                let mut neighbor_count = 0u32;

                if x > 0 {
                    neighbor_sum += self.values[y * self.width + (x - 1)];
                    neighbor_count += 1;
                }
                if x + 1 < self.width {
                    neighbor_sum += self.values[y * self.width + (x + 1)];
                    neighbor_count += 1;
                }
                if y > 0 {
                    neighbor_sum += self.values[(y - 1) * self.width + x];
                    neighbor_count += 1;
                }
                if y + 1 < self.height {
                    neighbor_sum += self.values[(y + 1) * self.width + x];
                    neighbor_count += 1;
                }

                let neighbor_avg = if neighbor_count > 0 {
                    neighbor_sum / neighbor_count as f32
                } else {
                    0.0
                };

                // New value: blend of current and neighbor average with decay.
                new_values[idx] = current * decay + neighbor_avg * (1.0 - decay) * decay;
            }
        }

        self.values = new_values;
    }

    /// Apply Gaussian blur for smoothing.
    ///
    /// Uses a 3x3 Gaussian kernel applied `iterations` times.
    pub fn blur(&mut self, iterations: usize) {
        // 3x3 Gaussian kernel (normalized).
        let kernel = [
            1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
            2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0,
            1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
        ];

        for _ in 0..iterations {
            let mut new_values = vec![0.0f32; self.values.len()];

            for y in 0..self.height {
                for x in 0..self.width {
                    let mut sum = 0.0f32;
                    let mut weight_sum = 0.0f32;

                    for ky in 0..3i32 {
                        for kx in 0..3i32 {
                            let nx = x as i32 + kx - 1;
                            let ny = y as i32 + ky - 1;

                            if nx >= 0
                                && ny >= 0
                                && (nx as usize) < self.width
                                && (ny as usize) < self.height
                            {
                                let k = kernel[(ky * 3 + kx) as usize];
                                sum += self.values[ny as usize * self.width + nx as usize] * k;
                                weight_sum += k;
                            }
                        }
                    }

                    if weight_sum > 0.0 {
                        new_values[y * self.width + x] = sum / weight_sum;
                    }
                }
            }

            self.values = new_values;
        }
    }

    /// Apply temporal decay: multiply all values by the decay factor.
    ///
    /// Values below the threshold are zeroed out.
    pub fn decay(&mut self, factor: f32, threshold: f32) {
        for v in &mut self.values {
            *v *= factor;
            if v.abs() < threshold {
                *v = 0.0;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Combination
    // -----------------------------------------------------------------------

    /// Combine this map with another map using the given operation.
    ///
    /// Both maps must have the same dimensions.
    pub fn combine(&mut self, other: &InfluenceMap, op: CombineOp) {
        assert_eq!(
            self.values.len(),
            other.values.len(),
            "Maps must have the same dimensions for combine"
        );

        for i in 0..self.values.len() {
            self.values[i] = match op {
                CombineOp::Add => self.values[i] + other.values[i],
                CombineOp::Max => self.values[i].max(other.values[i]),
                CombineOp::Min => self.values[i].min(other.values[i]),
                CombineOp::Multiply => self.values[i] * other.values[i],
            };
        }
    }

    /// Create a new map that is the combination of this map and another.
    pub fn combined(&self, other: &InfluenceMap, op: CombineOp) -> InfluenceMap {
        let mut result = self.clone();
        result.combine(other, op);
        result
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Find the position of the cell with the highest influence value.
    pub fn find_maximum(&self) -> Vec3 {
        let mut best_idx = 0;
        let mut best_val = f32::NEG_INFINITY;

        for (i, &v) in self.values.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }

        let x = best_idx % self.width;
        let y = best_idx / self.width;
        self.grid_to_world(x, y)
    }

    /// Find the position of the cell with the lowest influence value.
    pub fn find_minimum(&self) -> Vec3 {
        let mut best_idx = 0;
        let mut best_val = f32::INFINITY;

        for (i, &v) in self.values.iter().enumerate() {
            if v < best_val {
                best_val = v;
                best_idx = i;
            }
        }

        let x = best_idx % self.width;
        let y = best_idx / self.width;
        self.grid_to_world(x, y)
    }

    /// Find the maximum influence value in the map.
    pub fn max_value(&self) -> f32 {
        self.values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Find the minimum influence value in the map.
    pub fn min_value(&self) -> f32 {
        self.values.iter().copied().fold(f32::INFINITY, f32::min)
    }

    /// Find the average influence value.
    pub fn average_value(&self) -> f32 {
        if self.values.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.values.iter().sum();
        sum / self.values.len() as f32
    }

    /// Find all cells above a threshold value. Returns their world positions.
    pub fn find_above_threshold(&self, threshold: f32) -> Vec<Vec3> {
        let mut result = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                let val = self.values[y * self.width + x];
                if val > threshold {
                    result.push(self.grid_to_world(x, y));
                }
            }
        }
        result
    }

    /// Find the highest-value position within a radius of a given point.
    pub fn find_maximum_near(&self, position: Vec3, radius: f32) -> Vec3 {
        let (cx, cy) = self.world_to_grid(position);
        let cell_radius = (radius / self.cell_size).ceil() as i32;

        let mut best_pos = position;
        let mut best_val = f32::NEG_INFINITY;

        let min_x = (cx - cell_radius).max(0) as usize;
        let max_x = ((cx + cell_radius) as usize).min(self.width.saturating_sub(1));
        let min_y = (cy - cell_radius).max(0) as usize;
        let max_y = ((cy + cell_radius) as usize).min(self.height.saturating_sub(1));

        for gy in min_y..=max_y {
            for gx in min_x..=max_x {
                let wp = self.grid_to_world(gx, gy);
                let dist = ((wp.x - position.x).powi(2) + (wp.z - position.z).powi(2)).sqrt();
                if dist <= radius {
                    let val = self.values[gy * self.width + gx];
                    if val > best_val {
                        best_val = val;
                        best_pos = wp;
                    }
                }
            }
        }

        best_pos
    }

    /// Find the lowest-value position within a radius of a given point.
    pub fn find_minimum_near(&self, position: Vec3, radius: f32) -> Vec3 {
        let (cx, cy) = self.world_to_grid(position);
        let cell_radius = (radius / self.cell_size).ceil() as i32;

        let mut best_pos = position;
        let mut best_val = f32::INFINITY;

        let min_x = (cx - cell_radius).max(0) as usize;
        let max_x = ((cx + cell_radius) as usize).min(self.width.saturating_sub(1));
        let min_y = (cy - cell_radius).max(0) as usize;
        let max_y = ((cy + cell_radius) as usize).min(self.height.saturating_sub(1));

        for gy in min_y..=max_y {
            for gx in min_x..=max_x {
                let wp = self.grid_to_world(gx, gy);
                let dist = ((wp.x - position.x).powi(2) + (wp.z - position.z).powi(2)).sqrt();
                if dist <= radius {
                    let val = self.values[gy * self.width + gx];
                    if val < best_val {
                        best_val = val;
                        best_pos = wp;
                    }
                }
            }
        }

        best_pos
    }

    /// Normalize all values to the range [0, 1].
    pub fn normalize(&mut self) {
        let min_val = self.min_value();
        let max_val = self.max_value();
        let range = max_val - min_val;

        if range.abs() < 1e-8 {
            for v in &mut self.values {
                *v = 0.0;
            }
            return;
        }

        for v in &mut self.values {
            *v = (*v - min_val) / range;
        }
    }

    /// Clamp all values to the given range.
    pub fn clamp(&mut self, min: f32, max: f32) {
        for v in &mut self.values {
            *v = v.clamp(min, max);
        }
    }

    /// Invert all values: `new_value = max_value - value`.
    pub fn invert(&mut self) {
        let max_val = self.max_value();
        for v in &mut self.values {
            *v = max_val - *v;
        }
    }
}

// ---------------------------------------------------------------------------
// InfluenceMapManager
// ---------------------------------------------------------------------------

/// Manages multiple named influence maps for strategic AI decisions.
///
/// Common maps include:
/// - `"threat"` — enemy positions and danger zones
/// - `"resources"` — resource locations and density
/// - `"exploration"` — unvisited/unexplored areas
/// - `"tactical"` — combined strategic value
pub struct InfluenceMapManager {
    /// Named influence maps.
    maps: std::collections::HashMap<String, InfluenceMap>,
    /// Default map dimensions.
    default_width: usize,
    /// Default map dimensions.
    default_height: usize,
    /// Default cell size.
    default_cell_size: f32,
    /// Default origin.
    default_origin: Vec3,
}

impl InfluenceMapManager {
    /// Creates a new manager with default map dimensions.
    pub fn new(width: usize, height: usize, cell_size: f32) -> Self {
        Self {
            maps: std::collections::HashMap::new(),
            default_width: width,
            default_height: height,
            default_cell_size: cell_size,
            default_origin: Vec3::ZERO,
        }
    }

    /// Set the default origin for new maps.
    pub fn with_origin(mut self, origin: Vec3) -> Self {
        self.default_origin = origin;
        self
    }

    /// Create a standard set of maps (threat, resources, exploration, tactical).
    pub fn create_standard_maps(&mut self) {
        self.create_map("threat");
        self.create_map("resources");
        self.create_map("exploration");
        self.create_map("tactical");

        // Initialize exploration map with 1.0 (everything unexplored).
        if let Some(exploration) = self.maps.get_mut("exploration") {
            exploration.fill(1.0);
        }
    }

    /// Create a new named map with default dimensions.
    pub fn create_map(&mut self, name: impl Into<String>) {
        let map = InfluenceMap::new(
            self.default_width,
            self.default_height,
            self.default_cell_size,
        )
        .with_origin(self.default_origin);
        self.maps.insert(name.into(), map);
    }

    /// Get a reference to a named map.
    pub fn get(&self, name: &str) -> Option<&InfluenceMap> {
        self.maps.get(name)
    }

    /// Get a mutable reference to a named map.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut InfluenceMap> {
        self.maps.get_mut(name)
    }

    /// Remove a named map.
    pub fn remove(&mut self, name: &str) -> Option<InfluenceMap> {
        self.maps.remove(name)
    }

    /// Check if a named map exists.
    pub fn has_map(&self, name: &str) -> bool {
        self.maps.contains_key(name)
    }

    /// Get the names of all maps.
    pub fn map_names(&self) -> Vec<&str> {
        self.maps.keys().map(|s| s.as_str()).collect()
    }

    /// Clear all maps (set values to zero).
    pub fn clear_all(&mut self) {
        for map in self.maps.values_mut() {
            map.clear();
        }
    }

    /// Apply temporal decay to all maps.
    pub fn decay_all(&mut self, factor: f32, threshold: f32) {
        for map in self.maps.values_mut() {
            map.decay(factor, threshold);
        }
    }

    // -----------------------------------------------------------------------
    // Strategic queries
    // -----------------------------------------------------------------------

    /// Find the safest position (lowest threat, within a search area).
    pub fn find_safest_position(&self, near: Vec3, radius: f32) -> Option<Vec3> {
        self.maps
            .get("threat")
            .map(|map| map.find_minimum_near(near, radius))
    }

    /// Find the best attack position (highest tactical value near the target).
    pub fn find_best_attack_position(
        &self,
        target: Vec3,
        radius: f32,
    ) -> Option<Vec3> {
        // Combine threat (inverted) and tactical for best attack spot.
        if let (Some(threat), Some(tactical)) =
            (self.maps.get("threat"), self.maps.get("tactical"))
        {
            let mut combined = threat.clone();
            combined.invert();
            combined.normalize();
            let mut tac_norm = tactical.clone();
            tac_norm.normalize();
            combined.combine(&tac_norm, CombineOp::Add);
            Some(combined.find_maximum_near(target, radius))
        } else {
            None
        }
    }

    /// Find the nearest resource.
    pub fn find_nearest_resource(&self, position: Vec3, radius: f32) -> Option<Vec3> {
        self.maps
            .get("resources")
            .map(|map| map.find_maximum_near(position, radius))
    }

    /// Find the most unexplored area.
    pub fn find_unexplored(&self, position: Vec3, radius: f32) -> Option<Vec3> {
        self.maps
            .get("exploration")
            .map(|map| map.find_maximum_near(position, radius))
    }

    /// Mark an area as explored.
    pub fn mark_explored(&mut self, position: Vec3, radius: f32) {
        if let Some(exploration) = self.maps.get_mut("exploration") {
            exploration.set_influence(position, radius, 0.0, Falloff::Linear);
        }
    }

    /// Update the tactical map by combining other maps.
    ///
    /// The tactical map is computed as: resources - threat + exploration_bonus.
    pub fn update_tactical_map(&mut self) {
        let (resources, threat, exploration) = {
            let r = self.maps.get("resources").cloned();
            let t = self.maps.get("threat").cloned();
            let e = self.maps.get("exploration").cloned();
            (r, t, e)
        };

        if let Some(tactical) = self.maps.get_mut("tactical") {
            tactical.clear();

            if let Some(ref resources) = resources {
                tactical.combine(resources, CombineOp::Add);
            }

            if let Some(ref threat) = threat {
                // Subtract threat by inverting and using Min.
                let mut inverted_threat = threat.clone();
                for v in &mut inverted_threat.values {
                    *v = -*v;
                }
                tactical.combine(&inverted_threat, CombineOp::Add);
            }

            if let Some(ref exploration) = exploration {
                // Add a small exploration bonus.
                let mut exploration_bonus = exploration.clone();
                for v in &mut exploration_bonus.values {
                    *v *= 0.2; // 20% weight for exploration.
                }
                tactical.combine(&exploration_bonus, CombineOp::Add);
            }
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
    fn test_influence_map_creation() {
        let map = InfluenceMap::new(10, 10, 1.0);
        assert_eq!(map.width, 10);
        assert_eq!(map.height, 10);
        assert_eq!(map.cell_count(), 100);
        assert_eq!(map.get(0, 0), 0.0);
    }

    #[test]
    fn test_influence_map_set_get() {
        let mut map = InfluenceMap::new(10, 10, 1.0);
        map.set(5, 5, 42.0);
        assert_eq!(map.get(5, 5), 42.0);
        assert_eq!(map.get(0, 0), 0.0);
    }

    #[test]
    fn test_influence_map_clear() {
        let mut map = InfluenceMap::new(10, 10, 1.0);
        map.set(3, 3, 1.0);
        map.set(7, 7, 2.0);
        map.clear();
        assert_eq!(map.get(3, 3), 0.0);
        assert_eq!(map.get(7, 7), 0.0);
    }

    #[test]
    fn test_influence_map_world_grid_conversion() {
        let map = InfluenceMap::new(10, 10, 2.0);
        let world = map.grid_to_world(3, 4);
        assert!((world.x - 7.0).abs() < 0.01); // (3+0.5)*2.0
        assert!((world.z - 9.0).abs() < 0.01); // (4+0.5)*2.0

        let (gx, gy) = map.world_to_grid(world);
        assert_eq!(gx, 3);
        assert_eq!(gy, 4);
    }

    #[test]
    fn test_influence_map_add_influence_constant() {
        let mut map = InfluenceMap::new(20, 20, 1.0);
        map.add_influence(Vec3::new(10.5, 0.0, 10.5), 3.0, 5.0, Falloff::Constant);

        // The center cell should have influence = 5.0.
        assert!(map.get(10, 10) > 4.9);

        // A cell far away should have no influence.
        assert_eq!(map.get(0, 0), 0.0);
    }

    #[test]
    fn test_influence_map_add_influence_linear() {
        let mut map = InfluenceMap::new(20, 20, 1.0);
        map.add_influence(Vec3::new(10.5, 0.0, 10.5), 5.0, 10.0, Falloff::Linear);

        let center = map.get(10, 10);
        let edge_ish = map.get(10, 14); // ~4 cells away from center

        // Center should have higher influence than edge.
        assert!(center > edge_ish);
        // Center should be close to max.
        assert!(center > 8.0);
    }

    #[test]
    fn test_influence_map_add_influence_quadratic() {
        let mut map = InfluenceMap::new(20, 20, 1.0);
        map.add_influence(
            Vec3::new(10.5, 0.0, 10.5),
            5.0,
            10.0,
            Falloff::Quadratic,
        );

        let center = map.get(10, 10);
        assert!(center > 8.0);
    }

    #[test]
    fn test_influence_map_sample_bilinear() {
        let mut map = InfluenceMap::new(10, 10, 1.0);
        map.set(5, 5, 10.0);

        // Sampling at the cell center should return close to 10.0.
        let val = map.sample(Vec3::new(5.5, 0.0, 5.5));
        assert!(val > 5.0);

        // Sampling far away should return close to 0.0.
        let val = map.sample(Vec3::new(0.5, 0.0, 0.5));
        assert!(val < 0.1);
    }

    #[test]
    fn test_influence_map_propagate() {
        let mut map = InfluenceMap::new(10, 10, 1.0);
        map.set(5, 5, 10.0);

        // Neighbors should be 0 before propagation.
        assert_eq!(map.get(5, 6), 0.0);

        map.propagate(0.8);

        // After propagation, neighbors should have some influence.
        assert!(map.get(5, 6) > 0.0);
        // Center should be reduced.
        assert!(map.get(5, 5) < 10.0);
    }

    #[test]
    fn test_influence_map_blur() {
        let mut map = InfluenceMap::new(10, 10, 1.0);
        map.set(5, 5, 10.0);

        map.blur(1);

        // Center should be reduced after blur.
        assert!(map.get(5, 5) < 10.0);
        // Neighbors should have gained some value.
        assert!(map.get(5, 6) > 0.0);
        assert!(map.get(4, 5) > 0.0);
    }

    #[test]
    fn test_influence_map_decay() {
        let mut map = InfluenceMap::new(10, 10, 1.0);
        map.set(5, 5, 10.0);
        map.set(3, 3, 0.001);

        map.decay(0.5, 0.01);

        assert!((map.get(5, 5) - 5.0).abs() < 0.01);
        assert_eq!(map.get(3, 3), 0.0); // Below threshold
    }

    #[test]
    fn test_influence_map_combine_add() {
        let mut map1 = InfluenceMap::new(10, 10, 1.0);
        let mut map2 = InfluenceMap::new(10, 10, 1.0);
        map1.set(5, 5, 3.0);
        map2.set(5, 5, 7.0);

        map1.combine(&map2, CombineOp::Add);
        assert!((map1.get(5, 5) - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_influence_map_combine_max() {
        let mut map1 = InfluenceMap::new(10, 10, 1.0);
        let mut map2 = InfluenceMap::new(10, 10, 1.0);
        map1.set(5, 5, 3.0);
        map2.set(5, 5, 7.0);

        map1.combine(&map2, CombineOp::Max);
        assert!((map1.get(5, 5) - 7.0).abs() < 0.01);
    }

    #[test]
    fn test_influence_map_combine_min() {
        let mut map1 = InfluenceMap::new(10, 10, 1.0);
        let mut map2 = InfluenceMap::new(10, 10, 1.0);
        map1.set(5, 5, 3.0);
        map2.set(5, 5, 7.0);

        map1.combine(&map2, CombineOp::Min);
        assert!((map1.get(5, 5) - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_influence_map_find_maximum() {
        let mut map = InfluenceMap::new(10, 10, 1.0);
        map.set(3, 7, 15.0);
        map.set(5, 5, 10.0);

        let max_pos = map.find_maximum();
        let (gx, gy) = map.world_to_grid(max_pos);
        assert_eq!(gx, 3);
        assert_eq!(gy, 7);
    }

    #[test]
    fn test_influence_map_find_minimum() {
        let mut map = InfluenceMap::new(10, 10, 1.0);
        map.fill(10.0);
        map.set(2, 8, -5.0);

        let min_pos = map.find_minimum();
        let (gx, gy) = map.world_to_grid(min_pos);
        assert_eq!(gx, 2);
        assert_eq!(gy, 8);
    }

    #[test]
    fn test_influence_map_normalize() {
        let mut map = InfluenceMap::new(10, 10, 1.0);
        map.set(0, 0, -10.0);
        map.set(9, 9, 10.0);

        map.normalize();

        assert!((map.get(0, 0) - 0.0).abs() < 0.01);
        assert!((map.get(9, 9) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_influence_map_invert() {
        let mut map = InfluenceMap::new(10, 10, 1.0);
        map.set(0, 0, 2.0);
        map.set(5, 5, 8.0);

        map.invert();

        // max was 8.0: inverted 2.0 -> 6.0, 8.0 -> 0.0
        assert!((map.get(0, 0) - 6.0).abs() < 0.01);
        assert!((map.get(5, 5) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_influence_map_above_threshold() {
        let mut map = InfluenceMap::new(10, 10, 1.0);
        map.set(2, 2, 5.0);
        map.set(5, 5, 10.0);
        map.set(8, 8, 3.0);

        let above = map.find_above_threshold(4.0);
        assert_eq!(above.len(), 2); // 5.0 and 10.0 are above 4.0
    }

    #[test]
    fn test_influence_map_statistics() {
        let mut map = InfluenceMap::new(10, 10, 1.0);
        map.fill(5.0);
        map.set(0, 0, 0.0);
        map.set(9, 9, 10.0);

        assert!((map.min_value() - 0.0).abs() < 0.01);
        assert!((map.max_value() - 10.0).abs() < 0.01);

        let avg = map.average_value();
        assert!(avg > 4.0 && avg < 6.0);
    }

    #[test]
    fn test_influence_map_manager_creation() {
        let mut mgr = InfluenceMapManager::new(20, 20, 1.0);
        mgr.create_standard_maps();

        assert!(mgr.has_map("threat"));
        assert!(mgr.has_map("resources"));
        assert!(mgr.has_map("exploration"));
        assert!(mgr.has_map("tactical"));
    }

    #[test]
    fn test_influence_map_manager_get_set() {
        let mut mgr = InfluenceMapManager::new(10, 10, 1.0);
        mgr.create_map("test");

        let map = mgr.get_mut("test").unwrap();
        map.set(5, 5, 42.0);

        assert_eq!(mgr.get("test").unwrap().get(5, 5), 42.0);
    }

    #[test]
    fn test_influence_map_manager_safest_position() {
        let mut mgr = InfluenceMapManager::new(20, 20, 1.0);
        mgr.create_standard_maps();

        // Set threat at center.
        if let Some(threat) = mgr.get_mut("threat") {
            threat.add_influence(
                Vec3::new(10.5, 0.0, 10.5),
                5.0,
                10.0,
                Falloff::Linear,
            );
        }

        // The safest position should be away from the threat center.
        let safe = mgr
            .find_safest_position(Vec3::new(10.5, 0.0, 10.5), 15.0)
            .unwrap();

        // Verify it's not at the threat center.
        let dist = ((safe.x - 10.5).powi(2) + (safe.z - 10.5).powi(2)).sqrt();
        assert!(dist > 3.0);
    }

    #[test]
    fn test_influence_map_manager_exploration() {
        let mut mgr = InfluenceMapManager::new(20, 20, 1.0);
        mgr.create_standard_maps();

        // Exploration should start at 1.0 everywhere.
        let val = mgr.get("exploration").unwrap().get(5, 5);
        assert!((val - 1.0).abs() < 0.01);

        // Mark an area as explored.
        mgr.mark_explored(Vec3::new(5.5, 0.0, 5.5), 3.0);

        // The explored area should now be lower.
        let val = mgr.get("exploration").unwrap().get(5, 5);
        assert!(val < 0.5);
    }

    #[test]
    fn test_influence_map_manager_decay_all() {
        let mut mgr = InfluenceMapManager::new(10, 10, 1.0);
        mgr.create_map("test");

        mgr.get_mut("test").unwrap().set(5, 5, 10.0);
        mgr.decay_all(0.9, 0.01);

        assert!((mgr.get("test").unwrap().get(5, 5) - 9.0).abs() < 0.01);
    }

    #[test]
    fn test_influence_map_find_max_near() {
        let mut map = InfluenceMap::new(20, 20, 1.0);
        map.set(5, 5, 100.0);
        map.set(15, 15, 50.0);

        // Search near (5,5) with small radius.
        let result = map.find_maximum_near(Vec3::new(5.5, 0.0, 5.5), 3.0);
        let (gx, gy) = map.world_to_grid(result);
        assert_eq!(gx, 5);
        assert_eq!(gy, 5);
    }

    #[test]
    fn test_influence_map_clamp() {
        let mut map = InfluenceMap::new(10, 10, 1.0);
        map.set(0, 0, -5.0);
        map.set(5, 5, 15.0);

        map.clamp(0.0, 10.0);

        assert_eq!(map.get(0, 0), 0.0);
        assert_eq!(map.get(5, 5), 10.0);
    }
}
