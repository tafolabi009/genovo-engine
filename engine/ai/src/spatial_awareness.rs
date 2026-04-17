// engine/ai/src/spatial_awareness.rs
//
// AI spatial awareness system for the Genovo AI module.
//
// Provides:
// - Threat map computation with decay over distance and time
// - Cover evaluation scoring (exposure from each threat direction)
// - Path safety computation (integrate threat along path)
// - Predictive position estimation (where will enemy be in N seconds)
// - Flanking opportunity detection
// - Elevation advantage scoring
// - Line-of-fire analysis (can I shoot without hitting allies)
// - Environmental advantage map (choke points, high ground)
// - Visibility maps and line-of-sight checks
// - Tactical position evaluation

use std::collections::HashMap;

pub type EntityId = u64;

pub const GRID_DEFAULT_SIZE: u32 = 64;
pub const GRID_DEFAULT_CELL_SIZE: f32 = 2.0;
pub const MAX_DANGER_SOURCES: usize = 128;
pub const VISIBILITY_UPDATE_INTERVAL: f32 = 0.2;
pub const PREDICTION_HORIZON: f32 = 2.0;
pub const MAX_ALLIES: usize = 64;
pub const FLANKING_ANGLE_THRESHOLD: f32 = 60.0; // degrees
pub const ELEVATION_ADVANTAGE_PER_METER: f32 = 0.05;
pub const CHOKE_POINT_MIN_RATIO: f32 = 0.3;
pub const LINE_OF_FIRE_ALLY_RADIUS: f32 = 1.0;
pub const THREAT_TIME_DECAY_RATE: f32 = 0.2;
pub const COVER_SEARCH_RADIUS: f32 = 15.0;
pub const PATH_SAMPLE_DISTANCE: f32 = 1.0;

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }

    pub fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }

    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    pub fn normalize(self) -> Self {
        let l = self.length();
        if l > 1e-7 {
            self.scale(1.0 / l)
        } else {
            Self::ZERO
        }
    }

    pub fn distance(self, o: Self) -> f32 {
        self.sub(o).length()
    }

    pub fn distance_sq(self, o: Self) -> f32 {
        self.sub(o).length_sq()
    }

    pub fn lerp(self, o: Self, t: f32) -> Self {
        self.scale(1.0 - t).add(o.scale(t))
    }

    /// Horizontal distance (ignoring Y).
    pub fn distance_xz(self, o: Self) -> f32 {
        let dx = self.x - o.x;
        let dz = self.z - o.z;
        (dx * dx + dz * dz).sqrt()
    }

    /// Horizontal direction (normalized, Y=0).
    pub fn direction_xz(self, o: Self) -> Self {
        let d = Self::new(o.x - self.x, 0.0, o.z - self.z);
        d.normalize()
    }

    /// Angle between two vectors in radians.
    pub fn angle_between(self, other: Self) -> f32 {
        let d = self.dot(other) / (self.length() * other.length());
        d.clamp(-1.0, 1.0).acos()
    }

    /// Perpendicular vector in the XZ plane (rotated 90 degrees).
    pub fn perpendicular_xz(self) -> Self {
        Self::new(-self.z, 0.0, self.x)
    }

    /// Rotate around Y axis by angle in radians.
    pub fn rotate_y(self, angle: f32) -> Self {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        Self::new(
            self.x * cos_a + self.z * sin_a,
            self.y,
            -self.x * sin_a + self.z * cos_a,
        )
    }
}

// ---------------------------------------------------------------------------
// GridCoord
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridCoord {
    pub x: i32,
    pub z: i32,
}

impl GridCoord {
    pub fn new(x: i32, z: i32) -> Self {
        Self { x, z }
    }

    pub fn distance_sq(&self, other: &Self) -> i32 {
        let dx = self.x - other.x;
        let dz = self.z - other.z;
        dx * dx + dz * dz
    }

    pub fn manhattan_distance(&self, other: &Self) -> i32 {
        (self.x - other.x).abs() + (self.z - other.z).abs()
    }

    /// Get the 8 neighbors (including diagonals).
    pub fn neighbors_8(&self) -> [GridCoord; 8] {
        [
            GridCoord::new(self.x - 1, self.z - 1),
            GridCoord::new(self.x, self.z - 1),
            GridCoord::new(self.x + 1, self.z - 1),
            GridCoord::new(self.x - 1, self.z),
            GridCoord::new(self.x + 1, self.z),
            GridCoord::new(self.x - 1, self.z + 1),
            GridCoord::new(self.x, self.z + 1),
            GridCoord::new(self.x + 1, self.z + 1),
        ]
    }

    /// Get the 4 cardinal neighbors.
    pub fn neighbors_4(&self) -> [GridCoord; 4] {
        [
            GridCoord::new(self.x, self.z - 1),
            GridCoord::new(self.x - 1, self.z),
            GridCoord::new(self.x + 1, self.z),
            GridCoord::new(self.x, self.z + 1),
        ]
    }
}

// ---------------------------------------------------------------------------
// SpatialGrid
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SpatialGrid {
    pub width: u32,
    pub height: u32,
    pub cell_size: f32,
    pub origin: Vec3,
    pub cells: Vec<f32>,
}

impl SpatialGrid {
    pub fn new(width: u32, height: u32, cell_size: f32, origin: Vec3) -> Self {
        Self {
            width,
            height,
            cell_size,
            origin,
            cells: vec![0.0; (width * height) as usize],
        }
    }

    pub fn world_to_grid(&self, pos: Vec3) -> Option<GridCoord> {
        let lx = ((pos.x - self.origin.x) / self.cell_size) as i32;
        let lz = ((pos.z - self.origin.z) / self.cell_size) as i32;
        if lx >= 0 && lz >= 0 && (lx as u32) < self.width && (lz as u32) < self.height {
            Some(GridCoord::new(lx, lz))
        } else {
            None
        }
    }

    pub fn grid_to_world(&self, coord: GridCoord) -> Vec3 {
        Vec3::new(
            self.origin.x + coord.x as f32 * self.cell_size + self.cell_size * 0.5,
            self.origin.y,
            self.origin.z + coord.z as f32 * self.cell_size + self.cell_size * 0.5,
        )
    }

    fn index(&self, x: u32, z: u32) -> usize {
        (z * self.width + x) as usize
    }

    fn in_bounds(&self, x: i32, z: i32) -> bool {
        x >= 0 && z >= 0 && (x as u32) < self.width && (z as u32) < self.height
    }

    pub fn get(&self, x: u32, z: u32) -> f32 {
        if x < self.width && z < self.height {
            self.cells[self.index(x, z)]
        } else {
            0.0
        }
    }

    pub fn set(&mut self, x: u32, z: u32, val: f32) {
        if x < self.width && z < self.height {
            let idx = self.index(x, z);
            self.cells[idx] = val;
        }
    }

    pub fn get_at(&self, pos: Vec3) -> f32 {
        self.world_to_grid(pos)
            .map(|c| self.get(c.x as u32, c.z as u32))
            .unwrap_or(0.0)
    }

    pub fn set_at(&mut self, pos: Vec3, val: f32) {
        if let Some(c) = self.world_to_grid(pos) {
            self.set(c.x as u32, c.z as u32, val);
        }
    }

    pub fn clear(&mut self) {
        for c in &mut self.cells {
            *c = 0.0;
        }
    }

    /// Add influence centered at a position with inverse-square-law falloff.
    pub fn add_influence(&mut self, pos: Vec3, radius: f32, strength: f32) {
        let r_cells = (radius / self.cell_size).ceil() as i32;
        if let Some(center) = self.world_to_grid(pos) {
            for dz in -r_cells..=r_cells {
                for dx in -r_cells..=r_cells {
                    let gx = center.x + dx;
                    let gz = center.z + dz;
                    if self.in_bounds(gx, gz) {
                        let world = self.grid_to_world(GridCoord::new(gx, gz));
                        let dist = pos.distance(world);
                        if dist <= radius {
                            let falloff = 1.0 - dist / radius;
                            let idx = self.index(gx as u32, gz as u32);
                            self.cells[idx] += strength * falloff * falloff;
                        }
                    }
                }
            }
        }
    }

    /// Add influence with linear falloff.
    pub fn add_influence_linear(&mut self, pos: Vec3, radius: f32, strength: f32) {
        let r_cells = (radius / self.cell_size).ceil() as i32;
        if let Some(center) = self.world_to_grid(pos) {
            for dz in -r_cells..=r_cells {
                for dx in -r_cells..=r_cells {
                    let gx = center.x + dx;
                    let gz = center.z + dz;
                    if self.in_bounds(gx, gz) {
                        let world = self.grid_to_world(GridCoord::new(gx, gz));
                        let dist = pos.distance(world);
                        if dist <= radius {
                            let falloff = 1.0 - dist / radius;
                            let idx = self.index(gx as u32, gz as u32);
                            self.cells[idx] += strength * falloff;
                        }
                    }
                }
            }
        }
    }

    /// Add a directional cone of influence (e.g., weapon cone of fire).
    pub fn add_cone_influence(
        &mut self,
        origin: Vec3,
        direction: Vec3,
        cone_angle_rad: f32,
        range: f32,
        strength: f32,
    ) {
        let dir_norm = direction.normalize();
        let r_cells = (range / self.cell_size).ceil() as i32;
        if let Some(center) = self.world_to_grid(origin) {
            for dz in -r_cells..=r_cells {
                for dx in -r_cells..=r_cells {
                    let gx = center.x + dx;
                    let gz = center.z + dz;
                    if !self.in_bounds(gx, gz) {
                        continue;
                    }
                    let world = self.grid_to_world(GridCoord::new(gx, gz));
                    let to_cell = world.sub(origin);
                    let dist = to_cell.length();
                    if dist > range || dist < 0.01 {
                        continue;
                    }
                    let to_cell_norm = to_cell.normalize();
                    let angle = dir_norm.angle_between(to_cell_norm);
                    if angle <= cone_angle_rad {
                        let angle_falloff = 1.0 - angle / cone_angle_rad;
                        let dist_falloff = 1.0 - dist / range;
                        let idx = self.index(gx as u32, gz as u32);
                        self.cells[idx] += strength * angle_falloff * dist_falloff;
                    }
                }
            }
        }
    }

    /// Decay all values toward zero by a factor.
    pub fn decay_all(&mut self, factor: f32) {
        for c in &mut self.cells {
            *c *= factor;
        }
    }

    /// Blur the grid using a simple 3x3 averaging kernel.
    pub fn blur(&mut self) {
        let mut output = vec![0.0f32; self.cells.len()];
        for z in 0..self.height {
            for x in 0..self.width {
                let mut sum = 0.0f32;
                let mut count = 0u32;
                for dz in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = x as i32 + dx;
                        let nz = z as i32 + dz;
                        if self.in_bounds(nx, nz) {
                            sum += self.cells[self.index(nx as u32, nz as u32)];
                            count += 1;
                        }
                    }
                }
                output[self.index(x, z)] = sum / count as f32;
            }
        }
        self.cells = output;
    }

    pub fn max_value(&self) -> f32 {
        self.cells.iter().cloned().fold(f32::MIN, f32::max)
    }

    pub fn min_value(&self) -> f32 {
        self.cells.iter().cloned().fold(f32::MAX, f32::min)
    }

    /// Find the cell with the maximum value. Returns (coord, value).
    pub fn find_max(&self) -> (GridCoord, f32) {
        let mut best_val = f32::MIN;
        let mut best_coord = GridCoord::new(0, 0);
        for z in 0..self.height {
            for x in 0..self.width {
                let val = self.cells[self.index(x, z)];
                if val > best_val {
                    best_val = val;
                    best_coord = GridCoord::new(x as i32, z as i32);
                }
            }
        }
        (best_coord, best_val)
    }

    /// Find the cell with the minimum value.
    pub fn find_min(&self) -> (GridCoord, f32) {
        let mut best_val = f32::MAX;
        let mut best_coord = GridCoord::new(0, 0);
        for z in 0..self.height {
            for x in 0..self.width {
                let val = self.cells[self.index(x, z)];
                if val < best_val {
                    best_val = val;
                    best_coord = GridCoord::new(x as i32, z as i32);
                }
            }
        }
        (best_coord, best_val)
    }

    /// Bilinear interpolation at a world position.
    pub fn sample_bilinear(&self, pos: Vec3) -> f32 {
        let fx = (pos.x - self.origin.x) / self.cell_size - 0.5;
        let fz = (pos.z - self.origin.z) / self.cell_size - 0.5;

        let x0 = fx.floor() as i32;
        let z0 = fz.floor() as i32;
        let x1 = x0 + 1;
        let z1 = z0 + 1;

        let tx = fx - x0 as f32;
        let tz = fz - z0 as f32;

        let v00 = if self.in_bounds(x0, z0) { self.cells[self.index(x0 as u32, z0 as u32)] } else { 0.0 };
        let v10 = if self.in_bounds(x1, z0) { self.cells[self.index(x1 as u32, z0 as u32)] } else { 0.0 };
        let v01 = if self.in_bounds(x0, z1) { self.cells[self.index(x0 as u32, z1 as u32)] } else { 0.0 };
        let v11 = if self.in_bounds(x1, z1) { self.cells[self.index(x1 as u32, z1 as u32)] } else { 0.0 };

        let top = v00 + (v10 - v00) * tx;
        let bottom = v01 + (v11 - v01) * tx;
        top + (bottom - top) * tz
    }

    /// Compute the gradient (direction of steepest increase) at a position.
    pub fn gradient_at(&self, pos: Vec3) -> Vec3 {
        let coord = match self.world_to_grid(pos) {
            Some(c) => c,
            None => return Vec3::ZERO,
        };

        let x = coord.x as u32;
        let z = coord.z as u32;

        let vx_pos = if x + 1 < self.width { self.get(x + 1, z) } else { self.get(x, z) };
        let vx_neg = if x > 0 { self.get(x - 1, z) } else { self.get(x, z) };
        let vz_pos = if z + 1 < self.height { self.get(x, z + 1) } else { self.get(x, z) };
        let vz_neg = if z > 0 { self.get(x, z - 1) } else { self.get(x, z) };

        Vec3::new(
            (vx_pos - vx_neg) / (2.0 * self.cell_size),
            0.0,
            (vz_pos - vz_neg) / (2.0 * self.cell_size),
        )
    }

    /// Count cells above a threshold.
    pub fn count_above(&self, threshold: f32) -> u32 {
        self.cells.iter().filter(|&&v| v > threshold).count() as u32
    }
}

// ---------------------------------------------------------------------------
// HeightMap
// ---------------------------------------------------------------------------

/// A height map for elevation data.
#[derive(Debug, Clone)]
pub struct HeightMap {
    grid: SpatialGrid,
}

impl HeightMap {
    pub fn new(width: u32, height: u32, cell_size: f32, origin: Vec3) -> Self {
        Self {
            grid: SpatialGrid::new(width, height, cell_size, origin),
        }
    }

    pub fn set_height(&mut self, x: u32, z: u32, height: f32) {
        self.grid.set(x, z, height);
    }

    pub fn get_height(&self, pos: Vec3) -> f32 {
        self.grid.sample_bilinear(pos)
    }

    pub fn get_height_at_grid(&self, x: u32, z: u32) -> f32 {
        self.grid.get(x, z)
    }

    /// Compute the elevation advantage of position A over position B.
    /// Positive means A is higher. Capped at a maximum value.
    pub fn elevation_advantage(&self, pos_a: Vec3, pos_b: Vec3) -> f32 {
        let h_a = self.get_height(pos_a);
        let h_b = self.get_height(pos_b);
        let diff = h_a - h_b;
        (diff * ELEVATION_ADVANTAGE_PER_METER).clamp(-1.0, 1.0)
    }

    /// Get the slope at a position (0 = flat, 1 = very steep).
    pub fn slope_at(&self, pos: Vec3) -> f32 {
        let grad = self.grid.gradient_at(pos);
        grad.length().min(1.0)
    }
}

// ---------------------------------------------------------------------------
// DangerSource
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DangerSource {
    pub entity: EntityId,
    pub position: Vec3,
    pub danger_radius: f32,
    pub danger_strength: f32,
    pub velocity: Vec3,
    pub acceleration: Vec3,
    pub threat_level: f32,
    pub last_known_time: f32,
    /// Facing direction (for cone-of-fire calculations).
    pub facing: Vec3,
    /// Weapon range.
    pub weapon_range: f32,
    /// Weapon cone angle (radians, 0 = point, PI = omnidirectional).
    pub weapon_cone: f32,
    /// Whether this source is currently visible to us.
    pub visible: bool,
}

impl DangerSource {
    /// Simple linear prediction.
    pub fn predicted_position(&self, dt: f32) -> Vec3 {
        self.position
            .add(self.velocity.scale(dt))
            .add(self.acceleration.scale(0.5 * dt * dt))
    }

    /// Predict position using velocity with deceleration (enemy slowing down).
    pub fn predicted_position_decel(&self, dt: f32, decel_rate: f32) -> Vec3 {
        let speed = self.velocity.length();
        if speed < 0.01 {
            return self.position;
        }
        let direction = self.velocity.normalize();
        // v(t) = v0 - decel * t, clamped at 0
        let stop_time = speed / decel_rate;
        let effective_dt = dt.min(stop_time);
        let distance = speed * effective_dt - 0.5 * decel_rate * effective_dt * effective_dt;
        self.position.add(direction.scale(distance.max(0.0)))
    }

    /// Estimated time to reach a target position (assuming constant velocity).
    pub fn time_to_reach(&self, target: Vec3) -> f32 {
        let dist = self.position.distance(target);
        let speed = self.velocity.length();
        if speed < 0.01 {
            return f32::MAX;
        }
        dist / speed
    }

    /// Age of the last observation (how stale is this data).
    pub fn staleness(&self, current_time: f32) -> f32 {
        current_time - self.last_known_time
    }
}

impl Default for DangerSource {
    fn default() -> Self {
        Self {
            entity: 0,
            position: Vec3::ZERO,
            danger_radius: 10.0,
            danger_strength: 1.0,
            velocity: Vec3::ZERO,
            acceleration: Vec3::ZERO,
            threat_level: 1.0,
            last_known_time: 0.0,
            facing: Vec3::new(0.0, 0.0, 1.0),
            weapon_range: 20.0,
            weapon_cone: std::f32::consts::PI * 0.5,
            visible: true,
        }
    }
}

// ---------------------------------------------------------------------------
// AllyInfo
// ---------------------------------------------------------------------------

/// Information about a friendly entity (for line-of-fire checks).
#[derive(Debug, Clone)]
pub struct AllyInfo {
    pub entity: EntityId,
    pub position: Vec3,
    pub radius: f32,
}

// ---------------------------------------------------------------------------
// CoverPoint
// ---------------------------------------------------------------------------

/// A cover point in the environment.
#[derive(Debug, Clone)]
pub struct CoverPoint {
    pub position: Vec3,
    /// Normal direction of the cover (the direction the cover faces, i.e., the "safe" side).
    pub normal: Vec3,
    /// Height of cover (half = crouch, full = standing).
    pub height: CoverHeight,
    /// Width of the cover object (meters).
    pub width: f32,
    /// Whether the cover is destructible.
    pub destructible: bool,
    /// Current health (if destructible).
    pub health: f32,
    /// Whether this cover point is currently occupied by someone.
    pub occupied: bool,
    pub occupant: Option<EntityId>,
}

/// Height classification of cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverHeight {
    /// Low cover (must crouch).
    Half,
    /// Full-height cover (standing).
    Full,
}

impl CoverPoint {
    pub fn new(position: Vec3, normal: Vec3, height: CoverHeight) -> Self {
        Self {
            position,
            normal: normal.normalize(),
            height,
            width: 2.0,
            destructible: false,
            health: 100.0,
            occupied: false,
            occupant: None,
        }
    }

    /// Check if this cover protects against a threat at the given position.
    /// Cover protects if the threat is on the opposite side of the cover normal.
    pub fn protects_from(&self, threat_pos: Vec3) -> bool {
        let to_threat = threat_pos.sub(self.position).normalize();
        // Cover protects if the threat is behind the cover (dot product with normal < 0).
        let dot = to_threat.dot(self.normal);
        dot < 0.0
    }

    /// Compute the protection quality against a specific threat (0 = no protection, 1 = full).
    pub fn protection_quality(&self, threat_pos: Vec3) -> f32 {
        let to_threat = threat_pos.sub(self.position);
        let dist = to_threat.length();
        if dist < 0.01 {
            return 0.0;
        }
        let to_threat_norm = to_threat.normalize();

        // How much the threat is behind the cover.
        let alignment = -to_threat_norm.dot(self.normal);
        if alignment <= 0.0 {
            return 0.0; // Threat is on the same side as us.
        }

        // Angle-based coverage.
        let angle_quality = alignment.clamp(0.0, 1.0);

        // Width-based coverage: wider cover is better.
        let lateral_coverage = (self.width / 2.0 / dist.max(1.0)).min(1.0);

        // Height factor.
        let height_factor = match self.height {
            CoverHeight::Full => 1.0,
            CoverHeight::Half => 0.6,
        };

        angle_quality * lateral_coverage * height_factor
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct ExposureResult {
    pub exposure_score: f32,
    pub visible_threats: u32,
    pub nearest_cover_distance: f32,
    pub nearest_cover_direction: Vec3,
    pub exposed_angle_ratio: f32,
    /// Per-direction exposure (N, NE, E, SE, S, SW, W, NW).
    pub directional_exposure: [f32; 8],
}

#[derive(Debug, Clone, Copy)]
pub struct DefensiveScore {
    pub total_score: f32,
    pub cover_quality: f32,
    pub sight_lines: f32,
    pub escape_routes: f32,
    pub elevation_advantage: f32,
    pub flank_protection: f32,
    pub choke_point_value: f32,
}

#[derive(Debug, Clone)]
pub struct PathSafety {
    pub overall_safety: f32,
    pub max_danger: f32,
    pub avg_danger: f32,
    pub exposed_segments: u32,
    pub total_segments: u32,
    pub safest_alternative: Option<Vec3>,
    /// Danger values at each sampled point along the path.
    pub danger_profile: Vec<f32>,
    /// Most dangerous point on the path.
    pub danger_peak_position: Option<Vec3>,
}

/// Result of a flanking opportunity analysis.
#[derive(Debug, Clone)]
pub struct FlankingOpportunity {
    /// Position to move to for flanking.
    pub position: Vec3,
    /// Target entity being flanked.
    pub target: EntityId,
    /// Angle of approach relative to target's facing (degrees). 90+ is a good flank.
    pub flank_angle: f32,
    /// How safe the flanking route is.
    pub route_safety: f32,
    /// Distance to the flanking position.
    pub distance: f32,
    /// Quality score (higher is better).
    pub quality: f32,
}

/// Result of line-of-fire analysis.
#[derive(Debug, Clone)]
pub struct LineOfFireResult {
    /// Whether the line of fire is clear (no allies in the way).
    pub clear: bool,
    /// Allies in or near the line of fire.
    pub endangered_allies: Vec<EntityId>,
    /// Closest distance an ally is to the fire line.
    pub closest_ally_distance: f32,
    /// Safety margin (how far the nearest ally is from the fire line, normalized).
    pub safety_margin: f32,
}

/// Environmental advantage analysis for a position.
#[derive(Debug, Clone)]
pub struct EnvironmentalAdvantage {
    pub total_score: f32,
    pub is_choke_point: bool,
    pub is_high_ground: bool,
    pub nearby_cover_count: u32,
    pub escape_routes: u32,
    pub open_angles: f32,
    pub elevation: f32,
}

// ---------------------------------------------------------------------------
// SpatialAwareness
// ---------------------------------------------------------------------------

pub struct SpatialAwareness {
    danger_map: SpatialGrid,
    visibility_map: SpatialGrid,
    cover_map: SpatialGrid,
    tactical_map: SpatialGrid,
    flanking_map: SpatialGrid,
    height_map: Option<HeightMap>,
    danger_sources: Vec<DangerSource>,
    allies: Vec<AllyInfo>,
    cover_points: Vec<CoverPoint>,
    self_position: Vec3,
    self_forward: Vec3,
    self_entity: EntityId,
    grid_origin: Vec3,
    grid_size: u32,
    cell_size: f32,
    time: f32,
    update_timer: f32,
    /// Threat map with temporal decay (keeps memory of old threats).
    threat_memory: SpatialGrid,
    /// Passability map (0 = blocked, 1 = open).
    passability_map: SpatialGrid,
}

impl SpatialAwareness {
    pub fn new(origin: Vec3, grid_size: u32, cell_size: f32) -> Self {
        Self {
            danger_map: SpatialGrid::new(grid_size, grid_size, cell_size, origin),
            visibility_map: SpatialGrid::new(grid_size, grid_size, cell_size, origin),
            cover_map: SpatialGrid::new(grid_size, grid_size, cell_size, origin),
            tactical_map: SpatialGrid::new(grid_size, grid_size, cell_size, origin),
            flanking_map: SpatialGrid::new(grid_size, grid_size, cell_size, origin),
            height_map: None,
            danger_sources: Vec::new(),
            allies: Vec::new(),
            cover_points: Vec::new(),
            self_position: Vec3::ZERO,
            self_forward: Vec3::new(0.0, 0.0, 1.0),
            self_entity: 0,
            grid_origin: origin,
            grid_size,
            cell_size,
            time: 0.0,
            update_timer: 0.0,
            threat_memory: SpatialGrid::new(grid_size, grid_size, cell_size, origin),
            passability_map: SpatialGrid::new(grid_size, grid_size, cell_size, origin),
        }
    }

    // --- Setters ---

    pub fn set_self_position(&mut self, pos: Vec3, forward: Vec3) {
        self.self_position = pos;
        self.self_forward = forward.normalize();
    }

    pub fn set_self_entity(&mut self, entity: EntityId) {
        self.self_entity = entity;
    }

    pub fn set_height_map(&mut self, height_map: HeightMap) {
        self.height_map = Some(height_map);
    }

    pub fn add_danger_source(&mut self, source: DangerSource) {
        if self.danger_sources.len() < MAX_DANGER_SOURCES {
            self.danger_sources.push(source);
        }
    }

    pub fn update_danger_source(&mut self, entity: EntityId, position: Vec3, velocity: Vec3) {
        if let Some(source) = self.danger_sources.iter_mut().find(|s| s.entity == entity) {
            source.position = position;
            source.velocity = velocity;
            source.last_known_time = self.time;
            source.visible = true;
        }
    }

    pub fn clear_danger_sources(&mut self) {
        self.danger_sources.clear();
    }

    pub fn add_ally(&mut self, ally: AllyInfo) {
        if self.allies.len() < MAX_ALLIES {
            self.allies.push(ally);
        }
    }

    pub fn clear_allies(&mut self) {
        self.allies.clear();
    }

    pub fn add_cover_point(&mut self, cover: CoverPoint) {
        self.cover_points.push(cover);
    }

    pub fn clear_cover_points(&mut self) {
        self.cover_points.clear();
    }

    pub fn set_passability(&mut self, x: u32, z: u32, passable: f32) {
        self.passability_map.set(x, z, passable);
    }

    // --- Update ---

    pub fn update(&mut self, dt: f32) {
        self.time += dt;
        self.update_timer += dt;
        if self.update_timer >= VISIBILITY_UPDATE_INTERVAL {
            self.update_timer = 0.0;
            self.rebuild_danger_map();
            self.rebuild_cover_map();
            self.rebuild_tactical_map();
            self.rebuild_flanking_map();
            self.update_threat_memory(dt);
        }
    }

    fn rebuild_danger_map(&mut self) {
        self.danger_map.clear();
        for source in &self.danger_sources {
            // Decay threat based on staleness.
            let staleness = source.staleness(self.time);
            let time_decay = (-THREAT_TIME_DECAY_RATE * staleness).exp();

            // Predict near-future position.
            let predicted = source.predicted_position(0.5);

            // Add omnidirectional danger.
            let effective_strength = source.danger_strength * source.threat_level * time_decay;
            self.danger_map.add_influence(predicted, source.danger_radius, effective_strength);

            // Add cone-of-fire danger if the source has a weapon.
            if source.weapon_range > 0.0 && source.weapon_cone > 0.0 {
                self.danger_map.add_cone_influence(
                    source.position,
                    source.facing,
                    source.weapon_cone,
                    source.weapon_range,
                    effective_strength * 0.5,
                );
            }

            // Add predicted future positions at longer time horizons.
            for t in [1.0, 1.5, 2.0] {
                let future = source.predicted_position(t);
                let future_strength = effective_strength * (1.0 - t / 3.0).max(0.1);
                self.danger_map.add_influence(future, source.danger_radius * 0.5, future_strength * 0.3);
            }
        }
    }

    fn rebuild_cover_map(&mut self) {
        self.cover_map.clear();
        for cp in &self.cover_points {
            let cover_value = match cp.height {
                CoverHeight::Full => 1.0,
                CoverHeight::Half => 0.6,
            };
            let radius = cp.width.max(self.cell_size * 1.5);
            self.cover_map.add_influence(cp.position, radius, cover_value);
        }
    }

    fn rebuild_tactical_map(&mut self) {
        self.tactical_map.clear();
        // Tactical map combines danger, cover, and elevation into a single "tactical value" map.
        for z in 0..self.grid_size {
            for x in 0..self.grid_size {
                let danger = self.danger_map.get(x, z);
                let cover = self.cover_map.get(x, z);
                let elevation = self.height_map.as_ref()
                    .map(|h| h.get_height_at_grid(x, z))
                    .unwrap_or(0.0);

                // Higher value = better tactical position.
                let tactical_value = cover * 0.3
                    + (1.0 - danger.min(1.0)) * 0.4
                    + (elevation * ELEVATION_ADVANTAGE_PER_METER).min(0.3) * 0.3;

                self.tactical_map.set(x, z, tactical_value);
            }
        }
    }

    fn rebuild_flanking_map(&mut self) {
        self.flanking_map.clear();
        if self.danger_sources.is_empty() {
            return;
        }

        // For each threat, mark cells that are to the side or behind them
        // (relative to their facing direction) as flanking opportunities.
        for source in &self.danger_sources {
            let facing = source.facing.normalize();
            let right = facing.perpendicular_xz();

            for z in 0..self.grid_size {
                for x in 0..self.grid_size {
                    let world = self.flanking_map.grid_to_world(GridCoord::new(x as i32, z as i32));
                    let to_cell = world.sub(source.position);
                    let dist = to_cell.length();

                    if dist < 2.0 || dist > source.weapon_range * 1.5 {
                        continue;
                    }

                    let to_cell_norm = to_cell.normalize();
                    let forward_dot = to_cell_norm.dot(facing);
                    let right_dot = to_cell_norm.dot(right).abs();

                    // Flanking is best when we are to the side (high right_dot) or behind (negative forward_dot).
                    let flank_value = if forward_dot < -0.3 {
                        // Behind the enemy.
                        0.8 + right_dot * 0.2
                    } else if right_dot > 0.5 {
                        // To the side.
                        right_dot * 0.6
                    } else {
                        0.0
                    };

                    // Reduce flanking value if position is dangerous.
                    let danger = self.danger_map.get(x, z);
                    let adjusted = flank_value * (1.0 - danger.min(1.0) * 0.5);

                    let idx = (z * self.grid_size + x) as usize;
                    self.flanking_map.cells[idx] += adjusted;
                }
            }
        }
    }

    fn update_threat_memory(&mut self, dt: f32) {
        // Decay old threat memory.
        let decay_factor = (-THREAT_TIME_DECAY_RATE * dt * 5.0).exp();
        self.threat_memory.decay_all(decay_factor);

        // Add current danger to memory.
        for i in 0..self.threat_memory.cells.len() {
            self.threat_memory.cells[i] = self.threat_memory.cells[i].max(self.danger_map.cells[i] * 0.5);
        }
    }

    // --- Queries ---

    pub fn get_danger_at(&self, pos: Vec3) -> f32 {
        self.danger_map.get_at(pos)
    }

    pub fn get_tactical_value_at(&self, pos: Vec3) -> f32 {
        self.tactical_map.get_at(pos)
    }

    pub fn get_flanking_value_at(&self, pos: Vec3) -> f32 {
        self.flanking_map.get_at(pos)
    }

    pub fn get_threat_memory_at(&self, pos: Vec3) -> f32 {
        self.threat_memory.get_at(pos)
    }

    /// Evaluate the path safety along a series of waypoints.
    pub fn evaluate_path_safety(&self, path: &[Vec3]) -> PathSafety {
        if path.is_empty() {
            return PathSafety {
                overall_safety: 1.0,
                max_danger: 0.0,
                avg_danger: 0.0,
                exposed_segments: 0,
                total_segments: 0,
                safest_alternative: None,
                danger_profile: Vec::new(),
                danger_peak_position: None,
            };
        }

        // Sample points along the path at regular intervals.
        let mut sample_points: Vec<Vec3> = Vec::new();
        for i in 0..path.len() - 1 {
            let start = path[i];
            let end = path[i + 1];
            let segment_dist = start.distance(end);
            let num_samples = (segment_dist / PATH_SAMPLE_DISTANCE).ceil() as usize;
            for s in 0..=num_samples {
                let t = s as f32 / num_samples.max(1) as f32;
                sample_points.push(start.lerp(end, t));
            }
        }
        if sample_points.is_empty() {
            sample_points.push(path[0]);
        }

        let mut total_danger = 0.0f32;
        let mut max_danger = 0.0f32;
        let mut max_danger_pos = None;
        let mut exposed = 0u32;
        let mut danger_profile = Vec::new();

        for &p in &sample_points {
            let d = self.danger_map.sample_bilinear(p);
            danger_profile.push(d);
            total_danger += d;
            if d > max_danger {
                max_danger = d;
                max_danger_pos = Some(p);
            }
            if d > 0.5 {
                exposed += 1;
            }
        }

        let n = sample_points.len() as f32;
        let avg = total_danger / n;

        // Find a safer alternative: look at cells perpendicular to the most dangerous segment.
        let safest_alt = if let Some(peak_pos) = max_danger_pos {
            let perp = self.self_forward.perpendicular_xz().normalize();
            let left = peak_pos.add(perp.scale(5.0));
            let right = peak_pos.sub(perp.scale(5.0));
            let d_left = self.danger_map.get_at(left);
            let d_right = self.danger_map.get_at(right);
            if d_left < max_danger * 0.5 {
                Some(left)
            } else if d_right < max_danger * 0.5 {
                Some(right)
            } else {
                None
            }
        } else {
            None
        };

        PathSafety {
            overall_safety: (1.0 - avg).max(0.0),
            max_danger,
            avg_danger: avg,
            exposed_segments: exposed,
            total_segments: sample_points.len() as u32,
            safest_alternative: safest_alt,
            danger_profile,
            danger_peak_position: max_danger_pos,
        }
    }

    /// Evaluate exposure at a position from all directions.
    pub fn evaluate_exposure(&self, pos: Vec3) -> ExposureResult {
        let danger = self.danger_map.sample_bilinear(pos);
        let visible = self.danger_sources.iter()
            .filter(|s| s.position.distance(pos) < s.danger_radius * 1.5 && s.visible)
            .count() as u32;

        // Find nearest cover.
        let mut nearest_dist = f32::MAX;
        let mut nearest_dir = Vec3::ZERO;
        for cp in &self.cover_points {
            let d = pos.distance(cp.position);
            if d < nearest_dist {
                nearest_dist = d;
                nearest_dir = cp.position.sub(pos).normalize();
            }
        }

        // Compute directional exposure (8 directions).
        let mut directional_exposure = [0.0f32; 8];
        let directions = [
            Vec3::new(0.0, 0.0, 1.0),   // N
            Vec3::new(0.7, 0.0, 0.7),   // NE (approx normalized)
            Vec3::new(1.0, 0.0, 0.0),   // E
            Vec3::new(0.7, 0.0, -0.7),  // SE
            Vec3::new(0.0, 0.0, -1.0),  // S
            Vec3::new(-0.7, 0.0, -0.7), // SW
            Vec3::new(-1.0, 0.0, 0.0),  // W
            Vec3::new(-0.7, 0.0, 0.7),  // NW
        ];

        let half_pi = std::f32::consts::PI * 0.5;
        for source in &self.danger_sources {
            let to_threat = source.position.sub(pos);
            let dist = to_threat.length();
            if dist < 0.1 || dist > source.danger_radius * 2.0 {
                continue;
            }
            let to_threat_norm = to_threat.normalize();
            let threat_contribution = source.threat_level * (1.0 - dist / (source.danger_radius * 2.0)).max(0.0);

            for (i, dir) in directions.iter().enumerate() {
                let angle = dir.angle_between(to_threat_norm);
                if angle < half_pi {
                    let factor = 1.0 - angle / half_pi;
                    directional_exposure[i] += threat_contribution * factor;

                    // Reduce exposure if cover protects from this direction.
                    for cp in &self.cover_points {
                        if cp.position.distance(pos) < 3.0 && cp.protects_from(source.position) {
                            directional_exposure[i] *= 1.0 - cp.protection_quality(source.position);
                        }
                    }
                }
            }
        }

        // Compute exposed angle ratio: what fraction of angles have significant exposure.
        let exposed_dirs = directional_exposure.iter().filter(|&&e| e > 0.3).count();
        let exposed_ratio = exposed_dirs as f32 / 8.0;

        ExposureResult {
            exposure_score: danger,
            visible_threats: visible,
            nearest_cover_distance: nearest_dist,
            nearest_cover_direction: nearest_dir,
            exposed_angle_ratio: exposed_ratio,
            directional_exposure,
        }
    }

    /// Score a position for defensive value.
    pub fn score_defensive_position(&self, pos: Vec3) -> DefensiveScore {
        let danger = self.danger_map.sample_bilinear(pos);
        let cover = self.cover_map.sample_bilinear(pos);

        // Elevation advantage against the nearest threat.
        let elevation_adv = if let Some(height_map) = &self.height_map {
            self.danger_sources.iter()
                .map(|s| height_map.elevation_advantage(pos, s.position))
                .fold(0.0f32, |acc, e| acc + e.max(0.0))
                / self.danger_sources.len().max(1) as f32
        } else {
            0.0
        };

        // Escape routes: count cover points and open directions near this position.
        let nearby_cover = self.cover_points.iter()
            .filter(|cp| cp.position.distance(pos) < COVER_SEARCH_RADIUS && !cp.occupied)
            .count() as f32;
        let escape_score = (nearby_cover / 5.0).min(1.0);

        // Flank protection: check if threats can easily get behind us.
        let mut flank_protection = 1.0f32;
        for source in &self.danger_sources {
            let to_threat = source.position.sub(pos).normalize();
            let dot = to_threat.dot(self.self_forward);
            if dot < -0.3 {
                // Threat is behind us = poor flank protection.
                flank_protection -= 0.3;
            }
        }
        flank_protection = flank_protection.clamp(0.0, 1.0);

        // Check if this is a choke point.
        let choke_value = self.evaluate_choke_point(pos);

        let cover_quality = cover.min(1.0);
        let sight_lines = 1.0 - danger.min(1.0);

        let total = cover_quality * 0.25
            + sight_lines * 0.20
            + elevation_adv * 0.15
            + escape_score * 0.15
            + flank_protection * 0.15
            + choke_value * 0.10;

        DefensiveScore {
            total_score: total,
            cover_quality,
            sight_lines,
            escape_routes: escape_score,
            elevation_advantage: elevation_adv,
            flank_protection,
            choke_point_value: choke_value,
        }
    }

    /// Evaluate whether a position is a choke point.
    /// A choke point has restricted passability around it, funneling movement.
    fn evaluate_choke_point(&self, pos: Vec3) -> f32 {
        let coord = match self.passability_map.world_to_grid(pos) {
            Some(c) => c,
            None => return 0.0,
        };

        // Count passable vs blocked cells in a ring around this position.
        let check_radius = 3;
        let mut passable = 0u32;
        let mut blocked = 0u32;

        for dz in -check_radius..=check_radius {
            for dx in -check_radius..=check_radius {
                if dx == 0 && dz == 0 {
                    continue;
                }
                let cx = coord.x + dx;
                let cz = coord.z + dz;
                if self.passability_map.in_bounds(cx, cz) {
                    if self.passability_map.get(cx as u32, cz as u32) > 0.5 {
                        passable += 1;
                    } else {
                        blocked += 1;
                    }
                }
            }
        }

        let total = passable + blocked;
        if total == 0 {
            return 0.0;
        }

        let passable_ratio = passable as f32 / total as f32;
        // A choke point has a low passable ratio (lots of walls/obstacles).
        if passable_ratio < CHOKE_POINT_MIN_RATIO {
            (1.0 - passable_ratio / CHOKE_POINT_MIN_RATIO).min(1.0)
        } else {
            0.0
        }
    }

    // --- Predictive Position Estimation ---

    /// Predict where an entity will be at a future time.
    pub fn predict_position(current: Vec3, velocity: Vec3, acceleration: Vec3, time: f32) -> Vec3 {
        current
            .add(velocity.scale(time))
            .add(acceleration.scale(0.5 * time * time))
    }

    /// Predict entity position with multiple models and return the most likely.
    pub fn predict_position_multi(
        &self,
        source: &DangerSource,
        time_ahead: f32,
    ) -> Vec3 {
        // Model 1: Linear extrapolation.
        let linear = source.predicted_position(time_ahead);

        // Model 2: With deceleration.
        let decel = source.predicted_position_decel(time_ahead, 2.0);

        // Model 3: Toward us (assumes aggressive approach).
        let toward_us = {
            let dir = self.self_position.sub(source.position).normalize();
            let speed = source.velocity.length();
            source.position.add(dir.scale(speed * time_ahead))
        };

        // Weight models based on behavior.
        let speed = source.velocity.length();
        if speed < 0.5 {
            // Stationary: predict they stay put.
            source.position
        } else {
            // Moving: blend linear and toward-us.
            let approach_dot = source.velocity.normalize().dot(
                self.self_position.sub(source.position).normalize()
            );
            if approach_dot > 0.5 {
                // Moving toward us: weight toward-us model higher.
                linear.lerp(toward_us, 0.6)
            } else {
                // Moving away or laterally: weight linear + decel.
                linear.lerp(decel, 0.4)
            }
        }
    }

    // --- Flanking Opportunity Detection ---

    /// Find flanking opportunities against the threats.
    pub fn find_flanking_opportunities(&self) -> Vec<FlankingOpportunity> {
        let mut opportunities = Vec::new();

        for source in &self.danger_sources {
            let facing = source.facing.normalize();
            let to_threat = source.position.sub(self.self_position);
            let threat_dist = to_threat.length();

            if threat_dist < 2.0 || threat_dist > 50.0 {
                continue;
            }

            // Check candidate positions to the side and behind the enemy.
            let right = facing.perpendicular_xz().normalize();
            let candidates = [
                // Left flank.
                source.position.add(right.scale(-8.0)),
                source.position.add(right.scale(-12.0)),
                // Right flank.
                source.position.add(right.scale(8.0)),
                source.position.add(right.scale(12.0)),
                // Rear flank.
                source.position.sub(facing.scale(8.0)),
                source.position.sub(facing.scale(12.0)),
                // Diagonal flanks.
                source.position.add(right.scale(-6.0)).sub(facing.scale(6.0)),
                source.position.add(right.scale(6.0)).sub(facing.scale(6.0)),
            ];

            for &candidate in &candidates {
                let danger = self.danger_map.get_at(candidate);
                if danger > 0.8 {
                    continue; // Too dangerous.
                }

                let to_candidate = candidate.sub(source.position).normalize();
                let flank_angle = to_candidate.angle_between(facing).to_degrees();

                if flank_angle < FLANKING_ANGLE_THRESHOLD {
                    continue; // Not enough of a flank.
                }

                let distance = self.self_position.distance(candidate);

                // Evaluate the route safety (simplified: check a straight line).
                let route_points = [
                    self.self_position,
                    self.self_position.lerp(candidate, 0.25),
                    self.self_position.lerp(candidate, 0.5),
                    self.self_position.lerp(candidate, 0.75),
                    candidate,
                ];
                let route_safety = self.evaluate_path_safety(&route_points);

                let quality = flank_angle / 180.0 * 0.4
                    + route_safety.overall_safety * 0.3
                    + (1.0 - danger.min(1.0)) * 0.3;

                opportunities.push(FlankingOpportunity {
                    position: candidate,
                    target: source.entity,
                    flank_angle,
                    route_safety: route_safety.overall_safety,
                    distance,
                    quality,
                });
            }
        }

        // Sort by quality descending.
        opportunities.sort_by(|a, b| b.quality.partial_cmp(&a.quality).unwrap_or(std::cmp::Ordering::Equal));
        opportunities
    }

    // --- Line of Fire Analysis ---

    /// Check if the line of fire from our position to a target is clear of allies.
    pub fn check_line_of_fire(&self, target_pos: Vec3) -> LineOfFireResult {
        self.check_line_of_fire_from(self.self_position, target_pos)
    }

    /// Check if the line of fire from a specific position to a target is clear.
    pub fn check_line_of_fire_from(&self, from: Vec3, to: Vec3) -> LineOfFireResult {
        let fire_dir = to.sub(from);
        let fire_dist = fire_dir.length();
        if fire_dist < 0.1 {
            return LineOfFireResult {
                clear: true,
                endangered_allies: Vec::new(),
                closest_ally_distance: f32::MAX,
                safety_margin: 1.0,
            };
        }
        let fire_norm = fire_dir.normalize();

        let mut endangered = Vec::new();
        let mut closest_dist = f32::MAX;

        for ally in &self.allies {
            if ally.entity == self.self_entity {
                continue;
            }

            // Compute the closest distance from the ally to the line of fire.
            let to_ally = ally.position.sub(from);
            let projection = to_ally.dot(fire_norm);

            // Only check allies that are between us and the target.
            if projection < 0.0 || projection > fire_dist {
                continue;
            }

            let closest_point = from.add(fire_norm.scale(projection));
            let lateral_dist = ally.position.distance(closest_point);

            // Check if ally is within danger radius of the fire line.
            let danger_radius = ally.radius + LINE_OF_FIRE_ALLY_RADIUS;
            if lateral_dist < danger_radius {
                endangered.push(ally.entity);
            }

            closest_dist = closest_dist.min(lateral_dist);
        }

        let safety = if closest_dist < f32::MAX {
            (closest_dist / (LINE_OF_FIRE_ALLY_RADIUS * 3.0)).min(1.0)
        } else {
            1.0
        };

        LineOfFireResult {
            clear: endangered.is_empty(),
            endangered_allies: endangered,
            closest_ally_distance: closest_dist,
            safety_margin: safety,
        }
    }

    // --- Environmental Advantage ---

    /// Analyze the environmental advantage at a position.
    pub fn evaluate_environment(&self, pos: Vec3) -> EnvironmentalAdvantage {
        let elevation = self.height_map.as_ref()
            .map(|h| h.get_height(pos))
            .unwrap_or(pos.y);

        let nearby_cover = self.cover_points.iter()
            .filter(|cp| cp.position.distance(pos) < COVER_SEARCH_RADIUS)
            .count() as u32;

        let is_choke = self.evaluate_choke_point(pos) > 0.3;
        let is_high_ground = self.danger_sources.iter()
            .all(|s| {
                self.height_map.as_ref()
                    .map(|h| h.elevation_advantage(pos, s.position) > 0.0)
                    .unwrap_or(false)
            }) && !self.danger_sources.is_empty();

        // Count escape routes (open directions we can retreat to).
        let directions = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ];
        let mut escape_routes = 0u32;
        for dir in &directions {
            let test_pos = pos.add(dir.scale(5.0));
            let danger = self.danger_map.get_at(test_pos);
            let passable = self.passability_map.get_at(test_pos);
            if danger < 0.3 && passable > 0.5 {
                escape_routes += 1;
            }
        }

        // Open angles: what fraction of 360 degrees is clear of obstacles.
        let open_angle_samples = 16;
        let mut open_count = 0u32;
        for i in 0..open_angle_samples {
            let angle = (i as f32 / open_angle_samples as f32) * std::f32::consts::PI * 2.0;
            let dir = Vec3::new(angle.cos(), 0.0, angle.sin());
            let test_pos = pos.add(dir.scale(3.0));
            if self.passability_map.get_at(test_pos) > 0.5 {
                open_count += 1;
            }
        }
        let open_angles = open_count as f32 / open_angle_samples as f32;

        let total_score = if is_choke { 0.3 } else { 0.0 }
            + if is_high_ground { 0.25 } else { 0.0 }
            + (nearby_cover as f32 / 5.0).min(0.2)
            + (escape_routes as f32 / 4.0) * 0.15
            + open_angles * 0.1;

        EnvironmentalAdvantage {
            total_score,
            is_choke_point: is_choke,
            is_high_ground,
            nearby_cover_count: nearby_cover,
            escape_routes,
            open_angles,
            elevation,
        }
    }

    // --- Finding best positions ---

    pub fn find_safest_position(&self, candidates: &[Vec3]) -> Option<(Vec3, f32)> {
        candidates.iter()
            .map(|&pos| {
                let score = self.score_defensive_position(pos);
                (pos, score.total_score)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Find the best cover point considering threats and accessibility.
    pub fn find_best_cover(&self) -> Option<(Vec3, f32)> {
        let mut best: Option<(Vec3, f32)> = None;

        for cp in &self.cover_points {
            if cp.occupied {
                continue;
            }

            let dist = self.self_position.distance(cp.position);
            if dist > COVER_SEARCH_RADIUS * 2.0 {
                continue;
            }

            // How well does this cover protect against current threats?
            let mut total_protection = 0.0f32;
            for source in &self.danger_sources {
                total_protection += cp.protection_quality(source.position);
            }
            let avg_protection = if !self.danger_sources.is_empty() {
                total_protection / self.danger_sources.len() as f32
            } else {
                0.5
            };

            // Closer cover is better (but not too close to threats).
            let dist_score = (1.0 - dist / (COVER_SEARCH_RADIUS * 2.0)).max(0.0);

            // Check danger at the cover position.
            let danger = self.danger_map.get_at(cp.position);
            let danger_penalty = danger.min(1.0);

            let score = avg_protection * 0.4 + dist_score * 0.3 + (1.0 - danger_penalty) * 0.3;

            match &best {
                Some((_, best_score)) if score <= *best_score => {}
                _ => best = Some((cp.position, score)),
            }
        }

        best
    }

    /// Find the best flanking position.
    pub fn find_best_flank(&self) -> Option<FlankingOpportunity> {
        let opportunities = self.find_flanking_opportunities();
        opportunities.into_iter().next()
    }

    // --- Map accessors ---

    pub fn danger_map(&self) -> &SpatialGrid {
        &self.danger_map
    }

    pub fn visibility_map(&self) -> &SpatialGrid {
        &self.visibility_map
    }

    pub fn cover_map(&self) -> &SpatialGrid {
        &self.cover_map
    }

    pub fn tactical_map(&self) -> &SpatialGrid {
        &self.tactical_map
    }

    pub fn flanking_map(&self) -> &SpatialGrid {
        &self.flanking_map
    }

    pub fn threat_memory_map(&self) -> &SpatialGrid {
        &self.threat_memory
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_grid() {
        let mut grid = SpatialGrid::new(16, 16, 1.0, Vec3::ZERO);
        grid.set(5, 5, 1.0);
        assert!((grid.get(5, 5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_danger_influence() {
        let mut grid = SpatialGrid::new(16, 16, 1.0, Vec3::ZERO);
        grid.add_influence(Vec3::new(8.0, 0.0, 8.0), 3.0, 1.0);
        assert!(grid.get(8, 8) > 0.0);
        // Further away should have less influence.
        assert!(grid.get(8, 8) > grid.get(10, 10));
    }

    #[test]
    fn test_cone_influence() {
        let mut grid = SpatialGrid::new(32, 32, 1.0, Vec3::ZERO);
        grid.add_cone_influence(
            Vec3::new(16.0, 0.0, 16.0),
            Vec3::new(1.0, 0.0, 0.0),
            0.5,
            10.0,
            1.0,
        );
        // Cell in front should have influence.
        assert!(grid.get(20, 16) > 0.0);
        // Cell behind should have no influence.
        assert!((grid.get(12, 16)).abs() < 1e-6);
    }

    #[test]
    fn test_prediction() {
        let pos = SpatialAwareness::predict_position(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::ZERO,
            2.0,
        );
        assert!((pos.x - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_prediction_with_acceleration() {
        let pos = SpatialAwareness::predict_position(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            1.0,
        );
        // x = v*t + 0.5*a*t^2 = 1.0 + 1.0 = 2.0
        assert!((pos.x - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_spatial_awareness() {
        let mut sa = SpatialAwareness::new(Vec3::ZERO, 32, 1.0);
        sa.add_danger_source(DangerSource {
            entity: 1,
            position: Vec3::new(10.0, 0.0, 10.0),
            danger_radius: 5.0,
            danger_strength: 1.0,
            velocity: Vec3::ZERO,
            threat_level: 1.0,
            last_known_time: 0.0,
            ..Default::default()
        });
        sa.update(0.3);
        let danger = sa.get_danger_at(Vec3::new(10.0, 0.0, 10.0));
        assert!(danger > 0.0);
    }

    #[test]
    fn test_cover_protection() {
        let cover = CoverPoint::new(
            Vec3::new(5.0, 0.0, 5.0),
            Vec3::new(1.0, 0.0, 0.0), // Faces east.
            CoverHeight::Full,
        );

        // Threat from the west (behind cover) should be protected.
        assert!(cover.protects_from(Vec3::new(0.0, 0.0, 5.0)));
        // Threat from the east (same side as normal) should NOT be protected.
        assert!(!cover.protects_from(Vec3::new(10.0, 0.0, 5.0)));
    }

    #[test]
    fn test_line_of_fire() {
        let mut sa = SpatialAwareness::new(Vec3::ZERO, 32, 1.0);
        sa.set_self_position(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        sa.set_self_entity(0);

        // Add an ally directly between us and the target.
        sa.add_ally(AllyInfo {
            entity: 2,
            position: Vec3::new(5.0, 0.0, 0.0),
            radius: 0.5,
        });

        let result = sa.check_line_of_fire(Vec3::new(10.0, 0.0, 0.0));
        assert!(!result.clear);
        assert_eq!(result.endangered_allies.len(), 1);
    }

    #[test]
    fn test_path_safety() {
        let mut sa = SpatialAwareness::new(Vec3::ZERO, 32, 1.0);
        sa.add_danger_source(DangerSource {
            entity: 1,
            position: Vec3::new(10.0, 0.0, 5.0),
            danger_radius: 5.0,
            danger_strength: 2.0,
            velocity: Vec3::ZERO,
            threat_level: 1.0,
            last_known_time: 0.0,
            ..Default::default()
        });
        sa.update(0.3);

        // Path that goes through the danger zone.
        let dangerous_path = vec![
            Vec3::new(5.0, 0.0, 5.0),
            Vec3::new(10.0, 0.0, 5.0),
            Vec3::new(15.0, 0.0, 5.0),
        ];
        let safety = sa.evaluate_path_safety(&dangerous_path);
        assert!(safety.max_danger > 0.0);
        assert!(safety.overall_safety < 1.0);

        // Safe path far from danger.
        let safe_path = vec![
            Vec3::new(5.0, 0.0, 25.0),
            Vec3::new(10.0, 0.0, 25.0),
        ];
        let safe_result = sa.evaluate_path_safety(&safe_path);
        assert!(safe_result.overall_safety > safety.overall_safety);
    }

    #[test]
    fn test_grid_gradient() {
        let mut grid = SpatialGrid::new(16, 16, 1.0, Vec3::ZERO);
        // Create a gradient increasing toward +x.
        for x in 0..16 {
            for z in 0..16 {
                grid.set(x, z, x as f32);
            }
        }
        let grad = grid.gradient_at(Vec3::new(8.0, 0.0, 8.0));
        assert!(grad.x > 0.0); // Gradient should point in +x direction.
    }

    #[test]
    fn test_bilinear_interpolation() {
        let mut grid = SpatialGrid::new(4, 4, 1.0, Vec3::ZERO);
        grid.set(1, 1, 1.0);
        grid.set(2, 1, 1.0);
        grid.set(1, 2, 1.0);
        grid.set(2, 2, 1.0);
        let val = grid.sample_bilinear(Vec3::new(2.0, 0.0, 2.0));
        assert!(val > 0.5);
    }

    #[test]
    fn test_vec3_operations() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 0.0, 1.0);
        let cross = a.cross(b);
        assert!((cross.y - 1.0).abs() < 1e-6);

        let angle = a.angle_between(b);
        assert!((angle - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    }

    #[test]
    fn test_danger_source_staleness() {
        let source = DangerSource {
            last_known_time: 5.0,
            ..Default::default()
        };
        assert!((source.staleness(7.0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_flanking_detection() {
        let mut sa = SpatialAwareness::new(Vec3::ZERO, 64, 1.0);
        sa.set_self_position(Vec3::new(10.0, 0.0, 10.0), Vec3::new(1.0, 0.0, 0.0));

        sa.add_danger_source(DangerSource {
            entity: 1,
            position: Vec3::new(30.0, 0.0, 10.0),
            facing: Vec3::new(-1.0, 0.0, 0.0),
            weapon_range: 20.0,
            weapon_cone: 1.0,
            danger_radius: 10.0,
            danger_strength: 1.0,
            threat_level: 1.0,
            ..Default::default()
        });

        sa.update(0.3);

        let flanks = sa.find_flanking_opportunities();
        // There should be some flanking opportunities.
        assert!(!flanks.is_empty());
        // Best flank should have a significant angle.
        assert!(flanks[0].flank_angle > FLANKING_ANGLE_THRESHOLD);
    }
}
