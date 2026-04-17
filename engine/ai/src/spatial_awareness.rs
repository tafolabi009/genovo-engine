// engine/ai/src/spatial_awareness.rs
//
// AI spatial awareness system for the Genovo AI module.
//
// Provides visibility maps, danger maps, path safety evaluation, predictive
// position estimation, exposure analysis, and defensive position scoring.

use std::collections::HashMap;

pub type EntityId = u64;

pub const GRID_DEFAULT_SIZE: u32 = 64;
pub const GRID_DEFAULT_CELL_SIZE: f32 = 2.0;
pub const MAX_DANGER_SOURCES: usize = 128;
pub const VISIBILITY_UPDATE_INTERVAL: f32 = 0.2;
pub const PREDICTION_HORIZON: f32 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32, pub y: f32, pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn add(self, o: Self) -> Self { Self::new(self.x + o.x, self.y + o.y, self.z + o.z) }
    pub fn sub(self, o: Self) -> Self { Self::new(self.x - o.x, self.y - o.y, self.z - o.z) }
    pub fn scale(self, s: f32) -> Self { Self::new(self.x * s, self.y * s, self.z * s) }
    pub fn dot(self, o: Self) -> f32 { self.x * o.x + self.y * o.y + self.z * o.z }
    pub fn length(self) -> f32 { self.dot(self).sqrt() }
    pub fn length_sq(self) -> f32 { self.dot(self) }
    pub fn normalize(self) -> Self { let l = self.length(); if l > 1e-7 { self.scale(1.0 / l) } else { Self::ZERO } }
    pub fn distance(self, o: Self) -> f32 { self.sub(o).length() }
    pub fn distance_sq(self, o: Self) -> f32 { self.sub(o).length_sq() }
    pub fn lerp(self, o: Self, t: f32) -> Self { self.scale(1.0 - t).add(o.scale(t)) }
}

#[derive(Debug, Clone, Copy)]
pub struct GridCoord { pub x: i32, pub z: i32 }

impl GridCoord {
    pub fn new(x: i32, z: i32) -> Self { Self { x, z } }
    pub fn distance_sq(&self, other: &Self) -> i32 { let dx = self.x - other.x; let dz = self.z - other.z; dx * dx + dz * dz }
}

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
        Self { width, height, cell_size, origin, cells: vec![0.0; (width * height) as usize] }
    }
    pub fn world_to_grid(&self, pos: Vec3) -> Option<GridCoord> {
        let lx = ((pos.x - self.origin.x) / self.cell_size) as i32;
        let lz = ((pos.z - self.origin.z) / self.cell_size) as i32;
        if lx >= 0 && lz >= 0 && (lx as u32) < self.width && (lz as u32) < self.height { Some(GridCoord::new(lx, lz)) } else { None }
    }
    pub fn grid_to_world(&self, coord: GridCoord) -> Vec3 {
        Vec3::new(self.origin.x + coord.x as f32 * self.cell_size + self.cell_size * 0.5, self.origin.y, self.origin.z + coord.z as f32 * self.cell_size + self.cell_size * 0.5)
    }
    pub fn get(&self, x: u32, z: u32) -> f32 { if x < self.width && z < self.height { self.cells[(z * self.width + x) as usize] } else { 0.0 } }
    pub fn set(&mut self, x: u32, z: u32, val: f32) { if x < self.width && z < self.height { self.cells[(z * self.width + x) as usize] = val; } }
    pub fn get_at(&self, pos: Vec3) -> f32 { self.world_to_grid(pos).map(|c| self.get(c.x as u32, c.z as u32)).unwrap_or(0.0) }
    pub fn set_at(&mut self, pos: Vec3, val: f32) { if let Some(c) = self.world_to_grid(pos) { self.set(c.x as u32, c.z as u32, val); } }
    pub fn clear(&mut self) { for c in &mut self.cells { *c = 0.0; } }
    pub fn add_influence(&mut self, pos: Vec3, radius: f32, strength: f32) {
        let r_cells = (radius / self.cell_size).ceil() as i32;
        if let Some(center) = self.world_to_grid(pos) {
            for dz in -r_cells..=r_cells {
                for dx in -r_cells..=r_cells {
                    let gx = center.x + dx; let gz = center.z + dz;
                    if gx >= 0 && gz >= 0 && (gx as u32) < self.width && (gz as u32) < self.height {
                        let world = self.grid_to_world(GridCoord::new(gx, gz));
                        let dist = pos.distance(world);
                        if dist <= radius {
                            let falloff = 1.0 - dist / radius;
                            let idx = (gz as u32 * self.width + gx as u32) as usize;
                            self.cells[idx] += strength * falloff * falloff;
                        }
                    }
                }
            }
        }
    }
    pub fn max_value(&self) -> f32 { self.cells.iter().cloned().fold(f32::MIN, f32::max) }
}

#[derive(Debug, Clone)]
pub struct DangerSource {
    pub entity: EntityId,
    pub position: Vec3,
    pub danger_radius: f32,
    pub danger_strength: f32,
    pub velocity: Vec3,
    pub threat_level: f32,
    pub last_known_time: f32,
}

impl DangerSource {
    pub fn predicted_position(&self, dt: f32) -> Vec3 { self.position.add(self.velocity.scale(dt)) }
}

#[derive(Debug, Clone, Copy)]
pub struct ExposureResult {
    pub exposure_score: f32,
    pub visible_threats: u32,
    pub nearest_cover_distance: f32,
    pub nearest_cover_direction: Vec3,
    pub exposed_angle_ratio: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct DefensiveScore {
    pub total_score: f32,
    pub cover_quality: f32,
    pub sight_lines: f32,
    pub escape_routes: f32,
    pub elevation_advantage: f32,
    pub flank_protection: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct PathSafety {
    pub overall_safety: f32,
    pub max_danger: f32,
    pub avg_danger: f32,
    pub exposed_segments: u32,
    pub total_segments: u32,
    pub safest_alternative: Option<Vec3>,
}

pub struct SpatialAwareness {
    danger_map: SpatialGrid,
    visibility_map: SpatialGrid,
    cover_map: SpatialGrid,
    danger_sources: Vec<DangerSource>,
    cover_points: Vec<Vec3>,
    self_position: Vec3,
    self_forward: Vec3,
    grid_origin: Vec3,
    grid_size: u32,
    cell_size: f32,
    time: f32,
    update_timer: f32,
}

impl SpatialAwareness {
    pub fn new(origin: Vec3, grid_size: u32, cell_size: f32) -> Self {
        Self {
            danger_map: SpatialGrid::new(grid_size, grid_size, cell_size, origin),
            visibility_map: SpatialGrid::new(grid_size, grid_size, cell_size, origin),
            cover_map: SpatialGrid::new(grid_size, grid_size, cell_size, origin),
            danger_sources: Vec::new(), cover_points: Vec::new(),
            self_position: Vec3::ZERO, self_forward: Vec3::new(0.0, 0.0, 1.0),
            grid_origin: origin, grid_size, cell_size, time: 0.0, update_timer: 0.0,
        }
    }

    pub fn set_self_position(&mut self, pos: Vec3, forward: Vec3) { self.self_position = pos; self.self_forward = forward.normalize(); }
    pub fn add_danger_source(&mut self, source: DangerSource) { if self.danger_sources.len() < MAX_DANGER_SOURCES { self.danger_sources.push(source); } }
    pub fn clear_danger_sources(&mut self) { self.danger_sources.clear(); }
    pub fn add_cover_point(&mut self, pos: Vec3) { self.cover_points.push(pos); }
    pub fn clear_cover_points(&mut self) { self.cover_points.clear(); }

    pub fn update(&mut self, dt: f32) {
        self.time += dt;
        self.update_timer += dt;
        if self.update_timer >= VISIBILITY_UPDATE_INTERVAL {
            self.update_timer = 0.0;
            self.rebuild_danger_map();
            self.rebuild_cover_map();
        }
    }

    fn rebuild_danger_map(&mut self) {
        self.danger_map.clear();
        for source in &self.danger_sources {
            let predicted = source.predicted_position(0.5);
            self.danger_map.add_influence(predicted, source.danger_radius, source.danger_strength * source.threat_level);
        }
    }

    fn rebuild_cover_map(&mut self) {
        self.cover_map.clear();
        for &cp in &self.cover_points {
            self.cover_map.add_influence(cp, self.cell_size * 2.0, 1.0);
        }
    }

    pub fn get_danger_at(&self, pos: Vec3) -> f32 { self.danger_map.get_at(pos) }

    pub fn evaluate_path_safety(&self, path: &[Vec3]) -> PathSafety {
        if path.is_empty() { return PathSafety { overall_safety: 1.0, max_danger: 0.0, avg_danger: 0.0, exposed_segments: 0, total_segments: 0, safest_alternative: None }; }
        let mut total_danger = 0.0f32;
        let mut max_danger = 0.0f32;
        let mut exposed = 0u32;
        for p in path {
            let d = self.danger_map.get_at(*p);
            total_danger += d; max_danger = max_danger.max(d);
            if d > 0.5 { exposed += 1; }
        }
        let avg = total_danger / path.len() as f32;
        PathSafety { overall_safety: (1.0 - avg).max(0.0), max_danger, avg_danger: avg, exposed_segments: exposed, total_segments: path.len() as u32, safest_alternative: None }
    }

    pub fn evaluate_exposure(&self, pos: Vec3) -> ExposureResult {
        let danger = self.danger_map.get_at(pos);
        let visible = self.danger_sources.iter().filter(|s| s.position.distance(pos) < s.danger_radius * 1.5).count() as u32;
        let mut nearest_dist = f32::MAX;
        let mut nearest_dir = Vec3::ZERO;
        for &cp in &self.cover_points {
            let d = pos.distance(cp);
            if d < nearest_dist { nearest_dist = d; nearest_dir = cp.sub(pos).normalize(); }
        }
        ExposureResult { exposure_score: danger, visible_threats: visible, nearest_cover_distance: nearest_dist, nearest_cover_direction: nearest_dir, exposed_angle_ratio: danger.min(1.0) }
    }

    pub fn score_defensive_position(&self, pos: Vec3) -> DefensiveScore {
        let danger = self.danger_map.get_at(pos);
        let cover = self.cover_map.get_at(pos);
        let elevation = pos.y;
        let nearest_threat_dist = self.danger_sources.iter().map(|s| s.position.distance(pos)).fold(f32::MAX, f32::min);
        let cover_quality = cover.min(1.0);
        let elevation_adv = (elevation * 0.1).min(1.0).max(0.0);
        let escape = self.cover_points.iter().filter(|cp| cp.distance(pos) < 10.0).count() as f32 / 5.0;
        let flank = if nearest_threat_dist > 5.0 { 0.7 } else { 0.3 };
        let total = cover_quality * 0.3 + (1.0 - danger.min(1.0)) * 0.25 + elevation_adv * 0.15 + escape.min(1.0) * 0.15 + flank * 0.15;
        DefensiveScore { total_score: total, cover_quality, sight_lines: 1.0 - danger.min(1.0), escape_routes: escape.min(1.0), elevation_advantage: elevation_adv, flank_protection: flank }
    }

    pub fn predict_position(current: Vec3, velocity: Vec3, acceleration: Vec3, time: f32) -> Vec3 {
        current.add(velocity.scale(time)).add(acceleration.scale(0.5 * time * time))
    }

    pub fn find_safest_position(&self, candidates: &[Vec3]) -> Option<(Vec3, f32)> {
        candidates.iter().map(|&pos| {
            let score = self.score_defensive_position(pos);
            (pos, score.total_score)
        }).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    pub fn danger_map(&self) -> &SpatialGrid { &self.danger_map }
    pub fn visibility_map(&self) -> &SpatialGrid { &self.visibility_map }
    pub fn cover_map(&self) -> &SpatialGrid { &self.cover_map }
}

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
    }

    #[test]
    fn test_prediction() {
        let pos = SpatialAwareness::predict_position(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO, 2.0);
        assert!((pos.x - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_spatial_awareness() {
        let mut sa = SpatialAwareness::new(Vec3::ZERO, 32, 1.0);
        sa.add_danger_source(DangerSource { entity: 1, position: Vec3::new(10.0, 0.0, 10.0), danger_radius: 5.0, danger_strength: 1.0, velocity: Vec3::ZERO, threat_level: 1.0, last_known_time: 0.0 });
        sa.update(0.3);
        let danger = sa.get_danger_at(Vec3::new(10.0, 0.0, 10.0));
        assert!(danger > 0.0);
    }
}
