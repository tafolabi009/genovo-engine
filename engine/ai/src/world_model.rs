// engine/ai/src/world_model.rs
//
// AI world representation for planning and decision-making.
// Provides simplified world state, spatial occupancy grid, threat map,
// resource locations, ally positions, objective importance, and
// environmental conditions.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityClass { Player, Ally, Enemy, Neutral, Resource, Objective, Hazard, Cover }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreatLevel { None, Low, Medium, High, Critical }

#[derive(Debug, Clone)]
pub struct WorldEntity {
    pub id: EntityId,
    pub class: EntityClass,
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub health_fraction: f32,
    pub threat_level: ThreatLevel,
    pub last_seen_time: f64,
    pub visible: bool,
    pub alive: bool,
    pub faction_id: u32,
}

impl WorldEntity {
    pub fn new(id: EntityId, class: EntityClass, position: [f32; 3]) -> Self {
        Self { id, class, position, velocity: [0.0; 3], health_fraction: 1.0, threat_level: ThreatLevel::None, last_seen_time: 0.0, visible: false, alive: true, faction_id: 0 }
    }
    pub fn distance_to(&self, other: [f32; 3]) -> f32 {
        let dx = self.position[0] - other[0]; let dy = self.position[1] - other[1]; let dz = self.position[2] - other[2];
        (dx*dx + dy*dy + dz*dz).sqrt()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellOccupancy { Empty, Occupied, Blocked, Hazard, Cover, Unknown }

#[derive(Debug)]
pub struct OccupancyGrid {
    pub cells: Vec<CellOccupancy>,
    pub width: u32,
    pub height: u32,
    pub cell_size: f32,
    pub origin: [f32; 2],
}

impl OccupancyGrid {
    pub fn new(width: u32, height: u32, cell_size: f32) -> Self {
        Self { cells: vec![CellOccupancy::Unknown; (width * height) as usize], width, height, cell_size, origin: [0.0; 2] }
    }
    pub fn world_to_cell(&self, x: f32, z: f32) -> Option<(u32, u32)> {
        let cx = ((x - self.origin[0]) / self.cell_size) as i32;
        let cy = ((z - self.origin[1]) / self.cell_size) as i32;
        if cx >= 0 && cx < self.width as i32 && cy >= 0 && cy < self.height as i32 { Some((cx as u32, cy as u32)) } else { None }
    }
    pub fn get(&self, x: u32, y: u32) -> CellOccupancy {
        if x < self.width && y < self.height { self.cells[(y * self.width + x) as usize] } else { CellOccupancy::Blocked }
    }
    pub fn set(&mut self, x: u32, y: u32, v: CellOccupancy) {
        if x < self.width && y < self.height { self.cells[(y * self.width + x) as usize] = v; }
    }
    pub fn is_walkable(&self, x: u32, y: u32) -> bool { matches!(self.get(x, y), CellOccupancy::Empty | CellOccupancy::Cover) }
    pub fn clear(&mut self) { self.cells.fill(CellOccupancy::Unknown); }
}

#[derive(Debug)]
pub struct ThreatMap {
    pub values: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub cell_size: f32,
    pub decay_rate: f32,
}

impl ThreatMap {
    pub fn new(width: u32, height: u32, cell_size: f32) -> Self {
        Self { values: vec![0.0; (width * height) as usize], width, height, cell_size, decay_rate: 0.1 }
    }
    pub fn add_threat(&mut self, cx: u32, cy: u32, amount: f32, radius: u32) {
        let r = radius as i32;
        for dy in -r..=r { for dx in -r..=r {
            let nx = cx as i32 + dx; let ny = cy as i32 + dy;
            if nx >= 0 && nx < self.width as i32 && ny >= 0 && ny < self.height as i32 {
                let dist = ((dx*dx + dy*dy) as f32).sqrt();
                let falloff = (1.0 - dist / radius as f32).max(0.0);
                self.values[(ny as u32 * self.width + nx as u32) as usize] += amount * falloff;
            }
        }}
    }
    pub fn get(&self, x: u32, y: u32) -> f32 {
        if x < self.width && y < self.height { self.values[(y * self.width + x) as usize] } else { 0.0 }
    }
    pub fn decay(&mut self, dt: f32) {
        let factor = 1.0 - self.decay_rate * dt;
        for v in &mut self.values { *v *= factor; }
    }
    pub fn clear(&mut self) { self.values.fill(0.0); }
    pub fn max_threat(&self) -> f32 { self.values.iter().cloned().fold(0.0f32, f32::max) }
}

#[derive(Debug, Clone)]
pub struct ResourceLocation {
    pub position: [f32; 3],
    pub resource_type: String,
    pub amount: f32,
    pub claimed: bool,
    pub priority: f32,
}

#[derive(Debug, Clone)]
pub struct ObjectiveInfo {
    pub position: [f32; 3],
    pub objective_type: String,
    pub importance: f32,
    pub completed: bool,
    pub assigned_to: Option<EntityId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeatherCondition { Clear, Rain, Snow, Fog, Storm }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeOfDayPeriod { Dawn, Day, Dusk, Night }

#[derive(Debug, Clone)]
pub struct EnvironmentState {
    pub weather: WeatherCondition,
    pub time_of_day: TimeOfDayPeriod,
    pub visibility_range: f32,
    pub ambient_noise: f32,
    pub wind_direction: [f32; 3],
    pub wind_speed: f32,
    pub temperature: f32,
}

impl Default for EnvironmentState {
    fn default() -> Self {
        Self { weather: WeatherCondition::Clear, time_of_day: TimeOfDayPeriod::Day, visibility_range: 100.0, ambient_noise: 0.3, wind_direction: [1.0, 0.0, 0.0], wind_speed: 2.0, temperature: 20.0 }
    }
}

#[derive(Debug)]
pub struct AIWorldModel {
    pub entities: HashMap<EntityId, WorldEntity>,
    pub occupancy: OccupancyGrid,
    pub threat_map: ThreatMap,
    pub resources: Vec<ResourceLocation>,
    pub objectives: Vec<ObjectiveInfo>,
    pub environment: EnvironmentState,
    pub simulation_time: f64,
}

impl AIWorldModel {
    pub fn new(grid_size: u32, cell_size: f32) -> Self {
        Self {
            entities: HashMap::new(),
            occupancy: OccupancyGrid::new(grid_size, grid_size, cell_size),
            threat_map: ThreatMap::new(grid_size, grid_size, cell_size),
            resources: Vec::new(), objectives: Vec::new(),
            environment: EnvironmentState::default(), simulation_time: 0.0,
        }
    }
    pub fn update_entity(&mut self, entity: WorldEntity) { self.entities.insert(entity.id, entity); }
    pub fn remove_entity(&mut self, id: EntityId) { self.entities.remove(&id); }
    pub fn enemies_near(&self, pos: [f32; 3], radius: f32) -> Vec<&WorldEntity> {
        self.entities.values().filter(|e| e.class == EntityClass::Enemy && e.alive && e.distance_to(pos) <= radius).collect()
    }
    pub fn allies_near(&self, pos: [f32; 3], radius: f32) -> Vec<&WorldEntity> {
        self.entities.values().filter(|e| e.class == EntityClass::Ally && e.alive && e.distance_to(pos) <= radius).collect()
    }
    pub fn nearest_resource(&self, pos: [f32; 3], res_type: &str) -> Option<&ResourceLocation> {
        self.resources.iter().filter(|r| !r.claimed && r.resource_type == res_type)
            .min_by(|a, b| {
                let da = dist3(a.position, pos); let db = dist3(b.position, pos);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
    }
    pub fn update(&mut self, dt: f32) {
        self.simulation_time += dt as f64;
        self.threat_map.decay(dt);
    }
    pub fn entity_count(&self) -> usize { self.entities.len() }
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0]-b[0]; let dy = a[1]-b[1]; let dz = a[2]-b[2];
    (dx*dx+dy*dy+dz*dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_occupancy_grid() {
        let mut grid = OccupancyGrid::new(10, 10, 1.0);
        grid.set(5, 5, CellOccupancy::Blocked);
        assert!(!grid.is_walkable(5, 5));
        assert!(grid.is_walkable(0, 0) || grid.get(0, 0) == CellOccupancy::Unknown);
    }
    #[test]
    fn test_threat_map() {
        let mut map = ThreatMap::new(10, 10, 1.0);
        map.add_threat(5, 5, 10.0, 3);
        assert!(map.get(5, 5) > 0.0);
        assert!(map.max_threat() > 0.0);
    }
    #[test]
    fn test_world_model() {
        let mut model = AIWorldModel::new(20, 1.0);
        model.update_entity(WorldEntity::new(EntityId(1), EntityClass::Enemy, [5.0, 0.0, 5.0]));
        assert_eq!(model.entity_count(), 1);
        assert_eq!(model.enemies_near([5.0, 0.0, 5.0], 10.0).len(), 1);
    }
}
