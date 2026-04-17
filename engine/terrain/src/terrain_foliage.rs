//! Terrain foliage: per-cell foliage data, wind animation, LOD transitions,
//! instanced rendering data, and seasonal changes.

use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FoliageTypeId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FoliageCellId(pub u32, pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Season { Spring, Summer, Autumn, Winter }

impl Season {
    pub fn color_multiplier(&self) -> [f32; 3] {
        match self {
            Self::Spring => [0.7, 1.0, 0.5], Self::Summer => [0.5, 0.9, 0.3],
            Self::Autumn => [1.0, 0.6, 0.2], Self::Winter => [0.6, 0.6, 0.5],
        }
    }
    pub fn density_multiplier(&self) -> f32 {
        match self { Self::Spring => 0.8, Self::Summer => 1.0, Self::Autumn => 0.7, Self::Winter => 0.3 }
    }
    pub fn next(&self) -> Self {
        match self { Self::Spring => Self::Summer, Self::Summer => Self::Autumn, Self::Autumn => Self::Winter, Self::Winter => Self::Spring }
    }
}

#[derive(Debug, Clone)]
pub struct WindParams {
    pub direction: [f32; 3],
    pub speed: f32,
    pub gustiness: f32,
    pub gust_frequency: f32,
    pub gust_speed: f32,
    pub turbulence: f32,
    pub trunk_flexibility: f32,
    pub branch_flexibility: f32,
    pub leaf_flutter: f32,
    pub time: f32,
}

impl Default for WindParams {
    fn default() -> Self {
        Self {
            direction: [1.0, 0.0, 0.3], speed: 2.0, gustiness: 0.5,
            gust_frequency: 0.2, gust_speed: 5.0, turbulence: 0.3,
            trunk_flexibility: 0.1, branch_flexibility: 0.3, leaf_flutter: 1.0, time: 0.0,
        }
    }
}

impl WindParams {
    pub fn update(&mut self, dt: f32) { self.time += dt; }
    pub fn wind_at(&self, pos: [f32; 3]) -> [f32; 3] {
        let gust = (self.time * self.gust_frequency + pos[0] * 0.01 + pos[2] * 0.01).sin() * self.gustiness;
        let strength = self.speed + gust * self.gust_speed;
        let len = (self.direction[0]*self.direction[0] + self.direction[1]*self.direction[1] + self.direction[2]*self.direction[2]).sqrt().max(0.001);
        [self.direction[0]/len * strength, self.direction[1]/len * strength, self.direction[2]/len * strength]
    }
}

#[derive(Debug, Clone)]
pub struct FoliageTypeDef {
    pub id: FoliageTypeId,
    pub name: String,
    pub mesh_path: String,
    pub material_path: String,
    pub density: f32,
    pub min_scale: f32,
    pub max_scale: f32,
    pub min_slope: f32,
    pub max_slope: f32,
    pub min_height: f32,
    pub max_height: f32,
    pub random_rotation: bool,
    pub align_to_normal: bool,
    pub cast_shadows: bool,
    pub lod_distances: Vec<f32>,
    pub wind_response: f32,
    pub seasonal_tint: bool,
    pub collision_enabled: bool,
    pub cull_distance: f32,
    pub fade_distance: f32,
    pub placement_jitter: f32,
}

impl FoliageTypeDef {
    pub fn new(id: FoliageTypeId, name: impl Into<String>, mesh: impl Into<String>) -> Self {
        Self {
            id, name: name.into(), mesh_path: mesh.into(), material_path: String::new(),
            density: 1.0, min_scale: 0.8, max_scale: 1.2, min_slope: 0.0, max_slope: 45.0,
            min_height: -1000.0, max_height: 1000.0, random_rotation: true,
            align_to_normal: true, cast_shadows: true,
            lod_distances: vec![50.0, 100.0, 200.0], wind_response: 1.0,
            seasonal_tint: true, collision_enabled: false, cull_distance: 300.0,
            fade_distance: 280.0, placement_jitter: 0.3,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FoliageInstance {
    pub position: [f32; 3],
    pub rotation: f32,
    pub scale: f32,
    pub normal: [f32; 3],
    pub color_variation: f32,
    pub type_id: FoliageTypeId,
    pub lod_level: u8,
    pub visible: bool,
}

impl FoliageInstance {
    pub fn new(pos: [f32; 3], type_id: FoliageTypeId) -> Self {
        Self {
            position: pos, rotation: 0.0, scale: 1.0, normal: [0.0, 1.0, 0.0],
            color_variation: 0.0, type_id, lod_level: 0, visible: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FoliageCell {
    pub id: FoliageCellId,
    pub instances: Vec<FoliageInstance>,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub loaded: bool,
    pub visible: bool,
    pub instance_buffer_dirty: bool,
}

impl FoliageCell {
    pub fn new(id: FoliageCellId) -> Self {
        Self {
            id, instances: Vec::new(), bounds_min: [f32::MAX; 3], bounds_max: [f32::MIN; 3],
            loaded: false, visible: true, instance_buffer_dirty: true,
        }
    }

    pub fn add_instance(&mut self, inst: FoliageInstance) {
        for i in 0..3 {
            self.bounds_min[i] = self.bounds_min[i].min(inst.position[i]);
            self.bounds_max[i] = self.bounds_max[i].max(inst.position[i]);
        }
        self.instances.push(inst);
        self.instance_buffer_dirty = true;
    }

    pub fn remove_in_radius(&mut self, center: [f32; 2], radius: f32) -> usize {
        let r2 = radius * radius;
        let before = self.instances.len();
        self.instances.retain(|i| {
            let dx = i.position[0] - center[0];
            let dz = i.position[2] - center[1];
            dx * dx + dz * dz > r2
        });
        let removed = before - self.instances.len();
        if removed > 0 { self.instance_buffer_dirty = true; }
        removed
    }

    pub fn instance_count(&self) -> usize { self.instances.len() }

    pub fn update_lods(&mut self, camera_pos: [f32; 3], lod_distances: &[f32]) {
        for inst in &mut self.instances {
            let dx = inst.position[0] - camera_pos[0];
            let dy = inst.position[1] - camera_pos[1];
            let dz = inst.position[2] - camera_pos[2];
            let dist = (dx*dx + dy*dy + dz*dz).sqrt();
            let mut lod = 0u8;
            for (i, &d) in lod_distances.iter().enumerate() {
                if dist > d { lod = (i + 1) as u8; }
            }
            inst.lod_level = lod;
        }
    }
}

#[derive(Debug, Clone)]
pub enum FoliageEvent {
    CellLoaded(FoliageCellId), CellUnloaded(FoliageCellId),
    InstancesAdded(FoliageCellId, usize), InstancesRemoved(FoliageCellId, usize),
    SeasonChanged(Season), WindChanged,
}

pub struct TerrainFoliageSystem {
    pub types: HashMap<FoliageTypeId, FoliageTypeDef>,
    pub cells: HashMap<FoliageCellId, FoliageCell>,
    pub wind: WindParams,
    pub season: Season,
    pub season_transition: f32,
    pub events: Vec<FoliageEvent>,
    pub cell_size: f32,
    pub camera_position: [f32; 3],
    pub render_distance: f32,
    pub total_instances: u64,
    pub next_type_id: u32,
    pub density_multiplier: f32,
    pub global_wind_enabled: bool,
}

impl TerrainFoliageSystem {
    pub fn new(cell_size: f32) -> Self {
        Self {
            types: HashMap::new(), cells: HashMap::new(), wind: WindParams::default(),
            season: Season::Summer, season_transition: 0.0, events: Vec::new(),
            cell_size, camera_position: [0.0; 3], render_distance: 300.0,
            total_instances: 0, next_type_id: 1, density_multiplier: 1.0,
            global_wind_enabled: true,
        }
    }

    pub fn register_type(&mut self, name: impl Into<String>, mesh: impl Into<String>) -> FoliageTypeId {
        let id = FoliageTypeId(self.next_type_id);
        self.next_type_id += 1;
        self.types.insert(id, FoliageTypeDef::new(id, name, mesh));
        id
    }

    pub fn place_instance(&mut self, cell_x: u32, cell_z: u32, inst: FoliageInstance) {
        let id = FoliageCellId(cell_x, cell_z);
        let cell = self.cells.entry(id).or_insert_with(|| FoliageCell::new(id));
        cell.add_instance(inst);
        self.total_instances += 1;
    }

    pub fn remove_in_radius(&mut self, center: [f32; 2], radius: f32) -> usize {
        let mut total = 0;
        for cell in self.cells.values_mut() {
            let removed = cell.remove_in_radius(center, radius);
            total += removed;
        }
        self.total_instances -= total as u64;
        total
    }

    pub fn set_season(&mut self, season: Season) {
        self.season = season;
        self.events.push(FoliageEvent::SeasonChanged(season));
    }

    pub fn update(&mut self, dt: f32) {
        if self.global_wind_enabled { self.wind.update(dt); }
        for cell in self.cells.values_mut() {
            if cell.loaded && cell.visible {
                cell.update_lods(self.camera_position, &[50.0, 100.0, 200.0]);
            }
        }
    }

    pub fn set_camera(&mut self, pos: [f32; 3]) { self.camera_position = pos; }
    pub fn type_count(&self) -> usize { self.types.len() }
    pub fn cell_count(&self) -> usize { self.cells.len() }
    pub fn drain_events(&mut self) -> Vec<FoliageEvent> { std::mem::take(&mut self.events) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wind_calculation() {
        let w = WindParams::default();
        let v = w.wind_at([0.0, 0.0, 0.0]);
        assert!(v[0] != 0.0 || v[2] != 0.0);
    }
    #[test]
    fn foliage_placement() {
        let mut sys = TerrainFoliageSystem::new(64.0);
        let tid = sys.register_type("Grass", "meshes/grass.obj");
        sys.place_instance(0, 0, FoliageInstance::new([10.0, 0.0, 10.0], tid));
        assert_eq!(sys.total_instances, 1);
        let removed = sys.remove_in_radius([10.0, 10.0], 5.0);
        assert_eq!(removed, 1);
    }
    #[test]
    fn season_cycle() {
        assert_eq!(Season::Summer.next(), Season::Autumn);
        assert!(Season::Winter.density_multiplier() < Season::Summer.density_multiplier());
    }
}
