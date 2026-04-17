//! Terrain water: water bodies (rivers, lakes, ocean), shore detection,
//! underwater rendering trigger, water flow maps, and water level height query.

use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WaterBodyId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaterBodyType { Lake, River, Ocean, Pond, Waterfall, Stream }

impl WaterBodyType {
    pub fn display_name(&self) -> &'static str {
        match self { Self::Lake => "Lake", Self::River => "River", Self::Ocean => "Ocean", Self::Pond => "Pond", Self::Waterfall => "Waterfall", Self::Stream => "Stream" }
    }
    pub fn default_flow_speed(&self) -> f32 {
        match self { Self::Lake | Self::Pond => 0.0, Self::River => 2.0, Self::Ocean => 0.5, Self::Waterfall => 8.0, Self::Stream => 1.0 }
    }
}

#[derive(Debug, Clone)]
pub struct WaterMaterial {
    pub color_shallow: [f32; 4], pub color_deep: [f32; 4], pub opacity: f32,
    pub refraction_strength: f32, pub reflection_strength: f32, pub wave_amplitude: f32,
    pub wave_frequency: f32, pub wave_speed: f32, pub foam_threshold: f32,
    pub foam_color: [f32; 4], pub specular_power: f32, pub fresnel_power: f32,
    pub caustics_enabled: bool, pub caustics_scale: f32, pub scatter_color: [f32; 3],
    pub absorption: f32, pub shore_blend_distance: f32,
}

impl Default for WaterMaterial {
    fn default() -> Self {
        Self {
            color_shallow: [0.1, 0.4, 0.6, 0.7], color_deep: [0.02, 0.1, 0.2, 0.95],
            opacity: 0.8, refraction_strength: 0.1, reflection_strength: 0.5,
            wave_amplitude: 0.3, wave_frequency: 1.5, wave_speed: 1.0,
            foam_threshold: 0.8, foam_color: [0.9, 0.95, 1.0, 0.8], specular_power: 64.0,
            fresnel_power: 5.0, caustics_enabled: true, caustics_scale: 3.0,
            scatter_color: [0.0, 0.3, 0.2], absorption: 0.5, shore_blend_distance: 2.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FlowMap {
    pub resolution: u32,
    pub data: Vec<[f32; 2]>,
    pub world_size: f32,
    pub world_offset: [f32; 2],
}

impl FlowMap {
    pub fn new(resolution: u32, world_size: f32) -> Self {
        let count = (resolution * resolution) as usize;
        Self { resolution, data: vec![[0.0, 0.0]; count], world_size, world_offset: [0.0; 2] }
    }
    pub fn set_flow(&mut self, x: u32, z: u32, flow: [f32; 2]) {
        let idx = (z * self.resolution + x) as usize;
        if idx < self.data.len() { self.data[idx] = flow; }
    }
    pub fn sample(&self, world_x: f32, world_z: f32) -> [f32; 2] {
        let lx = (world_x - self.world_offset[0]) / self.world_size;
        let lz = (world_z - self.world_offset[1]) / self.world_size;
        let fx = (lx * (self.resolution - 1) as f32).clamp(0.0, (self.resolution - 2) as f32);
        let fz = (lz * (self.resolution - 1) as f32).clamp(0.0, (self.resolution - 2) as f32);
        let idx = (fz as u32 * self.resolution + fx as u32) as usize;
        if idx < self.data.len() { self.data[idx] } else { [0.0, 0.0] }
    }
}

#[derive(Debug, Clone)]
pub struct ShoreData {
    pub points: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 2]>,
    pub shore_distance_map: Option<Vec<f32>>,
    pub map_resolution: u32,
}

impl ShoreData {
    pub fn new() -> Self { Self { points: Vec::new(), normals: Vec::new(), shore_distance_map: None, map_resolution: 0 } }
    pub fn add_point(&mut self, pos: [f32; 3], normal: [f32; 2]) {
        self.points.push(pos);
        self.normals.push(normal);
    }
}

impl Default for ShoreData {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone)]
pub struct WaterBody {
    pub id: WaterBodyId,
    pub name: String,
    pub water_type: WaterBodyType,
    pub water_level: f32,
    pub bounds_min: [f32; 2],
    pub bounds_max: [f32; 2],
    pub material: WaterMaterial,
    pub flow_map: Option<FlowMap>,
    pub shore_data: ShoreData,
    pub flow_direction: [f32; 2],
    pub flow_speed: f32,
    pub visible: bool,
    pub enable_physics: bool,
    pub buoyancy_density: f32,
    pub drag_coefficient: f32,
    pub splash_particle_effect: Option<String>,
    pub underwater_fog_color: [f32; 3],
    pub underwater_fog_density: f32,
    pub surface_mesh_resolution: u32,
    pub depth_fade_distance: f32,
}

impl WaterBody {
    pub fn new(id: WaterBodyId, name: impl Into<String>, wt: WaterBodyType, level: f32) -> Self {
        Self {
            id, name: name.into(), water_type: wt, water_level: level,
            bounds_min: [-500.0, -500.0], bounds_max: [500.0, 500.0],
            material: WaterMaterial::default(), flow_map: None,
            shore_data: ShoreData::new(), flow_direction: [1.0, 0.0],
            flow_speed: wt.default_flow_speed(), visible: true,
            enable_physics: true, buoyancy_density: 1000.0, drag_coefficient: 0.5,
            splash_particle_effect: None, underwater_fog_color: [0.0, 0.15, 0.2],
            underwater_fog_density: 0.1, surface_mesh_resolution: 64,
            depth_fade_distance: 10.0,
        }
    }

    pub fn contains_point_2d(&self, x: f32, z: f32) -> bool {
        x >= self.bounds_min[0] && x <= self.bounds_max[0] && z >= self.bounds_min[1] && z <= self.bounds_max[1]
    }

    pub fn is_underwater(&self, x: f32, y: f32, z: f32) -> bool {
        self.contains_point_2d(x, z) && y < self.water_level
    }

    pub fn depth_at(&self, x: f32, y: f32, z: f32) -> f32 {
        if self.contains_point_2d(x, z) { (self.water_level - y).max(0.0) } else { 0.0 }
    }

    pub fn wave_height_at(&self, x: f32, z: f32, time: f32) -> f32 {
        let m = &self.material;
        let phase = (x * m.wave_frequency + z * m.wave_frequency * 0.7 + time * m.wave_speed);
        self.water_level + phase.sin() * m.wave_amplitude
    }

    pub fn flow_at(&self, x: f32, z: f32) -> [f32; 2] {
        if let Some(ref fm) = self.flow_map { fm.sample(x, z) }
        else { [self.flow_direction[0] * self.flow_speed, self.flow_direction[1] * self.flow_speed] }
    }
}

#[derive(Debug, Clone)]
pub enum WaterEvent {
    BodyCreated(WaterBodyId), BodyRemoved(WaterBodyId),
    WaterLevelChanged(WaterBodyId, f32), EntityEnteredWater(WaterBodyId, u64),
    EntityExitedWater(WaterBodyId, u64), SplashTriggered(WaterBodyId, [f32; 3]),
}

pub struct TerrainWaterSystem {
    pub bodies: HashMap<WaterBodyId, WaterBody>,
    pub events: Vec<WaterEvent>,
    pub next_id: u64,
    pub global_time: f32,
    pub underwater_entity: Option<u64>,
    pub underwater_body: Option<WaterBodyId>,
    pub enable_reflections: bool,
    pub enable_refractions: bool,
    pub enable_caustics: bool,
    pub reflection_resolution: u32,
    pub max_wave_distance: f32,
}

impl TerrainWaterSystem {
    pub fn new() -> Self {
        Self {
            bodies: HashMap::new(), events: Vec::new(), next_id: 1,
            global_time: 0.0, underwater_entity: None, underwater_body: None,
            enable_reflections: true, enable_refractions: true, enable_caustics: true,
            reflection_resolution: 512, max_wave_distance: 500.0,
        }
    }

    pub fn create_body(&mut self, name: impl Into<String>, wt: WaterBodyType, level: f32) -> WaterBodyId {
        let id = WaterBodyId(self.next_id); self.next_id += 1;
        self.bodies.insert(id, WaterBody::new(id, name, wt, level));
        self.events.push(WaterEvent::BodyCreated(id));
        id
    }

    pub fn remove_body(&mut self, id: WaterBodyId) -> bool {
        if self.bodies.remove(&id).is_some() { self.events.push(WaterEvent::BodyRemoved(id)); true } else { false }
    }

    pub fn query_water_level(&self, x: f32, z: f32) -> Option<f32> {
        for body in self.bodies.values() {
            if body.visible && body.contains_point_2d(x, z) { return Some(body.water_level); }
        }
        None
    }

    pub fn query_wave_height(&self, x: f32, z: f32) -> Option<f32> {
        for body in self.bodies.values() {
            if body.visible && body.contains_point_2d(x, z) {
                return Some(body.wave_height_at(x, z, self.global_time));
            }
        }
        None
    }

    pub fn is_underwater(&self, x: f32, y: f32, z: f32) -> Option<WaterBodyId> {
        for (&id, body) in &self.bodies {
            if body.visible && body.is_underwater(x, y, z) { return Some(id); }
        }
        None
    }

    pub fn update(&mut self, dt: f32) { self.global_time += dt; }
    pub fn body_count(&self) -> usize { self.bodies.len() }
    pub fn drain_events(&mut self) -> Vec<WaterEvent> { std::mem::take(&mut self.events) }
}

impl Default for TerrainWaterSystem {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn water_body_basic() {
        let mut sys = TerrainWaterSystem::new();
        let id = sys.create_body("Lake", WaterBodyType::Lake, 10.0);
        assert_eq!(sys.body_count(), 1);
        let level = sys.query_water_level(0.0, 0.0);
        assert_eq!(level, Some(10.0));
    }
    #[test]
    fn underwater_check() {
        let mut sys = TerrainWaterSystem::new();
        sys.create_body("Ocean", WaterBodyType::Ocean, 0.0);
        assert!(sys.is_underwater(0.0, -5.0, 0.0).is_some());
        assert!(sys.is_underwater(0.0, 5.0, 0.0).is_none());
    }
}
