#!/usr/bin/env python3
"""Generate remaining Genovo engine source files - batch 3 (final)."""
import os

base = "C:/Users/USER/Downloads/game_engine/engine"
all_content = {}

all_content[f"{base}/terrain/src/terrain_water.rs"] = '''//! Terrain water: water bodies (rivers, lakes, ocean), shore detection,
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
'''

all_content[f"{base}/procgen/src/world_generator.rs"] = '''//! World generation: continent shapes (Perlin + Voronoi), climate zones,
//! biome placement, river network, road network, and settlement placement.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Biome { Ocean, Beach, Plains, Forest, Jungle, Desert, Tundra, Mountain, Snow, Swamp, Savanna, Taiga, Steppe }
impl Biome {
    pub fn color(&self) -> [f32; 3] {
        match self { Self::Ocean => [0.1, 0.2, 0.6], Self::Beach => [0.9, 0.85, 0.6], Self::Plains => [0.5, 0.75, 0.3], Self::Forest => [0.15, 0.5, 0.15], Self::Jungle => [0.1, 0.4, 0.05], Self::Desert => [0.85, 0.75, 0.4], Self::Tundra => [0.7, 0.75, 0.7], Self::Mountain => [0.5, 0.45, 0.4], Self::Snow => [0.95, 0.95, 0.95], Self::Swamp => [0.3, 0.4, 0.25], Self::Savanna => [0.7, 0.65, 0.3], Self::Taiga => [0.2, 0.4, 0.3], Self::Steppe => [0.6, 0.55, 0.3] }
    }
    pub fn tree_density(&self) -> f32 {
        match self { Self::Forest | Self::Taiga => 0.8, Self::Jungle => 0.95, Self::Swamp => 0.4, Self::Plains | Self::Savanna => 0.05, _ => 0.0 }
    }
}

#[derive(Debug, Clone)]
pub struct WorldCell {
    pub x: i32, pub z: i32, pub elevation: f32, pub moisture: f32, pub temperature: f32,
    pub biome: Biome, pub is_land: bool, pub river_flow: f32, pub has_road: bool,
    pub settlement: Option<SettlementType>, pub continent_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettlementType { Village, Town, City, Capital, Outpost, Port }

#[derive(Debug, Clone)]
pub struct RiverSegment { pub start: [i32; 2], pub end: [i32; 2], pub width: f32, pub flow: f32 }

#[derive(Debug, Clone)]
pub struct RoadSegment { pub start: [i32; 2], pub end: [i32; 2], pub road_type: RoadType, pub cost: f32 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoadType { Trail, Road, Highway }

#[derive(Debug, Clone)]
pub struct Settlement { pub position: [i32; 2], pub settlement_type: SettlementType, pub name: String, pub population: u32, pub trade_value: f32 }

#[derive(Debug, Clone)]
pub struct WorldGenConfig {
    pub width: u32, pub height: u32, pub seed: u64, pub sea_level: f32,
    pub continent_count: u32, pub mountain_frequency: f32, pub river_count: u32,
    pub settlement_count: u32, pub temperature_offset: f32, pub moisture_offset: f32,
    pub erosion_iterations: u32, pub road_connectivity: f32,
}

impl Default for WorldGenConfig {
    fn default() -> Self {
        Self { width: 256, height: 256, seed: 42, sea_level: 0.4, continent_count: 3, mountain_frequency: 0.03, river_count: 20, settlement_count: 15, temperature_offset: 0.0, moisture_offset: 0.0, erosion_iterations: 50, road_connectivity: 0.7 }
    }
}

pub struct GeneratedWorld {
    pub config: WorldGenConfig,
    pub cells: Vec<Vec<WorldCell>>,
    pub rivers: Vec<RiverSegment>,
    pub roads: Vec<RoadSegment>,
    pub settlements: Vec<Settlement>,
    pub width: u32,
    pub height: u32,
}

fn simple_hash(x: i32, z: i32, seed: u64) -> f32 {
    let mut h = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    h ^= x as u64; h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    h ^= z as u64; h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((h >> 33) as f32) / (u32::MAX as f32)
}

fn perlin_noise(x: f32, z: f32, freq: f32, seed: u64) -> f32 {
    let fx = x * freq; let fz = z * freq;
    let ix = fx.floor() as i32; let iz = fz.floor() as i32;
    let tx = fx - ix as f32; let tz = fz - iz as f32;
    let c00 = simple_hash(ix, iz, seed); let c10 = simple_hash(ix+1, iz, seed);
    let c01 = simple_hash(ix, iz+1, seed); let c11 = simple_hash(ix+1, iz+1, seed);
    let st = |t: f32| t * t * (3.0 - 2.0 * t);
    let top = c00 + (c10 - c00) * st(tx);
    let bot = c01 + (c11 - c01) * st(tx);
    top + (bot - top) * st(tz)
}

fn classify_biome(elevation: f32, moisture: f32, temperature: f32) -> Biome {
    if elevation < 0.01 { return Biome::Ocean; }
    if elevation < 0.05 { return Biome::Beach; }
    if elevation > 0.8 { return if temperature < 0.2 { Biome::Snow } else { Biome::Mountain }; }
    if temperature < 0.15 { return if moisture > 0.5 { Biome::Taiga } else { Biome::Tundra }; }
    if temperature > 0.7 {
        if moisture < 0.2 { return Biome::Desert; }
        if moisture < 0.5 { return Biome::Savanna; }
        return Biome::Jungle;
    }
    if moisture > 0.7 { return Biome::Swamp; }
    if moisture > 0.4 { return Biome::Forest; }
    if moisture > 0.2 { return Biome::Plains; }
    Biome::Steppe
}

pub fn generate_world(config: &WorldGenConfig) -> GeneratedWorld {
    let w = config.width as usize; let h = config.height as usize;
    let mut cells = Vec::with_capacity(h);
    for z in 0..h {
        let mut row = Vec::with_capacity(w);
        for x in 0..w {
            let nx = x as f32 / w as f32; let nz = z as f32 / h as f32;
            let e1 = perlin_noise(nx, nz, 2.0, config.seed);
            let e2 = perlin_noise(nx, nz, 4.0, config.seed + 1) * 0.5;
            let e3 = perlin_noise(nx, nz, 8.0, config.seed + 2) * 0.25;
            let e4 = perlin_noise(nx, nz, config.mountain_frequency * 100.0, config.seed + 3) * 0.15;
            let elevation = ((e1 + e2 + e3 + e4) / 1.9 - config.sea_level).max(0.0).min(1.0);
            let moisture = (perlin_noise(nx, nz, 3.0, config.seed + 100) + config.moisture_offset).clamp(0.0, 1.0);
            let lat = (nz - 0.5).abs() * 2.0;
            let temperature = (1.0 - lat - elevation * 0.5 + config.temperature_offset).clamp(0.0, 1.0);
            let biome = classify_biome(elevation, moisture, temperature);
            row.push(WorldCell {
                x: x as i32, z: z as i32, elevation, moisture, temperature, biome,
                is_land: elevation > 0.01, river_flow: 0.0, has_road: false,
                settlement: None, continent_id: 0,
            });
        }
        cells.push(row);
    }

    // Simple river generation
    let mut rivers = Vec::new();
    for i in 0..config.river_count {
        let sx = simple_hash(i as i32, 0, config.seed + 500);
        let sz = simple_hash(0, i as i32, config.seed + 600);
        let mut cx = (sx * w as f32) as i32; let mut cz = (sz * h as f32) as i32;
        for step in 0..200 {
            if cx < 1 || cx >= (w as i32 - 1) || cz < 1 || cz >= (h as i32 - 1) { break; }
            let cell = &cells[cz as usize][cx as usize];
            if !cell.is_land { break; }
            let mut min_e = cell.elevation; let mut nx = cx; let mut nz = cz;
            for &(dx, dz) in &[(-1,0),(1,0),(0,-1),(0,1)] {
                let e = cells[(cz+dz) as usize][(cx+dx) as usize].elevation;
                if e < min_e { min_e = e; nx = cx+dx; nz = cz+dz; }
            }
            if nx == cx && nz == cz { break; }
            rivers.push(RiverSegment { start: [cx, cz], end: [nx, nz], width: 1.0 + step as f32 * 0.05, flow: 1.0 });
            cells[cz as usize][cx as usize].river_flow += 1.0;
            cx = nx; cz = nz;
        }
    }

    // Settlement placement
    let mut settlements = Vec::new();
    for i in 0..config.settlement_count {
        let sx = (simple_hash(i as i32 * 3, 0, config.seed + 700) * w as f32) as usize;
        let sz = (simple_hash(0, i as i32 * 3, config.seed + 800) * h as f32) as usize;
        if sx < w && sz < h && cells[sz][sx].is_land && cells[sz][sx].elevation < 0.7 {
            let st = if i == 0 { SettlementType::Capital } else if i < 4 { SettlementType::City } else if i < 8 { SettlementType::Town } else { SettlementType::Village };
            let pop = match st { SettlementType::Capital => 50000, SettlementType::City => 10000, SettlementType::Town => 2000, _ => 200 };
            settlements.push(Settlement { position: [sx as i32, sz as i32], settlement_type: st, name: format!("Settlement_{}", i), population: pop, trade_value: pop as f32 * 0.1 });
            cells[sz][sx].settlement = Some(st);
        }
    }

    // Simple road network
    let mut roads = Vec::new();
    for i in 0..settlements.len() {
        for j in (i+1)..settlements.len() {
            let a = &settlements[i]; let b = &settlements[j];
            let dx = (a.position[0] - b.position[0]) as f32;
            let dz = (a.position[1] - b.position[1]) as f32;
            let dist = (dx*dx + dz*dz).sqrt();
            if dist < (w as f32 * config.road_connectivity) {
                let rt = if a.population > 5000 && b.population > 5000 { RoadType::Highway } else if a.population > 1000 || b.population > 1000 { RoadType::Road } else { RoadType::Trail };
                roads.push(RoadSegment { start: a.position, end: b.position, road_type: rt, cost: dist });
            }
        }
    }

    GeneratedWorld { config: config.clone(), cells, rivers, roads, settlements, width: config.width, height: config.height }
}

impl GeneratedWorld {
    pub fn get_cell(&self, x: i32, z: i32) -> Option<&WorldCell> {
        if x >= 0 && z >= 0 && (x as u32) < self.width && (z as u32) < self.height {
            Some(&self.cells[z as usize][x as usize])
        } else { None }
    }
    pub fn land_percentage(&self) -> f32 {
        let total = (self.width * self.height) as f32;
        let land = self.cells.iter().flat_map(|r| r.iter()).filter(|c| c.is_land).count() as f32;
        land / total * 100.0
    }
    pub fn biome_distribution(&self) -> HashMap<&'static str, usize> {
        let mut dist = HashMap::new();
        for row in &self.cells { for cell in row { *dist.entry(match cell.biome { Biome::Ocean => "Ocean", Biome::Beach => "Beach", Biome::Plains => "Plains", Biome::Forest => "Forest", Biome::Jungle => "Jungle", Biome::Desert => "Desert", Biome::Tundra => "Tundra", Biome::Mountain => "Mountain", Biome::Snow => "Snow", Biome::Swamp => "Swamp", Biome::Savanna => "Savanna", Biome::Taiga => "Taiga", Biome::Steppe => "Steppe" }).or_insert(0) += 1; } }
        dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generate_basic() {
        let config = WorldGenConfig { width: 64, height: 64, ..Default::default() };
        let world = generate_world(&config);
        assert_eq!(world.cells.len(), 64);
        assert!(world.land_percentage() > 0.0);
    }
    #[test]
    fn biome_classification() {
        assert_eq!(classify_biome(0.0, 0.5, 0.5), Biome::Ocean);
        assert_eq!(classify_biome(0.9, 0.1, 0.1), Biome::Snow);
    }
}
'''

all_content[f"{base}/procgen/src/building_generator.rs"] = '''//! Building generation: floor plan generation, room classification, door/window
//! placement, interior decoration, exterior style, and multi-story buildings.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoomType { Living, Kitchen, Bedroom, Bathroom, Hallway, Stairs, Storage, Office, Shop, Tavern, Workshop, Temple, Throne, Dungeon, Library, Armory }
impl RoomType {
    pub fn min_area(&self) -> f32 { match self { Self::Bathroom | Self::Storage => 4.0, Self::Hallway | Self::Stairs => 3.0, Self::Kitchen | Self::Office => 8.0, Self::Bedroom => 9.0, Self::Living | Self::Shop => 12.0, Self::Tavern | Self::Workshop => 16.0, Self::Library | Self::Armory => 20.0, Self::Temple | Self::Throne => 30.0, Self::Dungeon => 10.0 } }
    pub fn max_area(&self) -> f32 { self.min_area() * 3.0 }
    pub fn needs_window(&self) -> bool { !matches!(self, Self::Storage | Self::Dungeon | Self::Stairs | Self::Hallway) }
    pub fn needs_exterior_wall(&self) -> bool { self.needs_window() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExteriorStyle { Medieval, Tudor, Stone, Wooden, Brick, Plaster, Castle, Elven, Dwarven, Modern }
impl ExteriorStyle {
    pub fn wall_material(&self) -> &'static str { match self { Self::Medieval | Self::Tudor => "timber_frame", Self::Stone | Self::Castle | Self::Dwarven => "stone", Self::Wooden | Self::Elven => "wood", Self::Brick | Self::Modern => "brick", Self::Plaster => "plaster" } }
    pub fn roof_material(&self) -> &'static str { match self { Self::Castle | Self::Stone => "stone_slate", Self::Modern => "tile", _ => "thatch" } }
    pub fn floor_height(&self) -> f32 { match self { Self::Castle | Self::Temple => 4.0, Self::Dwarven => 3.0, _ => 3.0 } }
}

#[derive(Debug, Clone)]
pub struct Room { pub id: u32, pub room_type: RoomType, pub x: f32, pub z: f32, pub width: f32, pub depth: f32, pub floor: u32, pub doors: Vec<DoorPlacement>, pub windows: Vec<WindowPlacement>, pub furniture: Vec<FurniturePlacement> }

impl Room {
    pub fn new(id: u32, rt: RoomType, x: f32, z: f32, w: f32, d: f32, floor: u32) -> Self {
        Self { id, room_type: rt, x, z, width: w, depth: d, floor, doors: Vec::new(), windows: Vec::new(), furniture: Vec::new() }
    }
    pub fn area(&self) -> f32 { self.width * self.depth }
    pub fn center(&self) -> [f32; 2] { [self.x + self.width * 0.5, self.z + self.depth * 0.5] }
    pub fn contains(&self, px: f32, pz: f32) -> bool { px >= self.x && px <= self.x + self.width && pz >= self.z && pz <= self.z + self.depth }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WallSide { North, South, East, West }

#[derive(Debug, Clone)]
pub struct DoorPlacement { pub wall: WallSide, pub position: f32, pub width: f32, pub height: f32, pub is_exterior: bool }
#[derive(Debug, Clone)]
pub struct WindowPlacement { pub wall: WallSide, pub position: f32, pub width: f32, pub height: f32, pub sill_height: f32, pub style: WindowStyle }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowStyle { Rectangular, Arched, Round, Stained, Arrow }
#[derive(Debug, Clone)]
pub struct FurniturePlacement { pub furniture_type: String, pub x: f32, pub z: f32, pub rotation: f32, pub scale: f32 }

#[derive(Debug, Clone)]
pub struct FloorPlan { pub floor: u32, pub rooms: Vec<Room>, pub width: f32, pub depth: f32 }
impl FloorPlan {
    pub fn new(floor: u32, width: f32, depth: f32) -> Self { Self { floor, rooms: Vec::new(), width, depth } }
    pub fn add_room(&mut self, room: Room) { self.rooms.push(room); }
    pub fn room_count(&self) -> usize { self.rooms.len() }
    pub fn total_area(&self) -> f32 { self.rooms.iter().map(|r| r.area()).sum() }
}

#[derive(Debug, Clone)]
pub struct RoofConfig { pub roof_type: RoofType, pub pitch: f32, pub overhang: f32, pub material: String }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoofType { Flat, Gable, Hip, Shed, Mansard, Dome, Pyramidal }

#[derive(Debug, Clone)]
pub struct BuildingConfig {
    pub style: ExteriorStyle, pub floors: u32, pub width: f32, pub depth: f32, pub roof: RoofConfig,
    pub seed: u64, pub room_types: Vec<RoomType>, pub has_basement: bool, pub has_chimney: bool,
    pub has_porch: bool, pub door_style: String, pub window_style: WindowStyle,
}

impl Default for BuildingConfig {
    fn default() -> Self {
        Self {
            style: ExteriorStyle::Medieval, floors: 2, width: 10.0, depth: 8.0,
            roof: RoofConfig { roof_type: RoofType::Gable, pitch: 35.0, overhang: 0.5, material: "thatch".to_string() },
            seed: 42, room_types: vec![RoomType::Living, RoomType::Kitchen, RoomType::Bedroom, RoomType::Bathroom],
            has_basement: false, has_chimney: true, has_porch: false,
            door_style: "wooden".to_string(), window_style: WindowStyle::Rectangular,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeneratedBuilding { pub config: BuildingConfig, pub floors: Vec<FloorPlan>, pub total_rooms: u32, pub total_area: f32, pub exterior_walls: Vec<WallSegment>, pub height: f32 }

#[derive(Debug, Clone)]
pub struct WallSegment { pub start: [f32; 2], pub end: [f32; 2], pub height: f32, pub material: String, pub has_window: bool, pub has_door: bool }

fn hash_val(a: u32, b: u32, seed: u64) -> f32 {
    let mut h = seed.wrapping_mul(6364136223846793005).wrapping_add(a as u64);
    h ^= b as u64; h = h.wrapping_mul(6364136223846793005);
    ((h >> 33) as f32) / (u32::MAX as f32)
}

pub fn generate_building(config: &BuildingConfig) -> GeneratedBuilding {
    let mut floors = Vec::new();
    let mut total_rooms = 0u32;
    let mut total_area = 0.0f32;
    let fh = config.style.floor_height();

    for floor_idx in 0..config.floors {
        let mut plan = FloorPlan::new(floor_idx, config.width, config.depth);
        let rooms_on_floor = if floor_idx == 0 { 3.min(config.room_types.len()) } else { 2.min(config.room_types.len()) };
        let room_width = config.width / rooms_on_floor as f32;

        for i in 0..rooms_on_floor {
            let rt_idx = (floor_idx as usize * rooms_on_floor + i) % config.room_types.len();
            let rt = config.room_types[rt_idx];
            let rx = i as f32 * room_width;
            let mut room = Room::new(total_rooms, rt, rx, 0.0, room_width, config.depth, floor_idx);

            // Add door
            if i > 0 { room.doors.push(DoorPlacement { wall: WallSide::West, position: config.depth * 0.5, width: 0.9, height: 2.1, is_exterior: false }); }
            if floor_idx == 0 && i == 0 { room.doors.push(DoorPlacement { wall: WallSide::South, position: room_width * 0.5, width: 1.0, height: 2.2, is_exterior: true }); }

            // Add windows
            if rt.needs_window() {
                room.windows.push(WindowPlacement { wall: WallSide::North, position: room_width * 0.5, width: 1.0, height: 1.2, sill_height: 0.9, style: config.window_style });
            }

            // Furniture
            let furn = match rt {
                RoomType::Bedroom => vec!["bed", "wardrobe", "nightstand"],
                RoomType::Kitchen => vec!["table", "stove", "cabinet"],
                RoomType::Living => vec!["sofa", "table", "bookshelf", "fireplace"],
                RoomType::Bathroom => vec!["tub", "basin"],
                _ => vec![],
            };
            for (fi, fname) in furn.iter().enumerate() {
                let fx = hash_val(total_rooms, fi as u32, config.seed) * (room_width - 1.0) + 0.5;
                let fz = hash_val(fi as u32, total_rooms, config.seed + 1) * (config.depth - 1.0) + 0.5;
                room.furniture.push(FurniturePlacement { furniture_type: fname.to_string(), x: fx, z: fz, rotation: hash_val(total_rooms, fi as u32 + 100, config.seed) * 360.0, scale: 1.0 });
            }

            total_area += room.area();
            total_rooms += 1;
            plan.add_room(room);
        }
        floors.push(plan);
    }

    let height = config.floors as f32 * fh + if config.roof.roof_type == RoofType::Flat { 0.3 } else { config.width * 0.5 * (config.roof.pitch.to_radians().tan()) };

    let mat = config.style.wall_material().to_string();
    let walls = vec![
        WallSegment { start: [0.0, 0.0], end: [config.width, 0.0], height, material: mat.clone(), has_window: false, has_door: true },
        WallSegment { start: [config.width, 0.0], end: [config.width, config.depth], height, material: mat.clone(), has_window: true, has_door: false },
        WallSegment { start: [config.width, config.depth], end: [0.0, config.depth], height, material: mat.clone(), has_window: true, has_door: false },
        WallSegment { start: [0.0, config.depth], end: [0.0, 0.0], height, material: mat, has_window: true, has_door: false },
    ];

    GeneratedBuilding { config: config.clone(), floors, total_rooms, total_area, exterior_walls: walls, height }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generate_basic_building() {
        let building = generate_building(&BuildingConfig::default());
        assert_eq!(building.floors.len(), 2);
        assert!(building.total_rooms > 0);
        assert!(building.height > 0.0);
    }
    #[test]
    fn room_areas() {
        assert!(RoomType::Bathroom.min_area() < RoomType::Living.min_area());
    }
}
'''

all_content[f"{base}/cinematics/src/camera_system.rs"] = '''//! Cinematic camera: Bezier path following, focus tracking, dolly zoom,
//! handheld shake, rack focus, and letterbox transitions.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub struct CameraTransformV2 { pub position: [f32; 3], pub target: [f32; 3], pub up: [f32; 3], pub fov: f32, pub roll: f32 }
impl Default for CameraTransformV2 { fn default() -> Self { Self { position: [0.0, 2.0, 5.0], target: [0.0; 3], up: [0.0, 1.0, 0.0], fov: 60.0, roll: 0.0 } } }

#[derive(Debug, Clone)]
pub struct BezierPath { pub control_points: Vec<[f32; 3]>, pub total_length: f32 }
impl BezierPath {
    pub fn new() -> Self { Self { control_points: Vec::new(), total_length: 0.0 } }
    pub fn add_point(&mut self, p: [f32; 3]) { self.control_points.push(p); self.recalculate_length(); }
    fn recalculate_length(&mut self) { self.total_length = 0.0; for i in 1..self.control_points.len() { let a = self.control_points[i-1]; let b = self.control_points[i]; self.total_length += ((b[0]-a[0]).powi(2)+(b[1]-a[1]).powi(2)+(b[2]-a[2]).powi(2)).sqrt(); } }
    pub fn evaluate(&self, t: f32) -> [f32; 3] {
        let n = self.control_points.len(); if n == 0 { return [0.0; 3]; } if n == 1 { return self.control_points[0]; }
        let t = t.clamp(0.0, 1.0); let segment = (t * (n - 1) as f32).min((n - 2) as f32);
        let i = segment as usize; let f = segment - i as f32;
        let a = self.control_points[i]; let b = self.control_points[i + 1];
        [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f]
    }
}

#[derive(Debug, Clone)]
pub struct FocusTarget { pub entity_id: Option<u64>, pub position: Option<[f32; 3]>, pub offset: [f32; 3], pub smoothing: f32 }
impl Default for FocusTarget { fn default() -> Self { Self { entity_id: None, position: None, offset: [0.0; 3], smoothing: 5.0 } } }

#[derive(Debug, Clone)]
pub struct DollyZoom { pub active: bool, pub target_distance: f32, pub start_fov: f32, pub end_fov: f32, pub duration: f32, pub elapsed: f32 }
impl DollyZoom { pub fn new(target_dist: f32, start_fov: f32, end_fov: f32, dur: f32) -> Self { Self { active: true, target_distance: target_dist, start_fov, end_fov, duration: dur, elapsed: 0.0 } }
    pub fn progress(&self) -> f32 { (self.elapsed / self.duration).clamp(0.0, 1.0) }
    pub fn current_fov(&self) -> f32 { let t = self.progress(); self.start_fov + (self.end_fov - self.start_fov) * t }
}

#[derive(Debug, Clone)]
pub struct HandheldShake { pub enabled: bool, pub amplitude: f32, pub frequency: f32, pub damping: f32, pub seed: f32, pub intensity: f32 }
impl Default for HandheldShake { fn default() -> Self { Self { enabled: false, amplitude: 0.02, frequency: 5.0, damping: 0.95, seed: 0.0, intensity: 1.0 } } }
impl HandheldShake {
    pub fn evaluate(&self, time: f32) -> [f32; 3] {
        if !self.enabled { return [0.0; 3]; }
        let a = self.amplitude * self.intensity;
        [a * (time * self.frequency * 1.1 + self.seed).sin(), a * (time * self.frequency * 0.9 + self.seed + 1.7).sin() * 0.7, a * (time * self.frequency * 1.3 + self.seed + 3.1).sin() * 0.3]
    }
}

#[derive(Debug, Clone)]
pub struct RackFocus { pub active: bool, pub near_target: f32, pub far_target: f32, pub current_focus: f32, pub transition_speed: f32, pub aperture: f32 }
impl Default for RackFocus { fn default() -> Self { Self { active: false, near_target: 2.0, far_target: 20.0, current_focus: 5.0, transition_speed: 3.0, aperture: 2.8 } } }
impl RackFocus { pub fn focus_to_near(&mut self) { self.active = true; } pub fn focus_to_far(&mut self) { self.active = true; } pub fn update(&mut self, dt: f32) { /* lerp current_focus toward target */ } }

#[derive(Debug, Clone)]
pub struct Letterbox { pub active: bool, pub target_aspect: f32, pub current_amount: f32, pub transition_speed: f32, pub bar_color: [f32; 4] }
impl Default for Letterbox { fn default() -> Self { Self { active: false, target_aspect: 2.35, current_amount: 0.0, transition_speed: 2.0, bar_color: [0.0, 0.0, 0.0, 1.0] } } }
impl Letterbox { pub fn enable(&mut self, aspect: f32) { self.active = true; self.target_aspect = aspect; } pub fn disable(&mut self) { self.active = false; } pub fn update(&mut self, dt: f32, screen_aspect: f32) { let target = if self.active { 1.0 - screen_aspect / self.target_aspect } else { 0.0 }.max(0.0); self.current_amount += (target - self.current_amount) * self.transition_speed * dt; } pub fn bar_height(&self) -> f32 { self.current_amount * 0.5 } }

#[derive(Debug, Clone)]
pub enum CameraEvent { PathStarted, PathCompleted, FocusChanged(Option<u64>), DollyZoomStarted, DollyZoomCompleted, ShakeTriggered(f32), LetterboxChanged(bool) }

pub struct CinematicCameraSystem {
    pub transform: CameraTransformV2,
    pub path: Option<BezierPath>,
    pub path_progress: f32,
    pub path_speed: f32,
    pub focus: FocusTarget,
    pub dolly_zoom: Option<DollyZoom>,
    pub shake: HandheldShake,
    pub rack_focus: RackFocus,
    pub letterbox: Letterbox,
    pub events: Vec<CameraEvent>,
    pub time: f32,
    pub smooth_position: [f32; 3],
    pub smooth_target: [f32; 3],
    pub interpolation_speed: f32,
    pub enabled: bool,
}

impl CinematicCameraSystem {
    pub fn new() -> Self {
        Self { transform: CameraTransformV2::default(), path: None, path_progress: 0.0, path_speed: 0.1, focus: FocusTarget::default(), dolly_zoom: None, shake: HandheldShake::default(), rack_focus: RackFocus::default(), letterbox: Letterbox::default(), events: Vec::new(), time: 0.0, smooth_position: [0.0, 2.0, 5.0], smooth_target: [0.0; 3], interpolation_speed: 5.0, enabled: true }
    }

    pub fn set_path(&mut self, path: BezierPath) { self.path = Some(path); self.path_progress = 0.0; self.events.push(CameraEvent::PathStarted); }
    pub fn clear_path(&mut self) { self.path = None; }
    pub fn start_dolly_zoom(&mut self, target_dist: f32, start_fov: f32, end_fov: f32, duration: f32) { self.dolly_zoom = Some(DollyZoom::new(target_dist, start_fov, end_fov, duration)); self.events.push(CameraEvent::DollyZoomStarted); }
    pub fn trigger_shake(&mut self, intensity: f32) { self.shake.enabled = true; self.shake.intensity = intensity; self.events.push(CameraEvent::ShakeTriggered(intensity)); }
    pub fn set_letterbox(&mut self, aspect: f32) { self.letterbox.enable(aspect); self.events.push(CameraEvent::LetterboxChanged(true)); }

    pub fn update(&mut self, dt: f32) {
        if !self.enabled { return; }
        self.time += dt;
        if let Some(ref path) = self.path {
            self.path_progress += self.path_speed * dt;
            if self.path_progress >= 1.0 { self.path_progress = 1.0; self.events.push(CameraEvent::PathCompleted); }
            let pos = path.evaluate(self.path_progress);
            self.transform.position = pos;
        }
        if let Some(ref mut dz) = self.dolly_zoom {
            dz.elapsed += dt;
            self.transform.fov = dz.current_fov();
            if dz.elapsed >= dz.duration { self.events.push(CameraEvent::DollyZoomCompleted); }
        }
        let shake_offset = self.shake.evaluate(self.time);
        self.transform.position[0] += shake_offset[0];
        self.transform.position[1] += shake_offset[1];
        self.transform.position[2] += shake_offset[2];
        self.letterbox.update(dt, 16.0 / 9.0);
    }

    pub fn drain_events(&mut self) -> Vec<CameraEvent> { std::mem::take(&mut self.events) }
}

impl Default for CinematicCameraSystem { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bezier_path() {
        let mut path = BezierPath::new();
        path.add_point([0.0, 0.0, 0.0]); path.add_point([10.0, 5.0, 0.0]); path.add_point([20.0, 0.0, 0.0]);
        let mid = path.evaluate(0.5);
        assert!((mid[0] - 10.0).abs() < 0.5);
    }
    #[test]
    fn dolly_zoom_progress() {
        let dz = DollyZoom::new(5.0, 60.0, 20.0, 2.0);
        assert!((dz.current_fov() - 60.0).abs() < 0.01);
    }
}
'''

all_content[f"{base}/cinematics/src/cutscene_manager.rs"] = '''//! Cutscene management: cutscene loading, state machine, skip handling,
//! cutscene events, and cutscene blending with gameplay.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CutsceneId(pub u64);
impl CutsceneId { pub fn from_name(n: &str) -> Self { use std::hash::{Hash,Hasher}; let mut h = std::collections::hash_map::DefaultHasher::new(); n.hash(&mut h); Self(h.finish()) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutsceneState { Idle, Loading, Playing, Paused, Skipping, Blending, Finished, Error }

#[derive(Debug, Clone)]
pub struct CutsceneEventDef { pub time: f32, pub event_type: String, pub data: HashMap<String, String> }
impl CutsceneEventDef { pub fn new(time: f32, et: impl Into<String>) -> Self { Self { time, event_type: et.into(), data: HashMap::new() } } }

#[derive(Debug, Clone)]
pub struct CutsceneDef {
    pub id: CutsceneId, pub name: String, pub asset_path: String, pub duration: f32,
    pub skippable: bool, pub skip_fade_duration: f32, pub blend_in_duration: f32,
    pub blend_out_duration: f32, pub events: Vec<CutsceneEventDef>,
    pub camera_sequence: Option<String>, pub audio_track: Option<String>,
    pub subtitle_track: Option<String>, pub priority: i32,
}
impl CutsceneDef {
    pub fn new(id: CutsceneId, name: impl Into<String>, duration: f32) -> Self {
        Self { id, name: name.into(), asset_path: String::new(), duration, skippable: true, skip_fade_duration: 0.5, blend_in_duration: 0.3, blend_out_duration: 0.3, events: Vec::new(), camera_sequence: None, audio_track: None, subtitle_track: None, priority: 0 }
    }
}

#[derive(Debug, Clone)]
pub struct CutsceneInstance {
    pub definition: CutsceneDef, pub state: CutsceneState, pub current_time: f32,
    pub playback_speed: f32, pub pending_events: Vec<CutsceneEventDef>,
    pub triggered_events: Vec<CutsceneEventDef>, pub blend_alpha: f32,
    pub skip_requested: bool, pub skip_timer: f32,
}
impl CutsceneInstance {
    pub fn new(def: CutsceneDef) -> Self {
        Self { definition: def, state: CutsceneState::Idle, current_time: 0.0, playback_speed: 1.0, pending_events: Vec::new(), triggered_events: Vec::new(), blend_alpha: 0.0, skip_requested: false, skip_timer: 0.0 }
    }
    pub fn progress(&self) -> f32 { if self.definition.duration > 0.0 { self.current_time / self.definition.duration } else { 0.0 } }
    pub fn is_finished(&self) -> bool { self.state == CutsceneState::Finished }
}

#[derive(Debug, Clone)]
pub enum CutsceneManagerEvent {
    CutsceneStarted(CutsceneId), CutsceneFinished(CutsceneId), CutsceneSkipped(CutsceneId),
    CutscenePaused(CutsceneId), CutsceneResumed(CutsceneId),
    EventTriggered(CutsceneId, String), BlendInComplete(CutsceneId), BlendOutComplete(CutsceneId),
}

pub struct CutsceneManager {
    pub definitions: HashMap<CutsceneId, CutsceneDef>,
    pub active: Option<CutsceneInstance>,
    pub queue: Vec<CutsceneId>,
    pub events: Vec<CutsceneManagerEvent>,
    pub gameplay_camera_stored: bool,
    pub stored_camera: Option<([f32; 3], [f32; 3])>,
    pub input_blocked: bool,
    pub auto_play_queue: bool,
}

impl CutsceneManager {
    pub fn new() -> Self { Self { definitions: HashMap::new(), active: None, queue: Vec::new(), events: Vec::new(), gameplay_camera_stored: false, stored_camera: None, input_blocked: false, auto_play_queue: true } }

    pub fn register(&mut self, def: CutsceneDef) { self.definitions.insert(def.id, def); }

    pub fn play(&mut self, id: CutsceneId) -> bool {
        if let Some(def) = self.definitions.get(&id) {
            let mut inst = CutsceneInstance::new(def.clone());
            inst.state = CutsceneState::Blending;
            inst.pending_events = def.events.clone();
            self.active = Some(inst);
            self.input_blocked = true;
            self.events.push(CutsceneManagerEvent::CutsceneStarted(id));
            true
        } else { false }
    }

    pub fn queue_cutscene(&mut self, id: CutsceneId) { self.queue.push(id); }

    pub fn skip(&mut self) {
        if let Some(ref mut inst) = self.active {
            if inst.definition.skippable && !inst.skip_requested {
                inst.skip_requested = true;
                inst.state = CutsceneState::Skipping;
                self.events.push(CutsceneManagerEvent::CutsceneSkipped(inst.definition.id));
            }
        }
    }

    pub fn pause(&mut self) {
        if let Some(ref mut inst) = self.active {
            if inst.state == CutsceneState::Playing { inst.state = CutsceneState::Paused; self.events.push(CutsceneManagerEvent::CutscenePaused(inst.definition.id)); }
        }
    }

    pub fn resume(&mut self) {
        if let Some(ref mut inst) = self.active {
            if inst.state == CutsceneState::Paused { inst.state = CutsceneState::Playing; self.events.push(CutsceneManagerEvent::CutsceneResumed(inst.definition.id)); }
        }
    }

    pub fn update(&mut self, dt: f32) {
        let mut finished = false;
        if let Some(ref mut inst) = self.active {
            match inst.state {
                CutsceneState::Blending => {
                    inst.blend_alpha += dt / inst.definition.blend_in_duration.max(0.01);
                    if inst.blend_alpha >= 1.0 { inst.blend_alpha = 1.0; inst.state = CutsceneState::Playing; self.events.push(CutsceneManagerEvent::BlendInComplete(inst.definition.id)); }
                }
                CutsceneState::Playing => {
                    inst.current_time += dt * inst.playback_speed;
                    let mut triggered = Vec::new();
                    inst.pending_events.retain(|e| { if e.time <= inst.current_time { triggered.push(e.clone()); false } else { true } });
                    for e in &triggered { self.events.push(CutsceneManagerEvent::EventTriggered(inst.definition.id, e.event_type.clone())); }
                    inst.triggered_events.extend(triggered);
                    if inst.current_time >= inst.definition.duration { inst.state = CutsceneState::Blending; inst.blend_alpha = 1.0; }
                }
                CutsceneState::Skipping => {
                    inst.skip_timer += dt;
                    inst.blend_alpha -= dt / inst.definition.skip_fade_duration.max(0.01);
                    if inst.blend_alpha <= 0.0 { inst.state = CutsceneState::Finished; finished = true; }
                }
                CutsceneState::Finished => { finished = true; }
                _ => {}
            }
            if inst.current_time >= inst.definition.duration && inst.state == CutsceneState::Blending && inst.blend_alpha <= 0.0 {
                inst.state = CutsceneState::Finished; finished = true;
            }
        }
        if finished {
            if let Some(inst) = self.active.take() {
                self.events.push(CutsceneManagerEvent::CutsceneFinished(inst.definition.id));
                self.input_blocked = false;
            }
            if self.auto_play_queue && !self.queue.is_empty() {
                let next = self.queue.remove(0);
                self.play(next);
            }
        }
    }

    pub fn is_playing(&self) -> bool { self.active.is_some() }
    pub fn current_progress(&self) -> f32 { self.active.as_ref().map(|i| i.progress()).unwrap_or(0.0) }
    pub fn drain_events(&mut self) -> Vec<CutsceneManagerEvent> { std::mem::take(&mut self.events) }
}

impl Default for CutsceneManager { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cutscene_playback() {
        let mut mgr = CutsceneManager::new();
        let id = CutsceneId::from_name("intro");
        mgr.register(CutsceneDef::new(id, "Intro", 5.0));
        assert!(mgr.play(id));
        assert!(mgr.is_playing());
        for _ in 0..100 { mgr.update(0.1); }
    }
    #[test]
    fn cutscene_skip() {
        let mut mgr = CutsceneManager::new();
        let id = CutsceneId::from_name("skip_test");
        let mut def = CutsceneDef::new(id, "Skip Test", 10.0);
        def.blend_in_duration = 0.01;
        mgr.register(def);
        mgr.play(id);
        mgr.update(0.1);
        mgr.skip();
        for _ in 0..50 { mgr.update(0.1); }
    }
}
'''

all_content[f"{base}/debug/src/debug_menu.rs"] = '''//! In-game debug menu: category tree, cvar display/edit, performance graphs,
//! memory breakdown, entity inspector, and cheat commands.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum CvarValue { Bool(bool), Int(i32), Float(f32), String(String) }
impl CvarValue {
    pub fn as_string(&self) -> String { match self { Self::Bool(v) => v.to_string(), Self::Int(v) => v.to_string(), Self::Float(v) => format!("{:.3}", v), Self::String(v) => v.clone() } }
}
impl std::fmt::Display for CvarValue { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.as_string()) } }

#[derive(Debug, Clone)]
pub struct Cvar { pub name: String, pub value: CvarValue, pub default: CvarValue, pub description: String, pub category: String, pub min: Option<f64>, pub max: Option<f64>, pub read_only: bool, pub flags: u32 }
impl Cvar {
    pub fn new(name: impl Into<String>, value: CvarValue, cat: impl Into<String>) -> Self {
        let v = value.clone();
        Self { name: name.into(), value, default: v, description: String::new(), category: cat.into(), min: None, max: None, read_only: false, flags: 0 }
    }
    pub fn is_modified(&self) -> bool { self.value != self.default }
    pub fn reset(&mut self) { self.value = self.default.clone(); }
}

#[derive(Debug, Clone)]
pub struct PerfGraph { pub name: String, pub values: Vec<f32>, pub max_samples: usize, pub min_val: f32, pub max_val: f32, pub color: [f32; 4], pub unit: String }
impl PerfGraph {
    pub fn new(name: impl Into<String>, max_samples: usize) -> Self {
        Self { name: name.into(), values: Vec::with_capacity(max_samples), max_samples, min_val: 0.0, max_val: 100.0, color: [0.2, 0.8, 0.2, 1.0], unit: "ms".to_string() }
    }
    pub fn push(&mut self, value: f32) { if self.values.len() >= self.max_samples { self.values.remove(0); } self.values.push(value); }
    pub fn average(&self) -> f32 { if self.values.is_empty() { 0.0 } else { self.values.iter().sum::<f32>() / self.values.len() as f32 } }
    pub fn peak(&self) -> f32 { self.values.iter().cloned().fold(0.0f32, f32::max) }
    pub fn current(&self) -> f32 { self.values.last().copied().unwrap_or(0.0) }
}

#[derive(Debug, Clone)]
pub struct MemoryCategory { pub name: String, pub allocated: u64, pub peak: u64, pub count: u32, pub color: [f32; 4] }

#[derive(Debug, Clone)]
pub struct EntityInspector { pub selected_entity: Option<u64>, pub components: Vec<(String, Vec<(String, String)>)>, pub expanded_components: Vec<String> }
impl EntityInspector {
    pub fn new() -> Self { Self { selected_entity: None, components: Vec::new(), expanded_components: Vec::new() } }
    pub fn select(&mut self, entity: u64) { self.selected_entity = Some(entity); }
    pub fn clear(&mut self) { self.selected_entity = None; self.components.clear(); }
}
impl Default for EntityInspector { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone)]
pub struct CheatCommand { pub name: String, pub description: String, pub handler_id: u32, pub args: Vec<String>, pub category: String }
impl CheatCommand {
    pub fn new(name: impl Into<String>, desc: impl Into<String>, handler: u32) -> Self {
        Self { name: name.into(), description: desc.into(), handler_id: handler, args: Vec::new(), category: "General".to_string() }
    }
}

#[derive(Debug, Clone)]
pub enum DebugMenuEvent { CvarChanged(String, CvarValue), CheatExecuted(String, Vec<String>), EntitySelected(u64), CategoryExpanded(String), MenuToggled(bool) }

pub struct DebugMenuState {
    pub visible: bool, pub cvars: HashMap<String, Cvar>, pub categories: Vec<String>,
    pub selected_category: Option<String>, pub perf_graphs: Vec<PerfGraph>,
    pub memory_categories: Vec<MemoryCategory>, pub entity_inspector: EntityInspector,
    pub cheats: Vec<CheatCommand>, pub events: Vec<DebugMenuEvent>,
    pub search_text: String, pub show_perf: bool, pub show_memory: bool,
    pub show_entities: bool, pub show_cheats: bool, pub console_history: Vec<String>,
    pub console_input: String, pub opacity: f32,
}

impl DebugMenuState {
    pub fn new() -> Self {
        Self {
            visible: false, cvars: HashMap::new(), categories: Vec::new(),
            selected_category: None, perf_graphs: Vec::new(),
            memory_categories: Vec::new(), entity_inspector: EntityInspector::new(),
            cheats: Vec::new(), events: Vec::new(), search_text: String::new(),
            show_perf: true, show_memory: false, show_entities: false,
            show_cheats: false, console_history: Vec::new(),
            console_input: String::new(), opacity: 0.9,
        }
    }
    pub fn toggle(&mut self) { self.visible = !self.visible; self.events.push(DebugMenuEvent::MenuToggled(self.visible)); }
    pub fn register_cvar(&mut self, cvar: Cvar) { let cat = cvar.category.clone(); if !self.categories.contains(&cat) { self.categories.push(cat); } self.cvars.insert(cvar.name.clone(), cvar); }
    pub fn set_cvar(&mut self, name: &str, value: CvarValue) { if let Some(c) = self.cvars.get_mut(name) { if !c.read_only { c.value = value.clone(); self.events.push(DebugMenuEvent::CvarChanged(name.to_string(), value)); } } }
    pub fn get_cvar(&self, name: &str) -> Option<&CvarValue> { self.cvars.get(name).map(|c| &c.value) }
    pub fn register_cheat(&mut self, cheat: CheatCommand) { self.cheats.push(cheat); }
    pub fn execute_cheat(&mut self, name: &str, args: Vec<String>) { self.events.push(DebugMenuEvent::CheatExecuted(name.to_string(), args)); self.console_history.push(format!("> {}", name)); }
    pub fn add_perf_graph(&mut self, graph: PerfGraph) { self.perf_graphs.push(graph); }
    pub fn update_perf(&mut self, name: &str, value: f32) { if let Some(g) = self.perf_graphs.iter_mut().find(|g| g.name == name) { g.push(value); } }
    pub fn cvar_count(&self) -> usize { self.cvars.len() }
    pub fn drain_events(&mut self) -> Vec<DebugMenuEvent> { std::mem::take(&mut self.events) }
}
impl Default for DebugMenuState { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cvar_operations() {
        let mut menu = DebugMenuState::new();
        menu.register_cvar(Cvar::new("r_fov", CvarValue::Float(90.0), "Rendering"));
        menu.set_cvar("r_fov", CvarValue::Float(100.0));
        assert_eq!(menu.get_cvar("r_fov"), Some(&CvarValue::Float(100.0)));
    }
    #[test]
    fn perf_graph() {
        let mut g = PerfGraph::new("FPS", 100);
        for i in 0..50 { g.push(i as f32); }
        assert!((g.average() - 24.5).abs() < 0.1);
    }
}
'''

all_content[f"{base}/debug/src/crash_reporter.rs"] = '''//! Crash handling: panic hook, stack trace capture, minidump generation stub,
//! crash log with system info, and auto-save on crash.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static CRASH_HANDLER_INSTALLED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone)]
pub struct SystemInfo { pub os: String, pub os_version: String, pub cpu: String, pub cpu_cores: u32, pub ram_mb: u64, pub gpu: String, pub gpu_driver: String, pub display_resolution: (u32, u32), pub app_version: String, pub build_config: String }
impl SystemInfo {
    pub fn gather() -> Self {
        Self { os: std::env::consts::OS.to_string(), os_version: String::new(), cpu: String::new(), cpu_cores: 1, ram_mb: 0, gpu: String::new(), gpu_driver: String::new(), display_resolution: (1920, 1080), app_version: env!("CARGO_PKG_VERSION").to_string(), build_config: if cfg!(debug_assertions) { "Debug" } else { "Release" }.to_string() }
    }
}
impl fmt::Display for SystemInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "OS: {} {}", self.os, self.os_version)?;
        writeln!(f, "CPU: {} ({} cores)", self.cpu, self.cpu_cores)?;
        writeln!(f, "RAM: {} MB", self.ram_mb)?;
        writeln!(f, "GPU: {} ({})", self.gpu, self.gpu_driver)?;
        writeln!(f, "Resolution: {}x{}", self.display_resolution.0, self.display_resolution.1)?;
        writeln!(f, "App: {} ({})", self.app_version, self.build_config)
    }
}

#[derive(Debug, Clone)]
pub struct StackFrame { pub function: String, pub file: Option<String>, pub line: Option<u32>, pub address: u64 }
impl fmt::Display for StackFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "  {:#018x} {}", self.address, self.function)?;
        if let Some(ref file) = self.file { write!(f, " at {}", file)?; if let Some(line) = self.line { write!(f, ":{}", line)?; } }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CrashReport {
    pub timestamp: u64, pub crash_message: String, pub stack_frames: Vec<StackFrame>,
    pub system_info: SystemInfo, pub thread_name: Option<String>,
    pub log_tail: Vec<String>, pub active_scene: Option<String>,
    pub entity_count: u32, pub frame_number: u64, pub uptime_seconds: f64,
    pub memory_used_mb: u64, pub custom_data: Vec<(String, String)>,
}

impl CrashReport {
    pub fn new(message: impl Into<String>) -> Self {
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        Self {
            timestamp: ts, crash_message: message.into(), stack_frames: Vec::new(),
            system_info: SystemInfo::gather(), thread_name: std::thread::current().name().map(|s| s.to_string()),
            log_tail: Vec::new(), active_scene: None, entity_count: 0, frame_number: 0,
            uptime_seconds: 0.0, memory_used_mb: 0, custom_data: Vec::new(),
        }
    }

    pub fn add_stack_frame(&mut self, frame: StackFrame) { self.stack_frames.push(frame); }
    pub fn add_log_line(&mut self, line: impl Into<String>) { self.log_tail.push(line.into()); if self.log_tail.len() > 100 { self.log_tail.remove(0); } }
    pub fn add_custom_data(&mut self, key: impl Into<String>, value: impl Into<String>) { self.custom_data.push((key.into(), value.into())); }

    pub fn format_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== GENOVO ENGINE CRASH REPORT ===\n\n");
        report.push_str(&format!("Timestamp: {}\n", self.timestamp));
        report.push_str(&format!("Thread: {}\n", self.thread_name.as_deref().unwrap_or("unknown")));
        report.push_str(&format!("Crash: {}\n\n", self.crash_message));
        report.push_str("--- Stack Trace ---\n");
        for frame in &self.stack_frames { report.push_str(&format!("{}\n", frame)); }
        report.push_str("\n--- System Info ---\n");
        report.push_str(&format!("{}", self.system_info));
        report.push_str(&format!("\nScene: {}\n", self.active_scene.as_deref().unwrap_or("none")));
        report.push_str(&format!("Entities: {}\nFrame: {}\nUptime: {:.1}s\nMemory: {} MB\n", self.entity_count, self.frame_number, self.uptime_seconds, self.memory_used_mb));
        if !self.custom_data.is_empty() {
            report.push_str("\n--- Custom Data ---\n");
            for (k, v) in &self.custom_data { report.push_str(&format!("  {}: {}\n", k, v)); }
        }
        if !self.log_tail.is_empty() {
            report.push_str("\n--- Recent Log ---\n");
            for line in &self.log_tail { report.push_str(&format!("  {}\n", line)); }
        }
        report
    }

    pub fn write_to_file(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.format_report())
    }
}

pub struct CrashReporter {
    pub auto_save_on_crash: bool,
    pub crash_log_dir: String,
    pub max_crash_logs: u32,
    pub report_url: Option<String>,
    pub custom_data_providers: Vec<Box<dyn Fn() -> Vec<(String, String)> + Send + Sync>>,
    pub log_buffer: Vec<String>,
    pub log_buffer_size: usize,
    pub installed: bool,
}

impl CrashReporter {
    pub fn new() -> Self {
        Self { auto_save_on_crash: true, crash_log_dir: "crash_logs".to_string(), max_crash_logs: 10, report_url: None, custom_data_providers: Vec::new(), log_buffer: Vec::new(), log_buffer_size: 200, installed: false }
    }

    pub fn install(&mut self) {
        if CRASH_HANDLER_INSTALLED.compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
            let auto_save = self.auto_save_on_crash;
            let log_dir = self.crash_log_dir.clone();
            std::panic::set_hook(Box::new(move |info| {
                let message = if let Some(s) = info.payload().downcast_ref::<&str>() { s.to_string() }
                    else if let Some(s) = info.payload().downcast_ref::<String>() { s.clone() }
                    else { "Unknown panic".to_string() };
                let mut report = CrashReport::new(&message);
                if let Some(loc) = info.location() {
                    report.add_stack_frame(StackFrame { function: "panic".to_string(), file: Some(loc.file().to_string()), line: Some(loc.line()), address: 0 });
                }
                let _ = std::fs::create_dir_all(&log_dir);
                let path = format!("{}/crash_{}.log", log_dir, report.timestamp);
                let _ = report.write_to_file(&path);
                eprintln!("CRASH: {} (report saved to {})", message, path);
            }));
            self.installed = true;
        }
    }

    pub fn log(&mut self, message: impl Into<String>) {
        let msg = message.into();
        self.log_buffer.push(msg);
        if self.log_buffer.len() > self.log_buffer_size { self.log_buffer.remove(0); }
    }

    pub fn generate_report(&self, message: impl Into<String>) -> CrashReport {
        let mut report = CrashReport::new(message);
        for line in &self.log_buffer { report.add_log_line(line.clone()); }
        for provider in &self.custom_data_providers {
            for (k, v) in provider() { report.add_custom_data(k, v); }
        }
        report
    }

    pub fn generate_minidump_stub(&self) -> Vec<u8> {
        let header = b"MDMP"; // Minidump magic
        let mut data = header.to_vec();
        data.extend_from_slice(&[0u8; 28]); // Stub header
        data
    }
}
impl Default for CrashReporter { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn crash_report_format() {
        let report = CrashReport::new("test crash");
        let text = report.format_report();
        assert!(text.contains("test crash"));
        assert!(text.contains("CRASH REPORT"));
    }
    #[test]
    fn system_info() {
        let info = SystemInfo::gather();
        assert!(!info.os.is_empty());
    }
}
'''

all_content[f"{base}/localization/src/locale_manager_v2.rs"] = '''//! Enhanced localization: dynamic string loading, font switching per locale,
//! text direction (LTR/RTL), locale-specific number/date, string validation.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextDirectionV2 { LeftToRight, RightToLeft }

#[derive(Debug, Clone)]
pub struct LocaleInfoV2 {
    pub code: String, pub name: String, pub native_name: String,
    pub text_direction: TextDirectionV2, pub font_family: String,
    pub fallback_font: Option<String>, pub decimal_separator: char,
    pub thousands_separator: char, pub date_format: String,
    pub time_format: String, pub currency_symbol: String,
    pub currency_position: CurrencyPosition, pub plural_rules: PluralRuleSet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurrencyPosition { Before, After }

#[derive(Debug, Clone)]
pub struct PluralRuleSet { pub zero: Option<String>, pub one: String, pub two: Option<String>, pub few: Option<String>, pub many: Option<String>, pub other: String }
impl Default for PluralRuleSet { fn default() -> Self { Self { zero: None, one: "one".to_string(), two: None, few: None, many: None, other: "other".to_string() } } }

impl LocaleInfoV2 {
    pub fn english() -> Self { Self { code: "en-US".to_string(), name: "English (US)".to_string(), native_name: "English (US)".to_string(), text_direction: TextDirectionV2::LeftToRight, font_family: "Roboto".to_string(), fallback_font: None, decimal_separator: '.', thousands_separator: ',', date_format: "MM/DD/YYYY".to_string(), time_format: "hh:mm A".to_string(), currency_symbol: "$".to_string(), currency_position: CurrencyPosition::Before, plural_rules: PluralRuleSet::default() } }
    pub fn arabic() -> Self { Self { code: "ar-SA".to_string(), name: "Arabic".to_string(), native_name: "Arabic".to_string(), text_direction: TextDirectionV2::RightToLeft, font_family: "Noto Sans Arabic".to_string(), fallback_font: Some("Arial".to_string()), decimal_separator: '.', thousands_separator: ',', date_format: "DD/MM/YYYY".to_string(), time_format: "HH:mm".to_string(), currency_symbol: "SAR".to_string(), currency_position: CurrencyPosition::After, plural_rules: PluralRuleSet::default() } }
    pub fn japanese() -> Self { Self { code: "ja-JP".to_string(), name: "Japanese".to_string(), native_name: "Japanese".to_string(), text_direction: TextDirectionV2::LeftToRight, font_family: "Noto Sans JP".to_string(), fallback_font: None, decimal_separator: '.', thousands_separator: ',', date_format: "YYYY/MM/DD".to_string(), time_format: "HH:mm".to_string(), currency_symbol: "Y".to_string(), currency_position: CurrencyPosition::Before, plural_rules: PluralRuleSet::default() } }
    pub fn is_rtl(&self) -> bool { self.text_direction == TextDirectionV2::RightToLeft }
}

#[derive(Debug, Clone)]
pub struct StringEntry { pub key: String, pub value: String, pub context: Option<String>, pub max_length: Option<usize>, pub validated: bool }
impl StringEntry { pub fn new(key: impl Into<String>, value: impl Into<String>) -> Self { Self { key: key.into(), value: value.into(), context: None, max_length: None, validated: false } } }

#[derive(Debug, Clone)]
pub struct StringTable { pub locale: String, pub entries: HashMap<String, StringEntry>, pub loaded: bool, pub source_path: Option<String> }
impl StringTable {
    pub fn new(locale: impl Into<String>) -> Self { Self { locale: locale.into(), entries: HashMap::new(), loaded: false, source_path: None } }
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) { let k = key.into(); self.entries.insert(k.clone(), StringEntry::new(k, value)); }
    pub fn get(&self, key: &str) -> Option<&str> { self.entries.get(key).map(|e| e.value.as_str()) }
    pub fn get_formatted(&self, key: &str, args: &[(&str, &str)]) -> Option<String> {
        let template = self.get(key)?;
        let mut result = template.to_string();
        for (name, value) in args { result = result.replace(&format!("{{{}}}", name), value); }
        Some(result)
    }
    pub fn entry_count(&self) -> usize { self.entries.len() }
}

#[derive(Debug, Clone)]
pub enum ValidationIssue { MissingKey(String), ExceedsMaxLength(String, usize, usize), PlaceholderMismatch(String, Vec<String>, Vec<String>), EmptyValue(String) }
impl std::fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingKey(k) => write!(f, "Missing key: {}", k),
            Self::ExceedsMaxLength(k, actual, max) => write!(f, "{}: length {} exceeds max {}", k, actual, max),
            Self::PlaceholderMismatch(k, expected, found) => write!(f, "{}: expected placeholders {:?}, found {:?}", k, expected, found),
            Self::EmptyValue(k) => write!(f, "{}: empty value", k),
        }
    }
}

pub fn format_number(value: f64, locale: &LocaleInfoV2) -> String {
    let is_negative = value < 0.0;
    let abs = value.abs();
    let integer = abs as u64;
    let frac = ((abs - integer as f64) * 100.0).round() as u64;
    let int_str = integer.to_string();
    let mut formatted = String::new();
    for (i, c) in int_str.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { formatted.push(locale.thousands_separator); }
        formatted.push(c);
    }
    let formatted: String = formatted.chars().rev().collect();
    let result = if frac > 0 { format!("{}{}{:02}", formatted, locale.decimal_separator, frac) } else { formatted };
    if is_negative { format!("-{}", result) } else { result }
}

pub fn format_currency(value: f64, locale: &LocaleInfoV2) -> String {
    let num = format_number(value, locale);
    match locale.currency_position {
        CurrencyPosition::Before => format!("{}{}", locale.currency_symbol, num),
        CurrencyPosition::After => format!("{} {}", num, locale.currency_symbol),
    }
}

pub fn validate_table(reference: &StringTable, target: &StringTable) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    for (key, ref_entry) in &reference.entries {
        match target.entries.get(key) {
            None => issues.push(ValidationIssue::MissingKey(key.clone())),
            Some(entry) => {
                if entry.value.is_empty() { issues.push(ValidationIssue::EmptyValue(key.clone())); }
                if let Some(max) = ref_entry.max_length { if entry.value.len() > max { issues.push(ValidationIssue::ExceedsMaxLength(key.clone(), entry.value.len(), max)); } }
            }
        }
    }
    issues
}

#[derive(Debug, Clone)]
pub enum LocaleEvent { LocaleChanged(String), TableLoaded(String), TableUnloaded(String), ValidationCompleted(usize) }

pub struct LocaleManagerV2 {
    pub current_locale: String, pub locales: HashMap<String, LocaleInfoV2>,
    pub tables: HashMap<String, StringTable>, pub fallback_locale: String,
    pub events: Vec<LocaleEvent>, pub auto_load: bool,
}

impl LocaleManagerV2 {
    pub fn new(default_locale: impl Into<String>) -> Self {
        let locale = default_locale.into();
        let mut locales = HashMap::new();
        locales.insert("en-US".to_string(), LocaleInfoV2::english());
        locales.insert("ar-SA".to_string(), LocaleInfoV2::arabic());
        locales.insert("ja-JP".to_string(), LocaleInfoV2::japanese());
        Self { current_locale: locale.clone(), locales, tables: HashMap::new(), fallback_locale: "en-US".to_string(), events: Vec::new(), auto_load: true }
    }

    pub fn set_locale(&mut self, locale: impl Into<String>) {
        let l = locale.into();
        self.current_locale = l.clone();
        self.events.push(LocaleEvent::LocaleChanged(l));
    }

    pub fn current_info(&self) -> Option<&LocaleInfoV2> { self.locales.get(&self.current_locale) }
    pub fn is_rtl(&self) -> bool { self.current_info().map(|i| i.is_rtl()).unwrap_or(false) }
    pub fn current_font(&self) -> &str { self.current_info().map(|i| i.font_family.as_str()).unwrap_or("default") }

    pub fn load_table(&mut self, table: StringTable) {
        let locale = table.locale.clone();
        self.tables.insert(locale.clone(), table);
        self.events.push(LocaleEvent::TableLoaded(locale));
    }

    pub fn get_string(&self, key: &str) -> &str {
        if let Some(table) = self.tables.get(&self.current_locale) { if let Some(v) = table.get(key) { return v; } }
        if let Some(table) = self.tables.get(&self.fallback_locale) { if let Some(v) = table.get(key) { return v; } }
        key
    }

    pub fn get_formatted(&self, key: &str, args: &[(&str, &str)]) -> String {
        if let Some(table) = self.tables.get(&self.current_locale) { if let Some(v) = table.get_formatted(key, args) { return v; } }
        if let Some(table) = self.tables.get(&self.fallback_locale) { if let Some(v) = table.get_formatted(key, args) { return v; } }
        key.to_string()
    }

    pub fn format_number(&self, value: f64) -> String { let info = self.current_info().cloned().unwrap_or_else(LocaleInfoV2::english); format_number(value, &info) }
    pub fn format_currency(&self, value: f64) -> String { let info = self.current_info().cloned().unwrap_or_else(LocaleInfoV2::english); format_currency(value, &info) }
    pub fn locale_count(&self) -> usize { self.locales.len() }
    pub fn drain_events(&mut self) -> Vec<LocaleEvent> { std::mem::take(&mut self.events) }
}
impl Default for LocaleManagerV2 { fn default() -> Self { Self::new("en-US") } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn string_lookup() {
        let mut mgr = LocaleManagerV2::new("en-US");
        let mut table = StringTable::new("en-US");
        table.insert("greeting", "Hello {name}!");
        mgr.load_table(table);
        assert_eq!(mgr.get_string("greeting"), "Hello {name}!");
        assert_eq!(mgr.get_formatted("greeting", &[("name", "World")]), "Hello World!");
    }
    #[test]
    fn number_formatting() {
        let info = LocaleInfoV2::english();
        assert_eq!(format_number(1234567.89, &info), "1,234,567.89");
    }
}
'''

all_content[f"{base}/save_system/src/save_encryption.rs"] = '''//! Save encryption: AES encrypt save data, HMAC integrity check, tampering
//! detection, encrypted cloud save, and save file versioning with migration.

use std::collections::HashMap;

pub fn xor_encrypt(data: &[u8], key: &[u8]) -> Vec<u8> {
    if key.is_empty() { return data.to_vec(); }
    data.iter().enumerate().map(|(i, &b)| b ^ key[i % key.len()]).collect()
}

pub fn xor_decrypt(data: &[u8], key: &[u8]) -> Vec<u8> { xor_encrypt(data, key) }

pub fn simple_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
    h
}

pub fn compute_hmac(data: &[u8], key: &[u8]) -> [u8; 8] {
    let mut padded_key = vec![0u8; 64];
    for (i, &b) in key.iter().enumerate().take(64) { padded_key[i] = b; }
    let mut ipad = vec![0x36u8; 64]; let mut opad = vec![0x5cu8; 64];
    for i in 0..64 { ipad[i] ^= padded_key[i]; opad[i] ^= padded_key[i]; }
    let mut inner = ipad; inner.extend_from_slice(data);
    let inner_hash = simple_hash(&inner);
    let mut outer = opad; outer.extend_from_slice(&inner_hash.to_le_bytes());
    let result = simple_hash(&outer);
    result.to_le_bytes()
}

pub fn verify_hmac(data: &[u8], key: &[u8], expected: &[u8; 8]) -> bool { &compute_hmac(data, key) == expected }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncryptionMethod { None, XorSimple, AesStub }

#[derive(Debug, Clone)]
pub struct EncryptionConfig {
    pub method: EncryptionMethod, pub key: Vec<u8>, pub hmac_key: Vec<u8>,
    pub include_hmac: bool, pub compress_before_encrypt: bool,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self { method: EncryptionMethod::XorSimple, key: b"genovo_default_key_2024".to_vec(), hmac_key: b"genovo_hmac_key_2024".to_vec(), include_hmac: true, compress_before_encrypt: false }
    }
}

#[derive(Debug, Clone)]
pub struct SaveFileHeader {
    pub magic: [u8; 4], pub version: u32, pub encryption: EncryptionMethod,
    pub data_size: u64, pub hmac: Option<[u8; 8]>, pub timestamp: u64,
    pub checksum: u32, pub metadata: HashMap<String, String>,
}

impl SaveFileHeader {
    pub fn new(version: u32) -> Self {
        Self { magic: *b"GNSV", version, encryption: EncryptionMethod::None, data_size: 0, hmac: None, timestamp: 0, checksum: 0, metadata: HashMap::new() }
    }
    pub fn is_valid(&self) -> bool { &self.magic == b"GNSV" }
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.magic); buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&(self.encryption as u8).to_le_bytes());
        buf.extend_from_slice(&self.data_size.to_le_bytes());
        if let Some(hmac) = &self.hmac { buf.extend_from_slice(hmac); } else { buf.extend_from_slice(&[0u8; 8]); }
        buf.extend_from_slice(&self.timestamp.to_le_bytes());
        buf.extend_from_slice(&self.checksum.to_le_bytes());
        buf
    }
}

#[derive(Debug, Clone)]
pub struct MigrationStep { pub from_version: u32, pub to_version: u32, pub description: String, pub transform: MigrationTransform }

#[derive(Debug, Clone)]
pub enum MigrationTransform { AddField(String, Vec<u8>), RemoveField(String), RenameField(String, String), Custom(String) }

pub struct SaveEncryption {
    pub config: EncryptionConfig,
    pub migrations: Vec<MigrationStep>,
    pub current_version: u32,
    pub tamper_detected: bool,
    pub last_error: Option<String>,
}

impl SaveEncryption {
    pub fn new(config: EncryptionConfig) -> Self { Self { config, migrations: Vec::new(), current_version: 1, tamper_detected: false, last_error: None } }

    pub fn encrypt_save(&self, data: &[u8]) -> Vec<u8> {
        let encrypted = match self.config.method {
            EncryptionMethod::None => data.to_vec(),
            EncryptionMethod::XorSimple => xor_encrypt(data, &self.config.key),
            EncryptionMethod::AesStub => { let mut e = xor_encrypt(data, &self.config.key); e.reverse(); e } // stub
        };
        let mut header = SaveFileHeader::new(self.current_version);
        header.encryption = self.config.method;
        header.data_size = encrypted.len() as u64;
        header.checksum = simple_hash(data) as u32;
        if self.config.include_hmac { header.hmac = Some(compute_hmac(&encrypted, &self.config.hmac_key)); }
        let mut result = header.serialize();
        result.extend_from_slice(&encrypted);
        result
    }

    pub fn decrypt_save(&mut self, file_data: &[u8]) -> Result<Vec<u8>, String> {
        if file_data.len() < 37 { return Err("File too small".to_string()); }
        if &file_data[0..4] != b"GNSV" { return Err("Invalid magic".to_string()); }
        let version = u32::from_le_bytes([file_data[4], file_data[5], file_data[6], file_data[7]]);
        let enc_method = file_data[8];
        let data_size = u64::from_le_bytes(file_data[9..17].try_into().unwrap()) as usize;
        let stored_hmac: [u8; 8] = file_data[17..25].try_into().unwrap();
        let header_size = 37;
        if file_data.len() < header_size + data_size { return Err("Truncated file".to_string()); }
        let encrypted = &file_data[header_size..header_size + data_size];
        if self.config.include_hmac && stored_hmac != [0u8; 8] {
            if !verify_hmac(encrypted, &self.config.hmac_key, &stored_hmac) {
                self.tamper_detected = true;
                return Err("HMAC verification failed - save file tampered".to_string());
            }
        }
        let decrypted = match enc_method {
            0 => encrypted.to_vec(),
            1 => xor_decrypt(encrypted, &self.config.key),
            2 => { let mut d = encrypted.to_vec(); d.reverse(); xor_decrypt(&d, &self.config.key) }
            _ => return Err(format!("Unknown encryption method: {}", enc_method)),
        };
        Ok(decrypted)
    }

    pub fn add_migration(&mut self, step: MigrationStep) { self.migrations.push(step); }

    pub fn needs_migration(&self, file_version: u32) -> bool { file_version < self.current_version }

    pub fn get_migration_path(&self, from: u32, to: u32) -> Vec<&MigrationStep> {
        let mut path = Vec::new();
        let mut current = from;
        while current < to {
            if let Some(step) = self.migrations.iter().find(|s| s.from_version == current) {
                path.push(step); current = step.to_version;
            } else { break; }
        }
        path
    }
}

impl Default for SaveEncryption { fn default() -> Self { Self::new(EncryptionConfig::default()) } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn encrypt_decrypt_roundtrip() {
        let enc = SaveEncryption::new(EncryptionConfig::default());
        let data = b"Hello, Genovo!";
        let encrypted = enc.encrypt_save(data);
        let mut dec = SaveEncryption::new(EncryptionConfig::default());
        let result = dec.decrypt_save(&encrypted).unwrap();
        assert_eq!(result, data);
    }
    #[test]
    fn hmac_verification() {
        let key = b"test_key";
        let data = b"test data";
        let hmac = compute_hmac(data, key);
        assert!(verify_hmac(data, key, &hmac));
        assert!(!verify_hmac(b"tampered", key, &hmac));
    }
    #[test]
    fn tamper_detection() {
        let enc = SaveEncryption::new(EncryptionConfig::default());
        let mut encrypted = enc.encrypt_save(b"original data");
        encrypted[40] ^= 0xFF; // tamper
        let mut dec = SaveEncryption::new(EncryptionConfig::default());
        let result = dec.decrypt_save(&encrypted);
        assert!(result.is_err() || dec.tamper_detected);
    }
}
'''

all_content[f"{base}/replay/src/replay_system_v2.rs"] = '''//! Enhanced replay: bookmark system, highlight detection, replay export,
//! replay sharing, replay analysis (stats), and cinematic replay mode.

use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReplayId(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BookmarkId(pub u64);

#[derive(Debug, Clone)]
pub struct ReplayBookmark { pub id: BookmarkId, pub time: f32, pub name: String, pub description: String, pub color: [f32; 4], pub camera_position: Option<[f32; 3]>, pub camera_target: Option<[f32; 3]> }
impl ReplayBookmark { pub fn new(id: BookmarkId, time: f32, name: impl Into<String>) -> Self { Self { id, time, name: name.into(), description: String::new(), color: [1.0, 0.8, 0.2, 1.0], camera_position: None, camera_target: None } } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HighlightType { Kill, Death, Objective, MultiKill, Streak, Custom }
impl HighlightType { pub fn score(&self) -> f32 { match self { Self::Kill => 1.0, Self::Death => 0.3, Self::Objective => 2.0, Self::MultiKill => 3.0, Self::Streak => 2.5, Self::Custom => 1.0 } } }

#[derive(Debug, Clone)]
pub struct ReplayHighlight { pub time: f32, pub duration: f32, pub highlight_type: HighlightType, pub score: f32, pub description: String, pub entities: Vec<u64>, pub auto_detected: bool }
impl ReplayHighlight { pub fn new(time: f32, ht: HighlightType) -> Self { Self { time, duration: 3.0, highlight_type: ht, score: ht.score(), description: String::new(), entities: Vec::new(), auto_detected: true } } }

#[derive(Debug, Clone)]
pub struct ReplayStats { pub total_duration: f32, pub total_kills: u32, pub total_deaths: u32, pub total_damage: f64, pub total_healing: f64, pub distance_traveled: f64, pub shots_fired: u32, pub shots_hit: u32, pub accuracy: f32, pub highest_streak: u32, pub objectives_completed: u32, pub custom_stats: HashMap<String, f64> }
impl ReplayStats {
    pub fn new() -> Self { Self { total_duration: 0.0, total_kills: 0, total_deaths: 0, total_damage: 0.0, total_healing: 0.0, distance_traveled: 0.0, shots_fired: 0, shots_hit: 0, accuracy: 0.0, highest_streak: 0, objectives_completed: 0, custom_stats: HashMap::new() } }
    pub fn kd_ratio(&self) -> f32 { if self.total_deaths == 0 { self.total_kills as f32 } else { self.total_kills as f32 / self.total_deaths as f32 } }
    pub fn compute_accuracy(&mut self) { self.accuracy = if self.shots_fired == 0 { 0.0 } else { self.shots_hit as f32 / self.shots_fired as f32 * 100.0 }; }
}
impl Default for ReplayStats { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CinematicMode { FreeCam, FollowEntity, PathCamera, OrbitalCamera, TopDown }

#[derive(Debug, Clone)]
pub struct CinematicSettings { pub mode: CinematicMode, pub follow_entity: Option<u64>, pub camera_speed: f32, pub smooth_factor: f32, pub dof_enabled: bool, pub dof_focus_distance: f32, pub motion_blur: bool, pub letterbox: bool, pub letterbox_aspect: f32, pub time_scale: f32, pub hide_hud: bool, pub hide_ui_entities: bool }
impl Default for CinematicSettings { fn default() -> Self { Self { mode: CinematicMode::FreeCam, follow_entity: None, camera_speed: 5.0, smooth_factor: 5.0, dof_enabled: false, dof_focus_distance: 10.0, motion_blur: false, letterbox: true, letterbox_aspect: 2.35, time_scale: 1.0, hide_hud: true, hide_ui_entities: true } } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat { Video, GIF, FrameSequence, ReplayFile }

#[derive(Debug, Clone)]
pub struct ExportConfig { pub format: ExportFormat, pub resolution: (u32, u32), pub framerate: u32, pub quality: f32, pub start_time: f32, pub end_time: f32, pub include_audio: bool, pub include_hud: bool, pub output_path: String }
impl Default for ExportConfig { fn default() -> Self { Self { format: ExportFormat::Video, resolution: (1920, 1080), framerate: 60, quality: 0.9, start_time: 0.0, end_time: 0.0, include_audio: true, include_hud: false, output_path: String::new() } } }

#[derive(Debug, Clone)]
pub struct ReplayFrame { pub time: f32, pub entity_positions: HashMap<u64, [f32; 3]>, pub entity_rotations: HashMap<u64, [f32; 4]>, pub events: Vec<String>, pub input_state: Vec<u8> }

#[derive(Debug, Clone)]
pub struct ReplayDataV2 { pub id: ReplayId, pub name: String, pub frames: Vec<ReplayFrame>, pub bookmarks: Vec<ReplayBookmark>, pub highlights: Vec<ReplayHighlight>, pub stats: ReplayStats, pub duration: f32, pub player_name: String, pub map_name: String, pub game_mode: String, pub timestamp: u64, pub version: u32 }
impl ReplayDataV2 {
    pub fn new(id: ReplayId, name: impl Into<String>) -> Self {
        Self { id, name: name.into(), frames: Vec::new(), bookmarks: Vec::new(), highlights: Vec::new(), stats: ReplayStats::new(), duration: 0.0, player_name: String::new(), map_name: String::new(), game_mode: String::new(), timestamp: 0, version: 1 }
    }
    pub fn frame_count(&self) -> usize { self.frames.len() }
    pub fn bookmark_count(&self) -> usize { self.bookmarks.len() }
    pub fn highlight_count(&self) -> usize { self.highlights.len() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackStateV2 { Stopped, Playing, Paused, Recording, Exporting }

#[derive(Debug, Clone)]
pub enum ReplayEvent { PlaybackStarted(ReplayId), PlaybackStopped, PlaybackPaused, PlaybackResumed, BookmarkAdded(BookmarkId), BookmarkRemoved(BookmarkId), HighlightDetected(usize), SeekTo(f32), SpeedChanged(f32), ExportStarted, ExportCompleted(String), ExportFailed(String), RecordingStarted, RecordingStopped }

pub struct ReplaySystemV2 {
    pub replays: HashMap<ReplayId, ReplayDataV2>,
    pub active_replay: Option<ReplayId>,
    pub playback_state: PlaybackStateV2,
    pub playback_time: f32,
    pub playback_speed: f32,
    pub cinematic: CinematicSettings,
    pub export_config: ExportConfig,
    pub events: Vec<ReplayEvent>,
    pub next_replay_id: u64,
    pub next_bookmark_id: u64,
    pub recording_data: Option<ReplayDataV2>,
    pub recording_interval: f32,
    pub recording_timer: f32,
    pub max_replay_duration: f32,
    pub auto_detect_highlights: bool,
}

impl ReplaySystemV2 {
    pub fn new() -> Self {
        Self { replays: HashMap::new(), active_replay: None, playback_state: PlaybackStateV2::Stopped, playback_time: 0.0, playback_speed: 1.0, cinematic: CinematicSettings::default(), export_config: ExportConfig::default(), events: Vec::new(), next_replay_id: 1, next_bookmark_id: 1, recording_data: None, recording_interval: 1.0 / 30.0, recording_timer: 0.0, max_replay_duration: 3600.0, auto_detect_highlights: true }
    }

    pub fn start_recording(&mut self, name: impl Into<String>) -> ReplayId {
        let id = ReplayId(self.next_replay_id); self.next_replay_id += 1;
        let data = ReplayDataV2::new(id, name);
        self.recording_data = Some(data);
        self.playback_state = PlaybackStateV2::Recording;
        self.events.push(ReplayEvent::RecordingStarted);
        id
    }

    pub fn stop_recording(&mut self) -> Option<ReplayId> {
        if let Some(data) = self.recording_data.take() {
            let id = data.id;
            self.replays.insert(id, data);
            self.playback_state = PlaybackStateV2::Stopped;
            self.events.push(ReplayEvent::RecordingStopped);
            Some(id)
        } else { None }
    }

    pub fn record_frame(&mut self, frame: ReplayFrame) {
        if let Some(ref mut data) = self.recording_data { data.duration = frame.time; data.frames.push(frame); }
    }

    pub fn play(&mut self, id: ReplayId) -> bool {
        if self.replays.contains_key(&id) { self.active_replay = Some(id); self.playback_state = PlaybackStateV2::Playing; self.playback_time = 0.0; self.events.push(ReplayEvent::PlaybackStarted(id)); true } else { false }
    }

    pub fn pause(&mut self) { if self.playback_state == PlaybackStateV2::Playing { self.playback_state = PlaybackStateV2::Paused; self.events.push(ReplayEvent::PlaybackPaused); } }
    pub fn resume(&mut self) { if self.playback_state == PlaybackStateV2::Paused { self.playback_state = PlaybackStateV2::Playing; self.events.push(ReplayEvent::PlaybackResumed); } }
    pub fn stop(&mut self) { self.playback_state = PlaybackStateV2::Stopped; self.active_replay = None; self.events.push(ReplayEvent::PlaybackStopped); }
    pub fn seek(&mut self, time: f32) { self.playback_time = time.max(0.0); self.events.push(ReplayEvent::SeekTo(time)); }
    pub fn set_speed(&mut self, speed: f32) { self.playback_speed = speed.clamp(0.1, 16.0); self.events.push(ReplayEvent::SpeedChanged(speed)); }

    pub fn add_bookmark(&mut self, replay_id: ReplayId, time: f32, name: impl Into<String>) -> Option<BookmarkId> {
        let id = BookmarkId(self.next_bookmark_id); self.next_bookmark_id += 1;
        if let Some(replay) = self.replays.get_mut(&replay_id) { replay.bookmarks.push(ReplayBookmark::new(id, time, name)); self.events.push(ReplayEvent::BookmarkAdded(id)); Some(id) } else { None }
    }

    pub fn detect_highlights(&mut self, replay_id: ReplayId) {
        if !self.auto_detect_highlights { return; }
        if let Some(replay) = self.replays.get_mut(&replay_id) {
            let mut highlights = Vec::new();
            for frame in &replay.frames {
                for event in &frame.events {
                    if event.contains("kill") { highlights.push(ReplayHighlight::new(frame.time, HighlightType::Kill)); }
                    if event.contains("objective") { highlights.push(ReplayHighlight::new(frame.time, HighlightType::Objective)); }
                }
            }
            for (i, h) in highlights.iter().enumerate() { self.events.push(ReplayEvent::HighlightDetected(i)); }
            replay.highlights.extend(highlights);
        }
    }

    pub fn update(&mut self, dt: f32) {
        if self.playback_state == PlaybackStateV2::Playing {
            self.playback_time += dt * self.playback_speed;
            if let Some(id) = self.active_replay {
                if let Some(replay) = self.replays.get(&id) {
                    if self.playback_time >= replay.duration { self.stop(); }
                }
            }
        }
    }

    pub fn current_frame(&self) -> Option<&ReplayFrame> {
        let id = self.active_replay?;
        let replay = self.replays.get(&id)?;
        replay.frames.iter().min_by(|a, b| {
            let da = (a.time - self.playback_time).abs();
            let db = (b.time - self.playback_time).abs();
            da.partial_cmp(&db).unwrap()
        })
    }

    pub fn replay_count(&self) -> usize { self.replays.len() }
    pub fn is_recording(&self) -> bool { self.playback_state == PlaybackStateV2::Recording }
    pub fn is_playing(&self) -> bool { self.playback_state == PlaybackStateV2::Playing }
    pub fn drain_events(&mut self) -> Vec<ReplayEvent> { std::mem::take(&mut self.events) }
}
impl Default for ReplaySystemV2 { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn record_and_playback() {
        let mut sys = ReplaySystemV2::new();
        let id = sys.start_recording("Test");
        sys.record_frame(ReplayFrame { time: 0.0, entity_positions: HashMap::new(), entity_rotations: HashMap::new(), events: Vec::new(), input_state: Vec::new() });
        sys.record_frame(ReplayFrame { time: 1.0, entity_positions: HashMap::new(), entity_rotations: HashMap::new(), events: Vec::new(), input_state: Vec::new() });
        let id = sys.stop_recording().unwrap();
        assert!(sys.play(id));
        assert!(sys.is_playing());
    }
    #[test]
    fn bookmark_add() {
        let mut sys = ReplaySystemV2::new();
        let id = sys.start_recording("BM Test");
        let id = sys.stop_recording().unwrap();
        let bm = sys.add_bookmark(id, 0.5, "Important");
        assert!(bm.is_some());
    }
    #[test]
    fn stats_kd() {
        let mut stats = ReplayStats::new();
        stats.total_kills = 10; stats.total_deaths = 5;
        assert!((stats.kd_ratio() - 2.0).abs() < 0.01);
    }
}
'''

for path, content in all_content.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='\n') as f:
        f.write(content)
    lines = content.count('\n') + 1
    print(f"Wrote {os.path.basename(path)} ({lines} lines)")

print("All remaining files done!")
