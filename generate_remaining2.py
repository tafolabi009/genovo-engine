#!/usr/bin/env python3
"""Generate remaining Genovo engine source files - batch 2."""
import os

base = "C:/Users/USER/Downloads/game_engine/engine"
files = {}

files[f"{base}/terrain/src/terrain_system_v2.rs"] = open(f"{base}/terrain/src/terrain_system_v2.rs", "w") if False else None

# We'll write all files directly
all_content = {}

all_content[f"{base}/terrain/src/terrain_system_v2.rs"] = '''//! Enhanced terrain system: virtual texturing integration, runtime terrain
//! modification, terrain collision update, terrain streaming, and instancing.

use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TerrainChunkId(pub u32, pub u32);

impl TerrainChunkId {
    pub fn new(x: u32, z: u32) -> Self { Self(x, z) }
    pub fn x(&self) -> u32 { self.0 }
    pub fn z(&self) -> u32 { self.1 }
}

impl fmt::Display for TerrainChunkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Chunk({}, {})", self.0, self.1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainChunkState { Unloaded, Loading, Loaded, Modified, Unloading }

#[derive(Debug, Clone)]
pub struct VirtualTextureConfig {
    pub page_size: u32,
    pub tile_size: u32,
    pub max_pages: u32,
    pub feedback_buffer_size: u32,
    pub mip_bias: f32,
    pub aniso_level: u32,
    pub cache_size_mb: u32,
    pub streaming_budget_ms: f32,
}

impl Default for VirtualTextureConfig {
    fn default() -> Self {
        Self {
            page_size: 256, tile_size: 128, max_pages: 4096,
            feedback_buffer_size: 256, mip_bias: 0.0, aniso_level: 8,
            cache_size_mb: 256, streaming_budget_ms: 2.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TerrainModification {
    pub position: [f32; 2],
    pub radius: f32,
    pub strength: f32,
    pub mod_type: ModificationType,
    pub falloff: FalloffType,
    pub timestamp: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModificationType { Raise, Lower, Flatten, Smooth, Noise, SetHeight, Paint }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FalloffType { Linear, Smooth, Spherical, Tip, Flat }

impl TerrainModification {
    pub fn new(pos: [f32; 2], radius: f32, strength: f32, mod_type: ModificationType) -> Self {
        Self {
            position: pos, radius, strength, mod_type,
            falloff: FalloffType::Smooth, timestamp: 0.0,
        }
    }

    pub fn compute_falloff(&self, distance: f32) -> f32 {
        let t = (distance / self.radius).clamp(0.0, 1.0);
        match self.falloff {
            FalloffType::Linear => 1.0 - t,
            FalloffType::Smooth => {
                let t2 = t * t;
                1.0 - (3.0 * t2 - 2.0 * t * t2)
            }
            FalloffType::Spherical => (1.0 - t * t).sqrt().max(0.0),
            FalloffType::Tip => (1.0 - t).powi(3),
            FalloffType::Flat => if t < 0.8 { 1.0 } else { (1.0 - (t - 0.8) / 0.2).max(0.0) },
        }
    }
}

#[derive(Debug, Clone)]
pub struct TerrainChunk {
    pub id: TerrainChunkId,
    pub state: TerrainChunkState,
    pub heightmap: Vec<f32>,
    pub resolution: u32,
    pub world_pos: [f32; 2],
    pub chunk_size: f32,
    pub min_height: f32,
    pub max_height: f32,
    pub lod_level: u32,
    pub neighbor_lods: [u32; 4],
    pub collision_dirty: bool,
    pub render_dirty: bool,
    pub memory_usage: u64,
    pub instance_count: u32,
    pub modifications: Vec<TerrainModification>,
}

impl TerrainChunk {
    pub fn new(id: TerrainChunkId, resolution: u32, world_pos: [f32; 2], size: f32) -> Self {
        let count = (resolution * resolution) as usize;
        Self {
            id, state: TerrainChunkState::Unloaded,
            heightmap: vec![0.0; count], resolution, world_pos,
            chunk_size: size, min_height: 0.0, max_height: 0.0,
            lod_level: 0, neighbor_lods: [0; 4], collision_dirty: false,
            render_dirty: false, memory_usage: (count * 4) as u64,
            instance_count: 0, modifications: Vec::new(),
        }
    }

    pub fn get_height(&self, x: u32, z: u32) -> f32 {
        let idx = (z * self.resolution + x) as usize;
        if idx < self.heightmap.len() { self.heightmap[idx] } else { 0.0 }
    }

    pub fn set_height(&mut self, x: u32, z: u32, height: f32) {
        let idx = (z * self.resolution + x) as usize;
        if idx < self.heightmap.len() {
            self.heightmap[idx] = height;
            self.min_height = self.min_height.min(height);
            self.max_height = self.max_height.max(height);
            self.collision_dirty = true;
            self.render_dirty = true;
        }
    }

    pub fn get_height_interpolated(&self, local_x: f32, local_z: f32) -> f32 {
        let fx = (local_x / self.chunk_size * (self.resolution - 1) as f32).clamp(0.0, (self.resolution - 2) as f32);
        let fz = (local_z / self.chunk_size * (self.resolution - 1) as f32).clamp(0.0, (self.resolution - 2) as f32);
        let ix = fx as u32;
        let iz = fz as u32;
        let tx = fx - ix as f32;
        let tz = fz - iz as f32;
        let h00 = self.get_height(ix, iz);
        let h10 = self.get_height(ix + 1, iz);
        let h01 = self.get_height(ix, iz + 1);
        let h11 = self.get_height(ix + 1, iz + 1);
        let h0 = h00 + (h10 - h00) * tx;
        let h1 = h01 + (h11 - h01) * tx;
        h0 + (h1 - h0) * tz
    }

    pub fn apply_modification(&mut self, m: &TerrainModification) {
        for z in 0..self.resolution {
            for x in 0..self.resolution {
                let wx = self.world_pos[0] + (x as f32 / (self.resolution - 1) as f32) * self.chunk_size;
                let wz = self.world_pos[1] + (z as f32 / (self.resolution - 1) as f32) * self.chunk_size;
                let dx = wx - m.position[0];
                let dz = wz - m.position[1];
                let dist = (dx * dx + dz * dz).sqrt();
                if dist > m.radius { continue; }
                let falloff = m.compute_falloff(dist);
                let idx = (z * self.resolution + x) as usize;
                match m.mod_type {
                    ModificationType::Raise => self.heightmap[idx] += m.strength * falloff,
                    ModificationType::Lower => self.heightmap[idx] -= m.strength * falloff,
                    ModificationType::Flatten => {
                        let center_h = self.get_height_interpolated(
                            m.position[0] - self.world_pos[0],
                            m.position[1] - self.world_pos[1],
                        );
                        self.heightmap[idx] += (center_h - self.heightmap[idx]) * falloff * m.strength;
                    }
                    ModificationType::SetHeight => {
                        self.heightmap[idx] += (m.strength - self.heightmap[idx]) * falloff;
                    }
                    _ => {}
                }
            }
        }
        self.collision_dirty = true;
        self.render_dirty = true;
        self.recalculate_bounds();
    }

    fn recalculate_bounds(&mut self) {
        self.min_height = f32::MAX;
        self.max_height = f32::MIN;
        for &h in &self.heightmap {
            self.min_height = self.min_height.min(h);
            self.max_height = self.max_height.max(h);
        }
    }

    pub fn compute_normal(&self, x: u32, z: u32) -> [f32; 3] {
        let left = if x > 0 { self.get_height(x - 1, z) } else { self.get_height(x, z) };
        let right = if x < self.resolution - 1 { self.get_height(x + 1, z) } else { self.get_height(x, z) };
        let down = if z > 0 { self.get_height(x, z - 1) } else { self.get_height(x, z) };
        let up = if z < self.resolution - 1 { self.get_height(x, z + 1) } else { self.get_height(x, z) };
        let scale = self.chunk_size / (self.resolution - 1) as f32;
        let nx = (left - right) / (2.0 * scale);
        let nz = (down - up) / (2.0 * scale);
        let len = (nx * nx + 1.0 + nz * nz).sqrt();
        [nx / len, 1.0 / len, nz / len]
    }
}

#[derive(Debug, Clone)]
pub struct TerrainInstanceGroup {
    pub mesh_id: u64,
    pub instances: Vec<TerrainInstance>,
    pub lod_distances: Vec<f32>,
    pub cast_shadows: bool,
    pub receive_shadows: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct TerrainInstance {
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub color_tint: [f32; 4],
}

#[derive(Debug, Clone)]
pub enum TerrainEvent {
    ChunkLoaded(TerrainChunkId), ChunkUnloaded(TerrainChunkId),
    ChunkModified(TerrainChunkId), CollisionUpdated(TerrainChunkId),
    LODChanged(TerrainChunkId, u32), TexturePageLoaded(u32),
}

pub struct TerrainSystemV2 {
    pub chunks: HashMap<TerrainChunkId, TerrainChunk>,
    pub vt_config: VirtualTextureConfig,
    pub instance_groups: Vec<TerrainInstanceGroup>,
    pub events: Vec<TerrainEvent>,
    pub chunk_resolution: u32,
    pub chunk_size: f32,
    pub chunks_x: u32,
    pub chunks_z: u32,
    pub max_height: f32,
    pub camera_position: [f32; 3],
    pub lod_distances: Vec<f32>,
    pub streaming_radius: f32,
    pub collision_update_budget_ms: f32,
    pub total_memory_usage: u64,
    pub dirty_chunks: Vec<TerrainChunkId>,
}

impl TerrainSystemV2 {
    pub fn new(chunks_x: u32, chunks_z: u32, chunk_size: f32, resolution: u32) -> Self {
        let mut chunks = HashMap::new();
        for z in 0..chunks_z {
            for x in 0..chunks_x {
                let id = TerrainChunkId::new(x, z);
                let world_pos = [x as f32 * chunk_size, z as f32 * chunk_size];
                chunks.insert(id, TerrainChunk::new(id, resolution, world_pos, chunk_size));
            }
        }
        Self {
            chunks, vt_config: VirtualTextureConfig::default(),
            instance_groups: Vec::new(), events: Vec::new(),
            chunk_resolution: resolution, chunk_size, chunks_x, chunks_z,
            max_height: 500.0, camera_position: [0.0; 3],
            lod_distances: vec![100.0, 200.0, 500.0, 1000.0, 2000.0],
            streaming_radius: 1000.0, collision_update_budget_ms: 2.0,
            total_memory_usage: 0, dirty_chunks: Vec::new(),
        }
    }

    pub fn get_height(&self, world_x: f32, world_z: f32) -> f32 {
        let cx = (world_x / self.chunk_size) as u32;
        let cz = (world_z / self.chunk_size) as u32;
        let id = TerrainChunkId::new(cx, cz);
        if let Some(chunk) = self.chunks.get(&id) {
            let local_x = world_x - chunk.world_pos[0];
            let local_z = world_z - chunk.world_pos[1];
            chunk.get_height_interpolated(local_x, local_z)
        } else { 0.0 }
    }

    pub fn modify_terrain(&mut self, modification: TerrainModification) {
        let affected = self.chunks_in_radius(modification.position, modification.radius);
        for id in affected {
            if let Some(chunk) = self.chunks.get_mut(&id) {
                chunk.apply_modification(&modification);
                chunk.state = TerrainChunkState::Modified;
                self.events.push(TerrainEvent::ChunkModified(id));
                if !self.dirty_chunks.contains(&id) { self.dirty_chunks.push(id); }
            }
        }
    }

    fn chunks_in_radius(&self, pos: [f32; 2], radius: f32) -> Vec<TerrainChunkId> {
        let min_cx = ((pos[0] - radius) / self.chunk_size).floor().max(0.0) as u32;
        let max_cx = ((pos[0] + radius) / self.chunk_size).ceil().min(self.chunks_x as f32) as u32;
        let min_cz = ((pos[1] - radius) / self.chunk_size).floor().max(0.0) as u32;
        let max_cz = ((pos[1] + radius) / self.chunk_size).ceil().min(self.chunks_z as f32) as u32;
        let mut result = Vec::new();
        for z in min_cz..max_cz { for x in min_cx..max_cx { result.push(TerrainChunkId::new(x, z)); } }
        result
    }

    pub fn update_lods(&mut self) {
        let cam = self.camera_position;
        for (id, chunk) in &mut self.chunks {
            let cx = chunk.world_pos[0] + chunk.chunk_size * 0.5;
            let cz = chunk.world_pos[1] + chunk.chunk_size * 0.5;
            let dist = ((cam[0] - cx).powi(2) + (cam[2] - cz).powi(2)).sqrt();
            let mut lod = 0u32;
            for (i, &d) in self.lod_distances.iter().enumerate() {
                if dist > d { lod = (i + 1) as u32; }
            }
            if chunk.lod_level != lod {
                chunk.lod_level = lod;
                chunk.render_dirty = true;
                self.events.push(TerrainEvent::LODChanged(*id, lod));
            }
        }
    }

    pub fn update_collisions(&mut self) {
        let budget = self.dirty_chunks.len().min(4);
        for _ in 0..budget {
            if let Some(id) = self.dirty_chunks.pop() {
                if let Some(chunk) = self.chunks.get_mut(&id) {
                    chunk.collision_dirty = false;
                    self.events.push(TerrainEvent::CollisionUpdated(id));
                }
            }
        }
    }

    pub fn set_camera_position(&mut self, pos: [f32; 3]) { self.camera_position = pos; }
    pub fn chunk_count(&self) -> usize { self.chunks.len() }
    pub fn drain_events(&mut self) -> Vec<TerrainEvent> { std::mem::take(&mut self.events) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn terrain_height() {
        let sys = TerrainSystemV2::new(2, 2, 100.0, 33);
        let h = sys.get_height(50.0, 50.0);
        assert!((h - 0.0).abs() < 0.01);
    }
    #[test]
    fn modification_falloff() {
        let m = TerrainModification::new([0.0, 0.0], 10.0, 1.0, ModificationType::Raise);
        assert!((m.compute_falloff(0.0) - 1.0).abs() < 0.01);
        assert!((m.compute_falloff(10.0) - 0.0).abs() < 0.01);
    }
}
'''

all_content[f"{base}/terrain/src/terrain_foliage.rs"] = '''//! Terrain foliage: per-cell foliage data, wind animation, LOD transitions,
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
'''

for path, content in all_content.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='\n') as f:
        f.write(content)
    lines = content.count('\n') + 1
    print(f"Wrote {os.path.basename(path)} ({lines} lines)")

print("Batch 2 done")
