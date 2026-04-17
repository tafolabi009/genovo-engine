//! World generation: continent shapes (Perlin + Voronoi), climate zones,
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
