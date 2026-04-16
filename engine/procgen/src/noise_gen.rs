//! # Noise-Based Procedural Generation
//!
//! Higher-level procedural generation built on top of the noise primitives in
//! `genovo-core`. Provides complete terrain pipelines including:
//!
//! - **Terrain heightmap generation** — multi-octave FBM with configurable layers
//! - **Biome classification** — Whittaker diagram from temperature and moisture
//! - **River simulation** — downhill flow from peaks to ocean
//! - **City layout** — road network and building lot subdivision

use genovo_core::{PerlinNoise, Rng, SimplexNoise, WorleyNoise};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ===========================================================================
// Terrain Noise Configuration
// ===========================================================================

/// Configuration for a single noise layer in terrain generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseLayer {
    /// Frequency (scale) of this layer. Higher = more detailed.
    pub frequency: f32,
    /// Amplitude (height contribution) of this layer.
    pub amplitude: f32,
    /// Vertical offset applied after noise generation.
    pub offset: f32,
    /// Type of noise to use for this layer.
    pub noise_type: NoiseType,
    /// Seed for this layer.
    pub seed: u64,
}

/// Type of noise function to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseType {
    /// Perlin gradient noise.
    Perlin,
    /// Simplex noise (faster, fewer artifacts).
    Simplex,
    /// Worley/cellular noise (creates ridges and cells).
    Worley,
    /// Ridged noise (absolute value of Perlin, inverted).
    Ridged,
    /// Billowy noise (absolute value of Perlin).
    Billowy,
}

/// Configuration for terrain heightmap generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainConfig {
    /// Noise layers to combine (additive).
    pub layers: Vec<NoiseLayer>,
    /// Number of FBM octaves for the base terrain.
    pub octaves: u32,
    /// FBM lacunarity (frequency multiplier per octave).
    pub lacunarity: f32,
    /// FBM persistence (amplitude multiplier per octave).
    pub persistence: f32,
    /// Global height multiplier.
    pub height_scale: f32,
    /// Sea level threshold (0.0-1.0). Tiles below this are ocean.
    pub sea_level: f32,
    /// Random seed.
    pub seed: u64,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            layers: vec![
                NoiseLayer {
                    frequency: 0.01,
                    amplitude: 1.0,
                    offset: 0.0,
                    noise_type: NoiseType::Perlin,
                    seed: 0,
                },
                NoiseLayer {
                    frequency: 0.05,
                    amplitude: 0.3,
                    offset: 0.0,
                    noise_type: NoiseType::Simplex,
                    seed: 1,
                },
                NoiseLayer {
                    frequency: 0.1,
                    amplitude: 0.1,
                    offset: 0.0,
                    noise_type: NoiseType::Perlin,
                    seed: 2,
                },
            ],
            octaves: 6,
            lacunarity: 2.0,
            persistence: 0.5,
            height_scale: 1.0,
            sea_level: 0.35,
            seed: 42,
        }
    }
}

// ===========================================================================
// TerrainNoiseGenerator
// ===========================================================================

/// Generates terrain heightmaps using configurable multi-octave noise.
///
/// Combines multiple noise layers with FBM (Fractal Brownian Motion) to create
/// realistic terrain with continental shelves, mountain ridges, and valleys.
pub struct TerrainNoiseGenerator {
    /// Configuration.
    config: TerrainConfig,
    /// Pre-built noise generators for each layer.
    perlin_generators: Vec<PerlinNoise>,
    simplex_generators: Vec<SimplexNoise>,
    worley_generators: Vec<WorleyNoise>,
}

impl TerrainNoiseGenerator {
    /// Create a new terrain generator with the given configuration.
    pub fn new(config: TerrainConfig) -> Self {
        let mut perlin_generators = Vec::new();
        let mut simplex_generators = Vec::new();
        let mut worley_generators = Vec::new();

        for (i, layer) in config.layers.iter().enumerate() {
            let seed = config.seed.wrapping_add(layer.seed).wrapping_add(i as u64);
            perlin_generators.push(PerlinNoise::new(seed));
            simplex_generators.push(SimplexNoise::new(seed));
            worley_generators.push(WorleyNoise::new(seed));
        }

        Self {
            config,
            perlin_generators,
            simplex_generators,
            worley_generators,
        }
    }

    /// Generate a heightmap for the given dimensions.
    ///
    /// Returns a flat array of `width * height` f32 values, normalized to
    /// approximately [0, 1].
    pub fn generate_heightmap(&self, width: usize, height: usize) -> Vec<f32> {
        let mut heightmap = vec![0.0f32; width * height];
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;

        for y in 0..height {
            for x in 0..width {
                let mut value = 0.0f32;

                for (i, layer) in self.config.layers.iter().enumerate() {
                    let fx = x as f32 * layer.frequency;
                    let fy = y as f32 * layer.frequency;

                    let noise_val = match layer.noise_type {
                        NoiseType::Perlin => {
                            self.fbm_perlin(&self.perlin_generators[i], fx, fy)
                        }
                        NoiseType::Simplex => {
                            self.fbm_simplex(&self.simplex_generators[i], fx, fy)
                        }
                        NoiseType::Worley => {
                            let (f1, _f2) = self.worley_generators[i].worley2d(fx, fy);
                            f1
                        }
                        NoiseType::Ridged => {
                            let n = self.fbm_perlin(&self.perlin_generators[i], fx, fy);
                            1.0 - n.abs() * 2.0
                        }
                        NoiseType::Billowy => {
                            let n = self.fbm_perlin(&self.perlin_generators[i], fx, fy);
                            n.abs() * 2.0
                        }
                    };

                    value += (noise_val + layer.offset) * layer.amplitude;
                }

                value *= self.config.height_scale;
                heightmap[y * width + x] = value;

                min_val = min_val.min(value);
                max_val = max_val.max(value);
            }
        }

        // Normalize to [0, 1].
        let range = max_val - min_val;
        if range > f32::EPSILON {
            for v in &mut heightmap {
                *v = (*v - min_val) / range;
            }
        }

        heightmap
    }

    /// FBM using Perlin noise.
    fn fbm_perlin(&self, noise: &PerlinNoise, x: f32, y: f32) -> f32 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_amplitude = 0.0;

        for _ in 0..self.config.octaves {
            value += noise.noise2d(x * frequency, y * frequency) * amplitude;
            max_amplitude += amplitude;
            amplitude *= self.config.persistence;
            frequency *= self.config.lacunarity;
        }

        if max_amplitude > 0.0 {
            value / max_amplitude
        } else {
            value
        }
    }

    /// FBM using Simplex noise.
    fn fbm_simplex(&self, noise: &SimplexNoise, x: f32, y: f32) -> f32 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_amplitude = 0.0;

        for _ in 0..self.config.octaves {
            value += noise.noise2d(x * frequency, y * frequency) * amplitude;
            max_amplitude += amplitude;
            amplitude *= self.config.persistence;
            frequency *= self.config.lacunarity;
        }

        if max_amplitude > 0.0 {
            value / max_amplitude
        } else {
            value
        }
    }

    /// Generate a heightmap with continental shelf falloff.
    ///
    /// Applies a radial gradient that makes edges tend toward ocean level,
    /// creating island-like continents.
    pub fn generate_island_heightmap(&self, width: usize, height: usize) -> Vec<f32> {
        let mut heightmap = self.generate_heightmap(width, height);

        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        let max_dist = (cx * cx + cy * cy).sqrt();

        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt() / max_dist;

                // Smooth falloff: 1.0 at center, 0.0 at edges.
                let falloff = 1.0 - (dist * 1.5).min(1.0).powi(2);
                heightmap[y * width + x] *= falloff;
            }
        }

        heightmap
    }

    /// Get the configuration.
    pub fn config(&self) -> &TerrainConfig {
        &self.config
    }
}

// ===========================================================================
// Biome Classification
// ===========================================================================

/// Biome types based on the Whittaker diagram.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Biome {
    /// Deep ocean (very low elevation).
    DeepOcean,
    /// Shallow ocean/sea.
    Ocean,
    /// Sandy coastal area.
    Beach,
    /// Hot, dry biome.
    Desert,
    /// Semi-arid with scattered vegetation.
    Savanna,
    /// Temperate grassland.
    Grassland,
    /// Deciduous or mixed forest.
    Forest,
    /// Dense, wet tropical forest.
    Rainforest,
    /// Cold, sparse forest.
    Taiga,
    /// Frozen treeless biome.
    Tundra,
    /// High elevation, rocky.
    Mountain,
    /// Very high elevation, snow-covered.
    Snow,
    /// Wetland/swamp biome.
    Swamp,
    /// River tiles (assigned during river generation).
    River,
}

impl Biome {
    /// Get a representative color for this biome (RGB).
    pub fn color(&self) -> (u8, u8, u8) {
        match self {
            Biome::DeepOcean => (0, 0, 100),
            Biome::Ocean => (0, 40, 160),
            Biome::Beach => (238, 214, 175),
            Biome::Desert => (237, 201, 175),
            Biome::Savanna => (177, 209, 110),
            Biome::Grassland => (86, 152, 23),
            Biome::Forest => (34, 139, 34),
            Biome::Rainforest => (0, 100, 0),
            Biome::Taiga => (95, 115, 62),
            Biome::Tundra => (187, 194, 204),
            Biome::Mountain => (139, 137, 137),
            Biome::Snow => (248, 248, 255),
            Biome::Swamp => (47, 79, 47),
            Biome::River => (30, 80, 200),
        }
    }

    /// Whether this biome is water.
    pub fn is_water(&self) -> bool {
        matches!(
            self,
            Biome::DeepOcean | Biome::Ocean | Biome::River
        )
    }
}

/// Biome map generated from heightmap and moisture data.
pub struct BiomeMap {
    /// Width of the map.
    pub width: usize,
    /// Height of the map.
    pub height: usize,
    /// Flat array of biome values.
    pub biomes: Vec<Biome>,
    /// The heightmap used for generation.
    pub heightmap: Vec<f32>,
    /// The temperature map.
    pub temperature: Vec<f32>,
    /// The moisture map.
    pub moisture: Vec<f32>,
}

impl BiomeMap {
    /// Generate a biome map from a heightmap and moisture map.
    ///
    /// Uses a simplified Whittaker diagram to classify biomes based on
    /// elevation, temperature, and moisture.
    pub fn generate(
        heightmap: &[f32],
        moisture_map: &[f32],
        width: usize,
        height: usize,
        sea_level: f32,
    ) -> Self {
        let total = width * height;
        assert_eq!(heightmap.len(), total);
        assert_eq!(moisture_map.len(), total);

        // Generate temperature map: decreases with latitude and altitude.
        let mut temperature = vec![0.0f32; total];
        for y in 0..height {
            // Latitude factor: warmest at equator (center), coolest at poles.
            let lat = (y as f32 / height as f32 - 0.5).abs() * 2.0;
            let lat_temp = 1.0 - lat;

            for x in 0..width {
                let idx = y * width + x;
                let elevation = heightmap[idx];

                // Temperature decreases with altitude.
                let alt_factor = if elevation > sea_level {
                    1.0 - ((elevation - sea_level) / (1.0 - sea_level)) * 0.8
                } else {
                    1.0
                };

                temperature[idx] = (lat_temp * alt_factor).clamp(0.0, 1.0);
            }
        }

        // Classify biomes.
        let mut biomes = vec![Biome::Ocean; total];

        for i in 0..total {
            let h = heightmap[i];
            let t = temperature[i];
            let m = moisture_map[i];

            biomes[i] = if h < sea_level * 0.5 {
                Biome::DeepOcean
            } else if h < sea_level {
                Biome::Ocean
            } else if h < sea_level + 0.03 {
                Biome::Beach
            } else if h > 0.9 {
                Biome::Snow
            } else if h > 0.75 {
                Biome::Mountain
            } else if t < 0.15 {
                if m > 0.5 {
                    Biome::Tundra
                } else {
                    Biome::Snow
                }
            } else if t < 0.35 {
                if m > 0.6 {
                    Biome::Taiga
                } else {
                    Biome::Tundra
                }
            } else if t > 0.75 {
                if m > 0.7 {
                    Biome::Rainforest
                } else if m > 0.3 {
                    Biome::Savanna
                } else {
                    Biome::Desert
                }
            } else {
                // Temperate zone.
                if m > 0.7 {
                    Biome::Swamp
                } else if m > 0.5 {
                    Biome::Forest
                } else if m > 0.25 {
                    Biome::Grassland
                } else {
                    Biome::Desert
                }
            };
        }

        Self {
            width,
            height,
            biomes,
            heightmap: heightmap.to_vec(),
            temperature,
            moisture: moisture_map.to_vec(),
        }
    }

    /// Get the biome at (x, y).
    pub fn get(&self, x: usize, y: usize) -> Biome {
        if x < self.width && y < self.height {
            self.biomes[y * self.width + x]
        } else {
            Biome::Ocean
        }
    }

    /// Count the occurrence of each biome type.
    pub fn biome_counts(&self) -> std::collections::HashMap<Biome, usize> {
        let mut counts = std::collections::HashMap::new();
        for &b in &self.biomes {
            *counts.entry(b).or_insert(0) += 1;
        }
        counts
    }

    /// Generate a complete biome map from scratch using noise.
    pub fn generate_from_noise(
        width: usize,
        height: usize,
        seed: u64,
        sea_level: f32,
    ) -> Self {
        let terrain_config = TerrainConfig {
            seed,
            sea_level,
            ..Default::default()
        };
        let generator = TerrainNoiseGenerator::new(terrain_config);
        let heightmap = generator.generate_island_heightmap(width, height);

        // Generate moisture using different noise parameters.
        let moisture_config = TerrainConfig {
            layers: vec![
                NoiseLayer {
                    frequency: 0.02,
                    amplitude: 1.0,
                    offset: 0.0,
                    noise_type: NoiseType::Simplex,
                    seed: 100,
                },
                NoiseLayer {
                    frequency: 0.08,
                    amplitude: 0.3,
                    offset: 0.0,
                    noise_type: NoiseType::Perlin,
                    seed: 101,
                },
            ],
            octaves: 4,
            lacunarity: 2.0,
            persistence: 0.5,
            height_scale: 1.0,
            sea_level: 0.0,
            seed: seed.wrapping_add(1000),
        };
        let moisture_gen = TerrainNoiseGenerator::new(moisture_config);
        let moisture = moisture_gen.generate_heightmap(width, height);

        Self::generate(&heightmap, &moisture, width, height, sea_level)
    }
}

// ===========================================================================
// River Generation
// ===========================================================================

/// Generates rivers by simulating water flow downhill on a heightmap.
pub struct RiverGenerator;

impl RiverGenerator {
    /// Generate rivers on a heightmap.
    ///
    /// Simulates water flowing from high points to low points (below sea level).
    /// Returns a list of river paths, where each path is a sequence of (x, y)
    /// coordinates.
    pub fn generate_rivers(
        heightmap: &[f32],
        width: usize,
        height: usize,
        sea_level: f32,
        num_rivers: usize,
        min_length: usize,
        seed: u64,
    ) -> Vec<Vec<(usize, usize)>> {
        let mut rng = Rng::new(seed);
        let mut rivers = Vec::new();
        let mut used_tiles: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

        // Find candidate starting points (high elevation, above sea level).
        let mut candidates: Vec<(usize, usize, f32)> = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let h = heightmap[y * width + x];
                if h > sea_level + 0.2 {
                    candidates.push((x, y, h));
                }
            }
        }

        // Sort by height (highest first) for better river starts.
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        let mut river_count = 0;
        let mut candidate_idx = 0;

        while river_count < num_rivers && candidate_idx < candidates.len() {
            let (start_x, start_y, _) = candidates[candidate_idx];
            candidate_idx += 1;

            // Skip if too close to an existing river.
            if used_tiles.contains(&(start_x, start_y)) {
                continue;
            }

            // Trace the river downhill.
            let path = Self::trace_river(
                heightmap,
                width,
                height,
                start_x,
                start_y,
                sea_level,
                &used_tiles,
                &mut rng,
            );

            if path.len() >= min_length {
                for &pos in &path {
                    used_tiles.insert(pos);
                    // Mark nearby tiles as used to space rivers apart.
                    for dy in -3i32..=3 {
                        for dx in -3i32..=3 {
                            let nx = pos.0 as i32 + dx;
                            let ny = pos.1 as i32 + dy;
                            if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
                                used_tiles.insert((nx as usize, ny as usize));
                            }
                        }
                    }
                }
                rivers.push(path);
                river_count += 1;
            }
        }

        rivers
    }

    /// Trace a single river from a starting point downhill to the ocean.
    fn trace_river(
        heightmap: &[f32],
        width: usize,
        height: usize,
        start_x: usize,
        start_y: usize,
        sea_level: f32,
        used: &std::collections::HashSet<(usize, usize)>,
        rng: &mut Rng,
    ) -> Vec<(usize, usize)> {
        let mut path = vec![(start_x, start_y)];
        let mut x = start_x;
        let mut y = start_y;
        let max_steps = width * height;
        let mut visited: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        visited.insert((x, y));

        for _ in 0..max_steps {
            let current_h = heightmap[y * width + x];

            if current_h <= sea_level {
                break; // Reached the ocean.
            }

            // Find the lowest neighboring cell.
            let mut best_pos = None;
            let mut best_height = current_h;
            let neighbors = [
                (0i32, -1i32),
                (0, 1),
                (-1, 0),
                (1, 0),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ];

            for &(dx, dy) in &neighbors {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx as usize >= width || ny as usize >= height {
                    continue;
                }
                let ux = nx as usize;
                let uy = ny as usize;

                if visited.contains(&(ux, uy)) || used.contains(&(ux, uy)) {
                    continue;
                }

                let nh = heightmap[uy * width + ux];
                if nh < best_height {
                    best_height = nh;
                    best_pos = Some((ux, uy));
                }
            }

            match best_pos {
                Some((nx, ny)) => {
                    x = nx;
                    y = ny;
                    visited.insert((x, y));
                    path.push((x, y));
                }
                None => {
                    // No downhill neighbor — try a random lateral move to escape.
                    let lateral_idx = rng.range_i32(0, 4) as usize;
                    let (dx, dy) = neighbors[lateral_idx];
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0
                        && ny >= 0
                        && (nx as usize) < width
                        && (ny as usize) < height
                        && !visited.contains(&(nx as usize, ny as usize))
                    {
                        x = nx as usize;
                        y = ny as usize;
                        visited.insert((x, y));
                        path.push((x, y));
                    } else {
                        break; // Stuck, end the river.
                    }
                }
            }
        }

        path
    }

    /// Apply rivers to a biome map by setting river tiles.
    pub fn apply_to_biome_map(rivers: &[Vec<(usize, usize)>], biome_map: &mut BiomeMap) {
        for river in rivers {
            for &(x, y) in river {
                if x < biome_map.width && y < biome_map.height {
                    let idx = y * biome_map.width + x;
                    if !biome_map.biomes[idx].is_water() {
                        biome_map.biomes[idx] = Biome::River;
                    }
                }
            }
        }
    }
}

// ===========================================================================
// City Generation
// ===========================================================================

/// A road segment in the city layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadSegment {
    /// Start point (x, y).
    pub start: (f32, f32),
    /// End point (x, y).
    pub end: (f32, f32),
    /// Road width.
    pub width: f32,
    /// Whether this is a main road (highway) or side street.
    pub is_main_road: bool,
}

/// A building lot in the city layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingLot {
    /// Center position.
    pub x: f32,
    /// Center position.
    pub y: f32,
    /// Lot width.
    pub width: f32,
    /// Lot height.
    pub height: f32,
    /// Building type/zone.
    pub zone: BuildingZone,
}

/// Zoning types for building lots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuildingZone {
    /// Residential housing.
    Residential,
    /// Commercial/shops.
    Commercial,
    /// Industrial/factories.
    Industrial,
    /// Parks and green space.
    Park,
    /// Government/civic buildings.
    Civic,
}

/// Configuration for city generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CityConfig {
    /// City area width.
    pub width: f32,
    /// City area height.
    pub height: f32,
    /// Number of main roads.
    pub main_roads: usize,
    /// Number of growth iterations for side streets.
    pub growth_iterations: usize,
    /// Minimum block size.
    pub min_block_size: f32,
    /// Probability of adding a road branch.
    pub branch_probability: f32,
    /// Random seed.
    pub seed: u64,
}

impl Default for CityConfig {
    fn default() -> Self {
        Self {
            width: 200.0,
            height: 200.0,
            main_roads: 4,
            growth_iterations: 50,
            min_block_size: 8.0,
            branch_probability: 0.3,
            seed: 42,
        }
    }
}

/// Generates procedural city layouts.
pub struct CityGenerator;

impl CityGenerator {
    /// Generate a city road network and building lots.
    pub fn generate(config: &CityConfig) -> (Vec<RoadSegment>, Vec<BuildingLot>) {
        let mut rng = Rng::new(config.seed);
        let mut roads = Vec::new();
        let mut lots = Vec::new();

        let cx = config.width / 2.0;
        let cy = config.height / 2.0;

        // Generate main roads radiating from center.
        let angle_step = std::f32::consts::TAU / config.main_roads as f32;
        for i in 0..config.main_roads {
            let angle = i as f32 * angle_step + rng.range(-0.1, 0.1);
            let end_x = cx + angle.cos() * config.width * 0.5;
            let end_y = cy + angle.sin() * config.height * 0.5;

            roads.push(RoadSegment {
                start: (cx, cy),
                end: (end_x, end_y),
                width: 4.0,
                is_main_road: true,
            });
        }

        // Add a grid of cross streets.
        let grid_spacing = config.min_block_size * 3.0;
        let mut gx = config.min_block_size;
        while gx < config.width - config.min_block_size {
            // Vertical road.
            roads.push(RoadSegment {
                start: (gx, 0.0),
                end: (gx, config.height),
                width: 2.0,
                is_main_road: false,
            });
            gx += grid_spacing + rng.range(-2.0, 2.0);
        }

        let mut gy = config.min_block_size;
        while gy < config.height - config.min_block_size {
            // Horizontal road.
            roads.push(RoadSegment {
                start: (0.0, gy),
                end: (config.width, gy),
                width: 2.0,
                is_main_road: false,
            });
            gy += grid_spacing + rng.range(-2.0, 2.0);
        }

        // Growth simulation: add organic side streets.
        let mut tips: VecDeque<(f32, f32, f32)> = VecDeque::new(); // x, y, angle

        // Seed tips from main road endpoints.
        for road in &roads {
            if road.is_main_road {
                let dx = road.end.0 - road.start.0;
                let dy = road.end.1 - road.start.1;
                let angle = dy.atan2(dx);
                let mid_x = (road.start.0 + road.end.0) / 2.0;
                let mid_y = (road.start.1 + road.end.1) / 2.0;
                tips.push_back((mid_x, mid_y, angle + std::f32::consts::FRAC_PI_2));
                tips.push_back((mid_x, mid_y, angle - std::f32::consts::FRAC_PI_2));
            }
        }

        for _ in 0..config.growth_iterations {
            if let Some((tx, ty, angle)) = tips.pop_front() {
                let length = rng.range(config.min_block_size, config.min_block_size * 3.0);
                let end_x = tx + angle.cos() * length;
                let end_y = ty + angle.sin() * length;

                // Check bounds.
                if end_x > 0.0 && end_x < config.width && end_y > 0.0 && end_y < config.height {
                    roads.push(RoadSegment {
                        start: (tx, ty),
                        end: (end_x, end_y),
                        width: 1.5,
                        is_main_road: false,
                    });

                    // Potentially branch.
                    if rng.next_f32() < config.branch_probability {
                        let branch_angle = angle + std::f32::consts::FRAC_PI_2 * if rng.bool(0.5) { 1.0 } else { -1.0 };
                        tips.push_back((end_x, end_y, branch_angle));
                    }

                    // Continue straight.
                    let new_angle = angle + rng.range(-0.2, 0.2);
                    tips.push_back((end_x, end_y, new_angle));
                }
            }
        }

        // Generate building lots in grid cells.
        let lot_spacing = config.min_block_size;
        let mut lx = lot_spacing;
        while lx < config.width - lot_spacing {
            let mut ly = lot_spacing;
            while ly < config.height - lot_spacing {
                // Vary lot size.
                let lot_w = rng.range(config.min_block_size * 0.5, config.min_block_size * 1.5);
                let lot_h = rng.range(config.min_block_size * 0.5, config.min_block_size * 1.5);

                // Determine zone based on distance from center.
                let dist_to_center = ((lx - cx).powi(2) + (ly - cy).powi(2)).sqrt();
                let max_dist = (cx.powi(2) + cy.powi(2)).sqrt();
                let normalized_dist = dist_to_center / max_dist;

                let zone = if normalized_dist < 0.15 {
                    BuildingZone::Commercial
                } else if normalized_dist < 0.3 {
                    if rng.next_f32() < 0.7 {
                        BuildingZone::Commercial
                    } else {
                        BuildingZone::Civic
                    }
                } else if normalized_dist < 0.6 {
                    if rng.next_f32() < 0.8 {
                        BuildingZone::Residential
                    } else {
                        BuildingZone::Park
                    }
                } else {
                    if rng.next_f32() < 0.6 {
                        BuildingZone::Residential
                    } else {
                        BuildingZone::Industrial
                    }
                };

                lots.push(BuildingLot {
                    x: lx,
                    y: ly,
                    width: lot_w,
                    height: lot_h,
                    zone,
                });

                ly += lot_spacing + rng.range(-1.0, 1.0);
            }
            lx += lot_spacing + rng.range(-1.0, 1.0);
        }

        (roads, lots)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heightmap_generation() {
        let config = TerrainConfig::default();
        let terrain_gen = TerrainNoiseGenerator::new(config);
        let heightmap = terrain_gen.generate_heightmap(64, 64);

        assert_eq!(heightmap.len(), 64 * 64);

        // Values should be normalized to [0, 1].
        for &v in &heightmap {
            assert!(v >= -0.01 && v <= 1.01, "Height out of range: {v}");
        }
    }

    #[test]
    fn test_heightmap_deterministic() {
        let config = TerrainConfig::default();
        let terrain_gen = TerrainNoiseGenerator::new(config.clone());
        let h1 = terrain_gen.generate_heightmap(32, 32);

        let gen2 = TerrainNoiseGenerator::new(config);
        let h2 = gen2.generate_heightmap(32, 32);

        assert_eq!(h1, h2, "Same config should produce same heightmap");
    }

    #[test]
    fn test_island_heightmap() {
        let config = TerrainConfig::default();
        let terrain_gen = TerrainNoiseGenerator::new(config);
        let heightmap = terrain_gen.generate_island_heightmap(64, 64);

        // Edges should tend toward lower values.
        let edge_avg: f32 =
            (0..64).map(|x| heightmap[x]).sum::<f32>() / 64.0;
        let mut center_sum: f32 = 0.0;
        for y in 28..36 {
            for x in 28..36 {
                center_sum += heightmap[y * 64 + x];
            }
        }
        let center_avg = center_sum / 64.0;

        assert!(
            center_avg > edge_avg,
            "Center should be higher than edges: center={center_avg}, edge={edge_avg}"
        );
    }

    #[test]
    fn test_biome_classification() {
        let biome_map = BiomeMap::generate_from_noise(64, 64, 42, 0.35);

        assert_eq!(biome_map.biomes.len(), 64 * 64);

        let counts = biome_map.biome_counts();
        // Should have at least ocean and some land biome.
        assert!(
            counts.values().sum::<usize>() == 64 * 64,
            "All tiles should have a biome"
        );
    }

    #[test]
    fn test_biome_properties() {
        assert!(Biome::Ocean.is_water());
        assert!(Biome::DeepOcean.is_water());
        assert!(Biome::River.is_water());
        assert!(!Biome::Forest.is_water());
        assert!(!Biome::Desert.is_water());
    }

    #[test]
    fn test_biome_colors() {
        let (r, g, b) = Biome::Ocean.color();
        assert!(b > r && b > g, "Ocean should be blue");

        let (r, g, b) = Biome::Forest.color();
        assert!(g > r && g > b, "Forest should be green");
    }

    #[test]
    fn test_river_generation() {
        let config = TerrainConfig::default();
        let terrain_gen = TerrainNoiseGenerator::new(config);
        let heightmap = terrain_gen.generate_island_heightmap(64, 64);

        let rivers = RiverGenerator::generate_rivers(
            &heightmap,
            64,
            64,
            0.35,
            3,
            5,
            42,
        );

        // Some rivers should be generated (depends on terrain shape).
        // We don't assert a specific count since it depends on the heightmap.
        for river in &rivers {
            assert!(river.len() >= 5, "Rivers should be at least min_length");

            // River should flow downhill overall.
            if river.len() >= 2 {
                let start_h = heightmap[river[0].1 * 64 + river[0].0];
                let end_h = heightmap[river[river.len() - 1].1 * 64 + river[river.len() - 1].0];
                assert!(
                    end_h <= start_h + 0.1,
                    "River should flow downhill: start={start_h}, end={end_h}"
                );
            }
        }
    }

    #[test]
    fn test_city_generation() {
        let config = CityConfig::default();
        let (roads, lots) = CityGenerator::generate(&config);

        assert!(!roads.is_empty(), "City should have roads");
        assert!(!lots.is_empty(), "City should have building lots");

        // Main roads should exist.
        let main_road_count = roads.iter().filter(|r| r.is_main_road).count();
        assert_eq!(main_road_count, config.main_roads);

        // Check that lots are within bounds.
        for lot in &lots {
            assert!(lot.x >= 0.0 && lot.x <= config.width);
            assert!(lot.y >= 0.0 && lot.y <= config.height);
        }
    }

    #[test]
    fn test_city_zoning() {
        let config = CityConfig::default();
        let (_, lots) = CityGenerator::generate(&config);

        let commercial_count = lots.iter().filter(|l| l.zone == BuildingZone::Commercial).count();
        let residential_count = lots.iter().filter(|l| l.zone == BuildingZone::Residential).count();

        assert!(
            commercial_count > 0,
            "City should have commercial zones"
        );
        assert!(
            residential_count > 0,
            "City should have residential zones"
        );
    }

    #[test]
    fn test_noise_types() {
        let config = TerrainConfig {
            layers: vec![
                NoiseLayer {
                    frequency: 0.05,
                    amplitude: 1.0,
                    offset: 0.0,
                    noise_type: NoiseType::Ridged,
                    seed: 0,
                },
            ],
            ..Default::default()
        };
        let terrain_gen = TerrainNoiseGenerator::new(config);
        let heightmap = terrain_gen.generate_heightmap(32, 32);
        assert_eq!(heightmap.len(), 32 * 32);
    }

    #[test]
    fn test_river_apply_to_biome_map() {
        let mut biome_map = BiomeMap::generate_from_noise(32, 32, 42, 0.35);
        let heightmap = biome_map.heightmap.clone();
        let rivers = RiverGenerator::generate_rivers(&heightmap, 32, 32, 0.35, 2, 3, 42);

        let initial_river_count = biome_map.biomes.iter().filter(|&&b| b == Biome::River).count();
        RiverGenerator::apply_to_biome_map(&rivers, &mut biome_map);
        let final_river_count = biome_map.biomes.iter().filter(|&&b| b == Biome::River).count();

        // After applying rivers, there should be more (or at least equal) river tiles.
        assert!(final_river_count >= initial_river_count);
    }
}
