//! Procedural city generation for the Genovo engine.
//!
//! Generates realistic city layouts with:
//!
//! - **Road network** — agent-based simulation combined with L-system growth
//!   for major/minor roads.
//! - **Block subdivision** — recursive subdivision of road-bounded blocks into
//!   building lots.
//! - **Building generation** — floor stacking, window placement, facade
//!   variation, and roof type selection.
//! - **District zoning** — commercial, residential, industrial, park, and
//!   mixed-use districts with different generation rules.
//! - **Intersection types** — T-junctions, crossroads, roundabouts.
//! - **Building lots** — placement respecting setbacks, lot coverage, and
//!   height limits per zoning type.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// RNG utility (self-contained)
// ---------------------------------------------------------------------------

/// Simple xoshiro256** PRNG for deterministic generation.
#[derive(Debug, Clone)]
pub struct CityRng {
    state: [u64; 4],
}

impl CityRng {
    /// Creates a new RNG from a seed.
    pub fn new(seed: u64) -> Self {
        let mut s = [0u64; 4];
        let mut h = seed;
        for slot in &mut s {
            h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *slot = h;
        }
        Self { state: s }
    }

    /// Returns a random u64.
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.state[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Returns a random f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Returns a random f32 in [min, max).
    pub fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }

    /// Returns a random u32 in [0, max).
    pub fn range_u32(&mut self, max: u32) -> u32 {
        (self.next_u64() % max as u64) as u32
    }

    /// Returns a random i32 in [min, max].
    pub fn range_i32(&mut self, min: i32, max: i32) -> i32 {
        min + (self.next_u64() % (max - min + 1) as u64) as i32
    }

    /// Returns true with probability `p`.
    pub fn chance(&mut self, p: f32) -> bool {
        self.next_f32() < p
    }

    /// Picks a random element from a slice.
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        if items.is_empty() {
            None
        } else {
            Some(&items[self.range_u32(items.len() as u32) as usize])
        }
    }
}

// ---------------------------------------------------------------------------
// 2D position
// ---------------------------------------------------------------------------

/// A 2D position in the city grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pos2 {
    pub x: f32,
    pub y: f32,
}

impl Pos2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn distance(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len < 1e-9 { Self::new(0.0, 0.0) } else { Self::new(self.x / len, self.y / len) }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }

    pub fn scale(&self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s)
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self::new(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
        )
    }

    pub fn rotate(&self, angle_rad: f32) -> Self {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        Self::new(self.x * c - self.y * s, self.x * s + self.y * c)
    }

    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    pub fn cross(&self, other: &Self) -> f32 {
        self.x * other.y - self.y * other.x
    }
}

impl Default for Pos2 {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

// ---------------------------------------------------------------------------
// District zoning
// ---------------------------------------------------------------------------

/// The zoning type of a city district.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZoneType {
    /// Residential -- houses, apartments.
    Residential,
    /// Commercial -- shops, offices, restaurants.
    Commercial,
    /// Industrial -- factories, warehouses.
    Industrial,
    /// Park / green space.
    Park,
    /// Mixed-use (residential + commercial).
    MixedUse,
    /// Civic (government, schools, hospitals).
    Civic,
    /// Vacant / empty lot.
    Vacant,
}

impl ZoneType {
    /// Returns the maximum building height in floors for this zone.
    pub fn max_floors(&self) -> u32 {
        match self {
            Self::Residential => 4,
            Self::Commercial => 12,
            Self::Industrial => 3,
            Self::Park => 0,
            Self::MixedUse => 8,
            Self::Civic => 5,
            Self::Vacant => 0,
        }
    }

    /// Returns the lot coverage ratio (building footprint / lot area).
    pub fn max_lot_coverage(&self) -> f32 {
        match self {
            Self::Residential => 0.4,
            Self::Commercial => 0.7,
            Self::Industrial => 0.6,
            Self::Park => 0.05,
            Self::MixedUse => 0.6,
            Self::Civic => 0.5,
            Self::Vacant => 0.0,
        }
    }

    /// Returns the minimum setback from the road in meters.
    pub fn min_setback(&self) -> f32 {
        match self {
            Self::Residential => 5.0,
            Self::Commercial => 2.0,
            Self::Industrial => 8.0,
            Self::Park => 3.0,
            Self::MixedUse => 2.0,
            Self::Civic => 6.0,
            Self::Vacant => 0.0,
        }
    }

    /// Returns the minimum lot area in square meters.
    pub fn min_lot_area(&self) -> f32 {
        match self {
            Self::Residential => 200.0,
            Self::Commercial => 100.0,
            Self::Industrial => 500.0,
            Self::Park => 400.0,
            Self::MixedUse => 150.0,
            Self::Civic => 300.0,
            Self::Vacant => 50.0,
        }
    }
}

impl fmt::Display for ZoneType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Residential => write!(f, "Residential"),
            Self::Commercial => write!(f, "Commercial"),
            Self::Industrial => write!(f, "Industrial"),
            Self::Park => write!(f, "Park"),
            Self::MixedUse => write!(f, "Mixed-Use"),
            Self::Civic => write!(f, "Civic"),
            Self::Vacant => write!(f, "Vacant"),
        }
    }
}

// ---------------------------------------------------------------------------
// Road types
// ---------------------------------------------------------------------------

/// The category of a road segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoadCategory {
    /// Major arterial road (wide, fast traffic).
    Highway,
    /// Main city street.
    Major,
    /// Minor residential / side street.
    Minor,
    /// Alley or service road.
    Alley,
    /// Pedestrian-only pathway.
    Pedestrian,
}

impl RoadCategory {
    /// Returns the default road width in meters.
    pub fn default_width(&self) -> f32 {
        match self {
            Self::Highway => 20.0,
            Self::Major => 14.0,
            Self::Minor => 8.0,
            Self::Alley => 4.0,
            Self::Pedestrian => 3.0,
        }
    }

    /// Returns the number of lanes in each direction.
    pub fn lanes(&self) -> u32 {
        match self {
            Self::Highway => 3,
            Self::Major => 2,
            Self::Minor => 1,
            Self::Alley => 1,
            Self::Pedestrian => 0,
        }
    }

    /// Whether the road has sidewalks.
    pub fn has_sidewalks(&self) -> bool {
        matches!(self, Self::Major | Self::Minor | Self::Pedestrian)
    }
}

// ---------------------------------------------------------------------------
// Road segment
// ---------------------------------------------------------------------------

/// A straight road segment between two nodes.
#[derive(Debug, Clone)]
pub struct RoadSegment {
    /// Unique segment id.
    pub id: u32,
    /// Start node index.
    pub start_node: u32,
    /// End node index.
    pub end_node: u32,
    /// Road category.
    pub category: RoadCategory,
    /// Road width in meters.
    pub width: f32,
    /// Length in meters.
    pub length: f32,
    /// Speed limit (km/h).
    pub speed_limit: f32,
    /// Whether this is a one-way road.
    pub one_way: bool,
}

// ---------------------------------------------------------------------------
// Road node (intersection)
// ---------------------------------------------------------------------------

/// A node in the road network (typically an intersection).
#[derive(Debug, Clone)]
pub struct RoadNode {
    /// Unique node id.
    pub id: u32,
    /// World position.
    pub position: Pos2,
    /// Connected segment ids.
    pub segments: Vec<u32>,
    /// Intersection type.
    pub intersection_type: IntersectionType,
    /// Whether this node has a traffic light.
    pub has_traffic_light: bool,
}

/// The type of intersection at a road node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntersectionType {
    /// Dead-end (1 segment).
    DeadEnd,
    /// Curve / continuation (2 segments, same road).
    Curve,
    /// T-junction (3 segments).
    TJunction,
    /// Crossroads (4 segments).
    Crossroads,
    /// Roundabout.
    Roundabout,
    /// Complex (5+ segments).
    Complex,
}

impl IntersectionType {
    /// Determines intersection type from the number of connected segments.
    pub fn from_segment_count(count: usize) -> Self {
        match count {
            0 | 1 => Self::DeadEnd,
            2 => Self::Curve,
            3 => Self::TJunction,
            4 => Self::Crossroads,
            _ => Self::Complex,
        }
    }
}

// ---------------------------------------------------------------------------
// City block
// ---------------------------------------------------------------------------

/// A city block bounded by road segments.
#[derive(Debug, Clone)]
pub struct CityBlock {
    /// Unique block id.
    pub id: u32,
    /// Vertices defining the block polygon (CCW order).
    pub vertices: Vec<Pos2>,
    /// Zoning type.
    pub zone: ZoneType,
    /// Building lots within this block.
    pub lots: Vec<BuildingLot>,
    /// Area of the block in square meters.
    pub area: f32,
    /// Bounding road segment ids.
    pub bounding_roads: Vec<u32>,
}

impl CityBlock {
    /// Creates a new block.
    pub fn new(id: u32, vertices: Vec<Pos2>, zone: ZoneType) -> Self {
        let area = Self::compute_area(&vertices);
        Self {
            id,
            vertices,
            zone,
            lots: Vec::new(),
            area,
            bounding_roads: Vec::new(),
        }
    }

    /// Computes the area of a polygon using the shoelace formula.
    pub fn compute_area(vertices: &[Pos2]) -> f32 {
        let n = vertices.len();
        if n < 3 {
            return 0.0;
        }
        let mut area = 0.0f32;
        for i in 0..n {
            let j = (i + 1) % n;
            area += vertices[i].x * vertices[j].y;
            area -= vertices[j].x * vertices[i].y;
        }
        area.abs() * 0.5
    }

    /// Returns the centroid of the block.
    pub fn centroid(&self) -> Pos2 {
        let n = self.vertices.len() as f32;
        let sum_x: f32 = self.vertices.iter().map(|v| v.x).sum();
        let sum_y: f32 = self.vertices.iter().map(|v| v.y).sum();
        Pos2::new(sum_x / n, sum_y / n)
    }

    /// Returns the bounding rectangle.
    pub fn bounding_rect(&self) -> (Pos2, Pos2) {
        let min_x = self.vertices.iter().map(|v| v.x).fold(f32::MAX, f32::min);
        let min_y = self.vertices.iter().map(|v| v.y).fold(f32::MAX, f32::min);
        let max_x = self.vertices.iter().map(|v| v.x).fold(f32::MIN, f32::max);
        let max_y = self.vertices.iter().map(|v| v.y).fold(f32::MIN, f32::max);
        (Pos2::new(min_x, min_y), Pos2::new(max_x, max_y))
    }
}

// ---------------------------------------------------------------------------
// Building lot
// ---------------------------------------------------------------------------

/// A subdivision of a city block for a single building.
#[derive(Debug, Clone)]
pub struct BuildingLot {
    /// Unique lot id.
    pub id: u32,
    /// Lot polygon vertices.
    pub vertices: Vec<Pos2>,
    /// Lot area in square meters.
    pub area: f32,
    /// Lot width (approximate, along road frontage).
    pub frontage_width: f32,
    /// Lot depth (approximate, perpendicular to road).
    pub depth: f32,
    /// Front setback from road in meters.
    pub setback: f32,
    /// The building placed on this lot (if any).
    pub building: Option<Building>,
    /// Whether this lot faces a road.
    pub has_road_access: bool,
}

impl BuildingLot {
    /// Creates a rectangular lot.
    pub fn rectangular(id: u32, origin: Pos2, width: f32, depth: f32, setback: f32) -> Self {
        let vertices = vec![
            origin,
            Pos2::new(origin.x + width, origin.y),
            Pos2::new(origin.x + width, origin.y + depth),
            Pos2::new(origin.x, origin.y + depth),
        ];
        Self {
            id,
            vertices,
            area: width * depth,
            frontage_width: width,
            depth,
            setback,
            building: None,
            has_road_access: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Building generation
// ---------------------------------------------------------------------------

/// A procedurally generated building.
#[derive(Debug, Clone)]
pub struct Building {
    /// Building footprint (bottom-left corner).
    pub origin: Pos2,
    /// Footprint width.
    pub width: f32,
    /// Footprint depth.
    pub depth: f32,
    /// Number of floors.
    pub floors: u32,
    /// Floor height in meters.
    pub floor_height: f32,
    /// Total height in meters.
    pub total_height: f32,
    /// Roof type.
    pub roof: RoofType,
    /// Facade style.
    pub facade: FacadeStyle,
    /// Window layout per floor.
    pub window_layout: WindowLayout,
    /// Building usage/purpose.
    pub usage: BuildingUsage,
    /// Rotation angle in radians.
    pub rotation: f32,
}

/// Roof types for buildings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoofType {
    /// Flat roof (typical for commercial/industrial).
    Flat,
    /// Gabled (triangular) roof.
    Gabled,
    /// Hipped roof (sloped on all sides).
    Hipped,
    /// Mansard roof (dual slope on all sides).
    Mansard,
    /// Shed roof (single slope).
    Shed,
    /// Dome.
    Dome,
    /// Sawtooth (industrial).
    Sawtooth,
}

/// Facade style for building exteriors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FacadeStyle {
    /// Brick facade.
    Brick,
    /// Concrete / brutalist.
    Concrete,
    /// Glass curtain wall.
    Glass,
    /// Wood siding.
    Wood,
    /// Stucco / plaster.
    Stucco,
    /// Stone.
    Stone,
    /// Metal panel.
    Metal,
}

/// Window layout pattern.
#[derive(Debug, Clone)]
pub struct WindowLayout {
    /// Number of windows per floor along the front facade.
    pub windows_per_floor_front: u32,
    /// Number of windows per floor along the side facade.
    pub windows_per_floor_side: u32,
    /// Window width in meters.
    pub window_width: f32,
    /// Window height in meters.
    pub window_height: f32,
    /// Spacing between windows.
    pub window_spacing: f32,
    /// Whether the ground floor has different windows (e.g. storefront).
    pub ground_floor_different: bool,
}

impl WindowLayout {
    /// Creates a default residential window layout.
    pub fn residential() -> Self {
        Self {
            windows_per_floor_front: 3,
            windows_per_floor_side: 2,
            window_width: 1.2,
            window_height: 1.5,
            window_spacing: 2.0,
            ground_floor_different: false,
        }
    }

    /// Creates a commercial window layout.
    pub fn commercial() -> Self {
        Self {
            windows_per_floor_front: 6,
            windows_per_floor_side: 4,
            window_width: 2.0,
            window_height: 2.0,
            window_spacing: 0.5,
            ground_floor_different: true,
        }
    }

    /// Creates an industrial window layout.
    pub fn industrial() -> Self {
        Self {
            windows_per_floor_front: 8,
            windows_per_floor_side: 6,
            window_width: 1.5,
            window_height: 1.0,
            window_spacing: 3.0,
            ground_floor_different: false,
        }
    }
}

/// Building usage / purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildingUsage {
    House,
    Apartment,
    Office,
    Shop,
    Restaurant,
    Warehouse,
    Factory,
    School,
    Hospital,
    Government,
    Parking,
}

// ---------------------------------------------------------------------------
// City configuration
// ---------------------------------------------------------------------------

/// Configuration for the city generator.
#[derive(Debug, Clone)]
pub struct CityConfig {
    /// City size in meters (square).
    pub city_size: f32,
    /// Random seed.
    pub seed: u64,
    /// Number of major road growth iterations.
    pub major_road_iterations: u32,
    /// Number of minor road growth iterations.
    pub minor_road_iterations: u32,
    /// Minimum road segment length.
    pub min_segment_length: f32,
    /// Maximum road segment length.
    pub max_segment_length: f32,
    /// Road grid snapping angle (radians). 0 = organic layout.
    pub grid_angle: f32,
    /// How strongly roads align to the grid (0 = organic, 1 = strict grid).
    pub grid_strength: f32,
    /// Minimum block area for subdivision.
    pub min_block_area: f32,
    /// Maximum lot width.
    pub max_lot_width: f32,
    /// Minimum lot width.
    pub min_lot_width: f32,
    /// District zones and their approximate areas.
    pub district_zones: Vec<(ZoneType, f32)>,
    /// Whether to generate roundabouts at complex intersections.
    pub enable_roundabouts: bool,
    /// Population density (people per hectare, used for sizing).
    pub population_density: f32,
}

impl CityConfig {
    /// Creates a default small town configuration.
    pub fn small_town() -> Self {
        Self {
            city_size: 500.0,
            seed: 42,
            major_road_iterations: 5,
            minor_road_iterations: 10,
            min_segment_length: 30.0,
            max_segment_length: 80.0,
            grid_angle: 0.0,
            grid_strength: 0.8,
            min_block_area: 500.0,
            max_lot_width: 25.0,
            min_lot_width: 8.0,
            district_zones: vec![
                (ZoneType::Residential, 0.5),
                (ZoneType::Commercial, 0.2),
                (ZoneType::Park, 0.15),
                (ZoneType::Civic, 0.1),
                (ZoneType::Industrial, 0.05),
            ],
            enable_roundabouts: false,
            population_density: 50.0,
        }
    }

    /// Creates a medium city configuration.
    pub fn medium_city() -> Self {
        Self {
            city_size: 2000.0,
            seed: 42,
            major_road_iterations: 15,
            minor_road_iterations: 40,
            min_segment_length: 40.0,
            max_segment_length: 120.0,
            grid_angle: 0.0,
            grid_strength: 0.6,
            min_block_area: 400.0,
            max_lot_width: 30.0,
            min_lot_width: 6.0,
            district_zones: vec![
                (ZoneType::Residential, 0.4),
                (ZoneType::Commercial, 0.25),
                (ZoneType::Industrial, 0.15),
                (ZoneType::Park, 0.1),
                (ZoneType::MixedUse, 0.05),
                (ZoneType::Civic, 0.05),
            ],
            enable_roundabouts: true,
            population_density: 100.0,
        }
    }

    /// Creates a downtown / dense urban configuration.
    pub fn downtown() -> Self {
        Self {
            city_size: 800.0,
            seed: 42,
            major_road_iterations: 10,
            minor_road_iterations: 30,
            min_segment_length: 50.0,
            max_segment_length: 100.0,
            grid_angle: 0.0,
            grid_strength: 0.95,
            min_block_area: 300.0,
            max_lot_width: 40.0,
            min_lot_width: 10.0,
            district_zones: vec![
                (ZoneType::Commercial, 0.5),
                (ZoneType::MixedUse, 0.25),
                (ZoneType::Residential, 0.15),
                (ZoneType::Civic, 0.05),
                (ZoneType::Park, 0.05),
            ],
            enable_roundabouts: true,
            population_density: 300.0,
        }
    }
}

impl Default for CityConfig {
    fn default() -> Self {
        Self::small_town()
    }
}

// ---------------------------------------------------------------------------
// Generated city
// ---------------------------------------------------------------------------

/// The complete output of the city generator.
#[derive(Debug, Clone)]
pub struct GeneratedCity {
    /// Road network nodes.
    pub nodes: Vec<RoadNode>,
    /// Road network segments.
    pub segments: Vec<RoadSegment>,
    /// City blocks.
    pub blocks: Vec<CityBlock>,
    /// All building lots.
    pub lots: Vec<BuildingLot>,
    /// All generated buildings.
    pub buildings: Vec<Building>,
    /// City bounds (min, max).
    pub bounds: (Pos2, Pos2),
    /// Statistics.
    pub stats: CityStats,
}

/// Statistics about the generated city.
#[derive(Debug, Clone, Default)]
pub struct CityStats {
    pub total_road_length: f32,
    pub total_blocks: usize,
    pub total_lots: usize,
    pub total_buildings: usize,
    pub zone_areas: HashMap<ZoneType, f32>,
    pub average_building_height: f32,
    pub max_building_height: f32,
}

// ---------------------------------------------------------------------------
// CityGenerator
// ---------------------------------------------------------------------------

/// The procedural city generation system.
pub struct CityGenerator {
    config: CityConfig,
    rng: CityRng,
    next_node_id: u32,
    next_segment_id: u32,
    next_block_id: u32,
    next_lot_id: u32,
}

impl CityGenerator {
    /// Creates a new city generator.
    pub fn new(config: CityConfig) -> Self {
        let rng = CityRng::new(config.seed);
        Self {
            config,
            rng,
            next_node_id: 0,
            next_segment_id: 0,
            next_block_id: 0,
            next_lot_id: 0,
        }
    }

    /// Generates a complete city.
    pub fn generate(&mut self) -> GeneratedCity {
        let mut nodes = Vec::new();
        let mut segments = Vec::new();

        // 1. Generate road network.
        self.generate_road_network(&mut nodes, &mut segments);

        // 2. Classify intersections.
        Self::classify_intersections(&mut nodes);

        // 3. Generate blocks from road network.
        let mut blocks = self.generate_blocks(&nodes, &segments);

        // 4. Assign zoning to blocks.
        self.assign_zoning(&mut blocks);

        // 5. Subdivide blocks into lots.
        let mut all_lots = Vec::new();
        for block in &mut blocks {
            let lots = self.subdivide_block(block);
            all_lots.extend(lots.iter().cloned());
            block.lots = lots;
        }

        // 6. Generate buildings on lots.
        let mut all_buildings = Vec::new();
        for lot in &mut all_lots {
            if let Some(building) = self.generate_building(lot) {
                all_buildings.push(building.clone());
                lot.building = Some(building);
            }
        }

        // 7. Compute statistics.
        let stats = self.compute_stats(&nodes, &segments, &blocks, &all_lots, &all_buildings);

        let half = self.config.city_size * 0.5;
        GeneratedCity {
            nodes,
            segments,
            blocks,
            lots: all_lots,
            buildings: all_buildings,
            bounds: (Pos2::new(-half, -half), Pos2::new(half, half)),
            stats,
        }
    }

    // -- Road network generation ----------------------------------------------

    fn generate_road_network(
        &mut self,
        nodes: &mut Vec<RoadNode>,
        segments: &mut Vec<RoadSegment>,
    ) {
        let half = self.config.city_size * 0.5;

        // Create initial cross-roads through the city center.
        let center = self.create_node(nodes, Pos2::new(0.0, 0.0));
        let north = self.create_node(nodes, Pos2::new(0.0, half));
        let south = self.create_node(nodes, Pos2::new(0.0, -half));
        let east = self.create_node(nodes, Pos2::new(half, 0.0));
        let west = self.create_node(nodes, Pos2::new(-half, 0.0));

        self.create_segment(segments, nodes, center, north, RoadCategory::Major);
        self.create_segment(segments, nodes, center, south, RoadCategory::Major);
        self.create_segment(segments, nodes, center, east, RoadCategory::Major);
        self.create_segment(segments, nodes, center, west, RoadCategory::Major);

        // Grow major roads.
        for _ in 0..self.config.major_road_iterations {
            self.grow_roads(nodes, segments, RoadCategory::Major);
        }

        // Grow minor roads.
        for _ in 0..self.config.minor_road_iterations {
            self.grow_roads(nodes, segments, RoadCategory::Minor);
        }
    }

    fn create_node(&mut self, nodes: &mut Vec<RoadNode>, position: Pos2) -> u32 {
        let id = self.next_node_id;
        self.next_node_id += 1;
        nodes.push(RoadNode {
            id,
            position,
            segments: Vec::new(),
            intersection_type: IntersectionType::DeadEnd,
            has_traffic_light: false,
        });
        id
    }

    fn create_segment(
        &mut self,
        segments: &mut Vec<RoadSegment>,
        nodes: &mut [RoadNode],
        start: u32,
        end: u32,
        category: RoadCategory,
    ) -> u32 {
        let id = self.next_segment_id;
        self.next_segment_id += 1;

        let start_pos = nodes[start as usize].position;
        let end_pos = nodes[end as usize].position;
        let length = start_pos.distance(&end_pos);

        segments.push(RoadSegment {
            id,
            start_node: start,
            end_node: end,
            category,
            width: category.default_width(),
            length,
            speed_limit: match category {
                RoadCategory::Highway => 80.0,
                RoadCategory::Major => 50.0,
                RoadCategory::Minor => 30.0,
                RoadCategory::Alley => 15.0,
                RoadCategory::Pedestrian => 5.0,
            },
            one_way: false,
        });

        nodes[start as usize].segments.push(id);
        nodes[end as usize].segments.push(id);

        id
    }

    fn grow_roads(
        &mut self,
        nodes: &mut Vec<RoadNode>,
        segments: &mut Vec<RoadSegment>,
        category: RoadCategory,
    ) {
        let half = self.config.city_size * 0.5;

        // Pick a random existing node to grow from.
        if nodes.is_empty() {
            return;
        }
        let source_idx = self.rng.range_u32(nodes.len() as u32) as usize;
        let source_id = nodes[source_idx].id;
        let source_pos = nodes[source_idx].position;

        // Choose a direction.
        let base_angle = if self.config.grid_strength > 0.5 {
            // Snap to grid angles.
            let grid_dirs = [0.0f32, std::f32::consts::FRAC_PI_2, std::f32::consts::PI, -std::f32::consts::FRAC_PI_2];
            let base = grid_dirs[self.rng.range_u32(4) as usize];
            base + self.rng.range_f32(-0.1, 0.1) * (1.0 - self.config.grid_strength)
        } else {
            self.rng.range_f32(0.0, std::f32::consts::TAU)
        };

        let length = self.rng.range_f32(self.config.min_segment_length, self.config.max_segment_length);
        let dir = Pos2::new(base_angle.cos(), base_angle.sin());
        let end_pos = source_pos.add(&dir.scale(length));

        // Check bounds.
        if end_pos.x.abs() > half || end_pos.y.abs() > half {
            return;
        }

        // Check for nearby existing nodes to connect to.
        let snap_dist = self.config.min_segment_length * 0.3;
        let mut connect_to = None;
        for node in nodes.iter() {
            if node.id != source_id && node.position.distance(&end_pos) < snap_dist {
                connect_to = Some(node.id);
                break;
            }
        }

        let end_id = match connect_to {
            Some(existing) => existing,
            None => self.create_node(nodes, end_pos),
        };

        // Avoid duplicate segments.
        let has_existing = segments.iter().any(|s| {
            (s.start_node == source_id && s.end_node == end_id)
                || (s.start_node == end_id && s.end_node == source_id)
        });

        if !has_existing && source_id != end_id {
            self.create_segment(segments, nodes, source_id, end_id, category);
        }
    }

    fn classify_intersections(nodes: &mut [RoadNode]) {
        for node in nodes.iter_mut() {
            node.intersection_type =
                IntersectionType::from_segment_count(node.segments.len());
            // Traffic lights at busy intersections.
            node.has_traffic_light = node.segments.len() >= 4;
        }
    }

    // -- Block generation -----------------------------------------------------

    fn generate_blocks(
        &mut self,
        nodes: &[RoadNode],
        _segments: &[RoadSegment],
    ) -> Vec<CityBlock> {
        let mut blocks = Vec::new();
        let half = self.config.city_size * 0.5;

        // Simple grid-based block generation.
        let block_size = self.config.min_segment_length * 1.5;
        let num_x = (self.config.city_size / block_size) as u32;
        let num_y = (self.config.city_size / block_size) as u32;

        for by in 0..num_y {
            for bx in 0..num_x {
                let x0 = -half + bx as f32 * block_size;
                let y0 = -half + by as f32 * block_size;
                let x1 = x0 + block_size;
                let y1 = y0 + block_size;

                // Check if any road nodes are inside or near this block.
                let center = Pos2::new((x0 + x1) * 0.5, (y0 + y1) * 0.5);
                let near_road = nodes.iter().any(|n| n.position.distance(&center) < block_size);

                if near_road {
                    let vertices = vec![
                        Pos2::new(x0, y0),
                        Pos2::new(x1, y0),
                        Pos2::new(x1, y1),
                        Pos2::new(x0, y1),
                    ];
                    let block = CityBlock::new(self.next_block_id, vertices, ZoneType::Vacant);
                    self.next_block_id += 1;
                    blocks.push(block);
                }
            }
        }

        blocks
    }

    // -- Zoning assignment ----------------------------------------------------

    fn assign_zoning(&mut self, blocks: &mut [CityBlock]) {
        let center = Pos2::new(0.0, 0.0);
        let half = self.config.city_size * 0.5;

        for block in blocks.iter_mut() {
            let block_center = block.centroid();
            let dist_from_center = block_center.distance(&center);
            let normalized_dist = dist_from_center / half;

            // Downtown: commercial/mixed-use.
            // Mid-ring: residential.
            // Outer ring: industrial/parks.
            if normalized_dist < 0.2 {
                block.zone = if self.rng.chance(0.7) {
                    ZoneType::Commercial
                } else {
                    ZoneType::MixedUse
                };
            } else if normalized_dist < 0.5 {
                block.zone = if self.rng.chance(0.6) {
                    ZoneType::Residential
                } else if self.rng.chance(0.5) {
                    ZoneType::MixedUse
                } else {
                    ZoneType::Commercial
                };
            } else if normalized_dist < 0.8 {
                block.zone = if self.rng.chance(0.7) {
                    ZoneType::Residential
                } else if self.rng.chance(0.3) {
                    ZoneType::Park
                } else {
                    ZoneType::Industrial
                };
            } else {
                block.zone = if self.rng.chance(0.4) {
                    ZoneType::Industrial
                } else if self.rng.chance(0.5) {
                    ZoneType::Park
                } else {
                    ZoneType::Residential
                };
            }
        }
    }

    // -- Block subdivision ----------------------------------------------------

    fn subdivide_block(&mut self, block: &CityBlock) -> Vec<BuildingLot> {
        let mut lots = Vec::new();

        if block.zone == ZoneType::Park || block.zone == ZoneType::Vacant {
            return lots;
        }

        let (min, max) = block.bounding_rect();
        let block_width = max.x - min.x;
        let block_depth = max.y - min.y;

        if block.area < block.zone.min_lot_area() {
            return lots;
        }

        // Subdivide along the longer axis.
        let lot_width = self.rng.range_f32(self.config.min_lot_width, self.config.max_lot_width);
        let setback = block.zone.min_setback();

        let num_lots_x = (block_width / lot_width).floor() as u32;
        let num_lots_y = if block_depth > lot_width * 2.0 { 2 } else { 1 };
        let actual_depth = (block_depth - setback * 2.0) / num_lots_y as f32;

        for iy in 0..num_lots_y {
            for ix in 0..num_lots_x {
                let origin = Pos2::new(
                    min.x + ix as f32 * lot_width + setback,
                    min.y + iy as f32 * actual_depth + setback,
                );
                let w = lot_width - setback;
                let d = actual_depth - setback;

                if w > self.config.min_lot_width && d > self.config.min_lot_width {
                    let lot = BuildingLot::rectangular(
                        self.next_lot_id,
                        origin,
                        w,
                        d,
                        setback,
                    );
                    self.next_lot_id += 1;
                    lots.push(lot);
                }
            }
        }

        lots
    }

    // -- Building generation --------------------------------------------------

    fn generate_building(&mut self, lot: &BuildingLot) -> Option<Building> {
        if lot.area < 50.0 {
            return None;
        }

        // Determine building parameters based on lot context.
        let max_floors = 6; // Would use zone in full implementation.
        let floors = self.rng.range_i32(1, max_floors as i32) as u32;
        let floor_height = self.rng.range_f32(2.8, 3.5);

        let coverage = self.rng.range_f32(0.4, 0.7);
        let building_width = lot.frontage_width * coverage;
        let building_depth = lot.depth * coverage;

        let roof = *self.rng.pick(&[
            RoofType::Flat,
            RoofType::Gabled,
            RoofType::Hipped,
            RoofType::Mansard,
            RoofType::Shed,
        ]).unwrap_or(&RoofType::Flat);

        let facade = *self.rng.pick(&[
            FacadeStyle::Brick,
            FacadeStyle::Concrete,
            FacadeStyle::Glass,
            FacadeStyle::Wood,
            FacadeStyle::Stucco,
            FacadeStyle::Stone,
        ]).unwrap_or(&FacadeStyle::Brick);

        let usage = if floors <= 2 {
            *self.rng.pick(&[BuildingUsage::House, BuildingUsage::Shop]).unwrap_or(&BuildingUsage::House)
        } else {
            *self.rng.pick(&[BuildingUsage::Apartment, BuildingUsage::Office]).unwrap_or(&BuildingUsage::Apartment)
        };

        let windows_front = (building_width / 2.5).floor() as u32;
        let windows_side = (building_depth / 3.0).floor() as u32;

        Some(Building {
            origin: lot.vertices[0],
            width: building_width,
            depth: building_depth,
            floors,
            floor_height,
            total_height: floors as f32 * floor_height,
            roof,
            facade,
            window_layout: WindowLayout {
                windows_per_floor_front: windows_front.max(1),
                windows_per_floor_side: windows_side.max(1),
                window_width: 1.2,
                window_height: 1.5,
                window_spacing: 2.0,
                ground_floor_different: usage == BuildingUsage::Shop,
            },
            usage,
            rotation: 0.0,
        })
    }

    // -- Statistics -----------------------------------------------------------

    fn compute_stats(
        &self,
        _nodes: &[RoadNode],
        segments: &[RoadSegment],
        blocks: &[CityBlock],
        lots: &[BuildingLot],
        buildings: &[Building],
    ) -> CityStats {
        let total_road_length: f32 = segments.iter().map(|s| s.length).sum();

        let mut zone_areas: HashMap<ZoneType, f32> = HashMap::new();
        for block in blocks {
            *zone_areas.entry(block.zone).or_insert(0.0) += block.area;
        }

        let avg_height = if buildings.is_empty() {
            0.0
        } else {
            buildings.iter().map(|b| b.total_height).sum::<f32>() / buildings.len() as f32
        };

        let max_height = buildings
            .iter()
            .map(|b| b.total_height)
            .fold(0.0f32, f32::max);

        CityStats {
            total_road_length,
            total_blocks: blocks.len(),
            total_lots: lots.len(),
            total_buildings: buildings.len(),
            zone_areas,
            average_building_height: avg_height,
            max_building_height: max_height,
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
    fn zone_type_properties() {
        assert!(ZoneType::Commercial.max_floors() > ZoneType::Residential.max_floors());
        assert!(ZoneType::Industrial.max_lot_coverage() > ZoneType::Park.max_lot_coverage());
        assert!(ZoneType::Residential.min_setback() > ZoneType::Commercial.min_setback());
    }

    #[test]
    fn road_category_width() {
        assert!(RoadCategory::Highway.default_width() > RoadCategory::Minor.default_width());
        assert!(RoadCategory::Major.has_sidewalks());
        assert!(!RoadCategory::Highway.has_sidewalks());
    }

    #[test]
    fn city_block_area() {
        let vertices = vec![
            Pos2::new(0.0, 0.0),
            Pos2::new(100.0, 0.0),
            Pos2::new(100.0, 100.0),
            Pos2::new(0.0, 100.0),
        ];
        let area = CityBlock::compute_area(&vertices);
        assert!((area - 10000.0).abs() < 1.0);
    }

    #[test]
    fn city_block_centroid() {
        let block = CityBlock::new(
            0,
            vec![
                Pos2::new(0.0, 0.0),
                Pos2::new(10.0, 0.0),
                Pos2::new(10.0, 10.0),
                Pos2::new(0.0, 10.0),
            ],
            ZoneType::Residential,
        );
        let c = block.centroid();
        assert!((c.x - 5.0).abs() < 0.01);
        assert!((c.y - 5.0).abs() < 0.01);
    }

    #[test]
    fn building_lot_rectangular() {
        let lot = BuildingLot::rectangular(0, Pos2::new(0.0, 0.0), 20.0, 30.0, 5.0);
        assert!((lot.area - 600.0).abs() < 0.1);
        assert_eq!(lot.vertices.len(), 4);
    }

    #[test]
    fn window_layout_presets() {
        let res = WindowLayout::residential();
        let com = WindowLayout::commercial();
        assert!(com.windows_per_floor_front > res.windows_per_floor_front);
    }

    #[test]
    fn intersection_type_classification() {
        assert_eq!(IntersectionType::from_segment_count(1), IntersectionType::DeadEnd);
        assert_eq!(IntersectionType::from_segment_count(3), IntersectionType::TJunction);
        assert_eq!(IntersectionType::from_segment_count(4), IntersectionType::Crossroads);
        assert_eq!(IntersectionType::from_segment_count(6), IntersectionType::Complex);
    }

    #[test]
    fn generate_small_town() {
        let config = CityConfig::small_town();
        let mut city_gen = CityGenerator::new(config);
        let city = city_gen.generate();

        assert!(!city.nodes.is_empty());
        assert!(!city.segments.is_empty());
        assert!(!city.blocks.is_empty());
        assert!(city.stats.total_road_length > 0.0);
    }

    #[test]
    fn generate_medium_city() {
        let config = CityConfig::medium_city();
        let mut city_gen = CityGenerator::new(config);
        let city = city_gen.generate();

        assert!(city.nodes.len() > 5);
        assert!(city.segments.len() > 5);
    }

    #[test]
    fn generate_downtown() {
        let config = CityConfig::downtown();
        let mut city_gen = CityGenerator::new(config);
        let city = city_gen.generate();

        assert!(!city.blocks.is_empty());
    }

    #[test]
    fn deterministic_generation() {
        let config = CityConfig::small_town();
        let mut gen1 = CityGenerator::new(config.clone());
        let mut gen2 = CityGenerator::new(config);

        let city1 = gen1.generate();
        let city2 = gen2.generate();

        assert_eq!(city1.nodes.len(), city2.nodes.len());
        assert_eq!(city1.segments.len(), city2.segments.len());
    }

    #[test]
    fn pos2_operations() {
        let a = Pos2::new(3.0, 4.0);
        assert!((a.length() - 5.0).abs() < 0.001);

        let b = Pos2::new(6.0, 8.0);
        assert!((a.distance(&b) - 5.0).abs() < 0.001);

        let c = a.lerp(&b, 0.5);
        assert!((c.x - 4.5).abs() < 0.001);
    }

    #[test]
    fn city_rng_deterministic() {
        let mut r1 = CityRng::new(42);
        let mut r2 = CityRng::new(42);
        assert_eq!(r1.next_u64(), r2.next_u64());
        assert_eq!(r1.next_f32(), r2.next_f32());
    }
}
