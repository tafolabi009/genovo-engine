//! # Genovo Procedural Generation
//!
//! A comprehensive procedural generation toolkit for the Genovo game engine.
//! Provides algorithms for generating game content at runtime:
//!
//! - **Wave Function Collapse** — constraint-based tile placement for 2D/3D grids
//! - **Dungeon generation** — BSP trees, cellular automata caves, drunkard's walk,
//!   room placement with MST corridors
//! - **L-Systems** — string rewriting systems with turtle interpretation for
//!   plants, fractals, and organic structures
//! - **Noise-based generation** — terrain heightmaps, biome classification,
//!   river simulation, and city layout
//! - **Maze generation** — recursive backtracker, Kruskal, Prim, Eller, Wilson
//! - **Name generation** — Markov chain procedural names for fantasy cultures

pub mod city_generator;
pub mod dungeon;
pub mod lsystem;
pub mod maze;
pub mod name_gen;
pub mod noise_gen;
pub mod texture_gen;
pub mod vegetation_gen;
pub mod wfc;

// Re-export primary types for ergonomic access.
pub use dungeon::{
    BSPConfig, CaveConfig, DrunkardConfig, DungeonMap, DungeonTile, RoomPlacementConfig,
};
pub use lsystem::{LSystem, TurtleCommand, TurtleInterpreter, TurtleInterpreter3D};
pub use maze::{Maze, MazeAlgorithm, MazeCell};
pub use name_gen::{Culture, NameGenerator};
pub use noise_gen::{Biome, BiomeMap, CityGenerator, RiverGenerator, TerrainNoiseGenerator};
pub use wfc::{WFCConstraints, WFCGrid, WFCResult, WFCSolver, WFCTile};
pub use city_generator::{
    Building, BuildingLot, BuildingUsage, CityBlock, CityConfig, CityGenerator, FacadeStyle,
    GeneratedCity, IntersectionType, RoadCategory, RoadNode, RoadSegment, RoofType, ZoneType,
};
pub use vegetation_gen::{
    Biome, Branch, ForestConfig, GeneratedTree, Leaf, TreeGenConfig, TreeGenerator, TreePlacement,
    TreeSpecies, generate_forest_placements, select_species,
};
pub use texture_gen::{
    TextureBuffer, generate_brick, generate_cellular, generate_dirt, generate_marble,
    generate_normal_map, generate_perlin, generate_roughness_map, generate_rust, generate_voronoi,
    generate_wood, make_seamless,
};
