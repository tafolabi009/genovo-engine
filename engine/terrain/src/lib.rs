//! # Genovo Terrain
//!
//! A complete terrain system for the Genovo game engine providing heightmap-based
//! terrain with procedural generation, erosion simulation, continuous LOD,
//! splatmap-based texturing, vegetation scattering, and ECS integration.
//!
//! # Crate Organisation
//!
//! - [`heightmap`] -- Heightmap storage, sampling, procedural generation.
//! - [`mesh_generation`] -- Vertex/index buffer generation from heightmaps.
//! - [`lod`] -- Continuous Distance-Dependent LOD (CDLOD) and GeoMipMap.
//! - [`texturing`] -- Splatmap blending, terrain materials, triplanar mapping.
//! - [`vegetation`] -- Foliage scattering, instanced rendering, grass.
//! - [`erosion`] -- Hydraulic, thermal, and wind erosion simulations.
//! - [`components`] -- ECS components and systems for terrain management.

pub mod heightmap;
pub mod mesh_generation;
pub mod lod;
pub mod texturing;
pub mod vegetation;
pub mod erosion;
pub mod components;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use heightmap::Heightmap;
pub use mesh_generation::{TerrainMesh, MeshData, TerrainVertex};
pub use lod::{TerrainQuadtree, LODSettings, TerrainChunkInfo, GeoMipMap};
pub use texturing::{SplatMap, TerrainMaterial, TerrainLayer};
pub use vegetation::{VegetationLayer, VegetationInstance, GrassRenderer};
pub use erosion::{HydraulicErosion, ThermalErosion, WindErosion, TerrainBrush};
pub use components::{TerrainComponent, TerrainSystem, TerrainCollider};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the terrain crate.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TerrainError {
    /// The heightmap dimensions are invalid (zero or non-power-of-two+1).
    #[error("Invalid heightmap dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    /// The provided data length does not match the expected dimensions.
    #[error("Data length mismatch: expected {expected}, got {actual}")]
    DataLengthMismatch { expected: usize, actual: usize },

    /// The heightmap image data could not be parsed.
    #[error("Failed to parse heightmap image: {0}")]
    ImageParseError(String),

    /// A mesh generation operation failed.
    #[error("Mesh generation error: {0}")]
    MeshGeneration(String),

    /// A texturing operation failed.
    #[error("Texturing error: {0}")]
    TexturingError(String),

    /// An erosion simulation produced invalid state.
    #[error("Erosion simulation error: {0}")]
    ErosionError(String),

    /// LOD system encountered an invalid configuration.
    #[error("LOD configuration error: {0}")]
    LodError(String),

    /// The coordinate is out of the heightmap bounds.
    #[error("Coordinate out of bounds: ({x}, {z}) not in [0..{width}, 0..{height}]")]
    OutOfBounds {
        x: f32,
        z: f32,
        width: u32,
        height: u32,
    },
}

/// Convenience alias for terrain results.
pub type TerrainResult<T> = Result<T, TerrainError>;
