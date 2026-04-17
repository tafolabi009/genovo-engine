//! # genovo-world
//!
//! World streaming, level management, LOD, and spline tools for the Genovo
//! game engine.
//!
//! This crate provides large-world support through spatial partitioning,
//! asynchronous cell streaming with memory budgets, level management with
//! seamless transitions, distance-based LOD evaluation, and spline/path
//! utilities for roads, rivers, and entity movement.

pub mod lod_manager;
pub mod level;
pub mod navigation;
pub mod occlusion;
pub mod partition;
pub mod spline;
pub mod streaming;
pub mod world_physics;

// Re-exports for convenience.
pub use level::{Level, LevelManager, LevelSettings, SubLevel, SubLevelState};
pub use lod_manager::{
    CrossfadeState, HLODCluster, ImpostorData, LODComponent, LODGroup, LODLevel, LODManager,
    LODStats, ShadowMode,
};
pub use partition::{
    CellCoord, StreamingOps, WorldCell, WorldLayer, WorldPartition, CellState,
};
pub use spline::{
    SplineComponent, SplineFollower, SplineMesh, SplinePath, SplinePoint, SplineScatter,
};
pub use occlusion::{
    HierarchicalZBuffer, OcclusionBuffer, OcclusionBufferStats, OcclusionCamera,
    OcclusionCullStats, OcclusionCuller, OccluderComponent, OccluderEntry, Portal,
    PortalOcclusion, PortalOcclusionConfig, PortalPlane, RenderQueueFilter, Room, RoomId,
};
pub use streaming::{
    StreamRequest, StreamRequestPriority, StreamingManager, StreamingStats, CellTransitionState,
};
pub use world_physics::{
    BodyType, PhysicsBody, PhysicsBodyId, PhysicsLOD, PhysicsLODConfig, PhysicsProxy,
    PhysicsRegion, RegionId, RegionShape, WorldPhysicsManager, WorldPhysicsStats,
};
pub use navigation::{
    DynamicObstacle, LinkId, LinkType, NavConfig, NavFlags, NavPath, NavPoly, NavigationManager,
    ObstacleId, OffMeshLink, PathCorridor, PolyId,
};

// Massive world: EVE Online-scale universe structure, server instancing with
// cross-instance travel, hierarchical spatial database, and world persistence
// with incremental sector saving.
pub mod massive_world;

pub use massive_world::{
    CelestialBody, CelestialType, GalaxyMap, GalaxyPosition, GalaxyStats,
    Instance, InstanceId, InstanceManager, InstanceManagerStats, InstanceState,
    MassiveWorldSystem, PlayerId, SaveHeader, SaveState, SectorId,
    SectorSaveData, SolarSystem, SpatialDatabase, SpatialDatabaseStats,
    SpatialEntry, SpatialEntityId, SystemId, SystemLOD, TransferRequest,
    WorldPersistence, PersistenceStats,
};
