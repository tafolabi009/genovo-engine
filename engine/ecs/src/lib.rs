//! # genovo-ecs
//!
//! Entity-Component-System framework for the Genovo game engine.
//!
//! This crate provides the core ECS primitives:
//!
//! - **Entities** -- lightweight handles identifying game objects.
//! - **Components** -- plain data attached to entities.
//! - **Systems** -- logic that operates on entities via `&mut World`.
//! - **Queries** -- iterate entities matching a set of component types.
//! - **Events** -- decoupled inter-system communication.
//! - **World** -- the top-level container tying everything together.
//! - **Archetypes** -- cache-friendly column storage grouped by component set.
//! - **Commands** -- deferred world mutations for safe structural changes.
//! - **Schedule** -- advanced system scheduling with stages and parallelism.
//! - **Change Detection** -- tick-based tracking of component modifications.
//! - **Resources** -- singleton values with change-tracked access wrappers.

pub mod archetype;
pub mod bundle;
pub mod change_detection;
pub mod commands;
pub mod component;
pub mod entity;
pub mod entity_commands_ext;
pub mod event;
pub mod observer;
pub mod parallel_executor;
pub mod query;
pub mod relations;
pub mod resource;
pub mod schedule;
pub mod system;
pub mod system_params;
pub mod world;
pub mod world_builder;
pub mod world_query;
pub mod query_cache;
pub mod sparse_set_v2;
pub mod system_graph;

// Query filter system: With<T>, Without<T>, Changed<T>, Added<T>, Or<A,B>,
// And<A,B>, Not<T>, Optional<T> -- composable filter combinators for queries.
pub mod query_filters;

// Optimized component storage: SoA (struct-of-arrays) layout, chunk iteration,
// prefetch hints, sorted iteration by entity ID, memory pool backing.
pub mod component_storage_v2;

// Component lifecycle hooks: on_insert, on_remove, on_modify callbacks per component
// type, hook registration, batch hook processing, hook priority ordering.
pub mod component_hooks;

// ECS world snapshot: serialize entire world state, diff two snapshots, restore
// from snapshot, used for undo/redo and networking.
pub mod world_snapshot;

// Entity recycling pool: pre-allocate entity IDs, recycle despawned entities,
// reduce allocation overhead, pool warm-up, statistics.
pub mod entity_pool;

// Re-exports for ergonomic use from downstream crates.
pub use component::{Component, ComponentId, ComponentStorage};
pub use entity::{Entity, EntityBuilder, EntityStorage};
pub use event::Events;
pub use query::QueryItem;
pub use system::{System, SystemSchedule};
pub use world::World;

// New re-exports.
pub use archetype::{Archetype, ArchetypeId, ComponentColumn, ComponentInfo};
pub use change_detection::{Added, Changed, ChangeTracker, ComponentTicks, Mut, Ref};
pub use commands::CommandQueue;
pub use query::{
    And, FilteredQueryIter, Has, Or, QueryBuilder, QueryFilter, QueryIter, QueryState,
    QueryStateIter, With, Without,
};
pub use observer::{
    EventCollector, LifecycleEvent, LifecycleEventKind, Observer, ObserverFlushSystem,
    ObserverId, ObserverRegistry, OnAdd, OnChange, OnRemove,
};
pub use relations::{ChildOf, Relation, RelationEdge, RelationId, RelationKind, RelationManager};
pub use resource::{Res, ResMut, ResourceTicks};
pub use schedule::{RunCriteria, Schedule, Stage, SystemAccess, SystemDescriptor, SystemSet};
pub use world_query::{
    AccessMode, AnyOf, AnyOfMarker, CompiledQuery, ComponentAccess, EntityMut, EntityRef,
    FilterDescriptor, JoinedQuery, Optional, QueryRow, QuerySnapshot, WorldQueryBuilder,
    WorldQueryBuilderBound,
};

// Bundle re-exports.
pub use bundle::{
    Bundle, BundleBuilder, CameraBundle, CameraData, ColliderData, ColliderShape, LightBundle,
    LightData, LightType, MaterialRef, MeshBundle, MeshRef, PhysicsBundle, ProjectionType,
    RigidBodyData, RigidBodyType, SpriteBundle, SpriteData, TransformData,
};

// System parameter re-exports.
pub use system_params::{
    Commands, EventReader, EventWriter, ExclusiveSystem, IntoSystem, Local, ParamSystemAdapter,
    QueryParamIter, ResParam, ResMutParam, RunCondition, RunConditionId, RunConditionRegistry,
    SystemMeta, SystemParam, SystemParamAccess, SystemParamState,
};

// World builder re-exports.
pub use world_builder::{
    EcsPlugin, SystemRegistration, SystemSetConfig, WorldBuilder, WorldMetadata,
};

// Parallel executor re-exports.
pub use parallel_executor::{
    AccessKind, BatchSchedule, ConflictGraph, ExecutionContext, ExecutionProfile,
    ExecutorConfig, ParallelBatch, ParallelExecutor, ParallelSystemDescriptor,
    SystemComponentAccess, SystemIndex,
};

// Entity commands extension re-exports.
pub use entity_commands_ext::{
    ChildSpec, CloneFilter, CloneRegistry, CloneResult, DespawnResult,
    EntityCommandsExt, EntityPrefab, HierarchyBuilder, MoveResult,
    PrefabRegistry, TransferRegistry,
};

// Sparse set v2 re-exports.
pub use sparse_set_v2::{
    IntersectionView, SparseSetIter, SparseSetIterMut, SparseSetStats, SparseSetV2,
};

// System graph re-exports.
pub use system_graph::{
    ComponentAccess as GraphComponentAccess, ComponentAccessKind, DependencyEdge,
    DependencyReason, GraphError, GraphSummary, ParallelBatch,
    SystemGraph, SystemNode, SystemIndex as GraphSystemIndex,
};

// Query cache re-exports.
pub use query_cache::{
    CacheConfig, CacheEntryInfo, CacheStats, CachedQueryResult, QueryCache, QueryKey,
};

// Archetype chunk storage: fixed-size chunks (16KB), cache-friendly iteration,
// chunk allocation pool, component data alignment, chunk iteration with change
// detection, chunk sorting by archetype.
pub mod archetype_storage;

// System execution pipeline: prepare/execute/cleanup phases, system parameter
// extraction, exclusive system support, startup/shutdown systems, system
// profiling hooks, pipeline visualization.
pub mod system_pipeline;
