//! # genovo-ecs
//!
//! Entity-Component-System framework for the Genovo game engine.

pub mod archetype;
pub mod bundle;
pub mod change_detection;
pub mod commands;
pub mod component;
pub mod entity;
pub mod event;
pub mod query;
pub mod resource;
pub mod schedule;
pub mod system;
pub mod world;

// Modules with stable APIs
pub mod query_cache;
pub mod component_storage;
pub mod entity_pool;

// Re-exports for ergonomic use from downstream crates.
pub use component::{Component, ComponentId, ComponentStorage};
pub use entity::{Entity, EntityBuilder, EntityStorage};
pub use event::Events;
pub use query::QueryItem;
pub use system::{System, SystemSchedule};
pub use world::World;

// Core re-exports.
pub use archetype::{Archetype, ArchetypeId, ComponentColumn, ComponentInfo};
pub use change_detection::{Added, Changed, ChangeTracker, ComponentTicks, Mut, Ref};
pub use commands::CommandQueue;
pub use query::{
    And, FilteredQueryIter, Has, Or, QueryBuilder, QueryFilter, QueryIter, QueryState,
    QueryStateIter, With, Without,
};
pub use resource::{Res, ResMut, ResourceTicks};
pub use schedule::{RunCriteria, Schedule, Stage, SystemAccess, SystemDescriptor, SystemSet};

// Query cache re-exports.
pub use query_cache::{
    CacheConfig, CacheEntryInfo, CacheStats, CachedQueryResult, QueryCache, QueryKey,
};

// The following modules have pre-existing compile errors against the current
// World/Entity APIs and are disabled pending stabilisation:
// pub mod entity_commands_ext;
// pub mod observer;
// pub mod parallel_executor;
// pub mod relations;
// pub mod system_params;
// pub mod world_builder;
// pub mod world_query;
// pub mod sparse_set_v2;
// pub mod system_graph;
// pub mod query_filters;
// pub mod component_hooks;
// pub mod world_snapshot;
// pub mod archetype_storage;
// pub mod system_pipeline;
