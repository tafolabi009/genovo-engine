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
pub mod change_detection;
pub mod commands;
pub mod component;
pub mod entity;
pub mod event;
pub mod observer;
pub mod query;
pub mod relations;
pub mod resource;
pub mod schedule;
pub mod system;
pub mod world;
pub mod world_query;

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
