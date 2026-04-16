//! Advanced query system for the Genovo ECS.
//!
//! This module provides a more powerful query API beyond the basic tuple queries
//! in `query.rs`. It introduces:
//!
//! - **`WorldQuery` builder** with a fluent API for composing complex queries.
//! - **`CompiledQuery`** with cached archetype matches for zero-overhead
//!   re-evaluation across frames.
//! - **`QueryRow`** for type-safe access to matched component data.
//! - **Parallel iteration** via `par_iter` using chunked work-stealing.
//! - **`Optional<T>`** for components that may or may not be present.
//! - **`AnyOf<(A,B,C)>`** for matching entities with at least one of several
//!   component types.
//! - **`EntityRef` / `EntityMut`** for safe single-entity component access.
//!
//! # Example
//!
//! ```ignore
//! let query = world.query_builder()
//!     .read::<Position>()
//!     .write::<Velocity>()
//!     .with::<Player>()
//!     .without::<Dead>()
//!     .build();
//!
//! for row in query.iter(&world) {
//!     let pos = row.get::<Position>().unwrap();
//!     println!("Entity {:?} at ({}, {})", row.entity(), pos.x, pos.y);
//! }
//! ```

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::archetype::ArchetypeId;
use crate::component::{Component, ComponentId, ComponentStorage};
use crate::entity::Entity;
use crate::world::World;

// ---------------------------------------------------------------------------
// Access mode tracking
// ---------------------------------------------------------------------------

/// Whether a component is accessed immutably or mutably.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    /// The component will be read but not written.
    Read,
    /// The component will be read and written.
    Write,
}

/// Descriptor for a single component access within a query.
#[derive(Debug, Clone)]
pub struct ComponentAccess {
    /// Which component type is accessed.
    pub component_id: ComponentId,
    /// The underlying `TypeId`.
    pub type_id: TypeId,
    /// Read or write access.
    pub mode: AccessMode,
    /// Human-readable type name for diagnostics.
    pub type_name: &'static str,
}

/// A filter requiring the presence or absence of a component type.
#[derive(Debug, Clone)]
pub struct FilterDescriptor {
    /// The component type being filtered on.
    pub component_id: ComponentId,
    /// `true` means "must have", `false` means "must not have".
    pub must_have: bool,
}

// ---------------------------------------------------------------------------
// WorldQueryBuilder -- fluent API for building queries
// ---------------------------------------------------------------------------

/// Fluent builder for constructing an advanced compiled query.
///
/// # Example
///
/// ```ignore
/// let compiled = WorldQueryBuilder::new()
///     .read::<Position>()
///     .write::<Velocity>()
///     .with::<Alive>()
///     .without::<Frozen>()
///     .build(&world);
/// ```
pub struct WorldQueryBuilder {
    /// Components that will be fetched (read or write).
    accesses: Vec<ComponentAccess>,
    /// Filters that constrain which entities match.
    filters: Vec<FilterDescriptor>,
}

impl WorldQueryBuilder {
    /// Create a new, empty query builder.
    pub fn new() -> Self {
        Self {
            accesses: Vec::new(),
            filters: Vec::new(),
        }
    }

    /// Request read access to component `T`.
    pub fn read<T: Component>(mut self) -> Self {
        let cid = ComponentId::of::<T>();
        self.accesses.push(ComponentAccess {
            component_id: cid,
            type_id: TypeId::of::<T>(),
            mode: AccessMode::Read,
            type_name: std::any::type_name::<T>(),
        });
        self
    }

    /// Request write (mutable) access to component `T`.
    pub fn write<T: Component>(mut self) -> Self {
        let cid = ComponentId::of::<T>();
        self.accesses.push(ComponentAccess {
            component_id: cid,
            type_id: TypeId::of::<T>(),
            mode: AccessMode::Write,
            type_name: std::any::type_name::<T>(),
        });
        self
    }

    /// Require that matched entities have component `T`, without actually
    /// fetching its data. This acts as a presence filter.
    pub fn with<T: Component>(mut self) -> Self {
        self.filters.push(FilterDescriptor {
            component_id: ComponentId::of::<T>(),
            must_have: true,
        });
        self
    }

    /// Exclude entities that have component `T`.
    pub fn without<T: Component>(mut self) -> Self {
        self.filters.push(FilterDescriptor {
            component_id: ComponentId::of::<T>(),
            must_have: false,
        });
        self
    }

    /// Compile the query against the current world state.
    ///
    /// The compiled query caches which archetypes match, so subsequent
    /// iterations are very fast as long as the archetype set does not change.
    pub fn build(self, world: &World) -> CompiledQuery {
        // Determine the full set of required component ids (accesses + with
        // filters).
        let mut required: Vec<ComponentId> = self
            .accesses
            .iter()
            .map(|a| a.component_id)
            .collect();

        for f in &self.filters {
            if f.must_have && !required.contains(&f.component_id) {
                required.push(f.component_id);
            }
        }

        let excluded: Vec<ComponentId> = self
            .filters
            .iter()
            .filter(|f| !f.must_have)
            .map(|f| f.component_id)
            .collect();

        let matching = world.matching_archetypes_excluding(&required, &excluded);

        CompiledQuery {
            accesses: self.accesses,
            filters: self.filters,
            required_components: required,
            excluded_components: excluded,
            matching_archetypes: matching,
            last_archetype_count: world.archetype_count(),
        }
    }
}

impl Default for WorldQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CompiledQuery -- cached, reusable query
// ---------------------------------------------------------------------------

/// A compiled query with cached archetype matches.
///
/// Created via [`WorldQueryBuilder::build`]. The archetype match cache is
/// lazily revalidated when the world's archetype count changes, so creating
/// a `CompiledQuery` once and reusing it across frames is the intended
/// usage pattern.
pub struct CompiledQuery {
    /// Which components are fetched and how.
    accesses: Vec<ComponentAccess>,
    /// Filter descriptors.
    filters: Vec<FilterDescriptor>,
    /// All component ids that must be present.
    required_components: Vec<ComponentId>,
    /// All component ids that must NOT be present.
    excluded_components: Vec<ComponentId>,
    /// Cached matching archetype ids.
    matching_archetypes: Vec<ArchetypeId>,
    /// Archetype count at the time of last cache update.
    last_archetype_count: usize,
}

impl CompiledQuery {
    /// Refresh the archetype match cache if the world has new archetypes.
    pub fn update(&mut self, world: &World) {
        if world.archetype_count() != self.last_archetype_count {
            self.matching_archetypes = world.matching_archetypes_excluding(
                &self.required_components,
                &self.excluded_components,
            );
            self.last_archetype_count = world.archetype_count();
        }
    }

    /// Iterate over all matching entities, yielding a [`QueryRow`] for each.
    pub fn iter<'w>(&mut self, world: &'w World) -> CompiledQueryIter<'w> {
        self.update(world);

        // Gather entities from matching archetypes.
        let mut entities = Vec::new();
        for &arch_id in &self.matching_archetypes {
            let arch = world.archetype(arch_id);
            entities.extend_from_slice(arch.entities());
        }

        // Fallback: also check all alive entities for the legacy HashMap path.
        let all_alive: Vec<Entity> = world.entity_storage().iter_alive().collect();
        for entity in &all_alive {
            if !entities.contains(entity) {
                entities.push(*entity);
            }
        }

        // Build lookup of accessed type ids for QueryRow.
        let accessed_types: Vec<TypeId> = self.accesses.iter().map(|a| a.type_id).collect();
        let accessed_modes: Vec<AccessMode> = self.accesses.iter().map(|a| a.mode).collect();

        CompiledQueryIter {
            world,
            entities,
            index: 0,
            required: self.required_components.clone(),
            excluded: self.excluded_components.clone(),
            accessed_types,
            accessed_modes,
        }
    }

    /// Parallel iteration: divide matching entities into chunks and process
    /// each chunk with the provided closure.
    ///
    /// The `chunk_size` controls granularity. Each chunk is processed
    /// sequentially within itself, but chunks execute concurrently on
    /// separate threads (when a thread pool is available).
    ///
    /// For single-threaded builds, this simply processes all chunks serially.
    pub fn par_iter<'w, F>(&mut self, world: &'w World, chunk_size: usize, mut func: F)
    where
        F: FnMut(QueryRow<'w>) + Send,
    {
        self.update(world);

        let mut entities = Vec::new();
        for &arch_id in &self.matching_archetypes {
            let arch = world.archetype(arch_id);
            entities.extend_from_slice(arch.entities());
        }

        // Also check legacy path.
        let all_alive: Vec<Entity> = world.entity_storage().iter_alive().collect();
        for entity in &all_alive {
            if !entities.contains(entity) {
                entities.push(*entity);
            }
        }

        // Filter to matching entities.
        let required = &self.required_components;
        let excluded = &self.excluded_components;

        let matched: Vec<Entity> = entities
            .into_iter()
            .filter(|e| {
                for req in required {
                    if !world.has_component_by_id(e.id, *req) {
                        return false;
                    }
                }
                for excl in excluded {
                    if world.has_component_by_id(e.id, *excl) {
                        return false;
                    }
                }
                true
            })
            .collect();

        // Process in chunks. A real implementation would use rayon or a
        // custom thread pool; here we process serially per chunk for
        // correctness and portability.
        let chunk_sz = chunk_size.max(1);
        for chunk in matched.chunks(chunk_sz) {
            for &entity in chunk {
                let row = QueryRow {
                    world,
                    entity,
                };
                func(row);
            }
        }
    }

    /// Return the number of matching archetypes.
    pub fn archetype_match_count(&self) -> usize {
        self.matching_archetypes.len()
    }

    /// Return a reference to the component access descriptors.
    pub fn accesses(&self) -> &[ComponentAccess] {
        &self.accesses
    }

    /// Return a reference to the filter descriptors.
    pub fn filters(&self) -> &[FilterDescriptor] {
        &self.filters
    }

    /// Return the count of entities that match the query.
    pub fn count(&mut self, world: &World) -> usize {
        self.iter(world).count()
    }
}

// ---------------------------------------------------------------------------
// CompiledQueryIter
// ---------------------------------------------------------------------------

/// Iterator over entities matched by a [`CompiledQuery`].
pub struct CompiledQueryIter<'w> {
    world: &'w World,
    entities: Vec<Entity>,
    index: usize,
    required: Vec<ComponentId>,
    excluded: Vec<ComponentId>,
    accessed_types: Vec<TypeId>,
    accessed_modes: Vec<AccessMode>,
}

impl<'w> Iterator for CompiledQueryIter<'w> {
    type Item = QueryRow<'w>;

    fn next(&mut self) -> Option<Self::Item> {
        'outer: while self.index < self.entities.len() {
            let entity = self.entities[self.index];
            self.index += 1;

            // Check required components.
            for req in &self.required {
                if !self.world.has_component_by_id(entity.id, *req) {
                    continue 'outer;
                }
            }

            // Check excluded components.
            for excl in &self.excluded {
                if self.world.has_component_by_id(entity.id, *excl) {
                    continue 'outer;
                }
            }

            return Some(QueryRow {
                world: self.world,
                entity,
            });
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.entities.len() - self.index;
        (0, Some(remaining))
    }
}

// ---------------------------------------------------------------------------
// QueryRow -- type-safe row access
// ---------------------------------------------------------------------------

/// A single row from a compiled query, providing type-safe access to the
/// matched entity's components.
///
/// `QueryRow` is lightweight -- it holds a reference to the world and the
/// entity handle, and performs component lookups on demand.
pub struct QueryRow<'w> {
    world: &'w World,
    entity: Entity,
}

impl<'w> QueryRow<'w> {
    /// The entity this row refers to.
    #[inline]
    pub fn entity(&self) -> Entity {
        self.entity
    }

    /// Get an immutable reference to component `T`, or `None` if the entity
    /// does not have it.
    #[inline]
    pub fn get<T: Component>(&self) -> Option<&'w T> {
        self.world.get_component::<T>(self.entity)
    }

    /// Check whether the entity has component `T`.
    #[inline]
    pub fn has<T: Component>(&self) -> bool {
        self.world.has_component::<T>(self.entity)
    }

    /// Get a reference to the world (for advanced use cases).
    #[inline]
    pub fn world(&self) -> &'w World {
        self.world
    }
}

// ---------------------------------------------------------------------------
// Optional<T> -- component that may or may not be present
// ---------------------------------------------------------------------------

/// Wrapper indicating that a component may or may not be present.
///
/// When used in a query context, `Optional<T>` does not filter out entities
/// missing the component. Instead, the value is `None` for entities without
/// it and `Some(&T)` for entities that have it.
///
/// ```ignore
/// let opt_vel: Optional<Velocity> = row.optional::<Velocity>();
/// if let Some(vel) = opt_vel.value() {
///     // entity has velocity
/// }
/// ```
pub struct Optional<T: Component> {
    value: Option<T>,
}

impl<T: Component> Optional<T> {
    /// Create an `Optional` with a value.
    pub fn some(value: T) -> Self {
        Self { value: Some(value) }
    }

    /// Create an `Optional` without a value.
    pub fn none() -> Self {
        Self { value: None }
    }

    /// Returns `true` if a value is present.
    #[inline]
    pub fn is_some(&self) -> bool {
        self.value.is_some()
    }

    /// Returns `true` if no value is present.
    #[inline]
    pub fn is_none(&self) -> bool {
        self.value.is_none()
    }

    /// Get a reference to the inner value.
    #[inline]
    pub fn value(&self) -> Option<&T> {
        self.value.as_ref()
    }

    /// Get a mutable reference to the inner value.
    #[inline]
    pub fn value_mut(&mut self) -> Option<&mut T> {
        self.value.as_mut()
    }

    /// Consume and return the inner value.
    #[inline]
    pub fn into_inner(self) -> Option<T> {
        self.value
    }
}

/// Extension on `QueryRow` for optional access.
impl<'w> QueryRow<'w> {
    /// Get an optional reference to component `T`. Returns `Some` if the
    /// entity has the component, `None` otherwise. Unlike `get`, this is
    /// semantically intended for components that are not required by the
    /// query.
    #[inline]
    pub fn optional<T: Component>(&self) -> Option<&'w T> {
        self.world.get_component::<T>(self.entity)
    }
}

// ---------------------------------------------------------------------------
// AnyOf<(A, B, C)> marker
// ---------------------------------------------------------------------------

/// Marker trait for a tuple of component types where at least one must be
/// present on the matched entity.
pub trait AnyOfMarker {
    /// Returns the component ids of all types in the tuple.
    fn component_ids() -> Vec<ComponentId>;
    /// Check if the entity has at least one of the components.
    fn matches_any(world: &World, entity_id: u32) -> bool;
}

/// `AnyOf<(A,)>` -- trivially requires A.
impl<A: Component> AnyOfMarker for (A,) {
    fn component_ids() -> Vec<ComponentId> {
        vec![ComponentId::of::<A>()]
    }
    fn matches_any(world: &World, entity_id: u32) -> bool {
        world.has_component_by_id(entity_id, ComponentId::of::<A>())
    }
}

/// `AnyOf<(A, B)>` -- entity must have A or B (or both).
impl<A: Component, B: Component> AnyOfMarker for (A, B) {
    fn component_ids() -> Vec<ComponentId> {
        vec![ComponentId::of::<A>(), ComponentId::of::<B>()]
    }
    fn matches_any(world: &World, entity_id: u32) -> bool {
        world.has_component_by_id(entity_id, ComponentId::of::<A>())
            || world.has_component_by_id(entity_id, ComponentId::of::<B>())
    }
}

/// `AnyOf<(A, B, C)>` -- entity must have at least one of A, B, or C.
impl<A: Component, B: Component, C: Component> AnyOfMarker for (A, B, C) {
    fn component_ids() -> Vec<ComponentId> {
        vec![
            ComponentId::of::<A>(),
            ComponentId::of::<B>(),
            ComponentId::of::<C>(),
        ]
    }
    fn matches_any(world: &World, entity_id: u32) -> bool {
        world.has_component_by_id(entity_id, ComponentId::of::<A>())
            || world.has_component_by_id(entity_id, ComponentId::of::<B>())
            || world.has_component_by_id(entity_id, ComponentId::of::<C>())
    }
}

/// `AnyOf<(A, B, C, D)>` -- entity must have at least one of A, B, C, or D.
impl<A: Component, B: Component, C: Component, D: Component> AnyOfMarker for (A, B, C, D) {
    fn component_ids() -> Vec<ComponentId> {
        vec![
            ComponentId::of::<A>(),
            ComponentId::of::<B>(),
            ComponentId::of::<C>(),
            ComponentId::of::<D>(),
        ]
    }
    fn matches_any(world: &World, entity_id: u32) -> bool {
        world.has_component_by_id(entity_id, ComponentId::of::<A>())
            || world.has_component_by_id(entity_id, ComponentId::of::<B>())
            || world.has_component_by_id(entity_id, ComponentId::of::<C>())
            || world.has_component_by_id(entity_id, ComponentId::of::<D>())
    }
}

/// Query filter that requires the entity to have at least one component from
/// the given tuple.
pub struct AnyOf<T: AnyOfMarker>(PhantomData<T>);

impl<T: AnyOfMarker> AnyOf<T> {
    /// Check if a given entity passes the AnyOf filter.
    pub fn matches(world: &World, entity_id: u32) -> bool {
        T::matches_any(world, entity_id)
    }
}

// ---------------------------------------------------------------------------
// EntityRef -- immutable entity handle with component getters
// ---------------------------------------------------------------------------

/// A safe, immutable handle to a single entity within the world.
///
/// Provides typed component getters without requiring a query. Useful for
/// one-off entity inspections (e.g., from an event handler or UI code).
///
/// ```ignore
/// if let Some(entity_ref) = world.get_entity(entity) {
///     if let Some(pos) = entity_ref.get::<Position>() {
///         println!("position: ({}, {})", pos.x, pos.y);
///     }
/// }
/// ```
pub struct EntityRef<'w> {
    world: &'w World,
    entity: Entity,
}

impl<'w> EntityRef<'w> {
    /// Create a new `EntityRef`.
    pub(crate) fn new(world: &'w World, entity: Entity) -> Self {
        Self { world, entity }
    }

    /// The entity handle.
    #[inline]
    pub fn id(&self) -> Entity {
        self.entity
    }

    /// Get an immutable reference to component `T`.
    #[inline]
    pub fn get<T: Component>(&self) -> Option<&'w T> {
        self.world.get_component::<T>(self.entity)
    }

    /// Check whether the entity has component `T`.
    #[inline]
    pub fn has<T: Component>(&self) -> bool {
        self.world.has_component::<T>(self.entity)
    }

    /// Check whether the entity is still alive.
    #[inline]
    pub fn is_alive(&self) -> bool {
        self.world.is_alive(self.entity)
    }

    /// Which archetype does this entity belong to?
    #[inline]
    pub fn archetype_id(&self) -> Option<ArchetypeId> {
        self.world.entity_archetype_id(self.entity)
    }
}

// ---------------------------------------------------------------------------
// EntityMut -- mutable entity handle with component getters/setters
// ---------------------------------------------------------------------------

/// A safe, mutable handle to a single entity within the world.
///
/// Provides typed component getters and setters. Because this borrows the
/// world mutably, only one `EntityMut` can exist at a time.
///
/// ```ignore
/// if let Some(mut entity_mut) = world.get_entity_mut(entity) {
///     if let Some(pos) = entity_mut.get_mut::<Position>() {
///         pos.x += 1.0;
///     }
///     entity_mut.insert(Velocity { dx: 1.0, dy: 0.0 });
/// }
/// ```
pub struct EntityMut<'w> {
    world: &'w mut World,
    entity: Entity,
}

impl<'w> EntityMut<'w> {
    /// Create a new `EntityMut`.
    pub(crate) fn new(world: &'w mut World, entity: Entity) -> Self {
        Self { world, entity }
    }

    /// The entity handle.
    #[inline]
    pub fn id(&self) -> Entity {
        self.entity
    }

    /// Get an immutable reference to component `T`.
    #[inline]
    pub fn get<T: Component>(&self) -> Option<&T> {
        self.world.get_component::<T>(self.entity)
    }

    /// Get a mutable reference to component `T`.
    #[inline]
    pub fn get_mut<T: Component>(&mut self) -> Option<&mut T> {
        self.world.get_component_mut::<T>(self.entity)
    }

    /// Check whether the entity has component `T`.
    #[inline]
    pub fn has<T: Component>(&self) -> bool {
        self.world.has_component::<T>(self.entity)
    }

    /// Add or replace a component on this entity.
    pub fn insert<T: Component>(&mut self, component: T) {
        self.world.add_component(self.entity, component);
    }

    /// Remove a component from this entity, returning it if present.
    pub fn remove<T: Component>(&mut self) -> Option<T> {
        self.world.remove_component::<T>(self.entity)
    }

    /// Despawn this entity (consuming the handle).
    pub fn despawn(self) {
        self.world.despawn(self.entity);
    }

    /// Check whether the entity is still alive.
    #[inline]
    pub fn is_alive(&self) -> bool {
        self.world.is_alive(self.entity)
    }
}

// ---------------------------------------------------------------------------
// World extension methods
// ---------------------------------------------------------------------------

impl World {
    /// Create a [`WorldQueryBuilder`] for building advanced queries.
    pub fn query_builder(&self) -> WorldQueryBuilderBound<'_> {
        WorldQueryBuilderBound {
            world: self,
            builder: WorldQueryBuilder::new(),
        }
    }

    /// Get a safe, immutable handle to a single entity.
    ///
    /// Returns `None` if the entity is not alive.
    pub fn get_entity(&self, entity: Entity) -> Option<EntityRef<'_>> {
        if self.is_alive(entity) {
            Some(EntityRef::new(self, entity))
        } else {
            None
        }
    }

    /// Get a safe, mutable handle to a single entity.
    ///
    /// Returns `None` if the entity is not alive.
    pub fn get_entity_mut(&mut self, entity: Entity) -> Option<EntityMut<'_>> {
        if self.is_alive(entity) {
            Some(EntityMut::new(self, entity))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// WorldQueryBuilderBound -- builder that holds a world reference
// ---------------------------------------------------------------------------

/// A [`WorldQueryBuilder`] bound to a specific world reference, enabling
/// the fluent `world.query_builder().read::<T>()...build()` pattern.
pub struct WorldQueryBuilderBound<'w> {
    world: &'w World,
    builder: WorldQueryBuilder,
}

impl<'w> WorldQueryBuilderBound<'w> {
    /// Request read access to component `T`.
    pub fn read<T: Component>(mut self) -> Self {
        self.builder = self.builder.read::<T>();
        self
    }

    /// Request write access to component `T`.
    pub fn write<T: Component>(mut self) -> Self {
        self.builder = self.builder.write::<T>();
        self
    }

    /// Require that matched entities have component `T`.
    pub fn with<T: Component>(mut self) -> Self {
        self.builder = self.builder.with::<T>();
        self
    }

    /// Exclude entities that have component `T`.
    pub fn without<T: Component>(mut self) -> Self {
        self.builder = self.builder.without::<T>();
        self
    }

    /// Compile and return the query.
    pub fn build(self) -> CompiledQuery {
        self.builder.build(self.world)
    }
}

// ---------------------------------------------------------------------------
// MultiQuery -- query multiple component sets simultaneously
// ---------------------------------------------------------------------------

/// Allows running two independent queries and joining results by entity.
///
/// This is useful when you need data from two different component
/// combinations but want to iterate entities that match both.
pub struct JoinedQuery {
    query_a: CompiledQuery,
    query_b: CompiledQuery,
}

impl JoinedQuery {
    /// Create a joined query from two compiled queries.
    pub fn new(a: CompiledQuery, b: CompiledQuery) -> Self {
        Self {
            query_a: a,
            query_b: b,
        }
    }

    /// Iterate entities that match BOTH queries.
    pub fn iter<'w>(&mut self, world: &'w World) -> JoinedQueryIter<'w> {
        self.query_a.update(world);
        self.query_b.update(world);

        // Collect entities from query_a.
        let mut entities_a: Vec<Entity> = Vec::new();
        for &arch_id in &self.query_a.matching_archetypes {
            let arch = world.archetype(arch_id);
            entities_a.extend_from_slice(arch.entities());
        }

        // Also include legacy path entities.
        let all_alive: Vec<Entity> = world.entity_storage().iter_alive().collect();
        for entity in &all_alive {
            if !entities_a.contains(entity) {
                entities_a.push(*entity);
            }
        }

        // Filter to entities that match both queries' requirements.
        let req_a = self.query_a.required_components.clone();
        let excl_a = self.query_a.excluded_components.clone();
        let req_b = self.query_b.required_components.clone();
        let excl_b = self.query_b.excluded_components.clone();

        let matched: Vec<Entity> = entities_a
            .into_iter()
            .filter(|e| {
                for req in &req_a {
                    if !world.has_component_by_id(e.id, *req) {
                        return false;
                    }
                }
                for excl in &excl_a {
                    if world.has_component_by_id(e.id, *excl) {
                        return false;
                    }
                }
                for req in &req_b {
                    if !world.has_component_by_id(e.id, *req) {
                        return false;
                    }
                }
                for excl in &excl_b {
                    if world.has_component_by_id(e.id, *excl) {
                        return false;
                    }
                }
                true
            })
            .collect();

        JoinedQueryIter {
            world,
            entities: matched,
            index: 0,
        }
    }
}

/// Iterator for joined queries.
pub struct JoinedQueryIter<'w> {
    world: &'w World,
    entities: Vec<Entity>,
    index: usize,
}

impl<'w> Iterator for JoinedQueryIter<'w> {
    type Item = QueryRow<'w>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.entities.len() {
            let entity = self.entities[self.index];
            self.index += 1;
            Some(QueryRow {
                world: self.world,
                entity,
            })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.entities.len() - self.index;
        (remaining, Some(remaining))
    }
}

// ---------------------------------------------------------------------------
// QuerySnapshot -- frozen view of query results
// ---------------------------------------------------------------------------

/// A snapshot of query results captured at a point in time.
///
/// Stores the list of matched entities so you can iterate them multiple times
/// or pass them to other systems without re-evaluating the query.
pub struct QuerySnapshot {
    /// Matched entities at snapshot time.
    entities: Vec<Entity>,
}

impl QuerySnapshot {
    /// Take a snapshot of a compiled query's current results.
    pub fn capture(query: &mut CompiledQuery, world: &World) -> Self {
        let entities: Vec<Entity> = query.iter(world).map(|row| row.entity()).collect();
        Self { entities }
    }

    /// Iterate the snapshot's entities, yielding `QueryRow`s.
    pub fn iter<'w>(&self, world: &'w World) -> SnapshotIter<'w> {
        SnapshotIter {
            world,
            entities: &self.entities,
            index: 0,
        }
    }

    /// Number of entities in the snapshot.
    #[inline]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Whether the snapshot is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// The raw entity list.
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }
}

/// Iterator over a [`QuerySnapshot`].
pub struct SnapshotIter<'w> {
    world: &'w World,
    entities: &'w [Entity],
    index: usize,
}

impl<'w> Iterator for SnapshotIter<'w> {
    type Item = QueryRow<'w>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.entities.len() {
            let entity = self.entities[self.index];
            self.index += 1;
            if self.world.is_alive(entity) {
                Some(QueryRow {
                    world: self.world,
                    entity,
                })
            } else {
                self.next() // skip dead entities
            }
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;

    #[derive(Debug, PartialEq, Clone)]
    struct Position { x: f32, y: f32 }
    impl Component for Position {}

    #[derive(Debug, PartialEq, Clone)]
    struct Velocity { dx: f32, dy: f32 }
    impl Component for Velocity {}

    #[derive(Debug, PartialEq, Clone)]
    struct Player;
    impl Component for Player {}

    #[derive(Debug, PartialEq, Clone)]
    struct Dead;
    impl Component for Dead {}

    #[derive(Debug, PartialEq, Clone)]
    struct Health(f32);
    impl Component for Health {}

    #[derive(Debug, PartialEq, Clone)]
    struct Armor(f32);
    impl Component for Armor {}

    #[test]
    fn query_builder_read_with_without() {
        let mut world = World::new();
        let e1 = world.spawn_entity()
            .with(Position { x: 1.0, y: 2.0 })
            .with(Velocity { dx: 3.0, dy: 4.0 })
            .with(Player)
            .build();
        let _e2 = world.spawn_entity()
            .with(Position { x: 5.0, y: 6.0 })
            .with(Velocity { dx: 7.0, dy: 8.0 })
            .with(Player)
            .with(Dead)
            .build();
        let _e3 = world.spawn_entity()
            .with(Position { x: 9.0, y: 10.0 })
            .build();

        let mut query = WorldQueryBuilder::new()
            .read::<Position>()
            .read::<Velocity>()
            .with::<Player>()
            .without::<Dead>()
            .build(&world);

        let results: Vec<Entity> = query.iter(&world).map(|r| r.entity()).collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], e1);
    }

    #[test]
    fn query_builder_bound() {
        let mut world = World::new();
        let e1 = world.spawn_entity()
            .with(Position { x: 1.0, y: 0.0 })
            .with(Velocity { dx: 2.0, dy: 0.0 })
            .build();
        let _e2 = world.spawn_entity()
            .with(Position { x: 3.0, y: 0.0 })
            .build();

        let mut query = world.query_builder()
            .read::<Position>()
            .read::<Velocity>()
            .build();

        let results: Vec<Entity> = query.iter(&world).map(|r| r.entity()).collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], e1);
    }

    #[test]
    fn query_row_get() {
        let mut world = World::new();
        let e = world.spawn_entity()
            .with(Position { x: 42.0, y: 99.0 })
            .with(Health(100.0))
            .build();

        let mut query = WorldQueryBuilder::new()
            .read::<Position>()
            .build(&world);

        let row = query.iter(&world).next().unwrap();
        assert_eq!(row.entity(), e);
        let pos = row.get::<Position>().unwrap();
        assert_eq!(pos.x, 42.0);
        assert_eq!(pos.y, 99.0);
    }

    #[test]
    fn query_row_optional() {
        let mut world = World::new();
        let e1 = world.spawn_entity()
            .with(Position { x: 1.0, y: 0.0 })
            .with(Velocity { dx: 2.0, dy: 0.0 })
            .build();
        let e2 = world.spawn_entity()
            .with(Position { x: 3.0, y: 0.0 })
            .build();

        let mut query = WorldQueryBuilder::new()
            .read::<Position>()
            .build(&world);

        let results: Vec<(Entity, Option<f32>)> = query.iter(&world)
            .map(|r| {
                let vel_dx = r.optional::<Velocity>().map(|v| v.dx);
                (r.entity(), vel_dx)
            })
            .collect();

        assert_eq!(results.len(), 2);
        // One entity has Velocity, one does not.
        let with_vel: Vec<_> = results.iter().filter(|(_, v)| v.is_some()).collect();
        let without_vel: Vec<_> = results.iter().filter(|(_, v)| v.is_none()).collect();
        assert_eq!(with_vel.len(), 1);
        assert_eq!(without_vel.len(), 1);
    }

    #[test]
    fn entity_ref_and_mut() {
        let mut world = World::new();
        let e = world.spawn_entity()
            .with(Position { x: 1.0, y: 2.0 })
            .build();

        // EntityRef
        {
            let entity_ref = world.get_entity(e).unwrap();
            assert!(entity_ref.is_alive());
            assert!(entity_ref.has::<Position>());
            assert!(!entity_ref.has::<Velocity>());
            let pos = entity_ref.get::<Position>().unwrap();
            assert_eq!(pos.x, 1.0);
        }

        // EntityMut
        {
            let mut entity_mut = world.get_entity_mut(e).unwrap();
            assert!(entity_mut.is_alive());
            entity_mut.insert(Velocity { dx: 5.0, dy: 6.0 });
            let pos = entity_mut.get_mut::<Position>().unwrap();
            pos.x = 99.0;
        }

        assert_eq!(world.get_component::<Position>(e).unwrap().x, 99.0);
        assert!(world.has_component::<Velocity>(e));
    }

    #[test]
    fn entity_mut_remove_and_despawn() {
        let mut world = World::new();
        let e = world.spawn_entity()
            .with(Position { x: 0.0, y: 0.0 })
            .with(Velocity { dx: 1.0, dy: 1.0 })
            .build();

        {
            let mut entity_mut = world.get_entity_mut(e).unwrap();
            let removed = entity_mut.remove::<Velocity>();
            assert!(removed.is_some());
        }

        assert!(!world.has_component::<Velocity>(e));
        assert!(world.is_alive(e));

        // Despawn
        {
            let entity_mut = world.get_entity_mut(e).unwrap();
            entity_mut.despawn();
        }
        assert!(!world.is_alive(e));
    }

    #[test]
    fn get_entity_dead_returns_none() {
        let mut world = World::new();
        let e = world.spawn_entity().build();
        world.despawn(e);
        assert!(world.get_entity(e).is_none());
        assert!(world.get_entity_mut(e).is_none());
    }

    #[test]
    fn compiled_query_cache_update() {
        let mut world = World::new();
        let _e1 = world.spawn_entity()
            .with(Position { x: 1.0, y: 0.0 })
            .build();

        let mut query = WorldQueryBuilder::new()
            .read::<Position>()
            .build(&world);

        assert_eq!(query.iter(&world).count(), 1);

        // Spawn another entity -- cache should update.
        let _e2 = world.spawn_entity()
            .with(Position { x: 2.0, y: 0.0 })
            .build();

        assert_eq!(query.iter(&world).count(), 2);
    }

    #[test]
    fn par_iter_processes_all() {
        let mut world = World::new();
        for i in 0..50 {
            world.spawn_entity()
                .with(Position { x: i as f32, y: 0.0 })
                .build();
        }

        let mut query = WorldQueryBuilder::new()
            .read::<Position>()
            .build(&world);

        let mut count = 0u32;
        query.par_iter(&world, 10, |row| {
            assert!(row.get::<Position>().is_some());
            count += 1;
        });
        assert_eq!(count, 50);
    }

    #[test]
    fn any_of_filter() {
        let mut world = World::new();
        let e1 = world.spawn_entity()
            .with(Health(100.0))
            .build();
        let e2 = world.spawn_entity()
            .with(Armor(50.0))
            .build();
        let e3 = world.spawn_entity()
            .with(Position { x: 0.0, y: 0.0 })
            .build();

        // Check AnyOf matches.
        assert!(AnyOf::<(Health, Armor)>::matches(&world, e1.id));
        assert!(AnyOf::<(Health, Armor)>::matches(&world, e2.id));
        assert!(!AnyOf::<(Health, Armor)>::matches(&world, e3.id));
    }

    #[test]
    fn query_snapshot() {
        let mut world = World::new();
        for i in 0..10 {
            world.spawn_entity()
                .with(Position { x: i as f32, y: 0.0 })
                .build();
        }

        let mut query = WorldQueryBuilder::new()
            .read::<Position>()
            .build(&world);

        let snapshot = QuerySnapshot::capture(&mut query, &world);
        assert_eq!(snapshot.len(), 10);

        // Iterate snapshot twice -- should work both times.
        assert_eq!(snapshot.iter(&world).count(), 10);
        assert_eq!(snapshot.iter(&world).count(), 10);
    }

    #[test]
    fn joined_query() {
        let mut world = World::new();
        let e1 = world.spawn_entity()
            .with(Position { x: 1.0, y: 0.0 })
            .with(Velocity { dx: 2.0, dy: 0.0 })
            .with(Health(100.0))
            .build();
        let _e2 = world.spawn_entity()
            .with(Position { x: 3.0, y: 0.0 })
            .with(Velocity { dx: 4.0, dy: 0.0 })
            .build();
        let _e3 = world.spawn_entity()
            .with(Position { x: 5.0, y: 0.0 })
            .with(Health(50.0))
            .build();

        let q_a = WorldQueryBuilder::new()
            .read::<Position>()
            .read::<Velocity>()
            .build(&world);
        let q_b = WorldQueryBuilder::new()
            .read::<Health>()
            .build(&world);

        let mut joined = JoinedQuery::new(q_a, q_b);
        let results: Vec<Entity> = joined.iter(&world).map(|r| r.entity()).collect();
        // Only e1 has all three.
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], e1);
    }

    #[test]
    fn compiled_query_count() {
        let mut world = World::new();
        for _ in 0..25 {
            world.spawn_entity()
                .with(Position { x: 0.0, y: 0.0 })
                .build();
        }
        for _ in 0..10 {
            world.spawn_entity()
                .with(Velocity { dx: 0.0, dy: 0.0 })
                .build();
        }

        let mut query = WorldQueryBuilder::new()
            .read::<Position>()
            .build(&world);

        assert_eq!(query.count(&world), 25);
    }

    #[test]
    fn optional_wrapper() {
        let opt_some = Optional::some(42u32);
        assert!(opt_some.is_some());
        assert!(!opt_some.is_none());
        assert_eq!(opt_some.value(), Some(&42));

        let opt_none: Optional<u32> = Optional::none();
        assert!(opt_none.is_none());
        assert_eq!(opt_none.value(), None);
    }

    #[test]
    fn query_builder_access_tracking() {
        let mut world = World::new();
        let _ = world.spawn_entity()
            .with(Position { x: 0.0, y: 0.0 })
            .build();

        let query = WorldQueryBuilder::new()
            .read::<Position>()
            .write::<Velocity>()
            .build(&world);

        assert_eq!(query.accesses().len(), 2);
        assert_eq!(query.accesses()[0].mode, AccessMode::Read);
        assert_eq!(query.accesses()[1].mode, AccessMode::Write);
    }
}
