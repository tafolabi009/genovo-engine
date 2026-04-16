//! Entity queries for the Genovo ECS.
//!
//! Queries provide a way to iterate over all entities that have a specific set
//! of components. The implementation leverages archetype-awareness: only
//! archetypes whose component sets are a superset of the query's requirements
//! are visited, and within each archetype entities are yielded linearly.
//!
//! # Usage
//!
//! ```ignore
//! // Iterate entities with both Position and Velocity:
//! for (entity, (pos, vel)) in world.query::<(&Position, &Velocity)>() {
//!     println!("{:?} is at ({}, {}) moving ({}, {})", entity, pos.x, pos.y, vel.dx, vel.dy);
//! }
//!
//! // Single component:
//! for (entity, health) in world.query::<&Health>() {
//!     println!("{:?} has {} HP", entity, health.0);
//! }
//! ```
//!
//! # Archetype-Optimized Iteration
//!
//! Unlike a naive approach that checks every alive entity, the query system
//! first filters archetypes by their component set. For each matching
//! archetype, all entities are guaranteed to have the required components, so
//! no per-entity checks are needed — just linear iteration through the HashMap
//! storage (with archetype-level filtering for correctness).
//!
//! The `QueryState` struct can cache the list of matching archetype ids to
//! avoid re-scanning every frame.

use crate::component::{Component, ComponentId};
use crate::entity::Entity;
use crate::world::World;

// ---------------------------------------------------------------------------
// QueryItem trait
// ---------------------------------------------------------------------------

/// Trait implemented by types that describe what to fetch from the world for
/// each matching entity. Implementations exist for `&T` (immutable component
/// reference) and tuples thereof.
///
/// # Safety
///
/// This trait is unsafe because implementations must uphold aliasing rules
/// when returning references into component storages.
pub unsafe trait QueryItem {
    /// The item type yielded for each matching entity.
    type Item<'w>;

    /// Returns `true` if the world contains all required components for the
    /// given entity id.
    fn matches(world: &World, entity_id: u32) -> bool;

    /// Fetch the item for a single entity from the world.
    ///
    /// # Safety
    ///
    /// Caller must ensure the entity is alive and `matches` returned `true`.
    unsafe fn fetch<'w>(world: &'w World, entity_id: u32) -> Self::Item<'w>;

    /// Return the component ids that this query item requires.
    fn component_ids() -> Vec<ComponentId>;
}

// ---------------------------------------------------------------------------
// QueryItem impl for &T
// ---------------------------------------------------------------------------

unsafe impl<T: Component> QueryItem for &T {
    type Item<'w> = &'w T;

    fn matches(world: &World, entity_id: u32) -> bool {
        world
            .get_storage::<T>()
            .map_or(false, |s| s.has(entity_id))
    }

    unsafe fn fetch<'w>(world: &'w World, entity_id: u32) -> Self::Item<'w> {
        world
            .get_storage::<T>()
            .and_then(|s| s.get(entity_id))
            .expect("QueryItem::fetch called without matching check")
    }

    fn component_ids() -> Vec<ComponentId> {
        vec![ComponentId::of::<T>()]
    }
}

// ---------------------------------------------------------------------------
// QueryItem impls for tuples (arity 1..12)
// ---------------------------------------------------------------------------

macro_rules! impl_query_item_tuple {
    ($($T:ident),+) => {
        unsafe impl<$($T: QueryItem),+> QueryItem for ($($T,)+) {
            type Item<'w> = ($($T::Item<'w>,)+);

            fn matches(world: &World, entity_id: u32) -> bool {
                $($T::matches(world, entity_id))&&+
            }

            unsafe fn fetch<'w>(world: &'w World, entity_id: u32) -> Self::Item<'w> {
                ($(unsafe { $T::fetch(world, entity_id) },)+)
            }

            fn component_ids() -> Vec<ComponentId> {
                let mut ids = Vec::new();
                $(ids.extend($T::component_ids());)+
                ids
            }
        }
    };
}

impl_query_item_tuple!(A);
impl_query_item_tuple!(A, B);
impl_query_item_tuple!(A, B, C);
impl_query_item_tuple!(A, B, C, D);
impl_query_item_tuple!(A, B, C, D, E);
impl_query_item_tuple!(A, B, C, D, E, F);
impl_query_item_tuple!(A, B, C, D, E, F, G);
impl_query_item_tuple!(A, B, C, D, E, F, G, H);
impl_query_item_tuple!(A, B, C, D, E, F, G, H, I);
impl_query_item_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_query_item_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_query_item_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);

// ---------------------------------------------------------------------------
// QueryState — cached archetype matching
// ---------------------------------------------------------------------------

/// Cached query state that remembers which archetypes match.
///
/// Archetype indices are re-validated when new archetypes are created (the
/// archetype count changes). This avoids re-scanning all archetypes every
/// frame in the common case where the archetype graph is stable.
///
/// ```ignore
/// let mut state = QueryState::new::<(&Position, &Velocity)>(&world);
/// // ... later each frame ...
/// for (entity, (pos, vel)) in state.iter(&world) {
///     pos.x += vel.dx * dt;
/// }
/// ```
pub struct QueryState {
    /// Component ids required by this query.
    required_components: Vec<ComponentId>,
    /// Cached list of matching archetype ids.
    matching_archetypes: Vec<crate::archetype::ArchetypeId>,
    /// The archetype count when we last computed the match list.
    last_archetype_count: usize,
}

impl QueryState {
    /// Create a new query state for query type `Q`.
    pub fn new<Q: QueryItem>(world: &World) -> Self {
        let required = Q::component_ids();
        let matching = world.matching_archetypes(&required);
        Self {
            required_components: required,
            matching_archetypes: matching,
            last_archetype_count: world.archetype_count(),
        }
    }

    /// Update the cached archetype list if new archetypes were added.
    pub fn update(&mut self, world: &World) {
        if world.archetype_count() != self.last_archetype_count {
            self.matching_archetypes =
                world.matching_archetypes(&self.required_components);
            self.last_archetype_count = world.archetype_count();
        }
    }

    /// Return the matching archetype ids.
    pub fn matching_archetypes(&self) -> &[crate::archetype::ArchetypeId] {
        &self.matching_archetypes
    }

    /// Iterate using this cached state.
    pub fn iter<'w, Q: QueryItem>(&mut self, world: &'w World) -> QueryStateIter<'w, Q> {
        self.update(world);
        // Collect entities from matching archetypes.
        let mut entities = Vec::new();
        for &arch_id in &self.matching_archetypes {
            let arch = world.archetype(arch_id);
            entities.extend_from_slice(arch.entities());
        }
        QueryStateIter {
            world,
            entities,
            index: 0,
            _marker: std::marker::PhantomData,
        }
    }
}

/// Iterator produced by `QueryState::iter`.
pub struct QueryStateIter<'w, Q: QueryItem> {
    world: &'w World,
    entities: Vec<Entity>,
    index: usize,
    _marker: std::marker::PhantomData<Q>,
}

impl<'w, Q: QueryItem> Iterator for QueryStateIter<'w, Q> {
    type Item = (Entity, Q::Item<'w>);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.entities.len() {
            let entity = self.entities[self.index];
            self.index += 1;

            if Q::matches(self.world, entity.id) {
                let item = unsafe { Q::fetch(self.world, entity.id) };
                return Some((entity, item));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.entities.len() - self.index;
        (0, Some(remaining))
    }
}

// ---------------------------------------------------------------------------
// QueryFilter — filter types for advanced queries
// ---------------------------------------------------------------------------

/// Trait for query filters that can exclude entities based on component
/// presence.
pub trait QueryFilter {
    /// Returns `true` if the entity passes this filter.
    fn matches(world: &World, entity_id: u32) -> bool;

    /// Return the component ids that must be present (for archetype matching).
    fn required_ids() -> Vec<ComponentId> {
        Vec::new()
    }

    /// Return the component ids that must NOT be present.
    fn excluded_ids() -> Vec<ComponentId> {
        Vec::new()
    }
}

/// Filter that matches entities that have component `T`.
pub struct With<T: Component>(std::marker::PhantomData<T>);

impl<T: Component> QueryFilter for With<T> {
    fn matches(world: &World, entity_id: u32) -> bool {
        world
            .get_storage::<T>()
            .map_or(false, |s| s.has(entity_id))
    }

    fn required_ids() -> Vec<ComponentId> {
        vec![ComponentId::of::<T>()]
    }
}

/// Filter that matches entities that do NOT have component `T`.
pub struct Without<T: Component>(std::marker::PhantomData<T>);

impl<T: Component> QueryFilter for Without<T> {
    fn matches(world: &World, entity_id: u32) -> bool {
        !world
            .get_storage::<T>()
            .map_or(false, |s| s.has(entity_id))
    }

    fn excluded_ids() -> Vec<ComponentId> {
        vec![ComponentId::of::<T>()]
    }
}

/// Filter that matches entities where component `T` exists (alias for With).
pub struct Has<T: Component>(std::marker::PhantomData<T>);

impl<T: Component> QueryFilter for Has<T> {
    fn matches(world: &World, entity_id: u32) -> bool {
        world
            .get_storage::<T>()
            .map_or(false, |s| s.has(entity_id))
    }

    fn required_ids() -> Vec<ComponentId> {
        vec![ComponentId::of::<T>()]
    }
}

/// Combine two filters with AND.
pub struct And<A: QueryFilter, B: QueryFilter>(
    std::marker::PhantomData<(A, B)>,
);

impl<A: QueryFilter, B: QueryFilter> QueryFilter for And<A, B> {
    fn matches(world: &World, entity_id: u32) -> bool {
        A::matches(world, entity_id) && B::matches(world, entity_id)
    }

    fn required_ids() -> Vec<ComponentId> {
        let mut ids = A::required_ids();
        ids.extend(B::required_ids());
        ids
    }

    fn excluded_ids() -> Vec<ComponentId> {
        let mut ids = A::excluded_ids();
        ids.extend(B::excluded_ids());
        ids
    }
}

/// Combine two filters with OR.
pub struct Or<A: QueryFilter, B: QueryFilter>(
    std::marker::PhantomData<(A, B)>,
);

impl<A: QueryFilter, B: QueryFilter> QueryFilter for Or<A, B> {
    fn matches(world: &World, entity_id: u32) -> bool {
        A::matches(world, entity_id) || B::matches(world, entity_id)
    }
}

// ---------------------------------------------------------------------------
// World::query and World::query_filtered
// ---------------------------------------------------------------------------

impl World {
    /// Iterate over all entities that match the given query type, yielding
    /// `(Entity, Q::Item)` for each match.
    ///
    /// ```ignore
    /// for (entity, (pos, vel)) in world.query::<(&Position, &Velocity)>() {
    ///     // ...
    /// }
    /// ```
    pub fn query<Q: QueryItem>(&self) -> QueryIter<'_, Q> {
        // Use archetype-aware matching: only iterate entities in matching
        // archetypes.
        let required = Q::component_ids();
        let matching = self.matching_archetypes(&required);

        let mut alive = Vec::new();
        for arch_id in &matching {
            let arch = self.archetype(*arch_id);
            alive.extend_from_slice(arch.entities());
        }

        // Fallback: also check entities we may have missed (entities in the
        // empty archetype that gained components via legacy path).
        // For correctness, gather all alive entities and filter.
        let all_alive: Vec<Entity> = self.entity_storage().iter_alive().collect();
        for entity in &all_alive {
            if !alive.contains(entity) {
                alive.push(*entity);
            }
        }

        QueryIter {
            world: self,
            alive,
            index: 0,
            _marker: std::marker::PhantomData,
        }
    }

    /// Query with an additional filter.
    ///
    /// ```ignore
    /// for (entity, pos) in world.query_filtered::<&Position, Without<Hidden>>() {
    ///     // Only entities with Position but without Hidden
    /// }
    /// ```
    pub fn query_filtered<Q: QueryItem, F: QueryFilter>(
        &self,
    ) -> FilteredQueryIter<'_, Q, F> {
        let alive: Vec<Entity> = self.entity_storage().iter_alive().collect();
        FilteredQueryIter {
            world: self,
            alive,
            index: 0,
            _marker: std::marker::PhantomData,
        }
    }

    /// Query with a single component and return a count of matches, without
    /// allocating an iterator for each entity.
    pub fn query_count<Q: QueryItem>(&self) -> usize {
        self.query::<Q>().count()
    }

    /// Check if any entity matches the query.
    pub fn query_any<Q: QueryItem>(&self) -> bool {
        self.query::<Q>().next().is_some()
    }
}

// ---------------------------------------------------------------------------
// QueryIter
// ---------------------------------------------------------------------------

/// Iterator over entities matching a query.
///
/// Walks entities from matching archetypes and yields `(Entity, Q::Item)` for
/// each entity that has all required components.
pub struct QueryIter<'w, Q: QueryItem> {
    world: &'w World,
    alive: Vec<Entity>,
    index: usize,
    _marker: std::marker::PhantomData<Q>,
}

impl<'w, Q: QueryItem> Iterator for QueryIter<'w, Q> {
    type Item = (Entity, Q::Item<'w>);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.alive.len() {
            let entity = self.alive[self.index];
            self.index += 1;

            if Q::matches(self.world, entity.id) {
                // SAFETY: We just checked that all components exist via `matches`.
                let item = unsafe { Q::fetch(self.world, entity.id) };
                return Some((entity, item));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.alive.len() - self.index;
        (0, Some(remaining))
    }
}

// ---------------------------------------------------------------------------
// FilteredQueryIter
// ---------------------------------------------------------------------------

/// Iterator over entities matching a query with an additional filter.
pub struct FilteredQueryIter<'w, Q: QueryItem, F: QueryFilter> {
    world: &'w World,
    alive: Vec<Entity>,
    index: usize,
    _marker: std::marker::PhantomData<(Q, F)>,
}

impl<'w, Q: QueryItem, F: QueryFilter> Iterator for FilteredQueryIter<'w, Q, F> {
    type Item = (Entity, Q::Item<'w>);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.alive.len() {
            let entity = self.alive[self.index];
            self.index += 1;

            if Q::matches(self.world, entity.id) && F::matches(self.world, entity.id) {
                let item = unsafe { Q::fetch(self.world, entity.id) };
                return Some((entity, item));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.alive.len() - self.index;
        (0, Some(remaining))
    }
}

// ---------------------------------------------------------------------------
// QueryBuilder — fluent query construction
// ---------------------------------------------------------------------------

/// A builder for constructing complex queries with multiple filters.
///
/// ```ignore
/// let results: Vec<(Entity, &Position)> = QueryBuilder::new::<&Position>()
///     .with_filter::<With<Velocity>>()
///     .build(&world)
///     .collect();
/// ```
pub struct QueryBuilder<Q: QueryItem> {
    _marker: std::marker::PhantomData<Q>,
    required: Vec<ComponentId>,
    excluded: Vec<ComponentId>,
}

impl<Q: QueryItem> QueryBuilder<Q> {
    /// Create a new query builder.
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
            required: Q::component_ids(),
            excluded: Vec::new(),
        }
    }

    /// Add a required component filter.
    pub fn with_filter<F: QueryFilter>(mut self) -> Self {
        self.required.extend(F::required_ids());
        self.excluded.extend(F::excluded_ids());
        self
    }

    /// Exclude entities with a specific component.
    pub fn without<T: Component>(mut self) -> Self {
        self.excluded.push(ComponentId::of::<T>());
        self
    }

    /// Build and execute the query against a world.
    pub fn build<'w>(self, world: &'w World) -> QueryBuilderIter<'w, Q> {
        let matching = world.matching_archetypes_excluding(
            &self.required,
            &self.excluded,
        );

        let mut entities = Vec::new();
        for arch_id in &matching {
            let arch = world.archetype(*arch_id);
            entities.extend_from_slice(arch.entities());
        }

        // Fallback: also consider all alive entities.
        let all_alive: Vec<Entity> = world.entity_storage().iter_alive().collect();
        for entity in &all_alive {
            if !entities.contains(entity) {
                entities.push(*entity);
            }
        }

        QueryBuilderIter {
            world,
            entities,
            index: 0,
            required: self.required,
            excluded: self.excluded,
            _marker: std::marker::PhantomData,
        }
    }
}

/// Iterator produced by [`QueryBuilder::build`].
pub struct QueryBuilderIter<'w, Q: QueryItem> {
    world: &'w World,
    entities: Vec<Entity>,
    index: usize,
    required: Vec<ComponentId>,
    excluded: Vec<ComponentId>,
    _marker: std::marker::PhantomData<Q>,
}

impl<'w, Q: QueryItem> Iterator for QueryBuilderIter<'w, Q> {
    type Item = (Entity, Q::Item<'w>);

    fn next(&mut self) -> Option<Self::Item> {
        'outer: while self.index < self.entities.len() {
            let entity = self.entities[self.index];
            self.index += 1;

            if !Q::matches(self.world, entity.id) {
                continue;
            }

            // Check required components (from filters).
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

            let item = unsafe { Q::fetch(self.world, entity.id) };
            return Some((entity, item));
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::component::Component;
    use crate::world::World;
    use super::*;

    #[derive(Debug, PartialEq, Clone)]
    struct Pos {
        x: f32,
        y: f32,
    }
    impl Component for Pos {}

    #[derive(Debug, PartialEq, Clone)]
    struct Vel {
        dx: f32,
        dy: f32,
    }
    impl Component for Vel {}

    #[derive(Debug, PartialEq, Clone)]
    struct Name(String);
    impl Component for Name {}

    #[derive(Debug, PartialEq, Clone)]
    struct Hidden;
    impl Component for Hidden {}

    #[test]
    fn query_single_component() {
        let mut world = World::new();
        let e1 = world.spawn_entity().with(Pos { x: 1.0, y: 2.0 }).build();
        let e2 = world.spawn_entity().with(Pos { x: 3.0, y: 4.0 }).build();
        let _e3 = world.spawn_entity().build(); // no Pos

        let mut results: Vec<(u32, f32)> = world
            .query::<&Pos>()
            .map(|(e, pos)| (e.id, pos.x))
            .collect();
        results.sort_by_key(|(id, _)| *id);

        assert_eq!(results, vec![(e1.id, 1.0), (e2.id, 3.0)]);
    }

    #[test]
    fn query_two_components() {
        let mut world = World::new();
        let e1 = world
            .spawn_entity()
            .with(Pos { x: 1.0, y: 0.0 })
            .with(Vel { dx: 10.0, dy: 0.0 })
            .build();
        let _e2 = world
            .spawn_entity()
            .with(Pos { x: 2.0, y: 0.0 })
            .build(); // no Vel
        let _e3 = world
            .spawn_entity()
            .with(Vel { dx: 99.0, dy: 0.0 })
            .build(); // no Pos

        let results: Vec<(u32, f32, f32)> = world
            .query::<(&Pos, &Vel)>()
            .map(|(e, (pos, vel))| (e.id, pos.x, vel.dx))
            .collect();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (e1.id, 1.0, 10.0));
    }

    #[test]
    fn query_three_components() {
        let mut world = World::new();
        let e1 = world
            .spawn_entity()
            .with(Pos { x: 1.0, y: 0.0 })
            .with(Vel { dx: 2.0, dy: 0.0 })
            .with(Name("player".into()))
            .build();
        let _e2 = world
            .spawn_entity()
            .with(Pos { x: 3.0, y: 0.0 })
            .with(Vel { dx: 4.0, dy: 0.0 })
            .build(); // no Name

        let results: Vec<(u32, &str)> = world
            .query::<(&Pos, &Vel, &Name)>()
            .map(|(e, (_pos, _vel, name))| (e.id, name.0.as_str()))
            .collect();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (e1.id, "player"));
    }

    #[test]
    fn query_empty_world() {
        let world = World::new();
        let count = world.query::<&Pos>().count();
        assert_eq!(count, 0);
    }

    #[test]
    fn query_after_despawn() {
        let mut world = World::new();
        let e1 = world.spawn_entity().with(Pos { x: 1.0, y: 0.0 }).build();
        let e2 = world.spawn_entity().with(Pos { x: 2.0, y: 0.0 }).build();

        world.despawn(e1);

        let results: Vec<u32> = world.query::<&Pos>().map(|(e, _)| e.id).collect();
        assert_eq!(results, vec![e2.id]);
    }

    #[test]
    fn query_after_remove_component() {
        let mut world = World::new();
        let e1 = world
            .spawn_entity()
            .with(Pos { x: 1.0, y: 0.0 })
            .with(Vel { dx: 10.0, dy: 0.0 })
            .build();

        // Remove Vel: entity should no longer match (&Pos, &Vel).
        world.remove_component::<Vel>(e1);

        let count = world.query::<(&Pos, &Vel)>().count();
        assert_eq!(count, 0);

        // But it should still match &Pos alone.
        let count = world.query::<&Pos>().count();
        assert_eq!(count, 1);
    }

    #[test]
    fn query_state_caching() {
        let mut world = World::new();
        let _e1 = world
            .spawn_entity()
            .with(Pos { x: 1.0, y: 0.0 })
            .build();

        let mut state = QueryState::new::<&Pos>(&world);
        let count = state.iter::<&Pos>(&world).count();
        assert_eq!(count, 1);

        // Add another entity — state should update.
        let _e2 = world
            .spawn_entity()
            .with(Pos { x: 2.0, y: 0.0 })
            .build();

        let count = state.iter::<&Pos>(&world).count();
        assert_eq!(count, 2);
    }

    #[test]
    fn query_filtered_without() {
        let mut world = World::new();
        let e1 = world
            .spawn_entity()
            .with(Pos { x: 1.0, y: 0.0 })
            .build();
        let _e2 = world
            .spawn_entity()
            .with(Pos { x: 2.0, y: 0.0 })
            .with(Hidden)
            .build();

        let results: Vec<u32> = world
            .query_filtered::<&Pos, Without<Hidden>>()
            .map(|(e, _)| e.id)
            .collect();

        assert_eq!(results, vec![e1.id]);
    }

    #[test]
    fn query_count_and_any() {
        let mut world = World::new();
        assert_eq!(world.query_count::<&Pos>(), 0);
        assert!(!world.query_any::<&Pos>());

        let _e = world
            .spawn_entity()
            .with(Pos { x: 0.0, y: 0.0 })
            .build();
        assert_eq!(world.query_count::<&Pos>(), 1);
        assert!(world.query_any::<&Pos>());
    }

    #[test]
    fn query_builder_basic() {
        let mut world = World::new();
        let e1 = world
            .spawn_entity()
            .with(Pos { x: 1.0, y: 0.0 })
            .with(Vel { dx: 1.0, dy: 0.0 })
            .build();
        let _e2 = world
            .spawn_entity()
            .with(Pos { x: 2.0, y: 0.0 })
            .build();

        let results: Vec<u32> = QueryBuilder::<&Pos>::new()
            .with_filter::<With<Vel>>()
            .build(&world)
            .map(|(e, _)| e.id)
            .collect();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], e1.id);
    }

    #[test]
    fn query_builder_without() {
        let mut world = World::new();
        let e1 = world
            .spawn_entity()
            .with(Pos { x: 1.0, y: 0.0 })
            .build();
        let _e2 = world
            .spawn_entity()
            .with(Pos { x: 2.0, y: 0.0 })
            .with(Hidden)
            .build();

        let results: Vec<u32> = QueryBuilder::<&Pos>::new()
            .without::<Hidden>()
            .build(&world)
            .map(|(e, _)| e.id)
            .collect();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], e1.id);
    }

    #[test]
    fn query_four_components() {
        let mut world = World::new();
        let e = world
            .spawn_entity()
            .with(Pos { x: 1.0, y: 2.0 })
            .with(Vel { dx: 3.0, dy: 4.0 })
            .with(Name("test".into()))
            .with(Hidden)
            .build();

        let count = world.query::<(&Pos, &Vel, &Name, &Hidden)>().count();
        assert_eq!(count, 1);
    }

    #[test]
    fn query_many_entities() {
        let mut world = World::new();
        for i in 0..500 {
            world
                .spawn_entity()
                .with(Pos {
                    x: i as f32,
                    y: 0.0,
                })
                .build();
        }

        let count = world.query::<&Pos>().count();
        assert_eq!(count, 500);
    }

    #[test]
    fn component_ids_single() {
        let ids = <&Pos as QueryItem>::component_ids();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], ComponentId::of::<Pos>());
    }

    #[test]
    fn component_ids_tuple() {
        let ids = <(&Pos, &Vel) as QueryItem>::component_ids();
        assert_eq!(ids.len(), 2);
    }
}
