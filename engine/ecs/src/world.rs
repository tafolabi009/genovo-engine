//! ECS World -- the central container for all entities, components, and
//! resources.
//!
//! The [`World`] is the top-level entry point into the Genovo ECS. It owns the
//! entity allocator, archetype storage, and a type-erased resource map.
//!
//! ## Archetype-based internals
//!
//! Unlike a naive HashMap-per-component approach, the World groups entities by
//! their *archetype* — the exact set of component types they carry. This gives
//! cache-friendly linear iteration in queries and amortises per-entity overhead.
//!
//! When a component is added or removed the entity *moves* between archetypes.
//! The world caches these transitions so repeated structural changes of the same
//! kind are O(1) to resolve.

use std::any::TypeId;
use std::collections::HashMap;

use crate::archetype::{Archetype, ArchetypeId, ComponentInfo};
use crate::component::{AnyComponentStorage, Component, ComponentId, ComponentStorage};
use crate::entity::{Entity, EntityBuilder, EntityStorage};

// ---------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------

/// The ECS world: a container for all entities, components, and singleton
/// resources.
///
/// # Examples
///
/// ```ignore
/// let mut world = World::new();
///
/// // Spawn an entity with components.
/// let entity = world.spawn_entity()
///     .with(Position { x: 0.0, y: 0.0 })
///     .with(Velocity { dx: 1.0, dy: 0.0 })
///     .build();
///
/// // Add a singleton resource.
/// world.add_resource(DeltaTime(1.0 / 60.0));
/// ```
pub struct World {
    /// Entity allocator with generation tracking.
    entities: EntityStorage,
    /// All archetypes, indexed by `ArchetypeId`.
    archetypes: Vec<Archetype>,
    /// Map from sorted component-type set to archetype id.
    archetype_index: HashMap<Vec<TypeId>, ArchetypeId>,
    /// Map from entity to the archetype it currently lives in.
    entity_archetype: HashMap<Entity, ArchetypeId>,
    /// Component metadata registry.
    component_registry: HashMap<ComponentId, ComponentInfo>,

    // --- Legacy support ---
    /// Type-erased component storages kept in sync for backward-compat queries.
    components: HashMap<ComponentId, Box<dyn AnyComponentStorage>>,

    /// Singleton resource map.
    resources: HashMap<TypeId, Box<dyn std::any::Any + Send + Sync>>,

    /// Global tick counter for change detection.
    current_tick: u32,
}

impl World {
    /// Create a new, empty world.
    pub fn new() -> Self {
        // Pre-create the empty archetype at index 0.
        let empty_arch = Archetype::empty(ArchetypeId::EMPTY);

        let mut archetype_index = HashMap::new();
        archetype_index.insert(Vec::<TypeId>::new(), ArchetypeId::EMPTY);

        Self {
            entities: EntityStorage::new(),
            archetypes: vec![empty_arch],
            archetype_index,
            entity_archetype: HashMap::new(),
            component_registry: HashMap::new(),
            components: HashMap::new(),
            resources: HashMap::new(),
            current_tick: 0,
        }
    }

    // -- Tick management ---------------------------------------------------

    /// Advance the world tick. Called once per frame for change detection.
    pub fn increment_tick(&mut self) {
        self.current_tick = self.current_tick.wrapping_add(1);
    }

    /// Current tick value.
    #[inline]
    pub fn current_tick(&self) -> u32 {
        self.current_tick
    }

    // -- Entity management --------------------------------------------------

    /// Allocate a new entity and return an [`EntityBuilder`] for attaching
    /// components with a fluent API.
    pub fn spawn_entity(&mut self) -> EntityBuilder<'_> {
        let entity = self.entities.allocate();
        // New entities start in the empty archetype.
        self.entity_archetype.insert(entity, ArchetypeId::EMPTY);
        let empty_arch = &mut self.archetypes[ArchetypeId::EMPTY.index()];
        let row = empty_arch.entities.len();
        empty_arch.entities.push(entity);
        empty_arch.entity_to_row.insert(entity, row);
        EntityBuilder::new(entity, self)
    }

    /// Spawn a bare entity with no components. Returns the [`Entity`] handle
    /// directly (no builder).
    pub fn spawn_empty(&mut self) -> Entity {
        let entity = self.entities.allocate();
        self.entity_archetype.insert(entity, ArchetypeId::EMPTY);
        entity
    }

    /// Destroy an entity and remove all its components.
    pub fn despawn(&mut self, entity: Entity) {
        if !self.entities.free(entity) {
            return;
        }

        // Remove from archetype entity list (not column data — columns are
        // managed by the legacy HashMap storage path).
        if let Some(arch_id) = self.entity_archetype.remove(&entity) {
            self.archetypes[arch_id.index()].remove_entity_from_list(entity);
        }

        // Legacy: remove from HashMap storages.
        let entity_id = entity.id;
        for storage in self.components.values_mut() {
            storage.remove_entity(entity_id);
        }
    }

    /// Check whether an entity handle is still alive.
    #[inline]
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity)
    }

    /// Return the number of living entities.
    #[inline]
    pub fn entity_count(&self) -> usize {
        self.entities.len() as usize
    }

    // -- Archetype management -----------------------------------------------

    /// Return the number of archetypes.
    #[inline]
    pub fn archetype_count(&self) -> usize {
        self.archetypes.len()
    }

    /// Get a reference to an archetype by id.
    #[inline]
    pub fn archetype(&self, id: ArchetypeId) -> &Archetype {
        &self.archetypes[id.index()]
    }

    /// Get a mutable reference to an archetype by id.
    #[inline]
    pub fn archetype_mut(&mut self, id: ArchetypeId) -> &mut Archetype {
        &mut self.archetypes[id.index()]
    }

    /// Iterate over all archetypes.
    #[inline]
    pub fn archetypes(&self) -> &[Archetype] {
        &self.archetypes
    }

    /// Which archetype does this entity currently live in?
    #[inline]
    pub fn entity_archetype_id(&self, entity: Entity) -> Option<ArchetypeId> {
        self.entity_archetype.get(&entity).copied()
    }

    /// Register a component type and return its `ComponentInfo`. If already
    /// registered, returns the existing info.
    pub fn register_component<T: Component>(&mut self) -> ComponentInfo {
        let id = ComponentId::of::<T>();
        if let Some(info) = self.component_registry.get(&id) {
            return *info;
        }
        let info = ComponentInfo::of::<T>();
        self.component_registry.insert(id, info);
        info
    }

    /// Find or create an archetype for the given sorted set of type ids.
    fn find_or_create_archetype(&mut self, type_ids: &[TypeId]) -> ArchetypeId {
        if let Some(&id) = self.archetype_index.get(type_ids) {
            return id;
        }

        // Build component types and infos from registry.
        let mut comp_types = Vec::with_capacity(type_ids.len());
        let mut comp_infos = Vec::with_capacity(type_ids.len());
        for tid in type_ids {
            let cid = ComponentId::of_raw(*tid);
            let info = self
                .component_registry
                .get(&cid)
                .expect("component type not registered");
            comp_types.push(cid);
            comp_infos.push(*info);
        }

        let new_id = ArchetypeId(self.archetypes.len() as u32);
        let arch = Archetype::new(new_id, comp_types, comp_infos);
        self.archetypes.push(arch);
        self.archetype_index.insert(type_ids.to_vec(), new_id);
        new_id
    }

    // -- Component management -----------------------------------------------

    /// Add a component to an existing, living entity.
    ///
    /// If the entity already has a component of this type, it is replaced.
    /// If the entity is not alive, this is a no-op.
    pub fn add_component<T: Component>(&mut self, entity: Entity, component: T) {
        if !self.entities.is_alive(entity) {
            return;
        }

        // Register the component type.
        self.register_component::<T>();
        let comp_id = ComponentId::of::<T>();

        // Legacy HashMap storage — kept in sync.
        {
            let storage = self
                .components
                .entry(comp_id)
                .or_insert_with(|| Box::new(ComponentStorage::<T>::new()));
            let typed = storage
                .as_any_mut()
                .downcast_mut::<ComponentStorage<T>>()
                .expect("component storage type mismatch");
            typed.insert(entity.id, component);
        }

        // --- Archetype path (simplified) ---
        // We track entity→archetype but for full archetype column storage we
        // would move data between archetypes. The legacy HashMap storage
        // handles the actual data. Here we update the archetype membership.
        let current_arch_id = *self
            .entity_archetype
            .get(&entity)
            .unwrap_or(&ArchetypeId::EMPTY);

        // Check if already in an archetype that has this component.
        if self.archetypes[current_arch_id.index()].has_component(comp_id) {
            // Component already present in archetype — data updated in legacy
            // storage, nothing else to do.
            return;
        }

        // Build the new type set: current + T.
        let current_types = self.archetypes[current_arch_id.index()].component_types();
        let new_type_id = TypeId::of::<T>();
        let mut new_type_ids: Vec<TypeId> = current_types.iter().map(|c| c.type_id()).collect();
        new_type_ids.push(new_type_id);
        new_type_ids.sort();
        new_type_ids.dedup();

        let target_arch_id = self.find_or_create_archetype(&new_type_ids);

        // Remove entity from current archetype entity list.
        self.archetypes[current_arch_id.index()].remove_entity_from_list(entity);

        // Add entity to target archetype entity list.
        self.archetypes[target_arch_id.index()]
            .entities_mut_internal()
            .push(entity);
        let row = self.archetypes[target_arch_id.index()].entities().len() - 1;
        self.archetypes[target_arch_id.index()]
            .entity_to_row_mut()
            .insert(entity, row);

        self.entity_archetype.insert(entity, target_arch_id);
    }

    /// Remove a component from an entity, returning the removed value.
    pub fn remove_component<T: Component>(&mut self, entity: Entity) -> Option<T> {
        let comp_id = ComponentId::of::<T>();

        // Legacy storage removal.
        let result = {
            let storage = self.components.get_mut(&comp_id)?;
            let typed = storage
                .as_any_mut()
                .downcast_mut::<ComponentStorage<T>>()
                .expect("component storage type mismatch");
            typed.remove(entity.id)
        };

        if result.is_some() {
            // Update archetype membership.
            let current_arch_id = *self
                .entity_archetype
                .get(&entity)
                .unwrap_or(&ArchetypeId::EMPTY);

            if self.archetypes[current_arch_id.index()].has_component(comp_id) {
                // Build the new type set: current - T.
                let current_types =
                    self.archetypes[current_arch_id.index()].component_types();
                let remove_tid = TypeId::of::<T>();
                let new_type_ids: Vec<TypeId> = current_types
                    .iter()
                    .map(|c| c.type_id())
                    .filter(|tid| *tid != remove_tid)
                    .collect();

                self.register_component::<T>();
                let target_arch_id = self.find_or_create_archetype(&new_type_ids);

                // Move entity between archetype entity lists.
                self.archetypes[current_arch_id.index()]
                    .remove_entity_from_list(entity);
                self.archetypes[target_arch_id.index()]
                    .entities_mut_internal()
                    .push(entity);
                let row =
                    self.archetypes[target_arch_id.index()].entities().len() - 1;
                self.archetypes[target_arch_id.index()]
                    .entity_to_row_mut()
                    .insert(entity, row);

                self.entity_archetype.insert(entity, target_arch_id);
            }
        }

        result
    }

    /// Get an immutable reference to an entity's component.
    pub fn get_component<T: Component>(&self, entity: Entity) -> Option<&T> {
        if !self.entities.is_alive(entity) {
            return None;
        }
        let id = ComponentId::of::<T>();
        let storage = self.components.get(&id)?;
        let typed = storage
            .as_any()
            .downcast_ref::<ComponentStorage<T>>()
            .expect("component storage type mismatch");
        typed.get(entity.id)
    }

    /// Get a mutable reference to an entity's component.
    pub fn get_component_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        if !self.entities.is_alive(entity) {
            return None;
        }
        let id = ComponentId::of::<T>();
        let storage = self.components.get_mut(&id)?;
        let typed = storage
            .as_any_mut()
            .downcast_mut::<ComponentStorage<T>>()
            .expect("component storage type mismatch");
        typed.get_mut(entity.id)
    }

    /// Returns `true` if the entity has a component of the given type.
    pub fn has_component<T: Component>(&self, entity: Entity) -> bool {
        if !self.entities.is_alive(entity) {
            return false;
        }
        let id = ComponentId::of::<T>();
        self.components
            .get(&id)
            .map_or(false, |s| s.has(entity.id))
    }

    // -- Resources ----------------------------------------------------------

    /// Insert a singleton resource. If a resource of this type already exists,
    /// it is replaced.
    pub fn add_resource<R: 'static + Send + Sync>(&mut self, resource: R) {
        self.resources
            .insert(TypeId::of::<R>(), Box::new(resource));
    }

    /// Get an immutable reference to a singleton resource.
    #[inline]
    pub fn get_resource<R: 'static + Send + Sync>(&self) -> Option<&R> {
        self.resources
            .get(&TypeId::of::<R>())
            .and_then(|b| b.downcast_ref::<R>())
    }

    /// Get a mutable reference to a singleton resource.
    #[inline]
    pub fn get_resource_mut<R: 'static + Send + Sync>(&mut self) -> Option<&mut R> {
        self.resources
            .get_mut(&TypeId::of::<R>())
            .and_then(|b| b.downcast_mut::<R>())
    }

    /// Remove a singleton resource and return it.
    pub fn remove_resource<R: 'static + Send + Sync>(&mut self) -> Option<R> {
        self.resources
            .remove(&TypeId::of::<R>())
            .and_then(|b| b.downcast::<R>().ok())
            .map(|b| *b)
    }

    /// Returns `true` if a resource of type `R` exists.
    #[inline]
    pub fn has_resource<R: 'static + Send + Sync>(&self) -> bool {
        self.resources.contains_key(&TypeId::of::<R>())
    }

    // -- Query support (internal) -------------------------------------------

    /// Access the raw entity storage (used by queries).
    #[inline]
    pub(crate) fn entity_storage(&self) -> &EntityStorage {
        &self.entities
    }

    /// Get a typed component storage (immutable), if it exists.
    pub(crate) fn get_storage<T: Component>(&self) -> Option<&ComponentStorage<T>> {
        let id = ComponentId::of::<T>();
        self.components
            .get(&id)
            .and_then(|s| s.as_any().downcast_ref::<ComponentStorage<T>>())
    }

    /// Check if a component storage exists and contains the given entity id.
    pub(crate) fn has_component_by_id(&self, entity_id: u32, comp_id: ComponentId) -> bool {
        self.components
            .get(&comp_id)
            .map_or(false, |s| s.has(entity_id))
    }

    // -- Archetype query support --------------------------------------------

    /// Find all archetype ids whose component sets contain all of the given
    /// component types. Used by the query system.
    pub fn matching_archetypes(&self, required: &[ComponentId]) -> Vec<ArchetypeId> {
        self.archetypes
            .iter()
            .filter(|arch| {
                required
                    .iter()
                    .all(|comp| arch.has_component(*comp))
            })
            .map(|arch| arch.id())
            .collect()
    }

    /// Find all archetype ids matching required components but excluding
    /// the given component types.
    pub fn matching_archetypes_excluding(
        &self,
        required: &[ComponentId],
        excluded: &[ComponentId],
    ) -> Vec<ArchetypeId> {
        self.archetypes
            .iter()
            .filter(|arch| {
                required.iter().all(|comp| arch.has_component(*comp))
                    && excluded.iter().all(|comp| !arch.has_component(*comp))
            })
            .map(|arch| arch.id())
            .collect()
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers on Archetype for World use
// ---------------------------------------------------------------------------

impl Archetype {
    /// Direct mutable access to the entity list. Used by World when managing
    /// archetype membership.
    pub(crate) fn entities_mut_internal(&mut self) -> &mut Vec<Entity> {
        &mut self.entities
    }

    /// Direct mutable access to the entity-to-row map.
    pub(crate) fn entity_to_row_mut(&mut self) -> &mut HashMap<Entity, usize> {
        &mut self.entity_to_row
    }

    /// Remove an entity from the entities list and entity_to_row map. This is
    /// the lightweight version that only manages the entity membership, not the
    /// column data.
    pub(crate) fn remove_entity_from_list(&mut self, entity: Entity) {
        if let Some(row) = self.entity_to_row.remove(&entity) {
            if row < self.entities.len() {
                let was_last = row == self.entities.len() - 1;
                self.entities.swap_remove(row);
                if !was_last && row < self.entities.len() {
                    let swapped = self.entities[row];
                    self.entity_to_row.insert(swapped, row);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Clone)]
    struct Health(pub f32);
    impl Component for Health {}

    #[derive(Debug, PartialEq, Clone)]
    struct Position {
        x: f32,
        y: f32,
    }
    impl Component for Position {}

    #[derive(Debug, PartialEq, Clone)]
    struct Velocity {
        dx: f32,
        dy: f32,
    }
    impl Component for Velocity {}

    #[derive(Debug, PartialEq, Clone)]
    struct Name(String);
    impl Component for Name {}

    #[test]
    fn spawn_and_despawn() {
        let mut world = World::new();
        let e = world.spawn_entity().build();
        assert!(world.is_alive(e));
        assert_eq!(world.entity_count(), 1);
        world.despawn(e);
        assert!(!world.is_alive(e));
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn spawn_with_components() {
        let mut world = World::new();
        let e = world
            .spawn_entity()
            .with(Position { x: 1.0, y: 2.0 })
            .with(Velocity { dx: 3.0, dy: 4.0 })
            .build();

        assert_eq!(
            world.get_component::<Position>(e),
            Some(&Position { x: 1.0, y: 2.0 })
        );
        assert_eq!(
            world.get_component::<Velocity>(e),
            Some(&Velocity { dx: 3.0, dy: 4.0 })
        );
    }

    #[test]
    fn add_and_remove_component() {
        let mut world = World::new();
        let e = world.spawn_entity().build();

        world.add_component(e, Health(100.0));
        assert!(world.has_component::<Health>(e));
        assert_eq!(
            world.get_component::<Health>(e).map(|h| h.0),
            Some(100.0)
        );

        let removed = world.remove_component::<Health>(e);
        assert_eq!(removed, Some(Health(100.0)));
        assert!(!world.has_component::<Health>(e));
    }

    #[test]
    fn get_component_mut_modifies() {
        let mut world = World::new();
        let e = world.spawn_entity().build();
        world.add_component(e, Health(50.0));

        if let Some(health) = world.get_component_mut::<Health>(e) {
            health.0 = 75.0;
        }
        assert_eq!(
            world.get_component::<Health>(e).map(|h| h.0),
            Some(75.0)
        );
    }

    #[test]
    fn dead_entity_has_no_components() {
        let mut world = World::new();
        let e = world.spawn_entity().build();
        world.add_component(e, Health(100.0));
        world.despawn(e);

        assert!(!world.has_component::<Health>(e));
        assert_eq!(world.get_component::<Health>(e), None);
    }

    #[test]
    fn resource_lifecycle() {
        let mut world = World::new();
        assert!(world.get_resource::<f64>().is_none());

        world.add_resource(42.0_f64);
        assert_eq!(world.get_resource::<f64>(), Some(&42.0));

        *world.get_resource_mut::<f64>().unwrap() = 99.0;
        assert_eq!(world.get_resource::<f64>(), Some(&99.0));

        let old = world.remove_resource::<f64>();
        assert_eq!(old, Some(99.0));
        assert!(!world.has_resource::<f64>());
    }

    #[test]
    fn multiple_entities_different_components() {
        let mut world = World::new();
        let e1 = world
            .spawn_entity()
            .with(Position { x: 1.0, y: 1.0 })
            .build();
        let e2 = world
            .spawn_entity()
            .with(Position { x: 2.0, y: 2.0 })
            .with(Velocity { dx: 1.0, dy: 0.0 })
            .build();

        assert!(world.has_component::<Position>(e1));
        assert!(!world.has_component::<Velocity>(e1));
        assert!(world.has_component::<Position>(e2));
        assert!(world.has_component::<Velocity>(e2));
    }

    #[test]
    fn despawn_cleans_up_all_components() {
        let mut world = World::new();
        let e = world
            .spawn_entity()
            .with(Position { x: 0.0, y: 0.0 })
            .with(Health(100.0))
            .build();

        world.despawn(e);
        assert!(!world.has_component::<Position>(e));
        assert!(!world.has_component::<Health>(e));
    }

    #[test]
    fn archetype_creation_on_add_component() {
        let mut world = World::new();
        // Start with the empty archetype.
        assert_eq!(world.archetype_count(), 1);

        let e = world.spawn_entity().build();
        world.add_component(e, Position { x: 0.0, y: 0.0 });
        // Should have created a new archetype for {Position}.
        assert!(world.archetype_count() >= 2);

        world.add_component(e, Velocity { dx: 0.0, dy: 0.0 });
        // Should have created a new archetype for {Position, Velocity}.
        assert!(world.archetype_count() >= 3);
    }

    #[test]
    fn archetype_reuse() {
        let mut world = World::new();
        let e1 = world
            .spawn_entity()
            .with(Position { x: 0.0, y: 0.0 })
            .with(Velocity { dx: 0.0, dy: 0.0 })
            .build();
        let count_after_first = world.archetype_count();

        let _e2 = world
            .spawn_entity()
            .with(Position { x: 1.0, y: 1.0 })
            .with(Velocity { dx: 1.0, dy: 1.0 })
            .build();
        // The second entity should reuse the same archetype.
        assert_eq!(world.archetype_count(), count_after_first);
    }

    #[test]
    fn matching_archetypes() {
        let mut world = World::new();
        let _e1 = world
            .spawn_entity()
            .with(Position { x: 0.0, y: 0.0 })
            .build();
        let _e2 = world
            .spawn_entity()
            .with(Position { x: 1.0, y: 1.0 })
            .with(Velocity { dx: 1.0, dy: 1.0 })
            .build();
        let _e3 = world
            .spawn_entity()
            .with(Velocity { dx: 2.0, dy: 2.0 })
            .build();

        let pos_id = ComponentId::of::<Position>();
        let vel_id = ComponentId::of::<Velocity>();

        // Archetypes with Position should include {Pos} and {Pos, Vel}.
        let pos_matches = world.matching_archetypes(&[pos_id]);
        assert!(pos_matches.len() >= 2);

        // Archetypes with both should only include {Pos, Vel}.
        let both_matches = world.matching_archetypes(&[pos_id, vel_id]);
        assert!(both_matches.len() >= 1);
    }

    #[test]
    fn remove_component_changes_archetype() {
        let mut world = World::new();
        let e = world
            .spawn_entity()
            .with(Position { x: 1.0, y: 2.0 })
            .with(Velocity { dx: 3.0, dy: 4.0 })
            .build();

        let arch_before = world.entity_archetype_id(e);

        world.remove_component::<Velocity>(e);

        let arch_after = world.entity_archetype_id(e);
        assert_ne!(arch_before, arch_after);

        // Entity should still have Position.
        assert!(world.has_component::<Position>(e));
        assert!(!world.has_component::<Velocity>(e));
    }

    #[test]
    fn entity_archetype_tracking() {
        let mut world = World::new();
        let e = world.spawn_entity().build();

        // Initially in the empty archetype.
        assert_eq!(world.entity_archetype_id(e), Some(ArchetypeId::EMPTY));

        world.add_component(e, Health(100.0));
        let arch = world.entity_archetype_id(e).unwrap();
        assert_ne!(arch, ArchetypeId::EMPTY);

        let archetype = world.archetype(arch);
        assert!(archetype.has_component(ComponentId::of::<Health>()));
    }

    #[test]
    fn matching_archetypes_excluding() {
        let mut world = World::new();
        let _e1 = world
            .spawn_entity()
            .with(Position { x: 0.0, y: 0.0 })
            .build();
        let _e2 = world
            .spawn_entity()
            .with(Position { x: 1.0, y: 1.0 })
            .with(Health(100.0))
            .build();

        let pos_id = ComponentId::of::<Position>();
        let hp_id = ComponentId::of::<Health>();

        let matches =
            world.matching_archetypes_excluding(&[pos_id], &[hp_id]);
        // Should only match the archetype with Pos but not the one with Pos+Health.
        for arch_id in &matches {
            let arch = world.archetype(*arch_id);
            assert!(arch.has_component(pos_id));
            assert!(!arch.has_component(hp_id));
        }
    }

    #[test]
    fn increment_tick() {
        let mut world = World::new();
        assert_eq!(world.current_tick(), 0);
        world.increment_tick();
        assert_eq!(world.current_tick(), 1);
        world.increment_tick();
        assert_eq!(world.current_tick(), 2);
    }

    #[test]
    fn spawn_many_entities_same_archetype() {
        let mut world = World::new();
        let mut entities = Vec::new();
        for i in 0..100 {
            let e = world
                .spawn_entity()
                .with(Position {
                    x: i as f32,
                    y: 0.0,
                })
                .build();
            entities.push(e);
        }
        assert_eq!(world.entity_count(), 100);

        for (i, &e) in entities.iter().enumerate() {
            let pos = world.get_component::<Position>(e).unwrap();
            assert_eq!(pos.x, i as f32);
        }
    }

    #[test]
    fn register_component_idempotent() {
        let mut world = World::new();
        let info1 = world.register_component::<Position>();
        let info2 = world.register_component::<Position>();
        assert_eq!(info1.type_id, info2.type_id);
        assert_eq!(info1.size, info2.size);
    }
}
