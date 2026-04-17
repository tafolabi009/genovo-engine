//! Entity references: EntityRef (immutable entity accessor), EntityMut (mutable),
//! EntityWorldMut (world + entity), and component iteration per entity.
//!
//! This module provides safe, ergonomic wrappers for accessing an entity's
//! components without raw pointer manipulation:
//!
//! - **EntityRefV2** — immutable view of an entity: read components, check
//!   component presence, iterate component types.
//! - **EntityMutV2** — mutable view: read/write components in-place.
//! - **EntityWorldMutV2** — full mutable access to the entity *and* the world:
//!   can add/remove components, despawn the entity, etc.
//! - **Component iteration** — iterate over all component data attached to
//!   a single entity.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Entity handle (re-use from world_v2 or define locally for self-containment)
// ---------------------------------------------------------------------------

/// Lightweight entity handle.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityHandle {
    pub index: u32,
    pub generation: u32,
}

impl EntityHandle {
    pub const INVALID: Self = Self {
        index: u32::MAX,
        generation: 0,
    };

    pub fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    pub fn is_valid(&self) -> bool {
        self.index != u32::MAX
    }
}

impl fmt::Debug for EntityHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Entity({}v{})", self.index, self.generation)
    }
}

impl fmt::Display for EntityHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}v{}", self.index, self.generation)
    }
}

// ---------------------------------------------------------------------------
// Component storage trait
// ---------------------------------------------------------------------------

/// Trait for type-erased component containers.
pub trait AnyComponentStore: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn has(&self, entity: EntityHandle) -> bool;
    fn remove(&mut self, entity: EntityHandle) -> bool;
    fn type_id(&self) -> TypeId;
    fn type_name(&self) -> &'static str;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}

/// A simple HashMap-based component store for a single component type.
pub struct ComponentStore<T: Send + Sync + 'static> {
    data: HashMap<EntityHandle, T>,
}

impl<T: Send + Sync + 'static> ComponentStore<T> {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    pub fn insert(&mut self, entity: EntityHandle, value: T) -> Option<T> {
        self.data.insert(entity, value)
    }

    pub fn get(&self, entity: EntityHandle) -> Option<&T> {
        self.data.get(&entity)
    }

    pub fn get_mut(&mut self, entity: EntityHandle) -> Option<&mut T> {
        self.data.get_mut(&entity)
    }

    pub fn remove_typed(&mut self, entity: EntityHandle) -> Option<T> {
        self.data.remove(&entity)
    }

    pub fn contains(&self, entity: EntityHandle) -> bool {
        self.data.contains_key(&entity)
    }

    pub fn iter(&self) -> impl Iterator<Item = (EntityHandle, &T)> {
        self.data.iter().map(|(&e, v)| (e, v))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (EntityHandle, &mut T)> {
        self.data.iter_mut().map(|(&e, v)| (e, v))
    }
}

impl<T: Send + Sync + 'static> Default for ComponentStore<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Send + Sync + 'static> AnyComponentStore for ComponentStore<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn has(&self, entity: EntityHandle) -> bool {
        self.data.contains_key(&entity)
    }

    fn remove(&mut self, entity: EntityHandle) -> bool {
        self.data.remove(&entity).is_some()
    }

    fn type_id(&self) -> TypeId {
        TypeId::of::<T>()
    }

    fn type_name(&self) -> &'static str {
        std::any::type_name::<T>()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Simple world for entity reference operations
// ---------------------------------------------------------------------------

/// A simplified world that owns component stores, used to demonstrate
/// entity reference patterns.
pub struct SimpleWorld {
    /// Entity generation tracker.
    generations: Vec<u32>,
    alive: Vec<bool>,
    free_list: Vec<u32>,
    next_fresh: u32,
    /// Component stores keyed by TypeId.
    stores: HashMap<TypeId, Box<dyn AnyComponentStore>>,
}

impl SimpleWorld {
    /// Create a new empty world.
    pub fn new() -> Self {
        Self {
            generations: Vec::new(),
            alive: Vec::new(),
            free_list: Vec::new(),
            next_fresh: 0,
            stores: HashMap::new(),
        }
    }

    /// Spawn a new entity.
    pub fn spawn(&mut self) -> EntityHandle {
        let index = if let Some(recycled) = self.free_list.pop() {
            recycled
        } else {
            let idx = self.next_fresh;
            self.next_fresh += 1;
            self.generations.push(0);
            self.alive.push(false);
            idx
        };

        self.alive[index as usize] = true;
        EntityHandle::new(index, self.generations[index as usize])
    }

    /// Despawn an entity.
    pub fn despawn(&mut self, entity: EntityHandle) -> bool {
        let idx = entity.index as usize;
        if idx >= self.alive.len()
            || !self.alive[idx]
            || self.generations[idx] != entity.generation
        {
            return false;
        }

        // Remove from all stores.
        for store in self.stores.values_mut() {
            store.remove(entity);
        }

        self.alive[idx] = false;
        self.generations[idx] = self.generations[idx].wrapping_add(1);
        self.free_list.push(entity.index);
        true
    }

    /// Check if an entity is alive.
    pub fn is_alive(&self, entity: EntityHandle) -> bool {
        let idx = entity.index as usize;
        idx < self.alive.len()
            && self.alive[idx]
            && self.generations[idx] == entity.generation
    }

    /// Ensure a component store exists for type T.
    fn ensure_store<T: Send + Sync + 'static>(&mut self) {
        self.stores
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(ComponentStore::<T>::new()));
    }

    /// Add a component to an entity.
    pub fn add_component<T: Send + Sync + 'static>(
        &mut self,
        entity: EntityHandle,
        component: T,
    ) {
        if !self.is_alive(entity) {
            return;
        }
        self.ensure_store::<T>();
        let store = self
            .stores
            .get_mut(&TypeId::of::<T>())
            .unwrap()
            .as_any_mut()
            .downcast_mut::<ComponentStore<T>>()
            .unwrap();
        store.insert(entity, component);
    }

    /// Remove a component from an entity.
    pub fn remove_component<T: Send + Sync + 'static>(
        &mut self,
        entity: EntityHandle,
    ) -> Option<T> {
        let store = self
            .stores
            .get_mut(&TypeId::of::<T>())?
            .as_any_mut()
            .downcast_mut::<ComponentStore<T>>()?;
        store.remove_typed(entity)
    }

    /// Get a component reference.
    pub fn get_component<T: Send + Sync + 'static>(
        &self,
        entity: EntityHandle,
    ) -> Option<&T> {
        let store = self
            .stores
            .get(&TypeId::of::<T>())?
            .as_any()
            .downcast_ref::<ComponentStore<T>>()?;
        store.get(entity)
    }

    /// Get a mutable component reference.
    pub fn get_component_mut<T: Send + Sync + 'static>(
        &mut self,
        entity: EntityHandle,
    ) -> Option<&mut T> {
        let store = self
            .stores
            .get_mut(&TypeId::of::<T>())?
            .as_any_mut()
            .downcast_mut::<ComponentStore<T>>()?;
        store.get_mut(entity)
    }

    /// Check if an entity has a component.
    pub fn has_component<T: 'static>(&self, entity: EntityHandle) -> bool {
        self.stores
            .get(&TypeId::of::<T>())
            .map_or(false, |s| s.has(entity))
    }

    /// Get an immutable entity reference.
    pub fn entity_ref(&self, entity: EntityHandle) -> Option<EntityRefV2<'_>> {
        if !self.is_alive(entity) {
            return None;
        }
        Some(EntityRefV2 {
            entity,
            stores: &self.stores,
        })
    }

    /// Get a mutable entity reference.
    pub fn entity_mut(&mut self, entity: EntityHandle) -> Option<EntityMutV2<'_>> {
        if !self.is_alive(entity) {
            return None;
        }
        Some(EntityMutV2 {
            entity,
            stores: &mut self.stores,
        })
    }

    /// Get a full world+entity mutable reference.
    pub fn entity_world_mut(
        &mut self,
        entity: EntityHandle,
    ) -> Option<EntityWorldMutV2<'_>> {
        if !self.is_alive(entity) {
            return None;
        }
        Some(EntityWorldMutV2 {
            entity,
            world: self,
        })
    }

    /// List all component type names attached to an entity.
    pub fn component_types_of(&self, entity: EntityHandle) -> Vec<&'static str> {
        self.stores
            .values()
            .filter(|s| s.has(entity))
            .map(|s| s.type_name())
            .collect()
    }

    /// Count components on an entity.
    pub fn component_count_of(&self, entity: EntityHandle) -> usize {
        self.stores.values().filter(|s| s.has(entity)).count()
    }

    /// Total live entity count.
    pub fn entity_count(&self) -> usize {
        self.alive.iter().filter(|&&a| a).count()
    }
}

impl Default for SimpleWorld {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// EntityRefV2 — immutable entity accessor
// ---------------------------------------------------------------------------

/// Immutable reference to an entity and its components.
///
/// Provides read-only access to any component on the entity without needing
/// to know the full component set at compile time.
pub struct EntityRefV2<'w> {
    /// The entity handle.
    entity: EntityHandle,
    /// Reference to all component stores.
    stores: &'w HashMap<TypeId, Box<dyn AnyComponentStore>>,
}

impl<'w> EntityRefV2<'w> {
    /// Get the entity handle.
    pub fn id(&self) -> EntityHandle {
        self.entity
    }

    /// Get a reference to a component of type T.
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&'w T> {
        let store = self
            .stores
            .get(&TypeId::of::<T>())?
            .as_any()
            .downcast_ref::<ComponentStore<T>>()?;
        store.get(self.entity)
    }

    /// Check if the entity has a component of type T.
    pub fn has<T: 'static>(&self) -> bool {
        self.stores
            .get(&TypeId::of::<T>())
            .map_or(false, |s| s.has(self.entity))
    }

    /// List all component type names on this entity.
    pub fn component_types(&self) -> Vec<&'static str> {
        self.stores
            .values()
            .filter(|s| s.has(self.entity))
            .map(|s| s.type_name())
            .collect()
    }

    /// Count the number of components on this entity.
    pub fn component_count(&self) -> usize {
        self.stores
            .values()
            .filter(|s| s.has(self.entity))
            .count()
    }

    /// Check if the entity has any components.
    pub fn has_any_components(&self) -> bool {
        self.stores.values().any(|s| s.has(self.entity))
    }

    /// Check if the entity has all of the given component types.
    pub fn has_all(&self, type_ids: &[TypeId]) -> bool {
        type_ids.iter().all(|tid| {
            self.stores
                .get(tid)
                .map_or(false, |s| s.has(self.entity))
        })
    }

    /// Check if the entity has any of the given component types.
    pub fn has_any(&self, type_ids: &[TypeId]) -> bool {
        type_ids.iter().any(|tid| {
            self.stores
                .get(tid)
                .map_or(false, |s| s.has(self.entity))
        })
    }
}

impl<'w> fmt::Debug for EntityRefV2<'w> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EntityRefV2")
            .field("entity", &self.entity)
            .field("component_count", &self.component_count())
            .field("components", &self.component_types())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// EntityMutV2 — mutable entity accessor
// ---------------------------------------------------------------------------

/// Mutable reference to an entity's components.
///
/// Can read and write components, but cannot add/remove components or
/// despawn the entity (use `EntityWorldMutV2` for that).
pub struct EntityMutV2<'w> {
    /// The entity handle.
    entity: EntityHandle,
    /// Mutable reference to all component stores.
    stores: &'w mut HashMap<TypeId, Box<dyn AnyComponentStore>>,
}

impl<'w> EntityMutV2<'w> {
    /// Get the entity handle.
    pub fn id(&self) -> EntityHandle {
        self.entity
    }

    /// Get a reference to a component.
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&T> {
        let store = self
            .stores
            .get(&TypeId::of::<T>())?
            .as_any()
            .downcast_ref::<ComponentStore<T>>()?;
        store.get(self.entity)
    }

    /// Get a mutable reference to a component.
    pub fn get_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        let store = self
            .stores
            .get_mut(&TypeId::of::<T>())?
            .as_any_mut()
            .downcast_mut::<ComponentStore<T>>()?;
        store.get_mut(self.entity)
    }

    /// Check if the entity has a component.
    pub fn has<T: 'static>(&self) -> bool {
        self.stores
            .get(&TypeId::of::<T>())
            .map_or(false, |s| s.has(self.entity))
    }

    /// List component type names.
    pub fn component_types(&self) -> Vec<&'static str> {
        self.stores
            .values()
            .filter(|s| s.has(self.entity))
            .map(|s| s.type_name())
            .collect()
    }

    /// Component count.
    pub fn component_count(&self) -> usize {
        self.stores
            .values()
            .filter(|s| s.has(self.entity))
            .count()
    }

    /// Apply a function to a component if it exists.
    pub fn modify<T: Send + Sync + 'static, R>(
        &mut self,
        f: impl FnOnce(&mut T) -> R,
    ) -> Option<R> {
        let store = self
            .stores
            .get_mut(&TypeId::of::<T>())?
            .as_any_mut()
            .downcast_mut::<ComponentStore<T>>()?;
        store.get_mut(self.entity).map(f)
    }

    /// Reborrow as an immutable reference.
    pub fn as_ref(&self) -> EntityRefV2<'_> {
        EntityRefV2 {
            entity: self.entity,
            stores: self.stores,
        }
    }
}

impl<'w> fmt::Debug for EntityMutV2<'w> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EntityMutV2")
            .field("entity", &self.entity)
            .field("component_count", &self.component_count())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// EntityWorldMutV2 — entity + world mutable accessor
// ---------------------------------------------------------------------------

/// Full mutable access to both an entity and the world.
///
/// Can add/remove components, despawn the entity, and perform any other
/// structural change.
pub struct EntityWorldMutV2<'w> {
    /// The entity handle.
    entity: EntityHandle,
    /// Mutable reference to the world.
    world: &'w mut SimpleWorld,
}

impl<'w> EntityWorldMutV2<'w> {
    /// Get the entity handle.
    pub fn id(&self) -> EntityHandle {
        self.entity
    }

    /// Get a reference to a component.
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.world.get_component::<T>(self.entity)
    }

    /// Get a mutable reference to a component.
    pub fn get_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.world.get_component_mut::<T>(self.entity)
    }

    /// Check if the entity has a component.
    pub fn has<T: 'static>(&self) -> bool {
        self.world.has_component::<T>(self.entity)
    }

    /// Add a component to the entity.
    pub fn insert<T: Send + Sync + 'static>(&mut self, component: T) -> &mut Self {
        self.world.add_component(self.entity, component);
        self
    }

    /// Remove a component from the entity.
    pub fn remove<T: Send + Sync + 'static>(&mut self) -> Option<T> {
        self.world.remove_component::<T>(self.entity)
    }

    /// Despawn this entity.
    pub fn despawn(self) -> bool {
        self.world.despawn(self.entity)
    }

    /// List component type names.
    pub fn component_types(&self) -> Vec<&'static str> {
        self.world.component_types_of(self.entity)
    }

    /// Component count.
    pub fn component_count(&self) -> usize {
        self.world.component_count_of(self.entity)
    }

    /// Retain only components of the specified types, removing all others.
    pub fn retain_components(&mut self, keep: &[TypeId]) {
        let to_remove: Vec<TypeId> = self
            .world
            .stores
            .keys()
            .filter(|tid| !keep.contains(tid))
            .filter(|tid| {
                self.world.stores.get(*tid).map_or(false, |s| s.has(self.entity))
            })
            .copied()
            .collect();

        for tid in to_remove {
            if let Some(store) = self.world.stores.get_mut(&tid) {
                store.remove(self.entity);
            }
        }
    }

    /// Remove all components from the entity (but keep it alive).
    pub fn clear_components(&mut self) {
        for store in self.world.stores.values_mut() {
            store.remove(self.entity);
        }
    }

    /// Check if entity is still alive.
    pub fn is_alive(&self) -> bool {
        self.world.is_alive(self.entity)
    }
}

impl<'w> fmt::Debug for EntityWorldMutV2<'w> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EntityWorldMutV2")
            .field("entity", &self.entity)
            .field("alive", &self.is_alive())
            .field("component_count", &self.component_count())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Entity builder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing entities with components.
pub struct EntityBuilderV2<'w> {
    entity: EntityHandle,
    world: &'w mut SimpleWorld,
}

impl<'w> EntityBuilderV2<'w> {
    /// Create a builder for a newly spawned entity.
    pub fn new(world: &'w mut SimpleWorld) -> Self {
        let entity = world.spawn();
        Self { entity, world }
    }

    /// Add a component.
    pub fn with<T: Send + Sync + 'static>(self, component: T) -> Self {
        self.world.add_component(self.entity, component);
        self
    }

    /// Finish building and return the entity handle.
    pub fn build(self) -> EntityHandle {
        self.entity
    }

    /// Get the entity handle without finishing.
    pub fn id(&self) -> EntityHandle {
        self.entity
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Velocity {
        dx: f32,
        dy: f32,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Name(String);

    #[test]
    fn entity_ref_read() {
        let mut world = SimpleWorld::new();
        let e = world.spawn();
        world.add_component(e, Position { x: 1.0, y: 2.0 });
        world.add_component(e, Name("test".to_string()));

        let eref = world.entity_ref(e).unwrap();
        assert_eq!(eref.get::<Position>().unwrap().x, 1.0);
        assert!(eref.has::<Name>());
        assert!(!eref.has::<Velocity>());
        assert_eq!(eref.component_count(), 2);
    }

    #[test]
    fn entity_mut_write() {
        let mut world = SimpleWorld::new();
        let e = world.spawn();
        world.add_component(e, Position { x: 0.0, y: 0.0 });

        {
            let mut emut = world.entity_mut(e).unwrap();
            emut.get_mut::<Position>().unwrap().x = 42.0;
        }

        assert_eq!(world.get_component::<Position>(e).unwrap().x, 42.0);
    }

    #[test]
    fn entity_world_mut_insert_remove() {
        let mut world = SimpleWorld::new();
        let e = world.spawn();

        {
            let mut ewm = world.entity_world_mut(e).unwrap();
            ewm.insert(Position { x: 1.0, y: 2.0 });
            ewm.insert(Velocity { dx: 3.0, dy: 4.0 });
            assert_eq!(ewm.component_count(), 2);

            ewm.remove::<Velocity>();
            assert_eq!(ewm.component_count(), 1);
        }

        assert!(world.has_component::<Position>(e));
        assert!(!world.has_component::<Velocity>(e));
    }

    #[test]
    fn entity_world_mut_despawn() {
        let mut world = SimpleWorld::new();
        let e = world.spawn();
        world.add_component(e, Position { x: 0.0, y: 0.0 });

        let ewm = world.entity_world_mut(e).unwrap();
        assert!(ewm.despawn());
        assert!(!world.is_alive(e));
    }

    #[test]
    fn entity_builder() {
        let mut world = SimpleWorld::new();
        let e = EntityBuilderV2::new(&mut world)
            .with(Position { x: 10.0, y: 20.0 })
            .with(Velocity { dx: 1.0, dy: 0.0 })
            .with(Name("player".to_string()))
            .build();

        assert!(world.is_alive(e));
        assert_eq!(world.component_count_of(e), 3);
    }

    #[test]
    fn clear_components() {
        let mut world = SimpleWorld::new();
        let e = world.spawn();
        world.add_component(e, Position { x: 0.0, y: 0.0 });
        world.add_component(e, Velocity { dx: 1.0, dy: 1.0 });

        let mut ewm = world.entity_world_mut(e).unwrap();
        ewm.clear_components();
        assert_eq!(ewm.component_count(), 0);
        assert!(ewm.is_alive());
    }

    #[test]
    fn entity_ref_debug() {
        let mut world = SimpleWorld::new();
        let e = world.spawn();
        world.add_component(e, Position { x: 0.0, y: 0.0 });
        let eref = world.entity_ref(e).unwrap();
        let debug = format!("{:?}", eref);
        assert!(debug.contains("EntityRefV2"));
    }
}
