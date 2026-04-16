//! Entity management for the Genovo ECS.
//!
//! Entities are lightweight identifiers composed of an index and a generation
//! counter. The generation prevents ABA problems when entity slots are reused.

use std::fmt;

// ---------------------------------------------------------------------------
// Entity handle
// ---------------------------------------------------------------------------

/// A lightweight, copyable handle that identifies a living entity inside a
/// [`World`](crate::World). The `generation` field is bumped every time the
/// slot at `id` is recycled so that stale handles can be detected.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Entity {
    /// Slot index into the dense entity array.
    pub id: u32,
    /// Generation counter for ABA-safety.
    pub generation: u32,
}

impl Entity {
    /// Sentinel value representing "no entity".
    pub const PLACEHOLDER: Self = Self {
        id: u32::MAX,
        generation: 0,
    };

    /// Create a new entity handle. Prefer using [`EntityStorage::allocate`].
    #[inline]
    pub const fn new(id: u32, generation: u32) -> Self {
        Self { id, generation }
    }

    /// Returns `true` if this handle is the placeholder sentinel.
    #[inline]
    pub const fn is_placeholder(&self) -> bool {
        self.id == u32::MAX
    }
}

impl fmt::Display for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}v{}", self.id, self.generation)
    }
}

// ---------------------------------------------------------------------------
// EntityBuilder
// ---------------------------------------------------------------------------

/// Fluent builder returned by [`World::spawn_entity`](crate::World::spawn_entity).
///
/// Components are collected into a deferred list and applied when `.build()`
/// is called, avoiding borrow-checker issues with `&mut World`.
///
/// ```ignore
/// let entity = world.spawn_entity()
///     .with(Position { x: 0.0, y: 0.0 })
///     .with(Velocity { dx: 1.0, dy: 0.0 })
///     .build();
/// ```
pub struct EntityBuilder<'w> {
    /// The entity being constructed.
    entity: Entity,
    /// Back-reference to the world so components can be inserted on build.
    world: &'w mut crate::World,
    /// Deferred component insertions: each closure inserts one component.
    pending: Vec<Box<dyn FnOnce(&mut crate::World)>>,
}

impl<'w> EntityBuilder<'w> {
    /// Create a new builder. Called internally by `World::spawn_entity`.
    pub(crate) fn new(entity: Entity, world: &'w mut crate::World) -> Self {
        Self {
            entity,
            world,
            pending: Vec::new(),
        }
    }

    /// Attach a component to the entity under construction.
    pub fn with<C: crate::Component>(mut self, component: C) -> Self {
        let entity = self.entity;
        self.pending
            .push(Box::new(move |world: &mut crate::World| {
                world.add_component(entity, component);
            }));
        self
    }

    /// Finalize the entity: apply all pending component insertions and return
    /// the entity handle.
    pub fn build(self) -> Entity {
        let entity = self.entity;
        // We need to move `pending` out before calling methods on `world`,
        // since `self.world` and `self.pending` are both owned by `self`.
        let pending = self.pending;
        let world = self.world;
        for insert_fn in pending {
            insert_fn(world);
        }
        entity
    }

    /// Returns the [`Entity`] handle that will be produced on build.
    #[inline]
    pub fn id(&self) -> Entity {
        self.entity
    }
}

// ---------------------------------------------------------------------------
// EntityStorage
// ---------------------------------------------------------------------------

/// Dense entity array with a free-list for slot recycling.
///
/// Allocating an entity is O(1) (pop from free-list or push to the end).
/// Deallocating is O(1) (push to free-list and bump generation).
pub struct EntityStorage {
    /// Per-slot generation counters. Index == slot id.
    generations: Vec<u32>,
    /// Alive flag per slot.
    alive: Vec<bool>,
    /// Free-list of recyclable slot indices.
    free_list: Vec<u32>,
    /// Total number of currently alive entities.
    living_count: u32,
}

impl EntityStorage {
    /// Create an empty storage with no pre-allocated capacity.
    pub fn new() -> Self {
        Self {
            generations: Vec::new(),
            alive: Vec::new(),
            free_list: Vec::new(),
            living_count: 0,
        }
    }

    /// Create storage pre-allocated for `capacity` entities.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            generations: Vec::with_capacity(capacity),
            alive: Vec::with_capacity(capacity),
            free_list: Vec::new(),
            living_count: 0,
        }
    }

    /// Allocate a new entity handle. Reuses a freed slot when available.
    pub fn allocate(&mut self) -> Entity {
        if let Some(id) = self.free_list.pop() {
            let generation = self.generations[id as usize];
            self.alive[id as usize] = true;
            self.living_count += 1;
            Entity::new(id, generation)
        } else {
            let id = self.generations.len() as u32;
            self.generations.push(0);
            self.alive.push(true);
            self.living_count += 1;
            Entity::new(id, 0)
        }
    }

    /// Free an entity slot, bumping its generation counter.
    ///
    /// Returns `true` if the entity was alive and is now freed.
    pub fn free(&mut self, entity: Entity) -> bool {
        let idx = entity.id as usize;
        if idx >= self.alive.len() {
            return false;
        }
        if !self.alive[idx] || self.generations[idx] != entity.generation {
            return false;
        }
        self.alive[idx] = false;
        self.generations[idx] = self.generations[idx].wrapping_add(1);
        self.free_list.push(entity.id);
        self.living_count -= 1;
        true
    }

    /// Check whether an entity handle refers to a currently alive entity.
    #[inline]
    pub fn is_alive(&self, entity: Entity) -> bool {
        let idx = entity.id as usize;
        idx < self.alive.len()
            && self.alive[idx]
            && self.generations[idx] == entity.generation
    }

    /// Returns the number of living entities.
    #[inline]
    pub fn len(&self) -> u32 {
        self.living_count
    }

    /// Returns `true` if there are no living entities.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.living_count == 0
    }

    /// Returns the total capacity (number of slots ever allocated).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.generations.len()
    }

    /// Return an iterator over all currently alive entities.
    pub fn iter_alive(&self) -> impl Iterator<Item = Entity> + '_ {
        self.alive
            .iter()
            .enumerate()
            .filter(|&(_, alive)| *alive)
            .map(|(idx, _)| Entity::new(idx as u32, self.generations[idx]))
    }
}

impl Default for EntityStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_and_free() {
        let mut storage = EntityStorage::new();
        let e = storage.allocate();
        assert!(storage.is_alive(e));
        assert_eq!(storage.len(), 1);
        assert!(storage.free(e));
        assert!(!storage.is_alive(e));
        assert_eq!(storage.len(), 0);
    }

    #[test]
    fn generation_increments_on_reuse() {
        let mut storage = EntityStorage::new();
        let e1 = storage.allocate();
        storage.free(e1);
        let e2 = storage.allocate();
        assert_eq!(e1.id, e2.id);
        assert_eq!(e2.generation, e1.generation + 1);
    }

    #[test]
    fn free_stale_entity_returns_false() {
        let mut storage = EntityStorage::new();
        let e1 = storage.allocate();
        storage.free(e1);
        // e1 is now stale -- freeing again should fail.
        assert!(!storage.free(e1));
    }

    #[test]
    fn iter_alive_yields_living_entities() {
        let mut storage = EntityStorage::new();
        let e1 = storage.allocate();
        let e2 = storage.allocate();
        let e3 = storage.allocate();
        storage.free(e2);
        let alive: Vec<Entity> = storage.iter_alive().collect();
        assert_eq!(alive.len(), 2);
        assert!(alive.contains(&e1));
        assert!(alive.contains(&e3));
    }

    #[test]
    fn placeholder_is_not_alive() {
        let storage = EntityStorage::new();
        assert!(!storage.is_alive(Entity::PLACEHOLDER));
        assert!(Entity::PLACEHOLDER.is_placeholder());
    }

    #[test]
    fn display_format() {
        let e = Entity::new(5, 3);
        assert_eq!(format!("{}", e), "5v3");
    }
}
