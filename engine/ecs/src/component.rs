//! Component storage for the Genovo ECS.
//!
//! Components are plain data types that satisfy [`Component`]. They are stored
//! in type-erased containers keyed by entity id.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;


// ---------------------------------------------------------------------------
// Component trait
// ---------------------------------------------------------------------------

/// Marker trait for all ECS components.
///
/// Any `'static + Send + Sync` type automatically qualifies, but types should
/// opt in explicitly so that derive macros and reflection can discover them.
///
/// ```ignore
/// #[derive(Debug)]
/// struct Position { x: f32, y: f32 }
/// impl Component for Position {}
/// ```
pub trait Component: 'static + Send + Sync {}

// ---------------------------------------------------------------------------
// ComponentId
// ---------------------------------------------------------------------------

/// Runtime identifier for a component type, derived from [`TypeId`].
///
/// Using a newtype rather than raw `TypeId` lets us keep the door open for
/// dynamic (scripting-originated) components in the future.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentId(TypeId);

impl ComponentId {
    /// Obtain the [`ComponentId`] for a concrete component type.
    #[inline]
    pub fn of<C: Component>() -> Self {
        Self(TypeId::of::<C>())
    }

    /// Construct a [`ComponentId`] from a raw [`TypeId`].
    ///
    /// This is used internally by the archetype system where we may not have
    /// the `Component` trait bound available but do know the type id.
    #[inline]
    pub fn of_raw(type_id: TypeId) -> Self {
        Self(type_id)
    }

    /// Return the inner [`TypeId`].
    #[inline]
    pub fn type_id(&self) -> TypeId {
        self.0
    }
}

impl fmt::Debug for ComponentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ComponentId({:?})", self.0)
    }
}

// ---------------------------------------------------------------------------
// AnyComponentStorage trait
// ---------------------------------------------------------------------------

/// Type-erased interface for component storage. Each concrete
/// `ComponentStorage<T>` implements this trait so the [`World`](crate::World)
/// can store all storages in a single heterogeneous map.
pub trait AnyComponentStorage: Any + Send + Sync {
    /// Remove the component for the given entity id.
    fn remove_entity(&mut self, entity_id: u32);

    /// Returns `true` if a component exists for the given entity id.
    fn has(&self, entity_id: u32) -> bool;

    /// Upcast to `&dyn Any` for downcasting to the concrete type.
    fn as_any(&self) -> &dyn Any;

    /// Upcast to `&mut dyn Any` for downcasting to the concrete type.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// ---------------------------------------------------------------------------
// ComponentStorage<T>
// ---------------------------------------------------------------------------

/// Simple `HashMap`-based storage for a single component type.
///
/// Provides O(1) insert, remove, and lookup by entity id. Not as
/// cache-friendly as archetype-based SoA storage, but correct and
/// straightforward.
pub struct ComponentStorage<T: Component> {
    data: HashMap<u32, T>,
}

impl<T: Component> ComponentStorage<T> {
    /// Create a new, empty component storage.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Insert a component for the given entity id. Returns the previous value
    /// if one existed.
    pub fn insert(&mut self, entity_id: u32, component: T) -> Option<T> {
        self.data.insert(entity_id, component)
    }

    /// Remove and return the component for the given entity id.
    pub fn remove(&mut self, entity_id: u32) -> Option<T> {
        self.data.remove(&entity_id)
    }

    /// Get an immutable reference to the component for the given entity id.
    #[inline]
    pub fn get(&self, entity_id: u32) -> Option<&T> {
        self.data.get(&entity_id)
    }

    /// Get a mutable reference to the component for the given entity id.
    #[inline]
    pub fn get_mut(&mut self, entity_id: u32) -> Option<&mut T> {
        self.data.get_mut(&entity_id)
    }

    /// Returns `true` if a component exists for the given entity id.
    #[inline]
    pub fn has(&self, entity_id: u32) -> bool {
        self.data.contains_key(&entity_id)
    }

    /// Returns the number of stored components.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if no components are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Iterate over all `(entity_id, &component)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &T)> {
        self.data.iter().map(|(&id, val)| (id, val))
    }

    /// Iterate over all `(entity_id, &mut component)` pairs.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u32, &mut T)> {
        self.data.iter_mut().map(|(&id, val)| (id, val))
    }
}

impl<T: Component> Default for ComponentStorage<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Component> AnyComponentStorage for ComponentStorage<T> {
    fn remove_entity(&mut self, entity_id: u32) {
        self.data.remove(&entity_id);
    }

    fn has(&self, entity_id: u32) -> bool {
        self.data.contains_key(&entity_id)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
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

    #[test]
    fn insert_get_remove() {
        let mut storage = ComponentStorage::<Pos>::new();
        assert!(storage.insert(0, Pos { x: 1.0, y: 2.0 }).is_none());
        assert_eq!(storage.get(0), Some(&Pos { x: 1.0, y: 2.0 }));
        assert!(storage.has(0));

        let removed = storage.remove(0);
        assert_eq!(removed, Some(Pos { x: 1.0, y: 2.0 }));
        assert!(!storage.has(0));
        assert_eq!(storage.get(0), None);
    }

    #[test]
    fn insert_replaces_existing() {
        let mut storage = ComponentStorage::<Pos>::new();
        storage.insert(5, Pos { x: 1.0, y: 2.0 });
        let old = storage.insert(5, Pos { x: 3.0, y: 4.0 });
        assert_eq!(old, Some(Pos { x: 1.0, y: 2.0 }));
        assert_eq!(storage.get(5), Some(&Pos { x: 3.0, y: 4.0 }));
    }

    #[test]
    fn get_mut_modifies_in_place() {
        let mut storage = ComponentStorage::<Pos>::new();
        storage.insert(0, Pos { x: 1.0, y: 2.0 });
        if let Some(pos) = storage.get_mut(0) {
            pos.x = 99.0;
        }
        assert_eq!(storage.get(0), Some(&Pos { x: 99.0, y: 2.0 }));
    }

    #[test]
    fn len_and_is_empty() {
        let mut storage = ComponentStorage::<Pos>::new();
        assert!(storage.is_empty());
        assert_eq!(storage.len(), 0);

        storage.insert(0, Pos { x: 0.0, y: 0.0 });
        storage.insert(1, Pos { x: 1.0, y: 1.0 });
        assert_eq!(storage.len(), 2);
        assert!(!storage.is_empty());

        storage.remove(0);
        assert_eq!(storage.len(), 1);
    }

    #[test]
    fn any_component_storage_trait() {
        let mut storage = ComponentStorage::<Pos>::new();
        storage.insert(0, Pos { x: 1.0, y: 2.0 });
        storage.insert(1, Pos { x: 3.0, y: 4.0 });

        let any_storage: &mut dyn AnyComponentStorage = &mut storage;
        assert!(any_storage.has(0));
        assert!(any_storage.has(1));

        any_storage.remove_entity(0);
        assert!(!any_storage.has(0));
        assert!(any_storage.has(1));
    }

    #[test]
    fn downcast_via_as_any() {
        let mut storage = ComponentStorage::<Pos>::new();
        storage.insert(0, Pos { x: 1.0, y: 2.0 });

        let any_storage: &dyn AnyComponentStorage = &storage;
        let downcasted = any_storage
            .as_any()
            .downcast_ref::<ComponentStorage<Pos>>()
            .expect("downcast should succeed");
        assert_eq!(downcasted.get(0), Some(&Pos { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn component_id_distinct_types() {
        let pos_id = ComponentId::of::<Pos>();
        let vel_id = ComponentId::of::<Vel>();
        assert_ne!(pos_id, vel_id);
        assert_eq!(pos_id, ComponentId::of::<Pos>());
    }

    #[test]
    fn iterate_all_components() {
        let mut storage = ComponentStorage::<Pos>::new();
        storage.insert(10, Pos { x: 1.0, y: 0.0 });
        storage.insert(20, Pos { x: 2.0, y: 0.0 });
        storage.insert(30, Pos { x: 3.0, y: 0.0 });

        let mut items: Vec<(u32, f32)> = storage.iter().map(|(id, p)| (id, p.x)).collect();
        items.sort_by_key(|(id, _)| *id);
        assert_eq!(
            items,
            vec![(10, 1.0), (20, 2.0), (30, 3.0)]
        );
    }
}
