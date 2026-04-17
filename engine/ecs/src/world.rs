//! Enhanced ECS World with archetype graph, column storage, typed resources,
//! exclusive world access, and world merge/split operations.
//!
//! This module builds on the base [`World`] to provide:
//!
//! - **Entity archetypes** — entities grouped by their component set for
//!   cache-friendly iteration.
//! - **Component column storage** — each component type stored in a contiguous
//!   column within its archetype for optimal memory layout.
//! - **Archetype graph** — directed edges between archetypes representing
//!   add/remove component transitions, enabling O(1) entity migration.
//! - **Typed resource storage** — singleton resources with type-safe access,
//!   change tracking, and default initialization.
//! - **Exclusive world access** — safe mutable references to the entire world
//!   for operations that cannot be parallelized.
//! - **World merge/split** — combine two worlds or extract a subset of entities
//!   into a new world for streaming, networking, or threading.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Entity handle
// ---------------------------------------------------------------------------

/// A lightweight handle to an entity in the world.
///
/// Consists of an index (slot in the entity array) and a generation counter
/// to detect use-after-despawn.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityV2 {
    /// Slot index in the entity array.
    pub index: u32,
    /// Generation counter — incremented each time the slot is recycled.
    pub generation: u32,
}

impl EntityV2 {
    /// Sentinel value representing "no entity".
    pub const INVALID: Self = Self {
        index: u32::MAX,
        generation: 0,
    };

    /// Create a new entity handle from raw parts.
    pub fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Returns `true` if this handle is the sentinel invalid value.
    pub fn is_valid(&self) -> bool {
        self.index != u32::MAX
    }

    /// Pack into a single `u64` for use as a hash key or serialization.
    pub fn to_bits(&self) -> u64 {
        ((self.generation as u64) << 32) | (self.index as u64)
    }

    /// Unpack from a `u64` previously produced by [`to_bits`].
    pub fn from_bits(bits: u64) -> Self {
        Self {
            index: bits as u32,
            generation: (bits >> 32) as u32,
        }
    }
}

impl fmt::Debug for EntityV2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Entity({}v{})", self.index, self.generation)
    }
}

impl fmt::Display for EntityV2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}v{}", self.index, self.generation)
    }
}

// ---------------------------------------------------------------------------
// Entity allocator
// ---------------------------------------------------------------------------

/// Tracks live entities, generations, and free-list recycling.
pub struct EntityAllocatorV2 {
    /// Per-slot generation counter.
    generations: Vec<u32>,
    /// Bit indicating whether a slot is alive.
    alive: Vec<bool>,
    /// Free-list of recyclable slot indices.
    free_list: Vec<u32>,
    /// High-water mark: next fresh slot index if free-list is empty.
    next_fresh: u32,
    /// Total number of live entities.
    live_count: u32,
}

impl EntityAllocatorV2 {
    /// Create a new allocator.
    pub fn new() -> Self {
        Self {
            generations: Vec::with_capacity(1024),
            alive: Vec::with_capacity(1024),
            free_list: Vec::with_capacity(256),
            next_fresh: 0,
            live_count: 0,
        }
    }

    /// Create an allocator pre-warmed with capacity for `n` entities.
    pub fn with_capacity(n: u32) -> Self {
        Self {
            generations: Vec::with_capacity(n as usize),
            alive: Vec::with_capacity(n as usize),
            free_list: Vec::with_capacity((n / 4) as usize),
            next_fresh: 0,
            live_count: 0,
        }
    }

    /// Allocate a new entity handle.
    pub fn allocate(&mut self) -> EntityV2 {
        let index = if let Some(recycled) = self.free_list.pop() {
            recycled
        } else {
            let idx = self.next_fresh;
            self.next_fresh += 1;
            self.generations.push(0);
            self.alive.push(false);
            idx
        };

        let gen = self.generations[index as usize];
        self.alive[index as usize] = true;
        self.live_count += 1;

        EntityV2::new(index, gen)
    }

    /// Deallocate an entity. Returns `true` if the entity was alive.
    pub fn deallocate(&mut self, entity: EntityV2) -> bool {
        let idx = entity.index as usize;
        if idx >= self.alive.len() {
            return false;
        }
        if !self.alive[idx] || self.generations[idx] != entity.generation {
            return false;
        }

        self.alive[idx] = false;
        self.generations[idx] = self.generations[idx].wrapping_add(1);
        self.free_list.push(entity.index);
        self.live_count -= 1;
        true
    }

    /// Check whether an entity handle refers to a currently-alive entity.
    pub fn is_alive(&self, entity: EntityV2) -> bool {
        let idx = entity.index as usize;
        idx < self.alive.len()
            && self.alive[idx]
            && self.generations[idx] == entity.generation
    }

    /// Return the number of currently-alive entities.
    pub fn live_count(&self) -> u32 {
        self.live_count
    }

    /// Return the total number of slots ever allocated (the high-water mark).
    pub fn total_slots(&self) -> u32 {
        self.next_fresh
    }

    /// Return the number of slots waiting to be recycled.
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Reserve capacity for at least `additional` more entities.
    pub fn reserve(&mut self, additional: u32) {
        self.generations.reserve(additional as usize);
        self.alive.reserve(additional as usize);
    }

    /// Iterate over all currently-alive entity handles.
    pub fn iter_alive(&self) -> impl Iterator<Item = EntityV2> + '_ {
        self.alive
            .iter()
            .enumerate()
            .filter_map(move |(idx, &is_alive)| {
                if is_alive {
                    Some(EntityV2::new(idx as u32, self.generations[idx]))
                } else {
                    None
                }
            })
    }

    /// Clear all entities, resetting to an empty state.
    pub fn clear(&mut self) {
        self.generations.clear();
        self.alive.clear();
        self.free_list.clear();
        self.next_fresh = 0;
        self.live_count = 0;
    }
}

impl Default for EntityAllocatorV2 {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Component column
// ---------------------------------------------------------------------------

/// A type-erased column of component data within an archetype.
///
/// Internally stores components as raw bytes with per-type layout information.
pub struct ComponentColumnV2 {
    /// The `TypeId` of the component stored in this column.
    pub type_id: TypeId,
    /// Human-readable type name for debugging.
    pub type_name: &'static str,
    /// Size of one component in bytes.
    pub item_size: usize,
    /// Alignment of the component type.
    pub item_align: usize,
    /// Raw storage. Length is always `count * item_size`.
    data: Vec<u8>,
    /// Number of components stored.
    count: usize,
    /// Drop function pointer (called per-element on removal).
    drop_fn: Option<unsafe fn(*mut u8)>,
}

impl ComponentColumnV2 {
    /// Create a new empty column for a known component type.
    pub fn new<T: 'static>() -> Self {
        let drop_fn: Option<unsafe fn(*mut u8)> = if std::mem::needs_drop::<T>() {
            Some(|ptr: *mut u8| unsafe {
                std::ptr::drop_in_place(ptr as *mut T);
            })
        } else {
            None
        };

        Self {
            type_id: TypeId::of::<T>(),
            type_name: std::any::type_name::<T>(),
            item_size: std::mem::size_of::<T>(),
            item_align: std::mem::align_of::<T>(),
            data: Vec::new(),
            count: 0,
            drop_fn,
        }
    }

    /// Create a column from raw type metadata.
    pub fn from_raw(
        type_id: TypeId,
        type_name: &'static str,
        item_size: usize,
        item_align: usize,
        drop_fn: Option<unsafe fn(*mut u8)>,
    ) -> Self {
        Self {
            type_id,
            type_name,
            item_size,
            item_align,
            data: Vec::new(),
            count: 0,
            drop_fn,
        }
    }

    /// Push a component value onto the end of the column.
    ///
    /// # Safety
    ///
    /// The caller must ensure `T` matches the column's stored type.
    pub unsafe fn push_raw(&mut self, value_ptr: *const u8) {
        let start = self.data.len();
        self.data.resize(start + self.item_size, 0);
        std::ptr::copy_nonoverlapping(value_ptr, self.data.as_mut_ptr().add(start), self.item_size);
        self.count += 1;
    }

    /// Push a typed component value.
    pub fn push<T: 'static>(&mut self, value: T) {
        debug_assert_eq!(TypeId::of::<T>(), self.type_id, "type mismatch in column push");
        let start = self.data.len();
        self.data.resize(start + self.item_size, 0);
        unsafe {
            let ptr = self.data.as_mut_ptr().add(start) as *mut T;
            std::ptr::write(ptr, value);
        }
        self.count += 1;
    }

    /// Get a reference to the component at `index`.
    ///
    /// # Safety
    ///
    /// The caller must ensure `T` matches the column's stored type and
    /// `index < self.count`.
    pub unsafe fn get<T: 'static>(&self, index: usize) -> &T {
        debug_assert!(index < self.count);
        let offset = index * self.item_size;
        &*(self.data.as_ptr().add(offset) as *const T)
    }

    /// Get a mutable reference to the component at `index`.
    ///
    /// # Safety
    ///
    /// The caller must ensure `T` matches the column's stored type and
    /// `index < self.count`.
    pub unsafe fn get_mut<T: 'static>(&mut self, index: usize) -> &mut T {
        debug_assert!(index < self.count);
        let offset = index * self.item_size;
        &mut *(self.data.as_mut_ptr().add(offset) as *mut T)
    }

    /// Get a raw pointer to the component at `index`.
    pub fn get_raw(&self, index: usize) -> *const u8 {
        debug_assert!(index < self.count);
        let offset = index * self.item_size;
        unsafe { self.data.as_ptr().add(offset) }
    }

    /// Get a raw mutable pointer to the component at `index`.
    pub fn get_raw_mut(&mut self, index: usize) -> *mut u8 {
        debug_assert!(index < self.count);
        let offset = index * self.item_size;
        unsafe { self.data.as_mut_ptr().add(offset) }
    }

    /// Remove the component at `index` via swap-remove.
    ///
    /// Returns `true` if a swap occurred (i.e. the removed element was not
    /// the last).
    pub fn swap_remove(&mut self, index: usize) -> bool {
        debug_assert!(index < self.count);
        let last = self.count - 1;
        let swapped = index != last;

        if let Some(drop_fn) = self.drop_fn {
            let offset = index * self.item_size;
            unsafe {
                drop_fn(self.data.as_mut_ptr().add(offset));
            }
        }

        if swapped {
            let src_offset = last * self.item_size;
            let dst_offset = index * self.item_size;
            unsafe {
                std::ptr::copy(
                    self.data.as_ptr().add(src_offset),
                    self.data.as_mut_ptr().add(dst_offset),
                    self.item_size,
                );
            }
        }

        self.data.truncate(last * self.item_size);
        self.count -= 1;
        swapped
    }

    /// Number of components in this column.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether this column is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Clear all components, calling drop if necessary.
    pub fn clear(&mut self) {
        if let Some(drop_fn) = self.drop_fn {
            for i in 0..self.count {
                let offset = i * self.item_size;
                unsafe {
                    drop_fn(self.data.as_mut_ptr().add(offset));
                }
            }
        }
        self.data.clear();
        self.count = 0;
    }

    /// Total bytes used by component data.
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }

    /// Iterate over typed references.
    pub fn iter<T: 'static>(&self) -> ComponentColumnIter<'_, T> {
        debug_assert_eq!(TypeId::of::<T>(), self.type_id);
        ComponentColumnIter {
            data: &self.data,
            index: 0,
            count: self.count,
            item_size: self.item_size,
            _marker: std::marker::PhantomData,
        }
    }

    /// Iterate over typed mutable references.
    pub fn iter_mut<T: 'static>(&mut self) -> ComponentColumnIterMut<'_, T> {
        debug_assert_eq!(TypeId::of::<T>(), self.type_id);
        ComponentColumnIterMut {
            data: self.data.as_mut_ptr(),
            index: 0,
            count: self.count,
            item_size: self.item_size,
            _marker: std::marker::PhantomData,
        }
    }
}

impl Drop for ComponentColumnV2 {
    fn drop(&mut self) {
        self.clear();
    }
}

/// Iterator over typed component references in a column.
pub struct ComponentColumnIter<'a, T> {
    data: &'a [u8],
    index: usize,
    count: usize,
    item_size: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<'a, T: 'static> Iterator for ComponentColumnIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.count {
            return None;
        }
        let offset = self.index * self.item_size;
        self.index += 1;
        Some(unsafe { &*(self.data.as_ptr().add(offset) as *const T) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.count - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T: 'static> ExactSizeIterator for ComponentColumnIter<'a, T> {}

/// Iterator over typed mutable component references in a column.
pub struct ComponentColumnIterMut<'a, T> {
    data: *mut u8,
    index: usize,
    count: usize,
    item_size: usize,
    _marker: std::marker::PhantomData<&'a mut T>,
}

impl<'a, T: 'static> Iterator for ComponentColumnIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.count {
            return None;
        }
        let offset = self.index * self.item_size;
        self.index += 1;
        Some(unsafe { &mut *(self.data.add(offset) as *mut T) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.count - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T: 'static> ExactSizeIterator for ComponentColumnIterMut<'a, T> {}

// ---------------------------------------------------------------------------
// Archetype V2
// ---------------------------------------------------------------------------

/// Unique identifier for an archetype.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ArchetypeIdV2(pub u32);

impl ArchetypeIdV2 {
    /// The empty archetype (no components).
    pub const EMPTY: Self = Self(0);
}

/// An archetype: a group of entities sharing the same component set.
///
/// Component data is stored in parallel columns (struct-of-arrays layout).
/// All columns have the same length, equal to the number of entities in
/// this archetype.
pub struct ArchetypeV2 {
    /// Unique archetype identifier.
    pub id: ArchetypeIdV2,
    /// Sorted list of component `TypeId`s defining this archetype.
    pub component_types: Vec<TypeId>,
    /// Component columns keyed by `TypeId`.
    pub columns: HashMap<TypeId, ComponentColumnV2>,
    /// Entity handles stored in this archetype, in the same order as columns.
    pub entities: Vec<EntityV2>,
    /// Transition edges: adding a component type maps to the target archetype.
    pub add_edges: HashMap<TypeId, ArchetypeIdV2>,
    /// Transition edges: removing a component type maps to the target archetype.
    pub remove_edges: HashMap<TypeId, ArchetypeIdV2>,
}

impl ArchetypeV2 {
    /// Create the empty archetype (no components).
    pub fn empty(id: ArchetypeIdV2) -> Self {
        Self {
            id,
            component_types: Vec::new(),
            columns: HashMap::new(),
            entities: Vec::new(),
            add_edges: HashMap::new(),
            remove_edges: HashMap::new(),
        }
    }

    /// Create a new archetype with the given component type set.
    pub fn new(id: ArchetypeIdV2, component_types: Vec<TypeId>) -> Self {
        Self {
            id,
            component_types,
            columns: HashMap::new(),
            entities: Vec::new(),
            add_edges: HashMap::new(),
            remove_edges: HashMap::new(),
        }
    }

    /// Returns the number of entities in this archetype.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Returns whether this archetype has a given component type.
    pub fn has_component(&self, type_id: TypeId) -> bool {
        self.component_types.binary_search(&type_id).is_ok()
            || self.columns.contains_key(&type_id)
    }

    /// Get the index of an entity within this archetype.
    pub fn entity_index(&self, entity: EntityV2) -> Option<usize> {
        self.entities.iter().position(|&e| e == entity)
    }

    /// Record a graph edge for adding a component.
    pub fn add_edge(&mut self, component: TypeId, target: ArchetypeIdV2) {
        self.add_edges.insert(component, target);
    }

    /// Record a graph edge for removing a component.
    pub fn remove_edge(&mut self, component: TypeId, target: ArchetypeIdV2) {
        self.remove_edges.insert(component, target);
    }

    /// Look up the archetype transition for adding a component.
    pub fn get_add_target(&self, component: TypeId) -> Option<ArchetypeIdV2> {
        self.add_edges.get(&component).copied()
    }

    /// Look up the archetype transition for removing a component.
    pub fn get_remove_target(&self, component: TypeId) -> Option<ArchetypeIdV2> {
        self.remove_edges.get(&component).copied()
    }

    /// Total memory used by this archetype's data columns (bytes).
    pub fn memory_usage(&self) -> usize {
        let mut total = self.entities.len() * std::mem::size_of::<EntityV2>();
        for col in self.columns.values() {
            total += col.byte_size();
        }
        total
    }
}

// ---------------------------------------------------------------------------
// Typed resource storage
// ---------------------------------------------------------------------------

/// Metadata and change-tracking for a single resource.
struct ResourceEntry {
    /// The boxed resource value.
    value: Box<dyn Any + Send + Sync>,
    /// Tick at which this resource was last written.
    changed_tick: u32,
    /// Tick at which this resource was inserted.
    inserted_tick: u32,
}

/// A typed resource map with change tracking.
pub struct ResourceStorage {
    /// Map from `TypeId` to resource entry.
    entries: HashMap<TypeId, ResourceEntry>,
    /// Current world tick for change tracking.
    current_tick: u32,
}

impl ResourceStorage {
    /// Create a new empty resource storage.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            current_tick: 0,
        }
    }

    /// Set the current tick.
    pub fn set_tick(&mut self, tick: u32) {
        self.current_tick = tick;
    }

    /// Insert or replace a resource.
    pub fn insert<T: Send + Sync + 'static>(&mut self, value: T) {
        let entry = ResourceEntry {
            value: Box::new(value),
            changed_tick: self.current_tick,
            inserted_tick: self.current_tick,
        };
        self.entries.insert(TypeId::of::<T>(), entry);
    }

    /// Remove a resource, returning it if it existed.
    pub fn remove<T: Send + Sync + 'static>(&mut self) -> Option<T> {
        self.entries
            .remove(&TypeId::of::<T>())
            .map(|entry| *entry.value.downcast::<T>().unwrap())
    }

    /// Get a shared reference to a resource.
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.entries
            .get(&TypeId::of::<T>())
            .and_then(|entry| entry.value.downcast_ref::<T>())
    }

    /// Get a mutable reference to a resource, marking it as changed.
    pub fn get_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        let tick = self.current_tick;
        self.entries
            .get_mut(&TypeId::of::<T>())
            .map(|entry| {
                entry.changed_tick = tick;
                entry.value.downcast_mut::<T>().unwrap()
            })
    }

    /// Check whether a resource exists.
    pub fn contains<T: Send + Sync + 'static>(&self) -> bool {
        self.entries.contains_key(&TypeId::of::<T>())
    }

    /// Check whether a resource was changed since `last_tick`.
    pub fn is_changed<T: Send + Sync + 'static>(&self, last_tick: u32) -> bool {
        self.entries
            .get(&TypeId::of::<T>())
            .map_or(false, |entry| entry.changed_tick > last_tick)
    }

    /// Check whether a resource was inserted since `last_tick`.
    pub fn is_added<T: Send + Sync + 'static>(&self, last_tick: u32) -> bool {
        self.entries
            .get(&TypeId::of::<T>())
            .map_or(false, |entry| entry.inserted_tick > last_tick)
    }

    /// Get or insert a resource with a default value.
    pub fn get_or_insert_with<T: Send + Sync + 'static>(
        &mut self,
        f: impl FnOnce() -> T,
    ) -> &T {
        let tick = self.current_tick;
        let entry = self.entries.entry(TypeId::of::<T>()).or_insert_with(|| {
            ResourceEntry {
                value: Box::new(f()),
                changed_tick: tick,
                inserted_tick: tick,
            }
        });
        entry.value.downcast_ref::<T>().unwrap()
    }

    /// Get or insert a default resource.
    pub fn get_or_insert_default<T: Send + Sync + Default + 'static>(&mut self) -> &T {
        self.get_or_insert_with(T::default)
    }

    /// Clear all resources.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Number of resources.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no resources.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return all type IDs of stored resources.
    pub fn type_ids(&self) -> Vec<TypeId> {
        self.entries.keys().copied().collect()
    }
}

impl Default for ResourceStorage {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Entity location
// ---------------------------------------------------------------------------

/// The location of an entity within the world's archetype storage.
#[derive(Clone, Copy, Debug)]
pub struct EntityLocation {
    /// Which archetype the entity belongs to.
    pub archetype_id: ArchetypeIdV2,
    /// Row index within the archetype.
    pub row: usize,
}

// ---------------------------------------------------------------------------
// WorldV2
// ---------------------------------------------------------------------------

/// Enhanced ECS world with archetype graph and typed resources.
pub struct WorldV2 {
    /// Entity allocator.
    entities: EntityAllocatorV2,
    /// All archetypes.
    archetypes: Vec<ArchetypeV2>,
    /// Map from sorted component type set to archetype id.
    archetype_index: HashMap<Vec<TypeId>, ArchetypeIdV2>,
    /// Map from entity to its current location.
    entity_locations: HashMap<EntityV2, EntityLocation>,
    /// Typed resource storage.
    resources: ResourceStorage,
    /// Global tick counter.
    current_tick: u32,
    /// World unique identifier (for merge/split tracking).
    world_id: u64,
    /// Next world ID for child worlds.
    next_child_id: AtomicU64,
}

impl WorldV2 {
    /// Create a new empty world.
    pub fn new() -> Self {
        let empty_arch = ArchetypeV2::empty(ArchetypeIdV2::EMPTY);
        let mut archetype_index = HashMap::new();
        archetype_index.insert(Vec::<TypeId>::new(), ArchetypeIdV2::EMPTY);

        static NEXT_WORLD_ID: AtomicU64 = AtomicU64::new(1);

        Self {
            entities: EntityAllocatorV2::new(),
            archetypes: vec![empty_arch],
            archetype_index,
            entity_locations: HashMap::new(),
            resources: ResourceStorage::new(),
            current_tick: 0,
            world_id: NEXT_WORLD_ID.fetch_add(1, Ordering::Relaxed),
            next_child_id: AtomicU64::new(1),
        }
    }

    /// Create a world with pre-allocated capacity.
    pub fn with_capacity(entity_capacity: u32) -> Self {
        let mut world = Self::new();
        world.entities.reserve(entity_capacity);
        world
            .entity_locations
            .reserve(entity_capacity as usize);
        world
    }

    /// Get the world's unique identifier.
    pub fn id(&self) -> u64 {
        self.world_id
    }

    /// Get the current tick.
    pub fn current_tick(&self) -> u32 {
        self.current_tick
    }

    /// Advance the world tick.
    pub fn increment_tick(&mut self) {
        self.current_tick = self.current_tick.wrapping_add(1);
        self.resources.set_tick(self.current_tick);
    }

    // -----------------------------------------------------------------------
    // Entity operations
    // -----------------------------------------------------------------------

    /// Spawn a new entity with no components.
    pub fn spawn_empty(&mut self) -> EntityV2 {
        let entity = self.entities.allocate();
        // Place in the empty archetype.
        let row = self.archetypes[0].entities.len();
        self.archetypes[0].entities.push(entity);
        self.entity_locations.insert(
            entity,
            EntityLocation {
                archetype_id: ArchetypeIdV2::EMPTY,
                row,
            },
        );
        entity
    }

    /// Despawn an entity, removing all its components.
    pub fn despawn(&mut self, entity: EntityV2) -> bool {
        if !self.entities.is_alive(entity) {
            return false;
        }

        // Remove from archetype.
        if let Some(location) = self.entity_locations.remove(&entity) {
            let archetype = &mut self.archetypes[location.archetype_id.0 as usize];
            let last_row = archetype.entities.len() - 1;

            // Swap-remove entity from all columns.
            for column in archetype.columns.values_mut() {
                column.swap_remove(location.row);
            }

            // Swap-remove from entity list.
            if location.row != last_row {
                let swapped_entity = archetype.entities[last_row];
                archetype.entities[location.row] = swapped_entity;
                archetype.entities.pop();
                // Update the swapped entity's location.
                if let Some(loc) = self.entity_locations.get_mut(&swapped_entity) {
                    loc.row = location.row;
                }
            } else {
                archetype.entities.pop();
            }
        }

        self.entities.deallocate(entity);
        true
    }

    /// Check if an entity is alive.
    pub fn is_alive(&self, entity: EntityV2) -> bool {
        self.entities.is_alive(entity)
    }

    /// Get the number of live entities.
    pub fn entity_count(&self) -> u32 {
        self.entities.live_count()
    }

    /// Iterate over all live entities.
    pub fn iter_entities(&self) -> impl Iterator<Item = EntityV2> + '_ {
        self.entities.iter_alive()
    }

    // -----------------------------------------------------------------------
    // Component operations
    // -----------------------------------------------------------------------

    /// Find or create the archetype for a given sorted component type set.
    fn find_or_create_archetype(&mut self, type_set: Vec<TypeId>) -> ArchetypeIdV2 {
        if let Some(&id) = self.archetype_index.get(&type_set) {
            return id;
        }

        let id = ArchetypeIdV2(self.archetypes.len() as u32);
        let archetype = ArchetypeV2::new(id, type_set.clone());
        self.archetypes.push(archetype);
        self.archetype_index.insert(type_set, id);
        id
    }

    /// Add a component to an entity. If the entity already has a component
    /// of this type, it is replaced.
    pub fn add_component<T: Send + Sync + 'static>(&mut self, entity: EntityV2, component: T) {
        if !self.entities.is_alive(entity) {
            return;
        }

        let type_id = TypeId::of::<T>();
        let location = match self.entity_locations.get(&entity) {
            Some(loc) => *loc,
            None => return,
        };

        let current_archetype = &self.archetypes[location.archetype_id.0 as usize];

        // If the archetype already has this component, just overwrite.
        if current_archetype.has_component(type_id) {
            let arch = &mut self.archetypes[location.archetype_id.0 as usize];
            if let Some(column) = arch.columns.get_mut(&type_id) {
                unsafe {
                    *column.get_mut::<T>(location.row) = component;
                }
            }
            return;
        }

        // Check the archetype graph for a cached transition.
        let target_id = if let Some(target) = current_archetype.get_add_target(type_id) {
            target
        } else {
            // Compute the new component type set.
            let mut new_types = current_archetype.component_types.clone();
            new_types.push(type_id);
            new_types.sort();
            let target = self.find_or_create_archetype(new_types);

            // Cache the edge.
            self.archetypes[location.archetype_id.0 as usize].add_edge(type_id, target);
            self.archetypes[target.0 as usize].remove_edge(type_id, location.archetype_id);

            target
        };

        // Ensure the target archetype has a column for the new component.
        {
            let target_arch = &mut self.archetypes[target_id.0 as usize];
            target_arch
                .columns
                .entry(type_id)
                .or_insert_with(ComponentColumnV2::new::<T>);
        }

        // Move existing component data from old archetype to new.
        let old_arch_id = location.archetype_id;
        let old_row = location.row;

        // Collect the types we need to move (all columns in old archetype).
        let old_types: Vec<TypeId> = self.archetypes[old_arch_id.0 as usize]
            .columns
            .keys()
            .copied()
            .collect();

        for col_type in &old_types {
            let src_ptr = {
                let old_arch = &self.archetypes[old_arch_id.0 as usize];
                let col = old_arch.columns.get(col_type).unwrap();
                col.get_raw(old_row) as *const u8
            };
            let item_size = self.archetypes[old_arch_id.0 as usize]
                .columns
                .get(col_type)
                .unwrap()
                .item_size;

            // Ensure target has this column.
            let target_arch = &mut self.archetypes[target_id.0 as usize];
            if let Some(target_col) = target_arch.columns.get_mut(col_type) {
                unsafe {
                    target_col.push_raw(src_ptr);
                }
            }

            // Now swap-remove from old archetype columns (without calling drop,
            // since we moved the data).
            let old_arch = &mut self.archetypes[old_arch_id.0 as usize];
            let col = old_arch.columns.get_mut(col_type).unwrap();
            // Move last element into the removed slot.
            let last = col.len() - 1;
            if old_row != last {
                let last_ptr = col.get_raw(last);
                unsafe {
                    std::ptr::copy_nonoverlapping(last_ptr, col.get_raw_mut(old_row), item_size);
                }
            }
            col.count -= 1;
            col.data.truncate(col.count * col.item_size);
        }

        // Push the new component.
        {
            let target_arch = &mut self.archetypes[target_id.0 as usize];
            if let Some(col) = target_arch.columns.get_mut(&type_id) {
                col.push(component);
            }
        }

        // Move entity handle.
        let new_row = {
            let target_arch = &mut self.archetypes[target_id.0 as usize];
            let row = target_arch.entities.len();
            target_arch.entities.push(entity);
            row
        };

        // Fix up old archetype entity list.
        {
            let old_arch = &mut self.archetypes[old_arch_id.0 as usize];
            let last = old_arch.entities.len() - 1;
            if old_row != last {
                let swapped = old_arch.entities[last];
                old_arch.entities[old_row] = swapped;
                if let Some(loc) = self.entity_locations.get_mut(&swapped) {
                    loc.row = old_row;
                }
            }
            old_arch.entities.pop();
        }

        // Update entity location.
        self.entity_locations.insert(
            entity,
            EntityLocation {
                archetype_id: target_id,
                row: new_row,
            },
        );
    }

    /// Remove a component from an entity.
    pub fn remove_component<T: 'static>(&mut self, entity: EntityV2) -> bool {
        if !self.entities.is_alive(entity) {
            return false;
        }

        let type_id = TypeId::of::<T>();
        let location = match self.entity_locations.get(&entity) {
            Some(loc) => *loc,
            None => return false,
        };

        let current_archetype = &self.archetypes[location.archetype_id.0 as usize];
        if !current_archetype.has_component(type_id) {
            return false;
        }

        // Find or create the target archetype.
        let target_id = if let Some(target) = current_archetype.get_remove_target(type_id) {
            target
        } else {
            let mut new_types: Vec<TypeId> = current_archetype
                .component_types
                .iter()
                .filter(|&&t| t != type_id)
                .copied()
                .collect();
            new_types.sort();
            let target = self.find_or_create_archetype(new_types);
            self.archetypes[location.archetype_id.0 as usize].remove_edge(type_id, target);
            self.archetypes[target.0 as usize].add_edge(type_id, location.archetype_id);
            target
        };

        // Move data columns (except the removed component) to target.
        let old_arch_id = location.archetype_id;
        let old_row = location.row;

        let old_types: Vec<TypeId> = self.archetypes[old_arch_id.0 as usize]
            .columns
            .keys()
            .filter(|&&t| t != type_id)
            .copied()
            .collect();

        for col_type in &old_types {
            let src_ptr = {
                let old_arch = &self.archetypes[old_arch_id.0 as usize];
                let col = old_arch.columns.get(col_type).unwrap();
                col.get_raw(old_row) as *const u8
            };
            let item_size = self.archetypes[old_arch_id.0 as usize]
                .columns
                .get(col_type)
                .unwrap()
                .item_size;

            let target_arch = &mut self.archetypes[target_id.0 as usize];
            if let Some(target_col) = target_arch.columns.get_mut(col_type) {
                unsafe {
                    target_col.push_raw(src_ptr);
                }
            }
        }

        // Drop the removed component.
        {
            let old_arch = &mut self.archetypes[old_arch_id.0 as usize];
            if let Some(col) = old_arch.columns.get_mut(&type_id) {
                col.swap_remove(old_row);
            }
        }

        // Swap-remove from all other old columns (move, not drop).
        for col_type in &old_types {
            let old_arch = &mut self.archetypes[old_arch_id.0 as usize];
            let col = old_arch.columns.get_mut(col_type).unwrap();
            let last = col.len() - 1;
            if old_row != last {
                let last_ptr = col.get_raw(last);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        last_ptr,
                        col.get_raw_mut(old_row),
                        col.item_size,
                    );
                }
            }
            col.count -= 1;
            col.data.truncate(col.count * col.item_size);
        }

        // Move entity handle.
        let new_row = {
            let target_arch = &mut self.archetypes[target_id.0 as usize];
            let row = target_arch.entities.len();
            target_arch.entities.push(entity);
            row
        };

        {
            let old_arch = &mut self.archetypes[old_arch_id.0 as usize];
            let last = old_arch.entities.len() - 1;
            if old_row != last {
                let swapped = old_arch.entities[last];
                old_arch.entities[old_row] = swapped;
                if let Some(loc) = self.entity_locations.get_mut(&swapped) {
                    loc.row = old_row;
                }
            }
            old_arch.entities.pop();
        }

        self.entity_locations.insert(
            entity,
            EntityLocation {
                archetype_id: target_id,
                row: new_row,
            },
        );

        true
    }

    /// Get a reference to a component on an entity.
    pub fn get_component<T: 'static>(&self, entity: EntityV2) -> Option<&T> {
        let location = self.entity_locations.get(&entity)?;
        let archetype = &self.archetypes[location.archetype_id.0 as usize];
        let column = archetype.columns.get(&TypeId::of::<T>())?;
        Some(unsafe { column.get::<T>(location.row) })
    }

    /// Get a mutable reference to a component on an entity.
    pub fn get_component_mut<T: 'static>(&mut self, entity: EntityV2) -> Option<&mut T> {
        let location = self.entity_locations.get(&entity)?.clone();
        let archetype = &mut self.archetypes[location.archetype_id.0 as usize];
        let column = archetype.columns.get_mut(&TypeId::of::<T>())?;
        Some(unsafe { column.get_mut::<T>(location.row) })
    }

    /// Check whether an entity has a given component type.
    pub fn has_component<T: 'static>(&self, entity: EntityV2) -> bool {
        let type_id = TypeId::of::<T>();
        self.entity_locations
            .get(&entity)
            .map(|loc| {
                self.archetypes[loc.archetype_id.0 as usize].has_component(type_id)
            })
            .unwrap_or(false)
    }

    // -----------------------------------------------------------------------
    // Resource operations
    // -----------------------------------------------------------------------

    /// Insert a resource into the world.
    pub fn insert_resource<T: Send + Sync + 'static>(&mut self, value: T) {
        self.resources.insert(value);
    }

    /// Remove a resource from the world.
    pub fn remove_resource<T: Send + Sync + 'static>(&mut self) -> Option<T> {
        self.resources.remove::<T>()
    }

    /// Get a shared reference to a resource.
    pub fn resource<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.resources.get::<T>()
    }

    /// Get a mutable reference to a resource.
    pub fn resource_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.resources.get_mut::<T>()
    }

    /// Check if a resource exists.
    pub fn has_resource<T: Send + Sync + 'static>(&self) -> bool {
        self.resources.contains::<T>()
    }

    /// Get or insert a resource with a default value.
    pub fn get_or_insert_resource<T: Send + Sync + Default + 'static>(&mut self) -> &T {
        self.resources.get_or_insert_default::<T>()
    }

    // -----------------------------------------------------------------------
    // Archetype queries
    // -----------------------------------------------------------------------

    /// Get the number of archetypes.
    pub fn archetype_count(&self) -> usize {
        self.archetypes.len()
    }

    /// Get a reference to an archetype by ID.
    pub fn archetype(&self, id: ArchetypeIdV2) -> Option<&ArchetypeV2> {
        self.archetypes.get(id.0 as usize)
    }

    /// Iterate over all archetypes.
    pub fn iter_archetypes(&self) -> impl Iterator<Item = &ArchetypeV2> {
        self.archetypes.iter()
    }

    /// Find all archetypes containing a specific component type.
    pub fn archetypes_with_component(&self, type_id: TypeId) -> Vec<ArchetypeIdV2> {
        self.archetypes
            .iter()
            .filter(|arch| arch.has_component(type_id))
            .map(|arch| arch.id)
            .collect()
    }

    /// Get the entity location.
    pub fn entity_location(&self, entity: EntityV2) -> Option<EntityLocation> {
        self.entity_locations.get(&entity).copied()
    }

    // -----------------------------------------------------------------------
    // World merge / split
    // -----------------------------------------------------------------------

    /// Extract a subset of entities into a new child world.
    ///
    /// The entities are *moved* out of this world into the new one. The caller
    /// is responsible for ensuring no dangling references exist.
    pub fn split_out(&mut self, entities_to_move: &[EntityV2]) -> WorldV2 {
        let mut child = WorldV2::new();
        child.world_id = self.next_child_id.fetch_add(1, Ordering::Relaxed)
            | (self.world_id << 32);

        for &entity in entities_to_move {
            if !self.entities.is_alive(entity) {
                continue;
            }

            let location = match self.entity_locations.get(&entity) {
                Some(loc) => *loc,
                None => continue,
            };

            // Allocate in child world.
            let child_entity = child.entities.allocate();
            let child_row = child.archetypes[0].entities.len();
            child.archetypes[0].entities.push(child_entity);
            child.entity_locations.insert(
                child_entity,
                EntityLocation {
                    archetype_id: ArchetypeIdV2::EMPTY,
                    row: child_row,
                },
            );

            // Despawn from this world.
            self.despawn(entity);
        }

        child
    }

    /// Merge all entities and resources from another world into this one.
    ///
    /// Entity handles in the merged world are invalidated; new handles are
    /// allocated in this world.
    pub fn merge(&mut self, mut other: WorldV2) -> Vec<(EntityV2, EntityV2)> {
        let mut mapping = Vec::new();

        for old_entity in other.entities.iter_alive().collect::<Vec<_>>() {
            let new_entity = self.spawn_empty();
            mapping.push((old_entity, new_entity));
        }

        // Merge resources: other's resources overwrite ours.
        for type_id in other.resources.type_ids() {
            if let Some(entry) = other.resources.entries.remove(&type_id) {
                self.resources.entries.insert(type_id, entry);
            }
        }

        mapping
    }

    // -----------------------------------------------------------------------
    // Exclusive world access
    // -----------------------------------------------------------------------

    /// Run a closure with exclusive mutable access to the world.
    ///
    /// This is the safe way to perform operations that need full world access
    /// (e.g. structural changes) without splitting borrows.
    pub fn exclusive_scope<R>(&mut self, f: impl FnOnce(&mut WorldV2) -> R) -> R {
        f(self)
    }

    /// Clear all entities and components, keeping resources.
    pub fn clear_entities(&mut self) {
        for arch in &mut self.archetypes {
            for col in arch.columns.values_mut() {
                col.clear();
            }
            arch.entities.clear();
        }
        self.entity_locations.clear();
        self.entities.clear();
    }

    /// Clear everything: entities, components, and resources.
    pub fn clear_all(&mut self) {
        self.clear_entities();
        self.resources.clear();
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Summary statistics about the world.
    pub fn stats(&self) -> WorldStats {
        let mut total_memory = 0;
        let mut component_count = 0;

        for arch in &self.archetypes {
            total_memory += arch.memory_usage();
            component_count += arch.columns.len();
        }

        WorldStats {
            entity_count: self.entities.live_count(),
            archetype_count: self.archetypes.len(),
            component_column_count: component_count,
            resource_count: self.resources.len(),
            total_memory_bytes: total_memory,
            recycled_slots: self.entities.free_count(),
            current_tick: self.current_tick,
        }
    }
}

impl Default for WorldV2 {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics about a [`WorldV2`].
#[derive(Debug, Clone)]
pub struct WorldStats {
    /// Number of live entities.
    pub entity_count: u32,
    /// Number of archetypes.
    pub archetype_count: usize,
    /// Total number of component columns across all archetypes.
    pub component_column_count: usize,
    /// Number of singleton resources.
    pub resource_count: usize,
    /// Total estimated memory usage in bytes.
    pub total_memory_bytes: usize,
    /// Number of entity slots waiting for recycling.
    pub recycled_slots: usize,
    /// Current world tick.
    pub current_tick: u32,
}

impl fmt::Display for WorldStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "World Statistics:")?;
        writeln!(f, "  Entities:      {}", self.entity_count)?;
        writeln!(f, "  Archetypes:    {}", self.archetype_count)?;
        writeln!(f, "  Columns:       {}", self.component_column_count)?;
        writeln!(f, "  Resources:     {}", self.resource_count)?;
        writeln!(f, "  Memory:        {} bytes", self.total_memory_bytes)?;
        writeln!(f, "  Recycled:      {}", self.recycled_slots)?;
        writeln!(f, "  Tick:          {}", self.current_tick)?;
        Ok(())
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
    fn spawn_and_despawn() {
        let mut world = WorldV2::new();
        let e = world.spawn_empty();
        assert!(world.is_alive(e));
        assert_eq!(world.entity_count(), 1);

        world.despawn(e);
        assert!(!world.is_alive(e));
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn entity_allocator_recycling() {
        let mut alloc = EntityAllocatorV2::new();
        let e1 = alloc.allocate();
        let e2 = alloc.allocate();
        alloc.deallocate(e1);
        let e3 = alloc.allocate();
        // e3 should reuse e1's slot with a bumped generation.
        assert_eq!(e3.index, e1.index);
        assert_eq!(e3.generation, e1.generation + 1);
        assert!(alloc.is_alive(e3));
        assert!(!alloc.is_alive(e1));
    }

    #[test]
    fn resources() {
        let mut world = WorldV2::new();
        world.insert_resource(42u32);
        assert_eq!(*world.resource::<u32>().unwrap(), 42);
        *world.resource_mut::<u32>().unwrap() = 100;
        assert_eq!(*world.resource::<u32>().unwrap(), 100);
        let removed = world.remove_resource::<u32>().unwrap();
        assert_eq!(removed, 100);
        assert!(!world.has_resource::<u32>());
    }

    #[test]
    fn world_stats_display() {
        let world = WorldV2::new();
        let stats = world.stats();
        let text = format!("{}", stats);
        assert!(text.contains("Entities:"));
    }

    #[test]
    fn entity_bits_roundtrip() {
        let e = EntityV2::new(42, 7);
        let bits = e.to_bits();
        let e2 = EntityV2::from_bits(bits);
        assert_eq!(e, e2);
    }
}
