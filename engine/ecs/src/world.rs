//! Enhanced ECS World with archetype graph, column storage, typed resources,
//! exclusive world access, and world merge/split operations.

/// Type alias: the primary `World` used by the engine is `WorldV2`.
pub type World = WorldV2;
/// Type alias: `Entity` maps to `EntityV2`.
pub type Entity = EntityV2;

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Entity handle
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityV2 {
    pub index: u32,
    pub generation: u32,
}

impl EntityV2 {
    pub const INVALID: Self = Self { index: u32::MAX, generation: 0 };
    pub fn new(index: u32, generation: u32) -> Self { Self { index, generation } }
    pub fn is_valid(&self) -> bool { self.index != u32::MAX }
    pub fn to_bits(&self) -> u64 { ((self.generation as u64) << 32) | (self.index as u64) }
    pub fn from_bits(bits: u64) -> Self { Self { index: bits as u32, generation: (bits >> 32) as u32 } }
}

impl fmt::Debug for EntityV2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "Entity({}v{})", self.index, self.generation) }
}

impl fmt::Display for EntityV2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}v{}", self.index, self.generation) }
}

// ---------------------------------------------------------------------------
// Entity allocator
// ---------------------------------------------------------------------------

pub struct EntityAllocatorV2 {
    generations: Vec<u32>,
    alive: Vec<bool>,
    free_list: Vec<u32>,
    next_fresh: u32,
    live_count: u32,
}

impl EntityAllocatorV2 {
    pub fn new() -> Self {
        Self { generations: Vec::with_capacity(1024), alive: Vec::with_capacity(1024), free_list: Vec::with_capacity(256), next_fresh: 0, live_count: 0 }
    }

    pub fn with_capacity(n: u32) -> Self {
        Self { generations: Vec::with_capacity(n as usize), alive: Vec::with_capacity(n as usize), free_list: Vec::with_capacity((n / 4) as usize), next_fresh: 0, live_count: 0 }
    }

    pub fn allocate(&mut self) -> EntityV2 {
        let index = if let Some(recycled) = self.free_list.pop() { recycled } else {
            let idx = self.next_fresh; self.next_fresh += 1;
            self.generations.push(0); self.alive.push(false); idx
        };
        let generation = self.generations[index as usize];
        self.alive[index as usize] = true;
        self.live_count += 1;
        EntityV2::new(index, generation)
    }

    pub fn deallocate(&mut self, entity: EntityV2) -> bool {
        let idx = entity.index as usize;
        if idx >= self.alive.len() { return false; }
        if !self.alive[idx] || self.generations[idx] != entity.generation { return false; }
        self.alive[idx] = false;
        self.generations[idx] = self.generations[idx].wrapping_add(1);
        self.free_list.push(entity.index);
        self.live_count -= 1;
        true
    }

    pub fn is_alive(&self, entity: EntityV2) -> bool {
        let idx = entity.index as usize;
        idx < self.alive.len() && self.alive[idx] && self.generations[idx] == entity.generation
    }

    pub fn live_count(&self) -> u32 { self.live_count }
    pub fn total_slots(&self) -> u32 { self.next_fresh }
    pub fn free_count(&self) -> usize { self.free_list.len() }
    pub fn reserve(&mut self, additional: u32) { self.generations.reserve(additional as usize); self.alive.reserve(additional as usize); }

    pub fn iter_alive(&self) -> impl Iterator<Item = EntityV2> + '_ {
        self.alive.iter().enumerate().filter_map(move |(idx, &is_alive)| {
            if is_alive { Some(EntityV2::new(idx as u32, self.generations[idx])) } else { None }
        })
    }

    pub fn clear(&mut self) { self.generations.clear(); self.alive.clear(); self.free_list.clear(); self.next_fresh = 0; self.live_count = 0; }
}

impl Default for EntityAllocatorV2 { fn default() -> Self { Self::new() } }

// ---------------------------------------------------------------------------
// Component column
// ---------------------------------------------------------------------------

pub struct ComponentColumnV2 {
    pub type_id: TypeId,
    pub type_name: &'static str,
    pub item_size: usize,
    pub item_align: usize,
    data: Vec<u8>,
    pub count: usize,
    drop_fn: Option<unsafe fn(*mut u8)>,
}

impl ComponentColumnV2 {
    pub fn new<T: 'static>() -> Self {
        let drop_fn: Option<unsafe fn(*mut u8)> = if std::mem::needs_drop::<T>() {
            Some(|ptr: *mut u8| unsafe { std::ptr::drop_in_place(ptr as *mut T); })
        } else { None };
        Self { type_id: TypeId::of::<T>(), type_name: std::any::type_name::<T>(), item_size: std::mem::size_of::<T>(), item_align: std::mem::align_of::<T>(), data: Vec::new(), count: 0, drop_fn }
    }

    pub fn from_raw(type_id: TypeId, type_name: &'static str, item_size: usize, item_align: usize, drop_fn: Option<unsafe fn(*mut u8)>) -> Self {
        Self { type_id, type_name, item_size, item_align, data: Vec::new(), count: 0, drop_fn }
    }

    pub unsafe fn push_raw(&mut self, value_ptr: *const u8) {
        let start = self.data.len();
        self.data.resize(start + self.item_size, 0);
        std::ptr::copy_nonoverlapping(value_ptr, self.data.as_mut_ptr().add(start), self.item_size);
        self.count += 1;
    }

    pub fn push<T: 'static>(&mut self, value: T) {
        debug_assert_eq!(TypeId::of::<T>(), self.type_id);
        let start = self.data.len();
        self.data.resize(start + self.item_size, 0);
        unsafe { let ptr = self.data.as_mut_ptr().add(start) as *mut T; std::ptr::write(ptr, value); }
        self.count += 1;
    }

    pub unsafe fn get<T: 'static>(&self, index: usize) -> &T {
        debug_assert!(index < self.count);
        &*(self.data.as_ptr().add(index * self.item_size) as *const T)
    }

    pub unsafe fn get_mut<T: 'static>(&mut self, index: usize) -> &mut T {
        debug_assert!(index < self.count);
        &mut *(self.data.as_mut_ptr().add(index * self.item_size) as *mut T)
    }

    pub fn get_raw(&self, index: usize) -> *const u8 {
        debug_assert!(index < self.count);
        unsafe { self.data.as_ptr().add(index * self.item_size) }
    }

    pub fn get_raw_mut(&mut self, index: usize) -> *mut u8 {
        debug_assert!(index < self.count);
        unsafe { self.data.as_mut_ptr().add(index * self.item_size) }
    }

    pub fn swap_remove(&mut self, index: usize) -> bool {
        debug_assert!(index < self.count);
        let last = self.count - 1;
        let swapped = index != last;
        if let Some(drop_fn) = self.drop_fn {
            unsafe { drop_fn(self.data.as_mut_ptr().add(index * self.item_size)); }
        }
        if swapped {
            unsafe { std::ptr::copy(self.data.as_ptr().add(last * self.item_size), self.data.as_mut_ptr().add(index * self.item_size), self.item_size); }
        }
        self.data.truncate(last * self.item_size);
        self.count -= 1;
        swapped
    }

    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }

    pub fn clear(&mut self) {
        if let Some(drop_fn) = self.drop_fn {
            for i in 0..self.count { unsafe { drop_fn(self.data.as_mut_ptr().add(i * self.item_size)); } }
        }
        self.data.clear(); self.count = 0;
    }

    pub fn byte_size(&self) -> usize { self.data.len() }
}

impl Drop for ComponentColumnV2 { fn drop(&mut self) { self.clear(); } }

// ---------------------------------------------------------------------------
// Archetype V2
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ArchetypeIdV2(pub u32);
impl ArchetypeIdV2 { pub const EMPTY: Self = Self(0); }

pub struct ArchetypeV2 {
    pub id: ArchetypeIdV2,
    pub component_types: Vec<TypeId>,
    pub columns: HashMap<TypeId, ComponentColumnV2>,
    pub entities: Vec<EntityV2>,
    pub add_edges: HashMap<TypeId, ArchetypeIdV2>,
    pub remove_edges: HashMap<TypeId, ArchetypeIdV2>,
}

impl ArchetypeV2 {
    pub fn empty(id: ArchetypeIdV2) -> Self {
        Self { id, component_types: Vec::new(), columns: HashMap::new(), entities: Vec::new(), add_edges: HashMap::new(), remove_edges: HashMap::new() }
    }
    pub fn new(id: ArchetypeIdV2, component_types: Vec<TypeId>) -> Self {
        Self { id, component_types, columns: HashMap::new(), entities: Vec::new(), add_edges: HashMap::new(), remove_edges: HashMap::new() }
    }
    pub fn entity_count(&self) -> usize { self.entities.len() }
    pub fn has_component(&self, type_id: TypeId) -> bool { self.columns.contains_key(&type_id) || self.component_types.binary_search(&type_id).is_ok() }
    pub fn entity_index(&self, entity: EntityV2) -> Option<usize> { self.entities.iter().position(|&e| e == entity) }
    pub fn add_edge(&mut self, component: TypeId, target: ArchetypeIdV2) { self.add_edges.insert(component, target); }
    pub fn remove_edge(&mut self, component: TypeId, target: ArchetypeIdV2) { self.remove_edges.insert(component, target); }
    pub fn get_add_target(&self, component: TypeId) -> Option<ArchetypeIdV2> { self.add_edges.get(&component).copied() }
    pub fn get_remove_target(&self, component: TypeId) -> Option<ArchetypeIdV2> { self.remove_edges.get(&component).copied() }
    pub fn memory_usage(&self) -> usize {
        let mut total = self.entities.len() * std::mem::size_of::<EntityV2>();
        for col in self.columns.values() { total += col.byte_size(); }
        total
    }
}

// ---------------------------------------------------------------------------
// Resource storage
// ---------------------------------------------------------------------------

struct ResourceEntry { value: Box<dyn Any + Send + Sync>, changed_tick: u32, inserted_tick: u32 }

pub struct ResourceStorage {
    pub entries: HashMap<TypeId, ResourceEntry>,
    current_tick: u32,
}

impl ResourceStorage {
    pub fn new() -> Self { Self { entries: HashMap::new(), current_tick: 0 } }
    pub fn set_tick(&mut self, tick: u32) { self.current_tick = tick; }
    pub fn insert<T: Send + Sync + 'static>(&mut self, value: T) {
        self.entries.insert(TypeId::of::<T>(), ResourceEntry { value: Box::new(value), changed_tick: self.current_tick, inserted_tick: self.current_tick });
    }
    pub fn remove<T: Send + Sync + 'static>(&mut self) -> Option<T> {
        self.entries.remove(&TypeId::of::<T>()).map(|e| *e.value.downcast::<T>().unwrap())
    }
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&T> { self.entries.get(&TypeId::of::<T>()).and_then(|e| e.value.downcast_ref::<T>()) }
    pub fn get_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        let tick = self.current_tick;
        self.entries.get_mut(&TypeId::of::<T>()).map(|e| { e.changed_tick = tick; e.value.downcast_mut::<T>().unwrap() })
    }
    pub fn contains<T: Send + Sync + 'static>(&self) -> bool { self.entries.contains_key(&TypeId::of::<T>()) }
    pub fn is_changed<T: Send + Sync + 'static>(&self, last_tick: u32) -> bool { self.entries.get(&TypeId::of::<T>()).map_or(false, |e| e.changed_tick > last_tick) }
    pub fn clear(&mut self) { self.entries.clear(); }
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    pub fn type_ids(&self) -> Vec<TypeId> { self.entries.keys().copied().collect() }
}

impl Default for ResourceStorage { fn default() -> Self { Self::new() } }

// ---------------------------------------------------------------------------
// Entity location
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct EntityLocation { pub archetype_id: ArchetypeIdV2, pub row: usize }

// ---------------------------------------------------------------------------
// WorldV2
// ---------------------------------------------------------------------------

pub struct WorldV2 {
    entities: EntityAllocatorV2,
    archetypes: Vec<ArchetypeV2>,
    archetype_index: HashMap<Vec<TypeId>, ArchetypeIdV2>,
    entity_locations: HashMap<EntityV2, EntityLocation>,
    resources: ResourceStorage,
    current_tick: u32,
    world_id: u64,
    next_child_id: AtomicU64,
}

impl WorldV2 {
    pub fn new() -> Self {
        let empty_arch = ArchetypeV2::empty(ArchetypeIdV2::EMPTY);
        let mut archetype_index = HashMap::new();
        archetype_index.insert(Vec::<TypeId>::new(), ArchetypeIdV2::EMPTY);
        static NEXT_WORLD_ID: AtomicU64 = AtomicU64::new(1);
        Self {
            entities: EntityAllocatorV2::new(), archetypes: vec![empty_arch], archetype_index,
            entity_locations: HashMap::new(), resources: ResourceStorage::new(), current_tick: 0,
            world_id: NEXT_WORLD_ID.fetch_add(1, Ordering::Relaxed), next_child_id: AtomicU64::new(1),
        }
    }

    pub fn with_capacity(entity_capacity: u32) -> Self {
        let mut world = Self::new(); world.entities.reserve(entity_capacity); world.entity_locations.reserve(entity_capacity as usize); world
    }

    pub fn id(&self) -> u64 { self.world_id }
    pub fn current_tick(&self) -> u32 { self.current_tick }
    pub fn increment_tick(&mut self) { self.current_tick = self.current_tick.wrapping_add(1); self.resources.set_tick(self.current_tick); }

    pub fn spawn_empty(&mut self) -> EntityV2 {
        let entity = self.entities.allocate();
        let row = self.archetypes[0].entities.len();
        self.archetypes[0].entities.push(entity);
        self.entity_locations.insert(entity, EntityLocation { archetype_id: ArchetypeIdV2::EMPTY, row });
        entity
    }

    pub fn despawn(&mut self, entity: EntityV2) -> bool {
        if !self.entities.is_alive(entity) { return false; }
        if let Some(location) = self.entity_locations.remove(&entity) {
            let archetype = &mut self.archetypes[location.archetype_id.0 as usize];
            let last_row = archetype.entities.len() - 1;
            for column in archetype.columns.values_mut() { column.swap_remove(location.row); }
            if location.row != last_row {
                let swapped_entity = archetype.entities[last_row];
                archetype.entities[location.row] = swapped_entity;
                archetype.entities.pop();
                if let Some(loc) = self.entity_locations.get_mut(&swapped_entity) { loc.row = location.row; }
            } else { archetype.entities.pop(); }
        }
        self.entities.deallocate(entity);
        true
    }

    pub fn is_alive(&self, entity: EntityV2) -> bool { self.entities.is_alive(entity) }
    pub fn entity_count(&self) -> u32 { self.entities.live_count() }
    pub fn iter_entities(&self) -> impl Iterator<Item = EntityV2> + '_ { self.entities.iter_alive() }

    fn find_or_create_archetype(&mut self, type_set: Vec<TypeId>) -> ArchetypeIdV2 {
        if let Some(&id) = self.archetype_index.get(&type_set) { return id; }
        let id = ArchetypeIdV2(self.archetypes.len() as u32);
        self.archetypes.push(ArchetypeV2::new(id, type_set.clone()));
        self.archetype_index.insert(type_set, id);
        id
    }

    pub fn get_component<T: 'static>(&self, entity: EntityV2) -> Option<&T> {
        let location = self.entity_locations.get(&entity)?;
        let archetype = &self.archetypes[location.archetype_id.0 as usize];
        let column = archetype.columns.get(&TypeId::of::<T>())?;
        Some(unsafe { column.get::<T>(location.row) })
    }

    pub fn has_component<T: 'static>(&self, entity: EntityV2) -> bool {
        self.entity_locations.get(&entity).map(|loc| self.archetypes[loc.archetype_id.0 as usize].has_component(TypeId::of::<T>())).unwrap_or(false)
    }

    pub fn insert_resource<T: Send + Sync + 'static>(&mut self, value: T) { self.resources.insert(value); }
    pub fn remove_resource<T: Send + Sync + 'static>(&mut self) -> Option<T> { self.resources.remove::<T>() }
    pub fn resource<T: Send + Sync + 'static>(&self) -> Option<&T> { self.resources.get::<T>() }
    pub fn resource_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> { self.resources.get_mut::<T>() }
    pub fn has_resource<T: Send + Sync + 'static>(&self) -> bool { self.resources.contains::<T>() }

    pub fn archetype_count(&self) -> usize { self.archetypes.len() }
    pub fn entity_location(&self, entity: EntityV2) -> Option<EntityLocation> { self.entity_locations.get(&entity).copied() }

    pub fn split_out(&mut self, entities_to_move: &[EntityV2]) -> WorldV2 {
        let mut child = WorldV2::new();
        child.world_id = self.next_child_id.fetch_add(1, Ordering::Relaxed) | (self.world_id << 32);
        for &entity in entities_to_move {
            if !self.entities.is_alive(entity) { continue; }
            let child_entity = child.entities.allocate();
            let child_row = child.archetypes[0].entities.len();
            child.archetypes[0].entities.push(child_entity);
            child.entity_locations.insert(child_entity, EntityLocation { archetype_id: ArchetypeIdV2::EMPTY, row: child_row });
            self.despawn(entity);
        }
        child
    }

    pub fn merge(&mut self, other: WorldV2) -> Vec<(EntityV2, EntityV2)> {
        let mut mapping = Vec::new();
        for old_entity in other.entities.iter_alive().collect::<Vec<_>>() {
            let new_entity = self.spawn_empty();
            mapping.push((old_entity, new_entity));
        }
        mapping
    }

    pub fn exclusive_scope<R>(&mut self, f: impl FnOnce(&mut WorldV2) -> R) -> R { f(self) }

    pub fn clear_entities(&mut self) {
        for arch in &mut self.archetypes { for col in arch.columns.values_mut() { col.clear(); } arch.entities.clear(); }
        self.entity_locations.clear(); self.entities.clear();
    }

    pub fn clear_all(&mut self) { self.clear_entities(); self.resources.clear(); }

    pub fn stats(&self) -> WorldStats {
        let mut total_memory = 0; let mut component_count = 0;
        for arch in &self.archetypes { total_memory += arch.memory_usage(); component_count += arch.columns.len(); }
        WorldStats { entity_count: self.entities.live_count(), archetype_count: self.archetypes.len(), component_column_count: component_count, resource_count: self.resources.len(), total_memory_bytes: total_memory, recycled_slots: self.entities.free_count(), current_tick: self.current_tick }
    }
}

impl Default for WorldV2 { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone)]
pub struct WorldStats {
    pub entity_count: u32, pub archetype_count: usize, pub component_column_count: usize,
    pub resource_count: usize, pub total_memory_bytes: usize, pub recycled_slots: usize, pub current_tick: u32,
}

impl fmt::Display for WorldStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "World Statistics:")?;
        writeln!(f, "  Entities: {}", self.entity_count)?; writeln!(f, "  Archetypes: {}", self.archetype_count)?;
        writeln!(f, "  Columns: {}", self.component_column_count)?; writeln!(f, "  Resources: {}", self.resource_count)?;
        writeln!(f, "  Memory: {} bytes", self.total_memory_bytes)?; writeln!(f, "  Tick: {}", self.current_tick)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let e1 = alloc.allocate(); let _e2 = alloc.allocate(); alloc.deallocate(e1);
        let e3 = alloc.allocate();
        assert_eq!(e3.index, e1.index);
        assert_eq!(e3.generation, e1.generation + 1);
    }
    #[test]
    fn resources() {
        let mut world = WorldV2::new();
        world.insert_resource(42u32);
        assert_eq!(*world.resource::<u32>().unwrap(), 42);
        *world.resource_mut::<u32>().unwrap() = 100;
        assert_eq!(*world.resource::<u32>().unwrap(), 100);
    }
}
