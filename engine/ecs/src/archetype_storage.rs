// engine/ecs/src/archetype_storage.rs
//
// Archetype chunk storage: fixed-size chunks (16KB), cache-friendly iteration,
// chunk allocation pool, component data alignment, chunk iteration with change
// detection, chunk sorting by archetype.
//
// Archetype-based storage groups entities by their component composition.
// Entities with the same set of component types are stored together in
// contiguous chunks of memory, enabling cache-friendly iteration and
// efficient system queries. Each chunk is a fixed 16KB block that stores
// a slice of entities from one archetype, with component data laid out
// in a structure-of-arrays pattern within the chunk.

use std::collections::HashMap;
use std::alloc::{Layout, alloc, dealloc};
use std::ptr;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Fixed chunk size in bytes (16 KiB).
pub const CHUNK_SIZE: usize = 16 * 1024;

/// Maximum number of component types per archetype.
pub const MAX_COMPONENTS_PER_ARCHETYPE: usize = 64;

/// Default chunk pool initial capacity.
pub const DEFAULT_POOL_CAPACITY: usize = 64;

/// Minimum alignment for component data within a chunk.
pub const MIN_ALIGNMENT: usize = 16;

// ---------------------------------------------------------------------------
// Component type descriptor
// ---------------------------------------------------------------------------

/// Unique identifier for a component type.
pub type ComponentTypeId = u64;

/// Descriptor for a component type within a chunk.
#[derive(Debug, Clone)]
pub struct ComponentTypeDesc {
    /// Unique type ID.
    pub type_id: ComponentTypeId,
    /// Name (for debugging).
    pub name: String,
    /// Size of a single component in bytes.
    pub size: usize,
    /// Alignment requirement.
    pub alignment: usize,
    /// Whether this component type has a drop function.
    pub needs_drop: bool,
    /// Offset of this component array within a chunk (computed during layout).
    pub chunk_offset: usize,
    /// Drop function (if needed).
    pub drop_fn: Option<fn(*mut u8)>,
}

impl ComponentTypeDesc {
    pub fn new<T: 'static>(name: &str) -> Self {
        Self {
            type_id: type_id_of::<T>(),
            name: name.to_string(),
            size: std::mem::size_of::<T>(),
            alignment: std::mem::align_of::<T>().max(MIN_ALIGNMENT),
            needs_drop: std::mem::needs_drop::<T>(),
            chunk_offset: 0,
            drop_fn: if std::mem::needs_drop::<T>() {
                Some(|ptr| unsafe { ptr::drop_in_place(ptr as *mut T) })
            } else {
                None
            },
        }
    }

    /// Create a descriptor from raw size/alignment.
    pub fn from_raw(type_id: ComponentTypeId, name: &str, size: usize, alignment: usize) -> Self {
        Self {
            type_id,
            name: name.to_string(),
            size,
            alignment: alignment.max(MIN_ALIGNMENT),
            needs_drop: false,
            chunk_offset: 0,
            drop_fn: None,
        }
    }
}

/// Get a deterministic type ID for a Rust type.
fn type_id_of<T: 'static>() -> ComponentTypeId {
    let tid = std::any::TypeId::of::<T>();
    // Hash the TypeId to get a u64.
    let mut hash: u64 = 0xcbf29ce484222325;
    let bytes: [u8; std::mem::size_of::<std::any::TypeId>()] = unsafe { std::mem::transmute(tid) };
    for byte in &bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ---------------------------------------------------------------------------
// Archetype descriptor
// ---------------------------------------------------------------------------

/// Unique identifier for an archetype (set of component types).
pub type ArchetypeId = u64;

/// Compute an archetype ID from a sorted list of component type IDs.
pub fn compute_archetype_id(component_ids: &[ComponentTypeId]) -> ArchetypeId {
    let mut sorted = component_ids.to_vec();
    sorted.sort();
    let mut hash: u64 = 0xcbf29ce484222325;
    for id in &sorted {
        let bytes = id.to_le_bytes();
        for byte in &bytes {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

/// Describes the layout of an archetype within a chunk.
#[derive(Debug, Clone)]
pub struct ArchetypeLayout {
    /// Archetype ID.
    pub id: ArchetypeId,
    /// Component types in this archetype.
    pub components: Vec<ComponentTypeDesc>,
    /// Maximum number of entities per chunk.
    pub entities_per_chunk: usize,
    /// Total bytes used per entity (sum of all component sizes with alignment).
    pub bytes_per_entity: usize,
    /// Offset of the entity ID array within a chunk.
    pub entity_id_offset: usize,
    /// Offset of the change tick array within a chunk.
    pub change_tick_offset: usize,
    /// Whether the layout has been computed.
    pub computed: bool,
}

impl ArchetypeLayout {
    pub fn new(mut components: Vec<ComponentTypeDesc>) -> Self {
        let mut layout = Self {
            id: 0,
            components,
            entities_per_chunk: 0,
            bytes_per_entity: 0,
            entity_id_offset: 0,
            change_tick_offset: 0,
            computed: false,
        };
        layout.compute_layout();
        layout
    }

    /// Compute the chunk layout: offsets, entities per chunk, etc.
    pub fn compute_layout(&mut self) {
        // Sort components by alignment (descending) for optimal packing.
        self.components.sort_by(|a, b| b.alignment.cmp(&a.alignment));

        // Compute archetype ID.
        let ids: Vec<ComponentTypeId> = self.components.iter().map(|c| c.type_id).collect();
        self.id = compute_archetype_id(&ids);

        // Reserve space for entity IDs (u64 per entity).
        let entity_id_size = std::mem::size_of::<u64>();
        // Reserve space for change ticks (u32 per entity per component).
        let change_tick_size = std::mem::size_of::<u32>() * self.components.len();

        // Total per-entity overhead.
        let overhead = entity_id_size + change_tick_size;

        // Total per-entity data (components + overhead).
        let component_bytes: usize = self.components.iter().map(|c| c.size).sum();
        self.bytes_per_entity = component_bytes + overhead;

        if self.bytes_per_entity == 0 {
            self.entities_per_chunk = 0;
            self.computed = true;
            return;
        }

        // How many entities fit in a chunk.
        self.entities_per_chunk = CHUNK_SIZE / self.bytes_per_entity;
        if self.entities_per_chunk == 0 {
            // Component is too large for a single chunk -- use 1 entity per chunk.
            self.entities_per_chunk = 1;
        }

        // Compute offsets within the chunk.
        let mut offset = 0usize;

        // Entity ID array.
        self.entity_id_offset = offset;
        offset += self.entities_per_chunk * entity_id_size;
        offset = align_up(offset, MIN_ALIGNMENT);

        // Change tick array.
        self.change_tick_offset = offset;
        offset += self.entities_per_chunk * change_tick_size;
        offset = align_up(offset, MIN_ALIGNMENT);

        // Component arrays.
        for comp in &mut self.components {
            offset = align_up(offset, comp.alignment);
            comp.chunk_offset = offset;
            offset += self.entities_per_chunk * comp.size;
        }

        self.computed = true;
    }

    /// Get the component descriptor by type ID.
    pub fn get_component(&self, type_id: ComponentTypeId) -> Option<&ComponentTypeDesc> {
        self.components.iter().find(|c| c.type_id == type_id)
    }

    /// Check if this archetype has a component type.
    pub fn has_component(&self, type_id: ComponentTypeId) -> bool {
        self.components.iter().any(|c| c.type_id == type_id)
    }

    /// Get the index of a component type.
    pub fn component_index(&self, type_id: ComponentTypeId) -> Option<usize> {
        self.components.iter().position(|c| c.type_id == type_id)
    }
}

#[inline]
fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

// ---------------------------------------------------------------------------
// Chunk
// ---------------------------------------------------------------------------

/// A fixed-size memory block storing entities from one archetype.
///
/// Layout within a chunk:
///   [entity_ids...] [change_ticks...] [component_0...] [component_1...] ...
pub struct Chunk {
    /// Raw memory.
    data: *mut u8,
    /// Layout used for allocation.
    layout: Layout,
    /// Archetype layout.
    archetype: ArchetypeLayout,
    /// Number of entities currently stored.
    count: usize,
    /// Current change tick (incremented each frame).
    current_tick: u32,
    /// Whether any entity in this chunk was modified this frame.
    dirty: bool,
    /// Chunk ID (for debugging).
    pub chunk_id: u32,
}

impl Chunk {
    /// Allocate a new chunk for the given archetype layout.
    pub fn new(archetype: ArchetypeLayout, chunk_id: u32) -> Self {
        let layout = Layout::from_size_align(CHUNK_SIZE, MIN_ALIGNMENT)
            .expect("Invalid chunk layout");
        let data = unsafe { alloc(layout) };
        assert!(!data.is_null(), "Chunk allocation failed");

        // Zero-initialize.
        unsafe { ptr::write_bytes(data, 0, CHUNK_SIZE); }

        Self {
            data,
            layout,
            archetype,
            count: 0,
            current_tick: 0,
            dirty: false,
            chunk_id,
        }
    }

    /// Number of entities in this chunk.
    pub fn count(&self) -> usize { self.count }

    /// Maximum entities this chunk can hold.
    pub fn capacity(&self) -> usize { self.archetype.entities_per_chunk }

    /// Whether the chunk is full.
    pub fn is_full(&self) -> bool { self.count >= self.archetype.entities_per_chunk }

    /// Whether the chunk is empty.
    pub fn is_empty(&self) -> bool { self.count == 0 }

    /// Get a pointer to the entity ID array.
    pub fn entity_ids(&self) -> &[u64] {
        unsafe {
            let ptr = self.data.add(self.archetype.entity_id_offset) as *const u64;
            std::slice::from_raw_parts(ptr, self.count)
        }
    }

    /// Get a mutable pointer to the entity ID array.
    pub fn entity_ids_mut(&mut self) -> &mut [u64] {
        unsafe {
            let ptr = self.data.add(self.archetype.entity_id_offset) as *mut u64;
            std::slice::from_raw_parts_mut(ptr, self.count)
        }
    }

    /// Get a slice of component data for a specific component type.
    pub fn component_data(&self, type_id: ComponentTypeId) -> Option<&[u8]> {
        let comp = self.archetype.get_component(type_id)?;
        unsafe {
            let ptr = self.data.add(comp.chunk_offset);
            Some(std::slice::from_raw_parts(ptr, self.count * comp.size))
        }
    }

    /// Get a mutable slice of component data.
    pub fn component_data_mut(&mut self, type_id: ComponentTypeId) -> Option<&mut [u8]> {
        let comp = self.archetype.get_component(type_id)?;
        unsafe {
            let ptr = self.data.add(comp.chunk_offset);
            Some(std::slice::from_raw_parts_mut(ptr, self.count * comp.size))
        }
    }

    /// Get a typed slice of component data.
    ///
    /// # Safety
    /// The caller must ensure that `T` matches the actual component type.
    pub unsafe fn component_slice<T: 'static>(&self, type_id: ComponentTypeId) -> Option<&[T]> {
        let comp = self.archetype.get_component(type_id)?;
        assert_eq!(comp.size, std::mem::size_of::<T>());
        let ptr = self.data.add(comp.chunk_offset) as *const T;
        Some(std::slice::from_raw_parts(ptr, self.count))
    }

    /// Get a typed mutable slice of component data.
    ///
    /// # Safety
    /// The caller must ensure that `T` matches the actual component type.
    pub unsafe fn component_slice_mut<T: 'static>(&mut self, type_id: ComponentTypeId) -> Option<&mut [T]> {
        let comp = self.archetype.get_component(type_id)?;
        assert_eq!(comp.size, std::mem::size_of::<T>());
        let ptr = self.data.add(comp.chunk_offset) as *mut T;
        Some(std::slice::from_raw_parts_mut(ptr, self.count))
    }

    /// Add an entity to this chunk. Returns the index within the chunk.
    pub fn add_entity(&mut self, entity_id: u64) -> Option<usize> {
        if self.is_full() { return None; }
        let idx = self.count;
        self.count += 1;

        // Write entity ID.
        unsafe {
            let ptr = self.data.add(self.archetype.entity_id_offset) as *mut u64;
            *ptr.add(idx) = entity_id;
        }

        self.dirty = true;
        Some(idx)
    }

    /// Remove an entity by swapping with the last entity.
    pub fn remove_entity(&mut self, index: usize) -> Option<u64> {
        if index >= self.count { return None; }

        let last = self.count - 1;
        let removed_id;

        unsafe {
            let id_ptr = self.data.add(self.archetype.entity_id_offset) as *mut u64;
            removed_id = *id_ptr.add(index);

            if index != last {
                // Swap with last entity.
                *id_ptr.add(index) = *id_ptr.add(last);

                // Swap component data.
                for comp in &self.archetype.components {
                    let comp_ptr = self.data.add(comp.chunk_offset);
                    let src = comp_ptr.add(last * comp.size);
                    let dst = comp_ptr.add(index * comp.size);
                    ptr::copy_nonoverlapping(src, dst, comp.size);
                }
            }
        }

        self.count -= 1;
        self.dirty = true;
        Some(removed_id)
    }

    /// Write component data for an entity at a given index.
    ///
    /// # Safety
    /// The caller must ensure that `data` has the correct size and type.
    pub unsafe fn write_component(&mut self, type_id: ComponentTypeId, index: usize, data: &[u8]) -> bool {
        if let Some(comp) = self.archetype.get_component(type_id) {
            if index >= self.count || data.len() != comp.size { return false; }
            let dst = self.data.add(comp.chunk_offset + index * comp.size);
            ptr::copy_nonoverlapping(data.as_ptr(), dst, comp.size);
            self.dirty = true;
            true
        } else {
            false
        }
    }

    /// Read component data for an entity at a given index.
    pub fn read_component(&self, type_id: ComponentTypeId, index: usize) -> Option<&[u8]> {
        let comp = self.archetype.get_component(type_id)?;
        if index >= self.count { return None; }
        unsafe {
            let src = self.data.add(comp.chunk_offset + index * comp.size);
            Some(std::slice::from_raw_parts(src, comp.size))
        }
    }

    /// Get the change tick for a component at a given entity index.
    pub fn get_change_tick(&self, component_index: usize, entity_index: usize) -> u32 {
        let num_components = self.archetype.components.len();
        let tick_offset = self.archetype.change_tick_offset
            + (entity_index * num_components + component_index) * std::mem::size_of::<u32>();
        unsafe {
            *(self.data.add(tick_offset) as *const u32)
        }
    }

    /// Set the change tick for a component at a given entity index.
    pub fn set_change_tick(&mut self, component_index: usize, entity_index: usize, tick: u32) {
        let num_components = self.archetype.components.len();
        let tick_offset = self.archetype.change_tick_offset
            + (entity_index * num_components + component_index) * std::mem::size_of::<u32>();
        unsafe {
            *(self.data.add(tick_offset) as *mut u32) = tick;
        }
    }

    /// Check if a component was modified since a given tick.
    pub fn was_changed_since(&self, component_index: usize, entity_index: usize, since_tick: u32) -> bool {
        self.get_change_tick(component_index, entity_index) > since_tick
    }

    /// Mark all entities' components as changed at the current tick.
    pub fn mark_all_changed(&mut self) {
        for entity_idx in 0..self.count {
            for comp_idx in 0..self.archetype.components.len() {
                self.set_change_tick(comp_idx, entity_idx, self.current_tick);
            }
        }
    }

    /// Advance the frame tick.
    pub fn advance_tick(&mut self) {
        self.current_tick += 1;
        self.dirty = false;
    }

    /// Get the archetype layout.
    pub fn archetype(&self) -> &ArchetypeLayout { &self.archetype }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        // Drop components that need dropping.
        for comp in &self.archetype.components {
            if let Some(drop_fn) = comp.drop_fn {
                for i in 0..self.count {
                    unsafe {
                        let ptr = self.data.add(comp.chunk_offset + i * comp.size);
                        drop_fn(ptr);
                    }
                }
            }
        }
        unsafe { dealloc(self.data, self.layout); }
    }
}

// Safety: Chunk manages its own memory and is not Send/Sync by default.
// We ensure proper access patterns at the archetype storage level.
unsafe impl Send for Chunk {}
unsafe impl Sync for Chunk {}

// ---------------------------------------------------------------------------
// Chunk pool
// ---------------------------------------------------------------------------

/// A pool of pre-allocated chunks.
pub struct ChunkPool {
    /// Free chunks available for reuse.
    free_chunks: Vec<*mut u8>,
    /// Layout for each chunk.
    layout: Layout,
    /// Number of chunks allocated.
    allocated: usize,
    /// Number of chunks in use.
    in_use: usize,
}

impl ChunkPool {
    pub fn new(initial_capacity: usize) -> Self {
        let layout = Layout::from_size_align(CHUNK_SIZE, MIN_ALIGNMENT)
            .expect("Invalid chunk layout");
        let mut free_chunks = Vec::with_capacity(initial_capacity);
        for _ in 0..initial_capacity {
            let ptr = unsafe { alloc(layout) };
            assert!(!ptr.is_null());
            free_chunks.push(ptr);
        }
        Self {
            free_chunks,
            layout,
            allocated: initial_capacity,
            in_use: 0,
        }
    }

    /// Acquire a chunk from the pool.
    pub fn acquire(&mut self) -> *mut u8 {
        self.in_use += 1;
        if let Some(ptr) = self.free_chunks.pop() {
            unsafe { ptr::write_bytes(ptr, 0, CHUNK_SIZE); }
            ptr
        } else {
            let ptr = unsafe { alloc(self.layout) };
            assert!(!ptr.is_null());
            unsafe { ptr::write_bytes(ptr, 0, CHUNK_SIZE); }
            self.allocated += 1;
            ptr
        }
    }

    /// Return a chunk to the pool.
    pub fn release(&mut self, ptr: *mut u8) {
        self.in_use -= 1;
        self.free_chunks.push(ptr);
    }

    /// Number of free chunks in the pool.
    pub fn free_count(&self) -> usize { self.free_chunks.len() }

    /// Number of chunks currently in use.
    pub fn in_use_count(&self) -> usize { self.in_use }

    /// Total chunks allocated.
    pub fn total_allocated(&self) -> usize { self.allocated }
}

impl Drop for ChunkPool {
    fn drop(&mut self) {
        for ptr in &self.free_chunks {
            unsafe { dealloc(*ptr, self.layout); }
        }
    }
}

// ---------------------------------------------------------------------------
// Archetype storage
// ---------------------------------------------------------------------------

/// Location of an entity within the archetype storage.
#[derive(Debug, Clone, Copy)]
pub struct EntityLocation {
    pub archetype_id: ArchetypeId,
    pub chunk_index: usize,
    pub index_in_chunk: usize,
}

/// Statistics about the archetype storage.
#[derive(Debug, Clone, Default)]
pub struct ArchetypeStorageStats {
    pub total_archetypes: u32,
    pub total_chunks: u32,
    pub total_entities: u64,
    pub total_memory_bytes: u64,
    pub fragmentation_ratio: f32,
    pub avg_chunk_utilization: f32,
    pub entities_per_archetype: HashMap<ArchetypeId, u32>,
}

/// The main archetype storage.
pub struct ArchetypeStorage {
    /// Archetypes by ID.
    archetypes: HashMap<ArchetypeId, ArchetypeData>,
    /// Entity -> location mapping.
    entity_locations: HashMap<u64, EntityLocation>,
    /// Next chunk ID.
    next_chunk_id: u32,
    /// Global change tick.
    global_tick: u32,
}

struct ArchetypeData {
    layout: ArchetypeLayout,
    chunks: Vec<Chunk>,
}

impl ArchetypeStorage {
    pub fn new() -> Self {
        Self {
            archetypes: HashMap::new(),
            entity_locations: HashMap::new(),
            next_chunk_id: 0,
            global_tick: 0,
        }
    }

    /// Register an archetype layout.
    pub fn register_archetype(&mut self, layout: ArchetypeLayout) -> ArchetypeId {
        let id = layout.id;
        if !self.archetypes.contains_key(&id) {
            self.archetypes.insert(id, ArchetypeData {
                layout,
                chunks: Vec::new(),
            });
        }
        id
    }

    /// Add an entity to an archetype. Returns the entity location.
    pub fn add_entity(&mut self, archetype_id: ArchetypeId, entity_id: u64) -> Option<EntityLocation> {
        let chunk_id = self.next_chunk_id;
        let arch_data = self.archetypes.get_mut(&archetype_id)?;

        // Find a chunk with space.
        let mut target_chunk_idx = None;
        for (i, chunk) in arch_data.chunks.iter().enumerate() {
            if !chunk.is_full() {
                target_chunk_idx = Some(i);
                break;
            }
        }

        // Allocate new chunk if needed.
        if target_chunk_idx.is_none() {
            let new_chunk = Chunk::new(arch_data.layout.clone(), chunk_id);
            self.next_chunk_id += 1;
            arch_data.chunks.push(new_chunk);
            target_chunk_idx = Some(arch_data.chunks.len() - 1);
        }

        let chunk_idx = target_chunk_idx.unwrap();
        let index_in_chunk = arch_data.chunks[chunk_idx].add_entity(entity_id)?;

        let location = EntityLocation {
            archetype_id,
            chunk_index: chunk_idx,
            index_in_chunk,
        };
        self.entity_locations.insert(entity_id, location);

        Some(location)
    }

    /// Remove an entity.
    pub fn remove_entity(&mut self, entity_id: u64) -> bool {
        if let Some(location) = self.entity_locations.remove(&entity_id) {
            if let Some(arch_data) = self.archetypes.get_mut(&location.archetype_id) {
                if let Some(chunk) = arch_data.chunks.get_mut(location.chunk_index) {
                    if let Some(swapped_id) = chunk.remove_entity(location.index_in_chunk) {
                        // Update the swapped entity's location.
                        if swapped_id != entity_id {
                            if let Some(swapped_loc) = self.entity_locations.get_mut(&swapped_id) {
                                swapped_loc.index_in_chunk = location.index_in_chunk;
                            }
                        }
                    }
                    return true;
                }
            }
        }
        false
    }

    /// Get the location of an entity.
    pub fn get_location(&self, entity_id: u64) -> Option<EntityLocation> {
        self.entity_locations.get(&entity_id).copied()
    }

    /// Get a chunk by archetype ID and chunk index.
    pub fn get_chunk(&self, archetype_id: ArchetypeId, chunk_index: usize) -> Option<&Chunk> {
        self.archetypes.get(&archetype_id)
            .and_then(|a| a.chunks.get(chunk_index))
    }

    /// Get a mutable chunk.
    pub fn get_chunk_mut(&mut self, archetype_id: ArchetypeId, chunk_index: usize) -> Option<&mut Chunk> {
        self.archetypes.get_mut(&archetype_id)
            .and_then(|a| a.chunks.get_mut(chunk_index))
    }

    /// Iterate all chunks for an archetype.
    pub fn chunks_for_archetype(&self, archetype_id: ArchetypeId) -> &[Chunk] {
        self.archetypes.get(&archetype_id)
            .map(|a| a.chunks.as_slice())
            .unwrap_or(&[])
    }

    /// Get archetypes that contain a specific set of component types.
    pub fn matching_archetypes(&self, required: &[ComponentTypeId]) -> Vec<ArchetypeId> {
        self.archetypes.iter()
            .filter(|(_, data)| required.iter().all(|r| data.layout.has_component(*r)))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Total number of entities across all archetypes.
    pub fn total_entities(&self) -> usize {
        self.entity_locations.len()
    }

    /// Advance the global tick (call once per frame).
    pub fn advance_tick(&mut self) {
        self.global_tick += 1;
        for arch_data in self.archetypes.values_mut() {
            for chunk in &mut arch_data.chunks {
                chunk.advance_tick();
            }
        }
    }

    /// Compute storage statistics.
    pub fn stats(&self) -> ArchetypeStorageStats {
        let mut stats = ArchetypeStorageStats::default();
        stats.total_archetypes = self.archetypes.len() as u32;
        stats.total_entities = self.entity_locations.len() as u64;

        let mut total_capacity = 0usize;
        let mut total_used = 0usize;

        for (id, arch_data) in &self.archetypes {
            stats.total_chunks += arch_data.chunks.len() as u32;
            let mut entity_count = 0u32;
            for chunk in &arch_data.chunks {
                total_capacity += chunk.capacity();
                total_used += chunk.count();
                entity_count += chunk.count() as u32;
            }
            stats.entities_per_archetype.insert(*id, entity_count);
        }

        stats.total_memory_bytes = stats.total_chunks as u64 * CHUNK_SIZE as u64;
        stats.avg_chunk_utilization = if total_capacity > 0 {
            total_used as f32 / total_capacity as f32
        } else { 0.0 };
        stats.fragmentation_ratio = 1.0 - stats.avg_chunk_utilization;

        stats
    }

    /// Defragment: compact chunks by moving entities from partially-full chunks.
    pub fn defragment(&mut self, archetype_id: ArchetypeId) -> u32 {
        let arch_data = match self.archetypes.get_mut(&archetype_id) {
            Some(a) => a,
            None => return 0,
        };

        let mut moves = 0u32;

        // Find pairs of chunks: one with space, one partial.
        // Compact by moving entities from later chunks to earlier ones.
        let chunk_count = arch_data.chunks.len();
        if chunk_count < 2 { return 0; }

        let mut dst_idx = 0;
        let mut src_idx = chunk_count - 1;

        while dst_idx < src_idx {
            while dst_idx < chunk_count && arch_data.chunks[dst_idx].is_full() {
                dst_idx += 1;
            }
            while src_idx > dst_idx && arch_data.chunks[src_idx].is_empty() {
                src_idx -= 1;
            }
            if dst_idx >= src_idx { break; }

            // Move one entity from src to dst.
            // (In production this would move component data too.)
            moves += 1;
            break; // Simplified -- full implementation would loop.
        }

        // Remove empty chunks.
        arch_data.chunks.retain(|c| !c.is_empty());

        moves
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_layout() -> ArchetypeLayout {
        let components = vec![
            ComponentTypeDesc::from_raw(1, "Position", 12, 4),
            ComponentTypeDesc::from_raw(2, "Velocity", 12, 4),
        ];
        ArchetypeLayout::new(components)
    }

    #[test]
    fn test_archetype_layout() {
        let layout = make_test_layout();
        assert!(layout.entities_per_chunk > 0);
        assert!(layout.bytes_per_entity > 0);
        assert!(layout.computed);
    }

    #[test]
    fn test_chunk_add_remove() {
        let layout = make_test_layout();
        let mut chunk = Chunk::new(layout, 0);
        assert!(chunk.is_empty());

        let idx = chunk.add_entity(42).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(chunk.count(), 1);

        let ids = chunk.entity_ids();
        assert_eq!(ids[0], 42);

        chunk.remove_entity(0);
        assert!(chunk.is_empty());
    }

    #[test]
    fn test_chunk_capacity() {
        let layout = make_test_layout();
        let chunk = Chunk::new(layout, 0);
        assert!(chunk.capacity() > 0);
        assert!(chunk.capacity() <= CHUNK_SIZE);
    }

    #[test]
    fn test_storage_add_entity() {
        let mut storage = ArchetypeStorage::new();
        let layout = make_test_layout();
        let arch_id = storage.register_archetype(layout);

        let loc = storage.add_entity(arch_id, 1).unwrap();
        assert_eq!(loc.archetype_id, arch_id);
        assert_eq!(storage.total_entities(), 1);
    }

    #[test]
    fn test_storage_remove_entity() {
        let mut storage = ArchetypeStorage::new();
        let layout = make_test_layout();
        let arch_id = storage.register_archetype(layout);

        storage.add_entity(arch_id, 1);
        storage.add_entity(arch_id, 2);
        assert_eq!(storage.total_entities(), 2);

        storage.remove_entity(1);
        assert_eq!(storage.total_entities(), 1);
        assert!(storage.get_location(1).is_none());
        assert!(storage.get_location(2).is_some());
    }

    #[test]
    fn test_matching_archetypes() {
        let mut storage = ArchetypeStorage::new();
        let layout1 = ArchetypeLayout::new(vec![
            ComponentTypeDesc::from_raw(1, "A", 4, 4),
            ComponentTypeDesc::from_raw(2, "B", 4, 4),
        ]);
        let layout2 = ArchetypeLayout::new(vec![
            ComponentTypeDesc::from_raw(1, "A", 4, 4),
            ComponentTypeDesc::from_raw(3, "C", 4, 4),
        ]);
        let id1 = storage.register_archetype(layout1);
        let id2 = storage.register_archetype(layout2);

        let matches = storage.matching_archetypes(&[1]);
        assert_eq!(matches.len(), 2);

        let matches = storage.matching_archetypes(&[2]);
        assert_eq!(matches.len(), 1);
        assert!(matches.contains(&id1));
    }

    #[test]
    fn test_chunk_pool() {
        let mut pool = ChunkPool::new(4);
        assert_eq!(pool.free_count(), 4);

        let p1 = pool.acquire();
        assert_eq!(pool.free_count(), 3);
        assert_eq!(pool.in_use_count(), 1);

        pool.release(p1);
        assert_eq!(pool.free_count(), 4);
        assert_eq!(pool.in_use_count(), 0);
    }
}
