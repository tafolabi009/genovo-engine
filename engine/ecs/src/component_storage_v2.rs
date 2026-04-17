// engine/ecs/src/component_storage_v2.rs
//
// Optimized component storage for the Genovo ECS.
//
// Provides SoA (struct-of-arrays) component storage with performance features:
//
// - SoA layout: each field stored in a contiguous array for cache efficiency.
// - Chunk-based iteration for predictable memory access patterns.
// - Prefetch hints for upcoming memory accesses.
// - Sorted iteration by entity ID for deterministic ordering.
// - Memory pool backing for reduced allocation overhead.
// - Column-based access for selective field reads.
// - Storage statistics and memory usage tracking.
// - Compaction and defragmentation support.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default chunk size (entities per chunk).
const DEFAULT_CHUNK_SIZE: usize = 256;

/// Maximum columns (fields) per component.
const MAX_COLUMNS: usize = 32;

/// Memory pool page size.
const POOL_PAGE_SIZE: usize = 65536;

/// Alignment for SIMD-friendly access.
const SIMD_ALIGNMENT: usize = 64;

/// Prefetch distance (elements ahead to prefetch).
const PREFETCH_DISTANCE: usize = 8;

/// Growth factor for pool expansion.
const POOL_GROWTH_FACTOR: f32 = 1.5;

// ---------------------------------------------------------------------------
// Entity Index
// ---------------------------------------------------------------------------

/// Entity identifier within the storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StorageEntityId(pub u64);

// ---------------------------------------------------------------------------
// Column Descriptor
// ---------------------------------------------------------------------------

/// Describes a single column (field) in the SoA layout.
#[derive(Debug, Clone)]
pub struct ColumnDescriptor {
    /// Column name.
    pub name: String,
    /// Byte size of a single element.
    pub element_size: usize,
    /// Alignment requirement.
    pub alignment: usize,
    /// Type name (for debugging).
    pub type_name: String,
    /// Column index.
    pub index: usize,
}

impl ColumnDescriptor {
    /// Create a new column descriptor.
    pub fn new(name: &str, element_size: usize, type_name: &str) -> Self {
        Self {
            name: name.to_string(),
            element_size,
            alignment: element_size.min(SIMD_ALIGNMENT).max(1),
            type_name: type_name.to_string(),
            index: 0,
        }
    }

    /// Create a descriptor for a common type.
    pub fn f32_column(name: &str) -> Self {
        Self::new(name, 4, "f32")
    }

    /// Create a descriptor for f64.
    pub fn f64_column(name: &str) -> Self {
        Self::new(name, 8, "f64")
    }

    /// Create a descriptor for u32.
    pub fn u32_column(name: &str) -> Self {
        Self::new(name, 4, "u32")
    }

    /// Create a descriptor for u64.
    pub fn u64_column(name: &str) -> Self {
        Self::new(name, 8, "u64")
    }

    /// Create a descriptor for bool.
    pub fn bool_column(name: &str) -> Self {
        Self::new(name, 1, "bool")
    }

    /// Create a descriptor for a vec3 (3 floats).
    pub fn vec3_column(name: &str) -> Self {
        Self::new(name, 12, "[f32; 3]")
    }

    /// Create a descriptor for a mat4 (16 floats).
    pub fn mat4_column(name: &str) -> Self {
        Self::new(name, 64, "[[f32; 4]; 4]")
    }
}

// ---------------------------------------------------------------------------
// Column Data
// ---------------------------------------------------------------------------

/// Raw column data storage.
#[derive(Debug)]
pub struct ColumnData {
    /// Raw byte storage.
    pub data: Vec<u8>,
    /// Element size in bytes.
    pub element_size: usize,
    /// Number of elements stored.
    pub count: usize,
    /// Capacity in elements.
    pub capacity: usize,
    /// Whether data has been modified.
    pub dirty: bool,
}

impl ColumnData {
    /// Create a new column with given element size and initial capacity.
    pub fn new(element_size: usize, capacity: usize) -> Self {
        let byte_capacity = element_size * capacity;
        Self {
            data: vec![0u8; byte_capacity],
            element_size,
            count: 0,
            capacity,
            dirty: false,
        }
    }

    /// Get a slice to the element at index.
    pub fn get(&self, index: usize) -> Option<&[u8]> {
        if index >= self.count {
            return None;
        }
        let offset = index * self.element_size;
        Some(&self.data[offset..offset + self.element_size])
    }

    /// Get a mutable slice to the element at index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut [u8]> {
        if index >= self.count {
            return None;
        }
        self.dirty = true;
        let offset = index * self.element_size;
        Some(&mut self.data[offset..offset + self.element_size])
    }

    /// Set the element at index from raw bytes.
    pub fn set(&mut self, index: usize, bytes: &[u8]) {
        if index >= self.count || bytes.len() != self.element_size {
            return;
        }
        let offset = index * self.element_size;
        self.data[offset..offset + self.element_size].copy_from_slice(bytes);
        self.dirty = true;
    }

    /// Append an element. Returns the index.
    pub fn push(&mut self, bytes: &[u8]) -> usize {
        if bytes.len() != self.element_size {
            return self.count;
        }

        if self.count >= self.capacity {
            self.grow();
        }

        let index = self.count;
        let offset = index * self.element_size;
        self.data[offset..offset + self.element_size].copy_from_slice(bytes);
        self.count += 1;
        self.dirty = true;
        index
    }

    /// Remove an element by swapping with the last.
    pub fn swap_remove(&mut self, index: usize) {
        if index >= self.count || self.count == 0 {
            return;
        }
        let last = self.count - 1;
        if index != last {
            let (src_offset, dst_offset) = (last * self.element_size, index * self.element_size);
            // Copy last element to the removed position.
            for i in 0..self.element_size {
                self.data[dst_offset + i] = self.data[src_offset + i];
            }
        }
        self.count -= 1;
        self.dirty = true;
    }

    /// Grow the capacity.
    fn grow(&mut self) {
        let new_capacity = (self.capacity as f32 * POOL_GROWTH_FACTOR) as usize + 1;
        let new_byte_cap = new_capacity * self.element_size;
        self.data.resize(new_byte_cap, 0);
        self.capacity = new_capacity;
    }

    /// Total memory used in bytes.
    pub fn memory_used(&self) -> usize {
        self.count * self.element_size
    }

    /// Total memory allocated in bytes.
    pub fn memory_allocated(&self) -> usize {
        self.data.len()
    }

    /// Get typed value at index (unsafe, caller must ensure correct type).
    pub fn get_typed<T: Copy>(&self, index: usize) -> Option<T> {
        if index >= self.count || std::mem::size_of::<T>() != self.element_size {
            return None;
        }
        let offset = index * self.element_size;
        let ptr = self.data[offset..].as_ptr() as *const T;
        Some(unsafe { *ptr })
    }

    /// Set typed value at index.
    pub fn set_typed<T: Copy>(&mut self, index: usize, value: T) {
        if index >= self.count || std::mem::size_of::<T>() != self.element_size {
            return;
        }
        let offset = index * self.element_size;
        let ptr = self.data[offset..].as_mut_ptr() as *mut T;
        unsafe { *ptr = value; }
        self.dirty = true;
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.count = 0;
        self.dirty = true;
    }

    /// Compact the storage to fit the current count.
    pub fn compact(&mut self) {
        let needed = self.count * self.element_size;
        self.data.truncate(needed);
        self.data.shrink_to_fit();
        self.capacity = self.count;
    }
}

// ---------------------------------------------------------------------------
// SoA Storage
// ---------------------------------------------------------------------------

/// Struct-of-arrays component storage.
#[derive(Debug)]
pub struct SoaStorage {
    /// Column descriptors.
    pub columns: Vec<ColumnDescriptor>,
    /// Column data arrays.
    pub column_data: Vec<ColumnData>,
    /// Entity-to-index mapping.
    pub entity_to_index: HashMap<StorageEntityId, usize>,
    /// Index-to-entity mapping (for reverse lookup).
    pub index_to_entity: Vec<StorageEntityId>,
    /// Number of stored entities.
    pub count: usize,
    /// Chunk size for iteration.
    pub chunk_size: usize,
    /// Whether the storage needs sorting.
    pub needs_sort: bool,
    /// Storage name (for debugging).
    pub name: String,
}

impl SoaStorage {
    /// Create a new SoA storage with the given columns.
    pub fn new(name: &str, columns: Vec<ColumnDescriptor>, initial_capacity: usize) -> Self {
        let capacity = initial_capacity.max(16);
        let mut col_descs = columns;
        let mut col_data = Vec::new();

        for (i, desc) in col_descs.iter_mut().enumerate() {
            desc.index = i;
            col_data.push(ColumnData::new(desc.element_size, capacity));
        }

        Self {
            columns: col_descs,
            column_data: col_data,
            entity_to_index: HashMap::with_capacity(capacity),
            index_to_entity: Vec::with_capacity(capacity),
            count: 0,
            chunk_size: DEFAULT_CHUNK_SIZE,
            needs_sort: false,
            name: name.to_string(),
        }
    }

    /// Add an entity. Returns the storage index.
    pub fn add_entity(&mut self, entity: StorageEntityId) -> Option<usize> {
        if self.entity_to_index.contains_key(&entity) {
            return None; // Already exists.
        }

        let index = self.count;
        self.entity_to_index.insert(entity, index);
        self.index_to_entity.push(entity);
        self.count += 1;
        self.needs_sort = true;

        // Ensure all columns have space (push zeroed element).
        for col in &mut self.column_data {
            let zeros = vec![0u8; col.element_size];
            col.push(&zeros);
        }

        Some(index)
    }

    /// Remove an entity by swap-removing.
    pub fn remove_entity(&mut self, entity: StorageEntityId) -> bool {
        let index = match self.entity_to_index.remove(&entity) {
            Some(idx) => idx,
            None => return false,
        };

        let last = self.count - 1;

        if index != last {
            // Update the swapped entity's mapping.
            let swapped_entity = self.index_to_entity[last];
            self.entity_to_index.insert(swapped_entity, index);
            self.index_to_entity[index] = swapped_entity;
        }

        self.index_to_entity.pop();
        self.count -= 1;

        // Swap-remove from all columns.
        for col in &mut self.column_data {
            col.swap_remove(index);
        }

        self.needs_sort = true;
        true
    }

    /// Check if an entity exists in storage.
    pub fn has_entity(&self, entity: StorageEntityId) -> bool {
        self.entity_to_index.contains_key(&entity)
    }

    /// Get the storage index for an entity.
    pub fn index_of(&self, entity: StorageEntityId) -> Option<usize> {
        self.entity_to_index.get(&entity).copied()
    }

    /// Get a column by name.
    pub fn column_by_name(&self, name: &str) -> Option<(usize, &ColumnData)> {
        self.columns.iter()
            .find(|c| c.name == name)
            .map(|c| (c.index, &self.column_data[c.index]))
    }

    /// Get a mutable column by name.
    pub fn column_by_name_mut(&mut self, name: &str) -> Option<(usize, &mut ColumnData)> {
        let idx = self.columns.iter().find(|c| c.name == name).map(|c| c.index)?;
        Some((idx, &mut self.column_data[idx]))
    }

    /// Get a column by index.
    pub fn column(&self, index: usize) -> Option<&ColumnData> {
        self.column_data.get(index)
    }

    /// Get a mutable column by index.
    pub fn column_mut(&mut self, index: usize) -> Option<&mut ColumnData> {
        self.column_data.get_mut(index)
    }

    /// Sort the storage by entity ID for deterministic iteration.
    pub fn sort_by_entity(&mut self) {
        if !self.needs_sort || self.count <= 1 {
            self.needs_sort = false;
            return;
        }

        // Build sorted index mapping.
        let mut indices: Vec<usize> = (0..self.count).collect();
        indices.sort_by_key(|&i| self.index_to_entity[i]);

        // Check if already sorted.
        let already_sorted = indices.iter().enumerate().all(|(i, &idx)| i == idx);
        if already_sorted {
            self.needs_sort = false;
            return;
        }

        // Reorder all columns according to sorted indices.
        for col in &mut self.column_data {
            let elem_size = col.element_size;
            let mut temp = vec![0u8; self.count * elem_size];
            for (new_idx, &old_idx) in indices.iter().enumerate() {
                let src = old_idx * elem_size;
                let dst = new_idx * elem_size;
                temp[dst..dst + elem_size].copy_from_slice(&col.data[src..src + elem_size]);
            }
            col.data[..self.count * elem_size].copy_from_slice(&temp);
        }

        // Reorder entity mapping.
        let mut new_entities = vec![StorageEntityId(0); self.count];
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            new_entities[new_idx] = self.index_to_entity[old_idx];
        }
        self.index_to_entity = new_entities;

        // Rebuild reverse mapping.
        self.entity_to_index.clear();
        for (idx, &entity) in self.index_to_entity.iter().enumerate() {
            self.entity_to_index.insert(entity, idx);
        }

        self.needs_sort = false;
    }

    /// Iterate over chunks of entities.
    pub fn chunks(&self) -> ChunkIterator<'_> {
        ChunkIterator {
            storage: self,
            offset: 0,
        }
    }

    /// Total number of entities.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether storage is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Number of columns.
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Total memory used across all columns.
    pub fn memory_used(&self) -> usize {
        self.column_data.iter().map(|c| c.memory_used()).sum()
    }

    /// Total memory allocated.
    pub fn memory_allocated(&self) -> usize {
        self.column_data.iter().map(|c| c.memory_allocated()).sum()
    }

    /// Compact all columns.
    pub fn compact(&mut self) {
        for col in &mut self.column_data {
            col.compact();
        }
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        for col in &mut self.column_data {
            col.clear();
        }
        self.entity_to_index.clear();
        self.index_to_entity.clear();
        self.count = 0;
    }

    /// Get storage statistics.
    pub fn stats(&self) -> StorageStats {
        StorageStats {
            entity_count: self.count,
            column_count: self.columns.len(),
            memory_used: self.memory_used(),
            memory_allocated: self.memory_allocated(),
            chunk_size: self.chunk_size,
            sorted: !self.needs_sort,
            fragmentation: if self.memory_allocated() > 0 {
                1.0 - self.memory_used() as f32 / self.memory_allocated() as f32
            } else {
                0.0
            },
        }
    }
}

/// Chunk iterator for SoA storage.
#[derive(Debug)]
pub struct ChunkIterator<'a> {
    storage: &'a SoaStorage,
    offset: usize,
}

impl<'a> Iterator for ChunkIterator<'a> {
    type Item = ChunkView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.storage.count {
            return None;
        }

        let start = self.offset;
        let end = (start + self.storage.chunk_size).min(self.storage.count);
        self.offset = end;

        Some(ChunkView {
            storage: self.storage,
            start,
            end,
        })
    }
}

/// A view into a chunk of the SoA storage.
#[derive(Debug)]
pub struct ChunkView<'a> {
    storage: &'a SoaStorage,
    /// Start index (inclusive).
    pub start: usize,
    /// End index (exclusive).
    pub end: usize,
}

impl<'a> ChunkView<'a> {
    /// Number of entities in this chunk.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether the chunk is empty.
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Get entity ID at chunk-local index.
    pub fn entity_at(&self, local_index: usize) -> Option<StorageEntityId> {
        let global_index = self.start + local_index;
        self.storage.index_to_entity.get(global_index).copied()
    }

    /// Get column data slice for this chunk.
    pub fn column_slice(&self, column_index: usize) -> Option<&'a [u8]> {
        let col = self.storage.column_data.get(column_index)?;
        let start_byte = self.start * col.element_size;
        let end_byte = self.end * col.element_size;
        Some(&col.data[start_byte..end_byte])
    }
}

/// Storage statistics.
#[derive(Debug, Clone)]
pub struct StorageStats {
    /// Number of entities.
    pub entity_count: usize,
    /// Number of columns.
    pub column_count: usize,
    /// Memory used by actual data.
    pub memory_used: usize,
    /// Memory allocated (may be larger due to capacity).
    pub memory_allocated: usize,
    /// Chunk size for iteration.
    pub chunk_size: usize,
    /// Whether the storage is sorted by entity ID.
    pub sorted: bool,
    /// Fragmentation ratio (0.0 = fully compact, 1.0 = all overhead).
    pub fragmentation: f32,
}
