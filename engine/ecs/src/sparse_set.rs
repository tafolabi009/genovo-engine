//! Optimized sparse set for cache-friendly component storage in the ECS.
//!
//! This implementation uses page-based sparse arrays to avoid huge contiguous
//! allocations while maintaining O(1) lookup. It supports column iteration,
//! sorted iteration, component moves between entities, and bulk operations.
//!
//! # Architecture
//!
//! ```text
//! Sparse Array (paged):
//!   Page 0: [_, _, 0, _, 1, _, _, _]   <- maps entity index to dense index
//!   Page 1: [_, _, _, 2, _, _, _, _]
//!   ...
//!
//! Dense Array:
//!   [entity_2, entity_4, entity_11]     <- entity IDs in insertion order
//!
//! Data Array:
//!   [comp_2, comp_4, comp_11]           <- component data, parallel to dense
//! ```

use std::fmt;
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of entries per page in the sparse array.
const PAGE_SIZE: usize = 4096;

/// Sentinel value indicating an empty slot in the sparse array.
const EMPTY: u32 = u32::MAX;

// ---------------------------------------------------------------------------
// SparsePage
// ---------------------------------------------------------------------------

/// A single page of the sparse array, allocated lazily.
struct SparsePage {
    /// Maps local index (entity_index % PAGE_SIZE) to dense index.
    entries: Box<[u32; PAGE_SIZE]>,
}

impl SparsePage {
    fn new() -> Self {
        Self {
            entries: Box::new([EMPTY; PAGE_SIZE]),
        }
    }

    fn get(&self, local: usize) -> Option<u32> {
        let val = self.entries[local];
        if val == EMPTY {
            None
        } else {
            Some(val)
        }
    }

    fn set(&mut self, local: usize, dense_index: u32) {
        self.entries[local] = dense_index;
    }

    fn clear(&mut self, local: usize) {
        self.entries[local] = EMPTY;
    }
}

// ---------------------------------------------------------------------------
// SparseSetV2
// ---------------------------------------------------------------------------

/// A page-based sparse set providing O(1) insert, remove, and lookup of
/// components keyed by entity index (u32).
///
/// The data array is kept tightly packed for efficient iteration.
pub struct SparseSetV2<T> {
    /// Paged sparse array: maps entity index -> dense index.
    sparse: Vec<Option<Box<SparsePage>>>,
    /// Dense array: entity indices in insertion order.
    dense: Vec<u32>,
    /// Data array: component values, parallel to `dense`.
    data: Vec<T>,
    /// Generation counter for cache invalidation.
    generation: u64,
}

impl<T> SparseSetV2<T> {
    /// Create a new, empty sparse set.
    pub fn new() -> Self {
        Self {
            sparse: Vec::new(),
            dense: Vec::new(),
            data: Vec::new(),
            generation: 0,
        }
    }

    /// Create with a pre-allocated capacity for the dense array.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            sparse: Vec::new(),
            dense: Vec::with_capacity(capacity),
            data: Vec::with_capacity(capacity),
            generation: 0,
        }
    }

    // -- Page management --

    fn page_index(entity: u32) -> usize {
        entity as usize / PAGE_SIZE
    }

    fn local_index(entity: u32) -> usize {
        entity as usize % PAGE_SIZE
    }

    fn ensure_page(&mut self, page_idx: usize) {
        if page_idx >= self.sparse.len() {
            self.sparse.resize_with(page_idx + 1, || None);
        }
        if self.sparse[page_idx].is_none() {
            self.sparse[page_idx] = Some(Box::new(SparsePage::new()));
        }
    }

    fn get_dense_index(&self, entity: u32) -> Option<u32> {
        let page_idx = Self::page_index(entity);
        let local = Self::local_index(entity);
        self.sparse
            .get(page_idx)
            .and_then(|p| p.as_ref())
            .and_then(|p| p.get(local))
    }

    // -- Core operations --

    /// Insert a component for the given entity. If the entity already has a
    /// component, it is replaced and the old one is returned.
    pub fn insert(&mut self, entity: u32, value: T) -> Option<T> {
        if let Some(dense_idx) = self.get_dense_index(entity) {
            // Replace existing.
            let old = std::mem::replace(&mut self.data[dense_idx as usize], value);
            self.generation += 1;
            Some(old)
        } else {
            // New entry.
            let dense_idx = self.dense.len() as u32;
            let page_idx = Self::page_index(entity);
            let local = Self::local_index(entity);

            self.ensure_page(page_idx);
            self.sparse[page_idx].as_mut().unwrap().set(local, dense_idx);

            self.dense.push(entity);
            self.data.push(value);
            self.generation += 1;
            None
        }
    }

    /// Remove a component for the given entity. Returns the removed value.
    pub fn remove(&mut self, entity: u32) -> Option<T> {
        let dense_idx = self.get_dense_index(entity)? as usize;
        let page_idx = Self::page_index(entity);
        let local = Self::local_index(entity);

        // Clear the sparse entry.
        if let Some(ref mut page) = self.sparse[page_idx] {
            page.clear(local);
        }

        // Swap-remove from dense + data.
        let last_dense = self.dense.len() - 1;
        let removed_value;

        if dense_idx < last_dense {
            // Swap with last.
            let last_entity = self.dense[last_dense];
            self.dense[dense_idx] = last_entity;
            self.data.swap(dense_idx, last_dense);

            // Update the swapped entity's sparse entry.
            let last_page = Self::page_index(last_entity);
            let last_local = Self::local_index(last_entity);
            self.sparse[last_page]
                .as_mut()
                .unwrap()
                .set(last_local, dense_idx as u32);

            removed_value = self.data.pop().unwrap();
            self.dense.pop();
        } else {
            removed_value = self.data.pop().unwrap();
            self.dense.pop();
        }

        self.generation += 1;
        Some(removed_value)
    }

    /// Check if the entity has a component in this set.
    pub fn contains(&self, entity: u32) -> bool {
        self.get_dense_index(entity).is_some()
    }

    /// Get a reference to the component for the given entity.
    pub fn get(&self, entity: u32) -> Option<&T> {
        let dense_idx = self.get_dense_index(entity)? as usize;
        self.data.get(dense_idx)
    }

    /// Get a mutable reference to the component for the given entity.
    pub fn get_mut(&mut self, entity: u32) -> Option<&mut T> {
        let dense_idx = self.get_dense_index(entity)? as usize;
        self.data.get_mut(dense_idx)
    }

    /// Returns the number of components stored.
    pub fn len(&self) -> usize {
        self.dense.len()
    }

    /// Returns `true` if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    /// Clear all components.
    pub fn clear(&mut self) {
        for page in &mut self.sparse {
            *page = None;
        }
        self.dense.clear();
        self.data.clear();
        self.generation += 1;
    }

    /// Returns the current generation (incremented on structural changes).
    pub fn generation(&self) -> u64 {
        self.generation
    }

    // -- Iteration --

    /// Returns a slice of all entity indices (in insertion order).
    pub fn entities(&self) -> &[u32] {
        &self.dense
    }

    /// Returns a slice of all component data (parallel to entities).
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Returns a mutable slice of all component data.
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Iterate over (entity, &component) pairs.
    pub fn iter(&self) -> SparseSetIter<'_, T> {
        SparseSetIter {
            dense_iter: self.dense.iter(),
            data_iter: self.data.iter(),
        }
    }

    /// Iterate over (entity, &mut component) pairs.
    pub fn iter_mut(&mut self) -> SparseSetIterMut<'_, T> {
        SparseSetIterMut {
            dense_iter: self.dense.iter(),
            data_iter: self.data.iter_mut(),
        }
    }

    // -- Bulk operations --

    /// Retain only entities matching a predicate.
    pub fn retain<F: FnMut(u32, &T) -> bool>(&mut self, mut f: F) {
        let mut i = 0;
        while i < self.dense.len() {
            let entity = self.dense[i];
            if f(entity, &self.data[i]) {
                i += 1;
            } else {
                self.remove(entity);
                // After remove, position i now holds a different entity
                // (swap-removed from end), so don't increment.
            }
        }
    }

    /// Apply a function to all components.
    pub fn for_each<F: FnMut(u32, &mut T)>(&mut self, mut f: F) {
        for i in 0..self.dense.len() {
            f(self.dense[i], &mut self.data[i]);
        }
    }

    /// Move a component from one entity to another.
    /// Returns `true` if the move was successful.
    pub fn move_component(&mut self, from: u32, to: u32) -> bool {
        if !self.contains(from) || self.contains(to) {
            return false;
        }

        if let Some(dense_idx) = self.get_dense_index(from) {
            // Update sparse: clear old, set new.
            let from_page = Self::page_index(from);
            let from_local = Self::local_index(from);
            if let Some(ref mut page) = self.sparse[from_page] {
                page.clear(from_local);
            }

            let to_page = Self::page_index(to);
            let to_local = Self::local_index(to);
            self.ensure_page(to_page);
            self.sparse[to_page]
                .as_mut()
                .unwrap()
                .set(to_local, dense_idx);

            // Update dense.
            self.dense[dense_idx as usize] = to;
            self.generation += 1;
            true
        } else {
            false
        }
    }

    /// Swap components between two entities (both must exist).
    pub fn swap_components(&mut self, a: u32, b: u32) -> bool {
        let dense_a = match self.get_dense_index(a) {
            Some(d) => d as usize,
            None => return false,
        };
        let dense_b = match self.get_dense_index(b) {
            Some(d) => d as usize,
            None => return false,
        };

        // Swap data.
        self.data.swap(dense_a, dense_b);

        // Swap dense entries.
        self.dense.swap(dense_a, dense_b);

        // Update sparse pointers.
        let page_a = Self::page_index(a);
        let local_a = Self::local_index(a);
        let page_b = Self::page_index(b);
        let local_b = Self::local_index(b);

        self.sparse[page_a]
            .as_mut()
            .unwrap()
            .set(local_a, dense_b as u32);
        self.sparse[page_b]
            .as_mut()
            .unwrap()
            .set(local_b, dense_a as u32);

        self.generation += 1;
        true
    }

    /// Number of allocated pages in the sparse array.
    pub fn page_count(&self) -> usize {
        self.sparse.iter().filter(|p| p.is_some()).count()
    }

    /// Total memory used by the sparse pages (approximate).
    pub fn sparse_memory_bytes(&self) -> usize {
        self.page_count() * PAGE_SIZE * std::mem::size_of::<u32>()
    }

    /// Total memory used by the dense + data arrays (approximate).
    pub fn dense_memory_bytes(&self) -> usize {
        self.dense.capacity() * std::mem::size_of::<u32>()
            + self.data.capacity() * std::mem::size_of::<T>()
    }
}

impl<T: Clone> SparseSetV2<T> {
    /// Clone all components from another set into this one.
    pub fn clone_from_set(&mut self, other: &SparseSetV2<T>) {
        self.clear();
        for (entity, data) in other.iter() {
            self.insert(entity, data.clone());
        }
    }
}

impl<T: Ord> SparseSetV2<T> {
    /// Sort the dense array by component value.
    /// After sorting, iteration visits components in sorted order.
    pub fn sort(&mut self) {
        // Build index permutation.
        let mut indices: Vec<usize> = (0..self.dense.len()).collect();
        indices.sort_by(|&a, &b| self.data[a].cmp(&self.data[b]));

        self.apply_permutation(&indices);
    }
}

impl<T> SparseSetV2<T> {
    /// Sort by a key function.
    pub fn sort_by_key<K: Ord, F: FnMut(&T) -> K>(&mut self, mut f: F) {
        let mut indices: Vec<usize> = (0..self.dense.len()).collect();
        indices.sort_by_key(|&i| f(&self.data[i]));
        self.apply_permutation(&indices);
    }

    /// Sort by entity index (ascending).
    pub fn sort_by_entity(&mut self) {
        let mut indices: Vec<usize> = (0..self.dense.len()).collect();
        indices.sort_by_key(|&i| self.dense[i]);
        self.apply_permutation(&indices);
    }

    fn apply_permutation(&mut self, perm: &[usize]) {
        let n = perm.len();
        let mut new_dense = Vec::with_capacity(n);
        let mut new_data = Vec::with_capacity(n);

        // We need to use unsafe or a temporary Vec because we can't move
        // out of a Vec by index without leaving holes. Use a workaround
        // with Option.
        let mut opt_data: Vec<Option<T>> = self.data.drain(..).map(Some).collect();

        for &idx in perm {
            new_dense.push(self.dense[idx]);
            new_data.push(opt_data[idx].take().unwrap());
        }

        self.dense = new_dense;
        self.data = new_data;

        // Rebuild sparse pointers.
        for (new_idx, &entity) in self.dense.iter().enumerate() {
            let page_idx = Self::page_index(entity);
            let local = Self::local_index(entity);
            self.sparse[page_idx]
                .as_mut()
                .unwrap()
                .set(local, new_idx as u32);
        }

        self.generation += 1;
    }
}

impl<T> Default for SparseSetV2<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Debug> fmt::Debug for SparseSetV2<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseSetV2")
            .field("len", &self.len())
            .field("pages", &self.page_count())
            .field("generation", &self.generation)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Iterators
// ---------------------------------------------------------------------------

/// Iterator over (entity_index, &component) pairs.
pub struct SparseSetIter<'a, T> {
    dense_iter: std::slice::Iter<'a, u32>,
    data_iter: std::slice::Iter<'a, T>,
}

impl<'a, T> Iterator for SparseSetIter<'a, T> {
    type Item = (u32, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        match (self.dense_iter.next(), self.data_iter.next()) {
            (Some(&entity), Some(data)) => Some((entity, data)),
            _ => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.dense_iter.size_hint()
    }
}

impl<'a, T> ExactSizeIterator for SparseSetIter<'a, T> {}

/// Iterator over (entity_index, &mut component) pairs.
pub struct SparseSetIterMut<'a, T> {
    dense_iter: std::slice::Iter<'a, u32>,
    data_iter: std::iter::IterMut<'a, T>,
}

impl<'a, T> Iterator for SparseSetIterMut<'a, T> {
    type Item = (u32, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        match (self.dense_iter.next(), self.data_iter.next()) {
            (Some(&entity), Some(data)) => Some((entity, data)),
            _ => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.dense_iter.size_hint()
    }
}

impl<'a, T> ExactSizeIterator for SparseSetIterMut<'a, T> {}

// ---------------------------------------------------------------------------
// MultiSparseSetView (read multiple sets simultaneously)
// ---------------------------------------------------------------------------

/// A view that allows iterating entities present in multiple sparse sets.
pub struct IntersectionView<'a, A, B> {
    set_a: &'a SparseSetV2<A>,
    set_b: &'a SparseSetV2<B>,
}

impl<'a, A, B> IntersectionView<'a, A, B> {
    /// Create an intersection view over two sparse sets.
    pub fn new(set_a: &'a SparseSetV2<A>, set_b: &'a SparseSetV2<B>) -> Self {
        Self { set_a, set_b }
    }

    /// Iterate over entities present in both sets.
    pub fn iter(&self) -> IntersectionIter<'a, A, B> {
        // Iterate the smaller set and look up in the larger one.
        let (primary_entities, primary_data) = if self.set_a.len() <= self.set_b.len() {
            (self.set_a.entities(), self.set_a.data())
        } else {
            // We swap below in the iterator.
            (self.set_a.entities(), self.set_a.data())
        };

        IntersectionIter {
            set_a: self.set_a,
            set_b: self.set_b,
            index: 0,
        }
    }

    /// Count entities present in both sets.
    pub fn count(&self) -> usize {
        let smaller = if self.set_a.len() <= self.set_b.len() {
            self.set_a
        } else {
            // Need to check both directions; just iterate smaller.
            self.set_a
        };

        let mut count = 0;
        for &entity in self.set_a.entities() {
            if self.set_b.contains(entity) {
                count += 1;
            }
        }
        count
    }
}

/// Iterator for the intersection of two sparse sets.
pub struct IntersectionIter<'a, A, B> {
    set_a: &'a SparseSetV2<A>,
    set_b: &'a SparseSetV2<B>,
    index: usize,
}

impl<'a, A, B> Iterator for IntersectionIter<'a, A, B> {
    type Item = (u32, &'a A, &'a B);

    fn next(&mut self) -> Option<Self::Item> {
        let entities = self.set_a.entities();
        while self.index < entities.len() {
            let entity = entities[self.index];
            self.index += 1;

            if let (Some(a), Some(b)) = (self.set_a.get(entity), self.set_b.get(entity)) {
                return Some((entity, a, b));
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// SparseSetStats
// ---------------------------------------------------------------------------

/// Statistics about a sparse set.
#[derive(Debug, Clone)]
pub struct SparseSetStats {
    /// Number of stored components.
    pub count: usize,
    /// Number of allocated pages.
    pub pages: usize,
    /// Approximate sparse memory (bytes).
    pub sparse_memory: usize,
    /// Approximate dense memory (bytes).
    pub dense_memory: usize,
    /// Total approximate memory (bytes).
    pub total_memory: usize,
    /// Current generation.
    pub generation: u64,
}

impl<T> SparseSetV2<T> {
    /// Compute statistics for this sparse set.
    pub fn stats(&self) -> SparseSetStats {
        let sparse_mem = self.sparse_memory_bytes();
        let dense_mem = self.dense_memory_bytes();
        SparseSetStats {
            count: self.len(),
            pages: self.page_count(),
            sparse_memory: sparse_mem,
            dense_memory: dense_mem,
            total_memory: sparse_mem + dense_mem,
            generation: self.generation,
        }
    }
}

impl fmt::Display for SparseSetStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SparseSet Statistics:")?;
        writeln!(f, "  count:         {}", self.count)?;
        writeln!(f, "  pages:         {}", self.pages)?;
        writeln!(f, "  sparse memory: {} KB", self.sparse_memory / 1024)?;
        writeln!(f, "  dense memory:  {} KB", self.dense_memory / 1024)?;
        writeln!(f, "  total memory:  {} KB", self.total_memory / 1024)?;
        writeln!(f, "  generation:    {}", self.generation)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let mut set = SparseSetV2::new();
        set.insert(5, "hello");
        set.insert(10, "world");
        set.insert(3, "foo");

        assert_eq!(set.get(5), Some(&"hello"));
        assert_eq!(set.get(10), Some(&"world"));
        assert_eq!(set.get(3), Some(&"foo"));
        assert_eq!(set.get(7), None);
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_replace() {
        let mut set = SparseSetV2::new();
        assert_eq!(set.insert(5, 100), None);
        assert_eq!(set.insert(5, 200), Some(100));
        assert_eq!(set.get(5), Some(&200));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_remove() {
        let mut set = SparseSetV2::new();
        set.insert(1, "a");
        set.insert(2, "b");
        set.insert(3, "c");

        assert_eq!(set.remove(2), Some("b"));
        assert_eq!(set.len(), 2);
        assert!(!set.contains(2));
        assert!(set.contains(1));
        assert!(set.contains(3));
    }

    #[test]
    fn test_large_entity_ids() {
        let mut set = SparseSetV2::new();
        set.insert(100_000, "far");
        set.insert(0, "near");

        assert_eq!(set.get(100_000), Some(&"far"));
        assert_eq!(set.get(0), Some(&"near"));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_iteration() {
        let mut set = SparseSetV2::new();
        set.insert(10, 100);
        set.insert(20, 200);
        set.insert(30, 300);

        let mut collected: Vec<(u32, i32)> = set.iter().map(|(e, &v)| (e, v)).collect();
        collected.sort_by_key(|&(e, _)| e);
        assert_eq!(collected, vec![(10, 100), (20, 200), (30, 300)]);
    }

    #[test]
    fn test_move_component() {
        let mut set = SparseSetV2::new();
        set.insert(5, "data");

        assert!(set.move_component(5, 10));
        assert!(!set.contains(5));
        assert_eq!(set.get(10), Some(&"data"));
    }

    #[test]
    fn test_swap_components() {
        let mut set = SparseSetV2::new();
        set.insert(1, "alpha");
        set.insert(2, "beta");

        assert!(set.swap_components(1, 2));
        assert_eq!(set.get(1), Some(&"beta"));
        assert_eq!(set.get(2), Some(&"alpha"));
    }

    #[test]
    fn test_sort_by_entity() {
        let mut set = SparseSetV2::new();
        set.insert(30, "c");
        set.insert(10, "a");
        set.insert(20, "b");

        set.sort_by_entity();
        let entities: Vec<u32> = set.entities().to_vec();
        assert_eq!(entities, vec![10, 20, 30]);
    }

    #[test]
    fn test_retain() {
        let mut set = SparseSetV2::new();
        set.insert(1, 10);
        set.insert(2, 20);
        set.insert(3, 30);
        set.insert(4, 40);

        set.retain(|_, &v| v > 15);
        assert_eq!(set.len(), 3);
        assert!(!set.contains(1));
    }

    #[test]
    fn test_intersection() {
        let mut set_a = SparseSetV2::new();
        set_a.insert(1, "a1");
        set_a.insert(2, "a2");
        set_a.insert(3, "a3");

        let mut set_b = SparseSetV2::new();
        set_b.insert(2, 200);
        set_b.insert(3, 300);
        set_b.insert(4, 400);

        let view = IntersectionView::new(&set_a, &set_b);
        let results: Vec<_> = view.iter().collect();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_clear() {
        let mut set = SparseSetV2::new();
        set.insert(1, 10);
        set.insert(2, 20);
        set.clear();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn test_stats() {
        let mut set = SparseSetV2::new();
        for i in 0..100 {
            set.insert(i, i * 10);
        }
        let stats = set.stats();
        assert_eq!(stats.count, 100);
        assert!(stats.total_memory > 0);
    }
}
