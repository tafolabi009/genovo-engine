//! Generational handle system.
//!
//! Handles provide type-safe, generation-checked references into dense storage.
//! When an entry is freed its generation is bumped so that any outstanding
//! handles to the old entry are detected as stale, preventing use-after-free
//! bugs without runtime borrow-checking overhead.

use std::marker::PhantomData;

use crate::error::{EngineError, EngineResult};

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

/// A lightweight, type-safe reference into a [`HandlePool`].
///
/// `T` is a phantom type tag that prevents mixing handles from different pools
/// (e.g., a mesh handle cannot be used where a texture handle is expected).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Handle<T> {
    /// Slot index inside the pool's dense storage.
    index: u32,
    /// Generation counter; must match the pool entry to be valid.
    generation: u32,
    /// Zero-sized marker so the type system distinguishes handle kinds.
    _marker: PhantomData<T>,
}

impl<T> Handle<T> {
    /// Creates a new handle with the given index and generation.
    ///
    /// # Safety (logical)
    ///
    /// Callers must ensure that the index and generation correspond to a valid
    /// entry in a [`HandlePool`]. This is exposed publicly so that sibling
    /// crates (e.g. genovo-render) can reconstruct handles from packed
    /// representations.
    pub fn new(index: u32, generation: u32) -> Self {
        Self {
            index,
            generation,
            _marker: PhantomData,
        }
    }

    /// Returns the slot index.
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Returns the generation counter.
    #[inline]
    pub fn generation(&self) -> u32 {
        self.generation
    }
}

// ---------------------------------------------------------------------------
// Pool entry
// ---------------------------------------------------------------------------

/// Internal bookkeeping for one slot in the pool.
struct PoolEntry<T> {
    /// Current generation of this slot. Incremented on every free.
    generation: u32,
    /// `Some` when the slot is occupied, `None` when free.
    value: Option<T>,
}

// ---------------------------------------------------------------------------
// HandlePool
// ---------------------------------------------------------------------------

/// A generational arena that pairs dense storage with [`Handle`]-based access.
///
/// # Example (conceptual)
///
/// ```ignore
/// let mut pool: HandlePool<Mesh> = HandlePool::new();
/// let h = pool.insert(my_mesh);
/// assert!(pool.get(h).is_some());
/// pool.remove(h);
/// assert!(pool.get(h).is_none()); // generation mismatch
/// ```
pub struct HandlePool<T> {
    /// Dense array of entries (occupied or free).
    entries: Vec<PoolEntry<T>>,
    /// Stack of free-list indices available for reuse.
    free_list: Vec<u32>,
}

impl<T> HandlePool<T> {
    /// Creates an empty pool.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            free_list: Vec::new(),
        }
    }

    /// Creates a pool pre-allocated for `capacity` entries.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            free_list: Vec::new(),
        }
    }

    /// Inserts `value` into the pool and returns a handle to it.
    pub fn insert(&mut self, value: T) -> Handle<T> {
        // Reclaim slots from the free list before allocating new ones.
        if let Some(free_index) = self.free_list.pop() {
            let entry = &mut self.entries[free_index as usize];
            entry.value = Some(value);
            Handle::new(free_index, entry.generation)
        } else {
            let index = self.entries.len() as u32;
            self.entries.push(PoolEntry {
                generation: 0,
                value: Some(value),
            });
            Handle::new(index, 0)
        }
    }

    /// Removes the value referenced by `handle`, returning it if the handle
    /// was valid.
    pub fn remove(&mut self, handle: Handle<T>) -> EngineResult<T> {
        let entry = self
            .entries
            .get_mut(handle.index() as usize)
            .ok_or_else(|| EngineError::InvalidHandle {
                index: handle.index(),
                generation: handle.generation(),
            })?;

        if entry.generation != handle.generation() {
            return Err(EngineError::InvalidHandle {
                index: handle.index(),
                generation: handle.generation(),
            });
        }

        let value = entry.value.take().ok_or_else(|| EngineError::InvalidHandle {
            index: handle.index(),
            generation: handle.generation(),
        })?;

        // Bump generation so old handles are invalidated.
        entry.generation = entry.generation.wrapping_add(1);
        self.free_list.push(handle.index());
        Ok(value)
    }

    /// Returns a shared reference to the value if the handle is still valid.
    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        let entry = self.entries.get(handle.index() as usize)?;
        if entry.generation != handle.generation() {
            return None;
        }
        entry.value.as_ref()
    }

    /// Returns an exclusive reference to the value if the handle is still valid.
    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        let entry = self.entries.get_mut(handle.index() as usize)?;
        if entry.generation != handle.generation() {
            return None;
        }
        entry.value.as_mut()
    }

    /// Returns `true` if the handle points to a live entry.
    pub fn is_valid(&self, handle: Handle<T>) -> bool {
        self.get(handle).is_some()
    }

    /// Returns the number of currently occupied slots.
    pub fn len(&self) -> usize {
        self.entries.len() - self.free_list.len()
    }

    /// Returns `true` when no slots are occupied.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Default for HandlePool<T> {
    fn default() -> Self {
        Self::new()
    }
}
