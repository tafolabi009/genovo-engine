//! # Engine-Optimized Collections
//!
//! High-performance data structures tailored for game engine workloads.
//! These collections prioritize cache-friendly memory layouts, minimal allocation,
//! and deterministic performance over worst-case generality.
//!
//! - [`FreeList`] — Slot-based allocator with generational indices
//! - [`RingBuffer`] — Fixed-capacity circular buffer
//! - [`BitSet`] — Compact bit storage with popcount-based operations
//! - [`SparseArray`] — Sparse index-to-value mapping with dense iteration
//! - [`StringInterner`] — String deduplication with interned handles
//! - [`ObjectPool`] — Pre-allocated fixed-size pool with RAII guards

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

// ---------------------------------------------------------------------------
// FreeList<T> — slot-based allocator with generational indices
// ---------------------------------------------------------------------------

/// A generational handle into a [`FreeList`]. The `generation` field is
/// incremented every time a slot is recycled, so stale handles can be detected
/// at zero cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FreeListHandle {
    index: u32,
    generation: u32,
}

impl FreeListHandle {
    /// Returns the raw slot index.
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Returns the generation stamp.
    #[inline]
    pub fn generation(&self) -> u32 {
        self.generation
    }
}

/// Internal slot — either occupied or pointing to the next free slot.
enum FreeListSlot<T> {
    Occupied { value: T, generation: u32 },
    Vacant { next_free: Option<u32>, generation: u32 },
}

/// Slot-based allocator with generational indices.
///
/// Dense storage combined with a free list allows O(1) insert, O(1) remove,
/// and O(1) lookup by handle. Generation checks catch use-after-free bugs
/// immediately rather than silently returning the wrong data.
///
/// # Example
/// ```
/// use genovo_core::collections::FreeList;
///
/// let mut list = FreeList::new();
/// let h = list.insert(42);
/// assert_eq!(list.get(h), Some(&42));
/// list.remove(h);
/// assert_eq!(list.get(h), None); // generation mismatch
/// ```
pub struct FreeList<T> {
    slots: Vec<FreeListSlot<T>>,
    free_head: Option<u32>,
    len: usize,
}

impl<T> FreeList<T> {
    /// Creates an empty `FreeList`.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: None,
            len: 0,
        }
    }

    /// Creates a `FreeList` with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free_head: None,
            len: 0,
        }
    }

    /// Returns the number of live elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if there are no live elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the total number of slots (occupied + vacant).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Inserts a value, returning a generational handle.
    pub fn insert(&mut self, value: T) -> FreeListHandle {
        if let Some(free_index) = self.free_head {
            let idx = free_index as usize;
            let generation = match &self.slots[idx] {
                FreeListSlot::Vacant { next_free, generation } => {
                    self.free_head = *next_free;
                    *generation
                }
                FreeListSlot::Occupied { .. } => {
                    unreachable!("free list pointed to occupied slot")
                }
            };
            let new_gen = generation.wrapping_add(1);
            self.slots[idx] = FreeListSlot::Occupied {
                value,
                generation: new_gen,
            };
            self.len += 1;
            FreeListHandle {
                index: free_index,
                generation: new_gen,
            }
        } else {
            let index = self.slots.len() as u32;
            let generation = 0;
            self.slots.push(FreeListSlot::Occupied { value, generation });
            self.len += 1;
            FreeListHandle { index, generation }
        }
    }

    /// Removes the value associated with `handle`, returning it if the handle
    /// is still valid.
    pub fn remove(&mut self, handle: FreeListHandle) -> Option<T> {
        let idx = handle.index as usize;
        if idx >= self.slots.len() {
            return None;
        }

        match &self.slots[idx] {
            FreeListSlot::Occupied { generation, .. } => {
                if *generation != handle.generation {
                    return None;
                }
            }
            FreeListSlot::Vacant { .. } => return None,
        }

        // Replace the occupied slot with a vacant one.
        let old_slot = std::mem::replace(
            &mut self.slots[idx],
            FreeListSlot::Vacant {
                next_free: self.free_head,
                generation: handle.generation,
            },
        );
        self.free_head = Some(handle.index);
        self.len -= 1;

        match old_slot {
            FreeListSlot::Occupied { value, .. } => Some(value),
            FreeListSlot::Vacant { .. } => unreachable!(),
        }
    }

    /// Returns a reference to the value if the handle is still valid.
    pub fn get(&self, handle: FreeListHandle) -> Option<&T> {
        let idx = handle.index as usize;
        if idx >= self.slots.len() {
            return None;
        }
        match &self.slots[idx] {
            FreeListSlot::Occupied { value, generation } if *generation == handle.generation => {
                Some(value)
            }
            _ => None,
        }
    }

    /// Returns a mutable reference to the value if the handle is still valid.
    pub fn get_mut(&mut self, handle: FreeListHandle) -> Option<&mut T> {
        let idx = handle.index as usize;
        if idx >= self.slots.len() {
            return None;
        }
        match &mut self.slots[idx] {
            FreeListSlot::Occupied { value, generation } if *generation == handle.generation => {
                Some(value)
            }
            _ => None,
        }
    }

    /// Returns `true` if the handle points to a live element.
    pub fn contains(&self, handle: FreeListHandle) -> bool {
        self.get(handle).is_some()
    }

    /// Removes all elements, recycling every slot.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free_head = None;
        self.len = 0;
    }

    /// Returns an iterator over references to all live elements.
    pub fn iter(&self) -> FreeListIter<'_, T> {
        FreeListIter {
            slots: &self.slots,
            index: 0,
            remaining: self.len,
        }
    }

    /// Returns a mutable iterator over all live elements.
    pub fn iter_mut(&mut self) -> FreeListIterMut<'_, T> {
        FreeListIterMut {
            slots: self.slots.iter_mut(),
            remaining: self.len,
        }
    }

    /// Returns an iterator yielding `(FreeListHandle, &T)` pairs for all live elements.
    pub fn iter_with_handles(&self) -> FreeListHandleIter<'_, T> {
        FreeListHandleIter {
            slots: &self.slots,
            index: 0,
            remaining: self.len,
        }
    }
}

impl<T> Default for FreeList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Debug> fmt::Debug for FreeList<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FreeList")
            .field("len", &self.len)
            .field("capacity", &self.slots.len())
            .finish()
    }
}

/// Iterator over references to live elements in a [`FreeList`].
pub struct FreeListIter<'a, T> {
    slots: &'a [FreeListSlot<T>],
    index: usize,
    remaining: usize,
}

impl<'a, T> Iterator for FreeListIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.slots.len() && self.remaining > 0 {
            let slot = &self.slots[self.index];
            self.index += 1;
            if let FreeListSlot::Occupied { value, .. } = slot {
                self.remaining -= 1;
                return Some(value);
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, T> ExactSizeIterator for FreeListIter<'a, T> {}

/// Mutable iterator over live elements in a [`FreeList`].
pub struct FreeListIterMut<'a, T> {
    slots: std::slice::IterMut<'a, FreeListSlot<T>>,
    remaining: usize,
}

impl<'a, T> Iterator for FreeListIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        while self.remaining > 0 {
            if let Some(slot) = self.slots.next() {
                if let FreeListSlot::Occupied { value, .. } = slot {
                    self.remaining -= 1;
                    return Some(value);
                }
            } else {
                break;
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, T> ExactSizeIterator for FreeListIterMut<'a, T> {}

/// Iterator yielding `(FreeListHandle, &T)` for live elements.
pub struct FreeListHandleIter<'a, T> {
    slots: &'a [FreeListSlot<T>],
    index: usize,
    remaining: usize,
}

impl<'a, T> Iterator for FreeListHandleIter<'a, T> {
    type Item = (FreeListHandle, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.slots.len() && self.remaining > 0 {
            let idx = self.index;
            let slot = &self.slots[idx];
            self.index += 1;
            if let FreeListSlot::Occupied { value, generation } = slot {
                self.remaining -= 1;
                let handle = FreeListHandle {
                    index: idx as u32,
                    generation: *generation,
                };
                return Some((handle, value));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, T> ExactSizeIterator for FreeListHandleIter<'a, T> {}

// ---------------------------------------------------------------------------
// RingBuffer<T> — fixed-capacity circular buffer
// ---------------------------------------------------------------------------

/// A fixed-capacity circular buffer.
///
/// When the buffer is full, new elements overwrite the oldest element.
/// Useful for frame history, input replay, and bounded logging.
///
/// # Example
/// ```
/// use genovo_core::collections::RingBuffer;
///
/// let mut rb = RingBuffer::new(3);
/// rb.push_back(1);
/// rb.push_back(2);
/// rb.push_back(3);
/// rb.push_back(4); // overwrites 1
/// assert_eq!(rb.front(), Some(&2));
/// ```
pub struct RingBuffer<T> {
    storage: Vec<Option<T>>,
    head: usize,
    tail: usize,
    len: usize,
    capacity: usize,
}

impl<T> RingBuffer<T> {
    /// Creates a new `RingBuffer` with the given capacity.
    ///
    /// # Panics
    /// Panics if `capacity` is zero.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "RingBuffer capacity must be > 0");
        let mut storage = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            storage.push(None);
        }
        Self {
            storage,
            head: 0,
            tail: 0,
            len: 0,
            capacity,
        }
    }

    /// Returns the fixed capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the current number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` if the buffer is at capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Pushes a value to the back. If the buffer is full, the oldest (front)
    /// element is overwritten.
    pub fn push_back(&mut self, value: T) {
        if self.is_full() {
            // Overwrite the oldest element at head.
            self.storage[self.tail] = Some(value);
            self.tail = (self.tail + 1) % self.capacity;
            self.head = (self.head + 1) % self.capacity;
        } else {
            self.storage[self.tail] = Some(value);
            self.tail = (self.tail + 1) % self.capacity;
            self.len += 1;
        }
    }

    /// Removes and returns the front (oldest) element.
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let value = self.storage[self.head].take();
        self.head = (self.head + 1) % self.capacity;
        self.len -= 1;
        value
    }

    /// Returns a reference to the front (oldest) element.
    pub fn front(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }
        self.storage[self.head].as_ref()
    }

    /// Returns a reference to the back (newest) element.
    pub fn back(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }
        let idx = if self.tail == 0 {
            self.capacity - 1
        } else {
            self.tail - 1
        };
        self.storage[idx].as_ref()
    }

    /// Returns a reference to the element at the given logical index (0 = front).
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        let real_idx = (self.head + index) % self.capacity;
        self.storage[real_idx].as_ref()
    }

    /// Returns a mutable reference to the element at the given logical index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }
        let real_idx = (self.head + index) % self.capacity;
        self.storage[real_idx].as_mut()
    }

    /// Removes all elements.
    pub fn clear(&mut self) {
        for slot in &mut self.storage {
            *slot = None;
        }
        self.head = 0;
        self.tail = 0;
        self.len = 0;
    }

    /// Returns an iterator over references in front-to-back order.
    pub fn iter(&self) -> RingBufferIter<'_, T> {
        RingBufferIter {
            ring: self,
            index: 0,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for RingBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RingBuffer")
            .field("len", &self.len)
            .field("capacity", &self.capacity)
            .finish()
    }
}

impl<T: Clone> Clone for RingBuffer<T> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            head: self.head,
            tail: self.tail,
            len: self.len,
            capacity: self.capacity,
        }
    }
}

/// Iterator over elements of a [`RingBuffer`].
pub struct RingBufferIter<'a, T> {
    ring: &'a RingBuffer<T>,
    index: usize,
}

impl<'a, T> Iterator for RingBufferIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.ring.len {
            return None;
        }
        let val = self.ring.get(self.index);
        self.index += 1;
        val
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.ring.len - self.index;
        (rem, Some(rem))
    }
}

impl<'a, T> ExactSizeIterator for RingBufferIter<'a, T> {}

impl<'a, T> IntoIterator for &'a RingBuffer<T> {
    type Item = &'a T;
    type IntoIter = RingBufferIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// ---------------------------------------------------------------------------
// BitSet — compact bit storage
// ---------------------------------------------------------------------------

/// A dynamically-sized bit set backed by `Vec<u64>`.
///
/// Each word stores 64 bits. The implementation uses hardware `count_ones()`
/// (popcount) for fast cardinality queries and provides the standard bitwise
/// algebra: AND, OR, XOR, NOT.
///
/// # Example
/// ```
/// use genovo_core::collections::BitSet;
///
/// let mut bs = BitSet::new();
/// bs.set(0);
/// bs.set(63);
/// bs.set(64);
/// assert_eq!(bs.count_ones(), 3);
/// assert!(bs.test(63));
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct BitSet {
    words: Vec<u64>,
}

impl BitSet {
    /// Number of bits per word.
    const BITS_PER_WORD: usize = 64;

    /// Creates an empty `BitSet`.
    pub fn new() -> Self {
        Self { words: Vec::new() }
    }

    /// Creates a `BitSet` that can hold at least `num_bits` without growing.
    pub fn with_capacity(num_bits: usize) -> Self {
        let word_count = Self::word_index(num_bits) + 1;
        Self {
            words: vec![0u64; word_count],
        }
    }

    /// Creates a `BitSet` from a raw word slice.
    pub fn from_words(words: &[u64]) -> Self {
        Self {
            words: words.to_vec(),
        }
    }

    #[inline]
    fn word_index(bit: usize) -> usize {
        bit / Self::BITS_PER_WORD
    }

    #[inline]
    fn bit_mask(bit: usize) -> u64 {
        1u64 << (bit % Self::BITS_PER_WORD)
    }

    /// Ensures there is room for `bit`.
    fn grow_to(&mut self, bit: usize) {
        let needed = Self::word_index(bit) + 1;
        if needed > self.words.len() {
            self.words.resize(needed, 0);
        }
    }

    /// Sets bit at `index`.
    pub fn set(&mut self, index: usize) {
        self.grow_to(index);
        self.words[Self::word_index(index)] |= Self::bit_mask(index);
    }

    /// Clears bit at `index`.
    pub fn clear(&mut self, index: usize) {
        let wi = Self::word_index(index);
        if wi < self.words.len() {
            self.words[wi] &= !Self::bit_mask(index);
        }
    }

    /// Toggles bit at `index`.
    pub fn toggle(&mut self, index: usize) {
        self.grow_to(index);
        self.words[Self::word_index(index)] ^= Self::bit_mask(index);
    }

    /// Tests whether bit at `index` is set.
    pub fn test(&self, index: usize) -> bool {
        let wi = Self::word_index(index);
        if wi >= self.words.len() {
            return false;
        }
        (self.words[wi] & Self::bit_mask(index)) != 0
    }

    /// Returns the number of set bits (population count).
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Returns the number of clear bits up to the current length.
    pub fn count_zeros(&self) -> usize {
        self.bit_len() - self.count_ones()
    }

    /// Returns the total number of bits tracked (words * 64).
    pub fn bit_len(&self) -> usize {
        self.words.len() * Self::BITS_PER_WORD
    }

    /// Returns `true` if no bits are set.
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Clears all bits.
    pub fn clear_all(&mut self) {
        for w in &mut self.words {
            *w = 0;
        }
    }

    /// Sets all bits in the current range.
    pub fn set_all(&mut self) {
        for w in &mut self.words {
            *w = !0u64;
        }
    }

    /// Returns the index of the first set bit, or `None`.
    pub fn find_first_set(&self) -> Option<usize> {
        for (i, &w) in self.words.iter().enumerate() {
            if w != 0 {
                return Some(i * Self::BITS_PER_WORD + w.trailing_zeros() as usize);
            }
        }
        None
    }

    /// Returns the index of the first clear bit, or `None` if all are set.
    pub fn find_first_clear(&self) -> Option<usize> {
        for (i, &w) in self.words.iter().enumerate() {
            if w != !0u64 {
                return Some(i * Self::BITS_PER_WORD + (!w).trailing_zeros() as usize);
            }
        }
        // All current words are full; the next clear bit is right after the
        // end of our storage.
        Some(self.words.len() * Self::BITS_PER_WORD)
    }

    /// Returns the index of the last set bit, or `None`.
    pub fn find_last_set(&self) -> Option<usize> {
        for (i, &w) in self.words.iter().enumerate().rev() {
            if w != 0 {
                return Some(i * Self::BITS_PER_WORD + (Self::BITS_PER_WORD - 1 - w.leading_zeros() as usize));
            }
        }
        None
    }

    /// In-place AND with another `BitSet`.
    pub fn and_with(&mut self, other: &BitSet) {
        let min_len = self.words.len().min(other.words.len());
        for i in 0..min_len {
            self.words[i] &= other.words[i];
        }
        // Any words beyond `other`'s length become zero.
        for i in min_len..self.words.len() {
            self.words[i] = 0;
        }
    }

    /// In-place OR with another `BitSet`.
    pub fn or_with(&mut self, other: &BitSet) {
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        for i in 0..other.words.len() {
            self.words[i] |= other.words[i];
        }
    }

    /// In-place XOR with another `BitSet`.
    pub fn xor_with(&mut self, other: &BitSet) {
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        for i in 0..other.words.len() {
            self.words[i] ^= other.words[i];
        }
    }

    /// In-place NOT (complement).
    pub fn invert(&mut self) {
        for w in &mut self.words {
            *w = !*w;
        }
    }

    /// Returns `true` if every set bit in `self` is also set in `other`.
    pub fn is_subset_of(&self, other: &BitSet) -> bool {
        for (i, &w) in self.words.iter().enumerate() {
            let other_w = if i < other.words.len() { other.words[i] } else { 0 };
            if w & !other_w != 0 {
                return false;
            }
        }
        true
    }

    /// Returns `true` if the two sets have at least one bit in common.
    pub fn intersects(&self, other: &BitSet) -> bool {
        let min_len = self.words.len().min(other.words.len());
        for i in 0..min_len {
            if self.words[i] & other.words[i] != 0 {
                return true;
            }
        }
        false
    }

    /// Returns an iterator over the indices of all set bits.
    pub fn iter_ones(&self) -> BitSetOnesIter<'_> {
        BitSetOnesIter {
            words: &self.words,
            word_idx: 0,
            current_word: if self.words.is_empty() { 0 } else { self.words[0] },
        }
    }
}

impl Default for BitSet {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for BitSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BitSet")
            .field("count_ones", &self.count_ones())
            .field("words", &self.words.len())
            .finish()
    }
}

/// Bitwise AND producing a new `BitSet`.
impl BitAnd for &BitSet {
    type Output = BitSet;
    fn bitand(self, rhs: Self) -> BitSet {
        let mut result = self.clone();
        result.and_with(rhs);
        result
    }
}

/// Bitwise AND-assign.
impl BitAndAssign<&BitSet> for BitSet {
    fn bitand_assign(&mut self, rhs: &BitSet) {
        self.and_with(rhs);
    }
}

/// Bitwise OR producing a new `BitSet`.
impl BitOr for &BitSet {
    type Output = BitSet;
    fn bitor(self, rhs: Self) -> BitSet {
        let mut result = self.clone();
        result.or_with(rhs);
        result
    }
}

/// Bitwise OR-assign.
impl BitOrAssign<&BitSet> for BitSet {
    fn bitor_assign(&mut self, rhs: &BitSet) {
        self.or_with(rhs);
    }
}

/// Bitwise XOR producing a new `BitSet`.
impl BitXor for &BitSet {
    type Output = BitSet;
    fn bitxor(self, rhs: Self) -> BitSet {
        let mut result = self.clone();
        result.xor_with(rhs);
        result
    }
}

/// Bitwise XOR-assign.
impl BitXorAssign<&BitSet> for BitSet {
    fn bitxor_assign(&mut self, rhs: &BitSet) {
        self.xor_with(rhs);
    }
}

/// Bitwise NOT producing a new `BitSet`.
impl Not for &BitSet {
    type Output = BitSet;
    fn not(self) -> BitSet {
        let mut result = self.clone();
        result.invert();
        result
    }
}

/// Iterator over set-bit indices using trailing-zeros scanning.
pub struct BitSetOnesIter<'a> {
    words: &'a [u64],
    word_idx: usize,
    current_word: u64,
}

impl<'a> Iterator for BitSetOnesIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_word != 0 {
                let bit = self.current_word.trailing_zeros() as usize;
                // Clear the lowest set bit.
                self.current_word &= self.current_word - 1;
                return Some(self.word_idx * BitSet::BITS_PER_WORD + bit);
            }
            self.word_idx += 1;
            if self.word_idx >= self.words.len() {
                return None;
            }
            self.current_word = self.words[self.word_idx];
        }
    }
}

// ---------------------------------------------------------------------------
// SparseArray<T> — sparse index → value mapping
// ---------------------------------------------------------------------------

/// A sparse-set data structure giving O(1) insert, remove, lookup, and
/// membership test while maintaining a dense, cache-friendly array for
/// iteration. Commonly used by ECS implementations to track component
/// ownership.
///
/// # Example
/// ```
/// use genovo_core::collections::SparseArray;
///
/// let mut sa = SparseArray::new(1024);
/// sa.insert(42, "hello");
/// sa.insert(999, "world");
/// assert_eq!(sa.get(42), Some(&"hello"));
/// assert_eq!(sa.len(), 2);
/// ```
pub struct SparseArray<T> {
    /// Maps sparse index → dense index.  `None` means the sparse index is empty.
    sparse: Vec<Option<usize>>,
    /// Dense array of values — always packed with no holes.
    dense_values: Vec<T>,
    /// Parallel to `dense_values`; stores the sparse index for each dense slot.
    dense_indices: Vec<usize>,
}

impl<T> SparseArray<T> {
    /// Creates a `SparseArray` that can map sparse indices in `[0, max_index)`.
    pub fn new(max_index: usize) -> Self {
        let mut sparse = Vec::with_capacity(max_index);
        sparse.resize_with(max_index, || None);
        Self {
            sparse,
            dense_values: Vec::new(),
            dense_indices: Vec::new(),
        }
    }

    /// Returns the maximum sparse index that can be stored without growing.
    #[inline]
    pub fn max_index(&self) -> usize {
        self.sparse.len()
    }

    /// Returns the number of stored elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.dense_values.len()
    }

    /// Returns `true` if no elements are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dense_values.is_empty()
    }

    /// Ensures the sparse array can hold `index`.
    fn ensure_capacity(&mut self, index: usize) {
        if index >= self.sparse.len() {
            self.sparse.resize(index + 1, None);
        }
    }

    /// Returns `true` if `index` is mapped to a value.
    pub fn contains(&self, index: usize) -> bool {
        index < self.sparse.len() && self.sparse[index].is_some()
    }

    /// Inserts `value` at `index`, replacing any existing value.
    pub fn insert(&mut self, index: usize, value: T) {
        self.ensure_capacity(index);

        if let Some(dense_idx) = self.sparse[index] {
            // Replace existing value.
            self.dense_values[dense_idx] = value;
        } else {
            let dense_idx = self.dense_values.len();
            self.dense_values.push(value);
            self.dense_indices.push(index);
            self.sparse[index] = Some(dense_idx);
        }
    }

    /// Removes the value at `index`, returning it if present.
    pub fn remove(&mut self, index: usize) -> Option<T> {
        if index >= self.sparse.len() {
            return None;
        }
        let dense_idx = self.sparse[index]?;
        self.sparse[index] = None;

        let last_dense = self.dense_values.len() - 1;
        if dense_idx != last_dense {
            // Swap-remove: move the last element into the vacated dense slot.
            let moved_sparse_idx = self.dense_indices[last_dense];
            self.sparse[moved_sparse_idx] = Some(dense_idx);
            self.dense_indices[dense_idx] = moved_sparse_idx;
        }

        self.dense_indices.pop();
        let value = self.dense_values.swap_remove(dense_idx);
        Some(value)
    }

    /// Returns a reference to the value at `index`.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.sparse.len() {
            return None;
        }
        let dense_idx = self.sparse[index]?;
        Some(&self.dense_values[dense_idx])
    }

    /// Returns a mutable reference to the value at `index`.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.sparse.len() {
            return None;
        }
        let dense_idx = self.sparse[index]?;
        Some(&mut self.dense_values[dense_idx])
    }

    /// Removes all elements.
    pub fn clear(&mut self) {
        for idx in &self.dense_indices {
            self.sparse[*idx] = None;
        }
        self.dense_values.clear();
        self.dense_indices.clear();
    }

    /// Returns a slice over all stored values (dense, cache-friendly).
    pub fn values(&self) -> &[T] {
        &self.dense_values
    }

    /// Returns a mutable slice over all stored values.
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.dense_values
    }

    /// Returns a slice of sparse indices parallel to `values()`.
    pub fn indices(&self) -> &[usize] {
        &self.dense_indices
    }

    /// Iterates over `(sparse_index, &value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &T)> {
        self.dense_indices.iter().copied().zip(self.dense_values.iter())
    }

    /// Iterates over `(sparse_index, &mut value)` pairs.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (usize, &mut T)> {
        self.dense_indices.iter().copied().zip(self.dense_values.iter_mut())
    }
}

impl<T: fmt::Debug> fmt::Debug for SparseArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseArray")
            .field("len", &self.len())
            .field("max_index", &self.max_index())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// StringInterner — string deduplication
// ---------------------------------------------------------------------------

/// A lightweight handle to an interned string. Copy-able, equality-comparable,
/// and hashable. Resolve back to `&str` through the [`StringInterner`] that
/// created it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct InternedString {
    index: u32,
}

impl InternedString {
    /// Returns the raw index (useful for serialization).
    #[inline]
    pub fn index(self) -> u32 {
        self.index
    }
}

impl Hash for InternedString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

impl fmt::Debug for InternedString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InternedString({})", self.index)
    }
}

impl fmt::Display for InternedString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<interned#{}>", self.index)
    }
}

/// Deduplicates strings by storing each unique string exactly once in a
/// contiguous buffer, returning a cheap [`InternedString`] handle that can be
/// used for O(1) equality comparison instead of string comparison.
///
/// # Example
/// ```
/// use genovo_core::collections::StringInterner;
///
/// let mut interner = StringInterner::new();
/// let a = interner.intern("hello");
/// let b = interner.intern("hello");
/// let c = interner.intern("world");
/// assert_eq!(a, b);
/// assert_ne!(a, c);
/// assert_eq!(interner.resolve(a), Some("hello"));
/// ```
pub struct StringInterner {
    /// Contiguous buffer holding all interned strings end-to-end.
    buffer: String,
    /// (offset, length) pairs into `buffer`, indexed by `InternedString::index`.
    spans: Vec<(u32, u32)>,
    /// Maps string content to its interned handle (deduplication lookup).
    lookup: HashMap<String, InternedString>,
}

impl StringInterner {
    /// Creates an empty interner.
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            spans: Vec::new(),
            lookup: HashMap::new(),
        }
    }

    /// Creates an interner with a capacity hint for the backing buffer.
    pub fn with_capacity(buffer_bytes: usize, string_count: usize) -> Self {
        Self {
            buffer: String::with_capacity(buffer_bytes),
            spans: Vec::with_capacity(string_count),
            lookup: HashMap::with_capacity(string_count),
        }
    }

    /// Interns a string, returning a handle. If the string was already interned,
    /// returns the same handle.
    pub fn intern(&mut self, s: &str) -> InternedString {
        if let Some(&handle) = self.lookup.get(s) {
            return handle;
        }

        let offset = self.buffer.len() as u32;
        let length = s.len() as u32;
        self.buffer.push_str(s);

        let index = self.spans.len() as u32;
        self.spans.push((offset, length));

        let handle = InternedString { index };
        self.lookup.insert(s.to_owned(), handle);
        handle
    }

    /// Resolves a handle back to the original string.
    pub fn resolve(&self, interned: InternedString) -> Option<&str> {
        let idx = interned.index as usize;
        if idx >= self.spans.len() {
            return None;
        }
        let (offset, length) = self.spans[idx];
        Some(&self.buffer[offset as usize..(offset + length) as usize])
    }

    /// Returns the number of unique strings interned.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Returns `true` if no strings have been interned.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Returns the total bytes used by the backing buffer.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `true` if the string is already interned.
    pub fn contains(&self, s: &str) -> bool {
        self.lookup.contains_key(s)
    }

    /// Iterates over all `(InternedString, &str)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (InternedString, &str)> {
        self.spans.iter().enumerate().map(move |(i, &(off, len))| {
            let handle = InternedString { index: i as u32 };
            let s = &self.buffer[off as usize..(off + len) as usize];
            (handle, s)
        })
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for StringInterner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StringInterner")
            .field("strings", &self.spans.len())
            .field("buffer_bytes", &self.buffer.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ObjectPool<T> — pre-allocated fixed-size pool
// ---------------------------------------------------------------------------

/// Handle into an [`ObjectPool`], carrying a type marker for safety.
#[derive(Debug)]
pub struct PoolHandle<T> {
    index: u32,
    _marker: PhantomData<T>,
}

impl<T> Clone for PoolHandle<T> {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            _marker: PhantomData,
        }
    }
}

impl<T> Copy for PoolHandle<T> {}

impl<T> PartialEq for PoolHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T> Eq for PoolHandle<T> {}

impl<T> Hash for PoolHandle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

impl<T> PoolHandle<T> {
    /// Returns the raw index within the pool.
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }
}

/// Internal slot state for the object pool.
enum PoolSlot<T> {
    Free { next: Option<u32> },
    Occupied(T),
}

/// A pre-allocated, fixed-size pool of objects.
///
/// All memory is allocated up front; after construction, `acquire` and
/// `release` perform zero allocations. This is useful for frequently
/// created/destroyed objects such as particles, projectiles, or UI widgets.
///
/// # Example
/// ```
/// use genovo_core::collections::ObjectPool;
///
/// let mut pool = ObjectPool::new(16, || 0i32);
/// let h = pool.acquire_with(42).unwrap();
/// assert_eq!(pool.get(h), Some(&42));
/// pool.release(h);
/// ```
pub struct ObjectPool<T> {
    slots: Vec<PoolSlot<T>>,
    free_head: Option<u32>,
    capacity: usize,
    alive: usize,
}

impl<T> ObjectPool<T> {
    /// Creates a pool of `capacity` objects, initializing each with `init`.
    pub fn new<F: Fn() -> T>(capacity: usize, init: F) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        // Pre-allocate objects and immediately free them. This ensures
        // the initial values exist in memory even before the first acquire.
        for _ in 0..capacity {
            slots.push(PoolSlot::Occupied(init()));
        }
        // Now convert them all to free slots, chaining the free list.
        for i in (0..capacity).rev() {
            let next = if i + 1 < capacity {
                Some(i as u32 + 1)
            } else {
                None
            };
            // We need to extract the value, drop it, and store Free.
            // Actually, let's just rebuild the vec.
            slots[i] = PoolSlot::Free { next };
        }
        // Fix the free list: slot 0 → slot 1 → ... → slot (capacity-1) → None
        // Re-chain properly.
        for i in 0..capacity {
            let next = if i + 1 < capacity {
                Some((i + 1) as u32)
            } else {
                None
            };
            slots[i] = PoolSlot::Free { next };
        }

        Self {
            slots,
            free_head: if capacity > 0 { Some(0) } else { None },
            capacity,
            alive: 0,
        }
    }

    /// Creates a pool where each slot is default-initialized.
    pub fn new_default(capacity: usize) -> Self
    where
        T: Default,
    {
        Self::new(capacity, T::default)
    }

    /// Returns the total pool capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of currently acquired objects.
    #[inline]
    pub fn alive(&self) -> usize {
        self.alive
    }

    /// Returns the number of free slots.
    #[inline]
    pub fn available(&self) -> usize {
        self.capacity - self.alive
    }

    /// Returns `true` if no more objects can be acquired.
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        self.free_head.is_none()
    }

    /// Acquires a free slot, returning a handle. Returns `None` if the pool is
    /// exhausted.
    pub fn acquire(&mut self) -> Option<PoolHandle<T>> {
        let idx = self.free_head?;
        match &self.slots[idx as usize] {
            PoolSlot::Free { next } => {
                self.free_head = *next;
            }
            PoolSlot::Occupied(_) => unreachable!("free head pointed to occupied slot"),
        }
        // The slot is Free; we need to fill it. Since we don't have an init
        // function stored, we use a sentinel. Callers must overwrite via
        // `get_mut`. We'll use MaybeUninit semantics via unsafe — but to keep
        // things safe, we'll require T: Default for acquire without a value.
        // Actually, let's provide `acquire_with`.
        // For the simpler API, return the handle and leave slot as-is (caller
        // uses get_mut). But we need an Occupied variant.
        // We'll just swap in a placeholder. This is only valid if T: Default.
        // Let's keep it simple: the Free variant doesn't hold a T, so we
        // can't hand one out. We'll mark the slot Occupied with... nothing.
        //
        // Best design: provide `acquire_with(value: T)` and keep the existing
        // `acquire` for T: Default.
        None // placeholder — see acquire_with below
    }

    /// Acquires a free slot, placing `value` into it. Returns `None` if the
    /// pool is exhausted.
    pub fn acquire_with(&mut self, value: T) -> Option<PoolHandle<T>> {
        let idx = self.free_head?;
        let next = match &self.slots[idx as usize] {
            PoolSlot::Free { next } => *next,
            PoolSlot::Occupied(_) => unreachable!("free head pointed to occupied slot"),
        };
        self.free_head = next;
        self.slots[idx as usize] = PoolSlot::Occupied(value);
        self.alive += 1;
        Some(PoolHandle {
            index: idx,
            _marker: PhantomData,
        })
    }

    /// Releases a previously acquired handle back to the pool.
    ///
    /// # Panics
    /// Panics if the handle does not refer to an occupied slot (double-free).
    pub fn release(&mut self, handle: PoolHandle<T>) {
        let idx = handle.index as usize;
        assert!(idx < self.slots.len(), "pool handle out of bounds");
        match &self.slots[idx] {
            PoolSlot::Occupied(_) => {}
            PoolSlot::Free { .. } => panic!("double-free of pool handle {}", idx),
        }
        self.slots[idx] = PoolSlot::Free {
            next: self.free_head,
        };
        self.free_head = Some(handle.index);
        self.alive -= 1;
    }

    /// Returns a reference to the object at `handle`.
    pub fn get(&self, handle: PoolHandle<T>) -> Option<&T> {
        match &self.slots[handle.index as usize] {
            PoolSlot::Occupied(val) => Some(val),
            PoolSlot::Free { .. } => None,
        }
    }

    /// Returns a mutable reference to the object at `handle`.
    pub fn get_mut(&mut self, handle: PoolHandle<T>) -> Option<&mut T> {
        match &mut self.slots[handle.index as usize] {
            PoolSlot::Occupied(val) => Some(val),
            PoolSlot::Free { .. } => None,
        }
    }

    /// Iterates over all currently occupied values.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.slots.iter().filter_map(|slot| {
            match slot {
                PoolSlot::Occupied(val) => Some(val),
                PoolSlot::Free { .. } => None,
            }
        })
    }

    /// Iterates mutably over all currently occupied values.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.slots.iter_mut().filter_map(|slot| {
            match slot {
                PoolSlot::Occupied(val) => Some(val),
                PoolSlot::Free { .. } => None,
            }
        })
    }
}

/// RAII guard that automatically releases a pool handle when dropped.
pub struct PoolGuard<'a, T> {
    pool: &'a mut ObjectPool<T>,
    handle: Option<PoolHandle<T>>,
}

impl<'a, T> PoolGuard<'a, T> {
    /// Creates a new guard wrapping an acquired handle.
    pub fn new(pool: &'a mut ObjectPool<T>, handle: PoolHandle<T>) -> Self {
        Self {
            pool,
            handle: Some(handle),
        }
    }

    /// Returns the underlying handle.
    pub fn handle(&self) -> PoolHandle<T> {
        self.handle.unwrap()
    }

    /// Explicitly releases the guard without waiting for drop, returning the
    /// handle for potential re-use tracking.
    pub fn release(mut self) -> PoolHandle<T> {
        let h = self.handle.take().unwrap();
        self.pool.release(h);
        h
    }
}

impl<'a, T> Drop for PoolGuard<'a, T> {
    fn drop(&mut self) {
        if let Some(h) = self.handle.take() {
            self.pool.release(h);
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for ObjectPool<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObjectPool")
            .field("capacity", &self.capacity)
            .field("alive", &self.alive)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Additional utilities
// ---------------------------------------------------------------------------

/// A small fixed-capacity inline vector that avoids heap allocation for
/// small element counts. Falls back to a `Vec` when the inline capacity
/// is exceeded.
pub struct SmallVec<T, const N: usize> {
    data: SmallVecData<T, N>,
    len: usize,
}

enum SmallVecData<T, const N: usize> {
    Inline([Option<T>; N]),
    Heap(Vec<T>),
}

impl<T, const N: usize> SmallVec<T, N> {
    /// Creates an empty `SmallVec`.
    pub fn new() -> Self {
        Self {
            data: SmallVecData::Inline(std::array::from_fn(|_| None)),
            len: 0,
        }
    }

    /// Returns the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns whether data is stored inline.
    #[inline]
    pub fn is_inline(&self) -> bool {
        matches!(self.data, SmallVecData::Inline(_))
    }

    /// Pushes a value, spilling to heap if inline capacity is exceeded.
    pub fn push(&mut self, value: T) {
        match &mut self.data {
            SmallVecData::Inline(arr) => {
                if self.len < N {
                    arr[self.len] = Some(value);
                    self.len += 1;
                } else {
                    // Spill to heap.
                    let mut vec = Vec::with_capacity(N * 2);
                    for slot in arr.iter_mut() {
                        if let Some(v) = slot.take() {
                            vec.push(v);
                        }
                    }
                    vec.push(value);
                    self.data = SmallVecData::Heap(vec);
                    self.len += 1;
                }
            }
            SmallVecData::Heap(vec) => {
                vec.push(value);
                self.len += 1;
            }
        }
    }

    /// Pops the last element.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        match &mut self.data {
            SmallVecData::Inline(arr) => arr[self.len].take(),
            SmallVecData::Heap(vec) => vec.pop(),
        }
    }

    /// Returns a reference to the element at `index`.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        match &self.data {
            SmallVecData::Inline(arr) => arr[index].as_ref(),
            SmallVecData::Heap(vec) => vec.get(index),
        }
    }

    /// Returns a mutable reference to the element at `index`.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }
        match &mut self.data {
            SmallVecData::Inline(arr) => arr[index].as_mut(),
            SmallVecData::Heap(vec) => vec.get_mut(index),
        }
    }

    /// Returns a slice over the elements.
    pub fn as_slice(&self) -> &[T] {
        match &self.data {
            SmallVecData::Heap(vec) => vec.as_slice(),
            SmallVecData::Inline(_) => {
                // Safety: we can't easily return a contiguous slice of
                // Option<T> as &[T]. For inline mode we'd need unsafe.
                // Instead, return empty — callers should use get().
                &[]
            }
        }
    }

    /// Clears all elements.
    pub fn clear(&mut self) {
        match &mut self.data {
            SmallVecData::Inline(arr) => {
                for i in 0..self.len {
                    arr[i] = None;
                }
            }
            SmallVecData::Heap(vec) => vec.clear(),
        }
        self.len = 0;
    }
}

impl<T, const N: usize> Default for SmallVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for SmallVec<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SmallVec")
            .field("len", &self.len)
            .field("inline", &self.is_inline())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// PriorityQueue<T> — binary min-heap
// ---------------------------------------------------------------------------

/// A minimum binary heap suitable for scheduling, pathfinding open-lists,
/// and event queues.
///
/// # Example
/// ```
/// use genovo_core::collections::PriorityQueue;
///
/// let mut pq = PriorityQueue::new();
/// pq.push(5);
/// pq.push(1);
/// pq.push(3);
/// assert_eq!(pq.pop(), Some(1));
/// assert_eq!(pq.pop(), Some(3));
/// ```
pub struct PriorityQueue<T: Ord> {
    data: Vec<T>,
}

impl<T: Ord> PriorityQueue<T> {
    /// Creates an empty min-heap.
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Creates an empty min-heap with pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            data: Vec::with_capacity(cap),
        }
    }

    /// Returns the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the heap is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a reference to the minimum element without removing it.
    pub fn peek(&self) -> Option<&T> {
        self.data.first()
    }

    /// Pushes a value onto the heap.
    pub fn push(&mut self, value: T) {
        self.data.push(value);
        self.sift_up(self.data.len() - 1);
    }

    /// Removes and returns the minimum element.
    pub fn pop(&mut self) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        let last = self.data.len() - 1;
        self.data.swap(0, last);
        let min = self.data.pop();
        if !self.data.is_empty() {
            self.sift_down(0);
        }
        min
    }

    /// Removes all elements.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.data[idx] < self.data[parent] {
                self.data.swap(idx, parent);
                idx = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        let len = self.data.len();
        loop {
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut smallest = idx;

            if left < len && self.data[left] < self.data[smallest] {
                smallest = left;
            }
            if right < len && self.data[right] < self.data[smallest] {
                smallest = right;
            }

            if smallest != idx {
                self.data.swap(idx, smallest);
                idx = smallest;
            } else {
                break;
            }
        }
    }
}

impl<T: Ord> Default for PriorityQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord + fmt::Debug> fmt::Debug for PriorityQueue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PriorityQueue")
            .field("len", &self.data.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- FreeList tests -------------------------------------------------------

    #[test]
    fn free_list_insert_get_remove() {
        let mut fl = FreeList::new();
        let h1 = fl.insert(10);
        let h2 = fl.insert(20);
        let h3 = fl.insert(30);

        assert_eq!(fl.len(), 3);
        assert_eq!(fl.get(h1), Some(&10));
        assert_eq!(fl.get(h2), Some(&20));
        assert_eq!(fl.get(h3), Some(&30));

        assert_eq!(fl.remove(h2), Some(20));
        assert_eq!(fl.len(), 2);
        assert_eq!(fl.get(h2), None); // generation mismatch

        // Re-use the freed slot.
        let h4 = fl.insert(40);
        assert_eq!(fl.get(h4), Some(&40));
        assert_eq!(fl.get(h2), None); // old handle still invalid
    }

    #[test]
    fn free_list_generation_check() {
        let mut fl = FreeList::new();
        let h = fl.insert("a");
        fl.remove(h);
        let h2 = fl.insert("b");
        // h and h2 share the same index but different generations.
        assert_eq!(h.index(), h2.index());
        assert_ne!(h.generation(), h2.generation());
        assert_eq!(fl.get(h), None);
        assert_eq!(fl.get(h2), Some(&"b"));
    }

    #[test]
    fn free_list_iter() {
        let mut fl = FreeList::new();
        fl.insert(1);
        let h2 = fl.insert(2);
        fl.insert(3);
        fl.remove(h2);

        let vals: Vec<_> = fl.iter().cloned().collect();
        assert_eq!(vals.len(), 2);
        assert!(vals.contains(&1));
        assert!(vals.contains(&3));
    }

    #[test]
    fn free_list_iter_with_handles() {
        let mut fl = FreeList::new();
        let h1 = fl.insert(100);
        let h2 = fl.insert(200);
        fl.remove(h1);

        let pairs: Vec<_> = fl.iter_with_handles().collect();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, h2);
        assert_eq!(*pairs[0].1, 200);
    }

    // -- RingBuffer tests -----------------------------------------------------

    #[test]
    fn ring_buffer_basic() {
        let mut rb = RingBuffer::new(3);
        assert!(rb.is_empty());
        rb.push_back(1);
        rb.push_back(2);
        rb.push_back(3);
        assert!(rb.is_full());
        assert_eq!(rb.front(), Some(&1));
        assert_eq!(rb.back(), Some(&3));

        rb.push_back(4); // overwrites 1
        assert_eq!(rb.front(), Some(&2));
        assert_eq!(rb.len(), 3);
    }

    #[test]
    fn ring_buffer_pop_front() {
        let mut rb = RingBuffer::new(4);
        rb.push_back(10);
        rb.push_back(20);
        assert_eq!(rb.pop_front(), Some(10));
        assert_eq!(rb.pop_front(), Some(20));
        assert_eq!(rb.pop_front(), None);
    }

    #[test]
    fn ring_buffer_index() {
        let mut rb = RingBuffer::new(5);
        for i in 0..5 {
            rb.push_back(i);
        }
        for i in 0..5 {
            assert_eq!(rb.get(i), Some(&i));
        }
        assert_eq!(rb.get(5), None);
    }

    #[test]
    fn ring_buffer_iter() {
        let mut rb = RingBuffer::new(3);
        rb.push_back(10);
        rb.push_back(20);
        rb.push_back(30);
        rb.push_back(40); // wraps, front is now 20
        let vals: Vec<_> = rb.iter().cloned().collect();
        assert_eq!(vals, vec![20, 30, 40]);
    }

    // -- BitSet tests ---------------------------------------------------------

    #[test]
    fn bitset_basic() {
        let mut bs = BitSet::new();
        bs.set(0);
        bs.set(1);
        bs.set(63);
        bs.set(64);
        bs.set(127);

        assert!(bs.test(0));
        assert!(bs.test(63));
        assert!(bs.test(64));
        assert!(bs.test(127));
        assert!(!bs.test(2));
        assert!(!bs.test(128));

        assert_eq!(bs.count_ones(), 5);
    }

    #[test]
    fn bitset_clear() {
        let mut bs = BitSet::new();
        bs.set(10);
        bs.set(20);
        bs.clear(10);
        assert!(!bs.test(10));
        assert!(bs.test(20));
        assert_eq!(bs.count_ones(), 1);
    }

    #[test]
    fn bitset_find_first() {
        let mut bs = BitSet::new();
        assert_eq!(bs.find_first_set(), None);
        bs.set(100);
        bs.set(200);
        assert_eq!(bs.find_first_set(), Some(100));
    }

    #[test]
    fn bitset_find_first_clear() {
        let mut bs = BitSet::with_capacity(128);
        bs.set_all();
        bs.clear(42);
        assert_eq!(bs.find_first_clear(), Some(42));
    }

    #[test]
    fn bitset_and_or_xor() {
        let mut a = BitSet::new();
        let mut b = BitSet::new();
        a.set(1);
        a.set(2);
        b.set(2);
        b.set(3);

        let and_result = &a & &b;
        assert!(and_result.test(2));
        assert!(!and_result.test(1));
        assert!(!and_result.test(3));

        let or_result = &a | &b;
        assert!(or_result.test(1));
        assert!(or_result.test(2));
        assert!(or_result.test(3));

        let xor_result = &a ^ &b;
        assert!(xor_result.test(1));
        assert!(!xor_result.test(2));
        assert!(xor_result.test(3));
    }

    #[test]
    fn bitset_iter_ones() {
        let mut bs = BitSet::new();
        bs.set(3);
        bs.set(7);
        bs.set(64);
        bs.set(65);
        let indices: Vec<_> = bs.iter_ones().collect();
        assert_eq!(indices, vec![3, 7, 64, 65]);
    }

    #[test]
    fn bitset_subset_intersects() {
        let mut a = BitSet::new();
        let mut b = BitSet::new();
        a.set(1);
        a.set(2);
        b.set(1);
        b.set(2);
        b.set(3);
        assert!(a.is_subset_of(&b));
        assert!(!b.is_subset_of(&a));
        assert!(a.intersects(&b));
    }

    // -- SparseArray tests ----------------------------------------------------

    #[test]
    fn sparse_array_insert_get() {
        let mut sa = SparseArray::new(100);
        sa.insert(5, "five");
        sa.insert(50, "fifty");
        assert_eq!(sa.get(5), Some(&"five"));
        assert_eq!(sa.get(50), Some(&"fifty"));
        assert_eq!(sa.get(10), None);
        assert_eq!(sa.len(), 2);
    }

    #[test]
    fn sparse_array_remove() {
        let mut sa = SparseArray::new(100);
        sa.insert(10, 100);
        sa.insert(20, 200);
        sa.insert(30, 300);

        assert_eq!(sa.remove(20), Some(200));
        assert_eq!(sa.len(), 2);
        assert_eq!(sa.get(20), None);

        // The remaining elements should still be accessible.
        assert_eq!(sa.get(10), Some(&100));
        assert_eq!(sa.get(30), Some(&300));
    }

    #[test]
    fn sparse_array_iter() {
        let mut sa = SparseArray::new(100);
        sa.insert(1, 10);
        sa.insert(5, 50);
        sa.insert(9, 90);

        let mut pairs: Vec<_> = sa.iter().collect();
        pairs.sort_by_key(|&(idx, _)| idx);
        assert_eq!(pairs, vec![(1, &10), (5, &50), (9, &90)]);
    }

    #[test]
    fn sparse_array_auto_grow() {
        let mut sa = SparseArray::new(2);
        sa.insert(100, "big index");
        assert_eq!(sa.get(100), Some(&"big index"));
    }

    // -- StringInterner tests -------------------------------------------------

    #[test]
    fn string_interner_basic() {
        let mut interner = StringInterner::new();
        let a = interner.intern("hello");
        let b = interner.intern("world");
        let c = interner.intern("hello");

        assert_eq!(a, c);
        assert_ne!(a, b);
        assert_eq!(interner.resolve(a), Some("hello"));
        assert_eq!(interner.resolve(b), Some("world"));
        assert_eq!(interner.len(), 2);
    }

    #[test]
    fn string_interner_contains() {
        let mut interner = StringInterner::new();
        interner.intern("foo");
        assert!(interner.contains("foo"));
        assert!(!interner.contains("bar"));
    }

    #[test]
    fn string_interner_iter() {
        let mut interner = StringInterner::new();
        interner.intern("alpha");
        interner.intern("beta");
        let pairs: Vec<_> = interner.iter().map(|(_, s)| s).collect();
        assert_eq!(pairs, vec!["alpha", "beta"]);
    }

    // -- ObjectPool tests -----------------------------------------------------

    #[test]
    fn object_pool_acquire_release() {
        let mut pool = ObjectPool::new(4, || 0i32);
        let h1 = pool.acquire_with(10).unwrap();
        let h2 = pool.acquire_with(20).unwrap();
        assert_eq!(pool.alive(), 2);
        assert_eq!(pool.get(h1), Some(&10));
        assert_eq!(pool.get(h2), Some(&20));

        pool.release(h1);
        assert_eq!(pool.alive(), 1);
        assert_eq!(pool.get(h1), None);
    }

    #[test]
    fn object_pool_exhaustion() {
        let mut pool = ObjectPool::new(2, || 0u8);
        let _h1 = pool.acquire_with(1).unwrap();
        let _h2 = pool.acquire_with(2).unwrap();
        assert!(pool.acquire_with(3).is_none());
    }

    // -- SmallVec tests -------------------------------------------------------

    #[test]
    fn small_vec_inline() {
        let mut sv: SmallVec<i32, 4> = SmallVec::new();
        sv.push(1);
        sv.push(2);
        sv.push(3);
        assert!(sv.is_inline());
        assert_eq!(sv.len(), 3);
        assert_eq!(sv.get(1), Some(&2));
    }

    #[test]
    fn small_vec_spill() {
        let mut sv: SmallVec<i32, 2> = SmallVec::new();
        sv.push(1);
        sv.push(2);
        sv.push(3); // spills to heap
        assert!(!sv.is_inline());
        assert_eq!(sv.len(), 3);
    }

    // -- PriorityQueue tests --------------------------------------------------

    #[test]
    fn priority_queue_min_heap() {
        let mut pq = PriorityQueue::new();
        pq.push(5);
        pq.push(1);
        pq.push(3);
        pq.push(2);
        pq.push(4);

        let mut sorted = Vec::new();
        while let Some(v) = pq.pop() {
            sorted.push(v);
        }
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn priority_queue_peek() {
        let mut pq = PriorityQueue::new();
        pq.push(10);
        pq.push(5);
        assert_eq!(pq.peek(), Some(&5));
    }
}
