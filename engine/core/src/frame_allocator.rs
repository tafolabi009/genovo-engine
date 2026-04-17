// engine/core/src/frame_allocator.rs
//
// Per-frame bump allocator: fast allocation that resets each frame.
// Features: typed allocation, arena with destructor callbacks,
// frame-scoped collections (Vec, String), alignment handling,
// memory usage tracking, and watermark statistics.

use std::cell::Cell;
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Alignment helpers
// ---------------------------------------------------------------------------

#[inline]
fn align_up(offset: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    (offset + alignment - 1) & !(alignment - 1)
}

// ---------------------------------------------------------------------------
// FrameAllocator
// ---------------------------------------------------------------------------

/// A bump allocator that resets every frame. All allocations within a frame
/// share the same backing buffer and are invalidated on `reset()`.
pub struct FrameAllocator {
    buffer: Vec<u8>,
    offset: Cell<usize>,
    capacity: usize,
    frame_count: u64,
    peak_usage: usize,
    total_allocated: u64,
    allocation_count: u64,
    reset_count: u64,
    destructor_list: Vec<Box<dyn FnOnce()>>,
}

impl FrameAllocator {
    /// Create a new frame allocator with the given capacity in bytes.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1024);
        Self {
            buffer: vec![0u8; capacity],
            offset: Cell::new(0),
            capacity,
            frame_count: 0,
            peak_usage: 0,
            total_allocated: 0,
            allocation_count: 0,
            reset_count: 0,
            destructor_list: Vec::new(),
        }
    }

    /// Allocate `size` bytes with the given alignment.
    /// Returns `None` if there is not enough space.
    pub fn alloc_raw(&self, size: usize, alignment: usize) -> Option<*mut u8> {
        let current = self.offset.get();
        let aligned = align_up(current, alignment);
        let new_offset = aligned + size;

        if new_offset > self.capacity {
            return None;
        }

        self.offset.set(new_offset);
        let ptr = unsafe { self.buffer.as_ptr().add(aligned) as *mut u8 };
        Some(ptr)
    }

    /// Allocate space for a single value of type T and write it.
    pub fn alloc<T: Copy>(&self, value: T) -> Option<&T> {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        let ptr = self.alloc_raw(size, align)? as *mut T;
        unsafe {
            std::ptr::write(ptr, value);
            Some(&*ptr)
        }
    }

    /// Allocate space for a single value and return a mutable reference.
    pub fn alloc_mut<T: Copy>(&self, value: T) -> Option<&mut T> {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        let ptr = self.alloc_raw(size, align)? as *mut T;
        unsafe {
            std::ptr::write(ptr, value);
            Some(&mut *ptr)
        }
    }

    /// Allocate an uninitialized slice of `count` elements.
    pub fn alloc_slice_uninit<T>(&self, count: usize) -> Option<&mut [std::mem::MaybeUninit<T>]> {
        let size = std::mem::size_of::<T>() * count;
        let align = std::mem::align_of::<T>();
        let ptr = self.alloc_raw(size, align)? as *mut std::mem::MaybeUninit<T>;
        unsafe { Some(std::slice::from_raw_parts_mut(ptr, count)) }
    }

    /// Allocate a slice initialized with a value.
    pub fn alloc_slice<T: Copy>(&self, count: usize, value: T) -> Option<&mut [T]> {
        let size = std::mem::size_of::<T>() * count;
        let align = std::mem::align_of::<T>();
        let ptr = self.alloc_raw(size, align)? as *mut T;
        unsafe {
            for i in 0..count {
                std::ptr::write(ptr.add(i), value);
            }
            Some(std::slice::from_raw_parts_mut(ptr, count))
        }
    }

    /// Allocate a slice from existing data.
    pub fn alloc_slice_copy<T: Copy>(&self, data: &[T]) -> Option<&mut [T]> {
        let size = std::mem::size_of::<T>() * data.len();
        let align = std::mem::align_of::<T>();
        let ptr = self.alloc_raw(size, align)? as *mut T;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            Some(std::slice::from_raw_parts_mut(ptr, data.len()))
        }
    }

    /// Allocate a string copy.
    pub fn alloc_str(&self, s: &str) -> Option<&str> {
        let bytes = self.alloc_slice_copy(s.as_bytes())?;
        Some(unsafe { std::str::from_utf8_unchecked(bytes) })
    }

    /// Register a destructor to be called on reset.
    pub fn register_destructor(&mut self, f: Box<dyn FnOnce()>) {
        self.destructor_list.push(f);
    }

    /// Reset the allocator, freeing all allocations.
    pub fn reset(&mut self) {
        let used = self.offset.get();
        self.peak_usage = self.peak_usage.max(used);
        self.total_allocated += used as u64;
        self.allocation_count = 0;
        self.offset.set(0);
        self.frame_count += 1;
        self.reset_count += 1;

        // Run destructors in reverse order.
        let destructors = std::mem::take(&mut self.destructor_list);
        for dtor in destructors.into_iter().rev() {
            dtor();
        }
    }

    /// Current bytes used.
    pub fn used(&self) -> usize {
        self.offset.get()
    }

    /// Remaining bytes available.
    pub fn available(&self) -> usize {
        self.capacity.saturating_sub(self.offset.get())
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Usage ratio (0.0 to 1.0).
    pub fn usage_ratio(&self) -> f32 {
        self.offset.get() as f32 / self.capacity as f32
    }

    /// Peak usage across all frames.
    pub fn peak_usage(&self) -> usize {
        self.peak_usage
    }

    /// Total bytes allocated across all frames.
    pub fn total_allocated(&self) -> u64 {
        self.total_allocated
    }

    /// Number of times reset has been called.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get statistics.
    pub fn stats(&self) -> FrameAllocatorStats {
        FrameAllocatorStats {
            capacity: self.capacity,
            used: self.offset.get(),
            available: self.available(),
            peak_usage: self.peak_usage,
            total_allocated: self.total_allocated,
            frame_count: self.frame_count,
            reset_count: self.reset_count,
            usage_ratio: self.usage_ratio(),
        }
    }
}

impl Drop for FrameAllocator {
    fn drop(&mut self) {
        // Run any remaining destructors.
        let destructors = std::mem::take(&mut self.destructor_list);
        for dtor in destructors.into_iter().rev() {
            dtor();
        }
    }
}

/// Statistics for the frame allocator.
#[derive(Debug, Clone)]
pub struct FrameAllocatorStats {
    pub capacity: usize,
    pub used: usize,
    pub available: usize,
    pub peak_usage: usize,
    pub total_allocated: u64,
    pub frame_count: u64,
    pub reset_count: u64,
    pub usage_ratio: f32,
}

// ---------------------------------------------------------------------------
// FrameVec — a Vec-like collection backed by the frame allocator
// ---------------------------------------------------------------------------

/// A growable array that uses the frame allocator for storage.
/// Only valid for the current frame.
pub struct FrameVec<'a, T: Copy> {
    data: &'a mut [T],
    len: usize,
    _phantom: PhantomData<T>,
}

impl<'a, T: Copy> FrameVec<'a, T> {
    /// Create a new FrameVec with the given initial capacity.
    pub fn new(allocator: &'a FrameAllocator, capacity: usize) -> Option<Self> {
        let data = allocator.alloc_slice(capacity, unsafe { std::mem::zeroed() })?;
        Some(Self {
            data,
            len: 0,
            _phantom: PhantomData,
        })
    }

    pub fn push(&mut self, value: T) -> bool {
        if self.len >= self.data.len() {
            return false;
        }
        self.data[self.len] = value;
        self.len += 1;
        true
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        Some(self.data[self.len])
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data[..self.len]
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[..self.len]
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len { Some(&self.data[index]) } else { None }
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len { Some(&mut self.data[index]) } else { None }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data[..self.len].iter()
    }
}

impl<'a, T: Copy> std::ops::Index<usize> for FrameVec<'a, T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        &self.data[index]
    }
}

impl<'a, T: Copy> std::ops::IndexMut<usize> for FrameVec<'a, T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.data[index]
    }
}

// ---------------------------------------------------------------------------
// ScratchBuffer — a reusable temporary buffer
// ---------------------------------------------------------------------------

/// A reusable scratch buffer that can be obtained from the frame allocator.
pub struct ScratchBuffer<'a> {
    data: &'a mut [u8],
    write_pos: usize,
}

impl<'a> ScratchBuffer<'a> {
    pub fn new(allocator: &'a FrameAllocator, size: usize) -> Option<Self> {
        let data = allocator.alloc_slice(size, 0u8)?;
        Some(Self { data, write_pos: 0 })
    }

    pub fn write(&mut self, bytes: &[u8]) -> usize {
        let available = self.data.len() - self.write_pos;
        let to_write = bytes.len().min(available);
        self.data[self.write_pos..self.write_pos + to_write].copy_from_slice(&bytes[..to_write]);
        self.write_pos += to_write;
        to_write
    }

    pub fn write_u32(&mut self, value: u32) -> bool {
        let bytes = value.to_le_bytes();
        self.write(&bytes) == 4
    }

    pub fn write_f32(&mut self, value: f32) -> bool {
        let bytes = value.to_le_bytes();
        self.write(&bytes) == 4
    }

    pub fn data(&self) -> &[u8] {
        &self.data[..self.write_pos]
    }

    pub fn remaining(&self) -> usize {
        self.data.len() - self.write_pos
    }

    pub fn reset(&mut self) {
        self.write_pos = 0;
    }

    pub fn len(&self) -> usize {
        self.write_pos
    }

    pub fn is_empty(&self) -> bool {
        self.write_pos == 0
    }

    pub fn capacity(&self) -> usize {
        self.data.len()
    }
}

// ---------------------------------------------------------------------------
// MultiFrameAllocator — ring buffer of frame allocators
// ---------------------------------------------------------------------------

/// A ring buffer of frame allocators for double/triple buffering.
pub struct MultiFrameAllocator {
    allocators: Vec<FrameAllocator>,
    current_index: usize,
    frame_count: u64,
}

impl MultiFrameAllocator {
    /// Create a multi-frame allocator with `frames_in_flight` buffers.
    pub fn new(capacity_per_frame: usize, frames_in_flight: usize) -> Self {
        let allocators = (0..frames_in_flight)
            .map(|_| FrameAllocator::new(capacity_per_frame))
            .collect();
        Self {
            allocators,
            current_index: 0,
            frame_count: 0,
        }
    }

    /// Begin a new frame, resetting the current allocator.
    pub fn begin_frame(&mut self) {
        self.frame_count += 1;
        self.current_index = (self.frame_count as usize) % self.allocators.len();
        self.allocators[self.current_index].reset();
    }

    /// Get the current frame's allocator.
    pub fn current(&self) -> &FrameAllocator {
        &self.allocators[self.current_index]
    }

    /// Get the current frame's allocator (mutable).
    pub fn current_mut(&mut self) -> &mut FrameAllocator {
        &mut self.allocators[self.current_index]
    }

    /// Get the allocator for `frames_ago` frames back (0 = current).
    pub fn get_frame(&self, frames_ago: usize) -> Option<&FrameAllocator> {
        if frames_ago >= self.allocators.len() {
            return None;
        }
        let idx = (self.current_index + self.allocators.len() - frames_ago) % self.allocators.len();
        Some(&self.allocators[idx])
    }

    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    pub fn frames_in_flight(&self) -> usize {
        self.allocators.len()
    }

    /// Total memory used across all frame allocators.
    pub fn total_capacity(&self) -> usize {
        self.allocators.iter().map(|a| a.capacity()).sum()
    }

    /// Peak usage across all frame allocators.
    pub fn overall_peak(&self) -> usize {
        self.allocators.iter().map(|a| a.peak_usage()).max().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_alloc() {
        let alloc = FrameAllocator::new(4096);
        let val = alloc.alloc(42u32).unwrap();
        assert_eq!(*val, 42);
        assert!(alloc.used() > 0);
    }

    #[test]
    fn test_alloc_slice() {
        let alloc = FrameAllocator::new(4096);
        let slice = alloc.alloc_slice(10, 0.0f32).unwrap();
        assert_eq!(slice.len(), 10);
        slice[0] = 1.0;
        slice[9] = 9.0;
        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[9], 9.0);
    }

    #[test]
    fn test_alloc_slice_copy() {
        let alloc = FrameAllocator::new(4096);
        let data = [1u32, 2, 3, 4, 5];
        let copy = alloc.alloc_slice_copy(&data).unwrap();
        assert_eq!(copy, &data);
    }

    #[test]
    fn test_alloc_str() {
        let alloc = FrameAllocator::new(4096);
        let s = alloc.alloc_str("hello world").unwrap();
        assert_eq!(s, "hello world");
    }

    #[test]
    fn test_reset() {
        let mut alloc = FrameAllocator::new(4096);
        let _ = alloc.alloc(42u32);
        assert!(alloc.used() > 0);
        alloc.reset();
        assert_eq!(alloc.used(), 0);
        assert!(alloc.peak_usage() > 0);
    }

    #[test]
    fn test_out_of_space() {
        let alloc = FrameAllocator::new(32);
        let result = alloc.alloc_slice(1000, 0u32);
        assert!(result.is_none());
    }

    #[test]
    fn test_frame_vec() {
        let alloc = FrameAllocator::new(4096);
        let mut fv = FrameVec::<u32>::new(&alloc, 10).unwrap();
        fv.push(1);
        fv.push(2);
        fv.push(3);
        assert_eq!(fv.len(), 3);
        assert_eq!(fv.as_slice(), &[1, 2, 3]);
        assert_eq!(fv.pop(), Some(3));
        assert_eq!(fv.len(), 2);
    }

    #[test]
    fn test_scratch_buffer() {
        let alloc = FrameAllocator::new(4096);
        let mut scratch = ScratchBuffer::new(&alloc, 256).unwrap();
        scratch.write_u32(42);
        scratch.write_f32(3.14);
        assert_eq!(scratch.len(), 8);
        scratch.reset();
        assert!(scratch.is_empty());
    }

    #[test]
    fn test_multi_frame() {
        let mut mfa = MultiFrameAllocator::new(4096, 3);
        mfa.begin_frame();
        let _ = mfa.current().alloc(1u32);
        mfa.begin_frame();
        let _ = mfa.current().alloc(2u32);
        assert_eq!(mfa.frame_count(), 2);
    }

    #[test]
    fn test_alignment() {
        let alloc = FrameAllocator::new(4096);
        let _ = alloc.alloc(1u8);
        let ptr = alloc.alloc(42u64).unwrap();
        let addr = ptr as *const u64 as usize;
        assert_eq!(addr % std::mem::align_of::<u64>(), 0);
    }

    #[test]
    fn test_destructor() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let flag = Arc::new(AtomicBool::new(false));
        let flag2 = flag.clone();

        let mut alloc = FrameAllocator::new(4096);
        alloc.register_destructor(Box::new(move || {
            flag2.store(true, Ordering::SeqCst);
        }));

        assert!(!flag.load(Ordering::SeqCst));
        alloc.reset();
        assert!(flag.load(Ordering::SeqCst));
    }
}
