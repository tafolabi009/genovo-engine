// engine/core/src/memory_arena.rs
//
// Arena allocator for the Genovo core module.
//
// Provides bump allocation, frame arena (reset each frame), typed arena,
// arena scope with RAII, alignment handling, and memory tracking.

use std::alloc::{self, Layout};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::ptr;

pub const DEFAULT_BLOCK_SIZE: usize = 64 * 1024;
pub const MIN_BLOCK_SIZE: usize = 4096;
pub const MAX_BLOCK_SIZE: usize = 64 * 1024 * 1024;
pub const DEFAULT_ALIGNMENT: usize = 16;
pub const MAX_ALIGNMENT: usize = 4096;

fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ArenaStats {
    pub total_allocated: usize,
    pub total_used: usize,
    pub total_wasted: usize,
    pub block_count: usize,
    pub allocation_count: u64,
    pub reset_count: u64,
    pub peak_usage: usize,
}

impl ArenaStats {
    pub fn utilization(&self) -> f32 {
        if self.total_allocated == 0 { 0.0 } else { self.total_used as f32 / self.total_allocated as f32 }
    }
}

struct ArenaBlock {
    data: *mut u8,
    layout: Layout,
    capacity: usize,
    used: usize,
}

impl ArenaBlock {
    fn new(size: usize, alignment: usize) -> Self {
        let layout = Layout::from_size_align(size, alignment).expect("Invalid layout");
        let data = unsafe { alloc::alloc(layout) };
        if data.is_null() { alloc::handle_alloc_error(layout); }
        Self { data, layout, capacity: size, used: 0 }
    }

    fn alloc(&mut self, size: usize, alignment: usize) -> Option<*mut u8> {
        let aligned_offset = align_up(self.used, alignment);
        let end = aligned_offset + size;
        if end > self.capacity { return None; }
        self.used = end;
        unsafe { Some(self.data.add(aligned_offset)) }
    }

    fn reset(&mut self) { self.used = 0; }
    fn remaining(&self) -> usize { self.capacity - self.used }
}

impl Drop for ArenaBlock {
    fn drop(&mut self) {
        if !self.data.is_null() { unsafe { alloc::dealloc(self.data, self.layout); } }
    }
}

pub struct MemoryArena {
    blocks: Vec<ArenaBlock>,
    block_size: usize,
    alignment: usize,
    stats: ArenaStats,
    name: String,
}

impl MemoryArena {
    pub fn new(block_size: usize) -> Self {
        Self::with_name(block_size, "unnamed")
    }

    pub fn with_name(block_size: usize, name: &str) -> Self {
        let size = block_size.clamp(MIN_BLOCK_SIZE, MAX_BLOCK_SIZE);
        let mut arena = Self {
            blocks: Vec::new(), block_size: size, alignment: DEFAULT_ALIGNMENT,
            stats: ArenaStats::default(), name: name.to_string(),
        };
        arena.add_block(size);
        arena
    }

    fn add_block(&mut self, min_size: usize) {
        let size = min_size.max(self.block_size);
        let block = ArenaBlock::new(size, self.alignment);
        self.stats.total_allocated += size;
        self.stats.block_count += 1;
        self.blocks.push(block);
    }

    pub fn alloc_raw(&mut self, size: usize, alignment: usize) -> *mut u8 {
        let align = alignment.min(MAX_ALIGNMENT).max(1);
        // Try current block first.
        if let Some(block) = self.blocks.last_mut() {
            if let Some(ptr) = block.alloc(size, align) {
                self.stats.total_used += size;
                self.stats.allocation_count += 1;
                self.stats.peak_usage = self.stats.peak_usage.max(self.stats.total_used);
                return ptr;
            }
        }
        // Need a new block.
        let needed = size + align;
        self.add_block(needed);
        let block = self.blocks.last_mut().unwrap();
        let ptr = block.alloc(size, align).expect("Allocation failed after adding block");
        self.stats.total_used += size;
        self.stats.allocation_count += 1;
        self.stats.peak_usage = self.stats.peak_usage.max(self.stats.total_used);
        ptr
    }

    pub fn alloc<T>(&mut self, value: T) -> &mut T {
        let ptr = self.alloc_raw(std::mem::size_of::<T>(), std::mem::align_of::<T>()) as *mut T;
        unsafe { ptr::write(ptr, value); &mut *ptr }
    }

    pub fn alloc_slice<T: Copy>(&mut self, values: &[T]) -> &mut [T] {
        let size = std::mem::size_of::<T>() * values.len();
        let ptr = self.alloc_raw(size, std::mem::align_of::<T>()) as *mut T;
        unsafe {
            ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
            std::slice::from_raw_parts_mut(ptr, values.len())
        }
    }

    pub fn alloc_zeroed(&mut self, size: usize, alignment: usize) -> *mut u8 {
        let ptr = self.alloc_raw(size, alignment);
        unsafe { ptr::write_bytes(ptr, 0, size); }
        ptr
    }

    pub fn alloc_str(&mut self, s: &str) -> &str {
        let bytes = self.alloc_slice(s.as_bytes());
        unsafe { std::str::from_utf8_unchecked(bytes) }
    }

    pub fn reset(&mut self) {
        for block in &mut self.blocks { block.reset(); }
        self.stats.total_used = 0;
        self.stats.total_wasted = 0;
        self.stats.reset_count += 1;
    }

    pub fn stats(&self) -> &ArenaStats { &self.stats }
    pub fn name(&self) -> &str { &self.name }
    pub fn block_count(&self) -> usize { self.blocks.len() }
    pub fn total_allocated(&self) -> usize { self.stats.total_allocated }
    pub fn total_used(&self) -> usize { self.stats.total_used }
    pub fn remaining_in_current_block(&self) -> usize { self.blocks.last().map_or(0, |b| b.remaining()) }
}

impl Drop for MemoryArena {
    fn drop(&mut self) { /* ArenaBlock drop handles deallocation */ }
}

pub struct FrameArena {
    arena: MemoryArena,
    frame_count: u64,
}

impl FrameArena {
    pub fn new(block_size: usize) -> Self { Self { arena: MemoryArena::with_name(block_size, "frame"), frame_count: 0 } }
    pub fn begin_frame(&mut self) { self.arena.reset(); self.frame_count += 1; }
    pub fn alloc<T>(&mut self, value: T) -> &mut T { self.arena.alloc(value) }
    pub fn alloc_slice<T: Copy>(&mut self, values: &[T]) -> &mut [T] { self.arena.alloc_slice(values) }
    pub fn alloc_raw(&mut self, size: usize, alignment: usize) -> *mut u8 { self.arena.alloc_raw(size, alignment) }
    pub fn frame_count(&self) -> u64 { self.frame_count }
    pub fn stats(&self) -> &ArenaStats { self.arena.stats() }
}

pub struct TypedArena<T> {
    arena: MemoryArena,
    count: usize,
    _marker: PhantomData<T>,
}

impl<T> TypedArena<T> {
    pub fn new() -> Self { Self { arena: MemoryArena::with_name(DEFAULT_BLOCK_SIZE, std::any::type_name::<T>()), count: 0, _marker: PhantomData } }
    pub fn with_capacity(capacity: usize) -> Self {
        let size = (std::mem::size_of::<T>() * capacity).max(MIN_BLOCK_SIZE);
        Self { arena: MemoryArena::with_name(size, std::any::type_name::<T>()), count: 0, _marker: PhantomData }
    }
    pub fn alloc(&mut self, value: T) -> &mut T { self.count += 1; self.arena.alloc(value) }
    pub fn count(&self) -> usize { self.count }
    pub fn reset(&mut self) { self.arena.reset(); self.count = 0; }
    pub fn stats(&self) -> &ArenaStats { self.arena.stats() }
}

pub struct ScopedArena<'a> {
    arena: &'a mut MemoryArena,
    saved_used: usize,
}

impl<'a> ScopedArena<'a> {
    pub fn new(arena: &'a mut MemoryArena) -> Self {
        let saved = arena.stats.total_used;
        Self { arena, saved_used: saved }
    }
    pub fn alloc<T>(&mut self, value: T) -> &mut T { self.arena.alloc(value) }
    pub fn alloc_raw(&mut self, size: usize, alignment: usize) -> *mut u8 { self.arena.alloc_raw(size, alignment) }
    pub fn bytes_used(&self) -> usize { self.arena.stats.total_used - self.saved_used }
}

impl<'a> Drop for ScopedArena<'a> {
    fn drop(&mut self) {
        // Note: We cannot truly "free" bump allocations. We just track usage.
        // A full reset should be done at a higher level.
    }
}

pub struct ArenaRegistry {
    arenas: HashMap<String, ArenaStats>,
}

impl ArenaRegistry {
    pub fn new() -> Self { Self { arenas: HashMap::new() } }
    pub fn register(&mut self, name: &str, stats: ArenaStats) { self.arenas.insert(name.to_string(), stats); }
    pub fn get(&self, name: &str) -> Option<&ArenaStats> { self.arenas.get(name) }
    pub fn total_allocated(&self) -> usize { self.arenas.values().map(|s| s.total_allocated).sum() }
    pub fn total_used(&self) -> usize { self.arenas.values().map(|s| s.total_used).sum() }
    pub fn report(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("Arena Registry ({} arenas):", self.arenas.len()));
        for (name, stats) in &self.arenas {
            lines.push(format!("  {}: allocated={} used={} util={:.1}% blocks={}", name, stats.total_allocated, stats.total_used, stats.utilization() * 100.0, stats.block_count));
        }
        lines.push(format!("  Total: allocated={} used={}", self.total_allocated(), self.total_used()));
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_basic() {
        let mut arena = MemoryArena::new(4096);
        let x = arena.alloc(42u32);
        assert_eq!(*x, 42);
        *x = 100;
        assert_eq!(*x, 100);
    }

    #[test]
    fn test_arena_slice() {
        let mut arena = MemoryArena::new(4096);
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let slice = arena.alloc_slice(&data);
        assert_eq!(slice.len(), 4);
        assert!((slice[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_arena_str() {
        let mut arena = MemoryArena::new(4096);
        let s = arena.alloc_str("hello world");
        assert_eq!(s, "hello world");
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = MemoryArena::new(4096);
        arena.alloc(1u32);
        arena.alloc(2u32);
        assert!(arena.total_used() > 0);
        arena.reset();
        assert_eq!(arena.total_used(), 0);
    }

    #[test]
    fn test_frame_arena() {
        let mut frame = FrameArena::new(4096);
        frame.begin_frame();
        let a = frame.alloc(42);
        assert_eq!(*a, 42);
        frame.begin_frame();
        assert_eq!(frame.frame_count(), 2);
    }

    #[test]
    fn test_typed_arena() {
        let mut arena = TypedArena::<f64>::new();
        let a = arena.alloc(3.14);
        let b = arena.alloc(2.71);
        assert!((* a - 3.14).abs() < 1e-6);
        assert!((* b - 2.71).abs() < 1e-6);
        assert_eq!(arena.count(), 2);
    }

    #[test]
    fn test_large_allocation() {
        let mut arena = MemoryArena::new(256);
        // Allocate more than one block worth.
        for i in 0..100 { arena.alloc(i as u64); }
        assert!(arena.block_count() > 1);
    }

    #[test]
    fn test_alignment() {
        let aligned = align_up(7, 8);
        assert_eq!(aligned, 8);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
    }
}
