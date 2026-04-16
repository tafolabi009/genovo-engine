//! Custom memory allocators.
//!
//! Game engines require predictable allocation patterns. The allocators in this
//! module are designed for specific usage patterns that arise in real-time
//! applications:
//!
//! | Allocator         | Pattern          | Example use case                    |
//! |-------------------|------------------|-------------------------------------|
//! | `LinearAllocator` | Bump / frame     | Per-frame scratch data              |
//! | `PoolAllocator`   | Fixed-size block | Particle systems, ECS components    |
//! | `StackAllocator`  | LIFO             | Nested scope temporaries            |
//! | `ScopedArena`     | Scoped region    | Render-pass–scoped command building |

use std::alloc::Layout;
use std::ptr::NonNull;

// ---------------------------------------------------------------------------
// Allocator trait
// ---------------------------------------------------------------------------

/// Core allocation interface.
///
/// Implementations provide raw byte allocation. Higher-level typed wrappers
/// (e.g., `Vec`-like containers backed by a custom allocator) build on top of
/// this trait.
///
/// # Safety
///
/// Implementors must guarantee that returned pointers are valid for the
/// requested layout and that `dealloc` / `realloc` are only called with
/// pointers previously returned by `alloc` (or `realloc`) on the same
/// allocator instance.
pub unsafe trait Allocator: Send + Sync {
    /// Allocates a block of memory satisfying `layout`.
    ///
    /// Returns `None` if the allocator cannot satisfy the request.
    fn alloc(&self, layout: Layout) -> Option<NonNull<u8>>;

    /// Deallocates a previously allocated block.
    ///
    /// # Safety
    ///
    /// `ptr` must have been returned by a prior call to [`alloc`] or
    /// [`realloc`] on this allocator, and `layout` must be the same layout
    /// (or compatible) used in that call.
    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout);

    /// Resizes a previously allocated block.
    ///
    /// The default implementation allocates a new block, copies, and frees
    /// the old one. Allocators that can grow in-place should override this.
    ///
    /// # Safety
    ///
    /// Same preconditions as [`dealloc`].
    unsafe fn realloc(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Option<NonNull<u8>> {
        let new_ptr = self.alloc(new_layout)?;
        let copy_size = old_layout.size().min(new_layout.size());
        std::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), copy_size);
        self.dealloc(ptr, old_layout);
        Some(new_ptr)
    }
}

// ---------------------------------------------------------------------------
// MemoryStats
// ---------------------------------------------------------------------------

/// Diagnostic counters for memory usage tracking.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total bytes currently allocated.
    pub bytes_allocated: usize,
    /// High-water mark (peak allocation).
    pub peak_bytes_allocated: usize,
    /// Total number of `alloc` calls.
    pub allocation_count: u64,
    /// Total number of `dealloc` calls.
    pub deallocation_count: u64,
    /// Total bytes ever allocated (cumulative).
    pub total_bytes_allocated: u64,
}

impl MemoryStats {
    /// Records an allocation of `size` bytes.
    pub fn record_alloc(&mut self, size: usize) {
        self.bytes_allocated += size;
        self.allocation_count += 1;
        self.total_bytes_allocated += size as u64;
        if self.bytes_allocated > self.peak_bytes_allocated {
            self.peak_bytes_allocated = self.bytes_allocated;
        }
    }

    /// Records a deallocation of `size` bytes.
    pub fn record_dealloc(&mut self, size: usize) {
        self.bytes_allocated = self.bytes_allocated.saturating_sub(size);
        self.deallocation_count += 1;
    }

    /// Resets all counters to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ---------------------------------------------------------------------------
// LinearAllocator (bump allocator)
// ---------------------------------------------------------------------------

/// A bump (linear) allocator for frame-temporary data.
///
/// Allocations are O(1) pointer bumps. Individual deallocations are no-ops;
/// the entire allocator is reset at once (typically at the start of each
/// frame).
///
/// # Thread safety
///
/// The internal offset is protected by a `parking_lot::Mutex`. For
/// single-threaded hot paths, prefer the unsafe `alloc_unchecked` variant
/// (not yet implemented).
pub struct LinearAllocator {
    /// Base pointer of the backing memory region.
    _base: NonNull<u8>,
    /// Total capacity in bytes.
    capacity: usize,
    /// Current allocation offset from base.
    offset: parking_lot::Mutex<usize>,
    /// Diagnostic stats.
    stats: parking_lot::Mutex<MemoryStats>,
}

// SAFETY: The backing memory is exclusively owned and the offset is mutex-guarded.
unsafe impl Send for LinearAllocator {}
unsafe impl Sync for LinearAllocator {}

impl LinearAllocator {
    /// Creates a new linear allocator backed by `capacity` bytes.
    pub fn new(capacity: usize) -> Self {
        // Design note: using the global allocator for the backing region is
        // intentional for portability. A virtual-memory (VirtualAlloc / mmap)
        // reserve-commit strategy could allow growing without copying, but the
        // current fixed-capacity design already suits the per-frame bump usage
        // pattern. Platform-specific VM backing can be added behind a feature
        // flag when needed.
        let layout = Layout::from_size_align(capacity, 16).expect("invalid layout");
        let base = unsafe { std::alloc::alloc(layout) };
        let base = NonNull::new(base).expect("LinearAllocator: allocation failed");
        Self {
            _base: base,
            capacity,
            offset: parking_lot::Mutex::new(0),
            stats: parking_lot::Mutex::new(MemoryStats::default()),
        }
    }

    /// Resets the allocator, invalidating all outstanding allocations.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no references into this allocator are live.
    pub unsafe fn reset(&self) {
        *self.offset.lock() = 0;
        // Note: stats are intentionally *not* reset here; call stats().reset()
        // explicitly if desired.
    }

    /// Returns current allocation stats.
    pub fn stats(&self) -> MemoryStats {
        self.stats.lock().clone()
    }

    /// Returns how many bytes remain available.
    pub fn remaining(&self) -> usize {
        self.capacity - *self.offset.lock()
    }
}

unsafe impl Allocator for LinearAllocator {
    fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        let mut offset = self.offset.lock();
        // Align up.
        let aligned = (*offset + layout.align() - 1) & !(layout.align() - 1);
        let new_offset = aligned + layout.size();
        if new_offset > self.capacity {
            return None;
        }
        *offset = new_offset;
        self.stats.lock().record_alloc(layout.size());
        // SAFETY: base + aligned is within the reserved region.
        Some(unsafe { NonNull::new_unchecked(self._base.as_ptr().add(aligned)) })
    }

    unsafe fn dealloc(&self, _ptr: NonNull<u8>, layout: Layout) {
        // Linear allocator does not support individual deallocation.
        // Track the stat for bookkeeping only.
        self.stats.lock().record_dealloc(layout.size());
    }
}

impl Drop for LinearAllocator {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.capacity, 16).expect("invalid layout");
        unsafe {
            std::alloc::dealloc(self._base.as_ptr(), layout);
        }
    }
}

// ---------------------------------------------------------------------------
// PoolAllocator
// ---------------------------------------------------------------------------

/// A fixed-size block allocator.
///
/// All allocations are the same size, making this allocator ideal for
/// homogeneous collections (particles, ECS component chunks, etc.).
/// Allocations and deallocations are O(1).
pub struct PoolAllocator {
    /// Size of each block in bytes (including alignment padding).
    block_size: usize,
    /// Alignment requirement for each block.
    _block_align: usize,
    /// Base pointer of the backing memory.
    _base: NonNull<u8>,
    /// Total number of blocks.
    _block_count: usize,
    /// Free-list head. Each free block stores the index of the next free
    /// block at its start.
    free_head: parking_lot::Mutex<Option<usize>>,
    /// Diagnostic stats.
    stats: parking_lot::Mutex<MemoryStats>,
}

unsafe impl Send for PoolAllocator {}
unsafe impl Sync for PoolAllocator {}

impl PoolAllocator {
    /// Creates a pool allocator for `block_count` blocks of `block_size` bytes
    /// each, aligned to `block_align`.
    pub fn new(block_size: usize, block_align: usize, block_count: usize) -> Self {
        // Free-list is threaded through the blocks: each free block stores
        // the index of the next free block at its start (as a usize).
        assert!(
            block_size >= std::mem::size_of::<usize>(),
            "block_size must be >= size_of::<usize>() to store the free-list pointer"
        );
        let padded_size = (block_size + block_align - 1) & !(block_align - 1);
        let total = padded_size * block_count;
        let layout = Layout::from_size_align(total, block_align).expect("invalid layout");
        let base = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) })
            .expect("PoolAllocator: allocation failed");

        // Thread free list.
        unsafe {
            for i in 0..block_count {
                let block_ptr = base.as_ptr().add(i * padded_size) as *mut usize;
                if i + 1 < block_count {
                    *block_ptr = i + 1;
                } else {
                    // Sentinel: no next block.
                    *block_ptr = usize::MAX;
                }
            }
        }

        Self {
            block_size: padded_size,
            _block_align: block_align,
            _base: base,
            _block_count: block_count,
            free_head: parking_lot::Mutex::new(if block_count > 0 { Some(0) } else { None }),
            stats: parking_lot::Mutex::new(MemoryStats::default()),
        }
    }

    /// Returns current allocation stats.
    pub fn stats(&self) -> MemoryStats {
        self.stats.lock().clone()
    }
}

unsafe impl Allocator for PoolAllocator {
    fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        if layout.size() > self.block_size || layout.align() > self._block_align {
            return None;
        }
        let mut head = self.free_head.lock();
        let index = (*head)?;
        // Read the next-free index stored at the beginning of this block.
        let ptr = unsafe { self._base.as_ptr().add(index * self.block_size) };
        let next = unsafe { *(ptr as *const usize) };
        *head = if next == usize::MAX { None } else { Some(next) };
        self.stats.lock().record_alloc(self.block_size);
        NonNull::new(ptr)
    }

    unsafe fn dealloc(&self, ptr: NonNull<u8>, _layout: Layout) {
        let offset = ptr.as_ptr() as usize - self._base.as_ptr() as usize;
        let index = offset / self.block_size;
        let mut head = self.free_head.lock();
        // Push this block onto the free list.
        let next = head.unwrap_or(usize::MAX);
        unsafe {
            *(ptr.as_ptr() as *mut usize) = next;
        }
        *head = Some(index);
        self.stats.lock().record_dealloc(self.block_size);
    }
}

impl Drop for PoolAllocator {
    fn drop(&mut self) {
        let total = self.block_size * self._block_count;
        let layout = Layout::from_size_align(total, self._block_align).expect("invalid layout");
        unsafe {
            std::alloc::dealloc(self._base.as_ptr(), layout);
        }
    }
}

// ---------------------------------------------------------------------------
// StackAllocator
// ---------------------------------------------------------------------------

/// A LIFO (stack) allocator.
///
/// Allocations grow upward; deallocations must happen in reverse order.
/// Each allocation is preceded by a small header that records the previous
/// offset so that `dealloc` can rewind correctly.
pub struct StackAllocator {
    /// Base pointer of the backing memory.
    _base: NonNull<u8>,
    /// Total capacity in bytes.
    capacity: usize,
    /// Current stack top.
    offset: parking_lot::Mutex<usize>,
    /// Diagnostic stats.
    stats: parking_lot::Mutex<MemoryStats>,
}

unsafe impl Send for StackAllocator {}
unsafe impl Sync for StackAllocator {}

/// Header stored before each stack allocation.
#[repr(C)]
struct StackHeader {
    /// Offset of the previous allocation (used to rewind on dealloc).
    prev_offset: usize,
    /// Padding bytes added for alignment (so we can rewind past them).
    padding: usize,
}

impl StackAllocator {
    /// Creates a stack allocator with the given capacity.
    pub fn new(capacity: usize) -> Self {
        // Uses the global allocator for the backing region. Virtual-memory
        // reserve/commit would allow dynamic growth but is not required for
        // the fixed-capacity LIFO pattern this allocator targets.
        let layout = Layout::from_size_align(capacity, 16).expect("invalid layout");
        let base = NonNull::new(unsafe { std::alloc::alloc(layout) })
            .expect("StackAllocator: allocation failed");
        Self {
            _base: base,
            capacity,
            offset: parking_lot::Mutex::new(0),
            stats: parking_lot::Mutex::new(MemoryStats::default()),
        }
    }

    /// Returns current allocation stats.
    pub fn stats(&self) -> MemoryStats {
        self.stats.lock().clone()
    }
}

unsafe impl Allocator for StackAllocator {
    fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        // Header-based LIFO deallocation: each allocation is preceded by a
        // StackHeader recording the previous offset. On dealloc we read the
        // header and rewind the stack top.
        let mut offset = self.offset.lock();
        let header_size = std::mem::size_of::<StackHeader>();

        // Reserve space for header, then align payload.
        let after_header = *offset + header_size;
        let aligned = (after_header + layout.align() - 1) & !(layout.align() - 1);
        let padding = aligned - after_header;
        let new_offset = aligned + layout.size();

        if new_offset > self.capacity {
            return None;
        }

        // Write header just before the aligned payload.
        let header_ptr = unsafe { self._base.as_ptr().add(aligned - header_size) } as *mut StackHeader;
        unsafe {
            (*header_ptr).prev_offset = *offset;
            (*header_ptr).padding = padding;
        }

        *offset = new_offset;
        self.stats.lock().record_alloc(layout.size());
        NonNull::new(unsafe { self._base.as_ptr().add(aligned) })
    }

    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        let mut offset = self.offset.lock();
        let header_size = std::mem::size_of::<StackHeader>();
        let header_ptr = unsafe { (ptr.as_ptr().sub(header_size)) as *const StackHeader };
        let header = unsafe { &*header_ptr };
        *offset = header.prev_offset;
        self.stats.lock().record_dealloc(layout.size());
    }
}

impl Drop for StackAllocator {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.capacity, 16).expect("invalid layout");
        unsafe {
            std::alloc::dealloc(self._base.as_ptr(), layout);
        }
    }
}

// ---------------------------------------------------------------------------
// ScopedArena
// ---------------------------------------------------------------------------

/// A scoped arena allocator.
///
/// Similar to [`LinearAllocator`], but supports nested scopes. Entering a
/// scope records the current offset; leaving a scope rewinds to that
/// saved offset, freeing everything allocated within the scope.
///
/// ```ignore
/// let arena = ScopedArena::new(4096);
/// {
///     let scope = arena.scope();
///     // allocations here ...
///     // all freed when `scope` drops
/// }
/// ```
pub struct ScopedArena {
    /// The underlying linear allocator.
    inner: LinearAllocator,
}

impl ScopedArena {
    /// Creates a scoped arena with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: LinearAllocator::new(capacity),
        }
    }

    /// Begins a new allocation scope.
    ///
    /// When the returned [`ArenaScope`] is dropped, all memory allocated
    /// through it is released.
    pub fn scope(&self) -> ArenaScope<'_> {
        let saved_offset = *self.inner.offset.lock();
        ArenaScope {
            arena: self,
            saved_offset,
        }
    }

    /// Allocates `layout.size()` bytes within the arena (outside of any scope).
    pub fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        <LinearAllocator as Allocator>::alloc(&self.inner, layout)
    }

    /// Returns current allocation stats.
    pub fn stats(&self) -> MemoryStats {
        self.inner.stats()
    }
}

/// RAII guard that rewinds the parent [`ScopedArena`] on drop.
pub struct ArenaScope<'a> {
    /// Reference to the parent arena.
    arena: &'a ScopedArena,
    /// Offset to restore when this scope ends.
    saved_offset: usize,
}

impl<'a> ArenaScope<'a> {
    /// Allocates within this scope.
    pub fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        self.arena.alloc(layout)
    }
}

impl<'a> Drop for ArenaScope<'a> {
    fn drop(&mut self) {
        *self.arena.inner.offset.lock() = self.saved_offset;
    }
}
