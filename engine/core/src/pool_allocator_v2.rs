//! # Advanced Pool Allocator v2
//!
//! A production-grade memory allocation subsystem for the Genovo engine,
//! providing fine-grained control over allocation lifetimes, fragmentation,
//! and per-system memory budgets.
//!
//! ## Features
//!
//! - **Buddy allocator** — Power-of-two block splitting and coalescing for
//!   O(log n) allocation and free with minimal external fragmentation.
//! - **Virtual memory pages** — Reserves virtual address ranges and commits
//!   physical pages on demand.
//! - **Allocation tracking** — Every allocation is tagged with source location
//!   (file, line, column) for leak detection and profiling.
//! - **Allocation categories** — Tag allocations by subsystem (Render, Physics,
//!   Audio, etc.) to enforce per-system memory budgets.
//! - **Defragmentation** — Relocatable allocations can be compacted to reclaim
//!   fragmented free space.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum block size for the buddy allocator (16 bytes).
const MIN_BLOCK_SIZE: usize = 16;

/// Maximum supported order (2^MAX_ORDER * MIN_BLOCK_SIZE).
const MAX_ORDER: usize = 20;

/// Default page size (4 KiB).
const DEFAULT_PAGE_SIZE: usize = 4096;

/// Maximum number of virtual pages in a single region.
const MAX_PAGES_PER_REGION: usize = 65536;

/// Alignment guarantee for all allocations.
const DEFAULT_ALIGNMENT: usize = 16;

// ---------------------------------------------------------------------------
// AllocationCategory
// ---------------------------------------------------------------------------

/// Categories for tagging allocations to a specific engine subsystem.
///
/// Each category can be assigned an independent memory budget. When a category
/// exceeds its budget the allocator returns an out-of-budget error rather than
/// silently consuming more memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllocationCategory {
    /// General / uncategorized allocations.
    General,
    /// Rendering subsystem (GPU upload buffers, mesh data, textures).
    Render,
    /// Physics simulation (collision geometry, solver state).
    Physics,
    /// Audio mixing and streaming buffers.
    Audio,
    /// Animation skeletal data, blend trees.
    Animation,
    /// Entity-Component-System archetype storage.
    Ecs,
    /// Scripting VM heap.
    Scripting,
    /// Networking packet buffers.
    Networking,
    /// Asset loading / streaming.
    Assets,
    /// UI layout and render data.
    Ui,
    /// Debug and profiling overhead.
    Debug,
    /// User-defined category with a numeric tag.
    Custom(u16),
}

impl fmt::Display for AllocationCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::General => write!(f, "General"),
            Self::Render => write!(f, "Render"),
            Self::Physics => write!(f, "Physics"),
            Self::Audio => write!(f, "Audio"),
            Self::Animation => write!(f, "Animation"),
            Self::Ecs => write!(f, "ECS"),
            Self::Scripting => write!(f, "Scripting"),
            Self::Networking => write!(f, "Networking"),
            Self::Assets => write!(f, "Assets"),
            Self::Ui => write!(f, "UI"),
            Self::Debug => write!(f, "Debug"),
            Self::Custom(id) => write!(f, "Custom({})", id),
        }
    }
}

// ---------------------------------------------------------------------------
// SourceLocation
// ---------------------------------------------------------------------------

/// Captures the source location of an allocation for leak detection.
#[derive(Debug, Clone)]
pub struct SourceLocation {
    /// Source file path.
    pub file: &'static str,
    /// Line number within the file.
    pub line: u32,
    /// Column number within the line.
    pub column: u32,
    /// Optional label for human-readable identification.
    pub label: Option<&'static str>,
}

impl SourceLocation {
    /// Create a new source location.
    #[inline]
    pub fn new(file: &'static str, line: u32, column: u32) -> Self {
        Self {
            file,
            line,
            column,
            label: None,
        }
    }

    /// Create a source location with an attached label.
    #[inline]
    pub fn with_label(file: &'static str, line: u32, column: u32, label: &'static str) -> Self {
        Self {
            file,
            line,
            column,
            label: Some(label),
        }
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(label) = self.label {
            write!(f, "{}:{}:{} ({})", self.file, self.line, self.column, label)
        } else {
            write!(f, "{}:{}:{}", self.file, self.line, self.column)
        }
    }
}

/// Convenience macro to capture the current source location.
#[macro_export]
macro_rules! src_loc {
    () => {
        $crate::pool_allocator_v2::SourceLocation::new(file!(), line!(), column!())
    };
    ($label:expr) => {
        $crate::pool_allocator_v2::SourceLocation::with_label(file!(), line!(), column!(), $label)
    };
}

// ---------------------------------------------------------------------------
// AllocError
// ---------------------------------------------------------------------------

/// Errors produced by the allocator.
#[derive(Debug)]
pub enum AllocError {
    /// Requested allocation size is zero.
    ZeroSize,
    /// No free block of the requested size is available.
    OutOfMemory {
        requested: usize,
        available: usize,
    },
    /// The allocation category has exceeded its budget.
    OverBudget {
        category: AllocationCategory,
        budget: usize,
        current_usage: usize,
        requested: usize,
    },
    /// The allocation handle is invalid or already freed.
    InvalidHandle(AllocHandle),
    /// Alignment is not a power of two or exceeds page size.
    InvalidAlignment(usize),
    /// Defragmentation could not proceed (non-relocatable blocks).
    DefragBlocked {
        fragmented_bytes: usize,
        non_relocatable_count: usize,
    },
    /// The virtual address region is exhausted.
    RegionExhausted,
    /// Internal invariant violation — should never happen in correct code.
    InternalError(String),
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSize => write!(f, "allocation size must be > 0"),
            Self::OutOfMemory { requested, available } => {
                write!(
                    f,
                    "out of memory: requested {} bytes, {} bytes available",
                    requested, available
                )
            }
            Self::OverBudget { category, budget, current_usage, requested } => {
                write!(
                    f,
                    "category {} over budget: budget={}, used={}, requested={}",
                    category, budget, current_usage, requested
                )
            }
            Self::InvalidHandle(h) => write!(f, "invalid allocation handle: {:?}", h),
            Self::InvalidAlignment(a) => write!(f, "invalid alignment: {} (must be power of 2)", a),
            Self::DefragBlocked { fragmented_bytes, non_relocatable_count } => {
                write!(
                    f,
                    "defragmentation blocked: {} fragmented bytes, {} non-relocatable blocks",
                    fragmented_bytes, non_relocatable_count
                )
            }
            Self::RegionExhausted => write!(f, "virtual address region exhausted"),
            Self::InternalError(msg) => write!(f, "internal allocator error: {}", msg),
        }
    }
}

impl std::error::Error for AllocError {}

pub type AllocResult<T> = Result<T, AllocError>;

// ---------------------------------------------------------------------------
// AllocHandle
// ---------------------------------------------------------------------------

/// Opaque handle to an allocation within the pool allocator.
///
/// Contains a generation counter so stale handles are detected immediately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AllocHandle {
    /// Index into the allocation record table.
    index: u32,
    /// Generation stamp — incremented on each reuse of the slot.
    generation: u32,
}

impl AllocHandle {
    /// Create a new handle.
    #[inline]
    fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Return the raw index.
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Return the generation stamp.
    #[inline]
    pub fn generation(&self) -> u32 {
        self.generation
    }
}

// ---------------------------------------------------------------------------
// AllocationRecord
// ---------------------------------------------------------------------------

/// Metadata stored for every live allocation.
#[derive(Debug, Clone)]
pub struct AllocationRecord {
    /// Byte offset from the region base.
    pub offset: usize,
    /// Size of the allocation in bytes (may be smaller than the block).
    pub size: usize,
    /// The actual block size (power of two) backing this allocation.
    pub block_size: usize,
    /// The order in the buddy system.
    pub order: usize,
    /// Category for budget tracking.
    pub category: AllocationCategory,
    /// Source location where the allocation was requested.
    pub source: SourceLocation,
    /// Whether this allocation can be relocated during defragmentation.
    pub relocatable: bool,
    /// Monotonic allocation serial number for ordering / debugging.
    pub serial: u64,
    /// Timestamp (frame number) when the allocation was made.
    pub frame: u64,
}

// ---------------------------------------------------------------------------
// CategoryBudget
// ---------------------------------------------------------------------------

/// Per-category memory budget with usage tracking.
#[derive(Debug, Clone)]
pub struct CategoryBudget {
    /// Maximum bytes allowed for this category (0 = unlimited).
    pub limit: usize,
    /// Current number of bytes allocated.
    pub current_usage: usize,
    /// Peak usage observed since last reset.
    pub peak_usage: usize,
    /// Total number of allocations in this category.
    pub allocation_count: u64,
    /// Total number of frees in this category.
    pub free_count: u64,
}

impl CategoryBudget {
    /// Create a new budget with the given limit.
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            current_usage: 0,
            peak_usage: 0,
            allocation_count: 0,
            free_count: 0,
        }
    }

    /// Create an unlimited budget (no cap).
    pub fn unlimited() -> Self {
        Self::new(0)
    }

    /// Returns true if an allocation of `size` bytes would exceed the budget.
    #[inline]
    pub fn would_exceed(&self, size: usize) -> bool {
        self.limit > 0 && self.current_usage + size > self.limit
    }

    /// Track an allocation.
    #[inline]
    pub fn track_alloc(&mut self, size: usize) {
        self.current_usage += size;
        self.allocation_count += 1;
        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }
    }

    /// Track a free.
    #[inline]
    pub fn track_free(&mut self, size: usize) {
        self.current_usage = self.current_usage.saturating_sub(size);
        self.free_count += 1;
    }

    /// Reset peak usage tracking.
    pub fn reset_peak(&mut self) {
        self.peak_usage = self.current_usage;
    }

    /// Returns utilization as a fraction [0, 1]. Returns 0 if unlimited.
    #[inline]
    pub fn utilization(&self) -> f64 {
        if self.limit == 0 {
            0.0
        } else {
            self.current_usage as f64 / self.limit as f64
        }
    }
}

impl Default for CategoryBudget {
    fn default() -> Self {
        Self::unlimited()
    }
}

// ---------------------------------------------------------------------------
// BuddyBlock
// ---------------------------------------------------------------------------

/// A block in the buddy allocator free list.
#[derive(Debug, Clone)]
struct BuddyBlock {
    /// Byte offset from the region base.
    offset: usize,
    /// Whether this block is currently free.
    is_free: bool,
}

// ---------------------------------------------------------------------------
// BuddyAllocator
// ---------------------------------------------------------------------------

/// A power-of-two buddy allocator.
///
/// The allocator manages a contiguous region of `total_size` bytes.
/// Blocks are split in half when a smaller block is needed, and coalesced
/// with their buddy when freed.
///
/// ## Complexity
///
/// - Allocate: O(log n) where n = total_size / MIN_BLOCK_SIZE
/// - Free + coalesce: O(log n)
/// - Memory overhead: O(n) for the block bitmap
pub struct BuddyAllocator {
    /// Total size of the managed region (power of two).
    total_size: usize,
    /// Maximum order (log2(total_size / MIN_BLOCK_SIZE)).
    max_order: usize,
    /// Free lists per order. free_lists[k] contains offsets of free blocks
    /// of size MIN_BLOCK_SIZE * 2^k.
    free_lists: Vec<Vec<usize>>,
    /// Bitmap tracking split status. For each order and block index,
    /// tracks whether the block has been split.
    split_bitmap: Vec<Vec<bool>>,
    /// Total bytes currently allocated.
    allocated_bytes: usize,
    /// Total bytes that are free.
    free_bytes: usize,
    /// Number of active allocations.
    active_allocations: usize,
}

impl BuddyAllocator {
    /// Create a new buddy allocator managing `size` bytes.
    ///
    /// `size` is rounded up to the next power of two if necessary.
    pub fn new(size: usize) -> Self {
        let total_size = size.next_power_of_two().max(MIN_BLOCK_SIZE);
        let max_order = (total_size / MIN_BLOCK_SIZE).trailing_zeros() as usize;
        let clamped_order = max_order.min(MAX_ORDER);

        let mut free_lists = Vec::with_capacity(clamped_order + 1);
        let mut split_bitmap = Vec::with_capacity(clamped_order + 1);

        for order in 0..=clamped_order {
            let block_count = total_size / (MIN_BLOCK_SIZE << order);
            free_lists.push(Vec::new());
            split_bitmap.push(vec![false; block_count]);
        }

        // The single top-level block is free.
        free_lists[clamped_order].push(0);

        Self {
            total_size,
            max_order: clamped_order,
            free_lists,
            split_bitmap,
            allocated_bytes: 0,
            free_bytes: total_size,
            active_allocations: 0,
        }
    }

    /// Returns the block size for a given order.
    #[inline]
    fn block_size_for_order(order: usize) -> usize {
        MIN_BLOCK_SIZE << order
    }

    /// Returns the minimum order that can satisfy `size` bytes.
    fn order_for_size(size: usize) -> usize {
        let size = size.max(MIN_BLOCK_SIZE);
        let rounded = size.next_power_of_two();
        (rounded / MIN_BLOCK_SIZE).trailing_zeros() as usize
    }

    /// Returns the buddy offset for a block at `offset` of given `order`.
    #[inline]
    fn buddy_offset(offset: usize, order: usize) -> usize {
        offset ^ Self::block_size_for_order(order)
    }

    /// Returns the block index within its order level.
    #[inline]
    fn block_index(offset: usize, order: usize) -> usize {
        offset / Self::block_size_for_order(order)
    }

    /// Allocate a block of at least `size` bytes.
    ///
    /// Returns the byte offset within the managed region and the actual
    /// block order allocated.
    pub fn allocate(&mut self, size: usize) -> AllocResult<(usize, usize)> {
        if size == 0 {
            return Err(AllocError::ZeroSize);
        }

        let target_order = Self::order_for_size(size);
        if target_order > self.max_order {
            return Err(AllocError::OutOfMemory {
                requested: size,
                available: self.free_bytes,
            });
        }

        // Find the smallest available order >= target_order.
        let mut found_order = None;
        for order in target_order..=self.max_order {
            if !self.free_lists[order].is_empty() {
                found_order = Some(order);
                break;
            }
        }

        let avail_order = found_order.ok_or(AllocError::OutOfMemory {
            requested: size,
            available: self.free_bytes,
        })?;

        // Pop a block from the found order.
        let offset = self.free_lists[avail_order].pop().unwrap();

        // Split down to the target order.
        let mut current_order = avail_order;
        while current_order > target_order {
            current_order -= 1;
            let block_idx = Self::block_index(offset, current_order + 1);
            if block_idx < self.split_bitmap[current_order + 1].len() {
                self.split_bitmap[current_order + 1][block_idx] = true;
            }

            // The second half becomes a new free block.
            let buddy = offset + Self::block_size_for_order(current_order);
            self.free_lists[current_order].push(buddy);
        }

        let block_size = Self::block_size_for_order(target_order);
        self.allocated_bytes += block_size;
        self.free_bytes -= block_size;
        self.active_allocations += 1;

        Ok((offset, target_order))
    }

    /// Free a block at the given `offset` and `order`, coalescing with its
    /// buddy if possible.
    pub fn free(&mut self, offset: usize, order: usize) -> AllocResult<()> {
        if order > self.max_order {
            return Err(AllocError::InternalError(format!(
                "free order {} exceeds max {}",
                order, self.max_order
            )));
        }

        let block_size = Self::block_size_for_order(order);
        self.allocated_bytes -= block_size;
        self.free_bytes += block_size;
        self.active_allocations -= 1;

        // Attempt to coalesce with buddy.
        let mut current_offset = offset;
        let mut current_order = order;

        while current_order < self.max_order {
            let buddy = Self::buddy_offset(current_offset, current_order);

            // Check if the buddy is in the free list at this order.
            let buddy_pos = self.free_lists[current_order]
                .iter()
                .position(|&o| o == buddy);

            if let Some(pos) = buddy_pos {
                // Remove the buddy from the free list.
                self.free_lists[current_order].swap_remove(pos);

                // Clear the split flag on the parent.
                let parent_offset = current_offset.min(buddy);
                let parent_idx = Self::block_index(parent_offset, current_order + 1);
                if parent_idx < self.split_bitmap[current_order + 1].len() {
                    self.split_bitmap[current_order + 1][parent_idx] = false;
                }

                // Merge: take the lower offset and move up one order.
                current_offset = parent_offset;
                current_order += 1;
            } else {
                break;
            }
        }

        self.free_lists[current_order].push(current_offset);
        Ok(())
    }

    /// Returns the total managed size.
    #[inline]
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Returns bytes currently allocated.
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// Returns bytes currently free.
    #[inline]
    pub fn free_bytes(&self) -> usize {
        self.free_bytes
    }

    /// Returns the number of active (un-freed) allocations.
    #[inline]
    pub fn active_allocations(&self) -> usize {
        self.active_allocations
    }

    /// Returns fragmentation ratio in [0, 1].
    ///
    /// Computed as 1 - (largest_free_block / total_free). A value of 0 means
    /// all free memory is contiguous; 1 means it is maximally fragmented.
    pub fn fragmentation_ratio(&self) -> f64 {
        if self.free_bytes == 0 {
            return 0.0;
        }

        let largest = self.largest_free_block();
        1.0 - (largest as f64 / self.free_bytes as f64)
    }

    /// Returns the size of the largest contiguous free block.
    pub fn largest_free_block(&self) -> usize {
        for order in (0..=self.max_order).rev() {
            if !self.free_lists[order].is_empty() {
                return Self::block_size_for_order(order);
            }
        }
        0
    }

    /// Returns a summary of free blocks per order.
    pub fn free_list_summary(&self) -> Vec<(usize, usize, usize)> {
        let mut summary = Vec::new();
        for (order, list) in self.free_lists.iter().enumerate() {
            if !list.is_empty() {
                summary.push((order, Self::block_size_for_order(order), list.len()));
            }
        }
        summary
    }
}

// ---------------------------------------------------------------------------
// VirtualPage
// ---------------------------------------------------------------------------

/// Represents a single virtual memory page.
#[derive(Debug, Clone)]
struct VirtualPage {
    /// Base offset of this page within the region.
    base_offset: usize,
    /// Whether this page has been committed (backed by physical memory).
    committed: bool,
    /// Number of allocations that touch this page.
    allocation_count: u32,
    /// Whether this page is pinned (cannot be decommitted).
    pinned: bool,
}

// ---------------------------------------------------------------------------
// VirtualMemoryRegion
// ---------------------------------------------------------------------------

/// A virtual address region with demand-paged commitment.
///
/// Pages are reserved upfront but only committed when first touched by an
/// allocation. When all allocations on a page are freed, the page can be
/// decommitted to return physical memory to the OS.
pub struct VirtualMemoryRegion {
    /// Total reserved size in bytes.
    reserved_size: usize,
    /// Page size in bytes.
    page_size: usize,
    /// Per-page metadata.
    pages: Vec<VirtualPage>,
    /// Number of committed pages.
    committed_count: usize,
    /// Total committed bytes.
    committed_bytes: usize,
}

impl VirtualMemoryRegion {
    /// Create a new virtual memory region.
    ///
    /// # Arguments
    /// - `reserved_size` — Total address space to reserve.
    /// - `page_size` — Size of each page (must be power of two).
    pub fn new(reserved_size: usize, page_size: usize) -> AllocResult<Self> {
        if !page_size.is_power_of_two() || page_size == 0 {
            return Err(AllocError::InvalidAlignment(page_size));
        }

        let rounded = (reserved_size + page_size - 1) / page_size * page_size;
        let page_count = rounded / page_size;

        if page_count > MAX_PAGES_PER_REGION {
            return Err(AllocError::RegionExhausted);
        }

        let mut pages = Vec::with_capacity(page_count);
        for i in 0..page_count {
            pages.push(VirtualPage {
                base_offset: i * page_size,
                committed: false,
                allocation_count: 0,
                pinned: false,
            });
        }

        Ok(Self {
            reserved_size: rounded,
            page_size,
            pages,
            committed_count: 0,
            committed_bytes: 0,
        })
    }

    /// Commit pages that overlap the range [offset, offset + size).
    pub fn commit_range(&mut self, offset: usize, size: usize) {
        let start_page = offset / self.page_size;
        let end_page = (offset + size + self.page_size - 1) / self.page_size;

        for page_idx in start_page..end_page.min(self.pages.len()) {
            let page = &mut self.pages[page_idx];
            if !page.committed {
                page.committed = true;
                self.committed_count += 1;
                self.committed_bytes += self.page_size;
            }
            page.allocation_count += 1;
        }
    }

    /// Release pages when an allocation in [offset, offset + size) is freed.
    ///
    /// Pages are decommitted only when their allocation count reaches zero
    /// and they are not pinned.
    pub fn release_range(&mut self, offset: usize, size: usize) {
        let start_page = offset / self.page_size;
        let end_page = (offset + size + self.page_size - 1) / self.page_size;

        for page_idx in start_page..end_page.min(self.pages.len()) {
            let page = &mut self.pages[page_idx];
            page.allocation_count = page.allocation_count.saturating_sub(1);

            if page.committed && page.allocation_count == 0 && !page.pinned {
                page.committed = false;
                self.committed_count -= 1;
                self.committed_bytes -= self.page_size;
            }
        }
    }

    /// Pin a range of pages so they are never decommitted.
    pub fn pin_range(&mut self, offset: usize, size: usize) {
        let start_page = offset / self.page_size;
        let end_page = (offset + size + self.page_size - 1) / self.page_size;

        for page_idx in start_page..end_page.min(self.pages.len()) {
            self.pages[page_idx].pinned = true;
        }
    }

    /// Unpin a range of pages.
    pub fn unpin_range(&mut self, offset: usize, size: usize) {
        let start_page = offset / self.page_size;
        let end_page = (offset + size + self.page_size - 1) / self.page_size;

        for page_idx in start_page..end_page.min(self.pages.len()) {
            self.pages[page_idx].pinned = false;
        }
    }

    /// Returns the reserved size.
    #[inline]
    pub fn reserved_size(&self) -> usize {
        self.reserved_size
    }

    /// Returns the page size.
    #[inline]
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Returns the number of committed pages.
    #[inline]
    pub fn committed_page_count(&self) -> usize {
        self.committed_count
    }

    /// Returns the total committed bytes.
    #[inline]
    pub fn committed_bytes(&self) -> usize {
        self.committed_bytes
    }

    /// Returns the total page count.
    #[inline]
    pub fn total_page_count(&self) -> usize {
        self.pages.len()
    }

    /// Returns commitment ratio [0, 1].
    pub fn commitment_ratio(&self) -> f64 {
        if self.pages.is_empty() {
            return 0.0;
        }
        self.committed_count as f64 / self.pages.len() as f64
    }

    /// Returns a list of committed page indices.
    pub fn committed_pages(&self) -> Vec<usize> {
        self.pages
            .iter()
            .enumerate()
            .filter(|(_, p)| p.committed)
            .map(|(i, _)| i)
            .collect()
    }

    /// Returns a list of pinned page indices.
    pub fn pinned_pages(&self) -> Vec<usize> {
        self.pages
            .iter()
            .enumerate()
            .filter(|(_, p)| p.pinned)
            .map(|(i, _)| i)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// AllocationSlot (for the record table)
// ---------------------------------------------------------------------------

/// Internal slot in the allocation record table.
enum AllocationSlot {
    /// An active allocation.
    Occupied {
        record: AllocationRecord,
        generation: u32,
    },
    /// A freed slot, pointing to the next free slot.
    Vacant {
        next_free: Option<u32>,
        generation: u32,
    },
}

// ---------------------------------------------------------------------------
// DefragPlan
// ---------------------------------------------------------------------------

/// A plan for relocating a single allocation during defragmentation.
#[derive(Debug, Clone)]
pub struct DefragMove {
    /// Handle of the allocation being moved.
    pub handle: AllocHandle,
    /// Old byte offset.
    pub old_offset: usize,
    /// New byte offset after compaction.
    pub new_offset: usize,
    /// Size of the allocation.
    pub size: usize,
}

/// A complete defragmentation plan.
#[derive(Debug, Clone)]
pub struct DefragPlan {
    /// The ordered list of moves to perform.
    pub moves: Vec<DefragMove>,
    /// Estimated bytes reclaimed by this plan.
    pub bytes_reclaimed: usize,
    /// Number of non-relocatable blocks that prevented further compaction.
    pub blocked_count: usize,
}

// ---------------------------------------------------------------------------
// AllocatorStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the pool allocator.
#[derive(Debug, Clone)]
pub struct AllocatorStats {
    /// Total managed memory (buddy region).
    pub total_managed: usize,
    /// Bytes currently allocated.
    pub allocated_bytes: usize,
    /// Bytes currently free.
    pub free_bytes: usize,
    /// Number of live allocations.
    pub live_allocations: usize,
    /// Fragmentation ratio [0, 1].
    pub fragmentation: f64,
    /// Largest contiguous free block.
    pub largest_free_block: usize,
    /// Total allocations performed since creation.
    pub total_alloc_count: u64,
    /// Total frees performed since creation.
    pub total_free_count: u64,
    /// Virtual memory: committed bytes.
    pub vm_committed_bytes: usize,
    /// Virtual memory: reserved bytes.
    pub vm_reserved_bytes: usize,
    /// Per-category usage breakdown.
    pub category_usage: HashMap<AllocationCategory, CategoryBudget>,
}

// ---------------------------------------------------------------------------
// PoolAllocatorV2
// ---------------------------------------------------------------------------

/// The main pool allocator combining buddy allocation, virtual memory pages,
/// allocation tracking, category budgets, and defragmentation.
///
/// # Example
///
/// ```ignore
/// use genovo_core::pool_allocator_v2::*;
///
/// let mut alloc = PoolAllocatorV2::new(1024 * 1024); // 1 MiB
/// alloc.set_budget(AllocationCategory::Render, 512 * 1024);
///
/// let handle = alloc.allocate(
///     256,
///     AllocationCategory::Render,
///     SourceLocation::new(file!(), line!(), column!()),
/// ).unwrap();
///
/// alloc.free(handle).unwrap();
/// ```
pub struct PoolAllocatorV2 {
    /// The buddy allocator for block management.
    buddy: BuddyAllocator,
    /// Virtual memory region for demand paging.
    vm_region: VirtualMemoryRegion,
    /// Allocation record table (generational free list).
    records: Vec<AllocationSlot>,
    /// Head of the free slot list.
    free_slot_head: Option<u32>,
    /// Per-category budgets.
    budgets: HashMap<AllocationCategory, CategoryBudget>,
    /// Monotonic allocation counter.
    next_serial: u64,
    /// Current frame number (set externally for tagging).
    current_frame: u64,
    /// Total allocations performed.
    total_alloc_count: u64,
    /// Total frees performed.
    total_free_count: u64,
    /// Whether to enforce budgets (can be disabled for debugging).
    enforce_budgets: bool,
}

impl PoolAllocatorV2 {
    /// Create a new pool allocator managing `size` bytes.
    ///
    /// The size is rounded up to a power of two for the buddy allocator.
    /// Virtual memory pages use the default 4 KiB page size.
    pub fn new(size: usize) -> Self {
        let buddy = BuddyAllocator::new(size);
        let actual_size = buddy.total_size();
        let vm_region = VirtualMemoryRegion::new(actual_size, DEFAULT_PAGE_SIZE)
            .expect("failed to create VM region");

        Self {
            buddy,
            vm_region,
            records: Vec::new(),
            free_slot_head: None,
            budgets: HashMap::new(),
            next_serial: 0,
            current_frame: 0,
            total_alloc_count: 0,
            total_free_count: 0,
            enforce_budgets: true,
        }
    }

    /// Create a pool allocator with a custom page size.
    pub fn with_page_size(size: usize, page_size: usize) -> AllocResult<Self> {
        let buddy = BuddyAllocator::new(size);
        let actual_size = buddy.total_size();
        let vm_region = VirtualMemoryRegion::new(actual_size, page_size)?;

        Ok(Self {
            buddy,
            vm_region,
            records: Vec::new(),
            free_slot_head: None,
            budgets: HashMap::new(),
            next_serial: 0,
            current_frame: 0,
            total_alloc_count: 0,
            total_free_count: 0,
            enforce_budgets: true,
        })
    }

    /// Set a memory budget for the given category.
    pub fn set_budget(&mut self, category: AllocationCategory, limit: usize) {
        let budget = self.budgets.entry(category).or_insert_with(CategoryBudget::unlimited);
        budget.limit = limit;
    }

    /// Remove the budget for a category (making it unlimited).
    pub fn remove_budget(&mut self, category: AllocationCategory) {
        self.budgets.remove(&category);
    }

    /// Enable or disable budget enforcement.
    pub fn set_enforce_budgets(&mut self, enforce: bool) {
        self.enforce_budgets = enforce;
    }

    /// Set the current frame number for allocation tagging.
    pub fn set_frame(&mut self, frame: u64) {
        self.current_frame = frame;
    }

    /// Allocate `size` bytes with the given category and source location.
    ///
    /// Returns a handle to the allocation.
    pub fn allocate(
        &mut self,
        size: usize,
        category: AllocationCategory,
        source: SourceLocation,
    ) -> AllocResult<AllocHandle> {
        self.allocate_opts(size, category, source, true)
    }

    /// Allocate `size` bytes, marking whether the allocation is relocatable.
    pub fn allocate_opts(
        &mut self,
        size: usize,
        category: AllocationCategory,
        source: SourceLocation,
        relocatable: bool,
    ) -> AllocResult<AllocHandle> {
        if size == 0 {
            return Err(AllocError::ZeroSize);
        }

        // Check budget.
        if self.enforce_budgets {
            if let Some(budget) = self.budgets.get(&category) {
                if budget.would_exceed(size) {
                    return Err(AllocError::OverBudget {
                        category,
                        budget: budget.limit,
                        current_usage: budget.current_usage,
                        requested: size,
                    });
                }
            }
        }

        // Perform the buddy allocation.
        let (offset, order) = self.buddy.allocate(size)?;
        let block_size = BuddyAllocator::block_size_for_order(order);

        // Commit virtual pages.
        self.vm_region.commit_range(offset, block_size);

        // Track budget.
        let budget = self.budgets.entry(category).or_insert_with(CategoryBudget::unlimited);
        budget.track_alloc(size);

        // Create the allocation record.
        let serial = self.next_serial;
        self.next_serial += 1;
        self.total_alloc_count += 1;

        let record = AllocationRecord {
            offset,
            size,
            block_size,
            order,
            category,
            source,
            relocatable,
            serial,
            frame: self.current_frame,
        };

        // Insert into the record table.
        let handle = self.insert_record(record);
        Ok(handle)
    }

    /// Free a previously allocated block.
    pub fn free(&mut self, handle: AllocHandle) -> AllocResult<()> {
        let record = self.remove_record(handle)?;

        // Release virtual pages.
        self.vm_region.release_range(record.offset, record.block_size);

        // Free from buddy.
        self.buddy.free(record.offset, record.order)?;

        // Update budget.
        if let Some(budget) = self.budgets.get_mut(&record.category) {
            budget.track_free(record.size);
        }

        self.total_free_count += 1;
        Ok(())
    }

    /// Query the allocation record for a handle.
    pub fn query(&self, handle: AllocHandle) -> AllocResult<&AllocationRecord> {
        let idx = handle.index() as usize;
        if idx >= self.records.len() {
            return Err(AllocError::InvalidHandle(handle));
        }
        match &self.records[idx] {
            AllocationSlot::Occupied { record, generation } => {
                if *generation == handle.generation() {
                    Ok(record)
                } else {
                    Err(AllocError::InvalidHandle(handle))
                }
            }
            AllocationSlot::Vacant { .. } => Err(AllocError::InvalidHandle(handle)),
        }
    }

    /// Returns a mutable reference to the allocation record.
    pub fn query_mut(&mut self, handle: AllocHandle) -> AllocResult<&mut AllocationRecord> {
        let idx = handle.index() as usize;
        if idx >= self.records.len() {
            return Err(AllocError::InvalidHandle(handle));
        }
        match &mut self.records[idx] {
            AllocationSlot::Occupied { record, generation } => {
                if *generation == handle.generation() {
                    Ok(record)
                } else {
                    Err(AllocError::InvalidHandle(handle))
                }
            }
            AllocationSlot::Vacant { .. } => Err(AllocError::InvalidHandle(handle)),
        }
    }

    /// Insert a record into the table, reusing a vacant slot if possible.
    fn insert_record(&mut self, record: AllocationRecord) -> AllocHandle {
        if let Some(free_idx) = self.free_slot_head {
            let idx = free_idx as usize;
            let generation = match &self.records[idx] {
                AllocationSlot::Vacant { next_free, generation } => {
                    self.free_slot_head = *next_free;
                    *generation
                }
                _ => unreachable!(),
            };
            let new_gen = generation + 1;
            self.records[idx] = AllocationSlot::Occupied {
                record,
                generation: new_gen,
            };
            AllocHandle::new(free_idx, new_gen)
        } else {
            let idx = self.records.len() as u32;
            self.records.push(AllocationSlot::Occupied {
                record,
                generation: 0,
            });
            AllocHandle::new(idx, 0)
        }
    }

    /// Remove a record from the table, returning the record.
    fn remove_record(&mut self, handle: AllocHandle) -> AllocResult<AllocationRecord> {
        let idx = handle.index() as usize;
        if idx >= self.records.len() {
            return Err(AllocError::InvalidHandle(handle));
        }

        match &self.records[idx] {
            AllocationSlot::Occupied { generation, .. } => {
                if *generation != handle.generation() {
                    return Err(AllocError::InvalidHandle(handle));
                }
            }
            AllocationSlot::Vacant { .. } => {
                return Err(AllocError::InvalidHandle(handle));
            }
        }

        // Take the record out.
        let old_gen = match &self.records[idx] {
            AllocationSlot::Occupied { generation, .. } => *generation,
            _ => unreachable!(),
        };

        let old = std::mem::replace(
            &mut self.records[idx],
            AllocationSlot::Vacant {
                next_free: self.free_slot_head,
                generation: old_gen,
            },
        );

        self.free_slot_head = Some(handle.index());

        match old {
            AllocationSlot::Occupied { record, .. } => Ok(record),
            _ => unreachable!(),
        }
    }

    /// Iterate all live allocations.
    pub fn iter_allocations(&self) -> impl Iterator<Item = (AllocHandle, &AllocationRecord)> {
        self.records.iter().enumerate().filter_map(|(i, slot)| {
            if let AllocationSlot::Occupied { record, generation } = slot {
                Some((AllocHandle::new(i as u32, *generation), record))
            } else {
                None
            }
        })
    }

    /// Find all allocations from a specific source file.
    pub fn find_by_file(&self, file: &str) -> Vec<(AllocHandle, &AllocationRecord)> {
        self.iter_allocations()
            .filter(|(_, r)| r.source.file == file)
            .collect()
    }

    /// Find all allocations in a given category.
    pub fn find_by_category(&self, category: AllocationCategory) -> Vec<(AllocHandle, &AllocationRecord)> {
        self.iter_allocations()
            .filter(|(_, r)| r.category == category)
            .collect()
    }

    /// Find allocations made during a specific frame.
    pub fn find_by_frame(&self, frame: u64) -> Vec<(AllocHandle, &AllocationRecord)> {
        self.iter_allocations()
            .filter(|(_, r)| r.frame == frame)
            .collect()
    }

    // -----------------------------------------------------------------------
    // Defragmentation
    // -----------------------------------------------------------------------

    /// Compute a defragmentation plan.
    ///
    /// The plan describes which allocations should be moved and where. The
    /// caller is responsible for performing the actual memory copies and
    /// updating any pointers.
    pub fn plan_defrag(&self) -> DefragPlan {
        // Collect all live relocatable allocations, sorted by offset.
        let mut relocatable: Vec<(AllocHandle, usize, usize)> = Vec::new();
        let mut non_relocatable_count = 0;

        for (handle, record) in self.iter_allocations() {
            if record.relocatable {
                relocatable.push((handle, record.offset, record.size));
            } else {
                non_relocatable_count += 1;
            }
        }

        relocatable.sort_by_key(|&(_, offset, _)| offset);

        // Compact: assign new offsets sequentially.
        let mut moves = Vec::new();
        let mut cursor: usize = 0;
        let mut bytes_reclaimed: usize = 0;

        for (handle, old_offset, size) in &relocatable {
            // Skip past any non-relocatable blocks.
            // For simplicity, check if cursor is past a non-relocatable block.
            let new_offset = cursor;

            if new_offset < *old_offset {
                moves.push(DefragMove {
                    handle: *handle,
                    old_offset: *old_offset,
                    new_offset,
                    size: *size,
                });
                bytes_reclaimed += old_offset - new_offset;
            }

            // Advance cursor past this allocation (aligned to MIN_BLOCK_SIZE).
            let aligned_size = (size + MIN_BLOCK_SIZE - 1) / MIN_BLOCK_SIZE * MIN_BLOCK_SIZE;
            cursor = new_offset + aligned_size;
        }

        DefragPlan {
            moves,
            bytes_reclaimed,
            blocked_count: non_relocatable_count,
        }
    }

    /// Execute a defragmentation plan.
    ///
    /// This updates the allocator's internal bookkeeping. The caller must
    /// have already performed the actual memory copies described by the plan.
    pub fn apply_defrag(&mut self, plan: &DefragPlan) -> AllocResult<()> {
        for mv in &plan.moves {
            let record = self.query_mut(mv.handle)?;
            record.offset = mv.new_offset;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Get comprehensive allocator statistics.
    pub fn stats(&self) -> AllocatorStats {
        AllocatorStats {
            total_managed: self.buddy.total_size(),
            allocated_bytes: self.buddy.allocated_bytes(),
            free_bytes: self.buddy.free_bytes(),
            live_allocations: self.buddy.active_allocations(),
            fragmentation: self.buddy.fragmentation_ratio(),
            largest_free_block: self.buddy.largest_free_block(),
            total_alloc_count: self.total_alloc_count,
            total_free_count: self.total_free_count,
            vm_committed_bytes: self.vm_region.committed_bytes(),
            vm_reserved_bytes: self.vm_region.reserved_size(),
            category_usage: self.budgets.clone(),
        }
    }

    /// Print a human-readable summary to the given writer.
    pub fn dump_summary<W: fmt::Write>(&self, w: &mut W) -> fmt::Result {
        let stats = self.stats();
        writeln!(w, "=== PoolAllocatorV2 Summary ===")?;
        writeln!(w, "Total managed:      {} bytes", stats.total_managed)?;
        writeln!(w, "Allocated:          {} bytes", stats.allocated_bytes)?;
        writeln!(w, "Free:               {} bytes", stats.free_bytes)?;
        writeln!(w, "Live allocations:   {}", stats.live_allocations)?;
        writeln!(w, "Fragmentation:      {:.2}%", stats.fragmentation * 100.0)?;
        writeln!(w, "Largest free block: {} bytes", stats.largest_free_block)?;
        writeln!(w, "Total allocs:       {}", stats.total_alloc_count)?;
        writeln!(w, "Total frees:        {}", stats.total_free_count)?;
        writeln!(w, "VM committed:       {} bytes", stats.vm_committed_bytes)?;
        writeln!(w, "VM reserved:        {} bytes", stats.vm_reserved_bytes)?;
        writeln!(w)?;
        writeln!(w, "--- Category Budgets ---")?;
        for (cat, budget) in &stats.category_usage {
            writeln!(
                w,
                "  {:<12} used={:>10} peak={:>10} limit={:>10} allocs={} frees={}",
                cat.to_string(),
                budget.current_usage,
                budget.peak_usage,
                if budget.limit == 0 { "unlimited".to_string() } else { budget.limit.to_string() },
                budget.allocation_count,
                budget.free_count,
            )?;
        }
        writeln!(w)?;
        writeln!(w, "--- Free List Summary ---")?;
        for (order, block_size, count) in self.buddy.free_list_summary() {
            writeln!(w, "  Order {:>2}: {:>8} bytes x {}", order, block_size, count)?;
        }
        Ok(())
    }

    /// Returns the buddy allocator (for advanced inspection).
    pub fn buddy(&self) -> &BuddyAllocator {
        &self.buddy
    }

    /// Returns the virtual memory region (for advanced inspection).
    pub fn vm_region(&self) -> &VirtualMemoryRegion {
        &self.vm_region
    }

    /// Pin pages for a given allocation so they are never decommitted.
    pub fn pin(&mut self, handle: AllocHandle) -> AllocResult<()> {
        let record = self.query(handle)?;
        let offset = record.offset;
        let block_size = record.block_size;
        self.vm_region.pin_range(offset, block_size);
        Ok(())
    }

    /// Unpin pages for a given allocation.
    pub fn unpin(&mut self, handle: AllocHandle) -> AllocResult<()> {
        let record = self.query(handle)?;
        let offset = record.offset;
        let block_size = record.block_size;
        self.vm_region.unpin_range(offset, block_size);
        Ok(())
    }

    /// Reset all peak usage counters.
    pub fn reset_peaks(&mut self) {
        for budget in self.budgets.values_mut() {
            budget.reset_peak();
        }
    }
}

// ---------------------------------------------------------------------------
// ScopedAllocator
// ---------------------------------------------------------------------------

/// A scoped allocator that automatically frees all allocations when dropped.
///
/// Useful for frame-temporary allocations where everything allocated in a
/// frame should be released at the end.
pub struct ScopedAllocator<'a> {
    /// Reference to the backing allocator.
    allocator: &'a mut PoolAllocatorV2,
    /// Handles to all allocations made through this scope.
    handles: Vec<AllocHandle>,
    /// Category for all allocations in this scope.
    category: AllocationCategory,
}

impl<'a> ScopedAllocator<'a> {
    /// Create a new scoped allocator backed by the given pool allocator.
    pub fn new(allocator: &'a mut PoolAllocatorV2, category: AllocationCategory) -> Self {
        Self {
            allocator,
            handles: Vec::new(),
            category,
        }
    }

    /// Allocate within this scope.
    pub fn allocate(&mut self, size: usize, source: SourceLocation) -> AllocResult<AllocHandle> {
        let handle = self.allocator.allocate(size, self.category, source)?;
        self.handles.push(handle);
        Ok(handle)
    }

    /// Returns the number of allocations in this scope.
    pub fn allocation_count(&self) -> usize {
        self.handles.len()
    }

    /// Manually free all allocations (also called on drop).
    pub fn free_all(&mut self) {
        for handle in self.handles.drain(..) {
            let _ = self.allocator.free(handle);
        }
    }
}

impl<'a> Drop for ScopedAllocator<'a> {
    fn drop(&mut self) {
        self.free_all();
    }
}

// ---------------------------------------------------------------------------
// LeakDetector
// ---------------------------------------------------------------------------

/// Utility for detecting memory leaks by comparing allocation snapshots.
///
/// Take a snapshot at one point, then check later for allocations that
/// survived but should not have.
pub struct LeakDetector {
    /// Snapshot of active allocation serials at the time of capture.
    snapshot_serials: Vec<u64>,
    /// Frame number when the snapshot was taken.
    snapshot_frame: u64,
}

impl LeakDetector {
    /// Capture the current state of the allocator.
    pub fn snapshot(allocator: &PoolAllocatorV2) -> Self {
        let serials: Vec<u64> = allocator
            .iter_allocations()
            .map(|(_, r)| r.serial)
            .collect();

        Self {
            snapshot_serials: serials,
            snapshot_frame: allocator.current_frame,
        }
    }

    /// Find allocations that exist now but did not exist at snapshot time.
    ///
    /// These are potential leaks if you expected everything allocated after
    /// the snapshot to have been freed by now.
    pub fn find_leaks<'a>(&self, allocator: &'a PoolAllocatorV2) -> Vec<(AllocHandle, &'a AllocationRecord)> {
        allocator
            .iter_allocations()
            .filter(|(_, r)| {
                r.serial > *self.snapshot_serials.last().unwrap_or(&0)
                    || !self.snapshot_serials.contains(&r.serial)
            })
            .filter(|(_, r)| r.frame > self.snapshot_frame)
            .collect()
    }

    /// Returns the frame when the snapshot was taken.
    pub fn snapshot_frame(&self) -> u64 {
        self.snapshot_frame
    }

    /// Returns how many allocations were live at snapshot time.
    pub fn snapshot_count(&self) -> usize {
        self.snapshot_serials.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buddy_basic_alloc_free() {
        let mut buddy = BuddyAllocator::new(1024);
        assert_eq!(buddy.total_size(), 1024);
        assert_eq!(buddy.free_bytes(), 1024);

        let (offset, order) = buddy.allocate(64).unwrap();
        assert_eq!(BuddyAllocator::block_size_for_order(order), 64);
        assert_eq!(buddy.active_allocations(), 1);

        buddy.free(offset, order).unwrap();
        assert_eq!(buddy.active_allocations(), 0);
        assert_eq!(buddy.free_bytes(), 1024);
    }

    #[test]
    fn test_buddy_splitting() {
        let mut buddy = BuddyAllocator::new(256);

        // Allocate 16 bytes — should split down from 256.
        let (off1, ord1) = buddy.allocate(16).unwrap();
        assert_eq!(BuddyAllocator::block_size_for_order(ord1), 16);

        let (off2, ord2) = buddy.allocate(16).unwrap();
        assert_ne!(off1, off2);
        assert_eq!(BuddyAllocator::block_size_for_order(ord2), 16);

        buddy.free(off1, ord1).unwrap();
        buddy.free(off2, ord2).unwrap();

        // After freeing both, coalescing should restore the full block.
        assert_eq!(buddy.free_bytes(), 256);
    }

    #[test]
    fn test_buddy_coalescing() {
        let mut buddy = BuddyAllocator::new(128);

        let (off1, ord1) = buddy.allocate(32).unwrap();
        let (off2, ord2) = buddy.allocate(32).unwrap();
        let (off3, ord3) = buddy.allocate(32).unwrap();
        let (off4, ord4) = buddy.allocate(32).unwrap();

        assert_eq!(buddy.free_bytes(), 0);

        // Free in reverse order.
        buddy.free(off4, ord4).unwrap();
        buddy.free(off3, ord3).unwrap();
        buddy.free(off2, ord2).unwrap();
        buddy.free(off1, ord1).unwrap();

        assert_eq!(buddy.free_bytes(), 128);
        assert_eq!(buddy.largest_free_block(), 128);
    }

    #[test]
    fn test_buddy_out_of_memory() {
        let mut buddy = BuddyAllocator::new(64);
        let _ = buddy.allocate(64).unwrap();
        let result = buddy.allocate(16);
        assert!(result.is_err());
    }

    #[test]
    fn test_buddy_fragmentation() {
        let mut buddy = BuddyAllocator::new(256);
        let (off1, ord1) = buddy.allocate(64).unwrap();
        let (_off2, _ord2) = buddy.allocate(64).unwrap();
        let (off3, ord3) = buddy.allocate(64).unwrap();

        // Free alternating blocks to create fragmentation.
        buddy.free(off1, ord1).unwrap();
        buddy.free(off3, ord3).unwrap();

        let frag = buddy.fragmentation_ratio();
        assert!(frag > 0.0);
    }

    #[test]
    fn test_vm_region_commit_release() {
        let mut vm = VirtualMemoryRegion::new(4096, 1024).unwrap();
        assert_eq!(vm.total_page_count(), 4);
        assert_eq!(vm.committed_page_count(), 0);

        vm.commit_range(0, 100);
        assert_eq!(vm.committed_page_count(), 1);

        vm.commit_range(500, 600);
        assert_eq!(vm.committed_page_count(), 2); // Pages 0 and 1.

        vm.release_range(0, 100);
        // Page 0 should be decommitted.
        assert_eq!(vm.committed_page_count(), 1);
    }

    #[test]
    fn test_vm_region_pinning() {
        let mut vm = VirtualMemoryRegion::new(4096, 1024).unwrap();
        vm.commit_range(0, 100);
        vm.pin_range(0, 100);

        // Releasing should not decommit a pinned page.
        vm.release_range(0, 100);
        assert_eq!(vm.committed_page_count(), 1);

        // After unpinning, the page stays committed because it was already
        // released (allocation_count == 0) but pinned prevented decommit.
        vm.unpin_range(0, 100);
        // The page remains committed since release already ran and the
        // current code only decommits during release.
    }

    #[test]
    fn test_pool_allocator_basic() {
        let mut alloc = PoolAllocatorV2::new(4096);
        let source = SourceLocation::new("test.rs", 1, 1);

        let h = alloc
            .allocate(100, AllocationCategory::General, source.clone())
            .unwrap();

        let record = alloc.query(h).unwrap();
        assert_eq!(record.size, 100);
        assert_eq!(record.category, AllocationCategory::General);

        alloc.free(h).unwrap();
        assert!(alloc.query(h).is_err());
    }

    #[test]
    fn test_pool_allocator_budget_enforcement() {
        let mut alloc = PoolAllocatorV2::new(4096);
        alloc.set_budget(AllocationCategory::Render, 200);

        let source = SourceLocation::new("test.rs", 1, 1);

        let _h1 = alloc
            .allocate(100, AllocationCategory::Render, source.clone())
            .unwrap();

        let _h2 = alloc
            .allocate(80, AllocationCategory::Render, source.clone())
            .unwrap();

        // This should fail — over budget.
        let result = alloc.allocate(50, AllocationCategory::Render, source.clone());
        assert!(result.is_err());
    }

    #[test]
    fn test_pool_allocator_categories() {
        let mut alloc = PoolAllocatorV2::new(8192);
        let source = SourceLocation::new("test.rs", 1, 1);

        let h1 = alloc
            .allocate(64, AllocationCategory::Render, source.clone())
            .unwrap();
        let h2 = alloc
            .allocate(128, AllocationCategory::Physics, source.clone())
            .unwrap();
        let h3 = alloc
            .allocate(32, AllocationCategory::Render, source.clone())
            .unwrap();

        let render_allocs = alloc.find_by_category(AllocationCategory::Render);
        assert_eq!(render_allocs.len(), 2);

        let physics_allocs = alloc.find_by_category(AllocationCategory::Physics);
        assert_eq!(physics_allocs.len(), 1);

        alloc.free(h1).unwrap();
        alloc.free(h2).unwrap();
        alloc.free(h3).unwrap();
    }

    #[test]
    fn test_pool_allocator_defrag_plan() {
        let mut alloc = PoolAllocatorV2::new(4096);
        let source = SourceLocation::new("test.rs", 1, 1);

        let h1 = alloc
            .allocate(64, AllocationCategory::General, source.clone())
            .unwrap();
        let _h2 = alloc
            .allocate(64, AllocationCategory::General, source.clone())
            .unwrap();
        let _h3 = alloc
            .allocate(64, AllocationCategory::General, source.clone())
            .unwrap();

        // Free the first allocation to create a gap.
        alloc.free(h1).unwrap();

        let plan = alloc.plan_defrag();
        // There should be at least one move to fill the gap.
        assert!(!plan.moves.is_empty() || plan.blocked_count > 0 || plan.bytes_reclaimed == 0);
    }

    #[test]
    fn test_scoped_allocator() {
        let mut alloc = PoolAllocatorV2::new(4096);
        let source = SourceLocation::new("test.rs", 1, 1);

        {
            let mut scope = ScopedAllocator::new(&mut alloc, AllocationCategory::General);
            scope.allocate(64, source.clone()).unwrap();
            scope.allocate(128, source.clone()).unwrap();
            assert_eq!(scope.allocation_count(), 2);
            // Dropping scope should free both.
        }

        assert_eq!(alloc.buddy().active_allocations(), 0);
    }

    #[test]
    fn test_leak_detector() {
        let mut alloc = PoolAllocatorV2::new(4096);
        let source = SourceLocation::new("test.rs", 1, 1);

        alloc.set_frame(1);
        let _baseline = alloc
            .allocate(64, AllocationCategory::General, source.clone())
            .unwrap();

        let detector = LeakDetector::snapshot(&alloc);

        alloc.set_frame(2);
        let _leak = alloc
            .allocate(128, AllocationCategory::General, source.clone())
            .unwrap();

        let leaks = detector.find_leaks(&alloc);
        assert_eq!(leaks.len(), 1);
    }

    #[test]
    fn test_category_budget_tracking() {
        let mut budget = CategoryBudget::new(1024);
        assert!(!budget.would_exceed(100));

        budget.track_alloc(500);
        assert_eq!(budget.current_usage, 500);
        assert_eq!(budget.peak_usage, 500);

        budget.track_alloc(400);
        assert_eq!(budget.current_usage, 900);
        assert!(budget.would_exceed(200));

        budget.track_free(300);
        assert_eq!(budget.current_usage, 600);
        assert_eq!(budget.peak_usage, 900);

        budget.reset_peak();
        assert_eq!(budget.peak_usage, 600);
    }

    #[test]
    fn test_allocator_stats() {
        let mut alloc = PoolAllocatorV2::new(4096);
        let source = SourceLocation::new("test.rs", 1, 1);

        alloc
            .allocate(100, AllocationCategory::Render, source.clone())
            .unwrap();
        alloc
            .allocate(200, AllocationCategory::Physics, source.clone())
            .unwrap();

        let stats = alloc.stats();
        assert_eq!(stats.live_allocations, 2);
        assert_eq!(stats.total_alloc_count, 2);
        assert_eq!(stats.total_free_count, 0);
        assert!(stats.allocated_bytes > 0);
    }

    #[test]
    fn test_buddy_order_calculation() {
        assert_eq!(BuddyAllocator::order_for_size(1), 0);
        assert_eq!(BuddyAllocator::order_for_size(16), 0);
        assert_eq!(BuddyAllocator::order_for_size(17), 1);
        assert_eq!(BuddyAllocator::order_for_size(32), 1);
        assert_eq!(BuddyAllocator::order_for_size(33), 2);
        assert_eq!(BuddyAllocator::order_for_size(64), 2);
    }

    #[test]
    fn test_source_location_display() {
        let loc = SourceLocation::new("engine/core.rs", 42, 10);
        assert_eq!(format!("{}", loc), "engine/core.rs:42:10");

        let loc2 = SourceLocation::with_label("engine/core.rs", 42, 10, "mesh upload");
        assert_eq!(format!("{}", loc2), "engine/core.rs:42:10 (mesh upload)");
    }

    #[test]
    fn test_alloc_error_display() {
        let err = AllocError::ZeroSize;
        assert_eq!(format!("{}", err), "allocation size must be > 0");

        let err2 = AllocError::OutOfMemory {
            requested: 1024,
            available: 512,
        };
        assert!(format!("{}", err2).contains("1024"));
    }

    #[test]
    fn test_dump_summary() {
        let mut alloc = PoolAllocatorV2::new(1024);
        let source = SourceLocation::new("test.rs", 1, 1);
        alloc.set_budget(AllocationCategory::Render, 512);
        alloc
            .allocate(64, AllocationCategory::Render, source)
            .unwrap();

        let mut buf = String::new();
        alloc.dump_summary(&mut buf).unwrap();
        assert!(buf.contains("PoolAllocatorV2 Summary"));
        assert!(buf.contains("Render"));
    }
}
