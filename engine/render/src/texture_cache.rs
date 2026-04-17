// engine/render/src/texture_cache.rs
//
// Texture resource caching for the Genovo engine.
//
// Provides an LRU-based texture cache that manages GPU texture lifetimes,
// enables sharing between materials, and enforces memory budgets:
//
// - **LRU texture pool** -- Textures are evicted in least-recently-used order
//   when the memory budget is exceeded.
// - **Reference counting** -- Materials can share textures; textures are only
//   eligible for eviction when their reference count drops to zero.
// - **Texture sharing** -- Multiple materials referencing the same source path
//   receive the same texture handle, avoiding duplicate GPU allocations.
// - **Cache statistics** -- Hit/miss rates, memory usage, eviction counts.
// - **Memory budget enforcement** -- Configurable VRAM budget with automatic
//   eviction of the least-recently-used unreferenced textures.
// - **Eviction priorities** -- Textures can be pinned (never evicted) or given
//   priority hints that modify eviction order.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default texture cache memory budget (512 MB).
const DEFAULT_BUDGET_BYTES: u64 = 512 * 1024 * 1024;

/// Maximum number of cached textures.
const MAX_CACHED_TEXTURES: usize = 8192;

// ---------------------------------------------------------------------------
// Texture handle
// ---------------------------------------------------------------------------

/// Opaque handle identifying a cached texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CachedTextureHandle {
    index: u32,
    generation: u32,
}

impl CachedTextureHandle {
    /// Create a new handle.
    fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Returns the internal index (for debugging).
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Returns the generation (for debugging).
    pub fn generation(&self) -> u32 {
        self.generation
    }
}

impl fmt::Display for CachedTextureHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tex({}:{})", self.index, self.generation)
    }
}

// ---------------------------------------------------------------------------
// Eviction priority
// ---------------------------------------------------------------------------

/// Eviction priority for a cached texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EvictionPriority {
    /// Never evict (pinned).
    Pinned,
    /// High priority -- evict last.
    High,
    /// Normal priority -- default eviction order.
    Normal,
    /// Low priority -- evict first.
    Low,
}

impl Default for EvictionPriority {
    fn default() -> Self {
        Self::Normal
    }
}

// ---------------------------------------------------------------------------
// Texture format info
// ---------------------------------------------------------------------------

/// Compressed or uncompressed texture format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CachedTextureFormat {
    Rgba8Unorm,
    Rgba8Srgb,
    Rgba16Float,
    Rgba32Float,
    R8Unorm,
    Rg8Unorm,
    Bc1Unorm,
    Bc3Unorm,
    Bc5Unorm,
    Bc7Unorm,
    Bc7Srgb,
    Depth32Float,
    Depth24Stencil8,
}

impl CachedTextureFormat {
    /// Bytes per pixel (or approximate for block-compressed formats).
    pub fn bytes_per_pixel(&self) -> f32 {
        match self {
            Self::Rgba8Unorm | Self::Rgba8Srgb => 4.0,
            Self::Rgba16Float => 8.0,
            Self::Rgba32Float => 16.0,
            Self::R8Unorm => 1.0,
            Self::Rg8Unorm => 2.0,
            Self::Bc1Unorm => 0.5,
            Self::Bc3Unorm | Self::Bc5Unorm => 1.0,
            Self::Bc7Unorm | Self::Bc7Srgb => 1.0,
            Self::Depth32Float => 4.0,
            Self::Depth24Stencil8 => 4.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Texture descriptor
// ---------------------------------------------------------------------------

/// Describes a texture to be cached.
#[derive(Debug, Clone)]
pub struct CachedTextureDesc {
    /// Source path or identifier used for deduplication.
    pub source_key: String,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Depth (1 for 2D textures).
    pub depth: u32,
    /// Number of mip levels.
    pub mip_levels: u32,
    /// Array layers (1 for non-array textures).
    pub array_layers: u32,
    /// Pixel format.
    pub format: CachedTextureFormat,
    /// Eviction priority.
    pub priority: EvictionPriority,
    /// Whether this texture is a render target.
    pub is_render_target: bool,
}

impl CachedTextureDesc {
    /// Estimate the GPU memory usage of this texture in bytes.
    pub fn estimated_size_bytes(&self) -> u64 {
        let bpp = self.format.bytes_per_pixel();
        let mut total: u64 = 0;
        let mut w = self.width as u64;
        let mut h = self.height as u64;
        let d = self.depth.max(1) as u64;
        let layers = self.array_layers.max(1) as u64;

        for _mip in 0..self.mip_levels.max(1) {
            total += (w * h * d) as u64;
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }
        (total as f64 * bpp as f64 * layers as f64) as u64
    }
}

// ---------------------------------------------------------------------------
// Cache entry
// ---------------------------------------------------------------------------

/// Internal entry for a cached texture.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Texture descriptor.
    desc: CachedTextureDesc,
    /// Handle for external reference.
    handle: CachedTextureHandle,
    /// Reference count from materials.
    ref_count: u32,
    /// Last frame this texture was accessed.
    last_access_frame: u64,
    /// Number of times this texture has been accessed.
    access_count: u64,
    /// Whether this entry is valid (not evicted).
    valid: bool,
    /// Estimated GPU memory in bytes.
    size_bytes: u64,
}

// ---------------------------------------------------------------------------
// Cache statistics
// ---------------------------------------------------------------------------

/// Statistics about the texture cache.
#[derive(Debug, Clone, Default)]
pub struct TextureCacheStats {
    /// Total number of cached textures.
    pub texture_count: usize,
    /// Total GPU memory used by cached textures.
    pub used_bytes: u64,
    /// Memory budget.
    pub budget_bytes: u64,
    /// Percentage of budget used.
    pub budget_usage_percent: f32,
    /// Number of cache hits this frame.
    pub hits: u64,
    /// Number of cache misses this frame.
    pub misses: u64,
    /// Total cache hits since creation.
    pub total_hits: u64,
    /// Total cache misses since creation.
    pub total_misses: u64,
    /// Number of evictions this frame.
    pub evictions_this_frame: u32,
    /// Total evictions since creation.
    pub total_evictions: u64,
    /// Number of pinned (non-evictable) textures.
    pub pinned_count: usize,
    /// Memory used by pinned textures.
    pub pinned_bytes: u64,
    /// Number of shared textures (ref_count > 1).
    pub shared_count: usize,
    /// Hit rate (0.0 -- 1.0).
    pub hit_rate: f32,
}

impl fmt::Display for TextureCacheStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TextureCache: {} textures, {:.1} MB / {:.1} MB ({:.1}%), hit rate: {:.1}%",
            self.texture_count,
            self.used_bytes as f64 / (1024.0 * 1024.0),
            self.budget_bytes as f64 / (1024.0 * 1024.0),
            self.budget_usage_percent,
            self.hit_rate * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Texture cache
// ---------------------------------------------------------------------------

/// LRU texture cache with reference counting and memory budget enforcement.
pub struct TextureCache {
    /// All cache entries indexed by slot.
    entries: Vec<Option<CacheEntry>>,
    /// Map from source key to entry index for deduplication.
    key_to_index: HashMap<String, usize>,
    /// Free slot indices.
    free_slots: Vec<usize>,
    /// Generation counter for handles.
    generation: u32,
    /// Current frame number.
    current_frame: u64,
    /// Memory budget in bytes.
    budget_bytes: u64,
    /// Current memory usage.
    used_bytes: u64,
    /// Statistics.
    stats: TextureCacheStats,
    /// Frame-local hit count.
    frame_hits: u64,
    /// Frame-local miss count.
    frame_misses: u64,
    /// Frame-local eviction count.
    frame_evictions: u32,
}

impl TextureCache {
    /// Create a new texture cache with the default memory budget.
    pub fn new() -> Self {
        Self::with_budget(DEFAULT_BUDGET_BYTES)
    }

    /// Create a new texture cache with a specific memory budget (in bytes).
    pub fn with_budget(budget_bytes: u64) -> Self {
        Self {
            entries: Vec::with_capacity(256),
            key_to_index: HashMap::with_capacity(256),
            free_slots: Vec::new(),
            generation: 0,
            current_frame: 0,
            budget_bytes,
            used_bytes: 0,
            stats: TextureCacheStats {
                budget_bytes,
                ..Default::default()
            },
            frame_hits: 0,
            frame_misses: 0,
            frame_evictions: 0,
        }
    }

    /// Set the memory budget.
    pub fn set_budget(&mut self, budget_bytes: u64) {
        self.budget_bytes = budget_bytes;
        self.stats.budget_bytes = budget_bytes;
        // Evict if over budget.
        self.enforce_budget();
    }

    /// Get the memory budget.
    pub fn budget(&self) -> u64 {
        self.budget_bytes
    }

    /// Get the current memory usage.
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    /// Begin a new frame. Resets per-frame statistics.
    pub fn begin_frame(&mut self, frame: u64) {
        self.current_frame = frame;
        self.frame_hits = 0;
        self.frame_misses = 0;
        self.frame_evictions = 0;
    }

    /// End the frame and update statistics.
    pub fn end_frame(&mut self) {
        self.stats.hits = self.frame_hits;
        self.stats.misses = self.frame_misses;
        self.stats.total_hits += self.frame_hits;
        self.stats.total_misses += self.frame_misses;
        self.stats.evictions_this_frame = self.frame_evictions;
        self.stats.texture_count = self.key_to_index.len();
        self.stats.used_bytes = self.used_bytes;
        self.stats.budget_usage_percent = if self.budget_bytes > 0 {
            (self.used_bytes as f64 / self.budget_bytes as f64 * 100.0) as f32
        } else {
            0.0
        };

        let total_accesses = self.stats.total_hits + self.stats.total_misses;
        self.stats.hit_rate = if total_accesses > 0 {
            self.stats.total_hits as f32 / total_accesses as f32
        } else {
            0.0
        };

        // Count pinned and shared.
        self.stats.pinned_count = 0;
        self.stats.pinned_bytes = 0;
        self.stats.shared_count = 0;
        for entry_opt in &self.entries {
            if let Some(entry) = entry_opt {
                if entry.valid {
                    if entry.desc.priority == EvictionPriority::Pinned {
                        self.stats.pinned_count += 1;
                        self.stats.pinned_bytes += entry.size_bytes;
                    }
                    if entry.ref_count > 1 {
                        self.stats.shared_count += 1;
                    }
                }
            }
        }
    }

    /// Get the current statistics.
    pub fn stats(&self) -> &TextureCacheStats {
        &self.stats
    }

    /// Look up a texture by source key. Returns the handle if found.
    pub fn lookup(&mut self, source_key: &str) -> Option<CachedTextureHandle> {
        if let Some(&index) = self.key_to_index.get(source_key) {
            if let Some(entry) = &mut self.entries[index] {
                if entry.valid {
                    entry.last_access_frame = self.current_frame;
                    entry.access_count += 1;
                    self.frame_hits += 1;
                    return Some(entry.handle);
                }
            }
        }
        self.frame_misses += 1;
        None
    }

    /// Insert a new texture into the cache.
    ///
    /// If a texture with the same source key already exists, increments its
    /// reference count and returns the existing handle.
    pub fn insert(&mut self, desc: CachedTextureDesc) -> CachedTextureHandle {
        // Check for existing entry.
        if let Some(&index) = self.key_to_index.get(&desc.source_key) {
            if let Some(entry) = &mut self.entries[index] {
                if entry.valid {
                    entry.ref_count += 1;
                    entry.last_access_frame = self.current_frame;
                    entry.access_count += 1;
                    self.frame_hits += 1;
                    return entry.handle;
                }
            }
        }

        // Enforce budget before inserting.
        let size = desc.estimated_size_bytes();
        while self.used_bytes + size > self.budget_bytes {
            if !self.evict_one() {
                break; // Nothing left to evict.
            }
        }

        // Allocate a slot.
        let index = if let Some(free) = self.free_slots.pop() {
            free
        } else {
            if self.entries.len() >= MAX_CACHED_TEXTURES {
                // Force evict the LRU to make room.
                self.evict_one();
                if let Some(free) = self.free_slots.pop() {
                    free
                } else {
                    // Expand the pool.
                    let idx = self.entries.len();
                    self.entries.push(None);
                    idx
                }
            } else {
                let idx = self.entries.len();
                self.entries.push(None);
                idx
            }
        };

        self.generation += 1;
        let handle = CachedTextureHandle::new(index as u32, self.generation);

        let source_key = desc.source_key.clone();

        let entry = CacheEntry {
            desc,
            handle,
            ref_count: 1,
            last_access_frame: self.current_frame,
            access_count: 1,
            valid: true,
            size_bytes: size,
        };

        self.key_to_index.insert(source_key, index);
        if index < self.entries.len() {
            self.entries[index] = Some(entry);
        }
        self.used_bytes += size;
        self.frame_misses += 1;

        handle
    }

    /// Add a reference to a cached texture.
    pub fn add_ref(&mut self, handle: CachedTextureHandle) -> bool {
        let index = handle.index as usize;
        if let Some(Some(entry)) = self.entries.get_mut(index) {
            if entry.valid && entry.handle.generation == handle.generation {
                entry.ref_count += 1;
                return true;
            }
        }
        false
    }

    /// Release a reference to a cached texture.
    ///
    /// The texture is not immediately evicted when ref_count reaches zero;
    /// it remains in the cache and can be reused until evicted by budget pressure.
    pub fn release(&mut self, handle: CachedTextureHandle) -> bool {
        let index = handle.index as usize;
        if let Some(Some(entry)) = self.entries.get_mut(index) {
            if entry.valid && entry.handle.generation == handle.generation {
                if entry.ref_count > 0 {
                    entry.ref_count -= 1;
                }
                return true;
            }
        }
        false
    }

    /// Get the reference count for a texture.
    pub fn ref_count(&self, handle: CachedTextureHandle) -> Option<u32> {
        let index = handle.index as usize;
        if let Some(Some(entry)) = self.entries.get(index) {
            if entry.valid && entry.handle.generation == handle.generation {
                return Some(entry.ref_count);
            }
        }
        None
    }

    /// Pin a texture so it is never evicted.
    pub fn pin(&mut self, handle: CachedTextureHandle) -> bool {
        let index = handle.index as usize;
        if let Some(Some(entry)) = self.entries.get_mut(index) {
            if entry.valid && entry.handle.generation == handle.generation {
                entry.desc.priority = EvictionPriority::Pinned;
                return true;
            }
        }
        false
    }

    /// Unpin a texture so it can be evicted normally.
    pub fn unpin(&mut self, handle: CachedTextureHandle) -> bool {
        let index = handle.index as usize;
        if let Some(Some(entry)) = self.entries.get_mut(index) {
            if entry.valid && entry.handle.generation == handle.generation {
                entry.desc.priority = EvictionPriority::Normal;
                return true;
            }
        }
        false
    }

    /// Set the eviction priority for a texture.
    pub fn set_priority(&mut self, handle: CachedTextureHandle, priority: EvictionPriority) -> bool {
        let index = handle.index as usize;
        if let Some(Some(entry)) = self.entries.get_mut(index) {
            if entry.valid && entry.handle.generation == handle.generation {
                entry.desc.priority = priority;
                return true;
            }
        }
        false
    }

    /// Check if a handle is still valid.
    pub fn is_valid(&self, handle: CachedTextureHandle) -> bool {
        let index = handle.index as usize;
        if let Some(Some(entry)) = self.entries.get(index) {
            entry.valid && entry.handle.generation == handle.generation
        } else {
            false
        }
    }

    /// Get the texture descriptor for a handle.
    pub fn descriptor(&self, handle: CachedTextureHandle) -> Option<&CachedTextureDesc> {
        let index = handle.index as usize;
        if let Some(Some(entry)) = self.entries.get(index) {
            if entry.valid && entry.handle.generation == handle.generation {
                return Some(&entry.desc);
            }
        }
        None
    }

    /// Evict a specific texture by handle (even if it has references).
    pub fn force_evict(&mut self, handle: CachedTextureHandle) -> bool {
        let index = handle.index as usize;
        if let Some(Some(entry)) = self.entries.get_mut(index) {
            if entry.valid && entry.handle.generation == handle.generation {
                self.used_bytes = self.used_bytes.saturating_sub(entry.size_bytes);
                self.key_to_index.remove(&entry.desc.source_key);
                entry.valid = false;
                self.free_slots.push(index);
                self.frame_evictions += 1;
                self.stats.total_evictions += 1;
                return true;
            }
        }
        false
    }

    /// Evict all unreferenced textures.
    pub fn evict_unreferenced(&mut self) -> u32 {
        let mut count = 0;
        let mut indices_to_evict = Vec::new();

        for (i, entry_opt) in self.entries.iter().enumerate() {
            if let Some(entry) = entry_opt {
                if entry.valid
                    && entry.ref_count == 0
                    && entry.desc.priority != EvictionPriority::Pinned
                {
                    indices_to_evict.push(i);
                }
            }
        }

        for index in indices_to_evict {
            if let Some(entry) = &mut self.entries[index] {
                if entry.valid {
                    self.used_bytes = self.used_bytes.saturating_sub(entry.size_bytes);
                    self.key_to_index.remove(&entry.desc.source_key);
                    entry.valid = false;
                    self.free_slots.push(index);
                    count += 1;
                    self.stats.total_evictions += 1;
                }
            }
        }
        self.frame_evictions += count;
        count
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        let count = self.entries.len();
        for i in 0..count {
            if let Some(entry) = &self.entries[i] {
                if entry.valid {
                    self.key_to_index.remove(&entry.desc.source_key);
                    self.stats.total_evictions += 1;
                }
            }
            self.entries[i] = None;
        }
        self.free_slots.clear();
        for i in (0..count).rev() {
            self.free_slots.push(i);
        }
        self.used_bytes = 0;
    }

    /// Enforce the memory budget by evicting LRU unreferenced textures.
    fn enforce_budget(&mut self) {
        while self.used_bytes > self.budget_bytes {
            if !self.evict_one() {
                break;
            }
        }
    }

    /// Evict the single best candidate texture.
    ///
    /// Prefers: unreferenced, lowest priority, oldest access frame.
    fn evict_one(&mut self) -> bool {
        let mut best_index: Option<usize> = None;
        let mut best_priority = EvictionPriority::Pinned;
        let mut best_frame = u64::MAX;

        for (i, entry_opt) in self.entries.iter().enumerate() {
            if let Some(entry) = entry_opt {
                if !entry.valid || entry.desc.priority == EvictionPriority::Pinned {
                    continue;
                }
                // Prefer unreferenced textures.
                if entry.ref_count > 0 {
                    continue;
                }
                // Among unreferenced, prefer lowest priority and oldest access.
                let dominated = match best_index {
                    None => true,
                    Some(_) => {
                        entry.desc.priority > best_priority
                            || (entry.desc.priority == best_priority
                                && entry.last_access_frame < best_frame)
                    }
                };
                if dominated {
                    best_index = Some(i);
                    best_priority = entry.desc.priority;
                    best_frame = entry.last_access_frame;
                }
            }
        }

        if let Some(index) = best_index {
            if let Some(entry) = &mut self.entries[index] {
                self.used_bytes = self.used_bytes.saturating_sub(entry.size_bytes);
                self.key_to_index.remove(&entry.desc.source_key);
                entry.valid = false;
                self.free_slots.push(index);
                self.frame_evictions += 1;
                self.stats.total_evictions += 1;
                return true;
            }
        }
        false
    }

    /// Get all source keys currently in the cache.
    pub fn cached_keys(&self) -> Vec<&str> {
        self.key_to_index.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of cached textures.
    pub fn count(&self) -> usize {
        self.key_to_index.len()
    }

    /// Get the number of free slots.
    pub fn free_slot_count(&self) -> usize {
        self.free_slots.len()
    }
}

impl Default for TextureCache {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for TextureCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TextureCache")
            .field("count", &self.count())
            .field("used_mb", &(self.used_bytes as f64 / (1024.0 * 1024.0)))
            .field("budget_mb", &(self.budget_bytes as f64 / (1024.0 * 1024.0)))
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_desc(key: &str, width: u32, height: u32) -> CachedTextureDesc {
        CachedTextureDesc {
            source_key: key.to_string(),
            width,
            height,
            depth: 1,
            mip_levels: 1,
            array_layers: 1,
            format: CachedTextureFormat::Rgba8Unorm,
            priority: EvictionPriority::Normal,
            is_render_target: false,
        }
    }

    #[test]
    fn test_insert_and_lookup() {
        let mut cache = TextureCache::new();
        let handle = cache.insert(make_desc("test.png", 256, 256));
        assert!(cache.is_valid(handle));
        assert_eq!(cache.ref_count(handle), Some(1));

        let found = cache.lookup("test.png");
        assert_eq!(found, Some(handle));
    }

    #[test]
    fn test_duplicate_insert_shares() {
        let mut cache = TextureCache::new();
        let h1 = cache.insert(make_desc("test.png", 256, 256));
        let h2 = cache.insert(make_desc("test.png", 256, 256));
        assert_eq!(h1, h2);
        assert_eq!(cache.ref_count(h1), Some(2));
    }

    #[test]
    fn test_release_and_evict() {
        let mut cache = TextureCache::with_budget(1024 * 1024);
        let h = cache.insert(make_desc("a.png", 256, 256));
        cache.release(h);
        assert_eq!(cache.ref_count(h), Some(0));
        assert_eq!(cache.evict_unreferenced(), 1);
        assert!(!cache.is_valid(h));
    }

    #[test]
    fn test_budget_enforcement() {
        // Budget of 256*256*4 bytes = one texture.
        let budget = 256 * 256 * 4;
        let mut cache = TextureCache::with_budget(budget);
        let h1 = cache.insert(make_desc("a.png", 256, 256));
        cache.release(h1);
        let _h2 = cache.insert(make_desc("b.png", 256, 256));
        // h1 should have been evicted.
        assert!(!cache.is_valid(h1));
    }

    #[test]
    fn test_pin() {
        let budget = 256 * 256 * 4;
        let mut cache = TextureCache::with_budget(budget);
        let h1 = cache.insert(make_desc("a.png", 256, 256));
        cache.release(h1);
        cache.pin(h1);
        // Try to insert another; h1 should NOT be evicted because it's pinned.
        let _h2 = cache.insert(make_desc("b.png", 256, 256));
        assert!(cache.is_valid(h1));
    }
}
