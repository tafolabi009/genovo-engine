// engine/render/src/texture_streaming.rs
//
// Texture streaming system for the Genovo engine. Manages GPU memory by
// streaming texture mip levels on demand based on screen-space coverage and
// camera distance. This avoids loading every texture at full resolution,
// dramatically reducing GPU memory usage in large open-world scenes.
//
// Architecture:
//   - `StreamableTexture`: a texture resource with metadata about its full
//     resolution, current loaded mip level, and priority.
//   - `TextureStreamingManager`: the central manager that orchestrates which
//     mip levels to load or evict each frame.
//   - `TexturePool`: GPU-side memory management with LRU eviction under a
//     configurable memory budget.
//   - Priority computation based on screen-space coverage, distance, and
//     visibility feedback from the renderer.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TextureId
// ---------------------------------------------------------------------------

/// Unique identifier for a streamable texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamTextureId(pub u64);

impl StreamTextureId {
    pub const INVALID: Self = Self(u64::MAX);

    pub fn is_valid(self) -> bool {
        self.0 != u64::MAX
    }
}

impl Default for StreamTextureId {
    fn default() -> Self {
        Self::INVALID
    }
}

// ---------------------------------------------------------------------------
// MipLevel
// ---------------------------------------------------------------------------

/// Describes the state of a single mip level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MipState {
    /// The mip level has not been loaded at all.
    NotLoaded,
    /// A load request is in flight.
    Loading,
    /// The mip level data is resident in GPU memory.
    Resident,
    /// The mip level was evicted from GPU memory but the data is still in
    /// the CPU-side cache.
    Cached,
    /// An error occurred loading this mip level.
    Error,
}

/// Per-mip metadata for a streamable texture.
#[derive(Debug, Clone)]
pub struct MipInfo {
    /// Mip level index (0 = highest resolution).
    pub level: u32,
    /// Width of this mip level in pixels.
    pub width: u32,
    /// Height of this mip level in pixels.
    pub height: u32,
    /// Uncompressed size in bytes.
    pub size_bytes: u64,
    /// Current state of this mip level.
    pub state: MipState,
    /// Frame number when this mip was last accessed.
    pub last_access_frame: u64,
}

impl MipInfo {
    pub fn new(level: u32, width: u32, height: u32) -> Self {
        let size = (width as u64) * (height as u64) * 4; // Assume RGBA8
        Self {
            level,
            width,
            height,
            size_bytes: size,
            state: MipState::NotLoaded,
            last_access_frame: 0,
        }
    }

    /// Whether this mip is available for rendering.
    pub fn is_available(&self) -> bool {
        self.state == MipState::Resident
    }
}

// ---------------------------------------------------------------------------
// StreamableTexture
// ---------------------------------------------------------------------------

/// A texture that supports mip-level streaming.
#[derive(Debug, Clone)]
pub struct StreamableTexture {
    /// Unique identifier.
    pub id: StreamTextureId,
    /// Human-readable name (for debugging).
    pub name: String,
    /// Path to the full-resolution texture on disk.
    pub asset_path: String,
    /// Full-resolution width.
    pub full_width: u32,
    /// Full-resolution height.
    pub full_height: u32,
    /// Total number of mip levels.
    pub mip_count: u32,
    /// Per-mip metadata.
    pub mips: Vec<MipInfo>,
    /// The finest (lowest-numbered) mip level currently resident on the GPU.
    pub current_resident_mip: u32,
    /// The mip level the system wants to have resident (target).
    pub target_mip: u32,
    /// Priority score (higher = more important to stream in).
    pub priority: f32,
    /// Screen-space coverage in pixels squared (from feedback).
    pub screen_coverage: f32,
    /// Minimum distance from the camera to any object using this texture.
    pub min_camera_distance: f32,
    /// Frame number when this texture was last referenced in a draw call.
    pub last_used_frame: u64,
    /// Whether this texture is currently visible (from the renderer's
    /// visibility feedback).
    pub visible: bool,
    /// Whether this texture is pinned (never evicted, always fully loaded).
    pub pinned: bool,
    /// Total bytes currently resident for this texture across all loaded mips.
    pub resident_bytes: u64,
    /// GPU texture handle (opaque).
    pub gpu_handle: Option<u64>,
    /// Texture format.
    pub format: TextureStreamFormat,
    /// Whether the texture uses sRGB encoding.
    pub srgb: bool,
}

/// Texture format for streaming purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureStreamFormat {
    RGBA8,
    RGBA16F,
    BC1,
    BC3,
    BC4,
    BC5,
    BC7,
    ASTC4x4,
    ASTC8x8,
}

impl TextureStreamFormat {
    /// Bytes per pixel (or per block, normalised to per-pixel).
    pub fn bytes_per_pixel(&self) -> f32 {
        match self {
            Self::RGBA8 => 4.0,
            Self::RGBA16F => 8.0,
            Self::BC1 => 0.5,
            Self::BC3 | Self::BC5 | Self::BC7 => 1.0,
            Self::BC4 => 0.5,
            Self::ASTC4x4 => 1.0,
            Self::ASTC8x8 => 0.25,
        }
    }

    /// Calculate the size in bytes for a given width and height.
    pub fn mip_size_bytes(&self, width: u32, height: u32) -> u64 {
        let pixels = (width as u64) * (height as u64);
        (pixels as f64 * self.bytes_per_pixel() as f64).ceil() as u64
    }
}

impl StreamableTexture {
    /// Create a new streamable texture descriptor.
    pub fn new(
        id: StreamTextureId,
        name: impl Into<String>,
        asset_path: impl Into<String>,
        width: u32,
        height: u32,
        format: TextureStreamFormat,
    ) -> Self {
        let mip_count = compute_mip_count(width, height);
        let mut mips = Vec::with_capacity(mip_count as usize);
        for level in 0..mip_count {
            let mip_w = (width >> level).max(1);
            let mip_h = (height >> level).max(1);
            let mut info = MipInfo::new(level, mip_w, mip_h);
            info.size_bytes = format.mip_size_bytes(mip_w, mip_h);
            mips.push(info);
        }

        Self {
            id,
            name: name.into(),
            asset_path: asset_path.into(),
            full_width: width,
            full_height: height,
            mip_count,
            mips,
            current_resident_mip: mip_count.saturating_sub(1), // Start with lowest mip
            target_mip: mip_count.saturating_sub(1),
            priority: 0.0,
            screen_coverage: 0.0,
            min_camera_distance: f32::MAX,
            last_used_frame: 0,
            visible: false,
            pinned: false,
            resident_bytes: 0,
            gpu_handle: None,
            format,
            srgb: false,
        }
    }

    /// Compute the total size in bytes if all mip levels were resident.
    pub fn total_size_bytes(&self) -> u64 {
        self.mips.iter().map(|m| m.size_bytes).sum()
    }

    /// Compute the size of mip levels that need to be loaded to reach the
    /// target mip.
    pub fn bytes_needed_for_target(&self) -> u64 {
        if self.target_mip >= self.current_resident_mip {
            return 0;
        }
        self.mips[self.target_mip as usize..self.current_resident_mip as usize]
            .iter()
            .map(|m| m.size_bytes)
            .sum()
    }

    /// Compute the bytes that would be freed by evicting mip levels down to
    /// `new_mip`.
    pub fn bytes_freed_by_eviction(&self, new_mip: u32) -> u64 {
        if new_mip <= self.current_resident_mip {
            return 0;
        }
        let start = self.current_resident_mip as usize;
        let end = new_mip.min(self.mip_count) as usize;
        self.mips[start..end].iter().map(|m| m.size_bytes).sum()
    }

    /// Mark a mip level as resident (loaded).
    pub fn mark_mip_resident(&mut self, level: u32, frame: u64) {
        if let Some(mip) = self.mips.get_mut(level as usize) {
            mip.state = MipState::Resident;
            mip.last_access_frame = frame;
        }
        self.recompute_resident_state();
    }

    /// Mark a mip level as evicted.
    pub fn mark_mip_evicted(&mut self, level: u32) {
        if let Some(mip) = self.mips.get_mut(level as usize) {
            mip.state = MipState::Cached;
        }
        self.recompute_resident_state();
    }

    /// Mark a mip level as loading.
    pub fn mark_mip_loading(&mut self, level: u32) {
        if let Some(mip) = self.mips.get_mut(level as usize) {
            mip.state = MipState::Loading;
        }
    }

    /// Recompute `current_resident_mip` and `resident_bytes` from the mip
    /// state array.
    fn recompute_resident_state(&mut self) {
        self.resident_bytes = 0;
        self.current_resident_mip = self.mip_count.saturating_sub(1);

        for mip in &self.mips {
            if mip.state == MipState::Resident {
                self.resident_bytes += mip.size_bytes;
                if mip.level < self.current_resident_mip {
                    self.current_resident_mip = mip.level;
                }
            }
        }
    }

    /// Check if the texture has reached its target mip level.
    pub fn is_at_target(&self) -> bool {
        self.current_resident_mip <= self.target_mip
    }

    /// Check if any mip levels are currently loading.
    pub fn is_loading(&self) -> bool {
        self.mips.iter().any(|m| m.state == MipState::Loading)
    }
}

/// Compute the number of mip levels for a given texture dimension.
pub fn compute_mip_count(width: u32, height: u32) -> u32 {
    let max_dim = width.max(height);
    if max_dim == 0 {
        return 1;
    }
    (max_dim as f32).log2().floor() as u32 + 1
}

// ---------------------------------------------------------------------------
// Priority computation
// ---------------------------------------------------------------------------

/// Parameters controlling how texture streaming priority is computed.
#[derive(Debug, Clone)]
pub struct PriorityParams {
    /// Weight for screen-space coverage in the priority formula.
    pub coverage_weight: f32,
    /// Weight for inverse camera distance.
    pub distance_weight: f32,
    /// Weight for recency of use (frames since last reference).
    pub recency_weight: f32,
    /// Bonus priority for textures that are currently visible.
    pub visibility_bonus: f32,
    /// Maximum distance at which a texture receives any priority.
    pub max_distance: f32,
    /// Exponent applied to coverage for non-linear priority scaling.
    pub coverage_exponent: f32,
}

impl Default for PriorityParams {
    fn default() -> Self {
        Self {
            coverage_weight: 1.0,
            distance_weight: 0.5,
            recency_weight: 0.2,
            visibility_bonus: 2.0,
            max_distance: 1000.0,
            coverage_exponent: 0.5,
        }
    }
}

/// Compute the streaming priority for a texture.
///
/// Higher values mean the texture should be streamed in more urgently.
/// A priority of 0 means the texture is a candidate for eviction.
pub fn compute_priority(
    texture: &StreamableTexture,
    current_frame: u64,
    params: &PriorityParams,
) -> f32 {
    if texture.pinned {
        return f32::MAX; // Pinned textures always have maximum priority.
    }

    if !texture.visible && texture.last_used_frame + 60 < current_frame {
        // Not visible and not recently used -- low priority.
        return 0.0;
    }

    // Screen-space coverage component: larger on-screen footprint = higher
    // priority. We use a sqrt-like curve so that very large textures don't
    // dominate excessively.
    let coverage_score = texture.screen_coverage.max(0.0).powf(params.coverage_exponent);
    let coverage_component = coverage_score * params.coverage_weight;

    // Distance component: closer textures get higher priority. Normalised
    // to [0, 1] using max_distance.
    let dist_norm = (1.0 - (texture.min_camera_distance / params.max_distance).min(1.0)).max(0.0);
    let distance_component = dist_norm * params.distance_weight;

    // Recency component: recently used textures get a boost.
    let frames_since_use = (current_frame - texture.last_used_frame) as f32;
    let recency_score = 1.0 / (1.0 + frames_since_use * 0.01);
    let recency_component = recency_score * params.recency_weight;

    // Visibility bonus.
    let vis_bonus = if texture.visible { params.visibility_bonus } else { 0.0 };

    coverage_component + distance_component + recency_component + vis_bonus
}

/// Compute the ideal mip level for a texture based on its screen-space
/// coverage. This determines the `target_mip` that the streaming system
/// tries to achieve.
///
/// The formula computes which mip level would result in approximately 1:1
/// texel-to-pixel mapping for the texture's screen coverage.
///
/// # Arguments
/// - `full_width`, `full_height`: texture's full (mip 0) dimensions.
/// - `screen_coverage_pixels`: screen-space area occupied by the textured
///   geometry, in pixels squared.
/// - `mip_count`: total number of mip levels.
/// - `mip_bias`: artist-controllable bias (negative = sharper, positive = blurrier).
pub fn compute_target_mip(
    full_width: u32,
    full_height: u32,
    screen_coverage_pixels: f32,
    mip_count: u32,
    mip_bias: f32,
) -> u32 {
    if screen_coverage_pixels <= 0.0 || mip_count == 0 {
        return mip_count.saturating_sub(1);
    }

    // The texture covers `screen_coverage_pixels` on screen. For 1:1 mapping,
    // we want the mip level whose dimensions squared roughly match this area.
    let texture_area = (full_width as f32) * (full_height as f32);
    if texture_area <= 0.0 {
        return mip_count.saturating_sub(1);
    }

    // Ratio of texture pixels to screen pixels.
    let texel_ratio = texture_area / screen_coverage_pixels;

    // Each mip level halves each dimension, so the area ratio per mip is 4x.
    // mip_level = log2(sqrt(texel_ratio)) = 0.5 * log2(texel_ratio)
    let ideal_mip = 0.5 * texel_ratio.max(1.0).log2() + mip_bias;

    // Clamp to valid range.
    let clamped = ideal_mip.round().max(0.0).min((mip_count - 1) as f32);
    clamped as u32
}

// ---------------------------------------------------------------------------
// TexturePool
// ---------------------------------------------------------------------------

/// An entry in the texture pool's LRU tracking.
#[derive(Debug, Clone)]
struct PoolEntry {
    texture_id: StreamTextureId,
    /// Total bytes this texture occupies in the pool.
    size_bytes: u64,
    /// Frame when last accessed.
    last_access_frame: u64,
    /// Priority at time of last update.
    priority: f32,
}

/// GPU texture memory pool with LRU eviction.
///
/// The pool tracks total memory usage and enforces a budget. When the budget
/// is exceeded, it evicts the least-recently-used, lowest-priority textures
/// until usage falls below the threshold.
pub struct TexturePool {
    /// Maximum GPU memory budget in bytes.
    pub budget_bytes: u64,
    /// Current total memory usage in bytes.
    pub used_bytes: u64,
    /// Per-texture entries.
    entries: HashMap<StreamTextureId, PoolEntry>,
    /// High-water mark threshold (fraction of budget at which eviction begins).
    pub eviction_threshold: f32,
    /// Low-water mark threshold (fraction of budget to evict down to).
    pub eviction_target: f32,
    /// Number of textures evicted in the last eviction pass.
    pub last_eviction_count: u32,
    /// Total bytes evicted in the last eviction pass.
    pub last_eviction_bytes: u64,
}

impl TexturePool {
    /// Create a new pool with the given memory budget.
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            used_bytes: 0,
            entries: HashMap::new(),
            eviction_threshold: 0.9,
            eviction_target: 0.7,
            last_eviction_count: 0,
            last_eviction_bytes: 0,
        }
    }

    /// Register a texture in the pool or update its size.
    pub fn register(&mut self, id: StreamTextureId, size_bytes: u64, frame: u64, priority: f32) {
        if let Some(entry) = self.entries.get_mut(&id) {
            self.used_bytes -= entry.size_bytes;
            entry.size_bytes = size_bytes;
            entry.last_access_frame = frame;
            entry.priority = priority;
            self.used_bytes += size_bytes;
        } else {
            self.entries.insert(id, PoolEntry {
                texture_id: id,
                size_bytes,
                last_access_frame: frame,
                priority,
            });
            self.used_bytes += size_bytes;
        }
    }

    /// Remove a texture from the pool entirely.
    pub fn remove(&mut self, id: StreamTextureId) {
        if let Some(entry) = self.entries.remove(&id) {
            self.used_bytes = self.used_bytes.saturating_sub(entry.size_bytes);
        }
    }

    /// Update the access frame and priority for a texture.
    pub fn touch(&mut self, id: StreamTextureId, frame: u64, priority: f32) {
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.last_access_frame = frame;
            entry.priority = priority;
        }
    }

    /// Check if the pool is over budget.
    pub fn is_over_budget(&self) -> bool {
        self.used_bytes > self.budget_bytes
    }

    /// Check if the pool has reached the eviction threshold.
    pub fn needs_eviction(&self) -> bool {
        let threshold = (self.budget_bytes as f64 * self.eviction_threshold as f64) as u64;
        self.used_bytes > threshold
    }

    /// How much memory needs to be freed to reach the eviction target.
    pub fn bytes_to_free(&self) -> u64 {
        let target = (self.budget_bytes as f64 * self.eviction_target as f64) as u64;
        self.used_bytes.saturating_sub(target)
    }

    /// Determine which textures should be evicted to bring usage under budget.
    ///
    /// Returns a list of `(texture_id, bytes_to_free)` pairs sorted by
    /// eviction priority (lowest priority first). The caller is responsible
    /// for actually evicting the mip levels and calling `remove()` or
    /// updating the entry.
    pub fn compute_eviction_candidates(&self, current_frame: u64) -> Vec<StreamTextureId> {
        if !self.needs_eviction() {
            return Vec::new();
        }

        let bytes_needed = self.bytes_to_free();
        let mut candidates: Vec<&PoolEntry> = self.entries.values().collect();

        // Sort by eviction priority: lower priority first, then older access.
        candidates.sort_by(|a, b| {
            let priority_cmp = a.priority.partial_cmp(&b.priority).unwrap_or(std::cmp::Ordering::Equal);
            if priority_cmp != std::cmp::Ordering::Equal {
                return priority_cmp;
            }
            // Older access = evict first.
            a.last_access_frame.cmp(&b.last_access_frame)
        });

        let mut freed = 0u64;
        let mut evict_list = Vec::new();
        for entry in candidates {
            if freed >= bytes_needed {
                break;
            }
            // Don't evict textures accessed this frame.
            if entry.last_access_frame >= current_frame {
                continue;
            }
            evict_list.push(entry.texture_id);
            freed += entry.size_bytes;
        }

        evict_list
    }

    /// Memory utilisation as a fraction of the budget.
    pub fn utilisation(&self) -> f32 {
        if self.budget_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f32 / self.budget_bytes as f32
    }

    /// Number of textures in the pool.
    pub fn texture_count(&self) -> usize {
        self.entries.len()
    }

    /// Available memory (budget minus used).
    pub fn available_bytes(&self) -> u64 {
        self.budget_bytes.saturating_sub(self.used_bytes)
    }
}

// ---------------------------------------------------------------------------
// MipLoadRequest
// ---------------------------------------------------------------------------

/// A request to load a specific mip level from the asset system.
#[derive(Debug, Clone)]
pub struct MipLoadRequest {
    /// The texture this request is for.
    pub texture_id: StreamTextureId,
    /// Which mip level to load.
    pub mip_level: u32,
    /// Priority of this request (higher = load sooner).
    pub priority: f32,
    /// Size of the mip data in bytes.
    pub size_bytes: u64,
    /// Frame when this request was issued.
    pub request_frame: u64,
}

/// A completed mip load (returned by the background loader).
#[derive(Debug, Clone)]
pub struct MipLoadResult {
    /// The texture this result is for.
    pub texture_id: StreamTextureId,
    /// Which mip level was loaded.
    pub mip_level: u32,
    /// The loaded pixel data (empty on error).
    pub data: Vec<u8>,
    /// Whether the load succeeded.
    pub success: bool,
    /// Error message if `success` is false.
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// StreamingStats
// ---------------------------------------------------------------------------

/// Statistics about the texture streaming system.
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    /// Total number of managed textures.
    pub total_textures: u32,
    /// Number of textures currently visible.
    pub visible_textures: u32,
    /// Number of textures at their target mip level.
    pub at_target_count: u32,
    /// Number of textures that need higher-res mips.
    pub needs_streaming_count: u32,
    /// Total resident GPU memory in bytes.
    pub resident_bytes: u64,
    /// Total memory that would be needed for all textures at full res.
    pub total_potential_bytes: u64,
    /// Number of pending load requests.
    pub pending_loads: u32,
    /// Number of mips loaded this frame.
    pub mips_loaded_this_frame: u32,
    /// Number of mips evicted this frame.
    pub mips_evicted_this_frame: u32,
    /// Pool utilisation (0..1).
    pub pool_utilisation: f32,
    /// Pool budget in bytes.
    pub pool_budget_bytes: u64,
}

// ---------------------------------------------------------------------------
// TextureStreamingManager
// ---------------------------------------------------------------------------

/// Central texture streaming manager. Call `update()` each frame after
/// visibility feedback has been provided.
pub struct TextureStreamingManager {
    /// All managed textures.
    textures: HashMap<StreamTextureId, StreamableTexture>,
    /// GPU memory pool.
    pool: TexturePool,
    /// Pending load requests (not yet started).
    pending_requests: Vec<MipLoadRequest>,
    /// In-flight load requests (submitted to the background loader).
    inflight_requests: Vec<MipLoadRequest>,
    /// Maximum number of concurrent mip load requests.
    pub max_inflight_requests: u32,
    /// Maximum bytes to request per frame.
    pub max_bytes_per_frame: u64,
    /// Mip bias applied to all target mip calculations.
    pub global_mip_bias: f32,
    /// Priority computation parameters.
    pub priority_params: PriorityParams,
    /// Current frame number.
    current_frame: u64,
    /// Next texture id.
    next_id: u64,
    /// Statistics.
    stats: StreamingStats,
}

impl TextureStreamingManager {
    /// Create a new streaming manager with the given GPU memory budget.
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            textures: HashMap::new(),
            pool: TexturePool::new(budget_bytes),
            pending_requests: Vec::new(),
            inflight_requests: Vec::new(),
            max_inflight_requests: 8,
            max_bytes_per_frame: 16 * 1024 * 1024, // 16 MB per frame
            global_mip_bias: 0.0,
            priority_params: PriorityParams::default(),
            current_frame: 0,
            next_id: 1,
            stats: StreamingStats::default(),
        }
    }

    /// Register a new texture for streaming. Returns its id.
    pub fn register_texture(
        &mut self,
        name: impl Into<String>,
        asset_path: impl Into<String>,
        width: u32,
        height: u32,
        format: TextureStreamFormat,
    ) -> StreamTextureId {
        let id = StreamTextureId(self.next_id);
        self.next_id += 1;

        let texture = StreamableTexture::new(id, name, asset_path, width, height, format);
        self.textures.insert(id, texture);
        id
    }

    /// Unregister a texture, freeing its pool allocation.
    pub fn unregister_texture(&mut self, id: StreamTextureId) {
        self.textures.remove(&id);
        self.pool.remove(id);
        self.pending_requests.retain(|r| r.texture_id != id);
        self.inflight_requests.retain(|r| r.texture_id != id);
    }

    /// Get a reference to a managed texture.
    pub fn get_texture(&self, id: StreamTextureId) -> Option<&StreamableTexture> {
        self.textures.get(&id)
    }

    /// Get a mutable reference to a managed texture.
    pub fn get_texture_mut(&mut self, id: StreamTextureId) -> Option<&mut StreamableTexture> {
        self.textures.get_mut(&id)
    }

    /// Pin a texture so it is never evicted and always fully loaded.
    pub fn pin_texture(&mut self, id: StreamTextureId) {
        if let Some(tex) = self.textures.get_mut(&id) {
            tex.pinned = true;
            tex.target_mip = 0;
        }
    }

    /// Unpin a texture.
    pub fn unpin_texture(&mut self, id: StreamTextureId) {
        if let Some(tex) = self.textures.get_mut(&id) {
            tex.pinned = false;
        }
    }

    // -- Visibility feedback -------------------------------------------------

    /// Report that a texture is visible this frame with the given screen-space
    /// coverage and camera distance. This is called by the renderer after
    /// visibility determination.
    pub fn report_visibility(
        &mut self,
        id: StreamTextureId,
        screen_coverage: f32,
        camera_distance: f32,
    ) {
        if let Some(tex) = self.textures.get_mut(&id) {
            tex.visible = true;
            tex.last_used_frame = self.current_frame;
            tex.screen_coverage = screen_coverage;
            if camera_distance < tex.min_camera_distance {
                tex.min_camera_distance = camera_distance;
            }
        }
    }

    /// Clear per-frame visibility data. Call at the start of each frame
    /// before visibility feedback arrives.
    pub fn begin_frame(&mut self) {
        self.current_frame += 1;
        for tex in self.textures.values_mut() {
            tex.visible = false;
            tex.screen_coverage = 0.0;
            tex.min_camera_distance = f32::MAX;
        }
    }

    // -- Main update loop ----------------------------------------------------

    /// Run the streaming update logic for this frame. Should be called after
    /// all `report_visibility` calls for the frame.
    ///
    /// This function:
    /// 1. Computes priority for every texture.
    /// 2. Determines target mip levels.
    /// 3. Issues load requests for needed mips.
    /// 4. Runs eviction if the pool is over budget.
    /// 5. Updates statistics.
    pub fn update(&mut self) {
        self.stats = StreamingStats::default();
        self.stats.pool_budget_bytes = self.pool.budget_bytes;

        // 1. Compute priorities and target mip levels.
        let frame = self.current_frame;
        let params = self.priority_params.clone();
        let mip_bias = self.global_mip_bias;

        let mut texture_priorities: Vec<(StreamTextureId, f32, u32)> = Vec::new();
        for tex in self.textures.values_mut() {
            tex.priority = compute_priority(tex, frame, &params);
            tex.target_mip = compute_target_mip(
                tex.full_width,
                tex.full_height,
                tex.screen_coverage,
                tex.mip_count,
                mip_bias,
            );

            texture_priorities.push((tex.id, tex.priority, tex.target_mip));

            // Update stats.
            self.stats.total_textures += 1;
            if tex.visible {
                self.stats.visible_textures += 1;
            }
            if tex.is_at_target() {
                self.stats.at_target_count += 1;
            } else {
                self.stats.needs_streaming_count += 1;
            }
            self.stats.resident_bytes += tex.resident_bytes;
            self.stats.total_potential_bytes += tex.total_size_bytes();
        }

        // 2. Sort by priority (highest first) for load scheduling.
        texture_priorities.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // 3. Issue load requests for textures that need finer mips.
        let mut bytes_requested_this_frame = 0u64;
        let max_bytes = self.max_bytes_per_frame;
        let max_inflight = self.max_inflight_requests as usize;

        for (tex_id, priority, target_mip) in &texture_priorities {
            if self.inflight_requests.len() + self.pending_requests.len() >= max_inflight {
                break;
            }
            if bytes_requested_this_frame >= max_bytes {
                break;
            }

            let tex = match self.textures.get(tex_id) {
                Some(t) => t,
                None => continue,
            };

            if tex.is_at_target() || tex.is_loading() {
                continue;
            }

            // Request the next finer mip level.
            let next_mip = tex.current_resident_mip.saturating_sub(1);
            if next_mip < *target_mip {
                continue; // Don't load finer than needed.
            }
            // Actually, we want mips from target up to current. Load the
            // finest needed mip that is not yet resident/loading.
            for level in *target_mip..tex.current_resident_mip {
                let mip = &tex.mips[level as usize];
                if mip.state == MipState::NotLoaded || mip.state == MipState::Cached {
                    if bytes_requested_this_frame + mip.size_bytes <= max_bytes {
                        self.pending_requests.push(MipLoadRequest {
                            texture_id: *tex_id,
                            mip_level: level,
                            priority: *priority,
                            size_bytes: mip.size_bytes,
                            request_frame: frame,
                        });
                        bytes_requested_this_frame += mip.size_bytes;

                        // Mark as loading in the texture.
                        if let Some(tex_mut) = self.textures.get_mut(tex_id) {
                            tex_mut.mark_mip_loading(level);
                        }
                    }
                    break; // One mip at a time per texture.
                }
            }
        }

        // Sort pending requests by priority.
        self.pending_requests
            .sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));

        // 4. Run eviction if needed.
        if self.pool.needs_eviction() {
            let eviction_candidates = self.pool.compute_eviction_candidates(frame);
            let mut evicted_bytes = 0u64;
            let mut evicted_count = 0u32;

            for tex_id in eviction_candidates {
                if !self.pool.needs_eviction() {
                    break;
                }
                if let Some(tex) = self.textures.get_mut(&tex_id) {
                    if tex.pinned {
                        continue;
                    }
                    // Evict the finest resident mip.
                    let evict_level = tex.current_resident_mip;
                    if evict_level < tex.mip_count.saturating_sub(1) {
                        let freed = tex.mips[evict_level as usize].size_bytes;
                        tex.mark_mip_evicted(evict_level);
                        evicted_bytes += freed;
                        evicted_count += 1;
                    }
                }
                self.pool.remove(tex_id);
            }

            self.pool.last_eviction_count = evicted_count;
            self.pool.last_eviction_bytes = evicted_bytes;
            self.stats.mips_evicted_this_frame = evicted_count;
        }

        // 5. Update pool entries for all textures.
        for tex in self.textures.values() {
            self.pool.register(tex.id, tex.resident_bytes, frame, tex.priority);
        }

        self.stats.pending_loads = self.pending_requests.len() as u32
            + self.inflight_requests.len() as u32;
        self.stats.pool_utilisation = self.pool.utilisation();
    }

    /// Process completed mip load results from the background loader.
    pub fn process_load_results(&mut self, results: &[MipLoadResult]) {
        for result in results {
            self.inflight_requests
                .retain(|r| !(r.texture_id == result.texture_id && r.mip_level == result.mip_level));

            if result.success {
                if let Some(tex) = self.textures.get_mut(&result.texture_id) {
                    tex.mark_mip_resident(result.mip_level, self.current_frame);
                    self.stats.mips_loaded_this_frame += 1;
                }
            } else {
                if let Some(tex) = self.textures.get_mut(&result.texture_id) {
                    if let Some(mip) = tex.mips.get_mut(result.mip_level as usize) {
                        mip.state = MipState::Error;
                    }
                }
            }
        }
    }

    /// Drain pending load requests. The caller should submit these to the
    /// background asset loading system and later call `process_load_results`
    /// with the completed loads.
    pub fn drain_pending_requests(&mut self) -> Vec<MipLoadRequest> {
        let requests = std::mem::take(&mut self.pending_requests);
        self.inflight_requests.extend(requests.clone());
        requests
    }

    // -- Queries -------------------------------------------------------------

    /// Get the current mip level for a texture (what is actually available
    /// for rendering).
    pub fn current_mip_level(&self, id: StreamTextureId) -> Option<u32> {
        self.textures.get(&id).map(|t| t.current_resident_mip)
    }

    /// Get the target mip level for a texture.
    pub fn target_mip_level(&self, id: StreamTextureId) -> Option<u32> {
        self.textures.get(&id).map(|t| t.target_mip)
    }

    /// Get streaming statistics.
    pub fn stats(&self) -> &StreamingStats {
        &self.stats
    }

    /// Get pool statistics.
    pub fn pool(&self) -> &TexturePool {
        &self.pool
    }

    /// Total number of managed textures.
    pub fn texture_count(&self) -> usize {
        self.textures.len()
    }

    /// Current frame number.
    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Set the memory budget.
    pub fn set_budget(&mut self, budget_bytes: u64) {
        self.pool.budget_bytes = budget_bytes;
    }

    /// Iterate over all managed textures.
    pub fn iter_textures(&self) -> impl Iterator<Item = &StreamableTexture> {
        self.textures.values()
    }

    /// Force-load a texture to a specific mip level immediately (bypasses
    /// the priority system). Used for thumbnails and editor previews.
    pub fn force_target_mip(&mut self, id: StreamTextureId, mip_level: u32) {
        if let Some(tex) = self.textures.get_mut(&id) {
            tex.target_mip = mip_level.min(tex.mip_count.saturating_sub(1));
            tex.priority = f32::MAX - 1.0;
        }
    }
}

impl Default for TextureStreamingManager {
    fn default() -> Self {
        Self::new(512 * 1024 * 1024) // 512 MB default budget
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_count_computation() {
        assert_eq!(compute_mip_count(1024, 1024), 11); // 1024 -> 512 -> ... -> 1
        assert_eq!(compute_mip_count(256, 256), 9);
        assert_eq!(compute_mip_count(1, 1), 1);
        assert_eq!(compute_mip_count(2048, 1024), 12); // max(2048,1024)=2048, log2=11, +1
        assert_eq!(compute_mip_count(4096, 4096), 13);
    }

    #[test]
    fn streamable_texture_creation() {
        let tex = StreamableTexture::new(
            StreamTextureId(1),
            "test_texture",
            "textures/test.dds",
            1024,
            1024,
            TextureStreamFormat::RGBA8,
        );
        assert_eq!(tex.mip_count, 11);
        assert_eq!(tex.mips.len(), 11);
        assert_eq!(tex.full_width, 1024);
        assert_eq!(tex.current_resident_mip, 10); // Start with lowest mip
        assert!(tex.total_size_bytes() > 0);

        // Mip 0 should be 1024x1024 * 4 bytes.
        assert_eq!(tex.mips[0].size_bytes, 1024 * 1024 * 4);
        // Mip 1 should be 512x512 * 4 bytes.
        assert_eq!(tex.mips[1].size_bytes, 512 * 512 * 4);
    }

    #[test]
    fn target_mip_computation() {
        // Texture is 1024x1024, covers 512x512 pixels on screen.
        // Texel ratio = (1024*1024) / (512*512) = 4
        // ideal_mip = 0.5 * log2(4) = 0.5 * 2 = 1
        let mip = compute_target_mip(1024, 1024, 512.0 * 512.0, 11, 0.0);
        assert_eq!(mip, 1);

        // Full coverage: 1:1 mapping -> mip 0.
        let mip = compute_target_mip(1024, 1024, 1024.0 * 1024.0, 11, 0.0);
        assert_eq!(mip, 0);

        // Very small coverage -> high mip level.
        let mip = compute_target_mip(1024, 1024, 16.0 * 16.0, 11, 0.0);
        assert!(mip >= 3);

        // Zero coverage -> highest mip.
        let mip = compute_target_mip(1024, 1024, 0.0, 11, 0.0);
        assert_eq!(mip, 10);

        // With positive bias (blurrier).
        let mip_biased = compute_target_mip(1024, 1024, 512.0 * 512.0, 11, 1.0);
        assert!(mip_biased > mip || mip_biased == mip);
    }

    #[test]
    fn priority_computation_basics() {
        let params = PriorityParams::default();

        let mut tex = StreamableTexture::new(
            StreamTextureId(1), "visible", "tex.dds", 1024, 1024, TextureStreamFormat::RGBA8,
        );
        tex.visible = true;
        tex.screen_coverage = 10000.0;
        tex.min_camera_distance = 10.0;
        tex.last_used_frame = 100;

        let priority_visible = compute_priority(&tex, 100, &params);
        assert!(priority_visible > 0.0);

        // Invisible, old texture should have lower priority.
        let mut tex2 = StreamableTexture::new(
            StreamTextureId(2), "old", "tex2.dds", 1024, 1024, TextureStreamFormat::RGBA8,
        );
        tex2.visible = false;
        tex2.screen_coverage = 0.0;
        tex2.min_camera_distance = 500.0;
        tex2.last_used_frame = 10;

        let priority_old = compute_priority(&tex2, 100, &params);
        assert!(priority_visible > priority_old);

        // Pinned texture should have MAX priority.
        let mut tex3 = tex.clone();
        tex3.pinned = true;
        let priority_pinned = compute_priority(&tex3, 100, &params);
        assert_eq!(priority_pinned, f32::MAX);
    }

    #[test]
    fn texture_pool_basics() {
        let mut pool = TexturePool::new(1024 * 1024); // 1 MB budget

        pool.register(StreamTextureId(1), 500_000, 0, 1.0);
        assert_eq!(pool.used_bytes, 500_000);
        assert!(!pool.is_over_budget());

        pool.register(StreamTextureId(2), 600_000, 0, 0.5);
        assert_eq!(pool.used_bytes, 1_100_000);
        assert!(pool.is_over_budget());

        // Eviction candidates should include the lower-priority texture.
        let candidates = pool.compute_eviction_candidates(1);
        assert!(!candidates.is_empty());
        // The lower-priority texture (id=2) should be first.
        assert_eq!(candidates[0], StreamTextureId(2));
    }

    #[test]
    fn texture_pool_eviction_threshold() {
        let mut pool = TexturePool::new(1_000_000);
        pool.eviction_threshold = 0.8;
        pool.eviction_target = 0.6;

        // Add 700KB -- under 80% threshold.
        pool.register(StreamTextureId(1), 700_000, 0, 1.0);
        assert!(!pool.needs_eviction());

        // Add 200KB more -- now at 90%, over 80% threshold.
        pool.register(StreamTextureId(2), 200_000, 0, 0.5);
        assert!(pool.needs_eviction());

        // Need to free down to 60% = 600KB. Currently at 900KB.
        assert_eq!(pool.bytes_to_free(), 300_000);
    }

    #[test]
    fn streaming_manager_lifecycle() {
        let mut mgr = TextureStreamingManager::new(256 * 1024 * 1024);

        // Register a texture.
        let id = mgr.register_texture(
            "test_texture",
            "textures/test.dds",
            2048,
            2048,
            TextureStreamFormat::BC3,
        );
        assert!(id.is_valid());
        assert_eq!(mgr.texture_count(), 1);

        // Report visibility.
        mgr.begin_frame();
        mgr.report_visibility(id, 1000.0 * 1000.0, 5.0);

        // Run update.
        mgr.update();

        let stats = mgr.stats();
        assert_eq!(stats.total_textures, 1);
        assert_eq!(stats.visible_textures, 1);

        // The target mip should be computed.
        let target = mgr.target_mip_level(id).unwrap();
        assert!(target < 12); // Should want a reasonably fine mip.

        // Unregister.
        mgr.unregister_texture(id);
        assert_eq!(mgr.texture_count(), 0);
    }

    #[test]
    fn mip_state_transitions() {
        let mut tex = StreamableTexture::new(
            StreamTextureId(1), "test", "tex.dds", 256, 256, TextureStreamFormat::RGBA8,
        );

        // Initial state: only the lowest mip is "resident" conceptually.
        assert_eq!(tex.mips[0].state, MipState::NotLoaded);

        // Start loading mip 5.
        tex.mark_mip_loading(5);
        assert_eq!(tex.mips[5].state, MipState::Loading);
        assert!(tex.is_loading());

        // Complete loading mip 5.
        tex.mark_mip_resident(5, 10);
        assert_eq!(tex.mips[5].state, MipState::Resident);

        // Evict mip 5.
        tex.mark_mip_evicted(5);
        assert_eq!(tex.mips[5].state, MipState::Cached);
    }

    #[test]
    fn format_size_calculations() {
        assert_eq!(TextureStreamFormat::RGBA8.mip_size_bytes(1024, 1024), 1024 * 1024 * 4);
        assert_eq!(TextureStreamFormat::BC1.mip_size_bytes(1024, 1024), 1024 * 1024 / 2);
        assert_eq!(TextureStreamFormat::BC7.mip_size_bytes(1024, 1024), 1024 * 1024);
        assert_eq!(TextureStreamFormat::RGBA16F.mip_size_bytes(512, 512), 512 * 512 * 8);
    }

    #[test]
    fn pinned_texture_never_evicted() {
        let mut mgr = TextureStreamingManager::new(1024); // Tiny budget

        let id = mgr.register_texture(
            "pinned", "tex.dds", 4096, 4096, TextureStreamFormat::RGBA8,
        );
        mgr.pin_texture(id);

        mgr.begin_frame();
        mgr.update();

        // Even though the budget is tiny, the pinned texture should have
        // target mip 0 (full res).
        let target = mgr.target_mip_level(id).unwrap();
        assert_eq!(target, 0);
    }

    #[test]
    fn bytes_needed_and_freed() {
        let mut tex = StreamableTexture::new(
            StreamTextureId(1), "test", "tex.dds", 256, 256, TextureStreamFormat::RGBA8,
        );

        // Mark mips 6, 7, 8 as resident.
        tex.mark_mip_resident(6, 1);
        tex.mark_mip_resident(7, 1);
        tex.mark_mip_resident(8, 1);

        // Target mip 4 -- need mips 4 and 5.
        tex.target_mip = 4;
        let needed = tex.bytes_needed_for_target();
        assert!(needed > 0);

        // If we evict down to mip 7, we free mip 6's bytes.
        let freed = tex.bytes_freed_by_eviction(7);
        assert!(freed > 0);
        assert_eq!(freed, tex.mips[6].size_bytes);
    }

    #[test]
    fn process_load_results_updates_state() {
        let mut mgr = TextureStreamingManager::new(256 * 1024 * 1024);

        let id = mgr.register_texture(
            "test", "tex.dds", 512, 512, TextureStreamFormat::RGBA8,
        );

        // Simulate a load completing.
        let result = MipLoadResult {
            texture_id: id,
            mip_level: 5,
            data: vec![0u8; 1024],
            success: true,
            error: None,
        };

        mgr.process_load_results(&[result]);

        let tex = mgr.get_texture(id).unwrap();
        assert_eq!(tex.mips[5].state, MipState::Resident);
    }

    #[test]
    fn load_error_marks_mip_as_error() {
        let mut mgr = TextureStreamingManager::new(256 * 1024 * 1024);

        let id = mgr.register_texture(
            "test", "tex.dds", 512, 512, TextureStreamFormat::RGBA8,
        );

        let result = MipLoadResult {
            texture_id: id,
            mip_level: 3,
            data: Vec::new(),
            success: false,
            error: Some("Disk read error".to_string()),
        };

        mgr.process_load_results(&[result]);

        let tex = mgr.get_texture(id).unwrap();
        assert_eq!(tex.mips[3].state, MipState::Error);
    }

    #[test]
    fn streaming_manager_multiple_textures() {
        let mut mgr = TextureStreamingManager::new(512 * 1024 * 1024);

        let id1 = mgr.register_texture("tex1", "t1.dds", 2048, 2048, TextureStreamFormat::BC3);
        let id2 = mgr.register_texture("tex2", "t2.dds", 1024, 1024, TextureStreamFormat::RGBA8);
        let id3 = mgr.register_texture("tex3", "t3.dds", 4096, 4096, TextureStreamFormat::BC7);

        mgr.begin_frame();
        mgr.report_visibility(id1, 500000.0, 10.0);
        mgr.report_visibility(id2, 100000.0, 50.0);
        // id3 is not visible.

        mgr.update();

        let stats = mgr.stats();
        assert_eq!(stats.total_textures, 3);
        assert_eq!(stats.visible_textures, 2);

        // The visible, closer texture should have higher priority.
        let tex1 = mgr.get_texture(id1).unwrap();
        let tex3 = mgr.get_texture(id3).unwrap();
        assert!(tex1.priority > tex3.priority);
    }
}
