// engine/render/src/virtual_shadow_maps/mod.rs
//
// Virtual Shadow Maps (VSM) for the Genovo engine. Implements an Unreal
// Engine 5-style virtual shadow map system with huge virtual shadow maps
// backed by a physical page pool, LRU eviction, clipmap mip levels for
// directional lights, and one-pass shadow projection.
//
// The virtual shadow map uses a page table to map virtual pages to physical
// pages on demand. Only pages that are needed (visible geometry projects
// into them) are allocated each frame, keeping memory usage bounded.

use glam::{Mat4, Vec2, Vec3, Vec4};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default virtual shadow map resolution (one axis).
pub const DEFAULT_VIRTUAL_RESOLUTION: u32 = 16384;

/// Default physical page size in texels.
pub const DEFAULT_PAGE_SIZE: u32 = 128;

/// Default number of physical pages in the pool.
pub const DEFAULT_TOTAL_PAGES: u32 = 4096;

/// Default number of mip levels for directional light clipmaps.
pub const DEFAULT_CLIPMAP_LEVELS: u32 = 8;

/// Maximum supported shadow maps (point + spot + directional).
pub const MAX_SHADOW_MAPS: u32 = 128;

// ---------------------------------------------------------------------------
// VirtualPageId
// ---------------------------------------------------------------------------

/// Uniquely identifies a virtual page within the virtual shadow map system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VirtualPageId {
    /// Which shadow map this page belongs to.
    pub shadow_map_id: u32,
    /// Mip level (0 = finest).
    pub mip_level: u32,
    /// Page X coordinate within the mip level.
    pub page_x: u32,
    /// Page Y coordinate within the mip level.
    pub page_y: u32,
}

impl VirtualPageId {
    /// Create a new virtual page ID.
    pub fn new(shadow_map_id: u32, mip_level: u32, page_x: u32, page_y: u32) -> Self {
        Self {
            shadow_map_id,
            mip_level,
            page_x,
            page_y,
        }
    }

    /// Pack into a single u64 for compact storage.
    pub fn pack(&self) -> u64 {
        (self.shadow_map_id as u64) << 48
            | (self.mip_level as u64) << 40
            | (self.page_x as u64) << 20
            | (self.page_y as u64)
    }

    /// Unpack from a u64.
    pub fn unpack(packed: u64) -> Self {
        Self {
            shadow_map_id: (packed >> 48) as u32,
            mip_level: ((packed >> 40) & 0xFF) as u32,
            page_x: ((packed >> 20) & 0xFFFFF) as u32,
            page_y: (packed & 0xFFFFF) as u32,
        }
    }
}

// ---------------------------------------------------------------------------
// PhysicalPageId
// ---------------------------------------------------------------------------

/// Index of a physical page in the page pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicalPageId(pub u32);

impl PhysicalPageId {
    /// Invalid page sentinel.
    pub const INVALID: Self = Self(u32::MAX);

    /// Check if this is a valid page ID.
    pub fn is_valid(&self) -> bool {
        self.0 != u32::MAX
    }
}

// ---------------------------------------------------------------------------
// PageTableEntry
// ---------------------------------------------------------------------------

/// An entry in the virtual-to-physical page table.
#[derive(Debug, Clone, Copy)]
pub struct PageTableEntry {
    /// Physical page index (INVALID if not allocated).
    pub physical_page: PhysicalPageId,
    /// Whether this page has valid depth data.
    pub valid: bool,
    /// Whether this page was requested this frame.
    pub requested: bool,
    /// Frame number when this page was last used.
    pub last_used_frame: u64,
    /// Whether this page has been rendered this frame.
    pub rendered: bool,
}

impl PageTableEntry {
    /// Create an empty (unmapped) page table entry.
    pub fn empty() -> Self {
        Self {
            physical_page: PhysicalPageId::INVALID,
            valid: false,
            requested: false,
            last_used_frame: 0,
            rendered: false,
        }
    }

    /// Whether this page has a physical allocation.
    pub fn is_allocated(&self) -> bool {
        self.physical_page.is_valid()
    }
}

// ---------------------------------------------------------------------------
// PhysicalPage
// ---------------------------------------------------------------------------

/// A physical page in the shadow map pool.
#[derive(Debug, Clone)]
pub struct PhysicalPage {
    /// Depth data for this page (page_size * page_size).
    pub depth_data: Vec<f32>,
    /// Which virtual page this physical page is currently mapped to.
    pub mapped_to: Option<VirtualPageId>,
    /// Frame when this page was last rendered.
    pub last_rendered_frame: u64,
    /// Whether this page's data is valid.
    pub data_valid: bool,
    /// Page size in texels.
    pub page_size: u32,
}

impl PhysicalPage {
    /// Create a new physical page.
    pub fn new(page_size: u32) -> Self {
        Self {
            depth_data: vec![1.0; (page_size * page_size) as usize],
            mapped_to: None,
            last_rendered_frame: 0,
            data_valid: false,
            page_size,
        }
    }

    /// Clear the page to maximum depth.
    pub fn clear(&mut self) {
        for d in &mut self.depth_data {
            *d = 1.0;
        }
        self.data_valid = false;
    }

    /// Write a depth value at a texel coordinate.
    #[inline]
    pub fn write_depth(&mut self, x: u32, y: u32, depth: f32) {
        let idx = (y * self.page_size + x) as usize;
        if idx < self.depth_data.len() {
            self.depth_data[idx] = self.depth_data[idx].min(depth);
        }
    }

    /// Read the depth at a texel coordinate.
    #[inline]
    pub fn read_depth(&self, x: u32, y: u32) -> f32 {
        let idx = (y * self.page_size + x) as usize;
        self.depth_data.get(idx).copied().unwrap_or(1.0)
    }

    /// Sample depth with bilinear interpolation.
    pub fn sample_depth_bilinear(&self, u: f32, v: f32) -> f32 {
        let fx = u * (self.page_size - 1) as f32;
        let fy = v * (self.page_size - 1) as f32;

        let x0 = fx.floor() as u32;
        let y0 = fy.floor() as u32;
        let x1 = (x0 + 1).min(self.page_size - 1);
        let y1 = (y0 + 1).min(self.page_size - 1);

        let frac_x = fx - fx.floor();
        let frac_y = fy - fy.floor();

        let d00 = self.read_depth(x0, y0);
        let d10 = self.read_depth(x1, y0);
        let d01 = self.read_depth(x0, y1);
        let d11 = self.read_depth(x1, y1);

        let top = d00 * (1.0 - frac_x) + d10 * frac_x;
        let bottom = d01 * (1.0 - frac_x) + d11 * frac_x;
        top * (1.0 - frac_y) + bottom * frac_y
    }
}

// ---------------------------------------------------------------------------
// PhysicalPagePool
// ---------------------------------------------------------------------------

/// Fixed-size pool of physical page textures with LRU eviction.
pub struct PhysicalPagePool {
    /// All physical pages.
    pub pages: Vec<PhysicalPage>,
    /// Free list: indices of pages not currently mapped.
    free_list: VecDeque<u32>,
    /// LRU order: front = least recently used.
    lru_order: VecDeque<u32>,
    /// Map from physical page index to its position in LRU list.
    /// (We use the LRU deque directly and scan when needed.)
    /// Page size in texels.
    pub page_size: u32,
    /// Total number of pages.
    pub total_pages: u32,
    /// Number of currently allocated (mapped) pages.
    pub allocated_count: u32,
    /// Statistics: total allocations.
    pub total_allocations: u64,
    /// Statistics: total evictions.
    pub total_evictions: u64,
}

impl PhysicalPagePool {
    /// Create a new page pool.
    pub fn new(total_pages: u32, page_size: u32) -> Self {
        let mut pages = Vec::with_capacity(total_pages as usize);
        let mut free_list = VecDeque::with_capacity(total_pages as usize);

        for i in 0..total_pages {
            pages.push(PhysicalPage::new(page_size));
            free_list.push_back(i);
        }

        Self {
            pages,
            free_list,
            lru_order: VecDeque::new(),
            page_size,
            total_pages,
            allocated_count: 0,
            total_allocations: 0,
            total_evictions: 0,
        }
    }

    /// Allocate a physical page. Returns `None` if the pool is exhausted
    /// and all pages are in use.
    pub fn allocate_page(&mut self) -> Option<PhysicalPageId> {
        if let Some(idx) = self.free_list.pop_front() {
            self.pages[idx as usize].clear();
            self.lru_order.push_back(idx);
            self.allocated_count += 1;
            self.total_allocations += 1;
            Some(PhysicalPageId(idx))
        } else {
            None
        }
    }

    /// Free a physical page back to the pool.
    pub fn free_page(&mut self, id: PhysicalPageId) {
        if !id.is_valid() || id.0 >= self.total_pages {
            return;
        }

        let page = &mut self.pages[id.0 as usize];
        page.mapped_to = None;
        page.data_valid = false;

        // Remove from LRU.
        self.lru_order.retain(|&x| x != id.0);

        self.free_list.push_back(id.0);
        self.allocated_count = self.allocated_count.saturating_sub(1);
    }

    /// Evict the least recently used page and return it.
    /// Returns the evicted physical page ID and the virtual page it was mapped to.
    pub fn evict_lru(&mut self) -> Option<(PhysicalPageId, Option<VirtualPageId>)> {
        if let Some(idx) = self.lru_order.pop_front() {
            let page = &mut self.pages[idx as usize];
            let old_mapping = page.mapped_to.take();
            page.data_valid = false;
            page.clear();

            self.total_evictions += 1;
            // Do not push to free list -- the caller will reuse it.
            Some((PhysicalPageId(idx), old_mapping))
        } else {
            None
        }
    }

    /// Mark a page as recently used (move to back of LRU).
    pub fn touch_page(&mut self, id: PhysicalPageId) {
        if !id.is_valid() {
            return;
        }
        // Remove from current position and push to back.
        self.lru_order.retain(|&x| x != id.0);
        self.lru_order.push_back(id.0);
    }

    /// Get a physical page by ID.
    pub fn get_page(&self, id: PhysicalPageId) -> Option<&PhysicalPage> {
        if id.is_valid() && (id.0 as usize) < self.pages.len() {
            Some(&self.pages[id.0 as usize])
        } else {
            None
        }
    }

    /// Get a mutable physical page by ID.
    pub fn get_page_mut(&mut self, id: PhysicalPageId) -> Option<&mut PhysicalPage> {
        if id.is_valid() && (id.0 as usize) < self.pages.len() {
            Some(&mut self.pages[id.0 as usize])
        } else {
            None
        }
    }

    /// Number of free pages.
    pub fn free_count(&self) -> u32 {
        self.free_list.len() as u32
    }

    /// Occupancy ratio (0 = empty, 1 = full).
    pub fn occupancy(&self) -> f32 {
        self.allocated_count as f32 / self.total_pages.max(1) as f32
    }

    /// Reset the pool: free all pages.
    pub fn reset(&mut self) {
        self.free_list.clear();
        self.lru_order.clear();
        for i in 0..self.total_pages {
            self.pages[i as usize].clear();
            self.pages[i as usize].mapped_to = None;
            self.free_list.push_back(i);
        }
        self.allocated_count = 0;
    }
}

// ---------------------------------------------------------------------------
// VirtualShadowMap
// ---------------------------------------------------------------------------

/// A single virtual shadow map (one per light or per directional light
/// clipmap level). Manages the page table mapping virtual pages to the
/// physical page pool.
pub struct VirtualShadowMap {
    /// ID of this shadow map.
    pub id: u32,
    /// Virtual resolution (one axis, e.g. 16384).
    pub virtual_resolution: u32,
    /// Page size in texels.
    pub page_size: u32,
    /// Number of pages per axis.
    pub pages_per_axis: u32,
    /// Number of mip levels.
    pub mip_count: u32,
    /// Page table: virtual page -> physical page mapping.
    /// Indexed by (mip * pages_per_mip + y * pages_per_axis_mip + x).
    pub page_table: Vec<PageTableEntry>,
    /// Light-space view-projection matrix.
    pub light_view_proj: Mat4,
    /// Inverse of the light view-projection.
    pub inv_light_view_proj: Mat4,
    /// Light type.
    pub light_type: ShadowLightType,
    /// Near/far planes for depth.
    pub near_plane: f32,
    pub far_plane: f32,
    /// Depth bias.
    pub depth_bias: f32,
    /// Normal bias.
    pub normal_bias: f32,
    /// Whether this shadow map is active this frame.
    pub active: bool,
    /// Current frame number.
    pub current_frame: u64,
}

/// Type of light casting a virtual shadow map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowLightType {
    Directional,
    Spot,
    Point,
}

impl VirtualShadowMap {
    /// Create a new virtual shadow map.
    pub fn new(
        id: u32,
        virtual_resolution: u32,
        page_size: u32,
        light_type: ShadowLightType,
    ) -> Self {
        let pages_per_axis = virtual_resolution / page_size;
        let mip_count = compute_mip_count(pages_per_axis);

        // Calculate total page table entries across all mip levels.
        let mut total_entries = 0usize;
        let mut ppa = pages_per_axis;
        for _ in 0..mip_count {
            total_entries += (ppa * ppa) as usize;
            ppa = (ppa / 2).max(1);
        }

        Self {
            id,
            virtual_resolution,
            page_size,
            pages_per_axis,
            mip_count,
            page_table: vec![PageTableEntry::empty(); total_entries],
            light_view_proj: Mat4::IDENTITY,
            inv_light_view_proj: Mat4::IDENTITY,
            light_type,
            near_plane: 0.1,
            far_plane: 500.0,
            depth_bias: 0.005,
            normal_bias: 0.02,
            active: false,
            current_frame: 0,
        }
    }

    /// Set the light-space matrix.
    pub fn set_light_matrix(&mut self, view_proj: Mat4) {
        self.light_view_proj = view_proj;
        self.inv_light_view_proj = view_proj.inverse();
    }

    /// Pages per axis at a given mip level.
    pub fn pages_per_axis_at_mip(&self, mip: u32) -> u32 {
        (self.pages_per_axis >> mip).max(1)
    }

    /// Get the page table offset for a mip level.
    fn mip_offset(&self, mip: u32) -> usize {
        let mut offset = 0usize;
        let mut ppa = self.pages_per_axis;
        for m in 0..mip {
            offset += (ppa * ppa) as usize;
            ppa = (ppa / 2).max(1);
        }
        offset
    }

    /// Page table index for a virtual page.
    fn page_index(&self, page: &VirtualPageId) -> usize {
        let ppa = self.pages_per_axis_at_mip(page.mip_level);
        let base = self.mip_offset(page.mip_level);
        base + (page.page_y * ppa + page.page_x) as usize
    }

    /// Get the page table entry for a virtual page.
    pub fn get_entry(&self, page: &VirtualPageId) -> &PageTableEntry {
        let idx = self.page_index(page);
        &self.page_table[idx]
    }

    /// Get a mutable page table entry.
    pub fn get_entry_mut(&mut self, page: &VirtualPageId) -> &mut PageTableEntry {
        let idx = self.page_index(page);
        &mut self.page_table[idx]
    }

    /// Mark a virtual page as requested (needed) this frame.
    pub fn request_page(&mut self, page: &VirtualPageId) {
        let idx = self.page_index(page);
        if idx < self.page_table.len() {
            self.page_table[idx].requested = true;
        }
    }

    /// Clear all page request flags (call at start of frame).
    pub fn begin_frame(&mut self, frame: u64) {
        self.current_frame = frame;
        for entry in &mut self.page_table {
            entry.requested = false;
            entry.rendered = false;
        }
    }

    /// Project a world-space point to virtual shadow map coordinates.
    ///
    /// Returns (page_x, page_y, mip_level, depth) or None if outside.
    pub fn project_point(&self, world_pos: Vec3) -> Option<(u32, u32, f32)> {
        let clip = self.light_view_proj
            * Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);
        if clip.w <= 0.0 {
            return None;
        }

        let ndc = Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);

        // NDC to UV [0, 1].
        let u = ndc.x * 0.5 + 0.5;
        let v = 1.0 - (ndc.y * 0.5 + 0.5);
        let depth = ndc.z * 0.5 + 0.5;

        if u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0 || depth < 0.0 || depth > 1.0 {
            return None;
        }

        let px = (u * self.virtual_resolution as f32) as u32;
        let py = (v * self.virtual_resolution as f32) as u32;

        Some((px, py, depth))
    }

    /// Determine which virtual page a pixel coordinate falls in.
    pub fn pixel_to_page(&self, px: u32, py: u32, mip: u32) -> VirtualPageId {
        let ppa = self.pages_per_axis_at_mip(mip);
        let page_x = (px / (self.page_size >> mip).max(1)).min(ppa - 1);
        let page_y = (py / (self.page_size >> mip).max(1)).min(ppa - 1);
        VirtualPageId::new(self.id, mip, page_x, page_y)
    }

    /// Mark pages needed by a set of projected triangles.
    ///
    /// For each triangle projected into shadow map space, marks the
    /// pages it overlaps as requested.
    pub fn mark_pages_for_triangles(
        &mut self,
        triangles_ndc: &[(Vec3, Vec3, Vec3)],
    ) {
        for &(v0, v1, v2) in triangles_ndc {
            // Compute the AABB of the triangle in UV space [0, 1].
            let min_u = v0.x.min(v1.x).min(v2.x) * 0.5 + 0.5;
            let max_u = v0.x.max(v1.x).max(v2.x) * 0.5 + 0.5;
            let min_v = 1.0 - (v0.y.max(v1.y).max(v2.y) * 0.5 + 0.5);
            let max_v = 1.0 - (v0.y.min(v1.y).min(v2.y) * 0.5 + 0.5);

            if max_u < 0.0 || min_u > 1.0 || max_v < 0.0 || min_v > 1.0 {
                continue;
            }

            let min_u = min_u.clamp(0.0, 1.0);
            let max_u = max_u.clamp(0.0, 1.0);
            let min_v = min_v.clamp(0.0, 1.0);
            let max_v = max_v.clamp(0.0, 1.0);

            // Convert UV to page coordinates at mip 0.
            let ppa = self.pages_per_axis;
            let px_min = (min_u * ppa as f32).floor() as u32;
            let px_max = (max_u * ppa as f32).ceil().min(ppa as f32) as u32;
            let py_min = (min_v * ppa as f32).floor() as u32;
            let py_max = (max_v * ppa as f32).ceil().min(ppa as f32) as u32;

            for py in py_min..py_max {
                for px in px_min..px_max {
                    let page = VirtualPageId::new(self.id, 0, px, py);
                    self.request_page(&page);
                }
            }
        }
    }

    /// Collect all requested pages for this frame.
    pub fn collect_requested_pages(&self) -> Vec<VirtualPageId> {
        let mut result = Vec::new();
        let mut ppa = self.pages_per_axis;
        let mut offset = 0usize;

        for mip in 0..self.mip_count {
            for py in 0..ppa {
                for px in 0..ppa {
                    let idx = offset + (py * ppa + px) as usize;
                    if idx < self.page_table.len() && self.page_table[idx].requested {
                        result.push(VirtualPageId::new(self.id, mip, px, py));
                    }
                }
            }
            offset += (ppa * ppa) as usize;
            ppa = (ppa / 2).max(1);
        }

        result
    }

    /// Count currently allocated (mapped) pages.
    pub fn allocated_page_count(&self) -> u32 {
        self.page_table
            .iter()
            .filter(|e| e.is_allocated())
            .count() as u32
    }

    /// Count requested pages this frame.
    pub fn requested_page_count(&self) -> u32 {
        self.page_table.iter().filter(|e| e.requested).count() as u32
    }

    /// Sample the shadow map at a world position.
    ///
    /// Returns (shadow_factor, valid) where shadow_factor is 1.0 for fully
    /// lit and 0.0 for fully shadowed.
    pub fn sample_shadow(
        &self,
        world_pos: Vec3,
        world_normal: Vec3,
        pool: &PhysicalPagePool,
    ) -> (f32, bool) {
        // Apply normal bias.
        let biased_pos = world_pos + world_normal * self.normal_bias;

        let (px, py, depth) = match self.project_point(biased_pos) {
            Some(v) => v,
            None => return (1.0, false),
        };

        // Find the page.
        let page = self.pixel_to_page(px, py, 0);
        let entry = self.get_entry(&page);

        if !entry.is_allocated() || !entry.valid {
            return (1.0, false); // No data, assume lit.
        }

        // Look up the physical page.
        let phys_page = match pool.get_page(entry.physical_page) {
            Some(p) => p,
            None => return (1.0, false),
        };

        // Compute UV within the page.
        let ppa = self.pages_per_axis;
        let page_uv_x = (px as f32 / self.virtual_resolution as f32 * ppa as f32)
            - page.page_x as f32;
        let page_uv_y = (py as f32 / self.virtual_resolution as f32 * ppa as f32)
            - page.page_y as f32;

        let stored_depth = phys_page.sample_depth_bilinear(
            page_uv_x.clamp(0.0, 1.0),
            page_uv_y.clamp(0.0, 1.0),
        );

        let biased_depth = depth - self.depth_bias;

        if biased_depth > stored_depth {
            (0.0, true) // In shadow.
        } else {
            (1.0, true) // Lit.
        }
    }

    /// Invalidate all pages (e.g., when the light changes direction).
    pub fn invalidate_all(&mut self) {
        for entry in &mut self.page_table {
            entry.valid = false;
        }
    }

    /// Total page table entries across all mips.
    pub fn total_page_entries(&self) -> usize {
        self.page_table.len()
    }
}

/// Compute the number of mip levels for a given pages-per-axis count.
fn compute_mip_count(pages_per_axis: u32) -> u32 {
    let mut count = 1u32;
    let mut ppa = pages_per_axis;
    while ppa > 1 {
        ppa /= 2;
        count += 1;
    }
    count
}

// ---------------------------------------------------------------------------
// VirtualShadowMapSettings
// ---------------------------------------------------------------------------

/// Global settings for the virtual shadow map system.
#[derive(Debug, Clone)]
pub struct VirtualShadowMapSettings {
    /// Page size in texels.
    pub page_size: u32,
    /// Total number of physical pages in the pool.
    pub total_pages: u32,
    /// Virtual resolution for directional lights.
    pub directional_resolution: u32,
    /// Virtual resolution for local lights (point/spot).
    pub local_resolution: u32,
    /// Number of clipmap levels for directional lights.
    pub directional_mip_levels: u32,
    /// Depth bias.
    pub depth_bias: f32,
    /// Normal bias.
    pub normal_bias: f32,
    /// Whether to use page caching across frames.
    pub enable_caching: bool,
    /// Maximum age (in frames) for a cached page before eviction.
    pub max_cache_age: u32,
    /// Whether to use large pages (256x256) for close-up detail.
    pub use_large_pages: bool,
}

impl Default for VirtualShadowMapSettings {
    fn default() -> Self {
        Self {
            page_size: DEFAULT_PAGE_SIZE,
            total_pages: DEFAULT_TOTAL_PAGES,
            directional_resolution: DEFAULT_VIRTUAL_RESOLUTION,
            local_resolution: 8192,
            directional_mip_levels: DEFAULT_CLIPMAP_LEVELS,
            depth_bias: 0.005,
            normal_bias: 0.02,
            enable_caching: true,
            max_cache_age: 4,
            use_large_pages: false,
        }
    }
}

impl VirtualShadowMapSettings {
    /// High quality settings.
    pub fn high_quality() -> Self {
        Self {
            page_size: 256,
            total_pages: 8192,
            directional_resolution: 32768,
            local_resolution: 16384,
            directional_mip_levels: 10,
            ..Default::default()
        }
    }

    /// Low quality settings for mobile.
    pub fn low_quality() -> Self {
        Self {
            page_size: 64,
            total_pages: 1024,
            directional_resolution: 8192,
            local_resolution: 4096,
            directional_mip_levels: 6,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// VirtualShadowMapManager
// ---------------------------------------------------------------------------

/// Manages all virtual shadow maps and the shared physical page pool.
pub struct VirtualShadowMapManager {
    /// Settings.
    pub settings: VirtualShadowMapSettings,
    /// The physical page pool shared by all shadow maps.
    pub page_pool: PhysicalPagePool,
    /// All active virtual shadow maps.
    pub shadow_maps: Vec<VirtualShadowMap>,
    /// Current frame number.
    pub current_frame: u64,
    /// Per-frame statistics.
    pub stats: VSMFrameStats,
}

/// Per-frame statistics for the virtual shadow map system.
#[derive(Debug, Default, Clone, Copy)]
pub struct VSMFrameStats {
    /// Number of pages requested this frame.
    pub pages_requested: u32,
    /// Number of pages newly allocated this frame.
    pub pages_allocated: u32,
    /// Number of pages evicted this frame.
    pub pages_evicted: u32,
    /// Number of pages reused from cache.
    pub pages_cached: u32,
    /// Number of pages rendered this frame.
    pub pages_rendered: u32,
    /// Total active shadow maps.
    pub active_shadow_maps: u32,
    /// Page pool occupancy.
    pub pool_occupancy: f32,
}

impl VirtualShadowMapManager {
    /// Create a new virtual shadow map manager.
    pub fn new(settings: VirtualShadowMapSettings) -> Self {
        let pool = PhysicalPagePool::new(settings.total_pages, settings.page_size);
        Self {
            settings,
            page_pool: pool,
            shadow_maps: Vec::new(),
            current_frame: 0,
            stats: VSMFrameStats::default(),
        }
    }

    /// Add a virtual shadow map. Returns its index.
    pub fn add_shadow_map(&mut self, light_type: ShadowLightType) -> u32 {
        let resolution = match light_type {
            ShadowLightType::Directional => self.settings.directional_resolution,
            _ => self.settings.local_resolution,
        };

        let id = self.shadow_maps.len() as u32;
        let mut vsm = VirtualShadowMap::new(
            id,
            resolution,
            self.settings.page_size,
            light_type,
        );
        vsm.depth_bias = self.settings.depth_bias;
        vsm.normal_bias = self.settings.normal_bias;
        self.shadow_maps.push(vsm);
        id
    }

    /// Remove a shadow map by index.
    pub fn remove_shadow_map(&mut self, index: u32) {
        if (index as usize) < self.shadow_maps.len() {
            // Free all physical pages mapped to this shadow map.
            let vsm = &self.shadow_maps[index as usize];
            for entry in &vsm.page_table {
                if entry.is_allocated() {
                    self.page_pool.free_page(entry.physical_page);
                }
            }
            self.shadow_maps.remove(index as usize);
        }
    }

    /// Begin a new frame: clear request flags, prepare for page management.
    pub fn begin_frame(&mut self) {
        self.current_frame += 1;
        self.stats = VSMFrameStats::default();

        for vsm in &mut self.shadow_maps {
            vsm.begin_frame(self.current_frame);
        }
    }

    /// Allocate physical pages for all requested virtual pages.
    ///
    /// This is the main per-frame page management step:
    /// 1. Collect all requested pages.
    /// 2. For pages already allocated, mark as cached.
    /// 3. For new pages, allocate from pool.
    /// 4. If pool is exhausted, evict LRU pages.
    pub fn allocate_requested_pages(&mut self) {
        let mut all_requests: Vec<(u32, VirtualPageId)> = Vec::new();

        for (sm_idx, vsm) in self.shadow_maps.iter().enumerate() {
            if !vsm.active {
                continue;
            }
            let pages = vsm.collect_requested_pages();
            for page in pages {
                all_requests.push((sm_idx as u32, page));
            }
        }

        self.stats.pages_requested = all_requests.len() as u32;

        for (sm_idx, page) in all_requests {
            {
                let entry = self.shadow_maps[sm_idx as usize].get_entry_mut(&page);
                if entry.is_allocated() {
                    // Already allocated -- cached hit.
                    entry.last_used_frame = self.current_frame;
                    self.page_pool.touch_page(entry.physical_page);
                    self.stats.pages_cached += 1;
                    continue;
                }
            }

            // Try to allocate a new physical page.
            let phys_id = match self.page_pool.allocate_page() {
                Some(id) => id,
                None => {
                    // Pool exhausted. Evict LRU.
                    match self.page_pool.evict_lru() {
                        Some((evicted_id, old_virtual)) => {
                            // Invalidate the old mapping.
                            if let Some(old_page) = old_virtual {
                                if (old_page.shadow_map_id as usize) < self.shadow_maps.len() {
                                    let old_entry = self.shadow_maps
                                        [old_page.shadow_map_id as usize]
                                        .get_entry_mut(&old_page);
                                    old_entry.physical_page = PhysicalPageId::INVALID;
                                    old_entry.valid = false;
                                }
                            }
                            self.stats.pages_evicted += 1;

                            // Reuse the evicted page.
                            self.page_pool.lru_order.push_back(evicted_id.0);
                            evicted_id
                        }
                        None => continue, // Cannot allocate at all.
                    }
                }
            };

            // Map the physical page.
            if let Some(phys) = self.page_pool.get_page_mut(phys_id) {
                phys.mapped_to = Some(page);
                phys.clear();
            }

            let entry = self.shadow_maps[sm_idx as usize].get_entry_mut(&page);
            entry.physical_page = phys_id;
            entry.valid = false; // Will be valid after rendering.
            entry.last_used_frame = self.current_frame;
            self.stats.pages_allocated += 1;
        }
    }

    /// Evict pages that haven't been used in `max_cache_age` frames.
    pub fn evict_stale_pages(&mut self) {
        if !self.settings.enable_caching {
            return;
        }

        let max_age = self.settings.max_cache_age as u64;
        let frame = self.current_frame;

        let mut to_free: Vec<PhysicalPageId> = Vec::new();

        for vsm in &mut self.shadow_maps {
            for entry in &mut vsm.page_table {
                if entry.is_allocated() && !entry.requested {
                    let age = frame.saturating_sub(entry.last_used_frame);
                    if age > max_age {
                        to_free.push(entry.physical_page);
                        entry.physical_page = PhysicalPageId::INVALID;
                        entry.valid = false;
                    }
                }
            }
        }

        for page_id in to_free {
            self.page_pool.free_page(page_id);
        }
    }

    /// Mark a page as rendered (its depth data is now valid).
    pub fn mark_page_rendered(&mut self, sm_idx: u32, page: &VirtualPageId) {
        if (sm_idx as usize) < self.shadow_maps.len() {
            let entry = self.shadow_maps[sm_idx as usize].get_entry_mut(page);
            entry.valid = true;
            entry.rendered = true;
            self.stats.pages_rendered += 1;
        }
    }

    /// Sample shadow at a world position from the specified shadow map.
    pub fn sample_shadow(
        &self,
        sm_idx: u32,
        world_pos: Vec3,
        world_normal: Vec3,
    ) -> f32 {
        if (sm_idx as usize) >= self.shadow_maps.len() {
            return 1.0;
        }
        let vsm = &self.shadow_maps[sm_idx as usize];
        let (shadow, _) = vsm.sample_shadow(world_pos, world_normal, &self.page_pool);
        shadow
    }

    /// Get the number of active shadow maps.
    pub fn active_shadow_map_count(&self) -> u32 {
        self.shadow_maps.iter().filter(|s| s.active).count() as u32
    }

    /// Update stats.
    pub fn finalize_frame(&mut self) {
        self.stats.active_shadow_maps = self.active_shadow_map_count();
        self.stats.pool_occupancy = self.page_pool.occupancy();
    }

    /// Get shadow map by index.
    pub fn get_shadow_map(&self, index: u32) -> Option<&VirtualShadowMap> {
        self.shadow_maps.get(index as usize)
    }

    /// Get mutable shadow map by index.
    pub fn get_shadow_map_mut(&mut self, index: u32) -> Option<&mut VirtualShadowMap> {
        self.shadow_maps.get_mut(index as usize)
    }

    /// Reset everything.
    pub fn reset(&mut self) {
        self.page_pool.reset();
        self.shadow_maps.clear();
        self.current_frame = 0;
    }
}

// ---------------------------------------------------------------------------
// ClipmapShadow -- directional light clipmap using VSM
// ---------------------------------------------------------------------------

/// Directional light shadow using a clipmap of virtual shadow maps.
/// Each level covers a larger area with lower resolution.
pub struct ClipmapShadow {
    /// Shadow map indices in the VSM manager (one per clipmap level).
    pub levels: Vec<u32>,
    /// World-space center of the clipmap (typically camera position).
    pub center: Vec3,
    /// Light direction.
    pub light_direction: Vec3,
    /// Base extent (world units) of the finest level.
    pub base_extent: f32,
    /// Number of clipmap levels.
    pub level_count: u32,
}

impl ClipmapShadow {
    /// Create a new clipmap shadow with the given number of levels.
    pub fn new(
        manager: &mut VirtualShadowMapManager,
        level_count: u32,
        base_extent: f32,
    ) -> Self {
        let mut levels = Vec::with_capacity(level_count as usize);
        for _ in 0..level_count {
            let idx = manager.add_shadow_map(ShadowLightType::Directional);
            levels.push(idx);
        }

        Self {
            levels,
            center: Vec3::ZERO,
            light_direction: Vec3::new(0.0, -1.0, 0.0),
            base_extent,
            level_count,
        }
    }

    /// Update the clipmap matrices based on camera position and light direction.
    pub fn update(
        &mut self,
        manager: &mut VirtualShadowMapManager,
        camera_pos: Vec3,
        light_dir: Vec3,
    ) {
        self.center = camera_pos;
        self.light_direction = light_dir.normalize_or_zero();

        for (i, &sm_idx) in self.levels.iter().enumerate() {
            let extent = self.base_extent * (1 << i) as f32;
            let far = extent * 2.0;

            // Build orthographic projection centered on the camera.
            let view = Mat4::look_at_rh(
                camera_pos - self.light_direction * far * 0.5,
                camera_pos,
                Vec3::Y,
            );
            let proj = Mat4::orthographic_rh(
                -extent, extent,
                -extent, extent,
                0.01, far,
            );
            let view_proj = proj * view;

            if let Some(vsm) = manager.get_shadow_map_mut(sm_idx) {
                vsm.set_light_matrix(view_proj);
                vsm.active = true;
            }
        }
    }

    /// Sample the shadow at a world position, choosing the appropriate
    /// clipmap level based on distance.
    pub fn sample_shadow(
        &self,
        manager: &VirtualShadowMapManager,
        world_pos: Vec3,
        world_normal: Vec3,
    ) -> f32 {
        let dist = (world_pos - self.center).length();

        // Find the finest level that covers this distance.
        for (i, &sm_idx) in self.levels.iter().enumerate() {
            let extent = self.base_extent * (1 << i) as f32;
            if dist <= extent * 1.2 {
                return manager.sample_shadow(sm_idx, world_pos, world_normal);
            }
        }

        // Beyond all levels: fully lit.
        1.0
    }

    /// Extent at a given level.
    pub fn level_extent(&self, level: u32) -> f32 {
        self.base_extent * (1 << level) as f32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn virtual_page_id_pack_unpack() {
        let page = VirtualPageId::new(3, 2, 100, 200);
        let packed = page.pack();
        let unpacked = VirtualPageId::unpack(packed);
        assert_eq!(page, unpacked);
    }

    #[test]
    fn physical_page_pool_alloc_free() {
        let mut pool = PhysicalPagePool::new(4, 64);
        assert_eq!(pool.free_count(), 4);

        let p1 = pool.allocate_page().unwrap();
        let p2 = pool.allocate_page().unwrap();
        assert_eq!(pool.allocated_count, 2);
        assert_eq!(pool.free_count(), 2);

        pool.free_page(p1);
        assert_eq!(pool.allocated_count, 1);
        assert_eq!(pool.free_count(), 3);

        pool.free_page(p2);
        assert_eq!(pool.free_count(), 4);
    }

    #[test]
    fn physical_page_pool_lru_eviction() {
        let mut pool = PhysicalPagePool::new(2, 64);
        let p1 = pool.allocate_page().unwrap();
        let p2 = pool.allocate_page().unwrap();

        // Pool full.
        assert!(pool.allocate_page().is_none());

        // Evict LRU (should be p1).
        let (evicted, _) = pool.evict_lru().unwrap();
        assert_eq!(evicted, p1);
    }

    #[test]
    fn physical_page_depth_write_read() {
        let mut page = PhysicalPage::new(4);
        page.write_depth(1, 2, 0.5);
        assert!((page.read_depth(1, 2) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn physical_page_bilinear_sample() {
        let mut page = PhysicalPage::new(4);
        // Write constant depth.
        for y in 0..4 {
            for x in 0..4 {
                page.write_depth(x, y, 0.3);
            }
        }
        page.data_valid = true;

        let d = page.sample_depth_bilinear(0.5, 0.5);
        assert!((d - 0.3).abs() < 1e-4);
    }

    #[test]
    fn virtual_shadow_map_creation() {
        let vsm = VirtualShadowMap::new(0, 1024, 128, ShadowLightType::Directional);
        assert_eq!(vsm.pages_per_axis, 8);
        assert!(vsm.mip_count >= 1);
        assert!(vsm.total_page_entries() > 0);
    }

    #[test]
    fn vsm_page_request_and_collect() {
        let mut vsm = VirtualShadowMap::new(0, 512, 128, ShadowLightType::Spot);
        vsm.begin_frame(1);

        let page = VirtualPageId::new(0, 0, 1, 2);
        vsm.request_page(&page);

        let requested = vsm.collect_requested_pages();
        assert!(requested.contains(&page));
    }

    #[test]
    fn vsm_manager_workflow() {
        let settings = VirtualShadowMapSettings {
            page_size: 64,
            total_pages: 16,
            directional_resolution: 256,
            local_resolution: 256,
            ..Default::default()
        };

        let mut mgr = VirtualShadowMapManager::new(settings);
        let sm_idx = mgr.add_shadow_map(ShadowLightType::Spot);

        mgr.begin_frame();

        // Request some pages.
        if let Some(vsm) = mgr.get_shadow_map_mut(sm_idx) {
            vsm.active = true;
            vsm.request_page(&VirtualPageId::new(sm_idx, 0, 0, 0));
            vsm.request_page(&VirtualPageId::new(sm_idx, 0, 1, 0));
        }

        mgr.allocate_requested_pages();
        assert!(mgr.stats.pages_allocated >= 2);

        // Mark rendered.
        mgr.mark_page_rendered(sm_idx, &VirtualPageId::new(sm_idx, 0, 0, 0));
        mgr.mark_page_rendered(sm_idx, &VirtualPageId::new(sm_idx, 0, 1, 0));
        assert_eq!(mgr.stats.pages_rendered, 2);

        mgr.finalize_frame();
        assert!(mgr.stats.pool_occupancy > 0.0);
    }

    #[test]
    fn vsm_settings_presets() {
        let low = VirtualShadowMapSettings::low_quality();
        let high = VirtualShadowMapSettings::high_quality();
        assert!(low.total_pages < high.total_pages);
        assert!(low.directional_resolution < high.directional_resolution);
    }

    #[test]
    fn mip_count_calculation() {
        assert_eq!(compute_mip_count(1), 1);
        assert_eq!(compute_mip_count(2), 2);
        assert_eq!(compute_mip_count(4), 3);
        assert_eq!(compute_mip_count(128), 8);
    }

    #[test]
    fn pool_occupancy() {
        let mut pool = PhysicalPagePool::new(10, 32);
        assert!((pool.occupancy() - 0.0).abs() < 1e-6);

        pool.allocate_page();
        pool.allocate_page();
        assert!((pool.occupancy() - 0.2).abs() < 1e-6);
    }
}
