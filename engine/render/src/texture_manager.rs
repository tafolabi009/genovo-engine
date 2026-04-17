// engine/render/src/texture_manager.rs
//
// Texture lifecycle management: create/destroy GPU textures, format conversion,
// mipmap generation, texture pool, and memory tracking. Operates on CPU-side
// pixel data; the actual GPU upload is abstracted behind handles.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Texture formats
// ---------------------------------------------------------------------------

/// Pixel format for texture data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TexFormat {
    R8Unorm,
    Rg8Unorm,
    Rgba8Unorm,
    Rgba8Srgb,
    R16Float,
    Rg16Float,
    Rgba16Float,
    R32Float,
    Rg32Float,
    Rgba32Float,
    Bgra8Unorm,
    Bgra8Srgb,
    Depth16,
    Depth24Stencil8,
    Depth32Float,
    Bc1Unorm,
    Bc3Unorm,
    Bc5Unorm,
    Bc7Unorm,
}

impl TexFormat {
    /// Bytes per pixel (for uncompressed formats).
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            Self::R8Unorm => 1,
            Self::Rg8Unorm => 2,
            Self::Rgba8Unorm | Self::Rgba8Srgb | Self::Bgra8Unorm | Self::Bgra8Srgb => 4,
            Self::R16Float => 2,
            Self::Rg16Float => 4,
            Self::Rgba16Float => 8,
            Self::R32Float => 4,
            Self::Rg32Float => 8,
            Self::Rgba32Float => 16,
            Self::Depth16 => 2,
            Self::Depth24Stencil8 => 4,
            Self::Depth32Float => 4,
            Self::Bc1Unorm => 0, // block-compressed
            Self::Bc3Unorm => 0,
            Self::Bc5Unorm => 0,
            Self::Bc7Unorm => 0,
        }
    }

    /// Block size for compressed formats (4x4 blocks).
    pub fn block_bytes(&self) -> Option<usize> {
        match self {
            Self::Bc1Unorm => Some(8),
            Self::Bc3Unorm => Some(16),
            Self::Bc5Unorm => Some(16),
            Self::Bc7Unorm => Some(16),
            _ => None,
        }
    }

    pub fn is_compressed(&self) -> bool {
        self.block_bytes().is_some()
    }

    pub fn is_depth(&self) -> bool {
        matches!(self, Self::Depth16 | Self::Depth24Stencil8 | Self::Depth32Float)
    }

    pub fn is_srgb(&self) -> bool {
        matches!(self, Self::Rgba8Srgb | Self::Bgra8Srgb)
    }

    pub fn channel_count(&self) -> u32 {
        match self {
            Self::R8Unorm | Self::R16Float | Self::R32Float => 1,
            Self::Rg8Unorm | Self::Rg16Float | Self::Rg32Float => 2,
            Self::Rgba8Unorm | Self::Rgba8Srgb | Self::Rgba16Float | Self::Rgba32Float => 4,
            Self::Bgra8Unorm | Self::Bgra8Srgb => 4,
            _ => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Texture dimension & usage
// ---------------------------------------------------------------------------

/// Dimension of a texture resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TexDimension {
    Tex1D,
    Tex2D,
    Tex3D,
    TexCube,
    Tex2DArray,
}

/// Texture usage bit flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TexUsage(u32);

impl TexUsage {
    pub const SAMPLED: Self = Self(0x01);
    pub const STORAGE: Self = Self(0x02);
    pub const RENDER_TARGET: Self = Self(0x04);
    pub const DEPTH_STENCIL: Self = Self(0x08);
    pub const TRANSFER_SRC: Self = Self(0x10);
    pub const TRANSFER_DST: Self = Self(0x20);

    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl std::ops::BitOr for TexUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

// ---------------------------------------------------------------------------
// Texture handle
// ---------------------------------------------------------------------------

/// A strongly-typed handle to a managed texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId {
    pub index: u32,
    pub generation: u32,
}

impl TextureId {
    pub const INVALID: Self = Self { index: u32::MAX, generation: 0 };
}

// ---------------------------------------------------------------------------
// Texture descriptor
// ---------------------------------------------------------------------------

/// Describes how to create a new texture.
#[derive(Debug, Clone)]
pub struct TextureDescriptor {
    pub label: String,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub format: TexFormat,
    pub dimension: TexDimension,
    pub usage: TexUsage,
    pub initial_data: Option<Vec<u8>>,
}

impl TextureDescriptor {
    pub fn new_2d(label: &str, width: u32, height: u32, format: TexFormat) -> Self {
        let mip_levels = Self::compute_mip_count(width, height);
        Self {
            label: label.to_string(),
            width,
            height,
            depth: 1,
            mip_levels,
            array_layers: 1,
            format,
            dimension: TexDimension::Tex2D,
            usage: TexUsage::SAMPLED | TexUsage::TRANSFER_DST,
            initial_data: None,
        }
    }

    pub fn new_cubemap(label: &str, size: u32, format: TexFormat) -> Self {
        Self {
            label: label.to_string(),
            width: size,
            height: size,
            depth: 1,
            mip_levels: Self::compute_mip_count(size, size),
            array_layers: 6,
            format,
            dimension: TexDimension::TexCube,
            usage: TexUsage::SAMPLED | TexUsage::TRANSFER_DST,
            initial_data: None,
        }
    }

    pub fn new_render_target(label: &str, width: u32, height: u32, format: TexFormat) -> Self {
        Self {
            label: label.to_string(),
            width,
            height,
            depth: 1,
            mip_levels: 1,
            array_layers: 1,
            format,
            dimension: TexDimension::Tex2D,
            usage: TexUsage::SAMPLED | TexUsage::RENDER_TARGET,
            initial_data: None,
        }
    }

    pub fn with_data(mut self, data: Vec<u8>) -> Self {
        self.initial_data = Some(data);
        self
    }

    pub fn with_mip_levels(mut self, levels: u32) -> Self {
        self.mip_levels = levels;
        self
    }

    pub fn with_usage(mut self, usage: TexUsage) -> Self {
        self.usage = usage;
        self
    }

    /// Compute the number of mip levels for the given dimensions.
    pub fn compute_mip_count(width: u32, height: u32) -> u32 {
        let max_dim = width.max(height);
        if max_dim == 0 {
            return 1;
        }
        (max_dim as f32).log2().floor() as u32 + 1
    }

    /// Total memory for all mip levels of the base face.
    pub fn total_memory_bytes(&self) -> usize {
        let mut total = 0usize;
        let bpp = self.format.bytes_per_pixel();
        let block = self.format.block_bytes();

        for mip in 0..self.mip_levels {
            let w = (self.width >> mip).max(1);
            let h = (self.height >> mip).max(1);
            let d = (self.depth >> mip).max(1);

            let mip_size = if let Some(block_bytes) = block {
                let bw = (w + 3) / 4;
                let bh = (h + 3) / 4;
                (bw * bh * d) as usize * block_bytes
            } else {
                (w * h * d) as usize * bpp
            };
            total += mip_size;
        }
        total * self.array_layers as usize
    }
}

// ---------------------------------------------------------------------------
// Internal texture record
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct TextureRecord {
    desc: TextureDescriptor,
    generation: u32,
    alive: bool,
    ref_count: u32,
    memory_bytes: usize,
    mip_data: Vec<Vec<u8>>,
    created_frame: u64,
}

// ---------------------------------------------------------------------------
// Mipmap generation
// ---------------------------------------------------------------------------

/// Generate mipmaps for RGBA8 data by box-filtering.
pub fn generate_mipmaps_rgba8(
    base_width: u32,
    base_height: u32,
    base_data: &[u8],
) -> Vec<Vec<u8>> {
    let mip_count = TextureDescriptor::compute_mip_count(base_width, base_height);
    let mut mips = Vec::with_capacity(mip_count as usize);
    mips.push(base_data.to_vec());

    let mut prev_w = base_width;
    let mut prev_h = base_height;

    for _ in 1..mip_count {
        let cur_w = (prev_w / 2).max(1);
        let cur_h = (prev_h / 2).max(1);
        let prev = mips.last().unwrap();
        let mut cur = vec![0u8; (cur_w * cur_h * 4) as usize];

        for y in 0..cur_h {
            for x in 0..cur_w {
                let sx = (x * 2).min(prev_w - 1);
                let sy = (y * 2).min(prev_h - 1);
                let sx1 = (sx + 1).min(prev_w - 1);
                let sy1 = (sy + 1).min(prev_h - 1);

                for c in 0..4 {
                    let i00 = (sy * prev_w + sx) as usize * 4 + c;
                    let i10 = (sy * prev_w + sx1) as usize * 4 + c;
                    let i01 = (sy1 * prev_w + sx) as usize * 4 + c;
                    let i11 = (sy1 * prev_w + sx1) as usize * 4 + c;

                    let avg = (prev[i00] as u32 + prev[i10] as u32
                        + prev[i01] as u32 + prev[i11] as u32 + 2) / 4;
                    cur[(y * cur_w + x) as usize * 4 + c] = avg as u8;
                }
            }
        }

        mips.push(cur);
        prev_w = cur_w;
        prev_h = cur_h;
    }

    mips
}

/// Generate mipmaps for single-channel R8 data.
pub fn generate_mipmaps_r8(base_width: u32, base_height: u32, base_data: &[u8]) -> Vec<Vec<u8>> {
    let mip_count = TextureDescriptor::compute_mip_count(base_width, base_height);
    let mut mips = Vec::with_capacity(mip_count as usize);
    mips.push(base_data.to_vec());

    let mut prev_w = base_width;
    let mut prev_h = base_height;

    for _ in 1..mip_count {
        let cur_w = (prev_w / 2).max(1);
        let cur_h = (prev_h / 2).max(1);
        let prev = mips.last().unwrap();
        let mut cur = vec![0u8; (cur_w * cur_h) as usize];

        for y in 0..cur_h {
            for x in 0..cur_w {
                let sx = (x * 2).min(prev_w - 1);
                let sy = (y * 2).min(prev_h - 1);
                let sx1 = (sx + 1).min(prev_w - 1);
                let sy1 = (sy + 1).min(prev_h - 1);

                let v00 = prev[(sy * prev_w + sx) as usize] as u32;
                let v10 = prev[(sy * prev_w + sx1) as usize] as u32;
                let v01 = prev[(sy1 * prev_w + sx) as usize] as u32;
                let v11 = prev[(sy1 * prev_w + sx1) as usize] as u32;

                cur[(y * cur_w + x) as usize] = ((v00 + v10 + v01 + v11 + 2) / 4) as u8;
            }
        }

        mips.push(cur);
        prev_w = cur_w;
        prev_h = cur_h;
    }

    mips
}

/// Generate mipmaps for Rgba32Float (f32x4) data.
pub fn generate_mipmaps_rgba32f(
    base_width: u32,
    base_height: u32,
    base_data: &[f32],
) -> Vec<Vec<f32>> {
    let mip_count = TextureDescriptor::compute_mip_count(base_width, base_height);
    let mut mips: Vec<Vec<f32>> = Vec::with_capacity(mip_count as usize);
    mips.push(base_data.to_vec());

    let mut prev_w = base_width;
    let mut prev_h = base_height;

    for _ in 1..mip_count {
        let cur_w = (prev_w / 2).max(1);
        let cur_h = (prev_h / 2).max(1);
        let prev = mips.last().unwrap();
        let mut cur = vec![0.0f32; (cur_w * cur_h * 4) as usize];

        for y in 0..cur_h {
            for x in 0..cur_w {
                let sx = (x * 2).min(prev_w - 1);
                let sy = (y * 2).min(prev_h - 1);
                let sx1 = (sx + 1).min(prev_w - 1);
                let sy1 = (sy + 1).min(prev_h - 1);

                for c in 0..4 {
                    let i00 = (sy * prev_w + sx) as usize * 4 + c;
                    let i10 = (sy * prev_w + sx1) as usize * 4 + c;
                    let i01 = (sy1 * prev_w + sx) as usize * 4 + c;
                    let i11 = (sy1 * prev_w + sx1) as usize * 4 + c;

                    cur[(y * cur_w + x) as usize * 4 + c] =
                        (prev[i00] + prev[i10] + prev[i01] + prev[i11]) * 0.25;
                }
            }
        }

        mips.push(cur);
        prev_w = cur_w;
        prev_h = cur_h;
    }

    mips
}

// ---------------------------------------------------------------------------
// Format conversion
// ---------------------------------------------------------------------------

/// Convert sRGB to linear.
#[inline]
pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert linear to sRGB.
#[inline]
pub fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Convert RGBA8 data from sRGB to linear (alpha unchanged).
pub fn convert_srgb_to_linear_rgba8(data: &mut [u8]) {
    for chunk in data.chunks_exact_mut(4) {
        chunk[0] = (srgb_to_linear(chunk[0] as f32 / 255.0) * 255.0) as u8;
        chunk[1] = (srgb_to_linear(chunk[1] as f32 / 255.0) * 255.0) as u8;
        chunk[2] = (srgb_to_linear(chunk[2] as f32 / 255.0) * 255.0) as u8;
        // alpha unchanged
    }
}

/// Convert RGBA8 data from linear to sRGB (alpha unchanged).
pub fn convert_linear_to_srgb_rgba8(data: &mut [u8]) {
    for chunk in data.chunks_exact_mut(4) {
        chunk[0] = (linear_to_srgb(chunk[0] as f32 / 255.0) * 255.0).min(255.0) as u8;
        chunk[1] = (linear_to_srgb(chunk[1] as f32 / 255.0) * 255.0).min(255.0) as u8;
        chunk[2] = (linear_to_srgb(chunk[2] as f32 / 255.0) * 255.0).min(255.0) as u8;
    }
}

/// Convert BGRA8 to RGBA8 in-place.
pub fn convert_bgra8_to_rgba8(data: &mut [u8]) {
    for chunk in data.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }
}

/// Convert RGBA8 to RG8 (take R and G channels).
pub fn convert_rgba8_to_rg8(data: &[u8]) -> Vec<u8> {
    let pixel_count = data.len() / 4;
    let mut out = Vec::with_capacity(pixel_count * 2);
    for i in 0..pixel_count {
        out.push(data[i * 4]);
        out.push(data[i * 4 + 1]);
    }
    out
}

/// Convert RGBA8 to R8 (take R channel).
pub fn convert_rgba8_to_r8(data: &[u8]) -> Vec<u8> {
    let pixel_count = data.len() / 4;
    let mut out = Vec::with_capacity(pixel_count);
    for i in 0..pixel_count {
        out.push(data[i * 4]);
    }
    out
}

/// Convert R8 to RGBA8 (replicate to RGB, alpha=255).
pub fn convert_r8_to_rgba8(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for &v in data {
        out.push(v);
        out.push(v);
        out.push(v);
        out.push(255);
    }
    out
}

/// Convert RGBA8 to RGBA32F.
pub fn convert_rgba8_to_rgba32f(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    for &b in data {
        out.push(b as f32 / 255.0);
    }
    out
}

/// Convert RGBA32F to RGBA8 (clamped).
pub fn convert_rgba32f_to_rgba8(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    for &v in data {
        out.push((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
    }
    out
}

// ---------------------------------------------------------------------------
// Memory tracking
// ---------------------------------------------------------------------------

/// GPU memory budget and tracking.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    pub budget_bytes: usize,
    pub used_bytes: usize,
    pub peak_bytes: usize,
    pub allocation_count: u32,
}

impl MemoryBudget {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            used_bytes: 0,
            peak_bytes: 0,
            allocation_count: 0,
        }
    }

    pub fn available(&self) -> usize {
        self.budget_bytes.saturating_sub(self.used_bytes)
    }

    pub fn usage_ratio(&self) -> f32 {
        if self.budget_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f32 / self.budget_bytes as f32
    }

    fn allocate(&mut self, bytes: usize) -> bool {
        if self.used_bytes + bytes > self.budget_bytes {
            return false;
        }
        self.used_bytes += bytes;
        self.peak_bytes = self.peak_bytes.max(self.used_bytes);
        self.allocation_count += 1;
        true
    }

    fn deallocate(&mut self, bytes: usize) {
        self.used_bytes = self.used_bytes.saturating_sub(bytes);
        self.allocation_count = self.allocation_count.saturating_sub(1);
    }
}

// ---------------------------------------------------------------------------
// Texture pool
// ---------------------------------------------------------------------------

/// Reusable texture from the pool.
#[derive(Debug)]
struct PooledTexture {
    id: TextureId,
    desc: TextureDescriptor,
    memory_bytes: usize,
    last_used_frame: u64,
    in_use: bool,
}

/// A pool of textures that can be reused to avoid allocation overhead.
pub struct TexturePool {
    entries: Vec<PooledTexture>,
    max_pool_size: usize,
    current_frame: u64,
}

impl TexturePool {
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_pool_size,
            current_frame: 0,
        }
    }

    /// Try to acquire a texture matching the descriptor from the pool.
    pub fn acquire(
        &mut self,
        width: u32,
        height: u32,
        format: TexFormat,
        usage: TexUsage,
    ) -> Option<TextureId> {
        for entry in self.entries.iter_mut() {
            if !entry.in_use
                && entry.desc.width == width
                && entry.desc.height == height
                && entry.desc.format == format
                && entry.desc.usage.0 & usage.0 == usage.0
            {
                entry.in_use = true;
                entry.last_used_frame = self.current_frame;
                return Some(entry.id);
            }
        }
        None
    }

    /// Return a texture to the pool.
    pub fn release(&mut self, id: TextureId) {
        for entry in self.entries.iter_mut() {
            if entry.id == id {
                entry.in_use = false;
                return;
            }
        }
    }

    /// Register a new texture in the pool.
    pub fn register(&mut self, id: TextureId, desc: TextureDescriptor, memory_bytes: usize) {
        if self.entries.len() >= self.max_pool_size {
            return;
        }
        self.entries.push(PooledTexture {
            id,
            desc,
            memory_bytes,
            last_used_frame: self.current_frame,
            in_use: true,
        });
    }

    /// Evict textures not used for `max_age_frames`.
    pub fn evict(&mut self, max_age_frames: u64) -> Vec<TextureId> {
        let threshold = self.current_frame.saturating_sub(max_age_frames);
        let mut evicted = Vec::new();
        self.entries.retain(|e| {
            if !e.in_use && e.last_used_frame < threshold {
                evicted.push(e.id);
                false
            } else {
                true
            }
        });
        evicted
    }

    pub fn advance_frame(&mut self) {
        self.current_frame += 1;
    }

    pub fn pool_size(&self) -> usize {
        self.entries.len()
    }

    pub fn in_use_count(&self) -> usize {
        self.entries.iter().filter(|e| e.in_use).count()
    }

    pub fn total_memory(&self) -> usize {
        self.entries.iter().map(|e| e.memory_bytes).sum()
    }
}

// ---------------------------------------------------------------------------
// TextureManager
// ---------------------------------------------------------------------------

/// Manages the lifecycle of all textures in the engine.
pub struct TextureManager {
    records: Vec<TextureRecord>,
    free_indices: Vec<u32>,
    name_to_id: HashMap<String, TextureId>,
    memory: MemoryBudget,
    pool: TexturePool,
    current_frame: u64,
    stats: TextureManagerStats,
}

/// Diagnostic statistics.
#[derive(Debug, Clone, Default)]
pub struct TextureManagerStats {
    pub total_textures: u32,
    pub total_memory_bytes: usize,
    pub peak_memory_bytes: usize,
    pub textures_created: u64,
    pub textures_destroyed: u64,
    pub pool_hits: u64,
    pub pool_misses: u64,
    pub format_conversions: u64,
    pub mipmap_generations: u64,
}

impl TextureManager {
    pub fn new(memory_budget_bytes: usize) -> Self {
        Self {
            records: Vec::new(),
            free_indices: Vec::new(),
            name_to_id: HashMap::new(),
            memory: MemoryBudget::new(memory_budget_bytes),
            pool: TexturePool::new(128),
            current_frame: 0,
            stats: TextureManagerStats::default(),
        }
    }

    pub fn stats(&self) -> &TextureManagerStats {
        &self.stats
    }

    pub fn memory(&self) -> &MemoryBudget {
        &self.memory
    }

    /// Create a new texture from a descriptor.
    pub fn create(&mut self, desc: TextureDescriptor) -> Result<TextureId, TextureError> {
        let mem_bytes = desc.total_memory_bytes();

        if !self.memory.allocate(mem_bytes) {
            return Err(TextureError::OutOfMemory {
                requested: mem_bytes,
                available: self.memory.available(),
            });
        }

        let (index, generation) = if let Some(free_idx) = self.free_indices.pop() {
            let gen = self.records[free_idx as usize].generation + 1;
            self.records[free_idx as usize] = TextureRecord {
                desc: desc.clone(),
                generation: gen,
                alive: true,
                ref_count: 1,
                memory_bytes: mem_bytes,
                mip_data: Vec::new(),
                created_frame: self.current_frame,
            };
            (free_idx, gen)
        } else {
            let idx = self.records.len() as u32;
            self.records.push(TextureRecord {
                desc: desc.clone(),
                generation: 0,
                alive: true,
                ref_count: 1,
                memory_bytes: mem_bytes,
                mip_data: Vec::new(),
                created_frame: self.current_frame,
            });
            (idx, 0)
        };

        let id = TextureId { index, generation };

        if !desc.label.is_empty() {
            self.name_to_id.insert(desc.label.clone(), id);
        }

        self.stats.total_textures += 1;
        self.stats.total_memory_bytes += mem_bytes;
        self.stats.peak_memory_bytes = self.stats.peak_memory_bytes.max(self.stats.total_memory_bytes);
        self.stats.textures_created += 1;

        Ok(id)
    }

    /// Destroy a texture, freeing its memory.
    pub fn destroy(&mut self, id: TextureId) -> Result<(), TextureError> {
        let record = self.get_record_mut(id)?;
        record.ref_count = record.ref_count.saturating_sub(1);
        if record.ref_count > 0 {
            return Ok(());
        }

        let mem = record.memory_bytes;
        let label = record.desc.label.clone();
        record.alive = false;
        record.mip_data.clear();

        self.memory.deallocate(mem);
        self.free_indices.push(id.index);

        if !label.is_empty() {
            self.name_to_id.remove(&label);
        }

        self.stats.total_textures -= 1;
        self.stats.total_memory_bytes = self.stats.total_memory_bytes.saturating_sub(mem);
        self.stats.textures_destroyed += 1;

        Ok(())
    }

    /// Add a reference to a texture (increment ref count).
    pub fn add_ref(&mut self, id: TextureId) -> Result<(), TextureError> {
        let record = self.get_record_mut(id)?;
        record.ref_count += 1;
        Ok(())
    }

    /// Look up a texture by name.
    pub fn find_by_name(&self, name: &str) -> Option<TextureId> {
        self.name_to_id.get(name).copied()
    }

    /// Get the descriptor of a texture.
    pub fn descriptor(&self, id: TextureId) -> Result<&TextureDescriptor, TextureError> {
        let record = self.get_record(id)?;
        Ok(&record.desc)
    }

    /// Upload mipmap data for a texture.
    pub fn upload_mips(&mut self, id: TextureId, mips: Vec<Vec<u8>>) -> Result<(), TextureError> {
        let record = self.get_record_mut(id)?;
        record.mip_data = mips;
        Ok(())
    }

    /// Generate and store mipmaps for an RGBA8 texture.
    pub fn generate_mipmaps(&mut self, id: TextureId) -> Result<(), TextureError> {
        let record = self.get_record(id)?;
        if record.desc.format != TexFormat::Rgba8Unorm && record.desc.format != TexFormat::Rgba8Srgb {
            return Err(TextureError::InvalidFormat(
                "Mipmap generation only supports RGBA8".to_string(),
            ));
        }
        let base = record.desc.initial_data.as_ref().ok_or_else(|| {
            TextureError::InvalidData("No initial data to generate mipmaps from".to_string())
        })?;
        let w = record.desc.width;
        let h = record.desc.height;
        let mips = generate_mipmaps_rgba8(w, h, base);
        self.stats.mipmap_generations += 1;

        let record = self.get_record_mut(id)?;
        record.mip_data = mips;
        Ok(())
    }

    /// Try to acquire a pooled texture, or create a new one.
    pub fn acquire_pooled(
        &mut self,
        label: &str,
        width: u32,
        height: u32,
        format: TexFormat,
        usage: TexUsage,
    ) -> Result<TextureId, TextureError> {
        if let Some(id) = self.pool.acquire(width, height, format, usage) {
            self.stats.pool_hits += 1;
            return Ok(id);
        }
        self.stats.pool_misses += 1;

        let desc = TextureDescriptor {
            label: label.to_string(),
            width,
            height,
            depth: 1,
            mip_levels: 1,
            array_layers: 1,
            format,
            dimension: TexDimension::Tex2D,
            usage,
            initial_data: None,
        };
        let id = self.create(desc.clone())?;
        let mem = self.get_record(id)?.memory_bytes;
        self.pool.register(id, desc, mem);
        Ok(id)
    }

    /// Release a pooled texture back to the pool.
    pub fn release_pooled(&mut self, id: TextureId) {
        self.pool.release(id);
    }

    /// Advance frame counter and evict old pooled textures.
    pub fn begin_frame(&mut self) {
        self.current_frame += 1;
        self.pool.advance_frame();
        let evicted = self.pool.evict(60);
        for id in evicted {
            let _ = self.destroy(id);
        }
    }

    fn get_record(&self, id: TextureId) -> Result<&TextureRecord, TextureError> {
        let record = self.records.get(id.index as usize).ok_or(TextureError::InvalidHandle)?;
        if !record.alive || record.generation != id.generation {
            return Err(TextureError::InvalidHandle);
        }
        Ok(record)
    }

    fn get_record_mut(&mut self, id: TextureId) -> Result<&mut TextureRecord, TextureError> {
        let record = self.records.get_mut(id.index as usize).ok_or(TextureError::InvalidHandle)?;
        if !record.alive || record.generation != id.generation {
            return Err(TextureError::InvalidHandle);
        }
        Ok(record)
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from the texture management system.
#[derive(Debug)]
pub enum TextureError {
    InvalidHandle,
    OutOfMemory { requested: usize, available: usize },
    InvalidFormat(String),
    InvalidData(String),
    InvalidDimensions { width: u32, height: u32 },
}

impl std::fmt::Display for TextureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidHandle => write!(f, "Invalid texture handle"),
            Self::OutOfMemory { requested, available } => {
                write!(f, "Out of texture memory: requested {requested} bytes, {available} available")
            }
            Self::InvalidFormat(msg) => write!(f, "Invalid texture format: {msg}"),
            Self::InvalidData(msg) => write!(f, "Invalid texture data: {msg}"),
            Self::InvalidDimensions { width, height } => {
                write!(f, "Invalid texture dimensions: {width}x{height}")
            }
        }
    }
}

impl std::error::Error for TextureError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mip_count() {
        assert_eq!(TextureDescriptor::compute_mip_count(256, 256), 9);
        assert_eq!(TextureDescriptor::compute_mip_count(512, 256), 10);
        assert_eq!(TextureDescriptor::compute_mip_count(1, 1), 1);
    }

    #[test]
    fn test_generate_mipmaps_rgba8() {
        let w = 4u32;
        let h = 4u32;
        let data = vec![128u8; (w * h * 4) as usize];
        let mips = generate_mipmaps_rgba8(w, h, &data);
        assert_eq!(mips.len(), 3); // 4x4, 2x2, 1x1
        assert_eq!(mips[1].len(), 2 * 2 * 4);
        assert_eq!(mips[2].len(), 1 * 1 * 4);
        assert_eq!(mips[2][0], 128);
    }

    #[test]
    fn test_srgb_roundtrip() {
        for v in [0.0f32, 0.5, 1.0] {
            let linear = srgb_to_linear(v);
            let back = linear_to_srgb(linear);
            assert!((v - back).abs() < 0.01, "roundtrip failed for {v}");
        }
    }

    #[test]
    fn test_bgra_to_rgba() {
        let mut data = vec![10, 20, 30, 255];
        convert_bgra8_to_rgba8(&mut data);
        assert_eq!(data, vec![30, 20, 10, 255]);
    }

    #[test]
    fn test_texture_manager_lifecycle() {
        let mut mgr = TextureManager::new(64 * 1024 * 1024);
        let desc = TextureDescriptor::new_2d("test", 64, 64, TexFormat::Rgba8Unorm);
        let id = mgr.create(desc).unwrap();
        assert_eq!(mgr.stats().total_textures, 1);
        assert!(mgr.find_by_name("test").is_some());
        mgr.destroy(id).unwrap();
        assert_eq!(mgr.stats().total_textures, 0);
    }

    #[test]
    fn test_oom() {
        let mut mgr = TextureManager::new(1024); // very small budget
        let desc = TextureDescriptor::new_2d("big", 1024, 1024, TexFormat::Rgba8Unorm);
        assert!(mgr.create(desc).is_err());
    }

    #[test]
    fn test_texture_pool() {
        let mut pool = TexturePool::new(64);
        let id = TextureId { index: 0, generation: 0 };
        let desc = TextureDescriptor::new_2d("rt", 128, 128, TexFormat::Rgba8Unorm);
        pool.register(id, desc, 128 * 128 * 4);
        pool.release(id);

        let acquired = pool.acquire(128, 128, TexFormat::Rgba8Unorm, TexUsage::SAMPLED);
        assert_eq!(acquired, Some(id));
    }
}
