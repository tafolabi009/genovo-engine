// engine/render/src/uniform_buffer.rs
//
// Uniform buffer management: dynamic uniform buffer with offset allocation,
// per-frame reset, alignment handling, bind group caching. Provides a
// sub-allocator that hands out aligned regions within a large backing buffer.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Alignment utilities
// ---------------------------------------------------------------------------

/// Round `value` up to the nearest multiple of `alignment`.
#[inline]
pub fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two(), "alignment must be power of 2");
    (value + alignment - 1) & !(alignment - 1)
}

/// Round `value` down to the nearest multiple of `alignment`.
#[inline]
pub fn align_down(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two(), "alignment must be power of 2");
    value & !(alignment - 1)
}

/// Check if `value` is aligned to `alignment`.
#[inline]
pub fn is_aligned(value: usize, alignment: usize) -> bool {
    value & (alignment - 1) == 0
}

// ---------------------------------------------------------------------------
// Uniform buffer types
// ---------------------------------------------------------------------------

/// A handle to a sub-allocation within a uniform buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UniformAllocation {
    /// Byte offset into the backing buffer.
    pub offset: u32,
    /// Size of the allocation in bytes.
    pub size: u32,
    /// The frame this allocation belongs to.
    pub frame: u64,
    /// Which ring buffer index this belongs to.
    pub buffer_index: u32,
}

impl UniformAllocation {
    pub const INVALID: Self = Self {
        offset: u32::MAX,
        size: 0,
        frame: 0,
        buffer_index: 0,
    };

    pub fn is_valid(&self) -> bool {
        self.offset != u32::MAX
    }

    /// The dynamic offset to pass to a bind group for this allocation.
    pub fn dynamic_offset(&self) -> u32 {
        self.offset
    }
}

/// Configuration for the dynamic uniform buffer.
#[derive(Debug, Clone)]
pub struct DynamicUniformBufferConfig {
    /// Total size of the backing buffer in bytes.
    pub buffer_size: usize,
    /// Minimum alignment for uniform buffer offsets (typically 256 bytes).
    pub min_alignment: usize,
    /// Number of frames in flight (ring buffer count).
    pub frames_in_flight: u32,
    /// Whether to zero-fill the buffer on reset.
    pub zero_on_reset: bool,
}

impl Default for DynamicUniformBufferConfig {
    fn default() -> Self {
        Self {
            buffer_size: 4 * 1024 * 1024, // 4 MB
            min_alignment: 256,
            frames_in_flight: 2,
            zero_on_reset: false,
        }
    }
}

/// A single frame's region within the ring buffer.
#[derive(Debug)]
struct FrameRegion {
    start: usize,
    end: usize,
    write_offset: usize,
    allocation_count: u32,
    bytes_used: usize,
}

impl FrameRegion {
    fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            write_offset: start,
            allocation_count: 0,
            bytes_used: 0,
        }
    }

    fn available(&self) -> usize {
        self.end.saturating_sub(self.write_offset)
    }

    fn reset(&mut self) {
        self.write_offset = self.start;
        self.allocation_count = 0;
        self.bytes_used = 0;
    }

    fn usage_ratio(&self) -> f32 {
        let total = self.end - self.start;
        if total == 0 {
            return 0.0;
        }
        self.bytes_used as f32 / total as f32
    }
}

/// Statistics for the uniform buffer system.
#[derive(Debug, Clone, Default)]
pub struct UniformBufferStats {
    pub total_buffer_size: usize,
    pub current_frame_used: usize,
    pub current_frame_available: usize,
    pub current_frame_allocations: u32,
    pub peak_frame_usage: usize,
    pub total_allocations: u64,
    pub total_bytes_written: u64,
    pub alignment_waste_bytes: u64,
    pub bind_group_cache_hits: u64,
    pub bind_group_cache_misses: u64,
}

/// A dynamic uniform buffer with per-frame sub-allocation.
///
/// Uses a ring buffer approach: the backing buffer is divided into
/// `frames_in_flight` regions, and each frame writes into its own region.
/// At the start of each frame, the current region is reset.
pub struct DynamicUniformBuffer {
    data: Vec<u8>,
    config: DynamicUniformBufferConfig,
    regions: Vec<FrameRegion>,
    current_frame: u64,
    current_region: usize,
    stats: UniformBufferStats,
    peak_usage: usize,
}

impl DynamicUniformBuffer {
    pub fn new(config: DynamicUniformBufferConfig) -> Self {
        let frames = config.frames_in_flight as usize;
        let region_size = config.buffer_size / frames;
        let data = vec![0u8; config.buffer_size];

        let mut regions = Vec::with_capacity(frames);
        for i in 0..frames {
            let start = i * region_size;
            let end = if i == frames - 1 {
                config.buffer_size
            } else {
                (i + 1) * region_size
            };
            regions.push(FrameRegion::new(start, end));
        }

        Self {
            data,
            config,
            regions,
            current_frame: 0,
            current_region: 0,
            stats: UniformBufferStats::default(),
            peak_usage: 0,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(DynamicUniformBufferConfig::default())
    }

    /// Begin a new frame, resetting the current region.
    pub fn begin_frame(&mut self) {
        self.current_frame += 1;
        self.current_region = (self.current_frame as usize) % self.regions.len();

        let region = &mut self.regions[self.current_region];
        if self.config.zero_on_reset {
            let start = region.start;
            let end = region.end;
            self.data[start..end].fill(0);
        }
        region.reset();

        self.update_stats();
    }

    /// Allocate space for `size` bytes, returning the offset.
    pub fn allocate(&mut self, size: usize) -> Result<UniformAllocation, UniformBufferError> {
        if size == 0 {
            return Err(UniformBufferError::ZeroSize);
        }

        let alignment = self.config.min_alignment;
        let aligned_size = align_up(size, alignment);

        let region = &mut self.regions[self.current_region];
        let aligned_offset = align_up(region.write_offset, alignment);

        if aligned_offset + aligned_size > region.end {
            return Err(UniformBufferError::OutOfSpace {
                requested: aligned_size,
                available: region.available(),
            });
        }

        let waste = aligned_offset - region.write_offset;
        region.write_offset = aligned_offset + aligned_size;
        region.allocation_count += 1;
        region.bytes_used += aligned_size;

        self.stats.total_allocations += 1;
        self.stats.alignment_waste_bytes += waste as u64;

        Ok(UniformAllocation {
            offset: aligned_offset as u32,
            size: aligned_size as u32,
            frame: self.current_frame,
            buffer_index: self.current_region as u32,
        })
    }

    /// Allocate and write data in one step.
    pub fn allocate_and_write(&mut self, data: &[u8]) -> Result<UniformAllocation, UniformBufferError> {
        let alloc = self.allocate(data.len())?;
        self.write(alloc, data)?;
        Ok(alloc)
    }

    /// Write typed data (any `Copy` type) into the buffer.
    pub fn write_typed<T: Copy>(&mut self, value: &T) -> Result<UniformAllocation, UniformBufferError> {
        let size = std::mem::size_of::<T>();
        let alloc = self.allocate(size)?;
        let bytes = unsafe {
            std::slice::from_raw_parts(value as *const T as *const u8, size)
        };
        self.write(alloc, bytes)?;
        Ok(alloc)
    }

    /// Write raw bytes at the given allocation.
    pub fn write(&mut self, alloc: UniformAllocation, data: &[u8]) -> Result<(), UniformBufferError> {
        if !alloc.is_valid() {
            return Err(UniformBufferError::InvalidAllocation);
        }
        let offset = alloc.offset as usize;
        let size = data.len().min(alloc.size as usize);
        if offset + size > self.data.len() {
            return Err(UniformBufferError::OutOfBounds);
        }
        self.data[offset..offset + size].copy_from_slice(&data[..size]);
        self.stats.total_bytes_written += size as u64;
        Ok(())
    }

    /// Read back data at an allocation.
    pub fn read(&self, alloc: UniformAllocation) -> Result<&[u8], UniformBufferError> {
        if !alloc.is_valid() {
            return Err(UniformBufferError::InvalidAllocation);
        }
        let offset = alloc.offset as usize;
        let size = alloc.size as usize;
        if offset + size > self.data.len() {
            return Err(UniformBufferError::OutOfBounds);
        }
        Ok(&self.data[offset..offset + size])
    }

    /// Get the raw buffer data (for uploading to GPU).
    pub fn buffer_data(&self) -> &[u8] {
        &self.data
    }

    /// Get the current frame's region data only.
    pub fn current_region_data(&self) -> &[u8] {
        let region = &self.regions[self.current_region];
        &self.data[region.start..region.write_offset]
    }

    /// Get the current frame's region start offset.
    pub fn current_region_offset(&self) -> usize {
        self.regions[self.current_region].start
    }

    /// Get the current frame's region end (write cursor).
    pub fn current_region_used(&self) -> usize {
        self.regions[self.current_region].bytes_used
    }

    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    pub fn config(&self) -> &DynamicUniformBufferConfig {
        &self.config
    }

    pub fn stats(&self) -> &UniformBufferStats {
        &self.stats
    }

    pub fn total_size(&self) -> usize {
        self.data.len()
    }

    fn update_stats(&mut self) {
        let region = &self.regions[self.current_region];
        self.stats.total_buffer_size = self.data.len();
        self.stats.current_frame_used = region.bytes_used;
        self.stats.current_frame_available = region.available();
        self.stats.current_frame_allocations = region.allocation_count;
        self.peak_usage = self.peak_usage.max(region.bytes_used);
        self.stats.peak_frame_usage = self.peak_usage;
    }
}

// ---------------------------------------------------------------------------
// Bind group cache
// ---------------------------------------------------------------------------

/// A key for looking up cached bind groups.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindGroupKey {
    pub layout_id: u64,
    pub entries: Vec<BindGroupEntry>,
}

/// A single entry in a bind group.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindGroupEntry {
    pub binding: u32,
    pub resource: BindResource,
}

/// A bind resource descriptor (for hashing/caching).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BindResource {
    UniformBuffer {
        buffer_id: u64,
        offset: u32,
        size: u32,
    },
    StorageBuffer {
        buffer_id: u64,
        offset: u32,
        size: u32,
    },
    Texture {
        texture_id: u64,
    },
    Sampler {
        sampler_id: u64,
    },
}

/// A cached bind group handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CachedBindGroupId(pub u64);

/// Cache for bind groups to avoid re-creation each frame.
pub struct BindGroupCache {
    cache: HashMap<BindGroupKey, CachedBindGroupId>,
    next_id: u64,
    max_entries: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
    access_order: Vec<(BindGroupKey, u64)>,
}

impl BindGroupCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_entries),
            next_id: 1,
            max_entries,
            hits: 0,
            misses: 0,
            evictions: 0,
            access_order: Vec::new(),
        }
    }

    /// Look up a bind group by key.
    pub fn get(&mut self, key: &BindGroupKey) -> Option<CachedBindGroupId> {
        if let Some(&id) = self.cache.get(key) {
            self.hits += 1;
            Some(id)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a new bind group into the cache.
    pub fn insert(&mut self, key: BindGroupKey) -> CachedBindGroupId {
        if self.cache.len() >= self.max_entries {
            self.evict_oldest();
        }

        let id = CachedBindGroupId(self.next_id);
        self.next_id += 1;
        self.access_order.push((key.clone(), self.next_id));
        self.cache.insert(key, id);
        id
    }

    /// Get or create a bind group.
    pub fn get_or_insert(&mut self, key: BindGroupKey) -> (CachedBindGroupId, bool) {
        if let Some(&id) = self.cache.get(&key) {
            self.hits += 1;
            (id, true)
        } else {
            self.misses += 1;
            let id = self.insert(key);
            (id, false)
        }
    }

    /// Invalidate all entries referencing a specific buffer.
    pub fn invalidate_buffer(&mut self, buffer_id: u64) {
        self.cache.retain(|key, _| {
            !key.entries.iter().any(|e| match &e.resource {
                BindResource::UniformBuffer { buffer_id: bid, .. } => *bid == buffer_id,
                BindResource::StorageBuffer { buffer_id: bid, .. } => *bid == buffer_id,
                _ => false,
            })
        });
    }

    /// Invalidate all entries referencing a specific texture.
    pub fn invalidate_texture(&mut self, texture_id: u64) {
        self.cache.retain(|key, _| {
            !key.entries.iter().any(|e| match &e.resource {
                BindResource::Texture { texture_id: tid } => *tid == texture_id,
                _ => false,
            })
        });
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
    }

    pub fn entry_count(&self) -> usize {
        self.cache.len()
    }

    pub fn hit_count(&self) -> u64 {
        self.hits
    }

    pub fn miss_count(&self) -> u64 {
        self.misses
    }

    pub fn hit_ratio(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f32 / total as f32
    }

    fn evict_oldest(&mut self) {
        if let Some((key, _)) = self.access_order.first().cloned() {
            self.cache.remove(&key);
            self.access_order.remove(0);
            self.evictions += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-material uniform buffer
// ---------------------------------------------------------------------------

/// A per-material uniform buffer that tracks dirty state.
#[derive(Debug, Clone)]
pub struct MaterialUniformBuffer {
    data: Vec<u8>,
    dirty: bool,
    layout: Vec<UniformField>,
    name_to_index: HashMap<String, usize>,
}

/// A field within a uniform buffer layout.
#[derive(Debug, Clone)]
pub struct UniformField {
    pub name: String,
    pub offset: usize,
    pub size: usize,
    pub field_type: UniformFieldType,
}

/// Type of a uniform field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniformFieldType {
    Float,
    Vec2,
    Vec3,
    Vec4,
    Mat3,
    Mat4,
    Int,
    IVec2,
    IVec3,
    IVec4,
    UInt,
}

impl UniformFieldType {
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Float => 4,
            Self::Vec2 => 8,
            Self::Vec3 => 12,
            Self::Vec4 => 16,
            Self::Mat3 => 48, // 3 x vec4 (std140 padding)
            Self::Mat4 => 64,
            Self::Int => 4,
            Self::IVec2 => 8,
            Self::IVec3 => 12,
            Self::IVec4 => 16,
            Self::UInt => 4,
        }
    }

    pub fn alignment(&self) -> usize {
        match self {
            Self::Float | Self::Int | Self::UInt => 4,
            Self::Vec2 | Self::IVec2 => 8,
            Self::Vec3 | Self::IVec3 | Self::Vec4 | Self::IVec4 => 16,
            Self::Mat3 | Self::Mat4 => 16,
        }
    }
}

impl MaterialUniformBuffer {
    /// Build a material uniform buffer from a list of fields.
    /// Fields are packed according to std140 rules.
    pub fn new(fields: &[(String, UniformFieldType)]) -> Self {
        let mut layout = Vec::with_capacity(fields.len());
        let mut name_to_index = HashMap::new();
        let mut current_offset = 0usize;

        for (i, (name, field_type)) in fields.iter().enumerate() {
            let alignment = field_type.alignment();
            current_offset = align_up(current_offset, alignment);

            layout.push(UniformField {
                name: name.clone(),
                offset: current_offset,
                size: field_type.size_bytes(),
                field_type: *field_type,
            });
            name_to_index.insert(name.clone(), i);

            current_offset += field_type.size_bytes();
        }

        // Align total size to 16 bytes (std140 struct alignment).
        let total_size = align_up(current_offset, 16);
        let data = vec![0u8; total_size];

        Self {
            data,
            dirty: true,
            layout,
            name_to_index,
        }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Set a float field by name.
    pub fn set_float(&mut self, name: &str, value: f32) -> bool {
        self.set_bytes(name, &value.to_le_bytes())
    }

    /// Set a vec2 field by name.
    pub fn set_vec2(&mut self, name: &str, value: [f32; 2]) -> bool {
        let bytes: Vec<u8> = value.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.set_bytes(name, &bytes)
    }

    /// Set a vec3 field by name.
    pub fn set_vec3(&mut self, name: &str, value: [f32; 3]) -> bool {
        let bytes: Vec<u8> = value.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.set_bytes(name, &bytes)
    }

    /// Set a vec4 field by name.
    pub fn set_vec4(&mut self, name: &str, value: [f32; 4]) -> bool {
        let bytes: Vec<u8> = value.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.set_bytes(name, &bytes)
    }

    /// Set a mat4 field by name (column-major layout).
    pub fn set_mat4(&mut self, name: &str, value: &[f32; 16]) -> bool {
        let bytes: Vec<u8> = value.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.set_bytes(name, &bytes)
    }

    /// Set an int field by name.
    pub fn set_int(&mut self, name: &str, value: i32) -> bool {
        self.set_bytes(name, &value.to_le_bytes())
    }

    /// Set a uint field by name.
    pub fn set_uint(&mut self, name: &str, value: u32) -> bool {
        self.set_bytes(name, &value.to_le_bytes())
    }

    /// Get a float field by name.
    pub fn get_float(&self, name: &str) -> Option<f32> {
        let bytes = self.get_bytes(name)?;
        if bytes.len() >= 4 {
            Some(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        } else {
            None
        }
    }

    /// Get a vec4 field by name.
    pub fn get_vec4(&self, name: &str) -> Option<[f32; 4]> {
        let bytes = self.get_bytes(name)?;
        if bytes.len() >= 16 {
            Some([
                f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
                f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
                f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
            ])
        } else {
            None
        }
    }

    fn set_bytes(&mut self, name: &str, bytes: &[u8]) -> bool {
        if let Some(&idx) = self.name_to_index.get(name) {
            let field = &self.layout[idx];
            let end = (field.offset + bytes.len()).min(self.data.len());
            let write_len = end - field.offset;
            self.data[field.offset..end].copy_from_slice(&bytes[..write_len]);
            self.dirty = true;
            true
        } else {
            false
        }
    }

    fn get_bytes(&self, name: &str) -> Option<&[u8]> {
        let idx = *self.name_to_index.get(name)?;
        let field = &self.layout[idx];
        Some(&self.data[field.offset..field.offset + field.size])
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from the uniform buffer system.
#[derive(Debug)]
pub enum UniformBufferError {
    OutOfSpace { requested: usize, available: usize },
    InvalidAllocation,
    OutOfBounds,
    ZeroSize,
}

impl std::fmt::Display for UniformBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfSpace { requested, available } => {
                write!(f, "Uniform buffer full: need {requested}, have {available}")
            }
            Self::InvalidAllocation => write!(f, "Invalid uniform buffer allocation"),
            Self::OutOfBounds => write!(f, "Write out of bounds"),
            Self::ZeroSize => write!(f, "Zero-size allocation requested"),
        }
    }
}

impl std::error::Error for UniformBufferError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
    }

    #[test]
    fn test_dynamic_uniform_buffer() {
        let mut buf = DynamicUniformBuffer::with_defaults();
        buf.begin_frame();

        let data = [1.0f32, 2.0, 3.0, 4.0];
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, 16)
        };
        let alloc = buf.allocate_and_write(bytes).unwrap();
        assert!(alloc.is_valid());
        assert!(is_aligned(alloc.offset as usize, 256));

        let read_back = buf.read(alloc).unwrap();
        assert_eq!(&read_back[..16], bytes);
    }

    #[test]
    fn test_multiple_allocations() {
        let config = DynamicUniformBufferConfig {
            buffer_size: 4096,
            min_alignment: 256,
            frames_in_flight: 1,
            zero_on_reset: false,
        };
        let mut buf = DynamicUniformBuffer::new(config);
        buf.begin_frame();

        let a1 = buf.allocate(64).unwrap();
        let a2 = buf.allocate(128).unwrap();

        assert_ne!(a1.offset, a2.offset);
        assert!(a2.offset >= a1.offset + a1.size);
    }

    #[test]
    fn test_out_of_space() {
        let config = DynamicUniformBufferConfig {
            buffer_size: 512,
            min_alignment: 256,
            frames_in_flight: 1,
            zero_on_reset: false,
        };
        let mut buf = DynamicUniformBuffer::new(config);
        buf.begin_frame();

        let _ = buf.allocate(256).unwrap();
        // 256 bytes left, but after alignment only 256-byte block fits
        let result = buf.allocate(512);
        assert!(result.is_err());
    }

    #[test]
    fn test_frame_reset() {
        let config = DynamicUniformBufferConfig {
            buffer_size: 2048,
            min_alignment: 256,
            frames_in_flight: 2,
            zero_on_reset: false,
        };
        let mut buf = DynamicUniformBuffer::new(config);

        buf.begin_frame();
        let a1 = buf.allocate(64).unwrap();

        buf.begin_frame();
        let a2 = buf.allocate(64).unwrap();

        // Different frames should use different regions
        assert_ne!(a1.buffer_index, a2.buffer_index);
    }

    #[test]
    fn test_material_uniform_buffer() {
        let mut ubuf = MaterialUniformBuffer::new(&[
            ("color".into(), UniformFieldType::Vec4),
            ("roughness".into(), UniformFieldType::Float),
            ("metallic".into(), UniformFieldType::Float),
        ]);

        assert!(ubuf.set_vec4("color", [1.0, 0.0, 0.0, 1.0]));
        assert!(ubuf.set_float("roughness", 0.5));
        assert!(ubuf.set_float("metallic", 1.0));

        assert_eq!(ubuf.get_float("roughness"), Some(0.5));
        assert_eq!(ubuf.get_vec4("color"), Some([1.0, 0.0, 0.0, 1.0]));
        assert!(ubuf.is_dirty());
    }

    #[test]
    fn test_bind_group_cache() {
        let mut cache = BindGroupCache::new(100);
        let key = BindGroupKey {
            layout_id: 1,
            entries: vec![BindGroupEntry {
                binding: 0,
                resource: BindResource::UniformBuffer {
                    buffer_id: 42,
                    offset: 0,
                    size: 256,
                },
            }],
        };

        let (id1, hit) = cache.get_or_insert(key.clone());
        assert!(!hit);

        let (id2, hit) = cache.get_or_insert(key);
        assert!(hit);
        assert_eq!(id1, id2);
        assert!(cache.hit_ratio() > 0.0);
    }
}
