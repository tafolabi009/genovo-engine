// engine/render/src/render_targets.rs
//
// Render target management for the Genovo engine.
//
// Provides a unified system for managing off-screen render textures:
//
// - **RenderTexture creation** — Typed render texture creation with format,
//   size, and usage flags.
// - **Render texture pooling** — Pooled temporary render textures to avoid
//   redundant allocations.
// - **Temporary render textures** — Auto-released render textures for
//   intermediate passes.
// - **Format management** — Format validation and conversion utilities.
// - **Render-to-texture** — Configure framebuffers for render-to-texture
//   operations.
// - **Multi-render-target (MRT)** — Configure multiple colour attachments for
//   G-buffer or deferred shading.
// - **Readback** — Asynchronous and synchronous readback of render texture
//   contents to CPU memory.
// - **Mip generation** — Automatic mip-map generation for render textures.
//
// # Architecture
//
// The `RenderTargetManager` owns all render textures and framebuffers. Passes
// in the render graph request temporary render textures by descriptor; the
// manager returns a pooled or freshly created texture. At the end of each
// frame, temporary textures are released back to the pool.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Render texture format
// ---------------------------------------------------------------------------

/// Pixel format for render textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RtFormat {
    /// 8-bit RGBA (sRGB).
    Rgba8Srgb,
    /// 8-bit RGBA (unorm/linear).
    Rgba8Unorm,
    /// 10-bit RGB + 2-bit alpha.
    Rgb10A2Unorm,
    /// 16-bit float RGBA.
    Rgba16Float,
    /// 32-bit float RGBA.
    Rgba32Float,
    /// 16-bit float RG (two-channel).
    Rg16Float,
    /// 32-bit float RG.
    Rg32Float,
    /// 16-bit float R (single channel).
    R16Float,
    /// 32-bit float R.
    R32Float,
    /// 8-bit R (single channel).
    R8Unorm,
    /// 16-bit depth.
    Depth16,
    /// 24-bit depth.
    Depth24,
    /// 32-bit float depth.
    Depth32Float,
    /// 24-bit depth + 8-bit stencil.
    Depth24Stencil8,
    /// 32-bit float depth + 8-bit stencil.
    Depth32FloatStencil8,
    /// 11-bit float R + 11-bit float G + 10-bit float B.
    Rg11B10Float,
    /// 8-bit BGRA (sRGB).
    Bgra8Srgb,
    /// 8-bit BGRA (unorm).
    Bgra8Unorm,
}

impl RtFormat {
    /// Get the number of bytes per pixel for this format.
    pub fn bytes_per_pixel(&self) -> u32 {
        match self {
            Self::R8Unorm => 1,
            Self::Depth16 | Self::R16Float => 2,
            Self::Depth24 | Self::Rg16Float => 4,
            Self::Rgba8Srgb | Self::Rgba8Unorm | Self::Rgb10A2Unorm
            | Self::Depth24Stencil8 | Self::Depth32Float | Self::Rg32Float
            | Self::R32Float | Self::Rg11B10Float | Self::Bgra8Srgb | Self::Bgra8Unorm => 4,
            Self::Depth32FloatStencil8 => 5,
            Self::Rgba16Float => 8,
            Self::Rgba32Float => 16,
        }
    }

    /// Whether this is a depth/stencil format.
    pub fn is_depth(&self) -> bool {
        matches!(
            self,
            Self::Depth16 | Self::Depth24 | Self::Depth32Float
            | Self::Depth24Stencil8 | Self::Depth32FloatStencil8
        )
    }

    /// Whether this format has a stencil component.
    pub fn has_stencil(&self) -> bool {
        matches!(self, Self::Depth24Stencil8 | Self::Depth32FloatStencil8)
    }

    /// Whether this is a floating-point format.
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            Self::Rgba16Float | Self::Rgba32Float | Self::Rg16Float | Self::Rg32Float
            | Self::R16Float | Self::R32Float | Self::Depth32Float
            | Self::Depth32FloatStencil8 | Self::Rg11B10Float
        )
    }

    /// Whether this format supports sRGB.
    pub fn is_srgb(&self) -> bool {
        matches!(self, Self::Rgba8Srgb | Self::Bgra8Srgb)
    }

    /// Number of colour channels.
    pub fn channel_count(&self) -> u32 {
        match self {
            Self::R8Unorm | Self::R16Float | Self::R32Float => 1,
            Self::Rg16Float | Self::Rg32Float => 2,
            Self::Rg11B10Float => 3,
            Self::Rgba8Srgb | Self::Rgba8Unorm | Self::Rgb10A2Unorm
            | Self::Rgba16Float | Self::Rgba32Float
            | Self::Bgra8Srgb | Self::Bgra8Unorm => 4,
            _ => 0, // Depth formats don't have traditional colour channels.
        }
    }
}

// ---------------------------------------------------------------------------
// Render texture descriptor
// ---------------------------------------------------------------------------

/// Usage flags for render textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RtUsage(u32);

impl RtUsage {
    /// Can be used as a colour attachment.
    pub const COLOR_ATTACHMENT: Self = Self(1 << 0);
    /// Can be used as a depth/stencil attachment.
    pub const DEPTH_STENCIL: Self = Self(1 << 1);
    /// Can be sampled in a shader.
    pub const SAMPLED: Self = Self(1 << 2);
    /// Can be used as a storage texture (UAV).
    pub const STORAGE: Self = Self(1 << 3);
    /// Can be copied from (source of a copy operation).
    pub const COPY_SRC: Self = Self(1 << 4);
    /// Can be copied to (destination of a copy operation).
    pub const COPY_DST: Self = Self(1 << 5);

    /// Combine two usage flags.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Check if a flag is set.
    pub fn contains(self, flag: Self) -> bool {
        (self.0 & flag.0) == flag.0
    }
}

/// Describes a render texture to be created or requested from the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderTextureDesc {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel format.
    pub format: RtFormat,
    /// Usage flags.
    pub usage: RtUsage,
    /// Number of mip levels (1 = no mips).
    pub mip_count: u32,
    /// Number of array layers (1 = non-array).
    pub array_layers: u32,
    /// Multisampling sample count (1 = no MSAA).
    pub sample_count: u32,
}

impl RenderTextureDesc {
    /// Create a simple colour render texture descriptor.
    pub fn color(width: u32, height: u32, format: RtFormat) -> Self {
        Self {
            width,
            height,
            format,
            usage: RtUsage::COLOR_ATTACHMENT.union(RtUsage::SAMPLED),
            mip_count: 1,
            array_layers: 1,
            sample_count: 1,
        }
    }

    /// Create a depth render texture descriptor.
    pub fn depth(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            format: RtFormat::Depth32Float,
            usage: RtUsage::DEPTH_STENCIL.union(RtUsage::SAMPLED),
            mip_count: 1,
            array_layers: 1,
            sample_count: 1,
        }
    }

    /// Create a depth+stencil render texture descriptor.
    pub fn depth_stencil(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            format: RtFormat::Depth24Stencil8,
            usage: RtUsage::DEPTH_STENCIL.union(RtUsage::SAMPLED),
            mip_count: 1,
            array_layers: 1,
            sample_count: 1,
        }
    }

    /// Create a HDR colour render texture.
    pub fn hdr(width: u32, height: u32) -> Self {
        Self::color(width, height, RtFormat::Rgba16Float)
    }

    /// With mip-map chain.
    pub fn with_mips(mut self) -> Self {
        self.mip_count = compute_mip_count(self.width, self.height);
        self
    }

    /// With specific mip count.
    pub fn with_mip_count(mut self, count: u32) -> Self {
        self.mip_count = count;
        self
    }

    /// With MSAA.
    pub fn with_msaa(mut self, samples: u32) -> Self {
        self.sample_count = samples;
        self
    }

    /// With storage usage.
    pub fn with_storage(mut self) -> Self {
        self.usage = self.usage.union(RtUsage::STORAGE);
        self
    }

    /// With copy source usage.
    pub fn with_copy_src(mut self) -> Self {
        self.usage = self.usage.union(RtUsage::COPY_SRC);
        self
    }

    /// With copy destination usage.
    pub fn with_copy_dst(mut self) -> Self {
        self.usage = self.usage.union(RtUsage::COPY_DST);
        self
    }

    /// Compute memory footprint in bytes.
    pub fn memory_size(&self) -> u64 {
        let bpp = self.format.bytes_per_pixel() as u64;
        let mut total = 0u64;
        let mut w = self.width as u64;
        let mut h = self.height as u64;

        for _ in 0..self.mip_count {
            total += w * h * bpp;
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }

        total * self.array_layers as u64 * self.sample_count as u64
    }

    /// Check whether this descriptor is compatible with another for pooling.
    pub fn is_compatible(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && self.format == other.format
            && self.mip_count == other.mip_count
            && self.array_layers == other.array_layers
            && self.sample_count == other.sample_count
            && self.usage == other.usage
    }
}

/// Compute the number of mip levels for a given resolution.
fn compute_mip_count(width: u32, height: u32) -> u32 {
    let max_dim = width.max(height) as f32;
    (max_dim.log2().floor() as u32 + 1).max(1)
}

// ---------------------------------------------------------------------------
// Render texture handle
// ---------------------------------------------------------------------------

/// Opaque handle to a render texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderTextureHandle {
    /// Internal index into the texture array.
    pub index: u32,
    /// Generation counter (for stale handle detection).
    pub generation: u32,
}

impl RenderTextureHandle {
    pub const INVALID: Self = Self { index: u32::MAX, generation: 0 };

    pub fn is_valid(&self) -> bool {
        self.index != u32::MAX
    }
}

// ---------------------------------------------------------------------------
// Render texture
// ---------------------------------------------------------------------------

/// A managed render texture.
#[derive(Debug, Clone)]
pub struct RenderTexture {
    /// Handle.
    pub handle: RenderTextureHandle,
    /// Descriptor.
    pub desc: RenderTextureDesc,
    /// GPU texture handle (opaque, backend-specific).
    pub gpu_handle: u64,
    /// Debug label.
    pub label: String,
    /// Whether this texture is currently in use.
    pub in_use: bool,
    /// Frame when this texture was last used.
    pub last_used_frame: u64,
    /// Whether this is a temporary (pooled) texture.
    pub temporary: bool,
}

// ---------------------------------------------------------------------------
// Framebuffer (MRT)
// ---------------------------------------------------------------------------

/// A colour attachment in a framebuffer.
#[derive(Debug, Clone, Copy)]
pub struct ColorAttachment {
    /// Render texture handle.
    pub texture: RenderTextureHandle,
    /// Mip level to render into.
    pub mip_level: u32,
    /// Array layer to render into.
    pub array_layer: u32,
    /// Clear value (RGBA).
    pub clear_value: [f32; 4],
    /// Whether to clear this attachment at the start of the pass.
    pub clear: bool,
    /// Whether to store the result (false = discard, e.g. MSAA resolve source).
    pub store: bool,
}

impl ColorAttachment {
    pub fn new(texture: RenderTextureHandle) -> Self {
        Self {
            texture,
            mip_level: 0,
            array_layer: 0,
            clear_value: [0.0, 0.0, 0.0, 0.0],
            clear: true,
            store: true,
        }
    }

    pub fn with_clear_color(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.clear_value = [r, g, b, a];
        self
    }

    pub fn without_clear(mut self) -> Self {
        self.clear = false;
        self
    }
}

/// A depth/stencil attachment in a framebuffer.
#[derive(Debug, Clone, Copy)]
pub struct DepthStencilAttachment {
    /// Render texture handle.
    pub texture: RenderTextureHandle,
    /// Depth clear value.
    pub depth_clear: f32,
    /// Stencil clear value.
    pub stencil_clear: u8,
    /// Whether to clear depth.
    pub clear_depth: bool,
    /// Whether to clear stencil.
    pub clear_stencil: bool,
    /// Whether to store depth.
    pub store_depth: bool,
    /// Whether to store stencil.
    pub store_stencil: bool,
    /// Read-only depth (no depth writes).
    pub read_only_depth: bool,
    /// Read-only stencil (no stencil writes).
    pub read_only_stencil: bool,
}

impl DepthStencilAttachment {
    pub fn new(texture: RenderTextureHandle) -> Self {
        Self {
            texture,
            depth_clear: 1.0,
            stencil_clear: 0,
            clear_depth: true,
            clear_stencil: true,
            store_depth: true,
            store_stencil: true,
            read_only_depth: false,
            read_only_stencil: false,
        }
    }

    pub fn read_only(texture: RenderTextureHandle) -> Self {
        Self {
            texture,
            depth_clear: 1.0,
            stencil_clear: 0,
            clear_depth: false,
            clear_stencil: false,
            store_depth: false,
            store_stencil: false,
            read_only_depth: true,
            read_only_stencil: true,
        }
    }
}

/// A framebuffer configuration (MRT).
#[derive(Debug, Clone)]
pub struct FramebufferConfig {
    /// Colour attachments (up to 8).
    pub color_attachments: Vec<ColorAttachment>,
    /// Depth/stencil attachment (optional).
    pub depth_stencil: Option<DepthStencilAttachment>,
    /// Render area width.
    pub width: u32,
    /// Render area height.
    pub height: u32,
    /// Debug label.
    pub label: String,
}

impl FramebufferConfig {
    /// Single colour attachment + depth.
    pub fn simple(
        color: RenderTextureHandle,
        depth: RenderTextureHandle,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            color_attachments: vec![ColorAttachment::new(color)],
            depth_stencil: Some(DepthStencilAttachment::new(depth)),
            width,
            height,
            label: String::new(),
        }
    }

    /// G-buffer style MRT (multiple colour attachments + depth).
    pub fn gbuffer(
        albedo: RenderTextureHandle,
        normal: RenderTextureHandle,
        material: RenderTextureHandle,
        depth: RenderTextureHandle,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            color_attachments: vec![
                ColorAttachment::new(albedo),
                ColorAttachment::new(normal),
                ColorAttachment::new(material),
            ],
            depth_stencil: Some(DepthStencilAttachment::new(depth)),
            width,
            height,
            label: "G-Buffer".to_string(),
        }
    }

    /// Colour-only (no depth).
    pub fn color_only(color: RenderTextureHandle, width: u32, height: u32) -> Self {
        Self {
            color_attachments: vec![ColorAttachment::new(color)],
            depth_stencil: None,
            width,
            height,
            label: String::new(),
        }
    }

    /// Number of colour attachments.
    pub fn color_count(&self) -> usize {
        self.color_attachments.len()
    }

    /// Whether this framebuffer has a depth attachment.
    pub fn has_depth(&self) -> bool {
        self.depth_stencil.is_some()
    }
}

// ---------------------------------------------------------------------------
// Readback request
// ---------------------------------------------------------------------------

/// Status of a GPU readback request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadbackStatus {
    /// Request is pending GPU completion.
    Pending,
    /// Data is ready to read.
    Ready,
    /// Request failed.
    Failed,
    /// Request was cancelled.
    Cancelled,
}

/// A request to read render texture data back to the CPU.
#[derive(Debug, Clone)]
pub struct ReadbackRequest {
    /// Unique request ID.
    pub id: u64,
    /// Source render texture.
    pub source: RenderTextureHandle,
    /// Mip level to read.
    pub mip_level: u32,
    /// Array layer to read.
    pub array_layer: u32,
    /// Region to read (x, y, width, height). None = full texture.
    pub region: Option<(u32, u32, u32, u32)>,
    /// Status.
    pub status: ReadbackStatus,
    /// Frame when the request was submitted.
    pub submit_frame: u64,
    /// Result data (filled when Ready).
    pub data: Option<Vec<u8>>,
    /// Row pitch of the result data.
    pub row_pitch: u32,
}

impl ReadbackRequest {
    /// Create a new readback request.
    pub fn new(id: u64, source: RenderTextureHandle) -> Self {
        Self {
            id,
            source,
            mip_level: 0,
            array_layer: 0,
            region: None,
            status: ReadbackStatus::Pending,
            submit_frame: 0,
            data: None,
            row_pitch: 0,
        }
    }

    /// Read a specific region.
    pub fn with_region(mut self, x: u32, y: u32, w: u32, h: u32) -> Self {
        self.region = Some((x, y, w, h));
        self
    }

    /// Is the data ready?
    pub fn is_ready(&self) -> bool {
        self.status == ReadbackStatus::Ready
    }
}

// ---------------------------------------------------------------------------
// Render target manager
// ---------------------------------------------------------------------------

/// Manages render textures, framebuffers, pooling, and readback.
#[derive(Debug)]
pub struct RenderTargetManager {
    /// All render textures.
    textures: Vec<RenderTexture>,
    /// Free pool: descriptor → list of available texture indices.
    pool: HashMap<u64, Vec<u32>>,
    /// Pending readback requests.
    readback_requests: Vec<ReadbackRequest>,
    /// Next handle generation.
    next_generation: u32,
    /// Next readback request ID.
    next_readback_id: u64,
    /// Current frame.
    frame: u64,
    /// Maximum number of frames a texture can be unused before being freed.
    pub max_idle_frames: u64,
    /// Total GPU memory used by render textures.
    pub total_memory: u64,
    /// Memory budget (0 = unlimited).
    pub memory_budget: u64,
}

impl RenderTargetManager {
    /// Create a new render target manager.
    pub fn new() -> Self {
        Self {
            textures: Vec::new(),
            pool: HashMap::new(),
            readback_requests: Vec::new(),
            next_generation: 1,
            next_readback_id: 1,
            frame: 0,
            max_idle_frames: 8,
            total_memory: 0,
            memory_budget: 0,
        }
    }

    /// Create a persistent (non-temporary) render texture.
    pub fn create(&mut self, desc: RenderTextureDesc, label: impl Into<String>) -> RenderTextureHandle {
        let handle = RenderTextureHandle {
            index: self.textures.len() as u32,
            generation: self.next_generation,
        };
        self.next_generation += 1;

        let mem = desc.memory_size();
        self.total_memory += mem;

        self.textures.push(RenderTexture {
            handle,
            desc,
            gpu_handle: 0, // Would be filled by the GPU backend.
            label: label.into(),
            in_use: true,
            last_used_frame: self.frame,
            temporary: false,
        });

        handle
    }

    /// Request a temporary render texture from the pool.
    ///
    /// If a compatible texture is available in the pool, it is reused.
    /// Otherwise, a new texture is created.
    pub fn get_temporary(&mut self, desc: RenderTextureDesc) -> RenderTextureHandle {
        let key = self.pool_key(&desc);

        // Try to find a pooled texture.
        if let Some(indices) = self.pool.get_mut(&key) {
            if let Some(idx) = indices.pop() {
                if let Some(tex) = self.textures.get_mut(idx as usize) {
                    tex.in_use = true;
                    tex.last_used_frame = self.frame;
                    return tex.handle;
                }
            }
        }

        // Create a new temporary texture.
        let handle = RenderTextureHandle {
            index: self.textures.len() as u32,
            generation: self.next_generation,
        };
        self.next_generation += 1;

        let mem = desc.memory_size();
        self.total_memory += mem;

        self.textures.push(RenderTexture {
            handle,
            desc,
            gpu_handle: 0,
            label: format!("Temp_{}x{}", desc.width, desc.height),
            in_use: true,
            last_used_frame: self.frame,
            temporary: true,
        });

        handle
    }

    /// Release a temporary render texture back to the pool.
    pub fn release_temporary(&mut self, handle: RenderTextureHandle) {
        if let Some(tex) = self.textures.get_mut(handle.index as usize) {
            if tex.handle.generation != handle.generation {
                return; // Stale handle.
            }
            tex.in_use = false;
            let desc_clone = tex.desc.clone();
            let key = Self::pool_key_static(&desc_clone);
            self.pool.entry(key).or_default().push(handle.index);
        }
    }

    /// Get a render texture by handle.
    pub fn get(&self, handle: RenderTextureHandle) -> Option<&RenderTexture> {
        self.textures.get(handle.index as usize).filter(|t| t.handle.generation == handle.generation)
    }

    /// Get a mutable render texture by handle.
    pub fn get_mut(&mut self, handle: RenderTextureHandle) -> Option<&mut RenderTexture> {
        self.textures.get_mut(handle.index as usize).filter(|t| t.handle.generation == handle.generation)
    }

    /// Submit a readback request.
    pub fn request_readback(&mut self, source: RenderTextureHandle) -> u64 {
        let id = self.next_readback_id;
        self.next_readback_id += 1;

        let mut req = ReadbackRequest::new(id, source);
        req.submit_frame = self.frame;
        self.readback_requests.push(req);
        id
    }

    /// Check if a readback request is complete.
    pub fn readback_status(&self, id: u64) -> ReadbackStatus {
        self.readback_requests
            .iter()
            .find(|r| r.id == id)
            .map(|r| r.status)
            .unwrap_or(ReadbackStatus::Failed)
    }

    /// Get readback data if ready.
    pub fn get_readback_data(&self, id: u64) -> Option<&[u8]> {
        self.readback_requests
            .iter()
            .find(|r| r.id == id && r.status == ReadbackStatus::Ready)
            .and_then(|r| r.data.as_deref())
    }

    /// Begin a new frame: evict old pool entries.
    pub fn begin_frame(&mut self) {
        self.frame += 1;

        // Evict textures that have been idle too long.
        let max_idle = self.max_idle_frames;
        let frame = self.frame;

        for entry in self.pool.values_mut() {
            entry.retain(|&idx| {
                if let Some(tex) = self.textures.get(idx as usize) {
                    frame - tex.last_used_frame < max_idle
                } else {
                    false
                }
            });
        }

        // Remove empty pool entries.
        self.pool.retain(|_, v| !v.is_empty());

        // Clean up completed readback requests older than a few frames.
        self.readback_requests.retain(|r| {
            r.status == ReadbackStatus::Pending || frame - r.submit_frame < 4
        });
    }

    /// End of frame: release all temporary textures still in use.
    pub fn end_frame(&mut self) {
        let mut to_release = Vec::new();
        for tex in &self.textures {
            if tex.temporary && tex.in_use {
                to_release.push(tex.handle);
            }
        }
        for handle in to_release {
            self.release_temporary(handle);
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> RenderTargetStats {
        let mut total_textures = 0u32;
        let mut active_textures = 0u32;
        let mut pooled_textures = 0u32;

        for tex in &self.textures {
            total_textures += 1;
            if tex.in_use {
                active_textures += 1;
            }
        }

        for indices in self.pool.values() {
            pooled_textures += indices.len() as u32;
        }

        RenderTargetStats {
            total_textures,
            active_textures,
            pooled_textures,
            total_memory_bytes: self.total_memory,
            pending_readbacks: self.readback_requests.iter().filter(|r| r.status == ReadbackStatus::Pending).count() as u32,
        }
    }

    /// Pool key from descriptor (static version to avoid borrow conflicts).
    fn pool_key_static(desc: &RenderTextureDesc) -> u64 {
        let mut key = desc.width as u64;
        key = key.wrapping_mul(65537).wrapping_add(desc.height as u64);
        key = key.wrapping_mul(65537).wrapping_add(desc.format as u64);
        key = key.wrapping_mul(65537).wrapping_add(desc.mip_count as u64);
        key = key.wrapping_mul(65537).wrapping_add(desc.sample_count as u64);
        key = key.wrapping_mul(65537).wrapping_add(desc.usage.0 as u64);
        key
    }

    /// Pool key from descriptor (hash-like key for grouping compatible textures).
    fn pool_key(&self, desc: &RenderTextureDesc) -> u64 {
        let mut key = desc.width as u64;
        key = key.wrapping_mul(65537).wrapping_add(desc.height as u64);
        key = key.wrapping_mul(65537).wrapping_add(desc.format as u64);
        key = key.wrapping_mul(65537).wrapping_add(desc.mip_count as u64);
        key = key.wrapping_mul(65537).wrapping_add(desc.sample_count as u64);
        key = key.wrapping_mul(65537).wrapping_add(desc.usage.0 as u64);
        key
    }
}

/// Render target statistics.
#[derive(Debug, Clone)]
pub struct RenderTargetStats {
    pub total_textures: u32,
    pub active_textures: u32,
    pub pooled_textures: u32,
    pub total_memory_bytes: u64,
    pub pending_readbacks: u32,
}

impl RenderTargetStats {
    /// Total memory in megabytes.
    pub fn total_memory_mb(&self) -> f64 {
        self.total_memory_bytes as f64 / (1024.0 * 1024.0)
    }
}

// ---------------------------------------------------------------------------
// Mip generation
// ---------------------------------------------------------------------------

/// Mip generation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MipGenMode {
    /// Box filter (average of 4 texels).
    Box,
    /// Bilinear filter.
    Bilinear,
    /// Kaiser filter (higher quality).
    Kaiser,
    /// Min filter (for depth/shadow maps).
    Min,
    /// Max filter (for hi-Z).
    Max,
}

/// Configuration for mip-map generation.
#[derive(Debug, Clone)]
pub struct MipGenConfig {
    /// Generation mode.
    pub mode: MipGenMode,
    /// Start mip level (0 = generate from the base level).
    pub start_mip: u32,
    /// Number of mips to generate (0 = all remaining).
    pub mip_count: u32,
    /// Array layer to generate mips for (u32::MAX = all layers).
    pub array_layer: u32,
    /// Whether to use compute shader (true) or graphics pipeline (false).
    pub use_compute: bool,
}

impl Default for MipGenConfig {
    fn default() -> Self {
        Self {
            mode: MipGenMode::Box,
            start_mip: 0,
            mip_count: 0,
            array_layer: u32::MAX,
            use_compute: true,
        }
    }
}

/// Compute the size of a mip level.
pub fn mip_size(base_width: u32, base_height: u32, mip: u32) -> (u32, u32) {
    let w = (base_width >> mip).max(1);
    let h = (base_height >> mip).max(1);
    (w, h)
}

/// Generate a mip-chain of sizes.
pub fn mip_chain_sizes(base_width: u32, base_height: u32, mip_count: u32) -> Vec<(u32, u32)> {
    (0..mip_count).map(|m| mip_size(base_width, base_height, m)).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_properties() {
        assert_eq!(RtFormat::Rgba8Srgb.bytes_per_pixel(), 4);
        assert!(RtFormat::Depth32Float.is_depth());
        assert!(!RtFormat::Rgba8Srgb.is_depth());
        assert!(RtFormat::Rgba16Float.is_float());
    }

    #[test]
    fn test_render_texture_desc() {
        let desc = RenderTextureDesc::hdr(1920, 1080);
        assert_eq!(desc.format, RtFormat::Rgba16Float);
        let mem = desc.memory_size();
        assert_eq!(mem, 1920 * 1080 * 8);
    }

    #[test]
    fn test_pool_round_trip() {
        let mut mgr = RenderTargetManager::new();
        let desc = RenderTextureDesc::color(256, 256, RtFormat::Rgba8Unorm);

        let h1 = mgr.get_temporary(desc);
        assert!(mgr.get(h1).is_some());

        mgr.release_temporary(h1);

        // Requesting the same desc should reuse the pooled texture.
        let h2 = mgr.get_temporary(desc);
        assert_eq!(h1.index, h2.index);
    }

    #[test]
    fn test_mip_size() {
        assert_eq!(mip_size(1024, 512, 0), (1024, 512));
        assert_eq!(mip_size(1024, 512, 1), (512, 256));
        assert_eq!(mip_size(1024, 512, 10), (1, 1));
    }

    #[test]
    fn test_framebuffer_config() {
        let h1 = RenderTextureHandle { index: 0, generation: 1 };
        let h2 = RenderTextureHandle { index: 1, generation: 1 };
        let fb = FramebufferConfig::simple(h1, h2, 1920, 1080);
        assert_eq!(fb.color_count(), 1);
        assert!(fb.has_depth());
    }
}
