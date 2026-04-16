// engine/render/src/interface/resource.rs
//
// GPU resource type definitions: buffers, textures, samplers, shaders, and
// synchronisation primitives used by every backend through the `RenderDevice`
// trait.

use bitflags::bitflags;

// ---------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------

/// Opaque handle to a GPU buffer allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(pub(crate) u64);

/// Opaque handle to a GPU texture (image) allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub(crate) u64);

/// Opaque handle to a sampler object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerHandle(pub(crate) u64);

/// Opaque handle to a compiled shader module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderHandle(pub(crate) u64);

/// Opaque handle to a pipeline (graphics or compute).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineHandle(pub(crate) u64);

/// Opaque handle to a render pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderPassHandle(pub(crate) u64);

/// Opaque handle to a framebuffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FramebufferHandle(pub(crate) u64);

/// Opaque handle to a GPU fence (CPU-GPU synchronisation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FenceHandle(pub(crate) u64);

/// Opaque handle to a GPU semaphore (GPU-GPU synchronisation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SemaphoreHandle(pub(crate) u64);

// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------

bitflags! {
    /// Describes how a buffer may be used by the GPU and CPU.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BufferUsage: u32 {
        /// Source of a transfer / copy operation.
        const TRANSFER_SRC    = 1 << 0;
        /// Destination of a transfer / copy operation.
        const TRANSFER_DST    = 1 << 1;
        /// Uniform (constant) buffer binding.
        const UNIFORM         = 1 << 2;
        /// Storage (read/write) buffer binding.
        const STORAGE         = 1 << 3;
        /// Index buffer binding.
        const INDEX           = 1 << 4;
        /// Vertex buffer binding.
        const VERTEX          = 1 << 5;
        /// Indirect draw/dispatch argument buffer.
        const INDIRECT        = 1 << 6;
    }
}

/// Descriptor used to create a GPU buffer via [`RenderDevice::create_buffer`].
#[derive(Debug, Clone)]
pub struct BufferDesc {
    /// Human-readable label for debug tooling.
    pub label: Option<String>,
    /// Size in bytes.
    pub size: u64,
    /// Intended usage flags.
    pub usage: BufferUsage,
    /// Where the allocation should live.
    pub memory: MemoryLocation,
}

// ---------------------------------------------------------------------------
// Texture
// ---------------------------------------------------------------------------

bitflags! {
    /// Describes how a texture may be used by the GPU.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct TextureUsage: u32 {
        /// Source of a transfer / copy operation.
        const TRANSFER_SRC       = 1 << 0;
        /// Destination of a transfer / copy operation.
        const TRANSFER_DST       = 1 << 1;
        /// Sampled in a shader (read-only).
        const SAMPLED            = 1 << 2;
        /// Used as a storage image (read/write in compute).
        const STORAGE            = 1 << 3;
        /// Colour attachment in a render pass.
        const COLOR_ATTACHMENT   = 1 << 4;
        /// Depth/stencil attachment in a render pass.
        const DEPTH_STENCIL      = 1 << 5;
        /// Input attachment (subpass input).
        const INPUT_ATTACHMENT   = 1 << 6;
    }
}

/// Comprehensive pixel format enumeration covering colour, depth, stencil, and
/// compressed block formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TextureFormat {
    // -- Unsigned normalised --
    R8Unorm,
    Rg8Unorm,
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Bgra8Unorm,
    Bgra8UnormSrgb,

    // -- Signed normalised --
    R8Snorm,
    Rg8Snorm,
    Rgba8Snorm,

    // -- Unsigned integer --
    R8Uint,
    Rg8Uint,
    Rgba8Uint,
    R16Uint,
    Rg16Uint,
    Rgba16Uint,
    R32Uint,
    Rg32Uint,
    Rgba32Uint,

    // -- Signed integer --
    R8Sint,
    Rg8Sint,
    Rgba8Sint,
    R16Sint,
    Rg16Sint,
    Rgba16Sint,
    R32Sint,
    Rg32Sint,
    Rgba32Sint,

    // -- Float --
    R16Float,
    Rg16Float,
    Rgba16Float,
    R32Float,
    Rg32Float,
    Rgba32Float,

    // -- Packed --
    Rgb10A2Unorm,
    Rg11B10Float,
    Rgb9E5Float,

    // -- Depth / stencil --
    Depth16Unorm,
    Depth24Plus,
    Depth24PlusStencil8,
    Depth32Float,
    Depth32FloatStencil8,
    Stencil8,

    // -- Block-compressed (BC) --
    Bc1RgbaUnorm,
    Bc1RgbaUnormSrgb,
    Bc2RgbaUnorm,
    Bc2RgbaUnormSrgb,
    Bc3RgbaUnorm,
    Bc3RgbaUnormSrgb,
    Bc4RUnorm,
    Bc4RSnorm,
    Bc5RgUnorm,
    Bc5RgSnorm,
    Bc6hRgbUfloat,
    Bc6hRgbSfloat,
    Bc7RgbaUnorm,
    Bc7RgbaUnormSrgb,

    // -- ASTC (mobile) --
    Astc4x4Unorm,
    Astc4x4UnormSrgb,
    Astc5x5Unorm,
    Astc5x5UnormSrgb,
    Astc6x6Unorm,
    Astc6x6UnormSrgb,
    Astc8x8Unorm,
    Astc8x8UnormSrgb,
    Astc10x10Unorm,
    Astc10x10UnormSrgb,
    Astc12x12Unorm,
    Astc12x12UnormSrgb,

    // -- ETC2 (mobile) --
    Etc2Rgb8Unorm,
    Etc2Rgb8UnormSrgb,
    Etc2Rgb8A1Unorm,
    Etc2Rgb8A1UnormSrgb,
    Etc2Rgba8Unorm,
    Etc2Rgba8UnormSrgb,
}

/// Texture dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureDimension {
    D1,
    D2,
    D3,
}

/// Descriptor used to create a GPU texture via [`RenderDevice::create_texture`].
#[derive(Debug, Clone)]
pub struct TextureDesc {
    /// Human-readable label for debug tooling.
    pub label: Option<String>,
    /// Format of each texel.
    pub format: TextureFormat,
    /// Dimensionality.
    pub dimension: TextureDimension,
    /// Width in texels.
    pub width: u32,
    /// Height in texels (1 for 1-D textures).
    pub height: u32,
    /// Depth or array layer count.
    pub depth_or_array_layers: u32,
    /// Number of mip levels.
    pub mip_levels: u32,
    /// Number of MSAA samples.
    pub sample_count: u32,
    /// Intended usage flags.
    pub usage: TextureUsage,
}

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------

/// Texture filtering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FilterMode {
    Nearest,
    Linear,
}

/// Texture address (wrap) mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
    ClampToBorder,
}

/// Descriptor used to create a sampler via [`RenderDevice::create_sampler`].
#[derive(Debug, Clone)]
pub struct SamplerDesc {
    /// Human-readable label for debug tooling.
    pub label: Option<String>,
    /// Minification filter.
    pub min_filter: FilterMode,
    /// Magnification filter.
    pub mag_filter: FilterMode,
    /// Mipmap filter.
    pub mipmap_filter: FilterMode,
    /// Addressing along U axis.
    pub address_mode_u: AddressMode,
    /// Addressing along V axis.
    pub address_mode_v: AddressMode,
    /// Addressing along W axis.
    pub address_mode_w: AddressMode,
    /// LOD bias added to the computed mip level.
    pub lod_bias: f32,
    /// Minimum LOD clamp.
    pub lod_min_clamp: f32,
    /// Maximum LOD clamp.
    pub lod_max_clamp: f32,
    /// Maximum anisotropy (1 = disabled).
    pub max_anisotropy: u8,
    /// Optional comparison function for shadow sampling.
    pub compare: Option<super::pipeline::CompareOp>,
}

// ---------------------------------------------------------------------------
// Shader
// ---------------------------------------------------------------------------

/// Pipeline stage a shader module targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
    Geometry,
    TessellationControl,
    TessellationEvaluation,
    Mesh,
    Task,
    RayGeneration,
    RayClosestHit,
    RayMiss,
    RayAnyHit,
    RayIntersection,
}

/// Descriptor for loading a compiled shader module (e.g. SPIR-V, DXIL, MSL).
#[derive(Debug, Clone)]
pub struct ShaderDesc {
    /// Human-readable label for debug tooling.
    pub label: Option<String>,
    /// Pipeline stage.
    pub stage: ShaderStage,
    /// Pre-compiled bytecode.
    pub bytecode: Vec<u8>,
    /// Entry point function name.
    pub entry_point: String,
}

// ---------------------------------------------------------------------------
// Render pass / Framebuffer
// ---------------------------------------------------------------------------

/// What to do with an attachment at the start of a render pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoadOp {
    /// Preserve existing contents.
    Load,
    /// Clear to a specified value.
    Clear,
    /// Contents are undefined (driver may discard).
    DontCare,
}

/// What to do with an attachment at the end of a render pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StoreOp {
    /// Write results to memory.
    Store,
    /// Results are not needed after the pass.
    DontCare,
}

/// Describes a single attachment within a render pass.
#[derive(Debug, Clone)]
pub struct AttachmentDesc {
    /// Texel format.
    pub format: TextureFormat,
    /// MSAA sample count.
    pub samples: u32,
    /// Load operation.
    pub load_op: LoadOp,
    /// Store operation.
    pub store_op: StoreOp,
    /// Load operation for stencil (if applicable).
    pub stencil_load_op: LoadOp,
    /// Store operation for stencil (if applicable).
    pub stencil_store_op: StoreOp,
}

/// Descriptor used to create a render pass via [`RenderDevice::create_render_pass`].
#[derive(Debug, Clone)]
pub struct RenderPassDesc {
    /// Human-readable label for debug tooling.
    pub label: Option<String>,
    /// Colour attachments.
    pub color_attachments: Vec<AttachmentDesc>,
    /// Optional depth/stencil attachment.
    pub depth_stencil_attachment: Option<AttachmentDesc>,
}

/// Descriptor used to create a framebuffer via [`RenderDevice::create_framebuffer`].
#[derive(Debug, Clone)]
pub struct FramebufferDesc {
    /// Human-readable label for debug tooling.
    pub label: Option<String>,
    /// Render pass this framebuffer is compatible with.
    pub render_pass: RenderPassHandle,
    /// Texture views bound as attachments, in order.
    pub attachments: Vec<TextureHandle>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Number of layers.
    pub layers: u32,
}

// ---------------------------------------------------------------------------
// Memory location
// ---------------------------------------------------------------------------

/// Hint for where a resource allocation should reside.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryLocation {
    /// Device-local memory (fastest for GPU access, not CPU-visible).
    GpuOnly,
    /// CPU-visible memory optimised for CPU-to-GPU uploads (staging, uniforms).
    CpuToGpu,
    /// CPU-visible memory optimised for GPU-to-CPU read-back.
    GpuToCpu,
}
