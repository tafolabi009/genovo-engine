// engine/render/src/lib.rs
//
// Root of the `genovo-render` crate. Provides a backend-agnostic rendering
// abstraction with pluggable GPU backends (Vulkan, DX12, Metal).
//
// # Crate organisation
//
// - [`interface`] -- Backend-agnostic traits and type definitions.
//   - [`interface::device`] -- Core `RenderDevice` trait.
//   - [`interface::command_buffer`] -- Command recording abstractions.
//   - [`interface::pipeline`] -- Pipeline state descriptors and enums.
//   - [`interface::resource`] -- GPU resource types and handles.
// - [`vulkan`] -- Vulkan backend (feature `vulkan`).
// - [`dx12`] -- DirectX 12 backend (feature `dx12`).
// - [`metal`] -- Metal backend (feature `metal`).
// - [`wgpu_backend`] -- wgpu cross-platform backend.
// - [`backend_factory`] -- Runtime backend construction.
// - [`renderer`] -- High-level renderer, render graph, and ECS components.
// - [`shader`] -- Shader loading, caching, and permutation management.

pub mod interface;

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "dx12")]
pub mod dx12;

#[cfg(feature = "metal")]
pub mod metal;

pub mod wgpu_backend;

pub mod backend_factory;
pub mod postprocess;
pub mod renderer;
pub mod shader;

pub mod particles;
pub mod trails;
pub mod decals;
pub mod decal_volume;
pub mod foliage;
pub mod volumetrics;

// PBR rendering pipeline modules.
pub mod pbr;
pub mod lighting;
pub mod shadows;
pub mod gi;

// Extended rendering pipeline modules.
pub mod deferred;
pub mod mesh;
pub mod camera_system;

// Sky, atmosphere, ocean, water, and GPU compute modules.
pub mod atmosphere;
pub mod compute;
pub mod gpu_particles;
pub mod ocean;
pub mod sky;
pub mod water;

// Lumen-style software ray tracing, virtual shadow maps, and unified GI.
pub mod raytracing;
pub mod virtual_shadow_maps;
pub mod lumen_gi;

// Nanite-style virtual geometry, GPU-driven rendering, and advanced shading.
pub mod virtual_geometry;
pub mod gpu_driven;
pub mod upscaling;
pub mod subsurface;
pub mod hair;

// Runtime material system, visual shader graph, and texture streaming.
pub mod material_system;
pub mod shader_graph;
pub mod texture_streaming;

// LOD mesh generation, billboard impostors, and instanced rendering.
pub mod lod_mesh;
pub mod billboard;
pub mod instanced_renderer;

// Frame graph, pipeline orchestration, atlas packing, culling, and debug viz.
pub mod render_graph_v2;
pub mod render_pipeline;
pub mod texture_atlas;
pub mod culling;
pub mod debug_visualization;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

// Interface re-exports (most commonly used types).
pub use interface::command_buffer::{
    CommandBuffer, CommandEncoder, ClearValue, RenderPassBeginInfo, RenderPassEncoder,
    ScissorRect, Viewport,
};
pub use interface::device::{DeviceCapabilities, QueueType, RenderDevice};
pub use interface::pipeline::{
    BlendFactor, BlendOp, BlendState, CompareOp, ComputePipelineDesc, CullMode,
    DepthStencilState, DescriptorSet, DescriptorSetLayout, FrontFace, GraphicsPipelineDesc,
    IndexFormat, PipelineLayout, PrimitiveTopology, RasterizerState, ShaderStageFlags,
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexInputDesc,
};
pub use interface::resource::{
    AddressMode, BufferDesc, BufferHandle, BufferUsage, FenceHandle, FilterMode,
    FramebufferDesc, FramebufferHandle, MemoryLocation, PipelineHandle, RenderPassDesc,
    RenderPassHandle, SamplerDesc, SamplerHandle, ShaderDesc, ShaderHandle, ShaderStage,
    TextureDesc, TextureDimension, TextureFormat, TextureHandle, TextureUsage,
};

// High-level re-exports.
pub use backend_factory::{create_render_device, detect_preferred_backend};
pub use renderer::{
    Camera, FrameContext, MeshRenderer, Projection, RenderGraph, RenderQueue, Renderer,
    WgpuRenderer,
};
pub use shader::{
    MaterialShader, ShaderLibrary, ShaderPermutation, BUILTIN_TRIANGLE_WGSL,
    BUILTIN_SOLID_COLOR_WGSL, BUILTIN_VERTEX_COLOR_WGSL, BUILTIN_TEXTURED_WGSL,
    BUILTIN_DEPTH_ONLY_WGSL,
};
pub use wgpu_backend::{WgpuDevice, WgpuSurface};

// ---------------------------------------------------------------------------
// RenderBackend enum
// ---------------------------------------------------------------------------

/// Selects which GPU backend to use at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderBackend {
    /// Vulkan 1.2+ (Windows, Linux, Android).
    Vulkan,
    /// DirectX 12 (Windows only).
    Dx12,
    /// Metal 3+ (macOS, iOS, visionOS).
    Metal,
    /// wgpu cross-platform backend (Vulkan, DX12, Metal, WebGPU under the hood).
    Wgpu,
    /// Automatically select the best available backend for the current platform.
    /// Prefers wgpu for its cross-platform support.
    Auto,
}

impl std::fmt::Display for RenderBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vulkan => write!(f, "Vulkan"),
            Self::Dx12 => write!(f, "DirectX 12"),
            Self::Metal => write!(f, "Metal"),
            Self::Wgpu => write!(f, "wgpu"),
            Self::Auto => write!(f, "Auto"),
        }
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the render crate.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RenderError {
    /// The requested backend feature was not compiled in.
    #[error("Render backend not available: {0}")]
    BackendNotAvailable(&'static str),

    /// The backend is compiled in but the operation is not yet implemented.
    #[error("Render backend not implemented: {0}")]
    BackendNotImplemented(&'static str),

    /// GPU resource creation failed.
    #[error("Resource creation failed: {0}")]
    ResourceCreation(String),

    /// Shader loading or compilation failed.
    #[error("Shader load error: {0}")]
    ShaderLoad(String),

    /// Command buffer recording or submission failed.
    #[error("Command submission error: {0}")]
    CommandSubmission(String),

    /// The swapchain is out of date and must be recreated.
    #[error("Swapchain out of date")]
    SwapchainOutOfDate,

    /// The surface was lost (e.g. window destroyed).
    #[error("Surface lost")]
    SurfaceLost,

    /// A device (GPU) was lost -- unrecoverable.
    #[error("Device lost")]
    DeviceLost,

    /// Out of GPU or host memory.
    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    /// A timeout expired while waiting for a GPU operation.
    #[error("GPU operation timed out")]
    Timeout,

    /// An invalid handle was passed to the device.
    #[error("Invalid resource handle")]
    InvalidHandle,

    /// Render graph contains a cycle or invalid dependency.
    #[error("Render graph error: {0}")]
    GraphError(String),

    /// Catch-all for backend-specific errors.
    #[error("Internal render error: {0}")]
    Internal(String),
}
