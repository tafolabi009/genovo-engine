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
pub mod render_graph;
pub mod render_pipeline;
pub mod texture_atlas;
pub mod culling;
pub mod debug_visualization;

// ---------------------------------------------------------------------------
// Extended rendering subsystems
// ---------------------------------------------------------------------------

// Volumetric cloud rendering with ray-marching, scattering, and temporal reprojection.
pub mod volumetric_clouds;

// GPU terrain rendering with clipmap LOD, tessellation, and material blending.
pub mod terrain_renderer;

// Screen-space global illumination with temporal accumulation and spatial denoising.
pub mod screen_space_gi;

// Per-pixel motion vector computation for TAA, motion blur, and temporal effects.
pub mod motion_vectors;

// Lightmap baking system with path tracing, denoising, and HDR encoding.
pub mod light_baker;

// Impostor/billboard baking with octahedral mapping and atlas generation.
pub mod impostor_baker;

// Selection/hover outline rendering using the Jump Flood Algorithm.
pub mod outline_renderer;

// Order-independent transparency: stochastic, WBOIT, depth peeling, linked lists.
pub mod stochastic_transparency;

// Camera lens simulation: flares, streaks, dirt, star-burst, and distortion.
pub mod lens_effects;

// GPU skeletal animation: bone palettes, LBS, DQS, compute hierarchy, morph targets.
pub mod gpu_skinning;

// ---------------------------------------------------------------------------
// Additional rendering subsystems (batch 2)
// ---------------------------------------------------------------------------

// SDF ray-marching: primitives, boolean ops, sphere tracing, AO, soft shadows.
pub mod raymarching;

// Comprehensive fog: linear, exponential, height fog, animated fog, fog volumes.
pub mod global_fog;

// Unified reflection: planar, cubemap, SSR, probe blending, parallax correction.
pub mod reflection_system;

// 2D sprite rendering: batching, atlas UVs, animation, nine-slice, pixel-perfect.
pub mod sprite_renderer;

// Anti-aliased line rendering: thick lines, dashes, caps, joins, bezier curves.
pub mod line_renderer;

// GPU text rendering: SDF fonts, glyph atlas, alignment, rich text, outlines.
pub mod text_renderer;

// Render target management: pooling, MRT, readback, mip generation.
pub mod render_targets;

// Shader preprocessing: defines, feature flags, variants, complexity analysis.
pub mod shader_defines;

// Advanced bloom: dual-filter, soft knee threshold, lens dirt, energy conservation.
pub mod bloom;

// Colour space conversions: sRGB, Rec.2020, DCI-P3, ACES, Oklab, CIE LAB/LCH, deltaE.
pub mod color_space;

// Image processing on GPU: blur, sharpen, edge detection, resize, histograms.
pub mod image_effects;

// Detailed render stats: per-pass timing, draw calls, memory, shader compiles.
pub mod render_statistics;

// Screenshot and video capture: PNG/TGA/BMP encoding, async readback, frame sequences.
pub mod screen_capture;

// Extended tone mapping: local tone mapping, HDR histogram, auto-exposure, EV100.
pub mod tone_map;

// Vertex data compression: position/UV quantization, octahedron normals, TBN quaternion.
pub mod vertex_compression;

// ---------------------------------------------------------------------------
// Additional rendering subsystems (batch 3)
// ---------------------------------------------------------------------------

// Generic temporal filtering: accumulation buffer, motion-vector reprojection,
// neighborhood clamping (AABB/variance), velocity rejection, sub-pixel jitter patterns.
pub mod temporal_filter;

// Indirect lighting collection: light probes, reflection probes, lightmaps, SSGI,
// volumetric probes; blend/weight all sources; fallback chain.
pub mod indirect_lighting;

// Mesh processing utilities: decimation (progressive), subdivision (Loop/Catmull-Clark),
// smoothing (Laplacian/Taubin), mesh boolean (CSG), welding, splitting by material.
pub mod geometry_processing;

// Batched wireframe boxes/spheres/lines/arrows for debug, instanced rendering,
// configurable line width, depth test toggle, color per primitive.
pub mod primitive_batch;

// Ambient lighting: flat, gradient (sky+ground), hemisphere (tri-directional),
// SH ambient, ambient from cubemap, ambient probe grid, AO integration.
pub mod ambient_system;

// ---------------------------------------------------------------------------
// Additional rendering subsystems (batch 4)
// ---------------------------------------------------------------------------

// Enhanced deferred: thin G-buffer (albedo+metallic in one RT, normal+roughness in one),
// stencil-based light volumes, light pre-pass (deferred lighting), tiled deferred,
// cluster debug visualization.
pub mod deferred;

// Ground Truth AO (GTAO): multi-bounce AO, bent normals, specular occlusion,
// temporal accumulation, spatial denoising, quality presets.
pub mod ground_truth_ao;

// Additional atmosphere: aurora borealis, rainbows, halos, sundog (parhelion),
// crepuscular rays (from volumetric light), heat haze (screen distortion), mirage effect.
pub mod atmospheric_effects;

// Enhanced procedural sky: physically-based sky with ozone layer, multiple scattering
// precomputation, aerial perspective LUT, planet rendering from space, ring system,
// nebula backdrop.
pub mod procedural_sky;

// Material layering: height-based blend between layers, detail materials (add
// wear/scratches), decal materials (project onto surfaces), material masks,
// parallax occlusion mapping, clearcoat layer.
pub mod material_layering;

// Enhanced shadows: contact shadows (screen-space ray march from light), area light
// shadows, shadow bias auto-tuning, shadow cache (don't re-render static shadows),
// shadow importance (skip shadows for distant/small lights).
pub mod shadow_system;

// Lightmap GI baking: progressive path tracing, UV2 unwrapping, denoising,
// HDR lightmap encoding, directional lightmaps, irradiance probe baking.
pub mod gi_baking;

// Hardware mesh instancing: per-instance data (transform, color, custom),
// instance buffer management, LOD per instance, frustum culling per batch.
pub mod mesh_instancing;

// GPU particle sorting: bitonic sort on GPU, depth-based sort keys,
// indirect dispatch, sort stability, multi-pass sorting.
pub mod particle_gpu_sort;

// Deferred decal rendering: project into G-buffer, normal blending,
// angle fade, lifetime management, sorting, material channels.
pub mod deferred_decals;

// GPU render profiler: timestamp queries per pass, pipeline statistics,
// frame graph timeline, bottleneck detection, multi-frame averaging.
pub mod render_profiler;

// Shadow cascade management: split schemes (uniform/log/PSSM), per-cascade viewport,
// cascade blending at boundaries, cascade stabilization, visualization colors,
// shadow quality per cascade.
pub mod cascade_selection;

// Texture resource caching: LRU texture pool, reference counting, texture sharing
// between materials, cache statistics, memory budget enforcement, eviction priorities.
pub mod texture_cache;

// Draw call optimization: sort by state (shader->material->mesh), merge compatible
// draw calls, indirect draw batching, instance merging, draw call statistics.
pub mod draw_call_optimizer;

// Shader hot-reloading: watch shader files, recompile on change, swap pipelines,
// error recovery (keep old shader on compile failure), shader edit history.
pub mod shader_hot_reload;

// Configurable post-process stack: ordered effect chain, per-effect enable/disable/
// weight, volume-based overrides (enter zone -> change effects), transition blending.
pub mod post_process_stack;

// Enhanced DOF: circular DOF with bokeh shapes, foreground/background separation,
// partial occlusion, smooth transitions, DOF from camera settings.
pub mod depth_of_field;

// Contact/screen-space shadows: ray-march from light direction in screen space,
// thickness estimation, soft contact shadows, temporal filtering.
pub mod screen_space_shadows;

// Static mesh merging: combine multiple static meshes into one draw call,
// per-material sub-meshes, bounding volume update, automatic merging.
pub mod mesh_merger;

// ---------------------------------------------------------------------------
// Scene renderer -- complete GPU rendering pipeline with PBR, primitives, grid.
// ---------------------------------------------------------------------------
pub mod scene_renderer;

// ---------------------------------------------------------------------------
// Additional rendering subsystems (batch 6)
// ---------------------------------------------------------------------------

// Forward+ (tiled forward) rendering pipeline: depth prepass, light assignment
// per tile (16x16 tiles), per-tile light list in SSBO, single-pass forward
// render with per-tile light loop, transparent objects support, tile debug
// visualization.
pub mod forward_plus;

// Compiled shader cache: hash shader source + defines, store compiled modules
// on disk, load from cache if hash matches, cache invalidation, cache size
// management, warm cache on startup.
pub mod shader_cache;

// Advanced render queue: sort by opaque (front-to-back), transparent
// (back-to-front), shadow casters (by cascade), sky (last), overlay (on top);
// priority override per material; queue clear/rebuild per frame.
pub mod render_queue;

// Material instances with GPU uniform buffers: per-material uniform buffer,
// dirty tracking, batch update, material parameter animation, material LOD
// (simplified shader at distance).
pub mod material_instance;

// G-Buffer configuration: configurable format (thin/standard/extended),
// encode/decode functions, octahedron normal mapping, bandwidth analysis.
pub mod gbuffer_layout;

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
