// engine/render/src/interface/pipeline.rs
//
// Pipeline state objects and related descriptor types. These mirror the
// configurable stages of a modern graphics pipeline while remaining
// API-agnostic.

use super::resource::{ShaderHandle, TextureFormat};
use smallvec::SmallVec;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Primitive topology used during input assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
    TriangleFan,
    LineListWithAdjacency,
    LineStripWithAdjacency,
    TriangleListWithAdjacency,
    TriangleStripWithAdjacency,
    PatchList,
}

/// Triangle face culling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CullMode {
    None,
    Front,
    Back,
    FrontAndBack,
}

/// Winding order that defines the front face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrontFace {
    CounterClockwise,
    Clockwise,
}

/// Polygon rasterisation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PolygonMode {
    Fill,
    Line,
    Point,
}

/// Comparison function used by depth/stencil tests and shadow samplers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareOp {
    Never,
    Less,
    Equal,
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    Always,
}

/// Stencil operation applied when a stencil test passes or fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StencilOp {
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
}

/// Blend factor for colour/alpha blending equations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    ConstantColor,
    OneMinusConstantColor,
    ConstantAlpha,
    OneMinusConstantAlpha,
    SrcAlphaSaturate,
    Src1Color,
    OneMinusSrc1Color,
    Src1Alpha,
    OneMinusSrc1Alpha,
}

/// Blend operation applied between source and destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

/// Vertex attribute data format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Sint32,
    Sint32x2,
    Sint32x3,
    Sint32x4,
    Uint32,
    Uint32x2,
    Uint32x3,
    Uint32x4,
    Sint16x2,
    Sint16x4,
    Uint16x2,
    Uint16x4,
    Snorm16x2,
    Snorm16x4,
    Unorm16x2,
    Unorm16x4,
    Sint8x2,
    Sint8x4,
    Uint8x2,
    Uint8x4,
    Snorm8x2,
    Snorm8x4,
    Unorm8x2,
    Unorm8x4,
    Unorm10_10_10_2,
}

/// Vertex input rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexStepMode {
    /// Advance per vertex.
    Vertex,
    /// Advance per instance.
    Instance,
}

/// Index buffer element size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexFormat {
    Uint16,
    Uint32,
}

/// Descriptor binding type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DescriptorType {
    UniformBuffer,
    StorageBuffer,
    SampledTexture,
    StorageTexture,
    Sampler,
    CombinedImageSampler,
    InputAttachment,
}

// ---------------------------------------------------------------------------
// Vertex input
// ---------------------------------------------------------------------------

/// Describes a single vertex attribute within a vertex buffer binding.
#[derive(Debug, Clone)]
pub struct VertexAttribute {
    /// Shader location index.
    pub location: u32,
    /// Data format.
    pub format: VertexFormat,
    /// Byte offset within the vertex.
    pub offset: u32,
}

/// Describes a single vertex buffer binding.
#[derive(Debug, Clone)]
pub struct VertexBufferLayout {
    /// Stride in bytes between consecutive vertices.
    pub stride: u32,
    /// Per-vertex or per-instance stepping.
    pub step_mode: VertexStepMode,
    /// Attributes sourced from this binding.
    pub attributes: Vec<VertexAttribute>,
}

/// Aggregated vertex input description.
#[derive(Debug, Clone, Default)]
pub struct VertexInputDesc {
    /// Buffer bindings.
    pub buffers: Vec<VertexBufferLayout>,
}

// ---------------------------------------------------------------------------
// Rasteriser state
// ---------------------------------------------------------------------------

/// Rasteriser configuration.
#[derive(Debug, Clone)]
pub struct RasterizerState {
    /// Polygon fill mode.
    pub polygon_mode: PolygonMode,
    /// Face culling mode.
    pub cull_mode: CullMode,
    /// Front-face winding order.
    pub front_face: FrontFace,
    /// Enable depth bias (polygon offset).
    pub depth_bias_enable: bool,
    /// Constant depth bias.
    pub depth_bias_constant: f32,
    /// Slope-scaled depth bias.
    pub depth_bias_slope: f32,
    /// Depth bias clamp.
    pub depth_bias_clamp: f32,
    /// Enable depth clamping (fragments beyond near/far are clamped, not clipped).
    pub depth_clamp_enable: bool,
    /// Line width for line primitives.
    pub line_width: f32,
}

impl Default for RasterizerState {
    fn default() -> Self {
        Self {
            polygon_mode: PolygonMode::Fill,
            cull_mode: CullMode::Back,
            front_face: FrontFace::CounterClockwise,
            depth_bias_enable: false,
            depth_bias_constant: 0.0,
            depth_bias_slope: 0.0,
            depth_bias_clamp: 0.0,
            depth_clamp_enable: false,
            line_width: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Depth / stencil state
// ---------------------------------------------------------------------------

/// Per-face stencil operation descriptor.
#[derive(Debug, Clone)]
pub struct StencilFaceState {
    /// Operation on stencil fail.
    pub fail_op: StencilOp,
    /// Operation on depth fail.
    pub depth_fail_op: StencilOp,
    /// Operation on pass.
    pub pass_op: StencilOp,
    /// Comparison function.
    pub compare: CompareOp,
    /// Read mask.
    pub read_mask: u32,
    /// Write mask.
    pub write_mask: u32,
    /// Reference value.
    pub reference: u32,
}

impl Default for StencilFaceState {
    fn default() -> Self {
        Self {
            fail_op: StencilOp::Keep,
            depth_fail_op: StencilOp::Keep,
            pass_op: StencilOp::Keep,
            compare: CompareOp::Always,
            read_mask: 0xFF,
            write_mask: 0xFF,
            reference: 0,
        }
    }
}

/// Depth and stencil testing configuration.
#[derive(Debug, Clone)]
pub struct DepthStencilState {
    /// Enable depth testing.
    pub depth_test_enable: bool,
    /// Enable depth writes.
    pub depth_write_enable: bool,
    /// Depth comparison function.
    pub depth_compare: CompareOp,
    /// Enable stencil testing.
    pub stencil_test_enable: bool,
    /// Front-face stencil operations.
    pub stencil_front: StencilFaceState,
    /// Back-face stencil operations.
    pub stencil_back: StencilFaceState,
}

impl Default for DepthStencilState {
    fn default() -> Self {
        Self {
            depth_test_enable: true,
            depth_write_enable: true,
            depth_compare: CompareOp::Less,
            stencil_test_enable: false,
            stencil_front: StencilFaceState::default(),
            stencil_back: StencilFaceState::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Blend state
// ---------------------------------------------------------------------------

/// Per-attachment colour blend configuration.
#[derive(Debug, Clone)]
pub struct ColorTargetState {
    /// Format of this colour target.
    pub format: TextureFormat,
    /// Enable blending on this target.
    pub blend_enable: bool,
    /// Source colour factor.
    pub src_color_blend_factor: BlendFactor,
    /// Destination colour factor.
    pub dst_color_blend_factor: BlendFactor,
    /// Colour blend operation.
    pub color_blend_op: BlendOp,
    /// Source alpha factor.
    pub src_alpha_blend_factor: BlendFactor,
    /// Destination alpha factor.
    pub dst_alpha_blend_factor: BlendFactor,
    /// Alpha blend operation.
    pub alpha_blend_op: BlendOp,
    /// Colour write mask (RGBA bits).
    pub write_mask: u8,
}

impl Default for ColorTargetState {
    fn default() -> Self {
        Self {
            format: TextureFormat::Rgba8Unorm,
            blend_enable: false,
            src_color_blend_factor: BlendFactor::One,
            dst_color_blend_factor: BlendFactor::Zero,
            color_blend_op: BlendOp::Add,
            src_alpha_blend_factor: BlendFactor::One,
            dst_alpha_blend_factor: BlendFactor::Zero,
            alpha_blend_op: BlendOp::Add,
            write_mask: 0xF, // RGBA
        }
    }
}

/// Aggregate blend state across all colour attachments.
#[derive(Debug, Clone, Default)]
pub struct BlendState {
    /// Per-target configurations.
    pub targets: Vec<ColorTargetState>,
    /// Constant blend colour.
    pub blend_constant: [f32; 4],
}

// ---------------------------------------------------------------------------
// Multisample state
// ---------------------------------------------------------------------------

/// MSAA configuration.
#[derive(Debug, Clone)]
pub struct MultisampleState {
    /// Number of samples per pixel.
    pub count: u32,
    /// Sample mask.
    pub mask: u64,
    /// Enable alpha-to-coverage.
    pub alpha_to_coverage_enable: bool,
}

impl Default for MultisampleState {
    fn default() -> Self {
        Self {
            count: 1,
            mask: !0,
            alpha_to_coverage_enable: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Descriptor set / Pipeline layout
// ---------------------------------------------------------------------------

/// Single binding within a descriptor set layout.
#[derive(Debug, Clone)]
pub struct DescriptorSetLayoutBinding {
    /// Binding index.
    pub binding: u32,
    /// Descriptor type.
    pub descriptor_type: DescriptorType,
    /// Number of descriptors at this binding (arrays).
    pub count: u32,
    /// Shader stages that may access this binding.
    pub stage_flags: ShaderStageFlags,
}

/// Layout of a single descriptor set.
#[derive(Debug, Clone)]
pub struct DescriptorSetLayout {
    /// Bindings within this set.
    pub bindings: Vec<DescriptorSetLayoutBinding>,
}

/// Push constant range visible to specific shader stages.
#[derive(Debug, Clone)]
pub struct PushConstantRange {
    /// Shader stage flags.
    pub stage_flags: ShaderStageFlags,
    /// Byte offset.
    pub offset: u32,
    /// Size in bytes.
    pub size: u32,
}

/// Layout of a full pipeline (descriptor sets + push constants).
#[derive(Debug, Clone)]
pub struct PipelineLayout {
    /// Descriptor set layouts, in set-index order.
    pub descriptor_set_layouts: Vec<DescriptorSetLayout>,
    /// Push constant ranges.
    pub push_constant_ranges: Vec<PushConstantRange>,
}

/// Opaque handle to a bound descriptor set.
#[derive(Debug, Clone)]
pub struct DescriptorSet {
    /// Internal identifier used by the backend.
    pub(crate) id: u64,
}

// ---------------------------------------------------------------------------
// Shader stage flags (bitflags)
// ---------------------------------------------------------------------------

use bitflags::bitflags;

bitflags! {
    /// Shader stage visibility flags for descriptor bindings.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ShaderStageFlags: u32 {
        const VERTEX                  = 1 << 0;
        const FRAGMENT                = 1 << 1;
        const COMPUTE                 = 1 << 2;
        const GEOMETRY                = 1 << 3;
        const TESSELLATION_CONTROL    = 1 << 4;
        const TESSELLATION_EVALUATION = 1 << 5;
        const MESH                    = 1 << 6;
        const TASK                    = 1 << 7;
        const ALL_GRAPHICS = Self::VERTEX.bits()
            | Self::FRAGMENT.bits()
            | Self::GEOMETRY.bits()
            | Self::TESSELLATION_CONTROL.bits()
            | Self::TESSELLATION_EVALUATION.bits();
        const ALL = Self::ALL_GRAPHICS.bits() | Self::COMPUTE.bits()
            | Self::MESH.bits() | Self::TASK.bits();
    }
}

// ---------------------------------------------------------------------------
// Shader stage descriptor (per-stage in a pipeline)
// ---------------------------------------------------------------------------

/// Describes a shader stage within a pipeline.
#[derive(Debug, Clone)]
pub struct ShaderStageDesc {
    /// Handle to the compiled shader module.
    pub module: ShaderHandle,
    /// Entry-point function name.
    pub entry_point: String,
    /// Specialisation constant overrides (id -> bytes).
    pub specialization_constants: Vec<(u32, Vec<u8>)>,
}

// ---------------------------------------------------------------------------
// Graphics pipeline
// ---------------------------------------------------------------------------

/// Complete descriptor for creating a graphics pipeline.
#[derive(Debug, Clone)]
pub struct GraphicsPipelineDesc {
    /// Human-readable label for debug tooling.
    pub label: Option<String>,
    /// Pipeline layout (descriptor sets + push constants).
    pub layout: PipelineLayout,
    /// Vertex shader stage.
    pub vertex_stage: ShaderStageDesc,
    /// Optional fragment shader stage.
    pub fragment_stage: Option<ShaderStageDesc>,
    /// Vertex input state.
    pub vertex_input: VertexInputDesc,
    /// Primitive topology.
    pub primitive_topology: PrimitiveTopology,
    /// Rasteriser state.
    pub rasterizer: RasterizerState,
    /// Depth/stencil state.
    pub depth_stencil: DepthStencilState,
    /// Blend state.
    pub blend: BlendState,
    /// Multisample state.
    pub multisample: MultisampleState,
    /// Colour attachment formats (for dynamic rendering / render-pass-less).
    pub color_attachment_formats: SmallVec<[TextureFormat; 4]>,
    /// Depth attachment format.
    pub depth_attachment_format: Option<TextureFormat>,
    /// Stencil attachment format.
    pub stencil_attachment_format: Option<TextureFormat>,
}

// ---------------------------------------------------------------------------
// Compute pipeline
// ---------------------------------------------------------------------------

/// Complete descriptor for creating a compute pipeline.
#[derive(Debug, Clone)]
pub struct ComputePipelineDesc {
    /// Human-readable label for debug tooling.
    pub label: Option<String>,
    /// Pipeline layout (descriptor sets + push constants).
    pub layout: PipelineLayout,
    /// Compute shader stage.
    pub compute_stage: ShaderStageDesc,
}
