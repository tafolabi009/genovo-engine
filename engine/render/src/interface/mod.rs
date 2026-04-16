// engine/render/src/interface/mod.rs
//
// Abstract rendering API module. Contains the backend-agnostic traits and type
// definitions that every GPU backend must satisfy.

pub mod command_buffer;
pub mod device;
pub mod pipeline;
pub mod resource;

// Re-export the most commonly used types at the interface level for
// convenience.
pub use command_buffer::{
    AccessFlags, BufferCopyRegion, ClearValue, CommandBuffer, CommandEncoder, MemoryBarrier,
    PipelineStage, RenderPassBeginInfo, RenderPassEncoder, ScissorRect, TextureCopyRegion,
    Viewport,
};
pub use device::{DeviceCapabilities, QueueType, RenderDevice, Result};
pub use pipeline::{
    BlendFactor, BlendOp, BlendState, ColorTargetState, CompareOp, ComputePipelineDesc,
    CullMode, DepthStencilState, DescriptorSet, DescriptorSetLayout, DescriptorSetLayoutBinding,
    DescriptorType, FrontFace, GraphicsPipelineDesc, IndexFormat, MultisampleState,
    PipelineLayout, PolygonMode, PrimitiveTopology, PushConstantRange, RasterizerState,
    ShaderStageDesc, ShaderStageFlags, StencilFaceState, StencilOp, VertexAttribute,
    VertexBufferLayout, VertexFormat, VertexInputDesc, VertexStepMode,
};
pub use resource::{
    AddressMode, AttachmentDesc, BufferDesc, BufferHandle, BufferUsage, FenceHandle, FilterMode,
    FramebufferDesc, FramebufferHandle, LoadOp, MemoryLocation, PipelineHandle, RenderPassDesc,
    RenderPassHandle, SamplerDesc, SamplerHandle, SemaphoreHandle, ShaderDesc, ShaderHandle,
    ShaderStage, StoreOp, TextureDesc, TextureDimension, TextureFormat, TextureHandle,
    TextureUsage,
};
