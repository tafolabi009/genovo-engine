// engine/render/src/interface/command_buffer.rs
//
// Command buffer recording abstraction. Commands are recorded into a
// `CommandBuffer` through the `CommandEncoder` trait, then submitted to a
// queue via `RenderDevice::submit_commands`.

use super::pipeline::{DescriptorSet, IndexFormat};
use super::resource::{
    BufferHandle, FramebufferHandle, PipelineHandle, RenderPassHandle, TextureHandle,
};
use crate::RenderError;
use glam::UVec3;

// ---------------------------------------------------------------------------
// Viewport / Scissor
// ---------------------------------------------------------------------------

/// Viewport rectangle with depth range.
#[derive(Debug, Clone, Copy)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

/// Scissor rectangle (integer pixel coordinates).
#[derive(Debug, Clone, Copy)]
pub struct ScissorRect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

// ---------------------------------------------------------------------------
// Clear values
// ---------------------------------------------------------------------------

/// Value used to clear an attachment at the beginning of a render pass.
#[derive(Debug, Clone, Copy)]
pub enum ClearValue {
    /// RGBA floating-point colour.
    Color([f32; 4]),
    /// Depth + stencil.
    DepthStencil { depth: f32, stencil: u32 },
}

// ---------------------------------------------------------------------------
// Pipeline barrier
// ---------------------------------------------------------------------------

/// Coarse pipeline stage flags for barrier synchronisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineStage {
    TopOfPipe,
    DrawIndirect,
    VertexInput,
    VertexShader,
    FragmentShader,
    EarlyFragmentTests,
    LateFragmentTests,
    ColorAttachmentOutput,
    ComputeShader,
    Transfer,
    BottomOfPipe,
    AllGraphics,
    AllCommands,
}

/// Memory access flags for barrier specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessFlags {
    None,
    IndirectCommandRead,
    IndexRead,
    VertexAttributeRead,
    UniformRead,
    InputAttachmentRead,
    ShaderRead,
    ShaderWrite,
    ColorAttachmentRead,
    ColorAttachmentWrite,
    DepthStencilAttachmentRead,
    DepthStencilAttachmentWrite,
    TransferRead,
    TransferWrite,
    HostRead,
    HostWrite,
    MemoryRead,
    MemoryWrite,
}

/// A memory barrier between pipeline stages.
#[derive(Debug, Clone, Copy)]
pub struct MemoryBarrier {
    pub src_stage: PipelineStage,
    pub dst_stage: PipelineStage,
    pub src_access: AccessFlags,
    pub dst_access: AccessFlags,
}

/// Buffer copy region.
#[derive(Debug, Clone, Copy)]
pub struct BufferCopyRegion {
    pub src_offset: u64,
    pub dst_offset: u64,
    pub size: u64,
}

/// Texture copy region.
#[derive(Debug, Clone, Copy)]
pub struct TextureCopyRegion {
    pub src_offset: UVec3,
    pub dst_offset: UVec3,
    pub extent: UVec3,
    pub src_mip_level: u32,
    pub dst_mip_level: u32,
    pub src_array_layer: u32,
    pub dst_array_layer: u32,
}

// ---------------------------------------------------------------------------
// Render pass begin info
// ---------------------------------------------------------------------------

/// Parameters for beginning a render pass inside a command buffer.
#[derive(Debug, Clone)]
pub struct RenderPassBeginInfo {
    /// Render pass handle.
    pub render_pass: RenderPassHandle,
    /// Framebuffer to render into.
    pub framebuffer: FramebufferHandle,
    /// Render area offset X.
    pub x: i32,
    /// Render area offset Y.
    pub y: i32,
    /// Render area width.
    pub width: u32,
    /// Render area height.
    pub height: u32,
    /// Clear values for each attachment, in order.
    pub clear_values: Vec<ClearValue>,
}

// ---------------------------------------------------------------------------
// CommandBuffer
// ---------------------------------------------------------------------------

/// A recorded command buffer ready for submission.
///
/// Command buffers are opaque containers of GPU commands. They are created and
/// recorded through a backend-specific `CommandEncoder` implementation, then
/// submitted as a batch via [`RenderDevice::submit_commands`].
#[derive(Debug)]
pub struct CommandBuffer {
    /// Backend-specific data. The engine treats this as opaque; each backend
    /// downcasts through its own internal types.
    pub(crate) inner: CommandBufferInner,
}

/// Backend-private command buffer storage.
///
/// This enum is extended per backend behind feature gates. All backends
/// currently delegate to wgpu, so the `Wgpu` variant is the primary path.
/// The `Empty` variant is retained for unit testing and headless scenarios
/// where no GPU commands are actually recorded.
#[derive(Debug)]
pub(crate) enum CommandBufferInner {
    /// Placeholder for headless or test scenarios.
    Empty,
    /// Wraps a finished wgpu command buffer, used by the Vulkan, DX12, and
    /// Metal wgpu-delegating backends as well as the primary wgpu backend.
    Wgpu(Option<wgpu::CommandBuffer>),
}

impl CommandBuffer {
    /// Create an empty placeholder command buffer.
    pub(crate) fn empty() -> Self {
        Self {
            inner: CommandBufferInner::Empty,
        }
    }

    /// Create a command buffer wrapping a finished wgpu command buffer.
    pub(crate) fn from_wgpu(cmd: wgpu::CommandBuffer) -> Self {
        Self {
            inner: CommandBufferInner::Wgpu(Some(cmd)),
        }
    }

    /// Take the inner wgpu command buffer, if present.
    pub(crate) fn take_wgpu(&mut self) -> Option<wgpu::CommandBuffer> {
        match &mut self.inner {
            CommandBufferInner::Wgpu(cmd) => cmd.take(),
            CommandBufferInner::Empty => None,
        }
    }
}

// ---------------------------------------------------------------------------
// CommandEncoder trait
// ---------------------------------------------------------------------------

/// Trait for recording GPU commands into a [`CommandBuffer`].
///
/// A `CommandEncoder` is obtained from the backend, records a sequence of
/// commands, and then produces a finished `CommandBuffer` via
/// [`finish`](Self::finish).
pub trait CommandEncoder {
    /// Finalise recording and produce a submittable command buffer.
    fn finish(self) -> std::result::Result<CommandBuffer, RenderError>;

    // -- Render pass commands -----------------------------------------------

    /// Begin a render pass. All draw commands between `begin_render_pass` and
    /// `end_render_pass` are scoped to the specified framebuffer.
    fn begin_render_pass(&mut self, info: &RenderPassBeginInfo);

    /// End the current render pass.
    fn end_render_pass(&mut self);

    // -- Pipeline binding ---------------------------------------------------

    /// Bind a graphics or compute pipeline.
    fn bind_pipeline(&mut self, pipeline: PipelineHandle);

    // -- Vertex / index buffers ---------------------------------------------

    /// Bind a vertex buffer at the given slot.
    fn bind_vertex_buffer(&mut self, slot: u32, buffer: BufferHandle, offset: u64);

    /// Bind an index buffer.
    fn bind_index_buffer(&mut self, buffer: BufferHandle, offset: u64, format: IndexFormat);

    // -- Descriptor sets ----------------------------------------------------

    /// Bind a descriptor set at the given set index.
    fn bind_descriptor_set(&mut self, set_index: u32, descriptor_set: &DescriptorSet);

    // -- Draw / dispatch ----------------------------------------------------

    /// Record a non-indexed draw call.
    fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32);

    /// Record an indexed draw call.
    fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    );

    /// Record a compute dispatch.
    fn dispatch_compute(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32);

    // -- Transfer -----------------------------------------------------------

    /// Copy data between buffers.
    fn copy_buffer(
        &mut self,
        src: BufferHandle,
        dst: BufferHandle,
        regions: &[BufferCopyRegion],
    );

    /// Copy data between textures.
    fn copy_texture(
        &mut self,
        src: TextureHandle,
        dst: TextureHandle,
        regions: &[TextureCopyRegion],
    );

    // -- Dynamic state ------------------------------------------------------

    /// Set the viewport.
    fn set_viewport(&mut self, viewport: &Viewport);

    /// Set the scissor rectangle.
    fn set_scissor(&mut self, scissor: &ScissorRect);

    // -- Synchronisation ----------------------------------------------------

    /// Insert a pipeline barrier for memory / execution dependency.
    fn pipeline_barrier(&mut self, barriers: &[MemoryBarrier]);

    // -- Push constants -----------------------------------------------------

    /// Upload push-constant data for the currently bound pipeline.
    fn push_constants(&mut self, offset: u32, data: &[u8]);
}

// ---------------------------------------------------------------------------
// RenderPassEncoder
// ---------------------------------------------------------------------------

/// A scoped encoder that only permits commands valid inside a render pass.
///
/// This type is a safety wrapper: it borrows a `CommandEncoder` for the
/// duration of a render pass and exposes only the subset of commands that are
/// legal within one.
pub struct RenderPassEncoder<'a> {
    encoder: &'a mut dyn CommandEncoder,
}

impl<'a> RenderPassEncoder<'a> {
    /// Create a new render-pass-scoped encoder.
    ///
    /// Callers must ensure `begin_render_pass` has already been called on the
    /// underlying encoder.
    pub fn new(encoder: &'a mut dyn CommandEncoder) -> Self {
        Self { encoder }
    }

    /// Bind a graphics pipeline.
    pub fn bind_pipeline(&mut self, pipeline: PipelineHandle) {
        self.encoder.bind_pipeline(pipeline);
    }

    /// Bind a vertex buffer at the given slot.
    pub fn bind_vertex_buffer(&mut self, slot: u32, buffer: BufferHandle, offset: u64) {
        self.encoder.bind_vertex_buffer(slot, buffer, offset);
    }

    /// Bind an index buffer.
    pub fn bind_index_buffer(&mut self, buffer: BufferHandle, offset: u64, format: IndexFormat) {
        self.encoder.bind_index_buffer(buffer, offset, format);
    }

    /// Bind a descriptor set.
    pub fn bind_descriptor_set(&mut self, set_index: u32, descriptor_set: &DescriptorSet) {
        self.encoder.bind_descriptor_set(set_index, descriptor_set);
    }

    /// Non-indexed draw.
    pub fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) {
        self.encoder.draw(vertex_count, instance_count, first_vertex, first_instance);
    }

    /// Indexed draw.
    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        self.encoder.draw_indexed(index_count, instance_count, first_index, vertex_offset, first_instance);
    }

    /// Set the viewport.
    pub fn set_viewport(&mut self, viewport: &Viewport) {
        self.encoder.set_viewport(viewport);
    }

    /// Set the scissor rectangle.
    pub fn set_scissor(&mut self, scissor: &ScissorRect) {
        self.encoder.set_scissor(scissor);
    }

    /// Upload push-constant data.
    pub fn push_constants(&mut self, offset: u32, data: &[u8]) {
        self.encoder.push_constants(offset, data);
    }
}
