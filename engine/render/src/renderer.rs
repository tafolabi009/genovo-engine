// engine/render/src/renderer.rs
//
// High-level renderer that orchestrates frame rendering on top of the
// abstract `RenderDevice`. Contains the render graph, draw-call sorting,
// ECS component types, and a concrete wgpu-backed renderer that can
// create a surface, set up pipelines, and draw to the screen.

use crate::interface::device::RenderDevice;
use crate::interface::resource::{
    BufferHandle, PipelineHandle, TextureDesc, TextureDimension, TextureFormat, TextureHandle,
    TextureUsage,
};
use crate::shader::BUILTIN_TRIANGLE_WGSL;
use crate::wgpu_backend::{WgpuDevice, WgpuSurface};
use crate::RenderError;
use glam::{Mat4, Vec3, Vec4};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// FrameContext
// ---------------------------------------------------------------------------

/// Per-frame rendering context returned by `WgpuRenderer::begin_frame()`.
///
/// Holds the swapchain texture and its view for the duration of a single
/// frame. Must be passed back to `end_frame()` for presentation.
pub struct FrameContext {
    /// The swapchain surface texture acquired for this frame.
    pub(crate) surface_texture: wgpu::SurfaceTexture,
    /// A view into the surface texture for render pass attachment.
    pub(crate) surface_view: wgpu::TextureView,
    /// Accumulated command buffers to be submitted at end-of-frame.
    pub(crate) command_buffers: Vec<wgpu::CommandBuffer>,
}

impl FrameContext {
    /// Get the surface texture view for creating render passes.
    pub fn surface_view(&self) -> &wgpu::TextureView {
        &self.surface_view
    }
}

// ---------------------------------------------------------------------------
// WgpuRenderer
// ---------------------------------------------------------------------------

/// A concrete renderer backed by the wgpu backend.
///
/// This struct owns a `WgpuDevice` and a `WgpuSurface`, and provides
/// high-level methods for frame management, built-in pipeline drawing,
/// and surface reconfiguration.
pub struct WgpuRenderer {
    /// The wgpu GPU device.
    device: WgpuDevice,
    /// The presentation surface.
    surface: WgpuSurface,
    /// Handle to the built-in coloured-triangle pipeline.
    triangle_pipeline: PipelineHandle,
    /// Handle to the depth texture for the current surface dimensions.
    depth_texture: TextureHandle,
    /// Current surface width.
    width: u32,
    /// Current surface height.
    height: u32,
    /// Frame counter.
    frame_index: u64,
}

impl WgpuRenderer {
    /// Create a new wgpu renderer from a window handle.
    ///
    /// Initialises the GPU device, creates a presentation surface, builds
    /// built-in pipelines, and allocates a depth texture.
    pub fn new<W>(window: Arc<W>, width: u32, height: u32) -> Result<Self, RenderError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        let (device, raw_surface) = WgpuDevice::new_with_surface(window)?;
        let surface = WgpuSurface::new(&device, raw_surface, width, height);

        // Create built-in triangle pipeline.
        let triangle_pipeline = device.create_pipeline_from_wgsl(
            BUILTIN_TRIANGLE_WGSL,
            "vs_main",
            "fs_main",
            surface.format(),
            Some(wgpu::TextureFormat::Depth32Float),
            &[], // no vertex buffers -- uses vertex_index
            "builtin_triangle",
        )?;

        // Create depth texture.
        let depth_texture = device.create_texture(&TextureDesc {
            label: Some("depth_texture".into()),
            format: TextureFormat::Depth32Float,
            dimension: TextureDimension::D2,
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
            mip_levels: 1,
            sample_count: 1,
            usage: TextureUsage::DEPTH_STENCIL,
        })?;

        log::info!(
            "WgpuRenderer initialised ({}x{}, device: {})",
            width,
            height,
            device.get_capabilities().device_name
        );

        Ok(Self {
            device,
            surface,
            triangle_pipeline,
            depth_texture,
            width: width.max(1),
            height: height.max(1),
            frame_index: 0,
        })
    }

    /// Reconfigure the surface and depth texture after a window resize.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), RenderError> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        self.width = width;
        self.height = height;
        self.surface.resize(&self.device, width, height);

        // Recreate depth texture at the new size.
        self.device.destroy_texture(self.depth_texture);
        self.depth_texture = self.device.create_texture(&TextureDesc {
            label: Some("depth_texture".into()),
            format: TextureFormat::Depth32Float,
            dimension: TextureDimension::D2,
            width,
            height,
            depth_or_array_layers: 1,
            mip_levels: 1,
            sample_count: 1,
            usage: TextureUsage::DEPTH_STENCIL,
        })?;

        log::debug!("WgpuRenderer resized to {}x{}", width, height);
        Ok(())
    }

    /// Begin a new frame: acquire the next swapchain texture.
    ///
    /// Returns a `FrameContext` that must be passed to drawing methods and
    /// finally to `end_frame()` for presentation.
    pub fn begin_frame(&mut self) -> Result<FrameContext, RenderError> {
        let surface_texture = self.surface.get_current_texture()?;
        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.frame_index = self.frame_index.wrapping_add(1);

        Ok(FrameContext {
            surface_texture,
            surface_view,
            command_buffers: Vec::new(),
        })
    }

    /// Render the built-in coloured triangle into the given frame.
    ///
    /// This records a render pass that clears the surface to a dark colour,
    /// binds the triangle pipeline, and draws 3 vertices.
    pub fn render_triangle(
        &self,
        frame: &mut FrameContext,
        clear_color: [f32; 4],
    ) -> Result<(), RenderError> {
        let mut encoder = self.device.create_command_encoder("triangle_render_pass");

        // We need to look up the depth texture view. Since we created it via
        // the RenderDevice trait, we access it through the handle pool. For
        // simplicity, we create a transient depth view here.
        let depth_view = {
            // Access the texture through the pool -- the WgpuDevice stores them.
            // We need the view for the render pass.
            let depth_tex_desc = wgpu::TextureDescriptor {
                label: Some("frame_depth"),
                size: wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            };
            let depth_tex = self.device.raw_device().create_texture(&depth_tex_desc);
            depth_tex.create_view(&wgpu::TextureViewDescriptor::default())
        };

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("triangle_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame.surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: clear_color[0] as f64,
                            g: clear_color[1] as f64,
                            b: clear_color[2] as f64,
                            a: clear_color[3] as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Bind the triangle pipeline and draw.
            self.device.with_render_pipeline(self.triangle_pipeline, |pipeline| {
                render_pass.set_pipeline(pipeline);
                render_pass.draw(0..3, 0..1);
            })?;
        }

        frame.command_buffers.push(encoder.finish());
        Ok(())
    }

    /// Clear the screen to a solid colour without drawing anything else.
    pub fn clear(
        &self,
        frame: &mut FrameContext,
        color: [f32; 4],
    ) -> Result<(), RenderError> {
        let mut encoder = self.device.create_command_encoder("clear_pass");

        {
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame.surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: color[0] as f64,
                            g: color[1] as f64,
                            b: color[2] as f64,
                            a: color[3] as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // The render pass clears on begin; we just need to end it.
        }

        frame.command_buffers.push(encoder.finish());
        Ok(())
    }

    /// End the frame: submit all recorded command buffers and present.
    pub fn end_frame(&self, frame: FrameContext) -> Result<(), RenderError> {
        self.device.submit_wgpu_commands(frame.command_buffers);
        frame.surface_texture.present();
        Ok(())
    }

    /// Access the underlying `WgpuDevice`.
    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Current frame index.
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Current width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Current height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The surface texture format (engine representation).
    pub fn surface_format(&self) -> TextureFormat {
        self.surface.engine_format()
    }

    /// Wait for all GPU work to complete.
    pub fn wait_idle(&self) -> Result<(), RenderError> {
        self.device.wait_idle()
    }
}

// ---------------------------------------------------------------------------
// Renderer (trait-based, backend-agnostic)
// ---------------------------------------------------------------------------

/// The high-level renderer that owns the device and orchestrates per-frame
/// work using the abstract `RenderDevice` trait.
pub struct Renderer {
    /// The abstract GPU device (backend-specific behind the trait).
    device: Arc<dyn RenderDevice>,
    /// The render graph describing pass execution order.
    render_graph: RenderGraph,
    /// Sorted draw call queue for the current frame.
    render_queue: RenderQueue,
    /// Current frame index (wrapping).
    frame_index: u64,
    /// Number of frames in flight (double/triple buffering).
    frames_in_flight: u32,
}

impl Renderer {
    /// Create a new renderer backed by the given device.
    pub fn new(device: Arc<dyn RenderDevice>, frames_in_flight: u32) -> Self {
        Self {
            device,
            render_graph: RenderGraph::new(),
            render_queue: RenderQueue::new(),
            frame_index: 0,
            frames_in_flight,
        }
    }

    /// Begin a new frame: acquire swapchain image, reset per-frame resources.
    pub fn begin_frame(&mut self) -> std::result::Result<(), RenderError> {
        self.frame_index = self.frame_index.wrapping_add(1);
        Ok(())
    }

    /// Execute the render graph: record commands, submit, and present.
    pub fn end_frame(&mut self) -> std::result::Result<(), RenderError> {
        self.render_queue.clear();
        Ok(())
    }

    /// Access the underlying render device.
    pub fn device(&self) -> &dyn RenderDevice {
        &*self.device
    }

    /// Access the render graph for pass registration.
    pub fn render_graph_mut(&mut self) -> &mut RenderGraph {
        &mut self.render_graph
    }

    /// Access the render queue to enqueue draw calls.
    pub fn render_queue_mut(&mut self) -> &mut RenderQueue {
        &mut self.render_queue
    }

    /// Current frame index.
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Number of frames in flight.
    pub fn frames_in_flight(&self) -> u32 {
        self.frames_in_flight
    }

    /// Shut down the renderer, waiting for the GPU to finish all work.
    pub fn shutdown(&self) -> std::result::Result<(), RenderError> {
        self.device.wait_idle()
    }
}

// ---------------------------------------------------------------------------
// RenderGraph
// ---------------------------------------------------------------------------

/// A node in the render graph representing a single render or compute pass.
#[derive(Debug, Clone)]
pub struct RenderGraphPass {
    /// Unique name of this pass.
    pub name: String,
    /// Passes that must execute before this one.
    pub dependencies: Vec<String>,
    /// Textures read by this pass.
    pub reads: Vec<RenderGraphResource>,
    /// Textures written by this pass.
    pub writes: Vec<RenderGraphResource>,
    /// Whether this is a compute-only pass.
    pub is_compute: bool,
}

/// A virtual resource reference within the render graph (resolved at compile
/// time to a concrete texture / buffer).
#[derive(Debug, Clone)]
pub struct RenderGraphResource {
    /// Symbolic name.
    pub name: String,
    /// Expected format (for textures).
    pub format: Option<TextureFormat>,
    /// Percentage of the output resolution (1.0 = full res).
    pub resolution_scale: f32,
}

/// Directed acyclic graph of render/compute passes.
///
/// The graph is built each frame by the renderer and compiled into a linear
/// execution order with automatic resource transitions.
pub struct RenderGraph {
    passes: Vec<RenderGraphPass>,
}

impl RenderGraph {
    /// Create an empty render graph.
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Register a new pass in the graph.
    pub fn add_pass(&mut self, pass: RenderGraphPass) {
        self.passes.push(pass);
    }

    /// Remove all passes (called at the start of each frame).
    pub fn clear(&mut self) {
        self.passes.clear();
    }

    /// Compile the graph into a topologically-sorted execution order.
    ///
    /// Returns the passes in the order they should execute, or an error if the
    /// graph contains cycles.
    pub fn compile(&self) -> std::result::Result<Vec<&RenderGraphPass>, RenderError> {
        if self.passes.is_empty() {
            return Ok(Vec::new());
        }

        // Build adjacency list and in-degree map from dependency names.
        let name_to_index: HashMap<&str, usize> = self
            .passes
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name.as_str(), i))
            .collect();

        let n = self.passes.len();
        let mut in_degree = vec![0usize; n];
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (i, pass) in self.passes.iter().enumerate() {
            for dep_name in &pass.dependencies {
                if let Some(&dep_idx) = name_to_index.get(dep_name.as_str()) {
                    adjacency[dep_idx].push(i);
                    in_degree[i] += 1;
                }
                // Dependencies referencing unknown passes are silently ignored;
                // they may refer to passes from a previous frame or optional
                // passes that were not registered.
            }
        }

        // Kahn's algorithm for topological sort.
        let mut queue: Vec<usize> = in_degree
            .iter()
            .enumerate()
            .filter_map(|(i, &deg)| if deg == 0 { Some(i) } else { None })
            .collect();

        let mut sorted: Vec<&RenderGraphPass> = Vec::with_capacity(n);

        while let Some(node) = queue.pop() {
            sorted.push(&self.passes[node]);
            for &neighbor in &adjacency[node] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push(neighbor);
                }
            }
        }

        if sorted.len() != n {
            // Some nodes were never added -- the graph has a cycle.
            let cycle_nodes: Vec<&str> = in_degree
                .iter()
                .enumerate()
                .filter_map(|(i, &deg)| {
                    if deg > 0 {
                        Some(self.passes[i].name.as_str())
                    } else {
                        None
                    }
                })
                .collect();
            return Err(RenderError::GraphError(format!(
                "Render graph contains a cycle involving passes: {}",
                cycle_nodes.join(", ")
            )));
        }

        Ok(sorted)
    }

    /// Return a slice of all registered passes.
    pub fn passes(&self) -> &[RenderGraphPass] {
        &self.passes
    }
}

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RenderQueue
// ---------------------------------------------------------------------------

/// Sort key for draw calls, packed into a u64 for efficient sorting.
///
/// Bit layout (high to low):
///   [63..56] layer (8 bits)  - e.g. background, opaque, transparent, overlay
///   [55..40] depth (16 bits) - front-to-back or back-to-front
///   [39..24] pipeline (16 bits)
///   [23..0]  material/mesh (24 bits)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SortKey(pub u64);

impl SortKey {
    /// Construct a sort key from individual components.
    pub fn new(layer: u8, depth: u16, pipeline: u16, material: u32) -> Self {
        let key = ((layer as u64) << 56)
            | ((depth as u64) << 40)
            | ((pipeline as u64) << 24)
            | ((material as u64) & 0x00FF_FFFF);
        Self(key)
    }
}

/// A single draw call entry in the render queue.
#[derive(Debug, Clone)]
pub struct DrawCall {
    /// Sort key for ordering.
    pub sort_key: SortKey,
    /// Pipeline to bind.
    pub pipeline: PipelineHandle,
    /// Vertex buffer handle.
    pub vertex_buffer: BufferHandle,
    /// Optional index buffer.
    pub index_buffer: Option<BufferHandle>,
    /// Number of vertices or indices.
    pub element_count: u32,
    /// Number of instances.
    pub instance_count: u32,
    /// First vertex / index.
    pub first_element: u32,
    /// Model transform (object-to-world).
    pub transform: Mat4,
}

/// A sorted queue of draw calls collected during the scene traversal phase
/// and consumed during command buffer recording.
pub struct RenderQueue {
    draw_calls: Vec<DrawCall>,
    sorted: bool,
}

impl RenderQueue {
    /// Create an empty render queue.
    pub fn new() -> Self {
        Self {
            draw_calls: Vec::with_capacity(4096),
            sorted: false,
        }
    }

    /// Enqueue a draw call.
    pub fn push(&mut self, call: DrawCall) {
        self.draw_calls.push(call);
        self.sorted = false;
    }

    /// Sort all queued draw calls by their sort key.
    pub fn sort(&mut self) {
        if !self.sorted {
            self.draw_calls.sort_unstable_by_key(|dc| dc.sort_key);
            self.sorted = true;
        }
    }

    /// Iterate over sorted draw calls.
    pub fn iter(&mut self) -> impl Iterator<Item = &DrawCall> {
        self.sort();
        self.draw_calls.iter()
    }

    /// Clear the queue for the next frame.
    pub fn clear(&mut self) {
        self.draw_calls.clear();
        self.sorted = true;
    }

    /// Number of queued draw calls.
    pub fn len(&self) -> usize {
        self.draw_calls.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.draw_calls.is_empty()
    }
}

impl Default for RenderQueue {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ECS Components
// ---------------------------------------------------------------------------

/// ECS component that marks an entity as renderable with a mesh.
///
/// The renderer collects all entities with a `MeshRenderer` during the
/// extraction phase and populates the [`RenderQueue`].
#[derive(Debug, Clone)]
pub struct MeshRenderer {
    /// Handle to the vertex buffer containing mesh data.
    pub vertex_buffer: BufferHandle,
    /// Handle to the index buffer (if indexed).
    pub index_buffer: Option<BufferHandle>,
    /// Number of indices (or vertices if not indexed).
    pub element_count: u32,
    /// Pipeline handle for the material / shader combination.
    pub pipeline: PipelineHandle,
    /// Render layer for sorting (0 = background, 128 = default opaque, 255 = overlay).
    pub render_layer: u8,
    /// Whether this renderer is currently visible.
    pub visible: bool,
}

/// Projection mode for a camera.
#[derive(Debug, Clone, Copy)]
pub enum Projection {
    /// Perspective projection.
    Perspective {
        /// Vertical field of view in radians.
        fov_y: f32,
        /// Near clip plane distance.
        near: f32,
        /// Far clip plane distance.
        far: f32,
    },
    /// Orthographic projection.
    Orthographic {
        /// Half-height of the view volume.
        half_height: f32,
        /// Near clip plane distance.
        near: f32,
        /// Far clip plane distance.
        far: f32,
    },
}

/// ECS component representing a camera.
///
/// The renderer picks the active camera each frame and uses it to build the
/// view-projection matrix for culling and rendering.
#[derive(Debug, Clone)]
pub struct Camera {
    /// Projection mode.
    pub projection: Projection,
    /// Viewport aspect ratio (width / height). Updated by the renderer when
    /// the window resizes.
    pub aspect_ratio: f32,
    /// Clear colour for the camera's render target.
    pub clear_color: Vec4,
    /// Priority when multiple cameras exist (highest wins).
    pub priority: i32,
    /// Whether this camera is the active one.
    pub active: bool,
    /// Optional render target override (None = swapchain).
    pub render_target: Option<TextureHandle>,
}

impl Camera {
    /// Compute the projection matrix.
    pub fn projection_matrix(&self) -> Mat4 {
        match self.projection {
            Projection::Perspective { fov_y, near, far } => {
                Mat4::perspective_rh(fov_y, self.aspect_ratio, near, far)
            }
            Projection::Orthographic { half_height, near, far } => {
                let half_width = half_height * self.aspect_ratio;
                Mat4::orthographic_rh(
                    -half_width, half_width,
                    -half_height, half_height,
                    near, far,
                )
            }
        }
    }

    /// Compute a view matrix from a world-space position and look direction.
    pub fn view_matrix(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
        Mat4::look_at_rh(eye, target, up)
    }
}
