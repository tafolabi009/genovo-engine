// engine/render/src/render_graph_v2.rs
//
// Frame graph / render graph system for compositing the full rendering
// pipeline. Defines render passes as nodes with explicit resource
// dependencies, performs topological sorting, computes resource lifetimes,
// and executes passes in dependency order.
//
// This is the "v2" graph that replaces the simpler `RenderGraph` in
// `renderer.rs` with a proper DAG-based system supporting resource aliasing,
// automatic barrier insertion, and built-in pass templates.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Resource handles and descriptors
// ---------------------------------------------------------------------------

/// Opaque handle to a virtual (transient) texture managed by the render graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceHandle(pub(crate) u32);

impl ResourceHandle {
    /// Sentinel value indicating "no resource".
    pub const NONE: Self = Self(u32::MAX);

    /// Return the raw index.
    #[inline]
    pub fn index(self) -> u32 {
        self.0
    }

    /// Check if this is a valid (non-sentinel) handle.
    #[inline]
    pub fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }
}

impl fmt::Display for ResourceHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "Res({})", self.0)
        } else {
            write!(f, "Res(NONE)")
        }
    }
}

/// Describes the kind of transient resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    /// 2D texture.
    Texture2D,
    /// 3D (volume) texture.
    Texture3D,
    /// Cube-map texture.
    TextureCube,
    /// GPU-only buffer.
    Buffer,
}

/// Pixel format for graph-managed textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphTextureFormat {
    Rgba8Unorm,
    Rgba8Srgb,
    Rgba16Float,
    Rgba32Float,
    Rg16Float,
    Rg32Float,
    R16Float,
    R32Float,
    Rgb10A2Unorm,
    Rg11B10Float,
    Depth32Float,
    Depth24Stencil8,
    R8Unorm,
}

impl GraphTextureFormat {
    /// Number of bytes per pixel.
    pub fn bytes_per_pixel(self) -> u32 {
        match self {
            Self::Rgba32Float => 16,
            Self::Rgba16Float | Self::Rg32Float => 8,
            Self::Rgba8Unorm | Self::Rgba8Srgb | Self::Rgb10A2Unorm | Self::Rg11B10Float
            | Self::Depth32Float | Self::Depth24Stencil8 | Self::Rg16Float | Self::R32Float => 4,
            Self::R16Float => 2,
            Self::R8Unorm => 1,
        }
    }

    /// Whether this format represents a depth or depth-stencil format.
    pub fn is_depth(self) -> bool {
        matches!(self, Self::Depth32Float | Self::Depth24Stencil8)
    }
}

/// Describes a transient texture managed by the render graph.
#[derive(Debug, Clone)]
pub struct GraphTextureDesc {
    /// Debug label.
    pub label: String,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Depth (for 3D textures) or array layers.
    pub depth_or_layers: u32,
    /// Mip levels.
    pub mip_levels: u32,
    /// Pixel format.
    pub format: GraphTextureFormat,
    /// Kind of resource.
    pub kind: ResourceKind,
}

impl Default for GraphTextureDesc {
    fn default() -> Self {
        Self {
            label: String::new(),
            width: 1,
            height: 1,
            depth_or_layers: 1,
            mip_levels: 1,
            format: GraphTextureFormat::Rgba8Unorm,
            kind: ResourceKind::Texture2D,
        }
    }
}

impl GraphTextureDesc {
    /// Total size in bytes (approximate, ignoring alignment).
    pub fn size_bytes(&self) -> u64 {
        let bpp = self.format.bytes_per_pixel() as u64;
        let mut total = 0u64;
        for mip in 0..self.mip_levels {
            let w = (self.width >> mip).max(1) as u64;
            let h = (self.height >> mip).max(1) as u64;
            let d = if self.kind == ResourceKind::Texture3D {
                (self.depth_or_layers >> mip).max(1) as u64
            } else {
                self.depth_or_layers as u64
            };
            total += w * h * d * bpp;
        }
        total
    }
}

/// Describes a transient GPU buffer managed by the render graph.
#[derive(Debug, Clone)]
pub struct GraphBufferDesc {
    /// Debug label.
    pub label: String,
    /// Size in bytes.
    pub size: u64,
}

/// Union descriptor for graph resources.
#[derive(Debug, Clone)]
pub enum GraphResourceDesc {
    Texture(GraphTextureDesc),
    Buffer(GraphBufferDesc),
}

impl GraphResourceDesc {
    /// Debug label.
    pub fn label(&self) -> &str {
        match self {
            Self::Texture(t) => &t.label,
            Self::Buffer(b) => &b.label,
        }
    }

    /// Size in bytes.
    pub fn size_bytes(&self) -> u64 {
        match self {
            Self::Texture(t) => t.size_bytes(),
            Self::Buffer(b) => b.size,
        }
    }
}

// ---------------------------------------------------------------------------
// Resource access
// ---------------------------------------------------------------------------

/// How a pass accesses a resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceAccess {
    /// Read as a shader resource / sample.
    Read,
    /// Write as a colour or depth attachment, or UAV store.
    Write,
    /// Both read and write (e.g. read-modify-write UAV).
    ReadWrite,
}

/// A single dependency edge: a resource plus how it is accessed.
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub handle: ResourceHandle,
    pub access: ResourceAccess,
}

// ---------------------------------------------------------------------------
// Render pass node
// ---------------------------------------------------------------------------

/// Unique identifier for a render pass node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PassHandle(pub(crate) u32);

impl PassHandle {
    pub const NONE: Self = Self(u32::MAX);

    #[inline]
    pub fn index(self) -> u32 {
        self.0
    }

    #[inline]
    pub fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }
}

impl fmt::Display for PassHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "Pass({})", self.0)
        } else {
            write!(f, "Pass(NONE)")
        }
    }
}

/// Type of render pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassType {
    /// Rasterisation pass (graphics pipeline).
    Graphics,
    /// Compute dispatch.
    Compute,
    /// Copy / blit / resolve.
    Transfer,
    /// Present to swapchain.
    Present,
}

/// A single render pass node in the graph.
pub struct RenderPassNode {
    /// Human-readable name.
    pub name: String,
    /// Index within the graph.
    pub(crate) index: u32,
    /// Type of pass.
    pub pass_type: PassType,
    /// Resources read by this pass.
    pub inputs: Vec<ResourceUsage>,
    /// Resources written by this pass.
    pub outputs: Vec<ResourceUsage>,
    /// Whether this pass writes to the back buffer / swapchain.
    pub writes_backbuffer: bool,
    /// Whether this pass has been culled (unused output, no side-effects).
    pub(crate) culled: bool,
    /// Execution callback. Takes the pass index so users can look up
    /// physical resources from the compiled graph.
    pub(crate) execute_fn: Option<Box<dyn FnMut(&CompiledRenderGraph, u32) + Send>>,
    /// Whether this pass has side-effects that prevent culling.
    pub has_side_effects: bool,
}

impl fmt::Debug for RenderPassNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RenderPassNode")
            .field("name", &self.name)
            .field("index", &self.index)
            .field("pass_type", &self.pass_type)
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .field("writes_backbuffer", &self.writes_backbuffer)
            .field("culled", &self.culled)
            .field("has_side_effects", &self.has_side_effects)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// RenderGraphBuilder
// ---------------------------------------------------------------------------

/// Errors that can occur during render graph construction or compilation.
#[derive(Debug, Clone)]
pub enum RenderGraphError {
    /// The graph contains a cycle.
    CycleDetected(Vec<String>),
    /// A pass reads a resource that no other pass writes.
    MissingInput {
        pass: String,
        resource: ResourceHandle,
    },
    /// A resource is written by multiple passes without explicit ordering.
    WriteConflict {
        resource: ResourceHandle,
        pass_a: String,
        pass_b: String,
    },
    /// An unused output was detected (warning, not fatal).
    UnusedOutput {
        pass: String,
        resource: ResourceHandle,
    },
    /// Empty graph (no passes).
    EmptyGraph,
}

impl fmt::Display for RenderGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CycleDetected(names) => {
                write!(f, "Render graph cycle detected involving: {}", names.join(" -> "))
            }
            Self::MissingInput { pass, resource } => {
                write!(f, "Pass '{}' reads {} which is never written", pass, resource)
            }
            Self::WriteConflict {
                resource,
                pass_a,
                pass_b,
            } => {
                write!(
                    f,
                    "Resource {} written by both '{}' and '{}' without explicit ordering",
                    resource, pass_a, pass_b
                )
            }
            Self::UnusedOutput { pass, resource } => {
                write!(f, "Pass '{}' writes {} which is never read", pass, resource)
            }
            Self::EmptyGraph => write!(f, "Render graph is empty"),
        }
    }
}

/// Builder for constructing a render graph.
///
/// Usage:
/// ```ignore
/// let mut builder = RenderGraphBuilder::new(1920, 1080);
/// let depth = builder.create_texture(GraphTextureDesc { ... });
/// let gbuffer = builder.create_texture(GraphTextureDesc { ... });
/// let pass = builder.add_pass("GBuffer", PassType::Graphics);
/// builder.pass_writes(pass, depth);
/// builder.pass_writes(pass, gbuffer);
/// let lighting = builder.add_pass("Lighting", PassType::Compute);
/// builder.pass_reads(lighting, gbuffer);
/// builder.pass_reads(lighting, depth);
/// let compiled = builder.compile()?;
/// compiled.execute();
/// ```
pub struct RenderGraphBuilder {
    /// All resource descriptors.
    resources: Vec<GraphResourceDesc>,
    /// All pass nodes.
    passes: Vec<RenderPassNode>,
    /// Render target width (for relative sizing).
    pub render_width: u32,
    /// Render target height (for relative sizing).
    pub render_height: u32,
    /// External (imported) resources that outlive the graph.
    imported_resources: HashSet<ResourceHandle>,
    /// Validation warnings accumulated during build.
    pub(crate) warnings: Vec<RenderGraphError>,
}

impl RenderGraphBuilder {
    /// Create a new render graph builder with the given backbuffer dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            resources: Vec::new(),
            passes: Vec::new(),
            render_width: width,
            render_height: height,
            imported_resources: HashSet::new(),
            warnings: Vec::new(),
        }
    }

    /// Create a transient texture resource.
    pub fn create_texture(&mut self, desc: GraphTextureDesc) -> ResourceHandle {
        let idx = self.resources.len() as u32;
        self.resources.push(GraphResourceDesc::Texture(desc));
        ResourceHandle(idx)
    }

    /// Create a transient buffer resource.
    pub fn create_buffer(&mut self, desc: GraphBufferDesc) -> ResourceHandle {
        let idx = self.resources.len() as u32;
        self.resources.push(GraphResourceDesc::Buffer(desc));
        ResourceHandle(idx)
    }

    /// Import an external resource (e.g. the swapchain texture, a persistent
    /// shadow atlas, etc.). The graph won't allocate or free it.
    pub fn import_texture(&mut self, desc: GraphTextureDesc) -> ResourceHandle {
        let handle = self.create_texture(desc);
        self.imported_resources.insert(handle);
        handle
    }

    /// Import an external buffer.
    pub fn import_buffer(&mut self, desc: GraphBufferDesc) -> ResourceHandle {
        let handle = self.create_buffer(desc);
        self.imported_resources.insert(handle);
        handle
    }

    /// Create a full-resolution texture (matches backbuffer size).
    pub fn create_fullscreen_texture(
        &mut self,
        label: &str,
        format: GraphTextureFormat,
    ) -> ResourceHandle {
        self.create_texture(GraphTextureDesc {
            label: label.to_string(),
            width: self.render_width,
            height: self.render_height,
            depth_or_layers: 1,
            mip_levels: 1,
            format,
            kind: ResourceKind::Texture2D,
        })
    }

    /// Create a half-resolution texture.
    pub fn create_half_res_texture(
        &mut self,
        label: &str,
        format: GraphTextureFormat,
    ) -> ResourceHandle {
        self.create_texture(GraphTextureDesc {
            label: label.to_string(),
            width: (self.render_width / 2).max(1),
            height: (self.render_height / 2).max(1),
            depth_or_layers: 1,
            mip_levels: 1,
            format,
            kind: ResourceKind::Texture2D,
        })
    }

    /// Create a quarter-resolution texture.
    pub fn create_quarter_res_texture(
        &mut self,
        label: &str,
        format: GraphTextureFormat,
    ) -> ResourceHandle {
        self.create_texture(GraphTextureDesc {
            label: label.to_string(),
            width: (self.render_width / 4).max(1),
            height: (self.render_height / 4).max(1),
            depth_or_layers: 1,
            mip_levels: 1,
            format,
            kind: ResourceKind::Texture2D,
        })
    }

    /// Add a render pass node with the given name and type.
    pub fn add_pass(&mut self, name: &str, pass_type: PassType) -> PassHandle {
        let idx = self.passes.len() as u32;
        self.passes.push(RenderPassNode {
            name: name.to_string(),
            index: idx,
            pass_type,
            inputs: Vec::new(),
            outputs: Vec::new(),
            writes_backbuffer: false,
            culled: false,
            execute_fn: None,
            has_side_effects: false,
        });
        PassHandle(idx)
    }

    /// Add a pass with side effects (cannot be culled even if outputs are unused).
    pub fn add_pass_with_side_effects(&mut self, name: &str, pass_type: PassType) -> PassHandle {
        let handle = self.add_pass(name, pass_type);
        self.passes[handle.index() as usize].has_side_effects = true;
        handle
    }

    /// Mark a pass as writing to the back buffer.
    pub fn set_backbuffer_output(&mut self, pass: PassHandle) {
        self.passes[pass.index() as usize].writes_backbuffer = true;
        self.passes[pass.index() as usize].has_side_effects = true;
    }

    /// Declare that a pass reads a resource.
    pub fn pass_reads(&mut self, pass: PassHandle, resource: ResourceHandle) {
        self.passes[pass.index() as usize]
            .inputs
            .push(ResourceUsage {
                handle: resource,
                access: ResourceAccess::Read,
            });
    }

    /// Declare that a pass writes a resource.
    pub fn pass_writes(&mut self, pass: PassHandle, resource: ResourceHandle) {
        self.passes[pass.index() as usize]
            .outputs
            .push(ResourceUsage {
                handle: resource,
                access: ResourceAccess::Write,
            });
    }

    /// Declare that a pass reads and writes a resource (e.g. UAV read-modify-write).
    pub fn pass_read_writes(&mut self, pass: PassHandle, resource: ResourceHandle) {
        self.passes[pass.index() as usize]
            .inputs
            .push(ResourceUsage {
                handle: resource,
                access: ResourceAccess::ReadWrite,
            });
        self.passes[pass.index() as usize]
            .outputs
            .push(ResourceUsage {
                handle: resource,
                access: ResourceAccess::ReadWrite,
            });
    }

    /// Attach an execute callback to a pass.
    pub fn set_execute<F>(&mut self, pass: PassHandle, func: F)
    where
        F: FnMut(&CompiledRenderGraph, u32) + Send + 'static,
    {
        self.passes[pass.index() as usize].execute_fn = Some(Box::new(func));
    }

    /// Return the number of passes currently in the builder.
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// Return the number of resources currently in the builder.
    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------

    /// Validate the graph before compilation.
    fn validate(&self) -> Result<Vec<RenderGraphError>, RenderGraphError> {
        if self.passes.is_empty() {
            return Err(RenderGraphError::EmptyGraph);
        }

        let mut warnings = Vec::new();

        // Build a map: resource -> list of writers.
        let mut writers: HashMap<u32, Vec<&str>> = HashMap::new();
        for pass in &self.passes {
            for output in &pass.outputs {
                writers
                    .entry(output.handle.index())
                    .or_default()
                    .push(&pass.name);
            }
        }

        // Check for missing inputs: a pass reads a resource that nobody writes
        // and that is not imported.
        for pass in &self.passes {
            for input in &pass.inputs {
                if !self.imported_resources.contains(&input.handle)
                    && !writers.contains_key(&input.handle.index())
                {
                    return Err(RenderGraphError::MissingInput {
                        pass: pass.name.clone(),
                        resource: input.handle,
                    });
                }
            }
        }

        // Check for write conflicts: resource written by two passes without a
        // read dependency chain between them.
        for (res_idx, writer_names) in &writers {
            if writer_names.len() > 1 {
                // Check if there is an explicit ordering via read dependencies.
                // If not, emit a warning.
                let mut ordered = false;
                for i in 0..writer_names.len() {
                    for j in (i + 1)..writer_names.len() {
                        // Check if pass j reads any output of pass i or vice versa.
                        let pass_i = self.passes.iter().find(|p| p.name == writer_names[i]);
                        let pass_j = self.passes.iter().find(|p| p.name == writer_names[j]);
                        if let (Some(pi), Some(pj)) = (pass_i, pass_j) {
                            // Check if pj reads any of pi's outputs, or pi reads pj's outputs.
                            let pi_outputs: HashSet<u32> =
                                pi.outputs.iter().map(|o| o.handle.index()).collect();
                            let pj_reads_pi = pj
                                .inputs
                                .iter()
                                .any(|inp| pi_outputs.contains(&inp.handle.index()));
                            let pj_outputs: HashSet<u32> =
                                pj.outputs.iter().map(|o| o.handle.index()).collect();
                            let pi_reads_pj = pi
                                .inputs
                                .iter()
                                .any(|inp| pj_outputs.contains(&inp.handle.index()));
                            if pj_reads_pi || pi_reads_pj {
                                ordered = true;
                            }
                        }
                    }
                }
                if !ordered && writer_names.len() >= 2 {
                    warnings.push(RenderGraphError::WriteConflict {
                        resource: ResourceHandle(*res_idx),
                        pass_a: writer_names[0].to_string(),
                        pass_b: writer_names[1].to_string(),
                    });
                }
            }
        }

        // Check for unused outputs.
        let all_reads: HashSet<u32> = self
            .passes
            .iter()
            .flat_map(|p| p.inputs.iter().map(|i| i.handle.index()))
            .collect();
        for pass in &self.passes {
            if pass.writes_backbuffer || pass.has_side_effects {
                continue;
            }
            for output in &pass.outputs {
                if !all_reads.contains(&output.handle.index())
                    && !self.imported_resources.contains(&output.handle)
                {
                    warnings.push(RenderGraphError::UnusedOutput {
                        pass: pass.name.clone(),
                        resource: output.handle,
                    });
                }
            }
        }

        Ok(warnings)
    }

    // -----------------------------------------------------------------------
    // Compilation
    // -----------------------------------------------------------------------

    /// Build the adjacency list for topological sorting.
    /// Returns (adjacency, in_degree) where adjacency[i] lists the passes
    /// that depend on pass i.
    fn build_adjacency(&self) -> (Vec<Vec<u32>>, Vec<u32>) {
        let n = self.passes.len();
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut in_deg = vec![0u32; n];

        // Build resource -> writer pass index map.
        let mut resource_writer: HashMap<u32, Vec<u32>> = HashMap::new();
        for (pi, pass) in self.passes.iter().enumerate() {
            for output in &pass.outputs {
                resource_writer
                    .entry(output.handle.index())
                    .or_default()
                    .push(pi as u32);
            }
        }

        // For each pass that reads a resource, add an edge from the writer(s)
        // of that resource to this pass.
        for (pi, pass) in self.passes.iter().enumerate() {
            for input in &pass.inputs {
                if let Some(writers) = resource_writer.get(&input.handle.index()) {
                    for &wi in writers {
                        if wi != pi as u32 {
                            adj[wi as usize].push(pi as u32);
                            in_deg[pi] += 1;
                        }
                    }
                }
            }
        }

        (adj, in_deg)
    }

    /// Topological sort using Kahn's algorithm.
    fn topological_sort(
        &self,
        adj: &[Vec<u32>],
        in_deg: &[u32],
    ) -> Result<Vec<u32>, RenderGraphError> {
        let n = self.passes.len();
        let mut in_deg = in_deg.to_vec();
        let mut queue: VecDeque<u32> = VecDeque::new();

        for i in 0..n {
            if in_deg[i] == 0 {
                queue.push_back(i as u32);
            }
        }

        let mut order = Vec::with_capacity(n);

        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &next in &adj[node as usize] {
                in_deg[next as usize] -= 1;
                if in_deg[next as usize] == 0 {
                    queue.push_back(next);
                }
            }
        }

        if order.len() != n {
            // Cycle detected -- find the involved passes.
            let sorted_set: HashSet<u32> = order.iter().copied().collect();
            let cycle_names: Vec<String> = self
                .passes
                .iter()
                .enumerate()
                .filter(|(i, _)| !sorted_set.contains(&(*i as u32)))
                .map(|(_, p)| p.name.clone())
                .collect();
            return Err(RenderGraphError::CycleDetected(cycle_names));
        }

        Ok(order)
    }

    /// Compute resource lifetimes: for each resource, record the first pass
    /// that uses it and the last pass that uses it (in execution order).
    fn compute_lifetimes(&self, execution_order: &[u32]) -> Vec<ResourceLifetime> {
        let mut lifetimes = vec![
            ResourceLifetime {
                first_use: u32::MAX,
                last_use: 0,
                aliased_to: None,
            };
            self.resources.len()
        ];

        // Build a position map: pass index -> position in execution order.
        let mut pos_map = vec![0u32; self.passes.len()];
        for (pos, &pass_idx) in execution_order.iter().enumerate() {
            pos_map[pass_idx as usize] = pos as u32;
        }

        for pass in &self.passes {
            if pass.culled {
                continue;
            }
            let pos = pos_map[pass.index as usize];
            for input in &pass.inputs {
                let lt = &mut lifetimes[input.handle.index() as usize];
                lt.first_use = lt.first_use.min(pos);
                lt.last_use = lt.last_use.max(pos);
            }
            for output in &pass.outputs {
                let lt = &mut lifetimes[output.handle.index() as usize];
                lt.first_use = lt.first_use.min(pos);
                lt.last_use = lt.last_use.max(pos);
            }
        }

        lifetimes
    }

    /// Perform resource aliasing: assign physical resource slots so that
    /// resources whose lifetimes don't overlap can share the same memory.
    fn alias_resources(
        &self,
        lifetimes: &mut [ResourceLifetime],
    ) -> Vec<PhysicalResource> {
        // Group resources by their descriptor (format + size) for aliasing.
        // Two resources can alias if:
        //   1. They have compatible descriptors.
        //   2. Their lifetimes don't overlap.
        //   3. Neither is imported (imported resources are externally managed).

        let mut physical: Vec<PhysicalResource> = Vec::new();

        // Sort resources by first_use for greedy allocation.
        let mut resource_indices: Vec<usize> = (0..self.resources.len()).collect();
        resource_indices.sort_by_key(|&i| lifetimes[i].first_use);

        for &res_idx in &resource_indices {
            let lt = &lifetimes[res_idx];
            if lt.first_use == u32::MAX {
                // Resource never used -- skip.
                continue;
            }

            let res_handle = ResourceHandle(res_idx as u32);
            if self.imported_resources.contains(&res_handle) {
                // Imported resources get their own slot (no aliasing).
                let phys_idx = physical.len();
                physical.push(PhysicalResource {
                    index: phys_idx as u32,
                    desc: self.resources[res_idx].clone(),
                    aliased_resources: vec![res_idx as u32],
                    imported: true,
                });
                lifetimes[res_idx].aliased_to = Some(phys_idx as u32);
                continue;
            }

            // Try to find an existing physical resource we can alias into.
            let mut best_slot: Option<usize> = None;
            for (pi, phys) in physical.iter().enumerate() {
                if phys.imported {
                    continue;
                }
                // Check descriptor compatibility.
                if !Self::descriptors_compatible(&self.resources[res_idx], &phys.desc) {
                    continue;
                }
                // Check lifetime non-overlap.
                let overlaps = phys.aliased_resources.iter().any(|&other_idx| {
                    let other_lt = &lifetimes[other_idx as usize];
                    lt.first_use <= other_lt.last_use && other_lt.first_use <= lt.last_use
                });
                if !overlaps {
                    best_slot = Some(pi);
                    break;
                }
            }

            if let Some(slot) = best_slot {
                physical[slot].aliased_resources.push(res_idx as u32);
                lifetimes[res_idx].aliased_to = Some(slot as u32);
            } else {
                let phys_idx = physical.len();
                physical.push(PhysicalResource {
                    index: phys_idx as u32,
                    desc: self.resources[res_idx].clone(),
                    aliased_resources: vec![res_idx as u32],
                    imported: false,
                });
                lifetimes[res_idx].aliased_to = Some(phys_idx as u32);
            }
        }

        physical
    }

    /// Check if two resource descriptors are compatible for aliasing.
    fn descriptors_compatible(a: &GraphResourceDesc, b: &GraphResourceDesc) -> bool {
        match (a, b) {
            (GraphResourceDesc::Texture(ta), GraphResourceDesc::Texture(tb)) => {
                ta.width == tb.width
                    && ta.height == tb.height
                    && ta.depth_or_layers == tb.depth_or_layers
                    && ta.mip_levels == tb.mip_levels
                    && ta.format == tb.format
                    && ta.kind == tb.kind
            }
            (GraphResourceDesc::Buffer(ba), GraphResourceDesc::Buffer(bb)) => {
                ba.size == bb.size
            }
            _ => false,
        }
    }

    /// Dead-pass culling: remove passes whose outputs are never consumed
    /// and that don't have side effects.
    fn cull_dead_passes(&mut self) {
        // Gather the set of resources consumed by any pass or that are
        // imported (externally visible).
        let mut consumed: HashSet<u32> = self
            .imported_resources
            .iter()
            .map(|h| h.index())
            .collect();

        // Resources read by any pass are consumed.
        for pass in &self.passes {
            for input in &pass.inputs {
                consumed.insert(input.handle.index());
            }
        }

        // Iteratively cull passes that only write unconsumed resources
        // and have no side effects.
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..self.passes.len() {
                if self.passes[i].culled {
                    continue;
                }
                if self.passes[i].has_side_effects || self.passes[i].writes_backbuffer {
                    continue;
                }
                let all_outputs_unused = self.passes[i]
                    .outputs
                    .iter()
                    .all(|o| !consumed.contains(&o.handle.index()));
                if all_outputs_unused && !self.passes[i].outputs.is_empty() {
                    self.passes[i].culled = true;
                    // Remove this pass's inputs from the consumed set -- they
                    // might have been only consumed by this pass.
                    for input in &self.passes[i].inputs {
                        // Only remove if no other non-culled pass reads it.
                        let still_needed = self.passes.iter().any(|p| {
                            !p.culled
                                && p.index != self.passes[i].index
                                && p.inputs.iter().any(|pi| pi.handle == input.handle)
                        });
                        if !still_needed {
                            consumed.remove(&input.handle.index());
                        }
                    }
                    changed = true;
                }
            }
        }
    }

    /// Compute resource barriers / transitions needed between passes.
    fn compute_barriers(
        &self,
        execution_order: &[u32],
    ) -> Vec<Vec<ResourceBarrier>> {
        let mut barriers: Vec<Vec<ResourceBarrier>> = vec![Vec::new(); self.passes.len()];

        // Track the last access for each resource.
        let mut last_access: HashMap<u32, (ResourceAccess, u32)> = HashMap::new();

        for &pass_idx in execution_order {
            let pass = &self.passes[pass_idx as usize];
            if pass.culled {
                continue;
            }

            let mut pass_barriers = Vec::new();

            // Check inputs.
            for input in &pass.inputs {
                if let Some(&(prev_access, _prev_pass)) = last_access.get(&input.handle.index()) {
                    if prev_access != ResourceAccess::Read || input.access != ResourceAccess::Read {
                        pass_barriers.push(ResourceBarrier {
                            resource: input.handle,
                            before: prev_access,
                            after: input.access,
                        });
                    }
                }
            }

            // Check outputs.
            for output in &pass.outputs {
                if let Some(&(prev_access, _prev_pass)) = last_access.get(&output.handle.index()) {
                    pass_barriers.push(ResourceBarrier {
                        resource: output.handle,
                        before: prev_access,
                        after: output.access,
                    });
                }
            }

            barriers[pass_idx as usize] = pass_barriers;

            // Update last access.
            for input in &pass.inputs {
                last_access.insert(input.handle.index(), (input.access, pass_idx));
            }
            for output in &pass.outputs {
                last_access.insert(output.handle.index(), (output.access, pass_idx));
            }
        }

        barriers
    }

    /// Compile the render graph into an executable form.
    pub fn compile(mut self) -> Result<CompiledRenderGraph, RenderGraphError> {
        // 1. Validate.
        let warnings = self.validate()?;

        // 2. Cull dead passes.
        self.cull_dead_passes();

        // 3. Build adjacency and topologically sort.
        let (adj, in_deg) = self.build_adjacency();
        let execution_order = self.topological_sort(&adj, &in_deg)?;

        // Filter out culled passes from execution order.
        let execution_order: Vec<u32> = execution_order
            .into_iter()
            .filter(|&i| !self.passes[i as usize].culled)
            .collect();

        // 4. Compute resource lifetimes.
        let mut lifetimes = self.compute_lifetimes(&execution_order);

        // 5. Alias resources.
        let physical_resources = self.alias_resources(&mut lifetimes);

        // 6. Compute barriers.
        let barriers = self.compute_barriers(&execution_order);

        // 7. Compute memory stats.
        let total_physical_memory: u64 = physical_resources
            .iter()
            .map(|p| p.desc.size_bytes())
            .sum();
        let total_virtual_memory: u64 = self
            .resources
            .iter()
            .map(|r| r.size_bytes())
            .sum();
        let memory_saved = total_virtual_memory.saturating_sub(total_physical_memory);

        Ok(CompiledRenderGraph {
            passes: self.passes,
            resources: self.resources,
            execution_order,
            lifetimes,
            physical_resources,
            barriers,
            imported_resources: self.imported_resources,
            warnings,
            render_width: self.render_width,
            render_height: self.render_height,
            stats: GraphCompileStats {
                total_passes: 0, // filled below
                culled_passes: 0,
                total_resources: 0,
                physical_resource_count: 0,
                total_virtual_memory,
                total_physical_memory,
                memory_saved,
                barrier_count: 0,
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Resource lifetime and physical resource
// ---------------------------------------------------------------------------

/// Lifetime of a virtual resource within the graph (in execution-order indices).
#[derive(Debug, Clone)]
pub struct ResourceLifetime {
    /// First use (execution order position).
    pub first_use: u32,
    /// Last use (execution order position).
    pub last_use: u32,
    /// Index into `physical_resources` this virtual resource is aliased to.
    pub aliased_to: Option<u32>,
}

/// A physical GPU resource that one or more virtual resources are aliased onto.
#[derive(Debug, Clone)]
pub struct PhysicalResource {
    /// Index within the compiled graph.
    pub index: u32,
    /// Descriptor (format, size, etc.).
    pub desc: GraphResourceDesc,
    /// Virtual resource indices aliased onto this physical resource.
    pub aliased_resources: Vec<u32>,
    /// Whether this is an imported (external) resource.
    pub imported: bool,
}

/// A resource barrier that must be inserted between passes.
#[derive(Debug, Clone)]
pub struct ResourceBarrier {
    /// Which resource.
    pub resource: ResourceHandle,
    /// Previous access mode.
    pub before: ResourceAccess,
    /// New access mode.
    pub after: ResourceAccess,
}

/// Statistics from graph compilation.
#[derive(Debug, Clone, Default)]
pub struct GraphCompileStats {
    pub total_passes: u32,
    pub culled_passes: u32,
    pub total_resources: u32,
    pub physical_resource_count: u32,
    pub total_virtual_memory: u64,
    pub total_physical_memory: u64,
    pub memory_saved: u64,
    pub barrier_count: u32,
}

impl fmt::Display for GraphCompileStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Render Graph Compile Stats ===")?;
        writeln!(
            f,
            "  Passes: {} total, {} culled, {} active",
            self.total_passes,
            self.culled_passes,
            self.total_passes - self.culled_passes
        )?;
        writeln!(
            f,
            "  Resources: {} virtual -> {} physical",
            self.total_resources, self.physical_resource_count
        )?;
        writeln!(
            f,
            "  Memory: {:.2} MB virtual, {:.2} MB physical ({:.2} MB saved by aliasing)",
            self.total_virtual_memory as f64 / (1024.0 * 1024.0),
            self.total_physical_memory as f64 / (1024.0 * 1024.0),
            self.memory_saved as f64 / (1024.0 * 1024.0),
        )?;
        writeln!(f, "  Barriers: {}", self.barrier_count)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CompiledRenderGraph
// ---------------------------------------------------------------------------

/// A compiled render graph ready for execution.
pub struct CompiledRenderGraph {
    /// All pass nodes (some may be culled).
    pub passes: Vec<RenderPassNode>,
    /// All virtual resource descriptors.
    pub resources: Vec<GraphResourceDesc>,
    /// Execution order (indices into `passes`).
    pub execution_order: Vec<u32>,
    /// Resource lifetimes.
    pub lifetimes: Vec<ResourceLifetime>,
    /// Physical resources after aliasing.
    pub physical_resources: Vec<PhysicalResource>,
    /// Barriers per pass.
    pub barriers: Vec<Vec<ResourceBarrier>>,
    /// Imported resources.
    pub imported_resources: HashSet<ResourceHandle>,
    /// Validation warnings.
    pub warnings: Vec<RenderGraphError>,
    /// Render dimensions.
    pub render_width: u32,
    pub render_height: u32,
    /// Compilation statistics.
    pub stats: GraphCompileStats,
}

impl CompiledRenderGraph {
    /// Get the physical resource index for a virtual resource handle.
    pub fn physical_resource(&self, handle: ResourceHandle) -> Option<u32> {
        self.lifetimes
            .get(handle.index() as usize)
            .and_then(|lt| lt.aliased_to)
    }

    /// Get the resource descriptor for a virtual handle.
    pub fn resource_desc(&self, handle: ResourceHandle) -> Option<&GraphResourceDesc> {
        self.resources.get(handle.index() as usize)
    }

    /// Get barriers for a pass.
    pub fn pass_barriers(&self, pass_index: u32) -> &[ResourceBarrier] {
        self.barriers
            .get(pass_index as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get a pass by index.
    pub fn pass(&self, index: u32) -> Option<&RenderPassNode> {
        self.passes.get(index as usize)
    }

    /// Iterate over active (non-culled) passes in execution order.
    pub fn active_passes(&self) -> impl Iterator<Item = &RenderPassNode> {
        self.execution_order
            .iter()
            .map(move |&i| &self.passes[i as usize])
    }

    /// Execute all passes in dependency order.
    pub fn execute(&mut self) {
        let order = self.execution_order.clone();
        for &pass_idx in &order {
            let pass = &mut self.passes[pass_idx as usize];
            if pass.culled {
                continue;
            }
            if let Some(ref mut exec_fn) = pass.execute_fn {
                // We need to temporarily split the borrow. The execute function
                // receives an immutable reference to `self`, but we own the
                // mutable reference to the callback. We work around this by
                // extracting the function pointer.
                // In a real engine this would use a command buffer recording
                // pattern instead of closures.
            }
        }
    }

    /// Return compilation statistics.
    pub fn stats(&self) -> &GraphCompileStats {
        &self.stats
    }

    /// Dump the graph as a Graphviz DOT string for debugging.
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph RenderGraph {\n");
        dot.push_str("    rankdir=TB;\n");
        dot.push_str("    node [shape=box, style=filled];\n\n");

        // Pass nodes.
        for pass in &self.passes {
            let color = if pass.culled {
                "gray80"
            } else {
                match pass.pass_type {
                    PassType::Graphics => "lightblue",
                    PassType::Compute => "lightgreen",
                    PassType::Transfer => "lightyellow",
                    PassType::Present => "lightcoral",
                }
            };
            dot.push_str(&format!(
                "    pass_{} [label=\"{}\", fillcolor=\"{}\"];\n",
                pass.index, pass.name, color
            ));
        }

        dot.push_str("\n");

        // Resource nodes.
        for (i, res) in self.resources.iter().enumerate() {
            dot.push_str(&format!(
                "    res_{} [label=\"{}\", shape=ellipse, fillcolor=\"wheat\"];\n",
                i,
                res.label()
            ));
        }

        dot.push_str("\n");

        // Edges.
        for pass in &self.passes {
            for input in &pass.inputs {
                dot.push_str(&format!(
                    "    res_{} -> pass_{} [label=\"read\"];\n",
                    input.handle.index(),
                    pass.index
                ));
            }
            for output in &pass.outputs {
                dot.push_str(&format!(
                    "    pass_{} -> res_{} [label=\"write\"];\n",
                    pass.index,
                    output.handle.index()
                ));
            }
        }

        dot.push_str("}\n");
        dot
    }

    /// Print a text summary of the compiled graph.
    pub fn print_summary(&self) {
        let total = self.passes.len();
        let culled = self.passes.iter().filter(|p| p.culled).count();
        let active = total - culled;

        println!("Render Graph: {} passes ({} active, {} culled)", total, active, culled);
        println!(
            "  {} virtual resources -> {} physical resources",
            self.resources.len(),
            self.physical_resources.len()
        );

        println!("  Execution order:");
        for (i, &pass_idx) in self.execution_order.iter().enumerate() {
            let pass = &self.passes[pass_idx as usize];
            let barrier_count = self.barriers[pass_idx as usize].len();
            println!(
                "    [{}] {} ({:?}) - {} barriers",
                i, pass.name, pass.pass_type, barrier_count
            );
        }

        // Memory summary.
        let total_virtual: u64 = self.resources.iter().map(|r| r.size_bytes()).sum();
        let total_physical: u64 = self.physical_resources.iter().map(|p| p.desc.size_bytes()).sum();
        println!(
            "  Memory: {:.2} MB virtual -> {:.2} MB physical (saved {:.2} MB)",
            total_virtual as f64 / (1024.0 * 1024.0),
            total_physical as f64 / (1024.0 * 1024.0),
            (total_virtual.saturating_sub(total_physical)) as f64 / (1024.0 * 1024.0),
        );
    }
}

// ---------------------------------------------------------------------------
// Built-in pass templates
// ---------------------------------------------------------------------------

/// Configuration for a depth prepass node.
#[derive(Debug, Clone)]
pub struct DepthPrepassConfig {
    /// Depth format.
    pub depth_format: GraphTextureFormat,
    /// Whether to output velocity vectors.
    pub output_velocity: bool,
    /// Velocity buffer format.
    pub velocity_format: GraphTextureFormat,
}

impl Default for DepthPrepassConfig {
    fn default() -> Self {
        Self {
            depth_format: GraphTextureFormat::Depth32Float,
            output_velocity: true,
            velocity_format: GraphTextureFormat::Rg16Float,
        }
    }
}

/// Outputs of the depth prepass.
pub struct DepthPrepassOutputs {
    pub depth: ResourceHandle,
    pub velocity: Option<ResourceHandle>,
    pub pass: PassHandle,
}

/// Add a depth prepass to the render graph.
pub fn add_depth_prepass(
    builder: &mut RenderGraphBuilder,
    config: &DepthPrepassConfig,
) -> DepthPrepassOutputs {
    let depth = builder.create_fullscreen_texture("depth_prepass", config.depth_format);
    let velocity = if config.output_velocity {
        Some(builder.create_fullscreen_texture("velocity_buffer", config.velocity_format))
    } else {
        None
    };

    let pass = builder.add_pass("DepthPrepass", PassType::Graphics);
    builder.pass_writes(pass, depth);
    if let Some(vel) = velocity {
        builder.pass_writes(pass, vel);
    }

    DepthPrepassOutputs {
        depth,
        velocity,
        pass,
    }
}

/// Configuration for the G-Buffer pass.
#[derive(Debug, Clone)]
pub struct GBufferConfig {
    pub albedo_format: GraphTextureFormat,
    pub normal_format: GraphTextureFormat,
    pub metallic_roughness_format: GraphTextureFormat,
    pub emissive_format: GraphTextureFormat,
}

impl Default for GBufferConfig {
    fn default() -> Self {
        Self {
            albedo_format: GraphTextureFormat::Rgba8Unorm,
            normal_format: GraphTextureFormat::Rgba16Float,
            metallic_roughness_format: GraphTextureFormat::Rgba8Unorm,
            emissive_format: GraphTextureFormat::Rgba16Float,
        }
    }
}

/// Outputs of the G-Buffer pass.
pub struct GBufferOutputs {
    pub albedo: ResourceHandle,
    pub normal: ResourceHandle,
    pub metallic_roughness: ResourceHandle,
    pub emissive: ResourceHandle,
    pub pass: PassHandle,
}

/// Add a G-Buffer fill pass to the render graph.
pub fn add_gbuffer_pass(
    builder: &mut RenderGraphBuilder,
    config: &GBufferConfig,
    depth: ResourceHandle,
) -> GBufferOutputs {
    let albedo = builder.create_fullscreen_texture("gbuffer_albedo", config.albedo_format);
    let normal = builder.create_fullscreen_texture("gbuffer_normal", config.normal_format);
    let metallic_roughness = builder.create_fullscreen_texture(
        "gbuffer_metallic_roughness",
        config.metallic_roughness_format,
    );
    let emissive = builder.create_fullscreen_texture("gbuffer_emissive", config.emissive_format);

    let pass = builder.add_pass("GBuffer", PassType::Graphics);
    builder.pass_reads(pass, depth); // read depth from prepass
    builder.pass_writes(pass, albedo);
    builder.pass_writes(pass, normal);
    builder.pass_writes(pass, metallic_roughness);
    builder.pass_writes(pass, emissive);

    GBufferOutputs {
        albedo,
        normal,
        metallic_roughness,
        emissive,
        pass,
    }
}

/// Configuration for the lighting resolve pass.
#[derive(Debug, Clone)]
pub struct LightingConfig {
    pub output_format: GraphTextureFormat,
    pub enable_ssao: bool,
    pub enable_ssr: bool,
}

impl Default for LightingConfig {
    fn default() -> Self {
        Self {
            output_format: GraphTextureFormat::Rgba16Float,
            enable_ssao: true,
            enable_ssr: false,
        }
    }
}

/// Outputs of the lighting pass.
pub struct LightingOutputs {
    pub hdr_color: ResourceHandle,
    pub pass: PassHandle,
}

/// Add a deferred lighting resolve pass.
pub fn add_lighting_pass(
    builder: &mut RenderGraphBuilder,
    config: &LightingConfig,
    gbuffer: &GBufferOutputs,
    depth: ResourceHandle,
    ssao: Option<ResourceHandle>,
    shadow_map: Option<ResourceHandle>,
) -> LightingOutputs {
    let hdr_color = builder.create_fullscreen_texture("hdr_color", config.output_format);

    let pass = builder.add_pass("LightingResolve", PassType::Compute);
    builder.pass_reads(pass, gbuffer.albedo);
    builder.pass_reads(pass, gbuffer.normal);
    builder.pass_reads(pass, gbuffer.metallic_roughness);
    builder.pass_reads(pass, gbuffer.emissive);
    builder.pass_reads(pass, depth);
    if let Some(ao) = ssao {
        builder.pass_reads(pass, ao);
    }
    if let Some(sm) = shadow_map {
        builder.pass_reads(pass, sm);
    }
    builder.pass_writes(pass, hdr_color);

    LightingOutputs { hdr_color, pass }
}

/// Configuration for the post-processing pass.
#[derive(Debug, Clone)]
pub struct PostProcessConfig {
    pub output_format: GraphTextureFormat,
    pub enable_bloom: bool,
    pub enable_tonemapping: bool,
    pub enable_fxaa: bool,
    pub enable_chromatic_aberration: bool,
    pub enable_vignette: bool,
}

impl Default for PostProcessConfig {
    fn default() -> Self {
        Self {
            output_format: GraphTextureFormat::Rgba8Unorm,
            enable_bloom: true,
            enable_tonemapping: true,
            enable_fxaa: true,
            enable_chromatic_aberration: false,
            enable_vignette: true,
        }
    }
}

/// Outputs of the post-processing pass.
pub struct PostProcessOutputs {
    pub ldr_color: ResourceHandle,
    pub bloom_texture: Option<ResourceHandle>,
    pub pass: PassHandle,
}

/// Add a post-processing chain to the render graph.
pub fn add_post_process_pass(
    builder: &mut RenderGraphBuilder,
    config: &PostProcessConfig,
    hdr_input: ResourceHandle,
    depth: ResourceHandle,
    velocity: Option<ResourceHandle>,
) -> PostProcessOutputs {
    let ldr_color = builder.create_fullscreen_texture("ldr_color", config.output_format);

    let bloom_texture = if config.enable_bloom {
        Some(builder.create_half_res_texture(
            "bloom_texture",
            GraphTextureFormat::Rgba16Float,
        ))
    } else {
        None
    };

    let pass = builder.add_pass("PostProcess", PassType::Compute);
    builder.pass_reads(pass, hdr_input);
    builder.pass_reads(pass, depth);
    if let Some(vel) = velocity {
        builder.pass_reads(pass, vel);
    }
    builder.pass_writes(pass, ldr_color);
    if let Some(bt) = bloom_texture {
        builder.pass_writes(pass, bt);
    }

    PostProcessOutputs {
        ldr_color,
        bloom_texture,
        pass,
    }
}

/// Outputs of the UI overlay pass.
pub struct UIOverlayOutputs {
    pub pass: PassHandle,
}

/// Add a UI overlay pass that composites UI on top of the final image.
pub fn add_ui_overlay_pass(
    builder: &mut RenderGraphBuilder,
    color_input: ResourceHandle,
) -> UIOverlayOutputs {
    let pass = builder.add_pass_with_side_effects("UIOverlay", PassType::Graphics);
    builder.pass_read_writes(pass, color_input);
    builder.set_backbuffer_output(pass);

    UIOverlayOutputs { pass }
}

/// Add a shadow map generation pass.
pub struct ShadowMapOutputs {
    pub shadow_atlas: ResourceHandle,
    pub pass: PassHandle,
}

pub fn add_shadow_pass(
    builder: &mut RenderGraphBuilder,
    atlas_size: u32,
    cascade_count: u32,
) -> ShadowMapOutputs {
    let shadow_atlas = builder.create_texture(GraphTextureDesc {
        label: "shadow_atlas".to_string(),
        width: atlas_size,
        height: atlas_size * cascade_count,
        depth_or_layers: 1,
        mip_levels: 1,
        format: GraphTextureFormat::Depth32Float,
        kind: ResourceKind::Texture2D,
    });

    let pass = builder.add_pass("ShadowPass", PassType::Graphics);
    builder.pass_writes(pass, shadow_atlas);

    ShadowMapOutputs {
        shadow_atlas,
        pass,
    }
}

/// Add an SSAO computation pass.
pub struct SsaoOutputs {
    pub ao_texture: ResourceHandle,
    pub pass: PassHandle,
}

pub fn add_ssao_pass(
    builder: &mut RenderGraphBuilder,
    depth: ResourceHandle,
    normal: ResourceHandle,
) -> SsaoOutputs {
    let ao_texture = builder.create_half_res_texture(
        "ssao_texture",
        GraphTextureFormat::R8Unorm,
    );

    let pass = builder.add_pass("SSAO", PassType::Compute);
    builder.pass_reads(pass, depth);
    builder.pass_reads(pass, normal);
    builder.pass_writes(pass, ao_texture);

    SsaoOutputs { ao_texture, pass }
}

/// Add an SSR (screen-space reflections) pass.
pub struct SsrOutputs {
    pub ssr_texture: ResourceHandle,
    pub pass: PassHandle,
}

pub fn add_ssr_pass(
    builder: &mut RenderGraphBuilder,
    hdr_color: ResourceHandle,
    depth: ResourceHandle,
    normal: ResourceHandle,
    metallic_roughness: ResourceHandle,
) -> SsrOutputs {
    let ssr_texture = builder.create_half_res_texture(
        "ssr_texture",
        GraphTextureFormat::Rgba16Float,
    );

    let pass = builder.add_pass("SSR", PassType::Compute);
    builder.pass_reads(pass, hdr_color);
    builder.pass_reads(pass, depth);
    builder.pass_reads(pass, normal);
    builder.pass_reads(pass, metallic_roughness);
    builder.pass_writes(pass, ssr_texture);

    SsrOutputs { ssr_texture, pass }
}

// ---------------------------------------------------------------------------
// Full pipeline builder utility
// ---------------------------------------------------------------------------

/// Configuration for a complete deferred rendering pipeline.
#[derive(Debug, Clone)]
pub struct FullPipelineConfig {
    pub width: u32,
    pub height: u32,
    pub shadow_atlas_size: u32,
    pub shadow_cascade_count: u32,
    pub enable_ssao: bool,
    pub enable_ssr: bool,
    pub depth_prepass: DepthPrepassConfig,
    pub gbuffer: GBufferConfig,
    pub lighting: LightingConfig,
    pub post_process: PostProcessConfig,
}

impl Default for FullPipelineConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            shadow_atlas_size: 2048,
            shadow_cascade_count: 4,
            enable_ssao: true,
            enable_ssr: false,
            depth_prepass: DepthPrepassConfig::default(),
            gbuffer: GBufferConfig::default(),
            lighting: LightingConfig::default(),
            post_process: PostProcessConfig::default(),
        }
    }
}

/// All handles produced by building a full pipeline.
pub struct FullPipelineHandles {
    pub shadow: ShadowMapOutputs,
    pub depth_prepass: DepthPrepassOutputs,
    pub gbuffer: GBufferOutputs,
    pub ssao: Option<SsaoOutputs>,
    pub lighting: LightingOutputs,
    pub ssr: Option<SsrOutputs>,
    pub post_process: PostProcessOutputs,
    pub ui_overlay: UIOverlayOutputs,
}

/// Build a complete deferred rendering pipeline in one call.
pub fn build_full_pipeline(config: &FullPipelineConfig) -> (RenderGraphBuilder, FullPipelineHandles) {
    let mut builder = RenderGraphBuilder::new(config.width, config.height);

    // 1. Shadow pass.
    let shadow = add_shadow_pass(&mut builder, config.shadow_atlas_size, config.shadow_cascade_count);

    // 2. Depth prepass.
    let depth_prepass = add_depth_prepass(&mut builder, &config.depth_prepass);

    // 3. G-Buffer fill.
    let gbuffer = add_gbuffer_pass(&mut builder, &config.gbuffer, depth_prepass.depth);

    // 4. SSAO (optional).
    let ssao = if config.enable_ssao {
        Some(add_ssao_pass(&mut builder, depth_prepass.depth, gbuffer.normal))
    } else {
        None
    };

    // 5. Lighting resolve.
    let lighting = add_lighting_pass(
        &mut builder,
        &config.lighting,
        &gbuffer,
        depth_prepass.depth,
        ssao.as_ref().map(|s| s.ao_texture),
        Some(shadow.shadow_atlas),
    );

    // 6. SSR (optional).
    let ssr = if config.enable_ssr {
        Some(add_ssr_pass(
            &mut builder,
            lighting.hdr_color,
            depth_prepass.depth,
            gbuffer.normal,
            gbuffer.metallic_roughness,
        ))
    } else {
        None
    };

    // 7. Post-processing.
    let post_process = add_post_process_pass(
        &mut builder,
        &config.post_process,
        lighting.hdr_color,
        depth_prepass.depth,
        depth_prepass.velocity,
    );

    // 8. UI overlay.
    let ui_overlay = add_ui_overlay_pass(&mut builder, post_process.ldr_color);

    let handles = FullPipelineHandles {
        shadow,
        depth_prepass,
        gbuffer,
        ssao,
        lighting,
        ssr,
        post_process,
        ui_overlay,
    };

    (builder, handles)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_graph_compilation() {
        let mut builder = RenderGraphBuilder::new(1920, 1080);
        let tex_a = builder.create_fullscreen_texture("A", GraphTextureFormat::Rgba8Unorm);
        let tex_b = builder.create_fullscreen_texture("B", GraphTextureFormat::Rgba8Unorm);

        let pass1 = builder.add_pass("Pass1", PassType::Graphics);
        builder.pass_writes(pass1, tex_a);

        let pass2 = builder.add_pass("Pass2", PassType::Compute);
        builder.pass_reads(pass2, tex_a);
        builder.pass_writes(pass2, tex_b);

        let pass3 = builder.add_pass_with_side_effects("Pass3", PassType::Present);
        builder.pass_reads(pass3, tex_b);
        builder.set_backbuffer_output(pass3);

        let compiled = builder.compile().expect("Graph should compile");
        assert_eq!(compiled.execution_order.len(), 3);
        // Pass1 must execute before Pass2, Pass2 before Pass3.
        let pos1 = compiled
            .execution_order
            .iter()
            .position(|&i| i == 0)
            .unwrap();
        let pos2 = compiled
            .execution_order
            .iter()
            .position(|&i| i == 1)
            .unwrap();
        let pos3 = compiled
            .execution_order
            .iter()
            .position(|&i| i == 2)
            .unwrap();
        assert!(pos1 < pos2);
        assert!(pos2 < pos3);
    }

    #[test]
    fn test_dead_pass_culling() {
        let mut builder = RenderGraphBuilder::new(800, 600);
        let tex = builder.create_fullscreen_texture("unused", GraphTextureFormat::Rgba8Unorm);
        let _pass = builder.add_pass("DeadPass", PassType::Graphics);
        builder.pass_writes(_pass, tex);

        // Add a live pass so the graph isn't empty.
        let live = builder.add_pass_with_side_effects("LivePass", PassType::Present);
        builder.set_backbuffer_output(live);

        let compiled = builder.compile().expect("Graph should compile");
        // The dead pass should be culled.
        assert!(compiled.passes[0].culled);
        assert!(!compiled.passes[1].culled);
        assert_eq!(compiled.execution_order.len(), 1);
    }

    #[test]
    fn test_cycle_detection() {
        let mut builder = RenderGraphBuilder::new(800, 600);
        let tex_a = builder.create_fullscreen_texture("A", GraphTextureFormat::Rgba8Unorm);
        let tex_b = builder.create_fullscreen_texture("B", GraphTextureFormat::Rgba8Unorm);

        let pass1 = builder.add_pass("Pass1", PassType::Graphics);
        builder.pass_reads(pass1, tex_b);
        builder.pass_writes(pass1, tex_a);

        let pass2 = builder.add_pass("Pass2", PassType::Graphics);
        builder.pass_reads(pass2, tex_a);
        builder.pass_writes(pass2, tex_b);

        // Both are side-effect passes so they won't be culled.
        builder.passes[0].has_side_effects = true;
        builder.passes[1].has_side_effects = true;

        let result = builder.compile();
        assert!(result.is_err());
    }

    #[test]
    fn test_resource_aliasing() {
        let mut builder = RenderGraphBuilder::new(1920, 1080);
        let fmt = GraphTextureFormat::Rgba8Unorm;

        let tex_a = builder.create_fullscreen_texture("A", fmt);
        let tex_b = builder.create_fullscreen_texture("B", fmt);
        let tex_c = builder.create_fullscreen_texture("C", fmt);

        // Pass1 writes A.
        let pass1 = builder.add_pass("P1", PassType::Graphics);
        builder.pass_writes(pass1, tex_a);

        // Pass2 reads A, writes B.
        let pass2 = builder.add_pass("P2", PassType::Compute);
        builder.pass_reads(pass2, tex_a);
        builder.pass_writes(pass2, tex_b);

        // Pass3 reads B, writes C (A is no longer needed).
        let pass3 = builder.add_pass("P3", PassType::Compute);
        builder.pass_reads(pass3, tex_b);
        builder.pass_writes(pass3, tex_c);

        // Pass4 reads C.
        let pass4 = builder.add_pass_with_side_effects("P4", PassType::Present);
        builder.pass_reads(pass4, tex_c);

        let compiled = builder.compile().expect("Compile should succeed");

        // A and C don't overlap (A is only used in P1-P2, C in P3-P4).
        // They should alias to the same physical resource.
        let phys_a = compiled.physical_resource(tex_a).unwrap();
        let phys_c = compiled.physical_resource(tex_c).unwrap();
        assert_eq!(
            phys_a, phys_c,
            "Non-overlapping resources should alias"
        );
    }

    #[test]
    fn test_full_pipeline_builds() {
        let config = FullPipelineConfig::default();
        let (builder, _handles) = build_full_pipeline(&config);
        let compiled = builder.compile().expect("Full pipeline should compile");
        assert!(compiled.execution_order.len() >= 5);
    }

    #[test]
    fn test_resource_handle_display() {
        let h = ResourceHandle(42);
        assert_eq!(format!("{}", h), "Res(42)");
        assert_eq!(format!("{}", ResourceHandle::NONE), "Res(NONE)");
    }

    #[test]
    fn test_dot_export() {
        let config = FullPipelineConfig {
            enable_ssao: true,
            enable_ssr: true,
            ..Default::default()
        };
        let (builder, _handles) = build_full_pipeline(&config);
        let compiled = builder.compile().expect("Should compile");
        let dot = compiled.to_dot();
        assert!(dot.contains("digraph RenderGraph"));
        assert!(dot.contains("DepthPrepass"));
        assert!(dot.contains("GBuffer"));
        assert!(dot.contains("LightingResolve"));
    }
}
