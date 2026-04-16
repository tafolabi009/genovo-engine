// engine/render/src/compute/mod.rs
//
// GPU Compute Pipeline abstraction for the Genovo engine.
//
// Provides a type-safe, ergonomic wrapper around wgpu compute pipelines
// and common GPU compute operations:
//
// - Compute pass recording and dispatch.
// - Storage/uniform buffer management.
// - Common parallel primitives: reduce, prefix sum, radix sort, histogram.
// - Tiled matrix multiply using shared memory.
// - GPU particle simulation via compute shader.
// - WGSL compute shader templates.
//
// The API is designed for the wgpu backend but the core types are
// backend-agnostic so they can be adapted to Vulkan/DX12/Metal as needed.

use glam::{Mat4, Vec2, Vec3, Vec4};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Compute dispatch parameters
// ---------------------------------------------------------------------------

/// Workgroup size in three dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkgroupSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl WorkgroupSize {
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub const fn linear(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    pub const fn flat_2d(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    /// Total number of invocations per workgroup.
    pub const fn total(&self) -> u32 {
        self.x * self.y * self.z
    }
}

impl Default for WorkgroupSize {
    fn default() -> Self {
        Self::new(64, 1, 1)
    }
}

/// Dispatch parameters: number of workgroups in each dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeDispatch {
    pub workgroups_x: u32,
    pub workgroups_y: u32,
    pub workgroups_z: u32,
}

impl ComputeDispatch {
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self {
            workgroups_x: x,
            workgroups_y: y,
            workgroups_z: z,
        }
    }

    pub const fn linear(x: u32) -> Self {
        Self::new(x, 1, 1)
    }

    pub const fn flat_2d(x: u32, y: u32) -> Self {
        Self::new(x, y, 1)
    }

    /// Computes the dispatch size needed to cover `total_invocations` given
    /// a workgroup size.
    pub fn for_elements(total: u32, workgroup_size: u32) -> Self {
        Self::linear((total + workgroup_size - 1) / workgroup_size)
    }

    /// Computes 2-D dispatch size to cover a `width × height` grid.
    pub fn for_grid_2d(width: u32, height: u32, wg_x: u32, wg_y: u32) -> Self {
        Self::flat_2d(
            (width + wg_x - 1) / wg_x,
            (height + wg_y - 1) / wg_y,
        )
    }

    /// Total number of workgroups dispatched.
    pub const fn total_workgroups(&self) -> u32 {
        self.workgroups_x * self.workgroups_y * self.workgroups_z
    }
}

impl Default for ComputeDispatch {
    fn default() -> Self {
        Self::new(1, 1, 1)
    }
}

// ---------------------------------------------------------------------------
// Buffer descriptors
// ---------------------------------------------------------------------------

/// Usage flags for a compute buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBufferUsage {
    /// Read-only storage buffer.
    StorageRead,
    /// Read-write storage buffer.
    StorageReadWrite,
    /// Uniform buffer.
    Uniform,
    /// Indirect dispatch arguments buffer.
    Indirect,
    /// Staging buffer for CPU read-back.
    Staging,
    /// Combined storage + vertex (e.g. particle positions).
    StorageVertex,
    /// Combined storage + index.
    StorageIndex,
}

/// Descriptor for creating a compute storage/uniform buffer.
#[derive(Debug, Clone)]
pub struct StorageBufferDesc {
    /// Debug label.
    pub label: String,
    /// Size in bytes.
    pub size: u64,
    /// Usage category.
    pub usage: ComputeBufferUsage,
    /// If `true`, the buffer is mapped at creation for initial upload.
    pub mapped_at_creation: bool,
}

impl StorageBufferDesc {
    pub fn new(label: impl Into<String>, size: u64, usage: ComputeBufferUsage) -> Self {
        Self {
            label: label.into(),
            size,
            usage,
            mapped_at_creation: false,
        }
    }

    pub fn with_mapped(mut self) -> Self {
        self.mapped_at_creation = true;
        self
    }
}

/// Handle to a GPU storage buffer managed by the compute system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StorageBufferHandle(pub u64);

/// Concrete storage buffer (wraps metadata; actual GPU handle is in the
/// backend layer).
#[derive(Debug, Clone)]
pub struct StorageBuffer {
    pub handle: StorageBufferHandle,
    pub label: String,
    pub size: u64,
    pub usage: ComputeBufferUsage,
}

// ---------------------------------------------------------------------------
// Compute shader
// ---------------------------------------------------------------------------

/// A compiled compute shader with metadata.
#[derive(Debug, Clone)]
pub struct ComputeShader {
    /// Unique identifier / name.
    pub name: String,
    /// WGSL source code.
    pub source: String,
    /// Entry point function name.
    pub entry_point: String,
    /// Declared workgroup size in the shader.
    pub workgroup_size: WorkgroupSize,
    /// Specialisation constants (name → value).
    pub constants: HashMap<String, f64>,
}

impl ComputeShader {
    pub fn new(
        name: impl Into<String>,
        source: impl Into<String>,
        entry_point: impl Into<String>,
        workgroup_size: WorkgroupSize,
    ) -> Self {
        Self {
            name: name.into(),
            source: source.into(),
            entry_point: entry_point.into(),
            workgroup_size,
            constants: HashMap::new(),
        }
    }

    pub fn with_constant(mut self, name: impl Into<String>, value: f64) -> Self {
        self.constants.insert(name.into(), value);
        self
    }
}

// ---------------------------------------------------------------------------
// Bind group layout
// ---------------------------------------------------------------------------

/// Type of a binding in a compute pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBindingType {
    UniformBuffer,
    StorageBufferReadOnly,
    StorageBufferReadWrite,
    Texture2D,
    StorageTexture2DWrite,
    StorageTexture2DRead,
    Sampler,
}

/// A single binding in a compute bind group layout.
#[derive(Debug, Clone)]
pub struct ComputeBinding {
    /// Binding index.
    pub binding: u32,
    /// Type of binding.
    pub binding_type: ComputeBindingType,
    /// Whether this binding is optional (may be `None` at dispatch time).
    pub optional: bool,
}

/// Layout for a bind group.
#[derive(Debug, Clone)]
pub struct ComputeBindGroupLayout {
    pub group: u32,
    pub bindings: Vec<ComputeBinding>,
}

impl ComputeBindGroupLayout {
    pub fn new(group: u32) -> Self {
        Self {
            group,
            bindings: Vec::new(),
        }
    }

    pub fn add_uniform(mut self, binding: u32) -> Self {
        self.bindings.push(ComputeBinding {
            binding,
            binding_type: ComputeBindingType::UniformBuffer,
            optional: false,
        });
        self
    }

    pub fn add_storage_ro(mut self, binding: u32) -> Self {
        self.bindings.push(ComputeBinding {
            binding,
            binding_type: ComputeBindingType::StorageBufferReadOnly,
            optional: false,
        });
        self
    }

    pub fn add_storage_rw(mut self, binding: u32) -> Self {
        self.bindings.push(ComputeBinding {
            binding,
            binding_type: ComputeBindingType::StorageBufferReadWrite,
            optional: false,
        });
        self
    }

    pub fn add_texture(mut self, binding: u32) -> Self {
        self.bindings.push(ComputeBinding {
            binding,
            binding_type: ComputeBindingType::Texture2D,
            optional: false,
        });
        self
    }

    pub fn add_storage_texture_write(mut self, binding: u32) -> Self {
        self.bindings.push(ComputeBinding {
            binding,
            binding_type: ComputeBindingType::StorageTexture2DWrite,
            optional: false,
        });
        self
    }

    pub fn add_sampler(mut self, binding: u32) -> Self {
        self.bindings.push(ComputeBinding {
            binding,
            binding_type: ComputeBindingType::Sampler,
            optional: false,
        });
        self
    }
}

// ---------------------------------------------------------------------------
// ComputePipeline
// ---------------------------------------------------------------------------

/// Handle to a created compute pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputePipelineHandle(pub u64);

/// Descriptor for creating a compute pipeline.
#[derive(Debug, Clone)]
pub struct ComputePipelineDesc {
    pub label: String,
    pub shader: ComputeShader,
    pub bind_group_layouts: Vec<ComputeBindGroupLayout>,
    pub push_constant_ranges: Vec<PushConstantRange>,
}

/// Push constant range descriptor.
#[derive(Debug, Clone, Copy)]
pub struct PushConstantRange {
    pub offset: u32,
    pub size: u32,
}

/// Represents a compiled compute pipeline ready for dispatch.
#[derive(Debug, Clone)]
pub struct ComputePipeline {
    pub handle: ComputePipelineHandle,
    pub label: String,
    pub workgroup_size: WorkgroupSize,
    pub bind_group_layouts: Vec<ComputeBindGroupLayout>,
}

impl ComputePipeline {
    /// Computes the dispatch parameters needed to process `n` elements.
    pub fn dispatch_for(&self, n: u32) -> ComputeDispatch {
        ComputeDispatch::for_elements(n, self.workgroup_size.x)
    }

    /// Computes the dispatch parameters for a 2-D grid.
    pub fn dispatch_for_2d(&self, width: u32, height: u32) -> ComputeDispatch {
        ComputeDispatch::for_grid_2d(
            width,
            height,
            self.workgroup_size.x,
            self.workgroup_size.y,
        )
    }
}

// ---------------------------------------------------------------------------
// Compute pass / encoder
// ---------------------------------------------------------------------------

/// A recorded compute dispatch command.
#[derive(Debug, Clone)]
pub struct ComputeCommand {
    pub pipeline: ComputePipelineHandle,
    pub dispatch: ComputeDispatch,
    pub push_constants: Option<Vec<u8>>,
    pub label: String,
}

/// Encoder for recording compute dispatches within a pass.
///
/// In a real engine this would encode into a wgpu `CommandEncoder`; here we
/// record the commands for later submission.
#[derive(Debug, Clone)]
pub struct ComputeEncoder {
    pub label: String,
    pub commands: Vec<ComputeCommand>,
    /// Barrier indices: positions in `commands` where a barrier is needed.
    pub barriers: Vec<usize>,
}

impl ComputeEncoder {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            commands: Vec::new(),
            barriers: Vec::new(),
        }
    }

    /// Records a dispatch command.
    pub fn dispatch(
        &mut self,
        pipeline: ComputePipelineHandle,
        dispatch: ComputeDispatch,
        label: impl Into<String>,
    ) {
        self.commands.push(ComputeCommand {
            pipeline,
            dispatch,
            push_constants: None,
            label: label.into(),
        });
    }

    /// Records a dispatch command with push constants.
    pub fn dispatch_with_push_constants(
        &mut self,
        pipeline: ComputePipelineHandle,
        dispatch: ComputeDispatch,
        push_constants: &[u8],
        label: impl Into<String>,
    ) {
        self.commands.push(ComputeCommand {
            pipeline,
            dispatch,
            push_constants: Some(push_constants.to_vec()),
            label: label.into(),
        });
    }

    /// Inserts a memory barrier before the next dispatch, ensuring all prior
    /// writes are visible.
    pub fn barrier(&mut self) {
        self.barriers.push(self.commands.len());
    }

    /// Returns the number of recorded dispatches.
    pub fn dispatch_count(&self) -> usize {
        self.commands.len()
    }
}

/// A complete compute pass containing one or more dispatches.
#[derive(Debug, Clone)]
pub struct ComputePass {
    pub label: String,
    pub encoder: ComputeEncoder,
    pub workgroup_size: WorkgroupSize,
    pub dispatch_size: ComputeDispatch,
}

impl ComputePass {
    pub fn new(
        label: impl Into<String>,
        workgroup_size: WorkgroupSize,
        dispatch_size: ComputeDispatch,
    ) -> Self {
        let label = label.into();
        Self {
            label: label.clone(),
            encoder: ComputeEncoder::new(label),
            workgroup_size,
            dispatch_size,
        }
    }
}

// ---------------------------------------------------------------------------
// Common compute operations
// ---------------------------------------------------------------------------

/// Reduction operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Min,
    Max,
    Product,
}

impl ReduceOp {
    /// Returns the WGSL identity element for this operation.
    pub fn identity_wgsl(&self) -> &'static str {
        match self {
            Self::Sum => "0.0",
            Self::Min => "3.402823466e+38",
            Self::Max => "-3.402823466e+38",
            Self::Product => "1.0",
        }
    }

    /// Returns the WGSL binary operator or function.
    pub fn op_wgsl(&self) -> &'static str {
        match self {
            Self::Sum => "+",
            Self::Min => "min",
            Self::Max => "max",
            Self::Product => "*",
        }
    }

    /// Returns whether the operation uses a function call or infix operator.
    pub fn is_function(&self) -> bool {
        matches!(self, Self::Min | Self::Max)
    }
}

/// Configuration for a parallel reduce operation.
#[derive(Debug, Clone)]
pub struct ParallelReduceConfig {
    /// Number of input elements.
    pub element_count: u32,
    /// Operation to perform.
    pub op: ReduceOp,
    /// Data type in WGSL (e.g. "f32", "u32", "vec4<f32>").
    pub data_type: String,
    /// Workgroup size (must be power of 2).
    pub workgroup_size: u32,
}

impl ParallelReduceConfig {
    pub fn new(element_count: u32, op: ReduceOp) -> Self {
        Self {
            element_count,
            op,
            data_type: "f32".to_string(),
            workgroup_size: 256,
        }
    }

    pub fn with_type(mut self, ty: impl Into<String>) -> Self {
        self.data_type = ty.into();
        self
    }

    /// Generates the WGSL shader source for this reduction.
    pub fn generate_shader(&self) -> String {
        let op = self.op;
        let ty = &self.data_type;
        let wg = self.workgroup_size;
        let identity = op.identity_wgsl();

        let combine = if op.is_function() {
            format!("{f}(a, b)", f = op.op_wgsl())
        } else {
            format!("a {op} b", op = op.op_wgsl())
        };

        format!(
            r#"// Parallel reduce ({op:?}) — generated by Genovo compute system
struct Params {{
    element_count: u32,
}};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_data: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> output_data: array<{ty}>;

var<workgroup> shared_data: array<{ty}, {wg}>;

fn combine(a: {ty}, b: {ty}) -> {ty} {{
    return {combine};
}}

@compute @workgroup_size({wg}, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {{
    let idx = gid.x;
    let local_idx = lid.x;

    // Load from global memory (or identity if out of bounds).
    if idx < params.element_count {{
        shared_data[local_idx] = input_data[idx];
    }} else {{
        shared_data[local_idx] = {ty}({identity});
    }}
    workgroupBarrier();

    // Tree reduction in shared memory.
    var stride: u32 = {wg}u / 2u;
    loop {{
        if stride == 0u {{ break; }}
        if local_idx < stride {{
            shared_data[local_idx] = combine(
                shared_data[local_idx],
                shared_data[local_idx + stride]
            );
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}

    // First thread writes result.
    if local_idx == 0u {{
        output_data[wid.x] = shared_data[0];
    }}
}}
"#,
        )
    }

    /// Returns the number of dispatches needed for a full reduction.
    pub fn num_passes(&self) -> u32 {
        let mut remaining = self.element_count;
        let mut passes = 0;
        while remaining > 1 {
            remaining = (remaining + self.workgroup_size - 1) / self.workgroup_size;
            passes += 1;
        }
        passes
    }
}

// ---------------------------------------------------------------------------
// Prefix sum (exclusive scan)
// ---------------------------------------------------------------------------

/// Configuration for a parallel prefix sum (exclusive scan).
#[derive(Debug, Clone)]
pub struct PrefixSumConfig {
    /// Number of input elements.
    pub element_count: u32,
    /// Data type in WGSL.
    pub data_type: String,
    /// Workgroup size (must be power of 2).
    pub workgroup_size: u32,
}

impl PrefixSumConfig {
    pub fn new(element_count: u32) -> Self {
        Self {
            element_count,
            data_type: "u32".to_string(),
            workgroup_size: 256,
        }
    }

    pub fn with_type(mut self, ty: impl Into<String>) -> Self {
        self.data_type = ty.into();
        self
    }

    /// Generates the WGSL source for the Blelloch prefix sum (up-sweep +
    /// down-sweep within a workgroup).
    pub fn generate_local_scan_shader(&self) -> String {
        let ty = &self.data_type;
        let wg = self.workgroup_size;

        format!(
            r#"// Prefix sum (exclusive scan) — workgroup-local pass
// Generated by Genovo compute system

struct Params {{
    element_count: u32,
}};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<{ty}>;

var<workgroup> temp: array<{ty}, {wg_x2}>;

@compute @workgroup_size({wg}, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {{
    let n = {wg}u * 2u;
    let base = wid.x * n;
    let ai = lid.x;
    let bi = lid.x + {wg}u;

    // Load into shared memory.
    if base + ai < params.element_count {{
        temp[ai] = data[base + ai];
    }} else {{
        temp[ai] = {ty}(0);
    }}
    if base + bi < params.element_count {{
        temp[bi] = data[base + bi];
    }} else {{
        temp[bi] = {ty}(0);
    }}

    // Up-sweep (reduce) phase.
    var offset: u32 = 1u;
    var d: u32 = n >> 1u;
    loop {{
        workgroupBarrier();
        if d == 0u {{ break; }}
        if lid.x < d {{
            let idx_a = offset * (2u * lid.x + 1u) - 1u;
            let idx_b = offset * (2u * lid.x + 2u) - 1u;
            temp[idx_b] = temp[idx_b] + temp[idx_a];
        }}
        offset = offset << 1u;
        d = d >> 1u;
    }}

    // Store block sum and clear last element.
    if lid.x == 0u {{
        block_sums[wid.x] = temp[n - 1u];
        temp[n - 1u] = {ty}(0);
    }}

    // Down-sweep phase.
    d = 1u;
    offset = n >> 1u;
    loop {{
        if offset == 0u {{ break; }}
        workgroupBarrier();
        if lid.x < d {{
            let idx_a = offset * (2u * lid.x + 1u) - 1u;
            let idx_b = offset * (2u * lid.x + 2u) - 1u;
            let t = temp[idx_a];
            temp[idx_a] = temp[idx_b];
            temp[idx_b] = temp[idx_b] + t;
        }}
        d = d << 1u;
        offset = offset >> 1u;
    }}
    workgroupBarrier();

    // Write results back.
    if base + ai < params.element_count {{
        data[base + ai] = temp[ai];
    }}
    if base + bi < params.element_count {{
        data[base + bi] = temp[bi];
    }}
}}
"#,
            wg_x2 = wg * 2,
        )
    }

    /// Generates the WGSL source for the propagation pass that adds block
    /// sums to each block's elements.
    pub fn generate_propagate_shader(&self) -> String {
        let ty = &self.data_type;
        let wg = self.workgroup_size;

        format!(
            r#"// Prefix sum — block-sum propagation pass
// Generated by Genovo compute system

struct Params {{
    element_count: u32,
}};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<{ty}>;
@group(0) @binding(2) var<storage, read> block_sums: array<{ty}>;

@compute @workgroup_size({wg}, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {{
    if wid.x == 0u {{ return; }} // First block needs no addition.
    let idx = gid.x;
    if idx >= params.element_count {{ return; }}

    data[idx] = data[idx] + block_sums[wid.x];
}}
"#,
        )
    }
}

// ---------------------------------------------------------------------------
// Radix sort
// ---------------------------------------------------------------------------

/// Configuration for a GPU radix sort.
#[derive(Debug, Clone)]
pub struct RadixSortConfig {
    /// Number of elements to sort.
    pub element_count: u32,
    /// Key type ("u32" or "i32").
    pub key_type: String,
    /// Whether to also rearrange a value buffer.
    pub has_values: bool,
    /// Value type (only used if `has_values`).
    pub value_type: String,
    /// Bits per radix pass (typically 4 = 16 buckets).
    pub radix_bits: u32,
    /// Workgroup size.
    pub workgroup_size: u32,
}

impl RadixSortConfig {
    pub fn new(element_count: u32) -> Self {
        Self {
            element_count,
            key_type: "u32".to_string(),
            has_values: false,
            value_type: "u32".to_string(),
            radix_bits: 4,
            workgroup_size: 256,
        }
    }

    pub fn with_values(mut self, value_type: impl Into<String>) -> Self {
        self.has_values = true;
        self.value_type = value_type.into();
        self
    }

    /// Number of radix passes needed (32-bit key / radix_bits).
    pub fn num_passes(&self) -> u32 {
        (32 + self.radix_bits - 1) / self.radix_bits
    }

    /// Number of histogram buckets per pass.
    pub fn bucket_count(&self) -> u32 {
        1 << self.radix_bits
    }

    /// Generates the WGSL shader for the histogram counting pass.
    pub fn generate_histogram_shader(&self) -> String {
        let wg = self.workgroup_size;
        let buckets = self.bucket_count();

        format!(
            r#"// Radix sort — histogram pass
// Generated by Genovo compute system

struct Params {{
    element_count: u32,
    radix_shift: u32,
}};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>>;

var<workgroup> local_hist: array<atomic<u32>, {buckets}>;

@compute @workgroup_size({wg}, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {{
    // Clear local histogram.
    if lid.x < {buckets}u {{
        atomicStore(&local_hist[lid.x], 0u);
    }}
    workgroupBarrier();

    // Count.
    let idx = gid.x;
    if idx < params.element_count {{
        let key = keys[idx];
        let bucket = (key >> params.radix_shift) & {mask}u;
        atomicAdd(&local_hist[bucket], 1u);
    }}
    workgroupBarrier();

    // Write to global histogram.
    if lid.x < {buckets}u {{
        let global_idx = wid.x * {buckets}u + lid.x;
        atomicAdd(&histogram[global_idx], atomicLoad(&local_hist[lid.x]));
    }}
}}
"#,
            mask = buckets - 1,
        )
    }

    /// Generates the WGSL shader for the scatter (reorder) pass.
    pub fn generate_scatter_shader(&self) -> String {
        let wg = self.workgroup_size;
        let buckets = self.bucket_count();
        let value_scatter = if self.has_values {
            "    output_values[dest] = input_values[idx];"
        } else {
            ""
        };
        let value_bindings = if self.has_values {
            r#"@group(0) @binding(4) var<storage, read> input_values: array<u32>;
@group(0) @binding(5) var<storage, read_write> output_values: array<u32>;"#
        } else {
            ""
        };

        format!(
            r#"// Radix sort — scatter pass
// Generated by Genovo compute system

struct Params {{
    element_count: u32,
    radix_shift: u32,
}};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_keys: array<u32>;
@group(0) @binding(3) var<storage, read> prefix_sums: array<u32>;
{value_bindings}

var<workgroup> local_offsets: array<atomic<u32>, {buckets}>;

@compute @workgroup_size({wg}, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {{
    // Init local offsets from global prefix sums.
    if lid.x < {buckets}u {{
        let hist_idx = wid.x * {buckets}u + lid.x;
        atomicStore(&local_offsets[lid.x], prefix_sums[hist_idx]);
    }}
    workgroupBarrier();

    let idx = gid.x;
    if idx >= params.element_count {{ return; }}

    let key = input_keys[idx];
    let bucket = (key >> params.radix_shift) & {mask}u;
    let dest = atomicAdd(&local_offsets[bucket], 1u);

    output_keys[dest] = key;
{value_scatter}
}}
"#,
            mask = buckets - 1,
        )
    }
}

// ---------------------------------------------------------------------------
// Histogram
// ---------------------------------------------------------------------------

/// Configuration for a GPU histogram computation.
#[derive(Debug, Clone)]
pub struct HistogramConfig {
    /// Number of input elements.
    pub element_count: u32,
    /// Number of bins.
    pub bin_count: u32,
    /// Minimum value of the input range.
    pub range_min: f32,
    /// Maximum value of the input range.
    pub range_max: f32,
    /// Workgroup size.
    pub workgroup_size: u32,
}

impl HistogramConfig {
    pub fn new(element_count: u32, bin_count: u32, range_min: f32, range_max: f32) -> Self {
        Self {
            element_count,
            bin_count,
            range_min,
            range_max,
            workgroup_size: 256,
        }
    }

    /// Generates the WGSL shader for histogram computation using shared
    /// memory atomics.
    pub fn generate_shader(&self) -> String {
        let wg = self.workgroup_size;
        let bins = self.bin_count;

        format!(
            r#"// Histogram compute shader
// Generated by Genovo compute system

struct Params {{
    element_count: u32,
    bin_count: u32,
    range_min: f32,
    range_max: f32,
}};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>>;

var<workgroup> local_bins: array<atomic<u32>, {bins}>;

@compute @workgroup_size({wg}, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    // Clear local bins.
    var clear_idx = lid.x;
    loop {{
        if clear_idx >= {bins}u {{ break; }}
        atomicStore(&local_bins[clear_idx], 0u);
        clear_idx += {wg}u;
    }}
    workgroupBarrier();

    let idx = gid.x;
    if idx < params.element_count {{
        let value = input_data[idx];
        let range = params.range_max - params.range_min;
        let normalised = clamp((value - params.range_min) / range, 0.0, 0.999999);
        let bin = u32(normalised * f32(params.bin_count));
        atomicAdd(&local_bins[bin], 1u);
    }}
    workgroupBarrier();

    // Merge local bins into global histogram.
    var merge_idx = lid.x;
    loop {{
        if merge_idx >= {bins}u {{ break; }}
        let count = atomicLoad(&local_bins[merge_idx]);
        if count > 0u {{
            atomicAdd(&histogram[merge_idx], count);
        }}
        merge_idx += {wg}u;
    }}
}}
"#,
        )
    }
}

// ---------------------------------------------------------------------------
// Matrix multiply (tiled, shared memory)
// ---------------------------------------------------------------------------

/// Configuration for a tiled matrix multiplication C = A × B.
#[derive(Debug, Clone)]
pub struct MatrixMultiplyConfig {
    /// Rows of A (and C).
    pub m: u32,
    /// Columns of A / rows of B.
    pub k: u32,
    /// Columns of B (and C).
    pub n: u32,
    /// Tile size (square tiles in shared memory).
    pub tile_size: u32,
}

impl MatrixMultiplyConfig {
    pub fn new(m: u32, k: u32, n: u32) -> Self {
        Self {
            m,
            k,
            n,
            tile_size: 16,
        }
    }

    pub fn with_tile_size(mut self, tile: u32) -> Self {
        self.tile_size = tile;
        self
    }

    /// Dispatch parameters for this matmul.
    pub fn dispatch(&self) -> ComputeDispatch {
        ComputeDispatch::flat_2d(
            (self.n + self.tile_size - 1) / self.tile_size,
            (self.m + self.tile_size - 1) / self.tile_size,
        )
    }

    /// Generates the WGSL shader for tiled matrix multiply.
    pub fn generate_shader(&self) -> String {
        let ts = self.tile_size;

        format!(
            r#"// Tiled matrix multiply C = A * B
// Generated by Genovo compute system
// A: M x K, B: K x N, C: M x N

struct Params {{
    M: u32,
    K: u32,
    N: u32,
}};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

const TILE_SIZE: u32 = {ts}u;

var<workgroup> tile_a: array<f32, {ts_sq}>;
var<workgroup> tile_b: array<f32, {ts_sq}>;

@compute @workgroup_size({ts}, {ts}, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {{
    let row = wid.y * TILE_SIZE + lid.y;
    let col = wid.x * TILE_SIZE + lid.x;

    var sum: f32 = 0.0;
    let num_tiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t++) {{
        // Load tile of A.
        let a_col = t * TILE_SIZE + lid.x;
        if row < params.M && a_col < params.K {{
            tile_a[lid.y * TILE_SIZE + lid.x] = A[row * params.K + a_col];
        }} else {{
            tile_a[lid.y * TILE_SIZE + lid.x] = 0.0;
        }}

        // Load tile of B.
        let b_row = t * TILE_SIZE + lid.y;
        if b_row < params.K && col < params.N {{
            tile_b[lid.y * TILE_SIZE + lid.x] = B[b_row * params.N + col];
        }} else {{
            tile_b[lid.y * TILE_SIZE + lid.x] = 0.0;
        }}

        workgroupBarrier();

        // Accumulate.
        for (var k: u32 = 0u; k < TILE_SIZE; k++) {{
            sum += tile_a[lid.y * TILE_SIZE + k] * tile_b[k * TILE_SIZE + lid.x];
        }}

        workgroupBarrier();
    }}

    // Write result.
    if row < params.M && col < params.N {{
        C[row * params.N + col] = sum;
    }}
}}
"#,
            ts_sq = ts * ts,
        )
    }
}

// ---------------------------------------------------------------------------
// GPU particle simulation
// ---------------------------------------------------------------------------

/// Particle structure used in GPU compute simulation.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuParticle {
    pub position: Vec3,
    pub lifetime: f32,
    pub velocity: Vec3,
    pub age: f32,
    pub color: Vec4,
    pub size: f32,
    pub rotation: f32,
    pub _padding: [f32; 2],
}

impl GpuParticle {
    pub fn zeroed() -> Self {
        Self {
            position: Vec3::ZERO,
            lifetime: 0.0,
            velocity: Vec3::ZERO,
            age: 0.0,
            color: Vec4::ONE,
            size: 1.0,
            rotation: 0.0,
            _padding: [0.0; 2],
        }
    }

    /// Size of the struct in bytes (for buffer sizing).
    pub const SIZE: u64 = std::mem::size_of::<Self>() as u64;
}

/// Particle emitter parameters for compute shader.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuEmitterParams {
    pub emit_position: Vec3,
    pub emit_rate: f32,
    pub emit_direction: Vec3,
    pub emit_spread: f32,
    pub min_speed: f32,
    pub max_speed: f32,
    pub min_lifetime: f32,
    pub max_lifetime: f32,
    pub gravity: Vec3,
    pub drag: f32,
    pub start_size: f32,
    pub end_size: f32,
    pub start_color: Vec4,
    pub end_color: Vec4,
}

impl Default for GpuEmitterParams {
    fn default() -> Self {
        Self {
            emit_position: Vec3::ZERO,
            emit_rate: 100.0,
            emit_direction: Vec3::Y,
            emit_spread: 0.5,
            min_speed: 1.0,
            max_speed: 3.0,
            min_lifetime: 1.0,
            max_lifetime: 3.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            drag: 0.1,
            start_size: 0.1,
            end_size: 0.01,
            start_color: Vec4::ONE,
            end_color: Vec4::new(1.0, 1.0, 1.0, 0.0),
        }
    }
}

/// Configuration for GPU particle simulation.
#[derive(Debug, Clone)]
pub struct GpuParticleSimulation {
    /// Maximum number of particles in the pool.
    pub max_particles: u32,
    /// Current number of alive particles.
    pub alive_count: u32,
    /// Emitter parameters.
    pub emitter: GpuEmitterParams,
    /// Workgroup size for the simulation dispatch.
    pub workgroup_size: u32,
    /// Whether to use GPU-side emission (vs CPU emit, GPU update).
    pub gpu_emission: bool,
    /// Sort particles by depth for correct alpha blending.
    pub depth_sort: bool,
    /// Enable collision with a ground plane at Y = ground_y.
    pub ground_collision: bool,
    /// Ground plane Y coordinate.
    pub ground_y: f32,
    /// Coefficient of restitution for ground collision.
    pub restitution: f32,
}

impl GpuParticleSimulation {
    pub fn new(max_particles: u32) -> Self {
        Self {
            max_particles,
            alive_count: 0,
            emitter: GpuEmitterParams::default(),
            workgroup_size: 256,
            gpu_emission: true,
            depth_sort: false,
            ground_collision: false,
            ground_y: 0.0,
            restitution: 0.5,
        }
    }

    /// Computes the dispatch size for the simulation update pass.
    pub fn dispatch(&self) -> ComputeDispatch {
        ComputeDispatch::for_elements(self.max_particles, self.workgroup_size)
    }

    /// Computes the storage buffer size needed for all particles.
    pub fn particle_buffer_size(&self) -> u64 {
        GpuParticle::SIZE * self.max_particles as u64
    }

    /// Generates the WGSL compute shader for particle update.
    pub fn generate_update_shader(&self) -> String {
        let wg = self.workgroup_size;
        let collision = if self.ground_collision {
            format!(
                r#"
    // Ground collision.
    if p.position.y < params.ground_y {{
        p.position.y = params.ground_y;
        p.velocity.y = -p.velocity.y * params.restitution;
        p.velocity = p.velocity * 0.95; // friction
    }}"#
            )
        } else {
            String::new()
        };

        format!(
            r#"// GPU particle simulation — update pass
// Generated by Genovo compute system

struct SimParams {{
    dt: f32,
    time: f32,
    max_particles: u32,
    alive_count: u32,
    gravity: vec3<f32>,
    drag: f32,
    ground_y: f32,
    restitution: f32,
    start_size: f32,
    end_size: f32,
    start_color: vec4<f32>,
    end_color: vec4<f32>,
}};

struct Particle {{
    position: vec3<f32>,
    lifetime: f32,
    velocity: vec3<f32>,
    age: f32,
    color: vec4<f32>,
    size: f32,
    rotation: f32,
    _pad0: f32,
    _pad1: f32,
}};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> alive_counter: atomic<u32>;

@compute @workgroup_size({wg}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx >= params.max_particles {{ return; }}

    var p = particles[idx];

    // Skip dead particles.
    if p.age >= p.lifetime {{ return; }}

    // Advance age.
    p.age += params.dt;
    if p.age >= p.lifetime {{
        // Kill particle.
        p.color.w = 0.0;
        particles[idx] = p;
        return;
    }}

    let t = p.age / p.lifetime; // normalised age [0, 1]

    // Physics integration (symplectic Euler).
    let accel = params.gravity - p.velocity * params.drag;
    p.velocity += accel * params.dt;
    p.position += p.velocity * params.dt;
{collision}

    // Interpolate visual properties.
    p.size = mix(params.start_size, params.end_size, t);
    p.color = mix(params.start_color, params.end_color, t);

    // Simple rotation.
    p.rotation += params.dt * 1.0;

    particles[idx] = p;

    // Count alive.
    atomicAdd(&alive_counter, 1u);
}}
"#,
        )
    }

    /// Generates the WGSL compute shader for particle emission.
    pub fn generate_emit_shader(&self) -> String {
        let wg = self.workgroup_size;

        format!(
            r#"// GPU particle simulation — emit pass
// Generated by Genovo compute system

struct EmitParams {{
    emit_count: u32,
    time: f32,
    emit_position: vec3<f32>,
    emit_direction: vec3<f32>,
    emit_spread: f32,
    min_speed: f32,
    max_speed: f32,
    min_lifetime: f32,
    max_lifetime: f32,
    max_particles: u32,
    start_color: vec4<f32>,
    start_size: f32,
}};

struct Particle {{
    position: vec3<f32>,
    lifetime: f32,
    velocity: vec3<f32>,
    age: f32,
    color: vec4<f32>,
    size: f32,
    rotation: f32,
    _pad0: f32,
    _pad1: f32,
}};

@group(0) @binding(0) var<uniform> params: EmitParams;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> dead_list: array<u32>;
@group(0) @binding(3) var<storage, read_write> dead_count: atomic<u32>;

// Simple hash-based PRNG.
fn pcg_hash(input: u32) -> u32 {{
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}}

fn rand_f32(seed: ptr<function, u32>) -> f32 {{
    *seed = pcg_hash(*seed);
    return f32(*seed) / 4294967295.0;
}}

fn rand_unit_sphere(seed: ptr<function, u32>) -> vec3<f32> {{
    let z = rand_f32(seed) * 2.0 - 1.0;
    let phi = rand_f32(seed) * 6.283185;
    let r = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}}

@compute @workgroup_size({wg}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let emit_idx = gid.x;
    if emit_idx >= params.emit_count {{ return; }}

    // Grab a dead particle slot.
    let slot_idx = atomicSub(&dead_count, 1u);
    if slot_idx == 0u {{
        atomicAdd(&dead_count, 1u); // undo
        return;
    }}
    let particle_idx = dead_list[slot_idx - 1u];
    if particle_idx >= params.max_particles {{ return; }}

    // Initialize particle.
    var seed = emit_idx * 1099u + u32(params.time * 1000.0);
    let spread = rand_unit_sphere(&seed) * params.emit_spread;
    let dir = normalize(params.emit_direction + spread);
    let speed = mix(params.min_speed, params.max_speed, rand_f32(&seed));
    let lifetime = mix(params.min_lifetime, params.max_lifetime, rand_f32(&seed));

    var p: Particle;
    p.position = params.emit_position + rand_unit_sphere(&seed) * 0.1;
    p.lifetime = lifetime;
    p.velocity = dir * speed;
    p.age = 0.0;
    p.color = params.start_color;
    p.size = params.start_size;
    p.rotation = rand_f32(&seed) * 6.283185;
    p._pad0 = 0.0;
    p._pad1 = 0.0;

    particles[particle_idx] = p;
}}
"#,
        )
    }
}

// ---------------------------------------------------------------------------
// Compute system manager
// ---------------------------------------------------------------------------

/// Top-level manager for compute resources within a frame.
#[derive(Debug)]
pub struct ComputeSystem {
    /// All registered compute pipelines.
    pub pipelines: HashMap<String, ComputePipeline>,
    /// All registered storage buffers.
    pub buffers: HashMap<String, StorageBuffer>,
    /// Next handle ID for pipelines.
    next_pipeline_id: u64,
    /// Next handle ID for buffers.
    next_buffer_id: u64,
    /// Maximum supported workgroup size (from device limits).
    pub max_workgroup_size: u32,
    /// Maximum supported dispatch size per dimension.
    pub max_dispatch_size: u32,
    /// Maximum storage buffer binding size.
    pub max_storage_buffer_size: u64,
}

impl ComputeSystem {
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
            buffers: HashMap::new(),
            next_pipeline_id: 1,
            next_buffer_id: 1,
            max_workgroup_size: 256,
            max_dispatch_size: 65535,
            max_storage_buffer_size: 128 * 1024 * 1024, // 128 MiB default
        }
    }

    /// Registers a compute pipeline and returns its handle.
    pub fn register_pipeline(&mut self, desc: ComputePipelineDesc) -> ComputePipelineHandle {
        let handle = ComputePipelineHandle(self.next_pipeline_id);
        self.next_pipeline_id += 1;

        let pipeline = ComputePipeline {
            handle,
            label: desc.label.clone(),
            workgroup_size: desc.shader.workgroup_size,
            bind_group_layouts: desc.bind_group_layouts,
        };

        self.pipelines.insert(desc.label, pipeline);
        handle
    }

    /// Registers a storage buffer and returns its handle.
    pub fn register_buffer(&mut self, desc: StorageBufferDesc) -> StorageBufferHandle {
        let handle = StorageBufferHandle(self.next_buffer_id);
        self.next_buffer_id += 1;

        let buffer = StorageBuffer {
            handle,
            label: desc.label.clone(),
            size: desc.size,
            usage: desc.usage,
        };

        self.buffers.insert(desc.label, buffer);
        handle
    }

    /// Looks up a pipeline by name.
    pub fn get_pipeline(&self, name: &str) -> Option<&ComputePipeline> {
        self.pipelines.get(name)
    }

    /// Looks up a buffer by name.
    pub fn get_buffer(&self, name: &str) -> Option<&StorageBuffer> {
        self.buffers.get(name)
    }

    /// Creates a `ComputeEncoder` for recording dispatches.
    pub fn create_encoder(&self, label: impl Into<String>) -> ComputeEncoder {
        ComputeEncoder::new(label)
    }

    /// Validates that a dispatch does not exceed device limits.
    pub fn validate_dispatch(&self, dispatch: &ComputeDispatch) -> Result<(), String> {
        if dispatch.workgroups_x > self.max_dispatch_size {
            return Err(format!(
                "Dispatch X ({}) exceeds max ({})",
                dispatch.workgroups_x, self.max_dispatch_size
            ));
        }
        if dispatch.workgroups_y > self.max_dispatch_size {
            return Err(format!(
                "Dispatch Y ({}) exceeds max ({})",
                dispatch.workgroups_y, self.max_dispatch_size
            ));
        }
        if dispatch.workgroups_z > self.max_dispatch_size {
            return Err(format!(
                "Dispatch Z ({}) exceeds max ({})",
                dispatch.workgroups_z, self.max_dispatch_size
            ));
        }
        Ok(())
    }

    /// Validates that a buffer size is within device limits.
    pub fn validate_buffer_size(&self, size: u64) -> Result<(), String> {
        if size > self.max_storage_buffer_size {
            return Err(format!(
                "Buffer size ({size}) exceeds max ({})",
                self.max_storage_buffer_size
            ));
        }
        Ok(())
    }
}

impl Default for ComputeSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Indirect dispatch buffer helpers
// ---------------------------------------------------------------------------

/// Layout of an indirect dispatch arguments buffer.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct IndirectDispatchArgs {
    pub workgroups_x: u32,
    pub workgroups_y: u32,
    pub workgroups_z: u32,
}

impl IndirectDispatchArgs {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self {
            workgroups_x: x,
            workgroups_y: y,
            workgroups_z: z,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// WGSL shader template library
// ---------------------------------------------------------------------------

/// Collection of common WGSL utility functions that can be `#include`d in
/// compute shaders.
pub const WGSL_COMMON_UTILS: &str = r#"
// -----------------------------------------------------------------------
// Common WGSL utilities (Genovo Engine)
// -----------------------------------------------------------------------

// PCG hash for pseudo-random numbers.
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(seed: ptr<function, u32>) -> f32 {
    *seed = pcg_hash(*seed);
    return f32(*seed) / 4294967295.0;
}

fn rand_vec2(seed: ptr<function, u32>) -> vec2<f32> {
    return vec2<f32>(rand_f32(seed), rand_f32(seed));
}

fn rand_vec3(seed: ptr<function, u32>) -> vec3<f32> {
    return vec3<f32>(rand_f32(seed), rand_f32(seed), rand_f32(seed));
}

fn rand_unit_sphere(seed: ptr<function, u32>) -> vec3<f32> {
    let z = rand_f32(seed) * 2.0 - 1.0;
    let phi = rand_f32(seed) * 6.283185307;
    let r = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

fn rand_unit_hemisphere(seed: ptr<function, u32>, normal: vec3<f32>) -> vec3<f32> {
    var v = rand_unit_sphere(seed);
    if dot(v, normal) < 0.0 { v = -v; }
    return v;
}

// Remapping utility.
fn remap(value: f32, from_min: f32, from_max: f32, to_min: f32, to_max: f32) -> f32 {
    let t = clamp((value - from_min) / (from_max - from_min), 0.0, 1.0);
    return to_min + t * (to_max - to_min);
}

// Smooth-step.
fn smooth_step(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Quintic smooth-step (C2 continuous).
fn smoother_step(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Linear interpolation.
fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

fn lerp_vec3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

fn lerp_vec4(a: vec4<f32>, b: vec4<f32>, t: f32) -> vec4<f32> {
    return a + (b - a) * t;
}

// Atomic workgroup min/max for f32 encoded as u32 (bit-casting).
fn f32_to_sortable_u32(f: f32) -> u32 {
    let bits = bitcast<u32>(f);
    let mask = select(0x80000000u, 0xFFFFFFFFu, (bits & 0x80000000u) != 0u);
    return bits ^ mask;
}

fn sortable_u32_to_f32(u: u32) -> f32 {
    let mask = select(0x80000000u, 0xFFFFFFFFu, (u & 0x80000000u) == 0u);
    return bitcast<f32>(u ^ mask);
}
"#;

/// WGSL template for a clear/fill buffer shader.
pub const WGSL_CLEAR_BUFFER: &str = r#"
// Clear / fill a storage buffer with a constant value.

struct Params {
    count: u32,
    value: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> buffer: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.count { return; }
    buffer[gid.x] = params.value;
}
"#;

/// WGSL template for a buffer copy shader.
pub const WGSL_BUFFER_COPY: &str = r#"
// Copy one storage buffer to another.

struct Params {
    count: u32,
    src_offset: u32,
    dst_offset: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.count { return; }
    dst[params.dst_offset + gid.x] = src[params.src_offset + gid.x];
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_for_elements() {
        let d = ComputeDispatch::for_elements(1000, 256);
        assert_eq!(d.workgroups_x, 4); // ceil(1000/256) = 4
        assert_eq!(d.workgroups_y, 1);
    }

    #[test]
    fn dispatch_for_grid_2d() {
        let d = ComputeDispatch::for_grid_2d(1920, 1080, 8, 8);
        assert_eq!(d.workgroups_x, 240);
        assert_eq!(d.workgroups_y, 135);
    }

    #[test]
    fn reduce_shader_generates() {
        let config = ParallelReduceConfig::new(1024, ReduceOp::Sum);
        let shader = config.generate_shader();
        assert!(shader.contains("fn main"));
        assert!(shader.contains("shared_data"));
    }

    #[test]
    fn prefix_sum_shader_generates() {
        let config = PrefixSumConfig::new(4096);
        let scan = config.generate_local_scan_shader();
        let prop = config.generate_propagate_shader();
        assert!(scan.contains("Up-sweep"));
        assert!(prop.contains("block_sums"));
    }

    #[test]
    fn radix_sort_passes() {
        let config = RadixSortConfig::new(1_000_000);
        assert_eq!(config.num_passes(), 8); // 32 / 4 = 8
        assert_eq!(config.bucket_count(), 16); // 2^4
    }

    #[test]
    fn histogram_shader_generates() {
        let config = HistogramConfig::new(10000, 64, 0.0, 1.0);
        let shader = config.generate_shader();
        assert!(shader.contains("local_bins"));
        assert!(shader.contains("atomicAdd"));
    }

    #[test]
    fn matmul_dispatch() {
        let config = MatrixMultiplyConfig::new(512, 256, 1024);
        let d = config.dispatch();
        assert_eq!(d.workgroups_x, 64); // 1024 / 16
        assert_eq!(d.workgroups_y, 32); // 512 / 16
    }

    #[test]
    fn matmul_shader_generates() {
        let config = MatrixMultiplyConfig::new(64, 64, 64);
        let shader = config.generate_shader();
        assert!(shader.contains("tile_a"));
        assert!(shader.contains("workgroupBarrier"));
    }

    #[test]
    fn particle_sim_shader_generates() {
        let sim = GpuParticleSimulation::new(10000);
        let update = sim.generate_update_shader();
        let emit = sim.generate_emit_shader();
        assert!(update.contains("symplectic Euler"));
        assert!(emit.contains("dead_list"));
    }

    #[test]
    fn compute_system_register() {
        let mut sys = ComputeSystem::new();
        let buf = sys.register_buffer(StorageBufferDesc::new(
            "test_buf",
            1024,
            ComputeBufferUsage::StorageReadWrite,
        ));
        assert_eq!(buf.0, 1);
        assert!(sys.get_buffer("test_buf").is_some());
    }

    #[test]
    fn compute_encoder_record() {
        let mut enc = ComputeEncoder::new("test_pass");
        enc.dispatch(ComputePipelineHandle(1), ComputeDispatch::linear(4), "d1");
        enc.barrier();
        enc.dispatch(ComputePipelineHandle(2), ComputeDispatch::linear(2), "d2");
        assert_eq!(enc.dispatch_count(), 2);
        assert_eq!(enc.barriers.len(), 1);
    }

    #[test]
    fn indirect_dispatch_args_size() {
        assert_eq!(
            std::mem::size_of::<IndirectDispatchArgs>(),
            12
        );
    }
}
