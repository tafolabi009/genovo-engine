// engine/render/src/mesh_instancing.rs
//
// Hardware mesh instancing system for the Genovo engine.
//
// Provides efficient rendering of many identical meshes with per-instance
// variation through GPU instanced draw calls:
//
// - Per-instance data: transform, color, custom float4 parameters.
// - Instance buffer management with dynamic growth and defragmentation.
// - LOD selection per instance based on screen-space size.
// - Frustum culling per instance batch for GPU load reduction.
// - Instance grouping by material and mesh for batched draws.
// - Indirect draw support for GPU-driven rendering pipelines.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default initial capacity for instance buffers.
const DEFAULT_INSTANCE_CAPACITY: usize = 1024;

/// Maximum instances per draw call (to avoid exceeding buffer limits).
const MAX_INSTANCES_PER_DRAW: u32 = 65536;

/// Growth factor when resizing instance buffers.
const BUFFER_GROWTH_FACTOR: f32 = 1.5;

/// Minimum buffer size in instances.
const MIN_BUFFER_SIZE: usize = 64;

// ---------------------------------------------------------------------------
// Per-Instance Data
// ---------------------------------------------------------------------------

/// Per-instance data sent to the GPU.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct InstanceData {
    /// Model matrix (4x4, stored column-major).
    pub model_matrix: [[f32; 4]; 4],
    /// Instance color tint (RGBA).
    pub color: [f32; 4],
    /// Custom parameters (application-defined).
    pub custom_params: [f32; 4],
    /// LOD level for this instance (0 = highest detail).
    pub lod_level: u32,
    /// Visibility flags (bitfield).
    pub visibility_flags: u32,
    /// Unique instance identifier.
    pub instance_id: u32,
    /// Padding for alignment.
    pub _padding: u32,
}

impl InstanceData {
    /// Create a new instance with an identity transform and white color.
    pub fn identity() -> Self {
        Self {
            model_matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            color: [1.0, 1.0, 1.0, 1.0],
            custom_params: [0.0; 4],
            lod_level: 0,
            visibility_flags: 0xFFFFFFFF,
            instance_id: 0,
            _padding: 0,
        }
    }

    /// Create an instance from a position and scale.
    pub fn from_position_scale(position: [f32; 3], scale: f32) -> Self {
        Self {
            model_matrix: [
                [scale, 0.0, 0.0, 0.0],
                [0.0, scale, 0.0, 0.0],
                [0.0, 0.0, scale, 0.0],
                [position[0], position[1], position[2], 1.0],
            ],
            color: [1.0, 1.0, 1.0, 1.0],
            custom_params: [0.0; 4],
            lod_level: 0,
            visibility_flags: 0xFFFFFFFF,
            instance_id: 0,
            _padding: 0,
        }
    }

    /// Create an instance from a full transform (position, rotation quaternion, scale).
    pub fn from_transform(position: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> Self {
        let qx = rotation[0];
        let qy = rotation[1];
        let qz = rotation[2];
        let qw = rotation[3];

        // Quaternion to rotation matrix.
        let xx = qx * qx;
        let yy = qy * qy;
        let zz = qz * qz;
        let xy = qx * qy;
        let xz = qx * qz;
        let yz = qy * qz;
        let wx = qw * qx;
        let wy = qw * qy;
        let wz = qw * qz;

        Self {
            model_matrix: [
                [(1.0 - 2.0 * (yy + zz)) * scale[0], 2.0 * (xy + wz) * scale[0], 2.0 * (xz - wy) * scale[0], 0.0],
                [2.0 * (xy - wz) * scale[1], (1.0 - 2.0 * (xx + zz)) * scale[1], 2.0 * (yz + wx) * scale[1], 0.0],
                [2.0 * (xz + wy) * scale[2], 2.0 * (yz - wx) * scale[2], (1.0 - 2.0 * (xx + yy)) * scale[2], 0.0],
                [position[0], position[1], position[2], 1.0],
            ],
            color: [1.0, 1.0, 1.0, 1.0],
            custom_params: [0.0; 4],
            lod_level: 0,
            visibility_flags: 0xFFFFFFFF,
            instance_id: 0,
            _padding: 0,
        }
    }

    /// Set the instance color.
    pub fn with_color(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.color = [r, g, b, a];
        self
    }

    /// Set custom parameters.
    pub fn with_custom(mut self, params: [f32; 4]) -> Self {
        self.custom_params = params;
        self
    }

    /// Set the LOD level.
    pub fn with_lod(mut self, level: u32) -> Self {
        self.lod_level = level;
        self
    }

    /// Extract the position from the model matrix.
    pub fn position(&self) -> [f32; 3] {
        [self.model_matrix[3][0], self.model_matrix[3][1], self.model_matrix[3][2]]
    }

    /// Extract the approximate uniform scale from the model matrix.
    pub fn approximate_scale(&self) -> f32 {
        let sx = (self.model_matrix[0][0] * self.model_matrix[0][0]
            + self.model_matrix[0][1] * self.model_matrix[0][1]
            + self.model_matrix[0][2] * self.model_matrix[0][2])
            .sqrt();
        sx
    }

    /// Size of a single instance data struct in bytes.
    pub fn stride() -> usize {
        std::mem::size_of::<Self>()
    }
}

// ---------------------------------------------------------------------------
// Instance Buffer
// ---------------------------------------------------------------------------

/// Manages a GPU instance buffer with CPU-side shadow data.
#[derive(Debug)]
pub struct InstanceBuffer {
    /// CPU-side instance data.
    pub data: Vec<InstanceData>,
    /// Current number of active instances.
    pub count: usize,
    /// Buffer capacity (number of instances).
    pub capacity: usize,
    /// Whether the buffer data has been modified since last upload.
    pub dirty: bool,
    /// GPU buffer handle (opaque, set by renderer).
    pub gpu_handle: u64,
    /// Total bytes uploaded to GPU.
    pub gpu_size_bytes: usize,
    /// Label for debugging.
    pub label: String,
}

impl InstanceBuffer {
    /// Create a new instance buffer with the given capacity.
    pub fn new(label: &str, capacity: usize) -> Self {
        let cap = capacity.max(MIN_BUFFER_SIZE);
        Self {
            data: Vec::with_capacity(cap),
            count: 0,
            capacity: cap,
            dirty: true,
            gpu_handle: 0,
            gpu_size_bytes: 0,
            label: label.to_string(),
        }
    }

    /// Add an instance to the buffer. Returns the index.
    pub fn add(&mut self, instance: InstanceData) -> usize {
        let idx = self.count;
        if idx < self.data.len() {
            self.data[idx] = instance;
        } else {
            self.data.push(instance);
        }
        self.count += 1;
        self.dirty = true;

        if self.count > self.capacity {
            self.capacity = (self.capacity as f32 * BUFFER_GROWTH_FACTOR) as usize;
        }

        idx
    }

    /// Remove an instance by swapping with the last instance.
    pub fn remove_swap(&mut self, index: usize) -> bool {
        if index >= self.count {
            return false;
        }
        self.count -= 1;
        if index < self.count {
            self.data[index] = self.data[self.count];
        }
        self.dirty = true;
        true
    }

    /// Update an existing instance.
    pub fn update(&mut self, index: usize, instance: InstanceData) -> bool {
        if index >= self.count {
            return false;
        }
        self.data[index] = instance;
        self.dirty = true;
        true
    }

    /// Clear all instances.
    pub fn clear(&mut self) {
        self.count = 0;
        self.dirty = true;
    }

    /// Returns the active instance count.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns the raw byte data for upload to GPU.
    pub fn as_bytes(&self) -> &[u8] {
        if self.count == 0 {
            return &[];
        }
        let ptr = self.data.as_ptr() as *const u8;
        let len = self.count * InstanceData::stride();
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// Mark the buffer as uploaded (no longer dirty).
    pub fn mark_uploaded(&mut self) {
        self.dirty = false;
        self.gpu_size_bytes = self.count * InstanceData::stride();
    }

    /// Get a reference to the instance at the given index.
    pub fn get(&self, index: usize) -> Option<&InstanceData> {
        if index < self.count {
            Some(&self.data[index])
        } else {
            None
        }
    }

    /// Get a mutable reference to the instance at the given index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut InstanceData> {
        if index < self.count {
            self.dirty = true;
            Some(&mut self.data[index])
        } else {
            None
        }
    }

    /// Defragment the buffer by compacting to the active count.
    pub fn defragment(&mut self) {
        self.data.truncate(self.count);
        self.data.shrink_to_fit();
        self.capacity = self.data.capacity();
        self.dirty = true;
    }
}

// ---------------------------------------------------------------------------
// LOD Selection
// ---------------------------------------------------------------------------

/// LOD level definition for instanced meshes.
#[derive(Debug, Clone)]
pub struct InstanceLodLevel {
    /// LOD index (0 = highest detail).
    pub level: u32,
    /// Minimum screen-space size to use this LOD (in pixels).
    pub min_screen_size: f32,
    /// Maximum screen-space size to use this LOD (in pixels).
    pub max_screen_size: f32,
    /// Mesh handle for this LOD.
    pub mesh_handle: u64,
    /// Vertex count at this LOD (for statistics).
    pub vertex_count: u32,
    /// Triangle count at this LOD (for statistics).
    pub triangle_count: u32,
}

/// LOD configuration for an instanced mesh group.
#[derive(Debug, Clone)]
pub struct InstanceLodConfig {
    /// LOD levels sorted by detail (index 0 = highest detail).
    pub levels: Vec<InstanceLodLevel>,
    /// Screen-space size bias (multiplier).
    pub screen_size_bias: f32,
    /// Hysteresis factor to prevent LOD popping.
    pub hysteresis: f32,
    /// Whether to fade between LOD levels.
    pub cross_fade: bool,
    /// Cross-fade duration in seconds.
    pub cross_fade_duration: f32,
    /// Maximum LOD level allowed (clamped).
    pub max_lod: u32,
}

impl Default for InstanceLodConfig {
    fn default() -> Self {
        Self {
            levels: Vec::new(),
            screen_size_bias: 1.0,
            hysteresis: 0.1,
            cross_fade: false,
            cross_fade_duration: 0.3,
            max_lod: 4,
        }
    }
}

impl InstanceLodConfig {
    /// Select the LOD level for a given screen-space size.
    pub fn select_lod(&self, screen_size: f32) -> u32 {
        let adjusted = screen_size * self.screen_size_bias;
        for level in &self.levels {
            if adjusted >= level.min_screen_size && adjusted < level.max_screen_size {
                return level.level;
            }
        }
        self.levels.last().map(|l| l.level).unwrap_or(0)
    }

    /// Select LOD with hysteresis (requires previous LOD level).
    pub fn select_lod_hysteresis(&self, screen_size: f32, previous_lod: u32) -> u32 {
        let base_lod = self.select_lod(screen_size);
        if base_lod == previous_lod {
            return base_lod;
        }

        // Apply hysteresis: require a larger change to switch.
        let threshold = if base_lod > previous_lod {
            screen_size * (1.0 + self.hysteresis)
        } else {
            screen_size * (1.0 - self.hysteresis)
        };

        self.select_lod(threshold)
    }

    /// Get the vertex count for a specific LOD level.
    pub fn vertex_count_at_lod(&self, lod: u32) -> u32 {
        self.levels.iter().find(|l| l.level == lod).map(|l| l.vertex_count).unwrap_or(0)
    }

    /// Get the triangle count for a specific LOD level.
    pub fn triangle_count_at_lod(&self, lod: u32) -> u32 {
        self.levels.iter().find(|l| l.level == lod).map(|l| l.triangle_count).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Frustum Culling for Instance Batches
// ---------------------------------------------------------------------------

/// A simple frustum defined by six planes for culling.
#[derive(Debug, Clone, Copy)]
pub struct CullFrustum {
    /// Frustum planes [left, right, bottom, top, near, far].
    /// Each plane is (A, B, C, D) where Ax + By + Cz + D >= 0 means inside.
    pub planes: [[f32; 4]; 6],
}

impl CullFrustum {
    /// Create a frustum from a view-projection matrix.
    pub fn from_view_projection(vp: &[[f32; 4]; 4]) -> Self {
        let mut planes = [[0.0f32; 4]; 6];

        // Left plane.
        planes[0] = [
            vp[0][3] + vp[0][0], vp[1][3] + vp[1][0],
            vp[2][3] + vp[2][0], vp[3][3] + vp[3][0],
        ];
        // Right plane.
        planes[1] = [
            vp[0][3] - vp[0][0], vp[1][3] - vp[1][0],
            vp[2][3] - vp[2][0], vp[3][3] - vp[3][0],
        ];
        // Bottom plane.
        planes[2] = [
            vp[0][3] + vp[0][1], vp[1][3] + vp[1][1],
            vp[2][3] + vp[2][1], vp[3][3] + vp[3][1],
        ];
        // Top plane.
        planes[3] = [
            vp[0][3] - vp[0][1], vp[1][3] - vp[1][1],
            vp[2][3] - vp[2][1], vp[3][3] - vp[3][1],
        ];
        // Near plane.
        planes[4] = [
            vp[0][3] + vp[0][2], vp[1][3] + vp[1][2],
            vp[2][3] + vp[2][2], vp[3][3] + vp[3][2],
        ];
        // Far plane.
        planes[5] = [
            vp[0][3] - vp[0][2], vp[1][3] - vp[1][2],
            vp[2][3] - vp[2][2], vp[3][3] - vp[3][2],
        ];

        // Normalize planes.
        for plane in &mut planes {
            let len = (plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]).sqrt();
            if len > 1e-8 {
                plane[0] /= len;
                plane[1] /= len;
                plane[2] /= len;
                plane[3] /= len;
            }
        }

        Self { planes }
    }

    /// Test if a sphere (center + radius) is inside or intersecting the frustum.
    pub fn test_sphere(&self, center: [f32; 3], radius: f32) -> bool {
        for plane in &self.planes {
            let dist = plane[0] * center[0] + plane[1] * center[1]
                + plane[2] * center[2] + plane[3];
            if dist < -radius {
                return false;
            }
        }
        true
    }

    /// Test if an AABB is inside or intersecting the frustum.
    pub fn test_aabb(&self, min: [f32; 3], max: [f32; 3]) -> bool {
        for plane in &self.planes {
            let px = if plane[0] >= 0.0 { max[0] } else { min[0] };
            let py = if plane[1] >= 0.0 { max[1] } else { min[1] };
            let pz = if plane[2] >= 0.0 { max[2] } else { min[2] };
            let dist = plane[0] * px + plane[1] * py + plane[2] * pz + plane[3];
            if dist < 0.0 {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Instance Batch
// ---------------------------------------------------------------------------

/// A batch of instances sharing the same mesh and material.
#[derive(Debug)]
pub struct InstanceBatch {
    /// Unique batch identifier.
    pub id: u64,
    /// Mesh handle.
    pub mesh_handle: u64,
    /// Material handle.
    pub material_handle: u64,
    /// Instance buffer for this batch.
    pub buffer: InstanceBuffer,
    /// LOD configuration.
    pub lod_config: InstanceLodConfig,
    /// Bounding sphere center (in world space).
    pub bounds_center: [f32; 3],
    /// Bounding sphere radius.
    pub bounds_radius: f32,
    /// AABB min corner.
    pub aabb_min: [f32; 3],
    /// AABB max corner.
    pub aabb_max: [f32; 3],
    /// Whether bounds need recalculation.
    pub bounds_dirty: bool,
    /// Whether this batch is visible (after culling).
    pub visible: bool,
    /// Cast shadows.
    pub cast_shadows: bool,
    /// Receive shadows.
    pub receive_shadows: bool,
    /// Rendering layer mask.
    pub render_layer: u32,
}

impl InstanceBatch {
    /// Create a new instance batch.
    pub fn new(id: u64, mesh_handle: u64, material_handle: u64) -> Self {
        Self {
            id,
            mesh_handle,
            material_handle,
            buffer: InstanceBuffer::new(&format!("batch_{}", id), DEFAULT_INSTANCE_CAPACITY),
            lod_config: InstanceLodConfig::default(),
            bounds_center: [0.0; 3],
            bounds_radius: 0.0,
            aabb_min: [f32::MAX; 3],
            aabb_max: [f32::MIN; 3],
            bounds_dirty: true,
            visible: true,
            cast_shadows: true,
            receive_shadows: true,
            render_layer: 0xFFFFFFFF,
        }
    }

    /// Add an instance to this batch.
    pub fn add_instance(&mut self, instance: InstanceData) -> usize {
        let idx = self.buffer.add(instance);
        self.bounds_dirty = true;
        idx
    }

    /// Remove an instance from this batch.
    pub fn remove_instance(&mut self, index: usize) -> bool {
        let result = self.buffer.remove_swap(index);
        if result {
            self.bounds_dirty = true;
        }
        result
    }

    /// Recompute the bounding volumes from all instances.
    pub fn recompute_bounds(&mut self, mesh_radius: f32) {
        if self.buffer.is_empty() {
            self.bounds_center = [0.0; 3];
            self.bounds_radius = 0.0;
            self.aabb_min = [0.0; 3];
            self.aabb_max = [0.0; 3];
            self.bounds_dirty = false;
            return;
        }

        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];

        for i in 0..self.buffer.count {
            let pos = self.buffer.data[i].position();
            let scale = self.buffer.data[i].approximate_scale();
            let r = mesh_radius * scale;

            for axis in 0..3 {
                min[axis] = min[axis].min(pos[axis] - r);
                max[axis] = max[axis].max(pos[axis] + r);
            }
        }

        self.aabb_min = min;
        self.aabb_max = max;

        self.bounds_center = [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ];

        let half_extent = [
            (max[0] - min[0]) * 0.5,
            (max[1] - min[1]) * 0.5,
            (max[2] - min[2]) * 0.5,
        ];
        self.bounds_radius = (half_extent[0] * half_extent[0]
            + half_extent[1] * half_extent[1]
            + half_extent[2] * half_extent[2])
            .sqrt();

        self.bounds_dirty = false;
    }

    /// Perform frustum culling for this batch.
    pub fn cull(&mut self, frustum: &CullFrustum) {
        if self.bounds_dirty {
            return; // Cannot cull without valid bounds.
        }
        self.visible = frustum.test_sphere(self.bounds_center, self.bounds_radius);
    }

    /// Returns the number of instances in this batch.
    pub fn instance_count(&self) -> usize {
        self.buffer.len()
    }
}

// ---------------------------------------------------------------------------
// Indirect Draw Arguments
// ---------------------------------------------------------------------------

/// Indirect draw arguments compatible with GPU indirect draw commands.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct IndirectDrawArgs {
    /// Number of indices per instance (for indexed draws).
    pub index_count: u32,
    /// Number of instances.
    pub instance_count: u32,
    /// First index in the index buffer.
    pub first_index: u32,
    /// Vertex offset.
    pub vertex_offset: i32,
    /// First instance (instance ID offset).
    pub first_instance: u32,
}

impl IndirectDrawArgs {
    /// Create new indirect draw arguments.
    pub fn new(index_count: u32, instance_count: u32) -> Self {
        Self {
            index_count,
            instance_count,
            first_index: 0,
            vertex_offset: 0,
            first_instance: 0,
        }
    }

    /// Size of this struct in bytes.
    pub fn stride() -> usize {
        std::mem::size_of::<Self>()
    }
}

// ---------------------------------------------------------------------------
// Instancing Manager
// ---------------------------------------------------------------------------

/// High-level instancing manager that organises batches by mesh and material.
#[derive(Debug)]
pub struct InstancingManager {
    /// All active batches, keyed by batch ID.
    pub batches: HashMap<u64, InstanceBatch>,
    /// Mapping from (mesh, material) to batch ID.
    pub batch_lookup: HashMap<(u64, u64), u64>,
    /// Next batch ID to assign.
    next_batch_id: u64,
    /// Total instance count across all batches.
    pub total_instances: usize,
    /// Total visible instances after culling.
    pub visible_instances: usize,
    /// Indirect draw argument buffer (for GPU-driven rendering).
    pub indirect_args: Vec<IndirectDrawArgs>,
    /// Whether to enable per-instance culling (more expensive).
    pub per_instance_culling: bool,
    /// Statistics.
    pub stats: InstancingStats,
}

impl InstancingManager {
    /// Create a new instancing manager.
    pub fn new() -> Self {
        Self {
            batches: HashMap::new(),
            batch_lookup: HashMap::new(),
            next_batch_id: 1,
            total_instances: 0,
            visible_instances: 0,
            indirect_args: Vec::new(),
            per_instance_culling: false,
            stats: InstancingStats::default(),
        }
    }

    /// Get or create a batch for the given mesh and material combination.
    pub fn get_or_create_batch(&mut self, mesh: u64, material: u64) -> u64 {
        if let Some(&id) = self.batch_lookup.get(&(mesh, material)) {
            return id;
        }

        let id = self.next_batch_id;
        self.next_batch_id += 1;

        let batch = InstanceBatch::new(id, mesh, material);
        self.batches.insert(id, batch);
        self.batch_lookup.insert((mesh, material), id);
        id
    }

    /// Add an instance to the appropriate batch.
    pub fn add_instance(&mut self, mesh: u64, material: u64, instance: InstanceData) -> (u64, usize) {
        let batch_id = self.get_or_create_batch(mesh, material);
        let idx = self.batches.get_mut(&batch_id).unwrap().add_instance(instance);
        self.total_instances += 1;
        (batch_id, idx)
    }

    /// Remove an instance from a batch.
    pub fn remove_instance(&mut self, batch_id: u64, index: usize) -> bool {
        if let Some(batch) = self.batches.get_mut(&batch_id) {
            if batch.remove_instance(index) {
                self.total_instances -= 1;
                return true;
            }
        }
        false
    }

    /// Remove an empty batch.
    pub fn remove_batch(&mut self, batch_id: u64) {
        if let Some(batch) = self.batches.remove(&batch_id) {
            self.total_instances -= batch.instance_count();
            self.batch_lookup.remove(&(batch.mesh_handle, batch.material_handle));
        }
    }

    /// Perform frustum culling on all batches.
    pub fn cull_batches(&mut self, frustum: &CullFrustum) {
        self.visible_instances = 0;
        for batch in self.batches.values_mut() {
            batch.cull(frustum);
            if batch.visible {
                self.visible_instances += batch.instance_count();
            }
        }
    }

    /// Recompute bounds for all dirty batches.
    pub fn update_bounds(&mut self, mesh_radii: &HashMap<u64, f32>) {
        for batch in self.batches.values_mut() {
            if batch.bounds_dirty {
                let radius = mesh_radii.get(&batch.mesh_handle).copied().unwrap_or(1.0);
                batch.recompute_bounds(radius);
            }
        }
    }

    /// Generate indirect draw arguments for all visible batches.
    pub fn generate_indirect_args(&mut self, mesh_index_counts: &HashMap<u64, u32>) {
        self.indirect_args.clear();
        for batch in self.batches.values() {
            if !batch.visible || batch.buffer.is_empty() {
                continue;
            }
            let index_count = mesh_index_counts.get(&batch.mesh_handle).copied().unwrap_or(0);
            self.indirect_args.push(IndirectDrawArgs::new(
                index_count,
                batch.buffer.count as u32,
            ));
        }
    }

    /// Returns the number of active batches.
    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    /// Returns the number of visible batches.
    pub fn visible_batch_count(&self) -> usize {
        self.batches.values().filter(|b| b.visible).count()
    }

    /// Update statistics.
    pub fn update_stats(&mut self) {
        self.stats.batch_count = self.batches.len() as u32;
        self.stats.total_instances = self.total_instances as u32;
        self.stats.visible_instances = self.visible_instances as u32;
        self.stats.draw_calls = self.indirect_args.len() as u32;
        self.stats.total_buffer_bytes = self.batches.values()
            .map(|b| b.buffer.count * InstanceData::stride())
            .sum();
    }

    /// Remove all empty batches.
    pub fn cleanup_empty_batches(&mut self) {
        let empty_ids: Vec<u64> = self.batches
            .iter()
            .filter(|(_, b)| b.buffer.is_empty())
            .map(|(&id, _)| id)
            .collect();

        for id in empty_ids {
            self.remove_batch(id);
        }
    }
}

/// Statistics for the instancing system.
#[derive(Debug, Clone, Default)]
pub struct InstancingStats {
    /// Number of active batches.
    pub batch_count: u32,
    /// Total instances across all batches.
    pub total_instances: u32,
    /// Visible instances after culling.
    pub visible_instances: u32,
    /// Number of draw calls generated.
    pub draw_calls: u32,
    /// Total instance buffer memory in bytes.
    pub total_buffer_bytes: usize,
    /// Instances culled this frame.
    pub instances_culled: u32,
    /// LOD transitions this frame.
    pub lod_transitions: u32,
}
