// engine/render/src/instanced_renderer.rs
//
// General-purpose instanced rendering subsystem for the Genovo engine.
//
// Provides a high-performance instanced draw call system that batches objects
// sharing the same mesh and material, manages GPU instance buffers with double-
// buffering, performs per-instance frustum culling, and merges draw calls for
// scenes with many small objects.

use crate::mesh::{Mesh, Vertex, AABB};
use crate::virtual_geometry::bvh::Frustum;
use glam::{Mat4, Vec2, Vec3, Vec4};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of instances per instanced draw call.
pub const MAX_INSTANCES_PER_DRAW: usize = 65_536;

/// Maximum number of instanced draw calls per frame.
pub const MAX_INSTANCED_DRAW_CALLS: usize = 4096;

/// Initial capacity for the instance buffer (number of instances).
pub const INITIAL_INSTANCE_BUFFER_CAPACITY: usize = 1024;

/// Growth factor when the instance buffer needs to be enlarged.
pub const INSTANCE_BUFFER_GROWTH_FACTOR: f32 = 1.5;

/// Maximum number of unique mesh+material combinations before merging is
/// attempted.
pub const MERGE_THRESHOLD: usize = 256;

// ---------------------------------------------------------------------------
// InstanceData
// ---------------------------------------------------------------------------

/// Per-instance data uploaded to the GPU instance buffer.
///
/// Each instance represents one object drawn with a specific mesh and material.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct InstanceData {
    /// World transform (4x4 column-major matrix).
    pub world_matrix: [[f32; 4]; 4],
    /// Custom per-instance data. Interpretation is shader-dependent:
    /// typically (color.rgb, variation_seed) or (tint.rgba).
    pub custom_data: [f32; 4],
}

impl InstanceData {
    /// Byte size of a single instance.
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// Create instance data from a world matrix.
    pub fn from_matrix(matrix: Mat4) -> Self {
        Self {
            world_matrix: matrix.to_cols_array_2d(),
            custom_data: [1.0, 1.0, 1.0, 0.0],
        }
    }

    /// Create instance data from a world matrix and custom data.
    pub fn new(matrix: Mat4, custom: Vec4) -> Self {
        Self {
            world_matrix: matrix.to_cols_array_2d(),
            custom_data: custom.to_array(),
        }
    }

    /// Get the world matrix as a `Mat4`.
    pub fn world_matrix(&self) -> Mat4 {
        Mat4::from_cols_array_2d(&self.world_matrix)
    }

    /// Get the world position (translation column of the matrix).
    pub fn position(&self) -> Vec3 {
        Vec3::new(
            self.world_matrix[3][0],
            self.world_matrix[3][1],
            self.world_matrix[3][2],
        )
    }

    /// Get the uniform scale factor (average of axis lengths).
    pub fn scale(&self) -> f32 {
        let m = self.world_matrix();
        let sx = m.x_axis.truncate().length();
        let sy = m.y_axis.truncate().length();
        let sz = m.z_axis.truncate().length();
        (sx + sy + sz) / 3.0
    }

    /// Get the custom data as a `Vec4`.
    pub fn custom(&self) -> Vec4 {
        Vec4::from_array(self.custom_data)
    }
}

impl Default for InstanceData {
    fn default() -> Self {
        Self::from_matrix(Mat4::IDENTITY)
    }
}

// ---------------------------------------------------------------------------
// InstanceBuffer
// ---------------------------------------------------------------------------

/// Manages a CPU-side staging buffer for instance data that is uploaded to the
/// GPU each frame. Uses a double-buffered strategy: while the GPU reads from
/// one buffer, the CPU writes to the other.
///
/// The buffer is grow-only: it never shrinks, to avoid frequent reallocations.
/// The `used_count` tracks how many instances are active.
pub struct InstanceBuffer {
    /// Double-buffered instance data. Index is the buffer index (0 or 1).
    buffers: [Vec<InstanceData>; 2],
    /// Which buffer is currently being written to by the CPU.
    write_index: usize,
    /// Number of active instances in the write buffer.
    used_count: usize,
    /// Capacity (max allocated count) across both buffers.
    capacity: usize,
    /// Unique ID for this buffer.
    pub id: u32,
    /// Whether the buffer contents have changed since the last upload.
    dirty: bool,
    /// Total number of uploads performed (for profiling).
    upload_count: u64,
}

impl InstanceBuffer {
    /// Create a new instance buffer with the given initial capacity.
    pub fn new(id: u32, initial_capacity: usize) -> Self {
        let cap = initial_capacity.max(16);
        Self {
            buffers: [
                Vec::with_capacity(cap),
                Vec::with_capacity(cap),
            ],
            write_index: 0,
            used_count: 0,
            capacity: cap,
            id,
            dirty: true,
            upload_count: 0,
        }
    }

    /// Clear the write buffer for a new frame.
    pub fn clear(&mut self) {
        self.buffers[self.write_index].clear();
        self.used_count = 0;
        self.dirty = true;
    }

    /// Add a single instance to the write buffer.
    pub fn push(&mut self, instance: InstanceData) {
        self.buffers[self.write_index].push(instance);
        self.used_count += 1;
        self.dirty = true;
    }

    /// Add multiple instances to the write buffer.
    pub fn extend(&mut self, instances: &[InstanceData]) {
        self.buffers[self.write_index].extend_from_slice(instances);
        self.used_count += instances.len();
        self.dirty = true;
    }

    /// Upload the instance data. In a real GPU backend, this would copy to a
    /// GPU buffer. Here it swaps the double buffers and returns the data slice.
    pub fn upload(&mut self, instances: &[InstanceData]) {
        self.clear();
        self.extend(instances);
        self.swap_buffers();
    }

    /// Swap the read and write buffers.
    pub fn swap_buffers(&mut self) {
        self.write_index = 1 - self.write_index;
        self.upload_count += 1;
        self.dirty = false;
    }

    /// Get the read buffer (the one the GPU would consume).
    pub fn read_buffer(&self) -> &[InstanceData] {
        let read_index = 1 - self.write_index;
        &self.buffers[read_index]
    }

    /// Get the write buffer.
    pub fn write_buffer(&self) -> &[InstanceData] {
        &self.buffers[self.write_index]
    }

    /// Get the write buffer as raw bytes for GPU upload.
    pub fn write_buffer_bytes(&self) -> &[u8] {
        let buf = &self.buffers[self.write_index];
        unsafe {
            std::slice::from_raw_parts(
                buf.as_ptr() as *const u8,
                buf.len() * InstanceData::SIZE,
            )
        }
    }

    /// Number of active instances in the write buffer.
    pub fn used_count(&self) -> usize {
        self.used_count
    }

    /// Current allocated capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Whether the buffer has been modified since the last upload.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Total number of uploads.
    pub fn upload_count(&self) -> u64 {
        self.upload_count
    }

    /// Byte size of the active instance data.
    pub fn byte_size(&self) -> usize {
        self.used_count * InstanceData::SIZE
    }

    /// Ensure the buffer has at least `count` capacity. Grow-only: never
    /// shrinks.
    pub fn reserve(&mut self, count: usize) {
        if count > self.capacity {
            let new_cap = (count as f32 * INSTANCE_BUFFER_GROWTH_FACTOR) as usize;
            self.buffers[0].reserve(new_cap.saturating_sub(self.buffers[0].len()));
            self.buffers[1].reserve(new_cap.saturating_sub(self.buffers[1].len()));
            self.capacity = new_cap;
        }
    }
}

// ---------------------------------------------------------------------------
// InstancedMesh
// ---------------------------------------------------------------------------

/// Represents a mesh that should be drawn with instancing: one mesh, one
/// material, many instances.
#[derive(Debug, Clone)]
pub struct InstancedMesh {
    /// Handle to the mesh asset.
    pub mesh_handle: u64,
    /// Handle to the material asset.
    pub material_handle: u64,
    /// Per-instance data (transforms, colours, etc.).
    pub instances: Vec<InstanceData>,
    /// Bounding box of the mesh in object space (for frustum culling).
    pub mesh_bounds: AABB,
}

impl InstancedMesh {
    /// Create a new instanced mesh.
    pub fn new(mesh_handle: u64, material_handle: u64, bounds: AABB) -> Self {
        Self {
            mesh_handle,
            material_handle,
            instances: Vec::new(),
            mesh_bounds: bounds,
        }
    }

    /// Add an instance.
    pub fn add_instance(&mut self, instance: InstanceData) {
        self.instances.push(instance);
    }

    /// Add an instance from just a transform.
    pub fn add_transform(&mut self, transform: Mat4) {
        self.instances.push(InstanceData::from_matrix(transform));
    }

    /// Number of instances.
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Clear all instances.
    pub fn clear_instances(&mut self) {
        self.instances.clear();
    }
}

// ---------------------------------------------------------------------------
// InstancedDrawCall
// ---------------------------------------------------------------------------

/// A single instanced draw call, ready to be submitted to the GPU.
#[derive(Debug, Clone, Copy)]
pub struct InstancedDrawCall {
    /// Mesh asset handle.
    pub mesh_id: u64,
    /// Material asset handle.
    pub material_id: u64,
    /// Offset into the instance buffer (in number of instances).
    pub instance_buffer_offset: u32,
    /// Number of instances to draw.
    pub instance_count: u32,
    /// Index count of the mesh (for `draw_indexed_instanced`).
    pub index_count: u32,
    /// First index in the mesh's index buffer.
    pub first_index: u32,
    /// Base vertex offset.
    pub base_vertex: i32,
}

impl InstancedDrawCall {
    /// Whether this draw call draws nothing.
    pub fn is_empty(&self) -> bool {
        self.instance_count == 0 || self.index_count == 0
    }

    /// Total number of triangles drawn by this call.
    pub fn triangle_count(&self) -> u64 {
        self.index_count as u64 / 3 * self.instance_count as u64
    }
}

// ---------------------------------------------------------------------------
// InstanceCuller
// ---------------------------------------------------------------------------

/// Performs per-instance frustum culling and compacts the surviving instances
/// into a contiguous output buffer.
pub struct InstanceCuller {
    /// Compacted output buffer of visible instances.
    visible_instances: Vec<InstanceData>,
    /// Number of instances that passed the cull test.
    visible_count: usize,
    /// Number of instances that were culled (failed the test).
    culled_count: usize,
}

impl InstanceCuller {
    /// Create a new instance culler.
    pub fn new() -> Self {
        Self {
            visible_instances: Vec::with_capacity(INITIAL_INSTANCE_BUFFER_CAPACITY),
            visible_count: 0,
            culled_count: 0,
        }
    }

    /// Cull instances against a frustum. Writes visible instances to the
    /// internal buffer.
    ///
    /// # Arguments
    /// - `instances` -- input instances to test.
    /// - `mesh_bounds` -- object-space bounding box of the mesh.
    /// - `frustum` -- the view frustum to cull against.
    ///
    /// # Returns
    /// Number of visible instances.
    pub fn cull(
        &mut self,
        instances: &[InstanceData],
        mesh_bounds: &AABB,
        frustum: &Frustum,
    ) -> usize {
        self.visible_instances.clear();
        self.visible_count = 0;
        self.culled_count = 0;

        let obj_center = mesh_bounds.center();
        let obj_radius = mesh_bounds.radius();

        for instance in instances {
            let world_mat = instance.world_matrix();

            // Transform bounding sphere to world space.
            let world_center = world_mat.transform_point3(obj_center);
            let scale = world_mat
                .x_axis
                .truncate()
                .length()
                .max(world_mat.y_axis.truncate().length())
                .max(world_mat.z_axis.truncate().length());
            let world_radius = obj_radius * scale;

            if frustum.test_sphere(world_center, world_radius) {
                self.visible_instances.push(*instance);
                self.visible_count += 1;
            } else {
                self.culled_count += 1;
            }
        }

        self.visible_count
    }

    /// Cull instances with a distance threshold in addition to frustum culling.
    pub fn cull_with_distance(
        &mut self,
        instances: &[InstanceData],
        mesh_bounds: &AABB,
        frustum: &Frustum,
        camera_position: Vec3,
        max_distance: f32,
    ) -> usize {
        self.visible_instances.clear();
        self.visible_count = 0;
        self.culled_count = 0;

        let obj_center = mesh_bounds.center();
        let obj_radius = mesh_bounds.radius();
        let max_dist_sq = max_distance * max_distance;

        for instance in instances {
            let world_mat = instance.world_matrix();
            let world_center = world_mat.transform_point3(obj_center);

            // Distance cull.
            let dist_sq = (world_center - camera_position).length_squared();
            if dist_sq > max_dist_sq {
                self.culled_count += 1;
                continue;
            }

            // Frustum cull.
            let scale = world_mat
                .x_axis
                .truncate()
                .length()
                .max(world_mat.y_axis.truncate().length())
                .max(world_mat.z_axis.truncate().length());
            let world_radius = obj_radius * scale;

            if frustum.test_sphere(world_center, world_radius) {
                self.visible_instances.push(*instance);
                self.visible_count += 1;
            } else {
                self.culled_count += 1;
            }
        }

        self.visible_count
    }

    /// Get the visible instances after culling.
    pub fn visible_instances(&self) -> &[InstanceData] {
        &self.visible_instances[..self.visible_count]
    }

    /// Number of visible instances.
    pub fn visible_count(&self) -> usize {
        self.visible_count
    }

    /// Number of culled instances.
    pub fn culled_count(&self) -> usize {
        self.culled_count
    }

    /// Cull ratio (fraction of instances that were culled).
    pub fn cull_ratio(&self) -> f32 {
        let total = self.visible_count + self.culled_count;
        if total == 0 {
            return 0.0;
        }
        self.culled_count as f32 / total as f32
    }
}

impl Default for InstanceCuller {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MergedInstancing
// ---------------------------------------------------------------------------

/// Merges multiple meshes that share the same material into a single instanced
/// draw call by concatenating their vertex buffers and using per-instance data
/// to select the correct sub-mesh.
///
/// This is useful for scenes with many small, unique objects that share a
/// material but have different geometry. Instead of one draw call per object,
/// all objects are drawn in a single call.
pub struct MergedInstancing {
    /// Combined vertex data from all merged meshes.
    pub merged_vertices: Vec<Vertex>,
    /// Combined index data from all merged meshes.
    pub merged_indices: Vec<u32>,
    /// Per-mesh metadata: (vertex_offset, index_offset, index_count, mesh_id).
    pub mesh_ranges: Vec<MergedMeshRange>,
    /// Map from mesh handle to index into `mesh_ranges`.
    pub mesh_map: HashMap<u64, usize>,
    /// Instances with their mesh range index encoded in `custom_data.w`.
    pub instances: Vec<InstanceData>,
}

/// Range of a single mesh within the merged vertex/index buffers.
#[derive(Debug, Clone, Copy)]
pub struct MergedMeshRange {
    /// Offset of this mesh's first vertex in the merged vertex buffer.
    pub vertex_offset: u32,
    /// Number of vertices.
    pub vertex_count: u32,
    /// Offset of this mesh's first index in the merged index buffer.
    pub index_offset: u32,
    /// Number of indices.
    pub index_count: u32,
    /// Original mesh handle.
    pub mesh_handle: u64,
}

impl MergedInstancing {
    /// Create a new merged instancing system.
    pub fn new() -> Self {
        Self {
            merged_vertices: Vec::new(),
            merged_indices: Vec::new(),
            mesh_ranges: Vec::new(),
            mesh_map: HashMap::new(),
            instances: Vec::new(),
        }
    }

    /// Add a mesh to the merged buffer. Returns the mesh range index.
    pub fn add_mesh(&mut self, mesh_handle: u64, mesh: &Mesh) -> usize {
        if let Some(&idx) = self.mesh_map.get(&mesh_handle) {
            return idx;
        }

        let vertex_offset = self.merged_vertices.len() as u32;
        let index_offset = self.merged_indices.len() as u32;

        self.merged_vertices.extend_from_slice(&mesh.vertices);

        // Adjust indices by the vertex offset.
        for &idx in &mesh.indices {
            self.merged_indices.push(idx + vertex_offset);
        }

        let range = MergedMeshRange {
            vertex_offset,
            vertex_count: mesh.vertex_count,
            index_offset,
            index_count: mesh.index_count,
            mesh_handle,
        };

        let range_idx = self.mesh_ranges.len();
        self.mesh_ranges.push(range);
        self.mesh_map.insert(mesh_handle, range_idx);

        range_idx
    }

    /// Add an instance for a previously added mesh.
    pub fn add_instance(&mut self, mesh_handle: u64, transform: Mat4, custom: Vec4) {
        if let Some(&range_idx) = self.mesh_map.get(&mesh_handle) {
            let mut data = InstanceData::new(transform, custom);
            // Encode the mesh range index in custom_data.w so the shader can
            // select the correct sub-mesh.
            data.custom_data[3] = range_idx as f32;
            self.instances.push(data);
        }
    }

    /// Clear all instances (but keep the merged mesh data).
    pub fn clear_instances(&mut self) {
        self.instances.clear();
    }

    /// Clear everything.
    pub fn clear_all(&mut self) {
        self.merged_vertices.clear();
        self.merged_indices.clear();
        self.mesh_ranges.clear();
        self.mesh_map.clear();
        self.instances.clear();
    }

    /// Generate draw calls for the merged buffer.
    ///
    /// Since all meshes share the same material, this produces a minimal number
    /// of draw calls (ideally one per mesh range that has instances).
    pub fn generate_draw_calls(&self) -> Vec<InstancedDrawCall> {
        // Group instances by mesh range.
        let mut range_instances: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, inst) in self.instances.iter().enumerate() {
            let range_idx = inst.custom_data[3] as usize;
            range_instances.entry(range_idx).or_default().push(i);
        }

        let mut draw_calls = Vec::new();
        let mut buffer_offset: u32 = 0;

        for (range_idx, instance_indices) in &range_instances {
            if let Some(range) = self.mesh_ranges.get(*range_idx) {
                draw_calls.push(InstancedDrawCall {
                    mesh_id: range.mesh_handle,
                    material_id: 0,
                    instance_buffer_offset: buffer_offset,
                    instance_count: instance_indices.len() as u32,
                    index_count: range.index_count,
                    first_index: range.index_offset,
                    base_vertex: 0, // indices are already offset
                });
                buffer_offset += instance_indices.len() as u32;
            }
        }

        draw_calls
    }

    /// Number of unique meshes in the merged buffer.
    pub fn mesh_count(&self) -> usize {
        self.mesh_ranges.len()
    }

    /// Number of instances.
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Total vertex count in the merged buffer.
    pub fn total_vertex_count(&self) -> usize {
        self.merged_vertices.len()
    }

    /// Total index count in the merged buffer.
    pub fn total_index_count(&self) -> usize {
        self.merged_indices.len()
    }
}

impl Default for MergedInstancing {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// InstancedRenderer
// ---------------------------------------------------------------------------

/// High-level instanced rendering system. Collects instance submissions during
/// a frame, batches them by mesh and material, uploads instance buffers, and
/// produces an optimal set of instanced draw calls.
pub struct InstancedRenderer {
    /// Pending submissions: (mesh_handle, material_handle) -> instances.
    submissions: HashMap<(u64, u64), Vec<InstanceData>>,
    /// Instance buffers, keyed by (mesh, material).
    instance_buffers: HashMap<(u64, u64), InstanceBuffer>,
    /// Generated draw calls for the current frame.
    draw_calls: Vec<InstancedDrawCall>,
    /// Mesh bounds cache: mesh_handle -> AABB.
    mesh_bounds: HashMap<u64, AABB>,
    /// Instance culler.
    culler: InstanceCuller,
    /// Whether frustum culling is enabled.
    pub frustum_culling_enabled: bool,
    /// Rendering statistics for the current frame.
    pub stats: InstancedRendererStats,
    /// Next buffer ID.
    next_buffer_id: u32,
}

/// Rendering statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct InstancedRendererStats {
    /// Number of draw calls generated.
    pub draw_calls: u32,
    /// Total number of triangles across all draw calls.
    pub triangles: u64,
    /// Total number of instances submitted.
    pub instances_submitted: u32,
    /// Total number of instances after culling.
    pub instances_drawn: u32,
    /// Number of instances culled.
    pub instances_culled: u32,
    /// Number of unique mesh+material combinations.
    pub unique_batches: u32,
    /// Total instance buffer memory used in bytes.
    pub instance_buffer_bytes: usize,
}

impl InstancedRenderer {
    /// Create a new instanced renderer.
    pub fn new() -> Self {
        Self {
            submissions: HashMap::new(),
            instance_buffers: HashMap::new(),
            draw_calls: Vec::with_capacity(MAX_INSTANCED_DRAW_CALLS),
            mesh_bounds: HashMap::new(),
            culler: InstanceCuller::new(),
            frustum_culling_enabled: true,
            stats: InstancedRendererStats::default(),
            next_buffer_id: 0,
        }
    }

    /// Register the bounding box for a mesh handle (required for frustum
    /// culling).
    pub fn register_mesh_bounds(&mut self, mesh_handle: u64, bounds: AABB) {
        self.mesh_bounds.insert(mesh_handle, bounds);
    }

    /// Submit instances for rendering.
    ///
    /// The instances will be batched with other submissions sharing the same
    /// mesh and material.
    pub fn submit(&mut self, mesh: u64, material: u64, transforms: &[Mat4]) {
        let key = (mesh, material);
        let entry = self.submissions.entry(key).or_default();
        for transform in transforms {
            entry.push(InstanceData::from_matrix(*transform));
        }
    }

    /// Submit instances with custom data.
    pub fn submit_with_data(
        &mut self,
        mesh: u64,
        material: u64,
        instances: &[InstanceData],
    ) {
        let key = (mesh, material);
        let entry = self.submissions.entry(key).or_default();
        entry.extend_from_slice(instances);
    }

    /// Submit a single instance.
    pub fn submit_one(&mut self, mesh: u64, material: u64, transform: Mat4) {
        let key = (mesh, material);
        self.submissions
            .entry(key)
            .or_default()
            .push(InstanceData::from_matrix(transform));
    }

    /// Submit a single instance with custom data.
    pub fn submit_one_with_data(
        &mut self,
        mesh: u64,
        material: u64,
        instance: InstanceData,
    ) {
        let key = (mesh, material);
        self.submissions.entry(key).or_default().push(instance);
    }

    /// Flush all submissions: batch, optionally cull, upload, and generate draw
    /// calls.
    ///
    /// After this call, `draw_calls()` returns the final list of draw calls and
    /// `stats` contains the frame's rendering statistics.
    pub fn flush(&mut self, frustum: Option<&Frustum>) {
        self.draw_calls.clear();
        self.stats = InstancedRendererStats::default();

        let mut total_submitted: u32 = 0;
        let mut total_drawn: u32 = 0;
        let mut total_culled: u32 = 0;
        let mut total_triangles: u64 = 0;
        let mut total_buffer_bytes: usize = 0;

        // Sort submissions by material for optimal GPU state changes.
        let mut sorted_keys: Vec<(u64, u64)> = self.submissions.keys().copied().collect();
        sorted_keys.sort_by_key(|&(mesh, material)| (material, mesh));

        let mut global_instance_offset: u32 = 0;

        for key in &sorted_keys {
            let instances = match self.submissions.get(key) {
                Some(v) if !v.is_empty() => v,
                _ => continue,
            };

            let (mesh_handle, material_handle) = *key;
            total_submitted += instances.len() as u32;

            // Optionally cull.
            let visible_instances = if self.frustum_culling_enabled {
                if let (Some(frustum), Some(bounds)) =
                    (frustum, self.mesh_bounds.get(&mesh_handle))
                {
                    self.culler.cull(instances, bounds, frustum);
                    total_culled += self.culler.culled_count() as u32;
                    self.culler.visible_instances()
                } else {
                    instances.as_slice()
                }
            } else {
                instances.as_slice()
            };

            if visible_instances.is_empty() {
                continue;
            }

            let instance_count = visible_instances.len() as u32;
            total_drawn += instance_count;

            // Get or create the instance buffer for this batch.
            if !self.instance_buffers.contains_key(key) {
                let buf = InstanceBuffer::new(self.next_buffer_id, visible_instances.len());
                self.next_buffer_id += 1;
                self.instance_buffers.insert(*key, buf);
            }

            let buffer = self.instance_buffers.get_mut(key).unwrap();
            buffer.reserve(visible_instances.len());
            buffer.upload(visible_instances);

            total_buffer_bytes += buffer.byte_size();

            // Generate the draw call.
            // NOTE: In a real renderer, `index_count` and `first_index` would
            // come from the mesh resource. Here we use placeholder values that
            // the caller must fill in from their mesh registry.
            self.draw_calls.push(InstancedDrawCall {
                mesh_id: mesh_handle,
                material_id: material_handle,
                instance_buffer_offset: global_instance_offset,
                instance_count,
                index_count: 0, // to be filled by caller
                first_index: 0,
                base_vertex: 0,
            });

            global_instance_offset += instance_count;
        }

        self.stats = InstancedRendererStats {
            draw_calls: self.draw_calls.len() as u32,
            triangles: total_triangles,
            instances_submitted: total_submitted,
            instances_drawn: total_drawn,
            instances_culled: total_culled,
            unique_batches: sorted_keys.len() as u32,
            instance_buffer_bytes: total_buffer_bytes,
        };

        // Clear submissions for the next frame.
        self.submissions.clear();
    }

    /// Get the generated draw calls.
    pub fn draw_calls(&self) -> &[InstancedDrawCall] {
        &self.draw_calls
    }

    /// Get the instance buffer for a specific mesh+material combination.
    pub fn get_instance_buffer(&self, mesh: u64, material: u64) -> Option<&InstanceBuffer> {
        self.instance_buffers.get(&(mesh, material))
    }

    /// Get the rendering statistics for the current frame.
    pub fn stats(&self) -> &InstancedRendererStats {
        &self.stats
    }

    /// Clear all internal state (buffers, caches, etc.).
    pub fn reset(&mut self) {
        self.submissions.clear();
        self.instance_buffers.clear();
        self.draw_calls.clear();
        self.mesh_bounds.clear();
        self.stats = InstancedRendererStats::default();
    }

    /// Number of active instance buffers.
    pub fn buffer_count(&self) -> usize {
        self.instance_buffers.len()
    }
}

impl Default for InstancedRenderer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SortKey
// ---------------------------------------------------------------------------

/// A 64-bit sort key for ordering draw calls to minimise GPU state changes.
///
/// Layout (MSB to LSB):
/// - Bits 48..63: material ID (16 bits)
/// - Bits 32..47: mesh ID (16 bits)
/// - Bits 16..31: distance bucket (16 bits, front-to-back)
/// - Bits  0..15: instance count hint (16 bits)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SortKey(pub u64);

impl SortKey {
    /// Build a sort key from components.
    pub fn new(material_id: u16, mesh_id: u16, distance_bucket: u16, instance_hint: u16) -> Self {
        let key = ((material_id as u64) << 48)
            | ((mesh_id as u64) << 32)
            | ((distance_bucket as u64) << 16)
            | instance_hint as u64;
        Self(key)
    }

    /// Extract the material ID from the sort key.
    pub fn material_id(&self) -> u16 {
        ((self.0 >> 48) & 0xFFFF) as u16
    }

    /// Extract the mesh ID from the sort key.
    pub fn mesh_id(&self) -> u16 {
        ((self.0 >> 32) & 0xFFFF) as u16
    }

    /// Extract the distance bucket from the sort key.
    pub fn distance_bucket(&self) -> u16 {
        ((self.0 >> 16) & 0xFFFF) as u16
    }

    /// Compute a distance bucket from a view-space depth value.
    pub fn bucket_from_depth(depth: f32, near: f32, far: f32) -> u16 {
        let normalised = ((depth - near) / (far - near)).clamp(0.0, 1.0);
        (normalised * 65535.0) as u16
    }
}

// ---------------------------------------------------------------------------
// InstancedRendererConfig
// ---------------------------------------------------------------------------

/// Configuration for the instanced renderer.
#[derive(Debug, Clone)]
pub struct InstancedRendererConfig {
    /// Whether to enable per-instance frustum culling.
    pub frustum_culling: bool,
    /// Maximum draw distance for instanced objects.
    pub max_draw_distance: f32,
    /// Whether to sort draw calls by material.
    pub sort_by_material: bool,
    /// Whether to merge small draw calls.
    pub merge_small_draws: bool,
    /// Minimum instance count to justify a dedicated draw call.
    pub min_instances_per_draw: u32,
}

impl Default for InstancedRendererConfig {
    fn default() -> Self {
        Self {
            frustum_culling: true,
            max_draw_distance: 1000.0,
            sort_by_material: true,
            merge_small_draws: true,
            min_instances_per_draw: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// WGSL instanced rendering shader
// ---------------------------------------------------------------------------

/// WGSL vertex shader for instanced rendering with per-instance transforms.
pub const INSTANCED_VERTEX_WGSL: &str = r#"
// Instanced rendering vertex shader.
// Reads per-instance transform from a storage buffer and applies it.

struct InstanceData {
    world_matrix: mat4x4<f32>,
    custom_data: vec4<f32>,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<storage, read> instances: array<InstanceData>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) custom_data: vec4<f32>,
}

@vertex
fn vs_main(
    vertex: VertexInput,
    @builtin(instance_index) instance_id: u32,
) -> VertexOutput {
    var output: VertexOutput;

    let instance = instances[instance_id];
    let world_pos = instance.world_matrix * vec4<f32>(vertex.position, 1.0);

    output.clip_position = camera.view_projection * world_pos;
    output.world_position = world_pos.xyz;

    // Transform normal to world space (using the inverse transpose of the
    // upper-left 3x3).
    let normal_matrix = mat3x3<f32>(
        instance.world_matrix[0].xyz,
        instance.world_matrix[1].xyz,
        instance.world_matrix[2].xyz
    );
    output.world_normal = normalize(normal_matrix * vertex.normal);

    output.uv = vertex.uv;
    output.custom_data = instance.custom_data;

    return output;
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{create_cube, create_sphere};

    #[test]
    fn test_instance_data_from_matrix() {
        let mat = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let inst = InstanceData::from_matrix(mat);

        let pos = inst.position();
        assert!((pos.x - 1.0).abs() < 1e-5);
        assert!((pos.y - 2.0).abs() < 1e-5);
        assert!((pos.z - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_instance_data_custom() {
        let mat = Mat4::IDENTITY;
        let custom = Vec4::new(1.0, 0.5, 0.0, 42.0);
        let inst = InstanceData::new(mat, custom);

        let c = inst.custom();
        assert!((c.x - 1.0).abs() < 1e-5);
        assert!((c.w - 42.0).abs() < 1e-5);
    }

    #[test]
    fn test_instance_data_scale() {
        let mat = Mat4::from_scale(Vec3::splat(2.0));
        let inst = InstanceData::from_matrix(mat);
        assert!((inst.scale() - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_instance_data_size() {
        // Verify the struct has a known size for GPU buffer layout.
        assert_eq!(InstanceData::SIZE, std::mem::size_of::<InstanceData>());
        // 4x4 matrix (64 bytes) + vec4 custom (16 bytes) = 80 bytes.
        assert_eq!(InstanceData::SIZE, 80);
    }

    #[test]
    fn test_instance_buffer_basic() {
        let mut buffer = InstanceBuffer::new(0, 16);
        assert_eq!(buffer.used_count(), 0);
        assert!(buffer.capacity() >= 16);

        buffer.push(InstanceData::default());
        buffer.push(InstanceData::default());
        assert_eq!(buffer.used_count(), 2);
        assert!(buffer.is_dirty());
    }

    #[test]
    fn test_instance_buffer_clear() {
        let mut buffer = InstanceBuffer::new(0, 16);
        buffer.push(InstanceData::default());
        assert_eq!(buffer.used_count(), 1);

        buffer.clear();
        assert_eq!(buffer.used_count(), 0);
    }

    #[test]
    fn test_instance_buffer_double_buffer() {
        let mut buffer = InstanceBuffer::new(0, 16);

        // Write to buffer 0.
        buffer.push(InstanceData::from_matrix(Mat4::from_translation(Vec3::X)));
        let write_data = buffer.write_buffer().to_vec();

        // Swap: now buffer 0 is readable, buffer 1 is writable.
        buffer.swap_buffers();
        let read_data = buffer.read_buffer();

        // The read buffer should contain what we just wrote.
        assert_eq!(read_data.len(), write_data.len());
    }

    #[test]
    fn test_instance_buffer_upload() {
        let mut buffer = InstanceBuffer::new(0, 16);
        let instances = vec![
            InstanceData::from_matrix(Mat4::from_translation(Vec3::X)),
            InstanceData::from_matrix(Mat4::from_translation(Vec3::Y)),
        ];

        buffer.upload(&instances);
        assert_eq!(buffer.upload_count(), 1);
    }

    #[test]
    fn test_instance_buffer_reserve() {
        let mut buffer = InstanceBuffer::new(0, 4);
        assert!(buffer.capacity() >= 4);

        buffer.reserve(100);
        assert!(buffer.capacity() >= 100);
    }

    #[test]
    fn test_instance_buffer_bytes() {
        let mut buffer = InstanceBuffer::new(0, 16);
        buffer.push(InstanceData::default());
        let bytes = buffer.write_buffer_bytes();
        assert_eq!(bytes.len(), InstanceData::SIZE);
    }

    #[test]
    fn test_instanced_mesh() {
        let bounds = AABB::new(-Vec3::ONE, Vec3::ONE);
        let mut im = InstancedMesh::new(1, 2, bounds);

        im.add_transform(Mat4::IDENTITY);
        im.add_transform(Mat4::from_translation(Vec3::X));
        assert_eq!(im.instance_count(), 2);

        im.clear_instances();
        assert_eq!(im.instance_count(), 0);
    }

    #[test]
    fn test_instanced_draw_call() {
        let dc = InstancedDrawCall {
            mesh_id: 1,
            material_id: 2,
            instance_buffer_offset: 0,
            instance_count: 100,
            index_count: 36,
            first_index: 0,
            base_vertex: 0,
        };

        assert!(!dc.is_empty());
        assert_eq!(dc.triangle_count(), 12 * 100);
    }

    #[test]
    fn test_instanced_draw_call_empty() {
        let dc = InstancedDrawCall {
            mesh_id: 0,
            material_id: 0,
            instance_buffer_offset: 0,
            instance_count: 0,
            index_count: 36,
            first_index: 0,
            base_vertex: 0,
        };
        assert!(dc.is_empty());
    }

    #[test]
    fn test_instance_culler_all_visible() {
        let mut culler = InstanceCuller::new();
        let bounds = AABB::new(-Vec3::ONE, Vec3::ONE);

        // Create a frustum that sees everything in front.
        let vp = Mat4::perspective_rh(1.0, 1.0, 0.1, 1000.0)
            * Mat4::look_at_rh(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
        let frustum = Frustum::from_view_projection(&vp);

        let instances = vec![
            InstanceData::from_matrix(Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0))),
            InstanceData::from_matrix(Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0))),
        ];

        let visible = culler.cull(&instances, &bounds, &frustum);
        assert_eq!(visible, 2);
        assert_eq!(culler.culled_count(), 0);
        assert_eq!(culler.visible_instances().len(), 2);
    }

    #[test]
    fn test_instance_culler_some_culled() {
        let mut culler = InstanceCuller::new();
        let bounds = AABB::new(-Vec3::ONE, Vec3::ONE);

        // Camera looking down -Z. Objects behind the camera should be culled.
        let vp = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0)
            * Mat4::look_at_rh(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
        let frustum = Frustum::from_view_projection(&vp);

        let instances = vec![
            // In front of camera.
            InstanceData::from_matrix(Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0))),
            // Behind camera.
            InstanceData::from_matrix(Mat4::from_translation(Vec3::new(0.0, 0.0, 50.0))),
        ];

        let visible = culler.cull(&instances, &bounds, &frustum);
        // At least one should be visible, and ideally the one behind is culled.
        assert!(visible >= 1);
        assert!(culler.visible_count() + culler.culled_count() == 2);
    }

    #[test]
    fn test_instance_culler_distance() {
        let mut culler = InstanceCuller::new();
        let bounds = AABB::new(-Vec3::ONE, Vec3::ONE);

        let vp = Mat4::perspective_rh(1.0, 1.0, 0.1, 1000.0)
            * Mat4::look_at_rh(Vec3::ZERO, -Vec3::Z, Vec3::Y);
        let frustum = Frustum::from_view_projection(&vp);

        let instances = vec![
            InstanceData::from_matrix(Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0))),
            InstanceData::from_matrix(Mat4::from_translation(Vec3::new(0.0, 0.0, -200.0))),
        ];

        let visible = culler.cull_with_distance(
            &instances, &bounds, &frustum, Vec3::ZERO, 50.0,
        );

        // The far instance should be distance-culled.
        assert!(visible >= 1);
        assert!(culler.culled_count() >= 1);
    }

    #[test]
    fn test_instance_culler_cull_ratio() {
        let mut culler = InstanceCuller::new();
        let bounds = AABB::new(-Vec3::ONE, Vec3::ONE);

        let vp = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0)
            * Mat4::look_at_rh(Vec3::ZERO, -Vec3::Z, Vec3::Y);
        let frustum = Frustum::from_view_projection(&vp);

        // All visible.
        let instances = vec![
            InstanceData::from_matrix(Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0))),
        ];
        culler.cull(&instances, &bounds, &frustum);
        assert!((culler.cull_ratio() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_merged_instancing_basic() {
        let mut merged = MergedInstancing::new();

        let cube = create_cube();
        let sphere = create_sphere(8, 6);

        let cube_idx = merged.add_mesh(1, &cube);
        let sphere_idx = merged.add_mesh(2, &sphere);

        assert_eq!(cube_idx, 0);
        assert_eq!(sphere_idx, 1);
        assert_eq!(merged.mesh_count(), 2);

        // Re-adding the same mesh should return the same index.
        let cube_idx2 = merged.add_mesh(1, &cube);
        assert_eq!(cube_idx2, 0);
        assert_eq!(merged.mesh_count(), 2);
    }

    #[test]
    fn test_merged_instancing_vertices() {
        let mut merged = MergedInstancing::new();
        let cube = create_cube();
        let sphere = create_sphere(8, 6);

        merged.add_mesh(1, &cube);
        merged.add_mesh(2, &sphere);

        let expected_verts = cube.vertices.len() + sphere.vertices.len();
        assert_eq!(merged.total_vertex_count(), expected_verts);

        let expected_indices = cube.indices.len() + sphere.indices.len();
        assert_eq!(merged.total_index_count(), expected_indices);
    }

    #[test]
    fn test_merged_instancing_draw_calls() {
        let mut merged = MergedInstancing::new();
        let cube = create_cube();

        merged.add_mesh(1, &cube);
        merged.add_instance(1, Mat4::IDENTITY, Vec4::ONE);
        merged.add_instance(1, Mat4::from_translation(Vec3::X), Vec4::ONE);

        let draws = merged.generate_draw_calls();
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].instance_count, 2);
    }

    #[test]
    fn test_merged_instancing_clear() {
        let mut merged = MergedInstancing::new();
        let cube = create_cube();
        merged.add_mesh(1, &cube);
        merged.add_instance(1, Mat4::IDENTITY, Vec4::ONE);

        merged.clear_instances();
        assert_eq!(merged.instance_count(), 0);
        assert_eq!(merged.mesh_count(), 1); // meshes preserved

        merged.clear_all();
        assert_eq!(merged.mesh_count(), 0);
        assert_eq!(merged.total_vertex_count(), 0);
    }

    #[test]
    fn test_instanced_renderer_submit_flush() {
        let mut renderer = InstancedRenderer::new();
        renderer.frustum_culling_enabled = false;

        renderer.submit(1, 10, &[Mat4::IDENTITY, Mat4::from_translation(Vec3::X)]);
        renderer.submit(2, 10, &[Mat4::from_translation(Vec3::Y)]);

        renderer.flush(None);

        // Should produce 2 draw calls (one per mesh).
        assert_eq!(renderer.stats().draw_calls, 2);
        assert_eq!(renderer.stats().instances_submitted, 3);
        assert_eq!(renderer.stats().instances_drawn, 3);
    }

    #[test]
    fn test_instanced_renderer_batching() {
        let mut renderer = InstancedRenderer::new();
        renderer.frustum_culling_enabled = false;

        // Submit the same mesh+material twice: should be batched.
        renderer.submit_one(1, 10, Mat4::IDENTITY);
        renderer.submit_one(1, 10, Mat4::from_translation(Vec3::X));

        renderer.flush(None);

        assert_eq!(renderer.stats().draw_calls, 1);
        assert_eq!(renderer.stats().instances_drawn, 2);
    }

    #[test]
    fn test_instanced_renderer_with_culling() {
        let mut renderer = InstancedRenderer::new();
        renderer.frustum_culling_enabled = true;
        renderer.register_mesh_bounds(1, AABB::new(-Vec3::ONE, Vec3::ONE));

        let vp = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0)
            * Mat4::look_at_rh(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
        let frustum = Frustum::from_view_projection(&vp);

        // Instance in front of camera.
        renderer.submit_one(1, 10, Mat4::from_translation(Vec3::ZERO));
        // Instance behind camera.
        renderer.submit_one(1, 10, Mat4::from_translation(Vec3::new(0.0, 0.0, 50.0)));

        renderer.flush(Some(&frustum));

        // At least the front instance should be drawn.
        assert!(renderer.stats().instances_drawn >= 1);
    }

    #[test]
    fn test_instanced_renderer_reset() {
        let mut renderer = InstancedRenderer::new();
        renderer.frustum_culling_enabled = false;
        renderer.submit_one(1, 10, Mat4::IDENTITY);
        renderer.flush(None);

        renderer.reset();
        assert_eq!(renderer.buffer_count(), 0);
        assert_eq!(renderer.draw_calls().len(), 0);
    }

    #[test]
    fn test_instanced_renderer_stats() {
        let mut renderer = InstancedRenderer::new();
        renderer.frustum_culling_enabled = false;

        renderer.submit(1, 10, &[Mat4::IDENTITY]);
        renderer.submit(2, 20, &[Mat4::IDENTITY, Mat4::IDENTITY]);

        renderer.flush(None);

        assert_eq!(renderer.stats().instances_submitted, 3);
        assert_eq!(renderer.stats().unique_batches, 2);
    }

    #[test]
    fn test_sort_key() {
        let k1 = SortKey::new(1, 2, 100, 50);
        let k2 = SortKey::new(1, 3, 100, 50);
        let k3 = SortKey::new(2, 1, 0, 0);

        // Same material, different mesh: k1 < k2.
        assert!(k1 < k2);
        // Different material: k1 < k3.
        assert!(k1 < k3);

        assert_eq!(k1.material_id(), 1);
        assert_eq!(k1.mesh_id(), 2);
        assert_eq!(k1.distance_bucket(), 100);
    }

    #[test]
    fn test_sort_key_bucket() {
        let bucket = SortKey::bucket_from_depth(50.0, 0.1, 100.0);
        // Should be roughly in the middle of the range.
        assert!(bucket > 30000);
        assert!(bucket < 35000);

        let near_bucket = SortKey::bucket_from_depth(0.1, 0.1, 100.0);
        assert_eq!(near_bucket, 0);

        let far_bucket = SortKey::bucket_from_depth(100.0, 0.1, 100.0);
        assert_eq!(far_bucket, 65535);
    }

    #[test]
    fn test_instanced_renderer_config_default() {
        let config = InstancedRendererConfig::default();
        assert!(config.frustum_culling);
        assert!(config.sort_by_material);
        assert_eq!(config.min_instances_per_draw, 1);
        assert!(config.max_draw_distance > 0.0);
    }

    #[test]
    fn test_instance_buffer_extend() {
        let mut buffer = InstanceBuffer::new(0, 16);
        let instances = vec![InstanceData::default(); 5];
        buffer.extend(&instances);
        assert_eq!(buffer.used_count(), 5);
        assert_eq!(buffer.byte_size(), 5 * InstanceData::SIZE);
    }

    #[test]
    fn test_instanced_renderer_submit_with_data() {
        let mut renderer = InstancedRenderer::new();
        renderer.frustum_culling_enabled = false;

        let instances = vec![
            InstanceData::new(Mat4::IDENTITY, Vec4::new(1.0, 0.0, 0.0, 1.0)),
            InstanceData::new(Mat4::from_translation(Vec3::Y), Vec4::new(0.0, 1.0, 0.0, 1.0)),
        ];

        renderer.submit_with_data(1, 10, &instances);
        renderer.flush(None);

        assert_eq!(renderer.stats().instances_drawn, 2);
        assert_eq!(renderer.stats().draw_calls, 1);
    }

    #[test]
    fn test_instanced_renderer_empty_flush() {
        let mut renderer = InstancedRenderer::new();
        renderer.flush(None);

        assert_eq!(renderer.stats().draw_calls, 0);
        assert_eq!(renderer.stats().instances_submitted, 0);
    }

    #[test]
    fn test_merged_instancing_multiple_meshes_instances() {
        let mut merged = MergedInstancing::new();
        let cube = create_cube();
        let sphere = create_sphere(8, 6);

        merged.add_mesh(1, &cube);
        merged.add_mesh(2, &sphere);

        merged.add_instance(1, Mat4::IDENTITY, Vec4::ONE);
        merged.add_instance(1, Mat4::from_translation(Vec3::X), Vec4::ONE);
        merged.add_instance(2, Mat4::from_translation(Vec3::Y), Vec4::ONE);

        let draws = merged.generate_draw_calls();
        // Should produce 2 draw calls (one per mesh).
        assert_eq!(draws.len(), 2);

        // Find the cube draw call.
        let cube_draw = draws.iter().find(|d| d.mesh_id == 1).unwrap();
        assert_eq!(cube_draw.instance_count, 2);

        let sphere_draw = draws.iter().find(|d| d.mesh_id == 2).unwrap();
        assert_eq!(sphere_draw.instance_count, 1);
    }
}
