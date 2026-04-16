// engine/render/src/gpu_driven/mod.rs
//
// GPU-driven rendering pipeline for the Genovo engine. Implements indirect
// draw command generation, GPU-side instance culling (frustum + occlusion),
// Hi-Z depth pyramid construction, and multi-draw indirect batching.
//
// This module enables rendering thousands of objects with minimal CPU overhead
// by moving culling and draw call generation to GPU compute shaders.

use crate::mesh::AABB;
use crate::virtual_geometry::bvh::Frustum;
use glam::{Mat4, Vec2, Vec3, Vec4};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of instances that can be managed.
pub const MAX_INSTANCES: usize = 1_000_000;

/// Maximum number of indirect draw commands per frame.
pub const MAX_DRAW_COMMANDS: usize = 65_536;

/// Maximum number of mesh descriptors.
pub const MAX_MESH_DESCRIPTORS: usize = 8192;

/// Size of Hi-Z mip chain (max resolution = 2^MAX_HIZ_MIPS).
pub const MAX_HIZ_MIPS: usize = 12;

/// Tile size for Hi-Z construction compute shader.
pub const HIZ_TILE_SIZE: u32 = 8;

/// Workgroup size for culling compute shader.
pub const CULL_WORKGROUP_SIZE: u32 = 256;

// ---------------------------------------------------------------------------
// IndirectDrawCommand
// ---------------------------------------------------------------------------

/// Matches the GPU `DrawIndexedIndirect` layout.
///
/// This struct is uploaded directly to a GPU buffer and consumed by
/// `draw_indexed_indirect` or `multi_draw_indexed_indirect` calls.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IndirectDrawCommand {
    /// Number of indices to draw.
    pub index_count: u32,
    /// Number of instances to draw.
    pub instance_count: u32,
    /// Byte offset into the index buffer.
    pub first_index: u32,
    /// Value added to each index before indexing into the vertex buffer.
    pub base_vertex: i32,
    /// First instance ID.
    pub first_instance: u32,
}

impl IndirectDrawCommand {
    /// Create a new draw command.
    pub fn new(
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) -> Self {
        Self {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        }
    }

    /// Size of this struct in bytes (for GPU buffer layout).
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// Whether this command would draw anything.
    pub fn is_empty(&self) -> bool {
        self.index_count == 0 || self.instance_count == 0
    }
}

// ---------------------------------------------------------------------------
// InstanceData
// ---------------------------------------------------------------------------

/// Per-instance data stored in a GPU buffer. Each instance represents one
/// object in the scene.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct InstanceData {
    /// World transform matrix (4x4 column-major).
    pub world_matrix: [[f32; 4]; 4],
    /// Previous frame's world matrix (for motion vectors).
    pub prev_world_matrix: [[f32; 4]; 4],
    /// Bounding sphere in object space: (center.xyz, radius).
    pub bounding_sphere: [f32; 4],
    /// AABB min in object space.
    pub aabb_min: [f32; 3],
    /// Material ID.
    pub material_id: u32,
    /// AABB max in object space.
    pub aabb_max: [f32; 3],
    /// Mesh descriptor ID.
    pub mesh_id: u32,
    /// LOD level (0 = highest detail).
    pub lod_level: u32,
    /// Visibility flag (written by the culling shader).
    pub visible: u32,
    /// Instance flags (shadow caster, receives shadows, etc.).
    pub flags: u32,
    /// Padding to align to 16 bytes.
    pub _pad: u32,
}

impl Default for InstanceData {
    fn default() -> Self {
        Self {
            world_matrix: Mat4::IDENTITY.to_cols_array_2d(),
            prev_world_matrix: Mat4::IDENTITY.to_cols_array_2d(),
            bounding_sphere: [0.0, 0.0, 0.0, 1.0],
            aabb_min: [0.0; 3],
            material_id: 0,
            aabb_max: [0.0; 3],
            mesh_id: 0,
            lod_level: 0,
            visible: 1,
            flags: 0xFFFF_FFFF,
            _pad: 0,
        }
    }
}

impl InstanceData {
    /// Size of this struct in bytes.
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// Create instance data from a world matrix, AABB, and IDs.
    pub fn new(
        world_matrix: Mat4,
        aabb: &AABB,
        material_id: u32,
        mesh_id: u32,
    ) -> Self {
        let center = aabb.center();
        let radius = aabb.radius();

        Self {
            world_matrix: world_matrix.to_cols_array_2d(),
            prev_world_matrix: world_matrix.to_cols_array_2d(),
            bounding_sphere: [center.x, center.y, center.z, radius],
            aabb_min: aabb.min.to_array(),
            material_id,
            aabb_max: aabb.max.to_array(),
            mesh_id,
            lod_level: 0,
            visible: 1,
            flags: 0xFFFF_FFFF,
            _pad: 0,
        }
    }

    /// Instance flags.
    pub const FLAG_CAST_SHADOW: u32 = 1 << 0;
    pub const FLAG_RECEIVE_SHADOW: u32 = 1 << 1;
    pub const FLAG_STATIC: u32 = 1 << 2;
    pub const FLAG_OCCLUDER: u32 = 1 << 3;
    pub const FLAG_TRANSPARENT: u32 = 1 << 4;

    /// Check if a flag is set.
    pub fn has_flag(&self, flag: u32) -> bool {
        self.flags & flag != 0
    }
}

// ---------------------------------------------------------------------------
// MeshDescriptor
// ---------------------------------------------------------------------------

/// Describes the location of a mesh's data in the global vertex/index buffers.
/// One descriptor per unique mesh. Multiple instances can reference the same
/// descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MeshDescriptor {
    /// Byte offset of the first vertex in the global vertex buffer.
    pub vertex_offset: u32,
    /// Number of vertices.
    pub vertex_count: u32,
    /// Offset of the first index in the global index buffer.
    pub index_offset: u32,
    /// Number of indices for this LOD.
    pub index_count: u32,
    /// Bounding sphere radius in object space.
    pub bounding_radius: f32,
    /// LOD level this descriptor represents.
    pub lod_level: u32,
    /// Screen-space error threshold for this LOD.
    pub lod_error: f32,
    /// Padding.
    pub _pad: u32,
}

impl MeshDescriptor {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    pub fn new(
        vertex_offset: u32,
        vertex_count: u32,
        index_offset: u32,
        index_count: u32,
        bounding_radius: f32,
    ) -> Self {
        Self {
            vertex_offset,
            vertex_count,
            index_offset,
            index_count,
            bounding_radius,
            lod_level: 0,
            lod_error: 0.0,
            _pad: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// GPUScene
// ---------------------------------------------------------------------------

/// Manages all scene data on the GPU side. Stores instances, mesh descriptors,
/// and handles uploads/updates.
pub struct GPUScene {
    /// All instances in the scene.
    instances: Vec<InstanceData>,
    /// All mesh descriptors.
    mesh_descriptors: Vec<MeshDescriptor>,
    /// Map from mesh asset ID to mesh descriptor index.
    mesh_id_map: HashMap<u64, u32>,
    /// Number of active (non-removed) instances.
    active_instance_count: usize,
    /// Free list of instance slots.
    free_instance_slots: Vec<usize>,
    /// Whether the instance buffer needs re-upload.
    instances_dirty: bool,
    /// Whether the mesh descriptor buffer needs re-upload.
    descriptors_dirty: bool,
    /// Frame counter for tracking updates.
    frame_number: u64,
}

impl GPUScene {
    /// Create a new GPU scene.
    pub fn new() -> Self {
        Self {
            instances: Vec::with_capacity(1024),
            mesh_descriptors: Vec::with_capacity(256),
            mesh_id_map: HashMap::new(),
            active_instance_count: 0,
            free_instance_slots: Vec::new(),
            instances_dirty: true,
            descriptors_dirty: true,
            frame_number: 0,
        }
    }

    /// Add an instance to the scene. Returns the instance index.
    pub fn add_instance(&mut self, instance: InstanceData) -> u32 {
        self.instances_dirty = true;

        if let Some(slot) = self.free_instance_slots.pop() {
            self.instances[slot] = instance;
            self.active_instance_count += 1;
            slot as u32
        } else {
            let idx = self.instances.len();
            self.instances.push(instance);
            self.active_instance_count += 1;
            idx as u32
        }
    }

    /// Remove an instance from the scene.
    pub fn remove_instance(&mut self, index: u32) {
        let idx = index as usize;
        if idx < self.instances.len() {
            self.instances[idx].visible = 0;
            self.instances[idx].flags = 0;
            self.free_instance_slots.push(idx);
            self.active_instance_count -= 1;
            self.instances_dirty = true;
        }
    }

    /// Update an instance's world matrix.
    pub fn update_instance_transform(&mut self, index: u32, new_matrix: Mat4) {
        let idx = index as usize;
        if idx < self.instances.len() {
            self.instances[idx].prev_world_matrix = self.instances[idx].world_matrix;
            self.instances[idx].world_matrix = new_matrix.to_cols_array_2d();
            self.instances_dirty = true;
        }
    }

    /// Update an instance's LOD level.
    pub fn update_instance_lod(&mut self, index: u32, lod_level: u32) {
        let idx = index as usize;
        if idx < self.instances.len() {
            self.instances[idx].lod_level = lod_level;
            self.instances_dirty = true;
        }
    }

    /// Register a mesh descriptor. Returns the mesh descriptor ID.
    pub fn register_mesh(&mut self, asset_id: u64, descriptor: MeshDescriptor) -> u32 {
        if let Some(&existing_id) = self.mesh_id_map.get(&asset_id) {
            self.mesh_descriptors[existing_id as usize] = descriptor;
            self.descriptors_dirty = true;
            return existing_id;
        }

        let id = self.mesh_descriptors.len() as u32;
        self.mesh_descriptors.push(descriptor);
        self.mesh_id_map.insert(asset_id, id);
        self.descriptors_dirty = true;
        id
    }

    /// Get the raw instance data for GPU upload.
    pub fn instance_data(&self) -> &[InstanceData] {
        &self.instances
    }

    /// Get the raw mesh descriptor data for GPU upload.
    pub fn mesh_descriptor_data(&self) -> &[MeshDescriptor] {
        &self.mesh_descriptors
    }

    /// Number of instances.
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Number of active (visible) instances.
    pub fn active_instance_count(&self) -> usize {
        self.active_instance_count
    }

    /// Number of registered meshes.
    pub fn mesh_count(&self) -> usize {
        self.mesh_descriptors.len()
    }

    /// Check and clear the dirty flag for instances.
    pub fn take_instances_dirty(&mut self) -> bool {
        let dirty = self.instances_dirty;
        self.instances_dirty = false;
        dirty
    }

    /// Check and clear the dirty flag for mesh descriptors.
    pub fn take_descriptors_dirty(&mut self) -> bool {
        let dirty = self.descriptors_dirty;
        self.descriptors_dirty = false;
        dirty
    }

    /// Advance to next frame.
    pub fn begin_frame(&mut self) {
        self.frame_number += 1;
    }

    /// Get the current frame number.
    pub fn frame_number(&self) -> u64 {
        self.frame_number
    }
}

impl Default for GPUScene {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DepthPyramid
// ---------------------------------------------------------------------------

/// Hi-Z depth pyramid for occlusion culling. Built from the depth buffer
/// via a compute shader that produces a mip chain where each texel contains
/// the maximum (farthest) depth in its footprint.
#[derive(Debug, Clone)]
pub struct DepthPyramid {
    /// Width of the base level.
    pub width: u32,
    /// Height of the base level.
    pub height: u32,
    /// Number of mip levels.
    pub mip_count: u32,
    /// Mip level dimensions.
    pub mip_dimensions: Vec<(u32, u32)>,
    /// CPU-side depth data per mip level (for software fallback).
    pub mip_data: Vec<Vec<f32>>,
}

impl DepthPyramid {
    /// Create a new depth pyramid descriptor for the given resolution.
    pub fn new(width: u32, height: u32) -> Self {
        let mip_count = ((width.max(height) as f32).log2().floor() as u32 + 1).min(MAX_HIZ_MIPS as u32);

        let mut mip_dimensions = Vec::with_capacity(mip_count as usize);
        let mut mip_data = Vec::with_capacity(mip_count as usize);

        let mut w = width;
        let mut h = height;

        for _ in 0..mip_count {
            mip_dimensions.push((w, h));
            mip_data.push(vec![0.0f32; (w * h) as usize]);
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }

        Self {
            width,
            height,
            mip_count,
            mip_dimensions,
            mip_data,
        }
    }

    /// Build the mip chain from a depth buffer (CPU fallback).
    ///
    /// Each mip level stores the maximum (farthest) depth value from
    /// the 2x2 block of the previous level.
    pub fn build_from_depth_buffer(&mut self, depth_data: &[f32]) {
        if depth_data.len() < (self.width * self.height) as usize {
            return;
        }

        // Copy base level.
        self.mip_data[0] = depth_data[..(self.width * self.height) as usize].to_vec();

        // Build mip chain.
        for level in 1..self.mip_count as usize {
            let (prev_w, prev_h) = self.mip_dimensions[level - 1];
            let (curr_w, curr_h) = self.mip_dimensions[level];

            let prev_data = self.mip_data[level - 1].clone();
            let curr_data = &mut self.mip_data[level];

            for y in 0..curr_h {
                for x in 0..curr_w {
                    let px = (x * 2).min(prev_w - 1);
                    let py = (y * 2).min(prev_h - 1);

                    let px1 = (px + 1).min(prev_w - 1);
                    let py1 = (py + 1).min(prev_h - 1);

                    let d00 = prev_data[(py * prev_w + px) as usize];
                    let d10 = prev_data[(py * prev_w + px1) as usize];
                    let d01 = prev_data[(py1 * prev_w + px) as usize];
                    let d11 = prev_data[(py1 * prev_w + px1) as usize];

                    // Use max for conservative depth (reversed-Z: use min for farthest).
                    // Standard depth: max = farthest.
                    curr_data[(y * curr_w + x) as usize] = d00.max(d10).max(d01).max(d11);
                }
            }
        }
    }

    /// Test if a screen-space rectangle is occluded at a given depth.
    ///
    /// Uses the appropriate mip level based on the rectangle size.
    pub fn test_rect_occluded(&self, min_x: f32, min_y: f32, max_x: f32, max_y: f32, test_depth: f32) -> bool {
        if self.mip_data.is_empty() || self.mip_data[0].is_empty() {
            return false; // No data -- assume visible.
        }

        // Determine mip level based on rect size.
        let rect_w = ((max_x - min_x) * self.width as f32).max(1.0);
        let rect_h = ((max_y - min_y) * self.height as f32).max(1.0);
        let max_dim = rect_w.max(rect_h);

        let mip_level = (max_dim.log2().ceil() as usize).min(self.mip_count as usize - 1);

        let (mip_w, mip_h) = self.mip_dimensions[mip_level];
        let mip = &self.mip_data[mip_level];

        // Convert normalised coords to mip-level texel coords.
        let tx = ((min_x + max_x) * 0.5 * mip_w as f32) as u32;
        let ty = ((min_y + max_y) * 0.5 * mip_h as f32) as u32;

        let tx = tx.min(mip_w - 1);
        let ty = ty.min(mip_h - 1);

        let pyramid_depth = mip[(ty * mip_w + tx) as usize];

        // Occluded if the test depth is farther than the pyramid depth.
        // (Standard depth: higher value = farther.)
        test_depth > pyramid_depth
    }

    /// Generate the WGSL compute shader source for Hi-Z pyramid construction.
    pub fn hiz_build_shader_source() -> &'static str {
        HIZ_BUILD_WGSL
    }
}

// ---------------------------------------------------------------------------
// OcclusionCulling
// ---------------------------------------------------------------------------

/// Two-pass occlusion culling system.
///
/// Pass 1: Render large occluders to build the depth pyramid.
/// Pass 2: Test all instances against the depth pyramid.
pub struct OcclusionCulling {
    /// Depth pyramid.
    pub depth_pyramid: DepthPyramid,
    /// Indices of occluder instances (large objects rendered first).
    pub occluder_indices: Vec<u32>,
    /// Whether occlusion culling is enabled.
    pub enabled: bool,
    /// Minimum screen-space size (in pixels) for an object to be an occluder.
    pub occluder_min_screen_size: f32,
    /// Conservative depth bias for occlusion testing.
    pub depth_bias: f32,
}

impl OcclusionCulling {
    /// Create a new occlusion culling system.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            depth_pyramid: DepthPyramid::new(width, height),
            occluder_indices: Vec::new(),
            enabled: true,
            occluder_min_screen_size: 64.0,
            depth_bias: 0.001,
        }
    }

    /// Resize the depth pyramid (e.g. on window resize).
    pub fn resize(&mut self, width: u32, height: u32) {
        self.depth_pyramid = DepthPyramid::new(width, height);
    }

    /// Classify instances into occluders and occludees.
    pub fn classify_instances(
        &mut self,
        scene: &GPUScene,
        view_projection: &Mat4,
        viewport_height: f32,
    ) {
        self.occluder_indices.clear();

        for (i, instance) in scene.instance_data().iter().enumerate() {
            if !instance.has_flag(InstanceData::FLAG_OCCLUDER) {
                continue;
            }

            // Estimate screen-space size.
            let world_mat = Mat4::from_cols_array_2d(&instance.world_matrix);
            let center = world_mat.transform_point3(Vec3::from_array(
                [(instance.aabb_min[0] + instance.aabb_max[0]) * 0.5,
                 (instance.aabb_min[1] + instance.aabb_max[1]) * 0.5,
                 (instance.aabb_min[2] + instance.aabb_max[2]) * 0.5],
            ));

            let clip = *view_projection * center.extend(1.0);
            if clip.w <= 0.0 {
                continue;
            }

            let radius = instance.bounding_sphere[3];
            let screen_size = radius * viewport_height / clip.w;

            if screen_size >= self.occluder_min_screen_size {
                self.occluder_indices.push(i as u32);
            }
        }

        // Sort occluders front-to-back for early-Z.
        let vp = *view_projection;
        let instances = scene.instance_data();
        self.occluder_indices.sort_by(|&a, &b| {
            let ia = &instances[a as usize];
            let ib = &instances[b as usize];

            let wa = Mat4::from_cols_array_2d(&ia.world_matrix);
            let wb = Mat4::from_cols_array_2d(&ib.world_matrix);

            let da = (vp * wa.col(3)).z;
            let db = (vp * wb.col(3)).z;

            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Test if an instance is occluded (CPU fallback).
    pub fn test_instance_occluded(
        &self,
        instance: &InstanceData,
        view_projection: &Mat4,
    ) -> bool {
        if !self.enabled {
            return false;
        }

        let world_mat = Mat4::from_cols_array_2d(&instance.world_matrix);

        // Transform AABB corners to clip space and compute screen-space rect.
        let aabb_min = Vec3::from_array(instance.aabb_min);
        let aabb_max = Vec3::from_array(instance.aabb_max);

        let corners = [
            Vec3::new(aabb_min.x, aabb_min.y, aabb_min.z),
            Vec3::new(aabb_max.x, aabb_min.y, aabb_min.z),
            Vec3::new(aabb_min.x, aabb_max.y, aabb_min.z),
            Vec3::new(aabb_max.x, aabb_max.y, aabb_min.z),
            Vec3::new(aabb_min.x, aabb_min.y, aabb_max.z),
            Vec3::new(aabb_max.x, aabb_min.y, aabb_max.z),
            Vec3::new(aabb_min.x, aabb_max.y, aabb_max.z),
            Vec3::new(aabb_max.x, aabb_max.y, aabb_max.z),
        ];

        let mut screen_min = Vec2::splat(f32::MAX);
        let mut screen_max = Vec2::splat(f32::MIN);
        let mut max_depth: f32 = 0.0;
        let mut all_behind = true;

        for corner in &corners {
            let world_pos = world_mat.transform_point3(*corner);
            let clip = *view_projection * world_pos.extend(1.0);

            if clip.w <= 0.0 {
                continue;
            }
            all_behind = false;

            let ndc = Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
            let screen = Vec2::new(ndc.x * 0.5 + 0.5, ndc.y * 0.5 + 0.5);

            screen_min = screen_min.min(screen);
            screen_max = screen_max.max(screen);
            max_depth = max_depth.max(ndc.z);
        }

        if all_behind {
            return true; // Behind camera -- not visible.
        }

        // Clamp to viewport.
        screen_min = screen_min.max(Vec2::ZERO);
        screen_max = screen_max.min(Vec2::ONE);

        if screen_min.x >= screen_max.x || screen_min.y >= screen_max.y {
            return true; // Outside viewport.
        }

        // Test against depth pyramid with conservative bias.
        self.depth_pyramid.test_rect_occluded(
            screen_min.x,
            screen_min.y,
            screen_max.x,
            screen_max.y,
            max_depth + self.depth_bias,
        )
    }

    /// Generate the WGSL compute shader source for GPU instance culling.
    pub fn cull_shader_source() -> &'static str {
        GPU_CULL_WGSL
    }
}

// ---------------------------------------------------------------------------
// IndirectDrawManager
// ---------------------------------------------------------------------------

/// Manages the generation and batching of indirect draw commands.
///
/// The draw manager takes a scene, performs frustum and occlusion culling,
/// then generates a buffer of `IndirectDrawCommand` entries sorted by
/// material/pipeline to minimise state changes.
pub struct IndirectDrawManager {
    /// Generated draw commands for the current frame.
    draw_commands: Vec<IndirectDrawCommand>,
    /// Material/pipeline sort keys for each draw command.
    sort_keys: Vec<u64>,
    /// Number of visible instances after culling.
    visible_instance_count: u32,
    /// Total draw call count after merging.
    draw_call_count: u32,
    /// Total triangle count of visible geometry.
    visible_triangle_count: u64,
    /// Whether to use GPU-driven culling (vs CPU fallback).
    pub gpu_culling_enabled: bool,
    /// Occlusion culling state.
    pub occlusion: OcclusionCulling,
}

impl IndirectDrawManager {
    /// Create a new indirect draw manager.
    pub fn new(viewport_width: u32, viewport_height: u32) -> Self {
        Self {
            draw_commands: Vec::with_capacity(MAX_DRAW_COMMANDS),
            sort_keys: Vec::with_capacity(MAX_DRAW_COMMANDS),
            visible_instance_count: 0,
            draw_call_count: 0,
            visible_triangle_count: 0,
            gpu_culling_enabled: true,
            occlusion: OcclusionCulling::new(viewport_width, viewport_height),
        }
    }

    /// Perform CPU-side culling and generate indirect draw commands.
    ///
    /// This is the CPU fallback path. When GPU culling is enabled, the
    /// compute shader writes directly to the indirect draw buffer.
    pub fn build_draw_commands(
        &mut self,
        scene: &GPUScene,
        frustum: &Frustum,
        view_projection: &Mat4,
    ) {
        self.draw_commands.clear();
        self.sort_keys.clear();
        self.visible_instance_count = 0;
        self.visible_triangle_count = 0;

        // Per-mesh draw command accumulator.
        // Key: (mesh_id, material_id) -> (command, sort_key)
        let mut draw_map: HashMap<(u32, u32), (IndirectDrawCommand, u64)> = HashMap::new();

        let instances = scene.instance_data();
        let descriptors = scene.mesh_descriptor_data();

        for (i, instance) in instances.iter().enumerate() {
            if instance.flags == 0 {
                continue; // Removed instance.
            }

            // Frustum cull.
            let world_mat = Mat4::from_cols_array_2d(&instance.world_matrix);
            let world_center = world_mat.transform_point3(Vec3::new(
                instance.bounding_sphere[0],
                instance.bounding_sphere[1],
                instance.bounding_sphere[2],
            ));
            let world_radius = instance.bounding_sphere[3]
                * world_mat.x_axis.length().max(world_mat.y_axis.length().max(world_mat.z_axis.length()));

            if !frustum.test_sphere(world_center, world_radius) {
                continue;
            }

            // Occlusion cull.
            if self.occlusion.enabled {
                if self.occlusion.test_instance_occluded(instance, view_projection) {
                    continue;
                }
            }

            self.visible_instance_count += 1;

            // Look up mesh descriptor.
            let mesh_id = instance.mesh_id;
            if (mesh_id as usize) >= descriptors.len() {
                continue;
            }

            let desc = &descriptors[mesh_id as usize];
            self.visible_triangle_count += desc.index_count as u64 / 3;

            // Accumulate into draw commands.
            let key = (mesh_id, instance.material_id);
            let sort_key = ((instance.material_id as u64) << 32) | mesh_id as u64;

            let entry = draw_map.entry(key).or_insert_with(|| {
                (
                    IndirectDrawCommand {
                        index_count: desc.index_count,
                        instance_count: 0,
                        first_index: desc.index_offset,
                        base_vertex: desc.vertex_offset as i32,
                        first_instance: i as u32,
                    },
                    sort_key,
                )
            });

            entry.0.instance_count += 1;
        }

        // Collect and sort draw commands.
        let mut commands: Vec<(u64, IndirectDrawCommand)> = draw_map
            .into_values()
            .map(|(cmd, key)| (key, cmd))
            .collect();

        commands.sort_by_key(|&(key, _)| key);

        for (key, cmd) in commands {
            if !cmd.is_empty() {
                self.sort_keys.push(key);
                self.draw_commands.push(cmd);
            }
        }

        self.draw_call_count = self.draw_commands.len() as u32;
    }

    /// Get the generated draw commands.
    pub fn draw_commands(&self) -> &[IndirectDrawCommand] {
        &self.draw_commands
    }

    /// Number of draw calls after merging.
    pub fn draw_call_count(&self) -> u32 {
        self.draw_call_count
    }

    /// Number of visible instances after culling.
    pub fn visible_instance_count(&self) -> u32 {
        self.visible_instance_count
    }

    /// Total triangle count of visible geometry.
    pub fn visible_triangle_count(&self) -> u64 {
        self.visible_triangle_count
    }

    /// Get the raw draw command data as bytes for GPU upload.
    pub fn draw_command_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.draw_commands.as_ptr() as *const u8,
                self.draw_commands.len() * IndirectDrawCommand::SIZE,
            )
        }
    }

    /// Resize the occlusion culling system.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.occlusion.resize(width, height);
    }
}

// ---------------------------------------------------------------------------
// DrawBatch
// ---------------------------------------------------------------------------

/// A batch of draw commands that share the same pipeline state and can be
/// issued as a single multi-draw-indirect call.
#[derive(Debug, Clone)]
pub struct DrawBatch {
    /// Pipeline/material hash for this batch.
    pub pipeline_hash: u64,
    /// Offset into the indirect draw buffer.
    pub command_offset: u32,
    /// Number of draw commands in this batch.
    pub command_count: u32,
    /// Total instance count across all commands.
    pub total_instances: u32,
    /// Total triangle count across all commands.
    pub total_triangles: u64,
}

/// Merge consecutive draw commands with the same pipeline into batches.
pub fn merge_draw_commands(
    commands: &[IndirectDrawCommand],
    sort_keys: &[u64],
) -> Vec<DrawBatch> {
    if commands.is_empty() {
        return Vec::new();
    }

    let mut batches = Vec::new();
    let mut current_key = sort_keys[0] >> 32; // Material portion.
    let mut batch_start = 0u32;
    let mut batch_instances = commands[0].instance_count;
    let mut batch_triangles = commands[0].index_count as u64 / 3 * commands[0].instance_count as u64;

    for i in 1..commands.len() {
        let material_key = sort_keys[i] >> 32;

        if material_key != current_key {
            // Flush current batch.
            batches.push(DrawBatch {
                pipeline_hash: current_key,
                command_offset: batch_start,
                command_count: i as u32 - batch_start,
                total_instances: batch_instances,
                total_triangles: batch_triangles,
            });

            current_key = material_key;
            batch_start = i as u32;
            batch_instances = 0;
            batch_triangles = 0;
        }

        batch_instances += commands[i].instance_count;
        batch_triangles += commands[i].index_count as u64 / 3 * commands[i].instance_count as u64;
    }

    // Final batch.
    batches.push(DrawBatch {
        pipeline_hash: current_key,
        command_offset: batch_start,
        command_count: commands.len() as u32 - batch_start,
        total_instances: batch_instances,
        total_triangles: batch_triangles,
    });

    batches
}

// ---------------------------------------------------------------------------
// WGSL Shaders
// ---------------------------------------------------------------------------

/// WGSL compute shader for Hi-Z depth pyramid construction.
const HIZ_BUILD_WGSL: &str = r#"
// Hi-Z Depth Pyramid Build Shader
// Reduces depth buffer to a mip chain using max operation.

@group(0) @binding(0) var input_depth: texture_2d<f32>;
@group(0) @binding(1) var output_mip: texture_storage_2d<r32float, write>;

struct PushConstants {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
}

@group(0) @binding(2) var<uniform> params: PushConstants;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.output_width || gid.y >= params.output_height) {
        return;
    }

    let src_x = gid.x * 2u;
    let src_y = gid.y * 2u;

    let d00 = textureLoad(input_depth, vec2<i32>(i32(src_x), i32(src_y)), 0).r;
    let d10 = textureLoad(input_depth, vec2<i32>(i32(min(src_x + 1u, params.input_width - 1u)), i32(src_y)), 0).r;
    let d01 = textureLoad(input_depth, vec2<i32>(i32(src_x), i32(min(src_y + 1u, params.input_height - 1u))), 0).r;
    let d11 = textureLoad(input_depth, vec2<i32>(i32(min(src_x + 1u, params.input_width - 1u)), i32(min(src_y + 1u, params.input_height - 1u))), 0).r;

    let max_depth = max(max(d00, d10), max(d01, d11));
    textureStore(output_mip, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(max_depth, 0.0, 0.0, 0.0));
}
"#;

/// WGSL compute shader for GPU instance culling (frustum + Hi-Z occlusion).
const GPU_CULL_WGSL: &str = r#"
// GPU Instance Culling Compute Shader
// Performs frustum culling and Hi-Z occlusion culling per instance.
// Outputs a compacted indirect draw buffer.

struct InstanceData {
    world_matrix: mat4x4<f32>,
    prev_world_matrix: mat4x4<f32>,
    bounding_sphere: vec4<f32>,
    aabb_min: vec3<f32>,
    material_id: u32,
    aabb_max: vec3<f32>,
    mesh_id: u32,
    lod_level: u32,
    visible: u32,
    flags: u32,
    _pad: u32,
}

struct DrawCommand {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct MeshDescriptor {
    vertex_offset: u32,
    vertex_count: u32,
    index_offset: u32,
    index_count: u32,
    bounding_radius: f32,
    lod_level: u32,
    lod_error: f32,
    _pad: u32,
}

struct CullUniforms {
    view_projection: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
    camera_position: vec4<f32>,
    viewport_size: vec2<f32>,
    enable_frustum_cull: u32,
    enable_occlusion_cull: u32,
    instance_count: u32,
    hiz_mip_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> instances: array<InstanceData>;
@group(0) @binding(1) var<storage, read> mesh_descriptors: array<MeshDescriptor>;
@group(0) @binding(2) var<storage, read_write> draw_commands: array<DrawCommand>;
@group(0) @binding(3) var<storage, read_write> visible_count: atomic<u32>;
@group(0) @binding(4) var<uniform> uniforms: CullUniforms;
@group(0) @binding(5) var hiz_texture: texture_2d<f32>;

fn test_sphere_frustum(center: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = uniforms.frustum_planes[i];
        let dist = dot(plane.xyz, center) + plane.w;
        if (dist < -radius) {
            return false;
        }
    }
    return true;
}

fn test_hiz_occlusion(min_ndc: vec2<f32>, max_ndc: vec2<f32>, test_depth: f32) -> bool {
    let screen_min = min_ndc * 0.5 + 0.5;
    let screen_max = max_ndc * 0.5 + 0.5;

    let rect_size = (screen_max - screen_min) * uniforms.viewport_size;
    let max_dim = max(rect_size.x, rect_size.y);
    let mip_level = u32(ceil(log2(max(max_dim, 1.0))));
    let safe_mip = min(mip_level, uniforms.hiz_mip_count - 1u);

    let center_uv = (screen_min + screen_max) * 0.5;
    let texel = vec2<i32>(center_uv * uniforms.viewport_size / f32(1u << safe_mip));

    let pyramid_depth = textureLoad(hiz_texture, texel, i32(safe_mip)).r;
    return test_depth > pyramid_depth;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let instance_id = gid.x;
    if (instance_id >= uniforms.instance_count) {
        return;
    }

    let instance = instances[instance_id];
    if (instance.flags == 0u) {
        return;
    }

    // Transform bounding sphere to world space.
    let world_center = (instance.world_matrix * vec4<f32>(instance.bounding_sphere.xyz, 1.0)).xyz;
    let scale = max(
        length(instance.world_matrix[0].xyz),
        max(length(instance.world_matrix[1].xyz), length(instance.world_matrix[2].xyz))
    );
    let world_radius = instance.bounding_sphere.w * scale;

    // Frustum cull.
    if (uniforms.enable_frustum_cull != 0u) {
        if (!test_sphere_frustum(world_center, world_radius)) {
            return;
        }
    }

    // Occlusion cull.
    if (uniforms.enable_occlusion_cull != 0u) {
        let clip = uniforms.view_projection * vec4<f32>(world_center, 1.0);
        if (clip.w > 0.0) {
            let ndc = clip.xyz / clip.w;
            let proj_radius = world_radius / clip.w;
            let min_ndc = ndc.xy - vec2<f32>(proj_radius);
            let max_ndc = ndc.xy + vec2<f32>(proj_radius);

            if (test_hiz_occlusion(min_ndc, max_ndc, ndc.z)) {
                return;
            }
        }
    }

    // Instance passed culling -- append a draw command.
    let mesh = mesh_descriptors[instance.mesh_id];
    let draw_idx = atomicAdd(&visible_count, 1u);

    draw_commands[draw_idx].index_count = mesh.index_count;
    draw_commands[draw_idx].instance_count = 1u;
    draw_commands[draw_idx].first_index = mesh.index_offset;
    draw_commands[draw_idx].base_vertex = i32(mesh.vertex_offset);
    draw_commands[draw_idx].first_instance = instance_id;
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indirect_draw_command_size() {
        assert_eq!(IndirectDrawCommand::SIZE, 20);
    }

    #[test]
    fn test_instance_data_default() {
        let inst = InstanceData::default();
        assert_eq!(inst.visible, 1);
        assert!(inst.has_flag(InstanceData::FLAG_CAST_SHADOW));
    }

    #[test]
    fn test_gpu_scene_add_remove() {
        let mut scene = GPUScene::new();
        let idx = scene.add_instance(InstanceData::default());
        assert_eq!(scene.active_instance_count(), 1);

        scene.remove_instance(idx);
        assert_eq!(scene.active_instance_count(), 0);
    }

    #[test]
    fn test_gpu_scene_reuse_slots() {
        let mut scene = GPUScene::new();
        let idx1 = scene.add_instance(InstanceData::default());
        scene.remove_instance(idx1);

        let idx2 = scene.add_instance(InstanceData::default());
        assert_eq!(idx2, idx1); // Should reuse the slot.
        assert_eq!(scene.active_instance_count(), 1);
    }

    #[test]
    fn test_mesh_descriptor_register() {
        let mut scene = GPUScene::new();
        let desc = MeshDescriptor::new(0, 100, 0, 300, 1.0);
        let id = scene.register_mesh(42, desc);
        assert_eq!(id, 0);

        // Re-registering same asset should return same ID.
        let id2 = scene.register_mesh(42, desc);
        assert_eq!(id2, 0);
        assert_eq!(scene.mesh_count(), 1);
    }

    #[test]
    fn test_depth_pyramid_creation() {
        let pyramid = DepthPyramid::new(1024, 768);
        assert!(pyramid.mip_count > 0);
        assert_eq!(pyramid.mip_dimensions[0], (1024, 768));
    }

    #[test]
    fn test_depth_pyramid_build() {
        let mut pyramid = DepthPyramid::new(4, 4);
        let depth_data = vec![
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
            0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9,
        ];
        pyramid.build_from_depth_buffer(&depth_data);

        // Mip 1 should have max of each 2x2 block.
        assert!(pyramid.mip_data[1].len() <= 4);
        assert!(pyramid.mip_data[1][0] >= 0.6); // max of top-left 2x2
    }

    #[test]
    fn test_draw_batch_merging() {
        let commands = vec![
            IndirectDrawCommand::new(36, 10, 0, 0, 0),  // mat 0, mesh 0
            IndirectDrawCommand::new(36, 5, 0, 0, 10),   // mat 0, mesh 1
            IndirectDrawCommand::new(24, 3, 100, 50, 0), // mat 1, mesh 0
        ];
        let sort_keys = vec![
            0u64 << 32 | 0,
            0u64 << 32 | 1,
            1u64 << 32 | 0,
        ];

        let batches = merge_draw_commands(&commands, &sort_keys);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].command_count, 2);
        assert_eq!(batches[0].total_instances, 15);
        assert_eq!(batches[1].command_count, 1);
    }

    #[test]
    fn test_build_draw_commands() {
        let mut scene = GPUScene::new();
        let desc = MeshDescriptor::new(0, 24, 0, 36, 1.0);
        scene.register_mesh(1, desc);

        let instance = InstanceData::new(Mat4::IDENTITY, &AABB::new(-Vec3::ONE, Vec3::ONE), 0, 0);
        scene.add_instance(instance);

        let frustum = Frustum::from_view_projection(
            &(Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0)
                * Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y)),
        );

        let vp = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0)
            * Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);

        let mut manager = IndirectDrawManager::new(1920, 1080);
        manager.occlusion.enabled = false;
        manager.build_draw_commands(&scene, &frustum, &vp);

        assert!(manager.visible_instance_count() >= 1);
        assert!(manager.draw_call_count() >= 1);
    }
}
