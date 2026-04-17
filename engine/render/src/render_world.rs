// engine/render/src/render_world.rs
//
// Render world: thread-safe extracted render data from the ECS world.
// Separates simulation from rendering by extracting mesh, material,
// transform, and visibility data into a dedicated render representation
// that can be consumed by the render thread independently.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Entity ID in the render world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderEntityId(pub u64);

/// Mesh handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshHandle(pub u32);

/// Material handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialHandle(pub u32);

/// Texture handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub u32);

// ---------------------------------------------------------------------------
// Math types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 { pub x: f32, pub y: f32, pub z: f32, pub w: f32 }
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat { pub x: f32, pub y: f32, pub z: f32, pub w: f32 }
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 { pub data: [f32; 16] }

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0, z: 1.0 };
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn distance(self, r: Self) -> f32 {
        let dx = self.x - r.x;
        let dy = self.y - r.y;
        let dz = self.z - r.z;
        (dx*dx + dy*dy + dz*dz).sqrt()
    }
}

impl Quat {
    pub const IDENTITY: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self { Self { x, y, z, w } }
}

impl Mat4 {
    pub const IDENTITY: Self = Self { data: [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]};

    pub fn from_trs(pos: Vec3, rot: Quat, scale: Vec3) -> Self {
        let (qx, qy, qz, qw) = (rot.x, rot.y, rot.z, rot.w);
        let x2 = qx + qx; let y2 = qy + qy; let z2 = qz + qz;
        let xx = qx * x2; let xy = qx * y2; let xz = qx * z2;
        let yy = qy * y2; let yz = qy * z2; let zz = qz * z2;
        let wx = qw * x2; let wy = qw * y2; let wz = qw * z2;
        Self { data: [
            (1.0 - (yy + zz)) * scale.x, (xy + wz) * scale.x, (xz - wy) * scale.x, 0.0,
            (xy - wz) * scale.y, (1.0 - (xx + zz)) * scale.y, (yz + wx) * scale.y, 0.0,
            (xz + wy) * scale.z, (yz - wx) * scale.z, (1.0 - (xx + yy)) * scale.z, 0.0,
            pos.x, pos.y, pos.z, 1.0,
        ]}
    }
}

// ---------------------------------------------------------------------------
// AABB
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub const ZERO: Self = Self { min: Vec3::ZERO, max: Vec3::ZERO };

    pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }

    pub fn center(&self) -> Vec3 {
        Vec3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    pub fn extents(&self) -> Vec3 {
        Vec3::new(
            (self.max.x - self.min.x) * 0.5,
            (self.max.y - self.min.y) * 0.5,
            (self.max.z - self.min.z) * 0.5,
        )
    }

    pub fn merge(self, other: Self) -> Self {
        Self {
            min: Vec3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Vec3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }

    pub fn transform(&self, matrix: &Mat4) -> Self {
        let m = &matrix.data;
        let center = self.center();
        let ext = self.extents();

        let new_center = Vec3::new(
            m[0]*center.x + m[4]*center.y + m[8]*center.z + m[12],
            m[1]*center.x + m[5]*center.y + m[9]*center.z + m[13],
            m[2]*center.x + m[6]*center.y + m[10]*center.z + m[14],
        );

        let new_ext = Vec3::new(
            m[0].abs()*ext.x + m[4].abs()*ext.y + m[8].abs()*ext.z,
            m[1].abs()*ext.x + m[5].abs()*ext.y + m[9].abs()*ext.z,
            m[2].abs()*ext.x + m[6].abs()*ext.y + m[10].abs()*ext.z,
        );

        Self {
            min: Vec3::new(new_center.x - new_ext.x, new_center.y - new_ext.y, new_center.z - new_ext.z),
            max: Vec3::new(new_center.x + new_ext.x, new_center.y + new_ext.y, new_center.z + new_ext.z),
        }
    }
}

// ---------------------------------------------------------------------------
// Render components (extracted from ECS)
// ---------------------------------------------------------------------------

/// Extracted transform for rendering.
#[derive(Debug, Clone)]
pub struct RenderTransform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub world_matrix: Mat4,
    pub prev_world_matrix: Mat4, // for motion vectors
    pub dirty: bool,
}

impl Default for RenderTransform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            world_matrix: Mat4::IDENTITY,
            prev_world_matrix: Mat4::IDENTITY,
            dirty: true,
        }
    }
}

impl RenderTransform {
    pub fn update_matrix(&mut self) {
        self.prev_world_matrix = self.world_matrix;
        self.world_matrix = Mat4::from_trs(self.position, self.rotation, self.scale);
        self.dirty = false;
    }
}

/// Extracted mesh renderer component.
#[derive(Debug, Clone)]
pub struct RenderMeshData {
    pub mesh: MeshHandle,
    pub materials: Vec<MaterialHandle>,
    pub sub_mesh_count: u32,
    pub cast_shadows: bool,
    pub receive_shadows: bool,
    pub lod_level: u32,
    pub local_bounds: AABB,
    pub world_bounds: AABB,
    pub layer_mask: u32,
    pub render_order: i32,
    pub is_static: bool,
    pub instance_data: Option<Vec<u8>>,
}

/// Extracted light data.
#[derive(Debug, Clone)]
pub struct RenderLight {
    pub light_type: LightType,
    pub position: Vec3,
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
    pub inner_angle: f32,
    pub outer_angle: f32,
    pub cast_shadows: bool,
    pub shadow_bias: f32,
    pub shadow_map_index: Option<u32>,
    pub layer_mask: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightType {
    Directional,
    Point,
    Spot,
    Area,
}

/// Visibility result from culling.
#[derive(Debug, Clone)]
pub struct VisibilityResult {
    pub visible_entities: Vec<RenderEntityId>,
    pub visible_lights: Vec<RenderEntityId>,
    pub shadow_casters: Vec<RenderEntityId>,
    pub frame: u64,
}

impl VisibilityResult {
    pub fn new() -> Self {
        Self {
            visible_entities: Vec::new(),
            visible_lights: Vec::new(),
            shadow_casters: Vec::new(),
            frame: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Camera data
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RenderCamera {
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
    pub view_matrix: Mat4,
    pub projection_matrix: Mat4,
    pub view_projection: Mat4,
    pub prev_view_projection: Mat4,
    pub near_plane: f32,
    pub far_plane: f32,
    pub fov_y: f32,
    pub aspect_ratio: f32,
    pub jitter: [f32; 2], // TAA jitter
    pub viewport: [u32; 4], // x, y, w, h
}

impl Default for RenderCamera {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            forward: Vec3::new(0.0, 0.0, -1.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            view_matrix: Mat4::IDENTITY,
            projection_matrix: Mat4::IDENTITY,
            view_projection: Mat4::IDENTITY,
            prev_view_projection: Mat4::IDENTITY,
            near_plane: 0.1,
            far_plane: 1000.0,
            fov_y: 60.0_f32.to_radians(),
            aspect_ratio: 16.0 / 9.0,
            jitter: [0.0, 0.0],
            viewport: [0, 0, 1920, 1080],
        }
    }
}

// ---------------------------------------------------------------------------
// Render world
// ---------------------------------------------------------------------------

/// Thread-safe render world that holds extracted render data.
pub struct RenderWorld {
    // Entity data
    transforms: HashMap<RenderEntityId, RenderTransform>,
    meshes: HashMap<RenderEntityId, RenderMeshData>,
    lights: HashMap<RenderEntityId, RenderLight>,

    // Camera
    pub camera: RenderCamera,

    // Visibility
    pub visibility: VisibilityResult,

    // Frame tracking
    pub frame_number: u64,
    pub extraction_time_ms: f32,

    // Change tracking
    added_entities: Vec<RenderEntityId>,
    removed_entities: Vec<RenderEntityId>,
    changed_transforms: Vec<RenderEntityId>,

    // Settings
    pub render_layer_mask: u32,
    pub shadow_distance: f32,
    pub lod_bias: f32,
}

impl RenderWorld {
    pub fn new() -> Self {
        Self {
            transforms: HashMap::new(),
            meshes: HashMap::new(),
            lights: HashMap::new(),
            camera: RenderCamera::default(),
            visibility: VisibilityResult::new(),
            frame_number: 0,
            extraction_time_ms: 0.0,
            added_entities: Vec::new(),
            removed_entities: Vec::new(),
            changed_transforms: Vec::new(),
            render_layer_mask: 0xFFFFFFFF,
            shadow_distance: 100.0,
            lod_bias: 0.0,
        }
    }

    /// Begin extraction for a new frame.
    pub fn begin_extraction(&mut self) {
        self.added_entities.clear();
        self.removed_entities.clear();
        self.changed_transforms.clear();
        self.frame_number += 1;
    }

    /// Add or update a render entity with a mesh.
    pub fn extract_mesh_renderer(
        &mut self,
        entity: RenderEntityId,
        transform: RenderTransform,
        mesh_data: RenderMeshData,
    ) {
        let is_new = !self.transforms.contains_key(&entity);
        self.transforms.insert(entity, transform);
        self.meshes.insert(entity, mesh_data);

        if is_new {
            self.added_entities.push(entity);
        } else {
            self.changed_transforms.push(entity);
        }
    }

    /// Add or update a light.
    pub fn extract_light(
        &mut self,
        entity: RenderEntityId,
        transform: RenderTransform,
        light: RenderLight,
    ) {
        self.transforms.insert(entity, transform);
        self.lights.insert(entity, light);
    }

    /// Remove an entity.
    pub fn remove_entity(&mut self, entity: RenderEntityId) {
        self.transforms.remove(&entity);
        self.meshes.remove(&entity);
        self.lights.remove(&entity);
        self.removed_entities.push(entity);
    }

    /// Update transforms (compute world matrices).
    pub fn update_transforms(&mut self) {
        for transform in self.transforms.values_mut() {
            if transform.dirty {
                transform.update_matrix();
            }
        }

        // Update world bounds for meshes
        let transforms = &self.transforms;
        for (entity, mesh) in &mut self.meshes {
            if let Some(transform) = transforms.get(entity) {
                mesh.world_bounds = mesh.local_bounds.transform(&transform.world_matrix);
            }
        }
    }

    /// Get transform for an entity.
    pub fn get_transform(&self, entity: RenderEntityId) -> Option<&RenderTransform> {
        self.transforms.get(&entity)
    }

    /// Get mesh data for an entity.
    pub fn get_mesh(&self, entity: RenderEntityId) -> Option<&RenderMeshData> {
        self.meshes.get(&entity)
    }

    /// Get light data for an entity.
    pub fn get_light(&self, entity: RenderEntityId) -> Option<&RenderLight> {
        self.lights.get(&entity)
    }

    /// Get all mesh entities.
    pub fn mesh_entities(&self) -> impl Iterator<Item = &RenderEntityId> {
        self.meshes.keys()
    }

    /// Get all light entities.
    pub fn light_entities(&self) -> impl Iterator<Item = &RenderEntityId> {
        self.lights.keys()
    }

    /// Perform frustum culling.
    pub fn cull(&mut self, frustum_planes: &[[f32; 4]; 6]) {
        self.visibility.visible_entities.clear();
        self.visibility.visible_lights.clear();
        self.visibility.shadow_casters.clear();
        self.visibility.frame = self.frame_number;

        for (&entity, mesh) in &self.meshes {
            if mesh.layer_mask & self.render_layer_mask == 0 {
                continue;
            }

            if frustum_aabb_test(frustum_planes, &mesh.world_bounds) {
                self.visibility.visible_entities.push(entity);

                if mesh.cast_shadows {
                    let dist = self.camera.position.distance(mesh.world_bounds.center());
                    if dist < self.shadow_distance {
                        self.visibility.shadow_casters.push(entity);
                    }
                }
            }
        }

        // All lights are visible (simplified; real engine would cull too)
        for &entity in self.lights.keys() {
            self.visibility.visible_lights.push(entity);
        }
    }

    /// Sort visible entities for rendering.
    pub fn sort_for_rendering(&mut self) {
        let camera_pos = self.camera.position;
        let transforms = &self.transforms;
        let meshes = &self.meshes;

        // Sort opaque front-to-back for early Z
        self.visibility.visible_entities.sort_by(|a, b| {
            let da = transforms.get(a).map(|t| camera_pos.distance(t.position)).unwrap_or(f32::MAX);
            let db = transforms.get(b).map(|t| camera_pos.distance(t.position)).unwrap_or(f32::MAX);

            // First by render order
            let oa = meshes.get(a).map(|m| m.render_order).unwrap_or(0);
            let ob = meshes.get(b).map(|m| m.render_order).unwrap_or(0);

            oa.cmp(&ob).then_with(|| da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal))
        });
    }

    /// Compute LOD level for an entity based on camera distance.
    pub fn compute_lod(&self, entity: RenderEntityId, lod_distances: &[f32]) -> u32 {
        let dist = match self.transforms.get(&entity) {
            Some(t) => self.camera.position.distance(t.position),
            None => return 0,
        };

        let biased_dist = dist * (1.0 + self.lod_bias);

        for (i, &threshold) in lod_distances.iter().enumerate() {
            if biased_dist < threshold {
                return i as u32;
            }
        }

        lod_distances.len() as u32
    }

    /// Get statistics about the render world.
    pub fn stats(&self) -> RenderWorldStats {
        RenderWorldStats {
            total_entities: self.transforms.len(),
            mesh_entities: self.meshes.len(),
            light_entities: self.lights.len(),
            visible_entities: self.visibility.visible_entities.len(),
            visible_lights: self.visibility.visible_lights.len(),
            shadow_casters: self.visibility.shadow_casters.len(),
            added_this_frame: self.added_entities.len(),
            removed_this_frame: self.removed_entities.len(),
            transforms_updated: self.changed_transforms.len(),
            frame_number: self.frame_number,
        }
    }

    /// Get entities added this frame.
    pub fn added_entities(&self) -> &[RenderEntityId] {
        &self.added_entities
    }

    /// Get entities removed this frame.
    pub fn removed_entities(&self) -> &[RenderEntityId] {
        &self.removed_entities
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.transforms.clear();
        self.meshes.clear();
        self.lights.clear();
        self.visibility = VisibilityResult::new();
    }
}

/// Frustum-AABB intersection test.
fn frustum_aabb_test(planes: &[[f32; 4]; 6], aabb: &AABB) -> bool {
    let center = aabb.center();
    let extents = aabb.extents();

    for plane in planes {
        let d = plane[0] * center.x + plane[1] * center.y + plane[2] * center.z + plane[3];
        let r = extents.x * plane[0].abs() + extents.y * plane[1].abs() + extents.z * plane[2].abs();
        if d + r < 0.0 {
            return false;
        }
    }
    true
}

#[derive(Debug, Clone)]
pub struct RenderWorldStats {
    pub total_entities: usize,
    pub mesh_entities: usize,
    pub light_entities: usize,
    pub visible_entities: usize,
    pub visible_lights: usize,
    pub shadow_casters: usize,
    pub added_this_frame: usize,
    pub removed_this_frame: usize,
    pub transforms_updated: usize,
    pub frame_number: u64,
}

// ---------------------------------------------------------------------------
// Shared render world (thread-safe wrapper)
// ---------------------------------------------------------------------------

/// Thread-safe shared render world using double-buffering.
pub struct SharedRenderWorld {
    front: Arc<RwLock<RenderWorld>>,
    back: Arc<RwLock<RenderWorld>>,
}

impl SharedRenderWorld {
    pub fn new() -> Self {
        Self {
            front: Arc::new(RwLock::new(RenderWorld::new())),
            back: Arc::new(RwLock::new(RenderWorld::new())),
        }
    }

    /// Get the front buffer for reading (render thread).
    pub fn front(&self) -> &Arc<RwLock<RenderWorld>> {
        &self.front
    }

    /// Get the back buffer for writing (game thread).
    pub fn back(&self) -> &Arc<RwLock<RenderWorld>> {
        &self.back
    }

    /// Swap front and back buffers. Call at the end of extraction.
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.front, &mut self.back);
    }
}

// ---------------------------------------------------------------------------
// Render batch
// ---------------------------------------------------------------------------

/// A render batch groups entities that share the same mesh and material
/// for instanced drawing.
#[derive(Debug, Clone)]
pub struct RenderBatch {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub entities: Vec<RenderEntityId>,
    pub instance_transforms: Vec<Mat4>,
    pub lod_level: u32,
    pub is_shadow_batch: bool,
    pub sort_key: u64,
}

impl RenderBatch {
    pub fn new(mesh: MeshHandle, material: MaterialHandle) -> Self {
        Self {
            mesh,
            material,
            entities: Vec::new(),
            instance_transforms: Vec::new(),
            lod_level: 0,
            is_shadow_batch: false,
            sort_key: 0,
        }
    }

    pub fn add_instance(&mut self, entity: RenderEntityId, transform: Mat4) {
        self.entities.push(entity);
        self.instance_transforms.push(transform);
    }

    pub fn instance_count(&self) -> usize {
        self.entities.len()
    }
}

/// Build render batches from the visible entities.
pub fn build_batches(world: &RenderWorld) -> Vec<RenderBatch> {
    let mut batch_map: HashMap<(u32, u32, u32), RenderBatch> = HashMap::new();

    for &entity in &world.visibility.visible_entities {
        let mesh_data = match world.get_mesh(entity) {
            Some(m) => m,
            None => continue,
        };
        let transform = match world.get_transform(entity) {
            Some(t) => t,
            None => continue,
        };

        for (sub_mesh, material) in mesh_data.materials.iter().enumerate() {
            let key = (mesh_data.mesh.0, material.0, mesh_data.lod_level);

            let batch = batch_map.entry(key).or_insert_with(|| {
                let mut b = RenderBatch::new(mesh_data.mesh, *material);
                b.lod_level = mesh_data.lod_level;
                // Sort key: material << 32 | mesh
                b.sort_key = ((material.0 as u64) << 32) | mesh_data.mesh.0 as u64;
                b
            });

            batch.add_instance(entity, transform.world_matrix);
        }
    }

    let mut batches: Vec<_> = batch_map.into_values().collect();
    batches.sort_by_key(|b| b.sort_key);
    batches
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_world_creation() {
        let world = RenderWorld::new();
        assert_eq!(world.frame_number, 0);
    }

    #[test]
    fn test_extract_mesh() {
        let mut world = RenderWorld::new();
        world.begin_extraction();

        let entity = RenderEntityId(1);
        world.extract_mesh_renderer(
            entity,
            RenderTransform::default(),
            RenderMeshData {
                mesh: MeshHandle(0),
                materials: vec![MaterialHandle(0)],
                sub_mesh_count: 1,
                cast_shadows: true,
                receive_shadows: true,
                lod_level: 0,
                local_bounds: AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
                world_bounds: AABB::ZERO,
                layer_mask: 0xFFFFFFFF,
                render_order: 0,
                is_static: true,
                instance_data: None,
            },
        );

        assert_eq!(world.stats().mesh_entities, 1);
        assert_eq!(world.added_entities().len(), 1);
    }

    #[test]
    fn test_remove_entity() {
        let mut world = RenderWorld::new();
        let entity = RenderEntityId(1);
        world.extract_mesh_renderer(entity, RenderTransform::default(), RenderMeshData {
            mesh: MeshHandle(0), materials: vec![], sub_mesh_count: 1,
            cast_shadows: false, receive_shadows: false, lod_level: 0,
            local_bounds: AABB::ZERO, world_bounds: AABB::ZERO,
            layer_mask: 0xFFFFFFFF, render_order: 0, is_static: false, instance_data: None,
        });
        world.remove_entity(entity);
        assert_eq!(world.stats().mesh_entities, 0);
    }

    #[test]
    fn test_frustum_cull() {
        let mut world = RenderWorld::new();
        let entity = RenderEntityId(1);
        world.extract_mesh_renderer(entity, RenderTransform::default(), RenderMeshData {
            mesh: MeshHandle(0), materials: vec![], sub_mesh_count: 1,
            cast_shadows: false, receive_shadows: false, lod_level: 0,
            local_bounds: AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
            world_bounds: AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
            layer_mask: 0xFFFFFFFF, render_order: 0, is_static: false, instance_data: None,
        });

        // Frustum that contains the origin
        let planes = [
            [1.0, 0.0, 0.0, 10.0],
            [-1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 10.0],
            [0.0, -1.0, 0.0, 10.0],
            [0.0, 0.0, 1.0, 10.0],
            [0.0, 0.0, -1.0, 10.0],
        ];
        world.cull(&planes);
        assert_eq!(world.visibility.visible_entities.len(), 1);
    }

    #[test]
    fn test_lod_selection() {
        let mut world = RenderWorld::new();
        let entity = RenderEntityId(1);
        let mut transform = RenderTransform::default();
        transform.position = Vec3::new(50.0, 0.0, 0.0);
        world.extract_mesh_renderer(entity, transform, RenderMeshData {
            mesh: MeshHandle(0), materials: vec![], sub_mesh_count: 1,
            cast_shadows: false, receive_shadows: false, lod_level: 0,
            local_bounds: AABB::ZERO, world_bounds: AABB::ZERO,
            layer_mask: 0xFFFFFFFF, render_order: 0, is_static: false, instance_data: None,
        });

        let lod = world.compute_lod(entity, &[20.0, 40.0, 80.0]);
        assert_eq!(lod, 2); // 50 > 40 but < 80 -> LOD 2
    }

    #[test]
    fn test_mat4_from_trs() {
        let mat = Mat4::from_trs(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY, Vec3::ONE);
        assert!((mat.data[12] - 1.0).abs() < 1e-6);
        assert!((mat.data[13] - 2.0).abs() < 1e-6);
        assert!((mat.data[14] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_batches() {
        let mut world = RenderWorld::new();
        let mat = MaterialHandle(0);
        let mesh = MeshHandle(0);

        for i in 0..5 {
            let entity = RenderEntityId(i);
            world.extract_mesh_renderer(entity, RenderTransform::default(), RenderMeshData {
                mesh, materials: vec![mat], sub_mesh_count: 1,
                cast_shadows: false, receive_shadows: false, lod_level: 0,
                local_bounds: AABB::ZERO, world_bounds: AABB::ZERO,
                layer_mask: 0xFFFFFFFF, render_order: 0, is_static: false, instance_data: None,
            });
        }

        // Make all visible
        world.visibility.visible_entities = (0..5).map(RenderEntityId).collect();

        let batches = build_batches(&world);
        assert_eq!(batches.len(), 1); // all same mesh+material
        assert_eq!(batches[0].instance_count(), 5);
    }
}
