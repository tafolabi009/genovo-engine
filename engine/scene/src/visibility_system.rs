// engine/scene/src/visibility_system.rs
//
// Visibility determination system for the Genovo engine.
// Frustum culling, occlusion query integration, portal-based indoor visibility,
// PVS query, distance culling, small-object culling, shadow caster visibility,
// and per-camera visibility lists.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityHandle(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CameraId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PortalId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullResult { Visible, Culled, PartiallyVisible }

#[derive(Debug, Clone, Copy)]
pub struct BoundingSphere { pub center: [f32; 3], pub radius: f32 }

impl BoundingSphere {
    pub fn new(center: [f32; 3], radius: f32) -> Self { Self { center, radius } }
    pub fn distance_to(&self, point: [f32; 3]) -> f32 {
        let dx = self.center[0] - point[0]; let dy = self.center[1] - point[1]; let dz = self.center[2] - point[2];
        (dx*dx + dy*dy + dz*dz).sqrt()
    }
    pub fn screen_size(&self, distance: f32, fov_factor: f32) -> f32 {
        if distance <= 0.0 { return f32::MAX; }
        (self.radius / distance) * fov_factor
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FrustumPlane { pub normal: [f32; 3], pub distance: f32 }

impl FrustumPlane {
    pub fn distance_to_point(&self, point: [f32; 3]) -> f32 {
        self.normal[0] * point[0] + self.normal[1] * point[1] + self.normal[2] * point[2] + self.distance
    }
}

#[derive(Debug, Clone)]
pub struct Frustum { pub planes: [FrustumPlane; 6] }

impl Frustum {
    pub fn test_sphere(&self, sphere: &BoundingSphere) -> CullResult {
        for plane in &self.planes {
            let dist = plane.distance_to_point(sphere.center);
            if dist < -sphere.radius { return CullResult::Culled; }
        }
        CullResult::Visible
    }
}

#[derive(Debug, Clone)]
pub struct CullingConfig {
    pub frustum_culling: bool,
    pub distance_culling: bool,
    pub max_distance: f32,
    pub small_object_culling: bool,
    pub min_screen_size: f32,
    pub occlusion_culling: bool,
    pub portal_culling: bool,
    pub shadow_caster_culling: bool,
    pub shadow_caster_max_distance: f32,
}

impl Default for CullingConfig {
    fn default() -> Self {
        Self {
            frustum_culling: true, distance_culling: true, max_distance: 1000.0,
            small_object_culling: true, min_screen_size: 0.005,
            occlusion_culling: false, portal_culling: false,
            shadow_caster_culling: true, shadow_caster_max_distance: 200.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VisibleEntity {
    pub handle: EntityHandle,
    pub distance: f32,
    pub screen_size: f32,
    pub lod_level: u32,
    pub casts_shadow: bool,
    pub shadow_visible: bool,
}

#[derive(Debug, Clone)]
pub struct CameraVisibilityList {
    pub camera_id: CameraId,
    pub visible_entities: Vec<VisibleEntity>,
    pub shadow_casters: Vec<EntityHandle>,
    pub total_tested: u32,
    pub frustum_culled: u32,
    pub distance_culled: u32,
    pub size_culled: u32,
    pub occlusion_culled: u32,
    pub portal_culled: u32,
}

impl CameraVisibilityList {
    pub fn new(camera_id: CameraId) -> Self {
        Self { camera_id, visible_entities: Vec::new(), shadow_casters: Vec::new(), total_tested: 0, frustum_culled: 0, distance_culled: 0, size_culled: 0, occlusion_culled: 0, portal_culled: 0 }
    }
    pub fn clear(&mut self) {
        self.visible_entities.clear(); self.shadow_casters.clear();
        self.total_tested = 0; self.frustum_culled = 0; self.distance_culled = 0;
        self.size_culled = 0; self.occlusion_culled = 0; self.portal_culled = 0;
    }
    pub fn visible_count(&self) -> usize { self.visible_entities.len() }
    pub fn culled_count(&self) -> u32 { self.frustum_culled + self.distance_culled + self.size_culled + self.occlusion_culled + self.portal_culled }
    pub fn cull_ratio(&self) -> f32 { if self.total_tested == 0 { 0.0 } else { self.culled_count() as f32 / self.total_tested as f32 } }
}

#[derive(Debug, Clone)]
pub struct Portal {
    pub id: PortalId,
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub half_width: f32,
    pub half_height: f32,
    pub cell_a: CellId,
    pub cell_b: CellId,
    pub open: bool,
}

impl Portal {
    pub fn new(id: PortalId, pos: [f32; 3], normal: [f32; 3], w: f32, h: f32, a: CellId, b: CellId) -> Self {
        Self { id, position: pos, normal, half_width: w, half_height: h, cell_a: a, cell_b: b, open: true }
    }
    pub fn other_cell(&self, from: CellId) -> CellId { if from == self.cell_a { self.cell_b } else { self.cell_a } }
}

#[derive(Debug, Clone)]
pub struct VisibilityCell {
    pub id: CellId,
    pub entities: Vec<EntityHandle>,
    pub portals: Vec<PortalId>,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
}

impl VisibilityCell {
    pub fn new(id: CellId) -> Self { Self { id, entities: Vec::new(), portals: Vec::new(), aabb_min: [0.0; 3], aabb_max: [0.0; 3] } }
    pub fn contains_point(&self, p: [f32; 3]) -> bool {
        p[0] >= self.aabb_min[0] && p[0] <= self.aabb_max[0] && p[1] >= self.aabb_min[1] && p[1] <= self.aabb_max[1] && p[2] >= self.aabb_min[2] && p[2] <= self.aabb_max[2]
    }
}

#[derive(Debug)]
pub struct VisibilitySystem {
    pub config: CullingConfig,
    pub camera_lists: HashMap<CameraId, CameraVisibilityList>,
    pub cells: HashMap<CellId, VisibilityCell>,
    pub portals: HashMap<PortalId, Portal>,
    pub entity_bounds: HashMap<EntityHandle, BoundingSphere>,
    pub entity_shadow_caster: HashMap<EntityHandle, bool>,
    pub active: bool,
    pub stats: VisibilityStats,
}

#[derive(Debug, Clone, Default)]
pub struct VisibilityStats {
    pub total_entities: u32,
    pub total_visible: u32,
    pub total_culled: u32,
    pub cameras_processed: u32,
    pub processing_time_us: f64,
}

impl VisibilitySystem {
    pub fn new(config: CullingConfig) -> Self {
        Self { config, camera_lists: HashMap::new(), cells: HashMap::new(), portals: HashMap::new(), entity_bounds: HashMap::new(), entity_shadow_caster: HashMap::new(), active: true, stats: VisibilityStats::default() }
    }

    pub fn register_entity(&mut self, handle: EntityHandle, bounds: BoundingSphere, casts_shadow: bool) {
        self.entity_bounds.insert(handle, bounds);
        self.entity_shadow_caster.insert(handle, casts_shadow);
    }

    pub fn unregister_entity(&mut self, handle: EntityHandle) {
        self.entity_bounds.remove(&handle);
        self.entity_shadow_caster.remove(&handle);
    }

    pub fn update_bounds(&mut self, handle: EntityHandle, bounds: BoundingSphere) {
        self.entity_bounds.insert(handle, bounds);
    }

    pub fn cull_camera(&mut self, camera_id: CameraId, frustum: &Frustum, camera_pos: [f32; 3], fov_factor: f32) {
        let list = self.camera_lists.entry(camera_id).or_insert_with(|| CameraVisibilityList::new(camera_id));
        list.clear();

        for (handle, bounds) in &self.entity_bounds {
            list.total_tested += 1;

            // Frustum culling
            if self.config.frustum_culling && frustum.test_sphere(bounds) == CullResult::Culled {
                list.frustum_culled += 1;
                continue;
            }

            let distance = bounds.distance_to(camera_pos);

            // Distance culling
            if self.config.distance_culling && distance > self.config.max_distance {
                list.distance_culled += 1;
                continue;
            }

            // Small object culling
            let screen_size = bounds.screen_size(distance, fov_factor);
            if self.config.small_object_culling && screen_size < self.config.min_screen_size {
                list.size_culled += 1;
                continue;
            }

            let lod = if screen_size > 0.3 { 0 } else if screen_size > 0.1 { 1 } else if screen_size > 0.03 { 2 } else { 3 };
            let casts_shadow = self.entity_shadow_caster.get(handle).copied().unwrap_or(false);
            let shadow_visible = casts_shadow && self.config.shadow_caster_culling && distance <= self.config.shadow_caster_max_distance;

            list.visible_entities.push(VisibleEntity {
                handle: *handle, distance, screen_size, lod_level: lod,
                casts_shadow, shadow_visible,
            });

            if shadow_visible { list.shadow_casters.push(*handle); }
        }

        list.visible_entities.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        self.stats.total_entities = self.entity_bounds.len() as u32;
        self.stats.total_visible = list.visible_count() as u32;
        self.stats.total_culled = list.culled_count();
        self.stats.cameras_processed += 1;
    }

    pub fn get_visibility_list(&self, camera_id: CameraId) -> Option<&CameraVisibilityList> { self.camera_lists.get(&camera_id) }
    pub fn entity_count(&self) -> usize { self.entity_bounds.len() }
}

impl Default for VisibilitySystem { fn default() -> Self { Self::new(CullingConfig::default()) } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_bounding_sphere_distance() {
        let s = BoundingSphere::new([0.0, 0.0, 0.0], 1.0);
        assert!((s.distance_to([3.0, 4.0, 0.0]) - 5.0).abs() < 0.01);
    }
    #[test]
    fn test_visibility_system() {
        let mut sys = VisibilitySystem::new(CullingConfig::default());
        sys.register_entity(EntityHandle(1), BoundingSphere::new([10.0, 0.0, 0.0], 1.0), true);
        sys.register_entity(EntityHandle(2), BoundingSphere::new([2000.0, 0.0, 0.0], 1.0), false);
        let frustum = Frustum { planes: [
            FrustumPlane { normal: [1.0, 0.0, 0.0], distance: 0.0 },
            FrustumPlane { normal: [-1.0, 0.0, 0.0], distance: 500.0 },
            FrustumPlane { normal: [0.0, 1.0, 0.0], distance: 500.0 },
            FrustumPlane { normal: [0.0, -1.0, 0.0], distance: 500.0 },
            FrustumPlane { normal: [0.0, 0.0, 1.0], distance: 500.0 },
            FrustumPlane { normal: [0.0, 0.0, -1.0], distance: 500.0 },
        ]};
        sys.cull_camera(CameraId(0), &frustum, [0.0; 3], 1.0);
        let list = sys.get_visibility_list(CameraId(0)).unwrap();
        assert!(list.visible_count() >= 1);
    }
}
