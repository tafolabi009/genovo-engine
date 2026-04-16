//! ECS integration for the terrain system.
//!
//! Provides components and systems that plug the terrain into the Genovo
//! entity-component-system framework.
//!
//! - [`TerrainComponent`] — links an entity to terrain data.
//! - [`TerrainSystem`] — per-frame LOD selection, chunk streaming, and
//!   vegetation culling.
//! - [`TerrainCollider`] — physics integration via heightfield collision.

use genovo_core::{Frustum, Transform, AABB};
use genovo_ecs::{Component, World};
use glam::{Mat4, Vec3};
use parking_lot::RwLock;
use std::sync::Arc;

use crate::heightmap::Heightmap;
use crate::lod::{LODSettings, TerrainChunkInfo, TerrainQuadtree, TerrainStreamer};
use crate::mesh_generation::TerrainMeshSettings;
use crate::texturing::TerrainMaterial;
use crate::vegetation::{VegetationInstance, VegetationLayer};

// ---------------------------------------------------------------------------
// TerrainComponent
// ---------------------------------------------------------------------------

/// ECS component that attaches terrain data to an entity.
///
/// This is the primary interface for the terrain subsystem within the ECS.
/// An entity with a `TerrainComponent` and a `Transform` will be rendered
/// as terrain.
pub struct TerrainComponent {
    /// The heightmap data.
    pub heightmap: Arc<RwLock<Heightmap>>,

    /// The terrain material (layers + splatmaps).
    pub material: TerrainMaterial,

    /// Mesh generation settings.
    pub mesh_settings: TerrainMeshSettings,

    /// LOD settings.
    pub lod_settings: LODSettings,

    /// Total terrain size in world units (square).
    pub terrain_size: f32,

    /// Vegetation layers.
    pub vegetation_layers: Vec<VegetationLayer>,

    /// Cached vegetation instances (regenerated when terrain changes).
    pub vegetation_instances: Vec<VegetationInstance>,

    /// The quadtree for LOD selection (built on init / terrain change).
    pub quadtree: Option<TerrainQuadtree>,

    /// Chunks currently selected for rendering (updated each frame).
    pub visible_chunks: Vec<TerrainChunkInfo>,

    /// Whether the terrain data has been modified and meshes need rebuilding.
    pub dirty: bool,

    /// Whether vegetation needs to be re-scattered.
    pub vegetation_dirty: bool,

    /// Terrain streaming state.
    pub streamer: Option<TerrainStreamer>,
}

impl Component for TerrainComponent {}

impl TerrainComponent {
    /// Creates a new terrain component with the given heightmap.
    pub fn new(
        heightmap: Heightmap,
        terrain_size: f32,
        material: TerrainMaterial,
    ) -> Self {
        Self {
            heightmap: Arc::new(RwLock::new(heightmap)),
            material,
            mesh_settings: TerrainMeshSettings::default(),
            lod_settings: LODSettings::default(),
            terrain_size,
            vegetation_layers: Vec::new(),
            vegetation_instances: Vec::new(),
            quadtree: None,
            visible_chunks: Vec::new(),
            dirty: true,
            vegetation_dirty: true,
            streamer: None,
        }
    }

    /// Creates a terrain component with a flat heightmap.
    pub fn new_flat(
        width: u32,
        height: u32,
        terrain_size: f32,
        initial_height: f32,
    ) -> Self {
        let hm = Heightmap::new_flat(width, height, initial_height)
            .expect("Valid heightmap dimensions");
        Self::new(hm, terrain_size, TerrainMaterial::default())
    }

    /// Creates a terrain component from a procedurally generated heightmap.
    pub fn new_procedural(
        size: u32,
        terrain_size: f32,
        roughness: f32,
        seed: u64,
    ) -> Self {
        let hm = Heightmap::generate_procedural(size, roughness, seed)
            .expect("Valid procedural parameters");
        Self::new(hm, terrain_size, TerrainMaterial::default())
    }

    /// Returns the height at the given world position.
    ///
    /// Transforms the world position into heightmap space and samples.
    pub fn height_at_world(&self, world_pos: Vec3, transform: &Transform) -> f32 {
        let hm = self.heightmap.read();
        let local = world_pos - transform.position;

        // Convert world space to heightmap coordinates
        let hm_x = local.x / self.terrain_size * (hm.width() - 1) as f32;
        let hm_z = local.z / self.terrain_size * (hm.height() - 1) as f32;

        let height = hm.sample(hm_x, hm_z);
        height * self.mesh_settings.height_scale + transform.position.y
    }

    /// Returns the surface normal at the given world position.
    pub fn normal_at_world(&self, world_pos: Vec3, transform: &Transform) -> Vec3 {
        let hm = self.heightmap.read();
        let local = world_pos - transform.position;

        let hm_x = local.x / self.terrain_size * (hm.width() - 1) as f32;
        let hm_z = local.z / self.terrain_size * (hm.height() - 1) as f32;

        hm.normal_at_scaled(
            hm_x,
            hm_z,
            self.mesh_settings.cell_size,
            self.mesh_settings.height_scale,
        )
    }

    /// Builds or rebuilds the LOD quadtree.
    pub fn build_quadtree(&mut self) {
        let hm = self.heightmap.read();
        let max_depth = self.lod_settings.max_lod.min(10) + 1;
        let qt = TerrainQuadtree::build(
            &hm,
            self.terrain_size,
            max_depth,
            self.lod_settings.clone(),
        );
        drop(hm);
        self.quadtree = Some(qt);
        self.dirty = false;
    }

    /// Marks the terrain as modified, requiring mesh and LOD rebuilds.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Marks vegetation as needing re-scatter.
    pub fn mark_vegetation_dirty(&mut self) {
        self.vegetation_dirty = true;
    }

    /// Returns the world-space AABB of the terrain.
    pub fn world_aabb(&self, transform: &Transform) -> AABB {
        let hm = self.heightmap.read();
        let min = transform.position + Vec3::new(0.0, hm.min_height() * self.mesh_settings.height_scale, 0.0);
        let max = transform.position
            + Vec3::new(
                self.terrain_size,
                hm.max_height() * self.mesh_settings.height_scale,
                self.terrain_size,
            );
        AABB::new(min, max)
    }

    /// Enables terrain streaming with the given chunk radius.
    pub fn enable_streaming(&mut self, load_radius: u32, unload_radius: u32) {
        let chunk_size = self.lod_settings.chunk_size;
        self.streamer = Some(TerrainStreamer::new(chunk_size, load_radius, unload_radius));
    }
}

// ---------------------------------------------------------------------------
// TerrainSystem
// ---------------------------------------------------------------------------

/// System that updates terrain LOD, streaming, and vegetation each frame.
///
/// This system reads camera data from the world and updates each terrain
/// entity's visible chunks and vegetation culling.
pub struct TerrainSystem {
    /// Camera position cache for the current frame.
    camera_pos: Vec3,
    /// Camera frustum for the current frame.
    frustum: Option<Frustum>,
    /// Viewport height for screen-space error metric.
    viewport_height: f32,
    /// Vertical field of view in radians.
    fov_y: f32,
    /// Whether to use screen-space error metric (vs fixed distance).
    use_sse: bool,
}

impl TerrainSystem {
    /// Creates a new terrain system.
    pub fn new() -> Self {
        Self {
            camera_pos: Vec3::ZERO,
            frustum: None,
            viewport_height: 1080.0,
            fov_y: std::f32::consts::FRAC_PI_4,
            use_sse: false,
        }
    }

    /// Sets the camera parameters for LOD selection.
    pub fn set_camera(
        &mut self,
        position: Vec3,
        view_projection: &Mat4,
        viewport_height: f32,
        fov_y: f32,
    ) {
        self.camera_pos = position;
        self.frustum = Some(Frustum::from_view_projection(view_projection));
        self.viewport_height = viewport_height;
        self.fov_y = fov_y;
    }

    /// Enables screen-space error metric for LOD selection.
    pub fn enable_sse(&mut self, enable: bool) {
        self.use_sse = enable;
    }

    /// Updates a single terrain component.
    ///
    /// Performs LOD selection, streaming updates, and vegetation culling.
    #[profiling::function]
    pub fn update_terrain(&self, terrain: &mut TerrainComponent) {
        // Rebuild quadtree if terrain data changed
        if terrain.dirty || terrain.quadtree.is_none() {
            terrain.build_quadtree();
        }

        // LOD selection
        if let (Some(qt), Some(frustum)) = (&terrain.quadtree, &self.frustum) {
            terrain.visible_chunks = if self.use_sse {
                qt.select_visible_nodes_sse(
                    self.camera_pos,
                    frustum,
                    self.fov_y,
                    self.viewport_height,
                )
            } else {
                qt.select_visible_nodes(self.camera_pos, frustum)
            };
        }

        // Streaming update
        if let Some(streamer) = &mut terrain.streamer {
            let chunks_x = (terrain.terrain_size / terrain.lod_settings.chunk_size).ceil() as u32;
            let chunks_z = chunks_x;
            streamer.update(self.camera_pos, chunks_x, chunks_z);
        }

        // Vegetation re-scatter if needed
        if terrain.vegetation_dirty && !terrain.vegetation_layers.is_empty() {
            let hm = terrain.heightmap.read();
            let splatmap = terrain.material.splatmaps.first();
            terrain.vegetation_instances = crate::vegetation::scatter_vegetation(
                &hm,
                splatmap,
                None,
                &terrain.vegetation_layers,
                terrain.terrain_size,
                terrain.mesh_settings.height_scale,
                42, // Fixed seed for consistency
            );
            terrain.vegetation_dirty = false;
        }
    }
}

impl Default for TerrainSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl genovo_ecs::System for TerrainSystem {
    fn run(&mut self, _world: &mut World) {
        // In a full integration, this would iterate entities with
        // TerrainComponent and Transform, calling update_terrain for each.
        // The ECS query API is used to find matching entities.
        //
        // For now, terrain updates are driven externally via
        // `update_terrain()` to keep the integration simple while the
        // ECS query API matures.
        log::trace!("TerrainSystem::run (LOD selection)");
    }
}

// ---------------------------------------------------------------------------
// TerrainCollider
// ---------------------------------------------------------------------------

/// Physics integration component for terrain collision.
///
/// Provides a heightfield collision shape that physics engines can use
/// for collision detection and raycasting against the terrain surface.
pub struct TerrainCollider {
    /// Reference to the terrain heightmap (shared with TerrainComponent).
    heightmap: Arc<RwLock<Heightmap>>,
    /// Terrain size in world units.
    terrain_size: f32,
    /// Vertical scale applied to heightmap values.
    height_scale: f32,
    /// Collision margin (skin thickness).
    margin: f32,
    /// Whether the collider is enabled.
    enabled: bool,
}

impl Component for TerrainCollider {}

impl TerrainCollider {
    /// Creates a new terrain collider.
    pub fn new(
        heightmap: Arc<RwLock<Heightmap>>,
        terrain_size: f32,
        height_scale: f32,
    ) -> Self {
        Self {
            heightmap,
            terrain_size,
            height_scale,
            margin: 0.01,
            enabled: true,
        }
    }

    /// Creates a terrain collider from a TerrainComponent.
    pub fn from_terrain(terrain: &TerrainComponent) -> Self {
        Self {
            heightmap: terrain.heightmap.clone(),
            terrain_size: terrain.terrain_size,
            height_scale: terrain.mesh_settings.height_scale,
            margin: 0.01,
            enabled: true,
        }
    }

    /// Returns the height at a world XZ position (Y is the height).
    pub fn height_at(&self, x: f32, z: f32) -> Option<f32> {
        if x < 0.0 || z < 0.0 || x > self.terrain_size || z > self.terrain_size {
            return None;
        }

        let hm = self.heightmap.read();
        let hm_x = x / self.terrain_size * (hm.width() - 1) as f32;
        let hm_z = z / self.terrain_size * (hm.height() - 1) as f32;
        Some(hm.sample(hm_x, hm_z) * self.height_scale)
    }

    /// Returns the surface normal at a world XZ position.
    pub fn normal_at(&self, x: f32, z: f32) -> Option<Vec3> {
        if x < 0.0 || z < 0.0 || x > self.terrain_size || z > self.terrain_size {
            return None;
        }

        let hm = self.heightmap.read();
        let hm_x = x / self.terrain_size * (hm.width() - 1) as f32;
        let hm_z = z / self.terrain_size * (hm.height() - 1) as f32;
        let cell_size = self.terrain_size / (hm.width() - 1) as f32;
        Some(hm.normal_at_scaled(hm_x, hm_z, cell_size, self.height_scale))
    }

    /// Performs a vertical raycast (from above) at position (x, z).
    ///
    /// Returns `Some(y)` with the terrain height at that position, or
    /// `None` if out of bounds.
    pub fn raycast_vertical(&self, x: f32, z: f32) -> Option<f32> {
        self.height_at(x, z)
    }

    /// Tests whether a sphere intersects the terrain.
    ///
    /// Returns `Some(penetration_depth)` if the sphere penetrates, or
    /// `None` if no intersection.
    pub fn sphere_test(&self, center: Vec3, radius: f32) -> Option<f32> {
        if !self.enabled {
            return None;
        }

        let terrain_height = self.height_at(center.x, center.z)?;
        let bottom = center.y - radius;

        if bottom < terrain_height + self.margin {
            Some(terrain_height + self.margin - bottom)
        } else {
            None
        }
    }

    /// Tests whether an AABB intersects the terrain.
    ///
    /// Samples the terrain at the four corners and center of the AABB's
    /// XZ projection and returns `true` if any sample is above the AABB
    /// bottom.
    pub fn aabb_test(&self, aabb: &AABB) -> bool {
        if !self.enabled {
            return false;
        }

        let bottom = aabb.min.y;
        let test_points = [
            (aabb.min.x, aabb.min.z),
            (aabb.max.x, aabb.min.z),
            (aabb.min.x, aabb.max.z),
            (aabb.max.x, aabb.max.z),
            (
                (aabb.min.x + aabb.max.x) * 0.5,
                (aabb.min.z + aabb.max.z) * 0.5,
            ),
        ];

        for (x, z) in test_points {
            if let Some(h) = self.height_at(x, z) {
                if h > bottom {
                    return true;
                }
            }
        }

        false
    }

    /// Sets the collision margin.
    pub fn set_margin(&mut self, margin: f32) {
        self.margin = margin.max(0.0);
    }

    /// Enables or disables the collider.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Returns whether the collider is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Returns the terrain size.
    pub fn terrain_size(&self) -> f32 {
        self.terrain_size
    }

    /// Returns the height scale.
    pub fn height_scale(&self) -> f32 {
        self.height_scale
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrain_component_creation() {
        let tc = TerrainComponent::new_flat(65, 65, 1024.0, 0.5);
        assert_eq!(tc.terrain_size, 1024.0);
        assert!(tc.dirty);
    }

    #[test]
    fn terrain_component_procedural() {
        let tc = TerrainComponent::new_procedural(65, 512.0, 0.5, 42);
        assert_eq!(tc.terrain_size, 512.0);
    }

    #[test]
    fn terrain_height_at_world() {
        let tc = TerrainComponent::new_flat(17, 17, 100.0, 0.5);
        let transform = Transform::from_position(Vec3::ZERO);
        let h = tc.height_at_world(Vec3::new(50.0, 0.0, 50.0), &transform);
        let expected = 0.5 * tc.mesh_settings.height_scale;
        assert!((h - expected).abs() < 0.1);
    }

    #[test]
    fn terrain_collider_height() {
        let hm = Heightmap::new_flat(17, 17, 0.5).unwrap();
        let collider = TerrainCollider::new(
            Arc::new(RwLock::new(hm)),
            100.0,
            50.0,
        );
        let h = collider.height_at(50.0, 50.0).unwrap();
        assert!((h - 25.0).abs() < 0.1); // 0.5 * 50.0 = 25.0
    }

    #[test]
    fn terrain_collider_sphere_test() {
        let hm = Heightmap::new_flat(17, 17, 0.0).unwrap();
        let collider = TerrainCollider::new(
            Arc::new(RwLock::new(hm)),
            100.0,
            50.0,
        );

        // Sphere above terrain: no intersection
        let result = collider.sphere_test(Vec3::new(50.0, 10.0, 50.0), 1.0);
        assert!(result.is_none());

        // Sphere at terrain: intersection
        let result = collider.sphere_test(Vec3::new(50.0, 0.5, 50.0), 1.0);
        assert!(result.is_some());
    }

    #[test]
    fn terrain_collider_aabb_test() {
        let hm = Heightmap::new_flat(17, 17, 0.5).unwrap();
        let collider = TerrainCollider::new(
            Arc::new(RwLock::new(hm)),
            100.0,
            50.0,
        );

        // AABB below terrain height (25.0): should intersect
        let aabb = AABB::new(Vec3::new(40.0, 20.0, 40.0), Vec3::new(60.0, 24.0, 60.0));
        assert!(collider.aabb_test(&aabb));

        // AABB above terrain: no intersection
        let aabb = AABB::new(Vec3::new(40.0, 30.0, 40.0), Vec3::new(60.0, 40.0, 60.0));
        assert!(!collider.aabb_test(&aabb));
    }

    #[test]
    fn terrain_system_creation() {
        let sys = TerrainSystem::new();
        assert!(!sys.use_sse);
    }

    #[test]
    fn terrain_world_aabb() {
        let tc = TerrainComponent::new_flat(17, 17, 100.0, 0.5);
        let transform = Transform::from_position(Vec3::new(10.0, 0.0, 10.0));
        let aabb = tc.world_aabb(&transform);
        assert!(aabb.min.x >= 10.0);
        assert!(aabb.max.x <= 110.0);
    }

    #[test]
    fn terrain_build_quadtree() {
        let mut tc = TerrainComponent::new_procedural(65, 512.0, 0.5, 42);
        tc.build_quadtree();
        assert!(tc.quadtree.is_some());
        assert!(!tc.dirty);
    }

    #[test]
    fn terrain_streaming_enable() {
        let mut tc = TerrainComponent::new_flat(65, 65, 1024.0, 0.0);
        tc.enable_streaming(5, 7);
        assert!(tc.streamer.is_some());
    }
}
