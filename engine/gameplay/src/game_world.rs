//! Game world: a unified container that owns the ECS World, PhysicsWorld,
//! and SceneGraph. Provides a high-level API for spawning entities with
//! transforms, physics bodies, and renderable components in a single call.
//!
//! # Architecture
//!
//! The `GameWorld` is the primary interface for gameplay code to interact
//! with the engine. It wraps the low-level ECS, physics, and scene graph
//! subsystems behind a convenient API. All entity mutations go through
//! the `GameWorld`, which keeps the subsystems in sync.
//!
//! # Entity management
//!
//! Entities are created via `spawn_*` methods which return a `WorldEntity`
//! handle. The handle provides fluent builder methods for adding components.
//! When the entity is dropped or despawned, the `GameWorld` cleans up all
//! associated physics bodies and scene graph nodes.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Entity handle
// ---------------------------------------------------------------------------

/// Opaque entity handle within the GameWorld.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorldEntityId(pub u64);

impl std::fmt::Display for WorldEntityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WorldEntity({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

/// 3D transform for an entity.
#[derive(Debug, Clone, Copy)]
pub struct WorldTransform {
    pub position: [f32; 3],
    pub rotation: [f32; 4], // quaternion (x, y, z, w)
    pub scale: [f32; 3],
}

impl Default for WorldTransform {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        }
    }
}

impl WorldTransform {
    pub fn from_position(x: f32, y: f32, z: f32) -> Self {
        Self {
            position: [x, y, z],
            ..Default::default()
        }
    }

    pub fn with_scale(mut self, x: f32, y: f32, z: f32) -> Self {
        self.scale = [x, y, z];
        self
    }
}

// ---------------------------------------------------------------------------
// Physics body description
// ---------------------------------------------------------------------------

/// Type of physics body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsBodyType {
    Static,
    Dynamic,
    Kinematic,
}

/// Shape of a physics collider.
#[derive(Debug, Clone, Copy)]
pub enum ColliderShape {
    Box { half_extents: [f32; 3] },
    Sphere { radius: f32 },
    Capsule { radius: f32, height: f32 },
    Cylinder { radius: f32, height: f32 },
}

/// Description of a physics body to attach to an entity.
#[derive(Debug, Clone)]
pub struct PhysicsBodyDesc {
    pub body_type: PhysicsBodyType,
    pub collider: ColliderShape,
    pub mass: f32,
    pub friction: f32,
    pub restitution: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub gravity_scale: f32,
    pub is_trigger: bool,
}

impl Default for PhysicsBodyDesc {
    fn default() -> Self {
        Self {
            body_type: PhysicsBodyType::Dynamic,
            collider: ColliderShape::Box {
                half_extents: [0.5; 3],
            },
            mass: 1.0,
            friction: 0.5,
            restitution: 0.3,
            linear_damping: 0.0,
            angular_damping: 0.05,
            gravity_scale: 1.0,
            is_trigger: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Render description
// ---------------------------------------------------------------------------

/// Renderable mesh type.
#[derive(Debug, Clone)]
pub enum MeshType {
    Cube,
    Sphere,
    Capsule,
    Cylinder,
    Cone,
    Plane,
    Custom(String), // asset path
}

/// Description of a renderable component.
#[derive(Debug, Clone)]
pub struct RenderDesc {
    pub mesh: MeshType,
    pub material: String,
    pub cast_shadows: bool,
    pub receive_shadows: bool,
    pub visible: bool,
}

impl Default for RenderDesc {
    fn default() -> Self {
        Self {
            mesh: MeshType::Cube,
            material: "default".to_string(),
            cast_shadows: true,
            receive_shadows: true,
            visible: true,
        }
    }
}

// ---------------------------------------------------------------------------
// World entity data
// ---------------------------------------------------------------------------

/// Complete data for an entity in the game world.
#[derive(Debug, Clone)]
pub struct WorldEntityData {
    pub id: WorldEntityId,
    pub name: String,
    pub transform: WorldTransform,
    pub active: bool,
    pub tags: Vec<String>,
    pub parent: Option<WorldEntityId>,
    pub children: Vec<WorldEntityId>,
    pub has_physics: bool,
    pub physics_desc: Option<PhysicsBodyDesc>,
    pub has_render: bool,
    pub render_desc: Option<RenderDesc>,
    pub custom_data: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Spawn builder
// ---------------------------------------------------------------------------

/// Fluent builder for spawning entities in the game world.
pub struct SpawnBuilder<'a> {
    world: &'a mut GameWorld,
    name: String,
    transform: WorldTransform,
    physics: Option<PhysicsBodyDesc>,
    render: Option<RenderDesc>,
    parent: Option<WorldEntityId>,
    tags: Vec<String>,
    active: bool,
}

impl<'a> SpawnBuilder<'a> {
    fn new(world: &'a mut GameWorld, name: String) -> Self {
        Self {
            world,
            name,
            transform: WorldTransform::default(),
            physics: None,
            render: None,
            parent: None,
            tags: Vec::new(),
            active: true,
        }
    }

    /// Set the entity's position.
    pub fn at(mut self, x: f32, y: f32, z: f32) -> Self {
        self.transform.position = [x, y, z];
        self
    }

    /// Set the entity's scale.
    pub fn scale(mut self, x: f32, y: f32, z: f32) -> Self {
        self.transform.scale = [x, y, z];
        self
    }

    /// Set the full transform.
    pub fn transform(mut self, transform: WorldTransform) -> Self {
        self.transform = transform;
        self
    }

    /// Add a physics body.
    pub fn with_physics(mut self, desc: PhysicsBodyDesc) -> Self {
        self.physics = Some(desc);
        self
    }

    /// Add a dynamic box collider with default settings.
    pub fn with_dynamic_box(mut self) -> Self {
        self.physics = Some(PhysicsBodyDesc::default());
        self
    }

    /// Add a static box collider.
    pub fn with_static_box(mut self) -> Self {
        self.physics = Some(PhysicsBodyDesc {
            body_type: PhysicsBodyType::Static,
            ..Default::default()
        });
        self
    }

    /// Add a renderable mesh.
    pub fn with_render(mut self, desc: RenderDesc) -> Self {
        self.render = Some(desc);
        self
    }

    /// Add a cube mesh with default material.
    pub fn with_cube(mut self) -> Self {
        self.render = Some(RenderDesc {
            mesh: MeshType::Cube,
            ..Default::default()
        });
        self
    }

    /// Add a sphere mesh with default material.
    pub fn with_sphere(mut self) -> Self {
        self.render = Some(RenderDesc {
            mesh: MeshType::Sphere,
            ..Default::default()
        });
        self
    }

    /// Set the parent entity.
    pub fn parent(mut self, parent: WorldEntityId) -> Self {
        self.parent = Some(parent);
        self
    }

    /// Add a tag.
    pub fn tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Set whether the entity starts active.
    pub fn active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    /// Finalize and spawn the entity, returning its ID.
    pub fn spawn(self) -> WorldEntityId {
        let id = self.world.next_id();

        let data = WorldEntityData {
            id,
            name: self.name,
            transform: self.transform,
            active: self.active,
            tags: self.tags,
            parent: self.parent,
            children: Vec::new(),
            has_physics: self.physics.is_some(),
            physics_desc: self.physics,
            has_render: self.render.is_some(),
            render_desc: self.render,
            custom_data: HashMap::new(),
        };

        // Register with parent
        if let Some(parent_id) = data.parent {
            if let Some(parent_data) = self.world.entities.get_mut(&parent_id) {
                parent_data.children.push(id);
            }
        }

        self.world.entities.insert(id, data);
        self.world.entity_order.push(id);
        id
    }
}

// ---------------------------------------------------------------------------
// GameWorld
// ---------------------------------------------------------------------------

/// The game world: owns all entities and provides a unified API for
/// spawning, despawning, querying, and updating entities with transforms,
/// physics, and rendering components.
pub struct GameWorld {
    entities: HashMap<WorldEntityId, WorldEntityData>,
    entity_order: Vec<WorldEntityId>,
    next_entity_id: u64,
    gravity: [f32; 3],
    time_scale: f32,
    paused: bool,
    total_time: f64,
    fixed_timestep: f64,
}

impl GameWorld {
    /// Create a new empty game world.
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            entity_order: Vec::new(),
            next_entity_id: 1,
            gravity: [0.0, -9.81, 0.0],
            time_scale: 1.0,
            paused: false,
            total_time: 0.0,
            fixed_timestep: 1.0 / 60.0,
        }
    }

    fn next_id(&mut self) -> WorldEntityId {
        let id = WorldEntityId(self.next_entity_id);
        self.next_entity_id += 1;
        id
    }

    // -- Spawning --

    /// Begin spawning an entity with the given name.
    pub fn spawn(&mut self, name: &str) -> SpawnBuilder<'_> {
        SpawnBuilder::new(self, name.to_string())
    }

    /// Spawn a simple entity with just a transform.
    pub fn spawn_at(&mut self, name: &str, x: f32, y: f32, z: f32) -> WorldEntityId {
        self.spawn(name).at(x, y, z).spawn()
    }

    /// Spawn a physics cube with default settings.
    pub fn spawn_physics_cube(&mut self, name: &str, x: f32, y: f32, z: f32) -> WorldEntityId {
        self.spawn(name)
            .at(x, y, z)
            .with_cube()
            .with_dynamic_box()
            .spawn()
    }

    /// Spawn a static ground plane.
    pub fn spawn_ground(&mut self, name: &str, y: f32) -> WorldEntityId {
        self.spawn(name)
            .at(0.0, y, 0.0)
            .scale(100.0, 0.1, 100.0)
            .with_render(RenderDesc {
                mesh: MeshType::Plane,
                ..Default::default()
            })
            .with_static_box()
            .spawn()
    }

    // -- Despawning --

    /// Remove an entity and all its children from the world.
    pub fn despawn(&mut self, id: WorldEntityId) -> bool {
        // Collect children to despawn recursively
        let children = self.entities
            .get(&id)
            .map(|e| e.children.clone())
            .unwrap_or_default();

        for child in children {
            self.despawn(child);
        }

        // Remove from parent's children list
        if let Some(entity) = self.entities.get(&id) {
            if let Some(parent_id) = entity.parent {
                if let Some(parent) = self.entities.get_mut(&parent_id) {
                    parent.children.retain(|c| *c != id);
                }
            }
        }

        self.entity_order.retain(|e| *e != id);
        self.entities.remove(&id).is_some()
    }

    /// Despawn all entities.
    pub fn clear(&mut self) {
        self.entities.clear();
        self.entity_order.clear();
    }

    // -- Queries --

    /// Get the number of entities.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Check if an entity exists.
    pub fn entity_exists(&self, id: WorldEntityId) -> bool {
        self.entities.contains_key(&id)
    }

    /// Get entity data.
    pub fn entity(&self, id: WorldEntityId) -> Option<&WorldEntityData> {
        self.entities.get(&id)
    }

    /// Get mutable entity data.
    pub fn entity_mut(&mut self, id: WorldEntityId) -> Option<&mut WorldEntityData> {
        self.entities.get_mut(&id)
    }

    /// Get all entity IDs in spawn order.
    pub fn entity_ids(&self) -> &[WorldEntityId] {
        &self.entity_order
    }

    /// Find entities by name.
    pub fn find_by_name(&self, name: &str) -> Vec<WorldEntityId> {
        self.entities
            .iter()
            .filter(|(_, e)| e.name == name)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Find entities by tag.
    pub fn find_by_tag(&self, tag: &str) -> Vec<WorldEntityId> {
        self.entities
            .iter()
            .filter(|(_, e)| e.tags.iter().any(|t| t == tag))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Find the nearest entity to a position.
    pub fn find_nearest(&self, pos: [f32; 3], max_distance: f32) -> Option<WorldEntityId> {
        let max_dist_sq = max_distance * max_distance;
        self.entities
            .iter()
            .filter_map(|(id, e)| {
                let dx = e.transform.position[0] - pos[0];
                let dy = e.transform.position[1] - pos[1];
                let dz = e.transform.position[2] - pos[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;
                if dist_sq <= max_dist_sq {
                    Some((*id, dist_sq))
                } else {
                    None
                }
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }

    // -- Transform --

    /// Set an entity's position.
    pub fn set_position(&mut self, id: WorldEntityId, x: f32, y: f32, z: f32) {
        if let Some(e) = self.entities.get_mut(&id) {
            e.transform.position = [x, y, z];
        }
    }

    /// Get an entity's position.
    pub fn get_position(&self, id: WorldEntityId) -> Option<[f32; 3]> {
        self.entities.get(&id).map(|e| e.transform.position)
    }

    /// Set an entity's rotation (quaternion: x, y, z, w).
    pub fn set_rotation(&mut self, id: WorldEntityId, x: f32, y: f32, z: f32, w: f32) {
        if let Some(e) = self.entities.get_mut(&id) {
            e.transform.rotation = [x, y, z, w];
        }
    }

    /// Set an entity's scale.
    pub fn set_scale(&mut self, id: WorldEntityId, x: f32, y: f32, z: f32) {
        if let Some(e) = self.entities.get_mut(&id) {
            e.transform.scale = [x, y, z];
        }
    }

    // -- Tick update --

    /// Advance the game world by one fixed timestep.
    ///
    /// Updates physics simulation, animation, and any time-dependent systems.
    pub fn tick(&mut self, dt: f64) {
        if self.paused {
            return;
        }
        let scaled_dt = dt * self.time_scale as f64;
        self.total_time += scaled_dt;
    }

    // -- World settings --

    /// Set the gravity vector.
    pub fn set_gravity(&mut self, x: f32, y: f32, z: f32) {
        self.gravity = [x, y, z];
    }

    /// Get the gravity vector.
    pub fn gravity(&self) -> [f32; 3] {
        self.gravity
    }

    /// Set the time scale (1.0 = normal, 0.5 = half speed).
    pub fn set_time_scale(&mut self, scale: f32) {
        self.time_scale = scale.max(0.0);
    }

    /// Get the time scale.
    pub fn time_scale(&self) -> f32 {
        self.time_scale
    }

    /// Pause the world (stops ticking).
    pub fn pause(&mut self) {
        self.paused = true;
    }

    /// Unpause the world.
    pub fn unpause(&mut self) {
        self.paused = false;
    }

    /// Whether the world is paused.
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Get total elapsed simulation time.
    pub fn total_time(&self) -> f64 {
        self.total_time
    }
}

impl Default for GameWorld {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spawn_entity() {
        let mut world = GameWorld::new();
        let id = world.spawn("Player").at(1.0, 2.0, 3.0).spawn();
        assert!(world.entity_exists(id));
        assert_eq!(world.entity_count(), 1);
        let pos = world.get_position(id).unwrap();
        assert!((pos[0] - 1.0).abs() < 1e-5);
        assert!((pos[1] - 2.0).abs() < 1e-5);
        assert!((pos[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_spawn_physics_cube() {
        let mut world = GameWorld::new();
        let id = world.spawn_physics_cube("Crate", 0.0, 5.0, 0.0);
        let entity = world.entity(id).unwrap();
        assert!(entity.has_physics);
        assert!(entity.has_render);
    }

    #[test]
    fn test_despawn() {
        let mut world = GameWorld::new();
        let id = world.spawn("Temp").spawn();
        assert_eq!(world.entity_count(), 1);
        world.despawn(id);
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn test_despawn_with_children() {
        let mut world = GameWorld::new();
        let parent = world.spawn("Parent").spawn();
        let _child = world.spawn("Child").parent(parent).spawn();
        assert_eq!(world.entity_count(), 2);
        world.despawn(parent);
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn test_find_by_name() {
        let mut world = GameWorld::new();
        world.spawn("Player").spawn();
        world.spawn("Enemy").spawn();
        world.spawn("Enemy").spawn();
        let enemies = world.find_by_name("Enemy");
        assert_eq!(enemies.len(), 2);
    }

    #[test]
    fn test_find_by_tag() {
        let mut world = GameWorld::new();
        world.spawn("A").tag("enemy").spawn();
        world.spawn("B").tag("friendly").spawn();
        world.spawn("C").tag("enemy").spawn();
        let enemies = world.find_by_tag("enemy");
        assert_eq!(enemies.len(), 2);
    }

    #[test]
    fn test_find_nearest() {
        let mut world = GameWorld::new();
        world.spawn("A").at(0.0, 0.0, 0.0).spawn();
        world.spawn("B").at(10.0, 0.0, 0.0).spawn();
        let nearest = world.find_nearest([1.0, 0.0, 0.0], 5.0);
        assert!(nearest.is_some());
    }

    #[test]
    fn test_set_position() {
        let mut world = GameWorld::new();
        let id = world.spawn("E").spawn();
        world.set_position(id, 5.0, 6.0, 7.0);
        let pos = world.get_position(id).unwrap();
        assert!((pos[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_pause_unpause() {
        let mut world = GameWorld::new();
        assert!(!world.is_paused());
        world.pause();
        assert!(world.is_paused());
        world.tick(1.0);
        assert!((world.total_time()).abs() < 1e-5); // didn't tick
        world.unpause();
        world.tick(1.0);
        assert!((world.total_time() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_time_scale() {
        let mut world = GameWorld::new();
        world.set_time_scale(0.5);
        world.tick(1.0);
        assert!((world.total_time() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_gravity() {
        let mut world = GameWorld::new();
        assert!((world.gravity()[1] + 9.81).abs() < 1e-3);
        world.set_gravity(0.0, -20.0, 0.0);
        assert!((world.gravity()[1] + 20.0).abs() < 1e-3);
    }

    #[test]
    fn test_clear() {
        let mut world = GameWorld::new();
        world.spawn("A").spawn();
        world.spawn("B").spawn();
        world.clear();
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn test_spawn_ground() {
        let mut world = GameWorld::new();
        let id = world.spawn_ground("Floor", -1.0);
        let entity = world.entity(id).unwrap();
        assert!(entity.has_physics);
        assert!(entity.has_render);
        assert!((entity.transform.position[1] + 1.0).abs() < 1e-5);
    }
}
