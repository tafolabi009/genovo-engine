//! Component bundles for spawning entities with pre-defined component sets.
//!
//! A [`Bundle`] is a group of components that can be inserted together into
//! an entity. This avoids the boilerplate of calling `add_component` for each
//! component individually, and ensures that common component groupings are
//! consistent across the codebase.
//!
//! # Usage
//!
//! ```ignore
//! // Spawn an entity with a named bundle:
//! let entity = world.spawn_bundle(SpriteBundle {
//!     transform: TransformData::from_position(Vec3::new(100.0, 200.0, 0.0)),
//!     sprite: SpriteData { width: 32.0, height: 32.0, ..Default::default() },
//!     material: MaterialRef::default(),
//! });
//!
//! // Or use tuples as bundles:
//! let entity = world.spawn_bundle((
//!     Position { x: 0.0, y: 0.0 },
//!     Velocity { dx: 1.0, dy: 0.0 },
//!     Health(100.0),
//! ));
//! ```
//!
//! # Tuple Bundles
//!
//! Tuples of components up to 12 elements automatically implement `Bundle`.
//! This is the most flexible approach for ad-hoc component groupings.
//!
//! # Named Bundles
//!
//! For commonly used groupings, named bundles provide documentation and
//! type safety. The engine provides several built-in bundles:
//!
//! - [`SpriteBundle`] — 2D sprite entity
//! - [`MeshBundle`] — 3D mesh entity
//! - [`LightBundle`] — light source entity
//! - [`CameraBundle`] — camera entity
//! - [`PhysicsBundle`] — physics-enabled entity

use crate::component::Component;
use crate::entity::Entity;
use crate::world::World;

// ---------------------------------------------------------------------------
// Bundle trait
// ---------------------------------------------------------------------------

/// Trait for a group of components that can be inserted together into an
/// entity.
///
/// Implementing this trait allows a struct to be used with
/// [`World::spawn_bundle`] and [`World::insert_bundle`], which insert all
/// the bundle's components in one call.
pub trait Bundle: Send + Sync + 'static {
    /// Insert all components from this bundle into the given entity.
    fn insert_into(self, world: &mut World, entity: Entity);

    /// Return the number of components in this bundle.
    fn component_count() -> usize;

    /// Return a human-readable description of the bundle contents.
    fn describe() -> &'static str {
        "Bundle"
    }
}

// ---------------------------------------------------------------------------
// World extensions for bundles
// ---------------------------------------------------------------------------

impl World {
    /// Spawn a new entity and insert all components from the bundle.
    ///
    /// ```ignore
    /// let entity = world.spawn_bundle((
    ///     Position { x: 0.0, y: 0.0 },
    ///     Velocity { dx: 1.0, dy: 0.0 },
    /// ));
    /// ```
    pub fn spawn_bundle<B: Bundle>(&mut self, bundle: B) -> Entity {
        let entity = self.spawn_empty();
        bundle.insert_into(self, entity);
        entity
    }

    /// Insert all components from a bundle into an existing entity.
    ///
    /// If the entity already has components of the same type, they are
    /// overwritten.
    pub fn insert_bundle<B: Bundle>(&mut self, entity: Entity, bundle: B) {
        if !self.is_alive(entity) {
            return;
        }
        bundle.insert_into(self, entity);
    }

    /// Spawn multiple entities from an iterator of bundles.
    pub fn spawn_bundle_batch<B: Bundle>(
        &mut self,
        bundles: impl IntoIterator<Item = B>,
    ) -> Vec<Entity> {
        bundles
            .into_iter()
            .map(|b| self.spawn_bundle(b))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Bundle impl for single component
// ---------------------------------------------------------------------------

impl<A: Component> Bundle for (A,) {
    fn insert_into(self, world: &mut World, entity: Entity) {
        world.add_component(entity, self.0);
    }

    fn component_count() -> usize {
        1
    }

    fn describe() -> &'static str {
        "Bundle(1)"
    }
}

// ---------------------------------------------------------------------------
// Bundle impl for tuples (arity 2..12)
// ---------------------------------------------------------------------------

macro_rules! impl_bundle_tuple {
    ($count:expr, $($idx:tt: $T:ident),+) => {
        impl<$($T: Component),+> Bundle for ($($T,)+) {
            fn insert_into(self, world: &mut World, entity: Entity) {
                $(world.add_component(entity, self.$idx);)+
            }

            fn component_count() -> usize {
                $count
            }

            fn describe() -> &'static str {
                concat!("Bundle(", stringify!($count), ")")
            }
        }
    };
}

impl_bundle_tuple!(2, 0: A, 1: B);
impl_bundle_tuple!(3, 0: A, 1: B, 2: C);
impl_bundle_tuple!(4, 0: A, 1: B, 2: C, 3: D);
impl_bundle_tuple!(5, 0: A, 1: B, 2: C, 3: D, 4: E);
impl_bundle_tuple!(6, 0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_bundle_tuple!(7, 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_bundle_tuple!(8, 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_bundle_tuple!(9, 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_bundle_tuple!(10, 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_bundle_tuple!(11, 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_bundle_tuple!(12, 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);

// ---------------------------------------------------------------------------
// Named bundle component types
// ---------------------------------------------------------------------------

// These are lightweight data-only structs used by the named bundles. In the
// full engine they would reference actual render/physics types; here we
// define self-contained versions for the ECS layer.

/// Transform data used in bundles. Represents position, rotation, and scale.
#[derive(Debug, Clone)]
pub struct TransformData {
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

impl TransformData {
    /// Identity transform.
    pub const IDENTITY: Self = Self {
        position: [0.0, 0.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    };

    /// Create from position only.
    pub fn from_position(pos: [f32; 3]) -> Self {
        Self {
            position: pos,
            ..Self::IDENTITY
        }
    }

    /// Create from position and scale.
    pub fn from_position_scale(pos: [f32; 3], scale: [f32; 3]) -> Self {
        Self {
            position: pos,
            scale,
            ..Self::IDENTITY
        }
    }

    /// Create with all fields.
    pub fn new(position: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> Self {
        Self {
            position,
            rotation,
            scale,
        }
    }
}

impl Default for TransformData {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Component for TransformData {}

/// Sprite data for 2D rendering.
#[derive(Debug, Clone)]
pub struct SpriteData {
    pub width: f32,
    pub height: f32,
    pub color: [f32; 4],
    pub uv_rect: [f32; 4],
    pub flip_x: bool,
    pub flip_y: bool,
    pub anchor: [f32; 2],
    pub sort_order: i32,
}

impl Default for SpriteData {
    fn default() -> Self {
        Self {
            width: 1.0,
            height: 1.0,
            color: [1.0, 1.0, 1.0, 1.0],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            flip_x: false,
            flip_y: false,
            anchor: [0.5, 0.5],
            sort_order: 0,
        }
    }
}

impl Component for SpriteData {}

/// Material reference for rendering.
#[derive(Debug, Clone)]
pub struct MaterialRef {
    pub material_id: u64,
    pub shader_variant: u32,
    pub render_queue: i32,
    pub double_sided: bool,
    pub cast_shadows: bool,
    pub receive_shadows: bool,
}

impl Default for MaterialRef {
    fn default() -> Self {
        Self {
            material_id: 0,
            shader_variant: 0,
            render_queue: 2000,
            double_sided: false,
            cast_shadows: true,
            receive_shadows: true,
        }
    }
}

impl Component for MaterialRef {}

/// Mesh reference for 3D rendering.
#[derive(Debug, Clone)]
pub struct MeshRef {
    pub mesh_id: u64,
    pub sub_mesh_index: u32,
    pub lod_bias: f32,
    pub lod_count: u32,
    pub instance_count: u32,
}

impl Default for MeshRef {
    fn default() -> Self {
        Self {
            mesh_id: 0,
            sub_mesh_index: 0,
            lod_bias: 1.0,
            lod_count: 1,
            instance_count: 1,
        }
    }
}

impl Component for MeshRef {}

/// Light data component.
#[derive(Debug, Clone)]
pub struct LightData {
    pub light_type: LightType,
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    pub inner_cone_angle: f32,
    pub outer_cone_angle: f32,
    pub cast_shadows: bool,
    pub shadow_bias: f32,
    pub shadow_resolution: u32,
}

/// Type of light source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightType {
    Directional,
    Point,
    Spot,
    Area,
}

impl Default for LightData {
    fn default() -> Self {
        Self {
            light_type: LightType::Point,
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            range: 10.0,
            inner_cone_angle: 30.0,
            outer_cone_angle: 45.0,
            cast_shadows: true,
            shadow_bias: 0.001,
            shadow_resolution: 1024,
        }
    }
}

impl Component for LightData {}

/// Camera data component.
#[derive(Debug, Clone)]
pub struct CameraData {
    pub projection: ProjectionType,
    pub fov_degrees: f32,
    pub near_plane: f32,
    pub far_plane: f32,
    pub ortho_size: f32,
    pub clear_color: [f32; 4],
    pub depth: i32,
    pub render_target: Option<u64>,
    pub is_active: bool,
}

/// Camera projection type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionType {
    Perspective,
    Orthographic,
}

impl Default for CameraData {
    fn default() -> Self {
        Self {
            projection: ProjectionType::Perspective,
            fov_degrees: 60.0,
            near_plane: 0.1,
            far_plane: 1000.0,
            ortho_size: 10.0,
            clear_color: [0.1, 0.1, 0.1, 1.0],
            depth: 0,
            render_target: None,
            is_active: true,
        }
    }
}

impl Component for CameraData {}

/// Rigid body data for physics.
#[derive(Debug, Clone)]
pub struct RigidBodyData {
    pub body_type: RigidBodyType,
    pub mass: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub gravity_scale: f32,
    pub lock_rotation_x: bool,
    pub lock_rotation_y: bool,
    pub lock_rotation_z: bool,
    pub continuous_collision: bool,
}

/// Type of rigid body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RigidBodyType {
    Dynamic,
    Static,
    Kinematic,
}

impl Default for RigidBodyData {
    fn default() -> Self {
        Self {
            body_type: RigidBodyType::Dynamic,
            mass: 1.0,
            linear_damping: 0.0,
            angular_damping: 0.05,
            gravity_scale: 1.0,
            lock_rotation_x: false,
            lock_rotation_y: false,
            lock_rotation_z: false,
            continuous_collision: false,
        }
    }
}

impl Component for RigidBodyData {}

/// Collider data for physics.
#[derive(Debug, Clone)]
pub struct ColliderData {
    pub shape: ColliderShape,
    pub is_trigger: bool,
    pub friction: f32,
    pub restitution: f32,
    pub density: f32,
    pub collision_group: u32,
    pub collision_mask: u32,
}

/// Shape of a physics collider.
#[derive(Debug, Clone)]
pub enum ColliderShape {
    Box { half_extents: [f32; 3] },
    Sphere { radius: f32 },
    Capsule { radius: f32, half_height: f32 },
    Cylinder { radius: f32, half_height: f32 },
    Mesh { mesh_id: u64 },
}

impl Default for ColliderData {
    fn default() -> Self {
        Self {
            shape: ColliderShape::Box {
                half_extents: [0.5, 0.5, 0.5],
            },
            is_trigger: false,
            friction: 0.5,
            restitution: 0.0,
            density: 1.0,
            collision_group: 0xFFFF_FFFF,
            collision_mask: 0xFFFF_FFFF,
        }
    }
}

impl Component for ColliderData {}

// ---------------------------------------------------------------------------
// Named bundles
// ---------------------------------------------------------------------------

/// Bundle for spawning a 2D sprite entity.
///
/// Contains a transform, sprite data, and material reference.
pub struct SpriteBundle {
    pub transform: TransformData,
    pub sprite: SpriteData,
    pub material: MaterialRef,
}

impl Default for SpriteBundle {
    fn default() -> Self {
        Self {
            transform: TransformData::default(),
            sprite: SpriteData::default(),
            material: MaterialRef::default(),
        }
    }
}

impl Bundle for SpriteBundle {
    fn insert_into(self, world: &mut World, entity: Entity) {
        world.add_component(entity, self.transform);
        world.add_component(entity, self.sprite);
        world.add_component(entity, self.material);
    }

    fn component_count() -> usize {
        3
    }

    fn describe() -> &'static str {
        "SpriteBundle(transform, sprite, material)"
    }
}

/// Bundle for spawning a 3D mesh entity.
///
/// Contains a transform, mesh reference, and material reference.
pub struct MeshBundle {
    pub transform: TransformData,
    pub mesh: MeshRef,
    pub material: MaterialRef,
}

impl Default for MeshBundle {
    fn default() -> Self {
        Self {
            transform: TransformData::default(),
            mesh: MeshRef::default(),
            material: MaterialRef::default(),
        }
    }
}

impl Bundle for MeshBundle {
    fn insert_into(self, world: &mut World, entity: Entity) {
        world.add_component(entity, self.transform);
        world.add_component(entity, self.mesh);
        world.add_component(entity, self.material);
    }

    fn component_count() -> usize {
        3
    }

    fn describe() -> &'static str {
        "MeshBundle(transform, mesh, material)"
    }
}

/// Bundle for spawning a light source entity.
///
/// Contains a transform and light data.
pub struct LightBundle {
    pub transform: TransformData,
    pub light: LightData,
}

impl Default for LightBundle {
    fn default() -> Self {
        Self {
            transform: TransformData::default(),
            light: LightData::default(),
        }
    }
}

impl Bundle for LightBundle {
    fn insert_into(self, world: &mut World, entity: Entity) {
        world.add_component(entity, self.transform);
        world.add_component(entity, self.light);
    }

    fn component_count() -> usize {
        2
    }

    fn describe() -> &'static str {
        "LightBundle(transform, light)"
    }
}

/// Bundle for spawning a camera entity.
///
/// Contains a transform, camera data.
pub struct CameraBundle {
    pub transform: TransformData,
    pub camera: CameraData,
}

impl Default for CameraBundle {
    fn default() -> Self {
        Self {
            transform: TransformData::default(),
            camera: CameraData::default(),
        }
    }
}

impl Bundle for CameraBundle {
    fn insert_into(self, world: &mut World, entity: Entity) {
        world.add_component(entity, self.transform);
        world.add_component(entity, self.camera);
    }

    fn component_count() -> usize {
        2
    }

    fn describe() -> &'static str {
        "CameraBundle(transform, camera)"
    }
}

/// Bundle for spawning a physics-enabled entity.
///
/// Contains a transform, rigid body, and collider.
pub struct PhysicsBundle {
    pub transform: TransformData,
    pub rigid_body: RigidBodyData,
    pub collider: ColliderData,
}

impl Default for PhysicsBundle {
    fn default() -> Self {
        Self {
            transform: TransformData::default(),
            rigid_body: RigidBodyData::default(),
            collider: ColliderData::default(),
        }
    }
}

impl Bundle for PhysicsBundle {
    fn insert_into(self, world: &mut World, entity: Entity) {
        world.add_component(entity, self.transform);
        world.add_component(entity, self.rigid_body);
        world.add_component(entity, self.collider);
    }

    fn component_count() -> usize {
        3
    }

    fn describe() -> &'static str {
        "PhysicsBundle(transform, rigid_body, collider)"
    }
}

// ---------------------------------------------------------------------------
// BundleBuilder — dynamic bundle construction
// ---------------------------------------------------------------------------

/// A builder for constructing bundles dynamically at runtime.
///
/// Unlike static bundles (tuples or named structs), `BundleBuilder` allows
/// adding components one at a time, which is useful for editor workflows
/// and deserialization.
pub struct BundleBuilder {
    insertions: Vec<Box<dyn FnOnce(&mut World, Entity) + Send + Sync>>,
}

impl BundleBuilder {
    /// Create a new, empty bundle builder.
    pub fn new() -> Self {
        Self {
            insertions: Vec::new(),
        }
    }

    /// Add a component to the bundle.
    pub fn add<T: Component>(mut self, component: T) -> Self {
        self.insertions
            .push(Box::new(move |world: &mut World, entity: Entity| {
                world.add_component(entity, component);
            }));
        self
    }

    /// Build and insert all components into the entity.
    pub fn insert_into(self, world: &mut World, entity: Entity) {
        for insert_fn in self.insertions {
            insert_fn(world, entity);
        }
    }

    /// Build by spawning a new entity.
    pub fn spawn(self, world: &mut World) -> Entity {
        let entity = world.spawn_empty();
        self.insert_into(world, entity);
        entity
    }

    /// Number of components in the builder.
    pub fn len(&self) -> usize {
        self.insertions.len()
    }

    /// Whether the builder has no components.
    pub fn is_empty(&self) -> bool {
        self.insertions.is_empty()
    }
}

impl Default for BundleBuilder {
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

    #[derive(Debug, PartialEq, Clone)]
    struct Position {
        x: f32,
        y: f32,
    }
    impl Component for Position {}

    #[derive(Debug, PartialEq, Clone)]
    struct Velocity {
        dx: f32,
        dy: f32,
    }
    impl Component for Velocity {}

    #[derive(Debug, PartialEq, Clone)]
    struct Health(f32);
    impl Component for Health {}

    #[derive(Debug, PartialEq, Clone)]
    struct Name(String);
    impl Component for Name {}

    #[derive(Debug, PartialEq, Clone)]
    struct Marker;
    impl Component for Marker {}

    // -- Tuple bundle tests -----------------------------------------------------

    #[test]
    fn spawn_bundle_single_tuple() {
        let mut world = World::new();
        let e = world.spawn_bundle((Position { x: 1.0, y: 2.0 },));
        assert!(world.has_component::<Position>(e));
        assert_eq!(
            world.get_component::<Position>(e),
            Some(&Position { x: 1.0, y: 2.0 })
        );
    }

    #[test]
    fn spawn_bundle_two_tuple() {
        let mut world = World::new();
        let e = world.spawn_bundle((
            Position { x: 1.0, y: 2.0 },
            Velocity { dx: 3.0, dy: 4.0 },
        ));
        assert!(world.has_component::<Position>(e));
        assert!(world.has_component::<Velocity>(e));
    }

    #[test]
    fn spawn_bundle_three_tuple() {
        let mut world = World::new();
        let e = world.spawn_bundle((
            Position { x: 1.0, y: 2.0 },
            Velocity { dx: 3.0, dy: 4.0 },
            Health(100.0),
        ));
        assert!(world.has_component::<Position>(e));
        assert!(world.has_component::<Velocity>(e));
        assert!(world.has_component::<Health>(e));
        assert_eq!(
            world.get_component::<Health>(e).map(|h| h.0),
            Some(100.0)
        );
    }

    #[test]
    fn spawn_bundle_four_tuple() {
        let mut world = World::new();
        let e = world.spawn_bundle((
            Position { x: 0.0, y: 0.0 },
            Velocity { dx: 1.0, dy: 0.0 },
            Health(50.0),
            Name("player".to_string()),
        ));
        assert!(world.has_component::<Position>(e));
        assert!(world.has_component::<Velocity>(e));
        assert!(world.has_component::<Health>(e));
        assert!(world.has_component::<Name>(e));
    }

    #[test]
    fn insert_bundle_existing_entity() {
        let mut world = World::new();
        let e = world.spawn_entity().build();
        assert!(!world.has_component::<Position>(e));

        world.insert_bundle(
            e,
            (
                Position { x: 5.0, y: 6.0 },
                Velocity { dx: 7.0, dy: 8.0 },
            ),
        );
        assert!(world.has_component::<Position>(e));
        assert!(world.has_component::<Velocity>(e));
        assert_eq!(
            world.get_component::<Position>(e),
            Some(&Position { x: 5.0, y: 6.0 })
        );
    }

    #[test]
    fn insert_bundle_overwrites() {
        let mut world = World::new();
        let e = world.spawn_bundle((Health(50.0),));
        assert_eq!(
            world.get_component::<Health>(e).map(|h| h.0),
            Some(50.0)
        );

        world.insert_bundle(e, (Health(100.0),));
        assert_eq!(
            world.get_component::<Health>(e).map(|h| h.0),
            Some(100.0)
        );
    }

    #[test]
    fn insert_bundle_dead_entity_noop() {
        let mut world = World::new();
        let e = world.spawn_entity().build();
        world.despawn(e);

        world.insert_bundle(e, (Position { x: 1.0, y: 2.0 },));
        assert!(!world.has_component::<Position>(e));
    }

    #[test]
    fn spawn_bundle_batch() {
        let mut world = World::new();
        let bundles = vec![
            (Position { x: 0.0, y: 0.0 }, Health(100.0)),
            (Position { x: 1.0, y: 1.0 }, Health(80.0)),
            (Position { x: 2.0, y: 2.0 }, Health(60.0)),
        ];

        let entities = world.spawn_bundle_batch(bundles);
        assert_eq!(entities.len(), 3);

        for e in &entities {
            assert!(world.has_component::<Position>(*e));
            assert!(world.has_component::<Health>(*e));
        }
    }

    #[test]
    fn bundle_component_count() {
        assert_eq!(<(Position,)>::component_count(), 1);
        assert_eq!(<(Position, Velocity)>::component_count(), 2);
        assert_eq!(<(Position, Velocity, Health)>::component_count(), 3);
    }

    // -- Named bundle tests -----------------------------------------------------

    #[test]
    fn sprite_bundle_spawn() {
        let mut world = World::new();
        let e = world.spawn_bundle(SpriteBundle {
            transform: TransformData::from_position([10.0, 20.0, 0.0]),
            sprite: SpriteData {
                width: 32.0,
                height: 32.0,
                ..Default::default()
            },
            material: MaterialRef::default(),
        });
        assert!(world.has_component::<TransformData>(e));
        assert!(world.has_component::<SpriteData>(e));
        assert!(world.has_component::<MaterialRef>(e));

        let td = world.get_component::<TransformData>(e).unwrap();
        assert_eq!(td.position, [10.0, 20.0, 0.0]);

        let sprite = world.get_component::<SpriteData>(e).unwrap();
        assert_eq!(sprite.width, 32.0);
    }

    #[test]
    fn mesh_bundle_spawn() {
        let mut world = World::new();
        let e = world.spawn_bundle(MeshBundle::default());
        assert!(world.has_component::<TransformData>(e));
        assert!(world.has_component::<MeshRef>(e));
        assert!(world.has_component::<MaterialRef>(e));
        assert_eq!(MeshBundle::component_count(), 3);
    }

    #[test]
    fn light_bundle_spawn() {
        let mut world = World::new();
        let e = world.spawn_bundle(LightBundle {
            transform: TransformData::from_position([0.0, 10.0, 0.0]),
            light: LightData {
                light_type: LightType::Directional,
                color: [1.0, 0.9, 0.8],
                intensity: 2.0,
                ..Default::default()
            },
        });
        assert!(world.has_component::<TransformData>(e));
        assert!(world.has_component::<LightData>(e));

        let light = world.get_component::<LightData>(e).unwrap();
        assert_eq!(light.light_type, LightType::Directional);
        assert_eq!(light.intensity, 2.0);
    }

    #[test]
    fn camera_bundle_spawn() {
        let mut world = World::new();
        let e = world.spawn_bundle(CameraBundle {
            transform: TransformData::from_position([0.0, 5.0, -10.0]),
            camera: CameraData {
                fov_degrees: 75.0,
                ..Default::default()
            },
        });
        assert!(world.has_component::<TransformData>(e));
        assert!(world.has_component::<CameraData>(e));

        let cam = world.get_component::<CameraData>(e).unwrap();
        assert_eq!(cam.fov_degrees, 75.0);
        assert!(cam.is_active);
    }

    #[test]
    fn physics_bundle_spawn() {
        let mut world = World::new();
        let e = world.spawn_bundle(PhysicsBundle {
            transform: TransformData::from_position([0.0, 5.0, 0.0]),
            rigid_body: RigidBodyData {
                mass: 10.0,
                ..Default::default()
            },
            collider: ColliderData {
                shape: ColliderShape::Sphere { radius: 1.0 },
                ..Default::default()
            },
        });
        assert!(world.has_component::<TransformData>(e));
        assert!(world.has_component::<RigidBodyData>(e));
        assert!(world.has_component::<ColliderData>(e));

        let rb = world.get_component::<RigidBodyData>(e).unwrap();
        assert_eq!(rb.mass, 10.0);
        assert_eq!(rb.body_type, RigidBodyType::Dynamic);
    }

    #[test]
    fn named_bundle_describe() {
        assert!(SpriteBundle::describe().contains("sprite"));
        assert!(MeshBundle::describe().contains("mesh"));
        assert!(LightBundle::describe().contains("light"));
        assert!(CameraBundle::describe().contains("camera"));
        assert!(PhysicsBundle::describe().contains("rigid_body"));
    }

    // -- BundleBuilder tests ----------------------------------------------------

    #[test]
    fn bundle_builder_spawn() {
        let mut world = World::new();
        let e = BundleBuilder::new()
            .add(Position { x: 1.0, y: 2.0 })
            .add(Velocity { dx: 3.0, dy: 4.0 })
            .add(Health(100.0))
            .spawn(&mut world);

        assert!(world.has_component::<Position>(e));
        assert!(world.has_component::<Velocity>(e));
        assert!(world.has_component::<Health>(e));
    }

    #[test]
    fn bundle_builder_insert_into() {
        let mut world = World::new();
        let e = world.spawn_entity().build();

        BundleBuilder::new()
            .add(Position { x: 5.0, y: 6.0 })
            .add(Marker)
            .insert_into(&mut world, e);

        assert!(world.has_component::<Position>(e));
        assert!(world.has_component::<Marker>(e));
    }

    #[test]
    fn bundle_builder_len() {
        let builder = BundleBuilder::new()
            .add(Position { x: 0.0, y: 0.0 })
            .add(Velocity { dx: 0.0, dy: 0.0 });

        assert_eq!(builder.len(), 2);
        assert!(!builder.is_empty());
    }

    #[test]
    fn bundle_builder_empty() {
        let builder = BundleBuilder::new();
        assert_eq!(builder.len(), 0);
        assert!(builder.is_empty());
    }

    // -- TransformData tests ----------------------------------------------------

    #[test]
    fn transform_data_identity() {
        let td = TransformData::IDENTITY;
        assert_eq!(td.position, [0.0, 0.0, 0.0]);
        assert_eq!(td.rotation, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(td.scale, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn transform_data_from_position() {
        let td = TransformData::from_position([1.0, 2.0, 3.0]);
        assert_eq!(td.position, [1.0, 2.0, 3.0]);
        assert_eq!(td.scale, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn transform_data_from_position_scale() {
        let td = TransformData::from_position_scale([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]);
        assert_eq!(td.position, [1.0, 2.0, 3.0]);
        assert_eq!(td.scale, [2.0, 2.0, 2.0]);
    }

    // -- Component data defaults ------------------------------------------------

    #[test]
    fn sprite_data_defaults() {
        let s = SpriteData::default();
        assert_eq!(s.width, 1.0);
        assert_eq!(s.height, 1.0);
        assert_eq!(s.color, [1.0, 1.0, 1.0, 1.0]);
        assert!(!s.flip_x);
        assert!(!s.flip_y);
    }

    #[test]
    fn material_ref_defaults() {
        let m = MaterialRef::default();
        assert!(m.cast_shadows);
        assert!(m.receive_shadows);
        assert!(!m.double_sided);
    }

    #[test]
    fn mesh_ref_defaults() {
        let m = MeshRef::default();
        assert_eq!(m.lod_bias, 1.0);
        assert_eq!(m.instance_count, 1);
    }

    #[test]
    fn light_data_defaults() {
        let l = LightData::default();
        assert_eq!(l.light_type, LightType::Point);
        assert!(l.cast_shadows);
        assert_eq!(l.intensity, 1.0);
    }

    #[test]
    fn camera_data_defaults() {
        let c = CameraData::default();
        assert_eq!(c.projection, ProjectionType::Perspective);
        assert_eq!(c.fov_degrees, 60.0);
        assert!(c.is_active);
    }

    #[test]
    fn rigid_body_defaults() {
        let rb = RigidBodyData::default();
        assert_eq!(rb.body_type, RigidBodyType::Dynamic);
        assert_eq!(rb.mass, 1.0);
        assert_eq!(rb.gravity_scale, 1.0);
    }

    #[test]
    fn collider_defaults() {
        let c = ColliderData::default();
        assert!(!c.is_trigger);
        assert_eq!(c.friction, 0.5);
    }
}
