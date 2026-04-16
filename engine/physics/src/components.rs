//! ECS components and systems for physics integration.
//!
//! Bridges the physics simulation with the entity-component-system by providing
//! components that can be attached to entities and a system that drives the
//! simulation each frame.

use glam::{Quat, Vec3};

use crate::collision::{ColliderDesc, CollisionEvent, CollisionShape};
use crate::dynamics::ForceMode;
use crate::interface::{
    BodyType, ColliderHandle, PhysicsBackend, PhysicsMaterial, PhysicsWorld, RaycastHit,
    RigidBodyDesc, RigidBodyHandle,
};

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Component that attaches a rigid body to an entity.
///
/// When added to an entity, the `PhysicsSystem` will create a corresponding
/// rigid body in the physics world and synchronize transforms each frame.
#[derive(Debug, Clone)]
pub struct RigidBodyComponent {
    /// Handle into the physics world (set by PhysicsSystem after body creation).
    pub handle: Option<RigidBodyHandle>,

    /// The body type (Static, Dynamic, Kinematic).
    pub body_type: BodyType,

    /// Mass in kilograms.
    pub mass: f32,

    /// Linear velocity in m/s (read-back from simulation for Dynamic bodies).
    pub linear_velocity: Vec3,

    /// Angular velocity in rad/s (read-back from simulation for Dynamic bodies).
    pub angular_velocity: Vec3,

    /// Linear damping factor.
    pub linear_damping: f32,

    /// Angular damping factor.
    pub angular_damping: f32,

    /// Gravity scale multiplier for this body.
    pub gravity_scale: f32,

    /// Friction coefficient.
    pub friction: f32,

    /// Restitution (bounciness).
    pub restitution: f32,

    /// Whether this body should start asleep.
    pub start_asleep: bool,

    /// If true, the component was modified by gameplay code and needs to be
    /// pushed back to the physics world.
    pub dirty: bool,
}

impl Default for RigidBodyComponent {
    fn default() -> Self {
        Self {
            handle: None,
            body_type: BodyType::Dynamic,
            mass: 1.0,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            linear_damping: 0.01,
            angular_damping: 0.05,
            gravity_scale: 1.0,
            friction: 0.5,
            restitution: 0.3,
            start_asleep: false,
            dirty: true,
        }
    }
}

impl RigidBodyComponent {
    /// Create a new dynamic body component with the given mass.
    pub fn dynamic(mass: f32) -> Self {
        Self {
            body_type: BodyType::Dynamic,
            mass,
            ..Default::default()
        }
    }

    /// Create a new static body component.
    pub fn static_body() -> Self {
        Self {
            body_type: BodyType::Static,
            mass: 0.0,
            ..Default::default()
        }
    }

    /// Create a new kinematic body component.
    pub fn kinematic() -> Self {
        Self {
            body_type: BodyType::Kinematic,
            mass: 1.0,
            ..Default::default()
        }
    }

    /// Create a descriptor suitable for adding this body to the physics world.
    pub fn to_desc(&self, position: Vec3, rotation: Quat) -> RigidBodyDesc {
        RigidBodyDesc {
            body_type: self.body_type,
            mass: self.mass,
            friction: self.friction,
            restitution: self.restitution,
            linear_damping: self.linear_damping,
            angular_damping: self.angular_damping,
            position,
            rotation,
        }
    }
}

/// Component that attaches a collider to an entity.
///
/// Must be paired with a [`RigidBodyComponent`] on the same entity
/// (or a parent entity for compound colliders).
#[derive(Debug, Clone)]
pub struct ColliderComponent {
    /// Handle into the physics world (set by PhysicsSystem after collider creation).
    pub handle: Option<ColliderHandle>,

    /// Collider configuration.
    pub desc: ColliderDesc,

    /// If true, the collider was modified and needs to be re-created.
    pub dirty: bool,
}

impl Default for ColliderComponent {
    fn default() -> Self {
        Self {
            handle: None,
            desc: ColliderDesc::default(),
            dirty: true,
        }
    }
}

impl ColliderComponent {
    /// Create a sphere collider.
    pub fn sphere(radius: f32) -> Self {
        Self {
            desc: ColliderDesc {
                shape: CollisionShape::Sphere { radius },
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create a box collider.
    pub fn cuboid(half_extents: Vec3) -> Self {
        Self {
            desc: ColliderDesc {
                shape: CollisionShape::Box { half_extents },
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create a capsule collider.
    pub fn capsule(radius: f32, half_height: f32) -> Self {
        Self {
            desc: ColliderDesc {
                shape: CollisionShape::Capsule {
                    radius,
                    half_height,
                },
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Set this collider as a sensor (trigger).
    pub fn as_sensor(mut self) -> Self {
        self.desc.is_sensor = true;
        self
    }

    /// Set the material for this collider.
    pub fn with_material(mut self, material: PhysicsMaterial) -> Self {
        self.desc.material = material;
        self
    }
}

// ---------------------------------------------------------------------------
// Physics System
// ---------------------------------------------------------------------------

/// System that drives the physics simulation and synchronizes ECS transforms.
///
/// Each frame the system:
/// 1. Registers new rigid bodies and colliders with the physics world.
/// 2. Pushes dirty component state to the physics world.
/// 3. Steps the simulation with fixed sub-stepping.
/// 4. Reads back transforms from the physics world to ECS components.
/// 5. Dispatches collision events.
pub struct PhysicsSystem {
    /// The active physics world.
    world: PhysicsWorld,

    /// Whether the world has been initialized.
    initialized: bool,

    /// Fixed timestep for physics updates (default 1/60 s).
    pub fixed_timestep: f32,

    /// Accumulated time for fixed-step sub-stepping.
    time_accumulator: f32,

    /// Maximum number of sub-steps per frame to prevent spiral of death.
    pub max_substeps: u32,

    /// Global gravity vector.
    pub gravity: Vec3,

    /// Collision events from the most recent step, available for gameplay queries.
    pub collision_events: Vec<CollisionEvent>,
}

impl Default for PhysicsSystem {
    fn default() -> Self {
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        Self {
            world: PhysicsWorld::new(gravity),
            initialized: true,
            fixed_timestep: 1.0 / 60.0,
            time_accumulator: 0.0,
            max_substeps: 8,
            gravity,
            collision_events: Vec::new(),
        }
    }
}

impl PhysicsSystem {
    /// Create a new physics system with the given gravity.
    pub fn new(gravity: Vec3) -> Self {
        Self {
            world: PhysicsWorld::new(gravity),
            initialized: true,
            gravity,
            ..Default::default()
        }
    }

    /// Initialize the physics world using the given backend.
    pub fn initialize(
        &mut self,
        backend: &dyn PhysicsBackend,
    ) -> crate::interface::PhysicsResult<()> {
        self.world = backend.create_world(self.gravity)?;
        self.initialized = true;
        log::info!(
            "Physics system initialized with backend: {}",
            backend.name()
        );
        Ok(())
    }

    /// Register a rigid body component and return its handle.
    ///
    /// Call this when a new entity with a RigidBodyComponent is added to the world.
    pub fn register_body(
        &mut self,
        component: &mut RigidBodyComponent,
        position: Vec3,
        rotation: Quat,
    ) -> crate::interface::PhysicsResult<RigidBodyHandle> {
        let desc = component.to_desc(position, rotation);
        let handle = self.world.add_body(&desc)?;

        // Set initial velocities
        if component.linear_velocity != Vec3::ZERO {
            self.world
                .apply_force(handle, component.linear_velocity, ForceMode::VelocityChange)?;
        }

        component.handle = Some(handle);
        component.dirty = false;
        Ok(handle)
    }

    /// Register a collider component and return its handle.
    ///
    /// The body_handle must be a valid handle returned by `register_body`.
    pub fn register_collider(
        &mut self,
        component: &mut ColliderComponent,
        body_handle: RigidBodyHandle,
    ) -> crate::interface::PhysicsResult<ColliderHandle> {
        let handle = self.world.add_collider(body_handle, &component.desc)?;
        component.handle = Some(handle);
        component.dirty = false;
        Ok(handle)
    }

    /// Unregister a rigid body and all its colliders.
    pub fn unregister_body(
        &mut self,
        handle: RigidBodyHandle,
    ) -> crate::interface::PhysicsResult<()> {
        self.world.remove_body(handle)
    }

    /// Advance the physics simulation by the given frame delta time.
    ///
    /// Uses fixed sub-stepping internally for deterministic results.
    pub fn update(&mut self, dt: f32) {
        if !self.initialized {
            log::warn!("PhysicsSystem::update called but physics world is not initialized");
            return;
        }

        profiling::scope!("PhysicsSystem::update");

        // Fixed-step sub-stepping
        self.time_accumulator += dt;
        let mut steps = 0u32;
        while self.time_accumulator >= self.fixed_timestep && steps < self.max_substeps {
            if let Err(e) = self.world.step(self.fixed_timestep) {
                log::error!("Physics step failed: {}", e);
                break;
            }
            self.time_accumulator -= self.fixed_timestep;
            steps += 1;
        }

        // Clamp accumulator to avoid spiral of death
        if self.time_accumulator > self.fixed_timestep {
            log::warn!(
                "Physics cannot keep up: dropping {:.1}ms of simulation time",
                self.time_accumulator * 1000.0
            );
            self.time_accumulator = 0.0;
        }

        // Drain collision events
        self.collision_events = self.world.drain_collision_events();
    }

    /// Synchronize a body component's velocities from the physics world.
    ///
    /// Call this after `update()` for each entity with a RigidBodyComponent.
    pub fn sync_component(
        &self,
        component: &mut RigidBodyComponent,
    ) -> Option<(Vec3, Quat)> {
        let handle = component.handle?;

        let position = self.world.get_position(handle).ok()?;
        let rotation = self.world.get_rotation(handle).ok()?;
        let lin_vel = self.world.get_linear_velocity(handle).ok()?;
        let ang_vel = self.world.get_angular_velocity(handle).ok()?;

        component.linear_velocity = lin_vel;
        component.angular_velocity = ang_vel;

        Some((position, rotation))
    }

    /// Push dirty component state back to the physics world.
    ///
    /// Call this before `update()` if a component's properties have changed.
    pub fn push_dirty_state(
        &mut self,
        component: &mut RigidBodyComponent,
        position: Vec3,
        rotation: Quat,
    ) {
        if !component.dirty {
            return;
        }

        if let Some(handle) = component.handle {
            let _ = self.world.set_position(handle, position);
            let _ = self.world.set_rotation(handle, rotation);
            if component.linear_velocity != Vec3::ZERO {
                let _ = self
                    .world
                    .set_linear_velocity(handle, component.linear_velocity);
            }
        }

        component.dirty = false;
    }

    /// Apply a force to a body through the system.
    pub fn apply_force(
        &mut self,
        handle: RigidBodyHandle,
        force: Vec3,
        mode: ForceMode,
    ) -> crate::interface::PhysicsResult<()> {
        self.world.apply_force(handle, force, mode)
    }

    /// Apply a torque to a body through the system.
    pub fn apply_torque(
        &mut self,
        handle: RigidBodyHandle,
        torque: Vec3,
        mode: ForceMode,
    ) -> crate::interface::PhysicsResult<()> {
        self.world.apply_torque(handle, torque, mode)
    }

    /// Cast a ray and return all hits.
    pub fn raycast(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
    ) -> Vec<RaycastHit> {
        self.world.raycast(origin, direction, max_distance)
    }

    /// Cast a ray and return the closest hit.
    pub fn raycast_closest(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
    ) -> Option<RaycastHit> {
        self.world.raycast_closest(origin, direction, max_distance, u32::MAX)
    }

    /// Returns a reference to the underlying physics world.
    pub fn world(&self) -> &PhysicsWorld {
        &self.world
    }

    /// Returns a mutable reference to the underlying physics world.
    pub fn world_mut(&mut self) -> &mut PhysicsWorld {
        &mut self.world
    }

    /// Set the global gravity vector.
    pub fn set_gravity(&mut self, gravity: Vec3) {
        self.gravity = gravity;
        self.world.set_gravity(gravity);
    }

    /// Get the number of bodies in the physics world.
    pub fn body_count(&self) -> usize {
        self.world.body_count()
    }

    /// Get the number of active (non-sleeping) bodies.
    pub fn active_body_count(&self) -> usize {
        self.world.active_body_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::CustomBackend;

    #[test]
    fn test_physics_system_default() {
        let system = PhysicsSystem::default();
        assert_eq!(system.body_count(), 0);
    }

    #[test]
    fn test_physics_system_initialize() {
        let mut system = PhysicsSystem::new(Vec3::new(0.0, -9.81, 0.0));
        let backend = CustomBackend::new();
        system.initialize(&backend).unwrap();
    }

    #[test]
    fn test_register_body() {
        let mut system = PhysicsSystem::new(Vec3::new(0.0, -9.81, 0.0));
        let mut rb = RigidBodyComponent::dynamic(1.0);

        let handle = system
            .register_body(&mut rb, Vec3::new(0.0, 5.0, 0.0), Quat::IDENTITY)
            .unwrap();

        assert!(rb.handle.is_some());
        assert_eq!(system.body_count(), 1);
    }

    #[test]
    fn test_register_collider() {
        let mut system = PhysicsSystem::new(Vec3::ZERO);

        let mut rb = RigidBodyComponent::dynamic(1.0);
        let body_handle = system
            .register_body(&mut rb, Vec3::ZERO, Quat::IDENTITY)
            .unwrap();

        let mut col = ColliderComponent::sphere(0.5);
        let col_handle = system.register_collider(&mut col, body_handle).unwrap();

        assert!(col.handle.is_some());
    }

    #[test]
    fn test_update_with_substeps() {
        let mut system = PhysicsSystem::new(Vec3::new(0.0, -9.81, 0.0));

        let mut rb = RigidBodyComponent::dynamic(1.0);
        let handle = system
            .register_body(&mut rb, Vec3::new(0.0, 10.0, 0.0), Quat::IDENTITY)
            .unwrap();

        let mut col = ColliderComponent::sphere(0.5);
        system.register_collider(&mut col, handle).unwrap();

        // Update with a large dt to trigger sub-stepping
        system.update(1.0 / 30.0);

        let (pos, _rot) = system.sync_component(&mut rb).unwrap();
        assert!(pos.y < 10.0, "Body should have fallen: y = {}", pos.y);
    }

    #[test]
    fn test_static_body_component() {
        let rb = RigidBodyComponent::static_body();
        assert_eq!(rb.body_type, BodyType::Static);
        assert_eq!(rb.mass, 0.0);
    }

    #[test]
    fn test_collider_component_sphere() {
        let col = ColliderComponent::sphere(2.0);
        match &col.desc.shape {
            CollisionShape::Sphere { radius } => assert_eq!(*radius, 2.0),
            _ => panic!("Expected sphere shape"),
        }
    }

    #[test]
    fn test_collider_component_cuboid() {
        let col = ColliderComponent::cuboid(Vec3::new(1.0, 2.0, 3.0));
        match &col.desc.shape {
            CollisionShape::Box { half_extents } => {
                assert_eq!(*half_extents, Vec3::new(1.0, 2.0, 3.0))
            }
            _ => panic!("Expected box shape"),
        }
    }

    #[test]
    fn test_collider_component_sensor() {
        let col = ColliderComponent::sphere(1.0).as_sensor();
        assert!(col.desc.is_sensor);
    }

    #[test]
    fn test_apply_force_through_system() {
        let mut system = PhysicsSystem::new(Vec3::ZERO);

        let mut rb = RigidBodyComponent::dynamic(1.0);
        let handle = system
            .register_body(&mut rb, Vec3::ZERO, Quat::IDENTITY)
            .unwrap();

        let mut col = ColliderComponent::sphere(0.5);
        system.register_collider(&mut col, handle).unwrap();

        system
            .apply_force(handle, Vec3::new(10.0, 0.0, 0.0), ForceMode::Impulse)
            .unwrap();

        system.update(1.0 / 60.0);

        let (pos, _) = system.sync_component(&mut rb).unwrap();
        assert!(pos.x > 0.0, "Body should have moved: x = {}", pos.x);
    }

    #[test]
    fn test_raycast_through_system() {
        let mut system = PhysicsSystem::new(Vec3::ZERO);

        let mut rb = RigidBodyComponent::static_body();
        let handle = system
            .register_body(&mut rb, Vec3::new(5.0, 0.0, 0.0), Quat::IDENTITY)
            .unwrap();

        let mut col = ColliderComponent::sphere(1.0);
        system.register_collider(&mut col, handle).unwrap();

        let hits = system.raycast(Vec3::ZERO, Vec3::X, 100.0);
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn test_set_gravity() {
        let mut system = PhysicsSystem::new(Vec3::ZERO);
        system.set_gravity(Vec3::new(0.0, -20.0, 0.0));
        assert_eq!(system.gravity, Vec3::new(0.0, -20.0, 0.0));
    }
}
